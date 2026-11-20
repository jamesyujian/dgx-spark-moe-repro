#!/usr/bin/env python3
"""S1-04 静默错误扫描：DeepGEMM grouped GEMM 的语义判定基线

为什么先做 DeepGEMM 而不是"三后端端到端对比"：
  端到端比 LLM 输出，采样噪声、batch 组合、prefix cache 都会让"输出不同"变成
  无法归因的信号。先在一个**可解析求值**的算子上把"什么算错"定义清楚，端到端
  比对才有基准。所以本脚本是 S1-04 的地基，不是替代。

判定分三档，区别在"能不能被原谅"：
  ok         : 落在量化误差内，且无 NaN/Inf
  suspicious : 数值"看起来对"但性质错（如应非零却全零、应有限却含 Inf）
  error      : 明确错或有 NaN/Inf/崩溃

危险用例（已知会触发 CUDA 719）一律跑在子进程里：
  一次非法内存访问会让整个 CUDA context 失效，同进程后续用例全部作废。
  这与 S2-01 的挂起回收同源——只有进程级隔离能回收。

用法:
  python s104_semantic.py            # 全部（危险项自动隔离子进程）
  python s104_semantic.py --safe     # 只跑不会崩的
  python s104_semantic.py --worker-ue8m0-off   # 内部
"""
import argparse
import hashlib
import json
import subprocess
import sys

import torch
import deep_gemm

RES = []


def record(name, expect, got_desc, verdict, detail=""):
    RES.append({"case": name, "expect": expect, "got": got_desc,
                "verdict": verdict, "detail": detail})
    print("%-38s %-8s %s" % (name, verdict, got_desc[:70]), flush=True)


def _stats(d):
    f = d.float()
    return {"nan": int(torch.isnan(f).sum()), "inf": int(torch.isinf(f).sum()),
            "zero_frac": round(float((f == 0).float().mean()), 4),
            "absmax": round(float(f.abs().max()), 4)}


def _hash(d):
    # bfloat16 不支持 numpy 直转，先升 float32
    return hashlib.sha256(d.float().cpu().numpy().tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------- bf16 基线
def bf16_masked(G, max_m, N, K, masked_m, fill_a=1.0, fill_b=1.0, b_eye=False):
    a = torch.full((G, max_m, K), fill_a, device="cuda", dtype=torch.bfloat16)
    if b_eye:
        b = torch.zeros((G, N, K), device="cuda", dtype=torch.bfloat16)
        for i in range(min(N, K)):
            b[:, i, i] = 1.0
    else:
        b = torch.full((G, N, K), fill_b, device="cuda", dtype=torch.bfloat16)
    d = torch.zeros((G, max_m, N), device="cuda", dtype=torch.bfloat16)
    mm = torch.tensor(masked_m, device="cuda", dtype=torch.int32)
    deep_gemm.m_grouped_bf16_gemm_nt_masked(a, b, d, mm, expected_m=max_m)
    torch.cuda.synchronize()
    return a, b, d, mm


def stage_bf16_single():
    """单组 A=1,B=1 → 每个元素应等于 K。这是对 API 用法本身的校验。"""
    G, M, N, K = 1, 128, 128, 128
    _, _, d, _ = bf16_masked(G, M, N, K, [M])
    got = float(d.float().mean())
    err = abs(got - K) / K
    record("bf16 single: A=1,B=1 → K", str(K), "mean=%.2f rel=%.2f%%" % (got, err * 100),
           "ok" if err < 0.02 else ("suspicious" if err < 0.1 else "error"),
           json.dumps(_stats(d)))


def stage_bf16_identity():
    """B=I → 输出应等于 A。"""
    G, M, N, K = 1, 128, 128, 128
    _, _, d, _ = bf16_masked(G, M, N, K, [M], fill_a=3.0, b_eye=True)
    got = float(d.float().mean())
    err = abs(got - 3.0) / 3.0
    record("bf16 identity: B=I → A", "3.0", "mean=%.3f rel=%.2f%%" % (got, err * 100),
           "ok" if err < 0.02 else "error")


def stage_bf16_multi_with_empty():
    """G=4，masked_m=[128,128,0,64]。空组不应写脏数据/产 NaN。"""
    G, M, N, K = 4, 128, 128, 128
    mm = [M, M, 0, 64]
    _, _, d, _ = bf16_masked(G, M, N, K, mm)
    s = _stats(d)
    # 第 3 组（index 2）应为空且不污染；第 4 组只有前 64 行有值
    g2 = d[2].float()
    g3 = d[3].float()
    g3_tail = g3[64:]
    verdict = "ok"
    detail = {"g2_absmax": round(float(g2.abs().max()), 3),
              "g3_tail_absmax": round(float(g3_tail.abs().max()), 3),
              "nan": s["nan"], "inf": s["inf"]}
    if s["nan"] or s["inf"]:
        verdict = "error"
    elif float(g3_tail.abs().max()) > 1.0:
        verdict = "suspicious"   # 超出 masked_m 的行被写了
    record("bf16 multi: empty group G2", "G2 不写脏、G3 尾 64 行不写",
           json.dumps(detail), verdict)


def stage_bf16_determinism():
    """同输入 10 次，输出必须 bit-exact 一致。"""
    G, M, N, K = 1, 128, 128, 128
    hs = []
    for _ in range(10):
        _, _, d, _ = bf16_masked(G, M, N, K, [M])
        hs.append(_hash(d))
    uniq = len(set(hs))
    record("bf16 determinism (10x)", "1 unique hash", "%d unique %s" % (uniq, hs[0]),
           "ok" if uniq == 1 else "error")


# ---------------------------------------------------------------- fp8 基线
def _fp8_masked(a_val, b_val, ue8m0=True, disable_cast=False):
    """a/b 必须传 (data, scale) 元组——`_unpack_ab_pair` 会解包。
    传裸 tensor 会报 `index 1 out of bounds`，那是调用错误，不是算子错误。"""
    G, M, N, K = 1, 128, 128, 128
    a_f = torch.full((M, K), a_val, device="cuda", dtype=torch.float32)
    b_f = torch.full((N, K), b_val, device="cuda", dtype=torch.float32)
    a_d, a_s = deep_gemm.per_token_cast_to_fp8(a_f, use_ue8m0=ue8m0)
    b_d, b_s = deep_gemm.per_block_cast_to_fp8(b_f, use_ue8m0=ue8m0)
    a = (a_d.unsqueeze(0), a_s.unsqueeze(0) if a_s.dim() == 2 else a_s)
    b = (b_d.unsqueeze(0), b_s.unsqueeze(0) if b_s.dim() == 2 else b_s)
    d = torch.zeros((G, M, N), device="cuda", dtype=torch.bfloat16)
    mm = torch.tensor([M], device="cuda", dtype=torch.int32)
    deep_gemm.m_grouped_fp8_gemm_nt_masked(a, b, d, mm, expected_m=M,
                                           recipe=(1, 128, 128),
                                           disable_ue8m0_cast=disable_cast)
    torch.cuda.synchronize()
    return d


def stage_fp8_ue8m0():
    """FP8 + ue8m0。生产路径口径（已知安全）。"""
    d = _fp8_masked(1.0, 1.0, ue8m0=True)
    got = float(d.float().mean())
    s = _stats(d)
    err = abs(got - 128) / 128
    v = "error" if (s["nan"] or s["inf"]) else ("ok" if err < 0.05 else "suspicious")
    record("fp8 ue8m0: A=1,B=1 → K", "128",
           "mean=%.2f rel=%.2f%% %s" % (got, err * 100, json.dumps(s)), v)


def stage_fp8_underflow():
    """FP8 极小值：应变为 0，不应变 NaN。"""
    d = _fp8_masked(1e-30, 1.0, ue8m0=True)
    s = _stats(d)
    v = "error" if (s["nan"] or s["inf"]) else "ok"
    record("fp8 underflow 1e-30", "全 0 或有限", json.dumps(s), v)


def stage_fp8_overflow():
    """FP8 大值：BF16 输出端不应出现 Inf。"""
    d = _fp8_masked(400.0, 400.0, ue8m0=True)
    s = _stats(d)
    v = "error" if s["inf"] else ("suspicious" if s["nan"] else "ok")
    record("fp8 overflow 400*400*128", "不应 Inf", json.dumps(s), v)


# ------------------------------------------------------- 危险用例（子进程）
def worker_ue8m0_off():
    """非 ue8m0 路径（use_ue8m0=False + disable_cast）：已知 SM120 崩 CUDA 719。"""
    M, N, K = 128, 128, 128
    a_f = torch.full((M, K), 1.0, device="cuda", dtype=torch.float32)
    b_f = torch.full((N, K), 1.0, device="cuda", dtype=torch.float32)
    a_d, a_s = deep_gemm.per_token_cast_to_fp8(a_f, use_ue8m0=False)
    b_d, b_s = deep_gemm.per_block_cast_to_fp8(b_f, use_ue8m0=False)
    a = (a_d.unsqueeze(0), a_s.unsqueeze(0) if a_s.dim() == 2 else a_s)
    b = (b_d.unsqueeze(0), b_s.unsqueeze(0) if b_s.dim() == 2 else b_s)
    d = torch.zeros((1, M, N), device="cuda", dtype=torch.bfloat16)
    mm = torch.tensor([M], device="cuda", dtype=torch.int32)
    deep_gemm.m_grouped_fp8_gemm_nt_masked(a, b, d, mm, expected_m=M,
                                           recipe=(1, 128, 128),
                                           disable_ue8m0_cast=True)
    torch.cuda.synchronize()
    f = d.float()
    print(json.dumps({"survived": True, "nan": int(torch.isnan(f).sum()),
                      "absmax": round(float(f.abs().max()), 3)}), flush=True)


def run_dangerous(name, flag, expect):
    try:
        r = subprocess.run([sys.executable, __file__, flag],
                           timeout=180, capture_output=True, text=True)
        tail = (r.stdout.strip().splitlines() or [""])[-1]
        code = r.returncode
        if "survived" in tail:
            payload = json.loads(tail)
            if payload["nan"] > 0:
                record(name, expect, "SURVIVED but NaN=%d" % payload["nan"], "error")
            else:
                record(name, expect, "SURVIVED absmax=%s" % payload["absmax"],
                       "suspicious" if payload["absmax"] < 1 else "ok")
        else:
            err = (r.stderr.strip().splitlines() or [""])[-1]
            record(name, expect, "CRASH exit=%d %s" % (code, err[:60]), "error")
    except subprocess.TimeoutExpired:
        record(name, expect, "TIMEOUT 180s（挂起，需进程级回收）", "error")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safe", action="store_true")
    ap.add_argument("--worker-ue8m0-off", action="store_true")
    args = ap.parse_args()

    if args.worker_ue8m0_off:
        worker_ue8m0_off()
        return

    import deep_gemm as dg
    torch.cuda.init()
    print("gpu:", torch.cuda.get_device_name(0), "| deep_gemm", dg.__version__, flush=True)
    for fn in (stage_bf16_single, stage_bf16_identity, stage_bf16_multi_with_empty,
               stage_bf16_determinism, stage_fp8_ue8m0, stage_fp8_underflow,
               stage_fp8_overflow):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            record(fn.__name__, "no exception", "EXCEPTION %s" % str(exc)[:80], "error")
        torch.cuda.empty_cache()

    if not args.safe:
        run_dangerous("fp8 NON-ue8m0 (disable cast)", "--worker-ue8m0-off",
                      "要么正常，要么明确崩；不得静默 NaN")

    bad = [r for r in RES if r["verdict"] != "ok"]
    print("\n=== SUMMARY: %d cases, %d not-ok ===" % (len(RES), len(bad)))
    for r in bad:
        print("  [%s] %s | expect=%s | got=%s" % (r["verdict"], r["case"], r["expect"], r["got"]))
    with open("/tmp/s104_result.json", "w") as f:
        json.dump(RES, f, ensure_ascii=False, indent=1)
    print("saved /tmp/s104_result.json")


if __name__ == "__main__":
    import subprocess  # noqa: E402  (worker 需要)
    main()
