#!/usr/bin/env python3
"""SM12x 矩阵 · GSM8K 全量评测(契约 dataset.yaml 口径)

口径(严格按 contracts/dataset.yaml):
  - 数据集: $GSM8K_DATA，默认 data/gsm8k/gsm8k_test_1319.jsonl（1319 题，md5 已固化）
  - decode: temperature=0, seed=0, max_new_tokens=512, stop=["\nQuestion:", "\nQ:"]
  - answer_extract: 取最后一个数值;无唯一数值即计为 syntax_invalid
  - 判据(thresholds.yaml round1):
      syntax_valid_rate >= 0.95
      gsm8k_accuracy 与同模型 BF16 参考相差 <= 5 个百分点  ← ⚠ 该门已废弃（2026-11-19）：
        1319 题单跑标准误 ≈1.0 pp，5 pp 门什么都筛不出来；改用逐题 McNemar 配对检验，
        且参照必须是「同版本同 TP」那一跑。见 docs/NOISE_POLICY.md 规则 4。

用法:
  python eval_gsm8k.py <port> <backend> <out_jsonl>
"""
import json, os, re, sys, time, urllib.request, urllib.error

DATA = os.environ.get("GSM8K_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gsm8k", "gsm8k_test_1319.jsonl"))
MODEL = os.environ.get("EVAL_MODEL", "qwen36-35b-nvfp4")
TEMP, SEED, MAXTOK = 0, 0, 512
STOPS = ["\nQuestion:", "\nQ:"]

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text):
    """取最后一个数值。无 → None(即 syntax_invalid)。"""
    nums = NUM_RE.findall(text.replace(",", ""))
    if not nums:
        return None
    return nums[-1]


def gold_answer(ans_field):
    """GSM8K 的 answer 字段最后一行是 '#### <number>'。"""
    if "####" in ans_field:
        tail = ans_field.split("####")[-1].strip()
        nums = NUM_RE.findall(tail.replace(",", ""))
        return nums[-1] if nums else None
    nums = NUM_RE.findall(ans_field.replace(",", ""))
    return nums[-1] if nums else None


def norm(x):
    """数值归一化: 去尾零、统一小数形态。"""
    if x is None:
        return None
    try:
        f = float(x)
    except ValueError:
        return None
    if f == int(f):
        return str(int(f))
    return ("%.6f" % f).rstrip("0").rstrip(".")


def chat(port, question):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": question}],
        "temperature": TEMP,
        "seed": SEED,
        "max_tokens": MAXTOK,
        "stop": STOPS,
    }
    # thinking 开关:Qwen3.5/3.6 chat_template 支持 enable_thinking。
    # ENABLE_THINKING=1 时开启(推理链占满 max_tokens,导致输出截断、答案抽取失真)。
    if os.environ.get("ENABLE_THINKING") == "1":
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    else:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % port,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())


def main():
    port = int(sys.argv[1])
    backend = sys.argv[2]
    out_path = sys.argv[3]

    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    print("dataset: %d 题" % len(rows))

    out = open(out_path, "w", encoding="utf-8")
    n_ok = n_syn = n_wrong = 0
    t0 = time.time()
    gen_tokens = 0

    for i, r in enumerate(rows):
        try:
            resp = chat(port, r["question"])
            content = resp["choices"][0]["message"]["content"] or ""
            usage = resp.get("usage", {}) or {}
            gen_tokens += usage.get("completion_tokens", 0) or 0
            finish = resp["choices"][0].get("finish_reason")
        except Exception as e:
            content = ""
            finish = "error"
            print("  [%d] ERROR %s" % (i, str(e)[:80]))

        pred = last_number(content)
        gold = gold_answer(r["answer"])
        if pred is None:
            n_syn += 1
            verdict = "syntax_invalid"
        elif norm(pred) == norm(gold):
            n_ok += 1
            verdict = "correct"
        else:
            n_wrong += 1
            verdict = "wrong"

        out.write(json.dumps({
            "idx": i, "question": r["question"][:300],
            "gold": gold, "pred": pred, "verdict": verdict,
            "content": content[:2000],
            "finish_reason": finish,
        }, ensure_ascii=False) + "\n")
        out.flush()

        if (i + 1) % 50 == 0:
            el = time.time() - t0
            acc = n_ok / (i + 1)
            print("  [%d/%d] acc=%.4f syn_invalid=%d wrong=%d  %.1f tok/s  elapsed=%.0fs" % (
                i + 1, len(rows), acc, n_syn, n_wrong, gen_tokens / max(el, 1), el))

    out.close()
    el = time.time() - t0
    total = len(rows)
    acc = n_ok / total
    syn = n_syn / total
    summary = {
        "backend": backend,
        "enable_thinking": os.environ.get("ENABLE_THINKING", "0"),
        "port": port,
        "n": total,
        "correct": n_ok,
        "syntax_invalid": n_syn,
        "wrong": n_wrong,
        "gsm8k_accuracy": round(acc, 6),
        "syntax_valid_rate": round(1.0 - syn, 6),
        "gen_tokens": gen_tokens,
        "elapsed_s": round(el, 1),
        "throughput_tok_s": round(gen_tokens / max(el, 1), 2),
    }
    sp = out_path.replace(".jsonl", "_summary.json")
    json.dump(summary, open(sp, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("summary ->", sp)


if __name__ == "__main__":
    main()
