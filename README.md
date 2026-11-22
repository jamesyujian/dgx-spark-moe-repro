# dgx-spark-moe-repro

**English**: [`README.en.md`](README.en.md) ｜ 中文（本文件为准）

> MoE 推理在 **SM120（RTX PRO 5000 / 4500 / 5080）** 与 **SM121（GB10 / DGX Spark）** 上的精度与复现基线，覆盖 **vLLM / SGLang / llama.cpp**。

## 本仓库里有什么

| 有什么 | 在哪 |
|---|---|
| **N1 精度安全公告（正文）** | `docs/N1_公告_v3_公开版.md`（中文，约 590 行）；英文执行摘要 `docs/N1_v3_executive_summary.en.md` |
| **闭合状态表** | `docs/N1_闭合状态表.md` — 每条结论现在能写到什么程度、哪一格还空着 |
| **数据索引（先看这个）** | `N1_data_index.md` — 每条结论 → 原始文件 / 条件 / 状态，37 行，不解释 |
| **复现 harness** | `scripts/serve_fp8_tp2.sh`、`serve_nvfp4_tp1.sh`、`probe_20_prompts.py`、`eval_gsm8k.py`、`attrib_tp_vs_dtype.py`、`run_matrix.sh`；**L1 判定原件** `scripts/s104_semantic.py`（非 ue8m0 直连 DeepGEMM 的 8 用例，可独立跑） |
| **坑档 / 噪声纪律** | `docs/NOISE_POLICY.md`、`docs/EVIDENCE_INDEX.md`、`docs/SE-03_teacher_forcing_verdict.md` |
| **原始 JSON 落盘路径** | 115 `$BASE/d4/results/s104i_*`、`s104j_*`、`s104k_*`；101 `$BASE/d4/results/s104x3_det_*_sm121_101_r{1..10}`、`$BASE/results/sglang_result_*`；逐条见 `N1_data_index.md` |
| **自检用参考跑** | `data/reference_fp8_tp2/`、`data/reference_replay_1120/`、`data/reference_crossmachine_nvfp4/`、**`data/reference_l1_semantic/`**（L1 8 用例的两份 verdict：SM120 / 2.8.1 的 `s104_result_sm120.json`，末条 `gemm.hpp:346`；SM121 / 0.2.3.post2 的 `s104_result_sm121_dg023post2.json`，末条 `gemm.hpp:415`）（含 meta 与期望 md5） |

**三秒导航**：要复核某个数字 → `N1_data_index.md` 找到行、按"原始文件"去那台机器；要自己跑一遍 → 跳到 §2 快速开始；要看结论边界 → `docs/N1_闭合状态表.md`。

**发布包分两个 zip**：`dgx-spark-moe-repro-docs-vX.zip`（文档 / harness / 版本清单）与 `dgx-spark-moe-repro-data-vX.zip`（`data/` 全部原始 JSON 与指纹）。两份解压到同一父目录即合并成一棵完整树；各自根目录下的 `MANIFEST.md` 写明另一半在哪、sha256 是多少。

---

## 0. 先看三件事（不然会白跑）

1. **精度结论可复现，吞吐结论不可跨窗口比较。** 同一栈同一参数下，20 题 × 10 轮的内容指纹应逐字节相同；而 tok/s 在同一台机 3 分钟窗口内就可能漂 ~18 %。⇒ 吞吐只比"同窗口配对比值"，不要比绝对值。详见 `docs/NOISE_POLICY.md`。
2. **起服必须显式指定 MoE 后端**（`--moe-backend marlin|humming|b12x|triton|...`），不要靠 auto。auto 选出来的 backend 因版本而异，`effective_backend` 不记下来，结论就无法归因。
3. **工具调用必须显式指定 parser**：Qwen3 系列输出的是 Qwen3 方言 XML，`--tool-call-parser qwen3_xml` 才能让 API 层 `tool_calls` 有值；用 `hermes` 会解析失败并静默退回纯文本（`tool_calls=null`）。判分脚本会把这种情况标记为 `parsed_from_content=True`。**同一栈只差这两个参数，20 题就从 15/20 变成 18/20**（差的 3 题全是 `tc_*`，不是精度）。
4. **算指纹要剔掉 `tool_calls[*].id`**：服务端每跑新生成 `chatcmpl-tool-xxxx`，含 id 时带工具调用的跑"永远不稳定"。实测：含 id 三轮三个值，剔 id 后三轮同一个 `c8db9a77c6d7b58a40df9d66f75344e7`。口径见 `data/fingerprints/expected.md5`。

---

## 1. 目录

```
N1_data_index.md      公告每条结论 → 原始文件 / 条件 / 状态（极简索引，先看这个）
LICENSE               Apache-2.0
CHANGELOG.md          版本记录

env/                 版本与镜像清单（复现前先对齐这一层）
  versions.md        三条实测栈的精确版本：驱动 / torch / vllm / transformers / flashinfer / NCCL
  requirements.txt   pip 侧最小依赖（仅作参考，vllm 版本以 versions.md 为准）
  docker.md          容器镜像引用与 digest（版本独占后端只能在该镜像里测）

data/
  prompts/s104b_prompts.json   20 题固定集（含 3 道带 tools 的 tc_*），sha256 见 fingerprints/
  prompts/README.md            每题考什么、判分类型、max_tokens
  fingerprints/expected.md5    期望指纹（内容投影 md5 + 原始文件 md5），附计算口径
  reference_fp8_tp2/           同一栈的参考跑（r1–r3 + meta），可离线复算指纹
  reference_l1_semantic/       N1 §1.1 L1 的 8 条 verdict（`s104_result_sm120.json` + `s104_result_sm121_dg023post2.json` + 来源说明）
  gsm8k/README.md              GSM8K 1319 子集从哪来、怎么切、许可注意

scripts/
  s104_semantic.py              ★ N1 §1.1 L1 判定原件（非 ue8m0 直连 DeepGEMM，8 用例，可独立跑）
  serve_fp8_tp2.sh              FP8 + triton + TP=2 起服模板
  serve_nvfp4_tp1.sh            NVFP4 + marlin + TP=1 起服模板
  probe_20_prompts.py           20 题判分器（含 XML fallback 标记）
  eval_gsm8k.py                 GSM8K 1319 评测器（计数字段是 verdict，不是布尔 correct）
  allreduce_lat.py              双卡 all_reduce / all_gather 延迟（必须 tee 落盘才算实测）
  attrib_tp_vs_dtype.py         TP1/TP2 十轮内容对比，输出翻转题集合与 Δlp
  run_matrix.sh                 一把梭：起服 → 探针 → 落盘 → 关服（矩阵里的一格）

docs/
  EVIDENCE_INDEX.md             公告每条结论 → 本包哪条命令 / 哪个产物
  NOISE_POLICY.md               噪声与引用纪律（公告 §12 的精简版）
  N1_公告_v3_公开版.md          N1 公告 v3 快照（路径已 `$BASE` 化，附 11-20 说明）
  N1_闭合状态表.md              ★ 公告 §10 十条的现行状态：6 闭 / 2 半开 / 2 开，逐条写"能引到哪、缺什么才算闭"
  A8_GSM8K全量1319_结论.md      GGUF 双 dtype 全量 1319 的结论与复算
  A8_GGUF卸载门栓矩阵.md        12 格子集矩阵（含 11-20 修订：总分恒定、逐题不恒定）
  SE-03_teacher-forcing判定.md  SE-03 灰区归因（机制 = spec-decode）
  N2_GB10双机MoE配方_v0.1.md    ★ GB10 双机 MoE 配方 v0.1：PP=2/TP=2 两条路线、长上下文八档、双栈、NVFP4 8 形状、覆盖边界（英文版 `docs/N2_dual_node_recipe_v0.1.en.md`）

results/                         空目录，跑出来的东西按 run_<tag>/ 放，不要覆盖别人的
```

**中英对照**（英文版均为全文翻译，术语与数字逐位对齐；中文为准）

| 中文 | 英文 |
|---|---|
| `README.md` | `README.en.md` |
| `CHANGELOG.md` | `CHANGELOG.en.md` |
| `N1_data_index.md` | `N1_data_index.en.md` |
| `docs/N1_公告_v3_公开版.md` | `docs/N1_v3_executive_summary.en.md`（执行摘要，全文仅中文） |
| `docs/N1_闭合状态表.md` | `docs/N1_closure_status.en.md` |
| `docs/NOISE_POLICY.md` | `docs/NOISE_POLICY.en.md` |
| `docs/EVIDENCE_INDEX.md` | `docs/EVIDENCE_INDEX.en.md` |
| `docs/A8_GSM8K全量1319_结论.md` | `docs/A8_GSM8K_full1319_conclusion.en.md` |
| `docs/A8_GGUF卸载门栓矩阵.md` | `docs/A8_GGUF_offload_matrix.en.md` |
| `docs/SE-03_teacher-forcing判定.md` | `docs/SE-03_teacher_forcing_verdict.en.md` |
| `docs/N2_GB10双机MoE配方_v0.1.md` | `docs/N2_dual_node_recipe_v0.1.en.md` |

---

## 2. 最短路径（单卡跑一次 FP8 TP=2 探针）

```bash
# 0) 准备：模型放本地，路径自己改
export MODEL_DIR=/path/to/Qwen3.6-35B-A3B-FP8
export WORK=/path/to/this/repo

# 1) 起服（后台，日志落盘）
bash scripts/serve_fp8_tp2.sh "$MODEL_DIR" 8230 results/run_fp8_tp2/serve.log

# 2) 等 health，再跑三轮 20 题
for r in 1 2 3; do
  python3 scripts/probe_20_prompts.py \
    --base-url http://127.0.0.1:8230 --model qwen36-fp8 \
    --tag "fp8_tp2_r$r" --out results/run_fp8_tp2/r$r.json --repeat 1
done

# 3) 对比期望指纹（口径见 data/fingerprints/expected.md5）
python3 - <<'PY'
import json, hashlib
def canon(p):
    d = json.load(open(p, encoding="utf-8"))
    proj = []
    for r in d.get("results", []):
        tcs = []
        for tc in (r.get("tool_calls") or []):
            f = tc.get("function") or {}
            tcs.append([tc.get("type"), f.get("name"), f.get("arguments")])   # 不含 tc.id
        tcs.sort(key=lambda x: json.dumps(x, ensure_ascii=False))
        proj.append([r["id"], r.get("content") or "", tcs])
    return hashlib.md5(json.dumps(proj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
print(canon("results/run_fp8_tp2/r1.json"))   # FP8@TP2 triton 三轮基准 = c8db9a77c6d7b58a40df9d66f75344e7
PY
```

期望结果：`results/run_fp8_tp2/r{1,2,3}.json` 的**内容投影 md5 三次相同**（`c8db9a77…`）；通过 **18 / 20**（0.90），失败题固定为 `num_mul` + `logic_weekday` —— 这两题在 BF16 上同样错，**不是量化掉点**。包内已附同一栈的参考跑 `data/reference_fp8_tp2/`（r1–r3 + meta），可离线复算指纹。

---

## 3. 复现"某个后端是否确定"的标准动作

```bash
# A. 单栈十轮（同参数，只变轮次）
for r in $(seq 1 10); do
  python3 scripts/probe_20_prompts.py --base-url ... --model ... \
    --tag "det_r$r" --out results/run_det/r$r.json --repeat 1
done
python3 scripts/attrib_tp_vs_dtype.py --glob "results/run_det/r*.json"

# B. 换一条轴（只变 TP，dtype 不动），再跑十轮 → 归因到底是 dtype 还是 TP
for r in $(seq 1 10); do
  python3 scripts/probe_20_prompts.py --base-url ... --model ... \
    --tag "tp2_r$r" --out results/run_tp2/r$r.json --repeat 1
done

# C. BF16 对照（同版本同 TP）
#    注意：BF16 权重 71.9 GB，单卡 48 G 装不下 → TP=1 这格是硬件不可达，不是"没做"
```

判据：
- 十轮内容投影 md5 **全同** ⇒ 该栈确定（可作参照）；
- 出现翻转 ⇒ 记**翻转题集合 + 最大 Δlp + 唯一轨迹数**，不要只报"有时对有时错"。

---

## 4. 不该做的事（踩过坑）

| 不要 | 为什么 |
|---|---|
| 把两组不同窗口的 tok/s 相减 | 同机 3 分钟可漂 ~18 %，相减得到的差值不是效应 |
| 用两次独立回归的绝对值算加速比 | 只能用配对（同窗口、交替）的比值 + 胜场 + 置换 p |
| 把 BF16 权重塞进主机内存当"能跑" | 通讯/带宽全变，失去测量意义 |
| 拿 `is_supported()` 当"可用后端清单" | 它只答"能不能编译"，不答 QuantKey 契约；`cutlass`、`emulation` 都是探测过、起服拒 |
| 用 `pgrep -f 'vllm serve'` 判断有没有服务 | 会匹配到自己的 `ssh bash -c` 命令行，必然假阳 |
| 把 `±5 pp` 当通过门 | 1319 题单跑标准误 ≈1.0 pp，5 pp 门什么都筛不出来 |
| 把整块 `tool_calls`（含服务端生成的 id）拿去算指纹 | id 每跑都变，会把"确定"误判成"不确定"；剔 id 只留 `type/name/arguments` |

---

## 5. 许可与数据

- 代码：**Apache-2.0**，全文见 `LICENSE`（含专利授权条款；允许商用与再分发，须保留版权与免责声明）。
- `s104b_prompts.json` 为自造题集，可随包分发；`gsm8k_test_1319.jsonl` 来自 GSM8K（MIT），随包只放**取数脚本与指纹**，不直接再分发全量数据。
- 所有路径在本包里一律写 `$WORK` / `$MODEL_DIR`，**不写机器名、内网 IP、绝对家目录**。

## 6. 已知缺口（诚实清单）

- 并发曲线（c=1…24）只有实测数字与日志，**没有一键脚本**（`vllm bench serve` 的参数组合在 `docs/EVIDENCE_INDEX.md` 里给了命令行原文）。
- GGUF / llama.cpp 那一族**本包不收脚本与权重**，只收 **N2 双机配方里的对照读数**（pp512 831.78 ± 12.07 / tg128 12.46 ± 0.05，见 `docs/N2_GB10双机MoE配方_v0.1.md` §7）。
- **D8 三项（长上下文八档 / SGLang·llama.cpp 双栈 / NVFP4 8 形状）已进本包**，全部走 `docs/N2_GB10双机MoE配方_v0.1.md`；原先「GGUF 一族不进包」的口径作废。
- SM121（GB10）上的复现脚本与 SM120 共用，但**镜像版本不同**，见 `env/docker.md`。
- `a6` teacher-forcing（灰区归因）脚本尚未验证过，**未纳入 scripts/**。
- **FP8 + `triton` + TP=2 格的跨机复现做不出来**：现有第二台 SM120 机器是单卡 32 GB 且无 FP8 权重；SM121/GB10 机器同样没有 FP8 权重。能补的只有 NVFP4 格（见 §7.4，且它被证明不确定）。

## 7. 验收记录

### 7.1 同机换目录（2026-11-19）

把本包整体拷到一台**全新目录**（不是原实验目录）后，用 `scripts/run_matrix.sh` 跑 FP8 + `triton` + TP=2：

| 判据 | 结果 |
|---|---|
| 起服 | `RESULT=HEALTH_OK`，60 s 到健康；KV 1,355,776 token / 8,192 ctx ⇒ 165.50× |
| 有效后端 | `Selected CutlassFp8BlockScaledMMKernel for Fp8LinearMethod` + `Using TRITON Fp8 MoE backend` |
| 判分 | **18 / 20 × 3**，失败 `num_mul` + `logic_weekday`（两题 BF16 同错） |
| 指纹 | 三轮同一 `c8db9a77c6d7b58a40df9d66f75344e7` |
| 路径可移植性 | 只改 `MODEL_DIR` 与端口，未回连原实验目录（`$BASE`）任何路径 |

**诚实边界**：这是"同机换目录"，验证的是**路径可移植、无隐藏依赖**；**不是**"全新机器"（该机仍装着同一套 venv 与模型）。

### 7.2 发布包自解包复跑（2026-11-20）

把 `sm12x-repro-v1.0.0.zip`（sha256 `7942b38788d39d4185b43b186566fb6da79892f7d6d2871d88f6aff045fe30c6`，29 文件）传到机器上 `unzip` 到空目录，**不触碰原实验目录 `$BASE`**，直接跑 `scripts/run_matrix.sh`（同一格 FP8 + `triton` + TP=2）：

| 判据 | 结果 |
|---|---|
| 解包完整性 | sha256 与发布值一致；29 文件，`py_compile` 全过 |
| 起服 | `RESULT=HEALTH_OK`，80 s 到健康 |
| 有效后端 | `Selected CutlassFp8BlockScaledMMKernel for Fp8LinearMethod` + `Using TRITON Fp8 MoE backend` |
| KV | 1,355,776 token / 8,192 ctx ⇒ 165.50× |
| 判分 | **18 / 20 × 3**，失败 `num_mul` + `logic_weekday`（与 7.1 完全同集合） |
| 指纹 | 三轮同一 `c8db9a77c6d7b58a40df9d66f75344e7`，与 7.1 基准**逐字节相同** |
| 证据 | `data/reference_replay_1120/`（r1–r3 + `meta.txt`） |

### 7.3 这两次验证到底证明了什么（必读，别外推）

- 7.1 / 7.2 跑在**同一台 48 GB 机器**（原 FP8 实验机）上，换的只是目录与解包副本 ⇒ 证明的是「**包能从压缩包零解包、自洽、与基准逐字节确定地复现**」，**不是**"跨机复现"。
- **机械面**确实在**第二台机器**上验过：`git clone` → 29 文件 → `python3 -m py_compile scripts/*.py` 全过 → 路径泄漏检查 `LEAK_FREE` → `probe_20_prompts.py --help` 正常。
- **GPU 面换机**在 FP8 格上仍未验：另一台 SM120 机器是**单卡 32 GB 且无 FP8 权重**，物理上跑不了 2-way FP8 —— 见 7.4 的替代做法与结论。按 `env/versions.md` 换机实测之前，**不要把 7.1 / 7.2 写成"换机复现"**。

### 7.4 换机复跑（2026-11-20，第二台机型）：只能落 NVFP4 格，且它不确定

第二台机器为 1× RTX PRO 4500 32 GB、vLLM 0.29.0、模型目录只有 NVFP4 ⇒ FP8 + TP=2 那一格**物理不可达**。于是用同一支 `run_matrix.sh` / 同一份 20 题 / 同一解析器 / 同一版本，在两台机型上各跑 NVFP4 + `marlin` + TP=1 三轮：

| 机器 | r1 | r2 | r3 | 三轮同 md5 |
|---|---|---|---|---|
| A：2×PRO 5000 48 G（用 GPU0） | 17/20 | 17/20 | 17/20 | 否 |
| B：1×PRO 4500 32 G | 17/20 | 17/20 | **18/20** | 否 |

两个必须分开说的结论：

1. **跨机首轮逐字节相同**：两机 r1 内容投影 md5 均为 `74dc12605c34eb0206372f5dcd48f02d`，仅 `latency_s` 不同 ⇒ 换机型、换显存容量，首轮回答不受影响。
2. **同一服务内跨轮不等价**：A 机 2 题、B 机 3 题文本变化，B 机 r3 的 `code_bug` 由错改对（17/20→18/20）⇒ **NVFP4 / marlin 这一格不能当确定性参照**。这与 `docs/NOISE_POLICY.md` §2.1（"NVFP4 的差异是内核非确定性"）在第二台机型上互相印证。

证据：`data/reference_crossmachine_nvfp4/`（两机 r1–r3 + 各机 meta）。

**一句话总账**：可当确定性参照的只有 FP8 + `triton` + TP=2 格（三轮恒 `c8db9a77…`），它的跨机证据本次补不出来；能补的 NVFP4 格恰好证明了"这格不确定"。**这个缺口要如实留在包里，不要用人机交替换算成"跨机已复现"。**
