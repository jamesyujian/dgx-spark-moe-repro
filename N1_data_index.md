# N1 数据索引

> 路径约定：`$BASE` = 原始实验根目录（Linux 侧）；本包内文件写相对路径。一行一条结论，不做解释、不做推荐。

| # | 结论（N1 条目） | 原始文件 | 条件 | 状态 |
|---|---|---|---|---|
| 1 | §1.1 L1 · SM120 明确失败（`gemm.hpp:346` / `CUDA 719`） | **原件已随包**：`scripts/s104_semantic.py`（7 个用例 + 1 个隔离子进程用例）、`data/reference_l1_semantic/s104_result_sm120.json`（8 条 `verdict`，末条 `error` = `gemm.hpp:346` 崩溃） | 116 `~/dgtest/` / PRO 4500 32G / DeepGEMM 2.8.1 直连 / `disable_ue8m0_cast=True` | ✔ 2026-11-20 取回并逐条核对：7 `ok`/`suspicious`、1 `error` |
| 2 | §1.1 L1 · SM121 失败形态（原记录：静默全 NaN；`exit=0`，无 warning） | **复跑产物随包**：`data/reference_l1_semantic/s104_result_sm121_dg023post2.json`（8 条 `verdict`，末条 `error` = `gemm.hpp:415` 断言）；原 2.8.1 静默输出仍无文件 | 101 `~/l1run/` / GB10 / SM121 / `deep_gemm 0.2.3.post2` / 同一脚本 / 子进程隔离 | ○ 2026-11-20 复跑：7 条与 SM120 逐字节同判（含 hash `f92674f9804a1e90`），第 8 条**明确断言** ⇒ 静默未复现；**版本未对齐（101/102 无 2.8.1 的 aarch64 构建）** |
| 3 | §1.2 L2 · TP1 34/40 | `$BASE/s104b/results/s104b_fp8_single_triton.json` | 115 / 0.29.0 / PRO 5000 48G / TP=1 / `--moe-backend triton` | ○ 在 115 |
| 4 | §1.2 L2 · TP2 36/40、失败集与 BF16 相同 | `s104b_fp8_tp2_triton.json`；参照 `s104b_bf16.json`（TP2，11-15） | 115 / 0.29.0 / TP=2 / triton | ○ 在 115 |
| 5 | §1.2 L2 · GSM8K 0.8453 vs BF16 0.8491，−0.38 pp，p=0.63 | `s104b/results/gsm8k_fp8_tp2_triton.jsonl`；BF16 侧 `$BASE/results/gsm8k_bf16_run2.jsonl`（同在 115） | 同版本同 TP / 1319 题 / 逐题配对 | ○ 在 115（两侧同机） |
| 6 | §1.3 L3 · `hpc` / `cutlass` / `flashinfer_cutlass` 三拒 | `$BASE/logs/fp8_matrix/{single_hpc,single_cutlass,p3_single_flashinfer_cutlass}.log` | 115 / 0.29.0 / TP=1 或 2 / 换 `--moe-backend` | ○ 在 115 |
| 7 | §4.1 · 115 / 48G / 0.29.0 九后端矩阵 | `$BASE/results3/manifest.csv`、`$BASE/logs3/*_serve.log`（27 个日志） | 115 / 0.29.0 / NVFP4 / 9 后端各一次 | ○ 在 115 |
| 8 | §4.1 · 116 / 32G / 0.31.0 各后端 | `$BASE/logs/probe2610_*_full.log`（b12x / cutlass / vllm_cutlass 等） | 116 / 0.31.0 / 容器口径 | ○ 在 116 |
| 9 | §4.1 · 101/102 GB10 0.27.1 各后端 | `$BASE/d4/logs/d4r*_sm121_*.log`、`$BASE/d4/results/d4r_nvfp4_*_sm121_*.json` | GB10 / 0.27.1 / 容器口径 | ○ 在 101/102（15 个 `d4r*_sm121_*.log`、每机 2 个 `d4r_nvfp4_*_sm121_*.json` 已确认） |
| 10 | §4.1 · `cutedsl@SM120` 设备级拒绝（0.31.0，40 s） | `$BASE/logs/a7_cutedsl_sm120{,_container}.log` | 116 / 0.31.0 / `--moe-backend flashinfer_cutedsl` | ○ 在 116 |
| 11 | §5.1 · 115 `marlin` 10 轮：logprob 全等 0/20、Δlp **1.263** | `$BASE/d4/results/s104i_base_marlin_r1..10.json` | 115 / 0.29.0 / 48G / TP=2 / temp=0 / seed=0 / 同进程 | ✔ 2026-11-20 重算 |
| 12 | §5.1 · 115 `humming` 10 轮：0/20、Δlp **1.529** | `s104i_base_humming_r1..10.json` | 同上，换后端 | ✔ 重算 |
| 13 | §5.1 · 116 `marlin` 10 轮：1/20、Δlp **1.899** | `$BASE/d4/results/s104j_116_marlin_r1..10.json` | 116 / 0.31.0 / 32G / `marlin` | ✔ 重算（文本/`num_unit`/logprob） |
| 14 | §5.1 · 116 `humming` 10 轮：0/20、Δlp **1.897** | `s104j_116_humming_r1..10.json` | 116 / 0.31.0 / 32G / `humming` | ✔ 重算 |
| 15 | §5.1 · BF16 TP=2 十轮 20/20、Δlp **0.000** | `$BASE/d4/results/s104k_bf16_r1..10.json` | 115 / 0.29.0 / TP=2 / 同批 20 题 | ✔ 重算（Δlp 精确 0.0000） |
| 16 | §5.1 · SM121 四栈 0/80、Δlp 1.62–2.09 | 101 `$BASE/d4/results/s104x3_det_{marlin,flashinfer_b12x}_sm121_101_r1..10.json` | GB10 / 0.27.1 / `marlin` 与 `b12x` 各 10 轮 | ○ 在 101（r1–r10 已确认） |
| 17 | §5.2 D · `--enforce-eager` 后 Δlp 上升（1.258 → **1.905**，同栈 3 轮对 3 轮） | `s104f_det_humming_r1..3.json`、`s104f_eager_humming_r1..3.json` | 115 / 0.29.0 / `humming` / `--enforce-eager` + `--max-num-seqs 1` + `--max-num-batched-tokens 2048` | ✔ 重算（+0.647；3 轮子集 SD 0.071） |
| 18 | §5.2 C · `VLLM_BATCH_INVARIANT=1` 起服失败 | `$BASE/d4/logs/`（起服失败日志）；摘录见 `2026-11-16_D4_执行记录.md` | 115 / 0.29.0 / NVFP4 W4A16 | ○ |
| 19 | §5.3 · 13 道稳定题（文本）× 4 栈 × 10 轮 = 40 次 | 第 11–14 行四组文件（`s104i_*`、`s104j_*`） | 115/116 / 0.29.0、0.31.0 / `marlin`、`humming` / 各 10 轮 | ✔ 重算（13/20 文本恒定；`zh_en` 仅 116 `humming` 翻 2 种） |
| 20 | §5.3 · TP 归因：TP1 翻 5/20、TP2 翻 6/20、BF16@TP2 翻 0/20 | `s104i_base_marlin_r1..10`（TP1）、`s104h_nvfp4_tp2_r1..10`（TP2）、`s104k_bf16_r1..10` | 115 / 0.29.0 / ctx 2048 / util 0.90 | ✔ 重算（TP2 侧 `num_unit` 10 轮 3 种值） |
| 21 | §5.3 · BF16@TP=1 硬件不可达（71.9 GB > 47.8 GiB） | 无非原文件；`du -sb` 与 `nvidia-smi` 输出见 `2026-11-19` 记录 | 115 / BF16 权重 71,926,861,539 B / 单卡 48,935 MiB | ⛔ 物理不可达 |
| 22 | §5.4 · `num_unit` 116 `marlin` 1/10、`humming` 8/10 | `s104j_116_{marlin,humming}_r1..10.json` | 116 / 0.31.0 / 32G / `seed=1234` / `max_tokens=40` | ✔ 重算（1/10、8/10） |
| 23 | §5.4 · `num_unit` 3 进程 × 10 次：4 / 6 / 0 | `$BASE/d4/results/num_unit_cycles.jsonl` | 116 / 0.31.0 / `marlin` / 三个独立进程 | ○ 在 116 |
| 24 | §5.4 · 115 三栈 `num_unit`：`marlin` 10/10、`humming` 9/10、BF16 10/10 | `s104i_base_{marlin,humming}_r1..10`、`s104k_bf16_r1..10` | 115 / 0.29.0 | ✔ 重算（10 / 9 / 10；答案文本恒为 `3584`，logprob 仍不逐位相同） |
| 25 | §5.5 · 四组 5 轮 correct（1098…1111） | 115 `$BASE/d4/results/gsm8k_0{29,31}_r*.jsonl`；116 `gsm8k_116{mar,hum}_r{1..5}.jsonl` | 四栈各自 5 轮 × 1319 | ○ 在 115 / 116（本轮仅取 r4–r5 算 `p_d`） |
| 26 | §5.5 · 分辨力 `p_d ≈ 4.7 %` → 阈值 1.17 pp | `gsm8k_031_r4.jsonl` vs `r5.jsonl`（b=29 / c=33 = 62/1319） | 115 / 0.31.0 / `marlin` / 同栈两跑逐题配对 | ✔ 重算（0.0470；另三对 0.0425 / 0.0478 / 0.0493） |
| 27 | §5.5 · 三版本 9 跑定标 83.573 / 83.472 / 83.321 % | `$BASE/results3/gsm8k_{A,B,C}_r{1,2,3}.jsonl` + `manifest.csv` | 115 / 0.27.1、0.29.0、0.31.0 / 每版 3 跑 | ○ 在 115 |
| 28 | §6 · SGLang `marlin` 比 `flashinfer_cutlass` 快 6.31 % | 101 `$BASE/results/gsm8k_sglang_marlin_r{2,3}.jsonl` + `gsm8k_cutlass101_r{2,3}.jsonl`（另有基线对 `gsm8k_sglang_{cutlass,sm121}.jsonl`）；102 `gsm8k_sglang_cutlass_r{2,3}.jsonl` | 101/102 / SGLang 0.5.23 / 3 对 3 | ○ 在 101/102（本轮只确认文件在机；r1 未见落盘，无编号那对为基线） |
| 29 | §6 · 精度侧 p=0.55（b=25 / c=20） | 同上 jsonl（逐题 verdict） | 101 / 同栈配对 | ○ 在 101/102（未重算 b/c） |
| 30 | §10.1 · tool calling 必须 `qwen3_xml`（tc 3/3） | `$BASE/d4/results/a10e_{fp8xml,nvfp4xml}_r{1,2,3}.json` | 115 / 0.29.0 / FP8 与 NVFP4 两栈各 3 轮 | ○ 在 115 |
| 31 | §10.9 · 并发曲线 209 → 1011 tok/s（c=1…24） | `$BASE/d4/logs/a9_concurrency_sweep_v{2,3}.log` | 115 / `vllm bench serve` / input 512 output 128 | ○ 在 115 |
| 32 | §3 通信 · all_reduce 4K 9.8 µs / 512K 80.8 µs | `$BASE/logs/fp8_matrix/allreduce_w2.txt` | 115 / world=2 / 200 次每档 / NCCL 2.30.7 | ○ 在 115（已落盘） |
| 33 | A6 / SE-03 · 首 token `109` vs `123`（5.38 / 2.88 nats）、机制 = spec-decode | 101 `$BASE/d4/results/se03_tf2_{b12x,dgemm,dgemm_formula}_101.json`；脚本 `/tmp/se03_tf2.py`、`iso_probe.py`（`/tmp` 未核） | 101+102 / DeepSeek-V4-Flash / TP=2 / temp=0 | ○ 在 101（3 个 json 已确认） |
| 34 | A8 · GGUF 全量 1319：Q4 = Q8 = 1128/1319 | 101 `$BASE/results/full_q4ngl99full.jsonl`；102 `$BASE/results/full_q8ngl99full_102.jsonl` | 101+102 / llama.cpp / `-ngl 99` / ctx 8192 / `max_new_tokens=512` | ○ 在 101/102（两个全量 jsonl 已确认） |
| 35 | C3 · FP8+triton+TP2 自解包复跑 18/20 × 3、md5 `c8db9a77…` | 本包 `data/reference_replay_1120/r1..3.json` + `meta.txt`；原件 115 `~/c3verify/pkg/results/` | 115 / 0.29.0 / 2×48G / 新解包副本 | ✔ 随包（可直接重算 md5） |
| 36 | C3 · NVFP4+marlin+TP1 跨机型首轮 md5 `74dc1260…` | 本包 `data/reference_crossmachine_nvfp4/{115,116}_r1..3.json` + `meta_*.txt` | 115（2×48G）/ 116（1×32G）/ 0.29.0 / TP=1 | ✔ 随包 |
| 37 | C3 · 基准轮（11-19）18/20 × 3、KV 165.50× | 本包 `data/reference_fp8_tp2/r1..3.json` + `meta.txt` | 115 / 0.29.0 / FP8+triton+TP=2 | ✔ 随包 |

## 状态标记

| 标记 | 含义 |
|---|---|
| ✔ | 2026-11-20 已按原始文件逐项重算，数字与正文一致 |
| ○ | 原始文件在机（115 或 116），本轮未重算 |
| ⏳ | 原文件应在 101/102，但本轮在落盘目录未定位到（不作为已验） |
| ⛔ | 物理不可达 / 无镜像 / 无落盘，已在正文写明原因 |
