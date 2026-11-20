# 证据索引：公告每一条结论 → 本包怎么跑出来 / 原始产物在哪

English：[`EVIDENCE_INDEX.en.md`](EVIDENCE_INDEX.en.md)（本文件为准）

> 路径约定：`$WORK` = 本包根目录；`$BASE` = 原始实验目录（跑这套实验时的根目录；按你的本地布局替换）。
> 原则：**没有可复现产物的结论不进正文。**

## 1. 精度类结论

| 公告条目 | 结论 | 怎么复现 | 原始产物 |
|---|---|---|---|
| §1.1 L1 禁用 | 非 ue8m0 直连 DeepGEMM → 明确崩（`gemm.hpp:346`），不是静默 NaN | **包内直接跑**：`python3 scripts/s104_semantic.py`（需 DeepGEMM 2.8.1 + 非 ue8m0 的 FP8 权重） | `data/reference_l1_semantic/s104_result_sm120.json`（8 条 verdict，末条 `error`；来源与 sha256 见同目录 `README.txt`） |
| §1 FP8 L2 可用 | block-scaled 128×128 W8A8 能跑 | `scripts/run_matrix.sh $MODEL 8230 fp8_tp2 --tensor-parallel-size 2 --moe-backend triton` | `$BASE/s104b/results/s104b_fp8_tp2_{triton,tc}.json` |
| §1 L3 三后端拒 | `hpc`/`cutlass`/`flashinfer_cutlass` 各自拒绝 | 同上，换 `--moe-backend` | `$BASE/logs/fp8_matrix/{single_hpc,single_cutlass,p3_single_flashinfer_cutlass}.log` |
| §1 GSM8K 差 −0.38 pp | 同版本同 TP 下不可区分 | `scripts/eval_gsm8k.py` 两次（BF16 / FP8 各一跑） | `$BASE/results/gsm8k_bf16_run2.jsonl`、`$BASE/s104b/results/gsm8k_fp8_tp2_triton.jsonl` |
| §5 NVFP4 十轮不可复现 | 内核非确定性 | 十轮 `probe_20_prompts.py` + `attrib_tp_vs_dtype.py` | `$BASE/d4/results/s104i_base_marlin_r{1..10}.json`、`s104i_base_humming_r{1..10}.json` |
| §5.3 TP 归因 | TP 不是来源，kernel 才是 | TP=1 / TP=2 各十轮 + BF16@TP2 十轮 | `$BASE/d4/results/{s104h_nvfp4_tp2,s104i_base_marlin,s104k_bf16}_r{1..10}.json` |
| §10 tool calling | 必须 `qwen3_xml` | `probe_20_prompts.py` 三轮 ×2 栈 | `$BASE/d4/results/a10e_*_r{1,2,3}.json` |

## 2. 后端可用性结论（§4.1 矩阵）

| 格 | 怎么跑 | 产物 |
|---|---|---|
| 115 / 48G / 0.29.0 | `run_matrix.sh` 换 9 个后端各一次 | `$BASE/logs3/*_serve.log`、`$BASE/results3/manifest.csv` |
| 116 / 32G / 0.31.0 | 容器口径（见 `env/docker.md` §3）各后端一次 | `$BASE/logs/probe2610_*_full.log` |
| 101/102 / GB10 / 0.27.1 | 容器口径各后端一次 | `$BASE/logs/probe_116_2610*.log`（同族脚本） |
| **cutedsl@SM120** | 单容器 + `--moe-backend flashinfer_cutedsl` | `$BASE/logs/a7_cutedsl_sm120{,_container}.log` |

## 3. 通信与并发

| 结论 | 怎么跑（命令原文） | 产物 |
|---|---|---|
| all_reduce 4K 9.8 µs / 32K 16.0 / 128K 45.7 / 512K 80.8 | `torch.distributed.run --nproc_per_node=2 scripts/allreduce_lat.py`（**必须 tee 落盘**） | `$BASE/logs/fp8_matrix/allreduce_w2.txt` |
| 并发曲线 209 → 1011 tok/s（c=1…24） | `vllm bench serve --backend openai-chat --base-url http://127.0.0.1:<P> --endpoint /v1/chat/completions --model <M> --tokenizer <MODEL_DIR> --dataset-name random --random-input-len 512 --random-output-len 128 --num-prompts 32 --max-concurrency <C> --ignore-eos --seed <S>` | `$BASE/d4/logs/a9_concurrency_sweep_v{2,3}.log` |

> ⚠ bench 的两个坑（踩过）：① 不给 `--tokenizer` 本地路径会在离线机器上卡在 HF 拉取超时；② 输出接管道会被 grep 吃掉指标行，必须写文件再 grep。

## 4. 报告里出现但**不由本包产出**的数

| 数 | 为什么不在本包 |
|---|---|
| SGLang 的 −6.31 % 吞吐（分母 = `marlin`；反向 +6.74 %） | 原始产物是 101 `~/sm12x/results/sglang_result_*.json` + `gsm8k_*.jsonl` 共 12 份（N1 §6 列全名），但**配对执行器未随包发布**，所以只能复核、不能一键重跑 |
| 5080 节点 3 分钟漂 18 % | 另一台机型，属噪声证据，不是本包结论 |
| `flashinfer_cutile` 容量数字（16/48 MiB） | 需要 32G 卡上单跑该后端，已在公告正文给数，脚本同 §2 |
