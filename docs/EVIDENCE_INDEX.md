# 证据索引：公告每一条结论 → 本包怎么跑出来 / 原始产物在哪

English：[`EVIDENCE_INDEX.en.md`](EVIDENCE_INDEX.en.md)（本文件为准）

> 路径约定：`$WORK` = 本包根目录；`$BASE` = 原始实验目录（跑这套实验时的根目录；按你的本地布局替换）。
> 原则：**没有可复现产物的结论不进正文。**

## 1. 精度类结论

| 公告条目 | 结论 | 怎么复现 | 原始产物 |
|---|---|---|---|
| §1.1 L1 禁用 | 非 ue8m0 直连 DeepGEMM → 明确崩（SM120 `gemm.hpp:346`；SM121 在 `0.2.3.post2` 上是 `:415`），不是静默 NaN | **包内直接跑**：`python3 scripts/s104_semantic.py`（需 DeepGEMM + 非 ue8m0 的 FP8 权重；行号随版本变） | `data/reference_l1_semantic/s104_result_sm120.json`（SM120 / 2.8.1，末条 `error`）+ `s104_result_sm121_dg023post2.json`（SM121 复跑，末条 `error`；来源与 sha256 见同目录 `README.txt`） |
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
| SGLang 的 −6.31 % 吞吐（分母 = `marlin`；反向 +6.74 %） | 原始产物是 101 `$BASE/sm12x/results/sglang_result_*.json` + `gsm8k_*.jsonl` 共 12 份（N1 §6 列全名），但**配对执行器未随包发布**，所以只能复核、不能一键重跑 |
| 5080 节点 3 分钟漂 18 % | 另一台机型，属噪声证据，不是本包结论 |
| `flashinfer_cutile` 容量数字（16/48 MiB） | 需要 32G 卡上单跑该后端，已在公告正文给数，脚本同 §2 |


## 5. N2（GB10 双机 MoE 配方）的证据映射

> 英文全文版：`docs/N2_dual_node_recipe_v0.1.en.md`（数字与本节逐位一致，中文为准）。

| 结论 | 本包位置 | 旁证 / 边界 |
|---|---|---|
| PP=2 打通（NGC 26.08 + Ray + `--enforce-eager`），起服 270 s、health 200 | `docs/N2_GB10双机MoE配方_v0.1.md` §3–§4 + §14.2 附录 `pp2_ngc_entry.sh` 原文 | 容器 run 参数见 §14.1（`--gpus all --ipc=host` 缺一即死）；原始日志未随包（48–59 KB），按 §10 清单在机器上取 |
| 单流 **14.71 / 14.72 t/s**（120 tok / 8.16 s），冷启首请求 4.4 弃用 | 同上 §5.1 | 探针 `pp2_e2e.py`（随包 `scripts/` 未含，见 §10）；三次日志 `pp2_e2e_r{1,2,3}` |
| 4 并发 **区间 25–43 t/s**（25.5 / 43.2，`max_tokens 300`） | 同上 §5.1 | **点值不可用**；12:35 那次 32.6 无落盘，仅存历史 |
| KV **243,733 token / 29.75×**，PP0 24.32 / PP1 25.01 GiB | 同上 §5.1 与起服日志 | 12:32 的 249,232 / 30.42× 属同工况 ±2%，其中 19.65 GiB / 2,688,671 那行**已整行撤回**（不可复现） |
| 跨机 **TP=2 可通**（anemll 0.1.1 + `mp`，`:8888`），长上下文八档跑在其上 | 同上 §6 + §14.3/§14.4 附录 | 起服收据 `tp2s.out`（HEALTH_OK 266 s）；八档源 `d8a_longctx.json` 逐格可复算 |
| SGLang prefill **4,468 / 4,597 t/s**、decode **31.5 / 31.7 t/s** | 同上 §7 | 13:39 复测落盘；warning 原文 `Config file not found … NVIDIA_GB10.json`；旧控制台 4,534–4,648 不作引用 |
| llama.cpp pp512 **831.78 ± 12.07** / tg128 **12.46 ± 0.05** | 同上 §7 | 仅作对照栈，脚本与权重不进包 |
| NVFP4 大形状 **3.6–4.3×**、小形状多数打平（例外 3.28×）、误差 13.9–16.7%、FP8 列 `Unknown recipe` | 同上 §8 | 8 格原始表在正文；构建环境必须 `$BASE/envs/sm120-t211`（torch 2.14.0+cu130）且先 `cd $BASE/src/DeepGEMM-vllm` |
| anemll 0.1.1 镜像 PP=2 判死（`stride 16`） | 同上 §9 | **无落盘、且不可重跑**（两机权重 `dsv4-abliterated` 均不存在） |

