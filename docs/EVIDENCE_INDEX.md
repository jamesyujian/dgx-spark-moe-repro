# 证据索引：公告每一条结论 → 本包怎么跑出来 / 原始产物在哪

English：[`EVIDENCE_INDEX.en.md`](EVIDENCE_INDEX.en.md)（本文件为准）

> **【权威声明 · 2026-11-25】** 本文件是 SM120/SM121 证据索引的**唯一权威版本**，`EVIDENCE_INDEX.en.md` 为同源双语对照（数字逐位一致，中文为准）。同名文件在本地共 7 份，其余 **5 份已于 2026-11-25 加冻结横幅并置为只读**（清单见 §6），一律不得作为引用依据。此后任何台账增补只改本文件及其 `.en.md`。

> **【数字定值 · 2026-11-25 冻结】** 下表为**对外唯一口径**：白皮书、看板、报告、Release note 引用时必须逐字一致；一旦不一致，视为该文档写错，**改文档、不改本表**。要改本表，必须同时给出新实测原件路径与新的证据等级。

| 量 | 定值 | 引用约束 |
|---|---|---|
| 2 节点 KV 池 | **243,733 token @ 8,192 = 29.75×** | 249,232 只作同工况 ±2 % 旁证；19.65 GiB / 2,688,671 行已整行撤回；**禁写「声明上限」** |
| 115 NVFP4/BF16 | **1.56×**（TP=2·c4，472.76 → 739.68） | **8.28× 永久作废**；引用必带「TP=2 / c4」 |
| 115 TP 扩展 | **1.13×**（c4，654.3 → 739.68） | 单机 PCIe 口径，禁作跨机推论 |
| 101 / 102 NVFP4/BF16 | **2.47× / 2.37×** | 单机 TP=1 口径 |
| 115 并发 | c1 **250.8** / c2 357.4 / c4 654.3 / c8 **1,017.2** / c12 1,314.02 / c16 **1,551.75** | 只引整段均值，禁引单窗（心跳窗口仅 15–23 请求） |
| 101 并发 | c1 **81.8** / c2 124.7 / c4 **188.2** / c8 267.9 | c4 为 24 h 参照；c8 倍率已降至 1.42× |
| 能效 | **6.7**（GB10，GPU 域）/ **3.7**（PRO5000 c8，整卡 TBP）/ **5.19**（c16，整卡 TBP） | 6.7 **不可当整机能效**；三个分母必须并列 |
| 零衰减 | **+0.37 / +0.16 / 0.00 %**（101 / 102 / 115） | 首 1 h → 末 3 h；24.00 h、errs 0 |
| decode 引擎差 | 8k 档 SGLang 领先 marlin **15.3–16.0 %**、领先 b12x **20.5–21.2 %**；长上下文 16k/32k/64k 领先 marlin **17.3 / 19.6 / 24.6 %** | 5 rep 中位；分母必须写明对谁 |
| 网络 | **27.7 GB/s**（双向聚合，单向线速 25）；**NVLink = 1/14** | 两个都必须带参照/口径，禁裸写 |
| 模型体积 | **155.4 GiB** / 165.27 B 参数 / **7.63 bit/param**（分子=张量本体 146.8 GiB）；**整仓口径 8.08 bit/param**（155.4 GiB ÷ 165.27 B） | **禁写「原形 156 GB」**（该 156 GB 本身即官方混合量化发行版）；原形 ≈330 GB 只可标推算；**I8 专家为 1 字节/参数（138.0 GiB），禁写「2 参/字节」**（照 2 参/字节整仓只剩 86 GiB，与 `du` 差 69 GiB） |
| 能力门 | **155.4 GiB > 单机可用 121 G（+34.4 GiB）** | 禁写「换 fp8 约 78 G 就能装」（算术前提已作废） |
| 社区数 | 4 台环网单流 **90** / c16 **325** | 自媒体单次、无复核无落盘，**必须三标注** |
| 116 | **11-23 曾 13.84 h 后冻结、机制未定**；**16 h 复跑跨过该点未复现（≠已修复）** | 禁写「疑似 forward compat」「驱动落后」，**禁写「已修复」**；6 h 不覆盖冻结点，16 h 未复现仍不定因 |

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

> 英文全文版：`docs/N2_dual_node_recipe_v0.2.en.md`（数字与本节逐位一致，中文为准）。

| 结论 | 本包位置 | 旁证 / 边界 |
|---|---|---|
| PP=2 打通（NGC 26.08 + Ray + `--enforce-eager`），起服 270 s、health 200 | `docs/N2_GB10双机MoE配方_v0.2.md` §3–§4 + §14.2 附录 `pp2_ngc_entry.sh` 原文 | 容器 run 参数见 §14.1（`--gpus all --ipc=host` 缺一即死）；原始日志未随包（48–59 KB），按 §10 清单在机器上取 |
| 单流 **14.71 / 14.72 t/s**（120 tok / 8.16 s），冷启首请求 4.4 弃用 | 同上 §5.1 | 探针 `pp2_e2e.py`（随包 `scripts/` 未含，见 §10）；三次日志 `pp2_e2e_r{1,2,3}` |
| 4 并发 **区间 25–43 t/s**（25.5 / 43.2，`max_tokens 300`） | 同上 §5.1 | **点值不可用**；12:35 那次 32.6 无落盘，仅存历史 |
| KV **243,733 token / 29.75×**，PP0 24.32 / PP1 25.01 GiB | 同上 §5.1 与起服日志 | 12:32 的 249,232 / 30.42× 属同工况 ±2%，其中 19.65 GiB / 2,688,671 那行**已整行撤回**（不可复现） |
| 跨机 **TP=2 可通**（anemll 0.1.1 + `mp`，`:8888`），长上下文八档跑在其上 | 同上 §6 + §14.3/§14.4 附录 | 起服收据 `tp2s.out`（HEALTH_OK 266 s）；八档源 `d8a_longctx.json` 逐格可复算 |
| SGLang prefill **4,468 / 4,597 t/s**、decode **31.5 / 31.7 t/s** | 同上 §7 | 13:39 复测落盘；warning 原文 `Config file not found … NVIDIA_GB10.json`；旧控制台 4,534–4,648 不作引用 |
| llama.cpp pp512 **831.78 ± 12.07** / tg128 **12.46 ± 0.05** | 同上 §7 | 仅作对照栈，脚本与权重不进包 |
| NVFP4 大形状 **3.6–4.3×**、小形状多数打平（例外 3.28×）、误差 13.9–16.7%、FP8 列 `Unknown recipe` | 同上 §8 | 8 格原始表在正文；构建环境必须 `$BASE/envs/sm120-t211`（torch 2.14.0+cu130）且先 `cd $BASE/src/DeepGEMM-vllm` |
| anemll 0.1.1 镜像 PP=2 判死（`stride 16`） | 同上 §9 | **无落盘、且不可重跑**（两机权重 `dsv4-abliterated` 均不存在） |

## 6. 同名文件清点与冻结记录（2026-11-25）

清点结果：本地共 7 份 `EVIDENCE_INDEX*.md`。**权威集 = 2 份（本文件 + `.en.md`，同源双语对，同改同发）**；其余 5 份为历史副本，已加冻结横幅并置 `IsReadOnly = True`。

| # | 路径 | 处置 | 依据 |
|---|---|---|---|
| 1 | `2026-11-20\new-chat\sm12x-repro-release\docs\EVIDENCE_INDEX.md` | **权威（中文）** | 62 行、2026-11-22 13:48，含 2 节点 KV 等全部最新值 |
| 2 | 同上目录 `EVIDENCE_INDEX.en.md` | **权威（英文对照）** | 与 #1 逐位同源，同改同发 |
| 3 | `2026-11-18\new-chat-1\sm12x-repro\docs\EVIDENCE_INDEX.md` | 冻结只读 | 41 行、11-19 09:55，最旧 |
| 4 | `2026-11-20\new-chat\_verify106\ext\dgx-spark-moe-repro\docs\EVIDENCE_INDEX.md` | 冻结只读 | 44 行、11-20 09:40 |
| 5 | 同上目录 `EVIDENCE_INDEX.en.md` | 冻结只读 | 同上 |
| 6 | `2026-11-20\new-chat\_verify107\dgx-spark-moe-repro\docs\EVIDENCE_INDEX.md` | 冻结只读 | 44 行、11-20 10:07 |
| 7 | 同上目录 `EVIDENCE_INDEX.en.md` | 冻结只读 | 同上 |

**冻结判据**：副本 ≤44 行且不含 `243,733`（＝缺 2 节点 KV 与 §5 全节），与权威版不可并存为引用源。**解冻流程**（若确需追溯）：先去 `IsReadOnly`、删横幅、在 §6 本表登记用途与日期，不得直接改数。

