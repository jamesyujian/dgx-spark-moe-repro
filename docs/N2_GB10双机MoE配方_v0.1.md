# GB10 双机 MoE 配方 v0.1（N2）

> **本包版本的脱敏说明**：家目录一律 `$BASE`（如 `p4/`、`envs/`）、权重目录 `$MODEL_DIR`、双机私网 IP `$HEAD_IP`（101）/ `$WORKER_IP`（102）；机号 101/102/115/116 按约定保留。附录脚本做了同批替换（`VLLM_HOST_IP`、`MASTER_ADDR` 处改回真实值即可跑）。原始未脱敏稿在个人工作区，不进包。
> 版本 v0.1 ｜ 2026-11-22 ｜ 机型：GB10（SM121，aarch64）× 2 台（node101 / node102）｜ 模型：DeepSeek-V4-Flash-DSpark（MoE，fp4 expert / fp8 KV）
> 本文只发**复现链 + 基准数字 + 覆盖边界**，不发推荐配置、不发调优建议。所有数字均标注是否落盘可复算。

---

## 0. 一句话结论

**两套跨机并行都通了，但走的是两个不同镜像**：PP=2 在 **NGC 26.08**，跨机 TP=2 在 **anemll 0.1.1**（详下）。容器 `nvcr.io/nvidia/vllm:26.08-py3`（vLLM 0.27.1）+ 容器内 `ray` + `--enforce-eager` + fp8 KV，head 起服到 health 200 约 **4 min**，单流 **14.71 / 14.72 t/s**（落盘）、4 并发 **只给区间 25–43 t/s**、KV **243,733 token**（均为 13:31 复测组，见 §5.1）。
**跨机 TP=2 可通**：`$BASE/p4/gen/head_tp2s.sh`（`--tensor-parallel-size 2 --nnodes 2 --node-rank … --distributed-executor-backend mp`）在 **anemll 0.1.1** 镜像上起服成功，`logs/tp2s.out` 记 `SERVE_RESULT=HEALTH_OK elapsed=266s 11:58:31 / DEPLOY_OK tp2s`，本文第 6 节的长上下文八档就是跑在这台 TP=2 服务（`:8888`）上。**注意镜像方向相反**：anemll 上 PP=2 不通（cutedsl 非 PP-aware，`stride 16` 硬失败），NGC 上 TP=2 未测。此前「vLLM 不支持跨节点 TP」的说法只对 **PyPI 公版 0.24 + `mp`** 成立（7-09 内部记录，机上无落盘），不能推广到 NGC。

---

## 1. 硬件 / 驱动 / 基线

| 项 | 值 |
|---|---|
| 节点 | 2 × GB10（NVIDIA GB10，compute capability **12.1**，aarch64），单卡 VRAM 124,546 MiB |
| 驱动 | **595.71.05**（101/102 实测同版本，容器内 `nvidia-smi` 直接透传；上层文档若出现别的驱动号一律按本条为准） |
| 互联 | 双机私网 **$HEAD_IP / $WORKER_IP**（bond0，slave `enP2p1s0f0np0`，active-backup，跨机 ping 0.6–0.9 ms） |
| 模型（PP=2 用） | 容器内 `/models/DeepSeek-V4-Flash-DSpark` ⇒ **宿主机** `$MODEL_DIR/DeepSeek-V4-Flash-DSpark`；MoE，`expert_dtype` 解析为 `fp4`，KV cache 格式 `fp8_ds_mla` |

> 路径注意：宿主机的 `/models` 是另一个 root 目录（不是容器里的 `/models`），而 `$MODEL_DIR/` 才是权重真实位置 ⇒ 直接在宿主跑 `vllm serve /models/DeepSeek-V4-Flash-DSpark` 会找不到；容器要 `-v $MODEL_DIR:/models`。
>
> 注意：bond0 的 `Primary Slave` 是运行时态，**重启会丢**；本配方不依赖哪个 slave 在 active，但两机必须同网段可达。

---

## 2. 组件清单

| 组件 | 版本 / 来源 | 说明 |
|---|---|---|
| 镜像 | `nvcr.io/nvidia/vllm:26.08-py3` | 容器内 vLLM **0.27.1**；**不要用 PyPI 公版 0.24**（DeepGEMM 与 NVFP4 modelopt 的 `scale_fmt=ue8m0` 不兼容，见 7-09 记录） |
| Ray | 容器内 `pip install ray==2.55.1 -i https://pypi.tuna.tsinghua.edu.cn/simple` | 镜像本身不带 ray；`--distributed-executor-backend ray` 硬依赖 |
| 入口脚本 | `$BASE/p4/pp2_ngc_entry.sh`（**两机同版本**，`NODE_RANK=0/1` 区分） | 内含「等 2.0 张 GPU 就绪」守卫，最多 72 × 5 s，超时打 `RAY_ONLY_1GPU_ABORT` 并 `sleep infinity`（不静默失败） |

---

## 3. 起服参数（两机一致，仅 `NODE_RANK` 不同）

```
vllm serve /models/DeepSeek-V4-Flash-DSpark \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --enforce-eager \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.82 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000 \
  --served-model-name dsv4-pp2
```

环境变量（两机一致）：

```
GLOO_SOCKET_IFNAME=bond0
NCCL_SOCKET_IFNAME=bond0
NCCL_IB_DISABLE=1
NCCL_NET=Socket
RAY_DISABLE_IPV6=1
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
HF_HUB_OFFLINE=1
```

后两个不是装饰：前者放开超过模型默认上限的 `--max-model-len`（实测 entrypoint 里就带着），后者让离线机不去连 HF Hub（不带会在无网环境卡在 tokenizer/config 拉取）。

三个参数不是可选项：

- `--enforce-eager`：跳过跨节点 CUDA graph 编译同步（7-09 与 122B 记录都验证过，不开就卡在编译同步）。
- `--distributed-executor-backend ray`：PP=2 跨机只能走 Ray。
- `--quantization fp8` + `--kv-cache-dtype fp8`：与容器内 DeepGEMM 路径对齐；换 bf16 会走出另一条未验证分支。

---

## 4. 起服流程（顺序是硬约束）

1. **node101 起 head**（`NODE_RANK=0`），等它打出 Ray head 地址。
2. **约 20 s 后起 node102 worker**（`NODE_RANK=1`）。
3. 轮询 `curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:8000/health`，200 即通。

顺序反了会看到：`RuntimeError: Engine core initialization failed. Failed core proc(s): {}`（GCS 已连上、引擎起不来，日志里看不出真因）。**此时不要原地重试 worker，先停 head 再按 1→2 重来。**

实测时间线（node 侧时间戳）：

| 事件 | 时刻 |
|---|---|
| head 容器启动 | **12:28:11** |
| health 200 | **12:32:32**（启动约 4 min，其中 DeepGEMM warmup 990 kernel 占 55 s） |
| 双容器拆除 | head `exit 0` 12:58:42 / worker `exit 137` 12:59:02 |

**跨机 TP=2 的起服（另一条路，anemll 0.1.1 + `mp`，`:8888`）**：与 PP=2 相反，**worker 先起**（`worker_tp2s.sh` 11:53:59）→ head 后起（`head_tp2s.sh` 11:54:04）→ `SERVE_RESULT=HEALTH_OK elapsed=266s 11:58:31 / DEPLOY_OK tp2s`（`$BASE/p4/logs/tp2s.out`）。启动约 4.4 min。注意 `tp2s.out`、`pp2_run.out` 开头的 `confirm: tp=1 pp=1` 是 wrapper 的固定回显，**与 `gen/*.sh` 实际参数不符，不能当参数依据**。

---

## 5. 基准数字（PP=2，`max-model-len 8192` / `max-num-seqs 4`）

> ⚠️ **角标：本表是 12:32 首次打通时的读数，已被 §5.1（13:31 带 tee 复测）取代。** 保留本表只为记录"首次打通那一刻"，**对外引用一律用 §5.1**；本表里的 14.72/14.79/14.82 与 32.6 均无落盘。

| 指标 | 值 | 备注 |
|---|---|---|
| 单流 decode | **14.8 t/s**（三次客户端读数 14.72 / 14.79 / 14.82） | ⚠ **未落盘**：三次读数只在控制台，机上只有服务端滚动行 14.7 / 14.8（`docker logs pp2ngc_head`，容器已于 12:58 拆除） |
| 4 并发 aggregate | **32.6 t/s**（1022 token / 31.3 s） | 折算单流 **8.16 t/s**，即并发下每流比单流低约 45% |
| KV cache 容量 | **249,232 token**，@8k = **30.42×** 并发 | 可复算（引擎启动日志） |
| 每 stage 可用 KV | **23.33 GiB（PP0） / 25.72 GiB（PP1）** | 同上日志 |
| KV 数值曾被误引 | ~~19.65 GiB / 2,688,671 token / 2.56×~~ | **已撤回**：该行来自已删容器的控制台，机上无日志；同次起服的 `$BASE/p4/logs/pp_worker_tp2s.log:5` 记的是 18.1 GiB |

### 5.1 13:31 复测（带 `tee` 落盘，可逐字复算）

同参数重启一次（13:24:25 起服 → 13:28:55 `HTTP server started`，**270 s**），探针与读数全部落盘：

| 指标 | 本轮落盘值 | 文件 |
|---|---|---|
| 单流 r2 / r3 | **14.71 / 14.72 t/s**（120 token / 8.16 s / 8.15 s） | `$BASE/p4/logs/pp2_e2e_r2/r3_20261122.log` |
| 单流 r1（冷启首请求） | 4.4 t/s（27.3 s，含首请求 warmup） | `pp2_e2e_r1_20261122.log` |
| 4 并发 r1 / r2 | **25.5 t/s**（1147 token / 44.91 s）｜**43.2 t/s**（1052 token / 24.33 s） | `pp2_conc_20261122.log` / `pp2_conc_r2_20261122.log` |
| KV | **243,733 token**，@8k = **29.75×** | `pp2ngc_head_20261122_1326.log` |
| 每 stage 可用 KV | **24.32 GiB（PP0） / 25.01 GiB（PP1）** | 同上 |

读法修正（重要）：

- **单流 14.7 可复算**，与 12:32 那组 14.72 / 14.79 / 14.82 一致；冷启首请求只有 4.4 t/s，**引用单流必须排除首请求**。
- **4 并发波动极大**：同参数相邻两跑 25.5 与 43.2（差 1.7 倍），12:35 那次 32.6 落在区间中间 ⇒ **该指标不能用单点值，只能给区间 25–43 t/s**，并注明跑了几次。原表「32.6 / 折算 8.16」保留为 12:35 读数（未落盘）。
- KV 本轮 243,733 / 29.75×，与 12:32 的 249,232 / 30.42× 相差 2%，属显存余量差异，不是版本差异。
- 日志文件由容器内 root 写出（`-rw-r--r-- root`），宿主机可读不可覆盖；要复用请先 `sudo chown $USER:$USER`。

> 14.8 t/s 与「TP=2 时测的长上下文」不是一个工况：下表 decode 18–20 t/s 是 TP=2 / spec on 的读数，不能与 PP=2 的单流 14.8 直接比。

---

## 6. 长上下文八档（跑在 TP=2 服务上）

工况：**anemll 0.1.1 镜像**、`--tensor-parallel-size 2 --nnodes 2 --distributed-executor-backend mp`、`:8888`、`--kv-cache-dtype nvfp4_ds_mla --block-size 256`、dspark spec on（5 token）、`--gpu-memory-utilization 0.85`、prefix caching + chunked prefill 开、`--max-model-len 1048576`。
源：`$BASE/p4/d8a_longctx.json`（落盘 12:07:24，逐格可复算）；探针 `$BASE/p4/d8a_longctx.py` 里 `BASE=127.0.0.1:8888`，就是这台 TP=2 服务。
⚠ **权重不同**：这台服务用的是 `/models/dsv4-abliterated`（宿主上已找不到同名目录），而第 5 节的 PP=2 用的是 `/models/DeepSeek-V4-Flash-DSpark`。两组读数不能当成同一个权重的两种并行方式。

| prompt tokens | TTFT (s) | prefill (t/s) | decode (t/s) |
|---|---|---|---|
| 7,963 | 6.73 | 1,184 | 18.6 |
| 15,999 | 9.80 | 1,632 | 18.8 |
| 31,989 | 9.48 | 3,373 | 17.9 |
| 63,969 | 16.72 | 3,827 | 20.1 |
| 127,929 | 35.33 | 3,621 | 19.4 |
| 199,925 | 44.57 | 4,486 | 19.4 |
| 319,973 | **87.51** | 3,656 | 19.2 |
| 511,935 | 173.18 | 2,956 | 18.3 |

结论：TTFT 随长度近似线性；**320k 实测首字 87.5 s，300k 按 200k→320k 斜率外推约 80 s**；512k 能跑但 prefill 相对 320k 掉 19% ⇒ **实用上限按 320k**。decode 全程 18–20 t/s 平，长上下文不加长解码开销。

---

## 7. 双栈对照（非 vLLM 路径，供复现环境选型参考）

| 栈 | 起服 | 读数 | 边界 |
|---|---|---|---|
| llama.cpp | `$BASE/llama.cpp/build-sm121/bin`（须 `LD_LIBRARY_PATH` 指过去），Qwen3.6-27B Q4_K_M 全层 offload（15.65 GiB / ngl 99） | **pp512 831.78 ± 12.07 t/s**、**tg128 12.46 ± 0.05 t/s**；设备识别 `NVIDIA GB10 cc 12.1 VRAM 124546 MiB` | 只能跑 GGUF；日志 `$BASE/p4/logs/d8b_llamacpp_bench.log`（12:08:54） |
| SGLang | 镜像 `sglang-sm12x:f832b5c39559`，Qwen3.6-35B-A3B-BF16，`--tp 1 --mem-fraction-static 0.80 --port 30000` | **13:39 复测（落盘）**：prefill **4,468 / 4,597 t/s**（r2/r3）、decode **31.5 / 31.7 t/s**、TTFT **0.088 / 0.086 s**；r1 冷启 971.8 t/s / 0.405 s（含 graph 捕获后首请求，弃用）。394-token 提示 / 128 输出 | 起服已把 warning 原文落盘：`Using default MoE kernel config. Performance might be sub-optimal! Config file not found at …/triton_3_8_0/E=256,N=512,device_name=NVIDIA_GB10.json`（缺 GB10 triton config）；`max_total_num_tokens=659,775`、`context_len=262,144`、`available_gpu_mem=20.53 GB`；decode CUDA graph bs=[1,2,4,8,12,16,24,32,37]。**跑不了 DeepSeek-V4**（能力判断，无日志）。旧控制台读数 4,534–4,648 / 31.9 保留为历史，不作为引用值 |

---

## 8. NVFP4 8-Shape（单卡，116 / RTX PRO 4500 Blackwell 32 GB）

构建：DeepGEMM **2.8.0** 源码自建（需补 `third-party/deep_jit` 与 `third-party/cutlass`），`CUDA_HOME=/usr/local/cuda-13.4`，python 3.14.4 + **torch 2.14.0+cu130**（环境 `$BASE/envs/sm120-t211`）；脚本 `$BASE/dgtest/dg_8shape_nvfp4.py`。

**必须在该目录下跑**：`deep_gemm` 没进 site-packages，只能从源码树导入 ⇒ 先 `cd $BASE/src/DeepGEMM-vllm`，再 `$BASE/envs/sm120-t211/bin/python $BASE/dgtest/dg_8shape_nvfp4.py`（或显式 `PYTHONPATH=$HOME/src/DeepGEMM-vllm`）。**跨目录 import 会失败**，这是最容易误判成“环境坏了”的一件事。

**环境可复现性（重要）**：当前 116 上三个环境实测为 `sm120-t211 = 2.14.0+cu130`（`import deep_gemm` **成功**，8 格就是它跑的）、`sm120-cu130 = 2.9.0+cu130`、`sm120 = 2.9.1+cu128`。后两个 import 现存 `_C.cpython-314-x86_64-linux-gnu.so` 会报 `undefined symbol: _ZN3c104cuda19CUDAErrorLogCaptureC1Ev`（ABI 不匹配）。**换环境即不可复现，必须用 t211**；且 `2.13.0+cu130`（`~/vllm-venv`）、`2.9.0+cu130`（`sm120-cu130`）都是同一个 ABI 报错 ⇒ 版本号不是唯一变量，`.so` 的构建 ABI 也要对上。

| M × N × K | bf16 (ms / TFLOPS / rel) | nvfp4 (ms / TFLOPS / rel) | 加速比 |
|---|---|---|---|
| 8192 × 4096 × 4096 | 1.814 / 151.5 / 0.28% | 0.431 / **638.2** / 16.68% | **4.212×** |
| 4096 × 4096 × 4096 | 0.901 / 152.6 / 0.28% | 0.218 / **630.5** / 16.19% | **4.132×** |
| 4096 × 8192 × 4096 | 1.691 / 162.6 / 0.26% | 0.468 / **586.9** / 16.55% | **3.610×** |
| 8192 × 2048 × 4096 | 1.022 / 134.4 / 0.29% | 0.238 / **576.3** / 16.72% | **4.286×** |
| 4 × 4096 × 4096 | 0.018 / 7.4 | 0.019 / 7.2 / 13.88% | 0.97× |
| 2 × 4096 × 4096 | 0.018 / 3.7 | 0.019 / 3.6 / 13.88% | 0.96× |
| 1 × 4096 × 4096 | 0.015 / 2.2 | 0.019 / 1.8 / 16.17% | 0.82× |
| 1 × 8192 × 2048 | 0.043 / 0.8 | 0.013 / **2.5** / 15.15% | **3.28×** |

读法：

- 大形状（M ≥ 4096）**3.6–4.3×**；误差 13.9–16.7%（FP4 元素级 E2M1 + per-32 block scale 量化误差，8 格无 NAN、无 SUSPECT）。
- 小形状多数与 bf16 打平（0.82–0.97×，launch-bound），**但 M1×N8192×K2048 是 3.28× 例外** —— 不能写成「小形状一律无优势」。
- **FP8 列 8 格全失败**：`Assertion error (csrc/apis/../heuristics/../../utils/layout.hpp:92): Unknown recipe`，`per_token_cast_to_fp8` / `per_block_cast_to_fp8` / packed ue8m0 / plain 四种组合皆过不去 ⇒ 与 FP8 表并列这一半未完成，需换 `fp8_gemm_nt` 或换 build。

---

## 9. 覆盖边界（发布必须一字不改地带上）

**打不通 / 不可用**

- **anemll 0.1.1 镜像 PP=2**（原文仍无落盘，且**已无法重跑补证**：两机宿主 `$MODEL_DIR/dsv4-abliterated` 都不存在、镜像内也无 `/models`，权重来源已断）：两轮同一错，102 worker `sparse_attn_compress_cutedsl.py::fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl` 报 `ValueError: Invalid state_cache.strides[0] on argument #0 ... expected to be divisible by 16`；旁证每 stage 仍占 74.86 GiB（≈整模）⇒ 该镜像实现不是 PP-aware。⚠ 失败**原文**与 74.86 GiB **只在控制台出现过**，`$BASE/p4`、`/tmp` grep 0 命中，`deploy_pp2.log` 只留 `DEPLOY_FAIL pp2 elapsed=829s`。
- **跨机 TP=2 的方向**：anemll 0.1.1 + `mp` **能起且能跑**（第 6 节即在其上，`tp2s.out` 有 HEALTH_OK）；此前「vLLM 不支持跨节点 TP」的结论只对 **PyPI 公版 0.24 + `mp`** 成立（7-09 内部记录，无落盘）。**NGC 26.08 上的 TP=2 今天没测**，不要写成「NGC 也不行」或「vLLM 都不行」。
- **PyPI 公版 vLLM 0.24**：DeepGEMM 与 NVFP4 modelopt 的 `scale_fmt=ue8m0` 不兼容 —— **内部记录（7-09），无落盘，引用需带此标注**。
- **SGLang 跑 DeepSeek-V4**：不支持 —— **内部记录，无落盘**（该组容器已删）。

**未验证 / 不承诺**

- 只保证 **101/102 同型号 GB10 + 同镜像 + 同 entrypoint 版本**；换机、换驱动、换镜像版本不在覆盖内。
- `max-model-len` 只验证到 8192（PP=2）；更长上下文的 PP=2 未测（八档那组是 TP=2 数据）。
- 文本层一致性口径（用户订正版，引用必须用这句）：**首 token 20/20 相同；16-token 文本 13/20 相同、7/20 在 16 token 内分叉（含数值不同的答案）⇒ 跨后端差可以落到输出层，只是尚未扩到全量题集**。不得写「不落到输出 / 不影响输出」。

---

## 10. 复现清单（文件级）

| 文件 | 位置 | 用途 |
|---|---|---|
| `pp2_ngc_entry.sh` | 101 `$BASE/p4/` + 102 `$BASE/p4/`（同版本） | 起服入口，`NODE_RANK` 区分 |
| `pp2_e2e.py` / `pp2_conc.py` | 101 `$BASE/p4/` | 单流 / 4 并发探针 |
| `d8a_longctx.py` / `d8a_longctx.json` / `logs/d8a.out` | 101 `$BASE/p4/` | 八档原始读数（12:07:24） |
| `logs/d8b_llamacpp_bench.log` | 101 `$BASE/p4/logs/` | llama.cpp pp512/tg128 |
| `logs/deploy_pp2.log` | 101 `$BASE/p4/logs/` | anemll PP=2 失败收据（11:53:53） |
| `gen/head_tp2s.sh` / `gen/worker_tp2s.sh` | 101 `$BASE/p4/gen/`（102 同套） | **跨机 TP=2** 起服脚本（anemll 0.1.1 + `mp`，`:8888`，`--nnodes 2`） |
| `logs/tp2s.out` | 101 `$BASE/p4/logs/` | TP=2 起服收据：`HEALTH_OK elapsed=266s 11:58:31 / DEPLOY_OK tp2s` |
| `logs/pp_worker_tp2s.log` | 101 `$BASE/p4/logs/` | TP=2 worker（`Available KV cache memory: 18.1 GiB` 出处） |
| `logs/sglang_20261122.log` / `logs/sglang_smoke_20261122.log` | 101 `$BASE/p4/logs/` | SGLang 起服 48 KB 全文 + 3 次探针（13:39）；副本不进包 |
| `dg_8shape_nvfp4.py` | 116 `$BASE/dgtest/` | NVFP4 8 格（12:34:31）；**跑前先 `cd $BASE/src/DeepGEMM-vllm`**（`deep_gemm` 只在源码树里） |
| `$BASE/src/DeepGEMM-vllm` | 116 | 2.8.0 自建源码与 `_C.cpython-314-x86_64-linux-gnu.so` |

**操作纪律（本 v0.1 起生效）**：起服一律 `... 2>&1 | tee $BASE/p4/logs/<tag>.log`；测速脚本 `python3 -u ... | tee`。本次 v0.1 里有 3 处「只有控制台」的数字（anemll 原文 / KV 容量行 / SGLang 整组），都是没 tee 造成的，不要在下一版重犯。

---

## 11. 已知坑（按踩到的顺序）

1. **`$BASE/p4` 变 root 所有**：`-v` 源文件不存在时 docker 会把挂载建成目录 ⇒ 容器内写日志失败。修：`sudo chown -R $USER:$USER $BASE/p4`。
2. **残留容器占显存**：101 上旧的 `sglang_t` 未拆时，起服报 `Free memory on device cuda:0 (12.69/121.63 GiB) on startup is less than desired GPU memory utilization (0.82, 99.73 GiB)`。修：先 `docker rm -f` 清干净。
3. **起服顺序**：head 先、worker 后 20 s；反了报 `Engine core initialization failed. Failed core proc(s): {}`。
4. **`pkill -f <脚本名>` 会自杀**：模式串在自己 cmdline 里，用 `pkill -x` 或按 PID。
5. **`serve` 名与探针不一致**：探针里写死 `model="m"` 会 404，必须从 `/v1/models` 取 `served-model-name`。
6. **`--ipc=host` 必须带**：不带时 `/dev/shm` 默认 64 MiB，PP=2 起服必死在 `Insufficient space in /dev/shm: 160 MiB required`（12:26 复测即此坑）。
7. **`--gpus all` 必须带**：`docker inspect` 里 `DeviceRequests=null`、`Runtime=runc` 时容器看不到卡，起服报 `Failed to infer device type`。
8. **首请求不是稳态**：同参数冷启首请求 4.4 t/s，第二请求起 14.7 → 单流读数一律丢掉第一发。

---

## 12. 发布前必须处理的两处边界（v1.0.9 已处理）

- `README:181` 原写「GGUF / llama.cpp 完全没进本包」——本配方第 7 节引了 llama.cpp 数字，已改为「不发脚本与权重，仅记录对照读数」。**v1.0.9 已改。**
- `docs/EVIDENCE_INDEX.md:42` 原只把 SGLang 当 N1 §6 的对比项——第 7 节随包发布后，已在此登记为 N2 证据。**v1.0.9 已改（新增 §5，9 行）。**

---

## 13. 变更记录

- **修订 r2（2026-11-22，审计回执 §十六 后；产品版本仍记 v0.1）**：修正 ①「跨机 TP=2 不可用」为「anemll 0.1.1 + `mp` 可通，八档即在其上；NGC 未测 TP=2」② 驱动版本按实机改正为 **595.71.05**（原值错） ③ 补 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` / `HF_HUB_OFFLINE=1` ④ §5 三次读数标「未落盘」⑤ §7 SGLang prefill 起点改 4,534、把「日志明示」改为「控制台、无落盘」⑥ §8 补环境可复现性（t211 才能 import）⑦ §10 补 TP=2 脚本与收据、补宿主机权重路径与 `/models` 陷阱 ⑧ PyPI 0.24 / SGLang 跑 V4 标为内部记录 ⑨ 补「先 `cd $BASE/src/DeepGEMM-vllm` 才能 import deep_gemm」与 2.13.0/2.9.0 同 ABI 报错说明 ⑩ 补 §14 附录（三份起服脚本原文 + 实测可用 run 参数）⑫ 补 §7 SGLang 13:39 落盘复测（prefill 4,468/4,597、decode 31.5/31.7、warning 原文、max_total_num_tokens 659,775）⑬ 记「anemll 原文已无法重跑补证」（两机权重 dsv4-abliterated 均不存在）⑪ 补 §5.1 13:31 复测（单流 14.71/14.72 落盘；4 并发 25.5/43.2 波动，点值作废改区间）。
- **v0.1（2026-11-22）**：首版。PP=2 NGC 路线打通并实测（14.8 / 32.6 t/s、KV 249,232）；补八档、双栈、NVFP4 8 格；写入覆盖边界与三处无落盘证据的标注；撤回 KV 容量行；修正加速比 3.6–4.3× 与小形状 3.28× 例外。
- **修订 r3（2026-11-22，发布版）**：§0 数字对齐 §5.1（14.71/14.72、25–43、243,733）；§9/§12 边界改为已处理口径；新增英文版 `docs/N2_dual_node_recipe_v0.1.en.md`。


---

## 14. 附录：起服脚本原文（逐字抄自节点，勿手改）

三个文件已 scp 到个人工作区 `n2_appendix/`（1482 / 3692 / 3544 B），与节点上 `$BASE/p4/pp2_ngc_entry.sh`、`$BASE/p4/gen/head_tp2s.sh`、`$BASE/p4/gen/worker_tp2s.sh` 逐字节一致；附录里的脚本正文即本包随附的完整版本。

### 14.1 双容器起法（PP=2，NGC 26.08）—— 13:31 实测可用的 run 参数

```
# 101（head）
docker run -d --name pp2ngc_head --network host --gpus all --ipc=host \
  -e NODE_RANK=0 \
  -v $MODEL_DIR:/models \
  -v $BASE/p4/pp2_ngc_entry.sh:/entry.sh \
  --entrypoint /bin/bash nvcr.io/nvidia/vllm:26.08-py3 /entry.sh

# 102（worker，同镜像同挂载，仅 rank 不同）
docker run -d --name pp2ngc_worker --network host --gpus all --ipc=host \
  -e NODE_RANK=1 \
  -v $MODEL_DIR:/models \
  -v $BASE/p4/pp2_ngc_entry.sh:/entry.sh \
  --entrypoint /bin/bash nvcr.io/nvidia/vllm:26.08-py3 /entry.sh
```

`--gpus all` 与 `--ipc=host` 不是可选：缺前者报 `RuntimeError: Failed to infer device type`（容器内看不到 GPU），缺后者报 `RuntimeError: Insufficient space in /dev/shm: 160 MiB required, 64 MiB free`（12:26 那次就是这么死的，日志 `pp2ngc_head_20261122_1310.log`）。原容器（12:28 那对）实测：`Binds = ["$MODEL_DIR:/models","$BASE/p4/pp2_ngc_entry.sh:/entry.sh"]`、`Entrypoint = ["/bin/bash"]`、`Cmd = ["/entry.sh"]`、`NetworkMode = host`。

**带落盘启动（v0.1 起的纪律）**：把 `Cmd` 换成 `bash -c "/entry.sh 2>&1 | tee /p4/logs/pp2ngc_<rank>_<tag>.log"`，并补挂 `-v $BASE/p4:/p4`，控制台与文件同时留痕。

### 14.2 `pp2_ngc_entry.sh`（PP=2，两机同一份，`NODE_RANK` 区分）

```bash
#!/bin/bash
# NGC + Ray 双机 PP=2 试跑（DeepSeek-V4-Flash-DSpark, fp8 权重 + fp8 KV, enforce-eager）
set -u
RANK="${NODE_RANK:-0}"
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export RAY_DISABLE_IPV6=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export HF_HUB_OFFLINE=1

pip install -q ray==2.55.1 -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null
python -c "import ray;print('ray',ray.__version__)"


if [ "$RANK" = "0" ]; then
  ray stop -f >/dev/null 2>&1; sleep 2
  ray start --head --port=6379 --num-gpus=1 --node-ip-address=$HEAD_IP --disable-usage-stats
  for i in $(seq 1 72); do
    if ray status 2>&1 | grep -qE "2\.0 GPU|2 GPU"; then echo "RAY_2GPU_READY"; break; fi
    sleep 5
  done
  ray status
  if ! ray status 2>&1 | grep -qE "2\.0 GPU|2 GPU"; then echo "RAY_ONLY_1GPU_ABORT"; sleep infinity; fi
  exec vllm serve /models/DeepSeek-V4-Flash-DSpark \
    --served-model-name dsv4-pp2 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray \
    --enforce-eager \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.82 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000
else
  ray stop -f >/dev/null 2>&1; sleep 2
  ray start --address=$HEAD_IP:6379 --num-gpus=1 --node-ip-address=$WORKER_IP --disable-usage-stats
  ray status
  sleep infinity
fi
```

### 14.3 `head_tp2s.sh`（跨机 TP=2，anemll 0.1.1，`:8888`）

```bash
#!/bin/bash
# vllm head — Anemll 0.1.1 完全 1:1 复刻 (从 anemll/dspark-vllm-gx10 docker-compose.yml)
set -e

# === NCCL / IP 基础 ===
export VLLM_HOST_IP=$HEAD_IP
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=roceP2p1s0f0,roceP2p1s0f1
export NCCL_IB_DISABLE=0
export NCCL_IB_ADDR_FAMILY=AF_INET
export NCCL_IB_ROCE_VERSION_NUM=2
export NCCL_CROSS_NIC=1
export NCCL_CUMEM_ENABLE=0
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export TP_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0

# === anemll DSpark 专属 17 个 env ===
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache
export DSPARK_SLOT_CLAMP=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_TRITON_MLA_SPARSE=1
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export VLLM_SKIP_INIT_MEMORY_CHECK=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_USE_B12X_MOE=1
export VLLM_USE_B12X_WO_PROJECTION=1
export VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM=0
export VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M=16
export B12X_W4A16_TC_DECODE=0
export VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.0
export VLLM_DSPARK_CONFIDENCE_SCHEDULER=off
export VLLM_DSPARK_LOCAL_ARGMAX=1
export VLLM_DSPARK_REPLICATE_MARKOV_W1=1
export VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0
export VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1
export VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT=0
export VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP=1
export VLLM_DSV4_B12X_COMPRESSED_MLA=0
export VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=0
export VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT=0

# === FlashInfer / DeepGEMM ===
export TORCH_CUDA_ARCH_LIST=12.1a
export FLASHINFER_CUDA_ARCH_LIST=12.1a
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TILELANG_CLEANUP_TEMP_FILES=1
export DG_JIT_USE_NVRTC=0
export DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === Models / 缓存路径 ===
export HF_HOME=/cache/huggingface

# === Node rank / master ===
export NODE_RANK=0
export MASTER_ADDR=$HEAD_IP
export MASTER_PORT=25000
export MTP_NUM_TOKENS=5

# === JIT monitor ===
export JIT_MONITOR_MODE=warn

cd /vllm-workspace

# === vllm serve args (from anemll docker-compose.yml command section)===
SPECULATIVE_CONFIG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
MAX_NUM_SEQS=4

exec vllm serve /models/dsv4-abliterated \
    --served-model-name deepseek-v4-flash \
    --host 0.0.0.0 --port 8888 \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 1 \
    --kv-cache-dtype nvfp4_ds_mla \
    --block-size 256 \
    --max-model-len 1048576 \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --max-num-batched-tokens 8192 \
    --max-cudagraph-capture-size $(( ${MAX_NUM_SEQS} * (${MTP_NUM_TOKENS} + 1) )) \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --async-scheduling \
    --enable-chunked-prefill \
    --speculative-config "${SPECULATIVE_CONFIG}" \
    --tokenizer-mode deepseek_v4 \
    --distributed-executor-backend mp \
    --moe-backend flashinfer_b12x \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<","reasoning_end_str":">"}' \
    --default-chat-template-kwargs '{"thinking":false}' \
    --generation-config vllm \
    --enable-flashinfer-autotune \
    --nnodes 2 \
    --node-rank ${NODE_RANK} \
    --master-addr ${MASTER_ADDR} \
    --master-port ${MASTER_PORT} \
    --jit-monitor-mode ${JIT_MONITOR_MODE}
```

### 14.4 `worker_tp2s.sh`（同上，`NODE_RANK=1` + `--headless`）

```bash
#!/bin/bash
# vllm worker — Anemll 0.1.1 完全 1:1 复刻 headless
set -e

# === NCCL ===
export VLLM_HOST_IP=$WORKER_IP
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=roceP2p1s0f0,roceP2p1s0f1
export NCCL_IB_DISABLE=0
export NCCL_IB_ADDR_FAMILY=AF_INET
export NCCL_IB_ROCE_VERSION_NUM=2
export NCCL_CROSS_NIC=1
export NCCL_CUMEM_ENABLE=0
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0
export TP_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0

# === anemll DSpark 17 env ===
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache
export DSPARK_SLOT_CLAMP=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_TRITON_MLA_SPARSE=1
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
export VLLM_SKIP_INIT_MEMORY_CHECK=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_USE_B12X_MOE=1
export VLLM_USE_B12X_WO_PROJECTION=1
export VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM=0
export VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M=16
export B12X_W4A16_TC_DECODE=0
export VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.0
export VLLM_DSPARK_CONFIDENCE_SCHEDULER=off
export VLLM_DSPARK_LOCAL_ARGMAX=1
export VLLM_DSPARK_REPLICATE_MARKOV_W1=1
export VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0
export VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1
export VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT=0
export VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP=1
export VLLM_DSV4_B12X_COMPRESSED_MLA=0
export VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE=0
export VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT=0

# === FlashInfer / DeepGEMM ===
export TORCH_CUDA_ARCH_LIST=12.1a
export FLASHINFER_CUDA_ARCH_LIST=12.1a
export FLASHINFER_DISABLE_VERSION_CHECK=1
export TILELANG_CLEANUP_TEMP_FILES=1
export DG_JIT_USE_NVRTC=0
export DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === Models ===
export HF_HOME=/cache/huggingface

# === Node rank / master ===
export NODE_RANK=1
export HEADLESS=1
export MASTER_ADDR=$HEAD_IP
export MASTER_PORT=25000
export MTP_NUM_TOKENS=5

# === JIT monitor ===
export JIT_MONITOR_MODE=warn

cd /vllm-workspace

SPECULATIVE_CONFIG='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
MAX_NUM_SEQS=4

exec vllm serve /models/dsv4-abliterated \
    --served-model-name deepseek-v4-flash \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 1 \
    --kv-cache-dtype nvfp4_ds_mla \
    --block-size 256 \
    --max-model-len 1048576 \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --max-num-batched-tokens 8192 \
    --max-cudagraph-capture-size $(( ${MAX_NUM_SEQS} * (${MTP_NUM_TOKENS} + 1) )) \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --async-scheduling \
    --enable-chunked-prefill \
    --speculative-config "${SPECULATIVE_CONFIG}" \
    --tokenizer-mode deepseek_v4 \
    --distributed-executor-backend mp \
    --moe-backend flashinfer_b12x \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<","reasoning_end_str":">"}' \
    --default-chat-template-kwargs '{"thinking":false}' \
    --generation-config vllm \
    --enable-flashinfer-autotune \
    --nnodes 2 \
    --node-rank ${NODE_RANK} \
    --master-addr ${MASTER_ADDR} \
    --master-port ${MASTER_PORT} \
    --jit-monitor-mode ${JIT_MONITOR_MODE} \
    --headless
```