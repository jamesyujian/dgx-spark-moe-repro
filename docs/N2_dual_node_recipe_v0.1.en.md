# GB10 Dual-Node MoE Recipe v0.1 (N2)

> **Sanitization note for this kit version**: home directories are written `$BASE` (e.g. `p4/`, `envs/`), the weights directory is `$MODEL_DIR`, and the private dual-node IPs are `$HEAD_IP` (101) / `$WORKER_IP` (102). Machine numbers 101/102/115/116 are kept by agreement. The scripts in the appendix were substituted the same way — set `VLLM_HOST_IP` and `MASTER_ADDR` back to real values before running. The unsanitized original stays in the personal workspace and is not shipped.
> Version v0.1 | 2026-11-22 | Hardware: GB10 (SM121, aarch64) × 2 (node101 / node102) | Model: DeepSeek-V4-Flash-DSpark (MoE, fp4 expert / fp8 KV)
> This document ships **the reproduction chain + baseline numbers + coverage boundaries** only. No recommended configurations, no tuning advice. Every number states whether it is backed by a log that can be recomputed.

---

## 0. One-paragraph conclusion

**Both cross-node parallel modes work, but on two different images**: PP=2 on **NGC 26.08**, cross-node TP=2 on **anemll 0.1.1** (details below). Container `nvcr.io/nvidia/vllm:26.08-py3` (vLLM 0.27.1) + `ray` inside the container + `--enforce-eager` + fp8 KV: head startup to health 200 takes about **4 min**; single stream **14.71 / 14.72 t/s** (logged), 4-way concurrency **only as a range, 25–43 t/s**, KV **243,733 tokens**.
**Cross-node TP=2 also works**: `$BASE/p4/gen/head_tp2s.sh` (`--tensor-parallel-size 2 --nnodes 2 --node-rank … --distributed-executor-backend mp`) serves successfully on the **anemll 0.1.1** image; `logs/tp2s.out` records `SERVE_RESULT=HEALTH_OK elapsed=266s 11:58:31 / DEPLOY_OK tp2s`, and the long-context sweep in §6 ran on exactly that TP=2 server (`:8888`). **Note the mirror-image result**: on anemll, PP=2 does not work (the cutedsl path is not PP-aware, hard failure at `stride 16`); TP=2 on NGC was **not tested**. The earlier claim "vLLM does not support cross-node TP" holds only for **PyPI 0.24 + `mp`** (internal note from 07-09, no log on disk) and must not be generalised to NGC.

---

## 1. Hardware / driver / baseline

| Item | Value |
|---|---|
| Nodes | 2 × GB10 (NVIDIA GB10, compute capability **12.1**, aarch64), 124,546 MiB VRAM per card |
| Driver | **595.71.05** (measured identical on 101/102; `nvidia-smi` passes straight through into the container; if an upstream document quotes any other driver number, this row wins) |
| Interconnect | Private dual-node network `$HEAD_IP` / `$WORKER_IP` (bond0, slave `enP2p1s0f0np0`, active-backup, cross-node ping 0.6–0.9 ms) |
| Model (used by PP=2) | `/models/DeepSeek-V4-Flash-DSpark` inside the container ⇒ host path `$MODEL_DIR/DeepSeek-V4-Flash-DSpark`; MoE, `expert_dtype` resolves to `fp4`, KV cache format `fp8_ds_mla` |

> Path caveat: `/models` on the host is a different root-level directory (not the container's `/models`), while `$MODEL_DIR/` is where the weights actually live ⇒ running `vllm serve /models/DeepSeek-V4-Flash-DSpark` directly on the host will not find the model; mount it with `-v $MODEL_DIR:/models`.
>
> Caveat: the `Primary Slave` of bond0 is runtime state and **is lost on reboot**; this recipe does not care which slave is active, but both nodes must reach each other on the same subnet.

---

## 2. Component list

| Component | Version / source | Notes |
|---|---|---|
| Image | `nvcr.io/nvidia/vllm:26.08-py3` | vLLM **0.27.1** inside; **do not use PyPI 0.24** (DeepGEMM is incompatible with the `scale_fmt=ue8m0` NVFP4 modelopt path, see the 07-09 note) |
| Ray | inside the container: `pip install ray==2.55.1 -i https://pypi.tuna.tsinghua.edu.cn/simple` | the image does not ship ray; `--distributed-executor-backend ray` hard-requires it |
| Entry script | `$BASE/p4/pp2_ngc_entry.sh` (**same file on both nodes**, `NODE_RANK=0/1` distinguishes them) | contains a "wait for 2.0 GPUs" guard, up to 72 × 5 s; on timeout it prints `RAY_ONLY_1GPU_ABORT` and `sleep infinity` (it never fails silently) |

---

## 3. Serve arguments (identical on both nodes, only `NODE_RANK` differs)

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

Environment variables (identical on both nodes):

```
GLOO_SOCKET_IFNAME=bond0
NCCL_SOCKET_IFNAME=bond0
NCCL_IB_DISABLE=1
NCCL_NET=Socket
RAY_DISABLE_IPV6=1
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
HF_HUB_OFFLINE=1
```

The last two are not decoration: the first lifts `--max-model-len` above the model's default ceiling (the measured entrypoint carries it), the second keeps an offline node from reaching for the HF Hub (without it, tokenizer/config fetch stalls on a machine with no internet).

Three arguments are not optional:

- `--enforce-eager`: skips cross-node CUDA-graph compile synchronisation (verified in the 07-09 and 122B records; without it startup hangs in that sync).
- `--distributed-executor-backend ray`: cross-node PP=2 can only go through Ray.
- `--quantization fp8` + `--kv-cache-dtype fp8`: keeps the DeepGEMM path inside the container; switching to bf16 walks an untested branch.

---

## 4. Startup procedure (ordering is a hard constraint)

1. **Start the head on node101** (`NODE_RANK=0`) and wait for it to print the Ray head address.
2. **About 20 s later, start the worker on node102** (`NODE_RANK=1`).
3. Poll `curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:8000/health`; 200 means it is up.

Reverse the order and you get `RuntimeError: Engine core initialization failed. Failed core proc(s): {}` (GCS is connected, the engine is not; the log does not reveal the real cause). **Do not retry the worker in place — stop the head and run 1→2 again.**

Measured timeline (node-side timestamps):

| Event | Time |
|---|---|
| head container started | **12:28:11** |
| health 200 | **12:32:32** (about 4 min of startup, of which the 990-kernel DeepGEMM warmup takes 55 s) |
| both containers removed | head `exit 0` 12:58:42 / worker `exit 137` 12:59:02 |

**Cross-node TP=2 startup (the other route, anemll 0.1.1 + `mp`, `:8888`)**: the order is reversed versus PP=2 — **worker first** (`worker_tp2s.sh` 11:53:59), then head (`head_tp2s.sh` 11:54:04), giving `SERVE_RESULT=HEALTH_OK elapsed=266s 11:58:31 / DEPLOY_OK tp2s` (`$BASE/p4/logs/tp2s.out`). About 4.4 min to serve. Note that the leading `confirm: tp=1 pp=1` line in `tp2s.out` / `pp2_run.out` is a fixed echo from the wrapper, **does not match the real arguments in `gen/*.sh`, and must not be used as parameter evidence**.

---

## 5. Baseline numbers (PP=2, `max-model-len 8192` / `max-num-seqs 4`)

> ⚠️ **Superseded marker: this table holds the readings from the first successful bring-up at 12:32 and has been superseded by §5.1 (the 13:31 re-measurement with `tee` logging).** It is kept only to record "the moment it first worked"; **any external citation must use §5.1**. The 14.72/14.79/14.82 and the 32.6 in this table have no log behind them.

| Metric | Value | Notes |
|---|---|---|
| Single-stream decode | **14.8 t/s** (three client-side readings: 14.72 / 14.79 / 14.82) | ⚠ **not logged**: the three readings existed on the console only; the machine retains just the server-side rolling lines 14.7 / 14.8 (`docker logs pp2ngc_head`, container removed at 12:58) |
| 4-way aggregate | **32.6 t/s** (1022 tokens / 31.3 s) | equals **8.16 t/s** per stream, i.e. each stream runs about 45% below single-stream under concurrency |
| KV cache capacity | **249,232 tokens**, @8k = **30.42×** concurrency | recomputable (engine startup log) |
| Available KV per stage | **23.33 GiB (PP0) / 25.72 GiB (PP1)** | same log |
| KV figures once mis-quoted | ~~19.65 GiB / 2,688,671 tokens / 2.56×~~ | **retracted**: that line came from a console of a container that has since been deleted and there is no log on disk; the same bring-up recorded 18.1 GiB in `$BASE/p4/logs/pp_worker_tp2s.log:5` |

### 5.1 13:31 re-measurement (with `tee`, every reading on disk)

Same arguments, restarted once (13:24:25 start → 13:28:55 `HTTP server started`, **270 s**); all probes and readings are logged:

| Metric | Value this run (logged) | File |
|---|---|---|
| Single stream r2 / r3 | **14.71 / 14.72 t/s** (120 tokens / 8.16 s / 8.15 s) | `$BASE/p4/logs/pp2_e2e_r2/r3_20261122.log` |
| Single stream r1 (first request after cold start) | 4.4 t/s (27.3 s, includes first-request warmup) | `pp2_e2e_r1_20261122.log` |
| 4-way r1 / r2 | **25.5 t/s** (1147 tokens / 44.91 s)｜**43.2 t/s** (1052 tokens / 24.33 s) | `pp2_conc_20261122.log` / `pp2_conc_r2_20261122.log` |
| KV | **243,733 tokens**, @8k = **29.75×** | `pp2ngc_head_20261122_1326.log` |
| Available KV per stage | **24.32 GiB (PP0) / 25.01 GiB (PP1)** | same log |

Reading corrections (important):

- **Single stream 14.7 is reproducible** and agrees with the 12:32 set (14.72 / 14.79 / 14.82); the cold first request reaches only 4.4 t/s, so **any single-stream citation must drop the first request**.
- **4-way concurrency swings wildly**: two adjacent runs with identical arguments gave 25.5 and 43.2 (a 1.7× spread), with the 32.6 reading from 12:35 sitting in the middle ⇒ **this metric must not be published as a point value, only as the range 25–43 t/s, with the number of runs stated.** The original "32.6 / 8.16 per stream" stays as the 12:35 reading (unlogged).
- KV this run is 243,733 / 29.75× versus 249,232 / 30.42× at 12:32 — a 2% difference from memory headroom, not from a version change.
- The log files are written by root inside the container (`-rw-r--r-- root`), readable but not overwritable from the host; `sudo chown $USER:$USER` them before reusing.

> 14.8 t/s and "the long-context numbers measured under TP=2" are different operating points: the 18–20 t/s decode below belongs to TP=2 with spec on and cannot be compared directly with the PP=2 single-stream 14.8.

---

## 6. Long-context sweep (measured on the TP=2 server)

Operating point: **anemll 0.1.1 image**, `--tensor-parallel-size 2 --nnodes 2 --distributed-executor-backend mp`, `:8888`, `--kv-cache-dtype nvfp4_ds_mla --block-size 256`, dspark spec on (5 tokens), `--gpu-memory-utilization 0.85`, prefix caching + chunked prefill on, `--max-model-len 1048576`.
Source: `$BASE/p4/d8a_longctx.json` (written 12:07:24, cell-by-cell recomputable); the probe `$BASE/p4/d8a_longctx.py` has `BASE=127.0.0.1:8888`, i.e. exactly this TP=2 server.
⚠ **Different weights**: this server used `/models/dsv4-abliterated` (no directory of that name remains on the host), while the PP=2 numbers in §5 used `/models/DeepSeek-V4-Flash-DSpark`. The two sets must not be presented as the same weights under two parallel modes.

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

Conclusion: TTFT grows roughly linearly with length; **320k measured 87.5 s to first token, and 300k extrapolates to about 80 s along the 200k→320k slope**; 512k runs but prefill drops 19% relative to 320k ⇒ **the practical ceiling is 320k**. Decode stays flat at 18–20 t/s throughout, so long context does not lengthen decode.

---

## 7. Dual-stack comparison (non-vLLM paths, for choosing a reproduction environment)

| Stack | How it serves | Readings | Boundary |
|---|---|---|---|
| llama.cpp | `$BASE/llama.cpp/build-sm121/bin` (requires `LD_LIBRARY_PATH` pointing there), Qwen3.6-27B Q4_K_M fully offloaded (15.65 GiB / ngl 99) | **pp512 831.78 ± 12.07 t/s**, **tg128 12.46 ± 0.05 t/s**; device detected as `NVIDIA GB10 cc 12.1 VRAM 124546 MiB` | GGUF only; log `$BASE/p4/logs/d8b_llamacpp_bench.log` (12:08:54) |
| SGLang | image `sglang-sm12x:f832b5c39559`, Qwen3.6-35B-A3B-BF16, `--tp 1 --mem-fraction-static 0.80 --port 30000` | **13:39 re-measured (logged)**: prefill **4,468 / 4,597 t/s** (r2/r3), decode **31.5 / 31.7 t/s**, TTFT **0.088 / 0.086 s**; r1 cold start 971.8 t/s / 0.405 s (discarded, it is the first request after graph capture). 394-token prompt / 128 output tokens | Startup logged the warning verbatim: `Using default MoE kernel config. Performance might be sub-optimal! Config file not found at …/triton_3_8_0/E=256,N=512,device_name=NVIDIA_GB10.json` (no GB10 triton config); `max_total_num_tokens=659,775`, `context_len=262,144`, `available_gpu_mem=20.53 GB`; decode CUDA graph bs=[1,2,4,8,12,16,24,32,37]. **Cannot run DeepSeek-V4** (capability judgement, no log). The older console readings 4,534–4,648 / 31.9 are kept as history and are not citable |

---

## 8. NVFP4 8-shape (single card, 116 / RTX PRO 4500 Blackwell 32 GB)

Build: DeepGEMM **2.8.0** built from source (needs `third-party/deep_jit` and `third-party/cutlass`), `CUDA_HOME=/usr/local/cuda-13.4`, python 3.14.4 + **torch 2.14.0+cu130** (environment `$BASE/envs/sm120-t211`); script `$BASE/dgtest/dg_8shape_nvfp4.py`.

**You must run from that directory**: `deep_gemm` is not installed into site-packages and can only be imported from the source tree ⇒ first `cd $BASE/src/DeepGEMM-vllm`, then `$BASE/envs/sm120-t211/bin/python $BASE/dgtest/dg_8shape_nvfp4.py` (or set `PYTHONPATH=$BASE/src/DeepGEMM-vllm` explicitly). **Importing from another directory fails**, which is the single most likely reason to think "the environment is broken".

**Environment reproducibility (important)**: the three environments currently on 116 measure `sm120-t211 = 2.14.0+cu130` (`import deep_gemm` **succeeds**; this is the one that produced the 8 shapes), `sm120-cu130 = 2.9.0+cu130`, `sm120 = 2.9.1+cu128`. The latter two raise `undefined symbol: _ZN3c104cuda19CUDAErrorLogCaptureC1Ev` when importing the existing `_C.cpython-314-x86_64-linux-gnu.so` (ABI mismatch). **Changing environment makes the numbers unreproducible — you must use t211**; and both `2.13.0+cu130` (`~/vllm-venv`) and `2.9.0+cu130` (`sm120-cu130`) fail with the same ABI error ⇒ the version number is not the only variable, the build ABI of the `.so` must match too.

| M × N × K | bf16 (ms / TFLOPS / rel) | nvfp4 (ms / TFLOPS / rel) | speed-up |
|---|---|---|---|
| 8192 × 4096 × 4096 | 1.814 / 151.5 / 0.28% | 0.431 / **638.2** / 16.68% | **4.212×** |
| 4096 × 4096 × 4096 | 0.901 / 152.6 / 0.28% | 0.218 / **630.5** / 16.19% | **4.132×** |
| 4096 × 8192 × 4096 | 1.691 / 162.6 / 0.26% | 0.468 / **586.9** / 16.55% | **3.610×** |
| 8192 × 2048 × 4096 | 1.022 / 134.4 / 0.29% | 0.238 / **576.3** / 16.72% | **4.286×** |
| 4 × 4096 × 4096 | 0.018 / 7.4 | 0.019 / 7.2 / 13.88% | 0.97× |
| 2 × 4096 × 4096 | 0.018 / 3.7 | 0.019 / 3.6 / 13.88% | 0.96× |
| 1 × 4096 × 4096 | 0.015 / 2.2 | 0.019 / 1.8 / 16.17% | 0.82× |
| 1 × 8192 × 2048 | 0.043 / 0.8 | 0.013 / **2.5** / 15.15% | **3.28×** |

How to read it:

- Large shapes (M ≥ 4096) reach **3.6–4.3×**; the relative error is 13.9–16.7% (FP4 element-wise E2M1 plus per-32 block-scale quantisation; no NAN and no SUSPECT in any of the 8 cells).
- Small shapes mostly tie with bf16 (0.82–0.97×, launch-bound), **but M1×N8192×K2048 is the 3.28× exception** — do not write "small shapes never win".
- **The FP8 column fails in all 8 cells**: `Assertion error (csrc/apis/../heuristics/../../utils/layout.hpp:92): Unknown recipe`; `per_token_cast_to_fp8`, `per_block_cast_to_fp8`, packed ue8m0 and plain all fail ⇒ the FP8 half of the comparison is unfinished and needs `fp8_gemm_nt` or a different build.

---

## 9. Coverage boundaries (must ship verbatim)

**Does not work / unavailable**

- **anemll 0.1.1 image, PP=2** (the failure text still has no log, and **it can no longer be re-run to obtain one**: neither node has `$MODEL_DIR/dsv4-abliterated`, and the image ships no `/models`, so the weights source is gone): two rounds failed identically; the worker on 102 raised `ValueError: Invalid state_cache.strides[0] on argument #0 ... expected to be divisible by 16` from `sparse_attn_compress_cutedsl.py::fused_kv_compress_norm_rope_insert_sparse_attn_cutedsl`; the side evidence is that each stage still holds 74.86 GiB (≈ the whole model) ⇒ that image's implementation is not PP-aware. ⚠ The **verbatim** failure text and the 74.86 GiB figure existed **on the console only**; grep over `$BASE/p4` and `/tmp` finds 0 hits, and `deploy_pp2.log` retains only `DEPLOY_FAIL pp2 elapsed=829s`.
- **Cross-node TP=2 direction**: anemll 0.1.1 + `mp` **serves and runs** (§6 ran on it, `tp2s.out` shows HEALTH_OK); the earlier conclusion "vLLM does not support cross-node TP" holds only for **PyPI 0.24 + `mp`** (07-09 internal note, no log). **TP=2 on NGC 26.08 was not tested today** — do not write "NGC can't do it either" or "vLLM can't do it at all".
- **PyPI vLLM 0.24**: DeepGEMM is incompatible with the NVFP4 modelopt `scale_fmt=ue8m0` path — **internal note (07-09), no log, cite it only with this caveat**.
- **SGLang running DeepSeek-V4**: unsupported — **internal note, no log** (that container has been deleted).

**Untested / not promised**

- Only **101/102 with the same GB10 model + same image + same entrypoint version** is guaranteed; other machines, drivers or image versions are outside coverage.
- `max-model-len` is verified only up to 8192 for PP=2; PP=2 beyond that is untested (the sweep in §6 is TP=2 data).
- Text-layer consistency, in the user-corrected wording that must be quoted as is: **first token identical in 20/20; 16-token text identical in 13/20, with 7/20 diverging inside 16 tokens (including answers whose numeric value differs) ⇒ cross-backend differences can reach the output layer, they simply have not been extended to the full question set yet**. Never write "does not reach the output" or "does not affect the output".

---

## 10. Reproduction checklist (file level)

| File | Location | Purpose |
|---|---|---|
| `pp2_ngc_entry.sh` | 101 `$BASE/p4/` + 102 `$BASE/p4/` (same version) | serve entrypoint, `NODE_RANK` distinguishes roles |
| `pp2_e2e.py` / `pp2_conc.py` | 101 `$BASE/p4/` | single-stream / 4-way probes |
| `d8a_longctx.py` / `d8a_longctx.json` / `logs/d8a.out` | 101 `$BASE/p4/` | raw sweep readings (12:07:24) |
| `logs/d8b_llamacpp_bench.log` | 101 `$BASE/p4/logs/` | llama.cpp pp512/tg128 |
| `logs/deploy_pp2.log` | 101 `$BASE/p4/logs/` | anemll PP=2 failure receipt (11:53:53) |
| `gen/head_tp2s.sh` / `gen/worker_tp2s.sh` | 101 `$BASE/p4/gen/` (same set on 102) | **cross-node TP=2** launchers (anemll 0.1.1 + `mp`, `:8888`, `--nnodes 2`) |
| `logs/tp2s.out` | 101 `$BASE/p4/logs/` | TP=2 receipt: `HEALTH_OK elapsed=266s 11:58:31 / DEPLOY_OK tp2s` |
| `logs/pp_worker_tp2s.log` | 101 `$BASE/p4/logs/` | TP=2 worker (the source of `Available KV cache memory: 18.1 GiB`) |
| `logs/sglang_20261122.log` / `logs/sglang_smoke_20261122.log` | 101 `$BASE/p4/logs/` | SGLang 48 KB startup log + three probes (13:39); copies are not shipped |
| `dg_8shape_nvfp4.py` | 116 `$BASE/dgtest/` | NVFP4 8 shapes (12:34:31); **`cd $BASE/src/DeepGEMM-vllm` first** (`deep_gemm` exists only in the source tree) |
| `$BASE/src/DeepGEMM-vllm` | 116 | 2.8.0 source build and `_C.cpython-314-x86_64-linux-gnu.so` |

**Operating discipline (in force from v0.1)**: always serve as `... 2>&1 | tee $BASE/p4/logs/<tag>.log`; always run benchmarks as `python3 -u ... | tee`. This v0.1 contains three console-only numbers (the anemll failure text / the KV capacity line / the whole SGLang group), all caused by missing `tee`; do not repeat it in the next revision.

---

## 11. Known pitfalls (in the order they were hit)

1. **`$BASE/p4` becomes root-owned**: when a `-v` source file does not exist, docker creates the mount as a directory ⇒ logging inside the container fails. Fix: `sudo chown -R $USER:$USER $BASE/p4`.
2. **A leftover container holds memory**: with the old `sglang_t` still running on 101, startup reports `Free memory on device cuda:0 (12.69/121.63 GiB) on startup is less than desired GPU memory utilization (0.82, 99.73 GiB)`. Fix: `docker rm -f` it first.
3. **Startup order**: head first, worker 20 s later; reversed gives `Engine core initialization failed. Failed core proc(s): {}`.
4. **`pkill -f <script>` kills itself**: the pattern is in its own cmdline; use `pkill -x` or kill by PID.
5. **Probe model name mismatch**: hard-coding `model="m"` in a probe returns 404; always read the id from `/v1/models`.
6. **`--ipc=host` is mandatory**: without it `/dev/shm` defaults to 64 MiB and PP=2 dies at `Insufficient space in /dev/shm: 160 MiB required` (this is exactly how the 12:26 retry died).
7. **`--gpus all` is mandatory**: with `DeviceRequests=null` and `Runtime=runc` in `docker inspect`, the container cannot see the GPU and startup reports `Failed to infer device type`.
8. **The first request is not steady state**: the cold first request reaches 4.4 t/s, the second and later reach 14.7 ⇒ always drop the first shot from single-stream readings.

---

## 12. Boundaries that had to be handled before publication (now done)

- `README:181` used to say "GGUF / llama.cpp is not covered by this kit at all" — §7 cites llama.cpp numbers, so the wording was rewritten to "no scripts or weights shipped, only the reference readings". **Done in v1.0.9.**
- `docs/EVIDENCE_INDEX.md:42` used to treat SGLang only as an N1 §6 comparison — §7 now ships as part of the kit, so it was registered as N2 evidence there. **Done in v1.0.9 (new §5, 9 rows).**

---

## 13. Change log

- **Revision r2 (2026-11-22, after audit receipt §16; product version stays v0.1)**: ① "cross-node TP=2 unavailable" corrected to "anemll 0.1.1 + `mp` works and the sweep ran on it; TP=2 on NGC untested" ② driver corrected to **595.71.05** per the machines ③ added `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` / `HF_HUB_OFFLINE=1` ④ §5 three readings marked "unlogged" ⑤ §7 SGLang prefill start corrected, "logged" wording replaced by "console, unlogged" ⑥ §8 added environment reproducibility (only t211 imports) ⑦ §10 added the TP=2 scripts and receipts, the host weights path and the `/models` trap ⑧ PyPI 0.24 and "SGLang cannot run V4" marked as internal notes ⑨ added "`cd $BASE/src/DeepGEMM-vllm` first" plus the shared ABI failure under 2.13.0/2.9.0 ⑩ added §14 appendix (three verbatim launch scripts + the run flags that actually worked) ⑪ added §5.1 (single stream 14.71/14.72 logged; 4-way 25.5/43.2 swings, point value withdrawn in favour of a range) ⑫ added §7 SGLang 13:39 logged re-measurement (prefill 4,468/4,597, decode 31.5/31.7, verbatim warning, `max_total_num_tokens` 659,775) ⑬ recorded "the anemll text can no longer be re-run for evidence" (no `dsv4-abliterated` weights on either node).
- **Revision r3 (2026-11-22, publication edit)**: §0 figures aligned with §5.1; §9/§12 boundaries updated; this English edition added.

---

## 14. Appendix: verbatim launch scripts (copied from the nodes, do not hand-edit)

The three files were copied to the personal workspace (`n2_appendix/`, 1482 / 3692 / 3544 B) and are byte-identical to `$BASE/p4/pp2_ngc_entry.sh`, `$BASE/p4/gen/head_tp2s.sh`, `$BASE/p4/gen/worker_tp2s.sh` on the nodes.

### 14.1 How the two containers are started (PP=2, NGC 26.08) — run flags that worked at 13:31

```
# 101 (head)
docker run -d --name pp2ngc_head --network host --gpus all --ipc=host \
  -e NODE_RANK=0 \
  -v $MODEL_DIR:/models \
  -v $BASE/p4/pp2_ngc_entry.sh:/entry.sh \
  --entrypoint /bin/bash nvcr.io/nvidia/vllm:26.08-py3 /entry.sh

# 102 (worker, same image and mounts, only the rank differs)
docker run -d --name pp2ngc_worker --network host --gpus all --ipc=host \
  -e NODE_RANK=1 \
  -v $MODEL_DIR:/models \
  -v $BASE/p4/pp2_ngc_entry.sh:/entry.sh \
  --entrypoint /bin/bash nvcr.io/nvidia/vllm:26.08-py3 /entry.sh
```

`--gpus all` and `--ipc=host` are not optional: without the first you get `RuntimeError: Failed to infer device type` (the container cannot see the GPU); without the second you get `RuntimeError: Insufficient space in /dev/shm: 160 MiB required, 64 MiB free` (exactly how the 12:26 attempt died, log `pp2ngc_head_20261122_1310.log`). The original pair (12:28) measured: `Binds = ["$MODEL_DIR:/models","$BASE/p4/pp2_ngc_entry.sh:/entry.sh"]`, `Entrypoint = ["/bin/bash"]`, `Cmd = ["/entry.sh"]`, `NetworkMode = host`.

**Starting with logging (discipline from v0.1 on)**: replace `Cmd` with `bash -c "/entry.sh 2>&1 | tee /p4/logs/pp2ngc_<rank>_<tag>.log"` and add `-v $BASE/p4:/p4`, so the console and the file both keep the record.

### 14.2 `pp2_ngc_entry.sh` (PP=2, one file for both nodes, `NODE_RANK` distinguishes them)

```bash
#!/bin/bash
# NGC + Ray dual-node PP=2 trial (DeepSeek-V4-Flash-DSpark, fp8 weights + fp8 KV, enforce-eager)
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

### 14.3 `head_tp2s.sh` (cross-node TP=2, anemll 0.1.1, `:8888`)

```bash
#!/bin/bash
# vllm head — verbatim 1:1 copy of Anemll 0.1.1 (from anemll/dspark-vllm-gx10 docker-compose.yml)
set -e

# === NCCL / IP basics ===
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

# === the 17 anemll DSpark env vars ===
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

# === Models / cache paths ===
export HF_HOME=/cache/huggingface

# === Node rank / master ===
export NODE_RANK=0
export MASTER_ADDR=$HEAD_IP
export MASTER_PORT=25000
export MTP_NUM_TOKENS=5

# === JIT monitor ===
export JIT_MONITOR_MODE=warn

cd /vllm-workspace

# === vllm serve args (from the anemll docker-compose.yml command section) ===
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

### 14.4 `worker_tp2s.sh` (same as above, `NODE_RANK=1` + `--headless`)

```bash
#!/bin/bash
# vllm worker — verbatim 1:1 copy of Anemll 0.1.1, headless
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
