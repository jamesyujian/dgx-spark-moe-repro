# 版本与栈清单（复现前先对齐这一层）

> 记法：**栈 = 机型 + SM 版本 + 引擎版本 + 后端 + dtype + TP**。缺任一项，数字就不可归因。
> 本清单只写实测过的三条栈，不写"理论上支持"。

## A. 115 · 2× RTX PRO 5000 48G · SM120 · venv（主力测试机）

| 项 | 值 |
|---|---|
| GPU | 2× RTX PRO 5000 Blackwell，**48,935 MiB / 卡**，capability **sm_120 (12,0)** |
| 驱动 | NVIDIA UNIX Open Kernel Module **615.71.09**（Release Build，2026-09-04） |
| 引擎 | vLLM **0.29.0**（pip venv） |
| torch | **2.13.0+cu130** |
| transformers | **5.17.0** |
| flashinfer | **0.6.18** |
| NCCL | **2.30.7** |
| 模型 | `Qwen3.6-35B-A3B-BF16`（**71.9 GB ≈67.0 GiB**）／`-FP8`（37.49 GB / 55 文件）／`-NVFP4`（≈22 GB） |
| 已测后端 | `marlin`、`humming` 可用；`triton` 在 FP8 MoE 路径可用；`b12x`、`flashinfer_cutlass`、`flashinfer_trtllm`、`flashinfer_cutedsl` 设备级拒；`cutlass`、`emulation` QuantKey 拒 |

**FP8 权重契约（关键）**：`quant_method=fp8`、`fmt=e4m3`、`activation_scheme=dynamic`、**`weight_block_size=[128,128]`**、**31,745 个 `weight_scale_inv`**、per-tensor `weight_scale` **0 个** ⇒ **W8A8 block-scaled 128×128 f32 权重 scale + dynamic per-token 激活**。起服日志应出现 `Selected CutlassFp8BlockScaledMMKernel for Fp8LinearMethod`。

## B. 116 · 1× RTX PRO 4500 32G · SM120 · 容器（版本独占后端的唯一来源）

| 项 | 值 |
|---|---|
| GPU | 1× RTX PRO 4500 Blackwell，**32,623 MiB** |
| 镜像 | `nvcr.io/nvidia/vllm:26.10-py3`（38.3 GB），**镜像 ID `0a0de9362545`** |
| 镜像内引擎 | vLLM **0.31.0+88641915.dev** |
| venv（非容器） | vLLM **0.29.0**（与 A 同版本，用于对照） |
| 模型 | `Qwen3.6-35B-A3B-NVFP4` |
| 已测后端 | `marlin`、`humming`、`b12x` 可用；`flashinfer_b12x`、`flashinfer_cutile` 容量 OOM；`flashinfer_cutedsl` **设备级拒绝**（40 s，`kernel does not support current device cuda`）；`cutlass`、`emulation` QuantKey 拒 |

## C. 101 / 102 · GB10 · SM121 · 容器（架构宽度的唯一来源）

| 项 | 值 |
|---|---|
| GPU | GB10（DGX Spark 一类），SM**121** |
| 镜像 | `nvcr.io/nvidia/vllm:26.08-py3`（ID `20b5b6d2f470`（101）／`3db89c5fcf8b`（102）） |
| 引擎 | vLLM **0.27.1** |
| 已测后端 | `marlin`、`flashinfer_b12x` 可用；`flashinfer_trtllm`、`flashinfer_cutedsl`、`humming`（**NVML 崩溃，归因未取证**）拒；`cutlass`、`flashinfer_cutlass`、`emulation` QuantKey 拒；`b12x`、`flashinfer_cutile` 非法 CLI 值 |

## D. 未覆盖的两格（写清楚，不留白）

- **0.31.0 + ≥48G** 这一组合**没有任何实测**：版本独占的后端（`cutile` 等）只在 0.31.0 的清单里，而 0.31.0 只在 116（32G）上跑过；115 是 0.29.0。⇒ 属"**无数据**"，不得写"不可用"。
- **32G × 0.27.1 / 0.29.0**、**GB10 × 0.29.0 / 0.31.0**：机器上没有对应镜像 ⇒ 属"**无数据（无镜像）**"。

## E. 采样参数基线（跨结论统一用这套，改任何一项都要在结果里注明）

`temperature=0`｜`seed=1234`｜`max_tokens` 按题给（`tc_single` 100、`tc_multi` 160、`tc_none` 40，其余 128）｜`--ignore-eos`（吞吐测试）｜`max-model-len` 8192｜`gpu-memory-utilization` 0.90（32G 上 0.85）｜`max-num-seqs` 64（并发测试 32）。
