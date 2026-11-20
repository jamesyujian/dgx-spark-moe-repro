# 容器镜像与"版本独占后端"的复现约束

## 1. 镜像清单（实测用过的三个）

| 镜像 | 大小 | 镜像 ID | 内含 vllm | 用于 |
|---|---|---|---|---|
| `nvcr.io/nvidia/vllm:26.10-py3` | 38.3 GB | `0a0de9362545` | **0.31.0+88641915.dev** | 32G / SM120 的后端矩阵；`flashinfer_cutile` 等版本独占后端 |
| `nvcr.io/nvidia/vllm:26.08-py3` | — | `20b5b6d2f470`（101）／`3db89c5fcf8b`（102） | **0.27.1** | GB10 / SM121 |
| （venv 口径，非容器） | — | — | **0.29.0** | 115 的主矩阵 |

## 2. 一条铁律：版本独占的后端只能在对应镜像里测

`flashinfer_cutile` 只出现在 0.31.0 的后端清单里（0.29.0 是 9 项，差集正是它）。
⇒ 「0.31.0 + ≥48G」这个组合**至今无数据**（115 只有 0.29.0、116 只有 32G），属"**无数据**"，**不得写成"不可用"**。

## 3. 起服模板（容器口径，用于 32G 与 GB10）

```bash
docker run -d --name q-<tag> --gpus all --ipc host --net host \
  --shm-size=64gb --ulimit memlock=-1:-1 \
  -v /path/to/models:/models --entrypoint python3 \
  nvcr.io/nvidia/vllm:26.10-py3 \
  -m vllm.entrypoints.openai.api_server \
  --model /models/Qwen3.6-35B-A3B-NVFP4 \
  --served-model-name qwen36-nvfp4 --host 127.0.0.1 --port <PORT> \
  --max-model-len 8192 --max-num-seqs 32 --gpu-memory-utilization 0.85 \
  --moe-backend <BACKEND> --trust-remote-code
```

## 4. 采集与清理（每次跑完必须做，否则下一格会被残留进程污染）

```bash
# 取日志（docker logs 不会自动落盘，必须显式重定向）
docker logs q-<tag> > logs/<tag>.log 2>&1

# 判定"服务真的没了"：不要用 pgrep -f 'vllm serve'（会匹配到 ssh 的 bash -c）
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
docker ps --format '{{.Names}} {{.Image}} {{.Status}}'

# 清场
docker rm -f q-<tag>
```

## 5. 三种拒绝要分开记（别混成一类）

| 类型 | 典型日志 | 含义 |
|---|---|---|
| **设备级拒绝** | `...does not support the deployment configuration since kernel does not support current device cuda.` | 该 kernel 没为这个 SM 编译/白名单不含它 ⇒ 换卡或换后端 |
| **QuantKey 拒绝** | `...does not support quantization scheme QuantKey(...)xQ` | 权重契约与该后端不匹配 ⇒ 换权重格式或换后端 |
| **容量 OOM** | `OutOfMemoryError` / 剩余显存不足的 MiB 报数 | 卡太小 ⇒ 换卡/降 ctx/降 util |

> 判错的代价很大：把"设备级拒绝"写成"容量不够"，会让人误以为换 48G 卡就能跑。
