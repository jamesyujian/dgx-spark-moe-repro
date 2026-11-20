#!/usr/bin/env bash
# FP8 + triton + TP=2 起服模板（SM120 / 2×PRO 5000 48G 实测口径）
# 用法: bash serve_fp8_tp2.sh <MODEL_DIR> <PORT> <LOG>
#   例: bash serve_fp8_tp2.sh /models/Qwen3.6-35B-A3B-FP8 8230 results/run_fp8/serve.log
set -eu

MODEL_DIR="${1:?need MODEL_DIR}"
PORT="${2:?need PORT}"
LOG="${3:?need LOG}"
mkdir -p "$(dirname "$LOG")"

# 关键：必须显式 --moe-backend，否则 auto 选谁因版本而异，结论无法归因
# 关键：max-num-seqs 别超默认会觉得越大越好 —— 该模型有 Mamba 块，实际撞上限会起服失败
exec vllm serve "$MODEL_DIR" \
  --served-model-name qwen36-fp8 \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --moe-backend triton \
  --trust-remote-code \
  > "$LOG" 2>&1
