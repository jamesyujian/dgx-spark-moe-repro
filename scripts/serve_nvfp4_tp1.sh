#!/usr/bin/env bash
# NVFP4 + marlin + TP=1 起服模板（SM120 / 单卡 PRO 5000 48G 或 PRO 4500 32G）
# 用法: bash serve_nvfp4_tp1.sh <MODEL_DIR> <PORT> <LOG> [gpu_index] [util]
set -eu

MODEL_DIR="${1:?need MODEL_DIR}"
PORT="${2:?need PORT}"
LOG="${3:?need LOG}"
GPU="${4:-0}"
UTIL="${5:-0.90}"
mkdir -p "$(dirname "$LOG")"

export CUDA_VISIBLE_DEVICES="$GPU"

# --enable-auto-tool-choice + qwen3_xml：Qwen3 输出的是 Qwen3 方言 XML，
# 用 hermes 会让 API 层 tool_calls 恒为 null（服务端报 JSONDecodeError、静默退回文本）。
exec vllm serve "$MODEL_DIR" \
  --served-model-name qwen36-nvfp4 \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --moe-backend marlin \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  > "$LOG" 2>&1
