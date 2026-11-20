#!/usr/bin/env bash
# run_matrix.sh —— 矩阵里的一格，一把梭：起服 → 三轮探针 → 落盘 → 关服
# 用法: bash run_matrix.sh <MODEL_DIR> <PORT> <TAG> [EXTRA_SERVE_ARGS...]
#   例: bash run_matrix.sh /models/Qwen3.6-35B-A3B-FP8 8230 fp8_tp2_triton --tensor-parallel-size 2 --moe-backend triton
set -eu

MODEL_DIR="${1:?need MODEL_DIR}"
PORT="${2:?need PORT}"
TAG="${3:?need TAG}"
shift 3
EXTRA="$*"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/run_${TAG}"
mkdir -p "$OUT"

# 1) 清场（先确认没有残留进程，否则显存被占，起服会以 OOM 假象失败）
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true); do
  kill -9 "$p" 2>/dev/null || true
done
sleep 4

# 2) 起服（后台，日志落盘）
#    ⚠ 这里把 tool parser 写成默认，是踩过坑的：Qwen3.6 输出 Qwen3 方言 XML，
#    不给 --tool-call-parser qwen3_xml 时 API 层 tool_calls 恒为空，tc_* 三题直接判 0/3
#    （2026-11-19 实测：不带解析器 15/20，带了解析器 18/20，别的题一分不动）。
echo "serve_start=$(date '+%F %T')  tag=$TAG  extra=$EXTRA" | tee "$OUT/meta.txt"
setsid nohup vllm serve "$MODEL_DIR" \
  --served-model-name "qwen36-$TAG" --host 127.0.0.1 --port "$PORT" \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --trust-remote-code $EXTRA \
  > "$OUT/serve.log" 2>&1 < /dev/null &

# 3) 等 health（最多 ~5 分钟）
RESULT=TIMEOUT
for i in $(seq 1 75); do
  sleep 4
  if curl -sf -o /dev/null "http://127.0.0.1:${PORT}/health"; then RESULT=HEALTH_OK; break; fi
  if grep -q "Engine core initialization failed" "$OUT/serve.log" 2>/dev/null; then RESULT=EXITED; break; fi
done
echo "RESULT=$RESULT elapsed~$((i*4))s" | tee -a "$OUT/meta.txt"

# 4) 记录 effective backend（没有这一行，结论无法归因）
grep -aE "Using .* MoE backend|Selected .*Kernel|KV cache size" "$OUT/serve.log" | tail -3 | tee -a "$OUT/meta.txt"

# 5) 三轮探针
if [ "$RESULT" = HEALTH_OK ]; then
  for R in 1 2 3; do
    python3 "$ROOT/scripts/probe_20_prompts.py" \
      --base-url "http://127.0.0.1:${PORT}" --model "qwen36-$TAG" \
      --tag "${TAG}_r$R" --out "$OUT/r$R.json" --repeat 1
  done
fi

# 6) 关服 + 清场
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true); do
  kill -9 "$p" 2>/dev/null || true
done
echo "done=$(date '+%F %T')" | tee -a "$OUT/meta.txt"
