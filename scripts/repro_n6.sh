#!/usr/bin/env bash
# N6 边界白皮书 v1.0 —— 一键复现脚本（SM120 / SM121 单机）
#
# 只做三件事，不做解释、不自动调参、不替你选档：
#   1) 按 digest 拉起与白皮书完全一致的镜像
#   2) 按白皮书 §3.3 / §9.3 的参数起服
#   3) 用 --包内的 s303_stress.py 跑同口径压测，产出与白皮书可逐列对比的 csv
#
# 用法：
#   bash repro_n6.sh                        # 默认 c4 跑 10 分钟
#   STAGES="c4:600,c8:600,c16:600" bash repro_n6.sh
#   KEEP=1 bash repro_n6.sh                 # 跑完不删容器，便于抓日志
#
# 可覆写环境变量：MODEL_DIR PORT STAGES GPUS UTIL IMAGE OUT SGLANG
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/home/james/models/Qwen3.6-35B-A3B-NVFP4}
PORT=${PORT:-8200}
STAGES=${STAGES:-c4:600}
GPUS=${GPUS:-0}
UTIL=${UTIL:-0.90}
MAXLEN=${MAXLEN:-8192}
MAXSEQS=${MAXSEQS:-32}
OUT=${OUT:-./results/n6_$(date +%Y%m%d_%H%M%S)}
KEEP=${KEEP:-0}

# 白皮书的锚点：默认用 26.10 的 registry digest，不是浮动 tag
IMAGE=${IMAGE:-nvcr.io/nvidia/vllm:26.10-py3@sha256:0a0de9362545e3e062750bec4ad4f722422a04a81182b5258a7deda4325e9313}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STRESS="$HERE/s303_stress.py"

log(){ printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die(){ printf '\n[FAIL] %s\n' "$*" >&2; exit 1; }

# ---------- 0. 环境自检：缺什么直接停，不猜 ----------
log "0/5 环境自检"
command -v docker  >/dev/null || die "docker 不存在"
command -v python3 >/dev/null || die "python3 不存在"
command -v nvidia-smi >/dev/null || die "nvidia-smi 不存在"
[ -f "$STRESS" ] || die "找不到 $STRESS（压测器必须与本脚本同目录）"
[ -d "$MODEL_DIR" ] || die "模型目录不存在: $MODEL_DIR（本包不含权重，需自备）"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/    GPU /'
echo "    IMAGE = $IMAGE"

# ---------- 1. 拉镜像并校验 digest ----------
log "1/5 拉镜像并校验 digest"
docker pull -q "$IMAGE" >/dev/null
GOT=$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$IMAGE" 2>/dev/null | grep -o 'sha256:[0-9a-f]*' | head -1 || true)
if [ -z "${GOT:-}" ]; then
  echo "    [warn] 该引用未带 RepoDigests（本地 tag 情况），跳过 digest 比对"
elif [ "$GOT" = "${IMAGE##*@}" ]; then
  echo "    digest OK: $GOT"
else
  die "digest 不一致：期望 ${IMAGE##*@} 实得 $GOT"
fi

# ---------- 2. 起服 ----------
log "2/5 起服（TP=1 / marlin / max-num-seqs=$MAXSEQS / util=$UTIL）"
mkdir -p "$OUT"
docker rm -f n6-repro >/dev/null 2>&1 || true
docker run -d --name n6-repro \
  --gpus all --ipc=host --network host \
  -e CUDA_VISIBLE_DEVICES="$GPUS" \
  -v "$(dirname "$MODEL_DIR")":/models:ro \
  "$IMAGE" \
  vllm serve "/models/$(basename "$MODEL_DIR")" \
    --served-model-name n6-repro \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$UTIL" \
    --max-model-len "$MAXLEN" \
    --max-num-seqs "$MAXSEQS" \
    --moe-backend marlin \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    >/dev/null
docker logs -f n6-repro > "$OUT/serve.log" 2>&1 &
LOGPID=$!
trap 'kill $LOGPID 2>/dev/null || true' EXIT

# 就绪判定：/v1/models 能返回才继续（不靠 sleep 猜）
log "3/5 等就绪（最长 900 s）"
ok=0
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ok=1; break; fi
  if ! docker ps --format '{{.Names}}' | grep -qx n6-repro; then
    echo; tail -20 "$OUT/serve.log"; die "容器已退出，起服失败"
  fi
  sleep 5
done
[ "$ok" = 1 ] || { tail -40 "$OUT/serve.log"; die "900 s 内未就绪"; }
echo "    就绪。起服日志已落 $OUT/serve.log"

# ---------- 4. 判据留痕（白皮书 §9.3 的两条按日志实测取，不写死数字）----------
log "4/5 记录 §9.3 判据（并发上限 / page-size 对齐）"
grep -iE 'Maximum concurrency|attention block size|mamba page size|Mamba cache mode' "$OUT/serve.log" \
  | tee "$OUT/serve_params.txt" || echo "    [warn] 未匹配到对齐行，请人工核对 serve.log"

# ---------- 5. 压测 ----------
log "5/5 压测 STAGES=$STAGES"
python3 "$STRESS" --port "$PORT" --model n6-repro --stages "$STAGES" --out-dir "$OUT"

# ---------- 收尾 ----------
log "落盘"
[ "$KEEP" = 1 ] && echo "    KEEP=1，容器 n6-repro 保留" || { docker rm -f n6-repro >/dev/null 2>&1 || true; echo "    容器已删除"; }
sha256sum "$OUT"/*.csv "$OUT"/serve.log 2>/dev/null | tee "$OUT/SHA256SUMS" || true
echo
echo "产物: $OUT"
echo "对比口径: 直接与 docs/N6_evidence.jsonl 同指标行逐列比；等级列写「已核」的应能对上，"
echo "          写「社区」「二手」的不必对上，白皮书正文对此有免责声明。"
