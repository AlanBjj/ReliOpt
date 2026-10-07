#!/usr/bin/env bash
# [E0] Start vLLM servers for one model: one tmux session vllm_<tag>_g<gpu> per GPU group, port 8100 + first GPU id.
# Usage: bash run/e0_serve.sh <llama31-8b|qwen3-8b> <gpu groups>      e.g. bash run/e0_serve.sh llama31-8b 0,1,2,3
#        "2+3" serves one instance with tensor parallelism over GPUs 2 and 3 (useful on 24 GB cards).
#        MODEL_DIR=/path/to/weights overrides the Hugging Face id. Running sessions are skipped, so re-running is safe.
#        Stop with: bash run/e0_serve_stop.sh <tag>
# qwen3-8b and qwen3-8b-think share one server (thinking mode is set per request through chat_template_kwargs).
set -euo pipefail
TAG="${1:?model tag: llama31-8b | qwen3-8b}"
GPUS="${2:?comma-separated GPU ids, e.g. 0,2,3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
VLLM="${VLLM:-vllm}"
# VLLM_USE_FLASHINFER_SAMPLER=0: vLLM's PyTorch sampler, as in the paper's runs (FlashInfer's sampler is JIT-compiled
# and needs an nvcc matching torch's CUDA; per-request seeds apply either way).
case "$TAG" in
  llama31-8b) MODEL=meta-llama/Llama-3.1-8B-Instruct ;;
  qwen3-8b)   MODEL=Qwen/Qwen3-8B ;;
  *) echo "unknown model tag $TAG"; exit 1 ;;
esac
MODEL="${MODEL_DIR:-$MODEL}"
mkdir -p logs
IFS=',' read -ra GLIST <<< "$GPUS"
for grp in "${GLIST[@]}"; do
  g="${grp%%+*}"; DEVS="${grp//+/,}"; TP=$(( $(tr -cd '+' <<< "$grp" | wc -c) + 1 ))
  S="vllm_${TAG}_g${g}"; PORT=$((8100 + g))
  if tmux has-session -t "=$S" 2>/dev/null; then echo "[E0] $S already running"; continue; fi
  echo "[E0] start $S on GPU $DEVS (tensor parallel $TP) port $PORT"
  tmux new-session -d -s "$S" "VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=$DEVS $VLLM serve $MODEL --served-model-name $TAG \
    --host 127.0.0.1 --port $PORT --dtype bfloat16 --gpu-memory-utilization 0.90 --max-model-len 16384 --seed 0 \
    --tensor-parallel-size $TP 2>&1 | tee -a logs/vllm_${TAG}_g${g}.log"
done
for grp in "${GLIST[@]}"; do
  g="${grp%%+*}"; PORT=$((8100 + g))
  for i in $(seq 1 120); do
    if curl -s --noproxy '*' -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" | grep -q 200; then
      echo "[E0] GPU $g port $PORT ready"; break
    fi
    [ "$i" = 120 ] && { echo "[E0] GPU $g port $PORT NOT ready after 10 min; see logs/vllm_${TAG}_g${g}.log"; exit 1; }
    sleep 5
  done
done
echo "[E0] all ready: $(for grp in "${GLIST[@]}"; do printf 'http://127.0.0.1:%s ' $((8100 + ${grp%%+*})); done)"
