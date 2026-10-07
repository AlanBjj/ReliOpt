#!/usr/bin/env bash
# [E8] Execution-design ablation of the typed plan without verification: switches off one design at a time
# (variants full / no_sink / short / k5 / cmp_llm / untyped / no_check of scripts/e8_exec_ablation.py) on the three test
# sets, then paired bootstrap against full (scripts/e8_summarize.py).
# "full" reproduces Plan-Only of e3 question by question only with the same LLM call cache (cache/llm/<tag>.sqlite).
# Usage: bash run/e8_exec_ablation.sh <tag> [gpus]      Resumable like run/e3_method.sh.
set -euo pipefail
TAG="${1:?model tag: llama31-8b | qwen3-8b}"
GPUS="${2:-1,2,3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY="${PY:-python}"
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)"
mkdir -p logs
LOG="logs/e8_${TAG}.log"
IFS=',' read -ra GLIST <<< "$GPUS"
for g in "${GLIST[@]}"; do
  PORT=$((8100 + g))
  curl -s --noproxy '*' -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" | grep -q 200 \
    || { echo "[E8] vLLM on GPU $g (port $PORT) is not up"; exit 1; }
done
echo "[E8] start $(date '+%m-%d %H:%M')  $TAG  GPUs=$GPUS" | tee -a "$LOG"
"$PY" -u scripts/e8_exec_ablation.py --tag "$TAG" --gpus "$GPUS" 2>&1 | grep --line-buffered -v -i warn | tee -a "$LOG"
"$PY" -u scripts/e8_summarize.py --tag "$TAG" 2>&1 | tee -a "$LOG"
echo "[E8] done $(date '+%m-%d %H:%M')  $TAG" | tee -a "$LOG"
