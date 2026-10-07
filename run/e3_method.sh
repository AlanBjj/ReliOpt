#!/usr/bin/env bash
# [E3] The method with step-level verification and all internal comparisons.
# 1) materialise the 500 calibration questions: plan, default execution, step labels and features, every operator on
#    every step; 2) fit the quality model; 3) price the verification budgets B_SC and B_SC/2 on the tuning questions
#    (B_SC = measured tokens of Self-Consistency, from e2); 4) run the method and its variants on the 500 test questions
#    (seeds 17, 29, 43); 5) materialise the test questions (diagnostics and oracle only) and summarise.
# Needs results/e3/<tag>/config.json (TXT reader chosen by scripts/e1_txt_tune.py) and results/e2/<tag>/test/summary.json.
# Usage: bash run/e3_method.sh <tag> [gpus] [datasets...]      e.g. bash run/e3_method.sh llama31-8b 0,1,2,3
#        Resumable: finished files are skipped and unfinished ones continue from the LLM call cache.
set -euo pipefail
TAG="${1:?model tag: llama31-8b | qwen3-8b}"
GPUS="${2:-0,1,2,3}"
DATASETS=("${@:3}")
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=(hotpotqa 2wikimultihopqa musique)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY="${PY:-python}"
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)"
# one thread per sklearn / BLAS call: the shard processes run 40 workers each, and OpenMP over every core in every call
# pushed the load to ~1000 with idle GPUs (qwen3-8b, 2026-10-07: the gradient-boosted variant stalled)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs
LOG="logs/e3_${TAG}.log"
[ -f "results/e3/$TAG/config.json" ] || { echo "[E3] results/e3/$TAG/config.json missing (round-4 TXT choice)"; exit 1; }
[ -f "results/e2/$TAG/test/summary.json" ] || { echo "[E3] run scripts/e2_summarize.py --tag $TAG first (B_SC)"; exit 1; }
IFS=',' read -ra GLIST <<< "$GPUS"
for g in "${GLIST[@]}"; do
  PORT=$((8100 + g))
  curl -s --noproxy '*' -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" | grep -q 200 \
    || { echo "[E3] vLLM on GPU $g (port $PORT) is not up"; exit 1; }
done
echo "[E3] start $(date '+%m-%d %H:%M')  $TAG  ${DATASETS[*]}  config $(cat results/e3/$TAG/config.json)" | tee -a "$LOG"
"$PY" -u scripts/e3_materialize.py --tag "$TAG" --gpus "$GPUS" --split cal --datasets "${DATASETS[@]}" 2>&1 | tee -a "$LOG"
"$PY" -u scripts/e3_fit.py --tag "$TAG" --datasets "${DATASETS[@]}" 2>&1 | tee -a "$LOG"
"$PY" -u scripts/e3_price.py --tag "$TAG" --gpus "$GPUS" --datasets "${DATASETS[@]}" 2>&1 | tee -a "$LOG"
# the runs are CPU-bound in Python: 8 processes over disjoint question shards (scripts/e3_run.py --shard)
for k in $(seq 0 7); do
  "$PY" -u scripts/e3_run.py --tag "$TAG" --gpus "$GPUS" --split test --datasets "${DATASETS[@]}" --shard $k --nshards 8 \
    --workers 40 2>&1 | grep --line-buffered -v -i warn | tee -a "$LOG" &
done
wait
"$PY" -u scripts/e3_materialize.py --tag "$TAG" --gpus "$GPUS" --split test --datasets "${DATASETS[@]}" 2>&1 | tee -a "$LOG"
"$PY" -u scripts/e3_summarize.py --tag "$TAG" 2>&1 | tee -a "$LOG"
echo "[E3] done $(date '+%m-%d %H:%M')  $TAG" | tee -a "$LOG"
