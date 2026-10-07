#!/usr/bin/env bash
# [E2] External baselines on one dataset (500 test questions).
# First tunes IRCoT's K in {4,6,8} and Plan-and-Budget's retrieval (question only / plus every sub-question) on the 100
# tuning questions, then runs on the test questions: CoT, Standard RAG and the Self-Consistency sample pool (3 seeds),
# Self-Ask, IRCoT (tuned K), Search-o1, PyRAG and Plan-and-Budget (3 seeds each), CoVe, and IRCoT at reduced budgets.
# Output: results/e2/<tag>/{tune,test}/<ds>/<method>[_k<K>][_s<seed>].jsonl; log logs/e2_<tag>_<ds>.log
# Usage: bash run/e2_baselines.sh <tag> <ds> [gpus]      e.g. bash run/e2_baselines.sh llama31-8b musique 0,1,2,3
#        TUNE_METHODS / TEST_METHODS override the method lists. Resumable: finished files are skipped and unfinished
#        ones continue from the LLM call cache.
set -euo pipefail
TAG="${1:?model tag: llama31-8b | qwen3-8b}"
DS="${2:?dataset: hotpotqa | 2wikimultihopqa | musique}"
GPUS="${3:-0,1,2,3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY="${PY:-python}"
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)"
mkdir -p logs
LOG="logs/e2_${TAG}_${DS}.log"
IFS=',' read -ra GLIST <<< "$GPUS"
for g in "${GLIST[@]}"; do
  PORT=$((8100 + g))
  if ! curl -s --noproxy '*' -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" | grep -q 200; then
    echo "[E2] vLLM on GPU $g (port $PORT) is not up; start it with: bash run/e0_serve.sh $TAG $GPUS"; exit 1
  fi
done
echo "[E2] start $(date '+%m-%d %H:%M')  $TAG $DS  GPUs=$GPUS" | tee -a "$LOG"
"$PY" -u scripts/e2_baselines.py --tag "$TAG" --gpus "$GPUS" --split tune --datasets "$DS" --methods ${TUNE_METHODS:-ircot planbudget} \
  --ircot-k 4 6 8 --pb-retrieval question subq --seeds 17 2>&1 | tee -a "$LOG"
"$PY" -u scripts/e2_baselines.py --tag "$TAG" --gpus "$GPUS" --split test --datasets "$DS" \
  --methods ${TEST_METHODS:-simple selfask ircot searcho1 pyrag planbudget cove ircot_budget} --ircot-k auto --pb-retrieval auto 2>&1 | tee -a "$LOG"
echo "[E2] done $(date '+%m-%d %H:%M')  $TAG $DS" | tee -a "$LOG"
