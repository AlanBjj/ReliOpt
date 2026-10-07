#!/usr/bin/env bash
# [E0] Data and retrieval indexes (CPU only).
# Unpacks IRCoT's processed data and the raw datasets from cache/ircot/downloads/, builds the three retrieval corpora as
# IRCoT does, fixes the test / tuning / calibration question ids (results/e0/ids/) and builds one BM25 index per dataset
# (cache/bm25/), then checks retrieval on the test subsets. About 1-1.5 h with 32 processes, peak memory < 60 GB.
# Usage: bash run/e0_data.sh      (resumable: every stage skips finished outputs)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY="${PY:-python}"
mkdir -p logs results/e0
echo "[E0-data] start $(date '+%m-%d %H:%M')"
CUDA_VISIBLE_DEVICES= "$PY" -u scripts/e0_prepare_data.py --workers 32 2>&1 | tee -a logs/e0_prepare_data.log
CUDA_VISIBLE_DEVICES= "$PY" -u scripts/e0_build_bm25.py 2>&1 | tee -a logs/e0_build_bm25.log
echo "[E0-data] done $(date '+%m-%d %H:%M')"
