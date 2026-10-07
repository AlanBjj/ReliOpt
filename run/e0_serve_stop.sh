#!/usr/bin/env bash
# [E0] Stop all vLLM sessions of one model started by run/e0_serve.sh (sessions vllm_<tag>_g*).
# Usage: bash run/e0_serve_stop.sh <llama31-8b|qwen3-8b>
set -uo pipefail
TAG="${1:?model tag}"
for S in $(tmux ls -F '#{session_name}' 2>/dev/null | grep "^vllm_${TAG}_g"); do
  echo "[E0] stop $S"; tmux kill-session -t "=$S"
done
