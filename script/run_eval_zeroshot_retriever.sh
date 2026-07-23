#!/bin/bash
# Zero-shot (no training) retriever baseline for Audio2Tool Tier-1.
# No vLLM server needed -- plain transformers embedding script.
#
# Usage:
#   ./script/run_eval_zeroshot_retriever.sh                # full 2,146-query run
#   N_QUERIES=200 ./script/run_eval_zeroshot_retriever.sh   # quick pilot
#   MODEL=model/Qwen3-1.7B ./script/run_eval_zeroshot_retriever.sh

set -euo pipefail
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="${MODEL:-$BASE_DIR/model/Qwen3-0.6B}"
N_QUERIES="${N_QUERIES:-}"
GPU="${GPU:-0}"

CMD=(python "$BASE_DIR/src/eval_zeroshot_retriever.py" --model "$MODEL" --device "cuda:0")
[[ -n "$N_QUERIES" ]] && CMD+=(--n_queries "$N_QUERIES")

CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}"
