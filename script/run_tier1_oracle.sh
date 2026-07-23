#!/bin/bash
# Audio2Tool Tier-1 oracle reproduction (paper Table 3 "Qwen 8B" row, target Acc=EM=85.6%).
#
# Usage:
#   ./script/run_tier1_oracle.sh              # full 2,146-query run
#   N_QUERIES=50 ./script/run_tier1_oracle.sh  # quick pilot
#   ENABLE_THINKING=1 ./script/run_tier1_oracle.sh

set -euo pipefail
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="${MODEL:-$BASE_DIR/model/Qwen3-8B}"
# vLLM registers the served model under its realpath (symlinks resolved), so the
# client --model must match exactly -- resolve here once, pass the same value to both.
MODEL="$(realpath "$MODEL")"
GPUS="${GPUS:-0}"
TP_SIZE="${TP_SIZE:-1}"
N_QUERIES="${N_QUERIES:-}"
TOOL_FORMAT="${TOOL_FORMAT:-domain}"
TOPK="${TOPK:-}"
THINK_FLAG=""
[[ "${ENABLE_THINKING:-0}" == "1" ]] && THINK_FLAG="--enable_thinking"

CMD=(python "$BASE_DIR/src/tier1_oracle.py" --model "$MODEL" --tool_format "$TOOL_FORMAT")
[[ -n "$N_QUERIES" ]] && CMD+=(--n_queries "$N_QUERIES")
[[ -n "$TOPK" ]] && CMD+=(--topk "$TOPK")
[[ -n "$THINK_FLAG" ]] && CMD+=("$THINK_FLAG")

"$BASE_DIR/script/run_with_vllm.sh" -m "$MODEL" -g "$GPUS" -t "$TP_SIZE" -- "${CMD[@]}"
