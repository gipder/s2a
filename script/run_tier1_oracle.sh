#!/bin/bash
# Audio2Tool Tier-1 oracle reproduction (paper Table 3 "Qwen 8B" row, target Acc=EM=85.6%).
#
# Usage:
#   ./script/run_tier1_oracle.sh                    # full 2,146-query run, all 152 tools
#   N_QUERIES=50 ./script/run_tier1_oracle.sh        # quick pilot
#   ENABLE_THINKING=1 ./script/run_tier1_oracle.sh
#   TOPK=5 ./script/run_tier1_oracle.sh              # retriever-shaped upper bound (oracle, GT guaranteed)
#   DOMAIN_FILTERED=1 ./script/run_tier1_oracle.sh   # all tools from GT's own domain (~53/86/13)
#   RETRIEVED_FROM=experiment/zeroshot_retriever/Qwen3-0.6B.json RETRIEVED_TOPK=10 \
#     ./script/run_tier1_oracle.sh                   # REAL retriever candidates, GT not guaranteed

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
RETRIEVED_FROM="${RETRIEVED_FROM:-}"
RETRIEVED_TOPK="${RETRIEVED_TOPK:-10}"
THINK_FLAG=""
[[ "${ENABLE_THINKING:-0}" == "1" ]] && THINK_FLAG="--enable_thinking"

CMD=(python "$BASE_DIR/src/tier1_oracle.py" --model "$MODEL" --tool_format "$TOOL_FORMAT")
[[ -n "$N_QUERIES" ]] && CMD+=(--n_queries "$N_QUERIES")
[[ -n "$TOPK" ]] && CMD+=(--topk "$TOPK")
[[ "${DOMAIN_FILTERED:-0}" == "1" ]] && CMD+=(--domain_filtered)
[[ -n "$RETRIEVED_FROM" ]] && CMD+=(--retrieved_from "$RETRIEVED_FROM" --retrieved_topk "$RETRIEVED_TOPK")
[[ -n "$THINK_FLAG" ]] && CMD+=("$THINK_FLAG")

"$BASE_DIR/script/run_with_vllm.sh" -m "$MODEL" -g "$GPUS" -t "$TP_SIZE" -- "${CMD[@]}"
