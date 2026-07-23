#!/bin/bash
# Generate synthetic per-tool training utterances via Qwen3-32B (4 GPU).
#
# Usage:
#   ./script/run_generate_synthetic_utterances.sh                  # full 152 tools, K=20
#   K=5 N_TOOLS=5 ./script/run_generate_synthetic_utterances.sh     # quick pilot

set -euo pipefail
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="${MODEL:-$BASE_DIR/model/Qwen3-32B}"
# vLLM registers the served model under its realpath (symlinks resolved), so the
# client --model must match exactly -- resolve here once, pass the same value to both.
MODEL="$(realpath "$MODEL")"
GPUS="${GPUS:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
K="${K:-20}"
N_TOOLS="${N_TOOLS:-}"

CMD=(python "$BASE_DIR/src/generate_synthetic_utterances.py" --model "$MODEL" --k "$K")
[[ -n "$N_TOOLS" ]] && CMD+=(--n_tools "$N_TOOLS")

"$BASE_DIR/script/run_with_vllm.sh" -m "$MODEL" -g "$GPUS" -t "$TP_SIZE" -- "${CMD[@]}"
