#!/bin/bash
# vLLM 서버를 자동 시작/종료하면서 실험 스크립트를 실행합니다.
# (reasoning_for_asr/scripts/run_with_vllm.sh 그대로 포팅, 기본 경로만 s2a 구조에 맞춤)
#
# Usage:
#   ./script/run_with_vllm.sh [옵션] -- <실험 명령>
#
# 옵션:
#   -m, --model PATH        모델 경로 (default: model/Qwen3-8B)
#   -g, --gpus GPUS         사용할 GPU (default: 0)
#   -t, --tp TP_SIZE        tensor parallel size (default: 1)
#   -p, --port PORT         vLLM 포트 (default: 8000)
#   --max-len N             max-model-len (default: 8192)
#
# 예시:
#   ./script/run_with_vllm.sh -- python src/tier1_oracle.py --model model/Qwen3-8B

set -euo pipefail

# http_proxy/https_proxy가 설정된 환경에서는 localhost로 가는 요청까지 프록시를 거치려다
# 실패한다 -- 서버(vLLM)는 정상적으로 떠서 GPU도 정상 동작하는데, 아래 헬스체크 curl과
# 클라이언트(OpenAI SDK)가 둘 다 이 프록시에 막혀서 응답을 못 받는 것처럼 보인다 (증상만
# 보면 서버가 멈춘 것 같지만 실제로는 클라이언트<->localhost 통신이 프록시에서 끊긴 것).
# noise_aware_slu/scripts/run_with_vllm.sh에서 이미 겪은 문제라 동일하게 우회.
no_proxy="localhost,127.0.0.1,::1${no_proxy:+,$no_proxy}"
export no_proxy
NO_PROXY="$no_proxy"
export NO_PROXY

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="$BASE_DIR/model/Qwen3-8B"
GPUS="0"
TP_SIZE=1
PORT=8000
MAX_LEN=8192

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)   MODEL="$2";   shift 2 ;;
        -g|--gpus)    GPUS="$2";    shift 2 ;;
        -t|--tp)      TP_SIZE="$2"; shift 2 ;;
        -p|--port)    PORT="$2";    shift 2 ;;
        --max-len)    MAX_LEN="$2"; shift 2 ;;
        --)           shift; break ;;
        *) echo "[ERROR] 알 수 없는 옵션: $1"; exit 1 ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "[ERROR] 실험 명령을 -- 뒤에 입력하세요."
    exit 1
fi

EXPERIMENT_CMD=("$@")

if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "[WARN] 포트 $PORT 에 이미 서버가 떠 있습니다. 기존 서버를 사용합니다."
    SKIP_START=1
else
    SKIP_START=0
fi

LOG_FILE="/tmp/vllm_$(date +%Y%m%d_%H%M%S).log"
ABS_MODEL="$(realpath "$MODEL")"
export VLLM_MODEL="$ABS_MODEL"

if [[ $SKIP_START -eq 0 ]]; then
    echo "[INFO] vLLM 서버 시작: $ABS_MODEL (GPU=$GPUS, TP=$TP_SIZE, port=$PORT)"
    CUDA_VISIBLE_DEVICES=$GPUS vllm serve "$ABS_MODEL" \
        --tensor-parallel-size "$TP_SIZE" \
        --port "$PORT" \
        --max-model-len "$MAX_LEN" \
        --gpu-memory-utilization 0.85 \
        > "$LOG_FILE" 2>&1 &
    VLLM_PID=$!

    trap '
        echo ""
        echo "[INFO] vLLM 서버 종료 (PID=$VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null
        wait "$VLLM_PID" 2>/dev/null || true
        echo "[INFO] 서버 로그: '"$LOG_FILE"'"
    ' EXIT

    echo "[INFO] 서버 준비 대기 중..."
    TIMEOUT=300
    ELAPSED=0
    until curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; do
        if [[ $ELAPSED -ge $TIMEOUT ]]; then
            echo "[ERROR] 서버가 ${TIMEOUT}초 내에 시작되지 않았습니다."
            echo "[ERROR] 로그 확인: $LOG_FILE"
            exit 1
        fi
        sleep 3
        ELAPSED=$((ELAPSED + 3))
        printf "."
    done
    echo ""
    echo "[INFO] 서버 준비 완료 (${ELAPSED}초 소요)"
fi

echo "[INFO] 실험 시작: ${EXPERIMENT_CMD[*]}"
echo "──────────────────────────────────────────"
cd "$BASE_DIR"
"${EXPERIMENT_CMD[@]}"
echo "──────────────────────────────────────────"
echo "[INFO] 실험 완료"
