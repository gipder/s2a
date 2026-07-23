# s2a

Audio2Tool 벤치마크([arXiv:2604.22821](https://arxiv.org/abs/2604.22821), "Speak, Call, Act")
Table 3 결과 재현 + retriever 기반 개선 프로젝트. STOP 데이터셋으로 먼저 시도했던
[noise_aware_slu/retriever](../noise_aware_slu/retriever)의 후속으로, STOP은 함수명 중복/파라미터만
다른 variant/nested 구조 때문에 retriever 효과가 잘 안 보였던 반면 Audio2Tool은 152개 도구가
명확히 분리돼 있어 (reasoning_for_asr의 Tier4 실험에서도 "도구 수 감소가 가장 큰 성능 요인"으로
확인됨) 다시 시도해볼 가치가 있음.

## 구조

```
.
├── src/         # python 코드
├── script/      # bash 실행 스크립트
├── model/       # -> noise_aware_slu/models 심링크 (git 제외)
├── data/        # -> reasoning_for_asr/data 심링크 (git 제외)
└── experiment/  # 실험 결과 (git 제외)
```

## 1차 목표: Tier-1 (Direct) 재현

논문 Table 3, `Qwen 8B` 행:

| 세팅 | Acc = EM | 스크립트 |
|---|---|---|
| Oracle (정답 텍스트, ASR 없음) | 85.6% | `src/tier1_oracle.py` |
| whisperv3 + Qwen 8B (실제 cascade) | 78.1% | TODO |

Tier-1은 파라미터가 없는 순수 tool-name 선택 과제라 F1 undefined, Acc=EM.

```bash
# 파일럿 (50개 샘플로 하니스부터 검증)
N_QUERIES=50 ./script/run_tier1_oracle.sh

# 전체 2,146 query
./script/run_tier1_oracle.sh

# thinking mode
ENABLE_THINKING=1 ./script/run_tier1_oracle.sh
```

결과는 `experiment/tier1_oracle/<model_name>.json`에 저장됨 (paper 목표치 대비 accuracy, 샘플별 예측/정답 포함).

## 재사용

- `src/action_metrics.py` — `noise_aware_slu/src/action_metrics.py`에서 그대로 포팅. Audio2Tool의
  `expected_tool_call` 필드가 STOP retriever와 동일한 canonical action 문법
  (`INTENT(SLOT="value", ...)`)이라 파서 수정 없이 재사용 가능.
- `script/run_with_vllm.sh` — `reasoning_for_asr/scripts/run_with_vllm.sh` 포팅 (vLLM 서버 자동 기동/종료).

## 다음 단계 (미착수)

- [ ] Whisper-large-v3 cascade 붙이기 (`whisperv3 + Qwen 8B`, 78.1% 목표)
- [ ] Tier-1 수치가 논문에 수렴하면 Tier-2 이상으로 확장
- [ ] retriever로 tool 후보 필터링해서 STOP에서 안 됐던 걸 Audio2Tool에서 재시도
