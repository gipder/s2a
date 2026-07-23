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
| Oracle (정답 텍스트, ASR 없음) | 85.6% (목표) | `src/tier1_oracle.py` |
| whisperv3 + Qwen 8B (실제 cascade) | 78.1% (목표) | TODO |

### Tier별 채점 정책이 다름 (중요)

[audio2tool.github.io](https://audio2tool.github.io/)의 worked example로 확인: **Tier-1의 ground
truth는 인자 없이 `Tool: setZoneTemperature`처럼 tool 이름만 표시**되고, Tier-2부터는 `Tool: X:
{args...}`처럼 인자까지 포함됨. 즉 Tier-1은 EM이 tool-name match와 동일하고(`expected_tool_call`의
trailing `()`는 "인자가 0개여야 정답"이 아니라 그냥 호출 문법 표기), 인자 채점은 Tier-2부터
시작됨. `tier1_oracle.py`는 이걸 반영해서 EM = Tool-Acc로 계산함(`action_metrics.em()`은 안 씀).
나중에 tier2_oracle.py 등을 만들 때는 `action_metrics.em()`(name+args 전체 매치)을 실제로 써야
함 — Tier-1 스크립트의 name-only 로직을 그대로 재사용하면 안 됨.

### 실행

```bash
# 파일럿 (50개 샘플로 하니스부터 검증)
N_QUERIES=50 ./script/run_tier1_oracle.sh

# 전체 2,146 query (전체 152개 도구)
./script/run_tier1_oracle.sh

# retriever-shaped top-k 실험: GT + 같은 도메인 랜덤 (k-1)개 (진짜 retriever는 아직 없음, 상한선 측정용)
TOPK=5 ./script/run_tier1_oracle.sh

# domain 필터링: GT의 실제 도메인에 속한 도구 전부 (도메인별 13~86개, 152개 전체와 top-k 사이 중간 지점)
DOMAIN_FILTERED=1 ./script/run_tier1_oracle.sh

# thinking mode
ENABLE_THINKING=1 ./script/run_tier1_oracle.sh
```

결과는 `experiment/tier1_oracle/<model_name>_<tool_format>_<topk_tag>_<think_tag>.json`에 저장됨
(paper 목표치 대비 accuracy, 샘플별 prompt/raw_output/예측/정답 포함 — 전체 152개 도구 케이스는
프롬프트의 도구 목록 부분만 `tools_registry.csv` 참조 문구로 축약해서 저장, top-k 케이스는 후보가
샘플마다 달라서 프롬프트 전체 저장).

### 결과 (2026-07-23, n=2,146, no-thinking, greedy)

| Tool 후보 개수 | Acc = EM | 비고 |
|---|---|---|
| 152개 전체 | 64.4% (1383/2146) | 논문 목표 85.6%에 -21.2%p |
| domain 필터링 (GT의 실제 도메인 전부, 13~86개) | 78.6% (1686/2146) | 152개 전체와 top-5 사이 |
| top-5 (GT + 같은 도메인 랜덤 4개) | 96.9% (2080/2146) | |
| top-3 | 98.1% (2105/2146) | |
| top-1 (GT만) | 98.5% (2114/2146) | 하니스 자체의 상한선 (파싱/포맷 신뢰도 체크) |

**도구 수를 152→5개로만 줄여도 64.4%→96.9%.** `reasoning_for_asr`의 Tier4 발견("도구 수가 가장
큰 성능 요인")이 Tier1에서도 재확인됨 — top-5는 무작위 distractor인데도 논문 목표를 이미 넘어섬.
STOP에서는 domain 필터링만으론 성능이 안 올랐던 것과 대조적이라, 이 프로젝트의 retriever 연구
방향에 좋은 신호.

domain 필터링 결과를 도메인별로 쪼개보면 (도구 수: smart_car 86 / smart_home 53 / wearables 13):

| 도메인 | Acc |
|---|---|
| wearables | 97.2% |
| smart_car | 86.1% |
| smart_home | 73.2% |

smart_home이 smart_car보다 도구 수가 적은데도(53 vs 86) 정확도가 더 낮음 — domain 필터링은
cross-domain 혼동(52.4%의 오답 원인)은 없애주지만, smart_home 내부의 근사-동의어 중복
(`setLighting`/`setLightState` 등, 오답의 45.5%)은 여전히 남기 때문. 즉 이 갭을 마저 메우려면
domain을 넘어 실제 tool 단위로 후보를 좁히는 retriever가 필요하다는 뜻 — top-5 결과(96.9%)가
그 상한선을 보여줌.

### 발견한 taxonomy(DB) 버그

152개 전체 도구를 줄 때 갭(64.4% vs 85.6%)의 상당 부분이 모델 실수가 아니라 `tools_registry.csv`
자체의 구조적 중복/라벨링 문제로 보임:

- **도메인 간 근사-쌍둥이 함수**: `getBatteryLevel`(smart_home)↔`getBatteryStatus`(smart_car),
  `getLockState`↔`getLockStatus`, `setFanMode`↔`setFanSpeed`, `getCameraStream`↔`viewRemoteCamera`,
  `setVolume`↔`setAudioVolume`, `setThermostatMode`↔`setHvacMode` 등 — 오답의 52.4%가 정답과 다른
  domain의 도구를 고른 경우.
- **도메인 내 근사-동의어**: `setLighting`↔`setLightState`(최다 오답), `rebootDevice`↔`restartDevice`,
  `setSecurityMode`↔`armSecuritySystem` — description까지 사실상 동일. 오답의 45.5%.
- **item의 `domain` 필드와 `tools_registry.csv`의 실제 도메인이 어긋나는 라벨링 버그**: 64/2146
  query(`setLockState`, `controlPlayback`, `getLockState`, `setVolume`)가 smart_home 쿼리인데
  gold `expected_tool_call`은 smart_car 전용 도구를 가리킴 — smart_home 전용 버전(`setLockState_home`
  등, "_home" suffix가 이름 충돌 회피용으로 붙어있음)이 따로 있는데도 안 씀. `tier1_oracle.py`의
  `sample_topk_candidates`는 이 문제를 우회하려고 item의 `domain` 필드 대신 tool 이름으로 registry를
  직접 조회함(자세한 배경은 함수 docstring 참고).

## Retriever: 실제로 만들 수 있을까

Audio2Tool은 8개 tier 전부 HF config가 `split: test`뿐이라 **train split이 없음** — STOP처럼
`train_retriever.py`(contrastive dual-encoder LoRA)를 바로 돌릴 학습 데이터가 없다. 학습 없이
얼마나 되는지부터 확인하려고, STOP retriever가 zero-shot 베이스라인으로 검증했던 것과 동일한
레시피(PromptEOL 프롬프트 wrapping + last-token pooling, `src/embedding_utils.py`)로
`src/eval_zeroshot_retriever.py`를 만들어 152개 도구 description을 코퍼스로 Tier-1 utterance를
검색해봄.

```bash
./script/run_eval_zeroshot_retriever.sh              # 전체 2,146 query
N_QUERIES=200 ./script/run_eval_zeroshot_retriever.sh # 파일럿
MODEL=model/Qwen3-1.7B ./script/run_eval_zeroshot_retriever.sh
```

### 결과 (2026-07-23, n=2,146, Qwen3-0.6B, 학습 없음)

`Recall@k`(정확한 tool이 top-k 안에 있는지)가 진짜 성능 지표. `Domain-only Recall@k`(top-k 중
하나라도 정답과 같은 domain인지)는 참고용으로 같이 쟀는데, **domain이 3개뿐이고 corpus 152개 중
smart_car 86개(57%)/smart_home 53개(35%)로 큰 domain이 절반 가까이를 차지**해서 k가 커지면
거의 랜덤과 구분이 안 됨 — 그래서 랜덤 베이스라인을 같이 표시함(도구를 무작위로 k개 뽑았을 때
같은 domain이 하나라도 걸릴 확률, 조합론으로 계산: `1 - C(N-m,k)/C(N,k)`, 실제 쿼리의 domain 분포로
가중평균).

| k | Recall@k (정확한 tool) | Domain-only Recall@k | 랜덤 베이스라인 (domain-only) |
|---|---|---|---|
| 1 | 28.1% | 47.8% | 31.8% |
| 3 | 45.1% | 71.3% | 65.3% |
| 5 | 52.4% | 80.6% | 80.0% |
| 10 | 61.9% | 91.6% | 91.9% |

**k=5부터는 domain-only recall이 랜덤과 사실상 동일**(80.6% vs 80.0%, 91.6% vs 91.9%) — zero-shot
retriever가 domain 레벨에서는 랜덤보다 나을 게 없다는 뜻. k=1,3에서만 랜덤 대비 유의미한 차이가
있음(+16%p, +6%p). 결론적으로 이 지표는 "retriever가 적어도 엉뚱한 domain으로 새지는 않는지"를
확인하는 용도로만 쓸모 있고, 실전 성능 판단은 정확한 tool의 Recall@k로 해야 함.

랜덤(5/152≈3.3%)보다 recall@k는 훨씬 낫지만 **top-5 안에 정답이 있을 확률이 52.4%뿐**이라, 이
retriever를 그대로 하드 컷오프로 파이프라인에 연결하면(top-5만 LLM에 줌) 전체 정확도가 대략
52.4%×0.97(top-5-with-GT일 때의 LLM 성공률) ≈ **51%** 로, 오히려 152개 전체를 그냥 주는 것(64.4%)보다
못하다 — retriever가 후보를 잘못 좁히면 정답 자체가 사라지는 하드 실패 모드가 생기기 때문. 즉
**학습 없는 zero-shot retriever는 아직 실전 투입 수준이 아님.** 두 가지 방향이 남음:
(1) k를 10~20 정도로 넉넉히 잡거나, (2) 8개 tier를 합친 ~16,843개 라벨 데이터에서 자체
train/dev/test를 만들어 실제로 학습.

### (1)번 확인: k를 넓혀서 진짜 파이프라인을 돌려보면?

`tier1_oracle.py --retrieved_from <retriever json> --retrieved_topk N`으로, oracle(GT 무조건 포함)이
아니라 zero-shot retriever가 **실제로 고른** top-N을 LLM에 그대로 넣어서 end-to-end 정확도를 쟀다.

```bash
RETRIEVED_FROM=experiment/zeroshot_retriever/Qwen3-0.6B.json RETRIEVED_TOPK=10 \
  ./script/run_tier1_oracle.sh
```

| RETRIEVED_TOPK | retriever exact recall@k | 실제 end-to-end Acc=EM |
|---|---|---|
| 10 | 61.9% | 51.8% (1112/2146) |
| 20 | 72.2% | 58.0% (1244/2146) |

둘 다 152개 전체(64.4%)보다 낮다 — **k를 10~20 정도로 넓히는 것만으로는 90%는커녕 지금 baseline도
못 넘는다.** recall@k 곡선 자체가 완만해서(recall@50=86.0%, recall@65=90.0%, recall@76=92.3%),
"실제로 90% 근처"를 찍으려면 k≈65까지 넓혀야 하는데 이건 전체 152개의 43%라 사실상 필터링이라고
부르기 민망한 수준이다. 결론: 학습 없이 k만 조절하는 접근(옵션 1)으로는 90%에 도달할 수 없고,
성능을 올리려면 실제 학습(옵션 2, train/dev/test 자체 구성)이 필요함이 정량적으로 확인됨.

### STOP으로 학습된 retriever를 그대로 가져오면 (Audio2Tool 학습 0)

Audio2Tool 전용 학습 데이터를 만들기 전에, **완전히 다른 데이터셋(STOP)으로 학습된 retriever가
Audio2Tool에 어느 정도나 전이(transfer)되는지** 먼저 확인. `eval_zeroshot_retriever.py
--lora_adapter`로 `noise_aware_slu`의 STOP 학습 체크포인트
(`qwen3-0.6b_depth0.1_seed44_5ep/epoch5`, contrastive dual-encoder LoRA, 5 epoch)를 그대로
로드해서 X/Y 인코더로 씀 — Audio2Tool 데이터는 학습에 전혀 안 들어감.

```bash
python src/eval_zeroshot_retriever.py --model model/Qwen3-0.6B \
  --lora_adapter ../noise_aware_slu/experiments/retriever_train/qwen3-0.6b_depth0.1_seed44_5ep/epoch5
```

| k | Zero-shot (학습 0) | STOP 학습 retriever (Audio2Tool 학습 0, transfer만) | 랜덤 베이스라인 (domain-only) |
|---|---|---|---|
| recall@1 | 28.1% | 26.9% | - |
| recall@3 | 45.1% | 46.9% | - |
| recall@5 | 52.4% | **55.7%** | - |
| recall@10 | 61.9% | **66.2%** | - |
| domain-only@5 | 80.6% (≈랜덤 80.0%) | **91.3%** | 80.0% |
| domain-only@10 | 91.6% (≈랜덤 91.9%) | **96.6%** | 91.9% |

STOP taxonomy를 한 번도 안 보고 Audio2Tool taxonomy도 학습에 전혀 안 썼는데, recall@1만 소폭
하락하고(STOP 관습에 맞춰진 편향 추정) recall@5/10은 확실히 개선됨. domain-only recall도 STOP
학습본은 랜덤(80.0%/91.9%)을 확실히 상회하는 반면(위 절에서 확인했듯 zero-shot 쪽은 랜덤과
거의 동일) — STOP 학습이 최소한 domain 레벨의 신호는 진짜로 배웠다는 뜻. "발화를 intent 의미공간에
매핑하는" 능력 자체가 데이터셋을 넘어 어느 정도 전이된다는 걸 recall@k(정확한 tool 기준)로도
확인. Audio2Tool 전용 학습(옵션 2)을 하면 이 전이 성능이 하한선 역할을 해줄 걸로 기대됨.

## 재사용

- `src/action_metrics.py` — `noise_aware_slu/src/action_metrics.py`에서 포팅 + 확장. Audio2Tool의
  `expected_tool_call` 필드가 STOP retriever와 동일한 canonical action 문법
  (`INTENT(SLOT="value", ...)`)이라 파서 재사용 가능. 단, Audio2Tool 쪽 gold 문자열은 작은따옴표
  (`deviceId='dryer_1'`)도 쓰길래 원래 큰따옴표(JSON)만 지원하던 `parse_string`에 작은따옴표 지원을
  추가함.
- `script/run_with_vllm.sh` — `reasoning_for_asr/scripts/run_with_vllm.sh` 포팅 (vLLM 서버 자동
  기동/종료 + `noise_aware_slu/scripts/run_with_vllm.sh`의 `no_proxy` 예외처리도 함께 포팅).

## 다음 단계 (미착수)

- [ ] Whisper-large-v3 cascade 붙이기 (`whisperv3 + Qwen 8B`, 78.1% 목표)
- [ ] tools_registry.csv의 도메인 라벨 버그(64건) 수정
- [ ] Tier-2 이상으로 확장 (이때부터 `action_metrics.em()`으로 진짜 인자까지 채점)
- [ ] 8개 tier(~16,843개 라벨) 합쳐서 retriever 학습용 train/dev/test 자체 split 만들기 —
      "논문 재현용 zero-shot 수치"와 "retriever 학습 데이터"가 데이터를 나눠 가져야 해서 설계 필요
- [ ] 학습된 retriever로 top-k recall 개선해서 zero-shot(52.4%@5) 대비 oracle 상한선(96.9%)에
      근접시키기
