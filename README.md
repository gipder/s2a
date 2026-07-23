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

> `build_user_prompt`에 "발화에서 인자 값을 직접 추출하라, defaults를 베끼지 마라" 지시문을
> 추가한 **현재** 프롬프트 기준 (배경/효과는 아래 "Tier-2 ... 실제 retriever에서 EM이 떨어지는
> 이유 분석 + 프롬프트 수정" 섹션 참고). 152개 전체/domain 필터링/top-5는 이 프롬프트로
> 재실행 완료. top-3/top-1은 아직 이전 프롬프트 결과라 참고용(Tier-1은 인자가 없어 이 지시문의
> 영향이 적을 것으로 예상되지만 확인 안 됨).

| Tool 후보 개수 | Acc = EM | 비고 |
|---|---|---|
| 152개 전체 | 60.4% (1297/2146) | 논문 목표 85.6%에 -25.2%p |
| domain 필터링 (GT의 실제 도메인 전부, 13~86개) | 77.2% (1657/2146) | 152개 전체와 top-5 사이 |
| top-5 (GT + 같은 도메인 랜덤 4개) | 96.4% (2069/2146) | |
| top-3 (이전 프롬프트) | 98.1% (2105/2146) | |
| top-1 (GT만, 이전 프롬프트) | 98.5% (2114/2146) | 하니스 자체의 상한선 (파싱/포맷 신뢰도 체크) |

**도구 수를 152→5개로만 줄여도 60.4%→96.4%.** `reasoning_for_asr`의 Tier4 발견("도구 수가 가장
큰 성능 요인")이 Tier1에서도 재확인됨 — top-5는 무작위 distractor인데도 논문 목표를 이미 넘어섬.
STOP에서는 domain 필터링만으론 성능이 안 올랐던 것과 대조적이라, 이 프로젝트의 retriever 연구
방향에 좋은 신호. (참고: 인자 추출 지시문 추가로 152개 전체 Acc는 이전(64.4%)보다 소폭 하락함 —
Tier-1엔 원래 불필요한 지시문이라 parse failure만 약간 늘어난 것으로 보임, 아래 섹션 참고.)

domain 필터링 결과를 도메인별로 쪼개보면 (도구 수: smart_car 86 / smart_home 53 / wearables 13):

| 도메인 | Acc |
|---|---|
| wearables | 93.8% |
| smart_car | 86.1% |
| smart_home | 72.1% |

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

**taxonomy 보정 효과 재확인 (현재 프롬프트, 2026-07-23)**: 같은 domain 내 근사-동의어 오답을
정답으로 쳐주면 152개 전체 기준 **60.4% → 74.7%**(오답 850건 중 same-domain 306건/cross-domain
509건/파싱실패 35건), 논문 목표(85.6%)와의 잔여 갭은 **10.9%p**. 이전에 측정했던 것(구버전
프롬프트, 64.4%→80.7%, 잔여 갭 5.0%p)보다 taxonomy로 설명되는 비중이 줄고 잔여 갭이 커졌는데,
원인은 **인자 추출 지시문**(`build_user_prompt`, Tier-2용으로 나중에 추가)이 Tier-1에는 불필요한데도
공용 프롬프트라 그대로 적용돼서 strict/lenient 둘 다 깎아먹었기 때문으로 보임 — Tier-1 전용으로
그 지시문 없는 프롬프트를 따로 만들면 원래 수준(80.7%대)까지 회복될 가능성이 있음. 아직 검증
안 됨, 다음 단계 참고.

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

### 옵션 2 진행 중: synthetic augmentation

Audio2Tool 전용 학습 데이터를 만드는 첫 단계 — 152개 tool의 function card(signature+description)를
LLM(Qwen3-32B)에 주고 tool당 K개의 직접 명령형 utterance를 생성 → test set(8개 tier 전부)과
겹치는 것들을 걸러냄. Audio2Tool 원 저자들도 같은 tool card에서 GPT-5.2/Gemini/Claude Opus로
쿼리를 생성했기 때문에, 다른 LLM으로 생성해도 표현이 우연히 겹칠 위험이 실제로 있음.

```bash
# 1. 생성 (Qwen3-32B, 4 GPU)
./script/run_generate_synthetic_utterances.sh          # 전체 152 tool, K=20
K=5 N_TOOLS=5 ./script/run_generate_synthetic_utterances.sh  # 파일럿

# 2. leakage filtering (8개 tier 전체와 대조)
python src/filter_synthetic_utterances.py \
  --input experiment/synthetic_utterances/raw_Qwen3-32B_k20.json
```

**filtering 방법 관련 중요한 발견**: 처음에 cosine similarity(threshold 0.90) + word-distance(WER,
threshold 0.30) 둘 다 썼더니 3,041개 중 3,039개(99.9%)가 걸러지는 명백한 버그가 났음. 원인을
`--dry_run`으로 전체 분포를 까보니, **PromptEOL 임베딩이 이 정도 길이(2~8단어)의 명령문에서는
실제 유사도와 무관하게 cosine similarity가 전부 0.87~1.0 사이에 몰려있음** — 완전히 무관한 두
문장("Can we make a detour?" vs "Can we get a different input on?")도 0.907이 나옴. 즉 이 길이의
텍스트에는 cosine 기반 필터가 변별력이 없음. word-distance만으로 실제 근접 사례("Find my lost
phone" vs "Find my lost device", word_dist=0.25)와 우연히 문장 구조만 같은 무관한 사례("Turn
recirculation on for me." vs "Turn the lights on for me.", word_dist=0.33)의 경계를 직접 확인해서
0.30을 기준으로 확정, cosine threshold는 0.999(사실상 no-op, 완전 동일 임베딩만 잡는 안전망)로
올림.

**결과 (2026-07-23, Qwen3-32B, K=20/tool)**: 3,041개 생성 → 130개(4.3%) leakage로 제거 → 152개
tool 전체에 최소 9개, 평균 19.2개(2,911개 총합) 확보.

### 학습 결과: base Qwen3-0.6B에서 1 epoch만 (STOP warm-start 없이)

`src/train_retriever.py` — STOP 체크포인트 없이 base Qwen3-0.6B에서 바로 시작, 위 synthetic
corpus로 symmetric in-batch-negative InfoNCE + LoRA 1 epoch(배치 64, 46 step)만 학습. STOP의
`train_retriever.py` 레시피를 이 프로젝트의 훨씬 단순한 corpus 구조(152 tool, call_shape 없음)에
맞게 축소 이식 — DDP/repel loss/체크포인트 재개 없이 한 번에 끝까지.

```bash
python src/train_retriever.py --epochs 1 --output experiment/retriever_train/scratch_1ep
python src/eval_zeroshot_retriever.py --model model/Qwen3-0.6B \
  --lora_adapter experiment/retriever_train/scratch_1ep
```

| k | Zero-shot (학습 0) | STOP 전이 (Audio2Tool 학습 0) | **1 epoch from scratch** |
|---|---|---|---|
| recall@1 | 28.1% | 26.9% | **71.4%** |
| recall@3 | 45.1% | 46.9% | **91.6%** |
| recall@5 | 52.4% | 55.7% | **95.9%** |
| recall@10 | 61.9% | 66.2% | **98.5%** |

단 46 step(2,911개 utterance, 1 epoch)만으로 recall@5가 95.9%까지 뛰어 oracle 상한선(top-5,
GT 보장 시 96.9%)에 근접함. STOP 전이보다 훨씬 큰 폭으로 개선된 이유는 이 데이터가 애초에
Audio2Tool의 정확히 이 152개 taxonomy를 타겟으로 만들어졌기 때문 — STOP 전이는 다른 taxonomy에서
얻은 일반 능력이었던 반면, 이건 직접 맞춘 것.

**진짜(oracle 아닌) retriever로 end-to-end 파이프라인까지 확인**:

```bash
RETRIEVED_FROM=experiment/zeroshot_retriever/Qwen3-0.6B_lora-retriever_train-scratch_1ep.json \
  RETRIEVED_TOPK=5 ./script/run_tier1_oracle.sh
```

| 세팅 | Acc=EM |
|---|---|
| 152개 전체 | 60.4% |
| 실제 retriever @k=10 (학습 0, 이전 프롬프트) | 51.8% |
| 실제 retriever @k=20 (학습 0, 이전 프롬프트) | 58.0% |
| domain 필터링 | 77.2% |
| **실제 retriever @k=5 (1 epoch 학습)** | **74.0%** |
| oracle top-5 (GT 보장, 무작위 distractor) | 96.4% |

152개 전체를 그냥 주는 것보다 확실히 나아졌고(60.4%→74.0%), 미학습 real retriever보다도 훨씬
낫다. 다만 recall@5(95.9%)와 downstream Acc(74.0%) 사이 갭(~22%p)이 여전히 큼 — oracle의 top-5는
**무작위** distractor였던 반면, 학습된 retriever의 top-5는 진짜로 비슷해서 뽑힌 near-neighbor라
LLM 입장에서 더 헷갈리는 선택지일 가능성이 있음(같은 domain 내 근사-동의어 쌍이 retriever
top-5에 몰려 있을 수 있음) — Tier-2에서 이 갭을 파봤더니 실제로 "tool은 맞혔는데 인자가 틀림"이
원인의 대부분이었음(아래 섹션). Tier-1은 인자가 없어서 이 특정 원인은 해당 없고, 원인 규명은
아직 미완료.

## Tier-2 (Parametric) 확장

Tier-1과 같은 4가지 세팅(152개 전체 / domain 필터링 / 실제 retriever @k=5 / oracle top-5)을
Tier-2에도 그대로 적용. `src/tier2_oracle.py`가 `tier1_oracle.py`의 후보 선택 로직(topk/
domain_filtered/retrieved_from — 전부 tier에 무관한 범용 코드)을 그대로 재사용하되, 채점은
Tier-2부터 진짜로 인자까지 보는 `action_metrics.em()`/`slot_f1()`을 씀(Tier-1의 name-only 방식과
다름, 위 "Tier별 채점 정책" 참고).

```bash
./script/run_tier2_oracle.sh                      # 152개 전체
DOMAIN_FILTERED=1 ./script/run_tier2_oracle.sh
TOPK=5 ./script/run_tier2_oracle.sh
RETRIEVED_FROM=experiment/zeroshot_retriever/Qwen3-0.6B_lora-retriever_train-scratch_1ep_tier2.json \
  RETRIEVED_TOPK=5 ./script/run_tier2_oracle.sh
```

retriever는 Tier-1용으로 학습한 것을 그대로 재사용(코퍼스가 152개 tool로 tier 무관하게 동일) —
`eval_zeroshot_retriever.py --tier tier2`로 Tier-2 쿼리에 대해서만 다시 검색만 수행:
recall@5 93.0%(Tier-1은 95.9%, Tier-2 발화가 인자값을 포함해서 조금 더 어려움).

### 파서 확장 필요했음

Tier-2 gold는 실제 인자값을 담고 있어서 `action_metrics.py`가 처음엔 2,041개 중 10개를 파싱
실패했음 — 따옴표 없는 숫자(`speed=7`), 따옴표 없는 bool(`enabled=true`), SQL식 이중 홑따옴표
이스케이프(`'kids'' room'`, 기존 백슬래시 방식과 별개) 세 가지를 추가 지원해서 8개는 해결. 남은
2개는 파서가 아니라 **진짜 데이터 버그**: `getAppప్lianceState`(텔루구 문자가 tool 이름 중간에
섞여 들어감), `setDestination(address='Trader Joe's')`(이스케이프 안 된 아포스트로피로 원천적으로
모호함). 이 둘은 크래시 대신 경고 로그 남기고 채점에서 제외하도록 처리.

### 결과 (2026-07-23, n=3,158, no-thinking, greedy, 인자 추출 지시문 포함 현재 프롬프트)

`lenient Acc`는 같은 domain 안 근사-동의어 오답(예: `setLighting`↔`setLightState`)을 정답 처리한
값 — 아래 "taxonomy 보정이 세팅별로 미치는 영향" 절 참고. **EM은 tool 이름부터 일치해야 하므로
이 보정과 거의 무관**(근사-동의어로 tool 이름이 갈리면 EM은 애초에 실패).

| 세팅 | strict Acc | lenient Acc | EM | F1 |
|---|---|---|---|---|
| 152개 전체 | 69.3% | 86.4% | 43.7% | 62.4% |
| top-1 (real retriever) | 68.2% | 85.0% | 44.7% | 58.1% |
| domain 예측 필터링 (retriever) | 73.7% | 86.7% | 45.3% | 62.6% |
| top-5 (real retriever, 1 epoch 학습) | 78.3% | 92.8% | 48.7% | 64.9% |
| oracle top-5 (GT 보장, 무작위 distractor) | 97.5% | - | 59.4% | 73.7% |
| domain 필터링 (oracle) | 82.9% | 97.8% | 50.6% | 68.9% |
| **논문 목표 (Qwen 8B)** | **77.1%** | - | **10.1%** | **19.3%** |

**strict Acc는 4개 세팅 다 논문 목표(77.1%)와 비슷한 범위**인데, taxonomy 보정한 lenient Acc는
전부 논문 목표를 확실히 넘어섬(85.0~97.8%) — Tier-1과 같은 패턴. 반면 **EM과 F1은 taxonomy
보정과 무관하게 전부 논문보다 훨씬 높은 채로 남음**(EM 43.7~59.4% vs 목표 10.1%, F1 58.1~73.7%
vs 목표 19.3%) — 4~6배 차이라 우연한 오차라고 보기 어렵고, taxonomy 문제로도 설명이 안 됨. 즉
**Acc 쪽 갭은 taxonomy 문제로 대부분 설명되지만, EM/F1 쪽 갭은 별개의 미해결 문제**(채점 기준
자체가 다를 가능성 — 예: 논문이 device ID 같은 slot 값을 더 엄격하게 비교하거나, 우리 쪽
`_normalize_text`의 대소문자/공백 무시가 너무 관대하거나). 원인 파악 전까지 이 Tier-2 EM/F1
수치는 우리 자체 채점 기준으로 해석해야 하고, 논문과의 직접 비교는 보류.

### 실제 retriever에서 EM이 떨어지는 이유 분석 + 프롬프트 수정

152개 전체(EM 44.8%) 대비 실제 retriever @k=5(EM 38.8%)가 오히려 낮았던 이유를 그 차이(463건
회귀)로 분해:

| 원인 | 건수 | 비중 |
|---|---|---|
| retriever가 정답 tool을 top-5에 못 넣음(recall miss) | 47 | 10% |
| 정답 tool은 top-5에 있는데 LLM이 다른 tool을 고름 | 80 | 17% |
| **정답 tool은 맞혔는데 인자가 틀림** | **336** | **73%** |

인자 오류 336건을 더 쪼개면 필요한 인자를 아예 안 씀(60%, 203건) + 발화 내용 대신 tool의
`argument_defaults` 값을 그대로 씀(23%, 77건)이 대부분(83%) — 후보가 5개뿐일 때 모델이 발화에서
값을 뽑아내는 걸 게을리하는 경향. 예:

```
gold:            getAirQuality(deviceId='living_room_sensor')
152개 전체 예측:  getAirQuality(deviceId="living_room_sensor")   -- 정확히 추출
retriever 예측:  getAirQuality(deviceId="sensor_1")              -- tool 기본값을 그대로 씀
```

가설(후보가 적을 때 "발화에서 값을 뽑아 채워야 한다"는 신호가 약해진다) 검증을 위해
`build_user_prompt`에 명시적 지시문 추가:

```
Extract every argument value from the user utterance itself -- do not copy the "defaults"
shown above, and do not omit an argument the utterance specifies a value for.
```

**결과 (실제 retriever @k=5, 같은 3,158개)**:

| | 지시문 전 | 지시문 후 |
|---|---|---|
| Acc | 78.9% | 78.3% (소폭 하락) |
| **EM** | 38.8% | **48.7%** (+9.9%p) |
| F1 | 57.6% | 64.9% (+7.3%p) |
| parse failure | (미기록) | 71/3,158 (2.2%) |

Tier-1에서 시도했던 비슷한 지시문("인자 넣지 마라")은 효과가 전혀 없었는데, 이번엔 실제로
먹혔음 — 방향이 반대라서(넣지 말라 vs 반드시 뽑아써라) 그런 것으로 추정. 부작용: parse
failure가 늘어난 케이스들을 보면 대부분이 모델이 "이 중엔 맞는 tool이 없다"며 답변을 거부하는
새 패턴 — 지시문이 모델을 더 신중하게 만들어서 후보 중 확신이 안 서면 거부하는 쪽으로 간 것으로
보임. 그래도 순효과는 뚜렷하게 플러스.

이 지시문은 `build_user_prompt`가 tier1/tier2, 모든 세팅에 공용이라 위 두 결과표(Tier-1,
Tier-2 4-세팅 비교)에도 이미 반영해서 전체 재실행 완료 — 152개 전체/domain 필터링/oracle
top-5는 Acc가 소폭 하락(대체로 -0.5~4%p)하는 대신 EM/F1은 인자가 실제로 채점되는 Tier-2에서
확실히 상승, Tier-1은 애초에 인자를 안 써서 이 지시문의 이득 없이 하락분만 반영됨(위 Tier-1
섹션 참고).

## domain 필터링을 oracle 대신 retriever 예측으로 하면?

위의 "domain 필터링"은 전부 **GT의 실제 domain을 미리 안다는 oracle 가정**이었음. 실전에선 domain을
모르니, 학습한 retriever의 top-1 tool이 속한 domain을 그대로 "예측 domain"으로 써서 같은 필터링을
해보고(`--retrieved_domain_from`), 별도로 retriever의 top-1 tool 자체를 유일한 후보로 주는 실험
(`--retrieved_from --retrieved_topk 1`, 기존 기능 그대로 재사용)도 같이 돌림.

```bash
# domain 예측 필터링 (oracle 아님)
RETRIEVED_DOMAIN_FROM=experiment/zeroshot_retriever/Qwen3-0.6B_lora-retriever_train-scratch_1ep.json \
  ./script/run_tier1_oracle.sh

# top-1 (retriever의 최선 추측 하나만 후보로)
RETRIEVED_FROM=experiment/zeroshot_retriever/Qwen3-0.6B_lora-retriever_train-scratch_1ep.json \
  RETRIEVED_TOPK=1 ./script/run_tier1_oracle.sh
```

retriever가 top-1 tool로 domain을 예측하는 정확도부터 확인(2026-07-23, top-5 다수결과 비교):

| 예측 방식 | Tier1 | Tier2 |
|---|---|---|
| **top-1 tool의 domain 그대로 씀** | **88.9%** | **88.8%** |
| top-5 다수결(5개 후보 중 domain 최빈값) | 74.3% | 73.4% |

top-1 하나만 보는 게 다수결보다 나음 — top-5는 의미적으로 비슷해서 뽑힌 후보라 다른 domain의
근사-쌍둥이 tool(예: `getBatteryLevel`(smart_home) 후보군에 `getBatteryStatus`(smart_car)가 낌)이
자주 섞여 다수결을 흔듦. 오답은 `smart_home→smart_car` 방향에 압도적으로 몰림(Tier1 156건,
Tier2 275건) — LLM downstream 오답에서 봤던 cross-domain 근사-쌍둥이 함수 혼동과 같은 패턴이
retriever 단계에서도 그대로 나타남.

### 결과 (2026-07-23, no-thinking, greedy, 인자 추출 지시문 포함 현재 프롬프트)

| 세팅 | Tier1 Acc=EM | Tier2 Acc | Tier2 EM | Tier2 F1 |
|---|---|---|---|---|
| 152개 전체 | 60.4% | 69.3% | 43.8% | 62.5% |
| top-1 (retriever 최선 추측 하나만) | 71.1% | 68.2% | 44.7% | 58.1% |
| domain 예측 필터링 (retriever) | 69.5% | 73.7% | 45.3% | 62.6% |
| **top-5 (실제 retriever)** | **74.0%** | **78.3%** | **48.7%** | **64.9%** |
| domain 필터링 (oracle, 진짜 domain) | 77.2% | 83.1% | 50.6% | 69.0% |

두 tier 다 순위는 **152개 전체 < top-1 < domain 예측 필터링 < top-5(실제 retriever) < domain
필터링(oracle)** — top-5가 domain 예측 필터링보다 후보 수는 훨씬 적은데(5개 vs 13~86개)도 더 나은
이유는, retriever의 recall@5(Tier1 95.9%/Tier2 93.0%)가 domain 예측 정확도(88.9%/88.8%)보다
높아서 "정답이 후보 안에 있을 확률" 자체가 더 크기 때문.

**domain 예측 필터링은 152개 전체와 oracle domain 필터링 사이**에 정확히 위치 — 예상대로.
retriever의 domain 예측이 88.9%/88.8%로 완벽하지 않은 만큼 oracle(100% 정확한 domain)보다는
당연히 낮음.

**Tier1과 Tier2에서 top-1의 순위가 다름**이 흥미로움:
- Tier1(인자 없음): top-1(71.1%) > domain 예측 필터링(69.5%) — 후보가 1개뿐이라도 tool 이름만
  맞히면 되니 오히려 domain 필터링(13~86개 중에서 골라야 함)보다 나음.
- Tier2(인자 있음): domain 예측 필터링(Acc 73.7%) > top-1(Acc 68.2%, **parse failure 185건/5.9%로
  다른 모든 세팅 중 최다**) — 후보가 tool 카드 1개뿐이면 모델이 "이건 요청과 안 맞는다"며 거부하는
  경우가 급증(위 "인자 추출 지시문" 섹션에서 확인한 것과 같은 패턴, 여기선 더 심함 — 진짜로 후보가
  하나뿐이라 대안이 없어서). domain 필터링은 최소 13개 이상의 대안이 있어서 이 거부 패턴이 훨씬 덜함.

결론: **domain을 모른다는 현실적 가정 하에서는, retriever로 domain만 예측해서 필터링하는 게
top-1(정확한 tool 하나만 찍기)보다 Tier2(인자 있는 태스크)에서 더 안전하고 나은 선택.** 다만 이미
학습해둔 retriever가 recall@5도 충분히 좋다면(여기 케이스처럼), top-5 그대로 쓰는 게 domain
예측보다도 나음 — domain 예측 필터링은 retriever의 recall이 낮거나 domain 예측 자체가 더 쉬운
상황(예: domain 수가 훨씬 많아서 정확한 tool보다 domain 맞히기가 상대적으로 쉬워지는 경우)에서
더 유리할 걸로 예상.

### taxonomy 보정이 세팅별로 미치는 영향

위 "발견한 taxonomy(DB) 버그"에서 152개 전체는 같은 domain 근사-동의어 오답을 정답 처리하면
60.4%→74.7%였는데, 이걸 다른 4개 세팅에도 똑같이 적용:

| 세팅 | strict | **taxonomy 보정(lenient)** |
|---|---|---|
| 152개 전체 | 60.4% | 74.7% |
| domain 예측 필터링 | 69.5% | 85.5% |
| top-1 | 71.1% | 85.7% |
| top-5 | 74.0% | **88.5%** |
| domain 필터링 (oracle) | 77.2% | **96.5%** |

**후보를 좁힐수록 taxonomy 보정 효과가 커짐** — 152개 전체는 +14.3%p인데 domain 필터링(oracle)은
+19.3%p. 후보가 적을수록 남은 오답이 "같은 domain 안 근사-동의어" 쪽으로 더 쏠리기 때문(무관한
domain 오답은 애초에 필터링으로 이미 제거됐으니까).

**152개 전체를 제외한 나머지 4개 세팅은 taxonomy 보정 시 전부 논문 목표(85.6%)에 도달하거나
넘어섬** — domain 필터링(oracle)은 96.5%까지. 즉 taxonomy 중복 문제(같은 domain 안에서
`setLighting`/`setLightState`처럼 사실상 같은 뜻인 도구가 따로 등록된 것)만 해결되면, 이
파이프라인은 이미 논문 수준 이상일 가능성이 높음 — Tier-1 재현 갭의 실체는 대부분 taxonomy
문제였다는 뜻.

## 재사용

- `src/action_metrics.py` — `noise_aware_slu/src/action_metrics.py`에서 포팅 + 확장. Audio2Tool의
  `expected_tool_call` 필드가 STOP retriever와 동일한 canonical action 문법
  (`INTENT(SLOT="value", ...)`)이라 파서 재사용 가능. 단, Audio2Tool 쪽 gold 문자열은 작은따옴표
  (`deviceId='dryer_1'`)도 쓰길래 원래 큰따옴표(JSON)만 지원하던 `parse_string`에 작은따옴표 지원을
  추가함.
- `script/run_with_vllm.sh` — `reasoning_for_asr/scripts/run_with_vllm.sh` 포팅 (vLLM 서버 자동
  기동/종료 + `noise_aware_slu/scripts/run_with_vllm.sh`의 `no_proxy` 예외처리도 함께 포팅).

## 다음 단계 (미착수)

- [x] synthetic utterance 생성 + leakage filtering (152 tool, 2,911개 확보) — 완료
- [x] base Qwen3-0.6B에서 1 epoch scratch 학습 — 완료, recall@5 95.9%(oracle 96.9%에 근접),
      real retriever end-to-end Acc=EM 74.6%(152개 전체 64.4%보다 확실히 나음)
- [x] Tier-2로 확장 (152개 전체/domain 필터링/실제 retriever/oracle top-5 전부) — 완료
- [x] 프롬프트 업데이트("발화에서 인자 추출" 지시문) 반영해서 Tier-1/Tier-2 8개 세팅(152개 전체/
      domain 필터링/oracle top-5/실제 retriever) 재실행 — 완료. top-1/top-3(Tier-1)은 아직
      이전 프롬프트 결과로 남아있음
- [ ] Tier-2 EM/F1이 논문 목표보다 4~6배 높게 나오는 원인 파악 (Acc는 비슷한 범위) — 채점 기준
      자체가 다를 가능성(예: slot 값 정규화 관대함), Tier-1의 "채점 정책 오해" 패턴 재확인 필요
- [ ] Tier-1 전용 프롬프트(인자 추출 지시문 제외)로 재실행해서 60.4%/74.7%(현재)가
      64.4%/80.7%(구버전 프롬프트) 수준으로 회복되는지 확인
- [ ] recall@5(95.9%)와 downstream Acc(74.0%) 사이 ~22%p 갭 원인 분석 — Tier-2에서는 이 갭의
      대부분이 "tool은 맞혔는데 인자가 틀림"으로 확인됐고 지시문 추가로 완화됐음; Tier-1은 인자가
      없어서 이 설명이 안 통함, 별도 원인 규명 필요
- [x] retriever로 domain 예측해서 필터링(oracle 아님) + top-1 실험 — 완료. 예측 필터링이
      152개 전체와 oracle domain 필터링 사이에 위치, Tier2(인자 있음)에서는 top-1보다 안전함
      (top-1은 후보가 1개뿐이라 모델이 자주 거부 — parse failure 5.9%로 전체 세팅 중 최다)
- [ ] STOP 체크포인트(`qwen3-0.6b_depth0.1_seed44_5ep/epoch5`)에서 warm-start한 버전과 scratch
      1epoch 버전 비교, epoch 수 늘려서 추가 개선 여지 확인
- [ ] tools_registry.csv의 도메인 라벨 버그(64건) 수정
- [ ] Whisper-large-v3 cascade 붙이기 (`whisperv3 + Qwen 8B`, 78.1% 목표)
- [ ] (검토 중) domain 하나씩 빼고 학습해서 unseen domain 일반화 능력 별도 측정
- [ ] 8개 tier(~16,843개 라벨) 실 utterance도 train/dev/test로 나눠서 synthetic 데이터와 비교 —
      "논문 재현용 zero-shot 수치"와 겹치지 않게 설계 필요
