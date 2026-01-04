# Lecture 5: LLM Tuning (Preference Tuning & RLHF)

# Materials

- [CME 295](https://cme295.stanford.edu/syllabus/)
- [slide](https://cme295.stanford.edu/slides/fall25-cme295-lecture5.pdf)
- [video](https://www.youtube.com/watch?v=PmW_TMQ3l0I&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=5)

# Table of Contents

- [Lecture 5: LLM Tuning (Preference Tuning \& RLHF)](#lecture-5-llm-tuning-preference-tuning--rlhf)
- [Materials](#materials)
- [Table of Contents](#table-of-contents)
- [강의 개요](#강의-개요)
  - [강의 목표](#강의-목표)
  - [주요 학습 내용](#주요-학습-내용)
- [1. LLM Training Pipeline 복습](#1-llm-training-pipeline-복습)
  - [1.1. Stage 1: Pretraining](#11-stage-1-pretraining)
  - [1.2. Stage 2: Supervised Fine-tuning (SFT)](#12-stage-2-supervised-fine-tuning-sft)
  - [1.3. Stage 3: Preference Tuning (이번 강의)](#13-stage-3-preference-tuning-이번-강의)
- [2. Preference Tuning이 필요한 이유](#2-preference-tuning이-필요한-이유)
  - [2.1. 문제 상황](#21-문제-상황)
  - [2.2. Preference Pair 개념](#22-preference-pair-개념)
  - [2.3. SFT vs Preference Tuning](#23-sft-vs-preference-tuning)
    - [이유 1: 데이터 수집이 더 쉬움](#이유-1-데이터-수집이-더-쉬움)
    - [이유 2: Prompt Distribution 관리](#이유-2-prompt-distribution-관리)
    - [이유 3: Negative Signal](#이유-3-negative-signal)
- [3. Preference Data Collection](#3-preference-data-collection)
  - [3.1. 데이터 수집 방법](#31-데이터-수집-방법)
  - [3.2. Pointwise vs Pairwise vs Listwise](#32-pointwise-vs-pairwise-vs-listwise)
    - [Pointwise (개별 점수)](#pointwise-개별-점수)
    - [Pairwise (쌍 비교) ⭐ 가장 일반적](#pairwise-쌍-비교--가장-일반적)
    - [Listwise (순위 매기기)](#listwise-순위-매기기)
  - [3.3. Preference Pair 생성](#33-preference-pair-생성)
  - [3.4. 평가 방법](#34-평가-방법)
    - [1. Human Ratings (인간 평가)](#1-human-ratings-인간-평가)
    - [2. LLM-as-a-Judge](#2-llm-as-a-judge)
    - [3. Rule-based Metrics](#3-rule-based-metrics)
    - [4. Nuanced Scale (세밀한 척도)](#4-nuanced-scale-세밀한-척도)
- [4. RLHF (Reinforcement Learning from Human Feedback)](#4-rlhf-reinforcement-learning-from-human-feedback)
  - [4.1. RL 기초](#41-rl-기초)
  - [4.2. LLM에서의 RL 개념](#42-llm에서의-rl-개념)
  - [4.3. RLHF 개요](#43-rlhf-개요)
- [5. Stage 1: Reward Model Training](#5-stage-1-reward-model-training)
  - [5.1. Reward Model이란?](#51-reward-model이란)
  - [5.2. Bradley-Terry Formulation](#52-bradley-terry-formulation)
  - [5.3. Loss Function 유도](#53-loss-function-유도)
  - [5.4. Reward Model 구현](#54-reward-model-구현)
    - [옵션 1: Decoder-only LLM + Value Head](#옵션-1-decoder-only-llm--value-head)
    - [옵션 2: Encoder-only (BERT) + Classifier](#옵션-2-encoder-only-bert--classifier)
  - [5.5. Reward Model 학습](#55-reward-model-학습)
- [6. Stage 2: RL Fine-tuning](#6-stage-2-rl-fine-tuning)
  - [6.1. RL Fine-tuning 개요](#61-rl-fine-tuning-개요)
  - [6.2. 왜 Base Model에서 멀어지지 않아야 하는가?](#62-왜-base-model에서-멀어지지-않아야-하는가)
  - [6.3. Reward Hacking](#63-reward-hacking)
- [7. PPO (Proximal Policy Optimization)](#7-ppo-proximal-policy-optimization)
  - [7.1. PPO란?](#71-ppo란)
  - [7.2. Advantage와 Value Function](#72-advantage와-value-function)
  - [7.3. PPO-Clip](#73-ppo-clip)
  - [7.4. PPO-KL](#74-ppo-kl)
  - [7.5. PPO 구현](#75-ppo-구현)
- [8. DPO (Direct Preference Optimization)](#8-dpo-direct-preference-optimization)
  - [8.1. DPO의 등장 배경](#81-dpo의-등장-배경)
  - [8.2. DPO 핵심 아이디어](#82-dpo-핵심-아이디어)
  - [8.3. DPO vs RLHF](#83-dpo-vs-rlhf)
- [9. 실전 고려사항](#9-실전-고려사항)
  - [9.1. 데이터 품질](#91-데이터-품질)
  - [9.2. Hyperparameter 선택](#92-hyperparameter-선택)
  - [9.3. 평가 방법](#93-평가-방법)
- [10. 요약](#10-요약)
- [11. 중요 용어 정리](#11-중요-용어-정리)

---

# 강의 개요

## 강의 목표

이번 강의에서는 LLM을 인간의 선호도에 맞게 정렬(align)하는 방법을 학습합니다.

**학습 목표:**
- Preference Tuning의 필요성 이해
- Preference Data 수집 방법 학습
- RLHF (Reinforcement Learning from Human Feedback) 과정 이해
- Reward Model 학습 방법
- PPO (Proximal Policy Optimization) 알고리즘
- DPO (Direct Preference Optimization) 소개

## 주요 학습 내용

**1. Preference Tuning**
- SFT의 한계
- Preference Pair 개념
- 데이터 수집 방법

**2. RLHF**
- Stage 1: Reward Model Training
- Stage 2: RL Fine-tuning
- Bradley-Terry Formulation

**3. PPO Algorithm**
- Advantage와 Value Function
- PPO-Clip
- PPO-KL

**4. 실전 응용**
- DPO (간소화된 방법)
- 데이터 품질 관리
- 평가 방법

---

# 1. LLM Training Pipeline 복습

**전체 LLM Training 과정:**

```
Stage 1: Pretraining
┌────────────────────────────────┐
│ 목표: 언어 구조 학습           │
│ 데이터: 인터넷의 모든 텍스트   │
│ 작업: Next Token Prediction    │
│ 결과: Base Model               │
│ → 훌륭한 autocompleter         │
│ → 하지만 유용한 assistant 아님 │
└────────────────────────────────┘
              ↓
Stage 2: Supervised Fine-tuning (SFT)
┌────────────────────────────────┐
│ 목표: 특정 작업 수행 학습       │
│ 데이터: Instruction-Response 쌍│
│ 작업: 행동 패턴 학습            │
│ 결과: Instruction Model        │
│ → 지시사항을 따를 수 있음      │
│ → 하지만 톤/안전성 부족 가능   │
└────────────────────────────────┘
              ↓
Stage 3: Preference Tuning (이번 강의!)
┌────────────────────────────────┐
│ 목표: 인간 선호도에 맞게 정렬   │
│ 데이터: Preference Pairs       │
│ 작업: 선호하는 응답 생성        │
│ 결과: Aligned Model            │
│ → 유용하고 (Helpful)           │
│ → 정직하고 (Honest)            │
│ → 무해함 (Harmless)            │
└────────────────────────────────┘
```

## 1.1. Stage 1: Pretraining

**특징:**
- 매우 비쌈 (수백만 ~ 수억 달러)
- 오래 걸림 (수주 ~ 수개월)
- 대규모 데이터 (수조 tokens)
- 최적화 기법: Data Parallelism, ZeRO, Model Parallelism

**결과:**
```
입력: "The cat sat on"
출력: "the mat" (next token prediction)

→ 언어 구조는 이해하지만
→ 도움이 되는 assistant는 아님
```

## 1.2. Stage 2: Supervised Fine-tuning (SFT)

**특징:**
- 고품질 데이터셋 (수만 examples)
- 특정 작업에 맞게 조정
- LoRA 같은 효율적 방법 사용 가능

**결과:**
```
입력: "파이썬에서 리스트 정렬하는 법?"
출력: "sort() 메서드나 sorted() 함수를 사용하세요..."

→ 지시사항을 따를 수 있음
→ 하지만 항상 이상적이지는 않음
```

## 1.3. Stage 3: Preference Tuning (이번 강의)

**목표:**

모델을 인간의 선호도에 맞게 정렬 (Alignment)

**예시:**

```
입력: "테디베어와 할 수 있는 새로운 활동을 추천해주세요."

SFT Model 응답:
"테디베어와는 시간을 별로 보내지 않는 것을 추천합니다."
→ 지시사항은 따르지만 도움이 안 됨!

Aligned Model 응답:
"물론이죠! 테디베어는 잠자리 친구일 뿐만 아니라
재미있는 활동의 좋은 친구가 될 수 있습니다:
1. 소풍 가기
2. 차 파티 하기
3. 이야기 만들기..."
→ 유용하고 긍정적!
```

---

# 2. Preference Tuning이 필요한 이유

## 2.1. 문제 상황

**SFT 모델의 한계:**

```python
# 예시 1: 도움이 안 되는 응답
Q: "내 테디베어를 세탁기에 넣어도 되나요?"
SFT Model: "테디베어는 보통 폴리에스터나 면으로 만들어집니다.
           세탁 시 주의가 필요한 소재입니다..."

원하는 응답: "라벨을 확인하세요. 세탁 가능 표시가 있다면
             저온으로 세탁하되, 가능하면 손세탁을 권장합니다."

# 예시 2: 무례한 응답
Q: "오늘 기분이 안 좋아요."
SFT Model: "그렇군요. 다음 질문이 있나요?"

원하는 응답: "힘든 하루를 보내고 계시는군요. 무슨 일이 있었나요?
             제가 도울 수 있는 것이 있을까요?"
```

## 2.2. Preference Pair 개념

**Preference Pair란?**

```
하나의 프롬프트에 대해 두 개의 응답:
- 선호하는 응답 (Chosen/Winner)
- 선호하지 않는 응답 (Rejected/Loser)

┌────────────────────────────────┐
│ Prompt (프롬프트)              │
│ "테디베어 활동 추천해주세요"    │
└────────────────────────────────┘
              ↓
     ┌────────┴────────┐
     ↓                  ↓
┌─────────────┐   ┌─────────────┐
│ Response A  │   │ Response B  │
│ (나쁨) ✗    │   │ (좋음) ✓    │
│ "시간 낭비" │   │ "소풍 가기" │
│  "별로"     │   │ "차 파티"   │
└─────────────┘   └─────────────┘
```

## 2.3. SFT vs Preference Tuning

**왜 SFT만으로는 부족한가?**

### 이유 1: 데이터 수집이 더 쉬움

```
SFT 데이터 수집:
작업: "완벽한 시를 처음부터 작성하세요"
난이도: ⭐⭐⭐⭐⭐ (매우 어려움)
시간: 오래 걸림
품질: 고품질 보장 어려움

Preference 데이터 수집:
작업: "두 시 중 어느 것이 더 나은가요?"
난이도: ⭐⭐ (훨씬 쉬움)
시간: 빠름
품질: 일관성 있음
```

### 이유 2: Prompt Distribution 관리

```
문제:
SFT 데이터에 특정 유형의 프롬프트가 너무 많으면
모델이 그 방향으로 편향됨

예시:
코드 생성 프롬프트: 60%
일반 대화: 20%
수학 문제: 20%

→ 모델이 모든 것을 코드로 답하려 함!

Preference Tuning:
- 프롬프트 분포를 신경 쓸 필요 없음
- "어떤 응답이 더 나은가?"만 판단
- 더 유연함
```

### 이유 3: Negative Signal

```
SFT:
- "이렇게 해라" (positive signal만)
- "이렇게 하지 마라"를 학습 못함

Preference Tuning:
- "이것이 저것보다 좋다" (positive)
- "이것은 나쁘다" (negative) ← 중요!
- 모델이 피해야 할 것을 학습
```

**비교표:**

| 측면 | SFT | Preference Tuning |
|------|-----|-------------------|
| 데이터 생성 | 어려움 (완벽한 응답 작성) | 쉬움 (비교만 하면 됨) |
| 데이터 품질 | 매우 높아야 함 | 비교만 정확하면 됨 |
| Prompt 분포 | 매우 중요 (편향 위험) | 덜 중요 |
| Negative signal | 없음 | 있음 |
| 데이터 규모 | 수만 examples | 수만 comparisons |
| 사용 시기 | 기본 행동 학습 | 선호도 정렬 |

---

# 3. Preference Data Collection

## 3.1. 데이터 수집 방법

**Preference Data의 기본 구조:**

```
{
  "prompt": "짧은 시를 써주세요.",
  "response_a": "장미는 빨갛고\n하늘은 파랗다\n...",
  "response_b": "달빛이 내리면\n그대 생각에...",
  "preferred": "response_b"  # A or B
}
```

## 3.2. Pointwise vs Pairwise vs Listwise

### Pointwise (개별 점수)

```
작업: 각 응답에 점수 부여

시 A: 점수는? _____ (0-10)
시 B: 점수는? _____ (0-10)
시 C: 점수는? _____ (0-10)

문제:
- 점수 scale이 애매함
- "7점"이 정확히 무엇을 의미?
- 사람마다 기준이 다름
- 일관성 유지 어려움

예시:
평가자 A: 시 X는 7점
평가자 B: 시 X는 5점
→ 동일한 시, 다른 점수!
```

### Pairwise (쌍 비교) ⭐ 가장 일반적

```
작업: 두 응답 중 어느 것이 더 나은가?

시 A vs 시 B: [ ] A가 더 좋음  [✓] B가 더 좋음

장점:
- 매우 직관적
- 일관성 있음
- scale 애매함 없음
- 빠르게 판단 가능

단점:
- 여러 응답 비교 시 O(n²) 비교 필요
```

### Listwise (순위 매기기)

```
작업: 여러 응답의 순위 매기기

시 A, B, C, D, E가 있습니다.
순위를 매기세요:
1위: ___
2위: ___
3위: ___
4위: ___
5위: ___

장점:
- 한 번에 여러 응답 비교
- 더 효율적

단점:
- Pointwise보다 어려움
- 순위 매기기가 복잡할 수 있음
- 일관성 유지가 더 어려움
```

**실전 선택:**

```
대부분의 RLHF 시스템: Pairwise

이유:
- 가장 단순하고 명확
- 높은 품질의 데이터
- 충분히 효과적
```

## 3.3. Preference Pair 생성

**방법 1: 모델로 생성 + 인간 평가**

```python
def generate_preference_pair(prompt, model, temperature=0.8):
    """
    하나의 프롬프트로 두 개의 다른 응답 생성

    Args:
        prompt: 질문/지시사항
        model: LLM (SFT 완료된 모델)
        temperature: > 0 (다양한 응답 위해)

    Returns:
        (response_1, response_2)
    """
    # 같은 프롬프트로 두 번 생성 (temperature > 0이므로 다른 결과)
    response_1 = model.generate(prompt, temperature=temperature)
    response_2 = model.generate(prompt, temperature=temperature)

    return response_1, response_2

# 예시
prompt = "건강한 아침 식사를 추천해주세요."

# 생성
response_1 = model.generate(prompt, temperature=0.8)
# → "계란, 토스트, 과일을 추천합니다..."

response_2 = model.generate(prompt, temperature=0.8)
# → "라면 한 그릇이면 충분합니다..."

# 인간 평가
print(f"Response 1: {response_1}")
print(f"Response 2: {response_2}")
print("Which is better? [1/2]:")
choice = input()  # Human rates: 1

preference_pair = {
    "prompt": prompt,
    "chosen": response_1,
    "rejected": response_2
}
```

**방법 2: 로그에서 Bad Response 찾아 Rewrite**

```python
# 실제 서비스 로그에서
bad_responses = find_bad_responses_from_logs(
    criteria={
        "user_feedback": "negative",
        "safety_score": < 0.5,
        "helpfulness": < 0.3
    }
)

for bad_response in bad_responses:
    # 인간이 좋은 버전으로 다시 작성
    good_response = human_rewrite(bad_response)

    preference_pair = {
        "prompt": bad_response["prompt"],
        "chosen": good_response,
        "rejected": bad_response["text"]
    }
```

**프롬프트 출처:**

```python
# 1. 사용자 로그
prompts = get_user_queries_from_logs(
    sample_size=10000,
    diverse=True  # 다양한 주제
)

# 2. 인공적으로 생성
prompts = [
    "파이썬 코드 작성: ...",
    "이메일 작성: ...",
    "수학 문제 풀이: ...",
    "창의적 글쓰기: ...",
]

# 원칙: 실제 사용자 분포를 따라야 함
```

## 3.4. 평가 방법

### 1. Human Ratings (인간 평가)

```
가장 정확하지만 비쌈

과정:
1. 평가자 모집
2. 평가 가이드라인 제공
3. 응답 쌍 제시
4. 선택: "A가 더 좋음" / "B가 더 좋음"

주의사항:
- 명확한 가이드라인 필요
- 주관적 작업 최소화
- 여러 평가자로 검증
- 품질 관리 중요

예시 가이드라인:
"다음 기준으로 평가하세요:
1. 유용성: 질문에 직접 답하는가?
2. 정확성: 정보가 정확한가?
3. 안전성: 유해한 내용이 없는가?
4. 친절함: 톤이 적절한가?"
```

### 2. LLM-as-a-Judge

```python
def llm_judge(prompt, response_a, response_b):
    """
    LLM을 사용하여 두 응답 비교
    """
    judge_prompt = f"""
    질문: {prompt}

    답변 A: {response_a}
    답변 B: {response_b}

    어느 답변이 더 유용하고, 정확하고, 안전한가요?
    다음 기준으로 평가하세요:
    1. 유용성
    2. 정확성
    3. 안전성

    답변: [A/B]
    이유: ...
    """

    result = judge_model.generate(judge_prompt)
    return parse_result(result)  # 'A' or 'B'

# 장점:
# - 빠름
# - 저렴
# - Scalable

# 단점:
# - 인간보다 정확도 낮을 수 있음
# - Judge 모델의 편향 가능
# - 주관적 판단에 약함
```

### 3. Rule-based Metrics

```python
# BLEU, ROUGE 등 (잘 사용 안 됨)
def rule_based_eval(response_a, response_b, reference):
    """
    Rule-based metrics (덜 사용됨)
    """
    bleu_a = calculate_bleu(response_a, reference)
    bleu_b = calculate_bleu(response_b, reference)

    if bleu_a > bleu_b:
        return "A"
    else:
        return "B"

# 문제:
# - BLEU/ROUGE는 표면적 유사도만 측정
# - 의미적 품질 측정 못함
# - 현대 LLM에는 부적합
```

### 4. Nuanced Scale (세밀한 척도)

```
Binary (기본):
[ ] A가 더 좋음
[ ] B가 더 좋음

Nuanced (선택적):
[ ] A가 훨씬 더 좋음
[ ] A가 약간 더 좋음
[ ] 비슷함
[ ] B가 약간 더 좋음
[ ] B가 훨씬 더 좋음

장점:
- 더 세밀한 정보

단점:
- 복잡함
- 주관적
- 일관성 유지 어려움

→ 대부분 Binary 사용
```

**실전 권장:**

```
규모별 선택:

소규모 (< 1K pairs):
- Human rating (high quality)

중규모 (1K-10K):
- Human + LLM-as-judge (mix)

대규모 (10K+):
- LLM-as-judge (scalability)
- + Human sampling (quality check)
```

---

# 4. RLHF (Reinforcement Learning from Human Feedback)

## 4.1. RL 기초

**Reinforcement Learning의 기본 개념:**

```
RL의 구성 요소:

┌─────────┐
│  Agent  │ ← 학습하는 주체
│         │   (여기서는 LLM)
└────┬────┘
     │
     │ Action (행동)
     ↓
┌─────────────┐
│Environment  │
│  (환경)     │
└─────────────┘
     │
     │ Reward (보상)
     ↓
   높은 reward 받는 action을 학습
```

**핵심 용어:**

```
State (상태) - S_t:
현재 상황
예: 체스판의 현재 배치

Action (행동) - A_t:
취할 수 있는 행동
예: 체스 말 이동

Policy (정책) - π(a|s):
상태 s에서 행동 a를 선택할 확률
예: 이 상황에서 어떤 수를 둘 것인가?

Reward (보상) - R_t:
행동에 대한 피드백
예: 체스에서 승리 = +1, 패배 = -1
```

**RL의 목표:**

```
π*(a|s) = arg max E[Σ R_t]

최적의 정책 π*를 찾는 것:
- 누적 reward를 최대화하는 행동 선택
```

## 4.2. LLM에서의 RL 개념

**RL 개념을 LLM에 매핑:**

```
┌──────────────────┬─────────────────────────────┐
│ RL 개념          │ LLM에서의 대응               │
├──────────────────┼─────────────────────────────┤
│ Agent            │ LLM (학습 중인 모델)         │
│ Environment      │ Vocabulary (토큰 집합)       │
│ State (S_t)      │ 현재까지의 입력 토큰들       │
│ Action (A_t)     │ 다음 토큰 선택               │
│ Policy π(a|s)    │ P(next_token | context)     │
│ Reward (R)       │ Reward Model 점수            │
└──────────────────┴─────────────────────────────┘
```

**구체적인 예시:**

```python
# State: 현재까지의 텍스트
state = "테디베어와 할 수 있는"

# Action: 다음 토큰 선택
possible_actions = ["활동은", "놀이는", "게임은", ...]

# Policy: LLM의 출력 확률 분포
policy_output = model(state)
# {
#   "활동은": 0.6,
#   "놀이는": 0.3,
#   "게임은": 0.1
# }

# 토큰 선택 (sampling)
next_token = sample(policy_output)  # "활동은"

# 새로운 state
new_state = state + " " + next_token
# "테디베어와 할 수 있는 활동은"

# 이를 반복하여 전체 응답 생성
# → completion/rollout이라고 부름

# Reward: 전체 응답에 대한 평가
full_response = "테디베어와 할 수 있는 활동은 소풍, 차 파티..."
reward = reward_model(prompt, full_response)  # 예: 2.5
```

**시각화:**

```
┌──────────────────────────────────────────────┐
│ LLM (Agent)                                  │
└──────────────┬───────────────────────────────┘
               │
     State: "테디베어"
               │
               ↓
     Policy: P(next | "테디베어")
               │
               ↓
     Action: "와" (token 선택)
               │
               ↓
     State: "테디베어와"
               │
               ↓
     ... (반복) ...
               │
               ↓
     Full response: "테디베어와 소풍..."
               │
               ↓
     ┌─────────────────┐
     │ Reward Model    │
     │ Score: +2.5     │
     └─────────────────┘
               │
               ↓
     LLM 파라미터 업데이트
     (높은 reward → 이런 응답 더 생성)
```

## 4.3. RLHF 개요

**RLHF = Reinforcement Learning from Human Feedback**

```
"Human Feedback"의 의미:
- Preference pairs가 인간이 평가한 것
- 인간의 선호도를 반영

RLAIF = RL from AI Feedback:
- Preference pairs가 AI가 평가한 것
- (예: LLM-as-judge)
```

**RLHF의 2단계:**

```
Stage 1: Reward Model Training
┌─────────────────────────────────┐
│ Input: Preference Pairs         │
│ - (prompt, chosen, rejected)    │
│                                 │
│ Goal: Good vs Bad 구분 학습     │
│                                 │
│ Output: Reward Model            │
│ - r(prompt, response) → score  │
└─────────────────────────────────┘
              ↓
Stage 2: RL Fine-tuning
┌─────────────────────────────────┐
│ Input: Prompts                  │
│ Goal: High reward 생성 학습     │
│                                 │
│ Process:                        │
│ 1. LLM이 응답 생성              │
│ 2. Reward Model이 점수 부여     │
│ 3. LLM 파라미터 업데이트        │
│                                 │
│ Output: Aligned LLM             │
└─────────────────────────────────┘
```

---

# 5. Stage 1: Reward Model Training

## 5.1. Reward Model이란?

**Reward Model의 역할:**

```
Reward Model (RM):
입력: (prompt, response)
출력: scalar score (얼마나 좋은가?)

┌─────────────────────────────────┐
│ Reward Model                    │
│                                 │
│ Input: "활동 추천" + "소풍 가기"│
│ Output: +2.5 (좋은 응답!)       │
│                                 │
│ Input: "활동 추천" + "시간 낭비"│
│ Output: -1.2 (나쁜 응답!)       │
└─────────────────────────────────┘
```

**예시:**

```python
# Reward Model 사용
reward_model = RewardModel.load("trained_rm")

# Good response
score_good = reward_model(
    prompt="테디베어 활동 추천",
    response="소풍 가기, 차 파티 하기, 이야기 만들기..."
)
# score_good = 2.8

# Bad response
score_bad = reward_model(
    prompt="테디베어 활동 추천",
    response="테디베어와는 시간을 보내지 마세요."
)
# score_bad = -1.5

# Reward Model의 목표:
# Good response → 높은 점수
# Bad response → 낮은 점수
```

## 5.2. Bradley-Terry Formulation

**핵심 공식:**

```
Bradley-Terry Model:

P(y_i > y_j | x) = exp(r(x, y_i)) / (exp(r(x, y_i)) + exp(r(x, y_j)))

= σ(r(x, y_i) - r(x, y_j))

여기서:
- x: prompt
- y_i: response i
- y_j: response j
- r(x, y): reward model의 점수
- σ: sigmoid 함수
- P(y_i > y_j | x): y_i가 y_j보다 좋을 확률
```

**Sigmoid 함수:**

```python
def sigmoid(x):
    return 1 / (1 + exp(-x))

# 그래프:
#   1.0 |        ╱────
#       |      ╱
#   0.5 |    ╱
#       |  ╱
#   0.0 |╱____
#       └──────────── x
#      -∞  0  +∞

# 특성:
# x → +∞: sigmoid(x) → 1
# x = 0:  sigmoid(x) = 0.5
# x → -∞: sigmoid(x) → 0
```

**직관적 이해:**

```
y_i가 y_j보다 좋은 경우:

r(x, y_i) = 2.0  (good response)
r(x, y_j) = -1.0 (bad response)

r_i - r_j = 2.0 - (-1.0) = 3.0

σ(3.0) ≈ 0.95

→ 95% 확률로 y_i가 더 좋다고 예측!


y_i와 y_j가 비슷한 경우:

r(x, y_i) = 1.0
r(x, y_j) = 0.9

r_i - r_j = 0.1

σ(0.1) ≈ 0.52

→ 52% 확률 (거의 비슷함)
```

## 5.3. Loss Function 유도

**목표: Preference data의 확률을 최대화**

```
데이터:
D = {(x, y_w, y_l)_1, (x, y_w, y_l)_2, ..., (x, y_w, y_l)_n}

여기서:
- x: prompt
- y_w: winner (chosen)
- y_l: loser (rejected)
```

**Maximum Likelihood Estimation:**

```
Step 1: 확률 곱
────────────────
P(D | θ) = ∏ P(y_w > y_l | x; θ)
           i=1 to n

= ∏ σ(r_θ(x, y_w) - r_θ(x, y_l))
  i=1 to n

목표: P(D | θ)를 최대화하는 θ 찾기


Step 2: Log 취하기
──────────────────
log P(D | θ) = Σ log σ(r_θ(x, y_w) - r_θ(x, y_l))
               i=1 to n

이유:
- 곱 → 합으로 변환 (수치 안정성)
- log는 단조 증가 → 최대화 목표 동일


Step 3: Loss Function
──────────────────────
Maximize: Σ log σ(r(x, y_w) - r(x, y_l))

↕ (equivalent)

Minimize: -Σ log σ(r(x, y_w) - r(x, y_l))
          = Σ -log σ(r(x, y_w) - r(x, y_l))

최종 Loss Function:
L = E[- log σ(r(x, y_w) - r(x, y_l))]
```

**수식 정리:**

```
Loss = E[- log σ(r_w - r_l)]

여기서:
- r_w = r(x, y_w): winner의 reward
- r_l = r(x, y_l): loser의 reward
- E[...]: 데이터셋 전체에 대한 기댓값 (평균)
```

**직관적 이해:**

```
Loss가 최소화되려면:

σ(r_w - r_l) → 1
↓
r_w - r_l → +∞
↓
r_w >> r_l

즉:
- Winner의 reward가 높아야 함
- Loser의 reward가 낮아야 함
- 차이가 클수록 loss 작아짐!
```

## 5.4. Reward Model 구현

**아키텍처 선택:**

### 옵션 1: Decoder-only LLM + Value Head

```python
class RewardModel(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        # Pre-trained LLM (보통 SFT 모델에서 초기화)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name
        )

        # Value head: scalar reward 출력
        self.value_head = nn.Linear(
            self.base_model.config.hidden_size,
            1  # Scalar output
        )

    def forward(self, input_ids, attention_mask):
        # LLM forward pass
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # 마지막 토큰의 hidden state
        last_hidden_state = outputs.hidden_states[-1]
        last_token_hidden = last_hidden_state[:, -1, :]

        # Scalar reward 예측
        reward = self.value_head(last_token_hidden)

        return reward  # (batch_size, 1)
```

### 옵션 2: Encoder-only (BERT) + Classifier

```python
class BERTRewardModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.value_head = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # BERT forward
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Reward score
        reward = self.value_head(cls_embedding)

        return reward
```

**실전 권장: Decoder-only LLM**

```
이유:
1. 최신 트렌드 (모든 것이 LLM)
2. SFT 모델에서 초기화 가능 → 전이 학습
3. 더 강력한 표현력
4. 일관성 있는 파이프라인
```

## 5.5. Reward Model 학습

**전체 Training 코드:**

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class RewardModelTrainer:
    def __init__(self, model_name, lr=1e-5):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = RewardModel(model_name)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def compute_loss(self, batch):
        """
        Bradley-Terry Loss 계산

        batch = {
            'prompt': [...],
            'chosen': [...],
            'rejected': [...]
        }
        """
        # Tokenize: prompt + chosen
        inputs_chosen = self.tokenizer(
            [p + c for p, c in zip(batch['prompt'], batch['chosen'])],
            return_tensors='pt',
            padding=True,
            truncation=True
        )

        # Tokenize: prompt + rejected
        inputs_rejected = self.tokenizer(
            [p + r for p, r in zip(batch['prompt'], batch['rejected'])],
            return_tensors='pt',
            padding=True,
            truncation=True
        )

        # Forward pass
        r_chosen = self.model(
            inputs_chosen['input_ids'],
            inputs_chosen['attention_mask']
        )  # (batch_size, 1)

        r_rejected = self.model(
            inputs_rejected['input_ids'],
            inputs_rejected['attention_mask']
        )  # (batch_size, 1)

        # Bradley-Terry Loss
        # L = -log σ(r_chosen - r_rejected)
        # = log(1 + exp(r_rejected - r_chosen))  (numerically stable)
        loss = nn.functional.softplus(r_rejected - r_chosen).mean()

        # Accuracy (for monitoring)
        acc = (r_chosen > r_rejected).float().mean()

        return loss, acc

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        total_acc = 0

        for batch in dataloader:
            self.optimizer.zero_grad()

            loss, acc = self.compute_loss(batch)
            loss.backward()

            # Gradient clipping (안정성)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)

        return avg_loss, avg_acc

    def evaluate(self, dataloader):
        """Validation"""
        self.model.eval()
        total_loss = 0
        total_acc = 0

        with torch.no_grad():
            for batch in dataloader:
                loss, acc = self.compute_loss(batch)
                total_loss += loss.item()
                total_acc += acc.item()

        return total_loss / len(dataloader), total_acc / len(dataloader)

# Training
trainer = RewardModelTrainer('gpt2')

for epoch in range(3):
    train_loss, train_acc = trainer.train_epoch(train_loader)
    val_loss, val_acc = trainer.evaluate(val_loader)

    print(f"Epoch {epoch}")
    print(f"  Train: loss={train_loss:.3f}, acc={train_acc:.3f}")
    print(f"  Val:   loss={val_loss:.3f}, acc={val_acc:.3f}")
```

**Loss Function 상세:**

```python
# Bradley-Terry Loss (두 가지 구현)

# 방법 1: 직접 구현 (수치 불안정할 수 있음)
def bradley_terry_loss_v1(r_chosen, r_rejected):
    """
    L = -log σ(r_chosen - r_rejected)
    """
    diff = r_chosen - r_rejected
    prob = torch.sigmoid(diff)
    loss = -torch.log(prob).mean()
    return loss

# 방법 2: 수치 안정성 버전 (권장)
def bradley_terry_loss_v2(r_chosen, r_rejected):
    """
    L = -log σ(r_chosen - r_rejected)

    수학적 변형:
    -log σ(x) = -log(1 / (1 + exp(-x)))
              = log(1 + exp(-x))
              = softplus(-x)

    -log σ(r_c - r_r) = -log σ(-(r_r - r_c))
                       = softplus(r_r - r_c)
    """
    loss = nn.functional.softplus(r_rejected - r_chosen).mean()
    return loss

# 왜 v2가 더 안정적인가?
"""
v1의 문제:
- sigmoid(x)가 0에 가까우면 log(0) → -∞
- 수치 오버플로우 가능

v2의 장점:
- softplus는 항상 안정적
- PyTorch가 내부적으로 최적화
- 추천 방법!
"""
```

**Reward Model 학습 팁:**

```python
# 1. Learning Rate
lr = 1e-5  # 작은 LR 사용 (LLM 기반이므로)

# 2. Batch Size
batch_size = 8  # GPU 메모리에 따라 조정

# 3. Epochs
epochs = 3  # 너무 많이 학습하면 overfitting

# 4. Gradient Clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 5. Monitoring
# - Loss가 감소하는지
# - Accuracy가 증가하는지 (chosen > rejected)
# - Validation loss도 함께 추적

# 6. Early Stopping
# Validation accuracy가 더 이상 증가하지 않으면 중단
```

**평가 지표:**

```python
def evaluate_reward_model(reward_model, test_data):
    """
    Reward Model 평가
    """
    correct = 0
    total = 0

    for prompt, chosen, rejected in test_data:
        # Reward 계산
        r_chosen = reward_model(prompt, chosen)
        r_rejected = reward_model(prompt, rejected)

        # Correct if chosen > rejected
        if r_chosen > r_rejected:
            correct += 1
        total += 1

    accuracy = correct / total
    return accuracy

# 좋은 Reward Model:
# - Test accuracy > 70%
# - Validation loss 안정적
# - 새로운 prompts에도 일반화
```

---

# 6. Stage 2: RL Fine-tuning

## 6.1. RL Fine-tuning 개요

**목표:**

```
Reward Model을 사용하여 LLM을 fine-tuning
→ 높은 reward를 받는 응답을 생성하도록 학습
```

**전체 프로세스:**

```
┌─────────────────────────────────────┐
│ 1. Prompt 입력                      │
│    "테디베어 활동 추천"              │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│ 2. LLM이 응답 생성 (Rollout)        │
│    π_θ: "소풍 가기, 차 파티..."     │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│ 3. Reward Model로 점수 계산         │
│    r(prompt, response) = +2.5       │
└────────────┬────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│ 4. RL 알고리즘으로 LLM 업데이트     │
│    높은 reward → 강화               │
│    낮은 reward → 약화               │
└─────────────────────────────────────┘
```

**핵심 공식:**

```
Objective:
maximize E[r(x, y)]
         y~π_θ(·|x)

여기서:
- x: prompt
- y: LLM이 생성한 response
- π_θ: LLM (파라미터 θ)
- r(x, y): Reward Model 점수

목표: 기댓값이 최대가 되도록 θ 조정
```

## 6.2. 왜 Base Model에서 멀어지지 않아야 하는가?

**문제: Reward Hacking**

```
Reward Model만 최대화하면:
→ LLM이 이상하게 변할 수 있음!

예시:
"!!!!!!!!!!!!!!!!" (느낌표 반복)
→ Reward Model이 높은 점수 줌 (버그)
→ LLM이 계속 느낌표만 생성
→ 의미 없는 출력!
```

**해결책: KL Divergence Penalty**

```
수정된 Objective:

maximize E[r(x, y)] - β * KL(π_θ || π_ref)
         y~π_θ

여기서:
- π_ref: Reference policy (SFT 모델, 고정)
- KL(π_θ || π_ref): KL divergence (거리 측정)
- β: penalty coefficient

의미:
- r(x, y): reward 높이기 (선호도 맞추기)
- KL penalty: π_θ가 π_ref에서 너무 멀어지지 않게
```

**KL Divergence란?**

```
KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
             x

두 확률 분포의 차이를 측정

예시:
P = [0.7, 0.2, 0.1]  (현재 모델)
Q = [0.6, 0.3, 0.1]  (base 모델)

→ KL이 작으면 비슷함
→ KL이 크면 많이 변했음
```

**시각화:**

```
┌────────────────────────────────────┐
│ β가 작을 때 (β = 0.01)             │
│ KL penalty 약함                    │
│ → Reward 최적화 우선               │
│ → Reward hacking 위험 ⚠            │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│ β가 클 때 (β = 1.0)                │
│ KL penalty 강함                    │
│ → Base model에 가까움              │
│ → 안전하지만 개선 제한됨           │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│ β가 적절할 때 (β = 0.1~0.2)       │
│ 균형 잡힘 ✓                        │
│ → Reward도 개선                    │
│ → Base model에서 크게 벗어나지 않음│
└────────────────────────────────────┘
```

## 6.3. Reward Hacking

**Reward Hacking이란?**

```
Reward Model의 약점을 악용하여
실제로는 나쁜 응답이지만 높은 점수를 받는 현상
```

**예시들:**

```python
# 예시 1: 반복
prompt = "짧은 시를 써주세요"
bad_response = "아름다운 아름다운 아름다운 아름다운..."
# Reward Model이 "아름다운"이라는 단어에 높은 점수 부여
# → LLM이 무한 반복 학습

# 예시 2: 길이
prompt = "파이썬 설명"
bad_response = "파이썬은 " + "..." * 1000  # 매우 긴 응답
# Reward Model이 긴 응답에 높은 점수 부여 (잘못 학습됨)
# → LLM이 불필요하게 긴 응답 생성

# 예시 3: 특정 단어
bad_response = "물론이죠! 정말! 확실히! ..."
# Reward Model이 긍정적 단어에 높은 점수 부여
# → LLM이 과도하게 긍정적 단어 사용
```

**방지 방법:**

```
1. KL Divergence Penalty 사용 ✓
   → Base model에서 크게 벗어나지 못하게

2. Reward Model 품질 개선
   → 더 많은 preference data
   → 다양한 edge cases 포함

3. Rule-based Constraints
   → 응답 길이 제한
   → 반복 패턴 감지 및 페널티

4. Ensemble Reward Models
   → 여러 Reward Model 사용
   → 평균 점수 사용
```

---

# 7. PPO (Proximal Policy Optimization)

## 7.1. PPO란?

**PPO = Proximal Policy Optimization**

```
"Proximal" = 가까운, 근접한
→ Policy update를 조금씩만 하자!

왜?
- 너무 큰 update: 학습 불안정
- 너무 작은 update: 학습 느림
- PPO: 적절한 크기로 update
```

**PPO의 핵심 아이디어:**

```
Old Policy (π_old):
이전 파라미터 θ_old

New Policy (π_new):
새로운 파라미터 θ

PPO 목표:
π_new가 π_old에서 너무 멀어지지 않게!
→ 안정적인 학습
```

## 7.2. Advantage와 Value Function

**Advantage Function**

```
A(s, a) = Q(s, a) - V(s)

여기서:
- Q(s, a): State-action value
  → "상태 s에서 행동 a를 하면 얼마나 좋은가?"

- V(s): State value
  → "상태 s에서 평균적으로 얼마나 좋은가?"

- A(s, a): Advantage
  → "행동 a가 평균보다 얼마나 좋은가?"
```

**직관적 이해:**

```
예시: 레스토랑 선택

State s: "배고픔"

Action a1: 이탈리안 레스토랑
Q(s, a1) = 8점

Action a2: 패스트푸드
Q(s, a2) = 5점

V(s) = (8 + 5) / 2 = 6.5점 (평균)

Advantage:
A(s, a1) = 8 - 6.5 = +1.5  (평균보다 좋음!)
A(s, a2) = 5 - 6.5 = -1.5  (평균보다 나쁨!)

→ a1을 더 자주 선택하도록 학습
```

**LLM에서의 Advantage:**

```python
# State: 현재까지의 텍스트
state = "테디베어와"

# Action: 다음 토큰
action = "소풍"  # vs "시간"

# Q value (Reward Model 사용)
full_response_1 = "테디베어와 소풍..."
q_1 = reward_model(prompt, full_response_1)  # 2.5

full_response_2 = "테디베어와 시간..."
q_2 = reward_model(prompt, full_response_2)  # -1.0

# V value (평균적인 reward)
v = expected_reward(state)  # 1.0

# Advantage
advantage_1 = q_1 - v = 2.5 - 1.0 = +1.5  (좋음!)
advantage_2 = q_2 - v = -1.0 - 1.0 = -2.0  (나쁨!)

→ "소풍" 토큰의 확률 증가
→ "시간" 토큰의 확률 감소
```

**Value Function 추정:**

```python
# Generalized Advantage Estimation (GAE)
def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    """
    Args:
        rewards: 각 step의 reward
        values: 각 step의 value 추정
        gamma: discount factor
        lam: GAE lambda

    Returns:
        advantages: 각 step의 advantage
    """
    advantages = []
    advantage = 0

    for t in reversed(range(len(rewards))):
        # TD error
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value - values[t]

        # GAE
        advantage = delta + gamma * lam * advantage
        advantages.insert(0, advantage)

    return advantages
```

## 7.3. PPO-Clip

**PPO-Clip Objective:**

```
L^CLIP(θ) = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]

여기서:
- r(θ) = π_θ(a|s) / π_old(a|s)  (probability ratio)
- A: Advantage
- ε: clip range (보통 0.1~0.2)
- clip(x, min, max): x를 [min, max] 범위로 제한
```

**Ratio r(θ):**

```python
# 예시
π_old(a|s) = 0.2  # 이전 모델의 확률
π_new(a|s) = 0.4  # 새로운 모델의 확률

r = π_new / π_old = 0.4 / 0.2 = 2.0

의미:
- r = 1.0: 확률 변화 없음
- r > 1.0: 확률 증가 (action을 더 자주 선택)
- r < 1.0: 확률 감소 (action을 덜 선택)
```

**Clipping의 역할:**

```
Case 1: Advantage > 0 (좋은 action)
────────────────────────────────────
r = 2.0 (확률이 2배 증가)
ε = 0.2

clip(r, 1-0.2, 1+0.2) = clip(2.0, 0.8, 1.2) = 1.2

→ r이 1.2로 제한됨
→ 확률이 너무 급격히 증가하지 않음!


Case 2: Advantage < 0 (나쁜 action)
────────────────────────────────────
r = 0.5 (확률이 절반으로 감소)
ε = 0.2

clip(r, 0.8, 1.2) = 0.8

→ r이 0.8로 제한됨
→ 확률이 너무 급격히 감소하지 않음!
```

**min(·, ·)의 역할:**

```python
# Advantage > 0일 때
objective = min(
    r * A,           # 원래 objective
    clip(r, 1-ε, 1+ε) * A  # clipped objective
)

# r > 1+ε (확률이 너무 많이 증가)
# → clip이 작용하여 update 제한
# → 안정적인 학습!

# Advantage < 0일 때
# → 마찬가지로 clip이 작용
# → 과도한 페널티 방지
```

**PPO-Clip 구현:**

```python
def ppo_clip_loss(
    log_probs_new,    # 새 모델의 log π_θ(a|s)
    log_probs_old,    # 옛 모델의 log π_old(a|s)
    advantages,       # Advantage 값들
    epsilon=0.2       # Clip 범위
):
    """
    PPO-Clip Loss 계산
    """
    # Ratio: r = π_new / π_old
    # = exp(log π_new - log π_old)
    ratio = torch.exp(log_probs_new - log_probs_old)

    # Unclipped objective
    obj_unclipped = ratio * advantages

    # Clipped objective
    ratio_clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    obj_clipped = ratio_clipped * advantages

    # PPO-Clip: min of the two
    loss = -torch.min(obj_unclipped, obj_clipped).mean()

    return loss

# 예시
log_probs_new = torch.tensor([-1.0, -2.0, -0.5])
log_probs_old = torch.tensor([-1.2, -1.8, -0.6])
advantages = torch.tensor([1.5, -0.8, 2.0])

loss = ppo_clip_loss(log_probs_new, log_probs_old, advantages)
```

## 7.4. PPO-KL

**PPO-KL = PPO with KL Penalty**

```
대안적 방법: Clipping 대신 KL divergence penalty 사용

Objective:
L^KL(θ) = E[π_θ(a|s) / π_old(a|s) * A] - β * KL(π_θ || π_old)

여기서:
- 첫 항: Reward 최대화
- 둘째 항: KL penalty (policy가 너무 많이 변하지 않게)
- β: penalty coefficient (adaptive)
```

**Adaptive β:**

```python
# β를 동적으로 조정
if KL > KL_target * 1.5:
    # KL이 너무 크면 β 증가
    β = β * 2
elif KL < KL_target / 1.5:
    # KL이 너무 작으면 β 감소
    β = β / 2

# 예시
KL_target = 0.01  # 목표 KL divergence
β = 0.1           # 초기 β
```

**PPO-Clip vs PPO-KL:**

```
┌──────────────┬─────────────────┬───────────────────┐
│              │ PPO-Clip        │ PPO-KL            │
├──────────────┼─────────────────┼───────────────────┤
│ 안정성       │ ⭐⭐⭐ (매우 안정)│ ⭐⭐ (안정)      │
│ 구현 복잡도  │ ⭐⭐ (간단)     │ ⭐⭐⭐ (복잡)    │
│ 하이퍼파라미터│ ε (고정)        │ β (adaptive)      │
│ 성능         │ ⭐⭐⭐          │ ⭐⭐⭐           │
│ 사용 빈도    │ ⭐⭐⭐ (많이 씀)│ ⭐⭐ (덜 씀)     │
└──────────────┴─────────────────┴───────────────────┘

실전 추천: PPO-Clip
- 더 간단
- 더 안정적
- 대부분의 RLHF에서 사용
```

## 7.5. PPO 구현

**전체 PPO Training Loop:**

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class PPOTrainer:
    def __init__(
        self,
        model_name,
        reward_model,
        ref_model,
        lr=1e-6,
        epsilon=0.2,
        beta=0.1
    ):
        # Policy model (학습할 모델)
        self.policy = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Reference model (고정)
        self.ref_model = ref_model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Reward model (고정)
        self.reward_model = reward_model
        self.reward_model.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Hyperparameters
        self.epsilon = epsilon  # PPO clip range
        self.beta = beta        # KL penalty coefficient

    def generate_rollout(self, prompts, max_length=128):
        """
        Step 1: LLM으로 응답 생성 (rollout)
        """
        self.policy.eval()

        responses = []
        log_probs_list = []

        for prompt in prompts:
            # Tokenize
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

            # Generate
            with torch.no_grad():
                output = self.policy.generate(
                    input_ids,
                    max_length=max_length,
                    do_sample=True,        # Sampling (not greedy)
                    temperature=0.7,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            # Response
            response_ids = output.sequences[0]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            responses.append(response)

            # Log probabilities
            log_probs = self._compute_log_probs(output.scores, response_ids)
            log_probs_list.append(log_probs)

        return responses, log_probs_list

    def compute_rewards(self, prompts, responses):
        """
        Step 2: Reward Model로 점수 계산
        """
        rewards = []

        for prompt, response in zip(prompts, responses):
            # Reward Model 평가
            reward = self.reward_model(prompt, response)

            # KL penalty 추가
            kl_penalty = self._compute_kl_penalty(prompt, response)

            # Final reward
            final_reward = reward - self.beta * kl_penalty
            rewards.append(final_reward)

        return rewards

    def _compute_kl_penalty(self, prompt, response):
        """
        KL divergence: KL(π || π_ref)
        """
        # Current policy
        log_probs_policy = self._get_log_probs(self.policy, prompt, response)

        # Reference policy
        with torch.no_grad():
            log_probs_ref = self._get_log_probs(self.ref_model, prompt, response)

        # KL(π || π_ref) = E[log π - log π_ref]
        kl = (log_probs_policy - log_probs_ref).sum()

        return kl

    def compute_advantages(self, rewards):
        """
        Step 3: Advantage 계산
        """
        # 간단한 버전: Reward를 그대로 사용
        # (실제로는 GAE 등 더 정교한 방법 사용)
        advantages = rewards

        # Normalize (선택적, 안정성 향상)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def ppo_update(self, prompts, responses, log_probs_old, advantages):
        """
        Step 4: PPO로 policy 업데이트
        """
        self.policy.train()

        # Forward pass: 새로운 log probs 계산
        log_probs_new = []
        for prompt, response in zip(prompts, responses):
            log_prob = self._get_log_probs(self.policy, prompt, response)
            log_probs_new.append(log_prob)

        log_probs_new = torch.stack(log_probs_new)
        log_probs_old = torch.stack(log_probs_old)
        advantages = torch.tensor(advantages)

        # PPO-Clip Loss
        ratio = torch.exp(log_probs_new - log_probs_old)
        obj_unclipped = ratio * advantages
        obj_clipped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        loss = -torch.min(obj_unclipped, obj_clipped).mean()

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def train_step(self, prompts):
        """
        전체 PPO training step
        """
        # 1. Generate responses (rollout)
        responses, log_probs_old = self.generate_rollout(prompts)

        # 2. Compute rewards
        rewards = self.compute_rewards(prompts, responses)

        # 3. Compute advantages
        advantages = self.compute_advantages(rewards)

        # 4. PPO update (여러 epoch)
        for _ in range(4):  # PPO는 여러 번 update
            loss = self.ppo_update(prompts, responses, log_probs_old, advantages)

        return loss, rewards

# Training
trainer = PPOTrainer(
    model_name='gpt2',
    reward_model=trained_reward_model,
    ref_model=reference_model
)

prompts = ["테디베어 활동 추천", "파이썬 코드 작성", ...]

for iteration in range(1000):
    loss, rewards = trainer.train_step(prompts)
    print(f"Iteration {iteration}: loss={loss:.3f}, avg_reward={sum(rewards)/len(rewards):.3f}")
```

**PPO 하이퍼파라미터:**

```python
# 중요한 하이퍼파라미터들
hyperparameters = {
    # Learning rate
    'lr': 1e-6,              # 매우 작게! (LLM이므로)

    # PPO-Clip epsilon
    'epsilon': 0.2,          # Clip range (0.1~0.3)

    # KL penalty
    'beta': 0.1,             # KL coefficient (0.01~0.5)

    # Training
    'batch_size': 32,        # Prompts per batch
    'ppo_epochs': 4,         # PPO update 반복 횟수
    'max_length': 128,       # Response 최대 길이

    # Generation
    'temperature': 0.7,      # Sampling temperature
    'top_p': 0.9,            # Nucleus sampling
}
```

---

# 8. DPO (Direct Preference Optimization)

## 8.1. DPO의 등장 배경

**RLHF의 문제점:**

```
RLHF는 복잡함:
1. Reward Model 따로 학습 (Stage 1)
2. RL로 LLM 학습 (Stage 2)

문제:
- 두 단계가 필요
- PPO 구현이 복잡
- 학습 불안정할 수 있음
- Reward Model을 별도로 유지해야 함
```

**DPO의 아이디어:**

```
"Reward Model을 명시적으로 학습하지 말고,
 직접 preference data로 LLM을 학습하자!"

RLHF:
Preference Data → Reward Model → RL Fine-tuning

DPO:
Preference Data → LLM (한 번에!)
```

## 8.2. DPO 핵심 아이디어

**수학적 유도:**

```
RLHF의 optimal policy:

π*(y|x) ∝ π_ref(y|x) * exp(r(x,y) / β)

여기서:
- π*: optimal policy
- π_ref: reference policy (SFT model)
- r(x,y): reward function
- β: temperature

이를 변형하면:
r(x,y) = β * log(π*(y|x) / π_ref(y|x)) + constant

즉, reward를 policy의 비율로 표현 가능!
```

**Bradley-Terry를 DPO로:**

```
Bradley-Terry (Reward Model 사용):
P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))

DPO (Reward 대신 Policy 사용):
P(y_w > y_l | x) = σ(β * log(π_θ(y_w|x) / π_ref(y_w|x))
                       - β * log(π_θ(y_l|x) / π_ref(y_l|x)))
```

**DPO Loss:**

```
L_DPO(θ) = -E[(x,y_w,y_l)~D] [log σ(β * log(π_θ(y_w|x) / π_ref(y_w|x))
                                       - β * log(π_θ(y_l|x) / π_ref(y_l|x)))]

간단히:
L_DPO = -E[log σ(score_chosen - score_rejected)]

여기서:
score_chosen = β * log(π_θ(y_chosen|x) / π_ref(y_chosen|x))
score_rejected = β * log(π_θ(y_rejected|x) / π_ref(y_rejected|x))
```

**DPO 구현:**

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class DPOTrainer:
    def __init__(self, model_name, ref_model, beta=0.1, lr=1e-6):
        # Policy model (학습할 모델)
        self.policy = AutoModelForCausalLM.from_pretrained(model_name)

        # Reference model (고정)
        self.ref_model = ref_model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Hyperparameters
        self.beta = beta
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def compute_loss(self, batch):
        """
        DPO Loss 계산

        batch = {
            'prompt': [...],
            'chosen': [...],
            'rejected': [...]
        }
        """
        prompts = batch['prompt']
        chosen = batch['chosen']
        rejected = batch['rejected']

        # Log probs: policy model
        log_probs_chosen_policy = self._get_log_probs(self.policy, prompts, chosen)
        log_probs_rejected_policy = self._get_log_probs(self.policy, prompts, rejected)

        # Log probs: reference model
        with torch.no_grad():
            log_probs_chosen_ref = self._get_log_probs(self.ref_model, prompts, chosen)
            log_probs_rejected_ref = self._get_log_probs(self.ref_model, prompts, rejected)

        # Score = β * log(π/π_ref)
        score_chosen = self.beta * (log_probs_chosen_policy - log_probs_chosen_ref)
        score_rejected = self.beta * (log_probs_rejected_policy - log_probs_rejected_ref)

        # DPO Loss: -log σ(score_chosen - score_rejected)
        loss = -nn.functional.logsigmoid(score_chosen - score_rejected).mean()

        # Accuracy (monitoring)
        acc = (score_chosen > score_rejected).float().mean()

        return loss, acc

    def _get_log_probs(self, model, prompts, responses):
        """
        Get log P(response | prompt) from model
        """
        log_probs = []

        for prompt, response in zip(prompts, responses):
            # Tokenize
            input_text = prompt + response
            input_ids = tokenizer.encode(input_text, return_tensors='pt')
            prompt_ids = tokenizer.encode(prompt, return_tensors='pt')

            # Forward
            with torch.set_grad_enabled(model.training):
                outputs = model(input_ids)
                logits = outputs.logits

            # Log probs of response tokens
            response_start = len(prompt_ids[0])
            response_logits = logits[0, response_start-1:-1, :]  # Shift by 1
            response_ids = input_ids[0, response_start:]

            # Log softmax
            log_probs_token = nn.functional.log_softmax(response_logits, dim=-1)

            # Select log probs of actual tokens
            log_prob = log_probs_token[range(len(response_ids)), response_ids].sum()
            log_probs.append(log_prob)

        return torch.stack(log_probs)

    def train_epoch(self, dataloader):
        """
        DPO Training Epoch
        """
        self.policy.train()
        total_loss = 0
        total_acc = 0

        for batch in dataloader:
            self.optimizer.zero_grad()

            loss, acc = self.compute_loss(batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()

        return total_loss / len(dataloader), total_acc / len(dataloader)

# Training
trainer = DPOTrainer(
    model_name='gpt2',
    ref_model=reference_model,
    beta=0.1
)

for epoch in range(3):
    train_loss, train_acc = trainer.train_epoch(train_loader)
    print(f"Epoch {epoch}: loss={train_loss:.3f}, acc={train_acc:.3f}")
```

## 8.3. DPO vs RLHF

**비교표:**

```
┌─────────────────┬──────────────────────┬─────────────────────┐
│                 │ RLHF (PPO)           │ DPO                 │
├─────────────────┼──────────────────────┼─────────────────────┤
│ 학습 단계       │ 2 stages             │ 1 stage             │
│                 │ (RM + RL)            │ (Direct)            │
├─────────────────┼──────────────────────┼─────────────────────┤
│ Reward Model    │ 필요 ✓               │ 불필요 ✗            │
├─────────────────┼──────────────────────┼─────────────────────┤
│ 구현 복잡도     │ ⭐⭐⭐⭐ (복잡)      │ ⭐⭐ (간단)         │
├─────────────────┼──────────────────────┼─────────────────────┤
│ 학습 안정성     │ ⭐⭐ (불안정할 수)   │ ⭐⭐⭐ (안정적)     │
├─────────────────┼──────────────────────┼─────────────────────┤
│ 계산 비용       │ 높음 (RM + policy)   │ 낮음 (policy만)     │
├─────────────────┼──────────────────────┼─────────────────────┤
│ 메모리 사용     │ 높음 (2개 모델)      │ 낮음 (1개 모델)     │
├─────────────────┼──────────────────────┼─────────────────────┤
│ 하이퍼파라미터  │ 많음 (lr, ε, β, ...) │ 적음 (lr, β)        │
├─────────────────┼──────────────────────┼─────────────────────┤
│ Reward 해석     │ 가능 (RM 존재)       │ 어려움 (암묵적)     │
├─────────────────┼──────────────────────┼─────────────────────┤
│ 유연성          │ ⭐⭐⭐ (높음)        │ ⭐⭐ (제한적)       │
│                 │ (RM 재사용 가능)     │                     │
├─────────────────┼──────────────────────┼─────────────────────┤
│ 성능            │ ⭐⭐⭐              │ ⭐⭐⭐              │
│                 │ (비슷함)             │ (비슷함)            │
├─────────────────┼──────────────────────┼─────────────────────┤
│ 최근 트렌드     │ 전통적 방법          │ 인기 상승 ↗          │
└─────────────────┴──────────────────────┴─────────────────────┘
```

**언제 무엇을 사용할까?**

```
DPO 추천:
✓ 빠른 프로토타이핑
✓ 제한된 계산 자원
✓ 간단한 preference tuning
✓ 안정적인 학습 원할 때

RLHF 추천:
✓ 복잡한 reward function 필요
✓ Reward를 명시적으로 분석하고 싶을 때
✓ 여러 policy에 동일 RM 재사용
✓ 더 정교한 제어 원할 때

최근 트렌드:
DPO가 점점 더 인기 (간단함 + 효과적)
```

---

# 9. 실전 고려사항

## 9.1. 데이터 품질

**Preference Data 품질이 가장 중요!**

```
Garbage In, Garbage Out

나쁜 데이터 → 나쁜 모델
좋은 데이터 → 좋은 모델
```

**데이터 품질 체크리스트:**

```python
def check_data_quality(preference_dataset):
    """
    Preference data 품질 검사
    """
    issues = []

    # 1. 다양성 체크
    prompt_types = categorize_prompts(preference_dataset['prompts'])
    if len(prompt_types) < 5:
        issues.append("Prompt diversity too low")

    # 2. 응답 길이 체크
    lengths = [len(r) for r in preference_dataset['chosen']]
    if max(lengths) > 10 * min(lengths):
        issues.append("Response length variance too high")

    # 3. Agreement 체크 (여러 평가자)
    if 'num_raters' in preference_dataset:
        agreement = calculate_agreement(preference_dataset)
        if agreement < 0.7:
            issues.append("Low inter-rater agreement")

    # 4. Preference strength 체크
    if 'preference_strength' in preference_dataset:
        weak_preferences = sum(s < 0.6 for s in preference_dataset['preference_strength'])
        if weak_preferences / len(preference_dataset) > 0.3:
            issues.append("Too many weak preferences")

    return issues
```

**데이터 수집 Best Practices:**

```
1. 명확한 가이드라인
   - 평가자에게 구체적 기준 제공
   - 예시 포함
   - 일관성 유지

2. 다양한 프롬프트
   - 여러 도메인 (코드, 글쓰기, 수학, ...)
   - 다양한 난이도
   - Edge cases 포함

3. 품질 관리
   - 여러 평가자 사용
   - Agreement 측정
   - 주기적 재평가

4. 규모
   - 최소: 10K pairs
   - 권장: 50K-100K pairs
   - 대규모: 1M+ pairs
```

## 9.2. Hyperparameter 선택

**중요한 하이퍼파라미터:**

```python
# DPO/RLHF 공통
hyperparameters = {
    # Learning rate
    'lr': 1e-6,              # 시작: 1e-6, 범위: 1e-7 ~ 1e-5

    # KL penalty (매우 중요!)
    'beta': 0.1,             # 시작: 0.1, 범위: 0.01 ~ 0.5
    # β 작음: 더 큰 변화 (reward 우선)
    # β 큼: 더 작은 변화 (안정성 우선)

    # Batch size
    'batch_size': 32,        # GPU 메모리에 따라

    # Epochs
    'epochs': 3,             # 너무 많으면 overfitting

    # Gradient clipping
    'max_grad_norm': 1.0,    # 안정성 위해 필수!
}

# PPO 추가
ppo_params = {
    'epsilon': 0.2,          # Clip range: 0.1 ~ 0.3
    'ppo_epochs': 4,         # PPO update 반복: 3 ~ 5
    'value_loss_coef': 0.5,  # Value loss coefficient
}
```

**β (KL penalty) 튜닝 가이드:**

```
β 너무 작음 (β < 0.01):
❌ 모델이 base model에서 너무 멀어짐
❌ Reward hacking 위험
❌ 불안정한 학습

β 적절함 (β = 0.05 ~ 0.2):
✅ 균형 잡힌 학습
✅ Base model 근처 유지
✅ 안정적 개선

β 너무 큼 (β > 0.5):
❌ 모델이 거의 변하지 않음
❌ Reward 개선 제한적
❌ 학습 느림

권장 시작값:
DPO: β = 0.1
RLHF: β = 0.1 ~ 0.2
```

**하이퍼파라미터 탐색 전략:**

```python
# Grid search (작은 규모)
betas = [0.01, 0.05, 0.1, 0.2, 0.5]
lrs = [1e-7, 1e-6, 1e-5]

best_val_reward = -float('inf')
best_params = None

for beta in betas:
    for lr in lrs:
        # Train with these params
        model = train_dpo(beta=beta, lr=lr)

        # Evaluate
        val_reward = evaluate(model)

        if val_reward > best_val_reward:
            best_val_reward = val_reward
            best_params = {'beta': beta, 'lr': lr}

print(f"Best params: {best_params}")
```

## 9.3. 평가 방법

**자동 평가 지표:**

```python
def evaluate_model(model, test_set):
    """
    모델 평가
    """
    metrics = {}

    # 1. Win Rate (선호도 비교)
    win_rate = compute_win_rate(model, test_set)
    metrics['win_rate'] = win_rate

    # 2. Reward Model Score
    avg_reward = compute_avg_reward(model, test_set)
    metrics['avg_reward'] = avg_reward

    # 3. KL Divergence (base model과의 거리)
    kl_div = compute_kl_divergence(model, base_model, test_set)
    metrics['kl_divergence'] = kl_div

    # 4. Perplexity (언어 모델링 품질)
    perplexity = compute_perplexity(model, test_set)
    metrics['perplexity'] = perplexity

    return metrics

# 예시 결과
"""
{
    'win_rate': 0.72,        # 72% 승률 (vs base model)
    'avg_reward': 1.5,       # 평균 reward
    'kl_divergence': 0.05,   # Base model에서의 거리
    'perplexity': 15.2       # 언어 모델링 품질
}
"""
```

**인간 평가:**

```
자동 지표의 한계:
- Reward model의 편향 반영
- 실제 인간 선호도와 불일치 가능

인간 평가 필수:
1. A/B Testing
   - Base model vs Aligned model
   - 어느 응답이 더 나은가?

2. 평가 기준
   - Helpfulness (유용성)
   - Harmlessness (무해성)
   - Honesty (정직성)

3. 규모
   - 최소 500 examples
   - 여러 평가자
   - Agreement 측정
```

**실전 평가 파이프라인:**

```python
def full_evaluation(model):
    """
    포괄적 평가
    """
    results = {}

    # 1. 자동 평가 (빠름, 저렴)
    auto_metrics = evaluate_model(model, test_set)
    results['auto'] = auto_metrics

    # 2. LLM-as-judge (중간)
    judge_scores = llm_judge_evaluation(model, test_prompts)
    results['llm_judge'] = judge_scores

    # 3. 인간 평가 (느림, 비쌈, 정확)
    if should_run_human_eval(auto_metrics):
        human_scores = human_evaluation(model, sample_prompts)
        results['human'] = human_scores

    # 4. 안전성 체크
    safety_score = safety_evaluation(model)
    results['safety'] = safety_score

    return results
```

---

# 10. 요약

**전체 LLM Training Pipeline:**

```
┌──────────────────────────────────────────────────┐
│ Stage 1: Pretraining                             │
│ - 대규모 텍스트로 언어 구조 학습                  │
│ - 결과: Base Model (autocompleter)               │
└────────────────┬─────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────┐
│ Stage 2: Supervised Fine-tuning (SFT)            │
│ - Instruction-response pairs로 학습              │
│ - 결과: Instruction-following Model              │
└────────────────┬─────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────┐
│ Stage 3: Preference Tuning (이번 강의!)          │
│                                                  │
│ 방법 1: RLHF (전통적)                            │
│   ├─ Step 1: Reward Model Training               │
│   │   - Preference pairs로 RM 학습               │
│   │   - Bradley-Terry formulation                │
│   └─ Step 2: RL Fine-tuning                      │
│       - PPO로 LLM 학습                           │
│       - KL penalty로 안정화                      │
│                                                  │
│ 방법 2: DPO (최신, 간단)                         │
│   └─ Preference pairs로 직접 LLM 학습            │
│       - Reward Model 불필요                      │
│       - 한 단계로 완료                           │
│                                                  │
│ 결과: Aligned Model (Helpful, Honest, Harmless)  │
└──────────────────────────────────────────────────┘
```

**핵심 개념 정리:**

```
1. Preference Tuning의 필요성
   - SFT: 지시사항은 따르지만 항상 이상적이지 않음
   - Preference: 인간 선호도에 맞게 정렬
   - 데이터 수집이 더 쉬움 (비교만 하면 됨)

2. Preference Data
   - Pairwise comparison (가장 일반적)
   - Prompt + Chosen + Rejected
   - 품질이 매우 중요!

3. RLHF (Reinforcement Learning from Human Feedback)
   - Stage 1: Reward Model Training
     * Bradley-Terry formulation
     * Loss: -log σ(r_chosen - r_rejected)
   - Stage 2: RL Fine-tuning
     * PPO (Proximal Policy Optimization)
     * KL penalty로 base model 근처 유지

4. PPO (Proximal Policy Optimization)
   - Policy update를 조금씩만
   - PPO-Clip: ratio를 [1-ε, 1+ε]로 제한
   - PPO-KL: KL divergence penalty 사용
   - 안정적인 RL 학습

5. DPO (Direct Preference Optimization)
   - Reward Model 불필요
   - Preference data로 직접 LLM 학습
   - 간단하고 효과적
   - 최근 인기 상승

6. 실전 고려사항
   - 데이터 품질이 가장 중요
   - β (KL penalty) 튜닝 중요
   - 자동 + 인간 평가 병행
```

**RLHF vs DPO 선택 가이드:**

```
┌─────────────────────┬────────────┬──────────┐
│ 상황                │ RLHF       │ DPO      │
├─────────────────────┼────────────┼──────────┤
│ 빠른 프로토타이핑   │            │ ✅       │
│ 제한된 계산 자원    │            │ ✅       │
│ 간단한 구현         │            │ ✅       │
│ 복잡한 reward       │ ✅         │          │
│ Reward 분석 필요    │ ✅         │          │
│ RM 재사용           │ ✅         │          │
│ 안정성 우선         │            │ ✅       │
└─────────────────────┴────────────┴──────────┘

결론: 대부분의 경우 DPO로 시작하는 것을 추천!
```

---

# 11. 중요 용어 정리

**Alignment (정렬)**
```
모델의 행동을 인간의 가치관과 선호도에 맞추는 과정
예: Helpful, Honest, Harmless (HHH)
```

**Preference Tuning (선호도 튜닝)**
```
Preference pairs를 사용하여 모델을 인간 선호도에 맞게 조정
SFT 이후의 3번째 학습 단계
```

**Preference Pair (선호도 쌍)**
```
{prompt, chosen_response, rejected_response}
하나의 프롬프트에 대한 좋은 응답과 나쁜 응답
```

**RLHF (Reinforcement Learning from Human Feedback)**
```
인간 피드백을 강화학습에 활용하는 방법
1. Reward Model 학습
2. RL로 LLM fine-tuning
```

**Reward Model (보상 모델)**
```
r(prompt, response) → scalar score
응답의 품질을 점수로 평가하는 모델
Preference pairs로 학습
```

**Bradley-Terry Formulation**
```
P(y_i > y_j | x) = σ(r(x, y_i) - r(x, y_j))
두 항목 간의 선호도를 확률로 모델링하는 방법
```

**Policy (정책)**
```
π(a|s): 상태 s에서 행동 a를 선택할 확률
LLM에서: P(next_token | context)
```

**Advantage Function (이점 함수)**
```
A(s, a) = Q(s, a) - V(s)
특정 행동이 평균보다 얼마나 좋은지를 나타냄
```

**PPO (Proximal Policy Optimization)**
```
안정적인 RL 알고리즘
Policy update를 조금씩만 수행
- PPO-Clip: ratio clipping으로 제한
- PPO-KL: KL divergence penalty 사용
```

**KL Divergence (KL 발산)**
```
두 확률 분포의 차이를 측정
KL(P || Q) = Σ P(x) log(P(x) / Q(x))
RLHF/DPO에서 base model과의 거리 유지에 사용
```

**KL Penalty**
```
β * KL(π_θ || π_ref)
모델이 base model에서 너무 멀어지지 않게 하는 제약
Reward hacking 방지
```

**Reward Hacking (보상 해킹)**
```
모델이 reward model의 약점을 악용하여
실제로는 나쁘지만 높은 점수를 받는 현상
예: 무의미한 반복, 과도한 길이
```

**DPO (Direct Preference Optimization)**
```
Reward Model 없이 preference data로 직접 LLM 학습
RLHF의 간소화된 버전
더 간단하고 안정적
```

**Rollout (전개)**
```
LLM이 프롬프트로부터 완전한 응답을 생성하는 과정
RL에서 전체 에피소드를 실행하는 것
```

**Temperature**
```
Sampling의 randomness 조절
- 높으면: 더 다양한 응답
- 낮으면: 더 deterministic한 응답
```

**Reference Model (참조 모델)**
```
π_ref: KL penalty 계산을 위한 기준 모델
보통 SFT 모델을 고정하여 사용
학습 중 파라미터 업데이트 안 됨
```

**Value Function (가치 함수)**
```
V(s): 상태 s의 기대 누적 reward
"이 상태는 평균적으로 얼마나 좋은가?"
```

**LLM-as-a-Judge**
```
LLM을 평가자로 사용하여 응답 품질 판단
인간 평가의 대안 (빠르고 저렴)
```

**Pairwise / Pointwise / Listwise**
```
Pairwise: 두 항목 비교 (A vs B)
Pointwise: 개별 항목 점수 부여
Listwise: 여러 항목 순위 매기기
```

**Win Rate (승률)**
```
평가 지표: 모델 A의 응답이 모델 B보다 선호되는 비율
예: 72% win rate = 100개 중 72개에서 승리
```

---

**참고 자료:**

- [Lecture Slides](https://cme295.stanford.edu/slides/fall25-cme295-lecture5.pdf)
- InstructGPT Paper: Training language models to follow instructions with human feedback (OpenAI, 2022)
- DPO Paper: Direct Preference Optimization (Rafailov et al., 2023)
- PPO Paper: Proximal Policy Optimization Algorithms (Schulman et al., 2017)
- RLHF Tutorial: https://huyenchip.com/2023/05/02/rlhf.html

**추가 학습 권장:**

1. Reinforcement Learning 기초
   - Sutton & Barto: Reinforcement Learning (교과서)
   - OpenAI Spinning Up (실습)

2. RLHF 구현
   - trlX (Carper AI)
   - TRL (Hugging Face)
   - DeepSpeed-Chat (Microsoft)

3. DPO 구현
   - DPO Trainer (Hugging Face TRL)
   - Implementation examples on GitHub

4. 최신 연구
   - RLAIF (RL from AI Feedback)
   - Constitutional AI
   - RAFT (Reward rAnked FineTuning)
   - PRO (Preference Ranking Optimization)