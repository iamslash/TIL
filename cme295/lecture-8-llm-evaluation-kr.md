# Lecture 8: LLM Evaluation

# Materials

- [CME 295](https://cme295.stanford.edu/syllabus/)
- [slide](https://cme295.stanford.edu/slides/fall25-cme295-lecture8.pdf)
- [video](https://www.youtube.com/watch?v=8fNP4N46RRo&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=9)

# Table of Contents

- [Lecture 8: LLM Evaluation](#lecture-8-llm-evaluation)
- [Materials](#materials)
- [Table of Contents](#table-of-contents)
- [강의 개요](#강의-개요)
  - [강의 목표](#강의-목표)
  - [주요 학습 내용](#주요-학습-내용)
- [1. Evaluation의 중요성](#1-evaluation의-중요성)
  - [1.1. Evaluation이란?](#11-evaluation이란)
  - [1.2. 왜 Evaluation이 중요한가?](#12-왜-evaluation이-중요한가)
- [2. Human Evaluation](#2-human-evaluation)
  - [2.1. Ideal Scenario: Human Ratings](#21-ideal-scenario-human-ratings)
  - [2.2. Inter-Rater Agreement](#22-inter-rater-agreement)
    - [Agreement Rate의 문제점](#agreement-rate의-문제점)
    - [Cohen's Kappa](#cohens-kappa)
    - [다른 Agreement Metrics](#다른-agreement-metrics)
  - [2.3. Human Evaluation의 한계](#23-human-evaluation의-한계)
- [3. Rule-Based Metrics](#3-rule-based-metrics)
  - [3.1. Rule-Based Metrics란?](#31-rule-based-metrics란)
  - [3.2. METEOR](#32-meteor)
  - [3.3. BLEU](#33-bleu)
  - [3.4. ROUGE](#34-rouge)
  - [3.5. Rule-Based Metrics의 한계](#35-rule-based-metrics의-한계)
- [4. LLM-as-a-Judge](#4-llm-as-a-judge)
  - [4.1. LLM-as-a-Judge란?](#41-llm-as-a-judge란)
  - [4.2. 기본 Setup](#42-기본-setup)
  - [4.3. Structured Outputs](#43-structured-outputs)
  - [4.4. LLM-as-a-Judge의 장점](#44-llm-as-a-judge의-장점)
  - [4.5. Variants](#45-variants)
    - [Pointwise Evaluation](#pointwise-evaluation)
    - [Pairwise Evaluation](#pairwise-evaluation)
- [5. LLM-as-a-Judge의 Biases](#5-llm-as-a-judge의-biases)
  - [5.1. Position Bias](#51-position-bias)
  - [5.2. Verbosity Bias](#52-verbosity-bias)
  - [5.3. Self-Enhancement Bias](#53-self-enhancement-bias)
- [6. Best Practices](#6-best-practices)
  - [6.1. 명확한 Guidelines](#61-명확한-guidelines)
  - [6.2. Binary Scale 사용](#62-binary-scale-사용)
  - [6.3. Rationale First](#63-rationale-first)
  - [6.4. Bias 완화](#64-bias-완화)
  - [6.5. Human Calibration](#65-human-calibration)
  - [6.6. Low Temperature](#66-low-temperature)
- [7. Evaluation Dimensions](#7-evaluation-dimensions)
  - [7.1. Task Performance](#71-task-performance)
  - [7.2. Factuality](#72-factuality)
    - [Fact Extraction](#fact-extraction)
    - [Fact Checking](#fact-checking)
    - [Aggregation](#aggregation)
- [8. Agent Evaluation](#8-agent-evaluation)
  - [8.1. Agent의 Inner Working](#81-agent의-inner-working)
  - [8.2. Tool Prediction Errors](#82-tool-prediction-errors)
    - [Tool Router Error](#tool-router-error)
    - [LLM이 Tool을 사용하지 않는 경우](#llm이-tool을-사용하지-않는-경우)
  - [8.3. Tool Hallucination](#83-tool-hallucination)
- [9. 요약](#9-요약)
  - [핵심 개념](#핵심-개념)
  - [Evaluation 방법 비교](#evaluation-방법-비교)
  - [실전 체크리스트](#실전-체크리스트)
- [10. 중요 용어 정리](#10-중요-용어-정리)

---

# 강의 개요

## 강의 목표

이번 강의에서는 LLM의 출력 품질을 어떻게 평가할 것인가에 대해 학습합니다. 평가는 LLM 개발에서 가장 중요한 단계 중 하나입니다. **측정할 수 없으면 개선할 수 없기 때문입니다.**

**학습 목표:**
- LLM Evaluation의 중요성 이해
- Human Evaluation의 장단점 파악
- Rule-based Metrics의 한계 인식
- LLM-as-a-Judge 개념과 활용법 습득
- Evaluation Biases 이해 및 완화 방법 학습
- Agent Evaluation 기법 이해

## 주요 학습 내용

**1. Evaluation 방법론**
- Human Evaluation: Inter-rater agreement
- Rule-based Metrics: METEOR, BLEU, ROUGE
- LLM-as-a-Judge: 현대적 접근법

**2. LLM-as-a-Judge**
- Structured Outputs
- Pointwise vs Pairwise Evaluation
- Biases와 완화 방법

**3. Evaluation Dimensions**
- Task Performance
- Factuality
- Safety & Alignment

**4. Agent Evaluation**
- Tool Call Evaluation
- Error Analysis

---

# 1. Evaluation의 중요성

## 1.1. Evaluation이란?

"LLM을 평가한다"는 말은 여러 의미를 가질 수 있습니다:

**다양한 평가 측면:**

```
평가 범위:
1. 출력 품질 (Output Quality)
   - Coherence (일관성)
   - Factuality (사실성)
   - Relevance (관련성)
   - Usefulness (유용성)

2. 시스템 메트릭 (System Metrics)
   - Latency (지연시간)
   - Throughput (처리량)
   - Uptime (가용성)

3. 비용 (Cost)
   - 토큰당 비용
   - 운영 비용
```

**이번 강의의 Focus:**

이 강의는 **출력 품질(Output Quality)** 평가에 집중합니다.

## 1.2. 왜 Evaluation이 중요한가?

**핵심 원칙:**

```
"You can't improve what you don't measure."
측정할 수 없으면 개선할 수 없다.
```

**LLM 출력의 특성:**

```python
# LLM은 자유 형식 텍스트를 생성합니다
prompt = "What birthday gift should I get?"

# 가능한 출력:
response1 = "A teddy bear is always sweet."
response2 = "Consider their hobbies and interests..."
response3 = "Here are 10 gift ideas:\n1. Books\n2. ..."

# 어떤 것이 더 좋은가?
# → Evaluation이 필요!
```

**평가가 중요한 이유:**

1. **모델 개선 방향 결정**
   - 어떤 부분이 약한지 파악
   - 개선 효과 측정

2. **모델 간 비교**
   - GPT-4 vs Claude vs LLaMA
   - Fine-tuned model vs Base model

3. **Production Readiness 판단**
   - 사용자에게 배포해도 되는가?
   - 안전한가?

---

# 2. Human Evaluation

## 2.1. Ideal Scenario: Human Ratings

**이상적인 시나리오:**

```
Workflow:
1. Prompt → LLM → Response
2. Human rates the response
3. Collect all ratings
4. Aggregate → Overall performance
```

**예시:**

```python
# Evaluation Process
prompts = [
    "What gift should I get?",
    "Explain quantum computing",
    "Write a Python function..."
]

for prompt in prompts:
    response = llm.generate(prompt)

    # Human rates
    rating = human_rater.rate(
        response=response,
        criteria="usefulness",
        scale="1-5"
    )

    store_rating(prompt, response, rating)
```

**문제점:**

1. **Cost-Intensive (비용이 많이 듦)**
   - 1000개 응답 평가 = 많은 시간과 비용

2. **Slow (느림)**
   - 모델 iteration마다 평가 불가능

3. **Subjective (주관적)**
   - 평가자마다 다른 기준

## 2.2. Inter-Rater Agreement

평가가 주관적일 수 있으므로, **평가자 간 일치도**를 측정해야 합니다.

**예시:**

```
Prompt: "What birthday gift should I get?"
Response: "A teddy bear is almost always a sweet gift.
          Just pick one that feels right for you."

평가 기준: Usefulness (유용성)

Rater 1: "Useful (5/5)"
  → 테디베어는 구체적인 제안이다

Rater 2: "Not useful (2/5)"
  → 어떤 테디베어인지 구체적이지 않다

→ Inter-rater disagreement!
```

### Agreement Rate의 문제점

**단순 Agreement Rate:**

```python
agreement_rate = (평가자들이 동의한 횟수) / (전체 평가 횟수)
```

**문제: Random Chance를 고려하지 않음**

**예시:**

```python
# 두 평가자가 random하게 평가하는 경우
# Binary scale: Good (1) or Bad (0)

P(Alice says Good) = 0.5
P(Bob says Good) = 0.5

# Agreement by random chance:
P(Both agree) = P(A=1, B=1) + P(A=0, B=0)
              = 0.5 * 0.5 + 0.5 * 0.5
              = 0.25 + 0.25
              = 0.5
```

**결론:** Random으로만 평가해도 50% agreement!

**일반화:**

```python
def agreement_by_chance(p_a, p_b):
    """
    p_a: 평가자 A가 1을 선택할 확률
    p_b: 평가자 B가 1을 선택할 확률
    """
    agree_on_1 = p_a * p_b
    agree_on_0 = (1 - p_a) * (1 - p_b)
    return agree_on_1 + agree_on_0

# 예시
print(agreement_by_chance(0.5, 0.5))  # 0.5
print(agreement_by_chance(0.7, 0.7))  # 0.58
```

### Cohen's Kappa

**문제 해결:** Random chance를 고려한 metric

**공식:**

```
κ (kappa) = (P_observed - P_chance) / (1 - P_chance)

여기서:
- P_observed: 실제 관찰된 agreement rate
- P_chance: Random chance에 의한 agreement rate
```

**해석:**

```python
κ = 1.0   # Perfect agreement
κ = 0.0   # Agreement = random chance
κ < 0.0   # Worse than random!
```

**예시:**

```python
# 실제 관찰된 agreement
P_observed = 0.8

# Random chance로 인한 agreement
P_chance = 0.5

# Cohen's Kappa
kappa = (0.8 - 0.5) / (1 - 0.5)
      = 0.3 / 0.5
      = 0.6
```

**Kappa 해석 가이드:**

| Kappa 값 | 해석 |
|----------|------|
| < 0.0 | Poor (나쁨) |
| 0.0 - 0.2 | Slight (약간) |
| 0.2 - 0.4 | Fair (보통) |
| 0.4 - 0.6 | Moderate (중간) |
| 0.6 - 0.8 | Substantial (상당함) |
| 0.8 - 1.0 | Almost Perfect (거의 완벽) |

### 다른 Agreement Metrics

**Cohen's Kappa의 확장:**

1. **Fleiss's Kappa**
   - 3명 이상의 평가자
   - 모든 평가자가 모든 항목을 평가

2. **Krippendorff's Alpha**
   - 임의 수의 평가자
   - Missing data 허용
   - 다양한 data type 지원

**구현 예시:**

```python
from sklearn.metrics import cohen_kappa_score

# 두 평가자의 ratings
rater1 = [1, 0, 1, 1, 0, 1, 0, 0]
rater2 = [1, 0, 1, 0, 0, 1, 1, 0]

kappa = cohen_kappa_score(rater1, rater2)
print(f"Cohen's Kappa: {kappa:.3f}")
```

**실전 활용:**

```python
# Inter-rater agreement 추적
def track_agreement(ratings_per_rater):
    """
    ratings_per_rater: {rater_id: [ratings]}
    """
    kappas = []
    rater_ids = list(ratings_per_rater.keys())

    # 모든 rater 쌍에 대해 kappa 계산
    for i in range(len(rater_ids)):
        for j in range(i+1, len(rater_ids)):
            r1 = ratings_per_rater[rater_ids[i]]
            r2 = ratings_per_rater[rater_ids[j]]
            kappa = cohen_kappa_score(r1, r2)
            kappas.append(kappa)

    avg_kappa = sum(kappas) / len(kappas)
    return avg_kappa

# 사용 예시
ratings = {
    "alice": [1, 0, 1, 1, 0],
    "bob": [1, 0, 1, 0, 0],
    "charlie": [1, 1, 1, 0, 0]
}

avg_kappa = track_agreement(ratings)
print(f"Average Kappa: {avg_kappa:.3f}")

if avg_kappa < 0.6:
    print("⚠️ Agreement is low. Hold calibration session!")
```

## 2.3. Human Evaluation의 한계

**요약:**

| 측면 | 문제점 | 해결 방법 |
|------|--------|-----------|
| Subjectivity | 평가자마다 다른 기준 | Inter-rater agreement 추적 |
| Speed | 매우 느림 | - |
| Cost | 비용이 많이 듦 | - |
| Scalability | 대규모 평가 불가능 | - |

**결론:**

모든 LLM 출력을 Human이 평가하는 것은 **실용적이지 않습니다.**

하지만 Human ratings는 여전히 중요:
- Ground truth로 활용
- 다른 평가 방법의 calibration

---

# 3. Rule-Based Metrics

## 3.1. Rule-Based Metrics란?

**새로운 접근:**

```
이전: Prompt → LLM → Response → Human rates
                                   ↓
                              (매번 평가)

새로운:
1. Prompt → Human writes → Reference (한 번만)
2. Prompt → LLM → Response
            ↓
   Compare(Response, Reference) → Score
```

**핵심 아이디어:**

1. **Reference 작성** (한 번만)
   - 각 prompt에 대한 이상적인 답변
   - Human이 미리 작성

2. **자동 비교**
   - LLM 응답 vs Reference
   - Rule-based formula로 점수 계산

**장점:**

- 반복 가능 (Repeatable)
- 빠름 (Fast)
- 비용 효율적 (Cost-effective)

## 3.2. METEOR

**METEOR: Metric for Evaluation of Translation with Explicit ORdering**

**사용 분야:** 기계 번역

**핵심 아이디어:**

1. Unigram matching (단어 매칭)
2. Ordering penalty (순서 페널티)

**공식:**

```
METEOR = F_score × (1 - Penalty)

F_score = (Precision × Recall) / (α × Precision + (1-α) × Recall)

Penalty = γ × (C / M)^β

여기서:
- Precision: prediction에서 매칭된 unigram 비율
- Recall: reference에서 매칭된 unigram 비율
- C: contiguous chunk 수 (연속된 매칭 덩어리 수)
- M: matched unigram 수
- α, γ, β: 하이퍼파라미터
```

**Precision vs Recall:**

```python
reference = "The cat sat on the mat"
prediction = "cat sat on mat"

# Matched words: cat, sat, on, mat (4개)

Precision = 매칭된 단어 수 / prediction의 단어 수
          = 4 / 4 = 1.0

Recall = 매칭된 단어 수 / reference의 단어 수
       = 4 / 6 = 0.67
```

**Ordering Penalty:**

```python
reference = "The cat sat on the mat"
prediction1 = "cat sat on mat"      # 순서 유지
prediction2 = "mat on sat cat"      # 순서 뒤섞임

# prediction1: C=1 (하나의 연속된 덩어리)
# prediction2: C=4 (4개의 분리된 덩어리)

# Penalty가 prediction2에서 더 높음
```

**예시:**

```python
def simple_meteor_score(reference, prediction):
    """
    Simplified METEOR implementation
    """
    ref_words = reference.lower().split()
    pred_words = prediction.lower().split()

    # Matched words
    matched = set(ref_words) & set(pred_words)

    # Precision & Recall
    precision = len(matched) / len(pred_words) if pred_words else 0
    recall = len(matched) / len(ref_words) if ref_words else 0

    # F-score (α=0.9)
    alpha = 0.9
    if precision + recall == 0:
        f_score = 0
    else:
        f_score = (precision * recall) / \
                  (alpha * precision + (1-alpha) * recall)

    # Simplified penalty (실제는 더 복잡)
    # C: contiguous chunks
    # 간단히 하기 위해 penalty=0으로 가정
    penalty = 0

    meteor = f_score * (1 - penalty)
    return meteor

# 예시
ref = "The cat sat on the mat"
pred1 = "The cat sat on the mat"
pred2 = "cat mat"

print(f"METEOR (pred1): {simple_meteor_score(ref, pred1):.3f}")
print(f"METEOR (pred2): {simple_meteor_score(ref, pred2):.3f}")
```

**장점:**

- Synonym 고려 (확장 가능)
- Ordering 고려
- Translation에 효과적

**단점:**

- 많은 하이퍼파라미터 (α, γ, β)
- Stylistic variation 허용 안함
- 여전히 reference 필요

## 3.3. BLEU

**BLEU: BiLingual Evaluation Understudy**

**특징:** Precision-focused metric

**핵심 아이디어:**

```
BLEU = BP × exp(∑(w_n × log(p_n)))

여기서:
- p_n: n-gram precision
- w_n: weight (보통 균등)
- BP: Brevity Penalty
```

**N-gram Precision:**

```python
reference = "The cat sat on the mat"
prediction = "The cat sat"

# Unigram (1-gram)
# "The", "cat", "sat" 모두 매칭
unigram_precision = 3/3 = 1.0

# Bigram (2-gram)
# "The cat", "cat sat" 모두 매칭
bigram_precision = 2/2 = 1.0
```

**Brevity Penalty (BP):**

```python
# Precision만 사용하면 짧은 문장에 유리
reference = "The cat sat on the mat"  # length=6
prediction = "The cat"                 # length=2

# 문제: precision=1.0이지만 내용이 부족!

# 해결: Brevity Penalty
BP = exp(1 - len(reference)/len(prediction))
   = exp(1 - 6/2)
   = exp(-2)
   ≈ 0.135

# BLEU score가 큰 폭으로 감소!
```

**구현 예시:**

```python
from collections import Counter

def ngrams(words, n):
    """Generate n-grams from words"""
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]

def bleu_score(reference, prediction, max_n=4):
    """
    Simplified BLEU implementation
    """
    ref_words = reference.split()
    pred_words = prediction.split()

    # Brevity Penalty
    ref_len = len(ref_words)
    pred_len = len(pred_words)

    if pred_len == 0:
        return 0.0

    if pred_len < ref_len:
        bp = math.exp(1 - ref_len / pred_len)
    else:
        bp = 1.0

    # N-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(ngrams(ref_words, n))
        pred_ngrams = Counter(ngrams(pred_words, n))

        # Clipped counts
        clipped_counts = sum(min(pred_ngrams[ng], ref_ngrams[ng])
                            for ng in pred_ngrams)
        total_counts = sum(pred_ngrams.values())

        if total_counts == 0:
            precision = 0
        else:
            precision = clipped_counts / total_counts

        precisions.append(precision)

    # Geometric mean
    if min(precisions) == 0:
        return 0.0

    geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))

    return bp * geo_mean

# 예시
ref = "The cat sat on the mat"
pred1 = "The cat sat on the mat"
pred2 = "cat mat"

print(f"BLEU (pred1): {bleu_score(ref, pred1):.3f}")
print(f"BLEU (pred2): {bleu_score(ref, pred2):.3f}")
```

**장점:**

- 널리 사용됨
- 번역 품질과 상관관계
- 빠른 계산

**단점:**

- Recall 무시
- 짧은 문장에 유리 (BP로 완화)
- Stylistic variation 허용 안함

## 3.4. ROUGE

**ROUGE: Recall-Oriented Understudy for Gisting Evaluation**

**사용 분야:** 요약 (Summarization)

**특징:** Recall-focused metric

**Variants:**

1. **ROUGE-N:** N-gram overlap

```python
reference = "The cat sat on the mat"
summary = "The cat sat"

# ROUGE-1 (unigram recall)
matched_unigrams = 3  # "The", "cat", "sat"
total_ref_unigrams = 6

ROUGE-1 = 3 / 6 = 0.5
```

2. **ROUGE-L:** Longest Common Subsequence

```python
reference = "The cat sat on the mat"
summary = "cat on mat"

# LCS: "cat on mat" (길이 3)
# (단어 순서 유지)

ROUGE-L = LCS_length / ref_length = 3 / 6 = 0.5
```

3. **ROUGE-W:** Weighted LCS (연속성 선호)

**구현 예시:**

```python
def rouge_n(reference, summary, n=1):
    """
    ROUGE-N score (recall-based)
    """
    ref_words = reference.split()
    sum_words = summary.split()

    ref_ngrams = Counter(ngrams(ref_words, n))
    sum_ngrams = Counter(ngrams(sum_words, n))

    # Overlapping n-grams
    overlap = sum(min(sum_ngrams[ng], ref_ngrams[ng])
                  for ng in sum_ngrams)
    total = sum(ref_ngrams.values())

    if total == 0:
        return 0.0

    return overlap / total

# 예시
ref = "The cat sat on the mat"
sum1 = "The cat sat on the mat"
sum2 = "cat mat"

print(f"ROUGE-1 (sum1): {rouge_n(ref, sum1, 1):.3f}")
print(f"ROUGE-1 (sum2): {rouge_n(ref, sum2, 1):.3f}")
print(f"ROUGE-2 (sum1): {rouge_n(ref, sum1, 2):.3f}")
```

## 3.5. Rule-Based Metrics의 한계

**핵심 문제:**

**1. Stylistic Variation을 허용하지 않음**

```python
# 같은 의미, 다른 표현
reference = "A plush teddy bear can comfort a child during bedtime."

variation1 = "Soft stuffed bears often help kids feel safe as they fall asleep."

variation2 = "Many youngsters rest more easily at night when they cuddle a gentle toy companion."

# METEOR, BLEU, ROUGE 모두 낮은 점수!
# 하지만 의미는 동일함
```

**2. Correlation이 높지 않음**

```python
# Human ratings와의 상관관계
correlation_with_human = {
    "BLEU": 0.4,      # 낮음
    "METEOR": 0.55,   # 중간
    "ROUGE": 0.5      # 중간
}

# 완벽한 상관관계 = 1.0
# 실제로는 그렇게 높지 않음
```

**3. 여전히 Reference가 필요**

```python
# Reference 작성 비용
num_prompts = 1000
time_per_reference = 5  # minutes

total_time = num_prompts * time_per_reference
           = 5000 minutes
           = 83 hours

# 여전히 비용이 많이 듦!
```

**비교표:**

| Metric | Focus | 장점 | 단점 | 사용 사례 |
|--------|-------|------|------|-----------|
| METEOR | Precision+Recall+Order | 순서 고려 | 많은 하이퍼파라미터 | 번역 |
| BLEU | Precision | 널리 사용됨 | Recall 무시 | 번역 |
| ROUGE | Recall | 요약에 적합 | Precision 무시 | 요약 |

**결론:**

Rule-based metrics는 **stylistic variation을 포착하지 못합니다.**

더 나은 방법이 필요합니다! → **LLM-as-a-Judge**

---

# 4. LLM-as-a-Judge

## 4.1. LLM-as-a-Judge란?

**핵심 아이디어:**

```
우리는 7개 강의 동안 LLM에 대해 배웠습니다:
- 대규모 데이터로 pre-training
- Human preference로 fine-tuning
- Human knowledge 내재화

그렇다면... LLM을 평가에 사용하면 어떨까?
```

**Setup:**

```
Input:
┌──────────────────────┐
│ Prompt              │  (원본 질문)
│ Response            │  (LLM 응답)
│ Evaluation Criteria │  (평가 기준)
└──────────────────────┘
         ↓
    LLM-as-a-Judge
         ↓
Output:
┌──────────────────────┐
│ Score               │  (점수)
│ Rationale           │  (이유)
└──────────────────────┘
```

**차별점:**

Rule-based metrics와 달리:
- Reference 불필요!
- Rationale 제공! (설명 가능)

## 4.2. 기본 Setup

**Prompt Template:**

```python
prompt_template = """
You are an expert evaluator. Your task is to evaluate the quality of a response.

**Evaluation Criteria:** {criteria}

**User Prompt:** {user_prompt}

**Model Response:** {model_response}

Please provide:
1. Rationale: Explain your reasoning step-by-step
2. Score: Choose either "pass" or "fail"

Output format:
Rationale: <your explanation>
Score: <pass/fail>
"""
```

**예시:**

```python
# Usefulness 평가
criteria = "Usefulness: Does the response provide helpful information to answer the question?"

user_prompt = "What birthday gift should I get?"

model_response = "A teddy bear is almost always a sweet gift. Just pick one that feels right for you."

# LLM-as-a-Judge call
judge_input = prompt_template.format(
    criteria=criteria,
    user_prompt=user_prompt,
    model_response=model_response
)

judge_output = judge_llm.generate(judge_input)
print(judge_output)
```

**출력 예시:**

```
Rationale: The response provides a concrete suggestion (teddy bear)
which gives the user a starting point. However, it lacks specifics
about age appropriateness, interests, or budget considerations.
The advice to "pick one that feels right" is somewhat generic.

Score: pass
```

**핵심 트릭: Rationale First!**

```python
# ✅ Good: Rationale 먼저
output_format = """
Rationale: <explanation>
Score: <pass/fail>
"""

# ❌ Bad: Score 먼저
output_format = """
Score: <pass/fail>
Rationale: <explanation>
"""
```

**왜 Rationale을 먼저 출력해야 하나?**

Lecture 6 (Reasoning)에서 배운 내용:
- Chain-of-Thought (CoT)
- Reasoning models (o1, o3)

→ 생각 과정을 먼저 externalize하면 성능 향상!

```python
# CoT와 같은 원리
# Think → Answer

# Judge에서도 동일
# Explain reasoning → Give score
```

## 4.3. Structured Outputs

**문제:**

```python
# LLM output은 probabilistic
judge_output = judge_llm.generate(prompt)

# 원하는 형식:
# Rationale: ...
# Score: pass

# 실제 output:
# "I think this is good..."  (파싱 불가!)
```

**해결: Constrained Decoding**

Lecture 3에서 배운 내용:
- Constrained-guided decoding
- Valid tokens만 sampling

**Provider API:**

```python
from pydantic import BaseModel

# 1. Response 형식 정의
class JudgeResponse(BaseModel):
    rationale: str
    score: str  # "pass" or "fail"

# 2. Structured output 사용
judge_output = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": judge_prompt}],
    response_format=JudgeResponse  # ✅ 형식 보장!
)

# 3. 파싱 보장됨
rationale = judge_output.rationale
score = judge_output.score
```

**OpenAI 예시:**

```python
import openai
from pydantic import BaseModel

class EvaluationResult(BaseModel):
    rationale: str
    score: str  # "pass" or "fail"

response = openai.beta.chat.completions.parse(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": "You are an expert evaluator."},
        {"role": "user", "content": judge_prompt}
    ],
    response_format=EvaluationResult
)

result = response.choices[0].message.parsed
print(f"Score: {result.score}")
print(f"Rationale: {result.rationale}")
```

**Anthropic 예시:**

```python
import anthropic

# Tool/function calling으로 구조화
tools = [{
    "name": "submit_evaluation",
    "description": "Submit evaluation result",
    "input_schema": {
        "type": "object",
        "properties": {
            "rationale": {"type": "string"},
            "score": {"type": "string", "enum": ["pass", "fail"]}
        },
        "required": ["rationale", "score"]
    }
}]

response = anthropic.messages.create(
    model="claude-3-opus-20240229",
    messages=[{"role": "user", "content": judge_prompt}],
    tools=tools,
    tool_choice={"type": "tool", "name": "submit_evaluation"}
)

# Structured output 보장
tool_use = response.content[0]
result = tool_use.input
```

## 4.4. LLM-as-a-Judge의 장점

**1. Reference 불필요**

```python
# Rule-based metrics
bleu_score(reference, prediction)  # reference 필요 ❌

# LLM-as-a-Judge
judge_score(prompt, response, criteria)  # reference 불필요 ✅
```

**2. Interpretable (해석 가능)**

```python
# Rule-based
score = 0.4523  # 무슨 의미? 🤷

# LLM-as-a-Judge
{
    "score": "fail",
    "rationale": "The response lacks specific details about pricing
                  and doesn't address the budget constraint mentioned
                  in the question."  # 명확한 이유! ✅
}
```

**3. Flexible Criteria**

```python
# 다양한 기준으로 평가 가능
criteria_list = [
    "Usefulness",
    "Factuality",
    "Safety",
    "Coherence",
    "Politeness",
    "Conciseness"
]

for criteria in criteria_list:
    score = judge(prompt, response, criteria)
```

**4. No Training Required**

```python
# Rule-based metrics
# → Hyperparameter tuning 필요 (α, β, γ...)

# LLM-as-a-Judge
# → Zero-shot으로 작동! (LLM의 사전 지식 활용)
```

## 4.5. Variants

### Pointwise Evaluation

**단일 응답 평가:**

```
Input:  1개의 response
Output: "pass" or "fail"
```

**예시:**

```python
def pointwise_judge(prompt, response, criteria):
    """
    단일 응답을 절대적으로 평가
    """
    judge_prompt = f"""
    Evaluate this response based on: {criteria}

    Prompt: {prompt}
    Response: {response}

    Output:
    Rationale: <explanation>
    Score: pass/fail
    """

    result = judge_llm.generate(judge_prompt)
    return parse_result(result)

# 사용
score = pointwise_judge(
    prompt="What is the capital of France?",
    response="The capital of France is Paris.",
    criteria="Factual accuracy"
)
```

### Pairwise Evaluation

**두 응답 비교:**

```
Input:  2개의 responses (A, B)
Output: "A is better" or "B is better"
```

**예시:**

```python
def pairwise_judge(prompt, response_a, response_b, criteria):
    """
    두 응답을 상대적으로 비교
    """
    judge_prompt = f"""
    Compare these two responses based on: {criteria}

    Prompt: {prompt}

    Response A: {response_a}
    Response B: {response_b}

    Which response is better?

    Output:
    Rationale: <explanation>
    Winner: A/B
    """

    result = judge_llm.generate(judge_prompt)
    return parse_result(result)

# 사용
winner = pairwise_judge(
    prompt="Explain quantum computing",
    response_a="Quantum computing uses qubits...",
    response_b="Quantum computers are fast...",
    criteria="Clarity and accuracy"
)
```

**Pairwise의 활용: Preference Data 생성**

Lecture 5 (Fine-tuning)에서 배운 내용:
- DPO, PPO 등은 preference data 필요
- (prompt, chosen, rejected) 쌍

```python
# Synthetic preference data generation
prompts = load_prompts()

for prompt in prompts:
    # 두 모델에서 응답 생성
    response_a = model_a.generate(prompt)
    response_b = model_b.generate(prompt)

    # LLM-as-a-Judge로 preference 판단
    winner = pairwise_judge(prompt, response_a, response_b, "quality")

    if winner == "A":
        preference_data.append({
            "prompt": prompt,
            "chosen": response_a,
            "rejected": response_b
        })
    else:
        preference_data.append({
            "prompt": prompt,
            "chosen": response_b,
            "rejected": response_a
        })

# DPO training에 사용!
train_dpo(model, preference_data)
```

**Pointwise vs Pairwise 비교:**

| 측면 | Pointwise | Pairwise |
|------|-----------|----------|
| 입력 | 1개 response | 2개 responses |
| 출력 | 절대적 평가 | 상대적 비교 |
| 사용 사례 | Quality check | Model comparison |
| Preference data | ❌ | ✅ |
| 속도 | 빠름 | 느림 (2배 입력) |

---

# 5. LLM-as-a-Judge의 Biases

LLM-as-a-Judge도 완벽하지 않습니다. 여러 bias가 존재합니다.

## 5.1. Position Bias

**정의:** 응답의 위치에 따라 평가가 달라지는 현상

**예시:**

```python
# Pairwise evaluation
prompt = "Explain AI"
response_a = "AI is ..."  # 좋은 응답
response_b = "AI means ..."  # 나쁜 응답

# Case 1: A를 먼저 제시
result1 = pairwise_judge(prompt, response_a, response_b)
# Output: "A is better" ✅

# Case 2: B를 먼저 제시
result2 = pairwise_judge(prompt, response_b, response_a)
# Output: "B is better" ❌ (Position bias!)
```

**원인:**

- LLM이 처음 본 것에 bias
- "Primacy effect" (첫 번째 것 선호)

**완화 방법:**

**1. Position Swapping + Majority Voting**

```python
def robust_pairwise_judge(prompt, response_a, response_b, criteria):
    """
    Position bias를 완화한 pairwise judge
    """
    # 1st call: A then B
    result1 = pairwise_judge(prompt, response_a, response_b, criteria)

    # 2nd call: B then A
    result2 = pairwise_judge(prompt, response_b, response_a, criteria)

    # Majority voting
    if result1 == "A" and result2 == "B":
        # Consistent: A is better
        return "A"
    elif result1 == "B" and result2 == "A":
        # Consistent: B is better
        return "B"
    else:
        # Inconsistent: Position bias detected
        # Tie-breaking strategy (예: 재평가)
        return "Uncertain"

# 사용
winner = robust_pairwise_judge(
    prompt="What is AI?",
    response_a="...",
    response_b="...",
    criteria="Clarity"
)
```

**2. 다중 평가 + 통계**

```python
def multi_trial_judge(prompt, response_a, response_b, criteria, n_trials=5):
    """
    여러 번 평가하여 robust한 결과 도출
    """
    votes = []

    for _ in range(n_trials):
        # Random order
        if random.random() < 0.5:
            result = pairwise_judge(prompt, response_a, response_b, criteria)
            votes.append("A" if result == "A" else "B")
        else:
            result = pairwise_judge(prompt, response_b, response_a, criteria)
            votes.append("B" if result == "A" else "A")

    # Majority vote
    a_count = votes.count("A")
    b_count = votes.count("B")

    if a_count > b_count:
        return "A"
    else:
        return "B"
```

## 5.2. Verbosity Bias

**정의:** 더 긴(verbose) 응답을 선호하는 경향

**예시:**

```python
prompt = "What is 2+2?"

# 짧고 정확한 응답
response_short = "4"

# 길지만 불필요한 응답
response_verbose = """
To answer this question, let's break it down step by step:
1. We start with the number 2
2. We add another 2 to it
3. The operation of addition combines these values
4. Using basic arithmetic principles
5. The result is 4

Additionally, this demonstrates fundamental mathematical concepts...
(continues for 3 more paragraphs)
"""

# LLM-as-a-Judge가 verbose를 선호할 수 있음!
result = pairwise_judge(prompt, response_short, response_verbose, "quality")
# Might prefer verbose ❌
```

**완화 방법:**

**1. Explicit Guidelines**

```python
criteria = """
Quality: Evaluate based on correctness and conciseness.
**Important:** Do NOT prefer responses simply because they are longer.
Focus on whether the response correctly and efficiently answers the question.
"""

result = pairwise_judge(prompt, response_a, response_b, criteria)
```

**2. In-Context Examples**

```python
judge_prompt = f"""
Here are examples of good evaluations:

Example 1:
Q: What is 2+2?
Response A: "4"
Response B: "The answer is 4. Let me explain..."
Better: A (concise and correct)

Example 2:
Q: Explain quantum entanglement
Response A: "It's complicated"
Response B: "Quantum entanglement is a phenomenon where..."
Better: B (provides actual explanation)

Now evaluate:
Q: {prompt}
Response A: {response_a}
Response B: {response_b}
"""
```

**3. Length Penalty**

```python
def length_adjusted_score(prompt, response, criteria):
    """
    길이를 고려한 점수 조정
    """
    # Base score
    score = pointwise_judge(prompt, response, criteria)

    # Length penalty
    words = len(response.split())
    expected_length = estimate_expected_length(prompt)

    if words > expected_length * 2:
        # Too verbose: apply penalty
        penalty = 0.1 * (words / expected_length - 2)
        score = score - penalty

    return score
```

## 5.3. Self-Enhancement Bias

**정의:** 자신이 생성한 응답을 선호하는 경향

**예시:**

```python
# GPT-4가 생성한 응답들
response_gpt4_a = gpt4.generate(prompt)
response_gpt4_b = gpt4.generate(prompt)

# GPT-4를 judge로 사용
judge = gpt4

# 문제: GPT-4가 자신의 출력을 선호할 수 있음
result = pairwise_judge(
    prompt=prompt,
    response_a=response_gpt4_a,
    response_b=response_claude,  # 다른 모델
    criteria="quality",
    judge_model=judge  # ⚠️ Same model!
)
# Might prefer response_gpt4_a due to self-enhancement bias
```

**원인:**

```
모델이 응답을 생성했다는 것은
→ 해당 sequence의 probability가 높다고 판단
→ Judge로 사용될 때도 그 응답을 "더 그럴듯하게" 인식
```

**완화 방법:**

**1. 다른 모델을 Judge로 사용**

```python
# ✅ Good: 생성 모델 ≠ Judge 모델
generator = gpt4
judge = claude  # 다른 모델!

response_a = generator.generate(prompt)
response_b = another_model.generate(prompt)

result = pairwise_judge(
    prompt=prompt,
    response_a=response_a,
    response_b=response_b,
    criteria="quality",
    judge_model=judge  # ✅ Different model
)
```

**2. 더 강력한 Judge 사용**

```python
# 생성: 작은 모델
generator = gpt_3_5_turbo

# Judge: 큰 모델 (reasoning 능력 우수)
judge = gpt_4_turbo  # 또는 o1, claude-opus

# 큰 모델이 작은 모델의 출력을 더 객관적으로 평가 가능
```

**3. Ensemble Judges**

```python
def ensemble_judge(prompt, response_a, response_b, criteria):
    """
    여러 모델을 judge로 사용
    """
    judges = [gpt4, claude, gemini]
    votes = []

    for judge in judges:
        result = pairwise_judge(
            prompt, response_a, response_b, criteria,
            judge_model=judge
        )
        votes.append(result)

    # Majority voting
    return max(set(votes), key=votes.count)
```

**Bias 요약:**

| Bias | 설명 | 완화 방법 |
|------|------|-----------|
| Position Bias | 먼저 제시된 응답 선호 | Position swapping + voting |
| Verbosity Bias | 긴 응답 선호 | Explicit guidelines, examples |
| Self-Enhancement | 자신의 출력 선호 | 다른 모델 사용, 큰 모델 사용 |

---

# 6. Best Practices

## 6.1. 명확한 Guidelines

**나쁜 예:**

```python
criteria = "Evaluate the quality"
# 너무 모호함! "quality"가 무엇을 의미하나?
```

**좋은 예:**

```python
criteria = """
Evaluate the response based on these specific dimensions:

1. Factual Accuracy:
   - Are all facts correct?
   - Are there any hallucinations?

2. Completeness:
   - Does it fully answer the question?
   - Are important details missing?

3. Clarity:
   - Is it easy to understand?
   - Is the structure logical?

Score "pass" if ALL dimensions are satisfactory.
Score "fail" if ANY dimension is unsatisfactory.
"""
```

**구체적인 예시 포함:**

```python
criteria = """
Usefulness for gift recommendations:

✅ Pass criteria:
- Provides specific gift suggestions
- Considers recipient's characteristics (age, interests)
- Explains why the gift is appropriate

❌ Fail criteria:
- Generic advice ("just get something nice")
- No specific suggestions
- Ignores context from the question

Example of PASS:
"For a 5-year-old who loves dinosaurs, consider a dinosaur toy set
or a picture book about dinosaurs."

Example of FAIL:
"Get them something nice."
"""
```

## 6.2. Binary Scale 사용

**나쁜 예: Granular Scale**

```python
# 1-5 scale
score_options = ["1", "2", "3", "4", "5"]
# 문제: 3과 4의 차이가 명확하지 않음
```

**좋은 예: Binary Scale**

```python
# Pass/Fail
score_options = ["pass", "fail"]
# 명확함: 기준을 충족하는가, 아닌가?
```

**이유:**

1. **판단이 쉬움**
   ```python
   # Human도 binary가 쉬움
   "Is this good enough?" → Yes/No

   # vs
   "Rate this 1-5" → 3? 4? 애매함
   ```

2. **Noise 감소**
   ```python
   # 5-scale
   rater1: 3
   rater2: 4
   # 의견 차이인가, noise인가?

   # Binary
   rater1: pass
   rater2: pass
   # 명확한 agreement
   ```

3. **Calibration이 쉬움**
   ```python
   # Binary: 하나의 threshold만 조정
   if quality_score > threshold:
       return "pass"

   # Multi-scale: 여러 threshold 조정 필요
   ```

## 6.3. Rationale First

**항상 rationale을 먼저 출력:**

```python
# ✅ Good
output_format = """
First, provide your reasoning:
Rationale: <step-by-step explanation>

Then, provide your score:
Score: pass/fail
"""

# ❌ Bad
output_format = """
Score: pass/fail
Rationale: <explanation>
"""
```

**이유: Chain-of-Thought 효과**

```python
# Analogy to CoT reasoning (Lecture 6)

# Without CoT:
Q: "What is 17 * 23?"
A: "391"  # May be wrong

# With CoT:
Q: "What is 17 * 23?"
A: "Let me break this down:
    17 * 20 = 340
    17 * 3 = 51
    340 + 51 = 391"  # More likely to be correct

# Same for Judge:
# Thinking first → Better judgment
```

## 6.4. Bias 완화

**Position Bias:**

```python
def evaluate_with_position_mitigation(prompt, response_a, response_b):
    # Evaluate both orders
    result_ab = judge(prompt, response_a, response_b)
    result_ba = judge(prompt, response_b, response_a)

    # Check consistency
    if result_ab == result_ba:
        return result_ab  # Consistent
    else:
        # Inconsistent: Run again or use tie-breaker
        return "uncertain"
```

**Verbosity Bias:**

```python
guidelines = """
IMPORTANT: Do not prefer a response simply because it is longer.
Evaluate based on:
1. Correctness
2. Completeness
3. Conciseness (brevity is a virtue!)
"""
```

**Self-Enhancement Bias:**

```python
# Use different model for judging
generator_model = "gpt-3.5-turbo"
judge_model = "gpt-4-turbo"  # Stronger, different model

# Or use specialized judge models
judge_model = "prometheus-eval"  # Specialized for evaluation
```

## 6.5. Human Calibration

**정기적으로 human ratings와 비교:**

```python
def calibration_study(judge_model, test_cases, human_ratings):
    """
    Judge와 human ratings 비교
    """
    judge_ratings = []

    for case in test_cases:
        judge_rating = judge_model.evaluate(
            prompt=case["prompt"],
            response=case["response"],
            criteria=case["criteria"]
        )
        judge_ratings.append(judge_rating)

    # Correlation analysis
    from scipy.stats import pearsonr

    correlation, p_value = pearsonr(human_ratings, judge_ratings)

    print(f"Correlation: {correlation:.3f}")
    print(f"P-value: {p_value:.3f}")

    if correlation < 0.7:
        print("⚠️ Low correlation! Revise judge prompt.")

    return correlation

# 사용
test_cases = load_test_cases()
human_ratings = load_human_ratings()

correlation = calibration_study(judge_model, test_cases, human_ratings)
```

**Calibration workflow:**

```python
"""
1. 소규모 human rating 수집 (100-500 examples)
2. Same examples에 대해 LLM-as-a-Judge 실행
3. Correlation 분석
4. Correlation < 0.7이면:
   - Judge prompt 수정
   - Guidelines 명확화
   - 다른 judge model 시도
5. Correlation >= 0.7이면:
   - LLM-as-a-Judge 신뢰하고 대규모 평가
"""
```

## 6.6. Low Temperature

**Reproducibility를 위해 낮은 temperature 사용:**

```python
# ✅ For evaluation: Low temperature
judge_result = judge_llm.generate(
    prompt=judge_prompt,
    temperature=0.1  # or 0.2
)
# 재현 가능한 평가

# ❌ For evaluation: High temperature
judge_result = judge_llm.generate(
    prompt=judge_prompt,
    temperature=0.9
)
# 매번 다른 결과 → 신뢰할 수 없음
```

**이유:**

```python
# 낮은 temperature
temperature = 0.1
# → 거의 deterministic
# → 같은 입력 → 같은 출력 (대부분)

# 높은 temperature
temperature = 0.9
# → 확률적 sampling
# → 같은 입력 → 다른 출력 가능
```

**Best Practices 요약:**

| Practice | Why | How |
|----------|-----|-----|
| 명확한 Guidelines | 모호함 제거 | 구체적 예시 포함 |
| Binary Scale | 판단 쉬움, Noise 감소 | Pass/Fail |
| Rationale First | CoT 효과 | Reasoning → Score |
| Bias 완화 | 공정한 평가 | Position swapping 등 |
| Human Calibration | Ground truth 확인 | Correlation 분석 |
| Low Temperature | 재현성 | temp=0.1-0.2 |

---

# 7. Evaluation Dimensions

## 7.1. Task Performance

**평가할 수 있는 다양한 차원:**

```python
evaluation_dimensions = {
    # Task Performance
    "usefulness": "Does the response help the user?",
    "relevance": "Is the response relevant to the question?",
    "completeness": "Does it fully answer the question?",

    # Response Format
    "coherence": "Is the response logically structured?",
    "tone": "Is the tone appropriate?",
    "style": "Does it match the desired style?",

    # Safety & Alignment
    "safety": "Is the response safe and ethical?",
    "factuality": "Are all facts correct?",
    "bias": "Is the response free from harmful biases?"
}
```

**예시: Usefulness 평가**

```python
criteria = """
Usefulness: Does the response provide actionable information
that helps the user achieve their goal?

Pass criteria:
- Provides specific, actionable advice
- Addresses the user's actual question
- Gives relevant details

Fail criteria:
- Too vague or generic
- Doesn't address the question
- Missing important information
"""

result = pointwise_judge(
    prompt="How do I bake a cake?",
    response="Mix ingredients and bake.",
    criteria=criteria
)
# Likely: "fail" (too vague)
```

## 7.2. Factuality

**Factuality는 특별한 처리가 필요합니다.**

**왜 특별한가?**

```python
# 다른 차원들
usefulness: "Is this useful?" → Yes/No (주관적)
coherence: "Is this coherent?" → Yes/No (주관적)

# Factuality
factuality: "Is this factually correct?" → Needs verification!
# 단순히 "맞다/틀리다"를 판단할 수 없음
# 외부 지식 필요
```

**문제:**

```python
text = """
Teddy bears, first created in the 1920s, were named after
President Theodore Roosevelt after he proudly wanted to shoot
a captured bear on a hunting trip.
"""

# 질문: 이 텍스트는 얼마나 사실적인가?

# 문제:
# 1. 여러 fact가 섞여 있음
# 2. 일부는 맞고 일부는 틀림
# 3. Binary (pass/fail)로는 nuance 포착 못함
```

**해결책: 3-Step Factuality Evaluation**

### Fact Extraction

**Step 1: 텍스트를 개별 fact로 분해**

```python
def extract_facts(text):
    """
    텍스트에서 개별 fact 추출
    """
    prompt = f"""
    Extract individual factual claims from this text.
    Each claim should be atomic (one fact per claim).

    Text: {text}

    Output format:
    1. <fact 1>
    2. <fact 2>
    ...
    """

    facts = llm.generate(prompt)
    return parse_facts(facts)

# 예시
text = """
Teddy bears, first created in the 1920s, were named after
President Theodore Roosevelt after he proudly wanted to shoot
a captured bear on a hunting trip.
"""

facts = extract_facts(text)
print(facts)
```

**출력:**

```
Facts:
1. Teddy bears were first created in the 1920s
2. Teddy bears were named after President Theodore Roosevelt
3. The name came from a hunting trip incident
4. President Roosevelt proudly wanted to shoot a captured bear
```

### Fact Checking

**Step 2: 각 fact를 개별적으로 검증**

```python
def check_fact(fact, use_rag=True, use_web_search=True):
    """
    개별 fact 검증
    """
    evidence = []

    # RAG: Knowledge base 검색
    if use_rag:
        relevant_docs = rag_system.retrieve(fact)
        evidence.extend(relevant_docs)

    # Web search
    if use_web_search:
        search_results = web_search(fact)
        evidence.extend(search_results)

    # Judge: Evidence 기반 검증
    prompt = f"""
    Fact to verify: {fact}

    Evidence:
    {format_evidence(evidence)}

    Is this fact correct?
    Rationale: <explanation>
    Verdict: correct/incorrect
    """

    result = judge_llm.generate(prompt)
    return parse_result(result)

# 예시
facts = [
    "Teddy bears were first created in the 1920s",
    "Teddy bears were named after President Theodore Roosevelt",
    "The name came from a hunting trip incident",
    "President Roosevelt proudly wanted to shoot a captured bear"
]

fact_results = []
for fact in facts:
    result = check_fact(fact)
    fact_results.append(result)
    print(f"{fact}: {result['verdict']}")
```

**출력:**

```
Teddy bears were first created in the 1920s: incorrect
  (Actually 1900s)

Teddy bears were named after President Theodore Roosevelt: correct

The name came from a hunting trip incident: correct

President Roosevelt proudly wanted to shoot a captured bear: incorrect
  (He actually refused to shoot it)
```

### Aggregation

**Step 3: 개별 fact 결과를 종합**

```python
def aggregate_factuality_score(fact_results, weights=None):
    """
    개별 fact 검증 결과를 종합
    """
    if weights is None:
        # Equal weights
        weights = [1.0] * len(fact_results)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Weighted sum
    score = sum(
        w * (1 if result['verdict'] == 'correct' else 0)
        for w, result in zip(weights, fact_results)
    )

    return score

# 예시: Equal weights
fact_results = [
    {'verdict': 'incorrect'},  # 1920s → 1900s
    {'verdict': 'correct'},    # Named after Roosevelt
    {'verdict': 'correct'},    # Hunting trip
    {'verdict': 'incorrect'}   # Proudly wanted → refused
]

score = aggregate_factuality_score(fact_results)
print(f"Factuality Score: {score:.2f}")
# Output: 0.50 (2 correct out of 4)
```

**중요도 가중치:**

```python
# 일부 fact가 더 중요할 수 있음
facts_with_importance = [
    {"fact": "Named after Roosevelt", "importance": 3},  # 핵심
    {"fact": "1920s", "importance": 1},  # 덜 중요
    {"fact": "Hunting trip", "importance": 2},
    {"fact": "Proudly wanted to shoot", "importance": 1}
]

weights = [f["importance"] for f in facts_with_importance]
score = aggregate_factuality_score(fact_results, weights)
print(f"Weighted Factuality Score: {score:.2f}")
```

**전체 Pipeline:**

```python
def evaluate_factuality(text):
    """
    Complete factuality evaluation pipeline
    """
    # Step 1: Extract facts
    facts = extract_facts(text)
    print(f"Extracted {len(facts)} facts")

    # Step 2: Check each fact
    fact_results = []
    for i, fact in enumerate(facts):
        print(f"Checking fact {i+1}/{len(facts)}...")
        result = check_fact(fact)
        fact_results.append(result)

    # Step 3: Aggregate
    score = aggregate_factuality_score(fact_results)

    # Detailed report
    report = {
        "overall_score": score,
        "total_facts": len(facts),
        "correct_facts": sum(1 for r in fact_results if r['verdict'] == 'correct'),
        "details": [
            {
                "fact": fact,
                "verdict": result['verdict'],
                "rationale": result['rationale']
            }
            for fact, result in zip(facts, fact_results)
        ]
    }

    return report

# 사용
text = """
Teddy bears, first created in the 1920s, were named after
President Theodore Roosevelt after he proudly wanted to shoot
a captured bear on a hunting trip.
"""

report = evaluate_factuality(text)
print(json.dumps(report, indent=2))
```

**출력 예시:**

```json
{
  "overall_score": 0.5,
  "total_facts": 4,
  "correct_facts": 2,
  "details": [
    {
      "fact": "Teddy bears were first created in the 1920s",
      "verdict": "incorrect",
      "rationale": "Teddy bears were actually first created around 1902-1903, not in the 1920s."
    },
    {
      "fact": "Teddy bears were named after President Theodore Roosevelt",
      "verdict": "correct",
      "rationale": "This is correct. The toy was named after Theodore 'Teddy' Roosevelt."
    },
    {
      "fact": "The name came from a hunting trip incident",
      "verdict": "correct",
      "rationale": "Correct. The name originated from a 1902 hunting trip in Mississippi."
    },
    {
      "fact": "President Roosevelt proudly wanted to shoot a captured bear",
      "verdict": "incorrect",
      "rationale": "Incorrect. Roosevelt actually refused to shoot the bear, considering it unsportsmanlike."
    }
  ]
}
```

---

# 8. Agent Evaluation

## 8.1. Agent의 Inner Working

Lecture 7에서 배운 내용:
- ReAct framework
- Observe, Plan, Act

**Agent 구조:**

```python
def agent_loop(query):
    """
    Agentic workflow
    """
    state = {"query": query, "observations": []}

    while not is_goal_achieved(state):
        # 1. Observe
        observation = observe(state)
        state["observations"].append(observation)

        # 2. Plan
        plan = llm.plan(state)

        # 3. Act
        if plan["action_type"] == "tool_call":
            # Tool selection & execution
            tool_name = plan["tool_name"]
            tool_args = plan["tool_args"]

            result = execute_tool(tool_name, tool_args)
            state["tool_results"].append(result)
        elif plan["action_type"] == "final_answer":
            return plan["answer"]

    return state["answer"]
```

**Evaluation 질문:**

```
Agent가 실패하면 어디서 문제가 발생했나?
1. Tool selection?
2. Argument extraction?
3. Tool execution?
4. Result interpretation?
```

## 8.2. Tool Prediction Errors

**Tool call decomposition:**

```
User Query
    ↓
[1] Tool Selection + Argument Extraction
    ↓
[2] Tool Execution
    ↓
[3] Result Interpretation
    ↓
Final Answer
```

**Step 1에서의 오류: Tool Prediction Errors**

### Tool Router Error

**문제: 필요한 tool이 function list에 포함되지 않음**

**예시:**

```python
# Available tools
all_tools = [
    "find_teddy_bear",  # ✅ 필요한 tool
    "get_weather",
    "search_web",
    "calculate"
]

# Query
query = "Find a teddy bear near me"

# Tool router
selected_tools = tool_router.select(query, all_tools, max_tools=3)
# Output: ["get_weather", "search_web", "calculate"]
# ❌ find_teddy_bear가 선택되지 않음!

# Result: LLM cannot call the right tool
# → Punt (error response)
response = "Sorry, I cannot help with that."
```

**원인:**

```python
# Tool router의 recall 문제
# Recall: 필요한 tool을 선택하는 비율

Recall = (선택한 relevant tools) / (모든 relevant tools)

# Tool router가 recall이 낮으면
# → 필요한 tool을 놓침
```

**해결:**

```python
# 1. Tool router 개선
# - Retrieval model fine-tuning
# - Better embeddings
# - Query expansion

def improved_tool_router(query, all_tools, max_tools=5):
    """
    Recall-oriented tool router
    """
    # Query expansion
    expanded_queries = [
        query,
        llm.rephrase(query),
        llm.extract_intent(query)
    ]

    # Retrieve for each query
    candidates = set()
    for q in expanded_queries:
        tools = retrieve_tools(q, all_tools, top_k=3)
        candidates.update(tools)

    # Re-rank
    ranked_tools = rerank(query, list(candidates))

    return ranked_tools[:max_tools]
```

### LLM이 Tool을 사용하지 않는 경우

**문제: Tool이 function list에 있지만 LLM이 사용하지 않음**

**예시:**

```python
# Available tools (이미 router에서 선택됨)
available_tools = [
    {
        "name": "find_teddy_bear",
        "description": "Find teddy bear stores near a location",
        "parameters": {"location": "string"}
    }
]

# Query
query = "Find a teddy bear near me"

# LLM response
response = llm.generate(query, tools=available_tools)
# Output: "You can try looking at toy stores nearby."
# ❌ find_teddy_bear tool을 호출하지 않음!
```

**원인 & 해결:**

**1. LLM이 tool 사용법을 모름**

```python
# 해결: Supervised Fine-Tuning (SFT)

# Training data
sft_examples = [
    {
        "input": "Find a teddy bear near me",
        "output": {
            "tool": "find_teddy_bear",
            "args": {"location": "user_location"}
        }
    },
    # More examples...
]

# Fine-tune
fine_tune(llm, sft_examples)
```

**2. Prompt가 불명확**

```python
# ❌ Bad prompt
prompt = f"""
Here are some tools: {tools}

User: {query}
"""

# ✅ Good prompt
prompt = f"""
You are an assistant with access to tools.

Available tools:
{format_tools(tools)}

IMPORTANT: You MUST use available tools to answer the user's question.
Do NOT provide answers without using tools.

User: {query}

Think step-by-step:
1. What tool should I use?
2. What arguments do I need?
3. Call the tool

Tool call:
"""
```

**3. Few-shot examples 부족**

```python
# ✅ In-context learning
prompt = f"""
You have access to these tools:
{format_tools(tools)}

Here are examples of correct tool usage:

Example 1:
User: "What's the weather in Paris?"
Tool call: get_weather(location="Paris")

Example 2:
User: "Find a teddy bear near me"
Tool call: find_teddy_bear(location="user_location")

Now, handle this query:
User: {query}
Tool call:
"""
```

## 8.3. Tool Hallucination

**문제: LLM이 존재하지 않는 tool을 호출**

**예시:**

```python
# Available tools
available_tools = [
    "find_teddy_bear",  # ✅ Exists
    "get_weather"
]

# Query
query = "Find a bear near me"

# LLM response
response = llm.generate(query, tools=available_tools)
# Output: find_bear(location="user_location")
#         ^^^^^^^^ ❌ This tool doesn't exist!

# Actual API: find_teddy_bear
# LLM hallucinated: find_bear
```

**원인:**

**1. Model capacity 부족**

```python
# 약한 모델이 tool API를 정확히 follow하지 못함
# → 비슷한 이름을 만들어냄

# 해결: 더 강력한 모델 사용
weak_model = "gpt-3.5-turbo"   # May hallucinate
strong_model = "gpt-4-turbo"   # Less likely to hallucinate

response = strong_model.generate(query, tools=available_tools)
```

**2. Tool API가 불명확**

```python
# ❌ Bad tool API
{
    "name": "ftb",  # Unclear abbreviation
    "description": "finds bears",  # Vague
    "parameters": {"loc": "string"}  # Unclear param name
}

# ✅ Good tool API
{
    "name": "find_teddy_bear",  # Clear, descriptive
    "description": "Searches for teddy bear stores near a given location",  # Detailed
    "parameters": {
        "location": {
            "type": "string",
            "description": "The location to search near (e.g., 'New York', 'user_location')"
        }
    }
}
```

**3. Global instructions 불명확**

```python
# ❌ Bad global instructions
system_prompt = "You can call functions."

# ✅ Good global instructions
system_prompt = """
You are an assistant with access to specific tools.

CRITICAL RULES:
1. You MUST ONLY use tools that are explicitly provided
2. Do NOT make up tool names or modify existing ones
3. Follow the exact API specification for each tool
4. If a suitable tool doesn't exist, say "I don't have a tool for that"

Available tools:
{format_tools(available_tools)}
"""
```

**Tool hallucination 감지:**

```python
def validate_tool_call(tool_call, available_tools):
    """
    Validate that tool call uses existing tools
    """
    tool_name = tool_call["name"]
    tool_args = tool_call["arguments"]

    # Check if tool exists
    if tool_name not in [t["name"] for t in available_tools]:
        return {
            "valid": False,
            "error": f"Tool '{tool_name}' does not exist",
            "suggestion": find_similar_tool(tool_name, available_tools)
        }

    # Check arguments
    tool_spec = get_tool_spec(tool_name, available_tools)
    required_args = tool_spec["parameters"]["required"]

    missing_args = set(required_args) - set(tool_args.keys())
    if missing_args:
        return {
            "valid": False,
            "error": f"Missing required arguments: {missing_args}"
        }

    return {"valid": True}

# 사용
tool_call = {"name": "find_bear", "arguments": {"location": "NYC"}}
validation = validate_tool_call(tool_call, available_tools)

if not validation["valid"]:
    print(f"❌ Invalid tool call: {validation['error']}")
    if "suggestion" in validation:
        print(f"Did you mean: {validation['suggestion']}?")
```

**Agent Evaluation 요약:**

| Error Type | Description | Solution |
|------------|-------------|----------|
| Tool Router Error | 필요한 tool 미선택 | Improve tool router (recall) |
| LLM doesn't use tool | Tool 있지만 사용 안함 | SFT, better prompts, examples |
| Tool Hallucination | 존재하지 않는 tool 호출 | Strong model, clear APIs |

---

# 9. 요약

## 핵심 개념

**Evaluation의 진화:**

```
Human Evaluation (Ideal but impractical)
    ↓
Rule-Based Metrics (Fast but limited)
    ↓
LLM-as-a-Judge (Best of both worlds)
```

**LLM-as-a-Judge의 핵심:**

1. **Reference 불필요**
   - LLM의 내재된 지식 활용

2. **Interpretable**
   - Rationale 제공

3. **Flexible**
   - 다양한 criteria 평가 가능

4. **Scalable**
   - 대규모 평가 가능

**주의사항:**

1. **Biases 존재**
   - Position, Verbosity, Self-Enhancement

2. **Human calibration 필요**
   - Correlation 확인

3. **Best practices 중요**
   - Binary scale, Rationale first, Low temperature

## Evaluation 방법 비교

| 방법 | 장점 | 단점 | 사용 시기 |
|------|------|------|-----------|
| Human Evaluation | - Ground truth<br>- 가장 정확 | - 느림<br>- 비쌈<br>- 확장 불가 | - Calibration<br>- 최종 검증 |
| Rule-Based (BLEU, METEOR) | - 빠름<br>- 재현 가능<br>- 비용 없음 | - Reference 필요<br>- Stylistic variation 불가<br>- 낮은 correlation | - 번역/요약<br>- 간단한 benchmark |
| LLM-as-a-Judge | - Reference 불필요<br>- 확장 가능<br>- Interpretable<br>- Flexible | - Biases 존재<br>- API 비용<br>- Calibration 필요 | - 대부분의 경우<br>- 현대적 LLM 평가 |

**언제 무엇을 사용하나:**

```python
# 1. 초기 개발 단계
# → LLM-as-a-Judge (빠른 iteration)

# 2. 모델 비교
# → LLM-as-a-Judge pairwise

# 3. 최종 검증
# → Human Evaluation (소규모)

# 4. 지속적 모니터링
# → LLM-as-a-Judge + periodic human calibration

# 5. Benchmark 제출
# → Rule-based metrics (BLEU 등) + LLM-as-a-Judge
```

## 실전 체크리스트

**LLM-as-a-Judge 구현 체크리스트:**

```python
evaluation_checklist = {
    "Prompt Design": [
        "✓ 명확한 criteria 정의",
        "✓ 구체적인 pass/fail 예시 포함",
        "✓ Binary scale 사용",
        "✓ Rationale을 먼저 요청"
    ],

    "Technical Setup": [
        "✓ Structured outputs 사용",
        "✓ Low temperature (0.1-0.2)",
        "✓ 생성 모델과 다른 judge 모델 사용",
        "✓ Error handling 구현"
    ],

    "Bias Mitigation": [
        "✓ Position bias: swapping 구현",
        "✓ Verbosity bias: guidelines에 명시",
        "✓ Self-enhancement: 다른 모델 사용"
    ],

    "Validation": [
        "✓ Human ratings 수집 (100+ examples)",
        "✓ Correlation 분석 (target: >0.7)",
        "✓ Inter-rater agreement 측정",
        "✓ 정기적 calibration"
    ],

    "Factuality (특별 처리)": [
        "✓ Fact extraction 단계",
        "✓ Fact checking with RAG/Web search",
        "✓ Weighted aggregation"
    ]
}
```

**Agent Evaluation 체크리스트:**

```python
agent_eval_checklist = {
    "Tool Selection": [
        "✓ Tool router recall 측정",
        "✓ 필요한 tool이 선택되는지 확인",
        "✓ Tool router 개선 (필요 시)"
    ],

    "Tool Usage": [
        "✓ LLM이 tool을 실제로 사용하는지 확인",
        "✓ SFT 데이터 준비 (필요 시)",
        "✓ Few-shot examples 포함"
    ],

    "Tool Hallucination": [
        "✓ Tool call validation 구현",
        "✓ Clear tool API 작성",
        "✓ Global instructions 명확화",
        "✓ Strong model 사용"
    ],

    "Error Analysis": [
        "✓ 실패 사례 분류",
        "✓ 병목 구간 식별",
        "✓ Systematic improvement"
    ]
}
```

**Production Deployment:**

```python
class ProductionEvaluator:
    """
    Production-ready LLM-as-a-Judge
    """

    def __init__(self, judge_model, criteria_config, calibration_data=None):
        self.judge_model = judge_model
        self.criteria_config = criteria_config
        self.calibration_data = calibration_data

        # Calibration
        if calibration_data:
            self.correlation = self.run_calibration()
            if self.correlation < 0.7:
                warnings.warn("Low correlation with human ratings!")

    def evaluate(self, prompt, response, criteria):
        """
        Evaluate with bias mitigation
        """
        # Main evaluation
        result = self._evaluate_once(prompt, response, criteria)

        # Confidence score
        confidence = self._estimate_confidence(result)

        # Flag low confidence cases for human review
        if confidence < 0.8:
            self._flag_for_human_review(prompt, response, result)

        return result

    def evaluate_pairwise(self, prompt, response_a, response_b, criteria):
        """
        Pairwise evaluation with position bias mitigation
        """
        # Evaluate both orders
        result_ab = self._evaluate_once(prompt, response_a, response_b, criteria, order="AB")
        result_ba = self._evaluate_once(prompt, response_b, response_a, criteria, order="BA")

        # Check consistency
        if result_ab["winner"] == result_ba["winner"]:
            return result_ab  # Consistent
        else:
            # Inconsistent: Flag for review
            return {
                "winner": "uncertain",
                "reason": "Position bias detected",
                "flag_for_review": True
            }

    def batch_evaluate(self, cases, parallel=True):
        """
        Batch evaluation with progress tracking
        """
        from tqdm import tqdm

        results = []
        for case in tqdm(cases):
            result = self.evaluate(
                case["prompt"],
                case["response"],
                case["criteria"]
            )
            results.append(result)

        return results

    def run_calibration(self):
        """
        Calibration with human ratings
        """
        judge_scores = []
        human_scores = []

        for case in self.calibration_data:
            judge_score = self.evaluate(
                case["prompt"],
                case["response"],
                case["criteria"]
            )
            judge_scores.append(judge_score["score"])
            human_scores.append(case["human_score"])

        from scipy.stats import pearsonr
        correlation, _ = pearsonr(human_scores, judge_scores)

        return correlation

# 사용
evaluator = ProductionEvaluator(
    judge_model=gpt4,
    criteria_config=load_criteria_config(),
    calibration_data=load_calibration_data()
)

# Single evaluation
result = evaluator.evaluate(prompt, response, "usefulness")

# Batch evaluation
results = evaluator.batch_evaluate(test_cases)

# Pairwise comparison
winner = evaluator.evaluate_pairwise(prompt, response_a, response_b, "quality")
```

---

# 10. 중요 용어 정리

**Evaluation 관련:**

- **LLM Evaluation**: LLM의 출력 품질을 정량적으로 측정하는 과정
- **Inter-Rater Agreement**: 여러 평가자 간의 평가 일치도
- **Cohen's Kappa**: 두 평가자 간 agreement를 random chance 대비 측정하는 metric
- **Fleiss's Kappa**: 3명 이상 평가자에 대한 Cohen's Kappa 확장
- **Krippendorff's Alpha**: Missing data를 허용하는 일반화된 agreement metric

**Rule-Based Metrics:**

- **METEOR**: Translation 평가를 위한 precision+recall 기반 metric (ordering 고려)
- **BLEU**: Translation 평가를 위한 n-gram precision 기반 metric
- **ROUGE**: Summarization 평가를 위한 recall 기반 metric
- **Brevity Penalty**: 짧은 문장에 대한 페널티 (BLEU에서 사용)
- **N-gram**: 연속된 n개의 단어 시퀀스

**LLM-as-a-Judge:**

- **LLM-as-a-Judge**: LLM을 평가자로 사용하는 평가 방법
- **Pointwise Evaluation**: 단일 응답의 절대적 품질 평가
- **Pairwise Evaluation**: 두 응답의 상대적 품질 비교
- **Rationale**: Judge가 제공하는 판단 근거 설명
- **Structured Outputs**: 정해진 형식으로 출력을 강제하는 기법

**Biases:**

- **Position Bias**: 먼저 제시된 응답을 선호하는 경향
- **Verbosity Bias**: 더 긴 응답을 선호하는 경향
- **Self-Enhancement Bias**: 자신이 생성한 응답을 선호하는 경향

**Evaluation Dimensions:**

- **Usefulness**: 응답이 사용자에게 도움이 되는 정도
- **Factuality**: 응답의 사실적 정확성
- **Relevance**: 응답이 질문과 관련된 정도
- **Coherence**: 응답의 논리적 일관성
- **Safety**: 응답의 안전성 및 윤리성

**Factuality Evaluation:**

- **Fact Extraction**: 텍스트에서 개별 fact를 추출하는 과정
- **Fact Checking**: 개별 fact의 사실 여부를 검증하는 과정
- **Atomic Fact**: 더 이상 분해할 수 없는 하나의 사실적 주장

**Agent Evaluation:**

- **Tool Router**: 주어진 쿼리에 적합한 tool을 선택하는 시스템
- **Tool Hallucination**: 존재하지 않는 tool을 호출하는 오류
- **Punt**: Agent가 작업을 수행하지 못하고 포기하는 것
- **Recall (Tool Selection)**: 필요한 tool을 얼마나 잘 선택하는지의 비율

**Best Practices:**

- **Calibration**: Judge의 평가를 human ratings와 비교하여 조정
- **Temperature**: 생성 과정의 randomness를 제어하는 파라미터
- **Binary Scale**: Pass/Fail과 같은 이진 평가 척도
- **Correlation**: 두 변수 간의 선형 관계 강도 (Judge vs Human)

---

**다음 강의 예고:**

Lecture 9에서는 Current Trends를 다룹니다. LLM 분야의 최신 동향과 미래 방향을 살펴봅니다.

---

**수고하셨습니다!** 🎉
