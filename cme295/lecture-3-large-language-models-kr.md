# Lecture 3: Large Language Models

# Materials

- [CME 295](https://cme295.stanford.edu/syllabus/)
- [slide](https://cme295.stanford.edu/slides/fall25-cme295-lecture3.pdf)
- [video](https://www.youtube.com/watch?v=Q5baLehv5So&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=3)

# Table of Contents

- [Lecture 3: Large Language Models](#lecture-3-large-language-models)
- [Materials](#materials)
- [Table of Contents](#table-of-contents)
- [강의 개요](#강의-개요)
  - [강의 목표](#강의-목표)
  - [주요 학습 내용](#주요-학습-내용)
- [1. Large Language Models (LLM) 정의](#1-large-language-models-llm-정의)
  - [1.1. LLM이란 무엇인가?](#11-llm이란-무엇인가)
  - [1.2. LLM의 특징](#12-llm의-특징)
    - [1. 모델 크기 (Model Size)](#1-모델-크기-model-size)
    - [2. 학습 데이터 크기 (Training Data)](#2-학습-데이터-크기-training-data)
    - [3. 컴퓨트 요구사항 (Compute Requirements)](#3-컴퓨트-요구사항-compute-requirements)
  - [1.3. LLM vs 기존 모델](#13-llm-vs-기존-모델)
- [2. Mixture of Experts (MoE)](#2-mixture-of-experts-moe)
  - [2.1. MoE의 핵심 아이디어](#21-moe의-핵심-아이디어)
  - [2.2. Dense MoE vs Sparse MoE](#22-dense-moe-vs-sparse-moe)
    - [Dense MoE](#dense-moe)
    - [Sparse MoE](#sparse-moe)
  - [2.3. LLM에서의 MoE 적용](#23-llm에서의-moe-적용)
  - [2.4. Routing Collapse 문제](#24-routing-collapse-문제)
  - [2.5. MoE 시각화](#25-moe-시각화)
- [3. Response Generation (응답 생성)](#3-response-generation-응답-생성)
  - [3.1. 응답 생성 개요](#31-응답-생성-개요)
  - [3.2. Greedy Decoding](#32-greedy-decoding)
  - [3.3. Beam Search](#33-beam-search)
  - [3.4. Sampling-based Methods](#34-sampling-based-methods)
    - [기본 Sampling](#기본-sampling)
    - [Top-k Sampling](#top-k-sampling)
    - [Top-p (Nucleus) Sampling](#top-p-nucleus-sampling)
  - [3.5. Temperature의 영향](#35-temperature의-영향)
- [4. Guided Decoding](#4-guided-decoding)
  - [4.1. Guided Decoding이란?](#41-guided-decoding이란)
  - [4.2. JSON 생성 예시](#42-json-생성-예시)
- [5. Context Length와 실전 고려사항](#5-context-length와-실전-고려사항)
  - [5.1. Context Length](#51-context-length)
  - [5.2. Context Rot 현상](#52-context-rot-현상)
- [6. 요약](#6-요약)
  - [핵심 개념](#핵심-개념)
    - [1. LLM (Large Language Models)](#1-llm-large-language-models)
    - [2. Mixture of Experts (MoE)](#2-mixture-of-experts-moe-1)
    - [3. Response Generation](#3-response-generation)
  - [Response Generation 비교](#response-generation-비교)
  - [실전 구현 체크리스트](#실전-구현-체크리스트)
    - [MoE 구현](#moe-구현)
    - [Response Generation 구현](#response-generation-구현)
    - [Context 관리](#context-관리)
    - [최적화](#최적화)
- [7. 중요 용어 정리](#7-중요-용어-정리)
  - [LLM 관련](#llm-관련)
  - [MoE 관련](#moe-관련)
  - [Response Generation 관련](#response-generation-관련)
  - [Temperature 관련](#temperature-관련)
  - [Context 관련](#context-관련)
  - [기타](#기타)

---

# 강의 개요

## 강의 목표

이번 강의에서는 Large Language Models (LLMs)의 핵심 개념과 실전 응용 기법을 학습합니다.

**학습 목표:**
- LLM의 정의와 특징 이해
- Mixture of Experts (MoE) 아키텍처 학습
- 다양한 응답 생성 전략 비교
- Temperature와 Sampling 기법 이해
- 실전 LLM 사용 시 고려사항 파악

## 주요 학습 내용

**1. LLM 정의**
- Language Model의 개념
- Large의 의미 (모델 크기, 데이터, 컴퓨트)
- Decoder-only 아키텍처

**2. Mixture of Experts**
- MoE의 핵심 아이디어
- Dense vs Sparse MoE
- Routing 메커니즘
- Training 시 고려사항

**3. Response Generation**
- Greedy Decoding
- Beam Search
- Sampling-based Methods
- Temperature 조절

**4. 실전 응용**
- Guided Decoding
- Context Length 관리
- Prompting Strategies

---

# 1. Large Language Models (LLM) 정의

## 1.1. LLM이란 무엇인가?

**LLM (Large Language Model)은 두 가지 특성을 가집니다:**

1. **Language Model**: 토큰 시퀀스에 확률을 할당하는 모델
2. **Large**: 규모가 매우 큰 모델

**Language Model의 정의:**

```
P(next_token | previous_tokens)
```

- 이전 토큰들이 주어졌을 때, 다음 토큰의 확률을 예측
- 모든 가능한 토큰에 대한 확률 분포 생성

**예시:**

```
입력: "A cute teddy"
모델 출력:
- "bear": 0.8
- "toy": 0.1
- "animal": 0.05
- "dog": 0.03
- ...
```

## 1.2. LLM의 특징

**LLM이 "Large"인 이유:**

### 1. 모델 크기 (Model Size)

```
규모별 분류:
- Small: ~1B parameters
- Medium: ~7B parameters
- Large: ~70B parameters
- Very Large: 100B+ parameters (GPT-3: 175B, GPT-4: ~1.7T 추정)
```

**예시: GPT-3**
- 파라미터: 175 billion
- 레이어: 96 layers
- Attention heads: 96
- d_model: 12,288

### 2. 학습 데이터 크기 (Training Data)

```
데이터 규모:
- 수백억 ~ 수조 개의 토큰
- 다양한 소스: 웹 텍스트, 책, 코드, 위키피디아 등

예시:
- GPT-3: ~300B tokens
- LLaMA: ~1.4T tokens
- LLaMA 2: ~2T tokens
```

### 3. 컴퓨트 요구사항 (Compute Requirements)

```
학습:
- 수천 개의 GPU
- 수주 ~ 수개월의 학습 시간
- 수백만 ~ 수천만 달러 비용

추론:
- 최소 수 GB ~ 수백 GB GPU 메모리
- 최적화 기법으로 consumer GPU에서도 가능
```

## 1.3. LLM vs 기존 모델

**역사적 맥락:**

```
2018-2019:
- BERT 등장 (Encoder-only)
- "LLM"이라는 용어가 명확하지 않음
- BERT도 LLM으로 분류되기도 함

2020-현재:
- GPT-3 이후 정의가 명확해짐
- LLM = Decoder-only + Text Generation + Large Scale
- BERT는 더 이상 LLM으로 분류되지 않음 (생성 불가)
```

**현대 LLM의 정의:**

| 조건 | 설명 |
|------|------|
| Architecture | Decoder-only (causal masking) |
| Task | Text generation (text-to-text) |
| Size | ≥ 1B parameters |
| Training Data | 수백억 개 이상의 토큰 |
| Capability | Next token prediction |

**주요 LLM 예시:**

- **GPT Family**: GPT-3, GPT-3.5, GPT-4
- **LLaMA**: LLaMA, LLaMA 2, LLaMA 3
- **Google**: Gemma, PaLM
- **Others**: Mistral, Mixtral, DeepSeek, Qwen, Falcon

**통계:**

```
현대 LLM의 90% 이상이 Decoder-only 아키텍처 사용
```

---

# 2. Mixture of Experts (MoE)

## 2.1. MoE의 핵심 아이디어

**문제 제기:**

수백억 개의 파라미터를 가진 LLM에서, 매 forward pass마다 모든 파라미터를 활성화해야 할까요?

**비유:**

```
상황: 당신이 방에 들어갑니다.
방 안에는:
- 수학자 (Mathematician)
- 물리학자 (Physicist)
- 화학자 (Chemist)
- 역사학자 (Historian)

질문: 수학 문제가 있습니다. 누구에게 물어볼까요?

옵션 A: 모든 사람에게 물어본다 (현재 방식)
옵션 B: 수학자에게만 물어본다 (MoE 방식)
```

**MoE의 핵심:**

입력에 따라 모델의 **일부 파라미터만** 활성화하여 효율성을 높입니다.

**구조:**

```
입력 x
  ↓
Gate/Router G(x) ← 어떤 experts를 사용할지 결정
  ↓
┌─────────┬─────────┬─────────┬─────────┐
│Expert 1 │Expert 2 │Expert 3 │Expert 4 │
└─────────┴─────────┴─────────┴─────────┘
     ↓
선택된 expert만 활성화
     ↓
출력 y
```

**수식:**

```
ŷ = Σ(i=1 to n) G(x)ᵢ · Eᵢ(x)

여기서:
- n: expert 개수
- G(x)ᵢ: gate의 출력 (expert i의 가중치)
- Eᵢ(x): expert i의 출력
```

## 2.2. Dense MoE vs Sparse MoE

### Dense MoE

**특징:**

- 모든 expert의 출력을 가중합
- Gate 가중치가 0과 1 사이의 연속값
- 실질적으로 모든 expert 활성화

**수식:**

```
ŷ = Σ(i=1 to n) G(x)ᵢ · Eᵢ(x)

여기서 G(x)ᵢ는 확률 분포:
Σ G(x)ᵢ = 1
0 ≤ G(x)ᵢ ≤ 1
```

**예시:**

```
수학 문제에 대한 응답:
- 수학자: 0.7 (높은 가중치)
- 물리학자: 0.2
- 화학자: 0.05
- 역사학자: 0.05 (낮은 가중치)

최종 답변 = 0.7 × 수학자 답변 + 0.2 × 물리학자 답변 + ...
```

**장점:**
- 모든 expert의 지식 활용
- 부드러운 전환 (smooth transition)

**단점:**
- 계산 비용 여전히 높음
- 메모리 효율성 낮음

### Sparse MoE

**특징:**

- Top-k expert만 선택하여 활성화
- k는 hyperparameter (보통 k=1 또는 k=2)
- 나머지 expert는 완전히 비활성화

**수식:**

```
ŷ = Σ(i ∈ Top-k) G(x)ᵢ · Eᵢ(x)

여기서 Top-k는 G(x)ᵢ가 가장 높은 k개의 expert
```

**예시:**

```
k=1인 경우:
- 수학자: 0.7 ← 선택됨!
- 물리학자: 0.2
- 화학자: 0.05
- 역사학자: 0.05

최종 답변 = 수학자 답변만 사용
```

**장점:**
- 계산 비용 대폭 감소
- 메모리 효율적
- 추론 속도 향상

**단점:**
- 일부 expert 지식 손실
- Training이 더 복잡

**FLOPS 비교:**

```
FLOPS (Floating-Point Operations):
forward pass에서 필요한 연산 수를 나타내는 단위

Dense MoE: n × (expert 연산 비용)
Sparse MoE (k=1): 1 × (expert 연산 비용)

예시: n=8, k=1
→ 8배 연산량 감소!
```

## 2.3. LLM에서의 MoE 적용

**질문: LLM의 어디에 MoE를 적용할까요?**

**Decoder Block 구조:**

```
입력 x
  ↓
┌─────────────────────┐
│ Masked Self-Attention│
└─────────────────────┘
  ↓
┌─────────────────────┐
│ Feedforward Network │ ← MoE는 여기에 적용!
└─────────────────────┘
  ↓
┌─────────────────────┐
│ Layer Normalization │
└─────────────────────┘
  ↓
출력
```

**왜 Feedforward에 적용하나요?**

**파라미터 수 비교:**

```
Attention Layer:
- 파라미터: O(d_model × d_k × num_heads × 4)
- 일반적으로 d_k는 작음 (64 ~ 128)

Feedforward Layer:
- 파라미터: O(d_model × d_ff × 2)
- d_ff는 매우 큼 (4 × d_model)

예시: d_model=4096, d_ff=16384, d_k=128, num_heads=32
Attention: 4096 × 128 × 32 × 4 ≈ 67M
Feedforward: 4096 × 16384 × 2 ≈ 134M

→ Feedforward가 2배 더 큼!
```

**MoE Feedforward 구조:**

```python
class MoEFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, k=1):
        super().__init__()
        self.num_experts = num_experts
        self.k = k

        # Gate/Router
        self.gate = nn.Linear(d_model, num_experts)

        # Experts (각각 독립적인 feedforward network)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape

        # Gate 계산: (batch, seq_len, num_experts)
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Top-k expert 선택
        top_k_probs, top_k_indices = torch.topk(
            gate_probs, self.k, dim=-1
        )

        # Sparse MoE: 선택된 expert만 계산
        output = torch.zeros_like(x)

        for i in range(self.k):
            expert_idx = top_k_indices[:, :, i]  # (batch, seq_len)
            expert_weight = top_k_probs[:, :, i].unsqueeze(-1)  # (batch, seq_len, 1)

            # 각 토큰을 해당 expert로 라우팅
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += expert_weight[mask] * expert_output

        return output
```

**토큰 레벨 라우팅:**

```
예시: "The cat sat on the mat"

Token | Layer 1 Expert | Layer 2 Expert | Layer 3 Expert
------|----------------|----------------|---------------
The   | Expert 2       | Expert 1       | Expert 3
cat   | Expert 1       | Expert 1       | Expert 2
sat   | Expert 3       | Expert 4       | Expert 1
on    | Expert 2       | Expert 2       | Expert 2
the   | Expert 2       | Expert 1       | Expert 3
mat   | Expert 1       | Expert 3       | Expert 1

→ 각 토큰이 각 레이어에서 다른 expert로 라우팅됨
```

## 2.4. Routing Collapse 문제

**문제:**

Training 중 일부 expert만 계속 선택되고, 나머지는 거의 사용되지 않는 현상

**예시:**

```
이상적인 경우 (8 experts):
Expert 1: 12.5% 사용
Expert 2: 12.5% 사용
...
Expert 8: 12.5% 사용

Routing Collapse 발생 시:
Expert 1: 60% 사용 ← 과다 사용!
Expert 2: 30% 사용 ← 과다 사용!
Expert 3: 5% 사용
Expert 4: 2% 사용
Expert 5: 1% 사용
...
Expert 8: 0.5% 사용 ← 거의 사용 안됨!
```

**해결책: Load Balancing Loss**

**추가 Loss Term:**

```
L_total = L_task + α × L_balance

여기서:
L_balance = n × Σ(i=1 to n) f(i) × P(i)

- n: expert 개수
- f(i): expert i로 라우팅된 토큰의 비율
- P(i): expert i의 평균 라우팅 확률
- α: hyperparameter (balancing weight)
```

**각 항의 의미:**

```python
# f(i): Fraction of tokens routed to expert i
f = torch.zeros(num_experts)
for token in batch:
    assigned_expert = argmax(gate(token))
    f[assigned_expert] += 1
f = f / total_tokens

# P(i): Average routing probability for expert i
P = torch.zeros(num_experts)
for token in batch:
    gate_probs = softmax(gate(token))
    P += gate_probs
P = P / total_tokens

# Load balancing loss
L_balance = num_experts * torch.sum(f * P)
```

**목표:**

```
이 loss는 f와 P가 uniform distribution에 가까워지도록 유도:

이상적인 경우:
f(i) = 1/n  (모든 i)
P(i) = 1/n  (모든 i)

이 경우:
L_balance = n × Σ (1/n × 1/n) = n × (n × 1/n²) = 1

Routing Collapse 시:
일부 f(i), P(i)가 크면 L_balance가 증가
→ 학습 시 이를 최소화하려고 함
→ uniform distribution으로 수렴
```

**기타 기법:**

**1. Noisy Gating:**

```python
def noisy_gate(x, noise_stddev=0.1):
    logits = gate_linear(x)

    # 학습 시에만 noise 추가
    if self.training:
        noise = torch.randn_like(logits) * noise_stddev
        logits = logits + noise

    return softmax(logits)
```

- Random noise를 추가하여 다른 expert도 선택될 기회 제공
- Dropout과 유사한 효과

**2. Expert Capacity:**

```python
# 각 expert가 처리할 수 있는 최대 토큰 수 제한
capacity = (total_tokens / num_experts) * capacity_factor

# capacity_factor > 1.0 (예: 1.25)
# 특정 expert로 토큰이 몰리는 것을 방지
```

**3. Auxiliary Losses:**

```python
# Expert usage를 균등하게 만드는 다양한 loss 추가
# - Entropy regularization
# - Diversity loss
# - Importance loss
```

## 2.5. MoE 시각화

**Mistral 논문의 시각화 예시:**

```
입력 텍스트: "In machine learning, gradient descent is a method..."

각 토큰의 expert 할당 (Layer 0):

Token      | Expert
-----------|--------
In         | ████ Expert 3
machine    | ██ Expert 1
learning   | ████ Expert 3
,          | ███ Expert 2
gradient   | ██████ Expert 5
descent    | ██████ Expert 5
is         | ███ Expert 2
a          | ████ Expert 3
method     | ████████ Expert 7
...        | ...

관찰:
- 비슷한 의미의 단어가 같은 expert로 가는 경향
- 예: "gradient", "descent" → Expert 5 (수학/최적화)
- 예: "machine", "learning", "method" → 다양한 expert
- Uniform하게 분포 (routing collapse 없음)
```

**모델별 MoE 사용 예시:**

| 모델 | 전체 파라미터 | Active 파라미터 | Experts | k |
|------|--------------|----------------|---------|---|
| Switch Transformer | 1.6T | 16B | 2048 | 1 |
| GLaM | 1.2T | 97B | 64 | 2 |
| Mixtral 8x7B | 47B | 13B | 8 | 2 |
| Mixtral 8x22B | 141B | 39B | 8 | 2 |

**해석:**

```
Mixtral 8x7B:
- 전체: 8개 experts × 7B parameters = 47B (실제로는 shared params 포함)
- Active: 2개 experts만 사용 = 13B parameters per forward pass
- 효율성: 13B 크기의 모델 속도로 47B 모델의 성능!
```

---

# 3. Response Generation (응답 생성)

## 3.1. 응답 생성 개요

**LLM의 기본 작동 방식:**

```
입력: "A cute teddy"
  ↓
LLM (Decoder)
  ↓
확률 분포:
{
  "bear": 0.65,
  "toy": 0.15,
  "animal": 0.08,
  "fluffy": 0.05,
  "doll": 0.03,
  "cat": 0.02,
  ...
}
  ↓
다음 토큰 선택: ???
```

**질문: 이 확률 분포에서 어떻게 다음 토큰을 선택할까요?**

## 3.2. Greedy Decoding

**가장 간단한 방법: 최고 확률 토큰 선택**

**알고리즘:**

```python
def greedy_decoding(model, input_tokens, max_length):
    generated = input_tokens.copy()

    for _ in range(max_length):
        # 모델 forward pass
        probs = model(generated)  # (vocab_size,)

        # 최고 확률 토큰 선택
        next_token = argmax(probs)

        generated.append(next_token)

        if next_token == EOS_TOKEN:
            break

    return generated
```

**예시:**

```
Step 1:
입력: "A cute teddy"
확률: {"bear": 0.65, "toy": 0.15, ...}
선택: "bear" (0.65)

Step 2:
입력: "A cute teddy bear"
확률: {"is": 0.7, "was": 0.2, ...}
선택: "is" (0.7)

Step 3:
입력: "A cute teddy bear is"
확률: {"sitting": 0.6, "standing": 0.3, ...}
선택: "sitting" (0.6)

최종: "A cute teddy bear is sitting"
```

**장점:**
- 구현이 매우 간단
- 빠른 추론 속도
- 결정론적 (deterministic)

**단점:**

**1. 다양성 부족:**
```
같은 입력 → 항상 같은 출력
ChatGPT가 매번 같은 답변만 하면 재미없겠죠?
```

**2. Locally Optimal, Not Globally Optimal:**

```
예시:

Path A (greedy):
"The" (0.8) → "cat" (0.3) → "meowed" (0.2)
전체 확률: 0.8 × 0.3 × 0.2 = 0.048

Path B:
"A" (0.7) → "dog" (0.8) → "barked" (0.9)
전체 확률: 0.7 × 0.8 × 0.9 = 0.504

→ Greedy는 Path A를 선택하지만, Path B가 더 높은 확률!
```

## 3.3. Beam Search

**핵심 아이디어: k개의 가장 가능성 높은 경로를 추적**

**알고리즘:**

```python
def beam_search(model, input_tokens, max_length, beam_width=3):
    # 초기: [(tokens, log_prob)]
    beams = [(input_tokens, 0.0)]

    for _ in range(max_length):
        candidates = []

        # 각 beam에 대해
        for tokens, log_prob in beams:
            # 다음 토큰 확률 계산
            probs = model(tokens)

            # Top-k 토큰 선택
            top_k_probs, top_k_tokens = torch.topk(probs, beam_width)

            # 새로운 candidate 생성
            for prob, token in zip(top_k_probs, top_k_tokens):
                new_tokens = tokens + [token]
                new_log_prob = log_prob + math.log(prob)
                candidates.append((new_tokens, new_log_prob))

        # 전체 candidate 중 top beam_width개 선택
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

    # 최고 확률의 beam 반환
    return beams[0][0]
```

**시각화 예시 (beam_width=2):**

```
Step 0:
[BOS]

Step 1:
┌─── "The" (0.8)
└─── "A" (0.7)

Step 2:
"The" ─┬─── "cat" (0.8 × 0.6 = 0.48)
       └─── "dog" (0.8 × 0.4 = 0.32)
"A" ───┬─── "cute" (0.7 × 0.7 = 0.49)  ← 최고!
       └─── "big" (0.7 × 0.5 = 0.35)

Top 2 유지:
- "A cute" (0.49)
- "The cat" (0.48)

Step 3:
"A cute" ──┬─── "teddy" (0.49 × 0.8 = 0.392)  ← 최고!
           └─── "puppy" (0.49 × 0.5 = 0.245)
"The cat" ─┬─── "sat" (0.48 × 0.7 = 0.336)
           └─── "ran" (0.48 × 0.4 = 0.192)

Top 2 유지:
- "A cute teddy" (0.392)
- "The cat sat" (0.336)

최종 선택: "A cute teddy" (계속...)
```

**확률 계산:**

```python
# 시퀀스 확률 = 각 토큰 확률의 곱
P(sequence) = P(t1) × P(t2|t1) × P(t3|t1,t2) × ...

# Log 공간에서 계산 (수치 안정성)
log P(sequence) = log P(t1) + log P(t2|t1) + log P(t3|t1,t2) + ...

# Length Normalization (긴 시퀀스 페널티 방지)
normalized_score = log P(sequence) / length^α

여기서 α는 hyperparameter (보통 0.6 ~ 1.0)
```

**Length Normalization의 필요성:**

```
문제: 확률의 곱은 시퀀스가 길어질수록 작아짐

예시:
Short: "Hi" = 0.5 → 짧지만 높은 확률
Long: "Hello, how are you today?" = 0.5^6 = 0.016 → 길지만 낮은 확률

해결: length로 normalize
Short: 0.5 / 1 = 0.5
Long: (0.5^6) / 6 = 0.0026 (여전히 낮지만 덜 불공평)

α < 1.0이면 긴 시퀀스에 더 관대함
```

**장점:**
- Greedy보다 더 globally optimal
- 다양한 경로 탐색
- 기계 번역 등에서 좋은 성능

**단점:**
- 계산 비용 높음 (beam_width배)
- 메모리 사용량 증가
- 여전히 deterministic (같은 출력)
- 창의성 부족

**사용 사례:**
- 기계 번역 (Machine Translation)
- 요약 (Summarization)
- 정확성이 중요한 작업

## 3.4. Sampling-based Methods

**핵심 아이디어: 확률 분포에서 샘플링**

### 기본 Sampling

**알고리즘:**

```python
def sampling_decoding(model, input_tokens, max_length):
    generated = input_tokens.copy()

    for _ in range(max_length):
        # 확률 분포 계산
        probs = model(generated)  # (vocab_size,)

        # 확률 분포에서 샘플링
        next_token = torch.multinomial(probs, num_samples=1)

        generated.append(next_token)

        if next_token == EOS_TOKEN:
            break

    return generated
```

**예시:**

```
확률 분포:
{
  "bear": 0.65,
  "toy": 0.15,
  "animal": 0.08,
  "fluffy": 0.05,
  "doll": 0.03,
  "cat": 0.02,
  "airplane": 0.01,  ← 낮은 확률이지만 선택 가능!
  ...
}

Sample 1: "bear" (높은 확률로 선택됨)
Sample 2: "bear" (다시 선택될 수 있음)
Sample 3: "toy" (다른 토큰도 선택 가능)
Sample 4: "airplane" (낮은 확률이지만 가끔 선택됨)
```

**장점:**
- 다양한 출력 생성 가능
- 창의적인 응답
- Non-deterministic

**단점:**
- 낮은 확률 토큰도 선택될 수 있음
- 품질이 불안정할 수 있음

### Top-k Sampling

**핵심: 상위 k개 토큰만 고려**

**알고리즘:**

```python
def top_k_sampling(model, input_tokens, max_length, k=50):
    generated = input_tokens.copy()

    for _ in range(max_length):
        probs = model(generated)

        # Top-k 토큰만 선택
        top_k_probs, top_k_indices = torch.topk(probs, k)

        # Top-k 내에서 재정규화
        top_k_probs = top_k_probs / top_k_probs.sum()

        # Top-k 중에서 샘플링
        sample_idx = torch.multinomial(top_k_probs, num_samples=1)
        next_token = top_k_indices[sample_idx]

        generated.append(next_token)

        if next_token == EOS_TOKEN:
            break

    return generated
```

**예시 (k=4):**

```
원본 확률:
{
  "bear": 0.65,
  "toy": 0.15,
  "animal": 0.08,
  "fluffy": 0.05,  ← Top-4 마지막
  "doll": 0.03,     ← 제외됨
  "cat": 0.02,      ← 제외됨
  "airplane": 0.01, ← 제외됨
  ...
}

Top-4 후 재정규화:
{
  "bear": 0.65 / 0.93 = 0.699,
  "toy": 0.15 / 0.93 = 0.161,
  "animal": 0.08 / 0.93 = 0.086,
  "fluffy": 0.05 / 0.93 = 0.054
}
(합계 = 1.0)

샘플링은 이 4개 중에서만 수행
```

**장점:**
- 낮은 확률 토큰 제외
- 품질과 다양성의 균형
- 구현 간단

**단점:**
- 고정된 k 값
- 상황에 따라 최적의 k가 다를 수 있음

**k 값의 영향:**

```
k=1: Greedy와 동일
k=10: 매우 제한적, 안전한 선택
k=50: 적절한 다양성
k=100: 더 창의적
k=vocab_size: 전체 샘플링과 동일
```

### Top-p (Nucleus) Sampling

**핵심: 누적 확률이 p를 넘는 최소 토큰 집합 선택**

**알고리즘:**

```python
def top_p_sampling(model, input_tokens, max_length, p=0.9):
    generated = input_tokens.copy()

    for _ in range(max_length):
        probs = model(generated)

        # 확률 내림차순 정렬
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # 누적 확률 계산
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # p를 초과하는 첫 index 찾기
        cutoff_index = (cumulative_probs > p).nonzero()[0].item()

        # Nucleus 선택
        nucleus_probs = sorted_probs[:cutoff_index+1]
        nucleus_indices = sorted_indices[:cutoff_index+1]

        # 재정규화
        nucleus_probs = nucleus_probs / nucleus_probs.sum()

        # Nucleus 내에서 샘플링
        sample_idx = torch.multinomial(nucleus_probs, num_samples=1)
        next_token = nucleus_indices[sample_idx]

        generated.append(next_token)

        if next_token == EOS_TOKEN:
            break

    return generated
```

**예시 (p=0.9):**

```
확률 (정렬됨):
{
  "bear": 0.65,     누적: 0.65
  "toy": 0.15,      누적: 0.80
  "animal": 0.08,   누적: 0.88
  "fluffy": 0.05,   누적: 0.93  ← p=0.9 초과!
  "doll": 0.03,     제외
  "cat": 0.02,      제외
  ...
}

Nucleus (상위 4개):
{
  "bear": 0.65 / 0.93 = 0.699,
  "toy": 0.15 / 0.93 = 0.161,
  "animal": 0.08 / 0.93 = 0.086,
  "fluffy": 0.05 / 0.93 = 0.054
}
```

**Top-k vs Top-p 비교:**

```
시나리오 1: 확률이 매우 집중됨
{
  "bear": 0.9,
  "toy": 0.05,
  "animal": 0.03,
  ...
}

Top-k (k=5): 5개 토큰 선택 (불필요하게 많음)
Top-p (p=0.9): 1개 토큰만 선택 (적절함)

시나리오 2: 확률이 고르게 분포
{
  "bear": 0.2,
  "toy": 0.18,
  "animal": 0.15,
  "fluffy": 0.12,
  "doll": 0.1,
  ...
}

Top-k (k=3): 3개 토큰 선택 (너무 적음)
Top-p (p=0.9): 6개 토큰 선택 (적절함)

→ Top-p가 더 adaptive!
```

**장점:**
- 상황에 따라 adaptive하게 조절
- Top-k보다 유연함
- 품질과 다양성의 균형

**단점:**
- Top-k보다 구현 복잡
- p 값 튜닝 필요

**실전 추천 값:**

```
p=0.9: 균형잡힌 선택 (가장 일반적)
p=0.95: 더 안전한 선택
p=0.8: 더 창의적
p=1.0: 전체 샘플링 (매우 창의적, 불안정)
```

## 3.5. Temperature의 영향

**Temperature (T)는 Softmax에서 사용되는 hyperparameter**

**Softmax 공식:**

```
P(token_i | context) = exp(x_i / T) / Σⱼ exp(x_j / T)

여기서:
- x_i: 토큰 i의 logit (모델 출력)
- T: temperature (> 0)
- Σⱼ: 모든 토큰에 대한 합
```

**Temperature의 수학적 분석:**

**Case 1: T → 0 (낮은 temperature)**

```
가장 높은 logit을 x_k라고 하면:

P(token_i) = exp(x_i / T) / Σⱼ exp(x_j / T)

분자와 분모에 exp(-x_k / T)를 곱하면:

P(token_i) = exp((x_i - x_k) / T) / Σⱼ exp((x_j - x_k) / T)

T → 0일 때:
- i = k인 경우: (x_k - x_k) / T = 0 / T = 0
  → exp(0) = 1

- i ≠ k인 경우: (x_i - x_k) / T → -∞ (x_i < x_k이므로)
  → exp(-∞) = 0

결과: P(token_k) = 1, P(token_i≠k) = 0
→ 가장 높은 확률의 토큰만 선택 (Greedy와 동일!)
```

**Case 2: T → ∞ (높은 temperature)**

```
T → ∞일 때:
x_i / T → 0 (모든 i)

exp(0) = 1

P(token_i) = 1 / Σⱼ 1 = 1 / vocab_size

→ Uniform distribution!
```

**시각화:**

```
원본 logits: [5.0, 3.0, 1.0, 0.5, 0.1]

T=0.5 (낮은 temperature) - 매우 뾰족한 분포:
Probs: [0.952, 0.043, 0.003, 0.001, 0.001]
         ████
         █

T=1.0 (기본) - 중간 분포:
Probs: [0.705, 0.191, 0.051, 0.031, 0.022]
         ███
         █

T=2.0 (높은 temperature) - 평평한 분포:
Probs: [0.468, 0.253, 0.138, 0.097, 0.044]
         ██
         █
         █
```

**구현:**

```python
def softmax_with_temperature(logits, temperature=1.0):
    """
    Temperature를 적용한 Softmax

    Args:
        logits: (vocab_size,) 모델 출력 logits
        temperature: float, temperature 값

    Returns:
        probs: (vocab_size,) 확률 분포
    """
    # Temperature scaling
    scaled_logits = logits / temperature

    # Softmax (수치 안정성을 위해 max 값 빼기)
    scaled_logits = scaled_logits - scaled_logits.max()
    exp_logits = torch.exp(scaled_logits)
    probs = exp_logits / exp_logits.sum()

    return probs

# 예시
logits = torch.tensor([5.0, 3.0, 1.0, 0.5, 0.1])

print("T=0.1:", softmax_with_temperature(logits, 0.1))
# T=0.1: [0.999, 0.001, 0.000, 0.000, 0.000]

print("T=1.0:", softmax_with_temperature(logits, 1.0))
# T=1.0: [0.705, 0.191, 0.051, 0.031, 0.022]

print("T=2.0:", softmax_with_temperature(logits, 2.0))
# T=2.0: [0.468, 0.253, 0.138, 0.097, 0.044]
```

**실전 사용:**

```python
def generate_with_temperature(model, prompt, temperature=1.0, max_length=50):
    tokens = tokenize(prompt)

    for _ in range(max_length):
        # 모델 forward pass
        logits = model(tokens)  # (vocab_size,)

        # Temperature 적용
        probs = softmax_with_temperature(logits, temperature)

        # 샘플링
        next_token = torch.multinomial(probs, num_samples=1)

        tokens.append(next_token)

        if next_token == EOS_TOKEN:
            break

    return detokenize(tokens)
```

**Temperature 선택 가이드:**

| Temperature | 특성 | 사용 사례 |
|-------------|------|-----------|
| T=0 | Deterministic, 가장 높은 확률 | 수학 문제, 코드 생성 |
| T=0.3~0.5 | 매우 집중적, 안전한 선택 | 번역, 요약 |
| T=0.7 | 균형잡힌 선택 (기본값) | 일반적인 대화 |
| T=1.0 | 원본 확률 분포 사용 | 표준 생성 |
| T=1.5~2.0 | 창의적, 다양함 | 창작 글쓰기, 브레인스토밍 |

**예시: ChatGPT 사용**

```
질문: "2+2는 무엇인가요?"

T=0.1: "4입니다." (항상 동일)
T=1.0: "2+2는 4입니다." 또는 "답은 4입니다."
T=2.0: "2 더하기 2는 4예요!" 또는 "아하, 그건 4죠~"

질문: "짧은 이야기를 써주세요."

T=0.1: 예측 가능하고 일반적인 이야기
T=1.0: 적절한 창의성의 이야기
T=2.0: 매우 창의적이지만 때로는 일관성 없는 이야기
```

**중요한 실전 노트:**

**1. Determinism:**

```
이론: T=0이면 완전히 deterministic

실전: 완전히 deterministic하지 않을 수 있음!

이유:
- GPU 연산의 부동소수점 오차
- 병렬 연산의 순서 차이
- 메모리 레이아웃 차이
```

**해결책:**

```python
# 1. Seed 고정
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# 2. Deterministic mode (PyTorch)
torch.use_deterministic_algorithms(True)

# 3. CUDA deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 그래도 100% 보장은 어려움!
# 추천 논문: "Defeating Non-determinism in LLM Inference"
```

**2. Temperature와 Top-p/Top-k 함께 사용:**

```python
def generate_advanced(model, prompt, temperature=0.8, top_p=0.9, top_k=50):
    """
    Temperature + Top-p + Top-k를 모두 적용
    """
    tokens = tokenize(prompt)

    for _ in range(max_length):
        logits = model(tokens)

        # 1. Temperature 적용
        probs = softmax_with_temperature(logits, temperature)

        # 2. Top-k 필터링
        if top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs)
            probs[top_k_indices] = top_k_probs

        # 3. Top-p 필터링
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Nucleus 선택
            nucleus_mask = cumulative_probs <= top_p
            nucleus_mask[0] = True  # 최소 1개는 포함

            probs = torch.zeros_like(probs)
            probs[sorted_indices[nucleus_mask]] = sorted_probs[nucleus_mask]

        # 재정규화
        probs = probs / probs.sum()

        # 샘플링
        next_token = torch.multinomial(probs, num_samples=1)
        tokens.append(next_token)

        if next_token == EOS_TOKEN:
            break

    return detokenize(tokens)

# 실전 추천 조합:
# 균형잡힌 생성: temperature=0.7, top_p=0.9, top_k=50
# 창의적 생성: temperature=1.2, top_p=0.95, top_k=100
# 안전한 생성: temperature=0.3, top_p=0.8, top_k=10
```

---

# 4. Guided Decoding

## 4.1. Guided Decoding이란?

**문제:**

특정 형식(JSON, XML 등)의 출력이 필요한 경우, 일반 생성 방법으로는 형식을 보장할 수 없습니다.

**Naive Approach:**

```python
# 반복해서 생성하고 검증
max_attempts = 10
for attempt in range(max_attempts):
    response = llm.generate("Generate a JSON object...")

    if is_valid_json(response):
        return response
    else:
        print(f"Attempt {attempt} failed, retrying...")

# 문제점:
# - 비효율적 (여러 번 생성)
# - 리소스 낭비
# - 성공 보장 없음
```

**Guided Decoding Approach:**

```
핵심 아이디어:
생성 과정에서 유효하지 않은 토큰을 필터링하여
항상 유효한 형식의 출력을 보장
```

## 4.2. JSON 생성 예시

**목표: 다음 JSON 형식으로 생성**

```json
{
  "name": "...",
  "age": ...,
  "city": "..."
}
```

**Guided Decoding 과정:**

```
Step 1: 시작
가능한 토큰: ["{"]
선택: "{"

Step 2: 첫 번째 key
가능한 토큰: [""name"", ""age"", ""city""]
선택: ""name""

Step 3: 콜론
가능한 토큰: [":"]
선택: ":"

Step 4: String 값
가능한 토큰: [모든 문자열]
선택: ""John""

Step 5: 다음 필드
가능한 토큰: [","]
선택: ","

... 계속

최종 생성:
{"name": "John", "age": 30, "city": "Seoul"}
                                           ↑
                                        100% valid JSON!
```

**구현 개념:**

```python
class GuidedDecoder:
    def __init__(self, schema):
        """
        schema: 원하는 출력 형식 (JSON schema, regex 등)
        """
        self.schema = schema
        self.fsm = self.build_fsm(schema)  # Finite State Machine

    def build_fsm(self, schema):
        """
        Schema로부터 FSM (Finite State Machine) 구축
        각 상태에서 가능한 다음 토큰 정의
        """
        # JSON schema를 FSM으로 변환
        # 예: JSON의 경우, 다음과 같은 상태들:
        # - START: "{" 만 가능
        # - KEY: property name만 가능
        # - COLON: ":" 만 가능
        # - VALUE: 값 타입에 맞는 토큰만 가능
        # - COMMA_OR_END: "," 또는 "}" 가능
        pass

    def filter_tokens(self, current_state, token_probs):
        """
        현재 상태에서 유효한 토큰만 남기고 필터링

        Args:
            current_state: FSM의 현재 상태
            token_probs: (vocab_size,) 원본 확률 분포

        Returns:
            filtered_probs: 유효한 토큰만 남은 확률 분포
        """
        valid_tokens = self.fsm.get_valid_tokens(current_state)

        # 유효하지 않은 토큰의 확률을 0으로 설정
        filtered_probs = token_probs.clone()
        invalid_mask = ~torch.isin(torch.arange(len(token_probs)), valid_tokens)
        filtered_probs[invalid_mask] = 0

        # 재정규화
        filtered_probs = filtered_probs / filtered_probs.sum()

        return filtered_probs

    def generate(self, model, prompt, max_length=100):
        """
        Guided decoding으로 생성
        """
        tokens = tokenize(prompt)
        current_state = self.fsm.initial_state

        for _ in range(max_length):
            # 모델 forward pass
            logits = model(tokens)
            probs = torch.softmax(logits, dim=-1)

            # 유효한 토큰만 필터링
            filtered_probs = self.filter_tokens(current_state, probs)

            # 샘플링 (유효한 토큰 중에서만)
            next_token = torch.multinomial(filtered_probs, num_samples=1)
            tokens.append(next_token)

            # FSM 상태 업데이트
            current_state = self.fsm.transition(current_state, next_token)

            # 종료 상태 확인
            if self.fsm.is_terminal(current_state):
                break

        return detokenize(tokens)
```

**실전 예시: JSON Schema**

```python
# JSON schema 정의
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string", "format": "email"}
    },
    "required": ["name", "age"]
}

# Guided decoder 생성
decoder = GuidedDecoder(schema)

# 생성
prompt = "Generate user information:"
result = decoder.generate(model, prompt)

print(result)
# 출력 (항상 유효한 JSON):
# {"name": "Alice", "age": 25, "email": "alice@example.com"}
```

**사용 가능한 라이브러리:**

```python
# 1. Outlines (Hugging Face)
from outlines import models, generate

model = models.transformers("mistralai/Mistral-7B-v0.1")
generator = generate.json(model, schema)
result = generator("Generate user data:")

# 2. LM Format Enforcer
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

parser = JsonSchemaParser(schema)
prefix_function = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

output = model.generate(
    input_ids,
    prefix_allowed_tokens_fn=prefix_function
)

# 3. Guidance (Microsoft)
from guidance import models, gen

lm = models.Transformers("gpt2")
lm += f'{gen("name", regex=r"[A-Z][a-z]+")} is {gen("age", regex=r"[0-9]+")} years old'
```

**Guided Decoding의 기반 기술:**

**1. Finite State Machines (FSM):**

```
JSON 생성을 위한 FSM 예시:

States:
- START: 시작 상태
- OBJECT_START: "{" 다음
- KEY: property name 위치
- COLON: ":" 위치
- VALUE: 값 위치
- COMMA: "," 위치
- OBJECT_END: "}" 다음 (종료)

Transitions:
START → OBJECT_START (token: "{")
OBJECT_START → KEY (token: any key)
KEY → COLON (token: ":")
COLON → VALUE (token: any value)
VALUE → COMMA (token: ",")
VALUE → OBJECT_END (token: "}")
COMMA → KEY (token: any key)
```

**2. Context-Free Grammars (CFG):**

```
더 복잡한 형식을 위한 문법 정의

예: 간단한 JSON 문법
json ::= object | array
object ::= "{" [pair ("," pair)*] "}"
pair ::= string ":" value
value ::= string | number | object | array | "true" | "false" | "null"
array ::= "[" [value ("," value)*] "]"
string ::= '"' [char]* '"'
number ::= [0-9]+
```

**장점:**
- 형식 보장
- 효율적 (한 번에 생성)
- 안정적

**단점:**
- 구현 복잡
- 추가 계산 비용
- 모든 형식을 지원하지는 않음

**사용 사례:**
- API 응답 생성
- 구조화된 데이터 추출
- 코드 생성 (특정 문법)
- 설정 파일 생성

---

# 5. Context Length와 실전 고려사항

## 5.1. Context Length

**용어 정의:**

```
Context Length = Context Size = Window Size = Input Length

의미: 모델이 한 번에 처리할 수 있는 최대 토큰 수
```

**현대 LLM의 Context Length:**

| 모델 | Context Length |
|------|----------------|
| GPT-3 | 2,048 tokens |
| GPT-3.5-turbo | 4,096 tokens |
| GPT-4 | 8,192 tokens |
| GPT-4-32k | 32,768 tokens |
| Claude 2 | 100,000 tokens |
| Claude 3 | 200,000 tokens |
| Gemini 1.5 Pro | 1,000,000 tokens (~2M) |
| GPT-4 Turbo | 128,000 tokens |

**토큰 수 예시:**

```
1 token ≈ 0.75 words (영어)
1 token ≈ 0.5 words (한국어, 더 많은 토큰 필요)

예시:
- "Hello, how are you?" ≈ 6 tokens
- "안녕하세요, 어떻게 지내세요?" ≈ 12 tokens

문서 크기:
- 1,000 tokens ≈ 750 words ≈ 1.5 pages
- 10,000 tokens ≈ 7,500 words ≈ 15 pages
- 100,000 tokens ≈ 75,000 words ≈ 150 pages
- 1,000,000 tokens ≈ 750,000 words ≈ 1,500 pages
```

## 5.2. Context Rot 현상

**문제: Context Length가 크다고 항상 좋은 것은 아닙니다!**

**Needle in a Haystack Test:**

```
테스트 방법:
1. 긴 문서에 특정 정보 ("needle") 삽입
2. 문서 길이를 점점 늘림
3. 모델이 해당 정보를 정확히 찾는지 테스트

예시:
문서: [5000 tokens의 일반 텍스트]
      "The secret code is X7Y9Z2"  ← Needle
      [5000 tokens의 일반 텍스트]

질문: "What is the secret code?"
```

**실험 결과:**

```
검색 정확도 vs Context Length

100% |                    *
     |                *       *
     |             *             *
 75% |          *                   *
     |       *                         *
     |    *                               *
 50% |  *                                   *
     | *                                       *
     |*_________________________________________*
     0    20k   40k   60k   80k  100k  120k
               Context Length

관찰:
- 짧은 context: 높은 정확도
- 중간 context: 정확도 감소 시작
- 긴 context: 정확도 크게 감소
- 매우 긴 context: 정확도 약간 회복 (하지만 낮은 수준)
```

**Context Rot의 원인:**

**1. Attention Dilution:**

```python
# Attention은 모든 토큰에 분산됨
attention_score = softmax(Q @ K^T / sqrt(d_k))

# Context가 길면:
# - 중요한 정보에 할당되는 attention 감소
# - Noise에도 attention이 분산됨

예시:
짧은 context (100 tokens):
- 중요 정보: 10% attention
- 나머지: 90%

긴 context (10,000 tokens):
- 중요 정보: 0.1% attention  ← 너무 적음!
- 나머지: 99.9%
```

**2. Distractors (방해 요소):**

```
Distractors가 많을수록 성능 저하

실험 결과:
- Clean context: 80% 정확도
- 1개 distractor: 70% 정확도
- 5개 distractors: 50% 정확도
- 10개 distractors: 30% 정확도

Distractor 예시:
문서: "The code is ABC123"
Distractor 1: "Some people think the code is XYZ789"
Distractor 2: "Another code mentioned is DEF456"
...
```

**3. Position Bias:**

```
모델은 특정 위치의 정보를 더 잘 인식

위치별 검색 정확도:
- 시작 부분 (0-10%): 85%
- 끝 부분 (90-100%): 80%
- 중간 부분 (40-60%): 40%  ← 가장 낮음!

→ "Lost in the middle" 현상
```

**실전 해결책:**

**1. Retrieval-Augmented Generation (RAG):**

```python
def rag_pipeline(query, documents, model, k=3):
    """
    긴 문서를 모두 넣지 않고, 관련있는 부분만 선택
    """
    # 1. 관련 문서 검색
    relevant_chunks = retrieve_top_k(query, documents, k=k)

    # 2. 검색된 chunk만 context로 사용
    context = "\n\n".join(relevant_chunks)

    # 3. LLM에 전달
    prompt = f"""
    Context:
    {context}

    Question: {query}

    Answer based on the context above:
    """

    response = model.generate(prompt)
    return response

# 장점:
# - 관련 정보만 포함 → Context Rot 방지
# - 효율적 (짧은 context)
# - 더 정확한 응답
```

**2. Context 최적화:**

```python
def optimize_context(query, long_document, model, max_tokens=4000):
    """
    긴 문서를 압축하거나 요약
    """
    # 옵션 1: 요약
    summary = summarize(long_document, target_length=max_tokens // 2)

    # 옵션 2: 핵심 문장 추출
    key_sentences = extract_key_sentences(long_document, n=20)

    # 옵션 3: Query 관련 부분만 추출
    relevant_parts = extract_relevant_sections(long_document, query)

    # 최적화된 context 구성
    optimized_context = f"""
    Summary: {summary}

    Relevant Details:
    {relevant_parts}
    """

    return optimized_context
```

**3. Sliding Window:**

```python
def sliding_window_generation(long_input, model, window_size=2000, overlap=200):
    """
    긴 입력을 window로 나누어 처리
    """
    results = []

    for i in range(0, len(long_input), window_size - overlap):
        window = long_input[i:i + window_size]
        result = model.generate(window)
        results.append(result)

    # 결과 통합
    final_result = merge_results(results)
    return final_result
```

**4. 계층적 처리:**

```python
def hierarchical_processing(long_document, query, model):
    """
    문서를 계층적으로 처리
    """
    # 1단계: 각 섹션 요약
    sections = split_document(long_document)
    summaries = [model.summarize(section) for section in sections]

    # 2단계: 관련 섹션 선택
    relevant_summaries = select_relevant(summaries, query)

    # 3단계: 선택된 섹션의 원본 텍스트로 최종 답변
    selected_sections = [sections[i] for i in relevant_summaries]
    context = "\n\n".join(selected_sections)

    answer = model.generate(f"{context}\n\nQuestion: {query}")
    return answer
```

**실전 권장사항:**

```
1. 항상 필요한 만큼만 context 사용
   - 긴 context ≠ 좋은 결과

2. RAG 적극 활용
   - 검색 + 생성 = 더 정확한 답변

3. Context 구조화
   - 중요한 정보는 시작이나 끝에 배치
   - 중간은 피하기

4. Prompt engineering
   - "Based on the information above" 같은 지시 추가
   - 답변 시 인용 요청

5. 실험과 측정
   - 실제 성능 측정
   - Context 길이 최적화
```

---

# 6. 요약

## 핵심 개념

### 1. LLM (Large Language Models)

```
정의:
- Language Model: 토큰 시퀀스에 확률 할당
- Large: 큰 모델 크기 + 많은 데이터 + 많은 컴퓨트

특징:
- Decoder-only 아키텍처
- ≥ 1B parameters
- Next token prediction

주요 모델:
GPT, LLaMA, Mistral, Gemma, etc.
```

### 2. Mixture of Experts (MoE)

```
핵심 아이디어:
입력에 따라 일부 파라미터만 활성화

구조:
┌───────────────┐
│ Gate/Router   │ ← 어떤 expert를 사용할지 결정
└───────────────┘
        ↓
┌──────┬──────┬──────┐
│Exp 1 │Exp 2 │Exp 3 │ ← 선택된 expert만 활성화
└──────┴──────┴──────┘

적용 위치:
Feedforward Layer (파라미터가 가장 많음)

종류:
- Dense MoE: 모든 expert 사용 (가중합)
- Sparse MoE: Top-k expert만 사용

장점:
- 모델 크기 확장 without 계산 비용 증가
- 더 sample efficient
```

### 3. Response Generation

```
방법                | 특성              | 사용 사례
--------------------|-------------------|------------------
Greedy Decoding     | 최고 확률 선택    | 간단한 작업
Beam Search         | k개 경로 추적     | 번역, 요약
Sampling            | 확률 분포 샘플링  | 창의적 생성
Top-k Sampling      | 상위 k개만 고려   | 품질-다양성 균형
Top-p Sampling      | 누적 p까지 고려   | Adaptive 선택

Temperature:
- T → 0: Deterministic (greedy와 유사)
- T = 0.7-1.0: 균형
- T > 1.0: 창의적

Guided Decoding:
특정 형식(JSON, XML) 보장
```

## Response Generation 비교

**종합 비교표:**

| 방법 | Deterministic | 다양성 | 품질 | 계산 비용 | 사용 |
|------|---------------|--------|------|-----------|------|
| Greedy | ✅ | ❌ | 보통 | 낮음 | 드물음 |
| Beam Search | ✅ | ❌ | 높음 | 높음 (k배) | 번역, 요약 |
| Sampling | ❌ | ✅ | 불안정 | 낮음 | 드물음 |
| Top-k | ❌ | 중간 | 중간 | 낮음 | 일반적 |
| Top-p | ❌ | 중간 | 높음 | 낮음 | 가장 일반적 |

**실전 추천 조합:**

```python
# 균형잡힌 생성 (가장 일반적)
temperature=0.7, top_p=0.9, top_k=50

# 창의적 생성
temperature=1.2, top_p=0.95, top_k=100

# 안전한 생성 (정확성 중요)
temperature=0.3, top_p=0.8, top_k=10

# Deterministic (디버깅, 테스트)
temperature=0.0
```

## 실전 구현 체크리스트

### MoE 구현

- [ ] Expert 개수 결정 (보통 8-64)
- [ ] k 값 결정 (보통 1-2)
- [ ] Feedforward layer에 적용
- [ ] Load balancing loss 추가
- [ ] Routing collapse 모니터링

### Response Generation 구현

- [ ] Sampling 방법 선택 (Top-p 권장)
- [ ] Temperature 값 설정 (0.7 시작)
- [ ] Top-k, Top-p 값 설정
- [ ] 특정 형식 필요 시 Guided Decoding 고려

### Context 관리

- [ ] Context length 확인
- [ ] RAG 시스템 고려
- [ ] 중요 정보는 시작/끝에 배치
- [ ] Context Rot 테스트

### 최적화

- [ ] KV Cache 구현 (메모리 절약)
- [ ] Batch inference (처리량 증가)
- [ ] Quantization (INT8/INT4)
- [ ] Flash Attention (속도 향상)

---

# 7. 중요 용어 정리

## LLM 관련

- **LLM (Large Language Model)**: 대규모 언어 모델. Decoder-only 아키텍처로 텍스트 생성하는 모델
- **Language Model**: 토큰 시퀀스에 확률을 할당하는 모델
- **Next Token Prediction**: 이전 토큰들이 주어졌을 때 다음 토큰을 예측하는 작업
- **Context Length/Size/Window**: 모델이 한 번에 처리할 수 있는 최대 토큰 수
- **Token**: 텍스트의 최소 단위 (단어의 일부, 단어, 구두점 등)

## MoE 관련

- **MoE (Mixture of Experts)**: 여러 expert 모델 중 일부만 선택적으로 활성화하는 아키텍처
- **Expert**: MoE에서 독립적인 하위 네트워크 (보통 feedforward network)
- **Gate/Router**: 입력에 따라 어떤 expert를 사용할지 결정하는 네트워크
- **Dense MoE**: 모든 expert의 출력을 가중합하는 방식
- **Sparse MoE**: Top-k expert만 선택하여 활성화하는 방식
- **Routing Collapse**: 일부 expert만 과도하게 사용되는 현상
- **Load Balancing Loss**: Expert 사용을 균등하게 만드는 추가 loss
- **Active Parameters**: Forward pass에서 실제로 사용되는 파라미터
- **FLOPS (Floating-Point Operations)**: 부동소수점 연산 수
- **Noisy Gating**: Gate에 noise를 추가하여 다양한 expert 사용 유도

## Response Generation 관련

- **Decoding**: 모델 출력으로부터 텍스트를 생성하는 과정
- **Greedy Decoding**: 매 step마다 최고 확률 토큰을 선택하는 방법
- **Beam Search**: k개의 가장 가능성 높은 경로를 추적하는 방법
- **Beam Width**: Beam search에서 유지하는 경로의 개수
- **Sampling**: 확률 분포에서 토큰을 샘플링하는 방법
- **Top-k Sampling**: 상위 k개 토큰 중에서만 샘플링
- **Top-p (Nucleus) Sampling**: 누적 확률이 p를 넘는 토큰 집합에서 샘플링
- **Temperature**: Softmax의 sharpness를 조절하는 hyperparameter
- **Logits**: 모델의 raw output (softmax 이전)
- **Guided Decoding**: 특정 형식을 보장하도록 토큰을 제한하는 방법

## Temperature 관련

- **Low Temperature (T→0)**: 뾰족한 분포, deterministic, 안전한 선택
- **High Temperature (T→∞)**: 평평한 분포, 창의적, uniform distribution에 가까움
- **Softmax**: Logits를 확률 분포로 변환하는 함수
- **Deterministic**: 같은 입력에 항상 같은 출력을 생성

## Context 관련

- **Context Rot**: Context가 길어질수록 정보 검색 능력이 감소하는 현상
- **Needle in a Haystack**: 긴 문서에서 특정 정보를 찾는 테스트
- **Distractor**: 모델을 혼란시키는 방해 정보
- **Position Bias**: 특정 위치의 정보를 더 잘 인식하는 현상
- **Lost in the Middle**: 중간 부분의 정보를 잘 인식하지 못하는 현상
- **RAG (Retrieval-Augmented Generation)**: 검색과 생성을 결합한 방법
- **Sliding Window**: 긴 입력을 window로 나누어 처리

## 기타

- **FSM (Finite State Machine)**: Guided decoding을 위한 상태 기계
- **CFG (Context-Free Grammar)**: 형식을 정의하는 문법
- **Length Normalization**: 시퀀스 길이로 확률을 정규화
- **KV Cache**: Key와 Value를 캐싱하여 추론 속도 향상
- **Quantization**: 모델 가중치를 낮은 precision으로 변환 (INT8, INT4)
- **Flash Attention**: 메모리 효율적인 attention 구현

---

**다음 강의 예고:**

Lecture 4에서는 LLM Training과 Fine-tuning에 대해 다룹니다:
- Pre-training strategies
- Instruction tuning
- RLHF (Reinforcement Learning from Human Feedback)
- Parameter-efficient fine-tuning (LoRA, QLoRA)
- Training infrastructure

---

**수고하셨습니다!** 🎉
