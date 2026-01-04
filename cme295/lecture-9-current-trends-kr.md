# Lecture 9: Current Trends in LLMs

# Materials

- [CME 295](https://cme295.stanford.edu/syllabus/)
- [slide](https://cme295.stanford.edu/slides/fall25-cme295-lecture9.pdf)
- [video](https://www.youtube.com/watch?v=Q86qzJ1K1Ss&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=9)

# Table of Contents

- [Lecture 9: Current Trends in LLMs](#lecture-9-current-trends-in-llms)
- [Materials](#materials)
- [Table of Contents](#table-of-contents)
- [강의 개요](#강의-개요)
  - [강의 목표](#강의-목표)
  - [강의 구성](#강의-구성)
- [Part 1: 전체 코스 복습](#part-1-전체-코스-복습)
  - [1. Transformer 기초](#1-transformer-기초)
    - [1.1. Tokenization](#11-tokenization)
    - [1.2. Embeddings](#12-embeddings)
    - [1.3. Self-Attention](#13-self-attention)
  - [2. Transformer 개선사항](#2-transformer-개선사항)
    - [2.1. Position Embeddings](#21-position-embeddings)
    - [2.2. Multi-Head Attention 최적화](#22-multi-head-attention-최적화)
    - [2.3. Normalization](#23-normalization)
  - [3. Large Language Models](#3-large-language-models)
    - [3.1. Mixture of Experts (MoE)](#31-mixture-of-experts-moe)
    - [3.2. Sampling 전략](#32-sampling-전략)
  - [4. LLM Training](#4-llm-training)
    - [4.1. Scaling Laws](#41-scaling-laws)
    - [4.2. Flash Attention](#42-flash-attention)
    - [4.3. Parallelism](#43-parallelism)
  - [5. LLM Tuning](#5-llm-tuning)
    - [5.1. 3단계 Training Pipeline](#51-3단계-training-pipeline)
    - [5.2. Reward Modeling](#52-reward-modeling)
    - [5.3. RL 기반 Tuning](#53-rl-기반-tuning)
  - [6. LLM Reasoning](#6-llm-reasoning)
    - [6.1. Chain of Thought](#61-chain-of-thought)
    - [6.2. GRPO vs PPO](#62-grpo-vs-ppo)
    - [6.3. GRPO의 문제점과 개선](#63-grpo의-문제점과-개선)
  - [7. Agentic LLMs](#7-agentic-llms)
    - [7.1. RAG (Retrieval Augmented Generation)](#71-rag-retrieval-augmented-generation)
    - [7.2. Tool Calling](#72-tool-calling)
  - [8. LLM Evaluation](#8-llm-evaluation)
    - [8.1. 전통적인 Metrics](#81-전통적인-metrics)
    - [8.2. LLM-as-a-Judge](#82-llm-as-a-judge)
    - [8.3. Benchmarks](#83-benchmarks)
- [Part 2: 2025년 현재 트렌드](#part-2-2025년-현재-트렌드)
  - [1. Vision Transformer (ViT)](#1-vision-transformer-vit)
    - [1.1. 동기: Transformer를 Vision에 적용할 수 있을까?](#11-동기-transformer를-vision에-적용할-수-있을까)
    - [1.2. ViT 아키텍처](#12-vit-아키텍처)
    - [1.3. Image Patching](#13-image-patching)
    - [1.4. End-to-End 예시](#14-end-to-end-예시)
    - [1.5. Inductive Bias](#15-inductive-bias)
  - [2. Multimodal LLMs](#2-multimodal-llms)
    - [2.1. Vision-Language Models (VLM)](#21-vision-language-models-vlm)
    - [2.2. 두 가지 접근 방식](#22-두-가지-접근-방식)
    - [2.3. LAVA 모델](#23-lava-모델)
  - [3. Diffusion-based LLMs](#3-diffusion-based-llms)
    - [3.1. 동기: Autoregressive의 한계](#31-동기-autoregressive의-한계)
    - [3.2. Diffusion이란?](#32-diffusion이란)
    - [3.3. Text에 Diffusion 적용하기](#33-text에-diffusion-적용하기)
    - [3.4. 최근 발전](#34-최근-발전)
  - [4. Transformer의 확장](#4-transformer의-확장)
- [Part 3: 요약 및 다음 단계](#part-3-요약-및-다음-단계)
  - [1. 코스 전체 요약](#1-코스-전체-요약)
  - [2. 핵심 Takeaways](#2-핵심-takeaways)
  - [3. Final 시험 범위](#3-final-시험-범위)
- [용어 정리](#용어-정리)
  - [Vision 관련](#vision-관련)
  - [Diffusion 관련](#diffusion-관련)
  - [Multimodal 관련](#multimodal-관련)

---

# 강의 개요

## 강의 목표

이번 강의는 CME 295의 마지막 강의로, 다음 세 가지 목표를 가지고 있습니다:

1. **전체 코스 복습**: Lecture 1-8의 핵심 내용을 정리하고 연결
2. **2025년 트렌드**: 현재와 가까운 미래의 트렌딩 토픽 소개
3. **다음 단계**: Final 시험 준비 및 향후 학습 방향

## 강의 구성

**Part 1: 전체 코스 복습 (Recap)**
- Transformer 기초부터 최신 기법까지
- Lecture 1-8의 핵심 개념 정리
- Final 시험 범위

**Part 2: 2025년 트렌드**
- Vision Transformer (ViT)
- Multimodal LLMs
- Diffusion-based LLMs
- Transformer의 다양한 응용

**Part 3: 마무리**
- 전체 요약
- Final 시험 안내
- 향후 학습 방향

---

# Part 1: 전체 코스 복습

이 섹션에서는 Lecture 1부터 Lecture 8까지 배운 모든 내용을 체계적으로 정리합니다. Final 시험 범위는 **Lecture 5-8**이지만, 전체적인 맥락을 이해하기 위해 모든 내용을 복습합니다.

## 1. Transformer 기초

Lecture 1에서는 Transformer의 기본 개념을 학습했습니다.

### 1.1. Tokenization

**텍스트를 처리하는 첫 단계**

```python
# 텍스트를 atomic units로 분할
Text: "The cat sat on the mat"

# Subword tokenization
Tokens: ["The", "cat", "sat", "on", "the", "mat"]
```

**핵심 포인트:**
- Subword level tokenizer가 가장 일반적
- 단어의 root를 재사용할 수 있어 효율적
- 예: "playing", "played", "player" → "play", "ing", "ed", "er"

### 1.2. Embeddings

**토큰을 벡터로 표현하기**

**초기 방법: Word2Vec**
```python
# 문맥 없는 고정 임베딩
"bank" → [0.1, -0.3, 0.5, ...]  # 항상 동일

문제점:
"I went to the bank to deposit money"  # 은행
"I sat by the river bank"              # 강둑
→ 동일한 임베딩! (문맥을 고려하지 않음)
```

**RNN의 등장**
- 순차적으로 토큰 처리
- 내부 상태로 문맥 유지
- **문제점**: Long-range dependency (먼 거리 토큰 정보 손실)

```python
# RNN의 한계
"The cat, which was sitting on the mat in the living room, meowed"
        ↑                                                    ↑
     주어                                                  동사
# "cat"과 "meowed"의 관계를 학습하기 어려움
```

### 1.3. Self-Attention

**모든 토큰이 직접 소통!**

```
Self-Attention = softmax(Q·K^T / sqrt(d_k)) · V

핵심 아이디어:
- Query: "나는 누구와 관련있나?"
- Key: "나는 이런 정보를 가지고 있어"
- Value: "실제 전달할 정보"
```

**구체적인 예시:**

```python
시퀀스: "The cat sat"

각 토큰의 attention:
"cat"의 입장에서:
  - "The"와의 similarity: 0.2
  - "cat"와의 similarity: 0.5 (자기 자신)
  - "sat"와의 similarity: 0.3

Output for "cat" = 0.2*V_The + 0.5*V_cat + 0.3*V_sat
```

**장점:**
- ✅ 거리와 상관없이 모든 토큰 간 직접 연결
- ✅ 병렬 처리 가능
- ✅ Long-range dependency 해결

**단점:**
- ❌ O(n²) 복잡도

## 2. Transformer 개선사항

Lecture 2에서는 Transformer를 개선하는 다양한 기법을 학습했습니다.

### 2.1. Position Embeddings

**왜 필요한가?**

Self-Attention은 위치 정보가 없습니다!

```python
"I love transformers"
"transformers love I"
# Self-Attention은 이 둘을 구분하지 못함!
```

**발전 과정:**

```
Learned (절대 위치)
  ↓
Sinusoidal (고정 수식)
  ↓
T5 Bias / ALiBi (attention에 직접)
  ↓
RoPE (회전 변환) ← 현대 표준!
```

**RoPE (Rotary Position Embeddings)**

```python
# Query와 Key를 회전시켜 상대 위치 인코딩
q_rotated = R(θ_pos) · q
k_rotated = R(θ_pos) · k

# 핵심: 상대 위치의 함수로 자연스럽게 표현됨
Attention = q_rotated · k_rotated^T
```

**사용 모델:**
- GPT-3/4
- LLaMA 시리즈
- Mistral
- 대부분의 현대 LLM

### 2.2. Multi-Head Attention 최적화

**KV Cache 메모리 문제**

```python
# GPT-3 (175B parameters)
num_heads = 96
d_k = 128
seq_len = 2048

KV Cache per head = 1 MB
Total = 96 MB per sample
Batch 100 = 9.6 GB!
```

**해결책: Grouped Query Attention (GQA)**

```
MHA (Multi-Head Attention):
  Head 1: Q1, K1, V1
  Head 2: Q2, K2, V2
  ...
  Head 96: Q96, K96, V96

GQA (Grouped Query Attention):
  Group 1 (Heads 1-8): Q1-Q8, K_shared, V_shared
  Group 2 (Heads 9-16): Q9-Q16, K_shared, V_shared
  ...

MQA (Multi-Query Attention):
  All Heads: Q1-Q96, K_shared (1개), V_shared (1개)
```

**메모리 비교:**

```
MHA: 96 MB
GQA (8 groups): 8 MB (12배 감소!)
MQA: 1 MB (96배 감소!)
```

**사용 예시:**
- LLaMA 2 (70B): GQA with 8 groups
- Mistral: GQA
- 작은 모델 (< 7B): MHA
- 매우 큰 모델 (> 70B): GQA 또는 MQA

### 2.3. Normalization

**Pre-norm vs Post-norm**

```python
# Post-norm (원본 Transformer)
x = LayerNorm(x + Sublayer(x))

# Pre-norm (현대 표준)
x = x + Sublayer(LayerNorm(x))
```

**Pre-norm이 더 나은 이유:**
1. Gradient flow가 더 안정적
2. Learning rate warmup 덜 필요
3. 학습 초기에 더 안정적

**RMSNorm (최신 기법)**

```python
# LayerNorm
LayerNorm(x) = γ · (x - μ) / sqrt(σ² + ε) + β

# RMSNorm (평균 제거, β 제거)
RMSNorm(x) = γ · x / sqrt(mean(x²) + ε)
```

**장점:**
- 25% 빠른 계산
- 파라미터 감소 (β 제거)
- 비슷한 성능

**사용 모델:**
- LLaMA
- Mistral
- Falcon

## 3. Large Language Models

Lecture 3에서는 LLM의 특별한 구조와 기법을 학습했습니다.

### 3.1. Mixture of Experts (MoE)

**핵심 아이디어: 모든 파라미터를 항상 사용하지 않기**

```
전통적인 LLM:
Input → All Parameters → Output

MoE LLM:
Input → Gating → Expert 1 (선택됨)
              → Expert 2
              → Expert 3
              → Expert 4 (선택됨)
              → ...
Only activated experts → Output
```

**구체적인 예시:**

```python
# Token-level routing
Text: "Write Python code for sorting"

Token "Python" → Expert 2, Expert 5 (Programming experts)
Token "sorting" → Expert 2, Expert 7 (Algorithm experts)
Token "code" → Expert 1, Expert 2 (Code experts)

# 각 토큰마다 다른 experts 활성화
```

**장점:**
- Forward pass 시 일부 파라미터만 사용
- 더 큰 모델 구축 가능
- 병렬화 가능 (experts를 다른 GPU에 배치)

**단점:**
- Training 복잡도 증가
- Load balancing 필요

### 3.2. Sampling 전략

**다음 토큰 예측하기**

```python
# Greedy (탐욕적)
"The cat" → "sat" (P=0.6, 가장 높은 확률)

# Sampling (샘플링)
"The cat" → "sat" (P=0.6) or "jumped" (P=0.2) or "meowed" (P=0.15)
```

**Temperature 조절**

```python
# Low temperature (T=0.1) - 더 deterministic
Probabilities: [0.95, 0.03, 0.01, 0.01]
Output: 거의 항상 첫 번째 선택

# High temperature (T=1.5) - 더 creative
Probabilities: [0.4, 0.25, 0.2, 0.15]
Output: 다양한 선택 가능
```

**언제 무엇을 사용하나?**

```
T = 0.0 (Greedy):
  - 사실 기반 QA
  - 코드 생성
  - 번역

T = 0.7-1.0:
  - 일반 대화
  - 균형잡힌 창의성

T = 1.5+:
  - 창의적 글쓰기
  - 브레인스토밍
  - 시 작성
```

## 4. LLM Training

Lecture 4에서는 대규모 LLM을 효율적으로 훈련하는 방법을 학습했습니다.

### 4.1. Scaling Laws

**더 큰 모델 = 더 좋은 성능?**

```
발견된 법칙:
1. 더 많은 compute → 더 낮은 loss
2. 더 큰 데이터셋 → 더 낮은 loss
3. 더 많은 파라미터 → 더 낮은 loss
```

**Chinchilla Scaling Laws (2022)**

```
Rule of thumb:
모델 파라미터 수 : 학습 토큰 수 = 1 : 20

예시:
100B parameter 모델 → 2T tokens로 학습
70B parameter 모델 → 1.4T tokens로 학습

문제: 대부분의 초기 LLM들은 undertrained!
```

**구체적인 예시:**

```python
# GPT-3
Parameters: 175B
Training tokens: ~300B
Chinchilla optimal: 175B × 20 = 3.5T tokens
→ Undertrained!

# LLaMA 2
Parameters: 70B
Training tokens: 2T
Chinchilla optimal: 70B × 20 = 1.4T tokens
→ Well-trained!
```

### 4.2. Flash Attention

**문제: Attention은 메모리 병목**

```
Standard Attention:
1. Q·K^T 계산 → HBM에 저장
2. Softmax 계산 → HBM에 저장
3. Attention·V 계산
→ HBM (큰/느린 메모리)에 많은 읽기/쓰기
```

**Flash Attention의 해결책**

```
Key Ideas:
1. Tiling: 계산을 작은 블록으로 분할
2. SRAM 활용: 작지만 빠른 메모리 사용
3. Recomputation: 저장 대신 재계산
```

**예시:**

```python
# Standard Attention
Attention Matrix (seq_len=2048):
  2048 × 2048 = 4M elements
  메모리: 16 MB (FP32)
  HBM 읽기/쓰기: 매우 느림

# Flash Attention
Block size: 128 × 128
  128 × 128 = 16K elements
  SRAM에 fit!
  훨씬 빠른 연산

결과:
- 2-4배 빠른 속도
- 동일한 결과 (exact, no approximation)
```

### 4.3. Parallelism

**하나의 GPU로는 부족합니다!**

**Data Parallelism**

```
GPU 1: Batch 1-32    ┐
GPU 2: Batch 33-64   ├─ 동일한 모델
GPU 3: Batch 65-96   ┘

각 GPU가 gradient 계산 → 평균 → 업데이트
```

**Model Parallelism**

```
GPU 1: Layer 1-10   ┐
GPU 2: Layer 11-20  ├─ 모델 분할
GPU 3: Layer 21-30  ┘

데이터가 GPU를 순차적으로 통과
```

**Pipeline Parallelism**

```
Time step 1:
  GPU 1: Process Batch 1
  GPU 2: Idle
  GPU 3: Idle

Time step 2:
  GPU 1: Process Batch 2
  GPU 2: Process Batch 1
  GPU 3: Idle

Time step 3:
  GPU 1: Process Batch 3
  GPU 2: Process Batch 2
  GPU 3: Process Batch 1
```

## 5. LLM Tuning

Lecture 5에서는 LLM을 유용하게 만드는 fine-tuning 과정을 학습했습니다.

### 5.1. 3단계 Training Pipeline

```
Step 1: Pre-training
  - 목적: 언어/코드 구조 학습
  - 데이터: Trillions of tokens
  - Task: Next token prediction
  - 결과: Autocomplete 가능한 모델

  ↓

Step 2: Supervised Fine-Tuning (SFT)
  - 목적: 원하는 행동 학습
  - 데이터: Input-output pairs
  - Task: 주어진 형식으로 응답
  - 결과: Helpful한 모델

  ↓

Step 3: Preference Tuning
  - 목적: 선호도 정렬
  - 데이터: Preference pairs
  - Task: 좋은 응답 vs 나쁜 응답
  - 결과: 인간 선호도와 정렬된 모델
```

**구체적인 예시:**

```python
# Pre-training
Input: "The cat sat on the"
Output: "mat"  # Next token prediction

# SFT
Input: "What is the capital of France?"
Output: "The capital of France is Paris."

# Preference Tuning
Input: "Explain quantum computing"
Preferred: "Quantum computing uses quantum bits..."
Rejected: "Idk, google it lol"
```

### 5.2. Reward Modeling

**Bradley-Terry 공식**

```
P(output_i > output_j) = exp(r_i) / (exp(r_i) + exp(r_j))

여기서:
- r_i: Output i의 reward score
- r_j: Output j의 reward score
```

**Reward Model 학습**

```python
# Pairwise training
Input: "Write a Python function"
Output A: "def foo():\n    pass"  # 선호됨
Output B: "idk"                    # 거부됨

# Model 학습
r_A = reward_model(Input, Output A)  # 높은 점수
r_B = reward_model(Input, Output B)  # 낮은 점수

Loss = -log(exp(r_A) / (exp(r_A) + exp(r_B)))
```

**추론 시에는:**

```python
# Single output scoring
output = llm.generate(prompt)
score = reward_model(prompt, output)
```

### 5.3. RL 기반 Tuning

**LLM as RL Agent**

```
State: 지금까지 생성된 토큰
Action: 다음 토큰 예측
Environment: 토큰 공간
Reward: Reward model의 점수
```

**PPO (Proximal Policy Optimization)**

```
Components:
1. Policy Model (LLM): 텍스트 생성
2. Reward Model: 품질 평가
3. Value Model: 예상 미래 보상
4. Reference Model: 원래 SFT 모델 (regularization)

Objective:
  Maximize reward
  - KL penalty (reference model로부터 너무 멀어지지 않기)
```

**구체적인 Flow:**

```python
# PPO Training Loop
for iteration in range(num_iterations):
    # 1. Generate rollouts
    prompts = sample_prompts()
    responses = policy_model.generate(prompts)

    # 2. Get rewards
    rewards = reward_model(prompts, responses)

    # 3. Get value predictions
    values = value_model(prompts, responses)

    # 4. Compute advantages
    advantages = compute_GAE(rewards, values)

    # 5. Update policy
    policy_model.update(advantages, reference_model)

    # 6. Update value model
    value_model.update(rewards)
```

## 6. LLM Reasoning

Lecture 6에서는 LLM의 추론 능력을 향상시키는 방법을 학습했습니다.

### 6.1. Chain of Thought

**핵심 아이디어: 단계별로 생각하기**

```
Without CoT:
Input: "Roger has 5 balls. He buys 2 more. How many balls does he have?"
Output: "7"

With CoT:
Input: "Roger has 5 balls. He buys 2 more. How many balls does he have?"
Output: "Let me think step by step:
1. Roger starts with 5 balls
2. He buys 2 more balls
3. Total = 5 + 2 = 7
Therefore, Roger has 7 balls."
```

**성능 향상**

```
실험 결과 (MATH benchmark):
Without CoT: 30% accuracy
With CoT: 65% accuracy
→ 2배 이상 향상!
```

### 6.2. GRPO vs PPO

**PPO의 구조**

```
Components (4개):
1. Policy Model (LLM)
2. Reward Model
3. Value Model
4. Reference Model

문제점:
- Value Model 학습/유지 비용
- 복잡한 구조
```

**GRPO (Group Relative Policy Optimization)**

```
Components (2개만!):
1. Policy Model (LLM)
2. Reference Model

핵심 아이디어:
- Value Model 제거
- 여러 completions 생성
- Reward를 서로 비교 (상대적)
```

**구체적인 비교:**

```python
# PPO
for each prompt:
    completion = policy.generate(prompt)
    reward = reward_model(prompt, completion)
    value = value_model(prompt, completion)
    advantage = GAE(reward, value)  # Value model 사용

# GRPO
for each prompt:
    completions = [policy.generate(prompt) for _ in range(K)]
    rewards = [reward_model(prompt, c) for c in completions]
    # Completions끼리 비교 (Value model 불필요!)
    advantages = rewards - mean(rewards)
```

**GRPO의 장점:**

1. **더 간단**: Value model 불필요
2. **더 효율적**: 모델 하나 덜 학습
3. **Verifiable rewards에 적합**: 수학 문제 등

### 6.3. GRPO의 문제점과 개선

**Length Bias 문제**

```
문제:
GRPO는 짧은 틀린 답을 더 많이 penalize함
→ 모델이 긴 틀린 답을 생성하게 됨

예시:
Short incorrect: "The answer is 5" (매우 penalize)
Long incorrect: "Let me explain... [500 tokens]... so it's 5" (덜 penalize)
→ 모델이 길게 쓰는 법을 학습!
```

**해결책: GRPO Done Right**

```python
# Original GRPO loss
loss = -advantages / length  # Length로 normalize → 문제!

# GRPO Done Right
loss = -advantages  # Normalization 제거
```

**DAPO (Direct Advantage Policy Optimization)**

```
추가 개선사항:
1. Length bias 제거
2. 더 안정적인 training
3. Reasoning tasks에 특화
```

## 7. Agentic LLMs

Lecture 7에서는 LLM을 외부 시스템과 연결하는 방법을 학습했습니다.

### 7.1. RAG (Retrieval Augmented Generation)

**왜 필요한가?**

```
문제:
LLM의 지식 = 학습 데이터까지만 (Knowledge Cutoff)

예시:
"2024년 10월에 무슨 일이 있었나요?"
→ LLM (2024년 1월 학습): 모름!
```

**RAG의 구조**

```
User Query
  ↓
1. Retrieval (검색)
   - Candidate Retrieval (Bi-encoder)
   - Reranking (Cross-encoder)
  ↓
2. Augmentation (증강)
   - 관련 문서를 prompt에 추가
  ↓
3. Generation (생성)
   - LLM이 문서 기반으로 답변
```

**구체적인 예시:**

```python
# Step 1: Candidate Retrieval (Bi-encoder)
query = "최신 iPhone 가격"
query_embedding = encoder(query)  # [0.1, -0.3, ...]

documents = [
    "iPhone 15 Pro는 $999입니다",
    "삼성 Galaxy는...",
    "MacBook Pro는..."
]
doc_embeddings = [encoder(d) for d in documents]

# Cosine similarity
scores = [cosine(query_embedding, d_emb) for d_emb in doc_embeddings]
top_k = select_top_k(documents, scores, k=10)

# Step 2: Reranking (Cross-encoder)
rerank_scores = [cross_encoder(query, doc) for doc in top_k]
final_docs = select_top_k(top_k, rerank_scores, k=3)

# Step 3: Generation
prompt = f"""
다음 문서를 참고하여 질문에 답하세요:

문서: {final_docs}

질문: {query}
답변:"""

answer = llm.generate(prompt)
```

**Bi-encoder vs Cross-encoder**

```
Bi-encoder:
Query와 Document를 독립적으로 인코딩
  + 빠름 (사전 계산 가능)
  + 대규모 검색에 적합
  - 정확도 낮음

Cross-encoder:
Query와 Document를 함께 인코딩
  + 정확도 높음
  + 더 정교한 유사도
  - 느림 (모든 pair 계산)
  - 소규모 reranking에 적합
```

### 7.2. Tool Calling

**LLM이 API 사용하기**

```
2-Step Process:

Step 1: Tool Selection
  LLM decides: 어떤 API? 어떤 arguments?

Step 2: Tool Execution → Final Answer
  API 실행 → 결과를 LLM에게 → 최종 답변
```

**구체적인 예시:**

```python
# User query
"What's the weather in Seoul?"

# Step 1: LLM decides to use weather API
llm_output = {
    "tool": "get_weather",
    "arguments": {
        "location": "Seoul",
        "units": "celsius"
    }
}

# Step 2: Execute API
weather_data = get_weather(location="Seoul", units="celsius")
# Returns: {"temperature": 15, "condition": "cloudy"}

# Step 3: LLM generates final answer
prompt = f"""
User asked: "What's the weather in Seoul?"
API returned: {weather_data}
Generate a natural response:
"""

final_answer = llm.generate(prompt)
# "The weather in Seoul is currently 15°C and cloudy."
```

**Modern Agentic Workflow**

```
User: "분석 보고서를 작성하고 이메일로 보내줘"

Agent Flow:
1. RAG: 관련 데이터 검색
   ↓
2. Tool: data_analysis API 호출
   ↓
3. Tool: generate_report API 호출
   ↓
4. Tool: send_email API 호출
   ↓
5. LLM: 최종 응답 생성
   "보고서를 작성하여 이메일로 발송했습니다."
```

## 8. LLM Evaluation

Lecture 8에서는 LLM의 성능을 평가하는 방법을 학습했습니다.

### 8.1. 전통적인 Metrics

**BLEU, ROUGE, METEOR**

```python
Reference: "The cat sat on the mat"
Hypothesis: "A cat sat on a mat"

BLEU Score: 0.75 (n-gram overlap 기반)

문제점:
Reference: "The movie was excellent"
Hypothesis: "The film was great"
→ BLEU: Low score (단어가 다름)
→ 하지만 의미는 동일!
```

**한계:**
- 단어 수준 일치만 고려
- 의미적 유사성 무시
- Paraphrase 처리 못함

### 8.2. LLM-as-a-Judge

**핵심 아이디어: LLM으로 평가하기**

```python
# Evaluation Prompt
prompt = f"""
다음 기준으로 응답을 평가하세요:
- 정확성
- 유용성
- 안전성

질문: {question}
응답: {response}

먼저 이유를 설명하고, 그 다음 Pass/Fail을 판단하세요.

평가:"""

judgment = judge_llm.generate(prompt)
```

**예시:**

```
질문: "Python에서 리스트를 정렬하는 방법은?"

응답: "sorted() 함수나 .sort() 메서드를 사용하면 됩니다."

LLM Judge 평가:
"이유: 응답이 두 가지 주요 방법을 정확히 언급했습니다.
sorted()는 새 리스트를 반환하고, .sort()는 in-place로
정렬합니다. 정확하고 유용합니다.

판단: Pass"
```

**Biases (편향)**

```
1. Position Bias
   Input: Compare A vs B
   LLM tends to prefer: A (첫 번째)
   해결: 순서를 바꿔서 2번 평가

2. Verbosity Bias
   Short answer: "Paris"
   Long answer: "Paris is the capital..."
   LLM tends to prefer: Long answer

3. Self-Enhancement Bias
   LLM prefers its own outputs
   해결: 다른 LLM을 judge로 사용
```

### 8.3. Benchmarks

**주요 벤치마크 카테고리**

```
1. Knowledge (지식)
   - MMLU: 다양한 주제 객관식
   - TriviaQA: 사실 질문

2. Reasoning (추론)
   - MATH: 수학 문제
   - GSM8K: 초등 수학

3. Coding (코딩)
   - HumanEval: Python 함수 작성
   - MBPP: 기본 프로그래밍

4. Safety (안전성)
   - TruthfulQA: 진실성
   - ToxiGen: 유해성 탐지
```

**모델 릴리스 예시:**

```
New LLM Release Announcement:
"Our model achieves state-of-the-art performance:
- MMLU: 87.5%
- GSM8K: 92.3%
- HumanEval: 85.1%
- TruthfulQA: 78.9%"
```

---

# Part 2: 2025년 현재 트렌드

이제 2025년 현재 트렌딩하고 있는 토픽들을 살펴봅니다. 이 내용은 Final 시험 범위에 **포함되지 않습니다**.

## 1. Vision Transformer (ViT)

### 1.1. 동기: Transformer를 Vision에 적용할 수 있을까?

**Self-Attention의 본질**

```
Self-Attention의 핵심:
- Query가 있음
- 다른 elements (Keys, Values)가 있음
- Query와 관련있는 elements 찾기

지금까지 elements = Text tokens (벡터)

질문: Elements를 이미지의 부분으로 바꾸면?
```

**Computer Vision의 전통적 접근**

```
Convolutional Neural Networks (CNN):
- Sliding window로 이미지 스캔
- Local patterns 인식
- Inductive bias: 이미지의 구조 가정

예시:
[3x3 filter]를 이미지 위로 slide
  → 엣지 감지
  → 텍스처 인식
  → 객체 인식
```

### 1.2. ViT 아키텍처

**핵심 아이디어: BERT를 Vision에 적용**

```
BERT (Encoder-only):
  Text tokens → Transformer Encoder → CLS embedding → Classification

ViT:
  Image patches → Transformer Encoder → CLS embedding → Classification
```

**왜 Encoder-only?**

```
Classification Task:
- 이미지가 무엇인지 분류
- 텍스트 생성 불필요
- BERT와 동일한 패러다임
```

### 1.3. Image Patching

**이미지를 토큰으로 변환**

```python
# Original Image: 224 x 224 x 3 (RGB)

# Step 1: Divide into patches
Patch size: 16 x 16
Number of patches: (224/16) x (224/16) = 14 x 14 = 196 patches

# Step 2: Flatten each patch
Each patch: 16 x 16 x 3 = 768 values
Flatten → 768-dim vector

# Step 3: Linear projection
patch_embedding = Linear(768, d_model)

# Result: 196 patch tokens
```

**시각화:**

```
Original Image:
┌─────────────────┐
│  ┌──┬──┬──┬──┐  │
│  │P1│P2│P3│P4│  │
│  ├──┼──┼──┼──┤  │
│  │P5│P6│P7│P8│  │
│  ├──┼──┼──┼──┤  │
│  │..│..│..│..│  │
│  └──┴──┴──┴──┘  │
└─────────────────┘

Tokens:
[CLS] [P1] [P2] [P3] ... [P196]
```

### 1.4. End-to-End 예시

**Teddy Bear 이미지 분류**

```python
# 1. Image → Patches
image = load_image("teddy_bear.jpg")  # 224x224x3
patches = divide_into_patches(image, patch_size=16)  # 196 patches

# 2. Patch Embedding
patch_tokens = []
for patch in patches:
    flat_patch = flatten(patch)  # 768 values
    embedding = linear_projection(flat_patch)  # d_model dim
    patch_tokens.append(embedding)

# 3. Add Position Embeddings
position_embeddings = learned_positions(196)
patch_tokens = [p + pos for p, pos in zip(patch_tokens, position_embeddings)]

# 4. Add CLS token
cls_token = learned_cls_embedding()
tokens = [cls_token] + patch_tokens

# 5. Transformer Encoder
for layer in transformer_layers:
    tokens = layer(tokens)  # Self-attention + FFN

# 6. Classification
cls_output = tokens[0]  # CLS token의 output
logits = classification_head(cls_output)  # [1000 classes]

prediction = argmax(logits)
# "Teddy Bear" (Class 853)
```

### 1.5. Inductive Bias

**CNN vs ViT**

```
CNN (Strong Inductive Bias):
- Local connectivity: 이웃 픽셀만 연결
- Translation invariance: 위치 무관하게 동일 패턴
- Hierarchical structure: Low → High level features

장점: 적은 데이터로도 학습 가능
단점: 유연성 제한

ViT (Low Inductive Bias):
- Global connectivity: 모든 patch가 상호작용
- Position은 embedding으로 학습
- Flat structure (레이어만 stacking)

장점: 더 유연함
단점: 많은 데이터 필요
```

**실험 결과 (2020 ViT Paper)**

```
Small dataset (ImageNet-1K, 1.3M images):
  CNN (ResNet): 85% accuracy
  ViT: 78% accuracy
  → CNN이 더 좋음 (inductive bias 덕분)

Large dataset (JFT-300M, 300M images):
  CNN (ResNet): 87% accuracy
  ViT: 90% accuracy
  → ViT가 더 좋음 (충분한 데이터!)
```

**핵심 결론:**

```
If you have a LOT of data:
  → ViT outperforms CNN
  → Low inductive bias가 장점으로 작용

If you have limited data:
  → CNN still better
  → Inductive bias가 중요
```

## 2. Multimodal LLMs

### 2.1. Vision-Language Models (VLM)

**목표: 이미지에 대한 질문 답변**

```
Input:
  - Image: [사진]
  - Text: "이 사진에 무엇이 있나요?"

Output:
  - Text: "테이블 위에 고양이가 앉아있습니다."
```

**챌린지:**

```
두 가지 다른 modality:
1. Image: Continuous, 2D structure
2. Text: Discrete, 1D sequence

어떻게 함께 처리할까?
```

### 2.2. 두 가지 접근 방식

**Method 1: Early Fusion (더 일반적)**

```
┌─────────┐
│  Image  │ → Vision Encoder → Image Tokens
└─────────┘                          ↓
                           [Concat with Text]
┌─────────┐                          ↓
│  Text   │ → Tokenizer  → Text Tokens
└─────────┘                          ↓
                           Decoder-only LLM
                                    ↓
                              Generated Text
```

**구체적인 예시:**

```python
# Image processing
image = load_image("cat.jpg")
image_tokens = vision_encoder(image)  # [256 tokens]

# Text processing
text = "What's in this image?"
text_tokens = tokenizer(text)  # [10 tokens]

# Concatenate
input_tokens = [
    "<image_start>",
    *image_tokens,  # 256 image tokens
    "<image_end>",
    *text_tokens    # 10 text tokens
]

# Generate
output = llm.generate(input_tokens)
# "This image shows a cat sitting on a table."
```

**Method 2: Cross-Attention (덜 일반적)**

```
┌─────────┐
│  Text   │ → Tokenizer → Text Tokens
└─────────┘                   ↓
                         Self-Attention
                              ↓
                    Cross-Attention ← Image Tokens
                              ↓                ↑
                         Feed-Forward         │
                                              │
┌─────────┐                                   │
│  Image  │ → Vision Encoder ─────────────────┘
└─────────┘
```

**비교:**

```
Early Fusion (Method 1):
  + 구현 간단
  + 더 일반적
  + 예시: LAVA, GPT-4V

Cross-Attention (Method 2):
  + 더 정교한 상호작용
  - 구현 복잡
  + 예시: Llama 3 (paper에서 언급)
```

### 2.3. LAVA 모델

**Architecture**

```
┌──────────────────┐
│  Image           │
│  [224x224x3]     │
└────────┬─────────┘
         │
    ┌────▼────┐
    │  CLIP   │ Vision Encoder
    │ Encoder │
    └────┬────┘
         │
    ┌────▼────────┐
    │ Projection  │ Image → LLM token space
    └────┬────────┘
         │
         │ Image Tokens
         ▼
    ┌─────────────────┐
    │                 │
    │   LLaMA LLM     │ ← Text Tokens
    │  (Decoder-only) │
    └────────┬────────┘
             │
    Generated Response
```

**Training Process**

```
Stage 1: Projection Layer Training
  - Vision encoder: Frozen (CLIP)
  - LLM: Frozen (LLaMA)
  - Projection: Trainable
  목표: Image tokens를 LLM이 이해할 수 있게 변환

Stage 2: Full Fine-tuning
  - Vision encoder: Frozen
  - LLM: Trainable
  - Projection: Trainable
  목표: Instruction following
```

**사용 예시:**

```python
# Load model
model = LAVA()

# Prepare inputs
image = load_image("scene.jpg")
prompt = "USER: <image>\nWhat objects are in this room?\nASSISTANT:"

# Generate
response = model.generate(image, prompt)
# "I can see a sofa, a coffee table, and a bookshelf in this room."
```

## 3. Diffusion-based LLMs

### 3.1. 동기: Autoregressive의 한계

**현재 LLM의 생성 방식**

```
Autoregressive Generation:
Token 1 → Token 2 → Token 3 → ... → Token N

문제점:
- 순차적 (Sequential)
- 병렬화 불가능 (Inference time)
- 긴 시퀀스 = 느린 생성
```

**구체적인 예시:**

```python
# Generate "The cat sat on the mat"

Step 1: [] → "The"
Step 2: ["The"] → "cat"
Step 3: ["The", "cat"] → "sat"
Step 4: ["The", "cat", "sat"] → "on"
Step 5: ["The", "cat", "sat", "on"] → "the"
Step 6: ["The", "cat", "sat", "on", "the"] → "mat"

총 6 steps (순차적)
→ GPU 병렬화 못함
```

**Training vs Inference**

```
Training (병렬 가능):
Input: ["The", "cat", "sat", "on", "the", "mat"]
       → Causal mask로 한번에 처리

Inference (순차적):
각 토큰을 하나씩 생성
→ 이 부분이 병목!
```

### 3.2. Diffusion이란?

**Vision에서의 Diffusion**

```
Image Generation:

Forward Process (Training):
Clean Image → Add Noise → ... → Pure Noise
[사진]      [약간 노이즈] ... [완전 노이즈]

Reverse Process (Generation):
Pure Noise → Denoise → ... → Clean Image
[완전 노이즈] [약간 노이즈] ... [사진]
```

**왜 Vision에서 잘 작동하는가?**

```
Images are Continuous:
- 픽셀 값: 0.0 ~ 255.0 (실수)
- 약간의 노이즈 추가 가능
- 점진적으로 denoise 가능

예시:
픽셀 [100, 150, 200]
→ 노이즈 추가: [105, 148, 203]
→ 더 추가: [110, 145, 207]
...
```

**Diffusion의 장점**

```
Parallel Generation:
- 모든 픽셀을 동시에 생성
- 점진적으로 개선
- GPU 병렬화 가능
```

### 3.3. Text에 Diffusion 적용하기

**문제: Text는 Discrete!**

```
Text Tokens:
"cat" (ID: 5234) → discrete
"dog" (ID: 7821) → discrete

문제:
"cat" + noise = ???
중간 값이 없음!

5234 + noise = 5239 → 다른 토큰!
```

**해결 방법들**

**Approach 1: Embedding Space에서 Diffusion**

```python
# Token → Continuous embedding
token = "cat"
embedding = token_embedding(token)  # [0.1, -0.3, 0.5, ...]

# Add noise to embedding
noisy_embedding = embedding + noise

# Denoise
denoised_embedding = diffusion_model.denoise(noisy_embedding)

# Embedding → Token
output_token = nearest_token(denoised_embedding)
```

**Approach 2: Logit Space에서 Diffusion**

```python
# Start from uniform distribution
logits = uniform([vocab_size])  # 모든 토큰 동일 확률

# Iterative refinement
for step in diffusion_steps:
    logits = diffusion_model.denoise(logits)
    # 점점 특정 토큰에 확률이 집중됨

# Final token
token = argmax(logits)
```

**Approach 3: Discrete Diffusion**

```python
# Discrete state space에서 직접 diffusion
# Mask tokens 사용

Start: [MASK] [MASK] [MASK] [MASK] [MASK]
Step 1: [The] [MASK] [MASK] [MASK] [MASK]
Step 2: [The] [cat] [MASK] [MASK] [MASK]
Step 3: [The] [cat] [sat] [MASK] [MASK]
Step 4: [The] [cat] [sat] [on] [MASK]
Step 5: [The] [cat] [sat] [on] [mat]
```

### 3.4. 최근 발전

**Google's Experimental Model (2024)**

```
발표 내용:
- Text diffusion model
- 2-4배 빠른 생성 속도
- 비슷한 품질

핵심 기술:
- Continuous embedding diffusion
- Efficient denoising network
```

**Inception AI (2024)**

```
발표 내용:
- Diffusion-based LLM
- 새로운 training paradigm
- 병렬 생성 가능

헤드라인:
"Breaking the sequential bottleneck"
```

**챌린지**

```
아직 해결해야 할 문제들:
1. 품질: Autoregressive만큼 좋은가?
2. 일관성: 긴 텍스트에서 coherence 유지
3. 학습: Training이 안정적인가?
4. 실용성: 실제 production에 사용 가능한가?

현재 상태:
- 매우 활발한 연구 분야
- Promising results
- Production-ready는 아직...
```

**왜 주목해야 하는가?**

```
Potential Benefits:
1. 더 빠른 추론 (병렬화)
2. 더 유연한 생성 (순서 무관)
3. 새로운 applications

Autoregressive 대비:
✅ 속도 (병렬화)
❓ 품질 (아직 연구 중)
❓ 안정성 (아직 연구 중)
```

## 4. Transformer의 확장

**Transformer는 Text에만 국한되지 않습니다!**

```
원래: Machine Translation (Text → Text)
  ↓
확장: Other Text Tasks
  ↓
확장: Vision (Image Understanding, Generation)
  ↓
확장: Audio (Speech Recognition, Generation)
  ↓
확장: Recommendation Systems
  ↓
확장: ...
```

**Vision 분야**

```
Image Understanding:
- ViT (Vision Transformer)
- CLIP (Contrastive Learning)
- DINOv2 (Self-supervised)

Image Generation:
- Diffusion Transformer (DiT)
- Stable Diffusion (U-Net + Attention)
- DALL-E 3
```

**Audio 분야**

```
Speech Recognition:
- Whisper (OpenAI)
- Wav2Vec 2.0 (Meta)

Speech Generation:
- VALL-E (Microsoft)
- AudioLM (Google)
```

**Recommendation 분야**

```
Self-Attention for User Behavior:
- SASRec (Self-Attentive Sequential Rec)
- BERT4Rec (BERT for Recommendation)

Item features → Transformer → Recommendations
```

**핵심 메시지**

```
Transformer는 범용 아키텍처:
- Self-attention은 범용 메커니즘
- 다양한 도메인에 적용 가능
- 각 도메인에 맞게 adaptation 필요

공통 패턴:
1. Input을 token-like units로 변환
2. Position information 추가
3. Self-attention으로 관계 학습
4. Task-specific head로 출력
```

---

# Part 3: 요약 및 다음 단계

## 1. 코스 전체 요약

**Journey Through CME 295**

```
Week 1-2: Transformer 기초
  - Tokenization, Embeddings
  - Self-Attention
  - Transformer Architecture

Week 3-4: Improvements & Training
  - RoPE, GQA, RMSNorm
  - Scaling Laws
  - Flash Attention

Week 5-6: Making LLMs Useful
  - Fine-tuning Pipeline
  - Reward Modeling
  - Reasoning (GRPO)

Week 7-8: Agents & Evaluation
  - RAG, Tool Calling
  - LLM-as-a-Judge
  - Benchmarks

Week 9: Current Trends
  - Vision Transformer
  - Multimodal LLMs
  - Diffusion-based LLMs
```

## 2. 핵심 Takeaways

**Technical Skills**

```
✅ Transformer 구조 이해
✅ Position embeddings (RoPE 등)
✅ Attention 최적화 (GQA, Flash)
✅ Training pipeline (Pre-training → SFT → RLHF)
✅ Reasoning techniques (CoT, GRPO)
✅ RAG 구현
✅ Evaluation 방법
```

**Conceptual Understanding**

```
✅ Scaling laws의 중요성
✅ Autoregressive의 장단점
✅ Reward modeling의 원리
✅ LLM의 한계와 해결 방법
✅ 최신 트렌드 방향
```

**Practical Knowledge**

```
✅ 언제 어떤 기법을 사용할지
✅ Trade-offs 이해
✅ Production considerations
✅ 최신 모델들의 선택 이유
```

## 3. Final 시험 범위

**포함되는 내용 (Lecture 5-8)**

```
✅ Lecture 5: LLM Tuning
   - SFT, Preference Tuning
   - Reward Modeling (Bradley-Terry)
   - PPO

✅ Lecture 6: LLM Reasoning
   - Chain of Thought
   - GRPO vs PPO
   - GRPO Done Right, DAPO

✅ Lecture 7: Agentic LLMs
   - RAG (Retrieval, Augmentation, Generation)
   - Bi-encoder vs Cross-encoder
   - Tool Calling

✅ Lecture 8: LLM Evaluation
   - Traditional metrics (BLEU, ROUGE)
   - LLM-as-a-Judge
   - Biases (Position, Verbosity, Self-enhancement)
   - Benchmarks (MMLU, GSM8K, HumanEval, etc.)
```

**포함되지 않는 내용**

```
❌ Lecture 9 (이번 강의):
   - Vision Transformer
   - Multimodal LLMs
   - Diffusion-based LLMs
```

**시험 준비 팁**

```
1. 핵심 개념 이해
   - 수식의 의미
   - 알고리즘의 동작 원리
   - Trade-offs

2. 구체적인 예시
   - RAG의 2-step process
   - GRPO vs PPO 차이
   - Reward modeling 학습 과정

3. 실전 적용
   - 언제 무엇을 사용하는가
   - 문제 해결 접근 방법
```

---

# 용어 정리

## Vision 관련

- **ViT (Vision Transformer)**: Transformer를 이미지 분류에 적용한 모델
- **Patch**: 이미지를 고정 크기로 분할한 단위
- **Patch Embedding**: 각 patch를 벡터로 변환하는 과정
- **Inductive Bias**: 모델이 가지고 있는 사전 가정
- **CNN (Convolutional Neural Network)**: 전통적인 컴퓨터 비전 모델
- **Local Connectivity**: 인접한 픽셀만 연결하는 구조
- **Translation Invariance**: 위치와 무관하게 동일한 패턴 인식

## Diffusion 관련

- **Diffusion Model**: 노이즈 제거를 통해 데이터를 생성하는 모델
- **Forward Process**: 깨끗한 데이터에 점진적으로 노이즈를 추가하는 과정
- **Reverse Process**: 노이즈가 있는 데이터를 점진적으로 깨끗하게 만드는 과정
- **Denoising**: 노이즈를 제거하는 과정
- **Continuous Space**: 연속적인 값을 가지는 공간 (이미지 픽셀)
- **Discrete Space**: 불연속적인 값을 가지는 공간 (텍스트 토큰)
- **Embedding Space Diffusion**: 임베딩 공간에서 diffusion 수행
- **Logit Space Diffusion**: Logit 공간에서 diffusion 수행
- **Discrete Diffusion**: Discrete 공간에서 직접 diffusion 수행

## Multimodal 관련

- **VLM (Vision-Language Model)**: 이미지와 텍스트를 함께 처리하는 모델
- **Early Fusion**: 입력 단계에서 modality를 결합
- **Cross-Attention**: 한 modality가 다른 modality를 참조하는 attention
- **Vision Encoder**: 이미지를 임베딩으로 변환하는 모듈
- **Projection Layer**: 한 공간을 다른 공간으로 변환하는 레이어
- **CLIP**: 이미지와 텍스트를 동일한 임베딩 공간으로 매핑하는 모델
- **LAVA**: 대표적인 open-source VLM
- **Modality**: 데이터의 형태 (이미지, 텍스트, 오디오 등)

---

**수고하셨습니다!** 🎉

이것으로 CME 295의 모든 강의가 끝났습니다. Final 시험을 잘 준비하시고, 앞으로도 LLM 분야의 발전을 계속 주시하시기 바랍니다!

**Final 시험 준비:**
- Lecture 5-8 복습
- 핵심 개념과 예시 숙지
- Trade-offs 이해
- 실전 적용 능력

**향후 학습 방향:**
- 최신 논문 follow
- 오픈소스 모델 실험
- Production 배포 경험
- 새로운 트렌드 주시
