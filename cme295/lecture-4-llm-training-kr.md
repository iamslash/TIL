# Lecture 4: LLM Training

# Materials

- [CME 295](https://cme295.stanford.edu/syllabus/)
- [slide](https://cme295.stanford.edu/slides/fall25-cme295-lecture4.pdf)
- [video](https://www.youtube.com/watch?v=VlA_jt_3Qc4&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=4)

# Table of Contents

- [Lecture 4: LLM Training](#lecture-4-llm-training)
- [Materials](#materials)
- [Table of Contents](#table-of-contents)
- [강의 개요](#강의-개요)
  - [강의 목표](#강의-목표)
  - [주요 학습 내용](#주요-학습-내용)
- [1. Transfer Learning과 LLM Training Paradigm](#1-transfer-learning과-llm-training-paradigm)
  - [1.1. 전통적인 ML vs Transfer Learning](#11-전통적인-ml-vs-transfer-learning)
  - [1.2. LLM Training의 두 단계](#12-llm-training의-두-단계)
- [2. Pretraining (사전학습)](#2-pretraining-사전학습)
  - [2.1. Pretraining이란?](#21-pretraining이란)
  - [2.2. Training Data](#22-training-data)
  - [2.3. FLOPs와 FLOPS](#23-flops와-flops)
    - [FLOPs (Floating Point Operations)](#flops-floating-point-operations)
    - [FLOPS (Floating Point Operations Per Second)](#flops-floating-point-operations-per-second)
  - [2.4. Scaling Laws](#24-scaling-laws)
  - [2.5. Chinchilla Law](#25-chinchilla-law)
  - [2.6. Pretraining의 과제](#26-pretraining의-과제)
- [3. Training Optimizations (학습 최적화)](#3-training-optimizations-학습-최적화)
  - [3.1. Training Process 복습](#31-training-process-복습)
  - [3.2. GPU Memory 제약](#32-gpu-memory-제약)
- [4. Data Parallelism (데이터 병렬화)](#4-data-parallelism-데이터-병렬화)
  - [4.1. Data Parallelism 개념](#41-data-parallelism-개념)
  - [4.2. ZeRO (Zero Redundancy Optimizer)](#42-zero-zero-redundancy-optimizer)
    - [ZeRO Stage 1: Optimizer State Sharding](#zero-stage-1-optimizer-state-sharding)
    - [ZeRO Stage 2: Gradient Sharding](#zero-stage-2-gradient-sharding)
    - [ZeRO Stage 3: Parameter Sharding](#zero-stage-3-parameter-sharding)
- [5. Model Parallelism (모델 병렬화)](#5-model-parallelism-모델-병렬화)
  - [5.1. Model Parallelism의 종류](#51-model-parallelism의-종류)
    - [1. Tensor Parallelism](#1-tensor-parallelism)
    - [2. Pipeline Parallelism](#2-pipeline-parallelism)
    - [3. Expert Parallelism (MOE)](#3-expert-parallelism-moe)
- [6. Flash Attention](#6-flash-attention)
  - [6.1. GPU Memory 구조](#61-gpu-memory-구조)
  - [6.2. 표준 Attention의 문제](#62-표준-attention의-문제)
  - [6.3. Flash Attention의 핵심 아이디어](#63-flash-attention의-핵심-아이디어)
    - [1. Tiling (블록 단위 계산)](#1-tiling-블록-단위-계산)
  - [6.4. Tiling과 Recomputation](#64-tiling과-recomputation)
    - [2. Recomputation (재계산)](#2-recomputation-재계산)
  - [6.5. Flash Attention의 효과](#65-flash-attention의-효과)
- [7. Quantization (양자화)](#7-quantization-양자화)
  - [7.1. Quantization이란?](#71-quantization이란)
  - [7.2. Floating Point 표현](#72-floating-point-표현)
  - [7.3. Mixed Precision Training](#73-mixed-precision-training)
- [8. Supervised Fine-tuning (SFT)](#8-supervised-fine-tuning-sft)
  - [8.1. Pretraining만으로는 부족한 이유](#81-pretraining만으로는-부족한-이유)
  - [8.2. SFT란?](#82-sft란)
  - [8.3. Instruction Data](#83-instruction-data)
  - [8.4. SFT 구현](#84-sft-구현)
- [9. 요약](#9-요약)
  - [LLM Training Pipeline](#llm-training-pipeline)
  - [Training Optimizations 요약](#training-optimizations-요약)
  - [실전 체크리스트](#실전-체크리스트)
- [10. 중요 용어 정리](#10-중요-용어-정리)
  - [Training 관련](#training-관련)
  - [Optimization 관련](#optimization-관련)
  - [Hardware 관련](#hardware-관련)
  - [Fine-tuning 관련](#fine-tuning-관련)

---

# 강의 개요

## 강의 목표

이번 강의에서는 LLM이 어떻게 학습되는지 전체 과정을 이해합니다.

**학습 목표:**
- Transfer Learning과 LLM training paradigm 이해
- Pretraining 과정과 scaling laws
- Training optimization 기법들
- Supervised Fine-tuning (SFT)

## 주요 학습 내용

**1. LLM Training Paradigm**
- Transfer Learning의 개념
- Pretraining → Fine-tuning 파이프라인

**2. Pretraining**
- Next token prediction
- Training data와 규모
- Scaling laws와 Chinchilla law

**3. Training Optimizations**
- Data Parallelism & ZeRO
- Model Parallelism
- Flash Attention
- Quantization & Mixed Precision

**4. Supervised Fine-tuning**
- Instruction-following 학습
- SFT 데이터와 방법

---

# 1. Transfer Learning과 LLM Training Paradigm

## 1.1. 전통적인 ML vs Transfer Learning

**전통적인 ML 접근법:**

```
과거 (10년 전):
각 작업마다 독립적인 모델 학습

Task 1: Spam Detection
└─> 처음부터 모델 학습
    └─> Train → Val → Test

Task 2: Sentiment Analysis
└─> 처음부터 모델 학습
    └─> Train → Val → Test

Task 3: Named Entity Recognition
└─> 처음부터 모델 학습
    └─> Train → Val → Test

문제:
- 매번 처음부터 시작
- 이전 학습 지식 재사용 안함
- 비효율적
```

**Transfer Learning 접근법:**

```
핵심 통찰:
모든 NLP 작업은 "언어 이해"라는 공통점이 있음!

└─> 언어의 일반적 지식을 먼저 학습
    └─> 이를 각 작업에 맞게 조정

┌────────────────────────────────┐
│ Step 1: Pre-training           │
│ 대규모 데이터로 언어 구조 학습 │
│ → Base Model 획득              │
└───────────┬────────────────────┘
            │
            ├──> Task 1: Spam Detection
            │    Base Model을 미세조정
            │
            ├──> Task 2: Sentiment Analysis
            │    Base Model을 미세조정
            │
            └──> Task 3: NER
                 Base Model을 미세조정

장점:
- 공통 지식 재사용
- 작업별 학습 데이터 적게 필요
- 더 나은 성능
```

**비유:**

```
전통적 ML = 요리사가 매번 식재료부터 준비
Transfer Learning = 기본 조리법을 배운 후 각 요리에 적용

예:
기본 조리 기술 학습 (Pre-training)
└─> 볶기, 끓이기, 굽기 등

특정 요리 조정 (Fine-tuning)
├─> 파스타 만들기
├─> 스테이크 굽기
└─> 케이크 굽기
```

## 1.2. LLM Training의 두 단계

**전체 파이프라인:**

```
┌─────────────────────────────────────────────┐
│ Stage 1: Pretraining (가장 비쌈)            │
│                                             │
│ 목표: 언어/코드의 구조 학습                  │
│ 데이터: 인터넷의 모든 텍스트                 │
│ 규모: 수백억 ~ 수조 tokens                  │
│ 비용: 수백만 ~ 수억 달러                     │
│ 시간: 수주 ~ 수개월                         │
│                                             │
│ 학습 방법: Next Token Prediction            │
│ "The cat sat on the" → "mat"                │
│                                             │
│ 결과: Base Model                            │
│ - 언어 구조 이해                            │
│ - 하지만 도움이 되는 assistant 아님          │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│ Stage 2: Fine-tuning (상대적으로 저렴)       │
│                                             │
│ 목표: 특정 작업에 맞게 조정                  │
│ 데이터: 작업별 고품질 데이터                 │
│ 규모: 수천 ~ 수만 examples                  │
│ 비용: 상대적으로 낮음                        │
│ 시간: 수시간 ~ 수일                         │
│                                             │
│ 학습 방법: Supervised Learning              │
│ "질문: ... → 답변: ..."                     │
│                                             │
│ 결과: Task-specific Model                   │
│ - 유용한 assistant                          │
│ - 또는 특정 도메인 전문가                    │
└─────────────────────────────────────────────┘
```

**비용 비교:**

```
Pretraining:
- 비용: $1,000,000 ~ $100,000,000+
- 시간: 수주 ~ 수개월
- 데이터: 수조 tokens
- GPU: 수천 개

Fine-tuning:
- 비용: $1,000 ~ $100,000
- 시간: 수시간 ~ 수일
- 데이터: 수만 examples
- GPU: 수십 개

비율: Pretraining이 1000배+ 더 비쌈!
```

---

# 2. Pretraining (사전학습)

## 2.1. Pretraining이란?

**정의:**

```
Pretraining = 대규모 데이터로 언어의 일반적 구조를 학습하는 과정

목표:
"다음 단어를 예측하라"

입력: "The cat sat on the"
목표: "mat"

입력: "I love machine"
목표: "learning"

→ 모델이 언어의 패턴, 문법, 의미를 학습
```

**학습 과정:**

```python
# Pseudo-code
for batch in massive_dataset:
    # 입력: 일부 텍스트
    input_text = "The cat sat on"

    # 목표: 다음 토큰
    target = "the"

    # Forward pass
    prediction = model(input_text)

    # Loss 계산
    loss = cross_entropy(prediction, target)

    # Backward pass & update
    loss.backward()
    optimizer.step()
```

**왜 Next Token Prediction인가?**

```
1. 자기지도학습 (Self-supervised)
   - 레이블 필요 없음
   - 텍스트 자체가 레이블
   - 무한한 데이터 활용 가능

2. 언어의 모든 측면 학습
   - 문법: "The cats ___" → "are" (not "is")
   - 의미: "Fire is ___" → "hot"
   - 상식: "Birds can ___" → "fly"
   - 패턴: 코드, 수식 등

3. 범용적
   - 하나의 목표로 모든 것 학습
   - 복잡한 작업 필요 없음
```

## 2.2. Training Data

**데이터 출처:**

```
┌─────────────────────────────────────────┐
│ 1. Web Crawls (웹 크롤링)               │
├─────────────────────────────────────────┤
│ • Common Crawl                          │
│   - 월 30억 페이지                      │
│   - 인터넷의 거의 모든 것               │
│   - 다양한 언어, 주제                   │
│                                         │
│ • 블로그, 뉴스, 포럼                    │
│ • 위키피디아                            │
│ • 소셜 미디어 (Reddit 등)               │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ 2. Code (코드)                          │
├─────────────────────────────────────────┤
│ • GitHub repositories                   │
│ • Stack Overflow                        │
│ • 다양한 프로그래밍 언어                 │
│   - Python, JavaScript, C++, ...        │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ 3. Books & Academic Papers              │
├─────────────────────────────────────────┤
│ • 전자책                                │
│ • 학술 논문 (arXiv 등)                  │
│ • 전문 서적                             │
└─────────────────────────────────────────┘
```

**데이터 규모:**

```
Order of Magnitude: 수백억 ~ 수조 tokens

예시:

GPT-3 (2020):
- 300 billion tokens
- ~45TB of text

Llama 2 (2023):
- 2 trillion tokens
- ~300TB of text

Llama 3 (2024):
- 15 trillion tokens
- ~2.25PB of text

비교:
- 1권의 책 ≈ 100,000 tokens
- Wikipedia 전체 ≈ 3 billion tokens
- GPT-3 = Wikipedia × 100
- Llama 3 = Wikipedia × 5,000
```

## 2.3. FLOPs와 FLOPS

**두 가지 중요한 개념:**

### FLOPs (Floating Point Operations)

```
FLOPs = 계산량의 단위

정의:
부동소수점 연산의 총 개수

예시:
행렬 곱셈 (A × B):
- A: (m × n)
- B: (n × p)
- FLOPs ≈ 2 × m × n × p

LLM Training:
- 규모: 10^24 ~ 10^26 FLOPs
- 10^24 = 1 septillion (1자)
```

**FLOPs 계산 (근사):**

```
FLOPs ≈ 6 × N × D

여기서:
- N: 모델 파라미터 수
- D: 학습 데이터 토큰 수
- 6: 상수 (forward + backward)

예시:
GPT-3:
- N = 175B parameters
- D = 300B tokens
- FLOPs ≈ 6 × 175B × 300B
         ≈ 3.15 × 10^23

Llama 3 405B:
- N = 405B parameters
- D = 15T tokens
- FLOPs ≈ 6 × 405B × 15T
         ≈ 3.6 × 10^25
```

### FLOPS (Floating Point Operations Per Second)

```
FLOPS = 계산 속도의 단위

정의:
초당 부동소수점 연산 수
= 하드웨어의 처리 속도

GPU 예시:

NVIDIA H100:
- FP32: 60 TFLOPS
  (60 trillion operations/second)
- FP16: 1,979 TFLOPS
  (약 2 quadrillion operations/second)

NVIDIA A100:
- FP32: 19.5 TFLOPS
- FP16: 312 TFLOPS
```

**Training Time 계산:**

```
Training Time = FLOPs / (FLOPS × num_GPUs × efficiency)

예시:
GPT-3 학습 시간:

FLOPs = 3.15 × 10^23
GPUs = 10,000 × A100
FLOPS per GPU = 312 × 10^12 (FP16)
Efficiency = 0.5 (50%)

Time = 3.15 × 10^23 / (312 × 10^12 × 10,000 × 0.5)
     ≈ 202,000 seconds
     ≈ 56 hours
     ≈ 2.3 days

실제로는:
- 더 오래 걸림 (수주)
- 통신 오버헤드
- 시스템 불안정성
- 여러 번 재시도
```

## 2.4. Scaling Laws

**Scaling Laws = 모델 성능과 규모의 관계**

**2020년 OpenAI 논문의 발견:**

```
"Scaling Laws for Neural Language Models"

핵심 발견:
Performance ∝ f(Compute, Model Size, Data Size)

1. 더 많은 Compute → 더 나은 성능
2. 더 큰 Model → 더 나은 성능
3. 더 많은 Data → 더 나은 성능
```

**시각화:**

```
Loss (낮을수록 좋음)
  ↑
  │
4 │●
  │
3 │  ●
  │
2 │    ●
  │
1 │      ●
  │        ●●●
0 │           ●●●●●●
  └─────────────────────> Compute (FLOPs)
  10^18  10^20  10^22  10^24

관찰:
- 계속 개선됨
- Smooth한 곡선
- 예측 가능
```

**Sample Efficiency:**

```
큰 모델 = 더 Sample Efficient

예시:
동일한 성능 달성을 위해:

Small Model (1B params):
- 1T tokens 필요

Large Model (100B params):
- 100B tokens만 필요

→ 10배 적은 데이터!
```

**왜 중요한가?**

```
2019-2024 트렌드:
모델이 점점 커짐

GPT-2 (2019):   1.5B
GPT-3 (2020):   175B
PaLM (2022):    540B
GPT-4 (2023):   ~1.7T (추정)

이유:
Scaling laws에 따르면
더 크면 더 좋기 때문!
```

## 2.5. Chinchilla Law

**핵심 질문:**

```
주어진 Compute 예산으로
어떻게 최적으로 학습할까?

두 가지 선택:
1. 큰 모델 + 적은 데이터
2. 작은 모델 + 많은 데이터

어느 것이 더 좋을까?
```

**Chinchilla 논문 (2022):**

```
"Training Compute-Optimal Large Language Models"

실험:
- 고정된 Compute budget
- 다양한 (Model Size, Data Size) 조합 시도

발견:
최적 비율 = Data : Model ≈ 20 : 1

공식:
D_optimal ≈ 20 × N

여기서:
- D: tokens 수
- N: parameters 수
```

**Chinchilla Table:**

```
┌────────────┬───────────────┬──────────────────┐
│ Parameters │ Optimal Tokens│ Compute (FLOPs)  │
├────────────┼───────────────┼──────────────────┤
│ 400M       │ 8B            │ 1.9 × 10^19      │
│ 1B         │ 20B           │ 1.2 × 10^20      │
│ 10B        │ 200B          │ 1.2 × 10^22      │
│ 67B        │ 1.3T          │ 5.4 × 10^23      │
│ 175B       │ 3.5T          │ 3.7 × 10^24      │
│ 280B       │ 5.6T          │ 9.4 × 10^24      │
│ 520B       │ 10.4T         │ 3.2 × 10^25      │
└────────────┴───────────────┴──────────────────┘
```

**실제 모델 비교:**

```
GPT-3:
- Parameters: 175B
- Tokens: 300B
- Ratio: 300B / 175B ≈ 1.7

Chinchilla 권장: 175B × 20 = 3.5T tokens
→ GPT-3는 undertrained! (10배 이상)

Llama 2 70B:
- Parameters: 70B
- Tokens: 2T
- Ratio: 2T / 70B ≈ 28.6

Chinchilla 권장: 70B × 20 = 1.4T tokens
→ Llama 2는 overtrained (약간)

Chinchilla 70B:
- Parameters: 70B
- Tokens: 1.4T
- Ratio: 정확히 20:1
→ Compute-optimal!
```

**왜 Undertrain하는가?**

```
이유:
1. Inference Cost
   - 큰 모델은 inference 비쌈
   - 더 오래 학습해도 inference는 빠름

2. Overtrain의 이점
   - 성능이 조금 더 좋을 수 있음
   - Diminishing returns는 있지만

3. 실전 고려사항
   - Training은 한 번
   - Inference는 수백만 번
   - → Inference 효율이 더 중요
```

## 2.6. Pretraining의 과제

**1. 비용 (Cost)**

```
재정적 비용:
- Minimum: $1M ~ $10M
- Typical: $10M ~ $50M
- Large models: $50M ~ $100M+

GPT-3 추정:
- Compute: ~$5M
- Total (infra, people): $10M+

GPT-4 추정:
- Compute: ~$100M
- Total: $200M+ (추정)

왜 비쌈?
- GPU 대여 비용
- 전기 비용
- 엔지니어 인건비
- 인프라 비용
```

**2. 시간 (Time)**

```
Training Duration:
- Small (1B-7B): 수일 ~ 수주
- Medium (10B-70B): 수주 ~ 수개월
- Large (100B+): 수개월

Llama 2:
- 70B model
- ~2M GPU hours
- ~3 months on 2000 GPUs

실제 문제:
- 긴 개발 주기
- 디버깅 어려움
- 재시작 비용 높음
```

**3. Knowledge Cutoff**

```
문제:
모델은 학습 데이터까지만 알고 있음

예시:
GPT-4 (knowledge cutoff: Sep 2023):

User: "2024년 미국 대통령은?"
GPT-4: "죄송하지만, 제 지식은 2023년 9월까지입니다..."

→ 최신 정보 없음!

해결책:
- Retrieval Augmented Generation (RAG)
- 정기적 재학습
- Fine-tuning
```

**4. 환경적 영향 (Environmental Cost)**

```
전력 소비:
GPT-3 학습:
- ~1,287 MWh
- ≈ 평균 가정 120년치 전기

탄소 배출:
- ~550 tons CO2
- ≈ 자동차 125대의 연간 배출량

최근 트렌드:
- 더 효율적인 하드웨어
- 재생 에너지 사용
- Carbon offset programs
```

**5. Plagiarism (표절) 위험**

```
문제:
모델이 학습 데이터를 그대로 복사?

우려:
- 저작권 침해
- 개인정보 노출
- 코드 라이센스 위반

완화 방법:
- Data filtering
- Memorization detection
- Output filtering
- Legal frameworks
```

---

# 3. Training Optimizations (학습 최적화)

## 3.1. Training Process 복습

**표준 Training Loop:**

```python
# Pseudo-code
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Forward Pass
        outputs = model(batch['input'])
        loss = criterion(outputs, batch['target'])

        # 2. Backward Pass
        loss.backward()  # Gradient 계산

        # 3. Optimizer Step
        optimizer.step()  # Weight 업데이트
        optimizer.zero_grad()
```

**메모리 사용:**

```
Training 중 저장해야 할 것들:

┌────────────────────────────────────┐
│ 1. Model Parameters (모델 가중치)  │
│    - Size: N parameters             │
│    - 예: 175B params × 4 bytes      │
│         = 700 GB (FP32)             │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│ 2. Gradients (기울기)              │
│    - Size: N (파라미터당 1개)       │
│    - 예: 175B × 4 bytes = 700 GB   │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│ 3. Optimizer States (최적화 상태)  │
│    Adam optimizer:                  │
│    - First moment (m): N            │
│    - Second moment (v): N           │
│    - 예: 175B × 2 × 4 bytes        │
│         = 1,400 GB                  │
└────────────────────────────────────┘

┌────────────────────────────────────┐
│ 4. Activations (활성화 값)         │
│    - 크기: Batch × Seq × Hidden    │
│    - 예: 32 × 2048 × 12288         │
│         × num_layers × 4 bytes      │
│         = 수십~수백 GB              │
└────────────────────────────────────┘

Total: 수 TB!
```

**Forward Pass 상세:**

```
Input: Batch of text
  ↓
┌─────────────────────┐
│ Embedding Layer     │ → Activation 저장
├─────────────────────┤
│ Transformer Block 1 │
│  - Self-Attention   │ → Activation 저장
│  - Feed-Forward     │ → Activation 저장
├─────────────────────┤
│ Transformer Block 2 │ → Activation 저장
├─────────────────────┤
│ ...                 │
├─────────────────────┤
│ Transformer Block N │ → Activation 저장
├─────────────────────┤
│ Output Layer        │
└─────────────────────┘
  ↓
Loss Computation

왜 Activation 저장?
→ Backward pass에서 gradient 계산에 필요!
```

**Backward Pass 상세:**

```
Loss
  ↓
∂Loss/∂output
  ↓
┌─────────────────────┐
│ Transformer Block N │
│  ∂L/∂params 계산    │ → Gradient 저장
├─────────────────────┤
│ ...                 │
├─────────────────────┤
│ Transformer Block 1 │
│  ∂L/∂params 계산    │ → Gradient 저장
├─────────────────────┤
│ Embedding Layer     │
│  ∂L/∂params 계산    │ → Gradient 저장
└─────────────────────┘

Chain Rule 사용:
∂L/∂W_i = ∂L/∂a_i × ∂a_i/∂W_i

→ Activation (a_i) 필요!
```

## 3.2. GPU Memory 제약

**NVIDIA H100 Spec:**

```
┌──────────────────────────────────────┐
│ NVIDIA H100 GPU                      │
├──────────────────────────────────────┤
│ GPU Memory:    80 GB                 │
│ Memory BW:     3.35 TB/s             │
│ FP32 TFLOPS:   60                    │
│ FP16 TFLOPS:   1,979                 │
│ Price:         ~$30,000              │
└──────────────────────────────────────┘
```

**문제:**

```
175B 모델 학습에 필요한 메모리:
- Parameters:      700 GB (FP32)
- Gradients:       700 GB
- Optimizer:     1,400 GB
- Activations:     100+ GB
─────────────────────────────
Total:          ~3,000 GB

H100 Memory:      80 GB

비율: 3,000 GB / 80 GB ≈ 38배 부족!

해결책:
여러 GPU 사용 + 최적화 기법
```

---

# 4. Data Parallelism (데이터 병렬화)

## 4.1. Data Parallelism 개념

**핵심 아이디어:**

```
배치를 여러 GPU에 분산

Example:
Batch size = 64
GPUs = 8

각 GPU:
- Mini-batch size = 64 / 8 = 8
- 독립적으로 forward/backward 수행
- Gradient만 통신으로 합산
```

**시각화:**

```
         Batch (64 samples)
              ↓
    ┌─────────┴──────────┐
    ↓         ↓          ↓
┌───────┐ ┌───────┐  ┌───────┐
│GPU 0  │ │GPU 1  │  │GPU 7  │
│       │ │       │  │       │
│Model  │ │Model  │  │Model  │
│Copy   │ │Copy   │  │Copy   │
│       │ │       │  │       │
│ 8     │ │ 8     │  │ 8     │
│samples│ │samples│  │samples│
└───┬───┘ └───┬───┘  └───┬───┘
    │         │          │
    └─────────┬──────────┘
              ↓
        Gradient 합산
              ↓
        Weight Update
```

**구현 예시:**

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 1. 초기화
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()

# 2. 모델 준비
model = MyLLM().to(rank)
model = DDP(model, device_ids=[rank])

# 3. Data 분산
sampler = torch.utils.data.distributed.DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank
)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=8)

# 4. Training
for batch in dataloader:
    # Forward (독립적)
    outputs = model(batch['input'].to(rank))
    loss = criterion(outputs, batch['target'].to(rank))

    # Backward (독립적)
    loss.backward()

    # Gradient averaging (자동으로 DDP가 처리)
    # All-Reduce operation

    # Update (동일하게)
    optimizer.step()
    optimizer.zero_grad()
```

**장점:**

```
1. 메모리 절약 (Activation)
   - Batch를 나눔
   - 각 GPU는 작은 batch만 처리

2. 속도 향상
   - 병렬 처리
   - Throughput 증가

3. 구현 간단
   - PyTorch DDP 사용
   - 코드 변경 최소
```

**단점:**

```
1. 모델 복사 필요
   - 각 GPU에 전체 모델
   - 175B 모델 = 각 GPU에 700GB (FP32)
   - → 여전히 메모리 부족!

2. Communication Overhead
   - Gradient 합산 필요
   - All-Reduce operation
   - 대역폭 제한

3. Scalability 제한
   - 모델이 GPU에 fit해야 함
   - 매우 큰 모델은 불가능
```

## 4.2. ZeRO (Zero Redundancy Optimizer)

**문제 인식:**

```
Data Parallelism의 중복:

GPU 0:  [Model][Grad][Opt State]
GPU 1:  [Model][Grad][Opt State]
GPU 2:  [Model][Grad][Opt State]
GPU 3:  [Model][Grad][Opt State]

→ 모든 것이 중복!
→ 비효율적!
```

**ZeRO의 핵심 아이디어:**

```
중복 제거 (Zero Redundancy)

대신 분할 (Shard):
- Parameters
- Gradients
- Optimizer States

각 GPU는 일부만 저장!
```

**ZeRO 단계:**

### ZeRO Stage 1: Optimizer State Sharding

```
Optimizer State만 분할

Before:
GPU 0: [Model][Grad][Opt_full]
GPU 1: [Model][Grad][Opt_full]
GPU 2: [Model][Grad][Opt_full]
GPU 3: [Model][Grad][Opt_full]

After (ZeRO-1):
GPU 0: [Model][Grad][Opt_0]
GPU 1: [Model][Grad][Opt_1]
GPU 2: [Model][Grad][Opt_2]
GPU 3: [Model][Grad][Opt_3]

메모리 절감:
Adam optimizer:
- Before: N × 2 × 4 bytes per GPU
- After:  N × 2 × 4 bytes / num_GPUs

Example (175B params, 4 GPUs):
- Before: 1,400 GB per GPU
- After:  350 GB per GPU
- 절감: 4배!
```

### ZeRO Stage 2: Gradient Sharding

```
Optimizer State + Gradient 분할

After (ZeRO-2):
GPU 0: [Model][Grad_0][Opt_0]
GPU 1: [Model][Grad_1][Opt_1]
GPU 2: [Model][Grad_2][Opt_2]
GPU 3: [Model][Grad_3][Opt_3]

메모리 절감:
- Gradients: N × 4 bytes / num_GPUs
- Optimizer:  N × 8 bytes / num_GPUs

Example (175B params, 4 GPUs):
- Gradients:  175 GB per GPU
- Optimizer:  350 GB per GPU
- 절감: 추가 4배!
```

### ZeRO Stage 3: Parameter Sharding

```
모든 것 분할!

After (ZeRO-3):
GPU 0: [Model_0][Grad_0][Opt_0]
GPU 1: [Model_1][Grad_1][Opt_1]
GPU 2: [Model_2][Grad_2][Opt_2]
GPU 3: [Model_3][Grad_3][Opt_3]

메모리 절감:
- Parameters: N × 4 bytes / num_GPUs
- Gradients:  N × 4 bytes / num_GPUs
- Optimizer:  N × 8 bytes / num_GPUs

Example (175B params, 4 GPUs):
- Parameters: 175 GB per GPU
- Gradients:  175 GB per GPU
- Optimizer:  350 GB per GPU
- Total:      700 GB per GPU
- 절감: 전체 4배!
```

**ZeRO-3 작동 방식:**

```python
# Forward pass with ZeRO-3
for layer in model.layers:
    # 1. All-Gather: 필요한 parameters 수집
    full_params = all_gather(layer.sharded_params)

    # 2. Forward computation
    output = layer.forward(input, full_params)

    # 3. 사용 후 버림 (메모리 절약)
    del full_params

# Backward pass
for layer in reversed(model.layers):
    # 1. All-Gather: parameters 다시 수집
    full_params = all_gather(layer.sharded_params)

    # 2. Backward computation
    grad = layer.backward(grad_output, full_params)

    # 3. Reduce-Scatter: gradient 분산
    my_grad_shard = reduce_scatter(grad)

    # 4. 버림
    del full_params, grad
```

**ZeRO 비교:**

```
┌────────┬──────────┬──────────┬──────────┬───────────┐
│ Stage  │ Params   │ Grads    │ Opt State│ Memory    │
├────────┼──────────┼──────────┼──────────┼───────────┤
│ None   │ Full     │ Full     │ Full     │ Baseline  │
│ ZeRO-1 │ Full     │ Full     │ Sharded  │ 4× better │
│ ZeRO-2 │ Full     │ Sharded  │ Sharded  │ 8× better │
│ ZeRO-3 │ Sharded  │ Sharded  │ Sharded  │ 12× better│
└────────┴──────────┴──────────┴──────────┴───────────┘

Communication:
None    < ZeRO-1 < ZeRO-2 < ZeRO-3
빠름                          느림

Memory Efficiency:
None    < ZeRO-1 < ZeRO-2 < ZeRO-3
나쁨                          좋음
```

**Trade-offs:**

```
ZeRO-1:
✓ 간단한 구현
✓ 낮은 communication overhead
✗ 제한된 메모리 절감

ZeRO-2:
✓ 좋은 균형
✓ 중간 수준의 메모리 절감
✓ 적절한 통신 비용

ZeRO-3:
✓ 최대 메모리 절감
✓ 매우 큰 모델 가능
✗ 높은 communication overhead
✗ 느린 학습 속도
```

**DeepSpeed 구현:**

```python
import deepspeed

# DeepSpeed config
ds_config = {
    "train_batch_size": 32,
    "zero_optimization": {
        "stage": 2,  # ZeRO-2
        "offload_optimizer": {
            "device": "cpu",  # CPU offload
            "pin_memory": True
        }
    },
    "fp16": {
        "enabled": True
    }
}

# 모델 초기화
model = MyLLM()
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)

# Training
for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

---

# 5. Model Parallelism (모델 병렬화)

## 5.1. Model Parallelism의 종류

**핵심 아이디어:**

```
Data Parallelism:
데이터를 나눔, 모델은 복사

Model Parallelism:
모델을 나눔, 배치는 공유
```

### 1. Tensor Parallelism

```
큰 행렬 연산을 GPU 간 분할

예: Linear layer
Y = X @ W + b

W를 분할:
W = [W1 | W2 | W3 | W4]

GPU 0: Y1 = X @ W1
GPU 1: Y2 = X @ W2
GPU 2: Y3 = X @ W3
GPU 3: Y4 = X @ W4

Final: Y = [Y1 | Y2 | Y3 | Y4]
```

**시각화:**

```
┌────────────┐
│   Input X  │
└──┬──┬──┬──┘
   │  │  │  │
   ↓  ↓  ↓  ↓
┌───┐┌───┐┌───┐┌───┐
│W1 ││W2 ││W3 ││W4 │
└─┬─┘└─┬─┘└─┬─┘└─┬─┘
  │    │    │    │
GPU0 GPU1 GPU2 GPU3
  │    │    │    │
  ↓    ↓    ↓    ↓
┌───┐┌───┐┌───┐┌───┐
│Y1 ││Y2 ││Y3 ││Y4 │
└─┬─┘└─┬─┘└─┬─┘└─┬─┘
  │    │    │    │
  └────┴────┴────┘
         ↓
    Concatenate
         ↓
    ┌────────┐
    │Output Y│
    └────────┘
```

### 2. Pipeline Parallelism

```
레이어를 GPU 간 분할

Model = [Layer1, Layer2, ..., Layer12]

GPU 0: Layer 1-3
GPU 1: Layer 4-6
GPU 2: Layer 7-9
GPU 3: Layer 10-12
```

**시각화:**

```
Input
  ↓
┌──────────┐
│ GPU 0    │
│ Layer 1-3│
└────┬─────┘
     ↓ (send to GPU 1)
┌──────────┐
│ GPU 1    │
│ Layer 4-6│
└────┬─────┘
     ↓ (send to GPU 2)
┌──────────┐
│ GPU 2    │
│ Layer 7-9│
└────┬─────┘
     ↓ (send to GPU 3)
┌──────────┐
│ GPU 3    │
│Layer10-12│
└────┬─────┘
     ↓
  Output
```

**Pipeline Bubbles 문제:**

```
문제: GPU들이 놀 수 있음

Time →
GPU 0: [████▒▒▒▒▒▒▒▒]
GPU 1: [▒▒▒▒████▒▒▒▒]
GPU 2: [▒▒▒▒▒▒▒▒████]
GPU 3: [▒▒▒▒▒▒▒▒▒▒▒▒]

해결: Micro-batching

Batch를 작은 micro-batches로 분할

GPU 0: [██▒▒██▒▒██▒▒]
GPU 1: [▒▒██▒▒██▒▒██]
GPU 2: [▒▒▒▒██▒▒██▒▒]
GPU 3: [▒▒▒▒▒▒██▒▒██]

효율 향상!
```

### 3. Expert Parallelism (MOE)

```
MOE 모델의 각 expert를 다른 GPU에 배치

Example: 8 experts, 8 GPUs

GPU 0: Expert 0
GPU 1: Expert 1
GPU 2: Expert 2
...
GPU 7: Expert 7

Token routing:
각 토큰이 해당 expert GPU로 전송
```

**시각화:**

```
         Input Tokens
              ↓
         Router Network
              ↓
    ┌─────────┴──────────┐
    ↓         ↓          ↓
┌───────┐ ┌───────┐  ┌───────┐
│GPU 0  │ │GPU 1  │  │GPU 7  │
│Expert0│ │Expert1│  │Expert7│
└───┬───┘ └───┬───┘  └───┬───┘
    │         │          │
    └─────────┬──────────┘
              ↓
        Combine Outputs
              ↓
          Output
```

**비교표:**

```
┌──────────────┬────────────┬────────────┬───────────┐
│ 방법         │ 장점       │ 단점       │ 사용 케이스│
├──────────────┼────────────┼────────────┼───────────┤
│ Tensor       │ 세밀한     │ 높은 통신  │ 매우 큰   │
│ Parallelism  │ 분할       │ 비용       │ 행렬 연산 │
├──────────────┼────────────┼────────────┼───────────┤
│ Pipeline     │ 낮은 통신  │ Bubble     │ 깊은      │
│ Parallelism  │ 비용       │ overhead   │ 네트워크  │
├──────────────┼────────────┼────────────┼───────────┤
│ Expert       │ MOE에      │ 불균형     │ MOE       │
│ Parallelism  │ 최적화     │ 로딩       │ 모델      │
└──────────────┴────────────┴────────────┴───────────┘
```

---

# 6. Flash Attention

## 6.1. GPU Memory 구조

**GPU의 두 가지 메모리:**

```
┌─────────────────────────────────────────┐
│ HBM (High Bandwidth Memory)             │
├─────────────────────────────────────────┤
│ 크기:    큼 (80 GB)                     │
│ 속도:    느림 (2 TB/s)                  │
│ 위치:    Off-chip                       │
│ 용도:    주 메모리                      │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ SRAM (Static RAM)                       │
├─────────────────────────────────────────┤
│ 크기:    작음 (20 MB)                   │
│ 속도:    빠름 (19 TB/s)                 │
│ 위치:    On-chip                        │
│ 용도:    캐시                           │
└─────────────────────────────────────────┘

비율:
크기: HBM / SRAM ≈ 4,000배
속도: SRAM / HBM ≈ 10배
```

**시각화:**

```
┌──────────────────────────────────────┐
│          GPU Chip                    │
│  ┌────────────────────────────────┐ │
│  │ Compute Units (SM)             │ │
│  │  - CUDA cores                  │ │
│  │  - Tensor cores                │ │
│  │                                │ │
│  │  ┌──────────┐                  │ │
│  │  │  SRAM    │ ← 매우 빠름!     │ │
│  │  │  20 MB   │                  │ │
│  │  └──────────┘                  │ │
│  └────────────────────────────────┘ │
│             ↕ (빠른 연결)           │
└──────────────┬───────────────────────┘
               ↕ (느린 연결)
    ┌──────────────────────┐
    │       HBM            │
    │       80 GB          │
    └──────────────────────┘
```

**병목 현상:**

```
GPU는 매우 빠름:
- FP16: 1,979 TFLOPS

하지만:
HBM과의 통신이 느림!

Example:
행렬 곱셈 A × B:
- 실제 계산:    0.1 ms
- HBM에서 로드: 10 ms
- HBM에 저장:   10 ms
───────────────────────────
Total:          20.1 ms

→ 계산은 0.5%만!
→ 99.5%는 메모리 전송!

결론:
Memory-bound computation
```

## 6.2. 표준 Attention의 문제

**Attention 공식:**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

크기:
Q: (N, d)  # N = sequence length
K: (N, d)
V: (N, d)
```

**표준 구현:**

```python
def standard_attention(Q, K, V):
    # 1. Q @ K^T
    S = Q @ K.T / sqrt(d_k)  # (N, N)
    # → HBM에 저장

    # 2. Softmax
    P = softmax(S)  # (N, N)
    # → HBM에 저장

    # 3. P @ V
    O = P @ V  # (N, d)

    return O
```

**메모리 접근 패턴:**

```
Step 1: Q @ K^T
┌─────┐
│ HBM │ → Load Q, K
└─────┘
   ↓
[Compute S]
   ↓
┌─────┐
│ HBM │ ← Store S  # (N, N) 매트릭스!
└─────┘

Step 2: Softmax(S)
┌─────┐
│ HBM │ → Load S
└─────┘
   ↓
[Compute P]
   ↓
┌─────┐
│ HBM │ ← Store P  # (N, N) 매트릭스!
└─────┘

Step 3: P @ V
┌─────┐
│ HBM │ → Load P, V
└─────┘
   ↓
[Compute O]
   ↓
┌─────┐
│ HBM │ ← Store O
└─────┘

문제:
- (N, N) 매트릭스를 HBM에 2번 저장
- N=2048: 4M elements × 2 = 8M elements
- N=8192: 67M elements × 2 = 134M elements
- 매우 큰 메모리 전송!
```

**왜 (N, N) 매트릭스를 저장?**

```
Softmax 때문:

softmax(S)_ij = exp(S_ij) / Σ_k exp(S_ik)

Row-wise 정규화:
각 행이 합이 1이 되어야 함

→ 전체 행을 알아야 함
→ 한 번에 계산해야 한다고 생각
```

## 6.3. Flash Attention의 핵심 아이디어

**두 가지 핵심 트릭:**

### 1. Tiling (블록 단위 계산)

```
핵심:
Softmax를 전체에 대해 한 번에 계산할 필요 없음!
작은 블록으로 나누어 계산 가능!
```

**Softmax의 수학적 성질:**

```
S = [S1 | S2 | S3 | ... | Sn]

softmax([S1, S2]) = [softmax(S1) × α, softmax(S2) × β]

여기서 α, β는 재정규화 상수

즉:
부분 softmax를 계산 후 나중에 보정 가능!
```

**Flash Attention 알고리즘:**

```
1. S와 V를 작은 블록으로 분할
   S = [S1, S2, ..., Sb]
   V = [V1, V2, ..., Vb]

2. 각 블록을 SRAM에 로드

3. SRAM 내에서 전체 계산 수행

4. 결과를 HBM에 한 번만 저장
```

**시각화:**

```
표준 Attention:
Q, K → HBM
       ↓ (load)
    S = Q @ K^T
       ↓ (store to HBM)
       ↓ (load from HBM)
    P = softmax(S)
       ↓ (store to HBM)
       ↓ (load from HBM)
    O = P @ V
       ↓ (store)

Flash Attention:
Q_block, K_block, V_block → SRAM
       ↓
    [모든 계산 in SRAM]
    S_block = Q_block @ K_block^T
    P_block = softmax(S_block)
    O_block = P_block @ V_block
       ↓
    O_block → HBM (최종 결과만)

HBM 접근:
표준: 5번 (Q, K, S, P, V, O)
Flash: 3번 (Q_blocks, K_blocks, V_blocks, O)
```

## 6.4. Tiling과 Recomputation

**Tiling 전략:**

```python
def flash_attention_forward(Q, K, V):
    # Q, K, V: (batch, seq_len, d_model)

    N = Q.shape[1]  # sequence length
    block_size = 256  # SRAM에 fit되는 크기
    num_blocks = N // block_size

    # Output 초기화
    O = torch.zeros_like(Q)

    # 정규화를 위한 통계 저장
    l = torch.zeros(Q.shape[0], N)  # row sum
    m = torch.full((Q.shape[0], N), -float('inf'))  # row max

    # Q를 블록으로 순회
    for i in range(num_blocks):
        # Q 블록 로드 (HBM → SRAM)
        Q_block = Q[:, i*block_size:(i+1)*block_size, :]

        # K, V를 블록으로 순회
        for j in range(num_blocks):
            # K, V 블록 로드 (HBM → SRAM)
            K_block = K[:, j*block_size:(j+1)*block_size, :]
            V_block = V[:, j*block_size:(j+1)*block_size, :]

            # SRAM 내에서 계산
            S_block = Q_block @ K_block.T / sqrt(d_k)

            # Online softmax (재정규화)
            m_new = torch.maximum(m[:, i*block_size:(i+1)*block_size],
                                  S_block.max(dim=-1, keepdim=True)[0])

            P_block = torch.exp(S_block - m_new)
            l_new = torch.exp(m_old - m_new) * l_old + P_block.sum(dim=-1, keepdim=True)

            # Output 업데이트
            O_block = P_block @ V_block
            O[:, i*block_size:(i+1)*block_size, :] = (
                O_old * (l_old / l_new) + O_block / l_new
            )

            # 통계 업데이트
            l[:, i*block_size:(i+1)*block_size] = l_new
            m[:, i*block_size:(i+1)*block_size] = m_new

    return O
```

### 2. Recomputation (재계산)

```
문제:
Backward pass에는 activation 필요
→ Forward에서 저장해야 함
→ 메모리 많이 사용

Flash Attention 해결책:
Forward에서 activation 저장 안함!
→ Backward에서 다시 계산
→ 메모리 절약
```

**표준 vs Flash Attention:**

```
표준 Attention:

Forward:
저장: S (N×N), P (N×N), O (N×d)
메모리: O(N²)

Backward:
로드: S, P
계산: gradients
메모리: O(N²)

────────────────────────────

Flash Attention:

Forward:
저장: O (N×d), l, m (통계만)
메모리: O(N) ← N² 아님!

Backward:
저장 안된 activation 재계산:
- Q, K, V 블록 다시 로드
- S, P 재계산 (SRAM에서)
- Gradient 계산

메모리: O(N)
연산: 더 많음 (재계산)
하지만: SRAM이 빠르므로 괜찮음!
```

## 6.5. Flash Attention의 효과

**성능 비교:**

```
┌────────────────┬──────────┬───────────┬─────────┐
│ Metric         │ Standard │ Flash     │ Speedup │
├────────────────┼──────────┼───────────┼─────────┤
│ HBM R/W        │ 40.3 GB  │ 4.0 GB    │ 10.1×   │
│ Runtime        │ 41.7 ms  │ 7.6 ms    │ 5.5×    │
│ Memory         │ 4.2 GB   │ 0.9 GB    │ 4.7×    │
│ GigaFLOPs      │ 65       │ 85        │ 0.77×   │
└────────────────┴──────────┴───────────┴─────────┘

놀라운 점:
- 더 많은 연산 (85 vs 65 GFLOPS)
- 하지만 더 빠름! (7.6ms vs 41.7ms)
- 메모리도 절약 (0.9GB vs 4.2GB)

이유:
Memory bandwidth가 병목
→ 메모리 전송을 줄이면 전체 성능 향상!
```

**Sequence Length에 따른 효과:**

```
Sequence Length = 512:
- Standard:  5 ms
- Flash:     3 ms
- Speedup:   1.7×

Sequence Length = 2048:
- Standard:  41 ms
- Flash:     8 ms
- Speedup:   5.1×

Sequence Length = 8192:
- Standard:  650 ms
- Flash:     120 ms
- Speedup:   5.4×

→ 긴 sequence일수록 더 효과적!
```

**Flash Attention 2, 3:**

```
Flash Attention 2 (2023):
- 더 나은 GPU 활용
- Parallelism 개선
- 2× faster than Flash Attention 1

Flash Attention 3 (2024):
- H100 GPU에 최적화
- Asynchronous 연산
- 추가 1.5-2× speedup

Evolution:
Standard → Flash 1 → Flash 2 → Flash 3
  100%      500%      1000%     2000%
```

**실전 활용:**

```python
# PyTorch 2.0+ built-in
import torch.nn.functional as F

output = F.scaled_dot_product_attention(
    query, key, value,
    is_causal=True,  # Autoregressive mask
    # 자동으로 Flash Attention 사용!
)

# Hugging Face Transformers
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"  # Flash Attention 2 사용
)
```

---

# 7. Quantization (양자화)

## 7.1. Quantization이란?

**핵심 질문:**

```
부동소수점 수의 정밀도를 줄여도
성능이 유지될까?

Example:
π = 3.141592653589793...

정밀도 줄이기:
π ≈ 3.14      (2 decimal places)
π ≈ 3.1       (1 decimal place)
π ≈ 3         (integer)

Trade-off:
정밀도 ↓ → 메모리 ↓, 속도 ↑
```

**Quantization의 정의:**

```
Quantization = 숫자의 표현 정밀도를 낮추는 과정

목표:
1. 메모리 사용량 감소
2. 계산 속도 향상
3. 성능 유지 (가장 중요!)
```

## 7.2. Floating Point 표현

**부동소수점 구조:**

```
Floating Point Number = Sign × Mantissa × 2^Exponent

Binary 표현:
[S][Exponent][Mantissa]
 ↑      ↑        ↑
 1bit  Ebits   Mbits

Example:
-12.75 = -1 × 1.59375 × 2^3

Binary:
Sign:     1 (negative)
Exponent: 3 (저장: 3 + 127 = 130)
Mantissa: 1.59375 = 1 + 0.5 + 0.09375
```

**일반적인 형식:**

```
┌──────────┬──────┬─────┬──────────┬──────────┐
│ Format   │ Total│ Sign│ Exponent │ Mantissa │
├──────────┼──────┼─────┼──────────┼──────────┤
│ FP64     │ 64   │ 1   │ 11       │ 52       │
│ FP32     │ 32   │ 1   │ 8        │ 23       │
│ FP16     │ 16   │ 1   │ 5        │ 10       │
│ BF16     │ 16   │ 1   │ 8        │ 7        │
│ FP8      │ 8    │ 1   │ 4        │ 3        │
│ INT8     │ 8    │ 1   │ 0        │ 7        │
└──────────┴──────┴─────┴──────────┴──────────┘
```

**FP32 vs FP16 비교:**

```
FP32 (Single Precision):
┌─┬────────┬───────────────────────┐
│S│EEEEEEEE│MMMMMMMMMMMMMMMMMMMMMMM│
└─┴────────┴───────────────────────┘
 1    8              23

Range:  ±1.4 × 10^-45 to ±3.4 × 10^38
Precision: ~7 decimal digits

FP16 (Half Precision):
┌─┬─────┬──────────┐
│S│EEEEE│MMMMMMMMMM│
└─┴─────┴──────────┘
 1   5       10

Range:  ±6 × 10^-8 to ±65,504
Precision: ~3 decimal digits

메모리: FP16 = FP32 / 2
```

**BF16 (Brain Float 16):**

```
BF16 = Google이 개발한 ML 특화 형식

FP16:
┌─┬─────┬──────────┐
│S│EEEEE│MMMMMMMMMM│
└─┴─────┴──────────┘
 1   5       10

BF16:
┌─┬────────┬───────┐
│S│EEEEEEEE│MMMMMMM│
└─┴────────┴───────┘
 1    8        7

특징:
- FP32와 동일한 exponent range
- Mantissa는 짧음
- FP32 ↔ BF16 변환 쉬움
- ML에서 더 안정적

이유:
Exponent가 중요 (범위)
Mantissa는 덜 중요 (정밀도)
→ Gradient 계산에 유리
```

**GPU Compute Speed:**

```
NVIDIA H100 FLOPS:

┌──────────┬───────────────┐
│ Precision│ TFLOPS        │
├──────────┼───────────────┤
│ FP64     │ 34            │
│ FP32     │ 60            │
│ TF32     │ 989           │
│ FP16     │ 1,979         │
│ BF16     │ 1,979         │
│ INT8     │ 3,958         │
└──────────┴───────────────┘

관찰:
정밀도 낮을수록 빠름!
FP16 = FP32의 33배 빠름!
```

## 7.3. Mixed Precision Training

**핵심 아이디어:**

```
서로 다른 정밀도를 섞어 사용

높은 정밀도: 중요한 것
낮은 정밀도: 덜 중요한 것

→ Best of both worlds!
```

**Mixed Precision Training 전략:**

```
┌────────────────────────────────────┐
│ Master Weights (FP32)              │
│ - 모델 가중치의 원본               │
│ - 높은 정밀도 유지                 │
│ - 메모리: 4 bytes × N              │
└────────────┬───────────────────────┘
             ↓ (Copy & Cast)
┌────────────────────────────────────┐
│ Working Copy (FP16)                │
│ - Forward/Backward에 사용          │
│ - 빠른 계산                        │
│ - 메모리: 2 bytes × N              │
└────────────┬───────────────────────┘
             ↓
┌────────────────────────────────────┐
│ Gradients (FP16)                   │
│ - Backward pass에서 계산           │
│ - 빠른 계산                        │
└────────────┬───────────────────────┘
             ↓ (Cast to FP32)
┌────────────────────────────────────┐
│ Weight Update (FP32)               │
│ - Master weights 업데이트          │
│ - 정밀한 누적                      │
└────────────────────────────────────┘
```

**구현 예시:**

```python
import torch
from torch.cuda.amp import autocast, GradScaler

model = MyLLM().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Gradient Scaler (underflow 방지)
scaler = GradScaler()

for batch in dataloader:
    inputs, targets = batch

    # Forward pass (FP16)
    with autocast():  # 자동으로 FP16 사용
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    # Backward pass
    scaler.scale(loss).backward()  # Loss scaling

    # Optimizer step (FP32)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

**Loss Scaling:**

```
문제:
FP16의 dynamic range가 작음
→ 작은 gradient가 0으로 underflow

해결:
Gradient Scaling

1. Loss에 큰 수 곱하기
   loss_scaled = loss × scale_factor

2. Backward (큰 gradient 계산)
   grad_scaled = ∂loss_scaled/∂w

3. Unscale before update
   grad = grad_scaled / scale_factor

4. Weight update (FP32)
   w_new = w - lr × grad

Example:
gradient = 1e-7 (FP16으로 0)
scale = 1024
gradient_scaled = 1e-7 × 1024 = 1e-4 (FP16 OK)
```

**왜 이게 작동하는가?**

```
Forward/Backward:
- Noisy한 계산
- 개별 activation 정밀도 덜 중요
- 방향만 대략 맞으면 OK
- FP16으로 충분!

Weight Update:
- 정밀한 누적 필요
- 작은 변화가 누적됨
- Precision 중요!
- FP32 필요!

비유:
Forward/Backward = 방향 찾기 (나침반)
  → 대략적으로 맞으면 됨

Weight Update = 정확한 이동 (GPS)
  → 정밀해야 함
```

**이점:**

```
1. 메모리 절감
   Activations (FP16):
   - 가장 큰 메모리 소비
   - 2배 절감!

2. 속도 향상
   Forward/Backward (FP16):
   - 대부분의 계산 시간
   - 2-3배 빠름!

3. 성능 유지
   Master Weights (FP32):
   - 정밀한 업데이트
   - 성능 거의 동일!

실제 결과:
- 메모리: ~40% 감소
- 속도: 2-3× 빠름
- 정확도: 0.1-0.2% 차이 (거의 없음)
```

**Mixed Precision 체크리스트:**

```python
# 1. 모델을 GPU로
model = model.cuda()

# 2. autocast 사용
from torch.cuda.amp import autocast

with autocast():
    output = model(input)
    loss = criterion(output, target)

# 3. GradScaler 사용
from torch.cuda.amp import GradScaler
scaler = GradScaler()

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 4. BF16 옵션 (A100+)
with autocast(dtype=torch.bfloat16):
    ...
```

---

# 8. Supervised Fine-tuning (SFT)

## 8.1. Pretraining만으로는 부족한 이유

**문제 상황:**

```
Pretraining 후 모델:
- 언어 구조 이해 ✓
- 다음 토큰 예측 잘함 ✓
- 하지만...

Example:
User: "Can I put my teddy bear in the washer?"

Pretrained Model:
"Most teddy bears are made of polyester or cotton.
 These materials typically require..."

→ 정보는 맞지만 도움이 안됨!
→ 질문에 직접 답하지 않음!
```

**왜 이런 일이?**

```
Pretraining objective:
"다음 단어를 예측하라"

학습한 것:
- 언어 패턴
- 통계적 연관성
- 일반적 지식

학습하지 못한 것:
- 도움이 되는 방법
- 질문에 답하는 방법
- Assistant 역할

비유:
책을 많이 읽음 (Pretraining)
→ 언어는 유창
→ 하지만 대화 방법은 모름
```

**원하는 행동:**

```
User: "Can I put my teddy bear in the washer?"

Ideal Assistant:
"Check the label first. If it says machine washable:
- Use cold water
- Gentle cycle
- Place in a pillowcase
Otherwise, hand wash is recommended."

→ 직접적으로 유용함!
→ 실용적 조언!
→ 친절한 톤!
```

## 8.2. SFT란?

**Supervised Fine-tuning (SFT):**

```
정의:
고품질 (Input, Output) 쌍으로 모델을 fine-tuning

목표:
Pretrained model을 유용한 assistant로 변환

Input  → Model → Output
(지시)         (응답)
```

**Training 형식:**

```
Pretrained Model:
"The cat sat on" → "the mat"
(Next token prediction)

SFT Model:
"Question: How to wash a teddy bear?
 Answer: Check the label first..."
(Instruction → Response)
```

## 8.3. Instruction Data

**데이터 형식:**

```json
{
  "instruction": "Can I put my teddy bear in the washer?",
  "response": "Check the care label on your teddy bear first. If it's machine washable: Use cold water on a gentle cycle, place the bear in a pillowcase or laundry bag for protection. If not machine washable, hand washing is recommended with mild detergent."
}
```

**데이터 출처:**

```
1. Human-written
   - 사람이 직접 작성
   - 높은 품질
   - 비쌈
   - 예: 수천~수만 examples

2. Model-generated
   - 다른 LLM이 생성
   - 대규모 생성 가능
   - 품질 검증 필요
   - 예: GPT-4로 생성

3. Existing datasets
   - Stack Overflow
   - Reddit conversations
   - Q&A websites
   - 형식 변환 필요
```

**데이터 예시:**

```
┌────────────────────────────────────────────┐
│ Example 1: Code Generation                 │
├────────────────────────────────────────────┤
│ Instruction:                               │
│ "Write a Python function to reverse a      │
│  string"                                   │
│                                            │
│ Response:                                  │
│ "def reverse_string(s):                    │
│      return s[::-1]"                       │
└────────────────────────────────────────────┘

┌────────────────────────────────────────────┐
│ Example 2: Question Answering              │
├────────────────────────────────────────────┤
│ Instruction:                               │
│ "What is the capital of France?"           │
│                                            │
│ Response:                                  │
│ "The capital of France is Paris."          │
└────────────────────────────────────────────┘

┌────────────────────────────────────────────┐
│ Example 3: Creative Writing                │
├────────────────────────────────────────────┤
│ Instruction:                               │
│ "Write a haiku about autumn"               │
│                                            │
│ Response:                                  │
│ "Leaves gently falling                     │
│  Colors paint the forest floor             │
│  Nature's quiet song"                      │
└────────────────────────────────────────────┘
```

**데이터 규모:**

```
Order of Magnitude:
- Minimum: 1K examples
- Typical: 10K-100K examples
- Large: 1M+ examples

비교:
Pretraining: 1T-15T tokens
SFT:         10K-1M examples

→ SFT 데이터가 훨씬 적음!
→ 품질이 더 중요!
```

## 8.4. SFT 구현

**Training Objective:**

```
Pretraining:
Loss = CrossEntropy(predicted_next_token, actual_next_token)

SFT:
Loss = CrossEntropy(predicted_response, target_response)

차이:
- 입력 형식이 instruction
- 목표가 helpful response
```

**구현 예시:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

# 1. Pretrained model 로드
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. Dataset 준비
class InstructionDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Instruction + Response 형식
        text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}"

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Labels = input_ids (autoregressive)
        # Instruction 부분은 loss 계산 안함
        labels = encoding["input_ids"].clone()

        # Instruction 부분 mask
        instruction_length = len(self.tokenizer.encode(
            f"### Instruction:\n{item['instruction']}\n\n### Response:\n"
        ))
        labels[:, :instruction_length] = -100  # Ignore in loss

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }

# 3. Dataset 생성
train_dataset = InstructionDataset(train_data, tokenizer)

# 4. Training arguments
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./sft-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    fp16=True,  # Mixed precision
    logging_steps=100,
    save_steps=1000,
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# 6. Train
trainer.train()
```

**Parameter-Efficient Fine-Tuning (PEFT):**

```
문제:
전체 모델 fine-tuning은 비쌈
- 175B model
- 700 GB memory
- 수천 GPU hours

해결:
일부만 tuning!

방법:
1. LoRA (Low-Rank Adaptation)
2. Prefix Tuning
3. Adapter Layers
```

**LoRA 예시:**

```python
from peft import LoraConfig, get_peft_model

# LoRA config
lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# LoRA 적용
model = get_peft_model(model, lora_config)

# Trainable parameters
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%

# → 0.06%만 학습!
# → 메모리/시간 대폭 절감!
```

**SFT의 효과:**

```
Before SFT (Pretrained only):
User: "How do I sort a list in Python?"
Model: "How do I sort a list in Python?
        This is a common question that..."
→ Continues the text, not answering

After SFT:
User: "How do I sort a list in Python?"
Model: "You can sort a list in Python using:
        1. list.sort() - sorts in place
        2. sorted(list) - returns new sorted list

        Example:
        numbers = [3, 1, 4, 1, 5]
        numbers.sort()
        print(numbers)  # [1, 1, 3, 4, 5]"
→ Directly helpful!
```

---

# 9. 요약

## LLM Training Pipeline

**전체 파이프라인:**

```
┌──────────────────────────────────────────────┐
│ Stage 1: Pretraining                         │
├──────────────────────────────────────────────┤
│ 목표: 언어 구조 학습                          │
│ 데이터: 인터넷의 모든 텍스트 (수조 tokens)     │
│ 학습: Next Token Prediction                  │
│ 비용: $1M-$100M+                             │
│ 시간: 수주-수개월                             │
│                                              │
│ 결과: Base Model                             │
│ - 언어 이해 ✓                                │
│ - 코드 이해 ✓                                │
│ - 하지만 유용한 assistant 아님 ✗             │
└────────────────┬─────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────┐
│ Stage 2: Supervised Fine-tuning (SFT)        │
├──────────────────────────────────────────────┤
│ 목표: 유용한 assistant로 변환                 │
│ 데이터: Instruction-Response 쌍 (수만)        │
│ 학습: Instruction Following                  │
│ 비용: $1K-$100K                              │
│ 시간: 수시간-수일                             │
│                                              │
│ 결과: Instruction-following Model            │
│ - 질문에 답함 ✓                              │
│ - 유용함 ✓                                   │
│ - 하지만 완벽하지 않음                        │
└────────────────┬─────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────┐
│ Stage 3: Preference Tuning (다음 강의!)       │
├──────────────────────────────────────────────┤
│ 목표: 인간 선호도에 맞게 정렬                  │
│ 방법: RLHF, DPO                              │
│                                              │
│ 결과: Aligned Model                          │
│ - Helpful, Honest, Harmless ✓                │
└──────────────────────────────────────────────┘
```

## Training Optimizations 요약

**메모리 최적화:**

```
┌───────────────────┬──────────────┬──────────────┐
│ 기법              │ 절감         │ Trade-off    │
├───────────────────┼──────────────┼──────────────┤
│ Data Parallelism  │ Activation   │ 통신 증가    │
│ ZeRO-1            │ 4× (Opt)     │ 약간 느림    │
│ ZeRO-2            │ 8× (Opt+Grad)│ 더 느림      │
│ ZeRO-3            │ 12× (모두)   │ 많이 느림    │
│ Flash Attention   │ 4-5×         │ 없음!        │
│ Mixed Precision   │ 2× (Act)     │ 거의 없음    │
│ Gradient Ckpt     │ O(√N)        │ 재계산       │
└───────────────────┴──────────────┴──────────────┘
```

**계산 최적화:**

```
┌─────────────────────┬──────────────┬──────────────┐
│ 기법                │ 속도 향상    │ 적용         │
├─────────────────────┼──────────────┼──────────────┤
│ Mixed Precision     │ 2-3×         │ Training     │
│ Flash Attention     │ 2-5×         │ Training/Inf │
│ Quantization (INT8) │ 2-4×         │ Inference    │
│ Model Parallelism   │ Linear       │ 큰 모델      │
│ Compilation         │ 1.5-2×       │ 모두         │
└─────────────────────┴──────────────┴──────────────┘
```

## 실전 체크리스트

**Small Model (<7B):**

```
□ Data Parallelism (DDP)
□ Mixed Precision (FP16/BF16)
□ Flash Attention
□ Gradient Accumulation
□ Learning rate scheduling
```

**Medium Model (7B-70B):**

```
□ ZeRO-2
□ Mixed Precision
□ Flash Attention
□ Gradient Checkpointing
□ LoRA for fine-tuning
```

**Large Model (>70B):**

```
□ ZeRO-3
□ Model Parallelism
□ Mixed Precision
□ Flash Attention
□ Gradient Checkpointing
□ Pipeline Parallelism
```

**Scaling Laws 요약:**

```
Chinchilla Optimal:
Data : Model = 20 : 1

Example:
70B model → 1.4T tokens optimal

실전에서는:
Often overtrain (data 더 많이)
이유: Inference efficiency
```

---

# 10. 중요 용어 정리

## Training 관련

**Pretraining (사전학습)**
```
대규모 데이터로 언어의 일반적 구조를 학습하는 단계
Next token prediction 목표
비용이 가장 많이 듦
```

**Fine-tuning (미세조정)**
```
Pretrained model을 특정 작업에 맞게 조정
SFT, RLHF, DPO 등 다양한 방법
상대적으로 저렴
```

**Transfer Learning (전이학습)**
```
한 작업에서 학습한 지식을 다른 작업에 활용
LLM의 기본 paradigm
Pretraining → Fine-tuning
```

**Supervised Fine-tuning (SFT)**
```
Instruction-Response 쌍으로 모델 학습
Assistant 행동 학습
10K-100K examples 필요
```

**Knowledge Cutoff**
```
모델이 학습한 데이터의 마지막 날짜
이후 지식은 모델이 모름
업데이트 필요
```

## Optimization 관련

**Data Parallelism**
```
배치를 여러 GPU에 분산
각 GPU가 모델 복사본 가짐
Gradient를 평균내어 업데이트
```

**Model Parallelism**
```
모델을 여러 GPU에 분산
Tensor, Pipeline, Expert parallelism
큰 모델에 필수
```

**ZeRO (Zero Redundancy Optimizer)**
```
중복 제거로 메모리 절약
Stage 1: Optimizer state
Stage 2: + Gradients
Stage 3: + Parameters
```

**Flash Attention**
```
Memory-efficient attention
Tiling + Recomputation
HBM 접근 최소화
2-5배 빠름
```

**Quantization (양자화)**
```
숫자 정밀도 감소
FP32 → FP16/BF16/INT8
메모리/속도 향상
```

**Mixed Precision Training**
```
서로 다른 정밀도 혼합 사용
Forward/Backward: FP16
Weight update: FP32
2-3배 속도 향상
```

## Hardware 관련

**FLOPs (Floating Point Operations)**
```
계산량의 단위
연산 개수
LLM training: 10^24-10^26
```

**FLOPS (Floating Point Operations Per Second)**
```
계산 속도의 단위
초당 연산 수
GPU 성능 지표
```

**HBM (High Bandwidth Memory)**
```
GPU의 주 메모리
큼 (80 GB)
느림 (2 TB/s)
```

**SRAM (Static RAM)**
```
GPU의 on-chip 메모리
작음 (20 MB)
빠름 (19 TB/s)
```

**GPU Memory Bandwidth**
```
메모리 전송 속도
병목의 주 원인
Flash Attention이 해결
```

## Fine-tuning 관련

**LoRA (Low-Rank Adaptation)**
```
Parameter-efficient fine-tuning
일부 parameter만 학습
0.1% parameters로 충분
```

**Instruction Data**
```
(Instruction, Response) 쌍
SFT 학습 데이터
품질이 매우 중요
```

**Gradient Checkpointing**
```
Activation 재계산
메모리 vs 속도 trade-off
긴 sequence에 유용
```

---

**다음 강의 예고:**

Lecture 5에서는 Preference Tuning과 RLHF를 다룹니다:
- RLHF (Reinforcement Learning from Human Feedback)
- Reward Model training
- PPO algorithm
- DPO (Direct Preference Optimization)
- Alignment techniques

---

**수고하셨습니다!** 🎉
