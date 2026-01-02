# Lecture 2: Transformer-based models & tricks

# Materials

- [CME 295](https://cme295.stanford.edu/syllabus/)
- [slide](https://cme295.stanford.edu/slides/fall25-cme295-lecture2.pdf)
- [video](https://www.youtube.com/watch?v=yT84Y5zCnaA&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=2)

# Table of Contents

- [Lecture 2: Transformer-based models \& tricks](#lecture-2-transformer-based-models--tricks)
- [Materials](#materials)
- [Table of Contents](#table-of-contents)
- [강의 개요](#강의-개요)
  - [강의 목표](#강의-목표)
  - [주요 학습 내용](#주요-학습-내용)
- [1. Transformer 복습](#1-transformer-복습)
  - [1.1. Self-Attention 메커니즘](#11-self-attention-메커니즘)
  - [1.2. Multi-Head Attention](#12-multi-head-attention)
- [2. Position Embeddings](#2-position-embeddings)
  - [2.1. 왜 Position Embedding이 필요한가?](#21-왜-position-embedding이-필요한가)
  - [2.2. Learned Position Embeddings](#22-learned-position-embeddings)
  - [2.3. Sinusoidal Position Embeddings](#23-sinusoidal-position-embeddings)
  - [2.4. Modern Approaches: T5 Bias \& ALiBi](#24-modern-approaches-t5-bias--alibi)
    - [T5 Relative Position Bias](#t5-relative-position-bias)
    - [ALiBi (Attention with Linear Biases)](#alibi-attention-with-linear-biases)
  - [2.5. RoPE (Rotary Position Embeddings)](#25-rope-rotary-position-embeddings)
  - [2.6. Position Embedding 비교](#26-position-embedding-비교)
- [3. Layer Normalization](#3-layer-normalization)
  - [3.1. LayerNorm이란?](#31-layernorm이란)
  - [3.2. Post-norm vs Pre-norm](#32-post-norm-vs-pre-norm)
    - [Post-norm (원본 Transformer)](#post-norm-원본-transformer)
    - [Pre-norm (현대 표준)](#pre-norm-현대-표준)
  - [3.3. RMSNorm](#33-rmsnorm)
- [4. Attention Variations](#4-attention-variations)
  - [4.1. Sparse Attention](#41-sparse-attention)
    - [Sliding Window Attention](#sliding-window-attention)
    - [Global + Local Attention](#global--local-attention)
  - [4.2. Grouped Query Attention (GQA)](#42-grouped-query-attention-gqa)
    - [Multi-Head Attention (MHA) - 원본](#multi-head-attention-mha---원본)
    - [Multi-Query Attention (MQA) - 극단적 공유](#multi-query-attention-mqa---극단적-공유)
    - [Grouped Query Attention (GQA) - 중간 접근](#grouped-query-attention-gqa---중간-접근)
- [5. Transformer Model Families](#5-transformer-model-families)
  - [5.1. Encoder-Decoder Models](#51-encoder-decoder-models)
    - [T5 (Text-to-Text Transfer Transformer)](#t5-text-to-text-transfer-transformer)
  - [5.2. Encoder-only Models](#52-encoder-only-models)
    - [BERT (2018)](#bert-2018)
  - [5.3. Decoder-only Models](#53-decoder-only-models)
    - [GPT Family](#gpt-family)
  - [5.4. 모델 선택 가이드](#54-모델-선택-가이드)
- [6. 요약](#6-요약)
  - [핵심 개선사항](#핵심-개선사항)
    - [1. Position Embeddings](#1-position-embeddings)
    - [2. Layer Normalization](#2-layer-normalization)
    - [3. Attention 최적화](#3-attention-최적화)
    - [4. Model Families](#4-model-families)
  - [현대 LLM 구성](#현대-llm-구성)
  - [실전 구현 체크리스트](#실전-구현-체크리스트)
  - [주요 Trade-offs](#주요-trade-offs)
    - [Position Embeddings](#position-embeddings)
    - [Attention](#attention)
    - [Model Architecture](#model-architecture)
- [7. 중요 용어 정리](#7-중요-용어-정리)
  - [Position Embedding 관련](#position-embedding-관련)
  - [Normalization 관련](#normalization-관련)
  - [Attention 관련](#attention-관련)
  - [Model Architecture 관련](#model-architecture-관련)
  - [Pre-training 관련](#pre-training-관련)
  - [최적화 관련](#최적화-관련)
  - [기타](#기타)

---

# 강의 개요

## 강의 목표

이번 강의에서는 Transformer 아키텍처를 기반으로 한 다양한 개선 기법들과 모델 변형들을 학습합니다.

**학습 목표:**
- Transformer의 핵심 메커니즘 복습
- 다양한 Position Embedding 방법 이해
- Layer Normalization 기법 비교
- Attention 메커니즘의 최적화 기법
- Transformer 기반 모델 families 이해

## 주요 학습 내용

**1. Position Embeddings (핵심)**
- Learned Position Embeddings
- Sinusoidal Position Embeddings
- RoPE (Rotary Position Embeddings) - 현대 LLM의 표준
- ALiBi, T5 Relative Position Bias

**2. Layer Normalization**
- LayerNorm, Pre-norm, Post-norm
- RMSNorm - 더 빠르고 효율적인 정규화

**3. Attention 최적화**
- Sparse Attention (Sliding Window)
- GQA/MQA - KV Cache 최적화

**4. Model Families**
- Encoder-Decoder (T5)
- Encoder-only (BERT)
- Decoder-only (GPT) - 현대 LLM의 주류

---

# 1. Transformer 복습

이번 섹션에서는 Lecture 1에서 학습한 Transformer의 핵심 메커니즘을 간단히 복습합니다.

## 1.1. Self-Attention 메커니즘

**Self-Attention의 기본 수식:**

```
Q = X · W_Q  # Query
K = X · W_K  # Key
V = X · W_V  # Value

Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V
```

**왜 sqrt(d_k)로 나누나요?**
- d_k가 크면 내적 값이 너무 커져서 softmax의 gradient가 작아집니다
- sqrt(d_k)로 나눠서 적절한 scale을 유지합니다
- 예: d_k=64라면 sqrt(64)=8로 나눕니다

**구체적인 예시:**

시퀀스: "I love transformers"

```
토큰:     [I,    love,      transformers]
입력:     [x1,   x2,        x3]           (각 벡터: d_model=512)

Query:    [q1,   q2,        q3]           (W_Q로 변환)
Key:      [k1,   k2,        k3]           (W_K로 변환)
Value:    [v1,   v2,        v3]           (W_V로 변환)

Attention Scores (q2·k 계산):
q2·k1 = 0.3
q2·k2 = 0.8  (자기 자신)
q2·k3 = 0.5

Softmax → [0.2, 0.5, 0.3]

출력:
y2 = 0.2*v1 + 0.5*v2 + 0.3*v3
```

**핵심 특징:**
- 각 토큰이 모든 토큰과 상호작용
- 병렬 처리 가능
- O(n²) 복잡도

## 1.2. Multi-Head Attention

**왜 여러 개의 head를 사용하나요?**

하나의 attention은 하나의 관계만 학습합니다. Multi-head를 사용하면:
- 다양한 종류의 관계를 동시에 학습
- 예: 문법 관계, 의미 관계, 위치 관계 등

**구조:**

```
입력: X (batch_size, seq_len, d_model)

Head 1: Attention(Q1, K1, V1) → output1
Head 2: Attention(Q2, K2, V2) → output2
...
Head h: Attention(Qh, Kh, Vh) → outputh

Concat: [output1, output2, ..., outputh]
최종 출력: Concat · W_O
```

**예시 (GPT-3 기준):**
- d_model = 12,288
- num_heads = 96
- d_k = d_v = d_model / num_heads = 128
- 각 head는 128차원에서 attention 수행

**Attention Map 시각화:**

```
       I    love  trans formers
I     [0.9  0.05  0.03  0.02]
love  [0.2  0.5   0.2   0.1]
trans [0.1  0.2   0.5   0.2]
formers [0.05 0.1  0.3  0.55]
```

- 대각선이 밝음 (자기 자신에 높은 attention)
- "transformers"는 "trans"에도 높은 attention

---

# 2. Position Embeddings

Position Embedding은 Transformer에서 가장 중요한 구성 요소 중 하나입니다. 이번 섹션에서는 다양한 Position Embedding 방법을 비교하고, 현대 LLM에서 사용되는 기법들을 학습합니다.

## 2.1. 왜 Position Embedding이 필요한가?

**Self-Attention의 문제:**

Self-Attention은 위치 정보를 전혀 고려하지 않습니다!

```python
# 두 문장이 같은 attention score를 가짐
"I love transformers"
"transformers love I"

# Self-Attention은 토큰의 순서를 모름
```

**수학적으로 보면:**

```
Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V
```

이 수식 어디에도 위치 정보가 없습니다!

**왜 문제인가요?**

자연어는 순서가 중요합니다:
- "Dog bites man" ≠ "Man bites dog"
- "Not good" ≠ "Good not"

**해결책:**

입력에 위치 정보를 추가합니다:

```
X_final = Token_Embedding + Position_Embedding
```

**다양한 방법들:**
1. Learned Position Embeddings (학습)
2. Sinusoidal Position Embeddings (고정 공식)
3. T5 Relative Position Bias (학습)
4. ALiBi (고정 공식)
5. RoPE (회전 변환) - 현대 표준

## 2.2. Learned Position Embeddings

**가장 간단한 방법: 위치마다 임베딩 벡터를 학습**

**구현:**

```python
class LearnedPositionEmbedding(nn.Module):
    def __init__(self, max_position, d_model):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_position, d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)

        # Position embedding 추가
        pos_emb = self.position_embeddings(positions)
        return x + pos_emb
```

**예시:**

```
max_position = 512
d_model = 768

Position Embedding Table:
위치 0: [0.12, -0.34, 0.56, ..., 0.78]  (768-dim)
위치 1: [-0.23, 0.45, -0.67, ..., 0.12]
위치 2: [0.34, 0.12, -0.89, ..., -0.45]
...
위치 511: [0.67, -0.23, 0.45, ..., 0.89]
```

**장점:**
- 구현이 매우 간단
- 데이터로부터 최적의 위치 표현을 학습
- 특정 위치에 특별한 패턴 학습 가능

**단점:**
- 학습 데이터보다 긴 시퀀스를 처리할 수 없음
  - 예: max_position=512로 학습했다면, 513 토큰은 처리 불가
- 과적합 위험
  - 특정 위치에 과도하게 맞춰질 수 있음
- 파라미터 추가
  - max_position × d_model 개의 파라미터

**사용 모델:**
- 초기 BERT
- GPT-1
- 일부 초기 Transformer 모델

## 2.3. Sinusoidal Position Embeddings

**원본 Transformer 논문의 방법: 고정된 수학 공식 사용**

**수식:**

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

여기서:
- pos: 위치 (0, 1, 2, ...)
- i: 차원 인덱스 (0, 1, 2, ..., d_model/2-1)
- 짝수 차원: sine 함수
- 홀수 차원: cosine 함수
```

**왜 이런 공식을 사용하나요?**

1. **다양한 주파수 사용**
   - 낮은 차원 (i=0): 빠른 주파수 → 가까운 위치 구분
   - 높은 차원 (i=d_model/2-1): 느린 주파수 → 먼 위치 구분

2. **상대적 위치 표현 가능**
   - 삼각함수 항등식 활용
   - PE(pos+k)를 PE(pos)의 선형 변환으로 표현 가능

**구체적인 예시:**

```python
# d_model = 512, pos = 0, 1, 2, ...

# i = 0 (차원 0, 1):
PE(0, 0) = sin(0 / 10000^0) = sin(0) = 0
PE(0, 1) = cos(0 / 10000^0) = cos(0) = 1
PE(1, 0) = sin(1 / 1) = sin(1) ≈ 0.841
PE(1, 1) = cos(1 / 1) = cos(1) ≈ 0.540

# i = 1 (차원 2, 3):
PE(0, 2) = sin(0 / 10000^(2/512)) = sin(0) = 0
PE(0, 3) = cos(0 / 10000^(2/512)) = cos(0) = 1
PE(1, 2) = sin(1 / 10000^(2/512)) ≈ 0.0010
PE(1, 3) = cos(1 / 10000^(2/512)) ≈ 0.9999

# i = 255 (차원 510, 511):
# 주파수가 매우 낮음 → 변화가 느림
```

**시각화:**

```
Position  Dim0    Dim1    Dim2    Dim3    ...  Dim510  Dim511
0         0.000   1.000   0.000   1.000   ...  0.000   1.000
1         0.841   0.540   0.001   1.000   ...  0.000   1.000
2         0.909  -0.416   0.002   1.000   ...  0.000   1.000
3         0.141  -0.990   0.003   1.000   ...  0.000   1.000
...
```

**삼각함수 항등식:**

```
sin(α + β) = sin(α)cos(β) + cos(α)sin(β)
cos(α + β) = cos(α)cos(β) - sin(α)sin(β)

이를 이용하면:
PE(pos + k) = Linear_Function(PE(pos), k)
```

**왜 중요한가요?**

모델이 상대적 위치를 학습할 수 있습니다:
- "k 토큰 떨어진 관계"를 인식
- 예: "주어와 3단어 떨어진 동사"

**구현:**

```python
def sinusoidal_position_embedding(seq_len, d_model):
    # 위치와 차원 인덱스 생성
    position = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 차원
    pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 차원

    return pe  # (seq_len, d_model)
```

**장점:**
- 학습이 필요 없음 (파라미터 0개)
- 임의 길이의 시퀀스 처리 가능
- 상대적 위치 정보 포함

**단점:**
- 고정된 공식 (데이터에 맞게 조정 불가)
- 매우 긴 시퀀스에서 성능 저하
- 학습된 방법보다 표현력이 제한적

**사용 모델:**
- 원본 Transformer (Attention is All You Need)
- 일부 초기 NMT 모델

## 2.4. Modern Approaches: T5 Bias & ALiBi

**기존 방법의 문제점:**

Learned와 Sinusoidal 모두 임베딩을 입력에 더합니다:
```
X = Token_Embedding + Position_Embedding
```

**새로운 접근: Attention Score에 직접 추가**

```
Attention = softmax(Q·K^T / sqrt(d_k) + Position_Bias) · V
```

### T5 Relative Position Bias

**T5 (2020)의 방법: 상대적 위치에 학습 가능한 bias 추가**

**핵심 아이디어:**

절대 위치가 아닌 상대 위치가 중요합니다:
- "I love transformers"에서
- "love"와 "transformers"의 절대 위치: 1, 2
- 상대 위치: +1 (1칸 차이)

**구현:**

```python
# 상대 거리 계산
relative_distance = position_j - position_i

# Bias 적용
attention_scores = Q @ K.T / sqrt(d_k)
attention_scores += relative_position_bias[relative_distance]
```

**예시:**

```
시퀀스 길이 = 4
토큰: [A, B, C, D]

상대 거리 행렬:
     A   B   C   D
A   [0, -1, -2, -3]
B   [1,  0, -1, -2]
C   [2,  1,  0, -1]
D   [3,  2,  1,  0]

학습된 Bias:
거리 -3: -0.5
거리 -2: -0.3
거리 -1: -0.1
거리  0:  0.5
거리  1: -0.1
거리  2: -0.3
거리  3: -0.5
```

**Bucketing:**

거리가 멀어질수록 중요도가 낮아지므로 그룹화:

```
거리  0: bucket 0
거리 ±1: bucket 1
거리 ±2~3: bucket 2
거리 ±4~7: bucket 3
거리 ±8~15: bucket 4
...
```

이렇게 하면 파라미터 수를 줄일 수 있습니다.

**장점:**
- 상대적 위치 직접 학습
- 긴 시퀀스에도 확장 가능 (bucketing)
- Attention에 직접 영향

**단점:**
- 파라미터 추가 필요
- Head마다 별도 학습

**사용 모델:**
- T5, mT5, ByT5

### ALiBi (Attention with Linear Biases)

**ALiBi (2022): 고정된 선형 bias 사용**

**핵심 아이디어:**

학습 없이 간단한 선형 공식 사용:

```python
bias = -slope * |i - j|

여기서:
- i, j: 토큰 위치
- slope: head마다 다른 상수
```

**예시:**

```
Head 1 (slope = 1):
     0   1   2   3   (Query 위치)
0 [  0  -1  -2  -3]
1 [ -1   0  -1  -2]  (Key 위치)
2 [ -2  -1   0  -1]
3 [ -3  -2  -1   0]

Head 2 (slope = 2):
     0   1   2   3
0 [  0  -2  -4  -6]
1 [ -2   0  -2  -4]
2 [ -4  -2   0  -2]
3 [ -6  -4  -2   0]
```

**Slope 계산:**

```python
# num_heads = 8
# Head별로 다른 slope 할당
slopes = [2^(-8*i/num_heads) for i in range(1, num_heads+1)]

# 예: num_heads=8
slopes = [0.5, 0.354, 0.25, 0.177, 0.125, 0.088, 0.063, 0.044]
```

**왜 작동하나요?**

- 거리가 멀수록 attention score 감소
- 로컬 정보에 더 집중
- Head마다 다른 범위 커버

**구현:**

```python
def get_alibi_bias(seq_len, num_heads):
    # Slope 계산
    slopes = torch.pow(2, -torch.arange(1, num_heads+1) * 8 / num_heads)

    # 거리 행렬
    positions = torch.arange(seq_len)
    distances = positions.unsqueeze(0) - positions.unsqueeze(1)

    # Bias: (num_heads, seq_len, seq_len)
    bias = -slopes.unsqueeze(-1).unsqueeze(-1) * distances.abs()
    return bias
```

**장점:**
- 학습 불필요 (파라미터 0개)
- 임의 길이 시퀀스 처리
- 구현 매우 간단
- 긴 시퀀스에서 좋은 성능

**단점:**
- 고정된 공식 (데이터 맞춤 불가)
- 선형 bias만 사용 (제한적 표현력)

**사용 모델:**
- BLOOM
- MPT
- Falcon

## 2.5. RoPE (Rotary Position Embeddings)

**RoPE: 현대 LLM의 표준 방법 (2021)**

RoPE는 현재 가장 널리 사용되는 Position Embedding 방법입니다.

**핵심 아이디어:**

Query와 Key를 회전 변환(rotation)으로 위치 정보 주입:

```
q_rotated = R(θ_pos) · q
k_rotated = R(θ_pos) · k

Attention = q_rotated · k_rotated^T
```

**2차원 회전 행렬:**

```
R(θ) = [cos(θ)  -sin(θ)]
       [sin(θ)   cos(θ)]
```

**회전의 성질:**

```
R(θ_i) · q · (R(θ_j) · k)^T
= q^T · R(θ_i)^T · R(θ_j) · k
= q^T · R(θ_j - θ_i) · k

→ 상대 위치 (θ_j - θ_i)의 함수!
```

**d차원으로 확장:**

d차원 벡터를 d/2개의 2차원 쌍으로 나눕니다:

```
q = [q_0, q_1, q_2, q_3, ..., q_{d-2}, q_{d-1}]
    └─────┘  └─────┘       └───────────┘
    쌍 0     쌍 1           쌍 d/2-1

각 쌍을 독립적으로 회전:
쌍 i: [q_{2i}, q_{2i+1}] → R(θ^i_pos) · [q_{2i}, q_{2i+1}]
```

**주파수 설정:**

```
θ^i = pos / 10000^(2i/d)

여기서:
- pos: 위치 (0, 1, 2, ...)
- i: 쌍 인덱스 (0, 1, 2, ..., d/2-1)
```

Sinusoidal과 비슷하지만, 더하는 대신 회전!

**구체적인 예시:**

```python
# d_model = 4, pos = 2
q = [q0, q1, q2, q3]

# 주파수 계산
θ^0 = 2 / 10000^(0/4) = 2
θ^1 = 2 / 10000^(2/4) = 0.02

# 쌍 0 회전 (q0, q1):
[q0']   [cos(2)   -sin(2)]   [q0]
[q1'] = [sin(2)    cos(2)] · [q1]

# 쌍 1 회전 (q2, q3):
[q2']   [cos(0.02)  -sin(0.02)]   [q2]
[q3'] = [sin(0.02)   cos(0.02)] · [q3]

q_rotated = [q0', q1', q2', q3']
```

**구현:**

```python
def apply_rope(x, position):
    # x: (batch, seq_len, d_model)
    seq_len, d_model = x.size(1), x.size(2)

    # 주파수 계산
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))

    # 각 위치의 각도
    t = torch.arange(seq_len).type_as(inv_freq)
    freqs = torch.einsum('i,j->ij', t, inv_freq)  # (seq_len, d_model/2)

    # cos, sin 계산
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, d_model)
    cos_emb = emb.cos()
    sin_emb = emb.sin()

    # 회전 적용
    x1, x2 = x[..., ::2], x[..., 1::2]  # 짝수/홀수 분리
    rotated = torch.cat([
        x1 * cos_emb[..., ::2] - x2 * sin_emb[..., 1::2],
        x1 * sin_emb[..., ::2] + x2 * cos_emb[..., 1::2]
    ], dim=-1)

    return rotated
```

**왜 현대 모델이 사용하나요?**

1. **상대 위치의 함수**
   - R(θ_i - θ_j) 형태로 자연스럽게 상대 위치 표현

2. **임의 길이 확장**
   - 학습 시보다 긴 시퀀스도 처리 가능

3. **파라미터 불필요**
   - 계산만으로 구현 (0개 파라미터)

4. **효율적**
   - 간단한 삼각함수 계산

5. **실증적 성능**
   - 대부분의 벤치마크에서 최고 성능

**시각화:**

```
Position 0: 각도 [0°,    0°,    0°,    ...]
Position 1: 각도 [57°,   0.6°,  0.006°, ...]
Position 2: 각도 [114°,  1.1°,  0.011°, ...]
Position 3: 각도 [172°,  1.7°,  0.017°, ...]

→ 낮은 차원: 빠른 회전
→ 높은 차원: 느린 회전
```

**장점:**
- 상대적 위치 자연스럽게 표현
- 임의 길이 시퀀스 처리
- 파라미터 불필요
- 실증적으로 우수한 성능
- 긴 시퀀스에서 안정적

**단점:**
- 구현이 복잡
- 추가 계산 비용 (회전 행렬)

**사용 모델:**
- GPT-3
- GPT-4
- LLaMA, LLaMA 2, LLaMA 3
- Mistral
- 대부분의 최신 LLM

## 2.6. Position Embedding 비교

**종합 비교표:**

| 방법 | 파라미터 | 임의 길이 | 상대 위치 | 성능 | 사용 모델 |
|------|----------|-----------|-----------|------|-----------|
| Learned | 많음 | ❌ | ❌ | 보통 | 초기 BERT, GPT-1 |
| Sinusoidal | 없음 | ✅ | 제한적 | 보통 | 원본 Transformer |
| T5 Bias | 보통 | ✅ | ✅ | 좋음 | T5 family |
| ALiBi | 없음 | ✅ | ✅ | 좋음 | BLOOM, MPT |
| RoPE | 없음 | ✅ | ✅ | 최고 | GPT-3/4, LLaMA |

**언제 무엇을 사용하나요?**

1. **최신 Decoder-only LLM**: RoPE
   - 예: GPT, LLaMA 스타일 모델

2. **메모리 효율이 중요**: ALiBi
   - 예: 제한된 자원 환경

3. **Encoder-Decoder**: T5 Bias
   - 예: 번역, 요약 작업

4. **간단한 실험**: Sinusoidal
   - 예: 프로토타입, 연구

**강의 중 질문:**

Q: "RoPE가 왜 이렇게 복잡한가요? Sinusoidal이 더 간단한데?"
A: RoPE는 상대 위치를 더 자연스럽게 표현합니다. 회전 행렬의 성질상 R(θ_i - θ_j) 형태가 자동으로 나오므로, 모델이 상대적 거리를 학습하기 쉽습니다.

Q: "Learned position embedding은 왜 더 이상 사용하지 않나요?"
A: 학습 시의 최대 길이로 제한되기 때문입니다. 현대 LLM은 점점 더 긴 context를 처리해야 하므로, 임의 길이를 지원하는 방법이 필수가 되었습니다.

---

# 3. Layer Normalization

Layer Normalization은 Transformer의 학습 안정성과 성능에 매우 중요한 역할을 합니다. 이번 섹션에서는 LayerNorm의 변형들을 비교합니다.

## 3.1. LayerNorm이란?

**정규화(Normalization)가 필요한 이유:**

딥러닝 학습 중 문제들:
- **Internal Covariate Shift**: 레이어마다 입력 분포가 변함
- **Gradient Vanishing/Exploding**: Gradient가 너무 작거나 큼
- **학습 불안정**: 학습률 조정이 어려움

**LayerNorm의 해결책:**

각 레이어의 출력을 정규화하여 안정적인 분포 유지

**수식:**

```
LayerNorm(x) = γ · (x - μ) / sqrt(σ² + ε) + β

여기서:
- μ = mean(x)        # 평균
- σ² = variance(x)   # 분산
- γ, β: 학습 가능한 파라미터
- ε: 안정성을 위한 작은 값 (예: 1e-5)
```

**구체적인 예시:**

```python
# 입력: x = (batch_size, seq_len, d_model)
x = [[1.0, 2.0, 3.0, 4.0],   # 토큰 1
     [2.0, 4.0, 6.0, 8.0]]    # 토큰 2
     # d_model = 4

# 토큰 1:
μ = (1.0 + 2.0 + 3.0 + 4.0) / 4 = 2.5
σ² = ((1.0-2.5)² + (2.0-2.5)² + (3.0-2.5)² + (4.0-2.5)²) / 4 = 1.25
σ = sqrt(1.25) ≈ 1.118

정규화:
x_norm = (x - 2.5) / 1.118
       = [-1.34, -0.45, 0.45, 1.34]

최종 출력 (γ=1, β=0 가정):
y = [-1.34, -0.45, 0.45, 1.34]
```

**구현:**

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

**BatchNorm vs LayerNorm:**

```
BatchNorm: 배치 차원에서 정규화
LayerNorm: 특성 차원에서 정규화

예시:
입력: (batch=2, seq=3, d=4)

BatchNorm: 각 특성마다 배치 전체의 평균/분산
  → 2 x 3 = 6개 샘플의 평균/분산 (차원마다)

LayerNorm: 각 샘플의 모든 특성에 대한 평균/분산
  → 4개 특성의 평균/분산 (샘플마다)
```

**왜 Transformer에서 LayerNorm을 사용하나요?**

1. **시퀀스 길이가 가변적**
   - BatchNorm은 고정된 시퀀스 길이 가정
   - LayerNorm은 시퀀스 길이와 무관

2. **작은 배치 크기에서 안정적**
   - BatchNorm은 큰 배치 필요
   - LayerNorm은 배치 크기와 무관

3. **Self-Attention과 잘 맞음**
   - 토큰마다 독립적으로 정규화

## 3.2. Post-norm vs Pre-norm

**Transformer에서 LayerNorm을 어디에 배치하나요?**

### Post-norm (원본 Transformer)

```
sublayer_output = Sublayer(x)
output = LayerNorm(x + sublayer_output)
```

**구조:**

```
x
↓
┌─────────────┐
│ Sublayer    │ (예: Self-Attention)
└─────────────┘
↓
Add (Residual)
↓
LayerNorm
↓
output
```

**특징:**
- 원본 "Attention is All You Need" 논문의 방법
- Residual connection 후 정규화

### Pre-norm (현대 표준)

```
sublayer_output = Sublayer(LayerNorm(x))
output = x + sublayer_output
```

**구조:**

```
x
↓
LayerNorm
↓
┌─────────────┐
│ Sublayer    │
└─────────────┘
↓
Add (Residual)
↓
output
```

**비교:**

| 측면 | Post-norm | Pre-norm |
|------|-----------|----------|
| 원본 논문 | ✅ | ❌ |
| 학습 안정성 | 낮음 | 높음 |
| Gradient flow | 불안정 | 안정 |
| Learning rate warmup | 필요 | 덜 필요 |
| 최종 성능 | 비슷 | 비슷 |
| 현대 사용 | 드물음 | 대부분 |

**왜 Pre-norm이 더 안정적인가요?**

1. **Gradient Flow:**
   ```
   Post-norm:
   Gradient → LayerNorm → Add → Sublayer
   (LayerNorm이 gradient를 방해할 수 있음)

   Pre-norm:
   Gradient → Add → Sublayer → LayerNorm
   (Residual connection으로 gradient 직접 전달)
   ```

2. **학습 초기 안정성:**
   - Pre-norm: 입력이 먼저 정규화되므로 안정적
   - Post-norm: 큰 출력이 그대로 전달될 수 있음

**전체 Transformer Block 비교:**

```python
# Post-norm (Original)
def transformer_block_postnorm(x):
    # Self-Attention
    attn_out = self_attention(x)
    x = layer_norm(x + attn_out)

    # Feed-Forward
    ff_out = feed_forward(x)
    x = layer_norm(x + ff_out)

    return x

# Pre-norm (Modern)
def transformer_block_prenorm(x):
    # Self-Attention
    attn_out = self_attention(layer_norm(x))
    x = x + attn_out

    # Feed-Forward
    ff_out = feed_forward(layer_norm(x))
    x = x + ff_out

    return x
```

**사용 모델:**
- Post-norm: 원본 Transformer, 초기 BERT
- Pre-norm: GPT-2/3/4, LLaMA, 대부분의 현대 모델

## 3.3. RMSNorm

**RMSNorm (Root Mean Square Normalization): LayerNorm의 단순화 버전**

**핵심 아이디어:**

LayerNorm에서 평균 계산을 제거하고 RMS만 사용:

```
RMSNorm(x) = γ · x / RMS(x)

RMS(x) = sqrt(mean(x²) + ε)
```

**LayerNorm vs RMSNorm:**

```
LayerNorm:
1. 평균 계산: μ = mean(x)
2. 분산 계산: σ² = mean((x - μ)²)
3. 정규화: (x - μ) / sqrt(σ² + ε)
4. Affine: γ · x_norm + β

RMSNorm:
1. RMS 계산: rms = sqrt(mean(x²))
2. 정규화: x / rms
3. Affine: γ · x_norm (β 제거)
```

**구현:**

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.gamma * x_norm
```

**예시:**

```python
x = [1.0, 2.0, 3.0, 4.0]

# LayerNorm:
μ = 2.5
σ = 1.118
x_norm = (x - 2.5) / 1.118 = [-1.34, -0.45, 0.45, 1.34]

# RMSNorm:
rms = sqrt((1² + 2² + 3² + 4²) / 4) = sqrt(7.5) ≈ 2.739
x_norm = x / 2.739 = [0.365, 0.730, 1.095, 1.460]
```

**장점:**
- 계산이 더 빠름 (평균 계산 제거)
- 파라미터 감소 (β 제거)
- 실증적으로 비슷한 성능
- 메모리 효율적

**단점:**
- 평균 제거로 인한 표현력 손실 (미미함)

**계산 복잡도:**

```
LayerNorm:
- 평균: O(d)
- 분산: O(d)
- 정규화: O(d)
- Affine: O(d)
→ 총 4 passes

RMSNorm:
- RMS: O(d)
- 정규화: O(d)
- Affine: O(d)
→ 총 3 passes (25% 감소)
```

**사용 모델:**
- LLaMA
- LLaMA 2
- Mistral
- Falcon
- 많은 최신 LLM

**종합 비교:**

| 방법 | 파라미터 | 계산량 | 성능 | 사용 |
|------|----------|--------|------|------|
| LayerNorm | 2d (γ, β) | 높음 | 기준 | BERT, GPT-2 |
| RMSNorm | d (γ) | 낮음 | 비슷 | LLaMA, Mistral |

---

# 4. Attention Variations

Attention의 가장 큰 문제는 O(n²) 복잡도입니다. 이번 섹션에서는 이를 해결하는 다양한 방법들을 학습합니다.

## 4.1. Sparse Attention

**문제: O(n²) 복잡도**

```
시퀀스 길이 n = 512:
Attention Matrix 크기 = 512 × 512 = 262,144

시퀀스 길이 n = 2048:
Attention Matrix 크기 = 2048 × 2048 = 4,194,304 (16배!)

시퀀스 길이 n = 8192:
Attention Matrix 크기 = 8192 × 8192 = 67,108,864 (256배!)
```

**해결책: Sparse Attention**

모든 토큰 쌍을 계산하지 않고 일부만 계산합니다.

### Sliding Window Attention

**핵심 아이디어:**

각 토큰은 주변 k개 토큰만 attention:

```
Full Attention:
     0  1  2  3  4  5  6  7
0   [■  ■  ■  ■  ■  ■  ■  ■]
1   [■  ■  ■  ■  ■  ■  ■  ■]
2   [■  ■  ■  ■  ■  ■  ■  ■]
3   [■  ■  ■  ■  ■  ■  ■  ■]
...

Sliding Window (k=2):
     0  1  2  3  4  5  6  7
0   [■  ■  □  □  □  □  □  □]
1   [■  ■  ■  □  □  □  □  □]
2   [■  ■  ■  ■  □  □  □  □]
3   [□  ■  ■  ■  ■  □  □  □]
4   [□  □  ■  ■  ■  ■  □  □]
...
```

**구현:**

```python
def sliding_window_attention(Q, K, V, window_size=256):
    seq_len = Q.size(1)

    # Attention scores
    scores = Q @ K.transpose(-2, -1) / sqrt(d_k)  # (batch, seq, seq)

    # Window mask 생성
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = False

    # Mask 적용
    scores = scores.masked_fill(mask, float('-inf'))

    # Softmax & output
    attn = torch.softmax(scores, dim=-1)
    output = attn @ V

    return output
```

**복잡도:**

```
Full Attention: O(n²)
Sliding Window: O(n × window_size) = O(n × k)

예: n=4096, window_size=512
Full: 16,777,216 operations
Sliding: 2,097,152 operations (8배 감소)
```

**Receptive Field:**

Computer Vision의 CNN과 비슷한 개념:

```
Layer 1: 각 토큰이 ±2 토큰 참조 (receptive field = 5)
Layer 2: 각 토큰이 ±4 토큰 참조 (receptive field = 9)
Layer 3: 각 토큰이 ±8 토큰 참조 (receptive field = 17)
...
Layer L: receptive field = 5 + 4×(L-1)

예: 12 layers, window=512
→ 최종 receptive field = 5 + 4×11 = 49 (충분히 큼!)
```

**장점:**
- 메모리 사용량 대폭 감소
- 긴 시퀀스 처리 가능
- 로컬 패턴 학습에 효과적

**단점:**
- 먼 거리 관계 학습 어려움
- Global 정보 손실

### Global + Local Attention

**LongFormer (2020) 방법:**

일부 토큰은 global attention, 나머지는 local:

```
     0  1  2  3  4  5  6  7  (0이 global token)
0   [■  ■  ■  ■  ■  ■  ■  ■]  (모든 토큰과 attention)
1   [■  ■  ■  □  □  □  □  □]  (local + global)
2   [■  □  ■  ■  □  □  □  □]  (local + global)
3   [■  □  □  ■  ■  □  □  □]  (local + global)
...
```

**구체적인 예시 (문서 요약):**

```
[CLS] The cat sat on the mat. The dog ran in the park.
 ↑
 Global token - 모든 토큰과 attention

나머지 토큰들: 주변 ±3 토큰만 attention
```

**복잡도:**

```
g: global tokens
l: local window size
n: 시퀀스 길이

Global attention: O(g × n)
Local attention: O((n - g) × l)
Total: O(g×n + n×l)

예: n=4096, g=2, l=256
= 2×4096 + 4096×256
= 8,192 + 1,048,576
= 1,056,768 (전체 대비 ~6%)
```

**사용 모델:**
- LongFormer
- BigBird
- LED (Longformer Encoder-Decoder)

**레이어별 Interleaving:**

일부 레이어는 sparse, 일부는 full:

```
Layer 1: Sliding Window Attention
Layer 2: Sliding Window Attention
Layer 3: Full Attention
Layer 4: Sliding Window Attention
Layer 5: Sliding Window Attention
Layer 6: Full Attention
...
```

이렇게 하면 효율성과 표현력의 균형을 맞출 수 있습니다.

## 4.2. Grouped Query Attention (GQA)

**KV Cache 문제:**

Decoder 추론 시 Key와 Value를 캐싱합니다:

```python
# 토큰 생성 시
for t in range(max_length):
    # 새 토큰의 Q, K, V 계산
    q_t = compute_query(x_t)
    k_t = compute_key(x_t)
    v_t = compute_value(x_t)

    # KV Cache에 추가
    k_cache = concat(k_cache, k_t)  # (seq_len, d_k)
    v_cache = concat(v_cache, v_t)  # (seq_len, d_v)

    # Attention 계산 (모든 이전 토큰 사용)
    output = attention(q_t, k_cache, v_cache)
```

**메모리 문제:**

```
GPT-3 (175B parameters):
- num_heads = 96
- d_k = d_v = 128
- seq_len = 2048

KV Cache per head:
= 2 (K, V) × seq_len × d_k × 2 bytes (fp16)
= 2 × 2048 × 128 × 2
= 1,048,576 bytes = 1 MB

Total (96 heads):
= 96 MB per sample
= 9.6 GB for batch_size=100
```

### Multi-Head Attention (MHA) - 원본

```
Query: num_heads개의 독립적인 projection
Key:   num_heads개의 독립적인 projection
Value: num_heads개의 독립적인 projection

Head 1: Q1, K1, V1
Head 2: Q2, K2, V2
...
Head h: Qh, Kh, Vh
```

**파라미터:**

```
Q: num_heads × d_model × d_k
K: num_heads × d_model × d_k
V: num_heads × d_model × d_v
```

### Multi-Query Attention (MQA) - 극단적 공유

```
Query: num_heads개의 독립적인 projection
Key:   1개만 (모든 head가 공유)
Value: 1개만 (모든 head가 공유)

Head 1: Q1, K_shared, V_shared
Head 2: Q2, K_shared, V_shared
...
Head h: Qh, K_shared, V_shared
```

**파라미터:**

```
Q: num_heads × d_model × d_k
K: 1 × d_model × d_k (공유)
V: 1 × d_model × d_v (공유)
```

**KV Cache 메모리:**

```
MHA: num_heads × seq_len × d_k × 2
MQA: 1 × seq_len × d_k × 2 (num_heads배 감소!)
```

### Grouped Query Attention (GQA) - 중간 접근

```
Query: num_heads개의 독립적인 projection
Key:   num_groups개 (head를 그룹으로 묶어 공유)
Value: num_groups개 (head를 그룹으로 묶어 공유)

예: num_heads=8, num_groups=2

Group 1 (Heads 1-4): Q1, Q2, Q3, Q4, K_group1, V_group1
Group 2 (Heads 5-8): Q5, Q6, Q7, Q8, K_group2, V_group2
```

**구체적인 예시:**

```
LLaMA 2 (70B):
- num_heads = 64
- num_kv_heads = 8 (GQA)
- heads_per_group = 64 / 8 = 8

Head 1-8:   share K1, V1
Head 9-16:  share K2, V2
Head 17-24: share K3, V3
...
Head 57-64: share K8, V8
```

**비교:**

| 방법 | Q projections | K projections | V projections | KV Cache | 성능 |
|------|---------------|---------------|---------------|----------|------|
| MHA | h | h | h | h × seq × d | 최고 |
| GQA | h | g | g | g × seq × d | 높음 |
| MQA | h | 1 | 1 | 1 × seq × d | 보통 |

**메모리 비교 (num_heads=96):**

```
MHA: 96 MB per sample
GQA (groups=8): 8 MB per sample (12배 감소)
MQA: 1 MB per sample (96배 감소)
```

**구현:**

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.heads_per_group = num_heads // num_kv_heads

        # Q는 모든 head에 대해
        self.W_q = nn.Linear(d_model, num_heads * d_k)

        # K, V는 group 수만큼
        self.W_k = nn.Linear(d_model, num_kv_heads * d_k)
        self.W_v = nn.Linear(d_model, num_kv_heads * d_v)

    def forward(self, x, kv_cache=None):
        batch, seq_len, d_model = x.shape

        # Q: (batch, seq, num_heads, d_k)
        Q = self.W_q(x).view(batch, seq_len, self.num_heads, d_k)

        # K, V: (batch, seq, num_kv_heads, d_k)
        K = self.W_k(x).view(batch, seq_len, self.num_kv_heads, d_k)
        V = self.W_v(x).view(batch, seq_len, self.num_kv_heads, d_v)

        # K, V를 각 그룹의 head에 복제
        # (batch, seq, num_kv_heads, d_k) → (batch, seq, num_heads, d_k)
        K = K.repeat_interleave(self.heads_per_group, dim=2)
        V = V.repeat_interleave(self.heads_per_group, dim=2)

        # 나머지는 일반 MHA와 동일
        ...
```

**장점:**
- KV Cache 메모리 대폭 감소
- 추론 속도 향상
- MHA 대비 성능 손실 최소

**단점:**
- MHA보다 약간 낮은 표현력
- 구현 복잡도 증가

**사용 모델:**
- LLaMA 2 (70B)
- Mistral
- Mixtral
- 대부분의 최신 대규모 LLM

**언제 무엇을 사용하나요?**

1. **작은 모델 (< 7B)**: MHA
   - KV Cache가 큰 문제 아님
   - 최대 성능 우선

2. **중간 모델 (7B-70B)**: GQA
   - 효율성과 성능의 균형

3. **매우 큰 모델 (> 70B)**: GQA 또는 MQA
   - 메모리 효율이 필수

---

# 5. Transformer Model Families

Transformer는 다양한 형태로 변형되어 사용됩니다. 이번 섹션에서는 세 가지 주요 아키텍처를 비교합니다.

## 5.1. Encoder-Decoder Models

**원본 Transformer 구조 (2017)**

```
입력 → Encoder → Encoder 출력
               ↓
        Decoder (cross-attention) → 출력
```

**구조:**

```
Encoder:
- Self-Attention (bidirectional)
- Feed-Forward
- 전체 입력을 한번에 처리

Decoder:
- Masked Self-Attention (autoregressive)
- Cross-Attention (Encoder 출력 참조)
- Feed-Forward
- 순차적으로 생성
```

**Cross-Attention:**

```python
# Decoder에서
encoder_output = encoder(source)  # 예: 영어 문장

for t in range(target_length):
    # Self-Attention: 이전 생성 토큰들
    decoder_hidden = self_attention(target[:t])

    # Cross-Attention: Encoder 출력 참조
    # Query: Decoder hidden
    # Key, Value: Encoder output
    context = cross_attention(
        query=decoder_hidden,
        key=encoder_output,
        value=encoder_output
    )

    # 다음 토큰 예측
    next_token = predict(context)
```

### T5 (Text-to-Text Transfer Transformer)

**핵심 아이디어:**

모든 NLP 태스크를 text-to-text 형식으로 통일:

```
번역:
입력: "translate English to German: Hello, how are you?"
출력: "Hallo, wie geht es dir?"

요약:
입력: "summarize: [긴 문서 텍스트...]"
출력: "[요약된 텍스트]"

질문 답변:
입력: "question: What is the capital of France? context: [문맥]"
출력: "Paris"
```

**Pre-training Task: Span Corruption**

```
원본 텍스트:
"The cat sat on the mat and the dog ran in the park."

Corrupted 입력:
"The cat <X> on the mat and <Y> ran in the park."
      ↑                    ↑
   span 1               span 2

목표 출력:
"<X> sat <Y> the dog </s>"
```

**알고리즘:**

```
1. 원본 텍스트에서 15%의 span 선택
2. 각 span을 sentinel 토큰 (<X>, <Y>, ...)로 대체
3. 모델이 sentinel 토큰의 내용 예측
```

**T5 Family:**

| 모델 | 파라미터 | 특징 |
|------|----------|------|
| T5-Small | 60M | 가벼운 실험용 |
| T5-Base | 220M | 기본 모델 |
| T5-Large | 770M | 중간 크기 |
| T5-XL | 3B | 대규모 |
| T5-XXL | 11B | 최대 크기 |
| mT5 | ~13B | 다국어 버전 |
| ByT5 | ~13B | Byte-level |

**장점:**
- 입력과 출력 모두 양방향 참조 가능
- 번역, 요약 등에 강력
- 통일된 프레임워크

**단점:**
- 구조가 복잡 (Encoder + Decoder)
- 파라미터가 많음
- 추론 속도가 느림

**사용 사례:**
- 기계 번역
- 문서 요약
- 질문 답변 (extractive)
- Seq-to-Seq 작업 전반

## 5.2. Encoder-only Models

**Encoder만 사용 (Decoder 제거)**

```
입력 → Encoder → [CLS] 토큰 → Classification
```

**특징:**
- Bidirectional attention (양방향 문맥)
- 모든 토큰이 서로를 참조
- 생성 불가 (classification만)

### BERT (2018)

**구조:**

```
[CLS] The cat sat on the mat [SEP]
  ↓
Encoder Layers (12 or 24)
  ↓
[CLS] 출력 → Classification
각 토큰 출력 → Token-level 작업
```

**Pre-training Tasks:**

1. **MLM (Masked Language Modeling)**

```
원본: "The cat sat on the mat"
입력: "The [MASK] sat on the [MASK]"
목표: "cat", "mat" 예측

마스킹 비율: 15%
```

2. **NSP (Next Sentence Prediction)**

```
Sentence A: "The cat sat on the mat."
Sentence B: "It was sleeping."
Label: IsNext (True/False)

입력: "[CLS] The cat sat... [SEP] It was sleeping. [SEP]"
목표: IsNext 예측
```

**Fine-tuning:**

```python
# Classification
output = bert(input_ids)
cls_output = output[0]  # [CLS] 토큰
logits = classifier(cls_output)
prediction = argmax(logits)

# Token Classification (NER, POS tagging)
token_outputs = output[1:]  # 각 토큰
token_logits = token_classifier(token_outputs)
predictions = argmax(token_logits, dim=-1)
```

**BERT Variants:**

| 모델 | 변경점 | 특징 |
|------|--------|------|
| RoBERTa | NSP 제거, 더 긴 학습 | BERT보다 성능 향상 |
| ALBERT | 파라미터 공유 | 모델 크기 대폭 감소 |
| DistilBERT | Knowledge distillation | 40% 작고 60% 빠름 |
| ELECTRA | Replaced token detection | 효율적 pre-training |

**장점:**
- Bidirectional context (양방향)
- Classification 작업에 강력
- 효율적 (Decoder 없음)

**단점:**
- 텍스트 생성 불가
- Autoregressive 작업 불가

**사용 사례:**
- 텍스트 분류 (sentiment, topic)
- Named Entity Recognition (NER)
- Question Answering (extractive)
- 문장 유사도

## 5.3. Decoder-only Models

**Decoder만 사용 (Encoder와 Cross-Attention 제거)**

```
입력 → Decoder (causal mask) → 다음 토큰 예측
```

**핵심: Causal Masking (인과적 마스킹)**

```
시퀀스: "The cat sat on"

Attention Mask:
        The  cat  sat  on
The    [1    0    0    0]   (The는 자기만 볼 수 있음)
cat    [1    1    0    0]   (cat은 The, cat 볼 수 있음)
sat    [1    1    1    0]   (sat은 The, cat, sat 볼 수 있음)
on     [1    1    1    1]   (on은 모두 볼 수 있음)

→ 이후 토큰을 볼 수 없음 (autoregressive)
```

### GPT Family

**Pre-training: Next Token Prediction**

```
입력:  "The cat sat on the"
목표:  "mat"

입력:  "The cat sat on the mat"
목표:  "."

→ 모든 위치에서 다음 토큰 예측
```

**GPT 발전:**

| 모델 | 연도 | 파라미터 | 특징 |
|------|------|----------|------|
| GPT-1 | 2018 | 117M | 첫 대규모 pre-training |
| GPT-2 | 2019 | 1.5B | Zero-shot learning |
| GPT-3 | 2020 | 175B | Few-shot learning |
| GPT-3.5 | 2022 | ~175B | ChatGPT, RLHF |
| GPT-4 | 2023 | ~1.7T (추정) | Multimodal |

**왜 Decoder-only가 주류가 되었나?**

1. **단순한 구조**
   ```
   Encoder-Decoder: Self-Attn + Cross-Attn + Feed-Forward
   Decoder-only:    Self-Attn (causal) + Feed-Forward
   → 구조가 더 단순
   ```

2. **확장성**
   - 모델 크기 증가가 쉬움
   - 파라미터 효율적

3. **범용성**
   - 생성 작업 자연스럽게 지원
   - In-context learning 가능

4. **실증적 성능**
   - 대규모에서 최고 성능
   - Emergence 현상 (규모가 커지면 새로운 능력 출현)

**In-Context Learning:**

```
입력 (Few-shot):
"Translate English to French:
Hello → Bonjour
Good morning → Bon matin
How are you? → "

모델 출력: "Comment allez-vous?"

→ Fine-tuning 없이 학습!
```

**사용 모델:**
- GPT series
- LLaMA series
- Mistral
- Falcon
- Claude (추정)
- 대부분의 최신 LLM

## 5.4. 모델 선택 가이드

**작업별 추천:**

| 작업 | 추천 아키텍처 | 이유 |
|------|---------------|------|
| 텍스트 생성 | Decoder-only | Autoregressive 생성 |
| 채팅/대화 | Decoder-only | 유연한 생성 |
| 기계 번역 | Encoder-Decoder | 양방향 문맥, 구조적 변환 |
| 문서 요약 | Encoder-Decoder | 압축 & 생성 |
| 분류 | Encoder-only | 양방향 문맥, 효율적 |
| NER/Tagging | Encoder-only | Token-level 작업 |
| Question Answering (extractive) | Encoder-only | 문맥 이해 |
| Question Answering (generative) | Decoder-only | 답변 생성 |

**종합 비교:**

| 측면 | Encoder-Decoder | Encoder-only | Decoder-only |
|------|-----------------|--------------|--------------|
| 구조 복잡도 | 높음 | 낮음 | 낮음 |
| 파라미터 | 많음 | 보통 | 많음 |
| Attention | Bidirectional + Cross | Bidirectional | Causal |
| 생성 가능 | ✅ | ❌ | ✅ |
| 분류 작업 | 보통 | 최고 | 가능 |
| 학습 효율 | 보통 | 높음 | 보통 |
| 추론 속도 | 느림 | 빠름 | 보통 |
| 대표 모델 | T5, BART | BERT, RoBERTa | GPT, LLaMA |
| 현재 트렌드 | 감소 | 감소 | 증가 (주류) |

**현대 LLM 트렌드:**

```
2018-2019: BERT 열풍 (Encoder-only)
2019-2020: T5 등장 (Encoder-Decoder)
2020-현재: GPT-3 이후 Decoder-only 주류

이유:
- 대규모 스케일링에서 최고 성능
- 단순한 구조 → 구현/최적화 용이
- In-context learning 능력
- 범용 작업 지원
```

---

# 6. 요약

이번 강의에서는 Transformer의 다양한 개선 기법들과 모델 variants를 학습했습니다.

## 핵심 개선사항

### 1. Position Embeddings

**발전 과정:**
```
Learned (초기)
  ↓ 길이 제한 문제
Sinusoidal (원본 Transformer)
  ↓ 상대 위치 표현 개선
T5 Bias / ALiBi (Attention에 직접)
  ↓ 더 나은 상대 위치
RoPE (현대 표준)
  → 회전 변환으로 자연스러운 상대 위치
```

**권장사항:**
- 새로운 LLM: RoPE
- 제한된 자원: ALiBi
- Encoder-Decoder: T5 Bias

### 2. Layer Normalization

**발전 과정:**
```
Post-norm (원본)
  ↓ 학습 불안정
Pre-norm (현대 표준)
  ↓ 파라미터 감소
RMSNorm (최신)
  → 빠르고 효율적
```

**권장사항:**
- 기본: Pre-norm
- 효율 중요: RMSNorm
- 원본 재현: Post-norm

### 3. Attention 최적화

**문제와 해결:**

| 문제 | 해결책 | 복잡도 | 사용 |
|------|--------|--------|------|
| O(n²) 복잡도 | Sliding Window | O(n×k) | LongFormer |
| 긴 시퀀스 | Sparse + Global | O(g×n + n×l) | BigBird |
| KV Cache 메모리 | GQA/MQA | 8-96배 감소 | LLaMA 2 |

**권장사항:**
- 긴 문서: Sparse Attention
- 대규모 모델: GQA (그룹화)
- 초대규모 모델: MQA (극단적 공유)

### 4. Model Families

**선택 가이드:**

```
작업 유형:

생성 (텍스트, 코드, 대화)
  → Decoder-only (GPT, LLaMA)

분류 (sentiment, NER)
  → Encoder-only (BERT, RoBERTa)

변환 (번역, 요약)
  → Encoder-Decoder (T5, BART)
```

## 현대 LLM 구성

**대부분의 최신 LLM (GPT-4, LLaMA, Claude):**

```python
class ModernLLM:
    def __init__(self):
        self.architecture = "Decoder-only"
        self.position_embedding = "RoPE"
        self.normalization = "RMSNorm"
        self.norm_position = "Pre-norm"
        self.attention = "GQA"  # 대규모 모델
        self.activation = "SwiGLU"  # 다음 강의 주제

    def forward(self, x):
        for layer in self.layers:
            # Pre-norm
            x_norm = rmsnorm(x)

            # Self-Attention with RoPE and GQA
            attn_out = gqa_attention(apply_rope(x_norm))
            x = x + attn_out

            # Pre-norm
            x_norm = rmsnorm(x)

            # Feed-Forward
            ff_out = feed_forward(x_norm)
            x = x + ff_out

        return x
```

## 실전 구현 체크리스트

**새로운 Transformer 모델 구현 시:**

1. **Position Embedding 선택**
   - [ ] RoPE 구현 (권장)
   - [ ] 또는 ALiBi (간단한 대안)

2. **Normalization 설정**
   - [ ] Pre-norm 구조
   - [ ] RMSNorm 사용 (효율적)

3. **Attention 최적화**
   - [ ] 모델 크기에 따라 MHA/GQA/MQA 선택
   - [ ] 긴 context면 Sliding Window 고려

4. **아키텍처 선택**
   - [ ] 생성 → Decoder-only
   - [ ] 분류 → Encoder-only
   - [ ] 변환 → Encoder-Decoder

5. **효율성 고려**
   - [ ] Flash Attention 구현
   - [ ] KV Cache 최적화
   - [ ] Gradient checkpointing

## 주요 Trade-offs

### Position Embeddings

```
표현력 vs 효율성:
Learned > RoPE > ALiBi > Sinusoidal

확장성:
RoPE = ALiBi = Sinusoidal > Learned
```

### Attention

```
성능:
MHA > GQA > MQA

메모리 효율:
MQA > GQA > MHA

속도:
MQA > GQA > MHA
```

### Model Architecture

```
생성 능력:
Decoder-only = Encoder-Decoder > Encoder-only

효율성:
Encoder-only > Decoder-only > Encoder-Decoder

범용성:
Decoder-only > Encoder-Decoder > Encoder-only
```

---

# 7. 중요 용어 정리

## Position Embedding 관련

- **Learned Position Embeddings**: 위치마다 학습 가능한 벡터를 할당하는 방법
- **Sinusoidal Position Embeddings**: sin/cos 함수로 고정된 위치 임베딩 생성
- **RoPE (Rotary Position Embeddings)**: 회전 행렬로 Query/Key에 위치 정보 주입
- **ALiBi (Attention with Linear Biases)**: Attention score에 선형 bias 추가
- **T5 Relative Position Bias**: 상대 거리에 따라 학습된 bias 추가
- **Bucketing**: 먼 거리를 그룹화하여 파라미터 수 감소

## Normalization 관련

- **LayerNorm**: 각 샘플의 특성 차원에 대해 정규화
- **Post-norm**: Residual connection 후 정규화 (원본 방법)
- **Pre-norm**: Sublayer 전에 정규화 (현대 표준)
- **RMSNorm**: 평균 제거, RMS만 사용하는 단순화된 정규화
- **BatchNorm**: 배치 차원에 대한 정규화 (Transformer에서 거의 사용 안함)

## Attention 관련

- **Sparse Attention**: 모든 토큰 쌍이 아닌 일부만 attention 계산
- **Sliding Window Attention**: 주변 k개 토큰만 attention
- **Global Attention**: 모든 토큰과 attention 계산
- **Receptive Field**: 각 레이어에서 토큰이 참조할 수 있는 범위
- **KV Cache**: 추론 시 Key와 Value를 캐싱하여 재계산 방지
- **MHA (Multi-Head Attention)**: 모든 head가 독립적인 Q, K, V
- **GQA (Grouped Query Attention)**: Head를 그룹으로 묶어 K, V 공유
- **MQA (Multi-Query Attention)**: 모든 head가 하나의 K, V 공유

## Model Architecture 관련

- **Encoder-Decoder**: 입력 인코딩 후 디코딩 (번역, 요약)
- **Encoder-only**: Encoder만 사용 (분류, NER)
- **Decoder-only**: Decoder만 사용 (생성, LLM)
- **Cross-Attention**: Decoder가 Encoder 출력을 참조
- **Causal Masking**: 이후 토큰을 볼 수 없게 마스킹 (autoregressive)
- **Bidirectional Attention**: 양방향 참조 가능 (Encoder)

## Pre-training 관련

- **MLM (Masked Language Modeling)**: 마스킹된 토큰 예측 (BERT)
- **NSP (Next Sentence Prediction)**: 다음 문장 여부 예측 (BERT)
- **Span Corruption**: Span을 마스킹하고 복원 (T5)
- **Next Token Prediction**: 다음 토큰 예측 (GPT)
- **In-Context Learning**: 예시를 입력으로 제공하여 학습

## 최적화 관련

- **Gradient Checkpointing**: 중간 activation 재계산으로 메모리 절약
- **Flash Attention**: 메모리 효율적인 attention 구현
- **Mixed Precision**: FP16/BF16과 FP32 혼합 사용
- **Gradient Accumulation**: 여러 step의 gradient 누적

## 기타

- **Emergence**: 모델 규모 증가로 새로운 능력 출현
- **Scaling Laws**: 모델 크기와 성능의 관계
- **Few-shot Learning**: 적은 예시로 학습
- **Zero-shot Learning**: 예시 없이 작업 수행
- **Fine-tuning**: Pre-trained 모델을 특정 작업에 맞게 재학습

---

**다음 강의 예고:**

Lecture 3에서는 Large Language Models (LLMs)의 구조와 학습 방법을 더 깊이 있게 다룹니다:
- LLM의 스케일링 법칙
- Training infrastructure & techniques
- Tokenization strategies
- Efficient training methods
- Multi-modal extensions

---

**수고하셨습니다!** 🎉