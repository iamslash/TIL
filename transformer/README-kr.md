- [Materials](#materials)
- [Basic](#basic)
  - [1. 전체 모델 구조: Encoder–Decoder 아키텍처](#1-전체-모델-구조-encoderdecoder-아키텍처)
  - [2. Encoder – 여러 층의 인코딩 레이어](#2-encoder--여러-층의-인코딩-레이어)
    - [2-1. 동일한 레이어 복제: `clones`](#2-1-동일한-레이어-복제-clones)
    - [2-2. Layer Normalization](#2-2-layer-normalization)
    - [2-3. Residual 연결 + Layer Normalization: `SublayerConnection`](#2-3-residual-연결--layer-normalization-sublayerconnection)
    - [2-4. Encoder Layer](#2-4-encoder-layer)
    - [2-5. 전체 Encoder](#2-5-전체-encoder)
  - [3. Decoder – 인코더-디코더 주의와 자기 주의](#3-decoder--인코더-디코더-주의와-자기-주의)
    - [3-1. Decoder Layer](#3-1-decoder-layer)
    - [3-2. 전체 Decoder](#3-2-전체-decoder)
    - [3-3. 미래 단어 마스킹: `subsequent_mask`](#3-3-미래-단어-마스킹-subsequent_mask)
  - [4. Attention 메커니즘](#4-attention-메커니즘)
    - [4-1. Scaled Dot-Product Attention](#4-1-scaled-dot-product-attention)
    - [4-2. Multi-Head Attention](#4-2-multi-head-attention)
  - [5. Position-wise Feed-Forward Network](#5-position-wise-feed-forward-network)
  - [6. 임베딩과 Positional Encoding](#6-임베딩과-positional-encoding)
    - [6-1. Embeddings](#6-1-embeddings)
    - [6-2. Positional Encoding](#6-2-positional-encoding)
  - [7. 전체 모델 구성: `make_model`](#7-전체-모델-구성-make_model)
  - [8. 학습 관련 구성 요소](#8-학습-관련-구성-요소)
    - [8-1. Batch와 마스킹 처리](#8-1-batch와-마스킹-처리)
    - [8-2. Label Smoothing](#8-2-label-smoothing)
    - [8-3. 학습률 스케줄러 (NoamOpt)](#8-3-학습률-스케줄러-noamopt)
  - [9. Greedy Decoding (추론)](#9-greedy-decoding-추론)
  - [10. 결론](#10-결론)
  - [Colab PyTorch Code](#colab-pytorch-code)

----

# Materials

- [The Illustrated Transformer](https://nlpinkorean.github.io/illustrated-transformer/)
* [Transformer 모델 (Attention is all you need)](https://gaussian37.github.io/dl-concept-transformer/)
  * [Attention 메커니즘의 이해](https://gaussian37.github.io/dl-concept-attention/)
* [[딥러닝 기계 번역] Transformer: Attention Is All You Need (꼼꼼한 딥러닝 논문 리뷰와 코드 실습) | youtube](https://www.youtube.com/watch?v=AA621UofTUA)
  * [pdf](https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/lecture_notes/Transformer.pdf)
  * [src](https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/Attention_is_All_You_Need_Tutorial_(German_English).ipynb)
* [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
* [Pytorch Transformers from Scratch (Attention is all you need) | youtube](https://www.youtube.com/watch?v=U0s0f995w14)
  * [pytorch src](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py)
* [Visual Guide to Transformer Neural Networks - (Episode 1) Position Embeddings | youtube](https://www.youtube.com/watch?v=dichIcUZfOw)
  * [Visual Guide to Transformer Neural Networks - (Episode 2) Multi-Head & Self-Attention | youtube](https://www.youtube.com/watch?v=mMa2PmYJlCo)
  * [Visual Guide to Transformer Neural Networks - (Episode 3) Decoder’s Masked Attention | youtube](https://www.youtube.com/watch?v=gJ9kaJsE78k)

# Basic

## 바깥쪽에서 안쪽으로: Transformer 이해하기

이 섹션은 Transformer를 큰 그림부터 세부사항까지 단계적으로 설명합니다. Junior Software Engineer도 이해할 수 있도록 구체적인 비유와 예시를 사용합니다.

### Level 1: Transformer는 번역기입니다

가장 바깥쪽 관점에서 보면, Transformer는 **한 언어를 다른 언어로 번역하는 기계**입니다.

```
입력: "I love cats"    →    [Transformer]    →    출력: "나는 고양이를 사랑해"
```

**핵심 개념:**
- 입력 문장을 받아서 출력 문장을 생성
- 한 번에 전체 입력을 보고, 한 단어씩 출력을 생성
- 기존 RNN과 달리 순서대로 처리하지 않고 병렬로 처리 가능

### Level 2: 두 개의 큰 블록 - Encoder와 Decoder

Transformer는 두 개의 큰 블록으로 구성됩니다:

```
[Encoder: 이해하는 부분]          [Decoder: 생성하는 부분]
      ↓                                ↓
입력 문장을 숫자로 이해         →    출력 문장을 한 단어씩 생성
```

**통역사 비유:**
- **Encoder**: 영어를 듣고 이해하는 통역사의 "듣기" 능력
  - "I love cats"를 듣고 의미를 파악
  - 결과: 문장의 의미를 담은 숫자들의 집합

- **Decoder**: 이해한 내용을 한국어로 말하는 통역사의 "말하기" 능력
  - Encoder가 이해한 의미를 보면서
  - "나는" → "고양이를" → "사랑해" 순서대로 생성

### Level 3: 6개의 층으로 쌓아올림

Encoder와 Decoder는 각각 **6개의 동일한 층(layer)**으로 구성됩니다:

```
Encoder (6층)                    Decoder (6층)
┌─────────────┐                 ┌─────────────┐
│   Layer 6   │                 │   Layer 6   │
├─────────────┤                 ├─────────────┤
│   Layer 5   │                 │   Layer 5   │
├─────────────┤                 ├─────────────┤
│   Layer 4   │                 │   Layer 4   │
├─────────────┤                 ├─────────────┤
│   Layer 3   │                 │   Layer 3   │
├─────────────┤                 ├─────────────┤
│   Layer 2   │                 │   Layer 2   │
├─────────────┤                 ├─────────────┤
│   Layer 1   │                 │   Layer 1   │
└─────────────┘                 └─────────────┘
      ↑                               ↑
 "I love cats"                  <start> + Encoder 출력
```

**왜 6층일까?**
- 1층만 있으면: 단순한 패턴만 학습
- 6층을 쌓으면: 점점 더 복잡하고 추상적인 패턴 학습
  - 1-2층: 단어 차원의 패턴
  - 3-4층: 구문 차원의 패턴
  - 5-6층: 문장 전체의 의미

### Level 4: 핵심 메커니즘 - Attention (주의 집중)

각 층 안에서 가장 중요한 것은 **Attention** 메커니즘입니다.

**회의실 비유:**

```
회의실에 3명이 있습니다:
- Query (질문자): "cats에 대해 알고 싶어요"
- Key (안내자들): ["I 정보", "love 정보", "cats 정보"]
- Value (실제 내용): ["I 상세", "love 상세", "cats 상세"]

과정:
1. Query가 각 Key에게 물어봄: "너는 내 질문과 얼마나 관련있어?"
   - "I"와의 관련성: 0.1
   - "love"와의 관련성: 0.3
   - "cats"와의 관련성: 0.6

2. 관련성 점수로 Value를 가중 평균:
   결과 = 0.1 × "I 상세" + 0.3 × "love 상세" + 0.6 × "cats 상세"
```

**구체적인 예시:**

```python
# "cats"가 다른 단어들과 얼마나 관련있는지 계산

입력 문장: "I love cats"

Q (Query): cats가 궁금함
K (Keys): [I, love, cats] 각각의 특징
V (Values): [I, love, cats] 각각의 실제 정보

단계 1: 유사도 계산 (Q와 K의 내적)
cats·I = 0.2
cats·love = 0.8
cats·cats = 1.0

단계 2: Softmax로 확률 변환
[0.1, 0.3, 0.6]  # 합이 1

단계 3: Value를 가중평균
출력 = 0.1×V[I] + 0.3×V[love] + 0.6×V[cats]
```

**Multi-Head Attention (여러 관점으로 보기):**

영화 리뷰 팀 비유:
- Head 1 (스토리 전문가): 줄거리 관점에서 분석
- Head 2 (연기 전문가): 배우 연기 관점에서 분석
- Head 3 (촬영 전문가): 영상미 관점에서 분석
- ...
- Head 8 (음악 전문가): 음악 관점에서 분석

→ 8명의 전문가 의견을 종합하여 최종 평가

Transformer에서:
- 8개의 Head가 각각 다른 관점으로 문장을 분석
- Head 1: 주어-동사 관계
- Head 2: 수식 관계
- Head 3: 의미적 유사성
- ...
- 모든 Head의 결과를 합쳐서 종합적인 이해

### Level 5: 위치 정보와 Masking

**Positional Encoding (위치 정보):**

사진 앨범 비유:
- 사진들이 뒤섞여 있으면 이야기 순서를 알 수 없음
- 각 사진에 번호표를 붙이면 순서를 알 수 있음

```
"I love cats"

I → 위치 0 + 단어 의미
love → 위치 1 + 단어 의미
cats → 위치 2 + 단어 의미
```

Transformer는 sine/cosine 함수로 위치 정보를 추가:
```
pos=0: [sin(0/10000), cos(0/10000), sin(0/100), cos(0/100), ...]
pos=1: [sin(1/10000), cos(1/10000), sin(1/100), cos(1/100), ...]
pos=2: [sin(2/10000), cos(2/10000), sin(2/100), cos(2/100), ...]
```

**Masking (가리기):**

시험 비유:
- 문제 1, 2, 3이 있는데
- 문제 1을 풀 때는 문제 2, 3을 가려야 함 (커닝 방지)

Decoder에서:
```
"나는"을 생성할 때 → "고양이를", "사랑해"를 볼 수 없음
"고양이를"을 생성할 때 → "사랑해"를 볼 수 없음

Mask:
[1, 0, 0]   # 첫 번째 단어: 자기만 볼 수 있음
[1, 1, 0]   # 두 번째 단어: 첫 번째와 자기만
[1, 1, 1]   # 세 번째 단어: 모두 볼 수 있음
```

### Level 6: 구체적인 숫자 예시

**전체 흐름을 숫자로:**

```python
# 1단계: 입력
입력: "I love cats"
vocab_size = 10000 (사전 크기)
d_model = 512 (벡터 차원)

# 2단계: 임베딩
"I" → [0.2, 0.5, 0.1, ..., 0.8]  # 512차원
"love" → [0.7, 0.3, 0.9, ..., 0.2]
"cats" → [0.4, 0.8, 0.2, ..., 0.6]

# 3단계: Positional Encoding 추가
"I" + pos(0) → [0.2+0.0, 0.5+0.1, 0.1+0.0, ...]
"love" + pos(1) → [0.7+0.01, 0.3+0.09, 0.9+0.01, ...]
"cats" + pos(2) → [0.4+0.02, 0.8+0.08, 0.2+0.02, ...]

# 4단계: Encoder Layer 1
## Self-Attention
Q = [입력] × W_Q  # W_Q는 512×512 행렬
K = [입력] × W_K
V = [입력] × W_V

Attention(Q,K,V) = softmax(Q×K^T / √512) × V

## Feed-Forward
출력 = ReLU([Attention결과] × W1 + b1) × W2 + b2
# W1: 512×2048, W2: 2048×512

# 5-9단계: Encoder Layer 2~6 반복

# 10단계: Decoder
<start> → "나는" → "고양이를" → "사랑해" → <end>

각 단계마다:
- Self-Attention (지금까지 생성한 단어들)
- Encoder-Decoder Attention (Encoder 출력 참조)
- Feed-Forward
- 다음 단어 확률 분포 생성

# 11단계: 출력
Linear + Softmax → [P("나는")=0.8, P("저는")=0.15, ...]
가장 높은 확률의 단어 선택
```

### 전체 흐름 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│                        입력: "I love cats"                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    ┌──────────────────┐
                    │  Token Embedding  │
                    └──────────────────┘
                              ↓
                    ┌──────────────────┐
                    │Positional Encoding│
                    └──────────────────┘
                              ↓
        ┌─────────────────────────────────────────┐
        │          Encoder (6 layers)             │
        │  ┌───────────────────────────────────┐  │
        │  │ Layer 1-6 각각:                   │  │
        │  │  • Multi-Head Self-Attention      │  │
        │  │  • Add & Norm                     │  │
        │  │  • Feed Forward                   │  │
        │  │  • Add & Norm                     │  │
        │  └───────────────────────────────────┘  │
        └─────────────────────────────────────────┘
                              ↓
                    [메모리: Encoder 출력]
                              ↓
        ┌─────────────────────────────────────────┐
        │          Decoder (6 layers)             │
        │  ┌───────────────────────────────────┐  │
        │  │ Layer 1-6 각각:                   │  │
        │  │  • Masked Multi-Head Self-Att    │  │
        │  │  • Add & Norm                     │  │
        │  │  • Multi-Head Encoder-Dec Att    │  │
        │  │  • Add & Norm                     │  │
        │  │  • Feed Forward                   │  │
        │  │  • Add & Norm                     │  │
        │  └───────────────────────────────────┘  │
        └─────────────────────────────────────────┘
                              ↓
                    ┌──────────────────┐
                    │ Linear + Softmax │
                    └──────────────────┘
                              ↓
            출력: "나는 고양이를 사랑해"
```

### 핵심 개념 5가지 요약

1. **Encoder-Decoder 구조**: 이해(Encoder) + 생성(Decoder)
2. **Self-Attention**: 문장 내 단어들 간의 관계 파악
3. **Multi-Head**: 여러 관점으로 동시에 분석
4. **Positional Encoding**: 단어 순서 정보 추가
5. **Masking**: 미래 정보를 보지 못하도록 차단

---

아래는 "The Annotated Transformer" 글의 주요 부분들을 단계별로 설명하면서, 관련된 PyTorch 코드를 조금씩 추가하는 방식의 설명입니다. 각 단계마다 Transformer의 구성요소와 그 역할, 그리고 해당 코드가 어떻게 구현되어 있는지 함께 살펴보겠습니다.

---

## 1. 전체 모델 구조: Encoder–Decoder 아키텍처

Transformer는 **Encoder**와 **Decoder**로 구성된 모델입니다.

**일상 비유: 통역사**

```
[Encoder: 듣기 전문가]                [Decoder: 말하기 전문가]
영어 문장 듣기                        한국어 문장 말하기
↓                                    ↓
"I love cats"                        "나는"
↓                                    ↓
뇌에서 의미 이해                       "고양이를"
↓                                    ↓
[의미 표현 저장]  ----전달---->         "사랑해"
```

- **Encoder**: 입력 문장을 "의미 표현"으로 변환
  - "I love cats" → [벡터들의 시퀀스]
  - 각 단어의 의미 + 문맥 정보를 숫자로 표현

- **Decoder**: 의미 표현을 보고 한 단어씩 생성
  - [벡터들] → "나는" → "고양이를" → "사랑해"
  - 이전에 생성한 단어와 Encoder 정보를 참고

**구체적인 흐름:**

```
입력: "I love cats"

1. 단어 임베딩 + 위치 정보
   I(0) → [0.1, 0.5, ...]
   love(1) → [0.8, 0.3, ...]
   cats(2) → [0.2, 0.9, ...]

2. Encoder (6개 층)
   각 층마다:
   - Self-Attention: 단어들 간의 관계 파악
   - Feed-Forward: 정보 변환
   최종 출력: "문장 전체의 의미 표현"

3. Decoder (6개 층)
   각 층마다:
   - Self-Attention: 지금까지 생성한 단어들 간 관계
   - Encoder-Decoder Attention: Encoder 정보 참조
   - Feed-Forward: 정보 변환

4. Generator
   최종 벡터 → 단어 확률 분포
   [0.01, 0.85, 0.02, ...] → "고양이를" 선택
```

이러한 구조를 반영한 기본 클래스는 다음과 같습니다:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy

class EncoderDecoder(nn.Module):
    """
    표준 Encoder-Decoder 아키텍처.
    encoder, decoder, src와 tgt 임베딩, 그리고 최종 출력(softmax)을 위한 generator를 포함합니다.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        매개변수:
            encoder (Encoder): 입력 문장을 처리하는 인코더 모듈
                예: Encoder(EncoderLayer(...), N=6)
            decoder (Decoder): 출력 문장을 생성하는 디코더 모듈
                예: Decoder(DecoderLayer(...), N=6)
            src_embed (nn.Sequential): 입력 단어 임베딩 + 위치 인코딩
                예: nn.Sequential(Embeddings(512, 10000), PositionalEncoding(512, 0.1))
            tgt_embed (nn.Sequential): 출력 단어 임베딩 + 위치 인코딩
                예: nn.Sequential(Embeddings(512, 10000), PositionalEncoding(512, 0.1))
            generator (Generator): 최종 단어 확률 계산기
                예: Generator(d_model=512, vocab=10000)
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        순전파 연산

        매개변수:
            src (Tensor): 입력 문장의 단어 인덱스
                shape: (batch_size, src_seq_len)
                예: tensor([[1, 234, 56, 789, 2]])  # [시작토큰, I, love, cats, 종료토큰]
            tgt (Tensor): 출력 문장의 단어 인덱스
                shape: (batch_size, tgt_seq_len)
                예: tensor([[1, 45, 678, 90, 2]])  # [시작토큰, 나는, 고양이를, 사랑해, 종료토큰]
            src_mask (Tensor): 입력 문장의 패딩 마스크
                shape: (batch_size, 1, src_seq_len)
                예: tensor([[[1, 1, 1, 1, 1]]])  # 모두 실제 단어 (1=사용, 0=패딩)
            tgt_mask (Tensor): 출력 문장의 미래 단어 + 패딩 마스크
                shape: (batch_size, tgt_seq_len, tgt_seq_len)
                예: tensor([[[1, 0, 0],
                           [1, 1, 0],
                           [1, 1, 1]]])  # 하삼각 행렬 (미래 보기 방지)

        반환값:
            output (Tensor): 각 위치의 다음 단어 확률 분포
                shape: (batch_size, tgt_seq_len, vocab_size)
                예: shape (1, 5, 10000)  # 배치1, 길이5, 어휘10000
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """
        입력 문장을 의미 벡터로 인코딩

        매개변수:
            src (Tensor): 입력 단어 인덱스, shape: (batch, src_len)
            src_mask (Tensor): 패딩 마스크, shape: (batch, 1, src_len)

        반환값:
            memory (Tensor): 인코딩된 문장 표현
                shape: (batch, src_len, d_model)
                예: shape (1, 5, 512)  # 5개 단어, 각 512차원 벡터
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        인코딩된 정보를 바탕으로 출력 문장 생성

        매개변수:
            memory (Tensor): 인코더 출력, shape: (batch, src_len, d_model)
                예: shape (1, 5, 512)
            src_mask (Tensor): 입력 마스크, shape: (batch, 1, src_len)
            tgt (Tensor): 출력 단어 인덱스, shape: (batch, tgt_len)
                예: tensor([[1, 45, 678, 90]])
            tgt_mask (Tensor): 출력 마스크, shape: (batch, tgt_len, tgt_len)

        반환값:
            output (Tensor): 디코더 출력 벡터
                shape: (batch, tgt_len, d_model)
                예: shape (1, 4, 512)
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "디코더 출력을 단어 확률 분포로 변환하는 모듈"

    def __init__(self, d_model, vocab):
        """
        매개변수:
            d_model (int): 모델의 hidden dimension 크기
                예: 512 (Transformer 원 논문 기준)
            vocab (int): 어휘 사전 크기 (가능한 단어 개수)
                예: 10000 (영어 단어 10,000개)
                예: 32000 (BPE 토크나이저 사용 시)
        """
        super(Generator, self).__init__()
        # 선형 변환: (batch, seq_len, 512) → (batch, seq_len, 10000)
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """
        hidden vector를 단어 확률로 변환

        매개변수:
            x (Tensor): 디코더의 출력 벡터
                shape: (batch, seq_len, d_model)
                예: shape (2, 10, 512)  # 배치 2, 길이 10, 차원 512
                예: tensor([[[0.2, 0.5, -0.1, ...],    # 위치 0의 벡터
                           [0.8, -0.3, 0.4, ...],   # 위치 1의 벡터
                           ...]])

        반환값:
            probs (Tensor): 각 단어의 로그 확률
                shape: (batch, seq_len, vocab)
                예: shape (2, 10, 10000)
                예: tensor([[[-8.5, -2.3, -0.1, ...],  # 위치 0: 각 단어 확률
                           [-7.2, -1.5, -0.05, ...], # 위치 1: 각 단어 확률
                           ...]])
                설명: 값이 클수록 해당 단어일 확률이 높음
                     log_softmax이므로 실제 확률은 exp(값)

        예시:
            >>> x = torch.randn(1, 3, 512)  # 배치1, 길이3, 차원512
            >>> generator = Generator(512, 10000)
            >>> output = generator(x)
            >>> output.shape
            torch.Size([1, 3, 10000])

            >>> # 가장 확률 높은 단어 찾기
            >>> next_word = output[0, -1].argmax()  # 마지막 위치의 예측 단어
            >>> print(next_word)  # 예: tensor(234) → "cats"
        """
        return F.log_softmax(self.proj(x), dim=-1)
```

> **설명:**
>
> **EncoderDecoder 클래스:**
> - 전체 Transformer 모델의 뼈대
> - encoder, decoder, 임베딩, generator를 하나로 연결
> - `forward()`: 입력 문장 → 출력 확률 분포 (end-to-end)
> - `encode()`: 입력 → 의미 벡터 (memory)
> - `decode()`: 의미 벡터 + 이전 출력 → 다음 단어 벡터
>
> **Generator 클래스:**
> - 디코더의 512차원 벡터를 10000차원 단어 확률로 변환
> - Linear(512, 10000): 각 단어에 대한 점수 계산
> - log_softmax: 점수를 로그 확률로 변환 (학습 시 수치 안정성)
>
> **실제 사용 예시:**
> ```python
> # 모델 생성
> model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)
>
> # 번역: "I love cats" → "나는 고양이를 사랑해"
> src = torch.LongTensor([[1, 234, 56, 789, 2]])  # [BOS, I, love, cats, EOS]
> tgt = torch.LongTensor([[1, 45, 678, 90]])      # [BOS, 나는, 고양이를, 사랑해]
> src_mask = (src != 0).unsqueeze(-2)              # 패딩 마스크
> tgt_mask = subsequent_mask(tgt.size(1))          # 미래 마스킹
>
> # 순전파
> output = model(src, tgt, src_mask, tgt_mask)     # shape: (1, 4, 10000)
>
> # 다음 단어 예측
> next_word_probs = output[0, -1]                  # 마지막 위치의 확률
> next_word_id = next_word_probs.argmax()          # 가장 높은 확률 단어
> print(f"다음 단어 ID: {next_word_id}")           # 예: 2 (EOS 토큰)
> ```

---

## 2. Encoder – 여러 층의 인코딩 레이어

Transformer의 인코더는 동일한 구조의 **N**개의 레이어를 쌓아 구성됩니다.  
각 레이어는 두 개의 서브 레이어로 구성되는데, 하나는 **Self-Attention** (자기 자신에 대한 주의집중)이고 다른 하나는 **Position-wise Feed-Forward Network**입니다.

### 2-1. 동일한 레이어 복제: `clones`

```python
def clones(module, N):
    "주어진 모듈을 N번 복제하여 ModuleList로 반환합니다."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```

### 2-2. Layer Normalization

Residual 연결과 함께 각 서브 레이어의 출력을 정규화하기 위해 LayerNorm을 사용합니다.

```python
class LayerNorm(nn.Module):
    "Layer Normalization 모듈."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))   # 스케일 파라미터
        self.b_2 = nn.Parameter(torch.zeros(features))  # shift 파라미터
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std  = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

### 2-3. Residual 연결 + Layer Normalization: `SublayerConnection`

각 서브 레이어는 residual 연결을 거친 후 정규화되며, 드롭아웃도 적용됩니다.

```python
class SublayerConnection(nn.Module):
    """
    Residual 연결 후 Layer Normalization을 적용합니다.
    (코드 단순화를 위해 먼저 정규화를 수행한 뒤 sublayer를 적용합니다.)
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "같은 차원의 sublayer에 residual 연결을 적용합니다."
        return x + self.dropout(sublayer(self.norm(x)))
```

### 2-4. Encoder Layer

인코더의 한 층은 self-attention과 feed-forward 네트워크 두 부분으로 구성됩니다.

```python
class EncoderLayer(nn.Module):
    "인코더 한 층: self-attention과 feed-forward로 구성됩니다."
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn              # self-attention 서브 레이어
        self.feed_forward = feed_forward        # 피드포워드 네트워크
        # 동일한 구성의 SublayerConnection을 두 번 사용
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "입력 x에 대해 self-attention과 feed-forward를 순차적으로 적용합니다."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
```

### 2-5. 전체 Encoder

여러 EncoderLayer를 쌓은 전체 인코더는 마지막에 LayerNorm을 한 번 더 적용합니다.

```python
class Encoder(nn.Module):
    "N개의 인코더 레이어로 구성된 전체 인코더."
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "입력 x와 마스크를 각 레이어에 순차적으로 적용한 후 정규화합니다."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

> **설명:**  
> 위 코드는 인코더를 구성하는 핵심 요소들입니다. 입력 임베딩을 받아 여러 인코더 레이어를 통과시키고, 마지막에 정규화를 수행하여 출력(hidden state)을 생성합니다.

---

## 3. Decoder – 인코더-디코더 주의와 자기 주의

디코더도 인코더와 유사하게 N개의 레이어로 구성되지만, 각 레이어는 세 개의 서브 레이어로 구성됩니다.
1. **자기 자기(self) attention:** 디코더 내에서 현재까지 생성된 단어들끼리의 관계를 모델링합니다.  
   – 단, 미래의 단어를 보지 못하도록 **mask**가 적용됩니다.
2. **인코더-디코더 attention:** 인코더의 출력(메모리)을 이용해 입력 문장과의 연관성을 고려합니다.
3. **Feed-Forward 네트워크:** 각 위치별로 동일하게 적용되는 피드포워드 네트워크.

### 3-1. Decoder Layer

```python
class DecoderLayer(nn.Module):
    "디코더 한 층: self-attention, src-attention, 그리고 feed-forward로 구성됩니다."
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn         # 디코더 자기 주의
        self.src_attn = src_attn           # 인코더-디코더 주의
        self.feed_forward = feed_forward   # 피드포워드 네트워크
        # 3개의 SublayerConnection을 사용
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "그림 1 (오른쪽)과 같이 순서대로 적용합니다."
        m = memory
        # 1) 디코더 자기 주의 (미래 단어를 보지 못하도록 tgt_mask 사용)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 2) 인코더-디코더 주의 (encoder의 메모리 m 사용)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 3) 피드포워드 네트워크
        return self.sublayer[2](x, self.feed_forward)
```

### 3-2. 전체 Decoder

```python
class Decoder(nn.Module):
    "N개의 디코더 레이어로 구성된 전체 디코더."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
```

### 3-3. 미래 단어 마스킹: `subsequent_mask`

디코더에서 자기 주의(self-attention)를 할 때, 미래의 단어(아직 생성되지 않은 단어)를 보지 못하도록 마스킹합니다.

**왜 마스킹이 필요한가요?**

시험 중에 답안지를 미리 보는 것을 방지하는 것과 같습니다.

```
번역 중: "I love cats" → "나는 고양이를 사랑해"

생성 과정:
1단계: "나는" 생성 - "I", "love", "cats" 참조 가능
2단계: "고양이를" 생성 - "나는"까지만 참조 가능 (미래인 "사랑해"를 보면 안 됨!)
3단계: "사랑해" 생성 - "나는", "고양이를"까지만 참조 가능

만약 미래를 볼 수 있다면?
- 학습: 답을 보고 풀기 = 부정행위
- 추론: 아직 생성 안 한 단어는 존재하지 않음 → 에러
```

**마스크 행렬 시각화:**

```
문장: "나는 고양이를 사랑해"

Attention 가능 여부 (1=가능, 0=불가능):
           나는  고양이를  사랑해
나는    [  1      0       0   ]  ← "나는"는 자기 자신만 참조
고양이를 [  1      1       0   ]  ← "고양이를"은 자기와 이전 단어 참조
사랑해  [  1      1       1   ]  ← "사랑해"는 모든 이전 단어 참조
```

```python
import numpy as np
import matplotlib.pyplot as plt

def subsequent_mask(size):
    "미래 단어들을 마스킹합니다. (lower-triangular matrix)"
    attn_shape = (1, size, size)
    # k=1부터 상삼각행렬 생성 -> 미래 단어에 해당하는 부분은 0
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

# 마스킹 결과 시각화 예시:
plt.figure(figsize=(5,5))
plt.imshow(subsequent_mask(20)[0])
plt.title("Subsequent Mask")
plt.show()
```

> **설명:**
>
> **구체적인 예시:**
> ```python
> # 길이 4인 시퀀스의 마스크
> mask = subsequent_mask(4)
> print(mask[0])
> # 결과:
> # [[1, 0, 0, 0],   ← 위치 0: 자기 자신만
> #  [1, 1, 0, 0],   ← 위치 1: 0, 1 참조 가능
> #  [1, 1, 1, 0],   ← 위치 2: 0, 1, 2 참조 가능
> #  [1, 1, 1, 1]]   ← 위치 3: 모든 이전 단어 참조 가능
> ```
>
> 이 마스크는 attention 계산 시 미래 단어에 해당하는 점수를 -∞로 만들어서, softmax 후 확률이 0이 되게 합니다. 학습 시 디코더가 올바른 auto-regressive(순차적) 생성을 하게끔 돕습니다.

---

## 4. Attention 메커니즘

Transformer의 핵심은 **Scaled Dot-Product Attention**입니다.

### 4-0. Query, Key, Value (Q, K, V)란 무엇인가?

**도서관 검색 시스템으로 이해하기:**

Attention을 이해하려면 먼저 Query, Key, Value의 의미를 알아야 합니다. 도서관에서 책을 찾는 상황으로 비유해보겠습니다.

```
도서관 상황:
- Query (검색어): "딥러닝 입문서"라고 검색합니다
- Key (책 제목/태그): 각 책마다 붙어있는 제목이나 태그
- Value (책 내용): 실제 책의 내용
```

**어떻게 작동하나요?**

1. **검색 (Query와 Key 비교)**
   - 당신의 검색어 "딥러닝 입문서"(Query)와 각 책의 제목/태그(Key)를 비교
   - "딥러닝 기초" 책 → 유사도 0.9 (매우 유사)
   - "머신러닝 실전" 책 → 유사도 0.7 (조금 유사)
   - "요리 레시피" 책 → 유사도 0.1 (거의 무관)

2. **가중치 계산 (Softmax)**
   - 유사도를 확률로 변환: [0.5, 0.4, 0.1]
   - 가장 관련 있는 책에 높은 가중치 부여

3. **정보 추출 (Value 가중합)**
   - 각 책의 내용(Value)을 가중치만큼 섞어서 가져옵니다
   - 결과 = 0.5 × "딥러닝 기초" 내용 + 0.4 × "머신러닝 실전" 내용 + 0.1 × "요리 레시피" 내용

**Transformer에서의 Q, K, V:**

```python
# 예시: "I love cats" 문장을 번역할 때
문장: ["I", "love", "cats"]

# "love"라는 단어를 처리할 때
Query (love):  "내가 지금 집중하고 싶은 정보는?"
Key (I):       "나는 주어입니다"
Key (love):    "나는 동사입니다"
Key (cats):    "나는 목적어입니다"
Value (I):     [주어의 실제 의미 벡터]
Value (love):  [동사의 실제 의미 벡터]
Value (cats):  [목적어의 실제 의미 벡터]

# Attention 계산
1. Query(love)와 각 Key를 비교 → 유사도 계산
2. cats와 관련성이 높으면 가중치 ↑
3. 가중치로 Value들을 섞어서 최종 표현 생성
```

**왜 Query, Key, Value로 나누나요?**

- **유연성**: 같은 단어라도 문맥에 따라 다른 역할
  - Key로서는 "나는 명사야"
  - Value로서는 [실제 의미 정보]
  - Query로서는 "나와 관련된 단어를 찾아줘"

- **효율성**: 한 번 계산한 Key와 Value는 재사용 가능

**구체적인 숫자 예시:**

```python
# 문장: "The cat sat on the mat"에서 "sat"를 처리

Query(sat) = [0.2, 0.8, 0.1, ...]  # "앉다"라는 동작에 대한 검색 벡터

Key(The)   = [0.1, 0.1, 0.9, ...]
Key(cat)   = [0.9, 0.2, 0.1, ...]  # 주어와 관련
Key(sat)   = [0.2, 0.8, 0.1, ...]
Key(on)    = [0.1, 0.7, 0.3, ...]
Key(the)   = [0.1, 0.1, 0.9, ...]
Key(mat)   = [0.3, 0.6, 0.2, ...]  # 장소와 관련

# 1단계: Query와 각 Key의 유사도 계산 (내적)
similarity(Query, cat) = 0.2×0.9 + 0.8×0.2 + ... = 0.5  # 주어니까 중요!
similarity(Query, mat) = 0.2×0.3 + 0.8×0.6 + ... = 0.7  # 장소니까 더 중요!

# 2단계: Softmax로 확률로 변환
weights = [0.05, 0.25, 0.15, 0.08, 0.05, 0.42]  # mat이 가장 높음

# 3단계: Value들을 가중합
output = 0.05×Value(The) + 0.25×Value(cat) + ... + 0.42×Value(mat)
```

### 4-1. Scaled Dot-Product Attention

이제 위 개념을 코드로 구현합니다.

```python
def attention(query, key, value, mask=None, dropout=None):
    "Scaled Dot Product Attention을 계산합니다."
    d_k = query.size(-1)

    # 1단계: Query와 Key의 유사도 계산 (내적)
    # 예: query가 "앉다", key가 "고양이", "매트"일 때 각각 얼마나 관련있는지 계산
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 2단계: 마스킹 적용 (선택적)
    # 미래 단어를 보지 못하게 하거나, 패딩을 무시하기 위함
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3단계: Softmax로 확률로 변환
    # scores [2.1, 0.3, 3.5] → p_attn [0.25, 0.05, 0.70]
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    # 4단계: 확률 가중치로 Value를 가중합
    # 0.25×Value(cat) + 0.05×Value(on) + 0.70×Value(mat)
    return torch.matmul(p_attn, value), p_attn
```

> **설명:**
>
> **전체 프로세스:**
> 1. **유사도 계산**: Query와 Key를 내적하여 얼마나 관련있는지 점수 계산
> 2. **스케일링**: `sqrt(d_k)`로 나누는 이유는 벡터 차원이 클수록 내적 값이 커져서 softmax가 한쪽으로 치우치는 문제 방지
> 3. **Softmax**: 점수를 0~1 사이 확률로 변환 (전체 합은 1)
> 4. **가중합**: 확률을 가중치로 Value들을 섞어서 최종 출력 생성
>
> **왜 `sqrt(d_k)`로 나누나요?**
> - d_k=64일 때, 두 랜덤 벡터의 내적은 평균적으로 0, 분산은 64
> - 분산이 크면 softmax 결과가 [0.001, 0.001, 0.998]처럼 극단적으로 됨
> - `sqrt(64)=8`로 나누면 분산이 1이 되어 [0.2, 0.3, 0.5]처럼 부드러운 분포 유지
> - 부드러운 분포 → 여러 단어에 attention 분산 → 더 풍부한 정보 학습

### 4-2. Multi-Head Attention

Multi-head attention은 여러 개의 attention "head"를 사용하여 서로 다른 표현 하위 공간에서 정보를 병렬로 추출합니다.

**왜 Multi-Head가 필요한가요?**

하나의 attention만 사용하면 한 가지 관점만 볼 수 있습니다. 여러 head를 사용하면 다양한 관점에서 문장을 이해할 수 있습니다.

**일상 비유: 영화 감상**
- **Head 1 (스토리 전문가)**: 줄거리의 흐름에 집중
- **Head 2 (연기 평론가)**: 배우들의 연기에 집중
- **Head 3 (촬영 감독)**: 카메라 워크와 구도에 집중
- **Head 4 (음악 감독)**: 배경음악과 음향에 집중

각 전문가가 다른 관점에서 영화를 보고, 최종적으로 모든 의견을 종합하여 종합 평가를 내립니다.

**Transformer에서의 Multi-Head:**

```python
# 예: "The cat sat on the mat" 문장 처리

Head 1: 문법적 관계에 집중
  - "sat"와 주어 "cat"의 관계 강조
  - attention weights: cat(0.7), sat(0.2), mat(0.1)

Head 2: 의미적 관계에 집중
  - "sat"와 장소 "mat"의 관계 강조
  - attention weights: cat(0.2), sat(0.1), mat(0.7)

Head 3: 품사 관계에 집중
  - 동사와 전치사 "on"의 관계
  - attention weights: on(0.6), sat(0.3), the(0.1)

최종 출력: 세 Head의 결과를 합쳐서 풍부한 표현 생성
```

**구체적인 작동 방식:**

1. **입력 분할**: 512차원 벡터를 8개 head로 나누면 각 head는 64차원 처리
2. **병렬 처리**: 각 head가 독립적으로 attention 계산
3. **결합**: 8개 head의 출력을 concat하여 다시 512차원으로 복원

```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "모델 차원(d_model)과 head의 개수 h를 입력받습니다. (d_k = d_model // h)"
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # d_model은 h로 나누어 떨어져야 합니다.

        # 예: d_model=512, h=8 → d_k=64 (각 head가 처리할 차원)
        self.d_k = d_model // h
        self.h = h

        # query, key, value 변환용 3개 + 최종 출력용 1개 = 총 4개의 Linear layer
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 동일한 마스크를 모든 head에 적용
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Q, K, V를 각 head용으로 분할
        # (batch, seq_len, 512) → (batch, seq_len, 8, 64) → (batch, 8, seq_len, 64)
        # 이렇게 하면 8개 head가 동시에 독립적으로 attention 계산 가능
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) 모든 head에 대해 병렬로 attention 적용
        # 각 head는 64차원에서 독립적으로 attention 계산
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) head들을 다시 합치기 (concat)
        # (batch, 8, seq_len, 64) → (batch, seq_len, 8, 64) → (batch, seq_len, 512)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        # 4) 최종 선형 변환으로 출력 생성
        return self.linears[-1](x)
```

> **설명:**
>
> **전체 프로세스 (d_model=512, h=8 예시):**
> 1. **선형 변환 + 분할**:
>    - 입력 (batch, seq_len, 512)를 선형 변환
>    - 8개 head로 분할: (batch, 8, seq_len, 64)
> 2. **병렬 Attention**:
>    - 각 head가 64차원에서 독립적으로 attention 계산
>    - Head 1은 문법 관계, Head 2는 의미 관계 등 다른 패턴 학습
> 3. **Concatenation**:
>    - 8개 head 결과를 합침: (batch, seq_len, 512)
> 4. **최종 변환**:
>    - 선형 변환으로 정보 통합
>
> **핵심 아이디어**: 하나의 512차원 attention보다, 8개의 64차원 attention이 더 다양한 패턴을 학습할 수 있습니다!

---

## 5. Position-wise Feed-Forward Network

각 인코더/디코더 레이어에는 각 위치마다 동일하게 적용되는 두 개의 선형 변환과 ReLU 활성화 함수가 포함된 피드포워드 네트워크가 있습니다.

```python
class PositionwiseFeedForward(nn.Module):
    "두 개의 선형 레이어와 ReLU를 사용한 피드포워드 네트워크."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)  # 차원 확장 (예: 512 -> 2048)
        self.w_2 = nn.Linear(d_ff, d_model)  # 원래 차원으로 축소
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
```

> **설명:**  
> 이 네트워크는 각 단어의 hidden state에 대해 독립적으로 동일한 변환을 적용합니다.

---

## 6. 임베딩과 Positional Encoding

**왜 Positional Encoding이 필요한가요?**

Transformer는 RNN이나 CNN과 달리 단어를 한 번에 모두 처리합니다. 따라서 단어의 순서 정보가 없으면 "고양이가 쥐를 잡았다"와 "쥐가 고양이를 잡았다"를 구분할 수 없습니다.

**비유: 셔플된 사진 앨범**
```
원본 순서: [사진1: 아기] → [사진2: 어린이] → [사진3: 청년] → [사진4: 노인]
셔플 후:   [사진3], [사진1], [사진4], [사진2]

문제: 사진들을 동시에 보면 시간 순서를 알 수 없음
해결: 각 사진에 타임스탬프를 추가
     - 사진1: "1990년 출생"
     - 사진2: "2000년 10살"
     - 사진3: "2010년 20살"
     - 사진4: "2050년 60살"
```

Positional Encoding은 각 단어에 "위치 타임스탬프"를 추가하는 것입니다.

### 6-1. Embeddings

```python
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # 임베딩 값에 sqrt(d_model)을 곱하여 스케일을 맞춰줍니다.
        return self.lut(x) * math.sqrt(self.d_model)
```

### 6-2. Positional Encoding

```python
class PositionalEncoding(nn.Module):
    "위치 정보를 sine과 cosine 함수를 이용해 생성합니다."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # max_len x d_model 크기의 positional encoding 행렬 생성
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # [0, 1, 2, 3, ...]

        # 주파수를 로그 스케일로 분포시키기 위한 term
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))

        # 짝수 인덱스: sin 함수 사용
        pe[:, 0::2] = torch.sin(position * div_term)
        # 홀수 인덱스: cos 함수 사용
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # 배치 차원을 위해 확장
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 입력 임베딩에 positional encoding을 더한 후 드롭아웃 적용
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

> **설명:**
>
> **왜 Sin/Cos 함수를 사용하나요?**
>
> 1. **고정된 패턴**: 학습 없이도 위치 정보 제공
>    ```python
>    위치 0: [sin(0), cos(0), sin(0), cos(0), ...] = [0, 1, 0, 1, ...]
>    위치 1: [sin(1), cos(1), sin(1), cos(1), ...] = [0.84, 0.54, ...]
>    위치 2: [sin(2), cos(2), sin(2), cos(2), ...] = [0.91, -0.42, ...]
>    ```
>    각 위치마다 고유한 "지문(fingerprint)" 생성
>
> 2. **상대적 위치 학습**:
>    - Sin/Cos의 수학적 성질로 "k칸 떨어진 단어" 관계 쉽게 학습
>    - PE(pos+k)를 PE(pos)의 선형 결합으로 표현 가능
>
> 3. **외삽 가능**: 학습 때 본 적 없는 긴 문장도 처리 가능
>    - 학습: 최대 100 단어
>    - 테스트: 150 단어 → Sin/Cos 패턴이 자동으로 확장됨
>
> **구체적인 예시:**
> ```python
> # 문장: "The cat sat"
> 단어 임베딩:
> "The" → [0.2, 0.5, 0.1, ...]  # 임베딩만
> "cat" → [0.8, 0.3, 0.9, ...]
> "sat" → [0.1, 0.7, 0.4, ...]
>
> + Positional Encoding:
> 위치0 → [0.0, 1.0, 0.0, ...]
> 위치1 → [0.8, 0.5, 0.0, ...]
> 위치2 → [0.9, -0.4, 0.0, ...]
>
> = 최종:
> "The"(위치0) → [0.2, 1.5, 0.1, ...]
> "cat"(위치1) → [1.6, 0.8, 0.9, ...]
> "sat"(위치2) → [1.0, 0.3, 0.4, ...]
> ```
>
> 이제 모델은 같은 단어도 위치에 따라 다른 표현을 가지므로 순서를 구분할 수 있습니다!

---

## 7. 전체 모델 구성: `make_model`

지금까지 구성한 각 구성요소들을 하나의 Transformer 모델로 조립하는 함수입니다.

```python
def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "하이퍼파라미터를 받아 Transformer 모델을 생성합니다."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        # src 임베딩 + positional encoding을 순차적으로 적용
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        # tgt 임베딩 + positional encoding을 순차적으로 적용
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )
    
    # 모델의 모든 파라미터를 Xavier 초기화로 초기화합니다.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

# 간단한 예시 모델 (어휘 크기 10, layer 2개)
tmp_model = make_model(10, 10, N=2)
print(tmp_model)
```

> **설명:**  
> 이 함수는 하이퍼파라미터(레이어 수, 모델 차원, 피드포워드 차원, head 수 등)를 받아 Encoder, Decoder, 임베딩, Generator를 모두 조립한 최종 Transformer 모델을 생성합니다.

---

## 8. 학습 관련 구성 요소

### 8-1. Batch와 마스킹 처리

훈련 시, 입력과 타겟 문장을 배치로 묶고 패딩을 마스킹할 수 있도록 Batch 객체를 정의합니다.

```python
class Batch:
    "마스킹 정보를 포함한 데이터 배치를 처리합니다."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        # src의 패딩 위치는 mask 처리 (1: 실제 token, 0: 패딩)
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            # 디코더 입력: 마지막 토큰 제거
            self.trg = trg[:, :-1]
            # 타겟 정답: 첫 토큰 제거
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "패딩과 미래 단어들을 마스킹하는 표준 mask 생성"
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask
```

> **설명:**  
> 입력 문장의 패딩 위치를 마스킹하고, 디코더에서는 타겟 시퀀스의 미래 단어를 마스킹하는 방식으로 학습에 필요한 마스크를 생성합니다.

### 8-2. Label Smoothing

Label Smoothing은 모델이 너무 자신감(confident)하게 예측하지 않도록, 정답 분포를 약간 부드럽게 만드는 기법입니다.

```python
class LabelSmoothing(nn.Module):
    "Label Smoothing을 구현합니다."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing  # 정답 토큰에 할당할 확률
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)
```

> **설명:**  
> 한-hot 벡터 대신, 정답 토큰에는 높은 확률(confidence)을, 나머지에는 조금의 확률을 분배하여 KL-divergence loss를 계산합니다.

### 8-3. 학습률 스케줄러 (NoamOpt)

Transformer 논문에서는 학습 초기에는 학습률을 선형으로 증가시킨 후, 이후에는 역제곱근으로 감소시키는 스케줄러를 사용합니다.

```python
class NoamOpt:
    "Noam 학습률 스케줄러. 매 step마다 학습률을 업데이트합니다."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "파라미터 업데이트와 함께 학습률을 갱신합니다."
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step=None):
        "학습률을 업데이트하는 공식: lr = factor * (model_size^(-0.5) * min(step^(-0.5), step * warmup^(-1.5)))"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
```

> **설명:**  
> 학습 초반 warmup 단계 동안에는 학습률을 선형으로 증가시키고, 이후에는 역제곱근으로 감소시키는 방식을 통해 안정적인 학습을 유도합니다.

---

## 9. Greedy Decoding (추론)

훈련 후, 모델을 사용해 한 문장을 번역할 때는 **Greedy Decoding**을 사용하여, 각 시점마다 가장 확률이 높은 단어를 선택합니다.

```python
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    "Greedy 방식으로 문장을 생성합니다."
    # encoder의 출력을 메모리(memory)로 얻습니다.
    memory = model.encode(src, src_mask)
    # 디코더 입력을 시작 기호로 초기화 (배치 크기 1)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    
    for i in range(max_len - 1):
        # 현재까지 생성된 문장을 디코더에 입력하고, 다음 단어 확률을 계산합니다.
        out = model.decode(memory, src_mask, ys, 
                           subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        # 가장 확률 높은 단어 선택
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        # 생성된 단어를 ys에 추가
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

# 예시: 모델 평가 시
src = torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]])
src_mask = (src != 0).unsqueeze(-2)
result = greedy_decode(tmp_model, src, src_mask, max_len=10, start_symbol=1)
print(result)
```

> **설명:**  
> 이 함수는 encoder의 출력을 얻은 후, 초기 토큰부터 시작해서 최대 길이까지 순차적으로 가장 높은 확률을 가진 단어를 선택해 문장을 생성합니다.

---

## 10. 결론

지금까지 Transformer의 주요 구성요소—Encoder, Decoder, Attention 메커니즘, Feed-Forward 네트워크, 임베딩 및 Positional Encoding, 그리고 학습 및 추론 관련 구성요소—를 하나씩 살펴보고, 각 부분에 해당하는 PyTorch 코드를 추가하며 설명했습니다.

### 전체 흐름 요약

```
입력 → 임베딩 + 위치정보 → Encoder → Decoder → 출력

각 단계별 역할:
1. 임베딩: 단어를 벡터로 변환
2. 위치 인코딩: 단어 순서 정보 추가
3. Encoder: 입력 문장의 의미 파악 (Self-Attention + FFN)
4. Decoder: 한 단어씩 생성 (Self-Attention + Enc-Dec Attention + FFN)
5. Generator: 다음 단어 확률 계산
```

### 핵심 개념 정리

**Query, Key, Value (가장 중요!)**
- Query: "무엇을 찾고 싶은가?"
- Key: "나는 이런 정보를 가지고 있어"
- Value: "실제 정보 내용"
- 프로세스: Query로 관련있는 Key 찾기 → 해당 Value 가져오기

**Multi-Head Attention**
- 하나의 attention보다 여러 관점에서 보는 게 더 풍부
- 8개 head = 8명의 전문가가 각자 다른 관점에서 분석

**Positional Encoding**
- Transformer는 순서를 모름 → Sin/Cos로 위치 정보 추가
- 학습 없이 고정된 패턴 사용

**Masking**
- 미래 단어를 보지 못하게 함 (부정행위 방지)
- 학습과 추론의 일관성 유지

### 실용 팁

**성능 최적화:**
- Warmup 중요: 초기 학습률을 천천히 올려서 안정화
- Label Smoothing: 과적합 방지
- Dropout: Attention과 Residual 연결 모두에 적용
- Batch Size: GPU 메모리 허용하는 최대로

**흔한 오류 해결:**
1. Shape 오류: Mask 차원 확인 (특히 batch 차원)
2. NaN 발생: 학습률 낮추기, Warmup 늘리기
3. 느린 수렴: Positional encoding 제대로 추가했는지 확인
4. 메모리 부족: Gradient accumulation 사용

**디버깅 체크리스트:**
```python
# 1. Attention weights 확인
print(model.encoder.layers[0].self_attn.attn)  # 합이 1인지 확인

# 2. Gradient 확인
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")  # 너무 크거나 작으면 문제

# 3. 출력 분포 확인
probs = F.softmax(output, dim=-1)
print(probs.max())  # 0.99 이상이면 너무 confident (label smoothing 필요)
```

### 추가 학습 자료

**시각화 도구:**
- [BertViz](https://github.com/jessevig/bertviz): Attention 패턴 시각화
- [Tensor2Tensor Visualization](https://github.com/tensorflow/tensor2tensor): 모델 내부 동작 확인

**실전 구현:**
- [Hugging Face Transformers](https://huggingface.co/transformers/): 사전 학습된 모델 사용
- [fairseq](https://github.com/pytorch/fairseq): Facebook의 Seq2Seq 라이브러리

**논문 읽기:**
- 원 논문: "Attention is All You Need" (2017)
- 개선 버전: BERT, GPT, T5 등

## Colab PyTorch Code

```py
# Google Colab에서 실행하기 위해 matplotlib inline 모드를 활성화합니다.
%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
import numpy as np
import matplotlib.pyplot as plt

###############################
# 1. 전체 모델 구조: Encoder-Decoder 아키텍처
###############################

class EncoderDecoder(nn.Module):
    """
    표준 Encoder-Decoder 아키텍처.
    encoder, decoder, src와 tgt 임베딩, 그리고 최종 출력(softmax)을 위한 generator를 포함합니다.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "출력 임베딩을 선형 변환한 후 softmax를 적용하는 모듈입니다."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

###############################
# 2. 인코더 관련 구성 요소
###############################

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std  = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

###############################
# 3. 디코더 관련 구성 요소
###############################

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

###############################
# 4. Attention 메커니즘
###############################

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    p_attn = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

###############################
# 5. Position-wise Feed-Forward Network
###############################

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

###############################
# 6. 임베딩과 Positional Encoding
###############################

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

###############################
# 7. 전체 모델 구성: make_model
###############################

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

###############################
# 8. 학습 관련 구성 요소
###############################

class Batch:
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask

class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

###############################
# 9. Greedy Decoding (추론)
###############################

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, ys, 
                           subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

###############################
# 10. 예시: 모델 생성 및 Greedy Decoding 테스트
###############################

# 어휘 크기를 11로 설정하여 토큰 인덱스 0~10 사용 가능
tmp_model = make_model(11, 11, N=2)
print(tmp_model)

plt.figure(figsize=(5,5))
plt.imshow(subsequent_mask(20)[0])
plt.title("Subsequent Mask")
plt.show()

# 예시 입력 (토큰 1~10 사용)
src = torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]])
src_mask = (src != 0).unsqueeze(-2)
result = greedy_decode(tmp_model, src, src_mask, max_len=10, start_symbol=1)
print("Greedy Decoding 결과:", result)

```

