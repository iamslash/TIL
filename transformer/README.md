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

아래는 “The Annotated Transformer” 글의 주요 부분들을 단계별로 설명하면서, 관련된 PyTorch 코드를 조금씩 추가하는 방식의 설명입니다. 각 단계마다 Transformer의 구성요소와 그 역할, 그리고 해당 코드가 어떻게 구현되어 있는지 함께 살펴보겠습니다.

---

## 1. 전체 모델 구조: Encoder–Decoder 아키텍처

Transformer는 **Encoder**와 **Decoder**로 구성된 모델입니다.  
- **Encoder**는 입력 문장을 연속적인(hidden) 표현(z₁, …, zₙ)으로 인코딩합니다.  
- **Decoder**는 인코딩된 표현을 바탕으로, 한 번에 한 단어씩 출력(y₁, …, yₘ)을 생성합니다.

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
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder        # 입력 시퀀스를 처리하는 인코더
        self.decoder = decoder        # 인코딩된 정보를 기반으로 출력 시퀀스를 생성하는 디코더
        self.src_embed = src_embed    # 입력 단어를 임베딩하는 레이어
        self.tgt_embed = tgt_embed    # 출력 단어를 임베딩하는 레이어
        self.generator = generator    # 최종 선형 변환 및 softmax

    def forward(self, src, tgt, src_mask, tgt_mask):
        "마스킹된 src와 tgt 시퀀스를 입력받아 인코딩 후 디코딩합니다."
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
```

> **설명:**  
> 위 코드는 전체 모델의 뼈대를 구성합니다. `EncoderDecoder` 클래스는 인코더, 디코더, 임베딩, 그리고 최종 출력 생성기(generator)를 연결합니다. `forward` 메서드에서는 먼저 입력(src)을 임베딩하고 인코딩한 후, 디코더에서 tgt를 임베딩하여 최종 출력을 생성합니다.

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

```python
import numpy as np
import matplotlib.pyplot as plt

def subsequent_mask(size):
    "미래 단어들을 마스킹합니다. (upper-triangular matrix)"
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
> 이 마스크는 디코더의 자기 주의에서 각 단어가 자기 자신보다 이후에 위치한 단어들을 참조하지 못하도록 만듭니다. 학습 시 디코더가 올바른 auto-regressive(순차적) 생성을 하게끔 돕습니다.

---

## 4. Attention 메커니즘

Transformer의 핵심은 **Scaled Dot-Product Attention**입니다.  
- 주어진 query, key, value 텐서를 사용해 각 query마다 관련된 value들을 가중합하여 출력합니다.  
- dot-product 값이 커지는 문제를 완화하기 위해 \( \frac{1}{\sqrt{d_k}} \)로 스케일 조정합니다.

### 4-1. Scaled Dot-Product Attention

```python
def attention(query, key, value, mask=None, dropout=None):
    "Scaled Dot Product Attention을 계산합니다."
    d_k = query.size(-1)
    # query와 key의 내적 후 스케일 조정
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 마스크가 주어지면, 마스크가 0인 부분은 매우 작은 값(-1e9)으로 채웁니다.
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # softmax를 적용하여 가중치 생성
    p_attn = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    # attention 가중치를 value에 곱해 최종 출력 생성
    return torch.matmul(p_attn, value), p_attn
```

> **설명:**  
> 각 query에 대해 모든 key와의 내적을 수행한 후, softmax를 통해 중요도를 구하고 그 가중치로 value들을 합산합니다. 여기서 스케일링을 통해 gradient vanishing 문제를 완화합니다.

### 4-2. Multi-Head Attention

Multi-head attention은 여러 개의 attention “head”를 사용하여 서로 다른 표현 하위 공간에서 정보를 병렬로 추출합니다.

```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "모델 차원(d_model)과 head의 개수 h를 입력받습니다. (d_k = d_model // h)"
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # d_model은 h로 나누어 떨어져야 합니다.
        self.d_k = d_model // h
        self.h = h
        # query, key, value, 그리고 마지막 선형 변환을 위한 4개의 Linear layer 생성
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "그림 2를 참고하여 multi-head attention을 구현합니다."
        if mask is not None:
            # 동일한 마스크를 모든 head에 적용
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) 각 선형 변환을 적용한 후, (batch, h, seq_len, d_k) 형태로 변환합니다.
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        
        # 2) 모든 head에 대해 attention 적용
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3) head들을 concat한 후 최종 선형 변환
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
```

> **설명:**  
> Multi-head attention은 각 head마다 선형 변환을 수행하여 낮은 차원으로 변환한 후 attention을 적용하고, 마지막에 다시 concat하여 전체 차원으로 복원합니다. 이를 통해 모델이 다양한 관점에서 입력을 바라볼 수 있게 합니다.

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

Transformer는 RNN이나 CNN과 달리 순서를 인식하는 구조가 없으므로, 단어 임베딩에 **위치 정보(positional encoding)**를 추가해 단어 순서를 모델에 주입합니다.

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
        position = torch.arange(0, max_len).unsqueeze(1)
        # 주파수를 로그 스케일로 분포시키기 위한 term
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스
        pe = pe.unsqueeze(0)  # 배치 차원을 위해 확장
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # 입력 임베딩에 positional encoding을 더한 후 드롭아웃 적용
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

> **설명:**  
> 각 단어의 임베딩에 위치 정보를 더해주면, 모델은 단어 순서를 인지할 수 있습니다. sine과 cosine을 사용하는 이유는 모델이 학습하지 않아도 자연스럽게 일반화할 수 있는 패턴을 제공하기 때문입니다.

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

전체 코드의 흐름은 아래와 같이 요약할 수 있습니다:

1. **입력 임베딩**에 positional encoding을 더해 입력을 준비  
2. **Encoder**가 입력 시퀀스를 여러 레이어를 통해 인코딩  
3. **Decoder**는 이전에 생성된 단어들과 encoder의 출력을 바탕으로 다음 단어를 예측  
4. **Attention 메커니즘**을 통해 각 단어 간의 관계를 효과적으로 학습  
5. **학습** 시에는 마스킹, label smoothing, 그리고 특별한 학습률 스케줄러를 사용하여 안정적인 학습을 수행  
6. **Greedy Decoding** (혹은 Beam Search 등)을 통해 실제 번역이나 텍스트 생성 수행

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

