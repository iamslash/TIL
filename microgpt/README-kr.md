- [Abstract](#abstract)
- [Materials](#materials)
- [전체 구조 한눈에 보기](#전체-구조-한눈에-보기)
- [섹션 1: 준비](#섹션-1-준비)
  - [코드](#코드)
  - [주요 변수 상태](#주요-변수-상태)
  - [토크나이저: 인코딩 ↔ 디코딩](#토크나이저-인코딩--디코딩)
- [섹션 2: Autograd 엔진 (Value 클래스)](#섹션-2-autograd-엔진-value-클래스)
  - [핵심 개념](#핵심-개념)
- [섹션 3: 모델 파라미터 초기화](#섹션-3-모델-파라미터-초기화)
  - [하이퍼파라미터](#하이퍼파라미터)
  - [state\_dict: 모든 파라미터 행렬](#state_dict-모든-파라미터-행렬)
  - [params: 평탄화된 파라미터 리스트](#params-평탄화된-파라미터-리스트)
- [섹션 4: 모델 아키텍처 (gpt 함수)](#섹션-4-모델-아키텍처-gpt-함수)
  - [x 의 변환 흐름](#x-의-변환-흐름)
  - [KV 캐시: pos\_id 가 늘어날수록 keys 가 쌓인다](#kv-캐시-pos_id-가-늘어날수록-keys-가-쌓인다)
  - [유틸리티 함수 요약](#유틸리티-함수-요약)
- [섹션 5: 학습 루프 ← 핵심](#섹션-5-학습-루프--핵심)
  - [전체 구조](#전체-구조)
  - [변수 상태 추적: doc="emma", step=0](#변수-상태-추적-docemma-step0)
    - [토큰화 직후](#토큰화-직후)
    - [순전파: pos\_id 별 변수 변화](#순전파-pos_id-별-변수-변화)
    - [역전파: backward() 전후](#역전파-backward-전후)
    - [Adam 업데이트: step=0, params\[0\].grad=0.5 가정](#adam-업데이트-step0-params0grad05-가정)
    - [step=0 → step=1: m, v 의 누적 효과](#step0--step1-m-v-의-누적-효과)
- [섹션 6: 추론](#섹션-6-추론)
  - [학습 루프와의 차이](#학습-루프와의-차이)
  - [변수 상태 추적: 모델이 "lia" 를 생성하는 경우](#변수-상태-추적-모델이-lia-를-생성하는-경우)
  - [temperature 의 역할](#temperature-의-역할)
- [전체 데이터 흐름 요약](#전체-데이터-흐름-요약)
- [원논문 Transformer 와의 비교](#원논문-transformer-와의-비교)
  - [구조적 차이: Encoder-Decoder vs Decoder-Only](#구조적-차이-encoder-decoder-vs-decoder-only)
  - [세부 차이 대조표](#세부-차이-대조표)
  - [핵심 차이 3가지 설명](#핵심-차이-3가지-설명)
    - [1. LayerNorm → RMSNorm + Pre-norm 위치 변경](#1-layernorm--rmsnorm--pre-norm-위치-변경)
    - [2. Sinusoidal → Learned Positional Embedding](#2-sinusoidal--learned-positional-embedding)
    - [3. 규모 비교](#3-규모-비교)
  - [공통점 (핵심 알고리즘은 동일)](#공통점-핵심-알고리즘은-동일)


--------

# Abstract

- [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) | gist
- Andrej Karpathy 가 GPT 를 외부 라이브러리 없이 순수 Python 으로 구현한 코드다.
- **학습 데이터**: 영어 이름 ~32,000 개
- **학습 목표**: 이름 패턴을 학습하여 새로운 이름을 생성

> *"This file is the complete algorithm. Everything else is just efficiency."*

# Materials

- [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) | gist
- [chainrule/README-kr.md](../chainrule/README-kr.md) | 체인 룰 & Autograd 원리
- [micrograd/README-kr.md](../micrograd/README-kr.md) | Value 클래스 상세 해설
- [adamoptimizer/README-kr.md](../adamoptimizer/README-kr.md) | Adam optimizer 상세 해설

# 전체 구조 한눈에 보기

```
microgpt.py 의 6개 섹션:

┌─────────────────────────────────────────────────────────────┐
│  섹션 1: 준비          import, 데이터셋, 토크나이저             │
├─────────────────────────────────────────────────────────────┤
│  섹션 2: Autograd      Value 클래스 (자동 미분 엔진)            │
├─────────────────────────────────────────────────────────────┤
│  섹션 3: 파라미터 초기화  state_dict, params                   │
├─────────────────────────────────────────────────────────────┤
│  섹션 4: 모델 아키텍처   linear, softmax, rmsnorm, gpt()       │
├─────────────────────────────────────────────────────────────┤
│  섹션 5: 학습 루프       forward → backward → Adam 업데이트    │  ← 핵심
├─────────────────────────────────────────────────────────────┤
│  섹션 6: 추론            학습된 모델로 이름 생성                │
└─────────────────────────────────────────────────────────────┘
```

데이터 흐름:

```
"emma" (문자열)
  ↓ 토크나이저
[26, 4, 12, 12, 0, 26] (토큰 ID)
  ↓ 순전파 (gpt 함수)
logits [27개 점수] → probs [27개 확률]
  ↓ 손실 계산
loss (스칼라 Value)
  ↓ 역전파
params[i].grad (모든 파라미터의 기울기)
  ↓ Adam 업데이트
params[i].data (파라미터 값 갱신)
  ↓ 1000 번 반복
  ↓ 추론
"amelia", "liam", ... (생성된 이름)
```

---

# 섹션 1: 준비

## 코드

```python
import os, math, random
random.seed(42)

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)

uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
```

## 주요 변수 상태

```
docs       = ['emma', 'olivia', 'ava', ...]   # 32,033 개 이름, 무작위 순서
uchars     = ['a','b','c',...,'z']            # 26 개 고유 문자, 정렬됨
BOS        = 26                               # 특수 토큰 ID (시작/끝 신호)
vocab_size = 27                               # 26 문자 + 1 BOS
```

## 토크나이저: 인코딩 ↔ 디코딩

```python
uchars.index('e')   # → 4   (문자 → 숫자, 학습/추론 입력 시)
uchars[4]           # → 'e' (숫자 → 문자, 추론 출력 시)
```

"emma" 토큰화 예시:

```
문자:   BOS   e   m   m   a   BOS
ID:     26    4   12  12  0   26
```

---

# 섹션 2: Autograd 엔진 (Value 클래스)

## 핵심 개념

`Value` 는 숫자를 감싸서 자동 미분을 가능하게 하는 래퍼다.
모든 파라미터(`state_dict` 안의 숫자들)가 `Value` 로 만들어진다.

```python
Value(random.gauss(0, 0.08))
# data        = 0.032  (파라미터의 현재 값)
# grad        = 0      (∂loss/∂data, backward 후 채워짐)
# _local_grads         (이 연산의 국소 미분, forward 시 저장됨)
```

세 속성이 채워지는 타이밍:

| 속성 | 채워지는 시점 |
|------|-------------|
| `data` | 파라미터: 생성 시 / 중간 노드: forward 시 |
| `_local_grads` | forward 시 (연산 직후 즉시) |
| `grad` | backward 후 |

→ 상세 원리: [chainrule/README-kr.md](../chainrule/README-kr.md)
→ 코드 해설: [micrograd/README-kr.md](../micrograd/README-kr.md)

---

# 섹션 3: 모델 파라미터 초기화

## 하이퍼파라미터

```python
n_layer    = 1   # 트랜스포머 레이어 수
n_embd     = 16  # 임베딩 차원 (각 토큰을 표현하는 벡터 크기)
block_size = 16  # 최대 처리 토큰 수 (가장 긴 이름이 15 글자)
n_head     = 4   # 어텐션 헤드 수
head_dim   = 4   # n_embd // n_head
```

## state_dict: 모든 파라미터 행렬

| 키 | 크기 | 역할 | 파라미터 수 |
|----|------|------|-----------|
| `wte` | 27×16 | 토큰 임베딩: 각 토큰 ID → 16-dim 벡터 | 432 |
| `wpe` | 16×16 | 위치 임베딩: 각 위치 ID → 16-dim 벡터 | 256 |
| `lm_head` | 27×16 | 출력 헤드: 16-dim → 27개 logits | 432 |
| `layer0.attn_wq` | 16×16 | Query 가중치 | 256 |
| `layer0.attn_wk` | 16×16 | Key 가중치 | 256 |
| `layer0.attn_wv` | 16×16 | Value 가중치 | 256 |
| `layer0.attn_wo` | 16×16 | Output 가중치 | 256 |
| `layer0.mlp_fc1` | 64×16 | MLP 확장 (16→64) | 1,024 |
| `layer0.mlp_fc2` | 16×64 | MLP 축소 (64→16) | 1,024 |
| **합계** | | | **4,192** |

## params: 평탄화된 파라미터 리스트

```python
params = [p for mat in state_dict.values() for row in mat for p in row]
# len(params) = 4,192
# params[0] 과 state_dict['wte'][0][0] 은 같은 Value 객체 (복사본 아님)
```

Adam 은 `for i, p in enumerate(params)` 로 모든 파라미터를 일괄 업데이트한다.

---

# 섹션 4: 모델 아키텍처 (gpt 함수)

## x 의 변환 흐름

`gpt(token_id=26, pos_id=0, keys=[[]], values=[[]])` 호출 시:

| 단계 | 코드 | x 의 크기 | 설명 |
|------|------|----------|------|
| 토큰 임베딩 | `tok_emb = wte[26]` | 16-dim | 토큰 26 에 해당하는 행 조회 |
| 위치 임베딩 | `pos_emb = wpe[0]` | 16-dim | 위치 0 에 해당하는 행 조회 |
| 합산 | `x = tok_emb + pos_emb` | 16-dim | 토큰과 위치 정보를 합침 |
| 정규화 | `x = rmsnorm(x)` | 16-dim | 크기를 ~1 로 정규화 |
| **어텐션 블록** | | | |
| 잔차 저장 | `x_residual = x` | 16-dim | 잔차 연결용 원본 보관 |
| Q,K,V 생성 | `q = linear(x, attn_wq)` | 16-dim each | 검색 질의/키/값 벡터 |
| KV 캐시 | `keys[0].append(k)` | — | pos_id=0 → 길이 1 |
| 멀티헤드 | 헤드 4개 × 4-dim | → 16-dim | Q·K 내적 → softmax → V 가중합 |
| 출력 변환 | `x = linear(x_attn, attn_wo)` | 16-dim | |
| 잔차 연결 | `x = x + x_residual` | 16-dim | 원본 정보 보존 |
| **MLP 블록** | | | |
| 잔차 저장 | `x_residual = x` | 16-dim | |
| 확장 | `x = linear(x, mlp_fc1)` | **64-dim** | 표현 공간 4배 확장 |
| 활성화 | `x = [xi.relu() for xi in x]` | 64-dim | 음수 → 0 |
| 축소 | `x = linear(x, mlp_fc2)` | 16-dim | 다시 원래 크기로 |
| 잔차 연결 | `x = x + x_residual` | 16-dim | |
| **출력** | `logits = linear(x, lm_head)` | **27-dim** | 각 토큰의 원시 점수 |

## KV 캐시: pos_id 가 늘어날수록 keys 가 쌓인다

```
pos_id=0: keys[0] = [k₀]              길이 1  (자기 자신만 참조)
pos_id=1: keys[0] = [k₀, k₁]         길이 2  (이전 + 현재 참조)
pos_id=2: keys[0] = [k₀, k₁, k₂]     길이 3
pos_id=4: keys[0] = [k₀,k₁,k₂,k₃,k₄] 길이 5
```

위치 t 의 어텐션은 0~t 의 모든 이전 토큰을 참조한다. 이전 k, v 를 재계산하지
않고 캐시에서 꺼내 쓰는 것이 KV 캐시다.

## 유틸리티 함수 요약

```python
linear(x, w)    # 행렬-벡터 곱: x(nin-dim) × w(nout×nin) → nout-dim
softmax(logits) # 실수 벡터 → 확률 분포 (합=1), max 빼기로 overflow 방지
rmsnorm(x)      # 크기 정규화: x_i × (1/√(mean(x²)+ε))
```

---

# 섹션 5: 학습 루프 ← 핵심

## 전체 구조

```python
for step in range(1000):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []

    # ① 순전파
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)

    loss = (1 / n) * sum(losses)

    # ② 역전파
    loss.backward()

    # ③ Adam 업데이트
    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0
```

## 변수 상태 추적: doc="emma", step=0

### 토큰화 직후

```
tokens = [26, 4, 12, 12, 0, 26]
n      = 5
keys   = [[]]   # 빈 캐시
values = [[]]
losses = []
```

### 순전파: pos_id 별 변수 변화

| pos_id | token_id | target_id | keys[0] 길이 | losses 길이 | loss_t (초반 예시) |
|--------|----------|-----------|--------------|-------------|-------------------|
| 0 | 26 (BOS) | 4 ('e') | 1 | 1 | ≈ 3.3 |
| 1 | 4 ('e') | 12 ('m') | 2 | 2 | ≈ 3.3 |
| 2 | 12 ('m') | 12 ('m') | 3 | 3 | ≈ 3.3 |
| 3 | 12 ('m') | 0 ('a') | 4 | 4 | ≈ 3.3 |
| 4 | 0 ('a') | 26 (BOS) | 5 | 5 | ≈ 3.3 |

```
loss = (1/5) × sum(losses) ≈ 3.3
# 참고: 27개 토큰을 랜덤으로 찍으면 loss ≈ -log(1/27) = 3.30
# 즉 학습 초반은 랜덤 수준의 손실
```

**손실 함수**: `loss_t = -probs[target_id].log()`

```
probs[target] = 0.04  →  -log(0.04) = 3.22  (많이 틀림)
probs[target] = 0.50  →  -log(0.50) = 0.69
probs[target] = 0.99  →  -log(0.99) = 0.01  (거의 맞음)
```

정답 토큰에 높은 확률을 부여할수록 loss 가 낮아진다.

### 역전파: backward() 전후

```
loss.backward() 호출 전:
  params[0].grad = 0  ... params[4191].grad = 0

loss.backward() 호출 후:
  params[0].grad = 0.023   (예시)
  params[1].grad = -0.041
  ...
  params[4191].grad = 0.007
```

각 `.grad` 의 의미: "이 파라미터를 1 바꾸면 loss 가 얼마나 변하는가"

→ 역전파 원리: [chainrule/README-kr.md](../chainrule/README-kr.md)

### Adam 업데이트: step=0, params[0].grad=0.5 가정

```
lr_t  = 0.01 × (1 - 0/1000) = 0.01          # 학습률 최대

m[0] = 0.85×0.0 + 0.15×0.5  = 0.075         # 1차 모멘트 (방향)
v[0] = 0.99×0.0 + 0.01×0.25 = 0.0025        # 2차 모멘트 (변동성)

m_hat = 0.075  / (1 - 0.85¹) = 0.5          # 편향 보정
v_hat = 0.0025 / (1 - 0.99¹) = 0.25         # 편향 보정

params[0].data -= 0.01 × 0.5 / (√0.25 + 1e-8)
                = 0.01 × 1.0 = 0.01          # 파라미터 이동량

params[0].grad = 0                            # 기울기 리셋
```

### step=0 → step=1: m, v 의 누적 효과

step=1 에서 params[0].grad = 0.3 이라면:

```
           step=0   →   step=1
m[0]:      0.075        0.85×0.075 + 0.15×0.3  = 0.109
v[0]:      0.0025       0.99×0.0025 + 0.01×0.09 = 0.003

m 은 "이전 방향 85% + 새 방향 15%" 로 부드럽게 이동한다 (관성)
v 는 기울기의 크기를 추적하여 파라미터별 학습률을 조정한다 (적응)
```

→ Adam 상세: [adamoptimizer/README-kr.md](../adamoptimizer/README-kr.md)

---

# 섹션 6: 추론

## 학습 루프와의 차이

| | 학습 | 추론 |
|--|------|------|
| `token_id` 출처 | 정답 데이터 (`tokens[pos_id]`) | 이전 스텝의 모델 출력 |
| probs 계산 | `softmax(logits)` | `softmax(logits / temperature)` |
| 다음 토큰 선택 | 없음 (정답이 주어짐) | `random.choices(weights=probs)` |
| 종료 조건 | pos_id 가 n 에 도달 | BOS 토큰 생성 시 |
| backward | 있음 | **없음** (파라미터 고정) |

## 변수 상태 추적: 모델이 "lia" 를 생성하는 경우

```
token_id = 26 (BOS)
sample   = []
```

| pos_id | token_id (입력) | keys[0] 길이 | probs 에서 선택 | sample |
|--------|-----------------|--------------|----------------|--------|
| 0 | 26 (BOS) | 1 | 11 ('l') | `['l']` |
| 1 | 11 ('l') | 2 | 8 ('i') | `['l','i']` |
| 2 | 8 ('i') | 3 | 0 ('a') | `['l','i','a']` |
| 3 | 0 ('a') | 4 | 26 (BOS) → **break** | `['l','i','a']` |

```python
print(''.join(sample))  # → "lia"
```

## temperature 의 역할

```
temperature=0.1: 확률 분포가 뾰족 → 가장 가능성 높은 글자만 선택 (보수적)
temperature=0.5: 균형 (microgpt 기본값)
temperature=1.0: 원래 확률 분포 그대로
temperature=2.0: 분포가 평탄 → 창의적이지만 엉뚱한 선택 가능
```

---

# 전체 데이터 흐름 요약

```
[준비]
  input.txt → docs(32,033) → uchars(26개) → vocab_size(27)

[파라미터 초기화]
  state_dict (9개 행렬) → params (4,192개 Value)
  m = [0.0 × 4,192],  v = [0.0 × 4,192]

[학습 루프, 1000 번]
  doc → tokens(n개)
    ↓
    pos_id=0..n-1 반복
      gpt() → token(16-dim) + pos(16-dim) → x(16-dim)
            → Attention: KV 캐시 누적, Q·K softmax V 가중합
            → MLP: 16→64→relu→16
            → lm_head: 16→logits(27-dim)
      loss_t = -log(probs[target_id])
    ↓
    loss = mean(losses)
    loss.backward()      → params[i].grad 채워짐
    Adam 업데이트         → m[i], v[i] 누적, params[i].data 갱신
    params[i].grad = 0    → 다음 스텝 준비

[추론]
  token_id=BOS 에서 시작
  gpt() → softmax(logits/T) → random.choices → 다음 token_id
  BOS 나오면 종료 → 새로운 이름 출력
```

---

# 원논문 Transformer 와의 비교

원논문: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)

## 구조적 차이: Encoder-Decoder vs Decoder-Only

원논문은 번역(seq2seq) 을 위한 **Encoder + Decoder** 구조다.
microgpt 는 텍스트 생성을 위한 **Decoder-Only** 구조다.

```
원논문 Transformer:                   microgpt (GPT 계열):

 Input                                 Input
   ↓                                     ↓
[Encoder] × 6                         [Decoder] × 1
  Self-Attention                         Masked Self-Attention
  FFN                                    FFN
   ↓                                     ↓
[Decoder] × 6                          lm_head → logits
  Masked Self-Attention
  Cross-Attention  ← Encoder 참조
  FFN
   ↓
 Output
```

Cross-Attention (Encoder 출력을 Decoder 가 참조) 이 microgpt 에는 없다.

## Decoder 만으로도 충분한 이유

### Encoder 가 존재하는 이유

원논문의 Encoder-Decoder 는 **두 개의 분리된 시퀀스**를 다루기 위해 만들어졌다.

```
번역 예시:
  "나는 밥을 먹었다"  ← 입력 (한국어, Encoder 가 처리)
          ↓
  "I ate rice"        ← 출력 (영어, Decoder 가 생성)
```

Encoder 의 역할:
- 입력 전체를 **양방향**으로 모두 보고 이해한다
- "나는/밥을/먹었다" 세 단어가 서로 어떻게 연관되는지 파악한다
- 이 이해를 압축하여 Cross-Attention 을 통해 Decoder 에 전달한다

### 생성 문제에는 별도 Encoder 가 필요 없다

**언어 생성 문제는 두 개의 시퀀스가 없다.** 입력과 출력이 같은 시퀀스 위에 있다.

```
이름 생성 예시:
  BOS → 'e' → 'm' → 'm' → 'a' → BOS
   ↑      ↑      ↑      ↑      ↑
  주어짐  예측   예측   예측   예측
```

각 위치에서 모델이 하는 일: "지금까지 본 토큰들을 보고 다음 토큰을 예측하라"

이것은 **과거 토큰들을 이해하는 것** (= Encoding) 과 **다음 토큰을 생성하는 것**
(= Decoding) 이 동시에 일어난다. 별도 Encoder 없이도 Self-Attention 이 두 역할을
모두 수행한다.

```
별도 Encoder 없이도:
  Causal Self-Attention 이 이전 토큰들을 "이해"하고    ← Encoding 역할
  lm_head 가 다음 토큰을 "생성"한다                    ← Decoding 역할
  → 한 스택이 두 역할을 모두 수행
```

### 결정적 차이: 어텐션 방향

```
Encoder Self-Attention (양방향):
  "밥을" 이 "나는", "먹었다" 를 모두 볼 수 있음 → 전체 문맥 이해에 유리

Decoder Causal Attention (단방향):
  "m" 은 "BOS", "e" 만 보고 미래 "m", "a" 는 못 봄 → 생성 시 치팅 방지
```

언어 생성에서는 **단방향이 오히려 필수**다. 미래를 보고 예측하면 학습이 의미없다.

### 현대 LLM 은 모두 Decoder-Only

GPT, LLaMA, Claude 등 현대 대형 언어 모델은 모두 Decoder-Only 다. 번역/요약처럼
"입력 전체를 먼저 이해해야 하는 태스크" 도 Decoder-Only 모델이 충분히 잘 처리한다.
시퀀스 앞에 프롬프트를 붙이면, Decoder 가 그 앞 내용 전체를 Self-Attention 으로
참조하여 사실상 Encoding 역할도 수행하기 때문이다.

```
프롬프트 방식:
  "Translate to English: 나는 밥을 먹었다 → "
  ↑────────────────────────────────────────↑
  Decoder 가 이 전체를 Causal Attention 으로 참조
  → 별도 Encoder 없이도 번역 가능
```

## 세부 차이 대조표

| 항목 | 원논문 (1706.03762) | microgpt.py | 비고 |
|------|--------------------|-|------|
| **구조** | Encoder-Decoder | Decoder-Only | GPT 계열의 핵심 차이 |
| **레이어 수 (N)** | 6 | 1 | |
| **임베딩 차원 (d_model)** | 512 | 16 | |
| **FFN 내부 차원 (d_ff)** | 2048 (×4) | 64 (×4) | 비율은 동일 |
| **어텐션 헤드 수 (h)** | 8 | 4 | |
| **헤드 차원 (d_k)** | 64 | 4 | |
| **위치 인코딩** | 사인파 공식 (고정) | **학습 가능한 `wpe`** | GPT-2 방식 |
| **정규화 방식** | **LayerNorm** (평균+분산) | **RMSNorm** (분산만) | LLaMA 방식 |
| **정규화 위치** | Post-norm | **Pre-norm** | GPT-2 이후 표준 |
| **활성화 함수** | ReLU | ReLU | 동일 |
| **바이어스** | 있음 | **없음** | |
| **Dropout** | 0.1 | **없음** | 극소 모델이라 생략 |
| **Cross-Attention** | 있음 | **없음** | Decoder-Only 이므로 |
| **파라미터 수** | ~65M | ~4,192 | 약 15,000 배 차이 |

## 핵심 차이 3가지 설명

### 1. LayerNorm → RMSNorm + Pre-norm 위치 변경

```
원논문 (Post-norm):   x → Sublayer(x) → + → LayerNorm → 출력
microgpt (Pre-norm):  x → RMSNorm → Sublayer(x) → + → 출력
```

Pre-norm 은 GPT-2 이후 표준이 됐다. 깊은 네트워크에서 학습이 더 안정적이다.
RMSNorm 은 평균을 빼지 않아 더 단순하면서 성능은 비슷하다 (LLaMA 등에서 채택).

```python
# 원논문: LayerNorm(x + Sublayer(x))
# microgpt:
x_residual = x
x = rmsnorm(x)           # 먼저 정규화
x = linear(x, ...)       # Sublayer
x = [a + b for ...]      # 잔차 연결은 나중에
```

### 2. Sinusoidal → Learned Positional Embedding

```
원논문: 수식으로 고정된 위치 인코딩
  PE(pos, 2i)   = sin(pos / 10000^(2i/512))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/512))

microgpt: 파라미터로 학습
  wpe = matrix(block_size, n_embd)  # 그냥 학습 가능한 행렬
  pos_emb = state_dict['wpe'][pos_id]
```

학습 가능한 방식이 특정 도메인(이름 생성 등)에서 더 잘 적응한다.

### 3. 규모 비교

```
             N     d_model  heads  params
원논문 base:  6     512      8      65M
GPT-2 small: 12    768      12     117M
GPT-3:       96    12,288   96     175B
microgpt:     1     16       4     4,192
```

## 공통점 (핵심 알고리즘은 동일)

원논문에서 제안한 핵심 아이디어는 microgpt 에도 그대로 살아있다.

| 공통 요소 | 수식 |
|----------|------|
| Scaled Dot-Product Attention | `softmax(QKᵀ / √d_k) × V` |
| Multi-Head Attention | h 개 헤드로 병렬 어텐션 후 concat |
| Position-wise FFN | Linear → ReLU → Linear (확장 비율 ×4) |
| Residual Connection | `x = x + Sublayer(x)` |
| Causal Masking | 미래 토큰을 참조하지 않음 |

Karpathy 가 *"Everything else is just efficiency"* 라고 한 이유가 바로 이것이다.
규모와 몇 가지 후속 개선(Pre-norm, RMSNorm, Learned PE)이 다를 뿐,
**핵심 알고리즘은 원논문과 동일하다.**
