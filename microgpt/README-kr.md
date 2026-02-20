# Abstract

- [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) |
  gist
- Andrej Karpathy 가 GPT(Generative Pre-trained Transformer) 를 외부 라이브러리
  없이 순수 Python 만으로 구현한 코드이다.
- PyTorch, TensorFlow 같은 딥러닝 프레임워크를 전혀 사용하지 않고, 자동
  미분(Autograd), 트랜스포머 아키텍처, 학습 루프, 텍스트 생성까지 모두 밑바닥부터
  만들었다.

> *"This file is the complete algorithm. Everything else is just efficiency."*
> (이 파일이 알고리즘의 전부다. 나머지는 모두 효율성의 문제일 뿐이다.)

- **학습 데이터**: 사람 이름 목록 (~32,000 개)
- **학습 목표**: 이름의 패턴을 학습하여 그럴듯한 새 이름을 생성

# Materials

- [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) |
  gist

# 전체 코드

```python
"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.

This file is the complete algorithm.

Everything else is just efficiency.

@karpathy
"""

import os
import math
import random

random.seed(42)

if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head

matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}

for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)

        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)

        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]

        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)

num_steps = 1000

for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []

    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)

    loss = (1 / n) * sum(losses)

    loss.backward()

    lr_t = learning_rate * (1 - step / num_steps)

    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

temperature = 0.5

print("\n--- inference (new, hallucinated names) ---")

for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []

    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]

        if token_id == BOS:
            break

        sample.append(uchars[token_id])

    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

# 섹션별 상세 해설

## 섹션 1: 임포트 및 랜덤 시드

```python
import os     # os.path.exists
import math   # math.log, math.exp
import random # random.seed, random.choices, random.gauss, random.shuffle

random.seed(42) # Let there be order among chaos
```

- `os`: 파일 존재 여부 확인용
- `math`: 수학 함수 (로그, 지수)용
- `random`: 난수 생성, 셔플, 가우시안 분포 등
- `random.seed(42)`: 랜덤 시드를 고정한다. 이렇게 하면 코드를 몇 번을 실행해도
  동일한 결과가 나온다. 실험의 재현성을 보장하기 위한 관행이다. 42 는 "은하수를
  여행하는 히치하이커를 위한 안내서"에서 유래한, 프로그래머들 사이에서 관습적으로
  쓰이는 숫자이다.

주석 "Let there be order among chaos"(혼돈 속에 질서가 있으라)는 Karpathy 특유의
유머이다. 랜덤이라는 혼돈에 시드라는 질서를 부여한다는 뜻이다.

## 섹션 2: 데이터셋 로딩

```python
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")
```

학습에 사용할 이름 데이터를 준비한다.

1. `input.txt` 파일이 로컬에 없으면 GitHub 에서 다운로드한다.
2. 파일을 한 줄씩 읽어서, 공백을 제거(`strip()`)하고, 빈 줄은 건너뛴다.
3. 데이터를 무작위로 섞는다 (`shuffle`). 학습 시 데이터 순서에 의한 편향을 방지하기
   위함이다.

`docs` 의 내용 예시: `['emma', 'olivia', 'ava', 'sophia', ...]` -- 약 32,000 개의
영어 이름

여기서 "document" 라는 표현을 쓰지만, 실제로는 한 줄짜리 이름 문자열이다. GPT 라는
범용 구조에 맞춰 "문서"라고 부르는 것이며, 이름 하나하나가 모델이 학습할 하나의
시퀀스(sequence)가 된다.

## 섹션 3: 토크나이저 (Tokenizer)

```python
uchars = sorted(set(''.join(docs)))  # unique characters
BOS = len(uchars)                     # Beginning of Sequence token
vocab_size = len(uchars) + 1          # total unique tokens
print(f"vocab size: {vocab_size}")
```

문자(글자)를 숫자로, 숫자를 문자로 변환하는 규칙을 만든다.

**왜 필요한가?** 컴퓨터(신경망)는 문자를 직접 처리할 수 없다. 숫자만 다룰 수 있기
때문에, 각 문자에 고유한 번호를 부여해야 한다.

동작 상세:

1. `''.join(docs)`: 모든 이름을 하나의 긴 문자열로 합친다. 예:
   `"emmaoliviaava..."`
2. `set(...)`: 중복을 제거하여 고유 문자만 남긴다. 예: `{'a', 'b', 'c', ..., 'z'}`
3. `sorted(...)`: 알파벳순으로 정렬한다. 이렇게 하면 `a=0, b=1, c=2, ...` 같은
   안정적인 매핑이 만들어진다.

`uchars` 결과 예시: `['a', 'b', 'c', ..., 'z']` (26 개 문자)

**BOS 토큰**:

- `BOS = 26` (알파벳 26 개 뒤의 번호)
- BOS 는 "Beginning of Sequence"의 약자로, 시퀀스의 시작을 나타내는 특수 토큰이다.
- 비유: 책의 표지 같은 역할이다. "지금부터 새 이름이 시작됩니다"라는 신호이다.
- 이 코드에서는 BOS 가 시퀀스의 끝(EOS)을 나타내는 데에도 함께 쓰인다. 즉, 하나의
  특수 토큰이 시작과 끝 모두를 표시한다.

`vocab_size = 27`: 알파벳 26 개 + BOS 1 개 = 총 27 개의 토큰

## 섹션 4: 자동 미분 엔진 (Autograd) -- Value 클래스

이 섹션이 코드에서 가장 핵심적이고 정교한 부분이다. 신경망 학습의 근본 메커니즘인
**역전파(backpropagation)** 를 구현한다.

### 자동 미분이란?

신경망 학습은 다음 과정의 반복이다:

1. **순전파(Forward)**: 입력을 넣고 출력을 계산한다.
2. **손실 계산**: 출력이 정답에서 얼마나 틀렸는지 측정한다.
3. **역전파(Backward)**: 각 파라미터가 손실에 얼마나 기여했는지 계산한다 (=
   기울기/gradient).
4. **파라미터 업데이트**: 기울기 방향으로 파라미터를 조정한다.

3 번 단계에서 "각 파라미터의 기울기"를 자동으로 계산해주는 것이 자동
미분(Autograd)이다.

**비유**: 요리에서 "짠맛이 너무 강하다"는 결과(손실)가 나왔을 때, 소금을 줄이면
얼마나 덜 짜질지, 물을 더 넣으면 얼마나 덜 짜질지를 각 재료별로 계산하는 것과
같다. 자동 미분은 이 계산을 모든 재료(파라미터)에 대해 자동으로 해준다.

### Value 클래스 기본 구조

```python
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data          # 실제 숫자값
        self.grad = 0             # 이 값의 기울기 (역전파 시 계산됨)
        self._children = children     # 이 값을 만든 입력값들
        self._local_grads = local_grads  # 각 입력에 대한 국소 미분값
```

`Value` 는 단순한 숫자가 아니다. 숫자이면서 동시에 **"어떤 계산을 통해
만들어졌는지"를 기억하는 숫자**이다.

각 속성의 의미:

- `data`: 실제 숫자 값. 예: `3.14`
- `grad`: 최종 손실(loss)에 대한 이 값의 기울기. "이 값이 조금 변하면 최종 손실이
  얼마나 변하는가?"의 답.
- `_children`: 이 값을 만들어낸 부모 값들. 계산 그래프의 간선(edge).
- `_local_grads`: 각 부모에 대한 국소 미분. 체인 룰 적용 시 사용.

`__slots__`: Python 최적화 기법이다. 일반적으로 Python 객체는 `__dict__` 라는
딕셔너리에 속성을 저장하는데, `__slots__` 를 지정하면 고정된 속성만 허용하여
메모리를 절약한다. 수십만 개의 `Value` 객체가 생성되므로 이 최적화가 중요하다.

**비유**: `Value` 는 "영수증이 달린 숫자"이다. 숫자 자체의 값(`data`)뿐 아니라, 그
숫자가 어떤 재료(`_children`)로 어떤 조리법(`_local_grads`)을 통해 만들어졌는지를
항상 기록한다.

### 산술 연산 -- 계산 그래프 구축

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data + other.data, (self, other), (1, 1))

def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data * other.data, (self, other), (other.data, self.data))
```

**덧셈 (`__add__`)**:

- `c = a + b` 를 실행하면, 새로운 `Value(a.data + b.data)` 가 만들어진다.
- `children = (a, b)`: c 는 a 와 b 로부터 만들어졌다.
- `local_grads = (1, 1)`: 덧셈의 미분은 양쪽 다 1 이다.
  - 직관: `c = a + b` 에서 a 가 1 늘어나면 c 도 정확히 1 늘어난다. b 도 마찬가지.

**곱셈 (`__mul__`)**:

- `c = a * b` 를 실행하면, 새로운 `Value(a.data * b.data)` 가 만들어진다.
- `local_grads = (b.data, a.data)`: 곱셈의 미분 규칙.
  - 직관: `c = a * b` 에서 a 가 조금 늘어나면 c 는 b 만큼 늘어난다. 반대도
    마찬가지.
  - 예: `c = 3 * 5 = 15` 에서, a(=3)가 1 늘어나면 c 는 5 늘어남. b(=5)가 1
    늘어나면 c 는 3 늘어남.

### 기타 수학 연산

```python
def __pow__(self, other):
    return Value(self.data**other, (self,), (other * self.data**(other-1),))

def log(self):
    return Value(math.log(self.data), (self,), (1/self.data,))

def exp(self):
    return Value(math.exp(self.data), (self,), (math.exp(self.data),))

def relu(self):
    return Value(max(0, self.data), (self,), (float(self.data > 0),))
```

각 연산과 그 미분(local_grad):

| 연산 | 수식 | 미분 | 직관 |
|------|------|------|------|
| `__pow__` | x^n | n * x^(n-1) | 거듭제곱의 미분 규칙 |
| `log` | ln(x) | 1/x | x 가 클수록 log 의 변화가 작아짐 |
| `exp` | e^x | e^x | 지수함수는 미분해도 자기 자신 |
| `relu` | max(0, x) | x>0 이면 1, 아니면 0 | 양수면 그대로 통과, 음수면 차단 |

**ReLU 비유**: 수도꼭지 같은 것이다. 물(값)이 양수 방향으로 흐르면 그대로
통과시키고, 음수 방향이면 완전히 막아버린다. 신경망에서 비선형성을 도입하는 역할을
한다. 비선형성이 없으면 아무리 층을 쌓아도 결국 하나의 선형 변환과 같아져서, 복잡한
패턴을 학습할 수 없다.

### 보조 연산자

```python
def __neg__(self): return self * -1
def __radd__(self, other): return self + other
def __sub__(self, other): return self + (-other)
def __rsub__(self, other): return other + (-self)
def __rmul__(self, other): return self * other
def __truediv__(self, other): return self * other**-1
def __rtruediv__(self, other): return other * self**-1
```

이들은 Python 의 연산자 오버로딩이다. 기존에 정의한 `__add__`, `__mul__`,
`__pow__` 를 조합하여 뺄셈, 나눗셈 등을 구현한다.

- `__neg__`: `-a` 는 `a * -1` 로 처리
- `__radd__`: `5 + Value(3)` 같은 경우를 처리 (왼쪽이 일반 숫자일 때)
- `__sub__`: 뺄셈은 "더하기 + 음수"로 처리
- `__truediv__`: 나눗셈은 "곱하기 + 역수(^-1)"로 처리

이렇게 하면 모든 복합 연산이 자동으로 계산 그래프에 기록된다.

### 역전파 (backward)

```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._children:
                build_topo(child)
            topo.append(v)
    build_topo(self)
    self.grad = 1
    for v in reversed(topo):
        for child, local_grad in zip(v._children, v._local_grads):
            child.grad += local_grad * v.grad
```

이 메서드가 자동 미분의 핵심이다. 두 단계로 동작한다.

**1 단계: 위상 정렬 (Topological Sort)**

```python
topo = []
visited = set()
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._children:
            build_topo(child)
        topo.append(v)
build_topo(self)
```

계산 그래프를 "의존성 순서"로 정렬한다. 자식(입력)이 부모(출력)보다 먼저 오도록
배치한다.

**비유**: 요리 레시피의 순서를 정리하는 것이다. "밀가루 반죽 -> 소스 만들기 -> 피자
굽기" 순서에서, 역전파 시에는 거꾸로 "피자 -> 소스 -> 밀가루" 순으로 기울기를
전파한다.

**2 단계: 기울기 전파 (Chain Rule)**

```python
self.grad = 1  # 출발점: 손실의 손실에 대한 기울기는 1
for v in reversed(topo):  # 출력에서 입력 방향으로 역순 순회
    for child, local_grad in zip(v._children, v._local_grads):
        child.grad += local_grad * v.grad
```

- `self.grad = 1`: 최종 출력(loss)의 자기 자신에 대한 기울기는 항상 1 이다.
- `reversed(topo)`: 출력에서 입력 방향으로 거꾸로 간다.
- `child.grad += local_grad * v.grad`: **체인 룰(chain rule)** 이다.

**체인 룰 비유**: 도미노를 상상하라. 마지막 도미노(loss)가 넘어진 충격이 이전
도미노에게 전달되는데, 각 연결 지점에서 충격의 크기가 `local_grad` 만큼 곱해져서
전달된다. `+=` 를 쓰는 이유는 하나의 값이 여러 경로를 통해 최종 결과에 기여할 수
있기 때문이다. 각 경로의 기울기를 모두 합산한다.

## 섹션 5: 하이퍼파라미터와 모델 파라미터 초기화

### 하이퍼파라미터

```python
n_layer = 1       # 트랜스포머 레이어 수
n_embd = 16       # 임베딩 차원 (각 토큰을 표현하는 벡터의 크기)
block_size = 16   # 최대 시퀀스 길이 (한 번에 처리할 수 있는 토큰 수)
n_head = 4        # 어텐션 헤드 수
head_dim = n_embd // n_head  # = 4 (각 헤드가 담당하는 차원 수)
```

**하이퍼파라미터란?** 모델의 "설계도 치수"이다. 학습 과정에서 변하지 않고, 사람이
미리 정해놓는 값이다.

| 파라미터 | 값 | 의미 | 비유 |
|---------|-----|------|------|
| `n_layer` | 1 | 트랜스포머 블록 개수 | 건물의 층 수 |
| `n_embd` | 16 | 임베딩 벡터 크기 | 각 문자를 설명하는 형용사 16 개 |
| `block_size` | 16 | 최대 문맥 길이 | 한 번에 볼 수 있는 글자 수 |
| `n_head` | 4 | 어텐션 헤드 수 | 동시에 다른 관점에서 보는 눈 4 개 |
| `head_dim` | 4 | 각 헤드의 차원 | 각 눈이 보는 특성 4 개 |

참고로 실제 GPT-3 는 `n_layer=96, n_embd=12288, n_head=96` 이다. 이 코드는 학습
가능성을 보여주기 위한 극소 규모 모델이다.

### 파라미터 행렬 생성

```python
matrix = lambda nout, nin, std=0.08: [
    [Value(random.gauss(0, std)) for _ in range(nin)]
    for _ in range(nout)
]
```

이 함수는 `nout x nin` 크기의 2 차원 행렬을 만든다. 각 원소는 평균 0, 표준편차
0.08 인 가우시안 분포에서 뽑은 난수로 초기화된 `Value` 객체이다.

`Value` 는 단순한 `float` 를 감싸서 자동 미분을 가능하게 하는 래퍼이다.
`random.gauss(0, 0.08)` 이 반환하는 `float` (예: `0.032`)를 `Value(0.032)` 로
감싸면, 이후 이 값이 참여하는 모든 연산이 계산 그래프에 기록되어 `.backward()` 시
gradient 를 자동으로 받을 수 있다. PyTorch 의 `torch.Tensor` 가 하는 일을 스칼라
단위로 순수 파이썬으로 구현한 것이다.

**왜 작은 난수로 초기화하는가?** 0 으로 초기화하면 모든 뉴런이 동일하게 작동하여
학습이 안 된다. 너무 큰 값이면 학습이 불안정하다. 작은 난수는 "각 뉴런에게 서로
다른 출발점을 주되, 안전한 범위 내에서" 시작하게 한다.

**왜 0.08 인가?** Xavier/He 초기화에서 표준편차로 흔히 쓰는 공식은 `1/√n_in`
이다. 이 모델에서 대부분의 행렬은 `n_in = n_embd = 16` 이므로 `1/√16 = 0.25`
이다. 0.08 은 이보다 약간 작은 **보수적인 초기화** 값으로, 이 작은 모델에서
경험적으로 잘 동작하는 상수를 선택한 것이다. GPT-2 원 논문에서도 `0.02` 를
사용했으며, 모델 크기에 따라 적절한 상수를 고르는 것이 일반적이다.

### state_dict 구성

```python
state_dict = {
    'wte': matrix(vocab_size, n_embd),   # 토큰 임베딩: 27 x 16
    'wpe': matrix(block_size, n_embd),   # 위치 임베딩: 16 x 16
    'lm_head': matrix(vocab_size, n_embd) # 출력 헤드: 27 x 16
}

for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)  # Query: 16x16
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)  # Key: 16x16
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)  # Value: 16x16
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)  # Output: 16x16
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4*n_embd, n_embd) # MLP 확장: 64x16
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4*n_embd) # MLP 축소: 16x64
```

`state_dict` 는 모델의 모든 학습 가능한 파라미터(가중치)를 담는 딕셔너리이다.
PyTorch 의 `model.state_dict()` 와 동일한 개념이다.

각 파라미터의 역할:

| 파라미터 | 크기 | 역할 |
|---------|------|------|
| `wte` | 27x16 | **Word Token Embedding**: 각 문자(토큰)를 16 차원 벡터로 변환 |
| `wpe` | 16x16 | **Word Position Embedding**: 각 위치를 16 차원 벡터로 변환 |
| `lm_head` | 27x16 | **Language Model Head**: 16 차원 벡터를 27 개 토큰 점수로 변환 |
| `attn_wq` | 16x16 | **Weight for Query**: "내가 찾고 싶은 것" 벡터 생성 |
| `attn_wk` | 16x16 | **Weight for Key**: "내가 제공하는 것" 벡터 생성 |
| `attn_wv` | 16x16 | **Weight for Value**: "실제로 전달할 정보" 벡터 생성 |
| `attn_wo` | 16x16 | **Weight for Output**: 모든 헤드의 결과를 합침 |
| `mlp_fc1` | 64x16 | **MLP 확장**: 16 차원을 64 차원으로 확장 |
| `mlp_fc2` | 16x64 | **MLP 축소**: 64 차원을 다시 16 차원으로 축소 |

**`wte`/`wpe` vs `lm_head` — 사용 방식의 차이**:

`matrix(nout, nin)` 에서 `nout`/`nin` 이라는 이름은 `linear()` 에서 행렬곱으로
사용할 때만 의미적으로 맞다. `wte` 와 `wpe` 는 행렬곱이 아니라 **행 인덱싱
(lookup)** 으로 사용된다.

| 행렬 | 사용 방식 | 첫 번째 차원의 의미 |
|---------|-------------|---------------------|
| `wte` | `wte[token_id]` — 행 하나를 꺼냄 | 토큰 개수 (27) |
| `wpe` | `wpe[pos_id]` — 행 하나를 꺼냄 | 위치 개수 (16) |
| `lm_head` | `linear(x, lm_head)` — 행렬곱 | 출력 차원 (27) |

따라서 `wte = matrix(16, 27)` 처럼 차원을 뒤집으면 `wte[token_id]` 에서
`token_id >= 16` 일 때 IndexError 가 발생한다. 첫 번째 차원이 반드시 인덱싱 범위
이상이어야 한다.

### 파라미터 리스트 평탄화

```python
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")
```

모든 행렬의 모든 원소를 하나의 1 차원 리스트로 펼친다. 이렇게 하면 학습 시 모든
파라미터를 일괄적으로 업데이트할 수 있다.

## 섹션 6: 유틸리티 함수들

### linear (선형 변환)

```python
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
```

행렬-벡터 곱셈이다. 입력 벡터 `x` 에 가중치 행렬 `w` 를 곱한다.

`w` 의 각 행(`wo`)과 입력 벡터 `x` 의 내적(dot product)을 계산한다.

**비유**: 설문지에 응답하는 것과 같다. 입력(`x`)이 설문 응답이고, 가중치 행렬의 각
행(`wo`)이 각 질문의 가중치라면, 내적은 "가중 합산 점수"이다. 행이 `nout` 개이므로
`nout` 개의 점수가 나온다.

**크기 변환**: 입력 `x` 가 `nin` 차원이고 `w` 가 `nout x nin` 행렬이면, 출력은
`nout` 차원 벡터이다.

### softmax (확률 변환)

```python
def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]
```

임의의 실수 벡터(logits)를 확률 분포로 변환한다. 모든 값이 0~1 사이가 되고, 합이
정확히 1 이 된다.

동작 단계:

1. `max_val`: 최댓값을 구한다 (수치 안정성을 위해).
2. `(val - max_val).exp()`: 각 값에서 최댓값을 빼고 지수를 취한다. 최댓값을 빼는
   이유는, 큰 값에 `exp()` 를 적용하면 숫자가 무한대로 폭발(overflow)할 수 있기
   때문이다. 최댓값을 빼면 가장 큰 값이 `exp(0) = 1` 이 되어 안전하다.
3. `total`: 모든 지수값의 합을 구한다.
4. `e / total`: 각 값을 총합으로 나눈다.

**비유**: 선거 결과를 발표하는 것과 같다. 각 후보의 원시 점수(logits)를 "득표율(%)"
로 변환하는 과정이다. 점수가 높을수록 확률이 높고, 모든 확률의 합은 100% 이다.

예시: `logits = [2.0, 1.0, 0.1]` 이면 `softmax = [0.659, 0.242, 0.099]`

### rmsnorm (RMS 정규화)

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

벡터의 크기(scale)를 정규화한다. 값들이 너무 커지거나 작아지는 것을 방지한다.

**RMS = Root Mean Square (제곱 평균 제곱근)**. 이름 그대로 세 단계를 거친다:

```
예시: x = [2.0, -1.0, 3.0, 0.0]

1) Square (제곱):       [4.0,  1.0,  9.0,  0.0]
2) Mean (평균):         (4 + 1 + 9 + 0) / 4 = 3.5
3) Root (제곱근):       √3.5 ≈ 1.871
```

코드의 `ms` 는 2 단계까지(Mean Square), `scale = ms ** -0.5` 이 3 단계(Root)를
적용하면서 동시에 역수를 취해 나눗셈 대신 곱셈으로 처리한 것이다.

동작:

1. `ms`: 각 원소의 제곱의 평균 (Mean Square)
2. `scale`: `1 / sqrt(ms + epsilon)`. epsilon(`1e-5`)은 0 으로 나누는 것을 방지.
3. 각 원소에 `scale` 을 곱한다.

결과적으로 벡터의 RMS 가 항상 약 1 이 되도록 스케일링한다. 방향은 보존하고
크기만 정규화한다.

**비유**: 볼륨 자동 조절 장치이다. 소리가 너무 크면 줄이고, 너무 작으면 키워서
일정한 범위를 유지한다. 이렇게 하면 신경망의 각 층에서 값이 폭발하거나 소멸하는
것을 막아 학습이 안정적으로 진행된다.

**LayerNorm 과의 차이**: 일반적인 LayerNorm 은 평균을 빼고 분산으로 나누지만,
RMSNorm 은 평균을 빼지 않고 제곱평균제곱근(RMS)으로만 나눈다. 더 단순하지만 성능은
비슷하여 LLaMA 등 최신 모델에서 많이 사용한다.

## 섹션 7: GPT 모델 (핵심 -- Transformer 아키텍처)

이 함수가 GPT 의 심장이다. 하나의 토큰을 입력받아 다음 토큰의 확률 분포를
출력한다.

### 임베딩 + 위치 인코딩

```python
def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]   # 토큰 임베딩 조회
    pos_emb = state_dict['wpe'][pos_id]     # 위치 임베딩 조회
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # 둘을 더함
    x = rmsnorm(x)
```

1. `wte[token_id]`: 토큰 ID 에 해당하는 16 차원 벡터를 가져온다. 예: 'a'(=0)이면
   `wte[0]`, 즉 16 개의 숫자로 구성된 'a'의 표현.
2. `wpe[pos_id]`: 위치에 해당하는 16 차원 벡터를 가져온다. 예: 3 번째 위치이면
   `wpe[3]`.
3. 두 벡터를 원소별로 더한다.
4. RMS 정규화를 적용한다.

**왜 위치 임베딩이 필요한가?** "cat"과 "tac"는 같은 글자들로 이루어져 있지만 완전히
다른 단어이다. 위치 정보가 없으면 모델은 글자의 순서를 알 수 없다. 위치 임베딩은 "이
글자가 몇 번째에 있는지"를 알려준다.

`keys` 와 `values` 파라미터: 이전 위치에서 계산한 Key 와 Value 를 저장하는
리스트이다. 이를 **KV 캐시**라고 한다. 이전 토큰들의 정보를 다시 계산하지 않고
재사용할 수 있게 해준다.

### Self-Attention (자기 어텐션)

```python
    for li in range(n_layer):
        x_residual = x          # 잔차 연결을 위해 원본 저장
        x = rmsnorm(x)          # 정규화
        q = linear(x, state_dict[f'layer{li}.attn_wq'])  # Query 계산
        k = linear(x, state_dict[f'layer{li}.attn_wk'])  # Key 계산
        v = linear(x, state_dict[f'layer{li}.attn_wv'])  # Value 계산
        keys[li].append(k)      # Key 캐시에 저장
        values[li].append(v)    # Value 캐시에 저장
```

**Self-Attention 의 핵심 개념 -- Q, K, V**:

이 부분은 Transformer 의 핵심인 "Attention" 메커니즘이다. 도서관에 비유하면 가장
이해하기 쉽다.

- **Query (Q)**: "나는 이런 정보가 필요해" -- 검색 질의
- **Key (K)**: "나는 이런 정보를 가지고 있어" -- 책의 제목/태그
- **Value (V)**: "내 실제 내용은 이거야" -- 책의 실제 내용

현재 토큰이 "나는 이런 정보가 필요해(Q)"라고 말하면, 이전의 모든 토큰들이 "나는
이런 정보를 가지고 있어(K)"라고 답한다. Q 와 K 의 유사도가 높은 토큰일수록 그
토큰의 실제 내용(V)을 더 많이 참고한다.

```python
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim     # 이 헤드가 담당하는 시작 인덱스
            q_h = q[hs:hs+head_dim]     # 이 헤드의 Query (4 차원)
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]   # 모든 위치의 Key
            v_h = [vi[hs:hs+head_dim] for vi in values[li]] # 모든 위치의 Value
```

**Multi-Head Attention**: 16 차원 벡터를 4 개의 헤드가 4 차원씩 나눠 담당한다.

**왜 여러 헤드가 필요한가?** 하나의 어텐션 헤드는 하나의 "관점"만 볼 수 있다.
`n_head=1` 이면 각 위치에서 attention score 가 **하나의 확률 분포**만 만들어진다.
하지만 자연어에서는 한 토큰이 여러 이유로 여러 위치를 동시에 참조해야 할 수 있다.
head 가 하나면 이 여러 관계를 하나의 분포로 표현해야 하니 정보가 뭉개진다.

**n_head = 4 의 선택**: `n_embd` 를 균등하게 나눌 수 있으면 된다. 16 차원이면
1, 2, 4, 8, 16 이 가능하다. 4 는 head 수와 head_dim(4) 사이의 균형이 적당한
값이다. head 가 너무 많으면 각 head 의 차원이 너무 작아 표현력이 부족하고, 너무
적으면 다양한 패턴을 포착하지 못한다.

**패턴은 학습으로 결정된다**: 각 head 가 구체적으로 어떤 패턴을 담당하는지는
코드에서 지정하지 않는다. 코드가 하는 것은 q, k, v 벡터를 4 등분하는 것뿐이다.
어떤 head 가 어떤 역할을 맡을지는 `attn_wq`, `attn_wk`, `attn_wv` 의 가중치가
반복적인 gradient 업데이트를 통해 조정되면서 **학습 과정에서 자연스럽게 결정**
된다. 보장되는 것은 4 개의 head 가 서로 다른 가중치 공간에서 독립적으로
attention 을 계산한다는 **구조적 사실**뿐이다.

**Ablation Study 로 검증**: 원 Transformer 논문 ("Attention Is All You Need",
Vaswani et al., 2017)에서 head 수를 1, 4, 8, 16 등으로 바꿔가며 성능을 비교하는
**ablation study** (시스템의 구성 요소를 하나씩 제거/변경하며 영향을 측정하는
실험)를 수행했다. 결과적으로 여러 head 가 단일 head 보다 확실히 우수했으며, 이후
모든 Transformer 가 multi-head attention 을 채택하게 되었다.

```python
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)
```

**어텐션 점수 계산**:

1. `sum(q_h[j] * k_h[t][j] ...)`: Q 와 각 K 의 내적. 유사도를 측정한다.
2. `/ head_dim**0.5`: **스케일링**. `head_dim` 이 4 이므로 `sqrt(4) = 2` 로
   나눈다. 내적 값이 차원 수에 비례하여 커지는 것을 방지한다. 이것 없으면 softmax
   입력이 너무 커져서 하나의 토큰에만 거의 100% 가중치가 집중되는 문제가 생긴다.
3. `softmax(...)`: 점수를 확률로 변환한다. "각 이전 토큰을 얼마나 참고할지"의 비율.

```python
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)
```

**가중 합산**: 어텐션 가중치에 따라 각 위치의 Value 를 가중 평균한다.

- 가중치가 0.7 인 토큰의 Value 는 70% 반영, 0.1 인 토큰은 10% 반영.
- `extend`: 각 헤드의 4 차원 출력을 이어붙여 최종 16 차원 벡터를 만든다.

### 어텐션 출력 + 잔차 연결

```python
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
```

1. `attn_wo`: 어텐션 출력을 한 번 더 선형 변환한다.
2. **잔차 연결(Residual Connection)**: 어텐션의 출력에 원래 입력(`x_residual`)을
   더한다.

**잔차 연결의 비유**: 원본 서류에 메모를 추가하는 것이다. 원본 정보는 유지하면서
새로운 정보(어텐션이 발견한 패턴)를 덧붙인다. 이렇게 하면 학습 시 기울기가 잔차
경로를 통해 쉽게 흐를 수 있어서 깊은 네트워크도 학습이 가능하다.

### MLP (Feed-Forward Network)

```python
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])  # 16 -> 64 확장
        x = [xi.relu() for xi in x]                       # ReLU 활성화
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])  # 64 -> 16 축소
        x = [a + b for a, b in zip(x, x_residual)]       # 잔차 연결
```

**MLP(Multi-Layer Perceptron)** 은 각 토큰을 독립적으로 처리하는 "개인 분석"
단계이다.

어텐션이 "다른 토큰들과의 관계"를 파악하는 단계였다면, MLP 는 "그 관계 정보를
바탕으로 해당 토큰의 표현을 풍부하게 만드는" 단계이다.

동작:

1. 16 차원 -> 64 차원으로 확장: 더 넓은 공간에서 다양한 특징을 포착
2. ReLU: 비선형성 추가 (중요하지 않은 특징을 0 으로 만들어 선택적 활성화)
3. 64 차원 -> 16 차원으로 축소: 다시 원래 크기로 압축
4. 잔차 연결: 원본 정보 보존

**확장-축소 패턴의 비유**: 사진을 확대해서 세밀한 부분을 보정한 뒤 다시 원래 크기로
축소하는 것과 같다. 일시적으로 더 넓은 공간에서 작업하면 더 다양한 패턴을 포착할 수
있다.

### 최종 출력

```python
    logits = linear(x, state_dict['lm_head'])
    return logits
```

최종 16 차원 벡터를 `lm_head` (27x16 행렬)를 통해 27 차원 벡터(=vocab_size)로
변환한다. 이 27 개의 값(logits)은 각 토큰이 다음에 올 가능성을 나타내는 원시
점수이다.

## 섹션 8: 학습 루프 (Training Loop)

### 옵티마이저 설정 (Adam)

```python
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)  # 1 차 모멘트 (기울기의 이동 평균)
v = [0.0] * len(params)  # 2 차 모멘트 (기울기 제곱의 이동 평균)
```

**Adam 옵티마이저**: 가장 널리 쓰이는 최적화 알고리즘이다. 단순한 경사
하강법(SGD)의 진화판이다.

| 변수 | 역할 | 비유 |
|------|------|------|
| `learning_rate` (0.01) | 한 번에 얼마나 크게 움직일지 | 걸음 크기 |
| `beta1` (0.85) | 1 차 모멘트의 감쇠율 | 관성의 강도 |
| `beta2` (0.99) | 2 차 모멘트의 감쇠율 | 지형 기억의 강도 |
| `m` | 기울기의 이동 평균 | 지금까지 움직인 평균 방향 |
| `v` | 기울기 제곱의 이동 평균 | 지금까지 경험한 경사의 급함 정도 |

**Adam 비유**: 언덕에서 공이 굴러 내려가는데, 단순히 현재 경사만 보는 것이 아니라:

- `m`: 지금까지의 움직임 방향을 기억한다 (관성/모멘텀). 같은 방향으로 계속 경사가
  있으면 가속한다.
- `v`: 지금까지 경사가 얼마나 급했는지 기억한다. 경사가 급한 방향은 조심스럽게,
  완만한 방향은 과감하게 움직인다.

### 메인 학습 루프

```python
num_steps = 1000

for step in range(num_steps):
    doc = docs[step % len(docs)]                              # 이름 하나 선택
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS] # 토큰화
    n = min(block_size, len(tokens) - 1)                      # 시퀀스 길이 제한
```

각 스텝에서 일어나는 일:

1. `docs[step % len(docs)]`: 데이터에서 이름 하나를 순서대로 선택한다. `%` 로 인해
   데이터 끝에 도달하면 처음부터 다시 시작한다.

2. **토큰화 예시**: 이름이 "emma"라면:
   - `[BOS]` = `[26]`
   - `[uchars.index('e'), uchars.index('m'), uchars.index('m'), uchars.index('a')]`
     = `[4, 12, 12, 0]`
   - `[BOS]` = `[26]`
   - 최종: `tokens = [26, 4, 12, 12, 0, 26]`
   - 해석: "시작 -> e -> m -> m -> a -> 끝"

3. `n = min(block_size, len(tokens) - 1)`: 학습에 사용할 위치 수. `-1` 인 이유는
   마지막 토큰은 예측 대상이지 입력이 아니기 때문이다.

### 순전파 및 손실 계산

```python
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []

    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)

    loss = (1 / n) * sum(losses)
```

단계별 설명 (이름 "emma" 기준):

각 위치에서 "현재 토큰을 보고 다음 토큰을 예측"한다:

| 위치 | 입력 (token_id) | 정답 (target_id) | 모델이 해야 할 일 |
|------|-----------------|-------------------|-------------------|
| 0 | BOS (26) | 'e' (4) | 시작 신호를 보고 'e'를 예측 |
| 1 | 'e' (4) | 'm' (12) | 'e' 다음에 'm'이 올 것을 예측 |
| 2 | 'm' (12) | 'm' (12) | 'm' 다음에 'm'이 올 것을 예측 |
| 3 | 'm' (12) | 'a' (0) | 'm' 다음에 'a'가 올 것을 예측 |
| 4 | 'a' (0) | BOS (26) | 'a' 다음에 끝남을 예측 |

### 변수 상태 추적 예시 (doc = "iamslash")

`doc = "iamslash"` 로 한 스텝의 forward pass 를 구체적으로 추적한다.

**토크나이징**:

```
doc = "iamslash"
tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
       = [26,    8,  0,  12, 18, 11,  0,  18, 7,    26]
          BOS    i   a   m   s   l   a   s   h    BOS
```

**n 의 값**:

```
n = min(block_size, len(tokens) - 1) = min(16, 9) = 9
```

**초기 상태** (`n_layer = 1` 이므로):

```
keys   = [[]]      # layer 0 에 대한 빈 리스트
values = [[]]
losses = []
```

**루프 반복별 변수 추적** (`for pos_id in range(9)`):

```
tokens:  [BOS,  i,   a,   m,   s,   l,   a,   s,   h,  BOS]
          26    8    0    12   18   11    0   18    7   26
pos:       0    1    2    3    4    5    6    7    8
```

| pos_id | token_id (입력) | target_id (정답) | keys[0] 길이 | loss_t |
|--------|-----------------|------------------|--------------|--------|
| 0 | 26 (BOS) | 8 ('i') | 1 | `-probs[8].log()` |
| 1 | 8 ('i') | 0 ('a') | 2 | `-probs[0].log()` |
| 2 | 0 ('a') | 12 ('m') | 3 | `-probs[12].log()` |
| 3 | 12 ('m') | 18 ('s') | 4 | `-probs[18].log()` |
| 4 | 18 ('s') | 11 ('l') | 5 | `-probs[11].log()` |
| 5 | 11 ('l') | 0 ('a') | 6 | `-probs[0].log()` |
| 6 | 0 ('a') | 18 ('s') | 7 | `-probs[18].log()` |
| 7 | 18 ('s') | 7 ('h') | 8 | `-probs[7].log()` |
| 8 | 7 ('h') | 26 (BOS) | 9 | `-probs[26].log()` |

- `keys[0]`, `values[0]`: 매 iteration 마다 현재 토큰의 k, v 벡터가 **append**
  된다. 이것이 KV cache 이며, 이전 위치들을 다시 계산하지 않고 attention 을 수행할
  수 있게 해준다. `pos_id=t` 일 때 attention 은 0~t 까지 총 t+1 개의 key/value 를
  참조한다.
- `probs`: 매 iteration 마다 **새로 계산**된다. 27 개 확률값이며,
  `probs[target_id]` 가 클수록 loss 가 작아진다.
- `losses`: 9 개의 개별 loss 가 모인 뒤, `loss = (1/9) * sum(losses)` 로 평균을
  내어 최종 loss 를 구한다.

**손실 함수: Cross-Entropy Loss**

```python
loss_t = -probs[target_id].log()
```

이것은 **크로스 엔트로피 손실**이다. 모델이 정답 토큰에 부여한 확률의 음의 로그를
취한다.

직관:

- 모델이 정답에 확률 1.0 을 부여하면: `-log(1.0) = 0` (완벽, 손실 없음)
- 모델이 정답에 확률 0.5 를 부여하면: `-log(0.5) = 0.693` (반반)
- 모델이 정답에 확률 0.01 을 부여하면: `-log(0.01) = 4.605` (거의 틀림, 큰 손실)

즉, 정답에 높은 확률을 부여할수록 손실이 작아지고, 학습이 잘 되고 있다는 뜻이다.

```python
loss = (1 / n) * sum(losses)
```

모든 위치의 손실을 평균 낸다. 이것이 이 학습 단계의 최종 손실값이다.

### 역전파

```python
    loss.backward()
```

이 한 줄이 앞서 정의한 `Value.backward()` 메서드를 호출한다. 최종 `loss` 에서
시작하여 계산 그래프를 역방향으로 거슬러 올라가며, 모든 파라미터의
기울기(`grad`)를 계산한다.

이 시점에서 모든 `params[i].grad` 에 "이 파라미터를 어느 방향으로 얼마나 바꾸면
손실이 줄어드는지"의 정보가 담기게 된다.

### 파라미터 업데이트 (Adam)

```python
    lr_t = learning_rate * (1 - step / num_steps)  # 학습률 선형 감소

    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad          # 1 차 모멘트 업데이트
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2     # 2 차 모멘트 업데이트
        m_hat = m[i] / (1 - beta1 ** (step + 1))             # 편향 보정
        v_hat = v[i] / (1 - beta2 ** (step + 1))             # 편향 보정
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)   # 파라미터 업데이트
        p.grad = 0                                            # 기울기 초기화
```

**학습률 스케줄링**:

```python
lr_t = learning_rate * (1 - step / num_steps)
```

학습률이 처음(0.01)에서 끝(0)까지 선형으로 감소한다. 학습 초반에는 큰 걸음으로
빠르게 이동하고, 후반에는 작은 걸음으로 섬세하게 조정한다.

**Adam 업데이트 각 줄 설명**:

1. **1 차 모멘트 업데이트** (`m`): 기울기의 지수이동평균. 85% 는 이전 방향을
   유지하고, 15% 만 새 기울기 방향을 반영한다. 이것이 "관성" 효과이다.

2. **2 차 모멘트 업데이트** (`v`): 기울기 제곱의 지수이동평균. 99% 는 이전 정보를
   유지한다. 기울기의 변동성(분산)을 추적한다.

3. **편향 보정** (`m_hat`, `v_hat`): `m` 과 `v` 가 0 으로 초기화되었기 때문에, 학습
   초기에는 실제보다 작은 값을 가진다. `1 - beta^(step+1)` 로 나눠서 이 편향을
   보정한다. step 이 커질수록 보정 계수가 1 에 가까워져 영향이 줄어든다.

4. **파라미터 업데이트**: `m_hat / (sqrt(v_hat) + epsilon)` 형태이다.
   - `m_hat`: 어느 방향으로 갈지 (방향)
   - `sqrt(v_hat)`: 얼마나 조심스럽게 갈지 (스케일 조정). 기울기가 자주 큰 방향은
     `v_hat` 이 크므로 나누면 작아진다 = 조심스럽게 이동. 기울기가 자주 작은 방향은
     `v_hat` 이 작으므로 상대적으로 크게 이동한다.
   - `eps_adam`: 0 으로 나누는 것 방지

5. **기울기 초기화** (`p.grad = 0`): 다음 스텝의 역전파를 위해 기울기를 0 으로
   리셋한다. 이걸 안 하면 이전 스텝의 기울기가 누적된다.

### 로그 출력

```python
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')
```

현재 학습 진행 상황을 출력한다. `end='\r'` 은 같은 줄에 덮어쓰기하여 깔끔한 진행
표시를 만든다.

## 섹션 9: 추론 (Inference -- 텍스트 생성)

```python
temperature = 0.5

print("\n--- inference (new, hallucinated names) ---")

for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []

    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]

        if token_id == BOS:
            break

        sample.append(uchars[token_id])

    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
```

학습된 모델을 사용하여 새로운 이름을 생성한다.

### 동작 흐름

1. **시작**: `token_id = BOS` 로 시작한다. "새 이름을 시작해라"는 신호.

2. **한 글자씩 생성 반복**:
   - GPT 에 현재 토큰과 위치를 넣어 다음 토큰의 확률 분포를 얻는다.
   - 확률에 따라 무작위로 다음 토큰을 선택한다.
   - BOS 토큰이 나오면 이름 생성을 종료한다 (= 이름 끝 신호).
   - 아니면 해당 문자를 `sample` 에 추가하고 계속한다.

### Temperature (온도)

```python
probs = softmax([l / temperature for l in logits])
```

`temperature = 0.5` 로 logits 를 나눈 뒤 softmax 를 적용한다.

**Temperature 의 효과**:

| Temperature | 효과 | 비유 |
|-------------|------|------|
| 낮음 (0.1) | 가장 확률 높은 토큰만 거의 선택 | 보수적, 안전한 선택 |
| 중간 (0.5) | 높은 확률 토큰 위주이나 다양성 있음 | 균형 잡힌 선택 |
| 높음 (1.0) | 원래 확률 분포 그대로 | 모델의 원래 판단 |
| 매우 높음 (2.0) | 거의 균일 분포, 랜덤에 가까움 | 모험적, 예측 불가 |

**왜 나누기인가?** logits 를 작은 수(0.5)로 나누면 값이 2 배로 커진다. softmax
입력이 커지면 확률 분포가 더 "뾰족"해진다(가장 큰 값의 확률이 더 커짐). 즉, 낮은
temperature = 더 확신에 찬 선택.

### random.choices

```python
token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
```

`random.choices(population, weights=..., k=1)` 는 **가중치 기반 확률 샘플링**
함수이다. 확률이 높은 토큰이 선택될 가능성이 높지만, 낮은 확률의 토큰도 가끔
선택될 수 있다. 이것이 생성의 다양성을 만든다.

**왜 `[0]` 이 필요한가?** `random.choices` 는 `k` 개의 샘플이 담긴 **리스트**를
반환한다. `k=1` (기본값)이라도 반환값은 `[8]` 같은 길이 1 인 리스트이므로,
스칼라 정수를 꺼내려면 `[0]` 으로 인덱싱해야 한다.

```python
random.choices(range(27), weights=...)    # → [8]   (리스트)
random.choices(range(27), weights=...)[0] # →  8    (정수)
```

### 변수 상태 추적 예시 (모델이 "lia" 를 생성한다고 가정)

| pos_id | token_id (입력) | keys[0] 길이 | probs | token_id (출력) | sample |
|--------|-----------------|--------------|-------|-----------------|--------|
| 0 | 26 (BOS) | 1 | softmax(logits/0.5) | 11 ('l') | `['l']` |
| 1 | 11 ('l') | 2 | softmax(logits/0.5) | 8 ('i') | `['l','i']` |
| 2 | 8 ('i') | 3 | softmax(logits/0.5) | 0 ('a') | `['l','i','a']` |
| 3 | 0 ('a') | 4 | softmax(logits/0.5) | 26 (BOS) → **break** | `['l','i','a']` |

학습과의 핵심 차이: `token_id` 가 정답 데이터에서 오는 것이 아니라 **이전 스텝의
출력이 다음 스텝의 입력**이 된다 (autoregressive generation).

### 학습 vs 추론 비교

| | 학습 (Training) | 추론 (Inference) |
|---|---|---|
| `token_id` | `tokens[pos_id]` (정답 데이터) | 이전 스텝의 출력 (모델이 생성) |
| `target_id` | `tokens[pos_id + 1]` (정답) | 없음 (정답이 없으므로 loss 도 없음) |
| `probs` | `softmax(logits)` | `softmax(logits / temperature)` |
| 출력 사용 | `loss = -log(probs[target_id])` | `random.choices(weights=probs)` |
| 종료 조건 | `n` 번 (토큰 수만큼) | BOS 나올 때 또는 `block_size` |

### 종료 조건

```python
if token_id == BOS:
    break
```

BOS 토큰이 생성되면 이름이 끝난 것으로 간주한다. 모델이 학습 과정에서 "이름의
끝에는 BOS 가 온다"는 패턴을 학습했기 때문에, 적절한 시점에 BOS 를 생성하여 이름을
자연스럽게 마무리한다.

# 전체 데이터 흐름 요약

```
이름 "emma"의 학습 과정:

1. 토큰화: "emma" -> [26, 4, 12, 12, 0, 26]

2. 각 위치에서:
   토큰 26(BOS) 입력
   -> 임베딩 조회 (27 개 중 26 번) -> 16 차원 벡터
   -> 위치 임베딩 더하기 (0 번 위치) -> 16 차원 벡터
   -> RMS 정규화 -> 16 차원 벡터
   -> Self-Attention (Q,K,V 계산, 4 개 헤드, 이전 토큰 참조)
   -> 잔차 연결
   -> MLP (16 -> 64 -> ReLU -> 16)
   -> 잔차 연결
   -> lm_head 변환 -> 27 차원 logits
   -> softmax -> 27 개 확률
   -> 정답 'e'(=4)의 확률로 손실 계산: -log(prob[4])

3. 모든 위치의 손실 평균 -> 최종 loss

4. loss.backward() -> 모든 파라미터의 기울기 계산

5. Adam 으로 파라미터 업데이트

6. 1000 번 반복

7. 추론: BOS 에서 시작, 한 글자씩 확률적으로 생성, BOS 나오면 종료
```

# 모델 크기 비교

| 항목 | microgpt.py | GPT-2 Small | GPT-3 |
|------|-------------|-------------|-------|
| 레이어 수 | 1 | 12 | 96 |
| 임베딩 차원 | 16 | 768 | 12,288 |
| 어텐션 헤드 | 4 | 12 | 96 |
| 파라미터 수 | ~3,000 개 | 1.17 억 | 1,750 억 |
| 학습 데이터 | 이름 32K 개 | 웹텍스트 40GB | 웹텍스트 570GB |
| 컨텍스트 길이 | 16 | 1,024 | 2,048 |

규모는 극단적으로 다르지만, **알고리즘 자체는 동일하다.** 이것이 Karpathy 가
"Everything else is just efficiency"라고 말한 이유이다. PyTorch, GPU, 분산 학습,
FlashAttention 등은 모두 이 동일한 알고리즘을 빠르게 실행하기 위한
엔지니어링이다.
