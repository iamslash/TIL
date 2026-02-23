- [Abstract](#abstract)
- [Materials](#materials)
- [micrograd vs microgpt 비교](#micrograd-vs-microgpt-비교)
- [전체 코드](#전체-코드)
  - [engine.py -- 자동 미분 엔진](#enginepy----자동-미분-엔진)
  - [nn.py -- 신경망 라이브러리](#nnpy----신경망-라이브러리)
- [섹션별 상세 해설](#섹션별-상세-해설)
  - [Part 1: engine.py -- 자동 미분 엔진](#part-1-enginepy----자동-미분-엔진)
    - [Value 클래스 생성자](#value-클래스-생성자)
    - [덧셈 (`__add__`)](#덧셈-__add__)
    - [곱셈 (`__mul__`)](#곱셈-__mul__)
    - [거듭제곱 (`__pow__`)](#거듭제곱-__pow__)
    - [ReLU](#relu)
    - [역전파 (`backward`)](#역전파-backward)
    - [보조 연산자](#보조-연산자)
  - [Part 2: nn.py -- 신경망 라이브러리](#part-2-nnpy----신경망-라이브러리)
    - [Module (기반 클래스)](#module-기반-클래스)
    - [Neuron (뉴런)](#neuron-뉴런)
    - [Layer (레이어)](#layer-레이어)
    - [MLP (Multi-Layer Perceptron)](#mlp-multi-layer-perceptron)
  - [Part 3: 사용 예시](#part-3-사용-예시)
    - [기본 사용법](#기본-사용법)
    - [MLP 학습 루프 예시](#mlp-학습-루프-예시)
  - [Part 4: 테스트 -- PyTorch 와의 검증](#part-4-테스트----pytorch-와의-검증)
- [전체 아키텍처 요약](#전체-아키텍처-요약)

--------

# Abstract

- [micrograd](https://github.com/karpathy/micrograd) | github
- Andrej Karpathy 가 만든 **최소 자동 미분(Autograd) 엔진**이다.
- 역전파(backpropagation)를 동적으로 구성되는 계산 그래프 위에서 수행한다.
- 엔진 약 100 줄, 신경망 라이브러리 약 50 줄로 구성된다.
- PyTorch 와 동일한 API 를 스칼라 단위로 순수 파이썬으로 구현한 것이다.

> microgpt.py 가 "GPT 전체를 한 파일로" 구현한 것이라면, micrograd 는 그 중
> **자동 미분 엔진과 신경망 레이어**만 분리하여 라이브러리 형태로 만든 것이다.

# Materials

- [micrograd](https://github.com/karpathy/micrograd) | github
- [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0) |
  youtube | Karpathy 본인의 2 시간 강의

# micrograd vs microgpt 비교

| | micrograd | microgpt |
|---|---|---|
| 목적 | 자동 미분 엔진 + 신경망 레이어 라이브러리 | GPT 전체 (학습 + 추론) |
| 구조 | `engine.py` + `nn.py` (모듈 분리) | 단일 파일 |
| Value 클래스 | `_backward` 클로저 방식 | `_local_grads` 튜플 방식 |
| 신경망 | Neuron, Layer, MLP 클래스 | gpt 함수 (인라인) |
| 학습 루프 | 없음 (사용자가 작성) | 포함 (Adam + 1000 스텝) |
| 데이터 | 없음 | 이름 데이터셋 포함 |

# 전체 코드

## engine.py -- 자동 미분 엔진

```python
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
```

## nn.py -- 신경망 라이브러리

```python
import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
```

# 섹션별 상세 해설

## Part 1: engine.py -- 자동 미분 엔진

### Value 클래스 생성자

```python
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
```

`Value` 는 하나의 스칼라 값을 감싸서, 자동 미분을 가능하게 하는 래퍼이다.

각 속성의 의미:

| 속성 | 타입 | 의미 |
|------|------|------|
| `data` | `float` | 실제 숫자 값 |
| `grad` | `float` | 최종 loss 에 대한 이 값의 민감도 (`∂loss/∂data`) |
| `_backward` | `함수` | 이 노드의 역전파 로직 (클로저) |
| `_prev` | `set` | 이 값을 만든 입력값들 (계산 그래프의 부모 노드) |
| `_op` | `str` | 이 값을 만든 연산 이름 (`'+'`, `'*'` 등, 디버깅용) |

**microgpt 의 Value 와의 핵심 차이 -- 클로저 방식**:

microgpt 에서는 `_local_grads` 튜플에 미분값을 저장하고, `backward()` 에서 일괄
계산했다. micrograd 에서는 **각 연산이 자신만의 `_backward` 함수를 클로저로
정의**하여 노드에 달아둔다.

```
microgpt:  out._local_grads = (1, 1)           → backward 에서 local_grad * v.grad
micrograd: out._backward = lambda: self.grad += out.grad  → backward 에서 v._backward()
```

클로저 방식의 장점:
- 역전파 로직이 연산 정의와 같은 위치에 있어 읽기 쉽다.
- 복잡한 연산도 자유롭게 정의할 수 있다 (튜플에 담기 어려운 경우).

**`_backward = lambda: None`**: 초기값은 "아무것도 하지 않는 함수"이다. 리프
노드(사용자가 직접 만든 `Value(3.0)` 같은 값)는 역전파할 부모가 없으므로 이
기본값이 유지된다.

**`_prev = set(_children)`**: `set` 으로 저장하는 이유는 중복 방지와 순서 무관
때문이다. microgpt 의 `_children` 은 튜플이었다.

**`_op`**: 순수하게 디버깅/시각화 용도이다. graphviz 로 계산 그래프를 그릴 때 각
노드에 연산 이름을 표시하는 데 사용한다. 계산에는 영향을 주지 않는다.

### 덧셈 (`__add__`)

```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
```

**순전파**: `self.data + other.data` 를 계산하여 새 `Value` 를 만든다.

**역전파 클로저**: `c = a + b` 에서 `∂c/∂a = 1`, `∂c/∂b = 1` 이므로:

```
a.grad += 1 * out.grad  →  a.grad += out.grad
b.grad += 1 * out.grad  →  b.grad += out.grad
```

덧셈의 미분은 1 이므로, `out.grad` 를 그대로 양쪽 부모에게 전달한다.

**`isinstance` 체크**: `Value(3) + 5` 처럼 일반 숫자와 연산할 수 있도록, 일반
숫자를 `Value` 로 감싸준다.

**`+=` 를 쓰는 이유**: 하나의 `Value` 가 계산 그래프에서 여러 경로로 사용될 수
있다. 예: `c = a + a` 에서 `a` 는 두 번 기여하므로 기울기가 누적되어야 한다.

### 곱셈 (`__mul__`)

```python
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
```

**역전파 클로저**: `c = a * b` 에서 `∂c/∂a = b`, `∂c/∂b = a` 이므로:

```
a.grad += b.data * out.grad
b.grad += a.data * out.grad
```

**클로저가 `other.data`, `self.data` 를 캡처하는 방식**: `_backward` 함수는
정의 시점의 `self`, `other`, `out` 을 클로저로 **캡처**한다. 나중에
`backward()` 에서 호출될 때, 순전파 시점에 존재했던 그 변수들을 참조하여 올바른
기울기를 계산한다.

### 거듭제곱 (`__pow__`)

```python
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
```

**`assert`**: 지수(`other`)는 `int` 또는 `float` 만 허용한다. `Value ** Value` 는
지원하지 않는다. 나눗셈(`a / b`)이 내부적으로 `a * b**-1` 로 구현되므로, `other`
에 `-1` 같은 정수가 오는 것만 지원하면 충분하다.

**역전파**: `x^n` 의 미분은 `n * x^(n-1)` 이다. 예: `x^3` 의 미분은 `3x^2`.

**`_children` 가 `(self,)` 인 이유**: 거듭제곱의 지수(`other`)는 `Value` 가
아니라 일반 숫자이므로, 계산 그래프에서 부모는 `self` 하나뿐이다.

### ReLU

```python
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
```

**순전파**: 입력이 음수면 0, 양수면 그대로 통과.

**역전파**: `out.data > 0` 은 Python 에서 `True`(=1) 또는 `False`(=0) 를
반환한다.

```
입력이 양수였으면: self.grad += 1 * out.grad  (그대로 전달)
입력이 음수였으면: self.grad += 0 * out.grad  (완전 차단)
```

**`out.data` 를 쓰는 이유**: `self.data` 가 아닌 `out.data` 를 체크한다. 순전파
결과가 양수인지를 기준으로 역전파 게이트를 결정하는 것이다. `self.data < 0` 이면
`out.data = 0` 이므로 `out.data > 0` 은 `False` 가 되어 기울기가 차단된다.
결과는 동일하지만, 이 방식이 "출력이 활성화되었는가?"를 직접 확인하는 의미가 더
명확하다.

### 역전파 (`backward`)

```python
    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()
```

microgpt 의 `backward` 와 구조가 동일하다. 두 단계로 동작한다.

**1 단계: 위상 정렬**

재귀적으로 계산 그래프를 탐색하여, 모든 노드를 의존성 순서(자식 먼저, 부모
나중)로 정렬한다.

**2 단계: 역순으로 기울기 전파**

```python
self.grad = 1   # loss 의 loss 에 대한 기울기는 1
for v in reversed(topo):
    v._backward() # 각 노드가 자신의 부모에게 기울기를 전파
```

**microgpt 와의 차이**:

```python
# microgpt: backward 에서 직접 체인 룰 계산
for child, local_grad in zip(v._children, v._local_grads):
    child.grad += local_grad * v.grad

# micrograd: 각 노드의 _backward 클로저를 호출
v._backward()
```

microgpt 는 `local_grad * v.grad` 를 backward 루프에서 계산한다. micrograd 는
각 연산이 미리 정의해둔 `_backward` 클로저를 호출한다. 결과는 동일하지만,
micrograd 방식이 연산별 역전파 로직을 **캡슐화**한다는 점에서 더 확장성이 높다.

### 보조 연산자

```python
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
```

기존에 정의한 `__add__`, `__mul__`, `__pow__` 를 조합하여 뺄셈, 나눗셈 등을
구현한다. microgpt 의 보조 연산자와 동일한 패턴이다.

| 연산자 | 호출 상황 | 구현 방식 |
|--------|----------|----------|
| `__neg__` | `-a` | `a * -1` |
| `__radd__` | `5 + a` (왼쪽이 일반 숫자) | `a + 5` 로 위임 |
| `__sub__` | `a - b` | `a + (-b)` |
| `__rsub__` | `5 - a` | `5 + (-a)` |
| `__rmul__` | `5 * a` | `a * 5` 로 위임 |
| `__truediv__` | `a / b` | `a * b^(-1)` |
| `__rtruediv__` | `5 / a` | `5 * a^(-1)` |
| `__repr__` | `print(a)` | `"Value(data=3.0, grad=0.5)"` |

**나눗셈이 `__pow__` 를 재사용하는 방식**: `a / b` 는 `a * b**(-1)` 로 변환된다.
`b**(-1)` 은 `__pow__` 에서 처리되고, 그 결과와 `a` 의 곱은 `__mul__` 에서
처리된다. 각 단계마다 `_backward` 클로저가 달려 있으므로, 역전파도 자동으로
체인된다.

---

## Part 2: nn.py -- 신경망 라이브러리

### Module (기반 클래스)

```python
class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
```

모든 신경망 컴포넌트의 부모 클래스이다. PyTorch 의 `torch.nn.Module` 에 해당한다.

- `zero_grad()`: 모든 파라미터의 기울기를 0 으로 초기화한다. 학습 루프에서 매
  스텝마다 호출해야 한다. (microgpt 에서 `p.grad = 0` 을 직접 한 것과 동일)
- `parameters()`: 기본값은 빈 리스트. 자식 클래스가 오버라이드한다.

### Neuron (뉴런)

```python
class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin
```

**하나의 뉴런**을 나타낸다. `nin` 개의 입력을 받아 하나의 출력을 만든다.

- `self.w`: 가중치 벡터. `nin` 개의 `Value` 로 구성. `[-1, 1]` 범위의 균일 분포로
  초기화. (microgpt 는 가우시안 분포를 사용했다)
- `self.b`: 바이어스 (편향). 0 으로 초기화. (microgpt 에는 바이어스가 없었다)
- `self.nonlin`: `True` 이면 ReLU 활성화 적용, `False` 이면 선형 출력.

```python
    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act
```

**`__call__`**: `neuron(x)` 형태로 호출 가능하게 한다.

동작:

```
act = w[0]*x[0] + w[1]*x[1] + ... + w[n-1]*x[n-1] + b
```

`sum(iterable, start)` 에서 `start=self.b` 로 시작값을 바이어스로 설정한다.
이렇게 하면 가중합에 바이어스가 자동으로 더해진다.

풀어쓰면:

```python
act = self.b
for wi, xi in zip(self.w, x):
    act = act + wi * xi
return act.relu() if self.nonlin else act
```

**microgpt 의 `linear()` 함수와의 차이**: microgpt 의 `linear(x, w)` 는 행렬의
각 행과 벡터의 내적을 계산하여 여러 출력을 만들었다. micrograd 의 `Neuron` 은
하나의 행(가중치 벡터)과 입력의 내적 + 바이어스 + 활성화를 하나의 객체로
캡슐화한다.

```python
    def parameters(self):
        return self.w + [self.b]
```

이 뉴런의 학습 가능한 파라미터를 반환한다. 가중치 리스트 + 바이어스 = `nin + 1`
개.

### Layer (레이어)

```python
class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
```

`nout` 개의 `Neuron` 을 모아놓은 것이다. 각 뉴런이 같은 입력을 받아 하나씩
출력하므로, 전체적으로 `nin → nout` 변환을 수행한다.

`**kwargs` 는 `Neuron` 에 전달할 추가 인자 (예: `nonlin=False`)를 그대로
전달한다.

```python
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
```

각 뉴런에 입력 `x` 를 넣고 결과를 모은다.

**`out[0] if len(out) == 1`**: 출력 뉴런이 1 개뿐이면 리스트가 아닌 스칼라
`Value` 를 반환한다. 이렇게 하면 마지막 레이어(출력 1 개)에서 `loss = model(x)` 로
바로 사용할 수 있다.

```python
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
```

모든 뉴런의 파라미터를 하나의 리스트로 펼친다.

### MLP (Multi-Layer Perceptron)

```python
class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
```

여러 `Layer` 를 순차적으로 쌓은 신경망이다.

**`sz = [nin] + nouts`**: 각 레이어의 입출력 크기를 하나의 리스트로 만든다.

예: `MLP(3, [4, 4, 1])` 이면:

```
sz = [3, 4, 4, 1]
Layer(3, 4, nonlin=True)   # i=0, 0 != 2  → ReLU
Layer(4, 4, nonlin=True)   # i=1, 1 != 2  → ReLU
Layer(4, 1, nonlin=False)  # i=2, 2 == 2  → 선형 (마지막 레이어)
```

**`nonlin=i!=len(nouts)-1`**: 마지막 레이어를 제외한 모든 레이어에 ReLU 를
적용한다. 마지막 레이어는 선형(활성화 없음)으로 두어 출력 범위를 제한하지 않는다.

```python
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

입력 `x` 를 첫 번째 레이어에 넣고, 그 출력을 다음 레이어의 입력으로 전달하는
것을 반복한다. 이전 레이어의 출력이 다음 레이어의 입력이 되는 **순차 연결**이다.

```python
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

모든 레이어의 모든 파라미터를 하나의 리스트로 펼친다. microgpt 의 `params`
평탄화와 동일한 목적이다.

---

## Part 3: 사용 예시

### 기본 사용법

```python
from micrograd.engine import Value

a = Value(2.0)
b = Value(-3.0)
c = a * b          # c.data = -6.0
d = c + Value(10)  # d.data = 4.0
d.backward()

print(a.grad)  # -3.0  (d 에 대한 a 의 민감도)
print(b.grad)  #  2.0  (d 에 대한 b 의 민감도)
```

**수동 추적**:

```
d = a * b + 10 = 2 * (-3) + 10 = 4

∂d/∂a = b = -3.0  → a 를 1 늘리면 d 는 3 줄어듦
∂d/∂b = a =  2.0  → b 를 1 늘리면 d 는 2 늘어남
```

### MLP 학습 루프 예시

```python
from micrograd.nn import MLP

# 2 입력, 은닉층 [16, 16], 1 출력
model = MLP(2, [16, 16, 1])

# 학습 데이터 (XOR 비슷한 문제)
xs = [[2.0, 3.0], [-1.0, -2.0], [3.0, -1.0]]
ys = [1.0, -1.0, 1.0]

for step in range(100):
    # 순전파: 각 입력에 대해 예측
    preds = [model(x) for x in xs]

    # 손실 계산: 평균 제곱 오차
    loss = sum((p - y)**2 for p, y in zip(preds, ys))

    # 역전파 전 기울기 초기화
    model.zero_grad()

    # 역전파
    loss.backward()

    # 파라미터 업데이트 (단순 SGD)
    for p in model.parameters():
        p.data -= 0.01 * p.grad

    print(f"step {step}, loss {loss.data:.4f}")
```

**microgpt 와의 비교**:

| 단계 | microgpt | micrograd |
|------|----------|-----------|
| 기울기 초기화 | `p.grad = 0` (직접) | `model.zero_grad()` (메서드) |
| 순전파 | `gpt(token_id, pos_id, ...)` | `model(x)` |
| 손실 | `-probs[target].log()` | `(pred - y)**2` |
| 역전파 | `loss.backward()` | `loss.backward()` (동일) |
| 업데이트 | Adam (m, v, bias correction) | SGD (`p.data -= lr * p.grad`) |

micrograd 는 Adam 대신 단순 SGD 를 사용한다. SGD 는 `p.data -= lr * p.grad`
한 줄이다. Adam 은 SGD 에 모멘텀과 적응적 학습률을 추가한 것이다.

---

## Part 4: 테스트 -- PyTorch 와의 검증

```python
import torch
from micrograd.engine import Value

def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()
```

**동일한 연산을 micrograd 와 PyTorch 로 각각 수행하고, 결과가 일치하는지
확인**한다.

**테스트 전략**:
1. 동일한 수식을 micrograd (`Value`) 와 PyTorch (`torch.Tensor`) 로 구성
2. 순전파 결과(`data`)가 동일한지 확인
3. 역전파 결과(`grad`)가 동일한지 확인

이것은 micrograd 의 자동 미분이 정확하다는 것을 PyTorch 라는 "정답지"와 비교하여
증명하는 것이다.

**수동 추적** (`x = -4.0`):

```
z = 2*(-4) + 2 + (-4) = -8 + 2 - 4 = -10
q = relu(-10) + (-10)*(-4) = 0 + 40 = 40
h = relu((-10)*(-10)) = relu(100) = 100
y = 100 + 40 + 40*(-4) = 100 + 40 - 160 = -20
```

# 전체 아키텍처 요약

```
engine.py (자동 미분 엔진)
├── Value 클래스
│   ├── data, grad, _backward, _prev, _op
│   ├── __add__, __mul__, __pow__, relu  (순전파 + 역전파 클로저)
│   ├── backward()  (위상 정렬 + 체인 룰)
│   └── 보조 연산자  (__neg__, __sub__, __truediv__, ...)
│
nn.py (신경망 라이브러리)
├── Module (기반 클래스: zero_grad, parameters)
├── Neuron (가중합 + 바이어스 + ReLU)
│   └── parameters: w + [b]
├── Layer  (Neuron 여러 개)
│   └── parameters: 모든 뉴런의 파라미터
└── MLP    (Layer 여러 개)
    └── parameters: 모든 레이어의 파라미터
```

**데이터 흐름**:

```
입력 x ──→ Layer 1 (nin→n1, ReLU) ──→ Layer 2 (n1→n2, ReLU) ──→ Layer 3 (n2→1, Linear) ──→ 출력
              │                            │                            │
              └── Neuron × n1 개            └── Neuron × n2 개           └── Neuron 1 개
```

**학습 루프 흐름**:

```
1. pred = model(x)          순전파: Value 들의 연산 → 계산 그래프 구축
2. loss = (pred - y)**2     손실 계산: 계산 그래프에 추가
3. model.zero_grad()        기울기 초기화 (이전 스텝 잔여값 제거)
4. loss.backward()          역전파: 위상 정렬 → 각 노드의 _backward() 호출
5. p.data -= lr * p.grad    파라미터 업데이트: 기울기의 반대 방향으로 이동
6. 1~5 반복
```
