- [Abstract](#abstract)
- [Materials](#materials)
- [체인 룰이란?](#체인-룰이란)
- [forward pass 와 backward pass 의 역할 분담](#forward-pass-와-backward-pass-의-역할-분담)
- [예제: `L = (a * b) + b`, a=2, b=3](#예제-l--a--b--b-a2-b3)
  - [계산 그래프](#계산-그래프)
  - [forward pass: `_local_grads` 스냅샷 저장](#forward-pass-_local_grads-스냅샷-저장)
  - [backward pass: 체인 룰로 `grad` 계산](#backward-pass-체인-룰로-grad-계산)
    - [초기화](#초기화)
    - [`a.grad` 계산 -- 체인 룰 전개](#agrad-계산----체인-룰-전개)
    - [`b.grad` 계산 -- 두 경로의 합산](#bgrad-계산----두-경로의-합산)
    - [최종 결과](#최종-결과)
    - [수학적 검증](#수학적-검증)
- [핵심 통찰 정리](#핵심-통찰-정리)
  - [왜 체인 룰이 Autograd 를 가능하게 하는가?](#왜-체인-룰이-autograd-를-가능하게-하는가)
  - [grad 와 \_local\_grads 의 관계 한 줄 요약](#grad-와-_local_grads-의-관계-한-줄-요약)
  - [`+=` 가 반드시 필요한 이유](#-가-반드시-필요한-이유)
  - [학습률이 작아야 하는 이유](#학습률이-작아야-하는-이유)

--------

# Abstract

- 체인 룰(Chain Rule)은 합성 함수의 미분 규칙이다.
- 신경망의 역전파(backpropagation)가 수학적으로 가능한 이유가 바로 체인 룰이다.
- `_local_grads`(forward 에서 저장) 와 `grad`(backward 에서 계산) 의 관계를
  체인 룰로 설명한다.

# Materials

- [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) | gist
- [micrograd](https://github.com/karpathy/micrograd) | github

# 체인 룰이란?

함수가 중첩되어 있을 때 미분하는 규칙이다.

```
L = f(g(x)) 라면:

dL/dx = (dL/dg) × (dg/dx)
         ↑              ↑
    g 에 대한 L 의 미분  x 에 대한 g 의 미분
    (backward 에서 옴)   (local_grad, forward 에서 저장)
```

> "최종 변화량 = 중간 변화량들의 연쇄 곱"

현실 비유: 서울 물가가 오르면 부산 물가도 오른다. 서울 물가가 1% 오를 때 부산이
0.8% 오르고, 부산 물가가 1% 오를 때 대구가 0.6% 오른다면, 서울이 1% 오를 때
대구는 0.8 × 0.6 = 0.48% 오른다. 중간 단계의 민감도를 곱해 나가는 것이 체인 룰이다.

# forward pass 와 backward pass 의 역할 분담

```
forward:  _local_grads 에 국소 미분값을 스냅샷 저장
backward: _local_grads × upstream_grad 를 체인으로 곱해 나감
```

| 단계 | 계산하는 것 | 저장 위치 | 수식 |
|------|-----------|----------|------|
| forward | 각 연산의 국소 미분 | `_local_grads` | `∂output / ∂input` |
| backward | 최종 loss 에 대한 전역 미분 | `grad` | `∂loss / ∂this` |

`grad` 와 `_local_grads` 의 관계:

```
child.grad += local_grad × v.grad
               ↑               ↑
          _local_grads 에     upstream 에서
          저장된 국소 미분     흘러온 grad
                    (체인 룰)
```

# 예제: `L = (a * b) + b`, a=2, b=3

## 계산 그래프

```
a(2) ─┐
      ├─→  c = a*b  (6)  ─┐
b(3) ─┘                    ├─→  L = c+b  (9)
b(3) ──────────────────────┘
```

## forward pass: `_local_grads` 스냅샷 저장

각 연산이 실행되는 순간 국소 미분값을 계산하여 저장한다.

```python
c = a * b
# c._children    = (a,      b    )
# c._local_grads = (∂c/∂a, ∂c/∂b) = (b.data, a.data) = (3.0, 2.0)

L = c + b
# L._children    = (c,      b    )
# L._local_grads = (∂L/∂c, ∂L/∂b) = (1,      1    )
```

**왜 `b.data = 3.0` 을 스냅샷으로 저장하는가?** `c = a * b` 의 미분은 `∂c/∂a = b`
인데, backward 시점에는 `b` 가 이미 다른 값으로 바뀌어 있을 수 있다. forward 가
실행되는 **그 순간의 `b.data`** 를 `_local_grads` 에 저장해 두어야 나중에
정확한 미분을 계산할 수 있다.

## backward pass: 체인 룰로 `grad` 계산

### 초기화

```python
L.grad = 1   # 출발점: ∂L/∂L = 1
```

### `a.grad` 계산 -- 체인 룰 전개

```
a → c → L  경로 하나만 있음

∂L/∂a = (∂L/∂c) × (∂c/∂a)
          ↑              ↑
       c.grad          c._local_grads[0]
       (backward 에서    (forward 에서
        L 처리 후 채워짐) 3.0 으로 저장됨)
```

단계별 계산:

```
① L 처리: c.grad += L._local_grads[0] × L.grad = 1 × 1 = 1.0
② c 처리: a.grad += c._local_grads[0] × c.grad = 3.0 × 1.0 = 3.0
```

코드 대응:

```python
# backward 루프
child.grad += local_grad * v.grad

# ① v=L, child=c:  c.grad += 1   × 1   = 1.0
# ② v=c, child=a:  a.grad += 3.0 × 1.0 = 3.0
```

### `b.grad` 계산 -- 두 경로의 합산

`b` 는 두 경로로 `L` 에 기여하므로 각 경로의 기울기를 `+=` 로 합산한다.

```
경로 1: b ──(덧셈)──→ L          경로 2: b ──(곱셈)──→ c ──(덧셈)──→ L
  ∂L/∂b = 1 × L.grad = 1.0        ∂L/∂b = (∂c/∂b) × c.grad
                                          = 2.0     × 1.0 = 2.0

b.grad = 경로 1 + 경로 2 = 1.0 + 2.0 = 3.0
```

코드 대응:

```python
# ① v=L, child=b:  b.grad += 1   × 1   = 1.0   ← 경로 1
# ② v=c, child=b:  b.grad += 2.0 × 1.0 = 2.0   ← 경로 2
#                  b.grad = 3.0
```

### 최종 결과

| 노드 | data | grad | 해석 |
|------|------|------|------|
| `a` | 2.0 | **3.0** | a 를 1 늘리면 L 이 3 늘어남 |
| `b` | 3.0 | **3.0** | b 를 1 늘리면 L 이 3 늘어남 (두 경로 합산) |
| `c` | 6.0 | 1.0 | c 를 1 늘리면 L 이 1 늘어남 |
| `L` | 9.0 | 1.0 | 자기 자신 |

### 수학적 검증

```
L = a*b + b 를 직접 편미분:

∂L/∂a = b = 3.0                    → a.grad = 3.0 ✓
∂L/∂b = a + 1 = 2.0 + 1 = 3.0     → b.grad = 3.0 ✓
```

실제로 값을 바꿔서도 확인:

```
기준: a=2, b=3  →  L = 2*3 + 3 = 9

a 를 2 → 3 으로 바꾸면:  L = 3*3 + 3 = 12  →  9 에서 3 증가  (a.grad=3.0 ✓)
b 를 3 → 4 으로 바꾸면:  L = 2*4 + 4 = 12  →  9 에서 3 증가  (b.grad=3.0 ✓)
```

# 핵심 통찰 정리

## 왜 체인 룰이 Autograd 를 가능하게 하는가?

사람이 `∂L/∂a` 를 수식으로 직접 유도하려면:

1. `L = a*b + b` 라는 전체 수식을 알아야 한다.
2. `a` 에 대해 편미분한다.

신경망은 수백만 개의 파라미터와 수십 개의 레이어가 중첩되어 있어서 전체 수식을
직접 유도하는 것이 불가능하다.

체인 룰을 쓰면:

1. 각 연산(`+`, `*`, `relu`, ...)은 자신의 **국소 미분만** 알면 된다.
2. backward 가 이것들을 체인으로 곱해 나가면 전체 미분이 자동으로 완성된다.

```
국소 미분 (연산별로 이미 알려진 수식)
  덧셈:  ∂(a+b)/∂a = 1,  ∂(a+b)/∂b = 1
  곱셈:  ∂(a*b)/∂a = b,  ∂(a*b)/∂b = a
  relu:  ∂relu(a)/∂a = 1 if a>0 else 0
  ...

체인 룰 (backward 루프)
  child.grad += local_grad × v.grad

→ 어떤 복잡한 수식도 자동으로 미분 가능
```

## grad 와 _local_grads 의 관계 한 줄 요약

```
grad       = ∂loss / ∂this          전역 민감도  (backward 후 채워짐)
_local_grad = ∂this / ∂child         국소 민감도  (forward 시 저장됨)

grad 전파: child.grad += _local_grad × this.grad
                                (체인 룰 = 두 민감도의 곱)
```

## `+=` 가 반드시 필요한 이유

하나의 노드가 여러 경로로 loss 에 기여할 때, 각 경로의 기울기를 모두 더해야
올바른 전체 기울기가 된다. `b` 가 두 경로로 `L` 에 기여하는 위 예제에서:

```
b.grad  = (경로 1 기여분) + (경로 2 기여분)
        = 1.0 + 2.0 = 3.0

만약 += 대신 = 를 쓰면:
  b.grad = 2.0  (마지막 경로만 남아서 틀린 결과)
```

## 학습률이 작아야 하는 이유

`grad` 는 엄밀히 `ε → 0` 극한에서의 비율이다. ReLU 처럼 비선형 연산이 있으면
큰 변화에서는 선형 근사가 깨진다. Adam 이 `lr = 0.01` 처럼 작은 값을 쓰는 이유가
이것이다. `grad` 가 알려주는 방향은 맞지만, 한 번에 너무 크게 움직이면 그 방향이
더 이상 유효하지 않게 된다.
