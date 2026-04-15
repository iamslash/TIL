# Chapter 14: 매칭 (Matching)

> **핵심 질문**: 회귀 대신, 비슷한 사람끼리 직접 짝지어 비교하면 어떨까?

---

## 14.1 매칭이란?

회귀가 **수식으로** 교란 변수를 통제한다면, 매칭은 **비슷한 사람끼리 골라서** 비교.

```
처치군: 25세 여성, 무료 유저, 주 50회 스와이프
         ↕ 매칭
대조군: 26세 여성, 무료 유저, 주 48회 스와이프
```

### 실전 예시: 앱 기능 채택 분석

```
기능 채택 유저와 비슷한 특성의 비채택 유저를 12,000쌍 매칭
→ 매칭된 쌍끼리 비교
→ 인게이지먼트 +3.8%, 전환율 +6.0%
```

---

## 14.2 매칭의 5가지 결정

### 1. 매칭 기준: 거리 vs 성향 점수

| 방법 | 원리 | 장단점 |
|------|------|--------|
| **거리 매칭** | 변수값 차이가 작은 쌍 | 직관적, 변수 많으면 어려움 |
| **성향 점수 매칭** | 처치 확률이 비슷한 쌍 | 여러 변수를 하나로 압축 |

**성향 점수 (Propensity Score)**:
```python
# 로짓 회귀로 처치 확률 추정
P(치료|X) = logit(β₀ + β₁·나이 + β₂·성별 + β₃·활동량 + ...)

# 처치 확률이 비슷한 사람끼리 매칭
# "이 두 사람은 기능을 채택할 확률이 비슷했는데, 하나는 채택하고 하나는 안 했다"
# → 마치 자연 실험처럼 비교 가능
```

### 2. 선택 vs 가중치

| 방식 | 원리 | 특징 |
|------|------|------|
| **선택** | 매칭되면 1, 안 되면 0 | 직관적, 노이즈 있음 |
| **가중치** | 비슷할수록 높은 가중치 | 통계적 성질 좋음 |

**역확률 가중치 (Inverse Probability Weighting, IPW)**:
```
처치군: 가중치 = 1/P(처치)
대조군: 가중치 = 1/(1-P(처치))

직관: 처치 확률이 낮은데 처치 받은 사람 = 가중치 큼 (귀한 정보)
     처치 확률이 높은데 처치 안 받은 사람 = 가중치 큼
```

### 3. 몇 명을 매칭할 것인가?

- 1:1 매칭: 매칭 품질 높음, 분산 큼
- k:1 매칭: 분산 줄지만, 먼 매칭 포함

### 4. 캘리퍼 (최대 허용 거리)

너무 먼 매칭은 제외. 좁은 캘리퍼 = 더 좋은 매칭, 더 작은 표본.

### 5. 균형 확인

매칭 후, 처치군과 대조군의 변수 분포가 비슷한지 확인:
```
표준화 차이 = (처치군 평균 - 대조군 평균) / 표준편차
→ 0에 가까울수록 좋은 매칭
```

---

## 14.3 다변량 매칭

### 마할라노비스 거리 (Mahalanobis Distance)

여러 변수를 동시에 매칭할 때 사용. 변수를 표준화하고 상관관계를 보정한 거리.

### 차원의 저주 (Curse of Dimensionality)

매칭 변수가 많아질수록 "가까운" 매칭을 찾기 어려워진다.

### 대안: Coarsened Exact Matching

연속형 변수를 구간으로 나누고, 모든 구간이 같은 관측치끼리 정확히 매칭.
```
나이: 20-25, 25-30, 30-35...
소득: 저/중/고
→ "25-30세, 중소득, 여성" 구간에서 처치/대조를 비교
```

### Entropy Balancing

매칭 대신, 대조군에 가중치를 부여하여 평균/분산/왜도가 처치군과 같아지게 함.

---

## 14.4 매칭 vs 회귀

| | 회귀 | 매칭 |
|--|------|------|
| 가정 | 관계가 선형 | 비슷한 사람끼리 비교 가능 |
| 비선형 | 명시적으로 지정해야 함 | 자동 처리 |
| 효과 유형 | 가중 ATE (분산 가중) | ATE 또는 ATT (선택 가능) |
| 표준오차 | 직접 계산 | 부트스트랩 필요 |

둘 다 **관찰 불가능한 교란 변수는 통제 못함** (동일한 한계).

---

## Python 예제: 성향 점수 매칭

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)
n = 3000

# DGP
age = np.random.normal(30, 8, n)
activity = np.random.normal(50, 15, n)

# 활동적이고 젊은 유저가 기능을 더 많이 채택
treat_logit = -3 + 0.02 * activity + 0.01 * age
treat_prob = 1 / (1 + np.exp(-treat_logit))
treated = np.random.binomial(1, treat_prob, n)

# 결과: 기능 효과 = +2, 활동량 효과 = +0.3, 나이 효과 = -0.1
outcome = 10 + 2 * treated + 0.3 * activity - 0.1 * age + np.random.normal(0, 5, n)

df = pd.DataFrame({
    'age': age, 'activity': activity,
    'treated': treated, 'outcome': outcome
})

# === 단순 비교 (편향) ===
naive = df.groupby('treated')['outcome'].mean()
print(f"=== 단순 비교 ===")
print(f"처치군: {naive[1]:.2f}, 대조군: {naive[0]:.2f}")
print(f"차이: {naive[1] - naive[0]:.2f} (진짜 효과는 2.0)")

# === 성향 점수 매칭 ===
# 1단계: 성향 점수 추정
X = df[['age', 'activity']]
lr = LogisticRegression()
lr.fit(X, df['treated'])
df['pscore'] = lr.predict_proba(X)[:, 1]

# 2단계: 1:1 최근접 매칭
treated_df = df[df['treated'] == 1].reset_index(drop=True)
control_df = df[df['treated'] == 0].reset_index(drop=True)

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control_df[['pscore']])
distances, indices = nn.kneighbors(treated_df[['pscore']])

matched_control = control_df.iloc[indices.flatten()]
att = treated_df['outcome'].mean() - matched_control['outcome'].mean()

print(f"\n=== 성향 점수 매칭 후 ===")
print(f"ATT 추정치: {att:.2f} (진짜 효과: 2.0)")

# 3단계: 균형 확인
print(f"\n=== 균형 확인 ===")
for var in ['age', 'activity']:
    d = (treated_df[var].mean() - matched_control[var].mean()) / df[var].std()
    print(f"{var} 표준화 차이: {d:.3f} (0에 가까울수록 좋음)")
```
