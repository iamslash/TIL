# Chapter 16: 고정효과 (Fixed Effects)

> **핵심 질문**: 관찰 불가능한 교란 변수가 **시간에 따라 변하지 않으면**, 어떻게 통제하는가?

---

## 16.1 핵심 아이디어

> **고정효과** = 그룹 자체를 통제하면, 그 그룹 내에서 변하지 않는
> **모든 변수가 자동으로 통제**된다 (관찰 불가능해도!)

```
예: 전기가 농촌 생산성에 미치는 영향
교란: 지리(토양, 기후, 수자원...) → 생산성에도 영향
     지리가 전기 인프라에도 영향

해결: 같은 마을의 전기 도입 전후를 비교
→ 마을 고정효과 = 마을 내 불변 특성 모두 통제
→ 토양, 기후, 수자원 등 측정 안 해도 됨
```

---

## 16.2 Between vs Within 변동

| | Between 변동 | Within 변동 |
|--|-------------|-------------|
| 의미 | 개체 **간** 차이 | 같은 개체의 **시간별** 변화 |
| 예시 | A는 B보다 키가 크다 | A가 작년보다 올해 키가 컸다 |
| 교란 위험 | 높음 (개체 특성 차이) | 낮음 (개체 특성 통제됨) |

**고정효과는 Between 변동을 버리고, Within 변동만 사용한다.**

---

## 16.3 구현 방법

### 방법 1: 평균 차감 (De-meaning)

```python
# 각 개체의 평균을 빼서, "개체 내 변동"만 남긴다
Y_it - Ȳ_i = β₁(X_it - X̄_i) + (ε_it - ε̄_i)

# 개체 평균을 빼면:
# - 성격, 유전, 성장배경 등 불변 특성이 모두 사라짐
# - 남는 것은 "이 사람이 평소보다 X가 높을 때, Y도 높은가?"
```

### 방법 2: 더미 변수

```python
# 각 개체마다 더미 변수 추가
Y = β₀ + β₁X + δ₁·개체2 + δ₂·개체3 + ... + ε

# 결과는 방법 1과 동일
# 단, 개체가 10만 명이면 더미가 10만 개 → 비효율적
```

### Stata/R/Python 구현

```python
import statsmodels.formula.api as smf

# 패널 데이터에서 고정효과
model = smf.ols('outcome ~ treatment + C(individual_id) + C(year)', data=df).fit()
# C(individual_id) = 개체 고정효과
# C(year) = 시간 고정효과
```

---

## 16.4 이원 고정효과 (Two-Way Fixed Effects)

개체 고정효과 + 시간 고정효과를 동시에:

```
Y_it = α_i + γ_t + β₁·Treatment_it + ε_it

α_i = 개체 고정효과 (개체 내 불변 특성 통제)
γ_t = 시간 고정효과 (모든 개체에 공통인 시간 효과 통제)
```

**이것이 바로 DiD(Ch18)의 기본 형태다!**

---

## 16.5 한계

1. **시간에 따라 변하는 교란 변수는 통제 못함**
2. **불변 변수의 효과는 추정 불가** (고정효과가 흡수)
3. **Within 변동이 적으면 추정이 부정확**

---

## 실전 적용: 기능 채택 분석에서의 이원 고정효과

DiD는 이원 고정효과의 응용이다:

```
유저 고정효과 → 유저의 불변 특성 (동기, 성격 등) 통제
시간 고정효과 → 시간에 따른 공통 변화 (계절, 앱 업데이트) 통제
남는 것 → 기능 채택에 의한 순수 변화
```

---

## Python 예제: 고정효과

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)

# 10개 마을, 5년간 관찰
n_towns = 10
n_years = 5
n = n_towns * n_years

town_id = np.repeat(range(n_towns), n_years)
year = np.tile(range(n_years), n_towns)

# 마을별 고정 특성 (관찰 불가)
town_quality = np.repeat(np.random.normal(0, 5, n_towns), n_years)

# 전기 도입: 좋은 마을이 먼저 도입 (교란!)
electricity = ((town_quality + year * 2) > 5).astype(float)

# 생산성: 전기 효과 = +3, 마을 품질 효과 = +2, 시간 추세 = +1
productivity = 10 + 3 * electricity + 2 * town_quality + 1 * year + np.random.normal(0, 2, n)

df = pd.DataFrame({
    'town': town_id, 'year': year,
    'electricity': electricity, 'productivity': productivity
})

# === 잘못: 고정효과 없이 ===
m1 = smf.ols('productivity ~ electricity', data=df).fit()
print(f"=== 고정효과 없이 ===")
print(f"전기 계수: {m1.params['electricity']:.2f} (진짜: 3.0)")
print(f"→ 좋은 마을이 먼저 전기 도입 → 과대추정!")

# === 올바름: 이원 고정효과 ===
m2 = smf.ols('productivity ~ electricity + C(town) + C(year)', data=df).fit()
print(f"\n=== 이원 고정효과 ===")
print(f"전기 계수: {m2.params['electricity']:.2f} (진짜: 3.0)")
print(f"→ 마을 + 시간 고정효과로 편향 제거!")
```
