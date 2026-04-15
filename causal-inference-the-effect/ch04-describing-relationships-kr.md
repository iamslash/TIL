# Chapter 4: 관계 설명하기 (Describing Relationships)

> **핵심 질문**: 두 변수 사이의 관계를 어떻게 설명하고, "다른 변수를 통제한다"는 것은 무엇인가?

---

## 4.1 관계란?

**관계(Relationship)** = "한 변수를 알면 다른 변수에 대해 무엇을 알 수 있는가"

예: 아이의 **나이**를 알면 **키**를 어느 정도 예측할 수 있다 → 양의 관계

| 관계 유형 | 설명 | 예시 |
|----------|------|------|
| 양의 관계 | X↑ → Y↑ | 나이↑ → 키↑ |
| 음의 관계 | X↑ → Y↓ | 운동량↑ → 체중↓ |
| 무관계 | X와 Y 독립 | 신발 사이즈 → 수학 점수 |

### 산점도 (Scatterplot)

X축과 Y축에 모든 데이터 포인트를 찍은 그래프. 관계의 방향과 형태를 직관적으로
볼 수 있다.

- **장점**: 모든 정보를 보여줌
- **단점**: 데이터가 많으면 읽기 어려움, 인과관계로 착각하기 쉬움

---

## 4.2 조건부 분포 (Conditional Distribution)

- **무조건부 분포**: "사람이 여성일 확률은 약 50%"
- **조건부 분포**: "이름이 Sarah인 사람이 여성일 확률은 50%보다 훨씬 높다"

### Emily Oster의 비타민E 연구

비타민E가 건강에 좋다는 권고가 있었다. 하지만:
- 비타민E를 복용하는 사람은 **원래 건강에 관심이 많은 사람**
- 운동도 하고, 담배도 안 피우고, 식단도 관리하는 사람
- 비타민E의 효과인가, 건강한 생활습관의 효과인가?

이것이 바로 **교란 변수(Confounder)** 문제다.

---

## 4.3 조건부 평균 (Conditional Mean)

**조건부 평균** = "X가 특정 값일 때 Y의 평균"

### 범주형 X일 때

간단히 그룹별 평균을 구하면 된다:

```python
# 비타민E 권고 시기별 복용률
df.groupby('recommendation_period')['takes_vitamin_e'].mean()
# 권고 전:  12%
# 권고 중:  25%
# 권고 후:  15%
```

### 연속형 X일 때

정확히 같은 X값을 가진 관측치가 거의 없으므로, 두 가지 방법을 쓴다:

#### 방법 1: 구간 나누기 (Binning)

X를 구간으로 잘라서 구간별 평균을 구한다.

```python
df['bmi_bin'] = pd.cut(df['bmi'], bins=10)
df.groupby('bmi_bin')['vitamin_e'].mean()
```

단점: 구간을 어떻게 나누느냐에 따라 결과가 달라진다.

#### 방법 2: LOESS (국소 회귀)

각 X 지점 근처의 데이터에 가중치를 두고 부드러운 곡선을 그린다.
구간 나누기보다 자연스럽고, 데이터가 적은 구간도 잘 처리한다.

---

## 4.4 회귀선 맞추기 (Line-Fitting)

### OLS 회귀 (Ordinary Least Squares)

Y = β₀ + β₁X 형태의 직선을 데이터에 맞추는 가장 일반적인 방법.

**원리**: 모든 데이터 포인트와 직선 사이의 **거리(잔차)의 제곱합**을 최소화하는
직선을 찾는다.

```
잔차 (Residual) = 실제값 - 예측값

예: 회귀선이 X=5에서 Y=10을 예측하는데, 실제 Y=13이면
    잔차 = 13 - 10 = 3
```

### 기울기의 의미

```
비타민E = 0.110 + 0.002 × BMI
```

해석: BMI가 1 단위 올라가면, 비타민E 복용 확률이 **0.2%p** 증가

### 상관계수 (Correlation)

OLS 기울기를 -1 ~ +1 사이로 표준화한 것.
- +1: 완벽한 양의 관계
- 0: 무관계
- -1: 완벽한 음의 관계

**장점**: 단위 무관, 비교 용이
**단점**: 원래 단위로 해석 불가

### 비선형 관계

OLS도 곡선을 맞출 수 있다:

```python
# 2차 함수 (포물선)
Y = β₀ + β₁·X + β₂·X²

# 로그 관계
Y = β₀ + β₁·log(X)
```

"계수에 대해 선형(linear in coefficients)"이면 OLS로 풀 수 있다.

---

## 4.5 변수 통제하기 (Controlling for a Variable)

이 섹션이 **인과추론의 핵심 아이디어**로 연결되는 가장 중요한 부분이다.

### 문제: 아이스크림과 반바지

```
관찰: 아이스크림 매출이 높은 날 → 반바지 판매도 높다
질문: 아이스크림이 반바지를 사게 만드는가?
현실: 기온이 둘 다에 영향
```

### 해결: 기온을 "통제"한다

"통제한다(Controlling for)" = **그 변수에 의한 변동을 모든 다른 변수에서 제거한다**

단계별로 설명하면:

```
1단계: 기온이 아이스크림 매출을 얼마나 설명하는지 계산
2단계: 아이스크림 매출에서 기온이 설명하는 부분을 뺀다 → "잔차 아이스크림"
3단계: 기온이 반바지 판매를 얼마나 설명하는지 계산
4단계: 반바지 판매에서 기온이 설명하는 부분을 뺀다 → "잔차 반바지"
5단계: "잔차 아이스크림"과 "잔차 반바지"의 관계를 본다
```

만약 기온을 제거했더니 관계가 사라지면 → **기온이 전부 설명** (아이스크림 ≠ 원인)
만약 관계가 남아있으면 → 기온 외에 다른 연결고리가 있을 수 있음

### 회귀에서의 통제

실무에서는 위 5단계를 일일이 안 하고, **변수를 추가**하면 된다:

```python
# 통제 없이
model1 = smf.ols('vitamin_e ~ bmi', data=df).fit()
# → BMI 계수: 0.002

# 나이와 성별을 통제
model2 = smf.ols('vitamin_e ~ bmi + age + female', data=df).fit()
# → BMI 계수: 0.001  (줄어들었다!)
# → 나이, 성별이 BMI-비타민E 관계의 일부를 설명했다는 뜻
```

### 실전 적용 예시

신규 기능의 효과를 분석할 때 바로 이 방법을 쓴다:

```python
# 기능 채택 효과 분석 (개념적)
model = smf.ols(
    'outcome ~ feature_adopted + activity + age + gender + account_age + subscriber',
    data=df
).fit()
# feature_adopted의 계수 = 활동량, 나이, 성별 등을 통제한 후의 순수 기능 효과
```

---

## Python 예제: 통제 변수의 힘

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)
n = 1000

# 진실: 기온이 아이스크림과 반바지 판매에 모두 영향
temperature = np.random.normal(25, 8, n)  # 기온 (섭씨)
ice_cream = 50 + 3 * temperature + np.random.normal(0, 10, n)
shorts = 20 + 2 * temperature + np.random.normal(0, 8, n)
# 아이스크림은 반바지에 직접 영향을 주지 않는다!

df = pd.DataFrame({
    'temperature': temperature,
    'ice_cream': ice_cream,
    'shorts': shorts
})

# --- 통제 없이 ---
model1 = smf.ols('shorts ~ ice_cream', data=df).fit()
print("=== 통제 없는 회귀 ===")
print(f"아이스크림 계수: {model1.params['ice_cream']:.4f}")
print(f"p-value: {model1.pvalues['ice_cream']:.6f}")
print("→ 아이스크림 매출이 올라가면 반바지도 올라간다? (거짓!)")

# --- 기온을 통제 ---
model2 = smf.ols('shorts ~ ice_cream + temperature', data=df).fit()
print("\n=== 기온 통제 후 ===")
print(f"아이스크림 계수: {model2.params['ice_cream']:.4f}")
print(f"p-value: {model2.pvalues['ice_cream']:.4f}")
print(f"기온 계수: {model2.params['temperature']:.4f}")
print("→ 기온을 통제하면 아이스크림 효과가 거의 사라진다!")
print("→ 진짜 원인은 기온이었다")
```

**실행 결과 해석:**
- 통제 없이: 아이스크림 계수 ≈ 0.6, 매우 유의 (p ≈ 0)
- 기온 통제 후: 아이스크림 계수 ≈ 0.0, 유의하지 않음
- **기온이라는 교란 변수를 통제하자, 거짓 관계가 사라졌다**

이것이 통제 변수의 핵심이며, 인과추론 전체의 출발점이다.
