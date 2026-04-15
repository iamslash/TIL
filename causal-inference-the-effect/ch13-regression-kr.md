# Chapter 13: 회귀분석 (Regression)

> **핵심 질문**: 회귀분석으로 백도어를 어떻게 닫고, 결과를 어떻게 해석하는가?

---

## 13.1 회귀분석의 기본

### OLS 회귀란?

```
Y = β₀ + β₁X + ε
```

- **β₀**: 절편 (X=0일 때 Y의 기대값)
- **β₁**: 기울기 (X가 1 단위 증가할 때 Y의 변화)
- **ε**: 오차항 (모델이 설명하지 못하는 나머지)

OLS는 **잔차(residual)의 제곱합을 최소화**하는 직선을 찾는다.

### 다중 회귀 = 백도어 닫기

```python
# 인과 다이어그램에서 회귀식으로 번역
# Treatment → Outcome ← Confounder → Treatment (백도어)

Y = β₀ + β₁·Treatment + β₂·Confounder + ε
#         ↑ 이 계수가 인과 효과 (백도어 닫힌 후)
```

**핵심 가정: 외생성 (Exogeneity)**
- 포함한 변수들이 오차항(ε)과 **무상관**이어야 함
- = 인과 다이어그램에서 **모든 백도어가 닫혀야** 함
- 위반되면 → **누락 변수 편향 (Omitted Variable Bias)**

### 표본 변동과 추론

OLS 계수의 **표준오차(Standard Error)**:

```
SE(β₁) ≈ σ / (sd(X) × √n)

줄이는 방법:
  σ ↓  (오차 분산 줄임 = 통제 변수 추가)
  sd(X) ↑  (X의 변동이 클수록)
  n ↑  (표본 크기 클수록)
```

### 가설 검정

```
귀무가설: β₁ = 0 (효과 없음)
t통계량 = β₁ / SE(β₁)
p-value = |t|가 이렇게 클 확률 (귀무가설 하에서)

p < 0.05 → 통계적으로 유의 (**)
p < 0.01 → 매우 유의 (***)
```

**주의사항:**
- 유의하지 않다 ≠ 효과가 없다 (표본이 작을 수도)
- 유의하다 ≠ 효과가 크다 (표본이 크면 작은 효과도 유의)
- 유의하게 만들기 위해 분석을 바꾸면 → **검정 자체가 무효화**

### 모델 적합도

| 지표 | 의미 |
|------|------|
| R² | Y 변동 중 모델이 설명하는 비율 (0~1) |
| Adjusted R² | 변수 수에 대해 보정된 R² |
| F통계량 | 모든 계수가 동시에 0인지 검정 |
| RMSE | 평균 예측 오차 크기 |

---

## 13.2 고급 회귀 기법

### 이진/범주형 변수

```python
# 이진 변수: 남녀 차이
Y = β₀ + β₁·Female + ...
# β₁ = 여성과 남성의 평균 Y 차이

# 범주형 변수: 교육 수준 (고졸 기준)
Y = β₀ + β₁·대졸 + β₂·석사 + ...
# β₁ = 대졸과 고졸의 차이
# β₂ = 석사와 고졸의 차이
```

### 다항식 (비선형 관계)

```python
Y = β₀ + β₁·X + β₂·X²
# X의 한계효과 = β₁ + 2β₂·X  (X값에 따라 다름!)
```

시각적으로 비선형이 보일 때만 사용. 3차 이상은 거의 불필요.

### 로그 변환

| 변환 | 해석 |
|------|------|
| log(Y) ~ X | X가 1 증가 → Y가 β₁×100 **%** 변화 |
| Y ~ log(X) | X가 1% 증가 → Y가 β₁/100 변화 |
| log(Y) ~ log(X) | X가 1% 증가 → Y가 β₁ **%** 변화 (탄력성) |

### 상호작용 (Interaction)

```python
Y = β₀ + β₁·Treatment + β₂·Group + β₃·Treatment×Group
# β₁ = 기본 그룹에서 Treatment의 효과
# β₃ = Group에 따라 Treatment 효과가 얼마나 다른지
```

---

## Python 예제: 회귀로 백도어 닫기

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)
n = 2000

# DGP: 교육 → 소득, 능력 → 교육 & 소득 (백도어)
ability = np.random.normal(0, 1, n)
education = 12 + 2 * ability + np.random.normal(0, 2, n)
income = 20000 + 3000 * education + 5000 * ability + np.random.normal(0, 8000, n)

df = pd.DataFrame({
    'ability': ability,
    'education': education,
    'income': income
})

# === 잘못: 능력 통제 안 함 (백도어 열림) ===
m1 = smf.ols('income ~ education', data=df).fit()
print("=== 능력 미통제 ===")
print(f"교육 계수: ${m1.params['education']:,.0f}")
print(f"(진짜는 $3,000이지만 능력 편향 때문에 과대추정)")

# === 올바름: 능력 통제 (백도어 닫힘) ===
m2 = smf.ols('income ~ education + ability', data=df).fit()
print(f"\n=== 능력 통제 ===")
print(f"교육 계수: ${m2.params['education']:,.0f}")
print(f"능력 계수: ${m2.params['ability']:,.0f}")
print(f"→ 교육의 진짜 효과 $3,000에 가깝다!")

# === 상호작용: 교육 효과가 성별로 다른가? ===
gender = np.random.binomial(1, 0.5, n)
income_gendered = income + 2000 * education * gender  # 여성의 교육 수익률이 더 높음
df['female'] = gender
df['income_g'] = income_gendered

m3 = smf.ols('income_g ~ education * female + ability', data=df).fit()
print(f"\n=== 상호작용 모델 ===")
print(f"교육 기본 효과 (남성): ${m3.params['education']:,.0f}")
print(f"교육×여성 상호작용:   ${m3.params['education:female']:,.0f}")
print(f"→ 여성은 교육 1년당 ${m3.params['education'] + m3.params['education:female']:,.0f} 소득 증가")
```
