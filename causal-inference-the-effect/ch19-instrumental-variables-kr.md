# Chapter 19: 도구변수 (Instrumental Variables, IV)

> **핵심 질문**: 백도어를 닫을 수 없을 때, **외생적 변동**만 골라 쓸 수 있는가?

---

## 19.1 핵심 아이디어

> 처치(X)의 변동 중에서 **도구변수(Z)로 설명되는 부분만** 사용하여
> 결과(Y)와의 관계를 추정

```
Z (도구변수) → X (처치) → Y (결과)
                  ↑
              Stuff (교란)
                  ↓
                  Y

Z로 인한 X 변동만 분리 → Stuff의 영향 제거
```

### Ch9의 바람-오염 예제가 바로 IV

```
바람 방향 (Z) → 오염 (X) → 운전량 (Y)
                    ↑
               경제활동, 계절 (교란)
                    ↓
                 운전량

바람으로 인한 오염 변동만 분석 → 오염의 인과 효과 추정
```

---

## 19.2 2단계 최소자승법 (2SLS)

```python
# 1단계: 도구변수로 처치를 예측
X̂ = α₀ + α₁·Z + (통제변수) + ε₁

# 2단계: 예측된 처치로 결과를 회귀
Y = β₀ + β₁·X̂ + (통제변수) + ε₂

# β₁ = 인과 효과 추정치 = Cov(Z,Y) / Cov(Z,X)
```

---

## 19.3 세 가지 핵심 가정

### 1. 관련성 (Relevance)

> Z가 X를 **실제로 예측**해야 한다

```
1단계 F-통계량 > 10 → 약한 도구변수 아님 (경험 법칙)
F < 10 → 약한 도구변수 문제 → 추정치 편향, 신뢰구간 의미 없음
```

### 2. 배제 제한 (Exclusion Restriction)

> Z → Y 의 경로가 **오직 X를 통해서만** 존재

```
올바름: 바람 → 오염 → 운전
잘못됨: 바람 → 날씨 → 운전 (바람이 날씨를 통해 직접 운전에 영향)
        → 계절/날씨를 통제해야 함
```

**가장 어렵고 중요한 가정.** 검증 불가능, 도메인 지식으로 정당화해야 함.

### 3. 단조성 (Monotonicity)

> Z가 X에 미치는 영향이 모든 사람에게 **같은 방향**이어야 한다

"거역자(defier)" 없음: 도구변수가 처치를 늘리는 방향인데,
오히려 처치를 줄이는 사람이 없어야 함.

---

## 19.4 IV가 추정하는 것: LATE

> **LATE (Local Average Treatment Effect)** = 도구변수에 **반응하는 사람들**의
> 평균 효과

```
순응자 (Complier): Z에 의해 처치가 바뀌는 사람 → 이들의 효과만 추정
항상 처치 (Always-taker): Z에 상관없이 처치 → 가중치 0
절대 비처치 (Never-taker): Z에 상관없이 미처치 → 가중치 0
```

---

## 19.5 대표적 IV 설계

| 도구변수 | 처치 | 연구 질문 |
|---------|------|----------|
| 군 징병 추첨 번호 | 군 복무 | 군 복무가 소득에 미치는 영향 |
| 판사의 엄격함 | 형량 | 형량이 재범에 미치는 영향 |
| 의무교육법 | 교육 연수 | 교육이 소득에 미치는 영향 |
| 복권 당첨금 | 부 | 부가 건강에 미치는 영향 |

---

## Python 예제: 2SLS

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

np.random.seed(42)
n = 5000

# DGP
motivation = np.random.normal(0, 1, n)  # 관찰 불가 교란
instrument = np.random.binomial(1, 0.4, n)  # 도구변수 (무작위)

# 처치: 도구변수 + 동기에 의해 결정
treatment = 0.5 * instrument + 0.3 * motivation + np.random.normal(0, 0.5, n)

# 결과: 처치 효과 = 2.0, 동기 효과 = 3.0
outcome = 2.0 * treatment + 3.0 * motivation + np.random.normal(0, 2, n)

df = pd.DataFrame({
    'outcome': outcome, 'treatment': treatment,
    'instrument': instrument, 'motivation': motivation
})

# === OLS: 동기 미통제 → 편향 ===
ols = smf.ols('outcome ~ treatment', data=df).fit()
print(f"=== OLS (편향) ===")
print(f"처치 계수: {ols.params['treatment']:.2f} (진짜: 2.0)")
print(f"→ 동기가 교란하여 과대추정")

# === 2SLS: 도구변수 사용 ===
# 1단계
stage1 = smf.ols('treatment ~ instrument', data=df).fit()
print(f"\n=== 1단계: 도구변수 → 처치 ===")
print(f"도구변수 계수: {stage1.params['instrument']:.3f}")
print(f"F-통계량: {stage1.fvalue:.1f} (>10이면 OK)")

# 2단계
df['treatment_hat'] = stage1.fittedvalues
stage2 = smf.ols('outcome ~ treatment_hat', data=df).fit()
print(f"\n=== 2SLS 추정 ===")
print(f"처치 계수: {stage2.params['treatment_hat']:.2f} (진짜: 2.0)")
print(f"→ 도구변수로 교란 제거!")
```
