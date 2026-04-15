# Chapter 5: 식별 (Identification)

> **핵심 질문**: 데이터에서 **어떤 변동(variation)**이 내 질문에 답하는가?

---

## 5.1 데이터 생성 과정 (Data Generating Process, DGP)

**DGP** = 관찰된 데이터를 만들어낸 **숨겨진 법칙들의 집합**

### 물리학 비유

뉴턴은 행성 궤도 데이터를 관찰하고, 그 뒤에 숨겨진 법칙(중력)을 발견했다.
사회과학도 마찬가지다 — 데이터 뒤에는 법칙(DGP)이 있고, 우리는 그것을
알아내려 한다.

### 시뮬레이션 예제: 머리색과 소득

```
DGP (진짜 규칙):
  1. 소득은 로그정규분포
  2. 갈색 머리 → 소득 +10% (진짜 효과)
  3. 자연적으로 갈색 머리: 20%
  4. 대졸 → 소득 +20%
  5. 대졸 비율: 30%
  6. 비대졸 + 비갈색 머리 중 40%가 염색 → 갈색 머리
```

**단순 분석 결과**: 갈색 머리 vs 아닌 머리 소득 차이 = **+1%** (진짜는 +10%)

**왜?** 비대졸자(소득 낮음)가 염색을 많이 해서 갈색 머리 그룹의 평균 소득을 깎았다.

**해결**: DGP를 이해하고 → 대졸자만 분석(염색 없는 그룹) → **+13%** (진짜에 가까움)

**교훈**: DGP를 모르면, 데이터에서 올바른 답을 꺼낼 수 없다.

---

## 5.2 변동이 어디 있는가? (Where's Your Variation?)

### 아보카도 가격 예제

캘리포니아 아보카도 데이터: 가격이 높을 때 판매량이 낮다.

**이건 뭘 의미하는가?**
- (a) 가격이 올라서 사람들이 안 산다? (수요)
- (b) 많이 팔려서 가격이 올라갔다? (공급)
- (c) 둘 다?

같은 데이터인데 해석이 완전히 다르다.

### 변동을 분리하기

**가정**: 판매자는 월 단위로 가격을 정하고, 월 중에는 바꾸지 않는다.

그러면:
- **월 간 변동** = 판매자의 결정 (공급 측)
- **월 내 변동** = 소비자의 반응 (수요 측)

월 간 변동을 제거하면 → 소비자 행동만 남는다 → "가격이 올라가면 덜 산다"

> **"연구 질문에 답하는 작업 = 올바른 변동을 찾는 작업"**

---

## 5.3 식별이란?

> **식별(Identification)** = 데이터의 변동 중에서, 연구 질문에 답하는 부분을
> 골라내는 과정

비유: 개가 탈출하는 경로를 찾기

```
1. 모든 가능한 출구를 나열한다 (강아지문, 뒷문, 창문, 환기구...)
2. 하나씩 차단한다
3. 차단한 상태에서 개가 여전히 탈출하면 → 그 경로가 아님
4. 하나만 남기고 다 차단 → 개가 탈출 → 그 경로가 범인!
```

연구에서도 마찬가지:
1. DGP를 가능한 한 정확히 그린다
2. 질문과 무관한 이유(대안 설명)를 파악한다
3. 그 대안 설명들을 **차단**하고 남은 변동을 분석한다

---

## 5.4 실전 사례: 음주와 사망률

Wood et al. (2018), *The Lancet*: 599,912명 분석

**발견**: 주당 알코올 ~100g(하루 1잔) 이상에서 사망 위험 증가

### DGP에서의 교란 변수

| 교란 변수 | 음주와의 관계 | 사망률과의 관계 |
|----------|-------------|--------------|
| 흡연 | 흡연자가 더 많이 마심 | 흡연이 사망 원인 |
| 위험 추구 성향 | 위험을 즐기는 사람이 더 마심 | 위험 행동이 사망 원인 |
| 건강 상태 | 너무 아파서 못 마시는 사람 존재 | 질병이 사망 원인 |

### 연구가 한 것

- 비음주자 제외 (병이 심하거나 과거 알코올 중독자 제거)
- 흡연, 나이, 성별, BMI, 당뇨 통제

### Chris Auld의 반박

같은 데이터, 같은 방법으로 분석했더니: **"음주가 남성일 확률을 높인다"** 는
결과가 나옴. 이건 명백히 말이 안 되는 결론이다.

**교훈**: 그럴듯해 보이는 결과라도 식별이 제대로 안 됐을 수 있다.
> **"그럴듯함(plausibility) ≠ 식별(identification)"**

---

## 5.5 맥락과 전지적 시점

> *"진보는 세부적인 제도 지식과, 특정 환경에서 작용하는 힘에 대한 신중한
> 조사에서 온다."* — Angrist & Krueger

연구 질문에 답하려면 **맥락(context)**을 깊이 이해해야 한다:
- 관련 책을 읽고
- 정책 문서를 검토하고
- 현장 사람들과 이야기하고
- 디테일을 파악해야 한다

인과추론은 **기법만으로 되는 것이 아니라, 도메인 지식이 필수**다.

---

## Python 예제: DGP를 알 때 vs 모를 때

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000

# === DGP (진짜 규칙) ===
has_college = np.random.binomial(1, 0.3, n)  # 30% 대졸
natural_brown = np.random.binomial(1, 0.2, n)  # 20% 자연 갈색머리

# 비대졸 + 비갈색 중 40%가 염색
dyed = np.where(
    (has_college == 0) & (natural_brown == 0),
    np.random.binomial(1, 0.4, n),
    0
)
brown_hair = np.where(natural_brown | dyed, 1, 0)

# 소득: 갈색머리 +10%, 대졸 +20%
log_income = (
    10
    + 0.10 * natural_brown  # 자연 갈색만 진짜 효과 (염색은 효과 없음!)
    + 0.20 * has_college
    + np.random.normal(0, 0.5, n)
)
income = np.exp(log_income)

df = pd.DataFrame({
    'brown_hair': brown_hair,
    'natural_brown': natural_brown,
    'has_college': has_college,
    'income': income
})

# === 분석 1: DGP를 모름 (단순 비교) ===
print("=== DGP 모름: 단순 비교 ===")
mean_brown = df[df['brown_hair'] == 1]['income'].mean()
mean_other = df[df['brown_hair'] == 0]['income'].mean()
print(f"갈색머리 평균 소득: ${mean_brown:,.0f}")
print(f"기타 평균 소득:     ${mean_other:,.0f}")
print(f"차이: {(mean_brown/mean_other - 1)*100:.1f}%")
print("→ 진짜 효과(+10%)보다 훨씬 작다!")

# === 분석 2: DGP를 알고, 대졸자만 분석 ===
college = df[df['has_college'] == 1]
print("\n=== DGP 앎: 대졸자만 분석 (염색 없는 그룹) ===")
mean_b_col = college[college['brown_hair'] == 1]['income'].mean()
mean_o_col = college[college['brown_hair'] == 0]['income'].mean()
print(f"갈색머리 대졸 평균: ${mean_b_col:,.0f}")
print(f"기타 대졸 평균:     ${mean_o_col:,.0f}")
print(f"차이: {(mean_b_col/mean_o_col - 1)*100:.1f}%")
print("→ 진짜 효과(+10%)에 가깝다!")

# === 분석 3: 회귀로 학력 통제 ===
import statsmodels.formula.api as smf
model = smf.ols('np.log(income) ~ brown_hair + has_college', data=df).fit()
print(f"\n=== 회귀로 학력 통제 ===")
print(f"갈색머리 계수: {model.params['brown_hair']:.4f}")
print(f"대졸 계수:     {model.params['has_college']:.4f}")
print("→ 갈색머리 계수가 0.10(=10%)보다 작다. 왜? 염색한 사람 때문!")
print("→ DGP를 정확히 모르면, 통제해도 편향이 남을 수 있다")
```

**핵심 교훈**: 어떤 통계 기법을 쓰든, **DGP(데이터가 어떻게 만들어졌는지)**를
이해하지 못하면 올바른 답을 얻을 수 없다.
