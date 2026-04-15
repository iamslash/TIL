# Chapter 6: 인과 다이어그램 (Causal Diagrams)

> **핵심 질문**: 인과관계를 어떻게 시각적으로 표현하고 추론하는가?

---

## 6.1 인과관계란?

> **"X가 Y를 유발한다"** = X에 개입하여 X를 바꾸면, 그 결과로 Y의 분포가 변한다

### 인과적 언어 vs 비인과적 언어

| 인과적 (방향 있음) | 비인과적 (방향 없음) |
|------------------|-------------------|
| 유발한다 (causes) | 관련이 있다 (associated) |
| 증가/감소시킨다 | 상관이 있다 (correlated) |
| ~에 영향을 준다 | 함께 나타나는 경향 |
| ~을 결정한다 | 예측한다 (predicts) |

**주의할 표현들 (Weasel Words)**:
- "~와 연결된다 (linked to)" ← 인과인 척하는 비인과 표현
- "X를 한 사람이 Y하는 경향이 있다" ← 마찬가지

### 확률적 인과

인과는 100% 확정적일 필요 없다:
- 아이에게 책을 사주면 읽을 **확률이 높아진다** (안 읽을 수도 있음)
- 페니실린을 넣으면 세균이 **죽을 확률이 높아진다**

---

## 6.2 인과 다이어그램이란?

Judea Pearl (컴퓨터 과학자)이 1990년대에 개발한 도구.

**구성 요소 2개:**
1. **노드 (Node)**: 변수 (동그라미나 텍스트)
2. **화살표 (Arrow)**: 인과 관계 (원인 → 결과)

### 간단한 예: 동전 던지기와 케이크

```
CoinFlip → Cake
```

- CoinFlip 은 하나의 변수 (앞면/뒷면 두 값)
- 화살표는 인과를 나타내지만, 양(+)인지 음(-)인지는 표시 안 함

### 복잡한 예: 케이크, 돈, 친구

```
CoinFlip → Cake
CoinFlip → Money
TerryInRoom → Money
```

- 하나의 변수가 여러 변수에 영향 가능 (CoinFlip → Cake, Money)
- 하나의 변수가 여러 원인을 가질 수 있음 (Money ← CoinFlip, TerryInRoom)

### 관찰되지 않는 변수 (Unobserved)

측정할 수 없거나 데이터에 없는 변수. 다이어그램에 회색으로 표시.

```
TerryMood → TerryInRoom → Money
  (측정 불가)
```

### 잠재 변수 (Latent Variable)

두 변수가 상관있지만 서로 원인이 아닐 때 → 공통 원인이 있다.

```
       Temperature (U1)
      ↙              ↘
  Shorts          IceCream
```

반바지와 아이스크림은 서로 원인이 아니다. **기온**이라는 잠재 변수가 둘 다에 영향.

---

## 6.3 현실 세계: 경찰과 범죄

**데이터**: 경찰이 많은 지역에서 범죄율이 더 높다 (!?)

직관적으로 경찰이 많으면 범죄가 줄어야 하는데, 왜?

### 인과 다이어그램

```
LaggedCrime ──→ PolicePerCapita ──→ ExpectedCrimePayout ──→ Crime
     │                                                        ↑
     └────────────────────────────────────────────────────────┘
                          (직접 영향도 있음)

LawAndOrderPolitics ──→ PolicePerCapita
         │
         └──→ SentencingLaws ──→ ExpectedCrimePayout
```

**핵심**: 과거 범죄(LaggedCrime)가 높은 지역에 경찰을 더 배치한다.
그래서 경찰이 많은 곳 = 원래 범죄가 많은 곳. **역인과(reverse causality)**!

### 다이어그램에 없는 것 = 가정

> "다이어그램에서 빠진 모든 변수와 화살표는 우리가 만드는 가정이다"

빠뜨리면 안 되는 것:
- 빈곤율 (PovertyRate) → 범죄에 영향
- 과거 경찰 수 → 현재 경찰 수에 영향

너무 많이 넣으면: 복잡해져서 사용 불가
적절한 균형을 찾아야 한다.

---

## 6.4 연구 질문과 다이어그램

### 직접 효과 vs 간접 효과

```
PolicePerCapita ──→ Crime              (직접 효과)
PolicePerCapita ──→ ExpectedPayout ──→ Crime  (간접 효과)
```

- **직접 효과**: 경찰 존재 자체가 범죄 억제
- **간접 효과**: 경찰이 체포 확률을 높여 → 범죄 기대 수익을 낮추어 → 범죄 감소

둘 다 "경찰이 범죄를 줄인다"에 포함된다.

---

## 6.5 조절 변수 (Moderator)

> **조절 변수** = X가 Y에 미치는 효과가 Z에 따라 달라질 때, Z가 조절 변수

예: 불임 치료약의 효과는 **자궁 유무**에 따라 완전히 다르다.

```python
# 조절 효과의 수학적 표현
Y = β₀ + β₁·X + β₂·Z + β₃·X·Z
#                         ↑ 이 항이 조절 효과
# β₃ ≠ 0 이면 Z가 X의 효과를 조절한다
```

---

## 실전 적용: 신규 기능의 인과 다이어그램

어떤 앱이 신규 기능을 출시하고 효과를 측정하려 한다:

```
        UserMotivation (관찰 불가)
       ↙              ↘
FeatureAdopted        Engagement
       ↓                    ↑
  BetterExperience ─────────┘

Age, Gender, Activity ──→ FeatureAdopted
Age, Gender, Activity ──→ Engagement
```

- **Good Path**: Feature → BetterExperience → Engagement (진짜 인과)
- **Bad Path**: UserMotivation → Feature, UserMotivation → Engagement (교란)

공변량 보정은 Age, Gender, Activity 등을 통제하여 Bad Path를 차단하려는 시도.
하지만 UserMotivation은 관찰 불가 → 완벽히 차단 못함 → "관찰적 추정치" 한계.

---

## Python 예제: 경찰과 범죄의 역인과

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)
n = 500

# DGP: 과거 범죄가 높으면 경찰을 더 배치
lagged_crime = np.random.normal(50, 15, n)
police = 10 + 0.3 * lagged_crime + np.random.normal(0, 5, n)

# 진짜 효과: 경찰은 범죄를 줄인다 (계수 = -0.5)
crime = 20 + 0.6 * lagged_crime - 0.5 * police + np.random.normal(0, 8, n)

df = pd.DataFrame({
    'crime': crime,
    'police': police,
    'lagged_crime': lagged_crime
})

# 잘못된 분석: 과거 범죄 통제 안 함
model_bad = smf.ols('crime ~ police', data=df).fit()
print("=== 과거 범죄 통제 안 함 (잘못된) ===")
print(f"경찰 계수: {model_bad.params['police']:.3f}")
print("→ 양수! 경찰이 많을수록 범죄가 많다고 나옴 (역인과 때문)")

# 올바른 분석: 과거 범죄를 통제
model_good = smf.ols('crime ~ police + lagged_crime', data=df).fit()
print(f"\n=== 과거 범죄 통제 (올바른) ===")
print(f"경찰 계수: {model_good.params['police']:.3f}")
print(f"과거범죄 계수: {model_good.params['lagged_crime']:.3f}")
print("→ 경찰 계수가 음수! 경찰이 범죄를 줄인다는 진짜 효과가 나옴")
```
