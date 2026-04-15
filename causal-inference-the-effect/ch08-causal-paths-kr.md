# Chapter 8: 인과 경로와 백도어 차단 (Causal Paths and Closing Back Doors)

> **핵심 질문**: 인과 다이어그램에서 어떤 경로를 열고 닫아야 진짜 효과를 식별하는가?

이 챕터는 인과추론의 **가장 핵심적인 메커니즘**을 다룬다.

---

## 8.1 경로 (Path)

> **경로** = 인과 다이어그램에서 한 변수에서 다른 변수로 가는 길
> (화살표 방향을 무시하고 이동)

### 와인과 수명 예제

```
Wine → Lifespan                              (경로 1)
Wine → Drugs → Lifespan                      (경로 2)
Wine ← Income → Lifespan                     (경로 3)
Wine ← Income ← U1 → Health → Lifespan       (경로 4)
Wine ← Health → Lifespan                     (경로 5)
Wine ← Health ← U1 → Income → Lifespan       (경로 6)
```

각 경로는 "와인과 수명이 관련되는 이유" 하나를 설명한다.

---

## 8.2 모든 경로 찾기

### 6단계 프로세스

1. 처치 변수에서 시작
2. 아무 화살표(방향 무관)를 따라 다음 변수로 이동
3. 다시 화살표를 따라 이동
4. 결과 변수에 도착하면 → 경로 기록 / 이미 방문한 변수면 → 중단
5. 뒤로 돌아가서 다른 화살표 시도
6. 모든 가능한 경로를 찾을 때까지 반복

---

## 8.3 좋은 경로 vs 나쁜 경로

### Front Door Path (좋은 경로)

> 모든 화살표가 처치에서 **멀어지는** 방향

```
Wine → Lifespan                     ← Front Door (직접 효과)
Wine → Drugs → Lifespan             ← Front Door (간접 효과)
```

**이 경로들은 "와인이 수명에 미치는 효과"의 일부다.**

### Back Door Path (나쁜 경로)

> 어딘가에서 화살표가 처치 **쪽으로** 향함

```
Wine ← Income → Lifespan           ← Back Door (소득이 교란)
Wine ← Health → Lifespan           ← Back Door (건강이 교란)
```

**이 경로들은 "대안 설명"이다. 와인 효과가 아니라 소득/건강 효과.**

### 식별의 핵심 원리

> **모든 Bad Path를 닫고, Good Path는 열어두면 → 인과 효과가 식별된다**

---

## 8.4 열린 경로와 닫힌 경로

### 열린 경로 (Open Path)

경로상 모든 변수에 변동이 있으면 → 열림 → 데이터에 영향

### 닫힌 경로 (Closed Path)

경로상 변수 중 하나라도 변동이 없으면 → 닫힘 → 데이터에 영향 없음

### 변수를 통제하면 경로가 닫힌다

```
경로: Wine ← Income → Lifespan
Income을 통제하면 → 이 경로는 닫힌다
```

### 와인 예제 해결

```
경로 1: Wine → Lifespan              ← Good, 열어둠
경로 2: Wine → Drugs → Lifespan      ← Good, 열어둠
경로 3: Wine ← Income → Lifespan     ← Bad, Income 통제 → 닫힘
경로 4: Wine ← Income ← U1 → Health  ← Bad, Income 통제 → 닫힘
경로 5: Wine ← Health → Lifespan     ← Bad, Health 통제 → 닫힘
경로 6: Wine ← Health ← U1 → Income  ← Bad, Health 통제 → 닫힘
```

**Income과 Health를 통제하면, 남는 관계는 오직 Good Path뿐!**

---

## 8.5 충돌변수 (Collider) — 가장 직관에 반하는 개념

> **충돌변수** = 경로에서 양쪽 화살표가 모두 **자기를 향하는** 변수

```
A → Collider ← B
```

### 충돌변수의 핵심 성질

> **충돌변수가 있는 경로는 기본적으로 닫혀 있다!**

A와 B가 모두 Collider에 영향을 주지만, Collider가 경로의 "막힌 지점"이 되어
A와 B 사이에 관계를 만들지 않는다.

### 샌드위치 예제

```
BuySandwich → AteSandwich ← GiftedSandwich
```

- 사는 것과 선물 받는 것은 **원래 무관하다**
- **AteSandwich를 통제하면?** (예: "샌드위치를 먹은 사람만" 분석)
- → 사지 않았는데 먹었으면 선물 받았을 것 → **갑자기 관계가 생김!**

### 위험: 충돌변수를 통제하면 닫힌 경로가 열린다!

```
Treatment → C ← Outcome
         (충돌변수)

통제 안 하면: 경로 닫힘 (좋음)
C를 통제하면: 경로 열림 (나쁨! 가짜 관계 생성)
```

### 실전에서 흔한 실수

**표본 선택 편향**: 대학생만 대상으로 연구하면, "대학 진학" 변수를 통제한 것.
만약 이것이 충돌변수라면, 없던 편향이 생긴다.

---

## 8.6 다이어그램 테스트: 위약 테스트 (Placebo Test)

다이어그램이 맞는지 확인하는 방법:

1. 처치/결과가 아닌 두 변수 A, B를 고른다
2. A-B 사이 모든 경로를 나열한다
3. 다이어그램에 따라 모든 열린 경로를 닫는다
4. 데이터에서 A-B 관계가 0인지 확인한다

> **0이 아니면 → 다이어그램에 뭔가 빠져있다!**

---

## 8.7 경로 용어 정리

| 용어 | 정의 |
|------|------|
| **Good Path** | 연구 질문에 답하는 경로 |
| **Bad Path** | 대안 설명을 나타내는 경로 |
| **Front Door** | 모든 화살표가 처치에서 멀어짐 |
| **Back Door** | 하나 이상의 화살표가 처치를 향함 |
| **Open Path** | 경로상 모든 변수에 변동 있음 |
| **Closed Path** | 경로상 하나 이상 변수에 변동 없음 |
| **Collider** | 양쪽 화살표가 자기를 향하는 변수 |

---

## Python 예제: 충돌변수 편향

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)
n = 5000

# DGP: Talent과 Looks는 완전히 독립
talent = np.random.normal(0, 1, n)
looks = np.random.normal(0, 1, n)

# Hollywood: Talent이나 Looks가 높으면 영화에 캐스팅
# → "Hollywood" 는 충돌변수 (Talent → Hollywood ← Looks)
hollywood = (talent + looks > 1).astype(int)

df = pd.DataFrame({
    'talent': talent,
    'looks': looks,
    'hollywood': hollywood
})

# === 전체 데이터: Talent과 Looks는 무관 ===
corr_all = df['talent'].corr(df['looks'])
print(f"=== 전체 데이터 ===")
print(f"Talent ↔ Looks 상관: {corr_all:.4f}")
print("→ 거의 0. 실제로 독립!")

# === 할리우드 배우만 분석: 충돌변수 통제! ===
hw = df[df['hollywood'] == 1]
corr_hw = hw['talent'].corr(hw['looks'])
print(f"\n=== 할리우드 배우만 (충돌변수 통제) ===")
print(f"Talent ↔ Looks 상관: {corr_hw:.4f}")
print("→ 음의 상관! 잘생긴데 캐스팅됐으면 연기력이 낮을 확률이 높다")
print("→ 충돌변수를 통제하면 없던 관계가 생긴다!")
print(f"\n전체 데이터 수: {len(df)}")
print(f"할리우드 배우 수: {len(hw)}")

# === 회귀로 확인 ===
model_all = smf.ols('talent ~ looks', data=df).fit()
model_hw = smf.ols('talent ~ looks', data=hw).fit()

print(f"\n=== 회귀 결과 ===")
print(f"전체: looks 계수 = {model_all.params['looks']:.4f} (p={model_all.pvalues['looks']:.4f})")
print(f"할리우드만: looks 계수 = {model_hw.params['looks']:.4f} (p={model_hw.pvalues['looks']:.4f})")
```

**실행 결과 해석:**
- 전체 데이터: Talent과 Looks는 상관 ≈ 0 (진짜 독립)
- 할리우드 배우만: Talent과 Looks는 **음의 상관** (가짜 관계!)
- → 이것이 **"잘생긴 배우는 연기를 못한다"** 는 속설의 통계적 원인
- → 충돌변수(캐스팅 여부)를 통제하면 없던 관계가 만들어진다

**핵심 교훈**: 통제하면 좋은 것만 있는 게 아니다. **충돌변수를 통제하면 오히려
편향이 생긴다.** 어떤 변수를 통제할지는 인과 다이어그램을 보고 결정해야 한다.
