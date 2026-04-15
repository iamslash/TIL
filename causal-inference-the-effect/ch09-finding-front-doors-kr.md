# Chapter 9: 프론트도어 찾기 (Finding Front Doors)

> **핵심 질문**: 백도어를 닫을 수 없을 때, 어떻게 인과 효과를 식별하는가?

---

## 9.1 백도어 대신 프론트도어로

Ch8까지는 "Bad Path(백도어)를 닫아서 식별"하는 전략이었다.
하지만 현실에서는 교란 변수를 **측정할 수 없는** 경우가 많다.

**대안 전략**: 프론트도어만 활용한다.
- 처치 변동 중에서 **백도어가 없는 부분**만 골라 쓴다
- 또는 프론트도어 경로의 개별 화살표를 따로 추정한다

### 복권 당첨 예제

```
질문: 부(wealth)가 수명(lifespan)에 영향을 주는가?

문제: 부의 원인은 너무 많다 (사업 능력, 교육, 유전 등)
     → 백도어를 다 닫을 수 없다

해결: 복권 당첨자만 본다!
     → 당첨 금액은 무작위 → 백도어 없음
     → 당첨금으로 인한 부의 변동만 분석
```

---

## 9.2 무작위 실험 (Randomized Controlled Experiment)

> 연구자가 처치를 **무작위로 배정**하고 결과 차이를 관찰

무작위 배정이 강력한 이유: **모든 백도어를 한 번에 닫는다**

```
예: 차터스쿨(특목학교) 효과를 알고 싶다

문제: 차터스쿨 선택 ← 인종, 배경, 성격, 학업 의지...
     같은 변수들이 성적에도 영향 → 수많은 백도어

해결: 추첨(lottery)으로 입학 결정
     → 추첨 당선 여부는 완전 무작위
     → 당선자 vs 탈락자 비교 = 깨끗한 인과 추정
```

```
LotteryWin → CharterSchool → Achievement
                                  ↑
AllKindsaStuff ──────────────────┘
       └──→ CharterSchool

추첨으로 인한 변동만 분리하면 AllKindsaStuff의 영향이 사라진다
```

**한계**: 추첨에 참여한 학생/학교만 대상 → 전체 모집단에 일반화가 어려울 수 있다.

---

## 9.3 자연 실험 (Natural Experiment)

> 연구자가 아니라 **현실 세계에서 무작위 배정이 일어나는** 상황을 활용

### 예제 1: 대기오염과 운전 (베이징)

```
문제: 자동차가 오염을 유발 & 오염이 운전에 영향 → 순환!
해결: 바람 방향이 서풍이면 오염물질이 베이징으로 → 오염 증가
     바람 방향은 운전 결정과 무관 (외생적)
     → 바람에 의한 오염 변동만 분석
결과: 오염 등급이 "오염"으로 올라가면 운전량 +3%
```

### 예제 2: 메디케이드 확장과 병원 환자 경험

```
2014년 일부 주에서 메디케이드 확장 → 보험 가입 증가 → 미보상 진료 감소
미보상 진료 감소 → 병원 자원 여유 → 환자 경험 개선?
메디케이드 확장 여부를 외생적 변동으로 활용
```

### 자연 실험 vs 무작위 실험

| 차이점 | 무작위 실험 | 자연 실험 |
|--------|-----------|----------|
| 백도어 | 없음 (보장) | 있을 수 있음 (확인 필요) |
| 자연스러움 | 참가자가 인지 | 참가자가 모름 |
| 표본 크기 | 보통 작음 | 보통 큼 |
| 신뢰성 | 높음 | 설득 필요 |

---

## 9.4 프론트도어 방법 (Front Door Method)

백도어를 닫을 수 없지만, 처치와 결과 사이에 **매개변수(M)**가 있을 때:

```
W (관찰 불가) → Treatment
W (관찰 불가) → Outcome
Treatment → M → Outcome
```

### 흡연과 폐암 예제

```
여러 요인 → Smoking
여러 요인 → Cancer
Smoking → TarInLungs → Cancer
```

1. Smoking → Tar 관계 추정: 매일 1개비 추가 → 10년간 타르 15g 증가
2. Tar → Cancer 관계 추정 (Smoking 통제): 타르 15g → 암 확률 2%p 증가
3. 곱하기: 매일 1개비 → 암 확률 2%p 증가

**현실적 한계**: M이 처치→결과의 **유일한 경로**여야 하고, M에 독립적 원인이
없어야 한다. 이 조건이 현실에서 거의 성립하지 않아 **실제로는 잘 안 쓰인다.**

---

## Python 예제: 자연 실험 — 바람과 대기오염

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)
n = 3000

# DGP
season = np.random.choice([0, 1, 2, 3], n)  # 0=봄, 1=여름, 2=가을, 3=겨울
wind_from_west = np.random.binomial(1, 0.3 + 0.1 * (season == 3), n)  # 겨울에 서풍 더 많음

# 오염: 서풍이면 높아짐, 겨울에도 높아짐
pollution = (
    30
    + 20 * wind_from_west
    + 10 * (season == 3)
    + np.random.normal(0, 10, n)
)

# 운전: 오염이 높으면 증가 (진짜 효과), 겨울에는 감소
driving = (
    100
    + 0.3 * pollution   # 진짜 효과: 오염 1단위 → 운전 0.3 증가
    - 5 * (season == 3)  # 겨울에 운전 감소
    + np.random.normal(0, 8, n)
)

df = pd.DataFrame({
    'wind_from_west': wind_from_west,
    'pollution': pollution,
    'driving': driving,
    'season': season,
    'winter': (season == 3).astype(int)
})

# === 단순 분석: 오염 → 운전 ===
model_naive = smf.ols('driving ~ pollution', data=df).fit()
print("=== 단순 분석 ===")
print(f"오염 계수: {model_naive.params['pollution']:.4f}")
print("→ 편향 가능 (계절이 교란)")

# === 자연 실험: 바람으로 인한 변동만 사용 (2단계 추정) ===
# 1단계: 바람 → 오염 (계절 통제)
stage1 = smf.ols('pollution ~ wind_from_west + winter', data=df).fit()
df['pollution_hat'] = stage1.fittedvalues

# 2단계: 예측된 오염 → 운전 (계절 통제)
stage2 = smf.ols('driving ~ pollution_hat + winter', data=df).fit()
print(f"\n=== 자연 실험 (바람을 도구변수로) ===")
print(f"오염 계수: {stage2.params['pollution_hat']:.4f}")
print(f"→ 진짜 효과 0.3에 가깝다!")
print(f"\n1단계: 서풍 → 오염 +{stage1.params['wind_from_west']:.1f} 단위")
```

**핵심**: 바람 방향은 운전 결정과 무관(외생적)하므로, 바람으로 인한 오염 변동만
분리하면 깨끗한 인과 추정이 가능하다. 이것이 **도구변수(IV)** 의 기본 아이디어이며
Ch19에서 자세히 다룬다.
