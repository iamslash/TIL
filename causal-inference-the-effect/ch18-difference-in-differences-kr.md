# Chapter 18: 이중차분법 (Difference-in-Differences, DiD)

> **이 책에서 가장 중요한 챕터.** 관찰 데이터에서 인과 효과를 추정하는 가장 널리 쓰이는 방법 중 하나다.

---

## 18.1 핵심 아이디어

```
DiD = (처치군의 변화) - (대조군의 변화)

    = (처치군 After - 처치군 Before) - (대조군 After - 대조군 Before)
```

### 왜 이중으로 차이를 구하는가?

**문제 1**: 처치 전후만 비교하면 → 시간 효과와 구분 못함
```
기능 채택자: 전환 8건 → 11건 (+3)
→ 3건 중에서 기능 효과는 얼마? 시간이 지나서 자연히 늘어난 건?
```

**문제 2**: 처치/대조만 비교하면 → 그룹 차이와 구분 못함
```
채택자 11개 vs 비채택자 7개 (차이 4)
→ 채택자가 원래 더 적극적인 유저일 수 있음
```

**DiD의 해결**:
```
채택자:   8 → 11 (변화 +3)
비채택자: 6 → 7  (변화 +1)  ← 자연적 추세
DiD = 3 - 1 = 2  ← 기능의 순수 효과
```

---

## 18.2 역사적 예제: John Snow의 콜레라 연구 (1855)

```
Lambeth (상류로 취수원 이동): 130.1 → 84.9 사망 (변화 -45.2)
기타 회사 (취수원 그대로):    134.9 → 146.6 사망 (변화 +11.7)

DiD = -45.2 - (+11.7) = -56.9 사망/만명
→ 깨끗한 물이 콜레라를 줄인다!
```

---

## 18.3 평행 추세 가정 (Parallel Trends Assumption)

> **DiD의 가장 중요한 가정**: "처치가 없었다면, 두 그룹은 같은 기울기로
> 변했을 것이다"

```
매칭 수
  ↑
11│            ╱ 채택자 (실제)
  │          ╱
 9│- - - - ╳ - - - 채택자 (처치 없었다면, 가상)
  │      ╱   ╱
 8│────╱───╱
  │  ╱  ╱
 7│╱──╱ 비채택자 (실제 = 가상)
  │ ╱
 6│╱
  └──────────────→ 시간
       전     후
```

**검증할 수 없다** (반사실이므로). 하지만 간접적으로 확인 가능:
- 처치 전 기간에 두 그룹의 추세가 평행한지 확인
- 위약 검정 (가짜 처치 날짜)

---

## 18.4 회귀로 구현: 이원 고정효과

```python
Y = β₀ + β₁·TreatedGroup + β₂·AfterPeriod + β₃·(TreatedGroup × AfterPeriod) + ε

β₁ = 그룹 차이 (처치 전)
β₂ = 시간 효과 (대조군의 변화)
β₃ = DiD 추정치 = 처치 효과!
```

또는 고정효과 형태로:
```python
Y = α_group + γ_time + β·Treated + ε
# α_group = 그룹 고정효과
# γ_time = 시간 고정효과
# β = 처치 효과 (DiD)
```

**표준오차**: 반드시 그룹 수준에서 **클러스터링** 해야 함.

---

## 18.5 평행 추세 검정

### 사전 추세 검정 (Prior Trends Test)

```python
# 처치 전 데이터만 사용
Y = α_group + β₁·Time + β₂·(Time × Group) + ε
# β₂ = 0 이면 → 사전 추세가 평행 (좋은 신호)
# β₂ ≠ 0 이면 → 평행 추세 의심
```

### 위약 검정 (Placebo Test)

```
처치 전 데이터에서 가짜 처치 날짜를 정하고 DiD 실행
→ 효과가 0이어야 함
→ 0이 아니면 → 평행 추세 위반 징후
```

---

## 18.6 동적 처치 효과 (Event Study)

처치 효과가 시간에 따라 달라지는지 확인:

```python
Y = α_group + γ_time + Σ βₜ · Treated_t + ε
# t = ..., -2, -1, 0, +1, +2, ...
# 처치 전 βₜ들은 0이어야 함 (평행 추세 확인)
# 처치 후 βₜ들이 효과를 보여줌
```

그래프로 그리면 "이벤트 스터디 플롯":
```
효과
 ↑
 │     *  *  *
 │   *
 │──────── 0 ─── * ──────
 │ * * *
 │
 └──────────────────→ 시간
  -3 -2 -1  0  1  2  3
           (처치)
```

---

## 18.7 단계적 도입 (Staggered Rollout) 문제

여러 그룹이 **다른 시점에** 처치를 받을 때:

```
호주: 2026년 2월 처치
미국: 2026년 6월 처치
캐나다: 아직 미처치
```

**문제**: 표준 이원 고정효과가 **이미 처치 받은 그룹을 대조군으로 사용**함.
효과가 시간에 따라 변하거나 그룹마다 다르면 → **편향 발생**.

### 해결 방법들

| 방법 | 핵심 아이디어 | 패키지 |
|------|-------------|--------|
| **Callaway & Sant'Anna** | 코호트별 효과를 따로 추정 후 합산 | R: `did`, Python: `differences` |
| **Wooldridge ETWFE** | 코호트×시기 상호작용 추가 | R: `etwfe`, Stata: `jwdid` |
| **BJS Imputation** | 미처치 관측치로 처치 결과 예측 후 비교 | R: `didimputation` |

---

## 18.8 매칭 + DiD

매칭과 DiD를 결합하면 더 강력:

```
1단계: 처치군과 비슷한 대조군을 성향 점수로 매칭
2단계: 매칭된 표본에서 DiD 실행
→ 더 좋은 비교 그룹 + 시간 차이 활용
```

**주의**: 처치 전 결과값으로 매칭하면 **평균 회귀(Regression to the Mean)** 문제.
→ 처치 전 공변량으로 매칭하는 것이 안전.

---

## 실전 적용 예시: 앱 기능 효과 분석

DiD를 기능 채택 분석에 적용한다면:

```
처치군: 기능 채택자
대조군: 비채택자
처치 전: 채택 전 7일
처치 후: 채택 후 7일

DiD = (채택자의 전환 변화) - (비채택자의 전환 변화)

공변량 보정: 활동량, 활동일수, 나이, 성별, 계정나이, 구독상태
→ 조건부 평행 추세: "이 변수들이 같다면 추세가 평행할 것"
```

**한계**:
- 관찰 불가능한 "동기"는 통제 못함
- 채택자 = 원래 적극적 유저일 수 있음
- 하지만 단순 비교보다 **훨씬 강한 인과 근거**

---

## Python 예제: DiD 완전 구현

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)

# === DGP: 기능 채택 시뮬레이션 ===
n_users = 2000
n_periods = 2  # before, after

# 유저별 고정 특성
user_motivation = np.random.normal(0, 1, n_users)

# 동기 높은 유저가 채택할 확률 높음
adopt_prob = 1 / (1 + np.exp(-1.0 * user_motivation))
adopted = np.random.binomial(1, adopt_prob, n_users)

# 패널 데이터 생성
rows = []
for i in range(n_users):
    for t in [0, 1]:  # 0=before, 1=after
        # 기본 결과 = 동기 효과 + 시간 추세 + 노이즈
        base = 5 + 2 * user_motivation[i] + 1.0 * t
        # 기능 효과: after 기간에만 채택자에게 적용
        feature_effect = 2.0 * adopted[i] * t  # 진짜 효과 = 2.0
        matches = base + feature_effect + np.random.normal(0, 2)
        rows.append({
            'user': i, 'period': t, 'adopted': adopted[i],
            'matches': matches, 'motivation': user_motivation[i]
        })

df = pd.DataFrame(rows)

# === 분석 1: 단순 비교 (편향) ===
after = df[df['period'] == 1]
naive = after.groupby('adopted')['matches'].mean()
print(f"=== 단순 비교 (after만) ===")
print(f"채택자: {naive[1]:.2f}, 비채택자: {naive[0]:.2f}")
print(f"차이: {naive[1] - naive[0]:.2f} (진짜 효과 2.0 + 동기 편향)")

# === 분석 2: DiD ===
model_did = smf.ols(
    'matches ~ adopted * period',
    data=df
).fit(cov_type='cluster', cov_kwds={'groups': df['user']})

print(f"\n=== DiD ===")
print(f"adopted:         {model_did.params['adopted']:.2f} (그룹 차이)")
print(f"period:          {model_did.params['period']:.2f} (시간 효과)")
print(f"adopted×period:  {model_did.params['adopted:period']:.2f} ← DiD 추정치")
print(f"(진짜 효과: 2.0)")

# === 분석 3: DiD + 공변량 보정 ===
model_adj = smf.ols(
    'matches ~ adopted * period + motivation',
    data=df
).fit(cov_type='cluster', cov_kwds={'groups': df['user']})

print(f"\n=== DiD + 동기 통제 (공변량 보정) ===")
print(f"adopted×period:  {model_adj.params['adopted:period']:.2f} ← 더 정확!")
print(f"motivation:      {model_adj.params['motivation']:.2f}")
```

**실행 결과 해석:**
- 단순 비교: 2.0보다 큰 값 (동기 편향 포함)
- DiD: 2.0에 가까움 (시간 효과와 그룹 차이 제거)
- DiD + 공변량: 2.0에 더 가까움 (동기까지 일부 통제)
