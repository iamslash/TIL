# Chapter 20: 회귀단절 설계 (Regression Discontinuity, RD)

> **핵심 질문**: 기준점(cutoff) 바로 위와 아래의 사람은 거의 같은데,
> 처치만 다르다면 → 인과 효과를 추정할 수 있는가?

---

## 20.1 핵심 아이디어

```
처치 확률
  ↑
1 │          ┌──────
  │          │
  │          │
0 │──────────┘
  └──────────────→ 점수
            기준점
```

기준점 바로 위(처치) vs 바로 아래(미처치) 비교
→ 거의 같은 사람들 → 마치 무작위 배정처럼

### 예: 장학금 기준 GPA 3.5

```
GPA 3.49인 학생 vs GPA 3.51인 학생
→ 실력 차이는 거의 없다
→ 하나는 장학금 받고, 하나는 못 받는다
→ 장학금의 인과 효과를 추정할 수 있다
```

---

## 20.2 Sharp vs Fuzzy RD

| 유형 | 처치 결정 | 분석 방법 |
|------|----------|----------|
| **Sharp RD** | 기준점에서 처치가 0→100% 전환 | 직접 비교 |
| **Fuzzy RD** | 기준점에서 처치 확률이 불연속적으로 증가 | 도구변수(IV)로 분석 |

```
Sharp: GPA ≥ 3.5 → 100% 장학금
Fuzzy: GPA ≥ 3.5 → 장학금 확률 40%→80% (다른 요인도 영향)
```

---

## 20.3 핵심 가정

### 1. 기준점에서의 매끄러움

> 처치를 제외하면, 기준점에서 **다른 모든 변수가 연속적**이어야 한다

### 2. 조작 불가

> 대상자가 기준점을 넘기 위해 **점수를 조작할 수 없어야** 한다

→ 밀도 검정: 기준점 근처에서 관측치가 비정상적으로 몰려있으면 조작 의심

### 3. 정밀한 측정

> 기준점 근처를 "줌인"할 수 있을 만큼 데이터가 충분해야

---

## 20.4 대역폭 (Bandwidth)

기준점에서 얼마나 멀리까지 데이터를 사용할 것인가?

| 좁은 대역폭 | 넓은 대역폭 |
|------------|------------|
| 비교 대상이 더 비슷 | 더 많은 데이터 |
| 편향 적음 | 분산 적음 |
| 분산 큼 | 편향 클 수 있음 |

최적 대역폭은 편향-분산 트레이드오프를 자동으로 계산하는 알고리즘으로 선택.

---

## 20.5 RD가 추정하는 것

> **LATE**: 기준점 **바로 근처** 사람들의 처치 효과

GPA 3.5 근처 학생의 장학금 효과. GPA 2.0이나 4.0인 학생에 대해서는 알 수 없음.

---

## 실전 적용: 자격 기준이 있는 기능

사진 기반 기능의 자격이 "사진 300장 이상"이라면:

```
자격 조건: 사진 ≥ 300장

299장인 유저 vs 301장인 유저
→ 거의 같은 유저인데, 기능 자격만 다르다
→ 사진 수 기준 RD 설계 가능

문제: 사진 수 데이터가 있는지 확인 필요
```

---

## Python 예제: Sharp RD

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)
n = 2000

# 점수 (running variable)
score = np.random.uniform(50, 100, n)
cutoff = 75

# 처치: 점수 ≥ 75이면 장학금
treatment = (score >= cutoff).astype(int)

# 결과: 장학금 효과 = +5, 점수 자체도 결과에 영향
outcome = 0.3 * score + 5.0 * treatment + np.random.normal(0, 3, n)

df = pd.DataFrame({
    'score': score, 'treatment': treatment,
    'outcome': outcome, 'centered': score - cutoff
})

# === Sharp RD: 기준점 근처만 사용 ===
bandwidth = 5
near = df[abs(df['centered']) <= bandwidth]

model = smf.ols('outcome ~ treatment + centered', data=near).fit()
print(f"=== RD 추정 (대역폭 ±{bandwidth}) ===")
print(f"처치 효과: {model.params['treatment']:.2f} (진짜: 5.0)")
print(f"표본 크기: {len(near)}")

# 대역폭 민감도 분석
print(f"\n=== 대역폭 민감도 ===")
for bw in [3, 5, 10, 15]:
    sub = df[abs(df['centered']) <= bw]
    m = smf.ols('outcome ~ treatment + centered', data=sub).fit()
    print(f"BW={bw:2d}: 효과={m.params['treatment']:.2f}, "
          f"SE={m.bse['treatment']:.2f}, N={len(sub)}")
```
