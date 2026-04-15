# Chapter 17: 이벤트 스터디 (Event Studies)

> **핵심 질문**: 특정 사건 전후를 비교하여 인과 효과를 추정할 수 있는가?

---

## 17.1 이벤트 스터디란?

> 특정 시점에 발생한 사건(처치) 전후의 결과를 비교하여,
> 사건이 없었다면의 **반사실(counterfactual)**과 실제 결과의 차이를 측정

```
결과
  ↑
  │        * * * (실제)
  │      *
  │- - -╳- - - - (사건 없었다면, 예측)
  │   *
  │ * *
  └────────────→ 시간
         사건
```

---

## 17.2 인과 식별의 도전

**백도어**: 시간 → 결과, 시간 → 처치

이벤트 스터디가 작동하려면: **"처치 시점에 바뀐 것이 처치뿐"**이어야 한다.

### 3가지 반사실 예측 방법

| 방법 | 적용 조건 | 예시 |
|------|----------|------|
| 시간 추세 무시 | 추세가 없거나 매우 짧은 기간 | 주가 반응 (분 단위) |
| 사전 추세 외삽 | 사건 전 추세가 안정적 | 정책 시행 전후 비교 |
| 관련 변수 활용 | 추세 + 다른 예측변수 | CAPM 기반 초과수익 |

---

## 17.3 DiD와의 관계

| | 이벤트 스터디 | DiD |
|--|------------|-----|
| 대조군 | 없음 (자체 사전 추세) | 있음 (미처치 그룹) |
| 가정 강도 | 더 강함 | 더 약함 |
| 장점 | 대조군 불필요 | 평행 추세로 교란 제거 |

DiD의 동적 효과 분석(Ch18.6)은 사실상 이벤트 스터디의 DiD 버전이다.

---

## Python 예제

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)
T = 20  # 20개 시점
event_time = 10  # 사건 발생 시점

time = np.arange(T)
trend = 0.5 * time  # 기존 추세
effect = np.where(time >= event_time, 3.0, 0)  # 사건 효과 = 3.0
y = 10 + trend + effect + np.random.normal(0, 1, T)

df = pd.DataFrame({'time': time, 'y': y, 'post': (time >= event_time).astype(int)})

# 사전 추세 외삽
pre = df[df['time'] < event_time]
model_pre = smf.ols('y ~ time', data=pre).fit()
df['predicted'] = model_pre.predict(df)
df['effect'] = df['y'] - df['predicted']

print("=== 사건 후 평균 효과 ===")
print(f"추정: {df[df['post']==1]['effect'].mean():.2f} (진짜: 3.0)")
```
