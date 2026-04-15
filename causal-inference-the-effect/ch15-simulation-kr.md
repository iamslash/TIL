# Chapter 15: 시뮬레이션 (Simulation)

> **핵심 질문**: 내가 선택한 분석 방법이 제대로 작동하는지 어떻게 확인하는가?

---

## 15.1 시뮬레이션이란?

> 연구자가 **DGP를 직접 설정**하고, 가상 데이터를 생성하여,
> 분석 방법이 **진짜 효과를 복원하는지** 검증하는 과정

```
1. DGP 설정: "진짜 효과 = 2.0" (내가 정함)
2. 가상 데이터 생성
3. 분석 방법 적용
4. 추정치가 2.0에 가까운가?
   → 가까우면: 방법이 작동
   → 멀면: 방법에 문제 있음
```

---

## 15.2 왜 필요한가?

- 실제 데이터에서는 **진짜 효과를 모른다** (알면 연구할 필요 없음)
- 시뮬레이션에서는 **내가 진짜를 알고 있으므로** 방법의 성능을 평가 가능
- DGP 가정을 바꿔가며 **방법의 한계**도 테스트 가능

---

## Python 예제: DiD가 작동하는지 시뮬레이션으로 검증

```python
import numpy as np
import statsmodels.formula.api as smf

true_effect = 3.0
n_simulations = 500
estimates = []

for sim in range(n_simulations):
    n = 1000
    group = np.repeat([0, 1], n // 2)
    period = np.tile([0, 1], n // 2)

    # DGP: 평행 추세 성립
    y = (
        5
        + 2 * group         # 그룹 차이
        + 1 * period         # 시간 효과
        + true_effect * group * period  # 처치 효과
        + np.random.normal(0, 3, n)
    )

    import pandas as pd
    df = pd.DataFrame({'y': y, 'group': group, 'period': period})
    model = smf.ols('y ~ group * period', data=df).fit()
    estimates.append(model.params['group:period'])

estimates = np.array(estimates)
print(f"=== {n_simulations}번 시뮬레이션 결과 ===")
print(f"진짜 효과: {true_effect}")
print(f"추정치 평균: {estimates.mean():.3f}")
print(f"추정치 표준편차: {estimates.std():.3f}")
print(f"95% 범위: [{np.percentile(estimates, 2.5):.2f}, {np.percentile(estimates, 97.5):.2f}]")
print(f"→ 평균이 {true_effect}에 가까우면 DiD가 비편향 추정치를 준다!")
```
