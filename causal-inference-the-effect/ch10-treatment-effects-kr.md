# Chapter 10: 처치 효과 (Treatment Effects)

> **핵심 질문**: "효과"라고 할 때, 정확히 **누구의**, **어떤** 효과를 추정하는 것인가?

---

## 10.1 효과는 사람마다 다르다

자궁경부암 약의 효과:
- Terry (자궁 있음): 암 확률 -2%p
- Angela (자궁 있음): 암 확률 -1%p
- Andrew (자궁 없음): 효과 0
- Mark (자궁 없음): 효과 0

**이질적 처치 효과 (Heterogeneous Treatment Effect)**:
처치 효과가 개인마다 다른 현상. 현실에서는 거의 항상 이질적이다.

---

## 10.2 다양한 "평균" 효과

### ATE (Average Treatment Effect) — 평균 처치 효과

> 모집단 전체에 처치를 적용했을 때의 평균 효과

```
Terry: -2%p, Angela: -1%p, Andrew: 0, Mark: 0
ATE = (-2 + -1 + 0 + 0) / 4 = -0.75%p
```

자궁 있는 사람만 분석하면: (-2 + -1) / 2 = -1.5%p → 이건 **조건부 ATE**

### ATT (Average Treatment on the Treated) — 처치군의 평균 효과

> **실제로 처치를 받은 사람들**의 평균 효과

앱 맥락: 신규 기능을 **실제로 채택한** 유저들의 효과.
채택하지 않은 유저에게는 다른 효과가 있을 수 있다.

### ATUT (Average Treatment on the Untreated) — 비처치군의 평균 효과

> 처치를 받지 않은 사람들에게 처치를 적용했다면의 효과

앱 맥락: 기능을 아직 안 쓴 유저들이 쓰면 어떤 효과가 있을까?
→ 채택률 확대 전략에서 중요한 지표

### LATE (Local Average Treatment Effect) — 국소 평균 효과

> 외생적 변동에 **강하게 반응하는 사람들**의 가중 평균 효과

자연 실험에서 주로 나온다. "도구변수에 의해 처치가 바뀌는 사람들"의 효과.

### Intent-to-Treat (ITT) — 처치 의도 효과

> 처치를 **배정한** 효과 (실제로 받았는지와 무관)

```
예: 4명에게 약을 배정
Chizue: 실제로 복용 (효과 -3)
Chizue2: 실제로 복용 (효과 -3)
Diego: 복용 안 함 (효과 0)
Diego2: 복용 안 함 (효과 0)

ITT = (-3 + -3 + 0 + 0) / 4 = -1.5
→ 배정의 효과 (복용의 효과 -3보다 작음)
```

---

## 10.3 어떤 효과를 얻게 되는가?

> **"사용하는 처치 변동의 출처가 어떤 효과를 얻는지 결정한다"**

### 경험 법칙

| 상황 | 얻는 효과 |
|------|----------|
| 대표 표본에서 무작위 배정, 보정 없음 | **ATE** |
| 특정 그룹에서 무작위 배정 | **조건부 ATE** |
| 관찰 데이터에서 백도어 닫기 (회귀 등) | **가중 ATE** (분산 가중) |
| 비처치군이 처치군의 반사실을 대리 | **ATT** |
| 외생적 변동으로 처치 변동 분리 | **LATE** |

### 분산 가중이란?

```
예: Brianna 1000명 (50% 처치), Diego 1000명 (90% 처치)

백도어를 닫은 후 남은 처치 변동:
  Brianna: 분산 = 0.5 × 0.5 = 0.25 (많음)
  Diego:   분산 = 0.9 × 0.1 = 0.09 (적음)

→ Brianna의 효과가 더 크게 반영된다
→ 추정치는 Brianna의 진짜 효과에 더 가깝다
```

---

## 10.4 왜 이게 중요한가?

### 납(lead)과 범죄

연구에서 특정 지역의 납 제거 효과(조건부 ATE)를 측정했다.
이 지역은 원래 범죄가 낮고 납도 적었다면 → 효과가 작게 나옴.
하지만 전국 ATE는 훨씬 클 수 있다 → 정책 결정을 그르칠 수 있다.

### 백신

헝가리에서 홍역 백신 테스트: 90%가 이미 접종 → ATE가 작게 나옴.
하지만 **미접종자** 대상 효과(조건부 ATE)는 매우 클 수 있다.

### 정책 결정과 매칭

| 정책 의도 | 필요한 효과 |
|----------|-----------|
| 전체에 적용 | ATE |
| 특정 그룹 대상 | 조건부 ATE |
| 이미 사용 중인 것 확대 | ATUT, 한계 효과 |
| 기존 정책 평가 | ATT |

---

## 실전 적용: 기능 채택 분석

관찰적 DiD 분석의 전형적 결과:
- 채택자 vs 비채택자 비교 → **ATT에 가까움**
- 채택자(관심 높은 유저)의 효과 ≠ 비채택자(관심 낮은 유저)의 효과
- 채택률을 늘리면 **새 유저의 효과(ATUT)**는 다를 수 있음

```
현재: 채택자(적극적 유저)의 효과 = +24%
미래: 비채택자(소극적 유저)도 쓰게 하면 효과가 더 클까 작을까?
→ ATUT 추정이 필요하지만, 현재 데이터로는 알기 어려움
```

---

## Python 예제: ATE vs ATT vs 가중 ATE

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 10000

# DGP: 두 그룹, 효과가 다름
group = np.random.choice(['high_motivation', 'low_motivation'], n, p=[0.3, 0.7])
true_effect = np.where(group == 'high_motivation', 5.0, 1.0)

# 높은 동기 유저가 처치를 더 많이 받음
treat_prob = np.where(group == 'high_motivation', 0.6, 0.15)
treated = np.random.binomial(1, treat_prob, n)

# 결과
y0 = np.random.normal(10, 3, n)  # 처치 안 받았을 때의 잠재 결과
y1 = y0 + true_effect            # 처치 받았을 때의 잠재 결과
y_observed = np.where(treated, y1, y0)

df = pd.DataFrame({
    'group': group,
    'treated': treated,
    'y': y_observed,
    'true_effect': true_effect
})

# === 진짜 효과 ===
ate = df['true_effect'].mean()
att = df[df['treated'] == 1]['true_effect'].mean()
atut = df[df['treated'] == 0]['true_effect'].mean()

print("=== 진짜 효과 (신의 시점) ===")
print(f"ATE  (전체):      {ate:.2f}")
print(f"ATT  (처치군):    {att:.2f}")
print(f"ATUT (비처치군):  {atut:.2f}")

# === 관찰 가능한 추정 ===
mean_treated = df[df['treated'] == 1]['y'].mean()
mean_control = df[df['treated'] == 0]['y'].mean()
naive_estimate = mean_treated - mean_control

print(f"\n=== 관찰 가능한 단순 비교 ===")
print(f"처치군 평균: {mean_treated:.2f}")
print(f"대조군 평균: {mean_control:.2f}")
print(f"단순 차이:   {naive_estimate:.2f}")
print(f"→ ATE({ate:.2f})보다 큼. 동기 높은 유저가 처치군에 많아서!")

# === 그룹별 효과 ===
for g in ['high_motivation', 'low_motivation']:
    sub = df[df['group'] == g]
    eff = sub[sub['treated'] == 1]['y'].mean() - sub[sub['treated'] == 0]['y'].mean()
    print(f"\n{g} 그룹 추정 효과: {eff:.2f} (진짜: {sub['true_effect'].iloc[0]:.1f})")
```
