# Chapter 11: 모델링 없는 인과추론 (Causality with Less Modeling)

> **핵심 질문**: 인과 다이어그램을 완벽히 그릴 수 없을 때, 어떻게 하는가?

---

## 11.1 자신감의 한계

Ch5~8에서는 DGP를 인과 다이어그램으로 그리고, 경로를 분석하고, 백도어를 닫는
전략을 배웠다. 하지만 현실에서는:

> "알고 있는 것을 그리고, 나머지는 '글쎄, 뭔가 있겠지'라고 표시하고,
> 거기서 거의 모든 곳으로 화살표를 그어야 한다."

완벽한 다이어그램은 불가능. 그래서 **완벽하지 않아도 쓸 수 있는 전략**이 필요하다.

---

## 11.2 열린 공간: 축소된 모형 (Reduced Form)

DGP를 완전히 모델링하는 대신, 세 가지만 놓는다:

```
    Stuff (온갖 교란 변수)
   ↙              ↘
Treatment        Outcome
    └──────→────────┘
```

"Stuff"에 정확히 뭐가 있는지 모르지만, 그것을 **간접적으로 통제**할 수 있다.

### 간접 통제 전략 4가지

#### 1. 프론트도어 방법 (Ch9)

외생적 변동을 찾아서 Stuff와 무관한 처치 변동만 사용.

#### 2. 계층적 통제 (Hierarchical Controls)

한 변수를 통제하면, 그 아래 변수들이 자동으로 통제된다.

```
예: "어린 시절 살던 집"을 통제하면
→ 동네, 도시/시골, 지역 모두 자동 통제
→ 이 모든 변수를 일일이 측정할 필요 없음
```

**고정효과(Fixed Effects)**: 같은 사람을 여러 시점에서 관찰하면, 그 사람의
시간 불변 특성(성격, 성장 배경 등)이 자동 통제된다. → Ch16

#### 3. 비교 가능한 그룹

처치군과 대조군이 **모든 교란 변수에서 비슷**하다고 가정.

- **실험**: 무작위 배정이 이것을 보장
- **관찰 연구**: 특정 비교에서만 가능

```
예: 지역사회 보조금의 효과
나쁜 비교: 보조금 받은 지역 vs 전체 비교
좋은 비교: 보조금 신청한 지역 중 받은 곳 vs 안 받은 곳
→ 신청 지역끼리는 특성이 비슷할 가능성 높음
```

#### 4. 변화량 비교

절대값이 아닌 **변화량**을 비교하면 가정이 약해진다.

```
가정 전환:
"두 그룹은 모든 Stuff에서 같다"
→ "두 그룹에서 Stuff의 변화량이 같다" (더 약한 가정)
→ 이것이 이중차분법(DiD)의 기본 아이디어 (Ch18)
```

---

## 11.3 틀렸어도, 얼마나 틀렸는가?

모든 가정은 어느 정도 틀리다. 문제는 **"얼마나"** 틀렸는가.

### 강건성 검정 (Robustness Test)

> 가정을 완화했을 때 결과가 얼마나 변하는지 확인

```
1. 가정 A를 세운다
2. 가정 A가 맞다면, 데이터에서 관계 X가 0이어야 한다
3. 관계 X를 실제로 확인한다
4. 0이 아니면 → 가정 A에 문제가 있다
```

### 위약 검정 (Placebo Test)

> 처치를 **받지 않은** 그룹에 **가짜 처치**를 적용하고, 효과가 0인지 확인

```
예: 전기 사용량 경고 편지의 효과
진짜: 1201-1250 kWh (편지 받음) vs 1151-1200 kWh (안 받음)
위약: 1151-1200 kWh (가짜 처치) vs 1101-1150 kWh (가짜 대조)
→ 가짜에서 효과가 0이 아니면? 두 그룹이 원래 다른 사람들!
```

### 부분 식별 (Partial Identification)

정확한 점추정 대신 **범위**를 제시

```
예: 스포츠카 → 과속 확률 +5%p (관찰)
교란: 위험 추구 성향 (스포츠카 구매 + 과속 모두에 영향)
→ 위험 추구를 통제하면 효과는 줄어들 것 → "5%p 이하"
→ 추가 가정: 위험 추구의 영향이 0~X 사이 → "2~5%p"
```

### 최후의 방어선: 직관 (Plausibility Check)

> "이 결과가 현실에서 가능한가?"

- 교육 개입의 효과가 0.1 표준편차를 넘기 어려움
- 30초짜리 실험실 메시지가 5년 행동을 바꿈 → 비현실적
- 물을 마시면 수명이 20년 늘어남 → 불가능

> "그래서 마지막 방어선은 당신의 직관이다. 이 결과가 가능한가?
> 가능하지 않다면, 실제로도 아닐 것이다."

---

## Python 예제: 위약 검정

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)
n = 5000

# DGP: 전기 사용량별 유저 유형이 다름
user_type = np.random.normal(0, 1, n)  # 높을수록 에너지 다소비 유형
usage = 1100 + 50 * user_type + np.random.normal(0, 30, n)

# 1200 kWh 이상이면 경고 편지 발송
letter = (usage >= 1200).astype(int)

# 편지의 효과: 다음 달 사용량 -30 (진짜 효과)
next_usage = usage - 30 * letter + np.random.normal(0, 20, n)

df = pd.DataFrame({
    'usage': usage,
    'letter': letter,
    'next_usage': next_usage,
    'user_type': user_type
})

# === 진짜 처치 효과 (1200 기준) ===
near_threshold = df[(df['usage'] >= 1170) & (df['usage'] <= 1230)]
model_real = smf.ols('next_usage ~ letter', data=near_threshold).fit()
print("=== 진짜 처치 (1200 kWh 기준) ===")
print(f"편지 효과: {model_real.params['letter']:.1f}")
print(f"p-value: {model_real.pvalues['letter']:.4f}")
print("→ 유의한 효과 (진짜)")

# === 위약 검정 (1100 기준 - 편지 없음) ===
placebo_group = df[(df['usage'] >= 1070) & (df['usage'] <= 1130)]
placebo_group = placebo_group.copy()
placebo_group['fake_letter'] = (placebo_group['usage'] >= 1100).astype(int)
model_placebo = smf.ols('next_usage ~ fake_letter', data=placebo_group).fit()
print(f"\n=== 위약 검정 (1100 kWh 기준, 편지 없음) ===")
print(f"가짜 편지 효과: {model_placebo.params['fake_letter']:.1f}")
print(f"p-value: {model_placebo.pvalues['fake_letter']:.4f}")
print("→ 유의하지 않으면 → 가정 통과!")
print("→ 유의하면 → 경계선 양쪽 유저가 다른 유형 → 가정 위반!")
```
