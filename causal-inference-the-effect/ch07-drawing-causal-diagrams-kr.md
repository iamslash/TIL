# Chapter 7: 인과 다이어그램 그리기 (Drawing Causal Diagrams)

> **핵심 질문**: 현실의 복잡한 DGP를 어떻게 적절한 인과 다이어그램으로 옮기는가?

---

## 7.1 세상에 대한 이해

인과 다이어그램을 그리려면, 먼저 **해당 주제에 대해 많이 공부**해야 한다.

- 관련 논문을 읽고
- 현장 전문가와 이야기하고
- 제도와 정책의 작동 방식을 이해해야 한다

다이어그램은 **기법이 아니라 도메인 지식의 표현**이다.

---

## 7.2 DGP를 생각하기

### 예제: 온라인 수업과 대학 중퇴

**연구 질문**: "온라인 수업을 듣는 것이 커뮤니티 칼리지 중퇴율에 영향을 주는가?"

### 단계 1: 변수 나열

```
처치 변수 (Treatment): OnlineClass
결과 변수 (Outcome):   Dropout
```

### 단계 2: 원인 나열

**온라인 수업을 듣는 이유:**
- 학생 선호도 (대면 vs 온라인)
- 가용 시간
- 인터넷 접근성
- 배경 (인종, 성별, 나이, 사회경제적 지위)
- 근무 시간

**중퇴의 원인:**
- 인종, 성별, 나이, 사회경제적 지위
- 학업 성적
- 근무 시간
- 가용 시간

### 단계 3: 양쪽에 겹치는 변수 → 교란 변수 후보

```
겹치는 것: 배경(인종, 성별 등), 가용 시간, 근무 시간
→ 이것들이 Back Door Path를 만든다
```

### 단계 4: 효과의 크기 판단

모든 변수를 넣을 수는 없다. **효과가 작은 것은 과감히 제외**.

> "조용한 카페가 근처에 있으면 온라인 수업을 더 선택할 수도 있지만,
> 평균적으로 아주 작은 효과일 것이다. 제외."

### 단계 5: 관찰 불가능한 공통 원인 추가

두 변수가 상관있지만 인과가 아닐 때 → U1, U2 등으로 표시

---

## 7.3 단순화하기

> "주유소 가는 길을 물었더니, 1 제곱마일 단위의 거대한 지도책을 건네는 것과
> 같다. 필요한 건 '고속도로 2번 출구, 웬디스 옆'이다."

### 4가지 단순화 기준

#### 1. 중요하지 않은 변수 (Unimportance)

효과가 아주 작은 변수 → 제거

#### 2. 중복 변수 (Redundancy)

들어오는 화살표와 나가는 화살표가 **동일한** 변수들 → 합치기

```
Before: Gender → OnlineClass, Gender → Dropout
        Race → OnlineClass,   Race → Dropout
After:  Demographics → OnlineClass
        Demographics → Dropout
```

#### 3. 매개 변수 (Mediator)

A → B → C 인데, B에 다른 연결이 없으면 → B 제거, A → C 직접 연결

```
Before: Demographics → Preferences → OnlineClass
After:  Demographics → OnlineClass
```

**주의**: 매개 변수가 분석에 중요한 도구일 수 있다! (예: 도구변수)
무조건 제거하면 안 된다.

#### 4. 무관한 변수 (Irrelevance)

처치와 결과 사이 경로에 없는 변수 → 제거 (Ch8에서 상세)

---

## 7.4 순환 피하기 (Avoiding Cycles)

인과 다이어그램에는 **순환(cycle)이 있으면 안 된다**.

```
잘못됨: A → B → C → A  (순환!)
```

하지만 현실에는 피드백 루프가 있다:
- 부자는 더 부자가 된다
- 네가 나를 때리면, 내가 너를 때린다

### 해결: 시간 차원 도입

```
너가 나를 때림 (t=0) → 내가 너를 때림 (t=1) → 너가 나를 때림 (t=2)
```

시간은 한 방향으로만 흐르므로 순환이 사라진다.

---

## 7.5 가정과 친해지기

인과 다이어그램은 **필연적으로 가정을 포함**한다.

### 가정을 다루는 법

1. **비판적 관점 취하기**: "회의적인 독자가 이 가정을 왜 거부할까?"
2. **근거 제시**: 선행 연구, 데이터 상관, 전문가 의견
3. **외부 검토**: 다른 사람에게 다이어그램을 보여주고 피드백 받기
4. **테스트 가능한 함의 확인**: 다이어그램이 맞다면 특정 변수들 사이 상관이
   0이어야 한다 → 실제 데이터에서 확인

---

## Python 예제: 인과 다이어그램의 테스트 가능한 함의

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

np.random.seed(42)
n = 2000

# DGP (진짜 구조)
# Demographics → OnlineClass, Dropout
# AvailableTime → OnlineClass, Dropout
# OnlineClass → Dropout (진짜 효과: +5%p)
demographics = np.random.normal(0, 1, n)
available_time = np.random.normal(0, 1, n)

online_prob = 1 / (1 + np.exp(-(0.3 * demographics - 0.5 * available_time)))
online_class = np.random.binomial(1, online_prob, n)

dropout_prob = 1 / (1 + np.exp(-(
    -1.5
    + 0.05 * online_class   # 진짜 효과 (작음)
    - 0.4 * demographics
    - 0.3 * available_time
)))
dropout = np.random.binomial(1, dropout_prob, n)

df = pd.DataFrame({
    'online_class': online_class,
    'dropout': dropout,
    'demographics': demographics,
    'available_time': available_time
})

# --- 다이어그램의 테스트 가능한 함의 ---
# 가정: demographics와 available_time은 서로 원인이 아님
# → 상관이 0에 가까워야 함
corr = df['demographics'].corr(df['available_time'])
print(f"demographics ↔ available_time 상관: {corr:.4f}")
print(f"→ 0에 가까움 → 다이어그램 가정과 일치!\n")

# --- 통제 없이 vs 통제 후 ---
model1 = smf.ols('dropout ~ online_class', data=df).fit()
model2 = smf.ols('dropout ~ online_class + demographics + available_time', data=df).fit()

print(f"=== 통제 없이 ===")
print(f"online_class 계수: {model1.params['online_class']:.4f}\n")

print(f"=== demographics + available_time 통제 후 ===")
print(f"online_class 계수: {model2.params['online_class']:.4f}")
print(f"→ 교란 변수를 통제하면 진짜 효과에 가까워진다")
```

---

## 소프트웨어 엔지니어를 위한 비유

인과 다이어그램 = **시스템 아키텍처 다이어그램**

| 인과 다이어그램 | 시스템 다이어그램 |
|--------------|----------------|
| 변수 (노드) | 서비스/컴포넌트 |
| 인과 화살표 | API 호출/데이터 흐름 |
| 교란 변수 | 공유 의존성 (공통 DB, 캐시) |
| 단순화 | 마이크로서비스 경계 설정 |
| 순환 금지 | 순환 참조 금지 |

둘 다 **핵심만 남기고 단순화**하면서도, **중요한 의존성을 빠뜨리지 않는 것**이 관건이다.
