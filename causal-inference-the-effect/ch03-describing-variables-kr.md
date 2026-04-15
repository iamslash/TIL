# Chapter 3: 변수 설명하기 (Describing Variables)

> **핵심 질문**: 데이터를 어떻게 요약하고 설명하는가?

---

## 3.1 변수란?

**변수(Variable)** = 같은 것을 여러 번 관찰한 값들의 집합

예시:
- 433명 남아프리카인의 월소득
- 프랑스의 연도별 기업 합병 건수 (1984~2014)
- 744명 아이의 신경증 점수
- 532개 꽃의 색깔

---

## 3.2 변수의 유형

| 유형 | 설명 | 예시 |
|------|------|------|
| **연속형 (Continuous)** | 어떤 값이든 가능 | 월소득, 키, 체중 |
| **카운트 (Count)** | 0 이상 정수 | 합병 건수, 주문 수 |
| **순서형 (Ordinal)** | 순서는 있지만 간격이 불명확 | 교육 수준 (고졸 < 대졸 < 석사) |
| **범주형 (Categorical)** | 순서 없는 분류 | 꽃 색깔, 국가, 직업 |
| **이진형 (Binary)** | 두 값만 가능 | 군 복무 여부, 약 투여 여부 |
| **질적 (Qualitative)** | 텍스트 등 비정형 | 신문 헤드라인, 리뷰 텍스트 |

### 소프트웨어 엔지니어라면

이미 친숙한 개념이다:
```python
# 연속형 → float
salary = 4500000.50

# 카운트 → int (non-negative)
order_count = 42

# 순서형 → enum with ordering
class Education(Enum):
    HIGH_SCHOOL = 1
    BACHELOR = 2
    MASTER = 3

# 범주형 → enum without ordering
class Color(Enum):
    RED = "red"
    BLUE = "blue"
    GREEN = "green"

# 이진형 → bool
is_subscriber = True
```

---

## 3.3 분포 (Distribution)

**분포** = 각 값이 나타날 확률을 설명한 것

### 범주형 변수의 분포: 빈도표

```
미국 대학 학위 유형 분포:
  2년 미만: 3,495개교 (47%)
  2년제:    1,647개교 (22%)
  4년제+:   2,282개교 (31%)
```

### 연속형 변수의 분포: 히스토그램과 밀도 곡선

- **히스토그램**: 값의 범위를 구간(bin)으로 나누고 빈도를 표시
- **밀도 곡선 (Density Plot)**: 히스토그램을 무한히 부드럽게 만든 것

---

## 3.4 분포 요약하기

### 중심을 나타내는 지표

#### 평균 (Mean)

```
데이터: 2, 5, 5, 6
평균 = (2 + 5 + 5 + 6) / 4 = 4.5
```

#### 중앙값 (Median, 50번째 백분위수)

절반은 이 값보다 작고, 절반은 크다.

**언제 평균 vs 중앙값?**

| 상황 | 적합한 지표 |
|------|-----------|
| 대칭 분포 (키, 시험 점수) | 평균 ≈ 중앙값, 아무거나 |
| 오른쪽 꼬리 (소득, 자산) | **중앙값** (극단값에 덜 민감) |

> 제프 베조스가 포함된 방의 평균 자산은 $15,000,000 이지만,
> 중앙값은 $90,000 이다. 보통 사람을 대표하는 것은 중앙값이다.

### 퍼짐을 나타내는 지표

#### 분산 (Variance)

```
데이터: 2, 5, 5, 6   평균: 4.5

1단계: 평균과의 차이    → -2.5, 0.5, 0.5, 1.5
2단계: 제곱            → 6.25, 0.25, 0.25, 2.25
3단계: 합계            → 9.0
4단계: (N-1)로 나누기  → 9.0 / 3 = 3.0
```

**문제**: 단위가 "달러의 제곱" 같은 이상한 것이 된다.

#### 표준편차 (Standard Deviation)

분산의 제곱근. 원래 단위로 돌아온다.

```
분산 = 3.0
표준편차 = √3.0 = 1.73
```

**활용**: "이 값이 평균에서 몇 표준편차 떨어져 있는가?"

> 대학 졸업자 평균 수입: $33,349, 표준편차: $12,381
> 어떤 대학 $38,000 → 평균에서 (38000-33349)/12381 = **0.38 표준편차** 위

#### 사분위범위 (IQR)

75번째 백분위수 - 25번째 백분위수. 가운데 50%가 얼마나 퍼져있는가.

### 비대칭 (Skew)

- **오른쪽 비대칭 (Right skew)**: 대부분 작은 값, 소수의 큰 값 → 소득 분포
- **왼쪽 비대칭 (Left skew)**: 반대
- **대칭 (Symmetric)**: 양쪽 균형 → 키 분포

**로그 변환**: 오른쪽 비대칭을 대칭에 가깝게 만들어준다.
- `log(소득)` 은 대략 정규분포를 따른다
- 로그 값이 0.01 증가 ≈ 원래 값이 1% 증가

---

## 3.5 이론적 분포

### 표본 vs 모집단

- **표본 (Sample)**: 내가 실제로 수집한 데이터
- **모집단 (Population)**: 관심 대상 전체
- 표본이 커질수록 모집단을 더 잘 대표한다

### 표기법

| 표기 | 의미 | 예시 |
|------|------|------|
| 영어/라틴 문자 | 데이터 | x = 관찰값 |
| 영어 + 바 | 데이터 계산 | x̄ = 표본 평균 |
| 그리스 문자 | 진실 (모수) | μ = 모집단 평균, σ = 모집단 표준편차 |
| 그리스 + 모자(hat) | 진실의 추정 | μ̂ = 추정된 평균 |

### 가설 검정 예제

> 농구 선수 100명 표본: 평균 102점, 표준편차 30
> 질문: 모집단 평균이 90일 수 있는가?

```
귀무가설: μ = 90
표준오차 = 30 / √100 = 3
Z = (102 - 90) / 3 = 4  (평균에서 4 표준편차)
→ 이런 결과가 나올 확률: 약 0.008%
→ μ = 90은 매우 가능성 낮음 → 기각
```

### 주요 이론적 분포

| 분포 | 특징 | 현실 예시 |
|------|------|----------|
| **정규분포** | 대칭, 종 모양 | 키, 지능 |
| **로그정규분포** | 오른쪽 비대칭, 로그 취하면 정규 | 소득, 자산 |
| **멱법칙분포** | 극단값이 자주 나타남 | 주가 폭락, SNS 팔로워 수 |

---

## Python 예제: 변수 설명의 모든 것

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# 소득 데이터 시뮬레이션 (로그정규분포 - 현실적)
log_income = np.random.normal(10.5, 0.8, 5000)  # log(소득)
income = np.exp(log_income)

df = pd.DataFrame({'income': income, 'log_income': log_income})

# --- 기본 요약 통계 ---
print("=== 소득 요약 통계 ===")
print(f"평균:       ${df['income'].mean():,.0f}")
print(f"중앙값:     ${df['income'].median():,.0f}")
print(f"표준편차:   ${df['income'].std():,.0f}")
print(f"25번째 %:   ${df['income'].quantile(0.25):,.0f}")
print(f"75번째 %:   ${df['income'].quantile(0.75):,.0f}")
print(f"IQR:        ${df['income'].quantile(0.75) - df['income'].quantile(0.25):,.0f}")
print(f"\n→ 평균이 중앙값보다 훨씬 높다 = 오른쪽 비대칭 (소수의 고소득자 때문)")

# --- 히스토그램: 원본 vs 로그 변환 ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df['income'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_title('Income Distribution (Right Skewed)')
axes[0].set_xlabel('Income ($)')

axes[1].hist(df['log_income'], bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[1].set_title('Log(Income) Distribution (Approx. Normal)')
axes[1].set_xlabel('Log(Income)')

plt.tight_layout()
plt.savefig('income_distribution.png', dpi=100)
plt.show()

# --- 가설 검정 예제 ---
sample = df['income'].sample(100, random_state=42)
sample_mean = sample.mean()
sample_std = sample.std()
n = len(sample)
se = sample_std / np.sqrt(n)

# 귀무가설: 모집단 평균 = $30,000
null_mu = 30000
z_stat = (sample_mean - null_mu) / se
p_value = 2 * (1 - __import__('scipy').stats.norm.cdf(abs(z_stat)))

print(f"\n=== 가설 검정 ===")
print(f"표본 평균: ${sample_mean:,.0f}")
print(f"표준오차:  ${se:,.0f}")
print(f"Z-통계량:  {z_stat:.2f}")
print(f"p-value:   {p_value:.4f}")
print(f"→ p < 0.05? {p_value < 0.05} → {'귀무가설 기각' if p_value < 0.05 else '귀무가설 유지'}")
```
