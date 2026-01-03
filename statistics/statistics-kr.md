- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
  - [1. 변수(Variable)와 변량(Variate)](#1-변수variable와-변량variate)
  - [2. 평균(Mean)](#2-평균mean)
  - [3. 분산(Variance)과 표준편차(Standard Deviation)](#3-분산variance과-표준편차standard-deviation)
  - [4. 정규분포(Normal Distribution)](#4-정규분포normal-distribution)
  - [5. 독립변수와 종속변수](#5-독립변수와-종속변수)
  - [6. 상관관계(Correlation) vs 인과관계(Causation)](#6-상관관계correlation-vs-인과관계causation)
  - [7. 가설 검정(Hypothesis Testing)과 p-값](#7-가설-검정hypothesis-testing과-p-값)
    - [가설 검정이란?](#가설-검정이란)
    - [p-값이란?](#p-값이란)
  - [8. 귀무가설(Null Hypothesis)과 대립가설(Alternative Hypothesis)](#8-귀무가설null-hypothesis과-대립가설alternative-hypothesis)
    - [왜 복잡한 절차를 사용할까?](#왜-복잡한-절차를-사용할까)
    - [가설 검정 절차](#가설-검정-절차)
  - [가설검정 예](#가설검정-예)
    - [상황](#상황)
    - [Step 1: 가설 설정](#step-1-가설-설정)
    - [Step 2: 실험 설계 및 데이터 수집](#step-2-실험-설계-및-데이터-수집)
    - [Step 3: p-값 계산](#step-3-p-값-계산)
    - [Step 4: 의사결정](#step-4-의사결정)
    - [Step 5: 결론](#step-5-결론)
    - [전체 과정 요약](#전체-과정-요약)
    - [주의사항](#주의사항)
- [Advanced](#advanced)
  - [통계학 계층도](#통계학-계층도)

-----

# Abstract

통계학(Statistics)은 **데이터를 수집, 정리, 분석하여 의미 있는 정보를 추출**하고, **불확실성 속에서 합리적인 의사결정**을 내리기 위한 학문입니다.

**통계의 핵심 개념:**
- **변수와 변량**: 변하는 값과 그 구체적인 관측값
- **평균, 분산, 표준편차**: 데이터의 중심과 퍼진 정도
- **정규분포**: 자연현상의 보편적 패턴
- **상관관계 vs 인과관계**: 관계가 있다 ≠ 원인이다
- **가설검정과 p-값**: 과학적 결론을 내리는 방법

# Materials

- [쉽게 배우는 데이터와 AI | 찹쓰 | 통계 딥러닝 | youtube](https://www.youtube.com/@data_chopsticks/videos)
- [[인공지능을 위한 머신러닝101] 머신러닝을 위한 기초통계개념들을 소개합니다 | youtube](https://www.youtube.com/watch?v=q8wCazJhhKo)

# Basic

## 1. 변수(Variable)와 변량(Variate)

**변수(Variable)**는 변하는 값으로, 숫자로 표현될 수 있고 값이 달라질 수 있는 것입니다.

**변량(Variate)**은 변수가 취할 수 있는 구체적인 값들입니다.

**예시: 농부의 10년간 수확량**
```
변수: 수확량
변량: 450kg, 620kg, 530kg, 710kg, ...
```

## 2. 평균(Mean)

**평균**은 모든 값을 더한 뒤 개수로 나눈 값으로, 데이터의 중심을 대표합니다.

```
평균 (μ 또는 x̄) = (x₁ + x₂ + ... + xₙ) / n

예: (450 + 620 + 530 + 710 + ... + 490) / 10 = 533.9 kg
```

## 3. 분산(Variance)과 표준편차(Standard Deviation)

**분산**은 각 값이 평균에서 얼마나 떨어져 있는지를 제곱한 값들의 평균입니다.

**표준편차**는 분산의 제곱근으로, 원래 단위로 환산한 값입니다.

```
분산 (σ²) = Σ(xᵢ - μ)² / n
표준편차 (σ) = √분산

예: 표준편차 118.2 kg
→ 대부분의 수확량은 평균 ± 118.2 kg 범위 (415.7 ~ 652.1 kg)
```

**왜 표준편차를 사용할까?**
분산은 단위가 kg²이 되어 직관적이지 않지만, 표준편차는 원래 단위(kg)로 돌아와 이해하기 쉽습니다.

## 4. 정규분포(Normal Distribution)

**정규분포**는 종 모양을 이루는 확률 분포로, 많은 자연현상이 이 분포를 따릅니다.

```
        *
      *   *
    *       *
  *           *
 *_____________*
      μ
```

**68-95-99.7 규칙:**
- 평균 ± 1σ: 약 68%의 데이터
- 평균 ± 2σ: 약 95%의 데이터
- 평균 ± 3σ: 약 99.7%의 데이터

**정규분포를 따르는 자연현상:** 사람의 키, 시험 점수, 측정 오차, 수확량 등

## 5. 독립변수와 종속변수

```
영향을 주는 변수 → 영향을 받는 변수
   독립변수    →    종속변수
   (원인)     →     (결과)
```

**예시:**
```
비료의 양 (독립변수) → 수확량 (종속변수)
```

## 6. 상관관계(Correlation) vs 인과관계(Causation)

**상관관계**는 두 변수가 함께 변하는 경향을 말합니다.

**중요한 원칙:**
> "상관관계가 있다고 해서 반드시 인과관계가 있는 것은 아닙니다"

**가짜 상관관계 예시:**
```
아이스크림 판매량 ↑ ↔ 익사 사고 ↑

강한 상관관계가 있지만, 인과관계는 없습니다!
실제 원인: 기온 (제3의 변수)
  여름 → 기온 ↑ → 아이스크림 판매 ↑
  여름 → 기온 ↑ → 수영 ↑ → 익사 ↑
```

**진짜 인과관계 확인 방법:**
1. 통제된 실험 (다른 조건은 동일하게)
2. 시간적 선후관계 (원인이 결과보다 먼저)
3. 제3의 변수 배제
4. 메커니즘 설명 가능

## 7. 가설 검정(Hypothesis Testing)과 p-값

### 가설 검정이란?

실험 결과가 우연인지, 실제 효과인지 판단하는 과학적 방법입니다.

**예시: 비료 실험**
```
가설: "비료의 양을 20% 늘리면 수확량이 증가할 것이다"

실험 결과:
- 대조군 (비료 100g): 평균 525kg
- 실험군 (비료 120g): 평균 617kg
- 차이: 92kg

질문: 이 차이가 우연일까, 실제 효과일까?
```

### p-값이란?

**p-값**은 **"효과가 없다"고 가정했을 때, 관찰된 결과(또는 더 극단적인 결과)가 우연히 발생할 확률**입니다.

```
p = 0.03의 의미:
"비료가 실제로 효과가 없다고 가정할 때,
우리가 본 결과가 우연히 나올 확률이 3%"

→ 우연치고는 너무 희귀함 (3%)
→ "효과가 없다"는 가정을 믿기 어려움
→ 비료가 실제로 효과가 있다고 결론
```

**유의수준 (α = 0.05):**
```
p < 0.05: 통계적으로 유의함 → 효과 있음
p ≥ 0.05: 통계적으로 유의하지 않음 → 효과 불명확
```

## 8. 귀무가설(Null Hypothesis)과 대립가설(Alternative Hypothesis)

### 왜 복잡한 절차를 사용할까?

**과학적 보수주의** 때문입니다. 법정의 무죄 추정의 원칙과 유사합니다:

```
법정:
기본 가정: 피고인은 무죄 (보수적)
증명 책임: 검사가 유죄 입증
원칙: "아홉 명의 범죄자를 놓쳐도, 한 명의 억울한 피해자를 막자"

과학:
기본 가정: 효과 없음 (보수적)
증명 책임: 연구자가 효과 있음 입증
원칙: "아홉 개의 발견을 놓쳐도, 한 개의 거짓 발견을 막자"
```

### 가설 검정 절차

**Step 1: 가설 설정**
```
귀무가설 (H₀): "비료는 효과가 없다" (보수적 가정, 기각하려는 것)
대립가설 (H₁): "비료는 효과가 있다" (증명하고 싶은 것)
```

**Step 2: 실험 및 데이터 수집**

**Step 3: p-값 계산**
```
H₀가 참이라고 가정
→ 관찰된 결과가 나올 확률 계산
→ p-값
```

**Step 4: 의사결정**
```
p < 0.05:
→ 귀무가설 기각 (Reject H₀)
→ 대립가설 채택 (Accept H₁)
→ "효과가 있다" ✓

p ≥ 0.05:
→ 귀무가설 기각 실패
→ "효과가 있다고 말할 수 없다"
```

## 가설검정 예

농부의 비료 실험을 통해 가설검정의 전체 과정을 단계별로 이해해봅시다.

### 상황

농부는 10년간 평균 534kg의 수확량을 기록했습니다. 이번에 새로운 비료를 구입했는데, 판매자는 "이 비료를 사용하면 수확량이 증가합니다"라고 주장합니다.

**농부의 질문:** "이 비료가 정말 효과가 있을까?"

### Step 1: 가설 설정

```
귀무가설 (H₀): 비료는 수확량에 영향이 없다
              μ_실험군 = μ_대조군

대립가설 (H₁): 비료는 수확량을 증가시킨다
              μ_실험군 > μ_대조군

유의수준 (α): 0.05 (5%)
```

**왜 이렇게 설정할까?**
- 귀무가설 (H₀): 보수적으로 "효과 없음"으로 시작
- 대립가설 (H₁): 증명하고 싶은 것 "효과 있음"
- 유의수준 α = 0.05: 5%의 오류는 감수 (Type I Error)

### Step 2: 실험 설계 및 데이터 수집

**실험 설계:**

```
똑같은 논 10개를 준비하여 무작위로 나눔:

대조군 (5개 논):
- 기존 비료 100g 사용
- 다른 조건 모두 동일 (물, 햇빛, 씨앗 등)

실험군 (5개 논):
- 새 비료 100g 사용
- 다른 조건 모두 동일
```

**수확 결과 (kg):**

```
대조군 (기존 비료):
논1: 520kg
논2: 530kg
논3: 510kg
논4: 540kg
논5: 525kg
→ 평균: 525kg

실험군 (새 비료):
논1: 610kg
논2: 625kg
논3: 605kg
논4: 630kg
논5: 615kg
→ 평균: 617kg

차이: 617 - 525 = 92kg (약 17.5% 증가!)
```

### Step 3: p-값 계산

**핵심 질문:**
> "만약 비료가 정말 효과가 없다면 (H₀가 참이라면), 이렇게 큰 차이(92kg)가 우연히 나올 확률은 얼마나 될까?"

**통계 검정 (t-test):**

```
1. 두 그룹의 평균 차이: 92kg
2. 각 그룹의 변동성 고려
3. t-통계량 계산
4. p-값 도출

결과: p-값 = 0.0001 (0.01%)
```

**p-값의 의미:**

```
p = 0.0001 = 0.01%

해석:
"비료가 효과가 없다고 가정했을 때,
이렇게 큰 차이가 우연히 나올 확률이 0.01%"

비유:
1000번 실험하면 997번은 작은 차이만 나타나고,
3번만 이렇게 큰 차이가 나타남
```

### Step 4: 의사결정

**판단 기준:**

```
p-값 (0.0001) < α (0.05) ?

0.0001 < 0.05 ✓ (만족!)

→ 귀무가설 기각 (Reject H₀)
→ 대립가설 채택 (Accept H₁)
```

**논리적 추론:**

```
1. 전제: "비료가 효과 없다면, 92kg 차이는 0.01% 확률"
2. 관찰: "그런데 실제로 92kg 차이가 나타남"
3. 결론: "0.01%의 희귀한 우연? 아니면 진짜 효과?"
        → 후자가 훨씬 설득력 있음!
        → H₀ 기각
```

### Step 5: 결론

**통계적 결론:**

```
✓ 새 비료는 수확량을 통계적으로 유의하게 증가시킨다
  (p < 0.05, 평균 92kg 증가, 약 17.5%)
```

**실용적 해석:**

```
1. 통계적 유의성: 효과가 우연이 아님 (p < 0.05)
2. 효과 크기: 92kg 증가 (약 17.5%)
3. 실용적 가치:
   - 비료 추가 비용: ?
   - 수확량 증가 가치: 92kg × 쌀 가격
   → 경제적 타당성 검토 필요
```

### 전체 과정 요약

```
Step 1: 가설 설정
   H₀: 효과 없음 (보수적 시작)
   H₁: 효과 있음 (증명하려는 것)
   ↓
Step 2: 실험
   대조군 vs 실험군 비교
   무작위 배정, 다른 조건 통제
   ↓
Step 3: 데이터 수집
   대조군: 525kg
   실험군: 617kg
   차이: 92kg
   ↓
Step 4: p-값 계산
   "H₀가 참이면 이런 결과가 나올 확률?"
   p = 0.0001 (매우 희귀!)
   ↓
Step 5: 의사결정
   p < 0.05 ?
   Yes! → H₀ 기각
   ↓
Step 6: 결론
   새 비료는 효과가 있다! ✓
```

### 주의사항

**통계적으로 유의 ≠ 실용적으로 가치**

```
예시:
- p-값 = 0.001 (통계적으로 매우 유의)
- 증가량 = 1kg (효과는 매우 작음)
- 비료 추가 비용 = 10만원

→ 통계적으로는 유의하지만, 경제적으로는 손해!
```

**Type I Error (위양성) 가능성:**

```
α = 0.05 의미:
→ 100번 실험 중 5번은 실수로 "효과 있음"이라고 결론
→ 실제로는 효과 없는데도!

이것이 과학적 보수주의의 비용
```

**재현성 중요:**

```
1번의 실험만으로 확신하지 말 것!
- 여러 계절에 반복
- 다른 지역에서 검증
- 독립적 연구팀의 확인
```

# Advanced

## 통계학 계층도

```
통계학 (Statistics)
│
├─ 📊 기술통계학 (Descriptive Statistics)
│  ├─ 중심경향 측도
│  │  ├─ 평균 (Mean)
│  │  ├─ 중앙값 (Median)
│  │  └─ 최빈값 (Mode)
│  │
│  ├─ 산포도 측도
│  │  ├─ 분산 (Variance)
│  │  ├─ 표준편차 (Standard Deviation)
│  │  ├─ 범위 (Range)
│  │  └─ 사분위수 (Quartiles)
│  │
│  └─ 분포 특성
│     ├─ 정규분포 (Normal Distribution)
│     ├─ 왜도 (Skewness)
│     └─ 첨도 (Kurtosis)
│
├─ 🔬 추론통계학 (Inferential Statistics)
│  ├─ 가설검정 (Hypothesis Testing)
│  │  ├─ t-test
│  │  ├─ ANOVA (Analysis of Variance)
│  │  ├─ Chi-square test
│  │  ├─ p-value
│  │  └─ 신뢰구간 (Confidence Interval)
│  │
│  ├─ 표본이론 (Sampling Theory)
│  │  ├─ 표본추출 방법
│  │  ├─ 표본분포
│  │  └─ 중심극한정리
│  │
│  └─ 추정 (Estimation)
│     ├─ 점추정 (Point Estimation)
│     └─ 구간추정 (Interval Estimation)
│
├─ 📈 상관 및 회귀분석 (Correlation & Regression)
│  ├─ 상관분석 (Correlation Analysis)
│  │  ├─ Pearson 상관계수
│  │  ├─ Spearman 상관계수
│  │  └─ ⚠️ 상관관계 ≠ 인과관계
│  │
│  └─ 회귀분석 (Regression Analysis)
│     ├─ 단순 선형회귀 (Simple Linear Regression)
│     ├─ 다중 회귀 (Multiple Regression)
│     ├─ 로지스틱 회귀 (Logistic Regression)
│     ├─ 다항 회귀 (Polynomial Regression)
│     └─ 비선형 회귀 (Nonlinear Regression)
│
├─ ⏱️ 시계열 분석 (Time Series Analysis)
│  ├─ ARIMA 모델
│  ├─ 지수평활법 (Exponential Smoothing)
│  ├─ 계절성 분석
│  └─ 예측 (Forecasting)
│
├─ 🎲 확률론 (Probability Theory)
│  ├─ 확률분포
│  │  ├─ 이산분포 (Binomial, Poisson 등)
│  │  └─ 연속분포 (Normal, Exponential 등)
│  │
│  ├─ 조건부 확률
│  ├─ 베이즈 정리
│  └─ 확률변수
│
├─ 🔀 다변량 분석 (Multivariate Analysis)
│  ├─ 주성분 분석 (PCA)
│  ├─ 요인분석 (Factor Analysis)
│  ├─ 판별분석 (Discriminant Analysis)
│  ├─ 군집분석 (Cluster Analysis)
│  └─ 다차원척도법 (MDS)
│
├─ 📊 베이지안 통계학 (Bayesian Statistics)
│  ├─ 사전확률 (Prior)
│  ├─ 사후확률 (Posterior)
│  ├─ 베이즈 추론
│  └─ MCMC 방법
│
├─ 🎯 실험계획법 (Design of Experiments)
│  ├─ 완전무작위 설계
│  ├─ 블록 설계
│  ├─ 요인 설계
│  └─ 반응표면 방법론
│
├─ 📉 비모수 통계학 (Nonparametric Statistics)
│  ├─ Mann-Whitney U test
│  ├─ Wilcoxon test
│  ├─ Kruskal-Wallis test
│  └─ Bootstrap 방법
│
├─ 🤖 예측 모델링 (Predictive Modeling)
│  ├─ 머신러닝 (Machine Learning)
│  │  ├─ 지도학습 (Supervised Learning)
│  │  ├─ 비지도학습 (Unsupervised Learning)
│  │  └─ 강화학습 (Reinforcement Learning)
│  │
│  ├─ 데이터 마이닝 (Data Mining)
│  └─ 앙상블 방법 (Ensemble Methods)
│
├─ 🔍 생존분석 (Survival Analysis)
│  ├─ Kaplan-Meier 추정
│  ├─ Cox 비례위험 모델
│  └─ 생존함수
│
└─ 🎯 인과 추론 (Causal Inference) ⭐
   │
   ├─ 실험적 방법 (Experimental Methods)
   │  └─ RCT (Randomized Controlled Trials) 🏆
   │
   └─ 관찰적 방법 (Observational Methods)
      ├─ 매칭 방법 (Matching)
      │  ├─ Propensity Score Matching (PSM)
      │  ├─ Exact Matching
      │  └─ Coarsened Exact Matching (CEM)
      │
      ├─ 차분 방법 (Difference Methods)
      │  ├─ Difference-in-Differences (DiD)
      │  └─ Triple Differences (DDD)
      │
      ├─ 도구변수 (Instrumental Variables, IV)
      │
      ├─ 회귀 불연속 설계 (Regression Discontinuity, RDD)
      │
      ├─ 합성 통제법 (Synthetic Control Method)
      │
      ├─ 인과 그래프 모델 (Causal Graphical Models)
      │  ├─ DAG (Directed Acyclic Graphs)
      │  ├─ Structural Causal Models (SCM)
      │  └─ Mediation Analysis
      │
      ├─ 처치 효과 추정 (Treatment Effect Estimation)
      │  ├─ ATE (Average Treatment Effect)
      │  ├─ ATT (Average Treatment Effect on Treated)
      │  ├─ CATE (Conditional ATE)
      │  └─ Uplift Modeling (ITE) 💡
      │
      ├─ 패널 데이터 방법 (Panel Data Methods)
      │  ├─ Fixed Effects Model
      │  └─ Random Effects Model
      │
      └─ 기타 고급 방법
         ├─ Meta-Analysis
         ├─ Bayesian Causal Inference
         └─ Machine Learning for Causal Inference
            ├─ Causal Forests
            ├─ Double/Debiased ML
            └─ Neural Causal Models
```

**핵심 경로:**
```
통계학 → 추론통계학 (가설검정)
      → 상관/회귀분석 (관계 찾기)
      → 인과 추론 (원인 찾기) ⭐
```

**3가지 핵심 질문:**
```
1. 기술통계학: "데이터가 어떻게 생겼나?"
2. 추론통계학: "모집단에 대해 뭘 말할 수 있나?"
3. 인과 추론:   "A가 B를 일으키는가?" ⭐
```


