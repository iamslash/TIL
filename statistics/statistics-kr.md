# Abstract

통계학(Statistics)은 **데이터를 수집, 정리, 분석하여 의미 있는 정보를 추출**하고, **불확실성 속에서 합리적인 의사결정**을 내리기 위한 학문입니다.

**통계의 핵심 개념:**
- **변수와 변량**: 변하는 값과 그 구체적인 관측값
- **평균, 분산, 표준편차**: 데이터의 중심과 퍼진 정도
- **정규분포**: 자연현상의 보편적 패턴
- **상관관계 vs 인과관계**: 관계가 있다 ≠ 원인이다
- **가설검정과 p-값**: 과학적 결론을 내리는 방법

# Materials

- [Khan Academy: Statistics and Probability](https://www.khanacademy.org/math/statistics-probability)
- [Statistics - Wikipedia](https://en.wikipedia.org/wiki/Statistics)
- [StatQuest: Statistics Fundamentals](https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9)
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

## 9. p-값 이해하기: 가장 흔한 오해와 올바른 논리

### 오해 1: 귀무가설은 "원래 가설"이다?

```
❌ 잘못된 이해:
귀무가설 = 우리가 원래 믿던 가설

✓ 올바른 이해:
귀무가설 = "효과가 없다"는 보수적 가정 (기각하려는 것)
대립가설 = "효과가 있다"는 우리가 증명하고 싶은 것
```

### 오해 2: p-값 = 귀무가설이 틀렸을 확률?

```
❌ 잘못된 해석:
p = 0.03 → "귀무가설이 틀렸을 확률이 3%"

✓ 올바른 해석:
p = 0.03 → "귀무가설이 참이라고 가정했을 때,
           이런 결과가 우연히 나올 확률이 3%"

핵심: p-값은 P(데이터 | H₀가 참) ≠ P(H₀가 참 | 데이터)
```

**동전 던지기 비유:**
```
상황: 동전을 10번 던졌더니 앞면이 9번

H₀: "동전은 공정하다"
p-값: "공정한 동전에서 10번 중 9번 이상 앞면 = 약 1%"

해석:
→ 공정한 동전에서 이런 극단적 결과가 나올 확률이 1%
→ 너무 희귀하므로 "동전이 공정하다"는 가정을 믿기 어려움
→ H₀ 기각, "동전이 불공정하다"고 결론

❌ 잘못: "동전이 공정할 확률이 1%"
✓ 올바름: "공정하다면 이런 결과는 1% 확률로만 나타남"
```

### p-값에서 결론까지: 논리적 연결

많은 사람들이 가장 혼란스러워하는 부분: **"p-값이 낮다는 것이 어떻게 '효과가 있다'는 결론으로 이어지는가?"**

#### 5단계 논리적 흐름

**상황: 비료 실험에서 p-값 = 0.003**

**Step 1: 가정**
```
H₀: "비료는 효과가 없다" ← 이것을 참이라고 일단 가정
```

**Step 2: 실험**
```
- 대조군: 525kg
- 실험군: 617kg
- 차이: 92kg (큰 차이!)
```

**Step 3: 질문**
```
"만약 비료가 정말 효과가 없다면,
이렇게 큰 차이가 우연히 나올 확률은?"
```

**Step 4: 답 (p-값)**
```
p = 0.003 = 0.3%

즉, 비료가 효과 없다면:
- 1000번 실험 중 3번만 큰 차이
- 997번은 작은 차이만
```

**Step 5: 논리적 추론**
```
전제: "효과 없다면, 큰 차이는 0.3% 확률만"
관찰: "그런데 실제로 큰 차이를 관찰"
결론: "전제('효과 없음')가 이상하다!"
     → H₀를 믿기 어렵다
     → H₀ 기각
     → "비료는 효과가 있을 수 있다" ✓
```

#### 귀류법(Proof by Contradiction)으로 이해

```
귀류법의 구조:
1. 가정: A가 참이다
2. 추론: A가 참이면 B가 일어나야 함
3. 관찰: 그런데 B가 일어나지 않음
4. 결론: 모순! → A는 거짓이다

가설 검정에 적용:
1. 가정: H₀가 참 ("비료는 효과 없다")
2. 추론: H₀가 참이면 작은 차이만 (큰 차이는 0.3%만)
3. 관찰: 그런데 큰 차이 나타남
4. 결론: 모순! → H₀는 거짓 → H₀ 기각
```

#### 두 가지 시나리오로 직관적 이해

**시나리오 A: H₀가 참인 세계 (비료 효과 없음)**
```
1000번 실험 반복 시:
- 997번: 작은 차이 (5~15kg)
- 3번: 큰 차이 (90kg 이상) ← 우리가 관찰한 것
```

**시나리오 B: H₀가 거짓인 세계 (비료 효과 있음)**
```
1000번 실험 반복 시:
- 950번: 큰 차이 (90kg 이상) ← 우리가 관찰한 것
- 50번: 중간 차이
```

**우리의 관찰: 큰 차이 (92kg)**
```
시나리오 A에서: 큰 차이는 0.3% (매우 희귀)
시나리오 B에서: 큰 차이는 95% (매우 흔함)

→ 시나리오 B가 훨씬 더 설득력 있음!
→ H₀ 기각, "비료는 효과가 있다"
```

#### 핵심 논리 체인

```
p-값이 낮다 (0.003)
    ↓
H₀ 하에서 이런 데이터는 매우 희귀 (0.3%)
    ↓
우연치고는 너무 희귀
    ↓
H₀가 이상하다
    ↓
H₀ 기각
    ↓
H₁ 채택 ("효과가 있다")
```

**중요:** 이것은 **확률적 추론**입니다:
- ✓ "H₀가 이상하다" (매우 높은 확률로)
- ✓ "H₁이 더 설득력 있다"
- ✗ "H₁이 100% 확실하다" (아님!)

### 핵심 요약

**귀무가설:**
- ❌ 우리가 믿는 가설
- ✓ 기각하기 위해 설정한 보수적 가정
- ✓ "효과 없음"에서 시작

**p-값:**
- ❌ 귀무가설이 틀렸을 확률
- ✓ 귀무가설이 참이라면, 이런 데이터가 우연히 나올 확률
- ✓ P(데이터 | H₀) ≠ P(H₀ | 데이터)

**의사결정:**
- p < 0.05: 너무 희귀 → H₀ 기각 → H₁ 채택
- p ≥ 0.05: 우연 가능 → H₀ 유지

**과학적 보수주의:**
```
α = 0.05 기준은 Type I Error (위양성)를 5% 이하로 제한
"잘못된 발견"을 최소화하기 위한 안전장치
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


