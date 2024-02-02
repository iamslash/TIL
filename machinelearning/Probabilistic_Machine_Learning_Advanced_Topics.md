- [Abstract](#abstract)
- [Brief Contents](#brief-contents)
  - [Fundamentals](#fundamentals)
  - [Inference](#inference)
  - [Prediction](#prediction)
  - [Generation](#generation)
  - [Discovery](#discovery)
  - [Action](#action)
- [Fundamentals](#fundamentals-1)
- [Inference](#inference-1)
- [Prediction](#prediction-1)
- [Generation](#generation-1)
- [Discovery](#discovery-1)
- [Decision making](#decision-making)

-----

# Abstract

Probabilistic Machine Learning: Advanced Topics 를 읽고 요약한다.

# Brief Contents

1 Introduction
소개

## Fundamentals
기초

2 Probability
확률: 확률론의 기초 개념을 다룸

3 Statistics
통계: 통계 기법을 소개

4 Graphical models
그래프 모델: 변수 사이의 확률관계를 보여주는 그림 표현

5 Information theory
정보이론: 정보를 효과적으로 인코딩하고 전송하는 방법 연구

6 Optimization
최적화: 주어진 목표하에 최적의 해를 찾는 연구

## Inference
추론

7 Inference algorithms: an overview
추론 알고리즘 개요: 여러 추론 알고리즘을 소개

8 Gaussian filtering and smoothing
가우시안 필터링 및 스무딩: 가우시안 기반 필터와 데이터를 스무딩하는 방법 론

9 Message passing algorithms
메시지 전달 알고리즘: 그래프 모델에서 정보를 전달하는 메시지 기반 알고리즘

10 Variational inference
변분 추론: 추론 문제를 최적화 문제로 변환하는 접근 방식

11 Monte Carlo methods
몬테 카를로 방법: 시뮬레이션을 사용하여 복잡한 수학적 문제 해결

12 Markov chain Monte Carlo
마르코프 체인 몬테 카를로: 마르코프 체인을 사용하는 몬테카를로 방법

13 Sequential Monte Carlo
순차 몬테 카를로: 순차 데이터에서 몬테 카를로 방법 적용

## Prediction
예측

14 Predictive models: an overview
예측 모델 개요: 다양한 예측 모델들을 소개

15 Generalized linear models
일반화 선형 모델: 선형 모델을 확장한 일반화된 모델

16 Deep neural networks
깊은 신경망: 복잡한 데이터에서 높은 성능을 내는 차세대 머신러닝 모델

17 Bayesian neural networks
베이지안 신경망: 불확실성을 고려한 신경망 모델

18 Gaussian processes
가우시안 프로세스: 데이터를 모델링하는 비모수적 방법

19 Beyond the iid assumption
iid 가정을 넘어서: 독립적 동일분포 가정이 아닌 데이터를 다룸

iid는 "independent and identically distributed"의 약자로, 독립적이고 동일한 분포를 가진 샘플들을 의미합니다. 쉽게 설명하자면, 학습(training) 및 테스트(test) 데이터에서 각각의 데이터가 서로 영향을 주지 않고 독립적으로 존재하며, 동일한 기준에 따라 추출된 것을 가정하는 것입니다. 이러한 iid 가정을 따르면, 데이터 분석에서 알고리즘들이 정확하고 일반화되어 정확한 예측을 할 확률이 높아집니다. 하지만 실제 문제에서는 테스트 데이터가 학습 데이터와 완벽하게 동일한 분포를 가지는 경우가 드물기 때문에, 글에서는 이러한 분포의 변화를 어떻게 다룰지에 대한 여러 상황을 설명하고 있습니다.

지도 학습 기계학습(ML)의 표준 접근법은 학습과 테스트 데이터 모두 동일한 분포에서 독립적이고 동일한 분포(iid)를 따르는 샘플을 포함한다고 가정합니다. 그러나 테스트 데이터의 분포가 학습 데이터의 분포와 다른 경우가 많이 있으며, 이를 분포 변화(distribution shift)라고 합니다(19.2 절에서 논의). 경우에 따라 학습과 테스트 외에도 여러 관련 분포의 데이터를 가질 수 있으며(19.6 절에서 논의), 데이터 분포가 지속적으로 변경되거나 조각별 상수 방식으로 데이터를 스트리밍하는 설정에서도 마주칠 수 있습니다(19.7 절에서 논의). 마지막으로, 예측 시스템의 성능을 최소화하기 위해 테스트 분포를 적대적으로 선택하는 설정에 대해서도 논의합니다(19.8 절).

iid가정을 잘 적용하려면 다음과 같은 방법들을 고려할 수 있습니다:

- 데이터 전처리: 원시 데이터에서 불필요한 노이즈를 제거하고 iid 가정에 가까운 데이터를 만드는 전처리 단계를 수행합니다.
- 교차 검증: 데이터를 여러 개의 하위 집합으로 나누고, 각각의 집합을 훈련 및 테스트로 사용하여 모델이 iid 가정에 얼마나 잘 맞는지 평가합니다.
- 이상치 제거: 특수한 경우나 예외적인 데이터 포인트를 확인하고, 이를 제거하거나 처리하여 훈련과 테스트 데이터셋 간의 일관성을 유지합니다.
- 여러 모델 비교: iid 가정에 대해 각기 다른 가정을 가지고 있는 여러 모델을 비교하여, 가장 적합한 모델을 선택합니다.

이러한 방법들을 통해 iid 가정에 더 가까운 데이터를 학습하고 테스트하는 것이 가능해집니다. 그렇게 함으로써 분포의 차이로 인한 오류를 줄이고 예측 성능을 향상시킬 수 있습니다.

## Generation
생성

20 Generative models: an overview
생성 모델 개요: 데이터 생성 과정을 모델링하는 방법들 소개

21 Variational autoencoders
변분 오토인코더: 생성 모델 중 하나로 높은 차원의 데이터를 압축

22 Autoregressive models
자기회귀 모델: 이전 데이터를 활용해 데이터 생성

23 Normalizing flows
정규화 플로우: 복잡한 확률 분포를 학습하는 생성 모델

24 Energy-based models
에너지 기반 모델: 에너지 함수를 사용한 생성 모델

25 Diffusion models
확산 모델: 데이터 생성 과정에서 노이즈를 추가하는 모델

26 Generative adversarial networks
생성적 적대 신경망: 경쟁적 학습을 통해 데이터 생성하는 모델

## Discovery
탐색

27 Discovery methods: an overview
탐색 방법 개요: 데이터에서 숨겨진 구조 및 패턴을 찾는 방법들 소개

28 Latent factor models
잠재 요인 모델: 데이터를 더 낮은 차원으로 압축하여 구조 발견

29 State-space models
상태-공간 모델: 순차 데이터에서 상태와 관측 사이의 관계를 모델링

30 Graph learning
그래프 학습: 그래프 구조를 학습하는 방법론

31 Nonparametric Bayesian models
비모수 베이지안 모델: 데이터 수가 늘어나면서 모델 복잡성이 자동 조정

32 Representation learning
표현 학습: 데이터에서 유용한 정보를 자동으로 추출하는 방법

33 Interpretability
해석 가능성: 머신러닝 모델의 예측을 명확하게 이해하고 설명하는 방법

## Action
행동

34 Decision making under uncertainty
불확실성 하에서의 의사결정: 불확실한 상황에서 최적의 결정을 내리기 위한 방법

35 Reinforcement learning
강화학습: 시행착오를 통해 학습하는 알고리즘

36 Causality
인과성: 인과 관계를 추론하고 이해하는 방법론

# Fundamentals

2. Probability

2.1 Introduction: 확률론의 기본 개념과 중요성을 소개합니다.

2.2 Some common univariate distributions: 단변량 분포들에 대해 설명합니다.
2.2.1 Some common discrete distributions: 일반적인 이산 확률 분포들을 설명합니다.
2.2.2 Some common continuous distributions: 일반적인 연속 확률 분포들에 대해 설명합니다.
2.2.3 Pareto distribution: 파레토 분포에 대해 설명합니다.

2.3 The multivariate Gaussian (normal) distribution: 다변량 가우시안(정규) 분포에 대해 다룹니다.
2.3.1 Definition: 다변량 가우시안 분포의 정의를 제시합니다.
2.3.2 Moment form and canonical form: 모멘트 형식과 표준형에 대해 설명합니다.
2.3.3 Marginals and conditionals of a MVN: 다변량 정규 분포의 주변분포와 조건부 분포에 대해 논합니다.
2.3.4 Bayes’ rule for Gaussians: 가우시안 분포에 대한 베이즈 규칙을 설명합니다.
2.3.5 Example: sensor fusion with known measurement noise: 알려진 측정 잡음을 가진 센서 융합의 예를 들어 설명합니다.
2.3.6 Handling missing data: 결측 데이터를 다루는 방법을 설명합니다.
2.3.7 A calculus for linear Gaussian models: 선형 가우시안 모델에 대한 계산법을 소개합니다.

2.4 Some other multivariate continuous distributions: 다른 일부 다변량 연속 분포들을 다룹니다.
2.4.1 Multivariate Student distribution: 다변량 스튜던트 분포에 대해 설명합니다.
2.4.2 Circular normal (von Mises Fisher) distribution: 원형 정규(폰 미세스 피셔) 분포에 대해 설명합니다.
2.4.3 Matrix-variate Gaussian (MVG) distribution: 행렬-변량 가우시안(MVG) 분포를 설명합니다.
2.4.4 Wishart distribution: 위샤트 분포에 대해 다룹니다.
2.4.5 Dirichlet distribution: 디리클레 분포에 대해 설명합니다.

2.5 The exponential family: 지수족에 대해 설명합니다.
2.5.1 Definition: 지수족의 정의를 제시합니다.
2.5.2 Examples: 지수족의 예시들을 들어 설명합니다.
2.5.3 Log partition function is cumulant generating function: 로그 분할 함수가 누적 생성 함수임을 설명합니다.
2.5.4 Canonical (natural) vs mean (moment) parameters: 정준(자연) 매개변수와 평균(모멘트) 매개변수의 차이를 논합니다.
목차에 나와 있는 내용을 기반으로 한 요약이며, 각 항목의 구체적인 내용은 책의 해당 장을 참조해야 합니다.
2.5.5 MLE for the exponential family: 지수족 분포에 대한 최대우도추정(Maximum Likelihood Estimation, MLE) 방법을 설명합니다.
2.5.6 Exponential dispersion family: 지수 분산족에 대해 설명하며, 지수족 분포를 일반화한 개념을 소개합니다.
2.5.7 Maximum entropy derivation of the exponential family: 최대 엔트로피 원리를 사용하여 지수족 분포를 도출하는 방법을 설명합니다.

2.6 Fisher information matrix (FIM): 피셔 정보 행렬에 대해 다룹니다.
2.6.1 Definition: 피셔 정보 행렬의 정의를 제시합니다.
2.6.2 Equivalence between the FIM and the Hessian of the NLL: 피셔 정보 행렬과 음의 로그 가능도(Negative Log Likelihood, NLL)의 헤시안 간의 동치성을 설명합니다.
2.6.3 Examples: 피셔 정보 행렬의 예시를 들어 설명합니다.
2.6.4 Approximating KL divergence using FIM: 피셔 정보 행렬을 사용하여 KL 다이버전스를 근사하는 방법을 소개합니다.
2.6.5 Fisher information matrix for exponential family: 지수족 분포에 대한 피셔 정보 행렬을 다룹니다.

2.7 Transformations of random variables: 확률변수의 변환에 대해 설명합니다.
2.7.1 Invertible transformations (bijections): 가역 변환(일대일 대응)에 대해 설명합니다.
2.7.2 Monte Carlo approximation: 몬테 카를로 근사 방법을 다룹니다.
2.7.3 Probability integral transform: 확률 적분 변환에 대해 설명합니다.

2.8 Markov chains: 마르코프 체인에 대해 소개합니다.
2.8.1 Parameterization: 마르코프 체인의 매개변수화에 대해 설명합니다.
2.8.2 Application: Language modeling: 언어 모델링에 마르코프 체인을 어떻게 적용하는지 설명합니다.
2.8.3 Parameter estimation: 마르코프 체인의 매개변수 추정 방법을 다룹니다.
2.8.4 Stationary distribution of a Markov chain: 마르코프 체인의 정상 분포에 대해 설명합니다.

2.9 Divergence measures between probability distributions: 확률 분포 간의 발산 척도에 대해 다룹니다.
2.9.1 f-divergence: f-발산에 대해 설명합니다.
2.9.2 Integral probability metrics: 적분 확률 메트릭에 대해 다룹니다.
2.9.3 Maximum mean discrepancy (MMD): 최대 평균 불일치에 대해 설명합니다.
2.9.4 Total variation distance: 전체 변동 거리에 대해 설명합니다.
2.9.5 Comparing distributions using binary classifiers: 이진 분류기를 사용하여 분포를 비교하는 방법을 다룹니다.

3. Statistics (통계): 통계적 방법론과 이론에 대한 전반적인 소개.

3.1 Introduction (서론): 통계학의 기본적인 개념과 베이지안 접근법의 개요 설명.
3.1.1 Frequentist statistics (빈도주의 통계): 장기적으로 반복되는 사건들의 빈도에 기초한 통계적 접근법.
3.1.2 Bayesian statistics (베이지안 통계): 사전 지식과 증거를 결합하여 확률을 업데이트하는 베이지안 논리.
3.1.3 Arguments for the Bayesian approach (베이지안 접근법을 위한 주장): 베이지안 통계의 장점과 사용 이유에 대한 논리적 근거.
3.1.4 Arguments against the Bayesian approach (베이지안 접근법에 대한 반론): 베이지안 접근법의 한계점과 비판에 대한 설명.
3.1.5 Why not just use MAP estimation? (왜 단지 MAP 추정만 사용하지 않는가?): 최대 사후 확률(Maximum a posteriori, MAP) 추정의 한계와 대안적인 방법들.

3.2 Closed-form analysis using conjugate priors (공액 사전 확률을 사용한 닫힌 형태의 분석): 사전 확률과 사후 확률이 같은 분포군에 속하는 경우의 분석 방법.
3.2.1 The binomial model (이항 모델): 두 가지 결과를 가지는 실험의 확률을 모델링하는 방법.
3.2.2 The multinomial model (다항 모델): 세 개 이상의 결과를 가지는 실험의 확률을 모델링하는 방법.
3.2.3 The univariate Gaussian model (단변량 가우시안 모델): 하나의 연속형 무작위 변수에 대한 정규 분포 모델.
3.2.4 The multivariate Gaussian model (다변량 가우시안 모델): 여러 연속형 무작위 변수에 대한 정규 분포 모델.
3.2.5 Conjugate-exponential models (공액 지수 모델): 지수적 가정을 가지는 확률 분포에 대한 모델.

3.3 Beyond conjugate priors (공액 사전 확률을 넘어서): 공액 사전 확률만으로는 충분하지 않은 경우의 대안적인 접근법.
3.3.1 Robust (heavy-tailed) priors (강건한 (무거운 꼬리를 가진) 사전 확률): 극단적인 값에 더 강건한 사전 확률 모델.
3.3.2 Priors for variance parameters (분산 파라미터를 위한 사전 확률): 분산과 같은 파라미터에 대한 사전 확률 설정 방법.

3.4 Noninformative priors (비정보 사전 확률): 데이터에 의해 주로 정보가 결정되도록 하는 약한 사전 확률.
3.4.1 Maximum entropy priors (최대 엔트로피 사전 확률): 가능한 모든 상태에 대해 무지의 원칙을 적용하는 방법.
3.4.2 Jeffreys priors (제프리 사전 확률): 불변성을 기반으로 한 사전 확률.
3.4.3 Invariant priors (불변 사전 확률): 변환에 대해 불변인 사전 확률.

3.5 Hierarchical priors (계층적 사전 확률): 변수들 간의 계층 구조를 모델링하는 사전 확률.
3.5.1 A hierarchical binomial model (계층적 이항 모델): 계층 구조를 가진 이항 분포 모델.
3.5.2 A hierarchical Gaussian model (계층적 가우시안 모델): 다중 수준의 가우시안 분포를 가진 계층적 모델.

3.6 Empirical Bayes (경험적 베이즈): 관측 데이터를 사용하여 사전 확률을 추정하는 방법.
3.6.1 A hierarchical binomial model (계층적 이항 모델): 데이터 기반의 계층 구조를 가진 이항 모델.
3.6.2 A hierarchical Gaussian model (계층적 가우시안 모델): 데이터를 통해 추정된 계층적 가우시안 분포.
3.6.3 Hierarchical Bayes for n-gram smoothing (n-gram 평활화를 위한 계층적 베이즈): 언어 모델링에서 n-gram 데이터의 평활화를 위한 계층적 베이즈 접근법.

3.7 Model selection and evaluation (모델 선택과 평가): 다양한 통계 모델을 선택하고 평가하는 기준과 방법.
3.7.1 Bayesian model selection (베이지안 모델 선택): 베이지안 원칙에 기반한 모델 선택 방법.
3.7.2 Estimating the marginal likelihood (주변 가능도 추정): 모델 적합도를 평가하는 주변 가능도의 계산.
3.7.3 Connection between cross validation and marginal likelihood (교차 검증과 주변 가능도 간의 연결): 교차 검증을 통한 모델의 성능 평가와 주변 가능도와의 관계.
3.7.4 Pareto-Smoothed Importance Sampling LOO estimate (파레토-평활 중요도 샘

# Inference

# Prediction

# Generation


# Discovery

# Decision making
