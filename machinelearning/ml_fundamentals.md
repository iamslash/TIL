- [References](#references)
- [Chain Rule](#chain-rule)
- [독립 변수 (Independent variable), 종속 변수 (Dependent variable)](#독립-변수-independent-variable-종속-변수-dependent-variable)
- [확률 변수 (Random Variable)](#확률-변수-random-variable)
- [모델 (Model)](#모델-model)
- [모델의 종류](#모델의-종류)
- [Simple Use Case](#simple-use-case)
- [Simple Linear Regression Implementation](#simple-linear-regression-implementation)
  - [최소제곱추정법(Least Squares Estimation, LSE) 구현](#최소제곱추정법least-squares-estimation-lse-구현)
  - [최대우도추정법(MLE) 구현](#최대우도추정법mle-구현)
  - [베이지안 추정법(Bayesian Estimation) 구현](#베이지안-추정법bayesian-estimation-구현)
  - [인공지능 신경망 (Artifical Neural Networks) 구현](#인공지능-신경망-artifical-neural-networks-구현)
  - [랜덤 포레스트 (Random Forest)](#랜덤-포레스트-random-forest)
- [모수 (Parameter)](#모수-parameter)
- [모수를 추정하는 추정법 (Parameter Estimation)](#모수를-추정하는-추정법-parameter-estimation)
- [확률 변수 (Random Variable)](#확률-변수-random-variable-1)
- [확률 분포(probability distribution)](#확률-분포probability-distribution)
- [p(x; μ, σ^2) 최대우도법 (Maximum Likelihood Estimation, MLE)](#px-μ-σ2-최대우도법-maximum-likelihood-estimation-mle)
- [p(y | x; μ, σ^2) 최대우도법 (Maximum Likelihood Estimation, MLE)](#py--x-μ-σ2-최대우도법-maximum-likelihood-estimation-mle)
- [Supervised Learning](#supervised-learning)
- [PyTorch Simple Linear Regression and Likelihood](#pytorch-simple-linear-regression-and-likelihood)
- [PyTorch Simple Linear Regression with Validation Data](#pytorch-simple-linear-regression-with-validation-data)
- [PyTorch Simple Linear Regression with Test Data](#pytorch-simple-linear-regression-with-test-data)
- [Bayesian Statistics MLE (Maximum Likelihood Estimation)](#bayesian-statistics-mle-maximum-likelihood-estimation)

-----

# References

> - [통계 기초 개념과 공식을 3시간만에 끝내드립니다ㅣ 고려대 통계학과 x AI연구원 강의 ㅣ표본, 기대값, 정규분포, 카이제곱, 모평균, t분포, 포아송분포, 조건부 확률 등](https://www.youtube.com/watch?v=YaCQrJCgbqg)
> - [데이터 사이언스 스쿨](https://datascienceschool.net/intro.html)
> - [공돌이의 수학정리노트 (Angelo's Math Notes)](https://angeloyeo.github.io/)
> - ["Probabilistic machine learning": a book series by Kevin Murphy](https://github.com/probml/pml-book)
>   - ["Machine Learning: A Probabilistic Perspective" (2012)](https://probml.github.io/pml-book/book0.html)
>   - ["Probabilistic Machine Learning: An Introduction" (2022)](https://probml.github.io/pml-book/book1.html)
>   - ["Probabilistic Machine Learning: Advanced Topics" (2023)](https://probml.github.io/pml-book/book2.html)
> - [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/syllabus.html)
>   - 화상인식을 위한 CNN
> - [CS224d: Deep Learning for Natural Language Processing](http://web.stanford.edu/class/cs224n/)
>   - 자연어처리를 위한 Deep Learning

# Chain Rule

체인룰(Chain Rule)은 미적분학에서 합성 함수를 미분할 때 사용되는 규칙입니다.
간단히 말해, 두 개 이상의 함수가 합성된 경우에 그 함수의 미분 결과를 계산하기
위해 사용됩니다.

합성 함수란 한 변수의 값을 다양한 함수를 통해 변환하는 과정으로 생성되는 최종
출력값에 대한 함수를 의미합니다. 예를 들어, 두 개의 함수 f(x)와 g(x)가 있고, 두
함수의 합성 h(x) = f(g(x))가 있다면, h(x)는 합성 함수입니다.

체인 룰의 공식은 다음과 같습니다:

```
(dy/dx) = (dy/du) * (du/dx)
```

여기서 `y = f(u)`이고 `u = g(x)`입니다. 이는 `y`가 u에 의존하고, `u`가 `x`에
의존하므로, `x`에 대한 `y`의 변화율`(dy/dx)`을 계산하기 위해 우선 중간변수인
`u`에 대한 `y`의 변화율`(dy/du)`과 `x`에 대한 `u`의 변화율`(du/dx)`을 각각
구하고 곱해주는 것입니다.

예를 들어, 함수 y = f(u) = u²와 u = g(x) = 2x + 3이 있다고 가정합시다. h(x) = f(g(x)) = (2x + 3)²를 구하려면 체인 룰을 적용해야 합니다.

dy/du = 2u
du/dx = 2
Chain rule 적용: (dy/dx) = (dy/du) * (du/dx) = 2u * 2 = 4u = 4(2x+3)
따라서 최종 결과는 h'(x) = 4(2x+3)가 됩니다. 이처럼 체인룰은 한 번에 여러 함수를 미분할 때 유용한 도구입니다.

# 독립 변수 (Independent variable), 종속 변수 (Dependent variable)

독립 변수와 종속 변수는 주로 통계학과 머신러닝에서 사용되는 용어로서 변수 간의
관계와 영향을 설명하는 데 사용됩니다.

**독립 변수(Independent variable)**: 독립 변수는 종속 변수에 영향을 주는
변수로서, 입력으로 주어지는 값입니다. 독립 변수는 다른 변수에 영향을 받지
않거나, 영향을 받는 정도가 상대적으로 낮습니다. 때로는 '특성', '예측 변수',
'회귀 계수' 또는 영어로 'feature', 'predictor', 'explanatory variable' 등과 같은
용어로 불립니다. 예를 들어, 집의 면적과 위치가 집 가격에 영향을 주는 경우 집의
면적과 위치는 독립 변수입니다.

**종속 변수(Dependent variable)**: 종속 변수는 독립 변수에 의해 영향을 받는
변수로서, 결과로 예측하거나 설명하려는 값입니다. 종속 변수는 독립 변수와의
관계를 통해 분석됩니다. 때로는 '반응 변수', '목표 변수', '또는 '결과 변수',
영어로는 'response variable', 'target variable', 'label', 'outcome variable'
등과 같은 용어로 불립니다. 예를 들어, 집의 면적과 위치에 따라 집 가격이 달라지는
경우 집 가격은 종속 변수입니다.

머신러닝 모델 구축 시 독립 변수를 사용하여 종속 변수를 예측하거나 분류하는 것이
목표입니다. 이때 학습 데이터에는 독립 변수와 종속 변수가 레이블로 함께 포함되어
있어야 합니다.

# 확률 변수 (Random Variable)

확률 과정에서 발생하는 불확실한 이벤트를 수치화하는 변수입니다. 일반적인 변수와
다르게 랜덤 변수는 결과 값이 확률론적(무작위적)으로 결정됩니다. 랜덤 변수는 주로
확률론과 통계학에서 사용되며, 확률 분포를 통해 이 변수의 확률적 성질을 설명할 수
있습니다.

랜덤 변수는 일반적으로 대문자 알파벳(예: X, Y, Z)으로 표기하며, 가능한 결과
값(실현 값)은 소문자 알파벳(예: x, y, z)으로 표기합니다.

독립 변수와 종속 변수는 머신러닝 및 회귀 분석과 같은 문맥에서 변수의 관계와
영향을 나타내는 데 사용되는 용어입니다. 반면 랜덤 변수는 확률과 통계학에서
불확실한 이벤트를 수치화하고 설명하는 데 사용되는 개념입니다.

독립 변수와 종속 변수도 어떤 상황에서는 랜덤 변수가 될 수 있습니다. 예를 들어,
어떤 표본 데이터를 수집할 때 데이터 포인트의 선택이 무작위라면, 이 표본 데이터의
독립 변수와 종속 변수는 랜덤 변수로 취급될 수 있습니다.

즉, 랜덤 변수, 독립 변수, 종속 변수는 상황과 문맥에 따라 서로 관련이 있을 수도
있고, 아닐 수도 있습니다. 중요한 것은, 이 용어들이 사용되는 목적과 범위를
이해하는 것입니다.

# 모델 (Model)

통계학에서 모델(model)은 현상이나 데이터를 설명하거나 예측하기 위해 수학적이나
개념적 도구로 표현된 간략화된 표현입니다. 현실 세계의 현상을 단순화하고
일반화하여 이해할 수 있게 만들어 주며, 변수 간의 관계, 패턴, 구조를 파악하거나
미래의 데이터를 예측하는 데 사용됩니다. 모델에는 선형 회귀, 로지스틱 회귀, 확률
모델 등 다양한 종류가 있습니다.

# 모델의 종류

**선형 회귀 모델(Linear Regression)**: 관찰된 데이터 사이의 선형 관계를
설명하며, **예측 변수**와 **종속 변수** 사이의 관계를 선형 방정식으로 나타냅니다.

**로지스틱 회귀 모델(Logistic Regression)**: 범주형 종속 변수와 연속형 독립 변수
사이의 관계를 이진 로지스틱 함수로 설명합니다. 분류 작업에 사용됩니다.

**일반화 선형 모델(Generalized Linear Models, GLM)**: 선형 및 비선형 관계를
설명할 수 있는 확장된 선형 회귀 모델입니다. 종속 변수의 분포가 정규 분포를
따르지 않을 경우에도 적용 가능합니다.

**시계열 모델(Time Series Models)**: 시간 종속적인 데이터를 분석하고 예측하는 데
사용되는 모델로, ARIMA (AutoRegressive Integrated Moving Average), Exponential
Smoothing 등이 있습니다.

**Decision Trees**: 복잡한 데이터 세트에서 결정 경계를 학습하는데 사용되는 트리
기반 모델입니다. 분류 및 회귀 문제에 사용됩니다.

**나이브 베이즈(Naive Bayes)**: 베이즈 정리를 기반으로 하며, 각 특성이
독립적임을 가정하는 확률 모델입니다. 분류 작업에 사용됩니다.

**서포트 벡터 머신(Support Vector Machines, SVM)**: 최적의 결정 경계를 찾기 위해
마진을 최대화하는 알고리즘을 사용하는 모델로, 분류 및 회귀 문제에 사용됩니다.

**신경망(Neural Networks)**: 인간의 뇌 구조를 모방한 인공 뉴런을 기반으로 한
복잡한 모델입니다. 분류, 회귀, 이미지 인식, 자연어 처리 등 다양한 분야에
적용됩니다.

**클러스터링 모델(Clustering Models)**: 비지도 학습 기법으로, 데이터 세트에서 구조,
관계, 패턴을 발견하기 위해 사용되는 모델입니다. K-Means, DBSCAN, Hierarchical
Clustering 등이 있습니다.

**앙상블 모델(Ensemble Models)**: 여러 개의 기본 모델을 조합하여 성능을
향상시키는 모델입니다. **배깅(Bagging)**, **부스팅(Boosting)**, 
**랜덤 포레스트(Random Forest)**, 
**그래디언트 부스팅(Gradient Boosting)** 등이 있습니다.

# Simple Use Case

**키(X)를 독립 변수로하고 몸무게(Y)를 종속 변수로 하는 데이터**를 분석하는 데
가장 적합한 통계적 모델은 **선형 회귀(Linear Regression)** 모델입니다. 키와
몸무게 간의 관계를 선형적인 관계로 가정하고, 몸무게를 예측하는데 사용할 수 있는
모델을 만들 수 있습니다.

선형 회귀모델은 다음과 같은 형태를 가집니다.

```
Y = w * X + b + ε
```

여기서 Y는 몸무게, X는 키, w와 b는 추정할 회귀 계수(기울기와 절편)이며, ε는
오차입니다.

선형 회귀 모델은 다양한 방법으로 추정할 수 있습니다. 가장 일반적인 방법은
**최소제곱추정법(Least Squares Estimation, LSE)**을 사용하여 w와 b를 찾는
것입니다. 이 방법 외에도 **최대우도추정법(MLE)**이나 
**베이지안 추정법(Bayesian Estimation)**을 사용하여 파라미터를 추정할 수 있습니다.

데이터가 선형적인 가정에 잘 부합하면, 선형 회귀 모델은 키와 몸무게 간의 관계에
대한 통찰력있는 예측을 제공할 것입니다. 그러나 데이터가 선형 가정에 맞지 않거나
좀 더 복잡한 관계를 가지고 있다면 비선형인 다항 회귀 모델, 스플라인 회귀, 지수
회귀 등 다른 형태의 회귀 모델을 고려할 수 있습니다. 이러한 고급 모델은 데이터가
가진 다양한 패턴에 더 잘 적합할 수 있습니다. 그러나 선형 회귀 모델은 주어진
데이터에 대한 단순하고 이해하기 쉬운 출발점을 제공하며, 경험적으로 키와 몸무게
사이에는 선형 회귀가 꽤 잘 작동한다는 것이 알려져 있습니다.

# Simple Linear Regression Implementation

## 최소제곱추정법(Least Squares Estimation, LSE) 구현

키(X)와 몸무게(Y) 데이터를 사용하여 독립 변수와 종속 변수간의 선형 회귀 모델을
만드는 Python 코드를 제공합니다. 여기서는 최소제곱추정법 (Least Squares
Estimation, LSE)을 사용하여 선형 회귀 매개변수를 찾습니다.

```py
import numpy as np
import matplotlib.pyplot as plt

# 키와 몸무게 데이터 (예시입니다. 실제 데이터로 대체하세요.)
X = np.array([152, 155, 163, 175, 189])
Y = np.array([45, 49, 60, 68, 77])

# 최소 제곱법을 이용한 선형 회귀 계수 추정
X_mean = np.mean(X)
Y_mean = np.mean(Y)
n = len(X)

numerator = np.sum((X - X_mean) * (Y - Y_mean))
denominator = np.sum((X - X_mean) ** 2)

w = numerator / denominator
b = Y_mean - w * X_mean

print("Optimal w (slope):", w)
print("Optimal b (intercept):", b)

# 그래프로 결과 시각화
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, w * X + b, color='red', label='Fitted Line')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.legend(loc='best')
plt.show()
```

![](img/2023-09-07-23-31-07.png)

위의 코드는 키(X)와 몸무게(Y)를 사용하여 최소제곱추정법으로 선형 회귀 모델의
매개 변수를 찾고 그래프로 결과를 시각화합니다. 이 코드에서 X와 Y 배열에 키와
몸무게 데이터를 제공해야 합니다. 몸무게를 예측하기 위해 구한 선형 회귀 모델에서
파라미터 w와 b를 사용하세요.

## 최대우도추정법(MLE) 구현

선형 회귀에서 최대우도추정법(MLE)을 구현하려면 가우시안 오차를 가정해야 합니다.
이 경우, 최대우도추정법은 최소제곱추정법(LSE)과 동일한 결과를 도출합니다.
그렇지만, MLE를 사용하여 선형 회귀를 구현하는 방법을 살펴 보겠습니다.

```py
import numpy as np
import matplotlib.pyplot as plt

# 키와 몸무게 데이터 (예시입니다. 실제 데이터로 대체하세요.)
X = np.array([152, 155, 163, 175, 189], dtype=float)
Y = np.array([45, 49, 60, 68, 77], dtype=float)

# 데이터 정규화 (수렴 속도 향상을 위해 추가)
X = (X - np.min(X)) / (np.max(X) - np.min(X))
Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))

# 학습률 및 반복 횟수 설정
alpha = 0.01
epochs = 5000

# 최대우도추정법 (MLE)를 사용하여 선형 회귀 매개 변수 찾기
n = len(X)
w = 0
b = 0

for epoch in range(epochs):
    Y_pred = w * X + b
    R = Y - Y_pred
    
    w_gradient = (-2 / n) * np.sum(X * R)
    b_gradient = (-2 / n) * np.sum(R)
    
    w = w - alpha * w_gradient
    b = b - alpha * b_gradient

print("Optimal w (slope):", w)
print("Optimal b (intercept):", b)

# 그래프로 결과 시각화
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, w * X + b, color='red', label='Fitted Line')
plt.xlabel('Normalized Height')
plt.ylabel('Normalized Weight')
plt.legend(loc='best')
plt.show()
```

![](img/2023-09-07-23-31-36.png)

위 코드는 LSE 예제와 거의 동일합니다. 그 이유는 선형 회귀 문제에서 LSE와 MLE가
동일한 결과를 도출하기 때문입니다. 코드에는 데이터 생성, 매개 변수 최적화 및
결과 시각화 외에 특별한 추가 단계가 없습니다.

## 베이지안 추정법(Bayesian Estimation) 구현

```py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge

# 키(X)와 몸무게(Y) 예시 데이터 (실제 데이터로 대체하십시오.)
X = np.array([[152], [155], [163], [175], [189]])
Y = np.array([45, 49, 60, 68, 77])

# BayesianRidge를 이용해 선형 회귀 모델 생성
bayesian_ridge = BayesianRidge()

# 모델 학습
bayesian_ridge.fit(X, Y)

# 베이지안 선형 회귀로 몸무게 예측
y_pred = bayesian_ridge.predict(X)

# 원래 키와 몸무게 데이터를 그려줍니다.
plt.scatter(X, Y, color='blue', label='Actual Data')

# 베이지안 선형 회귀로 예측값을 그려줍니다.
plt.plot(X, y_pred, color='red', label='Fitted Line')

plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.legend(loc='best')
plt.show()
```

![](img/2023-09-07-23-42-18.png)

## 인공지능 신경망 (Artifical Neural Networks) 구현 

다음은 scikit-learn의 `MLPRegressor`를 사용하여 인공 신경망 모델을 학습시키고
키에 대한 몸무게를 예측합니다. 실제 데이터를 사용하려면 `X`와 `y`에 해당하는 값을
실제 데이터로 바꾸면 됩니다.

```py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# 키(X)와 몸무게(Y) 예시 데이터 (실제 데이터로 대체하십시오.)
X = np.array([[152], [155], [163], [175], [189]])
Y = np.array([45, 49, 60, 68, 77])

# MLPRegressor를 사용하여 인공신경망 회귀 모델 생성
mlp_regressor = MLPRegressor(activation='identity', hidden_layer_sizes=(), solver='lbfgs', random_state=0, max_iter=1000)

# 데이터를 학습시키십시오
mlp_regressor.fit(X, Y)

# 회귀 모델에 따라 몸무게를 예측합니다.
y_pred = mlp_regressor.predict(X)

# 원래의 키와 몸무게 데이터를 그려줍니다.
plt.scatter(X, Y, color='blue', label='Actual Data')

# 인공신경망 회귀로 예측한 값을 그려줍니다.
plt.plot(X, y_pred, color='red', label='Predicted Data')

plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.legend(loc='best')
plt.show()
```

![](img/2023-09-07-23-46-26.png)

다음은 PyTorch 를 사용한 구현이다.

```py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

# 키(X)와 몸무게(Y) 데이터
X = np.array([[152], [155], [163], [175], [189]], dtype=np.float32)
Y = np.array([45, 49, 60, 68, 77], dtype=np.float32)

def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def de_normalize_data(data, original_data):
    mean = np.mean(original_data)
    std = np.std(original_data)
    return data * std + mean

X_normalized = normalize_data(X)
Y_normalized = normalize_data(Y)
X_tensor = torch.FloatTensor(X_normalized)
Y_tensor = torch.FloatTensor(Y_normalized).view(-1, 1)

# 선형 회귀 모델 정의
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

model = LinearRegressionModel()

# 손실 함수와 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 회귀 모델 학습
epochs = 10000
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()

    y_pred = model(X_tensor)
    loss = criterion(y_pred, Y_tensor)

    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

# 보다 넓은 범위의 X 값에 대해 예측을 계산합니다.
X_extended = np.expand_dims(np.linspace(np.min(X), np.max(X), 100).astype(np.float32), axis=-1)
y_extended_pred = model(torch.from_numpy(normalize_data(X_extended))).detach().numpy()
y_extended_pred = de_normalize_data(y_extended_pred, Y)

# 원래의 키와 몸무게 데이터와 예측선을 표시합니다.
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X_extended, y_extended_pred, color='red', label='Predicted Data')

plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.legend(loc='best')
plt.show()
```

![](img/2023-09-07-23-59-21.png)

## 랜덤 포레스트 (Random Forest)

Scikit-learn 라이브러리의 `RandomForestRegressor`를 사용하여 랜덤 포레스트
모델을 구현하고 학습시키는 예제입니다

```py
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# 키(X)와 몸무게(y) 데이터를 생성합니다. (예시 데이터)
X = np.array([150, 160, 170, 180, 190]).reshape(-1, 1)
y = np.array([55, 60, 65, 75, 85])

# 랜덤 포레스트 모델 객체를 생성합니다.
random_forest = RandomForestRegressor(n_estimators=100, random_state=0)

# 키(X)와 몸무게(y) 데이터를 사용하여 모델을 학습시킵니다.
random_forest.fit(X, y)

# 새로운 키 값에 대한 몸무게를 예측합니다.
new_X = np.array([167]).reshape(-1, 1)
predicted_y = random_forest.predict(new_X)

print("새로운 키:", new_X)
print("예측된 몸무게:", predicted_y)
```

# 모수 (Parameter)

통계에서 모수(parameter)란 모집단(population)의 특성을 나타내는 수치입니다.
이러한 모수는 모집단의 **평균**, **분산**, **상관계수** 등과 같이 분포의 성질과
특징을 요약하여 표현한 값입니다.

모수는 주어진 모집단의 확률분포를 규정하기 위해 사용되며, 모집단의 크기에 무관한
고정된 값입니다. 통계에서 주요 목적 중 하나는 이 모수를 **추정(Estimation)**하고
**기술통계(Descriptive statistics)**와 **추론통계(Inferential statistics)**를 이용하여
분석하는 것입니다. 이를 통해 모집단의 전반적인 성질을 이해하고 예측 및
의사결정에 도움이 되는 정보를 얻을 수 있습니다.

모수는 표본(sample) 추출을 통해 추정할 수 있습니다. 모집단으로부터 추출된 표본
데이터를 사용하여 모수에 대한 추정치를 계산하고, 이를 다양한 통계적 방법으로
검증하고 분석하는 과정을 거칩니다.

# 모수를 추정하는 추정법 (Parameter Estimation)

모수를 추정하는 추정법은 여러 가지가 있습니다. 대표적인 추정법으로는:

**모멘트추정법 (Method of Moments)**: 관측 데이터의 적률과 이론적인 모델의
적률을 같게 만드는 방식으로 모수를 추정합니다.

**최대우도추정법 (Maximum Likelihood Estimation, MLE)**: 관측된 데이터가 가장
그럴 듯한 모수를 찾아내는 방법으로, 전반적인 데이터의 우도를 최대화하는 값으로
모수를 추정합니다.

**베이즈추정법 (Bayesian Estimation)**: 사전 확률분포를 고려하며, 불확실성을
나타내는 사후 확률분포를 통해 모수를 추정합니다. 베이즈추정법은 사전정보와 관측
데이터를 이용하여 추정을 진행합니다.

**최소제곱추정법 (Least Squares Estimation, LSE)**: 관측값과 예측값의
차이(잔차)의 제곱합을 최소화하는 방식으로 모수를 추정합니다. LSE는 선형회귀와
같은 모델에서 널리 사용됩니다.

**최소마디오추정법 (Minimum Absolute Deviations, MAD)**: 관측값과 예측값의
차이(잔차)의 절대값의 합을 최소화하는 방식으로 모수를 추정합니다. MAD는 이상치에
대해 더 강인한 추정법으로 알려져 있습니다.

**최소자승 추정(LASSO, Ridge Regression 등)**: 정규화(regularization) 항을
추가하여 파라미터의 크기를 축소하는 방식으로 모수를 추정합니다. 이 방법은
과적합(overfitting)을 방지하는 데 도움이 됩니다.

# 확률 변수 (Random Variable)

불확실한 사건(experiment)의 결과에 대응하는 숫자 값으로 정의되는 변수입니다.
사건의 결과가 확률적으로 발생하기 때문에 확률 변수 값도 확률적으로 결정됩니다.
확률 변수는 주로 대문자 알파벳(X, Y, Z 등)으로 표현되고, 그 값은 소문자
알파벳(x, y, z 등)으로 나타냅니다.

확률 변수는 다음 두 가지 유형으로 분류됩니다:

**이산 확률 변수(Discrete random variable)**: 이산 확률 변수는 셀 수 있는(유한 또는
무한) 개의 가능한 값들 중 하나를 가질 수 있습니다. 일반적으로 정수 값을 가지며,
이산 확률 변수의 가능한 값들 각각에 확률을 직접 할당할 수 있습니다. 예를 들어
주사위 던지기, 동전 던지기, 프로세스에서 발생하는 에러 개수 등이 이산 확률
변수의 예입니다.

**연속 확률 변수(Continuous random variable)**: 연속 확률 변수는 값이 하나의 구간
내의 모든 실수 값을 가질 수 있습니다. 연속 확률 변수의 경우, 특정 값의 확률을
정의하기보다는 값의 범위에 확률을 할당하는 것이 더 일반적입니다. 예를 들어 온도,
길이, 시간, 중량 등이 연속 확률 변수의 예입니다.

확률 변수는 실험의 결과를 수치로 표현해 줌으로써 통계적 모델링, 확률 분포의
사용, 데이터 분석 등에 사용됩니다.

# 확률 분포(probability distribution)

확률 변수(random variable)의 가능한 값들과 그 값들이 나타날 가능성을 설명하는
수학적 표현입니다. 확률 분포는 각 사건이 발생할 확률을 알려주는 것으로, 이를
통해 데이터, 실험, 프로세스 등에서 불확실성을 처리할 수 있습니다. 확률 변수의
성격에 따라 확률 분포는 **이산 확률 분포(discrete probability distribution)**와
연속 **확률 분포(continuous probability distribution)**로 구분할 수 있습니다.

**이산 확률 분포(Discrete probability distribution)**: 이산 확률 변수가 가질 수
있는 가능한 값들의 확률을 나타내는 분포입니다. 
**확률질량함수(probability mass function, PMF)**로 나타낼 수 있으며, 
이 함수는 각 값을 확률로 매핑합니다. 이산
확률 분포의 예로는 **베르누이 분포**, **이항 분포**, **포아송 분포** 등이 있습니다.

**연속 확률 분포(Continuous probability distribution)**: 연속 확률 변수가 가질
수 있는 가능한 값들의 확률을 나타내는 분포입니다. 
**확률밀도함수(probability density function, PDF)**로 나타낼 수 있으며, 이 함수는 확률 변수의 값에 대한
확률 밀도를 매핑합니다. 연속 확률 분포에서 특정 값의 확률은 PDF의 해당 영역
아래의 면적을 통해 얻을 수 있습니다. 연속 확률 분포의 예로는 
**정규 분포**, **균일분포**, **지수 분포**, **베타 분포** 등이 있습니다.

# p(x; μ, σ^2) 최대우도법 (Maximum Likelihood Estimation, MLE)

**일반 정규 분포**를 따르는 확률변수 `X`가 있을 때, 확률밀도함수는 다음과 같습니다.

```
p(x; μ, σ^2) = (1 / √(2 * π * σ^2)) * exp(-(x - μ)^2 / (2 * σ^2))
```

![](img/2023-09-07-16-14-08.png)

```
p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
```

이제 관측된 데이터(sample) `x1, x2, ..., xn`이 있다고 가정하겠습니다. MLE 과정은
다음과 같습니다.

1. Likelihood 함수 정의:

데이터(sample) x1, x2, ..., xn에 대한 likelihood 함수는 각 데이터 포인트의 확률밀도함수를 곱한 형태입니다.

```
L(μ, σ^2 | x1, x2, ..., xn) = p(x1; μ, σ^2) * p(x2; μ, σ^2) * ... * p(xn; μ, σ^2)
```

![](img/2023-09-07-16-19-52.png)

```
L(\mu, \sigma^2 | x_1, x_2, \ldots, x_n) = p(x_1; \mu, \sigma^2) * p(x_2; \mu, \sigma^2) * \cdots * p(x_n; \mu, \sigma^2)
```

2. Log-likelihood 함수 정의:

로그 함수는 단조 증가 함수이므로 likelihood를 최대화하는 값과 log-likelihood를 최대화하는 값은 동일합니다. 타원 연산의 안정성과 연산의 단순화를 위해 log-likelihood 함수를 사용합니다.

```
log L(μ, σ^2 | x1, x2, ..., xn) = ∑[log(p(xi; μ, σ^2))] for i = 1 to n
```

3. Log-likelihood 함수 미분 및 최적화:

모수 `μ`와 `σ^2`에 대한 편미분을 통해 log-likelihood 함수를 최적화하고, 그 결과로 최적의 모수를 확인할 수 있습니다. 모수에 대해 미분한 값이 0이 되는 값을 찾습니다.

```
∂(log L) / ∂μ = 0, ∂(log L) / ∂(σ^2) = 0
```

4. 최적화 결과 해석:

위 최적화 과정을 통해 얻어진 최적의 모수 `μ`와 `σ^2`는 관측된 데이터를 가장 잘 설명하는 정규 분포의 모수입니다.

해당 과정을 통해 MLE는 주어진 데이터를 기반으로 일반 정규 분포의 모수인 `μ`와 `σ^2`을 추정합니다. 

다음은 주어진 데이터(표준 정규 분포를 따르는 데이터)에 대해 `μ` 및 `σ^2`에 대한
Likelihood를 계산하고 그 결과를 2D 로 시각화한다.

![](img/2023-09-07-16-28-45.png)

```py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 랜덤 데이터 생성 (표준 정규 분포를 따르는 데이터)
np.random.seed(42)
n = 100
data = np.random.normal(0, 1, n)

# Likelihood 함수 정의
def likelihood(mu, sigma_sq, data):
    return np.prod(norm.pdf(data, mu, np.sqrt(sigma_sq)))

# μ에 대한 likelihood 계산
mu_values = np.linspace(-5, 5, 100)
likelihood_mu = [likelihood(mu, 1, data) for mu in mu_values]

# σ^2에 대한 likelihood 계산
sigma_sq_values = np.linspace(0.1, 5, 100)
likelihood_sigma_sq = [likelihood(0, sigma_sq, data) for sigma_sq in sigma_sq_values]

# 결과를 시각화
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(mu_values, likelihood_mu)
axes[0].set_title("Likelihood with respect to $μ$")
axes[0].set_xlabel("$μ$")
axes[0].set_ylabel("Likelihood")

axes[1].plot(sigma_sq_values, likelihood_sigma_sq)
axes[1].set_title("Likelihood with respect to $σ^2$")
axes[1].set_xlabel("$σ^2$")
axes[1].set_ylabel("Likelihood")

plt.show()
```

다음은 주어진 데이터(표준 정규 분포를 따르는 데이터)에 대해 `μ` 및 `σ^2`에 대한
Likelihood를 계산하고 그 결과를 3D 로 시각화 한다.

![](img/2023-09-07-16-34-44.png)

```py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

# 랜덤 데이터 생성 (표준 정규 분포를 따르는 데이터)
np.random.seed(42)
n = 100
data = np.random.normal(0, 1, n)

# Likelihood 함수 정의
def likelihood(mu, sigma_sq, data):
    return np.prod(norm.pdf(data, mu, np.sqrt(sigma_sq)))

# μ와 σ^2 값의 범위를 설정
mu_values = np.linspace(-3, 3, 100)
sigma_sq_values = np.linspace(0.1, 5, 100)

# Meshgrid 생성
Mu, Sigma_sq = np.meshgrid(mu_values, sigma_sq_values)

# 각 μ와 σ^2 조합에 대한 Likelihood 계산
_likelihood = np.array([likelihood(mu, sigma_sq, data)
                        for mu, sigma_sq in zip(np.ravel(Mu), np.ravel(Sigma_sq))])

# Likelihood 값을 Meshgrid에 맞춤
Likelihood = _likelihood.reshape(Mu.shape)

# 3D로 시각화
fig = plt.figure(figsize=(8, 6))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(Mu, Sigma_sq, Likelihood, cmap='viridis', linewidth=0, antialiased=False)

ax.set_title('Likelihood Function ($μ$ and $σ^2$)')
ax.set_xlabel('$μ$')
ax.set_ylabel('$σ^2$')
ax.set_zlabel('Likelihood')

fig.colorbar(surf, shrink=0.5, aspect=10)

plt.show()
```

# p(y | x; μ, σ^2) 최대우도법 (Maximum Likelihood Estimation, MLE)

최대우도법 (Maximum Likelihood Estimation, MLE)은 주어진 데이터를 가장 잘 설명하는 모수 (μ, σ^2)를 추정하는 방법입니다. 이 과정은 확률밀도함수의 우도(likelihood)를 최대화하는 방향으로 진행되며, 이해를 돕기 위해 다음 단계로 설명하겠습니다.

데이터 수집: 먼저, 관측된 키(x)와 몸무게(y)에 대한 데이터 셋을 수집합니다.

확률밀도함수 정의: 키(x)에 대한 몸무게(y)의 확률밀도함수 p(y | x; μ, σ^2)를 정의합니다. 여기서 μ는 평균, σ^2는 분산을 나타냅니다. 일반적으로 정규분포를 가정하여 확률밀도함수를 정의합니다.

우도함수 정의: 데이터셋의 모든 관측치에 대한 확률밀도함수의 곱으로 우도함수 L(μ, σ^2)를 정의합니다. 여기서 우리는 모수 (μ, σ^2)가 주어진 경우 데이터 셋이 얼마나 그럴듯한지를 나타내는 값을 계산하게 됩니다.

L(μ, σ^2) = Π p(y_i | x_i; μ, σ^2)

로그우도함수: 곱셈의 형태인 우도함수를 덧셈 형태로 바꾸기 위해 우도함수에 로그를 취해 로그우도함수를 정의합니다. 이렇게 함으로써 계산이 용이해집니다.

log L(μ, σ^2) = Σ log p(y_i | x_i; μ, σ^2)

모수 추정: 로그우도함수를 최대화하는 모수 (μ, σ^2)을 찾습니다. 이를 위해 미분과 같은 최적화 기술을 사용하여 로그우도함수에 대한 모수의 최대값을 찾을 수 있습니다.

argmax_(μ, σ^2) log L(μ, σ^2)

결과 해석: 추정된 최적의 모수 (μ, σ^2)를 사용하여 데이터 셋의 몸무게 분포와 관련된 통계적인 해석을 할 수 있습니다.

요약하면, MLE 과정은 관측된 데이터를 가장 잘 설명하는 확률밀도함수의 모수 (μ, σ^2)를 찾기 위해 우도를 최대화하는 방식으로 진행됩니다. 이를 통해 키와 몸무게 간의 관계를 통계적으로 모형화하여 다양한 분석과 예측이 가능하게 됩니다.

# Supervised Learning

키를 입력하면 몸무게를 출력하는 모델을 만들고자 하는 경우, Supervised learning
(지도학습) 알고리즘 중 일부를 사용하여 회귀(regression) 문제를 해결할 수
있습니다. 주어진 데이터를 바탕으로 다음과 같은 모델들을 적용해볼 수 있습니다.

**선형 회귀 (Linear Regression)**: 키와 몸무게 간의 선형 관계를 가정하여
가중치와 편향을 학습합니다. 이 방법은 가장 간단하고 빠르게 구현할 수 있는
모델입니다.

**다항 회귀 (Polynomial Regression)**: 키와 몸무게 간의 비선형 관계를 포착하기
위해 더 복잡한 다항식을 사용합니다. 이 경우 데이터를 더 잘 설명할 수 있으나,
과적합(overfitting)의 위험이 있을 수 있습니다.

**결정 트리 (Decision Tree) 회귀**: 키 값을 기준으로 몸무게를 예측하는
트리구조를 학습합니다. 모델의 해석이 쉽고 작업 흐름을 이해하기 쉬운 장점이
있습니다.

**랜덤 포레스트 (Random Forest) 회귀**: 여러 결정 트리를 앙상블하여 몸무게를
예측하는 상대적으로 높은 성능의 모델입니다.

**서포트 벡터 머신(Support Vector Machine) 회귀**: 키와 몸무게 사이의 관계를
가장 잘 설명하는 초평면을 찾는 알고리즘 입니다. 적절한 커널 함수를 선택하면 선형
및 비선형 관계를 모두 캡처할 수 있습니다.

**인공신경망(Artificial Neural Networks) 회귀**: 신경망 구조를 사용하여 몸무게를
예측하는 딥러닝 기반 방법입니다. 높은 성능을 달성할 수 있지만, 설정 및 학습에
많은 계산 비용이 필요할 수 있습니다.

위의 방법 중에서 가장 적합한 모델을 선택하기 위해서는 주어진 데이터에 대한
이해도가 중요합니다. 데이터의 분포와 특성을 알고 있는 경우, 그에 맞는 적절한
모델과 기법을 선택할 수 있습니다. 또한 모델의 성능과 복잡성을 고려하여 적절한
알고리즘을 선택해야 합니다. 이러한 요인들을 고려하여 여러 모델들을 실험하고
최종적으로 가장 좋은 성능을 보이는 모델을 선택하여 사용하면 됩니다.

# PyTorch Simple Linear Regression and Likelihood

> [가능도함수 | 데이터사이언스스쿨](https://datascienceschool.net/02%20mathematics/09.02%20%EC%B5%9C%EB%8C%80%EA%B0%80%EB%8A%A5%EB%8F%84%20%EC%B6%94%EC%A0%95%EB%B2%95.html#id2)

다음은 PyTorch 로 구현한 Simple Linear Regression 이다.

```py
import torch
import torch.nn as nn
import torch.optim as optim

# Generate toy data
torch.manual_seed(42)
X = torch.rand(100, 1) * 10  # 100개의 데이터 생성
Y = 2 * X + 3 + torch.randn(100, 1)  # True line: y = 2x + 3, adding some noise

# Define the linear regression model
class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super(SimpleLinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleLinearRegression()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    Y_pred = model(X)
    loss = criterion(Y_pred, Y)
    loss.backward()
    optimizer.step()

print("After training:")
print("Weights:", model.linear.weight.item())
print("Bias:", model.linear.bias.item())
```

Frequentist Statistics 관점에서 가능도 함수(likelihood function)는 주어진
데이터와 모델 파라미터가 주어졌을 때, 발생할 확률을 나타내는 함수입니다. 이
코드에서 가능도 함수는 평균 제곱 오차(Mean Squared Error, MSE)를 최소화하는 선형
회귀 모델의 두 파라미터(가중치와 절편, weight와 bias)를 추정하는 것입니다.

이 경우 가능도 함수 L(θ)는 다음과 같이 표현할 수 있습니다.

```
L(θ) = ∏ P(Yi | Xi; θ)
```

여기서 θ는 선형 회귀 모델의 파라미터(가중치와 절편)를 나타내며, `P(Yi | Xi; θ)`는
조건부 확률을 나타내며, 각 데이터 포인트 Yi가 주어진 입력 Xi에 대해서 파라미터
θ를 가진 모델에 의해 생성될 확률입니다.

최대 가능도 추정(Maximum Likelihood Estimation, MLE)을 통해 모델의 파라미터를
찾을 때, 가능도 함수를 최대화하는 파라미터 값을 선택하게 됩니다. 이 경우, 가능도
함수와 관련된 손실 함수(loss function)는 `평균 제곱 오차(MSE)`이며, 이 값을
최소화함으로써 가능도 함수를 최대화하는 파라미터 값을 찾습니다.

# PyTorch Simple Linear Regression with Validation Data

```py
import torch
import torch.nn as nn
import torch.optim as optim

# Generate toy data
torch.manual_seed(42)
X = torch.rand(100, 1) * 10  # 100 data points
Y = 2 * X + 3 + torch.randn(100, 1)  # True line: y = 2x + 3, adding some noise

# Split data into train and validation sets (80% train, 20% validation)
train_ratio = 0.8
train_size = int(train_ratio * len(X))
X_train, X_val = X[:train_size], X[train_size:]
Y_train, Y_val = Y[:train_size], Y[train_size:]

# Define the linear regression model
class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super(SimpleLinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleLinearRegression()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    Y_pred_train = model(X_train)
    loss_train = criterion(Y_pred_train, Y_train)
    loss_train.backward()
    optimizer.step()

    # Compute the validation loss
    with torch.no_grad():
        Y_pred_val = model(X_val)
        loss_val = criterion(Y_pred_val, Y_val)

    # Print the training and validation loss every 100 epochs 
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss_train.item()}, Validation Loss = {loss_val.item()}")

print("After training:")
print("Weights:", model.linear.weight.item())
print("Bias:", model.linear.bias.item())
```

I've added a new variable `train_ratio` to specify the ratio of training data
points and split `X` and `Y` dataset into `train` and `validation` sets. The
training set is used for model training, while the validation¡ set is used for
calculating the validation loss, which can be used to monitor the performance of
the model.

# PyTorch Simple Linear Regression with Test Data

To add test data for evaluating the model's performance in terms of precision
and recall, first, you need to generate some test data points. Since precision
and recall metrics are generally used for classification problems, you'll need
to set a threshold for the predicted continuous values to convert them into
class labels (0 or 1). Then, you can compute the precision and recall scores
based on the true and predicted labels. Here's how to modify the code
accordingly:

```py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score

# Generate toy data
torch.manual_seed(42)
X = torch.rand(100, 1) * 10  # 100 data points
Y = 2 * X + 3 + torch.randn(100, 1)  # True line: y = 2x + 3, adding some noise

# Split data into train, validation, and test sets (70% train, 20% validation, 10% test)
train_ratio = 0.7
val_ratio = 0.2
train_size = int(train_ratio * len(X))
val_size = int(val_ratio * len(X))
X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
Y_train, Y_val, Y_test = Y[:train_size], Y[train_size:train_size+val_size], Y[train_size+val_size:]

# Define the linear regression model
class SimpleLinearRegression(nn.Module):
    def __init__(self):
        super(SimpleLinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleLinearRegression()

# Define threshold value to convert continuous values to class labels
threshold = 15

# Convert target values to class labels (0 or 1) based on the threshold
Y_train_labels = (Y_train > threshold).float()
Y_val_labels = (Y_val > threshold).float()
Y_test_labels = (Y_test > threshold).float()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Train the model
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    Y_pred_train = model(X_train)
    loss_train = criterion(Y_pred_train, Y_train)
    loss_train.backward()
    optimizer.step()

    # Compute the validation loss
    with torch.no_grad():
        Y_pred_val = model(X_val)
        loss_val = criterion(Y_pred_val, Y_val)

    # Print the training and validation loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss_train.item()}, Validation Loss = {loss_val.item()}")

# Evaluate the model on the test data
with torch.no_grad():
    Y_pred_test = model(X_test)
    # Convert predicted continuous values to class labels based on the threshold
    Y_pred_test_labels = (Y_pred_test > threshold).float()

# Calculate precision and recall
precision = precision_score(Y_test_labels, Y_pred_test_labels)
recall = recall_score(Y_test_labels, Y_pred_test_labels)

print("After training:")
print("Weights:", model.linear.weight.item())
print("Bias:", model.linear.bias.item())
print("Precision:", precision)
print("Recall:", recall)
```

In this modified code, I've added test data for evaluating precision and recall
metrics. The target values `Y` are converted to class labels (0 or 1) based on a
defined threshold value. The precision and recall scores are then computed using
the `precision_score` and `recall_score` functions from the `sklearn.metrics`
module, with the true and predicted labels of the test data. The precision and
recall values are printed after the training process. Keep in mind that
precision and recall metrics are not perfect for this problem since it's a
regression problem, not a classification problem.

# Bayesian Statistics MLE (Maximum Likelihood Estimation)

다음과 같은 데이터가 주어졌을 때 bayesian statistics 관점에서 키를 입력하면
몸무게를 출력하는 Simple Linear Regression 의 MLE 는 어떻게 해야 하는가?

```
X: 100, Y: 100
X: 110, Y: 110
X: 120, Y: 120
X: 130, Y: 130
X: 140, Y: 140
```

Bayesian statistics에서 일반적으로 최대 가능도 추정(MLE)를 사용하지 않습니다.
대신 베이지안 추정(Bayesian estimation) 방식을 사용하여 모수를 추정합니다. 이
방법은 사전 확률 분포와 사후 확률 분포를 사용하여 데이터 및 사전 지식에 근거한
추론을 수행합니다.

예를 들어 다음과 같이 주어진 데이터에 대해 베이즈 관점에서 Simple Linear
Regression을 수행하려면 다음 단계를 따르십시오.

데이터:
X: 100, Y: 100
X: 110, Y: 110
X: 120, Y: 120
X: 130, Y: 130
X: 140, Y: 140

데이터 이해: 키(X)와 몸무게(Y)에 대한 5개의 데이터 쌍이 있습니다.

선형 회귀 모델 정의: Simple Linear Regression을 구현하려면 y = β₀ + β₁x + ε 라는
선형 회귀 모델을 가정합니다.

사전 확률 분포 선택: β₀와 β₁에 대한 사전 확률 분포를 정의합니다. 이 배포는
도메인 지식에 기반하여 선택되거나 특정 시나리오(예: 균일하거나 정규 분포)에 대한
가정을 포함할 수 있습니다.

가능도 함수 설정: 샘플 데이터가 주어진 경우 모수에 대한 가능도를 계산합니다.
주어진 회귀 모델을 사용하여 가능도 함수를 설정합니다.

사후 확률 분포 계산: 베이즈 정리를 사용하여 사전 확률 분포와 가능도를 결합하고
사후 확률 분포를 구합니다. 사후 확률 분포는 주어진 데이터에 대한 모수의
노릇값이며, 분포 중간의 점(최대 사후 확률 추청치) 또는 평균을 사용하여 하나의
추론을 얻을 수 있습니다.

주어진 데이터에 대한 베이즈 관점에서의 Simple Linear Regression은 사전 확률 분포
및 실제 데이터를 사용하여 사후 확률 분포에 따라 회귀 계수를 추정합니다. 이
예에서는 특정 사전 확률 분포가 제공되지 않았기 때문에 원하는 사전 확률 분포를
가정한 뒤 사후 확률 분포를 계산해야 합니다. 결과적으로 얻어진 회귀 모델을
사용하여 키(X)에 대한 몸무게(Y) 값을 예측할 수 있습니다.
