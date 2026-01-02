# Abstract

Decision Tree(의사 결정 나무)는 머신 러닝에서 가장 직관적이고 이해하기 쉬운 알고리즘 중 하나입니다. 우리가 어릴 때 하던 "20 Questions" 게임처럼, 일련의 질문을 통해 답을 찾아가는 과정이 바로 Decision Tree의 핵심 원리입니다.

**Decision Tree의 장점:**
- 직관적이고 해석하기 쉬움
- 데이터 전처리가 거의 필요 없음 (정규화, 스케일링 불필요)
- 수치형/범주형 데이터 모두 처리 가능
- Feature Importance를 통해 중요한 변수 파악 가능

**Decision Tree의 단점:**
- 과적합(Overfitting) 경향이 강함
- 데이터의 작은 변화에 민감함 (불안정성)
- 복잡한 관계를 표현하기 어려움

이러한 단점을 보완하기 위해 **Random Forest**가 개발되었으며, 여러 개의 Decision Tree를 결합하여 더 안정적이고 정확한 예측을 가능하게 합니다.

# Materials

- [Scikit-learn Decision Tree Documentation](https://scikit-learn.org/stable/modules/tree.html)
- [Scikit-learn Random Forest Documentation](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk)
- [[인공지능을 위한 머신러닝 101] 의사결정나무와 랜덤포레스트를 알아보자 | youtube](https://www.youtube.com/watch?v=vutU-SLTZ-A)

# Basic

## Decision Tree란?

Decision Tree는 데이터를 분류하거나 값을 예측하기 위해 **일련의 질문을 통해 결정을 내리는 트리 구조**의 알고리즘입니다.

### 20 Questions 게임으로 이해하기

게임 상황을 상상해봅시다:

```
질문자: "제가 어떤 동물을 떠올렸어요. 맞춰보세요!"

플레이어: "그 동물은 날 수 있나요?"
질문자: "아니요."

플레이어: "그 동물은 네 발이 있나요?"
질문자: "네."

플레이어: "그 동물은 크나요?"
질문자: "아니요."

플레이어: "그 동물은 집에서 기르나요?"
질문자: "네."

플레이어: "강아지!"
질문자: "정답입니다!"
```

이 게임에서 우리는 각 질문을 통해 가능한 답의 범위를 좁혀갔습니다. 이것이 바로 **Decision Tree의 작동 원리**입니다.

## Decision Tree 구조

Decision Tree는 다음과 같은 구성 요소로 이루어져 있습니다:

```
                    [날 수 있나요?]  ← Root Node (뿌리 노드)
                    /              \
                  Yes              No
                  /                  \
          [새인가요?]          [네 발이 있나요?]  ← Internal Node (중간 노드)
          /      \              /              \
        Yes      No           Yes              No
        /          \          /                  \
    [참새]      [박쥐]   [크나요?]            [뱀]  ← Leaf Node (잎 노드)
                        /      \
                      Yes      No
                      /          \
                  [코끼리]    [강아지]
```

**구성 요소:**
- **Root Node (뿌리 노드)**: 첫 번째 질문, 전체 데이터를 처음으로 나누는 노드
- **Internal Node (중간 노드)**: 중간 단계의 질문들, 데이터를 더 세분화
- **Leaf Node (잎 노드)**: 최종 결정 또는 예측 값
- **Branch (가지)**: 노드들을 연결하는 선, 각 질문의 답(Yes/No)에 해당

## Gini Impurity (지니 계수)

Decision Tree는 어떻게 "가장 좋은 질문"을 선택할까요? 바로 **Gini Impurity (지니 계수)**를 사용합니다.

### Gini Impurity란?

Gini Impurity는 **데이터가 얼마나 혼잡하게 섞여 있는지**를 나타내는 척도입니다.

**공식:**
```
Gini = 1 - Σ(p_i)²
```

여기서:
- `p_i`: 각 클래스가 나타날 확률
- `Σ`: 모든 클래스에 대한 합

**예시:**

1. **완전히 순수한 경우 (Gini = 0)**
   ```
   상자 안에 파란 공만 10개
   Gini = 1 - (1.0)² = 0
   → 혼잡도가 0, 완벽하게 예측 가능
   ```

2. **완전히 혼잡한 경우 (Gini = 0.5)**
   ```
   상자 안에 파란 공 5개, 빨간 공 5개
   Gini = 1 - (0.5² + 0.5²) = 1 - 0.5 = 0.5
   → 혼잡도가 최대, 예측이 어려움
   ```

### 왜 제곱을 사용할까?

제곱의 의미를 이해해봅시다:

**상자에서 공을 두 번 뽑는다고 가정:**
- **두 번 다 같은 색**이 나올 확률이 높다 → 상자 속 공의 색깔이 통일되어 있다
- **서로 다른 색**이 나올 확률이 높다 → 상자 속 공의 색깔이 섞여 있다

```python
# 예시: 파란 공 80%, 빨간 공 20%
같은 색이 나올 확률 = (0.8 × 0.8) + (0.2 × 0.2) = 0.68
Gini = 1 - 0.68 = 0.32  (혼잡도가 낮음)

# 예시: 파란 공 50%, 빨간 공 50%
같은 색이 나올 확률 = (0.5 × 0.5) + (0.5 × 0.5) = 0.5
Gini = 1 - 0.5 = 0.5  (혼잡도가 높음)
```

**Decision Tree의 목표:** 각 분기에서 Gini Impurity를 최소화하는 질문을 선택하는 것!

## 실제 예제: 데이트 의사 결정

여자친구가 데이트 신청에 응할지 예측하는 Decision Tree를 만들어봅시다.

### 데이터셋

| 날씨 | 습도 | 바람 | 데이트 응답 |
|------|------|------|-------------|
| 맑음 | 높음 | 약함 | No |
| 맑음 | 높음 | 강함 | No |
| 흐림 | 높음 | 약함 | Yes |
| 비 | 보통 | 약함 | Yes |
| 비 | 낮음 | 약함 | Yes |
| 비 | 낮음 | 강함 | No |
| 흐림 | 낮음 | 강함 | Yes |
| 맑음 | 보통 | 약함 | No |

**목표:** 날씨, 습도, 바람을 보고 데이트 응답(Yes/No)을 예측

### Step 1: 전체 데이터의 Gini Impurity 계산

```python
# 전체 데이터: Yes 4개, No 4개
p_yes = 4/8 = 0.5
p_no = 4/8 = 0.5

Gini_total = 1 - (0.5² + 0.5²)
           = 1 - (0.25 + 0.25)
           = 0.5
```

### Step 2: 각 특성(Feature)별로 분기했을 때 Gini Impurity 계산

#### 날씨로 분기한 경우

**맑음 (3개):** Yes 0개, No 3개
```python
Gini_맑음 = 1 - (0² + 1²) = 0
```

**흐림 (2개):** Yes 2개, No 0개
```python
Gini_흐림 = 1 - (1² + 0²) = 0
```

**비 (3개):** Yes 2개, No 1개
```python
Gini_비 = 1 - ((2/3)² + (1/3)²)
        = 1 - (0.444 + 0.111)
        = 0.445
```

**날씨의 평균 Gini:**
```python
Gini_날씨 = (3/8 × 0) + (2/8 × 0) + (3/8 × 0.445)
          = 0.167
```

#### 습도로 분기한 경우

**높음 (3개):** Yes 1개, No 2개
```python
Gini_높음 = 1 - ((1/3)² + (2/3)²) = 0.444
```

**보통 (2개):** Yes 1개, No 1개
```python
Gini_보통 = 1 - (0.5² + 0.5²) = 0.5
```

**낮음 (3개):** Yes 2개, No 1개
```python
Gini_낮음 = 1 - ((2/3)² + (1/3)²) = 0.445
```

**습도의 평균 Gini:**
```python
Gini_습도 = (3/8 × 0.444) + (2/8 × 0.5) + (3/8 × 0.445)
          = 0.459
```

#### 바람으로 분기한 경우

**약함 (5개):** Yes 3개, No 2개
```python
Gini_약함 = 1 - ((3/5)² + (2/5)²) = 0.48
```

**강함 (3개):** Yes 1개, No 2개
```python
Gini_강함 = 1 - ((1/3)² + (2/3)²) = 0.444
```

**바람의 평균 Gini:**
```python
Gini_바람 = (5/8 × 0.48) + (3/8 × 0.444)
          = 0.467
```

### Step 3: 가장 낮은 Gini Impurity를 가진 특성 선택

```
Gini_날씨 = 0.167  ← 가장 낮음! (최적의 첫 번째 분기)
Gini_습도 = 0.459
Gini_바람 = 0.467
```

**결론:** 첫 번째 질문은 "날씨"로 결정!

### Step 4: 트리 구축

```
                    [날씨는?]
                 /      |      \
            맑음      흐림      비
            /         |         \
         No(3)     Yes(2)   [다음 분기 필요]
                             (Yes 2, No 1)
```

흐림일 때는 모두 Yes이므로 **Leaf Node** 완성!
맑음일 때는 모두 No이므로 **Leaf Node** 완성!
비일 때는 아직 혼잡하므로 다음 분기가 필요합니다.

### Step 5: "비" 경우에 대해 다음 분기 찾기

비 날씨 데이터 3개만 보면:

| 습도 | 바람 | 응답 |
|------|------|------|
| 보통 | 약함 | Yes |
| 낮음 | 약함 | Yes |
| 낮음 | 강함 | No |

습도와 바람 중 어느 것으로 나눌까요? 계산 결과, 습도로 나누면 완벽히 분류됩니다!

**최종 Decision Tree:**

```
                    [날씨는?]
                 /      |      \
            맑음      흐림      비
            /         |         \
         No(3)     Yes(2)    [습도는?]
                             /        \
                        높음/보통    낮음
                          /            \
                      Yes(2)       [바람은?]
                                   /      \
                                약함      강함
                                /          \
                            Yes(1)       No(1)
```

## Python 코드 예제

### 1. 기본 Decision Tree 분류

```python
"""
Decision Tree 기본 예제: Iris 꽃 분류
- 꽃받침(sepal)과 꽃잎(petal)의 길이/너비로 붓꽃 종류 분류
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# 1. 데이터 로드
iris = load_iris()
X = iris.data  # 특성: sepal length, sepal width, petal length, petal width
y = iris.target  # 타겟: 0(setosa), 1(versicolor), 2(virginica)

print("특성 이름:", iris.feature_names)
print("클래스 이름:", iris.target_names)
print("데이터 크기:", X.shape)

# 2. 학습/테스트 데이터 분리 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Decision Tree 모델 생성 및 학습
model = DecisionTreeClassifier(
    criterion='gini',      # 'gini' 또는 'entropy' (Information Gain)
    max_depth=3,           # 트리의 최대 깊이 (과적합 방지)
    min_samples_split=2,   # 노드를 분할하기 위한 최소 샘플 수
    min_samples_leaf=1,    # 리프 노드의 최소 샘플 수
    random_state=42
)

model.fit(X_train, y_train)

# 4. 예측 및 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n정확도: {accuracy:.4f}")
print("\n분류 리포트:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 5. Feature Importance (특성 중요도)
feature_importance = model.feature_importances_
for name, importance in zip(iris.feature_names, feature_importance):
    print(f"{name}: {importance:.4f}")

# 6. 트리 시각화
plt.figure(figsize=(15, 10))
tree.plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree for Iris Classification")
plt.savefig('decision_tree_iris.png', dpi=300, bbox_inches='tight')
print("\n트리 시각화가 'decision_tree_iris.png'로 저장되었습니다.")

# 7. 새로운 데이터 예측
new_sample = [[5.1, 3.5, 1.4, 0.2]]  # 새로운 붓꽃 측정값
prediction = model.predict(new_sample)
probability = model.predict_proba(new_sample)

print(f"\n새로운 샘플 예측: {iris.target_names[prediction[0]]}")
print("각 클래스별 확률:")
for class_name, prob in zip(iris.target_names, probability[0]):
    print(f"  {class_name}: {prob:.4f}")

"""
출력 예시:
특성 이름: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
클래스 이름: ['setosa' 'versicolor' 'virginica']
데이터 크기: (150, 4)

정확도: 1.0000

분류 리포트:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30

sepal length (cm): 0.0000
sepal width (cm): 0.0000
petal length (cm): 0.4333
petal width (cm): 0.5667

새로운 샘플 예측: setosa
각 클래스별 확률:
  setosa: 1.0000
  versicolor: 0.0000
  virginica: 0.0000
"""
```

### 2. 데이트 예제 구현

```python
"""
데이트 의사 결정 예제: 실제 Decision Tree 구현
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt

# 1. 데이터 생성
data = {
    '날씨': ['맑음', '맑음', '흐림', '비', '비', '비', '흐림', '맑음'],
    '습도': ['높음', '높음', '높음', '보통', '낮음', '낮음', '낮음', '보통'],
    '바람': ['약함', '강함', '약함', '약함', '약함', '강함', '강함', '약함'],
    '데이트': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)
print("원본 데이터:")
print(df)
print()

# 2. 범주형 데이터를 숫자로 변환 (Label Encoding)
le_weather = LabelEncoder()
le_humidity = LabelEncoder()
le_wind = LabelEncoder()
le_date = LabelEncoder()

df['날씨_encoded'] = le_weather.fit_transform(df['날씨'])
df['습도_encoded'] = le_humidity.fit_transform(df['습도'])
df['바람_encoded'] = le_wind.fit_transform(df['바람'])
df['데이트_encoded'] = le_date.fit_transform(df['데이트'])

print("인코딩된 데이터:")
print(df)
print()

# 3. 특성(X)과 타겟(y) 분리
X = df[['날씨_encoded', '습도_encoded', '바람_encoded']]
y = df['데이트_encoded']

# 4. Decision Tree 모델 생성 및 학습
model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X, y)

# 5. Gini Impurity 및 Feature Importance 출력
print("Feature Importance (특성 중요도):")
feature_names = ['날씨', '습도', '바람']
for name, importance in zip(feature_names, model.feature_importances_):
    print(f"  {name}: {importance:.4f}")
print()

# 6. 트리 시각화
plt.figure(figsize=(15, 10))
tree.plot_tree(
    model,
    feature_names=feature_names,
    class_names=['No', 'Yes'],
    filled=True,
    rounded=True,
    fontsize=12
)
plt.title("데이트 의사 결정 트리")
plt.savefig('decision_tree_date.png', dpi=300, bbox_inches='tight')
print("트리 시각화가 'decision_tree_date.png'로 저장되었습니다.\n")

# 7. 새로운 상황에 대한 예측
print("새로운 상황 예측:")

test_cases = [
    {'날씨': '맑음', '습도': '낮음', '바람': '약함'},
    {'날씨': '흐림', '습도': '높음', '바람': '강함'},
    {'날씨': '비', '습도': '보통', '바람': '약함'},
]

for i, case in enumerate(test_cases, 1):
    # 인코딩
    weather_encoded = le_weather.transform([case['날씨']])[0]
    humidity_encoded = le_humidity.transform([case['습도']])[0]
    wind_encoded = le_wind.transform([case['바람']])[0]

    # 예측
    X_new = [[weather_encoded, humidity_encoded, wind_encoded]]
    prediction = model.predict(X_new)
    probability = model.predict_proba(X_new)

    result = le_date.inverse_transform(prediction)[0]

    print(f"\n케이스 {i}: {case}")
    print(f"  예측 결과: {result}")
    print(f"  확률: No={probability[0][0]:.2f}, Yes={probability[0][1]:.2f}")

# 8. 트리 구조를 텍스트로 출력
print("\n\n트리 구조 (텍스트):")
text_representation = tree.export_text(model, feature_names=feature_names)
print(text_representation)

"""
출력 예시:
원본 데이터:
   날씨  습도  바람 데이트
0  맑음  높음  약함   No
1  맑음  높음  강함   No
2  흐림  높음  약함  Yes
3   비  보통  약함  Yes
4   비  낮음  약함  Yes
5   비  낮음  강함   No
6  흐림  낮음  강함  Yes
7  맑음  보통  약함   No

Feature Importance (특성 중요도):
  날씨: 0.6667
  습도: 0.2222
  바람: 0.1111

새로운 상황 예측:

케이스 1: {'날씨': '맑음', '습도': '낮음', '바람': '약함'}
  예측 결과: No
  확률: No=1.00, Yes=0.00

케이스 2: {'날씨': '흐림', '습도': '높음', '바람': '강함'}
  예측 결과: Yes
  확률: No=0.00, Yes=1.00

케이스 3: {'날씨': '비', '습도': '보통', '바람': '약함'}
  예측 결과: Yes
  확률: No=0.00, Yes=1.00
"""
```

### 3. Decision Tree Regressor (회귀)

```python
"""
Decision Tree 회귀 예제: 집값 예측
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 로드
housing = fetch_california_housing()
X = housing.data
y = housing.target  # 집값 (단위: $100,000)

print("특성 이름:", housing.feature_names)
print("데이터 크기:", X.shape)

# 2. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Decision Tree Regressor 학습
model = DecisionTreeRegressor(
    max_depth=5,           # 과적합 방지
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

model.fit(X_train, y_train)

# 4. 예측 및 평가
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nMSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# 5. Feature Importance
print("\nFeature Importance:")
feature_importance = sorted(
    zip(housing.feature_names, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
)
for name, importance in feature_importance:
    print(f"  {name}: {importance:.4f}")

# 6. 예측 vs 실제 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('실제 집값 ($100k)')
plt.ylabel('예측 집값 ($100k)')
plt.title('Decision Tree Regressor: 실제 vs 예측')
plt.savefig('decision_tree_regression.png', dpi=300, bbox_inches='tight')
print("\n예측 결과 시각화가 'decision_tree_regression.png'로 저장되었습니다.")

"""
출력 예시:
특성 이름: ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
데이터 크기: (20640, 8)

MSE: 0.5123
RMSE: 0.7158
R² Score: 0.6234

Feature Importance:
  MedInc: 0.5891
  Latitude: 0.1234
  Longitude: 0.1123
  HouseAge: 0.0789
  AveRooms: 0.0456
  AveOccup: 0.0234
  Population: 0.0189
  AveBedrms: 0.0084
"""
```

### 4. Decision Tree의 과적합 문제 시각화

```python
"""
Decision Tree의 과적합 문제 시연
- max_depth를 변화시키면서 학습/테스트 성능 비교
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 데이터 로드
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# 다양한 max_depth 값 테스트
depths = range(1, 20)
train_scores = []
test_scores = []

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_scores.append(accuracy_score(y_train, train_pred))
    test_scores.append(accuracy_score(y_test, test_pred))

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, 'o-', label='Training Accuracy', linewidth=2)
plt.plot(depths, test_scores, 's-', label='Test Accuracy', linewidth=2)
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree: Training vs Test Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('overfitting_example.png', dpi=300, bbox_inches='tight')

print("과적합 시각화가 'overfitting_example.png'로 저장되었습니다.")
print(f"\n최적의 max_depth: {depths[test_scores.index(max(test_scores))]}")
print(f"최고 테스트 정확도: {max(test_scores):.4f}")

"""
출력 예시:
과적합 시각화가 'overfitting_example.png'로 저장되었습니다.

최적의 max_depth: 3
최고 테스트 정확도: 0.9778

관찰:
- max_depth가 증가하면 학습 정확도는 계속 상승 (100%까지)
- 하지만 테스트 정확도는 특정 지점 이후 하락 또는 정체
- 이것이 바로 과적합(Overfitting) 현상!
"""
```

# Advanced: Random Forest

## Decision Tree의 단점

Decision Tree는 강력하지만 **치명적인 약점**이 있습니다:

### 1. 데이터 변화에 민감함 (불안정성)

작은 데이터 변화가 트리 구조를 크게 바꿀 수 있습니다.

**예시:** 데이트 데이터에 한 개의 샘플만 추가

```
새로운 데이터: 날씨=흐림, 습도=높음, 바람=강함, 응답=No
```

이 데이터 하나만 추가되면:
- 기존: "흐림 → 무조건 Yes" (완벽한 리프 노드)
- 변경 후: "흐림"이 더 이상 완벽한 분기가 아니게 됨
- **트리 구조가 완전히 재구성됨!**

### 2. 과적합(Overfitting) 경향

```python
# 제한이 없는 Decision Tree
model = DecisionTreeClassifier()  # max_depth=None (무제한)
model.fit(X_train, y_train)

# 결과:
# Training Accuracy: 100%  ← 완벽!
# Test Accuracy: 75%       ← 나쁨...
```

**원인:** 트리가 너무 깊어져서 학습 데이터의 **노이즈**까지 학습

### 3. 복잡한 경계선 표현의 어려움

Decision Tree는 **축에 수직인 직선**으로만 데이터를 나눕니다.

```
Decision Tree의 분기:
- X > 5 (수직선)
- Y > 3 (수평선)

표현하기 어려운 경우:
- 대각선 경계
- 원형 경계
- 복잡한 곡선
```

## Random Forest란?

**Random Forest = Decision Tree 여러 개를 합친 것!**

핵심 아이디어: "여러 명의 의견을 종합하면 더 정확하다"

### Random Forest의 원리

```
원본 데이터
    ↓
[부트스트래핑 + 피처 배깅]
    ↓
┌─────────┬─────────┬─────────┬─────────┐
│ Tree 1  │ Tree 2  │ Tree 3  │  ...    │
│ (Yes)   │ (Yes)   │ (No)    │ (Yes)   │
└─────────┴─────────┴─────────┴─────────┘
    ↓
[다수결 투표]
    ↓
최종 예측: Yes (3표 vs 1표)
```

**장점:**
1. **안정성**: 개별 트리의 불안정성이 평균화됨
2. **과적합 감소**: 여러 트리의 평균이므로 노이즈에 덜 민감
3. **높은 정확도**: 단일 트리보다 일반적으로 더 정확

**단점:**
1. **해석성 저하**: 수백 개의 트리를 한번에 이해하기 어려움
2. **계산 비용**: 여러 트리를 학습하므로 시간이 오래 걸림
3. **메모리 사용**: 많은 트리를 저장해야 함

## Bootstrapping과 Feature Bagging

Random Forest가 "다양한" 트리를 만드는 두 가지 핵심 기법:

### 1. Bootstrapping (부트스트래핑)

**정의:** 원본 데이터에서 **중복을 허용**하여 무작위로 샘플링

```python
원본 데이터 (8개):
[A, B, C, D, E, F, G, H]

Tree 1을 위한 샘플 (8개, 중복 허용):
[A, A, C, D, E, E, G, H]  ← A와 E가 중복, B와 F가 빠짐

Tree 2를 위한 샘플 (8개, 중복 허용):
[A, B, B, D, F, F, F, H]  ← B와 F가 중복, C, E, G가 빠짐

Tree 3을 위한 샘플 (8개, 중복 허용):
[B, C, C, D, E, F, G, G]  ← C와 G가 중복, A와 H가 빠짐
```

**효과:** 각 트리가 서로 다른 데이터로 학습 → 다양한 패턴 학습

### 2. Feature Bagging (피처 배깅)

**정의:** 각 노드에서 **일부 특성만** 무작위로 선택하여 분기

```python
전체 특성 (4개):
[날씨, 습도, 바람, 온도]

Tree 1의 Root Node: [날씨, 습도] 중에서만 선택
  → 날씨로 분기

Tree 2의 Root Node: [습도, 온도] 중에서만 선택
  → 습도로 분기

Tree 3의 Root Node: [바람, 온도] 중에서만 선택
  → 바람으로 분기
```

**효과:**
- 강력한 특성이 모든 트리를 지배하는 것을 방지
- 약한 특성도 활용 기회를 얻음
- 트리 간 다양성 증가

### Bootstrapping vs Feature Bagging 비교

| 구분 | Bootstrapping | Feature Bagging |
|------|---------------|-----------------|
| 대상 | 데이터 샘플 (행) | 특성 (열) |
| 시점 | 트리 생성 시 한 번 | 각 노드마다 매번 |
| 중복 | 허용 | 허용 안 함 (부분 선택) |
| 목적 | 데이터 다양성 | 특성 다양성 |

## Python 코드 예제

### 1. Random Forest 기본 사용

```python
"""
Random Forest 기본 예제: Iris 분류
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 로드
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# 2. Random Forest 모델 생성 및 학습
rf_model = RandomForestClassifier(
    n_estimators=100,        # 트리 개수 (많을수록 안정적, 하지만 느림)
    max_depth=3,             # 각 트리의 최대 깊이
    min_samples_split=2,     # 노드 분할 최소 샘플 수
    min_samples_leaf=1,      # 리프 노드 최소 샘플 수
    max_features='sqrt',     # 각 분기에서 고려할 특성 개수
                             # 'sqrt': sqrt(n_features)
                             # 'log2': log2(n_features)
                             # None: 모든 특성
    bootstrap=True,          # 부트스트래핑 사용 여부
    random_state=42,
    n_jobs=-1                # 병렬 처리 (모든 CPU 사용)
)

rf_model.fit(X_train, y_train)

# 3. 예측 및 평가
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Random Forest 정확도: {accuracy:.4f}")
print("\n분류 리포트:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 4. Feature Importance
feature_importance = rf_model.feature_importances_
print("\nFeature Importance:")
for name, importance in sorted(
    zip(iris.feature_names, feature_importance),
    key=lambda x: x[1],
    reverse=True
):
    print(f"  {name}: {importance:.4f}")

# 5. Feature Importance 시각화
plt.figure(figsize=(10, 6))
indices = np.argsort(feature_importance)[::-1]
plt.bar(range(len(feature_importance)), feature_importance[indices])
plt.xticks(range(len(feature_importance)),
           [iris.feature_names[i] for i in indices],
           rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('random_forest_importance.png', dpi=300, bbox_inches='tight')
print("\nFeature Importance 시각화가 저장되었습니다.")

"""
출력 예시:
Random Forest 정확도: 1.0000

분류 리포트:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        19
  versicolor       1.00      1.00      1.00        13
   virginica       1.00      1.00      1.00        13

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45

Feature Importance:
  petal width (cm): 0.4523
  petal length (cm): 0.4201
  sepal length (cm): 0.0891
  sepal width (cm): 0.0385
"""
```

### 2. Decision Tree vs Random Forest 비교

```python
"""
Decision Tree vs Random Forest 성능 비교
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# 1. Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

# Cross-validation으로 안정성 측정
dt_cv_scores = cross_val_score(dt_model, iris.data, iris.target, cv=5)

# 2. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Cross-validation으로 안정성 측정
rf_cv_scores = cross_val_score(rf_model, iris.data, iris.target, cv=5)

# 3. 결과 비교
print("=" * 60)
print("Decision Tree vs Random Forest 비교")
print("=" * 60)

print(f"\nDecision Tree:")
print(f"  테스트 정확도: {dt_accuracy:.4f}")
print(f"  CV 평균 정확도: {dt_cv_scores.mean():.4f}")
print(f"  CV 표준편차: {dt_cv_scores.std():.4f} ← 불안정성 지표")

print(f"\nRandom Forest:")
print(f"  테스트 정확도: {rf_accuracy:.4f}")
print(f"  CV 평균 정확도: {rf_cv_scores.mean():.4f}")
print(f"  CV 표준편차: {rf_cv_scores.std():.4f} ← 더 안정적!")

print(f"\n개선:")
print(f"  정확도 향상: {(rf_accuracy - dt_accuracy) * 100:.2f}%p")
print(f"  안정성 향상: {(dt_cv_scores.std() - rf_cv_scores.std()) * 100:.2f}%p")

# 4. Cross-validation 결과 시각화
plt.figure(figsize=(10, 6))
plt.boxplot([dt_cv_scores, rf_cv_scores], labels=['Decision Tree', 'Random Forest'])
plt.ylabel('Accuracy')
plt.title('Decision Tree vs Random Forest: Cross-Validation Scores')
plt.grid(True, alpha=0.3)
plt.savefig('dt_vs_rf_comparison.png', dpi=300, bbox_inches='tight')
print("\n비교 시각화가 'dt_vs_rf_comparison.png'로 저장되었습니다.")

"""
출력 예시:
============================================================
Decision Tree vs Random Forest 비교
============================================================

Decision Tree:
  테스트 정확도: 0.9778
  CV 평균 정확도: 0.9467
  CV 표준편차: 0.0326 ← 불안정성 지표

Random Forest:
  테스트 정확도: 1.0000
  CV 평균 정확도: 0.9600
  CV 표준편차: 0.0163 ← 더 안정적!

개선:
  정확도 향상: 2.22%p
  안정성 향상: 1.63%p
"""
```

### 3. 데이트 예제에 Random Forest 적용

```python
"""
데이트 예제: Random Forest로 개선
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import numpy as np

# 1. 데이터 생성 (원본 + 추가 데이터)
data = {
    '날씨': ['맑음', '맑음', '흐림', '비', '비', '비', '흐림', '맑음',
            '흐림', '비', '맑음', '맑음', '흐림', '비', '맑음', '비'],
    '습도': ['높음', '높음', '높음', '보통', '낮음', '낮음', '낮음', '보통',
            '보통', '높음', '낮음', '높음', '낮음', '보통', '보통', '낮음'],
    '바람': ['약함', '강함', '약함', '약함', '약함', '강함', '강함', '약함',
            '약함', '강함', '강함', '약함', '약함', '약함', '강함', '강함'],
    '데이트': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No',
              'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# 2. 인코딩
le_weather = LabelEncoder()
le_humidity = LabelEncoder()
le_wind = LabelEncoder()
le_date = LabelEncoder()

X = pd.DataFrame({
    '날씨': le_weather.fit_transform(df['날씨']),
    '습도': le_humidity.fit_transform(df['습도']),
    '바람': le_wind.fit_transform(df['바람'])
})
y = le_date.fit_transform(df['데이트'])

print("데이터 크기:", X.shape)
print()

# 3. Decision Tree vs Random Forest
dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross-validation (데이터가 작으므로)
dt_scores = cross_val_score(dt_model, X, y, cv=5)
rf_scores = cross_val_score(rf_model, X, y, cv=5)

print("Decision Tree:")
print(f"  CV 평균 정확도: {dt_scores.mean():.4f} ± {dt_scores.std():.4f}")
print(f"  CV 점수들: {dt_scores}")

print("\nRandom Forest:")
print(f"  CV 평균 정확도: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
print(f"  CV 점수들: {rf_scores}")

# 4. 전체 데이터로 학습 후 예측
dt_model.fit(X, y)
rf_model.fit(X, y)

# 새로운 케이스 예측
test_cases = [
    {'날씨': '맑음', '습도': '낮음', '바람': '약함'},
    {'날씨': '흐림', '습도': '높음', '바람': '강함'},
    {'날씨': '비', '습도': '보통', '바람': '약함'},
]

print("\n" + "=" * 60)
print("새로운 상황 예측 비교")
print("=" * 60)

for i, case in enumerate(test_cases, 1):
    # 인코딩
    weather_enc = le_weather.transform([case['날씨']])[0]
    humidity_enc = le_humidity.transform([case['습도']])[0]
    wind_enc = le_wind.transform([case['바람']])[0]

    X_new = [[weather_enc, humidity_enc, wind_enc]]

    # Decision Tree 예측
    dt_pred = dt_model.predict(X_new)
    dt_prob = dt_model.predict_proba(X_new)[0]
    dt_result = le_date.inverse_transform(dt_pred)[0]

    # Random Forest 예측
    rf_pred = rf_model.predict(X_new)
    rf_prob = rf_model.predict_proba(X_new)[0]
    rf_result = le_date.inverse_transform(rf_pred)[0]

    print(f"\n케이스 {i}: {case}")
    print(f"  Decision Tree: {dt_result} (확률: {dt_prob[1]:.2f})")
    print(f"  Random Forest: {rf_result} (확률: {rf_prob[1]:.2f})")

# 5. Feature Importance 비교
print("\n" + "=" * 60)
print("Feature Importance 비교")
print("=" * 60)

print("\nDecision Tree:")
for name, importance in zip(['날씨', '습도', '바람'], dt_model.feature_importances_):
    print(f"  {name}: {importance:.4f}")

print("\nRandom Forest:")
for name, importance in zip(['날씨', '습도', '바람'], rf_model.feature_importances_):
    print(f"  {name}: {importance:.4f}")

"""
출력 예시:
데이터 크기: (16, 3)

Decision Tree:
  CV 평균 정확도: 0.6875 ± 0.1563
  CV 점수들: [0.66666667 0.66666667 1.         0.33333333 0.66666667]

Random Forest:
  CV 평균 정확도: 0.7500 ± 0.1250
  CV 점수들: [0.66666667 1.         0.66666667 0.66666667 0.75      ]

============================================================
새로운 상황 예측 비교
============================================================

케이스 1: {'날씨': '맑음', '습도': '낮음', '바람': '약함'}
  Decision Tree: No (확률: 0.00)
  Random Forest: No (확률: 0.12)

케이스 2: {'날씨': '흐림', '습도': '높음', '바람': '강함'}
  Decision Tree: Yes (확률: 1.00)
  Random Forest: Yes (확률: 0.88)

케이스 3: {'날씨': '비', '습도': '보통', '바람': '약함'}
  Decision Tree: Yes (확률: 1.00)
  Random Forest: Yes (확률: 0.76)

============================================================
Feature Importance 비교
============================================================

Decision Tree:
  날씨: 0.4667
  습도: 0.3333
  바람: 0.2000

Random Forest:
  날씨: 0.3845
  습도: 0.3123
  바람: 0.3032

관찰:
- Random Forest가 더 안정적인 예측 (확률이 극단적이지 않음)
- Random Forest가 모든 특성을 더 균형있게 활용
- CV 점수의 표준편차가 Random Forest에서 더 낮음 (더 안정적)
"""
```

### 4. Random Forest 하이퍼파라미터 튜닝

```python
"""
Random Forest 하이퍼파라미터 최적화
"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로드 (유방암 데이터)
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.3, random_state=42
)

print("데이터 크기:", cancer.data.shape)
print("클래스:", cancer.target_names)

# 2. 기본 Random Forest
print("\n" + "=" * 60)
print("기본 Random Forest")
print("=" * 60)

rf_basic = RandomForestClassifier(random_state=42)
rf_basic.fit(X_train, y_train)
basic_score = rf_basic.score(X_test, y_test)
print(f"테스트 정확도: {basic_score:.4f}")

# 3. Grid Search로 최적 파라미터 찾기
print("\n" + "=" * 60)
print("Grid Search 진행 중...")
print("=" * 60)

param_grid = {
    'n_estimators': [50, 100, 200],           # 트리 개수
    'max_depth': [None, 10, 20, 30],          # 최대 깊이
    'min_samples_split': [2, 5, 10],          # 분할 최소 샘플
    'min_samples_leaf': [1, 2, 4],            # 리프 최소 샘플
    'max_features': ['sqrt', 'log2', None],   # 특성 개수
}

# Grid Search 실행 (시간이 걸릴 수 있음)
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,                    # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,              # 병렬 처리
    verbose=1
)

grid_search.fit(X_train, y_train)

# 4. 최적 파라미터 및 성능
print("\n최적 파라미터:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\n최적 모델 CV 점수: {grid_search.best_score_:.4f}")

# 5. 최적 모델로 테스트
rf_best = grid_search.best_estimator_
best_score = rf_best.score(X_test, y_test)
y_pred = rf_best.predict(X_test)

print(f"최적 모델 테스트 정확도: {best_score:.4f}")
print(f"개선: {(best_score - basic_score) * 100:.2f}%p")

# 6. 상세 분류 리포트
print("\n분류 리포트:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# 7. Confusion Matrix 시각화
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names)
plt.ylabel('실제')
plt.xlabel('예측')
plt.title('Random Forest Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nConfusion Matrix가 'confusion_matrix.png'로 저장되었습니다.")

# 8. Feature Importance Top 10
feature_importance = rf_best.feature_importances_
indices = np.argsort(feature_importance)[::-1][:10]

plt.figure(figsize=(12, 6))
plt.bar(range(10), feature_importance[indices])
plt.xticks(range(10), [cancer.feature_names[i] for i in indices],
           rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Top 10 Feature Importance')
plt.tight_layout()
plt.savefig('top10_features.png', dpi=300, bbox_inches='tight')
print("Top 10 Feature Importance가 'top10_features.png'로 저장되었습니다.")

"""
출력 예시:
데이터 크기: (569, 30)
클래스: ['malignant' 'benign']

============================================================
기본 Random Forest
============================================================
테스트 정확도: 0.9649

============================================================
Grid Search 진행 중...
============================================================
Fitting 5 folds for each of 324 candidates, totalling 1620 fits

최적 파라미터:
  n_estimators: 200
  max_depth: 20
  min_samples_split: 2
  min_samples_leaf: 1
  max_features: sqrt

최적 모델 CV 점수: 0.9724
최적 모델 테스트 정확도: 0.9766
개선: 1.17%p

분류 리포트:
              precision    recall  f1-score   support

   malignant       0.98      0.95      0.97        63
      benign       0.97      0.99      0.98       108

    accuracy                           0.98       171
   macro avg       0.98      0.97      0.97       171
weighted avg       0.98      0.98      0.98       171
"""
```

### 5. Out-of-Bag (OOB) Score

```python
"""
Out-of-Bag (OOB) Score 활용
- 부트스트래핑으로 선택되지 않은 샘플로 자동 검증
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# OOB Score 활성화
rf_oob = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,  # OOB Score 계산 활성화
    random_state=42
)

rf_oob.fit(X, y)

print("Out-of-Bag Score:", rf_oob.oob_score_)
print("\n설명:")
print("- 각 트리는 부트스트랩 샘플로 학습")
print("- 선택되지 않은 샘플(약 37%)로 자동 검증")
print("- 별도의 테스트 세트 없이도 성능 추정 가능!")

# 트리 개수에 따른 OOB Score 변화
n_estimators_range = range(10, 201, 10)
oob_scores = []

for n in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42)
    rf.fit(X, y)
    oob_scores.append(rf.oob_score_)

plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, oob_scores, 'o-', linewidth=2)
plt.xlabel('Number of Trees')
plt.ylabel('OOB Score')
plt.title('OOB Score vs Number of Trees')
plt.grid(True, alpha=0.3)
plt.savefig('oob_score_curve.png', dpi=300, bbox_inches='tight')
print("\nOOB Score 곡선이 'oob_score_curve.png'로 저장되었습니다.")

"""
출력 예시:
Out-of-Bag Score: 0.9533333333333334

설명:
- 각 트리는 부트스트랩 샘플로 학습
- 선택되지 않은 샘플(약 37%)로 자동 검증
- 별도의 테스트 세트 없이도 성능 추정 가능!

관찰:
- 트리가 50~100개 정도면 OOB Score가 안정화
- 그 이후로는 큰 개선이 없음
- 트리 개수와 성능/계산 시간의 트레이드오프 고려 필요
"""
```

## Random Forest 사용 가이드라인

### 언제 Decision Tree를 사용할까?

✅ **Decision Tree 추천:**
- 결과를 **해석하고 설명**해야 할 때
- 데이터가 **작고 단순**할 때
- **빠른 예측**이 필요할 때
- **메모리가 제한**적일 때

### 언제 Random Forest를 사용할까?

✅ **Random Forest 추천:**
- **높은 정확도**가 최우선일 때
- 과적합이 우려될 때
- 데이터가 **크고 복잡**할 때
- Feature Importance가 필요할 때
- 약간의 성능 향상을 위해 **계산 시간을 투자**할 수 있을 때

### 하이퍼파라미터 설정 팁

```python
# 기본적인 시작점 (대부분의 경우 잘 작동)
RandomForestClassifier(
    n_estimators=100,        # 보통 100~500이면 충분
    max_depth=None,          # 제한 없음 (데이터가 작으면 10~30 설정)
    min_samples_split=2,     # 기본값
    min_samples_leaf=1,      # 기본값
    max_features='sqrt',     # 분류: 'sqrt', 회귀: 'log2' 또는 1/3
    bootstrap=True,          # 반드시 True
    oob_score=True,          # 검증 세트가 없으면 True
    n_jobs=-1,               # 병렬 처리
    random_state=42          # 재현성
)

# 과적합이 심하면
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,            # 깊이 제한
    min_samples_split=10,    # 분할 조건 강화
    min_samples_leaf=5,      # 리프 크기 증가
    max_features='sqrt'
)

# 언더피팅이면
RandomForestClassifier(
    n_estimators=500,        # 트리 개수 증가
    max_depth=None,          # 제한 완화
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None        # 모든 특성 사용
)
```

# References

## 공식 문서
- [Scikit-learn Decision Tree](https://scikit-learn.org/stable/modules/tree.html)
- [Scikit-learn Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Scikit-learn User Guide: Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)

## 추천 학습 자료
- [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk)
- [StatQuest: Random Forests](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
- [Understanding Random Forests (Berkeley)](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm)

## 논문
- Breiman, L. (2001). "Random Forests". Machine Learning. 45 (1): 5–32.
- Breiman, L., et al. (1984). "Classification and Regression Trees" (CART)

## 관련 알고리즘
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost
- **Extra Trees**: Random Forest의 변형, 더 무작위적
- **Isolation Forest**: 이상치 탐지 전용
