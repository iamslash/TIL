# Abstract

KNN (K-Nearest Neighbors, K-최근접 이웃)은 머신 러닝에서 **가장 단순하면서도 직관적인** 알고리즘 중 하나입니다. "비슷한 것끼리 모인다"는 원리를 그대로 구현한 것으로, 새로운 데이터가 들어오면 **가장 가까운 K개의 이웃**을 찾아 그들의 다수결 또는 평균으로 예측합니다.

**KNN의 핵심 아이디어:**
> "당신의 친구를 보면 당신을 알 수 있다"
> "주변 이웃을 보고 판단하라"

KNN은 **학습이 필요 없는 게으른(Lazy) 알고리즘**입니다. 학습 단계에서는 데이터를 저장만 하고, 예측 시점에 비로소 계산을 시작합니다.

**KNN의 장점:**
- 알고리즘이 매우 단순하고 이해하기 쉬움
- 학습 시간이 거의 없음 (데이터만 저장)
- 분류와 회귀 모두 가능
- 비선형 데이터에도 잘 작동
- 새로운 데이터 추가가 쉬움

**KNN의 단점:**
- 예측 시간이 느림 (모든 데이터와 거리 계산)
- 메모리를 많이 사용 (모든 데이터 저장)
- 차원의 저주(Curse of Dimensionality)에 취약
- K값 선택에 민감
- 불균형 데이터에 약함

**주요 활용 분야:**
- 추천 시스템 (상품, 영화, 음악 추천)
- 이미지 인식 및 분류
- 의료 진단 (환자 데이터 기반 진단)
- 신용 평가
- 패턴 인식
- 이상 탐지

# Materials

- [Scikit-learn KNN Documentation](https://scikit-learn.org/stable/modules/neighbors.html)
- [KNN - Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [StatQuest: K-nearest neighbors](https://www.youtube.com/watch?v=HVXime0nQeI)
- [[인공지능을 위한 머신러닝 101] K-최근접 이웃 (K-Nearest Neighbors, KNN)에 대해 알아보자 | youtube](https://www.youtube.com/watch?v=uaCxu4yPZiI&t=4s)

# Basic

## KNN이란?

KNN (K-Nearest Neighbors)은 **새로운 데이터가 들어왔을 때, 가장 가까운 K개의 이웃을 찾아 그들의 투표(분류) 또는 평균(회귀)으로 예측**하는 알고리즘입니다.

### 범죄 수사 비유로 이해하기

KNN을 이해하는 가장 쉬운 방법은 **범죄 수사관의 프로파일링**을 생각하는 것입니다.

**상황:** 어떤 사람이 범죄 용의자 선상에 올랐습니다. 이 사람이 진짜 범죄자일 가능성이 높은지 판단하려면?

**수사관의 접근 방법:**
1. **행동 패턴 분석**: 용의자와 행동이 가장 비슷한 사람들을 찾는다
2. **이웃 확인**: 기존 데이터에서 가장 비슷한 K명을 찾는다
3. **다수결 판단**: 그 K명 중 범죄자가 많으면 → 범죄자로 판단
4. **확률 계산**: K명 중 범죄자 비율로 확률 추정

이것이 바로 **KNN의 핵심 원리**입니다!

### 실제 예제: 범죄 용의자 프로파일링

다음과 같은 사건 데이터가 있다고 가정합시다:

```
사람 | 야간활동(시간) | SNS활동(회) | 혐의 여부
-----|---------------|------------|----------
A    | 2             | 8          | 무혐의
B    | 7             | 3          | 혐의
C    | 3             | 6          | 무혐의
D    | 8             | 2          | 혐의
E    | 1             | 9          | 무혐의
F    | 6             | 4          | 혐의
G    | 5             | 5          | 무혐의

용의자 X | 4    | 7     | ???
```

**목표:** 용의자 X가 혐의가 있을지 무혐의일지 판단

#### Step 1: 거리 계산 (유사도)

용의자 X(4, 7)와 각 사람 사이의 **유클리드 거리**를 계산합니다.

**유클리드 거리 공식:**
```
d = √[(x₁ - x₂)² + (y₁ - y₂)²]
```

**계산:**

```python
# 용의자 X = (4, 7)

# A = (2, 8)
d(X, A) = √[(4-2)² + (7-8)²] = √[4 + 1] = √5 ≈ 2.24

# B = (7, 3)
d(X, B) = √[(4-7)² + (7-3)²] = √[9 + 16] = √25 = 5.00

# C = (3, 6)
d(X, C) = √[(4-3)² + (7-6)²] = √[1 + 1] = √2 ≈ 1.41

# D = (8, 2)
d(X, D) = √[(4-8)² + (7-2)²] = √[16 + 25] = √41 ≈ 6.40

# E = (1, 9)
d(X, E) = √[(4-1)² + (7-9)²] = √[9 + 4] = √13 ≈ 3.61

# F = (6, 4)
d(X, F) = √[(4-6)² + (7-4)²] = √[4 + 9] = √13 ≈ 3.61

# G = (5, 5)
d(X, G) = √[(4-5)² + (7-5)²] = √[1 + 4] = √5 ≈ 2.24
```

#### Step 2: 거리순 정렬

```
순위 | 사람 | 거리  | 혐의 여부
-----|------|-------|----------
1    | C    | 1.41  | 무혐의
2    | A    | 2.24  | 무혐의
3    | G    | 2.24  | 무혐의
4    | E    | 3.61  | 무혐의
5    | F    | 3.61  | 혐의
6    | B    | 5.00  | 혐의
7    | D    | 6.40  | 혐의
```

#### Step 3: K값에 따른 판단

**K=1 (가장 가까운 1명만 본다):**
```
가장 가까운 이웃: C (무혐의)
→ 예측: 무혐의
```

**K=3 (가장 가까운 3명을 본다):**
```
가까운 3명: C(무혐의), A(무혐의), G(무혐의)
→ 다수결: 무혐의 3표, 혐의 0표
→ 예측: 무혐의
```

**K=5 (가장 가까운 5명을 본다):**
```
가까운 5명: C(무혐의), A(무혐의), G(무혐의), E(무혐의), F(혐의)
→ 다수결: 무혐의 4표, 혐의 1표
→ 예측: 무혐의
```

**K=7 (모든 사람을 본다):**
```
모든 7명: 무혐의 4명, 혐의 3명
→ 다수결: 무혐의 4표, 혐의 3표
→ 예측: 무혐의
```

### K값의 영향

K값에 따라 결과가 달라질 수 있습니다:

| K값 | 특징 | 장점 | 단점 |
|-----|------|------|------|
| **작은 K (1~3)** | 국소적 패턴 | 세밀한 경계선 | 노이즈에 민감, 과적합 |
| **중간 K (5~10)** | 균형 | 안정적 | - |
| **큰 K (>10)** | 전역적 패턴 | 노이즈에 강함 | 경계가 단순해짐, 과소적합 |

**시각화:**

```
K=1: 경계선이 구불구불 (노이즈 포함)
    ●○●○●
    ○●○●○
    ●○●○●

K=5: 부드러운 경계선
    ●●●○○
    ●●●○○
    ○○○○○

K=큼: 거의 직선에 가까운 경계
    ●●●●●
    ○○○○○
    ○○○○○
```

**원칙:**
- **K가 너무 작으면**: 노이즈에 민감, 과적합
- **K가 너무 크면**: 전체 평균만 보게 되어 국소 정보 손실, 과소적합
- **홀수 K 권장**: 동점 방지 (2-class 문제)

## 거리 측정 방법

KNN은 "가까운 이웃"을 찾아야 하므로 **거리 측정**이 핵심입니다.

### 1. 유클리드 거리 (Euclidean Distance)

**가장 일반적**, 직선 거리

```
d = √[Σ(xᵢ - yᵢ)²]
```

**예시:**
```python
A = (1, 2)
B = (4, 6)
d = √[(1-4)² + (2-6)²] = √[9 + 16] = 5
```

**특징:**
- 직관적이고 자연스러움
- L2 norm
- 연속형 데이터에 적합

### 2. 맨해튼 거리 (Manhattan Distance)

**격자 형태** 이동, "택시 거리"

```
d = Σ|xᵢ - yᵢ|
```

**예시:**
```python
A = (1, 2)
B = (4, 6)
d = |1-4| + |2-6| = 3 + 4 = 7
```

**특징:**
- L1 norm
- 격자 구조 데이터에 적합 (도시 블록, 체스판)
- 이상치에 덜 민감

### 3. 민코프스키 거리 (Minkowski Distance)

유클리드와 맨해튼의 **일반화**

```
d = (Σ|xᵢ - yᵢ|ᵖ)^(1/p)

p=1: 맨해튼 거리
p=2: 유클리드 거리
p=∞: 체비셰프 거리
```

### 4. 코사인 유사도 (Cosine Similarity)

**각도**를 측정, 텍스트 데이터에 많이 사용

```
similarity = (A·B) / (||A|| × ||B||)
distance = 1 - similarity
```

**예시:**
```python
A = (1, 2, 3)
B = (2, 4, 6)

A·B = 1×2 + 2×4 + 3×6 = 28
||A|| = √(1² + 2² + 3²) = √14
||B|| = √(2² + 4² + 6²) = √56

similarity = 28 / (√14 × √56) = 1.0  (완전히 같은 방향)
```

**특징:**
- 벡터의 방향만 고려 (크기 무시)
- 문서 유사도, 추천 시스템에 적합

## Python 코드 예제

### 1. KNN 기본 사용 (Iris 데이터)

```python
"""
KNN 기본 예제: Iris 꽃 분류
- 붓꽃 데이터를 KNN으로 분류
"""

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 로드
iris = load_iris()
X = iris.data  # 4개 특성
y = iris.target  # 3개 클래스

print("데이터 크기:", X.shape)
print("특성 이름:", iris.feature_names)
print("클래스 이름:", iris.target_names)

# 2. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. 데이터 표준화 (중요!)
# KNN은 거리 기반이므로 스케일이 다르면 큰 특성이 지배함
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. KNN 모델 생성 및 학습
knn = KNeighborsClassifier(
    n_neighbors=5,        # K값 (이웃 개수)
    weights='uniform',    # 'uniform' 또는 'distance' (거리 가중치)
    metric='euclidean',   # 거리 측정 방법
    algorithm='auto'      # 'auto', 'ball_tree', 'kd_tree', 'brute'
)

# 학습 (실제로는 데이터만 저장)
knn.fit(X_train_scaled, y_train)

# 5. 예측
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n정확도: {accuracy:.4f}")
print("\n분류 리포트:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 6. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# 7. 새로운 데이터 예측
new_flower = [[5.1, 3.5, 1.4, 0.2]]  # 새로운 붓꽃
new_flower_scaled = scaler.transform(new_flower)

prediction = knn.predict(new_flower_scaled)
probabilities = knn.predict_proba(new_flower_scaled)

print(f"\n새로운 꽃 예측: {iris.target_names[prediction[0]]}")
print("각 클래스별 확률:")
for class_name, prob in zip(iris.target_names, probabilities[0]):
    print(f"  {class_name}: {prob:.4f}")

# 8. 이웃 찾기 (가장 가까운 K개)
distances, indices = knn.kneighbors(new_flower_scaled, n_neighbors=5)

print("\n가장 가까운 5개 이웃:")
print(f"거리: {distances[0]}")
print(f"인덱스: {indices[0]}")
for i, idx in enumerate(indices[0]):
    print(f"  {i+1}번째 이웃: {iris.target_names[y_train[idx]]} (거리: {distances[0][i]:.4f})")

# 9. 결정 경계 시각화 (2D로 축소)
plt.figure(figsize=(12, 5))

# 9-1. Petal Length vs Petal Width
plt.subplot(1, 2, 1)
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_test[y_test == i, 2], X_test[y_test == i, 3],
               label=target_name, alpha=0.6, s=50)
plt.scatter(X_test[y_pred != y_test, 2], X_test[y_pred != y_test, 3],
           c='red', marker='x', s=100, linewidths=3, label='Misclassified')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('실제 레이블')
plt.legend()
plt.grid(True, alpha=0.3)

# 9-2. 예측 결과
plt.subplot(1, 2, 2)
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_test[y_pred == i, 2], X_test[y_pred == i, 3],
               label=target_name, alpha=0.6, s=50)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title(f'KNN 예측 (K={knn.n_neighbors}, Accuracy={accuracy:.2f})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('knn_iris_basic.png', dpi=300, bbox_inches='tight')
print("\n시각화가 'knn_iris_basic.png'로 저장되었습니다.")

"""
출력 예시:
데이터 크기: (150, 4)
특성 이름: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
클래스 이름: ['setosa' 'versicolor' 'virginica']

정확도: 1.0000

분류 리포트:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        19
  versicolor       1.00      1.00      1.00        13
   virginica       1.00      1.00      1.00        13

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45

Confusion Matrix:
[[19  0  0]
 [ 0 13  0]
 [ 0  0 13]]

새로운 꽃 예측: setosa
각 클래스별 확률:
  setosa: 1.0000
  versicolor: 0.0000
  virginica: 0.0000

가장 가까운 5개 이웃:
거리: [0.     0.1414 0.1414 0.2236 0.2449]
인덱스: [34 23 12 45  8]
  1번째 이웃: setosa (거리: 0.0000)
  2번째 이웃: setosa (거리: 0.1414)
  3번째 이웃: setosa (거리: 0.1414)
  4번째 이웃: setosa (거리: 0.2236)
  5번째 이웃: setosa (거리: 0.2449)
"""
```

### 2. 범죄 용의자 예제 (동영상 재현)

```python
"""
범죄 용의자 프로파일링 예제
- 동영상 스크립트의 예제를 정확히 재현
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# 1. 데이터 정의
data = pd.DataFrame({
    '사람': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    '야간활동(시간)': [2, 7, 3, 8, 1, 6, 5],
    'SNS활동(회)': [8, 3, 6, 2, 9, 4, 5],
    '혐의여부': ['무혐의', '혐의', '무혐의', '혐의', '무혐의', '혐의', '무혐의']
})

print("범죄 데이터:")
print(data)
print()

# 용의자 X
suspect_X = np.array([[4, 7]])
print(f"용의자 X: 야간활동={suspect_X[0][0]}시간, SNS활동={suspect_X[0][1]}회")
print()

# 2. 특성과 레이블 분리
X = data[['야간활동(시간)', 'SNS활동(회)']].values
y = data['혐의여부'].values

# 레이블을 숫자로 변환
y_numeric = np.where(y == '혐의', 1, 0)

# 3. 거리 계산 (수동)
print("=" * 60)
print("용의자 X와 각 사람 간의 유클리드 거리 계산")
print("=" * 60)

distances = []
for i, person in data.iterrows():
    x1, y1 = person['야간활동(시간)'], person['SNS활동(회)']
    x2, y2 = suspect_X[0]

    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    distances.append({
        '사람': person['사람'],
        '좌표': f"({x1}, {y1})",
        '거리': dist,
        '혐의여부': person['혐의여부']
    })

    print(f"{person['사람']}({x1}, {y1}): "
          f"√[({x1}-4)² + ({y1}-7)²] = √[{(x1-4)**2} + {(y1-7)**2}] = {dist:.2f}")

# 거리순 정렬
distances_df = pd.DataFrame(distances).sort_values('거리')
print("\n" + "=" * 60)
print("거리순 정렬")
print("=" * 60)
print(distances_df.to_string(index=False))

# 4. K값에 따른 판단
print("\n" + "=" * 60)
print("K값에 따른 판단")
print("=" * 60)

for k in [1, 3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y_numeric)

    prediction = knn.predict(suspect_X)
    probabilities = knn.predict_proba(suspect_X)

    # 가장 가까운 K명 확인
    distances_k, indices = knn.kneighbors(suspect_X, n_neighbors=k)

    print(f"\nK={k}:")
    print(f"  가까운 {k}명:")

    count_guilty = 0
    count_innocent = 0

    for i, idx in enumerate(indices[0]):
        label = y[idx]
        print(f"    {i+1}. {data.iloc[idx]['사람']} - {label} (거리: {distances_k[0][i]:.2f})")
        if label == '혐의':
            count_guilty += 1
        else:
            count_innocent += 1

    print(f"  → 혐의 {count_guilty}명, 무혐의 {count_innocent}명")

    result = "혐의" if prediction[0] == 1 else "무혐의"
    print(f"  → 판단: {result}")
    print(f"  → 확률: 무혐의 {probabilities[0][0]:.2f}, 혐의 {probabilities[0][1]:.2f}")

# 5. 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for idx, k in enumerate([1, 3, 5, 7]):
    ax = axes[idx // 2, idx % 2]

    # KNN 학습 및 예측
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y_numeric)
    prediction = knn.predict(suspect_X)

    # 기존 데이터 플롯
    for i, label in enumerate(['무혐의', '혐의']):
        mask = y == label
        color = 'blue' if label == '무혐의' else 'red'
        ax.scatter(X[mask, 0], X[mask, 1], c=color, s=100,
                  label=label, alpha=0.6, edgecolors='black', linewidths=1.5)

        # 레이블 표시
        for j, person in data[mask].iterrows():
            ax.text(person['야간활동(시간)'] + 0.15, person['SNS활동(회)'] + 0.15,
                   person['사람'], fontsize=10, fontweight='bold')

    # 용의자 X
    result_color = 'red' if prediction[0] == 1 else 'blue'
    ax.scatter(suspect_X[0, 0], suspect_X[0, 1], c=result_color, marker='*',
              s=500, edgecolors='black', linewidths=2, label='용의자 X', zorder=5)
    ax.text(suspect_X[0, 0] + 0.15, suspect_X[0, 1] + 0.15, 'X',
           fontsize=12, fontweight='bold')

    # 가장 가까운 K명과 연결선
    distances_k, indices = knn.kneighbors(suspect_X, n_neighbors=k)
    for i, idx in enumerate(indices[0]):
        ax.plot([suspect_X[0, 0], X[idx, 0]],
               [suspect_X[0, 1], X[idx, 1]],
               'g--', alpha=0.5, linewidth=1.5)

    result = "혐의" if prediction[0] == 1 else "무혐의"
    ax.set_xlabel('야간활동 (시간)', fontsize=11)
    ax.set_ylabel('SNS활동 (회)', fontsize=11)
    ax.set_title(f'K={k} → 판단: {result}', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 10)

plt.tight_layout()
plt.savefig('knn_crime_profiling.png', dpi=300, bbox_inches='tight')
print("\n범죄 프로파일링 시각화가 'knn_crime_profiling.png'로 저장되었습니다.")

"""
출력 예시:
범죄 데이터:
  사람  야간활동(시간)  SNS활동(회) 혐의여부
0  A           2         8   무혐의
1  B           7         3    혐의
2  C           3         6   무혐의
3  D           8         2    혐의
4  E           1         9   무혐의
5  F           6         4    혐의
6  G           5         5   무혐의

용의자 X: 야간활동=4시간, SNS활동=7회

============================================================
용의자 X와 각 사람 간의 유클리드 거리 계산
============================================================
A(2, 8): √[(2-4)² + (8-7)²] = √[4 + 1] = 2.24
B(7, 3): √[(7-4)² + (3-7)²] = √[9 + 16] = 5.00
C(3, 6): √[(3-4)² + (6-7)²] = √[1 + 1] = 1.41
D(8, 2): √[(8-4)² + (2-7)²] = √[16 + 25] = 6.40
E(1, 9): √[(1-4)² + (9-7)²] = √[9 + 4] = 3.61
F(6, 4): √[(6-4)² + (4-7)²] = √[4 + 9] = 3.61
G(5, 5): √[(5-4)² + (5-7)²] = √[1 + 4] = 2.24

============================================================
거리순 정렬
============================================================
사람     좌표   거리 혐의여부
 C  (3, 6)  1.41   무혐의
 A  (2, 8)  2.24   무혐의
 G  (5, 5)  2.24   무혐의
 E  (1, 9)  3.61   무혐의
 F  (6, 4)  3.61    혐의
 B  (7, 3)  5.00    혐의
 D  (8, 2)  6.40    혐의

============================================================
K값에 따른 판단
============================================================

K=1:
  가까운 1명:
    1. C - 무혐의 (거리: 1.41)
  → 혐의 0명, 무혐의 1명
  → 판단: 무혐의
  → 확률: 무혐의 1.00, 혐의 0.00

K=3:
  가까운 3명:
    1. C - 무혐의 (거리: 1.41)
    2. A - 무혐의 (거리: 2.24)
    3. G - 무혐의 (거리: 2.24)
  → 혐의 0명, 무혐의 3명
  → 판단: 무혐의
  → 확률: 무혐의 1.00, 혐의 0.00

K=5:
  가까운 5명:
    1. C - 무혐의 (거리: 1.41)
    2. A - 무혐의 (거리: 2.24)
    3. G - 무혐의 (거리: 2.24)
    4. E - 무혐의 (거리: 3.61)
    5. F - 혐의 (거리: 3.61)
  → 혐의 1명, 무혐의 4명
  → 판단: 무혐의
  → 확률: 무혐의 0.80, 혐의 0.20

K=7:
  가까운 7명:
    1. C - 무혐의 (거리: 1.41)
    2. A - 무혐의 (거리: 2.24)
    3. G - 무혐의 (거리: 2.24)
    4. E - 무혐의 (거리: 3.61)
    5. F - 혐의 (거리: 3.61)
    6. B - 혐의 (거리: 5.00)
    7. D - 혐의 (거리: 6.40)
  → 혐의 3명, 무혐의 4명
  → 판단: 무혐의
  → 확률: 무혐의 0.57, 혐의 0.43

관찰:
- 이 예제에서는 모든 K값에서 "무혐의" 판단
- K가 커질수록 혐의 확률이 증가 (0.00 → 0.43)
- K=7이면 거의 동점 (4 vs 3)
"""
```

### 3. KNN 회귀 (Regression)

```python
"""
KNN 회귀 예제: 집값 예측
- KNN은 분류뿐만 아니라 회귀에도 사용 가능
"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 로드
housing = fetch_california_housing()
X = housing.data[:1000]  # 계산 속도를 위해 1000개만
y = housing.target[:1000]

print(f"데이터 크기: {X.shape}")
print(f"특성: {housing.feature_names}")

# 2. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. KNN 회귀 모델
knn_reg = KNeighborsRegressor(
    n_neighbors=5,
    weights='distance',  # 거리 가중치 사용 (가까울수록 영향 큼)
    metric='euclidean'
)

knn_reg.fit(X_train_scaled, y_train)

# 5. 예측 및 평가
y_pred = knn_reg.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nMSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# 6. 예측 vs 실제 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'r--', lw=2, label='Perfect Prediction')
plt.xlabel('실제 집값 ($100k)')
plt.ylabel('예측 집값 ($100k)')
plt.title(f'KNN 회귀: 실제 vs 예측 (K=5, R²={r2:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('knn_regression.png', dpi=300, bbox_inches='tight')
print("\nKNN 회귀 시각화가 저장되었습니다.")

"""
출력 예시:
데이터 크기: (1000, 8)
특성: ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

MSE: 0.4523
RMSE: 0.6725
R² Score: 0.6134

설명:
- KNN 회귀는 K개 이웃의 평균값을 예측값으로 사용
- weights='distance'를 사용하면 가까운 이웃에 더 큰 가중치
- 분류보다 회귀에서는 성능이 다소 떨어질 수 있음
"""
```

### 4. K값 비교 및 최적화

```python
"""
K값 변화에 따른 성능 비교
"""

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 데이터 준비
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# K=1부터 20까지 테스트
K_range = range(1, 21)
train_scores = []
test_scores = []
cv_scores = []

for k in K_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    # 학습 정확도
    train_scores.append(knn.score(X_train_scaled, y_train))

    # 테스트 정확도
    test_scores.append(knn.score(X_test_scaled, y_test))

    # Cross-validation 정확도
    cv_score = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    cv_scores.append(cv_score.mean())

# 시각화
plt.figure(figsize=(12, 6))

plt.plot(K_range, train_scores, 'o-', label='Training Accuracy', linewidth=2)
plt.plot(K_range, test_scores, 's-', label='Test Accuracy', linewidth=2)
plt.plot(K_range, cv_scores, '^-', label='CV Accuracy', linewidth=2)

# 최적 K
optimal_k_test = K_range[np.argmax(test_scores)]
optimal_k_cv = K_range[np.argmax(cv_scores)]

plt.axvline(x=optimal_k_test, color='red', linestyle='--',
           label=f'Optimal K (Test)={optimal_k_test}')
plt.axvline(x=optimal_k_cv, color='green', linestyle='--',
           label=f'Optimal K (CV)={optimal_k_cv}')

plt.xlabel('K (Number of Neighbors)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('KNN: K값에 따른 성능 변화', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(K_range)

plt.savefig('knn_k_optimization.png', dpi=300, bbox_inches='tight')
print(f"최적 K (Test): {optimal_k_test} (정확도: {max(test_scores):.4f})")
print(f"최적 K (CV): {optimal_k_cv} (정확도: {max(cv_scores):.4f})")

"""
출력 예시:
최적 K (Test): 7 (정확도: 1.0000)
최적 K (CV): 13 (정확도: 0.9714)

관찰:
- K=1: 학습 정확도 100%, 테스트는 낮음 (과적합)
- K 증가: 학습 정확도 감소, 테스트 정확도는 일정 지점까지 증가
- K가 너무 크면: 모든 정확도 감소 (과소적합)
- Cross-validation이 더 안정적인 K값 제시
"""
```

### 5. 거리 메트릭 비교

```python
"""
다양한 거리 측정 방법 비교
"""

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 데이터 준비
iris = load_iris()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris.data)
y = iris.target

# 다양한 거리 메트릭
metrics = {
    'euclidean': '유클리드 거리',
    'manhattan': '맨해튼 거리',
    'chebyshev': '체비셰프 거리',
    'minkowski': '민코프스키 거리 (p=3)'
}

# K=1부터 20까지 각 메트릭별 성능
K_range = range(1, 21)
results = {metric: [] for metric in metrics}

for k in K_range:
    for metric in metrics:
        if metric == 'minkowski':
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric, p=3)
        else:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)

        cv_score = cross_val_score(knn, X_scaled, y, cv=5)
        results[metric].append(cv_score.mean())

# 시각화
plt.figure(figsize=(12, 7))

for metric, label in metrics.items():
    plt.plot(K_range, results[metric], 'o-', label=label, linewidth=2, markersize=6)

plt.xlabel('K (Number of Neighbors)', fontsize=12)
plt.ylabel('Cross-Validation Accuracy', fontsize=12)
plt.title('거리 메트릭별 KNN 성능 비교', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(K_range)

plt.savefig('knn_metric_comparison.png', dpi=300, bbox_inches='tight')

# 최적 메트릭 찾기
print("각 거리 메트릭의 최고 정확도:")
for metric, label in metrics.items():
    max_acc = max(results[metric])
    optimal_k = K_range[np.argmax(results[metric])]
    print(f"  {label}: {max_acc:.4f} (K={optimal_k})")

"""
출력 예시:
각 거리 메트릭의 최고 정확도:
  유클리드 거리: 0.9733 (K=13)
  맨해튼 거리: 0.9800 (K=7)
  체비셰프 거리: 0.9667 (K=11)
  민코프스키 거리 (p=3): 0.9733 (K=13)

관찰:
- 데이터에 따라 최적 메트릭이 다름
- Iris 데이터에서는 맨해튼 거리가 가장 좋음
- 일반적으로 유클리드 거리가 무난
- 격자 구조 데이터는 맨해튼이 유리
"""
```

# Advanced

## 최적의 K 찾기

K값은 KNN의 가장 중요한 하이퍼파라미터입니다. 최적의 K를 찾는 방법:

### 1. Cross-Validation

```python
"""
Cross-Validation으로 최적 K 찾기
"""

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np

# 데이터 준비
iris = load_iris()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris.data)
y = iris.target

# Grid Search로 최적 K 찾기
param_grid = {
    'n_neighbors': list(range(1, 31)),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(
    knn,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_scaled, y)

print("최적 파라미터:")
print(grid_search.best_params_)
print(f"\n최적 CV 점수: {grid_search.best_score_:.4f}")

# 결과 시각화
results = grid_search.cv_results_
uniform_euclidean = results['mean_test_score'][
    (results['param_weights'] == 'uniform') &
    (results['param_metric'] == 'euclidean')
]
distance_euclidean = results['mean_test_score'][
    (results['param_weights'] == 'distance') &
    (results['param_metric'] == 'euclidean')
]

plt.figure(figsize=(10, 6))
plt.plot(range(1, 31), uniform_euclidean, 'o-', label='Uniform Weights')
plt.plot(range(1, 31), distance_euclidean, 's-', label='Distance Weights')
plt.axvline(x=grid_search.best_params_['n_neighbors'],
           color='red', linestyle='--', label='Optimal K')
plt.xlabel('K')
plt.ylabel('CV Accuracy')
plt.title('Grid Search: K Optimization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('knn_grid_search.png', dpi=300, bbox_inches='tight')

"""
출력 예시:
최적 파라미터:
{'metric': 'manhattan', 'n_neighbors': 13, 'weights': 'distance'}

최적 CV 점수: 0.9800
"""
```

### 2. Rule of Thumb

**일반적인 경험 법칙:**

```python
# K 선택 가이드라인
n_samples = len(X_train)

# 방법 1: sqrt(n)
k_sqrt = int(np.sqrt(n_samples))

# 방법 2: 홀수 선택 (동점 방지)
if k_sqrt % 2 == 0:
    k_sqrt += 1

# 방법 3: 클래스 개수의 배수 피하기
n_classes = len(np.unique(y_train))
while k_sqrt % n_classes == 0:
    k_sqrt += 2

print(f"추천 K (sqrt rule): {k_sqrt}")
```

## 가중치 KNN (Weighted KNN)

모든 이웃을 똑같이 취급하지 않고, **거리에 따라 가중치**를 부여합니다.

### 원리

```
uniform weights (기본):
  모든 이웃의 투표권이 동일
  vote = 1

distance weights:
  가까운 이웃일수록 더 큰 투표권
  vote = 1 / distance
  (distance가 0이면 무한대 가중치)
```

### Python 구현

```python
"""
Uniform vs Distance Weights 비교
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# 노이즈가 있는 데이터 생성
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    flip_y=0.1,  # 10% 노이즈
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Uniform vs Distance 비교
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, (weights, ax) in enumerate(zip(['uniform', 'distance'], axes)):
    knn = KNeighborsClassifier(n_neighbors=5, weights=weights)
    knn.fit(X_train, y_train)

    # 결정 경계 그리기
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
              cmap='RdYlBu', edgecolors='black', s=50)

    accuracy = knn.score(X_test, y_test)
    ax.set_title(f'{weights.capitalize()} Weights\nAccuracy: {accuracy:.3f}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.savefig('knn_weighted.png', dpi=300, bbox_inches='tight')
print("가중치 비교 시각화가 저장되었습니다.")

"""
관찰:
- Distance weights는 가까운 이웃에 더 의존
- Uniform은 모든 K개 이웃이 동등
- 노이즈가 많으면 distance weights가 유리
- 데이터가 균일하면 큰 차이 없음
"""
```

## 차원의 저주 (Curse of Dimensionality)

KNN의 가장 큰 약점은 **고차원 데이터**에서 성능이 급격히 저하된다는 것입니다.

### 문제점

```
1차원: 10개 점으로 선을 촘촘히 채움
2차원: 100개 점 필요 (10×10)
3차원: 1000개 점 필요 (10×10×10)
10차원: 10^10개 점 필요!

→ 차원이 증가하면 데이터가 희박해짐 (Sparse)
→ "가까운 이웃"이 실제로는 멀리 있음
```

### Python 시연

```python
"""
차원의 저주 시연
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

# 차원을 증가시키면서 성능 측정
dimensions = [2, 5, 10, 20, 50, 100, 200]
accuracies = []

n_samples = 1000  # 고정된 샘플 수

for n_features in dimensions:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(5, n_features),
        n_redundant=0,
        random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=5)
    cv_score = cross_val_score(knn, X, y, cv=5)
    accuracies.append(cv_score.mean())

    print(f"차원={n_features:3d}: 정확도={cv_score.mean():.4f}, "
          f"샘플당 차원={n_features/n_samples:.4f}")

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(dimensions, accuracies, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Number of Dimensions', fontsize=12)
plt.ylabel('Cross-Validation Accuracy', fontsize=12)
plt.title('차원의 저주: 차원 증가에 따른 KNN 성능 저하', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=0.5, color='red', linestyle='--',
           label='Random Guess (50%)', alpha=0.5)
plt.legend()
plt.savefig('curse_of_dimensionality.png', dpi=300, bbox_inches='tight')

"""
출력 예시:
차원=  2: 정확도=0.9180, 샘플당 차원=0.0020
차원=  5: 정확도=0.8740, 샘플당 차원=0.0050
차원= 10: 정확도=0.7960, 샘플당 차원=0.0100
차원= 20: 정확도=0.6920, 샘플당 차원=0.0200
차원= 50: 정확도=0.5980, 샘플당 차원=0.0500
차원=100: 정확도=0.5340, 샘플당 차원=0.1000
차원=200: 정확도=0.5120, 샘플당 차원=0.2000

관찰:
- 차원이 증가하면 정확도가 급격히 감소
- 200차원에서는 거의 랜덤 추측 수준 (50%)
- 샘플 수는 고정되었지만 공간은 기하급수적으로 증가
→ 이것이 "차원의 저주"
```

### 해결 방법

```python
"""
차원 축소로 성능 개선
"""

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 고차원 데이터
X, y = make_classification(
    n_samples=500,
    n_features=100,
    n_informative=10,
    n_redundant=0,
    random_state=42
)

# 원본 KNN
knn_original = KNeighborsClassifier(n_neighbors=5)
score_original = cross_val_score(knn_original, X, y, cv=5).mean()

# PCA + KNN
pca_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),  # 100차원 → 10차원
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
score_pca = cross_val_score(pca_knn, X, y, cv=5).mean()

print(f"원본 (100차원): {score_original:.4f}")
print(f"PCA (10차원): {score_pca:.4f}")
print(f"개선: {(score_pca - score_original) * 100:.2f}%p")

"""
출력 예시:
원본 (100차원): 0.5340
PCA (10차원): 0.8460
개선: 31.20%p

→ 차원 축소로 극적인 성능 향상!
"""
```

## 실전 예제: 영화 추천 시스템

```python
"""
실전 예제: 영화 추천 시스템
- 사용자 평점 기반으로 비슷한 사용자 찾기
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# 1. 가상의 영화 평점 데이터 생성
np.random.seed(42)

users = [f'User{i}' for i in range(1, 21)]
movies = [f'Movie{i}' for i in range(1, 16)]

# 평점 행렬 (User x Movie)
# 일부만 평점을 줌 (Sparse matrix)
ratings_data = []
for user in users:
    n_ratings = np.random.randint(5, 12)  # 각 사용자는 5~11개 영화 평가
    rated_movies = np.random.choice(movies, n_ratings, replace=False)
    for movie in rated_movies:
        rating = np.random.randint(1, 6)  # 1~5점
        ratings_data.append({'user': user, 'movie': movie, 'rating': rating})

ratings_df = pd.DataFrame(ratings_data)

print("평점 데이터 샘플:")
print(ratings_df.head(10))
print(f"\n총 평점 수: {len(ratings_df)}")

# 2. User-Movie 행렬 생성
user_movie_matrix = ratings_df.pivot_table(
    index='user',
    columns='movie',
    values='rating'
).fillna(0)

print(f"\nUser-Movie 행렬 크기: {user_movie_matrix.shape}")
print("\nUser-Movie 행렬 (일부):")
print(user_movie_matrix.iloc[:5, :5])

# 3. KNN 모델 (User-based Collaborative Filtering)
# Sparse matrix로 변환 (메모리 효율)
user_movie_sparse = csr_matrix(user_movie_matrix.values)

# KNN 학습
knn_model = NearestNeighbors(
    n_neighbors=5,
    metric='cosine',  # 코사인 유사도 (추천 시스템에 적합)
    algorithm='brute'
)
knn_model.fit(user_movie_sparse)

# 4. 특정 사용자에게 영화 추천
target_user = 'User1'
target_user_idx = list(user_movie_matrix.index).index(target_user)
target_user_vector = user_movie_sparse[target_user_idx]

# 가장 비슷한 5명의 사용자 찾기
distances, indices = knn_model.kneighbors(
    target_user_vector,
    n_neighbors=6  # 자기 자신 포함
)

print(f"\n{target_user}와 비슷한 사용자들:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    if i == 0:
        continue  # 자기 자신 제외
    similar_user = user_movie_matrix.index[idx]
    similarity = 1 - dist  # 코사인 거리 → 유사도
    print(f"  {i}. {similar_user} (유사도: {similarity:.4f})")

# 5. 추천 영화 생성
# 자신이 보지 않은 영화 중, 비슷한 사용자들이 높게 평가한 영화
target_user_ratings = user_movie_matrix.loc[target_user]
unwatched_movies = target_user_ratings[target_user_ratings == 0].index

similar_users_indices = indices[0][1:]  # 자기 자신 제외
similar_users = user_movie_matrix.index[similar_users_indices]

recommendations = {}
for movie in unwatched_movies:
    # 비슷한 사용자들의 평점 평균
    similar_ratings = user_movie_matrix.loc[similar_users, movie]
    similar_ratings = similar_ratings[similar_ratings > 0]  # 평가한 사용자만

    if len(similar_ratings) > 0:
        avg_rating = similar_ratings.mean()
        recommendations[movie] = avg_rating

# 추천 점수 순으로 정렬
recommendations_sorted = sorted(
    recommendations.items(),
    key=lambda x: x[1],
    reverse=True
)

print(f"\n{target_user}에게 추천하는 영화 (Top 5):")
for i, (movie, score) in enumerate(recommendations_sorted[:5], 1):
    print(f"  {i}. {movie} (예상 평점: {score:.2f})")

# 6. 시각화: 사용자 유사도 네트워크
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(12, 8))

# 타겟 사용자 중심에 배치
ax.scatter(0, 0, c='red', s=500, marker='*',
          edgecolors='black', linewidths=2, zorder=5)
ax.text(0, 0.15, target_user, ha='center', fontsize=12, fontweight='bold')

# 비슷한 사용자들을 원형으로 배치
n_neighbors = len(similar_users)
angles = np.linspace(0, 2*np.pi, n_neighbors, endpoint=False)

for i, (angle, user_idx) in enumerate(zip(angles, similar_users_indices)):
    user = user_movie_matrix.index[user_idx]
    similarity = 1 - distances[0][i+1]

    # 위치 계산
    x = 2 * np.cos(angle)
    y = 2 * np.sin(angle)

    # 사용자 표시
    ax.scatter(x, y, c='blue', s=300, alpha=0.7, edgecolors='black', linewidths=1.5)
    ax.text(x, y+0.25, user, ha='center', fontsize=10)
    ax.text(x, y-0.25, f'{similarity:.3f}', ha='center', fontsize=9, color='gray')

    # 연결선 (유사도에 따라 두께 조절)
    ax.plot([0, x], [0, y], 'k-', alpha=0.3, linewidth=similarity*5)

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title(f'KNN 기반 사용자 유사도 네트워크 ({target_user})',
            fontsize=14, fontweight='bold')

plt.savefig('knn_recommendation_network.png', dpi=300, bbox_inches='tight')
print("\n추천 시스템 네트워크 시각화가 저장되었습니다.")

"""
출력 예시:
평점 데이터 샘플:
     user   movie  rating
0   User1  Movie3       4
1   User1  Movie7       2
2   User1  Movie9       5
...

총 평점 수: 178

User-Movie 행렬 크기: (20, 15)

User1와 비슷한 사용자들:
  1. User15 (유사도: 0.8234)
  2. User7 (유사도: 0.7891)
  3. User12 (유사도: 0.7654)
  4. User3 (유사도: 0.7432)
  5. User18 (유사도: 0.7123)

User1에게 추천하는 영화 (Top 5):
  1. Movie5 (예상 평점: 4.33)
  2. Movie11 (예상 평점: 4.00)
  3. Movie2 (예상 평점: 3.67)
  4. Movie14 (예상 평점: 3.50)
  5. Movie8 (예상 평점: 3.33)

이렇게 KNN을 활용하면:
- Netflix, YouTube 같은 추천 시스템
- "이 상품을 구매한 고객이 함께 본 상품"
- Spotify 음악 추천
등을 구현할 수 있습니다!
"""
```

## KNN의 성능 최적화

KNN은 **예측 시 모든 데이터와 거리를 계산**해야 하므로 느립니다. 최적화 방법:

### 1. KD-Tree

고차원 데이터를 **트리 구조**로 저장하여 빠르게 검색

```python
from sklearn.neighbors import KNeighborsClassifier

# KD-Tree 사용 (저차원에 효과적, < 20차원)
knn_kdtree = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='kd_tree',  # 'ball_tree', 'brute', 'auto'
    leaf_size=30
)
```

**특징:**
- 차원이 낮을 때 매우 빠름 (2~20차원)
- 차원이 높으면 Brute Force보다 느릴 수 있음

### 2. Ball Tree

KD-Tree보다 **고차원에 강함**

```python
knn_balltree = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='ball_tree',
    leaf_size=30
)
```

### 3. 성능 비교

```python
"""
알고리즘별 속도 비교
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
import time

# 데이터 생성
X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
X_test_sample = X[:100]

algorithms = ['brute', 'kd_tree', 'ball_tree', 'auto']
times = {}

for algo in algorithms:
    knn = KNeighborsClassifier(n_neighbors=5, algorithm=algo)

    # 학습 시간
    start = time.time()
    knn.fit(X, y)
    fit_time = time.time() - start

    # 예측 시간
    start = time.time()
    knn.predict(X_test_sample)
    predict_time = time.time() - start

    times[algo] = {'fit': fit_time, 'predict': predict_time}
    print(f"{algo:10s}: Fit={fit_time:.4f}s, Predict={predict_time:.4f}s")

"""
출력 예시 (데이터와 환경에 따라 다름):
brute     : Fit=0.0012s, Predict=0.3456s
kd_tree   : Fit=0.0234s, Predict=0.0123s
ball_tree : Fit=0.0198s, Predict=0.0156s
auto      : Fit=0.0187s, Predict=0.0145s

관찰:
- Brute: 학습 빠름, 예측 매우 느림
- KD-Tree/Ball-Tree: 학습 느림, 예측 빠름
- Auto: 데이터 특성에 따라 자동 선택
"""
```

# References

## 공식 문서
- [Scikit-learn KNN](https://scikit-learn.org/stable/modules/neighbors.html)
- [Scikit-learn Nearest Neighbors Classification](https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html)

## 추천 학습 자료
- [StatQuest: K-nearest neighbors](https://www.youtube.com/watch?v=HVXime0nQeI)
- [KNN Visualization](https://www.cs.cmu.edu/~kdeng/thesis/feature.pdf)
- [The Elements of Statistical Learning - Chapter 13](https://web.stanford.edu/~hastie/ElemStatLearn/)

## 논문
- Fix, E., & Hodges, J. L. (1951). "Discriminatory Analysis. Nonparametric Discrimination: Consistency Properties"
- Cover, T., & Hart, P. (1967). "Nearest neighbor pattern classification". IEEE Transactions on Information Theory.

## KNN vs 다른 알고리즘

| 알고리즘 | 학습 시간 | 예측 시간 | 메모리 | 해석성 | 비선형 | 차원 민감도 |
|---------|----------|----------|--------|--------|--------|------------|
| **KNN** | 빠름 | 느림 | 많음 | 높음 | 가능 | 매우 높음 |
| Decision Tree | 느림 | 빠름 | 적음 | 매우 높음 | 가능 | 낮음 |
| Random Forest | 매우 느림 | 빠름 | 많음 | 중간 | 가능 | 낮음 |
| SVM | 느림 | 빠름 | 중간 | 낮음 | 가능 | 중간 |
| Logistic Reg | 빠름 | 매우 빠름 | 적음 | 높음 | 불가 | 낮음 |
| Neural Net | 매우 느림 | 빠름 | 많음 | 매우 낮음 | 가능 | 중간 |

## 관련 알고리즘
- **K-means Clustering**: 비슷하지만 비지도 학습
- **Radius Neighbors**: K개 대신 반경 내 모든 이웃
- **Local Outlier Factor (LOF)**: KNN 기반 이상치 탐지
- **DBSCAN**: 밀도 기반 클러스터링 (KNN 개념 활용)

## 실전 활용 팁

### 언제 KNN을 사용할까?

✅ **KNN 추천:**
- 데이터가 적을 때 (< 10,000개)
- 저차원일 때 (< 20차원)
- 비선형 패턴이 있을 때
- 새로운 데이터가 자주 추가될 때
- 해석 가능성이 중요할 때
- 빠른 프로토타입이 필요할 때

❌ **KNN 비추천:**
- 데이터가 많을 때 (> 100,000개)
- 고차원일 때 (> 50차원)
- 실시간 예측이 필요할 때
- 메모리가 제한적일 때
- 불균형 데이터일 때

### 성능 개선 체크리스트

```python
# 1. 데이터 표준화 (필수!)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 차원 축소 (고차원일 때)
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X_scaled)

# 3. 최적 K 찾기
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': range(1, 31)}
grid_search = GridSearchCV(knn, param_grid, cv=5)

# 4. Distance weights 사용 (노이즈가 많을 때)
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

# 5. 적절한 알고리즘 선택
# 저차원: kd_tree
# 고차원: ball_tree
# 모르겠으면: auto
knn = KNeighborsClassifier(algorithm='auto')

# 6. 불균형 데이터 처리
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
```

### 디버깅 가이드

```python
# 문제: 정확도가 낮다
# 해결:
# 1. K값 조정 (너무 작거나 크지 않게)
# 2. 데이터 표준화 확인
# 3. 차원 축소 시도
# 4. 특성 엔지니어링

# 문제: 예측이 너무 느리다
# 해결:
# 1. algorithm='kd_tree' 또는 'ball_tree' 사용
# 2. 데이터 샘플링
# 3. 차원 축소
# 4. 다른 알고리즘 고려

# 문제: 메모리 부족
# 해결:
# 1. 데이터 샘플링
# 2. Sparse matrix 사용
# 3. Mini-batch 처리
# 4. 클라우드 컴퓨팅
```
