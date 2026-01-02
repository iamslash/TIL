# Abstract

K-means Clustering은 머신 러닝에서 가장 기본적이고 널리 사용되는 **비지도 학습(Unsupervised Learning)** 알고리즘입니다. 이름 그대로 주어진 데이터를 **K개의 그룹(클러스터)**으로 묶어주는 알고리즘으로, 비슷한 특성을 가진 데이터를 그룹화하여 숨겨진 패턴을 찾아냅니다.

**K-means의 핵심 아이디어:**
> "비슷한 것끼리 가깝게, 다른 것끼리는 멀리"

흩어진 데이터들을 보면 우리는 자연스럽게 몇 개의 그룹으로 나눌 수 있습니다. K-means는 이러한 직관을 수학적으로 구현한 알고리즘입니다.

**K-means의 장점:**
- 알고리즘이 간단하고 이해하기 쉬움
- 계산적으로 효율적 (대용량 데이터에도 빠름)
- 구현이 쉽고 확장 가능
- 결과를 시각화하기 좋음

**K-means의 단점:**
- K값(클러스터 개수)을 사용자가 미리 지정해야 함
- 초기 중심점 위치에 따라 결과가 달라질 수 있음
- 구형(spherical) 클러스터만 잘 찾음 (복잡한 모양은 어려움)
- 이상치(outlier)에 민감함

**주요 활용 분야:**
- 고객 세그멘테이션 (Customer Segmentation)
- 이미지 압축 및 색상 양자화
- 추천 시스템
- 문서 분류
- 이상 탐지

# Materials

- [Scikit-learn K-means Documentation](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [K-means Clustering - Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering)
- [StatQuest: K-means Clustering](https://www.youtube.com/watch?v=4b5d3muPQmA)
- [[인공지능을 위한 머신러닝 101] K-평균 클러스터링 알고리즘을 소개합니다! | youtube](https://www.youtube.com/watch?v=0pr4gbuc5E0)

# Basic

## K-means란?

K-means Clustering은 데이터를 **K개의 클러스터로 분할**하는 알고리즘입니다.

### 직관적 이해

다음과 같이 데이터들이 흩어져 있다고 상상해봅시다:

```
    •     •  •
         •
  •           •
               •  •
    •  •
         •        •
```

사람이 보면 자연스럽게 3개의 그룹으로 나눌 수 있습니다:

```
   [그룹 1]     [그룹 2]
    •     •  •
         •
  •
               [그룹 3]
                 •  •
    •  •
         •        •
```

K-means는 이러한 "자연스러운 그룹핑"을 자동으로 수행합니다!

### K-means의 목표

**목표 함수 (Objective Function):**

각 클러스터는 **중심점(Centroid)**을 가지며, K-means의 목표는 다음을 최소화하는 것입니다:

```
J = Σ Σ ||x_i - μ_k||²
    k  i∈Ck

여기서:
- J: 총 클러스터 내 분산 (Within-Cluster Sum of Squares, WCSS)
- x_i: 각 데이터 포인트
- μ_k: k번째 클러스터의 중심점
- C_k: k번째 클러스터에 속한 데이터들
- ||·||: 유클리드 거리
```

**쉽게 말하면:**
- 같은 클러스터 내 데이터들은 **최대한 서로 가깝게**
- 다른 클러스터의 데이터들과는 **최대한 멀리**

## K-means 알고리즘 단계

K-means는 매우 간단한 2단계를 반복합니다:

### 전체 과정 요약

```
1. 초기화: K개의 중심점을 임의로 선택
2. 반복:
   a. 할당(Assignment): 각 데이터를 가장 가까운 중심점의 클러스터에 할당
   b. 업데이트(Update): 각 클러스터의 중심점을 재계산 (평균)
3. 종료: 클러스터 할당이 변하지 않으면 종료
```

### 상세 예제: 6개 데이터 포인트

다음 6개의 데이터 포인트를 K=2 (2개의 클러스터)로 묶어봅시다:

```
A = (1, 1)
B = (2, 1)
C = (4, 3)
D = (5, 4)
E = (6, 5)
F = (7, 6)
```

#### Step 1: 초기화 (Initialization)

K=2이므로 2개의 초기 중심점을 임의로 선택합니다.

```
C1 = A = (1, 1)  ← 첫 번째 중심점
C2 = F = (7, 6)  ← 두 번째 중심점
```

**시각화:**
```
   7 |                    F★
   6 |                E
   5 |           D
   4 |
   3 |       C
   2 |
   1 | A★  B
   0 +---+---+---+---+---+---+---
     0   1   2   3   4   5   6   7

★ = 중심점
```

#### Step 2: 할당 (Assignment)

각 데이터 포인트에서 두 중심점까지의 거리를 계산하여, 더 가까운 클러스터에 할당합니다.

**유클리드 거리 공식:**
```
d = √[(x₁ - x₂)² + (y₁ - y₂)²]
```

**데이터 B의 할당:**
```
B = (2, 1)

B와 C1의 거리:
d(B, C1) = √[(2-1)² + (1-1)²] = √[1 + 0] = 1.0

B와 C2의 거리:
d(B, C2) = √[(2-7)² + (1-6)²] = √[25 + 25] = √50 ≈ 7.07

결론: B는 C1에 더 가까우므로 Cluster 1에 할당
```

**데이터 C의 할당:**
```
C = (4, 3)

C와 C1의 거리:
d(C, C1) = √[(4-1)² + (3-1)²] = √[9 + 4] = √13 ≈ 3.61

C와 C2의 거리:
d(C, C2) = √[(4-7)² + (3-6)²] = √[9 + 9] = √18 ≈ 4.24

결론: C는 C1에 더 가까우므로 Cluster 1에 할당
```

**데이터 D의 할당:**
```
D = (5, 4)

D와 C1의 거리:
d(D, C1) = √[(5-1)² + (4-1)²] = √[16 + 9] = √25 = 5.0

D와 C2의 거리:
d(D, C2) = √[(5-7)² + (4-6)²] = √[4 + 4] = √8 ≈ 2.83

결론: D는 C2에 더 가까우므로 Cluster 2에 할당
```

**데이터 E의 할당:**
```
E = (6, 5)

E와 C1의 거리:
d(E, C1) = √[(6-1)² + (5-1)²] = √[25 + 16] = √41 ≈ 6.40

E와 C2의 거리:
d(E, C2) = √[(6-7)² + (5-6)²] = √[1 + 1] = √2 ≈ 1.41

결론: E는 C2에 더 가까우므로 Cluster 2에 할당
```

**1차 할당 결과:**
```
Cluster 1: {A, B, C}
Cluster 2: {D, E, F}

시각화:
   7 |                    F★
   6 |                E  [Cluster 2]
   5 |           D
   4 |
   3 |       C  [Cluster 1]
   2 |
   1 | A★  B
   0 +---+---+---+---+---+---+---
     0   1   2   3   4   5   6   7
```

#### Step 3: 업데이트 (Update)

각 클러스터에 속한 데이터들의 **평균값**을 계산하여 새로운 중심점을 구합니다.

**새로운 C1 계산:**
```
Cluster 1: A(1,1), B(2,1), C(4,3)

C1_new = (평균 x좌표, 평균 y좌표)
       = ((1+2+4)/3, (1+1+3)/3)
       = (7/3, 5/3)
       = (2.33, 1.67)
```

**새로운 C2 계산:**
```
Cluster 2: D(5,4), E(6,5), F(7,6)

C2_new = ((5+6+7)/3, (4+5+6)/3)
       = (18/3, 15/3)
       = (6.0, 5.0)
```

**업데이트된 중심점:**
```
   7 |                    F
   6 |                E
   5 |           D    C2★ (6.0, 5.0)
   4 |
   3 |       C
   2 | C1★ (2.33, 1.67)
   1 | A   B
   0 +---+---+---+---+---+---+---
     0   1   2   3   4   5   6   7
```

#### Step 4: 다시 할당 (Re-assignment)

새로운 중심점으로 다시 거리를 계산하고 할당합니다.

**데이터 C 재확인:**
```
C = (4, 3)

C와 C1_new의 거리:
d(C, C1) = √[(4-2.33)² + (3-1.67)²] = √[2.79 + 1.77] = √4.56 ≈ 2.14

C와 C2_new의 거리:
d(C, C2) = √[(4-6.0)² + (3-5.0)²] = √[4 + 4] = √8 ≈ 2.83

결론: C는 여전히 Cluster 1에 속함
```

**데이터 D 재확인:**
```
D = (5, 4)

D와 C1_new의 거리:
d(D, C1) = √[(5-2.33)² + (4-1.67)²] = √[7.13 + 5.43] = √12.56 ≈ 3.54

D와 C2_new의 거리:
d(D, C2) = √[(5-6.0)² + (4-5.0)²] = √[1 + 1] = √2 ≈ 1.41

결론: D는 여전히 Cluster 2에 속함
```

**모든 데이터를 재확인한 결과:**
- 모든 데이터가 여전히 같은 클러스터에 속함
- **클러스터 할당에 변화 없음 → 알고리즘 종료!**

**최종 결과:**
```
Cluster 1: {A, B, C}
중심점: (2.33, 1.67)

Cluster 2: {D, E, F}
중심점: (6.0, 5.0)
```

### 종료 조건

K-means는 다음 중 하나가 만족되면 종료합니다:

1. **클러스터 할당이 변하지 않음** (가장 일반적)
2. **중심점이 거의 움직이지 않음** (threshold 이하)
3. **최대 반복 횟수 도달** (무한 루프 방지)

## Python 코드 예제

### 1. K-means 기본 사용 (Iris 데이터)

```python
"""
K-means 기본 예제: Iris 데이터 클러스터링
- 붓꽃 데이터를 3개의 클러스터로 분류
"""

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 로드
iris = load_iris()
X = iris.data  # 4개 특성: sepal length, sepal width, petal length, petal width
y_true = iris.target  # 실제 레이블 (비교용)

print("데이터 크기:", X.shape)
print("특성 이름:", iris.feature_names)
print("클래스 이름:", iris.target_names)

# 2. 데이터 표준화 (선택사항이지만 권장)
# K-means는 거리 기반이므로 스케일이 중요!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. K-means 모델 생성 및 학습
kmeans = KMeans(
    n_clusters=3,           # 클러스터 개수
    init='k-means++',       # 초기화 방법 ('k-means++' 권장)
    n_init=10,              # 다른 초기 중심점으로 실행 횟수
    max_iter=300,           # 최대 반복 횟수
    random_state=42
)

# 학습 (fit)
kmeans.fit(X_scaled)

# 4. 결과 확인
y_kmeans = kmeans.labels_  # 각 데이터의 클러스터 레이블
centers = kmeans.cluster_centers_  # 각 클러스터의 중심점
inertia = kmeans.inertia_  # WCSS (Within-Cluster Sum of Squares)

print(f"\n클러스터 레이블: {y_kmeans[:10]}...")  # 처음 10개만 출력
print(f"클러스터 중심점:\n{centers}")
print(f"WCSS (Inertia): {inertia:.4f}")

# 5. 클러스터별 데이터 개수
unique, counts = np.unique(y_kmeans, return_counts=True)
print("\n클러스터별 데이터 개수:")
for cluster, count in zip(unique, counts):
    print(f"  Cluster {cluster}: {count}개")

# 6. 실제 레이블과 비교 (참고용)
from sklearn.metrics import adjusted_rand_score, silhouette_score

ari = adjusted_rand_score(y_true, y_kmeans)
silhouette = silhouette_score(X_scaled, y_kmeans)

print(f"\nAdjusted Rand Index: {ari:.4f}")  # 실제 레이블과 얼마나 비슷한지
print(f"Silhouette Score: {silhouette:.4f}")  # 클러스터링 품질 (-1 ~ 1, 높을수록 좋음)

# 7. 시각화 (2D로 축소: petal length vs petal width)
plt.figure(figsize=(12, 5))

# 7-1. 실제 레이블
plt.subplot(1, 2, 1)
plt.scatter(X[:, 2], X[:, 3], c=y_true, cmap='viridis', s=50, alpha=0.6)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('실제 레이블')
plt.colorbar(label='Species')

# 7-2. K-means 결과
plt.subplot(1, 2, 2)
plt.scatter(X[:, 2], X[:, 3], c=y_kmeans, cmap='viridis', s=50, alpha=0.6)

# 중심점 표시 (원래 스케일로 복원)
centers_original = scaler.inverse_transform(centers)
plt.scatter(centers_original[:, 2], centers_original[:, 3],
           c='red', marker='X', s=200, edgecolors='black', linewidths=2,
           label='Centroids')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('K-means 클러스터링 (K=3)')
plt.colorbar(label='Cluster')
plt.legend()

plt.tight_layout()
plt.savefig('kmeans_iris_basic.png', dpi=300, bbox_inches='tight')
print("\n시각화가 'kmeans_iris_basic.png'로 저장되었습니다.")

"""
출력 예시:
데이터 크기: (150, 4)
특성 이름: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
클래스 이름: ['setosa' 'versicolor' 'virginica']

클러스터 레이블: [1 1 1 1 1 1 1 1 1 1...]
클러스터 중심점:
[[-1.01457897  0.85326268 -1.30498732 -1.25489349]
 [ 1.13597027  0.08842168  0.99615451  1.01752612]
 [-0.05021989 -0.88337647  0.34773781  0.2815273 ]]
WCSS (Inertia): 78.8514

클러스터별 데이터 개수:
  Cluster 0: 50개
  Cluster 1: 47개
  Cluster 2: 53개

Adjusted Rand Index: 0.7302
Silhouette Score: 0.5528
"""
```

### 2. 수동 구현: 단계별 K-means

```python
"""
K-means 수동 구현: A, B, C, D, E, F 예제
- 알고리즘의 각 단계를 직접 구현하여 이해
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. 데이터 정의
data = np.array([
    [1, 1],  # A
    [2, 1],  # B
    [4, 3],  # C
    [5, 4],  # D
    [6, 5],  # E
    [7, 6]   # F
])

labels = ['A', 'B', 'C', 'D', 'E', 'F']
K = 2  # 클러스터 개수

print("데이터:")
for label, point in zip(labels, data):
    print(f"  {label}: {point}")

# 2. 유클리드 거리 계산 함수
def euclidean_distance(point1, point2):
    """두 점 사이의 유클리드 거리"""
    return np.sqrt(np.sum((point1 - point2) ** 2))

# 3. 초기 중심점 설정 (A와 F)
centroids = np.array([
    data[0],  # A = (1, 1)
    data[5]   # F = (7, 6)
])

print(f"\n초기 중심점:")
print(f"  C1: {centroids[0]}")
print(f"  C2: {centroids[1]}")

# 4. K-means 알고리즘 실행
max_iterations = 10
history = []  # 시각화를 위한 히스토리

for iteration in range(max_iterations):
    print(f"\n===== Iteration {iteration + 1} =====")

    # Step 1: 할당 (Assignment)
    clusters = [[] for _ in range(K)]
    cluster_indices = []

    for i, point in enumerate(data):
        # 각 중심점까지의 거리 계산
        distances = [euclidean_distance(point, centroid) for centroid in centroids]

        # 가장 가까운 중심점 찾기
        closest_centroid = np.argmin(distances)
        clusters[closest_centroid].append(point)
        cluster_indices.append(closest_centroid)

        print(f"  {labels[i]}{point}: ", end='')
        for j, dist in enumerate(distances):
            print(f"d(C{j+1})={dist:.2f}", end=' ')
        print(f"→ Cluster {closest_centroid + 1}")

    # 히스토리 저장
    history.append({
        'centroids': centroids.copy(),
        'clusters': cluster_indices.copy()
    })

    # Step 2: 업데이트 (Update)
    new_centroids = np.array([np.mean(cluster, axis=0) if cluster else centroids[i]
                              for i, cluster in enumerate(clusters)])

    print("\n새로운 중심점:")
    for i, centroid in enumerate(new_centroids):
        print(f"  C{i+1}: {centroid}")

    # 종료 조건: 중심점이 변하지 않으면 종료
    if np.allclose(centroids, new_centroids):
        print("\n중심점이 변하지 않음. 알고리즘 종료!")
        break

    centroids = new_centroids

# 최종 결과
print("\n" + "="*50)
print("최종 클러스터링 결과")
print("="*50)
for i in range(K):
    cluster_points = [labels[j] for j, c in enumerate(cluster_indices) if c == i]
    print(f"Cluster {i+1}: {cluster_points}")
    print(f"  중심점: {centroids[i]}")

# 5. 시각화
fig, axes = plt.subplots(1, len(history), figsize=(5*len(history), 4))
if len(history) == 1:
    axes = [axes]

colors = ['red', 'blue', 'green', 'orange']

for idx, (ax, state) in enumerate(zip(axes, history)):
    # 데이터 포인트
    for i, (point, label) in enumerate(zip(data, labels)):
        cluster = state['clusters'][i]
        ax.scatter(point[0], point[1], c=colors[cluster], s=100, alpha=0.6)
        ax.text(point[0], point[1]+0.2, label, ha='center', fontsize=12, fontweight='bold')

    # 중심점
    for i, centroid in enumerate(state['centroids']):
        ax.scatter(centroid[0], centroid[1], c=colors[i], marker='X',
                  s=300, edgecolors='black', linewidths=2, label=f'C{i+1}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Iteration {idx + 1}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-0.5, 7)

plt.tight_layout()
plt.savefig('kmeans_manual_steps.png', dpi=300, bbox_inches='tight')
print("\n단계별 시각화가 'kmeans_manual_steps.png'로 저장되었습니다.")

"""
출력 예시:
데이터:
  A: [1 1]
  B: [2 1]
  C: [4 3]
  D: [5 4]
  E: [6 5]
  F: [7 6]

초기 중심점:
  C1: [1 1]
  C2: [7 6]

===== Iteration 1 =====
  A[1 1]: d(C1)=0.00 d(C2)=7.81 → Cluster 1
  B[2 1]: d(C1)=1.00 d(C2)=7.07 → Cluster 1
  C[4 3]: d(C1)=3.61 d(C2)=4.24 → Cluster 1
  D[5 4]: d(C1)=5.00 d(C2)=2.83 → Cluster 2
  E[6 5]: d(C1)=6.40 d(C2)=1.41 → Cluster 2
  F[7 6]: d(C1)=7.81 d(C2)=0.00 → Cluster 2

새로운 중심점:
  C1: [2.33333333 1.66666667]
  C2: [6. 5.]

===== Iteration 2 =====
  A[1 1]: d(C1)=1.49 d(C2)=6.40 → Cluster 1
  B[2 1]: d(C1)=0.75 d(C2)=5.66 → Cluster 1
  C[4 3]: d(C1)=2.14 d(C2)=2.83 → Cluster 1
  D[5 4]: d(C1)=3.54 d(C2)=1.41 → Cluster 2
  E[6 5]: d(C1)=4.56 d(C2)=0.00 → Cluster 2
  F[7 6]: d(C1)=5.75 d(C2)=1.41 → Cluster 2

새로운 중심점:
  C1: [2.33333333 1.66666667]
  C2: [6. 5.]

중심점이 변하지 않음. 알고리즘 종료!

==================================================
최종 클러스터링 결과
==================================================
Cluster 1: ['A', 'B', 'C']
  중심점: [2.33333333 1.66666667]
Cluster 2: ['D', 'E', 'F']
  중심점: [6. 5.]
"""
```

### 3. Scikit-learn으로 간단히 구현

```python
"""
Scikit-learn K-means로 A~F 데이터 클러스터링
"""

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# 데이터
data = np.array([[1,1], [2,1], [4,3], [5,4], [6,5], [7,6]])
labels = ['A', 'B', 'C', 'D', 'E', 'F']

# K-means 학습
kmeans = KMeans(n_clusters=2, init=np.array([[1,1], [7,6]]),
                n_init=1, random_state=42)
kmeans.fit(data)

# 결과
print("클러스터 레이블:", kmeans.labels_)
print("중심점:\n", kmeans.cluster_centers_)
print("WCSS:", kmeans.inertia_)

# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='viridis', s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           c='red', marker='X', s=300, edgecolors='black', linewidths=2,
           label='Centroids')

for i, label in enumerate(labels):
    plt.text(data[i, 0], data[i, 1]+0.2, label, ha='center',
            fontsize=12, fontweight='bold')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-means Clustering (K=2)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('kmeans_sklearn_simple.png', dpi=300, bbox_inches='tight')
print("\n시각화가 저장되었습니다.")

"""
출력:
클러스터 레이블: [0 0 0 1 1 1]
중심점:
 [[2.33333333 1.66666667]
 [6.         5.        ]]
WCSS: 6.666666666666667
"""
```

### 4. K-means 클러스터링 애니메이션

```python
"""
K-means 알고리즘의 동작 과정을 애니메이션으로 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# 데이터 생성
np.random.seed(42)
X = np.vstack([
    np.random.randn(50, 2) + [2, 2],
    np.random.randn(50, 2) + [8, 8],
    np.random.randn(50, 2) + [8, 2]
])

# K-means 수동 구현 (상태 저장)
def kmeans_with_history(X, k, max_iter=20):
    # 초기 중심점 (랜덤)
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    history = []

    for iteration in range(max_iter):
        # 할당
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        history.append((centroids.copy(), labels.copy()))

        # 업데이트
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return history

history = kmeans_with_history(X, k=3)

# 애니메이션
fig, ax = plt.subplots(figsize=(10, 8))

def animate(i):
    ax.clear()
    centroids, labels = history[i]

    # 데이터 포인트
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                        s=30, alpha=0.6)

    # 중심점
    ax.scatter(centroids[:, 0], centroids[:, 1],
              c='red', marker='X', s=300, edgecolors='black', linewidths=2)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'K-means Iteration {i + 1}/{len(history)}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)

anim = FuncAnimation(fig, animate, frames=len(history), interval=800, repeat=True)
anim.save('kmeans_animation.gif', writer='pillow', fps=1)
print("애니메이션이 'kmeans_animation.gif'로 저장되었습니다.")

# 정적 이미지도 저장
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (ax, (centroids, labels)) in enumerate(zip(axes, history[:6])):
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, alpha=0.6)
    ax.scatter(centroids[:, 0], centroids[:, 1],
              c='red', marker='X', s=300, edgecolors='black', linewidths=2)
    ax.set_title(f'Iteration {idx + 1}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_iterations.png', dpi=300, bbox_inches='tight')
print("반복 과정이 'kmeans_iterations.png'로 저장되었습니다.")
```

# Advanced

## 최적의 K 찾기: Elbow Method

K-means의 가장 큰 난제는 **"K를 몇으로 해야 할까?"**입니다.

### Elbow Method (팔꿈치 방법)

**아이디어:** 다양한 K 값에 대해 WCSS를 계산하고, 그래프를 그려서 "꺾이는 지점"을 찾습니다.

**WCSS (Within-Cluster Sum of Squares):**
- 각 데이터와 해당 클러스터 중심점 간의 거리 제곱의 합
- 클러스터가 많을수록 WCSS는 감소
- K=N(데이터 개수)이면 WCSS=0

**Elbow Point:**
- WCSS가 급격히 감소하다가 완만해지는 지점
- 팔꿈치처럼 꺾이는 모양

### Python 구현

```python
"""
Elbow Method로 최적의 K 찾기
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 생성 (실제로는 5개의 클러스터)
X, y_true = make_blobs(n_samples=300, centers=5,
                       cluster_std=0.60, random_state=42)

print(f"데이터 크기: {X.shape}")
print(f"실제 클러스터 개수: {len(np.unique(y_true))}")

# 2. K=1부터 10까지 WCSS 계산
K_range = range(1, 11)
wcss_list = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss_list.append(kmeans.inertia_)
    print(f"K={k}: WCSS={kmeans.inertia_:.2f}")

# 3. Elbow Plot
plt.figure(figsize=(10, 6))
plt.plot(K_range, wcss_list, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
plt.title('Elbow Method for Optimal K', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(K_range)

# Elbow 지점 강조 (여기서는 K=5)
plt.axvline(x=5, color='red', linestyle='--', linewidth=2, label='Optimal K=5')
plt.legend()

plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
print("\nElbow Method 그래프가 'elbow_method.png'로 저장되었습니다.")

# 4. 실제로 K=5로 클러스터링
optimal_k = 5
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
y_pred = kmeans_optimal.fit_predict(X)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 실제 레이블
axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=30, alpha=0.6)
axes[0].set_title('실제 클러스터 (Ground Truth)')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# K-means 결과
axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=30, alpha=0.6)
axes[1].scatter(kmeans_optimal.cluster_centers_[:, 0],
               kmeans_optimal.cluster_centers_[:, 1],
               c='red', marker='X', s=300, edgecolors='black', linewidths=2)
axes[1].set_title(f'K-means (K={optimal_k})')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')

plt.tight_layout()
plt.savefig('optimal_k_result.png', dpi=300, bbox_inches='tight')
print("최적 K 결과가 'optimal_k_result.png'로 저장되었습니다.")

"""
출력 예시:
데이터 크기: (300, 2)
실제 클러스터 개수: 5
K=1: WCSS=2370.32
K=2: WCSS=1044.83
K=3: WCSS=626.09
K=4: WCSS=427.98
K=5: WCSS=279.29  ← Elbow!
K=6: WCSS=220.45
K=7: WCSS=188.63
K=8: WCSS=167.93
K=9: WCSS=149.89
K=10: WCSS=135.38

관찰:
- K=1→2: 급격히 감소 (-56%)
- K=2→3: 여전히 큰 감소 (-40%)
- K=3→4: 감소 (-32%)
- K=4→5: 감소 (-35%)
- K=5→6: 완만한 감소 (-21%) ← Elbow!
- K≥6: 완만한 감소 계속

→ K=5가 최적!
"""
```

## Silhouette Score (실루엣 점수)

Elbow Method는 주관적일 수 있습니다. **Silhouette Score**는 클러스터링 품질을 **정량적**으로 평가합니다.

### Silhouette Score란?

각 데이터 포인트에 대해:

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

여기서:
- a(i): i와 같은 클러스터 내 다른 점들과의 평균 거리 (작을수록 좋음)
- b(i): i와 가장 가까운 다른 클러스터의 점들과의 평균 거리 (클수록 좋음)
```

**해석:**
- `-1 ≤ s(i) ≤ 1`
- `s(i) ≈ 1`: 잘 분류됨 (같은 클러스터 내에서 가깝고, 다른 클러스터와 멀리 떨어짐)
- `s(i) ≈ 0`: 경계에 있음 (애매함)
- `s(i) < 0`: 잘못 분류됨 (다른 클러스터에 더 가까움)

**전체 Silhouette Score:** 모든 점의 s(i) 평균

### Python 구현

```python
"""
Silhouette Score로 최적의 K 찾기
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
X, _ = make_blobs(n_samples=300, centers=5, cluster_std=0.60, random_state=42)

# K=2부터 10까지 Silhouette Score 계산
K_range = range(2, 11)
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
    print(f"K={k}: Silhouette Score={score:.4f}")

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Silhouette Score for Different K', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(K_range)

# 최적 K 강조
optimal_k = K_range[np.argmax(silhouette_scores)]
plt.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
           label=f'Optimal K={optimal_k}')
plt.legend()

plt.savefig('silhouette_score_plot.png', dpi=300, bbox_inches='tight')
print(f"\n최적 K: {optimal_k} (Silhouette Score: {max(silhouette_scores):.4f})")

"""
출력 예시:
K=2: Silhouette Score=0.6485
K=3: Silhouette Score=0.6140
K=4: Silhouette Score=0.5820
K=5: Silhouette Score=0.6531  ← 최고!
K=6: Silhouette Score=0.6018
K=7: Silhouette Score=0.5650
K=8: Silhouette Score=0.5512
K=9: Silhouette Score=0.5389
K=10: Silhouette Score=0.5301

최적 K: 5 (Silhouette Score: 0.6531)
"""
```

### Silhouette Plot (상세 분석)

```python
"""
Silhouette Plot으로 각 클러스터의 품질 시각화
"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# K=5로 클러스터링
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X)

# Silhouette values 계산
silhouette_vals = silhouette_samples(X, labels)
silhouette_avg = silhouette_score(X, labels)

# Silhouette Plot
fig, ax = plt.subplots(figsize=(10, 7))

y_lower = 10
for i in range(5):
    # i번째 클러스터의 silhouette values
    cluster_silhouette_vals = silhouette_vals[labels == i]
    cluster_silhouette_vals.sort()

    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / 5)
    ax.fill_betweenx(np.arange(y_lower, y_upper),
                     0, cluster_silhouette_vals,
                     facecolor=color, edgecolor=color, alpha=0.7)

    # 클러스터 라벨
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    y_lower = y_upper + 10

ax.set_xlabel('Silhouette Coefficient Values')
ax.set_ylabel('Cluster Label')
ax.set_title(f'Silhouette Plot (K=5, Avg Score={silhouette_avg:.3f})')

# 평균 점수 라인
ax.axvline(x=silhouette_avg, color='red', linestyle='--', linewidth=2,
          label=f'Average Score={silhouette_avg:.3f}')
ax.legend()

plt.savefig('silhouette_plot.png', dpi=300, bbox_inches='tight')
print("Silhouette Plot이 'silhouette_plot.png'로 저장되었습니다.")

"""
해석:
- 각 클러스터별로 silhouette 값의 분포를 확인
- 모든 클러스터가 평균선(빨간 점선)보다 오른쪽에 있으면 좋음
- 클러스터 크기가 비슷하면 좋음
- 음수 값이 많으면 잘못된 클러스터링
"""
```

## K-means++ 초기화

K-means의 결과는 **초기 중심점 위치**에 민감합니다. K-means++는 더 좋은 초기 중심점을 선택하는 방법입니다.

### K-means++ 알고리즘

```
1. 첫 번째 중심점: 랜덤하게 선택
2. 나머지 중심점들:
   - 각 데이터에 대해 "가장 가까운 기존 중심점까지의 거리"를 계산
   - 거리가 먼 데이터일수록 높은 확률로 다음 중심점으로 선택
   - (이미 중심점이 있는 지역을 피하고, 먼 곳에 중심점을 배치)
3. K개의 중심점이 모두 선택될 때까지 반복
```

**장점:** 더 빠른 수렴, 더 좋은 결과, local minimum 회피

### Python 구현

```python
"""
Random Initialization vs K-means++ 비교
"""

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# 1. Random Initialization (여러 번 실행)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, ax in enumerate(axes[0]):
    kmeans_random = KMeans(n_clusters=4, init='random', n_init=1,
                          random_state=i)
    labels = kmeans_random.fit_predict(X)

    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, alpha=0.6)
    ax.scatter(kmeans_random.cluster_centers_[:, 0],
              kmeans_random.cluster_centers_[:, 1],
              c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    ax.set_title(f'Random Init (Run {i+1})\nWCSS={kmeans_random.inertia_:.2f}')

# 2. K-means++ Initialization
for i, ax in enumerate(axes[1]):
    kmeans_pp = KMeans(n_clusters=4, init='k-means++', n_init=1,
                      random_state=i)
    labels = kmeans_pp.fit_predict(X)

    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, alpha=0.6)
    ax.scatter(kmeans_pp.cluster_centers_[:, 0],
              kmeans_pp.cluster_centers_[:, 1],
              c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    ax.set_title(f'K-means++ (Run {i+1})\nWCSS={kmeans_pp.inertia_:.2f}')

plt.tight_layout()
plt.savefig('kmeans_initialization_comparison.png', dpi=300, bbox_inches='tight')
print("초기화 비교가 'kmeans_initialization_comparison.png'로 저장되었습니다.")

# 3. 수렴 속도 비교
n_runs = 50
random_iterations = []
kmeans_pp_iterations = []

for i in range(n_runs):
    # Random
    kmeans_random = KMeans(n_clusters=4, init='random', n_init=1,
                          max_iter=100, random_state=i)
    kmeans_random.fit(X)
    random_iterations.append(kmeans_random.n_iter_)

    # K-means++
    kmeans_pp = KMeans(n_clusters=4, init='k-means++', n_init=1,
                      max_iter=100, random_state=i)
    kmeans_pp.fit(X)
    kmeans_pp_iterations.append(kmeans_pp.n_iter_)

print(f"\n평균 수렴 반복 횟수 ({n_runs}회 실행):")
print(f"  Random Initialization: {np.mean(random_iterations):.2f}")
print(f"  K-means++: {np.mean(kmeans_pp_iterations):.2f}")
print(f"  개선: {np.mean(random_iterations) - np.mean(kmeans_pp_iterations):.2f} iterations")

"""
출력 예시:
평균 수렴 반복 횟수 (50회 실행):
  Random Initialization: 8.34
  K-means++: 5.12
  개선: 3.22 iterations

관찰:
- K-means++가 더 빠르게 수렴
- Random 초기화는 때때로 나쁜 local minimum에 빠짐
- K-means++는 일관되게 좋은 결과
"""
```

## 실전 예제: 고객 세그멘테이션

```python
"""
실전 예제: 고객 세그멘테이션 (Customer Segmentation)
- RFM 분석: Recency, Frequency, Monetary
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 가상의 고객 데이터 생성
np.random.seed(42)
n_customers = 500

data = pd.DataFrame({
    'CustomerID': range(1, n_customers + 1),
    'Recency': np.random.exponential(scale=30, size=n_customers).astype(int),  # 마지막 구매 후 일수
    'Frequency': np.random.poisson(lam=5, size=n_customers),  # 구매 횟수
    'Monetary': np.random.gamma(shape=2, scale=50, size=n_customers)  # 총 구매 금액
})

print("고객 데이터 샘플:")
print(data.head(10))
print(f"\n데이터 크기: {data.shape}")
print("\n기술 통계:")
print(data.describe())

# 2. 데이터 표준화 (필수!)
scaler = StandardScaler()
features = ['Recency', 'Frequency', 'Monetary']
X_scaled = scaler.fit_transform(data[features])

# 3. Elbow Method로 최적 K 찾기
wcss = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Customer Segmentation')
plt.grid(True, alpha=0.3)
plt.savefig('customer_segmentation_elbow.png', dpi=300, bbox_inches='tight')

# 4. K=4로 클러스터링
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X_scaled)

print(f"\n클러스터별 고객 수:")
print(data['Cluster'].value_counts().sort_index())

# 5. 각 클러스터의 특성 분석
print("\n클러스터별 평균값:")
cluster_summary = data.groupby('Cluster')[features].mean()
print(cluster_summary)

# 6. 클러스터에 이름 붙이기 (비즈니스 인사이트)
cluster_names = {
    0: 'Lost Customers',      # Recency 높음, Frequency/Monetary 낮음
    1: 'Loyal Customers',     # 모든 지표 우수
    2: 'Potential Loyalists', # Frequency/Monetary 중간
    3: 'New Customers'        # Recency 낮음, Frequency/Monetary 낮음
}

# 실제 데이터에 맞게 조정 (여기서는 예시)
data['Segment'] = data['Cluster'].map(cluster_names)

print("\n세그먼트별 고객 수:")
print(data['Segment'].value_counts())

# 7. 시각화
fig = plt.figure(figsize=(15, 10))

# 7-1. 3D 산점도
ax1 = fig.add_subplot(221, projection='3d')
scatter = ax1.scatter(data['Recency'], data['Frequency'], data['Monetary'],
                     c=data['Cluster'], cmap='viridis', s=50, alpha=0.6)
ax1.set_xlabel('Recency (days)')
ax1.set_ylabel('Frequency')
ax1.set_zlabel('Monetary ($)')
ax1.set_title('3D Customer Segmentation')
plt.colorbar(scatter, ax=ax1, label='Cluster')

# 7-2. Recency vs Frequency
ax2 = fig.add_subplot(222)
for cluster in range(optimal_k):
    cluster_data = data[data['Cluster'] == cluster]
    ax2.scatter(cluster_data['Recency'], cluster_data['Frequency'],
               label=f'Cluster {cluster}', s=50, alpha=0.6)
ax2.set_xlabel('Recency (days)')
ax2.set_ylabel('Frequency')
ax2.set_title('Recency vs Frequency')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 7-3. Frequency vs Monetary
ax3 = fig.add_subplot(223)
for cluster in range(optimal_k):
    cluster_data = data[data['Cluster'] == cluster]
    ax3.scatter(cluster_data['Frequency'], cluster_data['Monetary'],
               label=f'Cluster {cluster}', s=50, alpha=0.6)
ax3.set_xlabel('Frequency')
ax3.set_ylabel('Monetary ($)')
ax3.set_title('Frequency vs Monetary')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 7-4. 클러스터 프로파일 (Heatmap)
ax4 = fig.add_subplot(224)
cluster_summary_normalized = (cluster_summary - cluster_summary.min()) / (cluster_summary.max() - cluster_summary.min())
sns.heatmap(cluster_summary_normalized.T, annot=True, fmt='.2f', cmap='YlOrRd',
           cbar_kws={'label': 'Normalized Value'}, ax=ax4)
ax4.set_xlabel('Cluster')
ax4.set_ylabel('Feature')
ax4.set_title('Cluster Profile Heatmap')

plt.tight_layout()
plt.savefig('customer_segmentation_analysis.png', dpi=300, bbox_inches='tight')
print("\n고객 세그멘테이션 분석이 저장되었습니다.")

# 8. 비즈니스 액션 추천
print("\n" + "="*60)
print("비즈니스 액션 추천")
print("="*60)

for cluster in range(optimal_k):
    cluster_data = data[data['Cluster'] == cluster]
    avg_recency = cluster_data['Recency'].mean()
    avg_frequency = cluster_data['Frequency'].mean()
    avg_monetary = cluster_data['Monetary'].mean()

    print(f"\nCluster {cluster}:")
    print(f"  고객 수: {len(cluster_data)}")
    print(f"  평균 Recency: {avg_recency:.1f}일")
    print(f"  평균 Frequency: {avg_frequency:.1f}회")
    print(f"  평균 Monetary: ${avg_monetary:.2f}")

    # 간단한 추천 로직
    if avg_recency > 50 and avg_frequency < 3:
        print("  → 액션: 재활성화 캠페인 (할인 쿠폰, 이메일)")
    elif avg_frequency > 10 and avg_monetary > 200:
        print("  → 액션: VIP 프로그램, 개인화된 추천")
    elif avg_recency < 20 and avg_frequency < 5:
        print("  → 액션: 온보딩 프로그램, 제품 교육")
    else:
        print("  → 액션: 정기적인 프로모션, 크로스셀링")

"""
출력 예시:
고객 데이터 샘플:
   CustomerID  Recency  Frequency     Monetary
0           1       49          5   125.837421
1           2       34          6    89.234567
2           3       12          8   234.567890
...

데이터 크기: (500, 4)

클러스터별 고객 수:
0    128
1    142
2    115
3    115

클러스터별 평균값:
         Recency  Frequency    Monetary
Cluster
0          52.3        3.2      78.45
1          15.8       12.4     256.78
2          28.6        6.8     142.33
3          18.2        4.1      95.67

==============================================================
비즈니스 액션 추천
==============================================================

Cluster 0:
  고객 수: 128
  평균 Recency: 52.3일
  평균 Frequency: 3.2회
  평균 Monetary: $78.45
  → 액션: 재활성화 캠페인 (할인 쿠폰, 이메일)

Cluster 1:
  고객 수: 142
  평균 Recency: 15.8일
  평균 Frequency: 12.4회
  평균 Monetary: $256.78
  → 액션: VIP 프로그램, 개인화된 추천

... (생략)
"""
```

## K-means의 한계 시연

```python
"""
K-means가 잘 작동하지 않는 경우들
"""

from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. 초승달 모양 (Non-convex shapes)
X_moons, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
kmeans_moons = KMeans(n_clusters=2, random_state=42)
labels_moons = kmeans_moons.fit_predict(X_moons)

axes[0, 0].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_moons, cmap='viridis', s=30)
axes[0, 0].set_title('K-means on Moons (실패)')
axes[0, 0].text(0.5, -1.5, 'Non-convex 모양은\n K-means로 구분 불가',
               ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

# DBSCAN으로 비교
dbscan_moons = DBSCAN(eps=0.2, min_samples=5)
labels_dbscan_moons = dbscan_moons.fit_predict(X_moons)
axes[1, 0].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_dbscan_moons, cmap='viridis', s=30)
axes[1, 0].set_title('DBSCAN on Moons (성공)')

# 2. 동심원 (Concentric circles)
X_circles, _ = make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42)
kmeans_circles = KMeans(n_clusters=2, random_state=42)
labels_circles = kmeans_circles.fit_predict(X_circles)

axes[0, 1].scatter(X_circles[:, 0], X_circles[:, 1], c=labels_circles, cmap='viridis', s=30)
axes[0, 1].set_title('K-means on Circles (실패)')
axes[0, 1].text(0, -1.5, '동심원 모양은\nK-means로 구분 불가',
               ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

dbscan_circles = DBSCAN(eps=0.15, min_samples=5)
labels_dbscan_circles = dbscan_circles.fit_predict(X_circles)
axes[1, 1].scatter(X_circles[:, 0], X_circles[:, 1], c=labels_dbscan_circles, cmap='viridis', s=30)
axes[1, 1].set_title('DBSCAN on Circles (성공)')

# 3. 크기가 다른 클러스터
np.random.seed(42)
X_uneven = np.vstack([
    np.random.randn(200, 2) * 0.5 + [0, 0],     # 작고 밀집된 클러스터
    np.random.randn(50, 2) * 2.0 + [5, 5]       # 크고 흩어진 클러스터
])
kmeans_uneven = KMeans(n_clusters=2, random_state=42)
labels_uneven = kmeans_uneven.fit_predict(X_uneven)

axes[0, 2].scatter(X_uneven[:, 0], X_uneven[:, 1], c=labels_uneven, cmap='viridis', s=30)
axes[0, 2].set_title('K-means on Uneven Sizes (문제)')
axes[0, 2].text(2.5, -5, '크기가 다른 클러스터는\n부정확하게 분리',
               ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

# 실제 레이블 (참고용)
true_labels_uneven = np.array([0]*200 + [1]*50)
axes[1, 2].scatter(X_uneven[:, 0], X_uneven[:, 1], c=true_labels_uneven, cmap='viridis', s=30)
axes[1, 2].set_title('실제 클러스터')

plt.tight_layout()
plt.savefig('kmeans_limitations.png', dpi=300, bbox_inches='tight')
print("K-means 한계 시연이 'kmeans_limitations.png'로 저장되었습니다.")

print("\nK-means의 한계:")
print("1. Non-convex 모양: 초승달, S자 등 복잡한 모양은 구분 못함")
print("2. 동심원: 거리 기반이라 안쪽/바깥쪽 원 구분 못함")
print("3. 크기/밀도 차이: 모든 클러스터가 비슷한 크기와 밀도라고 가정")
print("4. 이상치 민감: 중심점이 이상치에 의해 왜곡될 수 있음")
print("\n대안: DBSCAN, Spectral Clustering, Hierarchical Clustering")
```

# References

## 공식 문서
- [Scikit-learn K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Scikit-learn Clustering Comparison](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html)

## 추천 학습 자료
- [StatQuest: K-means Clustering](https://www.youtube.com/watch?v=4b5d3muPQmA)
- [Stanford CS221: K-means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html)
- [K-means Visualization](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)

## 논문
- Lloyd, S. P. (1982). "Least squares quantization in PCM". IEEE Transactions on Information Theory.
- Arthur, D., & Vassilvitskii, S. (2007). "k-means++: The Advantages of Careful Seeding".

## 관련 알고리즘
- **K-medoids (PAM)**: 중심점 대신 실제 데이터 포인트 사용, 이상치에 강건
- **DBSCAN**: 밀도 기반, 복잡한 모양 클러스터 찾기, K 불필요
- **Hierarchical Clustering**: 계층 구조, 덴드로그램
- **Gaussian Mixture Model (GMM)**: 확률 기반, soft clustering
- **Spectral Clustering**: 그래프 기반, non-convex 모양 가능
- **Mean Shift**: 밀도 기반, K 불필요

## 실전 활용 사례
- **마케팅**: 고객 세그멘테이션, 타겟 마케팅
- **이미지 처리**: 이미지 압축, 색상 양자화
- **문서 분류**: 토픽 모델링, 뉴스 그룹화
- **추천 시스템**: 사용자/아이템 그룹화
- **이상 탐지**: 정상 패턴 학습 후 이상치 식별
- **지리 데이터**: 위치 기반 클러스터링 (상권 분석 등)
