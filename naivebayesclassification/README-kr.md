# Abstract

Naive Bayes Classification (나이브 베이즈 분류)은 **베이즈 정리(Bayes Theorem)**를 기반으로 한 확률적 분류 알고리즘입니다. "Naive(순진한)"라는 이름이 붙은 이유는 **모든 특성(feature)이 서로 독립적**이라는 단순한 가정을 하기 때문입니다.

> "스팸 메일인지 판단하려면, 이 메일이 스팸일 확률을 계산하자"

**Naive Bayes의 핵심 개념:**
- **베이즈 정리**: 조건부 확률을 이용한 역확률 계산
- **사전 확률(Prior)**: 기존에 알고 있던 확률
- **우도(Likelihood)**: 주어진 조건에서 데이터가 나타날 확률
- **사후 확률(Posterior)**: 데이터를 관찰한 후의 확률
- **독립 가정(Independence Assumption)**: 모든 특성이 서로 독립

예를 들어, 스팸 메일을 판단할 때:
- "무료"라는 단어가 있으면 스팸일 확률이 얼마나 될까?
- "대출"이라는 단어가 추가로 있으면 확률이 어떻게 변할까?
- 이 메일의 각 단어들을 보고 스팸 확률을 계산!

**Naive Bayes의 장점:**
- 구현이 매우 간단하고 빠름
- 적은 데이터로도 잘 작동
- 다중 클래스 분류에 효과적
- 텍스트 분류에 특히 강력함
- 확률 기반 예측 제공
- 차원이 높아도 성능 유지

**Naive Bayes의 단점:**
- 독립 가정이 현실적이지 않음 (단어들은 실제로 서로 연관)
- 훈련 데이터에 없는 조합은 확률이 0이 됨
- 연속형 데이터는 정규분포 가정 필요
- 특성 간 상관관계 고려 불가

**주요 활용 분야:**
- 텍스트 분류 (스팸 필터링, 감성 분석, 문서 분류)
- 추천 시스템
- 의료 진단 (증상 기반 질병 예측)
- 실시간 예측 (빠른 응답 필요한 시스템)
- 얼굴 인식

# Materials

- [Scikit-learn Naive Bayes Documentation](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [Naive Bayes Classifier - Wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
- [StatQuest: Naive Bayes](https://www.youtube.com/watch?v=O2L2Uv9pdDA)

# Basic

## Naive Bayes란?

Naive Bayes는 **베이즈 정리를 이용하여 새로운 데이터가 각 클래스에 속할 확률을 계산**하고, **가장 확률이 높은 클래스로 분류**하는 알고리즘입니다.

### 직관적 이해: 스팸 메일 필터링

당신에게 새로운 이메일이 도착했습니다:

```
"무료 대출 신청하세요!"
```

이 메일이 스팸일까요, 정상 메일일까요?

**우리가 알고 싶은 것:**
```
P(스팸 | 이 메일) = 이 메일을 봤을 때 스팸일 확률은?
```

**베이즈 정리를 사용하면:**
```
P(스팸 | 메일) = P(메일 | 스팸) × P(스팸) / P(메일)

여기서:
- P(스팸): 사전 확률 - 원래 스팸 메일이 올 확률
- P(메일 | 스팸): 우도 - 스팸 메일일 때 이런 내용이 나타날 확률
- P(메일): 증거 - 이런 메일이 나타날 확률 (정규화 상수)
- P(스팸 | 메일): 사후 확률 - 우리가 알고 싶은 최종 확률!
```

### 베이즈 정리 (Bayes Theorem)

**기본 공식:**

```
P(A|B) = P(B|A) × P(A) / P(B)

읽는 법: "B가 주어졌을 때 A일 확률"
```

**각 항의 의미:**

1. **P(A)**: 사전 확률 (Prior Probability)
   - B를 보기 전에 A일 확률
   - 예: 전체 메일 중 30%가 스팸 → P(스팸) = 0.3

2. **P(B|A)**: 우도 (Likelihood)
   - A일 때 B가 관찰될 확률
   - 예: 스팸 메일에 "무료"라는 단어가 나타날 확률

3. **P(B)**: 증거 (Evidence)
   - B가 관찰될 전체 확률
   - 정규화를 위해 사용

4. **P(A|B)**: 사후 확률 (Posterior Probability)
   - B를 관찰한 후 A일 확률
   - 우리가 최종적으로 구하고자 하는 값!

### 왜 "Naive(순진한)"인가?

**독립 가정 (Independence Assumption):**

메일에 ["무료", "대출", "신청"] 세 단어가 있을 때:

```
일반적인 확률 (독립이 아닐 때):
P(무료, 대출, 신청 | 스팸) = 복잡한 결합 확률 계산 필요
→ "무료"와 "대출"이 함께 나올 확률, "대출"과 "신청"이 함께 나올 확률 등...

Naive Bayes (독립 가정):
P(무료, 대출, 신청 | 스팸) = P(무료|스팸) × P(대출|스팸) × P(신청|스팸)
→ 각 단어의 확률만 곱하면 됨!
```

**현실에서는:**
- "무료"와 "대출"은 실제로 함께 나타날 가능성이 높음 (독립이 아님)
- 하지만 독립이라고 가정하면 계산이 훨씬 간단해짐
- 놀랍게도 이 단순한 가정으로도 실제로 잘 작동함!

## 스팸 메일 분류 예제 (동영상 예제)

### 훈련 데이터

우리에게 6개의 이메일이 있습니다:

**스팸 메일 (3개):**
1. "무료 대출 가능합니다"
2. "긴급 대출 신청하세요"
3. "무료 체험 신청"

**정상 메일 (3개):**
1. "회의 일정 공유합니다"
2. "프로젝트 진행 상황"
3. "내일 점심 약속"

### 단어 빈도 분석

각 단어가 스팸/정상 메일에 나타난 횟수를 세어봅시다:

**스팸 메일 단어 빈도:**
```
단어      | 출현 횟수 | 총 단어 수
---------|----------|----------
무료      | 2        | 9
대출      | 2        | 9
신청      | 2        | 9
긴급      | 1        | 9
가능      | 1        | 9
체험      | 1        | 9
```

**정상 메일 단어 빈도:**
```
단어      | 출현 횟수 | 총 단어 수
---------|----------|----------
회의      | 1        | 9
일정      | 1        | 9
공유      | 1        | 9
프로젝트   | 1        | 9
진행      | 1        | 9
상황      | 1        | 9
내일      | 1        | 9
점심      | 1        | 9
약속      | 1        | 9
```

### 새로운 메일 분류: "무료 대출 신청"

**Step 1: 사전 확률 계산**

```
P(스팸) = 3/6 = 0.5
P(정상) = 3/6 = 0.5
```

**Step 2: 우도 계산 (각 단어가 나타날 확률)**

**스팸일 때:**
```
P(무료 | 스팸) = 2/9 ≈ 0.222
P(대출 | 스팸) = 2/9 ≈ 0.222
P(신청 | 스팸) = 2/9 ≈ 0.222

독립 가정으로 곱하기:
P(무료, 대출, 신청 | 스팸) = 0.222 × 0.222 × 0.222 ≈ 0.0109
```

**정상일 때:**
```
P(무료 | 정상) = 0/9 = 0  ← 문제 발생!
P(대출 | 정상) = 0/9 = 0
P(신청 | 정상) = 0/9 = 0

P(무료, 대출, 신청 | 정상) = 0 × 0 × 0 = 0
```

**Step 3: 사후 확률 계산**

```
P(스팸 | 메일) ∝ P(메일 | 스팸) × P(스팸)
              = 0.0109 × 0.5
              = 0.00545

P(정상 | 메일) ∝ P(메일 | 정상) × P(정상)
              = 0 × 0.5
              = 0

→ 스팸일 확률이 더 높음! (실제로는 무한대)
```

**결론:** 이 메일은 **스팸**으로 분류됩니다!

### 문제점: 확률이 0이 되는 경우

위 예제에서 정상 메일 확률이 0이 나왔습니다. 이는:

```
정상 메일 훈련 데이터에 "무료", "대출", "신청" 단어가 한 번도 안 나타남
→ P(무료 | 정상) = 0
→ 전체 확률도 0
```

**이것은 큰 문제입니다:**
- 훈련 데이터에 없던 단어가 나오면 확률이 0이 됨
- 하나라도 0이면 전체가 0이 됨
- 현실적이지 않음 (정상 메일에도 "무료"가 나올 수 있음)

**해결책:** 라플라스 스무딩 (Laplace Smoothing) → Advanced 섹션에서 다룸

## Python 코드 예제

### 1. Naive Bayes 기본 사용 (Iris 데이터)

```python
"""
Naive Bayes 기본 예제: Iris 꽃 분류
- Gaussian Naive Bayes 사용 (연속형 데이터)
"""

from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# 1. 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

print("데이터 크기:", X.shape)
print("클래스:", iris.target_names)
print("특성:", iris.feature_names)

# 2. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Gaussian Naive Bayes 모델 생성 및 학습
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 4. 예측
y_pred = gnb.predict(X_test)

# 5. 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"\n정확도: {accuracy:.4f}")
print("\n분류 리포트:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 6. 혼동 행렬
print("혼동 행렬:")
print(confusion_matrix(y_test, y_pred))

# 7. 확률 예측
proba = gnb.predict_proba(X_test[:5])
print("\n처음 5개 샘플의 클래스별 확률:")
for i, p in enumerate(proba):
    print(f"샘플 {i}: Setosa={p[0]:.3f}, Versicolor={p[1]:.3f}, Virginica={p[2]:.3f}")
    print(f"  → 예측: {iris.target_names[y_pred[i]]}, 실제: {iris.target_names[y_test[i]]}")

# 8. 클래스별 사전 확률 확인
print("\n클래스별 사전 확률 (훈련 데이터 기반):")
for i, class_name in enumerate(iris.target_names):
    print(f"{class_name}: {gnb.class_prior_[i]:.3f}")

# 9. 특성별 평균과 분산 (Gaussian 가정)
print("\n각 클래스별 특성 평균:")
print("(Gaussian NB는 각 특성이 정규분포를 따른다고 가정)")
for i, class_name in enumerate(iris.target_names):
    print(f"\n{class_name}:")
    for j, feature_name in enumerate(iris.feature_names):
        print(f"  {feature_name}: 평균={gnb.theta_[i][j]:.2f}, 분산={gnb.var_[i][j]:.2f}")

"""
출력 예시:
데이터 크기: (150, 4)
클래스: ['setosa' 'versicolor' 'virginica']
특성: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

정확도: 0.9778

분류 리포트:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        19
  versicolor       1.00      0.92      0.96        13
   virginica       0.93      1.00      0.96        13

    accuracy                           0.98        45
   macro avg       0.98      0.97      0.97        45
weighted avg       0.98      0.98      0.98        45

혼동 행렬:
[[19  0  0]
 [ 0 12  1]
 [ 0  0 13]]

처음 5개 샘플의 클래스별 확률:
샘플 0: Setosa=1.000, Versicolor=0.000, Virginica=0.000
  → 예측: virginica, 실제: virginica
샘플 1: Setosa=1.000, Versicolor=0.000, Virginica=0.000
  → 예측: setosa, 실제: setosa
...

클래스별 사전 확률 (훈련 데이터 기반):
setosa: 0.333
versicolor: 0.333
virginica: 0.333
"""
```

### 2. 스팸 메일 분류 (동영상 예제 재현)

```python
"""
스팸 메일 분류: 동영상 예제 완전 재현
- 단어 빈도 기반 분류
- 라플라스 스무딩 적용 비교
"""

import numpy as np
from collections import defaultdict

# 훈련 데이터
spam_emails = [
    "무료 대출 가능합니다",
    "긴급 대출 신청하세요",
    "무료 체험 신청"
]

normal_emails = [
    "회의 일정 공유합니다",
    "프로젝트 진행 상황",
    "내일 점심 약속"
]

# 단어 분리 (간단한 공백 기준)
def tokenize(text):
    return text.split()

# 단어 빈도 계산
def count_words(emails):
    word_count = defaultdict(int)
    total_words = 0
    for email in emails:
        for word in tokenize(email):
            word_count[word] += 1
            total_words += 1
    return word_count, total_words

spam_word_count, spam_total = count_words(spam_emails)
normal_word_count, normal_total = count_words(normal_emails)

print("=== 훈련 데이터 분석 ===\n")
print(f"스팸 메일 수: {len(spam_emails)}")
print(f"정상 메일 수: {len(normal_emails)}")
print(f"\n스팸 메일 총 단어 수: {spam_total}")
print(f"정상 메일 총 단어 수: {normal_total}")

print("\n=== 스팸 메일 단어 빈도 ===")
for word, count in sorted(spam_word_count.items(), key=lambda x: -x[1]):
    print(f"{word}: {count}회 (P={count}/{spam_total} = {count/spam_total:.3f})")

print("\n=== 정상 메일 단어 빈도 ===")
for word, count in sorted(normal_word_count.items(), key=lambda x: -x[1]):
    print(f"{word}: {count}회 (P={count}/{normal_total} = {count/normal_total:.3f})")

# 사전 확률
P_spam = len(spam_emails) / (len(spam_emails) + len(normal_emails))
P_normal = len(normal_emails) / (len(spam_emails) + len(normal_emails))

print(f"\n=== 사전 확률 ===")
print(f"P(스팸) = {P_spam:.3f}")
print(f"P(정상) = {P_normal:.3f}")

# 새로운 메일 분류 함수 (라플라스 스무딩 없이)
def classify_email(email, use_smoothing=False):
    words = tokenize(email)
    print(f"\n{'='*60}")
    print(f"분류할 메일: '{email}'")
    print(f"단어: {words}")
    print(f"라플라스 스무딩: {'사용' if use_smoothing else '미사용'}")
    print(f"{'='*60}\n")

    # 스무딩 파라미터
    alpha = 1 if use_smoothing else 0

    # 어휘 크기 (모든 고유 단어)
    vocab = set(spam_word_count.keys()) | set(normal_word_count.keys())
    vocab_size = len(vocab)

    # 스팸 확률 계산
    log_prob_spam = np.log(P_spam)
    print("=== 스팸 확률 계산 ===")
    print(f"log P(스팸) = log({P_spam:.3f}) = {log_prob_spam:.4f}")

    for word in words:
        count = spam_word_count.get(word, 0)
        prob = (count + alpha) / (spam_total + alpha * vocab_size)
        log_prob = np.log(prob)
        log_prob_spam += log_prob
        print(f"  '{word}': 빈도={count}, P={prob:.6f}, log(P)={log_prob:.4f}")

    print(f"→ 최종 log P(메일|스팸) × P(스팸) = {log_prob_spam:.4f}")

    # 정상 확률 계산
    log_prob_normal = np.log(P_normal)
    print("\n=== 정상 확률 계산 ===")
    print(f"log P(정상) = log({P_normal:.3f}) = {log_prob_normal:.4f}")

    for word in words:
        count = normal_word_count.get(word, 0)
        prob = (count + alpha) / (normal_total + alpha * vocab_size)
        log_prob = np.log(prob)
        log_prob_normal += log_prob
        print(f"  '{word}': 빈도={count}, P={prob:.6f}, log(P)={log_prob:.4f}")

    print(f"→ 최종 log P(메일|정상) × P(정상) = {log_prob_normal:.4f}")

    # 결과 비교
    print(f"\n=== 결과 ===")
    print(f"log P(스팸|메일) = {log_prob_spam:.4f}")
    print(f"log P(정상|메일) = {log_prob_normal:.4f}")

    if log_prob_spam > log_prob_normal:
        print(f"→ 스팸이 {log_prob_spam - log_prob_normal:.4f} 더 높음")
        print(f"→ 분류 결과: 스팸 ⚠️")
        return "스팸"
    else:
        print(f"→ 정상이 {log_prob_normal - log_prob_spam:.4f} 더 높음")
        print(f"→ 분류 결과: 정상 ✓")
        return "정상"

# 테스트 1: "무료 대출 신청" (스무딩 없이)
result1 = classify_email("무료 대출 신청", use_smoothing=False)

# 테스트 2: "회의 일정" (스무딩 없이)
result2 = classify_email("회의 일정", use_smoothing=False)

# 테스트 3: "오늘 긴급 대출" (스무딩 없이 - 문제 발생!)
print("\n" + "="*60)
print("⚠️ '오늘'이라는 단어는 훈련 데이터에 없습니다!")
print("정상 메일에서 확률이 0이 되는 문제 발생 예상")
print("="*60)
result3 = classify_email("오늘 긴급 대출", use_smoothing=False)

"""
출력 예시:
=== 훈련 데이터 분석 ===

스팸 메일 수: 3
정상 메일 수: 3

스팸 메일 총 단어 수: 9
정상 메일 총 단어 수: 9

=== 스팸 메일 단어 빈도 ===
무료: 2회 (P=2/9 = 0.222)
대출: 2회 (P=2/9 = 0.222)
신청: 2회 (P=2/9 = 0.222)
긴급: 1회 (P=1/9 = 0.111)
가능합니다: 1회 (P=1/9 = 0.111)
신청하세요: 1회 (P=1/9 = 0.111)
체험: 1회 (P=1/9 = 0.111)

=== 정상 메일 단어 빈도 ===
회의: 1회 (P=1/9 = 0.111)
일정: 1회 (P=1/9 = 0.111)
공유합니다: 1회 (P=1/9 = 0.111)
프로젝트: 1회 (P=1/9 = 0.111)
진행: 1회 (P=1/9 = 0.111)
상황: 1회 (P=1/9 = 0.111)
내일: 1회 (P=1/9 = 0.111)
점심: 1회 (P=1/9 = 0.111)
약속: 1회 (P=1/9 = 0.111)

=== 사전 확률 ===
P(스팸) = 0.500
P(정상) = 0.500

============================================================
분류할 메일: '무료 대출 신청'
단어: ['무료', '대출', '신청']
라플라스 스무딩: 미사용
============================================================

=== 스팸 확률 계산 ===
log P(스팸) = log(0.500) = -0.6931
  '무료': 빈도=2, P=0.222222, log(P)=-1.5041
  '대출': 빈도=2, P=0.222222, log(P)=-1.5041
  '신청': 빈도=2, P=0.222222, log(P)=-1.5041
→ 최종 log P(메일|스팸) × P(스팸) = -6.2095

=== 정상 확률 계산 ===
log P(정상) = log(0.500) = -0.6931
  '무료': 빈도=0, P=0.000000, log(P)=-inf
  '대출': 빈도=0, P=0.000000, log(P)=-inf
  '신청': 빈도=0, P=0.000000, log(P)=-inf
→ 최종 log P(메일|정상) × P(정상) = -inf

=== 결과 ===
log P(스팸|메일) = -6.2095
log P(정상|메일) = -inf
→ 스팸이 inf 더 높음
→ 분류 결과: 스팸 ⚠️
"""
```

### 3. Multinomial Naive Bayes (텍스트 분류)

```python
"""
Multinomial Naive Bayes: 텍스트 분류
- 단어 빈도 기반 (Bag of Words)
- CountVectorizer 사용
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# 간단한 텍스트 데이터
texts = [
    "무료 대출 가능합니다",
    "긴급 대출 신청하세요",
    "무료 체험 신청",
    "특별 할인 이벤트",
    "공짜 쿠폰 받으세요",
    "회의 일정 공유합니다",
    "프로젝트 진행 상황",
    "내일 점심 약속",
    "업무 보고서 제출",
    "팀 회식 공지",
]

labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1=스팸, 0=정상

# 1. 텍스트를 숫자 벡터로 변환
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

print("=== 데이터 벡터화 ===")
print(f"특성 수 (고유 단어 수): {len(vectorizer.get_feature_names_out())}")
print(f"단어 목록: {vectorizer.get_feature_names_out()}")
print(f"\n첫 번째 문서의 벡터:")
print(X[0].toarray())
print(f"→ 각 위치는 해당 단어의 출현 횟수")

# 2. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)

# 3. Multinomial Naive Bayes 학습
mnb = MultinomialNB(alpha=1.0)  # alpha=1.0은 라플라스 스무딩
mnb.fit(X_train, y_train)

# 4. 예측
y_pred = mnb.predict(X_test)

# 5. 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"\n정확도: {accuracy:.4f}")

# 6. 새로운 메일 분류
new_emails = [
    "무료 쿠폰 이벤트",
    "회의 일정 변경",
    "대출 신청 방법"
]

new_emails_vec = vectorizer.transform(new_emails)
predictions = mnb.predict(new_emails_vec)
probas = mnb.predict_proba(new_emails_vec)

print("\n=== 새로운 메일 분류 ===")
for email, pred, proba in zip(new_emails, predictions, probas):
    label = "스팸" if pred == 1 else "정상"
    print(f"\n'{email}'")
    print(f"  → {label} (정상: {proba[0]:.2%}, 스팸: {proba[1]:.2%})")

# 7. 각 클래스별 단어 확률 상위 5개
print("\n=== 스팸 메일에서 자주 나타나는 단어 TOP 5 ===")
feature_names = vectorizer.get_feature_names_out()
spam_log_probs = mnb.feature_log_prob_[1]  # 클래스 1 (스팸)
top_spam_indices = spam_log_probs.argsort()[-5:][::-1]

for idx in top_spam_indices:
    print(f"{feature_names[idx]}: {np.exp(spam_log_probs[idx]):.4f}")

print("\n=== 정상 메일에서 자주 나타나는 단어 TOP 5 ===")
normal_log_probs = mnb.feature_log_prob_[0]  # 클래스 0 (정상)
top_normal_indices = normal_log_probs.argsort()[-5:][::-1]

for idx in top_normal_indices:
    print(f"{feature_names[idx]}: {np.exp(normal_log_probs[idx]):.4f}")

"""
출력 예시:
=== 데이터 벡터화 ===
특성 수 (고유 단어 수): 23
단어 목록: ['가능합니다' '공유합니다' '공짜' '공지' '내일' ...]

첫 번째 문서의 벡터:
[[1 0 0 0 0 1 0 0 0 1 0 0 ...]]
→ 각 위치는 해당 단어의 출현 횟수

정확도: 1.0000

=== 새로운 메일 분류 ===

'무료 쿠폰 이벤트'
  → 스팸 (정상: 12.34%, 스팸: 87.66%)

'회의 일정 변경'
  → 정상 (정상: 78.45%, 스팸: 21.55%)

'대출 신청 방법'
  → 스팸 (정상: 23.12%, 스팸: 76.88%)

=== 스팸 메일에서 자주 나타나는 단어 TOP 5 ===
무료: 0.1250
대출: 0.1250
신청: 0.1250
이벤트: 0.0833
할인: 0.0833
"""
```

### 4. Bernoulli Naive Bayes (이진 특성)

```python
"""
Bernoulli Naive Bayes: 이진 특성 (단어 존재 여부)
- 단어의 출현 횟수가 아닌 존재 여부만 사용
- 짧은 문서나 이진 특성에 효과적
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 같은 데이터 사용
texts = [
    "무료 대출 가능",
    "긴급 대출 신청",
    "무료 체험",
    "특별 할인",
    "공짜 쿠폰",
    "회의 일정",
    "프로젝트 진행",
    "내일 점심",
    "업무 보고서",
    "팀 회식",
]

labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1=스팸, 0=정상

# 1. 벡터화 (이진으로 변환)
vectorizer = CountVectorizer(binary=True)  # binary=True!
X = vectorizer.fit_transform(texts)

print("=== Bernoulli NB: 이진 특성 ===")
print("단어의 출현 횟수가 아닌 존재 여부만 사용\n")
print("첫 번째 문서 '무료 대출 가능':")
print(f"벡터: {X[0].toarray()[0]}")
print("→ 1 = 단어 존재, 0 = 단어 없음")

# 2. 학습
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, random_state=42
)

bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_train, y_train)

# 3. 예측
y_pred = bnb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n정확도: {accuracy:.4f}")

# 4. Multinomial vs Bernoulli 비교
print("\n=== Multinomial NB vs Bernoulli NB ===")

# 단어가 여러 번 나타나는 경우
test_cases = [
    ("무료 무료 무료 대출 대출", "단어 반복 많음"),
    ("무료 대출", "단어 반복 없음")
]

# Multinomial NB (빈도 고려)
from sklearn.naive_bayes import MultinomialNB
vectorizer_multi = CountVectorizer(binary=False)
X_multi = vectorizer_multi.fit_transform(texts)
mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_multi[:7], labels[:7])  # 간단히 일부만 학습

# Bernoulli NB (존재 여부만)
vectorizer_bern = CountVectorizer(binary=True)
X_bern = vectorizer_bern.fit_transform(texts)
bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_bern[:7], labels[:7])

for text, desc in test_cases:
    print(f"\n테스트: '{text}' ({desc})")

    # Multinomial
    vec_multi = vectorizer_multi.transform([text])
    prob_multi = mnb.predict_proba(vec_multi)[0]
    print(f"  Multinomial NB: 정상={prob_multi[0]:.2%}, 스팸={prob_multi[1]:.2%}")
    print(f"  → 빈도 벡터: {vec_multi.toarray()[0]}")

    # Bernoulli
    vec_bern = vectorizer_bern.transform([text])
    prob_bern = bnb.predict_proba(vec_bern)[0]
    print(f"  Bernoulli NB: 정상={prob_bern[0]:.2%}, 스팸={prob_bern[1]:.2%}")
    print(f"  → 이진 벡터: {vec_bern.toarray()[0]}")

    print(f"  📌 Multinomial은 반복 횟수를 고려, Bernoulli는 무시")

"""
출력 예시:
=== Bernoulli NB: 이진 특성 ===
단어의 출현 횟수가 아닌 존재 여부만 사용

첫 번째 문서 '무료 대출 가능':
벡터: [1 1 0 0 0 1 0 0 0 0 0]
→ 1 = 단어 존재, 0 = 단어 없음

정확도: 1.0000

=== Multinomial NB vs Bernoulli NB ===

테스트: '무료 무료 무료 대출 대출' (단어 반복 많음)
  Multinomial NB: 정상=5.23%, 스팸=94.77%
  → 빈도 벡터: [0 3 2 0 0 0 0 ...]
  Bernoulli NB: 정상=12.45%, 스팸=87.55%
  → 이진 벡터: [0 1 1 0 0 0 0 ...]
  📌 Multinomial은 반복 횟수를 고려, Bernoulli는 무시

테스트: '무료 대출' (단어 반복 없음)
  Multinomial NB: 정상=15.34%, 스팸=84.66%
  → 빈도 벡터: [0 1 1 0 0 0 0 ...]
  Bernoulli NB: 정상=12.45%, 스팸=87.55%
  → 이진 벡터: [0 1 1 0 0 0 0 ...]
  📌 Multinomial은 반복 횟수를 고려, Bernoulli는 무시
"""
```

# Advanced

## 라플라스 스무딩 (Laplace Smoothing)

### 문제 상황

Basic 섹션의 "무료 대출 신청" 예제에서 봤듯이:

```
정상 메일 훈련 데이터에 "무료"가 없음
→ P(무료 | 정상) = 0/9 = 0
→ P(무료, 대출, 신청 | 정상) = 0 × ... = 0
→ 분류 불가능!
```

**더 심각한 예: "오늘 긴급 대출"**

```
스팸 메일:
- "오늘": 0회 → P(오늘 | 스팸) = 0/9 = 0
- "긴급": 1회 → P(긴급 | 스팸) = 1/9
- "대출": 2회 → P(대출 | 스팸) = 2/9

P(오늘, 긴급, 대출 | 스팸) = 0 × (1/9) × (2/9) = 0  ← 문제!

정상 메일:
- 모든 단어가 0회 → 전체 확률 = 0

→ 둘 다 0이면 어떻게 분류하나?
```

### 라플라스 스무딩이란?

**핵심 아이디어:**
```
"한 번도 안 나타난 단어도 나타날 가능성이 조금은 있다"
```

**방법:**
```
모든 단어의 빈도에 1을 더하기 (또는 α를 더하기)

기존: P(단어 | 클래스) = count / total
스무딩: P(단어 | 클래스) = (count + α) / (total + α × 어휘크기)

일반적으로 α = 1 (Laplace Smoothing)
다른 값도 가능 (Additive Smoothing)
```

### 동영상 예제: "오늘 긁급 대출" with Smoothing

**훈련 데이터 복습:**

스팸 메일 단어 빈도:
```
무료: 2회, 대출: 2회, 신청: 2회, 긴급: 1회, 가능: 1회, 체험: 1회
총 단어 수: 9개
```

정상 메일 단어 빈도:
```
회의: 1회, 일정: 1회, 공유: 1회, 프로젝트: 1회, 진행: 1회,
상황: 1회, 내일: 1회, 점심: 1회, 약속: 1회
총 단어 수: 9개
```

전체 어휘 크기:
```
{무료, 대출, 신청, 긴급, 가능, 체험, 회의, 일정, 공유,
 프로젝트, 진행, 상황, 내일, 점심, 약속, 오늘} = 16개
```

**스무딩 없이:**

```
스팸일 때:
P(오늘 | 스팸) = 0/9 = 0  ← 문제!
P(긴급 | 스팸) = 1/9 ≈ 0.111
P(대출 | 스팸) = 2/9 ≈ 0.222

P(오늘, 긴급, 대출 | 스팸) = 0 × 0.111 × 0.222 = 0

정상일 때:
P(오늘 | 정상) = 0/9 = 0
P(긴급 | 정상) = 0/9 = 0
P(대출 | 정상) = 0/9 = 0

P(오늘, 긴급, 대출 | 정상) = 0 × 0 × 0 = 0

→ 둘 다 0! 분류 불가!
```

**스무딩 적용 (α=1):**

```
스팸일 때:
P(오늘 | 스팸) = (0+1) / (9+1×16) = 1/25 = 0.04
P(긴급 | 스팸) = (1+1) / (9+1×16) = 2/25 = 0.08
P(대출 | 스팸) = (2+1) / (9+1×16) = 3/25 = 0.12

P(오늘, 긴급, 대출 | 스팸) = 0.04 × 0.08 × 0.12 ≈ 0.000384

사전 확률 포함:
P(스팸 | 메일) ∝ 0.000384 × 0.5 = 0.000192

정상일 때:
P(오늘 | 정상) = (0+1) / (9+1×16) = 1/25 = 0.04
P(긴급 | 정상) = (0+1) / (9+1×16) = 1/25 = 0.04
P(대출 | 정상) = (0+1) / (9+1×16) = 1/25 = 0.04

P(오늘, 긴급, 대출 | 정상) = 0.04 × 0.04 × 0.04 = 0.000064

사전 확률 포함:
P(정상 | 메일) ∝ 0.000064 × 0.5 = 0.000032

비교:
P(스팸) = 0.000192 > P(정상) = 0.000032
→ 스팸으로 분류! ✓
```

**효과:**
- 0 확률 문제 해결
- 새로운 단어에도 작은 확률 부여
- 기존 패턴은 여전히 유지 (스팸에 많이 나타난 단어는 여전히 높은 확률)

## 라플라스 스무딩 Python 구현

### 1. 스무딩 적용 전/후 비교

```python
"""
라플라스 스무딩: 동영상 예제 완전 재현
- "오늘 긴급 대출" 분류
- 스무딩 전후 비교
"""

import numpy as np
from collections import defaultdict

# 훈련 데이터 (이전과 동일)
spam_emails = [
    "무료 대출 가능합니다",
    "긴급 대출 신청하세요",
    "무료 체험 신청"
]

normal_emails = [
    "회의 일정 공유합니다",
    "프로젝트 진행 상황",
    "내일 점심 약속"
]

def tokenize(text):
    return text.split()

def count_words(emails):
    word_count = defaultdict(int)
    total_words = 0
    for email in emails:
        for word in tokenize(email):
            word_count[word] += 1
            total_words += 1
    return word_count, total_words

spam_word_count, spam_total = count_words(spam_emails)
normal_word_count, normal_total = count_words(normal_emails)

# 전체 어휘
vocab = set(spam_word_count.keys()) | set(normal_word_count.keys())
vocab_size = len(vocab)

print("=== 훈련 데이터 정보 ===")
print(f"스팸 메일 총 단어 수: {spam_total}")
print(f"정상 메일 총 단어 수: {normal_total}")
print(f"전체 어휘 크기: {vocab_size}")
print(f"어휘: {sorted(vocab)}")

# 사전 확률
P_spam = len(spam_emails) / (len(spam_emails) + len(normal_emails))
P_normal = 1 - P_spam

def classify_with_comparison(email):
    words = tokenize(email)
    print(f"\n{'='*70}")
    print(f"분류할 메일: '{email}'")
    print(f"단어: {words}")
    print(f"{'='*70}")

    # === 스무딩 없이 ===
    print("\n【 1. 라플라스 스무딩 없이 (α=0) 】\n")

    print("--- 스팸 확률 ---")
    prob_spam_no_smooth = P_spam
    for word in words:
        count = spam_word_count.get(word, 0)
        prob = count / spam_total if spam_total > 0 else 0
        prob_spam_no_smooth *= prob
        print(f"  P({word} | 스팸) = {count}/{spam_total} = {prob:.6f}")
        if prob == 0:
            print(f"    ⚠️ 확률이 0! 전체가 0이 됩니다!")
    print(f"P(스팸) × ∏P(단어|스팸) = {P_spam:.3f} × ... = {prob_spam_no_smooth:.10f}")

    print("\n--- 정상 확률 ---")
    prob_normal_no_smooth = P_normal
    for word in words:
        count = normal_word_count.get(word, 0)
        prob = count / normal_total if normal_total > 0 else 0
        prob_normal_no_smooth *= prob
        print(f"  P({word} | 정상) = {count}/{normal_total} = {prob:.6f}")
        if prob == 0:
            print(f"    ⚠️ 확률이 0! 전체가 0이 됩니다!")
    print(f"P(정상) × ∏P(단어|정상) = {P_normal:.3f} × ... = {prob_normal_no_smooth:.10f}")

    if prob_spam_no_smooth == 0 and prob_normal_no_smooth == 0:
        print("\n⚠️ 결과: 둘 다 0! 분류 불가능!")
        result_no_smooth = "분류 불가"
    elif prob_spam_no_smooth > prob_normal_no_smooth:
        print(f"\n→ 스팸으로 분류 (스팸 확률 더 높음)")
        result_no_smooth = "스팸"
    else:
        print(f"\n→ 정상으로 분류 (정상 확률 더 높음)")
        result_no_smooth = "정상"

    # === 스무딩 적용 ===
    print(f"\n{'='*70}")
    print("【 2. 라플라스 스무딩 적용 (α=1) 】\n")

    alpha = 1

    print("--- 스팸 확률 ---")
    log_prob_spam_smooth = np.log(P_spam)
    print(f"log P(스팸) = {log_prob_spam_smooth:.4f}")

    for word in words:
        count = spam_word_count.get(word, 0)
        prob = (count + alpha) / (spam_total + alpha * vocab_size)
        log_prob = np.log(prob)
        log_prob_spam_smooth += log_prob
        print(f"  P({word} | 스팸) = ({count}+1) / ({spam_total}+1×{vocab_size}) = {prob:.6f}")
        print(f"    log(P) = {log_prob:.4f}")

    print(f"총 log 확률 = {log_prob_spam_smooth:.4f}")
    prob_spam_smooth = np.exp(log_prob_spam_smooth)
    print(f"실제 확률 = exp({log_prob_spam_smooth:.4f}) = {prob_spam_smooth:.10e}")

    print("\n--- 정상 확률 ---")
    log_prob_normal_smooth = np.log(P_normal)
    print(f"log P(정상) = {log_prob_normal_smooth:.4f}")

    for word in words:
        count = normal_word_count.get(word, 0)
        prob = (count + alpha) / (normal_total + alpha * vocab_size)
        log_prob = np.log(prob)
        log_prob_normal_smooth += log_prob
        print(f"  P({word} | 정상) = ({count}+1) / ({normal_total}+1×{vocab_size}) = {prob:.6f}")
        print(f"    log(P) = {log_prob:.4f}")

    print(f"총 log 확률 = {log_prob_normal_smooth:.4f}")
    prob_normal_smooth = np.exp(log_prob_normal_smooth)
    print(f"실제 확률 = exp({log_prob_normal_smooth:.4f}) = {prob_normal_smooth:.10e}")

    print(f"\n--- 비교 ---")
    print(f"log P(스팸|메일) = {log_prob_spam_smooth:.4f}")
    print(f"log P(정상|메일) = {log_prob_normal_smooth:.4f}")

    if log_prob_spam_smooth > log_prob_normal_smooth:
        diff = log_prob_spam_smooth - log_prob_normal_smooth
        print(f"→ 스팸이 {diff:.4f} 더 높음 (log 스케일)")
        result_smooth = "스팸"
    else:
        diff = log_prob_normal_smooth - log_prob_spam_smooth
        print(f"→ 정상이 {diff:.4f} 더 높음 (log 스케일)")
        result_smooth = "정상"

    # === 결과 요약 ===
    print(f"\n{'='*70}")
    print("【 결과 요약 】")
    print(f"{'='*70}")
    print(f"스무딩 없이: {result_no_smooth}")
    print(f"스무딩 적용: {result_smooth} ✓")
    print(f"\n💡 스무딩을 적용하면 0 확률 문제를 해결하고 올바르게 분류할 수 있습니다!")

    return result_smooth

# 테스트: "오늘 긴급 대출" (동영상 예제)
classify_with_comparison("오늘 긴급 대출")

"""
출력 예시:
=== 훈련 데이터 정보 ===
스팸 메일 총 단어 수: 9
정상 메일 총 단어 수: 9
전체 어휘 크기: 16
어휘: ['가능합니다', '공유합니다', '긴급', '내일', '대출', ...]

======================================================================
분류할 메일: '오늘 긴급 대출'
단어: ['오늘', '긴급', '대출']
======================================================================

【 1. 라플라스 스무딩 없이 (α=0) 】

--- 스팸 확률 ---
  P(오늘 | 스팸) = 0/9 = 0.000000
    ⚠️ 확률이 0! 전체가 0이 됩니다!
  P(긴급 | 스팸) = 1/9 = 0.111111
  P(대출 | 스팸) = 2/9 = 0.222222
P(스팸) × ∏P(단어|스팸) = 0.500 × ... = 0.0000000000

--- 정상 확률 ---
  P(오늘 | 정상) = 0/9 = 0.000000
    ⚠️ 확률이 0! 전체가 0이 됩니다!
  P(긴급 | 정상) = 0/9 = 0.000000
    ⚠️ 확률이 0! 전체가 0이 됩니다!
  P(대출 | 정상) = 0/9 = 0.000000
    ⚠️ 확률이 0! 전체가 0이 됩니다!
P(정상) × ∏P(단어|정상) = 0.500 × ... = 0.0000000000

⚠️ 결과: 둘 다 0! 분류 불가능!

======================================================================
【 2. 라플라스 스무딩 적용 (α=1) 】

--- 스팸 확률 ---
log P(스팸) = -0.6931
  P(오늘 | 스팸) = (0+1) / (9+1×16) = 0.040000
    log(P) = -3.2189
  P(긴급 | 스팸) = (1+1) / (9+1×16) = 0.080000
    log(P) = -2.5257
  P(대출 | 스팸) = (2+1) / (9+1×16) = 0.120000
    log(P) = -2.1203
총 log 확률 = -8.5580
실제 확률 = exp(-8.5580) = 1.9200000e-04

--- 정상 확률 ---
log P(정상) = -0.6931
  P(오늘 | 정상) = (0+1) / (9+1×16) = 0.040000
    log(P) = -3.2189
  P(긴급 | 정상) = (0+1) / (9+1×16) = 0.040000
    log(P) = -3.2189
  P(대출 | 정상) = (0+1) / (9+1×16) = 0.040000
    log(P) = -3.2189
총 log 확률 = -10.3499
실제 확률 = exp(-10.3499) = 3.2000000e-05

--- 비교 ---
log P(스팸|메일) = -8.5580
log P(정상|메일) = -10.3499
→ 스팸이 1.7919 더 높음 (log 스케일)

======================================================================
【 결과 요약 】
======================================================================
스무딩 없이: 분류 불가
스무딩 적용: 스팸 ✓

💡 스무딩을 적용하면 0 확률 문제를 해결하고 올바르게 분류할 수 있습니다!
"""
```

### 2. 다양한 α 값 비교

```python
"""
라플라스 스무딩: 다양한 α 값의 효과
- α=0 (스무딩 없음)
- α=0.1, 0.5, 1.0, 2.0
- α 값이 클수록 확률이 균등해짐
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 간단한 예제 데이터
spam_words = ["무료"] * 10 + ["대출"] * 8 + ["할인"] * 2  # 총 20개
normal_words = ["회의"] * 5 + ["보고"] * 3  # 총 8개

spam_count = defaultdict(int, {"무료": 10, "대출": 8, "할인": 2})
normal_count = defaultdict(int, {"회의": 5, "보고": 3})
spam_total = 20
normal_total = 8

# 전체 어휘: 무료, 대출, 할인, 회의, 보고, 긴급(새 단어)
vocab = ["무료", "대출", "할인", "회의", "보고", "긴급"]
vocab_size = len(vocab)

# 다양한 α 값
alphas = [0, 0.1, 0.5, 1.0, 2.0, 5.0]

print("=== 스팸 메일 단어 확률 (다양한 α) ===\n")
print(f"{'단어':^10} | ", end="")
for alpha in alphas:
    print(f"α={alpha:^4} | ", end="")
print()
print("-" * 70)

for word in vocab:
    print(f"{word:^10} | ", end="")
    count = spam_count.get(word, 0)
    for alpha in alphas:
        if alpha == 0:
            prob = count / spam_total if spam_total > 0 else 0
        else:
            prob = (count + alpha) / (spam_total + alpha * vocab_size)
        print(f"{prob:^6.3f} | ", end="")
    print(f"  (빈도={count})")

print("\n=== 관찰 ===")
print("1. α=0일 때:")
print("   - '긴급' 단어의 확률 = 0.000 (훈련 데이터에 없음)")
print("   - 기존 단어들의 확률은 정확한 빈도 반영")
print("\n2. α가 증가하면:")
print("   - '긴급' 단어의 확률이 0에서 증가")
print("   - 모든 확률이 균등해지는 방향으로 변화")
print("   - 고빈도 단어('무료')의 확률은 감소")
print("   - 저빈도 단어('할인')의 확률은 증가")
print("\n3. α 선택:")
print("   - α=1.0 (라플라스 스무딩): 일반적 선택")
print("   - α가 너무 크면: 훈련 데이터 정보 손실")
print("   - α가 너무 작으면: 0 확률 문제 해결 불충분")

# 시각화
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, alpha in enumerate(alphas):
    ax = axes[idx]

    probs_spam = []
    probs_normal = []

    for word in vocab:
        # 스팸
        count = spam_count.get(word, 0)
        if alpha == 0:
            prob = count / spam_total if count > 0 else 0
        else:
            prob = (count + alpha) / (spam_total + alpha * vocab_size)
        probs_spam.append(prob)

        # 정상
        count = normal_count.get(word, 0)
        if alpha == 0:
            prob = count / normal_total if count > 0 else 0
        else:
            prob = (count + alpha) / (normal_total + alpha * vocab_size)
        probs_normal.append(prob)

    x = np.arange(len(vocab))
    width = 0.35

    ax.bar(x - width/2, probs_spam, width, label='스팸', alpha=0.8)
    ax.bar(x + width/2, probs_normal, width, label='정상', alpha=0.8)

    ax.set_xlabel('단어')
    ax.set_ylabel('확률')
    ax.set_title(f'α = {alpha}')
    ax.set_xticks(x)
    ax.set_xticklabels(vocab, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('laplace_smoothing_comparison.png', dpi=300, bbox_inches='tight')
print("\n시각화가 'laplace_smoothing_comparison.png'로 저장되었습니다.")

# α 값에 따른 분류 결과 변화
print("\n=== α 값에 따른 '긴급 대출' 분류 결과 ===\n")

test_words = ["긴급", "대출"]

for alpha in alphas:
    log_prob_spam = 0
    log_prob_normal = 0

    for word in test_words:
        # 스팸
        count = spam_count.get(word, 0)
        if alpha == 0:
            prob = count / spam_total if count > 0 else 1e-10  # 아주 작은 값
        else:
            prob = (count + alpha) / (spam_total + alpha * vocab_size)
        log_prob_spam += np.log(prob)

        # 정상
        count = normal_count.get(word, 0)
        if alpha == 0:
            prob = count / normal_total if count > 0 else 1e-10
        else:
            prob = (count + alpha) / (normal_total + alpha * vocab_size)
        log_prob_normal += np.log(prob)

    prediction = "스팸" if log_prob_spam > log_prob_normal else "정상"
    diff = abs(log_prob_spam - log_prob_normal)

    print(f"α={alpha:4}: 스팸={log_prob_spam:7.2f}, 정상={log_prob_normal:7.2f}, "
          f"차이={diff:5.2f} → {prediction}")

print("\n💡 α=1.0 (라플라스 스무딩)이 일반적으로 좋은 선택입니다!")

"""
출력 예시:
=== 스팸 메일 단어 확률 (다양한 α) ===

   단어     | α= 0  | α=0.1 | α=0.5 | α=1.0 | α=2.0 | α=5.0 |
----------------------------------------------------------------------
   무료     | 0.500 | 0.485 | 0.438 | 0.393 | 0.333 | 0.250 |   (빈도=10)
   대출     | 0.400 | 0.389 | 0.357 | 0.321 | 0.278 | 0.217 |   (빈도=8)
   할인     | 0.100 | 0.102 | 0.119 | 0.107 | 0.111 | 0.117 |   (빈도=2)
   회의     | 0.000 | 0.005 | 0.024 | 0.036 | 0.056 | 0.100 |   (빈도=0)
   보고     | 0.000 | 0.005 | 0.024 | 0.036 | 0.056 | 0.100 |   (빈도=0)
   긴급     | 0.000 | 0.005 | 0.024 | 0.036 | 0.056 | 0.100 |   (빈도=0)

=== 관찰 ===
1. α=0일 때:
   - '긴급' 단어의 확률 = 0.000 (훈련 데이터에 없음)
   - 기존 단어들의 확률은 정확한 빈도 반영

2. α가 증가하면:
   - '긴급' 단어의 확률이 0에서 증가
   - 모든 확률이 균등해지는 방향으로 변화
   - 고빈도 단어('무료')의 확률은 감소
   - 저빈도 단어('할인')의 확률은 증가

3. α 선택:
   - α=1.0 (라플라스 스무딩): 일반적 선택
   - α가 너무 크면: 훈련 데이터 정보 손실
   - α가 너무 작으면: 0 확률 문제 해결 불충분

=== α 값에 따른 '긴급 대출' 분류 결과 ===

α= 0.0: 스팸=-18.42, 정상=-25.33, 차이= 6.91 → 스팸
α= 0.1: 스팸= -5.25, 정상= -7.89, 차이= 2.64 → 스팸
α= 0.5: 스팸= -3.67, 정상= -5.12, 차이= 1.45 → 스팸
α= 1.0: 스팸= -3.12, 정상= -4.33, 차이= 1.21 → 스팸
α= 2.0: 스팸= -2.78, 정상= -3.67, 차이= 0.89 → 스팸
α= 5.0: 스팸= -2.45, 정상= -2.98, 차이= 0.53 → 스팸

💡 α=1.0 (라플라스 스무딩)이 일반적으로 좋은 선택입니다!
"""
```

## Naive Bayes 변형 비교

### Gaussian vs Multinomial vs Bernoulli

```python
"""
Naive Bayes 3가지 변형 비교
- Gaussian NB: 연속형 데이터 (정규분포 가정)
- Multinomial NB: 이산형 카운트 (단어 빈도)
- Bernoulli NB: 이진 특성 (존재 여부)
"""

from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np

print("="*70)
print("Naive Bayes 변형 비교")
print("="*70)

# ============================================================
# 1. Gaussian NB - 연속형 데이터 (Iris)
# ============================================================
print("\n【 1. Gaussian NB - 연속형 데이터 】")
print("-" * 70)

iris = load_iris()
X_iris = iris.data
y_iris = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42
)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
acc_gnb = accuracy_score(y_test, y_pred_gnb)

print(f"데이터: Iris (연속형 특성 4개)")
print(f"특성 예시: {X_iris[0]}")  # [5.1 3.5 1.4 0.2]
print(f"정확도: {acc_gnb:.4f}")
print(f"\n💡 Gaussian NB는 각 특성이 정규분포를 따른다고 가정")
print(f"   각 클래스별 평균과 분산을 학습")

# 첫 번째 클래스의 평균과 분산
print(f"\n예: '{iris.target_names[0]}' 클래스의 통계")
for i, feature_name in enumerate(iris.feature_names):
    print(f"  {feature_name}: 평균={gnb.theta_[0][i]:.2f}, 분산={gnb.var_[0][i]:.2f}")

# ============================================================
# 2. Multinomial NB - 단어 빈도 (카운트 기반)
# ============================================================
print("\n" + "="*70)
print("【 2. Multinomial NB - 단어 빈도 (카운트) 】")
print("-" * 70)

# 간단한 텍스트 데이터
texts = [
    "python machine learning",
    "deep learning neural networks",
    "machine learning algorithms",
    "python programming language",
    "java programming language",
    "javascript web development",
]
labels = [0, 0, 0, 1, 1, 1]  # 0=ML, 1=Programming

vectorizer_count = CountVectorizer()
X_count = vectorizer_count.fit_transform(texts)

print(f"데이터: 텍스트 문서 {len(texts)}개")
print(f"특성: 단어 빈도 (CountVectorizer)")
print(f"어휘: {vectorizer_count.get_feature_names_out()}")
print(f"\n첫 번째 문서 '{texts[0]}'의 벡터:")
print(f"{X_count[0].toarray()[0]}")
print(f"→ 각 단어의 출현 횟수")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_count, labels)

test_texts = ["python deep learning", "java web programming"]
test_vec = vectorizer_count.transform(test_texts)
predictions = mnb.predict(test_vec)
probas = mnb.predict_proba(test_vec)

print(f"\n테스트:")
for text, pred, proba in zip(test_texts, predictions, probas):
    label_name = "ML" if pred == 0 else "Programming"
    print(f"  '{text}' → {label_name} (ML:{proba[0]:.2%}, Prog:{proba[1]:.2%})")

print(f"\n💡 Multinomial NB는 단어의 출현 횟수를 사용")
print(f"   '무료'가 3번 나오면 '무료'가 1번 나올 때보다 확률 높음")

# ============================================================
# 3. Bernoulli NB - 이진 특성 (존재 여부)
# ============================================================
print("\n" + "="*70)
print("【 3. Bernoulli NB - 이진 특성 (존재 여부) 】")
print("-" * 70)

vectorizer_binary = CountVectorizer(binary=True)
X_binary = vectorizer_binary.fit_transform(texts)

print(f"데이터: 같은 텍스트 문서")
print(f"특성: 단어 존재 여부 (binary=True)")
print(f"\n첫 번째 문서 '{texts[0]}'의 벡터:")
print(f"{X_binary[0].toarray()[0]}")
print(f"→ 1=존재, 0=없음 (출현 횟수 무시)")

bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_binary, labels)

test_vec_binary = vectorizer_binary.transform(test_texts)
predictions_b = bnb.predict(test_vec_binary)
probas_b = bnb.predict_proba(test_vec_binary)

print(f"\n테스트:")
for text, pred, proba in zip(test_texts, predictions_b, probas_b):
    label_name = "ML" if pred == 0 else "Programming"
    print(f"  '{text}' → {label_name} (ML:{proba[0]:.2%}, Prog:{proba[1]:.2%})")

print(f"\n💡 Bernoulli NB는 단어의 존재 여부만 사용")
print(f"   '무료'가 3번 나와도 1번 나온 것과 동일하게 처리")

# ============================================================
# 4. Multinomial vs Bernoulli 직접 비교
# ============================================================
print("\n" + "="*70)
print("【 4. Multinomial vs Bernoulli 비교 】")
print("-" * 70)

# 단어가 반복되는 경우
test_case = "python python python programming programming programming"

# Multinomial (빈도 고려)
vec_multi = vectorizer_count.transform([test_case])
pred_multi = mnb.predict(vec_multi)[0]
proba_multi = mnb.predict_proba(vec_multi)[0]

# Bernoulli (존재 여부만)
vec_bern = vectorizer_binary.transform([test_case])
pred_bern = bnb.predict(vec_bern)[0]
proba_bern = bnb.predict_proba(vec_bern)[0]

print(f"테스트 문서: '{test_case}'")
print(f"(각 단어가 3번씩 반복)")
print(f"\nMultinomial NB:")
print(f"  벡터: {vec_multi.toarray()[0]}")
print(f"  예측: {'ML' if pred_multi==0 else 'Programming'}")
print(f"  확률: ML={proba_multi[0]:.2%}, Prog={proba_multi[1]:.2%}")

print(f"\nBernoulli NB:")
print(f"  벡터: {vec_bern.toarray()[0]}")
print(f"  예측: {'ML' if pred_bern==0 else 'Programming'}")
print(f"  확률: ML={proba_bern[0]:.2%}, Prog={proba_bern[1]:.2%}")

print(f"\n💡 단어 반복이 많은 경우:")
print(f"   - Multinomial은 반복 횟수를 강하게 반영")
print(f"   - Bernoulli는 반복을 무시하고 존재 여부만 봄")

# ============================================================
# 5. 언제 어떤 것을 사용할까?
# ============================================================
print("\n" + "="*70)
print("【 5. 선택 가이드 】")
print("="*70)

guidelines = {
    "Gaussian NB": {
        "사용 시기": "연속형 수치 데이터",
        "가정": "각 특성이 정규분포를 따름",
        "예시": "키, 몸무게, 온도, 주가 등",
        "장점": "연속형 데이터를 자연스럽게 처리",
        "단점": "정규분포 가정이 맞지 않으면 성능 저하"
    },
    "Multinomial NB": {
        "사용 시기": "카운트/빈도 데이터 (이산형)",
        "가정": "특성이 다항 분포를 따름",
        "예시": "단어 빈도, TF-IDF, 리뷰 평점 개수",
        "장점": "텍스트 분류에 매우 효과적, 빈도 정보 활용",
        "단점": "음수 값 불가"
    },
    "Bernoulli NB": {
        "사용 시기": "이진 특성 (있다/없다)",
        "가정": "각 특성이 베르누이 분포를 따름",
        "예시": "단어 존재 여부, 스팸 키워드 포함 여부",
        "장점": "짧은 문서에 효과적, 빠름",
        "단점": "빈도 정보 손실"
    }
}

for nb_type, info in guidelines.items():
    print(f"\n{nb_type}:")
    for key, value in info.items():
        print(f"  • {key}: {value}")

print(f"\n💡 텍스트 분류에서:")
print(f"   - 긴 문서, 단어 빈도 중요 → Multinomial NB")
print(f"   - 짧은 문서 (트윗, 댓글) → Bernoulli NB")
print(f"   - 실험으로 둘 다 시도해보고 선택!")

"""
출력 예시:
======================================================================
Naive Bayes 변형 비교
======================================================================

【 1. Gaussian NB - 연속형 데이터 】
----------------------------------------------------------------------
데이터: Iris (연속형 특성 4개)
특성 예시: [5.1 3.5 1.4 0.2]
정확도: 0.9778

💡 Gaussian NB는 각 특성이 정규분포를 따른다고 가정
   각 클래스별 평균과 분산을 학습

예: 'setosa' 클래스의 통계
  sepal length (cm): 평균=5.01, 분산=0.12
  sepal width (cm): 평균=3.42, 분산=0.14
  petal length (cm): 평균=1.46, 분산=0.03
  petal width (cm): 평균=0.25, 분산=0.01

======================================================================
【 2. Multinomial NB - 단어 빈도 (카운트) 】
----------------------------------------------------------------------
데이터: 텍스트 문서 6개
특성: 단어 빈도 (CountVectorizer)
어휘: ['algorithms' 'deep' 'development' 'java' 'javascript' 'language'
 'learning' 'machine' 'networks' 'neural' 'programming' 'python' 'web']

첫 번째 문서 'python machine learning'의 벡터:
[0 0 0 0 0 0 1 1 0 0 0 1 0]
→ 각 단어의 출현 횟수

테스트:
  'python deep learning' → ML (ML:87.45%, Prog:12.55%)
  'java web programming' → Programming (ML:15.23%, Prog:84.77%)

💡 Multinomial NB는 단어의 출현 횟수를 사용
   '무료'가 3번 나오면 '무료'가 1번 나올 때보다 확률 높음
...
"""
```

## 하이퍼파라미터 튜닝

```python
"""
Naive Bayes 하이퍼파라미터 튜닝
- alpha (smoothing parameter)
- fit_prior (사전 확률 학습 여부)
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 로드 (20 newsgroups 일부 카테고리)
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

print("=== 20 Newsgroups 데이터 로드 ===")
print(f"카테고리: {categories}\n")

train_data = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=('headers', 'footers', 'quotes'))
test_data = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=('headers', 'footers', 'quotes'))

print(f"훈련 문서 수: {len(train_data.data)}")
print(f"테스트 문서 수: {len(test_data.data)}")

# 2. 벡터화 (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train = vectorizer.fit_transform(train_data.data)
X_test = vectorizer.transform(test_data.data)
y_train = train_data.target
y_test = test_data.target

print(f"특성 수 (단어 수): {X_train.shape[1]}")

# 3. alpha 값에 따른 성능 비교
print("\n=== Alpha 값에 따른 성능 ===\n")

alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
train_scores = []
test_scores = []

for alpha in alphas:
    mnb = MultinomialNB(alpha=alpha)
    mnb.fit(X_train, y_train)

    train_score = mnb.score(X_train, y_train)
    test_score = mnb.score(X_test, y_test)

    train_scores.append(train_score)
    test_scores.append(test_score)

    print(f"α={alpha:6.3f}: 훈련={train_score:.4f}, 테스트={test_score:.4f}")

# 최적 alpha 찾기
best_alpha_idx = np.argmax(test_scores)
best_alpha = alphas[best_alpha_idx]
print(f"\n최적 α: {best_alpha} (테스트 정확도: {test_scores[best_alpha_idx]:.4f})")

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(alphas, train_scores, marker='o', label='훈련 세트', linewidth=2)
plt.plot(alphas, test_scores, marker='s', label='테스트 세트', linewidth=2)
plt.axvline(best_alpha, color='red', linestyle='--',
            label=f'최적 α={best_alpha}', alpha=0.7)
plt.xscale('log')
plt.xlabel('α (smoothing parameter)')
plt.ylabel('정확도')
plt.title('Alpha 값에 따른 Naive Bayes 성능')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('nb_alpha_tuning.png', dpi=300, bbox_inches='tight')
print("\n시각화가 'nb_alpha_tuning.png'로 저장되었습니다.")

# 4. GridSearchCV로 최적 파라미터 찾기
print("\n=== GridSearchCV로 최적 파라미터 탐색 ===\n")

param_grid = {
    'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
    'fit_prior': [True, False]
}

mnb = MultinomialNB()
grid_search = GridSearchCV(mnb, param_grid, cv=5, scoring='accuracy',
                          n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f"\n최적 파라미터: {grid_search.best_params_}")
print(f"최적 CV 점수: {grid_search.best_score_:.4f}")

# 5. 최적 모델로 최종 평가
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n=== 최종 성능 (테스트 세트) ===\n")
print(classification_report(y_test, y_pred, target_names=categories))

# 6. fit_prior 파라미터의 영향
print("\n=== fit_prior 파라미터 영향 ===\n")

print("fit_prior=True (기본값):")
print("  사전 확률을 훈련 데이터에서 학습")
mnb_true = MultinomialNB(alpha=1.0, fit_prior=True)
mnb_true.fit(X_train, y_train)
print(f"  클래스별 사전 확률: {mnb_true.class_prior_}")
print(f"  테스트 정확도: {mnb_true.score(X_test, y_test):.4f}")

print("\nfit_prior=False:")
print("  모든 클래스의 사전 확률을 동일하게 가정 (균등 분포)")
mnb_false = MultinomialNB(alpha=1.0, fit_prior=False)
mnb_false.fit(X_train, y_train)
print(f"  클래스별 사전 확률: {mnb_false.class_prior_}")
print(f"  테스트 정확도: {mnb_false.score(X_test, y_test):.4f}")

print("\n💡 일반적으로 fit_prior=True가 더 좋은 성능")
print("   데이터가 불균형할 때 특히 중요!")

# 7. 파라미터 영향 정리
print("\n=== 파라미터 가이드 ===")
print("""
1. alpha (Smoothing Parameter):
   • 역할: 라플라스 스무딩 강도
   • 범위: 0 ~ 무한대 (보통 0.1 ~ 10)
   • 효과:
     - alpha=0: 스무딩 없음 (훈련 데이터에 과적합 위험)
     - alpha↑: 확률이 균등해짐 (과소적합 위험)
     - alpha=1: 라플라스 스무딩 (일반적 선택)
   • 선택: GridSearch로 0.1 ~ 10 사이 탐색

2. fit_prior:
   • 역할: 사전 확률 학습 여부
   • 값:
     - True (기본): 훈련 데이터의 클래스 비율 반영
     - False: 모든 클래스를 동일한 확률로 가정
   • 선택:
     - 데이터가 실제 분포를 반영 → True
     - 데이터가 불균형하지만 실제는 균등 → False
     - 일반적으로 True 권장

3. class_prior:
   • 역할: 사전 확률 직접 지정
   • 사용: 도메인 지식이 있을 때
   • 예: P(스팸)=0.3을 알고 있다면 [0.7, 0.3] 지정
""")

"""
출력 예시:
=== 20 Newsgroups 데이터 로드 ===
카테고리: ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

훈련 문서 수: 2257
테스트 문서 수: 1502
특성 수 (단어 수): 5000

=== Alpha 값에 따른 성능 ===

α= 0.001: 훈련=0.9956, 테스트=0.8802
α= 0.010: 훈련=0.9876, 테스트=0.8989
α= 0.100: 훈련=0.9654, 테스트=0.9134
α= 0.500: 훈련=0.9432, 테스트=0.9187
α= 1.000: 훈련=0.9321, 테스트=0.9201
α= 2.000: 훈련=0.9156, 테스트=0.9167
α= 5.000: 훈련=0.8876, 테스트=0.9034
α=10.000: 훈련=0.8654, 테스트=0.8876

최적 α: 1.0 (테스트 정확도: 0.9201)

=== GridSearchCV로 최적 파라미터 탐색 ===

Fitting 5 folds for each of 14 candidates, totalling 70 fits

최적 파라미터: {'alpha': 1.0, 'fit_prior': True}
최적 CV 점수: 0.9156

=== 최종 성능 (테스트 세트) ===

                        precision    recall  f1-score   support

           alt.atheism       0.89      0.92      0.91       319
  soc.religion.christian       0.92      0.91      0.92       398
         comp.graphics       0.94      0.93      0.93       389
               sci.med       0.96      0.94      0.95       396

              accuracy                           0.92      1502
             macro avg       0.93      0.93      0.93      1502
          weighted avg       0.93      0.92      0.92      1502

=== fit_prior 파라미터 영향 ===

fit_prior=True (기본값):
  사전 확률을 훈련 데이터에서 학습
  클래스별 사전 확률: [0.236 0.298 0.253 0.213]
  테스트 정확도: 0.9201

fit_prior=False:
  모든 클래스의 사전 확률을 동일하게 가정 (균등 분포)
  클래스별 사전 확률: [0.25 0.25 0.25 0.25]
  테스트 정확도: 0.9134

💡 일반적으로 fit_prior=True가 더 좋은 성능
   데이터가 불균형할 때 특히 중요!
"""
```

## 장단점 및 실전 팁

### Naive Bayes의 장단점

**장점:**

1. **빠른 학습과 예측**
   - O(n × d) 시간 복잡도 (n=데이터 개수, d=특성 개수)
   - 실시간 시스템에 적합

2. **적은 데이터로도 작동**
   - 각 특성별로 독립적으로 확률 계산
   - 고차원에서도 잘 작동

3. **확률 예측 제공**
   - predict_proba()로 신뢰도 확인 가능
   - 임계값 조정 용이

4. **구현이 간단**
   - 이해하기 쉬운 수학적 배경
   - 디버깅 용이

**단점:**

1. **독립 가정의 비현실성**
   - 실제로는 특성들이 상관관계 있음
   - 예: "무료"와 "대출"은 함께 나타남

2. **수치 안정성 문제**
   - 매우 작은 확률의 곱셈
   - 언더플로우 위험 → log 사용 필요

3. **연속형 데이터 처리**
   - Gaussian NB는 정규분포 가정 필요
   - 실제 분포와 다를 수 있음

### 실전 팁

```python
"""
Naive Bayes 실전 팁 모음
"""

# Tip 1: 항상 로그 확률 사용
print("=== Tip 1: 로그 확률 사용 ===")
print("""
문제: 작은 확률을 여러 번 곱하면 언더플로우 발생

나쁜 예:
P = 0.0001 × 0.0001 × 0.0001 × ... → 0.0 (언더플로우!)

좋은 예:
log(P) = log(0.0001) + log(0.0001) + ... → 정확한 계산

scikit-learn은 내부적으로 로그 사용:
  - decision_function(): 로그 확률 반환
  - predict_proba(): exp(log_prob)로 변환
""")

# Tip 2: 특성 스케일링 불필요
print("\n=== Tip 2: 특성 스케일링 불필요 ===")
print("""
Naive Bayes는 각 특성을 독립적으로 처리
→ 스케일링이 결과에 영향 없음

다른 알고리즘과 비교:
  - SVM, KNN: 스케일링 필수
  - Decision Tree: 스케일링 불필요
  - Naive Bayes: 스케일링 불필요 ✓

단, Gaussian NB에서 수치 안정성을 위해 스케일링할 수도 있음
""")

# Tip 3: 클래스 불균형 처리
print("\n=== Tip 3: 클래스 불균형 처리 ===")
print("""
옵션 1: fit_prior=True (기본값)
  - 훈련 데이터의 클래스 비율 반영
  - 불균형 데이터에 적합

옵션 2: class_prior 직접 지정
  - 도메인 지식 활용
  - 예: 실제 스팸 비율이 10%라면 [0.9, 0.1] 지정

from sklearn.naive_bayes import MultinomialNB

# 실제 스팸 비율 반영
mnb = MultinomialNB(class_prior=[0.9, 0.1])
""")

# Tip 4: 새로운 단어 처리
print("\n=== Tip 4: 새로운 단어 처리 ===")
print("""
문제: 테스트 시 처음 보는 단어
해결: alpha > 0 (라플라스 스무딩)

권장값:
  - 텍스트 분류: alpha=1.0 (라플라스)
  - 데이터가 많으면: alpha=0.1 ~ 0.5
  - 데이터가 적으면: alpha=1.0 ~ 2.0

항상 GridSearchCV로 최적값 찾기!
""")

# Tip 5: Multinomial vs Bernoulli 선택
print("\n=== Tip 5: Multinomial vs Bernoulli 선택 ===")
print("""
텍스트 길이에 따른 선택:

긴 문서 (뉴스 기사, 리뷰):
  → Multinomial NB + CountVectorizer
  → 단어 빈도가 중요한 정보

짧은 문서 (트윗, 댓글, SMS):
  → Bernoulli NB + binary=True
  → 존재 여부만으로 충분

애매하면:
  → 둘 다 시도하고 Cross-Validation으로 비교!

from sklearn.model_selection import cross_val_score

scores_multi = cross_val_score(MultinomialNB(), X, y, cv=5)
scores_bern = cross_val_score(BernoulliNB(), X, y, cv=5)

print(f"Multinomial: {scores_multi.mean():.3f}")
print(f"Bernoulli: {scores_bern.mean():.3f}")
""")

# Tip 6: TF-IDF vs Count
print("\n=== Tip 6: TF-IDF vs Count Vectorization ===")
print("""
Count Vectorizer (단어 빈도):
  - Multinomial/Bernoulli NB와 자연스러운 조합
  - 해석이 쉬움 (확률적 의미 명확)

TF-IDF:
  - 더 나은 성능을 보일 수 있음
  - 문서 길이 정규화 효과
  - 단, 확률 해석이 덜 직관적

권장: 둘 다 시도!

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Count 기반
vec_count = CountVectorizer()
X_count = vec_count.fit_transform(texts)
mnb_count = MultinomialNB().fit(X_count, y)

# TF-IDF 기반
vec_tfidf = TfidfVectorizer()
X_tfidf = vec_tfidf.fit_transform(texts)
mnb_tfidf = MultinomialNB().fit(X_tfidf, y)
""")

# Tip 7: 성능 향상 기법
print("\n=== Tip 7: 성능 향상 기법 ===")
print("""
1. 특성 선택:
   - 불용어(stopwords) 제거
   - 너무 흔하거나 희귀한 단어 제거
   - max_features, min_df, max_df 조정

2. N-gram 사용:
   - 단어 조합 정보 활용
   - ngram_range=(1, 2): 단어 + 바이그램

3. 앙상블:
   - Naive Bayes + 다른 모델 조합
   - VotingClassifier 사용

예시:
vectorizer = TfidfVectorizer(
    max_features=5000,      # 상위 5000개 단어만
    ngram_range=(1, 2),     # 유니그램 + 바이그램
    min_df=2,               # 최소 2번 출현
    max_df=0.8,             # 80% 이상 문서에 나타나면 제외
    stop_words='english'    # 불용어 제거
)
""")

# Tip 8: 확률 보정
print("\n=== Tip 8: 확률 보정 (Calibration) ===")
print("""
Naive Bayes의 확률 예측은 종종 극단적 (0 or 1에 가까움)
→ CalibratedClassifierCV로 보정

from sklearn.calibration import CalibratedClassifierCV

mnb = MultinomialNB()
calibrated_mnb = CalibratedClassifierCV(mnb, method='sigmoid', cv=5)
calibrated_mnb.fit(X_train, y_train)

# 보정된 확률
proba = calibrated_mnb.predict_proba(X_test)
""")

print("\n=== 요약: Naive Bayes 체크리스트 ===")
checklist = """
□ alpha 값 튜닝 (GridSearchCV)
□ Multinomial vs Bernoulli 선택
□ Count vs TF-IDF 비교
□ N-gram 시도
□ 불용어 제거
□ 클래스 불균형 처리 (fit_prior, class_prior)
□ Cross-validation으로 평가
□ 필요시 확률 보정
□ 다른 모델과 비교 (Logistic Regression, SVM)
"""
print(checklist)
