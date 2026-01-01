- [Lecture 1: Transformer](#lecture-1-transformer)
- [Materials](#materials)
- [강의 개요](#강의-개요)
  - [강의 목표](#강의-목표)
  - [강의 대상](#강의-대상)
  - [선수 과목](#선수-과목)
  - [강의 정보](#강의-정보)
  - [학습 자료](#학습-자료)
- [1. NLP 개요 (Natural Language Processing)](#1-nlp-개요-natural-language-processing)
  - [1.1. NLP란?](#11-nlp란)
    - [카테고리 1: Classification (분류)](#카테고리-1-classification-분류)
    - [카테고리 2: Multi-classification (다중 분류)](#카테고리-2-multi-classification-다중-분류)
    - [카테고리 3: Generation (생성)](#카테고리-3-generation-생성)
  - [1.2. NLP와 LLM의 역사](#12-nlp와-llm의-역사)
- [2. Tokenization (토큰화)](#2-tokenization-토큰화)
  - [2.1. 왜 Tokenization이 필요한가?](#21-왜-tokenization이-필요한가)
  - [2.2. Tokenization 방법](#22-tokenization-방법)
    - [방법 1: Arbitrary Units (임의 단위)](#방법-1-arbitrary-units-임의-단위)
    - [방법 2: Word-level (단어 수준)](#방법-2-word-level-단어-수준)
    - [방법 3: Subword-level (하위 단어 수준) ⭐ 권장](#방법-3-subword-level-하위-단어-수준--권장)
    - [방법 4: Character-level (문자 수준)](#방법-4-character-level-문자-수준)
  - [2.3. Tokenization 방법 비교](#23-tokenization-방법-비교)
  - [2.4. OOV (Out Of Vocabulary) 문제](#24-oov-out-of-vocabulary-문제)
  - [2.5. 실전 고려사항](#25-실전-고려사항)
- [3. Word Representation (단어 표현)](#3-word-representation-단어-표현)
  - [3.1. 문제 정의](#31-문제-정의)
  - [3.2. One-Hot Encoding (OHE)](#32-one-hot-encoding-ohe)
    - [Cosine Similarity (코사인 유사도)](#cosine-similarity-코사인-유사도)
  - [3.3. Word2vec (2013년)](#33-word2vec-2013년)
    - [3.3.1. 핵심 아이디어: Proxy Task (대리 작업)](#331-핵심-아이디어-proxy-task-대리-작업)
    - [3.3.2. 학습 과정 예시](#332-학습-과정-예시)
    - [3.3.3. 단계별 학습 과정](#333-단계별-학습-과정)
    - [3.3.4. 최종 결과: Word Embeddings](#334-최종-결과-word-embeddings)
  - [3.4. 학습 관련 질문 답변](#34-학습-관련-질문-답변)
    - [Q1: 언제 학습을 멈추나?](#q1-언제-학습을-멈추나)
    - [Q2: 생성은 어떻게 멈추나?](#q2-생성은-어떻게-멈추나)
    - [Q3: Hidden dimension 크기를 어떻게 정하나?](#q3-hidden-dimension-크기를-어떻게-정하나)
    - [Q4: 같은 단어, 다른 맥락은?](#q4-같은-단어-다른-맥락은)
  - [3.5. Word2vec의 한계](#35-word2vec의-한계)
- [4. Recurrent Neural Networks (RNN)](#4-recurrent-neural-networks-rnn)
  - [4.1. 문제 정의](#41-문제-정의)
  - [4.2. RNN의 아이디어](#42-rnn의-아이디어)
  - [4.3. RNN 동작 원리](#43-rnn-동작-원리)
  - [4.4. RNN 활용 방법](#44-rnn-활용-방법)
    - [4.4.1. Classification (분류)](#441-classification-분류)
    - [4.4.2. Multi-classification (다중 분류)](#442-multi-classification-다중-분류)
    - [4.4.3. Generation (생성)](#443-generation-생성)
  - [4.5. RNN의 한계](#45-rnn의-한계)
    - [문제 1: Long-range Dependencies (장거리 의존성)](#문제-1-long-range-dependencies-장거리-의존성)
    - [문제 2: Vanishing Gradient (기울기 소실)](#문제-2-vanishing-gradient-기울기-소실)
    - [문제 3: 순차 처리 (Sequential Processing)](#문제-3-순차-처리-sequential-processing)
  - [4.6. LSTM (Long Short-Term Memory)](#46-lstm-long-short-term-memory)
- [5. Attention Mechanism (주의 집중 메커니즘)](#5-attention-mechanism-주의-집중-메커니즘)
  - [5.1. 동기: 직접 연결](#51-동기-직접-연결)
  - [5.2. Attention의 역사](#52-attention의-역사)
  - [5.3. Self-Attention의 핵심 아이디어](#53-self-attention의-핵심-아이디어)
- [6. Transformer Architecture (다음 강의에서 상세히)](#6-transformer-architecture-다음-강의에서-상세히)
  - [6.1. 핵심 구성요소](#61-핵심-구성요소)
  - [6.2. Transformer의 장점](#62-transformer의-장점)
  - [6.3. Transformer의 영향](#63-transformer의-영향)
- [7. 요약](#7-요약)
  - [7.1. NLP 발전 과정](#71-nlp-발전-과정)
  - [7.2. 핵심 개념 정리](#72-핵심-개념-정리)
    - [Tokenization](#tokenization)
    - [Word Representation](#word-representation)
    - [Sequential Models](#sequential-models)
    - [Attention](#attention)
  - [7.3. 다음 강의 예고](#73-다음-강의-예고)
- [8. 중요 용어 정리](#8-중요-용어-정리)
- [9. 추가 자료](#9-추가-자료)
  - [9.1. 논문](#91-논문)
  - [9.2. 데이터셋](#92-데이터셋)
  - [9.3. 평가 지표](#93-평가-지표)
- [10. 강의 중 질문 답변](#10-강의-중-질문-답변)
  - [Q1: 왜 cosine similarity에서 norm을 고려하지 않나?](#q1-왜-cosine-similarity에서-norm을-고려하지-않나)
  - [Q2: Vocabulary 크기는 언어마다 다른가?](#q2-vocabulary-크기는-언어마다-다른가)
  - [Q3: Embedding은 어떻게 얻나?](#q3-embedding은-어떻게-얻나)
  - [Q4: Hidden dimension 크기는 어떻게 정하나?](#q4-hidden-dimension-크기는-어떻게-정하나)
  - [Q5: 학습은 언제 멈추나?](#q5-학습은-언제-멈추나)
  - [Q6: 생성은 언제 멈추나?](#q6-생성은-언제-멈추나)
  - [Q7: 같은 철자, 다른 의미는?](#q7-같은-철자-다른-의미는)
- [11. 핵심 메시지](#11-핵심-메시지)
  - [11.1. Transformer 이전](#111-transformer-이전)
  - [11.2. Transformer 이후](#112-transformer-이후)
  - [11.3. 왜 Transformer가 혁명인가?](#113-왜-transformer가-혁명인가)
- [다음 강의 예고](#다음-강의-예고)


----

# Lecture 1: Transformer

# Materials

- [CME 295](https://cme295.stanford.edu/syllabus/)
- [slide](https://cme295.stanford.edu/slides/fall25-cme295-lecture1.pdf)
- [video](https://www.youtube.com/watch?v=Ub3GoFaUcds&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=2)

---

# 강의 개요

**강의:** CME 295 - Transformers and Large Language Models
**강사:** Afshine & Shervine (쌍둥이 형제)
**배경:** MIT/Stanford ICME → Uber → Google → Netflix (LLM 전문)

## 강의 목표

1. **Transformer의 근본 메커니즘 이해**
   - LLM을 작동시키는 기초 아키텍처
2. **LLM의 학습 및 응용 방법**
   - 어떻게 학습되고 어디에 적용되는가

## 강의 대상

- LLM 분야 커리어를 원하는 사람 (연구 과학자, ML 과학자)
- LLM 기반 개인 프로젝트 개발자
- 다른 분야에서 AI/GenAI/LLM 적용을 원하는 사람

## 선수 과목

- **ML 기초**: 모델 학습, 신경망 개념
- **선형대수 기초**: 행렬 곱셈 등

## 강의 정보

- **시간:** 매주 금요일 3:30 - 5:20 PM
- **학점:** 2 units (Letter grade 또는 Credit/Non-credit)
- **평가:**
  - 중간고사 50% (10월 24일)
  - 기말고사 50% (12월 8일 주)
  - 과제 없음, 코딩 없음 (개념 중심)

## 학습 자료

- **교과서:** Super Study Guide - Transformer LLMs
- **VIP Cheat Sheet:** GitHub에서 제공 (다국어 지원)
- **공지:** Canvas Ed를 통한 질문/답변

---

# 1. NLP 개요 (Natural Language Processing)

## 1.1. NLP란?

**정의:** 텍스트를 다루고 계산하는 분야

NLP 작업은 크게 **3가지 카테고리**로 분류됩니다:

### 카테고리 1: Classification (분류)

**구조:** 텍스트 입력 → 하나의 예측값 출력

**예시:**
- **감정 분석 (Sentiment Analysis)**: 영화 리뷰 → 긍정/부정/중립
- **의도 감지 (Intent Detection)**: "내일 알람 만들어줘" → 의도: 알람 생성
- **언어 감지 (Language Detection)**: 프랑스어 텍스트 → "French"
- **주제 모델링 (Topic Modeling)**

**데이터셋 예시:**
- IMDb 영화 리뷰
- Amazon 제품 리뷰
- X (구 Twitter) 포스트

**평가 지표:**
- **Accuracy (정확도)**: 전체 중 올바르게 예측한 비율
- **Precision (정밀도)**: 양성으로 예측한 것 중 실제 양성인 비율
- **Recall (재현율)**: 실제 양성 중 올바르게 예측한 비율
- **F1 Score**: Precision과 Recall의 조화 평균

**왜 여러 지표가 필요한가?**
- 클래스 불균형 문제 (예: 99% 긍정, 1% 부정)
- Accuracy만으로는 오해의 소지 (다수 클래스만 예측해도 높은 정확도)

### 카테고리 2: Multi-classification (다중 분류)

**구조:** 텍스트 입력 → 여러 개의 예측값 출력

**예시:**
- **Named Entity Recognition (NER)**: 특정 단어를 카테고리로 라벨링
  - "Paris" → Location
  - "Tomorrow" → Time
- **Part-of-Speech Tagging**: 명사, 동사 등 품사 분류
- **Dependency/Constituency Parsing**: 문장 구조 분석

**평가:**
- 토큰 레벨 또는 엔티티 타입 레벨에서 분류 지표 사용
- 예: Location 카테고리의 정밀도/재현율

### 카테고리 3: Generation (생성)

**구조:** 텍스트 입력 → 가변 길이 텍스트 출력

**예시:**
- **기계 번역 (Machine Translation)**: 영어 → 독일어
- **질의응답 (Question Answering)**: ChatGPT, Gemini 같은 어시스턴트
- **요약 (Summarization)**: 긴 글 → 짧은 요약
- **코드/시 생성 (Generation)**: 다양한 텍스트 생성

**데이터셋 예시:**
- **WMT (Workshop on Machine Translation)**: 언어 쌍 데이터셋
  - 영어-프랑스어, 영어-독일어 (유럽 의회 데이터)

**평가 지표:**

1. **규칙 기반 지표 (과거)**
   - **BLEU (Bilingual Evaluation Under Study)**:
     - 번역과 참조 텍스트 비교
     - 값이 높을수록 좋음
   - **ROUGE (빨강)**:
     - 일련의 지표 모음
     - 값이 높을수록 좋음

   **문제점:** 항상 참조 텍스트(레이블)가 필요 → 비용이 많이 듦

2. **현대적 접근**
   - **Reference-free metrics**: 참조 텍스트 없이 평가
   - LLM의 발전으로 가능해짐 (강의 후반에 다룸)

3. **Perplexity (혼란도)**
   - 모델이 출력한 확률만 고려
   - 모델이 출력에 얼마나 "놀라는지" 측정
   - 값이 낮을수록 좋음

## 1.2. NLP와 LLM의 역사

**타임라인:**

```
1980년대: RNN 개념 등장
1990년대: LSTM 등장
2000년대: 인터넷/컴퓨팅 부족으로 제한적
2013년: Word2vec - 의미 있는 임베딩 계산
  예: "King - Queen = Paris - France = Berlin - Germany"
2017년: Transformer 논문 "Attention is All You Need"
  - 현재 모든 모델의 기초 아키텍처
2020년대: LLM의 시대
  - Compute와 Data 스케일 업
```

**제한 요인 (과거):**
- 인터넷 부족
- 컴퓨팅 파워 부족

**혁신 (현재):**
- 대규모 데이터셋
- 강력한 컴퓨팅 리소스
- Transformer 아키텍처

---

# 2. Tokenization (토큰화)

## 2.1. 왜 Tokenization이 필요한가?

**문제:** 모델은 숫자를 이해하지만, 텍스트는 이해하지 못함

**해결:** 텍스트를 모델이 이해할 수 있는 형태로 변환

**예시 문장:** "A cute teddy bear is reading"

이 문장을 어떻게 잘라야 할까?

## 2.2. Tokenization 방법

### 방법 1: Arbitrary Units (임의 단위)

```
"a" | "cute" | "teddy bear" | "is" | "reading"
```

- 완전히 임의적
- **Token**: 텍스트의 단위

### 방법 2: Word-level (단어 수준)

```
"a" | "cute" | "teddy" | "bear" | "is" | "reading"
```

**장점:**
- 직관적이고 간단

**단점:**
- 유사한 단어를 다르게 취급
  - "bear" vs "bears" → 다른 토큰
  - "run" vs "runs" → 다른 토큰
- 각각에 대해 별도의 임베딩을 학습해야 함

### 방법 3: Subword-level (하위 단어 수준) ⭐ 권장

```
"a" | "cute" | "teddy" | "bear" | "is" | "read" | "##ing"
```

**원리:** 단어의 어근(root)을 활용

**예시:**
- "bear", "bears" → 공통 부분: "bear"
- "run", "runs", "running" → 공통 부분: "run"

**장점:**
- ✅ 단어의 어근 활용
- ✅ OOV (Out Of Vocabulary) 위험 감소

**단점:**
- ❌ 시퀀스가 길어짐 → 계산 복잡도 증가

### 방법 4: Character-level (문자 수준)

```
"a" | " " | "c" | "u" | "t" | "e" | " " | ...
```

**장점:**
- ✅ 철자 오류에 강건함
- ✅ 대소문자 오류에 강건함
- ✅ OOV 문제 없음

**단점:**
- ❌ 시퀀스가 매우 길어짐
- ❌ 계산이 훨씬 느림
- ❌ 문자 하나의 의미를 파악하기 어려움 (예: "U"가 무엇을 의미하는가?)

## 2.3. Tokenization 방법 비교

| 방법 | 장점 | 단점 | OOV 위험 |
|------|------|------|----------|
| **Word-level** | 간단, 직관적 | 어근 활용 안 함 | 높음 |
| **Subword-level** ⭐ | 어근 활용, OOV 낮음 | 시퀀스 길어짐 | 낮음 |
| **Character-level** | 오류에 강건, OOV 없음 | 매우 느림, 의미 불명확 | 없음 |

## 2.4. OOV (Out Of Vocabulary) 문제

**정의:** 학습 시 보지 못한 토큰이 추론 시 나타나는 경우

**예시:**
```
학습: ["book", "soft", "teddy"]
추론: "fluffy" ← 본 적 없음!
```

**해결책:**
1. **Unknown token 예약**: `<UNK>` 토큰 사용
2. **Subword tokenization**: 어근 기반으로 분해
3. **Character-level**: 모든 문자 처리 가능

**방법별 OOV 빈도:**
- Word-level: 높음
- Subword-level: 낮음
- Character-level: 없음

## 2.5. 실전 고려사항

**Vocabulary Size (어휘 크기):**
- 단일 언어: 수만 개 (10,000 - 50,000)
- 다국어 모델: 수십만 개 (100,000+)
- 예: BERT - 30,522개, GPT-2 - 50,257개

**계산 복잡도:**
- 토큰 수 ↑ → 처리 시간 ↑
- Transformer는 시퀀스 길이에 민감

**Trade-off:**
- 어휘 크기 vs 시퀀스 길이 vs OOV 위험

---

# 3. Word Representation (단어 표현)

## 3.1. 문제 정의

**목표:** 각 토큰을 모델이 이해할 수 있는 숫자 표현으로 변환

**예시 vocabulary:**
```
["book", "soft", "teddy bear"]
```

각 단어를 어떻게 표현할까?

## 3.2. One-Hot Encoding (OHE)

**방법:** 각 단어에 고유한 위치에 1, 나머지는 0

**예시:**
```python
vocabulary_size = 3

"soft"        → [1, 0, 0]
"teddy bear"  → [0, 1, 0]
"book"        → [0, 0, 1]
```

**문제점: 유사도 계산 불가**

### Cosine Similarity (코사인 유사도)

**정의:** 벡터 간 각도로 유사도 측정

```
cosine_similarity = dot_product(v1, v2) / (norm(v1) × norm(v2))

범위: [-1, 1]
- 1: 같은 방향 (매우 유사)
- 0: 직교 (독립적)
- -1: 반대 방향 (반대)
```

**One-Hot의 문제:**
```python
"soft" · "teddy bear" = [1,0,0] · [0,1,0] = 0
"soft" · "book"       = [1,0,0] · [0,0,1] = 0
"teddy bear" · "book" = [0,1,0] · [0,0,1] = 0
```

→ 모든 벡터가 서로 직교 (유사도 0)

**이상적인 목표:**
```python
"teddy bear" · "soft" = 높은 값  (곰인형은 부드러움)
"teddy bear" · "book" = 0에 가까움 (관련 없음)
```

## 3.3. Word2vec (2013년)

**혁신:** 데이터로부터 의미 있는 임베딩 학습

**유명한 예시:**
```
King - Man + Woman = Queen
Paris - France + Germany = Berlin
```

→ 단어 간 의미 관계를 벡터 연산으로 표현!

### 3.3.1. 핵심 아이디어: Proxy Task (대리 작업)

**목표:** 단어의 의미 있는 표현 학습

**방법:** 텍스트로부터 뭔가를 예측하는 작업을 통해 학습

**철학:**
> "단어를 예측할 수 있다면, 언어를 이해하고 있다는 뜻이다"

**두 가지 방법:**

1. **CBOW (Continuous Bag of Words)**
   - 주변 단어들 → 중심 단어 예측

2. **Skip-gram**
   - 중심 단어 → 주변 단어들 예측

### 3.3.2. 학습 과정 예시

**문장:** "A cute teddy bear is reading"

**Proxy Task:** 다음 단어 예측

**Neural Network 구조:**
```
Input layer:    v (vocabulary_size)  예: 10,000
Hidden layer:   d (embedding_dim)    예: 512
Output layer:   v (vocabulary_size)  예: 10,000
```

**핵심:** Hidden layer의 가중치가 word embedding이 됨!

### 3.3.3. 단계별 학습 과정

**예시 Vocabulary:**
```
1: "a"
2: "cute"
3: "teddy bear"
4: "is"
5: "reading"
6: "book"

vocabulary_size (v) = 6
embedding_dim (d) = 2
```

**Step 1: 첫 단어 "a" → "cute" 예측**

```python
# Input
x = [1, 0, 0, 0, 0, 0]  # one-hot for "a"

# Hidden layer (embedding 학습)
W1: (6, 2) - 학습 대상
h = x @ W1 = [0.2, 0.9]  # 2차원 벡터

# Output layer
W2: (2, 6)
z = h @ W2 = [2.1, 1.8, 0.5, 1.2, 0.3, 0.7]

# Softmax
probs = softmax(z) = [0.2, 0.4, 0.1, 0.1, 0.1, 0.1]
                      ↑    ↑
                     "a" "cute" ← 가장 높은 확률

# Loss 계산
target = [0, 1, 0, 0, 0, 0]  # "cute"의 one-hot
loss = CrossEntropy(probs, target)

# Backpropagation
W1, W2를 업데이트하여 "cute" 예측 확률 증가
```

**Step 2: 두 번째 단어 "cute" → "teddy bear" 예측**

```python
x = [0, 1, 0, 0, 0, 0]  # one-hot for "cute"
h = x @ W1 = [0.8, 0.4]  # 다른 임베딩 벡터
probs = softmax(h @ W2) = [0.16, 0.16, 0.16, 0.16, 0.16, 0.16]  # 균등

target = [0, 0, 1, 0, 0, 0]  # "teddy bear"
loss = CrossEntropy(probs, target)

# Update weights to increase P("teddy bear")
```

**Step 3: 모든 단어에 대해 반복**

수천, 수만 번 반복하면:
- W1의 각 행 = 각 단어의 learned embedding
- 의미가 유사한 단어들의 벡터가 가까워짐

### 3.3.4. 최종 결과: Word Embeddings

**학습 완료 후:**

```python
# W1의 행들이 각 단어의 임베딩
embedding["a"]          = W1[0] = [0.15, 0.23]
embedding["cute"]       = W1[1] = [0.82, 0.91]
embedding["teddy bear"] = W1[2] = [0.79, 0.88]  # cute와 유사!
embedding["is"]         = W1[3] = [0.05, 0.12]
embedding["reading"]    = W1[4] = [0.43, 0.56]
embedding["book"]       = W1[5] = [0.38, 0.51]  # reading과 유사!

# 유사도 계산
cosine("teddy bear", "cute") = high     # 의미가 유사
cosine("teddy bear", "is") = low        # 의미가 다름
```

**학습 과정 시각화:**

```
Before training:
W1 = random weights
→ 모든 단어가 무작위 벡터

After millions of examples:
"king"와 "queen"은 가까워짐
"paris"와 "france"는 관계 형성
"soft"와 "teddy bear"는 연관성 학습
```

## 3.4. 학습 관련 질문 답변

### Q1: 언제 학습을 멈추나?

**답변:**
- Loss 함수가 수렴할 때까지
- Epoch별로 loss를 추적
- Loss가 더 이상 감소하지 않으면 종료
- 또는 validation set 성능으로 판단

### Q2: 생성은 어떻게 멈추나?

**답변:**
- 특수 토큰 사용: `<END>` 또는 `<EOS>` (End Of Sequence)
- 이 토큰이 생성되면 종료

### Q3: Hidden dimension 크기를 어떻게 정하나?

**답변:**
- Trade-off 고려:
  - 크면: 더 풍부한 표현, 하지만 계산 비용 ↑
  - 작으면: 빠른 계산, 하지만 표현력 ↓
- 일반적 크기:
  - 수백: 300, 512, 768
  - 수천: 1024, 2048 (큰 모델)
- 경험적으로 결정 (empirical)
- 다른 연구 결과 참고

### Q4: 같은 단어, 다른 맥락은?

**질문:** "bank" (은행 vs 강둑)를 어떻게 구분?

**답변:**
- Word2vec은 구분 못함 (문맥 무시)
- 해결책: Contextual embeddings (다음 섹션에서)

## 3.5. Word2vec의 한계

**문제점:**

1. **문맥 무시 (Context-agnostic)**
   ```python
   "bank" → 항상 같은 벡터

   "I went to the bank" (은행)
   "The river bank" (강둑)

   → 같은 임베딩 사용! (문제)
   ```

2. **단어 순서 무시**
   ```python
   "not good" vs "good"

   Average embedding:
   avg(["not", "good"]) ≈ 의미 모호
   ```

**해결책:** Recurrent Neural Networks (다음 섹션)

---

# 4. Recurrent Neural Networks (RNN)

## 4.1. 문제 정의

**Word2vec의 문제:**
- 문맥 무시
- 단어 순서 무시

**목표:**
- 문장 전체의 의미 표현
- 단어 순서 고려

**Naive 접근:**
```python
# 단어 임베딩의 평균
sentence_embedding = mean([embed("a"), embed("cute"), embed("teddy")])

문제: 순서와 문맥 정보 손실
```

## 4.2. RNN의 아이디어

**핵심:** 단어를 순차적으로 처리하며 "hidden state" 유지

**구조:**
```
Hidden State (h): 지금까지 본 문장의 의미 요약

Step 1: h0 (초기) + "a"        → h1
Step 2: h1        + "cute"     → h2
Step 3: h2        + "teddy"    → h3
Step 4: h3        + "bear"     → h4
...
```

**용어:**
- **Hidden State** = **Activation** = **Context Vector**
- 기호: `h` 또는 `a`

## 4.3. RNN 동작 원리

**Step-by-step:**

```
문장: "A cute teddy bear is reading"

Step 1:
  Input:  x1 = one_hot("a")        shape: (v,)
  Hidden: h0 = [0, 0, ..., 0]      shape: (d,)  초기값

  h1 = f(h0, x1)  # 함수 f는 neural network
  output1 = g(h1)  # "cute" 예측

Step 2:
  Input:  x2 = one_hot("cute")
  Hidden: h1 (이전 단계의 hidden state)

  h2 = f(h1, x2)
  output2 = g(h2)  # "teddy" 예측

Step 3:
  Input:  x3 = one_hot("teddy")
  Hidden: h2

  h3 = f(h2, x3)
  output3 = g(h3)  # "bear" 예측

...
```

**수식:**
```python
h_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b)
output_t = softmax(W_o @ h_t + b_o)

여기서:
- W_h: hidden-to-hidden 가중치
- W_x: input-to-hidden 가중치
- W_o: hidden-to-output 가중치
```

## 4.4. RNN 활용 방법

### 4.4.1. Classification (분류)

**방법:** 마지막 hidden state 사용

```python
문장: "This movie is great"

h1 ← "This"
h2 ← "movie"
h3 ← "is"
h4 ← "great"

sentiment = classifier(h4)  # 마지막 hidden state
            ↓
         "Positive"
```

### 4.4.2. Multi-classification (다중 분류)

**예: Named Entity Recognition**

```python
문장: "Paris is beautiful"

h1 ← "Paris"    → NER(h1) = "LOCATION"
h2 ← "is"       → NER(h2) = "O" (일반)
h3 ← "beautiful" → NER(h3) = "O"
```

각 토큰의 hidden state를 해당 토큰의 라벨 예측에 사용

### 4.4.3. Generation (생성)

**예: Machine Translation**

```python
Encoder RNN:
  "I love cats"
  h1 ← "I"
  h2 ← "love"
  h3 ← "cats"

  context = h3  # 전체 문장 의미

Decoder RNN:
  Start with context
  Generate: "나는" → "고양이를" → "사랑해" → <END>
```

## 4.5. RNN의 한계

### 문제 1: Long-range Dependencies (장거리 의존성)

**예시:**
```
"The cat, which we saw yesterday at the park, was sleeping."

"cat" → "was" 관계를 파악하려면 중간의 많은 단어를 거쳐야 함
```

**문제:** Hidden state에 모든 정보를 압축 → 멀리 있는 정보는 희미해짐

### 문제 2: Vanishing Gradient (기울기 소실)

**Backpropagation through time (BPTT):**

```python
# Forward
h1 = f(h0, x1)
h2 = f(h1, x2)
h3 = f(h2, x3)
...
h10 = f(h9, x10)

# Backward (h10의 loss를 h1까지 전파)
∂L/∂h1 = ∂L/∂h10 × ∂h10/∂h9 × ∂h9/∂h8 × ... × ∂h2/∂h1
         └─────────────────────────────────────────┘
                   많은 항들의 곱

문제:
- 각 항 < 1이면 → 곱하면 0으로 수렴 (vanishing)
- 각 항 > 1이면 → 곱하면 ∞로 발산 (exploding)
```

**결과:**
- 멀리 있는 단어로부터 학습 불가
- 긴 문장에서 성능 저하

### 문제 3: 순차 처리 (Sequential Processing)

```python
문장이 길면:
  h1 계산 → h2 계산 → h3 계산 → ... → h100 계산

  h50을 계산하려면 h1~h49를 모두 계산해야 함
  → 병렬 처리 불가
  → 학습/추론이 매우 느림
```

## 4.6. LSTM (Long Short-Term Memory)

**목적:** Vanishing gradient 문제 완화

**추가 메커니즘:**
1. **Cell State (c)**: 장기 기억 저장
2. **Gates**: 정보를 선택적으로 기억/망각

```python
# LSTM 구조
Forget gate:  어떤 정보를 잊을까?
Input gate:   어떤 정보를 기억할까?
Output gate:  어떤 정보를 출력할까?

c_t = forget_gate × c_{t-1} + input_gate × new_info
h_t = output_gate × tanh(c_t)
```

**개선점:**
- ✅ 더 긴 의존성 포착
- ✅ 기울기 소실 완화

**여전한 문제:**
- ❌ 순차 처리 (병렬화 불가)
- ❌ 완벽한 해결책은 아님

---

# 5. Attention Mechanism (주의 집중 메커니즘)

## 5.1. 동기: 직접 연결

**RNN의 근본 문제:**
```
"A cute teddy bear is reading" → "Un ours en peluche mignon lit"

h1 → h2 → h3 → h4 → h5 → h6 (Encoder)
                      ↓
                   context
                      ↓
                  Decoder
```

→ "reading"을 번역할 때 "is"를 거쳐서 정보 전달 (간접적)

**Attention의 아이디어:**
```
"reading"을 번역할 때 → 직접 "reading"을 참조!

Direct connection:
  decoder_state ──직접 연결──→ encoder states
```

## 5.2. Attention의 역사

**2014년:** Attention 메커니즘 도입
- RNN과 함께 사용
- Long-range dependency 문제 해결

**2017년:** Transformer 논문 "Attention is All You Need"
- RNN 완전히 제거
- Attention만으로 모델 구축
- 병렬 처리 가능

## 5.3. Self-Attention의 핵심 아이디어

**목표:** 문장 내 모든 단어 간 직접 연결

**예시:**
```
"The cat sat on the mat"

"sat"을 이해할 때:
- "cat"를 직접 참조  (주어)
- "mat"를 직접 참조  (장소)
- 순차 처리 불필요!
```

**장점:**
1. ✅ 모든 단어 간 직접 연결
2. ✅ 병렬 처리 가능
3. ✅ Long-range dependency 해결
4. ✅ Vanishing gradient 문제 없음

---

# 6. Transformer Architecture (다음 강의에서 상세히)

## 6.1. 핵심 구성요소

**Transformer = Self-Attention + Feedforward**

```
Input: "A cute teddy bear"

1. Tokenization
   ["A", "cute", "teddy", "bear"]

2. Embedding + Positional Encoding
   각 토큰 → 벡터 + 위치 정보

3. Self-Attention Layers (×N)
   모든 단어가 서로를 참조

4. Feedforward Layers (×N)
   각 위치별 변환

5. Output
```

## 6.2. Transformer의 장점

**vs RNN:**
- ✅ 병렬 처리 가능 → 훨씬 빠름
- ✅ Long-range dependency 자연스럽게 처리
- ✅ 기울기 소실 문제 없음

**vs LSTM:**
- ✅ 더 간단한 구조
- ✅ 더 나은 성능
- ✅ 확장성 (scalability)

## 6.3. Transformer의 영향

**현대 LLM의 기초:**
- GPT 시리즈 (GPT-2, GPT-3, GPT-4)
- BERT
- T5
- LLaMA
- ...

모두 Transformer 기반!

---

# 7. 요약

## 7.1. NLP 발전 과정

```
1980s-1990s: RNN, LSTM
  → 순차 처리, 기울기 소실 문제

2013: Word2vec
  → 의미 있는 word embedding
  → 하지만 문맥 무시

2014: Attention
  → 직접 연결, long-range dependency 해결

2017: Transformer
  → Attention만으로 모델 구축
  → 병렬 처리, 확장 가능

2020s: LLMs
  → Transformer 기반 대규모 모델
  → 엄청난 데이터 + 컴퓨팅
```

## 7.2. 핵심 개념 정리

### Tokenization
- Word-level: 간단하지만 OOV 많음
- **Subword-level**: 균형잡힌 선택 ⭐
- Character-level: OOV 없지만 너무 느림

### Word Representation
- One-hot: 유사도 계산 불가
- **Word2vec**: 의미 있는 임베딩 학습 ⭐
- Limitation: 문맥 무시

### Sequential Models
- RNN: 순차 처리, vanishing gradient
- LSTM: 개선되었지만 여전히 느림
- **Transformer**: Attention 기반, 병렬 처리 ⭐

### Attention
- 직접 연결
- Long-range dependency 해결
- **Transformer의 핵심** ⭐

## 7.3. 다음 강의 예고

**Lecture 2: Transformer Details**
- Self-Attention 수식
- Multi-Head Attention
- Positional Encoding
- Encoder-Decoder 구조
- Training techniques

---

# 8. 중요 용어 정리

| 용어 | 영어 | 설명 |
|------|------|------|
| 토큰화 | Tokenization | 텍스트를 단위로 자르기 |
| 임베딩 | Embedding | 단어의 벡터 표현 |
| OOV | Out Of Vocabulary | 학습에 없던 단어 |
| 은닉 상태 | Hidden State | RNN의 문장 요약 벡터 |
| 기울기 소실 | Vanishing Gradient | 역전파 시 기울기가 0으로 |
| 장거리 의존성 | Long-range Dependency | 멀리 떨어진 단어 간 관계 |
| 주의 집중 | Attention | 관련 정보에 직접 연결 |
| 대리 작업 | Proxy Task | 진짜 목표를 위한 보조 학습 |

---

# 9. 추가 자료

## 9.1. 논문

- **Word2vec (2013)**
  - "Efficient Estimation of Word Representations in Vector Space"
  - Mikolov et al.

- **Attention (2014)**
  - "Neural Machine Translation by Jointly Learning to Align and Translate"
  - Bahdanau et al.

- **Transformer (2017)** ⭐
  - "Attention is All You Need"
  - Vaswani et al.

## 9.2. 데이터셋

- **분류**: IMDb, Amazon Reviews, Twitter/X
- **NER**: CoNLL-2003, OntoNotes
- **번역**: WMT (Workshop on Machine Translation)
- **QA**: SQuAD, Natural Questions

## 9.3. 평가 지표

**Classification:**
- Accuracy
- Precision / Recall / F1

**Generation:**
- BLEU (reference-based)
- ROUGE (reference-based)
- Perplexity (model-based)
- Modern: Reference-free metrics (LLM-based)

---

# 10. 강의 중 질문 답변

## Q1: 왜 cosine similarity에서 norm을 고려하지 않나?

**답변:**
- Cosine similarity는 벡터의 방향만 고려
- 크기(norm)는 정규화되어 제거됨
- 방향이 같으면 유사하다고 판단
- 다른 지표(dot product)도 사용 가능

## Q2: Vocabulary 크기는 언어마다 다른가?

**답변:**
- 단일 언어: 수만 개 (영어 10,000-50,000)
- 다국어: 수십만 개
- 중국어 등: 문자 체계 차이 고려
- Subword tokenizer가 일반적
- Trade-off: 어휘 크기 vs 표현력

## Q3: Embedding은 어떻게 얻나?

**답변:**
- Word2vec의 W1 행렬 사용
- 학습 완료 후 각 행이 embedding
- 또는 사전 학습된 embedding 사용 (transfer learning)

## Q4: Hidden dimension 크기는 어떻게 정하나?

**답변:**
- 경험적 (empirical)
- 일반적: 수백 (300, 512, 768)
- 큰 모델: 수천 (1024, 2048)
- Trade-off: 표현력 vs 계산 비용
- 기존 연구 참고

## Q5: 학습은 언제 멈추나?

**답변:**
- Loss 함수 수렴 시
- Epoch별 추적
- Validation set 성능
- Early stopping 기법

## Q6: 생성은 언제 멈추나?

**답변:**
- `<END>` 또는 `<EOS>` 토큰
- 최대 길이 도달
- 확률이 일정 이하로 떨어짐

## Q7: 같은 철자, 다른 의미는?

**예:** "bank" (은행 vs 강둑)

**답변:**
- Word2vec은 구분 불가
- 해결: Contextual embeddings (BERT, GPT)
- Transformer 기반 모델에서 해결

---

# 11. 핵심 메시지

## 11.1. Transformer 이전

```
Text → Tokens → Embeddings → RNN/LSTM → Output

문제:
- 느린 순차 처리
- 기울기 소실
- Long-range dependency
```

## 11.2. Transformer 이후

```
Text → Tokens → Embeddings → Self-Attention → Output

해결:
- ✅ 빠른 병렬 처리
- ✅ 직접 연결
- ✅ 확장 가능
```

## 11.3. 왜 Transformer가 혁명인가?

1. **병렬화**: 학습/추론이 극적으로 빨라짐
2. **확장성**: 더 큰 모델, 더 많은 데이터
3. **성능**: 모든 NLP 태스크에서 SOTA
4. **범용성**: 비전, 음성 등 다른 분야로 확장

**결과:** GPT, BERT, 그리고 현재의 LLM 시대!

---

# 다음 강의 예고

**Lecture 2: Transformer-based models & tricks**
- Transformer 아키텍처 상세
- Self-Attention 수식
- Multi-Head Attention
- Positional Encoding
- Training tricks
- BERT, GPT 등 변형들

---

**강의 종료**

수고하셨습니다! 다음 시간에 Transformer의 세부 구조를 깊이 있게 다루겠습니다.
