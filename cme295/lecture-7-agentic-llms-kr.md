# Lecture 7: Agentic LLMs (RAG & Tool Calling)

# Materials

- [CME 295](https://cme295.stanford.edu/syllabus/)
- [slide](https://cme295.stanford.edu/slides/fall25-cme295-lecture7.pdf)
- [video](https://www.youtube.com/watch?v=h-7S6HNq0Vg&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=7)

# Table of Contents

- [Lecture 7: Agentic LLMs (RAG \& Tool Calling)](#lecture-7-agentic-llms-rag--tool-calling)
- [Materials](#materials)
- [Table of Contents](#table-of-contents)
- [강의 개요](#강의-개요)
  - [강의 목표](#강의-목표)
  - [주요 학습 내용](#주요-학습-내용)
- [1. 이전 강의 복습 (Lecture 6)](#1-이전-강의-복습-lecture-6)
  - [1.1. Reasoning Models 복습](#11-reasoning-models-복습)
  - [1.2. GRPO 알고리즘](#12-grpo-알고리즘)
  - [1.3. Length Bias 문제](#13-length-bias-문제)
- [2. Vanilla LLM의 약점](#2-vanilla-llm의-약점)
  - [2.1. Limited Reasoning (Lecture 6에서 해결)](#21-limited-reasoning-lecture-6에서-해결)
  - [2.2. Static Knowledge (이번 강의에서 해결)](#22-static-knowledge-이번-강의에서-해결)
  - [2.3. All Talk, No Action (이번 강의에서 해결)](#23-all-talk-no-action-이번-강의에서-해결)
- [3. RAG (Retrieval Augmented Generation)](#3-rag-retrieval-augmented-generation)
  - [3.1. Knowledge Cutoff 문제](#31-knowledge-cutoff-문제)
  - [3.2. 왜 Continue Training을 하지 않는가?](#32-왜-continue-training을-하지-않는가)
  - [3.3. 왜 모든 정보를 Context에 넣지 않는가?](#33-왜-모든-정보를-context에-넣지-않는가)
    - [문제 1: Context Length 제한](#문제-1-context-length-제한)
    - [문제 2: Needle in a Haystack](#문제-2-needle-in-a-haystack)
    - [문제 3: 비용](#문제-3-비용)
  - [3.4. RAG의 핵심 아이디어](#34-rag의-핵심-아이디어)
  - [3.5. RAG의 3단계](#35-rag의-3단계)
- [4. Knowledge Base 구축](#4-knowledge-base-구축)
  - [4.1. Chunking](#41-chunking)
  - [4.2. Embedding 생성](#42-embedding-생성)
  - [4.3. 하이퍼파라미터](#43-하이퍼파라미터)
    - [1. Embedding Size](#1-embedding-size)
    - [2. Chunk Size](#2-chunk-size)
    - [3. Overlap Size](#3-overlap-size)
- [5. Retrieval: 2-Stage Approach](#5-retrieval-2-stage-approach)
  - [5.1. Stage 1: Candidate Retrieval](#51-stage-1-candidate-retrieval)
    - [Semantic Similarity Search (Bi-encoder)](#semantic-similarity-search-bi-encoder)
    - [Heuristic Search (BM25)](#heuristic-search-bm25)
    - [Hybrid Approach](#hybrid-approach)
  - [5.2. Stage 2: Re-ranking](#52-stage-2-re-ranking)
- [6. Retrieval 개선 기법](#6-retrieval-개선-기법)
  - [6.1. HyDE (Hypothetical Document Embeddings)](#61-hyde-hypothetical-document-embeddings)
  - [6.2. Contextual Retrieval](#62-contextual-retrieval)
  - [6.3. Prompt Caching](#63-prompt-caching)
- [7. Retrieval Evaluation](#7-retrieval-evaluation)
  - [7.1. NDCG (Normalized Discounted Cumulative Gain)](#71-ndcg-normalized-discounted-cumulative-gain)
  - [7.2. MRR (Mean Reciprocal Rank)](#72-mrr-mean-reciprocal-rank)
  - [7.3. Precision@k와 Recall@k](#73-precisionk와-recallk)
  - [7.4. Benchmarks](#74-benchmarks)
- [8. Tool Calling](#8-tool-calling)
  - [8.1. Tool Calling이란?](#81-tool-calling이란)
  - [8.2. Function Calling](#82-function-calling)
  - [8.3. Tool Calling 예시](#83-tool-calling-예시)
- [9. 요약](#9-요약)
  - [핵심 개념](#핵심-개념)
    - [1. RAG (Retrieval Augmented Generation)](#1-rag-retrieval-augmented-generation)
    - [2. Knowledge Base 구축](#2-knowledge-base-구축)
    - [3. 2-Stage Retrieval](#3-2-stage-retrieval)
    - [4. Evaluation Metrics](#4-evaluation-metrics)
    - [5. Tool Calling](#5-tool-calling)
  - [실전 체크리스트](#실전-체크리스트)
- [10. 중요 용어 정리](#10-중요-용어-정리)
    - [RAG 관련](#rag-관련)
    - [Retrieval 관련](#retrieval-관련)
    - [Retrieval 개선 기법](#retrieval-개선-기법)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Tool Calling 관련](#tool-calling-관련)
    - [기타](#기타)
  - [다음 강의 예고](#다음-강의-예고)

---

# 강의 개요

## 강의 목표

이번 강의에서는 LLM이 외부 세계 및 다른 시스템과 상호작용하는 실용적인 기법들을 학습합니다.

**학습 목표:**
- RAG (Retrieval Augmented Generation) 이해하기
- Knowledge Base 구축 및 Retrieval 방법론
- Tool Calling과 Function Calling
- Agentic workflows 기초

## 주요 학습 내용

**1. RAG (Retrieval Augmented Generation)**
- Knowledge cutoff 문제 해결
- 2-stage retrieval: Candidate Retrieval + Re-ranking
- Evaluation metrics: NDCG, MRR, Precision@k, Recall@k

**2. Knowledge Base 구축**
- Chunking strategies
- Embedding models (SBERT)
- Bi-encoder vs Cross-encoder

**3. Retrieval 개선 기법**
- Semantic similarity (embeddings)
- Heuristic search (BM25)
- HyDE, Contextual Retrieval
- Prompt Caching

**4. Tool Calling**
- Function calling 개념
- External API 통합
- Structured data 처리

---

# 1. 이전 강의 복습 (Lecture 6)

## 1.1. Reasoning Models 복습

**Vanilla LLM vs Reasoning Model:**

```
Vanilla LLM:
Input: Prompt → LLM → Output: Answer

Reasoning Model:
Input: Prompt → LLM → Output: Reasoning Chain + Answer
                              (숨겨짐)
```

**차이점:**

```python
# Vanilla LLM
Prompt: "2020년에 태어난 곰은 2025년에 몇 살?"
Output: "5살"

# Reasoning Model
Prompt: "2020년에 태어난 곰은 2025년에 몇 살?"
Output:
"""
<think>
현재: 2025년
태어난 연도: 2020년
나이 = 2025 - 2020 = 5
</think>

답: 5살
"""
```

## 1.2. GRPO 알고리즘

**GRPO (Group Relative Policy Optimization):**

```
핵심 아이디어:
1. 같은 prompt로 여러 completions 생성
2. 각 completion의 reward 계산
3. Group 평균과 비교하여 advantage 계산
4. Value function 불필요!
```

**GRPO 프로세스:**

```
Query → Policy → Completion 1 → Reward 1
                Completion 2 → Reward 2
                Completion 3 → Reward 3
                Completion 4 → Reward 4

Advantage = (Reward - Group Mean) / Group Std
```

**Reward 설계:**

```python
def compute_reward(output, problem):
    """
    두 가지 reward 결합
    """
    # Reward 1: Reasoning chain 존재
    has_reasoning = 1.0 if "<think>" in output else 0.0

    # Reward 2: 정답 여부
    is_correct = verify_answer(output, problem)

    return has_reasoning * is_correct
```

## 1.3. Length Bias 문제

**문제:**

```
GRPO 학습 중 관찰:
- 성능은 plateau
- 하지만 output 길이는 계속 증가!

이유:
- Loss function에 length-dependent term 존재
- 긴 답변이 유리하게 작용
```

**학습 곡선:**

```
Performance (AIM)          Output Length
     │                          │
 70% ├─────────────             │
     │            ╱         2000├────────────╱
 60% ├───────────╱              │          ╱
     │         ╱            1500├────────╱
 50% ├───────╱                  │      ╱
     │     ╱                1000├────╱
 40% ├───╱                      │  ╱
     │                       500├╱
     └─────────────              └─────────────
      RL Steps                    RL Steps

문제: 성능은 정체되는데 길이만 증가
```

**해결책:**

**1. DAPO (2024):**

```python
# Length-independent normalization
advantage = reward / constant_factor  # 위치 독립적
```

**2. GRPO Done Right (2024):**

```python
# Normalization 제거
advantage = reward - group_mean  # std로 나누지 않음
```

---

# 2. Vanilla LLM의 약점

## 2.1. Limited Reasoning (Lecture 6에서 해결)

**문제:** 복잡한 수학/코딩 문제 해결 어려움

**해결:** Reasoning models (GRPO)

## 2.2. Static Knowledge (이번 강의에서 해결)

**문제:**

```
Pre-training data의 Cutoff Date:
- GPT-4: 2024년 9월 30일
- Claude: 특정 날짜

Cutoff 이후 정보는 모름!

예시:
Q: "2024년 11월 미국 대선 결과는?"
→ Cutoff이 9월이면 답 못함
```

**Knowledge Cutoff 예시:**

```python
# OpenAI GPT-4 Model Card
{
    "model": "gpt-4-turbo",
    "knowledge_cutoff": "2024-09-30",
    "context_window": 400000,  # 400K tokens
    "pricing": {
        "input": "$0.01 / 1K tokens",
        "output": "$0.03 / 1K tokens"
    }
}
```

**해결:** RAG (Retrieval Augmented Generation)

## 2.3. All Talk, No Action (이번 강의에서 해결)

**문제:**

```
User: "내일 오전 9시 회의 예약해줘"

Vanilla LLM:
"회의를 예약하겠습니다"  ← 말만 함

실제:
- 캘린더에 추가 안됨
- 알림 설정 안됨
- 이메일 발송 안됨
```

**해결:** Tool Calling & Agents

---

# 3. RAG (Retrieval Augmented Generation)

## 3.1. Knowledge Cutoff 문제

**시나리오:**

```
Model trained: 2024년 9월
Current date: 2024년 11월

Q: "최근 선거 결과는?"

Problem:
- Model은 9월까지만 알고 있음
- 11월 정보는 없음
- 틀린 답변 또는 "모름" 응답
```

## 3.2. 왜 Continue Training을 하지 않는가?

**Option 1: Continue Training (❌ 나쁜 방법)**

```python
# 새로운 데이터로 추가 학습
new_data = load_data("2024-10 ~ 2024-11")
model.train(new_data)

문제점:
1. Knowledge 변경 시 regression 발생
   - 기존 성능 저하 가능

2. 유지보수 overhead
   - 여러 use case마다 별도 학습 필요

3. 비용
   - Re-training은 매우 비쌈
```

**예시:**

```
Use Case 1: Medical QA
├─ Base Model + Medical Fine-tuning
└─ Knowledge update → 다시 학습 필요!

Use Case 2: Legal QA
├─ Base Model + Legal Fine-tuning
└─ Knowledge update → 다시 학습 필요!

→ 모든 use case에 대해 반복!
```

## 3.3. 왜 모든 정보를 Context에 넣지 않는가?

**Option 2: Put Everything in Context (❌ 나쁜 방법)**

```python
# 모든 새로운 정보를 prompt에 추가
prompt = f"""
모든 새로운 뉴스...
모든 업데이트된 문서...
모든 최신 정보...

질문: {user_question}
"""
```

### 문제 1: Context Length 제한

```
GPT-4: 400K tokens
≈ 400,000 / 4 = 100,000 characters
≈ 수백 페이지

충분히 크지만, 모든 정보를 담기에는 부족
```

### 문제 2: Needle in a Haystack

**실험 Setup:**

```python
# Haystack (건초더미): 큰 문서
haystack = long_document  # 예: 50K tokens

# Needle (바늘): 찾을 정보
needle = "The secret code is: 42"

# 다양한 위치에 needle 삽입
for position in [start, middle, end]:
    test_document = insert(haystack, needle, position)
    response = llm(test_document + "\nWhat is the secret code?")
    accuracy = check(response)
```

**GPT-4 Needle in Haystack 결과:**

```
Document Depth (Needle 위치)
    │
100%│ ██▓▓░░░░░░░░░░░░░░
    │ ██▓▓▓▓░░░░░░░░░░░░
 80%│ ███▓▓▓▓░░░░░░░░░░
    │ ███▓▓▓▓▓░░░░░░░░░
 60%│ ████▓▓▓▓▓░░░░░░░░
    │ ████▓▓▓▓▓▓░░░░░░░
 40%│ █████▓▓▓▓▓▓░░░░░░
    │ █████▓▓▓▓▓▓▓░░░░░
 20%│ ██████▓▓▓▓▓▓▓░░░░
    │ ██████▓▓▓▓▓▓▓▓░░░
  0%│ ███████▓▓▓▓▓▓▓▓░░
    └─────────────────────
     0K  10K  20K  30K  40K
     Context Length (tokens)

범례:
██ 90-100% 정확도
▓▓ 70-90%
░░ <70%

관찰:
1. 긴 context에서 성능 저하
2. 중간 부분(20-60%)에서 특히 안좋음
3. 문서 시작/끝에서 상대적으로 나음
```

### 문제 3: 비용

```
Pricing (GPT-4):
- $1 / 1M tokens (대략)

큰 context 사용 시:
- 매 query마다 수십만 tokens
- 비용이 빠르게 증가

예시:
- 100K tokens × 1000 queries = 100M tokens
- 비용: $100
```

## 3.4. RAG의 핵심 아이디어

**해결책:**

```
모든 정보를 넣지 말고,
관련된 (relevant) 정보만 선별해서 넣자!

Naive:
Prompt = Question + All New Info

RAG:
Prompt = Question + Relevant Info Only
                     ↑
                   핵심!
```

**RAG = Retrieval Augmented Generation**

```
Retrieval: 관련 정보 검색
Augmented: Prompt에 추가
Generation: LLM이 답변 생성
```

## 3.5. RAG의 3단계

**전체 프로세스:**

```
┌──────────────────────────────────────┐
│ Step 1: Retrieve                     │
│                                      │
│ Question → Knowledge Base            │
│              ↓                       │
│         Related Documents            │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│ Step 2: Augment                      │
│                                      │
│ Prompt = Question + Related Docs     │
└──────────────────────────────────────┘

┌──────────────────────────────────────┐
│ Step 3: Generate                     │
│                                      │
│ Augmented Prompt → LLM → Answer      │
└──────────────────────────────────────┘
```

**구체적 예시:**

```python
# Step 1: Retrieve
question = "2024년 11월 미국 대선 승자는?"
relevant_docs = retrieve(question, knowledge_base)

# relevant_docs:
"""
2024년 11월 5일, 미국 대선이 실시되었습니다.
승자는 Donald Trump입니다.
선거인단 득표는 312 vs 226 이었습니다.
"""

# Step 2: Augment
augmented_prompt = f"""
관련 정보:
{relevant_docs}

질문: {question}
"""

# Step 3: Generate
answer = llm(augmented_prompt)
# "2024년 11월 미국 대선 승자는 Donald Trump입니다."
```

**핵심:**

```
RAG의 성공 = Retrieval의 품질

Good Retrieval:
- 관련 있는 문서만 가져옴
- LLM이 정확한 답변 생성

Bad Retrieval:
- 관련 없는 문서 가져옴
- LLM이 혼란스러워함
- 틀린 답변 또는 "모름"

→ Retrieval이 가장 중요!
```

---

# 4. Knowledge Base 구축

**Knowledge Base란?**

```
외부 지식을 저장하는 DB

구성:
1. Documents: 원본 문서들
2. Chunks: 문서를 작은 조각으로 분할
3. Embeddings: 각 chunk의 vector 표현
```

## 4.1. Chunking

**왜 Chunking이 필요한가?**

```
문제:
- 전체 문서는 너무 큼
- Embedding model은 제한된 길이만 처리

해결:
- 문서를 작은 chunks로 분할
- 각 chunk를 independently 처리
```

**Chunking Process:**

```python
document = """
Large Language Models (LLMs) are...
(5000 words)
"""

# Chunking
chunks = split_document(
    document,
    chunk_size=500,      # 500 tokens per chunk
    overlap=100          # 100 tokens overlap
)

# Result:
chunks = [
    "Large Language Models (LLMs) are...",  # Chunk 1: 0-500
    "...are transformer-based models...",    # Chunk 2: 400-900
    "...trained on large datasets...",       # Chunk 3: 800-1300
    ...
]
```

**Overlap의 중요성:**

```
No Overlap:
Chunk 1: [0────500]
Chunk 2:           [500────1000]
                   ↑
                문맥 단절!

With Overlap (100 tokens):
Chunk 1: [0────500]
Chunk 2:      [400────900]
               ↑
          100 tokens 중복
          → 문맥 유지!
```

**예시:**

```
원본 문서:
"...Transformer는 attention mechanism을 사용합니다.
Attention은 입력 시퀀스의 각 위치를 다른 모든 위치와..."

No Overlap:
Chunk 1: "...Transformer는 attention mechanism을"
Chunk 2: "사용합니다. Attention은..."
         ↑ 문장이 잘림!

With Overlap:
Chunk 1: "...Transformer는 attention mechanism을 사용합니다."
Chunk 2: "mechanism을 사용합니다. Attention은..."
         ↑ 문맥 유지!
```

## 4.2. Embedding 생성

**Process:**

```python
# 각 chunk를 embedding으로 변환
embeddings = []

for chunk in chunks:
    # Encoder model (보통 BERT-like)
    embedding = encoder(chunk)  # → d-dimensional vector
    embeddings.append(embedding)

# 저장
knowledge_base = {
    "chunks": chunks,
    "embeddings": embeddings  # for fast retrieval
}
```

**Embedding이란?**

```
Text → Vector

예시:
"Transformer is a deep learning model"
→ [0.12, -0.34, 0.56, ..., 0.78]  # 1024-dim vector

특징:
- 의미가 비슷한 text는 비슷한 vector
- Cosine similarity로 비교 가능
```

## 4.3. 하이퍼파라미터

### 1. Embedding Size

```python
embedding_size = 768  # or 1024, 1536, etc.

Trade-off:
- 크면: 더 풍부한 표현, 하지만 메모리/compute 증가
- 작으면: 빠르고 효율적, 하지만 표현력 감소

권장:
- 간단한 문서: 512-768
- 복잡한 문서: 1024-1536
```

### 2. Chunk Size

```python
chunk_size = 500  # tokens

Trade-off:
- 작으면: 정확한 matching, 하지만 문맥 부족
- 크면: 풍부한 문맥, 하지만 noisy

권장:
- 일반적: 300-700 tokens
- 짧은 문서: 200-400
- 긴 문서: 500-1000
```

### 3. Overlap Size

```python
overlap = 100  # tokens

Trade-off:
- 없으면: 저장 효율적, 하지만 문맥 단절
- 크면: 문맥 유지, 하지만 중복 증가

권장:
- 일반적: chunk_size의 10-20%
- 예: chunk_size=500 → overlap=50-100
```

**전체 예시:**

```python
class KnowledgeBase:
    def __init__(
        self,
        embedding_size=1024,
        chunk_size=500,
        overlap=100
    ):
        self.embedding_size = embedding_size
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Encoder model
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Storage
        self.chunks = []
        self.embeddings = []

    def add_document(self, document):
        """문서를 knowledge base에 추가"""
        # 1. Chunking
        chunks = self.split_document(
            document,
            size=self.chunk_size,
            overlap=self.overlap
        )

        # 2. Embedding
        for chunk in chunks:
            embedding = self.encoder.encode(chunk)
            self.chunks.append(chunk)
            self.embeddings.append(embedding)

    def split_document(self, doc, size, overlap):
        """문서를 chunks로 분할"""
        chunks = []
        start = 0

        while start < len(doc):
            end = start + size
            chunk = doc[start:end]
            chunks.append(chunk)
            start += (size - overlap)

        return chunks
```

---

# 5. Retrieval: 2-Stage Approach

**왜 2 stages?**

```
Knowledge Base: 수백만 chunks

1-stage (naive):
- 모든 chunks와 query 비교
- 느림 (O(N) where N = millions)

2-stage (practical):
Stage 1: 빠르게 candidates 추출 (→ ~100)
Stage 2: 정확하게 re-ranking (→ top k)

→ 훨씬 빠르고 정확!
```

**전체 프로세스:**

```
Query: "Transformer attention이란?"

Knowledge Base (1M chunks)
        ↓
┌─────────────────────────────┐
│ Stage 1: Candidate Retrieval │
│ - Fast but rough             │
│ - Semantic similarity        │
│ - BM25 heuristic             │
└─────────────────────────────┘
        ↓
   ~100-500 candidates
        ↓
┌─────────────────────────────┐
│ Stage 2: Re-ranking          │
│ - Slow but accurate          │
│ - Cross-encoder              │
└─────────────────────────────┘
        ↓
     Top k (e.g., 5)
        ↓
  Return to LLM
```

## 5.1. Stage 1: Candidate Retrieval

**목표:**

```
Input:  1M chunks
Output: ~100 potentially relevant chunks

방법:
1. Semantic Similarity (embeddings)
2. Heuristic Search (BM25)
3. Hybrid (둘 다)
```

### Semantic Similarity Search (Bi-encoder)

**핵심 아이디어:**

```
1. Query를 embedding으로 변환
2. 모든 chunk embeddings와 비교
3. Cosine similarity 계산
4. Top candidates 선택
```

**Process:**

```python
def semantic_retrieval(query, knowledge_base, top_k=100):
    """
    Semantic similarity로 candidates 추출
    """
    # 1. Query embedding
    query_emb = encoder(query)  # (d,)

    # 2. 모든 chunk embeddings와 비교
    chunk_embs = knowledge_base.embeddings  # (N, d)

    # 3. Cosine similarity
    similarities = cosine_similarity(query_emb, chunk_embs)

    # 4. Top k
    top_indices = argsort(similarities)[-top_k:]
    candidates = [knowledge_base.chunks[i] for i in top_indices]

    return candidates
```

**Cosine Similarity:**

```python
def cosine_similarity(a, b):
    """
    두 벡터의 cosine similarity 계산

    cos(θ) = (a · b) / (||a|| × ||b||)

    범위: [-1, 1]
    - 1: 완전히 같은 방향
    - 0: 직교
    - -1: 완전히 반대 방향
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    return dot_product / (norm_a * norm_b)
```

**예시:**

```python
query = "Transformer attention mechanism"
query_emb = encoder(query)  # [0.1, 0.5, -0.3, ...]

chunks = [
    "Attention is all you need",           # Chunk 1
    "The cat sat on the mat",              # Chunk 2
    "Self-attention computes...",          # Chunk 3
]

chunk_embs = [
    [0.12, 0.48, -0.29, ...],  # Chunk 1 embedding
    [-0.5, 0.2, 0.8, ...],     # Chunk 2 embedding
    [0.11, 0.52, -0.31, ...],  # Chunk 3 embedding
]

# Cosine similarities
similarities = [
    cosine_sim(query_emb, chunk_embs[0]),  # 0.95 (high!)
    cosine_sim(query_emb, chunk_embs[1]),  # 0.12 (low)
    cosine_sim(query_emb, chunk_embs[2]),  # 0.93 (high!)
]

# Top candidates
candidates = [chunks[0], chunks[2]]  # 관련 있는 것만!
```

**Bi-encoder란?**

```
Query와 Chunk를 independently encode

        Query
          ↓
      Encoder₁
          ↓
    Query Embedding ─┐
                     │
                     ├→ Cosine Similarity
                     │
   Chunk Embedding ─┘
          ↑
      Encoder₂
          ↑
        Chunk

특징:
- 두 encoder가 독립적 (보통 같은 model 사용)
- 미리 chunk embeddings 계산 가능
- 빠름! (query time에 query만 encode)
```

**SBERT (Sentence-BERT):**

```
BERT 기반 embedding model

Training:
- Contrastive learning
- Similar sentences → similar embeddings
- Dissimilar sentences → different embeddings

Loss:
L = max(0, d(positive) - d(negative) + margin)

여기서:
- d(·) = distance between embeddings
- positive = relevant pair
- negative = irrelevant pair
- margin = separation (e.g., 0.5)
```

**Approximate Nearest Neighbor (ANN):**

```
문제:
- Knowledge base가 크면 (1M+ chunks)
- Linear search는 느림 (O(N))

해결:
- ANN algorithms
- 정확도를 약간 희생하고 속도 향상

방법:
- HNSW (Hierarchical Navigable Small World)
- FAISS (Facebook AI Similarity Search)
- Annoy (Spotify)

복잡도:
- Linear: O(N)
- ANN: O(log N) ~ O(√N)
```

**FAISS 예시:**

```python
import faiss

# Knowledge base embeddings (N=1M, d=768)
embeddings = np.array(knowledge_base.embeddings)  # (1M, 768)

# Build FAISS index
index = faiss.IndexFlatL2(768)  # L2 distance
index.add(embeddings)

# Query
query_emb = encoder(query)  # (768,)

# Fast search (top 100)
k = 100
distances, indices = index.search(query_emb, k)

# Get candidates
candidates = [knowledge_base.chunks[i] for i in indices[0]]
```

### Heuristic Search (BM25)

**문제:**

```
Semantic similarity만으로는 부족한 경우:

Query: "Where is Cuddly?"

Semantic:
- "Huggy is in the box" (teddy bear, semantically similar!)
- "The location of Snuggly" (location, semantically similar!)

하지만:
- "Cuddly"라는 키워드가 없음!

→ Keyword matching 필요!
```

**BM25란?**

```
BM25 (Best Matching 25):
- Heuristic ranking function
- Keyword overlap 기반
- Search engines에서 오래 사용됨

핵심:
- Query와 document의 word overlap 측정
- TF-IDF의 개선 버전
```

**BM25 Score:**

```
BM25(Q, D) = Σ IDF(qᵢ) × (f(qᵢ, D) × (k₁ + 1)) /
                         (f(qᵢ, D) + k₁ × (1 - b + b × |D|/avgdl))

여기서:
- Q: query
- D: document
- qᵢ: query term i
- f(qᵢ, D): term frequency of qᵢ in D
- IDF(qᵢ): inverse document frequency
- k₁, b: tuning parameters (보통 k₁=1.5, b=0.75)
- |D|: document length
- avgdl: average document length
```

**간단히:**

```python
def bm25_score(query, document):
    """
    BM25 score 계산 (simplified)
    """
    score = 0

    for term in query:
        if term in document:
            # Term frequency
            tf = document.count(term)

            # Document frequency
            df = count_docs_with(term, all_docs)
            idf = log((N - df + 0.5) / (df + 0.5))

            # BM25 component
            score += idf * tf / (tf + k1)

    return score
```

**예시:**

```python
query = "Where is Cuddly?"
query_terms = ["where", "is", "cuddly"]

docs = [
    "Cuddly is in the toybox",        # Doc 1
    "Huggy is in the bedroom",        # Doc 2
    "The location of Snuggly",        # Doc 3
]

# BM25 scores
scores = [
    bm25(query_terms, docs[0]),  # High! (has "cuddly", "is")
    bm25(query_terms, docs[1]),  # Low (only "is")
    bm25(query_terms, docs[2]),  # Low (no "cuddly")
]

# Top result
best_doc = docs[0]  # "Cuddly is in the toybox"
```

**Semantic vs BM25:**

| 측면 | Semantic Similarity | BM25 |
|------|---------------------|------|
| 기반 | Embeddings (의미) | Keywords (단어) |
| 장점 | 의미 이해 | Exact match 보장 |
| 단점 | Keyword miss 가능 | 의미 이해 안됨 |
| 예시 | "car" ≈ "automobile" | "car" ≠ "automobile" |

### Hybrid Approach

**아이디어:**

```
둘 다 사용하자!

Semantic: 의미적으로 관련된 문서
BM25: 키워드 포함 문서

Combined: 두 가지 모두 고려
```

**Implementation:**

```python
def hybrid_retrieval(query, knowledge_base, top_k=100, alpha=0.5):
    """
    Semantic + BM25 hybrid retrieval

    Args:
        alpha: weight for semantic (1-alpha for BM25)
    """
    # 1. Semantic scores
    query_emb = encoder(query)
    semantic_scores = cosine_similarity(query_emb, knowledge_base.embeddings)

    # 2. BM25 scores
    bm25_scores = [bm25(query, chunk) for chunk in knowledge_base.chunks]

    # 3. Normalize (0-1 range)
    semantic_scores = (semantic_scores - min(semantic_scores)) / \
                      (max(semantic_scores) - min(semantic_scores))
    bm25_scores = (bm25_scores - min(bm25_scores)) / \
                  (max(bm25_scores) - min(bm25_scores))

    # 4. Combine
    combined_scores = alpha * semantic_scores + (1 - alpha) * bm25_scores

    # 5. Top k
    top_indices = argsort(combined_scores)[-top_k:]
    candidates = [knowledge_base.chunks[i] for i in top_indices]

    return candidates
```

**Alpha 조정:**

```python
# Use case별 alpha 선택

# Case 1: 의미 중심 (일반적인 질문)
alpha = 0.7
query = "What is machine learning?"
→ Semantic 70%, BM25 30%

# Case 2: 키워드 중심 (특정 이름/코드)
alpha = 0.3
query = "Where is function compute_loss?"
→ Semantic 30%, BM25 70%

# Case 3: Balanced
alpha = 0.5
query = "How does Transformer work?"
→ Semantic 50%, BM25 50%
```

## 5.2. Stage 2: Re-ranking

**목표:**

```
Input:  ~100 candidates
Output: Top k (e.g., 5) most relevant

방법:
- Cross-encoder
- Query와 chunk를 함께 encode
- 더 정확하지만 느림
```

**Bi-encoder vs Cross-encoder:**

```
Bi-encoder (Stage 1):
Query → Encoder → Embedding ─┐
                              ├→ Similarity
Chunk → Encoder → Embedding ─┘

특징:
- 독립적으로 encode
- 빠름 (미리 chunk 계산 가능)
- 상호작용 없음

Cross-encoder (Stage 2):
[Query, Chunk] → Encoder → Relevance Score
        ↑
    함께 encode!

특징:
- 함께 encode
- 느림 (매번 계산)
- Query-chunk 상호작용 가능!
```

**Cross-encoder 구조:**

```python
class CrossEncoder(nn.Module):
    def __init__(self, model_name="bert-base"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(768, 1)  # Relevance score

    def forward(self, query, chunk):
        """
        Query와 chunk를 함께 encode
        """
        # 1. Concatenate
        text = f"[CLS] {query} [SEP] {chunk} [SEP]"
        tokens = tokenizer(text)

        # 2. BERT encoding (attention 발생!)
        output = self.bert(tokens)
        cls_embedding = output[0]  # [CLS] token

        # 3. Relevance score
        score = self.classifier(cls_embedding)

        return score  # Scalar
```

**왜 더 정확한가?**

```
Bi-encoder:
Query embedding: [0.1, 0.5, -0.3, ...]
Chunk embedding: [0.12, 0.48, -0.29, ...]
Similarity: dot product (no interaction!)

Cross-encoder:
[Query, Chunk] → BERT → Attention!
                   ↑
    Query의 "Transformer"가
    Chunk의 "attention"에 attend

→ 더 풍부한 interaction
→ 더 정확한 relevance
```

**사용 예시:**

```python
def rerank(query, candidates, top_k=5):
    """
    Cross-encoder로 re-ranking
    """
    # 1. Cross-encoder scores
    scores = []
    for candidate in candidates:
        score = cross_encoder(query, candidate)
        scores.append(score)

    # 2. Sort
    ranked_indices = argsort(scores)[::-1]  # Descending

    # 3. Top k
    top_docs = [candidates[i] for i in ranked_indices[:top_k]]

    return top_docs
```

**전체 Retrieval 예시:**

```python
def two_stage_retrieval(query, knowledge_base, top_k=5):
    """
    2-stage retrieval
    """
    # Stage 1: Candidate Retrieval (~100)
    candidates = hybrid_retrieval(
        query,
        knowledge_base,
        top_k=100,
        alpha=0.5
    )

    # Stage 2: Re-ranking (→ 5)
    final_docs = rerank(
        query,
        candidates,
        top_k=top_k
    )

    return final_docs

# 사용
query = "What is Transformer attention?"
top_docs = two_stage_retrieval(query, kb, top_k=5)

# RAG
augmented_prompt = f"""
관련 문서:
{top_docs}

질문: {query}
"""

answer = llm(augmented_prompt)
```

---

# 6. Retrieval 개선 기법

## 6.1. HyDE (Hypothetical Document Embeddings)

**문제:**

```
Query와 Document의 형식이 다름:

Query: "Transformer attention이란?"
→ 짧음, 질문 형식

Document: "Attention mechanism은 입력 시퀀스의..."
→ 길고, 서술 형식

→ Embedding이 다른 space에 있을 수 있음!
```

**HyDE 아이디어:**

```
Query를 직접 embed하지 말고,
가상의 답변(Hypothetical Document)을 생성하고
그것을 embed하자!
```

**Process:**

```python
def hyde_retrieval(query, knowledge_base, top_k=5):
    """
    HyDE: Hypothetical Document Embeddings
    """
    # 1. LLM으로 가상 문서 생성
    fake_doc = llm(f"""
    다음 질문에 대한 답변을 작성하세요:
    {query}
    """)

    # 예시:
    # Query: "Transformer attention이란?"
    # fake_doc: "Transformer attention은 입력 시퀀스의
    #            각 위치가 다른 모든 위치를 참조할 수 있게..."

    # 2. 가상 문서를 embed
    fake_doc_emb = encoder(fake_doc)

    # 3. Knowledge base에서 검색
    similarities = cosine_similarity(fake_doc_emb, kb.embeddings)
    top_indices = argsort(similarities)[-top_k:]

    # 4. Top documents
    docs = [kb.chunks[i] for i in top_indices]

    return docs
```

**왜 작동하는가?**

```
Query: "Transformer attention이란?"
→ 짧고 추상적

Hypothetical Document:
"Transformer attention은 입력 시퀀스의 각 위치가..."
→ 길고 구체적 (실제 문서와 비슷!)

→ Embedding space에서 더 가까움
→ 더 나은 retrieval
```

**장단점:**

```
장점:
- Query-document gap 해소
- 더 나은 semantic matching

단점:
- 추가 LLM call (비용, 시간)
- Hallucination 가능 (틀린 정보 생성)

→ Use case에 따라 선택
```

## 6.2. Contextual Retrieval

**문제:**

```
Naive chunking:
- 문서를 단순히 500 tokens씩 자름
- 문맥이 사라질 수 있음

예시:
Document: "...Apple의 CEO는 Tim Cook입니다.
           그는 2011년부터 CEO를 맡았습니다..."

Chunk 1: "...Apple의 CEO는 Tim Cook입니다."
Chunk 2: "그는 2011년부터 CEO를 맡았습니다..."
          ↑
       "그"가 누구인지 모름!
```

**Contextual Retrieval 아이디어:**

```
각 chunk에 context를 추가하자!

Chunk + Context:
"이 문서는 Apple 회사에 대한 내용입니다.
 그는 2011년부터 CEO를 맡았습니다..."
 ↑
 Context 추가!
```

**Process:**

```python
def contextual_retrieval(document, chunk_size=500):
    """
    각 chunk에 context 추가
    """
    # 1. 기본 chunking
    chunks = split_document(document, size=chunk_size)

    # 2. 각 chunk에 context 추가
    contextualized_chunks = []

    for chunk in chunks:
        # LLM으로 context 생성
        context = llm(f"""
        전체 문서:
        {document}

        다음 chunk를 이해하는데 필요한 짧은 context를 작성하세요:
        {chunk}

        Context (1-2 문장):
        """)

        # Chunk + context
        contextualized = f"{context}\n\n{chunk}"
        contextualized_chunks.append(contextualized)

    return contextualized_chunks
```

**예시:**

```python
# 원본 문서
document = """
Apple Inc.는 미국의 기술 회사입니다.
CEO는 Tim Cook입니다.
그는 2011년부터 CEO를 맡았습니다.
Apple의 본사는 Cupertino에 있습니다.
"""

# Naive chunking
chunk_2 = "그는 2011년부터 CEO를 맡았습니다."

# Contextual Retrieval
context = llm(...)  # → "이 문서는 Apple과 CEO Tim Cook에 대한 내용입니다."

contextualized_chunk = """
이 문서는 Apple과 CEO Tim Cook에 대한 내용입니다.

그는 2011년부터 CEO를 맡았습니다.
"""

→ "그"가 Tim Cook임을 알 수 있음!
```

**문제: 비용**

```
Contextual Retrieval:
- 각 chunk마다 LLM call
- 문서 1개 = 100 chunks → 100 LLM calls!

비용:
- 매우 비쌈

해결:
→ Prompt Caching!
```

## 6.3. Prompt Caching

**문제:**

```
Contextual Retrieval:
- 각 chunk마다 LLM call
- 하지만 "전체 문서"는 항상 같음!

Chunk 1 prompt:
"전체 문서: {document}  ← 같음!
 Chunk: {chunk_1}        ← 다름"

Chunk 2 prompt:
"전체 문서: {document}  ← 같음!
 Chunk: {chunk_2}        ← 다름"

→ "전체 문서"를 매번 처리하는게 낭비!
```

**Prompt Caching 아이디어:**

```
같은 prefix는 한 번만 처리하고,
activations를 cache!

First call:
"전체 문서: {document}" → Compute → Cache activations

Subsequent calls:
"전체 문서: {document}" → Lookup cache (빠름!)
"Chunk: {chunk_i}"      → Compute만 이것만
```

**작동 원리:**

```
LLM (Decoder-only):
- Autoregressive generation
- Left-to-right processing

Prompt: "A B C D E"

Forward pass:
1. Process A → hidden₁
2. Process B (attending to A) → hidden₂
3. Process C (attending to A, B) → hidden₃
...

If prefix "A B C" is same:
→ Cache hidden₁, hidden₂, hidden₃
→ Reuse for next prompt!
```

**구현:**

```python
# API에서 자동 지원

# OpenAI 예시
response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {
            "role": "system",
            "content": f"전체 문서: {document}"  # ← Cached
        },
        {
            "role": "user",
            "content": f"Chunk: {chunk}"  # ← Computed
        }
    ]
)

# Subsequent calls:
# - "전체 문서" part: cached (90% off!)
# - "Chunk" part: computed (full price)
```

**Pricing 예시:**

```
OpenAI GPT-4:
- Regular input: $0.01 / 1K tokens
- Cached input:  $0.001 / 1K tokens (10배 저렴!)

예시:
- Document: 10K tokens
- 100 chunks

No caching:
- 100 × 10K = 1M tokens
- Cost: $10

With caching:
- First call: 10K tokens (full price) = $0.10
- 99 calls: 99 × 10K (cached) = $0.99
- Total: $1.09 (90% 절약!)
```

**전체 예시:**

```python
def contextual_retrieval_with_caching(document, chunks):
    """
    Contextual retrieval + prompt caching
    """
    contextualized = []

    for chunk in chunks:
        # Prompt caching 활용
        context = llm(
            system=f"전체 문서: {document}",  # ← Cached!
            user=f"다음 chunk의 context 작성: {chunk}"
        )

        contextualized.append(f"{context}\n\n{chunk}")

    return contextualized
```

---

# 7. Retrieval Evaluation

**평가가 필요한 이유:**

```
Retrieval 시스템 개선 시:
- 어떤 방법이 더 좋은지?
- Hyperparameter 최적화는?

→ 정량적 평가 필요!
```

**Setup:**

```
1. Test set 준비:
   - Queries
   - Ground truth relevant documents

2. Retrieval 수행:
   - Query → Retriever → Top k documents

3. Evaluate:
   - Retrieved vs Ground truth
   - Metrics 계산
```

## 7.1. NDCG (Normalized Discounted Cumulative Gain)

**핵심 아이디어:**

```
Ranking을 평가할 때:
1. 관련 있는 문서를 retrieve했는가?
2. 높은 순위에 있는가?

NDCG:
- 높은 순위에 관련 문서가 있으면 high score
- 낮은 순위에 있으면 discounted
```

**DCG (Discounted Cumulative Gain):**

```
DCG@k = Σ (rel_i / log₂(i + 1))
        i=1 to k

여기서:
- rel_i: position i의 relevance (0 or 1)
- log₂(i + 1): discount factor

Discount:
- Position 1: 1 / log₂(2) = 1.0
- Position 2: 1 / log₂(3) ≈ 0.63
- Position 3: 1 / log₂(4) = 0.5
- Position 4: 1 / log₂(5) ≈ 0.43
...

→ 뒤로 갈수록 discount!
```

**NDCG (Normalized DCG):**

```
NDCG@k = DCG@k / IDCG@k

여기서:
- IDCG: Ideal DCG (최적의 ranking)

Normalization:
- 0 ≤ NDCG ≤ 1
- NDCG = 1: Perfect ranking
- NDCG = 0: 관련 문서 없음
```

**예시:**

```python
# Test case
query = "Transformer attention"
retrieved = ["Doc A", "Doc B", "Doc C", "Doc D", "Doc E"]  # Top 5

# Ground truth relevance
relevance = [
    1,  # Doc A: relevant
    0,  # Doc B: not relevant
    1,  # Doc C: relevant
    0,  # Doc D: not relevant
    1,  # Doc E: relevant
]

# DCG 계산
DCG = (1 / log₂(2)) + (0 / log₂(3)) + (1 / log₂(4)) +
      (0 / log₂(5)) + (1 / log₂(6))
    = 1.0 + 0 + 0.5 + 0 + 0.387
    = 1.887

# IDCG (ideal: 모든 relevant를 앞에)
# Ideal ranking: [Doc A, Doc C, Doc E, Doc B, Doc D]
# Relevance:     [1, 1, 1, 0, 0]

IDCG = (1 / log₂(2)) + (1 / log₂(3)) + (1 / log₂(4)) + 0 + 0
     = 1.0 + 0.631 + 0.5
     = 2.131

# NDCG
NDCG = DCG / IDCG = 1.887 / 2.131 ≈ 0.885

→ 88.5% 점수 (꽤 좋음!)
```

**구현:**

```python
import numpy as np

def dcg_at_k(relevances, k):
    """
    DCG@k 계산

    Args:
        relevances: relevance scores (0 or 1)
        k: top k
    """
    relevances = np.array(relevances)[:k]
    positions = np.arange(1, len(relevances) + 1)

    # DCG = Σ rel_i / log₂(i + 1)
    dcg = np.sum(relevances / np.log2(positions + 1))

    return dcg

def ndcg_at_k(relevances, k):
    """
    NDCG@k 계산
    """
    # DCG
    dcg = dcg_at_k(relevances, k)

    # IDCG (ideal: sort relevances descending)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)

    # NDCG
    if idcg == 0:
        return 0
    return dcg / idcg

# 예시
relevances = [1, 0, 1, 0, 1]
ndcg = ndcg_at_k(relevances, k=5)
print(f"NDCG@5: {ndcg:.3f}")  # 0.885
```

## 7.2. MRR (Mean Reciprocal Rank)

**핵심 아이디어:**

```
첫 번째 관련 문서의 위치만 고려

MRR = 1 / (첫 관련 문서의 rank)
```

**예시:**

```python
# Query 1
retrieved_1 = ["Doc A", "Doc B", "Doc C", "Doc D"]
relevance_1 = [0, 0, 1, 0]  # 첫 관련 문서: position 3

RR_1 = 1 / 3 ≈ 0.333

# Query 2
retrieved_2 = ["Doc X", "Doc Y", "Doc Z"]
relevance_2 = [1, 0, 0]  # 첫 관련 문서: position 1

RR_2 = 1 / 1 = 1.0

# Query 3
retrieved_3 = ["Doc M", "Doc N"]
relevance_3 = [0, 0]  # 관련 문서 없음

RR_3 = 0

# MRR (Mean)
MRR = (RR_1 + RR_2 + RR_3) / 3
    = (0.333 + 1.0 + 0) / 3
    ≈ 0.444
```

**구현:**

```python
def mrr(queries_relevances):
    """
    MRR (Mean Reciprocal Rank) 계산

    Args:
        queries_relevances: List of relevance lists
            [[1, 0, 0], [0, 1, 0], ...]
    """
    rrs = []

    for relevances in queries_relevances:
        # 첫 번째 관련 문서 찾기
        for i, rel in enumerate(relevances):
            if rel == 1:
                rr = 1.0 / (i + 1)
                rrs.append(rr)
                break
        else:
            # 관련 문서 없음
            rrs.append(0)

    # Mean
    return np.mean(rrs)

# 예시
queries_relevances = [
    [0, 0, 1, 0],  # First relevant at position 3
    [1, 0, 0],     # First relevant at position 1
    [0, 0, 0],     # No relevant document
]

mrr_score = mrr(queries_relevances)
print(f"MRR: {mrr_score:.3f}")  # 0.444
```

**NDCG vs MRR:**

| 측면 | NDCG | MRR |
|------|------|-----|
| 고려 | 모든 positions | 첫 관련 문서만 |
| Ranking | 중요 | 덜 중요 |
| 계산 | 복잡 | 간단 |
| 사용 | 상세 평가 | 빠른 평가 |

## 7.3. Precision@k와 Recall@k

**Precision@k:**

```
Precision@k = (Top k 중 관련 문서 수) / k

예시:
Top 5: [Doc A, Doc B, Doc C, Doc D, Doc E]
Relevant: [Doc A, Doc C, Doc E]  # 3개

Precision@5 = 3 / 5 = 0.6 (60%)

의미:
- Retrieve한 것 중 얼마나 정확한가?
```

**Recall@k:**

```
Recall@k = (Top k 중 관련 문서 수) / (전체 관련 문서 수)

예시:
Top 5: [Doc A, Doc C, Doc E]  # 3개 retrieve
Ground truth: [Doc A, Doc C, Doc E, Doc F, Doc G]  # 5개 실제로 관련

Recall@5 = 3 / 5 = 0.6 (60%)

의미:
- 전체 관련 문서 중 얼마나 찾았는가?
```

**Trade-off:**

```
k 증가:
- Precision 감소 (더 많이 가져오니 정확도 감소)
- Recall 증가 (더 많이 찾음)

예시:
Precision@3 = 0.8, Recall@3 = 0.4
Precision@10 = 0.5, Recall@10 = 0.8
```

**구현:**

```python
def precision_at_k(relevances, k):
    """
    Precision@k 계산
    """
    top_k_relevances = relevances[:k]
    return sum(top_k_relevances) / k

def recall_at_k(relevances, total_relevant, k):
    """
    Recall@k 계산

    Args:
        relevances: retrieved documents' relevance
        total_relevant: total number of relevant docs
        k: top k
    """
    top_k_relevances = relevances[:k]
    return sum(top_k_relevances) / total_relevant

# 예시
retrieved_relevances = [1, 0, 1, 0, 1]  # Top 5
total_relevant = 5  # Actually 5 relevant docs exist

precision = precision_at_k(retrieved_relevances, k=5)
recall = recall_at_k(retrieved_relevances, total_relevant, k=5)

print(f"Precision@5: {precision:.2f}")  # 0.60
print(f"Recall@5: {recall:.2f}")        # 0.60
```

## 7.4. Benchmarks

**MTEB (Massive Text Embedding Benchmark):**

```
Retrieval task benchmark

구성:
- 56 datasets
- 112 languages
- 8 task types:
  - Retrieval
  - Classification
  - Clustering
  - Semantic Textual Similarity
  ...

사용:
- Embedding model 평가
- NDCG, MRR, Recall@k 등 metric 포함
```

**사용 예시:**

```python
from mteb import MTEB
from sentence_transformers import SentenceTransformer

# Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Evaluation
evaluation = MTEB(tasks=["NFCorpus"])
results = evaluation.run(model)

# Results
print(results)
# {
#   "NFCorpus": {
#     "NDCG@10": 0.325,
#     "Recall@10": 0.252,
#     "Precision@10": 0.058
#   }
# }
```

---

# 8. Tool Calling

**Vanilla LLM의 또 다른 약점:**

```
Problem: All talk, no action

User: "내일 오전 9시에 회의 예약해줘"
LLM: "회의를 예약하겠습니다" ← 실제로 안함!

User: "주변 식당 찾아줘"
LLM: "죄송합니다. 실시간 정보에 접근할 수 없습니다"

→ Action 불가!
```

## 8.1. Tool Calling이란?

**정의 (IBM):**

```
Tool calling allows autonomous systems to complete
complex tasks by dynamically accessing and acting
upon external resources.

핵심:
1. Complete tasks (작업 완료)
2. External resources (외부 자원)

→ LLM이 외부 API/도구를 사용!
```

**예시:**

```
User: "주변 곰 인형 찾아줘"

Without Tools:
LLM: "죄송합니다. 실시간 정보가 없습니다"

With Tools:
LLM:
1. find_teddy_bear(location="Stanford") 호출
2. API 결과 받음: [{"name": "Cuddly", "distance": "0.5 miles"}, ...]
3. 결과를 자연어로 변환
"Cuddly가 0.5마일 거리에 있습니다!"

→ 실제로 검색함!
```

## 8.2. Function Calling

**Structured Data as Functions:**

```
Unstructured: RAG에서 다룸
- 문서, 뉴스, 기사 등

Structured: Tool Calling에서 다룸
- Table, API responses
- Input/Output 관계가 명확
- Function으로 표현 가능
```

**예시:**

```python
# Structured data as function

def find_teddy_bear(location: str) -> List[Dict]:
    """
    주변 곰 인형을 찾습니다.

    Args:
        location: 현재 위치

    Returns:
        곰 인형 목록
    """
    # API call
    response = requests.get(
        "https://api.teddybear.com/search",
        params={"location": location}
    )

    return response.json()
```

**Language Choice:**

```
Tool Calling에서는 Python을 많이 사용:
- 읽기 쉬움
- 명확한 type hints
- LLM이 잘 이해함

하지만:
- 다른 언어도 가능 (JavaScript, Java, etc.)
```

## 8.3. Tool Calling 예시

**예시: Teddy Bear Finder**

**1. Function Definition:**

```python
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class TeddyBear:
    """곰 인형 정보"""
    name: str
    location: str
    distance: float
    price: float

def find_teddy_bear(location: str, max_distance: float = 5.0) -> List[TeddyBear]:
    """
    주변 곰 인형을 찾습니다.

    Args:
        location: 현재 위치 (예: "Stanford, CA")
        max_distance: 최대 거리 (miles)

    Returns:
        가까운 곰 인형 목록

    Example:
        >>> bears = find_teddy_bear("Stanford, CA", max_distance=2.0)
        >>> print(bears[0].name)
        'Cuddly'
    """
    # Backend API call
    api_url = "https://api.teddybear.com/search"

    response = requests.get(
        api_url,
        params={
            "location": location,
            "radius": max_distance,
            "category": "teddy_bear"
        }
    )

    # Parse response
    data = response.json()

    # Convert to TeddyBear objects
    bears = [
        TeddyBear(
            name=item["name"],
            location=item["address"],
            distance=item["distance_miles"],
            price=item["price_usd"]
        )
        for item in data["results"]
    ]

    # Sort by distance
    bears.sort(key=lambda b: b.distance)

    return bears
```

**2. Tool Definition for LLM:**

```python
# LLM에게 제공할 tool specification

tool_spec = {
    "name": "find_teddy_bear",
    "description": "주변 곰 인형을 찾습니다. 위치 기반으로 가까운 곰 인형을 검색할 때 사용하세요.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "현재 위치 (예: 'Stanford, CA')"
            },
            "max_distance": {
                "type": "number",
                "description": "최대 거리 (miles). 기본값: 5.0"
            }
        },
        "required": ["location"]
    }
}

# LLM은 이 spec만 봄!
# 실제 구현 코드는 숨겨짐
```

**3. Tool Calling Process:**

```
┌─────────────────────────────────────┐
│ Step 1: User Query                  │
├─────────────────────────────────────┤
│ User: "Find teddy bear near me"     │
└─────────────────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│ Step 2: LLM Decides to Use Tool     │
├─────────────────────────────────────┤
│ LLM (internal):                     │
│ "User wants to find teddy bear.     │
│  I need location info.              │
│  → Use find_teddy_bear tool!"       │
│                                     │
│ Tool call:                          │
│ {                                   │
│   "tool": "find_teddy_bear",        │
│   "args": {                         │
│     "location": "Stanford, CA",     │
│     "max_distance": 5.0             │
│   }                                 │
│ }                                   │
└─────────────────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│ Step 3: Execute Tool                │
├─────────────────────────────────────┤
│ result = find_teddy_bear(           │
│     location="Stanford, CA",        │
│     max_distance=5.0                │
│ )                                   │
│                                     │
│ result = [                          │
│   TeddyBear(name="Cuddly",          │
│             distance=0.5, ...),     │
│   TeddyBear(name="Snuggly",         │
│             distance=1.2, ...)      │
│ ]                                   │
└─────────────────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│ Step 4: LLM Formats Response        │
├─────────────────────────────────────┤
│ LLM receives tool result:           │
│ [{"name": "Cuddly", ...}, ...]      │
│                                     │
│ LLM generates natural language:     │
│ "주변에 두 개의 곰 인형을 찾았습니다:   │
│  1. Cuddly (0.5 마일)               │
│  2. Snuggly (1.2 마일)              │
│  가장 가까운 것은 Cuddly입니다."      │
└─────────────────────────────────────┘
             ↓
┌─────────────────────────────────────┐
│ Step 5: User sees final answer      │
└─────────────────────────────────────┘
```

**4. Implementation Example:**

```python
import anthropic

client = anthropic.Anthropic()

# Tool definitions
tools = [
    {
        "name": "find_teddy_bear",
        "description": "주변 곰 인형을 찾습니다",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "현재 위치"
                },
                "max_distance": {
                    "type": "number",
                    "description": "최대 거리 (miles)"
                }
            },
            "required": ["location"]
        }
    }
]

# User query
user_message = "Find teddy bear near Stanford"

# LLM call with tools
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": user_message}]
)

# Check if LLM wants to use tool
if response.stop_reason == "tool_use":
    tool_use = response.content[0]

    # Extract tool name and arguments
    tool_name = tool_use.name  # "find_teddy_bear"
    tool_input = tool_use.input  # {"location": "Stanford, CA", ...}

    # Execute tool
    if tool_name == "find_teddy_bear":
        result = find_teddy_bear(**tool_input)

    # Send result back to LLM
    final_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=[
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": str(result)
                    }
                ]
            }
        ]
    )

    print(final_response.content[0].text)
```

---

# 9. 요약

## 핵심 개념

### 1. RAG (Retrieval Augmented Generation)

```
문제: Knowledge cutoff

해결:
1. Retrieve: Knowledge base에서 관련 문서 검색
2. Augment: Prompt에 추가
3. Generate: LLM이 답변 생성

핵심: Retrieval 품질이 가장 중요!
```

### 2. Knowledge Base 구축

```
Document → Chunks (500 tokens) → Embeddings

하이퍼파라미터:
- Embedding size: 768-1536
- Chunk size: 300-700 tokens
- Overlap: 10-20% of chunk size
```

### 3. 2-Stage Retrieval

```
Stage 1: Candidate Retrieval
- Semantic similarity (Bi-encoder)
- BM25 heuristic
- Hybrid approach
- 1M chunks → ~100 candidates

Stage 2: Re-ranking
- Cross-encoder
- Query-chunk interaction
- ~100 candidates → top k
```

### 4. Evaluation Metrics

```
NDCG@k: Ranking 품질 (0-1)
MRR: 첫 관련 문서 위치
Precision@k: 정확도
Recall@k: 재현율
```

### 5. Tool Calling

```
LLM이 external APIs/tools 사용

Process:
1. LLM decides to use tool
2. Execute tool
3. Return result to LLM
4. LLM formats natural language response

→ Action 가능!
```

## 실전 체크리스트

**RAG 시스템 구축 시:**

1. **Knowledge Base**
   - [ ] 문서 수집
   - [ ] Chunking strategy 결정
   - [ ] Embedding model 선택 (SBERT)
   - [ ] Hyperparameters 조정

2. **Retrieval**
   - [ ] Stage 1: Hybrid retrieval (Semantic + BM25)
   - [ ] Stage 2: Cross-encoder re-ranking
   - [ ] Top k parameter 설정
   - [ ] ANN index 구축 (FAISS)

3. **Evaluation**
   - [ ] Test set 준비 (queries + ground truth)
   - [ ] NDCG@k 계산
   - [ ] MRR 계산
   - [ ] Precision/Recall@k 계산
   - [ ] Hyperparameter tuning

4. **개선 기법**
   - [ ] HyDE 적용 검토
   - [ ] Contextual Retrieval 적용
   - [ ] Prompt Caching 설정
   - [ ] 비용 최적화

5. **Tool Calling (필요시)**
   - [ ] Tool specification 작성
   - [ ] Function implementation
   - [ ] Error handling
   - [ ] Testing

---

# 10. 중요 용어 정리

### RAG 관련

**RAG (Retrieval Augmented Generation)**
- LLM의 knowledge cutoff 문제를 해결하기 위해 외부 knowledge base에서 관련 정보를 검색하여 prompt에 추가하는 기법

**Knowledge Cutoff**
- LLM이 학습한 데이터의 마지막 시점. 그 이후의 정보는 모델이 알지 못함

**Knowledge Base**
- RAG에서 사용하는 외부 지식 저장소. 문서, chunks, embeddings로 구성됨

**Chunking**
- 긴 문서를 작은 조각(chunks)으로 분할하는 과정. 보통 500 tokens 정도

**Overlap**
- 인접한 chunks 사이의 중복 영역. 문맥 유지를 위해 10-20% 정도 설정

### Retrieval 관련

**Bi-encoder**
- Query와 document를 독립적으로 encode하는 방식. 빠르지만 상호작용 없음

**Cross-encoder**
- Query와 document를 함께 encode하는 방식. 느리지만 정확함

**Semantic Similarity**
- Embedding vector 간의 의미적 유사도. Cosine similarity로 측정

**BM25 (Best Matching 25)**
- Keyword 기반 heuristic ranking function. TF-IDF의 개선 버전

**Hybrid Retrieval**
- Semantic similarity와 BM25를 결합한 retrieval 방법

**ANN (Approximate Nearest Neighbor)**
- 정확도를 약간 희생하고 속도를 향상시킨 nearest neighbor search 알고리즘

**SBERT (Sentence-BERT)**
- BERT 기반의 문장 embedding model. Contrastive learning으로 학습됨

**2-Stage Retrieval**
- Stage 1 (Candidate Retrieval): 빠르게 ~100개 후보 추출
- Stage 2 (Re-ranking): 정확하게 top k 선택

### Retrieval 개선 기법

**HyDE (Hypothetical Document Embeddings)**
- Query 대신 LLM이 생성한 가상의 답변을 embed하여 retrieval하는 기법

**Contextual Retrieval**
- 각 chunk에 문서 전체의 context를 추가하여 문맥을 유지하는 기법

**Prompt Caching**
- 같은 prefix를 가진 prompt의 activations를 cache하여 비용을 절감하는 기법

**Needle in a Haystack**
- 긴 context 내에서 특정 정보를 찾는 능력을 평가하는 실험

### Evaluation Metrics

**NDCG (Normalized Discounted Cumulative Gain)**
- Ranking 품질을 평가하는 metric. 높은 순위의 관련 문서에 더 높은 가중치 (0-1)

**DCG (Discounted Cumulative Gain)**
- 위치별로 discount factor를 적용한 cumulative gain

**IDCG (Ideal DCG)**
- 최적의 ranking을 가정한 DCG. NDCG 계산 시 정규화에 사용

**MRR (Mean Reciprocal Rank)**
- 첫 번째 관련 문서의 위치의 역수의 평균

**Precision@k**
- Top k 중 관련 문서의 비율

**Recall@k**
- 전체 관련 문서 중 top k에 포함된 비율

**MTEB (Massive Text Embedding Benchmark)**
- Embedding model을 평가하는 대규모 벤치마크 (56 datasets, 112 languages)

### Tool Calling 관련

**Tool Calling**
- LLM이 외부 API나 도구를 호출하여 작업을 완료하는 기능

**Function Calling**
- Structured data를 function으로 표현하여 LLM이 호출할 수 있게 하는 방법

**Tool Specification**
- LLM에게 제공되는 tool의 명세 (이름, 설명, parameters)

**Agentic Workflows**
- LLM이 autonomously하게 여러 도구를 사용하여 복잡한 작업을 수행하는 워크플로우

### 기타

**Vanilla LLM**
- Reasoning augmentation이나 외부 도구 없이 기본적인 형태의 LLM

**Contrastive Learning**
- Similar pairs는 가깝게, dissimilar pairs는 멀게 학습하는 방법

**Cosine Similarity**
- 두 벡터의 방향 유사도. cos(θ) = (a·b) / (||a|| × ||b||), 범위: [-1, 1]

---

## 다음 강의 예고

다음 강의에서는 Agentic LLMs의 심화 내용을 다룹니다:
- Multi-agent systems
- Agent planning and reasoning
- Memory systems
- Advanced agentic workflows

---

**수고하셨습니다! 🎉**

이번 강의에서는 RAG와 Tool Calling을 통해 LLM의 실용적인 활용 방법을 배웠습니다. 다음 강의에서 만나요!