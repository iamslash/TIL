# DSPy

> **"프롬프트가 아닌 프로그래밍으로 LLM을 다루자"**
> Stanford NLP에서 만든 LLM 프로그래밍 프레임워크.
> 프롬프트를 수동으로 작성하는 대신, 입출력 구조와 예시를 정의하면 프레임워크가 프롬프트를 자동 생성/최적화한다.

- [Materials](#materials)
- [개요](#개요)
- [핵심 개념](#핵심-개념)
  - [Signature](#signature)
  - [Module](#module)
  - [Optimizer](#optimizer)
  - [Metric](#metric)
- [내장 모듈](#내장-모듈)
- [Optimizer 종류](#optimizer-종류)
- [LM 설정](#lm-설정)
- [평가 (Evaluation)](#평가-evaluation)
- [LangChain / LlamaIndex 비교](#langchain--llamaindex-비교)
- [실전 예제](#실전-예제)
  - [기본 사용법](#기본-사용법)
  - [멀티 스테이지 파이프라인](#멀티-스테이지-파이프라인)
  - [최적화](#최적화)
  - [RAG 파이프라인](#rag-파이프라인)
- [언제 DSPy를 써야 하는가](#언제-dspy를-써야-하는가)

---

# Materials

- [DSPy 공식 사이트](https://dspy.ai/)
- [GitHub - stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) - 32k+ stars
- [DSPy Cheatsheet](https://dspy.ai/cheatsheet/)
- [DSPy 원본 논문 (arXiv, 2023)](https://arxiv.org/pdf/2310.03714)
- [Programming Not Prompting 가이드 (Medium)](https://miptgirl.medium.com/programming-not-prompting-a-hands-on-guide-to-dspy-04ea2d966e6d)
- [DSPy Optimizers Deep Dive (Weaviate)](https://weaviate.io/blog/dspy-optimizers)
- [What is DSPy? (IBM)](https://www.ibm.com/think/topics/dspy)

---

# 개요

## 프롬프트 엔지니어링의 문제

```python
# 현재 방식: 프롬프트를 수동으로 작성하고, 결과 보고, 수정하고...
prompt = """
당신은 대화에서 사람 이름을 추출하는 시스템입니다.
규칙:
1. 사람 이름만 추출
2. 별명도 포함
3. 대명사는 특정 인물을 가리킬 때만...
(100줄 프롬프트)
"""
# 결과가 안 좋으면? → 프롬프트 수정 → 다시 테스트 → 또 수정 → ...
# 모델을 바꾸면? → 프롬프트 전부 다시 작성
```

## DSPy의 접근

```python
import dspy

# 프롬프트 대신 "무엇을 할 것인가"만 선언
class ExtractEntities(dspy.Signature):
    """대화에서 사람 이름을 추출한다."""
    conversation: str = dspy.InputField()
    entities: list[str] = dspy.OutputField()

# 예시 데이터 몇 개만 주면 DSPy가 프롬프트를 자동 생성/최적화
```

## 비유

| | 프롬프트 엔지니어링 | DSPy |
|--|-------------------|------|
| 비유 | 어셈블리 코딩 | Python 코딩 |
| 작성하는 것 | 자연어 프롬프트 (수백 줄) | 입출력 구조 + 예시 데이터 |
| 최적화 | 사람이 수동으로 수정 | **프레임워크가 자동 최적화** |
| 모델 바꾸면? | 프롬프트 다시 작성 | **재컴파일만** |
| 재현성 | 낮음 (감에 의존) | 높음 (코드로 관리) |

---

# 핵심 개념

```
Signature   →  "무엇을 할 것인가" (입출력 정의)
Module      →  "어떻게 할 것인가" (Signature를 실행하는 전략)
Optimizer   →  "자동으로 프롬프트를 최적화" (예시 데이터 기반)
Metric      →  "결과가 좋은지 판단하는 기준"
```

## Signature

입출력을 선언적으로 정의한다. **프롬프트가 아니다.** DSPy 컴파일러가 Signature를 보고 프롬프트를 자동 생성한다.

### 인라인 방식 (간단한 경우)

```python
# "question을 입력받아 answer를 출력"
"question -> answer"

# 타입 힌트 포함
"sentence -> sentiment: bool"

# 여러 입력
"context: list[str], question -> answer: str"
```

### 클래스 방식 (상세 제어)

```python
class Emotion(dspy.Signature):
    """문장의 감정을 분류한다."""
    sentence: str = dspy.InputField()
    sentiment: Literal['sadness', 'joy', 'anger', 'fear'] = dspy.OutputField()

class Summarize(dspy.Signature):
    """텍스트를 3줄로 요약한다."""
    text: str = dspy.InputField()
    summary: str = dspy.OutputField(desc="3줄 이내 요약")
```

**핵심**: 필드 이름이 의미를 가진다. `question`과 `sql_query`는 다른 프롬프트를 생성한다.

## Module

Signature를 실행하는 전략. PyTorch의 `nn.Module`과 비슷하다.

```python
# Predict: 가장 기본. 바로 답변
predictor = dspy.Predict(Summarize)

# ChainOfThought: 단계별 추론 후 답변
reasoner = dspy.ChainOfThought(Summarize)

# 여러 Module을 조합하여 파이프라인 구성
class MyPipeline(dspy.Module):
    def __init__(self):
        self.extract = dspy.ChainOfThought(ExtractEntities)
        self.classify = dspy.Predict(ClassifyRelation)

    def forward(self, text):
        entities = self.extract(text=text)
        return self.classify(entities=entities.entities)
```

## Optimizer

예시 데이터와 Metric을 주면, 프로그램의 파라미터(프롬프트 문구, few-shot 예시)를 **자동으로 최적화**한다.

```python
optimizer = dspy.BootstrapFewShot(metric=my_metric)
optimized_program = optimizer.compile(my_pipeline, trainset=examples)

# optimized_program은 원래 프로그램과 동일하게 사용하되,
# 내부 프롬프트가 최적화되어 있다.
result = optimized_program(text="어제 Jake랑 밥 먹었어")
```

## Metric

결과가 좋은지 판단하는 Python 함수. `example`(정답)과 `pred`(예측)를 받아 점수를 반환한다.

```python
# 간단한 정확도
def exact_match(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# F1 점수
def f1_metric(example, pred, trace=None):
    # ... F1 계산 ...
    return f1_value

# DSPy 프로그램 자체를 Metric으로 사용 가능 (!)
class QualityAssessor(dspy.Signature):
    """답변 품질을 1~5점으로 평가."""
    question: str = dspy.InputField()
    answer: str = dspy.InputField()
    score: int = dspy.OutputField(desc="1~5")
```

---

# 내장 모듈

| 모듈 | 용도 | 설명 |
|------|------|------|
| `dspy.Predict` | 기본 예측 | 가장 단순. Signature를 바로 실행 |
| `dspy.ChainOfThought` | 단계별 추론 | "생각하고 나서 답해" (CoT) |
| `dspy.ProgramOfThought` | 코드 기반 추론 | 코드를 생성/실행하여 답변 |
| `dspy.ReAct` | 도구 사용 에이전트 | 도구를 호출하며 답변 |
| `dspy.MultiChainComparison` | 앙상블 | 여러 ChainOfThought 결과를 비교/정제 |
| `dspy.RLM` | 재귀적 추론 (v3.1+) | 샌드박스 Python REPL + 재귀 호출 |

### 사용 예시

```python
# Predict: 바로 답변
predict = dspy.Predict("question -> answer")
result = predict(question="서울의 인구는?")
print(result.answer)  # "약 950만 명"

# ChainOfThought: 추론 과정 포함
cot = dspy.ChainOfThought("question -> answer")
result = cot(question="서울의 인구는?")
print(result.reasoning)  # "서울은 대한민국의 수도이며..."
print(result.answer)     # "약 950만 명"

# ReAct: 도구를 사용하여 답변
def search(query: str) -> str:
    """웹 검색"""
    return "검색 결과..."

react = dspy.ReAct("question -> answer", tools=[search])
result = react(question="오늘 서울 날씨는?")
```

---

# Optimizer 종류

## 4가지 카테고리

### A. Few-Shot 학습 (예시 자동 선택)

| Optimizer | 설명 | 데이터 | 권장 |
|-----------|------|--------|------|
| `LabeledFewShot` | 레이블된 데이터에서 예시 구성 | 소량 | 베이스라인 |
| `BootstrapFewShot` | 교사 모듈로 예시 생성, 메트릭으로 필터 | ~10개 | **시작점** |
| `BootstrapFewShotWithRandomSearch` | BootstrapFewShot를 여러 번 수행, 최선 선택 | 50개+ | 더 많은 데이터 |
| `KNNFewShot` | k-최근접 이웃으로 관련 예시 선택 | 다양한 | 다양한 데이터셋 |

### B. 지시문(Instruction) 최적화

| Optimizer | 설명 | 핵심 |
|-----------|------|------|
| `COPRO` | 좌표 상승법으로 지시문 생성/정제 | 체계적 탐색 |
| `MIPROv2` | 지시문 + few-shot을 **동시 최적화** | **베이지안 최적화**, 가장 강력 |
| `SIMBA` | 자기 성찰 규칙 생성 | 미니배치 |
| `GEPA` | LLM 성찰 기반 프롬프트 진화 | 경험 반영 |

### C. 파인튜닝

| Optimizer | 설명 |
|-----------|------|
| `BootstrapFinetune` | 프롬프트 기반 프로그램을 가중치 업데이트로 증류 |

### 선택 가이드

| 상황 | 추천 |
|------|------|
| 예시 ~10개, 빠르게 시작 | **BootstrapFewShot** |
| 예시 50개+, 더 좋은 성능 | **BootstrapFewShotWithRandomSearch** |
| 예시 200개+, 최대 성능 | **MIPROv2** |
| 소형 모델로 배포하고 싶을 때 | **BootstrapFinetune** |

### 최적화 비용

일반적으로 **$2, ~10분** 소요 (모델과 데이터에 따라 다름).

---

# LM 설정

## 기본 설정

```python
import dspy

lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)
```

## 지원 프로바이더

| 프로바이더 | 설정 |
|-----------|------|
| OpenAI | `OPENAI_API_KEY` 환경변수 |
| Anthropic Claude | `ANTHROPIC_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` |
| 로컬 모델 | SGLang (GPU) 또는 Ollama (노트북) |
| OpenAI 호환 | `openai/` 접두사 + `api_base` |

```python
# Anthropic
lm = dspy.LM('anthropic/claude-sonnet-4-20250514')

# 로컬 (Ollama)
lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434')

# 커스텀 엔드포인트
lm = dspy.LM('openai/my-model', api_base='https://my-server.com/v1')
```

## 생성 파라미터

```python
lm = dspy.LM(
    'openai/gpt-4o-mini',
    temperature=0.7,
    max_tokens=1000,
    cache=True,        # 동일 입력 캐싱 (기본 True)
)
```

## 여러 LM 사용

```python
fast_lm = dspy.LM('openai/gpt-4o-mini')
strong_lm = dspy.LM('openai/gpt-4o')

dspy.configure(lm=fast_lm)  # 기본 모델

# 특정 구간만 다른 모델 사용
with dspy.context(lm=strong_lm):
    result = complex_module(question="어려운 질문")
```

## 히스토리 확인

```python
lm.history  # 프롬프트, 출력, 토큰 사용량, 비용, 타임스탬프
```

---

# 평가 (Evaluation)

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(
    devset=test_examples,
    num_threads=4,         # 병렬 평가
    display_progress=True,
    display_table=5,       # 샘플 결과 테이블 표시
)

score = evaluator(my_program, metric=exact_match)
print(f"정확도: {score}%")
```

---

# LangChain / LlamaIndex 비교

| 기준 | LangChain | LlamaIndex | DSPy |
|------|-----------|------------|------|
| **철학** | 오케스트레이션 | 검색 최적화 | **프로그래밍 + 자동 최적화** |
| **강점** | 빠른 프로토타이핑, 광범위한 통합 | 문서 처리, RAG | **자동 프롬프트 최적화** |
| **프레임워크 오버헤드** | ~10ms | ~7ms | **~3.5ms** |
| **프롬프트 관리** | 수동 | 수동 | **자동** |
| **모델 교체** | 프롬프트 재작성 | 프롬프트 재작성 | **재컴파일** |
| **학습 곡선** | 낮음 | 중간 | 중간~높음 |
| **최적 용도** | 프로토타입, 에이전트 | 문서 RAG | **프로덕션 파이프라인** |

### 언제 어떤 것을 쓸까

| 상황 | 추천 |
|------|------|
| 빠른 프로토타입, 다양한 도구 연결 | **LangChain** |
| 대량 문서 기반 RAG | **LlamaIndex** |
| 프로덕션 최적화 필요, 멀티 스테이지 파이프라인 | **DSPy** |
| 프롬프트를 체계적으로 관리/개선 | **DSPy** |
| 모델을 자주 바꿀 예정 | **DSPy** |

---

# 실전 예제

## 기본 사용법

```python
import dspy

# 1. LM 설정
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)

# 2. Signature 정의
class QA(dspy.Signature):
    """질문에 답변한다."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# 3. Module로 실행
qa = dspy.ChainOfThought(QA)
result = qa(question="Python에서 리스트와 튜플의 차이는?")

print(result.reasoning)  # 추론 과정
print(result.answer)     # 최종 답변
```

## 멀티 스테이지 파이프라인

```python
class ExtractEntities(dspy.Signature):
    """대화에서 사람 이름을 추출."""
    conversation: str = dspy.InputField()
    entities: list[str] = dspy.OutputField()

class ClassifyRelation(dspy.Signature):
    """두 사람의 관계를 분류."""
    person1: str = dspy.InputField()
    person2: str = dspy.InputField()
    context: str = dspy.InputField()
    relation: str = dspy.OutputField(desc="friend, family, colleague, etc.")

class SocialGraphPipeline(dspy.Module):
    def __init__(self):
        self.extract = dspy.ChainOfThought(ExtractEntities)
        self.classify = dspy.Predict(ClassifyRelation)

    def forward(self, conversation):
        # 1단계: 엔티티 추출
        extracted = self.extract(conversation=conversation)

        # 2단계: 관계 분류 (엔티티 쌍마다)
        relations = []
        entities = extracted.entities
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                rel = self.classify(
                    person1=entities[i],
                    person2=entities[j],
                    context=conversation,
                )
                relations.append((entities[i], entities[j], rel.relation))

        return dspy.Prediction(entities=entities, relations=relations)

# 실행
pipeline = SocialGraphPipeline()
result = pipeline(conversation="어제 Jake랑 우리 엄마가 같이 밥 먹었대")
# entities: ["Jake", "엄마"]
# relations: [("Jake", "엄마", "acquaintance")]
```

## 최적화

```python
# 예시 데이터 준비
examples = [
    dspy.Example(
        conversation="어제 Jake랑 밥 먹었어",
        entities=["Jake"],
    ).with_inputs("conversation"),
    dspy.Example(
        conversation="우리 엄마가 Sarah 좋아하셔",
        entities=["엄마", "Sarah"],
    ).with_inputs("conversation"),
    # ... 10개 정도
]

# 메트릭 정의
def entity_match(example, pred, trace=None):
    gold = set(example.entities)
    predicted = set(pred.entities)
    if not gold:
        return len(predicted) == 0
    return len(gold & predicted) / len(gold)

# 최적화 (BootstrapFewShot)
optimizer = dspy.BootstrapFewShot(metric=entity_match, max_bootstrapped_demos=4)
optimized = optimizer.compile(SocialGraphPipeline(), trainset=examples)

# 최적화된 프로그램 사용 (원래와 동일한 인터페이스)
result = optimized(conversation="오늘 수진이랑 카페 갔어")

# 최적화된 프로그램 저장/로드
optimized.save("social_graph_v1")
loaded = SocialGraphPipeline()
loaded.load("social_graph_v1")
```

## RAG 파이프라인

```python
class GenerateAnswer(dspy.Signature):
    """컨텍스트를 참고하여 질문에 답변."""
    context: list[str] = dspy.InputField(desc="관련 문서 조각들")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

class RAG(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# 사용
rag = RAG()
result = rag(question="Momo의 가십 게이트는 어떻게 작동하나?")
print(result.answer)
```

---

# 언제 DSPy를 써야 하는가

## 쓰면 좋은 경우

- LLM 파이프라인이 **3개 이상** 있고, 각각 프롬프트를 관리해야 할 때
- 모델을 자주 교체할 예정일 때 (GPT → Claude → Gemini)
- 프롬프트 품질을 **체계적으로 측정/개선**하고 싶을 때
- 프로덕션에서 **재현 가능한** AI 시스템이 필요할 때

## 아직 불필요한 경우

- LLM 호출이 1~2개뿐일 때
- 프로토타입 단계에서 빠르게 검증만 하고 싶을 때
- 팀에 ML/NLP 경험이 없고 학습 비용이 부담될 때

## 버전 정보

| 항목 | 값 |
|------|-----|
| 최신 버전 | 3.1.3 (2026-02-05) |
| GitHub Stars | 32k+ |
| 라이선스 | MIT |
| 의존 프로젝트 | 1,500+ |
