# Lecture 6: LLM Reasoning

# Materials

- [CME 295](https://cme295.stanford.edu/syllabus/)
- [slide](https://cme295.stanford.edu/slides/fall25-cme295-lecture6.pdf)
- [video](https://www.youtube.com/watch?v=k5Fh-UgTuCo&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy&index=6)

# Table of Contents

- [Lecture 6: LLM Reasoning](#lecture-6-llm-reasoning)
- [Materials](#materials)
- [Table of Contents](#table-of-contents)
- [강의 개요](#강의-개요)
  - [강의 목표](#강의-목표)
  - [주요 학습 내용](#주요-학습-내용)
- [1. 이전 강의 복습](#1-이전-강의-복습)
  - [1.1. Pre-training (Lecture 4)](#11-pre-training-lecture-4)
  - [1.2. Fine-tuning (Lecture 4)](#12-fine-tuning-lecture-4)
  - [1.3. Preference Tuning (Lecture 5)](#13-preference-tuning-lecture-5)
  - [1.4. RL 프레임워크 복습](#14-rl-프레임워크-복습)
- [2. Vanilla LLM의 한계](#2-vanilla-llm의-한계)
  - [2.1. Vanilla LLM의 강점](#21-vanilla-llm의-강점)
  - [2.2. Vanilla LLM의 약점](#22-vanilla-llm의-약점)
    - [약점 1: Limited Reasoning (제한된 추론 능력)](#약점-1-limited-reasoning-제한된-추론-능력)
    - [약점 2: Static Knowledge (정적 지식)](#약점-2-static-knowledge-정적-지식)
    - [약점 3: All Talk, No Action (행동 불가)](#약점-3-all-talk-no-action-행동-불가)
    - [약점 4: Hard to Evaluate (평가의 어려움)](#약점-4-hard-to-evaluate-평가의-어려움)
- [3. Reasoning이란?](#3-reasoning이란)
  - [3.1. Reasoning의 정의](#31-reasoning의-정의)
  - [3.2. Reasoning vs Non-Reasoning 문제](#32-reasoning-vs-non-reasoning-문제)
    - [Non-Reasoning 문제 (지식 기반)](#non-reasoning-문제-지식-기반)
    - [Reasoning 문제 (다단계 사고)](#reasoning-문제-다단계-사고)
  - [3.3. Chain of Thought (CoT) 복습](#33-chain-of-thought-cot-복습)
- [4. Reasoning Models](#4-reasoning-models)
  - [4.1. Reasoning Models의 핵심 아이디어](#41-reasoning-models의-핵심-아이디어)
  - [4.2. Reasoning Models의 발전 타임라인](#42-reasoning-models의-발전-타임라인)
  - [4.3. Reasoning Models 식별하기](#43-reasoning-models-식별하기)
  - [4.4. Pricing 고려사항](#44-pricing-고려사항)
- [5. Benchmarks](#5-benchmarks)
  - [5.1. Coding Benchmarks](#51-coding-benchmarks)
    - [1. HumanEval](#1-humaneval)
    - [2. CodeForces](#2-codeforces)
    - [3. SWE-bench](#3-swe-bench)
  - [5.2. Math Benchmarks](#52-math-benchmarks)
    - [1. AIM (American Invitational Mathematics Examination)](#1-aim-american-invitational-mathematics-examination)
    - [2. GSM-8K (Grade School Math 8K)](#2-gsm-8k-grade-school-math-8k)
    - [3. MATH Dataset](#3-math-dataset)
  - [5.3. Pass@k Metric](#53-passk-metric)
    - [Pass@k의 정의](#passk의-정의)
    - [Pass@k 추정 공식 유도](#passk-추정-공식-유도)
    - [Pass@1의 특수한 경우](#pass1의-특수한-경우)
  - [5.4. Temperature와 Pass@k의 관계](#54-temperature와-passk의-관계)
    - [Temperature = 0 (너무 낮음)](#temperature--0-너무-낮음)
    - [Temperature = 0.2 (낮음)](#temperature--02-낮음)
    - [Temperature = 0.8 (적절함)](#temperature--08-적절함)
    - [Temperature = 1.2 (너무 높음)](#temperature--12-너무-높음)
  - [5.5. 기타 Metrics](#55-기타-metrics)
    - [Consensus@k](#consensusk)
    - [Accuracy](#accuracy)
    - [Exact Match](#exact-match)
- [6. Scaling with RL](#6-scaling-with-rl)
  - [6.1. 왜 SFT를 사용하지 않는가?](#61-왜-sft를-사용하지-않는가)
  - [6.2. 왜 RL을 사용하는가?](#62-왜-rl을-사용하는가)
    - [1. 데이터 효율성](#1-데이터-효율성)
    - [2. 자동 검증](#2-자동-검증)
    - [3. 모델 최적 Reasoning](#3-모델-최적-reasoning)
    - [4. 확장성](#4-확장성)
  - [6.3. Reward 설계](#63-reward-설계)
    - [Reward 1: Reasoning 존재 여부](#reward-1-reasoning-존재-여부)
    - [Reward 2: 정답 여부](#reward-2-정답-여부)
    - [최종 Reward](#최종-reward)
  - [6.4. RL 학습 결과](#64-rl-학습-결과)
  - [6.5. Compute Budget 제어](#65-compute-budget-제어)
    - [문제 1: Dynamic Budget](#문제-1-dynamic-budget)
    - [문제 2: Context Length Awareness](#문제-2-context-length-awareness)
    - [문제 3: Continuous Thought](#문제-3-continuous-thought)
- [7. GRPO (Group Relative Policy Optimization)](#7-grpo-group-relative-policy-optimization)
  - [7.1. GRPO란?](#71-grpo란)
  - [7.2. GRPO의 핵심 아이디어](#72-grpo의-핵심-아이디어)
  - [7.3. GRPO vs PPO](#73-grpo-vs-ppo)
    - [PPO 프로세스](#ppo-프로세스)
    - [GRPO 프로세스](#grpo-프로세스)
    - [상세 비교](#상세-비교)
  - [7.4. GRPO의 장점](#74-grpo의-장점)
    - [장점 1: No Value Function](#장점-1-no-value-function)
    - [장점 2: 간단한 구현](#장점-2-간단한-구현)
    - [장점 3: 상대적 비교](#장점-3-상대적-비교)
    - [장점 4: 자연스러운 Baseline](#장점-4-자연스러운-baseline)
  - [7.5. GRPO의 구현](#75-grpo의-구현)
    - [전체 알고리즘](#전체-알고리즘)
    - [Loss Function 상세](#loss-function-상세)
    - [실전 팁](#실전-팁)
- [8. 요약](#8-요약)
  - [핵심 개념](#핵심-개념)
    - [1. Reasoning이란?](#1-reasoning이란)
    - [2. Reasoning Models](#2-reasoning-models)
    - [3. 왜 RL?](#3-왜-rl)
    - [4. GRPO Algorithm](#4-grpo-algorithm)
  - [실전 체크리스트](#실전-체크리스트)
- [9. 중요 용어 정리](#9-중요-용어-정리)

---

# 강의 개요

## 강의 목표

이번 강의에서는 최근 1년간 가장 핫한 주제인 **LLM Reasoning**에 대해 학습합니다. 특히 Lecture 5에서 배운 preference tuning 기법들이 reasoning model 학습의 기초가 됩니다.

**학습 목표:**
- Reasoning models이 무엇인지 이해하기
- Reasoning models을 어떻게 학습시키는지 이해하기
- GRPO 알고리즘의 작동 원리 파악하기
- Pass@k metric 이해하기

## 주요 학습 내용

**1. Reasoning의 정의**
- Multi-step reasoning process
- Chain of Thought (CoT) 개념
- Reasoning vs Non-Reasoning 문제

**2. Reasoning Models**
- OpenAI O1, DeepSeek R1 등의 발전 과정
- Reasoning models 식별 방법
- Pricing과 compute budget

**3. Benchmarks**
- Coding: HumanEval, CodeForces, SWE-bench
- Math: AIM, GSM-8K
- Pass@k metric 유도 및 이해

**4. Training 방법론**
- 왜 RL을 사용하는가?
- GRPO (Group Relative Policy Optimization)
- GRPO vs PPO 비교

---

# 1. 이전 강의 복습

이번 섹션에서는 Lecture 4와 5에서 배운 내용을 간단히 복습합니다.

## 1.1. Pre-training (Lecture 4)

**목표:** 모델에게 텍스트와 코드의 구조를 가르치기

```
대규모 데이터 → Pre-training → Base Model

특징:
- 가장 compute-intensive한 단계
- Next token prediction 학습
- 텍스트/코드 구조 이해
```

**결과:**
- 코드와 언어를 이해하는 모델
- 하지만 오직 autocomplete만 가능

## 1.2. Fine-tuning (Lecture 4)

**목표:** Pre-trained 모델을 유용하게 만들기

```
Base Model + SFT Data → Fine-tuning → SFT Model

SFT Data 특징:
- 고품질 curated 데이터
- 특정 태스크에 맞춤 (예: Assistant)
- Instruction-following 능력 학습
```

**예시:**

```python
# SFT 데이터 형식
{
  "prompt": "Python에서 리스트를 정렬하는 방법을 알려주세요",
  "completion": "Python에서 리스트를 정렬하려면 sort() 메서드나 sorted() 함수를 사용할 수 있습니다..."
}
```

**결과:**
- 질문에 응답할 수 있는 모델
- 특정 태스크에 특화됨

## 1.3. Preference Tuning (Lecture 5)

**목표:** 모델을 human preferences와 align하기

**2단계 프로세스:**

```
1단계: Reward Model 학습
- Human preference data 수집
- Good vs Bad 구분 학습

2단계: RL 학습
- Reward Model 사용
- Policy 최적화
```

**RLHF (Reinforcement Learning from Human Feedback):**

```python
# Preference data 형식
{
  "prompt": "Python 함수 작성법",
  "chosen": "def hello():\n    print('Hello')",
  "rejected": "hello = lambda: print('Hello')"
}
```

## 1.4. RL 프레임워크 복습

**전통적인 RL vs LLM RL:**

| 요소 | 전통적인 RL | LLM RL |
|------|-------------|--------|
| Agent | 에이전트 | LLM |
| Environment | 환경 | Token space |
| State | 상태 | Input tokens so far |
| Action | 행동 | Next token prediction |
| Policy | 정책 π(a\|s) | P(token\|context) |
| Reward | 보상 | Human preference score |

**PPO Loss Function (복습):**

```
Loss = Advantage Maximization + KL Divergence Penalty

두 가지 구성요소:
1. Advantage Maximization: 보상 극대화
2. KL Divergence: 모델이 너무 많이 변하지 않도록

KL Penalty 대상:
- Old policy (이전 iteration)
- Base policy (SFT model)
```

**PPO Variants:**

1. **PPO-CLIP:**
```
L^CLIP = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)

여기서:
- r_t = π_new(a|s) / π_old(a|s) (ratio)
- A_t = advantage
- ε = clipping threshold (예: 0.2)
```

2. **PPO-KL Penalty:**
```
L = E[A_t] - β * KL(π_new || π_ref)

여기서:
- β = KL penalty coefficient
- π_ref = reference policy (보통 SFT model)
```

**현대 RLHF:**

```python
# 실제로는 두 가지를 혼합
Loss = PPO_CLIP + β * KL(π || π_SFT)

목적:
1. Clipping: iteration 간 변화 제한
2. KL penalty: base model로부터 deviation 제한
```

---

# 2. Vanilla LLM의 한계

## 2.1. Vanilla LLM의 강점

**Vanilla LLM:** 프롬프트를 입력받아 바로 답변을 생성하는 기본 LLM

```
Input: Prompt → LLM → Output: Answer
```

**강점:**

1. **풍부한 지식:**
   - 텍스트 구조 이해
   - 코드 구조 이해
   - 광범위한 도메인 지식

2. **코드 작업:**
   - 버그 디버깅
   - 코드 생성
   - 코드 설명

3. **텍스트 생성:**
   - 에세이 작성
   - 시 작성
   - 창의적 글쓰기

**예시:**

```python
# 간단한 코드 생성
Prompt: "Python에서 리스트의 합을 구하는 함수를 작성하세요"

Response:
def sum_list(numbers):
    return sum(numbers)
```

## 2.2. Vanilla LLM의 약점

### 약점 1: Limited Reasoning (제한된 추론 능력)

**문제:**

```python
# 복잡한 수학 문제
Prompt: """
한 회사에 직원이 120명 있습니다.
이 중 60%가 남성이고, 남성 중 40%가 엔지니어입니다.
여성 중에서는 50%가 엔지니어입니다.
전체 엔지니어는 몇 명입니까?
"""

Vanilla LLM의 문제:
- 중간 단계를 건너뛰고 바로 답변
- 복잡한 로직에서 길을 잃음
- 검증 없이 plausible한 답변 생성
```

**왜 이런 문제가 발생하나?**

```
LLM 학습 방식:
- Next token prediction에만 최적화
- 단계별 사고 과정 없음
- 최종 답만 출력하도록 학습
```

### 약점 2: Static Knowledge (정적 지식)

**문제:**

```python
Prompt: "2024년 11월 미국 대통령 선거 결과는?"

Cutoff Date가 2024년 9월이면:
- 선거 전 데이터만 학습
- 실제 결과를 모름
```

**Cutoff Date:**
```
Pre-training Data
┌─────────────────────────────────┐
│  2020  2021  2022  2023  2024.09│  ← Cutoff
└─────────────────────────────────┘
                                   │
                         이후 정보는 모름
```

### 약점 3: All Talk, No Action (행동 불가)

**문제:**

```
User: "내일 오전 9시에 회의 일정을 잡아주세요"

Vanilla LLM:
- 답변만 생성: "회의를 예약하겠습니다"
- 실제 액션은 없음 (캘린더에 추가 안됨)
```

### 약점 4: Hard to Evaluate (평가의 어려움)

**전통적인 NLP Metrics:**

```python
# 번역 평가
BLEU Score: n-gram overlap 측정

# 요약 평가
ROUGE Score: recall/precision 측정

문제: LLM은 자유형식 텍스트 생성
→ 단순 metric으로 평가 불가
```

**해결 방향:**

```
약점 1 (Reasoning): Lecture 6에서 다룸 ← 오늘!
약점 2 (Static): Lecture 7에서 다룸 (RAG, Tools)
약점 3 (Action): Lecture 7에서 다룸 (Agents)
약점 4 (Eval): Lecture 8에서 다룸
```

---

# 3. Reasoning이란?

## 3.1. Reasoning의 정의

**주의:** Reasoning에 대한 공통된 정의는 없음

**우리의 정의:**

```
Reasoning = 문제를 해결하는 능력

특징:
1. Multi-step reasoning process 필요
2. 주로 수학/코딩 문제
3. 단계별 분해와 해결
```

**비유:**

```
시험 문제 풀이 과정:
1. 문제 읽기
2. 하위 문제로 분해
3. 각 단계 해결
4. 최종 답 도출

→ Reasoning models도 동일한 패턴!
```

## 3.2. Reasoning vs Non-Reasoning 문제

### Non-Reasoning 문제 (지식 기반)

**예시 1:**

```
Q: "Stanford의 Transformer와 LLM 수업 코드는?"
A: "CME 295"

특징:
- 단순 지식 retrieval
- 한 단계로 답변 가능
- 중간 사고 과정 불필요
```

**예시 2:**

```
Q: "Python의 창시자는?"
A: "Guido van Rossum"

특징:
- 암기된 사실
- 추론 불필요
```

### Reasoning 문제 (다단계 사고)

**예시 1 (간단한 수학):**

```
Q: "곰이 2020년에 태어났습니다. 2025년 현재 몇 살입니까?"

Reasoning Process:
1. 현재 연도 확인: 2025년
2. 태어난 연도: 2020년
3. 나이 계산: 2025 - 2020 = 5
4. 답: 5살

특징:
- 여러 단계 필요
- 중간 계산 필요
```

**예시 2 (복잡한 수학):**

```
Q: """
한 상자에 빨간 공 15개, 파란 공 20개가 있습니다.
공을 하나씩 꺼낼 때, 같은 색 공 3개를 연속으로
꺼낼 확률은?
"""

Reasoning Process:
1. 전체 공 개수 계산
2. 첫 번째 공 확률 계산
3. 두 번째 공 조건부 확률
4. 세 번째 공 조건부 확률
5. 확률 곱하기
6. 두 가지 경우 (빨강/파랑) 더하기

→ 훨씬 복잡한 다단계 추론!
```

**예시 3 (코딩):**

```python
Q: "이진 트리에서 최대 경로 합을 찾는 알고리즘을 작성하세요"

Reasoning Process:
1. 문제 이해: 최대 경로 합 정의
2. 접근 방법: DFS vs BFS vs Dynamic Programming
3. 재귀 구조 설계
4. Base case 정의
5. Recursive case 정의
6. 코드 구현
7. 테스트 케이스 검증

→ 복잡한 알고리즘 설계 추론!
```

## 3.3. Chain of Thought (CoT) 복습

**Chain of Thought:** Lecture 2-3에서 배운 개념

**핵심 아이디어:**

```
기존 방식:
Q: "2+3×4는?"
A: "14"

Chain of Thought:
Q: "2+3×4는?"
A: "먼저 곱셈을 계산합니다: 3×4 = 12
    그 다음 덧셈: 2+12 = 14
    답: 14"
```

**In-Context Learning 예시:**

```python
Prompt = """
Q: 5 + 3 × 2는?
A: 먼저 곱셈: 3 × 2 = 6, 그 다음 덧셈: 5 + 6 = 11

Q: 10 - 4 × 2는?
A: 먼저 곱셈: 4 × 2 = 8, 그 다음 뺄셈: 10 - 8 = 2

Q: 7 + 2 × 3는?
A: """

→ 모델이 패턴을 따라 단계별로 사고
```

**왜 CoT가 작동하는가?**

1. **Decomposition (분해):**
   ```
   복잡한 문제 → 작은 문제들

   각 작은 문제는 training data에서 본 패턴
   ```

2. **More Compute:**
   ```
   더 많은 토큰 생성 = 더 많은 forward pass
   = 더 많은 compute = 더 나은 성능
   ```

3. **Training Data Coverage:**
   ```
   전체 문제: Training data에 없을 수 있음
   하위 문제들: Training data에 많이 있음

   예: "3 × 4" 같은 간단한 계산은 많이 봄
   ```

**Reasoning Models의 핵심:**

```
Chain of Thought를 대규모로 적용!

Vanilla LLM:
Input: Question → Output: Answer

Reasoning Model:
Input: Question → Output: Reasoning Chain + Answer
```

---

# 4. Reasoning Models

## 4.1. Reasoning Models의 핵심 아이디어

**기본 개념:**

```
Vanilla LLM:
┌──────────┐
│ Question │ → LLM → Answer
└──────────┘

Reasoning Model:
┌──────────┐
│ Question │ → LLM → Reasoning Chain → Answer
└──────────┘               ↑
                    여기가 핵심!
```

**구체적인 예시:**

```python
Question: "곰이 2020년에 태어났다. 2025년에 몇 살인가?"

Vanilla LLM Output:
"5살입니다."

Reasoning Model Output:
"""
<think>
현재 연도를 확인해보자: 2025년
곰이 태어난 연도: 2020년
나이 = 현재 연도 - 태어난 연도
나이 = 2025 - 2020 = 5
</think>

답: 5살입니다.
"""
```

**Output 구조:**

```
Reasoning Model Output = Reasoning Chain + Answer

Reasoning Chain:
- <think>와 </think> 사이
- 단계별 사고 과정
- 중간 계산
- 자기 검증

Answer:
- 최종 답변
- Reasoning chain 이후
```

## 4.2. Reasoning Models의 발전 타임라인

**2024년 이전:**
- Reasoning은 연구 주제였으나 production 모델은 없음
- Chain of Thought 등의 prompting techniques만 존재

**2024년 9월: OpenAI O1 Preview**

```
최초의 production reasoning model
- Extended thinking time
- Hidden reasoning chain
- 복잡한 수학/코딩 문제 해결
```

**2024년 12월: Google Gemini 2.0 Flash Thinking**

```
Google의 reasoning model
- Fast thinking mode
- 효율적인 inference
```

**2025년 1월: DeepSeek R1 (게임 체인저!)**

```
왜 중요한가?
1. OpenAI O1과 비슷한 성능
2. 학습 방법을 논문으로 공개!
3. 오픈소스 커뮤니티에 큰 영향

→ 모두가 reasoning model 학습 방법을 이해
```

**2025년 이후:**

```
모든 주요 AI lab이 reasoning model 출시:
- xAI (Grok)
- Anthropic (Claude with thinking)
- Mistral
- Meta (LLaMA reasoning variants)
```

**Timeline 요약:**

```
2024.09: O1 Preview (OpenAI) - 시작
2024.12: Gemini Thinking (Google)
2025.01: DeepSeek R1 - 방법론 공개!
2025.02~: 모든 lab이 참여

→ 불과 1년 만에 폭발적 성장!
```

## 4.3. Reasoning Models 식별하기

**UI에서 Reasoning Model 확인하기:**

**1. Thinking Indicator:**

```
ChatGPT 예시:
┌─────────────────────────────┐
│ Thinking... (3s)            │ ← 이것!
├─────────────────────────────┤
│ [답변 내용]                 │
└─────────────────────────────┘
```

**2. Thinking Mode 선택:**

```
OpenAI ChatGPT:
- Standard thinking
- Extended thinking ← compute budget 조절

Anthropic Claude:
- Extended thinking mode
```

**3. Thought Summary:**

```
실제 reasoning chain은 보통 숨겨짐

보여지는 것:
"I analyzed the problem by considering..."

숨겨진 것:
실제 reasoning chain (수백~수천 토큰)
```

**왜 Raw Reasoning Chain을 숨기는가?**

1. **Readability (가독성):**
   ```
   Raw chain: 수천 토큰, 읽기 어려움
   Summary: 핵심만 추출, 이해하기 쉬움
   ```

2. **Intelligibility (이해 가능성):**
   ```
   모델의 사고 과정 ≠ 인간의 사고 과정

   Raw chain 예시:
   "wait... let me reconsider...
    actually that's wrong...
    hmm... let me try another approach..."

   → 인간이 읽기에 혼란스러울 수 있음
   ```

3. **Competitive Advantage (경쟁 우위):**
   ```
   Raw reasoning chain 공개 시:
   - 다른 회사가 그 data로 모델 학습 가능
   - 회사의 기술적 우위 상실

   → Distillation attack 방지
   ```

**Distillation Attack:**

```python
# 공격자의 관점
for problem in problems:
    reasoning_chain = target_model.generate(problem)

    # 이 데이터로 자신의 모델 학습
    my_model.train(
        input=problem,
        output=reasoning_chain
    )

→ Target model의 능력을 복제!
```

## 4.4. Pricing 고려사항

**Reasoning Models의 비용 구조:**

```
기존 LLM:
Cost = Input tokens + Output tokens

Reasoning Model:
Cost = Input tokens + Reasoning tokens + Output tokens
                      ↑
                   추가 비용!
```

**OpenAI API 예시:**

```python
# GPT-4
Input:  $0.01 / 1K tokens
Output: $0.03 / 1K tokens

# GPT-4 with reasoning (O1)
Input:     $0.01 / 1K tokens
Reasoning: $0.06 / 1K tokens  ← 2배!
Output:    $0.03 / 1K tokens

→ Reasoning 사용 시 비용 크게 증가
```

**사용자 입장의 Incentive:**

```
목표: 최소 reasoning tokens로 최대 성능

Trade-off:
- 더 많은 thinking → 더 나은 답 → 더 비쌈
- 더 적은 thinking → 나쁜 답 → 저렴함

→ Compute budget 조절 필요!
```

**Compute Budget:**

```
Definition: 답변 생성에 사용할 수 있는 토큰 수

Standard mode:  ~1,000 tokens thinking
Extended mode: ~10,000 tokens thinking

선택 기준:
- 문제 난이도
- 비용 제약
- 응답 시간 요구사항
```

**예시 시나리오:**

```python
# 간단한 질문
Q: "Python에서 리스트 합계는?"
→ Standard mode로 충분 ($0.01)

# 복잡한 알고리즘
Q: "동적 프로그래밍으로 최장 증가 부분 수열 구현"
→ Extended mode 필요 ($0.10)

→ 문제에 따라 mode 선택!
```

---

# 5. Benchmarks

이번 섹션에서는 reasoning 능력을 측정하는 벤치마크들을 학습합니다.

## 5.1. Coding Benchmarks

**목표:** 코딩 문제 해결 능력 평가

**평가 방식:**

```
Input:  Problem description
Output: Code solution

Verification: Test cases로 검증
✓ 모든 test 통과 → Correct
✗ 하나라도 실패 → Incorrect
```

**주요 Benchmarks:**

### 1. HumanEval

```
특징:
- ~164개의 Python 문제
- Human-written problems
- Function implementation 중심

예시:
┌──────────────────────────────────┐
│ def has_close_elements(numbers,  │
│     threshold):                  │
│     \"\"\"                           │
│     Check if any two numbers are │
│     closer than threshold        │
│     \"\"\"                           │
│     # Your code here             │
└──────────────────────────────────┘

Test: assert has_close_elements([1.0, 2.0], 0.5) == False
```

### 2. CodeForces

```
특징:
- 경쟁 프로그래밍 사이트에서 가져옴
- 다양한 난이도 (Div 1, 2, 3)
- 알고리즘적 사고 필요

난이도 분포:
Easy:   기본 구현
Medium: 알고리즘 + 자료구조
Hard:   고급 알고리즘 + 최적화
```

### 3. SWE-bench

```
특징:
- Real-world GitHub issues
- 실제 코드베이스 수정
- 버그 수정, 기능 추가

예시:
Issue: "Fix memory leak in data processing pipeline"
→ 실제 production 코드 수정 필요

평가: PR이 original issue를 해결하는지
```

**Coding Benchmarks 비교:**

| Benchmark | 문제 수 | 난이도 | 실전성 |
|-----------|---------|--------|--------|
| HumanEval | ~164 | 중간 | 낮음 |
| CodeForces | 수천 | 다양 | 중간 |
| SWE-bench | ~2,000 | 높음 | 높음 |

## 5.2. Math Benchmarks

**목표:** 수학적 추론 능력 평가

**평가 방식:**

```
Input:  Math problem
Output: Reasoning chain + Answer

Verification: Answer parsing 후 정답 비교
```

**Answer Parsing 예시:**

```python
# Model output
output = """
<think>
계산 과정...
</think>

따라서 답은 \boxed{42}입니다.
"""

# Parsing
import re
answer = re.search(r'\\boxed{([^}]+)}', output).group(1)
# answer = "42"

# Verification
if answer == ground_truth:
    correct = True
```

**주요 Benchmarks:**

### 1. AIM (American Invitational Mathematics Examination)

```
특징:
- 미국 수학 올림피아드 예선
- 매우 어려운 문제
- 고급 수학 개념 필요

난이도:
- 미국 상위 1% 고등학생 대상
- 정답률 보통 10-30%

예시:
"정육면체 ABCD-EFGH에서 점 P는 모서리 AE 위에 있고
AP:PE = 2:1이다. 삼각형 BDP의 넓이를 구하시오."
```

### 2. GSM-8K (Grade School Math 8K)

```
특징:
- 초등학교 수준 문제
- ~8,000개 문제
- 자연어 수학 문제

예시:
"Sarah has 15 apples. She gives 3 to John and 5 to Mary.
Then she buys 8 more apples. How many apples does she have?"

Reasoning:
1. 시작: 15개
2. 주고 남은 것: 15 - 3 - 5 = 7개
3. 추가 구매: 7 + 8 = 15개
답: 15개
```

### 3. MATH Dataset

```
특징:
- 경쟁 수학 문제
- 7개 난이도 level
- 다양한 주제 (대수, 기하, 수론 등)

난이도 분포:
Level 1-2: 기초 (~50% 정답률)
Level 3-4: 중급 (~20% 정답률)
Level 5:   고급 (~5% 정답률)
```

**Math Benchmarks 비교:**

| Benchmark | 문제 수 | 난이도 | 주제 |
|-----------|---------|--------|------|
| GSM-8K | 8,000 | 초등 | 산수 |
| MATH | 12,500 | 다양 | 전반 |
| AIM | 수백 | 올림피아드 | 고급 |

## 5.3. Pass@k Metric

### Pass@k의 정의

**Motivation:** 여러 번 시도하면 성공 확률 높아짐

**정의:**

```
Pass@k = Probability(적어도 1개의 답이 맞음 | k번 시도)
```

**직관적 이해:**

```python
# k=1: 한 번만 시도
Q: "코딩 문제"
A1: [생성]
→ Pass@1 = P(A1이 맞음)

# k=5: 다섯 번 시도
Q: "코딩 문제"
A1, A2, A3, A4, A5: [생성]
→ Pass@5 = P(A1~A5 중 최소 1개 맞음)
```

**왜 Pass@k가 유용한가?**

1. **Multiple Attempts 허용:**
   ```
   실전 시나리오:
   - 코딩: 여러 솔루션 생성 후 test
   - 수학: 여러 접근법 시도

   Pass@k가 높으면:
   → 재시도로 정답 찾을 가능성 높음
   ```

2. **Best-of-n과 유사:**
   ```
   Best-of-n (Lecture 5):
   1. n개 답변 생성
   2. Reward model로 점수 매김
   3. 최고 점수 선택

   Pass@k:
   1. k개 답변 생성
   2. Deterministic verification (test cases)
   3. 맞는 것 하나 찾기

   → 검증 가능한 태스크에 이상적!
   ```

3. **Compute Budget Trade-off:**
   ```
   상황: 높은 정답률 필요

   Option 1: 더 큰 모델 사용 (비쌈)
   Option 2: 작은 모델 + 여러 번 시도 (저렴)

   Pass@k로 Option 2 평가 가능!
   ```

### Pass@k 추정 공식 유도

**Setup:**

```
n: 총 시도 횟수 (예: n=100)
c: 성공한 시도 수 (예: c=30)
k: 선택할 시도 수 (예: k=10)

목표: n개 중 k개를 랜덤하게 선택할 때,
     최소 1개가 성공할 확률?
```

**Step 1: Complementary Probability**

```
P(적어도 1개 성공) = 1 - P(모두 실패)

이유:
"적어도 1개"는 계산하기 복잡
"모두 실패"는 계산하기 쉬움
```

**Step 2: P(모두 실패) 계산**

```
Without replacement sampling:

1번째 선택이 실패: (n-c)/n
2번째도 실패 (given 1번째 실패): (n-c-1)/(n-1)
3번째도 실패 (given 1,2번째 실패): (n-c-2)/(n-2)
...
k번째도 실패: (n-c-k+1)/(n-k+1)

P(모두 실패) = (n-c)/n × (n-c-1)/(n-1) × ... × (n-c-k+1)/(n-k+1)
```

**Step 3: Factorial로 표현**

```
분자: (n-c) × (n-c-1) × ... × (n-c-k+1)
    = (n-c)! / (n-c-k)!

분모: n × (n-1) × ... × (n-k+1)
    = n! / (n-k)!

P(모두 실패) = [(n-c)! / (n-c-k)!] / [n! / (n-k)!]
```

**Step 4: Binomial Coefficient 도입**

```
조합 정의:
C(n, k) = n! / (k! × (n-k)!)

변형:
P(모두 실패) = [(n-c)! / (n-c-k)!] × [(n-k)! / n!]

분자분모에 k! 곱하기:
= [k! × (n-c)! / ((n-c-k)! × k!)] × [(n-k)! × k! / (n! × k!)]
= [C(n-c, k)] / [C(n, k)]
```

**최종 공식:**

```
Pass@k = 1 - C(n-c, k) / C(n, k)

여기서:
- n: 총 생성 수
- c: 성공 수
- k: 선택 수
- C(n, k) = n! / (k! × (n-k)!)
```

**구체적인 예시:**

```python
# 문제: 100번 시도, 30번 성공
n = 100
c = 30
k = 10

# Pass@10 계산
from math import comb

pass_at_10 = 1 - comb(100-30, 10) / comb(100, 10)
             = 1 - comb(70, 10) / comb(100, 10)
             = 1 - 0.0345
             = 0.9655
             = 96.55%

해석: 10번 시도하면 96.55% 확률로 최소 1개 맞음
```

**검증:**

```python
# k=1일 때 (특수 케이스)
pass_at_1 = 1 - comb(n-c, 1) / comb(n, 1)
          = 1 - (n-c) / n
          = c / n
          = 성공 비율

→ 직관과 일치!
```

### Pass@1의 특수한 경우

**정의:**

```
Pass@1 = 한 번 시도로 성공할 확률
```

**공식 대입:**

```
Pass@1 = 1 - C(n-c, 1) / C(n, 1)
       = 1 - (n-c) / n
       = c / n

→ 단순히 성공 비율!
```

**예시:**

```python
n = 100 (총 시도)
c = 75  (성공 수)

Pass@1 = 75/100 = 0.75 = 75%

해석: 모델이 한 번에 맞출 확률 75%
```

**Pass@1 vs Pass@k:**

```
Pass@1:  단일 시도 성능
Pass@k:  Multiple attempts 허용 시 성능

예시:
Pass@1  = 60%  ← 기본 성능
Pass@5  = 85%  ← 5번 시도 시
Pass@10 = 95%  ← 10번 시도 시

→ k 증가하면 Pass@k도 증가!
```

## 5.4. Temperature와 Pass@k의 관계

**Temperature 복습:**

```
Temperature = 샘플링 시 randomness 조절

T = 0:   Greedy (항상 최고 확률 토큰)
T = 0.8: Balanced (적당한 다양성)
T = 1.5: Random (매우 다양함)
```

**Pass@k에서 Temperature의 역할:**

### Temperature = 0 (너무 낮음)

```
문제점:
- 모든 생성이 동일
- Diversity 없음
- k를 늘려도 성능 향상 없음

그래프:
Pass@k
  │
75%├────────────────────
  │   flat line
  │
  └─────────────────── k
    1  5  10  50  100

→ k=1이나 k=100이나 동일!
```

### Temperature = 0.2 (낮음)

```
특징:
- 약간의 다양성
- 대부분 고품질
- k 증가 시 약간 향상

그래프:
Pass@k
  │
85%├──────────┌─────
  │         ╱
75%├────────┘
  │
  └─────────────────── k
    1  5  10  50  100

→ 초반에 향상, 이후 정체
```

### Temperature = 0.8 (적절함)

```
특징:
- 좋은 품질 + 다양성 균형
- k 증가 시 지속적 향상
- 최적의 선택!

그래프:
Pass@k
  │
95%├──────────────┌──
  │            ╱
85%├─────────╱
  │       ╱
75%├──────┘
  │
  └─────────────────── k
    1  5  10  50  100

→ k 증가에 따라 꾸준한 향상
```

### Temperature = 1.2 (너무 높음)

```
문제점:
- 매우 다양하지만 품질 낮음
- 낮은 확률 토큰도 자주 선택
- Pass@1이 낮음

그래프:
Pass@k
  │
75%├─────────────┌──
  │           ╱
65%├────────╱
  │      ╱
55%├─────┘
  │
  └─────────────────── k
    1  5  10  50  100

→ Pass@1은 낮지만 k 크면 회복
```

**최적 Temperature 선택:**

```python
# 실험 결과 (일반적 경향)
Best temperatures:
- Coding: 0.6 ~ 0.8
- Math:   0.7 ~ 0.9

이유:
1. 충분한 다양성 (exploration)
2. 높은 기본 품질 (exploitation)
3. k 증가 시 성능 향상 폭 최대

→ 논문에서 항상 temperature 명시!
```

**실전 가이드:**

```
목표에 따른 선택:

1. Pass@1 최대화:
   → T = 0.2~0.4 (품질 우선)

2. Pass@10 최대화:
   → T = 0.6~0.8 (균형)

3. Pass@100 최대화:
   → T = 0.8~1.0 (다양성 우선)
```

## 5.5. 기타 Metrics

### Consensus@k

**정의:**

```
Consensus@k = 가장 많이 나온 답을 최종 답으로 선택

예시:
k=5 생성:
Answer 1: "42"
Answer 2: "43"
Answer 3: "42"
Answer 4: "42"
Answer 5: "43"

Consensus: "42" (3번 등장)
```

**Self-Consistency와의 관계:**

```
Self-Consistency (Lecture 2-3):
1. 여러 번 생성
2. 가장 일관된 답 선택

Consensus@k:
→ Self-Consistency의 정량화!
```

**구현:**

```python
from collections import Counter

def consensus_at_k(answers):
    """
    answers: List of k answers
    returns: Most common answer
    """
    counter = Counter(answers)
    consensus_answer = counter.most_common(1)[0][0]
    return consensus_answer

# 예시
answers = ["42", "43", "42", "42", "43"]
result = consensus_at_k(answers)  # "42"
```

### Accuracy

**정의:**

```
Accuracy = 정답 수 / 전체 문제 수
```

**Coding에서:**

```python
correct = 0
total = 0

for problem in problems:
    solution = model.generate(problem)
    if passes_all_tests(solution):
        correct += 1
    total += 1

accuracy = correct / total
```

**Math에서:**

```python
correct = 0
total = 0

for problem, ground_truth in dataset:
    answer = model.generate_and_parse(problem)
    if answer == ground_truth:
        correct += 1
    total += 1

accuracy = correct / total
```

### Exact Match

**정의:**

```
Exact Match = 생성된 답이 정답과 정확히 일치하는 비율
```

**예시:**

```
Ground truth: "Paris"

Answer 1: "Paris"         → Exact Match ✓
Answer 2: "paris"         → Exact Match ✗ (대소문자)
Answer 3: "The answer is Paris" → Exact Match ✗ (추가 텍스트)
```

**수학 문제에서 Exact Match:**

```python
def parse_answer(text):
    """Extract answer from text"""
    # \boxed{answer} 형식에서 추출
    match = re.search(r'\\boxed{([^}]+)}', text)
    if match:
        return match.group(1).strip()
    return None

def exact_match(prediction, ground_truth):
    pred_answer = parse_answer(prediction)
    return pred_answer == ground_truth
```

---

# 6. Scaling with RL

이번 섹션에서는 reasoning model을 대규모로 학습하는 방법을 다룹니다.

## 6.1. 왜 SFT를 사용하지 않는가?

**목표:** 모델이 reasoning chain을 생성하도록 가르치기

**SFT 접근법:**

```
SFT Data:
{
  "prompt": "2020년에 태어난 곰은 2025년에 몇 살?",
  "completion": """
  <think>
  현재 연도: 2025
  태어난 연도: 2020
  나이 = 2025 - 2020 = 5
  </think>
  답: 5살
  """
}

→ 이런 데이터를 대량으로 수집 필요
```

**문제 1: 데이터 수집의 어려움**

```
Reasoning chains 작성은 매우 어려움:

간단한 문제:
- "2+3은?" → Chain 작성 쉬움

복잡한 문제:
- "동적 프로그래밍으로 LCS 구현"
- "복잡한 확률 문제 풀이"
→ Chain 작성 매우 어려움!

문제:
1. 전문가 필요 (비쌈)
2. 시간 소요 큼
3. 확장성 낮음
```

**구체적인 예시:**

```python
# 복잡한 문제
problem = """
이진 트리에서 두 노드의 최소 공통 조상을 찾는
알고리즘을 O(log n) 시간에 구현하세요.
"""

# Human이 작성해야 할 reasoning chain:
reasoning_chain = """
<think>
1. 문제 분석: LCA 찾기, O(log n) 제약
2. Naive 접근: O(n) - 제약 위반
3. 전처리 필요: Binary lifting?
4. Binary lifting 구조:
   - parent[node][i] = 2^i번째 조상
   - 전처리: O(n log n)
   - 쿼리: O(log n) ✓
5. 구현 전략:
   - DFS로 depth 계산
   - 2^i 조상 테이블 구축
   - LCA 쿼리 처리
6. 코드 작성...
</think>

[코드]
"""

→ 이런 chain을 수천 개 작성? 불가능!
```

**문제 2: Human Reasoning ≠ Model Reasoning**

**핵심 통찰:**

```
인간의 사고 방식 ≠ 모델의 사고 방식

인간:
- 추상적 개념 사용
- 경험 기반 직관
- 단계 생략 가능

모델:
- 토큰 기반 처리
- 통계적 패턴 의존
- 모든 단계 명시 필요
```

**예시:**

```python
# 인간의 reasoning
"이 문제는 DP로 풀면 되겠네"
→ 많은 중간 단계 생략

# 모델에게 더 유용한 reasoning
"""
1. 문제를 부분 문제로 분해
2. 부분 문제 간 관계: f(n) = f(n-1) + f(n-2)
3. 메모이제이션으로 중복 계산 제거
4. Bottom-up 구현
5. 초기값 설정
...
"""
→ 모든 단계를 토큰 레벨로 명시
```

**실제 관찰:**

```
OpenAI O1 논문에서:
"모델이 생성하는 reasoning chain은
 인간이 읽기 어려울 수 있음"

이유:
- 모델만의 internal representation
- 인간이 생각하지 않는 방식의 분해
- 반복적 시행착오 ("wait... let me try again")
```

**문제 3: Natural Reward 존재**

**핵심 관찰:**

```
Coding/Math 문제의 특별한 점:
→ 답이 맞는지 자동으로 검증 가능!

Coding:
if passes_all_tests(solution):
    reward = 1.0
else:
    reward = 0.0

Math:
if parsed_answer == ground_truth:
    reward = 1.0
else:
    reward = 0.0
```

**RL과의 완벽한 매칭:**

```
RL이 필요한 것:
1. Environment: 문제 ✓
2. Action: 답변 생성 ✓
3. Reward: 정답 여부 ✓ (자동 검증!)

→ RL을 사용하지 않을 이유가 없음!
```

**SFT vs RL 비교:**

| 측면 | SFT | RL |
|------|-----|-----|
| 데이터 | Human reasoning chains 필요 | 문제만 필요 |
| 비용 | 매우 비쌈 (전문가) | 저렴 (compute) |
| 확장성 | 낮음 | 높음 |
| Reasoning 품질 | Human-like | Model-optimized |
| Reward | 없음 | 자동 검증 |

## 6.2. 왜 RL을 사용하는가?

**RL이 이상적인 이유:**

### 1. 데이터 효율성

```
필요한 것:
- 문제 (이미 많음)
- Verifier (자동)

불필요한 것:
- Human reasoning chains
- 전문가 annotation
```

### 2. 자동 검증

```python
def verify_solution(problem, solution):
    """자동으로 답 검증"""
    if problem.type == "coding":
        return run_tests(solution, problem.tests)
    elif problem.type == "math":
        answer = parse_answer(solution)
        return answer == problem.ground_truth

→ Human in the loop 불필요!
```

### 3. 모델 최적 Reasoning

```
RL을 통한 학습:
- 모델이 자신만의 reasoning 전략 발견
- Human bias 없음
- 더 효과적인 경로 탐색

결과:
→ Human-written chain보다 더 나을 수 있음!
```

### 4. 확장성

```
SFT:
문제 10,000개 → Reasoning chains 10,000개 필요
→ 비용 $100,000+ (전문가 시간)

RL:
문제 10,000개 → 자동 학습
→ 비용: Compute만 (상대적으로 저렴)
```

## 6.3. Reward 설계

**목표:** 두 가지를 달성해야 함

```
1. Reasoning chain 생성하도록 유도
2. 정답을 맞추도록 유도
```

**Reward 구성요소:**

### Reward 1: Reasoning 존재 여부

```python
def check_reasoning_presence(output):
    """
    Reasoning chain이 있는지 확인
    """
    has_think_start = "<think>" in output
    has_think_end = "</think>" in output

    if has_think_start and has_think_end:
        return 1.0  # Reward
    else:
        return 0.0  # No reward
```

**예시:**

```
Good Output:
"<think>
계산: 2025 - 2020 = 5
</think>
답: 5살"
→ Reward = 1.0

Bad Output:
"답: 5살"
→ Reward = 0.0 (no thinking!)
```

### Reward 2: 정답 여부

```python
def check_correctness(output, problem):
    """
    답이 맞는지 확인
    """
    if problem.type == "coding":
        # 코딩 문제
        code = extract_code(output)
        passed = run_all_tests(code, problem.tests)
        return 1.0 if passed else 0.0

    elif problem.type == "math":
        # 수학 문제
        answer = parse_answer(output)
        correct = (answer == problem.ground_truth)
        return 1.0 if correct else 0.0
```

**예시:**

```python
# Coding
problem = {
    "description": "Sort a list",
    "tests": [
        ([3,1,2], [1,2,3]),
        ([5,5,5], [5,5,5]),
    ]
}

output = """
<think>리스트 정렬 필요</think>
def sort_list(arr):
    return sorted(arr)
"""

# Verification
code = extract_code(output)
test_results = [
    code([3,1,2]) == [1,2,3],  # True
    code([5,5,5]) == [5,5,5],  # True
]
reward = 1.0 if all(test_results) else 0.0
```

### 최종 Reward

```python
def compute_reward(output, problem):
    """
    최종 reward 계산
    """
    r1 = check_reasoning_presence(output)
    r2 = check_correctness(output, problem)

    # 두 가지 모두 만족해야 높은 reward
    if r1 == 1.0 and r2 == 1.0:
        return 1.0
    elif r1 == 1.0:
        return 0.3  # Reasoning은 있지만 오답
    elif r2 == 1.0:
        return 0.5  # 정답이지만 reasoning 없음
    else:
        return 0.0  # 둘 다 없음
```

**Reward Design 철학:**

```
단순함이 핵심!

복잡한 Reward Model 불필요:
✗ Human preference model
✗ Learned reward model

단순한 Rule-based Reward로 충분:
✓ Token 존재 여부 체크
✓ 자동 정답 검증

→ 이것만으로 훌륭한 성능!
```

## 6.4. RL 학습 결과

**DeepSeek R1-Zero 실험:**

**Setup:**

```
Model: DeepSeek Base (7B)
Task: AIM (수학 올림피아드)
Rewards:
  - Reasoning token presence
  - Answer correctness
```

**학습 곡선:**

```
Performance on AIM
  │
70%├─────────────────────┌─
  │                    ╱
60%├─────────────────╱
  │              ╱
50%├───────────╱
  │        ╱
40%├─────╱
  │   ╱
30%├─╱
  │
  └───────────────────────── RL Steps
    0  10K  20K  30K  40K  50K

관찰:
1. 초반 급격한 향상
2. 중반 완만한 향상
3. 후반 수렴
```

**주요 발견:**

```
1. Reasoning 능력 emergence:
   - RL 학습만으로 reasoning 능력 출현!
   - 명시적인 reasoning chain 없이도 학습

2. 성능 대폭 향상:
   - Baseline: 30% (SFT only)
   - After RL: 70% (+40%p!)

3. Reasoning 길이 증가:
   - 초기: ~100 tokens
   - 최종: ~1000 tokens
   → 모델이 더 깊게 사고하는 법 학습
```

**다른 벤치마크 결과:**

```
HumanEval (Coding):
- Before RL: 45%
- After RL:  72% (+27%p)

GSM-8K (Math):
- Before RL: 55%
- After RL:  88% (+33%p)

→ 모든 reasoning tasks에서 향상!
```

## 6.5. Compute Budget 제어

**문제:** 모든 문제가 같은 thinking을 필요로 하지 않음

**예시:**

```python
# 간단한 문제
Q: "2 + 2는?"
→ Thinking 1초면 충분

# 복잡한 문제
Q: "NP-완전 문제 증명"
→ Thinking 30초 필요

문제: 모델이 이를 어떻게 구분?
```

### 문제 1: Dynamic Budget

**목표:** 문제 난이도에 따라 thinking 시간 조절

**접근법 1: Classifier 사용**

```python
class DifficultyClassifier:
    """문제 난이도 분류"""
    def classify(self, problem):
        # 간단한 heuristics
        if len(problem) < 100:
            return "easy"
        elif "algorithm" in problem or "proof" in problem:
            return "hard"
        else:
            return "medium"

# 사용
difficulty = classifier.classify(problem)

if difficulty == "easy":
    max_thinking_tokens = 100
elif difficulty == "medium":
    max_thinking_tokens = 500
else:  # hard
    max_thinking_tokens = 2000
```

**접근법 2: Learned Budget**

```python
# 모델 자체가 필요한 compute 예측
budget_predictor = train_budget_model(
    inputs=problems,
    targets=optimal_thinking_tokens
)

# 추론 시
thinking_budget = budget_predictor(problem)
output = model.generate(
    problem,
    max_tokens=thinking_budget
)
```

### 문제 2: Context Length Awareness

**문제:**

```
LLM의 Context Window는 제한적:

GPT-4:     8K tokens
Claude:    100K tokens
LLaMA:     4K tokens

Thinking도 context를 소비:
Input (100) + Thinking (1000) + Answer (50)
= 1150 tokens

→ Context 관리 필요!
```

**해결책: Budget Forcing**

**기법 1: Continue Thinking**

```python
# 모델 output
output = """
<think>
단계 1: ...
단계 2: ...
단계 3: ...
</think>
"""

# 더 think하도록 강제
if not solved and tokens_left > 500:
    # "wait" token 주입
    forced_output = output + "\nWait, let me reconsider..."

    # 계속 생성
    continued = model.generate(
        prompt=problem + forced_output,
        max_tokens=500
    )
```

**기법 2: Stop Thinking**

```python
# Context limit 근접 시
if current_tokens > 0.9 * max_context:
    # Stop signal 주입
    forced_stop = "\nTime's up. My final answer is:"

    output = current_thinking + forced_stop
    answer = model.generate(
        prompt=problem + output,
        max_tokens=100  # 답변만
    )
```

**S1 Paper의 Budget Forcing:**

```
Special Tokens:
- "wait": 계속 생각하라는 신호
- "solution": 이제 답하라는 신호

예시 output:
"""
<think>
단계 1: 문제 분석...
단계 2: 접근법 고려...
wait  ← 모델 생성 또는 강제 주입
단계 3: 다른 방법 시도...
단계 4: 검증...
solution  ← 답변 시작 신호
</think>

답: ...
"""
```

### 문제 3: Continuous Thought

**Motivation:**

```
현재: Thinking in language space
- <think> ... 텍스트 ... </think>
- 인간이 읽을 수 있음
- 하지만 비효율적 (많은 토큰)

아이디어: Thinking in latent space
- Hidden states로 thinking
- 더 압축적
- 더 표현력 있음
```

**구현 아이디어:**

```python
class ContinuousThought(nn.Module):
    def __init__(self, hidden_dim):
        self.thought_layers = nn.TransformerEncoder(...)

    def forward(self, x):
        # Regular tokens
        hidden = self.embed(x)

        # Continuous thought
        # 특수 토큰 대신 hidden states로 thinking
        thought_states = self.thought_layers(hidden)

        # Continue generation
        output = self.decode(thought_states)

        return output
```

**장점:**

```
1. 압축:
   - Text: "First, let's calculate..." (5 tokens)
   - Latent: Single thought state

2. 표현력:
   - 언어로 표현 못할 개념도 가능

3. 효율성:
   - 적은 compute로 더 깊은 thinking
```

**최근 연구:**

```
Papers (2024-2025):
- "Thought Tokens" (2024.11)
- "Latent Reasoning" (2024.12)
- "Implicit Chain-of-Thought" (2025.01)

→ 활발한 연구 분야!
```

---

# 7. GRPO (Group Relative Policy Optimization)

## 7.1. GRPO란?

**정의:**

```
GRPO = Group Relative Policy Optimization

2024년 발표된 RL 알고리즘
Reasoning tasks에 최적화됨
현재 대부분의 reasoning models이 사용
```

**핵심 아이디어:**

```
PPO처럼 두 가지 목표:
1. Advantage 최대화
2. KL divergence로 변화 제한

차이점:
→ Advantage 계산 방식이 다름!
```

## 7.2. GRPO의 핵심 아이디어

**PPO의 Advantage:**

```
PPO Advantage:
- Reward 사용
- Value function 사용 (별도 학습!)
- GAE (Generalized Advantage Estimation)

문제점:
- Value function 학습 필요
- 복잡한 구현
- 불안정할 수 있음
```

**GRPO의 Advantage:**

```
GRPO Advantage:
- 같은 prompt에 대해 여러 completions 생성
- 각 completion의 reward 계산
- 그룹 평균과 비교!

핵심:
Advantage = (이 completion의 reward) - (그룹 평균 reward)
```

**직관적 이해:**

```python
# 수학 문제
problem = "2020년 태어난 곰은 2025년에 몇 살?"

# 같은 문제로 여러 번 생성 (g=5)
completions = [
    "<think>...</think> 답: 5살",    # Correct (r=1.0)
    "<think>...</think> 답: 4살",    # Wrong   (r=0.0)
    "<think>...</think> 답: 5살",    # Correct (r=1.0)
    "<think>...</think> 답: 6살",    # Wrong   (r=0.0)
    "<think>...</think> 답: 5살",    # Correct (r=1.0)
]

# Rewards
rewards = [1.0, 0.0, 1.0, 0.0, 1.0]
mean_reward = 0.6

# Advantages
advantages = [
    1.0 - 0.6 = +0.4,  # Good! 평균보다 좋음
    0.0 - 0.6 = -0.6,  # Bad! 평균보다 나쁨
    1.0 - 0.6 = +0.4,  # Good!
    0.0 - 0.6 = -0.6,  # Bad!
    1.0 - 0.6 = +0.4,  # Good!
]
```

**Standardization:**

실제로는 단순 평균이 아닌 표준화 사용:

```python
def compute_advantages(rewards):
    """
    GRPO advantage 계산
    """
    mean = np.mean(rewards)
    std = np.std(rewards)

    advantages = (rewards - mean) / (std + 1e-8)
    return advantages

# 예시
rewards = [1.0, 0.0, 1.0, 0.0, 1.0]
advantages = compute_advantages(rewards)
# advantages ≈ [0.63, -1.26, 0.63, -1.26, 0.63]
```

**왜 Standardization?**

```
1. Scale-invariant:
   - Rewards [1.0, 0.0] vs [10.0, 0.0]
   - 둘 다 같은 advantages

2. 안정적 학습:
   - Gradient가 안정적
   - Learning rate 조정 쉬움

3. 상대적 비교:
   - 절대값이 아닌 상대적 우열
```

## 7.3. GRPO vs PPO

### PPO 프로세스

**Step 1: 데이터 수집**

```
Query → Policy → Completion
  ↓
Reward Model → Reward
  ↓
Value Function → Value
```

**Step 2: Advantage 계산**

```python
# Per-token rewards
rewards[t] = {
    reward_completion  if t == last_token
    -β * KL[t]         otherwise
}

# Value prediction
values[t] = value_function(state[t])

# GAE (복잡!)
advantages = compute_gae(rewards, values, γ, λ)
```

**Step 3: 학습**

```python
# 두 개의 모델 업데이트
policy_loss = compute_policy_loss(advantages)
value_loss = compute_value_loss(rewards, values)

# 동시 학습
optimize(policy_loss + value_loss)
```

**PPO 다이어그램:**

```
     ┌────────┐
     │ Query  │
     └────┬───┘
          │
     ┌────▼────┐
     │ Policy  │
     └────┬────┘
          │
     ┌────▼─────────┐
     │ Completion   │
     └────┬─────────┘
          │
     ┌────▼────┐        ┌──────────────┐
     │ Reward  │        │ Value Func   │
     │ Model   │        │ (학습 필요!)  │
     └────┬────┘        └──────┬───────┘
          │                    │
          └────────┬───────────┘
                   │
              ┌────▼─────┐
              │   GAE    │
              │(복잡한 계산)│
              └────┬─────┘
                   │
            ┌──────▼──────┐
            │ Advantages  │
            └─────────────┘
```

### GRPO 프로세스

**Step 1: 데이터 수집**

```
Query → Policy → g Completions (같은 prompt!)
  ↓
Reward Model → g Rewards
```

**Step 2: Advantage 계산**

```python
# Group-wise advantage (간단!)
mean_reward = np.mean(rewards)
std_reward = np.std(rewards)

advantages = (rewards - mean_reward) / (std_reward + 1e-8)
```

**Step 3: 학습**

```python
# 하나의 모델만 업데이트
policy_loss = compute_policy_loss(advantages)

# Policy만 학습
optimize(policy_loss)
```

**GRPO 다이어그램:**

```
     ┌────────┐
     │ Query  │
     └────┬───┘
          │
     ┌────▼────┐
     │ Policy  │───┐
     └─────────┘   │
                   ├─→ Completion 1 ─→ Reward 1
                   ├─→ Completion 2 ─→ Reward 2
                   ├─→ Completion 3 ─→ Reward 3
                   ├─→ Completion 4 ─→ Reward 4
                   └─→ Completion 5 ─→ Reward 5
                          │
                     ┌────▼──────┐
                     │   Mean    │
                     │   Std     │
                     └────┬──────┘
                          │
                   ┌──────▼──────┐
                   │ Advantages  │
                   └─────────────┘
```

### 상세 비교

**Advantage 계산:**

```python
# PPO
class PPOAdvantage:
    def __init__(self):
        self.value_function = ValueNetwork()  # 별도 모델!

    def compute(self, rewards, states):
        values = self.value_function(states)
        advantages = compute_gae(rewards, values)  # 복잡!
        return advantages

# GRPO
class GRPOAdvantage:
    def compute(self, rewards):
        # 간단한 통계!
        mean = rewards.mean()
        std = rewards.std()
        advantages = (rewards - mean) / (std + 1e-8)
        return advantages
```

**메모리 사용:**

```
PPO:
- Policy network: X GB
- Value network: X GB (추가!)
- KV cache: Y GB
Total: 2X + Y GB

GRPO:
- Policy network: X GB
- KV cache: Y GB
Total: X + Y GB

→ GRPO가 더 메모리 효율적!
```

**구현 복잡도:**

```python
# PPO 구현 (Pseudo-code)
class PPO:
    def __init__(self):
        self.policy = PolicyNetwork()
        self.value_function = ValueNetwork()  # 추가!
        self.policy_optimizer = Adam()
        self.value_optimizer = Adam()  # 추가!

    def update(self, batch):
        # 1. Value function 학습
        value_loss = self.compute_value_loss(batch)
        self.value_optimizer.step(value_loss)

        # 2. GAE 계산
        advantages = self.compute_gae(batch)

        # 3. Policy 학습
        policy_loss = self.compute_policy_loss(advantages)
        self.policy_optimizer.step(policy_loss)

→ 복잡!

# GRPO 구현
class GRPO:
    def __init__(self):
        self.policy = PolicyNetwork()
        self.optimizer = Adam()

    def update(self, batch):
        # 1. Group advantage 계산
        advantages = (batch.rewards - batch.rewards.mean()) / batch.rewards.std()

        # 2. Policy 학습
        policy_loss = self.compute_policy_loss(advantages)
        self.optimizer.step(policy_loss)

→ 간단!
```

**비교 표:**

| 측면 | PPO | GRPO |
|------|-----|------|
| Value function | 필요 ✓ | 불필요 ✗ |
| 추가 모델 | 1개 | 0개 |
| 메모리 | 2X | X |
| 구현 복잡도 | 높음 | 낮음 |
| Advantage 계산 | GAE (복잡) | 통계 (간단) |
| Per-token rewards | 필요 | 불필요 |
| 그룹 생성 | 불필요 | 필요 (g개) |
| 학습 안정성 | 보통 | 높음 |

## 7.4. GRPO의 장점

### 장점 1: No Value Function

**의미:**

```
Value function이 없으면:
- 학습해야 할 모델 1개 감소
- 메모리 절약
- 구현 단순화
```

**Value function의 문제:**

```
1. 추가 학습 필요:
   - Separate optimizer
   - Separate loss
   - 수렴 어려움

2. Instability:
   - Value prediction 부정확하면
   - Advantage estimation 영향
   - 학습 불안정

3. 하이퍼파라미터:
   - Value loss weight
   - Value learning rate
   - 조정 어려움
```

### 장점 2: 간단한 구현

**코드 비교:**

```python
# GRPO - 핵심 부분만
def grpo_step(policy, problems, group_size=4):
    for problem in problems:
        # 1. 그룹 생성
        completions = [policy.generate(problem) for _ in range(group_size)]

        # 2. Rewards
        rewards = [compute_reward(c, problem) for c in completions]

        # 3. Advantages
        mean, std = np.mean(rewards), np.std(rewards)
        advantages = (rewards - mean) / (std + 1e-8)

        # 4. Policy update
        loss = -sum(log_prob(c) * adv for c, adv in zip(completions, advantages))
        loss.backward()
        optimizer.step()

→ ~20 lines!
```

### 장점 3: 상대적 비교

**왜 중요한가?**

```python
# 쉬운 문제
problem_easy = "2 + 2 = ?"

completions = ["4", "5", "4", "3", "4"]
rewards = [1.0, 0.0, 1.0, 0.0, 1.0]
mean = 0.6

advantages = [+0.4, -0.6, +0.4, -0.6, +0.4]
→ 정답이 평균보다 좋다는 것을 학습

# 어려운 문제
problem_hard = "Prove P≠NP"

completions = [wrong1, wrong2, wrong3, wrong4, wrong5]
rewards = [0.0, 0.0, 0.0, 0.0, 0.0]
mean = 0.0

advantages = [0.0, 0.0, 0.0, 0.0, 0.0]
→ 모두 같으므로 업데이트 없음!

의미:
- 어려운 문제에서 무의미한 업데이트 방지
- 상대적으로 더 나은 솔루션에 집중
```

### 장점 4: 자연스러운 Baseline

**Advantage의 의미:**

```
Advantage = Reward - Baseline

PPO의 Baseline: Value function (학습 필요)
GRPO의 Baseline: Group mean (자동 계산)

GRPO Baseline의 장점:
1. 학습 불필요
2. 항상 최신 (current policy의 평균)
3. Problem-specific (각 문제마다 다른 baseline)
```

**예시:**

```python
# Problem A (easy)
rewards_A = [1.0, 1.0, 0.0, 1.0, 1.0]
baseline_A = 0.8  # 높은 baseline

# Problem B (hard)
rewards_B = [0.0, 0.0, 0.0, 0.0, 0.0]
baseline_B = 0.0  # 낮은 baseline

→ 각 문제의 난이도를 자동으로 반영!
```

## 7.5. GRPO의 구현

### 전체 알고리즘

```python
def grpo_train(
    policy,              # LLM to train
    ref_policy,          # Reference policy (SFT)
    problems,            # Training problems
    group_size=4,        # g: samples per prompt
    β=0.01,             # KL penalty coefficient
    epochs=10,
    batch_size=32
):
    optimizer = Adam(policy.parameters())

    for epoch in range(epochs):
        for batch in batch_iterator(problems, batch_size):
            # 1. Generate completions
            all_completions = []
            all_rewards = []
            all_ref_logprobs = []

            for problem in batch:
                # Sample g completions
                completions = [
                    policy.generate(problem)
                    for _ in range(group_size)
                ]

                # Compute rewards
                rewards = [
                    compute_reward(c, problem)
                    for c in completions
                ]

                # Reference log probs (for KL)
                ref_logprobs = [
                    ref_policy.log_prob(problem, c)
                    for c in completions
                ]

                all_completions.extend(completions)
                all_rewards.extend(rewards)
                all_ref_logprobs.extend(ref_logprobs)

            # 2. Compute advantages (group-wise)
            advantages = []
            for i in range(0, len(all_rewards), group_size):
                group_rewards = all_rewards[i:i+group_size]

                # Standardize within group
                mean = np.mean(group_rewards)
                std = np.std(group_rewards) + 1e-8
                group_advantages = (group_rewards - mean) / std

                advantages.extend(group_advantages)

            # 3. Compute policy loss
            policy_loss = 0
            kl_loss = 0

            for completion, advantage, ref_logprob in zip(
                all_completions, advantages, all_ref_logprobs
            ):
                # Log probability
                logprob = policy.log_prob(problem, completion)

                # Policy gradient
                policy_loss -= logprob * advantage

                # KL divergence
                kl = logprob - ref_logprob
                kl_loss += β * kl

            # 4. Total loss
            total_loss = policy_loss + kl_loss

            # 5. Update
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {total_loss.item()}")
```

### Loss Function 상세

**GRPO Loss:**

```
L = L_policy + L_KL

L_policy = -E[log π(a|s) * A]
L_KL = β * KL(π || π_ref)
```

**구체적 구현:**

```python
def compute_grpo_loss(
    policy,
    ref_policy,
    problem,
    completions,
    rewards,
    β=0.01
):
    # 1. Compute advantages
    mean_reward = torch.mean(rewards)
    std_reward = torch.std(rewards) + 1e-8
    advantages = (rewards - mean_reward) / std_reward

    # 2. Policy loss
    policy_loss = 0
    for completion, advantage in zip(completions, advantages):
        logprob = policy.log_prob(problem, completion)
        policy_loss -= logprob * advantage

    # 3. KL loss
    kl_loss = 0
    for completion in completions:
        policy_logprob = policy.log_prob(problem, completion)
        ref_logprob = ref_policy.log_prob(problem, completion)
        kl = policy_logprob - ref_logprob
        kl_loss += β * kl

    return policy_loss + kl_loss
```

### 실전 팁

**1. Group Size 선택:**

```python
# Trade-off
group_size = 4:   빠름, advantage 추정 noisy
group_size = 8:   균형
group_size = 16:  느림, advantage 추정 정확

# 권장
training:   group_size = 8~16
evaluation: group_size = 1 (Pass@1 측정)
```

**2. Temperature 설정:**

```python
# Exploration-exploitation balance
temperature = 0.6~0.8

너무 낮으면: diversity 부족
너무 높으면: low quality samples
```

**3. Reward Shaping:**

```python
def shaped_reward(output, problem):
    """더 informative한 reward"""
    base_reward = compute_reward(output, problem)

    # Bonus for reasoning
    has_reasoning = 1.0 if "<think>" in output else 0.0

    # Bonus for length (적당한 reasoning 길이)
    reasoning_length = len(extract_reasoning(output))
    length_bonus = min(reasoning_length / 1000, 1.0)

    # Combined
    return base_reward + 0.1 * has_reasoning + 0.1 * length_bonus
```

**4. Learning Rate Schedule:**

```python
# Warm-up + Decay
def get_lr(step, warmup_steps=1000, total_steps=10000):
    if step < warmup_steps:
        # Linear warm-up
        return (step / warmup_steps) * base_lr
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return base_lr * 0.5 * (1 + cos(pi * progress))
```

---

# 8. 요약

## 핵심 개념

### 1. Reasoning이란?

```
정의: 문제를 multi-step process로 해결하는 능력

특징:
- 단계별 분해
- 중간 계산
- 자기 검증

적용: 수학, 코딩, 논리 문제
```

### 2. Reasoning Models

```
Vanilla LLM:
Input → Output

Reasoning Model:
Input → Reasoning Chain → Output

핵심:
- <think> tokens
- Extended inference
- Verifiable tasks
```

### 3. 왜 RL?

```
SFT 문제점:
✗ Reasoning chains 수집 어려움
✗ Human reasoning ≠ Model reasoning
✗ 확장성 낮음

RL 장점:
✓ 자동 검증 가능
✓ 모델 최적 reasoning
✓ 확장 가능
```

### 4. GRPO Algorithm

```
핵심 아이디어:
- Group-wise advantage
- No value function
- 상대적 비교

Process:
1. 같은 prompt로 g개 생성
2. Rewards 계산
3. Group mean과 비교
4. Policy 업데이트
```

## 실전 체크리스트

**Reasoning Model 구축 시:**

1. **데이터 준비**
   - [ ] 문제 데이터셋 수집
   - [ ] Verifier 구현 (test cases/ground truth)
   - [ ] Reward function 설계

2. **모델 선택**
   - [ ] Base model 선택 (SFT 완료된 모델)
   - [ ] Thinking tokens 정의 (<think>, </think>)
   - [ ] Context length 확인

3. **RL 설정**
   - [ ] Algorithm: GRPO 권장
   - [ ] Group size: 8-16
   - [ ] Temperature: 0.6-0.8
   - [ ] KL coefficient β: 0.01-0.1

4. **학습**
   - [ ] Warmup steps 설정
   - [ ] Learning rate schedule
   - [ ] Gradient clipping
   - [ ] Checkpointing

5. **평가**
   - [ ] Pass@1, Pass@5, Pass@10
   - [ ] Benchmark 성능
   - [ ] Reasoning length 분석
   - [ ] Cost 분석

6. **Deployment**
   - [ ] Inference optimization
   - [ ] Compute budget 제어
   - [ ] Caching strategy
   - [ ] Monitoring

**하이퍼파라미터 가이드:**

```python
# GRPO 기본 설정
config = {
    "group_size": 8,
    "temperature": 0.7,
    "β_kl": 0.05,
    "learning_rate": 1e-6,
    "warmup_steps": 100,
    "max_thinking_tokens": 2000,
    "batch_size": 32,
}

# 문제 유형별 조정
if task == "coding":
    config["group_size"] = 16      # 더 많은 diversity
    config["temperature"] = 0.8
elif task == "math":
    config["group_size"] = 8
    config["temperature"] = 0.7
```

---

# 9. 중요 용어 정리

**Reasoning 관련:**
- **Reasoning**: 문제를 multi-step process로 해결하는 능력
- **Chain of Thought (CoT)**: 단계별 사고 과정을 명시적으로 생성
- **Reasoning Chain**: 모델이 생성하는 사고 과정 (<think>...</think>)
- **Compute Budget**: 답변 생성에 사용할 수 있는 토큰/시간 예산

**Models 관련:**
- **Vanilla LLM**: 기본 LLM (reasoning chain 없이 바로 답변)
- **Reasoning Model**: Reasoning chain을 생성하는 LLM
- **DeepSeek R1**: 2025년 1월 공개된 reasoning model (오픈소스)
- **O1**: OpenAI의 reasoning model

**Benchmarks 관련:**
- **HumanEval**: 코딩 문제 benchmark (~164 problems)
- **CodeForces**: 경쟁 프로그래밍 문제
- **SWE-bench**: 실제 GitHub issues 기반 benchmark
- **AIM**: 미국 수학 올림피아드 예선 문제
- **GSM-8K**: 초등학교 수준 수학 문제 (8,000개)

**Metrics 관련:**
- **Pass@k**: k번 시도 시 최소 1개 성공 확률
- **Pass@1**: 한 번에 성공할 확률 (= accuracy)
- **Consensus@k**: k개 중 가장 많이 나온 답 선택
- **Exact Match**: 생성된 답이 정답과 정확히 일치하는 비율

**RL 관련:**
- **GRPO**: Group Relative Policy Optimization
- **Group Size (g)**: 같은 prompt에 대해 생성할 completions 수
- **Advantage**: 현재 action이 baseline보다 얼마나 좋은지
- **Group-wise Advantage**: 그룹 내 다른 samples과 비교한 advantage
- **Value Function**: Future rewards 예측 (PPO에서 사용)
- **GAE**: Generalized Advantage Estimation (PPO의 advantage 계산)

**Reward 관련:**
- **Verifiable Task**: 자동으로 정답 검증 가능한 task
- **Rule-based Reward**: 학습된 모델 대신 규칙 기반 reward
- **Reward Shaping**: Reward��� 추가 정보 포함하여 학습 유도

**Budget Control 관련:**
- **Dynamic Budget**: 문제 난이도에 따라 thinking 시간 조절
- **Budget Forcing**: 특수 토큰으로 thinking 제어
- **Continuous Thought**: Language space 대신 latent space에서 thinking

**Implementation 관련:**
- **Thinking Tokens**: Reasoning을 표시하는 특수 토큰 (<think>, </think>)
- **Thought Summary**: Raw reasoning chain의 요약 (사용자에게 표시)
- **Distillation Attack**: Reasoning chain을 훔쳐서 모델 학습
- **Context Length Awareness**: Context window 제한 고려

---

**다음 강의 예고:**

Lecture 7에서는 LLM의 나머지 약점들을 다룹니다:
- **Static Knowledge**: RAG (Retrieval-Augmented Generation)
- **No Action**: Agents와 Tool Use
- **Multi-modal**: Vision, Audio 등 확장

---

**수고하셨습니다!** 🎉
