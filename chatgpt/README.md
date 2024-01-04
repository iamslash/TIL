- [Abstract](#abstract)
- [Materials](#materials)
- [How To Train ChatGPT](#how-to-train-chatgpt)
  - [Pretrained model 만들기](#pretrained-model-만들기)
  - [Supervised Fine Tuning(SFT) model 만들기](#supervised-fine-tuningsft-model-만들기)
  - [Reward model 만들기](#reward-model-만들기)
- [Prompt Engineering](#prompt-engineering)
  - [구성 요소](#구성-요소)
  - [우선 순위](#우선-순위)
  - [Task](#task)
  - [Context](#context)
  - [Example](#example)
  - [Persona](#persona)
  - [Form](#form)
  - [Tone](#tone)

-----

# Abstract

ChatGpt 에 대해 정리한다.

# Materials

- [ChatGPT 모델의 동작원리](https://yoonheo91.tistory.com/26)

# How To Train ChatGPT

ChatGPT는 세 단계의 학습 과정을 거쳐 성능을 높입니다. 아래에서 각 단계를 더
자세히 설명하겠습니다.

## Pretrained model 만들기

- 데이터 수집: 수십 억 개의 문장, 웹 페이지, 문서 등으로부터 데이터를
  수집합니다. 이 데이터는 인터넷 상의 다양한 문서로부터 가져온 것입니다.
- 데이터 전처리: 토큰화를 통해 데이터를 처리 가능한 형태로 변환합니다. 토큰화는
  문장을 개별 토큰(단어, 구두점 등)으로 나누는 과정입니다.
- Unsupervised learning: 많은 데이터를 가지고, 다음 단어 예측에 초점을 맞추어
  학습하게 됩니다. 이미 주어진 문장 안에서 다음에 나올 단어가 무엇인지 추측하며
  학습합니다.

첫 번째 단계에서 학습한 모델은 문법, 도메인 지식, 문맥 이해 등 기본적인 언어
처리 능력을 갖추게 됩니다.

## Supervised Fine Tuning(SFT) model 만들기

- 레이블링: 전문가, 튜터, 교육받은 사람들이 직접 프롬프트(질문)과 이에 대한
  적절한 답변 데이터를 만듭니다.
- SFT 학습: 이렇게 얻어진 데이터셋을 가지고 모델을 더 정교하게 학습시킵니다.
  (supervised learning) 다음 단어 예측 방식은 유지되며, 이 과정을 통해 모델이 더
  정확한 답변을 생성하도록 만듭니다.

## Reward model 만들기

강화학습을 활용합니다.

- 대답 생성: 기존의 SFT 모델로 특정 프롬프트에 대한 여러 다양한 대답을
  생성합니다.
- 대답 평가: 평가자(레이블러)가 여러 개 생성된 대답에 순위를 매깁니다.
- Reward function 생성: 평가자의 순위 정보를 바탕으로 모델이 해당 대답에 대한
  보상을 계산할 수 있도록 보상 함수(reward model)를 만듭니다.
- 강화학습 적용: 보상 함수를 기반으로 모델이 최고의 대답을 출력하도록 강화학습을
  진행합니다. (LLM-RL, LLM-SFT)
- 보상 모델 업데이트: 각 결과마다 reward model을 업데이트 하여 점점 더
  개선됩니다.

이렇게 세 단계를 거치는 과정은 모델이 전반적으로 높은 성능을 낼 수 있도록
합니다. 또한, 많은 데이터와 인간의 도움을 받아 세부적인 교정 및 강화학습을 통해
지속적으로 성능을 개선합니다. 이로 인해 ChatGPT 모델은 사용자의 질문에 대해
상황에 적합하고, 정확한 답변을 생성할 수 있는 능력을 지니게 됩니다.

# Prompt Engineering

- [Cognitive Prompting 정리(1zlabs Ken. Ver.231207 공유용) | GoogleSlide](https://docs.google.com/presentation/d/1kayepoiTVT838Tetk02nxeqVmmS9BDc9O7n-4OzJdL8/edit#slide=id.g26309fa4a80_0_0)
  - [video](https://www.youtube.com/watch?v=CkCL8dV_mPk)
- [프롬프트사용법: 가장 완벽한 GPT 프롬프트 만드는법](https://www.youtube.com/watch?v=olRqEoiWy6Q)

## 구성 요소

1. 작업 (Task)
2. 맥락 (Context)
3. 예시 (Example)
4. 페르소나 (Persona)
5. 형식 (Form)
6. 어조 (Tone)

## 우선 순위

* Task 는 필수이다.
* Context, Example 은 더 좋은 답에 도움이 된다.
* Persona, Form, Tone 은 없어도 관계 없다.

## Task

명료하게 작성해야 한다. 

```
생성해 주세요.
요약해 주세요.
분석해 주세요.
```

## Context

작성하기 까다롭다. 다음과 같은 3 가지 요소가 중요하다.

- 배경 (나의 배경은?)
- 목표 (이 프롬프트의 목적은?)
- 환경 (환경적 요인은?)

```
나는 고혈압이 있고, 80 kg 의 남성이야
20kg 의 체중을 감량하려고 해
그러나 두변에 운동할 공간이 없어 
나에게 맞는 다이어트 프로그램을 생성해줘
```

## Example

예를 전해 주면 결과가 좋아진다.

```
다음은 입사지원서 양식의 예이다.
```

## Persona

인격을 정해 주면 결과가 좋아진다. 빌게이츠, 스티브잡스 등등.

```
너는 수학 선생님이야
너는 마케팅 전문가야
```

## Form

표, 리스트, markdown 등등 출력형식을 사용할 수 있다.

## Tone

다음은 주요 부사들이다. 

```
착하게
따뜻하게
부드럽게
강하고 명료하게
예의바르게
```

마땅한 부사가 생각나지 않으면 ChatGpt 에게 물어봐도 좋다.

```
사직서를 작성하려고 한다. 어떤 톤으로 쓰면 좋을까?
```
