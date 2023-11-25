# Abstract

# Materials

- [LLM 애플리케이션 아키텍처란? (RAG의 이해와 기술 스택 탐색) | facebook](https://www.facebook.com/aldente0630/posts/2579269092237469)
- [Building RAG-based LLM Applications for Production](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1)

# Basic

## LLM Architecture

대형 언어 모델(LLM) 애플리케이션 아키텍처에서 중요한 점은 정보 부족과 제한된
답변 능력을 보완하는 것이다. 이를 위해 검색 증강 생성(RAG) 아키텍처를 사용하여
데이터베이스에 저장된 정보를 검색하고, 프롬프트를 통해 LLM에 전달한다. RAG
아키텍처의 성능은 임베딩, 벡터 DB, LLM 선택에 따라 달라진다. 다양한 LLMOps
도구들이 생태계를 구축하고 있으며, 클라우드 공급자 서비스에서 전체 RAG 아키텍처
호스팅이 이루어진다.

## RAG

RAG(Retrieval Augmented Generation)는 대형 언어 모델(LLM)에 검색 기능을 추가하는
아키텍처입니다. 이 방식은 사용자의 질문에 대해 관련된 정보를 검색하고, 해당
정보가 포함된 문서를 프롬프트 콘텍스트로 LLM에 전달하여 더 정확한 응답을
생성하는 것을 목표로 합니다.

기본적으로 RAG는 크게 세 가지 작업 흐름으로 구성됩니다.

- 정보 저장: 관련 정보를 데이터베이스(DB)에 저장합니다. 자연어로 작성된 문서는
  벡터 형태(임베딩)로 변환한 후 벡터 DB에 저장합니다.
- 검색: 사용자의 질문이 들어오면, 질문을 임베딩 벡터로 변환하고 벡터 DB에서 가장
  가까운 문서들을 검색하여 가져옵니다.
- 생성: 관련 있는 문서들을 프롬프트 콘텍스트로 입력한 후, LLM에게 답변을
  요청하여 응답을 생성합니다.

이러한 RAG 아키텍처를 사용하면 LLM의 정보 부족과 제한된 답변 능력 문제를 어느 정도 해결할 수 있으며, 더욱 효율적인 대화형 AI 시스템을 구축할 수 있습니다.
