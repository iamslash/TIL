# Codex Under the Hood

Codex CLI는 OpenAI의 터미널 기반 코딩 에이전트이다. 사용자가 프롬프트를 입력하면 Codex가 지침, 로컬 설정, 세션 컨텍스트를 조합해 모델을 호출하고, 셸 명령이나 MCP 서버 같은 도구를 실행하며, 그 결과를 다시 모델 루프에 넣어 작업을 이어간다. 이 로컬 환경에는 oh-my-codex(OMX)가 함께 설치되어 있어 `AGENTS.md`, `prompts/`, `agents/`, `skills/`, MCP 서버를 통해 오케스트레이션 레이어가 추가된다.

이 문서들은 최근 로컬 설치 상태와 `pwd` 시나리오를 기준으로 Codex의 동작을 해부한 기술 메모이다.

> **근거 표기**: 각 문서 상단에 `source-inspected` (로컬 파일 또는 공식 문서 직접 확인), `runtime-observed` (실행 중 관찰), `inferred` (구조로부터 추정) 태그를 표기하였다. 단정형 문장이라도 태그가 `inferred`이면 추정이다.

## Contents

- [Codex CLI API 및 세션 이벤트 구조](codex-cli-api-kr.md) - Responses 스타일 아이템, session JSONL, tool call/event 구조
- [Codex CLI 동작 원리](codex-cli-architecture-kr.md) - CLI, AGENTS, prompts, agents, skills, MCP, persistence 레이어
- [Codex CLI 실행 흐름 다이어그램](codex-cli-flow-diagram-kr.md) - `pwd` 요청이 session JSONL과 도구 호출로 흘러가는 순서
- [Codex CLI 설정 파일 구조](codex-cli-config-kr.md) - `~/.codex/config.toml`, feature, project trust, cache/state 파일
- [Codex CLI 캐시 구조](codex-cli-caching-kr.md) - `cached_input_tokens`, model metadata cache, cloud requirements cache

## 표기 규칙

- 도구명: 로컬 session JSONL과 CLI 출력에서 관찰된 이름을 그대로 사용
- 경로/버전: 문서 상단의 관측 시점 블록 기준
- 예시 JSON/다이어그램: 실제 패킷 캡처가 아니라 구조 설명용 의사 코드 또는 재구성 시퀀스

## 관측 환경

> **관측 시점**: 2026-03-26. 아래 값은 이 시점 기준이며 Codex CLI/모델/OMX 업데이트에 따라 달라질 수 있다.

| 항목 | 값 |
|------|------|
| Codex CLI 버전 | `0.116.0` |
| 기본 모델 | `gpt-5.4` |
| 기본 reasoning effort | `high` |
| config 상 context window | `800000` |
| 주 작업 경로 | `/Users/david.sun/prj/github` |
| 로컬 오케스트레이션 | oh-my-codex 설치됨 |
