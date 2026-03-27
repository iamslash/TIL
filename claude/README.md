# Claude Under the Hood

Claude Code는 Anthropic의 CLI 기반 AI 코딩 도구이다. 사용자가 터미널에서 텍스트를 입력하면, Claude Code가 Anthropic Messages API를 통해 LLM과 통신하고, 도구(파일 읽기, 셸 실행 등)를 호출하며, 결과를 다시 LLM에 전달하는 루프를 반복한다. OMC(oh-my-claudecode)는 이 Claude Code 위에서 동작하는 서드파티 플러그인으로, 훅과 MCP 서버를 통해 자율 실행 모드, 스킬 시스템 등을 추가한다.

이 문서들은 "hello-slack"이라는 하나의 시나리오를 기준으로 내부 동작을 추적한 기술 메모이다.

> **근거 표기**: 각 문서 상단에 `source-inspected` (소스 코드 직접 확인), `runtime-observed` (실행 중 관찰), `inferred` (구조로부터 추정) 태그를 표기하였다. 단정형 진술이라도 태그가 `inferred`이면 추정이다.

## Contents

- [Claude Code API 통신 구조](claude-code-api-kr.md) - Messages API의 메시지 타입, 도구 호출 루프, 컨텍스트 누적
- [Claude Code 바이너리 내부 구조](claude-code-internals-kr.md) - Bun 바이너리 역분석: 시스템 프롬프트, 도구 시스템, ToolSearch, Speculation, Compaction
- [OMC 동작 원리](omc-architecture-kr.md) - 플러그인 구조, 훅, MCP 서버, 스킬 시스템, Python 브릿지
- [OMC 실행 흐름 다이어그램](omc-flow-diagram-kr.md) - "hello-slack" 실행 시 시퀀스 다이어그램
- [Claude Code 설정 파일 구조](claude-code-config-kr.md) - ~/.claude.json, settings.json, MCP 서버 등록 위치
- [Claude Code Prompt Caching](claude-code-caching-kr.md) - KV Cache 재사용, 비용 절약, Transformer Attention 내부 동작
- [Reverse Engineering 계획 & Prometheus 클론](reverse-engineering-plan-kr.md) - 바이너리 분석, 네트워크 캡처, 오픈소스 클론 구현 로드맵

## 표기 규칙

- 도구명: `mcp__glean_default__search` (실제 런타임에서 관찰된 이름 사용)
- 경로/버전: 본문이 아닌 관측 시점 블록에 기록 (예: `> **관측 시점**: 2026-03-26, OMC 4.4.4`)
- 예시 JSON: 실제 API 캡처가 아닌 **구조 설명용 의사 코드**. 코드블록에 `jsonc` 또는 `text` 사용

## 관측 환경

> **관측 시점**: 2026-03-26. 아래 값은 이 시점 기준이며 버전 업데이트에 따라 달라질 수 있다.

| 항목             | 값                            |
| -------------- | ---------------------------- |
| Claude Code 모델 | claude-opus-4-6 (1M context) |
| OMC 버전         | 4.4.4                        |
| OS             | macOS Darwin 24.6.0          |


