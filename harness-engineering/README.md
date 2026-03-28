# Harness Engineering - Claude Code & Codex 고급 활용 가이드

> Claude Code와 Codex를 활용한 **하네스 엔지니어링 고급 기술**을 구체적인 예시 중심으로 정리한 문서 모음입니다.

## 목차

### 실무 개발 작업 위임 (고급)

| # | 문서 | 주제 |
|---|------|------|
| 1 | [Codex 최신 실습 가이드](codex-latest-practical-guide-kr.md) | 최신 Codex CLI, 승인/샌드박스, exec/review/resume, MCP 실습 |
| 2 | [MCP로 외부 도구 연결하기](mcp-external-tools-kr.md) | GitHub/DB/브라우저 MCP 연결, 커스텀 MCP 서버 만들기 |
| 3 | [테스트와 검증 루프 만들기](tdd-verification-loop-kr.md) | TDD 워크플로우, 검증 루프 자동화, 디버깅 질문 패턴 |

### 반복 작업 자동화 워크플로우

| # | 문서 | 주제 |
|---|------|------|
| 4 | [스킬로 반복 작업 줄이기](skills-reusable-patterns-kr.md) | Claude Code/Codex 스킬 작성, 크로스 도구 재사용 |
| 5 | [훅과 정책으로 워크플로우 고정하기](hooks-policy-workflow-kr.md) | PreToolUse Hook, 자동 포맷/테스트, 팀 규칙 자동화 |
| 6 | [문서화 자동화 & 팀 운영 규칙](doc-automation-team-rules-kr.md) | 코드-문서 동기화, AI+사람 Git 운영 규칙 |

### Agent Harness: 멀티 에이전트

| # | 문서 | 주제 |
|---|------|------|
| 7 | [서브에이전트와 역할 분리](subagent-role-separation-kr.md) | 탐색/구현/검증 역할 분리, 컨텍스트 유지 |
| 8 | [멀티 에이전트 패턴과 선택 기준](multi-agent-patterns-kr.md) | 5가지 패턴 비교, 하이브리드 전략 |
| 9 | [나만의 Agent Harness 설계](agent-harness-design-kr.md) | SDD, Spec.md, 하네스 자기개선 루프 |

### Agent Harness 프로젝트 실습

| # | 문서 | 주제 |
|---|------|------|
| 10 | [프로젝트: 0 to 1](project-zero-to-one-kr.md) | PRD→MVP, SPEC 분해, 병렬 처리 자동화 |
| 11 | [프로젝트: 1 to 10](project-one-to-ten-kr.md) | 코드 리뷰, 리팩터링 루프, AI 위임 가이드 |

### Agent Harness 사례 연구

| # | 문서 | 주제 |
|---|------|------|
| 12 | [oh-my-claudecode 구조 분석](omc-teams-harness-kr.md) | Team/Autopilot/Ultrawork 비교, 패턴 이식 |
| 13 | [oh-my-openagent 하네스 설계](openagent-harness-kr.md) | 계층형 AGENTS.md, Sisyphus/Prometheus 역할 분석 |

## 대상 독자

- Claude Code와 Codex 기본 사용법을 이미 아는 개발자
- AI 코딩 에이전트를 실무 프로젝트에 본격 적용하려는 엔지니어
- 멀티 에이전트 하네스를 설계하고 싶은 기술 리더

## 작성 원칙

- 각 문서는 **실행 가능한 코드/설정/프롬프트 예시** 중심으로 작성되었습니다
- 코드와 CLI 명령어는 영문 그대로, 설명은 한글로 작성합니다
