# OMC 실행 흐름 다이어그램

> **근거**: `runtime-observed` (실제 "hello-slack" 실행 중 관찰한 훅 주입, API 호출 순서, 도구 결과 흐름), `source-inspected` (hooks.json, skill-injector.mjs 소스 코드), `inferred` (API 페이로드 내부 구조는 외부 관찰 기반 재구성)
>
> **관측 시점**: 2026-03-26, OMC 4.4.4. 버전에 따라 훅 구성, 도구명, 실행 순서가 달라질 수 있다.
>
> 아래 다이어그램은 실제 네트워크 캡처가 아니라, 관찰된 행동을 기반으로 **재구성한 시퀀스**이다.

"hello-slack" 입력 시 실제 일어나는 전체 흐름을 다이어그램으로 정리한다.

## 1. 전체 시퀀스 다이어그램

```
 User          Claude Code         Hooks (Node.js)      Opus API         MCP Servers
  |                |                     |                  |                  |
  | "hello-slack"  |                     |                  |                  |
  |--------------->|                     |                  |                  |
  |                |                     |                  |                  |
  |                |  UserPromptSubmit   |                  |                  |
  |                |-------------------->|                  |                  |
  |                |                     |                  |                  |
  |                |  #1 keyword-detector|                  |                  |
  |                |  stdin: {"prompt":  |                  |                  |
  |                |   "hello-slack"}    |                  |                  |
  |                |                     |                  |                  |
  |                |  stdout: (empty)    |                  |                  |
  |                |<--------------------|                  |                  |
  |                |                     |                  |                  |
  |                |  #2 skill-injector  |                  |                  |
  |                |  stdin: {"prompt":  |                  |                  |
  |                |   "hello-slack"}    |                  |                  |
  |                |                     |                  |                  |
  |                |  scan omc-learned/  |                  |                  |
  |                |  trigger match!     |                  |                  |
  |                |                     |                  |                  |
  |                |  stdout:            |                  |                  |
  |                |  <mnemosyne>        |                  |                  |
  |                |  # hello-slack...   |                  |                  |
  |                |  </mnemosyne>       |                  |                  |
  |                |<--------------------|                  |                  |
  |                |                     |                  |                  |
  |                |                     |                  |                  |
  |                |  ======= API Request #1 =============>|                  |
  |                |                     |                  |                  |
  |                |  system: CLAUDE.md + MEMORY.md         |                  |
  |                |  user: <system-reminder>               |                  |
  |                |          <mnemosyne>스킬내용            |                  |
  |                |        </mnemosyne>                    |                  |
  |                |        </system-reminder>              |                  |
  |                |        hello-slack                     |                  |
  |                |  tools: [Read, Write, Bash,            |                  |
  |                |    mcp__glean_default__search, ...]    |                  |
  |                |                     |                  |                  |
  |                |  <====== API Response #1 ==============|                  |
  |                |                     |                  |                  |
  |                |  thinking: "스킬 지침에 따라 검색..."   |                  |
  |                |  tool_use:                             |                  |
  |                |    mcp__glean_default__search x N      |                  |
  |                |  stop_reason: tool_use                 |                  |
  |                |                     |                  |                  |
  |  "채널을      |                     |                  |                  |
  |   검색합니다"  |                     |                  |                  |
  |<---------------|                     |                  |                  |
  |                |                     |                  |                  |
  |                |  PreToolUse 훅 xN   |                  |                  |
  |                |-------------------->|                  |                  |
  |                |<--------------------|                  |                  |
  |                |                     |                  |                  |
  |                |  실제 도구 실행 --------------------------->|
  |                |  mcp__glean_default__search xN (병렬)  |  Glean API 호출  |
  |                |  <-----------------------------------------|
  |                |                     |                  |  검색 결과 반환  |
  |                |                     |                  |                  |
  |                |  PostToolUse 훅 xN  |                  |                  |
  |                |-------------------->|                  |                  |
  |                |<--------------------|                  |                  |
  |                |                     |                  |                  |
  |                |                     |                  |                  |
  |                |  ======= API Request #2 =============>|                  |
  |                |                     |                  |                  |
  |                |  messages: [이전 대화 전체]             |                  |
  |                |  + tool_result xN (검색 결과)          |                  |
  |                |  + <system-reminder> (훅 컨텍스트)     |                  |
  |                |                     |                  |                  |
  |                |  <====== API Response #2 ==============|                  |
  |                |                     |                  |                  |
  |                |  thinking: "결과 필터링 후 파일 작성"   |                  |
  |                |  tool_use: Write (마크다운 파일)        |                  |
  |                |  stop_reason: tool_use                 |                  |
  |                |                     |                  |                  |
  |                |  파일 저장 (로컬)   |                  |                  |
  |                |                     |                  |                  |
  |                |                     |                  |                  |
  |                |  ======= API Request #3 =============>|                  |
  |                |                     |                  |                  |
  |                |  messages: [이전 대화 전체]             |                  |
  |                |  + tool_result: Write 성공             |                  |
  |                |                     |                  |                  |
  |                |  <====== API Response #3 ==============|                  |
  |                |                     |                  |                  |
  |                |  tool_use: Bash("cursor ...")          |                  |
  |                |  stop_reason: tool_use                 |                  |
  |                |                     |                  |                  |
  |                |  cursor 실행 (로컬) |                  |                  |
  |                |                     |                  |                  |
  |                |                     |                  |                  |
  |                |  ======= API Request #4 =============>|                  |
  |                |                     |                  |                  |
  |                |  <====== API Response #4 ==============|                  |
  |                |                     |                  |                  |
  |                |  text: "완료"                          |                  |
  |                |  stop_reason: end_turn                 |                  |
  |                |                     |                  |                  |
  |  "hello-slack  |                     |                  |                  |
  |   실행 완료"   |                     |                  |                  |
  |<---------------|                     |                  |                  |
```

## 2. API Request 누적 구조

매 Request마다 전체 대화 히스토리가 포함된다:

```
Request #1                Request #2                Request #3
┌──────────────┐          ┌──────────────┐          ┌──────────────┐
│ system:      │          │ system:      │          │ system:      │
│  CLAUDE.md   │          │  (동일)      │          │  (동일)      │
│  MEMORY.md   │          │              │          │              │
├──────────────┤          ├──────────────┤          ├──────────────┤
│ messages:    │          │ messages:    │          │ messages:    │
│              │          │              │          │              │
│ [user]       │          │ [user]       │          │ [user]       │
│ <mnemosyne>  │          │ <mnemosyne>  │          │ <mnemosyne>  │
│ 스킬내용     │          │ 스킬내용     │          │ 스킬내용     │
│ </mnemosyne> │          │ </mnemosyne> │          │ </mnemosyne> │
│ hello-slack  │          │ hello-slack  │          │ hello-slack  │
│              │          │              │          │              │
│              │          │ [assistant]  │          │ [assistant]  │
│              │          │ thinking     │          │ thinking     │
│              │          │ text         │          │ text         │
│              │          │ tool_use xN  │          │ tool_use xN  │
│              │          │              │          │              │
│              │          │ [user]       │          │ [user]       │
│              │          │ tool_result  │          │ tool_result  │
│              │          │  xN + 훅주입 │          │  xN + 훅주입 │
│              │          │              │          │              │
│              │          │              │          │ [assistant]  │
│              │          │              │          │ thinking     │
│              │          │              │          │ tool_use:    │
│              │          │              │          │  Write       │
│              │          │              │          │              │
│              │          │              │          │ [user]       │
│              │          │              │          │ tool_result: │
│              │          │              │          │  Write 성공  │
├──────────────┤          ├──────────────┤          ├──────────────┤
│ tools: [     │          │ tools: [     │          │ tools: [     │
│  (동일)      │          │  (동일)      │          │  (동일)      │
│ ]            │          │ ]            │          │ ]            │
└──────────────┘          └──────────────┘          └──────────────┘

 (system+user+tools)     (+ 도구 결과 N개)       (+ Write 결과)
                         크기가 크게 증가         점진적 증가
```

> 구체적인 KB 수치는 스킬 내용, 검색 결과 크기, 채널 수에 따라 크게 달라지므로 생략한다.

## 3. 훅 주입 포인트

```
사용자 메시지 도착
        │
        ▼
  ┌─────────────┐     stdout → <system-reminder> 로 감싸져서
  │ UserPrompt  │──────────── user message 앞에 붙음
  │ Submit 훅   │
  └─────────────┘
        │
        ▼
    API 호출 #1
        │
        ▼
    Opus 응답: tool_use
        │
        ▼
  ┌─────────────┐     stdout → <system-reminder> 로 감싸져서
  │ PreToolUse  │──────────── tool_result 앞에 붙음
  │ 훅          │
  └─────────────┘
        │
        ▼
    도구 실제 실행
        │
        ▼
  ┌─────────────┐     stdout → <system-reminder> 로 감싸져서
  │ PostToolUse │──────────── tool_result 뒤에 붙음
  │ 훅          │
  └─────────────┘
        │
        ▼
    API 호출 #2 (tool_result + 훅 컨텍스트 포함)
        │
        ▼
    Opus 응답: tool_use 또는 end_turn
        │
        ▼ (end_turn이면)
  ┌─────────────┐     stdout → Claude가 멈추지 못하게 강제
  │ Stop 훅     │──────────── (autopilot/ralph 모드일 때만)
  └─────────────┘
```

## 4. 컨텍스트 크기 누적

```
Request #1:  ████░░░░░░░░░░░░░░░░░░░░░░░░░░  (system + 스킬 + tools)
Request #2:  ████████████████░░░░░░░░░░░░░░░  (+ 검색 결과 N개 — 가장 큰 증가)
Request #3:  █████████████████░░░░░░░░░░░░░░  (+ Write 결과)
Request #4:  █████████████████░░░░░░░░░░░░░░  (+ Bash 결과)
...
Request #N:  ████████████████████████████████  → PreCompact 발동 → 압축
```

> 컨텍스트 한계에 가까워지면 Claude Code가 PreCompact 훅을 실행한 뒤 이전 메시지를 자동 압축한다.

## 5. 대화 턴 패턴

```
user:      [text]                          ← 사용자 입력
assistant: [thinking, text, tool_use]      ← 생각 + 응답 + 도구 호출
user:      [tool_result]                   ← 도구 결과 (Claude Code가 자동 생성)
assistant: [thinking, text, tool_use]      ← 다음 도구 호출
user:      [tool_result]                   ← 도구 결과
assistant: [thinking, text]                ← 최종 응답 (stop_reason: end_turn)
```

## 6. 전체 아키텍처

```
┌────────────────────────────────────────────────┐
│                 Claude Code                     │
│                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Skill    │  │ MCP Tool │  │ Hook     │     │
│  │ Tool     │  │ Calls    │  │ Events   │     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘     │
└───────┼──────────────┼─────────────┼───────────┘
        │              │             │
        ▼              ▼             ▼
┌──────────────┐ ┌───────────┐ ┌────────────────┐
│ skills/*.md  │ │ bridge/   │ │ scripts/*.mjs  │
│ (마크다운    │ │ mcp-      │ │ (Node.js 훅   │
│  지침서)     │ │ server.cjs│ │  스크립트)     │
└──────────────┘ └─────┬─────┘ └────────────────┘
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
     ┌──────────┐ ┌────────┐ ┌────────┐
     │ LSP      │ │ AST    │ │ Python │
     │ Servers  │ │ grep   │ │ REPL   │
     │ (언어별) │ │        │ │ (Unix  │
     └──────────┘ └────────┘ │ Socket)│
                              └────────┘
```
