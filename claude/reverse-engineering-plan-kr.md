# Claude Code Reverse Engineering 계획

> **목표**: Claude Code의 내부 동작을 완전히 이해하고, 오픈소스 클론 **Prometheus**를 구현한다.
>
> **클론 이름**: **Prometheus** (프로메테우스) — AI 코딩 능력을 모든 개발자에게 가져다주는 도구. 프로메테우스가 인류에게 불을 가져다준 것처럼.
>
> **관측 시점**: 2026-03-27, Claude Code 2.1.84, OMC 4.4.4

---

## Phase 1: 바이너리 분석 (1주)

### 1.1 소스 추출

Claude Code는 **Bun 런타임**으로 컴파일된 단일 바이너리이다. TypeScript 소스가 minified JS로 바이너리에 내장되어 있어 `strings`로 추출 가능하다.

```bash
# 바이너리 위치
~/.local/share/claude/versions/2.1.84  # 194MB Mach-O arm64

# 심볼릭 링크
~/.local/bin/claude -> ~/.local/share/claude/versions/2.1.84

# JS 소스 추출
strings ~/.local/share/claude/versions/2.1.84 > claude-raw-strings.txt

# Anthropic 코드 시작점 찾기
grep -n "Claude Code is a Beta product" claude-raw-strings.txt
```

**이미 확인된 사항:**
- 빌드 경로: `build/release/tmp_modules/bun/`
- Bun FFI, Bun SQLite, Bun Inspector 등 사용
- `(c) Anthropic PBC. All rights reserved.` 저작권 표시

### 1.2 소스 beautify 및 모듈 분리

```bash
# minified JS 추출 후 beautify
node -e "
  const fs = require('fs');
  const raw = fs.readFileSync('claude-raw-strings.txt', 'utf8');
  // Anthropic 메인 번들 찾기
  const start = raw.indexOf('Claude Code is a Beta product');
  const bundle = raw.slice(start);
  fs.writeFileSync('claude-bundle.js', bundle);
"

# prettier로 포맷팅
npx prettier --write claude-bundle.js

# 또는 js-beautify
npx js-beautify claude-bundle.js -o claude-beautified.js
```

### 1.3 모듈 구조 매핑

추출된 소스에서 주요 모듈 경계를 식별한다:

| 모듈 (추정) | 식별 키워드 | 역할 |
|---|---|---|
| Message Loop | `setMessages`, `messagesRef`, `readFileState` | 메시지 루프 핵심 |
| Tool System | `tool_use`, `tool_result`, `registerTool` | 도구 등록/실행 |
| Hook System | `PreToolUse`, `PostToolUse`, `hookEvent` | 훅 이벤트 처리 |
| ToolSearch | `DeferredTool`, `ToolSearch`, deferred | lazy tool loading |
| Speculation | `[Speculation]`, `boundary`, `pipelined` | 추측 실행 |
| Compaction | `context_management`, `compact` | 컨텍스트 압축 |
| UI (Ink) | `createElement`, `flexDirection`, `uT` | 터미널 UI |
| MCP Client | `mcp_tool_use`, `server_tool_use` | MCP 통신 |
| Plugin System | `pluginHook`, `pluginName`, `pluginRoot` | 플러그인 관리 |
| Skill System | `command-name`, `command-message`, `SKILL.md` | 스킬 로딩 |
| Permission | `permissionDecision`, `allow`, `deny`, `ask` | 권한 관리 |
| Privacy | `grove_enabled`, `tengu_grove_policy` | 프라이버시 설정 |

---

## Phase 2: 네트워크 트래픽 분석 (1주)

### 2.1 API 통신 캡처

Claude Code가 Anthropic Messages API에 보내는 실제 요청을 캡처한다.

**방법 A: mitmproxy**
```bash
# mitmproxy 설치
brew install mitmproxy

# 프록시 시작
mitmweb --mode regular --listen-port 8080

# Claude Code 실행 시 프록시 경유
HTTPS_PROXY=http://localhost:8080 claude
```

**방법 B: 환경변수 활용**
```bash
# Claude Code가 ANTHROPIC_BASE_URL을 지원하면 프록시로 리다이렉트
ANTHROPIC_BASE_URL=http://localhost:8080/proxy claude
```

### 2.2 캡처 대상

| 항목 | 확인 포인트 |
|---|---|
| System Prompt | 전체 내용, tool 정의 포함 여부 |
| Tool Definitions | `<functions>` 블록 구조 |
| Message Format | `user`/`assistant` 턴 구조 |
| Tool Call/Result | `tool_use` → `tool_result` 왕복 |
| Streaming | SSE 스트리밍 구조 |
| Cache Control | `cache_control` 필드, `ephemeral` 블록 |
| Token Usage | `input_tokens`, `output_tokens`, `cache_read` |

### 2.3 System Prompt 완전 복원

캡처된 첫 번째 API 요청에서 system prompt를 추출한다. 여기에 다음이 포함될 것으로 예상:

- 모든 built-in tool의 JSON Schema 정의
- 행동 지침 (안전성, 코딩 가이드라인)
- CLAUDE.md 내용 주입
- 환경 정보 (OS, shell, cwd)

---

## Phase 3: 훅 시스템 분석 (3일)

### 3.1 전체 훅 이벤트 목록 (바이너리에서 확인)

```
PreToolUse, PostToolUse, PostToolUseFailure,
Notification, UserPromptSubmit, SessionStart, SessionEnd,
Stop, StopFailure, SubagentStart, SubagentStop,
PreCompact, PostCompact, PermissionRequest, Setup,
TeammateIdle, TaskCreated, TaskCompleted,
Elicitation, ElicitationResult, ConfigChange,
WorktreeCreate, WorktreeRemove,
InstructionsLoaded, CwdChanged, FileChanged
```

### 3.2 훅 데이터 흐름 로깅

```json
// settings.json에 로깅 훅 추가
{
  "hooks": {
    "PreToolUse": [{
      "type": "command",
      "command": "tee -a /tmp/claude-hook-pre.jsonl"
    }],
    "PostToolUse": [{
      "type": "command",
      "command": "tee -a /tmp/claude-hook-post.jsonl"
    }],
    "UserPromptSubmit": [{
      "type": "command",
      "command": "tee -a /tmp/claude-hook-submit.jsonl"
    }],
    "SessionStart": [{
      "type": "command",
      "command": "tee -a /tmp/claude-hook-session.jsonl"
    }]
  }
}
```

### 3.3 훅 입출력 스키마 역추적

각 훅 이벤트별 stdin/stdout JSON 구조를 문서화한다.

---

## Phase 4: MCP 프로토콜 분석 (3일)

### 4.1 MCP 통신 캡처

MCP 서버와의 통신은 JSON-RPC 2.0 over stdio/HTTP 이다.

```bash
# MCP 서버 프로세스 확인
ps aux | grep mcp

# stdio 기반 MCP 서버의 통신 캡처 (strace/dtruss)
dtruss -f -t read -t write -p <MCP_PID>
```

### 4.2 MCP Tool 등록 흐름

```
Claude Code 시작
  → MCP 서버 연결 (stdio/HTTP)
  → tools/list 호출 → 도구 목록 수신
  → 도구를 deferred tools로 등록
  → ToolSearch로 필요 시 스키마 로드
  → mcp_tool_use로 실제 호출
```

---

## Phase 5: Prometheus 설계 (1주)

### 5.1 기술 스택

| 컴포넌트 | Claude Code (원본) | Prometheus (클론) |
|---|---|---|
| 런타임 | Bun | **Node.js 또는 Bun** |
| 언어 | TypeScript | **TypeScript** |
| UI | React Ink | **React Ink** |
| LLM API | Anthropic Messages API | **Anthropic + OpenAI + 기타** |
| 빌드 | Bun compile (single binary) | **esbuild + pkg 또는 Bun compile** |
| MCP | JSON-RPC 2.0 client | **@modelcontextprotocol/sdk** |

### 5.2 핵심 아키텍처

```
prometheus/
├── src/
│   ├── core/
│   │   ├── loop.ts              # Agent Loop (메시지 루프)
│   │   ├── messages.ts          # 메시지 관리
│   │   └── context.ts           # 컨텍스트 빌더 (system prompt)
│   ├── tools/
│   │   ├── registry.ts          # Tool 등록/관리
│   │   ├── builtin/             # 내장 도구 (Bash, Read, Write, Edit, Glob, Grep)
│   │   ├── deferred.ts          # Deferred tool (lazy loading)
│   │   └── mcp/                 # MCP client
│   ├── hooks/
│   │   ├── engine.ts            # 훅 엔진
│   │   ├── events.ts            # 이벤트 정의
│   │   └── runner.ts            # 훅 실행기 (command, prompt, http)
│   ├── plugins/
│   │   ├── loader.ts            # 플러그인 로더
│   │   ├── skills.ts            # 스킬 시스템
│   │   └── marketplace.ts       # 마켓플레이스 연동
│   ├── permissions/
│   │   ├── manager.ts           # 권한 관리
│   │   └── policies.ts          # 허용/차단 정책
│   ├── ui/
│   │   ├── app.tsx              # 메인 UI (React Ink)
│   │   ├── components/          # UI 컴포넌트
│   │   └── theme.ts             # 테마
│   ├── api/
│   │   ├── anthropic.ts         # Anthropic API 클라이언트
│   │   ├── openai.ts            # OpenAI API 클라이언트 (확장)
│   │   └── streaming.ts         # SSE 스트리밍
│   ├── features/
│   │   ├── speculation.ts       # 추측 실행
│   │   ├── compaction.ts        # 컨텍스트 압축
│   │   ├── agents.ts            # Sub-agent 시스템
│   │   └── teams.ts             # 팀 협업
│   └── config/
│       ├── settings.ts          # 설정 파일 관리
│       └── claude-md.ts         # CLAUDE.md 로딩
├── package.json
├── tsconfig.json
└── README.md
```

### 5.3 MVP 구현 순서

```
Phase 5a: 기본 루프 (1주)
  ├── Anthropic Messages API 통신
  ├── System Prompt 구성
  ├── User → Assistant → Tool → Assistant 루프
  └── 터미널 입출력

Phase 5b: 내장 도구 (1주)
  ├── Bash (셸 명령 실행)
  ├── Read (파일 읽기)
  ├── Write (파일 쓰기)
  ├── Edit (파일 수정)
  ├── Glob (파일 검색)
  └── Grep (내용 검색)

Phase 5c: MCP 지원 (1주)
  ├── MCP 클라이언트 구현
  ├── stdio/HTTP transport
  ├── Tool 자동 등록
  └── ToolSearch (deferred loading)

Phase 5d: 훅 시스템 (3일)
  ├── 이벤트 엔진
  ├── command/prompt/http 타입
  └── settings.json 로딩

Phase 5e: UI 개선 (3일)
  ├── React Ink 기반 UI
  ├── 진행 상태 표시
  └── 권한 확인 대화상자

Phase 5f: 플러그인/스킬 (1주)
  ├── 플러그인 로더
  ├── 스킬 시스템 (SKILL.md)
  └── 마켓플레이스 연동
```

---

## Phase 6: 검증 및 비교 (지속)

### 6.1 기능 대조표

| 기능 | Claude Code | Prometheus |
|---|---|---|
| Agent Loop | O | - |
| Built-in Tools (6종) | O | - |
| MCP Client | O | - |
| Hook System (25+ events) | O | - |
| ToolSearch (deferred) | O | - |
| Speculation | O | - |
| Compaction | O | - |
| Plugin/Skill | O | - |
| Permission System | O | - |
| Sub-agent/Team | O | - |
| Multi-provider (OpenAI 등) | X | - |

### 6.2 동일 시나리오 비교 테스트

1. "hello-slack" 스킬 실행
2. 파일 편집 + 테스트 실행
3. MCP 서버 연동 (Slack, GitHub)
4. 훅 기반 자동 포맷팅

---

## 리스크 및 고려사항

### 법적 고려
- Claude Code 라이선스: `Use is subject to the Legal Agreements outlined here: https://code.claude.com/docs/en/legal-and-compliance`
- 바이너리 역분석 결과를 **직접 복사하지 않고**, 동작 원리를 이해하여 **독립 구현**(clean room implementation)
- Prometheus는 완전히 새로운 코드로 작성

### 기술적 리스크
- System Prompt 전체 내용을 네트워크 캡처 없이는 정확히 알 수 없음
- Speculation 시스템의 정확한 알고리즘은 minified 코드 분석 필요
- Anthropic API의 비공개 파라미터가 있을 수 있음

---

## 참고 자료

### 이미 작성된 분석 문서
- [claude-code-api-kr.md](claude-code-api-kr.md) — Messages API 통신 구조
- [claude-code-internals-kr.md](claude-code-internals-kr.md) — 바이너리 내부 구조
- [claude-code-config-kr.md](claude-code-config-kr.md) — 설정 파일 구조
- [claude-code-caching-kr.md](claude-code-caching-kr.md) — Prompt Caching
- [omc-architecture-kr.md](omc-architecture-kr.md) — OMC 동작 원리
- [omc-flow-diagram-kr.md](omc-flow-diagram-kr.md) — 실행 흐름 다이어그램

### 외부 참고
- [Anthropic Messages API Docs](https://docs.anthropic.com/en/api/messages)
- [Model Context Protocol Spec](https://modelcontextprotocol.io)
- [React Ink](https://github.com/vadimdemedes/ink)
- [Bun](https://bun.sh)
