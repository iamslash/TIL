# Claude Code 바이너리 내부 구조 분석

> **근거 표기**: `binary-inspected` (바이너리 strings 추출로 확인), `inferred` (구조로부터 추정)
>
> **관측 시점**: 2026-03-26, Claude Code 2.1.84, Mach-O arm64 바이너리 (194MB)

## 개요

Claude Code는 **Bun 런타임으로 컴파일된 단일 네이티브 바이너리**이다. npm 패키지가 아니라 자체 업데이트되는 독립 실행 파일로 배포된다.

```text
~/.local/share/claude/versions/2.1.84   # Mach-O 64-bit executable arm64 (194MB)
~/.local/bin/claude -> 위 경로의 심링크
```

내부적으로는 TypeScript/JavaScript 소스가 Bun으로 번들·컴파일되어 있다. `strings` 명령으로 minified된 JS 코드 파편을 추출할 수 있으며, 이를 통해 아키텍처를 역추적할 수 있다.

## 빌드 환경 [binary-inspected]

빌드 경로에서 확인된 정보:

```text
/Users/runner/work/bun-internal/bun-internal/   # GitHub Actions runner
BUILD_TIME: "2026-03-25T23:49:18Z"
VERSION: "2.1.84"
PACKAGE_URL: "@anthropic-ai/claude-code"
```

- **런타임**: Bun (Node.js가 아님). `Bun.FFI`, `Bun.file()`, `Bun.env` API 참조 다수 확인
- **빌드 시스템**: GitHub Actions CI/CD
- **컴파일 방식**: Bun의 single-executable 컴파일 (`bun build --compile`과 유사)

## 내부 코드명 [binary-inspected]

프로젝트 내부 코드명은 **Tengu**(텐구)이다. 텔레메트리 이벤트 이름이 모두 `tengu_` 접두어를 사용한다:

```text
tengu_agent_tool_terminated
tengu_compact_*
tengu_memdir_disabled
tengu_mcp_instructions_pool_change
tengu_team_memdir_disabled
tengu_worktree_detection
tengu_turtle_carbon
tengu_tst_kx7
tengu_powershell_tool_command_executed
tengu_hawthorn_window
```

텔레메트리 수집에는 **OpenTelemetry** 프로토콜 (gRPC)이 사용된다.

## 아키텍처 총괄

```
┌──────────────────────────────────────────────────────────────┐
│                    Claude Code Binary (Bun)                  │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │ TUI (Ink)   │  │ System Prompt│  │  Tool Registry      │ │
│  │ Terminal UI │  │ Builder      │  │  (Built-in + MCP)   │ │
│  └──────┬──────┘  └──────┬───────┘  └──────────┬──────────┘ │
│         │                │                      │            │
│  ┌──────▼──────────────────────────────────────▼──────────┐ │
│  │                    Agent Loop                          │ │
│  │  user msg → API call → tool_use → execute → tool_result│ │
│  │       ↑                                        │       │ │
│  │       └────────────────────────────────────────┘       │ │
│  └──────────────────────┬─────────────────────────────────┘ │
│                         │                                    │
│  ┌──────────────────────▼─────────────────────────────────┐ │
│  │              Context Manager                           │ │
│  │  Prompt Caching │ Compaction │ Speculation │ ToolSearch│ │
│  └──────────────────────┬─────────────────────────────────┘ │
│                         │                                    │
│  ┌──────────────────────▼─────────────────────────────────┐ │
│  │         Anthropic Messages API (streaming)             │ │
│  │         + tool_reference (server-side ToolSearch)       │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────┐  ┌─────────────┐  ┌───────────────────────┐ │
│  │ Permission │  │ Hook System │  │ MCP Client            │ │
│  │ System     │  │ (pre/post)  │  │ (stdio/SSE transport) │ │
│  └────────────┘  └─────────────┘  └───────────────────────┘ │
│                                                              │
│  ┌────────────┐  ┌─────────────┐  ┌───────────────────────┐ │
│  │ Memory     │  │ Session     │  │ Team/Agent            │ │
│  │ System     │  │ Persistence │  │ Orchestration         │ │
│  └────────────┘  └─────────────┘  └───────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 시스템 프롬프트 구조 [binary-inspected]

시스템 프롬프트는 **동적으로 조립**된다. 고정 문자열이 아니라 여러 섹션을 런타임에 결합한다.

### 프롬프트 조립 함수

`dY6(model, context)` 함수가 시스템 프롬프트를 조립한다. 병렬로 여러 데이터를 로드한 뒤 결합한다:

```text
[모델 식별] "You are powered by the model named {modelName}. The exact model ID is {modelId}."
[지식 컷오프] "Assistant knowledge cutoff is {date}."
[작업 디렉토리] "Primary working directory: {cwd}"
[git 상태] "Is a git repository: ..."
[worktree] "This is a git worktree — an isolated copy of the repository."
```

### 에이전트 유형별 시스템 프롬프트 [binary-inspected]

바이너리에서 확인된 에이전트 identity 문자열들:

| 에이전트 유형 | 시스템 프롬프트 시작 |
|---|---|
| **메인 에이전트** | `"You are Claude Code, Anthropic's official CLI for Claude."` |
| **SDK 에이전트** | `"You are Claude Code, Anthropic's official CLI for Claude, running within the Claude Agent SDK."` |
| **서브 에이전트** | `"You are an agent for Claude Code, Anthropic's official CLI for Claude. Given the user's message, you should use the tools available to complete the task. Do what has been asked; nothing more, nothing less."` |
| **탐색 에이전트** | `"You are a file search specialist for Claude Code, Anthropic's official CLI for Claude. You excel at thoroughly navigating and exploring codebases."` |
| **심플 모드** | `"You are Claude Code, Anthropic's official CLI for Claude."` (최소한) |

### 동적 시스템 프롬프트 섹션 [binary-inspected]

`__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__` 마커로 정적/동적 섹션을 구분한다. 동적 섹션은 매 턴마다 갱신되어 프롬프트 캐시를 무효화하지 않도록 꼬리에 배치된다.

조립 순서 (추정):

```text
1. 에이전트 identity + 모델 정보
2. 환경 정보 (OS, cwd, git status)
3. 도구 정의 (built-in tools)
4. 코딩 지침 (규칙들)
5. __SYSTEM_PROMPT_DYNAMIC_BOUNDARY__
6. CLAUDE.md / 사용자 규칙
7. MCP 서버 instructions
8. 메모리 시스템 정보
9. 스크래치패드 경로
```

### 주요 코딩 지침 [binary-inspected]

시스템 프롬프트에 포함되는 핵심 지침들:

```text
"Don't add features, refactor code, or make 'improvements' beyond what was asked."
"Don't add error handling, fallbacks, or validation for scenarios that can't happen."
"NEVER create documentation files (*.md) or README files unless explicitly requested."
"NEVER create files unless they're absolutely necessary for achieving your goal."
"Go straight to the point. Try the simplest approach first without going in circles."
"When you encounter an obstacle, do not use destructive actions as a shortcut."
```

서브 에이전트(fork)에 대한 특수 지침:

```text
"Your system prompt says 'default to forking.' IGNORE IT — that's for the parent."
"You ARE the fork. Do NOT spawn sub-agents; execute directly."
"Do NOT converse, ask questions, or suggest next steps"
"Do NOT emit text between tool calls. Use tools silently, then report once at the end."
```

---

## 도구(Tool) 시스템 [binary-inspected]

### 도구 분류

바이너리에서 추출한 모든 Tool 클래스명 (`*Tool` 패턴):

**핵심 내장 도구 (Built-in)**

| 클래스명 | 설명 |
|---|---|
| `BashTool` | 셸 명령 실행 |
| `PowerShellTool` | Windows PowerShell 실행 |
| `FileReadTool` (= `ReadTool`) | 파일 읽기 (이미지 포함, multimodal) |
| `FileWriteTool` (= `WriteTool`) | 파일 쓰기 |
| `FileEditTool` (= `EditTool`, `CodeEditTool`) | 파일 편집 (old_string → new_string) |
| `GlobTool` | 파일 패턴 검색 |
| `GrepTool` | 내용 검색 (regex) |
| `NotebookEditTool` | Jupyter 노트북 편집 |
| `WebSearchTool` | 웹 검색 |
| `WebFetchTool` | URL 내용 가져오기 |
| `ToolSearchTool` | 지연 로드 도구 검색 |

**에이전트/팀 도구**

| 클래스명 | 설명 |
|---|---|
| `AgentTool` | 서브 에이전트(fork) 생성 |
| `TaskTool` | 태스크 생성/관리 |
| `SendMessageTool` | 팀 메시지 전송 |
| `TeamCreateTool` | 팀 생성 |
| `TeamDeleteTool` | 팀 삭제 |

**MCP 관련 도구**

| 클래스명 | 설명 |
|---|---|
| `McpTool` | MCP 서버 도구 래퍼 |
| `ListMcpResourcesTool` | MCP 리소스 목록 |
| `ReadMcpResourceTool` | MCP 리소스 읽기 |

**메모리/스케줄링**

| 클래스명 | 설명 |
|---|---|
| `MemoryTool` | 메모리 읽기/쓰기 |
| `FileSystemMemoryTool` | 파일 기반 메모리 |
| `CronCreateTool` | 예약 작업 생성 |
| `CronDeleteTool` | 예약 작업 삭제 |
| `CronListTool` | 예약 작업 목록 |

**특수 도구**

| 클래스명 | 설명 |
|---|---|
| `ChromeTool` | Chrome 브라우저 제어 |
| `SkillTool` | 스킬 실행 |
| `DeferredTool` | 지연 로드 대기 도구 |
| `DiscoveredTool` | ToolSearch로 발견된 도구 |
| `CompactDiscoveredTool` | 컴팩션 후 발견된 도구 |

### 도구 변수명 매핑 [binary-inspected]

minified 코드에서 도구 이름은 상수 변수로 관리된다:

```text
kq = "Bash"           (BashTool)
HR = "Glob"           (GlobTool)
G4 = "Grep"           (GrepTool)
pq = "Read"           (FileReadTool)
k5 = "WebFetch"       (WebFetchTool)
eG = "WebSearch"      (WebSearchTool)
W6 = "Write"          (FileWriteTool)
eA = "Edit"           (FileEditTool)
u2 = "NotebookEdit"   (NotebookEditTool)
z6 = "Task"           (AgentTool/TaskTool)
Yz = "ToolSearch"     (ToolSearchTool)
aN = (Plan approval tool)
s2, Iu, BKH, M0, kTH, XTH, VL = (team/task tools)
```

### 도구 그룹 [binary-inspected]

기본 에이전트 도구 세트:

```javascript
HM7 = [...il, HR, G4, pq, k5, eG]  // 탐색 도구: Glob, Grep, Read, WebFetch, WebSearch
_M7 = [W6, eA, u2]                   // 편집 도구: Write, Edit, NotebookEdit
```

탐색(explore) 에이전트는 편집 도구가 비활성화된다:

```text
disallowedTools: [z6, aN, W6, eA, u2]  // Task, Plan, Write, Edit, NotebookEdit 사용 불가
```

---

## ToolSearch 메커니즘 [binary-inspected]

### 개념

MCP 서버 등에서 제공하는 많은 도구를 모두 시스템 프롬프트에 포함하면 토큰이 낭비된다. ToolSearch는 도구를 **지연 로드(deferred loading)** 하여 필요할 때만 스키마를 로드한다.

### 동작 원리

```text
1. 대화 시작 시: 도구 이름 목록만 <available-deferred-tools> 블록에 포함
2. 모델이 도구가 필요하다고 판단: ToolSearch("query") 호출
3. 클라이언트가 매칭: 도구 이름/설명 기반으로 deferred tool 검색
4. 결과 반환: 매칭된 도구의 전체 JSONSchema를 <functions> 블록으로 반환
5. 도구 사용 가능: 스키마가 프롬프트에 있으므로 정상 호출 가능
```

### ToolSearch 시스템 프롬프트 [binary-inspected]

ToolSearch 전용 설명 텍스트:

```text
"Until fetched, only the name is known — there is no parameter schema,
so the tool cannot be invoked. This tool takes a query, matches it against
the deferred tool list, and returns the matched tools' complete JSONSchema
definitions inside a <functions> block. Once a tool's schema appears in
that result, it is callable exactly like any tool defined at the top of
the prompt."
```

### 캐시 관리 [binary-inspected]

```javascript
// 도구 목록의 해시로 캐시 유효성 검증
function m7$(tools) { return tools.map(t => t.name).sort().join(","); }

// 캐시 무효화: deferred tools가 변경되면 캐시 클리어
function p7$(tools) {
  let hash = m7$(tools);
  if (cachedHash !== hash) {
    log("ToolSearchTool: cache invalidated - deferred tools changed");
    cache.clear();
    cachedHash = hash;
  }
}
```

### 스키마 미로드 시 에러 처리 [binary-inspected]

도구 스키마가 로드되지 않은 상태에서 호출하면:

```text
"This tool's schema was not sent to the API — it was not in the
discovered-tool set derived from message history. Without the schema
in your prompt, typed parameters (arrays, numbers, booleans) get emitted
as strings and the client-side parser rejects them. Load the tool first:
call ToolSearch with query 'select:{toolName}', then retry this call."
```

### API 연동: tool_reference [binary-inspected]

ToolSearch는 Anthropic API의 `tool_reference` 기능과 연동된다. 이 기능은 서버 사이드에서 도구 스키마를 관리하여 프롬프트 크기를 줄인다.

```text
"The tool search tool lets Claude dynamically discover tools from large
libraries without loading all definitions into the context window."
URL: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool
```

### MCP 서버 연결 해제 시 [binary-inspected]

```text
"The following deferred tools are no longer available (their MCP server
disconnected). Do not search for them — ToolSearch will return no match:"
```

---

## 컨텍스트 관리 [binary-inspected]

### Compaction (자동 요약)

컨텍스트가 가득 차면 오래된 메시지를 자동으로 요약한다.

```text
"Auto-compact is enabled. When the context window is nearly full, older
messages will be automatically summarized so you can continue working
seamlessly. There is no need to stop or rush — you have unlimited context."
```

요약 시 `<summary></summary>` 태그 안에 감싸서 저장:

```text
"Wrap your summary in <summary></summary> tags."
```

요약 과정:
1. 이전 메시지들을 요약 요청으로 API에 전송
2. 반환된 요약을 `isCompactSummary: true` 플래그와 함께 저장
3. 원본 메시지 제거, 요약 메시지로 대체
4. `PostCompact` 훅 실행
5. `tengu_compact_*` 텔레메트리 이벤트 기록

### Speculation (투기적 실행) [binary-inspected]

모델 응답을 기다리는 동안 **도구 실행 결과를 미리 예측**하여 지연시간을 줄이는 메커니즘:

```text
[Speculation] enabled=false              # 비활성화 시
[Speculation] Pipelined suggestion: "..." # 파이프라인 제안
[Speculation] Accept                     # 예측 수락
[Speculation] Complete:                  # 완료
[Speculation] Denied                     # 거부됨
[Speculation] Stopping at bash:          # Bash 실행 전 중단
[Speculation] Stopping at file edit:     # 파일 편집 전 중단
  "Speculation paused: bash boundary"
  "Speculation paused: file edit requires permission"
[Speculation] Stopping at denied tool:   # 거부된 도구에서 중단
```

투기적 실행은 **안전한 도구 (읽기 전용)**에만 적용되고, Bash 실행이나 파일 편집 같은 위험한 작업 앞에서 중단된다.

### 프롬프트 캐싱 [binary-inspected]

토큰 사용량을 추적하는 구조:

```javascript
{
  inputTokens: 0,
  outputTokens: 0,
  cacheReadInputTokens: 0,      // 캐시에서 읽은 토큰
  cacheCreationInputTokens: 0,  // 캐시에 새로 쓴 토큰
  webSearchRequests: 0,
  costUSD: 0,
  contextWindow: 0,
  maxOutputTokens: 0
}
```

캐시 관련 주요 동작:
- 시스템 프롬프트를 `cache_control: ephemeral`로 캐싱
- 동적 섹션(`__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__` 이후)은 캐시에서 제외
- 프롬프트 캐시는 5분 후 만료: `"the prompt cache expires after 5 minutes of inactivity — balance accordingly"`

---

## 메모리 시스템 [binary-inspected]

### 파일 기반 메모리

```text
MEMORY.md          # 메모리 인덱스 파일
최대 200줄 로드     # Hd = 200
최대 25000 바이트   # tlH = 25000
```

두 개의 디렉토리:
- **프라이빗 디렉토리**: 개인 세션 메모리
- **공유 팀 디렉토리**: 팀원 간 공유 메모리

### 자동 메모리 (`auto memory`)

환경 변수 `CLAUDE_CODE_DISABLE_AUTO_MEMORY`로 비활성화 가능. 설정 `autoMemoryEnabled`로도 제어.

---

## 에이전트 유형 [binary-inspected]

### 내장 에이전트 정의

```javascript
// general-purpose 에이전트
{
  agentType: "general-purpose",
  whenToUse: "General-purpose agent for researching complex questions,
    searching for code, and executing multi-step tasks.",
  tools: ["*"],        // 모든 도구 사용 가능
  source: "built-in"
}

// explore 에이전트
{
  agentType: "explore",  // (= "Plan")
  whenToUse: "Fast agent specialized for exploring codebases...",
  disallowedTools: [Write, Edit, NotebookEdit, Task, Plan],
  source: "built-in"
}

// Plan 에이전트
{
  agentType: "Plan",
  whenToUse: "Software architect agent for designing implementation plans.",
  disallowedTools: [Task, Plan, Write, Edit, NotebookEdit],
  source: "projectSettings"
}
```

### 팀 에이전트 (in-process teammates)

```text
[inProcessRunner] {agentName} starting poll loop
[inProcessRunner] Claimed task #{id}: {subject}
[inProcessRunner] Failed to claim task #{id}: {reason}
```

팀 에이전트는 태스크 목록을 폴링하면서 작업을 claim하고 실행한다.

---

## 권한 시스템 [inferred]

도구 실행 전 권한 확인:

```text
type: "command_permissions"
allowedTools: [...]
model: "..."
```

`<system-reminder>` 태그를 사용한 권한 주입:

```text
/^<system-reminder>\n?([\s\S]*?)\n?<\/system-reminder>$/
```

---

## Hook 시스템 [binary-inspected]

Hook 이벤트 유형:

```text
hook_success
hook_additional_context
hook_cancelled
hook_stopped_continuation
command_permissions
agent_mention
budget_usd
critical_system_reminder
edited_image_file
edited_text_file
opened_file_in_ide
```

`hook_stopped_continuation` 예시:

```text
"{hookName} hook stopped continuation: {message}"
```

---

## 세션 관리 [binary-inspected]

### 세션 지속성

```text
shouldSkipPersistence():
  - TEST 환경이면서 TEST_ENABLE_SESSION_PERSISTENCE가 없는 경우
  - cleanupPeriodDays === 0
  - CLAUDE_CODE_SKIP_PROMPT_HISTORY 설정 시
```

### Git 통합

```text
auto-stash: "Claude Code auto-stash - {ISO timestamp}"
worktree 감지: tengu_worktree_detection
```

---

## 환경 변수 [binary-inspected]

바이너리에서 확인된 주요 환경 변수:

| 변수명 | 용도 |
|---|---|
| `CLAUDE_CODE_SIMPLE` | 최소 시스템 프롬프트 모드 |
| `CLAUDE_CODE_REMOTE` | 원격 실행 모드 |
| `CLAUDE_CODE_DISABLE_AUTO_MEMORY` | 자동 메모리 비활성화 |
| `CLAUDE_CODE_DISABLE_ATTACHMENTS` | 첨부 비활성화 |
| `CLAUDE_CODE_SKIP_PROMPT_HISTORY` | 프롬프트 히스토리 건너뛰기 |
| `CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS` | 파일 읽기 최대 토큰 |
| `CLAUDE_CODE_WEBSOCKET_AUTH_FILE_DESCRIPTOR` | WebSocket 인증 |
| `CLAUDE_ENV_FILE` | Bash 환경 파일 경로 |
| `CLAUDE_SESSION_*` | 세션 관련 |
| `CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS` | 내장 에이전트 비활성화 |
| `ANTHROPIC_BASE_URL` | API 엔드포인트 오버라이드 |
| `VCR_RECORD` | CI 테스트용 VCR 녹화 |

---

## 주요 상수 [binary-inspected]

```text
el7 = 25000     # 파일 읽기 기본 최대 토큰
Hd  = 200       # MEMORY.md 로드 최대 줄 수
tlH = 25000     # 메모리 디렉토리 최대 바이트
w16 = 3         # 탐색 에이전트 기본 결과 수
lxq = 180000    # (추정) 컨텍스트 윈도우 관련 상수
Qxq = 40000     # (추정) 최대 출력 토큰 관련 상수
Ld7 = 61440     # (추정) 컴팩션 관련 크기 한계
NQ6 = 700       # UI 관련 상수
kJ6 = (truncation limit for display)
```

---

## 참고

- 이 분석은 `strings` 명령으로 바이너리에서 추출한 텍스트 파편을 기반으로 한다
- minified/obfuscated 코드이므로 변수명(H, _, T, q 등)은 원본과 다르다
- 정확한 로직 흐름은 추정이며, 실제 구현과 다를 수 있다
- 바이너리 버전 2.1.84 기준이며, 버전 업데이트에 따라 내부 구조가 변경될 수 있다
