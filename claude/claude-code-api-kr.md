# Claude Code API 통신 구조

> **근거**: `source-inspected` (Anthropic 공식 Messages API 문서), `runtime-observed` (Claude Code 실행 중 대화 구조 관찰), `inferred` (일부 내부 동작은 외부에서 관찰 가능한 행동으로부터 추정)
>
> **주의**: 아래 JSON 예시들은 구조 설명을 위한 **의사 코드**이다. 실제 API 캡처가 아니며, 필드가 생략되거나 주석이 포함되어 있다.

Claude Code는 Anthropic Messages API를 사용하여 LLM과 통신한다.

## API Request 구조

```text
POST https://api.anthropic.com/v1/messages

{
  "model": "claude-opus-4-6",           // 설정에 따라 다른 모델 가능
  "max_tokens": 64000,                  // 설정에 따라 변동
  "thinking": {                         // extended thinking 활성화 시
    "type": "enabled",
    "budget_tokens": 10000              // 정확한 값은 미확인 (inferred)
  },
  "system": [
    {
      "type": "text",
      "text": "You are Claude Code...\n[CLAUDE.md 내용]\n[MEMORY.md 내용]"
    }
  ],
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": [...]},
    {"role": "user", "content": [...]}
  ],
  "tools": [
    {"name": "Read", "description": "...", "input_schema": {...}},
    {"name": "Bash", "description": "...", "input_schema": {...}},
    {"name": "mcp__glean_default__search", "description": "...", "input_schema": {...}}
  ]
}
```

### 핵심 필드

| 필드 | 설명 |
|------|------|
| `system` | CLAUDE.md + MEMORY.md 등 고정 컨텍스트. 매 요청마다 포함됨 |
| `messages` | 대화 히스토리 전체. 매 요청마다 누적됨 |
| `tools` | Claude가 호출 가능한 모든 도구 정의 |
| `thinking` | Extended thinking 활성화. 모델이 응답 전 사고하는 공간 |

## 도구 정의 출처 (Tools)

`tools` 배열에 포함되는 도구들은 3가지 출처에서 온다.

### 1. Built-in 도구 (Claude Code 내장)

`Read`, `Write`, `Edit`, `Bash`, `Glob`, `Grep`, `Agent`, `Skill`, `TaskCreate` 등. Claude Code 바이너리에 포함되어 있으며 사용자가 수정할 수 없다. (`runtime-observed`: 도구 목록은 대화 시작 시 system 메시지에서 확인됨. 바이너리 내장 여부는 `inferred`)

### 2. MCP 서버 도구 (플러그인/설정에서 정의)

MCP(Model Context Protocol) 서버가 제공하는 도구. 네이밍 규칙: (`runtime-observed`)

```text
mcp__{서버이름}__{도구이름}

mcp__glean_default__search                      ← Glean MCP 서버의 search 도구
mcp__plugin_oh-my-claudecode_t__python_repl     ← OMC 플러그인 "t" 서버의 python_repl 도구
mcp__github__search_code                        ← GitHub MCP 서버의 search_code 도구
```

MCP 서버 등록 위치:

| 스코프 | 파일 | 용도 |
|--------|------|------|
| 프로젝트 | `.mcp.json` (프로젝트 루트) | 프로젝트별 도구 |
| 글로벌 | `~/.claude/.mcp.json` | 모든 프로젝트에서 사용 |
| 플러그인 | 각 플러그인의 `.mcp.json` | 플러그인이 제공하는 도구 |

Claude Code가 시작되면 등록된 모든 MCP 서버를 자식 프로세스로 실행하고, 각 서버가 제공하는 도구 목록을 수집하여 `tools` 배열에 추가한다. (`inferred`: MCP 프로토콜 스펙 기반 추정)

### 3. Deferred 도구 (지연 로드)

도구가 너무 많으면 매 요청마다 모든 도구 스키마를 보내는 것이 비효율적이다. 이를 위해 **이름만 등록**해두고 실제 사용 시 `ToolSearch`로 스키마를 가져오는 지연 로드 메커니즘이 있다. (`runtime-observed`: 대화 시작 시 deferred tool 목록이 system-reminder에 표시됨)

```text
// 대화 시작 시 system 메시지에 이름만 나열
"The following deferred tools are available via ToolSearch:
 Read, Edit, Bash, Grep, Glob, Agent, mcp__glean_default__search..."

// 실제 사용 시 ToolSearch로 전체 스키마 로드
ToolSearch("select:Read,Bash") → 전체 JSON Schema 반환 → 이후 호출 가능
```

### 요약

```text
tools 배열
├── Built-in: Read, Write, Edit, Bash, Glob, Grep, Agent, Skill, Task*...
│   └── 대화 시작 시 항상 존재 (runtime-observed). 내장 방식은 inferred
├── MCP: mcp__glean_default__*, mcp__github__*, mcp__plugin_oh-my-claudecode_t__*...
│   └── .mcp.json에 등록된 MCP 서버들이 제공 (source-inspected)
└── Deferred: 이름만 등록, ToolSearch로 지연 로드
    └── 컨텍스트 절약을 위한 최적화 (runtime-observed)
```

## 메시지 타입 (Message Types)

Messages API에는 3가지 역할(role)이 있다: `system`, `user`, `assistant`. (`source-inspected`: Anthropic Messages API 문서)

### System 메시지

Request의 `system` 필드에 별도로 전달된다. `messages` 배열에는 포함되지 않는다.

```jsonc
{
  "system": [
    {
      "type": "text",
      "text": "You are Claude Code, Anthropic's official CLI..."
    }
  ]
}
```

Claude Code에서 system 메시지에 포함되는 내용: (`runtime-observed`)
- Claude Code 기본 동작 지침 (파일 읽기/쓰기 규칙, 보안 가이드라인 등)
- `CLAUDE.md` 파일 내용 (프로젝트별 지침)
- `MEMORY.md` 파일 내용 (자동 메모리)
- 현재 환경 정보 (OS, 셸, 모델명, 작업 디렉토리)
- 사용 가능한 스킬 목록

system 메시지는 **매 API 요청마다 동일하게 포함**된다. 대화 중 변하지 않는 "고정 컨텍스트"이다.

### User 메시지 (role: "user")

사용자 입력 + 도구 실행 결과를 전달한다. `content`는 문자열 또는 content block 배열이다.

```jsonc
// 단순 텍스트 입력
{"role": "user", "content": "hello-slack"}

// content block 배열 (훅 주입 + 텍스트)
{"role": "user", "content": "<system-reminder>...</system-reminder>\nhello-slack"}

// 도구 결과 반환 (Claude Code가 자동 생성)
{"role": "user", "content": [
  {"type": "tool_result", "tool_use_id": "toolu_01", "content": "파일 내용..."},
  {"type": "tool_result", "tool_use_id": "toolu_02", "content": "검색 결과..."}
]}
```

**OMC 훅 주입**: `<system-reminder>` 태그는 user 메시지의 텍스트 안에 포함된다. API 수준에서는 그냥 텍스트의 일부이지만, Claude는 이 태그를 시스템 지침으로 인식하도록 훈련되어 있다. (`inferred`: 훅 stdout이 system-reminder로 감싸지는 것은 runtime-observed이나, Claude의 인식 방식은 추정)

### Assistant 메시지 (role: "assistant")

모델의 응답이다. 항상 content block 배열로 구성된다.

```jsonc
{
  "role": "assistant",
  "content": [
    {"type": "thinking", "thinking": "스킬 지침을 분석하면..."},
    {"type": "text", "text": "6개 채널을 검색합니다."},
    {"type": "tool_use", "id": "toolu_01", "name": "mcp__glean_default__search", "input": {...}}
  ],
  "stop_reason": "tool_use"
}
```

### 대화 흐름에서의 메시지 교대 규칙

Messages API는 `user`와 `assistant`가 **반드시 교대**해야 한다: (`source-inspected`: API 문서)

```text
user → assistant → user → assistant → user → assistant
```

도구 호출 시 Claude Code가 자동으로 이 규칙을 맞춘다:
1. assistant가 `tool_use`로 응답
2. Claude Code가 도구를 실행
3. 결과를 `tool_result`로 감싸서 **user 메시지**로 전달
4. assistant가 다시 응답

## Content Block Types

### Assistant content blocks (role: "assistant")

| type | 설명 | 예시 |
|------|------|------|
| `text` | 일반 텍스트 응답 | `{"type": "text", "text": "완료"}` |
| `tool_use` | 도구 호출 요청 | `{"type": "tool_use", "id": "toolu_01", "name": "Bash", "input": {"command": "ls"}}` |
| `thinking` | 확장 사고 (scratch pad) | `{"type": "thinking", "thinking": "단계적 분석..."}` |

### User content blocks (role: "user")

| type | 설명 | 예시 |
|------|------|------|
| `text` | 일반 텍스트 입력 | `{"type": "text", "text": "hello-slack"}` |
| `tool_result` | 도구 실행 결과 반환 | `{"type": "tool_result", "tool_use_id": "toolu_01", "content": "..."}` |
| `image` | 이미지 (base64) | `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}` |

### 실제 대화 예시 (hello-slack)

> 아래는 구조 설명용 의사 코드이다. 실제 API 캡처가 아니며, 필드 생략과 주석이 포함되어 있다.

```jsonc
{
  "system": [{"type": "text", "text": "You are Claude Code...[CLAUDE.md]...[MEMORY.md]..."}],
  "messages": [
    // Turn 1: 사용자 입력 (+ skill-injector 훅이 주입한 <mnemosyne>)
    {
      "role": "user",
      "content": "<system-reminder>\n<mnemosyne>\n# hello-slack 스킬내용...\n</mnemosyne>\n</system-reminder>\nhello-slack"
    },

    // Turn 2: 모델 응답 (도구 호출)
    {
      "role": "assistant",
      "content": [
        {"type": "thinking", "thinking": "스킬 지침에 따라 채널을 병렬 검색..."},
        {"type": "text", "text": "채널을 병렬로 검색합니다."},
        {"type": "tool_use", "id": "toolu_01", "name": "mcp__glean_default__search",
         "input": {"query": "*", "app": "slack", "channel": "tinder-seoul-seek-internal"}},
        {"type": "tool_use", "id": "toolu_02", "name": "mcp__glean_default__search",
         "input": {"query": "*", "app": "slack", "channel": "ml-team-ml-seoul"}}
      ]
    },

    // Turn 3: 도구 결과 (Claude Code가 자동 생성, 훅 컨텍스트 포함)
    {
      "role": "user",
      "content": [
        {"type": "tool_result", "tool_use_id": "toolu_01",
         "content": "<system-reminder>PreToolUse hook: ...</system-reminder>\n{\"documents\":[...]}"},
        {"type": "tool_result", "tool_use_id": "toolu_02",
         "content": "<system-reminder>PreToolUse hook: ...</system-reminder>\n{\"documents\":[...]}"}
      ]
    },

    // Turn 4: 모델 응답 (파일 작성)
    {
      "role": "assistant",
      "content": [
        {"type": "thinking", "thinking": "검색 결과를 필터링하여 한글 요약 작성..."},
        {"type": "tool_use", "id": "toolu_03", "name": "Write",
         "input": {"file_path": ".../hello-slack-YYYY-MM-DD-kr.md", "content": "# 슬랙 주요 채널 요약..."}}
      ]
    },

    // Turn 5: 도구 결과
    {
      "role": "user",
      "content": [
        {"type": "tool_result", "tool_use_id": "toolu_03",
         "content": "File created successfully"}
      ]
    },

    // Turn 6: 모델 응답 (에디터 열기)
    {
      "role": "assistant",
      "content": [
        {"type": "tool_use", "id": "toolu_04", "name": "Bash",
         "input": {"command": "cursor ~/tmp/hello-slack-YYYY-MM-DD-kr.md"}}
      ]
    },

    // Turn 7: 도구 결과
    {
      "role": "user",
      "content": [
        {"type": "tool_result", "tool_use_id": "toolu_04",
         "content": ""}
      ]
    },

    // Turn 8: 최종 응답
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "hello-slack 실행 완료. 파일을 Cursor에서 열었습니다."}
      ]
      // stop_reason: "end_turn"
    }
  ]
}
```

## stop_reason

| 값 | 의미 | 근거 |
|----|------|------|
| `end_turn` | 모델이 응답 완료 | `source-inspected` |
| `tool_use` | 모델이 도구 호출을 요청하여 멈춤 | `source-inspected` |
| `max_tokens` | 토큰 한도 도달 | `source-inspected` |

## 도구 호출 루프

Claude Code는 `stop_reason`에 따라 루프를 돌린다: (`inferred`: 외부에서 관찰 가능한 행동으로부터 추정한 의사 코드)

```python
# 의사 코드 - 실제 구현이 아님
while True:
    response = api.call(messages)

    if response.stop_reason == "end_turn":
        display(response.text)
        break

    if response.stop_reason == "tool_use":
        for tool_call in response.tool_uses:
            result = execute_tool(tool_call)
            messages.append(assistant_message)
            messages.append(tool_result_message)
        continue  # 다시 API 호출
```

## Thinking (확장 사고)

`thinking` 블록은 모델이 답변 전에 단계적으로 사고하는 scratch pad이다. (`source-inspected`: Anthropic extended thinking 문서)

- **주 목적**: 모델의 추론 품질 향상 (chain-of-thought 강제)
- **부차적 목적**: 사용자가 디버깅 용도로 열람 가능
- **다음 요청에 포함**: 이전 턴의 thinking이 다시 전달되어 모델이 자신의 사고를 기억
- 활성화: `settings.json`의 `"alwaysThinkingEnabled": true` (`runtime-observed`)

## 컨텍스트 누적

매 API 요청에 전체 대화 히스토리가 포함된다. 대화가 길어지면 Claude Code가 자동으로 이전 메시지를 압축(compact)한다. (`runtime-observed`)

```text
Request #1:  ████░░░░░░░░░░░░░░░░░░  (system + user + tools)
Request #2:  ████████████████░░░░░░  (+ 도구 결과들)
Request #3:  █████████████████░░░░░  (+ 추가 도구 결과)
...
Request #N:  ██████████████████████  → PreCompact 발동 → 압축
```

> 구체적인 크기(KB)는 대화 내용에 따라 크게 달라지므로 생략한다.
