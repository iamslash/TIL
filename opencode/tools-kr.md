# OpenCode Tool Call

## 개요

OpenCode는 LLM에게 **도구(tool)**를 제공하여 파일 읽기/쓰기, 셸 명령 실행,
웹 검색 등을 수행할 수 있게 한다. LLM이 도구를 호출하면 OpenCode가 실행하고
결과를 다시 LLM에게 전달한다.

## Tool Call 흐름

```
LLM (GPT-4o 등)                     OpenCode
    │                                    │
    │ ① tool_call: Read                  │
    │    {file_path: "src/index.ts"}     │
    │ ──────────────────────────────────▶ │
    │                                    │ ② 파일 읽기 실행
    │                                    │
    │ ③ tool_result:                     │
    │    "import express from..."         │
    │ ◀────────────────────────────────── │
    │                                    │
    │ ④ 응답 생성 (파일 내용 기반)        │
    │ ──────────────────────────────────▶ │
```

LLM API(OpenAI Chat Completions 등)의 **function calling** 기능을 사용한다.
OpenCode는 Vercel AI SDK의 `streamText()`에 `tools` 파라미터로 도구를 등록한다.

```typescript
// packages/opencode/src/session/llm.ts
return streamText({
  model: languageModel,
  messages: input.messages,
  tools: input.tools,     // ← 여기에 도구 등록
  maxSteps: 100,          // 최대 100번 tool call 가능
})
```

## Tool 정의 구조

모든 도구는 `Tool.define()`으로 정의된다.

```typescript
// packages/opencode/src/tool/tool.ts
export function define<Parameters, Result>(
  id: string,
  init: {
    description: string,                        // LLM에게 보여주는 설명
    parameters: z.ZodType,                      // 입력 스키마 (JSON Schema)
    execute(args, ctx): Promise<{               // 실행 함수
      title: string,                            // UI에 표시할 제목
      metadata: Record<string, any>,            // 메타데이터
      output: string,                           // LLM에 반환할 결과
    }>
  }
)
```

각 도구의 설명(description)은 별도 `.txt` 파일에 저장되어 있다. LLM에게 도구의
용도와 사용법을 자세히 알려주기 위함이다.

## Tool 목록

### 파일 시스템

| Tool ID | 파일 | 설명 |
|---------|------|------|
| `read` | `read.ts` | 파일 내용 읽기. 라인 범위 지정 가능 |
| `write` | `write.ts` | 파일 전체 내용 쓰기 (새 파일 생성 또는 덮어쓰기) |
| `edit` | `edit.ts` | 파일의 특정 부분을 old/new 문자열로 교체 |
| `multiedit` | `multiedit.ts` | 하나의 파일에 여러 edit을 한번에 적용 |
| `apply_patch` | `apply_patch.ts` | unified diff 형식의 패치 적용 |
| `glob` | `glob.ts` | 파일 패턴 매칭으로 파일 목록 검색 |
| `grep` | `grep.ts` | 파일 내용에서 정규식 패턴 검색 |
| `ls` | `ls.ts` | 디렉토리 내용 목록 조회 |

### 실행

| Tool ID | 파일 | 설명 |
|---------|------|------|
| `bash` | `bash.ts` | 셸 명령 실행. timeout, 작업 디렉토리 지정 가능 |
| `batch` | `batch.ts` | 여러 도구를 동시에 병렬 실행 |
| `task` | `task.ts` | 서브 에이전트에게 작업 위임 (병렬 실행 가능) |

### 검색 / 정보

| Tool ID | 파일 | 설명 |
|---------|------|------|
| `codesearch` | `codesearch.ts` | 코드 심볼 검색 (LSP 기반) |
| `lsp` | `lsp.ts` | Language Server Protocol 작업 (정의 이동, 참조 찾기 등) |
| `websearch` | `websearch.ts` | 웹 검색 |
| `webfetch` | `webfetch.ts` | URL에서 웹 페이지 내용 가져오기 |

### 상호작용

| Tool ID | 파일 | 설명 |
|---------|------|------|
| `question` | `question.ts` | 사용자에게 질문하여 추가 정보 요청 |
| `todoread` | `todo.ts` | 할 일 목록 조회 |
| `todowrite` | `todo.ts` | 할 일 목록 추가/수정 |

### 계획 / 모드

| Tool ID | 파일 | 설명 |
|---------|------|------|
| `plan_enter` | `plan.ts` | 계획 모드 진입 (도구 실행 없이 계획만 작성) |
| `plan_exit` | `plan.ts` | 계획 모드 종료 |
| `skill` | `skill.ts` | 등록된 스킬(커맨드) 실행 |

### 유틸리티

| Tool ID | 파일 | 설명 |
|---------|------|------|
| `invalid` | `invalid.ts` | 잘못된 도구 호출 시 에러 메시지 반환 |

---

## 주요 도구 상세

### read

파일을 읽어 LLM에게 내용을 전달한다.

```
입력:
  file_path: "src/index.ts"     (필수)
  offset: 10                    (선택, 시작 라인)
  limit: 50                     (선택, 읽을 라인 수)

출력:
  "1: import express from 'express'\n2: const app = express()..."
```

### write

파일 전체를 작성하거나 새 파일을 생성한다.

```
입력:
  file_path: "src/health.ts"
  content: "export function health() { return { status: 'ok' } }"

출력:
  "File written successfully"
```

### edit

파일의 특정 부분을 교체한다. `old_string`을 `new_string`으로 바꾼다.

```
입력:
  file_path: "src/index.ts"
  old_string: "app.listen(3000)"
  new_string: "app.listen(process.env.PORT || 3000)"

출력:
  "File edited successfully"
```

### bash

셸 명령을 실행한다.

```
입력:
  command: "npm test"
  description: "Run unit tests"
  timeout: 30000                 (선택, 밀리초)

출력:
  "PASS  src/test/api.test.ts\n  ✓ GET /health (5ms)\n  ✓ GET / (3ms)"
```

### batch

여러 도구를 **병렬로** 실행한다. 독립적인 작업을 동시에 수행하여 속도를 높인다.

```
입력:
  invocations: [
    { tool: "read", input: { file_path: "src/a.ts" } },
    { tool: "read", input: { file_path: "src/b.ts" } },
    { tool: "read", input: { file_path: "src/c.ts" } }
  ]

출력:
  세 파일의 내용이 동시에 반환됨
```

### task

작업을 **서브 에이전트**에게 위임한다. 메인 에이전트의 컨텍스트와 독립적으로
실행되며, 복잡한 작업을 분할할 때 사용한다.

```
입력:
  description: "src/routes.ts에 CRUD 엔드포인트를 추가해줘"
  prompt: "REST API CRUD endpoints for users..."

출력:
  서브 에이전트의 실행 결과
```

### question

사용자에게 질문하여 추가 정보를 얻는다. LLM이 확신이 없을 때 사용한다.

```
입력:
  question: "PostgreSQL과 MySQL 중 어떤 데이터베이스를 사용하시겠습니까?"

출력:
  (사용자의 답변) "PostgreSQL로 해주세요"
```

---

## Tool Call의 SQLite 저장

도구 호출 결과는 `part` 테이블에 `type: "tool"` 로 저장된다.

```sql
INSERT INTO part (id, message_id, session_id, time_created, time_updated, data)
VALUES ('prt_010', 'msg_005', 'ses_001', ...,
  '{
    "type": "tool",
    "callID": "call_abc123",
    "tool": "read",
    "state": {
      "status": "completed",
      "input": {"file_path": "src/index.ts"},
      "output": "import express from ...",
      "title": "Read src/index.ts",
      "metadata": {"lines": 42},
      "time": {"start": 1711270000000, "end": 1711270001000}
    }
  }');
```

### Tool State 상태 흐름

```
pending → running → completed
                  → error
```

| 상태 | 설명 |
|------|------|
| `pending` | LLM이 tool call을 요청했지만 아직 실행 안 됨 |
| `running` | 도구가 실행 중 |
| `completed` | 실행 완료. `output`에 결과가 있음 |
| `error` | 실행 실패. `error`에 에러 메시지가 있음 |

---

## 권한 (Permission)

도구 실행 시 권한 확인이 필요할 수 있다. 파일 쓰기, 셸 실행 등은 사용자
승인을 요청한다.

```typescript
// 도구 실행 내부에서
await ctx.ask({
  description: "Write to src/index.ts",
  // 사용자에게 승인 요청
})
```

## Output Truncation

도구 출력이 너무 길면 자동으로 잘린다. LLM의 컨텍스트를 보호하기 위함이다.

```typescript
// packages/opencode/src/tool/tool.ts
const truncated = await Truncate.output(result.output, {}, agent)
return {
  ...result,
  output: truncated.content,
  metadata: { ...result.metadata, truncated: truncated.truncated },
}
```

## MCP (Model Context Protocol) 도구

위의 내장 도구 외에도 **MCP 서버**에서 제공하는 도구를 연결할 수 있다.
MCP 도구는 외부 프로세스로 실행되며, OpenCode가 MCP 프로토콜을 통해 통신한다.

```
내장 도구:  read, write, edit, bash, grep, glob, ...
MCP 도구:   외부 MCP 서버가 제공하는 도구 (GitHub, Slack, DB 등)
```
