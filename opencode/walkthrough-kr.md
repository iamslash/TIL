# OpenCode Long-term Memory 동작 원리 (Walkthrough)

## 시나리오

사용자가 OpenCode에서 OpenAI GPT-4o를 사용하여 두 번의 대화를 한다.

```
프롬프트 1: "src/index.ts 파일을 읽고 어떤 역할인지 설명해줘"
프롬프트 2: "이 파일에 health check endpoint를 추가해줘"
```

## 전체 흐름

```
사용자 입력
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ ① Session 생성 (또는 기존 세션 사용)                  │
│    → SQLite session 테이블에 INSERT                   │
│                                                      │
│ ② User Message 저장                                  │
│    → SQLite message 테이블에 INSERT (role: user)      │
│    → SQLite part 테이블에 INSERT (type: text)         │
│                                                      │
│ ③ 대화 이력 조회 (Long-term Memory)                   │
│    → SQLite에서 이 세션의 모든 message + part 조회     │
│    → filterCompacted() 로 compaction 이후만 필터       │
│    → toModelMessages() 로 LLM API 형식으로 변환       │
│                                                      │
│ ④ LLM API 호출 (OpenAI GPT-4o)                       │
│    → Vercel AI SDK streamText() 사용                  │
│    → system prompt + 대화 이력 + tools 전달           │
│                                                      │
│ ⑤ Assistant Message 저장                              │
│    → SQLite message 테이블에 INSERT (role: assistant)  │
│    → SQLite part 테이블에 INSERT (text, tool 등)      │
│                                                      │
│ ⑥ 다음 프롬프트에서 ②~⑤ 반복                         │
│    → ③에서 이전 대화가 모두 조회되어 컨텍스트 유지     │
└─────────────────────────────────────────────────────┘
```

---

## Step 1: Session 생성

사용자가 opencode를 실행하면 프로젝트를 식별하고 새 세션을 만든다.

### 프로젝트 식별

```typescript
// git repo의 최초 commit hash로 프로젝트를 식별
const result = await git(["rev-list", "--max-parents=0", "--all"], { cwd: worktree })
const projectID = result.text().split("\n").filter(Boolean).toSorted()[0]
// 예: "a1b2c3d4e5f6..."
```

### SQLite: project INSERT

```sql
INSERT INTO project (id, worktree, vcs, time_created, time_updated, sandboxes)
VALUES ('a1b2c3d4', '/Users/me/my-app', 'git', 1711234567000, 1711234567000, '[]');
```

```
project 테이블:
┌──────────┬──────────────────┬─────┬───────────────┐
│ id       │ worktree         │ vcs │ time_created   │
├──────────┼──────────────────┼─────┼───────────────┤
│ a1b2c3d4 │ /Users/me/my-app │ git │ 1711234567000 │
└──────────┴──────────────────┴─────┴───────────────┘
```

### Session 생성 코드

```typescript
// packages/opencode/src/session/index.ts
export async function createNext(input) {
  const result: Info = {
    id: SessionID.descending(),        // 시간 역순 정렬 가능한 ID
    slug: Slug.create(),               // URL-friendly 이름 (예: "brave-fox")
    version: Installation.VERSION,
    projectID: Instance.project.id,    // "a1b2c3d4"
    directory: input.directory,
    title: "New session - 2026-03-23T...",
    time: { created: Date.now(), updated: Date.now() },
  }

  Database.use((db) => {
    db.insert(SessionTable).values(toRow(result)).run()
  })

  return result
}
```

### SQLite: session INSERT

```sql
INSERT INTO session (id, project_id, slug, directory, title, version,
                     time_created, time_updated)
VALUES ('ses_001', 'a1b2c3d4', 'brave-fox', '/Users/me/my-app',
        'New session - 2026-03-23T10:00:00.000Z', '0.5.0',
        1711270800000, 1711270800000);
```

```
session 테이블:
┌─────────┬────────────┬───────────┬──────────────────────────────────────┐
│ id      │ project_id │ slug      │ title                                │
├─────────┼────────────┼───────────┼──────────────────────────────────────┤
│ ses_001 │ a1b2c3d4   │ brave-fox │ New session - 2026-03-23T10:00:00... │
└─────────┴────────────┴───────────┴──────────────────────────────────────┘
```

---

## Step 2: 첫 번째 프롬프트 — User Message 저장

사용자가 입력: **"src/index.ts 파일을 읽고 어떤 역할인지 설명해줘"**

### SQLite: message INSERT (User)

```sql
INSERT INTO message (id, session_id, time_created, time_updated, data)
VALUES ('msg_001', 'ses_001', 1711270801000, 1711270801000,
        '{"role":"user","time":{"created":1711270801000},
          "agent":"coder","model":{"providerID":"openai","modelID":"gpt-4o"}}');
```

### SQLite: part INSERT (Text)

```sql
INSERT INTO part (id, message_id, session_id, time_created, time_updated, data)
VALUES ('prt_001', 'msg_001', 'ses_001', 1711270801000, 1711270801000,
        '{"type":"text","text":"src/index.ts 파일을 읽고 어떤 역할인지 설명해줘"}');
```

```
message 테이블:
┌─────────┬────────────┬──────────────────────────────────────────────────┐
│ id      │ session_id │ data (JSON)                                      │
├─────────┼────────────┼──────────────────────────────────────────────────┤
│ msg_001 │ ses_001    │ {role:"user", model:{providerID:"openai",...}}   │
└─────────┴────────────┴──────────────────────────────────────────────────┘

part 테이블:
┌─────────┬────────────┬──────────────────────────────────────────────────┐
│ id      │ message_id │ data (JSON)                                      │
├─────────┼────────────┼──────────────────────────────────────────────────┤
│ prt_001 │ msg_001    │ {type:"text", text:"src/index.ts 파일을 읽고..."}│
└─────────┴────────────┴──────────────────────────────────────────────────┘
```

---

## Step 3: 대화 이력 조회 (Long-term Memory 핵심)

LLM API를 호출하기 전에 **이 세션의 모든 이전 대화를 SQLite에서 조회**한다.

### 조회 코드

```typescript
// packages/opencode/src/session/prompt.ts — SessionPrompt.prompt()
let msgs = await MessageV2.filterCompacted(MessageV2.stream(sessionID))
```

### SQL 실행 (내부)

```sql
-- 세션의 메시지를 시간 역순으로 페이지네이션하여 조회
SELECT * FROM message
WHERE session_id = 'ses_001'
ORDER BY time_created DESC, id DESC
LIMIT 51;  -- limit+1로 다음 페이지 존재 여부 확인

-- 조회된 메시지들의 part를 일괄 조회
SELECT * FROM part
WHERE message_id IN ('msg_001')
ORDER BY message_id, id;
```

### LLM API 형식으로 변환

```typescript
// packages/opencode/src/session/message-v2.ts — MessageV2.toModelMessages()
export function toModelMessages(input: WithParts[], model: Provider.Model): ModelMessage[]
```

이 함수가 SQLite에서 조회한 `{info, parts}[]` 배열을 Vercel AI SDK의
`ModelMessage[]` 형식으로 변환한다.

**중요: DB의 1개 message가 API의 여러 message로 확장될 수 있다.**
assistant message에 tool call + tool result + text가 모두 parts로 저장되어
있으면, API 전송 시에는 `assistant(tool_calls)` → `tool(result)` →
`assistant(text)` 여러 개의 message로 분리된다.

```
SQLite (1 message, 3 parts):
  {info: {role:"assistant"}, parts: [
    {type:"tool", tool:"read", state:{output:"..."}},
    {type:"text", text:"이 파일은..."},
    {type:"step-finish", cost:0.004}
  ]}

변환 후 (OpenAI API, 3 messages):
  {role: "assistant", tool_calls: [{function:{name:"read",...}}]}
  {role: "tool", tool_call_id: "call_abc", content: "..."}
  {role: "assistant", content: "이 파일은..."}
```

---

## Step 4: OpenAI GPT-4o API 호출

### streamText 호출

```typescript
// packages/opencode/src/session/llm.ts — LLM.stream()
return streamText({
  model: languageModel,          // OpenAI GPT-4o
  system: system,                // 시스템 프롬프트
  messages: input.messages,      // ← Step 3에서 변환한 대화 이력
  tools: input.tools,            // read, write, bash 등 도구
  maxSteps: 100,
  abortSignal: input.abort,
  // ...
})
```

### 실제 OpenAI API 요청 (Vercel AI SDK가 내부적으로 생성)

```
POST https://api.openai.com/v1/chat/completions
Authorization: Bearer sk-...
Content-Type: application/json

{
  "model": "gpt-4o",
  "stream": true,
  "messages": [
    {
      "role": "system",
      "content": "You are an AI coding assistant..."
    },
    {
      "role": "user",
      "content": "src/index.ts 파일을 읽고 어떤 역할인지 설명해줘"
    }
  ],
  "tools": [
    {"type":"function","function":{"name":"read","parameters":{...}}},
    {"type":"function","function":{"name":"write","parameters":{...}}},
    ...
  ]
}
```

### GPT-4o 응답 (tool_call)

```json
{
  "choices": [{
    "delta": {
      "role": "assistant",
      "tool_calls": [{
        "function": {
          "name": "read",
          "arguments": "{\"file_path\":\"src/index.ts\"}"
        }
      }]
    }
  }]
}
```

---

## Step 5: Assistant Message 저장

GPT-4o의 응답을 SQLite에 저장한다.

### SQLite: message INSERT (Assistant)

```sql
INSERT INTO message (id, session_id, time_created, time_updated, data)
VALUES ('msg_002', 'ses_001', 1711270802000, 1711270802000,
        '{"role":"assistant","parentID":"msg_001",
          "providerID":"openai","modelID":"gpt-4o",
          "agent":"coder","mode":"normal",
          "cost":0.0045,
          "tokens":{"input":1500,"output":200,"reasoning":0,
                    "cache":{"read":0,"write":0}},
          "time":{"created":1711270802000,"completed":1711270805000},
          "path":{"cwd":"/Users/me/my-app","root":"/Users/me/my-app"}}');
```

### SQLite: part INSERT (Tool — read 도구 호출)

```sql
INSERT INTO part (id, message_id, session_id, time_created, time_updated, data)
VALUES ('prt_002', 'msg_002', 'ses_001', 1711270802000, 1711270802000,
        '{"type":"tool","callID":"call_abc123","tool":"read",
          "state":{"status":"completed",
                   "input":{"file_path":"src/index.ts"},
                   "output":"import express from \"express\"...",
                   "title":"read src/index.ts",
                   "metadata":{},
                   "time":{"start":1711270802000,"end":1711270803000}}}');
```

### SQLite: part INSERT (Text — 설명)

```sql
INSERT INTO part (id, message_id, session_id, time_created, time_updated, data)
VALUES ('prt_003', 'msg_002', 'ses_001', 1711270803000, 1711270803000,
        '{"type":"text",
          "text":"이 파일은 Express.js 기반 API 서버의 진입점입니다..."}');
```

### SQLite: part INSERT (Step Finish — 토큰/비용)

```sql
INSERT INTO part (id, message_id, session_id, time_created, time_updated, data)
VALUES ('prt_004', 'msg_002', 'ses_001', 1711270805000, 1711270805000,
        '{"type":"step-finish","reason":"stop",
          "cost":0.0045,
          "tokens":{"input":1500,"output":200,"reasoning":0,
                    "cache":{"read":0,"write":0}}}');
```

### 첫 번째 프롬프트 후 DB 상태

```
message 테이블:
┌─────────┬────────────┬───────────────────────────────────────┐
│ id      │ session_id │ data.role  │ data.modelID             │
├─────────┼────────────┼───────────┼──────────────────────────┤
│ msg_001 │ ses_001    │ user      │ gpt-4o                   │
│ msg_002 │ ses_001    │ assistant │ gpt-4o                   │
└─────────┴────────────┴───────────┴──────────────────────────┘

part 테이블:
┌─────────┬────────────┬───────────────────────────────────────┐
│ id      │ message_id │ data.type  │ 요약                     │
├─────────┼────────────┼───────────┼──────────────────────────┤
│ prt_001 │ msg_001    │ text      │ "src/index.ts 파일을..."  │
│ prt_002 │ msg_002    │ tool      │ read src/index.ts         │
│ prt_003 │ msg_002    │ text      │ "이 파일은 Express..."    │
│ prt_004 │ msg_002    │ step-finish│ cost:0.0045, tokens:...  │
└─────────┴────────────┴───────────┴──────────────────────────┘
```

---

## Step 6: 두 번째 프롬프트 — 이전 대화를 기억한다

사용자가 입력: **"이 파일에 health check endpoint를 추가해줘"**

### ② User Message 저장

```sql
INSERT INTO message (id, session_id, ..., data)
VALUES ('msg_003', 'ses_001', ...,
        '{"role":"user",...}');

INSERT INTO part (id, message_id, ..., data)
VALUES ('prt_005', 'msg_003', ...,
        '{"type":"text","text":"이 파일에 health check endpoint를 추가해줘"}');
```

### ③ 대화 이력 조회 (핵심!)

```sql
-- 이번에는 msg_001, msg_002, msg_003 모두 조회된다
SELECT * FROM message WHERE session_id = 'ses_001'
ORDER BY time_created DESC, id DESC;

-- 모든 메시지의 part도 조회
SELECT * FROM part WHERE message_id IN ('msg_001', 'msg_002', 'msg_003')
ORDER BY message_id, id;
```

### ④ OpenAI API 호출 (이전 대화 포함)

```
POST https://api.openai.com/v1/chat/completions

{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "You are an AI coding assistant..."},

    // ← msg_001: 첫 번째 사용자 프롬프트
    {"role": "user",
     "content": "src/index.ts 파일을 읽고 어떤 역할인지 설명해줘"},

    // ← msg_002: 첫 번째 AI 응답 (tool call + 텍스트)
    {"role": "assistant",
     "tool_calls": [{"function":{"name":"read","arguments":"{...}"}}]},
    {"role": "tool",
     "tool_call_id": "call_abc123",
     "content": "import express from \"express\"..."},
    {"role": "assistant",
     "content": "이 파일은 Express.js 기반 API 서버의 진입점입니다..."},

    // ← msg_003: 두 번째 사용자 프롬프트
    {"role": "user",
     "content": "이 파일에 health check endpoint를 추가해줘"}
  ],
  "tools": [...]
}
```

**GPT-4o는 이전 대화를 모두 받으므로:**
- "이 파일"이 `src/index.ts`를 가리킨다는 것을 안다
- 이미 파일 내용을 읽었으므로 바로 수정 코드를 작성할 수 있다

### ⑤ GPT-4o가 write 도구 호출 → 결과 저장

```sql
-- Assistant message
INSERT INTO message (id, session_id, ..., data)
VALUES ('msg_004', 'ses_001', ..., '{"role":"assistant",...}');

-- Tool part: write 도구 호출
INSERT INTO part (id, message_id, ..., data)
VALUES ('prt_006', 'msg_004', ...,
        '{"type":"tool","tool":"write",
          "state":{"status":"completed",
                   "input":{"file_path":"src/index.ts","content":"..."},
                   "output":"File written successfully",...}}');

-- Text part: 설명
INSERT INTO part (id, message_id, ..., data)
VALUES ('prt_007', 'msg_004', ...,
        '{"type":"text","text":"health check endpoint를 추가했습니다..."}');

-- Step Finish part: 토큰/비용 (첫 번째 프롬프트와 동일하게 저장됨)
INSERT INTO part (id, message_id, ..., data)
VALUES ('prt_008', 'msg_004', ...,
        '{"type":"step-finish","reason":"stop",
          "cost":0.005,
          "tokens":{"input":3000,"output":300,"reasoning":0,
                    "cache":{"read":1500,"write":0}}}');
```

### 두 번째 프롬프트 후 최종 DB 상태

```
message 테이블:
┌─────────┬────────────┬───────────┐
│ id      │ session_id │ data.role │
├─────────┼────────────┼───────────┤
│ msg_001 │ ses_001    │ user      │  "src/index.ts 파일을 읽고..."
│ msg_002 │ ses_001    │ assistant │  read tool + 설명
│ msg_003 │ ses_001    │ user      │  "health check endpoint를..."
│ msg_004 │ ses_001    │ assistant │  write tool + 설명
└─────────┴────────────┴───────────┘

part 테이블:
┌─────────┬────────────┬────────────┬────────────────────────────┐
│ id      │ message_id │ data.type  │ 요약                       │
├─────────┼────────────┼────────────┼────────────────────────────┤
│ prt_001 │ msg_001    │ text       │ "src/index.ts 파일을..."   │
│ prt_002 │ msg_002    │ tool       │ Read src/index.ts          │
│ prt_003 │ msg_002    │ text       │ "이 파일은 Express..."     │
│ prt_004 │ msg_002    │ step-finish│ cost, tokens               │
│ prt_005 │ msg_003    │ text       │ "health check endpoint..." │
│ prt_006 │ msg_004    │ tool       │ write src/index.ts         │
│ prt_007 │ msg_004    │ text       │ "추가했습니다..."          │
│ prt_008 │ msg_004    │ step-finish│ cost, tokens               │
└─────────┴────────────┴────────────┴────────────────────────────┘
```

---

## Long-term Memory의 핵심 원리

```
세션 내 기억:
  SQLite에서 해당 세션의 message + part를 전부 조회
  → toModelMessages()로 OpenAI API 형식 변환
  → messages 파라미터로 전달
  → LLM이 이전 대화를 "기억"

세션 간 기억:
  새 세션을 만들면 이전 세션의 대화는 포함되지 않는다.
  하지만 이전 세션은 SQLite에 영구 보관되어 있어
  언제든 다시 열면 대화를 이어갈 수 있다.
```

**LLM 자체는 기억이 없다.** 매 API 호출마다 이전 대화를 함께 전송해야 한다.
OpenCode의 long-term memory는 **SQLite에 대화를 저장하고, 매 호출 시 전체
이력을 조회하여 API에 전달하는 것**이다.

```
호출 1: messages = [user_1]
호출 2: messages = [user_1, assistant_1, user_2]           ← DB에서 조회
호출 3: messages = [user_1, assistant_1, user_2, assistant_2, user_3]  ← DB에서 조회
```

대화가 길어져 컨텍스트 윈도우를 초과하면 **compaction**(요약/압축)이
발동하여 오래된 대화를 요약본으로 대체한다.
