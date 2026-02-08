# OpenClaw LLM API 호출 흐름 분석

> David과 Telegram Bot이 대화하는 시나리오를 통해 OpenClaw의 LLM API 호출 과정을 상세히 분석합니다.
> DeepSeek API를 예시로 사용합니다.

## 목차
- [1. 전체 아키텍처 개요](#1-전체-아키텍처-개요)
- [2. 메시지 수신 흐름](#2-메시지-수신-흐름)
- [3. 컨텍스트 빌드 과정](#3-컨텍스트-빌드-과정)
- [4. LLM API 호출 구조](#4-llm-api-호출-구조)
- [5. Request/Response JSON 상세](#5-requestresponse-json-상세)
- [6. DeepSeek 설정 방법](#6-deepseek-설정-방법)
- [7. 응답 전달 과정](#7-응답-전달-과정)
- [8. 생각하는 모델 (Reasoning Model) API 흐름](#8-생각하는-모델-reasoning-model-api-흐름)
- [9. Prompt Caching (프롬프트 캐싱)](#9-prompt-caching-프롬프트-캐싱)

---

## 1. 전체 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OpenClaw 메시지 처리 흐름                              │
└─────────────────────────────────────────────────────────────────────────────┘

  David (Telegram)                   OpenClaw                        DeepSeek
       │                                │                                │
       │  "안녕, 오늘 날씨 어때?"        │                                │
       ├───────────────────────────────▶│                                │
       │     [Telegram Webhook]         │                                │
       │                                │                                │
       │                         ┌──────┴──────┐                         │
       │                         │   Grammy    │                         │
       │                         │ Bot Handler │                         │
       │                         └──────┬──────┘                         │
       │                                │                                │
       │                         ┌──────┴──────┐                         │
       │                         │   Context   │                         │
       │                         │   Builder   │                         │
       │                         └──────┬──────┘                         │
       │                                │                                │
       │                         ┌──────┴──────┐                         │
       │                         │  Auto-Reply │                         │
       │                         │   System    │                         │
       │                         └──────┬──────┘                         │
       │                                │                                │
       │                         ┌──────┴──────┐                         │
       │                         │  Embedded   │                         │
       │                         │   Runner    │                         │
       │                         └──────┬──────┘                         │
       │                                │                                │
       │                                │   POST /v1/chat/completions    │
       │                                ├───────────────────────────────▶│
       │                                │                                │
       │                                │   SSE Stream Response          │
       │                                │◀───────────────────────────────┤
       │                                │                                │
       │     [Telegram API]             │                                │
       │◀───────────────────────────────┤                                │
       │  "안녕하세요! 현재 날씨..."      │                                │
       │                                │                                │
```

### 핵심 컴포넌트

| 컴포넌트 | 파일 위치 | 역할 |
|---------|----------|------|
| Webhook Server | `src/telegram/webhook.ts` | Telegram HTTP 요청 수신 |
| Grammy Bot | `src/telegram/bot.ts` | Telegram 메시지 처리 |
| Context Builder | `src/telegram/bot-message-context.ts` | 통합 컨텍스트 생성 |
| Auto-Reply | `src/auto-reply/reply/get-reply.ts` | 응답 생성 오케스트레이션 |
| Embedded Runner | `src/agents/pi-embedded-runner/run/attempt.ts` | LLM API 호출 실행 |
| pi-ai Library | `@mariozechner/pi-ai` | 실제 HTTP 요청 생성 |

---

## 2. 메시지 수신 흐름

### 2.1 Telegram Webhook 수신

David이 Telegram에서 메시지를 보내면:

```
David → Telegram Server → POST /telegram-webhook → OpenClaw
```

**Webhook 처리 코드** (`src/telegram/webhook.ts`):

```typescript
// Telegram이 보내는 Update 객체 예시
{
  "update_id": 123456789,
  "message": {
    "message_id": 1001,
    "from": {
      "id": 123456,
      "first_name": "David",
      "username": "david_sun"
    },
    "chat": {
      "id": 123456,
      "type": "private"
    },
    "date": 1706745600,
    "text": "안녕, 오늘 날씨 어때?"
  }
}
```

### 2.2 Grammy Bot Handler

```typescript
// src/telegram/bot-handlers.ts
bot.on("message", async (ctx) => {
  // 메시지 처리 시작
  await processMessage(ctx);
});
```

**처리 순서:**
1. 미디어 그룹 버퍼링 (사진 여러 장 전송 시)
2. 메시지 시퀀셜라이징 (순서 보장)
3. 스로틀링 (과부하 방지)

---

## 3. 컨텍스트 빌드 과정

### 3.1 MsgContext 생성

`buildTelegramMessageContext()` 함수가 Telegram 메시지를 통합 컨텍스트로 변환합니다.

**입력** (Telegram Message):
```typescript
{
  message_id: 1001,
  from: { id: 123456, first_name: "David", username: "david_sun" },
  chat: { id: 123456, type: "private" },
  text: "안녕, 오늘 날씨 어때?"
}
```

**출력** (MsgContext):
```typescript
{
  // 기본 정보
  Body: "안녕, 오늘 날씨 어때?",
  SenderId: "123456",
  SenderName: "David",
  SenderUsername: "david_sun",

  // 세션 정보
  SessionKey: "telegram:123456:private",
  MessageChannel: "telegram",

  // 채팅 정보
  ChatId: "123456",
  ChatType: "private",
  IsGroup: false,

  // 메타데이터
  MessageId: "1001",
  Timestamp: 1706745600,

  // 미디어 (있는 경우)
  MediaPaths: [],

  // 멘션/답장 정보
  WasMentioned: false,
  ReplyToMessageId: undefined
}
```

### 3.2 세션 키 구조

```
SessionKey = "{channel}:{chatId}:{scope}"

예시:
- telegram:123456:private     → David의 개인 채팅
- telegram:-100123:group      → 그룹 채팅
- discord:guild123:channel456 → 디스코드 채널
```

---

## 4. LLM API 호출 구조

### 4.1 Agent Runner 흐름

```
┌────────────────────────────────────────────────────────────────┐
│                    runEmbeddedAttempt()                        │
│                src/agents/pi-embedded-runner/run/attempt.ts    │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  1. resolveModel()                      │
        │     - ModelRegistry에서 모델 정보 조회    │
        │     - provider: "deepseek"              │
        │     - modelId: "deepseek-chat"          │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  2. getApiKeyForModel()                 │
        │     - auth.json에서 API 키 로드          │
        │     - 또는 환경변수에서 로드              │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  3. createAgentSession()                │
        │     - pi-coding-agent 라이브러리 사용    │
        │     - 시스템 프롬프트 설정               │
        │     - 도구(Tools) 등록                  │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  4. activeSession.prompt()              │
        │     - 사용자 메시지 전달                 │
        │     - streamSimple() 호출               │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │  5. @mariozechner/pi-ai                 │
        │     - HTTP POST 요청 생성               │
        │     - SSE 스트림 파싱                   │
        └─────────────────────────────────────────┘
```

### 4.2 streamSimple 함수

```typescript
// src/agents/pi-embedded-runner/run/attempt.ts:509
activeSession.agent.streamFn = streamSimple;

// streamSimple은 @mariozechner/pi-ai 라이브러리의 함수
// OpenAI 호환 API를 호출하는 핵심 함수
```

---

## 5. Request/Response JSON 상세

### 5.1 DeepSeek API Request

**HTTP 요청:**
```http
POST https://api.deepseek.com/v1/chat/completions
Content-Type: application/json
Authorization: Bearer sk-xxxxxxxxxxxxxxxx
```

**Request Body:**
```json
{
  "model": "deepseek-chat",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant. Current time: 2024-02-01 15:30:00 KST.\n\nUser info:\n- Name: David\n- Platform: Telegram\n- Chat type: private\n\nYou have access to the following tools:\n- bash: Execute shell commands\n- view: View file contents\n- edit: Edit files\n..."
    },
    {
      "role": "user",
      "content": "안녕, 오늘 날씨 어때?"
    }
  ],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 4096
}
```

**Request Body 상세 설명:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `model` | string | 사용할 모델 ID |
| `messages` | array | 대화 히스토리 |
| `messages[].role` | string | `system`, `user`, `assistant`, `tool` 중 하나 |
| `messages[].content` | string/array | 메시지 내용 |
| `stream` | boolean | SSE 스트리밍 사용 여부 |
| `temperature` | number | 응답 다양성 (0.0 ~ 2.0) |
| `max_tokens` | number | 최대 생성 토큰 수 |

### 5.2 System Prompt 구성

시스템 프롬프트는 `buildEmbeddedSystemPrompt()` 함수에서 생성됩니다:

```typescript
// src/agents/pi-embedded-runner/system-prompt.ts

const systemPrompt = `
You are OpenClaw, an AI assistant.

## Runtime Information
- Host: ${machineName}
- OS: ${os.type()} ${os.release()}
- Model: ${provider}/${modelId}
- Channel: ${messageChannel}

## User Context
- Sender: ${senderName} (@${senderUsername})
- Chat: ${chatType}
- Session: ${sessionKey}

## Available Tools
${toolDescriptions}

## Workspace
- Directory: ${workspaceDir}
- Bootstrap files loaded

## Instructions
- Be helpful and concise
- Use tools when needed
- Respect user privacy
`;
```

### 5.3 DeepSeek API Response (Streaming)

**SSE 스트림 형식:**

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1706745601,"model":"deepseek-chat","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1706745601,"model":"deepseek-chat","choices":[{"index":0,"delta":{"content":"안녕"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1706745601,"model":"deepseek-chat","choices":[{"index":0,"delta":{"content":"하세요"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1706745601,"model":"deepseek-chat","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1706745601,"model":"deepseek-chat","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":150,"completion_tokens":25,"total_tokens":175}}

data: [DONE]
```

**스트림 청크 구조:**

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1706745601,
  "model": "deepseek-chat",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "안녕"
      },
      "finish_reason": null
    }
  ]
}
```

### 5.4 Non-Streaming Response (참고용)

스트리밍을 사용하지 않을 경우:

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1706745601,
  "model": "deepseek-chat",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "안녕하세요! 현재 제가 실시간 날씨 정보에 접근할 수 없어서 정확한 날씨를 알려드리기 어렵습니다. 날씨 정보는 기상청 웹사이트나 날씨 앱을 확인해 주세요."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 45,
    "total_tokens": 195
  }
}
```

### 5.5 Tool Call Request/Response

Bot이 도구를 사용해야 할 때:

**Request (도구 사용 가능 모델):**
```json
{
  "model": "deepseek-chat",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "현재 디렉토리의 파일 목록을 보여줘"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "bash",
        "description": "Execute a shell command",
        "parameters": {
          "type": "object",
          "properties": {
            "command": {
              "type": "string",
              "description": "The command to execute"
            }
          },
          "required": ["command"]
        }
      }
    }
  ],
  "stream": true
}
```

**Response (Tool Call):**
```json
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "bash",
              "arguments": "{\"command\": \"ls -la\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

**Tool Result를 포함한 후속 Request:**
```json
{
  "model": "deepseek-chat",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "현재 디렉토리의 파일 목록을 보여줘"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_abc123",
          "type": "function",
          "function": {
            "name": "bash",
            "arguments": "{\"command\": \"ls -la\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_abc123",
      "content": "total 24\ndrwxr-xr-x  5 david staff  160 Feb  1 15:30 .\ndrwxr-xr-x 10 david staff  320 Feb  1 15:00 ..\n-rw-r--r--  1 david staff 1024 Feb  1 15:25 README.md\n-rw-r--r--  1 david staff 2048 Feb  1 15:30 main.ts"
    }
  ],
  "stream": true
}
```

---

## 6. DeepSeek 설정 방법

### 6.1 모델 설정 파일

**`~/.openclaw/agents/main/models.json`:**
```json
{
  "providers": {
    "deepseek": {
      "baseUrl": "https://api.deepseek.com",
      "api": "openai-completions",
      "models": [
        {
          "id": "deepseek-chat",
          "name": "DeepSeek Chat",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": 64000,
          "maxTokens": 4096,
          "cost": {
            "input": 0.14,
            "output": 0.28,
            "cacheRead": 0.014,
            "cacheWrite": 0.14
          }
        },
        {
          "id": "deepseek-reasoner",
          "name": "DeepSeek Reasoner",
          "reasoning": true,
          "input": ["text"],
          "contextWindow": 64000,
          "maxTokens": 8192,
          "cost": {
            "input": 0.55,
            "output": 2.19,
            "cacheRead": 0.055,
            "cacheWrite": 0.55
          }
        }
      ]
    }
  }
}
```

### 6.2 인증 설정

**`~/.openclaw/agents/main/auth.json`:**
```json
{
  "deepseek": {
    "apiKey": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  }
}
```

또는 **환경변수:**
```bash
export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 6.3 OpenClaw 설정

**`~/.openclaw/config.yaml`:**
```yaml
agents:
  defaults:
    model:
      primary: deepseek/deepseek-chat
      fallback: deepseek/deepseek-chat

models:
  providers:
    deepseek:
      baseUrl: https://api.deepseek.com
      api: openai-completions
```

### 6.4 API 유형 설명

| API 유형 | 설명 | 사용 프로바이더 |
|---------|------|----------------|
| `openai-completions` | OpenAI Chat Completions API 호환 | DeepSeek, OpenAI, Groq, Together |
| `openai-responses` | OpenAI Responses API (최신) | OpenAI GPT-4o |
| `anthropic-messages` | Anthropic Messages API | Claude |
| `google-generative-ai` | Google Gemini API | Gemini |
| `bedrock-converse-stream` | AWS Bedrock Converse API | Bedrock 모델들 |

---

## 7. 응답 전달 과정

### 7.1 스트림 처리

```typescript
// subscribeEmbeddedPiSession()에서 스트림 이벤트 구독
const subscription = subscribeEmbeddedPiSession({
  session: activeSession,
  onPartialReply: (text) => {
    // 스트리밍 중간 텍스트 업데이트
    updateDraftFromPartial(text);
  },
  onBlockReply: (block) => {
    // 완성된 블록 처리
    deliverBlock(block);
  }
});
```

### 7.2 Telegram 응답 전송

```typescript
// src/telegram/bot/delivery.ts
async function deliverReplies(payloads: ReplyPayload[]) {
  for (const payload of payloads) {
    if (payload.type === "text") {
      await bot.api.sendMessage(chatId, payload.text, {
        reply_to_message_id: replyToMessageId,
        parse_mode: "MarkdownV2"
      });
    }
  }
}
```

### 7.3 전체 응답 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                       응답 전달 파이프라인                         │
└─────────────────────────────────────────────────────────────────┘

  DeepSeek API                  OpenClaw                    Telegram
       │                           │                            │
       │  SSE: "안녕"              │                            │
       ├──────────────────────────▶│                            │
       │                           │  [Buffer: "안녕"]          │
       │                           │                            │
       │  SSE: "하세요"            │                            │
       ├──────────────────────────▶│                            │
       │                           │  [Buffer: "안녕하세요"]     │
       │                           │                            │
       │  SSE: "!"                 │                            │
       ├──────────────────────────▶│                            │
       │                           │  [Buffer: "안녕하세요!"]    │
       │                           │                            │
       │  SSE: [DONE]              │                            │
       ├──────────────────────────▶│                            │
       │                           │                            │
       │                           │  sendMessage()             │
       │                           ├───────────────────────────▶│
       │                           │                            │
       │                           │                      David │
       │                           │                       ◀────┤
       │                           │               "안녕하세요!" │
```

---

## 8. 생각하는 모델 (Reasoning Model) API 흐름

> 일반 모델과 달리, Reasoning Model은 최종 답변 전에 **내부 사고 과정(Chain of Thought)**을 거칩니다.
> 이 사고 과정은 API 응답에 별도 필드로 노출되며, 토큰 사용량도 별도로 집계됩니다.

### 8.1 일반 모델 vs 생각하는 모델

```
┌─────────────────────────────────────────────────────────────┐
│                    일반 모델 (deepseek-chat)                  │
│                                                             │
│   User Message ──▶ [모델 추론] ──▶ Assistant Response        │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              생각하는 모델 (deepseek-reasoner)                 │
│                                                             │
│   User Message ──▶ [내부 사고 과정] ──▶ [최종 답변 생성]      │
│                     reasoning_content      content          │
│                     (사용자에게 노출 가능)   (최종 응답)       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 프로바이더별 Reasoning Model 비교

| 프로바이더 | 모델 | API 유형 | Reasoning 필드 | 제어 파라미터 |
|-----------|------|---------|---------------|-------------|
| DeepSeek | `deepseek-reasoner` | `openai-completions` | `reasoning_content` | 없음 (항상 활성) |
| OpenAI | `o1`, `o3`, `o4-mini` | `openai-completions` | 내부 처리 (비공개) | `reasoning_effort` |
| Anthropic | `claude-sonnet-4-5-20250929` 등 | `anthropic-messages` | `thinking` block | `thinking.budget_tokens` |
| Google | `gemini-2.5-pro` 등 | `google-generative-ai` | `thought` 파트 | `thinkingConfig.thinkingBudget` |

### 8.3 DeepSeek Reasoner API 흐름

OpenClaw에서 `deepseek-reasoner`를 사용할 때의 전체 흐름입니다.

#### 8.3.1 모델 감지

```typescript
// models.json에서 reasoning: true로 설정된 모델
{
  "id": "deepseek-reasoner",
  "name": "DeepSeek Reasoner",
  "reasoning": true,     // ← 이 플래그로 reasoning 모델 판별
  "contextWindow": 64000,
  "maxTokens": 8192
}
```

OpenClaw의 `resolveModel()` 함수가 `reasoning: true`를 감지하면 reasoning 모델 전용 처리 경로를 사용합니다.

#### 8.3.2 Request

**HTTP 요청:**
```http
POST https://api.deepseek.com/v1/chat/completions
Content-Type: application/json
Authorization: Bearer sk-xxxxxxxxxxxxxxxx
```

**Request Body:**
```json
{
  "model": "deepseek-reasoner",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant. ..."
    },
    {
      "role": "user",
      "content": "피보나치 수열의 시간복잡도를 분석해줘"
    }
  ],
  "stream": true,
  "max_tokens": 8192
}
```

> **주의:** DeepSeek Reasoner는 `temperature`, `top_p` 등 샘플링 파라미터를 지원하지 않습니다.
> 이를 전송하면 API 오류가 발생합니다. OpenClaw은 reasoning 모델 감지 시 이 파라미터를 자동으로 제거합니다.

#### 8.3.3 Streaming Response

DeepSeek Reasoner의 SSE 스트림은 **두 단계**로 나뉩니다:

**1단계 - 사고 과정 (reasoning_content):**
```
data: {"id":"chatcmpl-456","object":"chat.completion.chunk","created":1706745601,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning_content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-456","object":"chat.completion.chunk","created":1706745601,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"reasoning_content":"피보나치 수열의 시간복잡도를 분석하기 위해"},"finish_reason":null}]}

data: {"id":"chatcmpl-456","object":"chat.completion.chunk","created":1706745601,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"reasoning_content":" 재귀적 구현과 동적 프로그래밍 구현을 비교해보자."},"finish_reason":null}]}

data: {"id":"chatcmpl-456","object":"chat.completion.chunk","created":1706745601,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"reasoning_content":" 재귀의 경우 T(n) = T(n-1) + T(n-2) + O(1)이므로..."},"finish_reason":null}]}
```

**2단계 - 최종 답변 (content):**
```
data: {"id":"chatcmpl-456","object":"chat.completion.chunk","created":1706745601,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"content":"# 피보나치 수열 시간복잡도 분석\n\n"},"finish_reason":null}]}

data: {"id":"chatcmpl-456","object":"chat.completion.chunk","created":1706745601,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"content":"## 1. 단순 재귀: O(2^n)\n"},"finish_reason":null}]}

data: {"id":"chatcmpl-456","object":"chat.completion.chunk","created":1706745601,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"content":"재귀 트리가 지수적으로 팽창합니다.\n\n"},"finish_reason":null}]}

data: {"id":"chatcmpl-456","object":"chat.completion.chunk","created":1706745601,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{"content":"## 2. DP (메모이제이션): O(n)\n"},"finish_reason":null}]}

data: {"id":"chatcmpl-456","object":"chat.completion.chunk","created":1706745601,"model":"deepseek-reasoner","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":120,"completion_tokens":85,"total_tokens":205,"completion_tokens_details":{"reasoning_tokens":340}}}

data: [DONE]
```

#### 8.3.4 스트림 청크 구조 비교

**일반 모델 (delta):**
```json
{
  "delta": {
    "content": "안녕하세요"
  }
}
```

**Reasoning 모델 (delta) - 사고 단계:**
```json
{
  "delta": {
    "reasoning_content": "사용자의 질문을 분석해보면..."
  }
}
```

**Reasoning 모델 (delta) - 답변 단계:**
```json
{
  "delta": {
    "content": "피보나치 수열의 시간복잡도는..."
  }
}
```

#### 8.3.5 Non-Streaming Response (참고용)

```json
{
  "id": "chatcmpl-456",
  "object": "chat.completion",
  "created": 1706745601,
  "model": "deepseek-reasoner",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "# 피보나치 수열 시간복잡도 분석\n\n## 1. 단순 재귀: O(2^n)\n재귀 트리가 지수적으로 팽창합니다.\n\n## 2. DP (메모이제이션): O(n)\n각 하위 문제를 한 번만 계산합니다.",
        "reasoning_content": "피보나치 수열의 시간복잡도를 분석하기 위해 재귀적 구현과 동적 프로그래밍 구현을 비교해보자. 재귀의 경우 T(n) = T(n-1) + T(n-2) + O(1)이므로 이 점화식은 피보나치 수열 자체와 같은 성장률을 가진다. 즉 O(φ^n) ≈ O(1.618^n)으로, 대략 O(2^n)이다..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 120,
    "completion_tokens": 85,
    "total_tokens": 205,
    "completion_tokens_details": {
      "reasoning_tokens": 340
    }
  }
}
```

> **토큰 비용 주의:** `reasoning_tokens`(340)는 `completion_tokens`(85)에 포함되지 않지만 **별도로 과금**됩니다.
> DeepSeek Reasoner의 경우 사고 토큰 비용이 출력 토큰과 동일합니다.

#### 8.3.6 멀티턴 대화에서의 reasoning_content 처리

Reasoning 모델과의 멀티턴 대화 시, 이전 턴의 `reasoning_content`를 포함할지 결정해야 합니다:

```json
{
  "model": "deepseek-reasoner",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "피보나치 수열의 시간복잡도를 분석해줘"},
    {
      "role": "assistant",
      "content": "# 피보나치 수열 시간복잡도 분석\n...",
      "reasoning_content": null
    },
    {"role": "user", "content": "행렬 거듭제곱 방법도 설명해줘"}
  ],
  "stream": true
}
```

> **OpenClaw 처리 방식:** 이전 턴의 `reasoning_content`는 컨텍스트 윈도우 절약을 위해 `null`로 설정합니다.
> 사고 과정은 해당 턴에서만 유효하며, 최종 `content`에 핵심 내용이 반영되어 있기 때문입니다.

### 8.4 OpenAI o-series API 흐름

OpenAI o1, o3, o4-mini 등의 reasoning 모델은 사고 과정을 **외부에 노출하지 않는** 방식입니다.

#### 8.4.1 Request

```json
{
  "model": "o3",
  "messages": [
    {"role": "user", "content": "피보나치 수열의 시간복잡도를 분석해줘"}
  ],
  "reasoning_effort": "medium",
  "stream": true
}
```

| 파라미터 | 값 | 설명 |
|---------|---|------|
| `reasoning_effort` | `"low"` / `"medium"` / `"high"` | 사고 깊이 제어. 높을수록 더 많은 reasoning 토큰 사용 |

> **주의:** o-series도 `temperature`, `top_p`를 지원하지 않습니다.

#### 8.4.2 Streaming Response

```
data: {"id":"chatcmpl-789","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-789","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"# 피보나치 수열 시간복잡도 분석\n\n"},"finish_reason":null}]}

data: {"id":"chatcmpl-789","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"## 1. 단순 재귀: O(2^n)\n"},"finish_reason":null}]}

data: {"id":"chatcmpl-789","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":15,"completion_tokens":80,"total_tokens":95,"completion_tokens_details":{"reasoning_tokens":256}}}

data: [DONE]
```

> **핵심 차이:** `reasoning_content` 필드가 없습니다. 사고 과정은 내부적으로만 수행되며,
> `usage.completion_tokens_details.reasoning_tokens`로 사용된 reasoning 토큰 수만 확인할 수 있습니다.

### 8.5 Anthropic Extended Thinking API 흐름

Anthropic Claude는 `anthropic-messages` API를 사용하며, 별도의 `thinking` 설정으로 제어합니다.

#### 8.5.1 Request

```json
{
  "model": "claude-sonnet-4-5-20250929",
  "max_tokens": 16000,
  "thinking": {
    "type": "enabled",
    "budget_tokens": 10000
  },
  "messages": [
    {
      "role": "user",
      "content": "피보나치 수열의 시간복잡도를 분석해줘"
    }
  ],
  "stream": true
}
```

| 파라미터 | 설명 |
|---------|------|
| `thinking.type` | `"enabled"` 로 설정하여 확장 사고 활성화 |
| `thinking.budget_tokens` | 사고에 사용할 최대 토큰 수. `max_tokens`보다 작아야 함 |

> **주의:** Extended Thinking 활성화 시 `temperature`는 반드시 `1`이어야 하며, `system` 프롬프트는 사용 가능합니다.

#### 8.5.2 Streaming Response

Anthropic은 **SSE가 아닌 자체 이벤트 형식**을 사용합니다:

**1단계 - 사고 과정 (thinking block):**
```
event: message_start
data: {"type":"message_start","message":{"id":"msg_01X","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5-20250929","usage":{"input_tokens":25,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"피보나치 수열의 시간복잡도를 분석해야 한다. "}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"재귀적 방법의 경우, T(n) = T(n-1) + T(n-2)이므로..."}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}
```

**2단계 - 최종 답변 (text block):**
```
event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"# 피보나치 수열 시간복잡도 분석\n\n"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"## 1. 단순 재귀: O(2^n)\n"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":95}}

event: message_stop
data: {"type":"message_stop"}
```

#### 8.5.3 Non-Streaming Response (참고용)

```json
{
  "id": "msg_01X",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "thinking",
      "thinking": "피보나치 수열의 시간복잡도를 분석해야 한다. 재귀적 방법의 경우, T(n) = T(n-1) + T(n-2)이므로 이 점화식의 해는 O(φ^n)이다..."
    },
    {
      "type": "text",
      "text": "# 피보나치 수열 시간복잡도 분석\n\n## 1. 단순 재귀: O(2^n)\n재귀 트리가 지수적으로 팽창합니다.\n\n## 2. DP (메모이제이션): O(n)\n각 하위 문제를 한 번만 계산합니다."
    }
  ],
  "model": "claude-sonnet-4-5-20250929",
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 25,
    "output_tokens": 95,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
  }
}
```

#### 8.5.4 멀티턴에서의 thinking block 처리

```json
{
  "model": "claude-sonnet-4-5-20250929",
  "max_tokens": 16000,
  "thinking": {
    "type": "enabled",
    "budget_tokens": 10000
  },
  "messages": [
    {"role": "user", "content": "피보나치 수열의 시간복잡도를 분석해줘"},
    {
      "role": "assistant",
      "content": [
        {"type": "thinking", "thinking": "피보나치 수열의 시간복잡도를..."},
        {"type": "text", "text": "# 피보나치 수열 시간복잡도 분석\n..."}
      ]
    },
    {"role": "user", "content": "행렬 거듭제곱 방법도 설명해줘"}
  ]
}
```

> **Anthropic 처리 방식:** DeepSeek과 달리, Anthropic은 이전 턴의 `thinking` block을 **반드시 포함**해야 합니다.
> 생략하면 API 오류가 발생합니다.

### 8.6 Google Gemini Thinking API 흐름

Gemini 2.5 시리즈는 `thinkingConfig`로 사고 과정을 제어합니다.

#### 8.6.1 Request

```json
{
  "contents": [
    {
      "role": "user",
      "parts": [{"text": "피보나치 수열의 시간복잡도를 분석해줘"}]
    }
  ],
  "generationConfig": {
    "thinkingConfig": {
      "thinkingBudget": 8192
    }
  }
}
```

#### 8.6.2 Response

```json
{
  "candidates": [
    {
      "content": {
        "parts": [
          {
            "thought": true,
            "text": "피보나치 수열의 시간복잡도를 분석하기 위해 여러 구현 방법을 비교해보자..."
          },
          {
            "text": "# 피보나치 수열 시간복잡도 분석\n\n## 1. 단순 재귀: O(2^n)\n..."
          }
        ],
        "role": "model"
      },
      "finishReason": "STOP"
    }
  ],
  "usageMetadata": {
    "promptTokenCount": 15,
    "candidatesTokenCount": 85,
    "totalTokenCount": 100,
    "thoughtsTokenCount": 280
  }
}
```

> `thought: true`인 파트가 사고 과정이며, 이후 파트가 최종 답변입니다.

### 8.7 OpenClaw에서의 통합 처리

OpenClaw은 다양한 프로바이더의 reasoning 모델을 통합적으로 처리합니다.

```
┌──────────────────────────────────────────────────────────────────────┐
│                  OpenClaw Reasoning Model 처리 흐름                    │
└──────────────────────────────────────────────────────────────────────┘

  models.json                   resolveModel()              API 호출
       │                            │                          │
       │  reasoning: true           │                          │
       ├───────────────────────────▶│                          │
       │                            │                          │
       │                     ┌──────┴──────┐                   │
       │                     │  Reasoning  │                   │
       │                     │  감지 분기   │                   │
       │                     └──────┬──────┘                   │
       │                            │                          │
       │              ┌─────────────┼─────────────┐            │
       │              ▼             ▼             ▼            │
       │        temperature    sampling      reasoning         │
       │        파라미터 제거    파라미터 조정   필드 추가          │
       │              │             │             │            │
       │              └─────────────┼─────────────┘            │
       │                            │                          │
       │                     ┌──────┴──────┐                   │
       │                     │  Provider별  │                   │
       │                     │  Request 변환│                   │
       │                     └──────┬──────┘                   │
       │                            │                          │
       │                            ├─────────────────────────▶│
       │                            │                          │
       │                     ┌──────┴──────┐                   │
       │                     │ Response     │                   │
       │                     │ 통합 파싱     │◀─────────────────┤
       │                     └──────┬──────┘                   │
       │                            │                          │
       │                            ▼                          │
       │                   reasoning_content                   │
       │                   또는 thinking block                  │
       │                   → 로그에 기록                        │
       │                   → content만 사용자에게 전달            │
```

#### 프로바이더별 파라미터 조정 요약

| 항목 | DeepSeek | OpenAI o-series | Anthropic | Google |
|------|----------|----------------|-----------|--------|
| `temperature` 제거 | Yes | Yes | 1 고정 | 자동 |
| `top_p` 제거 | Yes | Yes | 제거 | 자동 |
| Reasoning 제어 | 불가 | `reasoning_effort` | `budget_tokens` | `thinkingBudget` |
| 사고 과정 노출 | `reasoning_content` | 비공개 | `thinking` block | `thought` 파트 |
| 멀티턴 사고 포함 | `null` 처리 | 해당 없음 | **필수 포함** | 자동 제외 |
| 추가 토큰 비용 | `reasoning_tokens` | `reasoning_tokens` | `output_tokens`에 포함 | `thoughtsTokenCount` |

---

## 9. Prompt Caching (프롬프트 캐싱)

> LLM API 프로바이더가 제공하는 **Prefix Caching** 기능입니다.
> 반복되는 프롬프트 앞부분을 서버 측에 캐시하여 비용과 지연시간을 줄입니다.

### 9.1 Prompt Caching이란?

LLM API를 호출할 때, 매번 시스템 프롬프트와 이전 대화 기록을 전체 전송해야 합니다.
Prompt Caching은 이 **반복되는 앞부분(prefix)을 서버에 캐시**해서 재처리 비용을 줄이는 기능입니다.

```
첫 번째 호출:
┌──────────────────────────────────────────┐
│ 시스템 프롬프트 (5,000 토큰)              │ ← 전체 처리 (비쌈)
│ 이전 대화 기록 (10,000 토큰)              │ ← 전체 처리 (비쌈)
│ 새 사용자 메시지 (100 토큰)               │ ← 전체 처리
└──────────────────────────────────────────┘
  cacheWrite = 15,000  (캐시에 저장)
  cacheRead  = 0
  input      = 100

두 번째 호출:
┌──────────────────────────────────────────┐
│ 시스템 프롬프트 (5,000 토큰)              │ ← 캐시 히트! (저렴하게 처리)
│ 이전 대화 기록 (10,000 토큰)              │ ← 캐시 히트! (저렴하게 처리)
│ 새 사용자 메시지 (200 토큰)               │ ← 전체 처리
└──────────────────────────────────────────┘
  cacheWrite = 0
  cacheRead  = 15,000  (캐시에서 읽음)
  input      = 200
```

### 9.2 왜 Prefix Caching이 가능한가?

대화 메시지 배열은 **항상 뒤에 추가만 되고, 앞부분은 변하지 않습니다:**

```typescript
// 호출 1: messages 배열
[
  { role: "system",    content: "..." },         // 항상 같음
  { role: "user",      content: "안녕" },          // 새 메시지
]

// 호출 2: messages 배열 (앞부분 동일, 뒤에 추가)
[
  { role: "system",    content: "..." },         // ─┐
  { role: "user",      content: "안녕" },          // ─┤ 캐시 히트 (prefix 동일)
  { role: "assistant", content: "안녕하세요!" },    // ─┘
  { role: "user",      content: "날씨 알려줘" },    // ← 새로 추가된 부분만 처리
]

// 호출 3: messages 배열 (앞부분 동일, 뒤에 추가)
[
  { role: "system",    content: "..." },         // ─┐
  { role: "user",      content: "안녕" },          // ─┤
  { role: "assistant", content: "안녕하세요!" },    // ─┤ 캐시 히트 (prefix 동일)
  { role: "user",      content: "날씨 알려줘" },    // ─┤
  { role: "assistant", content: "서울은 맑습니다" }, // ─┘
  { role: "user",      content: "고마워" },         // ← 새로 추가된 부분만 처리
]
```

이전 대화는 수정되지 않고 append만 되므로, LLM 프로바이더는 **앞에서부터 일치하는 부분까지 캐시**합니다.

### 9.3 에이전트에서 특히 효과적인 이유

에이전트는 한 번의 사용자 메시지에 대해 **tool call 루프** 때문에 여러 번 LLM을 호출합니다:

```
User: "파일 읽고 수정해줘"

  → LLM 호출 1: tool_call(read_file)   ← 시스템+대화 전체 전송 (cacheWrite)
  → LLM 호출 2: tool_call(edit_file)   ← 호출1 + tool 결과 추가 (앞부분 cacheRead)
  → LLM 호출 3: "수정 완료했습니다"      ← 호출2 + tool 결과 추가 (앞부분 cacheRead)
```

사용자는 한 마디 했지만 LLM은 3번 호출되고, 매번 앞부분이 동일하므로 캐시 효과가 큽니다.

### 9.4 토큰 비용 필드

| 필드 | 의미 | 비용 |
|------|------|------|
| `input` | 캐시되지 않은 새 입력 토큰 | 정상 가격 |
| `cacheWrite` | 이번에 새로 캐시에 저장한 토큰 | 정상 가격보다 약간 비쌈 (Anthropic 기준 1.25배) |
| `cacheRead` | 캐시에서 읽어온 토큰 | 정상 가격보다 훨씬 쌈 (Anthropic 기준 0.1배) |
| `output` | LLM이 생성한 응답 토큰 | 정상 가격 |

### 9.5 프로바이더별 지원 현황

| 프로바이더 | 지원 여부 | 캐시 읽기 할인율 | 비고 |
|-----------|----------|----------------|------|
| Anthropic (Claude) | 지원 | 90% 할인 | `cache_creation_input_tokens`, `cache_read_input_tokens` |
| OpenAI (GPT) | 지원 | 50% 할인 | 자동 적용 |
| Google (Gemini) | 지원 | Context Caching | 명시적 API 호출 필요 |
| DeepSeek | 지원 | 90% 할인 | 자동 적용 |

### 9.6 models.json에서의 캐시 비용 설정

```json
{
  "id": "deepseek-chat",
  "cost": {
    "input": 0.14,
    "output": 0.28,
    "cacheRead": 0.014,
    "cacheWrite": 0.14
  }
}
```

- `cacheRead: 0.014`는 `input: 0.14`의 **10분의 1** → 90% 할인
- `cacheWrite: 0.14`는 `input: 0.14`와 동일 (DeepSeek의 경우)

이 비용 정보를 사용하여 `SessionStats.cost`가 계산됩니다.

---

## 10. Skill과 LLM API 호출 흐름

> Skill은 **재사용 가능한 프롬프트 조각**입니다.
> LLM API 호출 관점에서 Skill은 **시스템 프롬프트에 삽입되는 텍스트**에 불과합니다.

### 10.1 Skill이란?

Skill은 `.md` 파일에 작성된 프롬프트 조각으로, `/skill:name` 명령어로 호출합니다.
LLM 입장에서 Skill은 별도의 API나 프로토콜이 아니라, **시스템 프롬프트에 추가되는 텍스트**입니다.

```
~/.pi/agent/skills/            # 글로벌 스킬 (모든 프로젝트에서 사용)
  commit.md
  review/SKILL.md

~/my-project/.pi/skills/       # 프로젝트 스킬 (my-project에서만 사용)
  deploy.md
```

Skill 파일 예시 (`commit.md`):

```markdown
---
name: commit
description: 코드 변경사항을 커밋합니다
---

현재 변경사항을 분석하고 적절한 커밋 메시지를 작성해주세요.
git diff로 변경 내용을 확인한 후, Conventional Commits 형식으로 작성합니다.
```

### 10.2 Skill이 LLM API Request에 반영되는 과정

```
사용자: /skill:commit

         │
         ▼
┌─────────────────────────────────────────────────┐
│  1. Skill 로드                                    │
│     commit.md 파일 읽기                            │
│     → "현재 변경사항을 분석하고..."                   │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  2. 시스템 프롬프트 조립                             │
│     기존 시스템 프롬프트 + Skill 텍스트 삽입           │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  3. LLM API 호출                                  │
│     POST /v1/chat/completions                    │
│     messages: [                                  │
│       { role: "system", content: "..." },        │
│       { role: "user", content: "Skill 내용" }     │
│     ]                                            │
└─────────────────────────────────────────────────┘
```

### 10.3 Skill 적용 전후 Request Body 비교

**Skill 없이 일반 메시지를 보낸 경우:**

```json
{
  "model": "deepseek-chat",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant.\n\nAvailable tools: bash, view, edit ..."
    },
    {
      "role": "user",
      "content": "커밋 메시지 작성해줘"
    }
  ],
  "stream": true
}
```

**`/skill:commit` 으로 Skill을 호출한 경우:**

```json
{
  "model": "deepseek-chat",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant.\n\nAvailable tools: bash, view, edit ...\n\n## Available Skills\n- commit: 코드 변경사항을 커밋합니다"
    },
    {
      "role": "user",
      "content": "현재 변경사항을 분석하고 적절한 커밋 메시지를 작성해주세요.\ngit diff로 변경 내용을 확인한 후, Conventional Commits 형식으로 작성합니다."
    }
  ],
  "stream": true
}
```

핵심: **Skill의 description은 시스템 프롬프트에**, **Skill의 본문은 user 메시지에** 들어갑니다.

### 10.4 Skill이 시스템 프롬프트에 포함되는 구조

에이전트가 시작될 때, 등록된 모든 Skill의 목록이 시스템 프롬프트에 삽입됩니다:

```
┌─────────────────────────────────────────────────────────────┐
│                    시스템 프롬프트 구조                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  You are a helpful AI assistant.                            │
│                                                             │
│  ## Runtime Information                                     │
│  - Model: deepseek/deepseek-chat                           │
│  - OS: Darwin 24.6.0                                       │
│                                                             │
│  ## Available Tools                                         │
│  - bash: Execute shell commands                             │
│  - view: View file contents                                 │
│  - edit: Edit files                                         │
│                                                             │
│  ## Available Skills            ← Skill 목록이 여기 삽입됨    │
│  - commit: 코드 변경사항을 커밋합니다                          │
│  - review: 코드 리뷰를 수행합니다                              │
│  - deploy: 프로젝트를 배포합니다                               │
│                                                             │
│  ## Instructions                                            │
│  - Be helpful and concise                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 10.5 Skill과 Prompt Caching의 관계

Skill 목록은 시스템 프롬프트의 일부이므로, **Prompt Caching의 혜택을 받습니다:**

```
호출 1: [시스템 프롬프트(Skills 포함) + User:"커밋해줘"]
         ──────────────────────────── cacheWrite

호출 2: [시스템 프롬프트(Skills 포함) + User:"커밋해줘" + Asst:tool_call(bash) + Tool:결과 + User:"다음은?"]
         ────────────────────────────────────────────────────── cacheRead     ──── input
```

시스템 프롬프트에 포함된 Skill 목록은 세션 동안 변하지 않으므로, 첫 호출 이후 캐시에서 읽혀 비용이 절감됩니다.

### 10.6 정리: Skill은 LLM API 관점에서 무엇인가?

| 관점 | Skill의 정체 |
|------|-------------|
| 사용자 관점 | `/skill:name`으로 호출하는 명령어 |
| 에이전트 관점 | `.md` 파일에서 로드하는 프롬프트 조각 |
| **LLM API 관점** | **시스템 프롬프트에 삽입되는 텍스트 + user 메시지로 전송되는 본문** |
| 토큰 비용 관점 | 시스템 프롬프트의 Skill 목록은 Prompt Caching 혜택을 받음 |

Skill은 별도의 API 호출이나 특수한 프로토콜이 아닙니다. **기존 `messages` 배열의 `system`과 `user` 역할에 텍스트를 삽입하는 것**이 전부입니다.

---

## 부록: 디버깅 팁

### A. Cache Trace 활성화

```bash
export OPENCLAW_CACHE_TRACE=1
export OPENCLAW_CACHE_TRACE_FILE=~/openclaw-debug.jsonl
```

### B. Anthropic Payload Log 활성화

```bash
export OPENCLAW_ANTHROPIC_PAYLOAD_LOG=1
```

### C. 로그 파일 위치

```
~/.openclaw/state/logs/
├── cache-trace.jsonl      # 캐시 추적 로그
├── anthropic-payload.jsonl # API 페이로드 로그
└── agent-events.jsonl     # 에이전트 이벤트 로그
```

---

## 참고 자료

- [OpenClaw GitHub Repository](https://github.com/openclaw/openclaw)
- [DeepSeek API Documentation](https://api-docs.deepseek.com)
- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
- [Grammy (Telegram Bot Framework)](https://grammy.dev)
