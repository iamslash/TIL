# Compaction (대화 압축)

## 개요

LLM은 컨텍스트 윈도우(context window)라는 입력 길이 제한이 있다. 대화가
길어지면 이전 메시지 + 새 메시지의 토큰 합이 이 제한을 초과한다. Compaction은
**오래된 대화를 요약하여 토큰 수를 줄이면서 핵심 정보를 보존**하는 메커니즘이다.

## 트리거 조건

### 자동 Compaction (isOverflow)

매 LLM 응답 후 사용된 토큰 수를 확인한다. 모델의 컨텍스트 한계에 가까워지면
자동으로 compaction을 트리거한다.

```typescript
// packages/opencode/src/session/compaction.ts
const COMPACTION_BUFFER = 20_000

export async function isOverflow(input) {
  const context = input.model.limit.context     // 예: 128,000
  const count = input.tokens.total              // 현재 사용한 토큰
  const reserved = Math.min(COMPACTION_BUFFER, maxOutputTokens(input.model))
  const usable = context - maxOutputTokens(input.model)
  return count >= usable
}
```

예: GPT-4o (128K context, 16K max output)

```
reserved = min(COMPACTION_BUFFER, maxOutputTokens) = min(20,000, 16,000) = 16,000
usable = context - maxOutputTokens = 128,000 - 16,000 = 112,000
현재 토큰이 112,000 이상이면 → compaction 트리거
```

> `COMPACTION_BUFFER`(20,000)는 `reserved` 계산에 사용되며, `model.limit.input`이
> 설정된 경우 `input - reserved`로 계산된다. 위 예시는 `limit.input`이 없는
> 일반적인 경우이다.

### 수동 Compaction

사용자가 `/compact` 명령으로 직접 트리거할 수도 있다.

## Compaction의 두 단계

Compaction이 트리거되면 **Prune과 Summarize가 항상 순차적으로 실행**된다.
Prune만으로 토큰이 충분히 줄어들더라도 Summarize를 건너뛰지 않는다.
이는 다음 프롬프트에서 다시 초과하는 것을 미리 방지하기 위함이다.

```
Step 1: Prune (가지치기)
  → 오래된 tool call의 output을 제거하여 즉시 토큰 절약

Step 2: Summarize (요약)
  → 항상 Prune 직후 실행됨
  → LLM에게 대화 전체를 요약하게 하여 새 시작점 생성
```

---

## Step 1: Prune (가지치기)

오래된 tool call의 output(파일 내용, 명령 실행 결과 등)을
**`[Old tool result content cleared]` 플레이스홀더로 교체**한다.
최근 대화의 tool output은 보존한다.

### 동작 원리

```typescript
// packages/opencode/src/session/compaction.ts
export const PRUNE_MINIMUM = 20_000    // 최소 20K 토큰 이상 절약 가능할 때만
export const PRUNE_PROTECT = 40_000    // 최근 40K 토큰의 tool output은 보호

export async function prune(input) {
  // 메시지를 역순으로 순회
  for (let msgIndex = msgs.length - 1; msgIndex >= 0; msgIndex--) {
    // 최근 2턴은 건드리지 않음
    if (turns < 2) continue

    // tool output의 토큰 수를 누적
    total += Token.estimate(part.state.output)

    // 최근 40K 토큰은 보호, 그 이전의 tool output을 prune 대상에 추가
    if (total > PRUNE_PROTECT) {
      toPrune.push(part)
    }
  }

  // 절약 가능 토큰이 20K 이상일 때만 실행
  if (pruned > PRUNE_MINIMUM) {
    for (const part of toPrune) {
      part.state.time.compacted = Date.now()   // compacted 마킹
      await Session.updatePart(part)
    }
  }
}
```

### Prune 후 효과

```
Before prune:
  tool(Read src/index.ts) → output: "import express... (5,000 tokens)"
  tool(Bash npm test)     → output: "PASS 42 tests... (3,000 tokens)"

After prune:
  tool(Read src/index.ts) → output: "[Old tool result content cleared]"
  tool(Bash npm test)     → output: "[Old tool result content cleared]"
```

보호 대상:
- 최근 40K 토큰 이내의 tool output
- `skill` 도구의 output (항상 보호)

---

## Step 2: Summarize (요약)

Prune 직후 항상 실행된다. LLM에게 **전체 대화를 요약**하게 하여 요약 결과가
새로운 대화의 시작점이 된다.

### 요약 프롬프트

```
Provide a detailed prompt for continuing our conversation above.
Focus on information that would be helpful for continuing the conversation,
including what we did, what we're doing, which files we're working on,
and what we're going to do next.

When constructing the summary, try to stick to this template:
---
## Goal
[What goal(s) is the user trying to accomplish?]

## Instructions
- [What important instructions did the user give you that are relevant]
- [If there is a plan or spec, include information about it]

## Discoveries
[What notable things were learned during this conversation]

## Accomplished
[What work has been completed, what is still in progress, what is left?]

## Relevant files / directories
[Structured list of relevant files that have been read, edited, or created]
---
```

### 요약 과정

```typescript
// packages/opencode/src/session/compaction.ts — SessionCompaction.process()
const result = await processor.process({
  messages: [
    ...MessageV2.toModelMessages(msgs, model, { stripMedia: true }),
    {
      role: "user",
      content: [{ type: "text", text: promptText }],
    },
  ],
  tools: {},              // 도구 없음 (텍스트 응답만)
  model,
})
```

LLM에게 전체 대화 이력 + 요약 프롬프트를 보내면, LLM이 구조화된 요약을
반환한다.

### 요약 결과 저장 흐름

```
① user message (compaction 마커) INSERT
   → part: {type: "compaction", auto: true}

② assistant message (요약 본문) INSERT
   → part: {type: "text", text: "## Goal\n사용자는 Express API에..."}
   → summary: true, finish: "stop" 플래그 설정
   (filterCompacted가 summary + finish 둘 다 확인함)
```

---

## filterCompacted 함수

다음 LLM 호출 시 **compaction 이후의 메시지만** 가져온다.

```typescript
// packages/opencode/src/session/message-v2.ts — MessageV2.filterCompacted()
// 참고: compaction part가 있는 user message는 toModelMessages()에서
// "What did we do so far?"라는 합성 텍스트로 변환된다.
// (message-v2.ts의 toModelMessages 내부에서 compaction type을 감지하여 처리)
export async function filterCompacted(stream) {
  const result = []
  const completed = new Set()

  for await (const msg of stream) {
    result.push(msg)

    // compaction 마커가 있고, 그에 대한 요약이 완료된 경우 → 여기서 중단
    if (msg.info.role === "user" &&
        completed.has(msg.info.id) &&
        msg.parts.some(part => part.type === "compaction"))
      break

    // 요약 메시지가 정상 완료된 경우 → parentID를 완료 셋에 추가
    if (msg.info.role === "assistant" && msg.info.summary && msg.info.finish && !msg.info.error)
      completed.add(msg.info.parentID)
  }

  result.reverse()
  return result
}
```

이 함수가 하는 일: 메시지를 최신부터 역순으로 읽다가, **요약이 완료된
compaction 마커를 만나면 거기서 중단**한다. 그 이전의 오래된 메시지는 버린다.
