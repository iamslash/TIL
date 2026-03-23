# Session Resume & Fork

## 개요

OpenCode의 long-term memory는 SQLite에 영구 저장되므로 **이전 세션을 다시
열거나(resume)**, 특정 시점에서 **분기(fork)**할 수 있다. 이를 통해 대화
이력을 재활용하고 다양한 접근을 실험할 수 있다.

## Session Resume (세션 이어가기)

### 원리

세션을 다시 열면 SQLite에서 해당 세션의 모든 메시지를 조회하여 LLM 컨텍스트에
주입한다. LLM 입장에서는 대화가 계속되는 것과 같다.

```
Day 1: 세션 ses_001 에서 작업
  msg_001 (user):    "API 서버 만들어줘"
  msg_002 (assistant): Write src/index.ts
  → opencode 종료

Day 2: 세션 ses_001 다시 열기
  → SQLite에서 msg_001, msg_002 조회
  → LLM 컨텍스트에 주입
  msg_003 (user):    "인증 추가해줘"
  → LLM은 이전 대화를 "기억"하고 있음
```

### 데이터 흐름

```
① 세션 목록에서 ses_001 선택
     │
     ▼
② SELECT * FROM session WHERE id = 'ses_001'
     │
     ▼
③ SELECT * FROM message WHERE session_id = 'ses_001'
   ORDER BY time_created DESC
     │
     ▼
④ SELECT * FROM part WHERE message_id IN (...)
     │
     ▼
⑤ filterCompacted() → toModelMessages()
     │
     ▼
⑥ LLM API 호출 (이전 대화 + 새 프롬프트)
```

Compaction이 발생했던 세션이라면 `filterCompacted()`에 의해 요약 이후의
메시지만 사용된다.

## Session Fork (세션 분기)

### 원리

기존 세션의 대화를 **특정 시점까지 복사**하여 새 세션을 만든다. 원본 세션은
그대로 유지되고, 분기된 세션에서 다른 방향으로 작업할 수 있다.

```
원본 세션 (ses_001):
  msg_001 (user):    "API 서버 만들어줘"
  msg_002 (assistant): Express로 구현
  msg_003 (user):    "DB 연결 추가해줘"
  msg_004 (assistant): PostgreSQL 연결
  msg_005 (user):    "Redis 캐시 추가해줘"  ← fork(messageID=msg_005): 이 메시지 직전까지 복사
  msg_006 (assistant): Redis 구현

Fork 세션 (ses_002): msg_004까지만 복사
  msg_101 (user):    "API 서버 만들어줘"       ← msg_001 복사
  msg_102 (assistant): Express로 구현          ← msg_002 복사
  msg_103 (user):    "DB 연결 추가해줘"        ← msg_003 복사
  msg_104 (assistant): PostgreSQL 연결         ← msg_004 복사
  msg_105 (user):    "MongoDB로 변경해줘"      ← 새로운 방향!
```

### Fork 코드

```typescript
// packages/opencode/src/session/index.ts — Session.fork()
export const fork = fn(
  z.object({
    sessionID: SessionID.zod,
    messageID: MessageID.zod.optional(),   // 이 메시지 직전까지 복사
  }),
  async (input) => {
    const original = await get(input.sessionID)
    const title = getForkedTitle(original.title)  // "My Session (fork #1)"
    const session = await createNext({
      directory: Instance.directory,
      workspaceID: original.workspaceID,
      title,
    })

    // 원본 세션의 메시지를 순서대로 복사
    const msgs = await messages({ sessionID: input.sessionID })
    const idMap = new Map<string, MessageID>()

    for (const msg of msgs) {
      // messageID가 지정되면 그 메시지부터는 복사하지 않음 (exclusive)
      // ID가 시간순 정렬 가능하므로 >= 비교로 cutoff 판단
      if (input.messageID && msg.info.id >= input.messageID) break

      const newID = MessageID.ascending()
      idMap.set(msg.info.id, newID)

      // 메시지 복사 (새 ID, 새 세션 ID)
      const cloned = await updateMessage({
        ...msg.info,
        sessionID: session.id,
        id: newID,
      })

      // 파트도 복사
      for (const part of msg.parts) {
        await updatePart({
          ...part,
          id: PartID.ascending(),
          messageID: cloned.id,
          sessionID: session.id,
        })
      }
    }

    return session
  },
)
```

### Fork 후 DB 상태

```
session 테이블:
┌─────────┬────────────┬──────────────────────────────┐
│ id      │ project_id │ title                        │
├─────────┼────────────┼──────────────────────────────┤
│ ses_001 │ a1b2c3d4   │ Express API 서버             │  ← 원본 (변경 없음)
│ ses_002 │ a1b2c3d4   │ Express API 서버 (fork #1)   │  ← 분기
└─────────┴────────────┴──────────────────────────────┘

message 테이블 (ses_002):
┌─────────┬────────────┬───────────┬──────────────────────────┐
│ id      │ session_id │ role      │ 내용                     │
├─────────┼────────────┼───────────┼──────────────────────────┤
│ msg_101 │ ses_002    │ user      │ "API 서버 만들어줘"       │  ← 복사
│ msg_102 │ ses_002    │ assistant │ Express로 구현            │  ← 복사
│ msg_103 │ ses_002    │ user      │ "DB 연결 추가해줘"        │  ← 복사
│ msg_104 │ ses_002    │ assistant │ PostgreSQL 연결           │  ← 복사
│ msg_105 │ ses_002    │ user      │ "MongoDB로 변경해줘"      │  ← 새 메시지
└─────────┴────────────┴───────────┴──────────────────────────┘
```

### Parent-Child 관계

세션은 `parent_id`로 부모-자식 관계를 가질 수 있다.
**참고: `fork()`는 `parent_id`를 설정하지 않는다.** `parent_id`는 `Session.create()`에서
`parentID` 옵션을 명시적으로 전달할 때만 설정되며, 주로 서브태스크(child session)
생성 시 사용된다. fork는 독립적인 새 세션을 만들 뿐이다.

```sql
-- parent_id가 설정되는 경우: 서브태스크로 생성된 세션
SELECT id, parent_id, title FROM session WHERE project_id = 'a1b2c3d4';

┌─────────┬───────────┬──────────────────────────────┐
│ id      │ parent_id │ title                        │
├─────────┼───────────┼──────────────────────────────┤
│ ses_001 │ NULL      │ Express API 서버             │
│ ses_002 │ ses_001   │ DB 설계 (child session)      │  ← create(parentID)로 생성
│ ses_003 │ NULL      │ Express API 서버 (fork #1)   │  ← fork로 생성 (parent_id 없음)
└─────────┴───────────┴──────────────────────────────┘

주의: ses_002는 fork가 아니라 Session.create({parentID: "ses_001"})로 생성된
서브태스크이다. fork(ses_003)는 parent_id가 NULL이다.
```

## Fork 제목 규칙

```typescript
// packages/opencode/src/session/index.ts
function getForkedTitle(title: string): string {
  const match = title.match(/^(.+) \(fork #(\d+)\)$/)
  if (match) {
    const base = match[1]
    const num = parseInt(match[2], 10)
    return `${base} (fork #${num + 1})`     // "My Session (fork #2)"
  }
  return `${title} (fork #1)`               // "My Session (fork #1)"
}
```

## 사용 사례

### Resume

| 상황 | 설명 |
|------|------|
| 다음 날 이어서 작업 | 세션을 다시 열면 어제 대화가 복원됨 |
| 긴 작업을 여러 번에 나눠서 | 같은 세션에서 계속 진행 |
| 에러 후 재시도 | 같은 세션에서 다시 프롬프트 |

### Fork

| 상황 | 설명 |
|------|------|
| 다른 접근 시도 | "PostgreSQL 대신 MongoDB로 해보자" |
| 실험적 변경 | 원본을 보존하고 분기에서 실험 |
| A/B 비교 | 같은 시점에서 두 가지 방향으로 진행 |

## Long-term Memory 전체 아키텍처 요약

```
┌────────────────────────────────────────────────────────────┐
│                OpenCode Long-term Memory                    │
│                                                             │
│  Layer 1: SQLite DB                                        │
│    세션, 메시지, 파트 영구 저장                               │
│    → 세션 내 기억                                           │
│                                                             │
│  Layer 2: Compaction                                       │
│    Prune (tool output 제거) + Summarize (대화 요약)          │
│    → 컨텍스트 윈도우 한계 극복                               │
│                                                             │
│  Layer 3: Instructions (AGENTS.md / CLAUDE.md)             │
│    프로젝트/글로벌 지시사항                                   │
│    → 세션 간 지속 기억 (매 호출마다 system prompt에 주입)     │
│                                                             │
│  Layer 4: Session Resume & Fork                            │
│    이전 세션 이어가기, 특정 시점에서 분기                     │
│    → 대화 이력 재활용 및 실험                                │
│                                                             │
│  Layer 5: Project Identity                                 │
│    git 최초 commit hash로 프로젝트 식별                      │
│    → 같은 repo = 같은 세션 목록                              │
└────────────────────────────────────────────────────────────┘
```
