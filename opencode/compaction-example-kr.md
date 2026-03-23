# Compaction 동작 예시

## 시나리오

GPT-4o (context: 128K)를 사용하여 긴 대화를 한다. 20번의 대화 끝에
토큰이 112K를 초과하여 compaction이 자동 트리거된다.

---

## 대화 진행 (Compaction 전)

```
msg_001 (user):    "Express API 서버를 만들어줘"
msg_002 (assistant): Read package.json → text "Express 프로젝트를 만들겠습니다"
msg_003 (user):    "TypeScript로 해줘"
msg_004 (assistant): Write tsconfig.json → Write src/index.ts → text "TypeScript 설정..."
msg_005 (user):    "DB 연결 추가해줘"
msg_006 (assistant): Read src/index.ts → Write src/db.ts → text "PostgreSQL 연결..."
  ...
  (20번의 대화, 토큰 누적)
  ...
msg_039 (user):    "테스트 추가해줘"
msg_040 (assistant): Write src/test/api.test.ts → Bash "npm test" → text "테스트 통과"
```

### DB 상태 (Compaction 전)

```
message 테이블: 40행 (user 20 + assistant 20)
part 테이블:    ~120행 (text + tool + step-finish)
총 토큰:       ~115,000 (임계값 112,000 초과!)
```

---

## Compaction 트리거

msg_040의 응답이 끝난 후 `isOverflow()` 검사:

```
total tokens: 115,000
usable limit: 128,000 - 16,000 = 112,000
115,000 >= 112,000 → compaction 트리거!
```

---

## Phase 1: Prune (가지치기)

오래된 tool call의 output을 제거한다.

### Before Prune

```
part 테이블 (tool parts만 표시):
┌─────────┬────────┬─────────────────┬──────────────────────────────┐
│ id      │ msg_id │ tool            │ output (토큰 수)             │
├─────────┼────────┼─────────────────┼──────────────────────────────┤
│ prt_002 │ msg_002│ Read pkg.json   │ "{name: my-app...}" (500)    │ ← 오래됨
│ prt_005 │ msg_004│ Write tsconfig  │ "File written" (50)          │ ← 오래됨
│ prt_006 │ msg_004│ Write index.ts  │ "File written" (50)          │ ← 오래됨
│ prt_010 │ msg_006│ Read index.ts   │ "import express..." (3,000)  │ ← 오래됨
│ prt_011 │ msg_006│ Write db.ts     │ "File written" (50)          │ ← 오래됨
│ ...     │ ...    │ ...             │ ... (총 ~60,000 토큰)        │
│ prt_098 │ msg_038│ Write routes.ts │ "File written" (50)          │ ← 보호됨
│ prt_100 │ msg_040│ Write test.ts   │ "File written" (50)          │ ← 보호됨
│ prt_101 │ msg_040│ Bash npm test   │ "PASS 42 tests..." (2,000)  │ ← 보호됨
└─────────┴────────┴─────────────────┴──────────────────────────────┘
```

### Prune 로직

```
역순으로 tool output 토큰을 누적:
  prt_101: 2,000    (누적: 2,000)    → 보호 (< 40,000)
  prt_100: 50       (누적: 2,050)    → 보호
  prt_098: 50       (누적: 2,100)    → 보호
  ...
  prt_050: 1,500    (누적: 39,500)   → 보호
  prt_049: 2,000    (누적: 41,500)   → 41,500 > 40,000 → prune!
  prt_048: 3,000    (누적: 44,500)   → prune!
  ...
  prt_002: 500      (누적: 62,000)   → prune!
```

### After Prune

```sql
-- prt_002의 output을 "[Old tool result content cleared]"로 교체
UPDATE part SET data = json_set(data,
  '$.state.output', '[Old tool result content cleared]',
  '$.state.time.compacted', 1711271000000)
WHERE id = 'prt_002';

-- prt_010도 동일하게 처리
UPDATE part SET data = json_set(data,
  '$.state.output', '[Old tool result content cleared]',
  '$.state.time.compacted', 1711271000000)
WHERE id = 'prt_010';

-- ... 나머지 prune 대상도 동일
```

```
Prune 결과:
  제거된 토큰: ~22,000 (PRUNE_MINIMUM 20,000 초과 → 실행)
  남은 토큰:   ~93,000

  93,000 < 112,000 이므로 당장은 OK.
  하지만 다음 프롬프트에서 다시 초과할 가능성이 높으므로
  OpenCode는 Prune 직후 Summarize도 함께 진행한다.
  (Prune과 Summarize는 항상 순차 실행되며, Prune가 충분해도
   Summarize를 건너뛰지 않는다.)
```

---

## Phase 2: Summarize (요약)

### ① Compaction User Message 생성

```sql
INSERT INTO message (id, session_id, time_created, time_updated, data)
VALUES ('msg_041', 'ses_001', 1711271001000, 1711271001000,
        '{"role":"user","agent":"coder",
          "model":{"providerID":"openai","modelID":"gpt-4o"}}');

INSERT INTO part (id, message_id, session_id, time_created, time_updated, data)
VALUES ('prt_110', 'msg_041', 'ses_001', 1711271001000, 1711271001000,
        '{"type":"compaction","auto":true}');
```

### ② LLM에게 요약 요청

```
POST https://api.openai.com/v1/chat/completions

{
  "model": "gpt-4o",
  "messages": [
    // 전체 대화 이력 (media 제거, prune된 tool output 포함)
    {"role": "user", "content": "Express API 서버를 만들어줘"},
    {"role": "assistant", "content": "Express 프로젝트를 만들겠습니다"},
    {"role": "user", "content": "TypeScript로 해줘"},
    ...
    {"role": "assistant", "content": "테스트 통과"},

    // 요약 프롬프트
    {"role": "user", "content": "Provide a detailed prompt for continuing
     our conversation above...\n## Goal\n## Instructions\n..."}
  ]
}
```

### ③ LLM의 요약 응답

```markdown
## Goal
Express.js + TypeScript 기반 REST API 서버를 구축하고 테스트까지 완성하는 것.

## Instructions
- TypeScript 사용 필수
- PostgreSQL 데이터베이스 연결
- Jest 기반 테스트 작성

## Discoveries
- package.json에 기존 express 의존성이 있었음
- PostgreSQL 연결 시 connection pool 크기를 5로 설정
- API 라우트는 /api/v1 접두사 사용

## Accomplished
- [완료] TypeScript 설정 (tsconfig.json)
- [완료] Express 서버 기본 구조 (src/index.ts)
- [완료] PostgreSQL 연결 모듈 (src/db.ts)
- [완료] API 라우트 (src/routes.ts)
- [완료] Jest 테스트 (src/test/api.test.ts) - 42개 테스트 통과
- [미완료] 에러 핸들링 미들웨어
- [미완료] 인증/인가

## Relevant files / directories
- src/index.ts (서버 진입점)
- src/db.ts (PostgreSQL 연결)
- src/routes.ts (API 라우트)
- src/test/api.test.ts (테스트)
- tsconfig.json
- package.json
```

### ④ 요약 결과 저장

```sql
INSERT INTO message (id, session_id, time_created, time_updated, data)
VALUES ('msg_042', 'ses_001', 1711271002000, 1711271002000,
        '{"role":"assistant","parentID":"msg_041",
          "providerID":"openai","modelID":"gpt-4o",
          "agent":"compaction","mode":"compaction",
          "summary":true,
          "finish":"stop",
          "cost":0.003,
          "tokens":{"input":90000,"output":500,...}}');

INSERT INTO part (id, message_id, session_id, time_created, time_updated, data)
VALUES ('prt_111', 'msg_042', 'ses_001', 1711271002000, 1711271002000,
        '{"type":"text",
          "text":"## Goal\nExpress.js + TypeScript 기반..."}');
```

---

## Compaction 후 DB 상태

```
message 테이블:
┌─────────┬────────────┬───────────┬──────────────┬─────────┐
│ id      │ session_id │ role      │ agent        │ summary │
├─────────┼────────────┼───────────┼──────────────┼─────────┤
│ msg_001 │ ses_001    │ user      │ coder        │         │  ← 오래된 메시지
│ msg_002 │ ses_001    │ assistant │ coder        │         │     (DB에 남아있지만
│ ...     │ ...        │ ...       │ ...          │         │      다음 호출에서
│ msg_040 │ ses_001    │ assistant │ coder        │         │      사용되지 않음)
│ msg_041 │ ses_001    │ user      │ coder        │         │  ← compaction 마커
│ msg_042 │ ses_001    │ assistant │ compaction   │ true    │  ← 요약 (새 시작점)
└─────────┴────────────┴───────────┴──────────────┴─────────┘
```

---

## Compaction 후 다음 프롬프트

사용자: **"에러 핸들링 미들웨어를 추가해줘"**

### filterCompacted() 실행

```
메시지를 최신부터 역순으로 읽음:
  msg_043 (user, 새 메시지) → result에 추가
  msg_042 (assistant, summary=true, finish=ok) → result에 추가
         → completed.add("msg_041")
  msg_041 (user, compaction 마커) → completed에 있음 → 여기서 중단!

msg_001 ~ msg_040은 읽지 않음 (버림)
```

### OpenAI API에 보내는 messages

```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "You are an AI coding assistant..."},

    // msg_041: compaction 마커 → "What did we do so far?"로 변환
    {"role": "user", "content": "What did we do so far?"},

    // msg_042: 요약
    {"role": "assistant", "content": "## Goal\nExpress.js + TypeScript...
     ## Accomplished\n- [완료] TypeScript 설정...
     ## Relevant files\n- src/index.ts..."},

    // msg_043: 새 프롬프트
    {"role": "user", "content": "에러 핸들링 미들웨어를 추가해줘"}
  ]
}
```

### 토큰 비교

```
Compaction 전: ~115,000 토큰 (40개 메시지 전체)
Compaction 후:    ~2,000 토큰 (요약 + 새 메시지만)

절약: ~113,000 토큰 (98% 감소)
```

LLM은 요약을 통해 이전 대화의 핵심을 알고 있으므로,
"에러 핸들링 미들웨어"를 어디에 어떻게 추가할지 정확히 파악할 수 있다.

---

## 전체 흐름 요약

```
대화가 짧을 때:
  [msg_001] [msg_002] [msg_003] [msg_004] → 전부 LLM에 전달

대화가 길어지면 (토큰 초과):
  Phase 1 - Prune:
    오래된 tool output 제거 → 즉시 토큰 절약

  Phase 2 - Summarize:
    [msg_001~040] → LLM이 요약 → [msg_042: 요약문]
                                         │
                                         ▼
다음 호출:                          ┌──────────────────────────┐
  [msg_041: compaction marker]      │ 요약 (2,000 토큰)        │
  [msg_042: 요약문]        ← 여기서부터만 사용
  [msg_043: 새 프롬프트]
                            총 ~2,000 토큰 (115,000 → 2,000)
```

**Compaction = 오래된 대화를 "기억의 요약"으로 대체하여 LLM의 컨텍스트 제한을
극복하는 메커니즘**이다.
