# OpenCode Long-term Memory 시스템

## 개요

OpenCode는 대화 세션, 메시지, 프로젝트 정보 등을 **로컬 SQLite 데이터베이스**에
저장하여 장기 기억(long-term memory)을 관리한다. 이전에는 JSON 파일 기반
스토리지를 사용했으나, 현재는 SQLite + drizzle-orm으로 마이그레이션되었다.

## 스토리지 이중 구조

OpenCode는 두 가지 스토리지 시스템을 가지고 있다.

```
┌──────────────────────────────────────────────┐
│              OpenCode Storage                 │
│                                               │
│  ┌─────────────────────┐  ┌────────────────┐  │
│  │ SQLite Database     │  │ JSON File      │  │
│  │ (Primary, 현재)      │  │ Storage        │  │
│  │                     │  │ (Legacy)       │  │
│  │ - session           │  │                │  │
│  │ - message           │  │ - project      │  │
│  │ - part              │  │ - session      │  │
│  │ - project           │  │ - message      │  │
│  │ - todo              │  │ - part         │  │
│  │ - permission        │  │                │  │
│  │ - account           │  │ key-value 방식  │  │
│  │ - account_state     │  │ 파일 경로 = key │  │
│  │ - workspace         │  │                │  │
│  │ - session_share     │  │                │  │
│  └─────────────────────┘  └────────────────┘  │
└──────────────────────────────────────────────┘
```

---

## 1. SQLite Database (Primary Storage)

### DB 파일 경로

```
~/.local/share/opencode/opencode.db
```

채널(beta, latest 등)에 따라 파일명이 달라질 수 있다:

```typescript
// packages/opencode/src/storage/db.ts
if (["latest", "beta"].includes(channel) || Flag.OPENCODE_DISABLE_CHANNEL_DB)
  return path.join(Global.Path.data, "opencode.db")
const safe = channel.replace(/[^a-zA-Z0-9._-]/g, "-")
return path.join(Global.Path.data, `opencode-${safe}.db`)
```

### DB 초기화

SQLite는 Bun 환경에서는 `bun:sqlite`, Node.js 환경에서는 `node:sqlite`를
사용한다. drizzle-orm으로 추상화되어 있다.

```typescript
// packages/opencode/src/storage/db.bun.ts
import { Database } from "bun:sqlite"
import { drizzle } from "drizzle-orm/bun-sqlite"

export function init(path: string) {
  const sqlite = new Database(path, { create: true })
  const db = drizzle({ client: sqlite })
  return db
}
```

초기화 시 성능 최적화 PRAGMA를 설정한다:

```typescript
// packages/opencode/src/storage/db.ts
db.run("PRAGMA journal_mode = WAL")
db.run("PRAGMA synchronous = NORMAL")
db.run("PRAGMA busy_timeout = 5000")
db.run("PRAGMA cache_size = -64000")
db.run("PRAGMA foreign_keys = ON")
db.run("PRAGMA wal_checkpoint(PASSIVE)")
```

| PRAGMA | 설명 |
|--------|------|
| `journal_mode = WAL` | Write-Ahead Logging. 읽기/쓰기 동시성 향상 |
| `synchronous = NORMAL` | WAL 모드에서 안전하면서 빠른 설정 |
| `busy_timeout = 5000` | 락 대기 시간 5초 |
| `cache_size = -64000` | 64MB 캐시 |
| `foreign_keys = ON` | 외래 키 제약 조건 활성화 |

### 스키마 마이그레이션

drizzle-orm의 마이그레이션 시스템을 사용한다. `migration/` 디렉토리에
타임스탬프 기반 SQL 파일이 있다.

```
packages/opencode/migration/
├── 20260127222353_familiar_lady_ursula/migration.sql    # 초기 스키마
├── 20260211171708_add_project_commands/migration.sql
├── 20260213144116_wakeful_the_professor/migration.sql
├── 20260225215848_workspace/migration.sql
├── 20260227213759_add_session_workspace_id/migration.sql
├── 20260228203230_blue_harpoon/migration.sql
├── 20260303231226_add_workspace_fields/migration.sql
├── 20260309230000_move_org_to_state/migration.sql
└── 20260312043431_session_message_cursor/migration.sql
```

---

## 2. 데이터베이스 스키마

### 공통: Timestamps

대부분의 테이블에 공통으로 사용되는 타임스탬프 컬럼이다.
(`account_state`, `workspace` 등 일부 테이블은 제외)

```typescript
// packages/opencode/src/storage/schema.sql.ts
export const Timestamps = {
  time_created: integer().notNull().$default(() => Date.now()),
  time_updated: integer().notNull().$onUpdate(() => Date.now()),
}
```

### ER 다이어그램

```
┌──────────────┐
│   project    │
│──────────────│
│ id (PK)      │
│ worktree     │
│ vcs          │
│ name         │
│ icon_url     │
│ icon_color   │
│ sandboxes    │──JSON
│ commands     │──JSON
│ time_*       │
└──────┬───────┘
       │ 1:N
       ▼
┌──────────────┐     ┌──────────────┐
│   session    │     │  workspace   │
│──────────────│     │──────────────│
│ id (PK)      │     │ id (PK)      │
│ project_id   │──FK │ type         │
│ workspace_id │──FK─│ branch       │
│ parent_id    │──FK(self) │ name         │
│ slug         │     │ directory    │
│ directory    │     │ extra        │──JSON
│ title        │     │ project_id   │──FK
│ version      │     └──────────────┘
│ share_url    │
│ summary_*    │
│ revert       │──JSON
│ permission   │──JSON
│ time_*       │
└──────┬───────┘
       │ 1:N
       ▼
┌──────────────┐
│   message    │
│──────────────│
│ id (PK)      │
│ session_id   │──FK
│ data         │──JSON (role, content 등)
│ time_*       │
└──────┬───────┘
       │ 1:N
       ▼
┌──────────────┐
│    part      │
│──────────────│
│ id (PK)      │
│ message_id   │──FK
│ session_id   │
│ data         │──JSON (tool call, text 등)
│ time_*       │
└──────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    todo      │     │  permission  │     │session_share │
│──────────────│     │──────────────│     │──────────────│
│ session_id   │──FK │ project_id   │──FK │ session_id   │──FK
│ content      │     │ data         │──JSON│ id           │
│ status       │     │ time_*       │     │ secret       │
│ priority     │     └──────────────┘     │ url          │
│ position     │                          │ time_*       │
│ time_*       │                          └──────────────┘
└──────────────┘

┌──────────────┐     ┌──────────────┐
│   account    │     │account_state │
│──────────────│     │──────────────│
│ id (PK)      │     │ id (PK)      │
│ email        │     │ active_      │
│ url          │     │ account_id   │──FK
│ access_token │     │ active_org_id│
│ refresh_token│     └──────────────┘
│ token_expiry │
│ time_*       │
└──────────────┘
```

### 테이블별 상세

#### project

프로젝트(git 저장소) 정보를 저장한다.

```sql
CREATE TABLE `project` (
  `id` text PRIMARY KEY,          -- git 초기 commit hash
  `worktree` text NOT NULL,       -- 작업 디렉토리 경로
  `vcs` text,                     -- 버전 관리 시스템 (git)
  `name` text,                    -- 프로젝트 이름
  `icon_url` text,
  `icon_color` text,
  `time_created` integer NOT NULL,
  `time_updated` integer NOT NULL,
  `time_initialized` integer,
  `sandboxes` text NOT NULL,      -- JSON: 샌드박스 ID 배열
  `commands` text                 -- JSON: { start?: string }
);
```

`id`는 git repo의 **최초 commit hash**로 결정된다. 이렇게 하면 같은 repo를
다른 경로에서 열어도 동일한 프로젝트로 인식한다.

#### session

대화 세션을 저장한다. 하나의 프로젝트에 여러 세션이 있을 수 있다.

```sql
CREATE TABLE `session` (
  `id` text PRIMARY KEY,
  `project_id` text NOT NULL,       -- FK → project
  `workspace_id` text,              -- FK → workspace
  `parent_id` text,                 -- FK → session (child session/subtask용. fork는 설정하지 않음)
  `slug` text NOT NULL,             -- URL-friendly 이름
  `directory` text NOT NULL,        -- 세션 작업 디렉토리
  `title` text NOT NULL,            -- 세션 제목
  `version` text NOT NULL,          -- 프로토콜 버전
  `share_url` text,                 -- 공유 URL
  `summary_additions` integer,      -- 코드 변경 요약: 추가 행
  `summary_deletions` integer,      -- 코드 변경 요약: 삭제 행
  `summary_files` integer,          -- 변경된 파일 수
  `summary_diffs` text,             -- JSON: 파일별 diff 상세
  `revert` text,                    -- JSON: 되돌리기 정보
  `permission` text,                -- JSON: 권한 규칙
  `time_created` integer NOT NULL,
  `time_updated` integer NOT NULL,
  `time_compacting` integer,        -- 컨텍스트 압축 시점
  `time_archived` integer           -- 보관 시점
);

CREATE INDEX `session_project_idx` ON `session` (`project_id`);
CREATE INDEX `session_workspace_idx` ON `session` (`workspace_id`);
CREATE INDEX `session_parent_idx` ON `session` (`parent_id`);
```

#### message

세션 내의 개별 메시지를 저장한다. `data` 컬럼에 역할(user/assistant),
내용 등이 JSON으로 들어간다.

```sql
CREATE TABLE `message` (
  `id` text PRIMARY KEY,
  `session_id` text NOT NULL,       -- FK → session (CASCADE)
  `time_created` integer NOT NULL,
  `time_updated` integer NOT NULL,
  `data` text NOT NULL              -- JSON: { role, content, ... }
);

CREATE INDEX `message_session_time_created_id_idx`
  ON `message` (`session_id`, `time_created`, `id`);
```

#### part

메시지의 하위 구성 요소를 저장한다. 하나의 메시지가 여러 part를 가질 수
있다 (텍스트, tool call, tool result 등).

```sql
CREATE TABLE `part` (
  `id` text PRIMARY KEY,
  `message_id` text NOT NULL,       -- FK → message (CASCADE)
  `session_id` text NOT NULL,
  `time_created` integer NOT NULL,
  `time_updated` integer NOT NULL,
  `data` text NOT NULL              -- JSON: { type, content, ... }
);

CREATE INDEX `part_message_id_id_idx` ON `part` (`message_id`, `id`);
CREATE INDEX `part_session_idx` ON `part` (`session_id`);
```

#### todo

세션 내의 할 일 목록을 저장한다.

```sql
CREATE TABLE `todo` (
  `session_id` text NOT NULL,       -- FK → session (CASCADE)
  `content` text NOT NULL,
  `status` text NOT NULL,           -- pending, completed 등
  `priority` text NOT NULL,
  `position` integer NOT NULL,
  `time_created` integer NOT NULL,
  `time_updated` integer NOT NULL,
  PRIMARY KEY (`session_id`, `position`)
);
```

#### permission

프로젝트별 권한 규칙을 저장한다.

```sql
CREATE TABLE `permission` (
  `project_id` text PRIMARY KEY,    -- FK → project (CASCADE)
  `time_created` integer NOT NULL,
  `time_updated` integer NOT NULL,
  `data` text NOT NULL              -- JSON: 권한 규칙 배열
);
```

#### account / account_state

사용자 계정과 현재 활성 계정 상태를 저장한다.

```sql
CREATE TABLE `account` (
  `id` text PRIMARY KEY,
  `email` text NOT NULL,
  `url` text NOT NULL,
  `access_token` text NOT NULL,
  `refresh_token` text NOT NULL,
  `token_expiry` integer,
  `time_created` integer NOT NULL,
  `time_updated` integer NOT NULL
);

CREATE TABLE `account_state` (
  `id` integer PRIMARY KEY,
  `active_account_id` text,         -- FK → account (SET NULL)
  `active_org_id` text
);
```

#### workspace

프로젝트 내의 작업 공간(branch, 디렉토리 등)을 나타낸다.

```sql
CREATE TABLE `workspace` (
  `id` text PRIMARY KEY,
  `type` text NOT NULL,
  `branch` text,
  `name` text,
  `directory` text,
  `extra` text,                     -- JSON
  `project_id` text NOT NULL        -- FK → project (CASCADE)
);
```

#### session_share

세션 공유 정보를 저장한다.

```sql
CREATE TABLE `session_share` (
  `session_id` text PRIMARY KEY,    -- FK → session (CASCADE)
  `id` text NOT NULL,
  `secret` text NOT NULL,
  `url` text NOT NULL,
  `time_created` integer NOT NULL,
  `time_updated` integer NOT NULL
);
```

---

## 3. JSON File Storage (Legacy)

SQLite 이전에 사용했던 파일 기반 스토리지이다. 현재도 레거시 호환을 위해
코드가 남아있고, `opencode db migrate` 명령으로 SQLite로 마이그레이션할 수 있다.

### 파일 구조

```
~/.local/share/opencode/storage/
├── migration                          # 마이그레이션 버전 번호
├── project/
│   └── {projectID}.json               # 프로젝트 정보
├── session/
│   └── {projectID}/
│       └── {sessionID}.json           # 세션 정보
├── message/
│   └── {sessionID}/
│       └── {messageID}.json           # 메시지 정보
├── part/
│   └── {messageID}/
│       └── {partID}.json              # 파트 정보
└── session_diff/
    └── {sessionID}.json               # 세션 diff 요약
```

### Storage API

JSON 스토리지는 key-value 방식으로 동작한다. 키는 문자열 배열이고,
파일 시스템 경로로 변환된다.

```typescript
// packages/opencode/src/storage/storage.ts
export namespace Storage {
  // 읽기: ["session", "abc123", "def456"] → storage/session/abc123/def456.json
  export async function read<T>(key: string[]) { ... }

  // 쓰기
  export async function write<T>(key: string[], content: T) { ... }

  // 수정 (read → modify → write)
  export async function update<T>(key: string[], fn: (draft: T) => void) { ... }

  // 삭제
  export async function remove(key: string[]) { ... }

  // 목록
  export async function list(prefix: string[]) { ... }
}
```

파일 동시 접근 제어를 위해 **읽기/쓰기 락**을 사용한다:

```typescript
using _ = await Lock.read(target)   // 읽기 락
using _ = await Lock.write(target)  // 쓰기 락
```

### JSON → SQLite 마이그레이션

JSON 파일 스토리지에서 SQLite로 데이터를 이관하는 마이그레이션이 내장되어 있다.

```bash
# JSON 데이터를 SQLite로 마이그레이션
opencode db migrate
```

마이그레이션 시 프로그레스 바가 표시되며, 프로젝트/세션/메시지 단위로
이관 통계가 출력된다.

---

## 4. 데이터 흐름

### 새 세션 시작

```
사용자: opencode 실행
    │
    ▼
① Project 확인/생성
   - git rev-list로 최초 commit hash 조회 → project ID
   - project 테이블에 upsert
    │
    ▼
② Session 생성
   - SessionID.descending() 생성 → 시간 역순 정렬 가능한 ID
   - project_id, directory, title 등 설정
   - session 테이블에 insert
    │
    ▼
③ 사용자 메시지 입력
   - message 테이블에 insert (role: user)
   - part 테이블에 insert (type: text)
    │
    ▼
④ LLM 응답 수신
   - message 테이블에 insert (role: assistant)
   - part 테이블에 insert (type: text, tool_call 등)
    │
    ▼
⑤ 세션 종료
   - session의 summary 업데이트 (additions, deletions)
   - time_updated 갱신
```

### 이전 세션 복원

```
사용자: 세션 목록에서 이전 세션 선택
    │
    ▼
① session 테이블에서 해당 세션 조회
    │
    ▼
② message 테이블에서 해당 세션의 메시지 조회
   (session_id로 필터, time_created 순 정렬)
    │
    ▼
③ part 테이블에서 각 메시지의 파트 조회
    │
    ▼
④ 대화 이력 복원 → LLM 컨텍스트에 주입
```

---

## 5. CLI 도구

```bash
# DB 경로 확인
opencode db path

# SQL 쿼리 실행
opencode db "SELECT count(*) FROM session"

# 대화형 sqlite3 셸
opencode db

# JSON → SQLite 마이그레이션
opencode db migrate
```

---

## 6. Claude Code와의 비교

| 항목 | OpenCode | Claude Code |
|------|----------|-------------|
| 저장소 | SQLite (drizzle-orm) | 파일 기반 (JSON/Markdown) |
| 메모리 유형 | 세션/메시지/파트 구조화 | memory 디렉토리에 Markdown 파일 |
| 프로젝트 식별 | git 초기 commit hash | 디렉토리 경로 기반 |
| 쿼리 | SQL 쿼리 가능 | 파일 시스템 탐색 |
| 마이그레이션 | drizzle-orm 마이그레이션 | 수동 |
| 동시성 제어 | SQLite WAL + busy_timeout | 파일 기반 락 |

OpenCode는 구조화된 DB를 사용하므로 세션 검색, 통계, 필터링 등이 SQL로
가능하다는 장점이 있다.
