# PostgreSQL 한글 노트

> 영문 문서: [README.md](./README.md)

---

- [프로세스 아키텍처](#프로세스-아키텍처)
  - [프로세스 기반 모델](#프로세스-기반-모델)
  - [프로세스 목록](#프로세스-목록)
  - [상주 프로세스 (Background Processes)](#상주-프로세스-background-processes)
  - [클라이언트 커넥션 프로세스 (Backend Processes)](#클라이언트-커넥션-프로세스-backend-processes)
  - [데이터 흐름](#데이터-흐름)
  - [MySQL과의 비교](#mysql과의-비교)
  - [메모리 아키텍처](#메모리-아키텍처)
  - [프로세스 확인 명령어](#프로세스-확인-명령어)
  - [커넥션 풀이 중요한 이유](#커넥션-풀이-중요한-이유)
- [pgvector](#pgvector)
  - [설치 (소스 빌드)](#설치-소스-빌드)
  - [사용법](#사용법)
  - [인덱스](#인덱스)

---

# 프로세스 아키텍처

## 프로세스 기반 모델

PostgreSQL은 **프로세스 기반 아키텍처**를 사용한다. 쓰레드가 아닌 `fork()`로 프로세스를 생성한다.

```
postmaster (메인 프로세스, PID 1)
│
├── checkpointer              # WAL → 데이터 파일 반영
├── background writer         # 더티 페이지 미리 쓰기
├── walwriter                 # WAL 버퍼 → 디스크 flush
├── autovacuum launcher       # dead tuple 정리 조율
├── logical replication launcher  # 논리적 복제 관리
│
└── 클라이언트 접속마다 fork()
    ├── backend process #1    # 커넥션 1 전담
    ├── backend process #2    # 커넥션 2 전담
    └── ...                   # max_connections까지
```

## 프로세스 목록

실제 `ps aux | grep postgres` 출력:

```
postgres  /usr/lib/postgresql/16/bin/postgres -D /var/lib/...  # postmaster
postgres  postgres: 16/main: checkpointer
postgres  postgres: 16/main: background writer
postgres  postgres: 16/main: walwriter
postgres  postgres: 16/main: autovacuum launcher
postgres  postgres: 16/main: logical replication launcher
postgres  postgres: 16/main: myuser mydb 127.0.0.1(33976) idle   # 커넥션 1
postgres  postgres: 16/main: myuser mydb 127.0.0.1(48302) idle   # 커넥션 2
```

## 상주 프로세스 (Background Processes)

서버 시작 시 생성되어 항상 실행된다.

### postmaster

- **역할**: 모든 프로세스의 부모. 클라이언트 접속을 수락하고 `fork()`로 자식 프로세스(backend)를 생성한다.
- **포트**: 기본 5432
- **이 프로세스가 죽으면 PG 전체가 내려간다.**

### checkpointer

- **역할**: WAL(Write-Ahead Log)의 내용을 주기적으로 데이터 파일에 반영(flush)한다.
- **트리거**: `checkpoint_timeout`(기본 5분) 또는 WAL이 `max_wal_size`만큼 쌓이면 실행.
- **왜 필요한가**: checkpoint 이전의 WAL은 삭제해도 된다. 크래시 복구 시 마지막 checkpoint부터 WAL을 재생하면 된다.

```
WAL 세그먼트:  [===checkpoint===][새 WAL][새 WAL][새 WAL]
                    ↑                              ↑
              여기까지 데이터 파일에 반영됨      여기까지 WAL에만 존재
```

### background writer

- **역할**: 공유 버퍼(shared_buffers)에서 더티 페이지를 디스크에 미리 써둔다.
- **checkpointer와 차이**: checkpointer는 크게/간헐적으로 쓰고, background writer는 작게/자주 써서 I/O 스파이크를 완화한다.
- **비유**: checkpointer가 "대청소"라면, background writer는 "수시로 정리하는 청소부".

### walwriter

- **역할**: WAL 버퍼를 디스크에 주기적으로 flush한다.
- **왜 별도 프로세스인가**: 트랜잭션 커밋 시에도 WAL을 쓰지만, walwriter가 백그라운드에서 추가 flush하여 커밋 시 I/O 대기를 줄인다.
- `wal_writer_delay`(기본 200ms) 간격으로 실행.

### autovacuum launcher

- **역할**: VACUUM이 필요한 테이블을 모니터링하고 autovacuum worker를 자동 실행한다.
- **VACUUM이 하는 일**:
  1. 삭제/업데이트로 발생한 dead tuple의 공간 회수
  2. `pg_statistic` 테이블 통계 갱신 (쿼리 플래너가 사용)
  3. transaction ID wraparound 방지
- **이것이 없으면**: 테이블이 계속 비대해지고(bloat), 쿼리 성능이 저하된다.

```
autovacuum launcher (항상 실행)
  └── autovacuum worker (필요 시 생성, 동시에 여러 개 가능)
        └── 특정 테이블에 대해 VACUUM 수행
```

### logical replication launcher

- **역할**: 논리적 복제(logical replication) 구독이 있으면 worker를 실행한다.
- 복제를 사용하지 않으면 **유휴 상태**로 대기.
- PG 기본으로 항상 떠 있다.

## 클라이언트 커넥션 프로세스 (Backend Processes)

클라이언트가 접속하면 postmaster가 `fork()`로 생성한다.

```
postgres: 16/main: myuser mydb 127.0.0.1(33976) idle
          ──────   ────── ────  ───────────────  ────
          버전/클러스터  유저  DB    클라이언트IP(포트) 상태
```

### 상태 값

| 상태 | 의미 |
|------|------|
| `idle` | 쿼리 대기 중 (커넥션 풀에서 유지) |
| `active` | 쿼리 실행 중 |
| `idle in transaction` | 트랜잭션 안에서 다음 쿼리 대기 중 (주의: 오래 지속되면 문제) |
| `idle in transaction (aborted)` | 트랜잭션 에러 후 ROLLBACK 대기 |
| `fastpath function call` | 빠른 경로 함수 호출 중 |

### 활성 커넥션 조회 (SQL)

```sql
SELECT pid, usename, datname, client_addr, state, query
FROM pg_stat_activity
WHERE datname = 'momo'
ORDER BY backend_start;
```

## 데이터 흐름

```
클라이언트 → 커넥션 프로세스 (backend)
                │
                ▼
         shared_buffers (메모리)
          ┌─────┴─────┐
          │            │
   background writer  WAL 버퍼
   (더티 페이지 → 디스크) │
          │            ▼
          │       walwriter
          │       (WAL → 디스크)
          │            │
          ▼            ▼
     데이터 파일      WAL 파일
     (base/)        (pg_wal/)
          │
     checkpointer
     (주기적 전체 flush)
```

### 쓰기 순서 (WAL 먼저)

1. 트랜잭션 실행 → **WAL 버퍼**에 변경 내용 기록
2. 커밋 시 → WAL 버퍼를 **디스크(WAL 파일)**에 flush (walwriter 또는 동기적)
3. shared_buffers의 더티 페이지는 **나중에** background writer/checkpointer가 디스크에 기록

이것이 **Write-Ahead Logging**이다. 데이터 파일보다 WAL을 먼저 쓰므로 크래시 시 WAL을 재생하여 복구할 수 있다.

## MySQL과의 비교

| 항목 | PostgreSQL | MySQL (InnoDB) |
|------|-----------|----------------|
| 동시 접속 처리 | **프로세스** (fork) | **쓰레드** |
| 메모리 격리 | 프로세스별 독립 | 공유 메모리 |
| 커넥션당 메모리 | ~10-20MB | ~1-2MB |
| 안정성 | 하나가 죽어도 다른 커넥션 영향 없음 | 쓰레드 크래시 시 전체 영향 가능 |
| 최대 커넥션 실용치 | 수백 개 (커넥션 풀 필수) | 수천 개 가능 |
| MVCC 구현 | 튜플에 버전 저장 (dead tuple → VACUUM 필요) | Undo log 기반 (자동 정리) |

## 메모리 아키텍처

```
┌─────────────────────────────────────────┐
│           공유 메모리 (Shared Memory)      │
│                                          │
│  shared_buffers (128MB~수GB)             │  ← 데이터 페이지 캐시
│  WAL buffers (16MB)                      │  ← WAL 쓰기 버퍼
│  CLOG buffers                            │  ← 트랜잭션 커밋 상태
│  Lock space                              │  ← 잠금 정보
│                                          │
└─────────────────────────────────────────┘
       ↑ 모든 프로세스가 공유

┌──────────────┐  ┌──────────────┐
│ Backend #1   │  │ Backend #2   │    ← 프로세스별 독립
│              │  │              │
│ work_mem     │  │ work_mem     │    ← 정렬/해시 작업용
│ temp_buffers │  │ temp_buffers │    ← 임시 테이블용
│ maintenance  │  │ maintenance  │    ← VACUUM/인덱스 빌드용
│   _work_mem  │  │   _work_mem  │
└──────────────┘  └──────────────┘
```

### 주요 메모리 설정

| 파라미터 | 범위 | 권장값 (1.8GB RAM) | 설명 |
|---------|------|-------------------|------|
| `shared_buffers` | 전체 | 128MB (~RAM의 7%) | 데이터 캐시. 보통 RAM의 25%이지만 저메모리에서는 보수적으로 |
| `effective_cache_size` | 전체 | 256MB | 플래너에게 "OS 캐시 포함 이 정도 메모리가 있다"고 알려줌 (실제 할당 아님) |
| `work_mem` | 프로세스별 | 4MB | 정렬/해시 조인 시 사용. 커넥션 수 x work_mem 만큼 사용 가능 |
| `maintenance_work_mem` | 프로세스별 | 64MB | VACUUM, CREATE INDEX 시 사용 |

## 프로세스 확인 명령어

```bash
# 전체 프로세스 목록
ps aux | grep postgres

# 프로세스 트리로 보기
pstree -p $(head -1 /var/lib/postgresql/16/main/postmaster.pid)

# 활성 커넥션 수
psql -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"

# 커넥션별 상세 정보
psql -c "SELECT pid, usename, datname, state, query_start, query
         FROM pg_stat_activity
         WHERE datname IS NOT NULL
         ORDER BY query_start DESC;"

# 백그라운드 프로세스 상태
psql -c "SELECT * FROM pg_stat_bgwriter;"
```

## 커넥션 풀이 중요한 이유

PostgreSQL은 커넥션 = 프로세스이므로, 커넥션이 많아지면:

1. **메모리**: 커넥션 100개 = 프로세스 100개 = ~2GB 추가 메모리
2. **fork 비용**: 새 커넥션마다 fork() 시스템 콜 (수 ms)
3. **컨텍스트 스위칭**: 프로세스 수가 CPU 코어를 초과하면 성능 저하

그래서 **PgBouncer** 같은 커넥션 풀러가 필수이다:

```
앱 서버 (100개 커넥션) → PgBouncer (20개 커넥션 유지) → PostgreSQL (20개 backend)
```

Node.js에서는 `pg` 라이브러리의 내장 풀로 관리:

```typescript
import { Pool } from "pg";
const pool = new Pool({
  connectionString: "postgresql://user:pw@localhost:5432/db",
  min: 2,    // 최소 커넥션 (항상 유지)
  max: 10,   // 최대 커넥션
});
```

---

# pgvector

PostgreSQL에서 벡터 유사도 검색을 지원하는 확장. AI/ML 임베딩 검색에 사용한다.

## 설치 (소스 빌드)

apt 패키지가 없는 환경(ARM64 등)에서는 소스 빌드한다.

```bash
sudo apt install -y postgresql-server-dev-16

cd /tmp
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# DB에서 활성화
psql -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## 사용법

```sql
-- 1536차원 벡터 컬럼 생성
CREATE TABLE items (
  id SERIAL PRIMARY KEY,
  content TEXT,
  embedding vector(1536)
);

-- 벡터 삽입
INSERT INTO items (content, embedding)
VALUES ('hello', '[0.1, 0.2, ...]');

-- 코사인 유사도 검색 (가까운 순)
SELECT id, content, 1 - (embedding <=> $1) AS similarity
FROM items
ORDER BY embedding <=> $1
LIMIT 10;
```

### 거리 연산자

| 연산자 | 거리 함수 | 용도 |
|--------|----------|------|
| `<=>` | 코사인 거리 | 텍스트 임베딩 (가장 일반적) |
| `<->` | L2 (유클리드) 거리 | 이미지 임베딩 |
| `<#>` | 내적의 음수 | 정규화된 벡터에서 빠른 검색 |

## 인덱스

```sql
-- HNSW (권장, 정확도 높음)
CREATE INDEX idx_embedding_hnsw ON items
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- IVFFlat (메모리 적음, 주기적 재빌드 필요)
CREATE INDEX idx_embedding_ivf ON items
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);
```

| 특성 | HNSW | IVFFlat |
|------|------|---------|
| 정확도 | 99%+ | 95%+ |
| 메모리 | 높음 | 낮음 |
| 데이터 추가 시 | 즉시 반영 | 주기적 재빌드 권장 |
| 추천 규모 | ~100만 건 | 100만 건+ |
