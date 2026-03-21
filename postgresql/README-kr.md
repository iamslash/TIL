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
  - [임베딩이란?](#임베딩이란)
  - [코사인 유사도란?](#코사인-유사도란)
  - [실전 예시: 메모리 검색](#실전-예시-메모리-검색-agent-hub)
  - [임베딩 저장 흐름](#임베딩-저장-흐름)
  - [설치 (소스 빌드)](#설치-소스-빌드)
  - [사용법](#사용법)
  - [인덱스](#인덱스)
- [Full-Text Search (전문 검색)](#full-text-search-전문-검색)
  - [핵심 개념](#핵심-개념)
  - [tsquery 문법](#tsquery-문법)
  - [토크나이저 비교](#토크나이저-비교)
  - [한국어 조사 문제와 prefix 매칭](#한국어-조사-문제와-prefix-매칭)
  - [공백 포함 이름 처리](#공백-포함-이름-처리)
  - [GIN 인덱스](#gin-인덱스)
  - [실전 예시: 크로스유저 메모리 검색](#실전-예시-크로스유저-메모리-검색-agent-hub)
  - [pgvector vs Full-Text Search 비교](#pgvector-vs-full-text-search-비교)

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

## 임베딩이란?

텍스트를 **고차원 숫자 배열(벡터)**로 변환한 것. 의미가 비슷한 텍스트는 벡터 공간에서 가까이 위치한다.

```
"디자이너"       → [0.12, -0.45, 0.78, ... ] (1536차원)
"UI 설계자"      → [0.11, -0.44, 0.79, ... ] ← 의미가 비슷 → 벡터도 가까움
"축구 선수"      → [0.89, 0.23, -0.56, ... ] ← 의미가 다름 → 벡터가 멀리
```

OpenAI `text-embedding-3-small` 모델이 1536차원 벡터를 생성한다:

```python
from openai import AsyncOpenAI
client = AsyncOpenAI(api_key="sk-...")

response = await client.embeddings.create(
    model="text-embedding-3-small",
    input="Hailey는 합정에서 사는 디자이너이다",
)
embedding = response.data[0].embedding  # [0.0123, -0.0456, ...] (1536개 float)
```

## 코사인 유사도란?

두 벡터 사이의 "방향 유사도". 값이 1에 가까울수록 의미가 비슷하다.

```
코사인 유사도 (similarity) = cos(θ) = A·B / (|A| × |B|)
  1.0  → 완전 동일한 의미
  0.5  → 약간 관련 있음
  0.0  → 무관
 -1.0  → 정반대 의미

코사인 거리 (distance) = 1 - similarity
  0.0  → 완전 동일 (거리 없음)
  1.0  → 무관
  2.0  → 정반대
```

pgvector의 `<=>` 연산자는 **코사인 거리**를 반환한다:

```sql
-- 거리가 작을수록 유사 → ORDER BY ascending
ORDER BY embedding <=> $1::vector

-- 유사도로 변환하려면 1에서 빼기
SELECT 1 - (embedding <=> $1::vector) AS similarity
```

## 실전 예시: 메모리 검색 (agent-hub)

```
유저: "Hailey 누구야?"
  → "Hailey"를 OpenAI 임베딩 API로 벡터 변환
  → memories 테이블에서 코사인 유사도 검색

결과:
  similarity 0.615  "Hailey: 정보 없음"         ← 이름 매칭, 유사도 높음
  similarity 0.257  "Bob: 대학 친구"            ← "사람: 관계" 패턴 유사
  similarity 0.230  "Alice: 회사 동료"          ← 비슷한 패턴
  similarity 0.186  "테니스 치다가 허리 다침"    ← 무관, 유사도 낮음
```

```sql
-- 실제 쿼리
SELECT id, content, 1 - (embedding <=> $2::vector) AS similarity
FROM memories
WHERE user_id = $1
  AND embedding IS NOT NULL
ORDER BY embedding <=> $2::vector
LIMIT 10;

-- $1: 유저 UUID
-- $2: "Hailey"의 임베딩 벡터 '[0.0123, -0.0456, ...]'
```

## 임베딩 저장 흐름

```
대화: "나 다음주에 부산 해운대로 여행 가려고"
  → LLM 증류: "다음주 부산 해운대 여행 계획"
  → OpenAI embedding API → [0.xxx, ...] (1536차원)
  → INSERT INTO memories (content, embedding) VALUES ($1, $2::vector)
```

```
매 메시지 수신 시:
  → 유저 메시지를 OpenAI embedding API로 벡터 변환
  → memories 테이블에서 코사인 유사도 top-10 검색
  → 시스템 프롬프트에 주입 → LLM이 기억을 활용해 답변
```

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

벡터 검색은 **모든 행과 거리 비교**가 필요하므로 인덱스 없이는 매우 느리다. pgvector는 근사 최근접 이웃(ANN) 알고리즘 기반 인덱스를 제공한다.

### 인덱스 없이 검색 (Sequential Scan)

```
10만 행, 1536차원:
  → 모든 행의 벡터와 코사인 거리 계산 (10만 × 1536 float 연산)
  → ~수 초 소요
```

### HNSW 인덱스 (권장)

**HNSW (Hierarchical Navigable Small World)** — 그래프 기반 ANN 알고리즘. 데이터를 계층적 그래프로 구성하여 빠르게 근사 이웃을 찾는다.

```sql
CREATE INDEX idx_embedding_hnsw ON memories
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
```

| 파라미터 | 의미 | 기본값 |
|---------|------|--------|
| `m` | 각 노드의 연결 수. 높을수록 정확하지만 메모리 증가 | 16 |
| `ef_construction` | 인덱스 빌드 시 탐색 범위. 높을수록 정확하지만 빌드 느림 | 64 |
| `vector_cosine_ops` | 코사인 거리용. `<=>` 연산자에 최적화 | - |

```
10만 행, HNSW 인덱스:
  → 그래프 탐색 (~수백 노드만 비교)
  → ~수 밀리초 소요 (1000배+ 빠름)
```

### IVFFlat 인덱스

**IVFFlat (Inverted File with Flat vectors)** — 데이터를 클러스터로 나누고, 검색 시 가까운 클러스터만 탐색한다.

```sql
CREATE INDEX idx_embedding_ivf ON memories
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);
```

| 파라미터 | 의미 | 권장값 |
|---------|------|--------|
| `lists` | 클러스터 수. √(행수) 정도 | 100 (1만 행), 1000 (100만 행) |

### 비교

| 특성 | HNSW | IVFFlat |
|------|------|---------|
| 정확도 | 99%+ | 95%+ |
| 메모리 | 높음 | 낮음 |
| 데이터 추가 시 | 즉시 반영 | 주기적 재빌드 권장 |
| 빌드 시간 | 느림 | 빠름 |
| 추천 규모 | ~100만 건 | 100만 건+ |
| **일반 추천** | **대부분의 경우** | 메모리 제한 환경 |

---

# Full-Text Search (전문 검색)

PostgreSQL 내장 전문 검색 시스템. 외부 검색 엔진(ElasticSearch) 없이 텍스트 검색이 가능하다.

## 핵심 개념

### tsvector — 텍스트를 검색 가능한 토큰으로 분해

```sql
SELECT to_tsvector('simple', 'Hailey는 합정에서 사는 디자이너이다');
-- 결과: 'hailey는':1 '합정에서':2 '사는':3 '디자이너이다':4
--        ↑토큰        ↑위치
```

### tsquery — 검색 조건

```sql
SELECT to_tsquery('simple', 'hailey:*');
-- 결과: 'hailey':*   ← "hailey로 시작하는" 토큰을 찾아라
```

### @@ — 매칭 연산자

```sql
-- tsvector가 tsquery를 만족하는가?
SELECT to_tsvector('simple', 'Hailey는 디자이너이다')
    @@ to_tsquery('simple', 'hailey:*');
-- → true ('hailey는'이 'hailey:*'에 매칭)
```

## tsquery 문법

| 문법 | 의미 | 예시 |
|------|------|------|
| `word` | 정확히 일치 | `'hailey'` → 'hailey' 토큰만 |
| `word:*` | prefix 매칭 | `'hailey:*'` → 'hailey', 'hailey는', 'hailey가' 등 |
| `a & b` | AND | `'hailey:* & 디자이너:*'` → 둘 다 포함 |
| `a \| b` | OR | `'hailey:* \| alice:*'` → 하나라도 포함 |
| `(a & b)` | 그룹 | `'(los:* & angeles:*)'` → los와 angeles 모두 |

## 토크나이저 비교

| 토크나이저 | 동작 | 한국어 |
|-----------|------|--------|
| `english` | 어근 추출 (running→run, cities→city) | 한국어 무시 |
| `simple` | 소문자 변환 + 공백 분리만 | 조사 붙지만 `:*`로 해결 |
| `korean` | 한국어 형태소 분석 | PostgreSQL 기본 미지원 (확장 필요) |

## 한국어 조사 문제와 prefix 매칭

`simple` 토크나이저는 한국어 조사를 분리하지 못한다:

```sql
-- "Hailey는" → 토큰 'hailey는' (조사 "는"이 붙음)

-- 정확 매칭: 실패!
to_tsquery('simple', 'hailey')     -- 'hailey' ≠ 'hailey는'

-- prefix 매칭: 성공!
to_tsquery('simple', 'hailey:*')   -- 'hailey:*' → 'hailey는'에 매칭
```

**`:*` prefix 매칭이 한국어 FTS의 핵심 해결책이다.**

## 공백 포함 이름 처리

```sql
-- "Los Angeles" → 단어별 분리 후 AND 결합
to_tsquery('simple', '(los:* & angeles:*)')

-- 전체 엔티티 검색 (OR 결합)
to_tsquery('simple', '(vinay:* & kuruvila:*) | alice:* | hailey:*')
```

**공백 있는 이름을 그대로 넣으면 syntax error:**
```sql
-- 에러! "los angeles:*"는 유효한 tsquery가 아님
to_tsquery('simple', 'los angeles:*')

-- 올바른 방법: 단어별 분리
to_tsquery('simple', '(los:* & angeles:*)')
```

## GIN 인덱스

**GIN (Generalized Inverted Index)** = 역인덱스. 책 뒤의 색인(찾아보기)과 같다.

### 동작 원리

```
memories 테이블:
  row 1: "Hailey는 합정에서 사는 디자이너이다"
  row 2: "Alice는 회사 동료이다"
  row 3: "Hailey는 David의 회사 동료 디자이너이다"

GIN 인덱스 (역인덱스):
  'hailey는'    → [row 1, row 3]
  '합정에서'     → [row 1]
  '디자이너이다' → [row 1, row 3]
  'alice는'     → [row 2]
  '회사'        → [row 2, row 3]
  '동료이다'     → [row 2, row 3]
  'david의'     → [row 3]
```

### GIN 인덱스 없이 vs 있을 때

```
GIN 없음 (Sequential Scan):
  "hailey:*" 검색 → 모든 행에 to_tsvector() 계산 → 하나씩 비교
  10만 행 → 10만 번 계산 → 느림

GIN 있음 (Index Scan):
  "hailey:*" 검색 → 색인에서 'hailey*' 조회 → [row 1, row 3] 즉시 반환
  10만 행이어도 색인 1번 조회 → 빠름
```

### 인덱스 생성

```sql
CREATE INDEX idx_content_tsv ON memories
  USING gin (to_tsvector('simple', content));
```

### 성능 비교

| 행 수 | GIN 인덱스 없음 | GIN 인덱스 있음 |
|-------|----------------|---------------|
| 100행 | ~1ms | ~0.1ms |
| 10,000행 | ~50ms | ~0.2ms |
| 1,000,000행 | ~5초 | ~1ms |

### 저장 공간 비교 (ElasticSearch vs GIN)

| | PostgreSQL GIN | ElasticSearch |
|---|---|---|
| 역인덱스 | ✅ 있음 | ✅ 있음 (동일 구조) |
| 원본 데이터 복사 | ❌ 없음 (테이블 참조) | ✅ `_source`에 전체 복사 |
| Doc values | ❌ 없음 | ✅ 기본 생성 |
| 복제/샤드 | ❌ 단일 | ✅ replica 기본 1개 (×2) |
| **저장 비율** | **원본의 3~4배** | **원본의 10~15배** |

## 실전 예시: 크로스유저 메모리 검색 (agent-hub)

```sql
-- 다른 유저의 메모리에서 엔티티 이름으로 검색
SELECT content, sensitivity
FROM memories
WHERE user_id != $1                    -- 다른 유저의 메모리만
  AND is_secret = false                -- 비밀 아닌 것만
  AND sensitivity = ANY($4)            -- PUBLIC/SOCIAL만
  AND to_tsvector('simple', content) @@ to_tsquery('simple', $2)
ORDER BY created_at DESC
LIMIT $3;

-- $2 예시: '(vinay:* & kuruvila:*) | alice:* | hailey:*'
-- $4 예시: ARRAY['PUBLIC', 'SOCIAL']
```

## pgvector vs Full-Text Search 비교

| | pgvector (embedding) | Full-Text Search (tsvector) |
|---|---|---|
| 저장 위치 | 별도 컬럼 (`vector(1536)`) | 컬럼 없음 (실시간 생성, GIN 인덱스) |
| 생성 비용 | OpenAI API 호출 (~$0.0001/건) | 무료 (PostgreSQL 내부) |
| 검색 방식 | 코사인 유사도 `<=>` | 키워드 매칭 `@@` |
| 강점 | "의미" 이해 ("디자이너" → "UI 설계자"도 매칭) | 정확한 단어 매칭, 빠름 |
| 약점 | API 호출 필요, 느림 | 의미적 유사어 이해 불가 |
| 인덱스 | HNSW / IVFFlat | GIN |
| 용도 | "관련된 기억 찾기" | "특정 이름/키워드 찾기" |

**실전에서는 두 가지를 병합하여 사용한다:**

```python
# 1단계: pgvector 벡터 검색 (의미 유사도)
embedding = await generate_embedding("Hailey")
vector_results = await search_by_vector(user_id, embedding, limit=10)

# 2단계: FTS 텍스트 검색 (키워드 정확 매칭, 보완)
text_results = await search_by_text(user_id, "Hailey", limit=5)

# 병합 + 중복 제거
combined = deduplicate(vector_results + text_results)
```
