# ClickHouse 한글 가이드

- [개요](#개요)
- [왜 ClickHouse인가](#왜-clickhouse인가)
  - [PostgreSQL 의 한계](#postgresql-의-한계)
  - [ClickHouse 가 빛나는 순간](#clickhouse-가-빛나는-순간)
- [핵심 개념](#핵심-개념)
  - [컬럼 스토리지 (Column Storage)](#컬럼-스토리지-column-storage)
  - [MergeTree 엔진](#mergetree-엔진)
  - [파티션과 샤딩](#파티션과-샤딩)
  - [Materialized View](#materialized-view)
    - [MV 내부 동작 원리](#mv-내부-동작-원리)
    - [머지 전 중복 행 주의](#머지-전-중복-행-주의)
    - [MV 와 DELETE: 불일치 문제와 대처](#mv-와-delete-불일치-문제와-대처)
  - [압축과 성능](#압축과-성능)
- [아키텍처](#아키텍처)
  - [단일 노드](#단일-노드)
  - [클러스터 (Shard + Replica)](#클러스터-shard--replica)
- [학습 자료](#학습-자료)
- [관련 문서](#관련-문서)

---

# 개요

ClickHouse 는 **OLAP (Online Analytical Processing) 데이터베이스**이다. 대규모 데이터 집계 쿼리를 **매우 빠르게** 처리하도록 설계되었다.

한 문장 요약:

```
PostgreSQL:  행(row) 중심 저장 → 집계 쿼리 느림
ClickHouse:  컬럼(column) 중심 저장 → 집계 쿼리 초고속
```

쉽게 말해, "100만 개 레코드에서 월별 매출의 합계를 내려고 한다면?" 같은 분석 쿼리를 다룰 때, PostgreSQL 은 한계가 있지만 ClickHouse 는 밀리초 단위로 답한다.

# 왜 ClickHouse인가

## PostgreSQL 의 한계

PostgreSQL은 OLTP (Online Transaction Processing) 데이터베이스다. 행을 중심으로 저장한다:

```
행 중심 저장 (Row-oriented):

| user_id | date       | category | amount |
|---------|------------|----------|--------|
| 1       | 2024-01-01 | 음식     | 50000  |
| 2       | 2024-01-01 | 택시     | 10000  |
| 1       | 2024-01-02 | 음식     | 45000  |
...

저장 순서: [1, 2024-01-01, 음식, 50000] [2, 2024-01-01, 택시, 10000] ...
```

이 방식은 **하나의 레코드를 읽을 때** 최적이다. 하지만 분석 쿼리는 다르다:

```sql
-- "2024년 카테고리별 월별 합계"
SELECT category, DATE_TRUNC('month', date) AS month, SUM(amount)
FROM transactions
WHERE date >= '2024-01-01'
GROUP BY category, month
ORDER BY month, category;
```

이 쿼리를 실행하려면:
- 100만 행을 **모두 읽어야** 함
- 각 행에서 `user_id`, `date`, `category`, `amount` 4개 컬럼을 **모두 디코딩**
- 필요한 컬럼은 `category`, `date`, `amount` 3개뿐인데도 `user_id` 도 읽음

결과: **메모리 낭비, 디스크 I/O 낭비, CPU 낭비**. 테이블이 클수록 쿼리가 느려진다.

## ClickHouse 가 빛나는 순간

ClickHouse 는 **컬럼 중심**으로 저장한다:

```
컬럼 중심 저장 (Column-oriented):

user_id:   [1, 2, 1, ...]
date:      [2024-01-01, 2024-01-01, 2024-01-02, ...]
category:  [음식, 택시, 음식, ...]
amount:    [50000, 10000, 45000, ...]

저장 순서: 같은 컬럼끼리 연속해서 저장
```

같은 집계 쿼리를 실행하면:

```sql
SELECT category, DATE_TRUNC('month', date) AS month, SUM(amount)
FROM transactions
WHERE date >= '2024-01-01'
GROUP BY category, month
ORDER BY month, category;
```

ClickHouse 는:
- **필요한 3개 컬럼**(`category`, `date`, `amount`) 만 읽음
- 각 컬럼은 **연속 메모리**에 저장되어 CPU 캐시 효율이 높음
- **벡터 처리**로 1억 개 행을 한 번에 처리

성능 비교 (100만 행):

```
PostgreSQL:  ~2-3초
ClickHouse:  ~50-100ms (20-60배 빠름)
```

### 왜 컬럼 저장이 빠른가?

1. **선택적 읽기**: 4개 컬럼 중 3개만 필요하면, 25% 데이터만 읽음
2. **압축**: 같은 타입 데이터가 연속하면 매우 잘 압축됨. 수치는 델타 인코딩, 문자열은 dictionary encoding
3. **벡터화 처리**: SIMD 명령어로 여러 값을 동시에 처리
4. **캐시 친화**: 메모리 캐시 미스 확률이 낮음

# 핵심 개념

## 컬럼 스토리지 (Column Storage)

ClickHouse 가 데이터를 저장하는 기본 형식이다. 각 컬럼을 독립적인 블록으로 저장한다.

```clickhouse
-- 3개 행을 삽입
INSERT INTO users (id, name, age) VALUES
(1, 'Alice', 30),
(2, 'Bob', 28),
(3, 'Charlie', 35);
```

메모리 레이아웃:

```
id:    [1, 2, 3] (숫자, 최소 메모리)
name:  ["Alice", "Bob", "Charlie"]
age:   [30, 28, 35] (숫자, 쉽게 압축)
```

이를 통해:
- 쿼리가 `SELECT COUNT(*) GROUP BY name` 을 요청하면, `name` 컬럼만 읽으면 됨
- 전체 데이터 크기의 1/3 만 I/O

## MergeTree 엔진

ClickHouse 의 기본이자 가장 강력한 테이블 엔진이다. 삽입된 데이터를 자동으로 병합하면서 성능을 최적화한다.

```clickhouse
CREATE TABLE events (
    event_date Date,
    event_time DateTime,
    user_id UInt32,
    event_name String,
    properties Map(String, String)
) ENGINE = MergeTree()
ORDER BY (user_id, event_time)
PARTITION BY event_date;
```

특징:

- **ORDER BY**: 데이터 정렬 순서. 쿼리 성능에 직결됨
- **PARTITION BY**: 날짜 단위로 파티션 (예: 일일 파티션)
- **자동 압축**: 작은 부분들을 자동으로 병합
- **PRIMARY KEY**: ORDER BY 의 첫 번째 컬럼이 자동으로 PRIMARY KEY 역할

삽입된 데이터는 처음에 "part" 라는 독립 블록으로 저장되고, 백그라운드에서 자동으로 병합된다.

```
INSERT 직후:  [part_1] [part_2] [part_3]
수초 후:      [part_1_2_3_merged]
```

## 파티션과 샤딩

### Partition vs Shard 핵심 차이

| | Partition | Shard |
|---|---|---|
| **분할 대상** | 하나의 테이블 안의 데이터 | 테이블 전체를 여러 노드로 |
| **위치** | **같은 노드** 안에서 나뉨 | **다른 노드**로 나뉨 |
| **기준** | 보통 날짜 (`PARTITION BY toYYYYMM(date)`) | 보통 키 해시 (`user_id % shard_count`) |
| **목적** | 오래된 데이터 삭제, 쿼리 범위 축소 | 처리량 확장 (수평 스케일링) |

```
노드 A (Shard 1)
├── Partition 2026-01  ← 1월 데이터 (디스크에 별도 디렉토리)
├── Partition 2026-02  ← 2월 데이터
└── Partition 2026-03  ← 3월 데이터

노드 B (Shard 2)
├── Partition 2026-01
├── Partition 2026-02
└── Partition 2026-03
```

> "Partition = 논리적, Shard = 물리적" 이라고 단순화할 수도 있지만, **Partition 도 물리적으로 분리된다.** 각 Partition 은 디스크에 별도의 디렉토리(Part)로 저장되며, `DROP PARTITION` 하면 해당 디렉토리가 통째로 삭제된다. 더 정확하게는: **Partition = 같은 노드 안에서 조건 기준으로 물리 분리 (관리 편의)**, **Shard = 여러 노드에 데이터 분산 (성능 확장)** 이다.

### 파티션 (Partition)

같은 노드 안에서 테이블 데이터를 조건(보통 날짜) 기준으로 나누는 방식이다. 각 파티션은 디스크에 별도의 디렉토리로 저장된다.

```clickhouse
CREATE TABLE sales (
    date Date,
    amount UInt64,
    category String
) ENGINE = MergeTree()
ORDER BY category
PARTITION BY date;  -- 날짜별로 파티션
```

메리트:

- 오래된 파티션 삭제가 빠름 (`ALTER TABLE sales DROP PARTITION '2025-01-01'` — 즉시 삭제)
- 파티션별로 독립적으로 최적화
- 백업/복구가 쉬움
- 쿼리 시 불필요한 파티션을 건너뜀 (Partition Pruning)

### 샤딩 (Sharding)

대규모 데이터를 여러 노드에 분산하는 방식. ClickHouse 클러스터에서는 분산 테이블(Distributed Table)이 자동으로 관리한다.

```
Shard 1 (Node A):  user_id % 2 == 0 인 데이터
Shard 2 (Node B):  user_id % 2 == 1 인 데이터

SELECT SUM(amount) FROM dist_table  -- 자동으로 두 샤드에서 읽고 병합
```

Flink 로 비유하면:
- **Partition** ≈ RocksDB 안에서 키 범위별로 SST 파일이 나뉘는 것
- **Shard** ≈ `keyBy()` 로 데이터가 다른 Slot(다른 TaskManager)으로 가는 것

## Materialized View

사전에 계산된 뷰다. 데이터가 삽입될 때마다 자동으로 갱신된다.

```clickhouse
CREATE MATERIALIZED VIEW daily_summary
ENGINE = SummingMergeTree()
ORDER BY (date, category)
AS
SELECT
    DATE(event_time) AS date,
    category,
    COUNT(*) AS event_count,
    SUM(amount) AS total_amount
FROM events
GROUP BY date, category;
```

효과:

- 매번 쿼리할 때 GROUP BY 계산 안 함
- 데이터 삽입 시간에 미리 집계
- 대시보드 조회가 극도로 빠름

```clickhouse
-- 거의 즉시 반환
SELECT * FROM daily_summary WHERE date >= '2024-01-01' LIMIT 10;
```

### MV 내부 동작 원리

MV 는 전체 테이블을 다시 계산하지 않는다. **새로 INSERT 된 행에만** SELECT 를 적용하여 내부 테이블에 저장한다.

```
events 테이블에 INSERT 발생
  │
  ▼
새로 INSERT 된 행만 SELECT 쿼리에 통과
  │
  ▼
결과를 MV 의 내부 테이블 (.inner.daily_summary) 에 INSERT
  │
  ▼
SummingMergeTree 가 백그라운드에서 같은 (date, category) 키의 행을 합산
```

```
events 에 100만 행이 이미 있고 10행을 INSERT 하면:
  → MV 는 새 10행에 대해서만 집계
  → 기존 100만 행은 건드리지 않음
  → 이것이 실시간 처리가 가능한 이유
```

내부 테이블은 자동 생성된다:

```sql
SHOW TABLES LIKE '.inner%';
-- → .inner.daily_summary (독립적인 SummingMergeTree 테이블)
```

### 머지 전 중복 행 주의

SummingMergeTree 는 **백그라운드 머지** 시점에 합산한다. 머지 전에는 같은 키의 행이 여러 개 존재할 수 있다.

```sql
-- 머지 전: 중복 행이 보일 수 있음
SELECT * FROM daily_summary;
-- (2026-04-01, shoes, 3, 150.00)  ← 첫 번째 INSERT 분
-- (2026-04-01, shoes, 2, 80.00)   ← 두 번째 INSERT 분

-- 정확한 결과를 원하면 FINAL 또는 SUM 사용
SELECT * FROM daily_summary FINAL;
-- (2026-04-01, shoes, 5, 230.00)

-- 또는 직접 합산
SELECT date, category, SUM(event_count), SUM(total_amount)
FROM daily_summary
GROUP BY date, category;
```

### MV 와 DELETE: 불일치 문제와 대처

**MV 는 INSERT 에만 반응한다. UPDATE/DELETE 는 반영되지 않는다.**

```
INSERT INTO events → MV 트리거 됨
UPDATE events     → MV 트리거 안 됨
DELETE events     → MV 트리거 안 됨
```

이것은 버그가 아니라 **설계 철학**이다. ClickHouse 는 append-only 분석 엔진이므로 DELETE 가 거의 발생하지 않는 것이 전제이다.

| 상황 | 해결 방법 | MV 불일치? |
|------|----------|-----------|
| **오래된 데이터 정리** | `DROP PARTITION` (MV 도 함께 DROP) | 없음 |
| **잘못된 데이터 수정** | 보정 이벤트를 INSERT (음수 값) | 없음 |
| **GDPR 개인정보 삭제** | `ALTER TABLE DELETE` + MV 재생성 | 있음 (재생성 필요) |

**패턴 1: DROP PARTITION (가장 흔함)**

```sql
-- 원본과 MV 모두 같은 파티션 기준이면 함께 삭제
ALTER TABLE events DROP PARTITION '202601';
ALTER TABLE `.inner.daily_summary` DROP PARTITION '202601';
```

**패턴 2: 보정 이벤트 (Compensating Event)**

```sql
-- 원래: shoes 100원 구매
INSERT INTO events VALUES ('2026-04-01 10:00:00', 'shoes', 1, 100);
-- MV: (2026-04-01, shoes, 1, 100)

-- 취소: DELETE 대신 음수 값을 INSERT
INSERT INTO events VALUES ('2026-04-01 10:05:00', 'shoes', -1, -100);
-- MV 머지 후: (2026-04-01, shoes, 0, 0)  ← SummingMergeTree 가 자동 상쇄
```

**패턴 3: GDPR 삭제 (드문 경우)**

```sql
-- 원본에서 삭제
ALTER TABLE events DELETE WHERE user_id = 12345;
-- MV 에는 user_id 가 집계되어 사라졌으므로 특정 유저 기여분을 뺄 수 없음
-- → MV 를 DROP 하고 재생성하는 수밖에 없음
```

> 핵심: ClickHouse 는 **"삭제하지 않는다"** 가 기본 전제이다. 오래된 데이터는 TTL/DROP PARTITION 으로, 잘못된 데이터는 보정 이벤트로 처리한다. DELETE 가 필요한 GDPR 같은 경우만 MV 재생성을 감수한다.

## 압축과 성능

ClickHouse 는 자동으로 데이터를 압축한다. 컬럼별로 최적의 코덱을 선택한다:

```clickhouse
-- 압축 레벨 설정
CREATE TABLE compressed_data (
    id UInt64,
    name String,
    value Float64
) ENGINE = MergeTree()
ORDER BY id
SETTINGS compress_codec = 'ZSTD';  -- 압축 알고리즘 지정
```

일반적인 압축률:

```
미압축:   1,000MB
ZSTD:     100-200MB (5-10배 압축)
```

장점: 디스크 I/O 감소 + 네트워크 대역폭 절약 (클러스터 간 통신 시).

# 아키텍처

## 단일 노드

개발/테스트 환경이나 데이터가 100GB 미만인 경우에 충분하다.

```
┌─────────────────────────────────────┐
│      ClickHouse Server (1 Node)     │
│  ┌──────────────────────────────┐   │
│  │   Memory Buffer (InMemory)   │   │
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │  Disk Storage (MergeTree)    │   │
│  │  - Part 1 (압축)              │   │
│  │  - Part 2 (압축)              │   │
│  │  - Part N (병합 중)           │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

데이터 흐름:

```
INSERT → Memory Buffer → Disk (Part) → Auto-Merge
```

## 클러스터 (Shard + Replica)

대규모 데이터(TB 이상)나 고가용성이 필요한 경우.

```
┌─────────────────────────────────────────────────┐
│      ClickHouse Cluster (3 Shards)              │
├─────────────────────────────────────────────────┤
│ Shard 1          │ Shard 2          │ Shard 3  │
│ ┌────────────┐   │ ┌────────────┐   │          │
│ │ Replica 1a │   │ │ Replica 2a │   │          │
│ └────────────┘   │ └────────────┘   │          │
│ ┌────────────┐   │ ┌────────────┐   │          │
│ │ Replica 1b │   │ │ Replica 2b │   │          │
│ └────────────┘   │ └────────────┘   │          │
└─────────────────────────────────────────────────┘
```

용어:

- **Shard**: 데이터를 분산하는 단위. 샤드별로 다른 데이터를 저장
- **Replica**: 같은 샤드의 복제본. 고가용성과 읽기 확장
- **Distributed Table**: 클러스터 전체를 하나의 테이블처럼 보이게 하는 프록시 테이블

클러스터 쿼리:

```clickhouse
-- Distributed 테이블 생성 (자동으로 3개 샤드에 분산)
CREATE TABLE dist_events AS events
ENGINE = Distributed(cluster, database, events);

-- 조회 시 자동으로 모든 샤드에서 병렬 읽기
SELECT COUNT(*) FROM dist_events;  -- 빠름
```

메리트:

- **수평 확장**: 샤드 추가로 성능 증가
- **고가용성**: 샤드의 레플리카 중 1개 장애해도 동작
- **분산 처리**: 각 샤드에서 부분 집계 → 코디네이터에서 최종 병합

# 학습 자료

- [ClickHouse 공식 문서](https://clickhouse.com/docs/en/intro)
- [SQL 레퍼런스](https://clickhouse.com/docs/en/sql-reference)
- [MergeTree 엔진](https://clickhouse.com/docs/en/engines/table-engines/mergetree-family/mergetree)
- [데이터 타입](https://clickhouse.com/docs/en/sql-reference/data-types)
- [성능 최적화](https://clickhouse.com/docs/en/operations/optimizing-performance)

# 관련 문서

- [빠른 시작 (Docker Compose)](clickhouse-quickstart-kr.md) — 설치부터 첫 쿼리까지
- [SQL 기초](clickhouse-sql-basics-kr.md) — 쿼리 문법과 예제
- [Kafka 통합](clickhouse-kafka-integration-kr.md) — 실시간 데이터 수집
- [데이터 모델링](clickhouse-data-modeling-kr.md) — OLAP 설계 패턴
- [대안 솔루션 비교](clickhouse-vs-alternatives-kr.md) — ClickHouse vs Druid/Pinot/Redshift/BigQuery/ScyllaDB
