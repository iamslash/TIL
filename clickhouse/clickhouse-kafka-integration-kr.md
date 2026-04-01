# ClickHouse + Kafka 연동 — 실시간 이벤트 수집 완전 가이드

**작성일**: 2026-03-31
**대상**: 주니어 소프트웨어 엔지니어
**난이도**: 중급

---

## 목차

1. [왜 ClickHouse + Kafka인가?](#왜-clickhouse--kafka인가)
2. [아키텍처 개요](#아키텍처-개요)
3. [docker-compose로 로컬 환경 구성](#docker-compose로-로컬-환경-구성)
4. [Kafka 토픽 생성](#kafka-토픽-생성)
5. [Kafka Engine 테이블](#kafka-engine-테이블)
6. [Materialized View로 영속화](#materialized-view로-영속화)
7. [데이터 삽입 및 조회](#데이터-삽입-및-조회)
8. [Bronze/Silver/Gold 패턴](#bronzesilvergold-패턴)
9. [실전 예제: 사용자 행동 이벤트](#실전-예제-사용자-행동-이벤트)
10. [트러블슈팅](#트러블슈팅)

---

## 왜 ClickHouse + Kafka인가?

### 문제 상황

마이크로서비스들이 사용자 행동 이벤트를 발생시킨다:
- 사용자 로그인 → user-service
- 상품 클릭 → product-service
- 구매 완료 → order-service

각 서비스에서 직접 데이터베이스에 INSERT하면 **느리고 복잡하다**.

### 해결책: Kafka + ClickHouse

```
user-service ─┐
product-service ├─→ Kafka ─→ ClickHouse ─→ 실시간 분석
order-service ──┘
```

**장점**:
1. **느슨한 결합**: 각 서비스는 Kafka에만 이벤트를 보냄
2. **높은 처리량**: Kafka가 버퍼 역할 (초당 수십만 이벤트)
3. **실시간 분석**: ClickHouse에서 즉시 쿼리 가능
4. **다목적**: 같은 이벤트를 여러 곳에서 소비 가능 (분석, 머신러닝, 모니터링 등)

### 흐름도

```
1. 이벤트 발생 (JSON)
   {"user_id": 100, "action": "click", "product_id": 1001, "timestamp": "2026-03-31 10:00:00"}

2. Kafka에 발행
   Topic: user_events

3. ClickHouse가 소비
   - Kafka Engine 테이블에서 읽음
   - Materialized View로 자동 변환
   - 영속 테이블(MergeTree)에 저장

4. SQL로 쿼리
   SELECT COUNT(*) FROM user_events WHERE action = 'click' AND DATE(timestamp) = TODAY();
```

---

## 아키텍처 개요

### 핵심 개념 3가지

#### 1. Kafka Engine 테이블 (임시)
```sql
CREATE TABLE kafka_events (
    user_id UInt32,
    action String,
    product_id UInt32,
    timestamp DateTime
)
ENGINE = Kafka
...
```
- Kafka에서 직접 읽음 (메시지 큐처럼)
- 읽은 메시지는 ClickHouse에 **저장되지 않음** (메모리에만)
- 용도: Materialized View의 데이터 소스

#### 2. Materialized View (변환 + 저장)
```sql
CREATE MATERIALIZED VIEW events_mv
ENGINE = MergeTree(...)
AS
SELECT ... FROM kafka_events
```
- kafka_events 테이블에서 새 데이터를 감시
- 도착하는 데이터를 변환해 events_storage 테이블에 저장
- 자동으로 계속 실행됨

#### 3. 영속 테이블 (최종 저장소)
```sql
CREATE TABLE events_storage (
    user_id UInt32,
    action String,
    product_id UInt32,
    timestamp DateTime
)
ENGINE = MergeTree()
...
```
- Materialized View가 데이터를 저장
- 여기서 쿼리함

---

## docker-compose로 로컬 환경 구성

### docker-compose.yml

```yaml
version: '3.8'

services:
  # Kafka (KRaft 모드 — Zookeeper 불필요)
  kafka:
    image: confluentinc/cp-kafka:7.5.0
    container_name: clickhouse-kafka
    ports:
      - "9092:9092"      # PLAINTEXT (내부 통신)
      - "29092:29092"    # PLAINTEXT_HOST (Docker 호스트에서 접근)
    environment:
      # KRaft 설정 (Zookeeper 없이 실행)
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: 'broker,controller'
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_CONTROLLER_QUORUM_VOTERS: '1@kafka:29093'
      KAFKA_CONTROLLER_LISTENER_NAMES: 'CONTROLLER'
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: 'CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT'
      KAFKA_ADVERTISED_LISTENERS: 'PLAINTEXT://kafka:9092,PLAINTEXT_HOST://localhost:29092'
      KAFKA_INTER_BROKER_LISTENER_NAME: 'PLAINTEXT'
      KAFKA_LOG_DIRS: '/tmp/kraft-combined-logs'
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: 'true'
      KAFKA_OFFSETS_TOPIC_NUM_PARTITIONS: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      CLUSTER_ID: 'MkQkD3NiBkSRBZZ6k5NM_w'  # 고정값
    networks:
      - clickhouse-network
    healthcheck:
      test: kafka-topics --bootstrap-server localhost:9092 --list
      interval: 5s
      timeout: 10s
      retries: 10

  # ClickHouse
  clickhouse:
    image: clickhouse/clickhouse-server:latest
    container_name: clickhouse-server
    ports:
      - "8123:8123"   # HTTP API
      - "9000:9000"   # TCP native protocol
    environment:
      CLICKHOUSE_DB: default
      CLICKHOUSE_USER: default
      CLICKHOUSE_PASSWORD: ''
    volumes:
      - clickhouse_data:/var/lib/clickhouse
    networks:
      - clickhouse-network
    depends_on:
      kafka:
        condition: service_healthy

volumes:
  clickhouse_data:

networks:
  clickhouse-network:
    driver: bridge
```

### 실행 명령어

```bash
# docker-compose 시작
docker-compose -f docker-compose.yml up -d

# 로그 확인
docker-compose logs -f

# ClickHouse 접속
docker-compose exec clickhouse clickhouse-client

# Kafka 접속
docker-compose exec kafka bash
  # Kafka 내부에서:
  kafka-topics --bootstrap-server localhost:9092 --list
  kafka-console-producer --bootstrap-server localhost:9092 --topic user_events

# 종료
docker-compose down -v
```

### docker-compose.yml 상세 설명

#### Kafka LISTENER 설정 (흔한 실수!)

```
KAFKA_ADVERTISED_LISTENERS: 'PLAINTEXT://kafka:9092,PLAINTEXT_HOST://localhost:29092'
                            ↑ 컨테이너 내부용         ↑ Docker 호스트에서 접근
```

**INTERNAL vs EXTERNAL Listener**:
- **INTERNAL** (PLAINTEXT://kafka:9092): Kafka 컨테이너 내부 또는 같은 네트워크의 컨테이너가 접근
  - ClickHouse → Kafka: kafka:9092 사용
- **EXTERNAL** (PLAINTEXT_HOST://localhost:29092): Docker 호스트 또는 외부에서 접근
  - 로컬 PC의 kafka-console-producer: localhost:29092 사용

이 설정이 없으면:
```
Cannot assign requested address — "Can't get assignment"
```

---

## Kafka 토픽 생성

### 방법 1: docker-compose 내에서

```bash
docker-compose exec kafka kafka-topics --bootstrap-server localhost:9092 \
  --create \
  --topic user_events \
  --partitions 1 \
  --replication-factor 1 \
  --if-not-exists
```

### 방법 2: ClickHouse 설정으로 자동 생성

ClickHouse는 Kafka Engine 테이블을 만들 때 토픽이 없으면 자동 생성할 수 있다 (`auto_create_topics_enable=true`).

### 토픽 확인

```bash
docker-compose exec kafka kafka-topics --bootstrap-server localhost:9092 --list
```

출력:
```
user_events
__consumer_offsets
```

---

## Kafka Engine 테이블

### 기본 구조

```sql
CREATE TABLE kafka_events (
    user_id UInt32,
    action String,
    product_id UInt32,
    timestamp DateTime
)
ENGINE = Kafka
SETTINGS
  kafka_broker_list = 'kafka:9092',
  kafka_topic_list = 'user_events',
  kafka_group_id = 'clickhouse-consumer',
  kafka_format = 'JSONEachRow',
  kafka_skip_broken_messages = 1;
```

**설정 설명**:

| 설정 | 설명 |
|------|------|
| `kafka_broker_list` | Kafka 브로커 주소 (컨테이너 네트워크에서는 `kafka:9092`) |
| `kafka_topic_list` | 구독할 토픽 |
| `kafka_group_id` | Consumer Group ID (같은 그룹끼리 offset 공유) |
| `kafka_format` | 메시지 포맷 (`JSONEachRow`, `CSV`, `Avro` 등) |
| `kafka_skip_broken_messages` | 잘못된 메시지 스킵 (1 = 스킵, 0 = 에러 발생) |
| `kafka_num_consumers` | Consumer 스레드 수 (기본 1). Kafka 파티션 수 이하로 설정 |
| `kafka_thread_per_consumer` | 1 이면 Consumer 마다 독립 스레드. 0(기본)이면 하나의 스레드가 번갈아 처리 |
| `kafka_max_block_size` | 한 번에 가져오는 최대 메시지 수 (기본 65536) |
| `kafka_poll_timeout_ms` | poll 대기 시간 ms (기본 500) |

### Consumer 스레드 튜닝

처리량을 높이려면 `kafka_num_consumers` 를 올린다. Flink 의 병렬도와 같은 원리다 — **Kafka 파티션 수 이하**로 설정해야 의미가 있다.

```
파티션 6개, Consumer 3개 → 각 Consumer 가 2개씩 담당 (효율적)
파티션 6개, Consumer 6개 → 1:1 매핑 (최대 처리량)
파티션 3개, Consumer 6개 → 3개는 놀고 있음 (낭비)
```

```sql
CREATE TABLE kafka_user_events (...)
ENGINE = Kafka
SETTINGS
    kafka_broker_list = 'kafka:9092',
    kafka_topic_list = 'user_events',
    kafka_group_id = 'clickhouse_consumer',
    kafka_format = 'JSONEachRow',
    kafka_num_consumers = 6,             -- Consumer 6개
    kafka_thread_per_consumer = 1;       -- 각각 독립 스레드로 동작
```

> `kafka_thread_per_consumer = 1` 을 함께 설정해야 각 Consumer 가 진짜 독립 스레드로 동작한다. 기본값(0)이면 하나의 스레드가 여러 Consumer 를 번갈아 처리하므로 병렬 효과가 제한된다.

### Kafka Connect 가 필요 없다

일반적으로 Kafka → DB 파이프라인을 구축하려면 **Kafka Connect** 라는 별도 컴포넌트가 필요하다. 하지만 ClickHouse 는 Kafka Engine 을 내장하고 있어서 Kafka Connect 없이 직접 Consumer 역할을 한다.

```
일반적인 구성:
  Kafka → Kafka Connect (JDBC Sink Connector) → PostgreSQL/MySQL
  Kafka → Kafka Connect (ClickHouse Sink Connector) → ClickHouse
  → 별도 프로세스 운영 필요, 커넥터 설정/모니터링 부담

ClickHouse 구성:
  Kafka → ClickHouse (Kafka Engine + MV)
  → SQL 만으로 완성, 별도 프로세스 불필요
```

| 항목 | Kafka Connect | ClickHouse Kafka Engine |
|------|--------------|------------------------|
| **추가 프로세스** | 필요 (Connect 클러스터) | 불필요 |
| **설정 방식** | JSON/REST API | SQL DDL |
| **변환 로직** | SMT (Single Message Transform) | MV 의 SELECT 쿼리 |
| **모니터링** | Connect REST API | ClickHouse system 테이블 |
| **스키마 관리** | Schema Registry 권장 | ClickHouse 테이블 스키마로 관리 |

### 실제 ClickHouse 쿼리에서

```sql
-- Kafka 테이블 생성
CREATE TABLE kafka_user_events (
    user_id UInt32,
    action String,
    product_id UInt32,
    created_at DateTime
)
ENGINE = Kafka
SETTINGS
  kafka_broker_list = 'kafka:9092',
  kafka_topic_list = 'user_events',
  kafka_group_id = 'clickhouse_consumer_group_1',
  kafka_format = 'JSONEachRow',
  kafka_skip_broken_messages = 1;

-- 데이터 조회 (메모리에서만 읽음, 저장되지 않음)
SELECT * FROM kafka_user_events LIMIT 10;
```

---

## Materialized View로 영속화

### 문제점

Kafka Engine 테이블에서 읽은 데이터는 **저장되지 않는다**. 다시 접근하면 새 메시지만 보인다.

```sql
-- 첫 번째 조회: 메시지 10개 읽음
SELECT * FROM kafka_user_events;  -- 10개 반환

-- 두 번째 조회: 같은 메시지는 안 보임 (이미 소비됨)
SELECT * FROM kafka_user_events;  -- 0개 반환 (또는 새 메시지만)
```

### 해결책: Materialized View

```sql
-- 1. 영속 테이블 생성
CREATE TABLE user_events (
    user_id UInt32,
    action String,
    product_id UInt32,
    created_at DateTime
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)
ORDER BY created_at;

-- 2. Materialized View 생성 (자동 전환 + 저장)
CREATE MATERIALIZED VIEW user_events_mv
TO user_events
AS
SELECT
    user_id,
    action,
    product_id,
    created_at
FROM kafka_user_events;
```

### 동작 원리

```
Kafka 토픽에 새 메시지 도착
    ↓
kafka_user_events 테이블이 감지
    ↓
user_events_mv이 자동 실행 (SELECT * FROM kafka_user_events)
    ↓
데이터를 user_events 테이블에 INSERT
    ↓
영구 저장됨! 🎉
```

---

## 데이터 삽입 및 조회

### Kafka에 데이터 발송

#### 방법 1: kafka-console-producer 사용

```bash
docker-compose exec kafka kafka-console-producer \
  --bootstrap-server localhost:9092 \
  --topic user_events
```

그러면 프롬프트가 나타난다. JSON 메시지를 한 줄씩 입력:

```json
{"user_id": 100, "action": "click", "product_id": 1001, "created_at": "2026-03-31 10:00:00"}
{"user_id": 101, "action": "view", "product_id": 1002, "created_at": "2026-03-31 10:01:00"}
{"user_id": 100, "action": "purchase", "product_id": 1001, "created_at": "2026-03-31 10:02:00"}
```

(Ctrl+C로 종료)

#### 방법 2: 파일에서 일괄 삽입

```bash
cat > events.jsonl << 'EOF'
{"user_id": 100, "action": "click", "product_id": 1001, "created_at": "2026-03-31 10:00:00"}
{"user_id": 101, "action": "view", "product_id": 1002, "created_at": "2026-03-31 10:01:00"}
{"user_id": 100, "action": "purchase", "product_id": 1001, "created_at": "2026-03-31 10:02:00"}
EOF

docker-compose exec -T kafka kafka-console-producer \
  --bootstrap-server localhost:9092 \
  --topic user_events < events.jsonl
```

#### 방법 3: Python 스크립트로 발송

```python
from kafka import KafkaProducer
import json
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers=['localhost:29092'],  # 호스트 포트!
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

for i in range(100):
    event = {
        'user_id': 100 + (i % 5),
        'action': ['click', 'view', 'purchase'][i % 3],
        'product_id': 1001 + (i % 10),
        'created_at': datetime.now().isoformat()
    }
    producer.send('user_events', event)
    print(f"Sent: {event}")

producer.flush()
producer.close()
```

### ClickHouse에서 조회

```bash
# ClickHouse 접속
docker-compose exec clickhouse clickhouse-client

# 쿼리 실행
SELECT * FROM user_events LIMIT 10;

SELECT
    action,
    COUNT(*) as count
FROM user_events
GROUP BY action;

SELECT
    user_id,
    COUNT(*) as event_count
FROM user_events
GROUP BY user_id
ORDER BY event_count DESC;
```

---

## Bronze/Silver/Gold 패턴

**Bronze/Silver/Gold**는 Databricks Medallion Architecture에서 나온 패턴. ClickHouse에서도 적용 가능하다.

### 비유로 이해하기

데이팅 서비스의 데이터 엔지니어라고 상상하자. 유저가 앱에서 행동할 때마다 이벤트가 Kafka 로 들어온다. 정상 데이터와 버그로 생긴 쓰레기 데이터가 섞여서 초당 수천 건 들어온다.

```json
{"user_id": 42, "action": "swipe_right", "product_id": 7, "created_at": "2026-04-01 10:05:23"}
{"user_id": 0, "action": "", "product_id": 0, "created_at": "2026-04-01 10:05:24"}
{"user_id": 99, "action": "purchase", "product_id": 3, "created_at": "2026-04-01 10:05:25"}
```

이 데이터를 세 단계로 처리한다:

- **Bronze = 창고**: 택배가 도착하면 내용물을 확인하지 않고 일단 전부 넣는다
- **Silver = 검수대**: 창고에서 꺼내 망가진 건 버리고, 라벨을 통일하고, 분류 태그를 붙인다
- **Gold = 진열대**: 검수된 물건을 조합해서 바로 팔 수 있는 완성품(대시보드용 집계)을 만든다

### 3단계 구조

```
Kafka (초당 수천 건의 원시 이벤트)
  │
  ▼
Bronze (원본 보존, 필터링 없음)        ← 창고
  │
  ▼ MV: 유효성 검증, 컬럼 정규화
Silver (정제된 데이터)                 ← 검수 완료
  │
  ▼ MV: GROUP BY 집계
Gold (대시보드용 요약)                 ← 진열대
```

각 단계가 MV 로 연결되어 있으므로 **Kafka 에 이벤트가 들어오면 Bronze → Silver → Gold 가 자동으로 연쇄 업데이트**된다. 배치 작업을 돌릴 필요가 없다.

### 왜 Bronze 에 쓰레기 데이터를 남기는가?

Silver 에서 `user_id=0` 같은 쓰레기를 걸러내지만, Bronze 에는 원본이 그대로 남는다. 이것은 의도된 것이다:

- **정제 로직 버그 대응** — Silver MV 의 WHERE 조건이 잘못되어 정상 데이터를 버렸다면? Bronze 에 원본이 있으니 MV 를 고치고 Silver 를 재생성하면 된다
- **요구사항 변경** — "user_id=0 인 이벤트도 분석해야 해" 라는 요청이 나중에 올 수 있다
- **감사/디버깅** — "3월 15일에 이상한 이벤트가 뭐였지?" 할 때 Bronze 를 조회

> Bronze 는 **보험**이다. 공간은 좀 더 쓰지만, 원본을 버리면 되돌릴 수 없다.

### Gold 의 SummingMergeTree 동작

Gold 에 같은 날, 같은 action 의 이벤트가 들어오면 최종적으로 1행으로 합산된다. 단, 즉시 1행은 아니다:

```
Silver 에 100건 INSERT (같은 날, 같은 action)
  → MV 가 GROUP BY → Gold 에 1행 INSERT (count=100)

또 50건 INSERT
  → MV 가 GROUP BY → Gold 에 1행 INSERT (count=50)

이 시점에 Gold 에 2행 존재:
  (2026-04-01, swipe_right, 100)
  (2026-04-01, swipe_right, 50)

SummingMergeTree 백그라운드 머지 후 → 1행:
  (2026-04-01, swipe_right, 150)
```

머지 전에 정확한 값이 필요하면 `FINAL` 또는 `SUM()` 을 사용한다.

### MV 체인이 없다면?

```
MV 체인 없이:
  Kafka → Go/Python Consumer → Bronze INSERT
  cron 매 5분 → Bronze SELECT → 정제 → Silver INSERT
  cron 매 5분 → Silver SELECT → 집계 → Gold INSERT

MV 체인으로:
  Kafka → Bronze → Silver → Gold (자동, 코드 없음)
```

| 항목 | MV 체인 | 직접 코드 |
|------|---------|----------|
| **코드량** | SQL 만 | Consumer + cron + 정제 + 집계 로직 |
| **지연** | 거의 실시간 (INSERT 즉시 전파) | cron 주기만큼 지연 (5분~1시간) |
| **장애 포인트** | ClickHouse 하나 | Consumer, cron, 네트워크 등 여러 곳 |
| **운영** | DDL 만 관리 | 코드 배포, 모니터링, 재시작 필요 |

> MV 체인의 장점: **SQL 선언만으로 실시간 파이프라인이 완성**된다.

### 실전 구현

```sql
-- ============================================
-- Bronze: Kafka 원본 데이터 (그대로 저장)
-- ============================================
CREATE TABLE kafka_events_raw (
    user_id UInt32,
    action String,
    product_id UInt32,
    created_at DateTime
)
ENGINE = Kafka
SETTINGS
  kafka_broker_list = 'kafka:9092',
  kafka_topic_list = 'user_events',
  kafka_group_id = 'clickhouse_bronze',
  kafka_format = 'JSONEachRow';

CREATE TABLE bronze_events (
    user_id UInt32,
    action String,
    product_id UInt32,
    created_at DateTime,
    ingested_at DateTime DEFAULT now()  -- 수집 시간
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)
ORDER BY created_at;

CREATE MATERIALIZED VIEW bronze_events_mv
TO bronze_events
AS
SELECT
    user_id,
    action,
    product_id,
    created_at
FROM kafka_events_raw;

-- ============================================
-- Silver: 정제 & 표준화
-- ============================================
CREATE TABLE silver_events (
    user_id UInt32,
    action LowCardinality(String),  -- 카테고리화
    product_id UInt32,
    year UInt16,
    month UInt8,
    day UInt8,
    hour UInt8,
    created_at DateTime
)
ENGINE = MergeTree()
PARTITION BY (year, month)
ORDER BY (created_at, user_id);

CREATE MATERIALIZED VIEW silver_events_mv
TO silver_events
AS
SELECT
    user_id,
    action,
    product_id,
    toYear(created_at) as year,
    toMonth(created_at) as month,
    toDayOfMonth(created_at) as day,
    toHour(created_at) as hour,
    created_at
FROM bronze_events
WHERE user_id > 0 AND product_id > 0;  -- 유효성 검증

-- ============================================
-- Gold: 분석용 요약 데이터
-- ============================================
CREATE TABLE gold_daily_actions (
    date Date,
    action LowCardinality(String),
    action_count UInt64,
    unique_users UInt64
)
ENGINE = SummingMergeTree(action_count)
PARTITION BY toYYYYMM(date)
ORDER BY (date, action);

CREATE MATERIALIZED VIEW gold_daily_actions_mv
TO gold_daily_actions
AS
SELECT
    toDate(created_at) as date,
    action,
    COUNT(*) as action_count,
    uniq(user_id) as unique_users
FROM silver_events
GROUP BY date, action;

-- 사용자별 일별 활동 Gold 테이블
CREATE TABLE gold_user_daily_activity (
    date Date,
    user_id UInt32,
    click_count UInt64,
    view_count UInt64,
    purchase_count UInt64
)
ENGINE = SummingMergeTree(click_count, view_count, purchase_count)
PARTITION BY toYYYYMM(date)
ORDER BY (date, user_id);

CREATE MATERIALIZED VIEW gold_user_daily_mv
TO gold_user_daily_activity
AS
SELECT
    toDate(created_at) as date,
    user_id,
    countIf(action = 'click') as click_count,
    countIf(action = 'view') as view_count,
    countIf(action = 'purchase') as purchase_count
FROM silver_events
GROUP BY date, user_id;
```

### 쿼리 예제

```sql
-- Bronze: 원본 데이터 확인 (디버깅용)
SELECT * FROM bronze_events WHERE user_id = 100 LIMIT 5;

-- Silver: 정제 데이터 (중간 분석)
SELECT * FROM silver_events WHERE action = 'purchase' LIMIT 5;

-- Gold: 최종 분석 (빠름!)
SELECT * FROM gold_daily_actions WHERE date = '2026-03-31';

SELECT
    user_id,
    click_count,
    view_count,
    purchase_count
FROM gold_user_daily_activity
WHERE date = '2026-03-31'
ORDER BY purchase_count DESC
LIMIT 10;
```

---

## 실전 예제: 사용자 행동 이벤트

### 전체 흐름

```sql
-- 1. Kafka 원본 테이블 (임시, 메모리용)
CREATE TABLE kafka_user_actions (
    user_id UInt32,
    action String,      -- 'login', 'view_product', 'add_cart', 'purchase'
    product_id UInt32,
    product_price Float64,
    session_id String,
    created_at DateTime
)
ENGINE = Kafka
SETTINGS
  kafka_broker_list = 'kafka:9092',
  kafka_topic_list = 'user_actions',
  kafka_group_id = 'clickhouse_analytics',
  kafka_format = 'JSONEachRow';

-- 2. Bronze: 원본 저장
CREATE TABLE bronze_user_actions (
    user_id UInt32,
    action String,
    product_id UInt32,
    product_price Float64,
    session_id String,
    created_at DateTime,
    ingested_at DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)
ORDER BY created_at;

CREATE MATERIALIZED VIEW bronze_user_actions_mv
TO bronze_user_actions
AS
SELECT * FROM kafka_user_actions;

-- 3. Silver: 정제
CREATE TABLE silver_user_actions (
    user_id UInt32,
    action LowCardinality(String),
    product_id UInt32,
    product_price Float64,
    session_id String,
    hour UInt8,
    day_of_week UInt8,
    created_at DateTime
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)
ORDER BY (created_at, user_id);

CREATE MATERIALIZED VIEW silver_user_actions_mv
TO silver_user_actions
AS
SELECT
    user_id,
    action,
    product_id,
    product_price,
    session_id,
    toHour(created_at) as hour,
    toDayOfWeek(created_at) as day_of_week,
    created_at
FROM bronze_user_actions
WHERE user_id > 0;

-- 4. Gold: 사용자 세션 분석
CREATE TABLE gold_user_sessions (
    date Date,
    user_id UInt32,
    session_id String,
    login_count UInt64,
    view_count UInt64,
    cart_count UInt64,
    purchase_count UInt64,
    total_spent Float64,
    session_duration_seconds UInt32
)
ENGINE = SummingMergeTree(login_count, view_count, cart_count, purchase_count, total_spent)
PARTITION BY toYYYYMM(date)
ORDER BY (date, user_id);

CREATE MATERIALIZED VIEW gold_user_sessions_mv
TO gold_user_sessions
AS
SELECT
    toDate(created_at) as date,
    user_id,
    session_id,
    countIf(action = 'login') as login_count,
    countIf(action = 'view_product') as view_count,
    countIf(action = 'add_cart') as cart_count,
    countIf(action = 'purchase') as purchase_count,
    sumIf(product_price, action = 'purchase') as total_spent,
    CAST(dateDiff('second', min(created_at), max(created_at)) AS UInt32) as session_duration_seconds
FROM silver_user_actions
GROUP BY date, user_id, session_id;

-- 5. Gold: 시간대별 트래픽
CREATE TABLE gold_traffic_hourly (
    hour DateTime,
    action LowCardinality(String),
    action_count UInt64,
    unique_users UInt64
)
ENGINE = SummingMergeTree(action_count)
PARTITION BY toYYYYMM(toDate(hour))
ORDER BY hour;

CREATE MATERIALIZED VIEW gold_traffic_hourly_mv
TO gold_traffic_hourly
AS
SELECT
    toStartOfHour(created_at) as hour,
    action,
    COUNT(*) as action_count,
    uniq(user_id) as unique_users
FROM silver_user_actions
GROUP BY hour, action;
```

### 데이터 발송

```bash
cat > user_actions.jsonl << 'EOF'
{"user_id": 100, "action": "login", "product_id": 0, "product_price": 0, "session_id": "sess_1001", "created_at": "2026-03-31 10:00:00"}
{"user_id": 100, "action": "view_product", "product_id": 1001, "product_price": 99.99, "session_id": "sess_1001", "created_at": "2026-03-31 10:01:00"}
{"user_id": 100, "action": "add_cart", "product_id": 1001, "product_price": 99.99, "session_id": "sess_1001", "created_at": "2026-03-31 10:02:00"}
{"user_id": 100, "action": "view_product", "product_id": 1002, "product_price": 149.99, "session_id": "sess_1001", "created_at": "2026-03-31 10:03:00"}
{"user_id": 100, "action": "purchase", "product_id": 1001, "product_price": 99.99, "session_id": "sess_1001", "created_at": "2026-03-31 10:04:00"}
{"user_id": 101, "action": "login", "product_id": 0, "product_price": 0, "session_id": "sess_1002", "created_at": "2026-03-31 10:05:00"}
{"user_id": 101, "action": "view_product", "product_id": 1002, "product_price": 149.99, "session_id": "sess_1002", "created_at": "2026-03-31 10:06:00"}
{"user_id": 101, "action": "purchase", "product_id": 1002, "product_price": 149.99, "session_id": "sess_1002", "created_at": "2026-03-31 10:07:00"}
EOF

docker-compose exec -T kafka kafka-console-producer \
  --bootstrap-server localhost:9092 \
  --topic user_actions < user_actions.jsonl
```

### 분석 쿼리

```sql
-- 사용자별 세션 분석
SELECT
    user_id,
    COUNT(*) as session_count,
    SUM(purchase_count) as total_purchases,
    SUM(total_spent) as lifetime_value
FROM gold_user_sessions
GROUP BY user_id
ORDER BY lifetime_value DESC;

-- 시간대별 트래픽
SELECT
    hour,
    action,
    action_count,
    unique_users
FROM gold_traffic_hourly
ORDER BY hour DESC, action_count DESC;

-- 구매 전환율 (purchase / view_product)
SELECT
    toDate(hour) as date,
    ROUND(countIf(action = 'purchase') / countIf(action = 'view_product') * 100, 2) as conversion_rate
FROM gold_traffic_hourly
GROUP BY date
ORDER BY date DESC;
```

---

## 트러블슈팅

### 1. "Can't get assignment"

**증상**: Kafka 접속 실패

**원인**: LISTENER 설정 오류

```yaml
# 잘못됨 ❌
KAFKA_ADVERTISED_LISTENERS: 'PLAINTEXT://localhost:9092'

# 올바름 ✅
KAFKA_ADVERTISED_LISTENERS: 'PLAINTEXT://kafka:9092,PLAINTEXT_HOST://localhost:29092'
```

**해결책**:
1. docker-compose.yml 수정
2. `docker-compose down -v` (완전 제거)
3. `docker-compose up -d` (재시작)

### 2. "Broker is not available" in ClickHouse

**증상**: ClickHouse가 Kafka에 연결 불가

**원인**: 네트워크 또는 호스트명 오류

```sql
-- 잘못됨 ❌
CREATE TABLE ... ENGINE = Kafka
SETTINGS kafka_broker_list = 'localhost:9092';

-- 올바름 ✅
CREATE TABLE ... ENGINE = Kafka
SETTINGS kafka_broker_list = 'kafka:9092';
```

(ClickHouse와 Kafka가 같은 docker-compose 네트워크에 있으면 호스트명 `kafka:9092` 사용)

### 3. Materialized View가 자동 실행 안 됨

**원인**: Consumer Group이 이미 모든 메시지를 소비함

**해결책**:
```bash
# Consumer Group 초기화
docker-compose exec kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --group clickhouse_consumer_group_1 \
  --reset-offsets \
  --to-earliest \
  --execute \
  --all-topics

# 그 후 새로 메시지 발송
```

### 4. "Deserialize error" — 메시지 포맷 오류

**원인**: Kafka에 보낸 JSON이 테이블 스키마와 안 맞음

```json
// 테이블에 user_id가 UInt32인데 문자열이 옴 ❌
{"user_id": "abc", "action": "click"}

// 올바름 ✅
{"user_id": 100, "action": "click"}
```

**해결책**:
```sql
-- kafka_skip_broken_messages = 1로 설정 (잘못된 메시지 무시)
CREATE TABLE ... ENGINE = Kafka
SETTINGS kafka_skip_broken_messages = 1;
```

### 5. Consumer Lag이 계속 증가

**원인**: Kafka 토픽에 메시지가 들어오는 속도 > ClickHouse 처리 속도

**확인 방법**:
```bash
docker-compose exec kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --group clickhouse_consumer_group_1 \
  --describe
```

**출력 예**:
```
GROUP                   TOPIC       PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
clickhouse_consumer_group_1  user_events 0         100             200             100
```

LAG가 100 → Materialized View 처리가 느림

**해결책**:
1. Kafka 파티션 수 증가
2. Materialized View 쿼리 최적화
3. ClickHouse 메모리/CPU 증가

---

## 정리

| 항목 | 설명 |
|------|------|
| **Kafka Engine** | 실시간으로 토픽 구독 |
| **Materialized View** | 자동으로 변환 + 저장 |
| **Bronze** | 원본 데이터 (그대로) |
| **Silver** | 정제 데이터 |
| **Gold** | 분석 완료 데이터 |
| **LISTENER** | INTERNAL (kafka:9092) + EXTERNAL (localhost:29092) |

---

## 다음 단계

1. docker-compose.yml로 로컬 환경 구축
2. Kafka 토픽 생성, 메시지 발송
3. ClickHouse에서 Kafka Engine 테이블 생성
4. Materialized View로 자동 저장 설정
5. 분석 쿼리 작성
6. `clickhouse-sql-basics-kr.md`의 고급 쿼리 적용

