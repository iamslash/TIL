# ClickHouse SQL 기초 — 주니어 개발자를 위한 실전 가이드

**작성일**: 2026-03-31
**대상**: 주니어 소프트웨어 엔지니어
**난이도**: 초급~중급

---

## 목차

1. [ClickHouse란?](#clickhouse란)
2. [데이터 타입](#데이터-타입)
3. [테이블 엔진](#테이블-엔진)
4. [데이터 삽입 (INSERT)](#데이터-삽입-insert)
5. [SELECT 심화](#select-심화)
6. [집계 함수 (Aggregate Functions)](#집계-함수-aggregate-functions)
7. [윈도우 함수 (Window Functions)](#윈도우-함수-window-functions)
8. [Materialized View](#materialized-view)
9. [파티션과 TTL](#파티션과-ttl)
10. [실전 예제: 이커머스 분석](#실전-예제-이커머스-분석)

---

## ClickHouse란?

ClickHouse는 **OLAP(온라인 분석 처리)** 데이터베이스다. OLTP(트랜잭션)와는 다르다.

| 항목 | ClickHouse | 일반 SQL DB |
|------|-----------|-----------|
| 용도 | 대량 데이터 분석 | 실시간 거래 |
| 쓰기 | 배치 또는 스트림 | 단건 INSERT 빈번 |
| 읽기 | 열 기반 (컬럼) | 행 기반 (로우) |
| 압축 | 매우 효율적 | 보통 |
| 스키마 | 고정 | 유연 |

**핵심**: 같은 컬럼끼리 저장되므로 수조 건의 데이터도 빠르게 집계할 수 있다.

---

## 데이터 타입

### 기본 타입

```sql
-- 정수형
UInt8, UInt16, UInt32, UInt64  -- 부호 없음 (0 이상)
Int8, Int16, Int32, Int64       -- 부호 있음 (음수 가능)

-- 실수형
Float32, Float64                 -- 부동소수점

-- 문자열
String                           -- 가변 길이 (권장)
FixedString(N)                   -- 고정 길이 (성능 우위, 지금은 String 써도 됨)

-- 날짜/시간
Date                             -- YYYY-MM-DD (1970-01-01 ~ 2105-12-31)
DateTime                         -- YYYY-MM-DD HH:MM:SS (초 단위)
DateTime64(3)                    -- 밀리초 단위 (3 = 10^-3)

-- Boolean
UInt8                            -- ClickHouse는 Boolean 타입이 없다. 0/1로 저장

-- 배열
Array(String)                    -- 문자열 배열
Array(UInt32)                    -- 정수 배열
Array(Tuple(String, UInt32))     -- 복잡한 배열

-- Enum (카테고리)
Enum8('low' = 1, 'high' = 2)    -- 작은 범위
Enum16('bronze' = 1, 'silver' = 2, 'gold' = 3)

-- JSON (거의 쓰지 말 것 — 느림)
String                           -- JSON은 String으로 저장, 필요시 extractJSON 함수 사용
```

### LowCardinality — 반복되는 값을 위한 최적화

```sql
CREATE TABLE products (
    id UInt32,
    category LowCardinality(String),  -- "electronics", "clothing" 등 반복되는 값
    color LowCardinality(String)      -- "red", "blue", "black" 등
)
ENGINE = MergeTree()
ORDER BY id;
```

**언제 쓸까?**
- 값의 종류가 많지 않은 경우 (100개 이하가 목표)
- 예: 국가 코드, 카테고리, 상태(active/inactive), 색상

**효과**: 메모리 사용량 10배 이상 감소, 쿼리 빠름.

---

## 테이블 엔진

### 1. MergeTree — 기본 엔진 (가장 많이 쓰임)

```sql
CREATE TABLE orders (
    order_id UInt64,
    user_id UInt32,
    amount Float64,
    created_at DateTime,
    status String
)
ENGINE = MergeTree()
ORDER BY (user_id, order_id);  -- ORDER BY는 필수
```

**특징**:
- 순서대로 저장하고 압축 (빠른 검색)
- `ORDER BY` 컬럼으로 필터링하면 매우 빠름
- **언제**: 일반적인 로그, 이벤트, 시계열 데이터

### 2. ReplacingMergeTree — 최신 버전만 유지

```sql
CREATE TABLE user_state (
    user_id UInt32,
    name String,
    email String,
    status String,
    version UInt32  -- 버전 번호 (필수)
)
ENGINE = ReplacingMergeTree(version)
ORDER BY user_id;
```

**언제 쓸까?**: 같은 user_id에 대해 상태 업데이트가 반복되는 경우.
- 예: 사용자 프로필 변경 → 같은 user_id의 row 여러 개 존재
- version이 높은 row만 읽으면 최신 상태를 알 수 있음

```sql
-- 삽입 (버전 1)
INSERT INTO user_state VALUES (1, 'Alice', 'alice@example.com', 'active', 1);
INSERT INTO user_state VALUES (1, 'Alice', 'alice.new@example.com', 'active', 2);  -- 이메일 변경

-- 읽기: FINAL을 붙이면 최신 버전만 조회
SELECT * FROM user_state FINAL WHERE user_id = 1;
-- 결과: user_id=1, email='alice.new@example.com', version=2
```

### 3. SummingMergeTree — 미리 합산된 데이터 저장

```sql
CREATE TABLE daily_sales (
    date Date,
    product_id UInt32,
    quantity UInt64,  -- 합산할 컬럼
    revenue Float64   -- 합산할 컬럼
)
ENGINE = SummingMergeTree(quantity, revenue)
ORDER BY (date, product_id);
```

**언제 쓸까?**: 일별/시간별 통계를 미리 집계해 저장하고 싶을 때.
- 예: 일별 판매량, 일별 매출
- 같은 (date, product_id)를 여러 번 삽입하면 자동으로 합산됨

```sql
-- 두 번 삽입
INSERT INTO daily_sales VALUES ('2026-03-31', 100, 5, 50.00);
INSERT INTO daily_sales VALUES ('2026-03-31', 100, 3, 30.00);

-- 자동 합산 (optimize 후)
SELECT * FROM daily_sales;
-- 결과: date='2026-03-31', product_id=100, quantity=8, revenue=80.00
```

### 4. AggregatingMergeTree — 복잡한 집계 함수 저장

```sql
CREATE TABLE user_metrics (
    user_id UInt32,
    date Date,
    visit_count AggregateFunction(sum, UInt64),
    unique_pages AggregateFunction(uniq, String)
)
ENGINE = AggregatingMergeTree()
ORDER BY (date, user_id);
```

**언제 쓸까?**: `count(distinct ...)` 같은 복잡한 집계를 미리 계산할 때.
- 일일 사용자 수, 방문 수 등을 Materialized View로 자동 갱신

---

## 데이터 삽입 (INSERT)

### 방법 1: VALUES

```sql
INSERT INTO orders VALUES
    (1001, 100, 99.99, '2026-03-31 10:00:00', 'completed'),
    (1002, 101, 149.99, '2026-03-31 11:00:00', 'pending');
```

### 방법 2: SELECT (다른 테이블에서 복사)

```sql
INSERT INTO orders
SELECT * FROM orders_backup WHERE created_at > '2026-03-01';
```

### 방법 3: CSV 파일 읽기

```bash
# orders.csv 파일:
# 1003,102,199.99,2026-03-31 12:00:00,completed
# 1004,103,299.99,2026-03-31 13:00:00,pending

# ClickHouse CLI에서:
clickhouse-client -q "INSERT INTO orders FORMAT CSV" < orders.csv
```

### 방법 4: JSON 파일 읽기

```bash
# orders.json 파일:
# {"order_id":1005,"user_id":104,"amount":399.99,"created_at":"2026-03-31 14:00:00","status":"completed"}

clickhouse-client -q "INSERT INTO orders FORMAT JSONEachRow" < orders.json
```

---

## SELECT 심화

### WHERE — 조건 필터링

```sql
-- ORDER BY 컬럼으로 필터링 (빠름!)
SELECT * FROM orders WHERE user_id = 100;

-- 날짜 범위 필터링
SELECT * FROM orders WHERE created_at >= '2026-03-01' AND created_at < '2026-04-01';

-- IN 연산자
SELECT * FROM orders WHERE status IN ('completed', 'pending');

-- LIKE (정규식)
SELECT * FROM orders WHERE status LIKE '%ed';  -- 'completed', 'pending' 등
```

### GROUP BY — 그룹화

```sql
-- 사용자별 주문 건수
SELECT user_id, COUNT(*) as order_count
FROM orders
GROUP BY user_id;

-- 날짜별 총 매출
SELECT
    toDate(created_at) as order_date,
    SUM(amount) as daily_revenue
FROM orders
GROUP BY order_date;

-- 복합 그룹화
SELECT
    toDate(created_at) as order_date,
    status,
    COUNT(*) as count,
    SUM(amount) as total
FROM orders
GROUP BY order_date, status;
```

### HAVING — 그룹 조건 필터링

```sql
-- 주문이 10건 이상인 사용자만 조회
SELECT
    user_id,
    COUNT(*) as order_count
FROM orders
GROUP BY user_id
HAVING order_count >= 10;
```

### ORDER BY — 정렬

```sql
-- 내림차순 정렬
SELECT * FROM orders ORDER BY amount DESC LIMIT 10;

-- 여러 컬럼 정렬
SELECT * FROM orders
ORDER BY status ASC, created_at DESC;

-- 함수 적용 후 정렬
SELECT * FROM orders
ORDER BY toDate(created_at) DESC;
```

### LIMIT — 행 수 제한

```sql
-- 상위 10개만
SELECT * FROM orders LIMIT 10;

-- 10번째부터 20개 (OFFSET은 느리므로 주의)
SELECT * FROM orders LIMIT 10 OFFSET 10;
```

### 날짜 함수

```sql
-- 날짜/시간 추출
SELECT
    created_at,
    toDate(created_at) as date,           -- 날짜만
    toYear(created_at) as year,
    toMonth(created_at) as month,
    toDayOfMonth(created_at) as day,
    toHour(created_at) as hour
FROM orders;

-- 날짜 계산
SELECT
    created_at,
    created_at + INTERVAL 1 DAY as next_day,
    dateDiff('day', created_at, now()) as days_ago
FROM orders;

-- 주간/월간 그룹화
SELECT
    toStartOfWeek(created_at) as week,
    COUNT(*) as weekly_orders
FROM orders
GROUP BY week;
```

---

## 집계 함수 (Aggregate Functions)

```sql
SELECT
    COUNT(*) as total_rows,              -- 전체 행 수
    COUNT(DISTINCT user_id) as users,    -- 고유 사용자 수
    SUM(amount) as total_revenue,        -- 합
    AVG(amount) as avg_order_value,      -- 평균
    MIN(amount) as min_order,            -- 최소값
    MAX(amount) as max_order,            -- 최대값

    -- 백분위수
    quantile(0.5)(amount) as median,     -- 중앙값 (50%)
    quantile(0.95)(amount) as p95,       -- 95 백분위수
    quantile(0.99)(amount) as p99,       -- 99 백분위수

    -- 고유값 개수 (근사)
    uniq(user_id) as approx_users,
    uniq(status) as unique_statuses,

    -- 최대값과 해당 행의 다른 컬럼
    argMax(status, amount) as status_of_max_amount,

    -- 배열로 모든 값 수집 (주의: 메모리 사용 많음)
    groupArray(status) as all_statuses
FROM orders;
```

**주의사항**:
- `uniq()`: 정확하지 않음 (근사값). 정확한 값이 필요하면 `COUNT(DISTINCT)` 사용 (느림)
- `groupArray()`: 메모리 많이 사용. 값이 적을 때만 사용

---

## 윈도우 함수 (Window Functions)

윈도우 함수는 그룹 전체를 봤을 때 현재 행의 위치나 순위를 구한다.

### ROW_NUMBER — 순위 매기기

```sql
SELECT
    order_id,
    user_id,
    amount,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY amount DESC) as rank_in_user
FROM orders;
```

결과:
```
order_id | user_id | amount | rank_in_user
1001     | 100     | 99.99  | 1
1003     | 100     | 49.99  | 2
1002     | 101     | 149.99 | 1
```

### RANK — 중복 허용 순위

```sql
SELECT
    order_id,
    user_id,
    amount,
    RANK() OVER (ORDER BY amount DESC) as global_rank
FROM orders;
```

ROW_NUMBER와 다른 점: 같은 amount면 같은 rank 부여.

### LAG/LEAD — 이전/다음 행

```sql
SELECT
    order_id,
    user_id,
    created_at,
    amount,
    LAG(amount) OVER (PARTITION BY user_id ORDER BY created_at) as prev_amount,
    LEAD(amount) OVER (PARTITION BY user_id ORDER BY created_at) as next_amount
FROM orders;
```

결과:
```
order_id | user_id | amount | prev_amount | next_amount
1001     | 100     | 99.99  | NULL        | 49.99
1003     | 100     | 49.99  | 99.99       | NULL
```

**활용**: 구매 주기 분석, 일일 변화량 계산 등.

---

## Materialized View

### Materialized View 가 뭔가

한마디로: **"A 테이블에 데이터가 들어오면 → 변환해서 → B 테이블에 자동 INSERT 해줘"**

Materialized View 자체는 데이터를 갖고 있지 않다. 결과가 저장되는 곳은 `TO` 뒤의 목적지 테이블이다. Materialized View 는 그 사이의 **자동화된 파이프**일 뿐이다.

### 일반 View vs Materialized View

```sql
-- 일반 View: 쿼리할 때마다 매번 원본 테이블을 전체 스캔
CREATE VIEW daily_sales AS
SELECT toDate(created_at) AS day, sum(amount) AS total
FROM orders GROUP BY day;

SELECT * FROM daily_sales;  -- 매번 orders 전체를 스캔한다 (느림)
```

```sql
-- Materialized View: 데이터가 INSERT 될 때 자동으로 미리 계산해서 저장
CREATE TABLE daily_sales_table (day Date, total UInt64)
ENGINE = SummingMergeTree() ORDER BY day;

CREATE MATERIALIZED VIEW orders_to_daily TO daily_sales_table AS
SELECT toDate(created_at) AS day, sum(amount) AS total
FROM orders GROUP BY day;

SELECT * FROM daily_sales_table;  -- 이미 계산된 결과를 바로 읽는다 (빠름)
```

### 일반 테이블 2개 vs Materialized View

일반 테이블만 쓰면 INSERT 를 2번 해야 한다:

```sql
-- 수동: 개발자가 직접 2번 INSERT
INSERT INTO orders VALUES ('user-1', 10000, now());
INSERT INTO daily_sales_table VALUES (today(), 10000);  -- 빼먹으면 집계가 틀림
```

Materialized View 를 쓰면 1번만 INSERT 하면 된다:

```sql
-- 자동: 1번만 INSERT 하면 daily_sales_table 에도 자동으로 들어감
INSERT INTO orders VALUES ('user-1', 10000, now());
-- daily_sales_table 에는 Materialized View 가 알아서 넣어준다
```

### 기본 구조

```sql
-- 1. 원본 테이블
CREATE TABLE events (
    event_id UInt64,
    user_id UInt32,
    event_type String,
    created_at DateTime
)
ENGINE = MergeTree()
ORDER BY created_at;

-- 2. Materialized View (자동 갱신)
CREATE MATERIALIZED VIEW events_daily_stats
ENGINE = SummingMergeTree(event_count)
ORDER BY (date, event_type)
AS
SELECT
    toDate(created_at) as date,
    event_type,
    COUNT(*) as event_count
FROM events
GROUP BY date, event_type;
```

### 동작 원리

```sql
-- events에 데이터를 삽입하면...
INSERT INTO events VALUES
    (1, 100, 'login', '2026-03-31 10:00:00'),
    (2, 101, 'login', '2026-03-31 11:00:00'),
    (3, 100, 'purchase', '2026-03-31 12:00:00');

-- ...자동으로 events_daily_stats에도 집계된 데이터가 쌓인다.
SELECT * FROM events_daily_stats;
-- 결과:
-- date='2026-03-31', event_type='login', event_count=2
-- date='2026-03-31', event_type='purchase', event_count=1
```

### 실전 예제: 실시간 사용자 활동

```sql
-- 1. 사용자 활동 원본 테이블
CREATE TABLE user_actions (
    user_id UInt32,
    action String,  -- 'view', 'click', 'purchase'
    created_at DateTime
)
ENGINE = MergeTree()
ORDER BY created_at;

-- 2. 시간별 사용자 활동 통계 (자동 갱신)
CREATE MATERIALIZED VIEW user_actions_hourly
ENGINE = SummingMergeTree(action_count)
ORDER BY (hour, action)
AS
SELECT
    toStartOfHour(created_at) as hour,
    action,
    COUNT(*) as action_count
FROM user_actions
GROUP BY hour, action;

-- 3. 사용자별 일별 활동 통계
CREATE MATERIALIZED VIEW user_actions_daily
ENGINE = SummingMergeTree(action_count)
ORDER BY (date, user_id, action)
AS
SELECT
    toDate(created_at) as date,
    user_id,
    action,
    COUNT(*) as action_count
FROM user_actions
GROUP BY date, user_id, action;
```

**장점**:
- user_actions에 INSERT하면 자동으로 hour/day 단위 통계가 쌓인다
- 실시간 집계가 필요한 경우 유용
- 쿼리 응답 빠름 (미리 계산된 값)

**단점**:
- 스토리지 사용량 증가
- Materialized View가 많으면 INSERT 속도 저하

---

## 파티션과 TTL

### PARTITION BY — 날짜별 파티션

```sql
CREATE TABLE logs (
    timestamp DateTime,
    level String,      -- 'INFO', 'ERROR', 'WARNING'
    message String
)
ENGINE = MergeTree()
PARTITION BY toDate(timestamp)  -- 날짜별로 분할
ORDER BY timestamp;
```

**효과**:
- 날짜별로 파일이 분리 → 오래된 날짜 삭제 빠름
- 날짜 범위 쿼리가 빠름 (필요한 파티션만 읽음)

### TTL — 자동 삭제

```sql
CREATE TABLE logs (
    timestamp DateTime,
    level String,
    message String
)
ENGINE = MergeTree()
PARTITION BY toDate(timestamp)
ORDER BY timestamp
TTL timestamp + INTERVAL 30 DAY;  -- 30일 후 자동 삭제
```

실무에서는 보통 **날짜 컬럼 + 파티션 + TTL** 조합:

```sql
CREATE TABLE events (
    event_id UInt64,
    user_id UInt32,
    event_type String,
    created_at DateTime
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)  -- 월별 파티션
ORDER BY (created_at, user_id)
TTL created_at + INTERVAL 90 DAY;  -- 90일 후 자동 삭제
```

---

## 실전 예제: 이커머스 분석

### 시나리오

온라인 쇼핑몰이 있다. 이런 질문에 답해야 한다:
- 일별 매출은?
- 상품별 판매량은?
- 사용자별 구매 패턴은?

### 1단계: 원본 테이블 생성

```sql
CREATE TABLE orders (
    order_id UInt64,
    user_id UInt32,
    product_id UInt32,
    quantity UInt32,
    unit_price Float64,
    total_amount Float64,
    status LowCardinality(String),  -- 'pending', 'completed', 'cancelled'
    created_at DateTime
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)
ORDER BY (created_at, user_id)
TTL created_at + INTERVAL 2 YEAR;
```

### 2단계: 데이터 삽입

```sql
INSERT INTO orders VALUES
    (1, 100, 1001, 2, 49.99, 99.98, 'completed', '2026-03-31 10:00:00'),
    (2, 101, 1002, 1, 99.99, 99.99, 'completed', '2026-03-31 11:00:00'),
    (3, 100, 1003, 3, 29.99, 89.97, 'completed', '2026-03-31 12:00:00'),
    (4, 102, 1001, 1, 49.99, 49.99, 'pending', '2026-03-31 13:00:00');
```

### 3단계: 일별 매출 Materialized View

```sql
CREATE MATERIALIZED VIEW daily_sales_stats
ENGINE = SummingMergeTree(total_amount, order_count)
PARTITION BY toYYYYMM(date)
ORDER BY (date, status)
AS
SELECT
    toDate(created_at) as date,
    status,
    SUM(total_amount) as total_amount,
    COUNT(*) as order_count
FROM orders
WHERE status = 'completed'
GROUP BY date, status;
```

### 4단계: 상품별 판매 통계 View

```sql
CREATE MATERIALIZED VIEW product_sales_stats
ENGINE = SummingMergeTree(quantity, revenue)
ORDER BY (date, product_id)
AS
SELECT
    toDate(created_at) as date,
    product_id,
    SUM(quantity) as quantity,
    SUM(total_amount) as revenue
FROM orders
WHERE status = 'completed'
GROUP BY date, product_id;
```

### 5단계: 쿼리로 데이터 확인

```sql
-- 일별 매출
SELECT * FROM daily_sales_stats ORDER BY date DESC;

-- 상품별 판매량
SELECT * FROM product_sales_stats ORDER BY date DESC;

-- 사용자별 구매 이력 + 랭킹
SELECT
    user_id,
    COUNT(*) as purchase_count,
    SUM(total_amount) as lifetime_value,
    ROW_NUMBER() OVER (ORDER BY SUM(total_amount) DESC) as vip_rank
FROM orders
WHERE status = 'completed'
GROUP BY user_id
ORDER BY lifetime_value DESC
LIMIT 10;
```

### 6단계: 고급 분석

```sql
-- 사용자별 구매 주기 (평균 일수)
SELECT
    user_id,
    COUNT(*) as purchase_count,
    dateDiff('day', MIN(created_at), MAX(created_at)) / (COUNT(*) - 1) as avg_days_between_purchase
FROM orders
WHERE status = 'completed'
GROUP BY user_id
HAVING purchase_count > 1
ORDER BY avg_days_between_purchase DESC;

-- 제품별 인기도 (판매량 vs. 가격대)
SELECT
    product_id,
    COUNT(*) as sales_count,
    AVG(unit_price) as avg_price,
    SUM(total_amount) as total_revenue,
    quantile(0.5)(total_amount) as median_order_value
FROM orders
WHERE status = 'completed'
GROUP BY product_id
ORDER BY sales_count DESC;

-- 시간대별 매출
SELECT
    toStartOfHour(created_at) as hour,
    COUNT(*) as order_count,
    SUM(total_amount) as hourly_revenue
FROM orders
WHERE status = 'completed'
GROUP BY hour
ORDER BY hour DESC;
```

---

## 정리

| 항목 | 내용 |
|------|------|
| **MergeTree** | 기본 엔진, 대부분 여기서 시작 |
| **ReplacingMergeTree** | 상태 변경 추적 |
| **SummingMergeTree** | 미리 계산된 합계 |
| **LowCardinality** | 반복되는 값 최적화 |
| **Materialized View** | 자동 갱신되는 집계 |
| **파티션 + TTL** | 구형 데이터 자동 삭제 |
| **집계 함수** | COUNT, SUM, AVG, uniq, quantile 등 |
| **윈도우 함수** | ROW_NUMBER, RANK, LAG/LEAD |

---

## 다음 단계

1. 로컬에서 ClickHouse 실행 (`docker-compose` 사용)
2. 위 예제를 직접 실행해보기
3. `clickhouse-kafka-integration-kr.md` 읽기 (Kafka와 실시간 연동)

