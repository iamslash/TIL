# Flink SQL 가이드

## 목차

- [Flink SQL 이란?](#flink-sql-이란)
- [환경 설정](#환경-설정)
- [테이블 생성 (DDL)](#테이블-생성-ddl)
- [기본 쿼리](#기본-쿼리)
- [윈도우 집계](#윈도우-집계)
- [조인](#조인)
- [내장 함수와 UDF](#내장-함수와-udf)
- [Table API](#table-api)
- [카탈로그](#카탈로그)
- [실무 예제: 실시간 대시보드 파이프라인](#실무-예제-실시간-대시보드-파이프라인)

---

## Flink SQL 이란?

Flink SQL 은 Apache Flink 가 제공하는 선언적 쿼리 언어다. 표준 SQL 문법으로 스트리밍 데이터와 배치 데이터를 동일하게 처리할 수 있다. DataStream API 와 동일한 작업을 훨씬 적은 코드로 표현할 수 있어서 데이터 분석가나 SQL 에 익숙한 엔지니어도 쉽게 사용할 수 있다.

### DataStream API vs Flink SQL 비교

같은 작업 — Kafka 에서 주문 이벤트를 읽어서 사용자별 총 금액을 집계하는 파이프라인을 두 방식으로 비교한다.

**DataStream API 방식 (Java)**

```java
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

// Kafka 소스 설정
KafkaSource<String> source = KafkaSource.<String>builder()
    .setBootstrapServers("localhost:9092")
    .setTopics("orders")
    .setGroupId("flink-consumer-group")
    .setStartingOffsets(OffsetsInitializer.earliest())
    .setValueOnlyDeserializer(new SimpleStringSchema())
    .build();

DataStream<String> rawStream = env.fromSource(
    source, WatermarkStrategy.noWatermarks(), "Kafka Source");

// JSON 파싱, 필터링, 사용자별 합계 집계
DataStream<Tuple2<String, Double>> result = rawStream
    .map(json -> parseOrder(json))         // JSON -> Order 객체
    .filter(order -> order.getAmount() > 0)
    .keyBy(order -> order.getUserId())
    .sum("amount")
    .map(order -> Tuple2.of(order.getUserId(), order.getAmount()));

// Kafka 싱크로 출력
result.addSink(buildKafkaSink());
env.execute("Order Aggregation");
```

**Flink SQL 방식**

```sql
-- Kafka 소스 테이블 정의
CREATE TABLE orders (
    user_id   STRING,
    amount    DOUBLE,
    order_time TIMESTAMP(3),
    WATERMARK FOR order_time AS order_time - INTERVAL '5' SECOND
) WITH (
    'connector' = 'kafka',
    'topic'     = 'orders',
    'properties.bootstrap.servers' = 'localhost:9092',
    'properties.group.id'          = 'flink-consumer-group',
    'scan.startup.mode'            = 'earliest-offset',
    'format'                       = 'json'
);

-- Kafka 싱크 테이블 정의
CREATE TABLE order_totals (
    user_id     STRING,
    total_amount DOUBLE
) WITH (
    'connector' = 'kafka',
    'topic'     = 'order-totals',
    'properties.bootstrap.servers' = 'localhost:9092',
    'format'                       = 'json'
);

-- 사용자별 합계 집계 후 싱크에 삽입
INSERT INTO order_totals
SELECT user_id, SUM(amount) AS total_amount
FROM orders
WHERE amount > 0
GROUP BY user_id;
```

DataStream API 는 약 30줄이지만 Flink SQL 은 30줄 미만의 SQL 로 같은 동작을 표현한다.

### 언제 Flink SQL 을 쓰는가

- **간단한 집계와 필터링**: GROUP BY, SUM, COUNT, AVG 등 표준 집계 연산
- **ETL 파이프라인**: 소스에서 읽어서 변환 후 싱크에 쓰는 단순 파이프라인
- **윈도우 집계**: TUMBLE, HOP, SESSION 윈도우를 SQL 문법으로 간결하게 표현
- **데이터 분석가가 직접 운영**: SQL 에 익숙한 팀원이 Flink 코드 없이 직접 쿼리 작성
- **빠른 프로토타이핑**: 아이디어 검증 시 코드 작성 시간 단축

### 언제 DataStream API 를 쓰는가

- **복잡한 상태 관리**: ValueState, ListState, MapState 를 세밀하게 제어해야 할 때
- **CEP (복잡 이벤트 처리)**: 특정 패턴을 감지하는 복잡한 시퀀스 매칭
- **커스텀 타이머 로직**: ProcessFunction 으로 시간 기반 로직을 세밀하게 제어할 때
- **비표준 소스/싱크**: 커넥터가 없는 시스템과 통합할 때
- **낮은 수준의 최적화**: 성능이 극도로 중요한 경우 직접 연산자를 제어

---

## 환경 설정

### Maven 의존성

`pom.xml` 에 아래 의존성을 추가한다. Flink 버전은 1.18.x 기준이다.

```xml
<properties>
    <flink.version>1.18.1</flink.version>
    <java.version>11</java.version>
</properties>

<dependencies>
    <!-- Flink 스트리밍 핵심 -->
    <dependency>
        <groupId>org.apache.flink</groupId>
        <artifactId>flink-streaming-java</artifactId>
        <version>${flink.version}</version>
    </dependency>

    <!-- Table API와 SQL을 DataStream API와 연결하는 브릿지 -->
    <dependency>
        <groupId>org.apache.flink</groupId>
        <artifactId>flink-table-api-java-bridge</artifactId>
        <version>${flink.version}</version>
    </dependency>

    <!-- Flink SQL 플래너 (실제 SQL 실행 엔진) -->
    <!-- 테스트나 로컬 실행 시 필요. 클러스터에는 이미 포함됨 -->
    <dependency>
        <groupId>org.apache.flink</groupId>
        <artifactId>flink-table-planner-loader</artifactId>
        <version>${flink.version}</version>
    </dependency>

    <!-- Kafka 커넥터 -->
    <dependency>
        <groupId>org.apache.flink</groupId>
        <artifactId>flink-connector-kafka</artifactId>
        <version>3.1.0-1.18</version>
    </dependency>

    <!-- JSON 포맷 지원 -->
    <dependency>
        <groupId>org.apache.flink</groupId>
        <artifactId>flink-json</artifactId>
        <version>${flink.version}</version>
    </dependency>
</dependencies>
```

### TableEnvironment 생성

TableEnvironment 는 Flink SQL 과 Table API 의 진입점이다. 모든 DDL 과 DML 은 이 객체를 통해 실행된다.

```java
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.EnvironmentSettings;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;

public class FlinkSqlSetup {

    public static void main(String[] args) {
        // 방법 1: 스트리밍 모드 (실시간 파이프라인에 사용)
        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tableEnv = StreamTableEnvironment.create(env);

        // 방법 2: TableEnvironment 만 사용할 때 (DataStream 변환 불필요)
        EnvironmentSettings settings = EnvironmentSettings
            .newInstance()
            .inStreamingMode()   // 스트리밍 모드
            // .inBatchMode()    // 배치 모드로 전환하려면 이 줄 사용
            .build();
        TableEnvironment tableEnvOnly = TableEnvironment.create(settings);

        // SQL 실행 예시
        tableEnv.executeSql("SHOW TABLES");
    }
}
```

### SQL Client 사용법

SQL Client 는 Flink 배포판에 포함된 대화형 SQL 셸이다. 코드 없이 SQL 을 직접 실행해볼 수 있다.

```bash
# Flink 클러스터 시작
./bin/start-cluster.sh

# SQL Client 실행
./bin/sql-client.sh

# SQL Client 안에서 실행 모드 설정
SET 'execution.runtime-mode' = 'streaming';

# 결과를 테이블 형태로 보기
SET 'sql-client.execution.result-mode' = 'tableau';

# SQL 직접 실행
CREATE TABLE ...;
SELECT * FROM ...;
```

---

## 테이블 생성 (DDL)

### Kafka 소스 테이블 정의

```sql
-- 사용자 클릭 이벤트를 Kafka에서 읽는 소스 테이블
CREATE TABLE user_clicks (
    -- 필드 정의
    user_id      STRING        COMMENT '사용자 고유 식별자',
    page_url     STRING        COMMENT '클릭한 페이지 URL',
    click_count  INT           COMMENT '클릭 횟수',
    event_time   TIMESTAMP(3)  COMMENT '이벤트 발생 시각 (밀리초 정밀도)',

    -- 워터마크 정의: 최대 5초 지연을 허용
    -- event_time 기준으로 윈도우 집계를 하려면 워터마크가 필요
    WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
) WITH (
    -- 커넥터 종류
    'connector'                        = 'kafka',

    -- Kafka 토픽 이름
    'topic'                            = 'user-clicks',

    -- Kafka 브로커 주소
    'properties.bootstrap.servers'    = 'localhost:9092',

    -- 컨슈머 그룹 ID
    'properties.group.id'             = 'flink-sql-group',

    -- 시작 오프셋: earliest-offset(처음부터), latest-offset(최신부터)
    'scan.startup.mode'                = 'earliest-offset',

    -- 메시지 포맷
    'format'                           = 'json',

    -- JSON 필드가 없을 때 null 허용 (기본값 false)
    'json.ignore-parse-errors'         = 'true'
);
```

### Kafka 싱크 테이블 정의

```sql
-- 집계 결과를 Kafka에 쓰는 싱크 테이블
CREATE TABLE page_click_summary (
    page_url        STRING        COMMENT '페이지 URL',
    total_clicks    BIGINT        COMMENT '총 클릭 수',
    unique_users    BIGINT        COMMENT '순 방문자 수',
    window_start    TIMESTAMP(3)  COMMENT '윈도우 시작 시각',
    window_end      TIMESTAMP(3)  COMMENT '윈도우 종료 시각'
) WITH (
    'connector'                     = 'kafka',
    'topic'                         = 'page-click-summary',
    'properties.bootstrap.servers' = 'localhost:9092',

    -- 싱크 포맷
    'format'                        = 'json',

    -- JSON 출력 시 타임스탬프 포맷
    'json.timestamp-format.standard' = 'ISO-8601'
);
```

### WITH 절 커넥터 옵션 설명

| 옵션 | 설명 | 주요 값 |
|------|------|---------|
| `connector` | 커넥터 종류 | `kafka`, `jdbc`, `filesystem`, `blackhole` |
| `topic` | Kafka 토픽 이름 | 문자열 |
| `properties.bootstrap.servers` | Kafka 브로커 주소 | `host:port` 쉼표 구분 |
| `properties.group.id` | 컨슈머 그룹 ID | 문자열 |
| `scan.startup.mode` | 읽기 시작 위치 | `earliest-offset`, `latest-offset`, `timestamp` |
| `format` | 직렬화 포맷 | `json`, `avro`, `csv`, `debezium-json` |
| `sink.partitioner` | 싱크 파티션 전략 | `fixed`, `round-robin`, `custom` |

---

## 기본 쿼리

### SELECT, WHERE, GROUP BY

```sql
-- 기본 선택: 특정 페이지의 클릭만 조회
SELECT
    user_id,
    page_url,
    click_count,
    event_time
FROM user_clicks
WHERE page_url LIKE '%/product/%'
  AND click_count > 0;
```

```sql
-- 그룹 집계: 페이지별 총 클릭 수 (연속 집계, 무한 상태 축적)
-- 주의: GROUP BY 에 시간 속성이 없으면 상태가 무한히 커질 수 있음
SELECT
    page_url,
    COUNT(*)        AS total_events,
    SUM(click_count) AS total_clicks,
    COUNT(DISTINCT user_id) AS unique_users
FROM user_clicks
WHERE click_count > 0
GROUP BY page_url;
```

### 집계 함수 (COUNT, SUM, AVG, MAX, MIN)

```sql
-- 사용자별 다양한 집계 지표 계산
SELECT
    user_id,
    COUNT(*)              AS event_count,    -- 이벤트 발생 횟수
    SUM(click_count)      AS total_clicks,   -- 총 클릭 수
    AVG(click_count)      AS avg_clicks,     -- 평균 클릭 수
    MAX(click_count)      AS max_clicks,     -- 최대 클릭 수
    MIN(click_count)      AS min_clicks,     -- 최소 클릭 수
    MAX(event_time)       AS last_seen       -- 마지막 이벤트 시각
FROM user_clicks
GROUP BY user_id;
```

### 완전한 예제: Kafka 소스 → 필터링 → Kafka 싱크

```sql
-- 1단계: 소스 테이블 (Kafka 'orders' 토픽)
CREATE TABLE orders (
    order_id    STRING,
    user_id     STRING,
    product_id  STRING,
    amount      DOUBLE,
    status      STRING,      -- 'PAID', 'PENDING', 'CANCELLED'
    order_time  TIMESTAMP(3),
    WATERMARK FOR order_time AS order_time - INTERVAL '10' SECOND
) WITH (
    'connector'                     = 'kafka',
    'topic'                         = 'orders',
    'properties.bootstrap.servers' = 'localhost:9092',
    'properties.group.id'          = 'flink-orders-group',
    'scan.startup.mode'            = 'latest-offset',
    'format'                        = 'json',
    'json.ignore-parse-errors'     = 'true'
);

-- 2단계: 싱크 테이블 (Kafka 'paid-orders' 토픽)
CREATE TABLE paid_orders (
    order_id   STRING,
    user_id    STRING,
    product_id STRING,
    amount     DOUBLE,
    order_time TIMESTAMP(3)
) WITH (
    'connector'                     = 'kafka',
    'topic'                         = 'paid-orders',
    'properties.bootstrap.servers' = 'localhost:9092',
    'format'                        = 'json'
);

-- 3단계: 결제 완료 주문만 필터링해서 싱크에 삽입
INSERT INTO paid_orders
SELECT
    order_id,
    user_id,
    product_id,
    amount,
    order_time
FROM orders
WHERE status = 'PAID'
  AND amount > 0.0;
```

---

## 윈도우 집계

스트리밍 데이터는 끝이 없으므로 집계를 위해 데이터를 특정 시간 범위로 나눈다. 이를 윈도우라고 한다. Flink SQL 은 세 가지 윈도우 함수를 제공한다.

### TUMBLE (고정 윈도우)

겹치지 않는 고정 크기의 윈도우다. 예를 들어 5분마다 독립적으로 집계한다.

```
시간: |----5분----|----5분----|----5분----|
       [  윈도우1 ] [  윈도우2 ] [  윈도우3 ]
```

DataStream API 의 `TumblingEventTimeWindows.of(Time.minutes(5))` 에 대응한다.

```sql
-- 5분 고정 윈도우로 페이지별 클릭 집계
SELECT
    page_url,
    COUNT(*)              AS event_count,
    SUM(click_count)      AS total_clicks,
    window_start,
    window_end
FROM TABLE(
    -- TUMBLE(테이블, 시간컬럼, 윈도우크기)
    TUMBLE(TABLE user_clicks, DESCRIPTOR(event_time), INTERVAL '5' MINUTE)
)
GROUP BY page_url, window_start, window_end;
```

### HOP (슬라이딩 윈도우)

일정 간격으로 슬라이드하는 윈도우다. 윈도우가 겹칠 수 있어서 같은 이벤트가 여러 윈도우에 포함된다. 예를 들어 10분 크기 윈도우를 5분마다 슬라이드한다.

```
시간:  |--10분--|
            |--10분--|
                  |--10분--|
       [  윈도우1 ]
            [  윈도우2 ]
```

DataStream API 의 `SlidingEventTimeWindows.of(Time.minutes(10), Time.minutes(5))` 에 대응한다.

```sql
-- 10분 크기 윈도우를 5분마다 슬라이드: 최근 10분 클릭 집계를 5분마다 갱신
SELECT
    page_url,
    COUNT(*)          AS event_count,
    SUM(click_count)  AS total_clicks,
    window_start,
    window_end
FROM TABLE(
    -- HOP(테이블, 시간컬럼, 슬라이드간격, 윈도우크기)
    HOP(TABLE user_clicks, DESCRIPTOR(event_time),
        INTERVAL '5' MINUTE,    -- 슬라이드 간격
        INTERVAL '10' MINUTE)   -- 윈도우 크기
)
GROUP BY page_url, window_start, window_end;
```

### SESSION (세션 윈도우)

이벤트 사이의 간격이 일정 시간을 초과하면 새 윈도우를 시작한다. 사용자 세션처럼 활동이 집중된 시간대를 감지할 때 유용하다.

```
이벤트: --E--E--E-----gap(30분)-----E--E--
        [--세션1--]                [세션2]
```

DataStream API 의 `EventTimeSessionWindows.withGap(Time.minutes(30))` 에 대응한다.

```sql
-- 30분 동안 이벤트가 없으면 세션 종료: 사용자 세션별 클릭 집계
SELECT
    user_id,
    COUNT(*)          AS event_count,
    SUM(click_count)  AS total_clicks,
    window_start,
    window_end
FROM TABLE(
    -- SESSION(테이블, 시간컬럼, 세션간격)
    SESSION(TABLE user_clicks, DESCRIPTOR(event_time),
            INTERVAL '30' MINUTE)   -- 30분 이상 비활동 시 세션 종료
)
GROUP BY user_id, window_start, window_end;
```

---

## 조인

### Regular Join (양쪽 모두 변경 가능)

두 스트리밍 테이블을 조인한다. 양쪽 테이블의 모든 과거 데이터를 상태로 유지하므로 상태가 무한히 커질 수 있다. 오래된 상태는 TTL 설정으로 정리한다.

```sql
-- 주문과 사용자 정보를 조인
-- 두 스트림 모두 상태에 유지됨 (상태 크기 주의)
SELECT
    o.order_id,
    o.amount,
    u.user_name,
    u.email
FROM orders AS o
JOIN users AS u
    ON o.user_id = u.user_id;
```

### Interval Join (시간 범위 조건)

두 스트리밍 테이블을 시간 조건으로 조인한다. 특정 시간 범위 내의 이벤트만 매칭하므로 상태 크기가 제한된다.

```sql
-- 주문 후 1시간 이내에 발생한 배송 이벤트와 조인
-- 상태 크기가 시간 범위(1시간)로 제한됨
SELECT
    o.order_id,
    o.amount,
    s.tracking_number,
    s.ship_time
FROM orders AS o
JOIN shipments AS s
    ON  o.order_id = s.order_id
    -- 배송 시각이 주문 시각 이후, 주문 1시간 이내여야 함
    AND s.ship_time BETWEEN o.order_time
                        AND o.order_time + INTERVAL '1' HOUR;
```

### Temporal Join (버전 테이블 조인)

시간에 따라 값이 바뀌는 테이블 (예: 환율, 상품 가격) 에 대해 이벤트 발생 시점의 값으로 조인한다.

```sql
-- 환율 이력 테이블 (시간에 따라 환율이 변동)
CREATE TABLE exchange_rates (
    currency    STRING,
    rate        DOUBLE,
    valid_time  TIMESTAMP(3),
    PRIMARY KEY (currency) NOT ENFORCED
) WITH (
    'connector' = 'upsert-kafka',
    'topic'     = 'exchange-rates',
    'properties.bootstrap.servers' = 'localhost:9092',
    'key.format' = 'json',
    'value.format' = 'json'
);

-- 주문 발생 시점의 환율로 금액을 USD 로 변환
SELECT
    o.order_id,
    o.amount,
    o.currency,
    -- 주문 시점의 환율 적용
    o.amount * r.rate AS amount_usd,
    o.order_time
FROM orders AS o
-- FOR SYSTEM_TIME AS OF: 이벤트 발생 시점의 환율 조회
LEFT JOIN exchange_rates FOR SYSTEM_TIME AS OF o.order_time AS r
    ON o.currency = r.currency;
```

### Lookup Join (외부 DB 조회)

스트리밍 데이터를 외부 데이터베이스에서 실시간으로 조회해서 보강(enrich)한다. MySQL, PostgreSQL 등 JDBC 를 지원하는 DB 에 사용한다.

```sql
-- 외부 MySQL 의 사용자 정보 테이블 정의
CREATE TABLE user_profiles (
    user_id    STRING,
    user_name  STRING,
    tier       STRING,    -- 'gold', 'silver', 'bronze'
    PRIMARY KEY (user_id) NOT ENFORCED
) WITH (
    'connector'  = 'jdbc',
    'url'        = 'jdbc:mysql://localhost:3306/users_db',
    'table-name' = 'user_profiles',
    'username'   = 'flink',
    'password'   = 'password',
    -- 조회 결과 캐시 설정 (외부 DB 부하 감소)
    'lookup.cache'              = 'partial',
    'lookup.partial-cache.max-rows'     = '10000',
    'lookup.partial-cache.expire-after-write' = '10 min'
);

-- 주문 스트림에 사용자 등급 정보 추가
SELECT
    o.order_id,
    o.user_id,
    o.amount,
    u.user_name,
    u.tier
FROM orders AS o
-- Lookup Join: 실시간으로 MySQL 에서 사용자 정보 조회
JOIN user_profiles FOR SYSTEM_TIME AS OF o.order_time AS u
    ON o.user_id = u.user_id;
```

---

## 내장 함수와 UDF

### 주요 내장 함수

**날짜/시간 함수**

```sql
SELECT
    NOW()                                    AS current_timestamp,
    CURRENT_DATE                             AS today,
    TIMESTAMPADD(HOUR, 9, event_time)        AS kst_time,   -- UTC+9 변환
    DATE_FORMAT(event_time, 'yyyy-MM-dd')    AS event_date,
    EXTRACT(HOUR FROM event_time)            AS event_hour,
    DATEDIFF(NOW(), event_time)              AS days_ago
FROM user_clicks;
```

**문자열 함수**

```sql
SELECT
    UPPER(page_url)                    AS upper_url,
    LOWER(user_id)                     AS lower_user,
    SUBSTRING(page_url, 1, 50)         AS short_url,     -- 처음 50자
    CONCAT(user_id, '-', product_id)   AS composite_key,
    REGEXP_EXTRACT(page_url, '/(\d+)$', 1) AS product_id_from_url,
    CHAR_LENGTH(page_url)              AS url_length,
    TRIM(user_id)                      AS trimmed_id
FROM user_clicks;
```

**수학 함수**

```sql
SELECT
    ROUND(amount, 2)     AS rounded_amount,  -- 소수점 2자리 반올림
    FLOOR(amount)        AS floor_amount,
    CEIL(amount)         AS ceil_amount,
    ABS(amount)          AS absolute_amount,
    POWER(amount, 2)     AS amount_squared,
    LOG(amount)          AS log_amount,
    MOD(click_count, 10) AS click_mod_10
FROM orders;
```

### Scalar UDF 작성

스칼라 UDF 는 행 하나를 입력받아 값 하나를 반환한다.

```java
import org.apache.flink.table.annotation.DataTypeHint;
import org.apache.flink.table.functions.ScalarFunction;

// 사용자 등급을 계산하는 UDF
// 총 주문 금액에 따라 Gold/Silver/Bronze 등급 반환
public class UserTierFunction extends ScalarFunction {

    // eval 메서드가 실제 로직. 파라미터 타입이 SQL 타입과 매핑됨.
    public String eval(Double totalAmount) {
        if (totalAmount == null) {
            return "UNKNOWN";
        }
        if (totalAmount >= 1000000.0) {
            return "GOLD";
        } else if (totalAmount >= 100000.0) {
            return "SILVER";
        } else {
            return "BRONZE";
        }
    }
}
```

```java
// Java 코드에서 UDF 등록 후 SQL 에서 사용
tableEnv.createTemporaryFunction("USER_TIER", UserTierFunction.class);

// SQL 에서 사용
tableEnv.executeSql(
    "SELECT user_id, SUM(amount) AS total, USER_TIER(SUM(amount)) AS tier " +
    "FROM orders GROUP BY user_id"
);
```

```sql
-- SQL Client 에서 UDF 를 JAR 로 등록하는 방법
ADD JAR '/path/to/my-udfs.jar';

CREATE TEMPORARY FUNCTION USER_TIER
    AS 'com.example.udf.UserTierFunction';

-- 등록된 UDF 사용
SELECT user_id, USER_TIER(SUM(amount)) AS tier
FROM orders
GROUP BY user_id;
```

### Table UDF 작성

테이블 UDF 는 행 하나를 입력받아 0개 이상의 행을 반환한다. 하나의 값을 여러 행으로 분해할 때 사용한다.

```java
import org.apache.flink.table.annotation.DataTypeHint;
import org.apache.flink.table.annotation.FunctionHint;
import org.apache.flink.table.functions.TableFunction;
import org.apache.flink.types.Row;

// 쉼표로 구분된 태그 문자열을 개별 행으로 분해하는 UDF
// 예: "java,flink,sql" -> 3개 행 (java), (flink), (sql)
@FunctionHint(output = @DataTypeHint("ROW<tag STRING>"))
public class SplitTagsFunction extends TableFunction<Row> {

    public void eval(String tags) {
        if (tags == null || tags.isEmpty()) {
            return;
        }
        // 쉼표 구분자로 분리하여 각 태그를 개별 행으로 출력
        for (String tag : tags.split(",")) {
            collect(Row.of(tag.trim()));
        }
    }
}
```

```java
// 등록
tableEnv.createTemporaryFunction("SPLIT_TAGS", SplitTagsFunction.class);
```

```sql
-- CROSS JOIN LATERAL TABLE 로 Table UDF 호출
SELECT
    a.article_id,
    t.tag
FROM articles AS a
CROSS JOIN LATERAL TABLE(SPLIT_TAGS(a.tags)) AS t(tag);
```

### Aggregate UDF 작성

집계 UDF 는 여러 행을 입력받아 하나의 값을 반환한다. 기본 집계 함수로 표현할 수 없는 커스텀 집계를 구현할 때 사용한다.

```java
import org.apache.flink.table.functions.AggregateFunction;

// 가중 평균을 계산하는 집계 UDF
// 일반 AVG 와 달리 각 값의 가중치를 반영
public class WeightedAvgFunction
        extends AggregateFunction<Double, WeightedAvgFunction.Accumulator> {

    // 집계 상태를 저장하는 어큐뮬레이터
    public static class Accumulator {
        public double weightedSum = 0.0;  // 가중합
        public long   totalWeight = 0L;   // 총 가중치
    }

    // 초기 어큐뮬레이터 생성
    @Override
    public Accumulator createAccumulator() {
        return new Accumulator();
    }

    // 각 행의 값을 어큐뮬레이터에 누적
    // iValue: 집계할 값, iWeight: 가중치
    public void accumulate(Accumulator acc, Double iValue, Long iWeight) {
        if (iValue != null && iWeight != null) {
            acc.weightedSum += iValue * iWeight;
            acc.totalWeight += iWeight;
        }
    }

    // 최종 결과 계산
    @Override
    public Double getValue(Accumulator acc) {
        if (acc.totalWeight == 0) {
            return null;
        }
        return acc.weightedSum / acc.totalWeight;
    }
}
```

```java
// 등록
tableEnv.createTemporaryFunction("WEIGHTED_AVG", WeightedAvgFunction.class);
```

```sql
-- 상품별 가중 평균 가격 계산 (판매량을 가중치로 사용)
SELECT
    product_id,
    WEIGHTED_AVG(price, sales_count) AS weighted_avg_price
FROM product_sales
GROUP BY product_id;
```

---

## Table API

### Table API 란

Table API 는 SQL 문자열을 직접 쓰지 않고 Java/Python 메서드 체이닝으로 같은 표현을 작성하는 방식이다. SQL 과 DataStream API 의 중간 수준으로, 타입 안전성과 IDE 자동완성 지원이라는 장점이 있다.

- SQL: 문자열로 작성, 런타임에 파싱
- Table API: 메서드 체인으로 작성, 컴파일 타임에 타입 확인
- DataStream API: 연산자를 직접 조합, 가장 낮은 수준

### Table API 로 같은 작업 수행

SQL 과 Table API 는 동일한 실행 계획을 생성한다. 취향과 팀 컨벤션에 따라 선택한다.

```java
import org.apache.flink.table.api.*;
import static org.apache.flink.table.api.Expressions.*;

public class TableApiExample {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tableEnv = StreamTableEnvironment.create(env);

        // SQL DDL 로 소스 테이블 생성 (Table API 도 DDL 은 SQL 로 작성)
        tableEnv.executeSql(
            "CREATE TABLE orders (" +
            "  order_id   STRING," +
            "  user_id    STRING," +
            "  amount     DOUBLE," +
            "  status     STRING," +
            "  order_time TIMESTAMP(3)," +
            "  WATERMARK FOR order_time AS order_time - INTERVAL '5' SECOND" +
            ") WITH (" +
            "  'connector' = 'kafka'," +
            "  'topic'     = 'orders'," +
            "  'properties.bootstrap.servers' = 'localhost:9092'," +
            "  'format'    = 'json'" +
            ")"
        );

        // Table 객체 가져오기
        Table ordersTable = tableEnv.from("orders");

        // Table API 로 필터링, 그룹 집계
        Table result = ordersTable
            // WHERE status = 'PAID' AND amount > 0
            .filter(
                $("status").isEqual("PAID")
                    .and($("amount").isGreater(0.0))
            )
            // SELECT user_id, SUM(amount) AS total, COUNT(*) AS cnt
            // GROUP BY user_id
            .groupBy($("user_id"))
            .select(
                $("user_id"),
                $("amount").sum().as("total_amount"),
                lit(1).count().as("order_count")
            );

        // 결과를 SQL 로 조회할 수도 있음
        result.execute().print();
    }
}
```

### DataStream 과 Table 변환

DataStream API 의 스트림을 Table 로 변환하거나 반대로 변환할 수 있다. 두 API 를 혼합해서 사용할 때 필요하다.

```java
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;

public class DataStreamTableConversion {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();
        StreamTableEnvironment tableEnv = StreamTableEnvironment.create(env);

        // 1. DataStream -> Table 변환
        // 기존 DataStream API 로 만든 스트림을 Table API 로 가져옴
        DataStream<Order> orderStream = env
            .fromElements(
                new Order("o1", "u1", 1000.0, "PAID"),
                new Order("o2", "u2", 500.0,  "PAID"),
                new Order("o3", "u1", 200.0,  "PENDING")
            );

        // DataStream 을 Table 로 변환
        // 필드 이름을 명시적으로 지정
        Table ordersTable = tableEnv.fromDataStream(
            orderStream,
            Schema.newBuilder()
                .column("orderId",  DataTypes.STRING())
                .column("userId",   DataTypes.STRING())
                .column("amount",   DataTypes.DOUBLE())
                .column("status",   DataTypes.STRING())
                .build()
        );

        // Table API 로 처리
        Table paidOrders = ordersTable
            .filter($("status").isEqual("PAID"));

        // SQL 에서도 사용할 수 있도록 뷰로 등록
        tableEnv.createTemporaryView("paid_orders", paidOrders);
        tableEnv.executeSql("SELECT * FROM paid_orders").print();

        // 2. Table -> DataStream 변환
        // Table API 처리 결과를 다시 DataStream 으로 변환
        DataStream<Order> resultStream = tableEnv.toDataStream(
            paidOrders,
            Order.class
        );

        // DataStream API 로 추가 처리
        resultStream
            .filter(o -> o.getAmount() > 500.0)
            .print();

        env.execute("DataStream Table Conversion");
    }
}
```

---

## 카탈로그

### 카탈로그란

카탈로그는 테이블, 뷰, UDF 의 메타데이터를 저장하는 저장소다. Flink 는 기본적으로 인메모리 카탈로그를 사용하므로 프로그램이 종료되면 정의한 테이블이 사라진다. 영구적으로 테이블 정의를 유지하려면 외부 카탈로그를 사용한다.

- **InMemoryCatalog** (기본값): 세션 동안만 유지, 재시작 시 초기화
- **HiveCatalog**: Hive Metastore 에 저장, 재시작 후에도 유지
- **JdbcCatalog**: PostgreSQL 등 JDBC DB 에 저장

### HiveCatalog 사용

```xml
<!-- pom.xml 에 Hive 의존성 추가 -->
<dependency>
    <groupId>org.apache.flink</groupId>
    <artifactId>flink-connector-hive_2.12</artifactId>
    <version>${flink.version}</version>
</dependency>
<dependency>
    <groupId>org.apache.hive</groupId>
    <artifactId>hive-exec</artifactId>
    <version>3.1.3</version>
</dependency>
```

```java
import org.apache.flink.table.catalog.hive.HiveCatalog;

// HiveCatalog 생성 및 등록
String catalogName  = "my_hive_catalog";
String defaultDatabase = "flink_db";
String hiveConfDir  = "/etc/hive/conf";  // hive-site.xml 위치

HiveCatalog hiveCatalog = new HiveCatalog(
    catalogName,
    defaultDatabase,
    hiveConfDir
);

// TableEnvironment 에 카탈로그 등록
tableEnv.registerCatalog(catalogName, hiveCatalog);

// 등록한 카탈로그를 기본으로 설정
tableEnv.useCatalog(catalogName);
tableEnv.useDatabase(defaultDatabase);
```

```sql
-- SQL Client 에서 HiveCatalog 사용
CREATE CATALOG my_hive_catalog WITH (
    'type'            = 'hive',
    'hive-conf-dir'   = '/etc/hive/conf',
    'default-database' = 'flink_db'
);

USE CATALOG my_hive_catalog;
USE flink_db;

-- 이 테이블은 Hive Metastore 에 영구 저장됨
CREATE TABLE persistent_orders (
    order_id   STRING,
    amount     DOUBLE
) WITH (
    'connector' = 'kafka',
    'topic'     = 'orders',
    'properties.bootstrap.servers' = 'localhost:9092',
    'format'    = 'json'
);
```

### 영구 테이블 vs 임시 테이블

```sql
-- 임시 테이블: 현재 세션에서만 유효
CREATE TEMPORARY TABLE temp_orders (...) WITH (...);

-- 영구 테이블: 카탈로그에 저장되어 재시작 후에도 유지
-- HiveCatalog 등 영구 카탈로그를 사용 중일 때만 가능
CREATE TABLE permanent_orders (...) WITH (...);

-- 임시 뷰
CREATE TEMPORARY VIEW paid_view AS
SELECT * FROM orders WHERE status = 'PAID';

-- 영구 뷰
CREATE VIEW paid_view AS
SELECT * FROM orders WHERE status = 'PAID';
```

---

## 실무 예제: 실시간 대시보드 파이프라인

Kafka 에서 사용자 행동 이벤트를 읽어서 5분 윈도우로 집계하고, 결과를 다시 Kafka 에 쓴 뒤 Grafana 대시보드에 표시하는 파이프라인이다.

```
[사용자 앱] --이벤트--> [Kafka: user-events]
                              |
                        [Flink SQL]
                    (5분 윈도우 집계)
                              |
                   [Kafka: dashboard-metrics]
                              |
               [Grafana + Kafka Data Source 플러그인]
```

### 전체 SQL 쿼리

```sql
-- =============================
-- 1. 소스 테이블: 사용자 행동 이벤트
-- =============================
CREATE TABLE user_events (
    -- 이벤트 식별
    event_id     STRING     COMMENT '이벤트 고유 ID',
    user_id      STRING     COMMENT '사용자 ID',

    -- 이벤트 내용
    event_type   STRING     COMMENT '이벤트 종류 (click, view, purchase, search)',
    page         STRING     COMMENT '이벤트 발생 페이지',
    product_id   STRING     COMMENT '상품 ID (구매/조회 시)',
    amount       DOUBLE     COMMENT '금액 (구매 시)',

    -- 시간
    event_time   TIMESTAMP(3) COMMENT '이벤트 발생 시각',

    -- 워터마크: 최대 10초 지연 허용
    WATERMARK FOR event_time AS event_time - INTERVAL '10' SECOND
) WITH (
    'connector'                     = 'kafka',
    'topic'                         = 'user-events',
    'properties.bootstrap.servers' = 'kafka:9092',
    'properties.group.id'          = 'flink-dashboard-group',
    'scan.startup.mode'            = 'latest-offset',
    'format'                        = 'json',
    'json.ignore-parse-errors'     = 'true'
);

-- =============================
-- 2. 싱크 테이블: 대시보드 지표
-- =============================
CREATE TABLE dashboard_metrics (
    -- 집계 차원
    page            STRING    COMMENT '페이지',
    event_type      STRING    COMMENT '이벤트 종류',

    -- 집계 지표
    event_count     BIGINT    COMMENT '이벤트 건수',
    unique_users    BIGINT    COMMENT '순 방문자 수',
    total_amount    DOUBLE    COMMENT '총 거래 금액',
    avg_amount      DOUBLE    COMMENT '평균 거래 금액',

    -- 윈도우 시간
    window_start    TIMESTAMP(3) COMMENT '집계 윈도우 시작',
    window_end      TIMESTAMP(3) COMMENT '집계 윈도우 종료'
) WITH (
    'connector'                     = 'kafka',
    'topic'                         = 'dashboard-metrics',
    'properties.bootstrap.servers' = 'kafka:9092',
    'format'                        = 'json',
    'json.timestamp-format.standard' = 'ISO-8601'
);

-- =============================
-- 3. 집계 쿼리: 5분 윈도우로 집계 후 싱크에 삽입
-- =============================
INSERT INTO dashboard_metrics
SELECT
    page,
    event_type,

    -- 집계 지표 계산
    COUNT(*)                   AS event_count,
    COUNT(DISTINCT user_id)    AS unique_users,
    COALESCE(SUM(amount), 0.0) AS total_amount,     -- null 이면 0.0
    COALESCE(AVG(amount), 0.0) AS avg_amount,

    -- 윈도우 경계
    window_start,
    window_end

FROM TABLE(
    -- 5분 고정 윈도우 집계
    TUMBLE(TABLE user_events, DESCRIPTOR(event_time), INTERVAL '5' MINUTE)
)

-- 이벤트 종류와 페이지별로 그룹화
GROUP BY
    page,
    event_type,
    window_start,
    window_end;
```

### 추가 쿼리: 구매 전환율 계산

```sql
-- 페이지 조회(view) 대비 구매(purchase) 비율을 5분 윈도우로 집계
INSERT INTO conversion_metrics
SELECT
    page,
    window_start,
    window_end,

    -- 전체 이벤트 중 구매 비율
    COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) AS purchase_count,
    COUNT(CASE WHEN event_type = 'view'     THEN 1 END) AS view_count,

    -- 전환율 = 구매 수 / 조회 수 * 100
    CASE
        WHEN COUNT(CASE WHEN event_type = 'view' THEN 1 END) = 0
        THEN 0.0
        ELSE
            CAST(COUNT(CASE WHEN event_type = 'purchase' THEN 1 END) AS DOUBLE)
            / COUNT(CASE WHEN event_type = 'view' THEN 1 END)
            * 100.0
    END AS conversion_rate_pct

FROM TABLE(
    TUMBLE(TABLE user_events, DESCRIPTOR(event_time), INTERVAL '5' MINUTE)
)
GROUP BY page, window_start, window_end;
```

### Java 코드로 파이프라인 실행

```java
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.bridge.java.StreamTableEnvironment;

public class DashboardPipeline {

    public static void main(String[] args) throws Exception {
        // 환경 설정
        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();

        // 체크포인트: 장애 발생 시 최근 상태로 복구
        env.enableCheckpointing(60_000); // 60초마다 체크포인트

        StreamTableEnvironment tableEnv = StreamTableEnvironment.create(env);

        // DDL 과 DML 파일을 읽어서 실행하거나 아래처럼 직접 실행
        tableEnv.executeSql(CREATE_USER_EVENTS_SQL);
        tableEnv.executeSql(CREATE_DASHBOARD_METRICS_SQL);

        // INSERT 는 비동기로 실행됨 (스트리밍 잡 제출)
        tableEnv.executeSql(INSERT_METRICS_SQL);
    }

    private static final String CREATE_USER_EVENTS_SQL =
        "CREATE TABLE user_events (" +
        "  event_id   STRING," +
        "  user_id    STRING," +
        "  event_type STRING," +
        "  page       STRING," +
        "  product_id STRING," +
        "  amount     DOUBLE," +
        "  event_time TIMESTAMP(3)," +
        "  WATERMARK FOR event_time AS event_time - INTERVAL '10' SECOND" +
        ") WITH (" +
        "  'connector' = 'kafka'," +
        "  'topic'     = 'user-events'," +
        "  'properties.bootstrap.servers' = 'kafka:9092'," +
        "  'properties.group.id' = 'flink-dashboard-group'," +
        "  'scan.startup.mode'   = 'latest-offset'," +
        "  'format'              = 'json'" +
        ")";

    private static final String CREATE_DASHBOARD_METRICS_SQL =
        "CREATE TABLE dashboard_metrics (" +
        "  page         STRING," +
        "  event_type   STRING," +
        "  event_count  BIGINT," +
        "  unique_users BIGINT," +
        "  total_amount DOUBLE," +
        "  avg_amount   DOUBLE," +
        "  window_start TIMESTAMP(3)," +
        "  window_end   TIMESTAMP(3)" +
        ") WITH (" +
        "  'connector' = 'kafka'," +
        "  'topic'     = 'dashboard-metrics'," +
        "  'properties.bootstrap.servers' = 'kafka:9092'," +
        "  'format'    = 'json'" +
        ")";

    private static final String INSERT_METRICS_SQL =
        "INSERT INTO dashboard_metrics " +
        "SELECT page, event_type," +
        "       COUNT(*), COUNT(DISTINCT user_id)," +
        "       COALESCE(SUM(amount), 0.0), COALESCE(AVG(amount), 0.0)," +
        "       window_start, window_end " +
        "FROM TABLE(TUMBLE(TABLE user_events, DESCRIPTOR(event_time), INTERVAL '5' MINUTE)) " +
        "GROUP BY page, event_type, window_start, window_end";
}
```
