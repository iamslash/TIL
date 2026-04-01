# Flink 빠른 시작 (Hands-on)

- [가장 빠른 방법: Flink 로컬 클러스터](#가장-빠른-방법-flink-로컬-클러스터)
  - [1단계: 다운로드 및 시작](#1단계-다운로드-및-시작)
  - [2단계: Web UI 접속](#2단계-web-ui-접속)
  - [3단계: 예제 Job 실행](#3단계-예제-job-실행)
  - [4단계: 직접 데이터 넣고 결과 보기](#4단계-직접-데이터-넣고-결과-보기)
  - [5단계: 종료](#5단계-종료)
- [Docker Compose 로 실행하기](#docker-compose-로-실행하기)
  - [docker-compose.yml](#docker-composeyml)
  - [실행 및 확인](#실행-및-확인)
  - [예제 Job 실행](#예제-job-실행)
  - [종료](#종료)
- [Flink SQL Client 로 데이터 생산/소비 확인하기](#flink-sql-client-로-데이터-생산소비-확인하기)
  - [SQL Client 접속](#sql-client-접속)
  - [데이터 생성 (Source)](#데이터-생성-source)
  - [데이터 조회 (실시간 소비)](#데이터-조회-실시간-소비)
  - [윈도우 집계 확인](#윈도우-집계-확인)
- [Kafka 연동 (Docker Compose)](#kafka-연동-docker-compose)
  - [docker-compose-kafka.yml](#docker-compose-kafkayml)
  - [Kafka 에 데이터 생산하고 Flink 로 소비하기](#kafka-에-데이터-생산하고-flink-로-소비하기)

---

# 가장 빠른 방법: Flink 로컬 클러스터

Docker 도 필요 없다. Java 만 있으면 된다.

## 1단계: 다운로드 및 시작

```bash
# Java 11 이상 필요
java -version

# Flink 다운로드 및 압축 해제
curl -O https://dlcdn.apache.org/flink/flink-1.20.1/flink-1.20.1-bin-scala_2.12.tgz
tar xzf flink-1.20.1-bin-scala_2.12.tgz
cd flink-1.20.1

# 로컬 클러스터 시작 (JobManager + TaskManager)
./bin/start-cluster.sh
```

출력:
```
Starting cluster.
Starting standalonesession daemon on host my-mac.
Starting taskexecutor daemon on host my-mac.
```

## 2단계: Web UI 접속

브라우저에서 **http://localhost:8081** 을 열면 Flink 대시보드가 보인다.

- Running Jobs, Completed Jobs, TaskManagers 등을 확인할 수 있다.
- TaskManager 탭에서 Slot 수, 메모리 사용량 등을 볼 수 있다.

## 3단계: 예제 Job 실행

```bash
# 내장된 WordCount 예제 실행
./bin/flink run examples/streaming/WordCount.jar
```

출력:
```
Job has been submitted with JobID xxxxx
```

Web UI 에서 Job 이 실행되는 것을 확인한다. 결과는 `log/` 디렉토리에 저장된다.

```bash
# 결과 확인
cat log/flink-*-taskexecutor-*.out
```

```
(to,1)
(be,1)
(or,1)
(not,1)
(to,2)
(be,2)
...
```

## 4단계: 직접 데이터 넣고 결과 보기

소켓으로 데이터를 보내고 Flink 가 실시간으로 처리하는 것을 확인한다.

**터미널 1: 소켓 서버 시작 (데이터 생산자)**

```bash
nc -lk 9999
```

**터미널 2: Flink SocketWindowWordCount 실행 (소비자)**

```bash
./bin/flink run examples/streaming/SocketWindowWordCount.jar --port 9999
```

**터미널 1 에서 문장 입력:**

```
hello world
hello flink
flink is great
```

**터미널 3: 결과 확인**

```bash
tail -f log/flink-*-taskexecutor-*.out
```

```
(hello,2)
(world,1)
(flink,2)
(is,1)
(great,1)
```

타이핑할 때마다 5초 윈도우로 단어가 집계되는 것을 실시간으로 볼 수 있다.

## 5단계: 종료

```bash
./bin/stop-cluster.sh
```

---

# Docker Compose 로 실행하기

여러 명이 같은 환경을 쓰거나, 로컬에 Java 를 설치하고 싶지 않을 때 Docker Compose 를 사용한다.

## docker-compose.yml

```yaml
version: "3.8"
services:
  jobmanager:
    image: flink:1.20.1-scala_2.12-java11
    ports:
      - "8081:8081"   # Web UI
    command: jobmanager
    environment:
      - |
        FLINK_PROPERTIES=
        jobmanager.rpc.address: jobmanager
        taskmanager.numberOfTaskSlots: 4

  taskmanager:
    image: flink:1.20.1-scala_2.12-java11
    depends_on:
      - jobmanager
    command: taskmanager
    deploy:
      replicas: 2     # TaskManager 2개
    environment:
      - |
        FLINK_PROPERTIES=
        jobmanager.rpc.address: jobmanager
        taskmanager.numberOfTaskSlots: 4
```

> TaskManager 2개 x Slot 4개 = 최대 병렬도 8

## 실행 및 확인

```bash
# 시작
docker compose up -d

# 상태 확인
docker compose ps
```

브라우저에서 **http://localhost:8081** 접속. TaskManagers 탭에서 2개의 TaskManager 가 보인다.

## 예제 Job 실행

```bash
# JobManager 컨테이너에 접속하여 WordCount 실행
docker compose exec jobmanager ./bin/flink run examples/streaming/WordCount.jar

# 결과 확인 (TaskManager 컨테이너에서)
docker compose exec taskmanager cat /opt/flink/log/flink-*-taskexecutor-*.out
```

## 종료

```bash
docker compose down
```

---

# Flink SQL Client 로 데이터 생산/소비 확인하기

SQL 만으로 데이터를 생성하고 실시간으로 조회할 수 있다. **가장 직관적으로 Flink 를 체험하는 방법**이다.

## SQL Client 접속

```bash
# 로컬 클러스터 방식
./bin/sql-client.sh

# Docker Compose 방식
docker compose exec jobmanager ./bin/sql-client.sh
```

프롬프트가 나타나면 SQL 을 입력할 수 있다.

```
Flink SQL>
```

## 데이터 생성 (Source)

datagen 커넥터로 가짜 데이터를 자동 생성한다. Kafka 없이도 데이터가 끊임없이 들어온다.

```sql
-- 주문 이벤트 테이블 생성 (자동으로 데이터가 생성됨)
CREATE TABLE orders (
    order_id    INT,
    user_id     INT,
    product     STRING,
    amount      DOUBLE,
    order_time  TIMESTAMP(3),
    WATERMARK FOR order_time AS order_time - INTERVAL '5' SECOND
) WITH (
    'connector' = 'datagen',
    'rows-per-second' = '10',
    'fields.order_id.kind' = 'sequence',
    'fields.order_id.start' = '1',
    'fields.order_id.end' = '1000000',
    'fields.user_id.min' = '1',
    'fields.user_id.max' = '100',
    'fields.product.length' = '5',
    'fields.amount.min' = '1.0',
    'fields.amount.max' = '500.0'
);
```

## 데이터 조회 (실시간 소비)

```sql
-- 실시간 데이터 확인 (스트리밍 모드)
-- 데이터가 계속 흘러들어오는 것을 눈으로 볼 수 있다
SELECT * FROM orders LIMIT 20;
```

출력 (데이터가 실시간으로 나타난다):

```
+----------+---------+---------+--------+-------------------------+
| order_id | user_id | product | amount |      order_time         |
+----------+---------+---------+--------+-------------------------+
|        1 |      42 | aBcDe   | 123.45 | 2026-04-01 10:00:01.000 |
|        2 |       7 | xYzWq   |  89.99 | 2026-04-01 10:00:01.100 |
|        3 |      55 | pQrSt   | 456.78 | 2026-04-01 10:00:01.200 |
...
```

## 윈도우 집계 확인

```sql
-- 10초 단위로 유저별 주문 합계를 실시간 집계
SELECT
    user_id,
    TUMBLE_START(order_time, INTERVAL '10' SECOND) AS window_start,
    COUNT(*) AS order_count,
    ROUND(SUM(amount), 2) AS total_amount
FROM orders
GROUP BY
    user_id,
    TUMBLE(order_time, INTERVAL '10' SECOND);
```

10초마다 새로운 집계 결과가 나타난다:

```
+---------+-------------------------+-------------+--------------+
| user_id |     window_start        | order_count | total_amount |
+---------+-------------------------+-------------+--------------+
|      42 | 2026-04-01 10:00:00.000 |           3 |       567.89 |
|       7 | 2026-04-01 10:00:00.000 |           2 |       234.56 |
|      55 | 2026-04-01 10:00:00.000 |           1 |       456.78 |
...
```

> `Ctrl+C` 로 쿼리를 멈추고, `EXIT;` 로 SQL Client 를 종료한다.

---

# Kafka 연동 (Docker Compose)

실무에 가까운 환경을 만들고 싶다면 Kafka 를 함께 띄운다.

## docker-compose-kafka.yml

```yaml
version: "3.8"
services:
  # --- Kafka ---
  kafka:
    image: bitnami/kafka:3.7
    ports:
      - "9092:9092"
    environment:
      - KAFKA_CFG_NODE_ID=0
      - KAFKA_CFG_PROCESS_ROLES=broker,controller
      - KAFKA_CFG_CONTROLLER_QUORUM_VOTERS=0@kafka:9093
      - KAFKA_CFG_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093
      - KAFKA_CFG_ADVERTISED_LISTENERS=PLAINTEXT://kafka:9092
      - KAFKA_CFG_CONTROLLER_LISTENER_NAMES=CONTROLLER
      - KAFKA_CFG_LISTENER_SECURITY_PROTOCOL_MAP=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT

  # --- Flink ---
  jobmanager:
    image: flink:1.20.1-scala_2.12-java11
    ports:
      - "8081:8081"
    command: jobmanager
    environment:
      - |
        FLINK_PROPERTIES=
        jobmanager.rpc.address: jobmanager
        taskmanager.numberOfTaskSlots: 4
    volumes:
      - flink-sql-jars:/opt/flink/lib/extra

  taskmanager:
    image: flink:1.20.1-scala_2.12-java11
    depends_on:
      - jobmanager
    command: taskmanager
    environment:
      - |
        FLINK_PROPERTIES=
        jobmanager.rpc.address: jobmanager
        taskmanager.numberOfTaskSlots: 4
    volumes:
      - flink-sql-jars:/opt/flink/lib/extra

volumes:
  flink-sql-jars:
```

## Kafka 에 데이터 생산하고 Flink 로 소비하기

```bash
# 1. 시작
docker compose -f docker-compose-kafka.yml up -d

# 2. Kafka 토픽 생성
docker compose -f docker-compose-kafka.yml exec kafka \
  kafka-topics.sh --create --topic orders --partitions 3 --bootstrap-server localhost:9092

# 3. Kafka 에 데이터 생산 (터미널 1)
docker compose -f docker-compose-kafka.yml exec kafka \
  kafka-console-producer.sh --topic orders --bootstrap-server localhost:9092
```

터미널 1 에서 JSON 입력:

```json
{"user_id": 1, "product": "shoes", "amount": 99.99}
{"user_id": 2, "product": "shirt", "amount": 49.99}
{"user_id": 1, "product": "hat", "amount": 29.99}
```

```bash
# 4. Flink SQL Client 접속
docker compose -f docker-compose-kafka.yml exec jobmanager ./bin/sql-client.sh
```

```sql
-- 5. Kafka 소스 테이블 생성
CREATE TABLE kafka_orders (
    user_id   INT,
    product   STRING,
    amount    DOUBLE
) WITH (
    'connector' = 'kafka',
    'topic' = 'orders',
    'properties.bootstrap.servers' = 'kafka:9092',
    'properties.group.id' = 'flink-demo',
    'scan.startup.mode' = 'earliest-offset',
    'format' = 'json'
);

-- 6. 실시간 소비 확인
SELECT * FROM kafka_orders;
```

터미널 1 에서 JSON 을 추가로 입력하면, SQL Client 화면에 실시간으로 나타난다.

```sql
-- 7. 유저별 합계 (실시간 집계)
SELECT user_id, COUNT(*) AS cnt, ROUND(SUM(amount), 2) AS total
FROM kafka_orders
GROUP BY user_id;
```

```
+---------+-----+--------+
| user_id | cnt |  total |
+---------+-----+--------+
|       1 |   2 | 129.98 |
|       2 |   1 |  49.99 |
+---------+-----+--------+
```

터미널 1 에서 데이터를 더 입력하면 집계가 실시간으로 업데이트된다.

```bash
# 8. 종료
docker compose -f docker-compose-kafka.yml down
```

---

> **정리**: 가장 빠른 체험은 **로컬 클러스터 + nc 소켓** (5분), 가장 직관적인 체험은 **SQL Client + datagen** (SQL 만으로 확인), 실무에 가까운 체험은 **Docker Compose + Kafka** (JSON 입력 → 실시간 집계).
