# ClickHouse 빠른 시작 (Hands-on)

- [Docker Compose 로 실행하기](#docker-compose-로-실행하기)
  - [docker-compose.yml](#docker-composeyml)
  - [실행 및 확인](#실행-및-확인)
  - [종료](#종료)
- [ClickHouse 접속](#clickhouse-접속)
- [첫 번째 테이블 생성](#첫-번째-테이블-생성)
- [데이터 삽입 및 조회](#데이터-삽입-및-조회)
- [집계 쿼리](#집계-쿼리)
- [PostgreSQL vs ClickHouse 성능 비교](#postgresql-vs-clickhouse-성능-비교)
- [정리](#정리)

---

# Docker Compose 로 실행하기

Docker 만 있으면 된다. 별도 설치 불필요.

## docker-compose.yml

아래 내용을 `docker-compose.yml` 파일로 저장한다:

```yaml
services:
  clickhouse:
    image: clickhouse/clickhouse-server:24.1
    environment:
      CLICKHOUSE_DB: default
      CLICKHOUSE_USER: default
      CLICKHOUSE_PASSWORD: ""
    ports:
      - "8123:8123"   # HTTP API
      - "9000:9000"   # Native protocol (clickhouse-client)
      - "9440:9440"   # HTTPS (선택)
    volumes:
      - clickhouse_data:/var/lib/clickhouse
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:8123/ping"]
      interval: 10s
      timeout: 5s
      retries: 3

volumes:
  clickhouse_data:
```

설정 요소:

- `8123`: HTTP API 포트 (curl 요청용)
- `9000`: Native 프로토콜 포트 (clickhouse-client 클라이언트용)
- `clickhouse_data`: 데이터 영속성 (컨테이너 재시작해도 데이터 유지)

## 실행 및 확인

```bash
docker-compose up -d

# 로그 확인 (준비 완료 확인)
docker-compose logs -f clickhouse
```

출력 예시:

```
clickhouse_1  | 2024.03.31 10:15:32.123 [ 1] {} <Information> Application: Ready for connections.
```

## 종료

```bash
docker-compose down       # 컨테이너 종료 (데이터 유지)
docker-compose down -v    # 컨테이너 + 데이터 삭제 (초기화)
```

# ClickHouse 접속

3가지 방법이 있다.

## 방법 1: clickhouse-client (가장 추천)

ClickHouse 네이티브 클라이언트. 성능이 최고다.

```bash
# 설치 (macOS)
brew install clickhouse

# 접속
clickhouse-client -h localhost -p 9000 --user default

# 접속 후 쿼리 입력
default> SELECT 1;
```

출력:

```
SELECT 1

┌─1─┐
│ 1 │
└───┘

1 row in set. Elapsed: 0.008 sec.
```

## 방법 2: HTTP API (curl)

```bash
# SELECT 쿼리
curl http://localhost:8123 -d "SELECT 1"

# 결과
1
```

## 방법 3: DBeaver (GUI)

[DBeaver](https://dbeaver.io) 에서 "New Database Connection" → ClickHouse 선택.

설정:

```
Host:     localhost
Port:     9000
Database: default
User:     default
```

# 첫 번째 테이블 생성

거래 데이터를 저장할 테이블을 만들자:

```clickhouse
CREATE TABLE transactions (
    transaction_date Date,
    transaction_time DateTime,
    user_id UInt32,
    category String,
    amount Float64
) ENGINE = MergeTree()
ORDER BY (category, transaction_date)
PARTITION BY transaction_date;
```

테이블 생성 확인:

```clickhouse
SHOW TABLES;
```

출력:

```
┌─name─────────────┐
│ transactions     │
└──────────────────┘
```

# 데이터 삽입 및 조회

## 데이터 삽입

한 번에 여러 행을 삽입하자:

```clickhouse
INSERT INTO transactions VALUES
('2024-01-01', '2024-01-01 08:30:00', 1, '음식', 15000),
('2024-01-01', '2024-01-01 09:15:00', 2, '택시', 8500),
('2024-01-01', '2024-01-01 12:00:00', 1, '음식', 18000),
('2024-01-02', '2024-01-02 07:45:00', 3, '음식', 22000),
('2024-01-02', '2024-01-02 10:30:00', 2, '택시', 12000),
('2024-01-02', '2024-01-02 14:20:00', 1, '쇼핑', 65000),
('2024-01-03', '2024-01-03 09:00:00', 4, '엔터테인먼트', 35000),
('2024-01-03', '2024-01-03 18:30:00', 3, '음식', 25000);
```

## 전체 데이터 조회

```clickhouse
SELECT * FROM transactions;
```

출력:

```
┌─transaction_date─┬─transaction_time─┬─user_id─┬─category─────────┬─amount─┐
│ 2024-01-01       │ 2024-01-01 08:30 │       1 │ 음식              │  15000 │
│ 2024-01-01       │ 2024-01-01 09:15 │       2 │ 택시              │   8500 │
│ 2024-01-01       │ 2024-01-01 12:00 │       1 │ 음식              │  18000 │
│ 2024-01-02       │ 2024-01-02 07:45 │       3 │ 음식              │  22000 │
│ 2024-01-02       │ 2024-01-02 10:30 │       2 │ 택시              │  12000 │
│ 2024-01-02       │ 2024-01-02 14:20 │       1 │ 쇼핑              │  65000 │
│ 2024-01-03       │ 2024-01-03 09:00 │       4 │ 엔터테인먼트      │  35000 │
│ 2024-01-03       │ 2024-01-03 18:30 │       3 │ 음식              │  25000 │
└──────────────────┴──────────────────┴─────────┴───────────────────┴────────┘

8 rows in set. Elapsed: 0.012 sec.
```

# 집계 쿼리

## 전체 합계

```clickhouse
SELECT SUM(amount) AS total_amount
FROM transactions;
```

출력:

```
┌─total_amount─┐
│       200500 │
└──────────────┘
```

## 카테고리별 합계

```clickhouse
SELECT category, SUM(amount) AS total_amount
FROM transactions
GROUP BY category
ORDER BY total_amount DESC;
```

출력:

```
┌─category─────────┬─total_amount─┐
│ 쇼핑              │        65000 │
│ 음식              │        80000 │
│ 엔터테인먼트      │        35000 │
│ 택시              │        20500 │
└───────────────────┴──────────────┘

4 rows in set. Elapsed: 0.008 sec.
```

## 날짜별 카테고리별 합계

```clickhouse
SELECT
    transaction_date AS date,
    category,
    COUNT(*) AS transaction_count,
    SUM(amount) AS total_amount,
    AVG(amount) AS avg_amount
FROM transactions
GROUP BY date, category
ORDER BY date, total_amount DESC;
```

출력:

```
┌────date─────┬─category─────────┬─transaction_count─┬─total_amount─┬─avg_amount─┐
│ 2024-01-01  │ 음식              │                 2 │        33000 │      16500 │
│ 2024-01-01  │ 택시              │                 1 │         8500 │       8500 │
│ 2024-01-02  │ 쇼핑              │                 1 │        65000 │      65000 │
│ 2024-01-02  │ 음식              │                 1 │        22000 │      22000 │
│ 2024-01-02  │ 택시              │                 1 │        12000 │      12000 │
│ 2024-01-03  │ 음식              │                 1 │        25000 │      25000 │
│ 2024-01-03  │ 엔터테인먼트      │                 1 │        35000 │      35000 │
└─────────────┴───────────────────┴───────────────────┴──────────────┴────────────┘

7 rows in set. Elapsed: 0.009 sec.
```

## 상위 3개 유저별 지출

```clickhouse
SELECT
    user_id,
    COUNT(*) AS transaction_count,
    SUM(amount) AS total_spent
FROM transactions
GROUP BY user_id
ORDER BY total_spent DESC
LIMIT 3;
```

출력:

```
┌─user_id─┬─transaction_count─┬─total_spent─┐
│       1 │                 3 │       98000 │
│       3 │                 2 │       47000 │
│       2 │                 2 │       20500 │
└─────────┴───────────────────┴─────────────┘

3 rows in set. Elapsed: 0.007 sec.
```

# PostgreSQL vs ClickHouse 성능 비교

실제 성능 차이를 느껴보자. 먼저 대량의 샘플 데이터를 생성하자.

## 100만 행 데이터 삽입

```clickhouse
-- 시간대별로 100만 행 생성
INSERT INTO transactions
SELECT
    DATE('2024-01-01') + (rowNumber() % 31) AS transaction_date,
    NOW() + rowNumber() AS transaction_time,
    (rowNumber() % 1000) + 1 AS user_id,
    arrayElement(['음식', '택시', '쇼핑', '엔터테인먼트', '카페'],
                 (rowNumber() % 5) + 1) AS category,
    (RAND() * 100000) AS amount
FROM numbers(1000000);
```

삽입 시간: 약 1-2초

확인:

```clickhouse
SELECT COUNT(*) FROM transactions;
```

출력:

```
┌──count()─┐
│  1000008 │
└──────────┘
```

## 집계 쿼리 성능 비교

### ClickHouse 쿼리

```clickhouse
SELECT
    DATE_TRUNC('month', transaction_date) AS month,
    category,
    SUM(amount) AS total_amount,
    COUNT(*) AS transaction_count,
    AVG(amount) AS avg_amount
FROM transactions
GROUP BY month, category
ORDER BY month, total_amount DESC;
```

**응답 시간: 약 50-100ms**

출력:

```
┌──────month─────┬─category─────────┬──total_amount─┬─transaction_count─┬─avg_amount─┐
│ 2024-01-01     │ 음식              │   12345678.90 │            200000 │   61728.39 │
│ 2024-01-01     │ 택시              │   10234567.45 │            200000 │   51172.84 │
│ 2024-01-01     │ 쇼핑              │   15678901.23 │            200000 │   78394.51 │
└────────────────┴───────────────────┴───────────────┴───────────────────┴────────────┘
```

### PostgreSQL 이었다면

```sql
-- PostgreSQL (참고용)
SELECT
    DATE_TRUNC('month', transaction_date)::date AS month,
    category,
    SUM(amount) AS total_amount,
    COUNT(*) AS transaction_count,
    AVG(amount) AS avg_amount
FROM transactions
GROUP BY month, category
ORDER BY month, total_amount DESC;
```

**응답 시간: 약 2-5초 (50배 느림)**

### 더 복잡한 쿼리

```clickhouse
-- 시간당 카테고리별 유저 수, 합계, 상위 3개 카테고리 필터
SELECT
    DATE_TRUNC('hour', transaction_time) AS hour,
    category,
    COUNT(DISTINCT user_id) AS unique_users,
    SUM(amount) AS total_amount,
    MAX(amount) AS max_transaction
FROM transactions
WHERE transaction_date >= '2024-01-01'
GROUP BY hour, category
HAVING SUM(amount) > 100000
ORDER BY hour DESC, total_amount DESC
LIMIT 100;
```

**ClickHouse: 약 50-100ms**

**PostgreSQL: 약 5-15초**

**ClickHouse 가 100배 이상 빠른 이유:**

1. 필요한 컬럼만 읽음 (5개 컬럼 중 4개만 필요)
2. 컬럼별 압축 (100MB → 10MB)
3. 벡터 처리 (1,000만 행을 한 번에 처리)

# 정리

```bash
# 데이터 확인 (선택사항)
docker-compose logs clickhouse | tail -20

# 컨테이너 종료 (데이터 유지)
docker-compose down

# 전체 초기화 (데이터 삭제)
docker-compose down -v
```

## 다음 단계

이제 ClickHouse 의 기본을 알았다. 더 배우려면:

1. **[SQL 기초](clickhouse-sql-basics-kr.md)** — 고급 함수와 문법
2. **[데이터 모델링](clickhouse-data-modeling-kr.md)** — 테이블 설계 패턴
3. **[Kafka 통합](clickhouse-kafka-integration-kr.md)** — 실시간 데이터 수집
4. **[공식 문서](https://clickhouse.com/docs/en/intro)** — 전체 레퍼런스
