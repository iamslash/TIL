# ClickHouse vs 대안 솔루션 비교

- [OLAP 데이터베이스란](#olap-데이터베이스란)
- [Column-Oriented vs Wide-Column vs Row-Oriented](#column-oriented-vs-wide-column-vs-row-oriented)
- [ClickHouse 와 같은 카테고리의 솔루션](#clickhouse-와-같은-카테고리의-솔루션)
  - [자체 운영 (오픈소스)](#자체-운영-오픈소스)
  - [관리형 (클라우드)](#관리형-클라우드)
  - [임베디드](#임베디드)
- [솔루션별 상세 비교](#솔루션별-상세-비교)
  - [ClickHouse vs Apache Druid](#clickhouse-vs-apache-druid)
  - [ClickHouse vs Apache Pinot](#clickhouse-vs-apache-pinot)
  - [ClickHouse vs Amazon Redshift](#clickhouse-vs-amazon-redshift)
  - [ClickHouse vs Google BigQuery](#clickhouse-vs-google-bigquery)
  - [ClickHouse vs Snowflake](#clickhouse-vs-snowflake)
  - [ClickHouse vs DuckDB](#clickhouse-vs-duckdb)
  - [ClickHouse vs ScyllaDB](#clickhouse-vs-scylladb)
- [상황별 추천](#상황별-추천)
- [정리](#정리)

---

## OLAP 데이터베이스란

OLAP (Online Analytical Processing) 데이터베이스는 **대량 데이터를 빠르게 집계/분석** 하는 데 특화된 데이터베이스다. 일반적인 웹 서비스에서 사용하는 PostgreSQL, MySQL 같은 OLTP (Online Transaction Processing) 데이터베이스와 목적이 다르다.

```
OLTP (PostgreSQL, MySQL):
  "user-12345 의 프로필을 가져와"          → 1행 조회, 1ms
  "주문 1건을 INSERT 해줘"                → 1행 쓰기, 1ms

OLAP (ClickHouse, Redshift, BigQuery):
  "지난 1년간 월별 매출 합계를 구해줘"      → 10억 행 집계, 1초
  "사용자 활동 패턴이 시간대별로 어떻게 다른가" → 5억 행 GROUP BY, 2초
```

## Column-Oriented vs Wide-Column vs Row-Oriented

이름이 비슷해서 자주 혼동되는 세 가지 저장 방식이 있다.

### Row-Oriented (행 지향) — PostgreSQL, MySQL

데이터를 **행 단위로 저장**한다. 한 행의 모든 컬럼이 디스크에 나란히 위치한다.

```
디스크 저장:
  [1, "김철수", 25] [2, "이영희", 30] [3, "박민수", 28]
```

- 단일 행 조회가 빠르다 (한번 읽으면 모든 컬럼이 나온다)
- 집계가 느리다 (`SELECT avg(age)` 를 하려면 불필요한 name 도 전부 읽어야 한다)

### Column-Oriented (컬럼 지향) — ClickHouse, Redshift, BigQuery

데이터를 **컬럼 단위로 물리적으로 저장**한다.

```
디스크 저장:
  id 파일:   [1, 2, 3]
  name 파일: ["김철수", "이영희", "박민수"]
  age 파일:  [25, 30, 28]
```

- 집계가 빠르다 (`SELECT avg(age)` → age 파일만 읽으면 끝)
- 같은 타입 데이터가 모여서 압축률이 90% 이상
- 단일 행 조회가 느리다 (여러 컬럼 파일을 합쳐야 한다)

### Wide-Column (와이드 컬럼) — ScyllaDB, Cassandra, HBase

**행마다 컬럼이 다를 수 있는 유연한 스키마**를 제공한다. 저장은 행 단위다.

```
Partition Key: user-123
  ├── name: "김철수"
  ├── age: 25
  └── email: "kim@test.com"

Partition Key: user-456
  ├── name: "이영희"
  ├── phone: "010-1234"     ← user-123 에는 없는 컬럼
  └── premium: true         ← user-123 에는 없는 컬럼
```

"Wide-Column" 은 **행마다 컬럼 집합이 넓어질 수 있다**는 뜻이지, 컬럼 단위로 저장한다는 뜻이 아니다.

### 한눈에 비교


|         | Column-Oriented      | Wide-Column         | Row-Oriented      |
| ------- | -------------------- | ------------------- | ----------------- |
| 대표      | ClickHouse, Redshift | ScyllaDB, Cassandra | PostgreSQL, MySQL |
| 물리 저장   | 컬럼별 파일               | 행(파티션) 단위           | 행 단위              |
| 집계 쿼리   | 매우 빠름                | 느림 (불가에 가까움)        | 보통                |
| 단일 행 조회 | 느림                   | 매우 빠름 (< 1ms)       | 빠름                |
| 용도      | 분석/대시보드              | 대량 읽기/쓰기 서빙         | 범용                |


---

## ClickHouse 와 같은 카테고리의 솔루션

### 자체 운영 (오픈소스)


| 솔루션              | 특징                                      | 운영 난이도       |
| ---------------- | --------------------------------------- | ------------ |
| **ClickHouse**   | 가장 빠른 단일 노드 성능, Kafka Engine 내장, SQL 호환 | 중간           |
| **Apache Druid** | 실시간 수집 + 서브초 쿼리, Kafka 네이티브 수집          | 높음 (5개 프로세스) |
| **Apache Pinot** | LinkedIn 이 만듦, 실시간 OLAP, 스타 스키마         | 높음           |
| **Apache Doris** | MySQL 프로토콜 호환, 중국에서 인기                  | 중간           |
| **StarRocks**    | Doris 포크, 벡터화 실행 엔진                     | 중간           |


### 관리형 (클라우드)


| 솔루션                  | 클라우드          | 특징                            | 과금 방식       |
| -------------------- | ------------- | ----------------------------- | ----------- |
| **Amazon Redshift**  | AWS           | S3 연동, Spectrum 으로 데이터 레이크 쿼리 | 노드 시간       |
| **Google BigQuery**  | GCP           | 서버리스, 페타바이트 스케일               | 스캔한 데이터량    |
| **Snowflake**        | AWS/GCP/Azure | 컴퓨팅/스토리지 분리, 데이터 공유           | 크레딧         |
| **Databricks SQL**   | AWS/GCP/Azure | Delta Lake + Spark SQL, ML 통합 | DBU (처리 단위) |
| **ClickHouse Cloud** | AWS/GCP       | ClickHouse 관리형                | 컴퓨팅 + 스토리지  |


### 임베디드


| 솔루션        | 특징                                                                                   |
| ---------- | ------------------------------------------------------------------------------------ |
| **DuckDB** | SQLite 의 OLAP 버전. 서버 불필요, `pip install duckdb` 로 바로 사용. Python/R 에서 Parquet 파일 직접 쿼리 |


---

## 솔루션별 상세 비교

### ClickHouse vs Apache Druid


|        | ClickHouse   | Druid                                                         |
| ------ | ------------ | ------------------------------------------------------------- |
| 쿼리 언어  | SQL          | SQL (제한적) + Druid 네이티브 쿼리                                     |
| JOIN   | 지원           | 제한적 (lookup join 만)                                           |
| 실시간 수집 | Kafka Engine | 네이티브 Kafka Indexing                                           |
| 운영 구성  | 단일 바이너리      | Historical, Broker, Coordinator, Overlord, MiddleManager (5종) |
| 적합한 곳  | 범용 분석        | 실시간 대시보드 (Superset 등)                                         |


**선택 기준:** JOIN 이 필요하거나 SQL 을 그대로 쓰고 싶으면 ClickHouse. 서브초 실시간 대시보드가 최우선이면 Druid.

### ClickHouse vs Apache Pinot


|        | ClickHouse         | Pinot                       |
| ------ | ------------------ | --------------------------- |
| 만든 곳   | Yandex (러시아)       | LinkedIn                    |
| 스키마    | 자유 SQL             | 스타 스키마 (dimension + metric) |
| Upsert | ReplacingMergeTree | 네이티브 upsert 지원              |
| 생태계    | 넓음                 | 상대적으로 작음                    |


**선택 기준:** 범용성은 ClickHouse. LinkedIn 스타일 실시간 분석 파이프라인이면 Pinot.

### ClickHouse vs Amazon Redshift


|        | ClickHouse   | Redshift               |
| ------ | ------------ | ---------------------- |
| 운영     | 자체 운영        | AWS 관리형                |
| 비용     | 무료 (인프라 비용만) | ~~$0.25/시간/노드~~        |
| S3 연동  | S3 함수 지원     | Spectrum 으로 네이티브       |
| 동시 쿼리  | 수백           | WLM 으로 제한 (기본 5~50)    |
| AWS 연동 | 별도 구성        | IAM, Glue, Lambda 네이티브 |


**선택 기준:** AWS 올인 + 관리 부담 최소화면 Redshift. 비용 절감 + 높은 동시성이면 ClickHouse.

### ClickHouse vs Google BigQuery


|       | ClickHouse | BigQuery         |
| ----- | ---------- | ---------------- |
| 운영    | 자체 운영      | 완전 서버리스          |
| 과금    | 인프라 비용     | 스캔한 데이터량 ($5/TB) |
| 응답 속도 | ms~초       | 초~십초 (콜드 스타트)    |
| 스토리지  | 로컬/S3      | Google 관리 (자동)   |


**선택 기준:** GCP 환경이면 BigQuery. 빠른 응답 속도가 중요하면 ClickHouse.

### ClickHouse vs Snowflake


|        | ClickHouse         | Snowflake       |
| ------ | ------------------ | --------------- |
| 운영     | 자체 운영              | 관리형 (멀티 클라우드)   |
| 특장점    | 성능, 비용             | 데이터 공유, 멀티 클라우드 |
| 컴퓨팅 분리 | ClickHouse Cloud 만 | 기본 (웨어하우스 개념)   |
| 동시 쿼리  | 높음                 | 웨어하우스 크기에 비례    |


**선택 기준:** 여러 팀/조직간 데이터 공유가 중요하면 Snowflake. 단일 팀 분석이면 ClickHouse.

### ClickHouse vs DuckDB


|        | ClickHouse  | DuckDB                  |
| ------ | ----------- | ----------------------- |
| 배포     | 서버 (docker) | 라이브러리 (`import duckdb`) |
| 동시 접속  | 다수          | 단일 프로세스                 |
| 데이터 규모 | TB~PB       | GB~수십 GB                |
| 용도     | 서비스 백엔드 분석  | 로컬 노트북 분석               |


**선택 기준:** 서버로 운영하면 ClickHouse. Python 스크립트에서 빠른 분석이면 DuckDB.

### ClickHouse vs ScyllaDB

이 둘은 **경쟁 관계가 아니라 용도가 완전히 다르다.**


|     | ClickHouse               | ScyllaDB                       |
| --- | ------------------------ | ------------------------------ |
| 유형  | OLAP (분석)                | OLTP (서빙)                      |
| 저장  | Column-Oriented          | Wide-Column (행 기반)             |
| 강점  | `GROUP BY`, 집계           | 단일 행 읽기/쓰기 p99 < 1ms           |
| 약점  | 단일 행 조회 느림               | 집계/JOIN 불가                     |
| 쿼리  | SQL                      | CQL (Cassandra Query Language) |
| 용도  | 대시보드, 리포트, ML feature 계산 | 프로필 조회, 세션 저장, 타임라인            |


```
ClickHouse: "지난 30일간 사용자 활동 패턴을 분석해줘" (10억 행 → 1초)
ScyllaDB:   "user-12345 의 최근 메시지 50개 가져와" (0.5ms)
```

이 둘은 **함께 쓰는 것**이 일반적이다:

- ClickHouse 에서 분석 → ScyllaDB 에 결과 저장 → 서비스가 ScyllaDB 에서 조회

---

## 상황별 추천


| 상황               | 추천                     | 이유                          |
| ---------------- | ---------------------- | --------------------------- |
| AWS 올인, 관리형 원함   | **Redshift**           | IAM/Glue/Lambda 네이티브 연동     |
| GCP 올인, 서버리스 원함  | **BigQuery**           | 인프라 관리 0, 쿼리 기반 과금          |
| 멀티 클라우드 + 데이터 공유 | **Snowflake**          | 조직간 데이터 마켓플레이스              |
| ML + 분석 통합       | **Databricks SQL**     | Delta Lake + MLflow + Spark |
| 자체 운영, 최고 성능     | **ClickHouse**         | 오픈소스, 단일 노드 최강              |
| 실시간 대시보드 (서브초)   | **Druid** 또는 **Pinot** | 미리 집계된 세그먼트 서빙              |
| 노트북에서 로컬 분석      | **DuckDB**             | 서버 불필요, pip install         |
| 대량 행 읽기/쓰기 서빙    | **ScyllaDB**           | Cassandra 호환, p99 < 1ms     |
| 로컬 Databricks 대체 | **ClickHouse**         | Kafka Engine, SQL 호환, 가볍다   |


---

## 정리

```
분석/집계가 필요하면 → Column-Oriented DB (ClickHouse, Redshift, BigQuery)
대량 서빙이 필요하면 → Wide-Column DB (ScyllaDB, Cassandra, DynamoDB)
범용 트랜잭션이면   → Row-Oriented DB (PostgreSQL, MySQL)
```

ClickHouse 는 **자체 운영 OLAP 중 가장 쉽고 빠른 선택**이다. 관리형이 필요하면 클라우드에 맞는 서비스 (Redshift/BigQuery/Snowflake) 를 선택하면 된다.