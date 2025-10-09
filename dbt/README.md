- [Basic](#basic)
  - [1. DBT란 무엇인가?](#1-dbt란-무엇인가)
  - [2. DBT 도입 배경](#2-dbt-도입-배경)
  - [3. 주요 기능](#3-주요-기능)
  - [4. 장단점](#4-장단점)
    - [장점](#장점)
    - [단점](#단점)
  - [5. CLI vs Cloud](#5-cli-vs-cloud)
  - [6. 프로젝트 구조 예시](#6-프로젝트-구조-예시)
  - [7. 핵심 용어](#7-핵심-용어)
  - [8. 결론](#8-결론)
- [DuckDB Example](#duckdb-example)
  - [0) 사전 준비](#0-사전-준비)
  - [1) 프로젝트 뼈대 생성](#1-프로젝트-뼈대-생성)
  - [2) 프로필 설정 (DuckDB 접속)](#2-프로필-설정-duckdb-접속)
  - [3) 프로젝트 메타 설정](#3-프로젝트-메타-설정)
  - [4) 패키지(매크로) 사용 설정](#4-패키지매크로-사용-설정)
  - [5) Seed 데이터(CSV → 테이블)](#5-seed-데이터csv--테이블)
  - [6) Source 정의 (YAML)](#6-source-정의-yaml)
  - [7) 매크로(Jinja)](#7-매크로jinja)
  - [8) Staging 레이어 (정제 뷰)](#8-staging-레이어-정제-뷰)
  - [9) Marts 레이어 (Dim/Fct + 매크로 활용)](#9-marts-레이어-dimfct--매크로-활용)
  - [10) Snapshot (원천 변경 이력)](#10-snapshot-원천-변경-이력)
  - [11) Analysis (컴파일만; 실행 X)](#11-analysis-컴파일만-실행-x)
  - [12) 실행 · 테스트 · 문서](#12-실행--테스트--문서)
  - [13) (선택) 인크리멘탈 모델](#13-선택-인크리멘탈-모델)
  - [한눈에 보는 전체 흐름](#한눈에-보는-전체-흐름)
- [DataBricks Example](#databricks-example)
  - [0) 사전 준비](#0-사전-준비-1)
  - [1) 프로젝트 뼈대 생성](#1-프로젝트-뼈대-생성-1)
  - [2) 프로필 설정 (Databricks 접속)](#2-프로필-설정-databricks-접속)
  - [3) 프로젝트 메타 설정](#3-프로젝트-메타-설정-1)
  - [4) 패키지(매크로) 의존성](#4-패키지매크로-의존성)
  - [5) Seed 데이터 (CSV → 테이블)](#5-seed-데이터-csv--테이블)
  - [6) Source 정의 (YAML)](#6-source-정의-yaml-1)
  - [7) 매크로(Jinja)](#7-매크로jinja-1)
  - [8) Staging 레이어 (정제 뷰)](#8-staging-레이어-정제-뷰-1)
  - [9) Marts 레이어 (Dim/Fct + 매크로 활용)](#9-marts-레이어-dimfct--매크로-활용-1)
  - [10) Snapshot (원천 변경 이력)](#10-snapshot-원천-변경-이력-1)
  - [11) Analysis (컴파일만; 실행 X)](#11-analysis-컴파일만-실행-x-1)
  - [12) 실행 · 테스트 · 문서](#12-실행--테스트--문서-1)
  - [13) (선택) 인크리멘탈 모델 (MERGE 전략)](#13-선택-인크리멘탈-모델-merge-전략)
  - [한눈에 보는 전체 흐름](#한눈에-보는-전체-흐름-1)

-----

# Basic

## 1. DBT란 무엇인가?

DBT(Data Build Tool)는 **ELT(Extract, Load, Transform) 프로세스 중 Transform 단계**를 전담하는 오픈소스 도구입니다.
즉, 데이터를 추출(Extract)하거나 적재(Load)하는 기능은 없으며, **이미 데이터 웨어하우스에 적재된 데이터를 변환하고 관리하는 데 특화**되어 있습니다.

DBT는 SQL을 코드처럼 관리할 수 있도록 하여, **데이터 모델을 모듈화하고 테스트하며, Git 기반 버전 관리 및 협업**을 가능하게 합니다.

---

## 2. DBT 도입 배경

기존에는 BigQuery 상에서:

* Ad-hoc 쿼리 남발
* 쿼리 재사용 어려움
* 버전 관리 부재로 히스토리 추적 불가
* 성능 중심 SQL 작성으로 가독성과 유지보수성 저하

이런 문제들을 해결하기 위해 **SQL에도 소프트웨어 엔지니어링의 원칙(버전 관리, 모듈화, 협업, 테스트)을 적용**하려는 필요성이 생겼고, DBT가 그 해답으로 떠올랐습니다.

---

## 3. 주요 기능

1. **모델(Model) 기반 SQL 관리**

   * 결과물을 “데이터 모델”로 정의
   * 재사용 가능하고 모듈화된 SQL 작성 가능
   * Jinja 템플릿과 매크로 지원

2. **데이터 품질 보장**

   * 자동화된 테스트 제공 (`unique`, `not_null`, `equal_rowcount` 등)
   * 변환 과정에서 오류 방지

3. **버전 관리 및 협업**

   * Git, Bitbucket 같은 VCS와 연동
   * 코드 리뷰 및 협업 프로세스 지원

4. **확장성**

   * BigQuery, Snowflake, Redshift, Databricks 등 최신 DW와 연동
   * 대규모 데이터 변환에 적합

---

## 4. 장단점

### 장점

* 다양한 **유닛 테스트**로 데이터 품질 강화
* **Docs UI** 제공 → 커뮤니케이션 비용 절감
* **긴 SQL을 모듈화** 가능
* **SQL 리뷰 및 스타일 가이드 적용** 가능
* 풍부한 **매크로/패키지 오픈소스 생태계**
* **추가 리소스 불필요**, 기존 DW 활용

### 단점

* **YAML 관리 부담** (메뉴얼 작업 증가 가능)
* **러닝 커브 존재** (SQL + Jinja + YAML 학습 필요)
* Transform만 지원 → Extract/Load는 별도 도구 필요
* CLI 모드에서는 실행되는 쿼리 추적이 어려움 (Cloud 버전에서 해결 가능)

---

## 5. CLI vs Cloud

* **CLI (무료, 오픈소스)**

  * Python 패키지로 설치 (`pip install dbt-core`)
  * YAML & Jinja 기반 관리
  * 테이블 간 의존성 계산 지원

* **Cloud (유료, 관리형)**

  * 웹 IDE 제공 → SQL 사용자도 쉽게 협업
  * Git 연동 + PR 기반 워크플로우 지원
  * CronJob 기반 스케줄링 가능
  * 팀 단위 협업 및 유지보수 효율성 ↑

---

## 6. 프로젝트 구조 예시

```
├── dbt_project.yml        # 메타 정보
├── models/                # 변환 SQL (메인 디렉토리)
├── macros/                # 사용자 정의 매크로 (Jinja)
├── snapshots/             # 소스 테이블 스냅샷 관리
├── tests/                 # 모델/스냅샷 테스트 정의
├── analysis/              # 분석용 SQL (실행 안 함)
├── data/                  # seed 데이터 (CSV 등)
```

---

## 7. 핵심 용어

* **Model**: 변환된 결과 테이블 정의(SQL 파일 단위)
* **Source**: 원천 데이터/테이블
* **Schema**: 모델에 대한 명세와 테스트 정의(YAML)
* **Macro**: Jinja 기반 함수 정의
* **Snapshot**: 특정 시점의 원본 데이터를 추적

---

## 8. 결론

DBT는 **데이터 변환을 코드처럼 다룰 수 있게 하는 강력한 도구**입니다.
특히 ELT 환경에서 **SQL 품질 관리, 협업, 재사용성**을 확보하는 데 매우 유용합니다.

다만, **도입 시 학습 곡선(SQL+YAML+Jinja)**과 **추가 관리 오버헤드**가 존재하므로,
데이터 팀의 규모와 성숙도, 그리고 플랫폼 구조에 따라 적합성을 판단해야 합니다.

# DuckDB Example

좋아요! 프로젝트명을 **`duckdb_dbt`**로 바꿔서, **로컬에서 바로 실행 가능한 dbt 예제**를 단계별로 다시 정리했습니다. (DuckDB 어댑터 사용)

---

## 0) 사전 준비

```bash
# 0-1) 새 폴더
mkdir duckdb_dbt && cd duckdb_dbt

# 0-2) 가상환경(선택)
python -m venv .venv && source .venv/bin/activate   # (Windows) .venv\Scripts\activate

# 0-3) dbt 설치 (DuckDB 어댑터)
pip install "dbt-core>=1.7,<2.0" "dbt-duckdb>=1.7,<2.0"
```

> DuckDB는 파일형 DB라, 외부 클라우드 없이 바로 실습할 수 있습니다.

---

## 1) 프로젝트 뼈대 생성

```bash
# 1-1) dbt init (프로젝트명: duckdb_dbt)
dbt init duckdb_dbt
# 어댑터는 duckdb 선택 (나중에 profiles.yml로 지정해도 됨)

cd duckdb_dbt
```

생성 후 기본 구조:

```
duckdb_dbt/
├── dbt_project.yml
├── models/
├── macros/
├── snapshots/
├── analyses/          # (dbt 1.6+ 명칭: analyses)
├── seeds/
└── tests/
```

---

## 2) 프로필 설정 (DuckDB 접속)

dbt는 `profiles.yml`을 통해 접속정보를 읽습니다. **프로젝트 내부**에 두고 쓰도록 환경변수를 지정합니다.

```bash
# 현재 디렉토리를 프로필 디렉토리로 지정
export DBT_PROFILES_DIR=$(pwd)     # (Windows PowerShell) $env:DBT_PROFILES_DIR=(Get-Location).Path
```

**`profiles.yml`** (프로젝트 루트에 새로 생성)

```yaml
duckdb_dbt:                    # 프로필 이름 (프로젝트명과 동일 추천)
  target: dev
  outputs:
    dev:
      type: duckdb
      path: ./warehouse/duckdb_dbt.duckdb   # DuckDB DB 파일 (자동생성)
      threads: 4
      # external_root: ./external          # (선택) 외부 테이블 기본 경로
```

---

## 3) 프로젝트 메타 설정

**`dbt_project.yml`** (기존 파일 덮어쓰기)

```yaml
name: "duckdb_dbt"
version: "1.0"
config-version: 2

profile: "duckdb_dbt"

model-paths: ["models"]
macro-paths: ["macros"]
seed-paths: ["seeds"]
snapshot-paths: ["snapshots"]
analysis-paths: ["analyses"]
test-paths: ["tests"]

models:
  duckdb_dbt:
    +materialized: table
    staging:
      +materialized: view
    marts:
      +materialized: table
```

---

## 4) 패키지(매크로) 사용 설정

**`packages.yml`** (프로젝트 루트에 새로 생성)

```yaml
packages:
  - package: dbt-labs/dbt_utils
    version: [">=1.0.0", "<2.0.0"]
```

```bash
dbt deps
```

---

## 5) Seed 데이터(CSV → 테이블)

**`seeds/customers.csv`**

```csv
id,first_name,last_name,email
1,Ana,Robles,ana@example.com
2,Ben,Kim,ben@example.com
3,Chloe,Park,chloe@example.com
```

**`seeds/orders.csv`**

```csv
id,customer_id,order_date,status,amount_cents,updated_at
1,1,2024-01-05,completed,2500,2024-01-05 10:00:00
2,2,2024-01-06,completed,4000,2024-01-06 11:00:00
3,3,2024-01-07,canceled,1500,2024-01-07 12:00:00
4,1,2024-02-01,completed,3000,2024-02-01 09:30:00
```

**`seeds/payments.csv`**

```csv
id,order_id,amount_cents,payment_method,paid_at
10,1,2500,credit_card,2024-01-05 10:05:00
11,2,4000,bank_transfer,2024-01-06 11:10:00
12,4,3000,credit_card,2024-02-01 09:35:00
```

```bash
dbt seed
```

> 결과적으로 DuckDB 내 `main.customers`, `main.orders`, `main.payments`가 만들어집니다.

---

## 6) Source 정의 (YAML)

**`models/sources.yml`**

```yaml
version: 2

sources:
  - name: raw
    schema: main
    tables:
      - name: customers
      - name: orders
      - name: payments
```

---

## 7) 매크로(Jinja)

**`macros/cents_to_dollars.sql`**

```sql
{% macro cents_to_dollars(column_name, precision=2) -%}
  round({{ column_name }} / 100.0, {{ precision }})
{%- endmacro %}
```

---

## 8) Staging 레이어 (정제 뷰)

**`models/staging/stg_customers.sql`**

```sql
with src as (
  select * from {{ source('raw', 'customers') }}
)
select
  cast(id as integer) as customer_id,
  first_name,
  last_name,
  email
from src
```

**`models/staging/stg_orders.sql`**

```sql
with src as (
  select * from {{ source('raw', 'orders') }}
)
select
  cast(id as integer) as order_id,
  cast(customer_id as integer) as customer_id,
  cast(order_date as date) as order_date,
  status,
  cast(amount_cents as integer) as amount_cents,
  cast(updated_at as timestamp) as updated_at
from src
```

**`models/staging/stg_payments.sql`**

```sql
with src as (
  select * from {{ source('raw', 'payments') }}
)
select
  cast(id as integer) as payment_id,
  cast(order_id as integer) as order_id,
  cast(amount_cents as integer) as amount_cents,
  payment_method,
  cast(paid_at as timestamp) as paid_at
from src
```

**`models/staging/schema.yml`** (Staging 테스트)

```yaml
version: 2

models:
  - name: stg_customers
    columns:
      - name: customer_id
        tests: [not_null, unique]
      - name: email
        tests: [not_null]

  - name: stg_orders
    columns:
      - name: order_id
        tests: [not_null, unique]
      - name: customer_id
        tests:
          - not_null
          - relationships:
              to: ref('stg_customers')
              field: customer_id

  - name: stg_payments
    columns:
      - name: payment_id
        tests: [not_null, unique]
      - name: order_id
        tests:
          - not_null
          - relationships:
              to: ref('stg_orders')
              field: order_id
```

---

## 9) Marts 레이어 (Dim/Fct + 매크로 활용)

**`models/marts/dim_customers.sql`**

```sql
select
  c.customer_id,
  c.first_name,
  c.last_name,
  c.email,
  count(o.order_id) as order_count
from {{ ref('stg_customers') }} c
left join {{ ref('stg_orders') }} o
  on c.customer_id = o.customer_id
group by 1,2,3,4
```

**`models/marts/fct_orders.sql`** (매크로 사용)

```sql
with base as (
  select
    o.order_id,
    o.customer_id,
    o.order_date,
    o.status,
    {{ cents_to_dollars('o.amount_cents') }} as amount_usd
  from {{ ref('stg_orders') }} o
)
select * from base
```

**`models/marts/schema.yml`** (품질 테스트 + dbt_utils 예시)

```yaml
version: 2

models:
  - name: dim_customers
    description: "고객 차원 테이블"
    columns:
      - name: customer_id
        tests: [not_null, unique]
      - name: order_count
        tests:
          - not_null
          - dbt_utils.expression_is_true:
              expression: "order_count >= 0"

  - name: fct_orders
    description: "주문 사실 테이블"
    columns:
      - name: order_id
        tests: [not_null, unique]
      - name: amount_usd
        tests:
          - not_null
          - dbt_utils.expression_is_true:
              expression: "amount_usd >= 0"
```

---

## 10) Snapshot (원천 변경 이력)

**`snapshots/orders_snapshot.sql`**

```sql
{% snapshot orders_snapshot %}

{{
  config(
    target_schema='snapshots',
    unique_key='id',
    strategy='timestamp',
    updated_at='updated_at'
  )
}}

select * from {{ source('raw','orders') }}

{% endsnapshot %}
```

```bash
dbt snapshot
```

> 이후 `seeds/orders.csv`를 수정 → `dbt seed` → `dbt snapshot`을 반복하면 변경 이력이 누적됩니다.

---

## 11) Analysis (컴파일만; 실행 X)

**`analyses/adhoc_revenue_by_day.sql`**

```sql
-- dbt compile 시 컴파일 확인용 (dbt run 대상 아님)
select order_date, sum(amount_cents)/100.0 as revenue_usd
from {{ ref('stg_orders') }}
where status = 'completed'
group by order_date
order by order_date
```

```bash
dbt compile
```

---

## 12) 실행 · 테스트 · 문서

```bash
# 의존성
dbt deps

# 시드 적재
dbt seed

# 모델 빌드
dbt run

# 테스트
dbt test

# 스냅샷
dbt snapshot

# 문서 생성/서빙 (브라우저에서 DAG/컬럼 문서 확인)
dbt docs generate
dbt docs serve    # 종료: Ctrl+C
```

---

## 13) (선택) 인크리멘탈 모델

**`models/marts/fct_orders_incremental.sql`**

```sql
{{ config(materialized='incremental', unique_key='order_id') }}

with base as (
  select
    o.order_id,
    o.customer_id,
    o.order_date,
    o.status,
    {{ cents_to_dollars('o.amount_cents') }} as amount_usd
  from {{ ref('stg_orders') }} o
  {% if is_incremental() %}
    where o.order_date > (
      select coalesce(max(order_date), '1900-01-01') from {{ this }}
    )
  {% endif %}
)
select * from base
```

```bash
dbt run --select fct_orders_incremental
```

---

## 한눈에 보는 전체 흐름

```bash
dbt deps
dbt seed
dbt run
dbt test
dbt snapshot
dbt docs generate && dbt docs serve
```

이대로 파일을 생성하고 명령을 실행하면 **`duckdb_dbt`**라는 이름의 완전 동작하는 dbt 실습 프로젝트가 됩니다.

# DataBricks Example

좋아요! 앞서 만든 **폴더/파일 구조와 동일한 형태**를 그대로 유지하되, 실행 대상만 **Databricks(SQL Warehouse + Unity Catalog)** 로 바꾼 실습 프로젝트를 단계별로 만들어봅니다.
프로젝트명은 혼동을 피하기 위해 **`databricks_dbt`** 로 사용하겠습니다.

---

## 0) 사전 준비

```bash
# 0-1) 새 폴더
mkdir databricks_dbt && cd databricks_dbt

# 0-2) 가상환경(선택)
python -m venv .venv && source .venv/bin/activate   # (Windows) .venv\Scripts\activate

# 0-3) dbt 설치 (Databricks 어댑터)
pip install "dbt-core>=1.7,<2.0" "dbt-databricks>=1.7,<2.0"
```

> 실행에는 **Databricks SQL Warehouse**(옛 SQL Endpoint)와 **접속 토큰**이 필요합니다. (워크스페이스에서 미리 발급)

---

## 1) 프로젝트 뼈대 생성

```bash
# 1-1) dbt init (프로젝트명: databricks_dbt)
dbt init databricks_dbt
cd databricks_dbt
```

생성 후 기본 구조:

```
databricks_dbt/
├── dbt_project.yml
├── models/
├── macros/
├── snapshots/
├── analyses/
├── seeds/
└── tests/
```

---

## 2) 프로필 설정 (Databricks 접속)

dbt는 `profiles.yml`에서 접속정보를 읽습니다. **프로젝트 내부**의 profiles.yml을 쓰도록 환경변수를 지정합니다.

```bash
export DBT_PROFILES_DIR=$(pwd)     # (Windows PowerShell) $env:DBT_PROFILES_DIR=(Get-Location).Path
```

**`profiles.yml`** (프로젝트 루트에 생성 · 실제 값으로 교체)

```yaml
databricks_dbt:
  target: dev
  outputs:
    dev:
      type: databricks
      method: token
      host: adb-XXXXXXXX.XX.azuredatabricks.net        # 워크스페이스 호스트
      http_path: /sql/1.0/warehouses/XXXXXXXXXXXXXXX   # SQL Warehouse HTTP Path
      token: "{{ env_var('DATABRICKS_TOKEN') }}"       # 환경변수에 토큰 설정
      catalog: main                                    # Unity Catalog (예: main)
      schema: analytics                                # 스키마(데이터베이스)
      threads: 4
      # (선택) connect_timeout: 30
```

토큰 환경변수 설정 예:

```bash
export DATABRICKS_TOKEN=dapixxxxx...        # Windows PS: $env:DATABRICKS_TOKEN="dapixxxxx..."
```

> **Unity Catalog** 를 사용한다면 `catalog` 필드를 반드시 설정하세요. (예: `main`, `hive_metastore` 등)

---

## 3) 프로젝트 메타 설정

**`dbt_project.yml`** (기존 파일 덮어쓰기)

```yaml
name: "databricks_dbt"
version: "1.0"
config-version: 2

profile: "databricks_dbt"

model-paths: ["models"]
macro-paths: ["macros"]
seed-paths: ["seeds"]
snapshot-paths: ["snapshots"]
analysis-paths: ["analyses"]
test-paths: ["tests"]

models:
  databricks_dbt:
    +materialized: table
    staging:
      +materialized: view
    marts:
      +materialized: table

seeds:
  databricks_dbt:
    +schema: "{{ target.schema }}"   # seeds가 catalog.schema에 생성되도록 보장
```

---

## 4) 패키지(매크로) 의존성

**`packages.yml`**

```yaml
packages:
  - package: dbt-labs/dbt_utils
    version: [">=1.0.0", "<2.0.0"]
```

```bash
dbt deps
```

---

## 5) Seed 데이터 (CSV → 테이블)

**`seeds/customers.csv`**

```csv
id,first_name,last_name,email
1,Ana,Robles,ana@example.com
2,Ben,Kim,ben@example.com
3,Chloe,Park,chloe@example.com
```

**`seeds/orders.csv`**

```csv
id,customer_id,order_date,status,amount_cents,updated_at
1,1,2024-01-05,completed,2500,2024-01-05 10:00:00
2,2,2024-01-06,completed,4000,2024-01-06 11:00:00
3,3,2024-01-07,canceled,1500,2024-01-07 12:00:00
4,1,2024-02-01,completed,3000,2024-02-01 09:30:00
```

**`seeds/payments.csv`**

```csv
id,order_id,amount_cents,payment_method,paid_at
10,1,2500,credit_card,2024-01-05 10:05:00
11,2,4000,bank_transfer,2024-01-06 11:10:00
12,4,3000,credit_card,2024-02-01 09:35:00
```

```bash
dbt seed
```

> 결과적으로 `main.analytics`(= `catalog.schema`) 아래에 `customers`, `orders`, `payments` 테이블이 만들어집니다.

---

## 6) Source 정의 (YAML)

**`models/sources.yml`**

```yaml
version: 2

sources:
  - name: raw
    database: "{{ target.catalog }}"   # Unity Catalog
    schema: "{{ target.schema }}"      # 스키마
    tables:
      - name: customers
      - name: orders
      - name: payments
```

---

## 7) 매크로(Jinja)

**`macros/cents_to_dollars.sql`**

```sql
{% macro cents_to_dollars(column_name, precision=2) -%}
  round({{ column_name }} / 100.0, {{ precision }})
{%- endmacro %}
```

---

## 8) Staging 레이어 (정제 뷰)

**`models/staging/stg_customers.sql`**

```sql
with src as (
  select * from {{ source('raw', 'customers') }}
)
select
  cast(id as int) as customer_id,
  first_name,
  last_name,
  email
from src
```

**`models/staging/stg_orders.sql`**

```sql
with src as (
  select * from {{ source('raw', 'orders') }}
)
select
  cast(id as int) as order_id,
  cast(customer_id as int) as customer_id,
  cast(order_date as date) as order_date,
  status,
  cast(amount_cents as int) as amount_cents,
  cast(updated_at as timestamp) as updated_at
from src
```

**`models/staging/stg_payments.sql`**

```sql
with src as (
  select * from {{ source('raw', 'payments') }}
)
select
  cast(id as int) as payment_id,
  cast(order_id as int) as order_id,
  cast(amount_cents as int) as amount_cents,
  payment_method,
  cast(paid_at as timestamp) as paid_at
from src
```

**`models/staging/schema.yml`** (Staging 테스트)

```yaml
version: 2

models:
  - name: stg_customers
    columns:
      - name: customer_id
        tests: [not_null, unique]
      - name: email
        tests: [not_null]

  - name: stg_orders
    columns:
      - name: order_id
        tests: [not_null, unique]
      - name: customer_id
        tests:
          - not_null
          - relationships:
              to: ref('stg_customers')
              field: customer_id

  - name: stg_payments
    columns:
      - name: payment_id
        tests: [not_null, unique]
      - name: order_id
        tests:
          - not_null
          - relationships:
              to: ref('stg_orders')
              field: order_id
```

---

## 9) Marts 레이어 (Dim/Fct + 매크로 활용)

**`models/marts/dim_customers.sql`**

```sql
select
  c.customer_id,
  c.first_name,
  c.last_name,
  c.email,
  count(o.order_id) as order_count
from {{ ref('stg_customers') }} c
left join {{ ref('stg_orders') }} o
  on c.customer_id = o.customer_id
group by 1,2,3,4
```

**`models/marts/fct_orders.sql`**

```sql
with base as (
  select
    o.order_id,
    o.customer_id,
    o.order_date,
    o.status,
    {{ cents_to_dollars('o.amount_cents') }} as amount_usd
  from {{ ref('stg_orders') }} o
)
select * from base
```

**`models/marts/schema.yml`** (품질 테스트 + dbt_utils)

```yaml
version: 2

models:
  - name: dim_customers
    description: "고객 차원 테이블"
    columns:
      - name: customer_id
        tests: [not_null, unique]
      - name: order_count
        tests:
          - not_null
          - dbt_utils.expression_is_true:
              expression: "order_count >= 0"

  - name: fct_orders
    description: "주문 사실 테이블"
    columns:
      - name: order_id
        tests: [not_null, unique]
      - name: amount_usd
        tests:
          - not_null
          - dbt_utils.expression_is_true:
              expression: "amount_usd >= 0"
```

---

## 10) Snapshot (원천 변경 이력)

**`snapshots/orders_snapshot.sql`**

```sql
{% snapshot orders_snapshot %}

{{
  config(
    target_schema='snapshots',   -- catalog.schema 중 schema 파트
    unique_key='id',
    strategy='timestamp',
    updated_at='updated_at'
  )
}}

select * from {{ source('raw','orders') }}

{% endsnapshot %}
```

```bash
dbt snapshot
```

> 이후 `seeds/orders.csv` 수정 → `dbt seed` → `dbt snapshot` 반복 시 변경 이력이 누적됩니다.

---

## 11) Analysis (컴파일만; 실행 X)

**`analyses/adhoc_revenue_by_day.sql`**

```sql
-- dbt compile 대상(실행 대상 아님)
select order_date, sum(amount_cents)/100.0 as revenue_usd
from {{ ref('stg_orders') }}
where status = 'completed'
group by order_date
order by order_date
```

```bash
dbt compile
```

---

## 12) 실행 · 테스트 · 문서

```bash
# 의존성
dbt deps

# 시드 적재 (catalog.schema로 적재)
dbt seed

# 모델 빌드
dbt run

# 테스트
dbt test

# 스냅샷
dbt snapshot

# 문서 생성/서빙 (브라우저에서 DAG/컬럼 문서 확인)
dbt docs generate
dbt docs serve     # 종료: Ctrl+C
```

---

## 13) (선택) 인크리멘탈 모델 (MERGE 전략)

Databricks에서는 인크리멘탈 시 **MERGE** 전략을 흔히 사용합니다.

**`models/marts/fct_orders_incremental.sql`**

```sql
{{ config(
    materialized='incremental',
    unique_key='order_id',
    incremental_strategy='merge'   -- Databricks 권장
) }}

with base as (
  select
    o.order_id,
    o.customer_id,
    o.order_date,
    o.status,
    {{ cents_to_dollars('o.amount_cents') }} as amount_usd
  from {{ ref('stg_orders') }} o
  {% if is_incremental() %}
    where o.order_date > (select coalesce(max(order_date), to_date('1900-01-01')) from {{ this }})
  {% endif %}
)
select * from base
```

```bash
dbt run --select fct_orders_incremental
```

---

## 한눈에 보는 전체 흐름

```bash
# 사전: DATABRICKS_TOKEN / host / http_path 준비 & profiles.yml 작성
dbt deps
dbt seed
dbt run
dbt test
dbt snapshot
dbt docs generate && dbt docs serve
```

이대로 파일을 생성하고 값을 채우면, **Databricks(SQL Warehouse + Unity Catalog)** 상에서 **동일한 구조**로 dbt를 실습할 수 있습니다.
