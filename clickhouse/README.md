- [ClickHouse란?](#clickhouse란)
  - [기본 정보](#기본-정보)
  - [주요 특징](#주요-특징)
  - [예제 쿼리](#예제-쿼리)
  - [ClickHouse 활용 기업](#clickhouse-활용-기업)
  - [ClickHouse vs 전통적인 RDBMS](#clickhouse-vs-전통적인-rdbms)
  - [요약](#요약)
  - [참고 링크](#참고-링크)
- [ClickHouse `user_events` 테이블 저장 구조 설명](#clickhouse-user_events-테이블-저장-구조-설명)
  - [테이블 정의](#테이블-정의)
  - [저장 구조](#저장-구조)
  - [데이터 저장 단계](#데이터-저장-단계)
  - [예시: `user_id` 저장 구조](#예시-user_id-저장-구조)
  - [쿼리 최적화 요소](#쿼리-최적화-요소)
  - [요약](#요약-1)
  - [참고](#참고)
- [ClickHouse `.bin` 및 `.mrk3` 파일 상세 분석](#clickhouse-bin-및-mrk3-파일-상세-분석)
  - [1. `.bin` 파일: 컬럼 데이터 저장소](#1-bin-파일-컬럼-데이터-저장소)
    - [저장 개념](#저장-개념)
    - [구조](#구조)
    - [압축 포맷 예 (LZ4)](#압축-포맷-예-lz4)
  - [2. `.mrk3` 파일: block 위치 인덱스](#2-mrk3-파일-block-위치-인덱스)
    - [저장 개념](#저장-개념-1)
    - [구조 (1 mark = 16 bytes)](#구조-1-mark--16-bytes)
    - [예시](#예시)
  - [3. 쿼리 시 동작 예시](#3-쿼리-시-동작-예시)
  - [4. 파일 위치](#4-파일-위치)
  - [5. 파일 크기 계산](#5-파일-크기-계산)
  - [요약](#요약-2)
  - [참고](#참고-1)
- [ClickHouse에서 `ORDER BY` 변경 방법 및 내부 동작 분석](#clickhouse에서-order-by-변경-방법-및-내부-동작-분석)
  - [1. `ORDER BY`는 ALTER로 변경할 수 없다](#1-order-by는-alter로-변경할-수-없다)
  - [2. 해결 방법: 새 테이블 생성 후 데이터 복사](#2-해결-방법-새-테이블-생성-후-데이터-복사)
  - [3. 내부 동작: INSERT SELECT 실행 시](#3-내부-동작-insert-select-실행-시)
  - [4. 디스크 파일 변화 예](#4-디스크-파일-변화-예)
  - [5. 주의사항](#5-주의사항)
  - [6. 요약](#6-요약)
  - [참고](#참고-2)
- [ClickHouse 인덱스 종류 정리](#clickhouse-인덱스-종류-정리)
  - [인덱스 분류](#인덱스-분류)
  - [1. 기본 인덱스: `ORDER BY`](#1-기본-인덱스-order-by)
  - [2. Skip Index: `CREATE INDEX`로 명시적으로 생성](#2-skip-index-create-index로-명시적으로-생성)
    - [2.1 `minmax`](#21-minmax)
    - [2.2 `set(N)`](#22-setn)
    - [2.3 `bloom_filter(p)`](#23-bloom_filterp)
    - [2.4 `ngrambf_v1(n, p)`](#24-ngrambf_v1n-p)
    - [2.5 `tokenbf_v1(p)`](#25-tokenbf_v1p)
  - [인덱스 비교 표](#인덱스-비교-표)
  - [주의사항](#주의사항)
  - [요약](#요약-3)
  - [참고](#참고-3)
- [ClickHouse가 지원하는 ENGINE 종류](#clickhouse가-지원하는-engine-종류)
  - [ENGINE 분류 요약](#engine-분류-요약)
  - [1. MergeTree 계열 (저장용)](#1-mergetree-계열-저장용)
    - [MergeTree](#mergetree)
    - [ReplacingMergeTree](#replacingmergetree)
    - [SummingMergeTree](#summingmergetree)
    - [AggregatingMergeTree](#aggregatingmergetree)
  - [2. Distributed (분산 처리용)](#2-distributed-분산-처리용)
  - [3. 외부 연동용 엔진](#3-외부-연동용-엔진)
    - [Kafka](#kafka)
    - [MySQL](#mysql)
    - [S3](#s3)
  - [4. 메모리 및 임시용](#4-메모리-및-임시용)
    - [Memory](#memory)
    - [Null](#null)
    - [File](#file)
  - [5. View / MaterializedView](#5-view--materializedview)
    - [View](#view)
    - [MaterializedView](#materializedview)
  - [요약](#요약-4)
  - [참고](#참고-4)
- [ClickHouse가 제공하는 API 종류](#clickhouse가-제공하는-api-종류)
  - [API 요약 표](#api-요약-표)
  - [1. HTTP API](#1-http-api)
    - [특징](#특징)
    - [예시](#예시-1)
  - [2. Native TCP API](#2-native-tcp-api)
    - [예시](#예시-2)
  - [3. HTTPS API / TCP over TLS](#3-https-api--tcp-over-tls)
  - [4. ODBC / JDBC](#4-odbc--jdbc)
    - [JDBC 예시](#jdbc-예시)
    - [ODBC](#odbc)
  - [5. Prometheus API](#5-prometheus-api)
    - [엔드포인트](#엔드포인트)
  - [6. System Tables](#6-system-tables)
    - [예시](#예시-3)
  - [요약](#요약-5)

-----

# ClickHouse란?

**ClickHouse**는 초고속 대용량 분석(OLAP)을 위한 **열 지향(columnar) 데이터베이스**입니다.  
광고 클릭 로그 분석을 위해 Yandex에서 개발되었으며, 지금은 전 세계에서 실시간 로그 분석, BI 리포트, 대시보드 등에 널리 사용됩니다.

---

## 기본 정보

| 항목       | 설명                                       |
|------------|--------------------------------------------|
| 이름       | Click + Data Warehouse = ClickHouse         |
| 용도       | OLAP (분석용 DB)                           |
| 구조       | Columnar storage + SIMD + 인덱스 스킵       |
| 작성 언어  | C++                                        |
| 라이선스   | Apache 2.0 (오픈소스)                       |
| 사용 예시  | 광고 분석, 로그 집계, 실시간 대시보드       |

---

## 주요 특징

- Column-oriented 저장 방식
- Block-level skip index 지원
- 벡터화 실행(SIMD) 최적화
- 고압축 성능
- SQL 지원 (JOIN, GROUP BY, WHERE 등)
- 분산 클러스터 구성 가능 (Shard, Replica)

---

## 예제 쿼리

```sql
SELECT
  toDate(log_time) AS date,
  count() AS error_count
FROM logs
WHERE level = 'error'
GROUP BY date
ORDER BY date DESC
```

---

## ClickHouse 활용 기업

| 기업       | 사용 용도              |
|------------|------------------------|
| Yandex     | 검색/광고 로그 분석     |
| Cloudflare | 실시간 네트워크 분석    |
| Alibaba    | 광고 리포트, 실시간 집계 |
| Lyft       | 사용자 이벤트 분석      |
| Ubisoft    | 게임 이벤트 트래킹     |

---

## ClickHouse vs 전통적인 RDBMS

| 항목        | ClickHouse                        | MySQL / PostgreSQL           |
|-------------|-----------------------------------|-------------------------------|
| 저장 방식   | 열 지향(Columnar)                 | 행 지향(Row-based)           |
| 용도        | OLAP (분석)                       | OLTP (트랜잭션)              |
| INSERT      | 빠르지만 대량 삽입에 최적화       | 단건 삽입 빠름               |
| SELECT 성능 | 대량 집계 및 필터링에 매우 빠름    | 일반 SELECT는 빠르지만 대량 분석엔 부적합 |
| 인덱스      | Block skipping 기반 인덱스         | B-Tree, Hash, Fulltext 등 전통적 인덱스 |

---

## 요약

- ClickHouse는 대규모 분석/집계에 최적화된 열 지향 DBMS입니다.
- 실시간 대시보드, 로그 분석, 광고 리포트에 적합합니다.
- SQL을 그대로 사용하면서 수십억 건도 빠르게 쿼리할 수 있습니다.

---

## 참고 링크

- 공식 홈페이지: https://clickhouse.com/
- GitHub 저장소: https://github.com/ClickHouse/ClickHouse
- ClickHouse Play (온라인 SQL 테스트): https://play.clickhouse.com/

# ClickHouse `user_events` 테이블 저장 구조 설명

`user_events`는 사용자 이벤트 로그를 저장하는 테이블입니다.  
ClickHouse의 `MergeTree` 엔진을 사용하며, 데이터를 컬럼 단위로 압축 저장하고, mark 파일을 통해 빠른 block skipping을 지원합니다.

---

## 테이블 정의

```sql
CREATE TABLE user_events (
  event_id        UUID,
  user_id         UInt64,
  event_type      String,
  event_time      DateTime,
  page_url        String,
  device_type     LowCardinality(String),
  location        String
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(event_time)
ORDER BY (user_id, event_time)
SETTINGS index_granularity = 8192;
```

---

## 저장 구조

ClickHouse는 이 테이블을 다음 구조로 저장합니다:

```
/var/lib/clickhouse/data/<database>/user_events/
  └── 202406_1_1_0/
        ├── event_id.bin
        ├── event_id.mrk3
        ├── user_id.bin
        ├── user_id.mrk3
        ├── ...
        ├── checksums.txt
        ├── columns.txt
        └── count.txt
```

| 파일명             | 설명                                      |
|--------------------|-------------------------------------------|
| `*.bin`            | 컬럼 데이터를 block 단위로 압축 저장       |
| `*.mrk3`           | mark 파일. block 시작 offset 정보 포함     |
| `checksums.txt`    | 파일 무결성 확인용 해시/크기 정보         |
| `columns.txt`      | 이 part에 포함된 컬럼 목록                 |
| `count.txt`        | 이 part가 가진 row 수                      |

---

## 데이터 저장 단계

1. INSERT 발생
   - 데이터는 `event_time` 기준으로 파티션 (`202406`)에 배정됨

2. 컬럼 단위 저장
   - 각 컬럼은 `.bin` 파일에 압축 저장됨
   - 예: `user_id.bin`, `event_type.bin` 등

3. mark 파일 생성
   - `.mrk3` 파일은 각 컬럼의 block 위치 정보를 포함

4. 파티션/Part 구성
   - 디렉토리 이름: `202406_1_1_0`
   - 형식: `partition_min-max_level`

---

## 예시: `user_id` 저장 구조

```
user_id.bin   → 압축된 UInt64 데이터 블록
user_id.mrk3  → block 단위 offset 정보
```

block 예시:

- block 0: [1001, 1002, 1003] (압축됨)
- block 1: [1010, 1015, 1018] (압축됨)

---

## 쿼리 최적화 요소

| 항목               | 설명                                            |
|--------------------|-------------------------------------------------|
| `PARTITION BY`     | 파티션 단위로 디렉토리 구분, 빠른 파티션 프루닝 |
| `ORDER BY`         | 정렬된 컬럼 기준으로 block skipping 가능        |
| `index_granularity`| block 당 row 수 (기본 8192)                     |

---

## 요약

- 데이터는 컬럼 단위로 압축되어 `.bin`에 저장됨
- block 단위 위치는 `.mrk3`에 기록됨
- 쿼리 시 필요한 block만 빠르게 `seek()`해서 읽을 수 있음
- ClickHouse는 `MergeTree` 기반 저장 구조를 통해 초고속 분석이 가능함

---

## 참고

- mark 구조 (`mrk3`)는 block skipping의 핵심 요소
- `LowCardinality(String)`은 사전 압축(사전 인코딩)된 문자열 컬럼
- `checksums.txt`는 파일 손상 여부 검증에 사용됨


# ClickHouse `.bin` 및 `.mrk3` 파일 상세 분석

ClickHouse는 `MergeTree` 엔진을 사용할 때 데이터를 **컬럼 단위로 압축하여 `.bin` 파일에 저장**하고,  
**각 컬럼의 block 위치 정보를 `.mrk3` 파일에 기록**합니다.  
이를 통해 필요한 데이터 block만 `seek()`해서 읽어오는 고속 쿼리가 가능합니다.

---

## 1. `.bin` 파일: 컬럼 데이터 저장소

### 저장 개념

- 하나의 `.bin` 파일 = 하나의 컬럼의 값들
- row block 단위로 압축된 데이터가 연속 저장됨
- 기본 압축 방식: LZ4 (또는 ZSTD 등)

### 구조

```
[ Compressed Block 1 ]
  └─ Header
  └─ 압축된 row 값들

[ Compressed Block 2 ]
  └─ Header
  └─ 압축된 row 값들

...
```

- 한 block에는 `index_granularity` (기본: 8192)개의 row가 들어감
- 각 block은 독립적으로 압축되어 있음 → 병렬 처리 가능

### 압축 포맷 예 (LZ4)

- Codec ID (1 byte)
- Compressed size (3 bytes)
- Uncompressed size (3 bytes)
- 압축된 payload (N bytes)
- 패딩 (최대 7 bytes)

---

## 2. `.mrk3` 파일: block 위치 인덱스

### 저장 개념

- `.mrk3`는 각 컬럼의 block 위치를 담은 **"책갈피(index)" 파일**
- block 단위로 `.bin` 파일에서 어디를 읽어야 할지 알려줌
- block skipping과 빠른 `seek()`을 가능하게 함

### 구조 (1 mark = 16 bytes)

```
mark[n] = {
  offset_in_compressed_file: UInt64,  // .bin 내 시작 위치 (byte 단위)
  offset_in_decompressed_block: UInt64 // 압축 해제된 block 내 row offset (보통 0)
}
```

### 예시

만약 `user_id.bin` 파일이 다음과 같이 구성되어 있다고 가정:

```
Block 0 → 압축 후 23 bytes  
Block 1 → 압축 후 30 bytes  
Block 2 → 압축 후 17 bytes
```

그럼 `user_id.mrk3`는 다음 내용을 가질 수 있음:

```
mark[0] = (offset: 0, decompress_offset: 0)  
mark[1] = (offset: 23, decompress_offset: 0)  
mark[2] = (offset: 53, decompress_offset: 0)
```

→ 쿼리 시 mark를 참고해 `.bin` 파일에서 필요한 block만 읽음

---

## 3. 쿼리 시 동작 예시

쿼리:
```sql
SELECT user_id FROM user_events WHERE user_id > 5000;
```

동작 흐름:

1. ClickHouse는 `user_id.mrk3`를 읽어 offset 목록을 로딩
2. WHERE 조건이 적용되는 block만 판별 (minmax 또는 Skip Index 사용)
3. 해당 block의 offset을 기준으로 `user_id.bin`을 `seek()`
4. 압축 해제 후 필요한 row만 반환

---

## 4. 파일 위치

ClickHouse는 파티션별 디렉토리에 다음과 같이 파일을 저장합니다:

```
/var/lib/clickhouse/data/mydb/user_events/202406_1_1_0/
  ├── user_id.bin
  ├── user_id.mrk3
  ├── ...
```

`.bin`, `.mrk3`는 컬럼마다 존재하며, block 수에 따라 `.mrk3` 크기가 달라집니다.

---

## 5. 파일 크기 계산

- `.mrk3` 크기 = block 수 × 16 bytes
- block 수 = row 수 ÷ index_granularity

예:
- 1천만 row / 8192 = 약 1221 block  
- `.mrk3` = 1221 × 16 = 약 19.5 KB

---

## 요약

| 파일명      | 내용 |
|-------------|------|
| `*.bin`     | 컬럼 데이터 (block 단위 압축 저장) |
| `*.mrk3`    | 각 block의 시작 위치와 offset을 저장한 index |
| 활용 목적    | 쿼리 시 필요한 block만 seek해서 고속 처리 가능 |

ClickHouse의 `.bin` + `.mrk3` 구조는 **대용량 데이터를 빠르게 읽기 위한 핵심 설계**입니다.

---

## 참고

- block 수와 `index_granularity`를 튜닝하면 성능과 디스크 사용량을 조절할 수 있음
- `.bin` 파일 포맷은 내부 압축 형식에 따라 약간씩 다름 (ZSTD vs LZ4)
- `.mrk3`는 row-level이 아닌 **block-level index**임

# ClickHouse에서 `ORDER BY` 변경 방법 및 내부 동작 분석

ClickHouse의 MergeTree 엔진에서 테이블 생성 시 지정한 `ORDER BY`는  
**한 번 설정되면 변경할 수 없습니다.**

따라서, `ORDER BY (user_id)`로 만들었던 테이블을 `ORDER BY (user_id, event_time)`으로 바꾸고 싶다면  
**새 테이블을 만들고 데이터를 복사해야 합니다.**

---

## 1. `ORDER BY`는 ALTER로 변경할 수 없다

- ClickHouse는 `ALTER TABLE ... ORDER BY`를 지원하지 않음
- 이유: MergeTree는 데이터 파일들을 `ORDER BY` 기준으로 디스크에 정렬 저장하기 때문
- 해결 방법: **새 테이블 생성 + INSERT SELECT**

---

## 2. 해결 방법: 새 테이블 생성 후 데이터 복사

```sql
-- 1. 새 테이블 생성
CREATE TABLE user_events_new (
  event_id        UUID,
  user_id         UInt64,
  event_type      String,
  event_time      DateTime,
  page_url        String,
  device_type     LowCardinality(String),
  location        String
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(event_time)
ORDER BY (user_id, event_time)
SETTINGS index_granularity = 8192;

-- 2. 기존 테이블에서 데이터 복사
INSERT INTO user_events_new
SELECT * FROM user_events;

-- 3. (선택) 기존 테이블을 교체
RENAME TABLE user_events TO user_events_old, user_events_new TO user_events;
```

---

## 3. 내부 동작: INSERT SELECT 실행 시

1. 기존 테이블(`user_events`)에서 데이터를 block 단위로 읽음  
2. 새로운 테이블의 `ORDER BY (user_id, event_time)`에 따라 **정렬**  
3. 컬럼 단위로 `.bin` 파일 생성 (압축 저장)  
4. block 단위 인덱스(`.mrk3`) 새로 생성  
5. 각 파티션 디렉토리(`202406_1_1_0` 등)가 새로 만들어짐

---

## 4. 디스크 파일 변화 예

```
/var/lib/clickhouse/data/mydb/user_events_new/
  └── 202406_1_1_0/
        ├── event_id.bin
        ├── event_id.mrk3
        ├── ...
        ├── checksums.txt
        └── count.txt
```

- 모든 데이터 파트가 새로 생성됨
- 기존 테이블의 파일은 변경되지 않음

---

## 5. 주의사항

| 항목                         | 설명 |
|------------------------------|------|
| `ORDER BY`는 ALTER 불가      | CREATE TABLE로 새로 만들어야 함 |
| INSERT SELECT는 full rewrite | 모든 데이터가 새로 정렬되어 디스크에 저장됨 |
| 디스크 사용량 일시 증가      | 두 테이블이 동시에 존재함 |
| 대용량일 경우 비용 큼        | CPU, I/O 자원 소모 큼 |
| 트랜잭션 없음                | 중간 실패 시 중복 or 불완전 파트 주의 |

---

## 6. 요약

| 질문 | 답변 |
|------|------|
| 기존 테이블의 `ORDER BY`를 변경할 수 있나요? | ❌ ALTER로는 불가능 |
| 변경하려면 어떻게 해야 하나요? | ✅ 새 테이블 생성 후 INSERT SELECT로 데이터 이동 |
| 내부적으로 어떤 일이 발생하나요? | ✅ 데이터 재정렬, 새 파일 생성 (`.bin`, `.mrk3`) |
| 디스크 사용량은요? | ✅ 두 테이블이 존재하므로 일시적으로 2배로 증가할 수 있음 |

---

## 참고

- 대용량 마이그레이션 시 `ATTACH PARTITION`, `OPTIMIZE` 등을 조합하여 성능 개선 가능
- `ORDER BY`는 쿼리 성능 최적화의 핵심 요소이므로 처음 설계 시 신중히 고려할 것


# ClickHouse 인덱스 종류 정리

ClickHouse는 전통적인 RDBMS처럼 B-Tree 기반 인덱스를 사용하지 않습니다.  
대신, **block-level skipping index**를 통해 대용량 분석에 최적화된 인덱스 구조를 제공합니다.

---

## 인덱스 분류

| 분류                 | 종류                                                                 |
|----------------------|----------------------------------------------------------------------|
| 기본 인덱스          | `ORDER BY` 기반 인덱스 (`.mrk3` 파일)                                |
| Skip 인덱스 (`CREATE INDEX`) | `minmax`, `set`, `bloom_filter`, `ngrambf_v1`, `tokenbf_v1` 등 |

---

## 1. 기본 인덱스: `ORDER BY`

ClickHouse의 `MergeTree` 테이블은 반드시 `ORDER BY`를 포함하며,  
이 정렬 기준이 block skipping의 기본 인덱스로 사용됩니다.

예시:

```
CREATE TABLE logs (
  user_id UInt64,
  event_time DateTime
) ENGINE = MergeTree
ORDER BY (user_id, event_time);
```

- block 단위로 정렬되어 저장됨
- `.mrk3` 파일에 block 위치 정보 저장
- `user_id`, `event_time` 기반의 쿼리에서 빠른 성능 제공

---

## 2. Skip Index: `CREATE INDEX`로 명시적으로 생성

ClickHouse는 다음과 같은 block-level 인덱스를 생성할 수 있습니다.

---

### 2.1 `minmax`

- 각 block의 최소/최대 값을 기록
- WHERE 조건이 범위를 벗어나면 해당 block을 skip
- MergeTree에 자동 생성되지만 명시적으로 선언 가능

---

### 2.2 `set(N)`

- block마다 값 집합(Set)을 저장 (최대 N개)
- `IN (...)`, `=`, `!=` 조건에서 사용
- 값의 종류가 적고 반복되는 경우에 적합

```
CREATE INDEX status_idx ON logs (status_code)
TYPE set(1000) GRANULARITY 4;
```

---

### 2.3 `bloom_filter(p)`

- 존재 여부 확인용 확률 인덱스 (false positive 허용)
- 고유 값이 많은 컬럼에 적합
- `=`, `IN`, `!=` 등에서 효과적

```
CREATE INDEX user_idx ON logs (user_id)
TYPE bloom_filter(0.01) GRANULARITY 2;
```

---

### 2.4 `ngrambf_v1(n, p)`

- 문자열을 n-gram 단위로 잘라 bloom filter에 저장
- `LIKE '%xxx%'` 검색에 특화

```
CREATE INDEX msg_idx ON logs (message)
TYPE ngrambf_v1(3, 0.01) GRANULARITY 2;
```

---

### 2.5 `tokenbf_v1(p)`

- 문자열을 단어 단위로 분해 후 bloom filter에 저장
- 자연어, 문장형 텍스트 검색에 적합

```
CREATE INDEX token_idx ON logs (message)
TYPE tokenbf_v1(0.01) GRANULARITY 2;
```

---

## 인덱스 비교 표

| 인덱스 종류     | 사용 조건 예시            | 추천 컬럼 예시         |
|----------------|----------------------------|------------------------|
| `minmax`       | `WHERE age > 30`           | 숫자, 날짜             |
| `set`          | `status IN (...)`          | 상태 코드, Enum        |
| `bloom_filter` | `user_id = 123`            | 고유 ID                |
| `ngrambf_v1`   | `LIKE '%error%'`           | 로그 메시지            |
| `tokenbf_v1`   | `hasToken(message, 'fail')`| 자연어 문장             |

---

## 주의사항

- 인덱스는 모두 **block-level**에서 작동합니다 (row-level 아님).
- `CREATE INDEX`로 만든 인덱스는 `OPTIMIZE TABLE` 이후에 적용됩니다.
- 쿼리가 정렬 기준(`ORDER BY`)과 무관하면 효과가 제한될 수 있습니다.
- 인덱스를 생성하면 `.idx` 및 `.mrk3` 파일이 생성되어 디스크 사용량 증가

---

## 요약

| 항목                    | 내용                                              |
|-------------------------|---------------------------------------------------|
| 기본 인덱스             | `ORDER BY` 기준으로 정렬 + `.mrk3` 파일로 인덱싱 |
| 생성 가능한 인덱스       | `minmax`, `set`, `bloom_filter`, `ngrambf_v1`, `tokenbf_v1` |
| 동작 방식               | block 단위 skipping                              |
| 인덱스 생성 후 주의점    | `OPTIMIZE` 이후에만 전체 block에 적용됨           |

---

## 참고

ClickHouse는 전통적인 RDBMS의 인덱스 구조(B-Tree, Hash 등)와 다르며,  
**대량의 데이터 분석에서 불필요한 IO를 줄이기 위한 "읽기 최적화 인덱스"에 집중**되어 있습니다.

# ClickHouse가 지원하는 ENGINE 종류

ClickHouse는 테이블을 생성할 때 `ENGINE = ...` 구문을 통해  
데이터를 저장하거나 외부 시스템과 연결하는 방식을 지정합니다.  
각 엔진은 **저장 방식, 처리 목적, 성능 특성**이 다릅니다.

---

## ENGINE 분류 요약

| 유형               | 대표 엔진                             | 설명                             |
|--------------------|----------------------------------------|----------------------------------|
| 저장용             | `MergeTree`, `ReplacingMergeTree` 등   | 기본 분석/저장용 엔진           |
| 분산 처리용        | `Distributed`                         | 여러 노드에 분산된 테이블 처리 |
| 외부 연동용        | `Kafka`, `S3`, `MySQL`, `HDFS`, `JDBC` | 외부 데이터 소스 연결           |
| 메모리/임시 저장용 | `Memory`, `Null`, `File`               | 테스트, 비영속적 저장 등        |
| 뷰/자동 처리용     | `View`, `MaterializedView`             | 가상 테이블, 자동 insert 등     |

---

## 1. MergeTree 계열 (저장용)

ClickHouse의 핵심 저장 엔진이며, 대부분의 실전 테이블에서 사용됩니다.

### MergeTree

```
CREATE TABLE logs (
  user_id UInt64,
  event_time DateTime
)
ENGINE = MergeTree
ORDER BY (user_id, event_time);
```

- 컬럼 단위 압축 저장
- block skipping (`.mrk3`) 최적화
- 고속 집계 쿼리에 최적화

### ReplacingMergeTree

중복 데이터를 가장 최근 row로 대체 가능

```
ENGINE = ReplacingMergeTree
ORDER BY (user_id)
```

### SummingMergeTree

동일 키에 대해 SUM 자동 계산

```
ENGINE = SummingMergeTree
ORDER BY (user_id, date)
```

### AggregatingMergeTree

사전 집계 데이터 저장용 (State → Merge)

---

## 2. Distributed (분산 처리용)

여러 노드에 있는 테이블들을 하나처럼 조회할 수 있게 해줍니다.

```
CREATE TABLE global_logs AS logs
ENGINE = Distributed('my_cluster', 'mydb', 'logs', rand());
```

- 모든 노드에 동일한 구조의 테이블이 있어야 함
- 쿼리를 자동으로 shard에 분산 실행

---

## 3. 외부 연동용 엔진

### Kafka

```
CREATE TABLE kafka_input (
  message String
)
ENGINE = Kafka
SETTINGS kafka_broker_list = 'localhost:9092',
         kafka_topic_list = 'my_topic',
         kafka_format = 'JSONEachRow';
```

- Kafka topic에서 실시간으로 consume
- 반드시 Materialized View와 함께 사용

### MySQL

외부 MySQL 테이블을 조회 전용으로 연결

```
CREATE TABLE mysql_table
ENGINE = MySQL('host:3306', 'db', 'table', 'user', 'pass');
```

### S3

S3의 Parquet, CSV, JSON 파일을 읽기 전용 테이블로 연결

```
CREATE TABLE s3_data
ENGINE = S3('https://my-bucket.s3.amazonaws.com/data.json', 'JSONEachRow');
```

---

## 4. 메모리 및 임시용

### Memory

```
CREATE TABLE mem_table (
  id UInt32
) ENGINE = Memory;
```

- RAM에 저장됨 (재시작 시 사라짐)
- 테스트, 중간 결과용

### Null

```
CREATE TABLE null_table (
  x UInt32
) ENGINE = Null;
```

- 데이터를 받아도 저장하지 않음
- 출력 테스트, 파이프라인 디버깅에 사용

### File

```
CREATE TABLE csv_table
ENGINE = File(CSV, '/path/to/file.csv');
```

- 로컬 파일을 테이블처럼 읽기/쓰기

---

## 5. View / MaterializedView

### View

```
CREATE VIEW recent_logs AS
SELECT * FROM logs WHERE event_time > now() - INTERVAL 1 DAY;
```

- 쿼리 저장용 (virtual view)

### MaterializedView

```
CREATE MATERIALIZED VIEW kafka_to_logs
TO logs AS
SELECT * FROM kafka_input;
```

- source 테이블에 insert 시 자동 실행
- 자동 ETL 구성에 사용

---

## 요약

| 엔진           | 설명                                 | 사용 용도                 |
|----------------|--------------------------------------|----------------------------|
| MergeTree      | 기본 저장 엔진                       | 고속 쿼리, 분석용 테이블  |
| Distributed    | 여러 노드의 데이터를 통합 처리       | 분산 쿼리                  |
| Kafka          | Kafka에서 실시간 데이터 수신         | 실시간 ingestion           |
| Memory         | RAM 기반 저장                        | 테스트, 임시 계산용        |
| Null           | 입력을 무시                          | 파이프라인 디버깅          |
| File           | 파일 기반 테이블                     | CSV/JSON 파싱              |
| View           | 쿼리 저장용 가상 테이블              | 반복 쿼리 재사용           |
| MaterializedView| 자동 실행되는 트리거형 view         | 실시간 ETL 처리            |

---

## 참고

- 대부분의 ClickHouse 실무 테이블은 `MergeTree` 계열로 구축됩니다.
- Kafka 연동 시에는 반드시 Materialized View와 연결하여 실제 데이터 적재를 수행해야 합니다.
- `Distributed` 엔진은 클러스터 환경에서 필수이며, 클러스터 설정이 필요합니다.

# ClickHouse가 제공하는 API 종류

ClickHouse는 SQL 실행, 데이터 수집, 상태 조회, 외부 연동 등을 위해 다양한 API를 제공합니다.  
각 API는 목적과 사용 방식에 따라 구분되며, HTTP, TCP, ODBC, JDBC 등 여러 인터페이스를 제공합니다.

---

## API 요약 표

| API 종류             | 기본 포트 | 설명                                      |
|----------------------|------------|-------------------------------------------|
| HTTP API             | 8123       | REST 방식 SQL 실행, INSERT/SELECT          |
| Native TCP API       | 9000       | 고성능 바이너리 프로토콜 (CLI, 드라이버)   |
| HTTPS API            | 8443       | HTTP + TLS                                 |
| TCP over TLS         | 9440       | Native 프로토콜의 암호화 버전              |
| ODBC / JDBC          | -          | 외부 도구 및 언어 연동                     |
| Prometheus API       | 9363 (옵션)| `/metrics` 엔드포인트로 상태 제공          |
| System Tables        | -          | SQL로 내부 상태/메타데이터 조회 가능       |

---

## 1. HTTP API

ClickHouse는 REST 스타일의 HTTP 인터페이스를 제공합니다.

### 특징

- GET / POST 지원
- SQL 쿼리 실행
- 인증 (Basic Auth)
- 다양한 FORMAT 지원 (JSON, CSV 등)

### 예시

```
curl -u default:password 'http://localhost:8123/?query=SELECT+1'
```

데이터 삽입 예:

```
curl -X POST 'http://localhost:8123/?query=INSERT INTO logs FORMAT JSONEachRow' \
  --data-binary '{"event":"click", "user_id":42}'
```

---

## 2. Native TCP API

ClickHouse 고유의 바이너리 프로토콜입니다.

- 매우 빠름
- CLI 또는 드라이버(`clickhouse-client`, `clickhouse-jdbc`, `clickhouse-go`)에서 사용

### 예시

```
clickhouse-client --host=localhost --port=9000
```

---

## 3. HTTPS API / TCP over TLS

보안 연결이 필요한 경우 TLS로 암호화된 포트를 사용합니다.

- HTTPS: `port 8443` (HTTP API의 TLS 버전)
- TCP over TLS: `port 9440`

설정 예시 (`config.xml`에서 TLS 인증서 지정)

---

## 4. ODBC / JDBC

ClickHouse는 다양한 클라이언트 도구와 언어에서 사용할 수 있도록  
ODBC, JDBC 드라이버를 제공합니다.

### JDBC 예시

```
jdbc:clickhouse://localhost:8123/default
```

- Java 기반 앱에서 SQL 실행 가능

### ODBC

- Tableau, Excel, Power BI 연동
- Windows/Mac에 드라이버 설치 후 DSN 설정

---

## 5. Prometheus API

서버 내부 상태를 Prometheus 형식으로 수집할 수 있습니다.

### 엔드포인트

```
GET http://localhost:9363/metrics
```

노출되는 정보:

- 쿼리 수
- 실패율
- 디스크 I/O
- CPU 사용량 등

---

## 6. System Tables

SQL로 접근 가능한 메타데이터 및 내부 상태 정보 테이블입니다.

| 테이블 이름          | 설명                         |
|----------------------|------------------------------|
| `system.tables`      | DB 내 모든 테이블 목록       |
| `system.columns`     | 테이블의 컬럼 정보           |
| `system.parts`       | MergeTree 파티션 정보        |
| `system.query_log`   | 쿼리 실행 이력               |
| `system.processes`   | 현재 실행 중인 쿼리 목록     |
| `system.metrics`     | 내부 메트릭 수치             |

### 예시

```
SELECT *
FROM system.query_log
ORDER BY event_time DESC
LIMIT 10;
```

---

## 요약

| 목적                        | 추천 API                |
|-----------------------------|--------------------------|
| RESTful SQL 실행            | HTTP API (`8123`)        |
| 고성능 드라이버, CLI 사용   | Native TCP API (`9000`)  |
| 보안 연결                   | HTTPS (`8443`), TLS TCP (`9440`) |
| BI 도구 연동                | ODBC / JDBC              |
| 모니터링                    | Prometheus `/metrics`    |
| 내부 상태 분석              | System Tables            |

---

ClickHouse는 경량화된 아키텍처이지만,  
분석, 대시보드, 실시간 연동 등 다양한 목적에 대응할 수 있는 API를 제공합니다.
