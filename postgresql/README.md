- [Materials](#materials)
- [Install](#install)
- [Basic](#basic)
  - [vaccuum](#vaccuum)
- [Level 0: Sky Zone](#level-0-sky-zone)
  - [Data Types](#data-types)
  - [CREATE TABLE](#create-table)
  - [SELECT, INSERT, UPDATE, DELETE](#select-insert-update-delete)
  - [ORDERY BY](#ordery-by)
  - [LIMIT and OFFSET](#limit-and-offset)
  - [GROUP BY](#group-by)
  - [NULL](#null)
  - [Indexes](#indexes)
  - [Join](#join)
  - [Forein Keys](#forein-keys)
  - [ORMs](#orms)
- [Level 1: Surface Zone](#level-1-surface-zone)
  - [Transactions](#transactions)
  - [ACID](#acid)
  - [Query plans and EXPLAIN](#query-plans-and-explain)
  - [Inverted Indexes](#inverted-indexes)
  - [Keyset Pagination](#keyset-pagination)
  - [Computed Columns](#computed-columns)
  - [Stored Columns](#stored-columns)
  - [ORDER BY Aggregates](#order-by-aggregates)
  - [Window Functions](#window-functions)
  - [Outer Joins](#outer-joins)
  - [CTEs](#ctes)
  - [Normal Forms](#normal-forms)
- [Level 2: Sunlight Zone](#level-2-sunlight-zone)
  - [Connection Pools](#connection-pools)
  - [The DUAL Table](#the-dual-table)
  - [Laterial Joins](#laterial-joins)
  - [Recursive CTEs](#recursive-ctes)
  - [ORMs Create Bad Queries](#orms-create-bad-queries)
  - [Stored Procedures](#stored-procedures)
  - [Cursors](#cursors)
  - [There are no non-nullable types](#there-are-no-non-nullable-types)
  - [Optimizers don't work without table statistics](#optimizers-dont-work-without-table-statistics)
  - [Plan hints](#plan-hints)
  - [MVCC Garbage Collection](#mvcc-garbage-collection)
- [Twilight Zone](#twilight-zone)
  - [COUNT(\*) vs COUNT(1)](#count-vs-count1)
  - [Isolation Levels and Phantom Reads](#isolation-levels-and-phantom-reads)
  - [Write skew](#write-skew)
  - [Serializable restarts require retry loops on all statements](#serializable-restarts-require-retry-loops-on-all-statements)
  - [Partial Indexes](#partial-indexes)
  - [Generator functions zip when cross joined](#generator-functions-zip-when-cross-joined)
  - [Sharding](#sharding)
  - [ZigZag Join](#zigzag-join)
  - [MERGE](#merge)
  - [Triggers](#triggers)
  - [Grouping sets, Cube, Rollup](#grouping-sets-cube-rollup)
- [Level 4: Midnight Zone](#level-4-midnight-zone)
  - [Denormalization](#denormalization)
  - [NULLs in CHECK constraints are truthy](#nulls-in-check-constraints-are-truthy)
  - [Transaction Contention](#transaction-contention)
  - [SELECT FOR UPDATE](#select-for-update)
  - [Star Schemas](#star-schemas)
  - [Sargability](#sargability)
  - [Ascending Key Problem](#ascending-key-problem)
  - [Ambiguous Network Errors](#ambiguous-network-errors)
  - [utf8mb4](#utf8mb4)
- [Level 5: Abyssal Zone](#level-5-abyssal-zone)
  - [Cost models don't reflect reality](#cost-models-dont-reflect-reality)
  - [null::jsonb IS NULL = false](#nulljsonb-is-null--false)
  - [TPCC requires wait times](#tpcc-requires-wait-times)
  - [DEFERRABLE INITIALLY IMMEDIATE](#deferrable-initially-immediate)
  - [EXPLAIN approximates SELECT COUNT(\*)](#explain-approximates-select-count)
  - [MATCH PARTIAL Foreign Keys](#match-partial-foreign-keys)
  - [Causal Reverse](#causal-reverse)
- [Level 6: Hadal Zone](#level-6-hadal-zone)
  - [Vectorized doesn't mean SIMD](#vectorized-doesnt-mean-simd)
  - [NULLs are equal in DISTINCT but unequal in UNIQUE](#nulls-are-equal-in-distinct-but-unequal-in-unique)
  - [Volcano Model](#volcano-model)
  - [Join ordering is NP Hard](#join-ordering-is-np-hard)
  - [Database Cracking](#database-cracking)
  - [WCOJ](#wcoj)
  - [Learned Indexes](#learned-indexes)
  - [TXID Exhaustion](#txid-exhaustion)
- [Level 7: Pitch Black Zone](#level-7-pitch-black-zone)
  - [The halloween problem](#the-halloween-problem)
  - [Dee and Dum](#dee-and-dum)
  - [SERIAL is non-transactional](#serial-is-non-transactional)
  - [allballs](#allballs)
  - [fsyncgate](#fsyncgate)
  - [Every SQL operator is actually a join](#every-sql-operator-is-actually-a-join)

----

# Materials

* [Explaining The Postgres Meme](https://www.avestura.dev/blog/explaining-the-postgres-meme)
* [crunchdata](https://www.crunchydata.com/developers/tutorials)
  * browser 에서 PostgreSQL 을 띄우고 공부할 수 있다.
* [The Art of PostgreSQL](https://theartofpostgresql.com/)
  * mater piece

# Install

* [[Docker] Docker PostgreSQL 설치 및 실행 | tistory](https://kanoos-stu.tistory.com/23)

----

```console
$ docker run --rm -p 5432:5432 -e POSTGRES_PASSWORD=1 -e POSTGRES_USER=iamslash -e POSTGRES_DB=basicdb --name my-postgres -d postgres

$ docker exec -it my-postgres bash

$ psql -U iamslash basicdb
# Show databases
\list 
\dt
SELECT * FROM account;
```

# Basic

## vaccuum

* [PostgreSQL: 베큠(VACUUM)을 실행해야되는 이유 그리고 성능 향상](https://blog.gaerae.com/2015/09/postgresql-vacuum-fsm.html)

----

FSM (Free Space Map) 에 쌓여진 데이터를 지우는 것. 디스크 조각모으기와 비슷하다.

# Level 0: Sky Zone

## Data Types

> [Chapter 8. Data Types | postgresql](https://www.postgresql.org/docs/current/datatype.html)

| Category          | Name                          | Aliases         | Description                                      |
|-------------------|-------------------------------|-----------------|--------------------------------------------------|
| Numeric           | bigint                        | int8            | signed eight-byte integer                        |
| Numeric           | bigserial                     | serial8         | autoincrementing eight-byte integer              |
| Bit String        | bit [ (n) ]                   |                 | fixed-length bit string                          |
| Bit String        | bit varying [ (n) ]           | varbit [ (n) ]  | variable-length bit string                       |
| Boolean           | boolean                       | bool            | logical Boolean (true/false)                     |
| Geometric         | box                           |                 | rectangular box on a plane                       |
| Binary Data       | bytea                         |                 | binary data (“byte array”)                       |
| Character         | character [ (n) ]             | char [ (n) ]    | fixed-length character string                    |
| Character         | character varying [ (n) ]     | varchar [ (n) ] | variable-length character string                 |
| Network Address   | cidr                          |                 | IPv4 or IPv6 network address                     |
| Geometric         | circle                        |                 | circle on a plane                                |
| Date and Time     | date                          |                 | calendar date (year, month, day)                 |
| Numeric           | double precision              | float8          | double precision floating-point number (8 bytes) |
| Network Address   | inet                          |                 | IPv4 or IPv6 host address                        |
| Numeric           | integer                       | int, int4       | signed four-byte integer                         |
| Date and Time     | interval [ fields ] [ (p) ]   |                 | time span                                        |
| JSON Data Types   | json                          |                 | textual JSON data                                |
| JSON Data Types   | jsonb                         |                 | binary JSON data, decomposed                     |
| Geometric         | line                          |                 | infinite line on a plane                         |
| Geometric         | lseg                          |                 | line segment on a plane                          |
| Network Address   | macaddr                       |                 | MAC (Media Access Control) address               |
| Network Address   | macaddr8                      |                 | MAC (Media Access Control) address (EUI-64 format) |
| Monetary          | money                         |                 | currency amount                                  |
| Numeric           | numeric [ (p, s) ]            | decimal [ (p, s) ] | exact numeric of selectable precision       |
| Geometric         | path                          |                 | geometric path on a plane                         |
| System Data       | pg_lsn                        |                 | PostgreSQL Log Sequence Number                   |
| System Data       | pg_snapshot                   |                 | user-level transaction ID snapshot               |
| Geometric         | point                         |                 | geometric point on a plane                       |
| Geometric         | polygon                       |                 | closed geometric path on a plane                 |
| Numeric           | real                          | float4          | single precision floating-point number (4 bytes) |
| Numeric           | smallint                      | int2            | signed two-byte integer                          |
| Numeric           | smallserial                   | serial2         | autoincrementing two-byte integer                |
| Numeric           | serial                        | serial4         | autoincrementing four-byte integer               |
| Character         | text                          |                 | variable-length character string                 |
| Date and Time     | time [ (p) ] [ without time zone ] |           | time of day (no time zone)                       |
| Date and Time     | time [ (p) ] with time zone   | timetz          | time of day, including time zone                 |
| Date and Time     | timestamp [ (p) ] [ without time zone ] |       | date and time (no time zone)                     |
| Date and Time     | timestamp [ (p) ] with time zone | timestamptz  | date and time, including time zone               |
| Text Search       | tsquery                       |                 | text search query                                |
| Text Search       | tsvector                      |                 | text search document                             |
| System Data       | txid_snapshot                 |                 | user-level transaction ID snapshot (deprecated; see pg_snapshot) |
| UUID              | uuid                          |                 | universally unique identifier                    |
| XML Data          | xml                           |                 | XML data                                         |

## CREATE TABLE

```sql
CREATE TABLE "audit_log" (
	id serial primary key,
	ip inet,
	action text,
	actor text,
	description text,
	created_at timestamp default NOW()
)
```

## SELECT, INSERT, UPDATE, DELETE

```sql
-- Select
SELECT action, actor, description
  FROM "audit_log"
 WHERE ip = '127.0.0.1';
-- Insert
INSERT INTO "audit_log" (ip, action, actor, description) VALUES (
    '127.0.0.1',
    'delete user',
    'admin',
    'admin deleted the user x'
);
-- Update
UPDATE "audit_log"
   SET ip = '192.168.1.1'
 WHERE id = 1;
-- Delete
DELETE FROM "audit_log"
 WHERE id = 1;
```

## ORDERY BY

```sql
-- Order by
SELECT *
FROM "audit_log"
ORDER BY created_at DESC;
-- K-neareset-neighbor Ordering
SELECT "name", "location", "country"
FROM "circuits"
ORDER BY POINT(lng, lat) <-> POINT(2.349014, 48.864716)
LIMIT 10;
```

## LIMIT and OFFSET

[Pagination](/pagination/README.md) for efficient pagination.

```sql
SELECT *
  FROM "audit_log"
OFFSET 100
 LIMIT 10;
```

## GROUP BY

```sql
CREATE TABLE student (
    id SERIAL PRIMARY KEY,
    class_no INTEGER,
    grade INTEGER
);
-- Average grade of each class
SELECT class_no, AVG(grade) AS class_avg
  FROM student
 GROUP BY class_no;
```

## NULL

`NULL` means `undefined value`, or `simply not knowing the value`. That is why `true = NULL, false = NULL, and NULL = NULL` checks all result in a `NULL`.

```sql
SELECT
    TRUE = NULL AS a,
    FALSE = NULL AS b,
    NULL = NULL AS c;
-- result= a: NULL, b: NULL, c: NULL
```

## Indexes

## Join

## Forein Keys

## ORMs

# Level 1: Surface Zone

## Transactions

## ACID

## Query plans and EXPLAIN

## Inverted Indexes

## Keyset Pagination

## Computed Columns

## Stored Columns

## ORDER BY Aggregates

## Window Functions

## Outer Joins

## CTEs

## Normal Forms

# Level 2: Sunlight Zone

## Connection Pools

## The DUAL Table

## Laterial Joins

## Recursive CTEs

## ORMs Create Bad Queries

## Stored Procedures

## Cursors

## There are no non-nullable types

## Optimizers don't work without table statistics

## Plan hints

## MVCC Garbage Collection

# Twilight Zone

## COUNT(*) vs COUNT(1)

## Isolation Levels and Phantom Reads

## Write skew

## Serializable restarts require retry loops on all statements

## Partial Indexes

## Generator functions zip when cross joined

## Sharding

## ZigZag Join

## MERGE

## Triggers

## Grouping sets, Cube, Rollup

# Level 4: Midnight Zone

## Denormalization

## NULLs in CHECK constraints are truthy

## Transaction Contention

## SELECT FOR UPDATE

## Star Schemas

## Sargability

## Ascending Key Problem

## Ambiguous Network Errors

## utf8mb4

# Level 5: Abyssal Zone

## Cost models don't reflect reality

## null::jsonb IS NULL = false

## TPCC requires wait times

## DEFERRABLE INITIALLY IMMEDIATE

## EXPLAIN approximates SELECT COUNT(*)

## MATCH PARTIAL Foreign Keys

## Causal Reverse

# Level 6: Hadal Zone

## Vectorized doesn't mean SIMD

## NULLs are equal in DISTINCT but unequal in UNIQUE

## Volcano Model

## Join ordering is NP Hard

## Database Cracking

## WCOJ

## Learned Indexes

## TXID Exhaustion

# Level 7: Pitch Black Zone

## The halloween problem

## Dee and Dum

## SERIAL is non-transactional

## allballs

## fsyncgate

## Every SQL operator is actually a join
