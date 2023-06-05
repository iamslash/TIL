- [Materials](#materials)
- [Data Manipulation](#data-manipulation)
  - [Prevent Lock Contention For Updates On Hot Rows](#prevent-lock-contention-for-updates-on-hot-rows)
  - [Updates Based On A Select Query](#updates-based-on-a-select-query)
  - [Return The Values Of Modified Rows	(PostgreSQL)](#return-the-values-of-modified-rowspostgresql)
  - [Delete Duplicate Rows](#delete-duplicate-rows)
  - [Table Maintenance After Bulk Modifications](#table-maintenance-after-bulk-modifications)
- [Querying Data](#querying-data)
  - [Reduce The Amount Of Group By Columns](#reduce-the-amount-of-group-by-columns)
  - [Fill Tables With Large Amounts Of Test Data](#fill-tables-with-large-amounts-of-test-data)
  - [Simplified Inequality Checks With Nullable Columns](#simplified-inequality-checks-with-nullable-columns)
  - [Prevent Division By Zero Errors](#prevent-division-by-zero-errors)
  - [Sorting Order With Nullable Columns](#sorting-order-with-nullable-columns)
  - [Deterministic Ordering for Pagination](#deterministic-ordering-for-pagination)
  - [More Efficient Pagination Than LIMIT OFFSET](#more-efficient-pagination-than-limit-offset)
  - [Database-Backed Locks With Safety Guarantees](#database-backed-locks-with-safety-guarantees)
  - [Refinement Of Data With Common Table Expressions](#refinement-of-data-with-common-table-expressions)
  - [First Row Of Many Similar Ones	(PostgreSQL)](#first-row-of-many-similar-onespostgresql)
  - [Multiple Aggregates In One Query](#multiple-aggregates-in-one-query)
  - [Limit Rows Also Including Ties	(PostgreSQL)](#limit-rows-also-including-tiespostgresql)
  - [Fast Row Count Estimates](#fast-row-count-estimates)
  - [Date-Based Statistical Queries With Gap-Filling](#date-based-statistical-queries-with-gap-filling)
  - [Table Joins With A For-Each Loop](#table-joins-with-a-for-each-loop)
- [Schema](#schema)
  - [Rows Without Overlapping Dates	(PostgreSQL)](#rows-without-overlapping-datespostgresql)
  - [Store Trees As Materialized Paths](#store-trees-as-materialized-paths)
  - [JSON Columns to Combine NoSQL and Relational Databases](#json-columns-to-combine-nosql-and-relational-databases)
  - [Alternative Tag Storage With JSON Arrays](#alternative-tag-storage-with-json-arrays)
  - [Constraints for Improved Data Strictness](#constraints-for-improved-data-strictness)
  - [Validation Of JSON Colums Against A Schema](#validation-of-json-colums-against-a-schema)
  - [UUID Keys Against Enumeration Attacks](#uuid-keys-against-enumeration-attacks)
  - [Fast Delete Of Big Data With Partitions](#fast-delete-of-big-data-with-partitions)
  - [Pre-Sorted Tables For Faster Access](#pre-sorted-tables-for-faster-access)
  - [Pre-Aggregation of Values for Faster Queres](#pre-aggregation-of-values-for-faster-queres)
- [Indexes](#indexes)
  - [Indexes On Functions And Expressions](#indexes-on-functions-and-expressions)
  - [Find Unused Indexes](#find-unused-indexes)
  - [Safely Deleting Unused Indexes	(MySQL)](#safely-deleting-unused-indexesmysql)
  - [Index-Only Operations By Including More Columns](#index-only-operations-by-including-more-columns)
  - [Partial Indexes To Reduce Index Size	(PostgreSQL)](#partial-indexes-to-reduce-index-sizepostgresql)
  - [Partial Indexes For Uniqueness Constraints](#partial-indexes-for-uniqueness-constraints)
  - [Index Support For Wildcard Searches (PostgreSQL)](#index-support-for-wildcard-searches-postgresql)
  - [Rules For Multi-Column Indexes](#rules-for-multi-column-indexes)
  - [Hash Indexes To Descrease Index Size	(PostgreSQL)](#hash-indexes-to-descrease-index-sizepostgresql)
  - [Descending Indexes For Order By](#descending-indexes-for-order-by)
  - [Ghost Conditions Against Unindexed Columns](#ghost-conditions-against-unindexed-columns)

----

# Materials

- [The Database Cookbook For Developers](https://sqlfordevs.com/ebook)

# Data Manipulation	

## Prevent Lock Contention For Updates On Hot Rows	

하나의 record 에 모두 저장하는 것보다는 여러 record 에 나누어서 저장하자.
동시성을 해결할 수 있다. 다음은 특정한 `tweet_id` 의 좋아요 개수를 수정하는
예제이다.  

```sql
-- MySQL
CREATE TABLE tweet_statistics (
  tweet_id BIGINT,
  fanout INT,
  likes_count INT,
  PRIMARY KEY (tweet_id, fanout)
);

INSERT INTO tweet_statistics (
    tweet_id, fanout, likes_count) 
     VALUES (
    1475870220422107137, FLOOR(RAND() * 10), 1) 
ON DUPLICATE KEY UPDATE likes_count = likes_count + VALUES(likes_count);

  SELECT tweet_id, SUM(likes_count)
    FROM tweet_statistics
GROUP BY tweet_id;

-- PostgreSQL
INSERT INTO tweet_statistics (
  tweet_id, fanout, likes_count
) VALUES (
  1475870220422107137, FLOOR(RANDOM() * 10), 1
) ON CONFLICT (tweet_id, fanout) DO UPDATE SET likes_count =
tweet_statistics.likes_count + excluded.likes_count;
```

MySQL 의 `VALUES(likes_count)` 는 `INSERT INTO` 로 제공된 `likes_count` 를 말한다. 그러나 MySQL
8.0.20 이후로 deprecate 되었다. [Insert On Duplicate Key Update | TIL](/sql/README.md#insert-on-duplicate-key-update) 참고.

그런데 이렇게 SQL 에 `RAND()` 을 사용하게 되면 Global Lock 이 적용된다.
비효율적이다. Application 에서 random number 를 만들어 주입하자.

like 를 한 사람이 한번만 하도록 제한해 보자. 어떻게 구현하면 좋을까?
[HyperLogLogs](/redis/README.md#hyperl­oglogs) 가 유용하다.

## Updates Based On A Select Query	

`Update, Join` 을 이용한다면 한번의 sql 로 다른 table 의 record 를
참고하여 update 를 할 수 있다. 

```sql
-- MySQL
UPDATE products
  JOIN categories USING(category_id)
   SET price = price_base - price_base * categories.discount;

-- PostgreSQL
UPDATE products
   SET price = price_base - price_base * categories.discount
  FROM categories
 WHERE products.category_id = categories.category_id;
```

## Return The Values Of Modified Rows	(PostgreSQL)

`Delete` 후 record 의 일부를 읽을 수 있다.

```sql
-- PostgreSQL:
   DELETE FROM sessions
    WHERE ip = '127.0.0.1'
RETURNING id, user_agent, last_access;
```

## Delete Duplicate Rows	

**CTE (Common Table Expression)** 을 이용하면 단 하나의 sql
로 중복된 record 를 삭제할 수 있다.

```sql
-- MySQL
WITH duplicates AS (
      SELECT id, 
             ROW_NUMBER() OVER(
               PARTITION BY firstname, lastname, email
               ORDER BY age DESC) AS rownum
        FROM contacts)
DELETE contacts
  FROM contacts
  JOIN duplicates USING(id)
 WHERE duplicates.rownum > 1;

-- PostgreSQL
WITH duplicates AS (
      SELECT id, 
             ROW_NUMBER() OVER(
               PARTITION BY firstname, lastname, email
               ORDER BY age DESC) AS rownum
       FROM contacts)
DELETE FROM contacts
 USING duplicates
 WHERE contacts.id = duplicates.id AND 
       duplicates.rownum > 1;
```

## Table Maintenance After Bulk Modifications	

table 에 많은 양의 record 가 추가, 수정, 삭제 된다면 table 통계는 매번 계산되지
않는다. 추가, 수정, 삭제된 record 의 수가 threshold 를 넘기면 table 통계가 다시
게산된다. 

`ANALYZE` 를 이용하면 table 통계를 강제로 계산하도록 한다.

```sql
-- MySQL
ANALYZE TABLE users;
+-------------+---------+----------+----------+
| Table       | Op      | Msg_type | Msg_text |
+-------------+---------+----------+----------+
| hello.users | analyze | status   | OK       |
+-------------+---------+----------+----------+

-- PostgreSQL
ANALYZE SKIP_LOCKED users;
```

# Querying Data	

## Reduce The Amount Of Group By Columns	

Primary Key 를 Grouping 한다면 `GROUP BY` 에 Primary Key 만 적어주면 된다.
`GROUP BY actors.firstname, actors.lastname` 할 필요가 없다. SQL 이 간단해 진다. 

```sql
  SELECT actors.firstname, 
         actors.lastname, 
         COUNT(*) as count
    FROM actors
    JOIN actors_movies USING(actor_id)
GROUP BY actors.id
```

## Fill Tables With Large Amounts Of Test Data	

[MySQL](/mysql/README.md) test data 는 `WITH RECURSIVE` 를 이용하여 간단히
생성할 수 있다.

[PostgresSQL](/postgresql/README.md) test data 는 `generate_series` 를 이용하여
간단히 생성할 수 있다.

```sql
-- MySQL
   SET cte_max_recursion_depth = 4294967295;
INSERT INTO contacts (firstname, lastname)
  WITH RECURSIVE counter(n) AS(
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM counter WHERE n < 100000)
SELECT CONCAT('firstname-', counter.n), 
       CONCAT('lastname-', counter.n)
  FROM counter

-- PostgreSQL
INSERT INTO contacts (firstname, lastname)
SELECT CONCAT('firstname-', i), 
       CONCAT('lastname-', i)
  FROM generate_series(1, 100000) as i;
```

## Simplified Inequality Checks With Nullable Columns	

`col != 'value'` 는 `NULL` value record 를 가져오지 못한다. `NULL` value record
를 포함해서 가져오려면 `(col IS NULL OR col != 'value')` 를 사용해야 한다.

다음의 방법을 사용하면 간단한 SQL 로 `NULL` value record 를 포함해서 가져올 수
있다.

```sql
-- MySQL
SELECT * 
  FROM example 
 WHERE NOT(column <> 'value');

-- PostgreSQL
SELECT * 
  FROM example 
 WHERE column IS DISTINCT FROM 'value';
```

## Prevent Division By Zero Errors	

`NULLIF()` 를 사용하여 `Division By Zero Errors` 를 피하자.

아래는 `visitors_yesterday == 0` 이면 `NULL` 을 리턴한다. `visitors_yesterday != 0`
이면 `visitors_yesterday` 를 리턴한다.

```sql
SELECT visitors_today / NULLIF(visitors_yesterday, 0)
  FROM logs_aggregated;
```

`NULLIF(expr1, expr2)` 는 `expr1 == expr2` 일 때 `NULL` 를 리턴하고 `expr1 !=
NULL` 일 때 `expr1` 을 리턴한다.

참고로 임의의 숫자를 `NULL` 로 나누면 그 결과는 `NULL` 이다.

```sql
SELECT 3 / NULL;
+----------+
| 3 / NULL |
+----------+
|     NULL |
+----------+
```

## Sorting Order With Nullable Columns	

record 를 정렬할 때 다음과 같은 방법으로 `NULL` value record 의 위치를 조정할 수
있다.

```sql
-- MySQL: NULL values placed first (default)
SELECT * FROM customers ORDER BY country ASC;
SELECT * FROM customers ORDER BY country IS NOT NULL, country ASC;
-- MYSQL: NULL values placed last
SELECT * FROM customers ORDER BY country IS NULL, country ASC;

-- PostgreSQL: NULL values placed first
SELECT * FROM customers ORDER BY country ASC NULLS FIRST;
-- PostgreSQL_ NULL values placed last (default)
SELECT * FROM customers ORDER BY country ASC;
SELECT * FROM customers ORDER BY country ASC NULLS LAST;
```

## Deterministic Ordering for Pagination	

[pagination](/pagination/README.md) 결과가 동일하기 위해서는 `ORDER BY` 에 최대한 많은 column 을 적어야
한다. 그렇지 않으면 [pagination](/pagination/README.md) 결과가 SQL 을 요청할 때 마다 달라질 수 있다.

```sql
  SELECT *
    FROM users
ORDER BY firstname ASC, lastname ASC, user_id ASC
   LIMIT 20 OFFSET 60;
```

## More Efficient Pagination Than LIMIT OFFSET	

아래의 방법은 [Deterministic Ordering for Pagination](#deterministic-ordering-for-pagination) 보다 빠르다. `WHERE` 때문에?

```sql
-- MySQL, PostgreSQL
  SELECT *
    FROM users
   WHERE (firstname, lastname, id) > ('John', 'Doe', 3150)
ORDER BY firstname ASC, lastname ASC, user_id ASC
   LIMIT 30
```

## Database-Backed Locks With Safety Guarantees	

`SELECT ... FOR UPDATE` 를 사용하면 특정 SQL 에 대해 Serializable
Isolation Level 을 사용할 수 있다. [Practice of "SELECT ... FOR UPDATE"](#practice-of-select--for-update) 를 참고하자.

```sql
START TRANSACTION;

SELECT balance 
  FROM account 
 WHERE account_id = 7 FOR UPDATE;
-- Race condition free update after processing the data
UPDATE account 
   SET balance = 540 
 WHERE account_id = 7;

COMMIT;
```

## Refinement Of Data With Common Table Expressions	

[CTE (Common Table Expression)](/sql/README.md#with-as) 을 사용하면 여러 Table 을 Single SQL 에
질의할 수 있다. 그러나 OLAP 로 사용하자.

```sql
WITH most_popular_products AS (
  SELECT products.*, COUNT(*) as sales
    FROM products
    JOIN users_orders_products USING(product_id)
    JOIN users_orders USING(order_id)
   WHERE users_orders.created_at BETWEEN '2022-01-01' AND '2022-06-30'
GROUP BY products.product_id
ORDER BY COUNT(*) DESC
   LIMIT 10
), applicable_users (
  SELECT DISTINCT users.*
    FROM users
    JOIN users_raffle USING(user_id)
   WHERE users_raffle.correct_answers > 8
), applicable_users_bought_most_popular_product AS (
  SELECT applicable_users.user_id, most_popular_products.product_id
    FROM applicable_users
    JOIN users_orders USING(order_id)
    JOIN users_orders_products USING(product_id)
    JOIN most_popular_products USING(product_id)
) raffle AS (
  SELECT product_id, 
         user_id, 
         RANK() OVER(
           PARTITION BY product_id
           ORDER BY RANDOM()) AS winner_order
    FROM applicable_users_bought_most_popular_product
)
SELECT product_id, user_id 
  FROM raffle 
 WHERE winner_order = 1;
```

## First Row Of Many Similar Ones	(PostgreSQL)

`DISTINCT ON` 을 이용하면 단 하나의 row 를 가져올 수 있다???

```sql
-- PostgreSQL
  SELECT DISTINCT ON (customer_id) *
    FROM orders
   WHERE EXTRACT (YEAR FROM created_at) = 2022
ORDER BY customer_id ASC, price DESC;
```

## Multiple Aggregates In One Query	

단 하나의 SQL 로 많은 Aggregation 을 할 수 있다.

```sql
-- MySQL
SELECT SUM(released_at = 2001) AS released_2001,
       SUM(released_at = 2002) AS released_2002,
       SUM(director = 'Steven Spielberg') AS director_stevenspielberg,
       SUM(director = 'James Cameron') AS director_jamescameron
  FROM movies
 WHERE streamingservice = 'Netflix';
-- PostgreSQL
SELECT COUNT(*) FILTER (WHERE released_at = 2001) AS released_2001,
       COUNT(*) FILTER (WHERE released_at = 2002) AS released_2002,
       COUNT(*) FILTER (WHERE director = 'Steven Spielberg') AS 
         director_stevenspielberg,
       COUNT(*) FILTER (WHERE director = 'James Cameron') AS
         director_jamescameron
  FROM movies
 WHERE streamingservice = 'Netflix';
```

## Limit Rows Also Including Ties	(PostgreSQL)

일등 3 명의 record 를 가져오고 싶다. 아래와 같은 방법으로 값이 같더라도 3 개에
포함해서 가져온다.

```sql
-- PostgreSQL
  SELECT *
    FROM teams
ORDER BY winning_games DESC
   FETCH FIRST 3 ROWS WITH TIES;
```

## Fast Row Count Estimates	

record 의 개수가 많으면 빨리 그 개수를 가져올 수 없다. 아래와 같은 방법으로 대충
몇개인지 빨리 가져온다.

```sql
-- MySQL
EXPLAIN FORMAT=TREE 
 SELECT * 
   FROM movies 
  WHERE rating = 'NC-17' AND
        price < 4.99;

explain format=tree select * from games;
+---------------------------------------------+
| EXPLAIN                                     |
+---------------------------------------------+
| -> Table scan on games  (cost=0.55 rows=3)  |
+---------------------------------------------+

-- PostgreSQL
EXPLAIN 
 SELECT * 
   FROM movies 
  WHERE rating = 'NC-17' AND 
  price < 4.99;
```

## Date-Based Statistical Queries With Gap-Filling	

아래와 같은 방법으로 중간에 빠져있는 날짜를 채우자.

```sql
-- MySQL
SET cte_max_recursion_depth = 4294967295;
WITH RECURSIVE dates_without_gaps(day) AS (
     SELECT DATE_SUB(CURRENT_DATE, INTERVAL 14 DAY) as day
  UNION ALL
     SELECT DATE_ADD(day, INTERVAL 1 DAY) as day
       FROM dates_without_gaps
      WHERE day < CURRENT_DATE
)
   SELECT dates_without_gaps.day, 
          COALESCE(SUM(statistics.count), 0)
     FROM dates_without_gaps
LEFT JOIN statistics ON(statistics.day = dates_without_gaps.day)
 GROUP BY dates_without_gaps.day;

-- PostgreSQL
   SELECT dates_without_gaps.day, 
          COALESCE(SUM(statistics.count), 0)
     FROM generate_series(CURRENT_DATE - INTERVAL '14 days', CURRENT_DATE, '1 day') as 
          dates_without_gaps(day)
LEFT JOIN statistics ON(statistics.day = dates_without_gaps.day)
 GROUP BY dates_without_gaps.day;
```

## Table Joins With A For-Each Loop	

`JOIN LATERAL` 를 사용하여 `For-Each Loop` 를 실행할 수 있다.

```sql
-- MySQL, PostgreSQL
   SELECT customers.*, 
          recent_sales.*
     FROM customers
LEFT JOIN LATERAL (
     SELECT *
       FROM sales
      WHERE sales.customer_id = customers.id
   ORDER BY created_at DESC
      LIMIT 3
) AS recent_sales ON true;
```

# Schema	

## Rows Without Overlapping Dates	(PostgreSQL)

다음과 같은 방법으로 중복 예약 방지하자???

```sql
CREATE TABLE bookings (
  room_number int,
  reservation tstzrange,
  EXCLUDE USING gist (room_number WITH =, reservation WITH &&)
);
INSERT INTO meeting_rooms (
  room_number, reservation
) VALUES (
  5, '[2022-08-20 16:00:00+00,2022-08-20 17:30:00+00]',
  5, '[2022-08-20 17:30:00+00,2022-08-20 19:00:00+00]',
);
```

## Store Trees As Materialized Paths	

다음과 같은 방법으로 Tree 를 Table 로 표현하자.

```sql
-- MySQL
CREATE TABLE tree (path varchar(255));
INSERT INTO tree (path) VALUES ('Food');
INSERT INTO tree (path) VALUES ('Food.Fruit');
INSERT INTO tree (path) VALUES ('Food.Fruit.Cherry');
INSERT INTO tree (path) VALUES ('Food.Fruit.Banana');
INSERT INTO tree (path) VALUES ('Food.Meat');
INSERT INTO tree (path) VALUES ('Food.Meat.Beaf');
INSERT INTO tree (path) VALUES ('Food.Meat.Pork');
SELECT * FROM tree WHERE path like 'Food.Fruit.%';
SELECT * FROM tree WHERE path IN('Food', 'Food.Fruit');

-- PostgreSQL
CREATE EXTENSION ltree;
CREATE TABLE tree (path ltree);
INSERT INTO tree (path) VALUES ('Food');
INSERT INTO tree (path) VALUES ('Food.Fruit');
INSERT INTO tree (path) VALUES ('Food.Fruit.Cherry');
INSERT INTO tree (path) VALUES ('Food.Fruit.Banana');
INSERT INTO tree (path) VALUES ('Food.Meat');
INSERT INTO tree (path) VALUES ('Food.Meat.Beaf');
INSERT INTO tree (path) VALUES ('Food.Meat.Pork');
SELECT * FROM tree WHERE path ~ 'Food.Fruit.*{1,}';
SELECT * FROM tree WHERE path @> subpath('Food.Fruit.Banana', 0, -1);
```

## JSON Columns to Combine NoSQL and Relational Databases	

아래와 같은 방법으로 JSON column 을 사용할 수 있다. JSON Column 을 사용하여 denormalization
을 적용하자. RDBMS 와 NoSQL 을 동시에 구현할 수 있다.

```sql
-- MySQL
CREATE TABLE books (
  id bigint PRIMARY KEY,
  author_id bigint NOT NULL,
  category_id bigint NOT NULL,
  name varchar(255) NOT NULL,
  price numeric(15, 2) NOT NULL,
  attributes json NOT NULL DEFAULT '{}'
);

-- PostgreSQL
CREATE TABLE books (
  id bigint PRIMARY KEY,
  author_id bigint NOT NULL,
  category_id bigint NOT NULL,
  name text NOT NULL,
  price numeric(15, 2) NOT NULL,
  attributes jsonb NOT NULL DEFAULT '{}'
);
```

## Alternative Tag Storage With JSON Arrays	

`JOIN` 을 여러번 하는 것보다 JSON column 을 만들어 저장하자. 간단한 SQL 로
질의할 수 있다.

```sql
-- MySQL
CREATE TABLE products (
  id bigint,
  name varchar(255),
  tagids json
);
CREATE INDEX producttags ON products ((CAST(tagids as unsigned ARRAY)));
SELECT *
  FROM products
 WHERE JSON_ARRAY(3, 8) MEMBER OF(tagids) AND NOT(12 MEMBER OF(tagids));

-- PostgreSQL
CREATE TABLE products (
  id bigint,
  name text,
  tagids jsonb
);
CREATE INDEX producttags ON products (tags jsonb_path_ops);
SELECT *
  FROM products
 WHERE tagids @> '[3,8]' AND NOT(tagids @> '[12]');
```

## Constraints for Improved Data Strictness	

아래와 같이 제한 조건을 걸어두자. Application 에서 제한 조건을 검증할 수도 있다.
그러나 Operation 할 때 Application 의 제한 조건을 사용할 수 없다.

```sql
ALTER TABLE reservations
  ADD constraint start_before_end CHECK (checkin_at < checkout_at);

ALTER TABLE invoices
  ADD constraint eu_vat CHECK (
    NOT(is_europeanunion) OR vatid IS NOT NULL
  );
```

## Validation Of JSON Colums Against A Schema	

아래와 같은 방법으로 JSON column 의 제한 조건을 걸 수 있다. 그렇다면
JSON column 을 왜 사용해야 할까?

```sql
-- MySQL
ALTER TABLE products ADD CONSTRAINT CHECK(
  JSON_SCHEMA_VALID(
    '{
      "$schema": "http://json-schema.org/draft-04/schema#",
      "type": "object",
      "properties": {
        "tags": {
          "type": "array",
          "items": { "type": "string" }
} },
      "additionalProperties": false
    }',
attributes
) );
ALTER TABLE products ADD CONSTRAINT data_is_valid CHECK(
  validate_json_schema(
    '{
      "type": "object",
      "properties": {
        "tags": {
          "type": "array",
          "items": { "type": "string" }
} },
      "additionalProperties": false
    }',
attributes
) );
```

## UUID Keys Against Enumeration Attacks	

아래와 같이 `uuid` 를 순서대로 만들지 말자. `Enumeration Attacks` 를 
방지할 수 있다.

```sql
-- MySQL
      ALTER TABLE users 
        ADD COLUMN uuid char(36);
     UPDATE users SET uuid = (SELECT uuid_v4());

      ALTER TABLE users 
     CHANGE COLUMN uuid uuid char(36) NOT NULL;
CREATE UNIQUE INDEX users_uuid ON users (uuid);

-- PostgreSQL
ALTER TABLE users 
  ADD COLUMN uuid uuid NOT NULL DEFAULT
      gen_random_uuid();
CREATE UNIQUE INDEX users_uuid ON users (uuid);
```

## Fast Delete Of Big Data With Partitions	

`Table` 을 `Partitioning` 해 놓으면 많은 data 를 `Partition` 별로 지울 수 있어서
효율적이다.

```sql
ALTER TABLE logs 
 DROP PARTITION logs_2022_january;
```

## Pre-Sorted Tables For Faster Access	

`Clustered Index` 를 잘 설정해 놓으면 Sorted SELECT 를 빠르게
실행할 수 있다.

```sql
  SELECT *
    FROM product_comments
   WHERE product_id = 2
ORDER BY comment_id ASC
   LIMIT 10

-- MySQL
CREATE TABLE product_comments (
  product_id bigint,
  comment_id bigint auto_increment UNIQUE KEY,
  message text,
  PRIMARY KEY (product_id, comment_id)
);

-- PostgreSQL
CREATE TABLE product_comments (
  product_id bigint,
  comment_id bigint GENERATED ALWAYS AS IDENTITY,
  message text,
  PRIMARY KEY (product_id, comment_id)
);
CLUSTER product_comments USING product_comments_pkey;
```

## Pre-Aggregation of Values for Faster Queres	

빠른 SELECT 를 위해 미리 aggregation 해 놓자.

```sql
SELECT SUM(likes_count)
  FROM articles
 WHERE user_id = 1 and 
       publish_year = 2022;
```

# Indexes	

## Indexes On Functions And Expressions	

함수의 결과도 인덱싱 된다.

```sql
SELECT * FROM users WHERE lower(email) = 'test@example.com';
-- MySQL
CREATE INDEX users_email_lower ON users ((lower(email)));
-- PostgreSQL
CREATE INDEX users_email_lower ON users (lower(email));
```

## Find Unused Indexes	

아래와 같은 방법으로 Unused Index 들을 찾을 수 있다.

```sql
-- MySQL
  SELECT object_schema AS `database`,
         object_name AS `table`,
         index_name AS `index`,
         count_star as `io_operations`
    FROM performance_schema.table_io_waits_summary_by_index_usage
   WHERE object_schema NOT IN('mysql', 'performance_schema') AND 
         index_name IS NOT NULL AND index_name != 'PRIMARY'
ORDER BY object_schema, object_name, index_name;

-- PostgreSQL
   SELECT pg_tables.schemaname AS schema,
          pg_tables.tablename AS table,
          pg_stat_all_indexes.indexrelname AS index,
          pg_stat_all_indexes.idx_scan AS number_of_scans,
          pg_stat_all_indexes.idx_tup_read AS tuples_read,
          pg_stat_all_indexes.idx_tup_fetch AS tuples_fetched
     FROM pg_tables
LEFT JOIN pg_class ON(pg_tables.tablename = pg_class.relname)
LEFT JOIN pg_index ON(pg_class.oid = pg_index.indrelid)
LEFT JOIN pg_stat_all_indexes USING(indexrelid)
    WHERE pg_tables.schemaname NOT IN ('pg_catalog', 'information_schema')
 ORDER BY pg_tables.schemaname, pg_tables.tablename;
```

## Safely Deleting Unused Indexes	(MySQL)

다음과 같은 방법으로 Unused Index 들을 비활성화 하자. Index 를 지우는 것은
오래걸릴 수 있다.

```sql
-- MySQL
ALTER TABLE website_visits ALTER INDEX twitter_referrals INVISIBLE;
ALTER TABLE website_visits ALTER INDEX twitter_referrals VISIBLE;
```

## Index-Only Operations By Including More Columns	

아래와 같은 방법으로 기존의 index 에 column 을 추가할 수 있다. `price` 를 기존의
index 에 추가하면 aggregation 이 빠르겠지?

```sql
SELECT SUM(price) 
  FROM invoices 
 WHERE customer_id = 42 AND 
       year = 2022;

-- MySQL
CREATE INDEX ON invoices (customer_id, year, price);

-- PostgreSQL
CREATE INDEX ON invoices (customer_id, year) INCLUDE (price);
```

## Partial Indexes To Reduce Index Size	(PostgreSQL)

아래와 같은 방법으로 일부데이터만 index 를 적용할 수 있다. index
의 크기를 줄일 수 있다.

```sql
SELECT * 
  FROM invoices 
 WHERE specialcase_law3421 = TRUE;

-- PostgreSQL
CREATE INDEX invoices_specialcase_law3421 
    ON invoices (specialcase_law3421)
 WHERE specialcase_law3421 = TRUE;
```

## Partial Indexes For Uniqueness Constraints	

아래와 같은 방법으로 Unique Index 를 부분적용할 수 있다.

```sql
-- MySQL
CREATE UNIQUE INDEX email_unique
    ON users (email, (IF(deleted_at, NULL, 1)));

-- PostgreSQL
CREATE UNIQUE INDEX email_unique
    ON users (email)
 WHERE deleted_at IS NULL;
```

## Index Support For Wildcard Searches (PostgreSQL)

아래와 같은 방법으로 `%Tobias%` 와 같은 검색을 위한 index 를 생성할 수 있다.

```sql
SELECT * 
  FROM speakers 
 WHERE name LIKE '%Tobias%';

-- PostgreSQL
CREATE EXTENSION pg_trgm;
CREATE INDEX trgm_idx 
    ON speakers 
 USING GIN (name gin_trgm_ops);
```

## Rules For Multi-Column Indexes	

Multi-Column Index 는 다음의 규칙을 유의하여 생성하자.

* equality column 은 different values 가 많은 것을 앞으로 둔다.
* range column 은 가장 나중에 둔다.

```sql
SELECT *
  FROM shop_articles
 WHERE tenant_id = 6382 AND 
       category = 'books' AND 
       price < 49.99;

CREATE INDEX shop_articles_key ON shop_articles (
  tenant_id, -- type = equality, different_values = 7293
  category, -- type = equality, different_values = 628
  price -- type = range, different_values = 142
);
```

## Hash Indexes To Descrease Index Size	(PostgreSQL)

Hash Index 는 B-Tree Index 보다 `INSERT, SELECT` 의 수행속도가 빠르다.
크기도 작다. 그러나 B-Tree Index 처럼 Unique 는 보장하지 않는다. 

```sql
-- PostgreSQL
CREATE INDEX invoices_uniqid ON invoices USING HASH (uniqid)
```

## Descending Indexes For Order By	

아래와 같은 방법으로 Multi-Column Index 를 생성할 때 순서를 정해두자. `SELECT`
가 빠르다.

```sql
  SELECT *
    FROM highscores
ORDER BY score DESC, created_at ASC
   LIMIT 10;

CREATE INDEX highscores_correct ON highscores (
  score DESC, created_at ASC
);
```

## Ghost Conditions Against Unindexed Columns	

Multi-Column Index `(status, type)` 가 적용되어 있다고 하자. `status = 'open'`
이고 보험가입이 되어 있는 선박을 검색해 보자. 

`type` 이 SQL 에 포함되지 않아도 된다. 그러나 `type` 을 사용하면 검색을 빠르게
수행할 수 있다.

```sql
-- Before
SELECT *
  FROM shipments
 WHERE status = 'open' AND 
       transportinsurance = 1;

-- After
SELECT *
  FROM shipments
 WHERE status = 'open' AND 
       transportinsurance = 1 AND 
       type = IN(3, 6, 11);
```
