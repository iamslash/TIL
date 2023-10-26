- [Abstract](#abstract)
- [Materials](#materials)
- [Fundamentals](#fundamentals)
  - [Index](#index)
  - [Pages](#pages)
  - [Clustered Index vs Non-clustered Index](#clustered-index-vs-non-clustered-index)
- [Index Access Principles](#index-access-principles)
  - [Principle 1: Fast Lookup](#principle-1-fast-lookup)
  - [Principle 2: Scan in One Direction](#principle-2-scan-in-one-direction)
  - [Principle 3: From Left To Right](#principle-3-from-left-to-right)
  - [Principle 4: Scan On Range Conditions](#principle-4-scan-on-range-conditions)
- [Index Supported Operations](#index-supported-operations)
  - [Inequality (!=)](#inequality-)
  - [Nullable Values (IS NULL and IS NOT NULL)](#nullable-values-is-null-and-is-not-null)
  - [Pattern Matching (LIKE)](#pattern-matching-like)
  - [Sorting Values (ORDER BY)](#sorting-values-order-by)
  - [Aggregating Values (DISTINCT and GROUP BY)](#aggregating-values-distinct-and-group-by)
  - [Joins](#joins)
  - [Subqueries](#subqueries)
  - [Data Manipulation (UPDATE and DELETE)](#data-manipulation-update-and-delete)
- [Why isn't Database Using My Index?](#why-isnt-database-using-my-index)
  - [The Index Can’t Be Used](#the-index-cant-be-used)
  - [No Index Will Be the Fastest](#no-index-will-be-the-fastest)
  - [Another index is faster](#another-index-is-faster)
- [Pitfalls and Tips](#pitfalls-and-tips)
  - [Indexes on Functions](#indexes-on-functions)
  - [Boolean Flags](#boolean-flags)
  - [Transforming Range Conditions](#transforming-range-conditions)
  - [Leading Wildcard Search](#leading-wildcard-search)
  - [Type Juggling](#type-juggling)
  - [Index-Only Queries](#index-only-queries)
  - [Filtering and Sorting With Joins](#filtering-and-sorting-with-joins)
  - [Exceeding the Maximum Index Size](#exceeding-the-maximum-index-size)
  - [JSON Objects and Arrays](#json-objects-and-arrays)
  - [Unique Indexes and Null](#unique-indexes-and-null)
  - [Location-Based Searching With Bounding-Boxes](#location-based-searching-with-bounding-boxes)

----

# Abstract

Database index. Hands on with [MySQL](/mysql/README.md).

# Materials

[Indexing Beyond the Basics](https://sqlfordevs.com/ebooks/indexing)

# Fundamentals

## Index

> * [SQL Unplugged 2013] 쉽고 재미있는 인덱스 이야기/ 씨퀄로 이장래](https://www.youtube.com/watch?v=TRfVeco4wZM)
> * [RDB index 정리](https://hyungjoon6876.github.io/jlog/2018/07/18/rdb-indexing.html)
> * [The chidb File Format](http://chi.cs.uchicago.edu/chidb/fileformat.html)

----

![](img/physlog.png)

위의 그림을 잘 살펴보자. xdb file 은 여러개의 page 로 구성되어 있다. 하나의 page
는 하나의 BTreeNode 를 저장한다. 하나의 BTreeNode 는 하나 이상의 record 를
저장한다. 또한 하나의 page 는 여러개의 cell 로 이루어져 있다. cell 은 record 에
대응된다. 

Courses 테이블은 schema 가 `CREATE TABLE Courses(Id INTEGER PRIMARY KEY, Name TEXT, Instructor INTEGER, Dept INTEGER)` 라고 해보자. primary key 인 id 를 hashing 한 값을 key 로 BTree 를 제작한다. 당연히 Id 를 조건으로 검색하면 빠르다. 

예를 들어 위 그림에서 id 를 hashing 한 값 `86` 을 검색하기 위해 BTree 를
`42->85->86` 순으로 검색에 성공했다.

그러나 Dept 를 조건으로 검색하면 느리다. Dept 의 인덱스를 제작한다. 즉, Dept 의
hashing 값을 key 로 B+Tree 를 제작한다. 당연히 Dept 를 조건으로 검색하면
빨라진다.

예를 들어 Dept 를 hashing 한 값 `42` 를 검색하기 위해 B+Tree 를 `10->42` 순으로
검색에 성공했다. `42` 는 primary key 인 ID 를 hashing 한 값 `67` 를 소유한 cell
을 가리키고 있다.

## Pages

* [How does SQLite work? Part 2: btrees! (or: disk seeks are slow don't do them!)](https://jvns.ca/blog/2014/10/02/how-does-sqlite-work-part-2-btrees/)
  * sqlite 의 database file 이 왜 pages 로 구성되어 있는지 설명한다.

----

database file 은 물리적으로 여러개의 pages 로 구성되야 한다.

CPU 는 disk 에서 데이터를 읽어들여 memory 로 로드해야 한다. 이때 데이터의 크기는 내가 찾고자 하는
것에 가까운 최소의 크기여야 한다. 내가 찾고자하는 데이터는 16 Byte 인데 1 MB 를 메모리에 로드하는 것은
낭비이다. 

그리고 filesystem 의 block size 는 4 KB 이다. 따라서 한번에 1 MB 를 메모리에서 읽는 것은 불가능하다.
여러번에 걸쳐 disk access 를 해야 한다. I/O 낭비가 발생한다.

따라서 데이터를 page 에 저장하고 그것의 크기는 block size 보다는 작고 
너무 작지 않게 설정하여 L1, L2 Cache hit 가 이루어 지도록 해야 한다.

참고로 chidb, sqlite 는 page 의 size 가 `1K` 이다.

## Clustered Index vs Non-clustered Index

> * [Difference between Clustered and Non-clustered index](https://www.geeksforgeeks.org/difference-between-clustered-and-non-clustered-index/)
> * [[MySQL] 인덱스구조 : 클러스터링인덱스/넌 클러스터링인덱스](https://pearlluck.tistory.com/m/54)

----

![](img/2022-10-23-07-57-59.png)

**Clustered Index** 는 Leaf Node 에 모든 Attribute Values 가 있다. SELECT 할 때
한번에 읽어 온다. 반면에 **Non-Clustered Index** 는 Leaf Node 가 **Clustered
Index** 의 주소를 갖는다. 다시한번 **Clustered Index** 를 찾아갈 필요가 있다. 추가적인 I/O 가 필요하다.

Index of `Roll_no is` is a **Clustered Index**.

```sql
create table Student
( Roll_No int primary key, 
Name varchar(50), 
Gender varchar(30), 
Mob_No bigint );

insert into Student
values (4, 'ankita', 'female', 9876543210 );

insert into Student 
values (3, 'anita', 'female', 9675432890 );

insert into Student 
values (5, 'mahima', 'female', 8976453201 ); 
```

There is no additional index data for `Roll_no`.

| Roll_No | Name | 	Gender | 	Mob_No |
|---|---|---|---|
| 3 | 	anita |	female	| 9675432890 |
| 4 |	ankita	| female |	9876543210 |
| 5 |	mahima |	female |	8976453201 |

Index of `Name` is a **Non-Clustered Index**.

```sql
create table Student
( Roll_No int primary key, 
Name varchar(50), 
Gender varchar(30), 
Mob_No bigint );

insert into Student 
values (4, 'afzal', 'male', 9876543210 );

insert into Student 
values (3, 'sudhir', 'male', 9675432890 );

insert into Student 
values (5, 'zoya', 'female', 8976453201 );

create nonclustered index NIX_FTE_Name
on Student (Name ASC); 
```

There is additional index data for `Name`. This needs record lookup.

| Name |	Row address |
|---|----|
| Afzal |	3452 |
| Sudhir |	5643 |
| zoya |	9876 |

일반적으로 테이블에 여러 개의 non-clustered index가 있을 때, clustered index를
auto-increment로 생성하는 것이 좋습니다. 그 이유는 다음과 같습니다.

* 인덱스 크기: auto-increment를 사용하여 생성한 clustered index는 숫자로 이루어져
있어 인덱스 크기가 작습니다. 이는 I/O 효율과 디스크 공간 절약에 도움이 됩니다.
* 삽입 성능: auto-increment 기반의 인덱스는 새 데이터가 인덱스 페이지 끝에
추가되기 때문에 페이지 분할 및 재구성이 최소화됩니다. 이로 인해 인덱스 생성과
유지에 관련된 오버헤드가 줄어들어 삽입 성능이 좋아집니다.
* 단순성: auto-increment로 생성된 clustered index는 단일 컬럼에 기반하기 때문에
인덱스 관리가 비교적 쉽습니다.


# Index Access Principles

## Principle 1: Fast Lookup

## Principle 2: Scan in One Direction

## Principle 3: From Left To Right

## Principle 4: Scan On Range Conditions

# Index Supported Operations

## Inequality (!=)

## Nullable Values (IS NULL and IS NOT NULL)

## Pattern Matching (LIKE)

## Sorting Values (ORDER BY)

## Aggregating Values (DISTINCT and GROUP BY)

## Joins

## Subqueries

## Data Manipulation (UPDATE and DELETE)

# Why isn't Database Using My Index?

## The Index Can’t Be Used

## No Index Will Be the Fastest

## Another index is faster

# Pitfalls and Tips

## Indexes on Functions

**Functional Indexes**

```sql
CREATE TABLE contacts (
    id bigint PRIMARY KEY AUTO_INCREMENT,
    birthday datetime NOT NULL
);
INSERT INTO contacts(birthday) VALUES (NOW());
INSERT INTO contacts(birthday) VALUES (NOW());
INSERT INTO contacts(birthday) VALUES (NOW());

-- Full scan
EXPLAIN SELECT * FROM contacts;
+----+-------------+----------+------------+------+---------------+------+---------+------+------+----------+-------+
| id | select_type | table    | partitions | type | possible_keys | key  | key_len | ref  | rows | filtered | Extra |
+----+-------------+----------+------------+------+---------------+------+---------+------+------+----------+-------+
|  1 | SIMPLE      | contacts | NULL       | ALL  | NULL          | NULL | NULL    | NULL |    3 |   100.00 | NULL  |
+----+-------------+----------+------------+------+---------------+------+---------+------+------+----------+-------+

-- Index scan
EXPLAIN SELECT * FROM contacts WHERE id = 1;
+----+-------------+----------+------------+-------+---------------+---------+---------+-------+------+----------+-------+
| id | select_type | table    | partitions | type  | possible_keys | key     | key_len | ref   | rows | filtered | Extra |
+----+-------------+----------+------------+-------+---------------+---------+---------+-------+------+----------+-------+
|  1 | SIMPLE      | contacts | NULL       | const | PRIMARY       | PRIMARY | 8       | const |    1 |   100.00 | NULL  |
+----+-------------+----------+------------+-------+---------------+---------+---------+-------+------+----------+-------+

-- Full scan
EXPLAIN SELECT * FROM contacts WHERE month(birthday) = 5;
+----+-------------+----------+------------+------+---------------+------+---------+------+------+----------+-------------+
| id | select_type | table    | partitions | type | possible_keys | key  | key_len | ref  | rows | filtered | Extra       |
+----+-------------+----------+------------+------+---------------+------+---------+------+------+----------+-------------+
|  1 | SIMPLE      | contacts | NULL       | ALL  | NULL          | NULL | NULL    | NULL |    3 |   100.00 | Using where |
+----+-------------+----------+------------+------+---------------+------+---------+------+------+----------+-------------+

-- Index scan
CREATE INDEX idx_contacts_birthmonth ON contacts ((month(birthday)));
EXPLAIN SELECT * FROM contacts WHERE month(birthday) = 5;
+----+-------------+----------+------------+------+-------------------------+-------------------------+---------+-------+------+----------+-------+
| id | select_type | table    | partitions | type | possible_keys           | key                     | key_len | ref   | rows | filtered | Extra |
+----+-------------+----------+------------+------+-------------------------+-------------------------+---------+-------+------+----------+-------+
|  1 | SIMPLE      | contacts | NULL       | ref  | idx_contacts_birthmonth | idx_contacts_birthmonth | 5       | const |    1 |   100.00 | NULL  |
+----+-------------+----------+------------+------+-------------------------+-------------------------+---------+-------+------+----------+-------+
```

**Vitual Column Indexes**

```sql
DROP TABLE contacts;
CREATE TABLE contacts (
    id bigint PRIMARY KEY AUTO_INCREMENT,
    birthday datetime NOT NULL,
    birthday_month TINYINT AS (month(birthday)) VIRTUAL NOT NULL,
    INDEX idx_contacts_birthmonth (birthday_month)
);
DESC contacts;
+----------------+----------+------+-----+---------+-------------------+
| Field          | Type     | Null | Key | Default | Extra             |
+----------------+----------+------+-----+---------+-------------------+
| id             | bigint   | NO   | PRI | NULL    | auto_increment    |
| birthday       | datetime | NO   |     | NULL    |                   |
| birthday_month | tinyint  | NO   | MUL | NULL    | VIRTUAL GENERATED |
+----------------+----------+------+-----+---------+-------------------+
INSERT INTO contacts(birthday) VALUES (NOW());
INSERT INTO contacts(birthday) VALUES (NOW());
INSERT INTO contacts(birthday) VALUES (NOW());

-- Index scan
EXPLAIN SELECT * FROM contacts WHERE month(birthday) = 5;
+----+-------------+----------+------------+------+-------------------------+-------------------------+---------+-------+------+----------+-------+
| id | select_type | table    | partitions | type | possible_keys           | key                     | key_len | ref   | rows | filtered | Extra |
+----+-------------+----------+------------+------+-------------------------+-------------------------+---------+-------+------+----------+-------+
|  1 | SIMPLE      | contacts | NULL       | ref  | idx_contacts_birthmonth | idx_contacts_birthmonth | 1       | const |    1 |   100.00 | NULL  |
+----+-------------+----------+------------+------+-------------------------+-------------------------+---------+-------+------+----------+-------+

-- Index scan
EXPLAIN SELECT * FROM contacts WHERE birthday_month = 5;
+----+-------------+----------+------------+------+-------------------------+-------------------------+---------+-------+------+----------+-------+
| id | select_type | table    | partitions | type | possible_keys           | key                     | key_len | ref   | rows | filtered | Extra |
+----+-------------+----------+------------+------+-------------------------+-------------------------+---------+-------+------+----------+-------+
|  1 | SIMPLE      | contacts | NULL       | ref  | idx_contacts_birthmonth | idx_contacts_birthmonth | 1       | const |    1 |   100.00 | NULL  |
+----+-------------+----------+------------+------+-------------------------+-------------------------+---------+-------+------+----------+-------+
```

## Boolean Flags

Boolean flags are not suitable for indexing in MySQL for several reasons:

* **Low cardinality**: Boolean flags have very low cardinality because they can
  only have two possible values: true or false (1 or 0). Indexes are typically
  more effective when the indexed column has a high number of distinct values. In
  the case of a Boolean flag, the index would end up pointing to many records with
  the same value, which decreases the index's efficiency and selectivity.
* **Poor query optimization**: Since Boolean flags have low cardinality, it
  makes it difficult for the query optimizer to make efficient execution plans.
  The optimizer may not choose to use an index on a Boolean flag due to its low
  selectivity, and even when it does, the full table scan might be faster than
  using the index, which defeats the purpose of having an index in the first
  place.
* **Larger index size**: Indexing a Boolean flag can increase the size of the
  index and add extra storage and maintenance overhead. Since the index may not
  be utilized effectively, the additional cost and complexity may not be
  worthwhile.
* **Limited performance improvement**: Using an index on a Boolean flag may not
  significantly improve query performance in scenarios where the distribution of
  values is highly skewed. For example, if 95% of records have the flag set to
  true and the query filters on the false value, the index may not provide
  substantial performance gains compared to a full table scan.

In conclusion, Boolean flags are not suitable for indexing in MySQL primarily
due to their low cardinality and limited effectiveness in query optimization. It
is generally more beneficial to index columns with higher cardinality and use
other query optimization techniques to improve performance for Boolean flag
columns.

## Transforming Range Conditions

Remove range conditions such as `WHERE stars > 1000` for better performance. You
might fetch unnecessary rows in the middle of fetching with the index because of
range conditions.

For solving range condition problems **virtual columns**, **functional indexes**
are good solutions.

**Index of (stars, language)**

```sql
CREATE TABLE trends (
    id bigint PRIMARY KEY AUTO_INCREMENT,
    language VARCHAR(128) NOT NULL,
    stars INT NOT NULL,
    sponsors INT NOT NULL
);
show index from trends;
+--------+------------+----------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
| Table  | Non_unique | Key_name | Seq_in_index | Column_name | Collation | Cardinality | Sub_part | Packed | Null | Index_type | Comment | Index_comment | Visible | Expression |
+--------+------------+----------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
| trends |          0 | PRIMARY  |            1 | id          | A         |           2 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
+--------+------------+----------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+

INSERT INTO trends(language, stars, sponsors) VALUES('TypeScript', 20, 10);
INSERT INTO trends(language, stars, sponsors) VALUES('TypeScript', 21, 11);
INSERT INTO trends(language, stars, sponsors) VALUES('Python', 22, 12);
INSERT INTO trends(language, stars, sponsors) VALUES('Python', 23, 13);
INSERT INTO trends(language, stars, sponsors) VALUES('Java', 24, 14);
INSERT INTO trends(language, stars, sponsors) VALUES('Java', 25, 15);
CREATE INDEX idx_stars_language ON trends (stars, language);
+--------+------------+--------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
| Table  | Non_unique | Key_name           | Seq_in_index | Column_name | Collation | Cardinality | Sub_part | Packed | Null | Index_type | Comment | Index_comment | Visible | Expression |
+--------+------------+--------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
| trends |          0 | PRIMARY            |            1 | id          | A         |           2 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_stars_language |            1 | stars       | A         |           6 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_stars_language |            2 | language    | A         |           6 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
+--------+------------+--------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+


-- Full scan
explain select * from trends where language = 'TypeScript';
+----+-------------+--------+------------+------+---------------+------+---------+------+------+----------+-------------+
| id | select_type | table  | partitions | type | possible_keys | key  | key_len | ref  | rows | filtered | Extra       |
+----+-------------+--------+------------+------+---------------+------+---------+------+------+----------+-------------+
|  1 | SIMPLE      | trends | NULL       | ALL  | NULL          | NULL | NULL    | NULL |    6 |    16.67 | Using where |
+----+-------------+--------+------------+------+---------------+------+---------+------+------+----------+-------------+

-- Index scan
explain select * from trends where stars > 20;
+----+-------------+--------+------------+-------+--------------------+--------------------+---------+------+------+----------+-----------------------+
| id | select_type | table  | partitions | type  | possible_keys      | key                | key_len | ref  | rows | filtered | Extra                 |
+----+-------------+--------+------------+-------+--------------------+--------------------+---------+------+------+----------+-----------------------+
|  1 | SIMPLE      | trends | NULL       | range | idx_stars_language | idx_stars_language | 4       | NULL |    5 |   100.00 | Using index condition |
+----+-------------+--------+------------+-------+--------------------+--------------------+---------+------+------+----------+-----------------------+

-- Index scan
explain select * from trends where language = 'TypeScript' AND stars > 20;
+----+-------------+--------+------------+-------+--------------------+--------------------+---------+------+------+----------+-----------------------+
| id | select_type | table  | partitions | type  | possible_keys      | key                | key_len | ref  | rows | filtered | Extra                 |
+----+-------------+--------+------------+-------+--------------------+--------------------+---------+------+------+----------+-----------------------+
|  1 | SIMPLE      | trends | NULL       | range | idx_stars_language | idx_stars_language | 4       | NULL |    5 |    16.67 | Using index condition |
+----+-------------+--------+------------+-------+--------------------+--------------------+---------+------+------+----------+-----------------------+

-- Index scan
explain select * from trends where stars > 20 AND language = 'TypeScript';
+----+-------------+--------+------------+-------+--------------------+--------------------+---------+------+------+----------+-----------------------+
| id | select_type | table  | partitions | type  | possible_keys      | key                | key_len | ref  | rows | filtered | Extra                 |
+----+-------------+--------+------------+-------+--------------------+--------------------+---------+------+------+----------+-----------------------+
|  1 | SIMPLE      | trends | NULL       | range | idx_stars_language | idx_stars_language | 4       | NULL |    5 |    16.67 | Using index condition |
+----+-------------+--------+------------+-------+--------------------+--------------------+---------+------+------+----------+-----------------------+

-- Index scan
explain analyze select * from trends where stars > 20 AND language = 'TypeScript';
-> Index range scan on trends using idx_stars_language over (20 < stars), with index condition: 
     ((trends.`language` = 'TypeScript') and (trends.stars > 20))  
       (cost=2.51 rows=5) (actual time=0.0513..0.0618 rows=1 loops=1)

-- Index scan (Query optimizer worked)
explain analyze select * from trends where language = 'TypeScript' AND stars > 20;
-> Index range scan on trends using idx_stars_language over (20 < stars), with index condition: 
     ((trends.`language` = 'TypeScript') and (trends.stars > 20))  
       (cost=2.51 rows=5) (actual time=0.0511..0.0675 rows=1 loops=1)
```

**Index of (language, stars)**

```sql
CREATE INDEX idx_language_stars ON trends (language, stars);

show index from trends;
+--------+------------+--------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
| Table  | Non_unique | Key_name           | Seq_in_index | Column_name | Collation | Cardinality | Sub_part | Packed | Null | Index_type | Comment | Index_comment | Visible | Expression |
+--------+------------+--------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
| trends |          0 | PRIMARY            |            1 | id          | A         |           2 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_stars_language |            1 | stars       | A         |           6 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_stars_language |            2 | language    | A         |           6 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_language_stars |            1 | language    | A         |           3 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_language_stars |            2 | stars       | A         |           6 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
+--------+------------+--------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+

-- Index scan
explain select * from trends where language = 'TypeScript';
+----+-------------+--------+------------+------+--------------------+--------------------+---------+-------+------+----------+-------+
| id | select_type | table  | partitions | type | possible_keys      | key                | key_len | ref   | rows | filtered | Extra |
+----+-------------+--------+------------+------+--------------------+--------------------+---------+-------+------+----------+-------+
|  1 | SIMPLE      | trends | NULL       | ref  | idx_language_stars | idx_language_stars | 514     | const |    2 |   100.00 | NULL  |
+----+-------------+--------+------------+------+--------------------+--------------------+---------+-------+------+----------+-------+
```

**Index of (language, stars, sponsors)**

```sql
CREATE INDEX idx_language_stars_sponsors ON trends (language, stars, sponsors);

show index from trends;
+--------+------------+-----------------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
| Table  | Non_unique | Key_name                    | Seq_in_index | Column_name | Collation | Cardinality | Sub_part | Packed | Null | Index_type | Comment | Index_comment | Visible | Expression |
+--------+------------+-----------------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
| trends |          0 | PRIMARY                     |            1 | id          | A         |           2 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_stars_language          |            1 | stars       | A         |           6 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_stars_language          |            2 | language    | A         |           6 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_language_stars          |            1 | language    | A         |           3 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_language_stars          |            2 | stars       | A         |           6 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_language_stars_sponsors |            1 | language    | A         |           3 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_language_stars_sponsors |            2 | stars       | A         |           6 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_language_stars_sponsors |            3 | sponsors    | A         |           6 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
+--------+------------+-----------------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+

-- Index scan
explain select * from trends where language = 'TypeScript';
+----+-------------+--------+------------+------+------------------------------------------------+-----------------------------+---------+-------+------+----------+-------------+
| id | select_type | table  | partitions | type | possible_keys                                  | key                         | key_len | ref   | rows | filtered | Extra       |
+----+-------------+--------+------------+------+------------------------------------------------+-----------------------------+---------+-------+------+----------+-------------+
|  1 | SIMPLE      | trends | NULL       | ref  | idx_language_stars,idx_language_stars_sponsors | idx_language_stars_sponsors | 514     | const |    2 |   100.00 | Using index |
+----+-------------+--------+------------+------+------------------------------------------------+-----------------------------+---------+-------+------+----------+-------------+

-- Index scan
explain select * from trends where language = 'TypeScript' ORDER BY sponsors ASC;
+----+-------------+--------+------------+------+------------------------------------------------+-----------------------------+---------+-------+------+----------+-----------------------------+
| id | select_type | table  | partitions | type | possible_keys                                  | key                         | key_len | ref   | rows | filtered | Extra                       |
+----+-------------+--------+------------+------+------------------------------------------------+-----------------------------+---------+-------+------+----------+-----------------------------+
|  1 | SIMPLE      | trends | NULL       | ref  | idx_language_stars,idx_language_stars_sponsors | idx_language_stars_sponsors | 514     | const |    2 |   100.00 | Using index; Using filesort |
+----+-------------+--------+------------+------+------------------------------------------------+-----------------------------+---------+-------+------+----------+-----------------------------+
```

**Virtual Columns (No need for range conditions)**

Index of (language, popular, sponsors) is better than range conditions. But
[PostgreSQL](/postgresql/README.md) doesn't support this.

```sql
CREATE TABLE trends (
    id bigint PRIMARY KEY AUTO_INCREMENT,
    language VARCHAR(128) NOT NULL,
    stars INT NOT NULL,
    popular TINYINT AS(IF(stars > 20, 1, 0)) VIRTUAL NOT NULL,
    sponsors INT NOT NULL
);
CREATE INDEX idx_language_popular_sponsors on trends(language, popular, sponsors);
SHOW INDEX from trends
+--------+------------+-------------------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
| Table  | Non_unique | Key_name                      | Seq_in_index | Column_name | Collation | Cardinality | Sub_part | Packed | Null | Index_type | Comment | Index_comment | Visible | Expression |
+--------+------------+-------------------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+
| trends |          0 | PRIMARY                       |            1 | id          | A         |           0 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_language_popular_sponsors |            1 | language    | A         |           0 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_language_popular_sponsors |            2 | popular     | A         |           0 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
| trends |          1 | idx_language_popular_sponsors |            3 | sponsors    | A         |           0 |     NULL |   NULL |      | BTREE      |         |               | YES     | NULL       |
+--------+------------+-------------------------------+--------------+-------------+-----------+-------------+----------+--------+------+------------+---------+---------------+---------+------------+

INSERT INTO trends(language, stars, sponsors) VALUES('TypeScript', 20, 10);
INSERT INTO trends(language, stars, sponsors) VALUES('TypeScript', 21, 11);
INSERT INTO trends(language, stars, sponsors) VALUES('Python', 22, 12);
INSERT INTO trends(language, stars, sponsors) VALUES('Python', 23, 13);
INSERT INTO trends(language, stars, sponsors) VALUES('Java', 24, 14);
INSERT INTO trends(language, stars, sponsors) VALUES('Java', 25, 15);

explain analyze select * from trends where language = 'TypeScript' AND popular = 1 ORDER BY sponsors ASC;
-> Index lookup on trends using idx_language_popular_sponsors (language='TypeScript', popular=1)  
     (cost=0.35 rows=1) (actual time=0.0563..0.0614 rows=1 loops=1)
```

**Function Indexes (No need for range conditions)**

```sql
CREATE TABLE trends (
    id bigint PRIMARY KEY AUTO_INCREMENT,
    language VARCHAR(128) NOT NULL,
    stars INT NOT NULL,
    sponsors INT NOT NULL
);
INSERT INTO trends(language, stars, sponsors) VALUES('TypeScript', 20, 10);
INSERT INTO trends(language, stars, sponsors) VALUES('TypeScript', 21, 11);
INSERT INTO trends(language, stars, sponsors) VALUES('Python', 22, 12);
INSERT INTO trends(language, stars, sponsors) VALUES('Python', 23, 13);
INSERT INTO trends(language, stars, sponsors) VALUES('Java', 24, 14);
INSERT INTO trends(language, stars, sponsors) VALUES('Java', 25, 15);
CREATE INDEX idx_language_popular_sponsors ON trends (language, (IF(stars > 20, 1, 0)), sponsors);

explain analyze 
select * 
  from trends 
 where language = 'TypeScript' AND 
       IF(stars > 20, 1, 0) = 1 
 ORDER BY sponsors ASC;
-> Index lookup on trends using idx_language_popular_sponsors (language='TypeScript', if((stars > 20),1,0)=1)   
     (cost=0.35 rows=1) (actual time=0.0497..0.0507 rows=1 loops=1)
```

## Leading Wildcard Search

Wildcard condition is converted to range condition. For example, `WHERE name
LIKE 'Tob%s'` is converted to `WHEREname >= 'Tob' AND name < 'Toc`. 

Remove leading wildcard for better performance. `WHERE name LIKE '%obs'` can
lead to poor performance When a leading wildcard is used, the database engine
cannot utilize the index efficiently since it has to search for all possible
combinations of characters preceding the search term (e.g., `{%Iobs, Aobs, Bobs,..., zobs, ...}` ). This results in a full table scan, which can be slow for
large tables.

But [PostgreSQL](/postgresql/README.md) supports indexing of wildcard any position.

```sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX trgm_idx ON contacts USING GIN (name gin_trgm_ops);
```

## Type Juggling

When we query `int` attributes for `varchar` column index, **full scan**
happens. But when we query `string` attributes for `int` column index, 
**index scan** happens.

The index `idx_orders` is not used for 
`EXPLAIN SELECT * FROM orders WHERE payment_id = '57013925718'`. 
Because `EXPLAIN SELECT * FROM orders WHERE payment_id = '57013925718'`
is changed to `SELECT * FROM orders WHERE CAST(payment_id AS UNSIGNED) = 57013925718`. 

```sql
CREATE TABLE orders (
    id bigint PRIMARY KEY AUTO_INCREMENT,
    payment_id VARCHAR(255) NOT NULL
);
INSERT INTO orders(payment_id) VALUES (`57013925718`);
INSERT INTO orders(payment_id) VALUES (`57013925728`);
INSERT INTO orders(payment_id) VALUES (`57013925738`);
CREATE INDEX idx_orders ON orders (payment_id);

-- Index scan
EXPLAIN SELECT * FROM orders WHERE payment_id = '57013925718';
+----+-------------+--------+------------+------+---------------+------------+---------+-------+------+----------+-------------+
| id | select_type | table  | partitions | type | possible_keys | key        | key_len | ref   | rows | filtered | Extra       |
+----+-------------+--------+------------+------+---------------+------------+---------+-------+------+----------+-------------+
|  1 | SIMPLE      | orders | NULL       | ref  | idx_orders    | idx_orders | 1022    | const |    1 |   100.00 | Using index |
+----+-------------+--------+------------+------+---------------+------------+---------+-------+------+----------+-------------+

-- Full scan
EXPLAIN SELECT * FROM orders WHERE payment_id = 57013925718;
+----+-------------+--------+------------+-------+---------------+------------+---------+------+------+----------+--------------------------+
| id | select_type | table  | partitions | type  | possible_keys | key        | key_len | ref  | rows | filtered | Extra                    |
+----+-------------+--------+------------+-------+---------------+------------+---------+------+------+----------+--------------------------+
|  1 | SIMPLE      | orders | NULL       | index | idx_orders    | idx_orders | 1022    | NULL |    3 |    33.33 | Using where; Using index |
+----+-------------+--------+------------+-------+---------------+------------+---------+------+------+----------+--------------------------+
```

The opposite case works.

```sql
CREATE TABLE orders (
    id bigint PRIMARY KEY AUTO_INCREMENT,
    payment_id bigint NOT NULL
);
INSERT INTO orders(payment_id) VALUES (57013925718);
INSERT INTO orders(payment_id) VALUES (57013925728);
INSERT INTO orders(payment_id) VALUES (57013925738);
CREATE INDEX idx_orders ON orders (payment_id);

-- Index Scan
EXPLAIN SELECT * FROM orders WHERE payment_id = '57013925718';
+----+-------------+--------+------------+------+---------------+------------+---------+-------+------+----------+--------------------------+
| id | select_type | table  | partitions | type | possible_keys | key        | key_len | ref   | rows | filtered | Extra                    |
+----+-------------+--------+------------+------+---------------+------------+---------+-------+------+----------+--------------------------+
|  1 | SIMPLE      | orders | NULL       | ref  | idx_orders    | idx_orders | 8       | const |    1 |   100.00 | Using where; Using index |
+----+-------------+--------+------------+------+---------------+------------+---------+-------+------+----------+--------------------------+

-- Index Scan
EXPLAIN SELECT * FROM orders WHERE payment_id = 57013925718;
+----+-------------+--------+------------+------+---------------+------------+---------+-------+------+----------+-------------+
| id | select_type | table  | partitions | type | possible_keys | key        | key_len | ref   | rows | filtered | Extra       |
+----+-------------+--------+------------+------+---------------+------------+---------+-------+------+----------+-------------+
|  1 | SIMPLE      | orders | NULL       | ref  | idx_orders    | idx_orders | 8       | const |    1 |   100.00 | Using index |
+----+-------------+--------+------------+------+---------------+------------+---------+-------+------+----------+-------------+
```

## Index-Only Queries

Index-only queries, also known as "covering indexes" or "covered queries," are a
type of query in MySQL wherein all the data required to satisfy the query can be
retrieved from the index itself, without accessing the actual table data (rows).

When a query uses a covering index, the MySQL query optimizer can avoid reading
the rows from the table, making the query execution faster and more efficient.
This can happen when all the columns needed for the query are part of the index,
either in the key columns or as included columns.

For example, let's assume we have the following table and index:

```sql
CREATE TABLE users (
  id INT(11) NOT NULL,
  first_name VARCHAR(50) NOT NULL,
  last_name VARCHAR(50) NOT NULL,
  email VARCHAR(50) NOT NULL,
  PRIMARY KEY (id),
  INDEX idx_name_email (first_name, last_name, email)
);
```

We can now run an index-only query by selecting only the columns available in
the index:

```sql
SELECT first_name, 
       last_name, 
       email
  FROM users
 WHERE first_name = 'John' AND 
       last_name = 'Doe';
```

This query can be executed using only the `idx_name_email` index, without
accessing the table data, leading to faster and more efficient query execution.

Index-only queries require all the necessary columns for the query to be present
in the index. This can lead to larger index sizes, taking up more disk space and
memory, which can negatively impact overall system performance.

## Filtering and Sorting With Joins

Let's say the number of tasks is 20,000 and the number of projects is 40 and the
number of joined records is 120,000 and the the number of result is 40.

```
                20,000                40
------> tasks ---------> projects ---------->
<------       <---------          <----------
   40          120,000
```

The number of filtered `tasks` should be small for better performance.

```sql
SELECT tasks.*
  FROM tasks
  JOIN projects USING(project_id)
 WHERE tasks.team_id = 4 AND 
       taks.status = 'open' AND 
       projects.status = 'open';
```

The number of filtered `invoices` should be small for better performacne.

```sql
SELECT *
  FROM invoices
  JOIN invoices_metadata USING(invoice_id)
 WHERE invoices.tenant_id = 4236
 ORDER BY invoices_metadata.due_date
 LIMIT 30
```

## Exceeding the Maximum Index Size

The size of index should be lower than data. Unless it will make following
error. `The index row size of 3480 bytes exceeds the maximum size of 2712 bytes for index contacts_fullname`

There are good strategy for index size optimization including **Prefix Index**, **Indexing Hash Values**, **Multiple Columns**.

**Prefix Index**

The main advantage of using a prefix index is that it significantly reduces the
size of the index, which in turn can speed up search operations. This can be
particularly beneficial when working with large volumes of text data, such as
with VARCHAR or TEXT fields.

```sql
CREATE INDEX articles_search ON articles (type, (substring(title, 1, 20)));
SELECT *
  FROM articles
 WHERE substring(title, 1, 20) = '...20 chars truncated title...' AND
       title = '...full title...'
```
However, there are some drawbacks and limitations to using prefix indexes:

* Prefix indexes can only be used with MyISAM and InnoDB storage engines. 
* When using a prefix index, MySQL may not be able to use the index for certain
  queries or operations like sorting or searching with the LIKE operator
  (depending on the query and storage engine). 
* Maintenance and updates to the index can be slower because it involves string
  comparisons.

```sql
CREATE INDEX articles_search ON articles (type, title(20));
SELECT * FROM articles WHERE title = '...full title...'
```

In this example, a prefix index named articles_search is created on the articles
table, including both the 'type' column and the 'title' column with a length of
20 characters. The storage engine will only index the first 20 characters of the
'title' field, which can be beneficial for optimizing the performance of search
queries on large VARCHAR or TEXT fields.  

This approach can improve the query execution speed by utilizing the prefix
index for efficient searching while still maintaining the accuracy of the
results based on the full title. However, keep in mind the considerations and
limitations of using a prefix index as mentioned in the previous response, such
as compatibility with specific storage engines and potential impact on index
maintenance.


**Indexing Hash Values**

An index based on hash values involves creating an index using a hash function,
such as SHA1, applied to the indexed column values. This can be useful for
speeding up search operations, especially for large text or binary columns.

```sql
CREATE INDEX articles_search ON articles (type, (sha1(title)));
SELECT *
  FROM articles
 WHERE sha1(title) = '...hash of title...' AND 
       title = '...full title...'
```

In the example you have provided, a hash index is created on the articles table
named `articles_search`, including the 'type' column and an SHA1 hash of the
'title' column. Applying the SHA1 hash function to the 'title' column values
helps to create a fixed-length hash value (40 characters), which can be used to
improve search performance and reduce the index size.

The subsequent SELECT statement uses both the SHA1 hash value and the full title
as search conditions in the query. The search operation first compares the SHA1
hash value through the index, which helps to quickly narrow down the potential
search results. Once the search is filtered based on the hash value, MySQL then
compares the full title to find the exact match for the query.

Using a hash index can improve search performance on large text columns, as the
hash values can be compared faster than the larger original text values.
However, there are some caveats and limitations to consider when using hash
indexes:

* In MySQL, hash index support is available only for the **MEMORY** storage engine,
  not for **MyISAM** or **InnoDB**, which limits its general use.
* Hash functions, such as **SHA1**, are not guaranteed to produce unique hash
  values for unique inputs (i.e., collisions can occur), which might result in
  false positives during the search. Including the full title comparison in the
  WHERE clause, as demonstrated in the example, helps ensure accurate results.
* Updates and modifications to the data may involve recalculating the hash
  value, as well as updating the index, affecting the performance of write
  operations.

```sql
CREATE INDEX articles_search ON articles (type, (SUBSTRING(sha1(title), 1, 20)));
SELECT *
  FROM articles
 WHERE SUBSTRING(sha1(title), 1, 20)) = '...shortened hash of title...' AND 
       title = '...full title...'
```

In this example, you are attempting to create an index on the articles table
named articles_search by incorporating the 'type' column and a truncated hash
value of the 'title' column. The truncation is achieved by applying the
SUBSTRING function to the result of the SHA1 hash function on the 'title',
reducing the hash value to a length of 20 characters. The idea behind this is to
create smaller indexed values which would potentially reduce the index size and
improve search performance.

However, creating an index in this manner is not directly supported by MySQL for
InnoDB or MyISAM storage engines in the way you have provided in your example.  

```sql
CREATE INDEX articles_search ON articles (type, (md5(title)::uuid));
SELECT *
  FROM articles
 WHERE md5(title)::uuid = md5('...full title...')::uuid AND 
       title = '...full title...'       
```

In this case, you are creating an index on the articles table named
articles_search, which includes the 'type' column and the MD5 hash of the
'title' column cast as a UUID (Universal Unique Identifier) data type. The
purpose is to create a fixed-length indexed value (128-bit) based on the hash,
which could potentially improve search performance. PostgreSQL supports the
`::uuid` casting operator, unlike MySQL.

However, even in PostgreSQL, there are important factors to consider when
relying on hashed column values for indexing:

* Hash collisions: Although rare, MD5 is not guaranteed to produce unique hash
  values for unique text inputs. Including the full title comparison in the
  WHERE clause helps to ensure the accuracy of the search results.
* Write performance: When any change is made to the 'title' column, the hash
  value and the index have to be updated, which may impact write performance.
* Query planning: In some cases, PostgreSQL’s query planner might not choose the
  index in query execution, and this could affect search performance.

**Multiple Columns**

Here are some key points to consider for using multiple columns for indexing
optimization:

- **Column order**: The order of columns in the index plays a crucial role, as it
  determines the search efficiency for different query patterns. In general, you
  should place columns with a higher level of selectivity, or columns with
  higher cardinality (i.e., a larger number of distinct values), earlier in the
  index.
- **Index efficiency**: For a multi-column index to be effective, the indexed
  WHERE, ORDER BY, or JOIN clauses. This helps in utilizing the index more
  columns should frequently appear together in query conditions, such as in
  efficiently, improving the performance of the query execution.
- **Covering index**: An index that includes all the columns needed for processing a
  query is called a covering index. When a query can be fulfilled entirely using
  the data stored in an index, the database engine doesn't have to access the
  underlying table, resulting in faster query execution.
- **Index size**: Multi-column indexes may have a smaller size compared to
  maintaining multiple single-column indexes, as the additional overhead for
  each index is reduced. A smaller index size also means less disk and memory
  usage, potentially improving performance.
- **Write performance**: Keep in mind that having multiple indexes or composite
  indexes on a table may affect the performance of INSERT, UPDATE, and DELETE
  operations since the indexes must be updated each time a relevant modification
  is made to the data. Finding the right balance between read performance and
  write performance is crucial.

```sql
CREATE INDEX articles_search ON articles (type, author, title);
```

In this example, a multi-column index named articles_search is created on the
articles table, comprising the 'type', 'author', and 'title' columns.

Overall, using multiple columns in an index can be beneficial for optimizing
size and improving query performance. However, it is important to consider the
specific needs and query patterns of your application to determine the optimal
index design. Be sure to analyze and monitor the index usage to continuously
fine-tune the indexing strategy.

## JSON Objects and Arrays

There 3 stretegy in JSO Object Indexing including **Virtual Columns**, 
**Functional Indexes**, **GIN Indexes**.

**Virtual Columns**

```sql
CREATE TABLE contacts (
  id bigint PRIMARY KEY AUTO_INCREMENT,
  attributes json NOT NULL,
  email varchar(255)  AS (attributes->>"$.email") VIRTUAL NOT NULL,
  INDEX contacts_email (email)
);
SELECT * 
  FROM contacts 
 WHERE attributes->>"$.email" ='admin@example.com';
```

Using virtual columns of JSON objects in MySQL provides the following advantages:

* Enhances the query performance of JSON data by creating indexes on specific
  JSON attributes using virtual columns.
* Enables better schema flexibility and design, by allowing a combination of
  structured and semi-structured data in one table.
* Reduces storage space requirements since virtual column values aren't
  physically stored and are only calculated when accessed.

However, it's important to consider the balance between performance gains and
potential computation overhead, as virtual columns are calculated at runtime. In
most cases, the overall benefits of using virtual columns to extract and index
JSON attributes outweigh the potential overhead.

**Functional Indexes**

```sql
-- PostgreSQL
CREATE INDEX contacts_email ON contacts ((attributes->>'email'));
SELECT * 
  FROM contacts 
 WHERE attributes->>'email' = 'admin@example.com';

-- MySQL
CREATE INDEX contacts_email ON contacts ((
  CAST(attributes->>"$.email" AS CHAR(255)) COLLATE utf8mb4_bin
));
SELECT * 
  FROM contacts  
 WHERE attributes->>"$.email" = 'admin@example.com';
```

Using a function index for JSON objects in SQL databases provides the following advantages:

* Enhances query performance by speeding up searches on specific JSON attributes
  through the use of indexes. 
* Enables better handling of JSON data by simplifying the query conditions for
  attribute filtering.
* Allows for indexing and searching on computed values, making complex queries
  more efficient. 
  
However, there are some potential trade-offs to consider when using function indexes:

* Function indexes may have an impact on write performance, as the index needs
  to be updated whenever there is a change in the indexed data.
* The computation of function-based index values can create some overhead in
  cases where the function is complex or computationally intensive.
* Syntax and support for function indexes may vary between different SQL
  database systems, so it is crucial to understand the specific capabilities and
  limitations of your chosen database.

**GIN Indexes**

```sql
CREATE INDEX contacts_attributes ON contacts USING gin (attributes);
```

A GIN (Generalized Inverted Index) index is a type of index available in
**PostgreSQL** that is specifically designed to index complex data types, such
as arrays, full-text search tsvector, and JSON/JSONB objects. GIN indexes
provide high-performance search capabilities on such complex data fields and are
especially useful for searching JSON data based on key-value pairs, keys, or
values.

Using GIN indexes for JSON objects in PostgreSQL provides the following advantages:

* Enhances query performance for complex JSON queries, such as containment and
  existence queries, by utilizing a high-performance search index.
* Provides an efficient way to index key-value pairs, keys, or values in JSON
  objects. Suits well for read-heavy workloads and infrequent writes.
* Supports various query types and operators, making it suitable for different
  use cases when working with JSON data.

However, it's important to consider some potential trade-offs when using GIN indexes:

* **GIN** indexes may have an impact on write performance since the index needs to
  be updated whenever there's a change in the indexed data.
* **GIN** indexes consume more storage space compared to other index types, such as
  **B-tree** or **GiST** indexes.
* Building a **GIN** index can be time-consuming, especially for large data sets.

```sql
-- Containment query:
SELECT * FROM contacts WHERE attributes @> '{"email": "admin@example.com"}';
-- This query returns all rows where the attributes JSON object contains an 
-- 'email' key with the value "admin@example.com". The @> operator checks 
-- if the JSON on the left contains the JSON on the right.

-- Existence query:
SELECT * FROM contacts WHERE attributes ? 'email';
-- This query returns all rows with the 'email' key present 
-- in the attributes JSON object. The ? operator checks if the key 
-- on the right exists in the JSON object on the left.

-- Key existence (OR) query:
SELECT * FROM contacts WHERE attributes ?| array['email', 'phone'];
-- This query returns all rows where either 'email' key or 'phone' key 
-- is present in the attributes JSON object. The ?| operator checks 
-- if any key in the array on the right exists in the JSON object on the left.

-- Key existence (AND) query:
SELECT * FROM contacts WHERE attributes ?& array['email', 'phone'];
-- This query returns all rows where both 'email' key and 'phone' 
-- key are present in the attributes JSON object. The ?& operator checks 
-- if all keys in the array on the right exist in the JSON object on the left.

-- JSON path query:
SELECT * FROM contacts WHERE attributes @? '$.tags[*] ? (@.type == "note" && @.severity > 3)';
-- This query uses the JSON path language, specified by the '@?' operator, 
-- to filter rows based on more complex criteria. In this case, 
-- it returns all rows where the attributes JSON data has elements 
-- in the 'tags' array with a 'type' key equal to "note" and 
-- a 'severity' key with a value greater than 3.
```

**JSON Arrays**

```sql
-- PostgreSQL
CREATE INDEX products_categories ON products USING gin(categories);
SELECT * 
  FROM products 
 WHERE categories @> '["printed book", "ebook"]' AND 
       NOT categories @> '["audiobook"]';

-- MySQL
CREATE INDEX products_categories on products ((CAST(categories AS UNSIGNED ARRAY)));
SELECT * 
  FROM contacts 
 WHERE JSON_CONTAINS(attributes, CAST('[17, 23]' AS JSON)) AND 
       NOT JSON_CONTAINS(attributes, CAST('[11]' AS JSON))
```

## Unique Indexes and Null

When dealing with unique indexes, NULL values are considered distinct from each
other. This means that a unique index allows multiple rows to have NULL values
in the indexed column(s) without violating the uniqueness constraint.

```sql
-- PostgreSQL
-- Creating the 'users' table
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  email VARCHAR(255) NOT NULL,
  alternate_email VARCHAR(255)
);

-- Creating a unique index on the 'email' column
CREATE UNIQUE INDEX users_email ON users (email);

-- Creating a unique index on the 'alternate_email' column
CREATE UNIQUE INDEX users_alternate_email ON users (alternate_email);

-- Inserting data (valid)
INSERT INTO users (email, alternate_email) VALUES ('user1@example.com', NULL);
INSERT INTO users (email, alternate_email) VALUES ('user2@example.com', NULL);

-- Inserting a duplicate value (violates unique constraint)
-- This will fail
INSERT INTO users (email, alternate_email) VALUES ('user1@example.com', NULL);
```

```sql
-- MySQL
-- Creating the 'users' table
CREATE TABLE users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  email VARCHAR(255) NOT NULL,
  alternate_email VARCHAR(255),
  UNIQUE INDEX users_email (email),
  UNIQUE INDEX users_alternate_email (alternate_email)
);

-- Inserting data (valid)
INSERT INTO users (email, alternate_email) VALUES ('user1@example.com', NULL);
INSERT INTO users (email, alternate_email) VALUES ('user2@example.com', NULL);

-- Inserting a duplicate value (violates unique constraint)
-- This will fail
INSERT INTO users (email, alternate_email) VALUES ('user1@example.com', NULL);
```

This will fix that problem which does not suits for not null unique index.

```sql
-- Creating table
CREATE TABLE orders (
  id SERIAL PRIMARY KEY,
  customer_id INT,
  shipment_id INT
);

-- Creating a partial unique index for non-null customer_id and shipment_id
CREATE UNIQUE INDEX uniqueness_idx_customer_shipment_not_null ON orders (
  customer_id, shipment_id
) WHERE customer_id IS NOT NULL AND shipment_id IS NOT NULL;

-- Adding a check constraint to prevent both columns from being NULL at the same time
ALTER TABLE orders ADD CONSTRAINT chk_customer_shipment_not_both_null CHECK (
  customer_id IS NOT NULL OR shipment_id IS NOT NULL
);
```

This will fix not null discint problem.

```sql
CREATE UNIQUE INDEX uniqueness_idx ON orders (
  customer_id,
  (CASE WHEN shipment_id IS NULL THEN -1 ELSE shipment_id END)
);
```

However, there are several considerations when using this approach:

- The solution assumes that the replaced value (-1, in this case) will never be
  used as a valid value for the `shipment_id` column, as it might lead to unexpected
  behavior when inserting or updating data. You need to ensure that this value is
  not used accidentally as an actual value.
- Replacing the `NULL` values with distinct values like this can be seen as a bit
  "hacky" and might introduce some confusion when maintaining or debugging the
  database schema in the future.

## Location-Based Searching With Bounding-Boxes

```sql
CREATE TABLE businesses (
  id bigint PRIMARY KEY NOT NULL,
  type varchar(255) NOT NULL,
  name varchar(255) NOT NULL,
  latitude float NOT NULL,
  longitude float NOT NULL
);
CREATE INDEX search_idx ON businesses (longitude, latitude);

SELECT *
  FROM businesses
 WHERE type = 'restaurant' longitude BETWEEN -73.9752 AND -74.0083 AND 
       latitude BETWEEN 40.7216 AND 40.7422
```

Location-based searching with bounding boxes is a technique used to search and
filter results based on their geographical locations. This can be useful for
applications like restaurant finders, real estate websites, and
geolocation-based search engines. Bounding boxes are defined as rectangular
areas within which a search is performed, and they are typically specified by
their top-left and bottom-right coordinates (latitude and longitude).

In the example provided, we have a table called 'businesses' with columns 'id',
'type', 'name', 'latitude', and 'longitude'. We create an index named
'search_idx' on the 'longitude' and 'latitude' columns to improve the
performance of location-based queries.

The SQL query in the example searches for businesses with the type 'restaurant'
located within specified longitude and latitude boundaries. The longitude
boundaries are defined as -73.9752 (left) and -74.0083 (right), while the
latitude boundaries are set to 40.7216 (bottom) and 40.7422 (top). This query
will return all restaurants within the specified bounding box.

To execute efficient location-based searches in SQL databases like MySQL and
PostgreSQL, it is essential to make use of spatial indexing techniques such as
`R-trees` or `GiST` (Generalized Search Tree) for faster retrieval of records based
on their geographical location. Using bounding boxes is a simple but effective
method for narrowing down results on location-based searches, allowing for
efficient and fast retrieval of relevant information.

location-based searching involves finding and filtering records based on their
geographical locations, which is particularly useful for applications like
restaurant finders, real estate websites, and geolocation-based search engines.
Several methods and data structures can be used to improve the efficiency of
location-based searches, such as:

- Bounding Boxes: A simple technique where a rectangular area is defined by its
  top-left and bottom-right coordinates, and searching is performed within that
  area.
- Quadtrees: A tree data structure used for partitioning a two-dimensional space
  by recursively subdividing it into four equal quadrants or regions, enabling
  efficient storage and retrieval of spatial data.
- R-tree: A spatial indexing method that uses bounding rectangles with variable
  sizes and overlaps to partition the space, improving search performance for
  datasets with irregular spatial distribution.
- k-d tree: A tree data structure for partitioning k-dimensional space,
  particularly efficient for nearest neighbor and point-radius searches.
- Geohash: A geocoding system using short alphanumeric strings to represent
  rectangular areas on Earth, based on Z-order curve space-filling techniques,
  useful for searching in distributed systems or big data platforms.
- Spatial Partitioning: Dividing the dataset into pre-defined partitions or
  zones according to geographical boundaries for more efficient search
  performance.
- Haversine Formula: A formula for calculating the great-circle distance between
  two points on Earth's surface, used for filtering search results based on the
  distance between the user's location and the records in the dataset.
