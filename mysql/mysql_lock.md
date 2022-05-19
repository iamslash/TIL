- [Abstract](#abstract)
- [Materials](#materials)
- [MySQL Lock](#mysql-lock)
- [Shared and Exclusive Locks](#shared-and-exclusive-locks)
- [Intention Locks](#intention-locks)
- [Record Locks](#record-locks)
- [Gap Locks](#gap-locks)
- [Next-Key Locks](#next-key-locks)
- [Insert Intention Locks](#insert-intention-locks)
- [AUTO-INC Locks](#auto-inc-locks)
- [Predicate Locks for Spatial Indexes](#predicate-locks-for-spatial-indexes)
- [Experiment](#experiment)
- [Inno-db Deadlock](#inno-db-deadlock)
- [MySQL Optimistic Locking](#mysql-optimistic-locking)

----

# Abstract

MySQL InnoDB 의 Lock 에 대해 정리한다. MySQL 은 Lock 으로 [Concurrency
Problems](/database/README.md#concurrency-problems-in-transactions) 을 해결한다.
즉, [Isolation Level](/isolation/README.md) 을 Lock 으로 구현한다.

# Materials

> * [Lock으로 이해하는 Transaction의 Isolation Level](https://suhwan.dev/2019/06/09/transaction-isolation-level-and-lock/)
> * [MySQL InnoDB lock & deadlock 이해하기](https://www.letmecompile.com/mysql-innodb-lock-deadlock/)
> * [14.7.1 InnoDB Locking @ mysql](https://dev.mysql.com/doc/refman/5.7/en/innodb-locking.html)
> * [InnoDB locking](https://github.com/octachrome/innodb-locks)
> * [20-MySQL의 잠금 | MySQL DBA 튜토리얼 | MySQL 8 DBA 튜토리얼 | youtube](https://www.youtube.com/watch?v=8NlElO5-Xbk)
> * [MySQL Gap Lock 다시보기](https://medium.com/daangn/mysql-gap-lock-%EB%8B%A4%EC%8B%9C%EB%B3%B4%EA%B8%B0-7f47ea3f68bc)

# MySQL Lock

mysql 의 innodb 는 다음과 같은 종류의 lock 을 갖는다.

* Shared and Exclusive Locks
* Intention Locks
* Record Locks
* Gap Locks
* Next-Key Locks
* Insert Intention Locks
* AUTO-INC Locks
* Predicate Locks for Spatial Indexes

다음과 같이 [mysql](/mysql/README.md) 을 실행한다.

```bash
$ docker run -p 3306:3306 --rm --name my-mysql -e MYSQL_ROOT_PASSWORD=1 -e MYSQL_DATABASE=hello -e MYSQL_USER=iamslash -e MYSQL_PASSWORD=1 -d mysql

$ docker exec -it my-mysql bash
# mysql -u root -p
```

다음과 같이 database, table 을 생성한다.

```sql
-- sessionA
create database foo;
use foo;
create table tab(
  k int primary key,
  v int not null
);
insert into tab values(1,1),(5,5),(10,10);

-- Check locks
SELECT * FROM performance_schema.data_locks;
-- not useful
SELECT * FROM performance_schema.metadata_locks where OBJECT_SCHEMA not in ('information_schema', 'performance_schema', 'mysql');
-- Check transactions
SELECT * FROM information_schema.innodb_trx;
-- not useful
show engine innodb status;
```

# Shared and Exclusive Locks

MySQL 은 2 가지 row-level lock 을 제공한다. 

* shared lock `(S)` 
  * shared lock 을 가지면 read 할 수 있다.
* exclusive lock `(X)`
  * exclusive lock 을 가지면 update, delete 할 수 있다.

transaction T1 이 row r 에 대해 `(S)` 를 가지고 있다고 해보자. transaction T2 는
row r 에 대해 다음과 같이 동작한다.

* T2 가 row r 에 대해 `(S)` 를 요청하면 바로 획득할 수 있다.
* T2 가 row r 에 대해 `(X)` 를 요청하면 바로 획득할 수 없다.

transaction T1 이 row r 에 대해 `(X)` 를 가지고 있다고 해보자. transaction T2 는
row r 에 대해 `(S)` 혹은 `(X)` 를 요청하면 바로 획득할 수 없다.

# Intention Locks

MySQL 은 2 가지 table-level lock 을 제공한다.  

* intention shared lock `(IS)`
  * 곧 row-level `(S)` lock 을 요청한다는 의미이다.
  * `SELECT ... LOCK IN SHARE MODE` 는 `(IS)` 를 요청한다.
* intention exclusive lock `(IX)`
  * 곧 row-level `(X)` lock 을 요청한다는 의미이다.
  * `SELECT ... LOCK FOR UPDATE` 는 `(IX)` 를 요청한다.

transaction 은 intention lock 에 대해 다음과 같이 동작한다.

* transaction 이 row r 에 대해 `(S)` 을 얻기 위해서는 먼저 그 table 에 대해
`(IS)` 혹은 `(IX)` 를 얻어야 한다.
* transaction 이 row r 에 대해 `(X)` 을 얻기 위해서는 먼저 그 table 에 대해
`(IX)` 를 얻어야 한다. 

다음은 intention lock 의 compatibility table 이다. 

| .    | `X`      | `IX`       | `S`        |       `IS` |
|------|----------|------------|------------|------------|
| `X`  | Conflict | Conflict   | Conflict   | Conflict   | 
| `IX` | Conflict | Compatible | Conflict   | Compatible | 
| `S`  | Conflict | Conflict   | Compatible | Compatible |
| `IS` | Conflict | Compatible | Compatible | Compatible |

Compatible 은 lock 을 획득할 수 있다. Conflict 가 발생하면 lock 이
release 될 때 까지 기다리고 획득한다.

intention lock 은 full table lock ([LOCK TABLES ...
WRITE](https://dev.mysql.com/doc/refman/5.7/en/lock-tables.html)) 을 제외하면
모두 Compatible 이다.

intention lock 은 곧 특정 row 에 대해 row-level lock 이 예정되어 있다는 것을
알려주는 것이 목적이다. 

# Record Locks

Record Lock 은 index record 에 걸리는 lock 을 말한다. index record 는 index
table 에서 data table 의 특정 row 를 가리키는 record 를 말한다. 

```
    Index table              Data table
-------------------          ---------
| id  | row addr  |          |   k   |
-------------------          ---------
|  1  | addr to 1 |--------->|   1   |
|  5  | addr to 5 |--------->|   5   |
-------------------          ---------
```

Record Lock 은 다음과 같이 2가지가 있다.

* record lock `(S)`
* record lock `(X)`

예를 들어 transaction t1 에서 `SELECT c1 FROM t WHERE c1 = 10 FOR UPDATE;` 를
실행하면 transaction t2 에서 `t.c1 = 10` 에 해당하는 row 에 대해 insert, update,
delete 을 수행할 수 없다.

```sql
> SHOW ENGINE INNODB STATUS;
RECORD LOCKS space id 58 page no 3 n bits 72 index `PRIMARY` of table `test`.`t`
trx id 10078 lock_mode X locks rec but not gap
Record lock, heap no 2 PHYSICAL RECORD: n_fields 3; compact format; info bits 0
 0: len 4; hex 8000000a; asc     ;;
 1: len 6; hex 00000000274f; asc     'O;;
 2: len 7; hex b60000019d0110; asc        ;;
```

다음은 `intention lock(IS), record lock(S)` 의 예이다.

```sql
-- session 1
begin;
select * from tab;
+----+----+
| k  | v  |
+----+----+
|  1 |  1 |
|  5 |  5 |
| 10 | 10 |
+----+----+
select * from tab where k=1 for share;

-- session 2
  SELECT EVENT_ID, OBJECT_SCHEMA, OBJECT_NAME, INDEX_NAME, LOCK_TYPE, 
         LOCK_MODE, LOCK_STATUS, LOCK_DATA 
    FROM performance_schema.data_locks 
ORDER BY EVENT_ID;
+----------+---------------+-------------+------------+-----------+---------------+-------------+-----------+
| EVENT_ID | OBJECT_SCHEMA | OBJECT_NAME | INDEX_NAME | LOCK_TYPE | LOCK_MODE     | LOCK_STATUS | LOCK_DATA |
+----------+---------------+-------------+------------+-----------+---------------+-------------+-----------+
|      221 | foo           | tab         | NULL       | TABLE     | IS            | GRANTED     | NULL      |
|      221 | foo           | tab         | PRIMARY    | RECORD    | S,REC_NOT_GAP | GRANTED     | 1         |
+----------+---------------+-------------+------------+-----------+---------------+-------------+-----------+
```

다음은 `intention lock(IX), record lock(X)` 의 예이다.

```sql
-- session 1
begin;
select * from tab;
+----+----+
| k  | v  |
+----+----+
|  1 |  1 |
|  5 |  5 |
| 10 | 10 |
+----+----+
select * from tab where k=1 for update;

-- session 2
  SELECT EVENT_ID, OBJECT_SCHEMA, OBJECT_NAME, INDEX_NAME, LOCK_TYPE, 
         LOCK_MODE, LOCK_STATUS, LOCK_DATA 
    FROM performance_schema.data_locks 
ORDER BY EVENT_ID;
+----------+---------------+-------------+------------+-----------+---------------+-------------+-----------+
| EVENT_ID | OBJECT_SCHEMA | OBJECT_NAME | INDEX_NAME | LOCK_TYPE | LOCK_MODE     | LOCK_STATUS | LOCK_DATA |
+----------+---------------+-------------+------------+-----------+---------------+-------------+-----------+
|      225 | foo           | tab         | NULL       | TABLE     | IX            | GRANTED     | NULL      |
|      225 | foo           | tab         | PRIMARY    | RECORD    | X,REC_NOT_GAP | GRANTED     | 1         |
+----------+---------------+-------------+------------+-----------+---------------+-------------+-----------+
```

# Gap Locks

Gap Lock 은 index record 들 사이에 걸리는 lock 이다. 즉, data table 에 없는 index 에 대해 걸리는 lock 이다. 

예를 들어 다음과 같이 index table, data table 이 있다고 해보자.

```
    Index table              Data table
-------------------          ---------
| id  | row addr  |          |  id   |
-------------------          ---------
|  3  | addr to 3 |--------->|   3   |
|  7  | addr to 7 |--------->|   7   |
-------------------          ---------
```

`id <= 2, 4 <= id <= 6, 8 <= id` 에 해당하는 index 는 record 가 없다. 이것이
바로 gab 을 의미한다. gap lock 은 이 gab 에 걸리는 lock 이다. gab 에는 index
record 가 없다. 따라서 gab lock 은 다른 transaction 이 새로운 record 를 삽입할
때 동시성을 제어할 수 있다.

예를 들어 transaction t1 에서 `SELECT c1 FROM t WHERE c1 BETWEEN 0 and 10 FOR
UPDATE;` 를 수행하면 transaction t2 는 `t.c1 = 15` 에 해댕하는 row 를 insert 할
수 없다. transaction t1 이 commit 혹은 roll back 을 수행하면 transaction t2 는
새로운 row 를 insert 할 수 있다.

[isolation level](/isolation/README.md) 이 read committed 이면 gap lock 이 비활성화 된다.

다음은 `gap lock(X)` 의 예이다.

```sql
-- session 1
begin;
select * from tab where k between 6 AND 9 for update;

-- session 2
begin;
  SELECT EVENT_ID, OBJECT_SCHEMA, OBJECT_NAME, INDEX_NAME, LOCK_TYPE, 
         LOCK_MODE, LOCK_STATUS, LOCK_DATA 
    FROM performance_schema.data_locks 
ORDER BY EVENT_ID;
+----------+---------------+-------------+------------+-----------+-----------+-------------+-----------+
| EVENT_ID | OBJECT_SCHEMA | OBJECT_NAME | INDEX_NAME | LOCK_TYPE | LOCK_MODE | LOCK_STATUS | LOCK_DATA |
+----------+---------------+-------------+------------+-----------+-----------+-------------+-----------+
|      252 | foo           | tab         | NULL       | TABLE     | IX        | GRANTED     | NULL      |
|      252 | foo           | tab         | PRIMARY    | RECORD    | X,GAP     | GRANTED     | 10        |
+----------+---------------+-------------+------------+-----------+-----------+-------------+-----------+
insert into tab values(6,6);
-- session 2 blocked

-- session 1
rollback;
-- session 2 unblocked

-- session 2
rollback;
```

# Next-Key Locks

Next-Key Lock 은 index record 에 대한 record lock 과 그 index record 의 이전
index record 들에 대한 gab lock 을 합한 것이다.

# Insert Intention Locks

Insert intention lock 은 gap lock 의 종류이다. `INSERT ...` 를 실행할 때 획득한다.
서로 다른 두 transaction 은 gap 에서 같은 위치의 record 를 삽입하지 않는다면 conflict 는
없다.

다음은 `insert intention lock(X)` 의 예이다.


```sql
-- session 1
begin;
select * from tab where k between 6 AND 9 for update;

-- session 2
begin;
  SELECT EVENT_ID, OBJECT_SCHEMA, OBJECT_NAME, INDEX_NAME, LOCK_TYPE, 
         LOCK_MODE, LOCK_STATUS, LOCK_DATA 
    FROM performance_schema.data_locks 
ORDER BY EVENT_ID;
+----------+---------------+-------------+------------+-----------+-----------+-------------+-----------+
| EVENT_ID | OBJECT_SCHEMA | OBJECT_NAME | INDEX_NAME | LOCK_TYPE | LOCK_MODE | LOCK_STATUS | LOCK_DATA |
+----------+---------------+-------------+------------+-----------+-----------+-------------+-----------+
|      252 | foo           | tab         | NULL       | TABLE     | IX        | GRANTED     | NULL      |
|      252 | foo           | tab         | PRIMARY    | RECORD    | X,GAP     | GRANTED     | 10        |
+----------+---------------+-------------+------------+-----------+-----------+-------------+-----------+
insert into tab values(6,6);
-- session 2 blocked

-- session 1
  SELECT EVENT_ID, OBJECT_SCHEMA, OBJECT_NAME, INDEX_NAME, LOCK_TYPE, 
         LOCK_MODE, LOCK_STATUS, LOCK_DATA 
    FROM performance_schema.data_locks 
ORDER BY EVENT_ID;
+----------+---------------+-------------+------------+-----------+------------------------+-------------+-----------+
| EVENT_ID | OBJECT_SCHEMA | OBJECT_NAME | INDEX_NAME | LOCK_TYPE | LOCK_MODE              | LOCK_STATUS | LOCK_DATA |
+----------+---------------+-------------+------------+-----------+------------------------+-------------+-----------+
|       53 | foo           | tab         | NULL       | TABLE     | IX                     | GRANTED     | NULL      |
|       53 | foo           | tab         | PRIMARY    | RECORD    | X,GAP,INSERT_INTENTION | WAITING     | 10        |
|      260 | foo           | tab         | NULL       | TABLE     | IX                     | GRANTED     | NULL      |
|      260 | foo           | tab         | PRIMARY    | RECORD    | X,GAP                  | GRANTED     | 10        |
+----------+---------------+-------------+------------+-----------+------------------------+-------------+-----------+
-- session 2 waiting
rollback;
-- session 2 unblocked

-- session 2
rollback;
```

# AUTO-INC Locks

An AUTO-INC lock is a special table-level lock taken by transactions inserting
into tables with AUTO_INCREMENT columns.

# Predicate Locks for Spatial Indexes

InnoDB supports SPATIAL indexing of columns containing spatial columns???

# Experiment

다음은 몇가지 실험을 한 것이다. 아직 이해가 가지 않는다.

```sql
-- session 1
begin;
select * from tab where v=5;
-- v is not a primary key

-- session 2
  SELECT EVENT_ID, OBJECT_SCHEMA, OBJECT_NAME, INDEX_NAME, LOCK_TYPE, 
         LOCK_MODE, LOCK_STATUS, LOCK_DATA 
    FROM performance_schema.data_locks 
ORDER BY EVENT_ID;
+----------+---------------+-------------+------------+-----------+-----------+-------------+------------------------+
| EVENT_ID | OBJECT_SCHEMA | OBJECT_NAME | INDEX_NAME | LOCK_TYPE | LOCK_MODE | LOCK_STATUS | LOCK_DATA              |
+----------+---------------+-------------+------------+-----------+-----------+-------------+------------------------+
|      271 | foo           | tab         | NULL       | TABLE     | IX        | GRANTED     | NULL                   |
|      271 | foo           | tab         | PRIMARY    | RECORD    | X         | GRANTED     | supremum pseudo-record |
|      271 | foo           | tab         | PRIMARY    | RECORD    | X         | GRANTED     | 1                      |
|      271 | foo           | tab         | PRIMARY    | RECORD    | X         | GRANTED     | 5                      |
|      271 | foo           | tab         | PRIMARY    | RECORD    | X         | GRANTED     | 10                     |
+----------+---------------+-------------+------------+-----------+-----------+-------------+------------------------+
-- Why all data records were locked???
-- supremum psedu-record means infinite range data records???
```

```sql
-- session 1
begin;
select * from tab where k between 1 and 10 for update;

-- session 2
begin;
  SELECT EVENT_ID, OBJECT_SCHEMA, OBJECT_NAME, INDEX_NAME, LOCK_TYPE, 
         LOCK_MODE, LOCK_STATUS, LOCK_DATA 
    FROM performance_schema.data_locks 
ORDER BY EVENT_ID;
+----------+---------------+-------------+------------+-----------+---------------+-------------+------------------------+
| EVENT_ID | OBJECT_SCHEMA | OBJECT_NAME | INDEX_NAME | LOCK_TYPE | LOCK_MODE     | LOCK_STATUS | LOCK_DATA              |
+----------+---------------+-------------+------------+-----------+---------------+-------------+------------------------+
|      277 | foo           | tab         | NULL       | TABLE     | IX            | GRANTED     | NULL                   |
|      277 | foo           | tab         | PRIMARY    | RECORD    | X,REC_NOT_GAP | GRANTED     | 1                      |
|      277 | foo           | tab         | PRIMARY    | RECORD    | X             | GRANTED     | supremum pseudo-record |
|      277 | foo           | tab         | PRIMARY    | RECORD    | X             | GRANTED     | 5                      |
|      277 | foo           | tab         | PRIMARY    | RECORD    | X             | GRANTED     | 10                     |
+----------+---------------+-------------+------------+-----------+---------------+-------------+------------------------+
-- supremum psudo-record means [2..4], [6..9] ???
insert into tab values(6,6);
-- session 2 blocked
-- supremum psudo-record means gap block ???

-- session 1
  SELECT EVENT_ID, OBJECT_SCHEMA, OBJECT_NAME, INDEX_NAME, LOCK_TYPE, 
         LOCK_MODE, LOCK_STATUS, LOCK_DATA 
    FROM performance_schema.data_locks 
ORDER BY EVENT_ID;
+----------+---------------+-------------+------------+-----------+------------------------+-------------+------------------------+
| EVENT_ID | OBJECT_SCHEMA | OBJECT_NAME | INDEX_NAME | LOCK_TYPE | LOCK_MODE              | LOCK_STATUS | LOCK_DATA              |
+----------+---------------+-------------+------------+-----------+------------------------+-------------+------------------------+
|       61 | foo           | tab         | NULL       | TABLE     | IX                     | GRANTED     | NULL                   |
|       61 | foo           | tab         | PRIMARY    | RECORD    | X,GAP,INSERT_INTENTION | WAITING     | 10                     |
|      277 | foo           | tab         | NULL       | TABLE     | IX                     | GRANTED     | NULL                   |
|      277 | foo           | tab         | PRIMARY    | RECORD    | X,REC_NOT_GAP          | GRANTED     | 1                      |
|      277 | foo           | tab         | PRIMARY    | RECORD    | X                      | GRANTED     | supremum pseudo-record |
|      277 | foo           | tab         | PRIMARY    | RECORD    | X                      | GRANTED     | 5                      |
|      277 | foo           | tab         | PRIMARY    | RECORD    | X                      | GRANTED     | 10                     |
+----------+---------------+-------------+------------+-----------+------------------------+-------------+------------------------+
rollback;
-- session 2 unblocked;

-- session 2
rollback;
```

```sql
-- session 1
begin;
select * from tab where k>=20 for update;

-- session 2
  SELECT EVENT_ID, OBJECT_SCHEMA, OBJECT_NAME, INDEX_NAME, LOCK_TYPE, 
         LOCK_MODE, LOCK_STATUS, LOCK_DATA 
    FROM performance_schema.data_locks 
ORDER BY EVENT_ID;
+----------+---------------+-------------+------------+-----------+-----------+-------------+------------------------+
| EVENT_ID | OBJECT_SCHEMA | OBJECT_NAME | INDEX_NAME | LOCK_TYPE | LOCK_MODE | LOCK_STATUS | LOCK_DATA              |
+----------+---------------+-------------+------------+-----------+-----------+-------------+------------------------+
|      283 | foo           | tab         | NULL       | TABLE     | IX        | GRANTED     | NULL                   |
|      283 | foo           | tab         | PRIMARY    | RECORD    | X         | GRANTED     | supremum pseudo-record |
+----------+---------------+-------------+------------+-----------+-----------+-------------+------------------------+
-- supremum pseudo-record means what???
```

# Inno-db Deadlock

> * [MySQL InnoDB lock & deadlock 이해하기](https://www.letmecompile.com/mysql-innodb-lock-deadlock/)
> * [14.7.5.1 An InnoDB Deadlock Example @ mysql](https://dev.mysql.com/doc/refman/5.7/en/innodb-deadlock-example.html)

mysql 의 innodb 는 Deadlock 을 detect 할 수 있다. 만약 mysql 이 Deadlock 을
detect 하면 어느 한 transaction 의 lock wait 을 중지하여 Deadlock 을 해결한다.
즉, 바로 error 를 리턴한다.

[Deadlock](/isolation/README.md#consistent-read)

# MySQL Optimistic Locking

* [Optimistic locking in MySQL](https://stackoverflow.com/questions/17431338/optimistic-locking-in-mysql)

-----

MySQL 로 Optimistic Locking 을 다음과 같이 구현할 수 있다.

다음과 같이 table 을 만들자.

```sql
CREATE TABLE theTable(
    iD int NOT NULL,
    val1 int NOT NULL,
    val2 int NOT NULL
);
INSERT INTO theTable (iD, val1, val2) VALUES (1, 2 ,3);
```

다음과 같은 query 를 수행하고 싶다.

```sql
SELECT iD, val1, val2
  FROM theTable
 WHERE iD = @theId;
{code that calculates new values}
UPDATE theTable
   SET val1 = @newVal1,
       val2 = @newVal2
 WHERE iD = @theId;
{go on with your other code}
```

동시성을 옮바르게 처리하기 위해 다음과 같이 Optimistic Locking 을 구현한다.

```sql
SELECT iD, val1, val2
  FROM theTable
 WHERE iD = @theId;
{code that calculates new values}
UPDATE theTable
   SET val1 = @newVal1,
       val2 = @newVal2
 WHERE iD = @theId
   AND val1 = @oldVal1
   AND val2 = @oldVal2;
{if AffectedRows == 1 }
  {go on with your other code}
{else}
  {decide what to do since it has gone bad... in your code}
{endif}
```

트랜잭션을 사용해서 동시성을 제어할 수도 있다. `UPDATE ... WHERE` 에서 `WHERE` 에 field 들을 추가했다.  
[Isolation Level](/isolation/README.md#solution-of-non-repeatable-read-in-repeatable-read-isolation-level) 참고.

```sql
SELECT iD, val1, val2
  FROM theTable
 WHERE iD = @theId;
{code that calculates new values}
BEGIN TRANSACTION;
UPDATE anotherTable
   SET col1 = @newCol1,
       col2 = @newCol2
 WHERE iD = @theId;
UPDATE theTable
   SET val1 = @newVal1,
       val2 = @newVal2
 WHERE iD = @theId
   AND val1 = @oldVal1
   AND val2 = @oldVal2;
{if AffectedRows == 1 }
  COMMIT TRANSACTION;
  {go on with your other code}
{else}
  ROLLBACK TRANSACTION;
  {decide what to do since it has gone bad... in your code}
{endif}
```

version 을 이용한 Optimistic Locking 이다.

```sql
SELECT iD, val1, val2, version
  FROM theTable
 WHERE iD = @theId;
{code that calculates new values}
UPDATE theTable
   SET val1 = @newVal1,
       val2 = @newVal2,
       version = version + 1
 WHERE iD = @theId
   AND version = @oldversion;
{if AffectedRows == 1 }
  {go on with your other code}
{else}
  {decide what to do since it has gone bad... in your code}
{endif}
```
