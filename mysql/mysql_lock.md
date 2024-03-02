- [Abstract](#abstract)
- [Materials](#materials)
- [MySQL Lock](#mysql-lock)
- [Shared and Exclusive Locks](#shared-and-exclusive-locks)
- [Intention Locks](#intention-locks)
- [Record Locks](#record-locks)
- [Gap Locks](#gap-locks)
- [Next-Key Locks](#next-key-locks)
- [Gap Locks vs Next-Key Locks](#gap-locks-vs-next-key-locks)
- [infimum pseudo-record, supremum pseudo-record](#infimum-pseudo-record-supremum-pseudo-record)
- [Insert Intention Locks](#insert-intention-locks)
- [AUTO-INC Locks](#auto-inc-locks)
- [Predicate Locks for Spatial Indexes](#predicate-locks-for-spatial-indexes)
- [Inno-db Deadlock](#inno-db-deadlock)
- [Repeatable Read vs Serializable Isolation Level](#repeatable-read-vs-serializable-isolation-level)
- [MySQL Optimistic Locking](#mysql-optimistic-locking)

----

# Abstract

MySQL InnoDB 의 Lock 에 대해 정리한다. MySQL 은 Lock 으로 [Concurrency Problems](/database/README.md#concurrency-problems-in-transactions) 을 해결한다. 즉, [Isolation Level](/isolation/README.md) 을 Lock 으로 구현한다.

다음은 [MySQL InnoDB](/mysql/README.md) 의 isolation level 과 lock 활성화의 관계이다.

| Isolation Level | Record Lock | Gap Lock | Next-Key Lock | Phantom Reads Prevention |
|---|---|---|---|---|
| READ UNCOMMITTED | Yes | No | No | No |
| READ COMMITTED   | Yes | No*| No | No |
| REPEATABLE READ  | Yes | Yes| Yes| Yes|
| SERIALIZABLE     | Yes | Yes| Yes| Yes|

# Materials

* [Lock으로 이해하는 Transaction의 Isolation Level](https://suhwan.dev/2019/06/09/transaction-isolation-level-and-lock/)
* [MySQL InnoDB lock & deadlock 이해하기](https://www.letmecompile.com/mysql-innodb-lock-deadlock/)
* [14.7.1 InnoDB Locking @ mysql](https://dev.mysql.com/doc/refman/5.7/en/innodb-locking.html)
* [InnoDB locking](https://github.com/octachrome/innodb-locks)
* [20-MySQL의 잠금 | MySQL DBA 튜토리얼 | MySQL 8 DBA 튜토리얼 | youtube](https://www.youtube.com/watch?v=8NlElO5-Xbk)
* [MySQL Gap Lock 다시보기](https://medium.com/daangn/mysql-gap-lock-%EB%8B%A4%EC%8B%9C%EB%B3%B4%EA%B8%B0-7f47ea3f68bc)

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

MySQL 은 2 가지 **row-level lock** 을 제공한다. 

* shared lock `(S)` 
  * shared lock 을 가지면 `read` 할 수 있다.
* exclusive lock `(X)`
  * exclusive lock 을 가지면 `update, delete` 할 수 있다.

transaction T1 이 row r 에 대해 `(S)` 를 가지고 있다고 해보자. transaction T2 는 row r 에 대해 다음과 같이 동작한다.

* T2 가 row r 에 대해 `(S)` 를 요청하면 바로 획득할 수 **있다**.
* T2 가 row r 에 대해 `(X)` 를 요청하면 바로 획득할 수 **없다**.

transaction T1 이 row r 에 대해 `(X)` 를 가지고 있다고 해보자. transaction T2 는 row r 에 대해 다음과 같이 동작한다.

* T2 가 row r 에 대해 `(S)` 를 요청하면 바로 획득할 수 **없다**.
* T2 가 row r 에 대해 `(X)` 를 요청하면 바로 획득할 수 **없다**.

sql example 은 [Record Lock](#record-locks) 을 참고한다.
  
# Intention Locks

MySQL 은 2 가지 **table-level lock** 을 제공한다.  

* intention shared lock `(IS)`
  * 곧 row-level `(S)` lock 을 요청한다는 의미이다.
  * `SELECT ... LOCK IN SHARE MODE` 는 `(IS)` 를 요청한다.
* intention exclusive lock `(IX)`
  * 곧 row-level `(X)` lock 을 요청한다는 의미이다.
  * `SELECT ... LOCK FOR UPDATE` 는 `(IX)` 를 요청한다.

transaction 은 intention lock 에 대해 다음과 같이 동작한다.

* transaction 이 row r 에 대해 `(S)` 을 얻기 위해서는 먼저 그 table 에 대해 `(IS)` 혹은 `(IX)` 를 얻어야 한다.
* transaction 이 row r 에 대해 `(X)` 을 얻기 위해서는 먼저 그 table 에 대해 `(IX)` 를 얻어야 한다. 

다음은 intention lock 의 compatibility table 이다. 

| .    | `X`      | `IX`       | `S`        |       `IS` |
|------|----------|------------|------------|------------|
| `X`  | Conflict | Conflict   | Conflict   | Conflict   | 
| `IX` | Conflict | Compatible | Conflict   | Compatible | 
| `S`  | Conflict | Conflict   | Compatible | Compatible |
| `IS` | Conflict | Compatible | Compatible | Compatible |

Compatible 은 lock 을 획득할 수 있다. Conflict 가 발생하면 lock 이 release 될 때 까지 기다리고 획득한다.

intention lock 은 full table lock ([LOCK TABLES ... WRITE](https://dev.mysql.com/doc/refman/5.7/en/lock-tables.html)) 을 제외하면 모두 Compatible 이다.

intention lock 은 곧 특정 row 에 대해 row-level lock 이 예정되어 있다는 것을 알려주는 것이 목적이다. 

sql example 은 [Record Lock](#record-locks) 을 참고한다.

# Record Locks

Record Lock 은 index record 에 걸리는 lock 을 말한다. index record 는 index table 에서 data table 의 특정 row 를 가리키는 record 를 말한다. 

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

* `record lock (S)`
* `record lock (X)`

예를 들어 transaction t1 에서 `SELECT c1 FROM t WHERE c1 = 10 FOR UPDATE;` 를 실행하면 transaction t2 에서 `t.c1 = 10` 에 해당하는 row 에 대해 `insert, update, delete` 을 수행할 수 없다.

다음은 `intention lock(IS), record lock(S)` 의 예이다.

```sql
-- session 1
create database foo;
use foo;
create table tab(
  k int primary key,
  v int not null
);
insert into tab values(1,1),(5,5),(10,10);

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

Gap lock은 인덱스 레코드들 사이의 공간, 즉 gap에 걸리는 락입니다. 데이터 테이블에 존재하지 않는 인덱스 값에 대해 설정됩니다. 이 락은 주로 phantom read를 방지하기 위해 사용되며, 다른 트랜잭션이 gap 영역에 새로운 레코드를 삽입하는 것을 제어합니다. 예를 들어, `SELECT ... FOR UPDATE`로 특정 범위를 조회하면 해당 범위 내에서 새로운 레코드의 삽입을 방지하기 위해 gap lock이 사용됩니다. 이는 레코드가 아직 존재하지 않는 영역에 대한 락이기 때문에, 특정 레코드에 대한 직접적인 락이 아닙니다.

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

`id <= 2, 4 <= id <= 6, 8 <= id` 에 해당하는 index 는 record 가 없다. 이것이 바로 gap 을 의미한다. gap lock 은 이 gap 에 걸리는 lock 이다. gap 에는 index record 가 없다. 따라서 gap lock 은 다른 transaction 이 새로운 record 를 삽입할 때 동시성을 제어할 수 있다.

예를 들어 transaction t1 에서 `SELECT c1 FROM t WHERE c1 BETWEEN 0 and 10 FOR UPDATE;` 를 수행하면 transaction t2 는 `t.c1 = 15` 에 해댕하는 row 를 insert 할 수 없다. transaction t1 이 commit 혹은 roll back 을 수행하면 transaction t2 는 새로운 row 를 insert 할 수 있다.

참고로 [isolation level](/isolation/README.md) 이 `read committed` 이면 gap lock 이 비활성화 된다.

다음은 `gap lock(X)` for `select ... for udpate` 의 예이다.

```sql
-- session 1
create database foo;
use foo;
create table tab(
  k int primary key,
  v int not null
);
insert into tab values(1,1),(5,5),(10,10);

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

다음은 `gap lock(X)` for `update` 의 예이다.

```sql
-- session 1
begin;
update tab set v = v + 10 where k=2;

-- session 2
begin;
  SELECT EVENT_ID, OBJECT_SCHEMA, OBJECT_NAME, INDEX_NAME, LOCK_TYPE, 
         LOCK_MODE, LOCK_STATUS, LOCK_DATA 
    FROM performance_schema.data_locks 
ORDER BY EVENT_ID;
+----------+---------------+-------------+------------+-----------+-----------+-------------+-----------+
| EVENT_ID | OBJECT_SCHEMA | OBJECT_NAME | INDEX_NAME | LOCK_TYPE | LOCK_MODE | LOCK_STATUS | LOCK_DATA |
+----------+---------------+-------------+------------+-----------+-----------+-------------+-----------+
|      317 | foo           | tab         | NULL       | TABLE     | IX        | GRANTED     | NULL      |
|      317 | foo           | tab         | PRIMARY    | RECORD    | X,GAP     | GRANTED     | 5         |
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

Next-key lock은 gap lock과 record lock의 결합입니다. 즉, 특정 인덱스 레코드에 대한 락(Record Lock)과 그 인덱스 레코드의 이전 인덱스 레코드들에 대한 gap lock을 합친 것입니다. 이 락은 특정 레코드와 그 이전의 gap에 대해서도 락을 걸어, 동시에 여러 트랜잭션이 같은 데이터에 접근하는 것을 제어하고, phantom read를 방지합니다. 예를 들어, `SELECT ... FOR UPDATE`를 사용해 특정 레코드를 조회하면, 해당 레코드뿐만 아니라 그 레코드 이전의 gap에도 락이 걸립니다.

```sql
-- session 1
create database foo;
use foo;
create table tab(
  k int primary key,
  v int not null
);
insert into tab values(1,1),(5,5),(10,10);

-- session 1
set session transaction isolation level repeatable read;
begin;
select * from tab where k between 2 and 9 for update;
+---+---+
| k | v |
+---+---+
| 5 | 5 |
+---+---+
  SELECT EVENT_ID, OBJECT_SCHEMA, OBJECT_NAME, INDEX_NAME, LOCK_TYPE, 
         LOCK_MODE, LOCK_STATUS, LOCK_DATA 
    FROM performance_schema.data_locks 
ORDER BY EVENT_ID;
+----------+---------------+-------------+------------+-----------+-----------+-------------+-----------+
| EVENT_ID | OBJECT_SCHEMA | OBJECT_NAME | INDEX_NAME | LOCK_TYPE | LOCK_MODE | LOCK_STATUS | LOCK_DATA |
+----------+---------------+-------------+------------+-----------+-----------+-------------+-----------+
|       62 | foo           | tab         | NULL       | TABLE     | IX        | GRANTED     | NULL      |
|       62 | foo           | tab         | PRIMARY    | RECORD    | X         | GRANTED     | 5         |
|       62 | foo           | tab         | PRIMARY    | RECORD    | X,GAP     | GRANTED     | 10        |
+----------+---------------+-------------+------------+-----------+-----------+-------------+-----------+

-- session 2
set session transaction isolation level repeatable read;
begin;
insert into tab values(2,2);
-- blocked
```

# Gap Locks vs Next-Key Locks

- 용도의 차이: Gap lock은 주로 phantom read 방지를 위해 인덱스의 빈 공간(gap)에 적용되며, next-key lock은 특정 레코드와 그 이전의 gap 모두에 적용되어 보다 광범위한 방지 기능을 제공합니다.
- 적용 범위의 차이: Gap lock은 특정 레코드가 아닌, 레코드들 사이의 공간에 적용됩니다. 반면, next-key lock은 특정 레코드와 해당 레코드 이전의 gap에 모두 적용되어, 레코드 자체와 그 앞의 공간에 대한 락을 동시에 제공합니다.

즉, gap lock은 데이터 삽입에 대한 동시성 제어에 초점을 맞추며, next-key lock은 데이터의 읽기 및 삽입에 대한 더 광범위한 제어를 가능하게 합니다. 

# infimum pseudo-record, supremum pseudo-record

`supremum pseudo-record`는 next-key lock과 직접적인 연관이 있지는 않지만, InnoDB 스토리지 엔진에서 사용하는 내부 메커니즘의 일부입니다. InnoDB에서 각 인덱스는 두 개의 특수한 레코드를 가지고 있습니다: `infimum`과 `supremum pseudo-records`입니다. 이들은 각각 인덱스의 최소값과 최대값을 대표하는 가상의 레코드로, 실제 데이터 레코드가 아니라 인덱스의 경계를 나타냅니다.

- Infimum pseudo-record: 인덱스의 최소 가능한 값을 대표합니다. 이는 인덱스 내에서 가장 작은 값을 가진 레코드보다도 작은 가상의 값입니다.
- Supremum pseudo-record: 인덱스의 최대 가능한 값을 대표합니다. 이는 인덱스 내에서 가장 큰 값을 가진 레코드보다도 큰 가상의 값으로, 인덱스의 상한을 나타냅니다.

**Supremum Pseudo-record와 Next-key Locks**

`supremum pseudo-record`에 대한 배타적 락(LOCK_MODE가 X)이 보여주는 것은, 해당 세션에서 인덱스의 최대 경계에 대한 락을 획득했다는 것을 의미합니다. 이것은 직접적으로 next-key lock이라고 할 수는 없으나, next-key locks의 작동 방식을 이해하는 데 도움이 됩니다. Next-key lock은 특정 레코드와 그 레코드의 앞에 있는 간격(gap)에 대한 락을 포함합니다. 따라서, 인덱스의 마지막 레코드와 supremum pseudo-record 사이의 gap에 대한 락을 설정하는 것과 유사한 개념이 될 수 있습니다.

다음은 `supremum pseudo-record` 의 예이다. 

```sql
-- session 1
create database foo;
use foo;
create table tab(
  k int primary key,
  v int not null
);
insert into tab values(1,1),(5,5),(10,10);

-- session 1
begin;
select * from tab where v = 5 for update;
-- v is not a primary key
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
```

**실험 결과 해석**

실험 결과에서 `v = 5`에 대해 `FOR UPDATE`를 수행한 후, 모든 레코드`(1, 5, 10)`에 대해 배타적 락(LOCK_MODE가 X)이 설정된 것을 볼 수 있습니다. 이는 `v`가 기본 키가 아님에도 불구하고, InnoDB가 내부적으로 효율적인 락 관리를 위해 전체 테이블을 스캔하며 해당 조건에 일치하는 모든 레코드에 대한 배타적 락을 설정했음을 나타냅니다. `supremum pseudo-record`에 대한 락이 설정된 것은, 해당 쿼리가 인덱스의 끝까지 영향을 미쳤음을 보여줍니다.

결론적으로, supremum pseudo-record에 대한 락은 next-key lock의 직접적인 예시는 아니지만, InnoDB가 인덱스와 관련된 동작을 처리하는 방식의 일부입니다. 이러한 내부 메커니즘의 이해는 데이터베이스의 락 동작을 예측하고 최적화하는 데 도움이 될 수 있습니다.

# Insert Intention Locks

Insert intention lock 은 gap lock 의 종류이다. `INSERT ...` 를 실행할 때 획득한다. 서로 다른 두 transaction 은 gap 에서 같은 위치의 record 를 삽입하지 않는다면 conflict 는 없다.

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
into tables with `AUTO_INCREMENT` columns.

# Predicate Locks for Spatial Indexes

InnoDB supports `SPATIAL` indexing of columns containing spatial columns???

# Inno-db Deadlock

> * [MySQL InnoDB lock & deadlock 이해하기](https://www.letmecompile.com/mysql-innodb-lock-deadlock/)
> * [14.7.5.1 An InnoDB Deadlock Example @ mysql](https://dev.mysql.com/doc/refman/5.7/en/innodb-deadlock-example.html)

mysql 의 innodb 는 Deadlock 을 detect 할 수 있다. 만약 mysql 이 Deadlock 을 detect 하면 어느 한 transaction 의 lock wait 을 중지하여 Deadlock 을 해결한다. 즉, 바로 error 를 리턴한다. Error 발생하면 User 다시 시도하게 한다.

[Deadlock](/isolation/README.md#consistent-read)

# Repeatable Read vs Serializable Isolation Level

`REPEATABLE READ`와 `SERIALIZABLE` 격리 수준 모두에서 잠금(`Record Lock, Gap Lock, Next-Key Lock`)이 활성화되어 있음에도 불구하고, `SERIALIZABLE` 격리 수준을 선택해야 하는 경우가 있습니다. 이 두 격리 수준 사이의 주요 차이점은 `SERIALIZABLE` 격리 수준에서 모든 `SELECT` 쿼리에 대해 임시적인 읽기 잠금이 걸린다는 점입니다. 이러한 차이점은 동시성과 데이터 일관성 사이의 균형을 어떻게 맞출지에 대한 데이터베이스의 접근 방식에서 비롯됩니다.

다음은 isolation level 이 `REPEATABLE READ` 인 경우 예이다. blocking 이 없다.

```sql
-- Transaction 1
create database foo;
use foo;
create table tab(
  k int primary key,
  v int not null
);
insert into tab values(1,1),(5,5),(10,10);

-- Transaction 1
use foo;
set session transaction isolation level repeatable read;
START TRANSACTION;
SELECT * FROM tab;
+----+----+
| k  | v  |
+----+----+
|  1 |  1 |
|  5 |  5 |
| 10 | 10 |
+----+----+
  SELECT EVENT_ID, OBJECT_SCHEMA, OBJECT_NAME, INDEX_NAME, LOCK_TYPE, 
         LOCK_MODE, LOCK_STATUS, LOCK_DATA 
    FROM performance_schema.data_locks 
ORDER BY EVENT_ID;
-- Empty set

-- 다른 트랜잭션에서 orders 테이블에 새로운 주문을 삽입
-- Transaction 2
use foo;
set session transaction isolation level repeatable read;
START TRANSACTION;
insert into tab values(2,2);
-- Query OK, 1 row affected
COMMIT;

-- Transaction 1
SELECT * FROM tab;
+----+----+
| k  | v  |
+----+----+
|  1 |  1 |
|  2 |  2 |
|  5 |  5 |
| 10 | 10 |
+----+----+
COMMIT;
```

다음은 isolation level 이 `SERIALIZABLE` 인 경우 예이다. blocking 이 있다.

```sql
-- Transaction 1
create database foo;
use foo;
create table tab(
  k int primary key,
  v int not null
);
insert into tab values(1,1),(5,5),(10,10);

-- Transaction 1
use foo;
set session transaction isolation level serializable;
START TRANSACTION;
SELECT * FROM tab;
+----+----+
| k  | v  |
+----+----+
|  1 |  1 |
|  2 |  2 |
|  5 |  5 |
| 10 | 10 |
+----+----+
  SELECT EVENT_ID, OBJECT_SCHEMA, OBJECT_NAME, INDEX_NAME, LOCK_TYPE, 
         LOCK_MODE, LOCK_STATUS, LOCK_DATA 
    FROM performance_schema.data_locks 
ORDER BY EVENT_ID;
+----------+---------------+-------------+------------+-----------+-----------+-------------+------------------------+
| EVENT_ID | OBJECT_SCHEMA | OBJECT_NAME | INDEX_NAME | LOCK_TYPE | LOCK_MODE | LOCK_STATUS | LOCK_DATA              |
+----------+---------------+-------------+------------+-----------+-----------+-------------+------------------------+
|       91 | foo           | tab         | NULL       | TABLE     | IS        | GRANTED     | NULL                   |
|       91 | foo           | tab         | PRIMARY    | RECORD    | S         | GRANTED     | supremum pseudo-record |
|       91 | foo           | tab         | PRIMARY    | RECORD    | S         | GRANTED     | 1                      |
|       91 | foo           | tab         | PRIMARY    | RECORD    | S         | GRANTED     | 5                      |
|       91 | foo           | tab         | PRIMARY    | RECORD    | S         | GRANTED     | 10                     |
|       91 | foo           | tab         | PRIMARY    | RECORD    | S         | GRANTED     | 2                      |
+----------+---------------+-------------+------------+-----------+-----------+-------------+------------------------+

-- 다른 트랜잭션에서 tab 테이블에 새로운 주문을 삽입
-- Transaction 2
use foo;
set session transaction isolation level serializable;
START TRANSACTION;
insert into tab values(3,3);
-- blocked

-- Transaction 1
SELECT * FROM tab;
+----+----+
| k  | v  |
+----+----+
|  1 |  1 |
|  5 |  5 |
| 10 | 10 |
+----+----+
COMMIT;
-- unblocked Transaction 2
  SELECT EVENT_ID, OBJECT_SCHEMA, OBJECT_NAME, INDEX_NAME, LOCK_TYPE, 
         LOCK_MODE, LOCK_STATUS, LOCK_DATA 
    FROM performance_schema.data_locks 
ORDER BY EVENT_ID;
+----------+---------------+-------------+------------+-----------+------------------------+-------------+-----------+
| EVENT_ID | OBJECT_SCHEMA | OBJECT_NAME | INDEX_NAME | LOCK_TYPE | LOCK_MODE              | LOCK_STATUS | LOCK_DATA |
+----------+---------------+-------------+------------+-----------+------------------------+-------------+-----------+
|       27 | foo           | tab         | NULL       | TABLE     | IX                     | GRANTED     | NULL      |
|       27 | foo           | tab         | PRIMARY    | RECORD    | X,GAP,INSERT_INTENTION | GRANTED     | 5         |
+----------+---------------+-------------+------------+-----------+------------------------+-------------+-----------+

-- Transaction 2
  SELECT EVENT_ID, OBJECT_SCHEMA, OBJECT_NAME, INDEX_NAME, LOCK_TYPE, 
         LOCK_MODE, LOCK_STATUS, LOCK_DATA 
    FROM performance_schema.data_locks 
ORDER BY EVENT_ID;
+----------+---------------+-------------+------------+-----------+------------------------+-------------+-----------+
| EVENT_ID | OBJECT_SCHEMA | OBJECT_NAME | INDEX_NAME | LOCK_TYPE | LOCK_MODE              | LOCK_STATUS | LOCK_DATA |
+----------+---------------+-------------+------------+-----------+------------------------+-------------+-----------+
|       27 | foo           | tab         | NULL       | TABLE     | IX                     | GRANTED     | NULL      |
|       27 | foo           | tab         | PRIMARY    | RECORD    | X,GAP,INSERT_INTENTION | GRANTED     | 5         |
+----------+---------------+-------------+------------+-----------+------------------------+-------------+-----------+
COMMIT;
  SELECT EVENT_ID, OBJECT_SCHEMA, OBJECT_NAME, INDEX_NAME, LOCK_TYPE, 
         LOCK_MODE, LOCK_STATUS, LOCK_DATA 
    FROM performance_schema.data_locks 
ORDER BY EVENT_ID;
-- Empty set
```

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

트랜잭션을 사용해서 동시성을 제어할 수도 있다. `UPDATE ... WHERE` 에서 `WHERE` 에 field 들을 추가했다. [Isolation Level](/isolation/README.md#solution-of-non-repeatable-read-in-repeatable-read-isolation-level) 참고.

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
