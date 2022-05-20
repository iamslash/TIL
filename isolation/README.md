- [Abstract](#abstract)
- [Materials](#materials)
- [Launch mysql docker container](#launch-mysql-docker-container)
- [Create Table](#create-table)
- [Consistent Read](#consistent-read)
- [Solution of Non-repeatable read in repeatable read isolation level](#solution-of-non-repeatable-read-in-repeatable-read-isolation-level)
- [Practice of Read Uncommitted](#practice-of-read-uncommitted)
- [Practice of Read Committed](#practice-of-read-committed)
- [Practice of Repeatable Read](#practice-of-repeatable-read)
- [Practice of Searializable](#practice-of-searializable)
- [Practice of "SELECT ... FOR UPDATE"](#practice-of-select--for-update)

----

# Abstract

[mysql](/mysql/README.md) 은 [Lock](/mysql/mysql_lock.md) 이용하여
동시성을 제어한다. Isolation Level 을 달리하여 [Concurrency Problems](/database/README.md#concurrency-problems-in-transactions) 을 해결할 수 있다. Isolation Level 이 높을 수록
System throughput 은 낮아진다. 보통 Isolation Level 을 read committed 으로 설정한다.

Isolation Level 의 동작방식을 MySQL Lock 으로 이해하고 싶다.
[mysql](/mysql/README.md) 로 실습해 본다.

# Materials

> * [15.7.2.1 Transaction Isolation Levels | mysql](https://dev.mysql.com/doc/refman/8.0/en/innodb-transaction-isolation-levels.html)
> * [Mysql innoDB Lock, isolation level과 Lock 경쟁](https://taes-k.github.io/2020/05/17/mysql-transaction-lock/)
> * [Optimistic locking in MySQL | stackoverflow](https://stackoverflow.com/questions/17431338/optimistic-locking-in-mysql)
> * [SQL 트랜잭션 - 믿는 도끼에 발등 찍힌다](https://blog.sapzil.org/2017/04/01/do-not-trust-sql-transaction/)
>   * repeatable read 로 막을 수 없는 문제에 대해 설명한다. 
> * [MySQL의 Transaction Isolation Levels](https://jupiny.com/2018/11/30/mysql-transaction-isolation-levels/)

# Launch mysql docker container

```bash
$ docker run -p 3306:3306 --rm --name my-mysql -e MYSQL_ROOT_PASSWORD=1 -e MYSQL_DATABASE=hello -e MYSQL_USER=iamslash -e MYSQL_PASSWORD=1 -d mysql

$ docker ps
$ docker exec -it my-mysql /bin/bash

$ mysql -u iamslash -p
mysql> show databases
mysql> use hello
```

```bash
$ vim docker-compose.yml
version: "3.9"
services:
  mysql_0:
    image: mysql
    command: --character-set-server=utf8mb4
    restart: always
    mem_limit: 512m
    ports:
      - 3306:3306
    environment:
      MYSQL_ROOT_PASSWORD: 1234
$ docker-compose up
```

# Create Table

```sql
CREATE TABLE user (
  id int,
  name varchar(255)
);
insert into user (id, name) values (1, "foo");
insert into user (id, name) values (2, "bar");
insert into user (id, name) values (3, "baz");

-- check isolation level
> SHOW VARIABLES WHERE VARIABLE_NAME='transaction_isolation';
+-----------------------+-----------------+
| Variable_name         | Value           |
+-----------------------+-----------------+
| transaction_isolation | REPEATABLE-READ |
+-----------------------+-----------------+
-- check global transaction level (mysql8+)
> SELECT @@transaction_ISOLATION;
-- check session transaction level (mysql8+)
> SELECT @@global.transaction_ISOLATION;
```

# Consistent Read

* [Lock으로 이해하는 Transaction의 Isolation Level](https://suhwan.dev/2019/06/09/transaction-isolation-level-and-lock/)
* [consistent read | mysql](https://dev.mysql.com/doc/refman/8.0/en/glossary.html#glos_consistent_read)
* [15.7.2.3 Consistent Nonlocking Reads | mysql](https://dev.mysql.com/doc/refman/8.0/en/innodb-consistent-read.html)

----

Consistent Read 는 특정 시점의 snapshot 에서 data 를 읽는 것을 말한다. MySQL 은
lock 을 걸고 특정 시점의 snapshot 에서 data 를 읽지는 않는다. 이 것은 system
throughput 을 낮추기 때문이다.

특정 시점의 snapshot 에서 data 를 읽는다는 것은 무엇을 말하는 걸까?. MySQL 은
commit log 를 기록한다. snapshot 은 특정 시점까지의 commit log 모음을 말한다.

[Isolation Level](/isolation/README.md) 이 **read uncommitted** 인 경우를 생각해
보자. transaction 1 은 read 할 때 다른 transaction 이 commit 하지 않는 data 도
읽어온다. **dirty read** 가 발생한다. 물론 **non-repeatable read, phantom read** 도
발생한다.

[Isolation Level](/isolation/README.md) 이 **read committed** 인 경우를 생각해 보자.
transaction 1 에서 read 할 때 마다 snapshot 을 기록해 둔다. transaction 2
에서 특정 row 를 변경하고 commit 했다면 transaction 1 에서 다시 read 할 때 
새로운 snaptshot 에서 data 를 읽기 때문에 **non-repeatable read** 가 발생한다.

또한 [Isolation Level](/isolation/README.md) 이 **read committed** 인 경우는
gab lobck 을 사용하지 않는다. 따라서 **phantom read** 가 발생한다.

[Isolation Level](/isolation/README.md) 이 **repeatable read** 인 경우를 생각해
보자. 처음 read 한 때의 snapshot 을 기록해 둔다. 이후 read 할 때는 처음 read 한
때의 snapshot 에서 data 를 읽어온다. 따라서 **non-repeatable read** 가 발생하지 않는다.
그러나 **phantom read** 는 여전히 발생한다.

[Isolation Level](/isolation/README.md) 이 **serializable** 인 경우를 생각해
보자. 기본적으로 **repeatble read** 와 같다. 단, `SELECT ...` 가 `SELECT ... FOR
SHARE` 로 변경된다. (autocommit 이 꺼진 경우) 즉, `(S)` lock 이 걸린다.
isolation level 이 너무 강력하여 deadlock 이 자주 발생되는 것을 주의 하자.

예를 들어 다음과 같이 deadlock 을 발생시키고 확인해 보자. `update` 수행시
`money` 를 읽어올 때는 consistent read 가 아니다. 새로운 snapshot 에서 `money`
를 읽어온다.

```sql
create database foo;
use foo;
create table account(
  id int primary key,
  state varchar(10) not null,
  money int
);
insert into account values(0,'poor',10),(1,'poor',20);

-- session a
> use foo;
> set session transaction isolation level serializable;
> begin;
> SELECT state FROM account WHERE id = 1;

-- session b
> use foo;
> set session transaction isolation level serializable;
> begin;
> SELECT state FROM account WHERE id = 1;
> UPDATE account SET state = 'rich', money = money * 1000 WHERE id = 1;
-- session b blocked

-- session a
> UPDATE account SET state = 'rich', money = money * 1000 WHERE id = 1;
ERROR 1213 (40001): Deadlock found when trying to get lock; try restarting transaction
-- session a dead lock detected, rollbacked

-- session b
Query OK, 1 row affected (21.69 sec)
-- session b executed
> COMMIT;
```

# Solution of Non-repeatable read in repeatable read isolation level

* [SQL 트랜잭션 - 믿는 도끼에 발등 찍힌다](https://blog.sapzil.org/2017/04/01/do-not-trust-sql-transaction/)

----

MySQL 은 기본적으로 isolation level 이 repeatable read 이다. non-repeatable read 를 해결한다.
phantom read 도 해결된다. 왜지??

그러나 `update, delete` 를 수행할 때 consistent read 를 하지 않는다. 즉, 새로운
snapshot 에서 data 를 읽어오기 때문에 non-repeatable read 의 위험이 있다.

```sql
create database foo;
use foo;
create table account(
  id int primary key,
  state varchar(10) not null,
  money int
);
insert into account values(0,'poor',10),(1,'poor',20);

-- session a
> use foo;
> set session transaction isolation level repeatable read;
> begin;
> SELECT state FROM account WHERE id = 1;

-- session b
> use foo;
> set session transaction isolation level repeatable read;
> begin;
> SELECT state FROM account WHERE id = 1;
> UPDATE account SET state = 'rich', money = money * 1000 WHERE id = 1;
Query OK, 1 row affected (21.69 sec)
> commit;

-- session a
> UPDATE account SET state = 'rich', money = money * 1000 WHERE id = 1;
> COMMIT;
> SELECT * FROM account WHERE id = 1;

-- money became 10,000,000 which was not we wanted.
```

이것을 해결하기 위해 다음과 같은 3 가지 방법을 생각할 수 있다.

* isolation level of serializable
  * 가장 쉽다. dead lock 이 발견되고 바로 rollback 된다. 그러나 system
    throughput 이 낮아진다.
* `select for update`
  * intention lock `(IS)` 이 걸린다. 두번 째 transaction 은 대기한다. 역시
    system throughput 이 낮아진다.
* `update ... where ...`
  * 다른 field 의 조건을 where 에 추가한다. 가장 합리적이다. 그러나 where 에
    추가할 field 가 없다면 적당한 방법이 아니다.
* Optimistic Lock
  * version field 를 추가하고 optimistic lock 을 이용한다. 가장 합리적이다.

`UPDATE ... WHERE ...` 으로 안되면 [Optimistic Locking](/systemdesign/README.md#optimistic-lock-vs-pessimistic-lock) 으로 해결하자. Spring Data JPA 는
[Optimistic Locking](/spring/SpringDataJpa.md#optimistic-locking) 을 제공한다.

# Practice of Read Uncommitted

```sql
-- session 1
set session transaction isolation level read uncommitted;
start transaction;
select * from user;
-- no lock
+------+------+
| id   | name |
+------+------+
|    1 | foo  |
|    2 | bar  |
|    3 | baz  |
+------+------+

-- session 2
set session transaction isolation level read uncommitted;
start transaction;
update user SET name="foofoo" where id=1;
-- session 2 acquired (IX) of the row(is=1)
insert into user (id, name) values (4, "tree");
-- session 2 acquired (IX) of the row(is=4)

-- session 1
select * from user;
+------+--------+
| id   | name   |
+------+--------+
|    1 | foofoo |
|    2 | bar    |
|    3 | baz    |
|    4 | tree   |
+------+--------+
-- session 1 reads new snapshot which includes uncommited update from session 2
commit;

-- session 2
commit;
```

# Practice of Read Committed

```sql
-- session 1
> set session transaction isolation level read committed;
> start transaction;
> select * from user;
+------+--------+
| id   | name   |
+------+--------+
|    1 | foofoo |
|    2 | bar    |
|    3 | baz    |
|    4 | tree   |
+------+--------+

-- session 2
> set session transaction isolation level read committed;
> start transaction;
> update user set name="foo" where id=1;
-- session 2 acquired (IX) of the row(is=1)
> insert into user (id, name) values(5, "bear");
-- session 2 acquired (IX) of the row(is=5)

-- session 1
> select * from user;
+------+--------+
| id   | name   |
+------+--------+
|    1 | foofoo |
|    2 | bar    |
|    3 | baz    |
|    4 | tree   |
+------+--------+
-- session 1 read not uncommited data but committed data from session 2 

-- session 2
> commit;

-- session1
> select * from user;
-- session 1 reads new snapshot which includes committed update, insert from session 2
+------+------+
| id   | name |
+------+------+
|    1 | foo  |
|    2 | bar  |
|    3 | baz  |
|    4 | tree |
|    5 | bear |
+------+------+
-- session 1 non-repeatable read happened
commit;
```

# Practice of Repeatable Read

```sql
-- session 1
> set session transaction isolation level repeatable read;
> start transaction;
> select * from user;
+------+------+
| id   | name |
+------+------+
|    1 | foo  |
|    2 | bar  |
|    3 | baz  |
|    4 | tree |
|    5 | bear |
+------+------+

-- session 2
> start transaction;
> update user set name = "foofoo" WHERE id = 1;
-- session 2 acquired (IX) of the row(id=1)
> insert into user (id, name) values (6, "lion");
-- session 2 acquired (IX) of the row(id=6)

-- session 1
> select * from user;
-- session 1 reads data from the first snapshot
+------+------+
| id   | name |
+------+------+
|    1 | foo  |
|    2 | bar  |
|    3 | baz  |
|    4 | tree |
|    5 | bear |
+------+------+
-- where is phantom read??? where is the row(id=6)???

-- session 2
> commit;

-- session 1
> select * from user;
-- session 1 reads data form the first snapshot which does not include commited update from session 2
+------+------+
| id   | name |
+------+------+
|    1 | foo  |
|    2 | bar  |
|    3 | baz  |
|    4 | tree |
|    5 | bear |
+------+------+
> commit;
> select * from user;
-- session 1 reads new snapshot which includes committed update from session 2
+------+--------+
| id   | name   |
+------+--------+
|    1 | foofoo |
|    2 | bar    |
|    3 | baz    |
|    4 | tree   |
|    5 | bear   |
|    6 | lion   |
+------+--------+
-- session 1 reads data from the specific snapshot was created when transaction started
-- MySQL repeatable read looks like cover the phantom read because of consistent read.
```

# Practice of Searializable

```sql
-- session 1
> set session transaction isolation level serializable;
> start transaction;
> select * from user;
-- same with "select * from user for share"
-- session 1 acquired (IS)
+------+--------+
| id   | name   |
+------+--------+
|    1 | foofoo |
|    2 | bar    |
|    3 | baz    |
|    4 | tree   |
|    5 | bear   |
|    6 | lion   |
+------+--------+

-- session 2
> start transaction;
> update user set name = "barbar" where id = 2;
-- session 2 try to acquire (IX) of the row(id=2)
-- session 2 blocked

-- session 1
> commit;
-- session 2 unblocked

-- session 2
> commit;
```

# Practice of "SELECT ... FOR UPDATE"

`SELECT ... FOR UPDATE` 를 언제 사용하면 좋은지 생각해 보자.

session 1 이 update 를 수행한 후 session 2 가 udpate 를 수행할 때 block 된다.
session 2 입장에서 제대로 된 값을 읽어오지 않은채 update 를 수행할 수 있다.
이것을 개선해 보자.

session 1 이 `select...` 대신 `select ... from update` 를 사용하면 `(IX)` 를
획득할 것이다. 이때 session 2 가 `select ... from update` 를 실행할 때 `(IX)` 를
획득하기 위해 block 될 것이다. session 2 가 제대로 된 값을 읽어 오고 update 를
실행할 수 있다.

```sql
-- AsIs
-- session 1
create database foo;
use foo;
create table tab(
  k int primary key,
  v int not null
);
insert into tab values(1,1),(2,2),(3,3);

-- session 1
set session transaction isolation level read committed;
begin;
select v from tab where k = 1;
update tab set v = 100 where k = 1 and v = 1;

-- session 2
set session transaction isolation level read committed;
begin;
select v from tab where k = 1;
update tab set v = 200 where k = 1 and v = 1;
-- session 2 try to acquire (IX) of the row(k=1,v=1)

-- session 1
commit;
-- session 2 unblocked

-- session 2
commit;
```

```sql
-- ToBe
-- session 1
create database foo;
use foo;
create table tab(
  k int primary key,
  v int not null
);
insert into tab values(1,1),(2,2),(3,3);

-- session 1
set session transaction isolation level read committed;
begin;
select v from tab where k = 1 for update;
update tab set v = 100 where k = 1 and v = 1;

-- session 2
set session transaction isolation level read committed;
begin;
select v from tab where k = 1 for update;
-- session 2 try to acquire (IX) of the row(k=1)
-- session 2 blocked

-- session 1
commit;
-- session 2 unblocked

-- session 2
update tab set v = 200 where k = 1 and v = 1;
-- session 2 acquired (IX) of the row(k=1,v=1)

-- session 1
commit;
-- session 2 unblocked

-- session 2
+-----+
| v   |
+-----+
| 100 |
+-----+
1 row in set (35.15 sec)
commit;
```
