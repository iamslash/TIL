- [Abstract](#abstract)
- [Materials](#materials)
- [Launch mysql docker container](#launch-mysql-docker-container)
- [Create Table](#create-table)
- [Consistent Read](#consistent-read)
- [Read Uncommitted](#read-uncommitted)
- [Read Committed](#read-committed)
- [Repeatable Read](#repeatable-read)
- [Searializable](#searializable)
- [Update locking within transaction](#update-locking-within-transaction)
- [Innodb Lock](#innodb-lock)

----

# Abstract

[mysql](/mysql/README.md) 은 [Lock](/mysql/README.md#inno-db-locking) 이용하여
동시성을 제어한다. Isolation Level 을 달리하여 [Concurrency Problems](/database/README.md#concurrency-problems-in-transactions) 을 해결할 수 있다. Isolation Level 이 높을 수록
System throughput 은 낮아진다. 보통 Isolation Level 을 read committed 으로 설정한다.

[mysql](/mysql/README.md) 로 실습해 본다.

# Materials

> * [Mysql innoDB Lock, isolation level과 Lock 경쟁](https://taes-k.github.io/2020/05/17/mysql-transaction-lock/)
> * [Optimistic locking in MySQL](https://stackoverflow.com/questions/17431338/optimistic-locking-in-mysql)

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
> CREATE TABLE user (
    id int,
    name varchar(255)
);
> insert into user (id, name) values (1, "foo");
> insert into user (id, name) values (2, "bar");
> insert into user (id, name) values (3, "baz");

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

----

Consistent Read 는 특정 시점의 Data 를 읽는 것을 말한다. MySQL 은 lock 을 걸고
특정 시점의 Data 를 읽지는 않는다. 이 것은 System throughput 을 낮추기 때문이다.

특정 시점의 Snapshot 을 읽는다. MySQL 은 commit log 를 기록한다. Snapshot 은
특정 시점까지의 commit log 모음을 말한다. 이것을 어떻게 조합하여 특정 시점의 Snapshot
을 만든다는 것일까???

[Isolation Level](/isolation/README.md) 이 **read uncommitted** 인 경우를 생각해
보자. transaction 1 은 read 할 때 다른 transaction 이 commit 하지 않는 data 도
읽어온다. dirty read 가 발생한다. 물론 non-repeatable read, phantom read 도
발생한다.

[Isolation Level](/isolation/README.md) 이 **read committed** 인 경우를 생각해 보자.
transaction 1 에서 read 할 때 마다 snapshot 을 기록해 둔다. transaction 2
에서 특정 row 를 변경하고 commit 했다면 transaction 1 에서 다시 read 할 때 
새로운 snaptshot 을 읽기 때문에 non-repeatable read 가 발생한다. 

또한 [Isolation Level](/isolation/README.md) 이 **read committed** 인 경우는
gab lobck 을 사용하지 않는다. 따라서 phantom read 가 발생한다.

[Isolation Level](/isolation/README.md) 이 **repeatable read** 인 경우를 생각해
보자. 처음 read 한 때의 snapshot 을 기록해 둔다. 이후 read 할 때는 처음 read 한
때의 snapshot 을 사용한다. 따라서 non-repeatable read 가 발생하지 않는다.

[Isolation Level](/isolation/README.md) 이 **serializable** 인 경우를 생각해
보자. 기본적으로 repeatble read 와 같다. 단, `SELECT ...` 가 `SELECT ... FOR
SHARE` 로 변경된다. (autocommit 이 꺼진 경우) 즉, `(S)` lock 이 걸린다.
isolation level 이 너무 강력하여 deadlock 이 자주 발생되는 것을 주의 하자.

예를 들어 다음과 같은 상황을 살펴보자. 왜 deadlock detect 가 안되지???

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
> SELECT state FROM account WHERE id = 1;

-- session b
> use foo;
> set session transaction isolation level serializable;
> SELECT state FROM account WHERE id = 1;
> UPDATE account SET state = 'rich', money = money * 1000 WHERE id = 1;
> COMMIT;

-- session a
> UPDATE account SET state = 'rich', money = money * 1000 WHERE id = 1;
> COMMIT;
```

# Read Uncommitted

```sql
-- session 1
> set session transaction isolation level read uncommitted;
> start transaction;
> select * from user;
+------+------+
| id   | name |
+------+------+
|    1 | foo  |
|    2 | bar  |
|    3 | baz  |
+------+------+

-- session 2
> start transaction;
> update user SET name="foofoo" where id=1;
> insert into user (id, name) values (4, "tree");

-- session 1
-- Can read uncommitted read
> select * from user;
+------+--------+
| id   | name   |
+------+--------+
|    1 | foofoo |
|    2 | bar    |
|    3 | baz    |
|    4 | tree   |
+------+--------+
```

# Read Committed

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
> start transaction;
> update user SET name="foo" Where id=1;
> insert into user (id, name) values(5, "bear");

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

-- Can not read uncommitted data
-- Can not read consistent data repeatedly

-- session 2
> commit;

-- session1
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
```

# Repeatable Read

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
> update user SET name = "foofoo" WHERE id = 1;
> insert into user (id, name) values (6, "lion");

-- session 1
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
> commit;

-- session 1
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
> commit;
> select * from user;
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

-- Can not read even though session 2 commited
-- Can read consistent data repeatedly
-- Can read committed data after commit;
-- session 1 read data from snapshot was created at start transaction
```

# Searializable

```sql
-- session 1
> set session transaction isolation level serializable;
> start transaction;
> select * from user;
-- table lock???
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
-- session 2 locked

-- session 1
> commit;
-- session 2 unlocked

-- Can not update data even from selected table
```

# Update locking within transaction

`session 2` 의 update 가 blocking 되는 것을 유의하자. isolation level 이 read
committed 이지만 transaction 안의 update 는 blocking 된다. `session 2` 가
`select` 대신 `select ... for update` 을 사용하면 `update` 대신 `select ... for
update` 에서 blocking 될 것이다. 제대로 된 값을 읽어서 처리하는 것이 더욱 좋다. 
따라서 `select ... for update` 을 사용하는 것이 좋다.

```sql
-- session 1
create database foo;
use foo;
create table tab(
  k int primary key,
  v int not null
);
insert into tab values(1,1),(2,2),(3,3);
set session transaction isolation level read committed;

-- session 1
begin;
select v from tab where k = 1;
update tab set v = 100 where k = 1 and v = 1;

-- session 2
begin;
select v from tab where k = 1;
update tab set v = 200 where k = 1 and v = 1;
-- session 2 locked

-- session 1
commit;
-- session 2 unlocked
```

# Innodb Lock

> [mysql Inno-db locks @ TIL](/mysql/README.md#inno-db-locking)
