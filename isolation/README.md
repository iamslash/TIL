- [Abstract](#abstract)
- [Materials](#materials)
- [Launch mysql docker container](#launch-mysql-docker-container)
- [Create Table](#create-table)
- [Read Uncommitted](#read-uncommitted)
- [Read Committed](#read-committed)
- [Repeatable Read](#repeatable-read)
- [Searializable](#searializable)
- [Innodb Lock](#innodb-lock)

----

# Abstract

mysql 을 이용하여 isolation level 을 실습해 본다. [transaction @ TIL](/spring/README.md#transactional)

# Materials

> * [Mysql innoDB Lock, isolation level과 Lock 경쟁](https://taes-k.github.io/2020/05/17/mysql-transaction-lock/)

# Launch mysql docker container

```bash
$ docker run -p 3306:3306 --rm --name my-mysql -e MYSQL_ROOT_PASSWORD=1 -e MYSQL_DATABASE=hello -e MYSQL_USER=iamslash -e MYSQL_PASSWORD=1 -d mysql

$ docker ps
$ docker exec -it my-mysql /bin/bash

$ mysql -u iamslash -p
mysql> show databases
mysql> use hello
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
> SHOW VARIABLES WHERE VARIABLE_NAME='transaction_isolation';
+-----------------------+-----------------+
| Variable_name         | Value           |
+-----------------------+-----------------+
| transaction_isolation | REPEATABLE-READ |
+-----------------------+-----------------+
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
> select * from user;
+------+--------+
| id   | name   |
+------+--------+
|    1 | foofoo |
|    2 | bar    |
|    3 | baz    |
|    4 | tree   |
+------+--------+

-- Can read uncommitted read
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
-- Can committed data after commit;
-- session 1 read data from snapshot was created at start trasaction
```

# Searializable

```sql
-- session 1
> set session transaction isolation level serializable;
> start transaction;
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

-- session 2
> start transaction;
> update user set name = "barbar" where id = 2;
-- locked

-- session 1
> commit;
-- unlocked from session2

-- Can not update data from selected table
```

# Innodb Lock

WIP...
