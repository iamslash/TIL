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
- [IX practice](#ix-practice)
- [Inno-db Deadlock](#inno-db-deadlock)

----

# Abstract

MySQL InnoDB 의 Lock 에 대해 정리한다.  

# Materials

> * [MySQL InnoDB lock & deadlock 이해하기](https://www.letmecompile.com/mysql-innodb-lock-deadlock/)
> * [14.7.1 InnoDB Locking @ mysql](https://dev.mysql.com/doc/refman/5.7/en/innodb-locking.html)
> * [InnoDB locking](https://github.com/octachrome/innodb-locks)

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

-- sessionA
create database foo;
use foo;
create table tab(
  k int primary key,
  v int not null
);
insert into tab values(1,1),(2,2),(3,3);
set session transaction isolation level read committed;
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

intention lock 은 다음과 같이 `show engine innodb status` 로 확인할 수 있다.

```sql
> show engine innodb status;
...
TABLE LOCK table `test`.`t` trx id 10080 lock mode IX
...
```

# Record Locks

Record Lock 은 index record 에 걸리는 lock 을 말한다. 예를 들어 transaction
t1 에서 `SELECT c1 FROM t WHERE c1 = 10 FOR UPDATE;` 를 실행하면 transaction t2
에서 `t.c1 = 10` 에 해당하는 row 에 대해 insert, update, delete 을 수행할 수
없다.

```sql
> SHOW ENGINE INNODB STATUS;
RECORD LOCKS space id 58 page no 3 n bits 72 index `PRIMARY` of table `test`.`t`
trx id 10078 lock_mode X locks rec but not gap
Record lock, heap no 2 PHYSICAL RECORD: n_fields 3; compact format; info bits 0
 0: len 4; hex 8000000a; asc     ;;
 1: len 6; hex 00000000274f; asc     'O;;
 2: len 7; hex b60000019d0110; asc        ;;
```

# Gap Locks

Gap Lock 은 index record 들 사이, 특정 index record 이전, 특정 index record 이후에
걸리는 lock 이다. 예를 들어 transaction t1 에서  `SELECT c1 FROM t WHERE c1 BETWEEN 10 and 20 FOR UPDATE;` 를 수행하면 transaction t2 는 `t.c1 = 15` 에 해댕하는 row 를 insert 할 수 없다. 

[isolation level](/isolation/README.md) 이 read commited 이면 gap lock 이 비활성화 된다.

# Next-Key Locks

Next-Key Lock 은 index record 에 대한 record lock 과 그 index record 의 이전
index record 들에 대한 gab lock 을 합한 것이다.

# Insert Intention Locks

Insert intention lock 은 gap lock 의 종류이다. `INSERT ...` 를 실행할 때 획득한다.
서로 다른 두 transaction 은 gap 에서 같은 위치의 record 를 삽입하지 않는다면 conflict 는
없다.

예를 들어 다음과 같이 `sessionA, sessionB` 를 살펴보자.

```sql
-- sessionA
mysql> CREATE TABLE child (id int(11) NOT NULL, PRIMARY KEY(id)) ENGINE=InnoDB;
mysql> INSERT INTO child (id) values (90),(102);

mysql> START TRANSACTION;
mysql> SELECT * FROM child WHERE id > 100 FOR UPDATE;
+-----+
| id  |
+-----+
| 102 |
+-----+
-- sessionA acquired insert intention lock

-- sessionB
mysql> START TRANSACTION;
mysql> INSERT INTO child (id) VALUES (101);
-- sessionB acquired insert intention lock

mysql> show engine innodb status;
RECORD LOCKS space id 31 page no 3 n bits 72 index `PRIMARY` of table `test`.`child`
trx id 8731 lock_mode X locks gap before rec insert intention waiting
Record lock, heap no 3 PHYSICAL RECORD: n_fields 3; compact format; info bits 0
 0: len 4; hex 80000066; asc    f;;
 1: len 6; hex 000000002215; asc     " ;;
 2: len 7; hex 9000000172011c; asc     r  ;;...
```

# AUTO-INC Locks

An AUTO-INC lock is a special table-level lock taken by transactions inserting
into tables with AUTO_INCREMENT columns.

# Predicate Locks for Spatial Indexes

InnoDB supports SPATIAL indexing of columns containing spatial columns???

# IX practice

다음은 `select ... for update` 와 `update` 를 두개의 transaction
을 실습한 것이다. innodb 의 transaction 상태가 어떻게 변하는지 살펴보자.

```sql
-- sessionA
> begin;
> SELECT * from tab where k=1 for update;
> show engine innodb status;
------------
TRANSACTIONS
------------
Trx id counter 1841
Purge done for trx's n:o < 1835 undo n:o < 0 state: running but idle
History list length 0
LIST OF TRANSACTIONS FOR EACH SESSION:
---TRANSACTION 421609938063360, not started
0 lock struct(s), heap size 1128, 0 row lock(s)
---TRANSACTION 421609938061744, not started
0 lock struct(s), heap size 1128, 0 row lock(s)
---TRANSACTION 421609938060936, not started
0 lock struct(s), heap size 1128, 0 row lock(s)
---TRANSACTION 1840, ACTIVE 4 sec
2 lock struct(s), heap size 1128, 1 row lock(s)
MySQL thread id 10, OS thread handle 140134792005376, query id 124 localhost root starting
show engine innodb status
```

```sql
-- sessionB
> begin;
> UPDATE set v=11 where k=1;
-- sessionB blocked for a while
```

```sql
-- sessionA
> show engine innodb status;
------------
TRANSACTIONS
------------
Trx id counter 1842
Purge done for trx's n:o < 1835 undo n:o < 0 state: running but idle
History list length 0
LIST OF TRANSACTIONS FOR EACH SESSION:
---TRANSACTION 421609938061744, not started
0 lock struct(s), heap size 1128, 0 row lock(s)
---TRANSACTION 421609938060936, not started
0 lock struct(s), heap size 1128, 0 row lock(s)
---TRANSACTION 1841, ACTIVE 4 sec starting index read
mysql tables in use 1, locked 1
LOCK WAIT 2 lock struct(s), heap size 1128, 1 row lock(s)
MySQL thread id 11, OS thread handle 140134790948608, query id 127 localhost root updating
UPDATE tab set v=11 where k=1
------- TRX HAS BEEN WAITING 4 SEC FOR THIS LOCK TO BE GRANTED:
RECORD LOCKS space id 3 page no 4 n bits 72 index PRIMARY of table `foo`.`tab` trx id 1841 lock_mode X locks rec but not gap waiting
Record lock, heap no 2 PHYSICAL RECORD: n_fields 4; compact format; info bits 0
 0: len 4; hex 80000001; asc     ;;
 1: len 6; hex 000000000726; asc      &;;
 2: len 7; hex 81000001180110; asc        ;;
 3: len 4; hex 80000001; asc     ;;

------------------
---TRANSACTION 1840, ACTIVE 56 sec
2 lock struct(s), heap size 1128, 1 row lock(s)
MySQL thread id 10, OS thread handle 140134792005376, query id 128 localhost root starting
show engine innodb status
```

```sql
-- sessionB unblocked after for a while
ERROR 1205 (HY000): Lock wait timeout exceeded; try restarting transaction
> show engine innodb status;
------------
TRANSACTIONS
------------
Trx id counter 1842
Purge done for trx's n:o < 1835 undo n:o < 0 state: running but idle
History list length 0
LIST OF TRANSACTIONS FOR EACH SESSION:
---TRANSACTION 421609938061744, not started
0 lock struct(s), heap size 1128, 0 row lock(s)
---TRANSACTION 421609938060936, not started
0 lock struct(s), heap size 1128, 0 row lock(s)
---TRANSACTION 1841, ACTIVE 97 sec
1 lock struct(s), heap size 1128, 1 row lock(s)
MySQL thread id 11, OS thread handle 140134790948608, query id 129 localhost root starting
show engine innodb status
---TRANSACTION 1840, ACTIVE 149 sec
2 lock struct(s), heap size 1128, 1 row lock(s)
MySQL thread id 10, OS thread handle 140134792005376, query id 128 localhost root
```

```sql
-- sessionA
> rollback;
> show engine innodb status;
------------
TRANSACTIONS
------------
Trx id counter 1842
Purge done for trx's n:o < 1835 undo n:o < 0 state: running but idle
History list length 0
LIST OF TRANSACTIONS FOR EACH SESSION:
---TRANSACTION 421609938062552, not started
0 lock struct(s), heap size 1128, 0 row lock(s)
---TRANSACTION 421609938061744, not started
0 lock struct(s), heap size 1128, 0 row lock(s)
---TRANSACTION 421609938060936, not started
0 lock struct(s), heap size 1128, 0 row lock(s)
---TRANSACTION 1841, ACTIVE 224 sec
1 lock struct(s), heap size 1128, 1 row lock(s)
MySQL thread id 11, OS thread handle 140134790948608, query id 129 localhost root
```

```sql
-- sessionB
> rollback;
> show engine innodb status;
------------
TRANSACTIONS
------------
Trx id counter 1842
Purge done for trx's n:o < 1835 undo n:o < 0 state: running but idle
History list length 0
LIST OF TRANSACTIONS FOR EACH SESSION:
---TRANSACTION 421609938063360, not started
0 lock struct(s), heap size 1128, 0 row lock(s)
---TRANSACTION 421609938062552, not started
0 lock struct(s), heap size 1128, 0 row lock(s)
---TRANSACTION 421609938061744, not started
0 lock struct(s), heap size 1128, 0 row lock(s)
---TRANSACTION 421609938060936, not started
0 lock struct(s), heap size 1128, 0 row lock(s)
```

# Inno-db Deadlock

> * [MySQL InnoDB lock & deadlock 이해하기](https://www.letmecompile.com/mysql-innodb-lock-deadlock/)
> * [14.7.5.1 An InnoDB Deadlock Example @ mysql](https://dev.mysql.com/doc/refman/5.7/en/innodb-deadlock-example.html)

mysql 의 innodb 는 Deadlock 을 detect 할 수 있다. 만약 mysql 이 Deadlock 을
detect 하면 어느 한 transaction 의 lock wait 을 중지하여 Deadlock 을 해결한다.
즉, 바로 error 를 리턴한다.

예를 들어 다음과 같이 deadlock 을 일으켜보자. manual 처럼 deadlock detect
안되는데... isolation level 이 read committed, repeatable read 일 때 똑같이
deadlock detect 안된다. 왜지???

```sql
-- sessionA
mysql> set session transaction isolation level read committed;
mysql> CREATE TABLE t (i INT) ENGINE = InnoDB;
Query OK, 0 rows affected (1.07 sec)

mysql> INSERT INTO t (i) VALUES(1);
Query OK, 1 row affected (0.09 sec)

mysql> START TRANSACTION;
Query OK, 0 rows affected (0.00 sec)

mysql> SELECT * FROM t WHERE i = 1 LOCK IN SHARE MODE;
+------+
| i    |
+------+
|    1 |
+------+
-- sessionA acquired (S) lock for a row (i = 1)

-- sessionB
mysql> set session transaction isolation level read committed;
mysql> START TRANSACTION;
Query OK, 0 rows affected (0.00 sec)

mysql> DELETE FROM t WHERE i = 1;
-- sessionB tried to acquire (X) lock for a row (i = 1)
-- sessionB blocked

-- sessionA
mysql> DELETE FROM t WHERE i = 1;
Query OK, 1 row affected (0.00 sec)
-- sessionA tried to acquire (X) lock for a row (i = 1)
-- why deadlock not found???
```
