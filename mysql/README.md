- [Abstract](#abstract)
- [Materials](#materials)
- [Install with docker](#install-with-docker)
- [Basic](#basic)
  - [Bin Log](#bin-log)
  - [Types of Keys](#types-of-keys)
    - [Key](#key)
    - [Primary Key](#primary-key)
    - [Unique Kye](#unique-kye)
    - [Foreign Key](#foreign-key)
  - [BIGINT(20) vs INT(20)](#bigint20-vs-int20)
  - [SKIP LOCKED, NOWAIT](#skip-locked-nowait)
  - [SAVEPOINT](#savepoint)
  - [Explain](#explain)
- [Advanced](#advanced)
  - [how to reset password](#how-to-reset-password)
  - [Inno-db Locking](#inno-db-locking)
  - [how to run multiple mysqld instances](#how-to-run-multiple-mysqld-instances)
  - [XA](#xa)

----

# Abstract

- mysqld 에 대해 적어보자.
- msyql 의 HELP를 주로 사용하자.

# Materials

- [이곳](https://www.percona.com/resources/videos)에
  쓸만한 webinar동영상들이 있다.

# Install with docker

```console
$ docker run -p 3306:3306 --rm --name my-mysql -e MYSQL_ROOT_PASSWORD=1 -e MYSQL_DATABASE=hello -e MYSQL_USER=iamslash -e MYSQL_PASSWORD=1 -d mysql

$ docker ps
$ docker exec -it my-mysql /bin/bash

$ mysql -u iamslash -p
mysql> show databases
mysql> use hello
```

# Basic

## Bin Log

* [Binary Log(binlog) 사용하기](https://systemv.tistory.com/52)

-----

`bin log` 는 PIT (Point In Time) restore 혹은 replication 에 꼭 필요하다.

## Types of Keys

* [MySQL 에서 사용하는 Key 의 정의와 종류들에 대하여](https://jins-dev.tistory.com/entry/MySQL-%EC%97%90%EC%84%9C-%EC%82%AC%EC%9A%A9%ED%95%98%EB%8A%94-Key-%EC%9D%98-%EC%A0%95%EC%9D%98%EC%99%80-%EC%A2%85%EB%A5%98%EB%93%A4%EC%97%90-%EB%8C%80%ED%95%98%EC%97%AC)

-------

### Key

Duplication, NULL 을 허용한다. 그러나 NULL 은 performance issue 때문에 추천하지 않는다. Index 가 설정된다.

### Primary Key

Duplication, NULL 을 허용하지 않는 Key 이다. 즉, NOT NULL & UNIQUE option 이 포함된다. Index 가 설정된다.

### Unique Kye

Duplication 은 허용하지 않고 NULL 을 허용하는 Key 이다. Index 가 설정된다.

### Foreign Key

다른 Table 의 Primary Key 를 참조하는 Column 을 말한다. 다음과 같은 option 이 있다. 

- RESTRICT : FK 관계를 맺고 있는 데이터 ROW 의 변경(UPDATE) 또는 삭제(DELETE) 를 막는다.
- CASCADE : FK 관계를 맺을 때 가장 흔하게 접할 수 있는 옵션으로, FK 와 관계를 맺은 상대 PK 를 직접 연결해서 DELETE 또는 UPDATE 시, 상대 Key 값도 삭제 또는 갱신시킨다. 이 때에는 Trigger 가 발생하지 않으니 주의하자.
- SET NULL : 논리적 관계상 부모의 테이블, 즉 참조되는 테이블의 값이 변경 또는 삭제될 때 자식 테이블의 값을 NULL 로 만든다. UPDATE 쿼리로 인해 SET NULL 이 허용된 경우에만 동작한다.
- NO ACTION : RESTRICT 옵션과 동작이 같지만, 체크를 뒤로 미룬다.
- SET DEFAULT : 변경 또는 삭제 시에 값을 DEFAULT 값으로 세팅한다.

## BIGINT(20) vs INT(20)

* [mysql 에서 bigint(20), int(11) 의 의미](https://netmaid.tistory.com/42)
  * [Types in MySQL: BigInt(20) vs Int(20)](https://stackoverflow.com/questions/3135804/types-in-mysql-bigint20-vs-int20)
* [11.1.2 Integer Types (Exact Value) - INTEGER, INT, SMALLINT, TINYINT, MEDIUMINT, BIGINT @ mysql](https://dev.mysql.com/doc/refman/8.0/en/numeric-types.html)
  
------

mysql `bigint(20)` 의 `20` 은 display hint 일 뿐이다. 실제 저장공간과는 상관없다. mysql 은 다음과 같은 Data Type Range 를 지원한다.

| Type	| Storage (Bytes)	|Minimum Value Signed	|Minimum Value Unsigned	|Maximum Value Signed	|Maximum Value Unsigned |
|--|--|--|--|--|--|
|TINYINT |	1 |	-128	|0|	127|	255|
|SMALLINT	|2|	-32768|	0|	32767|	65535|
|MEDIUMINT	|3|	-8388608	|0	|8388607	|16777215|
|INT	|4	|-2147483648|	0	|2147483647	|4294967295|
|BIGINT	|8	|-2^63|	0|	2^63-1|	2^64-1|

다음과 같이 `bigint(20) ZEROFILL` 을 수행하면 `1234` 의 값이 20 자리의 숫자로 표현된다.

```
CREATE TABLE foo ( bar INT(20) ZEROFILL );
INSERT INTO foo (bar) VALUES (1234);
SELECT bar from foo;

+----------------------+
| bar                  |
+----------------------+
| 00000000000000001234 |
+----------------------+
```

## SKIP LOCKED, NOWAIT

* [MySQL 8.0.1: Using SKIP LOCKED and NOWAIT to handle hot rows](https://dev.mysql.com/blog-archive/mysql-8-0-1-using-skip-locked-and-nowait-to-handle-hot-rows/)
* [MySQL8.0 SKIP LOCKED / NOWAIT 활용하기](https://kimdubi.github.io/mysql/skip_locked/)

MySQL 8.0.1 부터 지원한다.

* NOWAIT: row 에 lock 이 걸려있으면 실패처리한다. innodb_lock_wait_timeout 만큼 기다리지 않는다.
* SKIP LOCKED: lock 이 걸려있지 않은 row 만 모아서 리턴한다.

예를 들어 다음과 같이 progress 테이블에 data 를 삽입하자.

```sql
> CREATE TABLE progress (
  seq INT PRIMARY KEY,
  progress ENUM('PREPARE', 'DONE') DEFAULT 'PREPARE'
);
INSERT INTO progress (seq)
WITH RECURSIVE my_cte AS
(
SELECT 1 AS n
UNION ALL
SELECT 1+n FROM my_cte WHERE n<10
)
SELECT * FROM my_cte;
```

다음과 같은 경우 session 2 는 lock 을 얻기위해 기다리다가 실패한다.

```sql
-- session 1
> begin;
Query OK, 0 rows affected (0.00 sec)

> select * from progress where progress='PREPARE' order by seq limit 3 for update;
+-----+----------+
| seq | progress |
+-----+----------+
|   1 | PREPARE  |
|   2 | PREPARE  |
|   3 | PREPARE  |
+-----+----------+
3 rows in set (0.01 sec)

-- session 2
> begin;
Query OK, 0 rows affected (0.00 sec)
> select * from progress where progress='PREPARE' order by seq limit 3 for update;

ERROR 1205 (HY000): Lock wait timeout exceeded; try restarting transaction 
```

다음과 같이 skip locked 를 사용하여 unlocked rows 를 리턴해 보자.

```sql
--- session 1
> begin;
Query OK, 0 rows affected (0.00 sec)

> select * from progress where progress='PREPARE' order by seq limit 3 for update ;
+-----+----------+
| seq | progress |
+-----+----------+
|   1 | PREPARE  |
|   2 | PREPARE  |
|   3 | PREPARE  |
+-----+----------+
3 rows in set (0.00 sec)


--- session 2 
> begin;
Query OK, 0 rows affected (0.00 sec)

> select * from progress where progress='PREPARE' order by seq limit 3 for update skip locked;
+-----+----------+
| seq | progress |
+-----+----------+
|   4 | PREPARE  |
|   5 | PREPARE  |
|   6 | PREPARE  |
+-----+----------+
3 rows in set (0.00 sec) 
```

이번에는 영화관의 좌석을 예매하는 예를 살펴보자. 다음과 같이 seats 테이블에 data 를 삽입한다.

```sql
CREATE TABLE seats (
  seat_no INT PRIMARY KEY,
  booked ENUM('YES', 'NO') DEFAULT 'NO'
);
-- generate 100 sample rows
INSERT INTO seats (seat_no)
WITH RECURSIVE my_cte AS
(
SELECT 1 AS n
UNION ALL
SELECT 1+n FROM my_cte WHERE n<100
)
SELECT * FROM my_cte;
```

다음과 같이 skip locked 를 사용하여 unlocked rows 를 반환한다.

```sql
--- session 1  1~90번 좌석을 예매 중

> begin;
Query OK, 0 rows affected (0.00 sec)

> SELECT * FROM seats WHERE seat_no BETWEEN 1 AND 90 AND booked = 'NO' FOR UPDATE ;

--- session 2 
> begin;
Query OK, 0 rows affected (0.00 sec)

> SELECT * FROM seats where booked = 'NO' for update skip locked;
+---------+--------+
| seat_no | booked |
+---------+--------+
|      92 | NO     |
|      93 | NO     |
|      94 | NO     |
|      95 | NO     |
|      96 | NO     |
|      97 | NO     |
|      98 | NO     |
|      99 | NO     |
|     100 | NO     |
+---------+--------+
9 rows in set (0.00 sec)
```

## SAVEPOINT

* [13.3.4 SAVEPOINT, ROLLBACK TO SAVEPOINT, and RELEASE SAVEPOINT Statements | mysql](https://dev.mysql.com/doc/refman/8.0/en/savepoint.html)
* [[MYSQL] SAVEPOINT | tistory](https://88240.tistory.com/458)

----

SAVEPOINT 는 다음과 같이 특정 시점으로 ROLLBACK 할 수 있도록 한다.

```sql
-- AUTO COMMIT OFF
SET AUTOCOMMIT = FALSE;

-- Begin transaction
START TRANSACTION;

-- Update
SAVEPOINT A;
UPDATE a SET name = 'foo' WHERE USER_NAME = 'iamslash'

SAVEPOINT B;
UPDATE a SET name = 'bar' WHERE USER_NAME = 'foo'

SAVEPOINT C;
UPDATE a SET name = 'baz' WHERE USER_NAME = 'bar'

-- Rollback to SAVEPOINT C
ROLLBACK TO SAVEPOINT C
```

## Explain

* [Mysql Explain](https://cheese10yun.github.io/mysql-explian/)
* [MySQL Workbench의 VISUAL EXPLAIN으로 인덱스 동작 확인하기](https://engineering.linecorp.com/ko/blog/mysql-workbench-visual-explain-index/)

----

다음과 같은 table 을 제작하자.

```sql
create table if not exists members
(
    id              bigint auto_increment primary key,
    email           varchar(255)  not null,
    name            varchar(255)  not null
);

create table if not exists trxs
(
    id              bigint auto_increment primary key,
    code            varchar(255)  not null,
    partner_trx_id  varchar(255)  not null,
    payment_type    varchar(255)  not null
);


create table if not exists orders
(
    id              bigint auto_increment primary key,
    order_num       varchar(255) not null,
    member_id       bigint        null,
    trx_id          bigint        null,
    constraint member_id_fk
        foreign key (member_id) references members (id),
    constraint trx_id_fk
        foreign key (trx_id) references trxs (id)
);

insert into members (email, name) values ('iamslash@gmail.com', name='David Sun');
```

`explain` 을 이용하면 실행 계획을 알 수 있다.

```sql
explain select * from members where id='1';

+----+-------------+---------+------------+-------+---------------+---------+---------+-------+------+----------+-------+
| id | select_type | table   | partitions | type  | possible_keys | key     | key_len | ref   | rows | filtered | Extra |
+----+-------------+---------+------------+-------+---------------+---------+---------+-------+------+----------+-------+
|  1 | SIMPLE      | members | NULL       | const | PRIMARY       | PRIMARY | 8       | const |    1 |   100.00 | NULL  |
+----+-------------+---------+------------+-------+---------------+---------+---------+-------+------+----------+-------+

explain analyze select * from members where id='1';

-> Rows fetched before execution  (cost=0.00..0.00 rows=1) (actual time=0.025..0.054 rows=1 loops=1)

----------------------------------------------------------------------

explain select * from members where email='iamslash@gmail.com';

+----+-------------+---------+------------+------+---------------+------+---------+------+------+----------+-------------+
| id | select_type | table   | partitions | type | possible_keys | key  | key_len | ref  | rows | filtered | Extra       |
+----+-------------+---------+------------+------+---------------+------+---------+------+------+----------+-------------+
|  1 | SIMPLE      | members | NULL       | ALL  | NULL          | NULL | NULL    | NULL |    1 |   100.00 | Using where |
+----+-------------+---------+------------+------+---------------+------+---------+------+------+----------+-------------+

explain analyze select * from members where email='iamslash@gmail.com';

-> Filter: (members.email = 'iamslash@gmail.com')  (cost=0.35 rows=1) (actual time=0.070..0.095 rows=1 loops=1)
  -> Table scan on members  (cost=0.35 rows=1) (actual time=0.040..0.051 rows=1 loops=1)

----------------------------------------------------------------------

explain 
select m.*, o.*, t.* from members  m
inner join orders o on m.id = o.member_id
inner join trxs t on o.trx_id = t.id
where m.id in (1, 2, 33);

+----+-------------+-------+------------+--------+------------------------+-----------+---------+-------------------+------+----------+-------------+
| id | select_type | table | partitions | type   | possible_keys          | key       | key_len | ref               | rows | filtered | Extra       |
+----+-------------+-------+------------+--------+------------------------+-----------+---------+-------------------+------+----------+-------------+
|  1 | SIMPLE      | t     | NULL       | ALL    | PRIMARY                | NULL      | NULL    | NULL              |    1 |   100.00 | NULL        |
|  1 | SIMPLE      | o     | NULL       | ref    | member_id_fk,trx_id_fk | trx_id_fk | 9       | hello.t.id        |    1 |   100.00 | Using where |
|  1 | SIMPLE      | m     | NULL       | eq_ref | PRIMARY                | PRIMARY   | 8       | hello.o.member_id |    1 |   100.00 | NULL        |
+----+-------------+-------+------------+--------+------------------------+-----------+---------+-------------------+------+----------+-------------+

----------------------------------------------------------------------

explain analyze
select m.*, o.*, t.* from members  m
inner join orders o on m.id = o.member_id
inner join trxs t on o.trx_id = t.id
where m.id in (1, 2, 33);

-> Nested loop inner join  (cost=1.05 rows=1) (actual time=0.089..0.089 rows=0 loops=1)
  -> Nested loop inner join  (cost=0.70 rows=1) (actual time=0.066..0.066 rows=0 loops=1)
      -> Table scan on t  (cost=0.35 rows=1) (actual time=0.033..0.033 rows=0 loops=1)
      -> Filter: ((o.member_id in (1,2,33)) and (o.member_id is not null))  (cost=0.35 rows=1) (never executed)
          -> Index lookup on o using trx_id_fk (trx_id=t.id)  (cost=0.35 rows=1) (never executed)
  -> Single-row index lookup on m using PRIMARY (id=o.member_id)  (cost=0.35 rows=1) (never executed)
```

# Advanced

## how to reset password

```
pkill mysqld
sudo /usr/local/mysql/bin/mysqld_safe --skip-grant-tables
mysqld
mysql -u root
> GRANT ALL PRIVILEGES ON *.* to 'root'@'localhost' WITH GRANT OPTION
> FLUSH PRIVILEGES;
```

## Inno-db Locking

[mysql lock](mysql_lock.md)

## how to run multiple mysqld instances

- vim /Users/iamslash/etc/mysql_multi.cnf

```
# Example MySQL config file for small systems.
#
# This is for a system with little memory (<= 64M) where MySQL is only used
# from time to time and it's important that the mysqld daemon
# doesn't use much resources.
#
# MySQL programs look for option files in a set of
# locations which depend on the deployment platform.
# You can copy this option file to one of those
# locations. For information about these locations, see:
# http://dev.mysql.com/doc/mysql/en/option-files.html
#
# In this file, you can use all long options that a program supports.
# If you want to know which options a program supports, run the program
# with the "--help" option.

# The following options will be passed to all MySQL clients
[client]
#password	= your_password
port		= 3306
socket		= /tmp/mysql.sock

# Here follows entries for some specific programs

# The MySQL server
[mysqld]
port		= 3306
socket		= /tmp/mysql.sock
pid-file	= /tmp/mysql.pid
datadir         = /Users/iamslash/var/mysql_db
skip-external-locking
key_buffer_size = 16K
max_allowed_packet = 1M
table_open_cache = 4
sort_buffer_size = 64K
read_buffer_size = 256K
read_rnd_buffer_size = 256K
net_buffer_length = 2K
thread_stack = 128K

[mysqld1]
port		= 3307
socket		= /tmp/mysql1.sock
pid-file	= /tmp/mysql1.pid
datadir         = /Users/iamslash/var/mysql1_db
skip-external-locking
key_buffer_size = 16K
max_allowed_packet = 1M
table_open_cache = 4
sort_buffer_size = 64K
read_buffer_size = 256K
read_rnd_buffer_size = 256K
net_buffer_length = 2K
thread_stack = 128K

[mysqld2]
port		= 3308
socket		= /tmp/mysql2.sock
pid-file	= /tmp/mysql2.pid
datadir         = /Users/iamslash/var/mysql2_db
skip-external-locking
key_buffer_size = 16K
max_allowed_packet = 1M
table_open_cache = 4
sort_buffer_size = 64K
read_buffer_size = 256K
read_rnd_buffer_size = 256K
net_buffer_length = 2K
thread_stack = 128K


# Don't listen on a TCP/IP port at all. This can be a security enhancement,
# if all processes that need to connect to mysqld run on the same host.
# All interaction with mysqld must be made via Unix sockets or named pipes.
# Note that using this option without enabling named pipes on Windows
# (using the "enable-named-pipe" option) will render mysqld useless!
# 
#skip-networking
server-id	= 1

# Uncomment the following if you want to log updates
#log-bin=mysql-bin

# binary logging format - mixed recommended
#binlog_format=mixed

# Causes updates to non-transactional engines using statement format to be
# written directly to binary log. Before using this option make sure that
# there are no dependencies between transactional and non-transactional
# tables such as in the statement INSERT INTO t_myisam SELECT * FROM
# t_innodb; otherwise, slaves may diverge from the master.
#binlog_direct_non_transactional_updates=TRUE

# Uncomment the following if you are using InnoDB tables
#innodb_data_home_dir = /usr/local/mysql/data
#innodb_data_file_path = ibdata1:10M:autoextend
#innodb_log_group_home_dir = /usr/local/mysql/data
# You can set .._buffer_pool_size up to 50 - 80 %
# of RAM but beware of setting memory usage too high
#innodb_buffer_pool_size = 16M
#innodb_additional_mem_pool_size = 2M
# Set .._log_file_size to 25 % of buffer pool size
#innodb_log_file_size = 5M
#innodb_log_buffer_size = 8M
#innodb_flush_log_at_trx_commit = 1
#innodb_lock_wait_timeout = 50

[mysqldump]
quick
max_allowed_packet = 16M

[mysql]
no-auto-rehash
# Remove the next comment character if you are not familiar with SQL
#safe-updates

[myisamchk]
key_buffer_size = 8M
sort_buffer_size = 8M

[mysqlhotcopy]
interactive-timeout
```

- install data-dir

```
cd /usr/local/mysql/
scripts/mysql_install_db --user=iamslash --datadir=/Users/iamslash/var/mysql1_db
scripts/mysql_install_db --user=iamslash --datadir=/Users/iamslash/var/mysql2_db
```

- run

```
mysqld_multi --defaults-file=/Users/iamslash/etc/mysql_multi.cnf start
mysqld_multi --defaults-file=/Users/iamslash/etc/mysql_multi.cnf report
mysqld_multi --defaults-file=/Users/iamslash/etc/mysql_multi.cnf stop
mysqld_multi --defaults-file=/Users/iamslash/etc/mysql_multi.cnf report
```

## XA

- distributed transaction을 위해 제공되는 기능이다. 예를 들어서 mysql
  server A, B가 있을때 A와 B에 걸쳐 transaction을 해야 하는 경우를
  생각해 보자. A, B를 묶어서 생각한 transaction을 global
  transaction이라고 하고 A, B각각 에서 이루어 지는 transaction을 local
  transaction 이라고 하자. A local transaction과 B local transaction이 
- A local transaction, B local transaction이 모두 성공하면 global
  transaction 을 commit하고 둘중 하나라도 성공하지 못했다면 global
  transaction을 rollback한다. 이때 A local transaction이 XA PREPARE를
  통해 PREPARED상태로 변하는 것을 one phase라고 하고 A local transaction이
  XA COMMIT를 통해 COMMITTED상태로 변하는 것을 two phase라고 하자.
- XA trnasaction id는 gtrid (global transaction id), bqual (branch
  qualifier), formatID로 구성된다. bqual과 formatID는 option이다. 같은
  global transaction에 속한 local transaction의 XA transaction id는
  gtrid는 같아야하고 bqual은 달라야한다. XA RECOVER를 이용하면
  PREPARED상태에 있는 local transaction의 XA transaction id를 얻어 올
  수 있다. 이것을 기반으로 local transaction성공여부를 판단하자.
  
![](Xa-diag-768x576.jpg)

- 보다 자세한 내용은 [이곳](http://mysqlserverteam.com/improvements-to-xa-support-in-mysql-5-7)을 참고하자.
  [이곳](https://www.percona.com/live/mysql-conference-2013/sites/default/files/slides/XA_final.pdf)은 distributed transaction이 성능이 좋지 않으니 최대한 피하자는 내용의 문서이다.
  혹시나 해서 [pdf](XA_final.pdf)으로 저장
  
- 다음은 XA문법이다.

```
mysql -h localhost -u iamslash -p
> XA START 'xa1';
> UPDATE t1 SET fld1=10;
> XA RECOVER;
> XA END 'xa1';
> XA PREPARE 'xa1';
> XA RECOVER;
> XA ROLLBACK 'xa1'; //XA COMMIT
```

- 다음은 python으로 XA를 이용해서 distributed transaction을
  구현해보았다. XA RECOVER를 실행하면 PREPARED상태의 XA transaction
  id들을 확인할수 있다. 검색해서 내가 수행하고 있는 XA transaction
  id가 PREPARED상태인지 확인해야 한다. exception이 발생하면
  ROLLBACK처리가 중요하다. 그렇지 않으면 XA RECOVER했을때 남아 있게
  된다. 

- a.sql
```
CREATE TABLE `users` (
    `id` int(11) NOT NULL AUTO_INCREMENT,
    `email` varchar(255) COLLATE utf8_bin NOT NULL,
    `password` varchar(255) COLLATE utf8_bin NOT NULL,
    PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin
AUTO_INCREMENT=1 ;
```

```bash
mysql -h localhost -u iamslash -p test < a.sql
pip install pymysql
```

```python
import pymysql

def OpenConnection(n_port=3306):
    return pymysql.connect(host='localhost',
                        port=n_port,
                        user='iamslash',
                        password='12345678',
                        db='test',
                        charset='utf8mb4',
                        cursorclass=pymysql.cursors.DictCursor)   

def Foo(c):
    try:
        with c.cursor() as cursor:
            # Create a new record
            sql = "INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)"
            cursor.execute(sql, ('webmaster@python.org', 'very-secret'))

        # connection is not autocommit by default. So you must commit to save
        # your changes.
        c.commit()

        with c.cursor() as cursor:
            # Read a single record
            sql = "SELECT `id`, `password` FROM `users` WHERE `email`=%s"
            cursor.execute(sql, ('webmaster@python.org',))
            result = cursor.fetchone()
            print(result)
    except Exception as e:
        print("ERR: exception " + e.value)
    finally:
        c.close()

def Bar(c, xa_id):
    """
    one phase commit exapmle using XA
    """
    try:
        with c.cursor() as cursor:
            cursor.execute("XA START %s", (xa_id))
            # Create a new record
            sql = "INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)"
            cursor.execute(sql, ('webmaster@python.org', 'very-secret'))          
            cursor.execute("XA END %s", (xa_id))
            cursor.execute("XA PREPARE %s", (xa_id))
            cursor.execute("XA RECOVER")
            result = cursor.fetchone()
            print(result)
            cursor.execute("XA COMMIT %s", (xa_id))
            #cursor.execute("XA ROLLBACK %s", (xa_id))
    except Exception as e:
        print("ERR: exception " + e)
    finally:
        c.close()

def CheckXaId(l, xa_id):
    for d in l:
        if d["data"] == xa_id:
            return True
    return False

def Baz(c0, xa_id_0, c1, xa_id_1):
    """
    two phase commit exapmle using XA
    """
    try:
        result_xa_id_0 = None
        result_xa_id_1 = None

        with c0.cursor() as cursor0, c1.cursor() as cursor1:
            cursor0.execute("XA START %s", (xa_id_0))
            cursor0.execute("INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)", ('webmaster@python.org', 'very-secret'))          
            cursor0.execute("XA END %s", (xa_id_0))
            cursor0.execute("XA PREPARE %s", (xa_id_0))
            cursor0.execute("XA RECOVER")
            result_xa_id_0 = cursor0.fetchall()
            print(result_xa_id_0)

            cursor1.execute("XA START %s", (xa_id_1))
            cursor1.execute("INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)", ('webmaster@python.org', 'very-secret'))          
            cursor1.execute("XA END %s", (xa_id_1))
            cursor1.execute("XA PREPARE %s", (xa_id_1))
            cursor1.execute("XA RECOVER")
            result_xa_id_1 = cursor1.fetchall()
            print(result_xa_id_1)

            if (result_xa_id_0 != None and result_xa_id_1 != None and
                CheckXaId(result_xa_id_0, xa_id_0) == True and CheckXaId(result_xa_id_1, xa_id_1) == True):
                cursor0.execute("XA COMMIT %s", (xa_id_0))
                cursor1.execute("XA COMMIT %s", (xa_id_1))
            else:
                cursor0.execute("XA ROLLBACK %s", (xa_id_0))
                cursor1.execute("XA ROLLBACK %s", (xa_id_1))

    except Exception as e:
        import traceback, sys
        print("ERR: exception")
        traceback.print_exc(file=sys.stdout)
        with c0.cursor() as cursor0, c1.cursor() as cursor1:
            cursor0.execute("XA ROLLBACK %s", (xa_id_0))
            cursor1.execute("XA ROLLBACK %s", (xa_id_1))
    finally:
        c0.close()
        c1.close()

if __name__ == '__main__':    
    #Foo(OpenConnection())
    #Bar(OpenConnection(), 'xa-1')
    Baz(OpenConnection(), 'xa-7', OpenConnection(), 'xa-8')
```
