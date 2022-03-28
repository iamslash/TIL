- [Abstract](#abstract)
- [Material](#material)
- [Relational Algebra](#relational-algebra)
- [Normalization](#normalization)
- [ACID](#acid)
- [SQL](#sql)
- [SQL Optimization](#sql-optimization)
- [Index](#index)
- [Pages](#pages)
- [Clustered Index vs Non-clustered Index](#clustered-index-vs-non-clustered-index)

-----

# Abstract

database 를 만들어 보자.

# Material

* [정보처리 실기_데이터베이스 @ youtube](https://www.youtube.com/playlist?list=PLimVTOIIZt2aP6msQIw0011mfVP-oJGab)
  * 관계대수, 정규화 등등 속시원한 설명
  * 강의자료 pdf 가 너무 좋다.
* [chidb](http://chi.cs.uchicago.edu/chidb/index.html)
  * c로 제작된 database, transaction이 없다.
  * [src](https://github.com/uchicago-cs/chidb)
* [cubrid @ github](https://github.com/CUBRID/cubrid)
  * 국산 opensource rdbms
* [sqlite @ github](https://github.com/smparkes/sqlite)

# Relational Algebra

* [ra](/ra/README.md)

# Normalization

* [normalization](/normalization/README.md)

# ACID

Transaction 이 안전하게 수행된다는 것을 보장하기 위한 성질이다. James Nicholas
"Jim" Gray 가 1970년대 말에 신뢰할 수 있는 트랜잭션 시스템의 특성들을 정의하고
개발했다.

* **Atomicity** (원자성)
  * Transaction 은 완전히 실행되거나 실행되지 않는 성질을 말한다. all-or-nothing
    이라고 한다.
* **Consistency** (일관성)
  * Transaction 이 완료되면 일관성 있는 데이터베이스 상태로 유지하는 것을
    의미한다. 모든 계좌는 잔고가 있어야 하는 무결성 제약이 있다면 이를 위반하는
    Transaction 은 중단된다.
* **Isolation** (고립성)
  * Transaction 의 concurrency 가 보장되는 성질을 말한다. A 와 B 두개의
    Transaction 이 실행되고 있다고 하자. A 의 작업들이 B 에게 보여지는 정도를
    [Isolation Level](/spring/README.md#transactional) 이라고 하며 모두 4
    가지로 구성된다.

    | Isolation Level | Dirty Read | Non-repeatable Read | Phantom Read |
    | --------------- | ---------- | ------------------- | ------------ |
    | Read uncommited | O          | O                   | O            |
    | Read commited   | X          | O                   | O            |
    | Repeatable Read | X          | X                   | O            |
    | Serializable    | X          | X                   | X            |

* **Durability** (영구성)
  * 성공적으로 수행된 Transaction 영원히 반영되어야 한다. 장애가
    발생했을 때 data 를 recover 할 수 있다. Transaction 은 logging 된다. 따라서
    Transaction 은 replay 될 수 있다. DataBase 가 제대로 backup 되고 있다면
    언제든지 recover 할 수 있다.

[Isolation](/isolation/README.md) 에서 MySQL 실습을 확인할 수 있다.

# SQL

* [sql](/sql/README.md)

# SQL Optimization

chidb 의 sql 최적화 내용중 일부이다. `.opt` 는 chidbshell 에서 최적화를 수행하는 shell command 이다.
원본과 최적화 된 relational algebra 를 출력한다. 다음과 같은 경우 join 을 하고 select 를 하는 것보다
select 를 하고 join 을 하는 것이 성능이 좋다. 

일반적으로 sql 작성자의 경우 WHERE 를 join 이후에 동작한다고 생각한다. NaturalJoin 을 하기 위해서는
cartesian product 를 해야 한다. 아래와 같이 최적화 하면 cartesian product 의 대상이 줄어들어 성능이
향상된다.

```
chidb> .opt "SELECT * FROM t |><| u WHERE t.a>10;"
Project([*],
        Select(t.a > int 10,
                NaturalJoin(
                        Table(t),
                        Table(u)
                )
        )
)

Project([*],
        NaturalJoin(
                Select(t.a > int 10,
                        Table(t)
                ),
                Table(u)
        )
)
chidb>
```

# Index

* [SQL Unplugged 2013] 쉽고 재미있는 인덱스 이야기/ 씨퀄로 이장래](https://www.youtube.com/watch?v=TRfVeco4wZM)
* [RDB index 정리](https://hyungjoon6876.github.io/jlog/2018/07/18/rdb-indexing.html)
* [The chidb File Format](http://chi.cs.uchicago.edu/chidb/fileformat.html)

----

![](physlog.png)

위의 그림을 잘 살펴보자. xdb file 은 여러개의 page 로 구성되어 있다. 하나의 page 는 하나의 BTreeNode 를 저장한다. 하나의 BTreeNode 는 하나 이상의 record 를 저장한다. 또한 하나의 page 는 여러개의 cell 로 이루어져 있다. cell 은 record 에 대응된다. 

Courses 테이블은 schema 가 `CREATE TABLE Courses(Id INTEGER PRIMARY KEY, Name TEXT, Instructor INTEGER, Dept INTEGER)` 라고 해보자. primary key 인 id 를 hashing 한 값을 key 로 BTree 를 제작한다. 당연히 Id 를 조건으로 검색하면 빠르다. 

예를 들어 위 그림에서 id 를 hashing 한 값 `86` 을 검색하기 위해 BTree 를 `42->85->86` 순으로 검색에 성공했다.

그러나 Dept 를 조건으로 검색하면 느리다. Dept 의 인덱스를 제작한다. 즉, Dept 의 hashing 값을 key 로 B+Tree 를 제작한다. 당연히 Dept 를 조건으로 검색하면 빨라진다.

예를 들어 Dept 를 hashing 한 값 `42` 를 검색하기 위해 B+Tree 를 `10->42` 순으로 검색에 성공했다. `42` 는 primary key 인 ID 를 hashing 한 값 `67` 를 소유한 cell 을 가리키고 있다.

# Pages

* [How does SQLite work? Part 2: btrees! (or: disk seeks are slow don't do them!)](https://jvns.ca/blog/2014/10/02/how-does-sqlite-work-part-2-btrees/)
  * sqlite 의 database file 이 왜 pages 로 구성되어 있는지 설명한다.

----

database file 은 물리적으로 여러개의 pages 로 구성되야 한다.

CPU 는 disk 에서 데이터를 읽어들여 memeory 로 로드해야 한다. 이때 데이터의 크기는 내가 찾고자 하는
것에 가까운 최소의 크기여야 한다. 내가 찾고자하는 데이터는 16 Byte 인데 1 MB 를 메모리에 로드하는 것은
낭비이다. 

그리고 filesystem 의 block size 는 4 KB 이다. 따라서 한번에 1 MB 를 메모리에서 읽는 것은 불가능하다.
여러번에 걸쳐 disk access 를 해야 한다. I/O 낭비가 발생한다.

따라서 데이터를 page 에 저장하고 그것의 크기는 block size 보다는 작고 
너무 작지 않게 설정하여 L1, L2 Cache hit 가 이루어 지도록 해야 한다.

참고로 chidb, sqlite 는 page 의 size 가 1K 이다.

# Clustered Index vs Non-clustered Index

* [Difference between Clustered and Non-clustered index](https://www.geeksforgeeks.org/difference-between-clustered-and-non-clustered-index/)

----

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
