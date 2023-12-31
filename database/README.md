- [Abstract](#abstract)
- [Material](#material)
- [Relational Algebra](#relational-algebra)
- [Normalization](#normalization)
- [Concurrency Problems In Transactions](#concurrency-problems-in-transactions)
- [ACID](#acid)
- [SQL](#sql)
- [SQL Optimization](#sql-optimization)
- [Index](#index)

-----

# Abstract

database 를 만들어 보자. [chidb](http://chi.cs.uchicago.edu/chidb/index.html),
[sqlite @ github](https://github.com/smparkes/sqlite) 분석부터 시작해 본다.

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
* [SQLite Internals: How The World's Most Used Database Works](https://www.compileralchemy.com/books/sqlite-internals/)

# Relational Algebra

* [ra](/ra/README.md)

# Normalization

* [Normalization](/normalization/README.md)

# Concurrency Problems In Transactions

* [SQL 트랜잭션 - 믿는 도끼에 발등 찍힌다](https://blog.sapzil.org/2017/04/01/do-not-trust-sql-transaction/)
  * non-repeatable read 를 설명한다.

----

Concurrency Problems 는 다음과 같다.

* Dirty Read
  * A transaction 이 값을 1 에서 2 로 수정하고 아직 commit 하지 않았다. B transaction 은 값을 2 로 읽어들인다. 만약 A transaction 이 rollback 되면 B transaction 은 잘못된 값 2 을 읽게 된다.

* Non-repeatable Read
  * A transaction 이 한번 읽어온다. B transaction 이 Update 한다. A transaction 이 다시 한번 읽어온다. 이때 처음 읽었던 값과 다른 값을 읽어온다.
  
    ```
    BEGIN TRAN
      SELECT SUM(Revenue) AS Total FROM Data;
      --another tran updates a row
      SELECT Revenue AS Detail FROM Data;
    COMMIT  
    ```

* Phantom Read
  * A transaction 이 한번 읽어온다. B transaction 이 insert 한다. A transaction 이 다시 한번 읽어온다. 이때 처음 읽었던 record 들에 하나 더 추가된 혹은 하나 삭제된 record 들을 읽어온다.

    ```
    BEGIN TRAN
      SELECT SUM(Revenue) AS Total FROM Data;
      --another tran inserts/deletes a row
      SELECT Revenue AS Detail FROM Data;
    COMMIT  
    ```

# ACID

- [ACID Transactions](https://redis.com/glossary/acid-transactions/)

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

[Index](/index/README.md)
