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

[Let's Build a Simple Database](https://cstack.github.io/db_tutorial/) 를 읽고 정리한다.

[chidb](http://chi.cs.uchicago.edu/chidb/index.html), [sqlite | github](https://github.com/smparkes/sqlite) 도 나쁘지 않다.

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

- [Conssurrency Problems](/dbconcurrencyprob/README.md)

# ACID

- [ACID Transactions](https://redis.com/glossary/acid-transactions/)

Transaction 이 안전하게 수행된다는 것을 보장하기 위한 성질이다. James Nicholas
"Jim" Gray 가 1970년대 말에 신뢰할 수 있는 트랜잭션 시스템의 특성들을 정의하고
개발했다.

* **Atomicity** (원자성)
  * all-or-nothing
  * Transaction 은 완전히 실행되거나 실행되지 않는 성질을 말한다. 
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
  * 성공적으로 수행된 Transaction 영원히 반영되어야 한다. 장애가 발생했을 때
    data 를 recover 할 수 있다. Transaction 은 logging 된다. 따라서 Transaction
    은 replay 될 수 있다. DataBase 가 제대로 backup 되고 있다면 언제든지 recover
    할 수 있다.

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
