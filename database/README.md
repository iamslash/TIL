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
* [The chidb File Format](http://chi.cs.uchicago.edu/chidb/fileformat.html)

----

![](physlog.png)

위의 그림을 잘 살펴보자. xdb file 은 여러개의 page 로 구성되어 있다. 하나의 page 는 하나의 BTreeNode 를 저장한다. 하나의 BTreeNode 는 하나 이상의 record 를 저장한다. 또한 하나의 page 는 여러개의 cell 로 이루어져 있다. cell 은 record 에 대응된다. 

Courses 테이블은 schema 가 `CREATE TABLE Courses(Id INTEGER PRIMARY KEY, Name TEXT, Instructor INTEGER, Dept INTEGER)` 라고 해보자. primary key 인 id 를 hasing 한 값을 key 로 BTree 를 제작한다. 당연히 Id 를 조건으로 검색하면 빠르다. 

그러나 Dept 를 조건으로 검색하면 느리다. Dept 의 인덱스를 제작한다. 즉, Dept 의 hashing 값을 key 로 B+Tree 를 제작한다. 당연히 Dept 를 조건으로 검색하면 빨라진다.