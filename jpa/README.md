# Abstract

java 에서 DB 에 access 할 때 사용할 API

# Materials

* [Java Hibernate Tutorial @ youtube](https://www.youtube.com/watch?v=_7BuLOCRJc4)

# Basic

## Pessimistic Locking

* [Pessimistic Locking in JPA](https://www.baeldung.com/jpa-pessimistic-locking)
* [MySQL InnoDB lock & deadlock 이해하기](https://www.letmecompile.com/mysql-innodb-lock-deadlock/)
  
----

isolation level 은 DB connection 이 생성될 때 설정된다. 그리고 모든 statement 의 영향을 준다. 그러나 pessimistic locking 은 isolation level 보다 융통성이 있다.

Lock 을 얻기 위한 방법은 exclusive lock, shared lock 과 같이 2 가지가 있다. 만약 다른 사람이 shared lock 을 가지고 있다면 우리는 read 할 수는 있지만 write 할 수는 없다. 만약 write 하고 싶다면 exclusive lock 이 필요하다. 

`SELECT ... LOCK IN SHARE MODE` 는 shared lock 을 획득하는 방법이다. `SELECT ... FOR UPDATE` 는 exclusive lock 을 획득하는 방법이다.

JPA 의 Pessimistic Locking 의 종류는 다음과 같이 3 가지가 있다.

* PESSIMISTIC_READ – allows us to obtain a shared lock and prevent the data from being updated or deleted
* PESSIMISTIC_WRITE – allows us to obtain an exclusive lock and prevent the data from being read, updated or deleted
* PESSIMISTIC_FORCE_INCREMENT – works like PESSIMISTIC_WRITE and it additionally increments a version attribute of a versioned entity
