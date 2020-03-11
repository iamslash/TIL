# Materials

* [스프링 프레임워크 핵심 기술 @ inflearn](https://www.inflearn.com/course/spring-framework_core/)
* [자바 ORM 표준 JPA 프로그래밍:스프링 데이터 예제 프로젝트로 배우는 전자정부 표준 데이터베이스 프레임 - 김영한](https://www.coupang.com/vp/products/20488571?itemId=80660090&vendorItemId=3314421212&q=%EA%B9%80%EC%98%81%ED%95%9C+JPA&itemsCount=4&searchId=13ac45f1095144b5bd41dfc0783f0478&rank=0&isAddedCart=)
  * [src](https://github.com/holyeye/jpabook)

# Basics

## RDBMS and Java

* dependency

```
<dependency>
  <groupId>org.postgresql</groupId>
<artifactId>postgre
```

* run postgres

```bash
$ docker run -p 5432:5432 -e POSTGRES_PASSWORD=xxxx -e POSTGRES_USER=iamslash -e POSTGRES_DB=basicdb --name my-postgres -d postgres

$ docker exec -i -t my-postgres

$ su - postgres

$ psql basicdb
\list
\dt
SELECT * FROM account;
```

* Java

```java
public class Appliation {
  public static void main(String[] args) throws SQLException {
    String url = "jdbc:postgresql://localhost:5432/basicdb";
    String username = "iamslash";
    String password = "xxxx";

    try (Connection connection = DriverManager.getConnection(url, username, password)) {
      System.out.println("Connection created: " + connection);
      String sql = "INSERT INTO ACCOUNT VALUES(1, 'iamslash', 'xxxx')";
      try (PreparedStatement statement = connection.prepareStatement(
        statement.execute());)
    }
  }
}
```

* Cons
  * Have to handle connection pools.
  * SQL is different depends on RDMBS server.
  * It's not easy to use lazy query.

## ORM

* Using Domain models

```java
Account account = new Account("iamslash", "xxxx");
accountRepository.save(account);
```

* Pros
  * Can use OOP.
  * Can use design pattern.
  * Can reuse codes.

* In a nutshell, object/relational mapping is the automated (and transparent) persistence of objects in a Java application to the tables in an SQL database, using metadata that describes the mapping between the classes of the application and the schema of the SQL database.
  * Java Persistence with Hibernate, Second Edition

Using JPA is better than Using JDBC.

## JPA Programming: Setting JPA project

## JPA Programming: Entity mapping

## JPA Programming: Value type mapping

## JPA Programming: 1 to n mapping

## JPA Programming: Cascade

## JPA Programming: Fetch

## JPA Programming: Query

## Introduction of JPA

## Core concepts

# Advanced

## Introduction of JPA

## Spring Data Common: Repository

## Spring Data Common: Repository Interface

## Spring Data Common: Handling Null

## Spring Data Common: Making a query

## Spring Data Common: Async Query

## Spring Data Common: Custom Repository

## Spring Data Common: Basic Repository Customizing

## Spring Data Common: Domain Event

## Spring Data Common: QueryDSL

## Spring Data Common: Web: Web Support Features

## Spring Data Common: Web: DomainClassConverter

## Spring Data Common: Web: Pageable and Sort Parameters

## Spring Data Common: Web: HATEOAS

## Spring Data Common: Summary

## Spring Data JPA: JPA Repository

## Spring Data JPA: Saving Entity

## Spring Data JPA: Query method

## Spring Data JPA: Query method Sort

## Spring Data JPA: Named Parameter and SpEL

## Spring Data JPA: Update query method

## Spring Data JPA: EntityGraph

## Spring Data JPA: Projection

## Spring Data JPA: Specifications

## Spring Data JPA: Query by Example

## Spring Data JPA: Transaction

## Spring Data JPA: Auditing

## Spring Data JPA: Summary




