- [Materials](#materials)
- [Basics](#basics)
  - [RDBMS and Java](#rdbms-and-java)
  - [ORM](#orm)
  - [JPA Programming: Setting JPA project](#jpa-programming-setting-jpa-project)
  - [JPA Programming: Entity mapping](#jpa-programming-entity-mapping)
  - [JPA Programming: Value type mapping](#jpa-programming-value-type-mapping)
  - [JPA Programming: 1 to n mapping](#jpa-programming-1-to-n-mapping)
  - [JPA Programming: Cascade](#jpa-programming-cascade)
  - [JPA Programming: Fetch](#jpa-programming-fetch)
  - [JPA Programming: Query](#jpa-programming-query)
  - [Introduction of JPA](#introduction-of-jpa)
  - [Core concepts](#core-concepts)
- [Advanced](#advanced)
  - [Introduction of JPA](#introduction-of-jpa-1)
  - [Spring Data Common: Repository](#spring-data-common-repository)
  - [Spring Data Common: Repository Interface](#spring-data-common-repository-interface)
  - [Spring Data Common: Handling Null](#spring-data-common-handling-null)
  - [Spring Data Common: Making a query](#spring-data-common-making-a-query)
  - [Spring Data Common: Async Query](#spring-data-common-async-query)
  - [Spring Data Common: Custom Repository](#spring-data-common-custom-repository)
  - [Spring Data Common: Basic Repository Customizing](#spring-data-common-basic-repository-customizing)
  - [Spring Data Common: Domain Event](#spring-data-common-domain-event)
  - [Spring Data Common: QueryDSL](#spring-data-common-querydsl)
  - [Spring Data Common: Web: Web Support Features](#spring-data-common-web-web-support-features)
  - [Spring Data Common: Web: DomainClassConverter](#spring-data-common-web-domainclassconverter)
  - [Spring Data Common: Web: Pageable and Sort Parameters](#spring-data-common-web-pageable-and-sort-parameters)
  - [Spring Data Common: Web: HATEOAS](#spring-data-common-web-hateoas)
  - [Spring Data JPA: JPA Repository](#spring-data-jpa-jpa-repository)
  - [Spring Data JPA: Saving Entity](#spring-data-jpa-saving-entity)
  - [Spring Data JPA: Query method](#spring-data-jpa-query-method)
  - [Spring Data JPA: Query method Sort](#spring-data-jpa-query-method-sort)
  - [Spring Data JPA: Named Parameter and SpEL](#spring-data-jpa-named-parameter-and-spel)
  - [Spring Data JPA: Update query method](#spring-data-jpa-update-query-method)
  - [Spring Data JPA: EntityGraph](#spring-data-jpa-entitygraph)
  - [Spring Data JPA: Projection](#spring-data-jpa-projection)
  - [Spring Data JPA: Specifications](#spring-data-jpa-specifications)
  - [Spring Data JPA: Query by Example](#spring-data-jpa-query-by-example)
  - [Spring Data JPA: Transaction](#spring-data-jpa-transaction)
  - [Spring Data JPA: Auditing](#spring-data-jpa-auditing)
  - [JPA Cache](#jpa-cache)

----

# Materials

* [Spring Data JPA - Reference Documentation](https://docs.spring.io/spring-data/jpa/docs/current/reference/html/#reference)
* [스프링 프레임워크 핵심 기술 @ inflearn](https://www.inflearn.com/course/spring-framework_core/)
* [자바 ORM 표준 JPA 프로그래밍:스프링 데이터 예제 프로젝트로 배우는 전자정부 표준 데이터베이스 프레임 - 김영한](https://www.coupang.com/vp/products/20488571?itemId=80660090&vendorItemId=3314421212&q=%EA%B9%80%EC%98%81%ED%95%9C+JPA&itemsCount=4&searchId=13ac45f1095144b5bd41dfc0783f0478&rank=0&isAddedCart=)
  * [src](https://github.com/holyeye/jpabook)

# Basics

## RDBMS and Java

* dependency

```xml
<dependency>
  <groupId>org.postgresql</groupId>
<artifactId>postgre
```

* run postgres

```bash
$ docker run -p 5432:5432 -e POSTGRES_PASSWORD=1 -e POSTGRES_USER=iamslash -e POSTGRES_DB=iamslash --name my-postgres -d postgres

$ docker exec -i -t my-postgres

$ su - postgres

$ psql iamslash
\list
\dt
SELECT * FROM account;
```

* Java

```java
public class Appliation {
  public static void main(String[] args) throws SQLException {
    String url = "jdbc:postgresql://localhost:5432/iamslash";
    String username = "iamslash";
    String password = "1";

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

> * [데이터베이스 초기화 @ TIL](https://github.com/iamslash/TIL/blob/3833c3dba5db0c33551741d9c9b8cc09954b86e2/spring/SpringBoot.md#%EC%8A%A4%ED%94%84%EB%A7%81-%EB%8D%B0%EC%9D%B4%ED%84%B0-7-%EB%B6%80-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4-%EC%B4%88%EA%B8%B0%ED%99%94)

> * [Show Hibernate/JPA SQL Statements from Spring Boot @ baeldung](https://www.baeldung.com/sql-logging-spring-boot)

```conf
spring.jpa.show-sql=true
spring.jpa.properties.hibernate.format_sql=true
```

## JPA Programming: Entity mapping

> * [Defining JPA Entities @ baeldung](https://www.baeldung.com/jpa-entities)

----

> `@Entity`

This is nothing but POJOs representing data that can be persisted to the database. 

```java
@Entity
public class Student {    
}
```

> `@Table`

the name of the table in the database and the name of the entity will not be the same.

```java
@Entity
@Table(name="STUDENT")
public class Student {    
}
```

> `@Id`

the primary key

> `@GeneratedValue`

We can generate the identifiers in different ways which are specified by the @GeneratedValue annotation.

```java
@Entity
public class Student {
    @Id
    @GeneratedValue(strategy=GenerationType.AUTO)
    private Long id;    
    private String name;
}
```

> `@Column`

the details of a column in the table.

```java
@Entity
@Table(name="STUDENT")
public class Student {
    @Id
    @GeneratedValue(strategy=GenerationType.AUTO)
    private Long id;
    
    @Column(name="STUDENT_NAME", length=50, nullable=false, unique=false)
    private String name;
}
```

> `@Temporal`

we may have to save temporal values in our table

```java
@Entity
@Table(name="STUDENT")
public class Student {
    @Id
    @GeneratedValue(strategy=GenerationType.AUTO)
    private Long id;
    
    @Column(name="STUDENT_NAME", length=50, nullable=false, unique=false)
    private String name;
    
    @Transient
    private Integer age;
    
    @Temporal(TemporalType.DATE)
    private Date birthDate;
}
```

> `@Transient`

It specifies that the field will not be persisted.

```java
@Entity
@Table(name="STUDENT")
public class Student {
    @Id
    @GeneratedValue(strategy=GenerationType.AUTO)
    private Long id;
    
    @Column(name="STUDENT_NAME", length=50, nullable=false)
    private String name;
    
    @Transient
    private Integer age;
}
```

> application.properties

```conf
spring.jp.show-sql=true
spring.jpa.properties.hibernate.format_sql=true
```

## JPA Programming: Value type mapping

> * [JPA @Embedded And @Embeddable @ baeldung](https://www.baeldung.com/jpa-embedded-embeddable)

----

> `@Embeddable`

a class will be embedded by other entities.

```java
@Embeddable
public class ContactPerson {
    private String firstName;
    private String lastName;
    private String phone;
}
```

> `@Embedded`

embed a type into another entity.

```java
@Entity
public class Company {

    @Id
    @GeneratedValue
    private Integer id;

    private String name;

    private String address;

    private String phone;

    @Embedded
    private ContactPerson contactPerson;
}
```

> `@AttributeOverrides, @AttributeOverride`

override the column properties of our embedded type.

```java
@Embedded
@AttributeOverrides({
  @AttributeOverride( name = "firstName", column = @Column(name = "contact_first_name")),
  @AttributeOverride( name = "lastName", column = @Column(name = "contact_last_name")),
  @AttributeOverride( name = "phone", column = @Column(name = "contact_phone"))
})
private ContactPerson contactPerson;
```

## JPA Programming: 1 to n mapping

> * [[JPA] @ManyToMany, 다대다[N:M] 관계 @ tistory](https://ict-nroo.tistory.com/127)
> * [Many-To-Many Relationship in JPA @ baeldung](https://www.baeldung.com/jpa-many-to-many)
> * [Hibernate One to Many Annotation Tutorial @ baeldung](https://www.baeldung.com/hibernate-one-to-many)

----------

Many to Many relationship 은 anomolies 를 발생시킨다. 두 테이블 사이에 relation table 을 만들어서 Many to one, one to many 로 해결해야 한다. JPA 의 `@JoinTable` 을 이용해서 relation table 을 generate 할 수 있다. 그러나 JPA 가 만들어주는 naming convention 보다는 manual 하게 table 을 만들 것을 추천한다. 서비스를 운영하다 보면 직접 table 에 operation 을 할 수도 있기 때문이다.

This is an example of schema.

```sql
CREATE TABLE `Cart` (
  `cart_id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  PRIMARY KEY (`cart_id`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8;

CREATE TABLE `Items` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `cart_id` int(11) unsigned NOT NULL,
  PRIMARY KEY (`id`),
  KEY `cart_id` (`cart_id`),
  CONSTRAINT `items_ibfk_1` FOREIGN KEY (`cart_id`) REFERENCES `Cart` (`cart_id`)
) ENGINE=InnoDB AUTO_INCREMENT=7 DEFAULT CHARSET=utf8;
```

This is an example of `@OneToMany`

 `@OneToMany` annotation is used to define the property in Items class that will be used to map the mappedBy variable. That is why we have a property named `cart` in the Items class

 It's also important to note that the `@ManyToOne` annotation is associated with the Cart class variable. `@JoinColumn` annotation references the mapped column.

```java
@Entity
@Table(name="CART")
public class Cart {

    //...
    @OneToMany(mappedBy="cart")
    private Set<Items> items;	
    // getters and setters
}
...
@Entity
@Table(name="ITEMS")
public class Items {
    
    //...
    @ManyToOne
    @JoinColumn(name="cart_id", nullable=false)
    private Cart cart;

    public Items() {}    
    // getters and setters
}
```

## JPA Programming: Cascade

* [Overview of JPA/Hibernate Cascade Types @ baeldung](https://www.baeldung.com/jpa-cascade-types)
* [JPA’s 4 Lifecycle States](https://thorben-janssen.com/entity-lifecycle-model/)

----

When we perform some action on the target entity, the same action will be applied to the associated entity.

* ALL
* PERSIST
* MERGE
* REMOVE
* REFRESH
* DETACH

![](https://thorben-janssen.com/wp-content/uploads/2020/07/Lifecycle-Model-1024x576.webp)

## JPA Programming: Fetch

* `@OneToMany`: Lazy Fetch
* `@ManyToOne`: Eager Fetch

## JPA Programming: Query

## Introduction of JPA

* [Introduction to Spring Data JPA @ baeldung](https://www.baeldung.com/the-persistence-layer-with-spring-data-jpa)

-----

`@EnableJpaRepositories` makes the Spring JPA repository support and specify the
package that contains the DAO interfaces.

`@EnableJpaRepositories` 

```java
@EnableJpaRepositories(basePackages = "com.baeldung.spring.data.persistence.repository") 
public class PersistenceConfig { 
}
```

`@Repository` 가 없어도 Bean 으로 등록해준다.

```java
public interface IFooDAO extends JpaRepository<Foo, Long> {
    Foo findByName(String name);
}
```

## Core concepts

This is an application.properties for debugging.

```conf
# application.properties
logging.level.org.hibernate.SQL=debug
logging.level.org.hibernate.type.descriptor.sql=trace
```

# Advanced

## Introduction of JPA

## Spring Data Common: Repository

`@NoRepositoryBean` 은 Repository Bean 이지만 Bean 으로 등록하지 않기 위해
사용한다. 주로 `Respository` 를 상속한 중간의 Repository interface 들에
부착한다. 참고로 `Respository` interface 는 Marker interface 이다.

```java
// com.iamslash.exjpa.AppRunner
@Component
@Transactional
public class AppRunner implements ApplicationRunner {
  @Autowired
  PostRepository postRepository;

  @Override
  public void run(ApplicationArguments args) throws Exception {
    Post post = new Post();
    post.setTitle("spring");

    Comment comment = new Comment();
    comment.setComment("hello");

    postRepository.save(post);
  }
}

// com.iamslash.exjpa.PostRepository
public interface PostRepository extends JpaRepository<Post, Long> {
...
}

// org\springframework\data\jpa\repository\JpaRepository.java
@NoRepositoryBean
public interface JpaRepository<T, ID> extends PagingAndSortingRepository<T, ID>, QueryByExampleExecutor<T> {
...  
}

// org\springframework\data\repository\PagingAndSortingRepository.java
@NoRepositoryBean
public interface PagingAndSortingRepository<T, ID> extends CrudRepository<T, ID> {
...
}

// org\springframework\data\repository\CrudRepository.java
@NoRepositoryBean
public interface CrudRepository<T, ID> extends Repository<T, ID> {
...
}

// org\springframework\data\repository\Repository.java
@Indexed
public interface Repository<T, ID> {
...  
}
```

다음은 `PostRepository` 의 unit test 구현이다. `@DataJpaTest` 를 사용하면 in
memory db 인 `h2` 를 사용한다. `build.gradle` 의 dependency 에 `h2` 를 추가해야
한다.

```java
// src/test/java/com.iamslash.exjpa.ExjpaApplication.PostRepositoryTest.java
@RunWith(SpringRunner.class)
@DataJpaTest
public class PostRepositoryTest {
  
  @Autowired
  PostRepository postRepository;

  @Test
  public void crudRepository() {
    // Given
    Post pos = new Post();
    post.setTitle("Hello world");
    assertThat(post.getId()).isNull();

    // When
    Post newPost = postRepository.save(post);

    // Then
    assertThat(newPost.getId()).isNotNull();

    // When
    List<Post> posts = postRepository.findAll();

    // Then
    assertThat(posts.size()).is(1);
    assertThat(posts).contains(newPost);

    // When
    Page<Post> page = postRepository.findAll(PageRequest.of(0, 10));
    assertThat(page.getTotalElements()).isEqualTo(1);
    assertThat(page.getNumber()).isEqualsTo(0);
    assertThat(page.getSize()).isEqualTo(10);
    assertThat(page.getNumberOfElements()).isEqualsTo(1);

    // When
    page = postRepository.findByTitleContains("spring", PageRequest.of(0, 10));
    // Then
    assertThat(page.getTotalElements()).isEqualTo(1);
    assertThat(page.getNumber()).isEqualsTo(0);
    assertThat(page.getSize()).isEqualTo(10);
    assertThat(page.getNumberOfElements()).isEqualsTo(1);

    // When
    long spring = postRepository.countByTitleContains("spring");
    // Then
    assertThat(spring).isEqualTo(1);
  }
}
```

위에서 사용한 `findByTitleContains` 는 `PostRepository` 에 다음과 같이 선언해야
한다. JPA 는 method name 을 보고 query 를 생성해 준다.

```java
public interface PostRepository extends JpaRepository<Post, Long> {
  Page<Post> findByTitleContains(String title, Pageable pageable);
  long countByTitleContains(String title);
}
```

## Spring Data Common: Repository Interface

`JpaRepository` 가 제공하는 method 들을 모두 사용하지 않고 꼭 필요한 method 만
사용하고 싶다. `JpaRepository` 를 상속하지 않고 Repository interface 를 제작해
보자.

```java
// src/main/java/com.iamslash.exjpa/CommentRepository.java
@RepositoryDefinition(domainClass = Comment.class, idClass = Long.class)
public interface CommentRepository {
  Comment save(Comment comment);
  List<Comment> findAll();
}

// src/test/java/com.iamslash.exjpa/CommentRepositoryTest.java
@RunWith(SpringRunner.class)
@DataJpaTest
public class CommentRepositoryTest {

  @Autowired
  CommentRepository commentRepository;

  @Test
  public void crud() {
    Comment comment = new Comment();
    comment.setComment("Hello Comment");
    commentRepository.save(comment);

    List<Comment> all = commentRepository.findAll();
    assertThat(all.size()).isEqualTo(1);
  }
}
```

이번에는 Repository interface 를 만들고 그것을 상속하는 CommentRepository 를
제작해 보자.

```java
// src/main/java/com.iamslash.exjpa/MyRepository
@NoRepositoryBean
public interface MyRepository<T, Id extends Serializable> extends Repository<T, Id> {
  <E extends T> E save(E entity);
  List<T> findAll();
  long count();
}

// src/main/java/com.iamslash.exjpa/CommentRepository
public interface CommentRepository implement MyRepository {
}

// src/test/java/com.iamslash.exjpa/CommentRepositoryTest.java
@RunWith(SpringRunner.class)
@DataJpaTest
public class CommentRepositoryTest {

  @Autowired
  CommentRepository commentRepository;

  @Test
  public void crud() {
    Comment comment = new Comment();
    comment.setComment("Hello Comment");
    commentRepository.save(comment);

    List<Comment> all = commentRepository.findAll();
    assertThat(all.size()).isEqualTo(1);

    long count = commentRepository.findAll();
    assertThat(count).isEqualTo(1);
  }
}
```

## Spring Data Common: Handling Null

`@NonNull, @Nullable` 를 이용하면 annotation 만으로 Null 을 check 하는 코드를
생성할 수 있다.

```java
// src/main/java/com.iamslash.exjpa/MyRepository
@NoRepositoryBean
public interface MyRepository<T, Id extends Serializable> extends Repository<T, Id> {
  <E extends T> E save(@NonNull E entity);
  
  List<T> findAll();
  
  long count();
  
  @Nullable
  <E extends T> E findById(Id id);
}
```

IntelliJ 에서 `@NonNull, @Nullable` 에 대한 intelli sense 를 원한다면 다음과 같이 spring 의 nonnull, nonnullable 을 추가한다.

![](img/preferences_compile.png)

![](img/preferences_compile_configuration.png)

## Spring Data Common: Making a query

Spring Data 가 Repository interface 의 Method 이름을 읽고 Query 를 생성한다. 만약 Method 이름에 문제가 있다면 실행할 때 error 가 발생할 것이다. 반드시 Repository interface 의 unit test 를 만들어 method 이름에 문제가 없는지 검증한다.

```java
// src/main/java/com.iamslash.exjpa/CommentRepository
public interface CommentRepository extends MyRepository<Comment, Long> {
  @Query("SELECT c FROM Comment AS c", nativeQuery = true)
  List<Comment> findByCommentContains(String keyword);

  Page<Comment> findbyLikeCountGreaterThanAndPost(int likeCount, Post post, Pageable pageable)
}
...
//src/main/java/com.iamslash.exjpa/ExjpaApplication.java
@SpringBootApplication
@EnableJpaRepositories(queryLookupStrategy = QueryLookupStrategy.Key.CREATE)
publi class ExjpaApplication {
  public static void main(String[] args) {
    SpringApplication.run(ExjpaApplication.class);
  }
}
```

Query 를 선택하는 전략은 다음과 같이 3 가지가 있다. `@EnableJpaRepositories` 에
`queryLookupStrategy` 의 value 로 설정할 수 있다.

* CREATE
* USE_DECLARED_QUERY
* CREATE_IF_NOT_FOUND (default)

Query Method 의 형식은 다음과 같다.

```
리턴타입 {접두어}{도입부}By{프로퍼티 표현식}{조건식}
  [{And|Or}{프로퍼티 표현식}{조건식}]{정렬조건}{매개변수}
```

| Part | Examples |
|----|----|
| 접두어 | Find, Get, Query, Count, etc... |
| 도입부 | Distinct, First(N), Top(N) |
| 프로퍼티 표현식 | Person, Address, ZipCode => find{Person}ByAddress_ZipCode(...) |
| 조건식 | IgnoreCase, Between, LessThan, GreaterThan, LIke, Contains, etc... |
| 정렬조건 | OrderBy{프로퍼티}Asc|Desc |
| 리턴타입 | E, Optional<E>, List<E>, Page<E>, Slice<E>, Stream<E> |
| 매개변수 | Pageable, Sort |

다음은 Query Method 의 예이다.

```java
/// basic
List<Person> findByEmailAddressAndLastname(EmailAddress emailAddress, String lastname);
// distinct
List<Person> findDistinctPeopleByLastnameOrFirstname(String lastname, String firstname);
List<Person> findPeopleDistinctByLastnameOrFirstname(String lastname, String firstname);
// ignoring case
List<Person> findByLastnameIgnoreCase(String lastname);
// ignoring case
List<Person> findByLastnameAndFirstnameAllIgnoreCase(String lastname, String firstname);

/// sort
List<Person> findByLastnameOrderByFirstnameAsc(String lastname);
List<Person> findByLastnameOrderByFirstnameDesc(String lastname);

/// page
Page<User> findByLastname(String lastname, Pageable pageable);
Slice<User> findByLastname(String lastname, Pageable pageable);
List<User> findByLastname(String lastname, Sort sort);
List<User> findByLastname(String lastname, Pageable pageable);

/// stream
// try-with-resource 를 사용하자.
// stream 은 사용을 마치고 close() 해야 한다.
Stream<User> readAllByFirstnameNotNull();
```

`Page` 는 `Slice` 를 extend 한다. `Slice` 는 바로 이전 혹은 다음 데이터 모음이 있는지 알 수 있다. `Page` 는 전체 데이터가 몇개의  모음인지 알수 있다.

다음은 Query Method 의 basic unit test 일부이다.

```java
// test/java/com.iamslash.exjpa/CommentRepositoryTest.java
import java.util.*;
import static org.assertj.core.api.Assertions.assertThat;

@RunWith(SpringRunner.class)
@DataJpaTest
public class CommentRepositoryTest {
  
  @Autowired
  CommentRepository commentRepository;

  @Test
  public void crud() {
    // Given
    Comment comment = new Comment();
    comment.setLikeCount(1);
    comment.setComment("spring data jpa");
    commentRepository.save(comment);

    // Then
    List<Comment> comments = commentRepository.findByCommentContainsIgnoreCase("Spring");
    assertThat(comments.size()).isEqualTo(0);
  }
}
```

다음은 Query Method 의 page unit test 일부이다.

```java
import java.util.*;
import static org.assertj.core.api.Assertions.assertThat;

@RunWith(SpringRunner.class)
@DataJpaTest
public class CommentRepositoryTest {
  
  @Autowired
  CommentRepository commentRepository;

  @Test
  public void crud() {
    this.createComment(100, "spring data jpa");
    this.createComment(55, "HIBERNATE SPRING");

    PageRequest pageRequest = PageRequest.of(0, 10, Sort.Direction.DESC, "LikeCount");

    Page<Comment> comments = commentRepository.findByCommentContainsIgnoreCase()
    assertThat(comments.getNumberOfElements()).isEqualTo(2);
    assertThat(comments).first().hasFieldOrPropertyWithValue("likeCount", 55);
  }

  private void createComment(int likeCount, String comment) {
    // Given
    Comment newComment = new Comment();
    comment.setLikeCount(likecount);
    comment.setComment(comment);
    commentRepository.save(newComment);
  }
}
```

다음은 Query Method 의 stream unit test 일부이다.

```java
import java.util.*;
import static org.assertj.core.api.Assertions.assertThat;

@RunWith(SpringRunner.class)
@DataJpaTest
public class CommentRepositoryTest {
  
  @Autowired
  CommentRepository commentRepository;

  @Test
  public void crud() {
    this.createComment(100, "spring data jpa");
    this.createComment(55, "HIBERNATE SPRING");

    PageRequest pageRequest = PageRequest.of(0, 10, Sort.Direction.DESC, "LikeCount");

    try (Stream<Comment> comments = commentRepository.findByCommentContainsIgnoreCase("Spring", pageRequest)) {
      Comment firstComment = comment.findFirst().get();
      assertThat(firstComment.getLikeCount()).isEqualTo(100);
    }
  }

  private void createComment(int likeCount, String comment) {
    // Given
    Comment newComment = new Comment();
    comment.setLikeCount(likecount);
    comment.setComment(comment);
    commentRepository.save(newComment);
  }
}
```

## Spring Data Common: Async Query

`@Async` 를 부착하여 Async Query 를 사용할 수 있다. 그러나 추천하지 않는다.

## Spring Data Common: Custom Repository

다음과 같이 `application.properties` 를 설정한다.

```conf
spring.datasource.url=jdbc:postgresql://localhost:5432/iamslash
spring.datasource.username=iamslash
spring.datasource.password=1

spring.jpa.hibernate.ddl-auto=update
spring.jpa.properties.hibernate.jdbc.lob.non_contextual_create=true

spring.jpa.show-sql=true
spring.jpa.properties.hibernate.format_sql=true
logging.level.org.hibernate.type.descriptor.sql=trace
```

다음과 같이 `Post` Entity Class 를 제작한다.

```java
// src/main/java/com.iamslash.exjpa/Post.java
@Entity
public class Post {
  @Id @GeneratedValue
  private Long id;

  private String title;

  // length is greater than 255
  @Lob
  private String content;

  @Temporal(TemporalType.TIMESTAMP)
  private Date created;

  // getters, setters
  ...
}
```

다음과 같이 `PostRepository` Repository interface 를 제작한다.

```java
// src/main/java/com.iamslash.exjpa/PostRepository.java
public interface PostRespository extends JpaRepository<Post, Long> {

}
```

다음과 같이 `PostRepositoryTest` Repository test class 를 제작한다. Test class 에서 `spring.jpa.show-sql=true` 는 기본이다.

```java
// src/test/java/com.iamslash.exjpa/PostRepositoryTest.java
@RunWith(SpringRunner.class)
@DataJpaTest
public class PostRepositoryTest {
  @Autowired
  PostRepository postRepository;

  @Test
  public void crud() {

  }
}
```

다음과 같이 `PostCustomRepository` Custom repository interface 를 제작한다.

```java
// src/main/java/com.iamslash.exjpa/PostCustomRepository.java
public interface PostCustomRespository {

}
```

다음과 같이 `PostCustomRepositoryImpl` Custom repositoryImpl class 를 제작한다. `Impl` 이라는 suffix 를 사용해야 한다.

```java
// src/main/java/com.iamslash.exjpa/PostCustomRepository.java
@Repository
@Transactional
public interface PostCustomRespositoryImpl implements PostCustomRepository {

  @Autowired
  EntityManager entityManager;

  @Override
  public List<Post> findMyPost() {
    System.out.println("custom findMyPost");
    return entityManager.createQuery("SELECT p FROM Post AS p", Post.class).getResultList();
  }
}
```

이제 `PostRepository` Repository interface 를 수정하여 PostCustomRepository 를 상속하자.

```java
// src/main/java/com.iamslash.exjpa/PostRepository.java
public interface PostRespository extends JpaRepository<Post, Long>, PostCustomRepository {

}
```

`PostRepositoryTest` Repository test class 에서 `PostCustomRepository` 를 사용해 보자.

```java
// src/test/java/com.iamslash.exjpa/PostRepositoryTest.java
@RunWith(SpringRunner.class)
@DataJpaTest
public class PostRepositoryTest {
  @Autowired
  PostRepository postRepository;

  @Test
  public void crud() {
    postRepository.findMyPost();
  }
}
```

한편 `SimpleJpaRepository.java` 를 확인하면 `delete` 이 비효율적으로 구현되어 있다. entity 가 entityManager 에 없다면 굳이 merge 하고 remove 하는 것은 비효율적이다.

```java
@Transactional
public void delete(T entity) {
  Assert.notNull(entity, "The entity must not be null");
  em.remove(em.contains(entity) ? entity : em.merge(entity));
}
```

다음과 같이 delete 를 개선해 보자. 

```java
// src/main/java/com.iamslash.exjpa/PostCustomRepository.java
public interface PostCustomRespository<T> {
  
  List<Post> findMyPost();

  void delete(T entity);
}

// src/main/java/com.iamslash.exjpa/PostCustomRepository.java
@Repository
@Transactional
public interface PostCustomRespositoryImpl implements PostCustomRepository {

  @Autowired
  EntityManager entityManager;

  @Override
  public List<Post> findMyPost() {
    System.out.println("custom findMyPost");
    return entityManager.createQuery("SELECT p FROM Post AS p", Post.class).getResultList();
  }

  @Override
  public void delete(Post entity) {
    System.out.println("custom delete");
    entityManager.remove(entity);
  }
}

// src/test/java/com.iamslash.exjpa/PostRepositoryTest.java
@RunWith(SpringRunner.class)
@DataJpaTest
public class PostRepositoryTest {

  @Autowired
  PostRepository postRepository;

  @Test
  public void crud() {
    postRepository.findMyPost();
    Post pos = new Post();
    post.setTitle("Hello World");
    postRepository.save(post);
    postRepositoty.findMyPost(); // send Insert Query

    postRepositoty.delete(post);
    postRepositoty.flush(); // send Delete Query
  }
}
```

`@Test` 가 부착된 Method 는 `@Transactional` 이 적용된다. 그리고 Method 가 실행된 후 RollBack 된다. 따라서 flush 가 수행되지 않을 수도 있다. 예를 들어 다음과 같이 `crud` 를 작성하면 Insert, Delete Query 는 수행되지 않는다.

```java
  @Test
  public void crud() {
    postRepository.findMyPost();
    Post pos = new Post();
    post.setTitle("Hello World");
    postRepository.save(post);
    postRepositoty.delete(post);
  }
```

`Impl` suffix 를 수정하고 싶다면 `@EnableJpaRepository` 를 다음과 같이 수정한다.

```java
// src/main/java/com.iamslash.exjpa/ExjpaApplication.java
@SpringBootApplication
@EnableJpaRepositories(repositoryImplementationPostfix = "Default")
public class Application {
  public static void main(String[] args) {
    SpringApplication.run(ExjpaApplication.class);
  }
}
```

## Spring Data Common: Basic Repository Customizing

이번에는 모든 entity 에 대해 적용할 수 있는 Custom Repository 를 만들어 보자.

```java
// src/main/java/com.iamslash.exjpa/MyRepository.java
// Bean 등록은 필요 없다.
@NoRepositoryBean
public interface MyRepository<T, ID extends Serializable> extends JpaRepository<T, ID> {
  boolean contains(T entity);
}

// src/main/java/com.iamslash.exjpa/SimpleMyRepository.java
public class SimpleMyRepository<T, ID extends Serializable> extends SimpleJpaRepository<T, ID> implements MyRepository<T, ID> {
  
  private EntityManager entityManager;

  public SimpleMyRepositoty(JpaEntityInformation<T, ?> entityInformation, EntityManager entityManager) {
    super(entityInformation, entityManager);
    this.entityManager = entityManager;
  }

  @Override
  public boolean contains(T entity) {
    return entityManager.contains(entity);
  }
}

// src/main/java/com.iamslash.exjpa/ExjpaApplication.java
@SpringBootApplication
@EnableJpaRepositories(repositoryBaseClass = SimpleMyRepository.class)
public class Application {
  public static void main(String[] args) {
    SpringApplication.run(ExjpaApplication.class);
  }
}

// src/main/java/com.iamslash.exjpa/PostRepository.java
public interface PostRespository extends MyRepository<Post, Long> {

}

// src/test/java/com.iamslash.exjpa/PostRepositoryTest.java
@RunWith(SpringRunner.class)
@DataJpaTest
public class PostRepositoryTest {

  @Autowired
  PostRepository postRepository;

  @Test
  public void crud() {
    postRepository.findMyPost();
    Post pos = new Post();
    post.setTitle("Hello World");

    assertThat(postRepository.contains(post)).isFalse();
    
    postRepository.save(post);

    assertThat(postRepository.contains(post)).isTrue();
    
    postRepositoty.delete(post);
    postRepositoty.flush(); // send Delete Query
  }
}
```

## Spring Data Common: Domain Event

Domain Event 를 publish 하고 listener 에서 handle 해보자. 예를 들어 `Post` 가 하나 만들어 지고 publish 라는 event 가 발생하면 DB 에 저장하도록 구현해 보자. 먼저 `ApplicationContext` 를 이용하여 Event 를 publish 해보자.

```java
// src/main/java/com.iamslash.exjpa/PostPublishedEvent.java
public class PostPublishedEvent extends ApplicationEvent {

  private final Post post;

  public PostPublishedEvent(Object source) {
    super(source);
    this.post = (Post) source;
  }

  public Post getPost() {
    return post;
  }
}

// src/main/java/com.iamslash.exjpa/PostListener.java
public class PostListener implements ApplicationListener<PostPublishedEvent> {

  @Override
  public void onApplicationEvent(PostPublishedEvent event) {
    System.out.println("----------");
    System.out.println("event.getPost() + is published!!");
    System.out.println("----------");
  }
}

// src/test/java/com.iamslash.exjpa/PostRepositoryTestConfig.java
// PostRepositoryTest 는 slicing test 이다. PostRepositoryTest 에
// @Import(PostRepositoryTestConfig) 를 부착해야 PostListener 가
// Bean 으로 등록된다.
@Configuration
public class PostRepositoryTestConfig {

  @Bean
  public PostListener postListener {
    return new PostListener();
  }
}

// src/test/java/com.iamslash.exjpa/PostRepositoryTest.java
@RunWith(SpringRunner.class)
@DataJpaTest
@Import(PostRepositotyTestConfig.class)
public class PostRepositoryTest {

  @Autowired
  PostRepository postRepository;

  @Autowired
  ApplicationContext applicationContext;

  @Test
  public void event() {
    Post post = new Post();
    post.setTitle("event");
    PostPublishedEvent event = new PostPublishedEvent(post);

    applicationContext.publishEvent(event);
  }
}
```

`@EventListener` 를 이용하면 `PostListener` 를 더욱 간단히 구현할 수 있다.

```java
// src/main/java/com.iamslash.exjpa/PostListener.java
public class PostListener {

  @EventListener
  public void onApplicationEvent(PostPublishedEvent event) {
    System.out.println("----------");
    System.out.println(event.getPost().getTitle() + " is published");
    System.out.println("----------");
  }
}
```

또한 다음과 같이 `PostRepositoryTestConfig` 에 Listener 를 구현하면 `PostListener` 는 더이상 필요 없다.

```java
@configuration
public class PostRepositoryTestConfig {

  @Bean
  public ApplicationListener<PostPublishedEvent> postListener() {
    return new ApplicationListener<PostPublishedEvent>() {
      @Overide
      public void onApplicationEvent(PostPublishedEvent event) {
        System.out.println("-----------");
        System.out.println(event.getPost().getTitle() + " is published");
        System.out.println("-----------");
      }
    }
  }
}
```

이번에는 `Post` 가 `AbstractAggregateRoot<E>` 를 상속하여 Domain Event 를 더욱 간단히 구현해 보자. `ApplicationContext` 는 더이상 필요 없다. `entityManager.save()` 를 호출할 때 등록되어 있던 Domain Event 가 publish 된다.

```java
// src/main/java/com.iamslash.exjpa/PostPublishedEvent.java
public class PostPublishedEvent extends ApplicationEvent {
  ...
}

// src/main/java/com.iamslash.exjpa/PostListener.java
public class PostListener {

  @EventListener
  public void onApplicationEvent(PostPublishedEvent event) {
    System.out.println("----------");
    System.out.println(event.getPost().getTitle() + " is published");
    System.out.println("----------");
  }
}

@Configuration
public class PostRepositoryTestConfig {

  @Bean
  public PostListener postListener {
    return new PostListener();
  }
}

// src/main/java/com.iamslash.exjpa/Post.java
@Entity
public class Post extends AbstractAggregateRoot<Post> {
  ...
  public Post publish() {
    this.registerEvent(new PostPublishedEvent(this));
    return this;
  }
}

// src/test/java/com.iamslash.exjpa/PostRepositoryTest.java
@RunWith(SpringRunner.class)
@DataJpaTest
@Import(PostRepositotyTestConfig.class)
public class PostRepositoryTest {

  @Autowired
  PostRepository postRepository;

  @Test
  public void evcrudent() {
    Post post = new Post();
    post.setTitle("event");

    assertThat(postRepository.contains(post)).isFalse();
    postRepository.save(post.publish());
    assertThat(postRepositoty.contains(post)).isTrue();
  }
}
```

`AbstractAggregateRoot.java` 를 살펴보면 다음과 같은 함수들을 확인할 수 있다.

* `registerEvent` : 이벤트를 등록한다.
* `clearDomainEvents` : 등록된 이벤트를 모두 삭제한다. `@AfterDomainEventPublication` 사용.
* `domainEvents` : 등록된 이벤트를 리턴한다. `DomainEvents` 사용

```java
// AbstractAggregateRoot.java
protected <T> T registerEvent(T event) {
  Assert.notNull(event, "Domain event must not be null!");

  this.domainEvents.add(event);
  return event;
}

@AfterDomainEventPublication
protected void clearDomainEvents() {
  this.domainEvents.clear();
}

@DomainEvents
protected Collection<Object> domainEvent() {
  return collections.unmodifiableList(domainEvents);
}
```

## Spring Data Common: QueryDSL

Repository interface 의 `findByFirstNameIngoreCaseAndLastNameStartsWithIgnoreCase(String firstName, String lastName)` 와 같은 Method 는 readability 가 떨어진다. QueryDSL 을 사용하면 type-safe 한 조건문 (predicate) 을 만들어서 Query 를 구현할 수 있다. 또한 readability 가 개선된다.

예를 들어 다음과 같이 사용한다.

```java
QAccount acccount = QAccount.account;
Predicate predicate = ...;
Optional<Account> one = accountRepository.findOne(predicate);
```

QueryDSL 를 사용하기 위해 다음과 같은 dependency 들을 `build.gradle` 에 추가한다.

```groovy
buildscript {
	ext {
		queryDslVersion = "4.4.0"
	}
}

plugins {
	id "org.springframework.boot" version "2.4.0-SNAPSHOT"
	id "io.spring.dependency-management" version "1.0.9.RELEASE"
	id "java"
}
group = "com.iamslash.exjpa"
version = "0.0.1-SNAPSHOT"
sourceCompatibility = "1.8"

repositories {
	mavenCentral()
	maven { url "https://repo.spring.io/milestone" }
	maven { url "https://repo.spring.io/snapshot" }
}

dependencies {
	implementation "org.springframework.boot:spring-boot-starter-data-jpa"
	implementation "org.springframework.boot:spring-boot-starter-web"
	implementation "org.springframework.boot:spring-boot-devtools"
  // QueryDSL
	implementation ("com.querydsl:querydsl-jpa:${queryDslVersion}")
	annotationProcessor ("com.querydsl:querydsl-apt:${queryDslVersion}:jpa")
	testImplementation ("com.querydsl:querydsl-jpa:${queryDslVersion}")
	testAnnotationProcessor ("com.querydsl:querydsl-apt:${queryDslVersion}:jpa")
  // Lombok
	implementation "org.projectlombok:lombok"
	annotationProcessor ("org.projectlombok:lombok")
	testImplementation ("org.projectlombok:lombok")
	testAnnotationProcessor ("org.projectlombok:lombok")
	
  runtimeOnly "com.h2database:h2"
	
  testImplementation "org.springframework.boot:spring-boot-starter-test"
}
test {
	useJUnitPlatform()
}
```

spring data jpa debugging 을 위해 `application.properties` 를 다음과 같이 수정한다.

```conf
spring.jpa.show-sql=true
spring.jpa.properties.hibernate.format_sql=true
logging.level.org.hibernate.type.descriptor.sql=trace
```

그리고 다음과 같이 구현한다.

```java
// src/main/java/com.iamslash.exjpa/Application.java

// src/main/java/com.iamslash.exjpa/Account/Account.java
@Entity
public class Account {
  
  @Id
  @GeneratedValue
  private Log id;

  private String username;

  private String firstname;

  private String lastName;

  ...
}

// src/main/java/com.iamslash.exjpa/Account/AccountRepository.java
public interface AccountRepository extends JpaRepository<Account, Long>, QuerydslPredicateExecutor<Account> {
}

// src/test/java/com.iamslash.exjpa/Account/AccountRepositoryTest.java
@RunWith(SpringRunner.class)
@DataJpaTest
public class AccountRepositoryTest {

  @Autowired
  AccountRepository accountRepository;

  @Test
  public void crud() {
    QAccount account = QAccount.account;
    Predicate predicate = account
      .firstName.containsIgnoreCase("iamslash")
      .and(QAccount.account.lastname.startsWith("iamslash"));
    Optional<Account> one = accountRepository.findOne(predicate);
    assertThat(one).isEmpty();
  }
}
```

이번에는 모든 entity 에 대해 적용할 수 있는 Custom Repository 를 QueryDSL 을 이용하여 만들어 보자.

```java
// src/main/java/com.iamslash.exjpa/SimpleMyRepository.java
// Bean 등록은 필요 없다.
@NoRepositoryBean
public class SimpleMyRepository<T, ID extends Serializable> extends QuerydslJpaRepository<T, ID> implements MyRepository<T, ID> {
  
  private EntityManager entityManager;

  public SimpleMyRepositoty(JpaEntityInformation<T, Id> entityInformation, EntityManager entityManager) {
    super(entityInformation, entityManager);
    this.entityManager = entityManager;
  }

  @Override
  public boolean contains(T entity) {
    return entityManager.contains(entity);
  }
}

// src/main/java/com.iamslash.exjpa/ExjpaApplication.java
@SpringBootApplication
@EnableJpaRepositories(repositoryBaseClass = SimpleMyRepository.class)
public class Application {
  public static void main(String[] args) {
    SpringApplication.run(ExjpaApplication.class);
  }
}

// src/main/java/com.iamslash.exjpa/PostRepository.java
public interface PostRespository extends MyRepository<Post, Long>, QuerydslPredicateExecutor<Post> {

}

// src/test/java/com.iamslash.exjpa/PostRepositoryTest.java
@RunWith(SpringRunner.class)
@DataJpaTest
public class PostRepositoryTest {

  @Autowired
  PostRepository postRepository;

  @Test
  public void crud() {
    postRepository.findMyPost();
    Post pos = new Post();
    post.setTitle("Hello World");

    Predicate predicate = QPost.post
      .title.containsIgnoreCase("iam");
    Optional<Post> one = postRepository.findOne(predicate);
    assertThat(one).isNotEmpty();
  }
}
```

이번에는 Book Entity class 를 QueryDSL 을 이용하여 구현해 본다.

```java
// src/main/java/com.iamslash.exjpa/Application.java
@SpringBootApplication
public class ExjpaApplication {
  public static void main(String[] args) {
    SpringApplication.run(ExjpaApplication.class);
  }
}

// src/main/java/com.iamslash.exjpa/Book.java
@Data
@Entity
public class Book {

  @Id
  @GeneratedValue
  private Long id;

  private String title;

  @Lob
  private String content;

  ...
}

// src/main/java/com.iamslash.exjpa/BookRepository.java
public interface BookRepository extends JpaRepository<Book, Long>, QuerydslPredicateExecutor<Book> {

}

// src/test/java/com.iamslash.exjpa/BookRepositoryTest.java
@DataJpaTest
class BookRepositoryTest {

  @Autowired 
  BookRepository bookRepository;

  @Test
  void test() {
    Book book = new Book();
    book.setTitle("spring");
    book.setContent("data");
    bookRepository.save(book);

    assertEquals(1, bookRepository.findAll().size());

    Optional<Book> one = bookRepository.findOne(QBook.book.title.contains("iam"));
    assertTrue(one.isPresent());
    Optional<Book> two = bookRepository.findOne(QBook.book.title.contains("jpa"));
    assertTrue(two.isEmpty());
  }
}
```

QuerydslJpaRepository 는 deprecated 되었다. 대신 SimpleJpaRepository 를 사용하도록 한다.

```java
// src/main/java/com.iamslash.exjpa/SimpleMyRepository.java
// Bean 등록은 필요 없다.
@NoRepositoryBean
public class SimpleMyRepository<T, ID extends Serializable> extends SimpleJpaRepository<T, ID> implements MyRepository<T, ID> {
...  
}
```

## Spring Data Common: Web: Web Support Features

`@SpringBootApplication` 를 사용한다면 특별히 설정할 것이 없다.
`@SpringBootApplication` 를 사용하지 않는 다면 다음과 같이
`@EnableSpringDataWebSupport` 를 `WebConfiguration` class 에 부착한다.

```java
@EnableWebMvc
@EnableSpringDataWebSupport
class WebConfiguration {

}
```

`@EnableSpringDataWebSupport` 를 부착하면 다음과 같은 기능들을 제공한다. 그러나 Domain Class Converter 와 `Pageable, Sort` 만 유용할 뿐이다.

* DomainClassConverter
* `Pageable, Sort` parameters
* HATEOAS
  * PagedResourcesAssembler
  * PagedResource
* Payload Projection
  * `@ProjectedPayload, @XBRead, @JsonPath`

## Spring Data Common: Web: DomainClassConverter

다음은 `getPost` 에서 `id` 를 `Post` 로 변환하는 예이다.

```java
// src/main/java/com.iamslash.exjpa/Application.java
@SpringBootApplication
public class Application {

  public static void main(String[] args) {
    SpringApplication.run(Application.class, args);
  }
}

// src/main/java/com.iamslash.exjpa/post/Post.java
Entity
public class Post {
  @Id @GeneratedValue
  private Long id;

  private String title;

  // length is greater than 255
  @Lob
  private String content;

  @Temporal(TemporalType.TIMESTAMP)
  private Date created;

  // getters, setters
  ...
}

// src/main/java/com.iamslash.exjpa/post/PostRepository.java
public interface PostRepository extends JpaRepository<Post, Long> {
...
}

// src/main/java/com.iamslash.exjpa/post/PostController.java
@RestController
public class PostController {

  @Autowired
  PostRepository postRepository;

  @GetMapping("/post/{id}")
  public Post getPost(@PathVariable Long id) {
    Optional<Post> byId = postRepository.findById(id);
    Post post = byId.get();
    return post.getTitle();
  }
}

// src/test/resources/application-test.properties
spring.jp.show-sql=true
spring.jpa.properties.hibernate.format_sql=true
logging.level.org.hibernate.type.descriptor.sql=trace

// src/test/java/com.iamslash.exjpa/post/PostController.java
// Integration Test
@RunWith(SpringRunner.class)
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
public class PostControllerTest {

  @Autowired
  MockMvc mockMvc;

  @Autowired
  PostRepository postRepository

  @Test
  public void getPost() {
    Post post = new Post();
    post.setTitle("jpa");
    postRepository.save(post);

    mockMvc.perform(get("/posts/1"))
      .andDo(print())
      .andExpect(status().isOk())
      .andExpect(content().string("jpa"));
  }

}
```

`DomainClassConverter` class 덕분에 argument 에서 `id` 가 `Post` 로 변환된다.

```java
  @GetMapping("/post/{id}")
  public Post getPost(@PathVariable("id") Post post) {
    return post.getTitle();
  }
```

`DomainClassConverter` class 에서 다음과 같은 부분을 주목하자.

```java
public class DomainClassConverter<T extends ConversionService & ConverterRegistry>
  implements ConditionalGenericConverter, ApplicationContextAware {

  }
  // id 가 주어지면 Entity 로 변환
  private class ToEntityConverter implements ConditionalGenericConverter {

  }
  // Entity 가 주어지면 id 로 변환
  private class ToIdConverter implements ConditionalGenericConverter {

  }
```

## Spring Data Common: Web: Pageable and Sort Parameters

Spring MVC 의 [HandlerMethodArgumentResolver](https://docs.spring.io/spring-framework/docs/current/javadoc-api/org/springframework/web/method/support/HandlerMethodArgumentResolver.html) 를 구현한 class 들을 Rest api 의  paramter 로 이용하면 다양한 기능들을 구현할 수 있다. 특히 `Pageable` 이 유용하다.

```java
// src/main/java/com.iamslash.exjpa/Application.java
@SpringBootApplication
public class Application {

  public static void main(String[] args) {
    SpringApplication.run(Application.class, args);
  }
}

// src/main/java/com.iamslash.exjpa/post/Post.java
Entity
public class Post {
  @Id @GeneratedValue
  private Long id;

  private String title;

  // length is greater than 255
  @Lob
  private String content;

  @Temporal(TemporalType.TIMESTAMP)
  private Date created;

  // getters, setters
  ...
}

// src/main/java/com.iamslash.exjpa/post/PostRepository.java
public interface PostRepository extends JpaRepository<Post, Long> {
...
}

// src/main/java/com.iamslash.exjpa/post/PostController.java
@RestController
public class PostController {

  @Autowired
  PostRepository postRepository;

  @GetMapping("/post/{id}")
  public Post getPost(@PathVariable Long id) {
    Optional<Post> byId = postRepository.findById(id);
    Post post = byId.get();
    return post.getTitle();
  }

  @GetMapping("/posts")
  public Page<Post> getPosts(Pageable pageable) {
    return postRepository.findAll(pageable);
  }
}

// src/test/resources/application-test.properties
spring.jp.show-sql=true
spring.jpa.properties.hibernate.format_sql=true
logging.level.org.hibernate.type.descriptor.sql=trace

// src/test/java/com.iamslash.exjpa/post/PostController.java
// Integration Test
@RunWith(SpringRunner.class)
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
public class PostControllerTest {

  @Autowired
  MockMvc mockMvc;

  @Autowired
  PostRepository postRepository

  @Test
  public void getPost() {
    Post post = new Post();
    post.setTitle("jpa");
    postRepository.save(post);

    mockMvc.perform(get("/posts/1"))
      .andDo(print())
      .andExpect(status().isOk())
      .andExpect(content().string("jpa"));
  }

  @Test
  public void getPosts() {
    Post post = new Post();
    post.setTitle("jpa");
    postRepository.save(post);

    mockMvc.perform(get("/posts")
        .param("page", "0")
        .param("size", "10")
        .param("sort", "created,desc")
        .param("sort", "title"))
      .andDo(print())
      .andExpect(status().isOk())
      .andExpect(jsonpath("$.content[0].title", is("jpa")));
  }

}
```

## Spring Data Common: Web: HATEOAS

* [An Intro to Spring HATEOAS](https://www.baeldung.com/spring-hateoas-tutorial)

----

The Spring HATEOAS project is a library of APIs that we can use to easily create REST representations that follow the principle of HATEOAS (Hypertext as the Engine of Application State).

쉽게 얘기하면 HTTP API 의 Response Body 에 link 를 포함한 것이다.

다음과 같은 dependency 를 build.gradle 에 선언한다.

```groovy
implementation("org.springframework.boot:spring-boot-starter-hateoas:2.1.4.RELEASE")
```

```java
// src/main/java/com.iamslash.exjpa/Application.java
@SpringBootApplication
public class Application {

  public static void main(String[] args) {
    SpringApplication.run(Application.class, args);
  }
}

// src/main/java/com.iamslash.exjpa/post/Post.java
Entity
public class Post {
  @Id @GeneratedValue
  private Long id;

  private String title;

  // length is greater than 255
  @Lob
  private String content;

  @Temporal(TemporalType.TIMESTAMP)
  private Date created;

  // getters, setters
  ...
}

// src/main/java/com.iamslash.exjpa/post/PostRepository.java
public interface PostRepository extends JpaRepository<Post, Long> {
...
}

// src/main/java/com.iamslash.exjpa/post/PostController.java
@RestController
public class PostController {

  @Autowired
  PostRepository postRepository;

  @GetMapping("/post/{id}")
  public Post getPost(@PathVariable Long id) {
    Optional<Post> byId = postRepository.findById(id);
    Post post = byId.get();
    return post.getTitle();
  }

  @GetMapping("/posts")
  public PagedResources<Resource<Post>> getPosts(Pageable pageable, PagedResourcesAssembler<Post> assembler) {
    Page<Post> all = postRepository.findAll(pageable);
    return assmbler.toResource(postRepository.findAll(all));
  }
}

// src/test/resources/application-test.properties
spring.jp.show-sql=true
spring.jpa.properties.hibernate.format_sql=true
logging.level.org.hibernate.type.descriptor.sql=trace

// src/test/java/com.iamslash.exjpa/post/PostController.java
// Integration Test
@RunWith(SpringRunner.class)
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
public class PostControllerTest {

  @Autowired
  MockMvc mockMvc;

  @Autowired
  PostRepository postRepository

  @Test
  public void getPost() {
    Post post = new Post();
    post.setTitle("jpa");
    postRepository.save(post);

    mockMvc.perform(get("/posts/1"))
      .andDo(print())
      .andExpect(status().isOk())
      .andExpect(content().string("jpa"));
  }

  @Test
  public void getPosts() {
    Post post = new Post();
    post.setTitle("jpa");
    postRepository.save(post);

    mockMvc.perform(get("/posts")
        .param("page", "0")
        .param("size", "10")
        .param("sort", "created,desc")
        .param("sort", "title"))
      .andDo(print())
      .andExpect(status().isOk())
      .andExpect(jsonpath("$.content[0].title", is("jpa")));
  }

  private void createPosts() {
    int postsCount = 100;
    while (postsCount-- > 0) {
      Post = new Post();
      post.setTitle("jpa");
      postRepository.save(post);      
    }
  }

}
```

## Spring Data JPA: JPA Repository

`@SpringBootApplication` 을 사용한다면 `@EnableJpaRepositories` 를 부착할 필요가 없다.

`@Repository` 는 부착하지 않아도 된다. 부착한다고 별 문제가 되지는 않는다.

`@Repository` 가 부착된다면 `SQLExcpetion` 또는 JPA Exception 을 Spring 의 DataAccessException 으로 변환한다???

## Spring Data JPA: Saving Entity

`SimpleJpaRepository::save` class 는 다음과 같이 구현되어 있다. 즉, entity 가 Transient status 이면 persist 를 호출하고 entity 가 Detached status 이면 merge 를 호출한다.

```java
@Transactional
public <S extends T> S save(S entity) {
  if (entityInformation.isNew(entity)) {
    em.persist(entity);
    return entity;
  } else {
    return em.merge(entity);
  }
}
```

`em.persist` 는 entity 를 persistent context 에 caching 한다. 이후 entity 에 변화가 생기면 DataBase 와 sync 한다. `em.merge` 는 entity 의 복사본을 만들고 그 복사본을 persistent context 에 caching 하고 return 한다. 이후 copied entity 에 변화가 생기면 DataBase 와 sync 한다.

```java
// src/test/java/com.iamslash.exjpa/post/PostController.java
// Integration Test
@RunWith(SpringRunner.class)
@DataJpaTest
public class PostControllerTest {

  @Autowired
  PostRepository postRepository

  @PersistenceContext
  private EntityManager entityManager;

  @Test
  public void save() {
    Post post = new Post();
    post.setTitle("jpa");
    Post savedPost = postRepository.save(post); // persist

    assertThat(entityManager.contains(post)).isTrue();

    Post postUpdate = new Post();
    postUpdate.setId(post.getId());
    postUpdate.setTitle("hibernate");
    postRepository.save(postUpdate); // merge

    List<Post> all = postRepository.findAll();
    assertThat(all.size()).isEqualTo(1);
  }
}
```

`entityReturn = postRepository.save(entityArg)` 를 호출하고 나서는 반드시 return 된 post 를 사용하는 것이 bug free 하다. entityArg 가 transient status 혹은 detached status 일 때 entityReturn 는 항상 persistent status 임이 보장되기 때문이다.

## Spring Data JPA: Query method

* [6.3.2. Query Creation @ spring.io](https://docs.spring.io/spring-data/jpa/docs/current/reference/html/#jpa.query-methods.query-creation)

----

다음과 같은 KeyWord 를 이용하여 Repository interface 의 Query method 생성이 가능하다.

* And, Or
* Is, Equals
* LessThan, LessThanEqual, GreaterThan, GreaterThanEqual
* After, Before
* IsNull, IsNotNull, NotNull
* Like, NotLike
* StartingWith, EndingWith, Containing
* OrderBy
* Not, In, NotIn
* True, False
* IgnoreCase

다음과 같이 Entity 에 `@NamedQuery` 를 부착하자. Repository interface 의 Query Method 중 matchiing 된 것이 있을 때 `@NamedQeury` 의 `query` 가 실행된다.

```java
// src/main/java/com.iamslash.exjpa/post/Post.java
@Entity
@NamedQuery(name = "Post.findByTitle", query = "SELECT p FROM Post AS p WHERE p.title = ?1")
public class Post {
  @Id @GeneratedValue
  private Long id;

  private String title;

  // length is greater than 255
  @Lob
  private String content;

  @Temporal(TemporalType.TIMESTAMP)
  private Date created;

  // getters, setters
  ...
}

// src/main/java/com.iamslash.exjpa/post/PostRepository.java
public interface PostRepository extends JpaRepository<Post, Long> {
  List<Post> findByTitle(String title);
}

// src/test/java/com.iamslash.exjpa/post/PostController.java
// Integration Test
@RunWith(SpringRunner.class)
@DataJpaTest
public class PostControllerTest {

  @Autowired
  PostRepository postRepository

  @PersistenceContext
  private EntityManager entityManager;

  private void savePost() {
    Post post = new Post();
    post.setTitle("Spring");
    postRepository.save(post);
  }

  @Test
  public void findByTitle() {
    savePost();
    List<Post> all = postRepository.findByTitle("Spring");
    assertThat(all.size()).isEqualTo(1);
  }
}
```

또한 다음과 같이 `@Query` 를 Repository interface 에 부착하여 Custom query 를 사용할 수 있다.

```java
// src/main/java/com.iamslash.exjpa/post/Post.java
@Entity
public class Post {
  @Id @GeneratedValue
  private Long id;

  private String title;

  // length is greater than 255
  @Lob
  private String content;

  @Temporal(TemporalType.TIMESTAMP)
  private Date created;

  // getters, setters
  ...
}

// src/main/java/com.iamslash.exjpa/post/PostRepository.java
public interface PostRepository extends JpaRepository<Post, Long> {
  @Query("SELECT p FROM post AS p WHERE p.title = ?1")
  List<Post> findByTitle(String title);
}

// src/test/java/com.iamslash.exjpa/post/PostController.java
@RunWith(SpringRunner.class)
@DataJpaTest
public class PostControllerTest {

  @Autowired
  PostRepository postRepository

  @PersistenceContext
  private EntityManager entityManager;

  private void savePost() {
    Post post = new Post();
    post.setTitle("Spring");
    postRepository.save(post);
  }

  @Test
  public void findByTitle() {
    savePost();
    List<Post> all = postRepository.findByTitle("Spring");
    assertThat(all.size()).isEqualTo(1);
  }
}
```

만약 jpql 대신 nativeQuery 를 사용하고 싶다면 `@Query` 에 다음을 추가한다. 

```java
// src/main/java/com.iamslash.exjpa/post/PostRepository.java
public interface PostRepository extends JpaRepository<Post, Long> {
  @Query("SELECT p FROM post AS p WHERE p.title = ?1", nativeQuery = true)
  List<Post> findByTitle(String title);
}
```

## Spring Data JPA: Query method Sort

`@Query` 를 사용하여 Sort 를 수행할 때 한가지 제약사항이 있다. `Sort.by()` 의
매개변수로 사용한 문자열은 Entity 의 property 혹은 alias 이어야 한다.

```java
// src/test/java/com.iamslash.exjpa/post/PostController.java
@RunWith(SpringRunner.class)
@DataJpaTest
public class PostControllerTest {

  @Autowired
  PostRepository postRepository

  @PersistenceContext
  private EntityManager entityManager;

  private void savePost() {
    Post post = new Post();
    post.setTitle("Spring");
    postRepository.save(post);
  }

  @Test
  public void findByTitle() {
    savePost();
    // title is a property of the entity
    List<Post> all = postRepository.findByTitle("Spring", Sort.by("title");
    assertThat(all.size()).isEqualTo(1);
  }
}

// src/main/java/com.iamslash.exjpa/post/PostRepository.java
public interface PostRepository extends JpaRepository<Post, Long> {
  // title is a alias of the entity
  @Query("SELECT p, p.title AS title FROM post AS p WHERE p.title = ?1")
  List<Post> findByTitle(String title, Sort sort);
}    
```

## Spring Data JPA: Named Parameter and SpEL

## Spring Data JPA: Update query method

## Spring Data JPA: EntityGraph

## Spring Data JPA: Projection

## Spring Data JPA: Specifications

## Spring Data JPA: Query by Example

## Spring Data JPA: Transaction

* [transactional @ TIL](https://github.com/iamslash/TIL/blob/3833c3dba5db0c33551741d9c9b8cc09954b86e2/spring/README.md#transactional)

## Spring Data JPA: Auditing

* [Auditing with JPA, Hibernate, and Spring Data JPA](https://www.baeldung.com/database-auditing-jpa)

----

There 3 ways for Auditing including Auditing With JPA, Hibernate Envers, Spring Data JPA in Spring. This is a Auditing using Spring Data JPA.

0. Enableing JPA Auditing. 

just add @EnableJpaAuditing on your @Configuration class.

```java
@Configuration
@EnableTransactionManagement
@EnableJpaRepositories
@EnableJpaAuditing
public class PersistenceConfig { ... }
```

1. Adding Spring's Entity Callback Listener.

```java
@Entity
@EntityListeners(AuditingEntityListener.class)
public class Hello { ... }
```

2. Tracking Created and Last Modified Date.

```java
@Entity
@EntityListeners(AuditingEntityListener.class)
public class Hello {     
    //...     
    @Column(name = "created_date", nullable = false, updatable = false)
    @CreatedDate
    private long createdDate;
 
    @Column(name = "modified_date")
    @LastModifiedDate
    private long modifiedDate;     
    //...     
}
```

3. Auditing the Author of Changes With Spring Security.

```java
@Entity
@EntityListeners(AuditingEntityListener.class)
public class Bar {     
    //...     
    @Column(name = "created_by")
    @CreatedBy
    private String createdBy;
 
    @Column(name = "modified_by")
    @LastModifiedBy
    private String modifiedBy;     
    //...     
}
```

4. Getting the author from SecurityContext's Authentication.

```java
public class AuditorAwareImpl implements AuditorAware<String> {
  
    @Override
    public String getCurrentAuditor() {
        // your custom logic
    }
 
}
```

5. Configuring to use AuditorAwareImpl to look up the current principal.

```java
@EnableJpaAuditing(auditorAwareRef="auditorProvider")
public class PersistenceConfig {     
    //...     
    @Bean
    AuditorAware<String> auditorProvider() {
        return new AuditorAwareImpl();
    }     
    //...    
}
```

## JPA Cache

* [JPA 캐시](https://gunju-ko.github.io/jpa/2019/01/14/JPA-2%EC%B0%A8%EC%BA%90%EC%8B%9C.html)

-----

JPA Cache 는 1-level Cache, 2-level Cache 가 있다.

**1-level Cache** 는 Persistence Context 안에 존재한다. 하나의 HTTP Request 가 시작되고 종료될 때까지 발생하는 하나의 Transaction 동안 유지된다. 

2-level Cache 는 Application 이 시작해서 종료될때까지 유지된다. Redis 와 같이 외부의 Cache 도 2-level Cache 이다???
