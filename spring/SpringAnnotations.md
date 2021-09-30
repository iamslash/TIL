- [Abstract](#abstract)
- [@Configuration](#configuration)
- [@ConfigurationProperties](#configurationproperties)
- [@EnableConfigurationProperties](#enableconfigurationproperties)
- [@TestPropertySource](#testpropertysource)
- [@Autowired](#autowired)
- [@ExceptionHandler](#exceptionhandler)
- [@ControllerAdvice, @RestControllerAdvice](#controlleradvice-restcontrolleradvice)
- [@Import](#import)
- [@EnableAutoConfiguration](#enableautoconfiguration)
- [@DynamicInsert, @DynamicUpdate](#dynamicinsert-dynamicupdate)
- [@Validation](#validation)

-----

# Abstract

This is about annotations of Spring Boot Framework.

# @Configuration

특정 Bean 들을 생성할 때 필요한 설정들을 구현한 class 를 `Configuration Bean Class` 혹은 간단히 `Configuration Class` 라고 하자. `Configuration Class` 는 역시 또 다른 Bean 으로 등록하기 위해 `@Configuration` 을 부착한다. Spring Framework 는 Component Scan 할 때 `@Configuration` 이 부착된 `Configuration Class` 를 읽는다. 그리고 그 Class 의 함수들중 `@Bean` 이 부착된 method 를 실행하여 Bean 을 생성한다.

예를 들어 다음과 같이 `MyBean, MyBeanImpl, MyConfig` 를 참고하자. `MyConfig` 라는 `Configuration Class` 를 통해 `MyBean` 이라는 Bean 을 생성한다.

```java
@Configuration
public class MyConfig {
	@Bean
	public MyBean getBean() {
		return new MyBeanImpl();
	}	
}

...

public interface MyBean {
	public String getBeanName();
}

...

public class MyBeanImpl implements MyBean {
	public String getBeanName() {
		return "My Bean";
	}
}

...

public class AppMain {
	public static void main(String[] args) {
		AnnotationConfigApplicationContexet context = new AnnotationConfigApplicationContexet(MyConfig.class);
		MyBean bean = context.getBean(MyBean.class);
		System.out.println(bean.getBeanName());
		context.close(0);
	}
}

```

# @ConfigurationProperties

특정 Bean 은 Configuration Class 를 이용하여 생성한다. 이때 그 Bean 의 설정을 넘겨줘야 한다. 이 설정을 `ConfigurationProperties Class` 라고 한다. `@ConfigurationProperties` 는 `ConfigurationProperties Class` 에 생성할 Bean 의 이름과 함께 attach 한다.

```java
@ConfigurationProperties("user")
public class UserProperties {
	private String name;
	private int age;
	...
}
```

# @EnableConfigurationProperties

`Configuration Class` 에 생성할 Bean 의 `ConfigurationProperties Class` 를 넘겨줘야 한다. `@EnableConfigurationProperties` 는 `Configuration Class` 에 넘겨줄 `ConfigurationProperties Class` 와 함께 attach 한다.

```java
@Configuration
@EnableConfigurationProperties(UserProperties.class)
public class UserConfiguration {
	@Bean
	@ConditionalOnMissingBean
	public User user(UserProperties properties) {
		User user = new User();
		user.setAge(properties.getAge());
		user.setName(properties.getName());
		return user;
	}
}
```

# @TestPropertySource

`@TestPropertySource` 를 이용하여 Properties 를 overriding 할 수도 있다.

```java
@RunWith(SpringRunnger.class)
@TestPropertySource(properties = {"iamslash.name=likechad,iamslash.Age=35"})
@SpringBootTest
public class ExbasicApplicationTests {

	@Autowired
	Environment environment;

	@Test
	public void contextLoads() {
		assertThat(environment.getProperty("iamslash.name"))
			.isEqualTo("davidsun");
	}
}
```

# @Autowired

* [Autowired 분석](https://galid1.tistory.com/512)

----

`@Autowired` injects a bean.

You can inject a bean in the field.

```java
// BookService.java
@Service
public class BookService { 
  @Autowired(required = false)
  BookRepository bookRepository;
}
// BookRepository.java
@Repository
public class BookRepository {  
}

// DemoApplication.java
```

You can inject a bean in the constructor. `bookRepository` is injected without `@Autowired`. 

```java
// BookService.java
@Service
public class BookService {
  BookRepository bookRepository;
  
  @Autowired
  public BookService(BookRepository bookRepository) {
    this.bookRepository = bookRepository;
  }
}
// BookRepository.java
@Repository
public class BookRepository {  
}
```

You can inject a bean in the setter. `required = false` means bookRepository can not be registered as a bean.

```java
// BookService.java
@Service
public class BookService {
  BookRepository bookRepository;
  
  @Autowired(required = false)
  public setBookRepository(BookRepository bookRepository) {
    this.bookRepository = bookRepository;
  }
}
// BookRepository.java
@Repository
public class BookRepository {  
}
```

You can inject a bean with the priority. The targeted bean class with `@Primary` will be injected.

```java
@Repository
@Primary
public class HelloBook implements BookRepository {

}

@Repository
public class WorldBook implements BookRepository {

}

@Service
public class BookService {
	@Autowired
	private BookRepository bookRepository;
}
```

You can inject a bean with the bean id.

```java
@Service
public class BookService {
	@Autowired
	@Qualifier("HelloBook")
	private BookRepository bookRepository;

	public void printBookRepository() {
		System.out.println(bookRepository.getClass());
	}
}
```

You can inject beans as list<beans>.

```java
@Service
public class BookService {
	@Autowired
	private List<BookRepository> bookRespositories;

	public void printBookRepositories() {
		this.bookRepositories.forEach(System.out::println);
	}
}
```

# @ExceptionHandler

* [@ControllerAdvice, @ExceptionHandler를 이용한 예외처리 분리, 통합하기(Spring에서 예외 관리하는 방법, 실무에서는 어떻게?)](https://jeong-pro.tistory.com/195)

----

You can handle exceptions in a Controller class. It just works in `@Controller, @RestController` class. It doesn't work in `@Service` class.

```java
@RestController
public class HelloController {
	...
	@ExceptionHandler(NullPointerException.class)
	public Object nullex(Exception e) {
		System.err.println(e.getClass());
		return "helloService";
	}
}
```

# @ControllerAdvice, @RestControllerAdvice

You can handle global exceptions with `@ControllerAdvice`.

```java
@RestControllerAdvice
public class HelloAdvice {
	@ExceptionHandler(CustomException.class)
	public String custom() {
		return "hello custom";
	}
}
```

# @Import

* [[Spring] @Import 어노테이션 사용](https://hilucky.tistory.com/244)
* [4.3.  Aggregating @Configuration classes with @Import](https://docs.spring.io/spring-javaconfig/docs/1.0.0.M4/reference/html/ch04s03.html)

----

`Component Scan` 이 없다면 모든 `@Configuration class` 들은 Bean 으로 등록될 수 없다. 이때 하나의 `@Configuration class` 를 bean 으로 등록한다면 `@Import` 와 함께 사용된 다른 `@Configuration class` 들이 bean 으로 등록된다.

`@Configuration class` 에서 또 다른 `@Configuration class` 를 bean 으로 등록할 수 있다. 예를 들어 다음과 같이 `AppConfig` 를 bean 으로 등록하면 `DataSource` 도 bean 으로 등록된다. 

```java
@Configuration
public class DataSourceConfig {
    @Bean
    public DataSource dataSource() {
        return new DriverManagerDataSource(...);
    }
}

@Configuration
@AnnotationDrivenConfig
@Import(DataSourceConfig.class)
public class AppConfig extends ConfigurationSupport {
    @Autowired DataSourceConfig dataSourceConfig;

    @Bean
    public void TransferService transferService() {
        return new TransferServiceImpl(dataSourceConfig.dataSource());
    }
}

public class Main {
	public static void main(String[] args) {
		JavaConfigApplicationContext ctx =
				new JavaConfigApplicationContext(AppConfig.class);
		...
	}
}
```

다음과 같이 다수의 `@Configuration class` 들을 `@Import` 할 수도 있다.

```java
@Configuration
@Import({ DataSourceConfig.class, TransactionConfig.class })
public class AppConfig extends ConfigurationSupport {
	// @Bean methods here can reference @Bean methods in DataSourceConfig or TransactionConfig
}
```

# @EnableAutoConfiguration

> * [[Spring Boot] 자동 설정 @EnableAutoConfiguration](https://velog.io/@max9106/Spring-Boot-EnableAutoConfiguration)

`spring.factories` 에 Configuration 이 있다. 조건에 따라 Bean 을 등록한다.

# @DynamicInsert, @DynamicUpdate

> * [jpa insert 시 default 값 적용](https://dotoridev.tistory.com/6)

특정 Entity 의 Column 중 기본값이 의도와 달리 NULL 이 저장되는 경우가 있다. 
`@DynamicInsert` 를 사용하면 Insert SQL 을 생성할 때 NULL 인 field 는
제외시킨다. 따라서 Entity class 에서 설정된 `ColumDefault(0)` 가 삽입된다.
`@DynamicUpdate` 는 똑같은 원리로 Update SQL 을 생성할 때 NULL 인 field 는
제외시킨다.

다음과 같은 Profile Entity Class 가 있다고 해보자.

```java
// Profile.java
@Entity
@Getter
@Table(name = "PROFILE")
@NoArgsConstructor
public class Profile {

    @Id
    @GeneratedValue(strategy = IDENTITY)
    @Column(name = "idx")
    private Long idx;

    @Column(name = "name")
    private String name;

    @Column(name = "age")
    private Integer age;

    @Column(name = "friends_count")
    @ColumnDefault("0") //default 0
    private Integer friendsCount;

    @Builder
    public Profile(String name, Integer age, Integer friendsCount) {
        this.name = name;
        this.age = age;
        this.friendsCount = friendsCount;
    }
}
```

Profile Entity Class 의 DDL 은 다음과 같다.

```sql
create table PROFILE (
	idx bigint generated by default as identity,
	friend_count integer default 0,
	name varchar(255),
	age varchar(255),
	primary key (idx)
)
```

그리고 다음과 같이 Profile Entity 를 하나 저장해 보자.

```java
@Test
public void insert_test() {
    Profile profile = Profile.builder()
            .name("iamslash")
            .age(29)
            .build();
    Profile resultProfile = ProfileRepository.save(profile);
	// Fail
    assertThat(resultProfile.getFriendsCount(), Is.is(0)); 
}
```

이때 생성되는 SQl 은 다음과 같다.

```sql
    INSERT 
    INTO
        PROFILE
        (idx, friends_count, name, age) 
    VALUES
        (null, ?, ?, ?)
```

다음과 같이 Prfile Entity Class 에 `@DynamicInsert` 를 추가한다.

```java
@Entity
@Getter
@Table(name = "PROFILE")
@DynamicInsert
@NoArgsConstructor
public class Profile {
    ...    
}
```

그렇다면 생성되는 SQL 은 다음과 같다. firends_count field 가 제외되었다. 따라서 
`@ColumnDefault("0")` 이 저장된다.

```sql
    INSERT 
    INTO
        PROFILE
        (idx, name, age) 
    VALUES
        (null, ?, ?)
```

다음과 같이 Test Code 가 성공하는 것을 확인할 수 있다.

```java
@Test
public void insert_test() {
    Profile profile = Profile.builder()
            .name("iamslash")
            .age(29)
            .build();
    Profile resultProfile = ProfileRepository.save(profile);
	// Success
    assertThat(resultProfile.getFriendsCount(), Is.is(0)); 
}
```

위에서 언급한 default NULL problem 은 `@PrePersist, @PostPersist, @PreRemove, @PostRemove, @PreUpdate, @PostUpdate, @PostLoad` 를 사용하여 해결할 수도 있다. 

`@PrePersist, @PostPersist, @PreRemove, @PostRemove, @PreUpdate, @PostUpdate, @PostLoad` 를 사용하면 특정 Entity 가 persist, remove, update, load 될 때 event handling 할 수 있다.

```java
// Profile.java
@Entity
@Getter
@Table(name = "PROFILE")
@NoArgsConstructor
public class Profile {

    @Id
    @GeneratedValue(strategy = IDENTITY)
    @Column(name = "idx")
    private Long idx;

    @Column(name = "name")
    private String name;

    @Column(name = "age")
    private Integer age;

    @Column(name = "friends_count")
    @ColumnDefault("0") //default 0
    private Integer friendsCount;

    @Builder
    public Profile(String name, Integer age, Integer friendsCount) {
        this.name = name;
        this.age = age;
        this.friendsCount = friendsCount;
    }

    // prePresist will be called before being persisted
    @PrePersist
    public void prePersist() {
        this.friendsCount = this.friendsCount == null ? 0 : this.friendsCount;
    }	
}

// ProfileTest.java
@Test
public void insert_test() {
    Profile profile = Profile.builder()
            .name("name")
			.age(29)
            .build();

    Profile resultProfile = ProfileRepository.save(profile);
    assertThat(profile.getFriendsCount(), Is.is(0));
}
```

# @Validation

* [Java Bean Validation 제대로 알고 쓰자](https://kapentaz.github.io/java/Java-Bean-Validation-%EC%A0%9C%EB%8C%80%EB%A1%9C-%EC%95%8C%EA%B3%A0-%EC%93%B0%EC%9E%90/#)
* [Package javax.validation.constraints @ java](https://docs.oracle.com/javaee/7/api/javax/validation/constraints/package-summary.html)
* [Validation in Spring Boot](https://www.baeldung.com/spring-boot-bean-validation)

----

값의 제한 조건 constraint 를 annotation 으로 달아둔다. constraint 를 어기는 값이 저장되면 Exception 을 던진다.
