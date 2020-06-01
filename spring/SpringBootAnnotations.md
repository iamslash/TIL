- [Abstract](#abstract)
- [@ConfigurationProperties](#configurationproperties)
- [@EnableConfigurationProperties](#enableconfigurationproperties)
- [@TestPropertySource](#testpropertysource)
- [@Autowired](#autowired)
- [@ExceptionHandler](#exceptionhandler)
- [@ControllerAdvice, @RestControllerAdvice](#controlleradvice-restcontrolleradvice)
- [@Import](#import)

-----

# Abstract

This is about annotations of Spring Boot Framework.

# @ConfigurationProperties

특정 Bean 은 Configuration class 를 이용하여 생성한다. 이때 그 Bean 의 설정을 넘겨줘야 한다. 이 설정을 Properties class 라고 한다. `@ConfigurationProperties` 는 Properties class 에 생성할 Bean 의 이름과 함께 attach 한다.

```java
@ConfigurationProperties("user")
public class UserProperties {
	private String name;
	private int age;
	...
}
```

# @EnableConfigurationProperties

Configuration class 에 생성할 Bean 의 Properties class 를 넘겨줘야 한다. `@EnableConfigurationProperties` 는 Configuration class 에 넘겨줄 Properties class 와 함께 attach 한다.

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

`@TestPropertySource` 를 이용하여 Properties 들을 overriding 할 수도 있다.

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

You can inject a bean with the priority. The targeted bean  class with `@Primary` will be injected.

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

`@Configuration` class 에서 또 다른 `@Configuration` bean 을 생성할 수 있다. 예를 들어 다음과 같이 AppConfig 를 bean 으로 등록하면 DataSource 도 bean 으로 등록된다.

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

다음과 같이 다수의 class 들을 `@Import` 할 수도 있다.

```java
@Configuration
@Import({ DataSourceConfig.class, TransactionConfig.class })
public class AppConfig extends ConfigurationSupport {
	// @Bean methods here can reference @Bean methods in DataSourceConfig or TransactionConfig
}
```
