- [Abstract](#abstract)
- [@Configuration](#configuration)
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
