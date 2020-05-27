- [Abstract](#abstract)
- [@ConfigurationProperties](#configurationproperties)
- [@EnableConfigurationProperties](#enableconfigurationproperties)
- [@TestPropertySource](#testpropertysource)
- [@Autowired](#autowired)

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
