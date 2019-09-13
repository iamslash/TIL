
# IoC Container and Bean

## Spring IoC Container and Bean

IOC Container 가 관리하는 객체를 Bean 이라고 한다. 마치 MS 의 COM 과 비슷한 것 같다. IOC Container 가 생성하고 다른 class 에 DI (Dependency Injection) 한다. Bean 은 주로 Singleton 이다.

다음은 `BookService` Bean 의 구현이다. `@service` 를 사용해서 Bean 이 되었다. `@Autowired` 를 사용해서 IOC Container 가 생성한 `BookRepository` Bean 을 얻어올 수 있다. `BookRepository` Bean 을 Dependency Injection 에 의해 constructor 에서 argument 로 전달받는다. `@PostConstruct` 를 사용해서 `BookService` Bean 이 생성된 후 함수가 실행되도록 구현했다.

```java
@service
public class BookService {
  @Autowired
  private BookRepository bookRepository;
  
  public BookService(BookRepository bookRepository) {
    this.bookRepository = bookRepository;
  }
  
  public Book save(Book book) {
    book.setCreated(new Date());
    book.setBookStatus(BookStatus.DRAFT);
    return bookRepository.save(book);
  }
  
  @PostConstruct
  public void postConstruct() {
    System.out.println("==============================");
    System.out.println("BookService::postConstruct");
  }
}
...
@Repository
public class BookRepository {
  public Book save(Book book) {
    return null;
  }
}
```

IOC 의 핵심은 [BeanFactory interface](https://docs.spring.io/spring-framework/docs/current/javadoc-api/org/springframework/beans/factory/BeanFactory.html) 이다. 다음과 같이 Bean 의 lifecycle 을 이해할 수 있다. 

* BeanNameAware's setBeanName
* BeanClassLoaderAware's setBeanClassLoader
* BeanFactoryAware's setBeanFactory
* EnvironmentAware's setEnvironment
* EmbeddedValueResolverAware's setEmbeddedValueResolver
* ResourceLoaderAware's setResourceLoader (only applicable when running in an application context)
* ApplicationEventPublisherAware's setApplicationEventPublisher (only applicable when running in an application context)
* MessageSourceAware's setMessageSource (only applicable when running in an application context)
* ApplicationContextAware's setApplicationContext (only applicable when running in an application context)
* ServletContextAware's setServletContext (only applicable when running in a web application context)
* postProcessBeforeInitialization methods of BeanPostProcessors
* InitializingBean's afterPropertiesSet
* a custom init-method definition
* postProcessAfterInitialization methods of BeanPostProcessors

[ApplicationContext](https://docs.spring.io/spring-framework/docs/current/javadoc-api/org/springframework/context/ApplicationContext.html) 는 가장 많이 사용하는 [BeanFactory](https://docs.spring.io/spring-framework/docs/current/javadoc-api/org/springframework/beans/factory/BeanFactory.html) 이다.

## Application Context and Setting Bean

Bean 을 생성하는 방법크게 xml 혹은 class 를 이용하는 방법이 있다.

먼저 xml 에 생성하고자 하는 Bean 을 모두 표기해보자. `bookService` 에서 `bookRepository` 를 DI 해야한다. 따라서 `bookRepository` Bean 을 reference 하기 위해 `property` 가 사용되었음을 주목하자. Bean 이 늘어날 때 마다 application.xml 에 모두 등록해야 한다. 상당히 귀찮다.

```xml
<bean id="bookService" 
      class="..."
      autowire="default">
  <property name="bookRespository" ref="bookRespository"/>
</bean>      
<bean id="bookRespository"
      class="..."/>
```

```java
public class DemoApplication {
  public static void main(String[] args) {
    ApplicationContext ctx = new ClassPathXmlApplicationContext(...);
    String[] names = ctx.getBeanDefinitionNames();
    System.out.println(Arrays.toString(names));
    BookService bookService = (BookService) ctx.getBean(s:"bookService");
    System.out.println(bookService.bookRepository != null);    
  }
}
```

또 다른 방법은 application.xml 에 component scan 을 사용하여 간단히 Bean 설정할 수 있다. Bean class 들은 `@Component` 를 사용해야 한다. `@Bean, @Service, @Repository` 는 `@Component` 를 확장한 annotation 이다. DI 를 위해 `@Autowired` 를 사용한다.

```xml
<context:component-scan base-package="..."/>
```

ApplicationConfig class 를 사용해서 Bean 을 등록할 수 있다. 여전히 component scan 은 xml 에서 수행한다.

```java
@Configuration
public class ApplicationConfig {
  @Bean
  public BookRepository bookRepository() {
    return new BookRespotiroy();
  }
  @Bean
  public BookService bookService() {
    BookService bookService = new BookService();
    bookService.setBookRepository(bookRepository());
    return bookService;
  }
}
```

```java
public class DemoApplication {
  public static void main(String[] args) {
    ApplicationContext ctx = new AnnotationConfigApplicationContext(ApplicationConfig.class);
    String[] names = ctx.getBeanDefinitionNames();
    System.out.println(Arrays.toString(names));
    BookService bookService = (BookService) ctx.getBean(s:"bookService");
    System.out.println(bookService.bookRepository != null);    
  }
}
```

이번에는 component scan 마저 annotation 을 이용해보자. 더 이상 xml 은 필요 없다.

```java
@Configuration
@ComponentScan(basePackageClasses = DemoApplication.class)
public class ApplicationConfig {
}
...
@SpringBootApplication
public class DemoApplication {
  public static void main(String[] args) {

  }
}
```

## @Autowire

Bean 의 constructor 에 `@Autowire` 를 사용하여 DI 해보자.

```java
// BookService.java
@Service
public class BookService {
  BookRepository bookRepository;
  
  @Autowire
  public BookService(BookRepository bookRepository) {
    this.bookRepository = bookRepository;
  }
}
// BookRepository.java
@Repository
public class BookRepository {  
}

// DemoApplication.java
```

Bean 의 setter 에 `@Autowire` 를 사용하여 DI 해보자.

```java
// BookService.java
@Service
public class BookService {
  BookRepository bookRepository;
  
  @Autowire(required = false)
  public setBookRepository(BookRepository bookRepository) {
    this.bookRepository = bookRepository;
  }
}
// BookRepository.java
@Repository
public class BookRepository {  
}

// DemoApplication.java
```

Bean 의 field 에 `@Autowire` 를 사용하여 DI 해보자.

```java
// BookService.java
@Service
public class BookService { 
  @Autowire(required = false)
  BookRepository bookRepository;
}
// BookRepository.java
@Repository
public class BookRepository {  
}

// DemoApplication.java
```

이번에는 여러개의 Bean 을 DI 해보자. `@Primary`, `@Qulifier`, `List` 를 사용할 수 있다.

다음은 `@Primary` 를 사용하여 `BookRepository` 를 상속받은 class 들 중 어느것을 사용할지 선택한 구현이다.

```java
// BookService.java
@Service
public class BookService { 
  @Autowire
  BookRepository bookRepository;

  public void printBookRepository() {
    System.out.println(bookRespository.getClass());
  }
}

// BookRepository.java
@Repository
public class BookRepository {  
}

// MyBookRepository.java
@Repository @Primary
public class MyBookRepository implements BookRepository {
}

// YourBookRepository.java
@Repository
public class YourBookRepository implements BookRepository {
}

// BookServiceRunner.java
@Component
public class BookServiceRunner implements ApplicationRunner {
  @Autowired
  BookRepository bookRepository;

  @Override
  public void run(ApplicationArguments args) throws Exception {
    bookRepository.printBookRepository();
  }
}
```

다음은 `@Qualifier` 를 사용해서 `@Autowired` 할 때 사용할 Bean 을 고르는 구현이다.

```java
// BookService.java
@Service
public class BookService { 
  @Autowired @Qualifier("MyBookRepository")
  BookRepository bookRepository;

  public void printBookRepository() {
    System.out.println(bookRespository.getClass());
  }
}

// BookRepository.java
@Repository
public class BookRepository {  
}

// MyBookRepository.java
@Repository
public class MyBookRepository implements BookRepository {
}

// YourBookRepository.java
@Repository
public class YourBookRepository implements BookRepository {
}
```

이번에는 List 를 사용하여 여러개의 Bean 을 모두 DI 하는 방법이다.

```java
// BookService.java
@Service
public class BookService { 
  @Autowired
  List<BookRepository> bookRepositories;

  public void printBookRepository() {
    this.bookRepositories.forEach(System.out::println);
  }
}

// BookRepository.java
@Repository
public class BookRepository {  
}

// MyBookRepository.java
@Repository
public class MyBookRepository implements BookRepository {
}

// YourBookRepository.java
@Repository
public class YourBookRepository implements BookRepository {
}
```

field 의 이름을 DI 하고 싶은 Bean 의 이름으로 설정하면 그 Bean 을 DI 할 수 있다.

```java
// BookService.java
@Service
public class BookService { 
  @Autowired
  BookRepository myBookRepository;

  public void printBookRepository() {
    System.out.println(mybookRepository.getClass());
  }
}

// BookRepository.java
@Repository
public class BookRepository {  
}

// MyBookRepository.java
@Repository
public class MyBookRepository implements BookRepository {
}

// YourBookRepository.java
@Repository
public class YourBookRepository implements BookRepository {
}
```

주로 `@Primary` 를 사용하는 것이 좋다.

그렇다면 `@Autowired` 는 어떤 시점에 object 를 생성하여 DI 를 하는 것인가? 

`BeanFactory` 가 `BeanPostProcessor` 를 검색한다. `AutowiredAnnotationBeanPostProcess` 이미 등록되어 있는 `BeanPostProcessor` 이다. `BeanFactory` 는 `AutowiredAnnotationBeanPostProcess` Bean 을 검색하여 다른 Bean 들을 순회하고 그 Bean 의 initialization 전에 `AutowiredAnnotationBeanPostProcess` 의 logic 즉 DI 를 수행한다. 

다음과 같이 `@PostConstruct` 를 사용하면 `myBookRepository` 가 이미 DI 되었음을 알 수 있다.

```java
@Service
public class BookService {
  @Autowired
  BookRepository myBookREpository;

  @PostConstruct
  public void setup() {
    System.out.println(myBookRepository.getClass());
  }
}
```

다음은 `ApplicationRunner` 를 사용하여 `AutowiredAnnotationBeanPostProcess` 가 등록되어 있는지 확인하는 구현이다.

```java
@Component
public class MyRunner implements ApplicationRunner {
  @Autowired
  ApplicationContext applicationContext;

  @Override
  public void run(ApplicationArguments args) throws Exception {
    AutowiredAnnotationBeanPostProcessor bean = applicationContext.getBean(AutowiredAnnotationBeanPostProcessor.class);
    System.out.println(bean);
  }
}
```

## @Component and Component Scan

## Scope of Bean

## Environment Profile

## Environment Property

## MessageSource

## ApplicationEventPublisher

## ResourceLoader

# Resource and Validation

## Resource Abstraction

## Validation Abstraction

# Data Binding

## PropertyEditor

## Converter and Formatter

# SpEL (Spring Expression Language)

# Spring AOP (Aspected Oriented Programming)

## Overview

## Proxy Based AOP

## @AOP

# Null-Safty