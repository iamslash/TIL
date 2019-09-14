
- [Materials](#materials)
- [IoC Container and Bean](#ioc-container-and-bean)
  - [Spring IoC Container and Bean](#spring-ioc-container-and-bean)
  - [Application Context and Setting Bean](#application-context-and-setting-bean)
  - [@Autowire](#autowire)
  - [@Component and Component Scan](#component-and-component-scan)
  - [Scope of Bean](#scope-of-bean)
  - [Environment Profile](#environment-profile)
  - [Environment Property](#environment-property)
  - [MessageSource](#messagesource)
  - [ApplicationEventPublisher](#applicationeventpublisher)
  - [ResourceLoader](#resourceloader)
- [Resource and Validation](#resource-and-validation)
  - [Resource Abstraction](#resource-abstraction)
  - [Validation Abstraction](#validation-abstraction)
- [Data Binding](#data-binding)
  - [PropertyEditor](#propertyeditor)
  - [Converter and Formatter](#converter-and-formatter)
- [SpEL (Spring Expression Language)](#spel-spring-expression-language)
- [Spring AOP (Aspected Oriented Programming)](#spring-aop-aspected-oriented-programming)
  - [Overview](#overview)
  - [Proxy Based AOP](#proxy-based-aop)
  - [@AOP](#aop)
- [Null-Safty](#null-safty)

----

# Materials

* [스프링 프레임워크 핵심 기술 @ inflearn](https://www.inflearn.com/course/spring-framework_core/)

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

객체 그래프를 조회하고 조작하는 기능을 제공한다. `#{}` 를 이용하면 expression 을 사용할 수 있다. `${}` 를 이용하면 properties 의 값을 얻어올 수 있다.

다음은 `@Value` annotation 에 `SpEL` 을 사용한 에이다.

```java
@Component
public class AppRunner implements ApplicationRunner {
  @Value("#{1 + 1}")
  int value;

  @Value("#{'Foo ' + 'Bar'}")
  String greeting;

  @Value("#{Baz}")
  String something;

  @Value("#{1 eq 1}")
  boolean bval;

  @Override
  public void run(ApplicationArguments args) throws Exception {
    System.out.println("==========");
    System.out.println(value);
    System.out.println(greeting);
    System.out.println(bval);    
    System.out.println(something);    
  }
}
```

다음은 `${}` 을 이용하여 properties 의 값을 얻어오는 구현이다.

```java
// application.properties
name=Foo

// AppRunner.java
@Component
public class AppRunner implements ApplicationRunner {
  @Value("${name.value}")
  String name

  @Override
  public void run(ApplicationArguments args) throws Exception {
    System.out.println("==========");
    System.out.println(name);
  }
}
```

expression 안에 properties 는 가능하다. 그러나 반대는 불가능하다.

```java
// application.properties
name=Foo

// AppRunner.java
@Component
public class AppRunner implements ApplicationRunner {
  @Value("#{${name.value} eq 100}")
  bool bval;

  @Override
  public void run(ApplicationArguments args) throws Exception {
    System.out.println("==========");
    System.out.println(bval);
  }
}
```

Bean 의 field 를 접근할 수 있다. 다음은 `Foo` Bean 의 `data` field 를 접근한 예이다.

```java
// AppRunner.java
@Component
public class AppRunner implements ApplicationRunner {
  @Value("#{Foo.data}")
  int data;

  @Override
  public void run(ApplicationArguments args) throws Exception {
    System.out.println("==========");
    System.out.println(data);
  }
}
```

SpEL 은 ExpressionParser 를 사용하여 evaluation 한다. 다음은 ExpressionParser 를 사용하여 직접 evaluation 하는 예이다.

```java
public void run(ApplicationArguements args) throws Exception {
  ExpressionParser parser = new SpelExpressionParser();
  Expression exp = parser.parseExpression("1 + 2");
  Integer val = exp.getValue(Integer.class);
  System.out.println(val);
}
```

# Spring AOP (Aspected Oriented Programming)

## Overview

반복되는 코드를 분리해서 모듈화하는 프로그래밍 기법이다. 반복되는 코드를 `cross-cutting`, 분리된 모듈을 `aspect` 라고 한다. 따라서 AOP 를 적용하면 반복되는 코드를 줄일 수 있다. 이때 반복되는 코드와 같이 해야할 일들을 `advice`, 어디에 적용해야 하는지를 `pointcut`, 적용해야할 class 를 `target`, method 를 호출할 때 aspect 를 삽입하는 지점을 `joinpoint` 라고 한다. 

AOP 는 언어별로 다양한 구현체가 있다. java 는 주로 AspectJ 를 사용한다. 또한 AOP 는 compile, load, run time 에 적용 가능하다. 만약 Foo 라는 class 에 A 라는 aspect 를 적용한다고 해보자. 

* compile time 에 AOP 를 적용한다면 Foo 의 compile time 에 aspect 가 적용된 byte 코드를 생성한다. 그러나 compile time 이 느려진다.
* load time 에 AOP 를 적용한다면 VM 이 Foo 를 load 할 때 aspect 가 적용된 Foo 를 메모리에 로드한다. 이것을 AOP weaving 이라고 한다. AOP weaving 을 위해서는 agent 를 포함하여 복잡한 설정을 해야 한다.
* rum time 에 AOP 를 적용한다면 VM 이 Foo 를 실행할 때 aspect 를 적용한다. 수행성능은 load time 과 비슷할 것이다. 대신 복잡한 설정이 필요없다.

## Proxy Based AOP

Spring 은 run time 에 Proxy Bean 을 만들어서 특정 Bean 의 methods 가 호출될 때 apect 를 실행하도록 한다.

예를 `A, B, C` 라는 class 를 구현한다고 해보자. `A, B, C` 의 methods 의 수행성능을 측정하기 위해 코드를 삽입하려고 한다. 수행속도를 측정하는 코드는 모든 methods 에서 반복되기 마련이다. 다음과 같이 Proxy Bean 을 만들어서 run time 에 AOP 를 적용해보자.

```java
// IService
public class IService {
  void create();
  void puslish();
  void delete();
}

// AService
@Service
public class AService implements IService {
  @Override
  public void create() {
    try {
      Thread.sleep(1000);
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
    System.out.println("Created");
  }

  @Override
  public void publish() {
    try {
      Thread.sleep(2000);
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
    System.out.println("Published");
  }

  @Override
  public void delete() {
    System.out.println("Deleted");
  }
}

// ProxyService
@Primary
@Service
public class ProxyService implements IService {

  @Autowired
  Iservice iservice;

  @Override
  public void create() {    
    long begin = System.currentTimeMillis();
    iservice.create();
    System.out.println(System.currentTimeMillis() - begin);
  }

  @Override
  public void publish() {
    long begin = System.currentTimeMillis();
    iservice.publish();
    System.out.println(System.currentTimeMillis() - begin);
  }

  @Override
  public void delete() {
    iservice.delete();
  }
}

// AppRunner
@Component
public class AppRunner implements ApplicationRunner {
  @Autowired
  IService iservice;

  @Override
  public void run(ApplicationArguements args) throws Exception {
    iservice.create();
    iservice.publish();
    iservice.delete();
  }
}

// spring 의 web 을 이용하면 runtime 이 느려지므로 
// 다음과 같이 web 을 제거하여 실행할 수 있다.
// DemoApplication
@SpringBootApplication
public class DemoApplication {
  public static void main(String[] args) {
    SpringApplication app = new SpringApplication(DemoApplication.class);
    app.setWebApplicationType(WebApplicationType.NONE);
    app.run(args);
  }
}
```

그러나 Spring 에서 프로그래머가 위와 같이 Proxy class 를 제공할 필요는 없다. `AbstractAutoProxyCreate` 가 runtime 에 Proxy class 를 제공해 준다. `AbstractAutoProxyCreate` 는 `BeanPostProcessor` 를 구현한다.

## @AOP

annotation 을 이용하여 AOP 를 구현해보자.

pom.xml 에 dependency 를 입력한다.

```xml
<dependencies>
  <dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-app</artifactId>
  </dependency>
```

`AbstractAutoProxyCreate` 에게 `PerfAspect` 가 `aspect` 임을 알리기 위해 `@Aspect` 를 선언한다. 그리고 component scan 을 위해 `@Component` 를 선언한다. `logPerf` 의 내용은 `advice` 라고 할 수 있다. `@Around` advice 를 사용하면 `proceed()` 전후로 advice 를 적용할 수 있다. `@Around` 의 `"execution(* com.iamslash.*.IService.*(..))"` 는 `IService` 의 모든 method 들을 의미한다. 즉, `pointcut` 이다. `pointcut` 을 재사용할 수도 있다.

```java
@Component
@Aspect
public class PerfAspect {

  @Around("execution(* com.iamslash.*.IService.*(..))")
  public Object logPerf(ProceedingJointPoint pjp) {
    long begin = System.currentTimeMillis();
    Object retVal = pjp.proceed();
    System.out.println(System.currentTimeMillis() - begin);
    return retVal;
  }
}
```

위와 같이 `PerAspect` 를 작성하면 `IService` 의 모든 method `create, publish, delete` 에 `advice` 가 적용된다. 만약 `delete` 에는 적용되는 것을 원하지 않는다면 다음과 같이 별도의 annotation `PerfLogging` 을 제작하고 사용하기를 원하는 method 에만 선언하자.

```java
// PerfLogging.java
@Documented
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.CLASS)
public @interface PerfLogging {

}

// AService
@Primary
@Service
public class AService implements IService {

  @Autowired
  Iservice iservice;

  @PerfLogging
  @Override
  public void create() {    
    iservice.create();
  }

  @PerfLogging
  @Override
  public void publish() {
    iservice.publish();
  }

  @Override
  public void delete() {
    iservice.delete();
  }
}
```

advice 의 종류는 `@Around, @Before, @AfterReturning, @AfterThrowing` 이 있다. 

# Null-Safty

Spring 은 argument 와 return value 가 null 인지 IDE 의 도움을 받아 compile time 에 검증할 수 있는 annotation `@NonNull` 을 제공한다. `@NonNull` 을 사용하려면 IntelliJ 의 `Menu | Preferences | Build, Excecution, Deployment | Compiler` 을 선택한다. `Add Runtime Assertions for notnullable-annotated methods and parameters` 를 체크하고 `Configure Annotations` 를 클릭한다. 그리고 `Nullable, NonNullable` annotation 을 추가해야 한다.

```java
// AService.java
public class AService {
  @NonNull
  public String createStr(@NonNull String name) {
    return "created : " + name;
  }
}

// ARunner.java
public class ARunner implements ApplicationRunner {
  @Autowired
  AService aservice;

  @Override
  public void run(ApplicationArguments args) throws Exception {
    aservice.createStr(null);
  }
}
```

또한 package 위에 `@NonNull` 을 사용하면 그 패키지에서 사용하는 모든 methods 의 parameter, return value 가 notnullable 인지 IDE 를 통해서 검증할 수 있다.

```java
@NonNullApi
package com.iamslash.A
```