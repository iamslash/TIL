# Abstract

- spring framework에 대해 적는다.

# Materials

- [예제로 배우는 스프링 입문 (개정판) @ inflearn](https://www.inflearn.com/course/spring_revised_edition#)
  - [spring-petclinic @ github](https://github.com/spring-projects/spring-petclinic)
- [스프링 프레임워크 핵심 기술 @ inflearn](https://www.inflearn.com/course/spring-framework_core/dashboard)
- [All Tutorials on Mkyong.com](https://www.mkyong.com/tutorials/spring-boot-tutorials/)
  - spring boot 를 포함한 여러 java tech turotials 
- [Spring Framework Documentation](https://docs.spring.io/spring/docs/current/spring-framework-reference/)
  - Overview	
    - history, design philosophy, feedback, getting started.
  - Core	
    - IoC Container, Events, Resources, i18n, Validation, Data Binding, Type Conversion, SpEL, AOP.
  - Testing	
    - Mock Objects, TestContext Framework, Spring MVC Test, WebTestClient.
  - Data Access	
    - Transactions, DAO Support, JDBC, O/R Mapping, XML Marshalling.
  - Web Servlet	
    - Spring MVC, WebSocket, SockJS, STOMP Messaging.
  - Web Reactive	
    - Spring WebFlux, WebClient, WebSocket.
  - Integration	
    - Remoting, JMS, JCA, JMX, Email, Tasks, Scheduling, Caching.
  - Languages	
    - Kotlin, Groovy, Dynamic Languages.

# Feature

## IOC (Inversion Of Control)

* [IOC 와 DI 에 대해서 @ tistory](https://mo-world.tistory.com/entry/IOC%EC%99%80-DI-%EC%97%90-%EB%8C%80%ED%95%98%EC%97%AC-%EC%8A%A4%ED%94%84%EB%A7%81-%EA%B0%9C%EB%85%90-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0-%EC%89%BD%EA%B2%8C-%EC%84%A4%EB%AA%85)

----

Spring Container 가 코드의 흐름을 제어할 수 있다. 즉, 내가 코드의 흐름을 제어하는 것을
Spring Container 가 가져갔다고 해서 inversion of control 이라고 한다.
Spring Container 가 object 의 life cycle 을 나 대신 관리한다. 
IOC 때문에 DI, AOP 가 가능하다.

## DI (Dependency Injection)

object 를 내가 생성하지 않고 Spring Container 가 생성해서 주입해준다.

## AOP (Aspect Oriented Programming)

annotation 을 사용하여 반복되는 코드를 줄일 수 있다.

다음은 함수의 수행속도를 측정하는 AOP 의 예이다.

```java
```

## PSA (Portable Service Abstraction)

annotation 을 사용하여 service 와 loosely coupled 한 코드를 만들 수 있다.

예를 들어 `@Controller, @RequestMapping` 을 사용한 코드는 tomcat, jetty, netty, undertow 와 같은 servlet container 중 어느 것을 사용해도 많은 수정을 할 필요 없다. 즉, 여러 Spring Web MVC 들을 추상화했다고 할 수 있다.

다음은 Spring Web MVC 를 추상화한 구현이다.

```java
```

또한 `@Transactional` 을 사용한 코드는 JpaTransactionManager, DatasourceTransactionManager, HibernateTransactionManager 중 어느 것을 사용해도 많은 수정을 할 필요 없다. 즉, 여러 Transaction Manager 들을 추상화했다고 할 수 있다.

다음은 TransactionManager 를 추상화한 구현이다.

```java
```

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


# Tutorial of STS

- [Spring Tool Suite](https://spring.io/tools)를 설치한다.
- STS를 시작하고 File | New | Spring Starter Project를 선택하고 적당히 설정하자.
  - com.iamslash.firstspring
- 다음과 같은 파일을 com.iamslash.firstspring에 추가하자.

```java
package com.iamslash.firstspring;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class A {
	@RequestMapping("/")
	public String index() {
		return "helloworld!";
	}
}
```
- firstspring을 R-click후에 Run As | Spring Boot App선택해서 실행하자.
- 브라우저를 이용하여 http://localhost:8080으로 접속하자.

# Tutorial of springboot 

- [springboot manual installation](https://docs.spring.io/spring-boot/docs/current/reference/html/getting-started-installing-spring-boot.html#getting-started-manual-cli-installation)
  에서 zip을 다운받아 `d:\local\src\`에 압축을 해제하자.
- 환경설정변수 SPRING_HOME을 만들자. `d:\local\src\D:\local\src\spring-1.5.6.RELEASE`
- 환경설정변수 PATH에 `%SPRING_HOME%\bin`을 추가하자.
- command shell에서 `spring version`을 실행한다.
- command shell에서 `spring init'을 이용하여 새로운 프로젝트를 제작할 수 있다.
