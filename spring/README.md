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

# Spring Framework Core

[Spring Framework Core](SpringFrameworkCore.md)

# Spring Boot

[Spring Boot](SpringBoot.md)

# Spring Web MVC

[Spring Web MVC](SpringWebMvc.md)

# Spring Data JPA

[Spring Data JPA](SpringDataJpa.md)

# Spring REST API

[Spring REST API](SpringRestApi.md)

# Spring Security

[Spring Security](SpringSecurity.md)

