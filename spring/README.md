- [Abstract](#abstract)
- [Materials](#materials)
- [Feature](#feature)
  - [IOC (Inversion Of Control)](#ioc-inversion-of-control)
  - [DI (Dependency Injection)](#di-dependency-injection)
  - [AOP (Aspect Oriented Programming)](#aop-aspect-oriented-programming)
  - [PSA (Portable Service Abstraction)](#psa-portable-service-abstraction)
- [Spring Framework Core](#spring-framework-core)
- [Spring Boot](#spring-boot)
- [Spring Web MVC](#spring-web-mvc)
- [Spring Data JPA](#spring-data-jpa)
- [Spring REST API](#spring-rest-api)
- [Spring Security](#spring-security)
- [Spring Batch](#spring-batch)
- [Tips](#tips)
  - [Active profile](#active-profile)
  - [Test Active profile](#test-active-profile)
  - [ConfigurationProperties](#configurationproperties)
  - [Http requests logging](#http-requests-logging)
  - [Http responses logging](#http-responses-logging)
  - [Sprint Boot Test](#sprint-boot-test)
  - [Spring Boot Exception Handling](#spring-boot-exception-handling)
  - [Spring WebMvcConfigure](#spring-webmvcconfigure)

----

# Abstract

- spring framework에 대해 적는다.

# Materials

- [Spring Guides](https://spring.io/guides)
  - Topics Guides are very useful.
- [baeldung spring](https://www.baeldung.com/start-here)
  - 킹왕짱 튜토리얼
  - [src](https://github.com/eugenp/tutorials)
- [All Tutorials on Mkyong.com](https://www.mkyong.com/tutorials/spring-boot-tutorials/)
  - spring boot 를 포함한 여러 java tech turotials 
  - [src](https://github.com/mkyong/spring-boot)
- [예제로 배우는 스프링 입문 (개정판) @ inflearn](https://www.inflearn.com/course/spring_revised_edition#)
  - [spring-petclinic @ github](https://github.com/spring-projects/spring-petclinic)
- [백기선의 Spring 완전 정복 로드맵 - 에이스 개발자가 되자! @ inflearn](https://www.inflearn.com/roadmaps/8)
  - 유료이긴 하지만 유용하다
- [스프링 레퍼런스 번역](https://blog.outsider.ne.kr/category/JAVA?page=1)
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

반복되는 코드를 분리해서 모듈화하는 프로그래밍 기법이다. 반복되는 코드를 `cross-cutting`, 분리된 모듈을 `aspect` 라고 한다. 따라서 AOP 를 적용하면 반복되는 코드를 줄일 수 있다. 이때 반복되는 코드와 같이 해야할 일들을 `advice`, 어디에 적용해야 하는지를 `pointcut`, 적용해야할 class 를 `target`, method 를 호출할 때 aspect 를 삽입하는 지점을 `joinpoint` 라고 한다. 

AOP 는 언어별로 다양한 구현체가 있다. java 는 주로 AspectJ 를 사용한다. 또한 AOP 는 compile, load, run time 에 적용 가능하다. 만약 Foo 라는 class 에 A 라는 aspect 를 적용한다고 해보자. 

* compile time 에 AOP 를 적용한다면 Foo 의 compile time 에 aspect 가 적용된 byte 코드를 생성한다. 그러나 compile time 이 느려진다.
* load time 에 AOP 를 적용한다면 VM 이 Foo 를 load 할 때 aspect 가 적용된 Foo 를 메모리에 로드한다. 이것을 AOP weaving 이라고 한다. AOP weaving 을 위해서는 agent 를 포함하여 복잡한 설정을 해야 한다.
* rum time 에 AOP 를 적용한다면 VM 이 Foo 를 실행할 때 aspect 를 적용한다. 수행성능은 load time 과 비슷할 것이다. 대신 복잡한 설정이 필요없다.

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

# Spring Batch

batch job 을 spring library 를 이용해서 만들어 보자.

* [Creating a Batch Service](https://spring.io/guides/gs/batch-processing/)
  * [src](https://github.com/spring-guides/gs-batch-processing)
* [2. Spring Batch 가이드 - Batch Job 실행해보기](https://jojoldu.tistory.com/325)
* [Spring Batch](https://spring.io/projects/spring-batch)

# Tips

## Active profile

* [spring profile 을 사용하여 환경에 맞게 deploy 하기](https://www.lesstif.com/pages/viewpage.action?pageId=18220309)

Spring application 을 시작할 때 JVM option 으로 profile 을 선택할 수 있다. 두개 이상을 선택해도 됨.

```bash
-Dspring.profiles.active=local
-Dspring.profiles.active=local,develop
```

## Test Active profile

test class 를 작성할 때 `application.yaml` 대신 `application-test.yaml` 을 사용하고 싶다면 다음과 같이 `@ActiveProfiles("test")` 를 사용한다.

```java
@RunWith(SpringRunner.class)
@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
public class PostControllerTest {
}
```

## ConfigurationProperties

* [스프링 부트 커스텀 설정 프로퍼티 클래스 사용하기](https://javacan.tistory.com/entry/springboot-configuration-properties-class)

`application.properties` 에 설정 값을 저장하고 java 에서 읽어 들이자.

```
iamslash.authcookie: HelloWorld
iamslash.authcookieSalt: HelloWorld
```

다음은 `application.properties` 의 Mapping class 이다.

```java
@ConfigurationProperties(prefix = "iamslash")
public class FooSetting {
  private String authcookie;
  private String authcookieSalt;
  public String getAuthcookie() {
    return authcookie;
  }
  public void setAuthcookie(String authcookie) {
    this.authcookie = authcookie;
  }
  public String getAuthcookieSalt() {
    return authcookieSalt;
  }
  public void setAuthcookieSalt(String authcookieSalt) {
    this.authcookieSalt = authcookieSalt;
  }
}
```

`FooSetting` 을 `@EnableConfigurationProperties` 을 사용하여 Bean 으로 등록하고 값을 복사해 온다.

```java
@SpringBootApplication
@EnableConfigurationProperties(FooSetting.class)
public class Application { ... }
```

또는 `FooSetting` 에 `@Configuration` 을 추가하면 Bean 으로 등록된다. 그리고 다음과 같이 `@AutoWired` 을 사용하여 DI 할 수 있다.

```java
@Configuration
public class SecurityConfig {
  @Autowired
  private FooSetting fooSetting;
  @Bean
  public Encryptor encryptor() {
    Encryptor encryptor = new Encryptor();
    encryptor.setSalt(fooSetting.getAuthcookieSalt());
    ...
  }
```

## Http requests logging

* [Spring – Log Incoming Requests](https://www.baeldung.com/spring-http-logging)

* `src/main/java/com.iamslash.alpha.common.RequestLoggingFilterConfig.java`

```java
package org.springframework.security.oauth.samples.common;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.filter.CommonsRequestLoggingFilter;

@Configuration
public class RequestLoggingFilterConfig {

  @Bean
  public CommonsRequestLoggingFilter logFilter() {
    CommonsRequestLoggingFilter filter
            = new CommonsRequestLoggingFilter();
    filter.setIncludeQueryString(true);
    filter.setIncludePayload(true);
    filter.setMaxPayloadLength(10000);
    filter.setIncludeHeaders(true);
    filter.setAfterMessagePrefix("REQUEST DATA : \n");
    return filter;
  }
}
```

* `src/main/resources/application.yaml`

```yaml
server:
  port: 8092

logging:
  level:
    root: INFO
    org.springframework.web: DEBUG
    org.springframework.web.filter: DEBUG
    org.springframework.web.filter.CommonsRequestLoggingFilter: DEBUG
    org.springframework.security: INFO
    org.springframework.security.oauth2: INFO
#    org.springframework.boot.autoconfigure: DEBUG
```

## Http responses logging

* [Logging Spring WebClient Calls](https://www.baeldung.com/spring-log-webclient-calls)

## Sprint Boot Test

* [Spring Boot Test](https://cheese10yun.github.io/spring-boot-test/#null)
* [Spring Boot에서 테스트를 - 1](https://hyper-cube.io/2017/08/06/spring-boot-test-1/)

----

* Spring Boot Test
* @WebMvcTest
* @DataJpaTest
* @RestClientTest
* @JsonTest

* Dependencies of `build.gradle` 

  ```groovy
  dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-web'
    testImplementation('org.springframework.boot:spring-boot-starter-test') {
      exclude group: 'org.junit.vintage', module: 'junit-vintage-engine'
    }
  }
  ```

* Test Class
  * @SpringBootTest 의 WebEnvironment 는 기본값이 SpringBootTest.WebEnvironment.MOCK 이다.
  * Mock Dispatcher 가 실행되어 Controller 를 test 할 수 있다.

  ```java
  @RunWith(SpringRunner.class)
  @SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.MOCK)
  @AutoConfigureMockMvc
  public class DemoTest {
    @Autowired
    MockMvc mockMvc;
    @Test
    public void hello() throws Exception {
      mockMvc.perform(get("/hello"))
        .andExpect(status().isOk())
        .andExpect(content().string("hello"))
        .andDo(print());
    }
  }
  ```

* RandomPort 를 사용하면 Servlet Container 가 실행된다. 
  * mockMvc 대신 TestRestTemplate 를 사용해야 한다.

  ```java
  @RunWith(SpringRunner.class)
  @SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
  public class DemoTest {
    @Autowired
    TestRestTemplate testRestTemplate;
    @Test
    public void hello() throws Exception {
      String result = testRestTemplate.getForObject("/hello", String.class);
      assertThat(result).isEqualTo("hello");
    }
  }
  ```

* Service 를 mocking 해보자.
  
  ```java
  @RunWith(SpringRunner.class)
  @SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
  public class DemoTest {
    @Autowired
    TestRestTemplate testRestTemplate;
    @MockBean
    HelloService mockHelloService;
    @Test
    public void hello() throws Exception {
      when(mockHelloService.getName()).thenReturn("hello");

      mockMvc.perform(get("/hello"))
        .andExpect(status().isOk())
        .andExpect(content().string("hello"))
        .andDo(print());
    }
  }
  ```

* WebTestClient 를 사용하면 Asynchronous http client 를 사용할 수 있다.

  ```groovy
  dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-web'
    implementation 'org.springframework.boot:spring-boot-starter-webflux'
    testImplementation('org.springframework.boot:spring-boot-starter-test') {
      exclude group: 'org.junit.vintage', module: 'junit-vintage-engine'
    }
  }
  ```

  ```java
  @RunWith(SpringRunner.class)
  @SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
  public class DemoTest {
    @Autowired
    WebTestClient webTestClient;
    @MockBean
    HelloService mockHelloService;
    @Test
    public void hello() throws Exception {
      when(mockHelloService.getName()).thenReturn("hello");

      webTestClient.get().uri("/hello").exchange()
        .expectStatus().isOk()
        .expectBody(String.class)
        .isEqualTo("hell");
    }
  }
  ```

* Json 만 lightweight 하게 test 할 수 있다.

  ```java
  @RunWith(SpringRunner.class)
  @JsonTest
  public class DemoTest {
    ...
  }
  ```

* `@SpringBootTest` 에는 많은 test annotation 들이 포함되어 있다. 그것을 각각 이용하면 layer 별로 test 할 수 있다. 이것을 slicing test 라고 한다. `@WebMvcTest` 를 이용하여 특정 Controller 만 사용해 보자.

  ```java
  @RunWith(SpringRunner.class)
  @WebMvcTest(DemoController.class)
  public class DemoControllerTest {
    @MockBean
    DemoService demoService;

    @Autowired
    MockMvc mockMbc

    @Test
    public void hello() throws Exception {
      when(mockHelloService.getName()).thenReturn("hello");

      webTestClient.get().uri("/hello").exchange()
        .expectStatus().isOk()
        .expectBody(String.class)
        .isEqualTo("hello");
    }
  }
  ```

* OutputCapture 를 이용하면 log message 를 test 할 수 있다.

  ```java
  @RestController
  public class DemoController {
    Logger logger = LoggerFactory.getLogger(DemoController.class);
    @Autowired
    private DemoService demoService;

    @GetMapping("/hello")
    public String hello() {
      logger.info("Foo");
      System.out.println("Bar");
      return "hello";
    }
  }
  ```

  ```java
  @RunWith(SpringRunner.class)
  @WebMvcTest(DemoController.class)
  public class DemoControllerTest {
    @Rule
    public OutputCapture outputCapture = new OutputCapture();

    @MockBean
    DemoService demoService;

    @Autowired
    MockMvc mockMbc

    @Test
    public void hello() throws Exception {
      when(mockHelloService.getName()).thenReturn("hello");

      mockMvc.perform(get("/hello"))
        .andExpect(content().string("hello"));

      assertThat(outputCapture.toString())
        .contains("Foo")
        .contains("Bar");        
    }
  }
  ```

## Spring Boot Exception Handling

* [(Spring Boot)오류 처리에 대해](https://supawer0728.github.io/2019/04/04/spring-error-handling/)
* [스프링부트 : REST어플리케이션에서 예외처리하기](https://springboot.tistory.com/33)
* [Error Handling for REST with Spring](https://www.baeldung.com/exception-handling-for-rest-with-spring)

## Spring WebMvcConfigure

spring Web Mvc 는 다음과 같은 설정들 덕분에 사용가능하다.

`org.springframework.boot:spring-boot-autoconfigure` 에 Web Mvc 설정이 포함되어 있다.

* `META-INF/spring.factories` 에 WebMvcAutoConfiguration 가 enable 되어 있다.

```
org.springframework.boot.autoconfigure.EnableAutoConfiguration=\
...
org.springframework.boot.autoconfigure.web.servlet.WebMvcAutoConfiguration,\
...
```

* `org.springframework.boot.autoconfigure/web/servlet/WebMvcAutoConfiguration` 에 `@Configuration` 이 첨부되어 있다.

```java
@Configuration(proxyBeanMethods = false)
@ConditionalOnWebApplication(type = Type.SERVLET)
@ConditionalOnClass({ Servlet.class, DispatcherServlet.class, WebMvcConfigurer.class })
@ConditionalOnMissingBean(WebMvcConfigurationSupport.class)
@AutoConfigureOrder(Ordered.HIGHEST_PRECEDENCE + 10)
@AutoConfigureAfter({ DispatcherServletAutoConfiguration.class, TaskExecutionAutoConfiguration.class,
		ValidationAutoConfiguration.class })
public class WebMvcAutoConfiguration {
  ...
```

* `org.springframework.boot.autoconfigure/web.servlet/WebMvcProperties` 에 `spring.mvc` 로 시작하는 properties 들이 포함된다.

```java
@ConfigurationProperties(prefix = "spring.mvc")
public class WebMvcProperties {
  ...
```
