- [Materials](#materials)
- [스프링 부트 원리](#스프링-부트-원리)
	- [의존성 관리 이해](#의존성-관리-이해)
	- [의존성 관리 응용](#의존성-관리-응용)
	- [자동 설정 이해](#자동-설정-이해)
	- [자동설정 만들기 1 부 : Starter 와 AutoConfigure](#자동설정-만들기-1-부--starter-와-autoconfigure)
	- [자동설정 만들기 2 부 : @ConfigurationProperties](#자동설정-만들기-2-부--configurationproperties)
	- [내장 웹 서버 이해](#내장-웹-서버-이해)
	- [내장 웹 서버 응용 1 부: 컨테이너와 포트](#내장-웹-서버-응용-1-부-컨테이너와-포트)
	- [내장 웹 서버 응용 2 부: HTTPS 와 HTTP2](#내장-웹-서버-응용-2-부-https-와-http2)
	- [톰캣 HTTP2](#톰캣-http2)
	- [독립적으로 실행가능한 JAR](#독립적으로-실행가능한-jar)
	- [스프링 부트 원리 정리](#스프링-부트-원리-정리)
- [스프링 부트 활용](#스프링-부트-활용)
	- [스프링 부트 활용 소개](#스프링-부트-활용-소개)
	- [SpringApplication 1 부](#springapplication-1-부)
	- [SpringApplication 2 부](#springapplication-2-부)
	- [외부 설정 1 부](#외부-설정-1-부)
	- [외부 설정 2 부 (1)](#외부-설정-2-부-1)
	- [외부 설정 2 부 (2)](#외부-설정-2-부-2)
	- [프로파일](#프로파일)
	- [로깅 1부 : 스프링 부트 기본 로거설정](#로깅-1부--스프링-부트-기본-로거설정)
	- [로깅 2부 : 커스터마이징](#로깅-2부--커스터마이징)
	- [테스트](#테스트)
	- [테스트 유틸](#테스트-유틸)
	- [Spring-Boot-Devtools](#spring-boot-devtools)
	- [스프링 웹 MVC 1 부: 소개](#스프링-웹-mvc-1-부-소개)
	- [스프링 웹 MVC 2 부: HttpMessageconverters](#스프링-웹-mvc-2-부-httpmessageconverters)
	- [스프링 웹 MVC 3 부: ViewResolve](#스프링-웹-mvc-3-부-viewresolve)
	- [스프링 웹 MVC 4 부: 정적 리소스 지원](#스프링-웹-mvc-4-부-정적-리소스-지원)
	- [스프링 웹 MVC 5 부: 웹 JAR](#스프링-웹-mvc-5-부-웹-jar)
	- [스프링 웹 MVC 6 부: index 페이지와 파비콘](#스프링-웹-mvc-6-부-index-페이지와-파비콘)
	- [스프링 웹 MVC 7 부: Thymeleaf](#스프링-웹-mvc-7-부-thymeleaf)
	- [스프링 웹 MVC 8 부: HtmlUnit](#스프링-웹-mvc-8-부-htmlunit)
	- [스프링 웹 MVC 9 부: ExceptionHandler](#스프링-웹-mvc-9-부-exceptionhandler)
	- [스프링 웹 MVC 10 부: Spring HATEOAS](#스프링-웹-mvc-10-부-spring-hateoas)
	- [스프링 웹 MVC 11 부: CORS](#스프링-웹-mvc-11-부-cors)
	- [스프링 데이터 1 부: 소개](#스프링-데이터-1-부-소개)
	- [스프링 데이터 2 부: 인메모리 데이터베이스](#스프링-데이터-2-부-인메모리-데이터베이스)
	- [스프링 데이터 3 부: MySQL](#스프링-데이터-3-부-mysql)
	- [스프링 데이터 4 부: PostgreSQL](#스프링-데이터-4-부-postgresql)
	- [스프링 데이터 5 부: 스프링 데이터 JPA 소개](#스프링-데이터-5-부-스프링-데이터-jpa-소개)
	- [스프링 데이터 6 부: 스프링 데이터 JPA 연동](#스프링-데이터-6-부-스프링-데이터-jpa-연동)
	- [스프링 데이터 7 부: 데이터베이스 초기화](#스프링-데이터-7-부-데이터베이스-초기화)
	- [스프링 데이터 8 부: 데이터베이스 마이그레이션](#스프링-데이터-8-부-데이터베이스-마이그레이션)
	- [스프링 데이터 9 부: Redis](#스프링-데이터-9-부-redis)
	- [스프링 데이터 10 부: MongoDB](#스프링-데이터-10-부-mongodb)
	- [스프링 데이터 11 부: Neo4J](#스프링-데이터-11-부-neo4j)
	- [스프링 데이터 12 부: 정리](#스프링-데이터-12-부-정리)
	- [스프링 시큐리티 1 부: StarterSecurity](#스프링-시큐리티-1-부-startersecurity)
	- [스프링 시큐리티 2 부: 시큐리티 설정 커스터마이징](#스프링-시큐리티-2-부-시큐리티-설정-커스터마이징)
	- [스프링 REST 클라이언트 1 부: RestTemplate vs WebClient](#스프링-rest-클라이언트-1-부-resttemplate-vs-webclient)
	- [스프링 REST 클라이언트 2 부: Customizing](#스프링-rest-클라이언트-2-부-customizing)
	- [그밖에 다양한 기술 연동](#그밖에-다양한-기술-연동)
- [스프링 부트 운영](#스프링-부트-운영)
	- [스프링 부트 Actuator 1 부: 소개](#스프링-부트-actuator-1-부-소개)
	- [스프링 부트 Actuator 2 부: JMX 와 HTTP](#스프링-부트-actuator-2-부-jmx-와-http)
	- [스프링 부트 Actuator 3 부: 스프링 부트 어드민](#스프링-부트-actuator-3-부-스프링-부트-어드민)

-----

# Materials

* [스프링 부트 개념과 활용 @ inflearn](https://www.inflearn.com/course/%EC%8A%A4%ED%94%84%EB%A7%81%EB%B6%80%ED%8A%B8)
  * [src blog ict-nroo](https://ict-nroo.tistory.com/category/ICT%20Eng/Spring?page=3)
  * [src blog engkimbs](https://engkimbs.tistory.com/category/Spring/Spring%20Boot?page=3)

# 스프링 부트 원리

## 의존성 관리 이해

* [Spring Boot Gradle Plugin(1)](https://brunch.co.kr/@springboot/186)
* [Spring Boot Gradle Plugin Reference Guide](https://docs.spring.io/spring-boot/docs/current/gradle-plugin/reference/html/)
  * [spring boot gradle plugin @ github](https://github.com/spring-projects/spring-boot/blob/master/spring-boot-project/spring-boot-tools/spring-boot-gradle-plugin)
* [Dependency Management Plugin](https://docs.spring.io/dependency-management-plugin/docs/current/reference/html/)
  * [dependency-management-plugin @ github](https://github.com/spring-gradle-plugins/dependency-management-plugin)

----

spring boot gradle plugin 은 spring boot dependency, application packaging 등을 지원한다. 'io.spring.dependency-management' plugin 은 spring boot dependency 를 담당한다. 이것은 spring boot gradle plugin 의 build.gradle 에 dependencies 설정되어 있다.

```gradle

dependencies {
	api(platform(project(":spring-boot-project:spring-boot-dependencies")))
...
	implementation("io.spring.gradle:dependency-management-plugin")
...
}
```

[spring-projects/spring-boot/spring-boot-project/spring-boot-dependencies/pom.xml](https://github.com/spring-projects/spring-boot/blob/2.2.x/spring-boot-project/spring-boot-dependencies/pom.xml) 을 살펴보면 sprint-boot 2.2.x version 의 dpenencies 를 확인할 수 있다.

spring boot gradle plugin 을 이용하려면 다음과 같이 build.gradle 을 설정한다.

```gradle
plugins {
	id 'org.springframework.boot' version '2.2.6.RELEASE'
	id 'io.spring.dependency-management' version '1.0.9.RELEASE'
	id 'java'
}
```

이제 build.gradle 의 dependencies 에 version 을 명시하지 않아도 library 가 import 된다.

```gradle
dependencies {
	implementation 'org.springframework.boot:spring-boot-starter'
	implementation 'org.springframework.boot:spring-boot-starter-web'
	testImplementation('org.springframework.boot:spring-boot-starter-test') {
		exclude group: 'org.junit.vintage', module: 'junit-vintage-engine'
	}
}
```

## 의존성 관리 응용

## 자동 설정 이해

`@SpringBootApplication` 은 `@ComponentScan, @EnableAutoConfiguration` 을 포함한다.

`@ComponentScan` 은 `@Component, @Configuration, @Repository, @Service, @Controller, @RestController` 가 attatch 된 Class 들을 검색하여 Bean 으로 등록한다.

`@Configuration` 은 `@Bean` 이 부착된 method 들을 순회하고 return 된 Bean 을 IOC container 에 등록한다.

`@EnableAutoConfiguration` 은 `org.springframework.boot:spring-boot-autoconfigure/META-INF/spring.factories` 를 포함한 `spring.factories` 파일들을 읽고 `org.springframework.boot.autoconfigure.EnableAutoConfiguration` 의 value 에 해당하는 class 들을 순회하면서 Bean 으로 등록한다. 이 class 들은 모두 `@Configuration` 이 attatch 되어 있다. 그러나 `@ConditionalOnWebApplication, @ConditionalOnClass, @ConditionalOnMissingBean` 등에 의해 조건에 따라 Bean 이 등록될 수도 있고 등록되지 않을수도 있다.

host application 에서 library 들을 `@ComponentScan` 할 수는 없기 때문에 `@EnableAutoConfiguration` 은 매우 유용하다.

## 자동설정 만들기 1 부 : Starter 와 AutoConfigure

`xxx-spring-boot-autoconfigure` module 은 자동설정을 담당한다. `xxx-spring-boot-starter` module 은 의존성 관리를 담당한다. 주로 `POM.xml` 만을 포함한다.

두가지를 합쳐서 `xxx-spring-boot-starter` 에 저장하기도 한다.

* `src/main/java/com.iamslash/User.java`

```java
public class User {
	String name;
	int age;
}
```

* `src/main/java/com.iamslash/UserConfiguration.java`

```java
@Configuration
public class UserConfiguration {
	@Bean
	public User user() {
		User user = new User();
		user.setAge(10);
		user.setName("David");
		return user;
	}
}
```

* `src/main/resources/META-INF/spring.factories`

```conf
org.springframework.boot.autoconfigure.EnableAutoConfiguration=\
  com.iamslash.UserConfiguration
```

`xxx-spring-boot-autoconfigure` 혹은 `xxx-spring-boot-starter` 가 완성되면 반드시 `$ mvn install` 를 이용하여 설치해야 host application 에서 사용할 수 있다.

`@EnableAutoConfiguration` 이 실행되면 `com.iamslash.UserConfiguration` 의 `@Configuration` 이 처리된다.

그러나 다음과 같이 `DemoApplication` 에서 `User` Bean 을 생성하는 경우 `spring.factories` 에서 생성된 `User` Bean 이 over write 한다. over write 을 막기 위해서는 `@ConditionalOnMissingBean` 등을 활용하여 Bean 등록을 제어할 수 있다.

```java
@SpringBootApplication
public class DemoApplication {
	public static void main(String[] args) {
		SpringApplication application = new SpringApplication(Application.class);
		application.setWebApplicationType(WebApplicationType.NONE);
		application.run(args);
	}

	@Bean
	public User user() {
		User user = new User();
		user.setName("Richard");
		user.setAge(20);
		return user;
	}
}
```

## 자동설정 만들기 2 부 : @ConfigurationProperties

다음과 같이 `@ConditionalOnMissingBean` 을 `src/main/java/com.iamslash/UserConfiguration.java` 에 attatch 해주면 `DemoApplication` 의 `@ComponentScan` 에 의한 Bean 을 우선시 할 수 있다.

* `src/main/java/com.iamslash/UserConfiguration.java`

```java
@Configuration
public class UserConfiguration {
	@Bean
	@ConditionalOnMissingBean
	public User user() {
		User user = new User();
		user.setAge(10);
		user.setName("David");
		return user;
	}
}
```

위와 같이 동일한 Bean 을 여러곳에서 정의하는 것보다 편리한 방법을 찾아보자.  `application.properties` 에 Bean 의 properties 를 삽입하고 그것을 기반으로 Bean 을 만들어 보자.

* `src/main/resources/appilcation.properties` 

```conf
user.name=Hello
user.age=30
```

* `src/main/java/com.iamslash/UserConfiguration.java`

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

* `src/main/java/com.iamslash/UserProperties.java`

```java
@ConfigurationProperties("user")
public class UserProperties {
	private String name;
	private int age;
	...
}
```

User Bean 을 등록하려면 반드시 gradle task 중 Install 을 실행해야 한다.

## 내장 웹 서버 이해

## 내장 웹 서버 응용 1 부: 컨테이너와 포트

## 내장 웹 서버 응용 2 부: HTTPS 와 HTTP2

## 톰캣 HTTP2

## 독립적으로 실행가능한 JAR

## 스프링 부트 원리 정리

# 스프링 부트 활용

## 스프링 부트 활용 소개

## SpringApplication 1 부

VM optoin `-Ddebug` 혹은 program arguments `--debug` 을 사용하여 IntelliJ 에서 실행하면 debug mode 로 logging 이 가능하다.

FailureAnalyzer 는 fail 출력을 pretty print 해준다.

`src/main/resources/banner.txt` 를 작성하면 banner 를 customizing 할 수 있다. 혹은 application.properties 에서 `spring.banner.location` 의 값으로 banner.txt 의 경로를 설정할 수 있다.

다음과 같이 code 로 banner 를 출력할 수도 있다.

```java
@SpringBootApplication
public class ExbasicApplication {

	public static void main(String[] args) {
		SpringApplication app = new SpringApplication(ExbasicApplication.class);
		app.setBanner(((environment, sourceClass, out) -> {
			out.println("==================================================");
			out.println("I' am david sun.");
			out.println("==================================================");
		}));
		app.setBannerMode(Banner.Mode.CONSOLE);
		app.run(args);
	}

}
```

Spring Application 은 SpringApplicationBuilder 를 통해 실행할 수도 있다. SpringApplication.run 은 customizing 할 수가 없다.

```java
@SpringBootApplication
public class ExbasicApplication {

	public static void main(String[] args) {
//		SpringApplication app = new SpringApplication(ExbasicApplication.class);
//		app.setBanner(((environment, sourceClass, out) -> {
//			out.println("==================================================");
//			out.println("I' am david sun.");
//			out.println("==================================================");
//		}));
//		app.setBannerMode(Banner.Mode.CONSOLE);
//		app.run(args);
		new SpringApplicationBuilder()
				.sources(ExbasicApplication.class)
				.run(args);
	}

}
```

## SpringApplication 2 부

ApplicationLister 를 implement 하면 Spring Application Event 를 handling 할 수 있다.

```java
@Component
public class HelloListener implements ApplicationListener<ApplicationStartingEvent> {

	@Override
	public void onApplicationEvent(ApplicationStartingEvent applicationStartingEvent) {
		System.out.println("I got you.");
	}
}
```

그러나 ApplicationStartingEvent 는 ApplitionContext 가 init 되기전에 발생한다. ApplicationContext 가 init 되야 HelloListener Bean 이 생성이 될 것이다. 따라서 onApplicationEvent 는 호출되지 않는다.

다음과 같이 직접 HelloLitsener 를 등록해야 한다. HelloListener 는 Bean 일 필요가 없다.

```java
@SpringBootApplication
public class ExbasicApplication {
	public static void main(String[] args) {
		SpringApplication app = new SpringApplication(ExbasicApplication.class);
		app.addListeners(new HelloListener());
		app.run(args);
	}
}
```

ApplicationstartedEvent 는 ApplicationContext 가 init 되고 나서 발생한다.

```java
@Component
public class HelloListener implements ApplicationListener<ApplicationStartedEvent> {

	@Override
	public void onApplicationEvent(ApplicationStartedEvent applicationStartingEvent) {
		System.out.println("I got you.");
	}
}
```

SpringApplication 의 동작 방식을 다음과 같이 조정할 수 있다. `SERVLET, REACTIVE, NONE` 등이 가능하다. 예를 들어 WebMvc, WebFlux 가 둘다 포함되어 있다면 `SERVLET` 혹은 `REACTIVE` 를 선택해야 한다.

```java
@SpringBootApplication
public class ExbasicApplication {
	public static void main(String[] args) {
		SpringApplication app = new SpringApplication(ExbasicApplication.class);
		app.setWebApplicationType(WebApplicationType.REACTIVE);
		app.run(args);
	}
}
```

SpringApplication 의 arguments 를 HelloListener 로 전달해 보자. VM option `-Dfoo` 와 Program arguements `--bar` 와 함께 실행해보자. `--bar` 만 args 에 담겨온다. `-Dfoo` 는 ApplicationArguements 가 아니다.

```java
@Component
public class HelloListener {
	public HelloListener(ApplicationArguements args) {
		System.out.println("foo: " + args.containsOption("foo"));
		System.out.println("bar: " + args.containsOption("bar"));
	}
}
```

SpringApplication 이 실행되고 추가로 뭔가 실행하고 싶다면 ApplicationRunner 를 구현한다. 이번에도 `--bar` 만 출력된다.

```java
@Component
public class HelloRunner implements ApplicationRunner {
	@Override
	public void run(ApplicationArguments args) throws Exception {
	}
}
```

또한 `@Order` 를 사용하여 여러 ApplicationRunner 의 순서를 조정할 수 있다.

```java
@Component
@Order(1)
public class HelloRunner implements ApplicationRunner {
	@Override
	public void run(ApplicationArguments args) throws Exception {
	}
}
```

## 외부 설정 1 부

application.properties 에 SpringApplication 의 설정들을 `key=value` 형태로 저장할 수 있다. 

```
iamslash.name = davidsun
```

그리고 다음과 같이 `@Value` 이용하여 Binding 할 수 있다.

```java
@Component
public class HelloRunner implements ApplicationRunner {
	@Value("${iamslash.name}")
	private String iamslashName;

	@Override
	public void run(ApplicationArguments args) throws Exception {
		System.out.println("==============================");
		System.out.println(iamslashName);
		System.out.println("==============================");
	}
}
```

Properties 의 우선순위는 다음과 같다. [4.2. Externalized Configuration](https://docs.spring.io/spring-boot/docs/current/reference/htmlsingle/#boot-features-external-config)

1. Devtools global settings properties in the $HOME/.config/spring-boot directory when devtools is active.
2. @TestPropertySource annotations on your tests.
3. properties attribute on your tests. Available on @SpringBootTest and the test annotations for testing a particular slice of your application.
4. Command line arguments.
5. Properties from SPRING_APPLICATION_JSON (inline JSON embedded in an environment variable or system property).
6. ServletConfig init parameters.
7. ServletContext init parameters.
8. JNDI attributes from java:comp/env.
9. Java System properties (System.getProperties()).
10. OS environment variables.
11. A RandomValuePropertySource that has properties only in random.*.
12. Profile-specific application properties outside of your packaged jar (application-{profile}.properties and YAML variants).
13. Profile-specific application properties packaged inside your jar (application-{profile}.properties and YAML variants).
14. Application properties outside of your packaged jar (application.properties and YAML variants).
15. Application properties packaged inside your jar (application.properties and YAML variants).
16. @PropertySource annotations on your @Configuration classes. Please note that such property sources are not added to the Environment until the application context is being refreshed. This is too late to configure certain properties such as logging.* and spring.main.* which are read before refresh begins.
17. Default properties (specified by setting SpringApplication.setDefaultProperties).

또한 다음과 같이 Environment 를 이용하여 Properties 를 읽어볼 수 있다.

```java
@RunWith(SpringRunnger.class)
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

Test 전용의 `src/test/resources/application.properties` 를 추가할 수도 있다. InteliJ 에서 module 에 등록해야 한다. compile 이 되면 `src/main/**` 를 먼저 compile 하고 `src/test/**` 를 compile 한다. 따라서 `src/main/resources/application.properties` 가 `src/test/resources/application.properties` 로 overwritting 된다. overriding 이 아니라서 문제가 된다. 

그러나 `src/main/resources/application.properties` 의 내용이 `src/main/resources/application.properties` 와 다르다면 문제가 될 수 있다.

예를 들어 `src/main/resources/application.properties` 는 다음과 같다.

```
iamslash.name = davidsun
iamslash.age = ${random.int}
server.pot = 0
```

그리고 `src/test/resources/application.properties` 는 다음과 같다.

```
iamslash.name = davidsun
```

또한 `src/main/java/com.iamslash.exbasic/HelloRunner.java` 를 다음과 같이 수정한다.

```java
@Component
public class HelloRunner implements ApplicationRunner {
	@Value("${iamslash.name}")
	private String iamslashName;

	@Value("${iamslash.age}")
	private String iamslashAge;

	@Override
	public void run(ApplicationArguments args) throws Exception {
		System.out.println("==============================");
		System.out.println(iamslashName);
		System.out.println(iamslashAge);
		System.out.println("==============================");
	}
}
```

다음과 같이 `contextLoads` 가 실행할 때 `src/mmain/java/**` 에서 필요한 age 가 없으므로 error 가 발생한다.

```java
@RunWith(SpringRunnger.class)
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

`ExbasicApplicationTests` 를 다음과 같이 수정하여 임의의 Properties 를 주입할 수 있다.

```java
@RunWith(SpringRunnger.class)
@SpringBootTest(properties = "iamslash.Age=30")
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

또한 `@TestPropertySource` 를 이용하여 Properties 들을 overriding 할 수도 있다.

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

`@TestPropertySource` 의 항목이 너무 많다면 `src/test/resources/test.properties` 를 만들고 `@TestPropertySource` 를 이용하여 경로를 설정한다.

```java
@RunWith(SpringRunnger.class)
@TestPropertySource(locations = "classpath:/test.properties")
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

`@TestPropertySource` 는 Properties 우선순위가 2 위이다. 매우 높다. home directory 의 `spring-boot-dev-tools.properties` 가 우선순위가 1 위이다. 그러나 잘 사용하지 않는다.

`application.properties` 는 여러 곳에 제작할 수 있고 우선순위는 다음과 같다.

1. `file:./config/`
2. `file:./`
3. `classpath:/config`
4. `classpath:/`

application.properties 에 random value 가 가능하다.

```conf
iamslash.age = ${random.int}
```

## 외부 설정 2 부 (1)

Properties 를 묶어서 Bean 으로 Binding 할 수 있다.

예를 들어 `src/main/resources/application.properties` 를 다음과 같이 작성한다.

```
iamslash.name = davidsun
iamslash.age = ${random.int(0,100)}
iamslash.fullname = ${iamslash.name} runs
server.pot = 0
```

다음과 같이 `@ConfigurationProperties("iamslash")` 을 사용하여 Mapping 할 Bean 을 정의한다.

```java
@Component
@ConfigurationProperties("iamslash")
public class IamslashProperties {
	private String name;
	private Integer age;
	private String fullName;
	...
}
```

그리고 `@EnableConfigurationProperties` 를 SpringApplication 에 attach 한다.

```java
@SpringBootApplication
@EnableConfigurationProperties(IamslashProperties.class)
public class ExbasicApplication {
	public static void main(String[] args) {
		SpringApplication app = new SpringApplication(ExbasicApplication.class);
		app.setWebApplicationType(WebApplicationType.REACTIVE);
		app.run(args);
	}
}
```

이제 `IamslashProperties` 를 `@Autowired` 해서 사용하자.

```java
@Component
public class HelloRunner implements ApplicationRunner {

	@Autowired
	IamslashProperties iamslashProperties;

	@Override
	public void run(ApplicationArguments args) throws Exception {
		System.out.println("==============================");
		System.out.println(iamslashProperties.getName());
		System.out.println(iamslashProperties.getAge());
		System.out.println("==============================");
	}
}
```

## 외부 설정 2 부 (2)

Properties class 를 사용하면 type safe 를 지킬 수 있다.

Properties 의 key 는 `iamslash.full_name` 혹은 `iamslash.full-name` 및 `iamslash.fullName` 이 가능하다. 각 kebab, underscore, camel case 라고 한다. 이것을 Relaxed Binding 이라고 한다.

시간을 Mapping 하기 위해서는 다음과 같이 `@DurationUnit` 을 사용한다.

```java
public class AppSystemProperties {
	@DurationUnit(ChronoUnit.SECONDS)
	private Duration sessionTimeout = Duration.ofSeconds(30);
	...
}
```

`@DurationUnit` 을 사용하지 않고 `application.properties` 에 postfix 를 사용하면 시간으로 Binding 할 수 있다.

```conf
iamslash.settionTimeout=25s
```

`@Validated, @NotEmpty` 를 사용하면 `JSR-303` 를 이용하여 검증이 가능하다.

```java
@Component
@ConfigurationProperties("iamslash")
@Validated
public class IamslashProperties {
	@NotEmpty
	private String name;
	...
}
```

이때 `@NotEmpty` 에 의해 발생되는 `must not empty` 를 포함한 error message 는 Failure Analyzer Bean 에 의해 출력된다.

`@Value` 는 SpEL 이 가능하다. 그러나 다른 annotation 은 불가능하다.

## 프로파일

`@Profile` 을 이용하여 특정 profile 에 대해 active 시킬 수 있다. `@Configuration, @Component` 에 사용할 수 있다.

다음과 같이 두개의 Configuration class 를 제작한다.

```java
@Profile("production")
@Configuration
public class BaseConfiguration {

	@Bean
	public String hello() {
		return "Hello Production";
	}
}
```

```java
@Profile("local")
@Configuration
public class LocalConfiguration {

	@Bean
	public String hello() {
		return "Hello Local";
	}
}
```

그리고 Program arguments `--spring.profiles.active=production` 과 함께 실행하면 profile 에 따라 Bean 생성을 핸들링할 수 있다.

또한 Program arguements `--spring.profiles.include=proddb` 와 함께 실행하면 `application-production.properties` 와 함께 `application-proddb.properties` 도 사용할 수 있다.

## 로깅 1부 : 스프링 부트 기본 로거설정

결국 SLF4j 를 사용하여 구현하면 LogBack 이 logging 한다.

* build.gradle

```gradle
dependencies {
	implementation 'org.springframework.boot:spring-boot-starter-logging'
}
```

다음과 같이 log file 의 경로와 level 을 설정한다.

* application.properties

```conf
logging.path=logs
logging.level.com.iamslash.demo=DEBUG
```

* `src/main/java/com.iamslash.demo/SampleRunner.java`

```java 
@Component
public class SampleRunner implements ApplicationRunner {
	private Logger logger = LoggerFactory.getLogger(SampleRunner.class);

	@Override
	public void run(ApplicationArguments args) throws Exception {
		logger.info("===================");
		logger.info("This is inside run");
		logger.info("===================");
	}
}
```

## 로깅 2부 : 커스터마이징

* [4. Logging](https://docs.spring.io/spring-boot/docs/2.2.6.RELEASE/reference/html/spring-boot-features.html#boot-features-logging)

----

logback 을 다음과 같이 customizing 해보자.

* `src/main/resources/logback-spring.xml` 

```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <include resource="org/springframework/boot/logging/logback/base.xml"/>
    <logger name="com.iamslash" level="DEBUG"/>
</configuration>
```

## 테스트

build.gradle 의 dependency 에 `org.springframework.boot:spring-boot-starter-test` 를 추가한다.

```gradle
dependencies {
	implementation 'org.springframework.boot:spring-boot-starter-parent:2.2.6.RELEASE'
	testImplementation('org.springframework.boot:spring-boot-starter-test') {
		exclude group: 'org.junit.vintage', module: 'junit-vintage-engine'
	}
}
```

`org.springframework.boot:spring-boot-starter-test` 에 종속적인 library 들의 version 관리는 `org.springframework.boot:spring-boot-starter-parent` 가 해준다.


그리고 다음과 같이 `src/main/java/com.iamslash.exbasic.HelloController` 및 `src/main/java/com.iamslash.exbasic.HelloService` 를 제작한다.

```java
@RestController
public class HelloController {
	@AutoWired
	private HelloService helloService;

	@GetMapping("/hello")
	public String hello() {
		return "hello " + helloService.getName();
	}
}
```

```java
@Service
public class HelloService{
	public String getName() {
		return "iamslash";
	}
}
```

그리고 다음과 같이 `src/test/java/com.iamslash.exbasic.HelloControllerTest` 를 제작한다.

```java
@RunWith(SpringRunner.class)
@SpringBootTest(WebEnvironment = SpringBootTest.WebEnvironment.MOCK)
@AutoConfigureMockMbc
public class HelloControllerTest {

	@Autowired
	MockMvc mockMvc;

	@Test
	public void hello() {
		mockMvc.perform(get("/hello"))
		  .andExpect(status().isOk())
			.andExpect(content().string("hello world"))
			.andDo(print());
	}
}
```

`@SpringBootTest(WebEnvironment = SpringBootTest.WebEnvironment.MOCK)` 는 Servlet 를 mocking 해준다. 반드시 `MockMvc` 를 이용해서 client 를 작성해야 한다. `@AutoConfigureMockMbc` 를 추가하고 `MockMvc` 를 AutoWired 해야 한다.

이제 Servlet 를 mocking 하지 않고 실제로 띄워서 테스트 해보자. 반드시 `TestRestTemplate` 를 이용해서 client 를 작성해야 한다.

```java
@RunWith(SpringRunner.class)
@SpringBootTest(WebEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
public class HelloControllerTest {

	@Autowired
	TestRestTemplate testRestTemplate;

	@Test
	public void hello() {
		String response = testRestTemplate.getForObject("/hello", String.class);
		assertThat(response).isEqualTo("hello world");
	}
}
```

만약 test 를 Controller 까지만 수행하고 HelloService 는 제외시키고 싶다면 `@MockBean` 을 이용하여 Service 를 mocking 한다.

```java
@RunWith(SpringRunner.class)
@SpringBootTest(WebEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
public class HelloControllerTest {

	@Autowired
	TestRestTemplate testRestTemplate;

	@MockBean
	HelloService mockHelloSerivce;

	@Test
	public void hello() {

		when(mockHelloService.getName()).thenReturn("iamslash");

		String response = testRestTemplate.getForObject("/hello", String.class);
		assertThat(response).isEqualTo("hello iamslash");
	}
}
```

만약 Asyncronous 한 client 를 사용하여 테스트하고 싶다면 `WebTestClient` 를 사용하자.

build.gradle 의 dependency 에 `org.springframework.boot:spring-boot-starter-webflux` 를 추가한다.

```gradle
dependencies {
	implementation 'org.springframework.boot:spring-boot-starter-webflux'
}
```

`WebTestClient` 를 다음과 같이 사용한다. method chaining 이 되기도 하고 performance 가 `TestRestTemplate` 보다 좋다.

```java
@RunWith(SpringRunner.class)
@SpringBootTest(WebEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
public class HelloControllerTest {

	@Autowired
	WebTestClient webTestClient;

	@MockBean
	HelloService mockHelloSerivce;

	@Test
	public void hello() {

		when(mockHelloService.getName()).thenReturn("iamslash");

		webTestClient.get().url("/hello").exchange()
		  .expectStatus().isOk()
		  .expectBody(String.class)
			.isEqualTo("hello iamslash");
	}
}
```

`@SpringBootTest` 를 사용하면 모든 Bean 들이 등록되기 때문에 비효율적이다. `@WebMvcTest, @JsonTest, @WebFluxTest, @DataJpaTest` 등을 이용하여 특정 Bean 들만 등록해서 test 를 할 수 있다.

다음은 `@WebMvcTest` 를 사용하여 `HelloController` 를 test 한 예이다. `@WebMvcTest` 를 사용하면 반드시 `MockMvc` 로 client를 작성해야 한다.

```java
@RunWith(SpringRunner.class)
@WebMvcTest(HelloController.class)
public class HelloControllerTest {

	@MockBean
	HelloService mockHelloService;

	@Autowired
	MockMvc mockMvc;

	@Test
	public void hello() {

		when(mockHelloService.getName()).thenReturn("iamslash");

		mockMvc.perform(get("/hello"))
		  .andExpect(status().isOk())
			.andExpect(content().string("hello iamslash"))
			.andDo(print());
	}
}
```

`@SpringBooTest` 는 integration test 에서 사용하고 `@WebMvcTest` 는 unit test 에서 사용한다.

## 테스트 유틸

`@OutputCapture` 를 사용하면 log message 를 test 할 수 있다.

다음과 같이 logging 한다.

```java
@RestController
public class HelloController {

	Logger logger = LoggerFactory.getLogger(HelloController.class);

	@AutoWired
	private HelloService helloService;

	@GetMapping("/hello")
	public String hello() {
		logger.info("iamslash");
		System.out.println("hello");
		return "hello " + helloService.getName();
	}
}
```

이제 `OutputCapture` 를 사용하여 log message 를 테스트해 보자.

```java
@RunWith(SpringRunner.class)
@WebMvcTest(HelloController.class)
public class HelloControllerTest {

	@Rule
	public OutputCapture outputCapture = new OutputCapture();

	@MockBean
	HelloService mockHelloService;

	@Autowired
	MockMvc mockMvc;

	@Test
	public void hello() {

		when(mockHelloService.getName()).thenReturn("iamslash");

		mockMvc.perform(get("/hello"))
		  .andExpect(status().isOk())
			.andExpect(content().string("hello iamslash"))
			.andDo(print());

		assertThat(outputCapture.toString())
		  .contains("iamslash")
			.contains("hello");
	}
}
```

## Spring-Boot-Devtools

## 스프링 웹 MVC 1 부: 소개

## 스프링 웹 MVC 2 부: HttpMessageconverters

* HelloController.java

```java
@RestController
public class HelloController {
	
	@GetMapping("/hello")
	public String hello() {
		return "hello";
	}
}
```

* UserControllerTest.java

```java
@RunWith(SpringRunner.class)
@WebMvcTest(UserController.class)
public class UserControllerTest {

	@Autowired 
	MockMvc mockMvc;

	@Test
	public void createUser_JSON() {
		String userJson = "{\"username\":\"iamslash\", \"password\":\"world\"}";
		mockMvc.perform(post("/users/create")
		    .contentType(MediaType.APPLICATION_JSON_UTF8)
			  .accept(MediaType.APPLICATION_JSON_UTF8)
			  .content(userJson))
			.andExpect(status().isOk())
			.andExpect(jsonPath("$.username", is(euqalTo("iamslash"))))
	}
}
```

* UserController.java

```java
@RestController
public class UserController {
	@PostMapping("/users/create")
	public User create(@RequesetBody user) {
		return user;
	}
}
```

* User.java

```java
public class User {
  private Logn id;
	private String username;
	private STring password;
	...
}
```

## 스프링 웹 MVC 3 부: ViewResolve

ContentNegotiatingViewResolver 는 Client 가 보내온 Accept Header 를 보고 Client 가 원하는 format 을 결정한다. 

다음은 Client 가 xml 형식을 원할 때 xml 을 보내 주는 예이다.

* build.gradle

```groovy
dependency {
	implementation 'com.fasterxml.jackson.dataformat:jackson-dataformat-xml:2.9.6'
}
```

* User.java
  * same with before
* UserController.java
  * same with before
* UserControllerTest.java

```java
@RunWith(SpringRunner.class)
@WebMvcTest(UserController.class)
public class UserControllerTest {

	@Autowired 
	MockMvc mockMvc;

	@Test
	public void createUser_XML() {
		String userJson = "{\"username\":\"iamslash\", \"password\":\"world\"}";
		mockMvc.perform(post("/users/create")
		    .contentType(MediaType.APPLICATION_JSON_UTF8)
			  .accept(MediaType.APPLICATION_XML)
			  .content(userJson))
			.andExpect(status().isOk())
			.andExpect(xpath("/User/username", is(euqalTo("iamslash"))))
	}
}
```

## 스프링 웹 MVC 4 부: 정적 리소스 지원

* 다음의 경로에 static resources 들을 복사한다.
* 이후 `/**` 로 request 가 가능하다.

```
classpath:/static
classpath:/public
classpath:/resources
classpath:/META-INF/resources/
```

* `spring.mvc-static-path-pattern=/static/**` 로 수정하면 `/static/**` 로 request 가 가능하다.

* WebMvcConfigure 를 상속받은 class 를 정의하면 static resource path 를 추가할 수 있다. `/m/` 는 `/` 으로 끝남을 주의하자.

```java
@Configuration
public class WebConfig implements WebMvcConfigurer {
	@Override
	public void addResourceHandlers(ResourceHandlerRegistry registry) {
		registry.addResourceHandler("/m/**")
		  .addResourceLocations("classpath:/m/")
			.setCachePeriod(20);
	}
}
```

## 스프링 웹 MVC 5 부: 웹 JAR

client 에서 사용하는 javascript library 를 web jar 에 포함시켜 배포할 수 있다.

* build.gradle

```groovy
dependency {
	implementation 'org.webjars.bower:jqeury:3.3.1'
}
```

* hello.html

```html
<script src="/webjars/jquery/3.3.1/dist/jqeury.min.js"></script>
<script>
  $(function(){
		alert("hello");
	});
</script>
```

webjars-locator-core 를 build.gradle 에 추가하면 hello.html 에서 jquery 의 version 을 생략할 수 있다.


* build.gradle

```groovy
dependency {
	implementation 'org.webjars:webjars-locator-core:0.35'
}
```

* hello.html

```html
<script src="/webjars/jquery/dist/jqeury.min.js"></script>
<script>
  $(function(){
		alert("hello");
	});
</script>
```

## 스프링 웹 MVC 6 부: index 페이지와 파비콘

다음과 같은 위치에 `index.html` 을 위치하면 된다.

```
classpath:/static
classpath:/public
classpath:/resources
classpath:/META-INF/resources/
```

favicon.io 에서 favicon.ico 를 다운로드 받아 앞서 언급한 static resource directory 에 복사한다. 그러나 caching 되기 때문에 favicon.ico 를 request 해보고 browser 를 껐다 켜야 한다.

## 스프링 웹 MVC 7 부: Thymeleaf

* build.gradle 에 dpendency 를 추가한다.

```groovy
dependency {
	implementation 'spring-boot-starter-thymeleaf'
}
```

* HelloControllerTest.java

```java
@RunWith(SpringRunner.class)
@WebMvcTest(HelloController.class)
public class HelloControllerTest {
	@Autowired
	MockMvc mockMvc;

	@Test
	public void hello() throws Exception {
		mocMvc.perform(get("/"))
		  .andExpect(status().isOk())
			.andExpect(view().name("hello"))
			.andExpect(model().attribute("name", is("iamslash")));
	}
}
```

* HelloController.java

```java
@Controller
public class HelloController {
	@GetMapping("/hello")
	public String hello(Mode model) {
		model.addAttribute("name", "iamslash");
		return "hello";
	}
}
```

* `src/main/resources/templates/hello.html` 을 작성한다.

```html
<!DOCTYPE HTML>
<HTML LANG="en" xmlns:th="http://www.thymeleaf.org">
<head>
  <meta charset="UTF-8">
	<title>Title</title>
</head>
<body>
<h1 th:test="${name}"></h1>
</body>
</html>
```

## 스프링 웹 MVC 8 부: HtmlUnit

html 을 unit test 하기 위한 library

* build.gradle 에 dependency 추가

```groovy
dependency {
	testImplementation 'org.seleniumhq.selenium:htmlunit-driver'
	testImplementation 'net.sourceforge.htmlunit:htmlunit'
}
```

* HelloControllerTest.java

```java
@RunWith(SpringRunner.class)
@WebMvcTest(HelloController.class)
public class HelloControllerTest {
	@Autowired
	WebClient webClient;

	@Test
	public void hello() throws Exception {
		HtmlPage page = webClient.getPage("/hello");
		Object firstByXPath = page.getFirstByXPath("//h1");
		assertThat(h1.getTextConent()).isEqualToIgnoreCase("iamslash");
	}
}
```

## 스프링 웹 MVC 9 부: ExceptionHandler

기본적으로 `org.springframework.boot.autoconfigure.web.servlet.BasicErrorController` 에 구현되어 있다.

```java
@Controller
@RequestMapping("${server.error.path:${error.path:/error}}")
public class BasicErrorController extends AbstractErrorController {

	private final ErrorProperties errorProperties;
```

`${server.error.path}` 가 정의되어 있지 않다면 `${error.path:/error}` 를 사용한다???

다음과 같이 특정 Controller 에 대해 ExceptionHandler 를 구현할 수 있다.

```java
public class HelloException extends RuntimeException {

}
```

```java
@Controller
public class HelloController {
	@GetMapping("/hello")
	public String hello() {
		throw new HelloException;
	}

	@ExceptionHandler(HelloException.class)
	public @ResponseBody AppError helloError(HelloException e) {
		AppError appError = new AppError();
		appError.setMessage("error.app.key");
		appError.setResponse("Hello Wrror")
		return appError
	}
}
```

```java
public class AppError {
	String message;
	String response;
}
```

error code 에 따라 다른 page 를 보여주자.

```
src/main/resources/static.error/5xx.html
src/main/resources/static.error/404.html
```

ErrorViewResolver 를 구현하면 더욱 customizing 할 수 있다.

## 스프링 웹 MVC 10 부: Spring HATEOAS

Hypermedia As The Engine Of Applicaiton State

REST API 를 만들 때 Resource 와 연관된 link 정보까지 같이 전달한다.

예를 들어 다음과 같은 reponse 를 전달한다.

```json
{
	"name": "iamslash",
	"links": [{
		"rel": "self",
		"href": "http://localhost:8080/hello",
	}]
}
```

다음과 같이 biuld.gradle 을 수정한다.

```groovy
dependency {
	implementation 'org.springframework.boot:spring-boot-starter-hateoas'
}
```

```java
@RunWith(SpringRunner.calss)
@WebMvcTest(HelloController.class)
public class HelloControllerTest {
	@Autowired
	MockMvc mockMvc;

	@Test
	public void hello() {
		mockMvc.perform(get("/hello"))
		  .andExpect(status().isOk())
			.andExpect(JsonPath("$._links.self").exists());
	}
}
```

```java
@RestController
public class HelloController {
	@GetMapping("/hello")
	public Hello hello() {
		Hello hello = new Hello();
		hello.setPrefx("Hey,");
		hello.setName("iamslash");

		Resource<Hello> helloResource = new Resource<>(hello);
		helloResource.add(linkTo(methodOn(HelloController.class).hello())).withSelfRel();

		return helloResource;
	}
}
```

```java
public class Hello {
	private String prefix;
	private String name;
	...
}
```

## 스프링 웹 MVC 11 부: CORS

Cross Origin Resource Sharing 의 약자이다. Single-Origin Policy 는 하나의 origin 의 request 를 허용하는 것이고 Cross-Origin Resource Sharing 은 서로 다른 Origin 의 request 를 허용하는 것이다.

A spring boot application 은 8080 port 에서 service 한다. B spring boot application 은 1080 port 에서 service 한다. `http://localhost:1080/index.html` 울 request 하면 client 는 A application 에게 Ajax 로 request 를 한다. CrossOrigin 이 설정되어 있지 않으면 error 가 발생한다.

```java
@RestController
public class HelloController {
	@GetMapping("/hello")
	public String hello() {
		return "Hello";
	}
}
```

다음은 client 의 Ajax 예이다. `src/main/resources/static/index.html`

```html
<!DOCTYPE HTML>
<HTML LANG="en">
<head>
  <meta charset="UTF-8">
	<title>Title</title>
</head>
<body>
<script src="/webjars/jquery/3.3.1/dist/jquery.min.js"></script>
<script>
  $(function() {
		$.ajax("http://localhost:8080/hello")
		  .done(function(msg) {
				alert(msg);
			})
			.fail(function() {
				alert("fail");
			});
	})
</script>
</body>
</html>
```

다음과 같이 `$CrossOrigin` 을 이용하면 여러 origin 을 허용할 수 있다.

```java
@RestController
public class HelloController {
	@CrossOrigin(origins = "http://localhost:1080")
	@GetMapping("/hello")
	public String hello() {
		return "Hello";
	}
}
```

다음과 같이 WebMvcConfigurer 를 이용하면 CrossOrigin 을 global 하게 설정할 수 있다.

```java
@Configuration
public class WebConfig implements WebMvcConfigurer {
	@Override
	public void addCorsMappings(CorsRegistry registry) {
		registry.addMapping("/hello")
		  .allowedOrigins("http://localhost:1080");
	}
}
```

## 스프링 데이터 1 부: 소개
## 스프링 데이터 2 부: 인메모리 데이터베이스

H2 를 이용한다.

다음과 같이 build.gradle 을 설정한다.

```groovy
dependency {
	implementation 'org.springframework.boot:spring-boot-starter-jdbc'
	implementation 'com.h2database:h2'
}
```

다음과 같이 `H2Runner.java` 를 작성한다.

```java
@Component
public class H2Runner implements ApplicationRunner {
	@Autowired
	DataSource dataSource;

	@Autowired
	JdbcTemplate jdbcTemplate;

	@Override
	public void run(ApplicationArguments args) throws Exception {
		try (Connection connection = dataSource.getConnection()) {
			System.out.println(connection.getMetaData().getURL());
			System.out.println(connection.getMetaData9).getUserName());

			Statement statement = connection.createStatement();
			String sql = "CREATE TABLE USER(ID INTEGER NOT NULL, name VARCHAR(255), PRIMARY KEY(id))";
			statement.executeUpdate(sql);
		}

		jdbcTemplate.execute("INSERT INTO USER VALUES(1, 'iamslash')");
	}
}
```

Open browser with `localhost:8080/h2-console/login.do`.

## 스프링 데이터 3 부: MySQL

docker 를 사용하여 mysql 를 설치 실행한다.

```bash
$ docker run -p3306:3306 --name my-mysql -e MYSQL_ROOT_PASSWORD=1 -e MYSQL_DATABASE=hello -e MYSQL_USER=iamslash -e MYSQL_PASSWORD=1 -d mysql
$ docker ps
$ docker exec -it my-mysql /bin/bash
$ mysql -u iamslash -p
mysql> show databases
mysql> use hello
```

spring boot 는 주로 Data Base Connection Pool 로 HikariCP 를 사용한다.

다음과 같이 `application.properties` 에 DBCP 설정을 한다.

```
spring.datasource.hikari.maximum-pool-size=4
spring.datasource.url=jdbc:mysql://localhost:3306/hello?userSSL=false
spring.datasource.username=iamslash
spring.datasource.password=1
```

다음과 같이 build.gradle 에 dependency 를 설정한다.

```gradle
dependency {
  implementation 'org.springframework.boot:spring-boot-starter-jdbc'
  implementation 'mysql:mysql-connector-java'
}
```

```java
@Component
public class MysqlRunner implements ApplicationRunner {
	@Autowired
	DataSource dataSource;

	@Autowired
	JdbcTemplate jdbcTemplate;

	@Override
	public void run(ApplicationArguments args) throws Exception {
		try (Connection connection = dataSource.getConnection()) {
			System.out.println(connection.getMetaData().getURL());
			System.out.println(connection.getMetaData().getUserName());

			Statement statement = connection.createStatement();
			String sql = "CREATE TABLE USER(ID INTEGER NOT NULL, name VARCHAR(255), PRIMARY KEY(id))";
			statement.executeUpdate(sql);
		}

		jdbcTemplate.execute("INSERT INTO USER VALUES(1, 'iamslash')");
	}
}
```

```bash
mysql> SELECT * FROM USER
```

## 스프링 데이터 4 부: PostgreSQL

docker 를 사용하여 postresql 를 설치 실행한다.

```bash
$ docker run -p 5432:5432 -d -e POSTGRES_PASSWORD=1 -e POSTGRES_USER=iamslash -e POSTGRES_DB=hello --rm --name my-postgres postgres
$ docker ps
$ docker exec -it postgres_boot /bin/bash

$ su - postgres
$ psql -U iamslash -W hello
> \list
> \dt
> SELECT * FROM account;
```

다음과 같이 `application.perperties` 에 DBCP 설정을 한다.

```
spring.datasource.hikari.maximum-pool-size=4
spring.datasource.url=jdbc:postgresql://localhost:5432/hello?userSSL=false
spring.datasource.username=iamslash
spring.datasource.password=1
```

다음과 같이 build.gradle 에 dependency 를 설정한다.

```gradle
dependency {
  implementation 'org.springframework.boot:spring-boot-starter-jdbc'
	implementation 'org.postgresql:postgresql'
}
```

```java
@Component
public class PostgresRunner implements ApplicationRunner {
	@Autowired
	DataSource dataSource;

	@Autowired
	JdbcTemplate jdbcTemplate;

	@Override
	public void run(ApplicationArguments args) throws Exception {
		try (Connection connection = dataSource.getConnection()) {
			System.out.println(connection.getMetaData().getURL());
			System.out.println(connection.getMetaData().getUserName());

			Statement statement = connection.createStatement();
			String sql = "CREATE TABLE USER(ID INTEGER NOT NULL, name VARCHAR(255), PRIMARY KEY(id))";
			statement.executeUpdate(sql);
		}

		jdbcTemplate.execute("INSERT INTO USER VALUES(1, 'iamslash')");
	}
}
```

## 스프링 데이터 5 부: 스프링 데이터 JPA 소개

## 스프링 데이터 6 부: 스프링 데이터 JPA 연동

다음과 같이 build.gradle 에 dependency 를 추가한다.

```groovy
dependency {
	implementation 'org.springframework.boot:spring-boot-starter-data-jpa'
}
```

다음과 같이 Account Entity Class 를 제작한다.

```java
@Entity
public class Account {
	@Id @GeneratedValue
	private Long id;
	private String username;
	private String password;
}
```

다음과 같이 Account Repository Interface 를 제작한다.

```java
public intervace AccountRepository extends JpaRepository<Account, Long> {

}
```

H2 를 build.gradle 의 test dependency 에 추가한다.

```groovy
dependency {
	testImplementation 'com.h2database:h2'
}
```

다음과 같이 build.gradle 에 dependency 를 설정한다.

```groovy
dependency {
	implementation 'org.postgresql:postgresql'
}
```

postgresql 를 실행한다.

```bash
$ docker run -p 5432:5432 --name postgres_boot -e POSTGRES_PASSWORD=pass -e POSTGRES_USER=iamslash -e POSTGRES_DB=springboot --name postgres_boot -d postgres
$ docker ps
$ docker exec -it postgres_boot /bin/bash
$ su - postgres
$ psql springboot
```

다음과 같이 `application.yml` 에 DBCP 설정을 한다.

```
spring.datasource.hikari.maximum-pool-size=4
spring.datasource.url=jdbc:postgresql://localhost:3306/springboot?userSSL=false
spring.datasource.username=iamslash
spring.datasource.password=pass
```

다음과 같이 slicing test class 를 제작한다.

```java
@RunWith(SpringRunner.class)
@DataJpaTest
public class AccountRepositoryTest {
	@Autowired
	DataSource dataSource;

	@Autowired
	JdbcTemplate jdbcTemplate;

	@Autowired
	AccountRepository accountRepository;

	@Test
	public void di() {
		try (Connection connection = dataSource.getConnection()) {
			DatabaseMetaData metaData = conneciton.getMetaData();
			System.out.println(metaData.getURL());
			System.out.println(metaData.getDriverName());
			System.out.println(metaData.getUserName());			
		}
	}

	@Test
	public void account() {
		Account account = new Account();
		account.setUsername("iamslash");
		account.setPassword("password");
		Accont newAccount = accountRepository.save(account);
		assertThat(newAccount).isNotNull();
		
		Account existingAccount = accountRepository.findByUserName(newAccount.getUsername());
		assertThat(ExistingAccount).isNotNull();

		Account nonExistingAccount = accountRepository.findByUserName("iamslash");
		assertThat(nonExistingAccount).isNotNull();
	}
}
```

Application 을 실행하면 postgresql 를 사용하고 test 를 실행하면 H2 를 사용한다.
`@DataJpaTest` 대신 `@SpringBootTest` 를 사용하면 postgresql 를 사용한다.
다음과 같이 AccountRepository Repository Interface 에 findByUserName 의 선언을 추가한다.
JPA 가 findByUserName 의 구현을 해준다.

```java
public interface AccountRepository extends JpaRepository<Account, Long> {
	Account findByUserName(String username);
}
```

다음과 같이 AccountRepository Repository Interface 에 Optional 을 사용해보자.

```java
public interface AccountRepository extends JpaRepository<Account, Long> {
	Optional<Account> findByUserName(String username);
}
```

이제 AccountRepositoryTest class 역시 Optional 을 사용해 본다.

```java
```java
@RunWith(SpringRunner.class)
@DataJpaTest
public class AccountRepositoryTest {
	@Autowired
	DataSource dataSource;

	@Autowired
	JdbcTemplate jdbcTemplate;

	@Autowired
	AccountRepository accountRepository;

	@Test
	public void di() {
		try (Connection connection = dataSource.getConnection()) {
			DatabaseMetaData metaData = conneciton.getMetaData();
			System.out.println(metaData.getURL());
			System.out.println(metaData.getDriverName());
			System.out.println(metaData.getUserName());			
		}
	}

	@Test
	public void account() {
		Account account = new Account();
		account.setUsername("iamslash");
		account.setPassword("password");
		Accont newAccount = accountRepository.save(account);
		assertThat(newAccount).isNotNull();
		
		Optional<Account> existingAccount = accountRepository.findByUserName(newAccount.getUsername());
		assertThat(ExistingAccount).isNotEmpty();

		Optional<Account> nonExistingAccount = accountRepository.findByUserName("iamslash");
		assertThat(nonExistingAccount).isEmpty();
	}
}
```

## 스프링 데이터 7 부: 데이터베이스 초기화

spring application 이 실행될때 기존의 Data Base 에 table 이 없다면 table 을 생성해 보자. table 의 column 의 이름을
바꾼다 해도 column 이 추가될 뿐 alter 가 되지는 않는다.

```
spring.jpa.hibernate.ddl-auto=update
spring.jpa.generate-ddl=true
spring.jpa.show-sql=true
```

spring application 이 실행될때 기존의 Data Base 의 table 을 삭제하고 생성해 보자.

```
spring.jpa.hibernate.ddl-auto=create
spring.jpa.generate-ddl=true
spring.jpa.show-sql=true
```

spring application 이 실행될때 기존의 Data Base 의 schema 를 validation 한다. 주로 production 에서 사용한다.

```
spring.jpa.hibernate.ddl-auto=validate
spring.jpa.generate-ddl=true
spring.jpa.show-sql=true
```

`src/main/resources/shcema.sql` 을 추가해서 Data Base 를 초기화할 수도 있다.

## 스프링 데이터 8 부: 데이터베이스 마이그레이션

Flyway 를 사용한다. Database schema 와 Data 를 versioning 할 수 있다.

다음과 같이 build.gradle 에 dependency 를 추가한다.

```groovy
dependency {
	implementation 'org.flywaydb:flyway-core'
}
```

`src/main/resources/db/migration` 에 `V숫자__이름.sql` 의 파일을 저장한다. 
`V` 는 대문자이고 `_` 가 두개임을 주의하자.

* `V1__init.sql`

```sql
drop table if exists account;
drop sequence if exists hibernate_sequence;
create sequence hibernate_sequence start with 1 increment by 1;
create table account (id bigint not null, email varchar(255), password varchar(255), username varchar(255), primary key (id));
```

spring application 이 실행될때 기존의 Data Base 의 schema 를 validation 한다. 주로 production 에서 사용한다.

```
spring.datasource.url=jdbc:postgresql://localhost:5432/springboot
spring.datasource.username=iamslash
spring.datasource.password=pass

spring.jpa.properties.hibernate.jdbc.lob.non_contextual_create=true

spring.jpa.hibernate.ddl-auto=validate
spring.jpa.generate-ddl=true
spring.jpa.show-sql=true
```

만약 schema 를 변경하고 싶다면 `V2__add_active.sql` 를 제작한다.

```sql
ALTER TABLE account ADD COLUMN active BOOLEAN;
```

Flyway 는 `flyway_schema_history` table 를 이용해서 versionning 한다.

## 스프링 데이터 9 부: Redis

다음과 같이 build.gradle 에 dependency 를 추가한다.

```groovy
dependency {
	implementation 'org.springframework.boot:spring-boot-starter-data-redis'
}
```

redis 를 실행한다.

```bash
$ docker run -p 6379:6379 --name redis_boot -d redis
$ docker exec -it redis_boot redis-cli
```

RedisRunner 를 제작한다.

```java
@Component
public class RedisRunner implements ApplicationRunner {
	@Autowired
	StringRedisTemplate redisTemplate;

	@Override
	public void run(ApplicationArguments args) throws Exception {
		ValueOperations<String, String> values = redisTemplate.opsForValue();
		values.set("Foo", "Bar");
		values.set("springboot", "2.0");
		values.set("Hello", "World");
	}
}
```

redis-cli 로 test 한다.

```bash
> keys *
> get Foo
> get springboot
> get Hello
```

redis 는 `application.yml` 을 에서 설정한다.

```
spring.redis.
```

Account Entity Class 를 제작한다.

```java
@RedisHash("accounts")
public class Account {
	@Id
	private String id;
	private String username;
	private String email;
}
```

Redis Repository Interface 를 제작한다.

```java
public interface AccountRepository extends CrudRepository<Acount, String> {

}
```

RedisRunner 를 수정한다.

```java
@Component
public class RedisRunner implements ApplicationRunner {
	@Autowired
	StringRedisTemplate redisTemplate;

	@Autowired
	AccountRepository accountRepository;

	@Override
	public void run(ApplicationArguments args) throws Exception {
		ValueOperations<String, String> values = redisTemplate.opsForValue();
		values.set("Foo", "Bar");
		values.set("springboot", "2.0");
		values.set("Hello", "World");

		Account account = new Account();
		account.setEmail("iamslash@gmail.com");
		account.setUsername("iamslash");

		accountRepository.save(account);

		Optional<Account> byId = accountRepository.findById(account.getId());
		System.out.println(byId.get().getUsername());
		System.out.println(byId.get().getEmail());
	}
}
```

```bash
> get accounts:bcsdljfdjisildifjsldfillsidfij
> hget accounts:bcsdljfdjisildifjsldfillsidfij email
> hgetall accounts:bcsdljfdjisildifjsldfillsidfij email
```

## 스프링 데이터 10 부: MongoDB

다음과 같이 build.gradle 에 dependency 를 추가한다.

```groovy
dependency {
	implementation 'org.springframework.boot:spring-boot-starter-data-mongodb'
}
```

mongoDB 를 실행한다.

```bash
$ docker run -d -p 27017:27017 --rm --name my-mongo -d mongo

$ docker exec -it my-mongo bash
> mongo
```

다음과 같이 Application class 에 ApplicationRunner 를 구현한다.

```java
@SpringBootApplication
public class ExmongoApplication {

    @Autowired
    MongoTemplate mongoTemplate;

    public static void main(String[] args) {
        SpringApplication.run(ExmongoApplication.class, args);
    }

    @Bean
    public ApplicationRunner applicationRunner() {
        return args -> {
            Account account = new Account();
            account.setEmail("iamslash@gmail.com");
            account.setUsername("iamslash");
            mongoTemplate.insert(account);
        };
    }
}
```

다음과 같이 Account Document Class 를 제작한다.

```java
@Document(collection = "accounts")
public class Account {
...
}
```

다음과 같이 mongo 를 이용하여 테스트해본다.

```bash
$ mongo
> db
> use test
> db.accounts.find({})
```

MongoTemplate 대신 Account Repository Interface 를 정의하여 mongoDB 를 접근할 수도 있다.

```java
public interface AccountRepository extends MongoRepository<Account, String> {
}
```

```java
@SpringBootApplication
public class SpringbootmongoApplication {

	@Autowired
	AccountRepository accountRepository;

	public static void main(String[] args) {
		SpringApplication.run(SpringbootmongoApplication.class, args);
	}

	@Bean
	public ApplicationRunner applicationRunner() {
		return args -> {
			Account account = new Account();
			account.setEmail("iamslash@gmail.com");
			account.setUsername("iamslash");
			accountRepository.insert(account);
		}
	}
}
```

Mongo DB Test class 를 제작해보자. 내장형 mongoDB 를 사용한다.

```groovy
dependency {
	testImplementation 'de.flapdoodle.embed:de.flapdoodle.embed.mongo'
}
```

```java
@RunWith(SpringRunner.class)
@DataMongoTest
public class ACcountRepositoryTest {
	@Autowired
	AccountRepository accountRepository;

	@Test
	public void findByEmail() {
		Account account = new Account();
		account.setUsername("iamslash");
		account.setEmail("iamslash@gmail.com");
		accountRepository.save(account);

		Optional<Account> byId = accountRepository.findById(account.getId());
		assertThat(byId).isNotEmpty();

		Optional<Account> byEmail = accountRepository.findByEmail(account.getEmail());
		assertThat(byEmail).isNotEmpty();
		assertThat(byEmail.get().getUsername()).isEqualTo("iamslash");
	}
}
```

## 스프링 데이터 11 부: Neo4J

Neo4J 는 Graph Data Base 이다.

다음과 같이 build.gradle 에 dependency 를 추가한다.

```groovy
dependency {
	implementation 'org.springframework.boot:spring-boot-starter-data-neo4j'
}
```

다음과 같이 Neo4J 를 실행한다.

```console
$ docker run -p 7474:7474 -p 7687:7687 -d --name neo4j_boot neo4j
```

browser 로 lcoalhost:7474/browser/ 를 접근한다.

다음과 같이 application.properties 를 수정한다.

```
spring.data.neo4j.password=1111
spring.data.neo4j.password=neo4j
```

다음과 같이 Runner, Account 를 제작한다.

```java
@Commponent
public class Neo4jRunner implements ApplicationRunner {

	@Autowired
	SessionFactory sessionFactory;

	@Override
	public void run(ApplicationArguments args) throws Exception {
		Account account = new Account();
		account.setEmail("iamslash@gmail.com");
		account.setUserName("iamslash");

		Session session = sessionFactory.openSession();
		session.save();
		sessionFactory.close();

		System.out.println("finished");
	}
}

@NodeEntity
public class Account{
	@Id @GeneratedValue
	private Long id;
	private String username;
	private String password;
	...
}
```

## 스프링 데이터 12 부: 정리

## 스프링 시큐리티 1 부: StarterSecurity

먼저 build.gradle 에 thymleaf dependency 를 추가한다.

```groovy
dependency {
	implementation 'org.springframework.boot:spring-boot-starter-thymeleaf'
}
```

다음과 같이 HomeController 를 추가한다.

```java
@Controller
public class HomeController {
	@GetMapping("/hello")
	public String hello() {
		return "hello";
	} 

	@GetMapping("/my")
	public String my() {
		return "my";
	}
}
```

다음과 같이 index.html, hello.html, my.html 을 `resources/templates` 에 작성한다.

* index.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
	<title>Title</title>
</head>	
<body>
<h1>Welcome</h1>
<a href="/hello">hello</a>
<a herf="/my">my page</a>
</body>
</html>
```

* hello.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
	<title>Title</title>
</head>	
<body>
<h1>Hello</h1>
</body>
</html>
```

* my.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
	<title>Title</title>
</head>	
<body>
<h1>Hello</h1>
</body>
</html>
```

다음과 같이 HelloControllerTest 를 작성한다.

```java
@RunWith(SpringRunner.class)
@WebMvcTest(HomeController.class)
public class HomeControllerTest {

	@Autowired
	MockMvc mockMvc;

	@Test
	public void hello() {
		mockMvc.perform(get("/hello"))
		  .andDo(print())
		  .andExpect(status().isOk())
			.ansExpect(view().name("hello"));
	}

	@Test
	public void my() {
		mockMvc.perform(get("/my"))
		  .andDo(print())
		  .andExpect(status().isOk())
			.ansExpect(view().name("my"));
	}
}
```

이제 Spring Security 를 추가해서 인증한 사람만 my.html 을 볼 수 있도록 하자.

build.gradle 에 spring security dependency 를 추가하자.

```groovy
dependency {
	implementation 'org.springframework.boot:spring-boot-starter-security'
}
```

이제 HelloControllerTest 는 test 를 실패한다. Spring Security 가 적용되어 모든 Request 는 Authentication 이 되어야 성공할 수 있다. 만약 `localhost:8080` 을 브라우저로 접근하면 form login 이 보여진다. username 은 `user` 이고 password 는 Spring Application 의 log 에 보여진다.

```java
@RunWith(SpringRunner.class)
@WebMvcTest(HomeController.class)
public class HomeControllerTest {

	@Autowired
	MockMvc mockMvc;

	@Test
	public void hello() {
		mockMvc.perform(get("/hello").accept(MediaType.TEXT_HTML))
		  .andDo(print())
		  .andExpect(status().isOk())
			.ansExpect(view().name("hello"));
	}

	@Test
	public void my() {
		mockMvc.perform(get("/my"))
		  .andDo(print())
		  .andExpect(status().isOk())
			.ansExpect(view().name("my"));
	}
}
```

다음과 같이 WebSecurityConfig 를 제작하면 Spring Security 가 기본적으로 제공하는 기능과 동일한 기능을 제공할 수 있다.

```java
@Configuration
public class WebSecurityConfig extends WebSecurityConfigurerAdapter {

}
```

Spring Security 가 적용된 HelloControllerTest 의 test 를 통과시켜보자.

먼저 build.gradle 에 dependency 를 추가하자.

```groovy
dependency {
	testImplementation 'org.springframework.security:spring-security-test:${spring-security.version}'
}
```

`@WithMockUser` 를 이용하면 가상으로 로그인이 되었다고 설정할 수 있다.

```java
@RunWith(SpringRunner.class)
@WebMvcTest(HomeController.class)
public class HomeControllerTest {

	@Autowired
	MockMvc mockMvc;

	@Test
	@WithMockUser
	public void hello() {
		mockMvc.perform(get("/hello").accept(MediaType.TEXT_HTML))
		  .andDo(print())
		  .andExpect(status().isOk())
			.ansExpect(view().name("hello"));
	}

	@Test
	public void hello_withoutUser() throws Exception {
		mockMvc.perform(get("/hello").
		  .andExpect(status().isUnauthorized());
	}

	@Test
	@WithMockUser
	public void my() {
		mockMvc.perform(get("/my"))
		  .andDo(print())
		  .andExpect(status().isOk())
			.ansExpect(view().name("my"));
	}
}
```

## 스프링 시큐리티 2 부: 시큐리티 설정 커스터마이징

다음과 같이 WebSecurityConfig 를 정의한다. `/, /hello` 를 제외한 request 는 Authentication 이 필요하다.

```java
@Configuration
public class WebSecurityConfig extends WebSecurityConfigureAdapter {
	@Override
	protected void configure(HttpSecurity http) throws Exception {
		http.authorizeRequests()
				.andMatchers("/", "/hello").permialAll()
				.anyRequest().authenticated()
				.and()
			.formLogin()
			  .and()
			.httpBasic();
	}
}
```

이제 user 설정을 customize 해보자.

먼저 Account 를 제작한다.

```java
@Entity
public class Account {
	@Id @GeneratedValue
	private Long id;
	private String password;
	private String email;
	...
}
```

이제 UserDetailsService 를 구현해 보자.

build.gralde 에 jpa dependency 를 추가한다.

```groovy
dependency {
	implementation 'org.springframework.boot:spring-boot-starter-data-jpa'
	implementation 'com.h2database:h2'
}
```

Account Repository 를 제작한다.

```java
public interface AccountRepository extends JpaRepository<Account, Long> {
	Optional<Account> findByUsername(String username);
}
```

AccountService 는 반드시 Bean 이어야 하고 UserDetialsService 를 상속받아야 한다. 그래야
Spring Security 는 기본적인 user 를 제공하지 않는다. loadUserByUsername 은 입력한 password 
를 검증할 때 호출된다. Spring Security 는 기본적으로 UserDetails 를 구현한 User class 를 제공한다.

```java
@Service
public class AccountService implements UserDetailsService {
	@Autowired
	private AccountRepository accountRepository;

	public Account createAccount(String username, String password) {
		Account account = new Account();
		account.setUsername(username);
		account.setPassword(password);
		return accontRepository.save(account);
	}

	@Override
	public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
		Optional<Account> byUsername = accountRespository.findByUsername(username);
		Account account = byUsername.orEleseThrow(()->new UsernameNotFoundException(username))
		return new User(account.getUsername(), account.getPassword(), authorites());
	}

	private Collection<? extends GrantedAuthority> authroties() {
		return Arrays.asList(new SimpleGrantedAuthority("ROLE_USER"));
	}
}
```

로그인할 User 를 Runner 를 이용하여 만들어 보자.

```java
@Component
public class AccountRunner implements ApplicationRunner {
	@Autowired
	AccountService accountService;

	@Override
	public void run(ApplicationArguments args) throws Exception {
		accountService.createAccount("iamslash", "1234");
	}
}
```

password 를 DB 에 그대로 저장하기 때문에 login 이 되지 않는다. `There is no PasswordEncoder mapped for the id "null"`.

가장 간단하게 해결할 수 있는 방법은 NoOpPasswordEncoder 를 Bean 으로 제공하는 것이다. 그러나
password 를 DB 에 그대로 저자하기 때문에 매우 안 좋은 방법이다.

```java
@Configuration
public class WebSecurityConfig extends WebSecurityConfigureAdapter {
	@Override
	protected void configure(HttpSecurity http) throws Exception {
		http.authorizeRequests()
				.andMatchers("/", "/hello").permialAll()
				.anyRequest().authenticated()
				.and()
			.formLogin()
			  .and()
			.httpBasic();
	}
	@Bean
	public PasswordEncoder passwordEncoder() {
		return NoOpPasswordEncoder.getInstance();
	}
}
```

Spring Security 는 `PasswordEncoderFactories.createDelegatingPasswordEncoder()` 를 권장한다.

```java
@Configuration
public class WebSecurityConfig extends WebSecurityConfigureAdapter {
	@Override
	protected void configure(HttpSecurity http) throws Exception {
		http.authorizeRequests()
				.andMatchers("/", "/hello").permialAll()
				.anyRequest().authenticated()
				.and()
			.formLogin()
			  .and()
			.httpBasic();
	}
	@Bean
	public PasswordEncoder passwordEncoder() {
		return PasswordEncoderFactories.createDelegatingPasswordEncoder();
	}
}
```

그럼 PasswordEncoder 를 AccountService 에 적용하여 password 를 encoding 하여 저장한다.

```java
@Service
public class AccountService implements UserDetailsService {
	@Autowired
	private AccountRepository accountRepository;
	
	@Autowired
	private PasswordEncoder passwordEncoder;

	public Account createAccount(String username, String password) {
		Account account = new Account();
		account.setUsername(username);
		account.setPassword(passwordEncoder.encode(pssword));
		return accontRepository.save(account);
	}

	@Override
	public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
		Optional<Account> byUsername = accountRespository.findByUsername(username);
		Account account = byUsername.orEleseThrow(()->new UsernameNotFoundException(username))
		return new User(account.getUsername(), account.getPassword(), authorites());
	}

	private Collection<? extends GrantedAuthority> authroties() {
		return Arrays.asList(new SimpleGrantedAuthority("ROLE_USER"));
	}
}
```

## 스프링 REST 클라이언트 1 부: RestTemplate vs WebClient

RestTemplateBuilder 혹은 WebClientBuilder 를 Bean 으로 주입받아서 사용한다.

RestTemplate 는 Synchronous, WebClient 는 Asynchronous API 이다.

다음은 RestTemplate 을 사용하는 예이다.

```java
@Component
public class RestRunner implements ApplicationRunner {
	@Autowired
	RestTemplateBuilder restTemplateBuilder;

	@Override
	public void run(ApplicationArguments args) throws Exception {
		RestTemplate restTemplate = restTemplateBuilder.build();

		StopWatch stopWatch = new StopWatch();
		stopWatch.start();

		String helloResult = restTemplate.getForObject("http://localhost:8080/hello");
		System.out.println(helloResult);

		String worldResult = restTemplate.getForObject("http://localhost:8080/world");
		System.out.println(worldResult);

		stopWatch.stop();
		System.out.println(stopWatch.prettyPrint());
	}
}
```

다음은 WebClient 을 사용하는 예이다. 먼저 webflux 를 build.gradle 에 추가해야 한다.


```java
@Component
public class RestRunner implements ApplicationRunner {
	@Autowired
	WebClient.Builder builder;

	@Override
	public void run(ApplicationArguments args) throws Exception {
		WebClient webClient = builder.build();

		StopWatch stopWatch = new StopWatch();
		stopWatch.start();
		
		Mono<String> helloMono = webClient.get().uri("http://localhost:8080/hello")
		  .retrieve()
			.bodyToMono(String.class)
		helloMono.subscribe(s -> {
				System.out.println(s);
				if (stopWatch.isRunning()) {
					stopWatch.stop();
				}
				System.out.println(stopWatch.prettyPrint());
				stopWatch.start();
			});
		Mono<String> worldMono = webClient.get().uri("http://localhost:8080/world")
		  .retrieve()
			.bodyToMono(String.class)
		worldMono.subscribe(s -> {
				System.out.println(s);
				if (stopWatch.isRunning()) {
					stopWatch.stop();
				}
				System.out.println(stopWatch.prettyPrint());
				stopWatch.start();
			});
	}
}
```

## 스프링 REST 클라이언트 2 부: Customizing

## 그밖에 다양한 기술 연동

# 스프링 부트 운영

## 스프링 부트 Actuator 1 부: 소개

Actuator 는 운영을 위한 여러 정보를 endpoint 들을 통해 제공해 준다.

Actuator 를 사용하기 위해 다음과 같이 build.gradle 에 dependency 를 추가한다.

```groovy
dependency {
	implemnentation 'org.springframework.boot:spring-boot-starter-actuator'
}
```

open browser `http://localhost:8080/actuator`

다음과 같이 application.properties 에서 endpoint 를 활성화 할 수 있다. 반드시 expose 를 해야 browser 에서 볼 수 있다.

```
management.endpoint.shutdown.enabled=true
```

다음과 같이 application.properties 에서 endpoint 를 expose 한다.

```
management.endpoints.jmx.exposure.include=health,info
```

## 스프링 부트 Actuator 2 부: JMX 와 HTTP

JMX 는 `jconsole` 로 확인 가능하다. 그러나 불편하다.

`jvisualvm` 이 훨씬 좋다. 그러나 설치를 해야한다.

만약 browser 에서 모두 보고 싶다면 다음과 같이 application.properties 를
수정한다.

```
management.endpoints.web.exposure.include=*
management.endpoints.web.exposure.exclude=env.beans
```

open url `http://localhost:8080/actuator/beans` 

그러나 가독성이 떨어지는 것은 똑같다. Spring Security 를 이용하면 endpoint 를
인증된 유저만 볼 수 있게 할 수 있다.

## 스프링 부트 Actuator 3 부: 스프링 부트 어드민

[Spring boot admin](https://github.com/codecentric/spring-boot-admin) 은 JMX 의 UI 이다. opensource 이다.

prometheus, grafana 를 사용하겠다.
