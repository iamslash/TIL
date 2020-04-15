# Materials

* [스프링 부트 개념과 활용 @ inflearn](https://www.inflearn.com/course/%EC%8A%A4%ED%94%84%EB%A7%81%EB%B6%80%ED%8A%B8)

## Tutorial of STS

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

## Tutorial of springboot 

- [springboot manual installation](https://docs.spring.io/spring-boot/docs/current/reference/html/getting-started-installing-spring-boot.html#getting-started-manual-cli-installation)
  에서 zip을 다운받아 `d:\local\src\`에 압축을 해제하자.
- 환경설정변수 SPRING_HOME을 만들자. `d:\local\src\D:\local\src\spring-1.5.6.RELEASE`
- 환경설정변수 PATH에 `%SPRING_HOME%\bin`을 추가하자.
- command shell에서 `spring version`을 실행한다.
- command shell에서 `spring init'을 이용하여 새로운 프로젝트를 제작할 수 있다.


# 스프링 부트 원리

## 의존성 관리 이해

## 의존성 관리 응용

## 자동 설정 이해

## 자동설정 만들기 1 부 : Starter 와 AutoConfigure

## 자동설정 만들기 2 부 : @ConfigurationProperties

## 내장 웹 서버 이해

## 내장 웹 서버 응용 1 부: 컨테이너와 포트

## 내장 웹 서버 응용 2 부: HTTPS 와 HTTP2

## 톰캣 HTTP2

## 독립적으로 실행가능한 JAR

## 스프링 부트 원리 정리

# 스프링 부트 활용

## 스프링 부트 활용 소개

## SpringApplication 1 부

## SpringApplication 2 부

## 외부 설정 1 부

## 외부 설정 2 부 (1)

## 외부 설정 2 부 (2)

## 프로파일

## 로깅 1부 : 스프링 부트 기본 로거설정

## 로깅 2부 : 커스터마이징

## 테스트
## 테스트 유틸
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
$ docker run -p3306:3306 --name mysql_boot -e MYSQL_ROOT_PASSWORD=1 -e MYSQL_DATABASE=springboot -e MYSQL_USER=iamslash -e MYSQL_PASSWORD=pas -d mysql
$ docker ps
$ docker exec -it mysql_boot /bin/bash
$ mysql -u iamslash -p
mysql> show databases
mysql> use springboot
```

spring boot 는 주로 Data Base Connection Pool 로 HikariCP 를 사용한다.

다음과 같이 `application.yml` 에 DBCP 설정을 한다.

```
spring.datasource.hikari.maximum-pool-size=4
spring.datasource.url=jdbc:mysql://localhost:3306/springboot?userSSL=false
spring.datasource.username=iamslash
spring.datasource.password=pass
```

다음과 같이 build.gradle 에 dependency 를 설정한다.

```
dependency {
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
			System.out.println(connection.getMetaData9).getUserName());

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
$ docker run -p 5432:5432 --name postgres_boot -e POSTGRES_PASSWORD=pass -e POSTGRES_USER=iamslash -e POSTGRES_DB=springboot --name postgres_boot -d postgres
$ docker ps
$ docker exec -it postgres_boot /bin/bash
$ su - postgres
$ psql springboot
> \list
> \dt
> SELECT * FROM account;
```

다음과 같이 `application.yml` 에 DBCP 설정을 한다.

```
spring.datasource.hikari.maximum-pool-size=4
spring.datasource.url=jdbc:postgresql://localhost:3306/springboot?userSSL=false
spring.datasource.username=iamslash
spring.datasource.password=pass
```

다음과 같이 build.gradle 에 dependency 를 설정한다.

```
dependency {
	implementation 'org.postgresql:postgresql'
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
$ docker run -p 27017:27017 --name mongo_boot -d mongo
$ docker exec -it mongo_boot bash
$ mongo
```

다음과 같이 Application class 에 ApplicationRunner 를 구현한다.

```java
@SpringBootApplication
public class SpringbootmongoApplication {

	@Autowired
	MongoTemplate mongoTemplate;

	public static void main(String[] args) {
		SpringApplication.run(SpringbootmongoApplication.class, args);
	}

	@Bean
	public ApplicationRunner applicationRunner() {
		return args -> {
			Account account = new Account();
			account.setEmail("iamslash@gmail.com");
			account.setUsername("iamslash");
			mongoTemplate.insert(account);
		}
	}
}
```

다음과 같이 Account Document Class 를 제작한다.

```java
@Document(collection = "accounts")
public class Account {

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
## 스프링 데이터 12 부: 정리

## 스프링 시큐리티 1 부: StarterSecurity
## 스프링 시큐리티 2 부: 시큐리티 설정 커스터마이징

## 스프링 REST 클라이언트 1 부: RestTemplate vs WebClient
## 스프링 REST 클라이언트 2 부: Customizing
## 그밖에 다양한 기술 연동

# 스프링 부트 운영

## 스프링 부트 Actuator 1 부: 소개

## 스프링 부트 Actuator 2 부: JMX 와 HTTP

## 스프링 부트 Actuator 3 부: 스프링 부트 어드민

