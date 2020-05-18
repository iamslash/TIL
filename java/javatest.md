- [Abstract](#abstract)
- [Materials](#materials)
- [JUnit5](#junit5)
  - [Junit 5 소개](#junit-5-%ec%86%8c%ea%b0%9c)
  - [JUnit 5 시작하기](#junit-5-%ec%8b%9c%ec%9e%91%ed%95%98%ea%b8%b0)
  - [JUnit 5 테스트 이름 표시하기](#junit-5-%ed%85%8c%ec%8a%a4%ed%8a%b8-%ec%9d%b4%eb%a6%84-%ed%91%9c%ec%8b%9c%ed%95%98%ea%b8%b0)
  - [JUnit 5 Assertion](#junit-5-assertion)
  - [JUnit 5 조건에 따라 테스트 실행하기](#junit-5-%ec%a1%b0%ea%b1%b4%ec%97%90-%eb%94%b0%eb%9d%bc-%ed%85%8c%ec%8a%a4%ed%8a%b8-%ec%8b%a4%ed%96%89%ed%95%98%ea%b8%b0)
  - [JUnit 5 태깅과 필터링](#junit-5-%ed%83%9c%ea%b9%85%ea%b3%bc-%ed%95%84%ed%84%b0%eb%a7%81)
  - [JUnit 5 테스트 반복하기 1부](#junit-5-%ed%85%8c%ec%8a%a4%ed%8a%b8-%eb%b0%98%eb%b3%b5%ed%95%98%ea%b8%b0-1%eb%b6%80)
  - [JUnit 5 테스트 반복하기 2부](#junit-5-%ed%85%8c%ec%8a%a4%ed%8a%b8-%eb%b0%98%eb%b3%b5%ed%95%98%ea%b8%b0-2%eb%b6%80)
  - [JUnit 5 테스트 인스턴스](#junit-5-%ed%85%8c%ec%8a%a4%ed%8a%b8-%ec%9d%b8%ec%8a%a4%ed%84%b4%ec%8a%a4)
  - [JUnit 5 테스트 순서](#junit-5-%ed%85%8c%ec%8a%a4%ed%8a%b8-%ec%88%9c%ec%84%9c)
  - [JUnit 5 junit-platform.properties](#junit-5-junit-platformproperties)
  - [JUnit 5 확장 모델](#junit-5-%ed%99%95%ec%9e%a5-%eb%aa%a8%eb%8d%b8)
  - [JUnit 5 migration](#junit-5-migration)
  - [Junit 5 연습문제](#junit-5-%ec%97%b0%ec%8a%b5%eb%ac%b8%ec%a0%9c)
- [Mockito](#mockito)
  - [Mockito 소개](#mockito-%ec%86%8c%ea%b0%9c)
  - [Mockito 시작하기](#mockito-%ec%8b%9c%ec%9e%91%ed%95%98%ea%b8%b0)
  - [Mock 객체 만들기](#mock-%ea%b0%9d%ec%b2%b4-%eb%a7%8c%eb%93%a4%ea%b8%b0)
  - [Mock 객체 Stubbing](#mock-%ea%b0%9d%ec%b2%b4-stubbing)
  - [Mock 객체 Stubbing 연습 문제](#mock-%ea%b0%9d%ec%b2%b4-stubbing-%ec%97%b0%ec%8a%b5-%eb%ac%b8%ec%a0%9c)
  - [Mock 객체 확인](#mock-%ea%b0%9d%ec%b2%b4-%ed%99%95%ec%9d%b8)
  - [BDD 스타일 Mockito API](#bdd-%ec%8a%a4%ed%83%80%ec%9d%bc-mockito-api)
  - [Mockito 연습문제](#mockito-%ec%97%b0%ec%8a%b5%eb%ac%b8%ec%a0%9c)
- [Docker and Test](#docker-and-test)
  - [Testcontainers 소개](#testcontainers-%ec%86%8c%ea%b0%9c)
  - [Testcontainers 설치](#testcontainers-%ec%84%a4%ec%b9%98)
  - [컨테이너 정보를 스프링 테스트에서 참조하기](#%ec%bb%a8%ed%85%8c%ec%9d%b4%eb%84%88-%ec%a0%95%eb%b3%b4%eb%a5%bc-%ec%8a%a4%ed%94%84%eb%a7%81-%ed%85%8c%ec%8a%a4%ed%8a%b8%ec%97%90%ec%84%9c-%ec%b0%b8%ec%a1%b0%ed%95%98%ea%b8%b0)
  - [Testcontainers Docker Compose 사용하기 1 부](#testcontainers-docker-compose-%ec%82%ac%ec%9a%a9%ed%95%98%ea%b8%b0-1-%eb%b6%80)
  - [Testcontainers Docker Compose 사용하기 2 부](#testcontainers-docker-compose-%ec%82%ac%ec%9a%a9%ed%95%98%ea%b8%b0-2-%eb%b6%80)
- [Performance Test](#performance-test)
  - [JMeter 소개](#jmeter-%ec%86%8c%ea%b0%9c)
  - [JMeter 설치](#jmeter-%ec%84%a4%ec%b9%98)
  - [JMeter 사용하기](#jmeter-%ec%82%ac%ec%9a%a9%ed%95%98%ea%b8%b0)
- [Operation Issue Test](#operation-issue-test)
  - [Chaos Monkey 소개](#chaos-monkey-%ec%86%8c%ea%b0%9c)
  - [CM4SB 설치](#cm4sb-%ec%84%a4%ec%b9%98)
  - [CM4SB 응답 지연](#cm4sb-%ec%9d%91%eb%8b%b5-%ec%a7%80%ec%97%b0)
  - [CM4SB 에러 발생](#cm4sb-%ec%97%90%eb%9f%ac-%eb%b0%9c%ec%83%9d)
- [Architecture Test](#architecture-test)
  - [ArchUnit 소개](#archunit-%ec%86%8c%ea%b0%9c)
  - [ArchUnit 설치](#archunit-%ec%84%a4%ec%b9%98)
  - [ArchUnit 패키지 의존성 확인하기](#archunit-%ed%8c%a8%ed%82%a4%ec%a7%80-%ec%9d%98%ec%a1%b4%ec%84%b1-%ed%99%95%ec%9d%b8%ed%95%98%ea%b8%b0)
  - [ArchUnit JUnit 5 얀동](#archunit-junit-5-%ec%96%80%eb%8f%99)
  - [ArchUnit 클래스 의존성 확인하기](#archunit-%ed%81%b4%eb%9e%98%ec%8a%a4-%ec%9d%98%ec%a1%b4%ec%84%b1-%ed%99%95%ec%9d%b8%ed%95%98%ea%b8%b0)

----

# Abstract

java TDD 를 정리한다.

# Materials

* [더 자바, 애플리케이션을 테스트하는 다양한 방법 by 백기선](https://www.inflearn.com/course/the-java-application-test)
  * [src](https://github.com/keesun/inflearn-the-java-test)
* [JUnit5 in Spring](https://brunch.co.kr/@springboot/77)

# JUnit5

## Junit 5 소개

* [Junit5-samples](https://github.com/junit-team/junit5-samples/tree/master/junit5-jupiter-starter-gradle)
  * [junit5-jupiter-starter-gradle](https://github.com/junit-team/junit5-samples/tree/master/junit5-jupiter-starter-gradle)

----

![](img/junit5.png)

JUnit 5 는 JUnitPlatform, Jupiter, Vintage 과 같이 3 개의 component 로 구성된다. 
JUnit Vintage 는 JUnit3, JUnit4 를 위한 test engine 을 제공한다.
JUnit Jupiter 는 JUnit5 를 위한 test engine 을 제공한다.
JUnit Platform 은 test engine interface 를 제공한다.

## JUnit 5 시작하기

[02. JUnit 시작하기 src](https://github.com/keesun/inflearn-the-java-test/commit/cf41bcd4b2d85d6c956e31d8af5e2f2e339d92fa)

----

만약 spring boot application 이 아니라면 즉, `org.springframework.boot:spring-boot-starter-test` 가 dpendency 에 없다면  `org.junit.jupiter:junit-jupiter-engine:5.5.2` 를 dependency 에 추가해 준다.

다음은 기본적인 JUnit 5 code 이다. `@Disabled` 는 test 실행을 잠시 꺼둘 수 있다.

```java
class FooServiceTest {
  @Test
  void create() {
    FooService fooService = new FooService();
    assertNotNull(fooService);
  }

  @Before All
  static void beforeAll() {
    System.out.println("before all");
  }

  @AfterAll
  static void afterAll() {
    System.out.println("after all");
  }

  @BeforeEach
  @Disabled
  static void beforeEach() {
    System.out.println("before each");
  }

  @AfterEach
  static void afterEach() {
    System.out.println("after each");
  }
}
```

## JUnit 5 테스트 이름 표시하기

* [03. 테스트 이름 표기 src](https://github.com/keesun/inflearn-the-java-test/commit/d9b5f3b51c2788365d2bdbeecbb0b2b5c50fdf5f)

----

`DisplayNameGeneration` 을 사용하면 global 하게 test 실행시 표시된 이름을 조작할 수 있다. `@DisplayName` 을 사용하면 특정 method 의 test 실행시 표시될 이름을 조작할 수 있다.

```java
@DisplayNameGeneration(DisplayNameGenerator.ReplaceUnderscores.class)
class FooServiceTest {} {
  @Test
  @DisplayName("푸 만들기")
  void create_new_foo() {
    FooService fooService = new FooService();
    assertNotNull(fooService);
    System.out.println("create");
  }
}
```

## JUnit 5 Assertion

* [04. Assertion src](https://github.com/keesun/inflearn-the-java-test/commit/fce5cc4e381ba65fc3a566453fd427cea64aaced)

----


`assertEquals, assertNotNull, assertTrue, assertAll, assertThrows, assertTimeout` 을 이용하며 asserting 하자.

```java
@DisplayNameGeneration(DisplayNameGenerator.ReplaceUnderscores.class)
class FooServiceTest {} {
  @Test
  @DisplayName("푸 만들기")
  void create_new_foo() {
    FooService fooService = new FooService();
    assertNotNull(fooService);
    assertEquals(fooService.DRAFT, fooService.getStatus(), () -> "Done.");
    assertTrue(fooService.age > 0, "Exceeded age.")
    System.out.println("create");
  }
}
```

`assertAll` 은 여러 lambda expression 을 이용하여 asserting 을 한번에 할 수 있다.

```java
@DisplayNameGeneration(DisplayNameGenerator.ReplaceUnderscores.class)
class FooServiceTest {
  @Test
  @DisplayName("푸 만들기")
  void create_new_foo() {
    FooService fooService = new FooService();
    assertAll(
      () -> assertNotNull(fooService),
      () -> assertEquals(fooService.DRAFT, fooService.getStatus(), () -> "Done."),
      () -> assertTrue(fooService.age > 0, "Exceeded age.")
    );
  }
}
```

`assertThrows` 는 특정 조건에 특정 exception 이 발생하는지 asserting 할 수 있다. `assertTimeout`
은 특정 구문을 실행했을 때 timeout 을 asserting 할 수 있다.

```java
@DisplayNameGeneration(DisplayNameGenerator.ReplaceUnderscores.class)
class FooServiceTest {
  @Test
  @DisplayName("푸 만들기")
  void create_new_foo() {
    assertThrows(IllegalArgumentException.class, () -> new FooService(-1));
    assertTimeout(Duration.ofMillis(100), () -> {
      new FooService(10);
      Thread.Sleep(300);
    });
  }
  ...
}
```

`assumeTrue, assumingThat` 은 특정 조건에서만 asserting 할 수 있게 한다.

```java
@DisplayNameGeneration(DisplayNameGenerator.ReplaceUnderscores.class)
class FooServiceTest {
  @Test
  @DisplayName("푸 만들기")
  void create_new_foo() {
    String env = System.getenv("FOO");
    System.out.println(env);
    assumeTrue("LOCAL".equalsIgnoreCase(env));
  }
  ...
}
```

```java
@DisplayNameGeneration(DisplayNameGenerator.ReplaceUnderscores.class)
class FooServiceTest {
  @Test
  @DisplayName("푸 만들기")
  void create_new_foo() {
    String env = System.getenv("FOO");
    assumingThat("LOCAL".equalsIgnoreCase(env), () -> {
      System.out.println("local");
      ...
    });
  }
  ...
}
```

## JUnit 5 조건에 따라 테스트 실행하기

* []()

----

## JUnit 5 태깅과 필터링

* []()

----

## JUnit 5 테스트 반복하기 1부

* []()

----

## JUnit 5 테스트 반복하기 2부

* []()

----

## JUnit 5 테스트 인스턴스

* []()

----

## JUnit 5 테스트 순서

* []()

----

## JUnit 5 junit-platform.properties

* []()

----

## JUnit 5 확장 모델

* []()

----

## JUnit 5 migration

* []()

----

## Junit 5 연습문제

* []()

----

# Mockito

* []()

----

## Mockito 소개

* []()

----

## Mockito 시작하기

* []()

----

## Mock 객체 만들기

* []()

----

## Mock 객체 Stubbing

* []()

----

## Mock 객체 Stubbing 연습 문제

* []()

----

## Mock 객체 확인

* []()

----

## BDD 스타일 Mockito API

* []()

----

## Mockito 연습문제

* []()

----

# Docker and Test

## Testcontainers 소개

* []()

----

## Testcontainers 설치

* []()

----

## 컨테이너 정보를 스프링 테스트에서 참조하기

## Testcontainers Docker Compose 사용하기 1 부

## Testcontainers Docker Compose 사용하기 2 부

# Performance Test

## JMeter 소개

## JMeter 설치

## JMeter 사용하기

# Operation Issue Test

## Chaos Monkey 소개

## CM4SB 설치

## CM4SB 응답 지연

## CM4SB 에러 발생

# Architecture Test

## ArchUnit 소개

## ArchUnit 설치

## ArchUnit 패키지 의존성 확인하기

## ArchUnit JUnit 5 얀동

## ArchUnit 클래스 의존성 확인하기

----

`@Enabled__, @Disabled__, EnabledIfEnvironmentVariable` 는 특정 OS, JRE, ENV 등등에서 test code 실행을 제어한다.

```java
@DisplayNameGeneration(DisplayNameGenerator.ReplaceUnderscores.class)
class FooServiceTest {
  @Test
  @DisplayName("푸 만들기")
  @EnabledOnOS({OS.MAX, OS.LINUX})
  @EnabledOnJre({JRE.JAVA_8})
  @EnabledIfEnvironmentVariable(named = "FOO")
  void create_new_foo() {
    String env = System.getenv("FOO");
    assumingThat("LOCAL".equalsIgnoreCase(env), () -> {
      System.out.println("local");
      ...
    });
  }
  ...
}
```

`@Tag` 를 이용하여 코드를 작성한다. 그리고 intelliJ 에서 `EditConfiguration | Test kind | Tags | Tag expression` 에 특정 tag expression ("slow | fast") 을 입력하고 test 를 실행하면 특정 tag 의 test code 만 실행된다.

```java
@DisplayNameGeneration(DisplayNameGenerator.ReplaceUnderscores.class)
class FooServiceTest {
  @Test
  @DisplayName("푸 만들기")
  @Tag("bar")
  void create_new_foo() {
    String env = System.getenv("FOO");
    assumingThat("LOCAL".equalsIgnoreCase(env), () -> {
      System.out.println("local");
      ...
    });
  }
  ...
}
```

다음과 같이 Annotation 을 이용하여 Custom tag 를 만들어 보자.

```java
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Tag("bar")
@Test
public @interface BarTest {  
}
```

```java
@BarTest
@DisplayName("Bar test")
void create_new_bar() {
}
```

`RepeatedTest, ParameterizedTest` 를 이용하여 반복 테스트를 할 수 있다.

```java
@RepeatedTest(10)
void repeatTest(RepetitionInfo repetitionInfo) {
  System.out.println("Test " + repetitionInfo.getCurrentRepetition() + "/" + repetitionInfo.getTotalRepetitions());
}

@RepeatedTest(value=10, name = "{displayName}, {currentRepetition}/{totalRepetition}")
void repeatTest(RepetitionInfo repetitionInfo) {
  System.out.println("Test " + repetitionInfo.getCurrentRepetition() + "/" + repetitionInfo.getTotalRepetitions());
}

@ParameterizedTest
@ValueSource(string = {"Foo", "Bar", "Baz"})
void parameterizedTest(String msg) {
  System.out.println(msg);
}

@DisplayName("Parameterized Test")
@ParameterizedTest(name = "{index} {displayName} message={0}")
@ValueSource(string = {"Foo", "Bar", "Baz"})
void parameterizedTest(String msg) {
  System.out.println(msg);
}
```

`@ValueSource, @NullSource, @EmptySource, @NullAndEmptySource, @EnumSource, @MethodSource, @CvsSource, @CvsFileSource, @ArgumentSource` 를 이용하여 ParameterizedTest 의 argument value source 를 다양하게 설정할 수 있다.

JUnit 은 test method 마다 test class 의 instance 를 새로 생성한다. test method 의 실행 순서에 상관없이 test 가 가능하도록 하기 위함이다. `@TestInstance` 를 이용하면 test class 하나에 instance 를 하나만 생성할 수 있다. 따라서 `@BeforeAll, @AfterAll, @BeforeEach, @AfterEach` 가 부탁된 method 들은 static 일 필요가 없다.

```java
@TestInstance(Lifecycle.PER_CLASS)
class FooServiceTest {
}
```

`@TestMethodOrder` 를 이용하여 test method 의 실행순서를 변경할 수 있다.

```java
@TestInstance(TestInstance.LifeCycle.PER_CLASS)
@TestMethodOrder(MethodOrder.OrderAnnotation.class)
class FooServiceTest {
  @Order(1)
  void create_new_foo() {
  }
  @Order(2)
  void create_new_bar() {
  }
}
```

`test/resources/junit-platform.properties` 를 생성하여 JUnit 설정을 저장하자.

```
junit.jupiter.testinstance.lifecycle.default = per_class
junit.jupiter.extensions.autodetection.enabled = true
junit.jupiter.conditions.deactivate = org.junit."DisabledCondition"
junit.jupiter.displayname.generator.default = org.junit.jupiter.api.DisplayNameGenerator@REplaceUnderscores
```

JUnit 5 는 `Extension` 을 이용하여 확장할 수 있다.

```java

public class FindSlowTestExtention implements BeforeTestExecutionCallback, AfterTestExecutionCallback {
  @Override 
  public void beforeTestExecution(ExtensionContext ctx) throws Exception {
    ExtensionContext.Store store = getStore(ctx);
    store.put("START_TIME", System.currentTimeMillis());
  }

  @Override
  public void afterTestExecution(ExtensionContext ctx) throws Exception {
    String testMethodName = ctx.getRequiredTestMethod().getName();
    ExtensionContext.Store store = getStore(ctx)
    long start_time = store.remove("START_TIME", long.class);
    long duration = System.currentTimeMillis() - start_time;
    if (duration > THRESHOLD) {
      System.out.printf("Please consider mark method [%s] with @SlowTest.\n", testMethodName);
    }
  }

  private ExtensionContext.Store getStore(ExtensionContext ctx) {
    String testClassName = ctx.getRequiredTestClass().getName();
    String testMethodName = ctx.getRequiredTestMethod().getName();
    return ctx.getStore(ExtensionContext.Namespace.create(testClassName, testMethodName));
  }
}
```

가짜 객체를 Mock 객체라고 한다. [Mockito](https://site.mockito.org/) 는 Mock 객체를 쉽게 만들고 관리할 수 있도록 하는 library 이다.
spring boot 를 사용하지 않는다면 `org.mockito:mockito-core:3.1.0` 과 `org.mockito:mockito-junit-jupiter:3.1.0` 를 dependency 로 추가한다.

```java
BarService barService = mock(BarService.class)
FooRepository fooRepository = mock(FooRepository.class)
```

```java
@ExtendWith(MockitoExtension.clas)
class FooServiceTest {
  @Mock BarService barService;
  @Mock FooRepository fooRepository;
}
```

```java
@ExtendWith(MockitoExtension.clas)
class FooServiceTest {

  @Test
  void createFooService(@Mock BarService barService,
                        @Mock FooRepository fooRespotiry) {
    FooService fooService = new FooService(barService, fooRepository);
    assertNotNull(fooService);
  }
}
```

Mock 객체의 행동을 조작하는 것을 Stubbing 이라고 한다.

```java
@ExtendWith(MockitoExtension.clas)
class FooServiceTest {

  @Test
  void createFooService(@Mock BarService barService,
                        @Mock FooRepository fooRespotiry) {
    FooService fooService = new FooService(barService, fooRespository);
    assertNotNull(fooService);

    Person person = new Person();
    person.setId(1L);
    person.setEmail("iamslash@gmail.com");
    when(barService.findById(1L)).thenReturn(Optional.of(member))
    Lecture lecture = new Lecture(10, "java");
    Optional<Person> findId = barService.findById(1L);
    assertEquals("iamslash@gmail.com", findById.get().getEmail());
  }
}
```

