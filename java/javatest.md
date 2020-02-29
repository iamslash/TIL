# Abstract

java TDD 를 정리한다.

# Materials

* [더 자바, 애플리케이션을 테스트하는 다양한 방법 by 백기선](https://www.inflearn.com/course/the-java-application-test)
  * [src](https://github.com/keesun/inflearn-the-java-test)

# Basic

JUnit 는 JUnitPlatform, Jupiter, Vintage 등으로 구성된다. Jupiter 는 JUnitPlatform 을 구현한 API 이다. JUnit 5 에 해당한다. Vintage 는 JUnit 3, 4 에 해당한다.

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

