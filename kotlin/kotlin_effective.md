- [Abstract](#abstract)
- [Matrials](#matrials)
- [Part 1: Good code](#part-1-good-code)
  - [Chapter 1: Safety](#chapter-1-safety)
    - [Item 1: Limit mutability](#item-1-limit-mutability)
    - [Item 2: Minimize the scope of variables](#item-2-minimize-the-scope-of-variables)
    - [Item 3: Eliminate platform types as soon as possible](#item-3-eliminate-platform-types-as-soon-as-possible)
    - [Item 4: Do not expose inferred types](#item-4-do-not-expose-inferred-types)
    - [Item 5: Specify your expectations on arguments and state](#item-5-specify-your-expectations-on-arguments-and-state)
    - [Item 6: Prefer standard errors to custom ones](#item-6-prefer-standard-errors-to-custom-ones)
    - [Item 7: Prefer null or Failure result when the lack of result is possible](#item-7-prefer-null-or-failure-result-when-the-lack-of-result-is-possible)
    - [Item 8: Handle nulls properly](#item-8-handle-nulls-properly)
    - [Item 9: Close resources with use](#item-9-close-resources-with-use)
    - [Item 10: Write unit tests](#item-10-write-unit-tests)
  - [Chapter 2: Readability](#chapter-2-readability)
    - [Item 11: Design for readability](#item-11-design-for-readability)
    - [Item 12: Operator meaning should be clearly consistent with its function name](#item-12-operator-meaning-should-be-clearly-consistent-with-its-function-name)
    - [Item 13: Avoid returning or operating on Unit?](#item-13-avoid-returning-or-operating-on-unit)
    - [Item 14: Specify the variable type when it is not clear](#item-14-specify-the-variable-type-when-it-is-not-clear)
    - [Item 15: Consider referencing receivers explicitly](#item-15-consider-referencing-receivers-explicitly)
    - [Item 16: Properties should represent state, not behavior](#item-16-properties-should-represent-state-not-behavior)
    - [Item 17: Consider naming arguments](#item-17-consider-naming-arguments)
    - [Item 18: Respect coding conventions](#item-18-respect-coding-conventions)
- [Part 2: Code design](#part-2-code-design)
  - [Chapter 3: Reusability](#chapter-3-reusability)
    - [Item 19: Do not repeat knowledge](#item-19-do-not-repeat-knowledge)
    - [Item 20: Do not repeat common algorithms](#item-20-do-not-repeat-common-algorithms)
    - [Item 21: Use property delegation to extract common property patterns](#item-21-use-property-delegation-to-extract-common-property-patterns)
    - [Item 22: Use generics when implementing common algorithms](#item-22-use-generics-when-implementing-common-algorithms)
    - [Item 23: Avoid shadowing type parameters](#item-23-avoid-shadowing-type-parameters)
    - [Item 24: Consider variance for generic types](#item-24-consider-variance-for-generic-types)
    - [Item 25: Reuse between different platforms by extracting common modules](#item-25-reuse-between-different-platforms-by-extracting-common-modules)
  - [Chapter 4: Abstraction design](#chapter-4-abstraction-design)
    - [Item 26: Each function should be written in terms of a single level of abstraction](#item-26-each-function-should-be-written-in-terms-of-a-single-level-of-abstraction)
    - [Item 27: Use abstraction to protect code against changes](#item-27-use-abstraction-to-protect-code-against-changes)
    - [Item 28: Specify API stability](#item-28-specify-api-stability)
    - [Item 29: Consider wrapping external API](#item-29-consider-wrapping-external-api)
    - [Item 30: Minimize elements visibility](#item-30-minimize-elements-visibility)
    - [Item 31: Define contract with documentation](#item-31-define-contract-with-documentation)
    - [Item 32: Respect abstraction contracts](#item-32-respect-abstraction-contracts)
  - [Chapter 5: Object creation](#chapter-5-object-creation)
    - [Item 33: Consider factory functions instead of constructors](#item-33-consider-factory-functions-instead-of-constructors)
      - [Companion Object Factory Method](#companion-object-factory-method)
      - [Extention Factory Method](#extention-factory-method)
      - [Top-Level Factory Method](#top-level-factory-method)
      - [Fake Constructor](#fake-constructor)
      - [Factory Class Method](#factory-class-method)
    - [Item 34: Consider a primary constructor with named optional arguments](#item-34-consider-a-primary-constructor-with-named-optional-arguments)
      - [Telescoping Constructor Pattern](#telescoping-constructor-pattern)
      - [Builder pattern](#builder-pattern)
    - [Item 35: Consider defining a DSL for complex object creation](#item-35-consider-defining-a-dsl-for-complex-object-creation)
  - [Chapter 6: Class design](#chapter-6-class-design)
    - [Item 36: Prefer composition over inheritance](#item-36-prefer-composition-over-inheritance)
    - [Item 37: Use the data modifier to represent a bundle of data](#item-37-use-the-data-modifier-to-represent-a-bundle-of-data)
    - [Item 38: Use function types or functional interfaces to pass operations and actions](#item-38-use-function-types-or-functional-interfaces-to-pass-operations-and-actions)
    - [Item 39: Use sealed classes and interfaces to express restricted hierarchies](#item-39-use-sealed-classes-and-interfaces-to-express-restricted-hierarchies)
    - [Item 40: Prefer class hierarchies to tagged classes](#item-40-prefer-class-hierarchies-to-tagged-classes)
    - [Item 41: Use enum to represent a list of values](#item-41-use-enum-to-represent-a-list-of-values)
    - [Item 42: Respect the contract of equals](#item-42-respect-the-contract-of-equals)
    - [Item 43: Respect the contract of hashCode](#item-43-respect-the-contract-of-hashcode)
    - [Item 44: Respect the contract of compareTo](#item-44-respect-the-contract-of-compareto)
    - [Item 45: Consider extracting non-essential parts of your API into extensions](#item-45-consider-extracting-non-essential-parts-of-your-api-into-extensions)
    - [Item 46: Avoid member extensions](#item-46-avoid-member-extensions)
- [Part 3: Efficiency](#part-3-efficiency)
  - [Chapter 7: Make it cheap](#chapter-7-make-it-cheap)
    - [Item 47: Avoid unnecessary object creation](#item-47-avoid-unnecessary-object-creation)
    - [Item 48: Use inline modifier for functions with parameters of functional types](#item-48-use-inline-modifier-for-functions-with-parameters-of-functional-types)
    - [Item 49: Consider using inline value classes](#item-49-consider-using-inline-value-classes)
    - [Item 50: Eliminate obsolete object references](#item-50-eliminate-obsolete-object-references)
  - [Chapter 8: Efficient collection processing](#chapter-8-efficient-collection-processing)
    - [Item 51: Prefer Sequence for big collections with more than one processing step](#item-51-prefer-sequence-for-big-collections-with-more-than-one-processing-step)
    - [Item 52: Consider associating elements to a map](#item-52-consider-associating-elements-to-a-map)
    - [Item 53: Consider using groupingBy instead of groupBy](#item-53-consider-using-groupingby-instead-of-groupby)
    - [Item 54: Limit the number of operations](#item-54-limit-the-number-of-operations)
    - [Item 55: Consider Arrays with primitives for performance-critical processing](#item-55-consider-arrays-with-primitives-for-performance-critical-processing)
    - [Item 56: Consider using mutable collections](#item-56-consider-using-mutable-collections)

----

# Abstract

This is about how to use kotlin effectively.

[exercise of effective kotlin @ github](https://github.com/iamslash/kotlin-ex/tree/main/ex-effective)

# Matrials

* [Effective Kotlin](https://kt.academy/book/effectivekotlin)
  * [Effective Kotlin @ amazon](https://www.amazon.com/Effective-Kotlin-practices-Marcin-Moskala/dp/8395452837)
  * [Moskala Marcin](https://marcinmoskala.com/#page-top)

# Part 1: Good code
## Chapter 1: Safety
### Item 1: Limit mutability
### Item 2: Minimize the scope of variables
### Item 3: Eliminate platform types as soon as possible
### Item 4: Do not expose inferred types

```kotlin
// As is
// 다음과 같이 interface 의 return type 을 생략하지 말아라.
// CarFactory 를 구현한 Class 들은 Fiat126P 밖에 생산할 수 밖에 없다.
// CarFactory interface 가 우리가 수정할 수 없는 library 에
// 있다면 문제를 해결할 수 없다.  
interface CarFactory {
  func produce() = Fiat126P()
}

// To Be
interface CarFactory {
  func product(): Car = Fiat126P()
}
```

### Item 5: Specify your expectations on arguments and state

```kotlin
// require, check, assert, elivis operator 를 이용하면 
// 제한사항을 깔끔하게 구현할 수 있다.

//////////////////////////////////////////////////
// require, check , assert
// require 는 조건을 만족하지 못할 때 IllegalArgumentException 을 던진다.
// check 는 조건을 만족하지 못할 때 IllegalStatusException 을 던진다.
// assert 는 조건을 만족하지 못할 때 AssertionError 를 던진다.
// assert 는 실행시 "-ea JVM" 을 사용해야 작동한다. 주로 test 실행시 활성화 된다. 
fun pop(num: Int = 1): List<T> {
  require(num <= size) {
    "Cannot remove more elements than current size"
  }
  check(isOpen) {
    "Cannot pop from closed stack"
  }
  var ret = collection.take(num)
  collection = collection.drop(num)
  assert(ret.size == num)
  return ret
}

// require, check 를 사용해서 조건이 참이라면
// 이후 smart cast 를 수행할 수 있다???
fun changeDress(person: Person) {
  require(person.outfit is Dress)
  val dress: Dress = person.outfit
}

//////////////////////////////////////////////////
// elvis operator
// person.email 이 null 이면 바로 return 이 가능하다.
fun sendEmail(person: Person, text: String) {
  val email: String = person.email ?: return
}
// run scope function 을 사용하면 logic 을 추가하고
// 바로 return 이 가능하다.
fun sendEmail(person: Person, text: String) {
  val email: String = person.email ?: run {
    log("Email not sent, no email address")
    return
  }
}
```

### Item 6: Prefer standard errors to custom ones

```kotlin
// Exception 은 만들지 말고 있는 거 써라.
// 원하는 Exception 이 없을 때만 만들어 써라.
// 다음은 JsonParsingException 를 만들어 쓴 예이다.
inline fun <reified T> String.readObject(): T {
  //...
  if (incorrectSign) {
    throw JsonParsingException()
  }
  //...
  return result
}

// Standard Exceptions
// IllegalArgumentException
// IllegalStatusException
// IndexOutOfBoundsException
// ConcurrentModificationException
// UnsupportedOperationException
// NoSuchElementException
```

### Item 7: Prefer null or Failure result when the lack of result is possible

```kotlin
// 함수가 오류를 발생했을 때 Exception 을 던지는 것보다는
// null, Failure result 를 리턴하는 편이 좋다.
// Exception 은 효율적이지 못하고 함수를 이용하는 입장에서
// 처리를 하지 않을 수도 있다. kotlin 의 Exception 은 
// unchecked Exception 이다. 즉, 그 함수를 호출하는 쪽에서 처리하지 않아도 
// compile 된다.
//
// 오류를 예측할 수 있다면 null, Failure result 를
// 리턴하는 것이 더욱 명시적이다. 예측할 수 없는 오류라면
// Exception 을 던지자.
//
// 일반적인 경우 null 을 리턴한다. 추가 정보를 포함하고 싶다면
// Failure Result 를 리턴한다.
inline fun <reified T> String.readObjectOrNull(): T? {
  //...
  if (incorrectSign) {
    return null
  }
  //...
  return result
}
inline fun <reified T> String.readObjectOrNull(): Result<T> {
  //...
  if (incorrectSign) {
    return Failure(JsonParsingException())
  }
  //...
  return Success(result)
}
sealed class Result<out T>
class Success<out T>(val result: T) Result<T>()
class Failure<val throwable: Throwable>: Result<Nothing>()
class JsonParsingException: Exception()

// null 을 사용한다면 Elvis operator 로 오류처리를 간결하게 구현할 수 있다. 
val age = userText.readObjectOrNull<Person>()?.age ?: -1

// Failure result 를 사용한다면 when 으로 오류처리를 간결하게 구현할 수 있다.
val person = userText.readObjectOrNull<Person>()
val age = when(person) {
  is Success -> person.age
  is Failure -> -1
}

// 무언가 null 을 리턴할 수 있는 함수는 get 보다는 getOrNull 이라고
// 이름짓는 것이 더욱 명시적이다. 
```

### Item 8: Handle nulls properly

```kotlin
// nullable type 을 제대로 다루는 방법은 다음과 같이 있다.
// * ?., smart cast, ?:, !!
// * throw Exception
// * lateinit, notNull delegate

// As Is
// nullable type 은 그대로 사용할 수 없다.
val printer: Printer? = getPrinter()
printer.print() // compile error
// To Be
// ?., smart cast, ?:, !! 를 사용해야 한다.
printer?.print()  // ?.
if (printer != null) {
  printer.print() // smart cast
}
printer!!.print() // Not-null assertion

// throw, !!, requireNotNull, checkNotNull 등을 활용하여
// Exception 을 던지자.
fun process(user: User) {
  requireNotNull(user.name)
  val context = checkNotNull(context)
  val networkService = getNetworkService(context) ?:
    throw NoInternetConnection()
  networkService.getData { data, userData ->
    show(data!!, userData!!)
  }
}

// not-null assertion 은 꼭 필요할 때만 쓰자.
// 문제가 발생할 수 있다??? 가독성을 떨어뜨린다.

// As Is
class UserControllerTest {
  private var dao: UserDao? = null
  private var controller: UserController? = null

  @BeforeEach
  fun init() {
    dao = mockk()
    controller = UserController(dao!!)
  }

  @Test
  fun test() {
    controller!!.doSomething()
  }
}
// To Be
// lateinit property 를 사용하면 나중에 초기화 하겠다는 의미이다.
// 초기화 하지 않고 사용하면 Exception 을 던진다. 
// not-null assertion 을 사용할 필요가 없다. 의미가 명확하다.
class UserControllerTest {
  private lateinit var dao: UserDao
  private lateinit var controller: UserController

  @BeforeEach
  fun init() {
    dao = mockk()
    controller = UserController(dao)
  }

  @Test
  fun test() {
    controller.doSomething()
  }
}

// JVM 의 Int, Long, Double, Boolean 과 같은 Primary type 은
// lateinit 을 사용할 수 없다. Delegates.notNull 을 사용한다.
// lateinit 보다는 느리다.
class DoctorActivity: Activity() {
  private var doctorId: Int by Delegates.notNull()
  private var fromNotification: Boolean by Delegates.notNull()
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    doctorId = intent.extras.getInt(DOCTOR_ID_ARG)
    fromNotification = intent.extras.getBoolean(FROM_NOTIFICATION_ARG)
  }
}
// Delegates.notNull() 대신 property delegation 을 사용할 수도 있다???
class DoctorActivity: Activity() {
  private var doctorId: Int by arg(DOCTOR_ID_ARG)
  private var fromNotification: Boolean by arg(FROM_NOTIFICATION_ARG)
}
```

### Item 9: Close resources with use

```kotlin
// Closeable interface 를 구현한 Object 라면 use function 을 사용하여
// 자동으로 close() 를 호출하도록 하자.

// 다음의 리소스들은 AutoCloseable interface 을 상속받는 Closeable interface
// 를 구현한 것들이다.
// * InputStream, OutputStream 
// * java.sql.Connection
// * java.io.Reader(FileReader, BufferedReader, CSSParser)
// * java.new.Socket, java.util.Scanner

// As Is
// 명시적으로 close() 를 호출해야 한다. try, catch 를 사용하여
// 가독성이 떨어진다.
fun countCharactersInFile(path: String): Int {
  val reader = BufferedReader(FileReader(path))
  try {
    return reader.lineSequence().sumBy { it.length }
  } finally {
    reader.close()
  }
}
// To Be
// user 를 사용한다. 자동으로 close() 가 호출된다.
fun countCharactersInFile(path: String): Int {
  val reader = BufferedReader(FileReader(path))
  reader.use {
    return reader.lineSequence().sumBy { it.length }
  }
}
// lambda 를 사용해도 좋다.
fun countCharactersInFile(path: String): Int {
  BufferedReader(FileReader(path)).use { reader ->
    return reader.lineSequence().sumBy { it.length }
  }
}
// useLines() 를 사용하여 file 를 통째로 읽지 않고 한줄씩 읽게 해보자.
fun countCharactersInFile(path: String): Int {
  File(path).useLines { lines ->
    return lines.sumBy { it.length }
  }
}
```

### Item 10: Write unit tests

[Kotlin Test @ TIL](/kotlin/kotlin_test.md)

## Chapter 2: Readability

### Item 11: Design for readability

```kotlin
// 읽기 쉬운 코드를 작성하자. 적절한 관용구 사용은 좋다.
// 무엇이 적절한 관용구인지는 팀의 컨벤션을 따른다.

// 구현 A 가 구현 B 보다 읽기 쉽다.
// 구현 A
if (person != null && person.isAdult) {
  view.showPerson(person)
} else {
  view.showError()
}
// 구현 B
person?.takeIf { it.Adult }
  ?.let(view::showPerson)
  ?: view.showError()

// 다음의 코드는 적절한 관용구를 사용했다고 할 수 있다.
students
  .filter { it.result => 50}
  .joinToString(separator = "\n") {
    "${it.name} ${it.surname}, ${it.result}"
  }
  .let(::print)
var obj = FileIntputStream("/file.gz")
  .let(::BufferedInputStream)
  .let(::ZipInputStream)
  .let(::ObjectInputStream)
  .readObject() as SomeObject
```

### Item 12: Operator meaning should be clearly consistent with its function name

```kotlin
// operator function 을 overload 해서 사용할 때 의미가 명확해야 한다.
// Good:
// 예를 들어 다음과 같은 factorial extention function 을 살펴보자.
fun Int.factorial(): Int = (1..this).product()
fun Iterable<Int>.product(): Int = fold(1) { acc, i -> acc * i }
print(10 * 6.factorial())  // 7200
// Bad:
// 10 * 6! 을 Int object 의 factorial() 을 호출하여 깔끔하게 구현했다.
// 이번에는 not() operator function 을 overload 하여 factorial 을 구현해보자.
operator fun Int.not() = factorial()
print(10 * 6!)  // 7200
// 답은 7200 으로 같지만 ! 은 not 의미가 아니므로 명확하지 않다.

// 다음은 Kotlin 에서 제공하는 oeprator function 의 목록이다.
// +a     a.unaryPlus()
// -a     a.unaryMinus()
// !a     a.not()
// ++a    a.inc()
// --a    a.dec()
// a+b    a.plus(b)
// a-b    a.minus(b)
// a*b    a.times(b)
// a/b    a.div(b)
// a..b   a.rangeTo(b)
// a in b a.contains(b)
// a += b a.plusAssign(b)
// a -= b a.minusAssign(b)
// a *= b a.timesAssign(b)
// a /= b a.divAssign(b)
// a == b a.equals(b)
// a > b  a.compareTo(b) > 0
// a < b  a.compareTo(b) < 0
// a >= b a.compareTo(b) >= 0
// a <= b a.compareTo(b) <= 0

// 함수의 이름은 의미가 명확행야 한다.
// 예를 들어 다음과 같은 코드를 살펴보자.
operator fun Int.times(operations: () -> Unit): ()->Unit = 
  { repeat(this) { operattion() } }
val tripledHello = 3 * { print("Hello") }
tripledHello()  // Output: HelloHelloHello
// times operator function 을 overload 한 것보다는 
// timesRepeated 라는 extention function 을 사용하는 것이
// 의미가 명확하다.
infix fun Int.timesRepeated(operation: ()->Unit) = {
  repeat(this) { operation() }
}
val tripledHello = 3 timesRepeated { print("Hello") }
tripledHello()  // Output: HelloHelloHello
// 혹은 top level function 을 그대로 사용해도 좋다.
repeat(3) { print("Hello") }
// 이 것은 적당한 예는 아닌 것 같음. repeat() 을 감싸는 또 다른 함수를 만들 필요가 있는가?
// 정리하면 함수의 이름을 의미가 명확하게 만들자는 얘기다.
```

### Item 13: Avoid returning or operating on Unit?

```kotlin
// Unit? 대신 Boolean 을 리턴하도록 하자. 그것이 더욱 명확하다.
// Unit? 를 리턴하고 Evlis operator 를 사용하는 것은
// 가독성을 떨어뜨린다.

// AsIs:
// 예를 들어 다음과 같은 코드를 보자.
// ?: 를 사용하기 위해 Unit? 를 리턴했다.
fun verifyKey(key: String): Unit? = { ... }
  verifyKey(key) ?: return
// ToBe:
// 다음과 같이 개선하자. 가독성이 더욱 좋다.
fun keyIsCorrect(key: String): Boolean = { ... }
if (!keyIsCorrect(key)) return

// AsIs:
// 읽기 어렵다.
getData()?.let{ view.showData(it) } ?: view.showError()
// ToBe:
// 읽기 쉽다.
if (person != null && person.isAdult) {
  view.showPerson(person)
} else {
  view.showError()
}
```

### Item 14: Specify the variable type when it is not clear

```kotlin
// type 은 상황에 따라 명시하는 것이 명확하다.
// AsIs:
val data = getSomeData()
// ToBe:
val data: UserData = getSomeData()
// custom type 의 경우는 명시하는 것이 좋을 것 같다.
```

### Item 15: Consider referencing receivers explicitly

```java
// 명확한 구현을 위해 receiver 를 명시하자.

// 예를 들어 다음과 같이 사용하는 quickSort() 를 Extension Function
// 으로 구현해 보자.
listOf(3, 2, 5, 1, 6).quickSort()  // [1, 2, 3, 4, 5]
listOf("C", "D", "A", "B").quickSort()  // [A, B, C, D]
// AsIs:
// 명확하지 못하다.
fun <T : Comparable<T>> List<T>.quickSort(): List<T> {
  if (size < 2) {
    return this
  }
  val pivot = first()
  val (smaller, bigger) = drop(1).partition { it < pivot }
  return smaller.quickSort() + pivot + bigger.quickSort()
}
// ToBe:
// 명확하다. this.first(), this.drop(1)
fun <T : Comparable<T>> List<T>.quickSort(): List<T> {
  if (size < 2) {
    return this
  }
  val pivot = this.first()
  val (smaller, bigger) = this.drop(1).partition { it < pivot }
  return smaller.quickSort() + pivot + bigger.quickSort()
}

// receiver 가 여러개 일 경우 꼭 명시하자.
// 예를 들어 다음처럼 사용하는 Node 를 구현해 보자.
fun main() {
  val node = Node("parent")
  node.makeChild("child")
}
// AsIs:
// "Created parent" 가 출력된다. 오류이다.
class Node(val name: String) {
  fun makeChild(childName: String) =
    create("$name.$childName")
      .apply { print("Created ${name}") }
  fun create(name: String): Node? = Node(name)
}
// ToBe:
// "Created parent.child" 가 출력된다. 정상이다. ${this?.name}
class Node(val name: String) {
  fun makeChild(childName: String) =
    create("$name.$childName")
      .apply { print("Created ${this?.name}") }
  fun create(name: String): Node? = Node(name)
}
// ToBe:
// apply 는 receiver 가 this 이다. Object 의 this 와 
// 구분이 되지 않는다. 그것보다 also, let 을 사용하는 것이 좋다. 
// receiver 가 it 이다. nullable receiver 를 다룰 때 편리하다.
class Node(val name: String) {
  fun makeChild(childName: String) =
    create("$name.$childName")
      .also { print("Created ${it?.name}") }
  fun create(name: String): Node? = Node(name)
}
// ToBe:
// 레이블을 이용한 receiver 를 사용해도 좋다. 레이블을 생략하면
// 가장 가까운 receiver 를 의미한다.
class Node(val name: String) {
  fun makeChild(childName: String) =
    create("$name.$childName")
      .apply { print("Created ${this?.name} in " +
        " ${this@Node.name}") }
  fun create(name: String): Node? = Node(name)
}
```

```java
// DSL 의 경우 다음처럼 label 사용이 가능하다???
table {
  tr {
    td { +"Column 1" }
    td { +"Column 2" }
    this@table.tr {
      td { +"Value 1" }
      td { +"Value 2" }
    }
  }
}
```

### Item 16: Properties should represent state, not behavior

```java
// class 에서 상태는 property 를 이용하자. 동작은 method 를 이용하자.
// 다음은 일반적인 property 의 예이다. custom get(), set(value) 를
// 정의할 수 있다. field 는 backing field 의 reference 이다.
var name: String? = null
  get() = field?.toUpperCase()
  set(value) {
    if (!value.isNullOrBlank()) {
      field = value
    }
  }
// readonly property 는 val 을 이용하여 선언한다. field 가 만들어
// 지지 않는다.
val fullName: String
  get() = "$name $surname"

// property 는 override 할 수 있다.
open class Supercomputer {
  open val theAnswer: Long = 42
}
class AppleComputer: Supercomputer() {
  override val theAnswer: Long = 1_800_275_2273
}
// property 를 위임할 수도 있다???
val db: Database by lazy { connectToDb() }

// extention property 도 만들 수 있다.
val Context.preference: SharedPreferences
  get() = PreferenceManager
    .getDefaultSharedPreferences(this)
val Context.infalter: LayoutInflater
  get() = getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater
val Context.notificationManager: NotificationManager
  get() = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

// class 의 동작은 method 로 구현한다. property 를 사용하자. 말자.
// AsIs:
val Tree<Int>.sum: Int
  get() = when (this) {
    is Leaf -> value
    is None -> left.sum + right.sum
  }
// ToBe:
fun Tree<Int>.sum(): Int = when (this) {
  is Leaf -> value
  is Node -> left.sum() + right.sum()
}
```

### Item 17: Consider naming arguments

```kotlin
// named arguments 를 사용하면 더욱 명확하다.
// AsIs:
val text = (1..10).joinToString("|")
// ToBe:
val text = (1..10).joinToString(separator = "|")

// function arguments 의 경우 주로 마지막으로 두자.
thread {
  //...
}

// 두 개 이상의 function arguments 의 경우 named arguments
// 를 이용하면 더욱 명확하다.
// AsIs:
// 다음은 Java code 이다. 가독성을 높이기 위해 comment 를 사용했다.
observable.getUsers()
  .subscribe((List<User> users) -> { // onNext
    //...
  }, (Throwable throwable) -> {      // onError
    //...
  }, () -> {                         // onCompleted
    //...
  });
// ToBe:
// 다음은 Kotlin code 이다. named arguments 를 사용해서 명확하다.
observable.getUsers()
  .subscribeBy(
    onNext = { users: List<User> -> 
      //... 
    }, 
    onError = { throwable: Throwable ->
      //...
    },
    onCompleted = {
      //...
    })
  )
```

### Item 18: Respect coding conventions

* [Coding conventions @ Kotlin](https://kotlinlang.org/docs/coding-conventions.html)

# Part 2: Code design

## Chapter 3: Reusability

### Item 19: Do not repeat knowledge

같은 code 를 반복해서 작성하지 말자. 추출해서 공통 code 로 만들자.

### Item 20: Do not repeat common algorithms

직접 만들지 말고 [stdlib @ Kotlin](https://kotlinlang.org/api/latest/jvm/stdlib/) 을 사용하자. [stdlib @ Kotlin](https://kotlinlang.org/api/latest/jvm/stdlib/) 은 공부할 만 하다.

```java
// 직접 공통 code 를 만든다면 다음을 유의하자.
// top level function 보다는 extention method 가 좋다.
// AsIs:
TextUtils.isEmpty("Text")
// ToBe:
"Text".isEmpty()
```

### Item 21: Use property delegation to extract common property patterns

```kotlin
// property delegation 을 사용하면 boiler plate code 를 제거할 수 있다.
// property delegation 은 by 와 delegate 를 이용하여 동작을 전달하는 것이다.
// 이때 delegate 는 val 의 경우 getValue, set 의 경우 getValue, setValue 가
// 정의되야 한다.

// lazy property 는 선언된 이후 처음 사용될 때 초기화되는 property 를 말한다.
// Kotlin 은 stdlib 의 lazy function 으로 lazy property 기능을 제공한다.
val value by lazy { createValue() }
// stdlib 의 Delegates.observable() 을 이용하여 observable pattern 을
// 손 쉽게 구현할 수 있다. 
var items: List<Item> by
  Delegates.observable(listOf()) { _, _, _ ->
    notifyDataSetChanged()
  }
var key: String? by
  Delgeates.observable(null) { _, old, new ->
    Log.e("key changed from $old to $new")
  }

// 다음은 property delegation 을 활용한 예이다.
// Android에서 뷰와 리소스 바인딩
private val button: button by bindView(R.id.button)
private val textSize by bindDimension(R.dimen.font_size)
private val doctor: Doctor by argExra(DOCTOR_ARG)
// Koin 에서 종속성 주입
// Koin 은 jinjection library 이다.
private val presenter: MainPresenter by inject()
private val repository: NetworkRepository by inject()
private val vm: MainViewModel by viewModel()
// 데이터 바인딩
private val port by bindconfiguration("port")
private val token: String by preferences.bind(TOKEN_KEY)

// property delegate 를 만들어 보자.
// AsIs:
// get(), set(value) 는 boiler plate 이다.
var token: String? = null
  get() {
    print("token returned value $field")
    return field
  }
  set(value) {
    print("token changed from $field to $value")
    field = value
  }
var attempts: Int = 0
  get() {
    print("attempts returned value $field")
    return field
  }
  set(value) {
    print("attempts changed from $field to $value")
    field = value
  }
// ToBe:
var token: String? by LoggingProperty(null)
var attempts: Int by LoggingProperty(0)
private class LoggingProperty<T>(var value: T) {
  operator fun getValue(
    thisRef: Any?,
    prop: KProperty<*>
  ): T {
    print("${prop.name} returned value $value")
    return value
  }
  operator fun setValue(
    thisRef: Any?,
    prop: KProperty<*>,
    newValue: T
  ) {
    val name = prop.name
    print("$name changed from $value to $newValue")
    value = newValue
  }
}
// token 은 다음과 같이 compile 될 것이다.
@JvmField
private val 'token$delegate' = LoggingProperty<String?>(null)
var token: String?
  get() = 'token$delegate'.gateValue(this, ::token)
  set(value) {
    'token$delegate'.setValue(this, ::token, value)
  }

// getValue, setValue 가 여러개 있는 delegate 를 만들어 보자.
class SwipeRefreshBinderDelegate(val id: Int) {
  private var cache: SwipeRefreshLayout? = null
  operator fun getValue(
    activity: Activity,
    prop: KProperty<*>
  ): SwipeRefreshLayout {
    return cache ?: activity
      .findViewById<SwipeRefreshLayout>(id)
      .also { cache = it }
  }
  operator fun getValue(
    fragment: Fragment,
    prop: KProperty<*>
  ): SwipeRefreshLayout {
    return cache ?: fragment.view
    .findViewById<SwipeRefreshLayout>(id)
    .also { cache = it }
  }
}

// property delegation 을 위해서 delegate 가 필요하다.
// delegate 는 val 의 경우 getValue(), var 의 경우 getValue(), setValue()
// 가 필요하다. stdlib 의 extention method 를 이용해서 property delegation 을
// 구현해 보자.
val map: Map<String, Any> = mapOf(
  "name" to "David", 
  "programmer" to true,
)
val name by map
print(name)  // Output: David
// 다음과 같은 extention method 가 stdlib 에 정의되어 있다. 
inline operator fun <V, V1 : V> Map<in String, V>
  .getValue(
    thisRef: Any?,
    property: KProperty<*>
  ): V1 = getOrImplicitDefault(property.name) as V1

// stdlib 의 다음과 같은 property delegate 는 자주 사용할 만 하다.
// lazy
// Delegates.observable
// Delegates.vetoable
// Delegates.notNull
```

### Item 22: Use generics when implementing common algorithms

```kotlin
// Generics 를 사용하면 type-safe algorithm, generic class
// 를 구현할 수 있다.

// Type arguments 를 사용하는 함수를 Generic function 이라고 한다.
// stdlib 의 filter 는 대표적인 Generic function 이다.
inline fun <T> Iterable<T>.filter(
  predicate: (T) -> Boolean
): List<T> {
  val destination = ArrayList<T>()
  for (element in this) {
    if (predicate(element)) {
      destination.add(element)
    }
  }
  return destination
}

// Type 을 제한할 수 도 있다. Type 제한은 어렵다. 특히 in, out 을 유의하자.
fun <T: Comparable<T>> Iterable<T>.sorted(): List<T> {
  //...
}
fun <T, C: MutableCollection<in T>> Iterable<T>.toCollection(destination: C): C {
  //...
}
class ListAdapter<T : ItemAdapter>(/*...*/) { /*...*/ }
// Any 는 nullable 이 아닌 type 을 나타낸다.
inline fun <T, R: Any> Iterable<T>.mapNotNull(
  transform: (T) -> R?
): List<R> {
  return mapNotNullTo(ArrayList<R>(), transform)
}
// 하나의 type 에 2개 이상의 제한을 걸 수도 있다.
fun <T: Animal> pet(animal: T) where T: GoodTempered {
  //...
}
fun <T> pet(animal: T) where T: Animal, T: GoodTempered {
  //...
}
```

### Item 23: Avoid shadowing type parameters

```kotlin
// type parameter shadowing 은 피하자.

// addTree 의 parameter 인 name 이 Forest class 의 property
// 인 name 을 가린다. 즉, shadowing 한다. 명확하지 못하다.
class Forest(val name: String) {
  fun addTree(name: String) {
    //...
  }
}

// AsIs:
// Forest 의 T 가 addTree 의 T 와 이름이 같다. 명확하지 못하다.
interface Tree
class Birch: Tree
class Spruce: Tree
class Forest<T: Tree> {
  fun <T: Tree> addTree(tree: T) {
    //...
  }
} 
val forest = Forest<Birch>()
forest.addTree(Birch())
forest.addTree(Spruce())
// AsIs:
// addTree 는 Forest 의 type parameter T 를 사용한다.
// ERROR 가 발생한다.
class Forest<T: Tree> {
  fun <T: Tree> addTree(tree: T) {
    //...
  }
} 
val forest = Forest<Birch>()
forest.addTree(Birch())
forest.addTree(Spruce())  // ERROR, type mismatch
// ToBe:
// 차라리 addTree 는 다른 type parameter 를 사용하자. 명확하다.
class Forest<T: Tree> {
  fun <ST: T> addTree(tree: ST) {
    //...
  }
}
```

### Item 24: Consider variance for generic types

매우 어렵다. [generics, int, out @ TIL](README.md##generics-inout) 를 이해하고 다시 정리하자.

### Item 25: Reuse between different platforms by extracting common modules

Kotlin 은 다양한 platform 을 지원한다. 공통 코드를 잘 만들어 두고 `Kotlin/JVM, Kotlin/Natitive, Kotlin/JS` 등에서 사용하자.

## Chapter 4: Abstraction design

### Item 26: Each function should be written in terms of a single level of abstraction

[Layered Architecture](/ddd/README.md#layered-architecture) 를 기반으로 설계하는 것이 유지보수가 용이하기 때문에 좋다. code 를 작성할 때 layer level 을 신경써서 작성하자. 이 것을 추상화 level 이라고도 한다.

```kotlin
// AsIs:
class CoffeeMachine {
  fun makeCoffee() {
    // 수백 개의 변수를 선언한다.
    // 복잡한 로직을 처리한다.
    // 낮은 수준의 최적화도 여기서 한다.
  }
}
// ToBe:
// Layer level 에 맞게 추상화 되었다.
class CoffeeMachine {
  fun makeCoffee() {
    boilWater()
    brewCoffee()
    pourCoffee()
    pourMilk()
  }
  private fun boilWater() { /*...*/ }
  private fun brewCoffee() { /*...*/ }
  private fun pourCoffee() { /*...*/ }
  private fun pourMil() { /*...*/ }
}
// 만약 ExpressoCoffee 를 만들고 싶다면 
// 다음과 같이 손쉽게 구현할 수 있다.
fun makeEspressoCoffee() {
  boilWater()
  brewCoffee()
  pourCoffee()
}
```

### Item 27: Use abstraction to protect code against changes

컴퓨터 프로그래밍에서의
[추상화](https://developer.mozilla.org/ko/docs/Glossary/Abstraction)란 복잡한
소프트웨어 시스템을 효율적으로 설계하고 구현할 수 있는 방법입니다.
[추상화](https://developer.mozilla.org/ko/docs/Glossary/Abstraction)는 뒷편
시스템의 기술적 복잡함을 단순한 API 뒤에 숨깁니다. 

지나친 추상화는 좋지 않다. [FizzBuzzEnterpriseEdition @ github](https://github.com/EnterpriseQualityCoding/FizzBuzzEnterpriseEdition) 는 지나친 추상화를 한 예이다. 10 줄 이면 구현할 내용을 61 개의 클래스와 26 개의 인터페이스로 구현했다.

```kotlin
// 상수, 함수, 클래스, 인터페이스, ID 추출을 통해 추상화를 할 수 있다.

// 상수 추상화
// AsIS:
fun isPasswordValid(text: String): Boolean {
  if (text.length < 7) return false
  //...
}
// ToBe:
const val MIN_PASSWORD_LENGTH = 7
fun isPasswordValid(text: String): Boolean {
  if (text.length < MIN_PASSWORD_LENGTH) return false
}

// 함수 추상화
Toast.makeText(this, message, Toast.LENGTH_LONG).show()
// AsIs:
// 다음과 같이 extention method 로 구현할 수 있다.
fun Context.toast(
  message: String,
  duration: Int = Toast.LENGTH_LONG
) {
  Toast.makeText(this, message, duration).show()
}
// 일반적으로 사용
context.toast(message)
// Activity, Context 의 sub class 에서 사용
toast(message)
// snack bar 도 필요하게 되어 다음과 같이 구현했다.
fun Context.snackbar(
  message: String,
  duration: Int = Toast.LENGTH_LONG
) {
  //...
}
// ToBe:
// toast, snackbar 를 추출하여 showMessage 로 구현했다.
fun Context.showMessage(
  message: String,
  duration: MessageLength = MessageLength.LONG
) {
  val toastDuration = when(duration) {
    SHORT -> Length.LENGTH_SHORT
    LONG -> Length.LENGTH_LONG
  }
  Toast.makeText(this, message, toastDuration).show()
}
enum class MessageLength { SHORT, LONG }

// 클래스 추상화
// ToBe:
// 이전의 메시지 출력을 클래스로 추상화 해보자. 클래스로 만들면
// 많은 함수를 가질 수 있고 상태도 가질 수 있다.
class MessageDisplay(val context: Context) {
  fun show(
    message: String,
    duration: MessageLength = MessageLength.LONG
  ) {
    val toastDuration = when(duration) {
      SHORT -> Length.LENGTH_SHORT
      LONG -> Length.LENGTH_LONG
    }
    Toast.makeText(this, message, toastDuration).show()
  }
}
enum class MessageLength { SHORT, LONG }
// 사용
val messageDisplay = messageDisplay(context)
messageDisplay.show("Message")

// 인터페이스 추상화
// ToBe:
interface MessageDisplay(
  fun show(
    message: String,
    duration: MessageLength = LONG
  )
)
class ToastMessageDisplay(val context: Context): MessageDisplay {
  override fun show(
    message: String,
    duration: MessageLength
  ) {
    val toastDuration = when(duration) {
      SHORT -> Length.LENGTH_SHORT
      LONG -> Length.LENGTH_LONG
    }
    Toast.makeText(this, message, toastDuration).show()
  }
}
enum class MessageLength { SHORT, LONG }
// 사용
val messageDisplay: MessageDisplay = ToastMessageDisplay()

// ID 만들기 추상화
// AsIs:
// thread-safe 하지 못하다.
var nextId: Int = 0
val newId = nextId++
// ToBe:
// 함수로 추상화한다. 나중에 함수를 thread-safe 하게 구현할 수 있다.
private var nextId: Int = 0
fun getNextId(): Int = nextId++
val newId = getNextId()
// ToBe:
// data class 로 추상화한다. 나중에 id 의 type 을 바꿀 수 있다.
data class Id(private val id: Int)
private var nextId: Int = 0
fun getNextId(): Id = Id(nextId++)
```

### Item 28: Specify API stability

annotation 을 이용해서 public API 들을 유지보수 하자.

```java
@Experimental(level = Experimental.level.WARNING)
annotation class ExperimentalNewApi

@ExperimentalNewApi
suspend fun getUsers(): List<User> {
  //...
}

@Deprecated("Use suspending getUsers instead")
fun getUsers(callback: (List<User>)->Unit) {
  //...
}

@Deprecated("Use suspending getUsers instead",
  ReplaceWith("getUsers()"))
fun getUsers(callback: (List<User>)->Unit) {
  //...
}
```

### Item 29: Consider wrapping external API

API 도 믿을 수 없으니 layer 하나 만들어 API 를 wrapping 하자.

### Item 30: Minimize elements visibility

[Visibility modifiers](https://kotlinlang.org/docs/visibility-modifiers.html) 의 종류는 다음과 같다. 

* private
* protected
* internal
* public (default)

[Visibility modifiers](https://kotlinlang.org/docs/visibility-modifiers.html) 의 수준은 항상 최소로 하자.

Packages, Classes 별로 의미가 다르다.

| visibility | Packages | Classes |
|--|--|--|
| public | any package | any class |
| private | same file | same class |
| internal | same module | same module |
| proteted | not available | same class, sub class |

다음은 Packages 의 Visibility modifiers 의 예이다.

```kotlin
// file name: example.kt
package foo
private fun foo() { /*...*/ } // visible inside example.kt
public var bar: Int = 5 // property is visible everywhere
    private set         // setter is visible only in example.kt
internal val baz = 6    // visible inside the same module
```

다음은 Classes 의 Visibility modifieres 의 예이다.

```kotlin
open class Outer {
    private val a = 1
    protected open val b = 2
    internal open val c = 3
    val d = 4  // public by default

    protected class Nested {
        public val e: Int = 5
    }
}

class Subclass : Outer() {
    // a is not visible
    // b, c and d are visible
    // Nested and e are visible
    override val b = 5   // 'b' is protected
    override val c = 7   // 'c' is internal
}

class Unrelated(o: Outer) {
    // o.a, o.b are not visible
    // o.c and o.d are visible (same module)
    // Outer.Nested is not visible, and Nested::e is not visible either
}
```

### Item 31: Define contract with documentation

* [Document Kotlin code: KDoc and Dokka](https://kotlinlang.org/docs/kotlin-doc.html)

---

Kotlin 의 document 를 표현한 language 를 KDoc 이라고 한다. [dokka](https://github.com/Kotlin/dokka) 는 Kotlin code 에서 Kotlin documentation 을 뽑아낸다. 

기본적인 문법은 다음과 같다. 

```kotlin
/**
 * A group of *members*.
 *
 * This class has no useful logic; it's just a documentation example.
 *
 * @param T the type of a member in this group.
 * @property name the name of this group.
 * @constructor Creates an empty group.
 */
class Group<T>(val name: String) {
    /**
     * Adds a [member] to this group.
     * @return the new size of the group.
     */
    fun add(member: T): Int { ... }
}
```

다음은 주요 block tags 이다.

| tag | description |
|--|--|
| `@param` | |
| `@return` | |
| `@constructor` | |
| `@receiver` | |
| `@property name` | |
| `@throws class, @exception class` | |
| `@sample identifier` | 사용예  |
| `@see` | see also |
| `@author` | |
| `@since` | |
| `@suppress` | 특정 버전의 문서에 포함시키지 않는다. |

KDoc 은 `@deprecated` 를 지원하지 않는다. `@Deprecated` 를 사용해야 함.

### Item 32: Respect abstraction contracts

class 가 제공하는 의도를 잘 파악해서 사용하자. 예를 들어 `equals` 를
의도와 다르게 구현했다면 문제가 발생한다.

```kotlin
class Id(val id: Int) {
  override fun equals(other: Any?) =
    other is Id && other.id == id
}
val set = mutableSetOf(Id(1))
set.add(Id(1))
set.add(Id(1))
println(set.size) // Output: 3
```

## Chapter 5: Object creation
### Item 33: Consider factory functions instead of constructors

```kotlin
// 생성자를 이용하여 객체를 생성하지 말고 팩토리 함수를 이용하여 객체를 생성하자.
// AsIs: 생성자를 이용하여 객체를 생성한다.
class ListNode<T> {
  val head: T,
  val tail: ListNode<T>?
}
val list = ListNode(1, ListNode(2, null))
// ToBe: 책토리 함수를 이용하여 객체를 생성한다.
fun <T> ListNodeOf(
  vararg elements: T
): ListNode<T>? {
  if (elements.isEmpty()) {
    return null
  }
  val head = elements.first()
  val elementsTail = elements.copyOfRange(1, elements.size)
  val tail = ListNodeOf(*elementsTail)
  return ListNode(head, tail)
}
val list = ListNodeOf(1, 2)
```

다음은 [factory method](/designpattern/factorymethod/factorymethod.md) 를 이용할 때 얻을 수 있는 장점이다.

* 함수에 이름을 붙일 수 있다. 이름을 보고 생성절차에 대해 알 수 있다.
* 함수가 원하는 타입을 리턴할 수 있다. 다른 객체를 생성할 때 좋다.
* 호출될 때 마다 새 객체를 만들 필요가 없다. 예를 들어 싱글톤 객체를 리턴할 수도
  있고 null 을 리턴할 수도 있다.
* 아직 존재하지 않는 객체를 리턴할 수도 있다.
* 가시성을 조절할 수 있다. factory method 를 top level function 으로 만들어 같은
  파일 혹은 같은 모듈에서만 접근하게 할 수도 있다.
* inline function 으로 만들 수 있다. type parameter 를 reified 으로 만들 수도
  있다.
* 생성자는 부모 클래스 또는 기본 생성자를 호출해야 한다. 팩토리 함수는 원하는
  때에 생성자를 호출 할 수 있다.
  ```kotlin
  fun makeListView(config: Config): ListView {
    val items = // Read something from config
    return ListView(items) // Call real constructor
  }
  ```
* 한 가지 단점이 있다. 서브클래스 생성에는 슈퍼클래스의 생성자가 필요가 하기
  때문에 서브클래스를 만들어낼 수 없다.
  ```kotlin
  class IntListNode: ListNode<T>() {
    // ListNode 가 open 이라면
    constructor(vararg ints: Int): ListNodeOf(*ints)
    // ERROR
  }
  ```
* 그러나 팩토리 함수로 슈퍼클래스를 만들기로 했다면 그 서브클래스에도 팩토리 함수를 만들면 된다.
  ```kotlin
  class IntListNode(head: Int, tail: IntListNode?): ListNode<Int>(head, tail)

  fun IntListNodeOf(vararg elements: Int): IntListNode? {
    if (elements.isEmpty()) {
      return null
    }
    val head = elements.first()
    val elementsTail = elements.copyOfRange(1, lements.size)
    val tail = IntListNodeOf(*elementsTail)
    return IntListNode(head, tail)
  }
  ```

다음은 Kotlin 에서 구현할 수 있는 [factory method](/designpattern/factorymethod/factorymethod.md) 의 종류이다.

* companion 객체 팩토리 함수
* 확장 팩토리 함수
* 톱레벨 팩토리 함수
* 가짜 생성자
* 팩토리 클래스의 메서드

#### Companion Object Factory Method

```kotlin
// Java 의 static factory function 과 같다.
class MyLinkedListNode<T> (
  val head: T,
  val tail: MyLinkedListNode<T>?
) {
  companion object {
    fun <T> of(vararg elements: T): MyLinkedListNode<T>? {

    }
  }
}
val list = MyLinkedListNode.of(1, 2)

// interface 에도 구현할 수 있다.
class MyLinkedListNode<T> {
  val head: T,
  val tail: MyLinkedListNode<T>?
}: MyListNode<T> {
  //...
}
interface MyListNode<T> {
  //...
  companion object {
    fun <T> of(vararg elements: T): MyListNode<T>? {
      //...
    }
  }
}
val list = MyListNode.of(1, 2)
```

```kotlin
// 주로 다음과 같은 이름들을 factory method 의 이름으로 사용합니다.

// from: 파라미터를 하나 받고 같은 타입의 인스턴스 하나를 리턴
val date: Date = Date.from(instant)

// of: 파라미터를 여러개 받고 이것을 합친 인스턴스를 생성
val faceCards: Set<Rank> = EnumSet.of(JACK, QUEEN, KING)

// valueOf: from 또는 of 와 비슷하다.
val prime: BigInteger = BigInteger.valueOf(Integer.MAX_VALUE)

// instance or getInstance: singleton instance 를 리턴한다.
val luke: StackWalker = StackWalker.getInstance(options)

// createInstance or newInstance: 호출될 때 마다 새로운 instance 를 리턴한다.
val newArray = Array.newInstance(classObject, arrayLen)

// getType: getInstance 와 같다. 팩토리 함수가 다른 클래스에 있을 때 사용한다.
val fs: FileStore = Files.getFileStore(path)

// newType: newInstance 와 같다. 팩토리 함수가 다른 클래스에 있을 때 사용한다.
val br: bufferedReader = Files.newBufferedReader(path)
```

companion object 는 interface 를 구현할 수 있고 또한 클래스를 상속받을 수도
있다.

```kotlin
abstract class AcivityFactory {
  abstract fun getIntent(context: Context): Intent

  fun start(context: Context) {
    val intent = getIntent(context)
    context.startActivity(intent)
  }

  fun startForResult(activity: Activity, requestCode: Int) {
    val intent = getIntent(activity)
    activity.startActivityForResult(intent, requestCode)
  }
}

class MainActivity: AppCompatActivity() {
  //...
  companion object: ActivityFactory() {
    override fun getIntent(context: Context): Intent =
      Intent(context, MainActivity::class.java)
  }
}

val intent = MainActivity.getIntent(context)
MainActivity.start(context)
MainActivity.startForResult(activity, requestCode)
```

#### Extention Factory Method

이미 companion object 가 존재할 때 이 객체의 함수처럼 사용할 팩토리 함수를 만들어 보자.

```kotlin
// Tool 의 코드를 수정할 수 없는 상황이다.
interface Tool {
  companion object { /*...*/ }
}
// companion object 의 extention method 를 이용하여 factory method 를 만들자.
fun Tool.Companion.createBigTool(): BigTool {
  //...
}
ToolcreateBigTool()
```

#### Top-Level Factory Method

```kotlin
// Intent object 를 companion factory method 를 이용하여 생성해보자.
class MainActivity: Activity {
  companion object {
    fun getIntent(context: Context) =
      Intent(context, MainActivity::class.java)
  }
}
// top level factory method 를 만들어 사용하자. 이름이 더욱 쉽다.
intentFor<MainActivity>()
intentFor<MainActivity>("page" to 2, "row" to 10)
// listOf(1, 2, 3) 이 List.of(1, 2, 3) 보다 가독성이 높다.s
```

#### Fake Constructor

fake constructor 를 제작하는 방법은 다음과 같이 2 가지가 있다.

* top level function
* companion object with invoke function

일반 function 이지만 역할은 constructor 와 같은 것을 facke factory method
라고 한다. 생성자는 아니지만 생성자처럼 동작한다. 다음과 같은 경우 사용한다.

* 인터페이스를 위한 생성자를 만들고 싶을 때
* retified type argument 를 갖게 하고 싶을 때

다음은 top level function 으로 fake constructor 를 구현한 것이다.

```kotlin
// AsIs: normal constructor
class A
val a = A()
val reference: ()->A = ::A
// ToBe: face constructor
List(4) { "User$it" } // [User0, User1, User2, User3]
// List, MutableList are fake constructors defined in stdlib since kotlin 1.1
public inline fun <T> List(
  size: Int,
  init: (index: int) -> T
): List<T> = MutableList(size, init)

public inline fun <T> MutableList(
  size: Int,
  init: (index: int) -> T
): MutableList<T> {
  val list = ArrayList<T>(size)
  repeat(size) { index -> list.add(init(index)) }
  return list
}
```

다음은 companion object 에 invoke 를 정의하여 fake constructor 를
사용하는 방법이다.

```kotlin
class Tree<T> {
  companion object {
    operator fun <T> invoke(size: Int, generator: (Int) -> T): Tree<T> {
      //...
    }
  }
}
Tree(10) { "$it" }
```

companion object with invoke function 은 복잡하다. 가급적 top level function 을
이용하여 fake constructor 를 정의하자.

```kotlin
// constructor
val f: () -> Tree = ::Tree
// top level function fake constructor
val f: () -> Tree = ::Tree
// companion object fake constructor
val f: () -> Tree = Tree.Companion::invoke
```

#### Factory Class Method

Factory Class 는 Factory Method 와 달리 상태를 가질 수 있다. 

```kotlin
data class Student(
  val id: Int,
  val name: String,
  val surname: String
)

class StudentsFactory {
  var nextId = 0
  fun next(name: String, surname: String) =
    Student(nextId++, name, surname)
}

val factory = StudentsFactory()
val s1 = factory.next("david", "sun")
println(s1) // Student(id=0, name=david, surname=sun)
val s2 = factory.next("hello", "world")
println(s2) // Student(id=1, name=hello, surname=world)
```

### Item 34: Consider a primary constructor with named optional arguments

Kotlin 에서 객체를 생성할 때 주로 primary constructor 를 이용한다.

```kotlin
// primary constructor of normal class
class User(var name: String, var surname: String)
val user = User("David", "Sun")

// primary constructor of data class
data class Studen(
  val name: String,
  val surname: String,
  val age: Int
)
```

primary constructor 에 property 를 추가할 수도 있다. 이 방법은
복잡하다.

```kotlin
class QuotationPresenter(
  private val view: QuotationView,
  private val repo: QuotationRepository
) {
  private var nextQuoteId = -1
  fun onStart() {
    onNext()
  }
  fun onNext() {
    nextQuoteId = (nextQuoteId + 1) % repo.quotesNumber
    val quote = repo.getQuote(nextQuoteId)
    view.showQuote(quote)
  }
}
```

constructor 에 option arguments 전달하는 방법은 다음과 같이 2 가지가 있다.

* 점층적 생성자 패턴 (telescoping constructor pattern)
* 빌더 패턴 (builder pattern)

#### Telescoping Constructor Pattern

telescoping constructor pattern 은 다양한 arguments 를 갖는
여러 생성자를 정의하는 방식이다.

```kotlin
// AsIs: telescoping constructor pattern 을 이용했다. 그러나 보기 좋지 않다.
class Pizza {
  val size: String
  val cheese: Int
  val olives: Int
  val bacon: Int

  constructor(size: String, cheese: Int, olives: Int, bacon: Int) {
    this.size = size
    this.cheese = cheese
    this.olives = olives
    this.bacon = bacon
  }

  constructor(size: String, cheese: Int, olives: Int): this(size, cheese, olives, 0)
  constructor(size: String, cheese: Int): this(size, cheese, 0)
  constructor(size: String): this(size, 0)
}
// ToBe: default arguments 를 이용하는 편이 더욱 명확하다.
// baned arguments 와 함께 더욱 깔끔해 졌다.
class Pize(
  val size: String,
  val cheese: Int = 0,
  val olives: Int = 0,
  val bacon: Int = 0
)
val myFavorite = Pizza("L", olives = 3)
val myFavorite = Pizza("L", olives = 3, cheese = 1)
```

#### Builder pattern

> [builder pattern](/designpattern/builder/builder.md)

Java 의 경우 builder pattern 이 유용하다. 그러나 kotlin 의 경우는 builder
pattern 보다 primary constructor with default arguments 를 이용하는 편이 더욱
좋다.

* 더 짧다.
* 더 명확하다.
* 더 쉽다.
* 동시성 문제가 없다. builder pattern 의 function 은 thread-safe 하게 구현해야
  한다. 대부분 mutable property 를 사용하기 때문이다.

그러나 Kotlin 에서도 primary constructor 보다 builder pattern 이 유용한 경우가
있다.

```kotlin
// AsIs: primary constructor 를 이용했더니 복잡하다.
val dialog = Alertdialog(context,
  message = R.string.fire_missiles,
  positiveButtonDescription = ButtonDescription(R.string.fire, { d, id ->
    // launch missiles
  }),
  negativeButtonDescription = ButtonDescription(R.string.cancel, { d, id ->
    // cancel missiles
  }))
val router = Router(
  routes = listOf(
    Route("/hone", ::showHome),
    Route("/users", ::showUsers)
  )
)
// ToBe: builder pattern 을 이용하는 것이 더욱 깔끔하다.
val dialog = AlertDialog.Builder(context)
  .showMessage(R.string.fire_missiles)
  .setPositiveButton(R.string.fire, { d, id ->
    // launch missiles
  })
  .setNegativeButton(R.string.cancel, { d, id ->
    // cancel missiles
  })
  .create()
val router = Router.Builder()
  .addRoute(path = "/home", ::showHome)
  .addRoute(path = "/users", ::showUsers)
  .create()
// ToBe: DSL 을 이용하는 것이 더욱 깔끔하다.
val dialog = context.alert(R.string.fire_missiles) {
  positiveButton(R.string.fire) {
    // launch missiles
  }
  negativeButton {
    // cancel missiles
  }
}
val route = {
  "/home" directsTo ::showHome
  "/users" directsTo ::showUsers
}
```

Kotlin 에서 builder pattern, DSL pattern 을 이용하여 객체를 생성하는 방법은
피하자. primary constructor 를 이용하자.

### Item 35: Consider defining a DSL for complex object creation

* [Building DSLs in Kotlin @ baeldung](https://www.baeldung.com/kotlin/dsl)

-----

복잡하다. 과연 사용할 일이 많을까? 그때 정리해야 겠다.

## Chapter 6: Class design

### Item 36: Prefer composition over inheritance

composition 은 필요한 기능을 제공하는 class 를 member 로 갖는 것이다.
compoisition 이 inheritance 보다 더욱 명확하다.

```kotlin
// AsIs: ProfileLoader, ImageLoader 가 있다.
class ProfileLoader {
  fun load() {
    // Show progress bar
    // Read profile
    // Hide progress bar
  }
}
class ImageLoader {
  fun load() {
    // Show progress bar
    // Read image
    // Hide progress bar
  }
}
// AsIs: ProfileLoader, ImageLoader 는 LoaderWithProgress 를 상속 받도록 하자.
// 다음과 같은 단점들이 있다.
// * 상속은 하나의 클래스만을 대상으로 할 수 있다. 추출을 해야 하므로 BaseXXX 라는 것이
//   생기고 뚱둥해진다.
// * 상속은 클래스의 모든 것을 가져온다. 자식 class 입장에서 부모로 부터 사용하지 않는 method,
//   variable 을 가져온다.
// * 상속은 이해하기 어렵다. 복잡한 hierarchy 는 가독성을 떨어뜨린다.
abstract class LoaderWithProgress {
  fun load() {
    // Show progress bar
    innerLoad()
    // Hide progress bar
  }
}

class ProfileLoader: LoaderWithProgress {
  override fun innerLoad()
}

class ImageLoader: LoaderWithProgress {
  override fun innserLoad()
}

// ToBe: composition 이 더욱 갈끔하다.
class Progress {
  fun showProgress() { /* Show progress */ }
  fun hideProgress() { /* hide progress */ }
}

class ProfielLoader {
  val progress = Progress()
  fun load() {
    progress.showProgress()
    // read profile
    progress.hideProgress()
  }
}

class ImageLoader {
  val progress = Progress()
  fun load() {
    progress.showProgress()
    // read profile
    progress.hideProgress()
  }
}
```

### Item 37: Use the data modifier to represent a bundle of data

```kotlin
// Kotlin 에서 data class 는 매우 유용하다.
// data class 는 다음과 같은 method 를 자동으로 만들어 준다.
// * toString
// * equals
// * hashCode
// * copy
// * componentN(component1, component2, etc...)
data class Player(
  val id: Int,
  val name: String,
  val points: Int
)
val player = Player(0, "Gecko", 9999)

// toString
print(player)  // Player(id=0, name=Gecko, points=9999)
// equals
player == Player(0, "Gecko", 9999)  // true
player == Player(0, "Ross", 9999)   // false
// copy, this is shallow copy not deep copy
val newObj = player.copy(name = "Thor")
print(newObj)  // Player(id=0, name=Thor, points=9999)
// copy should be implemented like this
fun copy(
  id: Int = this.id,
  name: String = this.name,
  points: Int = this.points
) = Player(id, name, points)
// componentN makes destructuring possible 
val (id, name, points) = player
// this is after compiled
val id: Int = player.component1()
val name: String = plyer.component2()
val points: Int = player.component3()
// omponentN is convenient for collection
val visited = listOf("china", "Russia", "India")
val (first, second, third) = visited
println("$first $second $third")
// China Russia India
val trip = mapOf(
  "China" to "Tianjiin",
  "Russia" to "Petersburg",
  "India" to "Rishikesh"
)
for ((country, city) in trip) {
  println("We loved $city in $country")
  // We loved Tianjin in China
  // We loved Petersburg in Russia
  // We loved Rishikesh in Indea
}
```

Tuple 을 사용하지 말고 Data Class 를 사용하라.

```kotlin
// Pair, Triple 은 Kotlin 에 남은 유일한 Tuple 이다.
public data class Pair<out A, out B>(
  public val first: A,
  public val second: B
): Serializable {
  public override fun toString(): String = 
    "($first, $second)"
}
public data class Triple<out A, out B, out C>(
  public val first: A,
  public val second: B,
  public val third: C,
): Serializable {
  public override fun toString(): String = 
    "($first, $second, $third)"
}
// 다음은 Pair, Triple 의 사용예이다.
val (description, color) = when {
  degrees < 5 -> "cold" to Color.BLUE
  degrees < 23 -> "mild" to Color.YELLOW
  else -> "hot" to Color.RED
}
val (odd, even) = numbers.partition { it % 2 == 1 }
val map = mapOf(1 to "San Francisco", 2 to "Seoul")
// 이런 경우들을 제외하고는 무조건 data class 를 사용하는 것이 좋다.
```

Tuple 보다 Data Class 가 더욱 명확한 경우를 살펴보자. 

```kotlin
// AsIs: firstName 이 첫번째 인지 두번째 인지 명확하지 않다.
fun String.parseName(): Pair<String, String>? {
  val indexOfLastSpace = this.trim().lastIndexOf(' ')
  if (indexOfLastSpace < 0) {
    return null
  }
  val firstName = this.take(indexOfLastSpace)
  val lastName = this.take(indexOfLastSpace)
  return Pair(firstName, lastName)
} 
val fullName = "David Sun"
val (firstName, lastName) = fullName.parseName() ?: return
print("His name is $firstName") // His name is David
// ToBe: firstName, lastName 의 위치가 명확하다.
data class FullName(
  val firstName: String,
  val lastName: String
)
fun String.parseName(): FullName? {
  val indexOfLastSpace = this.trim().lastIndexOf(' ')
  if (indexOfLastSpace < 0) {
    return null
  }
  val firstName = this.take(indexOfLastSpace)
  val lastName = this.take(indexOfLastSpace)
  return FullName(firstName, lastName)
} 
val fullName = "David Sun"
val (firstName, lastName) = fullName.parseName() ?: return
```

### Item 38: Use function types or functional interfaces to pass operations and actions

action 을 함수에 전할 때 세가지 방법이 있다.
* SAM (Single Abstract Method) 
  * function 이 하나인 interface
  * functional interface 라고도 한다.
* Function Types
* Method Reference

SAM 대신 Function Types, Method Reference 를 사용하자.

```java
// AsIs: SAM
interface OnClick {
  fun clicked(view: View)
}
fun setOnClickListener(listener: OnLick) {
  //...
}
setOnClickListener(object: OnClick {
  override fun clicked(view: View) {
    //...
  }
})
// ToBe: Function Types, Method Reference 가 더욱 깔끔하다.
fun setOnClickListener(listener: (View) -> Unit) {
  //...
}
// Lambda Function
setOnClickListener { /*...*/ }
// Anonymous Function which represent return type
setOnClickListener(fun(view) { /*...*/ })
// Method Reference
setOnClickListener(::println)
setOnClickListener(this::showUsers)
```

JAVA 에서 사용할 API 를 Kotlin 으로 제공할 때는 SAM 을 사용하자.
Java 에서는 interface 가 더욱 명확하다. IntelliJ 의 지원을 받을 수 있다.
100% 이해 못함.

```kotlin
// Kotlin
class CalendarView() {
  var onDateClicked: ((data: Date) -> Unit)? = null)
  var onPageChanged: OnDateClicked? = null
}
interface OnDateClicked {
  fun onClick(date: Date)
}
// Java
CalendarView c = new CalendarView();
c.setOnDateClicked(date -> Unit.INSTANCE);
c.setOnPageChanged(date -> {});
```

### Item 39: Use sealed classes and interfaces to express restricted hierarchies
### Item 40: Prefer class hierarchies to tagged classes

상수 모드를 tag 라고한다. tag 를 포함한 class 를 tagged class 라고 부른다.
아래의 예에서 ValueMatcher 가 tagged class 이다. matcher 가 상수 모드를 갖는다.

```kotlin
// AsIs: 다음과 같은 단점 들이 있다.
// * 한 클래스에 여러 모드를 처리하기 위해 boiler plate code 가 필요하다.
// * property 의 목적이 일관되지 못하다. value 는 LIST_EMPTY, LIST_NOT_EMPTY 에 필요 없다.
// * 일관성 정확성면에서 부족하다???
// * 팩토리 메서드를 사용해야 한다.
class ValueMatcher<T> private constructor(
  private val value: T? = null,
  private val matcher: Matcher
) {
  fun match(value: T?) = when(matcher) {
    Matcher.EQUAL -> value == this.value
    Matcher.NOT_EQUAL -> value != this.value
    Matcher.LIST_EMPTY -> value is List<*> && value.isEmpty()
    Matcher.LIST_NOT_EMPTY -> value is List<*> && value.isNotEmpty()
  }
  enum class Matcher {
    EUQAL,
    NOT_EQUAL,
    LIST_EMPTY,
    LIST_NO_EMPTY
  }
  companion object {
    fun <T> equal(value: T) =
      ValueMatcher<T>(value=value, matcher=Matcher.EQUAL)
    fun <T> noEqual(value: T) =
      ValueMatcher<T>(value=value, matcher=Matcher.NOT_EQUAL)
    fun <T> emptyList() =
      ValueMatcher<T>(matcher=Matcher.LIST_EMPTY)
    fun <T> notEmptyList() =
      ValueMatcher<T>(matcher=Matcher.LIST_NOT_EMPTY)      
  }
}
// ToBe: 상수 모드를 모두 class 로 정규화 하자. 무엇이 개선된건가???
// sealed 사용했기 때문에 다른 곳에서 mode 를 추가할 수 없다.
sealed class ValueMatcher<T> {
  abstract fun match(value: T): Boolean
  class Equal<T>(val value: T): ValueMatcher<T>() {
    override fun match(value: T): Boolean =
      value == this.value
  }
  class NotEqual<T>(val value: T): ValueMatcher<t>() {
    override fun match(value: T): Boolean =
      value != this.value
  }
  class EmptyList<T>(): ValueMatcher<t>() {
    override fun match(value: T): Boolean =
      value is List<*> && value.isEmpty()
  }
  class NotEmptyList<T>(): ValueMatcher<t>() {
    override fun match(value: T): Boolean =
      value is List<*> && value.isNotEmpty()
  }
}
```

### Item 41: Use enum to represent a list of values

### Item 42: Respect the contract of equals

Kotlin 의 Any 는 다음과 같은 method 를 갖는다.

* `equals`
* `hashCode`
* `toString`

Kotlin 은 두가지 종류의 equality 가 있다.

* 구조적 동등성 `structural equality`
  * `a == b`
* 레퍼런스적 동등성 `referential equality`
  * `a === b`

```kotlin
open class Animal
class Book
Animal() == Book()  // ERROR: can't use == for Animal, Book
Animal() === Book() // ERROR: can't use === for Animal, Book

class Cat: Animal()
Animal() == Cat()   // OK
Animal() === Cat()  // OK

class Name(val name: String)
val name1 = Name("David")
val name2 = Name("David")
val name1Ref = name1
name1 == name1     // true
name1 == name2     // false
name1 == name1Ref  // true
name1 === name1    // true
name1 === name2    // false
name1 === name1Ref // true
```

`equals()` 를 직접 구현해야할 경우는 거의 없다. data class 를 주로 사용하자.
그러나 다음의 경우에 구현할 필요가 있다.

* 기본적으로 제공되는 동작과 다른 동작을 해야 하는 경우
* 일부 프로퍼티만 비교해야하는 경우
* data 한정자를 붙이는 것을 원하지 않거나, 비교해야 하는 프로퍼티가 기본생성자에 없는 경우

```kotlin
// AsIs: DateTime 에 equals() 를 구현했다. data class 를 사용하면 구현할
// 필요가 없다. 
class DateTime(
  private var millis: Long = 0L,
  private var timeZone: TimeZone? = null
) {
  private var asStringCache = ""
  private var changed = false

  override fun equals(other: Any?): Boolean = 
    other is DateTime &&
      other.millis == millis &&
      other.timeZone == timeZone
}
// ToBe: data class 를 사용해 보자. 
// asStringCache, changed 의 비교대상이 아니다.
data class DateTime(
  private var millis: Long = 0L,
  private var timeZone: TimeZone? = null
) {
  private var asStringCache = ""
  private var changed = false
}
```

`equals()` 를 구현한다면 다음의 규칙을 충족해야 한다.

* 반사성 (reflexive)
  * `x != null` 이면 `x.equals(x) == ture` 이어야 한다.
* 대칭성 (symmetric)
  * `x != null && y != null` 이면 `x.equals(y) == y.equals(x)` 이어야 한다.
* 연속성 (transitive)
  * `x != null && y != null && z != null` 이라고 하자. `x.equals(y) ==
    y.equals(z)` 이면 `x.equals(z)` 이다. 
* 일관성 (consistent)
  * `x != null && y != null` 이라고 하자. `x.equals(y)` 는 여러번 실행해도 결과는 같다.
* null 과 관련된 동작
  * `x != null` 이면 `x.equals(null) == false` 이다.

### Item 43: Respect the contract of hashCode

`hashCode()` 를 구현한다면 다음과 같은 규칙을 충족해야 한다.

* 어떤 객체를 변경하지 않았다면 `hashCode()` 를 여러번 호출해도 항상 같은 값이
  나와야 한다.
* `equals()` 로 서로 같은 객체가 있다면 `hashCode()` 역시 같은 값이 나와야 한다.

`hashCode()` 는 `equals()` 처러 직접 구현할 일이 별로 없다. data class 를
사용하자.

다음은 `equals(), hashCode()` 의 구현예이다.

```kotlin
class DateTime(
  private var millis: Long = 0L,
  private var timeZone: TimeZone? = null
) {
  private var asStringCache = ""
  private var changed = false

  override fun equals(other: Any?): Boolean =
    other is DateTime &&
      other.millis == millis &&
      other.timeZone == timeZone

  override fun hashCode(): Int {
    var result = millis.hashCode()
    // 관례적으로 31 을 사용한다. prime number???
    result = result * 31 + timeZone.hashCode()
    return result
  }
}
```

### Item 44: Respect the contract of compareTo

다음과 같은 연산들이 `compareTo()` 를 호출한다. `compareTo()` 는 
`Comparable<T> interface` 에도 들어 있다.

```kotlin
a > b  // a.compareTo(b) > 0
a < b  // a.compareTo(b) < 0
a >= b // a.compareTo(b) >= 0
a <= b // a.compareTo(b) <= 0
```

`compareTo()` 를 구현한다면 다음의 규칙을 충족해야 한다.

* 비대칭성
  * `a >= b && b >= a` 이면 `a == b` 이다.
* 연속성
  * `a >= b && b >= c` 이면 `a >= c` 이다.
* 코넥스성 (connex relation)
  * a, b 에 대해 `a >= b` 혹은 `b >= a` 중 적어도 하나는 항상 true 여야 한다.

### Item 45: Consider extracting non-essential parts of your API into extensions

Extention method 는 member method 에 비해 다음과 같은 차이가 있다.

* Extention method 는 읽어 들어야 한다.
* Extention method 는 virtual 이 아니다.
* Member method 의 우선순위가 높다.
* Extention method 는 class 가 아니라 type 위에 만들어 진다.
  * Generic Type 에 Extention method 를 정의할 수도 있다.
* Extention method 는 class 의 reference 에 나오지 않는다.
  * Extention method 는 annotation processor 의 대상이 되지 않는다. 

```kotlin
// Member method, Extention method 는 정말 비슷하다.
// Member method
class Workshop() {
  fun makeEvent(date: DateTime): Event = //...
  val permalink
    get() = "/workshop/$name"
}
// Extention method
class Workshop() {
}
fun Workshop.makeEvent(date: DateTime): Event = //...
val Workshop.permalink
  get() = "/workshop/$name"
// usage
fun useWorkshop(workshop: Workshop) {
  val event = workshop.makeEvent(date)
  val permalink = workshop.permalink
  val makeEventRef = Workshop::makeEvent
  val permalinkPropRef = Workshop::permalink
}

// Extention method 는 virtual 이 아니다. 상속을 해야한다면
// Extention method 는 사용하면 안된다.
// Extention method 는 첫번째 인자로 리시버가 들어가는 함수이다.
open class C
class D: C()
fun C.foo() = "c"  // fun foo('this$receiver': C) = "c"
fun D.foo() = "d"  // fun foo('this$receiver': C) = "d"

// usage
fun main() {
  val d = D()
  print(d.foo()) // d
  val c: C = d
  print(c.foo()) // c
  print(D().foo()) // d
  print((D() as C).foo()) // c
}

// Extention method 는 class 가 아닌 type 에 정의하는 것이다.
// 따라서 nullable 혹은 Concrete Generic type 에도 구현할 수 있다.
inline fun CharSequence?.isNullOrBlank(): Boolean {
  contract {
    returns(false) implies (this@isNullOrBlank != null)
  }
  return this == null || this.isBlank()
}

public fun Iterable<Int>.sum(): Int {
  var sum: Int = 0
  for (element in this) {
    sum += element
  }
  return sum
}
```

### Item 46: Avoid member extensions

Extention method 는 class 의 member 로 두지 말자. 혼란스럽다.

```kotlin
// 다음은 Extention method 이다.
fun String.isPhoneNumber(): Boolean {
  length == 7 && all { it.isDigit() }
}
// compile 되면 다음과 같이 변한다.
fun isPhoneNumber('$this': String): Boolean {
  '$this'.length == 7 && '$this'.all { it.isDigit() }
}

// 따라서 다음과 같이 interface 혹은 class 의 member 로 
// Extention method 를 정의할 수 있다. 이 것은 매우 좋지 않다.
interface PhoneBook {
  fun String.isPhoneNumber(): Boolean
}
class Fizz: PhoneBook {
  override fun String.isPhoneNumber(): Boolean =
    length == 7 && all { it.isDigit() }
}
class PhoneBookIncorrect {
  fun String.isPhoneNumber() = 
    length == 7 && all { it.isDigit() }
}

// 가시성 때문에 이러헥 하고 싶다면 top level Extention method 에
// access modifier 를 추가하라.
private fun String.isPhoneNumber() = 
  length == 7 && all { it.isDigit() }

// reference 를 지원하지 않는다.
val ref = String::isPhoneNumber
val sr = "1234567890"
val boundedRef = str::isPhoneNumber

val refX = PhoneBookIncorrect::isPhoneNumber // ERROR
val book = PhoneBookIncorrect()
val boundedRefX = book::isPhoneNumber // ERRPR
```

# Part 3: Efficiency

## Chapter 7: Make it cheap

### Item 47: Avoid unnecessary object creation

객체는 최대한 생성하지 말자. 주로 다음과 같은 방법들이 있다.

* 객체를 하나 선언해서 재활용한다.
* 캐시를 이용하는 팩토리 함수를 이용한다.
* 무거운 객체는 외부로 보내자.
* 무거운 객체는 lazy init 하자.
* primitive type 을 사용하자.

Kotlin 에서 `Int` 는 `int` 로 `Int?` 는 `Integer` 로 compile 된다. Kotlin 은
문자열 혹은 `[-127,127]` 의 Int 를 caching 한다.

```kotlin
val a = "Hello World"
val b = "Hello World"
print(a == b)  // true
print(a === b) // true

val a: Int? = 1
val b: Int? = 1
print(a == b)  // true
print(a === b) // true

val a: Int? = 1234
val b: Int? = 1234
print(a == b)  // true
print(a === b) // false
```

객체를 하나 선언해서 재활용해보자.

```kotlin
// AsIs: Empty() 를 매번 생성한다. 비효율적이다.
sealed class LinkedList<T>
class Node<T>(
  val head: T,
  val tail: LinkedList<T>
): LinkedList<T>
class Empty<T>: LinkedList<T>
val list1: LinkedList<Int> = 
  Node(1, Node(2, Node(3, Empty())))
val list2: LinkedList<Int> =
  Node("A", Node("B", Empty()))
// ToBe: 
sealed class LinkedList<out T>
class Node<out T>(
  val head: T,
  val tail: LinkedList<T>
): LinkedList<T>()
object Empty: LinkedList<Nothing>()
val list1: LinkedList<Int> = 
  Node(1, Node(2, Node(3, Empty())))
val list2: LinkedList<Int> =
  Node("A", Node("B", Empty()))
```

캐시를 이용하는 팩토리 함수를 사용하자.

```kotlin
// connection 을 caching 한다. 객체를 재활용한다.
private val connections 
  = mutableMapOf<String, Connection>()
fun getConnection(host: String) =
  connections.getOrPut(host) { createConnection(host) }

// fibonacci 를 memoization 을 이용해 구현한다. 객체를 재활용한다.
private val FIB_CACHE = mutableMapOf<Int, BigInteger>()
fun fib(n: Int): BigInteger = FIB_CACHE.getOrPut(n) {
  if (n <= 1) BigIntger.ONE else fib(n - 1) + fib(n - 2)
}
```

무거운 객체는 외부 스코프로 보내보자. 

```kotlin
// AsIs: 매번 this.max() 가 호출된다.
fun <T: Comparable<T>> Iterable<T>.countMax(): Int = 
  count { it == this.max() }
// ToBe: 한번 this.max() 가 호출된다.
fun <T: Comparable<T>> Iterable<T>.countMax(): Int {
  val max = this.max()
  return count { it == max }
}

// AsIs: 매번 matches() 를 호출한다.
fun String.isValidIpAddress(): Boolean {
  return this.matches("\\A(?:(?:25[0-5]|2[0-4][0-9]
|[01]?[0-9][0-9]?)\\.){3}(?:25[0-25]|2[0-4]
[0-9]|[01]?[0-9][0-9]?\\z".toRegex())
}
print("5.173.80.254".isValidIpAddress())  // true
// ToBe: 한번 matches() 를 호출한다.
private val IS_VALID_EMAIL_REGEX = "\\A(?:(?:25[0-5]|2[0-4][0-9]
|[01]?[0-9][0-9]?)\\.){3}(?:25[0-25]|2[0-4]
[0-9]|[01]?[0-9][0-9]?\\z".toRegex()
fun String.isValidIpAddress(): Boolean = 
  matches(IS_VALID_EMAIL_REGEX)
// ToBe: lazy init 
private val IS_VALID_EMAIL_REGEX by lazy {
"\\A(?:(?:25[0-5]|2[0-4][0-9]
|[01]?[0-9][0-9]?)\\.){3}(?:25[0-25]|2[0-4]
[0-9]|[01]?[0-9][0-9]?\\z".toRegex()
}
```

무거운 객체는 lazy init 를 하자.

```kotlin
// AsIs: B, C, D 는 무거운 class 라고 하자.
class A {
  val b = B()
  val c = C()
  val d = D()
}
// ToBe:
class A {
  val b by lazy { B() } 
  val c by lazy { C() } 
  val d by lazy { D() } 
}
```

primitive type 을 이용하라.

```kotlin
// AsIs: 
// 매번 Elvis 연산을 사용해야 한다.
// Int? 는 Integer 와 같다. primitive type 이 더욱 좋다. 
fun Iterable<Int>.maxOrNull(): Int? {
  var max: Int? = null
  for (i in this) }{
    max = if (i > (max ?: Int.MIN_VALUE)) i else max
  }
  return max
}
// ToBe
fun Iterable<T>.maxOrNull(): Int? {
  val iterator = iterator()
  if (!iterator.hasNext()) {
    return null
  }
  var max: Int = iterator.next()
  while (iterator.hasNext()) {
    val e = iterator.next()
    if (max < e) {
      max = e
    }
  }
}
// ToBe: Extention method 로 제작해 보자.
public fun <T: Comparable<T>> Iterable<T>.max(): T? {
  val iterator = iterator()
  if (!iterator.hasNext()) {
    return null
  }
  var max = iterator.next()
  while (iterator.hasNext()) {
    val e = iterator.next()
    if (max < e) {
      max = e
    }
  }
  return max
}
```

### Item 48: Use inline modifier for functions with parameters of functional types

funtional type parameter 를 사용한다면 inline 을 추가하자. inline function 은 functional type argument 의 body 를 그대로 붙여넣는다.

```kotlin
inline fun repeat(times: Int, action: (Int) -> Unit) {
  for (index in 0 until times) {
    action(index)
  }
}
// This will be compiled
repeat(10) {
  print(it)
}
// like this
for (index in 0 until 10) {
  action(index)
}
```

inline function 은 다음과 같은 장점이 있다.

* type argument 에 reified 를 붙일 수 있다.
* functional type parameter 를 사용하는 함수의 경우 더 빠르다.
* non-local return 을 사용할 수 있다.

inline function 은 다음과 같은 단점이 있다.

* code 를 붙여넣기 하기 때문에 code 가 커질 수 있다.

하나씩 깊게 들어가 보자.

type argument 에 reified 를 붙일 수 있다. reified 를 붙이면 type parameter 를
사용한 부분이 type argument 로 대체 된다.

```kotlin
inline fun <reified T> printTypeName() {
  print(T::class.simpleName)
}
// 이 것이 대체된다.
printTypeName<Int>()    // Int
printTypeName<Char>()   // Char
printTypeName<String>() // String
// 이렇게
print(Int::class.simpleName) // Int
print(Char::class.simpleName) // Char
print(String::class.simpleName) // String

// filterIsInstance 는 reified 가 포함되어 정의되었다.
class Worker
class Manager
val employees: List<Any> =
  listOf(Worker(), Manager(), Worker())
val workers: List<Worker> =
  employees.filterIsInstance<Worker>()
```

functional type parameter 를 사용하는 함수의 경우 더 빠르다.

```kotlin
// 다음은 inline 을 부착하더라도 이득이 없다. IntelliJ 가 warning 을 보여줄 것임
inline fun print(msg: Any?) {
  System.out.print(msg)
}

// Kotlin/JVM 에서 Lambda Expression 은 Class 로 compile 된다.
val lambda: () -> Unit = {
  //...
}
Function0<Unit> lamda = new Function0<Unit>() {
  public Unit invoke() {
    //...
  }
}
```

다음은 kotlin 의 각 함수와 컴파일된 Java Class 이다.

| Kotlin | Java |
|--|--|
| `() -> Unit` | `Function0<Unit>` |
| `() -> Int` | `Function0<Int>` |
| `(Int) -> Int` | `Function1<Int, Int>` |
| `(Int, Int) -> Int` | `Function2<Int, Int, Int>` |

non-local return 을 사용할 수 있다.

```kotlin
// AsIs
if (value != null) {
  print(value)
}
for (i in 1..10) {
  print(i)
}
repeatNoninline(10) {
  print(it)
}
fun main() {
  repeatNoninline(10) {
    print(it)
    return // ERROR: not allowed
  }
}
// ToBe: repeat 은 inline function 이다.
fun main() {
  rpeat(10) {
    print(it)
    return // OK
  }
}
// 아래 코드는 non-local return 을 사용하기 때문에 자연스럽다.
fun getSomeMoney(): Money? {
  repeat(100) {
    val money = searchForMoney()
    if (money != null) {
      return money
    }
  }
  return null
}
```

inline function 으로 만들고 싶다. 일부 functional type parameter 는
inline 으로 받고 싶지 않다. 그럴 때 **crossinline, noninline** 을 사용하자.

```kotlin
inline fun requestNewToken(
  hasToken: Boolean,
  crossinline onRefresh: () -> Unit,
  noinline onGenerate: () -> Unit
) {
  if (hasToken) {
    httpCall("get-token", onGenerate)
    // inline function argument 를 받고 싶지 않을 때
    // noinline 을 사용한다.
  } else [
    httpCall("refresh-token") {
      onRefresh()
      // non-local return 이 허용되지 않는 context 에서
      // inline function argument 를 사용하고 싶다면
      // crossinline 을 사용한다. 
      onGenerate()
    } 
  ]
}
fun httpCall(url: String, callback: () -> Unit) {
  //...
}
```

### Item 49: Consider using inline value classes

inline function 뿐만 아니라 inline class 도 가능하다. inline class 는 object 를
새로 만들지 않기 때문에 overhead 가 없다.

```kotlin
// inline class 를 정의한다.
inline class Name(private val value: String) {
  //...
}

val name: Name = Name("David")
// 위 코드는 컴파일되면 아래와 같다.
val name: String = "David"

// inline class 의 method 는 static method 이다.
inline class Name(private val value: String) {
  fun greet() {
    print("Hello I am $value")
  }
}

val name: Name = Name("David")
name.greet()
// 위의 코드가 컴파일되면 아래와 같다.
val name: String = "David")
Name.'greet-impl'(name)
```

inline class 는 주로 다음과 같은 경우에 사용한다.

* 측정 단위를 표현할 때
* 타입 오용으로 발생하는 문제를 막을 때

하나씩 깊게 들어가보자.

특정 단위를 표현할 때

```kotlin
// AsIs: time 은 millis 인가??? minute인가??? second인가???
interface Timer {
  fun callAfter(time: Int, callback: () -> Unit)
}
// ToBe: named argument 를 사용하면 더 명확하다. 그러나 강제하고 싶다.
// named return 은 지원되지 않는다.
interface Timer {
  fun callAfter(timeMillis: Int, callback: () -> Unit)
}

// AsIs:
interface User {
  fun decideAboutTime(): Int
  fun wakeUp()
}
interface Timer {
  fun callAfter(timeMillis: Int, callback: () -> Unit)
}
fun setUpUserWakeUpUser(user: User, timer: Timer) {
  val time: Int = user.decideAboutTime()
  timer.callAfter(time) {
    user.wakeUp()
  }
}
// ToBe: 타입에 제한을 걸자
inline class Minutes(val minutes: Int) {
  fun toMillis(): Millis = Millis(minutes * 60 * 1000)
}
inline class Millis(val milliseconds: Int) {
  //...
}
interface User {
  fun decideAboutTime(): Minutes
  fun wakeUp()
}
interface Timer {
  fun callAfter(timeMillis: Miilis, callback: () -> Unit)
}
fun setUpUserWakeUpUser(user: User, timer: Timer) {
  val time: Minutes = user.decideAboutTime()
  timer.callAfter(time.toMillis()) {
    user.wakeUp()
  }
}
```

타입 오용으로 발생하는 문제를 막을 때

```kotlin
// AsIs: type 이 강제되지 않고 있다. 잘못된 숫자가 들어가도
// 동작한다.
@Entity(tableName = "grades")
class Grades(
  @ColumnInfo(name = "studentId")
  val studentId: Int,
  @ColumnInfo(name = "teacherId")
  val teacherId: Int,
  @ColumnInfo(name = "schoolId")
  val schoolId: Int,
)
// ToBe: type 을 강제하고 있다. 잘못된 숫자가 들어갈 수 없다.
inline class StudentId(val studentId: Int)
inline class TeacherId(val teacherId: Int)
inline class SchoolId(val schoolId: Int)
@Entity(tableName = "grades")
class Grades(
  @ColumnInfo(name = "studentId")
  val studentId: StudentId,
  @ColumnInfo(name = "teacherId")
  val teacherId: TeacherId,
  @ColumnInfo(name = "schoolId")
  val schoolId: SchoolId,
)
```

inline class 는 interface 를 구현할 수도 있다. incline class 는 더이상 inline 으로 동작하지 않는다. 따라서 아무런 장점이 없다.

```kotlin
interface TimeUnit {
  val millis: Long
}
inline class Minutes(val minutes: Long): TimeUnit {
  override val millis: Long get() = minutes * 60 * 1000
}
inline class Millis(val milliseconds: Long): TimeUnit {
  override val millis: Long get() = milliseconds
}
fun setUpTimer(time: TimeUnit) {
  val millis = time.millis
}
setUpTimer(Minutes(123))
setUpTimer(Millis(456789))
```

typealias 를 이용하면 type 에 새로운 이름을 붙여 줄 수 있다. 그러나 안전하지
않다.

```kotlin
// 다음과 같이 사용한다.
typealias NewName = Int
val n: NewName = 10

// 이렇게 반복적으로 사용할 type 을 정의하면 편하다.
typealias ClickListener = 
  (view: View, event: Event) -> Unit
class View {
  fun addclickListener(listener: ClickListener) {}
  fun removeClickListener(listener: ClickListener) {}
  //...
}

// typealias 를 사용하면 type 안전하지 못하다. 그냥 class 써라. 
typealias Seconds = Int
typealias Millis = Int

fun getTime(): Millis = 10
fun setUpTimer(time: Seconds) {}

fun main() {
  val seconds: Seconds = 10
  val millis: Millis = seconds // ERROR 가 아니라니 !!!
  setUpTimer(getTime())
}
```

### Item 50: Eliminate obsolete object references

사용하지 않는 객체의 레퍼런스를 null 로 저장해두자. garbage 
collector 가 회수할 것이다.

## Chapter 8: Efficient collection processing

### Item 51: Prefer Sequence for big collections with more than one processing step

다음은 Iterable 과 Sequence 의 차이이다.

* Iterable 은 operator function 이 수행될 때 마다 실행된다.
* Sequence 는 중간 연산이 모여졌다가 `toList(), count()` 와 같은 최종 연산이 수행될 때
  모여진 operator function 이 수행된다.

```kotlin
// Iterable 과 Sequence 는 정의가 비슷하다.
interface Iterable<out T> {
  operator fun iterator(): Iterator<T>
}
interface Sequence<out T> {
  operator fun iterator(): Iterator<T>
}

// Iterable
public inline fun <T> Iterable<T>.filter(
  predicate: (T) -> Boolean
): List<T> {
  return filterTo(ArrayList<T>(), predicate)
}
// Sequence
public inline fun <T> Iterable<T>.filter(
  predicate: (T) -> Boolean
): Sequence<T> {
  return filteringSequence(this, true, predicate)
}

// Iterable
val list = listOf(1, 2, 3)
val listFiltered = list
  .filter { print("f$it "); it % 2 == 1 }
// f1 f2 f3
println(listFiltered) // [1, 2, 3]

// Sequence
val seq = sequenceOf(1, 2, 3)
val filtered = seq.filter { print("f$it "); it % 2 == 1 }
println(filtered)  // FilterinSgSequence@...

val asList = filtered.toList()
// f1 f2 f3
println(asList)  // [1, 2, 3]
```

Iterable 과 Sequence 는 연산의 처리방식이 다르다.

* `element-by-element order (lazy order)`
  * element 에 모여진 연산을 적용한다.
* `step-by-step order (eager order)`
  * 모든 element 에 대해 연산을 하나씩 적용한다.

```kotlin
// Sequence
sequenceOf(1, 2, 3)
  .filter { print("F$it, "); it % 2 == 1 }
  .map { print("M$it, "); it * 2 }
  .forEach { print("E$it, ") }
// Output:
// F1, M1, E2, F2, F3, M3, M6

// Iterable
listOf(1, 2, 3)
  .filter { print("F$it, "); it % 2 == 1 }
  .map { print("M$it, "); it * 2 }
  .forEach { print("E$it, ") }
// Output:
// F1, F2, F3, M1, M3, E2, E6

// Not using collection operators and it is same with Sequence
// lazy order
for (e in listOf(1, 2, 3)) {
  print("F$e, ")
  if (e % 2 == 1) {
    print("M$e, ")
    val mapped = e * 2
    print("E$mapped, ")
  }
}
// Output:
// F1, M1, E2, F2, F3, M3, M6
```

필요한 것을 찾고 싶을 때 Squence 를 사용하자. 필요한 연산만 수행할 수 있다.
`find()` 를 사용하면 찾을 때까지 중간 연산을 수행한다.

```kotlin
// Sequence
(1..10).asSequence()
  .filter { print("F$it, "); it % 2 == 1 }
  .map { print("M$it, "); it * 2 }
  .find { it > 5 }
// Output:
// F1, M1, F2, F3, M3

// Iterable
(1..10)
  .filter { print("F$it, "); it % 2 == 1 }
  .map { print("M$it, "); it * 2 }
  .find { it > 5 }
// Output:
// F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
// M1, M3, M5, M7, M9
```

다음과 같은 방법으로 Infinite Sequence 를 구현할 수 있다.

* `generateSequence()`
* `sequence()`

Initinite Sequence 를 사용할 때는 무한 loop 를 조심하자.

```kotlin
// generateSequence 로 Infinite Sequence 를 만들자.
// 첫번째 element 를 만들어내는 방법과 두번째 element 를
// 만들어내는 방법이 필요하다.
generateSequence(1) { it + 1 }
  .map { it * 2 }
  .take(10)
  .forEach { print("$it, ") }
// Output:
// 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,

// sequence 로 Infinite Sequence 를 만들자.
// yield 를 이용한다.
val fibonacci = sequence {
  yield(1)
  var current = 1
  var prev = 1
  while (true) {
    yield(current)
    val temp = prev
    prev = current
    current += temp
  }
}
```

Sequence 는 중간 연산에서 collection 을 만들어 내지 않는다.

```kotlin
// Iterable
members
  .filter { it % 10 == 0 } // Create collection
  .map { it * 2 }          // Create collection
  .sum()
// Sequence
members
  .asSequence()
  .filter { it % 10 == 0 }
  .map { it * 2 }         
  .sum()

// AsIs: 큰 파일의 경우 Iterable 은 OutOfMemory 를 발생시킨다.
// "ChicagoCrimes.csv" is 1.53GB
File("ChicagoCrimes.csv").readLines() // List<String>
  .drop(1) // remove column row
  .mapNotNull { it.split(",").getOrNull(6) }
  .filter { "CANNABIS" in it }
  .count()
  .let(::println)
// ToBe:
File("ChicagoCrimes.csv").useLines() // Sequence<String>
  .drop(1) // remove column row
  .mapNotNull { it.split(",").getOrNull(6) }
  .filter { "CANNABIS" in it }
  .count()
  .let(::println)

// 중간 연산이 하나이면 큰 차이가 없다.
fun singleStepListProcessing(): List<Product> {
  return productsList.filter { it.bought }
}
fun singleStepSequenceProcessin(): List<Product> {
  return productsList.asSequence()
    .filter { it.brought }
    .toList()
}

// 중간 연산이 두개 이상이면 차이가 많다.
fun twoStepListProcessing(): List<Double> {
  return productsList
    .filter { it.bought }
    .map { it.price }
}
fun twoStepSequenceProcessing(): List<Double> {
  return productsList.asSequence()
    .filter { it.bought }
    .map { it.price }
    .toList()
}
fun threeStepListProcessing(): Double {
  return productsList
    .filter { it.bought }
    .map { it.price }
    .average()
}
fun threeStepSequenceProcessing(): Double {
  return productsList.asSequence()
    .filter { it.bought}
    .map { it.price }
    .average
}

// twoStepListProcessing        81,095 ns
// twoStepSequenceProcessing    55,685 ns
// threeStepListProcessing      83,307 ns
// threeStepSequenceProcessing   6,928 ns
```

`sorted()` 의 경우는 Sequence 보다 Iterable 이 더 빠르다. Sequence 의 경우
Iterable 로 변환하고 정렬하기 때문이다. 중간연산이 두개 이상이면 Sequence 가 더
빠르다.

```kotlin
// AsIs: 150,482 ns 
fun productsSortAndProcessingList(): Double {
  return productsList
    .sortedBy { it. price }
    .filter { it. bought }
    .map { it.price }
    .average()
}
// ToBe: 96,811 ns
fun productsSortAndProcessingSequence(): Double {
  return productsList.asSequence()
    .sortedBy { it. price }
    .filter { it. bought }
    .map { it.price }
    .average()
}
```

Java Stream 은 Sequence 와 비슷하다. 다음과 같은 차이가 있다.

* Kotlin/JVM 만 지원한다.
* Kotlin 의 Sequence 보다 처리함수가 적다.
* 병렬 함수를 사용해서 병렬 모드로 실행할 수 있다.

```kotlin
// Sequence
productsList.asSequence()
  .filter { it.bought }
  .map { it.price }
  .average()
// Stream
productsList.stream()
  .filter { it.bought }
  .map { it.price }
  .average()
  .orElese(0.0)
```

### Item 52: Consider associating elements to a map

### Item 53: Consider using groupingBy instead of groupBy

### Item 54: Limit the number of operations

collection 의 operation function 을 최소로 사용하자.

```kotlin
class Student(val name: String?)
// AsIs
fun List<Student>.getNames(): List<String> = this
  .map { it.name }
  .filter { it != null }
  .map { it!! }
// ToBe
fun List<Student>.getNames(): List<String> = this
  .map { it.name }
  .filterNotNull()
// ToBe
fun List<Student>.getNames(): List<String> = this
  .mapNotNull { it.name }
```

다음은 복잡한 operation function 을 최적화 하는 방법이다.

```kotlin
// AsIs:
// .filter { it != null }
// .map { it!! }
// ToBe:
// .filterNotNull()

// AsIs:
// .map { <Transformation> }
// .filterNotNull()
// ToBe:
// .mapNotNull { <Transformation> }

// AsIs:
// .map { <Transformation> }
// .joinToString()
// ToBe:
// .joinToString { <Transformation> }

// AsIs:
// .filter { <Predicate 1> }
// .filter { <Predicate 2> }
// ToBe:
// .filter { <Predicat 1> && <Predicate 2> }

// AsIs:
// .filter { it is Type }
// .map { it as Type }
// ToBe:
// .filterIsInstance<Type>()

// AsIs:
// .sortedBy { <Key 2> }
// .sortedBy { <Key 1> }
// ToBe:
// .sortedWith( compareBy({ <Key 1> }, { <Key 2> }))

// AsIs:
// listOf(..)
// .filterNotNull()
// ToBe:
// listOfNotNull(...)

// AsIs:
// .withIndex()
// .filter { (index, elem) ->
//   <Predicate using index>
// }
// .map { it.value }
// ToBe:
// .filterIndexed { index, elem ->
//   <Predicate using index>
// }
```

### Item 55: Consider Arrays with primitives for performance-critical processing

성능이 중요하다면 primitive array 를 사용하라.

다음은 Kotlin 과 Java 의 중요타입을 비교한 것이다.

| Kotlin | Java |
|--|--|
| `Int` | `int` |
| `List<Int>` | `List<Integer>` |
| `Array<Int>` | `Integer[]` |
| `IntArray` | `int[]` |


```kotlin
open class InlineFilterBenchmark {
  lateinit var list: List<Int>
  lateinit var array: IntArray

  @Setup
  fun init() {
    list = List(1_000_000) { it }
    array = IntArray(1_000_000) { it }
  }

  // 1,260,593 ns in average
  @Benchmark
  fun averageOnIntList(): Double {
    return list.average()
  }

  // 868,509 ns in average
  @Benchmark
  fun averageOnIntArray(): Double {
    return array.average()
  }
}
```

### Item 56: Consider using mutable collections

immutable collection 은 객체 생성을 자주한다. mutable collection 은 객체 생성을
한번만 한다. thread-safety 가 문제되지 않는다면 mutable cllection 을 사용하자.

```kotlin
// ArrayList<T> object 가 생성된다. 비효율적이다.
operator fun <T> Iterable<T>.plus(element: T): List<t> {
  if (this is Collection) {
    return this.plus(element)
  }
  val result = ArrayList<T>()
  result.addAll(this)
  result.add(element)
  return result
}

// AsIs: immutable collection 을 사용하는 것은 비효율적이다.
inline fun <T, R> Iterable<T>.map(
  transform: (T) -> R
): List<R> {
  val destination = listOf<R>()
  for (item in this) {
    destination.add(transform(item))
  }
  return destination
}
// ToBe: stdlib 의 map 은 mutable collection 을 사용한다.
inline fun <T, R> Iterable<T>.map(
  transform: (T) -> R
): List<R> {
  val size = if (this is Collection<*>) this.size else 10
  val destination = ArrayList<R>(size)
  for (item in this) {
    destination.add(transform(item))
  }
  return destination
}
```
