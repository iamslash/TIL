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
    - [Item 34: Consider a primary constructor with named optional arguments](#item-34-consider-a-primary-constructor-with-named-optional-arguments)
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
### Item 23: Avoid shadowing type parameters
### Item 24: Consider variance for generic types
### Item 25: Reuse between different platforms by extracting common modules
## Chapter 4: Abstraction design
### Item 26: Each function should be written in terms of a single level of abstraction
### Item 27: Use abstraction to protect code against changes
### Item 28: Specify API stability
### Item 29: Consider wrapping external API
### Item 30: Minimize elements visibility
### Item 31: Define contract with documentation
### Item 32: Respect abstraction contracts
## Chapter 5: Object creation
### Item 33: Consider factory functions instead of constructors
### Item 34: Consider a primary constructor with named optional arguments
### Item 35: Consider defining a DSL for complex object creation
## Chapter 6: Class design
### Item 36: Prefer composition over inheritance
### Item 37: Use the data modifier to represent a bundle of data
### Item 38: Use function types or functional interfaces to pass operations and actions
### Item 39: Use sealed classes and interfaces to express restricted hierarchies
### Item 40: Prefer class hierarchies to tagged classes
### Item 41: Use enum to represent a list of values
### Item 42: Respect the contract of equals
### Item 43: Respect the contract of hashCode
### Item 44: Respect the contract of compareTo
### Item 45: Consider extracting non-essential parts of your API into extensions
### Item 46: Avoid member extensions
# Part 3: Efficiency
## Chapter 7: Make it cheap
### Item 47: Avoid unnecessary object creation
### Item 48: Use inline modifier for functions with parameters of functional types
### Item 49: Consider using inline value classes
### Item 50: Eliminate obsolete object references
## Chapter 8: Efficient collection processing
### Item 51: Prefer Sequence for big collections with more than one processing step
### Item 52: Consider associating elements to a map
### Item 53: Consider using groupingBy instead of groupBy
### Item 54: Limit the number of operations
### Item 55: Consider Arrays with primitives for performance-critical processing
### Item 56: Consider using mutable collections
