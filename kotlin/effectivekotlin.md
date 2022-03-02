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
    - [Item 22: Reuse between different platforms by extracting common modules](#item-22-reuse-between-different-platforms-by-extracting-common-modules)
    - [Chapter 4: Abstraction design](#chapter-4-abstraction-design)
    - [Item 23: Each function should be written in terms of a single level of abstraction](#item-23-each-function-should-be-written-in-terms-of-a-single-level-of-abstraction)
    - [Item 24: Use abstraction to protect code against changes](#item-24-use-abstraction-to-protect-code-against-changes)
    - [Item 25: Specify API stability](#item-25-specify-api-stability)
    - [Item 26: Consider wrapping external API](#item-26-consider-wrapping-external-api)
    - [Item 27: Minimize elements visibility](#item-27-minimize-elements-visibility)
    - [Item 28: Define contract with documentation](#item-28-define-contract-with-documentation)
    - [Item 29: Respect abstraction contracts](#item-29-respect-abstraction-contracts)
  - [Chapter 5: Object creation](#chapter-5-object-creation)
    - [Item 30: Consider factory functions instead of constructors](#item-30-consider-factory-functions-instead-of-constructors)
    - [Item 31: Consider a primary constructor with named optional arguments](#item-31-consider-a-primary-constructor-with-named-optional-arguments)
    - [Item 32: Consider defining a DSL for complex object creation](#item-32-consider-defining-a-dsl-for-complex-object-creation)
  - [Chapter 6: Class design](#chapter-6-class-design)
    - [Item 33: Prefer composition over inheritance](#item-33-prefer-composition-over-inheritance)
    - [Item 34: Use the data modifier to represent a bundle of data](#item-34-use-the-data-modifier-to-represent-a-bundle-of-data)
    - [Item 35: Use function types instead of interfaces to pass operations and actions](#item-35-use-function-types-instead-of-interfaces-to-pass-operations-and-actions)
    - [Item 36: Prefer class hierarchies to tagged classes](#item-36-prefer-class-hierarchies-to-tagged-classes)
    - [Item 37: Respect the contract of equals](#item-37-respect-the-contract-of-equals)
    - [Item 38: Respect the contract of hashCode](#item-38-respect-the-contract-of-hashcode)
    - [Item 39: Respect the contract of compareTo](#item-39-respect-the-contract-of-compareto)
    - [Item 40: Consider extracting non-essential parts of your API into extensions](#item-40-consider-extracting-non-essential-parts-of-your-api-into-extensions)
    - [Item 41: Avoid member extensions](#item-41-avoid-member-extensions)
- [Part 3: Efficiency](#part-3-efficiency)
  - [Chapter 7: Make it cheap](#chapter-7-make-it-cheap)
    - [Item 42: Avoid unnecessary object creation](#item-42-avoid-unnecessary-object-creation)
    - [Item 43: Use inline modifier for functions with parameters of functional types](#item-43-use-inline-modifier-for-functions-with-parameters-of-functional-types)
    - [Item 44: Consider using inline classes](#item-44-consider-using-inline-classes)
    - [Item 45: Eliminate obsolete object references](#item-45-eliminate-obsolete-object-references)
  - [Chapter 8: Efficient collection processing](#chapter-8-efficient-collection-processing)
    - [Item 46: Prefer Sequence for big collections with more than one processing step](#item-46-prefer-sequence-for-big-collections-with-more-than-one-processing-step)
    - [Item 47: Limit number of operations](#item-47-limit-number-of-operations)
    - [Item 48: Consider Arrays with primitives for performance-critical processing](#item-48-consider-arrays-with-primitives-for-performance-critical-processing)
    - [Item 49: Consider using mutable collections](#item-49-consider-using-mutable-collections)

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
### Item 7: Prefer null or Failure result when the lack of result is possible
### Item 8: Handle nulls properly
### Item 9: Close resources with use
### Item 10: Write unit tests
## Chapter 2: Readability
### Item 11: Design for readability
### Item 12: Operator meaning should be clearly consistent with its function name
### Item 13: Avoid returning or operating on Unit?
### Item 14: Specify the variable type when it is not clear
### Item 15: Consider referencing receivers explicitly
### Item 16: Properties should represent state, not behavior
### Item 17: Consider naming arguments
### Item 18: Respect coding conventions
# Part 2: Code design
## Chapter 3: Reusability
### Item 19: Do not repeat knowledge
### Item 20: Do not repeat common algorithms
### Item 21: Use property delegation to extract common property patterns
### Item 22: Reuse between different platforms by extracting common modules
### Chapter 4: Abstraction design
### Item 23: Each function should be written in terms of a single level of abstraction
### Item 24: Use abstraction to protect code against changes
### Item 25: Specify API stability
### Item 26: Consider wrapping external API
### Item 27: Minimize elements visibility
### Item 28: Define contract with documentation
### Item 29: Respect abstraction contracts
## Chapter 5: Object creation
### Item 30: Consider factory functions instead of constructors
### Item 31: Consider a primary constructor with named optional arguments
### Item 32: Consider defining a DSL for complex object creation
## Chapter 6: Class design
### Item 33: Prefer composition over inheritance
### Item 34: Use the data modifier to represent a bundle of data
### Item 35: Use function types instead of interfaces to pass operations and actions
### Item 36: Prefer class hierarchies to tagged classes
### Item 37: Respect the contract of equals
### Item 38: Respect the contract of hashCode
### Item 39: Respect the contract of compareTo
### Item 40: Consider extracting non-essential parts of your API into extensions
### Item 41: Avoid member extensions
# Part 3: Efficiency
## Chapter 7: Make it cheap
### Item 42: Avoid unnecessary object creation
### Item 43: Use inline modifier for functions with parameters of functional types
### Item 44: Consider using inline classes
### Item 45: Eliminate obsolete object references
## Chapter 8: Efficient collection processing
### Item 46: Prefer Sequence for big collections with more than one processing step
### Item 47: Limit number of operations
### Item 48: Consider Arrays with primitives for performance-critical processing
### Item 49: Consider using mutable collections
