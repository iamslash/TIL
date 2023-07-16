- [Abstract](#abstract)
- [References](#references)
- [Materials](#materials)
- [Basic](#basic)
  - [Emacs Config](#emacs-config)
  - [Play Ground](#play-ground)
  - [Build and Run](#build-and-run)
  - [Keywords](#keywords)
  - [let, var](#let-var)
  - [Data Types](#data-types)
  - [Min, Max Values](#min-max-values)
  - [Absolute Value](#absolute-value)
  - [Bit Manipulation](#bit-manipulation)
  - [String](#string)
  - [Random](#random)
  - [Print Out](#print-out)
  - [Formatted String](#formatted-string)
  - [Function](#function)
  - [Basic Operators](#basic-operators)
  - [Control Flow](#control-flow)
    - [Conditional](#conditional)
    - [Loop](#loop)
  - [Collections Compared With C++ Container](#collections-compared-with-c-container)
  - [Collections](#collections)
    - [Array](#array)
    - [Set](#set)
    - [Dictionary](#dictionary)
  - [Collection Conversions](#collection-conversions)
  - [Sort](#sort)
  - [Search](#search)
  - [Multi Dimensional Array](#multi-dimensional-array)
  - [Optional](#optional)
  - [Struct](#struct)
  - [Class](#class)
  - [Enumerations](#enumerations)
  - [Value Type vs Reference Type](#value-type-vs-reference-type)
  - [Closures](#closures)
  - [Properties](#properties)
  - [Methods](#methods)
  - [Subscripts](#subscripts)
  - [Inheritance](#inheritance)
  - [Initialization, Deinitialization](#initialization-deinitialization)
  - [Optional Chaining](#optional-chaining)
  - [Error Handling](#error-handling)
  - [Concurrency](#concurrency)
  - [Macros](#macros)
  - [Print Type](#print-type)
  - [Type Casting](#type-casting)
  - [Assert, Guard](#assert-guard)
  - [Nested Types](#nested-types)
  - [Protocol](#protocol)
  - [Extension](#extension)
  - [Higher Order Function](#higher-order-function)
  - [Core Libraries](#core-libraries)
  - [Generics](#generics)
  - [Opaque and Boxed Types](#opaque-and-boxed-types)
  - [Automatic Reference Counting](#automatic-reference-counting)
  - [Memory Safety](#memory-safety)
  - [Access Control](#access-control)
  - [Advanced Operators](#advanced-operators)
- [Advanced](#advanced)
  - [Renaming Objective-C APIs for Swift](#renaming-objective-c-apis-for-swift)
  - [Property Wrapper](#property-wrapper)
  - [`@escaping`](#escaping)
  - [Closure vs Async/Await](#closure-vs-asyncawait)
  - [Diagnosing memory, thread, and crash issues early](#diagnosing-memory-thread-and-crash-issues-early)
- [Libraries](#libraries)
- [Style Guide](#style-guide)
- [Refactoring](#refactoring)
- [Effective](#effective)
- [GOF Design Pattern](#gof-design-pattern)
- [Architecture](#architecture)
  - [MV (Model View)](#mv-model-view)
  - [MVVM (Model-View-View-Model)](#mvvm-model-view-view-model)

-------------------------------------------------------------------------------

# Abstract

swift에 대해 정리한다.

# References

* [The Swift Programming Language | swift.org](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/)
  * [kor](https://bbiguduk.gitbook.io/swift)
* [Awesome Swift](https://github.com/matteocrippa/awesome-swift)

# Materials

* [야곰의 스위프트 기본 문법 강좌](https://www.inflearn.com/course/%EC%8A%A4%EC%9C%84%ED%94%84%ED%8A%B8-%EA%B8%B0%EB%B3%B8-%EB%AC%B8%EB%B2%95/)
  * swift 3 를 잘 설명함.
  * [src](https://github.com/yagom/swift_basic)
* [스위프트 프로그래밍 by 야곰 | yes24](https://www.yes24.com/Product/Goods/78907450)
  * Swift 5 를 잘 설명함.
  * [src](https://bitbucket.org/yagom/swift_programming/src/master/)

# Basic

## Emacs Config

* [swift-mode | github](https://github.com/swift-emacs/swift-mode)

## Play Ground

[online Swift playground](http://online.swiftplayground.run/) 가 있다. xcode 에서 Play Ground 를 만들 수 도 있다.

## Build and Run

> [10 Tips to Run Swift From Your Terminal](https://betterprogramming.pub/10-tips-to-run-swift-from-your-terminal-b5832cd9cd8c)

> `a.swift`

```swift
print("Hello World")
```

```bash
# Build
$ swiftc a.swift

# Run
$ ./a
$ swift a.swift

# REPL
$ swift
```

## Keywords

> [Swift Keywords](https://www.tutorialkart.com/swift-tutorial/swift-keywords/)

```swift
// Keywords in declarations
class	    deinit	    Enum	extension
func	    import	    Init	internal
let	        operator	private	protocol
public	    static	    struct	subscript
typealias	var

// Keywords in statements
break	case	continue	default
do	    else	fallthrough	for
if	    in	    return	    switch
where	while

// Keywords in expressions and types
as	    dynamicType	false	is
nil	    self	    Self	super
true	_COLUMN_	_FILE_	_FUNCTION_
_LINE_

// Keywords in specific contexts
associativity	convenience	dynamic	    didSet
final	        get	        infix	    inout
lazy	        left	    mutating	none
nonmutating	    optional	override	postfix
precedence	    prefix	    Protocol	required
right	        set	        Type	    unowned
weak	        willSet
```

## let, var

```swift
let c: String = "Hello World"
var v: String = "Hello World"
v = "Foo Bar"
// c = "Foo Bar" // error

let sum: Int
let a: Int = 100
let b: Int = 200

sum = a + b
// sum = a // error
var nickName: String
nickName = "Foo"
nickName = "Bar"
```

## Data Types

```swift
Bool, 
Int8, Int16, Int32, Int64(Int)
UInt8, UInt16, UInt32, UInt64(UInt)
Float, Double, Character, String
Any, AnyObject, nil
```

```swift
var b: Bool = true
b = false
// b = 0 // error
// b = 1 // error

var i: Int = -100
// i = 100.1 // error

var ui: UInt = 100
// ui = -100
// ui = i // error

var f: Float = 3.14
f = 3

var c: Character = "K"
c = "."
c = "가"
c = "A"
// c = "하하하" // error
print(c)

var s: String = "하하하"
s = s + "어어오오옹"
print(s)
// s = c // error

s = """
ㄴㄹ애ㅑㅓㄴㅇ래ㅑㅑㅓㅐㄴㅇㄹㄴㅇ
ㄴㅇㄹ
ㄴㅇ
ㄹ
ㄴ
ㅇㄹ
ㄴㅇㄹㄴㅇㄹㄴㅇㄹ
"""

var va: Any = 100
va = "ㄹ너ㅐㅑㄴ래ㅑㅓㅑㄴ애ㅓㄹ"
va = 123.12

let ld: Double = va // error

class someClass {}
var someAnyObject: AnyObject = someClass()
someAnyObject = 123.12 // error

someAny = nil // error
someAnyObject = nil // error
```

## Min, Max Values

> [Swift – Integer, Floating-Point Numbers](https://www.geeksforgeeks.org/swift-integer-floating-point-numbers/)

```swift
print("Integer Type        Min                    Max")
print("UInt8           \(UInt8.min)         \(UInt8.max)")
print("UInt16          \(UInt16.min)        \(UInt16.max)")
print("UInt32          \(UInt32.min)        \(UInt32.max)")
print("UInt64          \(UInt64.min)        \(UInt64.max)")
print("Int8            \(Int8.min)          \(Int8.max)")
print("Int16           \(Int16.min)         \(Int16.max)")
print("Int32           \(Int32.min)         \(Int32.max)")
print("Int64           \(Int64.min)         \(Int64.max)")
// Output:
// Integer Type     Min                    Max
// UInt8             0                     255
// UInt16            0                     65535
// UInt32            0                     4294967295
// UInt64            0                     18446744073709551615
// Int8             -128                   127
// Int16            -32768                 32767
// Int32            -2147483648            2147483647
// Int64            -9223372036854775808   9223372036854775807
```

## Absolute Value

```swift
print(abs(-18))   // 18
print(abs(-18.7)) // 18.7
```

## Bit Manipulation

```swift
let a = 0
let b = 1
print(String(a & b, radix: 2))  // 0
print(String(a | b, radix: 2))  // 1
print(String(a ^ b, radix: 2))  // 1
print(String(b << 1, radix: 2)) // 10
print(String(~a, radix: 2))     // -1
```

## String

```swift
// String interpolation
let a: String = "Hello"
print("\(a)")

// String of binary representation
let num = 22
let str = String(num, radix: 2)
print(str)

// String formatting
import Foundation
print(String(format: "Hello %d", 7))  // Helo 7
```

## Random

```swift
for _ in 1...3 {
    print(Int.random(in: 1..<100))
}
```

## Print Out

```swift
let num: Int = 100
print("Hello World \(num)")
```

## Formatted String

* [String Format Specifiers | apple](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/Strings/Articles/formatSpecifiers.html)

```swift
import Foundation

print(String(format: "%x %X", 13, 13))  // a D
```

## Function

```swift
func sum(a: Int, b: Int) -> Int {
    return a + b
}

func printMyName(name: String) -> Void {
    print(name)
}

func maximumIntegerValue() -> Int {
    return Int.max
}

func hello() -> Void { print("hello") }

func bye() { print("bye") }

sum(a: 3, b: 5) // 8
printMyName(name: "Foo") // Foo
printYourName(name: "hana") // hana
printYourName("hana") // hana
maximumIntegerValue() 
hello()
bye()

func greeting(friend: String, me: String = "Foo") {
    print("Hello \(friend)! I'm \(me)")    
}
greeting(friend: "hana")
greeting(friend: "john", me: "eric")

func greeting(to friend: String, from me: String) {
    print("Hello \(friend)! I'm \(me)")
}
greeting(to: "hana", from: "john")

func sayHelloToFriends(me: String, friends: String...) -> String {
    return "Hello \n(friends)! I'm \(me)!"
}
print(sayHelloToFriends(me: "Foo", friends: "Bar", "Baz"))
print(sayHelloToFriends(me: "Foo"))

var someFunction: (String, String) -> Void = greeting(to:from:)
someFunction("eric", "john")
someFunction = greeting(friend:me:)
func runAnother(function: (String, String) -> Void) {
    function("jenny", "mike")
}
runAnother(function: greeting(friend:me:))
runAnother(function: someFunction)

// Function Argument Labels and Parameter Names
//   Argument label for input
//   Parameter name for implementation
// 함수를 호출할 때 argument 전달시 label 을 표기해야 한다. label 을
// 표기하기 싫다면 함수를 정의할 때 label 자리에 _ 를 사용한다. 함수 정의에
// label 이 없다면 parameter 이름을 label 이름으로 대신한다.
func foo(key a: String, val b: String) -> String {
    return "\(a) \(b)"
}
func bar(_ a: String, _ b: String) -> String {
    return "\(a) \(b)"
}
func baz(a: String, b: String) -> String {
    return "\(a) \(b)"
}
print(foo(key: "Hello", val: "World"))
print(bar("Hello", "World"))
print(baz(a: "Hello", b: "World"))
```

## Basic Operators

Arithmetic, comparison, assignment, and logical operators.

```swift
// Arithmetic operators
let sum = 3 + 2 // 5
let difference = 3 - 2 // 1
let product = 3 * 2 // 6
let quotient = 6 / 2 // 3
let remainder = 7 % 3 // 1

// Comparison operators
let isEqual = 5 == 5 // true
let isNotEqual = 5 != 6 // true
let isGreater = 7 > 6 // true
let isLess = 4 < 5 // true
let isGreaterOrEqual = 4 >= 3 // true
let isLessOrEqual = 3 <= 4 // true

// Assignment operators
var a = 10
var b = 5
a = b // a is now 5

// Logical operators
let conditionA = true
let conditionB = false

let andResult = conditionA && conditionB // false
let orResult = conditionA || conditionB // true
let notResult = !conditionA // false
```

## Control Flow

### Conditional

```swift
let someInteger = 100
if someInteger < 100 {
    print("100미만")
} else if someInteger > 100 {
    print("100초과")
} else {
    print("100")
}

switch someInteger {
case 0:
    print("zero")    
case 1..<100:
    print("1~99")
case 100:
    print("100")
case 101...Int.max:
    print("over 100")
default:
    print("unknown")
}

switch "Foo" {
case "jake":
    print("jake")    
case "mina":
    print("mina")
case "Foo":
    print("Foo")
default:
    print("unkown")
}
```

### Loop

```swift
// for range
for i in 0...3 {
    print("\(i)", terminator: " ")
}
// 0 1 2 3

print("")
for i in 0..<3 {
    print("\(i)", terminator: " ")
}
// 0 1 2

print("")
for i in (0...3).reversed() {
    print("\(i)", terminator: " ")
}
// 3 2 1 0

print("")
for i in (0..<3).reversed() {
    print("\(i)", terminator: " ")
}
// 2 1 0

// for in
var integers = [1, 2, 3]
let people = ["foo": 10, "bar": 15, "baz": 12]
for integer in integers {
    print(integer)
}
for (name, age) in people {
    print("\(name): \n(age)")
}

// while
while integers.count > 1 {
    integers.removeLast()
}

// repeat while
repeat {
    integers.removeLast()
} while integers.count > 0
```

## Collections Compared With C++ Container

[Collection Types](https://docs.swift.org/swift-book/LanguageGuide/CollectionTypes.html) 에 의하면 swift 의 collection 은 `Array, Set, Dictionary` 가 있다.

[swift-collections | github](https://github.com/apple/swift-collections) 는
`Deque<>, OrderedSet<>, OrderedDictionary<>` 를 제공한다.

| c++                  | swift                  |
|:---------------------|:----------------------|
| `if, else, switch`   | `if, else, switch`    |
| `for, while, do`     | `for, in, while, repeat` |
| `array`              | `let, Array`          |
| `vector`             | `var, Array`          |
| `deque`              | ``                    |
| `forward_list`       | ``                    |
| `list`               | ``                    |
| `stack`              | ``                    |
| `queue`              | ``                    |
| `priority_queue`     | ``                    |
| `set`                | ``                    |
| `multiset`           | ``                    |
| `map`                | ``                    |
| `multimap`           | ``                    |
| `unordered_set`      | `Set`                 |
| `unordered_multiset` | ``        |
| `unordered_map`      | `Dictionary`          |
| `unordered_multimap` | ``                    |

## Collections

### Array

> [Documentation | Swift | Array](https://developer.apple.com/documentation/swift/array)

```swift
var i: Array<Int> = Array<Int>()
// var i : Array<Int> = [Int]()
// var i : Array<Int> = []
// var i : [Int] = Arry<int>()
// var i : [Int] = [Int]()
// var i : [Int] = []
// var i = [Int]()

i.append(1)
i.append(100)
// i.append(101.1) // Error
print(integers)        // [1, 100]
print(integers.max())  // 100
print(integers.last)   // 100
print(integers.lastIndex(of: 100)) // 1
print(i.contains(100)) // true
print(i.contains(90))  // false

i[0] = 99   // [99, 100]
i.remove(at: 0)  // [100]
i.removeLast()   // []
i.removeLast(1)  // []
i.removeAll()    // []
print(i.count)   // 0
// i[0] // Error

let ii = [1, 2, 3]
// ii.append(4) // Error
```

### Set

```swift
var s: Set<Int> = Set<Int>()
s.insert(1)
s.insert(100)
s.insert(99)
s.insert(99)

print(s) // [100, 99, 1]
print(s.contains(1)) // true
print(s.contains(2)) // false

s.remove(100)
s.removeFirst()
print(s.count) // 1

let s0: Set<Int> = [1, 2, 3, 4, 5]
let s1: Set<Int> = [3, 4, 5, 6, 7]

let union: Set<Int> = s0.union(s1)
print(union)

let sortedUnion: [Int] = union.sorted()
print(sortedUnion) // [1, 2, 3, 4, 5, 6, 7]

let intersection: Set<Int> = s0.intersection(s1)
print(intersection) // [5, 3, 4]

let subtraction: Set<Int> = s0.subtracting(s1)
print(subtraction) // [2, 1]
```

### Dictionary

```swift
var d: Dictionary<String, Any> = [String: Any]()
// var d: Dictionary<String, Any> = Dictionary<String, Any>()
// var d: Dictionary<String, Any> = [:]
// var d: Dictionary[String: Any] = Dictionary<String, Any>()
// var d: Dictionary[String: Any] = [String: Any]()
// var d: Dictionary[String: Any] = [:]
// var d = [String: Any]()

d["someKey"] = "value"
d["anotherKey"] = 100
print(d) // ["someKey": "value", "anotherKey": 100]

d["someKey"] = "dictionary"
print(d) // ["dictionary": "value", "anotherKey": 100]

d.removeValue(forKey: "anotherKey")
d["someKey"] = nil
print(d)

let d0: [String: String] = [:]
let d1: [String: String] = ["name": "foo", "gender": "male"]
// d0["key"] = "value" // error
// let somevalue: String = d1["name"] // error
```

## Collection Conversions

```swift
// Array to Set
let s = Set(["a", "b", "c"])
print(s)  // ["a", "c", "b"]

// Set to Array
let a = Array(s)
print(a)  // ["a", "c", "b"]
```

## Sort

* `sort()` is in-place sorting.
* `sroted()` returns sroted array copy.

```swift
var a = [1, 5, 3, 9, 7]

// Sort ASC
a.sort()
print(a) // [1, 3, 5, 7, 9]
// Sort DESC
a.sort(by: >)
print(a) // [9, 7, 5, 3, 1]
print(a.sort()) // []

// Sorted ASC
print(a.sorted()) // [1, 3, 5, 7, 9]
// Sort DESC
print(a.sorted(by: >)) // [9, 7, 5, 3, 1]
```

## Search

Array 에 binary search 없는 거임?

* [BinarySearch | github](https://github.com/mkeiser/BinarySearch)

## Multi Dimensional Array

> * [Multidimensional Array Example in Swift](https://www.appsdeveloperblog.com/multidimensional-array-example-in-swift/)

```swift
// one-dim array
var a = [String]()
var b = ["a", "b", "c"]

// mult-dim array
var c = [[String]]()
var d = [
  ["a", "b", "c"],
  ["d", "e", "f"]
]
print(a)
print(b)
print(c)
print(d)
```

## Optional

```swift
//> Declaring and using optionals:
// Declare an optional String
var optionalStr: String? = "Hello"
// Set it to nil (absence of a value)
optionalStr = nil

//> Forced Unwrapping: 
// To access the value inside an optional, you can use forced
// unwrapping by appending an exclamation mark (!). 
// However, it can cause a runtime crash if the optional contains nil.
var optionalValue: Int? = 42
// Forced unwrapping (dangerous, may crash if optionalValue is nil)
let value = optionalValue!

//> Optional Binding: 
// A safer way to unwrap optionals using the if let or 
// guard let constructs.
var optionalValue: Int? = 42
// Using if let (safe, only runs the block if optionalValue is not nil)
if let value = optionalValue {
    print("Unwrapped value is \(value)")
} else {
    print("optionalValue is nil")
}
// Using guard let (ensures value is available within the block,
// or exits block early if not)
func printValue() {
    guard let value = optionalValue else {
        print("optionalValue is nil")
        return
    }
    print("Unwrapped value is \(value)")
}

//> Optional Chaining: 
// Allows you to call properties, methods, or subscripts on optionals
// that might be nil. It returns an optional and short-circuits 
// when an optional is nil.
class Person {
    var name: String?

    init(name: String?) {
        self.name = name
    }

    func greet() {
        print("Hello, \(name ?? "unknown")!")
    }
}
let person: Person? = Person(name: "Alice")
// Optional chaining (greet() is called only if person is not nil)
person?.greet()

//> Nil Coalescing Operator (??): 
// Provides a default value to use when the optional 
// on the left is nil.
let optionalValue: Int? = nil
let defaultValue = 42
// Nil coalescing (returns defaultValue if optionalValue is nil)
let value = optionalValue ?? defaultValue // value is now 42
```

## Struct

**Structures** (or structs) in Swift are value types that define a custom data
structure. They can have properties, methods, and initializers.

```swift
// Declare a struct
struct Point {
    var x: Double
    var y: Double
    
    func description() -> String {
        return "(\(x), \(y))"
    }
}

// Create an instance of the Point struct
var pointA = Point(x: 3.0, y: 4.0)

// Access its properties
print("pointA x: \(pointA.x), y: \(pointA.y)") // Output: pointA x: 3.0, y: 4.0

// Call method on the instance
print("pointA description: \(pointA.description())") // Output: pointA description: (3.0, 4.0)

// Assign pointA to another variable (creates a copy)
var pointB = pointA
pointB.x = 5.0
pointB.y = 6.0

// Both pointA and pointB refer to different copies
print("pointA: \(pointA.description()), pointB: \(pointB.description())") // Output: pointA: (3.0, 4.0), pointB: (5.0, 6.0)
```

## Class

Classes in Swift are reference types that define a custom data structure. They
can have properties, methods, initializers, and support inheritance,
sub-classing, and casting.

```swift
// Declare a class
class Circle {
    var radius: Double
    
    init(radius: Double) {
        self.radius = radius
    }
    
    func area() -> Double {
        return Double.pi * radius * radius
    }
    
    func description() -> String {
        return "Circle with radius: \(radius)"
    }
}

// Create an instance of the Circle class
let circleA = Circle(radius: 4.0)

// Access its properties
print("circleA radius: \(circleA.radius)") // Output: circleA radius: 4.0

// Call methods on the instance
print("circleA area: \(circleA.area())") // Output: circleA area: 50.26548245743669
print("circleA description: \(circleA.description())") // Output: circleA description: Circle with radius: 4.0

// Assign circleA to another variable (creates a reference to the same instance)
let circleB = circleA
circleB.radius = 5.0

// Both circleA and circleB refer to the same instance
print("circleA: \(circleA.description()), circleB: \(circleB.description())") // Output: circleA: Circle with radius: 5.0, circleB: Circle with radius: 5.0
```

## Enumerations

Enumerations (or enums) in Swift are used to define a common type for a group of
related values, allowing you to work with those values in a type-safe and
expressive way.

```swift
// Declare an enumeration for compass directions
enum CompassDirection {
    case north
    case south
    case east
    case west
}

// Create a variable using the enum type
var currentDirection: CompassDirection = .north

// Switch between enum cases using a switch statement
func printDirection(direction: CompassDirection) {
    switch direction {
    case .north:
        print("Heading north")
    case .south:
        print("Heading south")
    case .east:
        print("Heading east")
    case .west:
        print("Heading west")
    }
}

printDirection(direction: currentDirection) // Output: Heading north

currentDirection = .south
printDirection(direction: currentDirection) // Output: Heading south
```

## Value Type vs Reference Type

```swift
struct ValueType {
    var property = 1
}
class ReferenceType {
    var property = 1
}
let firstStructInstance = ValueType()
var secondStructInstance = firstStructInstance
secondStructInstance.property = 2

print("first struct instance property : \(firstStructInstance.property)")    // 1
print("second struct instance property : \(secondStructInstance.property)")  // 2

let firstClassReference = ReferenceType()
let secondClassReference = firstClassReference
secondClassReference.property = 2
print("first class reference property : \(firstClassReference.property)")    // 2
print("second class reference property : \(secondClassReference.property)")  // 2
```

## Closures

클로저는 코드의 블럭이다. 일급 시민 (First-Citizen) 으로 전달인자, 변수, 상수
등으로 저장, 전달이 가능하다. 함수는 클로저의 일종이다. 이름이 있는 클로저라고
생각하자.

```swift
import Swift

func sumFunction(a: Int, b: Int) -> Int {
    return a + b
}

var sumResult: Int = sumFunction(a: 1, b: 2)

print(sumResult) // 3

var sum: (Int, Int) -> Int = { (a: Int, b: Int) -> Int in
    return a + b
}

sumResult = sum(1, 2)
print(sumResult) // 3
sum = sumFunction(a:b:)

sumResult = sum(1, 2)
print(sumResult) // 3

let add: (Int, Int) -> Int
add = { (a: Int, b: Int) -> Int in
    return a + b
}

let substract: (Int, Int) -> Int
substract = { (a: Int, b: Int) -> Int in
    return a - b
}

let divide: (Int, Int) -> Int
divide = { (a: Int, b: Int) -> Int in
    return a / b
}

func calculate(a: Int, b: Int, method: (Int, Int) -> Int) -> Int {
    return method(a, b)
}

var calculated: Int

calculated = calculate(a: 50, b: 10, method: add)

print(calculated) // 60
calculated = calculate(a: 50, b: 10, method: substract)

print(calculated) // 40
calculated = calculate(a: 50, b: 10, method: divide)

print(calculated) // 5
calculated = calculate(a: 50, b: 10, method: { (left: Int, right: Int) -> Int in
    return left * right
})

print(calculated) // 500

var result: Int

result = calculate(a: 10, b: 10) { (left: Int, right: Int) -> Int in
    return left + right
}

print(result) // 20

result = calculate(a: 10, b: 10, method: { (left: Int, right: Int) in
    return left + right
})

print(result) // 20
result = calculate(a: 10, b: 10) { (left: Int, right: Int) in
    return left + right
}

result = calculate(a: 10, b: 10, method: {
    return $0 + $1
})

print(result) // 20
result = calculate(a: 10, b: 10) {
    return $0 + $1
}

print(result) // 20

result = calculate(a: 10, b: 10) {
    $0 + $1
}

print(result) // 20
result = calculate(a: 10, b: 10) { $0 + $1 }

print(result) // 20
result = calculate(a: 10, b: 10, method: { (left: Int, right: Int) -> Int in
    return left + right
})

result = calculate(a: 10, b: 10) { $0 + $1 }

print(result) // 20
```

## Properties

```swift
struct Student {
    var name: String = ""
    var `class`: String = "Swift"
    var koreanAge: Int = 0
    
    var westernAge: Int {
        get {
            return koreanAge - 1
        }
        
        set(inputValue) {
            koreanAge = inputValue + 1
        }
    }
    
    static var typeDescription: String = "학생"
    
    /*
    func selfIntroduce() {
        print("저는 \(self.class)반 \(name)입니다")
    }
     */
    
    var selfIntroduction: String {
        get {
            return "저는 \(self.class)반 \(name)입니다"
        }
    }
        
    /*
     static func selfIntroduce() {
     print("학생타입입니다")
     }
     */

    // read only type property    
    static var selfIntroduction: String {
        return "학생타입입니다"
    }
}

print(Student.selfIntroduction)
// 학생타입입니다

var foo: Student = Student()
foo.koreanAge = 10

foo.name = "foo"
print(foo.name)
// foo

print(foo.selfIntroduction)
// 저는 Swift반 foo입니다

print("제 한국나이는 \(foo.koreanAge)살이고, 미쿡나이는 \(foo.westernAge)살입니다.")
// 제 한국나이는 10살이고, 미쿡나이는 9살입니다.

struct Money {
    var currencyRate: Double = 1100
    var dollar: Double = 0
    var won: Double {
        get {
            return dollar * currencyRate
        }
        set {
            dollar = newValue / currencyRate
        }
    }
}

var moneyInMyPocket = Money()
moneyInMyPocket.won = 11000
print(moneyInMyPocket.won)
// 11000.0

moneyInMyPocket.dollar = 10
print(moneyInMyPocket.won)
// 11000.0
```
```swift
struct Money {
    var currencyRate: Double = 1100 {
        willSet(newRate) {
            print("환율이 \(currencyRate)에서 \(newRate)으로 변경될 예정입니다")
        }
        
        didSet(oldRate) {
            print("환율이 \(oldRate)에서 \(currencyRate)으로 변경되었습니다")
        }
    }

    var dollar: Double = 0 {
        willSet {
            print("\(dollar)달러에서 \(newValue)달러로 변경될 예정입니다")
        }
        
        didSet {
            print("\(oldValue)달러에서 \(dollar)달러로 변경되었습니다")
        }
    }

    var won: Double {
        get {
            return dollar * currencyRate
        }
        set {
            dollar = newValue / currencyRate
        }
    }    
}

var moneyInMyPocket: Money = Money()

// 환율이 1100.0에서 1150.0으로 변경될 예정입니다
moneyInMyPocket.currencyRate = 1150
// 환율이 1100.0에서 1150.0으로 변경되었습니다

// 0.0달러에서 10.0달러로 변경될 예정입니다
moneyInMyPocket.dollar = 10
// 0.0달러에서 10.0달러로 변경되었습니다

print(moneyInMyPocket.won)
// 11500.0
```

## Methods

> * [Methods](https://docs.swift.org/swift-book/LanguageGuide/Methods.html)
>   * [메서드](https://bbiguduk.gitbook.io/swift/language-guide-1/methods)

```swift
// Instant methods
class Counter {
    var count = 0
    func increment() {
        count += 1
    }
    func increment(by amount: Int) {
        count += amount
    }
    func reset() {
        count = 0
    }
}

// Type methods
class SomeClass {
    class func someTypeMethod() {
        // type method implementation goes here
    }
}
SomeClass.someTypeMethod()
```

## Subscripts

> * [Subscripts](https://docs.swift.org/swift-book/LanguageGuide/Subscripts.html)
>   * [서브스크립트](https://bbiguduk.gitbook.io/swift/language-guide-1/subscripts)

**Classes**, **structures**, and **enumerations** can define subscripts, which are shortcuts for accessing the member elements of a collection, list, or sequence.

```swift
struct TimesTable {
    let multiplier: Int
    subscript(index: Int) -> Int {
        return multiplier * index
    }
}
let threeTimesTable = TimesTable(multiplier: 3)
print("six times three is \(threeTimesTable[6])")
// Prints "six times three is 18"
```

## Inheritance

```swift
class Person {
    var name: String = ""
    
    func selfIntroduce() {
        print("저는 \(name)입니다")
    }
    
    final func sayHello() {
        print("hello")
    }
    
    static func typeMethod() {
        print("type method - static")
    }
    
    class func classMethod() {
        print("type method - class")
    }
    
    final class func finalCalssMethod() {
        print("type method - final class")
    }
}

class Student: Person {
    var major: String = ""
    
    override func selfIntroduce() {
        print("저는 \(name)이고, 전공은 \(major)입니다")
    }
    
    override class func classMethod() {
        print("overriden type method - class")
    }
    
//    override static func typeMethod() {    }
    
//    override func sayHello() {    }
//    override class func finalClassMethod() {    }

}
```

```swift
let foo: Person = Person()
let hana: Student = Student()

foo.name = "foo"
hana.name = "hana"
hana.major = "Swift"

foo.selfIntroduce()
// 저는 foo입니다

hana.selfIntroduce()
// 저는 hana이고, 전공은 Swift입니다

Person.classMethod()
// type method - class

Person.typeMethod()
// type method - static

Person.finalCalssMethod()
// type method - final class


Student.classMethod()
// overriden type method - class

Student.typeMethod()
// type method - static

Student.finalCalssMethod()
// type method - final class
```

## Initialization, Deinitialization

```swift
class PersonA {
    var name: String = "unknown"
    var age: Int = 0
    var nickName: String = "nick"
}

let jason: PersonA = PersonA()

jason.name = "jason"
jason.age = 30
jason.nickName = "j"

class PersonB {
    var name: String
    var age: Int
    var nickName: String
    
    // initializer
    init(name: String, age: Int, nickName: String) {
        self.name = name
        self.age = age
        self.nickName = nickName
    }
}

let hana: PersonB = PersonB(name: "hana", age: 20, nickName: "하나")

class PersonC {
    var name: String
    var age: Int
    var nickName: String?
    
    init(name: String, age: Int, nickName: String) {
        self.name = name
        self.age = age
        self.nickName = nickName
    }
    
    init(name: String, age: Int) {
        self.name = name
        self.age = age
    }
}

let jenny: PersonC = PersonC(name: "jenny", age: 10)
let mike: PersonC = PersonC(name: "mike", age: 15, nickName: "m")

class Puppy {
    var name: String
    var owner: PersonC!
    
    init(name: String) {
        self.name = name
    }
    
    func goOut() {
        print("\(name)가 주인 \(owner.name)와 산책을 합니다")
    }
}

let happy: Puppy = Puppy(name: "happy")
//happy.goOut() // error
happy.owner = jenny
happy.goOut()
// happy가 주인 jenny와 산책을 합니다

class PersonD {
    var name: String
    var age: Int
    var nickName: String?
    
    init?(name: String, age: Int) {
        if (0...120).contains(age) == false {
            return nil
        }
        
        if name.characters.count == 0 {
            return nil
        }
        
        self.name = name
        self.age = age
    }
}

//let john: PersonD = PersonD(name: "john", age: 23)
let john: PersonD? = PersonD(name: "john", age: 23)
let joker: PersonD? = PersonD(name: "joker", age: 123)
let batman: PersonD? = PersonD(name: "", age: 10)

print(joker) // nil
print(batman) // nil

class PersonE {
    var name: String
    var pet: Puppy?
    var child: PersonC
    
    init(name: String, child: PersonC) {
        self.name = name
        self.child = child
    }
    
    deinit {
        if let petName = pet?.name {
            print("\(name)가 \(child.name)에게 \(petName)를 인도합니다")
            self.pet?.owner = child
        }
    }
}

var donald: PersonE? = PersonE(name: "donald", child: jenny)
donald?.pet = happy
donald = nil // donald 인스턴스가 더이상 필요없으므로 메모리에서 해제됩니다
// donald가 jenny에게 happy를 인도합니다
```

## Optional Chaining

```swift
class Person {
    var name: String
    var job: String?
    var home: Apartment?
    
    init(name: String) {
        self.name = name
    }
}

class Apartment {
    var buildingNumber: String
    var roomNumber: String
    var `guard`: Person?
    var owner: Person?
    
    init(dong: String, ho: String) {
        buildingNumber = dong
        roomNumber = ho
    }
}

let foo: Person? = Person(name: "foo")
let apart: Apartment? = Apartment(dong: "101", ho: "202")
let superman: Person? = Person(name: "superman")

// 옵셔널 체이닝을 사용하지 않는다면...
func guardJob(owner: Person?) {
    if let owner = owner {
        if let home = owner.home {
            if let `guard` = home.guard {
                if let guardJob = `guard`.job {
                    print("우리집 경비원의 직업은 \(guardJob)입니다")
                } else {
                    print("우리집 경비원은 직업이 없어요")
                }
            }
        }
    }
}

guardJob(owner: foo)

// 옵셔널 체이닝을 사용한다면
func guardJobWithOptionalChaining(owner: Person?) {
    if let guardJob = owner?.home?.guard?.job {
        print("우리집 경비원의 직업은 \(guardJob)입니다")
    } else {
        print("우리집 경비원은 직업이 없어요")
    }
}

guardJobWithOptionalChaining(owner: foo)
// 우리집 경비원은 직업이 없어요

foo?.home?.guard?.job // nil
foo?.home = apart
foo?.home // Optional(Apartment)
foo?.home?.guard // nil
foo?.home?.guard = superman
foo?.home?.guard // Optional(Person)
foo?.home?.guard?.name // superman
foo?.home?.guard?.job // nil
foo?.home?.guard?.job = "경비원"


// 옵셔널 값이 nil일 경우, 우측의 값을 반환합니다. 띄어쓰기에 주의하여야 합니다.
var guardJob: String
    
guardJob = foo?.home?.guard?.job ?? "슈퍼맨"
print(guardJob) // 경비원

foo?.home?.guard?.job = nil

guardJob = foo?.home?.guard?.job ?? "슈퍼맨"
print(guardJob) // 슈퍼맨
```

## Error Handling

```swift

import Swift

//MARK: - 오류표현
//Error 프로토콜과 (주로) 열거형을 통해서 오류를 표현합니다
/*
enum <#오류종류이름#>: Error {
    case <#종류1#>
    case <#종류2#>
    case <#종류3#>
    //...
}
*/


// 자판기 동작 오류의 종류를 표현한 VendingMachineError 열거형
enum VendingMachineError: Error {
    case invalidInput
    case insufficientFunds(moneyNeeded: Int)
    case outOfStock
}

//MARK:- 함수에서 발생한 오류 던지기
// 자판기 동작 도중 발생한 오류 던지기
// 오류 발생의 여지가 있는 메서드는 throws를 사용하여
// 오류를 내포하는 함수임을 표시합니다
class VendingMachine {
    let itemPrice: Int = 100
    var itemCount: Int = 5
    var deposited: Int = 0
    
    // 돈 받기 메서드
    func receiveMoney(_ money: Int) throws {
        
        // 입력한 돈이 0이하면 오류를 던집니다
        guard money > 0 else {
            throw VendingMachineError.invalidInput
        }
        
        // 오류가 없으면 정상처리를 합니다
        self.deposited += money
        print("\(money)원 받음")
    }
    
    // 물건 팔기 메서드
    func vend(numberOfItems numberOfItemsToVend: Int) throws -> String {
        
        // 원하는 아이템의 수량이 잘못 입력되었으면 오류를 던집니다
        guard numberOfItemsToVend > 0 else {
            throw VendingMachineError.invalidInput
        }
        
        // 구매하려는 수량보다 미리 넣어둔 돈이 적으면 오류를 던집니다
        guard numberOfItemsToVend * itemPrice <= deposited else {
            let moneyNeeded: Int
            moneyNeeded = numberOfItemsToVend * itemPrice - deposited
            
            throw VendingMachineError.insufficientFunds(moneyNeeded: moneyNeeded)
        }
        
        // 구매하려는 수량보다 요구하는 수량이 많으면 오류를 던집니다
        guard itemCount >= numberOfItemsToVend else {
            throw VendingMachineError.outOfStock
        }
        
        // 오류가 없으면 정상처리를 합니다
        let totalPrice = numberOfItemsToVend * itemPrice
        
        self.deposited -= totalPrice
        self.itemCount -= numberOfItemsToVend
        
        return "\(numberOfItemsToVend)개 제공함"
    }
}

// 자판기 인스턴스
let machine: VendingMachine = VendingMachine()

// 판매 결과를 전달받을 변수
var result: String?


//MARK:- 오류처리
//오류발생의 여지가 있는 throws 함수(메서드)는
//try를 사용하여 호출해야합니다
//try, try?, try!

//MARK: do-catch
//오류발생의 여지가 있는 throws 함수(메서드)는
//do-catch 구문을 활용하여
//오류발생에 대비합니다
// 가장 정석적인 방법으로 모든 오류 케이스에 대응합니다
do {
    try machine.receiveMoney(0)
} catch VendingMachineError.invalidInput {
    print("입력이 잘못되었습니다")
} catch VendingMachineError.insufficientFunds(let moneyNeeded) {
    print("\(moneyNeeded)원이 부족합니다")
} catch VendingMachineError.outOfStock {
    print("수량이 부족합니다")
} // 입력이 잘못되었습니다

// 하나의 catch 블럭에서 switch 구문을 사용하여
// 오류를 분류해봅니다
// 굳이 위의 것과 크게 다를 것이 없습니다
do {
    try machine.receiveMoney(300)
} catch /*(let error)*/ {
    
    switch error {
    case VendingMachineError.invalidInput:
        print("입력이 잘못되었습니다")
    case VendingMachineError.insufficientFunds(let moneyNeeded):
        print("\(moneyNeeded)원이 부족합니다")
    case VendingMachineError.outOfStock:
        print("수량이 부족합니다")
    default:
        print("알수없는 오류 \(error)")
    }
} // 300원 받음

// 딱히 케이스별로 오류처리 할 필요가 없으면 
// catch 구문 내부를 간략화해도 무방합니다
do {
    result = try machine.vend(numberOfItems: 4)
} catch {
    print(error)
} // insufficientFunds(100)
// 딱히 케이스별로 오류처리 할 필요가 없으면 do 구문만 써도 무방합니다
do {
    result = try machine.vend(numberOfItems: 4)
}


//MARK: try? 와 try!
//try?
//별도의 오류처리 결과를 통보받지 않고
//오류가 발생했으면 결과값을 nil로 돌려받을 수 있습니다
//정상동작 후에는 옵셔널 타입으로 정상 반환값을 돌려 받습니다
result = try? machine.vend(numberOfItems: 2)
result // Optional("2개 제공함")
result = try? machine.vend(numberOfItems: 2)
result // nil
//try!
//오류가 발생하지 않을 것이라는 강력한 확신을 가질 때
//try!를 사용하면 정상동작 후에 바로 결과값을 돌려받습니다
//오류가 발생하면 런타임 오류가 발생하여 
//애플리케이션 동작이 중지됩니다
result = try! machine.vend(numberOfItems: 1)
result // 1개 제공함
//result = try! machine.vend(numberOfItems: 1)
// 런타임 오류 발생!

/*
더 알아보기 : rethrows, defer
*/
```

## Concurrency

There are several approaches to manage concurrency in Swift, including
**Dispatch Queues (Grand Central Dispatch)** and Swift 5.1 Concurrency model
that uses **async**, **await**, and **actors**. Swift 5.5 Concurrency model 
introduced **Task** and **TaskGroup**.

**Dispatch Queues (Grand Central Dispatch or GCD)**:

**GCD** is a low-level API for managing concurrent tasks. It uses dispatch queues to schedule tasks for execution.

* **Serial Dispatch Queue**: Executes tasks one at a time, in the order they were added.
* **Concurrent Dispatch Queue**: Executes tasks concurrently, without any specific order.

```swift
import Dispatch

// Create a serial dispatch queue
let serialQueue = DispatchQueue(label: "serial_queue")

// Add tasks to the serial dispatch queue
serialQueue.async {
    print("Task 1")
}

serialQueue.async {
    print("Task 2")
}

// Create a concurrent dispatch queue
let concurrentQueue = DispatchQueue(label: "concurrent_queue", attributes: .concurrent)

// Add tasks to the concurrent dispatch queue
concurrentQueue.async {
    print("Task A")
}

concurrentQueue.async {
    print("Task B")
}
```

Swift Concurrency (**async**, **await**, and **actors**):

Introduced in Swift 5.5

* **async function**: Functions marked async can run concurrently with other
  tasks.
* **await**: Used to call async functions, it suspends the current task until
  the async function completes.
* **actors**: A new type, similar to classes, that manage concurrent access to
  their state using async functions.

```swift
import Foundation

// Define an async function
func fetchData() async -> String {
    // Simulate a network request
    Thread.sleep(forTimeInterval: 2.0)
    return "Fetched Data"
}

// Call the async function using 'await'
async {
    let data = await fetchData()
    print("Data: \(data)")
}

// Define an actor
actor Counter {
    private(set) var count = 0
    
    func increment() {
        count += 1
    }
    
    func reset() {
        count = 0
    }
}

// Spawn multiple tasks using the 'Counter' actor
async {
    let counter = Counter()
    
    // Increment the 'Counter' actor concurrently
    async let task1 = Task { await counter.increment() }
    async let task2 = Task { await counter.increment() }
    
    // Wait for tasks to complete and print the result
    await task1.value
    await task2.value
    print("Counter: \(await counter.count)")
}
```

**Task**: A Task represents an asynchronous unit of work, and it provides a
mechanism to produce a result or complete with an error. You can also use tasks
for cancellation, prioritization, and suspension of tasks.

```swift
import Foundation

func fetchData() async -> String {
  // Simulate a network request
  Thread.sleep(forTimeInterval: 2.0)
  return "Fetched data"
}

async {
  let task = Task { () -> String in
    return await fetchData()
  }

  // Get the result of the task
  let result = await task.value
  print("Task result: \(result)")
}
```

**TaskGroup**: A TaskGroup is a group of tasks that can be handled together. You
can use the TaskGroup concept to perform a series of tasks with a common goal.
The TaskGroup lets you spawn tasks, await their completion, and collect their
results.

```swift
import Foundation

// Function to perform async tasks
func fetchData(index: Int) async -> Int {
  // Simulate a network request
  Thread.sleep(forTimeInterval: 0.5)
  return index
}

// Main async scope
async {
  do {
    let results: [Int] = try await withThrowingTaskGroup(of: Int.self) { group in
      for index in 1...5 {
        group.addTask {
          return await fetchData(index: index)
        }
      }
      
      // Collect the results
      var resultArray: [Int] = []
      for try await result in group {
        resultArray.append(result)
      }
      
      return resultArray
    }
    print("TaskGroup results: \(results)") // Output: TaskGroup results: [1, 2, 3, 4, 5]
  } catch {
    print("TaskGroup error: \(error)")
  }
}
```

## Macros

Macros transform your source code when you compile it, 
letting you avoid writing **repetitive code** by hand.


```swift
//> Freestanding Macros:
func myFunction() {
    print("Currently running \(#function)")
    // Currently running myFunction()

    #warning("Something's wrong")

}

//> Attached Macros:
// as-is
struct SundaeToppings: OptionSet {
    let rawValue: Int
    static let nuts = SundaeToppings(rawValue: 1 << 0)
    static let cherry = SundaeToppings(rawValue: 1 << 1)
    static let fudge = SundaeToppings(rawValue: 1 << 2)
}
// to-be
// @OptionSet is a macro from the Swift standard library.
@OptionSet<Int>
struct SundaeToppings {
    private enum Options: Int {
        case nuts
        case cherry
        case fudge
    }
}
// Expanded code
struct SundaeToppings {
    private enum Options: Int {
        case nuts
        case cherry
        case fudge
    }
    typealias RawValue = Int
    var rawValue: RawValue
    init() { self.rawValue = 0 }
    init(rawValue: RawValue) { self.rawValue = rawValue }
    static let nuts: Self = Self(rawValue: 1 << Options.nuts.rawValue)
    static let cherry: Self = Self(rawValue: 1 << Options.cherry.rawValue)
    static let fudge: Self = Self(rawValue: 1 << Options.fudge.rawValue)
}
extension SundaeToppings: OptionSet { }

//> Macro Declarations
public macro OptionSet<RawType>() =
        #externalMacro(module: "SwiftMacros", type: "OptionSetMacro")

//> Macro Expansion
let magicNumber = #fourCharacterCode("ABCD")
// Expanded code
let magicNumber = 1145258561

//> Implementing a Macro
// Pakcage.swift of Macro library
targets: [
    // Macro implementation that performs the source transformations.
    .macro(
        name: "MyProjectMacros",
        dependencies: [
            .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
            .product(name: "SwiftCompilerPlugin", package: "swift-syntax")
        ]
    ),
    // Library that exposes a macro as part of its API.
    .target(name: "MyProject", dependencies: ["MyProjectMacros"]),
]
// Package.swift of Host application
dependencies: [
    .package(url: "https://github.com/apple/swift-syntax.git", from: "some-tag"),
],
// FourCharacterCode macro implementation
public struct FourCharacterCode: ExpressionMacro {
    public static func expansion(
        of node: some FreestandingMacroExpansionSyntax,
        in context: some MacroExpansionContext
    ) throws -> ExprSyntax {
        guard let argument = node.argumentList.first?.expression,
              let segments = argument.as(StringLiteralExprSyntax.self)?.segments,
              segments.count == 1,
              case .stringSegment(let literalSegment)? = segments.first
        else {
            throw CustomError.message("Need a static string")
        }

        let string = literalSegment.content.text
        guard let result = fourCharacterCode(for: string) else {
            throw CustomError.message("Invalid four-character code")
        }


        return "\(raw: result)"
    }
}
private func fourCharacterCode(for characters: String) -> UInt32? {
    guard characters.count == 4 else { return nil }

    var result: UInt32 = 0
    for character in characters {
        result = result << 8
        guard let asciiValue = character.asciiValue else { return nil }
        result += UInt32(asciiValue)
    }
    return result.bigEndian
}
enum CustomError: Error { case message(String) }

//> Developing and Debugging Macros
// Test of the #fourCharacterCode macro
let source: SourceFileSyntax =
    """
    let abcd = #fourCharacterCode("ABCD")
    """
let file = BasicMacroExpansionContext.KnownSourceFile(
    moduleName: "MyModule",
    fullFilePath: "test.swift"
)
let context = BasicMacroExpansionContext(sourceFiles: [source: file])
let transformedSF = source.expand(
    macros:["fourCharacterCode": FourCC.self],
    in: context
)
let expectedDescription =
    """
    let abcd = 1145258561
    """
precondition(transformedSF.description == expectedDescription)
```

## Print Type

```swift
print(type(of: [1, 2, 3])) 
// Array<Int>
print(type(of: 0...3))
// ClosedRange<Int>
print(type(of: (0...3).reversed()))
// ReversedCollection<ClosedRange<Int>>
print(type(of: 0..<3))
// Range<Int>
print(type(of: (0..<3).reversed()))
// ReversedCollection<Range<Int>>
```

## Type Casting

> * [Type Casting](https://docs.swift.org/swift-book/LanguageGuide/TypeCasting.html)
>   * [타입 캐스팅](https://bbiguduk.gitbook.io/swift/language-guide-1/type-casting)

* `type(of:)`
  * 타입확인
* `is`
  * 타입비교
* `as`
  * 성공하면 Up-casting or Self-casting, 실패하면 Compile Error
* `as?`
  * 성공하면 Optional, 실패하면 nil
* `as!`
  * 성공하면 Unwrapped, 실패하면 Runtime Error

```swift
// Print type
print(type(of: [1, 2, 3])) // Array<Int>

// Compare type
print([1, 2, 3] is Array<Int>)   // true
print([1, 2, 3] is Array<Float>) // false

// Type casting
let a = [1, 2, 3]
print(a as Int)    // Compile Error
print(a as? Int)   // nil
print(a as! Int)   // Runtime Error
```

## Assert, Guard

**assert** and **guard** are used to handle edge cases during program execution.

**assert**: The `assert` function is used for validating assumptions about your
program and checking for potential runtime errors during the development phase.
If the given condition is true, the program continues execution normally.
Otherwise, a custom error message is displayed, and the program is halted.
Assertions are only active in debug builds and removed from production/release
builds.

```swift
func calculateArea(length: Int, width: Int) -> Int {
    // Asserting that length and width are positive numbers
    assert(length > 0, "Length must be a positive number.")
    assert(width > 0, "Width must be a positive number.")
    
    return length * width
}

let area = calculateArea(length: 5, width: 3)
print("Area: \(area)") // Output: Area: 15

// let invalidArea = calculateArea(length: -1, width: 4)
// Assertion failed: Length must be a positive number.
```

**guard**: The `guard` statement is used to handle early exits from a scope
(function, loop, or control block) based on a specified condition. It allows you
to check for specific conditions and exit early if they are not met, keeping
your code cleaner and more readable.

```swift
func greet(person: [String: String]) {
    // Verify if the "name" key is present in the dictionary
    guard let name = person["name"] else {
        print("Name is missing.")
        return
    }

    print("Hello, \(name)!")
    
    // Check for other optional keys
    guard let age = person["age"] else {
        print("Age is not provided.")
        return
    }
    
    print("You are \(age) years old.")
}

let personA = ["name": "Alice", "age": "32"]
greet(person: personA) // Output: Hello, Alice! You are 32 years old.

```

## Nested Types

Swift allows you to define nested types, which means you can nest supporting
enumerations, classes, and structures within the definition of the type they
support. This can be useful to encapsulate the functionality and data specific
to a certain type and maintain a clean structure in your code, improving
readability and organization.

```swift
// Outer structure which represents a Bank
struct Bank {
    
    // Nested structure which represents Bank Account
    struct Account {
        var balance: Double
        
        // Nested enumeration which represents Account Type
        enum AccountType {
            case checking, savings
        }
        
        let type: AccountType
        
        // Function to deposit money into the account.
        mutating func deposit(amount: Double) {
            balance += amount
        }
        
        // Function to withdraw money from the account.
        mutating func withdraw(amount: Double) {
            if amount <= balance {
                balance -= amount
            } else {
                print("Insufficient balance.")
            }
        }
    }
    
    private var accounts: [Account] = [] // Array to store accounts
    
    // Function to create a new account.
    mutating func createAccount(accountType: Account.AccountType) {
        let newAccount = Account(balance: 0, type: accountType)
        accounts.append(newAccount)
    }
    
    // Function to get the balance of a specific account.
    func getBalance(accountIndex: Int) -> Double? {
        if accountIndex < 0 || accountIndex >= accounts.count {
            print("Invalid account index.")
            return nil
        }
        
        return accounts[accountIndex].balance
    }
}
```

we have an outer structure `Bank`, with a nested structure `Account` inside it.
The `Account` structure also has a nested enum `AccountType` that contains the
types of accounts (checking and savings). We also have some functions to create
accounts, deposit, withdraw, and get the balance of a specific account.

## Protocol

[Kotlin](/kotlin/README.md) 의 `interface` 와 같다.

```swift
/* 프로토콜 */

//프로토콜은 특정 역할을 수행하기 위한 
//메서드, 프로퍼티, 이니셜라이저 등의 요구사항을 정의합니다.
//구조체, 클래스, 열거형은 프로토콜을 채택(Adopted)해서
//프로토콜의 요구사항을 실제로 구현할 수 있습니다. 
//어떤 프로토콜의 요구사항을 모두 따르는 타입은 
//그 ‘프로토콜을 준수한다(Conform)’고 표현합니다. 
//프로토콜의 요구사항을 충족시키려면 프로토콜이 제시하는 기능을 
//모두 구현해야 합니다.
import Swift

//MARK: - 정의 문법
/*
protocol <#프로토콜 이름#> {
    /* 정의부 */
}
 */

protocol Talkable {
    
    // 프로퍼티 요구
    // 프로퍼티 요구는 항상 var 키워드를 사용합니다
    // get은 읽기만 가능해도 상관 없다는 뜻이며
    // get과 set을 모두 명시하면 
    // 읽기 쓰기 모두 가능한 프로퍼티여야 합니다
    var topic: String { get set }
    var language: String { get }
    
    // 메서드 요구
    func talk()
    
    // 이니셜라이저 요구
    init(topic: String, language: String)
}

//MARK: - 프로토콜 채택 및 준수
// Person 구조체는 Talkable 프로토콜을 채택했습니다
struct Person: Talkable {
    // 프로퍼티 요구 준수
    var topic: String
    let language: String
    
    // 메서드 요구 준수
    func talk() {
        print("\(topic)에 대해 \(language)로 말합니다")
    }
    
    // 이니셜라이저 요구 준수
    init(topic: String, language: String) {
        self.topic = topic
        self.language = language
    }
}


// MARK: - 프로토콜 상속
// 프로토콜은 클래스와 다르게 다중상속이 가능합니다
/*
 protocol <#프로토콜 이름#>: <#부모 프로토콜 이름 목록#> {
 /* 정의부 */
 }
 */

protocol Readable {
    func read()
}
protocol Writeable {
    func write()
}
protocol ReadSpeakable: Readable {
//    func read()
    func speak()
}
protocol ReadWriteSpeakable: Readable, Writeable {
//    func read()
//    func write()
    func speak()
}

struct SomeType: ReadWriteSpeakable {
    func read() {
        print("Read")
    }
    
    func write() {
        print("Write")
    }
    
    func speak() {
        print("Speak")
    }
}

//MARK: 클래스 상속과 프로토콜
// 클래스에서 상속과 프로토콜 채택을 동시에 하려면 
// 상속받으려는 클래스를 먼저 명시하고
// 그 뒤에 채택할 프로토콜 목록을 작성합니다
class SuperClass: Readable {
    func read() { }
}

class SubClass: SuperClass, Writeable, ReadSpeakable {
    func write() { }
    func speak() { }
}

//MARK:- 프로토콜 준수 확인
// 인스턴스가 특정 프로토콜을 준수하는지 확인할 수 있습니다
// is, as 연산자 사용
let sup: SuperClass = SuperClass()
let sub: SubClass = SubClass()

var someAny: Any = sup
someAny is Readable // true
someAny is ReadSpeakable // false
someAny = sub

someAny is Readable // true
someAny is ReadSpeakable // true
someAny = sup

if let someReadable: Readable = someAny as? Readable {
    someReadable.read()
} // read
if let someReadSpeakable: ReadSpeakable = someAny as? ReadSpeakable {
    someReadSpeakable.speak()
} // 동작하지 않음
someAny = sub

if let someReadable: Readable = someAny as? Readable {
    someReadable.read()
} // read
```

## Extension

[Kotlin](/kotlin/README.md) 의 `extension` 과 같다.

```swift
/* 익스텐션 */

//익스텐션은 구조체, 클래스, 열거형, 프로토콜 타입에 
//새로운 기능을 추가할 수 있는 기능입니다. 
//기능을 추가하려는 타입의 구현된 소스 코드를 
//알지 못하거나 볼 수 없다 해도, 
//타입만 알고 있다면 그 타입의 기능을 확장할 수도 있습니다.
//익스텐션으로 추가할 수 있는 기능
//연산 타입 프로퍼티 / 연산 인스턴스 프로퍼티
//타입 메서드 / 인스턴스 메서드
//이니셜라이저
//서브스크립트
//중첩 타입
//특정 프로토콜을 준수할 수 있도록 기능 추가
//기존에 존재하는 기능을 재정의할 수는 없습니다
import Swift

//MARK: - 정의 문법
/*
extension <#확장할 타입 이름#> {
    /* 타입에 추가될 새로운 기능 구현 */
}
 */

//익스텐션은 기존에 존재하는 타입이
//추가적으로 다른 프로토콜을 채택할 수 있도록 
//확장할 수도 있습니다.
/*
extension <#확장할 타입 이름#>: <#프로토콜1#>, <#프로토콜2#>, <#프로토콜3#>... {
    /* 프로토콜 요구사항 구현 */
}
 */

//MARK: - 익스텐션 구현
//MARK: 연산 프로퍼티 추가
extension Int {
    var isEven: Bool {
        return self % 2 == 0
    }
    var isOdd: Bool {
        return self % 2 == 1
    }
}

print(1.isEven) // false
print(2.isEven) // true
print(1.isOdd)  // true
print(2.isOdd)  // false
var number: Int = 3
print(number.isEven) // false
print(number.isOdd) // true
number = 2
print(number.isEven) // true
print(number.isOdd) // false


//MARK: 메서드 추가
extension Int {
    func multiply(by n: Int) -> Int {
        return self * n
    }
}
print(3.multiply(by: 2))  // 6
print(4.multiply(by: 5))  // 20
number = 3
print(number.multiply(by: 2))   // 6
print(number.multiply(by: 3))   // 9

//MARK: 이니셜라이저 추가
extension String {
    init(int: Int) {
        self = "\(int)"
    }
    
    init(double: Double) {
        self = "\(double)"
    }
}

let stringFromInt: String = String(int: 100)
// "100"
let stringFromDouble: String = String(double: 100.0)
// "100.0"
```

## Higher Order Function

```swift

import Swift

//전달인자로 함수를 전달받거나
//함수실행의 결과를 함수로 반환하는 함수
//스위프트 표준라이브러리에서 제공하는
//유용한 고차함수에 대해 알아봅니다
//map, filter, reduce
//컨테이너 타입(Array, Set, Dictionary 등)에 구현되어 있습니다
//MARK:- map
//컨테이너 내부의 기존 데이터를 변형(transform)하여 새로운 컨테이너 생성
let numbers: [Int] = [0, 1, 2, 3, 4]
var doubledNumbers: [Int]
var strings: [String]

// for 구문 사용
doubledNumbers = [Int]()
strings = [String]()

for number in numbers {
    doubledNumbers.append(number * 2)
    strings.append("\(number)")
}

print(doubledNumbers) // [0, 2, 4, 6, 8]
print(strings) // ["0", "1", "2", "3", "4"]
// map 메서드 사용
// numbers의 각 요소를 2배하여 새로운 배열 반환
doubledNumbers = numbers.map({ (number: Int) -> Int in
    return number * 2
})

// numbers의 각 요소를 문자열로 변환하여 새로운 배열 반환
strings = numbers.map({ (number: Int) -> String in
    return "\(number)"
})

print(doubledNumbers) // [0, 2, 4, 6, 8]
print(strings) // ["0", "1", "2", "3", "4"]
// 매개변수, 반환 타입, 반환 키워드(return) 생략, 후행 클로저
doubledNumbers = numbers.map { $0 * 2 }
print(doubledNumbers) // [0, 2, 4, 6, 8]

//MARK:- filter
//컨테이너 내부의 값을 걸러서 새로운 컨테이너로 추출
// for 구문 사용
// 변수 사용에 주목하세요
var filtered: [Int] = [Int]()

for number in numbers {
    if number % 2 == 0 {
        filtered.append(number)
    }
}

print(filtered) // [0, 2, 4]
// filter 메서드 사용
// numbers의 요소 중 짝수를 걸러내어 새로운 배열로 반환
let evenNumbers: [Int] = numbers.filter { (number: Int) -> Bool in
    return number % 2 == 0
}
print(evenNumbers) // [0, 2, 4]
// 매개변수, 반환 타입, 반환 키워드(return) 생략, 후행 클로저
let oddNumbers: [Int] = numbers.filter {
    $0 % 2 != 0
}
print(oddNumbers) // [1, 3]


//MARK:- reduce
// 컨테이너 내부의 콘텐츠를 하나로 통합
let someNumbers: [Int] = [2, 8, 15]

// for 구문 사용
// 변수 사용에 주목하세요
var result: Int = 0

// someNumbers의 모든 요소를 더합니다
for number in someNumbers {
    result += number
}

print(result) // 25

// reduce 메서드 사용
// 초깃값이 0이고 someNumbers 내부의 모든 값을 더합니다.
let sum: Int = someNumbers.reduce(0, { (first: Int, second: Int) -> Int in
    //print("\(first) + \(second)") //어떻게 동작하는지 확인해보세요
    return first + second
})

print(sum)  // 25
// 초깃값이 0이고 someNumbers 내부의 모든 값을 뺍니다.
var subtract: Int = someNumbers.reduce(0, { (first: Int, second: Int) -> Int in
    //print("\(first) - \(second)") //어떻게 동작하는지 확인해보세요
    return first - second
})

print(subtract) // -25
// 초깃값이 3이고 someNumbers 내부의 모든 값을 더합니다.
let sumFromThree = someNumbers.reduce(3) { $0 + $1 }

print(sumFromThree) // 28
/*
 더 알아보기 : flatMap
 */
```

## Core Libraries

> * [Swift Core Libraries](https://www.swift.org/core-libraries/)

Swift Standard Library 이외에 다음과 같은 3 가지 core library 를 학습해 두자. 

* [Foundation](https://developer.apple.com/documentation/foundation)
  * Provide primitive classes and introduces several paradigms that define functionality not provided by the language or runtime.
  * Provide formatted String.
* [libdispatch](https://github.com/apple/swift-corelibs-libdispatch)
  * Provide concurrency on multicore hardware.
* [XCTest](https://github.com/apple/swift-corelibs-xctest)
  * Provide unit test.

## Generics

Generic code enables you to write flexible, reusable functions and types that
can work with any type

Generic functions:

```swift
func swapValues<T>(_ a: inout T, _ b: inout T) {
    let temp = a
    a = b
    b = temp
}

var intA = 5
var intB = 10
swapValues(&intA, &intB)
print("intA: \(intA), intB: \(intB)") // Output: intA: 10, intB: 5

var strA = "Hello"
var strB = "World"
swapValues(&strA, &strB)
print("strA: \(strA), strB: \(strB)") // Output: strA: World, strB: Hello
```

`swapValues` function is a generic function that swaps the values of two
variables. The `<T>` syntax indicates it's a generic function with a type
placeholder `T`. The function operates on any type `T`, and it swaps the values
of variables a and b without casting or type-checking their types.

Generic types:

```swift
// Generic Stack using an Array
struct Stack<Element> {
    private var items: [Element] = []
    
    mutating func push(_ item: Element) {
        items.append(item)
    }
    
    mutating func pop() -> Element? {
        return items.isEmpty ? nil : items.removeLast()
    }
    
    func peek() -> Element? {
        return items.last
    }
    
    func isEmpty() -> Bool {
        return items.isEmpty
    }
}

var intStack = Stack<Int>()
intStack.push(1)
intStack.push(2)
print(intStack.pop()) // Output: Optional(2)

var strStack = Stack<String>()
strStack.push("Hello")
strStack.push("World")
print(strStack.peek()) // Output: Optional("World")
```

`Stack` is a generic type that represents a stack data structure. It has a type
placeholder `Element` that represents the data type of elements in the stack. When
we create an instance of the St`ack, we provide the actual data type (e.g., Int,
String) to be used in place of `Element`.

## Opaque and Boxed Types

Swift provides two ways to hide details about a value’s type: **opaque types**
and **boxed protocol types**. Both are related to dealing with protocols and
their associated types.

**Opaque types** are introduced in Swift 5.1 with the keyword some. They are used to
hide the underlying concrete type that conforms to a protocol. When you return
an opaque type, you are returning a type that conforms to a specific protocol,
but the actual type is not exposed to the caller. This is useful when you want
to hide implementation details and maintain type identity without revealing the
concrete type.

```swift
func createShape() -> some Shape {
    return Circle()
}
```

The caller of `createShape()` will know that it returns a type conforming to the
`Shape` protocol, but it won't know that the underlying concrete type is `Circle`.

**Boxed protocol types**, also known as existential types, are used when you
need to store or pass around heterogeneous instances of different types that
conform to the same protocol. When you use a protocol as a type, the Swift
compiler boxes the value into an existential container, allowing it to hold any
value that conforms to the protocol.

```swift
var shapes: [Shape] = [Circle(), Rectangle()]
```

The shapes array can store any object that conforms to the `Shape` protocol.
However, when using boxed protocol types, you lose the information about the
specific type of the object conforming to the protocol.

Key Differences:

**Opaque types** maintain type identity, while **boxed protocol types** lose type
identity. In other words, when using **opaque types**, the compiler has knowledge of
the specific concrete type at compile-time, whereas with **boxed protocol types**,
the specific type information is lost.

**Opaque types** provide better type safety and optimization because the Swift
compiler knows the concrete type during compilation, while **boxed protocol types**
may require dynamic dispatch and runtime checks for type compatibility, which
can result in decreased performance.

**Opaque types** can only be used as return types in functions, while **boxed protocol
types** can be used as return types, parameter types, and stored as properties.

**Opaque types** cannot be used with associated types that have **Self** or **associated
type** requirements, while **boxed protocol types** can handle these scenarios by
using a type-erased container, like an `Any-` wrapper, to deal with associated
types.

## Automatic Reference Counting

**ARC** manages memory by tracking class instance references increasing,
decreasing reference counts. 

**Weak** and **unowned references** prevent memory leaks.

```swift
//> ARC
// You should use `weak var` not just `var`.`
class Person {
    let name: String
    weak var bestFriend: Person?
    init(name: String) {
        self.name = name
    }    
    deinit {
        print("\(name) is being deallocated.")
    }
}
// Creating two Person instances
var personA: Person? = Person(name: "Alice")
var personB: Person? = Person(name: "Bob")
// Set their bestFriend property to each other (Weak reference avoids strong reference cycle)
personA?.bestFriend = personB
personB?.bestFriend = personA
// ARC will deallocate the instances when there are no strong references left
personA = nil // Output: "Alice is being deallocated."
personB = nil // Output: "Bob is being deallocated."

//> Weak references
// Declare weak reference
class ObjectA {
    weak var objectB: ObjectB?
}
// Usage
let objA = ObjectA()
var objB: ObjectB? = ObjectB()
objA.objectB = objB
objB = nil // Instance of ObjectB deallocated, objA.objectB becomes nil

//> Unowned references 
// Declare unowned reference
class ObjectA {
    unowned let objectB: ObjectB
    init(objB: ObjectB) { objectB = objB }
}
// Usage
let objB = ObjectB()
// Instance of ObjectB will outlive objA.objectB reference
let objA = ObjectA(objB: objB) 
```

## Memory Safety

There are three primary aspects of memory safety in Swift:

* Initialization before use
* Bounds checks on array access
* Exclusive access to memory

**Initialization before use**: Swift ensures that all properties of a value are
initialized before the value is used. It means that you cannot use an
uninitialized variable, preventing accidental access to garbage data.

```swift
var count: Int
// Using 'count' below without initializing will lead to a compile-time error.
// print(count) - Error: variable 'count' used before being initialized
count = 5
print(count)  // 5
```

**Bounds checks on array access**: Swift performs bounds checks on array access
to ensure you won't access memory outside the array's storage. If you attempt to
read or write to an array index that's out-of-bounds, Swift will trigger a
runtime error.

```swift
var numbers = [1, 2, 3]
// Accessing invalid index will result in a runtime error
// print(numbers[3]) - Fatal error: Index out of range
print(number[3]) // Fatal error: Index out of range
```

**Exclusive access to memory**: Swift prevents simultaneous access (read and
write) to the same memory location, avoiding racing conditions and unpredictable
behavior. This is enforced through compile-time checks and runtime checks.

```swift
// Example of conflict due to improper simultaneous access
func modify(value: inout Int) {
    value += 1
    value *= 2
}
var sharedValue = 3
modify(value: &sharedValue)
print(sharedValue) // Expected output: (3 + 1) * 2 = 8
```

## Access Control

Swift provides five access levels:

* **private**: Accessible only within the same source file where the entity is
  defined.
* **fileprivate**: Accessible only within the same source file.
* **internal**: Accessible within the entire module (by default, if no access
  modifier is specified).
* **public**: Accessible anywhere, but only sub-classable within the module. For
  properties and functions, it means they can be read but not overridden outside
  the module.
* **open**: Accessible and sub-classable from any module. For properties and
  functions, it means they can be read and overridden from any module.

```swift
// MyClass.swift
class MyClass {
    private var privateVar = "Only accessible within MyClass.swift"
    fileprivate var fileprivateVar = "Accessible within the same source file"    
    internal var internalVar = "Accessible within the entire module (default)"
    public var publicVar = "Readable anywhere, but not modifiable outside the module"    
    open var openVar = "Readable and modifiable from any module"
}

// AnotherSourceFile.swift
import SomeModule

let myClass = MyClass()
// myClass.privateVar - Error: 'privateVar' is inaccessible due to 'private' protection level
// Can access 'fileprivate' if in the same source file as MyClass
print(myClass.fileprivateVar)
// Accessible as they are within the same module
print(myClass.internalVar)
print(myClass.publicVar)
print(myClass.openVar)
```

## Advanced Operators

Bitwise, overflow, compound assignment, and other custom operators.

```swift
// Bitwise operators
let a: UInt8 = 0b1100
let b: UInt8 = 0b0110

let bitwiseNOT = ~a // 0b0011
let bitwiseAND = a & b // 0b0100
let bitwiseOR = a | b // 0b1110
let bitwiseXOR = a ^ b // 0b1010
let leftShift = a << 1 // 0b11000 (0b1000 after removing the overflowed bit)
let rightShift = a >> 1 // 0b0110

// Overflow operators
//   Overflow addition (&+): Adds two numbers and wraps the result on overflow.
//   Overflow subtraction (&-): Subtracts two numbers and wraps the result on overflow.
//   Overflow multiplication (&*): Multiplies two numbers and wraps the result on overflow.
let maxInt = UInt8.max
let overflowAdd = maxInt &+ 1 // 0 (wrapped overflow)
let overflowSubtract = UInt8.min &- 1 // 255 (wrapped overflow)

// Compound Assignment Operators
var a = 7
a += 2 // a = a + 2, now a is 9
a -= 3 // a = a - 3, now a is 6
a *= 4 // a = a * 4, now a is 24
a /= 2 // a = a / 2, now a is 12

// Custom Operators:
// Swift allows you to create custom operators, 
// which can either be prefix, infix, or postfix.
// Custom operators need to be declared first with 
// the operator keyword and associated with a special symbol.

// Declare a custom power operator
infix operator **
// Implementing the custom operator function
func **(base: Double, power: Double) -> Double {
    return pow(base, power)
}
// Using the custom power operator
let result = 2.0 ** 3.0 // 8.0
```

# Advanced

## Renaming Objective-C APIs for Swift

> * [Renaming Objective-C APIs for Swift](https://developer.apple.com/documentation/swift/renaming-objective-c-apis-for-swift)
> * [[Objective-C] Swift 에서 사용할 함수의 첫번째 인자 이름 지정하기](https://sujinnaljin.medium.com/objective-c-%ED%95%A8%EC%88%98%EC%9D%98-%EC%B2%AB%EB%B2%88%EC%A7%B8-%EC%9D%B8%EC%9E%90-%EC%9D%B4%EB%A6%84-%EC%A7%80%EC%A0%95%ED%95%98%EA%B8%B0-863aae7a3533)

----

Swift function 을 를 objc method 로 바꾸고 그 method 를 Swift 에서 호출해 보자.

```swift
// swift
func add(first a: Int, second b: Int) -> Int { 
   return a + b
}
add(first: 1, second: 2)

// objc
- (NSUInteger) add: (NSUInteger)a second: (NSInteger) b {
   return a + b;
}

// Call from swift
add(1, second: 2)
```

objc method 를 살펴보자. swift 에서 사용한 first label `first` 가 없다. objc
method 는 first label 을 생략하기 때문에 method name 을 다음과 같이 수정해야
한다.

```swift
// objc
- (NSUInteger) addToFirst: (NSUInteger)a second: (NSInteger) b {
   return a + b;
}

// Call from swift
add(toFirst: 1, second: 2)
```

그러나 objc method name 에 따라 일부가 사라지는 경우가 있다. `Color` 가 사라졌다. (테스트 못함)

```swift
// objc
- (void) makeBorderWithColor: (UIColor*) color width: (CGFloat) width;

// Call from swift
self.makeBorder(with: .red, width: 10)
```

`NS_SWIFT_NAME` macro 를 사용하면 해결된다.

```swift
// As-is
// objc 
- (void) makeRounded: (CGFloat) cornerRadius;

// Call from swift
myView.makeRounded(16)

// To-be
// objc
- (void) makeRounded: (CGFloat) cornerRadius 
NS_SWIFT_NAME(makeRounded(cornerRadius:));

// Call from swift
myView.makeRounded(cornerRadius: 16)
```

## Property Wrapper

* [Property Wrapper](https://zeddios.tistory.com/1221)

----

Property 의 Wrapper 를 만들 수 있다. 즉, Property 의 boiler plate code 를
재활용할 수 있다. 

```swift
// Define Uppercase property wrapper
@propertyWrapper
struct Uppercase {
    
    private var value: String = ""
    
    var wrappedValue: String {
        get { self.value }
        set { self.value = newValue.uppercased() }
    }
    
    init(wrappedValue initialValue: String) {
        self.wrappedValue = initialValue
    }
}

// Use it
struct Address {
    @Uppercase var town: String
}
let address = Address(town: "earth")
print(address.town)  // EARTH
```

```swift
// Define UserDefault<T> property wrapper
@propertyWrapper
struct UserDefault<T> {
    
    let key: String
    let defaultValue: T
    let storage: UserDefaults

    var wrappedValue: T {
        get { self.storage.object(forKey: self.key) as? T ?? self.defaultValue }
        set { self.storage.set(newValue, forKey: self.key) }
    }
    
    init(key: String, defaultValue: T, storage: UserDefaults = .standard) {
        self.key = key
        self.defaultValue = defaultValue
        self.storage = storage
    }
}

// Use it
class UserManager {
    
    @UserDefault(key: "usesTouchID", defaultValue: false)
    static var usesTouchID: Bool
    
    @UserDefault(key: "myEmail", defaultValue: nil)
    static var myEmail: String?
    
    @UserDefault(key: "isLoggedIn", defaultValue: false)
    static var isLoggedIn: Bool
}
```

`@State` 는 대표적인 Property Wrapper 중 하나이다.

```swift
@available(iOS 13.0, macOS 10.15, tvOS 13.0, watchOS 6.0, *)
@frozen @propertyWrapper public struct State<Value> : DynamicProperty {
```

## `@escaping`

`@escaping` is an attribute used in Swift for function parameters that take
closures as arguments. By default, closures passed to a function are
"non-escaping", which means that the closure is executed and completes within
the scope of the function. When a closure is marked with `@escaping`, it
indicates that the closure may "escape" the function and could be stored, used,
or executed later, outside the scope of the function, allowing the closure to
outlive the function call.

`@escaping` is often used when dealing with asynchronous operations, completion
handlers, or APIs that store closures as properties for later execution. Since
escaping closures may outlive the function, they need to capture and maintain
any values they reference, which may affect memory management and retain cycles.

```swift
func fetchData(completion: @escaping (Data?, Error?) -> Void) {
    let url = URL(string: "https://iamslash.com/data")!
    
    URLSession.shared.dataTask(with: url) { (data, response, error) in
        completion(data, error)
    }.resume()
}
```

## Closure vs Async/Await

Closure

```swift
import Foundation

func fetchData(completion: @escaping (Data?, Error?) -> Void) {
    let url = URL(string: "https://example.com/data")!
    
    URLSession.shared.dataTask(with: url) { (data, response, error) in
        completion(data, error)
    }.resume()
}

fetchData { data, error in
    if let error = error {
        print("Error: \(error)")
    } else {
        print("Data: \(data)")
    }
}
```

async/await (since swift 5.5)

```swift
import Foundation

func fetchData() async throws -> Data {
    let url = URL(string: "https://iamslash.com/data")!
    let (data, _) = try await URLSession.shared.data(from: url)
    return data
}

Task {
    do {
        let data = try await fetchData()
        print("Data: \(data)")
    } catch {
        print("Error: \(error)")
    }
}
```

## Diagnosing memory, thread, and crash issues early

[Diagnosing memory, thread, and crash issues early](https://developer.apple.com/documentation/xcode/diagnosing-memory-thread-and-crash-issues-early)

* **Address Sanitizer** — The ASan tool identifies potential memory-related
  corruption issues.
* **Thread Sanitizer** — The TSan tool detects race conditions between threads.
* **Main Thread Checker** — This tool verifies that system APIs that must run on
  the main thread actually do run on that thread.
* **Undefined Behavior Sanitizer** — The UBSan tool detects divide-by-zero
  errors, attempts to access memory using a misaligned pointer, and other
  undefined behaviors.

# Libraries

* [Alamofire](https://github.com/Alamofire/Alamofire)
  * HTTP Client Library

# Style Guide

[Swift Style Guide](swift_style_guide.md)

# Refactoring

[Refactoring Swift](refactoring_swift.md)

# Effective

[Effective Swift](effective_swift.md)

# GOF Design Pattern

[Swift GOF Design Pattern](swift_gof_designpattern.md)

# Architecture

## MV (Model View)

* [MVC](/java/java_designpattern.md#model-view-controller)

```swift
// User.swift - Model
struct User: Identifiable {
    let id: Int
    let name: String
    let age: Int
    let email: String
}

// UserListView.swift - View
import SwiftUI

struct UserListView: View {
    let users = [
        User(id: 1, name: "John Doe", age: 24, email: "john@example.com"),
        User(id: 2, name: "Jane Smith", age: 30, email: "jane@example.com")
    ]

    var body: some View {
        NavigationView {
            List(users) { user in
                VStack(alignment: .leading) {
                    Text(user.name)
                        .font(.headline)
                    Text("Age: \(user.age)")
                    Text(user.email)
                        .font(.subheadline)
                }
            }.navigationTitle("Users")
        }
    }
}
```

## MVVM (Model-View-View-Model)

* [MVVM](/java/java_designpattern.md#model-view-viewmodel)

```swift
// User.swift - Model
struct User: Identifiable {
    let id: Int
    let name: String
    let age: Int
    let email: String
}

// UserListView - View
import SwiftUI

struct UserListView: View {
    // Instantiate the ViewModel
    @StateObject private var viewModel = UserListViewModel()

    var body: some View {
        NavigationView {
            List(viewModel.users) { user in
                VStack(alignment: .leading) {
                    Text(user.name)
                        .font(.headline)
                    Text("Age: \(user.age)")
                    Text(user.email)
                        .font(.subheadline)
                }
            }.navigationTitle("Users")
        }
    }
}

// UserListViewModel.swift - View Model
import SwiftUI

class UserListViewModel: ObservableObject {
    @Published var users: [User] = [
        User(id: 1, name: "John Doe", age: 24, email: "john@example.com"),
        User(id: 2, name: "Jane Smith", age: 30, email: "jane@example.com")
    ]
}
```
