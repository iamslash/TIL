
<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Abstract](#abstract)
- [Materials](#materials)
- [Basic Usages](#basic-usages)
    - [Collections compared c++ container](#collections-compared-c-container)
    - [Collections](#collections)
    - [let, var](#let-var)
    - [Data Types](#data-types)
    - [function](#function)
    - [conditional](#conditional)
    - [loop](#loop)
    - [optional](#optional)
    - [struct](#struct)
    - [class](#class)
    - [enum](#enum)
    - [value reference](#value-reference)
    - [closure](#closure)
    - [property](#property)
    - [inheritance](#inheritance)
    - [init, deinit](#init-deinit)
    - [optional chaining](#optional-chaining)
    - [type casting](#type-casting)
    - [assert guard](#assert-guard)
    - [protocol](#protocol)
    - [extension](#extension)
    - [error handling](#error-handling)
    - [higher order function](#higher-order-function)
- [Advanced](#advanced)
    - [Generics](#generics)
    - [Subscript](#subscript)
    - [Access Control](#access-control)
    - [ARC (Automatic Reference Counting)](#arc-automatic-reference-counting)
    - [Nested Types](#nested-types)
    - [Custom Operators](#custom-operators)

<!-- markdown-toc end -->

-------------------------------------------------------------------------------

# Abstract

swift에 대해 정리한다.

# Materials

* [the swift programming language
swift 4.2](https://docs.swift.org/swift-book/LanguageGuide/TheBasics.html)
* [야곰의 스위프트 기본 문법 강좌](https://www.inflearn.com/course/%EC%8A%A4%EC%9C%84%ED%94%84%ED%8A%B8-%EA%B8%B0%EB%B3%B8-%EB%AC%B8%EB%B2%95/)
  * swift 3 기본 문법
  * [src](https://github.com/yagom/swift_basic)

# Basic Usages

## Collections compared c++ container

[Collection Types](https://docs.swift.org/swift-book/LanguageGuide/CollectionTypes.html) 에 의하면 swift 의 collection 은 `Array, Set, Dictionary` 가 있다.

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

* Array

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
// i.append(101.1) // error
print(integers) // [1, 100]

print(i.contains(100)) // true
print(i.contains(90)) // false

i[0] = 99

i.remove(at: 0)
i.removeLast()
i.removeAll()

print(i.count)

// i[0] // error

let ii = [1, 2, 3]
// ii.append(4) // error
```

* Set

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

* Dictionary

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
let d1: [String: String] = ["name": "yagom", "gender": "male"]
// d0["key"] = "value" // error
// let somevalue: String = d1["name"] // error
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

* Bool, Int, UInt, Float, Double, Character, String
* Any, AnyObject, nil

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

## function

```swift
func sum(a: Int, b: Int) -> Int {
    return a + b
}

func printMyName(name: String) -> Void {
    print(name)
}

func printYourName(name: String) {
    print(name)
}

func maximumIntegerValue() -> Int {
    return Int.max
}

func hello() -> Void { print("hello") }

func bye() {print("bye") }

sum(a: 3, b: 5) // 8
printMyName(name: "Foo") // Foo
printYourName(name: "hana") // hana
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
```

## conditional

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
    print("zero)    
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

## loop

```swift
var integers = [1, 2, 3]
let people = ["foo": 10, "bar": 15, "baz": 12]
for integer in integers {
    print(integer)
}
for (name, age) in people {
    print("\(name): \n(age)")
}

while integers.count > 1 {
    integers.removeLast()
}

repeat {
    integers.removeLast()
} wihle integers.count > 0
```

## optional

```swift
import Swift

var implicitlyUnwrappedOptionalValue: Int! = 100
switch implicitlyUnwrappedOptionalValue {
case .nome:
    print("This Optional variable is nil")
case .some(let value)    
    print("Value is \(value)")
}

implicitlyUnwrappedOptionalValue = implicitlyUnwrappedOptionalValue + 1
implicitlyUnwrappedOptionalValue = nil
// implicitlyUnwrappedOptionalValue = implicitlyUnwrappedOptionalValue + 1 // error

var optionalValue: Int? = 100
switch optionalValue {
case .none:
    print("This Optional variable is nil")
case .some(let value):
    print("Value is \(value)")
}
optionalValue = nil
// optionalValue = optionalvalue + 1 // error
```

```swift
import Swift

func printName(_ name: String) {
    print(name)
}
var myName: String? = nil
if let name: String = myName {
    printName(name)
} else {
    print("myName == nil")
}

var yourName: String! = nil
if let name: String yourName {
    printName(name)
} else {
    print("yourName == nil")
}

myName = "Foo"
yourName = nil

if let name = myName, let friend = yourName {
    print("\n(name) and \(friend)")
}
yourName = "hana"
if let name = myName, let firned = yourName {
    print("\n(name) and \(friend)")
}
// Foo and hana

printName(myName!) // Foo
myName = nil
//print(myName!) // error
yourName = nil
//print(yourName) // error

```

## struct

```swift
struct Sample {
    var mutableProperty: Int = 100
    let immutableProperty: Int = 100
    static var typeProperty: Int = 100
    func instanceMethod() {
        print("instance method")
    }
    static func typeMethod() {
        print("type method)
    }
}

var mutable: Sample = Sample()
mutable.mutableProperty = 200
// mutable.immutableProperty = 200 // error
let immutable: Sample = Sample()
//immutable.mutableProperty = 200 // error
//immutable.immutableProperty = 200 // error

sample.typeProperty = 300
sample.typeMethod()

//mutable.typeProperty = 400 // error
//mutable.typeMethod() // error

struct Student {
    var name: String = "unknow"
    var `class`: String = "Swfit"
    static func selfIntroduce() {
        print("학생타입입니다.")
    }
    func selfIntroduce() {
        print("저는 \(self.class)반 \(name)입니다.")
    }
}
Student.selfIntroduct()

var foo: Student = Student()
foo.name = "Foo"
foo.class = "스위프트"
foo.selfIntroduce()

let jina: Student = Student()
//jina.name = "jina" // error
jina.selfIntroduce() 
```

## class

```swift

```

## enum

```swift
```

## value reference

```swift
```

## closure

```swift
```

## property

```swift
```

## inheritance

```swift
```

## init, deinit

```swift
```

## optional chaining

```swift
```

## type casting

```swift
```

## assert guard

```swift
```

## protocol

```swift
```

## extension

```swift
```

## error handling

```swift
```

## higher order function

```swift
```

# Advanced

## Generics

## Subscript

## Access Control

## ARC (Automatic Reference Counting)

## Nested Types

## Custom Operators