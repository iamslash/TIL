- [Materials](#materials)
- [Install](#install)
  - [Install on Windows 10](#install-on-windows-10)
  - [Install on OSX](#install-on-osx)
- [DSL (Domain Specific Language)](#dsl-domain-specific-language)
- [Basic](#basic)
  - [Compile, Run](#compile-run)
  - [Hello World](#hello-world)
  - [Reserved Words](#reserved-words)
  - [Data Types](#data-types)
  - [Print Formatted Text](#print-formatted-text)
  - [Control Flows](#control-flows)
  - [Loops](#loops)
  - [Operators](#operators)
  - [Collections compared to c++ containers](#collections-compared-to-c-containers)
  - [Collections](#collections)
  - [Functions](#functions)
  - [Struct, Class, Interface, AbstractClass](#struct-class-interface-abstractclass)
  - [Closure](#closure)
  - [Lambda](#lambda)
  - [Exception](#exception)
  - [Structure of Project](#structure-of-project)
- [Advanced](#advanced)
  - [Introspection](#introspection)
  - [Traits](#traits)

----

# Materials

* [Groovy](https://code-maven.com/groovy)
  * Groovy Tutotial for Jenkins Pipeline.
* [groovy  Documentation](http://groovy-lang.org/documentation.html)
  * [Differences with Java](http://groovy-lang.org/differences.html)
* [GROOVY TUTORIAL](http://www.newthinktank.com/2016/04/groovy-tutorial/)
  * 한 페이지로 배우는 groovy
* [Groovy Tutorial @ tutorial point](https://www.tutorialspoint.com/groovy/index.htm)
  * groovy tutorial
* [Groovy for Java developers @ youtube](https://www.youtube.com/watch?v=BXRDTiJfrSE)
  * java 개발자를 위한 groovy

# Install

## Install on Windows 10

```
choco install groovy
```

## Install on OSX

```bash
brew install groovy
```

# DSL (Domain Specific Language)

groovy 의 top level statement 는 argument 주변의 `()` 와 function call 사이의 `.` 를 생략할 수 있다. 이것을 `command chain` 이라고 한다.

```groovy
// equivalent to: turn(left).then(right)
turn left then right
// equivalent to: take(2.pills).of(chloroquinine).after(6.hours)
take 2.pills of chloroquinine after 6.hours
// equivalent to: paint(wall).with(red, green).and(yellow)
paint wall with red, green and yellow
// with named parameters too
// equivalent to: check(that: margarita).tastes(good)
check that: margarita tastes good
// with closures as parameters
// equivalent to: given({}).when({}).then({})
given { } when { } then { }
// equivalent to: select(all).unique().from(names)
select all unique() from names
// equivalent to: take(3).cookies
// and also this: take(3).getCookies()
take 3 cookies
```

다음은 `EmailDs1` 을 DSL 에서 사용하기 위한 구현이다. [Groovy - DSLS @ tutorialpoint](https://www.tutorialspoint.com/groovy/groovy_dsls.htm)

```groovy
class EmailDsl {  
   String toText 
   String fromText 
   String body 
	
   /** 
   * This method accepts a closure which is essentially the DSL. Delegate the 
   * closure methods to 
   * the DSL class so the calls can be processed 
   */ 
   
   def static make(closure) { 
      EmailDsl emailDsl = new EmailDsl() 
      // any method called in closure will be delegated to the EmailDsl class 
      closure.delegate = emailDsl
      closure() 
   }
   
   /** 
   * Store the parameter as a variable and use it later to output a memo 
   */ 
	
   def to(String toText) { 
      this.toText = toText 
   }
   
   def from(String fromText) { 
      this.fromText = fromText 
   }
   
   def body(String bodyText) { 
      this.body = bodyText 
   } 
}

EmailDsl.make { 
   to "Nirav Assar" 
   from "Barack Obama" 
   body "How are things? We are doing well. Take care" 
}
```

# Basic 

## Compile, Run

```bash
$ groovy a.groovy
```

## Hello World

* a.groovy

```groovy
class A {
    static void main(String[] args){
    // Print to the screen
    println("Hello World");
    }
}
```

## Reserved Words

```groovy
as     assert  break      case
catch  class   const      continue
def    default do         else  
enum   extends false      finally
for    goto    if         implements 
import in      instanceof interface
new    null    package    return
super  switch  this       throw
throws trait   true       try
while
```

## Data Types

* [Built-in Data Types @ tutorialpoints](https://www.tutorialspoint.com/groovy/groovy_data_types.htm)

----

```groovy
byte
short
int
long
float
double
char
Boolean
String
```

## Print Formatted Text

```groovy
println(String.format("%s : %s", "name", "pass"))
```

## Control Flows

```groovy
//// if-else
if (a < 100) {
   println("The value is less than 100")
} else if (a < 50) {
   println("The value is less than 50")
} else {
   println("The value is too small")
}

//// switch
switch(a) {
   case 1:
     println("The value is one")
     break;
   case 2:
     println("The value is two")
     break;
   default:
     println("The value is unknown")
     break;
}
```

## Loops

```groovy
//// while
while (cnt < 10) {
   println(cnt)
   cnt++
}

//// for
for (int i = 0; i < 5; i++) {
   println(i)
}

//// for-in
for (int i in 1..5) {
   println(i)
}
```

## Operators

```groovy
//// Arithmetic operators
1 + 2 // 3
2 - 1 // 1
2 * 2 // 4
3 / 2 // 1.5
3 % 2 // 1

//// Relational operators
2 == 2
3 != 2
2 < 3
2 <= 3
3 > 2
3 >= 2

//// Logical operators
true && true
ture || true
!false

//// Bitwise operators
1 & 1
1 | 0
1 ^ 1
~1

//// Assignment operators
def A = 5; 
A += 3
A -= 3
A *= 3
A /= 3
A %= 3
```

## Collections compared to c++ containers


| c++                  | groovy                            |
|:---------------------|:--------------------------------|
| `if, else`           | `if, else`                      |
| `for, while`         | `for, while`                    |
| `array`              | `Collections.unmodifiableList`  |
| `vector`             | `Vector, ArrayList`             |
| `deque`              | `Deque, ArrayDeque`             |
| `forward_list`       | ``                              |
| `list`               | `List, LinkedList`              |
| `stack`              | `Stack, Deque`                  |
| `queue`              | `Queue, LinkedList`             |
| `priority_queue`     | `Queue, PriorityQueue`          |
| `set`                | `SortedSet, TreeSet`       |
| `multiset`           | ``                              |
| `map`                | `SortedMap, TreeMap`       |
| `multimap`           | ``                              |
| `unordered_set`      | `Set, HashSet`                  |
| `unordered_multiset` | ``                              |
| `unordered_map`      | `Map, HashMap`                  |
| `unordered_multimap` | ``                              |

## Collections

Lists, Maps, Ranges

* Lists

```groovy
def list = [5, 6, 7, 8]
assert list.get(2) == 7
assert list[2] == 7
assert list instanceof java.util.List

def emptyList = []
assert emptyList.size() == 0
emptyList.add(5)
assert emptyList.size() == 1
```

* Maps

```groovy
def map = [name: 'Gromit', likes: 'cheese', id: 1234]
assert map.get('name') == 'Gromit'
assert map.get('id') == 1234
assert map['name'] == 'Gromit'
assert map['id'] == 1234
assert map instanceof java.util.Map

def emptyMap = [:]
assert emptyMap.size() == 0
emptyMap.put("foo", 5)
assert emptyMap.size() == 1
assert emptyMap.get("foo") == 5
```

* Ranges
  * Range extends java.util.List.

```groovy
// an inclusive range
def range = 5..8
assert range.size() == 4
assert range.get(2) == 7
assert range[2] == 7
assert range instanceof java.util.List
assert range.contains(5)
assert range.contains(8)

// lets use a half-open range
range = 5..<8
assert range.size() == 3
assert range.get(2) == 7
assert range[2] == 7
assert range instanceof java.util.List
assert range.contains(5)
assert !range.contains(8)

//get the end points of the range without using indexes
range = 1..10
assert range.from == 1
assert range.to == 10
```

## Functions

TODO

## Struct, Class, Interface, AbstractClass

TODO

## Closure

TODO

## Lambda

TODO

## Exception

TODO

## Structure of Project

TODO

# Advanced

## Introspection

* [What's the Groovy equivalent to Python's dir()? @ stackoverflow](https://stackoverflow.com/questions/10882469/whats-the-groovy-equivalent-to-pythons-dir)

----

```groovy
// Introspection, know all the details about classes :
// List all constructors of a class
String.constructors.each{println it}

// List all interfaces implemented by a class
String.interfaces.each{println it}

// List all methods offered by a class
String.methods.each{println it}

// Just list the methods names
String.methods.name

// Get the fields of an object (with their values)
d = new Date()
d.properties.each{ println(it) }
```

## Traits

TODO