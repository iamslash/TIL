- [Abstract](#abstract)
- [References](#references)
- [Materials](#materials)
- [Articles](#articles)
  - [Tools](#tools)
  - [Best Practices](#best-practices)
  - [Concurrency](#concurrency)
  - [Error Handling](#error-handling)
  - [Interface](#interface)
  - [Struct](#struct)
  - [Pointer](#pointer)
  - [Map, Slice](#map-slice)
  - [Logging](#logging)
  - [Docker image](#docker-image)
  - [Encoding, JSON](#encoding-json)
  - [Profile](#profile)
- [Basic](#basic)
  - [Build and Run](#build-and-run)
  - [Hello World](#hello-world)
  - [Reserved Words](#reserved-words)
  - [min, max values](#min-max-values)
  - [abs, fabs](#abs-fabs)
  - [Bit Manipulation](#bit-manipulation)
  - [String](#string)
  - [Random](#random)
  - [Print Out](#print-out)
  - [Declarations](#declarations)
  - [Data Types](#data-types)
  - [Control Flows](#control-flows)
    - [If](#if)
    - [Switch](#switch)
    - [select](#select)
  - [Loops](#loops)
  - [Operators](#operators)
    - [Arithmetic](#arithmetic)
    - [Comparison](#comparison)
    - [Logical](#logical)
    - [Other](#other)
  - [Collections compared to c++ containers](#collections-compared-to-c-containers)
  - [Collections by examples](#collections-by-examples)
    - [array](#array)
    - [slice](#slice)
    - [map](#map)
    - [Nested Maps](#nested-maps)
    - [heap](#heap)
    - [list](#list)
    - [ring](#ring)
    - [sort](#sort)
    - [search](#search)
  - [Multidimensional Array](#multidimensional-array)
  - [Constants](#constants)
  - [Functions](#functions)
    - [Functions As Values And Closures](#functions-as-values-and-closures)
    - [Variadic Functions](#variadic-functions)
  - [Type Conversions](#type-conversions)
  - [Packages](#packages)
  - [Arrays, Slices, Ranges](#arrays-slices-ranges)
    - [Arrays](#arrays)
    - [Slices](#slices)
    - [Operations on Arrays and Slices](#operations-on-arrays-and-slices)
  - [Maps](#maps)
  - [Structs](#structs)
  - [Pointers](#pointers)
  - [Interfaces](#interfaces)
  - [Embedding](#embedding)
  - [Errors](#errors)
  - [Goroutines](#goroutines)
  - [Channels](#channels)
    - [Channel Axioms](#channel-axioms)
  - [Type Assertion](#type-assertion)
  - [Context](#context)
  - [module](#module)
- [Advanced](#advanced)
  - [Go memory ballast](#go-memory-ballast)
  - [go commands](#go-commands)
    - [go build](#go-build)
    - [go test](#go-test)
    - [go mod](#go-mod)
    - [go generate](#go-generate)
    - [go-wrk](#go-wrk)
    - [go-torch](#go-torch)
  - [bazel](#bazel)
  - [gazelle](#gazelle)
  - [present](#present)
  - [Debug](#debug)
  - [Testify](#testify)
  - [Gomock](#gomock)
  - [Benchmarks](#benchmarks)
  - [Profile](#profile-1)
  - [Vfsgen](#vfsgen)
  - [IntelliJ IDEA](#intellij-idea)
  - [Managing Multiple go versions](#managing-multiple-go-versions)
- [Effective Go](#effective-go)
- [Design Patterns](#design-patterns)
-------------------------------------------------------------------------------

# Abstract

오랜만에 go 를 다시 사용해야 한다면 [go wiki](https://github.com/golang/go/wiki) 를 다시 읽어보자.

# References

* [A Tour of Go](https://go.dev/tour/welcome/1)
  * 따라하기를 통해 Go 의 대부분을 익힐 수 있다.
* [effective go](https://golang.org/doc/effective_go.html)
  * [한글](https://gosudaweb.gitbooks.io/effective-go-in-korean/content/)
* [How to Write Go Code](https://go.dev/doc/code)
  * module, package 를 작성하는 방법을 설명한다.
* [go wiki](https://github.com/golang/go/wiki)
  * [CodeReviewComments](https://github.com/golang/go/wiki/CodeReviewComments)
  * [TestComments](https://github.com/golang/go/wiki/TestComments)
* [go blog](https://go.dev/blog/)
  * [Testable Examples in Go](https://go.dev/blog/examples)
* [go tool](https://pkg.go.dev/cmd/go)
  * the standard way to fetch, build, and install Go modules, packages, and commands.
* [The Go Programming Language Specification](https://go.dev/ref/spec)
  * Go 언어의 spec 을 확인한다.
* [uber-go style guide](https://github.com/uber-go/guide)
  * uber 의 go style guide 이다. 참고할만한 것이 많다.
* [upspin @ github](https://github.com/upspin/upspin)
  * Rob Pike 의 repo 이다. 배울 것이 많다.
* [go by example](https://gobyexample.com/)
  * 최고의 예제들
* [golang doc](https://golang.org/doc/)
* [go src](https://go.dev/src/)
* [Scheduling In Go : Part I - OS Scheduler](https://www.ardanlabs.com/blog/2018/08/scheduling-in-go-part1.html)
  * [Scheduling In Go : Part II - Go Scheduler](https://www.ardanlabs.com/blog/2018/08/scheduling-in-go-part2.html)
  * [Scheduling In Go : Part III - Concurrency](https://www.ardanlabs.com/blog/2018/12/scheduling-in-go-part3.html)

# Materials

* [Golang Design Patterns in Kubernetes](https://aly.arriqaaq.com/golang-design-patterns/?fbclid=IwAR20DyiTILpa3cMe0wt4JwF_Ll83Dluwnq6QPQpXyA3rkvELGZEmwDxsNoA)
* [DESIGN PATTERNS in GO](https://refactoring.guru/design-patterns/go)
* [디스커버리 Go 언어](http://www.yes24.com/Product/Goods/24759320)
  * [src](https://github.com/jaeyeom/gogo)
* [GoForCPPProgrammers](https://github.com/golang/go/wiki/GoForCPPProgrammers)
* [learn go in Y minutes](https://learnxinyminutes.com/docs/go/)
* [go @ tutorialspoint](https://www.tutorialspoint.com/go/)
* [가장 빨리 만나는 Go 언어](http://pyrasis.com/go.html)
  * 킹왕짱 golang 기본문법
  * [src](https://github.com/pyrasis/golangbook)
* [1ambda golang](https://github.com/1ambda/golang)
  * 유용한 go links
* [Go Bootcamp](http://www.golangbootcamp.com/book/collection_types)
  * 예제위주의 책
* [예제로 배우는 GO프로그래밍](http://golang.site/)
  * 최고의 한글 예제들
* [golang cheatsheet](https://github.com/a8m/go-lang-cheat-sheet)
  * 최고의 요약
* [A Tour of Go video](https://research.swtch.com/gotour)
  * interface, concurrency 에 관한 screencast
* [Rob Pike: Simplicity is Complicated](https://www.youtube.com/watch?v=rFejpH_tAHM)
* [Rob Pike: Go at Google](https://www.infoq.com/presentations/Go-Google)
  * [Article](https://talks.golang.org/2012/splash.article)
* [Golang FAQ: Design](https://golang.org/doc/faq#Design)
* [Golang FAQ: Types](https://golang.org/doc/faq#types)
* [Campoy: Functional Go?](https://www.youtube.com/watch?v=ouyHp2nJl0I)

# Articles

## Tools

* [Go Tooling in Action](https://www.youtube.com/watch?v=uBjoTxosSys)
  * go run, go build, go install, go test, go list, go doc, go-wrk,
    go-torch, debugging 등등을 다루는 킹왕짱 동영상

## Best Practices

* [Go best practices, six years in @ infoq](https://www.infoq.com/presentations/go-patterns)
  * [article](https://peter.bourgon.org/go-best-practices-2016/)
* [Twelve Go Best Practices @ youtube](https://www.youtube.com/watch?v=8D3Vmm1BGoY)
  * [slide](https://talks.golang.org/2013/bestpractices.slide#1)
* [Go Proverbs - Rob Pike](https://go-proverbs.github.io/)
  * rob pike가 전하는 golang철학

## Concurrency

* [go concurrency @ TIL](go_concurrency.md)
* [Rob Pike: Concurrency is not Parallelism @ youtube](https://www.youtube.com/watch?v=B9lP-E4J_lc)
  * [Slide](https://talks.golang.org/2012/waza.slide)
* [Go Concurrency Patterns](https://www.youtube.com/watch?v=f6kdp27TYZs)
* [Curious Channels](https://dave.cheney.net/2013/04/30/curious-channels)
* [Complex Concurrency Patterns With Go](https://www.youtube.com/watch?v=2HOO5gIgyMg)
* [Advanced Go Concurrency Patterns](https://www.youtube.com/watch?v=QDDwwePbDtw)

## Error Handling

- [Go Blog: Defer, Panic, and Recover](https://blog.golang.org/defer-panic-and-recover)
- [Go Blog: Error Handling and Go](https://blog.golang.org/error-handling-and-go)
- [Go Blog: Errors are Values](https://blog.golang.org/errors-are-values)
- [Dave Cheney: Don't just check errors, handle them gracefully @ youtube](https://www.youtube.com/watch?v=lsBF58Q-DnY)
  - [Article](https://dave.cheney.net/2016/04/27/dont-just-check-errors-handle-them-gracefully)
- [Dave Cheney: Why Go gets exceptions right](https://dave.cheney.net/2012/01/18/why-go-gets-exceptions-right)
- [Dave Cheney: Inspecting errors](https://dave.cheney.net/2014/12/24/inspecting-errors)
- [Dave Cheney: Error handling vs. exceptions redux](https://dave.cheney.net/2014/11/04/error-handling-vs-exceptions-redux)
- [Dave Cheney: Errors and Exceptions, redux](https://dave.cheney.net/2015/01/26/errors-and-exceptions-redux)
- [Dave Cheney: Constant errors](https://dave.cheney.net/2016/04/07/constant-errors)
- [Dave Cheney: Stack traces and the errors package](https://dave.cheney.net/2016/06/12/stack-traces-and-the-errors-package)

## Interface

- [Stackoverflow: What's the mearning of interface{} ?](http://stackoverflow.com/questions/23148812/go-whats-the-meaning-of-interface)
- [How to use interfaces in Go](http://jordanorelli.com/post/32665860244/how-to-use-interfaces-in-go)

## Struct

- [Dave Cheney: Struct composition with Go](https://dave.cheney.net/2015/05/22/struct-composition-with-go)
- [Dave Cheney: The empty struct](https://dave.cheney.net/2014/03/25/the-empty-struct)

## Pointer

- [Dave Cheney: Pointers in Go](https://dave.cheney.net/2014/03/17/pointers-in-go)
- [Things I Wish Someone Had Told Me About Go](http://openmymind.net/Things-I-Wish-Someone-Had-Told-Me-About-Go/)
- [Dave Cheney: Go has both make and new functions, what gives?](https://dave.cheney.net/2014/08/17/go-has-both-make-and-new-functions-what-gives)
- [Dave Cheney: Should methods be declared on T or *T](https://dave.cheney.net/2016/03/19/should-methods-be-declared-on-t-or-t)

## Map, Slice

- [Go Blog: Map in Action](https://blog.golang.org/go-maps-in-action)
- [Go Blog: Slices Usage and Internals](https://blog.golang.org/go-slices-usage-and-internals)

## Logging

- [The Hunt for a Logger Interface](http://go-talks.appspot.com/github.com/ChrisHines/talks/structured-logging/structured-logging.slide#1)
- [Logging v. instrumentation](https://peter.bourgon.org/blog/2016/02/07/logging-v-instrumentation.html)
- [Dave Cheney: Let’s talk about logging](https://dave.cheney.net/2015/11/05/lets-talk-about-logging)

## Docker image

- [Reducing Docker Image Size with UPX in Golang Projects](https://suaybsimsek58.medium.com/reducing-spring-native-and-golang-docker-image-size-by-70-with-upx-daf84e4f9227)

## Encoding, JSON

- [JSON, interface, and go generate](https://www.youtube.com/watch?v=YgnD27GFcyA)

## Profile

* [High Performance Go Workshop](https://dave.cheney.net/high-performance-go-workshop/dotgo-paris.html)
  * [video](https://www.youtube.com/watch?v=nok0aYiGiYA)
  * [src](https://github.com/davecheney/high-performance-go-workshop)
  * [번역](https://ziwon.github.io/post/high-performance-go-workshop/)

# Basic

## Build and Run

```bash
$ go build a.go
$ go build ./cmd/basic/...
$ go run a.go
$ go run ./cmd/basic/...
```

## Hello World

* a.go

```go
package main
import "fmt"
func main() {
    fmt.Println("Hello World")
}
// go run a.go
```

## Reserved Words

```go
break    default     func   interface select
case     defer       go     map       struct
chan     else        goto   package   switch
const    fallthrough if     range     type
continue for         import return    var
```

## min, max values

```go
// math package
const (
    MaxInt8   = 1<<7 - 1
    MinInt8   = -1 << 7
    MaxInt16  = 1<<15 - 1
    MinInt16  = -1 << 15
    MaxInt32  = 1<<31 - 1
    MinInt32  = -1 << 31
    MaxInt64  = 1<<63 - 1
    MinInt64  = -1 << 63
    MaxUint8  = 1<<8 - 1
    MaxUint16 = 1<<16 - 1
    MaxUint32 = 1<<32 - 1
    MaxUint64 = 1<<64 - 1
)

fmt.Println(math.MaxInt32)
fmt.Println(math.MinInt32)

const (
    MaxFloat32             = 3.40282346638528859811704183484516925440e+38  // 2**127 * (2**24 - 1) / 2**23
    SmallestNonzeroFloat32 = 1.401298464324817070923729583289916131280e-45 // 1 / 2**(127 - 1 + 23)

    MaxFloat64             = 1.797693134862315708145274237317043567981e+308 // 2**1023 * (2**53 - 1) / 2**52
    SmallestNonzeroFloat64 = 4.940656458412465441765687928682213723651e-324 // 1 / 2**(1023 - 1 + 52)
)

fmt.Println(math.MaxFloat32)
fmt.Println(math.SmallestNonzeroFloat32)
fmt.Println(math.MaxFloat64)
fmt.Println(math.SmallestNonzeroFloat64)

const MaxUint = ^uint(0) 
const MinUint = 0 
const MaxInt = int(MaxUint >> 1) 
const MinInt = -MaxInt - 1
```

## abs, fabs

```go
// Abs for int
func Abs(x int64) int64 {
	if x < 0 {
		return -x
	}
	return x
}
fmt.Println(Abs(-2)) // -2

// Abs for float
// func Abs(x float64) float64
import math
fmt.Println(math.Abs(-2.0)) // -2.0
```

## Bit Manipulation

WIP

## String

```go
// Sub string
s := "Hello World"
s = s[0:5]

// Convert string, int
import strconv
s, err := strconv.Itoa(12)
n := strconv.Atoi("12")

// creating strings
var greeting =  "Hello world!"

fmt.Printf("normal string: ")
fmt.Printf("%s", greeting)
fmt.Printf("\n")
fmt.Printf("hex bytes: ")

for i := 0; i < len(greeting); i++ {
    fmt.Printf("%x ", greeting[i])
}

fmt.Printf("\n")
const sampleText = "\xbd\xb2\x3d\xbc\x20\xe2\x8c\x98" 

/*q flag escapes unprintable characters, with + flag it escapses non-ascii 
characters as well to make output unambigous */
fmt.Printf("quoted string: ")
fmt.Printf("%+q", sampleText)
fmt.Printf("\n") 
// normal string: Hello world!
// hex bytes: 48 65 6c 6c 6f 20 77 6f 72 6c 64 21 
// quoted string: "\xbd\xb2=\xbc \u2318"

// length
var greeting =  "Hello world!"

fmt.Printf("String Length is: ")
fmt.Println(len(greeting))  
// 
// String Length is : 12

// concatenating strings
greetings :=  []string{"Hello","world!"}   
fmt.Println(strings.Join(greetings, " "))
// Hello world!

// append strings
a := "hello"
b := "world"
c := a + " " + b
d := c + string(' ')
e := a + b[0:1]
fmt.Printf("%s, %s, %s\n", c, d, e)
fmt.Printf("%T %T %d\n", a[0], a[0:1], len(a[0:1]))
// hello world, hello world , hellow
// uint8 string 1

a := "hello world"
for i, c := range a {
  fmt.Printf("%d:%T %c:%T %t\n", i, c, i, c, c == ' ')
}
// 0:int32 h:int32 false
// ...
fmt.Printf("%T\n", ' ')
// int32
```

## Random

* [Crypto Rand](https://github.com/golang/go/wiki/CodeReviewComments#crypto-rand)

-----

Do not use package `math/rand` to generate keys, even throwaway ones. Instead, use `crypto/rand's Reader`, and if you need text, print to hexadecimal or base64

```go
import (
	"crypto/rand"
	// "encoding/base64"
	// "encoding/hex"
	"fmt"
)

func Key() string {
	buf := make([]byte, 16)
	_, err := rand.Read(buf)
	if err != nil {
		panic(err)  // out of randomness, should never happen
	}
	return fmt.Sprintf("%x", buf)
	// or hex.EncodeToString(buf)
	// or base64.StdEncoding.EncodeToString(buf)
}
```

## Print Out

```go
import fmt

fmt.Println("Hello World")
fmt.Printf("%d\n", 12)


fmt.Println("Hello, 你好, नमस्ते, Привет, ᎣᏏᏲ") // basic print, plus newline
p := struct { X, Y int }{ 17, 2 }
fmt.Println( "My point:", p, "x coord=", p.X ) // print structs, ints, etc
s := fmt.Sprintln( "My point:", p, "x coord=", p.X ) // print to string variable

fmt.Printf("%d hex:%x bin:%b fp:%f sci:%e",17,17,17,17.0,17.0) // c-ish format
s2 := fmt.Sprintf( "%d %f", 17, 17.0 ) // formatted print to string variable

hellomsg := `
 "Hello" in Chinese is 你好 ('Ni Hao')
 "Hello" in Hindi is नमस्ते ('Namaste')
` // multi-line string literal, using back-tick at beginning and end
```

## Declarations

타입은 변수이름 뒤에 위치한다.

```go
var foo int // declaration without initialization
var foo int = 42 // declaration with initialization
var foo, bar int = 42, 1302 // declare and init multiple vars at once
var foo = 42 // type omitted, will be inferred
foo := 42 // shorthand, only in func bodies, omit var keyword, type is always implicit
const constant = "This is a constant"
```

## Data Types

```go
bool

string

int  int8  int16  int32  int64
uint uint8 uint16 uint32 uint64 uintptr

byte // alias for uint8

rune // alias for int32 ~= a character (Unicode code point) - very Viking

float32 float64

complex64 complex128
```

## Control Flows

### If
```go
func main() {
	// Basic one
	if x > 0 {
		return x
	} else {
		return -x
	}
    	
	// You can put one statement before the condition
	if a := b + c; a < 42 {
		return a
	} else {
		return a - 42
	}
    
	// Type assertion inside if
	var val interface{}
	val = "foo"
	if str, ok := val.(string); ok {
		fmt.Println(str)
	}
}
```

### Switch

```go
    // switch statement
    switch operatingSystem {
    case "darwin":
        fmt.Println("Mac OS Hipster")
        // cases break automatically, no fallthrough by default
    case "linux":
        fmt.Println("Linux Geek")
    default:
        // Windows, BSD, ...
        fmt.Println("Other")
    }

    // as with for and if, you can have an assignment statement before the switch value 
    switch os := runtime.GOOS; os {
    case "darwin": ...
    }

    // you can also make comparisons in switch cases
    number := 42
    switch {
        case number < 42:
            fmt.Println("Smaller")
        case number == 42:
            fmt.Println("Equal")
        case number > 42:
            fmt.Println("Greater")
    }
```

### select

* [[이더리움에서 배우는 Go언어] select 의 거의 모든 패턴들](https://hamait.tistory.com/1017)

----

```go
// select {
//    case communication clause  :
//       statement(s);      
//    case communication clause  :
//       statement(s); 
//    /* you can have any number of case statements */
//    default : /* Optional */
//       statement(s);
// }
package main

import "fmt"

func main() {
   var c1, c2, c3 chan int
   var i1, i2 int
   select {
      case i1 = <-c1:
         fmt.Printf("received ", i1, " from c1\n")
      case c2 <- i2:
         fmt.Printf("sent ", i2, " to c2\n")
      case i3, ok := (<-c3):  // same as: i3, ok := <-c3
         if ok {
            fmt.Printf("received ", i3, " from c3\n")
         } else {
            fmt.Printf("c3 is closed\n")
         }
      default:
         fmt.Printf("no communication\n")
   }    
}  
```

## Loops

```go
    // There's only `for`, no `while`, no `until`
    for i := 1; i < 10; i++ {
    }
    for ; i < 10; { // while - loop
    }
    for i < 10 { // you can omit semicolons if there is only a condition
    }
    for { // you can omit the condition ~ while (true)
    }

// break
   /* local variable definition */
   var a int = 10

   /* for loop execution */
   for a < 20 {
      fmt.Printf("value of a: %d\n", a);
      a++;
      if a > 15 {
         /* terminate the loop using break statement */
         break;
      }
   }

// continue
   /* local variable definition */
   var a int = 10

   /* do loop execution */
   for a < 20 {
      if a == 15 {
         /* skip the iteration */
         a = a + 1;
         continue;
      }
      fmt.Printf("value of a: %d\n", a);
      a++;     
   }  

// goto
   /* local variable definition */
   var a int = 10

   /* do loop execution */
   LOOP: for a < 20 {
      if a == 15 {
         /* skip the iteration */
         a = a + 1
         goto LOOP
      }
      fmt.Printf("value of a: %d\n", a)
      a++     
   }  

// string iteration
for i, c := range "Hello, 世界" {
        fmt.Printf("%d: %c\n", i, c)
}
```

## Operators

### Arithmetic

| Operator | Description         |            |
|----------|---------------------|------------|
| `+`      | addition            |            |
| `-`      | subtraction         |            |
| `*`      | multiplication      |            |
| `/`      | quotient            |            |
| `%`      | remainder           |            |
| `&`      | bitwise and         |            |
| '`|`'      | bitwise or          |            |
| `^`      | bitwise xor         |            |
| `&^`     | bit clear (and not) |            |
| `<<`     | left shift          |            |
| `>>`     | right shift         |            |

### Comparison

| Operator | Description           |
|----------|-----------------------|
| `==`     | equal                 |
| `!=`     | not equal             |
| `<`      | less than             |
| `<=`     | less than or equal    |
| `>`      | greater than          |
| `>=`     | greater than or equal |

### Logical

| Operator | Description |   |            |
|----------|-------------|---|------------|
| `&&`     | logical and |   |            |
| `||`     | logical or  |   |            |
| '`!`'      | logical not |   |            |

### Other

| Operator | Description                                    |
|----------|------------------------------------------------|
| `&`      | address of / create pointer                    |
| `*`      | dereference pointer                            |
| `<-`     | send / receive operator (see 'Channels' below) |

## Collections compared to c++ containers

| c++                  | go                   | 
|:---------------------|:---------------------|
| `if, else`           | `if, else`           |
| `for, while`         | `for`                |
| `array`              | `array`              |
| `vector`             | `slice`              |
| `deque`              | ``                   |
| `forward_list`       | ``                   |
| `list`               | `container/list`     |
| `stack`              | ``                   |
| `queue`              | ``                   |
| `priority_queue`     | `container/heap`     |
| `set`                | `map[keytype]struct{}`|
| `multiset`           | ``                   |
| `map`                | ``                   |
| `multimap`           | ``                   |
| `unordered_set`      | ``                   |
| `unordered_multiset` | ``                   |
| `unordered_map`      | `map`                |
| `unordered_multimap` | ``                   |

## Collections by examples

### array

```go
// In Go, an _array_ is a numbered sequence of elements of a
// specific length.

package main

import "fmt"

func main() {

    // Here we create an array `a` that will hold exactly
    // 5 `int`s. The type of elements and length are both
    // part of the array's type. By default an array is
    // zero-valued, which for `int`s means `0`s.
    var a [5]int
    fmt.Println("emp:", a)

    // We can set a value at an index using the
    // `array[index] = value` syntax, and get a value with
    // `array[index]`.
    a[4] = 100
    fmt.Println("set:", a)
    fmt.Println("get:", a[4])

    // The builtin `len` returns the length of an array.
    fmt.Println("len:", len(a))

    // Use this syntax to declare and initialize an array
    // in one line.
    b := [5]int{1, 2, 3, 4, 5}
    fmt.Println("dcl:", b)

    // Array types are one-dimensional, but you can
    // compose types to build multi-dimensional data
    // structures.
    var twoD [2][3]int
    for i := 0; i < 2; i++ {
        for j := 0; j < 3; j++ {
            twoD[i][j] = i + j
        }
    }
    fmt.Println("2d: ", twoD)
}
// $ go run arrays.go
// emp: [0 0 0 0 0]
// set: [0 0 0 0 100]
// get: 100
// len: 5
// dcl: [1 2 3 4 5]
// 2d:  [[0 1 2] [1 2 3]]
```

### slice

```go
// _Slices_ are a key data type in Go, giving a more
// powerful interface to sequences than arrays.

package main

import "fmt"

func main() {

    // Unlike arrays, slices are typed only by the
    // elements they contain (not the number of elements).
    // To create an empty slice with non-zero length, use
    // the builtin `make`. Here we make a slice of
    // `string`s of length `3` (initially zero-valued).
    s := make([]string, 3)
    fmt.Println("emp:", s)

    // We can set and get just like with arrays.
    s[0] = "a"
    s[1] = "b"
    s[2] = "c"
    fmt.Println("set:", s)
    fmt.Println("get:", s[2])

    // `len` returns the length of the slice as expected.
    fmt.Println("len:", len(s))

    // In addition to these basic operations, slices
    // support several more that make them richer than
    // arrays. One is the builtin `append`, which
    // returns a slice containing one or more new values.
    // Note that we need to accept a return value from
    // `append` as we may get a new slice value.
    s = append(s, "d")
    s = append(s, "e", "f")
    fmt.Println("apd:", s)

    // Slices can also be `copy`'d. Here we create an
    // empty slice `c` of the same length as `s` and copy
    // into `c` from `s`.
    c := make([]string, len(s))
    copy(c, s)
    fmt.Println("cpy:", c)

    // Slices support a "slice" operator with the syntax
    // `slice[low:high]`. For example, this gets a slice
    // of the elements `s[2]`, `s[3]`, and `s[4]`.
    l := s[2:5]
    fmt.Println("sl1:", l)

    // This slices up to (but excluding) `s[5]`.
    l = s[:5]
    fmt.Println("sl2:", l)

    // And this slices up from (and including) `s[2]`.
    l = s[2:]
    fmt.Println("sl3:", l)

    // We can declare and initialize a variable for slice
    // in a single line as well.
    t := []string{"g", "h", "i"}
    fmt.Println("dcl:", t)

    // Slices can be composed into multi-dimensional data
    // structures. The length of the inner slices can
    // vary, unlike with multi-dimensional arrays.
    twoD := make([][]int, 3)
    for i := 0; i < 3; i++ {
        innerLen := i + 1
        twoD[i] = make([]int, innerLen)
        for j := 0; j < innerLen; j++ {
            twoD[i][j] = i + j
        }
    }
    fmt.Println("2d: ", twoD)
}

// $ go run slices.go
// emp: [  ]
// set: [a b c]
// get: c
// len: 3
// apd: [a b c d e f]
// cpy: [a b c d e f]
// sl1: [c d e]
// sl2: [a b c d e]
// sl3: [c d e f]
// dcl: [g h i]
// 2d:  [[0] [1 2] [2 3 4]]
```

### map

```go
// _Maps_ are Go's built-in [associative data type](http://en.wikipedia.org/wiki/Associative_array)
// (sometimes called _hashes_ or _dicts_ in other languages).

package main

import "fmt"

func main() {

    // To create an empty map, use the builtin `make`:
    // `make(map[key-type]val-type)`.
    m := make(map[string]int)

    // Set key/value pairs using typical `name[key] = val`
    // syntax.
    m["k1"] = 7
    m["k2"] = 13

    // Printing a map with e.g. `fmt.Println` will show all of
    // its key/value pairs.
    fmt.Println("map:", m)

    // Get a value for a key with `name[key]`.
    v1 := m["k1"]
    fmt.Println("v1: ", v1)

    // The builtin `len` returns the number of key/value
    // pairs when called on a map.
    fmt.Println("len:", len(m))

    // The builtin `delete` removes key/value pairs from
    // a map.
    delete(m, "k2")
    fmt.Println("map:", m)

    // The optional second return value when getting a
    // value from a map indicates if the key was present
    // in the map. This can be used to disambiguate
    // between missing keys and keys with zero values
    // like `0` or `""`. Here we didn't need the value
    // itself, so we ignored it with the _blank identifier_
    // `_`.
    _, prs := m["k2"]
    fmt.Println("prs:", prs)

    // You can also declare and initialize a new map in
    // the same line with this syntax.
    n := map[string]int{"foo": 1, "bar": 2}
    fmt.Println("map:", n)

    // You can compare map using relfect.DeepEqual
    f := map[rune]int{'a' : 1, 'b' : 2}
    g := map[rune]int{'a' : 1, 'b' : 2}
    h := map[rune]int{'a' : 1, 'b' : 3}
    fmt.Println(reflect.DeepEqual(f, g))
    fmt.Println(reflect.DeepEqual(f, h))

    // not existed value
    M := make(map[int]int)
	i, ok := M[3]
	fmt.Println(i)
	fmt.Println(ok)
	fmt.Println(len(M))
}


// $ go run maps.go 
// map: map[k1:7 k2:13]
// v1:  7
// len: 2
// map: map[k1:7]
// prs: false
// map: map[foo:1 bar:2]
```

### Nested Maps

```go
var data = map[string]map[string]string{}

data["a"] = map[string]string{}
data["b"] = make(map[string]string)
data["c"] = make(map[string]string)

data["a"]["w"] = "x"
data["b"]["w"] = "x"
data["c"]["w"] = "x"
fmt.Println(data)
```

### heap

```go
// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This example demonstrates a priority queue built using the heap interface.
package heap_test

import (
	"container/heap"
	"fmt"
)

// An Item is something we manage in a priority queue.
type Item struct {
	value    string // The value of the item; arbitrary.
	priority int    // The priority of the item in the queue.
	// The index is needed by update and is maintained by the heap.Interface methods.
	index int // The index of the item in the heap.
}

// A PriorityQueue implements heap.Interface and holds Items.
type PriorityQueue []*Item

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
	// We want Pop to give us the highest, not lowest, priority so we use greater than here.
	return pq[i].priority > pq[j].priority
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
	n := len(*pq)
	item := x.(*Item)
	item.index = n
	*pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	item.index = -1 // for safety
	*pq = old[0 : n-1]
	return item
}

// update modifies the priority and value of an Item in the queue.
func (pq *PriorityQueue) update(item *Item, value string, priority int) {
	item.value = value
	item.priority = priority
	heap.Fix(pq, item.index)
}

// This example creates a PriorityQueue with some items, adds and manipulates an item,
// and then removes the items in priority order.
func Example_priorityQueue() {
	// Some items and their priorities.
	items := map[string]int{
		"banana": 3, "apple": 2, "pear": 4,
	}

	// Create a priority queue, put the items in it, and
	// establish the priority queue (heap) invariants.
	pq := make(PriorityQueue, len(items))
	i := 0
	for value, priority := range items {
		pq[i] = &Item{
			value:    value,
			priority: priority,
			index:    i,
		}
		i++
	}
	heap.Init(&pq)

	// Insert a new item and then modify its priority.
	item := &Item{
		value:    "orange",
		priority: 1,
	}
	heap.Push(&pq, item)
	pq.update(item, item.value, 5)

	// Take the items out; they arrive in decreasing priority order.
	for pq.Len() > 0 {
		item := heap.Pop(&pq).(*Item)
		fmt.Printf("%.2d:%s ", item.priority, item.value)
	}
	// Output:
	// 05:orange 04:pear 03:banana 02:apple
}
```

### list

```go
// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package list_test

import (
	"container/list"
	"fmt"
)

func Example() {
	// Create a new list and put some numbers in it.
	l := list.New()
	e4 := l.PushBack(4)
	e1 := l.PushFront(1)
	l.InsertBefore(3, e4)
	l.InsertAfter(2, e1)

	// Iterate through list and print its contents.
	for e := l.Front(); e != nil; e = e.Next() {
		fmt.Println(e.Value)
	}

	// Output:
	// 1
	// 2
	// 3
	// 4
}
```

### ring

```go
// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ring_test

import (
	"container/ring"
	"fmt"
)

func ExampleRing_Len() {
	// Create a new ring of size 4
	r := ring.New(4)

	// Print out its length
	fmt.Println(r.Len())

	// Output:
	// 4
}

func ExampleRing_Next() {
	// Create a new ring of size 5
	r := ring.New(5)

	// Get the length of the ring
	n := r.Len()

	// Initialize the ring with some integer values
	for i := 0; i < n; i++ {
		r.Value = i
		r = r.Next()
	}

	// Iterate through the ring and print its contents
	for j := 0; j < n; j++ {
		fmt.Println(r.Value)
		r = r.Next()
	}

	// Output:
	// 0
	// 1
	// 2
	// 3
	// 4
}

func ExampleRing_Prev() {
	// Create a new ring of size 5
	r := ring.New(5)

	// Get the length of the ring
	n := r.Len()

	// Initialize the ring with some integer values
	for i := 0; i < n; i++ {
		r.Value = i
		r = r.Next()
	}

	// Iterate through the ring backwards and print its contents
	for j := 0; j < n; j++ {
		r = r.Prev()
		fmt.Println(r.Value)
	}

	// Output:
	// 4
	// 3
	// 2
	// 1
	// 0
}

func ExampleRing_Do() {
	// Create a new ring of size 5
	r := ring.New(5)

	// Get the length of the ring
	n := r.Len()

	// Initialize the ring with some integer values
	for i := 0; i < n; i++ {
		r.Value = i
		r = r.Next()
	}

	// Iterate through the ring and print its contents
	r.Do(func(p interface{}) {
		fmt.Println(p.(int))
	})

	// Output:
	// 0
	// 1
	// 2
	// 3
	// 4
}

func ExampleRing_Move() {
	// Create a new ring of size 5
	r := ring.New(5)

	// Get the length of the ring
	n := r.Len()

	// Initialize the ring with some integer values
	for i := 0; i < n; i++ {
		r.Value = i
		r = r.Next()
	}

	// Move the pointer forward by three steps
	r = r.Move(3)

	// Iterate through the ring and print its contents
	r.Do(func(p interface{}) {
		fmt.Println(p.(int))
	})

	// Output:
	// 3
	// 4
	// 0
	// 1
	// 2
}

func ExampleRing_Link() {
	// Create two rings, r and s, of size 2
	r := ring.New(2)
	s := ring.New(2)

	// Get the length of the ring
	lr := r.Len()
	ls := s.Len()

	// Initialize r with 0s
	for i := 0; i < lr; i++ {
		r.Value = 0
		r = r.Next()
	}

	// Initialize s with 1s
	for j := 0; j < ls; j++ {
		s.Value = 1
		s = s.Next()
	}

	// Link ring r and ring s
	rs := r.Link(s)

	// Iterate through the combined ring and print its contents
	rs.Do(func(p interface{}) {
		fmt.Println(p.(int))
	})

	// Output:
	// 0
	// 0
	// 1
	// 1
}

func ExampleRing_Unlink() {
	// Create a new ring of size 6
	r := ring.New(6)

	// Get the length of the ring
	n := r.Len()

	// Initialize the ring with some integer values
	for i := 0; i < n; i++ {
		r.Value = i
		r = r.Next()
	}

	// Unlink three elements from r, starting from r.Next()
	r.Unlink(3)

	// Iterate through the remaining ring and print its contents
	r.Do(func(p interface{}) {
		fmt.Println(p.(int))
	})

	// Output:
	// 0
	// 4
	// 5
}
```

### sort

  * [The 3 ways to sort in Go](https://yourbasic.org/golang/how-to-sort-in-go/)
  * [golang.org/src/sort/example_test.go](https://golang.org/src/sort/example_test.go)

----

```go
//// array
// ERROR: cannot use a (type [5]int) as type []int in argument to sort.Ints
a := [5]int{5, 4, 3, 2, 1}
sort.Ints(a)

//// slice 
// ascending order
a := []int{5, 3, 4, 7, 8, 9}
sort.Slice(a, func(i, j int) bool {
    return a[i] < a[j]
})
// descending order
sort.Sort(sort.Reverse(sort.IntSlice(a)))

//// custom comparator
family := []struct {
    Name string
    Age  int
}{
    {"Alice", 23},
    {"David", 2},
    {"Eve", 2},
    {"Bob", 25},
}
// Sort by age, keeping original order or equal elements.
sort.SliceStable(family, func(i, j int) bool {
    return family[i].Age < family[j].Age
})
fmt.Println(family) // [{David 2} {Eve 2} {Alice 23} {Bob 25}]

//// custom data structures
// this is sort.Interface
type Interface interface {
  Len() int
  Less(i, j int) bool
  Swap(i, j int)
}

type Person struct {
  Name string
  Age  int
}
// ByAge implements sort.Interface based on the Age field.
type ByAge []Person
func (a ByAge) Len() int           { return len(a) }
func (a ByAge) Less(i, j int) bool { return a[i].Age < a[j].Age }
func (a ByAge) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

func main() {
family := []Person{
  {"Alice", 23},
  {"Eve", 2},
  {"Bob", 25},
}
sort.Sort(ByAge(family))
fmt.Println(family) // [{Eve 2} {Alice 23} {Bob 25}]
}
 
//// map by keys
m := map[string]int{"Alice": 2, "Cecil": 1, "Bob": 3}
keys := make([]string, 0, len(m))
for k := range m {
  keys = append(keys, k)
}
sort.Strings(keys)
for _, k := range keys {
  fmt.Println(k, m[k])
}
// Output:
// Alice 2
// Bob 3
// Cecil 1
```

### search

```go
// binary search
A := []int{2, 3, 5, 6}
fmt.Println(sort.SearchInts(A, 1)) // 0
fmt.Println(sort.SearchInts(A, 3)) // 1
fmt.Println(sort.SearchInts(A, 4)) // 2
fmt.Println(sort.SearchInts(A, 7)) // 4
```

## Multidimensional Array

* [How is two dimensional array's memory representation @ stackoverflow](https://stackoverflow.com/questions/39561140/how-is-two-dimensional-arrays-memory-representation)
* [Arrays, slices (and strings): The mechanics of 'append'](https://blog.golang.org/slices)

----

```go
//// 2d slice
var n = len(s)
var C = make([][]int, n+1)
for i := range C {
  C[i] = make([]int, n+1) 
}

a := [][]uint8{
    {0, 1, 2, 3},
    {4, 5, 6, 7},
}
fmt.Println(a) // Output is [[0 1 2 3] [4 5 6 7]]

//// partial initialization
b := []uint{10: 1, 2}
fmt.Println(b) // Prints [0 0 0 0 0 0 0 0 0 0 1 2]

//// 2d array
c := [5][5]uint8{}
fmt.Println(c)
// [[0 0 0 0 0] [0 0 0 0 0] [0 0 0 0 0] [0 0 0 0 0] [0 0 0 0 0]]

//// 2d array initialization partially
var n = len("DID")
var C = make([][]int, n+1)
for i := range C {
  C[i] = make([]int, n+1) 
}
for i := range C[0] {
  C[0][i] = 1
}
fmt.Println(C)
// [[1 1 1 1] [0 0 0 0] [0 0 0 0] [0 0 0 0]]
```

## Constants

```go
// Integer literals
212         /* Legal */
215u        /* Legal */
0xFeeL      /* Legal */
078         /* Illegal: 8 is not an octal digit */
032UU       /* Illegal: cannot repeat a suffix */
85         /* decimal */
0213       /* octal */
0x4b       /* hexadecimal */
30         /* int */
30u        /* unsigned int */
30l        /* long */
30ul       /* unsigned long */
1e1        // 10
10e1       // 100

// floating-point literals
3.14159       /* Legal */
314159E-5L    /* Legal */
510E          /* Illegal: incomplete exponent */
210f          /* Illegal: no decimal or exponent */
.e55          /* Illegal: missing integer or fraction */

// string literals
"hello, dear"

"hello, \

dear"

"hello, " "d" "ear"

// const
package main

import "fmt"

func main() {
   const LENGTH int = 10
   const WIDTH int = 5   
   var area int

   area = LENGTH * WIDTH
   fmt.Printf("value of area : %d", area)   
}
```

## Functions

```go
// a simple function
func functionName() {}

// function with parameters (again, types go after identifiers)
func functionName(param1 string, param2 int) {}

// multiple parameters of the same type
func functionName(param1, param2 int) {}

// return type declaration
func functionName() int {
    return 42
}

// Can return multiple values at once
func returnMulti() (int, string) {
    return 42, "foobar"
}
var x, str = returnMulti()

// Return multiple named results simply by return
func returnMulti2() (n int, s string) {
    n = 42
    s = "foobar"
    // n and s will be returned
    return
}
var x, str = returnMulti2()
```

### Functions As Values And Closures

A closure is a function value that references variables from outside its body. [Function closures](https://go.dev/tour/moretypes/25) 참고.

```go
func main() {
    // assign a function to a name
    add := func(a, b int) int {
        return a + b
    }
    // use the name to call the function
    fmt.Println(add(3, 4))
}

// Closures, lexically scoped: Functions can access values that were
// in scope when defining the function
func scope() func() int{
    outer_var := 2
    foo := func() int { return outer_var}
    return foo
}

func another_scope() func() int{
    // won't compile because outer_var and foo not defined in this scope
    outer_var = 444
    return foo
}


// Closures: don't mutate outer vars, instead redefine them!
func outer() (func() int, int) {
    outer_var := 2
    inner := func() int {
        outer_var += 99 // attempt to mutate outer_var from outer scope
        return outer_var // => 101 (but outer_var is a newly redefined
                         //         variable visible only inside inner)
    }
    return inner, outer_var // => 101, 2 (outer_var is still 2, not mutated by inner!)
}
```

### Variadic Functions

* [Ultimate Guide to Go Variadic Functions](https://blog.learngoprogramming.com/golang-variadic-funcs-how-to-patterns-369408f19085)

----

```go
func main() {
	fmt.Println(adder(1, 2, 3)) 	// 6
	fmt.Println(adder(9, 9))	// 18
	
	nums := []int{10, 20, 30}
	fmt.Println(adder(nums...))	// 60
}

// By using ... before the type name of the last parameter you can indicate that it takes zero or more of those parameters.
// The function is invoked like any other function except we can pass as many arguments as we want.
func adder(args ...int) int {
	total := 0
	for _, v := range args { // Iterates over the arguments whatever the number.
		total += v
	}
	return total
}
```

## Type Conversions

```go
var i int = 42
var f float64 = float64(i)
var u uint = uint(f)

// alternative syntax
i := 42
f := float64(i)
u := uint(f)
```

## Packages 

* Package declaration at top of every source file
* Executables are in package `main`
* Convention: package name == last name of import path (import path `math/rand` => package `rand`)
* Upper case identifier: exported (visible from other packages)
* Lower case identifier: private (not visible from other packages)

## Arrays, Slices, Ranges

### Arrays

```go
var a [10]int // declare an int array with length 10. Array length is part of the type!
a[3] = 42     // set elements
i := a[3]     // read elements

// declare and initialize
var a = [2]int{1, 2}
a := [2]int{1, 2} //shorthand
a := [...]int{1, 2} // elipsis -> Compiler figures out array length
```

### Slices

```go
var a []int                              // declare a slice - similar to an array, but length is unspecified
var a = []int {1, 2, 3, 4}               // declare and initialize a slice (backed by the array given implicitly)
a := []int{1, 2, 3, 4}                   // shorthand
chars := []string{0:"a", 2:"c", 1: "b"}  // ["a", "b", "c"]

var b = a[lo:hi]	// creates a slice (view of the array) from index lo to hi-1
var b = a[1:4]		// slice from index 1 to 3
var b = a[:3]		// missing low index implies 0
var b = a[3:]		// missing high index implies len(a)
a =  append(a,17,3)	// append items to slice a
c := append(a,b...)	// concatenate slices a and b

// create a slice with make
a = make([]byte, 5, 5)	// first arg length, second capacity
a = make([]byte, 5)	// capacity is optional

// create a slice from an array
x := [3]string{"Лайка", "Белка", "Стрелка"}
s := x[:] // a slice referencing the storage of x
```

### Operations on Arrays and Slices

`len(a)` gives you the length of an array/a slice. It's a built-in function, not a attribute/method on the array.

```go
// loop over an array/a slice
for i, e := range a {
    // i is the index, e the element
}

// if you only need e:
for _, e := range a {
    // e is the element
}

// ...and if you only need the index
for i := range a {
}

// In Go pre-1.4, you'll get a compiler error if you're not using i and e.
// Go 1.4 introduced a variable-free form, so that you can do this
for range time.Tick(time.Second) {
    // do it once a sec
}

```

## Maps

```go
var m map[string]int
m = make(map[string]int)
m["key"] = 42
fmt.Println(m["key"])

delete(m, "key")

elem, ok := m["key"] // test if key "key" is present and retrieve it, if so

// map literal
var m = map[string]Vertex{
    "Bell Labs": {40.68433, -74.39967},
    "Google":    {37.42202, -122.08408},
}

```

## Structs

There are no classes, only structs. Structs can have methods.

```go
// A struct is a type. It's also a collection of fields 

// Declaration
type Vertex struct {
    X, Y int
}

// Creating
var v = Vertex{1, 2}
var v = Vertex{X: 1, Y: 2} // Creates a struct by defining values with keys 
var v = []Vertex{{1,2},{5,2},{5,5}} // Initialize a slice of structs

// Accessing members
v.X = 4

// You can declare methods on structs. The struct you want to declare the
// method on (the receiving type) comes between the the func keyword and
// the method name. The struct is copied on each method call(!)
func (v Vertex) Abs() float64 {
    return math.Sqrt(v.X*v.X + v.Y*v.Y)
}

// Call method
v.Abs()

// For mutating methods, you need to use a pointer (see below) to the Struct
// as the type. With this, the struct value is not copied for the method call.
func (v *Vertex) add(n float64) {
    v.X += n
    v.Y += n
}

// struct{} for signal without allocating memory
func main() {
    done := make(chan struct{})
    go func() {
        time.Sleep(1 * time.Second)
        close(done)
    }()

    fmt.Println("Wait...")
    <-done
    fmt.Println("done.")
}
```

**Anonymous structs:**  

Cheaper and safer than using `map[string]interface{}`.

```go
point := struct {
	X, Y int
}{1, 2}
```

**struct tags**

> [Custom struct field tags in Golang](https://sosedoff.com/2016/07/16/golang-struct-tags.html)

struct 의 field 에 tag 를 달아두고 Runtime 에 얻어올 수 있다. tag 는 field 의 추가 정보이다.

```go
type User struct {
  Id        int       `json:"id"`
  Name      string    `json:"name"`
  Bio       string    `json:"about,omitempty"`
  Active    bool      `json:"active"`
  Admin     bool      `json:"-"`
  CreatedAt time.Time `json:"created_at"`
}
```

## Pointers


----

`*` 은 Type 과 Value 앞에 올 수 있다. `T` 를 type, `V` 를 value 라고 하자. 

`*T` 는 `pointer of T type` 를 의미한다. `*V` 는
`dereference of V value` 를 의미한다. `V` 는 pointer type value 이다.

`&` 는 Value 앞에 올 수 있다. `&V` 는 `address of V value` 를 의미한다.

Pointers to structs 의 경우 explicit pionter 를 생략할 수 있다. 즉, `(*p).X` 대신 `p.X` 가 가능하다.

```go
func main() {
	i, j := 42, 2701

	p := &i         // point to i
	fmt.Println(*p) // read i through the pointer
	*p = 21         // set i through the pointer
	fmt.Println(i)  // see the new value of i

	p = &j         // point to j
	*p = *p / 37   // divide j through the pointer
	fmt.Println(j) // see the new value of j
}

type Vertex struct {
	X int
	Y int
}

func main() {
	v := Vertex{1, 2}
	p := &v
	p.X = 1e9
	fmt.Println(v)  // {1e+09 2}
  (*p).X = 333
  fmt.Println(v)  // {333 2}
}
```

```go
p := Vertex{1, 2}  // p is a Vertex
q := &p            // q is a pointer to a Vertex
r := &Vertex{1, 2} // r is also a pointer to a Vertex

// The type of a pointer to a Vertex is *Vertex

var s *Vertex = new(Vertex) // new creates a pointer to a new struct instance 
```

## Interfaces

```go
// interface declaration
type Awesomizer interface {
    Awesomize() string
}

// types do *not* declare to implement interfaces
type Foo struct {}

// instead, types implicitly satisfy an interface if they implement all required methods
func (foo Foo) Awesomize() string {
    return "Awesome!"
}
```

**empty interface**

> [The Go Empty Interface Explained](https://flaviocopes.com/go-empty-interface/)

`interface {}` 는 마치 `c` 의 `void*` 와 같다. [Type Assertion](#type-assertion) 을 사용하여
type casting 할 수 있다. `switch` 와 함께 `.(type)` 을 사용하면 type 별로 business logic 을 처리할 수 있다. 

```go
// basic empty interface
t := []int{1, 2, 3, 4}
s := make([]interface{}, len(t))
for i, v := range t {
    s[i] = v
}

// [Type Assertion](#type-assertion)

// switch with type assertion
var t interface{}
t = functionOfSomeType()
switch t := t.(type) {
default:
    fmt.Printf("unexpected type %T\n", t)     // %T prints whatever type t has
case bool:
    fmt.Printf("boolean %t\n", t)             // t has type bool
case int:
    fmt.Printf("integer %d\n", t)             // t has type int
case *bool:
    fmt.Printf("pointer to boolean %t\n", *t) // t has type *bool
case *int:
    fmt.Printf("pointer to integer %d\n", *t) // t has type *int
}
```

## Embedding

There is no subclassing in Go. Instead, there is interface and struct embedding.

```go
// ReadWriter implementations must satisfy both Reader and Writer
type ReadWriter interface {
    Reader
    Writer
}

// Server exposes all the methods that Logger has
type Server struct {
    Host string
    Port int
    *log.Logger
}

// initialize the embedded type the usual way
server := &Server{"localhost", 80, log.New(...)}

// methods implemented on the embedded struct are passed through
server.Log(...) // calls server.Logger.Log(...)

// the field name of the embedded type is its type name (in this case Logger)
var logger *log.Logger = server.Logger
```

## Errors

There is no exception handling. Functions that might produce an error just declare an additional return value of type `Error`. This is the `Error` interface:

```go
type error interface {
    Error() string
}
```

A function that might return an error:

```go
func doStuff() (int, error) {
}

func main() {
    result, error := doStuff()
    if (error != nil) {
        // handle error
    } else {
        // all is good, use result
    }
}
```

## Goroutines

Goroutines are lightweight threads (managed by Go, not OS threads). `go f(a, b)` starts a new goroutine which runs `f` (given `f` is a function).

```go
// just a function (which can be later started as a goroutine)
func doStuff(s string) {
}

func main() {
    // using a named function in a goroutine
    go doStuff("foobar")

    // using an anonymous inner function in a goroutine
    go func (x int) {
        // function body goes here
    }(42)
}
```

## Channels

* [How to Gracefully Close Channels](https://go101.org/article/channel-closing.html)
  * 안전하게 보내는 방법도 있다.

-----

```go
ch := make(chan int) // create a channel of type int
ch <- 42             // Send a value to the channel ch.
v := <-ch            // Receive a value from ch

// Non-buffered channels block. Read blocks when no value is available, write blocks if a value already has been written but not read.

// Create a buffered channel. Writing to a buffered channels does not block if less than <buffer size> unread values have been written.
ch := make(chan int, 100)

close(ch) // closes the channel (only sender should close)

// read from channel and test if it has been closed
v, ok := <-ch

// if ok is false, channel has been closed

// Read from channel until it is closed
for i := range ch {
    fmt.Println(i)
}

// select blocks on multiple channel operations, if one unblocks, the corresponding case is executed
func doStuff(channelOut, channelIn chan int) {
    select {
    case channelOut <- 42:
        fmt.Println("We could write to channelOut!")
    case x := <- channelIn:
        fmt.Println("We could read from channelIn")
    case <-time.After(time.Second * 1):
        fmt.Println("timeout")
    }
}

//
func SafeSend(ch chan T, value T) (closed bool) {
	defer func() {
		if recover() != nil {
			closed = true
		}
	}()

	ch <- value  // panic if ch is closed
	return false // <=> closed = false; return
}
```

### Channel Axioms

- A send to a nil channel blocks forever

  ```go
  var c chan string
  c <- "Hello, World!"
  // fatal error: all goroutines are asleep - deadlock!
  ```

- A receive from a nil channel blocks forever

  ```go
  var c chan string
  fmt.Println(<-c)
  // fatal error: all goroutines are asleep - deadlock!
  ```

- A send to a closed channel panics

  ```go
  var c = make(chan string, 1)
  c <- "Hello, World!"
  close(c)
  c <- "Hello, Panic!"
  // panic: send on closed channel
  ```
  
- A receive from a closed channel returns the zero value immediately

  ```go
  var c = make(chan int, 2)
  c <- 1
  c <- 2
  close(c)
  for i := 0; i < 3; i++ {
      fmt.Printf("%d ", <-c) 
  }
  // 1 2 0
  ```

## Type Assertion

> * [What is the meaning of “dot parenthesis” syntax? [duplicate] @ stackoverflow](https://stackoverflow.com/questions/24492868/what-is-the-meaning-of-dot-parenthesis-syntax)
> * [Type assertions @ go](https://golang.org/ref/spec#Type_assertions)

```go
var i interface{}
i = int(42)

a, ok := i.(int)
// a == 42 and ok == true

b, ok := i.(string)
// b == "" (default value) and ok == false
```

## Context

> * [Using Context in Golang - Cancellation, Timeouts and Values (With Examples)](https://www.sohamkamani.com/golang/context-cancellation-and-values/)
>   * [src](https://github.com/sohamkamani/blog-example-go-context-cancellation)
> * [Go Concurrency Patterns: Context @ go blog](https://go.dev/blog/context)
> * [Discover Packages | Standard library | context](https://pkg.go.dev/context)
> * [Contexts and structs @ go blog](https://go.dev/blog/context-and-structs)

----

여러 go routine 들을 하나의 Context 로 묶는다. Context 를 취소하면 모든 go
routine 들이 취소된다. 

```go
// Open browser and exit broser. This will cancel Context.
func main() {
	// Create an HTTP server that listens on port 8000
	http.ListenAndServe(":8010", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()
		// This prints to STDOUT to show that processing has started
		fmt.Fprint(os.Stdout, "processing request\n")
		// We use `select` to execute a peice of code depending on which
		// channel receives a message first
		select {
		case <-time.After(2 * time.Second):
			// If we receive a message after 2 seconds
			// that means the request has been processed
			w.Write([]byte("request processed"))
		case <-ctx.Done():
			// If the request gets cancelled, log it
			// to STDERR
			fmt.Fprint(os.Stderr, "request cancelled\n")
		}
	}))
}

// Create Context with timeout.
func main() {
	// Create a new context
	// With a deadline of 100 milliseconds
	ctx := context.Background()
	ctx, _ = context.WithTimeout(ctx, 100*time.Millisecond)

	// Make a request, that will call the google homepage
	req, _ := http.NewRequest(http.MethodGet, "http://google.com", nil)
	// Associate the cancellable context we just created to the request
	req = req.WithContext(ctx)

	// Create a new HTTP client and execute the request
	client := &http.Client{}
	res, err := client.Do(req)
	// If the request failed, log to STDOUT
	if err != nil {
		fmt.Println("Request failed:", err)
		return
	}
	// Print the statuscode if the request succeeds
	fmt.Println("Response received, status code:", res.StatusCode)
}

// Call cancel function to cancel the Context.
func operation1(ctx context.Context) error {
	// Let's assume that this operation failed for some reason
	// We use time.Sleep to simulate a resource intensive operation
	time.Sleep(100 * time.Millisecond)
	return errors.New("failed")
}
func operation2(ctx context.Context) {
	// We use a similar pattern to the HTTP server
	// that we saw in the earlier example
	select {
	case <-time.After(500 * time.Millisecond):
		fmt.Println("done")
	case <-ctx.Done():
		fmt.Println("halted operation2")
	}
}
func main() {
	// Create a new context
	ctx := context.Background()
	// Create a new context, with its cancellation function
	// from the original context
	ctx, cancel := context.WithCancel(ctx)

	// Run two operations: one in a different go routine
	go func() {
		err := operation1(ctx)
		// If this operation returns an error
		// cancel all operations using this context
		if err != nil {
			cancel()
		}
	}()

	// Run operation2 with the same context we use for operation1
	operation2(ctx)
}

// Write key, value to the Context.
// we need to set a key that tells us where the data is stored
const keyID = "id"
func main() {
	rand.Seed(time.Now().Unix())
	ctx := context.WithValue(context.Background(), keyID, rand.Int())
	operation1(ctx)
}
func operation1(ctx context.Context) {
	// do some work

	// we can get the value from the context by passing in the key
	log.Println("operation1 for id:", ctx.Value(keyID), " completed")
	operation2(ctx)
}
func operation2(ctx context.Context) {
	// do some work

	// this way, the same ID is passed from one function call to the next
	log.Println("operation2 for id:", ctx.Value(keyID), " completed")
}
```

## module

* [go module @ TIL](go_module.md)

# Advanced

## Go memory ballast

> [Go memory ballast: How I learnt to stop worrying and love the heap](https://blog.twitch.tv/ko-kr/2019/04/10/go-memory-ballast-how-i-learnt-to-stop-worrying-and-love-the-heap-26c2462549a2/)

다음과 같이 아주 큰 heap 을 할당해 두자. Virtual Memory 에 10 GB Heap 이 할당된다. Page Fault 가 발생되기 전까지 Physical Memory 를 소모하지는 않는다. 아주 큰 Heap 이 미리 할당되었으므로 GC 를 위한 CPU utilization 은 낮다. 따라서 CPU utilization 을 API 처리에 더욱 사용할 수 있다. API Latency 는 더욱 낮아진다.

```go
func main() {

	// Create a large heap allocation of 10 GiB
	ballast := make([]byte, 10<<30)

	// Application execution continues
	// ...
}
```

## go commands

* [go tool](https://pkg.go.dev/cmd/go)
  * the standard way to fetch, build, and install Go modules, packages, and commands.

----

```
go run
go build
go install
go get
go fmt
go vet
go help build
```

### go build

* [Command go](https://golang.org/cmd/go/)
* [Go Modules - Local Modules](https://jusths.tistory.com/107)
* [Modules](https://github.com/golang/go/wiki/Modules)

----

```bash
$ tree .
.
├── cmd
│   └── main
│       └── main.go
├── go.mod
├── internal
│   └── hello
│       ├── hello.go
│       └── hello_test.go
└── main
```

* `cmd/main/main.go`

```go
package main

import "fmt"
import "iamslash.com/HelloWorld/internal/hello"

func main() {
	fmt.Println(hello.HelloWorld())
	fmt.Println(hello.HelloFoo())
}
```

* `internal/hello/hello.go`

```go
package hello

func HelloWorld() string {
	return "Hello World"
}

func HelloFoo() string {
	return "Hello Foo"
}
```

* `internal/hello/hello_test.go`

```go
package hello

import "testing"

func TestHello(t *testing.T) {
	want := "Hello, world."
	if got := Hello(); got != want {
		t.Errorf("Hello() = %q, want %q", got, want)
	}
}
```

```bash
$ go mod init iamslash.com/HelloWorld
$ cat go.mod
module iamslash.com/HelloWorld

go 1.13

$ go build ./internal/...
$ go build ./internal/hello
$ go build ./internal/hello/
$ go build ./internal/hello/...
$ go build ./cmd/...

$ ./main
$ go run ./cmd/main/main.go
```

### go test

* [Testing in Go: go test](https://ieftimov.com/post/testing-in-go-go-test/)

----

```bash
$ tree .
.
└── internal
    └── person
        ├── person.go
        └── person_test.go
# test all
$ go test -v ./...

# test specific funtion in same package
$ cd internal/person
$ go test -v -run Testxxx_xxx
$ go test -v -run ^Testxxx_xxx$
```

### go mod

* [go mod](go_module.md)

### go generate

[go generate](go_generate.md)

### go-wrk

an HTTP benchmarking tool

```
go-wrk -c 5 -d 5 http://localhost:8080/
```

### go-torch

Tool for stochastically profiling Go programs. Collects stack traces
and synthesizes them into a flame graph. Uses Go's built in pprof
library.

## bazel

* [bazel](/bazel/README.md)

## gazelle

* [gazelle](/gazelle/README.md)

## present

[present](https://godoc.org/golang.org/x/tools/present) is a tool for a slide. You can install like this.

```
go get golang.org/x/tools/cmd/present
```

And you have to make a present `*.slide` with [present format](https://godoc.org/golang.org/x/tools/present). Finally you can see the present like this.

```
present <slide-filename>.slide
```

If you have a github repository you can see that using [talks.godoc.org](https://talks.godoc.org). Just open the browser with the url like this.

```
https://talks.godoc.org/github.com/owner/project/file.ext
https://talks.godoc.org/github.com/owner/project/sub/directory/file.ext
```

This is a major present format.

```
#_italic_
#*bold*
#`program`
#Markup—_especially_italic_text_—can easily be overused.
#_Why_use_scoped__ptr_? Use plain ***ptr* instead.
#.code scmmig/a.go /^func main/,/^}/
#.code scmmig/a.go 
#.play scmmig/a.go
#.image scmmig/a.png
#.background scmmig/a.png
#.iframe http://www.iamslash.com 560 960
#.link http://www.google.co.kr Google
#.html scmmig/a.html
#.caption _Gopher_ by [[https://www.instagram.com/reneefrench/][Renée French]]
```

## Debug

IntelliJ 는 `^D` 로 debugging 시작.

## Testify

* [testify](https://github.com/stretchr/testify)

## Gomock

* [gomock](go_mock.md)

## Benchmarks

```
go test-bench
```

## Profile

[go profile](go_profile.md)

## Vfsgen

[vfsgen](https://github.com/shurcooL/vfsgen) takes an http.FileSystem and generates Go code that statically implements the provided http.FileSystem.

[vfsgen](https://github.com/shurcooL/vfsgen) is bettern than [packr](https://github.com/gobuffalo/packr). Because of the simple dependecy.

* directory

```
.
├── cmd
│   └── main
│       └── main.go
├── go.mod
├── go.sum
└── internal
    └── resource_manager
        └── provision
            ├── assets_generate.go
            ├── assets_vfsdata.go
            ├── template_builder.go
            └── templates
                ├── a.yaml
                ├── b.yaml
                ├── c.yaml
                ├── d.yaml
                ├── e.yaml
                └── f.yaml
```

* `cmd/main/main.go`

```go
package main

import (
	"iamslash.com/alpha/internal/resource_manager/provision"
)

func main() {
	provision.ReadFiles()
	//fmt.Println("Hello World !!!")
}
```

* `internal/reousrce_manager/provision/assets_generate.go`

```go
// +build ignore
//go:generate go run assets_generate.go

package main

import (
	"log"
	"net/http"

	"github.com/shurcooL/vfsgen"
)

func main() {

	var fs http.FileSystem = http.Dir("templates")

	err := vfsgen.Generate(fs, vfsgen.Options{
		PackageName:  "provision",
		// BuildTags:    "!dev",
		VariableName: "Assets",
	})
	if err != nil {
		log.Fatalln(err)
	}
}
```

* generate `assets_vfsdata.go`
  * run generate go file
    ```bash
    $ cd internal/resource_manager/provision
    $ go run assets_generate.go
    ```
  * why `go generate` does not work ???
  * why `vfsgendev` does not work ???

* `internal/resource_manager/provision/template_builder.go`
  * read from `assets_vfsdata.go`
  
```go
package provision

import (
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/shurcooL/httpfs/vfsutil"
)

func ReadFiles() {
	var fs http.FileSystem = assets

	b, err := vfsutil.ReadFile(fs, "/a.yaml")
	if err != nil {
		panic(err)
	}
	strData := string(b)
	fmt.Println(strData)
}
```

* build

```bash
$ go build ./...
$ go build cmd/main/main.go
```

## IntelliJ IDEA

* use plugins such as "Protobuf Support", "Go", "File Watchers".
* set [File Watcher](https://tech.flyclops.com/posts/2016-06-14-goimports-intellij.html) for goimports.

## Managing Multiple go versions

export GOROOT, GOPATH, PATH with specific go version. execute this script with `$ source switch_go_1.14.13.sh`.

* `switch_go_1.14.13.sh`

```bash
#!/usr/bin/env bash

export GOPATH=$HOME/my/gopath-1.14.13
export GOROOT=/Users/davidsun/sdk/go1.14.13
export PATH=$GOROOT/bin:$GOPATH/bin:$PATH
echo "done..."
```

# Effective Go

* [effective go](https://golang.org/doc/effective_go.html)
  * [한글](https://gosudaweb.gitbooks.io/effective-go-in-korean/content/)

# Design Patterns

* [Design Patterns in Go](go_design_pattern.md)
