- [Abstract](#abstract)
- [Materials](#materials)
- [References](#references)
  - [Tools](#tools)
  - [Best Practices](#best-practices)
  - [Concurrency](#concurrency)
  - [Error Handling](#error-handling)
  - [Interface](#interface)
  - [Struct](#struct)
  - [Pointer](#pointer)
  - [Map, Slice](#map-slice)
  - [Logging](#logging)
  - [Encoding, JSON](#encoding-json)
- [Basic Usages](#basic-usages)
  - [Hello World](#hello-world)
  - [Collections compared to c++ containers](#collections-compared-to-c-containers)
  - [Collections by examples](#collections-by-examples)
  - [Operators](#operators)
    - [Arithmetic](#arithmetic)
    - [Comparison](#comparison)
    - [Logical](#logical)
    - [Other](#other)
  - [Declarations](#declarations)
  - [Functions](#functions)
    - [Functions As Values And Closures](#functions-as-values-and-closures)
    - [Variadic Functions](#variadic-functions)
  - [Built-in Types](#built-in-types)
  - [Type Conversions](#type-conversions)
  - [Packages](#packages)
  - [Control structures](#control-structures)
    - [If](#if)
    - [Loops](#loops)
    - [Switch](#switch)
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
  - [Printing](#printing)
- [Snippets](#snippets)
  - [HTTP Server](#http-server)
- [Tools](#tools-1)
  - [go](#go)
  - [go-wrk](#go-wrk)
  - [go-torch](#go-torch)
- [Debug](#debug)
- [Test](#test)
- [Benchmarks](#benchmarks)
- [Profile](#profile)

-------------------------------------------------------------------------------

# Abstract

golang에 대해 정리한다. IDE는 VScode가 좋다.

# Materials

* [가장 빨리 만나는 Go 언어](http://pyrasis.com/go.html)
  * 킹왕짱 golang 기본문법
  * [src](https://github.com/pyrasis/golangbook)
* [1ambda golang](https://github.com/1ambda/golang)
  * 유용한 go links
* [effective go](https://golang.org/doc/effective_go.html)
* [Go Bootcamp](http://www.golangbootcamp.com/book/collection_types)
  * 예제위주의 책
* [예제로 배우는 GO프로그래밍](http://golang.site/)
  * 최고의 한글 예제들
* [go by example](https://gobyexample.com/)
  * 최고의 예제들
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


# References

* [golang doc](https://golang.org/doc/)

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

## Encoding, JSON

- [JSON, interface, and go generate](https://www.youtube.com/watch?v=YgnD27GFcyA)


# Basic Usages

## Hello World

* a.go

```go
package main
import "fmt"
func main() {
    fmt.Println("Hello World")
}
```

`go run a.go`

## Collections compared to c++ containers

## Collections by examples

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
| `|`      | bitwise or          |            |
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
| `\       | \           | ` | logical or |
| `!`      | logical not |   |            |

### Other

| Operator | Description                                    |
|----------|------------------------------------------------|
| `&`      | address of / create pointer                    |
| `*`      | dereference pointer                            |
| `<-`     | send / receive operator (see 'Channels' below) |

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

## Built-in Types

```
bool

string

int  int8  int16  int32  int64
uint uint8 uint16 uint32 uint64 uintptr

byte // alias for uint8

rune // alias for int32 ~= a character (Unicode code point) - very Viking

float32 float64

complex64 complex128
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

## Control structures

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

### Loops
```go
    // There's only `for`, no `while`, no `until`
    for i := 1; i < 10; i++ {
    }
    for ; i < 10;  { // while - loop
    }
    for i < 10  { // you can omit semicolons if there is only a condition
    }
    for { // you can omit the condition ~ while (true)
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

```

**Anonymous structs:**  
Cheaper and safer than using `map[string]interface{}`.
```go
point := struct {
	X, Y int
}{1, 2}
```

## Pointers

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

## Printing

```go
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

# Snippets

## HTTP Server
```go
package main

import (
    "fmt"
    "net/http"
)

// define a type for the response
type Hello struct{}

// let that type implement the ServeHTTP method (defined in interface http.Handler)
func (h Hello) ServeHTTP(w http.ResponseWriter, r *http.Request) {
    fmt.Fprint(w, "Hello!")
}

func main() {
    var h Hello
    http.ListenAndServe("localhost:4000", h)
}

// Here's the method signature of http.ServeHTTP:
// type Handler interface {
//     ServeHTTP(w http.ResponseWriter, r *http.Request)
// }
```

# Tools

## go

주로 사용하는 command는 다음과 같다. 도움말은 go help를 이용하자.

```
go run
go build
go install
go get
go fmt
go vet
```

## go-wrk

an HTTP benchmarking tool

```
go-wrk -c 5 -d 5 http://localhost:8080/
```

## go-torch

Tool for stochastically profiling Go programs. Collects stack traces
and synthesizes them into a flame graph. Uses Go's built in pprof
library.

# Debug

VS Code를 사용한다면 debug mode로 launch하자.

# Test

tests from VS Code

code coverage

table driven tests

# Benchmarks

```
go test-bench
```

# Profile

go-torch
