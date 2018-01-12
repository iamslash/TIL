# Abstract

golang에 대해 정리한다. IDE는 VScode가 좋다.

1. [Hello World](#hello-world)
2. [Operators](#operators)
  * [Arithmetic](#arithmetic)
  * [Comparison](#comparison)
  * [Logical](#logical)
  * [Other](#other)
3. [Declarations](#declarations)
4. [Functions](#functions)
  * [Functions as values and closures](#functions-as-values-and-closures)
  * [Variadic Functions](#variadic-functions)
5. [Built-in Types](#built-in-types)
6. [Type Conversions](#type-conversions)
7. [Packages](#packages)
8. [Control structures](#control-structures)
  * [If](#if)
  * [Loops](#loops)
  * [Switch](#switch)
9. [Arrays, Slices, Ranges](#arrays-slices-ranges)
  * [Arrays](#arrays)
  * [Slices](#slices)
  * [Operations on Arrays and Slices](#operations-on-arrays-and-slices)
10. [Maps](#maps)
11. [Structs](#structs)
12. [Pointers](#pointers)
13. [Interfaces](#interfaces)
14. [Embedding](#embedding)
15. [Errors](#errors)
16. [Concurrency](#concurrency)
  * [Goroutines](#goroutines)
  * [Channels](#channels)
  * [Channel Axioms](#channel-axioms)
17. [Printing](#printing)
18. [Snippets](#snippets)
  * [Http-Server](#http-server)

# Materials

* [1ambda golang](https://github.com/1ambda/golang)
  * 유용한 go links
* [effective go](https://golang.org/doc/effective_go.html)

## Tutorial

* [golang cheatsheet](https://github.com/a8m/go-lang-cheat-sheet)
  * 최고의 요약
* [A Tour of Go video](https://research.swtch.com/gotour)
  * interface, concurrency 에 관한 screencast

## Language 

* [Rob Pike: Simplicity is Complicated](https://www.youtube.com/watch?v=rFejpH_tAHM)
* [Rob Pike: Go at Google](https://www.infoq.com/presentations/Go-Google)
  * [Article](https://talks.golang.org/2012/splash.article)
* [Golang FAQ: Design](https://golang.org/doc/faq#Design)
* [Golang FAQ: Types](https://golang.org/doc/faq#types)
* [Campoy: Functional Go?](https://www.youtube.com/watch?v=ouyHp2nJl0I)

## Tools

* [Go Tooling in Action](https://www.youtube.com/watch?v=uBjoTxosSys)
  * go run, go build, go install, go test, go list, go doc, go-wrk,
    go-torch, debugging등등을 다루는 킹왕짱 동영상

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

# References

* [golang doc](https://golang.org/doc/)

# Language

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
| `\       | `                   | bitwise or |
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

# Examples

## Building a simple web server

net/http, errcheck

## regexp

regexp
