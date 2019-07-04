- [Abstract](#Abstract)
- [Materials](#Materials)
- [References](#References)
  - [Tools](#Tools)
  - [Best Practices](#Best-Practices)
  - [Concurrency](#Concurrency)
  - [Error Handling](#Error-Handling)
  - [Interface](#Interface)
  - [Struct](#Struct)
  - [Pointer](#Pointer)
  - [Map, Slice](#Map-Slice)
  - [Logging](#Logging)
  - [Encoding, JSON](#Encoding-JSON)
- [Basic Usages](#Basic-Usages)
  - [Hello World](#Hello-World)
  - [Collections compared to c++ containers](#Collections-compared-to-c-containers)
  - [Collections by examples](#Collections-by-examples)
  - [Reserved Words](#Reserved-Words)
  - [Data Types](#Data-Types)
  - [Declarations](#Declarations)
  - [Constants](#Constants)
  - [Operators](#Operators)
    - [Arithmetic](#Arithmetic)
    - [Comparison](#Comparison)
    - [Logical](#Logical)
    - [Other](#Other)
  - [Decision Making](#Decision-Making)
    - [If](#If)
    - [Switch](#Switch)
    - [select](#select)
  - [Loops](#Loops)
  - [Functions](#Functions)
    - [Functions As Values And Closures](#Functions-As-Values-And-Closures)
    - [Variadic Functions](#Variadic-Functions)
  - [Type Conversions](#Type-Conversions)
  - [Packages](#Packages)
  - [Strings](#Strings)
  - [Arrays, Slices, Ranges](#Arrays-Slices-Ranges)
    - [Arrays](#Arrays)
    - [Slices](#Slices)
    - [Operations on Arrays and Slices](#Operations-on-Arrays-and-Slices)
  - [Maps](#Maps)
  - [Structs](#Structs)
  - [Pointers](#Pointers)
  - [Interfaces](#Interfaces)
  - [Embedding](#Embedding)
  - [Errors](#Errors)
  - [Goroutines](#Goroutines)
  - [Channels](#Channels)
    - [Channel Axioms](#Channel-Axioms)
  - [Printing](#Printing)
- [Advanced Usages](#Advanced-Usages)
  - [Tools](#Tools-1)
    - [go](#go)
    - [go-wrk](#go-wrk)
    - [go-torch](#go-torch)
  - [Debug](#Debug)
  - [Test](#Test)
  - [Benchmarks](#Benchmarks)
  - [Profile](#Profile)
- [Snippets](#Snippets)
  - [HTTP Server](#HTTP-Server)
-------------------------------------------------------------------------------

# Abstract

golang에 대해 정리한다. IDE는 VScode가 좋다.

# Materials


* [learn go in Y minutes](https://learnxinyminutes.com/docs/go/)
* [go @ tutorialspoint](https://www.tutorialspoint.com/go/)
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
// go run a.go
```

## Collections compared to c++ containers

| c++                  | go                   | 
|:---------------------|:---------------------|
| `if, else`           | `if, else`           |
| `for, while`         | `for`                |
| `array`              | `array`              |
| `vector`             | `slice`              |
| `deque`              | ``                   |
| `forward_list`       | ``                   |
| `list`               | `list`               |
| `stack`              | ``                   |
| `queue`              | ``                   |
| `priority_queue`     | `heap`               |
| `set`                | `map[keytype]struct{}`|
| `multiset`           | ``                   |
| `map`                | ``                   |
| `multimap`           | ``                   |
| `unordered_set`      | ``                   |
| `unordered_multiset` | ``                   |
| `unordered_map`      | `map`                |
| `unordered_multimap` | ``                   |

## Collections by examples

* array

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

* slice

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

* map

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
}


// $ go run maps.go 
// map: map[k1:7 k2:13]
// v1:  7
// len: 2
// map: map[k1:7]
// prs: false
// map: map[foo:1 bar:2]
```

* heap

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

* list

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

* ring

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

## Reserved Words

```go
break    default     func   interface select
case     defer       go     map       struct
chan     else        goto   package   switch
const    fallthrough if     range     type
continue for         import return    var
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


## Decision Making

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


## Strings

```go
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


# Advanced Usages

## Tools

### go

주로 사용하는 command는 다음과 같다. 도움말은 go help를 이용하자.

```
go run
go build
go install
go get
go fmt
go vet
```

### go-wrk

an HTTP benchmarking tool

```
go-wrk -c 5 -d 5 http://localhost:8080/
```

### go-torch

Tool for stochastically profiling Go programs. Collects stack traces
and synthesizes them into a flame graph. Uses Go's built in pprof
library.

## Debug

VS Code를 사용한다면 debug mode로 launch하자.

## Test

tests from VS Code

code coverage

table driven tests

## Benchmarks

```
go test-bench
```

## Profile

go-torch

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