- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
  - [Concurrency & parallelism](#concurrency--parallelism)
  - [goroutine](#goroutine)
  - [syn.WaitGroup](#synwaitgroup)
  - [channel](#channel)
  - [generator pattern with channel](#generator-pattern-with-channel)
  - [How to debug concurrency](#how-to-debug-concurrency)
  - [atomic](#atomic)
- [sync.Mutex](#syncmutex)
- [sync.RWMutex](#syncrwmutex)
- [sync.Once](#synconce)
- [go routine patterns](#go-routine-patterns)
  - [Pipeline pattern](#pipeline-pattern)
  - [Fan out](#fan-out)
  - [Fan in](#fan-in)
  - [distribute](#distribute)
  - [select](#select)
  - [Stop pipeline](#stop-pipeline)
  - [context.Context](#contextcontext)
  - [Paring request, response](#paring-request-response)
  - [Assemble go routine dynamically](#assemble-go-routine-dynamically)

-----

# Abstract

go routine 의 pattern 에 대해 정리한다.

# Materials

* [Rob Pike: Concurrency is not Parallelism @ youtube](https://www.youtube.com/watch?v=B9lP-E4J_lc)
  * [Slide](https://talks.golang.org/2012/waza.slide)
* [Go Concurrency Patterns](https://www.youtube.com/watch?v=f6kdp27TYZs)
* [Curious Channels](https://dave.cheney.net/2013/04/30/curious-channels)
* [Complex Concurrency Patterns With Go](https://www.youtube.com/watch?v=2HOO5gIgyMg)
* [Advanced Go Concurrency Patterns](https://www.youtube.com/watch?v=QDDwwePbDtw)

# Basic

## Concurrency & parallelism

A, B 두 개의 일을 물리적으로 동시에 진행하는 것을 parallelism (병행성) 이라고 한다. A, B 두 개의 일을 논리적으로 동시에 진행하는 것을 Concurrency (동시성) 이라고 한다.

예를 들어 신문을 보는 것을 A, 커피를 마시는 것을 B 라고 하자. A 를 잠깐 하다가 B 를 잠깐 하다가 다시 A 를 할 수 잇기 때문에 Concurrency 가 있다고 할 수 있다. 또한 신문을 보면서 커피를 마실 수 있기 때문에 Parallism 이 있다고 할 수 있다. 

다른 예를 들어 양말을 신는 것을 A, 신발을 신는 것을 B 라고 하자. B 를 하고 나면 A 를 할 수 없으므로 Concurrency 가 없다고 할 수 있다. 역시 물리적으로 B 다음 A 를 할 수 없으므로 Parallelism 은 없다고 할 수 있다.

이와 같이 Concurrency 가 존재해야 Parallelism 이 존재할 수 있다.

## goroutine

* [go concurrency](https://github.com/jaeyeom/gogo/tree/master/concurrency)

go routine 은 user level thread 이다. n 개의 go routine 은 1 개의 kernel thread 와 mapping 된다. 

go routine 및 `WaitGroup` 을 사용하여 최소값을 찾는 함수를 작성해 보자. WaitGroup 은 `Add, Done, Wait` 으로 go routine 의 흐름을 제어한다.

```go
func Min(a []int) int {
	if len(a) == 0 {
		return 0
	}
	min := a[0]
	for _, e := range a[1:] {
		if min > e {
			min = e
		}
	}
	return min
}

func ParallelMin(a []int, n int) int {
	if len(a) < n {
		return Min(a)
	}
	mins := make([]int, n)
	bucketSize := (len(a) + n - 1) / n
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			begin, end := i*bucketSize, (i+1)*bucketSize
			if end > len(a) {
				end = len(a)
			}
			mins[i] = Min(a[begin:end])
		}(i)
	}
	wg.Wait()
	return Min(mins)
}
```

## syn.WaitGroup

WaitGroup makes it possible to wait for multiple goroutines to finish. `Add` increase one, `Done` decrease one, `Wait` waits until the count is zero.

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

func worker(id int, wg *sync.WaitGroup) {
    defer wg.Done()
    fmt.Printf("Worker %d starting\n", id)
    time.Sleep(time.Second)
    fmt.Printf("Worker %d done\n", id)
}

func main() {
    var wg sync.WaitGroup
    for i := 1; i <= 5; i++ {
        wg.Add(1)
        go worker(i, &wg)
    }
    wg.Wait()
}
```

## channel

channel 은 buffer 와 lock 을 포함한 data sturucture 이다. 
* channel 은 항상 보내는 쪽에서 닫아야 한다. 
* closed channel 을 다시 close 하면 panic 이 발생한다. 
* closed channel 을 읽어들이면 기본값과 false 가 넘어온다. 
 

다음은 양방향 채널의 에이다.

```go
package main

import "fmt"

func main() {
  messages := make(chan string)
  go func() { 
    messages <- "ping" 
  }()
  msg := <-messages
  fmt.Println(msg)
}
```

또한 send, receive 와 같이 단방향 channel 을 만들 수도 있다.

```go
// receive only channel
var a <-chan string = make(chan string)
fmt.Println(<-a)
// cannot close receive-only channel

// send only channel
var b chan<- string = make(chan string)
b <- "hello"
close(b)
```

channel 의 내용은 range, select 등으로 가져오자.

```go
package main

import "fmt"

func main() {
	c := make(chan int)
	go func() {
		c <- 1
		c <- 2
		c <- 3
		close(c)
	}()
	for num := range c {
		fmt.Println(num)
	}
}
```

select 을 이용하면 default clause 를 이용하여 non-blocking sneds, receives 가 가능하다.

```go
package main

import "fmt"

func main() {
    messages := make(chan string)
    signals := make(chan bool)

    select {
    case msg := <-messages:
        fmt.Println("received message", msg)
    default:
        fmt.Println("no message received")
    }

    msg := "hi"
    select {
    case messages <- msg:
        fmt.Println("sent message", msg)
    default:
        fmt.Println("no message sent")
    }

    select {
    case msg := <-messages:
        fmt.Println("received message", msg)
    case sig := <-signals:
        fmt.Println("received signal", sig)
    default:
        fmt.Println("no activity")
    }
// Output
// no message received
// no message sent
// no activity
}
```

## generator pattern with channel

closure 를 이용해서 Fibonacci generator 를 만들어 보자.

```go
func FibonacciGenerator(max int) func() int {
  next, a, b := 0, 0, 1
  return funct() int {
    next, a, b = a, b, a+b
    if next > max {
      return -1
    }
    return next
  }
}

func ExampleFibonacciGenerator() {
  fib := FibonacciGenerator(15)
  for n := fib(); n >= 0; n = fib() {
    fmt.Println(n, ",")
  }
  // Output:  
  // 0,1,1,2,3,5,8,13
}
```

이제 channel 을 이용해서 Fibonacci generator 를 만들어 보자.

```go
func Fibonacci(max int) <-chan int {
  c := make(chan int)
  go func() {
    defer close(c)
    a, b := 0, 1
    for a <= max {
      c <- a
      a, b = b, a+b
    }
  }()
  return c
}

func ExampleFibonacci() {
  for fib := range Fibonacci(15) {
    fmt.Println(fib, ",")
  }
  // Output:
  // 0,1,1,2,3,5,8,13,
}
```

channel generator 가 closure generator 보다 다음과 같은 장점을 갖는다.

* producer 입장에서 상태 저장 방법을 단순히 처리할 수 있다.
* consumer 입장에서 for, range 를 사용할 수 있어서 구현이 용이하다.
* channel buffer 를 이용하면 multi core 를 활용하거나 입출력 성능이 좋아진다.

## How to debug concurrency

* [Data Race Detector](https://golang.org/doc/articles/race_detector.html)

```console
$ go test -race foo
$ go run -race foo.go
$ go build -race foo
$ go install -race foo
```

## atomic

atomic is a thread safe counter.

```go
package main

import (
    "fmt"
    "sync"
    "sync/atomic"
)

func main() {
    var ops uint64
    var wg sync.WaitGroup

    for i := 0; i < 50; i++ {
        wg.Add(1)
        go func() {
            for c := 0; c < 1000; c++ {
                atomic.AddUint64(&ops, 1)
            }
            wg.Done()
        }()
    }

    wg.Wait()
    fmt.Println("ops:", ops)
}
```

# sync.Mutex

```go
package main

import (
  "fmt"
  "math/rand"
  "sync"
  "sync/atomic"
  "time"
)

func main() {
  var state = make(map[int]int)
  var mutex = &sync.Mutex{}
  var readOps uint64
  var writeOps uint64

  for r := 0; r < 100; r++ {
    go func() {
      total := 0
      for {
        key := rand.Intn(5)
        mutex.Lock()
        total += state[key]
        mutex.Unlock()
        atomic.AddUint64(&readOps, 1)
        time.Sleep(time.Millisecond)
      }
    }()
  }

  for w := 0; w < 10; w++ {
    go func() {
      for {
        key := rand.Intn(5)
        val := rand.Intn(100)
        mutex.Lock()
        state[key] = val
        mutex.Unlock()
        atomic.AddUint64(&writeOps, 1)
        time.Sleep(time.Millisecond)
      }
    }()
  }

  time.Sleep(time.Second)

  readOpsFinal := atomic.LoadUint64(&readOps)
  fmt.Println("readOps:", readOpsFinal)
  writeOpsFinal := atomic.LoadUint64(&writeOps)
  fmt.Println("writeOps:", writeOpsFinal)

  mutex.Lock()
  fmt.Println("state:", state)
  mutex.Unlock()
}
```

# sync.RWMutex

sync.Mutex 와 유사하다. 그러나 읽기동작과 쓰기 동작을 나누어서 잠금처리할 수 있다.

```go
func (rw *RWMutex) Lock(): 쓰기 잠금
func (rw *RWMutex) Unlock(): 쓰기 잠금 해제
func (rw *RWMutex) RLock(): 읽기 잠금
func (rw *RWMutex) RUnlock(): 읽기 잠금 해제
```

읽기 잠금하면 다른 go routine 에서 읽기는 되지만 쓰기는 못한다.
쓰기 잠금하면 다른 go routine 에서 읽기, 쓰기 모두 할 수 없다.

# sync.Once

Execute a function just one time.

```go
func ExampleOnce() {
	var once sync.Once
	onceBody := func() {
		fmt.Println("Only once")
	}
	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func() {
			once.Do(onceBody)
			done <- true
		}()
	}
	for i := 0; i < 10; i++ {
		<-done
	}
	// Output:
	// Only once
}
```

# go routine patterns

## Pipeline pattern

* [Go Concurrency Patterns: Pipelines and cancellation](https://blog.golang.org/pipelines)

-----

pipeline 은 한 단계의 출력이 다음 단계의 입력으로 이어지는 구조를 말한다. channel 을 입력과 출력으로 하는 function 을 만들자. 그리고 그 function 들을 연결하면 pipeline 이 된다.

```go
package main

import "fmt"

func PlusOne(in <-chan int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for num := range in {
			out <- num + 1
		}
	}()
	return out
}

func ExamplePlusOne() {
	c := make(chan int)
	go func() {
		defer close(c)
		c <- 5
		c <- 3
		c <- 8
	}()
	for num := range PlusOne(PlusOne(c)) {
		fmt.Println(num)
	}
	// Output:
	// 7
	// 5
	// 10
}
```

## Fan out

## Fan in

## distribute

## select

## Stop pipeline

## context.Context

## Paring request, response

## Assemble go routine dynamically




