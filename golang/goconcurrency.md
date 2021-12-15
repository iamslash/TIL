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
	- [Distributed processing](#distributed-processing)
	- [select](#select)
	- [context.Context](#contextcontext)
	- [Paring request, response](#paring-request-response)
	- [Assemble goroutine dynamically](#assemble-goroutine-dynamically)
	- [Anti pattern of goroutine](#anti-pattern-of-goroutine)

-----

# Abstract

go routine 의 pattern 에 대해 정리한다.

# Materials

* [Scheduling In Go : Part I - OS Scheduler](https://www.ardanlabs.com/blog/2018/08/scheduling-in-go-part1.html)
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

다음과 같은 function type 을 선언하면 더욱 간단하게 pipeline 을 구성할 수 있다.
다음은 PluginOne 을 두개 pipelining 해서 PlusTwo 를 만든 예이다.

```go
type IntPipe func(<-chan int) <-chan int

func Chain(ps ...IntPipe) IntPipe {
	return func(in <-chan int) <-chan int {
		c := in
		for _, p := range ps {
			c = p(c)
		}
		return c
	}
}

func ExamplePlusTwo() {
	PlusTwo := Chain(PlusOne, PlusOne)
	c := make(chan int)
	go func() {
		defer close(c)
		c <- 5
		c <- 3
		c <- 8
	}()
	for num := range PlusTwo(c) {
		fmt.Println(num)
	}
	// Output:
	// 7
	// 5
	// 10
}
```

pipeline 과 같이 function 이 여러개 중첩되는 경우 복잡해 보일 수 있다.
그러나 function 의 input, output arguement 의 type 이 무엇인지를 확실히 이해하고
top to the bottom 방향으로 이해하면 수월하게 접근할 수 있다. 항상 high level 에서 먼저
이해하는 것이 중요하다.

## Fan out

하나의 출력이 여러 입력으로 흘러가는 형태를 Fan out 이라고 한다. 다음은 하나의 출력을
3 개의 goroutine 의 입력으로 흘러가서 각각 출력되는 예이다. context switching 을 위해 time.Sleep 을 사용했다. goroutine 에서 input argument 로 i 를 넘기지 않고 lexical binding 한 i 를 사용하면 race condition 이 발생되어 잘 못된 i 를 출력하게 된다. 반드시 host code 의 local variable 을 lexical binding 하지 않고 goroutine 에 input argument 로 전달하도록 하자.

```go
func main() {
	c := make(chan int)
	for i := 0; i < 3; i++ {
		go func(i int) {
			for n := range c {
				time.Sleep(1)
				fmt.Println(i, n)
			}
		}(i)
	}
	for i := 0; i < 10; i++ {
		c <- i
	}
	close(c)
}

// Output
// 1 2
// 2 1
// 0 0
// 1 3
// 2 4
// 1 6
// 0 5
// 2 7
```

## Fan in

여러개의 출력을 하나의 입력으로 흘러가는 형태를 Fan in 이라고 한다.

```go
func FanIn(ins ...<-chan int) <-chan int {
	out := make(chan int)
	var wg sync.WaitGroup
	wg.Add(len(ins))
	for _, in := range ins {
		go func(in <-chan int) {
			defer wg.Done()
			for num := range in {
				out <- num
			}
		}(in)
	}
	go func() {
		wg.Wait()
		close(out)
	}()
	return out
}
```

## Distributed processing

하나의 출력을 Fan out 하고 Fan in 하는 것을 Distributed processing 이라고 한다.

```go
func Distribute(p IntPipe, n int) IntPipe {
	return func(in <-chan int) <-chan int {
		cs := make([]<-chan int, n)
		for i := 0; i < n; i++ {
			cs[i] = p(in)
		}
		return FanIn(cs...)
	}
}
```

Distribute 와 Chain 을 이용하면 다양한 pipeline 을 구성할 수 있다.

```go
out := Chain(Cut, Distribute(Chain(Draw, Paint, Decorate), 10), Box)(in)
```

위와 같이 하면 in 으로 들어온 data 가 다음의 흐름을 거친다.

* 하나의 Cut goroutine 을 거친다. (1)
* 10 개로 나누어져 각각 Draw, Paint, Decorate 의 pipeline 을 거친다. (30)
* 하나의 Box goroutine 으로 합쳐진다. (1)

goroutine 의 개수는 대략 32 개가 된다.

다음은 또 다른 pipeline 이다.

```go
out := Chain(Cut, Distribute(Draw, 6), Distribute(Paint, 10), Distribute(Decorate, 3), Box)(in)
```

위와 같이 하면 in 으로 들어온 data 가 다음의 흐름을 거친다.

* 하나의 Cut goroutine 을 거친다. (1)
* 6 개의 Draw goroutine 으로 Fan out 했다가 Fan in 한다. (6)
* 10 개의 Paint goroutine 으로 Fan out 했다가 Fan in 한다. (10)
* 3 개의 Decorate goroutine 으로 Fan out 했다가 Fan in 한다. (3)
* 하나의 Box goroutine 을 거친다. (1)

goroutine 의 개수는 대략 21 개이다.

goroutine 은 user level thread 이다. context switching 의 부담이 적다. 많이 사용해도 된다.

## select

select 를 이용하면 동시에 여러 channel 과 통신할 수 있다.

```go
package main

import (
  "fmt"
  "time"
)

func main() {
  c1 := make(chan string)
  c2 := make(chan string)

  go func() {
    time.Sleep(1 * time.Second)
    c1 <- "one"
  }()
  go func() {
    time.Sleep(2 * time.Second)
    c2 <- "two"
  }()

  for i := 0; i < 2; i++ {
    select {
    case msg1 := <-c1:
      fmt.Println("received", msg1)
    case msg2 := <-c2:
      fmt.Println("received", msg2)
    }
  }
}
```

앞서 구현한 Fan in 을 여러개의 goroutine 을 사용하지 않고 select 로 구현해 보자.

```go
package main

import "fmt"

func FanIn3(in1, in2, in3 <-chan int) <-chan int {
	out := make(chan int)
	openCnt := 3
	closeChan := func(c *<-chan int) bool {
		*c = nil
		openCnt--
		return openCnt == 0
	}
	go func() {
		defer close(out)
		for {
			select {
			case n, ok := <-in1:
				if ok {
					out <- n
				} else if closeChan(&in1) {
					return
				}
			case n, ok := <-in2:
				if ok {
					out <- n
				} else if closeChan(&in2) {
					return
				}
			case n, ok := <-in3:
				if ok {
					out <- n
				} else if closeChan(&in3) {
					return
				}
			}
		}
	}()
	return out
}

func main() {
	c1, c2, c3 := make(chan int), make(chan int), make(chan int)
	sendInts := func(c chan<- int, begin, end int) {
		defer close(c)
		for i := begin; i < end; i++ {
			c <- i
		}
	}
	go sendInts(c1, 11, 14)
	go sendInts(c2, 21, 23)
	go sendInts(c3, 31, 35)
	for n := range FanIn3(c1, c2, c3) {
		fmt.Println(n, ",")
	}
}
```

닫힌 채널은 nil 로 저장했다. nil 채널은 보내기 및 받기가 모두 blocking 된다.

또한 default 를 이용하여 select 를 non-blocking 으로 구현할 수 있다.

```go
select {
  case n := <-c:
    fmt.Println(n)
  default:
    fmt.Println("Data is not ready. Skipping...")
}
```

time.After 를 이용하면 일정시간동안 기다리게 할 수도 있다.

```golang
timeout := time.After(5 * time.Second)
for {
  select {
    case n := <-recv:
      fmt.Println(n)
    case send <- 1:
      fmt.Println("sent 1")
    case <-timeout:
      fmt.Println("Communication wasn't finisehd in 5 sec.")
      return
  }
}
```

## context.Context

* [Go Concurrency Patterns: Context](https://blog.golang.org/context)

-----

context.Cotext 를 이용하면 goroutine 을 중지시킬 수 있다. 다음과 같이 context.Context 를 설치한다.

```console
$ go get golang.org/x/net/context
```

다음은 context.Context 를 이용하여 goroutine 을 취소할 수 있는 예이다.

```go
package main

import (
	"context"
	"fmt"
)

func PlusOne(ctx context.Context, in <-chan int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for num := range in {
			select {
			case out <- num + 1:
			case <-ctx.Done():
				return
			}
		}
	}()
	return out
}

func main() {
	c := make(chan int)
	go func() {
		defer close(c)
		for i := 3; i < 103; i += 10 {
			c <- i
		}
	}()
	ctx, cancel := context.WithCancel(context.Background())
	nums := PlusOne(ctx, PlusOne(ctx, PlusOne(ctx, PlusOne(ctx, c))))
	for num := range nums {
		fmt.Println(num)
		if num == 17 {
			cancel()
			break
		}
	}
}
// Output
// 7
// 17
```

## Paring request, response

특정한 request, response 를 짝지어 주기 위해서는 request, response 에 id field 가 필요하다.

다음은 request, response 를 Request, Response struct 를 이용하여 구현했다. 물론 각각 id field 를 포함한다.

```go
package main

import (
	"fmt"
	"sync"
)

type Request struct {
	Num  int
	Resp chan Response
}
type Response struct {
	Num      int
	WorkerId int
}

func PlusOneService(reqs <-chan Request, workerID int) {
	for req := range reqs {
		go func(req Request) {
			defer close(req.Resp)
			req.Resp <- Response{req.Num + 1, workerID}
		}(req)
	}
}

func main() {
	reqs := make(chan Request)
	defer close(reqs)
	for i := 0; i < 3; i++ {
		go PlusOneService(reqs, i)
	}
	var wg sync.WaitGroup
	for i := 3; i < 53; i += 10 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			resps := make(chan Response)
			reqs <- Request{i, resps}
			fmt.Println(i, "=>", <-resps)
		}(i)
	}
	wg.Wait()
}
```

## Assemble goroutine dynamically

* [A concurrent prime sieve](https://golang.org/doc/play/sieve.go)

----

sieve of eratosthenes 를 goroutine 으로 구현해 보자. 다음은 goroutine 을 동적으로 이어 붙여서 소수를 찾는 예이다.

```go
// A concurrent prime sieve

package main

import "fmt"

// Send the sequence 2, 3, 4, ... to channel 'ch'.
func Generate(ch chan<- int) {
	for i := 2; ; i++ {
		ch <- i // Send 'i' to channel 'ch'.
	}
}

// Copy the values from channel 'in' to channel 'out',
// removing those divisible by 'prime'.
func Filter(in <-chan int, out chan<- int, prime int) {
	for {
		i := <-in // Receive value from 'in'.
		if i%prime != 0 {
			out <- i // Send 'i' to 'out'.
		}
	}
}

// The prime sieve: Daisy-chain Filter processes.
func main() {
	ch := make(chan int) // Create a new channel.
	go Generate(ch)      // Launch Generate goroutine.
	for i := 0; i < 10; i++ {
		prime := <-ch
		fmt.Println(prime)
		ch1 := make(chan int)
		go Filter(ch, ch1, prime)
		ch = ch1
	}
}
```

이제 generator, context.Context 등을 이용하여 prime number generator 를 만들어 보자.

```go
package main

import (
	"context"
	"fmt"
)

type IntPipe func(ctx context.Context, in <-chan int) <-chan int

func Range(ctx context.Context, start, step int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for i := start; ; i += step {
			select {
			case out <- i:
			case <-ctx.Done():
				return
			}
		}
	}()
	return out
}

func FilterMultiple(n int) IntPipe {
	return func(ctx context.Context, in <-chan int) <-chan int {
		out := make(chan int)
		go func() {
			defer close(out)
			for x := range in {
				if x%n == 0 {
					continue
				}
				select {
				case out <- x:
				case <-ctx.Done():
					return
				}
			}
		}()
		return out
	}
}

func Primes(ctx context.Context) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		c := Range(ctx, 2, 1)
		for {
			select {
			case i := <-c:
				c = FilterMultiple(i)(ctx, c)
				select {
				case out <- i:
				case <-ctx.Done():
					return
				}
			case <-ctx.Done():
				return
			}
		}
	}()
	return out
}

func PrintPrimes(max int) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	for prime := range Primes(ctx) {
		if prime > max {
			break
		}
		fmt.Print(prime, " ")
	}
	fmt.Println()
}

func main() {
	PrintPrimes(100)
}
```

## Anti pattern of goroutine

다음의 예에서 두번째 goroutine 은 consumer 이다. 두번째 goroutine 은 끝나지 않는다. 이 코드가 반복적으로 실행되면 goroutine 은 무한히 증가할 것이다.

```go
package main

import (
	"fmt"
)

func NeverStop() {
	c := make(chan int)
	done := make(chan bool)
	go func() {
		for i := 0; i < 10; i++ {
			c <- i
		}
		done <- true
	}()
	go func() {
		for {
			fmt.Println(<-c)
		}
	}()
	<-done
}

func main() {
	NeverStop()
}
```
