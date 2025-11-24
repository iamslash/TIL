- [Coroutine](#coroutine)
  - [Java](#java)
    - [Virtual Threads (Java 19+)](#virtual-threads-java-19)
    - [CompletableFuture (Java 8+)](#completablefuture-java-8)
  - [Kotlin](#kotlin)
    - [Basic Coroutine](#basic-coroutine)
    - [Async/Await Pattern](#asyncawait-pattern)
    - [Multiple Coroutines](#multiple-coroutines)
  - [Swift](#swift)
    - [Basic Async/Await](#basic-asyncawait)
    - [Multiple Concurrent Tasks](#multiple-concurrent-tasks)
    - [Task Group](#task-group)
  - [Python 3](#python-3)
    - [Basic Async/Await](#basic-asyncawait-1)
    - [Multiple Coroutines](#multiple-coroutines-1)
    - [Async Generator](#async-generator)
  - [TypeScript](#typescript)
    - [Basic Async/Await](#basic-asyncawait-2)
    - [Multiple Promises](#multiple-promises)
    - [Async Generator](#async-generator-1)
  - [C++](#c)
    - [Basic Coroutine](#basic-coroutine-1)
    - [Generator Pattern](#generator-pattern)
  - [C](#c-1)
    - [Using setjmp/longjmp (Simple Example)](#using-setjmplongjmp-simple-example)
    - [Using libcoro (Third-party Library)](#using-libcoro-third-party-library)
  - [C#](#c-2)
    - [Basic Async/Await](#basic-asyncawait-3)
    - [Multiple Tasks](#multiple-tasks)
    - [Async Generator](#async-generator-2)
  - [Go](#go)
    - [Basic Goroutine](#basic-goroutine)
    - [Using Channels](#using-channels)
    - [Select Statement](#select-statement)
  - [Rust](#rust)
    - [Basic Async/Await (with tokio)](#basic-asyncawait-with-tokio)
    - [Multiple Tasks](#multiple-tasks-1)
    - [Stream (Async Iterator)](#stream-async-iterator)
  - [Summary](#summary)

--------

# Coroutine

Coroutine은 서브루틴의 일반화된 형태로, 실행을 일시 중단하고 나중에 재개할 수 있는 함수입니다. 다양한 프로그래밍 언어에서 coroutine을 구현하는 방법을 정리합니다.

## Java

Java는 전통적으로 coroutine을 직접 지원하지 않았지만, Java 19부터 Project Loom의 Virtual Threads가 도입되었습니다. 또한 `CompletableFuture`를 사용한 비동기 프로그래밍도 가능합니다.

### Virtual Threads (Java 19+)

```java
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class CoroutineExample {
    public static void main(String[] args) {
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            Future<String> future = executor.submit(() -> {
                System.out.println("Task 1 running on: " + Thread.currentThread());
                Thread.sleep(1000);
                return "Task 1 completed";
            });
            
            System.out.println("Main thread continues");
            System.out.println(future.get());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

### CompletableFuture (Java 8+)

```java
import java.util.concurrent.CompletableFuture;

public class AsyncExample {
    public static void main(String[] args) {
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> {
            try {
                Thread.sleep(1000);
                return "Async task completed";
            } catch (InterruptedException e) {
                return "Interrupted";
            }
        });
        
        future.thenAccept(result -> System.out.println(result));
        System.out.println("Main thread continues");
        
        // Wait for completion
        future.join();
    }
}
```

## Kotlin

Kotlin은 언어 레벨에서 coroutine을 지원합니다. `kotlinx.coroutines` 라이브러리를 사용합니다.

### Basic Coroutine

```kotlin
import kotlinx.coroutines.*

fun main() = runBlocking {
    // Launch a coroutine
    launch {
        delay(1000L)
        println("World!")
    }
    println("Hello,")
    delay(2000L)
}
```

### Async/Await Pattern

```kotlin
import kotlinx.coroutines.*

suspend fun fetchData(): String {
    delay(1000L)
    return "Data from network"
}

fun main() = runBlocking {
    val deferred = async { fetchData() }
    println("Waiting for data...")
    val result = deferred.await()
    println(result)
}
```

### Multiple Coroutines

```kotlin
import kotlinx.coroutines.*

suspend fun task(name: String, delay: Long): String {
    delay(delay)
    return "Task $name completed"
}

fun main() = runBlocking {
    val jobs = listOf(
        async { task("A", 1000L) },
        async { task("B", 500L) },
        async { task("C", 1500L) }
    )
    
    jobs.forEach { println(it.await()) }
}
```

## Swift

Swift는 `async/await`와 structured concurrency를 지원합니다.

### Basic Async/Await

```swift
import Foundation

func fetchData() async -> String {
    try? await Task.sleep(nanoseconds: 1_000_000_000)
    return "Data fetched"
}

func main() async {
    print("Starting...")
    let result = await fetchData()
    print(result)
}

Task {
    await main()
}
```

### Multiple Concurrent Tasks

```swift
import Foundation

func task(name: String, delay: UInt64) async -> String {
    try? await Task.sleep(nanoseconds: delay * 1_000_000_000)
    return "Task \(name) completed"
}

func main() async {
    async let task1 = task(name: "A", delay: 1)
    async let task2 = task(name: "B", delay: 2)
    async let task3 = task(name: "C", delay: 1)
    
    let results = await [task1, task2, task3]
    results.forEach { print($0) }
}

Task {
    await main()
}
```

### Task Group

```swift
import Foundation

func processItem(_ item: Int) async -> Int {
    try? await Task.sleep(nanoseconds: 500_000_000)
    return item * 2
}

func main() async {
    await withTaskGroup(of: Int.self) { group in
        for i in 1...5 {
            group.addTask {
                await processItem(i)
            }
        }
        
        for await result in group {
            print("Result: \(result)")
        }
    }
}

Task {
    await main()
}
```

## Python 3

Python은 `async/await` 키워드와 `asyncio` 모듈을 사용하여 coroutine을 지원합니다.

### Basic Async/Await

```python
import asyncio

async def fetch_data():
    await asyncio.sleep(1)
    return "Data fetched"

async def main():
    print("Starting...")
    result = await fetch_data()
    print(result)

asyncio.run(main())
```

### Multiple Coroutines

```python
import asyncio

async def task(name, delay):
    await asyncio.sleep(delay)
    return f"Task {name} completed"

async def main():
    tasks = [
        task("A", 1),
        task("B", 2),
        task("C", 1)
    ]
    
    results = await asyncio.gather(*tasks)
    for result in results:
        print(result)

asyncio.run(main())
```

### Async Generator

```python
import asyncio

async def count_up_to(n):
    for i in range(1, n + 1):
        await asyncio.sleep(0.5)
        yield i

async def main():
    async for number in count_up_to(5):
        print(f"Count: {number}")

asyncio.run(main())
```

## TypeScript

TypeScript는 JavaScript의 `async/await`를 사용합니다.

### Basic Async/Await

```typescript
async function fetchData(): Promise<string> {
    await new Promise(resolve => setTimeout(resolve, 1000));
    return "Data fetched";
}

async function main() {
    console.log("Starting...");
    const result = await fetchData();
    console.log(result);
}

main();
```

### Multiple Promises

```typescript
async function task(name: string, delay: number): Promise<string> {
    await new Promise(resolve => setTimeout(resolve, delay * 1000));
    return `Task ${name} completed`;
}

async function main() {
    const promises = [
        task("A", 1),
        task("B", 2),
        task("C", 1)
    ];
    
    const results = await Promise.all(promises);
    results.forEach(result => console.log(result));
}

main();
```

### Async Generator

```typescript
async function* countUpTo(n: number): AsyncGenerator<number> {
    for (let i = 1; i <= n; i++) {
        await new Promise(resolve => setTimeout(resolve, 500));
        yield i;
    }
}

async function main() {
    for await (const number of countUpTo(5)) {
        console.log(`Count: ${number}`);
    }
}

main();
```

## C++

C++20부터 coroutine이 언어 표준에 추가되었습니다.

### Basic Coroutine

```cpp
#include <coroutine>
#include <iostream>
#include <thread>
#include <chrono>

struct Task {
    struct promise_type {
        Task get_return_object() {
            return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_void() {}
        void unhandled_exception() {}
    };

    std::coroutine_handle<promise_type> coro;

    Task(std::coroutine_handle<promise_type> h) : coro(h) {}
    ~Task() { if (coro) coro.destroy(); }

    void resume() { coro.resume(); }
    bool done() { return coro.done(); }
};

Task coroutine_example() {
    std::cout << "Coroutine started\n";
    co_await std::suspend_always{};
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "Coroutine resumed\n";
}

int main() {
    Task task = coroutine_example();
    std::cout << "Main thread\n";
    task.resume();
    return 0;
}
```

### Generator Pattern

```cpp
#include <coroutine>
#include <iostream>

template<typename T>
struct Generator {
    struct promise_type {
        T current_value;
        
        Generator get_return_object() {
            return Generator{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        std::suspend_always yield_value(T value) {
            current_value = value;
            return {};
        }
        void return_void() {}
        void unhandled_exception() {}
    };

    std::coroutine_handle<promise_type> coro;

    Generator(std::coroutine_handle<promise_type> h) : coro(h) {}
    ~Generator() { if (coro) coro.destroy(); }

    bool next() {
        coro.resume();
        return !coro.done();
    }
    T value() { return coro.promise().current_value; }
};

Generator<int> count_up_to(int n) {
    for (int i = 1; i <= n; ++i) {
        co_yield i;
    }
}

int main() {
    auto gen = count_up_to(5);
    while (gen.next()) {
        std::cout << "Count: " << gen.value() << "\n";
    }
    return 0;
}
```

## C

C는 언어 레벨에서 coroutine을 지원하지 않지만, 라이브러리나 매크로를 사용하여 구현할 수 있습니다. 가장 유명한 방법은 `setjmp`/`longjmp`를 사용하거나 라이브러리를 사용하는 것입니다.

### Using setjmp/longjmp (Simple Example)

```c
#include <stdio.h>
#include <setjmp.h>
#include <unistd.h>

jmp_buf env;

void coroutine_example() {
    printf("Coroutine started\n");
    setjmp(env);
    sleep(1);
    printf("Coroutine resumed\n");
}

int main() {
    printf("Main thread\n");
    coroutine_example();
    longjmp(env, 1);
    return 0;
}
```

### Using libcoro (Third-party Library)

```c
#include <stdio.h>
#include <coro.h>

void* coroutine_function(void* arg) {
    printf("Coroutine started\n");
    coro_yield();
    printf("Coroutine resumed\n");
    return NULL;
}

int main() {
    coro_context ctx;
    coro_stack_alloc(&ctx.stack, 8192);
    coro_create(&ctx, coroutine_function, NULL, NULL, ctx.stack.sptr, ctx.stack.ssze);
    
    printf("Main thread\n");
    coro_transfer(&ctx, &ctx);
    coro_destroy(&ctx);
    return 0;
}
```

## C#

C#은 `async/await` 키워드를 사용하여 coroutine을 지원합니다.

### Basic Async/Await

```csharp
using System;
using System.Threading.Tasks;

class Program
{
    static async Task<string> FetchDataAsync()
    {
        await Task.Delay(1000);
        return "Data fetched";
    }

    static async Task Main(string[] args)
    {
        Console.WriteLine("Starting...");
        string result = await FetchDataAsync();
        Console.WriteLine(result);
    }
}
```

### Multiple Tasks

```csharp
using System;
using System.Threading.Tasks;
using System.Linq;

class Program
{
    static async Task<string> TaskAsync(string name, int delay)
    {
        await Task.Delay(delay * 1000);
        return $"Task {name} completed";
    }

    static async Task Main(string[] args)
    {
        var tasks = new[]
        {
            TaskAsync("A", 1),
            TaskAsync("B", 2),
            TaskAsync("C", 1)
        };

        var results = await Task.WhenAll(tasks);
        foreach (var result in results)
        {
            Console.WriteLine(result);
        }
    }
}
```

### Async Generator

```csharp
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

class Program
{
    static async IAsyncEnumerable<int> CountUpToAsync(int n)
    {
        for (int i = 1; i <= n; i++)
        {
            await Task.Delay(500);
            yield return i;
        }
    }

    static async Task Main(string[] args)
    {
        await foreach (var number in CountUpToAsync(5))
        {
            Console.WriteLine($"Count: {number}");
        }
    }
}
```

## Go

Go는 `goroutine`을 사용하여 경량 스레드를 제공합니다. `channel`을 통해 통신합니다.

### Basic Goroutine

```go
package main

import (
    "fmt"
    "time"
)

func fetchData() string {
    time.Sleep(1 * time.Second)
    return "Data fetched"
}

func main() {
    fmt.Println("Starting...")
    go func() {
        result := fetchData()
        fmt.Println(result)
    }()
    time.Sleep(2 * time.Second)
}
```

### Using Channels

```go
package main

import (
    "fmt"
    "time"
)

func task(name string, delay time.Duration, ch chan<- string) {
    time.Sleep(delay)
    ch <- fmt.Sprintf("Task %s completed", name)
}

func main() {
    ch := make(chan string, 3)
    
    go task("A", 1*time.Second, ch)
    go task("B", 2*time.Second, ch)
    go task("C", 1*time.Second, ch)
    
    for i := 0; i < 3; i++ {
        fmt.Println(<-ch)
    }
}
```

### Select Statement

```go
package main

import (
    "fmt"
    "time"
)

func main() {
    ch1 := make(chan string)
    ch2 := make(chan string)
    
    go func() {
        time.Sleep(1 * time.Second)
        ch1 <- "Message from channel 1"
    }()
    
    go func() {
        time.Sleep(2 * time.Second)
        ch2 <- "Message from channel 2"
    }()
    
    select {
    case msg1 := <-ch1:
        fmt.Println(msg1)
    case msg2 := <-ch2:
        fmt.Println(msg2)
    case <-time.After(3 * time.Second):
        fmt.Println("Timeout")
    }
}
```

## Rust

Rust는 `async/await`와 `Future` 트레이트를 사용합니다. `tokio`나 `async-std` 같은 런타임이 필요합니다.

### Basic Async/Await (with tokio)

```rust
use tokio::time::{sleep, Duration};

async fn fetch_data() -> String {
    sleep(Duration::from_secs(1)).await;
    "Data fetched".to_string()
}

#[tokio::main]
async fn main() {
    println!("Starting...");
    let result = fetch_data().await;
    println!("{}", result);
}
```

### Multiple Tasks

```rust
use tokio::time::{sleep, Duration};

async fn task(name: &str, delay: u64) -> String {
    sleep(Duration::from_secs(delay)).await;
    format!("Task {} completed", name)
}

#[tokio::main]
async fn main() {
    let tasks = vec![
        tokio::spawn(task("A", 1)),
        tokio::spawn(task("B", 2)),
        tokio::spawn(task("C", 1)),
    ];
    
    for task in tasks {
        println!("{}", task.await.unwrap());
    }
}
```

### Stream (Async Iterator)

```rust
use tokio_stream::{self as stream, StreamExt};
use tokio::time::{sleep, Duration};

async fn count_up_to(n: u32) -> impl stream::Stream<Item = u32> {
    stream::unfold(1, move |mut i| async move {
        if i <= n {
            sleep(Duration::from_millis(500)).await;
            let value = i;
            i += 1;
            Some((value, i))
        } else {
            None
        }
    })
}

#[tokio::main]
async fn main() {
    let mut stream = count_up_to(5).await;
    while let Some(number) = stream.next().await {
        println!("Count: {}", number);
    }
}
```

## Summary

각 언어별 coroutine 구현 방법:

| Language | Feature | Library/Runtime |
|----------|---------|-----------------|
| Java | Virtual Threads (19+), CompletableFuture | JDK |
| Kotlin | Coroutines | kotlinx.coroutines |
| Swift | async/await | Swift Concurrency |
| Python 3 | async/await | asyncio |
| TypeScript | async/await | Native (Promise) |
| C++ | Coroutines (C++20) | Standard Library |
| C | setjmp/longjmp or libraries | libcoro |
| C# | async/await | .NET |
| Go | goroutines | Native |
| Rust | async/await | tokio, async-std |

