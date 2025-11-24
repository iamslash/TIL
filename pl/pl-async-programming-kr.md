- [Async Programming](#async-programming)
  - [Java](#java)
    - [CompletableFuture](#completablefuture)
    - [Future](#future)
  - [Kotlin](#kotlin)
    - [Coroutine with async/await](#coroutine-with-asyncawait)
  - [Swift](#swift)
    - [async/await](#asyncawait)
    - [Combine Framework](#combine-framework)
  - [Python 3](#python-3)
    - [asyncio](#asyncio)
    - [async/await](#asyncawait-1)
  - [TypeScript](#typescript)
    - [Promise](#promise)
    - [async/await](#asyncawait-2)
  - [C++](#c)
    - [std::future](#stdfuture)
    - [std::async](#stdasync)
  - [C](#c-1)
    - [Callback Pattern](#callback-pattern)
    - [Event Loop (libuv)](#event-loop-libuv)
  - [C#](#c-2)
    - [async/await](#asyncawait-3)
    - [Task](#task)
  - [Go](#go)
    - [Goroutine with Channel](#goroutine-with-channel)
  - [Rust](#rust)
    - [async/await with tokio](#asyncawait-with-tokio)

--------

# Async Programming

비동기 프로그래밍은 작업을 블로킹하지 않고 다른 작업을 계속 수행할 수 있게 하는 프로그래밍 패러다임입니다.

## Java

Java는 `CompletableFuture`와 `Future`를 사용하여 비동기 프로그래밍을 지원합니다.

### CompletableFuture

```java
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

public class AsyncExample {
    public static void main(String[] args) {
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> {
            try {
                Thread.sleep(1000);
                return "Data fetched";
            } catch (InterruptedException e) {
                return "Interrupted";
            }
        });
        
        future.thenAccept(result -> System.out.println(result));
        future.thenApply(String::toUpperCase)
              .thenAccept(System.out::println);
        
        // 여러 작업 조합
        CompletableFuture<String> future1 = CompletableFuture.supplyAsync(() -> "Hello");
        CompletableFuture<String> future2 = CompletableFuture.supplyAsync(() -> "World");
        
        CompletableFuture<String> combined = future1.thenCombine(future2, (a, b) -> a + " " + b);
        combined.thenAccept(System.out::println);
        
        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
```

### Future

```java
import java.util.concurrent.*;

public class FutureExample {
    public static void main(String[] args) throws ExecutionException, InterruptedException {
        ExecutorService executor = Executors.newFixedThreadPool(2);
        
        Future<String> future = executor.submit(() -> {
            Thread.sleep(1000);
            return "Task completed";
        });
        
        System.out.println("Doing other work...");
        String result = future.get();  // 블로킹
        System.out.println(result);
        
        executor.shutdown();
    }
}
```

## Kotlin

Kotlin은 coroutine을 사용하여 비동기 프로그래밍을 지원합니다.

### Coroutine with async/await

```kotlin
import kotlinx.coroutines.*

suspend fun fetchData(): String {
    delay(1000L)
    return "Data fetched"
}

suspend fun fetchMoreData(): String {
    delay(1500L)
    return "More data fetched"
}

fun main() = runBlocking {
    // 순차 실행
    val data1 = fetchData()
    val data2 = fetchMoreData()
    println("$data1, $data2")
    
    // 병렬 실행
    val deferred1 = async { fetchData() }
    val deferred2 = async { fetchMoreData() }
    println("${deferred1.await()}, ${deferred2.await()}")
}
```

## Swift

Swift는 `async/await`와 Combine 프레임워크를 제공합니다.

### async/await

```swift
import Foundation

func fetchData() async throws -> String {
    try await Task.sleep(nanoseconds: 1_000_000_000)
    return "Data fetched"
}

func fetchMoreData() async throws -> String {
    try await Task.sleep(nanoseconds: 1_500_000_000)
    return "More data fetched"
}

func main() async {
    // 순차 실행
    let data1 = try await fetchData()
    let data2 = try await fetchMoreData()
    print("\(data1), \(data2)")
    
    // 병렬 실행
    async let data3 = fetchData()
    async let data4 = fetchMoreData()
    let results = try await [data3, data4]
    print(results)
}

Task {
    try? await main()
}
```

### Combine Framework

```swift
import Combine

let publisher = Just("Hello")
    .delay(for: .seconds(1), scheduler: DispatchQueue.main)
    .map { $0 + " World" }

let cancellable = publisher.sink { value in
    print(value)
}
```

## Python 3

Python은 `asyncio` 모듈을 사용하여 비동기 프로그래밍을 지원합니다.

### asyncio

```python
import asyncio

async def fetch_data():
    await asyncio.sleep(1)
    return "Data fetched"

async def fetch_more_data():
    await asyncio.sleep(1.5)
    return "More data fetched"

async def main():
    # 순차 실행
    data1 = await fetch_data()
    data2 = await fetch_more_data()
    print(f"{data1}, {data2}")
    
    # 병렬 실행
    results = await asyncio.gather(
        fetch_data(),
        fetch_more_data()
    )
    print(results)

asyncio.run(main())
```

### async/await

```python
import asyncio

async def process_item(item):
    await asyncio.sleep(0.5)
    return f"Processed {item}"

async def main():
    items = [1, 2, 3, 4, 5]
    tasks = [process_item(item) for item in items]
    results = await asyncio.gather(*tasks)
    for result in results:
        print(result)

asyncio.run(main())
```

## TypeScript

TypeScript는 Promise와 async/await를 사용합니다.

### Promise

```typescript
function fetchData(): Promise<string> {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve("Data fetched");
        }, 1000);
    });
}

function fetchMoreData(): Promise<string> {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve("More data fetched");
        }, 1500);
    });
}

// Promise 체이닝
fetchData()
    .then(data => {
        console.log(data);
        return fetchMoreData();
    })
    .then(data => {
        console.log(data);
    });

// Promise.all
Promise.all([fetchData(), fetchMoreData()])
    .then(results => {
        console.log(results);
    });
```

### async/await

```typescript
async function fetchData(): Promise<string> {
    await new Promise(resolve => setTimeout(resolve, 1000));
    return "Data fetched";
}

async function fetchMoreData(): Promise<string> {
    await new Promise(resolve => setTimeout(resolve, 1500));
    return "More data fetched";
}

async function main() {
    // 순차 실행
    const data1 = await fetchData();
    const data2 = await fetchMoreData();
    console.log(`${data1}, ${data2}`);
    
    // 병렬 실행
    const results = await Promise.all([fetchData(), fetchMoreData()]);
    console.log(results);
}

main();
```

## C++

C++는 `std::future`와 `std::async`를 제공합니다.

### std::future

```cpp
#include <future>
#include <iostream>
#include <thread>
#include <chrono>

std::string fetchData() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return "Data fetched";
}

int main() {
    std::future<std::string> future = std::async(std::launch::async, fetchData);
    
    std::cout << "Doing other work..." << std::endl;
    std::string result = future.get();  // 블로킹
    std::cout << result << std::endl;
    
    return 0;
}
```

### std::async

```cpp
#include <future>
#include <iostream>
#include <vector>

int compute(int n) {
    return n * n;
}

int main() {
    std::vector<std::future<int>> futures;
    
    for (int i = 1; i <= 5; i++) {
        futures.push_back(std::async(std::launch::async, compute, i));
    }
    
    for (auto& future : futures) {
        std::cout << "Result: " << future.get() << std::endl;
    }
    
    return 0;
}
```

## C

C는 언어 레벨에서 비동기 프로그래밍을 직접 지원하지 않지만, callback 패턴이나 이벤트 루프 라이브러리(libuv, libevent 등)를 사용하여 비동기 프로그래밍을 구현할 수 있습니다.

### Callback Pattern

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

typedef void (*callback_t)(const char* result);

void fetch_data_async(callback_t callback) {
    pthread_t thread;
    
    void* fetch_thread(void* arg) {
        sleep(1);  // 비동기 작업 시뮬레이션
        callback("Data fetched");
        return NULL;
    }
    
    pthread_create(&thread, NULL, fetch_thread, NULL);
    pthread_detach(thread);
}

void on_data_received(const char* result) {
    printf("%s\n", result);
}

int main() {
    printf("Starting async operation...\n");
    fetch_data_async(on_data_received);
    
    printf("Doing other work...\n");
    sleep(2);  // 메인 스레드가 다른 작업 수행
    
    return 0;
}
```

### Event Loop (libuv)

```c
#include <stdio.h>
#include <uv.h>

void fetch_data(uv_work_t* req) {
    // 백그라운드 스레드에서 실행되는 작업
    sleep(1);
}

void after_fetch(uv_work_t* req, int status) {
    // 메인 스레드로 돌아와서 실행되는 콜백
    printf("Data fetched\n");
}

void fetch_more_data(uv_work_t* req) {
    sleep(1);
}

void after_fetch_more(uv_work_t* req, int status) {
    printf("More data fetched\n");
}

int main() {
    uv_loop_t* loop = uv_default_loop();
    
    uv_work_t req1, req2;
    
    // 비동기 작업 큐에 추가
    uv_queue_work(loop, &req1, fetch_data, after_fetch);
    uv_queue_work(loop, &req2, fetch_more_data, after_fetch_more);
    
    printf("Doing other work...\n");
    
    // 이벤트 루프 실행
    uv_run(loop, UV_RUN_DEFAULT);
    
    return 0;
}
```

## C#

C#은 `async/await` 키워드를 제공합니다.

### async/await

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
    
    static async Task<string> FetchMoreDataAsync()
    {
        await Task.Delay(1500);
        return "More data fetched";
    }
    
    static async Task Main(string[] args)
    {
        // 순차 실행
        var data1 = await FetchDataAsync();
        var data2 = await FetchMoreDataAsync();
        Console.WriteLine($"{data1}, {data2}");
        
        // 병렬 실행
        var task1 = FetchDataAsync();
        var task2 = FetchMoreDataAsync();
        var results = await Task.WhenAll(task1, task2);
        Console.WriteLine(string.Join(", ", results));
    }
}
```

### Task

```csharp
using System;
using System.Threading.Tasks;

class Program
{
    static async Task ProcessItemAsync(int item)
    {
        await Task.Delay(500);
        Console.WriteLine($"Processed {item}");
    }
    
    static async Task Main(string[] args)
    {
        var tasks = new Task[5];
        for (int i = 1; i <= 5; i++)
        {
            int item = i;
            tasks[i - 1] = ProcessItemAsync(item);
        }
        
        await Task.WhenAll(tasks);
    }
}
```

## Go

Go는 goroutine과 channel을 사용합니다.

### Goroutine with Channel

```go
package main

import (
    "fmt"
    "time"
)

func fetchData(ch chan<- string) {
    time.Sleep(1 * time.Second)
    ch <- "Data fetched"
}

func fetchMoreData(ch chan<- string) {
    time.Sleep(1500 * time.Millisecond)
    ch <- "More data fetched"
}

func main() {
    ch1 := make(chan string)
    ch2 := make(chan string)
    
    go fetchData(ch1)
    go fetchMoreData(ch2)
    
    // Select를 사용하여 먼저 도착하는 결과 처리
    select {
    case result1 := <-ch1:
        fmt.Println(result1)
    case result2 := <-ch2:
        fmt.Println(result2)
    }
    
    // 모든 결과 대기
    result1 := <-ch1
    result2 := <-ch2
    fmt.Printf("%s, %s\n", result1, result2)
}
```

## Rust

Rust는 `async/await`와 `tokio` 런타임을 사용합니다.

### async/await with tokio

```rust
use tokio::time::{sleep, Duration};

async fn fetch_data() -> String {
    sleep(Duration::from_secs(1)).await;
    "Data fetched".to_string()
}

async fn fetch_more_data() -> String {
    sleep(Duration::from_secs(1)).await;
    "More data fetched".to_string()
}

#[tokio::main]
async fn main() {
    // 순차 실행
    let data1 = fetch_data().await;
    let data2 = fetch_more_data().await;
    println!("{}, {}", data1, data2);
    
    // 병렬 실행
    let task1 = tokio::spawn(fetch_data());
    let task2 = tokio::spawn(fetch_more_data());
    
    let (result1, result2) = tokio::join!(task1, task2);
    println!("{}, {}", result1.unwrap(), result2.unwrap());
}
```

