- [Synchronization](#synchronization)
  - [Java](#java)
    - [synchronized](#synchronized)
    - [ReentrantLock](#reentrantlock)
    - [Semaphore](#semaphore)
    - [CountDownLatch](#countdownlatch)
  - [Kotlin](#kotlin)
    - [Mutex](#mutex)
    - [Semaphore](#semaphore-1)
  - [Swift](#swift)
    - [NSLock](#nslock)
    - [DispatchSemaphore](#dispatchsemaphore)
  - [Python 3](#python-3)
    - [Lock](#lock)
    - [Semaphore](#semaphore-2)
  - [TypeScript](#typescript)
    - [Mutex (SharedArrayBuffer)](#mutex-sharedarraybuffer)
  - [C++](#c)
    - [mutex](#mutex-1)
    - [semaphore](#semaphore-3)
    - [condition_variable](#condition_variable)
  - [C](#c-1)
    - [pthread_mutex](#pthread_mutex)
    - [semaphore](#semaphore-4)
  - [C#](#c-2)
    - [lock](#lock-1)
    - [SemaphoreSlim](#semaphoreslim)
  - [Go](#go)
    - [Mutex](#mutex-2)
    - [Channel (동기화 수단)](#channel-동기화-수단)
  - [Rust](#rust)
    - [Mutex](#mutex-3)
    - [Semaphore](#semaphore-5)

--------

# Synchronization

동기화는 여러 스레드가 공유 자원에 안전하게 접근할 수 있도록 보장하는 메커니즘입니다. 주요 동기화 도구로는 mutex, semaphore, condition variable 등이 있습니다.

## Java

Java는 `synchronized` 키워드와 `java.util.concurrent` 패키지를 제공합니다.

### synchronized

```java
class Counter {
    private int count = 0;
    
    public synchronized void increment() {
        count++;
    }
    
    public synchronized int getCount() {
        return count;
    }
}

// 또는 synchronized 블록 사용
class Counter2 {
    private int count = 0;
    private final Object lock = new Object();
    
    public void increment() {
        synchronized(lock) {
            count++;
        }
    }
}
```

### ReentrantLock

```java
import java.util.concurrent.locks.ReentrantLock;

class Counter {
    private int count = 0;
    private final ReentrantLock lock = new ReentrantLock();
    
    public void increment() {
        lock.lock();
        try {
            count++;
        } finally {
            lock.unlock();
        }
    }
}
```

### Semaphore

```java
import java.util.concurrent.Semaphore;

class ResourcePool {
    private final Semaphore semaphore;
    
    public ResourcePool(int permits) {
        this.semaphore = new Semaphore(permits);
    }
    
    public void useResource() throws InterruptedException {
        semaphore.acquire();
        try {
            // 공유 자원 사용
            System.out.println("Resource in use");
        } finally {
            semaphore.release();
        }
    }
}
```

### CountDownLatch

```java
import java.util.concurrent.CountDownLatch;

public class CountDownLatchExample {
    public static void main(String[] args) throws InterruptedException {
        CountDownLatch latch = new CountDownLatch(3);
        
        for (int i = 0; i < 3; i++) {
            new Thread(() -> {
                System.out.println("Task completed");
                latch.countDown();
            }).start();
        }
        
        latch.await();
        System.out.println("All tasks completed");
    }
}
```

## Kotlin

Kotlin은 coroutine의 동기화 도구를 제공합니다.

### Mutex

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.*

val mutex = Mutex()
var counter = 0

suspend fun increment() {
    mutex.withLock {
        counter++
    }
}

fun main() = runBlocking {
    repeat(100) {
        launch {
            increment()
        }
    }
    delay(1000)
    println("Counter: $counter")
}
```

### Semaphore

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.*

val semaphore = Semaphore(permits = 3)

suspend fun useResource() {
    semaphore.withPermit {
        println("Resource in use")
        delay(1000)
    }
}

fun main() = runBlocking {
    repeat(10) {
        launch {
            useResource()
        }
    }
    delay(5000)
}
```

## Swift

Swift는 `NSLock`과 `DispatchSemaphore`를 제공합니다.

### NSLock

```swift
import Foundation

class Counter {
    private var count = 0
    private let lock = NSLock()
    
    func increment() {
        lock.lock()
        defer { lock.unlock() }
        count += 1
    }
    
    func getCount() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }
}
```

### DispatchSemaphore

```swift
import Foundation

let semaphore = DispatchSemaphore(value: 3)

func useResource() {
    semaphore.wait()
    defer { semaphore.signal() }
    
    print("Resource in use")
    Thread.sleep(forTimeInterval: 1.0)
}

for _ in 0..<10 {
    DispatchQueue.global().async {
        useResource()
    }
}
```

## Python 3

Python은 `threading` 모듈의 `Lock`과 `Semaphore`를 제공합니다.

### Lock

```python
import threading

counter = 0
lock = threading.Lock()

def increment():
    global counter
    with lock:
        counter += 1

threads = []
for _ in range(100):
    t = threading.Thread(target=increment)
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"Counter: {counter}")
```

### Semaphore

```python
import threading
import time

semaphore = threading.Semaphore(3)

def use_resource():
    semaphore.acquire()
    try:
        print("Resource in use")
        time.sleep(1)
    finally:
        semaphore.release()

threads = []
for i in range(10):
    t = threading.Thread(target=use_resource)
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```

## TypeScript

TypeScript는 SharedArrayBuffer와 Atomics를 사용하여 동기화를 구현할 수 있습니다.

### Mutex (SharedArrayBuffer)

```typescript
class Mutex {
    private buffer: SharedArrayBuffer;
    private view: Int32Array;
    
    constructor() {
        this.buffer = new SharedArrayBuffer(4);
        this.view = new Int32Array(this.buffer);
    }
    
    lock(): void {
        while (Atomics.compareExchange(this.view, 0, 0, 1) !== 0) {
            Atomics.wait(this.view, 0, 1);
        }
    }
    
    unlock(): void {
        Atomics.store(this.view, 0, 0);
        Atomics.notify(this.view, 0, 1);
    }
}
```

## C++

C++11부터 `<mutex>`, `<semaphore>`, `<condition_variable>`을 제공합니다.

### mutex

```cpp
#include <mutex>
#include <thread>
#include <iostream>

std::mutex mtx;
int counter = 0;

void increment() {
    std::lock_guard<std::mutex> lock(mtx);
    counter++;
}

int main() {
    std::thread t1(increment);
    std::thread t2(increment);
    
    t1.join();
    t2.join();
    
    std::cout << "Counter: " << counter << std::endl;
    return 0;
}
```

### semaphore

```cpp
#include <semaphore>
#include <thread>
#include <iostream>

std::counting_semaphore<3> semaphore(3);

void useResource() {
    semaphore.acquire();
    std::cout << "Resource in use" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    semaphore.release();
}

int main() {
    std::thread threads[10];
    for (int i = 0; i < 10; i++) {
        threads[i] = std::thread(useResource);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    return 0;
}
```

### condition_variable

```cpp
#include <condition_variable>
#include <mutex>
#include <thread>
#include <queue>

std::mutex mtx;
std::condition_variable cv;
std::queue<int> queue;

void producer() {
    for (int i = 0; i < 10; i++) {
        std::unique_lock<std::mutex> lock(mtx);
        queue.push(i);
        cv.notify_one();
    }
}

void consumer() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, []{ return !queue.empty(); });
        int value = queue.front();
        queue.pop();
        std::cout << "Consumed: " << value << std::endl;
        if (value == 9) break;
    }
}

int main() {
    std::thread t1(producer);
    std::thread t2(consumer);
    
    t1.join();
    t2.join();
    
    return 0;
}
```

## C

C는 `pthread` 라이브러리의 동기화 도구를 사용합니다.

### pthread_mutex

```c
#include <pthread.h>
#include <stdio.h>

int counter = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void* increment(void* arg) {
    pthread_mutex_lock(&mutex);
    counter++;
    pthread_mutex_unlock(&mutex);
    return NULL;
}

int main() {
    pthread_t t1, t2;
    pthread_create(&t1, NULL, increment, NULL);
    pthread_create(&t2, NULL, increment, NULL);
    
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    
    printf("Counter: %d\n", counter);
    return 0;
}
```

### semaphore

```c
#include <semaphore.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

sem_t semaphore;

void* useResource(void* arg) {
    sem_wait(&semaphore);
    printf("Resource in use\n");
    sleep(1);
    sem_post(&semaphore);
    return NULL;
}

int main() {
    sem_init(&semaphore, 0, 3);
    
    pthread_t threads[10];
    for (int i = 0; i < 10; i++) {
        pthread_create(&threads[i], NULL, useResource, NULL);
    }
    
    for (int i = 0; i < 10; i++) {
        pthread_join(threads[i], NULL);
    }
    
    sem_destroy(&semaphore);
    return 0;
}
```

## C#

C#은 `lock` 키워드와 `System.Threading` 네임스페이스를 제공합니다.

### lock

```csharp
using System;
using System.Threading;

class Counter
{
    private int count = 0;
    private readonly object lockObject = new object();
    
    public void Increment()
    {
        lock (lockObject)
        {
            count++;
        }
    }
    
    public int GetCount()
    {
        lock (lockObject)
        {
            return count;
        }
    }
}
```

### SemaphoreSlim

```csharp
using System;
using System.Threading;
using System.Threading.Tasks;

class Program
{
    static SemaphoreSlim semaphore = new SemaphoreSlim(3, 3);
    
    static void UseResource()
    {
        semaphore.Wait();
        try
        {
            Console.WriteLine("Resource in use");
            Thread.Sleep(1000);
        }
        finally
        {
            semaphore.Release();
        }
    }
    
    static void Main(string[] args)
    {
        Task[] tasks = new Task[10];
        for (int i = 0; i < 10; i++)
        {
            tasks[i] = Task.Run(UseResource);
        }
        Task.WaitAll(tasks);
    }
}
```

## Go

Go는 `sync` 패키지와 channel을 통한 동기화를 제공합니다.

### Mutex

```go
package main

import (
    "fmt"
    "sync"
)

type Counter struct {
    mu    sync.Mutex
    count int
}

func (c *Counter) Increment() {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.count++
}

func (c *Counter) GetCount() int {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.count
}

func main() {
    counter := &Counter{}
    var wg sync.WaitGroup
    
    for i := 0; i < 100; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            counter.Increment()
        }()
    }
    
    wg.Wait()
    fmt.Printf("Counter: %d\n", counter.GetCount())
}
```

### Channel (동기화 수단)

```go
package main

import (
    "fmt"
    "time"
)

func worker(id int, jobs <-chan int, results chan<- int) {
    for j := range jobs {
        fmt.Printf("Worker %d processing job %d\n", id, j)
        time.Sleep(time.Second)
        results <- j * 2
    }
}

func main() {
    jobs := make(chan int, 5)
    results := make(chan int, 5)
    
    for w := 1; w <= 3; w++ {
        go worker(w, jobs, results)
    }
    
    for j := 1; j <= 5; j++ {
        jobs <- j
    }
    close(jobs)
    
    for r := 1; r <= 5; r++ {
        fmt.Printf("Result: %d\n", <-results)
    }
}
```

## Rust

Rust는 `std::sync` 모듈을 통해 동기화 도구를 제공합니다.

### Mutex

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Counter: {}", *counter.lock().unwrap());
}
```

### Semaphore

```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

struct Semaphore {
    count: AtomicUsize,
}

impl Semaphore {
    fn new(count: usize) -> Self {
        Semaphore {
            count: AtomicUsize::new(count),
        }
    }
    
    fn acquire(&self) {
        while self.count.load(Ordering::Acquire) == 0 {
            thread::yield_now();
        }
        self.count.fetch_sub(1, Ordering::Release);
    }
    
    fn release(&self) {
        self.count.fetch_add(1, Ordering::Release);
    }
}

fn main() {
    let semaphore = Arc::new(Semaphore::new(3));
    let mut handles = vec![];

    for i in 0..10 {
        let sem = Arc::clone(&semaphore);
        let handle = thread::spawn(move || {
            sem.acquire();
            println!("Resource in use: {}", i);
            thread::sleep(std::time::Duration::from_secs(1));
            sem.release();
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}
```

