- [Deadlock and Race Condition](#deadlock-and-race-condition)
  - [Deadlock](#deadlock)
    - [Java](#java)
    - [Python 3](#python-3)
    - [C++](#c)
    - [Kotlin](#kotlin)
    - [Swift](#swift)
    - [Go](#go)
    - [TypeScript](#typescript)
    - [C](#c-1)
    - [C#](#c-2)
    - [Rust](#rust)
  - [Race Condition](#race-condition)
    - [Java](#java-1)
    - [Python 3](#python-3-1)
    - [C++](#c-3)
    - [Kotlin](#kotlin-1)
    - [Swift](#swift-1)
    - [Go](#go-1)
    - [TypeScript](#typescript-1)
    - [C](#c-4)
    - [C#](#c-5)
    - [Rust](#rust-1)
  - [해결 방법](#해결-방법)
    - [Lock 순서 통일](#lock-순서-통일)
    - [Atomic 연산 사용](#atomic-연산-사용)
    - [불변성 보장](#불변성-보장)

--------

# Deadlock and Race Condition

데드락(Deadlock)과 경쟁 상태(Race Condition)는 동시성 프로그래밍에서 발생하는 주요 문제입니다.

## Deadlock

데드락은 두 개 이상의 스레드가 서로의 자원을 기다리며 무한 대기하는 상황입니다.

### Java

```java
class DeadlockExample {
    private final Object lock1 = new Object();
    private final Object lock2 = new Object();
    
    public void method1() {
        synchronized(lock1) {
            System.out.println("Thread 1: Holding lock1");
            try { Thread.sleep(100); } catch (InterruptedException e) {}
            synchronized(lock2) {
                System.out.println("Thread 1: Holding lock1 and lock2");
            }
        }
    }
    
    public void method2() {
        synchronized(lock2) {
            System.out.println("Thread 2: Holding lock2");
            try { Thread.sleep(100); } catch (InterruptedException e) {}
            synchronized(lock1) {
                System.out.println("Thread 2: Holding lock2 and lock1");
            }
        }
    }
    
    public static void main(String[] args) {
        DeadlockExample example = new DeadlockExample();
        
        Thread t1 = new Thread(example::method1);
        Thread t2 = new Thread(example::method2);
        
        t1.start();
        t2.start();
    }
}
```

### Python 3

```python
import threading
import time

lock1 = threading.Lock()
lock2 = threading.Lock()

def method1():
    with lock1:
        print("Thread 1: Holding lock1")
        time.sleep(0.1)
        with lock2:
            print("Thread 1: Holding lock1 and lock2")

def method2():
    with lock2:
        print("Thread 2: Holding lock2")
        time.sleep(0.1)
        with lock1:
            print("Thread 2: Holding lock2 and lock1")

t1 = threading.Thread(target=method1)
t2 = threading.Thread(target=method2)

t1.start()
t2.start()

t1.join()
t2.join()
```

### C++

```cpp
#include <mutex>
#include <thread>
#include <chrono>

std::mutex mutex1, mutex2;

void method1() {
    std::lock_guard<std::mutex> lock1(mutex1);
    std::cout << "Thread 1: Holding mutex1" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::lock_guard<std::mutex> lock2(mutex2);
    std::cout << "Thread 1: Holding mutex1 and mutex2" << std::endl;
}

void method2() {
    std::lock_guard<std::mutex> lock2(mutex2);
    std::cout << "Thread 2: Holding mutex2" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::lock_guard<std::mutex> lock1(mutex1);
    std::cout << "Thread 2: Holding mutex2 and mutex1" << std::endl;
}

int main() {
    std::thread t1(method1);
    std::thread t2(method2);
    
    t1.join();
    t2.join();
    
    return 0;
}
```

### Kotlin

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.*

val mutex1 = Mutex()
val mutex2 = Mutex()

suspend fun method1() {
    mutex1.lock()
    println("Thread 1: Holding mutex1")
    delay(100L)
    mutex2.lock()
    println("Thread 1: Holding mutex1 and mutex2")
    mutex2.unlock()
    mutex1.unlock()
}

suspend fun method2() {
    mutex2.lock()
    println("Thread 2: Holding mutex2")
    delay(100L)
    mutex1.lock()
    println("Thread 2: Holding mutex2 and mutex1")
    mutex1.unlock()
    mutex2.unlock()
}

fun main() = runBlocking {
    launch { method1() }
    launch { method2() }
    delay(2000L)
}
```

### Swift

```swift
import Foundation

let lock1 = NSLock()
let lock2 = NSLock()

func method1() {
    lock1.lock()
    print("Thread 1: Holding lock1")
    Thread.sleep(forTimeInterval: 0.1)
    lock2.lock()
    print("Thread 1: Holding lock1 and lock2")
    lock2.unlock()
    lock1.unlock()
}

func method2() {
    lock2.lock()
    print("Thread 2: Holding lock2")
    Thread.sleep(forTimeInterval: 0.1)
    lock1.lock()
    print("Thread 2: Holding lock2 and lock1")
    lock1.unlock()
    lock2.unlock()
}

let thread1 = Thread { method1() }
let thread2 = Thread { method2() }

thread1.start()
thread2.start()

thread1.cancel()
thread2.cancel()
```

### Go

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

var mutex1, mutex2 sync.Mutex

func method1() {
    mutex1.Lock()
    fmt.Println("Thread 1: Holding mutex1")
    time.Sleep(100 * time.Millisecond)
    mutex2.Lock()
    fmt.Println("Thread 1: Holding mutex1 and mutex2")
    mutex2.Unlock()
    mutex1.Unlock()
}

func method2() {
    mutex2.Lock()
    fmt.Println("Thread 2: Holding mutex2")
    time.Sleep(100 * time.Millisecond)
    mutex1.Lock()
    fmt.Println("Thread 2: Holding mutex2 and mutex1")
    mutex1.Unlock()
    mutex2.Unlock()
}

func main() {
    go method1()
    go method2()
    time.Sleep(2 * time.Second)
}
```

### TypeScript

```typescript
class Mutex {
    private locked = false;
    private queue: Array<() => void> = [];
    
    async lock(): Promise<void> {
        return new Promise((resolve) => {
            if (!this.locked) {
                this.locked = true;
                resolve();
            } else {
                this.queue.push(resolve);
            }
        });
    }
    
    unlock(): void {
        if (this.queue.length > 0) {
            const next = this.queue.shift();
            if (next) next();
        } else {
            this.locked = false;
        }
    }
}

const mutex1 = new Mutex();
const mutex2 = new Mutex();

async function method1() {
    await mutex1.lock();
    console.log("Thread 1: Holding mutex1");
    await new Promise(resolve => setTimeout(resolve, 100));
    await mutex2.lock();
    console.log("Thread 1: Holding mutex1 and mutex2");
    mutex2.unlock();
    mutex1.unlock();
}

async function method2() {
    await mutex2.lock();
    console.log("Thread 2: Holding mutex2");
    await new Promise(resolve => setTimeout(resolve, 100));
    await mutex1.lock();
    console.log("Thread 2: Holding mutex2 and mutex1");
    mutex1.unlock();
    mutex2.unlock();
}

method1();
method2();
```

### C

```c
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;

void* method1(void* arg) {
    pthread_mutex_lock(&mutex1);
    printf("Thread 1: Holding mutex1\n");
    usleep(100000);  // 100ms
    pthread_mutex_lock(&mutex2);
    printf("Thread 1: Holding mutex1 and mutex2\n");
    pthread_mutex_unlock(&mutex2);
    pthread_mutex_unlock(&mutex1);
    return NULL;
}

void* method2(void* arg) {
    pthread_mutex_lock(&mutex2);
    printf("Thread 2: Holding mutex2\n");
    usleep(100000);  // 100ms
    pthread_mutex_lock(&mutex1);
    printf("Thread 2: Holding mutex2 and mutex1\n");
    pthread_mutex_unlock(&mutex1);
    pthread_mutex_unlock(&mutex2);
    return NULL;
}

int main() {
    pthread_t t1, t2;
    pthread_create(&t1, NULL, method1, NULL);
    pthread_create(&t2, NULL, method2, NULL);
    
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    
    return 0;
}
```

### C#

```csharp
using System;
using System.Threading;

class DeadlockExample
{
    private static readonly object lock1 = new object();
    private static readonly object lock2 = new object();
    
    static void Method1()
    {
        lock (lock1)
        {
            Console.WriteLine("Thread 1: Holding lock1");
            Thread.Sleep(100);
            lock (lock2)
            {
                Console.WriteLine("Thread 1: Holding lock1 and lock2");
            }
        }
    }
    
    static void Method2()
    {
        lock (lock2)
        {
            Console.WriteLine("Thread 2: Holding lock2");
            Thread.Sleep(100);
            lock (lock1)
            {
                Console.WriteLine("Thread 2: Holding lock2 and lock1");
            }
        }
    }
    
    static void Main(string[] args)
    {
        Thread t1 = new Thread(Method1);
        Thread t2 = new Thread(Method2);
        
        t1.Start();
        t2.Start();
        
        t1.Join();
        t2.Join();
    }
}
```

### Rust

```rust
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

fn main() {
    let mutex1 = Arc::new(Mutex::new(()));
    let mutex2 = Arc::new(Mutex::new(()));
    
    let m1_clone1 = Arc::clone(&mutex1);
    let m2_clone1 = Arc::clone(&mutex2);
    let handle1 = thread::spawn(move || {
        let _lock1 = m1_clone1.lock().unwrap();
        println!("Thread 1: Holding mutex1");
        thread::sleep(Duration::from_millis(100));
        let _lock2 = m2_clone1.lock().unwrap();
        println!("Thread 1: Holding mutex1 and mutex2");
    });
    
    let m1_clone2 = Arc::clone(&mutex1);
    let m2_clone2 = Arc::clone(&mutex2);
    let handle2 = thread::spawn(move || {
        let _lock2 = m2_clone2.lock().unwrap();
        println!("Thread 2: Holding mutex2");
        thread::sleep(Duration::from_millis(100));
        let _lock1 = m1_clone2.lock().unwrap();
        println!("Thread 2: Holding mutex2 and mutex1");
    });
    
    handle1.join().unwrap();
    handle2.join().unwrap();
}
```

## Race Condition

경쟁 상태는 여러 스레드가 공유 자원에 동시에 접근하여 예측 불가능한 결과를 만드는 상황입니다.

### Java

```java
class RaceConditionExample {
    private int counter = 0;
    
    public void increment() {
        counter++;  // Race condition 발생 가능
    }
    
    public int getCounter() {
        return counter;
    }
    
    public static void main(String[] args) throws InterruptedException {
        RaceConditionExample example = new RaceConditionExample();
        
        Thread[] threads = new Thread[10];
        for (int i = 0; i < 10; i++) {
            threads[i] = new Thread(() -> {
                for (int j = 0; j < 1000; j++) {
                    example.increment();
                }
            });
            threads[i].start();
        }
        
        for (Thread t : threads) {
            t.join();
        }
        
        System.out.println("Counter: " + example.getCounter());  // 예상: 10000, 실제: 10000 미만 가능
    }
}
```

### Python 3

```python
import threading

counter = 0

def increment():
    global counter
    for _ in range(1000):
        counter += 1  # Race condition 발생 가능

threads = []
for _ in range(10):
    t = threading.Thread(target=increment)
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"Counter: {counter}")  # 예상: 10000, 실제: 10000 미만 가능
```

### C++

```cpp
#include <thread>
#include <iostream>

int counter = 0;

void increment() {
    for (int i = 0; i < 1000; i++) {
        counter++;  // Race condition 발생 가능
    }
}

int main() {
    std::thread threads[10];
    
    for (int i = 0; i < 10; i++) {
        threads[i] = std::thread(increment);
    }
    
    for (int i = 0; i < 10; i++) {
        threads[i].join();
    }
    
    std::cout << "Counter: " << counter << std::endl;  // 예상: 10000, 실제: 10000 미만 가능
    return 0;
}
```

### Kotlin

```kotlin
var counter = 0

fun increment() {
    for (i in 0 until 1000) {
        counter++  // Race condition 발생 가능
    }
}

fun main() = runBlocking {
    val jobs = List(10) {
        launch {
            increment()
        }
    }
    jobs.forEach { it.join() }
    println("Counter: $counter")  // 예상: 10000, 실제: 10000 미만 가능
}
```

### Swift

```swift
import Foundation

var counter = 0

func increment() {
    for _ in 0..<1000 {
        counter += 1  // Race condition 발생 가능
    }
}

let group = DispatchGroup()
let queue = DispatchQueue(label: "concurrent", attributes: .concurrent)

for _ in 0..<10 {
    group.enter()
    queue.async {
        increment()
        group.leave()
    }
}

group.wait()
print("Counter: \(counter)")  // 예상: 10000, 실제: 10000 미만 가능
```

### Go

```go
package main

import (
    "fmt"
    "sync"
)

var counter int

func increment() {
    for i := 0; i < 1000; i++ {
        counter++  // Race condition 발생 가능
    }
}

func main() {
    var wg sync.WaitGroup
    
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            increment()
        }()
    }
    
    wg.Wait()
    fmt.Printf("Counter: %d\n", counter)  // 예상: 10000, 실제: 10000 미만 가능
}
```

### TypeScript

```typescript
let counter = 0;

function increment() {
    for (let i = 0; i < 1000; i++) {
        counter++;  // Race condition 발생 가능
    }
}

const promises: Promise<void>[] = [];

for (let i = 0; i < 10; i++) {
    promises.push(
        new Promise((resolve) => {
            increment();
            resolve();
        })
    );
}

Promise.all(promises).then(() => {
    console.log(`Counter: ${counter}`);  // 예상: 10000, 실제: 10000 미만 가능
});
```

### C

```c
#include <pthread.h>
#include <stdio.h>

int counter = 0;

void* increment(void* arg) {
    for (int i = 0; i < 1000; i++) {
        counter++;  // Race condition 발생 가능
    }
    return NULL;
}

int main() {
    pthread_t threads[10];
    
    for (int i = 0; i < 10; i++) {
        pthread_create(&threads[i], NULL, increment, NULL);
    }
    
    for (int i = 0; i < 10; i++) {
        pthread_join(threads[i], NULL);
    }
    
    printf("Counter: %d\n", counter);  // 예상: 10000, 실제: 10000 미만 가능
    return 0;
}
```

### C#

```csharp
using System;
using System.Threading;

class RaceConditionExample
{
    private static int counter = 0;
    
    static void Increment()
    {
        for (int i = 0; i < 1000; i++)
        {
            counter++;  // Race condition 발생 가능
        }
    }
    
    static void Main(string[] args)
    {
        Thread[] threads = new Thread[10];
        
        for (int i = 0; i < 10; i++)
        {
            threads[i] = new Thread(Increment);
            threads[i].Start();
        }
        
        for (int i = 0; i < 10; i++)
        {
            threads[i].Join();
        }
        
        Console.WriteLine($"Counter: {counter}");  // 예상: 10000, 실제: 10000 미만 가능
    }
}
```

### Rust

```rust
use std::thread;

static mut COUNTER: i32 = 0;

fn increment() {
    for _ in 0..1000 {
        unsafe {
            COUNTER += 1;  // Race condition 발생 가능
        }
    }
}

fn main() {
    let mut handles = vec![];
    
    for _ in 0..10 {
        let handle = thread::spawn(|| {
            increment();
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    unsafe {
        println!("Counter: {}", COUNTER);  // 예상: 10000, 실제: 10000 미만 가능
    }
}
```

## 해결 방법

### Lock 순서 통일

데드락을 방지하기 위해 항상 같은 순서로 lock을 획득합니다.

```java
class DeadlockPrevention {
    private final Object lock1 = new Object();
    private final Object lock2 = new Object();
    
    public void method1() {
        synchronized(lock1) {  // 항상 lock1 먼저
            synchronized(lock2) {
                // 작업 수행
            }
        }
    }
    
    public void method2() {
        synchronized(lock1) {  // lock1 먼저 (순서 통일)
            synchronized(lock2) {
                // 작업 수행
            }
        }
    }
}
```

```kotlin
import kotlinx.coroutines.sync.*

val mutex1 = Mutex()
val mutex2 = Mutex()

suspend fun method1() {
    mutex1.lock()  // 항상 mutex1 먼저
    mutex2.lock()
    // 작업 수행
    mutex2.unlock()
    mutex1.unlock()
}

suspend fun method2() {
    mutex1.lock()  // mutex1 먼저 (순서 통일)
    mutex2.lock()
    // 작업 수행
    mutex2.unlock()
    mutex1.unlock()
}
```

```swift
import Foundation

let lock1 = NSLock()
let lock2 = NSLock()

func method1() {
    lock1.lock()  // 항상 lock1 먼저
    lock2.lock()
    // 작업 수행
    lock2.unlock()
    lock1.unlock()
}

func method2() {
    lock1.lock()  // lock1 먼저 (순서 통일)
    lock2.lock()
    // 작업 수행
    lock2.unlock()
    lock1.unlock()
}
```

```go
package main

import "sync"

var mutex1, mutex2 sync.Mutex

func method1() {
    mutex1.Lock()  // 항상 mutex1 먼저
    mutex2.Lock()
    // 작업 수행
    mutex2.Unlock()
    mutex1.Unlock()
}

func method2() {
    mutex1.Lock()  // mutex1 먼저 (순서 통일)
    mutex2.Lock()
    // 작업 수행
    mutex2.Unlock()
    mutex1.Unlock()
}
```

### Atomic 연산 사용

Race condition을 방지하기 위해 atomic 연산을 사용합니다.

```java
import java.util.concurrent.atomic.AtomicInteger;

class RaceConditionFix {
    private AtomicInteger counter = new AtomicInteger(0);
    
    public void increment() {
        counter.incrementAndGet();  // Atomic 연산
    }
    
    public int getCounter() {
        return counter.get();
    }
}
```

```python
import threading

counter = 0
lock = threading.Lock()

def increment():
    global counter
    for _ in range(1000):
        with lock:  # Lock 사용
            counter += 1
```

```cpp
#include <atomic>
#include <thread>

std::atomic<int> counter(0);

void increment() {
    for (int i = 0; i < 1000; i++) {
        counter++;  // Atomic 연산
    }
}
```

```go
package main

import (
    "fmt"
    "sync"
    "sync/atomic"
)

var counter int64

func increment() {
    for i := 0; i < 1000; i++ {
        atomic.AddInt64(&counter, 1)  // Atomic 연산
    }
}

func main() {
    var wg sync.WaitGroup
    
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            increment()
        }()
    }
    
    wg.Wait()
    fmt.Printf("Counter: %d\n", counter)
}
```

```kotlin
import java.util.concurrent.atomic.AtomicInteger

val counter = AtomicInteger(0)

fun increment() {
    for (i in 0 until 1000) {
        counter.incrementAndGet()  // Atomic 연산
    }
}
```

```swift
import Foundation

let counter = OSAllocatedUnfairLock(initialState: 0)

func increment() {
    for _ in 0..<1000 {
        counter.withLock { $0 += 1 }  // Lock 사용
    }
}
```

```typescript
// SharedArrayBuffer와 Atomics 사용
const buffer = new SharedArrayBuffer(4);
const view = new Int32Array(buffer);

function increment() {
    for (let i = 0; i < 1000; i++) {
        Atomics.add(view, 0, 1);  // Atomic 연산
    }
}
```

```c
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>

atomic_int counter = 0;

void* increment(void* arg) {
    for (int i = 0; i < 1000; i++) {
        atomic_fetch_add(&counter, 1);  // Atomic 연산
    }
    return NULL;
}
```

```csharp
using System.Threading;

class RaceConditionFix
{
    private int counter = 0;
    private readonly object lockObject = new object();
    
    public void Increment()
    {
        for (int i = 0; i < 1000; i++)
        {
            Interlocked.Increment(ref counter);  // Atomic 연산
        }
    }
}
```

```rust
use std::sync::atomic::{AtomicI32, Ordering};
use std::thread;

static COUNTER: AtomicI32 = AtomicI32::new(0);

fn increment() {
    for _ in 0..1000 {
        COUNTER.fetch_add(1, Ordering::SeqCst);  // Atomic 연산
    }
}

fn main() {
    let mut handles = vec![];
    
    for _ in 0..10 {
        let handle = thread::spawn(|| {
            increment();
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Counter: {}", COUNTER.load(Ordering::SeqCst));
}
```

### 불변성 보장

공유 데이터를 불변(immutable)으로 만들어 race condition을 방지합니다.

```java
// 불변 클래스
final class ImmutableCounter {
    private final int value;
    
    public ImmutableCounter(int value) {
        this.value = value;
    }
    
    public ImmutableCounter increment() {
        return new ImmutableCounter(value + 1);  // 새 객체 반환
    }
    
    public int getValue() {
        return value;
    }
}
```

```kotlin
// 불변 데이터 클래스
data class ImmutableCounter(val value: Int) {
    fun increment() = ImmutableCounter(value + 1)  // 새 객체 반환
}
```

```swift
// 불변 구조체
struct ImmutableCounter {
    let value: Int
    
    func increment() -> ImmutableCounter {
        return ImmutableCounter(value: value + 1)  // 새 객체 반환
    }
}
```

```python
from dataclasses import dataclass

@dataclass(frozen=True)  # 불변 클래스
class ImmutableCounter:
    value: int
    
    def increment(self):
        return ImmutableCounter(self.value + 1)  # 새 객체 반환
```

```rust
// 불변 구조체
#[derive(Clone, Copy)]
struct ImmutableCounter {
    value: i32,
}

impl ImmutableCounter {
    fn increment(self) -> ImmutableCounter {
        ImmutableCounter { value: self.value + 1 }  // 새 객체 반환
    }
}
```
