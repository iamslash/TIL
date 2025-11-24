- [Thread and Process](#thread-and-process)
  - [Java](#java)
    - [Thread 생성](#thread-생성)
    - [Thread Pool](#thread-pool)
    - [Process 생성](#process-생성)
  - [Kotlin](#kotlin)
    - [Thread 생성](#thread-생성-1)
    - [Coroutine (경량 스레드)](#coroutine-경량-스레드)
  - [Swift](#swift)
    - [Thread 생성](#thread-생성-2)
    - [Task (비동기 작업)](#task-비동기-작업)
  - [Python 3](#python-3)
    - [Thread 생성](#thread-생성-3)
    - [Process 생성](#process-생성-1)
  - [TypeScript](#typescript)
    - [Worker Thread](#worker-thread)
  - [C++](#c)
    - [Thread 생성](#thread-생성-4)
    - [Process 생성](#process-생성-2)
  - [C](#c-1)
    - [Thread 생성 (pthread)](#thread-생성-pthread)
    - [Process 생성 (fork)](#process-생성-fork)
  - [C#](#c-2)
    - [Thread 생성](#thread-생성-5)
    - [Task (비동기 작업)](#task-비동기-작업-1)
  - [Go](#go)
    - [Goroutine](#goroutine)
  - [Rust](#rust)
    - [Thread 생성](#thread-생성-6)
    - [Async Task](#async-task)

--------

# Thread and Process

스레드와 프로세스는 동시성을 구현하는 기본 단위입니다. 프로세스는 독립적인 메모리 공간을 가지는 실행 단위이고, 스레드는 프로세스 내에서 공유 메모리를 사용하는 실행 단위입니다.

## Java

Java는 `Thread` 클래스와 `ProcessBuilder`를 사용하여 스레드와 프로세스를 생성할 수 있습니다.

### Thread 생성

```java
// Thread 클래스 상속
class MyThread extends Thread {
    public void run() {
        System.out.println("Thread running: " + Thread.currentThread().getName());
    }
}

// Runnable 인터페이스 구현
class MyRunnable implements Runnable {
    public void run() {
        System.out.println("Runnable running: " + Thread.currentThread().getName());
    }
}

public class ThreadExample {
    public static void main(String[] args) {
        // 방법 1: Thread 상속
        Thread t1 = new MyThread();
        t1.start();
        
        // 방법 2: Runnable 구현
        Thread t2 = new Thread(new MyRunnable());
        t2.start();
        
        // 방법 3: Lambda 표현식
        Thread t3 = new Thread(() -> {
            System.out.println("Lambda thread: " + Thread.currentThread().getName());
        });
        t3.start();
        
        try {
            t1.join();
            t2.join();
            t3.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
```

### Thread Pool

```java
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ThreadPoolExample {
    public static void main(String[] args) {
        ExecutorService executor = Executors.newFixedThreadPool(5);
        
        for (int i = 0; i < 10; i++) {
            final int taskId = i;
            executor.submit(() -> {
                System.out.println("Task " + taskId + " running on: " + 
                    Thread.currentThread().getName());
            });
        }
        
        executor.shutdown();
    }
}
```

### Process 생성

```java
import java.io.IOException;

public class ProcessExample {
    public static void main(String[] args) {
        try {
            ProcessBuilder pb = new ProcessBuilder("ls", "-l");
            Process process = pb.start();
            
            int exitCode = process.waitFor();
            System.out.println("Exit code: " + exitCode);
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }
}
```

## Kotlin

Kotlin은 Java의 Thread API를 사용할 수 있으며, coroutine을 통해 경량 스레드를 제공합니다.

### Thread 생성

```kotlin
fun main() {
    val thread = Thread {
        println("Thread running: ${Thread.currentThread().name}")
    }
    thread.start()
    thread.join()
}
```

### Coroutine (경량 스레드)

```kotlin
import kotlinx.coroutines.*

fun main() = runBlocking {
    launch {
        delay(1000L)
        println("Coroutine running: ${Thread.currentThread().name}")
    }
    println("Main thread: ${Thread.currentThread().name}")
    delay(2000L)
}
```

## Swift

Swift는 `Thread` 클래스와 `Task`를 사용합니다.

### Thread 생성

```swift
import Foundation

class MyThread: Thread {
    override func main() {
        print("Thread running: \(Thread.current.name ?? "unnamed")")
    }
}

let thread = MyThread()
thread.name = "MyThread"
thread.start()
thread.cancel()
```

### Task (비동기 작업)

```swift
import Foundation

Task {
    print("Task running on: \(Thread.current)")
    try? await Task.sleep(nanoseconds: 1_000_000_000)
    print("Task completed")
}
```

## Python 3

Python은 `threading` 모듈과 `multiprocessing` 모듈을 제공합니다.

### Thread 생성

```python
import threading
import time

def worker():
    print(f"Thread running: {threading.current_thread().name}")
    time.sleep(1)

thread = threading.Thread(target=worker, name="WorkerThread")
thread.start()
thread.join()
```

### Process 생성

```python
import multiprocessing
import os

def worker():
    print(f"Process ID: {os.getpid()}")
    print(f"Parent Process ID: {os.getppid()}")

if __name__ == "__main__":
    process = multiprocessing.Process(target=worker)
    process.start()
    process.join()
```

## TypeScript

TypeScript는 Node.js의 `worker_threads` 모듈을 사용합니다.

### Worker Thread

```typescript
import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';

if (isMainThread) {
    const worker = new Worker(__filename, {
        workerData: { message: 'Hello from main thread' }
    });
    
    worker.on('message', (msg) => {
        console.log('Main thread received:', msg);
    });
} else {
    console.log('Worker thread:', workerData.message);
    parentPort?.postMessage('Hello from worker thread');
}
```

## C++

C++11부터 `<thread>` 헤더를 통해 스레드를 지원합니다.

### Thread 생성

```cpp
#include <thread>
#include <iostream>

void threadFunction() {
    std::cout << "Thread running: " << std::this_thread::get_id() << std::endl;
}

int main() {
    std::thread t1(threadFunction);
    std::thread t2([]() {
        std::cout << "Lambda thread: " << std::this_thread::get_id() << std::endl;
    });
    
    t1.join();
    t2.join();
    
    return 0;
}
```

### Process 생성

```cpp
#include <unistd.h>
#include <sys/wait.h>
#include <iostream>

int main() {
    pid_t pid = fork();
    
    if (pid == 0) {
        // Child process
        std::cout << "Child process PID: " << getpid() << std::endl;
        execl("/bin/ls", "ls", "-l", nullptr);
    } else if (pid > 0) {
        // Parent process
        std::cout << "Parent process PID: " << getpid() << std::endl;
        wait(nullptr);
    }
    
    return 0;
}
```

## C

C는 `pthread` 라이브러리와 `fork` 시스템 콜을 사용합니다.

### Thread 생성 (pthread)

```c
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

void* threadFunction(void* arg) {
    printf("Thread running: %lu\n", pthread_self());
    return NULL;
}

int main() {
    pthread_t thread;
    pthread_create(&thread, NULL, threadFunction, NULL);
    pthread_join(thread, NULL);
    return 0;
}
```

### Process 생성 (fork)

```c
#include <unistd.h>
#include <sys/wait.h>
#include <stdio.h>

int main() {
    pid_t pid = fork();
    
    if (pid == 0) {
        printf("Child process PID: %d\n", getpid());
        execl("/bin/ls", "ls", "-l", NULL);
    } else if (pid > 0) {
        printf("Parent process PID: %d\n", getpid());
        wait(NULL);
    }
    
    return 0;
}
```

## C#

C#은 `Thread` 클래스와 `Task`를 제공합니다.

### Thread 생성

```csharp
using System;
using System.Threading;

class Program
{
    static void ThreadFunction()
    {
        Console.WriteLine($"Thread running: {Thread.CurrentThread.Name}");
    }

    static void Main(string[] args)
    {
        Thread thread = new Thread(ThreadFunction);
        thread.Name = "MyThread";
        thread.Start();
        thread.Join();
    }
}
```

### Task (비동기 작업)

```csharp
using System;
using System.Threading.Tasks;

class Program
{
    static async Task Main(string[] args)
    {
        Task task = Task.Run(() => {
            Console.WriteLine($"Task running on thread: {Thread.CurrentThread.ManagedThreadId}");
        });
        
        await task;
    }
}
```

## Go

Go는 `goroutine`을 통해 경량 스레드를 제공합니다.

### Goroutine

```go
package main

import (
    "fmt"
    "time"
)

func worker(id int) {
    fmt.Printf("Goroutine %d running\n", id)
    time.Sleep(1 * time.Second)
}

func main() {
    for i := 0; i < 5; i++ {
        go worker(i)
    }
    
    time.Sleep(2 * time.Second)
}
```

## Rust

Rust는 `std::thread`와 async 런타임을 제공합니다.

### Thread 생성

```rust
use std::thread;

fn main() {
    let handle = thread::spawn(|| {
        println!("Thread running: {:?}", thread::current().id());
    });
    
    handle.join().unwrap();
}
```

### Async Task

```rust
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    let task = tokio::spawn(async {
        println!("Async task running");
        sleep(Duration::from_secs(1)).await;
    });
    
    task.await.unwrap();
}
```

