- [Concurrency Patterns](#concurrency-patterns)
  - [Producer-Consumer Pattern](#producer-consumer-pattern)
    - [Java](#java)
    - [Python 3](#python-3)
    - [Kotlin](#kotlin)
    - [Swift](#swift)
    - [TypeScript](#typescript)
    - [C++](#c)
    - [C](#c-1)
    - [C#](#c-2)
    - [Go](#go)
    - [Rust](#rust)
  - [Reader-Writer Pattern](#reader-writer-pattern)
    - [Java](#java-1)
    - [Kotlin](#kotlin-1)
    - [Swift](#swift-1)
    - [Python 3](#python-3-1)
    - [TypeScript](#typescript-1)
    - [C++](#c-3)
    - [C](#c-4)
    - [C#](#c-5)
    - [Go](#go-1)
    - [Rust](#rust-1)
  - [Worker Pool Pattern](#worker-pool-pattern)
    - [Java](#java-2)
    - [Kotlin](#kotlin-2)
    - [Swift](#swift-2)
    - [Python 3](#python-3-2)
    - [TypeScript](#typescript-2)
    - [C++](#c-6)
    - [C](#c-7)
    - [C#](#c-8)
    - [Go](#go-2)
    - [Rust](#rust-2)
  - [Future/Promise Pattern](#futurepromise-pattern)
    - [Java](#java-3)
    - [Kotlin](#kotlin-3)
    - [Swift](#swift-3)
    - [Python 3](#python-3-3)
    - [TypeScript](#typescript-3)
    - [C++](#c-9)
    - [C](#c-10)
    - [C#](#c-11)
    - [Go](#go-3)
    - [Rust](#rust-3)
  - [Actor Pattern](#actor-pattern)
    - [Java (Akka)](#java-akka)
    - [Kotlin](#kotlin-4)
    - [Swift](#swift-4)
    - [Python 3](#python-3-4)
    - [TypeScript](#typescript-4)
    - [C++](#c-12)
    - [C](#c-13)
    - [C#](#c-14)
    - [Go](#go-4)
    - [Rust (Actix)](#rust-actix)

--------

# Concurrency Patterns

동시성 제어 패턴은 동시성 프로그래밍에서 자주 사용되는 설계 패턴입니다.

## Producer-Consumer Pattern

생산자-소비자 패턴은 생산자가 데이터를 생성하고 소비자가 데이터를 처리하는 패턴입니다.

### Java

```java
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

class Producer implements Runnable {
    private final BlockingQueue<Integer> queue;
    
    public Producer(BlockingQueue<Integer> queue) {
        this.queue = queue;
    }
    
    public void run() {
        try {
            for (int i = 0; i < 10; i++) {
                queue.put(i);
                System.out.println("Produced: " + i);
                Thread.sleep(100);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}

class Consumer implements Runnable {
    private final BlockingQueue<Integer> queue;
    
    public Consumer(BlockingQueue<Integer> queue) {
        this.queue = queue;
    }
    
    public void run() {
        try {
            while (true) {
                Integer item = queue.take();
                System.out.println("Consumed: " + item);
                if (item == 9) break;
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}

public class ProducerConsumerExample {
    public static void main(String[] args) {
        BlockingQueue<Integer> queue = new LinkedBlockingQueue<>(5);
        
        Thread producer = new Thread(new Producer(queue));
        Thread consumer = new Thread(new Consumer(queue));
        
        producer.start();
        consumer.start();
    }
}
```

### Python 3

```python
import threading
import queue
import time

def producer(q):
    for i in range(10):
        q.put(i)
        print(f"Produced: {i}")
        time.sleep(0.1)

def consumer(q):
    while True:
        item = q.get()
        if item is None:
            break
        print(f"Consumed: {item}")
        q.task_done()

q = queue.Queue(maxsize=5)
t1 = threading.Thread(target=producer, args=(q,))
t2 = threading.Thread(target=consumer, args=(q,))

t1.start()
t2.start()

t1.join()
q.put(None)  # 종료 신호
t2.join()
```

### Kotlin

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.*

suspend fun producer(channel: Channel<Int>) {
    for (i in 0 until 10) {
        channel.send(i)
        println("Produced: $i")
        delay(100L)
    }
    channel.close()
}

suspend fun consumer(channel: Channel<Int>) {
    for (item in channel) {
        println("Consumed: $item")
    }
}

fun main() = runBlocking {
    val channel = Channel<Int>(5)
    launch { producer(channel) }
    launch { consumer(channel) }
    delay(2000L)
}
```

### Swift

```swift
import Foundation

let queue = DispatchQueue(label: "producer-consumer", attributes: .concurrent)
let semaphore = DispatchSemaphore(value: 5)
let group = DispatchGroup()

func producer() {
    for i in 0..<10 {
        semaphore.wait()
        queue.async {
            print("Produced: \(i)")
            semaphore.signal()
        }
    }
}

func consumer() {
    for i in 0..<10 {
        queue.async {
            print("Consumed: \(i)")
        }
    }
}

producer()
consumer()
```

### TypeScript

```typescript
class Queue<T> {
    private items: T[] = [];
    private maxSize: number;
    
    constructor(maxSize: number) {
        this.maxSize = maxSize;
    }
    
    async put(item: T): Promise<void> {
        while (this.items.length >= this.maxSize) {
            await new Promise(resolve => setTimeout(resolve, 10));
        }
        this.items.push(item);
    }
    
    async take(): Promise<T> {
        while (this.items.length === 0) {
            await new Promise(resolve => setTimeout(resolve, 10));
        }
        return this.items.shift()!;
    }
}

async function producer(queue: Queue<number>) {
    for (let i = 0; i < 10; i++) {
        await queue.put(i);
        console.log(`Produced: ${i}`);
        await new Promise(resolve => setTimeout(resolve, 100));
    }
}

async function consumer(queue: Queue<number>) {
    for (let i = 0; i < 10; i++) {
        const item = await queue.take();
        console.log(`Consumed: ${item}`);
    }
}

const queue = new Queue<number>(5);
producer(queue);
consumer(queue);
```

### C++

```cpp
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <iostream>

template<typename T>
class BlockingQueue {
private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
    size_t max_size_;
    
public:
    BlockingQueue(size_t max_size) : max_size_(max_size) {}
    
    void put(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this] { return queue_.size() < max_size_; });
        queue_.push(item);
        not_empty_.notify_one();
    }
    
    T take() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return !queue_.empty(); });
        T item = queue_.front();
        queue_.pop();
        not_full_.notify_one();
        return item;
    }
};

void producer(BlockingQueue<int>& queue) {
    for (int i = 0; i < 10; i++) {
        queue.put(i);
        std::cout << "Produced: " << i << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void consumer(BlockingQueue<int>& queue) {
    for (int i = 0; i < 10; i++) {
        int item = queue.take();
        std::cout << "Consumed: " << item << std::endl;
    }
}

int main() {
    BlockingQueue<int> queue(5);
    std::thread t1(producer, std::ref(queue));
    std::thread t2(consumer, std::ref(queue));
    
    t1.join();
    t2.join();
    
    return 0;
}
```

### C

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define QUEUE_SIZE 5

typedef struct {
    int items[QUEUE_SIZE];
    int front, rear, count;
    pthread_mutex_t mutex;
    pthread_cond_t not_full, not_empty;
} BlockingQueue;

void queue_init(BlockingQueue* q) {
    q->front = q->rear = q->count = 0;
    pthread_mutex_init(&q->mutex, NULL);
    pthread_cond_init(&q->not_full, NULL);
    pthread_cond_init(&q->not_empty, NULL);
}

void queue_put(BlockingQueue* q, int item) {
    pthread_mutex_lock(&q->mutex);
    while (q->count >= QUEUE_SIZE) {
        pthread_cond_wait(&q->not_full, &q->mutex);
    }
    q->items[q->rear] = item;
    q->rear = (q->rear + 1) % QUEUE_SIZE;
    q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mutex);
}

int queue_take(BlockingQueue* q) {
    pthread_mutex_lock(&q->mutex);
    while (q->count == 0) {
        pthread_cond_wait(&q->not_empty, &q->mutex);
    }
    int item = q->items[q->front];
    q->front = (q->front + 1) % QUEUE_SIZE;
    q->count--;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mutex);
    return item;
}

void* producer(void* arg) {
    BlockingQueue* q = (BlockingQueue*)arg;
    for (int i = 0; i < 10; i++) {
        queue_put(q, i);
        printf("Produced: %d\n", i);
        usleep(100000);
    }
    return NULL;
}

void* consumer(void* arg) {
    BlockingQueue* q = (BlockingQueue*)arg;
    for (int i = 0; i < 10; i++) {
        int item = queue_take(q);
        printf("Consumed: %d\n", item);
    }
    return NULL;
}

int main() {
    BlockingQueue q;
    queue_init(&q);
    
    pthread_t t1, t2;
    pthread_create(&t1, NULL, producer, &q);
    pthread_create(&t2, NULL, consumer, &q);
    
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    
    return 0;
}
```

### C#

```csharp
using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;

class ProducerConsumerExample
{
    static void Main(string[] args)
    {
        BlockingCollection<int> queue = new BlockingCollection<int>(5);
        
        Task producer = Task.Run(() => {
            for (int i = 0; i < 10; i++) {
                queue.Add(i);
                Console.WriteLine($"Produced: {i}");
                Thread.Sleep(100);
            }
            queue.CompleteAdding();
        });
        
        Task consumer = Task.Run(() => {
            foreach (int item in queue.GetConsumingEnumerable()) {
                Console.WriteLine($"Consumed: {item}");
            }
        });
        
        Task.WaitAll(producer, consumer);
    }
}
```

### Go

```go
package main

import (
    "fmt"
    "time"
)

func producer(ch chan<- int) {
    for i := 0; i < 10; i++ {
        ch <- i
        fmt.Printf("Produced: %d\n", i)
        time.Sleep(100 * time.Millisecond)
    }
    close(ch)
}

func consumer(ch <-chan int) {
    for item := range ch {
        fmt.Printf("Consumed: %d\n", item)
    }
}

func main() {
    ch := make(chan int, 5)
    go producer(ch)
    consumer(ch)
}
```

### Rust

```rust
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn main() {
    let (tx, rx) = mpsc::channel();
    
    let producer = thread::spawn(move || {
        for i in 0..10 {
            tx.send(i).unwrap();
            println!("Produced: {}", i);
            thread::sleep(Duration::from_millis(100));
        }
    });
    
    let consumer = thread::spawn(move || {
        for received in rx {
            println!("Consumed: {}", received);
        }
    });
    
    producer.join().unwrap();
    consumer.join().unwrap();
}
```

## Reader-Writer Pattern

Reader-Writer 패턴은 여러 reader가 동시에 읽을 수 있지만, writer는 단독으로 쓸 수 있는 패턴입니다.

### Java

```java
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

class DataStore {
    private int data = 0;
    private final ReadWriteLock lock = new ReentrantReadWriteLock();
    
    public int read() {
        lock.readLock().lock();
        try {
            return data;
        } finally {
            lock.readLock().unlock();
        }
    }
    
    public void write(int value) {
        lock.writeLock().lock();
        try {
            data = value;
        } finally {
            lock.writeLock().unlock();
        }
    }
}

class Reader implements Runnable {
    private final DataStore store;
    
    public Reader(DataStore store) {
        this.store = store;
    }
    
    public void run() {
        for (int i = 0; i < 5; i++) {
            System.out.println("Reader: " + store.read());
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
}

class Writer implements Runnable {
    private final DataStore store;
    
    public Writer(DataStore store) {
        this.store = store;
    }
    
    public void run() {
        for (int i = 0; i < 5; i++) {
            store.write(i);
            System.out.println("Writer: " + i);
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
}

public class ReaderWriterExample {
    public static void main(String[] args) {
        DataStore store = new DataStore();
        
        Thread reader1 = new Thread(new Reader(store));
        Thread reader2 = new Thread(new Reader(store));
        Thread writer = new Thread(new Writer(store));
        
        reader1.start();
        reader2.start();
        writer.start();
    }
}
```

### Kotlin

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.*

class DataStore {
    private var data = 0
    private val rwLock = java.util.concurrent.locks.ReentrantReadWriteLock()
    
    fun read(): Int {
        rwLock.readLock().lock()
        try {
            return data
        } finally {
            rwLock.readLock().unlock()
        }
    }
    
    fun write(value: Int) {
        rwLock.writeLock().lock()
        try {
            data = value
        } finally {
            rwLock.writeLock().unlock()
        }
    }
}

fun main() = runBlocking {
    val store = DataStore()
    
    launch {
        repeat(5) {
            println("Reader 1: ${store.read()}")
            delay(100L)
        }
    }
    
    launch {
        repeat(5) {
            println("Reader 2: ${store.read()}")
            delay(100L)
        }
    }
    
    launch {
        repeat(5) {
            store.write(it)
            println("Writer: $it")
            delay(200L)
        }
    }
    
    delay(2000L)
}
```

### Swift

```swift
import Foundation

class DataStore {
    private var data = 0
    private let queue = DispatchQueue(label: "dataStore", attributes: .concurrent)
    
    func read() -> Int {
        return queue.sync {
            return data
        }
    }
    
    func write(_ value: Int) {
        queue.async(flags: .barrier) {
            self.data = value
        }
    }
}

let store = DataStore()

DispatchQueue.global().async {
    for i in 0..<5 {
        print("Reader 1: \(store.read())")
        Thread.sleep(forTimeInterval: 0.1)
    }
}

DispatchQueue.global().async {
    for i in 0..<5 {
        print("Reader 2: \(store.read())")
        Thread.sleep(forTimeInterval: 0.1)
    }
}

DispatchQueue.global().async {
    for i in 0..<5 {
        store.write(i)
        print("Writer: \(i)")
        Thread.sleep(forTimeInterval: 0.2)
    }
}

Thread.sleep(forTimeInterval: 2.0)
```

### Python 3

```python
import threading
import time

class DataStore:
    def __init__(self):
        self.data = 0
        self.rw_lock = threading.RWLock()
    
    def read(self):
        with self.rw_lock.reader():
            return self.data
    
    def write(self, value):
        with self.rw_lock.writer():
            self.data = value

store = DataStore()

def reader(id):
    for i in range(5):
        print(f"Reader {id}: {store.read()}")
        time.sleep(0.1)

def writer():
    for i in range(5):
        store.write(i)
        print(f"Writer: {i}")
        time.sleep(0.2)

t1 = threading.Thread(target=reader, args=(1,))
t2 = threading.Thread(target=reader, args=(2,))
t3 = threading.Thread(target=writer)

t1.start()
t2.start()
t3.start()

t1.join()
t2.join()
t3.join()
```

### TypeScript

```typescript
class DataStore {
    private data: number = 0;
    private readers: number = 0;
    private writing: boolean = false;
    private mutex: Promise<void> = Promise.resolve();
    
    async read(): Promise<number> {
        await this.mutex;
        this.readers++;
        const result = this.data;
        this.readers--;
        return result;
    }
    
    async write(value: number): Promise<void> {
        await this.mutex;
        while (this.readers > 0 || this.writing) {
            await new Promise(resolve => setTimeout(resolve, 10));
        }
        this.writing = true;
        this.data = value;
        this.writing = false;
    }
}

const store = new DataStore();

async function reader(id: number) {
    for (let i = 0; i < 5; i++) {
        const value = await store.read();
        console.log(`Reader ${id}: ${value}`);
        await new Promise(resolve => setTimeout(resolve, 100));
    }
}

async function writer() {
    for (let i = 0; i < 5; i++) {
        await store.write(i);
        console.log(`Writer: ${i}`);
        await new Promise(resolve => setTimeout(resolve, 200));
    }
}

reader(1);
reader(2);
writer();
```

### C++

```cpp
#include <shared_mutex>
#include <thread>
#include <iostream>

class DataStore {
private:
    int data = 0;
    std::shared_mutex mutex;
    
public:
    int read() {
        std::shared_lock<std::shared_mutex> lock(mutex);
        return data;
    }
    
    void write(int value) {
        std::unique_lock<std::shared_mutex> lock(mutex);
        data = value;
    }
};

void reader(DataStore& store, int id) {
    for (int i = 0; i < 5; i++) {
        std::cout << "Reader " << id << ": " << store.read() << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void writer(DataStore& store) {
    for (int i = 0; i < 5; i++) {
        store.write(i);
        std::cout << "Writer: " << i << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

int main() {
    DataStore store;
    
    std::thread reader1(reader, std::ref(store), 1);
    std::thread reader2(reader, std::ref(store), 2);
    std::thread writer_thread(writer, std::ref(store));
    
    reader1.join();
    reader2.join();
    writer_thread.join();
    
    return 0;
}
```

### C

```c
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

typedef struct {
    int data;
    pthread_rwlock_t rwlock;
} DataStore;

void* reader(void* arg) {
    DataStore* store = (DataStore*)arg;
    for (int i = 0; i < 5; i++) {
        pthread_rwlock_rdlock(&store->rwlock);
        printf("Reader: %d\n", store->data);
        pthread_rwlock_unlock(&store->rwlock);
        usleep(100000);
    }
    return NULL;
}

void* writer(void* arg) {
    DataStore* store = (DataStore*)arg;
    for (int i = 0; i < 5; i++) {
        pthread_rwlock_wrlock(&store->rwlock);
        store->data = i;
        printf("Writer: %d\n", i);
        pthread_rwlock_unlock(&store->rwlock);
        usleep(200000);
    }
    return NULL;
}

int main() {
    DataStore store = {0, PTHREAD_RWLOCK_INITIALIZER};
    
    pthread_t reader1, reader2, writer_thread;
    pthread_create(&reader1, NULL, reader, &store);
    pthread_create(&reader2, NULL, reader, &store);
    pthread_create(&writer_thread, NULL, writer, &store);
    
    pthread_join(reader1, NULL);
    pthread_join(reader2, NULL);
    pthread_join(writer_thread, NULL);
    
    return 0;
}
```

### C#

```csharp
using System;
using System.Threading;
using System.Threading.Tasks;

class DataStore
{
    private int data = 0;
    private readonly ReaderWriterLockSlim rwLock = new ReaderWriterLockSlim();
    
    public int Read()
    {
        rwLock.EnterReadLock();
        try {
            return data;
        } finally {
            rwLock.ExitReadLock();
        }
    }
    
    public void Write(int value)
    {
        rwLock.EnterWriteLock();
        try {
            data = value;
        } finally {
            rwLock.ExitWriteLock();
        }
    }
}

class Program
{
    static void Main(string[] args)
    {
        DataStore store = new DataStore();
        
        Task reader1 = Task.Run(() => {
            for (int i = 0; i < 5; i++) {
                Console.WriteLine($"Reader 1: {store.Read()}");
                Thread.Sleep(100);
            }
        });
        
        Task reader2 = Task.Run(() => {
            for (int i = 0; i < 5; i++) {
                Console.WriteLine($"Reader 2: {store.Read()}");
                Thread.Sleep(100);
            }
        });
        
        Task writer = Task.Run(() => {
            for (int i = 0; i < 5; i++) {
                store.Write(i);
                Console.WriteLine($"Writer: {i}");
                Thread.Sleep(200);
            }
        });
        
        Task.WaitAll(reader1, reader2, writer);
    }
}
```

### Go

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

type DataStore struct {
    data  int
    rwMutex sync.RWMutex
}

func (ds *DataStore) Read() int {
    ds.rwMutex.RLock()
    defer ds.rwMutex.RUnlock()
    return ds.data
}

func (ds *DataStore) Write(value int) {
    ds.rwMutex.Lock()
    defer ds.rwMutex.Unlock()
    ds.data = value
}

func reader(ds *DataStore, id int) {
    for i := 0; i < 5; i++ {
        value := ds.Read()
        fmt.Printf("Reader %d: %d\n", id, value)
        time.Sleep(100 * time.Millisecond)
    }
}

func writer(ds *DataStore) {
    for i := 0; i < 5; i++ {
        ds.Write(i)
        fmt.Printf("Writer: %d\n", i)
        time.Sleep(200 * time.Millisecond)
    }
}

func main() {
    store := &DataStore{}
    
    go reader(store, 1)
    go reader(store, 2)
    go writer(store)
    
    time.Sleep(2 * time.Second)
}
```

### Rust

```rust
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

fn main() {
    let store = Arc::new(RwLock::new(0));
    
    let store1 = Arc::clone(&store);
    let reader1 = thread::spawn(move || {
        for _ in 0..5 {
            let value = store1.read().unwrap();
            println!("Reader 1: {}", *value);
            thread::sleep(Duration::from_millis(100));
        }
    });
    
    let store2 = Arc::clone(&store);
    let reader2 = thread::spawn(move || {
        for _ in 0..5 {
            let value = store2.read().unwrap();
            println!("Reader 2: {}", *value);
            thread::sleep(Duration::from_millis(100));
        }
    });
    
    let writer = thread::spawn(move || {
        for i in 0..5 {
            let mut value = store.write().unwrap();
            *value = i;
            println!("Writer: {}", i);
            thread::sleep(Duration::from_millis(200));
        }
    });
    
    reader1.join().unwrap();
    reader2.join().unwrap();
    writer.join().unwrap();
}
```

## Worker Pool Pattern

Worker Pool 패턴은 고정된 수의 worker 스레드가 작업 큐에서 작업을 가져와 처리하는 패턴입니다.

### Java

```java
import java.util.concurrent.*;

class WorkerPoolExample {
    public static void main(String[] args) {
        ExecutorService executor = Executors.newFixedThreadPool(3);
        
        for (int i = 0; i < 10; i++) {
            final int taskId = i;
            executor.submit(() -> {
                System.out.println("Task " + taskId + " executed by " + 
                    Thread.currentThread().getName());
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            });
        }
        
        executor.shutdown();
        try {
            executor.awaitTermination(30, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            executor.shutdownNow();
        }
    }
}
```

### Kotlin

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.*

suspend fun worker(id: Int, jobs: ReceiveChannel<Int>, results: SendChannel<Int>) {
    for (job in jobs) {
        println("Worker $id processing job $job")
        delay(1000L)
        results.send(job * 2)
    }
}

fun main() = runBlocking {
    val jobs = Channel<Int>(Channel.UNLIMITED)
    val results = Channel<Int>(Channel.UNLIMITED)
    
    // 3개의 worker 생성
    val workers = List(3) { id ->
        launch {
            worker(id + 1, jobs, results)
        }
    }
    
    // 작업 전송
    for (j in 1..10) {
        jobs.send(j)
    }
    jobs.close()
    
    // 결과 수집
    launch {
        workers.forEach { it.join() }
        results.close()
    }
    
    for (result in results) {
        println("Result: $result")
    }
}
```

### Swift

```swift
import Foundation

let queue = OperationQueue()
queue.maxConcurrentOperationCount = 3

for i in 1...10 {
    queue.addOperation {
        print("Task \(i) executed by \(Thread.current)")
        Thread.sleep(forTimeInterval: 1.0)
    }
}

queue.waitUntilAllOperationsAreFinished()
```

### Python 3

```python
import concurrent.futures
import time

def process_task(task_id):
    print(f"Task {task_id} executed by {threading.current_thread().name}")
    time.sleep(1)
    return task_id * 2

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(process_task, i) for i in range(1, 11)]
    
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        print(f"Result: {result}")
```

### TypeScript

```typescript
class WorkerPool {
    private workers: Worker[] = [];
    private queue: Array<{task: number, resolve: (value: number) => void}> = [];
    private active = 0;
    
    constructor(private maxWorkers: number) {}
    
    async execute(task: number): Promise<number> {
        return new Promise((resolve) => {
            this.queue.push({task, resolve});
            this.process();
        });
    }
    
    private process() {
        if (this.active >= this.maxWorkers || this.queue.length === 0) {
            return;
        }
        
        this.active++;
        const {task, resolve} = this.queue.shift()!;
        
        setTimeout(() => {
            const result = task * 2;
            resolve(result);
            this.active--;
            this.process();
        }, 1000);
    }
}

const pool = new WorkerPool(3);
const promises = [];

for (let i = 1; i <= 10; i++) {
    promises.push(pool.execute(i).then(result => {
        console.log(`Result: ${result}`);
    }));
}

Promise.all(promises);
```

### C++

```cpp
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <vector>

class WorkerPool {
private:
    std::queue<int> jobs;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool stop_ = false;
    
public:
    void addJob(int job) {
        std::unique_lock<std::mutex> lock(mutex_);
        jobs.push(job);
        condition_.notify_one();
    }
    
    void stop() {
        std::unique_lock<std::mutex> lock(mutex_);
        stop_ = true;
        condition_.notify_all();
    }
    
    void worker(int id) {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { return !jobs.empty() || stop_; });
            
            if (stop_ && jobs.empty()) break;
            
            int job = jobs.front();
            jobs.pop();
            lock.unlock();
            
            std::cout << "Worker " << id << " processing job " << job << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
};

int main() {
    WorkerPool pool;
    std::vector<std::thread> workers;
    
    for (int i = 1; i <= 3; i++) {
        workers.emplace_back(&WorkerPool::worker, &pool, i);
    }
    
    for (int i = 1; i <= 10; i++) {
        pool.addJob(i);
    }
    
    pool.stop();
    
    for (auto& worker : workers) {
        worker.join();
    }
    
    return 0;
}
```

### C

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define MAX_WORKERS 3
#define MAX_JOBS 10

typedef struct {
    int jobs[MAX_JOBS];
    int front, rear, count;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    int stop;
} JobQueue;

void queue_init(JobQueue* q) {
    q->front = q->rear = q->count = q->stop = 0;
    pthread_mutex_init(&q->mutex, NULL);
    pthread_cond_init(&q->not_empty, NULL);
}

void queue_add(JobQueue* q, int job) {
    pthread_mutex_lock(&q->mutex);
    q->jobs[q->rear] = job;
    q->rear = (q->rear + 1) % MAX_JOBS;
    q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mutex);
}

int queue_get(JobQueue* q) {
    pthread_mutex_lock(&q->mutex);
    while (q->count == 0 && !q->stop) {
        pthread_cond_wait(&q->not_empty, &q->mutex);
    }
    if (q->stop && q->count == 0) {
        pthread_mutex_unlock(&q->mutex);
        return -1;
    }
    int job = q->jobs[q->front];
    q->front = (q->front + 1) % MAX_JOBS;
    q->count--;
    pthread_mutex_unlock(&q->mutex);
    return job;
}

void* worker(void* arg) {
    JobQueue* q = (JobQueue*)arg;
    int worker_id = (int)(long)arg;
    
    while (1) {
        int job = queue_get(q);
        if (job == -1) break;
        printf("Worker %d processing job %d\n", worker_id, job);
        sleep(1);
    }
    return NULL;
}

int main() {
    JobQueue q;
    queue_init(&q);
    
    pthread_t workers[MAX_WORKERS];
    for (int i = 0; i < MAX_WORKERS; i++) {
        pthread_create(&workers[i], NULL, worker, (void*)(long)(i + 1));
    }
    
    for (int i = 1; i <= MAX_JOBS; i++) {
        queue_add(&q, i);
    }
    
    q.stop = 1;
    pthread_cond_broadcast(&q.not_empty);
    
    for (int i = 0; i < MAX_WORKERS; i++) {
        pthread_join(workers[i], NULL);
    }
    
    return 0;
}
```

### C#

```csharp
using System;
using System.Threading;
using System.Threading.Tasks;

class WorkerPool
{
    private SemaphoreSlim semaphore;
    
    public WorkerPool(int maxWorkers)
    {
        semaphore = new SemaphoreSlim(maxWorkers, maxWorkers);
    }
    
    public async Task<int> ExecuteAsync(int taskId)
    {
        await semaphore.WaitAsync();
        try
        {
            Console.WriteLine($"Task {taskId} executed by {Thread.CurrentThread.ManagedThreadId}");
            await Task.Delay(1000);
            return taskId * 2;
        }
        finally
        {
            semaphore.Release();
        }
    }
}

class Program
{
    static async Task Main(string[] args)
    {
        WorkerPool pool = new WorkerPool(3);
        Task<int>[] tasks = new Task<int>[10];
        
        for (int i = 1; i <= 10; i++)
        {
            int taskId = i;
            tasks[i - 1] = pool.ExecuteAsync(taskId);
        }
        
        int[] results = await Task.WhenAll(tasks);
        foreach (var result in results)
        {
            Console.WriteLine($"Result: {result}");
        }
    }
}
```

### Go

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

func worker(id int, jobs <-chan int, results chan<- int, wg *sync.WaitGroup) {
    defer wg.Done()
    for job := range jobs {
        fmt.Printf("Worker %d processing job %d\n", id, job)
        time.Sleep(1 * time.Second)
        results <- job * 2
    }
}

func main() {
    jobs := make(chan int, 10)
    results := make(chan int, 10)
    
    var wg sync.WaitGroup
    
    // 3개의 worker 생성
    for w := 1; w <= 3; w++ {
        wg.Add(1)
        go worker(w, jobs, results, &wg)
    }
    
    // 작업 전송
    for j := 1; j <= 10; j++ {
        jobs <- j
    }
    close(jobs)
    
    // 결과 수집
    go func() {
        wg.Wait()
        close(results)
    }()
    
    for result := range results {
        fmt.Printf("Result: %d\n", result)
    }
}
```

### Rust

```rust
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn main() {
    let (tx, rx) = mpsc::channel();
    
    let pool = thread::scope(|s| {
        for worker_id in 1..=3 {
            let rx = rx.clone();
            s.spawn(move || {
                for job in rx {
                    println!("Worker {} processing job {}", worker_id, job);
                    thread::sleep(Duration::from_secs(1));
                }
            });
        }
    });
    
    for job in 1..=10 {
        tx.send(job).unwrap();
    }
    drop(tx);
}
```

## Future/Promise Pattern

Future/Promise 패턴은 비동기 작업의 결과를 나타내는 패턴입니다.

### Java

```java
import java.util.concurrent.CompletableFuture;

public class FuturePromiseExample {
    public static void main(String[] args) {
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> {
            try {
                Thread.sleep(1000);
                return "Hello";
            } catch (InterruptedException e) {
                return "Interrupted";
            }
        });
        
        future.thenApply(s -> s + " World")
              .thenApply(String::toUpperCase)
              .thenAccept(System.out::println);
        
        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
```

### Kotlin

```kotlin
import kotlinx.coroutines.*

fun createFuture(): Deferred<String> {
    return GlobalScope.async {
        delay(1000L)
        "Hello"
    }
}

fun main() = runBlocking {
    val future = createFuture()
    val result = future.await()
    println("$result World")
}
```

### Swift

```swift
import Foundation

func createPromise() async -> String {
    try? await Task.sleep(nanoseconds: 1_000_000_000)
    return "Hello"
}

Task {
    let promise = createPromise()
    let result = await promise
    print(result + " World")
}
```

### Python 3

```python
import asyncio

async def create_promise():
    await asyncio.sleep(1)
    return "Hello"

async def main():
    promise = create_promise()
    result = await promise
    print(result + " World")

asyncio.run(main())
```

### TypeScript

```typescript
function createPromise(): Promise<string> {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            resolve("Hello");
        }, 1000);
    });
}

async function main() {
    const promise = createPromise();
    const result = await promise;
    console.log(result + " World");
}

main();
```

### C++

```cpp
#include <future>
#include <iostream>
#include <thread>
#include <chrono>

std::string createPromise() {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return "Hello";
}

int main() {
    std::future<std::string> promise = std::async(std::launch::async, createPromise);
    std::string result = promise.get();
    std::cout << result << " World" << std::endl;
    return 0;
}
```

### C

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct {
    char* result;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int ready;
} Promise;

void* createPromise(void* arg) {
    Promise* p = (Promise*)arg;
    sleep(1);
    pthread_mutex_lock(&p->mutex);
    p->result = "Hello";
    p->ready = 1;
    pthread_cond_signal(&p->cond);
    pthread_mutex_unlock(&p->mutex);
    return NULL;
}

int main() {
    Promise p = {NULL, PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0};
    pthread_t thread;
    
    pthread_create(&thread, NULL, createPromise, &p);
    
    pthread_mutex_lock(&p.mutex);
    while (!p.ready) {
        pthread_cond_wait(&p.cond, &p.mutex);
    }
    printf("%s World\n", p.result);
    pthread_mutex_unlock(&p.mutex);
    
    pthread_join(thread, NULL);
    return 0;
}
```

### C#

```csharp
using System;
using System.Threading.Tasks;

class Program
{
    static async Task<string> CreatePromise()
    {
        await Task.Delay(1000);
        return "Hello";
    }
    
    static async Task Main(string[] args)
    {
        Task<string> promise = CreatePromise();
        string result = await promise;
        Console.WriteLine(result + " World");
    }
}
```

### Go

```go
package main

import (
    "fmt"
    "time"
)

func createPromise() <-chan string {
    ch := make(chan string, 1)
    go func() {
        time.Sleep(1 * time.Second)
        ch <- "Hello"
    }()
    return ch
}

func main() {
    promise := createPromise()
    result := <-promise
    fmt.Println(result + " World")
}
```

### Rust

```rust
use std::thread;
use std::time::Duration;
use std::sync::mpsc;

fn create_promise() -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(1));
        tx.send("Hello".to_string()).unwrap();
    });
    rx
}

fn main() {
    let promise = create_promise();
    let result = promise.recv().unwrap();
    println!("{} World", result);
}
```

## Actor Pattern

Actor 패턴은 각 actor가 독립적인 상태와 메시지 큐를 가지는 패턴입니다.

### Java (Akka)

```java
import akka.actor.AbstractActor;
import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;

class Greeter extends AbstractActor {
    @Override
    public Receive createReceive() {
        return receiveBuilder()
            .match(String.class, message -> {
                System.out.println("Received: " + message);
                getSender().tell("Hello back!", getSelf());
            })
            .build();
    }
}

public class ActorExample {
    public static void main(String[] args) {
        ActorSystem system = ActorSystem.create("MySystem");
        ActorRef greeter = system.actorOf(Props.create(Greeter.class), "greeter");
        
        // 메시지 전송
        greeter.tell("Hello", ActorRef.noSender());
        
        system.terminate();
    }
}
```

### Kotlin

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.*

class GreeterActor {
    private val mailbox = Channel<String>(Channel.UNLIMITED)
    
    suspend fun send(message: String) {
        mailbox.send(message)
    }
    
    suspend fun start() = coroutineScope {
        launch {
            for (message in mailbox) {
                println("Received: $message")
            }
        }
    }
}

fun main() = runBlocking {
    val greeter = GreeterActor()
    greeter.start()
    greeter.send("Hello")
    delay(100L)
}
```

### Swift

```swift
import Foundation

actor Greeter {
    func receive(_ message: String) {
        print("Received: \(message)")
    }
}

Task {
    let greeter = Greeter()
    await greeter.receive("Hello")
}
```

### Python 3

```python
import asyncio
from typing import Optional

class GreeterActor:
    def __init__(self):
        self.mailbox = asyncio.Queue()
        self.running = False
    
    async def send(self, message: str):
        await self.mailbox.put(message)
    
    async def start(self):
        self.running = True
        while self.running:
            message = await self.mailbox.get()
            print(f"Received: {message}")
    
    def stop(self):
        self.running = False

async def main():
    greeter = GreeterActor()
    asyncio.create_task(greeter.start())
    await greeter.send("Hello")
    await asyncio.sleep(0.1)
    greeter.stop()

asyncio.run(main())
```

### TypeScript

```typescript
class Actor {
    private mailbox: string[] = [];
    private processing = false;
    
    async send(message: string): Promise<void> {
        this.mailbox.push(message);
        if (!this.processing) {
            this.process();
        }
    }
    
    private async process(): Promise<void> {
        this.processing = true;
        while (this.mailbox.length > 0) {
            const message = this.mailbox.shift()!;
            console.log(`Received: ${message}`);
            await new Promise(resolve => setTimeout(resolve, 0));
        }
        this.processing = false;
    }
}

const greeter = new Actor();
greeter.send("Hello");
```

### C++

```cpp
#include <queue>
#include <mutex>
#include <thread>
#include <iostream>
#include <string>

class Actor {
private:
    std::queue<std::string> mailbox_;
    std::mutex mutex_;
    std::thread thread_;
    bool running_ = true;
    
    void process() {
        while (running_) {
            std::unique_lock<std::mutex> lock(mutex_);
            if (!mailbox_.empty()) {
                std::string message = mailbox_.front();
                mailbox_.pop();
                lock.unlock();
                std::cout << "Received: " << message << std::endl;
            }
        }
    }
    
public:
    Actor() : thread_(&Actor::process, this) {}
    
    void send(const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);
        mailbox_.push(message);
    }
    
    ~Actor() {
        running_ = false;
        thread_.join();
    }
};

int main() {
    Actor greeter;
    greeter.send("Hello");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return 0;
}
```

### C

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    char** messages;
    int count;
    int capacity;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int running;
} Actor;

void* actor_process(void* arg) {
    Actor* actor = (Actor*)arg;
    while (actor->running) {
        pthread_mutex_lock(&actor->mutex);
        while (actor->count == 0 && actor->running) {
            pthread_cond_wait(&actor->cond, &actor->mutex);
        }
        if (actor->count > 0) {
            printf("Received: %s\n", actor->messages[0]);
            memmove(actor->messages, actor->messages + 1, 
                   (actor->count - 1) * sizeof(char*));
            actor->count--;
        }
        pthread_mutex_unlock(&actor->mutex);
        usleep(10000);
    }
    return NULL;
}

void actor_send(Actor* actor, const char* message) {
    pthread_mutex_lock(&actor->mutex);
    if (actor->count < actor->capacity) {
        actor->messages[actor->count++] = strdup(message);
        pthread_cond_signal(&actor->cond);
    }
    pthread_mutex_unlock(&actor->mutex);
}

int main() {
    Actor actor = {NULL, 0, 10, PTHREAD_MUTEX_INITIALIZER, 
                   PTHREAD_COND_INITIALIZER, 1};
    actor.messages = malloc(10 * sizeof(char*));
    
    pthread_t thread;
    pthread_create(&thread, NULL, actor_process, &actor);
    
    actor_send(&actor, "Hello");
    sleep(1);
    
    actor.running = 0;
    pthread_cond_signal(&actor.cond);
    pthread_join(thread, NULL);
    
    return 0;
}
```

### C#

```csharp
using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;

class Actor
{
    private ConcurrentQueue<string> mailbox = new ConcurrentQueue<string>();
    private CancellationTokenSource cts = new CancellationTokenSource();
    
    public Actor()
    {
        Task.Run(() => Process(cts.Token));
    }
    
    public void Send(string message)
    {
        mailbox.Enqueue(message);
    }
    
    private async Task Process(CancellationToken token)
    {
        while (!token.IsCancellationRequested)
        {
            if (mailbox.TryDequeue(out string message))
            {
                Console.WriteLine($"Received: {message}");
            }
            await Task.Delay(10, token);
        }
    }
    
    public void Stop()
    {
        cts.Cancel();
    }
}

class Program
{
    static void Main(string[] args)
    {
        Actor greeter = new Actor();
        greeter.Send("Hello");
        Thread.Sleep(100);
        greeter.Stop();
    }
}
```

### Go

```go
package main

import (
    "fmt"
    "time"
)

type Actor struct {
    mailbox chan string
}

func NewActor() *Actor {
    a := &Actor{
        mailbox: make(chan string, 10),
    }
    go a.process()
    return a
}

func (a *Actor) Send(message string) {
    a.mailbox <- message
}

func (a *Actor) process() {
    for message := range a.mailbox {
        fmt.Printf("Received: %s\n", message)
    }
}

func main() {
    greeter := NewActor()
    greeter.Send("Hello")
    time.Sleep(100 * time.Millisecond)
    close(greeter.mailbox)
}
```

### Rust (Actix)

```rust
use actix::prelude::*;

struct Greeter;

impl Actor for Greeter {
    type Context = Context<Self>;
}

#[derive(Message)]
#[rtype(result = "String")]
struct Greet {
    name: String,
}

impl Handler<Greet> for Greeter {
    type Result = String;
    
    fn handle(&mut self, msg: Greet, _ctx: &mut Context<Self>) -> Self::Result {
        format!("Hello, {}!", msg.name)
    }
}

#[actix::main]
async fn main() {
    let greeter = Greeter.start();
    let result = greeter.send(Greet { name: "World".to_string() }).await;
    println!("{}", result.unwrap());
}
```

