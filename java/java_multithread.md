- [Materials](#materials)
- [Start Thread](#start-thread)
- [Runnable vs Thread](#runnable-vs-thread)
- [Status of Thread](#status-of-thread)
- [Kill Thread](#kill-thread)
- [Atomic Variables](#atomic-variables)
- [Syncronized](#syncronized)
- [wait(), notify(), notifyAll()](#wait-notify-notifyall)
- [Lock](#lock)
- [Condition](#condition)
- [CountDownLatch](#countdownlatch)
- [Semaphore](#semaphore)

-----

# Materials

* [Multithread design pattern @ slideshare](https://www.slideshare.net/ohyecloudy/multithread-design-pattern)
* [Guide to java.util.concurrent.Locks @ baeldung](https://www.baeldung.com/java-concurrent-locks)
* [Concurrency @ leetcode](https://leetcode.com/problemset/concurrency/)

# Start Thread

* [How to Start a Thread in Java @ baeldung](https://www.baeldung.com/java-start-thread)
* [Implementing a Runnable vs Extending a Thread](https://www.baeldung.com/java-runnable-vs-extending-thread)

-----

This is a simple way to create thread

```java
public class NewThread extends Thread {
    public void run() {
        long startTime = System.currentTimeMillis();
        int i = 0;
        while (true) {
            System.out.println(this.getName() + ": New Thread is running..." + i++);
            try {
                //Wait for one sec so it doesn't print too fast
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            ...
        }
    }
}

public class SingleThreadExample {
    public static void main(String[] args) {
        NewThread t = new NewThread();
        t.start();
    }
}
```

We use ExecutorService in a real world.

```java
ExecutorService executor = Executors.newFixedThreadPool(10);
...
executor.submit(() -> {
    new Task();
});
```

These are usages of them

```java
public class SimpleThread extends Thread {

    private String message;

    // standard logger, constructor

    @Override
    public void run() {
        log.info(message);
    }
}

@Test
public void givenAThread_whenRunIt_thenResult()
  throws Exception {
 
    Thread thread = new SimpleThread(
      "SimpleThread executed using Thread");
    thread.start();
    thread.join();
}

@Test
public void givenAThread_whenSubmitToES_thenResult()
  throws Exception {
    
    executorService.submit(new SimpleThread(
      "SimpleThread executed using ExecutorService")).get();
}
```

This is a runnable way to make a thread.

```java
class SimpleRunnable implements Runnable {
	
    private String message;
	
    // standard logger, constructor
    
    @Override
    public void run() {
        log.info(message);
    }
}

@Test
public void givenRunnable_whenRunIt_thenResult()
 throws Exception {
    Thread thread = new Thread(new SimpleRunnable(
      "SimpleRunnable executed using Thread"));
    thread.start();
    thread.join();
}

@Test
public void givenARunnable_whenSubmitToES_thenResult()
 throws Exception {    
    executorService.submit(new SimpleRunnable(
      "SimpleRunnable executed using ExecutorService")).get();
}

@Test
public void givenARunnableLambda_whenSubmitToES_thenResult() 
  throws Exception {    
    executorService.submit(
      () -> log.info("Lambda runnable executed!"));
}
```

This is a TimerTask which creates background thread.

```java
TimerTask task = new TimerTask() {
    public void run() {
        System.out.println("Task performed on: " + new Date() + "n" 
          + "Thread's name: " + Thread.currentThread().getName());
    }
};
Timer timer = new Timer("Timer");
long delay = 1000L;
// This is an onetime task
timer.schedule(task, delay);

// This is a regular task
timer.scheduleAtFixedRate(repeatedTask, delay, period);
```

This is a ScheduledThreadPoolExecutor.

```java
ScheduledExecutorService executorService = Executors.newScheduledThreadPool(2);
// This is for one time tasks
ScheduledFuture<Object> resultFuture
  = executorService.schedule(callableTask, 1, TimeUnit.SECONDS);

// This is for regular tasks
ScheduledFuture<Object> resultFuture
  = executorService.scheduleAtFixedRate(runnableTask, 100, 450, TimeUnit.MILLISECONDS);
```

# Runnable vs Thread

* [Implementing a Runnable vs Extending a Thread](https://www.baeldung.com/java-runnable-vs-extending-thread)

-----

* When extending the Thread class, we're not overriding any of its methods. Instead, we override the method of Runnable (which Thread happens to implement). This is a clear violation of IS-A Thread principle
* Creating an implementation of Runnable and passing it to the Thread class utilizes composition and not inheritance – which is more flexible
* After extending the Thread class, we can't extend any other class
* From Java 8 onwards, Runnables can be represented as lambda expressions

# Status of Thread

* [Life Cycle of a Thread in Java @ baeldung](https://www.baeldung.com/java-thread-lifecycle)

----

There are 5 statuses of Thread in JAVA including `NEW, RUNNABLE, BLOCKED, WAITING, TIMED_WAITING, TERMINATED`

**NEW**

```java
Runnable runnable = new NewState();
Thread t = new Thread(runnable);
Log.info(t.getState());
```

**RUNNABLE**

```java
Runnable runnable = new NewState();
Thread t = new Thread(runnable);
t.start();
Log.info(t.getState());
```

**BLOCKED**

```java
public class BlockedState {
    public static void main(String[] args) throws InterruptedException {
        Thread t1 = new Thread(new DemoThreadB());
        Thread t2 = new Thread(new DemoThreadB());
        
        t1.start();
        t2.start();
        
        Thread.sleep(1000);
        
        Log.info(t2.getState());
        System.exit(0);
    }
}

class DemoThreadB implements Runnable {
    @Override
    public void run() {
        commonResource();
    }
    
    public static synchronized void commonResource() {
        while(true) {
            // Infinite loop to mimic heavy processing
            // 't1' won't leave this method
            // when 't2' try to enter this
        }
    }
}
```

**WAITING**

* object.wait()
* thread.join() 
* LockSupport.park()

```java
public class WaitingState implements Runnable {
    public static Thread t1;

    public static void main(String[] args) {
        t1 = new Thread(new WaitingState());
        t1.start();
    }

    public void run() {
        Thread t2 = new Thread(new DemoThreadWS());
        t2.start();

        try {
            t2.join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            Log.error("Thread interrupted", e);
        }
    }
}

class DemoThreadWS implements Runnable {
    public void run() {
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            Log.error("Thread interrupted", e);
        }
        
        Log.info(WaitingState.t1.getState());
    }
}
```

**TIMED_WAITING**

* thread.sleep(long millis)
* wait(int timeout) or wait(int timeout, int nanos)
* thread.join(long millis)
* LockSupport.parkNanos
* LockSupport.parkUntil

```java
public class TimedWaitingState {
    public static void main(String[] args) throws InterruptedException {
        DemoThread obj1 = new DemoThread();
        Thread t1 = new Thread(obj1);
        t1.start();
        
        // The following sleep will give enough time for ThreadScheduler
        // to start processing of thread t1
        Thread.sleep(1000);
        Log.info(t1.getState());
    }
}

class DemoThread implements Runnable {
    @Override
    public void run() {
        try {
            Thread.sleep(5000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            Log.error("Thread interrupted", e);
        }
    }
}
```

**TERMINATED**

```java
public class TerminatedState implements Runnable {
    public static void main(String[] args) throws InterruptedException {
        Thread t1 = new Thread(new TerminatedState());
        t1.start();
        // The following sleep method will give enough time for 
        // thread t1 to complete
        Thread.sleep(1000);
        Log.info(t1.getState());
    }
    
    @Override
    public void run() {
        // No processing in this block
    }
}
```


# Kill Thread

* [How to Kill a Java Thread @ baeldung](https://www.baeldung.com/java-thread-stop)

----

Thread.stop() method is deprecated.

This is a using flag.

```java
public class ControlSubThread implements Runnable {

    private Thread worker;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private int interval;

    public ControlSubThread(int sleepInterval) {
        interval = sleepInterval;
    }
 
    public void start() {
        worker = new Thread(this);
        worker.start();
    }
 
    public void stop() {
        running.set(false);
    }

    public void run() { 
        running.set(true);
        while (running.get()) {
            try { 
                Thread.sleep(interval); 
            } catch (InterruptedException e){ 
                Thread.currentThread().interrupt();
                System.out.println(
                  "Thread was interrupted, Failed to complete operation");
            }
            // do something here 
         } 
    } 
}
```

This is a interrupting a thread

```java
public class ControlSubThread implements Runnable {

    private Thread worker;
    private AtomicBoolean running = new AtomicBoolean(false);
    private int interval;

    // ...

    public void interrupt() {
        running.set(false);
        worker.interrupt();
    }

    boolean isRunning() {
        return running.get();
    }

    boolean isStopped() {
        return stopped.get();
    }

    public void run() {
        running.set(true);
        stopped.set(false);
        while (running.get()) {
            try {
                Thread.sleep(interval);
            } catch (InterruptedException e){
                Thread.currentThread().interrupt();
                System.out.println(
                  "Thread was interrupted, Failed to complete operation");
            }
            // do something
        }
        stopped.set(true);
    }
}
```

# Atomic Variables

* [An Introduction to Atomic Variables in Java @ baeldung](https://www.baeldung.com/java-atomic-variables)

----

This is a Counter class which is not thread-safe.

```java
public class Counter {
    int counter; 
 
    public void increment() {
        counter++;
    }
}
```

We can make a thread-safe Counter class using **syncronized**.

```java
public class SafeCounterWithLock {
    private volatile int counter;
 
    public synchronized void increment() {
        counter++;
    }
}
```

Automic variables are very convinient ways to implement thread-safe data. There are 4 kinds of Atomic variables including **AtomicInteger**, **AtomicLong**, **AtomicBoolean**, and **AtomicReference**. They keep the rule compare-and-swap (CAS).

* compareAndSet() – returns true when it succeeds, else false

```java
public class SafeCounterWithoutLock {
    private final AtomicInteger counter = new AtomicInteger(0);
    
    public int getValue() {
        return counter.get();
    }
    public void increment() {
        while(true) {
            int existingValue = getValue();
            int newValue = existingValue + 1;
            if(counter.compareAndSet(existingValue, newValue)) {
                return;
            }
        }
    }
}
```

# Syncronized

* [Guide to the Synchronized Keyword in Java](https://www.baeldung.com/java-synchronized)

-----

We can use the synchronized keyword in places like these:

* Instance methods
* Static methods
* Code blocks

```java
public synchronized void synchronisedCalculate() {
    setSum(getSum() + 1);
}
@Test
public void givenMultiThread_whenMethodSync() {
    ExecutorService service = Executors.newFixedThreadPool(3);
    SynchronizedMethods method = new SynchronizedMethods();

    IntStream.range(0, 1000)
      .forEach(count -> service.submit(method::synchronisedCalculate));
    service.awaitTermination(1000, TimeUnit.MILLISECONDS);

    assertEquals(1000, method.getSum());
}
```

```java
public static synchronized void syncStaticCalculate() {
    staticSum = staticSum + 1;
}
@Test
public void givenMultiThread_whenStaticSyncMethod() {
    ExecutorService service = Executors.newCachedThreadPool();

    IntStream.range(0, 1000)
      .forEach(count -> 
        service.submit(BaeldungSynchronizedMethods::syncStaticCalculate));
    service.awaitTermination(100, TimeUnit.MILLISECONDS);

    assertEquals(1000, BaeldungSynchronizedMethods.staticSum);
}
```

```java
public void performSynchronisedTask() {
    synchronized (this) {
        setCount(getCount()+1);
    }
}
@Test
public void givenMultiThread_whenBlockSync() {
    ExecutorService service = Executors.newFixedThreadPool(3);
    BaeldungSynchronizedBlocks synchronizedBlocks = new BaeldungSynchronizedBlocks();

    IntStream.range(0, 1000)
      .forEach(count -> 
        service.submit(synchronizedBlocks::performSynchronisedTask));
    service.awaitTermination(100, TimeUnit.MILLISECONDS);

    assertEquals(1000, synchronizedBlocks.getCount());
}
```

The lock behind the synchronized methods and blocks is reentrant.

```java
Object lock = new Object();
synchronized (lock) {
    System.out.println("First time acquiring it");

    synchronized (lock) {
        System.out.println("Entering again");

         synchronized (lock) {
             System.out.println("And again");
         }
    }
}
```

# wait(), notify(), notifyAll()

* [wait and notify() Methods in Java @ baeldung](https://www.baeldung.com/java-wait-notify)

----

* wait()
  * The wait() method causes the current thread to wait indefinitely until another thread either invokes notify() for this object or notifyAll().
* notify()
  * the method notify() notifies any one of them to wake up arbitrarily.
* notifyAll()
  * This method simply wakes all threads that are waiting on this object's monitor.

```java
public class Data {
    private String packet;
    
    // True if receiver should wait
    // False if sender should wait
    private boolean transfer = true;
 
    public synchronized void send(String packet) {
        while (!transfer) {
            try { 
                wait();
            } catch (InterruptedException e)  {
                Thread.currentThread().interrupt(); 
                Log.error("Thread interrupted", e); 
            }
        }
        transfer = false;
        
        this.packet = packet;
        notifyAll();
    }
 
    public synchronized String receive() {
        while (transfer) {
            try {
                wait();
            } catch (InterruptedException e)  {
                Thread.currentThread().interrupt(); 
                Log.error("Thread interrupted", e); 
            }
        }
        transfer = true;

        notifyAll();
        return packet;
    }
}
public class Sender implements Runnable {
    private Data data;
 
    // standard constructors
 
    public void run() {
        String packets[] = {
          "First packet",
          "Second packet",
          "Third packet",
          "Fourth packet",
          "End"
        };
 
        for (String packet : packets) {
            data.send(packet);

            // Thread.sleep() to mimic heavy server-side processing
            try {
                Thread.sleep(ThreadLocalRandom.current().nextInt(1000, 5000));
            } catch (InterruptedException e)  {
                Thread.currentThread().interrupt(); 
                Log.error("Thread interrupted", e); 
            }
        }
    }
}
public class Receiver implements Runnable {
    private Data load;
 
    // standard constructors
 
    public void run() {
        for(String receivedMessage = load.receive();
          !"End".equals(receivedMessage);
          receivedMessage = load.receive()) {
            
            System.out.println(receivedMessage);

            // ...
            try {
                Thread.sleep(ThreadLocalRandom.current().nextInt(1000, 5000));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt(); 
                Log.error("Thread interrupted", e); 
            }
        }
    }
}
public static void main(String[] args) {
    Data data = new Data();
    Thread sender = new Thread(new Sender(data));
    Thread receiver = new Thread(new Receiver(data));
    
    sender.start();
    receiver.start();
}
//
// First packet
// Second packet
// Third packet
// Fourth packet
```

# Lock

* [Guide to java.util.concurrent.Locks](https://www.baeldung.com/java-concurrent-locks)

----

Differences Between Lock and Synchronized Block

* A synchronized block is fully contained within a method – we can have Lock API's lock() and unlock() operation in separate methods
* Synchronized doesn't support fairness property. We can achieve fairness within the Lock APIs by specifying the fairness property. 
* A thread gets blocked if it can't get an access to the synchronized block. The Lock API provides tryLock() method. The thread acquires lock only if it's available and not held by any other thread.
* A thread which is in “waiting” state to acquire the access to synchronized block, can't be interrupted. The Lock API provides a method lockInterruptibly() which can be used to interrupt the thread when it's waiting for the lock

Lock's APIs

* `void lock()` – acquire the lock if it's available; if the lock isn't available a thread gets blocked until the lock is released
* `void lockInterruptibly()` – this is similar to the lock(), but it allows the blocked thread to be interrupted and resume the execution through a thrown java.lang.InterruptedException
* `boolean tryLock()` – this is a non-blocking version of lock() method; it attempts to acquire the lock immediately, return true if locking succeeds
* `boolean tryLock(long timeout, TimeUnit timeUnit)` – this is similar to tryLock(), except it waits up the given timeout before giving up trying to acquire the Lock
* `void unlock()` – unlocks the Lock instance
* `Lock readLock()` – returns the lock that's used for reading
* `Lock writeLock()` – returns the lock that's used for writing

```java
Lock lock = ...; 
lock.lock();
try {
    // access to the shared resource
} finally {
    lock.unlock();
}
```

There 3 implementations including **ReentrantLock**, **ReentrantReadWriteLock**, **StampedLock**.

**ReentrantLock**

```java
public class SharedObject {
    //...
    ReentrantLock lock = new ReentrantLock();
    int counter = 0;

    public void perform() {
        lock.lock();
        try {
            // Critical section here
            count++;
        } finally {
            lock.unlock();
        }
    }
    //...
}

public void performTryLock(){
    //...
    boolean isLockAcquired = lock.tryLock(1, TimeUnit.SECONDS);
    
    if(isLockAcquired) {
        try {
            //Critical section here
        } finally {
            lock.unlock();
        }
    }
    //...
}
```

**ReentrantReadWriteLock**

```java
public class SynchronizedHashMapWithReadWriteLock {

    Map<String,String> syncHashMap = new HashMap<>();
    ReadWriteLock lock = new ReentrantReadWriteLock();
    // ...
    Lock writeLock = lock.writeLock();

    public void put(String key, String value) {
        try {
            writeLock.lock();
            syncHashMap.put(key, value);
        } finally {
            writeLock.unlock();
        }
    }
    ...
    public String remove(String key){
        try {
            writeLock.lock();
            return syncHashMap.remove(key);
        } finally {
            writeLock.unlock();
        }
    }
    //...
}

Lock readLock = lock.readLock();
//...
public String get(String key){
    try {
        readLock.lock();
        return syncHashMap.get(key);
    } finally {
        readLock.unlock();
    }
}

public boolean containsKey(String key) {
    try {
        readLock.lock();
        return syncHashMap.containsKey(key);
    } finally {
        readLock.unlock();
    }
}
```

**StampedLock**

lock acquisition methods return a stamp that is used to release a lock or to check if the lock is still valid:

```java
public class StampedLockDemo {
    Map<String,String> map = new HashMap<>();
    private StampedLock lock = new StampedLock();

    public void put(String key, String value){
        long stamp = lock.writeLock();
        try {
            map.put(key, value);
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    public String get(String key) throws InterruptedException {
        long stamp = lock.readLock();
        try {
            return map.get(key);
        } finally {
            lock.unlockRead(stamp);
        }
    }
}
```

Another feature provided by StampedLock is optimistic locking. Most of the time read operations don't need to wait for write operation completion and as a result of this, the full-fledged read lock isn't required.

```java
public String readWithOptimisticLock(String key) {
    long stamp = lock.tryOptimisticRead();
    String value = map.get(key);

    if(!lock.validate(stamp)) {
        stamp = lock.readLock();
        try {
            return map.get(key);
        } finally {
            lock.unlock(stamp);               
        }
    }
    return value;
}
```

# Condition

* [Guide to java.util.concurrent.Locks](https://www.baeldung.com/java-concurrent-locks)

----

Traditionally Java provides `wait()`, `notify()` and `notifyAll()` methods for thread intercommunication. Conditions have similar mechanisms, but in addition, we can specify multiple conditions.

```java
public class ReentrantLockWithCondition {

    Stack<String> stack = new Stack<>();
    int CAPACITY = 5;

    ReentrantLock lock = new ReentrantLock();
    Condition stackEmptyCondition = lock.newCondition();
    Condition stackFullCondition = lock.newCondition();

    public void pushToStack(String item){
        try {
            lock.lock();
            while(stack.size() == CAPACITY) {
                stackFullCondition.await();
            }
            stack.push(item);
            stackEmptyCondition.signalAll();
        } finally {
            lock.unlock();
        }
    }

    public String popFromStack() {
        try {
            lock.lock();
            while(stack.size() == 0) {
                stackEmptyCondition.await();
            }
            return stack.pop();
        } finally {
            stackFullCondition.signalAll();
            lock.unlock();
        }
    }
}
```


# CountDownLatch

* [Guide to CountDownLatch in Java](https://www.baeldung.com/java-countdown-latch)

----

When it's  been counted down to zero, it will wake-up blocked threads.

```java
public class Worker implements Runnable {
    private List<String> outputScraper;
    private CountDownLatch countDownLatch;

    public Worker(List<String> outputScraper, CountDownLatch countDownLatch) {
        this.outputScraper = outputScraper;
        this.countDownLatch = countDownLatch;
    }

    @Override
    public void run() {
        doSomeWork();
        outputScraper.add("Counted down");
        countDownLatch.countDown();
    }
}

@Test
public void whenParallelProcessing_thenMainThreadWillBlockUntilCompletion()
  throws InterruptedException {

    List<String> outputScraper = Collections.synchronizedList(new ArrayList<>());
    CountDownLatch countDownLatch = new CountDownLatch(5);
    List<Thread> workers = Stream
      .generate(() -> new Thread(new Worker(outputScraper, countDownLatch)))
      .limit(5)
      .collect(toList());

      workers.forEach(Thread::start);
      countDownLatch.await(); 
      outputScraper.add("Latch released");

      assertThat(outputScraper)
        .containsExactly(
          "Counted down",
          "Counted down",
          "Counted down",
          "Counted down",
          "Counted down",
          "Latch released"
        );
    }
```

# Semaphore

* [Semaphores in Java](https://www.baeldung.com/java-semaphore)

------

* `tryAcquire()` – return true if a permit is available immediately and acquire it otherwise return false, but acquire() acquires a permit and blocking until one is available
* `release()` – release a permit
* `availablePermits()` – return number of current permits available

```java
class LoginQueueUsingSemaphore {

    private Semaphore semaphore;

    public LoginQueueUsingSemaphore(int slotLimit) {
        semaphore = new Semaphore(slotLimit);
    }

    boolean tryLogin() {
        return semaphore.tryAcquire();
    }

    void logout() {
        semaphore.release();
    }

    int availableSlots() {
        return semaphore.availablePermits();
    }

}

@Test
public void givenLoginQueue_whenReachLimit_thenBlocked() {
    int slots = 10;
    ExecutorService executorService = Executors.newFixedThreadPool(slots);
    LoginQueueUsingSemaphore loginQueue = new LoginQueueUsingSemaphore(slots);
    IntStream
      .range(0, slots)
      .forEach(user -> executorService.execute(loginQueue::tryLogin));
    executorService.shutdown();

    assertEquals(0, loginQueue.availableSlots());
    assertFalse(loginQueue.tryLogin());
}

@Test
public void givenLoginQueue_whenLogout_thenSlotsAvailable() {
    int slots = 10;
    ExecutorService executorService = Executors.newFixedThreadPool(slots);
    LoginQueueUsingSemaphore loginQueue = new LoginQueueUsingSemaphore(slots);
    IntStream
      .range(0, slots)
      .forEach(user -> executorService.execute(loginQueue::tryLogin));
    executorService.shutdown();
    assertEquals(0, loginQueue.availableSlots());
    loginQueue.logout();

    assertTrue(loginQueue.availableSlots() > 0);
    assertTrue(loginQueue.tryLogin());
}
```
