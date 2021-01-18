# Materials

* [Multithread design pattern @ slideshare](https://www.slideshare.net/ohyecloudy/multithread-design-pattern)
* [Guide to java.util.concurrent.Locks @ baeldung](https://www.baeldung.com/java-concurrent-locks)

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

# Condition
