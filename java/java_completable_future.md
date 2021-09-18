- [Abstract](#abstract)
- [Materials](#materials)
- [Basics](#basics)
  - [Overview](#overview)
  - [Simple Future](#simple-future)
  - [runAsync, supplyAsync](#runasync-supplyasync)
  - [thenApply](#thenapply)
  - [thenAccept](#thenaccept)
  - [thenRun](#thenrun)

----

# Abstract

Completable Future 에 대해 정리한다.

# Materials

* [Guide To CompletableFuture @ baeldung](https://www.baeldung.com/java-completablefuture)

# Basics

## Overview

**CompletableFuture** 는 다음과 Future, CompletionStage 를 구현한 class 이다. 
미래에 완성될 수 있는 작업을 표현한 것이다.

```java
// 
public class CompletableFuture<T> implements Future<T>, CompletionStage<T> {
    ...
}
```

**Future** 는 다음과 같은 interface 이다. 비동기 작업의 결과를 표현한 것이다. 지금 수행하고 있는
작업이 취소됬는지 잘 수행됬는지 등을 알 수 있다. 또한 취소시킬수도 있다. 그리고 결과를 기다릴 수도 있다.

```java
// A Future represents the result of an asynchronous computation.
public interface Future<V> {
    boolean cancel(boolean mayInterruptIfRunning);
    boolean isCancelled();
    boolean isDone();
    V get() throws InterruptedException, ExecutionException;
    V get(long timeout, TimeUnit unit)
        throws InterruptedException, ExecutionException, TimeoutException;
}
```

**CompletionStage** 는 다음과 같은 interface 이다. 비동기 작업의 한 단계를 표현한 것이다. 지금 수행하고 있는
작업에 특별한 일을 시킬 수 있다.

```java
// A stage of a possibly asynchronous computation.
public interface CompletionStage<T> {
    public <U> CompletionStage<U> thenApply(Function<? super T,? extends U> fn);
    public <U> CompletionStage<U> thenApplyAsync(Function<? super T,? extends U> fn);

    public CompletionStage<Void> thenAccept(Consumer<? super T> action);
    public CompletionStage<Void> thenAcceptAsync(Consumer<? super T> action);
    public CompletionStage<Void> thenAcceptAsync(Consumer<? super T> action,
                                                 Executor executor);

    public CompletionStage<Void> thenRun(Runnable action);
    public CompletionStage<Void> thenRunAsync(Runnable action);
    public CompletionStage<Void> thenRunAsync(Runnable action,
                                              Executor executor);

    public <U,V> CompletionStage<V> thenCombine(CompletionStage<? extends U> other,
         BiFunction<? super T,? super U,? extends V> fn);
    public <U,V> CompletionStage<V> thenCombineAsync(CompletionStage<? extends U> other,
         BiFunction<? super T,? super U,? extends V> fn);
    public <U,V> CompletionStage<V> thenCombineAsync(CompletionStage<? extends U> other,
         BiFunction<? super T,? super U,? extends V> fn,
         Executor executor);
    
    public <U> CompletionStage<Void> thenAcceptBoth
        (CompletionStage<? extends U> other,
         BiConsumer<? super T, ? super U> action);
    public <U> CompletionStage<Void> thenAcceptBothAsync
        (CompletionStage<? extends U> other,
         BiConsumer<? super T, ? super U> action);
    public <U> CompletionStage<Void> thenAcceptBothAsync
        (CompletionStage<? extends U> other,
         BiConsumer<? super T, ? super U> action,
         Executor executor);

    public CompletionStage<Void> runAfterBoth(CompletionStage<?> other,
                                              Runnable action);
    public CompletionStage<Void> runAfterBothAsync(CompletionStage<?> other,
                                                   Runnable action);
    public CompletionStage<Void> runAfterBothAsync(CompletionStage<?> other,
                                                   Runnable action,
                                                   Executor executor);
    
    public <U> CompletionStage<U> applyToEither
        (CompletionStage<? extends T> other,
         Function<? super T, U> fn);
    public <U> CompletionStage<U> applyToEitherAsync
        (CompletionStage<? extends T> other,
         Function<? super T, U> fn);
    public <U> CompletionStage<U> applyToEitherAsync
        (CompletionStage<? extends T> other,
         Function<? super T, U> fn,
         Executor executor);

    public CompletionStage<Void> acceptEither
        (CompletionStage<? extends T> other,
         Consumer<? super T> action);
    public CompletionStage<Void> acceptEitherAsync
        (CompletionStage<? extends T> other,
         Consumer<? super T> action);
    public CompletionStage<Void> acceptEitherAsync
        (CompletionStage<? extends T> other,
         Consumer<? super T> action,
         Executor executor);

    public CompletionStage<Void> runAfterEither(CompletionStage<?> other,
                                                Runnable action);
    public CompletionStage<Void> runAfterEitherAsync
        (CompletionStage<?> other,
         Runnable action);
    public CompletionStage<Void> runAfterEitherAsync
        (CompletionStage<?> other,
         Runnable action,
         Executor executor);

    public <U> CompletionStage<U> thenCompose
        (Function<? super T, ? extends CompletionStage<U>> fn);
    public <U> CompletionStage<U> thenComposeAsync
        (Function<? super T, ? extends CompletionStage<U>> fn);
    public <U> CompletionStage<U> thenComposeAsync
        (Function<? super T, ? extends CompletionStage<U>> fn,
         Executor executor);

    public <U> CompletionStage<U> handle
        (BiFunction<? super T, Throwable, ? extends U> fn);
    public <U> CompletionStage<U> handleAsync
        (BiFunction<? super T, Throwable, ? extends U> fn);
    public <U> CompletionStage<U> handleAsync
        (BiFunction<? super T, Throwable, ? extends U> fn,
         Executor executor);

    public CompletionStage<T> whenComplete
        (BiConsumer<? super T, ? super Throwable> action);
    public CompletionStage<T> whenCompleteAsync
        (BiConsumer<? super T, ? super Throwable> action);
    public CompletionStage<T> whenCompleteAsync
        (BiConsumer<? super T, ? super Throwable> action,
         Executor executor);

    public CompletionStage<T> exceptionally
        (Function<Throwable, ? extends T> fn);
    public CompletableFuture<T> toCompletableFuture();
}
```

## Simple Future

CompletableFuture 를 단순한 Future 로 사용해보자. 예를 들어 다음과 같이 calculateAsync 를
다음과 같이 정의한다. `Thread.sleep(500)` 를 호출하고 있기 때문에 blocking 된다.

```java
public Future<String> calculateAsync() throws InterruptedException {
    CompletableFuture<String> completableFuture = new CompletableFuture<>();

    Executors.newCachedThreadPool().submit(() -> {
        Thread.sleep(500);
        completableFuture.complete("Hello");
        return null;
    });

    return completableFuture;
}
```

이제 calculateAsync 를 다음과 같이 사용해 보자. `completableFuture.get()` 에서
blocking 된다.

```java
Future<String> completableFuture = calculateAsync();
String result = completableFuture.get();
assertEquals("Hello", result);
```

만약 blocking 되지 않게 하고 싶다면 `CompletableFuture.completedFuture`
를 사용한다. 미리 결과를 알고 있어야 사용할 수 있다.

```java
Future<String> completableFuture = 
  CompletableFuture.completedFuture("Hello");
String result = completableFuture.get();
assertEquals("Hello", result);
```

## runAsync, supplyAsync

`calculateAsync` 보다 더욱 단순한 방법으로 **CompletableFuture** 를
만들어 보자. **runAsync**, **supplyAsync** 를 lambda argument 와 함께 사용하여
생성해 보자. **runAsync** 에 전달하는 lambda argument 는 **Runnable** 이라고 한다.
**supplyAsync** 에 전달하는 lambda argument 는 **Supplier** 라고 한다. **Runnable**,
**Supplier** 는 모두 **Functional Interface** 이다.

```java
CompletableFuture<String> future
  = CompletableFuture.supplyAsync(() -> "Hello");
assertEquals("Hello", future.get());
```

## thenApply

CompletableFuture 를 생성하고 **thenApply** 에 lambda argument
를 전달하여 새로운 비동기 작업을 시켜보자. 이때 전달하는 lambda argument 는
argument 도 있고 return 도 있다. lambda 는 Future chain 에 삽입된다.

```java
CompletableFuture<String> completableFuture
  = CompletableFuture.supplyAsync(() -> "Hello");
CompletableFuture<String> future = completableFuture
  .thenApply(s -> s + " World");
assertEquals("Hello World", future.get());
```

## thenAccept

CompletableFuture 를 생성하고 **thenAccept** 에 lambda argument
를 전달하여 새로운 비동기 작업을 시켜보자. 이때 전달하는 lambda argument 는
argument 는 있는데 return 은 없다. 이러한 lambda 를 **Consumer** functional
interface 라고 한다.

```java
CompletableFuture<String> completableFuture
  = CompletableFuture.supplyAsync(() -> "Hello");
CompletableFuture<Void> future = completableFuture
  .thenAccept(s -> System.out.println("Computation returned: " + s));
future.get();
```

## thenRun

CompletableFuture 를 생성하고 **thenAccept** 에 lambda argument
를 전달하여 새로운 비동기 작업을 시켜보자. 이때 전달하는 lambda argument 는
argument 도 없고 return 도 없다. 이러한 lambda 를 **Runnable** functional
interface 라고 한다.

```java
CompletableFuture<String> completableFuture 
  = CompletableFuture.supplyAsync(() -> "Hello");
CompletableFuture<Void> future = completableFuture
  .thenRun(() -> System.out.println("Computation finished."));
future.get();
```
