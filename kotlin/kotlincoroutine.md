- [Abstract](#abstract)
- [Basic](#basic)
  - [First coroutine](#first-coroutine)
  - [Structured concurrency](#structured-concurrency)

----

# Abstract

kotlin 의 coroutine 에 대해 정리한다.

# Basic

## First coroutine

```kotlin
fun main() = runBlocking { // this: CoroutineScope
    launch { // launch a new coroutine and continue
        delay(1000L) // non-blocking delay for 1 second (default time unit is ms)
        println("World!") // print after delay
    }
    println("Hello") // main coroutine continues while a previous one is delayed
}
```

* `launch` 는 coroutine builder 이다. block 을 입력으로 받아서 coroutine 을 생성한다.
* `delay` 는 suspend function 이다.
* `runBlocking` 는 coroutine builder 이다. block 을 입력으로 받아서 coroutine 을 생성한다. main thread 는 그 coroutine 이 모두 실행될 때까지 blocking 된다.

## Structured concurrency

kotlin 의 coroutine 은 structured concurrency 를 따른다. 부모 coroutine 은 특정 CoroutineScope 에서 자식 coroutine 을 만들 수 있다. 그리고 자식 coroutine 들이 모두 완료되야 부모 coroutine 이 흐름을 완료할 수 있다.

예를 들어 다음과 같은 code 에서 `delay(1000L)` 의 실행이 완료되야 `println("World!")` 가 수행된다.

```kotlin
fun main() = runBlocking { // this: CoroutineScope
    launch { // launch a new coroutine and continue
        delay(1000L) // non-blocking delay for 1 second (default time unit is ms)
        println("World!") // print after delay
    }
    println("Hello") // main coroutine continues while a previous one is delayed
}
```
