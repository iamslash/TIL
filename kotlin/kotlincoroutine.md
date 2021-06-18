- [Abstract](#abstract)
- [Basic](#basic)
  - [First coroutine](#first-coroutine)
  - [Structured concurrency](#structured-concurrency)
  - [Coroutine Builders](#coroutine-builders)

----

# Abstract

kotlin 의 coroutine 에 대해 정리한다.

# Basic

## First coroutine

다음과 같이 간단한 예를 살펴보자. 다음과 같은 특징들을 생각할 수 있다.

* `runBlocking` 는 coroutine builder 이다. code block 을 입력으로 받아서 coroutine 을 생성한다. main thread 는 그 coroutine 이 모두 실행될 때까지 blocking 된다.
* `launch` 역시 coroutine builder 이다. code block 을 입력으로 받아서 coroutine 을 생성한다.
* `delay` 는 suspend function 이다. 실행의 흐름이 suspend/resume 될 수 있다.


```kotlin
fun main() = runBlocking { // this: CoroutineScope
    launch { // launch a new coroutine and continue
        delay(1000L) // non-blocking delay for 1 second (default time unit is ms)
        println("World!") // print after delay
    }
    println("Hello") // main coroutine continues while a previous one is delayed
}
// Hello
// World!
```

이번에는 `doWorld` 라는 suspend function 으로 refactoring 해보자. 다음과 같은 특징들을 생각할 수 있다.

* `doWorld` 는 suspend function 이다. coroutine 에서 suspend function 은 suspend/resume 되어 실행의 흐름이 완료될 때까지 return 되지 않는다.

```kotlin
import kotlinx.coroutines.*

fun main() = runBlocking { // this: CoroutineScope
	doWorld()
//  launch { doWorld() }
    println("Hello")
}

// this is your first suspending function
suspend fun doWorld() {
    delay(1000L)
    println("World!")
}
// World!
// Hello
```

만약 다음과 같이 `launch` 에서 `doWorld` 를 실행하면 위의 code 와 출력 순서가 달라진다. `launch` 에 의해 자식 coroutine 을 생성하고 실행의 흐름을 이어간다. `doWorld()` 는 자식 coroutine 에서 실행된다.

```kotlin
import kotlinx.coroutines.*

fun main() = runBlocking { // this: CoroutineScope
//	doWorld()
    launch { doWorld() }
    println("Hello")
}

// this is your first suspending function
suspend fun doWorld() {
    delay(1000L)
    println("World!")
}
// Hello
// World!
```

## Structured concurrency

kotlin 의 coroutine 은 structured concurrency 를 따른다. 부모 coroutine 은 특정 CoroutineScope 에서 자식 coroutine 을 만들 수 있다. 그리고 자식 coroutine 들이 모두 완료되야 부모 coroutine 이 흐름을 완료할 수 있다.

다음과 같은 예를 살펴보자. `runBlocking` 은 code block 을 입력으로 받아서 coroutine 을 생성한다. `doWorld()` 는 suspend function 이다. 실행의 흐름이 suspend/resume 될 수 있다. 

`coroutineScope` 을 이용하여 code block 을 입력으로 받아 coroutine 을 생성한다. `coroutineScope` 은 `runBlocking` 처럼 생성한 coroutine 이 모두 완료될 때까지 기다린다. 그러나 `runBlocking` 처럼 main thread 가 block 되지는 않는다. 단지 suspend 된다. 

`launch` 로 두개의 자식 coroutine 을 생성한다. 두개의 자식 coroutine 이 모두 완료되기 전까지 `doWorld` 는 return 되지 않는다.

```kotlin
import kotlinx.coroutines.*

// Sequentially executes doWorld followed by "Done"
fun main() = runBlocking {
    doWorld()
    println("Done")
}

// Concurrently executes both sections
suspend fun doWorld() = coroutineScope { // this: CoroutineScope
    launch {
        delay(2000L)
        println("World 2")
    }
    launch {
        delay(1000L)
        println("World 1")
    }
    println("Hello")
}
```

## Coroutine Builders


