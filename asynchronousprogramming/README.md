# Abstract

[Asynchronous programming techniques](https://kotlinlang.org/docs/async-programming.html) 를 읽고 Asynchronous programming 에 정리한다. 

Asynchronous 는 A, B 두개의 job 이 있을 때 서로 신경쓰지 않고 각자 실행하는 것을 의미한다. [Asynchronous VS non-blocking](#asynchronous-vs-non-blocking) 를 참고한다.

오랫동안 개발자들은 blocking 없이 logic 을 수행하기 위해 다음과 같은 방법들을 생각해 냈다.

* Threading
* Callbacks
* Futures, promises, and others
* Reactive Extensions
* Coroutines

예를 들어 다음과 같은 code 를 살펴보자. `preparePost` 는 blocking 되는 함수이다. main thread 를 blocking 할 수 있다. 앞서 언급한 다양한 방법들로 blocking 문제를 해결해 보자.

```kotlin
fun postItem(item: Item) {
    val token = preparePost()
    val post = submitPost(token, item)
    processPost(post)
}

fun preparePost(): Token {
    // makes a request and consequently blocks the main thread
    return token
}
```

# Threading

`preparePost` 를 다른 thread 에서 실행한다. 그러나 다음과 같은 결점들이 존재한다.

* thread 는 context switching 의 비용이 크다.
* thread 의 개수는 정해져 있다.
* thread 가 항상 지원되는 것은 아니다. JavaScript 의 경우 thread 는 지원되지 않는다.
* thread 는 debugging 이 어렵다.

# Callbacks

다음가 같은 callback 을 이용한다.

```kotlin
fun postItem(item: Item) {
    preparePostAsync { token ->
        submitPostAsync(token, item) { post ->
            processPost(post)
        }
    }
}

fun preparePostAsync(callback: (Token) -> Unit) {
    // make request and return immediately
    // arrange callback to be invoked later
}
```

그러나 다음과 같은 결점이 존재한다.

* nested callback 들 때문에 readability 가 떨어진다.
* error handling 이 복잡하다.

# Futures, promises, and others

다음과 같이 return 된 Future, Promise 를 이용한다. language 에 따라 이름은 다를 수 있다.

```kotlin
fun postItem(item: Item) {
    preparePostAsync()
        .thenCompose { token ->
            submitPostAsync(token, item)
        }
        .thenAccept { post ->
            processPost(post)
        }

}

fun preparePostAsync(): Promise<Token> {
    // makes request and returns a promise that is completed later
    return promise
}
```

그러나 다음과 같은 결점들이 존재한다.

* 우리가 익숙한 top-down programming 과 다르다.
* `thenCompose, thenAccept` 와 같은 새로운 APIs 를 익혀야 한다.
* return type 은 Futures, Promise 이다. 우리가 원하는 값을 얻기 위해 한번 더 읽어야 한다.
* error handling 이 복잡하다.

# Reactive Extensions

Reactive Extensions (Rx) 는 Erik Meijer 에 의해 c# 에 처음 소개되었다. 이후 Netflix 가 이것을 Java 로 porting 하면서 대중화 되었다. 이것을 RxJava 라고 한다. 이후 수많은 언어들이 Rx 를 도입했다. [reactive programming](/reactiveprogramming/README.md)

# Coroutines

새로운 API 를 배울 필요가 없다. synchronous code 를 작성하는 것과 같다. error handling 이 수월하다. kotlin 에서 asynchronous programming 을 할 수 있는 가장 세련된 방법이다. [coroutine](/coroutine/README.md)

```kotlin
fun postItem(item: Item) {
    launch {
        val token = preparePost()
        val post = submitPost(token, item)
        processPost(post)
    }
}

suspend fun preparePost(): Token {
    // makes a request and suspends the coroutine
    return suspendCoroutine { /* ... */ }
}
```

`preparePost` 는 suspend function 이다. 실행의 흐름을 pause, resume 할 수 있다.
