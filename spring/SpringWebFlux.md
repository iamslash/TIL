- [Abstract](#abstract)
- [Materials](#materials)
- [Basics](#basics)
  - [Project Reactor](#project-reactor)
  - [Example Application](#example-application)
- [Advanced](#advanced)
  - [Reactor MeltDown](#reactor-meltdown)
  - [Spring MVC vs Spring WebFlux](#spring-mvc-vs-spring-webflux)
  - [Why WebFlux Slow](#why-webflux-slow)
  - [Reactive Cache](#reactive-cache)
- [Question](#question)
  - [map vs flatMap for async function](#map-vs-flatmap-for-async-function)

----

# Abstract

Spring WebFlux 에 대해 정리한다.

# Materials

* [Webflux로 막힘없는 프로젝트 만들기 @ ifkakao2021](https://if.kakao.com/session/107)
  * event-loop 를 당당하는 single thread 가 blocking 되면 Reactor meltdown 을 일으킨다. 그것을 해결하는 방법을 제시한다. 
  * [pdf](https://t1.kakaocdn.net/service_if_kakao_prod/file/file-1636524397590)
  * [src](https://github.com/beryh/event-loop-demo)
* [내가 만든 WebFlux가 느렸던 이유 @ nhn](https://forward.nhn.com/session/26)
  * [pdf](https://rlxuc0ppd.toastcdn.net/presentation/%5BNHN%20FORWARD%202020%5D%EB%82%B4%EA%B0%80%20%EB%A7%8C%EB%93%A0%20WebFlux%EA%B0%80%20%EB%8A%90%EB%A0%B8%EB%8D%98%20%EC%9D%B4%EC%9C%A0.pdf)
* [Spring WebFlux @ 권남](https://kwonnam.pe.kr/wiki/springframework/webflux)
* [스프링 부트 실전 활용 마스터 @ yes24](http://www.yes24.com/Product/Goods/101803558)
  * [src](https://github.com/onlybooks/spring-boot-reactive)
* [Guide to Spring 5 WebFlux @ baeldung](https://www.baeldung.com/spring-webflux)
* [Web on Reactive Stack @ spring.io](https://docs.spring.io/spring-framework/docs/5.2.6.RELEASE/spring-framework-reference/web-reactive.html#webflux)
  * [kor](https://godekdls.github.io/Reactive%20Spring/contents/)

# Basics

## Project Reactor

Spring WebFlux 는 내부적으로 [reactor @ TIL](/reactor/README.md) 를 사용하여 구현되었다. [reactor @ TIL](/reactor/README.md) 를 이해하는 것이 중요하다.

## Example Application

[wfmongo](https://github.com/iamslash/spring-examples/tree/master/wfmongo) 는 WebFlux, MongoDB 를 구현한 예이다.

# Advanced

## Reactor MeltDown

* [Webflux로 막힘없는 프로젝트 만들기 @ ifkakao2021](https://if.kakao.com/session/107)
  * event-loop 를 당당하는 single thread 가 blocking 되면 Reactor meltdown 을 일으킨다. 그것을 해결하는 방법을 제시한다. 
  * [pdf](https://t1.kakaocdn.net/service_if_kakao_prod/file/file-1636524397590)
  * [src](https://github.com/beryh/event-loop-demo)
  
----

Netty 의 Event-Loop 는 Single Thread 으로 동작한다. Single Thread 가 blocking 되면 Reactor MeltDown 을 일으킨다. 즉, Event-Loop 가 멈춘다. 따라서 blocking call 은 모두 제거해야 한다.

[Block Hound](https://github.com/reactor/BlockHound) 를 이용하여 run-time 에 blocking call 을 발견하자. [Block Hound](https://github.com/reactor/BlockHound) 는 별도의 agent 에서 동작한다. byte code 를 조작한다. 따라서 production profile 에서 사용하지 말자. test profile 에서만 사용하자. [Block Hound](https://github.com/reactor/BlockHound) 는 다음과 같이 간단히 실행할 수 있다.

```java
// 
static {
  BlockHound.install()
}

// 
@Execution(ExecutionMode.CONCURRENT)
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.DEFINED_PORT)
class EventLoopDemoTest {

    @BeforeAll
    public static void beforeAll() {
        BlockHound.install();
        System.setProperty("reactor.netty.ioWorkerCount", "4");
    }
...
}
```

다음은 blocking 을 일으키는 code 이다. Reactor Meltdown 이 발생하여 `health()` 가 hang 되는 것을 확인할 수 있다.

```java
@RestController
public class EventLoopDemoController {
    @GetMapping(value = "/sleep", produces = MediaType.TEXT_PLAIN_VALUE)
    public Mono<String> sleep() {
        return Mono.fromSupplier(() -> blockingFunction(10_000L));
    }

    private String blockingFunction(long sleepMs) {
        try {
            Thread.sleep(sleepMs);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        return "OK";
    }

    @GetMapping(value = "/ok", produces = MediaType.TEXT_PLAIN_VALUE)
    public String health() {
        return "OK";
    }
}
```

blocking 을 일으키는 code 는 `subscribeOn()` 을 사용하여 별도의 thread-pool 즉, Scheduler Work 에서 실행되도록 하자.

```java
@RestController
public class EventLoopDemoController {
    @GetMapping(value = "/sleep", produces = MediaType.TEXT_PLAIN_VALUE)
    public Mono<String> sleep() {
        return Mono.fromSupplier(() -> blockingFunction(10_000L))
                .subscribeOn(Schedulers.boundedElastic());
    }

    private String blockingFunction(long sleepMs) {
        try {
            Thread.sleep(sleepMs);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        return "OK";
    }

    @GetMapping(value = "/ok", produces = MediaType.TEXT_PLAIN_VALUE)
    public String health() {
        return "OK";
    }
}
```

`Scheduler Worker` 의 종류는 다음과 같다. comment 와 같이 blocking call 은 `Schedulers.boundedElastic()` 에서 실행하는 것이 좋다.

```java
Schedulers.single();          // for low-latency
Schedulers.parellel();        // for fast & non-blocking
Schedulers.immediate();       // immediately run instead of scheduling 
Schedulers.boundedElastic();  // for long, an alternative for blocking tasks
```

다음의 code 를 보자. `method1(), method2()` 는 synchronous method 이다. `method3()` 는 asynchronous method 이다. synchronous method 는 호출하는 쪽에서 blocking 되지 않도록 해야 한다. 즉, `subscribeOn(Schedulers.boundedElastic())` 을 사용하여 별도의 thread 에서 실행해야 한다. `method3()` 는 `method3()` 를 작성하는 쪽에서 blocking 처리를 해야 한다. 

```java
Result method1() throws InterruptedException;
Result method2();
Mono<Result> method3();
```

`method1()` 는 `InterruptedException` 를 던지고 있다. method signature 만으로 blocking call 이 있는지 구분할 수 있다. `method2()` 는 synchronous method 이므로 blocking call 이 있을만 하다. code 를 확인하기 전까지는 알 수 없다. `method3()` 는 asynchronous method 이다. 그러나 `method3()` 작성자가 blocking 되지 않게 작성했다고 확신할 수 없다.

code 를 모두 확인하지 않고 blocking call 을 알아낼 수는 없을까? [Block Hound](https://github.com/reactor/BlockHound) 를 이용하여 blocking code 를 발견할 수 있다.

## Spring MVC vs Spring WebFlux

* [SpringMVC vs WebFlux @ velog](https://velog.io/@minsuk/SpringMVC-vs-WebFlux)

Spring MVC 는 하나의 request 를 하나의 thread 가 처리한다. 별도의 설정이 없다면 Spring MVC 는 200 개의 thread 를 thread pool 에서 생성한다. 200 개의 thread 가 4 개 혹은 8 개의 CPU core 를 두고 경합하는 것은 많은 context switching 을 발생시킨다. 비효율적이다.

Spring WebFlux 는 하나의 thread 가 event-loop 을 처리한다. 그리고 몇개의 worker-thread 가 있다. thread 의 개수가 적기 때문에 context switching overhead 가 적다. 또한 4 개 혹은 8 개의 CPU core 를 두고 경합하는 thread 의 수도 적다. 효율적이다.

![](https://media.vlpt.us/images/minsuk/post/b62e9387-1c38-42c0-9f23-2e5fb900e1a3/%EC%BA%A1%EC%B2%98.PNG)

## Why WebFlux Slow

* [내가 만든 WebFlux가 느렸던 이유 @ nhn](https://forward.nhn.com/session/26)
  * [pdf](https://rlxuc0ppd.toastcdn.net/presentation/%5BNHN%20FORWARD%202020%5D%EB%82%B4%EA%B0%80%20%EB%A7%8C%EB%93%A0%20WebFlux%EA%B0%80%20%EB%8A%90%EB%A0%B8%EB%8D%98%20%EC%9D%B4%EC%9C%A0.pdf)

------

```java
public class AdHandler {
    public Mono<ServerResponse> fetchByAdRequest(ServerRequest serverRequest) {
        return serverRequest.bodyToMono(AdRequest.class)
            .log()
            .map(AdRequest::getCode)
            .map(AdCodeId::of)
            .map(adCodeId -> {
                    log.warn("Requested AdCodeId = {}", adCodeId.toKeyString());
                    return adCodeId;
                })
            .map(adCodeId -> cacheStorageAdapter.getAdValue(adCodeId))
            .flatMap(adValue ->
                     ServerResponse.ok().contentType(MediaType.APPLICATION_JSON)
                     .body(adValue, adValue.class)
                     );
    }
}
```

위의 코드는 다음과 같은 이유로 performance 가 좋지 않다.

* `log()` 는 blocking I/O 를 일으킨다. 성능이 좋지 않다.
* `map()` 호출이 너무 많다. immutable object instance 생성이 많다. GC 의 연산량이 증가한다.
* `Mono::map()` 은 동기식으로 동작한다. `Mono::flatMap()` 은 비동기식으로 동작한다. `cacheStorageAdapter.getAdValue(adCodeId)` 은 비동기식 method 이다. `Mono::flatMap()` 에서 사용해야 한다. 왜지??? 
* blocking call 은 별도의 Scheduler Worker 에서 실행하자. `publishOn()` 은 method chaining 에서 `publishOn()` 의 다음 method 부터 별도의 Scheduler Worker 에서 실행한다. 이것은 다음 `publishOn()` 이 호출될 때까지 유지된다. `subscribeOn` 은 전체 method 를 별도의 Scheduler Worker 에서 실행한다. 역시 다음 `publishOn()` 이 호출될 때까지 유지된다.
  * **publishOn** : Run **onNext**, **onComplete** and **onError** on a supplied Scheduler Worker. This operator influences the threading context where the rest of the operators in the chain below it will execute, up to a new occurrence of publishOn.
  * **subscribeOn** : Run **subscribe**, **onSubscribe** and **request** on a specified Scheduler's Scheduler.Worker. As such, placing this operator anywhere in the chain will also impact the execution context of **onNext/onError/onComplete** signals from the beginning of the chain up to the next occurrence of a publishOn.

다음과 같이 수정하자.

* `log()` 는 제거해야 한다.
* 너무 많은 `map()` 을 사용하지 않도록 하자.
* `cacheStorageAdapter.getAdValue(adCodeId)` 는 asynchronous method 이다. 그러나 `map()` 에서 사용하고 있다. `flatMap()` 에서 해야한다. 왜지???
* `AdRequest.getCode()` 는 blocking call 이다. event-loop 가 blocking 될 수 있다. 별도의 thread-pool 즉, 별도의 Scheduler Worker 에서 실행하자.
 
```java
public class AdHandler {
    public Mono<ServerResponse> fetchByAdRequest(ServerRequest serverRequest) {
        Mono<AdValue> adValueMono = serverRequest.bodyToMono(AdRequest.class)
            .publishOn(Schedulers.boundedElastic())
            .map(adRequest -> {
                    AdCodeId adCodeId = AdCodeId.of(AdRequest.getCode());
                    log.warn("Requested AdCodeId = {}", adCodeId.toKeyString());
                    return adCodeId;
                })
            .flatMap(adCodeId -> cacheStorageAdapter.getAdValue(adCodeId));
        return ServerResponse.ok()
            .contentType(MediaType.APPLICATION_JSON)
            .body(adValueMono, AdValue.class);
    }
}
```

Lettuce 를 사용하여 Redis 를 접근한다면 다음과 같이 `factory.setValidateConnection(true)` 호출을 피해야 한다.

```java
@Bean(name = "redisConnectionFactory")
public ReactiveRedisConnectionFactory connectionFactory() {
...
    RedisStandaloneConfiguration redisConfig
        = new RedisStandaloneConfiguration(redisHost,
                                           Integer.valueOf(redisPort));
    LettuceConnectionFactory factory
        = new LettuceConnectionFactory(redisConfig, clientConfig);
    factory.setValidateConnection(true);
    factory.setShareNativeConnection(true);
    return factory;
}
```

`factory.setValidateConnection(true)` 가 호출되면 Redis Command 를 전송할 때 마다 Connection Validation 을 수행한다.
이때 Redis Command 를 보낼 때마다 Ping Command 를 보내는 데 이것이 `sync().ping()` 호출에 이해 이루어 진다.
그러나 `sync().ping()` 은 blocking call 이다. 이 것을 피하면 성능을 향상할 수 있다.

```java
public class LettuceConnectionFactory {
    void validateConnection() { 
...
        synchronized(this.connectionMonitor) {
            boolean valid = false;
            if (this.connection != null && this.connection.isOpen()) {
                try {
                    if (this.connection instanceof StatefulRedisConnection) {
                        ((StatefulRedisConnection)this.connection).sync().ping();
                    }
...
                }
            }
        }
    }
}

public class StatefulRedisConnectionImpl<K, V> extends RedisChannelHandler<K, V> implements StatefulRedisConnection<K, V> {
    protected final RedisCodec<K, V> codec;
    protected final RedisCommands<K, V> sync;
    protected final RedisAsyncCommandsImpl<K, V> async; protected final RedisReactiveCommandsImpl<K, V> reactive;
    // 중략
}
```

## Reactive Cache

* [Spring Webflux Cache @ tistory](https://dreamchaser3.tistory.com/17)

----

WIP...

# Question

## map vs flatMap for async function

`cacheStorageAdapter.getAdValue(adCodeId)` 는 non-blocking function 이다. 다음의 code 에서 `flatMap(adCodeId -> cacheStorageAdapter.getAdValue(adCodeId))` 과 `map(adCodeId -> cacheStorageAdapter.getAdValue(adCodeId))` 은 어떤 차이가 있는 걸까?

```java
public class AdHandler {
    public Mono<ServerResponse> fetchByAdRequest(ServerRequest serverRequest) {
        Mono<AdValue> adValueMono = serverRequest.bodyToMono(AdRequest.class)
            .publishOn(Schedulers.boundedElastic())
            .map(adRequest -> {
                    AdCodeId adCodeId = AdCodeId.of(AdRequest.getCode());
                    log.warn("Requested AdCodeId = {}", adCodeId.toKeyString());
                    return adCodeId;
                })
            .flatMap(adCodeId -> cacheStorageAdapter.getAdValue(adCodeId));
        return ServerResponse.ok()
            .contentType(MediaType.APPLICATION_JSON)
            .body(adValueMono, AdValue.class);
    }
}
```

`Mono::map` 의 mapper 는 `AdValue` 를 return 할 테고 `Mono::flatMap` 의 mapper 는 `Mono<AdValue>` 를 
return 할 것이다. `Mono::map()` 과 `Mono::flatMap` 를 바꿔가면서 사용할 수 있는 것인가?
