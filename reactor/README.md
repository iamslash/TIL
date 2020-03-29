# Materials

* [Reactive Programming with Reactor 3 @ tech.io](https://tech.io/playgrounds/929/reactive-programming-with-reactor-3/Intro)
* [[리액터] 리액티브 프로그래밍 1부 리액티브 프로그래밍 소개 @ youtube](https://www.youtube.com/watch?v=VeSHa_Xsd2U&list=PLfI752FpVCS9hh_FE8uDuRVgPPnAivZTY)


# Asynchronous VS non-blocking

* [Blocking-NonBlocking-Synchronous-Asynchronous](https://homoefficio.github.io/2017/02/19/Blocking-NonBlocking-Synchronous-Asynchronous/)

Synchronous 와 Asynchronous 의 관심사는 job 이다. 즉, 어떤 A job, B job 을 수행할 때 A job 이 B job 과 시간을 맞추면서 실행하면 Synchronous 이다. 그러나 시간을 맞추지 않고 각자 수행하면 Asynchronous 이다.

Blocking 과 Non-blocking 의 관심사는 function 이다. 즉, A function 이 B function 을 호출할 때 B function 이 리턴할 때까지 A function 이 기다린다면 Blocking 이다. 그러나 B function 이 리턴하기 전에 A function 이 수행할 수 있다면 Non-blocking 이다.

# Basic

## Flux

Flux is a publisher which implements Publisher.

```java
//        Flux is a publisher which implements Publisher
//        basic Flux
        Flux<String> flux = Flux.just("foo", "bar", "baz");
        flux.map(a -> a + " hello");
        flux.subscribe(System.out::println);
        Flux<String> flux = Flux.empty();
        Flux<String> flux = Flux.just("foo", "bar", "baz");
        Flux<String> flux = Flux.fromIterable(Arrays.asList("foo", "bar", "baz"));
        Flux.fromIterable(Arrays.asList("foo", "bar", "baz"))
                .doOnNext(System.out::println)
                .blockLast();
        Flux.interval(Duration.ofMillis(100))
                .take(10)
                .subscribe(System.out::println);
        Flux.error(new IllegalStateException());
```

## Mono

Mono is a publisher which implements Publisher.

```java
        // Mono is a publisher which implements Publisher
        // basic Mono
        Mono<String> mono = Mono.just("foo");
        mono.map(a -> a + " bar baz");
        mono.subscribe(System.out::println);

        Mono<Long> delay = Mono.delay(Duration.ofMillis(100));
        Mono.just(1l)
                .map(i -> i * 2)
                .or(delay)
                .subscribe(System.out::println);

        Mono.just(1)
                .map(i -> i * 2)
                .or(Mono.just(100))
                .subscribe(System.out::println);

        Mono<String> mono = Mono.empty();

        Mono<String> mono = Mono.never();

        Mono<String> mono = Mono.just("foo");

        Mono.error(new IllegalStateException());
```

## StepVerifier

You can validate Mono, Flux with StepVerifier.

```java
        // StepVerifier
        // You can validate Mono, Flux with StepVerifier.
        // for example think about test codes.
        Flux<String> flux = Flux.just("foo", "bar");
        StepVerifier.create(flux)
                .expectNext("foo")
                .expectNext("bar")
                .verifyComplete();
        StepVerifier.create(flux)
                .expectNext("foo")
                .expectNext("bar")
                .verifyError(RuntimeException.class);
        StepVerifier.create(flux)
                .assertNext(a -> Assertions.assertSame(a, "foo"))
                .assertNext(a -> Assertions.assertSame(a, "bar"))
                .verifyComplete();

        Flux<Long> take10 = Flux.interval(Duration.ofMillis(100))
                .take(10);
        StepVerifier.create(take10)
                .expectNextCount(10)
                .verifyComplete();
         Use VirtualTimer when test big Flux.
        StepVerifier.withVirtualTime(() -> Mono.delay(Duration.ofHours(3)))
                .expectSubscription()
                .expectNoEvent(Duration.ofHours(2))
                .thenAwait(Duration.ofHours(1))
                .expectNextCount(1)
                .expectComplete()
                .verify();
        StepVerifier.withVirtualTime(flux)
                .thenAwait(Duration.ofHours(1))
                .expectNextCount(3600)
                .verifyComplete();
```

## Transform

map is synchronous, flatMap is asynchronous.

![](https://raw.githubusercontent.com/reactor/projectreactor.io/master/src/main/static/assets/img/marble/flatmap.png)

flatMapSequential supports sequnce with parallel.

![](https://projectreactor.io/docs/core/release/api/reactor/core/publisher/doc-files/marbles/flatMapSequential.svg)

```java
        Flux<String> flux = Flux.just("a", "b", "c", "d", "e", "f", "g", "h", "i");
        flux
                .map(a -> a + " 0")
                .doOnNext(System.out::println)
                .blockLast();
        flux
                .flatMap(a -> Mono.just(a + " 0"))
                .doOnNext(System.out::println)
                .blockLast();
        flux
                .window(3)
                .flatMap(a -> a.map(this::toUpperCase))
                .doOnNext(System.out::println)
                .blockLast();
        // parallel support parallel
        flux
                .window(3)
                .flatMap(a -> a.map(this::toUpperCase))
                .subscribeOn(parallel())
                .doOnNext(System.out::println)
                .blockLast();
        // concatMap support sequence so parallel is no use
        flux
                .window(3)
                .concatMap(a -> a.map(this::toUpperCase))
                .subscribeOn(parallel())
                .doOnNext(System.out::println)
                .blockLast();
        // flaflatMapSequential tMap support sequence with parallel.
        flux
                .window(3)
                .flatMapSequential(a -> a.map(this::toUpperCase))
                .subscribeOn(parallel())
                .doOnNext(System.out::println)
                .blockLast();

    private List<String> toUpperCase(String s) {
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return Arrays.asList(s.toUpperCase(), Thread.currentThread().getName());
    }
```

## Merge

```java
        Flux<Long> flux1 = Flux.interval(Duration.ofMillis(100)).take(10);
        Flux<Long> flux2 = Flux.just(100l, 101l, 102l);
        // mergeWith
        flux1.mergeWith(flux2)
                .doOnNext(System.out::println)
                .blockLast();
        // concatWith
        flux1.concatWith(flux2)
                .doOnNext(System.out::println)
                .blockLast();
        // concat
        Mono<Integer> mono1 = Mono.just(1);
        Mono<Integer> mono2 = Mono.just(2);
        Flux.concat(mono1, mono2)
                .doOnNext(System.out::println)
                .blockLast();
```

## Request

## Error

## Adapt

## Others Operations
