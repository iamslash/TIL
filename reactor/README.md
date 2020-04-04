# Materials

* [Reactive Programming with Reactor 3 @ tech.io](https://tech.io/playgrounds/929/reactive-programming-with-reactor-3/Intro)
  * [sol](https://gist.github.com/kjs850/a29addc92b98b51ea05a09587be34071)
* [[리액터] 리액티브 프로그래밍 1부 리액티브 프로그래밍 소개 @ youtube](https://www.youtube.com/watch?v=VeSHa_Xsd2U&list=PLfI752FpVCS9hh_FE8uDuRVgPPnAivZTY)
* [reactor reference](https://projectreactor.io/docs/core/release/reference/)

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
        // Flux from empty
        Flux<String> flux = Flux.empty();
        // Flux from String
        Flux<String> flux = Flux.just("foo", "bar", "baz");
        // Flux from Iterable
        Flux<String> flux = Flux.fromIterable(Arrays.asList("foo", "bar", "baz"));
        Flux.fromIterable(Arrays.asList("foo", "bar", "baz"))
                .doOnNext(System.out::println)
                .blockLast();
        // take 10 elements of Flux and cancel
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
        // Use VirtualTimer when test big Flux.
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

* [Spring’s WebFlux / Reactor Parallelism and Backpressure](https://www.e4developer.com/2018/04/28/springs-webflux-reactor-parallelism-and-backpressure/)

Publisher 가 제공한 data 를 Subscriber 가 throttling 하는 것을 Back pressure 하고 한다. 즉, Subscriber 가 요청한 만큼만 data 가 넘어온다. request() 를 통해 back pressure 를 할 수 있다. 그러나 request() 는 기본적으로 unbounded 이다. Subscriber class 를 통해 back pressure 할 수 있다. 

![](https://i1.wp.com/www.e4developer.com/wp-content/uploads/2018/04/backpressure-handling.jpg?w=1680&ssl=1)

```java
        // StepVerifier handle request (backpressure) with create function.
        Flux<Long> flux = Flux.just(1l, 2l, 3l, 4l);
        StepVerifier.create(flux)
                .expectNextCount(4)
                .expectComplete();
        // StepVerifier handle request (backpressure) with create, thenRequest.
        // thenCancel will cancel the reactor stream.
        StepVerifier.create(flux, 1)
                .expectNext(User.SKLER)
                .thenRequest(1)
                .expectNext(User.JESSE)
                .thenCancel();
        //
        repository.findAll().log();
        //
        repository.findAll().logs()
                .doONSubscribe(a -> System.out.println("Starting:"))
                .doOnNext(p -> System.out.println(p.getFirst() + p.getLastName()))
                .doOnComplete(() -> System.out.println("The end!"));
        //
        Flux<Long> flux = Flux.interval(Duration.ofMillis(100))
                .take(4)
                .log();
        flux.doOnNext(a -> {
            System.out.println(a);
            System.out.println(Thread.currentThread().getName());
        }).blockLast();
        // Spring will handle Subscription request.
        List<Integer> vals = new ArrayList<>();
        for (int i = 0; i < Integer.MAX_VALUE; i++)
            vals.add(i);
        Flux<Integer> flux = Flux.fromIterable(vals).log();
        flux.doOnNext(a -> {
            System.out.println(a);
            System.out.println(Thread.currentThread().getName());
        }).blockLast();
        // verify should be after expectComplete
        Flux<Integer> flux = Flux.just(0, 1, 2).log();
        StepVerifier.create(flux)
                .expectNext(0)
                .expectComplete()
                .verify();
        // You can cancel the reactor stream with thenCancel.
        // StepVerifier should be end with verify().
        Flux<Integer> flux = Flux.just(0, 1, 2).log();
        StepVerifier.create(flux)
                .expectNext(0)
                .thenRequest(1)
                .expectNext(1)
                .thenCancel()
                .verify();
        // request is unbounded
        Flux.range(1, 100)
                .doOnNext(System.out::println)
                .subscribe();
        // You can handle request by Subscriber
        // This is a back pressure.
        // This will request 10 after requesting 10.
        Flux.range(1, 100)
                .log()
                .doOnNext(System.out::println)
                .subscribe(new Subscriber<Integer>() {
                    private Subscription subscription;
                    private int count;
                    @Override
                    public void onSubscribe(Subscription s) {
                        this.subscription = s;
                        this.subscription.request(10);
                    }

                    @Override
                    public void onNext(Integer integer) {
                        count++;
                        if (count % 10 == 0) {
                            this.subscription.request(10);
                        }
                    }

                    @Override
                    public void onError(Throwable t) {

                    }

                    @Override
                    public void onComplete() {

                    }
                });
```

## Error

```java
        // Return 2 when error has occurred.
        Mono<Object> mono = Mono.error(new RuntimeException());
        mono
                .log()
                .onErrorReturn(2)
                .doOnNext(System.out::println)
                .subscribe();
        // Return Mono when error has occurred.
        Mono<Object> mono = Mono.error(new RuntimeException());
        mono
                .log()
                .onErrorReturn(Mono.just(2))
                .doOnNext(System.out::println)
                .subscribe();
        // Return reactor stream when error. You should use lambda for onErrorResume.
        Mono<Object> mono = Mono.error(new RuntimeException());
        mono
                .log()
                .onErrorResume(e -> Mono.just(2))
                .doOnNext(System.out::println)
                .subscribe();
        // answer 1
        return mono.onErrorResume(e -> Mono.just(User.SAUL));
        // answer 2
        return flux.onErrorResume(e -> Flux.just(User.SAUL, User.JESSE));
        // answer 3
        // GetOutOfHereException is checked exception because it is not decendant of RuntimeException.
        // You can propagate the checked exception switching with the runtime exception.
        return flux.map(u -> {
            try {
                return capitalizedUser(u);
            }
            catch (GetOutOfHereException e) {
                throw Exceptions.propagate(e);
            }
        });
        User capitalizeUser(User user) throws GetOutOfHereException {
            if (user.equals(User.SAUL)) {
                throw new GetOutOfHereException();
            }
            return new User(user.getUsername(), user.getFirstname(), user.getLastname());
        }
        protected final class GetOutOfHereException extends Exception {
        }
        Mono.just("Foo")
                .log()
                .map(s -> {
                    int val = 0;
                    try {
                        Integer.parseInt(s);
                    } catch (Exception e) {
                        throw Exceptions.propagate(e);
                    }
                    return val;
                })
                .onErrorReturn(200)
                .doOnNext(System.out::println)
                .subscribe();
```

## Adapt

Rxjava 2 와 Reactor 3 를 switching 한다. 즉, Flux 와 Flowable 를 switching 한다.

```java
       // answer 1
       Flowable flowable = Flowable.fromPublisher(Flux.just(2));
       // answer 2
       Flux<Integer> flux = Flux.from(flowable);
       // answer 3
       Flux<Integer> flux = Flux.just(2);
       Observable<Integer> observable = Observable.just(2);
       // answer 4
       Flux.from(observable.toFlowable(BackpressureStrategy.BUFFER));
       // answer 5
       Mono<Integer> mono = Mono.just(2);
       Single<Integer> single = Single.just(2);
       Mono.from(single.toFlowable());
       Single.fromPublisher(mono);
       CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> "hello");
       future.thenApply(s -> s.toUpperCase());
       future.get();
       Mono<String> mono = Mono.just("Hello").map(s -> s.toUpperCase());
       Mono.fromFuture(future);
       mono.toFuture();
```

## Others Operations

```java
        // zip
        Flux<Integer> f1 = Flux.range(0, 10);
        Flux<Integer> f2 = Flux.range(11, 20);
        Flux<Integer> f3 = Flux.range(21, 30);
        Flux.zip(f1, f2, f3)
                .map(tuple -> tuple.getT1());
        // answer 1
        Flux.zip(usernameFlux, firstnameFlux, lastnameFlux)
                .map((tuple) -> new User(tuple.getT1(), tuple.getT2(), tuple.getT3()));
        // answer 2
        Mono.first(ono1,mono2);
        // answer 3
        Flux.first(flux1, flux2);
        // answer 4
        flux.then();
        // answer 5
        Mono.justOrEmpty(user);
        // answer 6
        mono.defaultIfEmpty(User.SKLER);
```

## Reactive to Blocking

```java
public class Part10ReactiveToBlocking {
        User monoToValue(Mono<User> mono) {
                return mono.block();
        }
        Iterable<User> fluxToValues(Flux<User> flux) {
                return flux.toIterable();
        }
}
```

## Blocking to Reactive

The subscribeOn method allow to isolate a sequence from the start on a provided Scheduler.
subscribeOn run subscribe, onSubscribe and request on a specified Scheduler's Scheduler.Worker.

publishOn run onComplete and onError on a supplied Scheduler Worker. This operator influences the treading context where the rest of the operators in the chain below it will execute, up to a new occurence of publishOn.

```java
public class Part11BlockingToReactive {
	Flux<User> blockingRepositoryToFlux(BlockingRepository<User> repository) {
		return Flux.defer(() -> Flux.fromIterable(repository.findAll()))
                        .subscribeOn(Schedulers.elastic());
	}
	Mono<Void> fluxToBlockingRepository(Flux<User> flux, BlockingRepository<User> repository) {
		return flux.publishOn(Schedulers.elastic())
                        .doOnNext(repository::save)
                        .then();
	}
}
```
