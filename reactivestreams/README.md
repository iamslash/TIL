- [Abstract](#abstract)
- [Materials](#materials)
- [Basics](#basics)

----

# Abstract

Reactive Streams 는 non-blocking back pressure 와 함께 asynchronous stream processing 을 제공하기 위한 제안이다. rxJava, [reactor](/reactor/README.md) 는 대표적인 implementation 들이다.

# Materials

* [Reactive Streams](https://www.reactive-streams.org/)

# Basics

[Package org.reactivestreams API](https://www.reactive-streams.org/reactive-streams-1.0.3-javadoc/org/reactivestreams/package-summary.html) 를 살펴보면 Reactive Streams 의 API 를 이해할 수 있다.

[SPECIFICATION](https://github.com/reactive-streams/reactive-streams-jvm/blob/v1.0.3/README.md#specification) 을 살펴보면 Reactive Streams 를 어떻게 구현해야 하는지를 이해할 수 있다. Reactive Streams 는 다음과 같이 4 가지 Components 로 구성된다.

* Publisher
* Subscriber
* Subscription
* Processor

각각의 interface 는 다음과 같다. 요것만 잘 구현하면 끝인가???

```java
public interface Publisher<T> {
    public void subscribe(Subscriber<? super T> s);
}

public interface Subscriber<T> {
    public void onSubscribe(Subscription s);
    public void onNext(T t);
    public void onError(Throwable t);
    public void onComplete();
}

public interface Subscription {
    public void request(long n);
    public void cancel();
}

public interface Processor<T, R> extends Subscriber<T>, Publisher<R> {
}
```

[example implementations](https://github.com/reactive-streams/reactive-streams-jvm/tree/v1.0.3/examples/src/main/java/org/reactivestreams/example/unicast) 는 implementation 의 예이다.

[The Technology Compatibility Kit (TCK)](https://github.com/reactive-streams/reactive-streams-jvm/tree/v1.0.3/tck) 는 제대로 구현했는지 검증하기 위한 Test Kit 이다.
