# Abstract

**reactive programming** is a declarative programming paradigm concerned with data streams and the propagation of change.

For example, in an **imperative programming**, `a := b + c` would mean that `a` is being assigned the result of `b + c` in the instant the expression is evaluated, and later, the values of `b` and `c` can be changed with no effect on the value of `a`.

On the other hand, in **reactive programming**, the value of `a` is automatically updated whenever the values of `b` or `c` change, without the program having to re-execute the statement `a := b + c` to determine the presently assigned value of`a`.

Reactive runtime 은 request, response 를 적당히 조율해서 system 의 자원이 허용하는 한도에서 thread 사용효율을 최대화 한다. 따라서 System 의 throughput 이 높아진다. Spring data project leader 인 Mark Paluch 는 이것을 "시스템 자원 가용성에 반응한다." 라고 말했다.

Reactive runtime 은 backpressure 를 지원한다. Consumer 가 필요한 만큼만 Producer 에게서 얻어온다. 이것은 thread 사용효율을 최대화하는 것과 같다. reactive code 를 선언형으로 잘 만들어 놔도 Consumer 가 소비하지 않으면 아무일도 일어나지 않는다.

# Materials

* [스프링 부트 실전 활용 마스터 스프링 부트 개발과 운영부터 웹플럭스, R소켓, 메시징을 활용한 고급 리액티브 웹 개발까지 @ yes24](http://www.yes24.com/Product/Goods/101803558)
  * [src](https://github.com/onlybooks/spring-boot-reactive)
* [3월 우아한 Tech 세미나 후기, 스프링 리액티브 프로그래밍 @ 우아한형제들](https://woowabros.github.io/experience/2019/03/18/tech-toby-reactive.html)
* [사용하면서 알게 된 Reactor, 예제 코드로 살펴보기 @ kakao](https://tech.kakao.com/2018/05/29/reactor-programming/)
* [스프링캠프 2017 [Day1 A3] : Spring Web Flux @ youtube](https://www.youtube.com/watch?reload=9&v=2E_1yb8iLKk)
* [스프링5 웹플럭스와 테스트 전략 @ kakaoTV](https://tv.kakao.com/channel/3150758/cliplink/391418995)

# Concepts

[Reactive Streams @ wikipedia](https://en.wikipedia.org/wiki/Reactive_Streams) is a JVM standard for asynchronous stream processing with non-blocking backpressure

[Reacdtive manifesto](https://www.reactivemanifesto.org/ko)

* Responsive
* Resilient
* Elastic
* Message Driven

# Why spring 5 was born???

* [Reactive와 Spring 4 (C10K, 리액티브 선언문, 리액티브 스프링 등장 전)](https://sjh836.tistory.com/179)

----

To solve C 10K problems we needed reactive programming. That's why spring 5 was born with reactive framework.
