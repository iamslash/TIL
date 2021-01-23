# Abstract

**reactive programming** is a declarative programming paradigm concerned with data streams and the propagation of change.

For example, in an **imperative programming**, `a := b + c` would mean that `a` is being assigned the result of `b + c` in the instant the expression is evaluated, and later, the values of `b` and `c` can be changed with no effect on the value of `a`.

On the other hand, in **reactive programming**, the value of `a` is automatically updated whenever the values of `b` or `c` change, without the program having to re-execute the statement `a := b + c` to determine the presently assigned value of`a`.

# Materials

* [3월 우아한 Tech 세미나 후기, 스프링 리액티브 프로그래밍 @ 우아한형제들](https://woowabros.github.io/experience/2019/03/18/tech-toby-reactive.html)
* [사용하면서 알게 된 Reactor, 예제 코드로 살펴보기 @ kakao](https://tech.kakao.com/2018/05/29/reactor-programming/)
* [스프링캠프 2017 [Day1 A3] : Spring Web Flux @ youtube](https://www.youtube.com/watch?reload=9&v=2E_1yb8iLKk)
* [스프링5 웹플럭스와 테스트 전략 @ kakaoTV](https://tv.kakao.com/channel/3150758/cliplink/391418995)

# Concepts

[Reactive Streams @ wikipedia](https://en.wikipedia.org/wiki/Reactive_Streams) is a JVM standard for asynchronous stream processing with non-blocking backpressure
