- [Abstract](#abstract)
- [Materials](#materials)
- [Basics](#basics)
  - [Project Reactor](#project-reactor)
  - [Example Application](#example-application)
- [Advanced](#advanced)
  - [Reactor MeltDown](#reactor-meltdown)

----

# Abstract

Spring WebFlux 에 대해 정리한다.

# Materials

* [Webflux로 막힘없는 프로젝트 만들기 @ ifkakao2021](https://if.kakao.com/session/107)
  * event-loop 를 당당하는 single thread 가 blocking 되면 Reactor meltdown 을 일으킨다. 그것을 해결하는 방법을 제시한다. 
  * [src](https://github.com/beryh/event-loop-demo)
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

----

Netty 의 Event-Loop 는 Single Thread 으로 동작한다. Single Thread 가 blocking 되면 Reactor MeltDown 을 일으킨다. 즉, Event-Loop 가 멈춘다. blocking call 은 모두 제거해야 한다.

[Block Hound](https://github.com/reactor/BlockHound) 를 이용하여 run-time 에 blocking call 을 발견하자. [Block Hound](https://github.com/reactor/BlockHound) 는 별도의 agent 에서 동작한다. byte code 를 조작한다. 따라서 production profile 에서 사용하지 말자. test 에서만 사용하자. [Block Hound](https://github.com/reactor/BlockHound) 는 다음과 같이 간단히 실행할 수 있다.

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




