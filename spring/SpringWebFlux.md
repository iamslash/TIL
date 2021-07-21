- [Abstract](#abstract)
- [Materials](#materials)
- [Basics](#basics)
  - [Project Reactor](#project-reactor)
  - [Example Application](#example-application)

----

# Abstract

Spring WebFlux 에 대해 정리한다.

# Materials

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
