# Abstract

DDD (Domain Driven Design) 은 Domain 을 중심으로 System 을 Design 하는 것이다. 보통 Event storming 을 통해 Bounded Context 들을 구성한다. Boris Diagram, SnapE 의 과정을 거쳐 MSA (Micro Service Architecture) 를 디자인 한다.

# Materials

* [Domain Driven Design 이란 무엇인가?](https://frontalnh.github.io/2018/05/17/z_domain-driven-design/)
* [마이크로서비스 개발을 위한 Domain Driven Design @ youtube](https://www.youtube.com/watch?v=QUMERCN3rZs&feature=youtu.be)
* [MSA 전략 1: 마이크로서비스, 어떻게 디자인 할 것인가? @ slideshare](https://www.slideshare.net/PivotalKorea/msa-1-154454835)
* ["마이크로 서비스 어디까지 해봤니?" 정리](https://syundev.tistory.com/125)

# Basic

## Event Storming

* [마이크로서비스 개발을 위한 Domain Driven Design @ youtube](https://www.youtube.com/watch?v=QUMERCN3rZs&feature=youtu.be)
* [KCD 2020 [Track 2] 도메인 지식 탐구를 위한 이벤트 스토밍 Event Storming | youtube](https://www.youtube.com/watch?v=hUcpv5fdCIk)

-----

* Orange Sticker : Domain Event, something that happended.
  * `Item Added to Cart`
* Blue Sticker : Command, Request to trigger, source of domain devent.
  * `Item Add to Cart`
* Yellow Sticker : Aggregate, Object which has several attributes
 
  ```java
  @entity
  public class Item {
    private String sku;
    private String name;
    private Int quantity;
    private ProductId productId;
  }
  ```
* Red sticker : External system

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FmUee9%2FbtqBil20Nun%2FfLOZVDQ6d2CngWnxXkiOO1%2Fimg.png)

## Boris Diagram

Aggreate 들 간에 synchronous, asynchronous event 들을 표기한다.

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FmWJYL%2FbtqBkqISPTh%2FIgxDNQPqxPRoyVsolHKqWk%2Fimg.png)

## SnapE

Aggregate 는 하나의 Micro Service 에 대응된다. API, Data, Stories, Risk, UI 를 기술한다.

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FNNQEH%2FbtqBjAMfua2%2F7dMIJawGC5ZI8pCpvOGQF0%2Fimg.png)
