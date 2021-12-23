- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
  - [Terms](#terms)
  - [DDD Tutorial](#ddd-tutorial)
  - [Layered Architecture](#layered-architecture)
  - [Domain Layer](#domain-layer)
  - [DIP (Dependency Inversion Principle)](#dip-dependency-inversion-principle)
  - [Aggregate](#aggregate)
  - [Bounded Context](#bounded-context)

------

# Abstract

DDD (Domain Driven Design) 를 정리한다.

# Materials

* [DDD START! 도메인 주도 설계 구현과 핵심 개념 익히기](http://www.yes24.com/Product/Goods/27750871)
  * [src](https://github.com/madvirus/ddd-start)
  * [DDD START! 수다 #1 @ youtube](https://www.youtube.com/watch?v=N3NSISzolSw)
  * [DDD START! 수다 #2 @ youtube](https://www.youtube.com/watch?v=OjMshMPVx5I)
  * [DDD START! 수다 #3 @ youtube](https://www.youtube.com/watch?v=BE5ysejA2cQ)

# Basic

## Terms

| Term | Description  |
|--|--|
| Domain | 해결하고자 하는 문제영역 |
| Domain Model | Domain 을 개념적으로 표현한 것. 주로 UML (Class, Activity, Sequence Diagram) 로 표현한다. | |

## DDD Tutorial

* 요구사항을 파악한다. 파악한 것을 그림 및 글로 정리한다. 
  * Context map
  * 개념모델을 구현모델로 바꾸어 간다. 구현모델은 Entity 와 Value 로 나뉘어진다.
  * 도메인 용어 사전을 작성하여 기획자와 sync-up 한다.
  * UML (Class Diagram, Activity Diagram, Sequence Diagram) 을 작성한다.
  * 도메인 규칙을 글로정리해본다. 그리고 Class 별로 Access pattern 을 정리한다.
* Layered Architecture 를 설계한다.
  * Presentation
  * Application
  * Domain
  * Infrastructure
* 구현한다. 
* 디버깅한다.

## Layered Architecture

* [Layered Architecture](/architecture/README.md)

-----

| Layer | Description |
|---|---|
| Presentation | User 에 보여지는 화면 또는 데이터를 구현한다. |
| Application | User 가 원하는 기능을 Domain layer 를 조합하여 구현한다. |
| Domain | Domain model 을 구현한다. | 
| Infrastructure | RDBMS, MQ 와 같은 외부시스템 연동을 구현한다. |

## Domain Layer

Domain Layer 는 Entity, Value, Aggregate, Repository, Domain Service 로 구성된다. Aggregate 가 명사로 쓰일 때는 애그리거트라고 발음한다. 동사로 쓰일때는 애그리게이트이다.

| Component | Description |
|---|----|
| Entity  | 식별자를 갖는 Domain Model 이다. 자신의 라이프사이클을 갖는다. 예를 들어 Order, Member, Product 등이 해당된다. |
| Value | 식별자를 갖지 않는 Domain Model 이다. 주로 Entity 의 속성이다. 예를 들어 Address, Money 가 해당된다. |
| Aggregate | 관련된 Entity, Value 객체를 개념적으로 하나로 묶은 것이다. 예를 들어, 주문 Aggregate 은 Order Entity, OrderLine Value, Orderer Value 를 포함한다. |
| Repository | Domain Model 의 영속성을 제공한다. 예를 들어 DBMS 테이블의 CRUD 를 제공한다. |
| Domain Service | Domain Logic 을 제공한다. 예를 들어 "할인 금액 계산" Domain Service 는 상품, 쿠폰, 회원등급, 구매 금액등 다양한 조건을 이용해서 할인된 금액을 계산한다. |

## DIP (Dependency Inversion Principle)

* [DIP @ learntocode](/solid/README.md#dip)

## Aggregate

Aggregate 는 Entity, Value 로 구성된다. 하나의 Aggregate 는 하나의 Repositoty 에 대응된다.

Aggregate Root 는 Aggregate 의 대표 Domain Model 즉 대표 Entity 를 말한다. Aggregate 의 모든 Domain Model 은 직접 혹은 간접적으로 Aggregate Root 와 관련되어 있다. 

하나의 Aggregate 에서 다른 Aggregate 를 참조할 때는 Aggregate Root 를 참조한다. 예를 들어 Order Aggregate 의 Orderer Entity 는 Member Aggregate 의 Member Entity 를 참조한다.

그림 필요 TODO

Aggregate 의 도메인 규칙을 지키기 위해서는 모든 도메인 모델이 일관성을 가지고 있어야 한다. Aggregate Root 는 모든 도메인 모델의 상태를 관리하는 역할을 한다. 예를 들어 주문 Aggregate 에서 Aggregate Root 는 Order Entity 이다.


그림 필요 TODO

## Bounded Context
