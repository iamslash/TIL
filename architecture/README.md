- [Abstract](#abstract)
- [Materials](#materials)
- [Architectures](#architectures)
  - [Layered Architecture Advantages, Disadvantages](#layered-architecture-advantages-disadvantages)
  - [Clean Architecture](#clean-architecture)
  - [Unified Architecture for Data Infrastructure](#unified-architecture-for-data-infrastructure)

----

# Abstract

Architecture 들에 대해 정리한다.

# Materials

* [Explicit Architecture](https://herbertograca.com/2017/11/16/explicit-architecture-01-ddd-hexagonal-onion-clean-cqrs-how-i-put-it-all-together/)

# Architectures

## Layered Architecture Advantages, Disadvantages

* [Layered Architecture](https://www.baeldung.com/cs/layered-architecture)

**Advantages**

* 새로운 사람이 파악하기 쉽다. Hand over 가 쉽고 빠르다. 
* 유지보수가 쉽다. 새로운 feature 를 구현할 때 패턴이 명확해서 구현이 쉽고 빠르다.  
* 의존성이 적다. 단 하나의 레이어에만 의존성이 있다.
* 의존성이 적어서 테스트가 쉽다.

**Disadvantages**

* 큰 변경은 어렵다. 하나의 레이어가 크게 바뀐다면 다른 레이어를 바꿔야할 수도 있다.

## Clean Architecture

* [Clean Architecure](/cleanarchitecture/README.md)
* [Hexagonal Architecture](/hexagonalarchitecture/README.md)
* [Explicit Architecture](https://herbertograca.com/2017/11/16/explicit-architecture-01-ddd-hexagonal-onion-clean-cqrs-how-i-put-it-all-together/)

## Unified Architecture for Data Infrastructure

* [Emerging Architectures for Modern Data Infrastructure](https://future.a16z.com/emerging-architectures-modern-data-infrastructure/)
* [최신 데이터 인프라 이해하기 @ youtube](https://www.youtube.com/watch?v=g_c742vW8dQ&list=PLL-_zEJctPoJ92HmbGxFv1Pv_ugsggGD2)

----

Data 를 기반으로 의사결정을 하기 위해 Data Infrastructure 가 필요하다 (Data-Driven Decision Making). 아래 그림의 용어들을 모두 이해해 보자.

![](img/unified_data_infrastructure_architecture.png)
  