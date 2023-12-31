- [Abstract](#abstract)
- [Materials](#materials)
- [History](#history)
- [Spring Cloud VS Kubernetes](#spring-cloud-vs-kubernetes)
- [Pattern](#pattern)
  - [Application architecture patterns](#application-architecture-patterns)
  - [Decomposition](#decomposition)
  - [Refactoring to microservices](#refactoring-to-microservices)
  - [Data management](#data-management)
  - [Transactional messaging](#transactional-messaging)
  - [Testing](#testing)
  - [Deployment patterns](#deployment-patterns)
  - [Cross cutting concerns](#cross-cutting-concerns)
  - [Communication style](#communication-style)
  - [External API](#external-api)
  - [Service discovery](#service-discovery)
  - [Reliability](#reliability)
  - [Security](#security)
  - [Observability](#observability)
  - [UI patterns](#ui-patterns)

----

# Abstract

마이크로 서비스 아키텍처는 애플리케이션을 작은 크기의 모듈식 및 독립적으로 배포
가능한 서비스들의 모음으로 구조화하는 소프트웨어 개발 방식입니다. 각
마이크로서비스는 단일하고 구체적인 기능을 수행하도록 설계되며, 애플리케이션의
나머지 부분에서 독립적으로 개발, 배포 및 확장할 수 있습니다.
마이크로서비스끼리는 경량 API, 표준화된 프로토콜 또는 메시징 메커니즘을 사용하여
통신합니다.

이 아키텍처는 전통적인 모노리식 아키텍처에 비해 더 세밀하고 유연한 접근 방식을
제공하여 복잡한 애플리케이션을 구축할 수 있습니다. 전통적인 모노리식
아키텍처에서는 전체 애플리케이션이 하나의 연결된 단위로 구축됩니다.

마이크로서비스 아키텍처의 주요 특성은 다음과 같습니다:

- 분산 및 모듈식: 각 마이크로서비스는 자체 코드베이스, 데이터 저장 및 인프라를
  갖추고 있어 독립적으로 개발 및 관리할 수 있습니다.
- 도메인 주도 설계: 마이크로서비스는 특정 비즈니스 기능이나 도메인 개념과
  일치하므로 애플리케이션의 목적 및 기능에 대한 이해를 촉진합니다.
- 확장성: 각 마이크로서비스는 독립적으로 확장할 수 있어 효율적인 자원 활용 및
  애플리케이션의 다른 부분에 대한 가변적인 부하 처리가 가능합니다.
- 유연성: 다른 마이크로서비스는 서로 다른 프로그래밍 언어, 프레임워크, 도구를
  사용하여 개발할 수 있어 팀이 특정 서비스에 가장 적합한 기술을 선택할 수
  있습니다.
- 탄력성: 각 서비스가 격리되어 있고 필요한 장애 허용 및 대체 메커니즘을 설계할
  수 있으므로 한 마이크로서비스의 실패 또는 문제가 전체 애플리케이션에 미치는
  영향이 줄어듭니다.

그러나 마이크로 서비스 아키텍처는 서비스 간 통신, 서비스 검색, 데이터 일치성
관리의 복잡성이 증가하는 등의 일부 도전 과제를 초래하기도 합니다. 이러한 도전
과제에 대응하기 위해 조직들은 종종 컨테이너화, 서비스 메시, API 게이트웨이, 분산
추적과 같은 도구와 기법에 의존합니다.

마이크로서비스 아키텍처는 큰 규모의 복잡하고 고성능 애플리케이션을 구축하며 쉽게
업데이트하고 유지 관리할 수 있는 시스템을 구축하려는 조직들에게 인기가 있습니다.
마이크로 서비스 아키텍처를 사용하는 회사들의 예로는 아마존, 넷플릭스, 스포티파이
등이 있습니다.

# Materials

- [The What, Why, and How of a Microservices Architecture](https://medium.com/hashmapinc/the-what-why-and-how-of-a-microservices-architecture-4179579423a9)
- [Microservices Lessons From Netflix](https://newsletter.systemdesign.one/p/netflix-microservices)
- [도서 마이크로서비스 아키텍처 구축 가이드(한빛미디어)](https://m.hanbit.co.kr/store/books/book_view.html?p_code=B9403623796)
  - [src](https://github.com/wharup/book-examples)
- [A pattern language for microservices](https://microservices.io/patterns/index.html)
  - microservices 의 기본개념
  - [ftgo-monolith src](https://github.com/microservices-patterns/ftgo-monolith)
  - [ftgo-msa src](https://github.com/microservices-patterns/ftgo-application)
  - [ftgo-msa src from gilbut](https://github.com/gilbutITbook/007035) 
  - [eventuate-tram](https://eventuate.io/abouteventuatetram.html)
    - sagas, CQRS, transactional outbox 등을 지원하는 framework 이다. ftgo 에 사용되었다.
  - [eventuate-local](https://eventuate.io/usingeventuate.html)
    - event sourcing 을 지원하는 framework 이다. ftgo 에 사용되었다.
  - [eventuate-foundation](https://github.com/orgs/eventuate-foundation/repositories)
    - eventuate-tram 이 사용하는 library
- [MSA 제대로 이해하기 -(1) MSA의 기본 개념](https://velog.io/@tedigom/MSA-%EC%A0%9C%EB%8C%80%EB%A1%9C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0-1-MSA%EC%9D%98-%EA%B8%B0%EB%B3%B8-%EA%B0%9C%EB%85%90-3sk28yrv0e)
- [Microservices | wikipedia](https://en.wikipedia.org/wiki/Microservices)
- [[우아콘2020] 배달의민족 마이크로서비스 여행기 | youtube](https://www.youtube.com/watch?v=BnS6343GTkY) 

# History

As early as 2005, **Peter Rodgers** introduced the term "Micro-Web-Services" during
a presentation at the Web Services Edge conference.

# Spring Cloud VS Kubernetes

Spring Cloud 는 Microservices 를 위해 탄생한 framework 이다.
microservices 의 대표적인 implementation 중 Spring Cloud 와 Kubernetes 를 MSA 관점에 따라 비교해 본다.

| Concern                               | Spring Cloud                          | Kubernetes                                           |
| ------------------------------------- | ------------------------------------- | ---------------------------------------------------- |
| Configuration Management              | Spring Config Server                  | Kubernetes ConfigMaps                                |
| Service Discovery                     | Spring Cloud Eureka                   | Kubernetes Services                                  |
| Load Balancing                        | Spring Cloud Ribbon                   | Kubernetes Services                                  |
| API gateway                           | Spring Cloud Zuul                     | Kubernetes Services, Ingress                         |
| Security                              | Spring Cloud Security                 | Service meshes (ex. Istio)                           |
| Centralized logging                   | ELK (Elasticsearch, LogStash, Kibana) | EFK (Eleasticsearch, Fluentd, Kibana)                |
| Centralized metrics                   | Spring Spectator & Atlas              | Heapster, Prometheus & Grafana                       |
| Distributed tracing                   | Spring Cloud Sleuth                   | Hawkular, Jaeger                                     |
| Resilience and fault tolerance        | Spring Hystrix, Turbine & Ribbon      | Health check, Service meshes (ex. Istio)             |
| Autoscaling and self-healing          | Spring hystrix, Turbine, & Ribbon     | Health check, service meshes (ex. Istio)             |
| Packaging, deployment, and scheduling | Spring Boot, Apache Maven             | Docker, Rkt, Kubernetes Scheduler & Deployment, Helm |
| Job management                        | Spring Batch                          | Kubernetes Pods                                      |
| Singleton application                 | Spring Cloud Cluster                  | Kubernetes Pods                                      |
 
# Pattern

## Application architecture patterns

* **Monolithic architecture**
  * 서비스를 하나의 Application 으로 구현한 pattern 을 말한다. 
  * 적은 인원으로 운영할 수 있다. 그러나 서비스 확장이 쉽지 않다.
* **Microservice architecture**
  * 서비스를 여러개의 Application 으로 구현한 pattern 을 말한다.
  * 많은 인원으로 운영해야 한다. Application 별로 배포가 이루어지므로 동시 개발 및 배포가 가능하다.

## Decomposition

* Decompose by business capability
* Decompose by subdomain
* Self-contained Service
* Service per team

## Refactoring to microservices

* **Strangler Application**
  * 새로운 service 로 migration 을 위해 legacy application 을 조금씩 일부를
    개선하는 pattern 을 말한다.
* **Anti-corruption layer**
  * legacy service 와 new service 에 layer 를 두어 서비스 충돌이 발생하지 않게
    하는 pattern 을 말한다.

## Data management

* **Database per Service**
  * 하나의 application 이 하나의 DataBase 를 사용한는 pattern 을 말한다.
* **Shared database**
  * 하나의 DataBase 를 여러 application 들이 사용하는 pattern 을 말한다.
* **Saga**
  * local transaction 들을 모아서 처리하는 pattern 을 말한다.
  * **choreography-based saga**, **orchestration-based saga** 와 같이 2 가지가 있다.
  * [Distributed Transaction](/distributedtransaction/README.md) 참고
* **API Composition**
  * 하나의 request 를 다수의 request 로 나누고 결과를 합하여 response 로
    돌려주는 pattern 을 말한다.
* **CQRS**
  * 쓰기와 읽기를 분리해서 구현하는 pattern 을 말한다.
* **Domain Event**
  * 여러개의 domain service 들 event 를 주고받도록 구현한 pattern 을 말한다.
  * Domain event 는 [DDD](/domaindrivendesign/README.md) 에서 Business Logic 을
    위해 발생한 어떤 것을 말한다.
* **Event sourcing**
  * service 의 이력을 모두 event 로 만들어 저장하는 pattern 을 말한다.
  * 예를 들어 order service 의 이력을 
    **orderCreated, orderApproved, orderCanceled, orderShipped** 와 같이 모두 event 로 만들어 저장한다. 
  * service 의 흐름을 상태별로 구분해서 구현할 수 있다.

## Transactional messaging

* **[Transactional Outbox](msa_transactional_outbox.md)**
* **Transaction log tailing**
  * outbox table 을 polling 하지 않고 transaction log 를 polling 하다가 message 가 삽입되면 message broker 에 전달하는 pattern
  * DynamoDB Streams 가 해당된다.
  * Polling publisher 와 차이는???
* **Polling publisher**
  * outbox 를 polling 하다가 message 가 삽입되면 message broker 에 전달하는 pattern 을 말한다.

## Testing

* Service Component Test
* Consumer-driven contract test
* Consumer-side contract test

## Deployment patterns

* **Multiple service instances per host**
  * 하나의 host 에 여러개의 service 들을 운영하는 pattern 을 말한다.
* **Service instance per host**
  * 하나의 host 에 하나의 service 를 운영하는 pattern 을 말한다.
* **Service instance per VM**
  * 하나의 virtual machine image 에 하나의 service 를 운영하는 pattern 을 말한다.
* **Service instance per Container**
  * 하나의 docker container 에 하나의 service 를 운영하는 pattern 을 말한다.
* **Serverless deployment**
  * server 가 없이 service 를 운영할 수 있는 pattern 을 말한다.
  * AWS Lambda 와 같이 code 를 업로드하면 service 를 운영할 수 있다.
* **Service deployment platform**
  * deployment platform 을 사용하여 service 를 배포하는 pattern 을 말한다.
  * deployment platform 은 loadbalancer 를 포함한 infrastructure 를 배포한다.

## Cross cutting concerns

* **Microservice chassis**
  * cross cutting concern 을 해결할 수 있는 도구들을 모아둔 pattern 을 말한다.
  * micro service 를 설계할 때 Externalized configuration, Logging, Health checks, Metrics, Distributed tracing 과 같은 cross cutting concern 들을 경험할 수 있다. 이러한 cross cutting concern 을 해결할 framework 를 사용하면 쉽게 micro service 를 구현할 수 있다. 이러한 framework 는 Microservice chassis 패턴을 구현했다고 할 수 있다.
  * 예를 들어 Java 의 Spring Boot, Spring Cloud, Dropwizard 가 Microservice cahssis 에 해당한다. 또한 Go 의 Gizmo, Micro, Go Kit 도 역시 해당한다.
* **Externalized configuration**
  * Application 의 설정을 외부에서 읽어 들이는 pattern 을 말한다.
  * 예를 들어 Java 의 Spring Boot Application 은 Environment variables, property files, command line arguments 등의 외부 설정을 사용한다.

## Communication style

* Remote Procedure Invocation
* Messaging
* Domain-specific protocol
* Idempotent Consumer

## External API

* **API gateway**
  * 다수의 client 들이 single entrypoint 에 request 하도록 구현한 pattern 을 말한다.
  * Authentication, Authorization 을 API gateway 한 곳에서 처리하면 유지보수가 수월하다.
* **Backend for front-end**
  * front-end 를 위한 backend 를 별도로 운영하는 pattern 을 말한다.

## Service discovery

* Client-side discovery
* Server-side discovery
* Service registry
* Self registration
* 3rd party registration

## Reliability

* **Circuit Breaker**
  * request 의 실패횟수가 임계점을 넘어가면 request 를 차단하는 pattern 을 말한다.
  * 차단하고 일정시간이 지나면 다시 request 를 보낼 수 있게 한다.
  * circuit breaker 를 구현하면 traffic 을 throttling 할 수 있다.

## Security

* **Access Token**
  * token 을 사용하여 인증을 구현한 pattern 을 말한다.
  * JSON Web Token 은 대표적인 예이다.

## Observability

* **Log aggregation**
  * 여러 service 의 log 를 한곳을 모아 검색 및 알림에 사용하는 pattern 을 말한다.
* Application metrics
* **Audit logging**
  * user 의 행동을 logging 하는 pattern
* Distributed tracing
* Exception tracking
* Health check API
* Log deployments and changes

## UI patterns

* Server-side page fragment composition
* Client-side UI composition
