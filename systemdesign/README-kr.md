# 시스템 디자인 (System Design)

- [시스템 디자인 (System Design)](#시스템-디자인-system-design)
- [개요 (Abstract)](#개요-abstract)
- [참고자료 (References)](#참고자료-references)
- [학습자료 (Materials)](#학습자료-materials)
- [System Design Takeaways](#system-design-takeaways)
- [추정 계산 (Estimations)](#추정-계산-estimations)
  - [주요 항목](#주요-항목)
  - [기본 숫자](#기본-숫자)
  - [2의 거듭제곱](#2의-거듭제곱)
  - [개발자가 알아야 할 지연시간](#개발자가-알아야-할-지연시간)
  - [가용성](#가용성)
  - [80/20 규칙 (80/20 Rule)](#8020-규칙-8020-rule)
  - [70% 용량 모델 (70% Capacity Model)](#70-용량-모델-70-capacity-model)
  - [SLA, SLO, SLI](#sla-slo-sli)
    - [정의](#정의)
    - [관계도](#관계도)
    - [실전 예시](#실전-예시)
    - [SLA 위반 시 보상 정책](#sla-위반-시-보상-정책)
    - [모니터링 대시보드](#모니터링-대시보드)
  - [Time](#time)
  - [실전 예제: Twitter 규모 추정](#실전-예제-twitter-규모-추정)
- [기초 개념 (Fundamentals)](#기초-개념-fundamentals)
  - [네트워크 (Network)](#네트워크-network)
    - [Network Overview](#network-overview)
    - [HTTP/3와 QUIC](#http3와-quic)
    - [REST API](#rest-api)
    - [RPC](#rpc)
    - [gRPC](#grpc)
    - [gRPC vs REST API](#grpc-vs-rest-api)
    - [GraphQL](#graphql)
    - [Long Polling](#long-polling)
    - [Web Socket](#web-socket)
    - [Server-Sent Events (SSE)](#server-sent-events-sse)
    - [Circuit Breaker](#circuit-breaker)
    - [Rate Limiter](#rate-limiter)
    - [Asynchronism](#asynchronism)
    - [Message Queue](#message-queue)
    - [Message Queue VS Event Streaming Platform](#message-queue-vs-event-streaming-platform)
  - [위치 기반 서비스 (Location-Based Services)](#위치-기반-서비스-location-based-services)
    - [Geohashing](#geohashing)
    - [Quadtrees](#quadtrees)
  - [로드 밸런싱 (Load Balancing)](#로드-밸런싱-load-balancing)
    - [알고리즘 비교](#알고리즘-비교)
    - [Nginx 설정 예제](#nginx-설정-예제)
    - [Layer 4 vs Layer 7 로드 밸런싱](#layer-4-vs-layer-7-로드-밸런싱)
  - [캐싱 (Caching)](#캐싱-caching)
    - [캐시 무효화 전략](#캐시-무효화-전략)
    - [Cache Stampede 문제](#cache-stampede-문제)
    - [Multi-level Caching](#multi-level-caching)
  - [데이터베이스 (Database)](#데이터베이스-database)
    - [SQL vs NoSQL](#sql-vs-nosql)
      - [SQL (Structured Query Language)](#sql-structured-query-language)
      - [NoSQL (Not Only SQL)](#nosql-not-only-sql)
      - [상세 비교](#상세-비교)
      - [ACID vs BASE](#acid-vs-base)
      - [선택 가이드](#선택-가이드)
      - [하이브리드 접근 (Polyglot Persistence)](#하이브리드-접근-polyglot-persistence)
      - [마이그레이션 고려사항](#마이그레이션-고려사항)
    - [인덱싱 전략](#인덱싱-전략)
    - [Replication Lag 처리](#replication-lag-처리)
    - [Database Connection Pooling](#database-connection-pooling)
    - [Bloom Filter](#bloom-filter)
  - [분산 시스템 (Distributed Systems)](#분산-시스템-distributed-systems)
    - [CAP Theorem (일관성, 가용성, 분할 허용)](#cap-theorem-일관성-가용성-분할-허용)
    - [PACELC Theorem](#pacelc-theorem)
    - [DNS (Domain Name System)](#dns-domain-name-system)
    - [CDN (Content Delivery Network)](#cdn-content-delivery-network)
    - [Saga Pattern 상세](#saga-pattern-상세)
    - [Outbox Pattern](#outbox-pattern)
    - [Distributed Primary Key](#distributed-primary-key)
    - [Consistent Hashing](#consistent-hashing)
    - [합의 알고리즘 (Consensus Algorithm)](#합의-알고리즘-consensus-algorithm)
    - [Paxos](#paxos)
    - [Gossip Protocol](#gossip-protocol)
    - [Raft](#raft)
    - [Chubby](#chubby)
    - [Distributed Locking](#distributed-locking)
      - [Redis Redlock의 한계](#redis-redlock의-한계)
      - [Redis Redisson 구현 예제](#redis-redisson-구현-예제)
      - [ZooKeeper를 사용한 안전한 분산 락](#zookeeper를-사용한-안전한-분산-락)
      - [ZooKeeper Curator 구현 예제](#zookeeper-curator-구현-예제)
    - [Distributed Tracing](#distributed-tracing)
      - [개요](#개요)
      - [필요성](#필요성)
  - [낙관적 락 vs 비관적 락 (Optimistic Lock vs Pessimistic Lock)](#낙관적-락-vs-비관적-락-optimistic-lock-vs-pessimistic-lock)
    - [개념](#개념)
    - [비교](#비교)
    - [낙관적 락 (Optimistic Lock)](#낙관적-락-optimistic-lock)
    - [비관적 락 (Pessimistic Lock)](#비관적-락-pessimistic-lock)
    - [실전 시나리오](#실전-시나리오)
    - [선택 가이드](#선택-가이드-1)
  - [분산 트랜잭션 (Distributed Transaction)](#분산-트랜잭션-distributed-transaction)
    - [개념](#개념-1)
    - [ACID vs BASE](#acid-vs-base-1)
    - [분산 트랜잭션 패턴](#분산-트랜잭션-패턴)
    - [패턴 비교](#패턴-비교)
    - [보상 트랜잭션 설계 원칙](#보상-트랜잭션-설계-원칙)
    - [분산 트랜잭션 모니터링](#분산-트랜잭션-모니터링)
    - [설계 체크리스트](#설계-체크리스트)
  - [멱등성 (Idempotency)](#멱등성-idempotency)
    - [개념](#개념-2)
    - [HTTP 메서드 멱등성](#http-메서드-멱등성)
    - [멱등성 키 (Idempotency Key) 패턴](#멱등성-키-idempotency-key-패턴)
    - [멱등성 보장 전략](#멱등성-보장-전략)
    - [실전 시나리오](#실전-시나리오-1)
    - [멱등성 구현 시 고려사항](#멱등성-구현-시-고려사항)
  - [메시지 큐 (Message Queue)](#메시지-큐-message-queue)
    - [전송 보장 방식](#전송-보장-방식)
    - [Dead Letter Queue](#dead-letter-queue)
  - [Observability](#observability)
  - [Load Test](#load-test)
  - [Incidenct](#incidenct)
  - [재해 복구 (Disaster Recovery, DR)](#재해-복구-disaster-recovery-dr)
    - [개념](#개념-3)
    - [핵심 지표](#핵심-지표)
    - [DR 전략 비교](#dr-전략-비교)
    - [DR 전략 구현](#dr-전략-구현)
    - [DR 테스트 및 검증](#dr-테스트-및-검증)
    - [DR 체크리스트](#dr-체크리스트)
    - [비용 최적화 전략](#비용-최적화-전략)
    - [주요 클라우드 DR 서비스](#주요-클라우드-dr-서비스)
  - [구성 관리 데이터베이스 (Configuration Management Database, CMDB)](#구성-관리-데이터베이스-configuration-management-database-cmdb)
  - [보안 (Security)](#보안-security)
    - [Security Overview](#security-overview)
    - [Web Application Firewall (WAF)](#web-application-firewall-waf)
    - [Cross Site Scripting (XSS)](#cross-site-scripting-xss)
    - [Cross Site Request Forgery (CSRF)](#cross-site-request-forgery-csrf)
    - [XSS vs CSRF](#xss-vs-csrf)
    - [Cross Origin Resource Sharing (CORS)](#cross-origin-resource-sharing-cors)
    - [SSL/TLS](#ssltls)
    - [JWT 상세 구조](#jwt-상세-구조)
    - [OAuth 2.0 Flow](#oauth-20-flow)
    - [mTLS (Mutual TLS)](#mtls-mutual-tls)
    - [OpenID Connect (OIDC)](#openid-connect-oidc)
    - [OIDC 구현 예제](#oidc-구현-예제)
    - [Single Sign-On (SSO)](#single-sign-on-sso)
    - [SSO 구현 예제](#sso-구현-예제)
    - [API 보안 (API Security)](#api-보안-api-security)
      - [API 보안 핵심 구성 요소](#api-보안-핵심-구성-요소)
      - [1. HTTPS 사용](#1-https-사용)
      - [2. OAuth2 사용](#2-oauth2-사용)
      - [3. WebAuthn 사용](#3-webauthn-사용)
      - [4. 레벨별 API 키 사용](#4-레벨별-api-키-사용)
      - [5. 인가 (Authorization)](#5-인가-authorization)
      - [6. Rate Limiting (속도 제한)](#6-rate-limiting-속도-제한)
      - [7. API 버전 관리](#7-api-버전-관리)
      - [8. 화이트리스트 (Whitelisting)](#8-화이트리스트-whitelisting)
      - [9. OWASP API 보안 위험 점검](#9-owasp-api-보안-위험-점검)
      - [10. API Gateway 사용](#10-api-gateway-사용)
      - [11. 오류 처리 (Error Handling)](#11-오류-처리-error-handling)
      - [12. 입력 검증 (Input Validation)](#12-입력-검증-input-validation)
      - [API 보안 체크리스트](#api-보안-체크리스트)
      - [모니터링 및 감사](#모니터링-및-감사)
      - [결론](#결론)
  - [Big Data](#big-data)
    - [Hadoop](#hadoop)
      - [Hadoop 핵심 구성요소](#hadoop-핵심-구성요소)
      - [Hadoop 생태계](#hadoop-생태계)
      - [실전 사용 사례](#실전-사용-사례)
      - [Hadoop 장단점](#hadoop-장단점)
    - [MapReduce](#mapreduce)
      - [MapReduce 두 단계](#mapreduce-두-단계)
      - [상세 처리 흐름](#상세-처리-흐름)
      - [MapReduce 자동 처리 기능](#mapreduce-자동-처리-기능)
      - [실전 예제](#실전-예제)
      - [MapReduce 장단점](#mapreduce-장단점)
      - [MapReduce vs Spark](#mapreduce-vs-spark)
      - [언제 MapReduce를 사용할까?](#언제-mapreduce를-사용할까)
    - [Data Lake (데이터 레이크)](#data-lake-데이터-레이크)
      - [주요 장점](#주요-장점)
      - [한계점](#한계점)
    - [Data Warehouse (데이터 웨어하우스)](#data-warehouse-데이터-웨어하우스)
      - [주요 특징](#주요-특징)
      - [일반적인 사용 패턴](#일반적인-사용-패턴)
    - [Data Lakehouse (데이터 레이크하우스)](#data-lakehouse-데이터-레이크하우스)
      - [핵심 목표](#핵심-목표)
      - [주요 기능](#주요-기능)
      - [오픈소스 기술](#오픈소스-기술)
      - [주요 벤더](#주요-벤더)
      - [장점](#장점)
      - [사용 사례](#사용-사례)
      - [아키텍처 비교](#아키텍처-비교)
      - [결론](#결론-1)
  - [A/B 테스트 (A/B Testing)](#ab-테스트-ab-testing)
  - [Actor Model](#actor-model)
  - [Reactor vs Proactor](#reactor-vs-proactor)
  - [배치 처리 vs 스트림 처리 (Batch Processing vs Stream Processing)](#배치-처리-vs-스트림-처리-batch-processing-vs-stream-processing)
    - [아키텍처 비교](#아키텍처-비교-1)
    - [1. 데이터 처리 방식](#1-데이터-처리-방식)
      - [배치 처리 (Batch Processing)](#배치-처리-batch-processing)
      - [스트림 처리 (Stream Processing)](#스트림-처리-stream-processing)
    - [2. 지연 시간 (Latency)](#2-지연-시간-latency)
    - [3. 사용 사례](#3-사용-사례)
      - [배치 처리 사용 사례](#배치-처리-사용-사례)
      - [스트림 처리 사용 사례](#스트림-처리-사용-사례)
    - [4. 장애 허용성 및 신뢰성](#4-장애-허용성-및-신뢰성)
      - [배치 처리](#배치-처리)
      - [스트림 처리](#스트림-처리)
    - [5. 확장성 및 성능](#5-확장성-및-성능)
    - [6. 복잡성](#6-복잡성)
      - [배치 처리](#배치-처리-1)
      - [스트림 처리](#스트림-처리-1)
    - [7. 도구 및 플랫폼](#7-도구-및-플랫폼)
      - [배치 처리 도구](#배치-처리-도구)
      - [스트림 처리 도구](#스트림-처리-도구)
    - [8. 하이브리드 접근: Lambda \& Kappa 아키텍처](#8-하이브리드-접근-lambda--kappa-아키텍처)
      - [Lambda 아키텍처](#lambda-아키텍처)
      - [Kappa 아키텍처 (단순화)](#kappa-아키텍처-단순화)
    - [비교 요약](#비교-요약)
    - [선택 가이드](#선택-가이드-2)
  - [Checksum](#checksum)
    - [개념](#개념-4)
    - [사용 목적](#사용-목적)
    - [체크섬 계산기](#체크섬-계산기)
  - [MSA (Micro Service Architecture)](#msa-micro-service-architecture)
  - [Cloud Design Patterns](#cloud-design-patterns)
  - [Enterprise Integration Patterns](#enterprise-integration-patterns)
  - [DDD](#ddd)
  - [Architecture](#architecture)
- [최신 기술 트렌드](#최신-기술-트렌드)
  - [Serverless Architecture](#serverless-architecture)
  - [Container Orchestration - Kubernetes](#container-orchestration---kubernetes)
  - [Control Plane vs Data Plane vs Management Plane](#control-plane-vs-data-plane-vs-management-plane)
    - [개념](#개념-5)
    - [상세 설명](#상세-설명)
    - [아키텍처 다이어그램](#아키텍처-다이어그램)
    - [실제 예시](#실제-예시)
    - [평면 간 통신](#평면-간-통신)
    - [평면 분리의 이점](#평면-분리의-이점)
    - [설계 고려사항](#설계-고려사항)
  - [Service Mesh](#service-mesh)
  - [Edge Computing](#edge-computing)
- [시스템 디자인 인터뷰](#시스템-디자인-인터뷰)
  - [인터뷰 진행 방법](#인터뷰-진행-방법)
  - [난이도별 문제](#난이도별-문제)
    - [Easy](#easy)
    - [Medium](#medium)
    - [Hard](#hard)
- [실제 시스템 아키텍처 사례](#실제-시스템-아키텍처-사례)
  - [주요 기업 아키텍처](#주요-기업-아키텍처)
    - [Twitter](#twitter)
    - [Instagram](#instagram)
    - [Netflix](#netflix)
    - [Uber](#uber)
- [추가 학습 리소스](#추가-학습-리소스)
  - [데이터베이스 특화](#데이터베이스-특화)
    - [Database Sharding 전략](#database-sharding-전략)
  - [SQL vs NoSQL 선택 가이드](#sql-vs-nosql-선택-가이드)
  - [성능 최적화 체크리스트](#성능-최적화-체크리스트)
    - [Backend 최적화](#backend-최적화)
    - [Frontend 최적화](#frontend-최적화)
    - [Infrastructure 최적화](#infrastructure-최적화)
  - [추가 학습 자료](#추가-학습-자료)
- [System Design Interview](#system-design-interview)
  - [Easy](#easy-1)
  - [Medium](#medium-1)
  - [Hard](#hard-1)
- [Scalability Articles](#scalability-articles)
- [Real World Architecture](#real-world-architecture)
- [Company Architectures](#company-architectures)
- [Company Engineering Blog](#company-engineering-blog)

----

# 개요 (Abstract)

시스템 디자인은 대규모 분산 시스템을 설계하고 구축하는 방법론입니다. 이 문서는 시스템 디자인의 핵심 개념부터 실전 예제까지 포괄적으로 다룹니다.

<p align="center">
  <img src="http://i.imgur.com/jj3A5N8.png"/>
  <br/>
</p>

# 참고자료 (References)

* [systemdesignone archive](https://newsletter.systemdesign.one/archive)
* [system-design-101 | github](https://github.com/ByteByteGoHq/system-design-101) - 한 페이지로 보는 시스템 디자인
* [System Design Interview – An insider's guide by Alex](https://bytebytego.com/courses/system-design-interview/)
* [Design Microservices Architecture with Patterns & Principles | udemy](https://www.udemy.com/course/design-microservices-architecture-with-patterns-principles/)

# 학습자료 (Materials)

* [System Design Roadmap | roadmap.sh](https://roadmap.sh/system-design)
* [Awesome System Design Resources | github](https://github.com/ashishps1/awesome-system-design-resources)
* [DreamOfTheRedChamber/system-design @ github](https://github.com/DreamOfTheRedChamber/system-design)

# System Design Takeaways

- [System Design Template](https://systemdesign.one/system-design-interview-cheatsheet/#system-design-template)

| Category          | Topic                        | Sub-Topic                    |
| ----------------- | ---------------------------- | ---------------------------- |
| **Requirements**  | Functional Requirements      |                              |
|                   | Non-Functional Requirements  |                              |
|                   | Daily Active Users           |                              |
|                   | Read-to-Write Ratio          |                              |
|                   | Usage Patterns               |                              |
|                   | Peak and Seasonal Events     |                              |
| **Database**      | Data Model                   |                              |
|                   | Entity Relationship Diagram  |                              |
|                   | SQL                          |                              |
|                   | Type of Database             |                              |
| **API Design**    | HTTP Verb                    |                              |
|                   | Request-Response Headers     |                              |
|                   | Request-Response Contract    |                              |
|                   | Data format                  | JSON                         |
|                   |                              | XML                          |
|                   |                              | Protocol Buffer              |
| **Capacity Planning** | Query Per Second (Read-Write) |                        |
|                   | Bandwidth (Read-Write)       |                              |
|                   | Storage                      |                              |
|                   | Memory                       | Cache (80-20 Rule)           |
| **High Level Design** | Basic Algorithm          |                              |
|                   | Data Flow                    | Read-Write Scenario          |
|                   | Tradeoffs                    |                              |
|                   | Alternatives                 |                              |
|                   | Network Protocols            | TCP                          |
|                   |                              | UDP                          |
|                   |                              | REST                         |
|                   |                              | RPC                          |
|                   |                              | WebSocket                    |
|                   |                              | SSE                          |
|                   |                              | Long Polling                 |
|                   | Cloud Patterns               | CQRS                         |
|                   |                              | Publish-Subscribe            |
|                   | Serverless Functions         |                              |
|                   | Data Structures              | CRDT (conflict-free replicated data type) |
|                   |                              | Trie                         |
| **Design Deep Dive** | Single Point of Failures  |                              |
|                   | Bottlenecks (Hot spots)      |                              |
|                   | Concurrency                  |                              |
|                   | Distributed Transactions     | Two-Phase Commit             |
|                   |                              | Sagas                        |
|                   | Probabilistic Data Structures | Bloom Filter                |
|                   |                              | HyperLogLog                  |
|                   |                              | Count-Min Sketch             |
|                   | Coordination Service         | Zookeeper                    |
|                   | Logging                      |                              |
|                   | Monitoring                   |                              |
|                   | Alerting                     |                              |
|                   | Tracing                      |                              |
|                   | Deployment                   |                              |
|                   | Security                     | Authorization                |
|                   |                              | Authentication               |
|                   | Consensus Algorithms         | Raft                         |
|                   |                              | Paxos                        |
| **Components**    | DNS                          |                              |
|                   | CDN                          |                              |
|                   | Load Balancer                | Push-Pull                    |
|                   |                              | Layer 4-7                    |
|                   | Reverse Proxy                |                              |
|                   | Application Layer            | Microservice-Monolith        |
|                   |                              | Service Discovery            |
|                   |                              | Leader-Follower              |
|                   |                              | Leader-Leader                |
|                   | SQL Data Store               | Indexing                     |
|                   |                              | Federation                   |
|                   |                              | Sharding                     |
|                   |                              | Denormalization              |
|                   |                              | SQL Tuning                   |
|                   | NoSQL Data Store             | Graph                        |
|                   |                              | Document                     |
|                   |                              | Key-Value                    |
|                   |                              | Wide-Column                  |
|                   | Message Queue                |                              |
|                   | Task Queue                   |                              |
|                   | Cache                        | Query-Object Level           |
|                   |                              | Client                       |
|                   |                              | CDN                          |
|                   |                              | Webserver                    |
|                   |                              | Database                     |
|                   |                              | Application                  |
|                   | Cache Update Pattern         | Cache Aside                  |
|                   |                              | Read Through                 |
|                   |                              | Write Through                |
|                   |                              | Write Behind                 |
|                   |                              | Refresh Ahead                |
|                   | Cache Eviction Policy        | LRU                          |
|                   |                              | LFU                          |
|                   |                              | FIFO                         |
|                   | Clocks                       | Physical clock               |
|                   |                              | Lamport clock (logical)      |
|                   |                              | Vector clock                 |

# 추정 계산 (Estimations)

## 주요 항목

시스템 설계 시 추정해야 할 주요 항목들:
- QPS (Queries Per Second)
- Peak QPS
- Storage 용량
- Cache 크기
- 필요한 서버 대수
- Network Bandwidth

## 기본 숫자

| Value | Short-scale | 한글   | SI-symbol | SI-prefix |
| ----- | ----------- | ------ | --------- | --------- |
| 10^3  | Thousand    | 천     | K         | Kilo-     |
| 10^6  | Million     | 백만   | M         | Mega-     |
| 10^9  | Billion     | 십억   | G         | Giga-     |
| 10^12 | Trillion    | 조     | T         | Tera-     |
| 10^15 | Quadrillion | 천조   | P         | Peta-     |

## 2의 거듭제곱

| Power | Approximate value | Full name | Short name |
|-------|-------------------|-----------|------------|
| 10    | 1 Thousand        | 1 Kilobyte | 1 KB      |
| 20    | 1 Million         | 1 Megabyte | 1 MB      |
| 30    | 1 Billion         | 1 Gigabyte | 1 GB      |
| 40    | 1 Trillion        | 1 Terabyte | 1 TB      |
| 50    | 1 Quadrillion     | 1 Petabyte | 1 PB      |

## 개발자가 알아야 할 지연시간

| Operation | Time | |
|-----------|------|--|
| L1 cache reference | 0.5 ns | |
| Branch mispredict | 5 ns | |
| L2 cache reference | 7 ns | 14x L1 cache |
| Mutex lock/unlock | 100 ns | |
| Main memory reference | 100 ns | 20x L2 cache, 200x L1 cache |
| Compress 1K bytes with Zippy | 10,000 ns = 10 us | |
| Send 1 KB over 1 Gbps network | 20,000 ns = 20 us | |
| Read 1 MB sequentially from memory | 250,000 ns = 250 us | |
| Round trip within same datacenter | 500,000 ns = 500 us | |
| Disk seek | 10,000,000 ns = 10 ms | 20x datacenter roundtrip |
| Read 1 MB sequentially from disk | 30,000,000 ns = 30 ms | |
| Send packet CA->Netherlands->CA | 150,000,000 ns = 150 ms | |

```
Notes
-----
1 ns = 10^-9 seconds
1 us = 10^-6 seconds = 1,000 ns
1 ms = 10^-3 seconds = 1,000 us = 1,000,000 ns
```

## 가용성

| Availability % | Downtime per year | Downtime per month | Downtime per week |
|----------------|-------------------|-------------------|-------------------|
| 99%            | 3.65 days         | 7.20 hours        | 1.68 hours        |
| 99.9%          | 8.77 hours        | 43.8 minutes      | 10.1 minutes      |
| 99.99%         | 52.6 minutes      | 4.32 minutes      | 1.01 minutes      |
| 99.999%        | 5.26 minutes      | 25.9 seconds      | 6.05 seconds      |

## 80/20 규칙 (80/20 Rule)

전체 데이터의 20%만 자주 사용된다는 규칙입니다. 주로 캐시 데이터 크기를 추정할 때 활용됩니다.

**적용 예시**:
- 전체 데이터 크기가 100GB라면 캐시 데이터 크기는 20GB로 예측
- 전체 트래픽의 80%가 20%의 인기 콘텐츠에 집중
- 100만 개의 URL 중 20만 개가 대부분의 요청 처리

**실전 활용**:
```
총 데이터 크기: 100 GB
캐시 크기 (20%): 20 GB
캐시 적중률 목표: 80%

일일 저장 데이터: 1 TB
Hot data (캐시): 200 GB
Cold data (영구 저장): 800 GB
```

**적용 분야**:
- CDN 캐싱 전략: 인기 콘텐츠 우선 캐싱
- 데이터베이스 쿼리 캐싱: 자주 조회되는 쿼리 결과 캐싱
- 상품 재고 관리: 인기 상품 20%에 재고 집중
- API Rate Limiting: 상위 20% 사용자에 대한 별도 정책

## 70% 용량 모델 (70% Capacity Model)

예상 데이터 크기는 전체 데이터 크기의 70%라는 규칙입니다. 시스템 용량 계획 시 안전 마진을 확보하기 위해 사용됩니다.

**기본 공식**:
```
total data size : estimated data size = 100 : 70
```

**적용 예시**:
- 예상 데이터 크기가 70GB이면 전체 데이터 크기는 100GB로 계획
- 70% 이상 사용 시 확장 알림 설정
- 30%의 버퍼로 급격한 트래픽 증가 대응

**실전 계산**:
```java
public class CapacityPlanner {

    private static final double CAPACITY_THRESHOLD = 0.7;

    public long calculateTotalCapacity(long estimatedDataSize) {
        // 예상 데이터 크기가 70%가 되도록 전체 용량 계산
        return (long) (estimatedDataSize / CAPACITY_THRESHOLD);
    }

    public boolean needsScaling(long currentUsage, long totalCapacity) {
        double utilizationRate = (double) currentUsage / totalCapacity;
        return utilizationRate >= CAPACITY_THRESHOLD;
    }

    public static void main(String[] args) {
        CapacityPlanner planner = new CapacityPlanner();

        // 예상 사용량: 70GB
        long estimatedSize = 70L * 1024 * 1024 * 1024; // bytes
        long totalCapacity = planner.calculateTotalCapacity(estimatedSize);

        System.out.printf("예상 데이터: %d GB%n", estimatedSize / (1024*1024*1024));
        System.out.printf("필요 전체 용량: %d GB%n", totalCapacity / (1024*1024*1024));
        // 출력: 100 GB

        // 현재 사용량 확인
        long currentUsage = 72L * 1024 * 1024 * 1024;
        boolean needsScaling = planner.needsScaling(currentUsage, totalCapacity);

        System.out.printf("현재 사용량: %d GB (%.1f%%)%n",
            currentUsage / (1024*1024*1024),
            (double) currentUsage / totalCapacity * 100);
        System.out.printf("스케일링 필요: %s%n", needsScaling ? "예" : "아니오");
    }
}
```

**모니터링 전략**:
```java
@Service
public class CapacityMonitoringService {

    @Autowired
    private MetricsCollector metricsCollector;

    @Scheduled(cron = "0 */5 * * * *") // 5분마다 실행
    public void monitorCapacity() {
        long totalCapacity = getTotalCapacity();
        long currentUsage = getCurrentUsage();
        double utilizationRate = (double) currentUsage / totalCapacity;

        metricsCollector.recordGauge("system.capacity.utilization", utilizationRate);

        // 70% 임계값 초과 시 알림
        if (utilizationRate >= 0.7) {
            alertService.sendAlert(
                AlertLevel.WARNING,
                String.format("용량 사용률 %.1f%% - 스케일링 고려 필요", utilizationRate * 100)
            );
        }

        // 85% 초과 시 긴급 알림
        if (utilizationRate >= 0.85) {
            alertService.sendAlert(
                AlertLevel.CRITICAL,
                String.format("용량 사용률 %.1f%% - 즉시 스케일링 필요", utilizationRate * 100)
            );
        }
    }
}
```

**적용 시나리오**:
| 시나리오 | 예상 크기 | 전체 용량 | 확장 임계값 |
|---------|----------|----------|-----------|
| 데이터베이스 스토리지 | 700 GB | 1 TB | 700 GB (70%) |
| 메모리 캐시 | 14 GB | 20 GB | 14 GB (70%) |
| 디스크 I/O | 7K IOPS | 10K IOPS | 7K IOPS (70%) |
| 네트워크 대역폭 | 700 Mbps | 1 Gbps | 700 Mbps (70%) |

## SLA, SLO, SLI

> * [The Difference between SLI, SLO, and SLA](https://enqueuezero.com/the-difference-between-sli-slo-and-sla.html)
> * [Uptime Calculation Web](https://uptime.is/)

서비스 수준 관리를 위한 세 가지 핵심 개념입니다.

### 정의

**SLA (Service Level Agreement - 서비스 수준 협약)**
- 서비스 제공자가 고객에게 약속하는 공식적인 계약
- 서비스 가용성, 성능, 응답 시간 등에 대한 보장
- 위반 시 보상 조건 명시

**SLO (Service Level Objective - 서비스 수준 목표)**
- 서비스 제공자가 달성하고자 하는 내부 목표
- SLA보다 더 엄격한 기준 설정 (안전 마진 확보)
- 팀의 운영 목표와 우선순위 결정

**SLI (Service Level Indicator - 서비스 수준 지표)**
- 서비스 품질을 측정하는 구체적인 지표
- SLO 달성 여부를 판단하는 데이터
- 실제 시스템에서 수집되는 메트릭

### 관계도

```
┌─────────────────────────────────────────┐
│ SLA (고객 약속)                          │
│ 예: 99.9% 가용성 보장                    │
└──────────────┬──────────────────────────┘
               │
               │ 기준
               ↓
┌─────────────────────────────────────────┐
│ SLO (내부 목표)                          │
│ 예: 99.95% 가용성 목표 (안전 마진)       │
└──────────────┬──────────────────────────┘
               │
               │ 측정
               ↓
┌─────────────────────────────────────────┐
│ SLI (실제 측정값)                        │
│ 예: 지난 30일간 99.97% 가용성 달성       │
└─────────────────────────────────────────┘
```

### 실전 예시

**1. API 서비스**

```java
public class ServiceLevelMetrics {

    // SLI: 실제 측정 지표
    private double actualAvailability;      // 99.97%
    private long averageResponseTimeMs;     // 45ms
    private double errorRate;               // 0.05%

    // SLO: 내부 목표
    private static final double TARGET_AVAILABILITY = 0.9995;     // 99.95%
    private static final long TARGET_RESPONSE_TIME_MS = 50;       // 50ms
    private static final double TARGET_ERROR_RATE = 0.001;        // 0.1%

    // SLA: 고객 약속
    private static final double SLA_AVAILABILITY = 0.999;          // 99.9%
    private static final long SLA_RESPONSE_TIME_MS = 100;         // 100ms
    private static final double SLA_ERROR_RATE = 0.005;           // 0.5%

    public ServiceHealthStatus checkServiceHealth() {
        // SLI 수집
        this.actualAvailability = calculateAvailability();
        this.averageResponseTimeMs = calculateAverageResponseTime();
        this.errorRate = calculateErrorRate();

        // SLO 달성 여부 확인
        boolean meetsSLO = checkSLO();

        // SLA 준수 여부 확인
        boolean meetsSLA = checkSLA();

        return new ServiceHealthStatus(
            actualAvailability,
            averageResponseTimeMs,
            errorRate,
            meetsSLO,
            meetsSLA
        );
    }

    private boolean checkSLO() {
        return actualAvailability >= TARGET_AVAILABILITY
            && averageResponseTimeMs <= TARGET_RESPONSE_TIME_MS
            && errorRate <= TARGET_ERROR_RATE;
    }

    private boolean checkSLA() {
        return actualAvailability >= SLA_AVAILABILITY
            && averageResponseTimeMs <= SLA_RESPONSE_TIME_MS
            && errorRate <= SLA_ERROR_RATE;
    }
}
```

**2. 데이터베이스 서비스**

| 구분 | 가용성 | 응답 시간 (P95) | 처리량 |
|------|--------|----------------|--------|
| **SLI (측정값)** | 99.98% | 8ms | 12,000 QPS |
| **SLO (목표)** | 99.95% | 10ms | 10,000 QPS |
| **SLA (약속)** | 99.9% | 20ms | 8,000 QPS |

**3. Error Budget (오류 예산)**

```java
@Service
public class ErrorBudgetCalculator {

    // SLO 목표: 99.95% 가용성
    private static final double SLO_TARGET = 0.9995;

    public ErrorBudget calculateMonthlyBudget() {
        // 한 달 = 30일 = 43,200분
        long totalMinutesInMonth = 30L * 24 * 60;

        // 허용 가능한 다운타임 = (1 - SLO) * 전체 시간
        double allowedDowntimeMinutes = (1 - SLO_TARGET) * totalMinutesInMonth;

        return new ErrorBudget(
            totalMinutesInMonth,
            allowedDowntimeMinutes,
            SLO_TARGET
        );
    }

    public ErrorBudgetStatus getCurrentStatus(long actualDowntimeMinutes) {
        ErrorBudget budget = calculateMonthlyBudget();

        double budgetUsedPercentage =
            (double) actualDowntimeMinutes / budget.getAllowedDowntimeMinutes() * 100;

        boolean budgetExhausted = actualDowntimeMinutes >= budget.getAllowedDowntimeMinutes();

        return new ErrorBudgetStatus(
            actualDowntimeMinutes,
            budget.getAllowedDowntimeMinutes(),
            budgetUsedPercentage,
            budgetExhausted
        );
    }
}

@Data
class ErrorBudget {
    private final long totalMinutes;              // 43,200분
    private final double allowedDowntimeMinutes;  // 21.6분 (0.05%)
    private final double sloTarget;               // 99.95%
}

// 사용 예시
public class ErrorBudgetExample {
    public static void main(String[] args) {
        ErrorBudgetCalculator calculator = new ErrorBudgetCalculator();

        // 이번 달 오류 예산 계산
        ErrorBudget budget = calculator.calculateMonthlyBudget();
        System.out.printf("월간 허용 다운타임: %.1f분%n",
            budget.getAllowedDowntimeMinutes()); // 21.6분

        // 현재 상태 확인 (15분 다운타임 발생)
        ErrorBudgetStatus status = calculator.getCurrentStatus(15L);
        System.out.printf("예산 사용률: %.1f%%%n",
            status.getBudgetUsedPercentage()); // 69.4%
        System.out.printf("예산 소진 여부: %s%n",
            status.isBudgetExhausted() ? "예" : "아니오"); // 아니오

        // 30분 다운타임 발생 시
        ErrorBudgetStatus exceeded = calculator.getCurrentStatus(30L);
        System.out.printf("예산 초과: %.1f분%n",
            30 - budget.getAllowedDowntimeMinutes()); // 8.4분 초과
    }
}
```

### SLA 위반 시 보상 정책

```java
public class SLACompensationCalculator {

    // SLA 위반에 따른 보상 정책
    public Compensation calculateCompensation(double actualAvailability, double monthlyCost) {
        // SLA 99.9% 기준
        if (actualAvailability >= 0.999) {
            return new Compensation(0, "SLA 충족");
        }
        // 99.0% ~ 99.9%: 10% 환불
        else if (actualAvailability >= 0.99) {
            return new Compensation(monthlyCost * 0.1, "10% 크레딧");
        }
        // 95.0% ~ 99.0%: 25% 환불
        else if (actualAvailability >= 0.95) {
            return new Compensation(monthlyCost * 0.25, "25% 크레딧");
        }
        // 95% 미만: 50% 환불
        else {
            return new Compensation(monthlyCost * 0.5, "50% 크레딧");
        }
    }
}
```

### 모니터링 대시보드

```java
@RestController
@RequestMapping("/api/sli")
public class SLIController {

    @Autowired
    private MetricsService metricsService;

    @GetMapping("/dashboard")
    public SLIDashboard getSLIDashboard() {
        // 지난 30일간 SLI 데이터 수집
        return SLIDashboard.builder()
            .availability(metricsService.getAvailability(30))
            .p50ResponseTime(metricsService.getPercentile(50, 30))
            .p95ResponseTime(metricsService.getPercentile(95, 30))
            .p99ResponseTime(metricsService.getPercentile(99, 30))
            .errorRate(metricsService.getErrorRate(30))
            .throughput(metricsService.getThroughput(30))
            .sloCompliance(metricsService.checkSLOCompliance())
            .errorBudgetRemaining(metricsService.getRemainingErrorBudget())
            .build();
    }
}
```

**핵심 포인트**:
1. **SLA ⊇ SLO ⊇ SLI**: SLA는 가장 느슨하고, SLO는 중간, SLI는 실제 측정값
2. **안전 마진**: SLO를 SLA보다 엄격하게 설정하여 버퍼 확보
3. **측정 가능성**: 모든 지표는 자동으로 측정 가능해야 함
4. **Error Budget**: SLO 미달성 시 새 기능 개발보다 안정성 개선 우선

## Time

| years | days | hours |    mins |       secs |
| ----: | ---: | ----: | ------: | ---------: |
|     1 |  365 | 8,760 | 525,600 | 31,536,000 |
|       |    1 |    24 |   1,440 |     86,400 |
|       |      |     1 |      60 |      3,600 |
|       |      |       |       1 |         60 |

## 실전 예제: Twitter 규모 추정

**요구사항:**
- 3억 명의 월간 활성 사용자 (MAU)
- 50%가 매일 사용 (DAU = 1.5억)
- 평균적으로 사용자당 하루 2개의 트윗
- 10%의 트윗에 미디어 포함
- 데이터는 5년간 보관

**계산:**

1. **QPS 추정**
```
일일 트윗 수 = 1.5억 * 2 = 3억 트윗/일
평균 QPS = 3억 / 86,400초 ≈ 3,500 QPS
피크 QPS = 평균 QPS * 2 = 7,000 QPS
```

2. **저장 공간 추정**
```
트윗당 평균 크기:
- tweet_id: 64 bytes
- text: 140 bytes
- media: 1 MB (10%만 해당)

텍스트 저장 공간/일 = 3억 * (64 + 140) bytes = 61.2 GB/일
미디어 저장 공간/일 = 3억 * 10% * 1 MB = 30 TB/일

5년간 총 저장 공간:
- 텍스트: 61.2 GB * 365 * 5 ≈ 112 TB
- 미디어: 30 TB * 365 * 5 ≈ 55 PB
```

3. **캐시 메모리 추정 (80-20 규칙)**
```
일일 트윗 중 20%가 80%의 트래픽 생성
캐시 필요 용량 = 61.2 GB * 0.2 ≈ 12 GB
```

4. **대역폭 추정**
```
유입(Write) 대역폭:
- 텍스트: 61.2 GB / 86,400초 ≈ 0.7 MB/s
- 미디어: 30 TB / 86,400초 ≈ 347 MB/s
- 총: ≈ 348 MB/s

유출(Read) 대역폭 (읽기:쓰기 = 10:1 가정):
- 총: 348 MB/s * 10 = 3.48 GB/s
```

**실전 예제 2: URL Shortener 규모 추정**

**요구사항:**
- 하루 100M 개의 URL 생성
- Read:Write 비율 = 100:1
- 단축 URL은 5년간 보관
- 평균 URL 길이: 100 bytes

**계산:**

1. **QPS**
```
Write QPS = 100M / 86,400 ≈ 1,160 QPS
Read QPS = 1,160 * 100 = 116,000 QPS
```

2. **저장 공간**
```
일일 저장 = 100M * 100 bytes = 10 GB/일
5년 저장 = 10 GB * 365 * 5 = 18.25 TB
```

3. **캐시**
```
80-20 규칙 적용
캐시 크기 = 10 GB * 0.2 = 2 GB/일
```

# 기초 개념 (Fundamentals)

## 네트워크 (Network)

### Network Overview

- [Network](/network/README.md)
- [ipv4](/network/README.md#ipv4)
- [ipv6](/network/README.md#ipv6)
- [OSI 7 layer](/network/README.md#osi-7-layer)
- [TCP](/network/README.md#tcp)
- [UDP](/network/README.md#udp)
- [RestApi](/restapi/README.md)
- [HTTP](/HTTP/README.md)
- [HTTP Flow](/HTTP/README.md#http-flow)
- [Long-Polling vs WebSockets vs Server-Sent Events](fundamentals/Long-PollingvsWebSocketsvsServer-SentEvents.md)

### HTTP/3와 QUIC

**HTTP/3의 주요 특징:**
- UDP 기반의 QUIC 프로토콜 사용
- TCP의 Head-of-line blocking 문제 해결
- 0-RTT 연결 재개 (더 빠른 재연결)
- 내장된 암호화 (TLS 1.3)
- 연결 마이그레이션 지원 (IP 변경 시에도 연결 유지)

**QUIC의 장점:**
- 멀티플렉싱: 여러 스트림이 독립적으로 동작
- 패킷 손실 시 해당 스트림만 영향 받음
- 연결 설정 시간 단축

**사용 사례:**
- 모바일 네트워크 환경
- 실시간 스트리밍
- 고지연 네트워크

```python
# HTTP/3 연결 예제 (Python)
import httpx

async def fetch_with_http3(url):
    async with httpx.AsyncClient(http2=True) as client:
        response = await client.get(url)
        return response.text
```

### REST API

**REST (REpresentational State Transfer) API**는 2000년 로이 필딩(Roy Fielding)의 박사학위 논문에서 최초로 소개된 아키텍처 스타일입니다.

**REST의 핵심 제약 조건:**

1. **Client-Server**: 클라이언트와 서버의 관심사 분리
2. **Stateless**: 각 요청은 독립적이며 서버는 세션 상태를 저장하지 않음
3. **Cache**: 응답은 캐시 가능 여부를 명시해야 함
4. **Uniform Interface**: 일관된 인터페이스
   - 리소스가 URI로 식별되어야 함
   - 리소스 조작은 HTTP 메시지로 표현되어야 함
   - 메시지는 스스로 설명 가능해야 함 (Self-descriptive)
   - HATEOAS (Hypermedia As The Engine Of Application State)
5. **Layered System**: 계층화된 시스템 구조
6. **Code-on-Demand** (선택사항): 서버가 클라이언트에 실행 가능한 코드 전송

**HTTP 메소드와 CRUD 매핑:**

| HTTP Method | CRUD Operation | Idempotent | Safe |
|-------------|----------------|------------|------|
| GET         | Read           | ✓          | ✓    |
| POST        | Create         | ✗          | ✗    |
| PUT         | Update/Replace | ✓          | ✗    |
| PATCH       | Update/Modify  | ✗          | ✗    |
| DELETE      | Delete         | ✓          | ✗    |

**REST API 설계 예제 (Spring Boot):**

```java
import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;
import org.springframework.http.HttpStatus;
import java.net.URI;
import java.util.*;

@RestController
@RequestMapping("/users")
public class UserController {
    private Map<Integer, User> users = new HashMap<>();
    private int nextId = 1;

    // GET /users - 사용자 목록 조회
    @GetMapping
    public ResponseEntity<List<User>> getUsers() {
        return ResponseEntity.ok(new ArrayList<>(users.values()));
    }

    // GET /users/{id} - 특정 사용자 조회
    @GetMapping("/{id}")
    public ResponseEntity<User> getUser(@PathVariable int id) {
        User user = users.get(id);
        if (user == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(user);
    }

    // POST /users - 사용자 생성
    @PostMapping
    public ResponseEntity<User> createUser(@RequestBody UserRequest request) {
        User user = new User(nextId, request.getName(), request.getEmail());
        users.put(nextId, user);

        // 201 Created와 Location 헤더 반환
        URI location = URI.create("/users/" + nextId);
        nextId++;
        return ResponseEntity.created(location).body(user);
    }

    // PUT /users/{id} - 사용자 전체 업데이트
    @PutMapping("/{id}")
    public ResponseEntity<User> updateUser(
        @PathVariable int id,
        @RequestBody UserRequest request) {

        if (!users.containsKey(id)) {
            return ResponseEntity.notFound().build();
        }

        User user = new User(id, request.getName(), request.getEmail());
        users.put(id, user);
        return ResponseEntity.ok(user);
    }

    // PATCH /users/{id} - 사용자 부분 업데이트
    @PatchMapping("/{id}")
    public ResponseEntity<User> patchUser(
        @PathVariable int id,
        @RequestBody Map<String, Object> updates) {

        User user = users.get(id);
        if (user == null) {
            return ResponseEntity.notFound().build();
        }

        if (updates.containsKey("name")) {
            user.setName((String) updates.get("name"));
        }
        if (updates.containsKey("email")) {
            user.setEmail((String) updates.get("email"));
        }

        return ResponseEntity.ok(user);
    }

    // DELETE /users/{id} - 사용자 삭제
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteUser(@PathVariable int id) {
        if (!users.containsKey(id)) {
            return ResponseEntity.notFound().build();
        }

        users.remove(id);
        return ResponseEntity.noContent().build();  // 204 No Content
    }

    // GET /users/{id}/orders - 사용자의 주문 목록 (중첩 리소스)
    @GetMapping("/{id}/orders")
    public ResponseEntity<List<Order>> getUserOrders(@PathVariable int id) {
        if (!users.containsKey(id)) {
            return ResponseEntity.notFound().build();
        }
        // 주문 목록 반환 로직
        return ResponseEntity.ok(new ArrayList<>());
    }
}

class User {
    private int id;
    private String name;
    private String email;

    public User(int id, String name, String email) {
        this.id = id;
        this.name = name;
        this.email = email;
    }

    // Getters and Setters
    public int getId() { return id; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }
}

class UserRequest {
    private String name;
    private String email;

    // Getters and Setters
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }
}
```

**HTTP 상태 코드:**

```
2xx 성공:
- 200 OK: 요청 성공
- 201 Created: 리소스 생성 성공
- 204 No Content: 성공했지만 반환할 내용 없음

3xx 리다이렉션:
- 301 Moved Permanently: 영구 이동
- 302 Found: 임시 이동
- 304 Not Modified: 캐시된 버전 사용

4xx 클라이언트 에러:
- 400 Bad Request: 잘못된 요청
- 401 Unauthorized: 인증 필요
- 403 Forbidden: 권한 없음
- 404 Not Found: 리소스 없음
- 409 Conflict: 충돌 (중복 생성 등)
- 422 Unprocessable Entity: 검증 실패

5xx 서버 에러:
- 500 Internal Server Error: 서버 오류
- 502 Bad Gateway: 게이트웨이 오류
- 503 Service Unavailable: 서비스 일시 불가
```

### RPC

**RPC (Remote Procedure Call)**는 네트워크로 연결된 다른 컴퓨터나 프로그램의 프로시저(함수)를 로컬에서 호출하는 것처럼 실행할 수 있게 하는 통신 프로토콜입니다.

**RPC의 특징:**

- **동작 중심 (Action-oriented)**: 함수/메소드 호출에 초점
- **긴밀한 결합 (Tight coupling)**: 클라이언트가 서버의 함수 시그니처를 정확히 알아야 함
- **효율적인 바이너리 프로토콜**: JSON보다 빠른 직렬화
- **언어 독립적**: 다양한 언어 간 통신 가능

**RPC 예제 (Apache Thrift):**

```java
// 1. Thrift IDL 정의 (calculator.thrift)
/*
service CalculatorService {
    i32 add(1:i32 x, 2:i32 y),
    i32 subtract(1:i32 x, 2:i32 y),
    i32 multiply(1:i32 x, 2:i32 y),
    double divide(1:i32 x, 2:i32 y)
}
*/

// 2. RPC 서버 구현
import org.apache.thrift.server.TServer;
import org.apache.thrift.server.TSimpleServer;
import org.apache.thrift.transport.TServerSocket;
import org.apache.thrift.transport.TServerTransport;

public class CalculatorServer {
    public static class CalculatorHandler implements CalculatorService.Iface {
        @Override
        public int add(int x, int y) {
            System.out.println("add(" + x + ", " + y + ")");
            return x + y;
        }

        @Override
        public int subtract(int x, int y) {
            System.out.println("subtract(" + x + ", " + y + ")");
            return x - y;
        }

        @Override
        public int multiply(int x, int y) {
            System.out.println("multiply(" + x + ", " + y + ")");
            return x * y;
        }

        @Override
        public double divide(int x, int y) throws DivisionByZeroException {
            System.out.println("divide(" + x + ", " + y + ")");
            if (y == 0) {
                throw new DivisionByZeroException();
            }
            return (double) x / y;
        }
    }

    public static void main(String[] args) {
        try {
            CalculatorHandler handler = new CalculatorHandler();
            CalculatorService.Processor processor =
                new CalculatorService.Processor<>(handler);

            TServerTransport serverTransport = new TServerSocket(9090);
            TServer server = new TSimpleServer(
                new TServer.Args(serverTransport).processor(processor)
            );

            System.out.println("RPC Server running on port 9090...");
            server.serve();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

// 3. RPC 클라이언트
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.protocol.TProtocol;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransport;

public class CalculatorClient {
    public static void main(String[] args) {
        try {
            // 서버 연결
            TTransport transport = new TSocket("localhost", 9090);
            transport.open();

            TProtocol protocol = new TBinaryProtocol(transport);
            CalculatorService.Client client =
                new CalculatorService.Client(protocol);

            // 원격 함수 호출 (로컬 함수처럼 사용)
            int sum = client.add(5, 3);
            System.out.println("5 + 3 = " + sum);  // 8

            int product = client.multiply(4, 7);
            System.out.println("4 * 7 = " + product);  // 28

            double quotient = client.divide(10, 2);
            System.out.println("10 / 2 = " + quotient);  // 5.0

            transport.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

**RPC vs REST 비교:**

| 작업 | RPC | REST |
|------|-----|------|
| 회원가입 | POST /signup | POST /users |
| 회원탈퇴 | POST /resign<br/>{"userId": "1234"} | DELETE /users/1234 |
| 사용자 조회 | GET /readUser?userId=1234 | GET /users/1234 |
| 사용자 아이템 목록 | GET /readUserItems?userId=1234 | GET /users/1234/items |
| 아이템 추가 | POST /addItemToUser<br/>{"userId": "1234", "itemId": "456"} | POST /users/1234/items<br/>{"itemId": "456"} |
| 아이템 수정 | POST /modifyItem<br/>{"itemId": "456", "key": "value"} | PUT /items/456<br/>{"key": "value"} |
| 아이템 삭제 | POST /removeItem<br/>{"itemId": "456"} | DELETE /items/456 |

### gRPC

**gRPC (gRPC Remote Procedure Calls)**는 Google에서 개발한 고성능 오픈소스 RPC 프레임워크입니다. HTTP/2 프로토콜 기반으로 마이크로서비스 간 효율적인 통신을 제공합니다.

**gRPC의 주요 특징:**

1. **Protocol Buffers (protobuf)**
   - IDL (Interface Definition Language)로 사용
   - JSON/XML보다 효율적이고 빠름
   - 더 작은 메시지 크기

2. **언어 독립적 (Language-agnostic)**
   - Java, Go, C++, Python, Node.js 등 다양한 언어 지원
   - 서로 다른 언어 간 통신 가능

3. **양방향 스트리밍 (Bi-directional streaming)**
   - HTTP/2의 멀티플렉싱 활용
   - 클라이언트와 서버가 동시에 메시지 송수신

4. **강타입 (Strongly-typed)**
   - 컴파일 타임에 타입 체크
   - 개발 경험 향상, 에러 감소

5. **압축 (Compression)**
   - 네트워크 대역폭 절약
   - 성능 향상

6. **보안 (Security)**
   - TLS 기본 지원
   - 클라이언트 인증서 지원

**gRPC 예제:**

**1. Protocol Buffer 정의 (user.proto):**

```protobuf
syntax = "proto3";

option java_package = "com.example.grpc";
option java_outer_classname = "UserProto";

package user;

// 사용자 서비스 정의
service UserService {
  // 단일 요청-응답 (Unary RPC)
  rpc GetUser(GetUserRequest) returns (User) {}

  // 서버 스트리밍
  rpc ListUsers(ListUsersRequest) returns (stream User) {}

  // 클라이언트 스트리밍
  rpc CreateUsers(stream User) returns (CreateUsersResponse) {}

  // 양방향 스트리밍
  rpc Chat(stream ChatMessage) returns (stream ChatMessage) {}
}

message User {
  int32 id = 1;
  string name = 2;
  string email = 3;
  int32 age = 4;
}

message GetUserRequest {
  int32 id = 1;
}

message ListUsersRequest {
  int32 page = 1;
  int32 page_size = 2;
}

message CreateUsersResponse {
  int32 count = 1;
}

message ChatMessage {
  string user_id = 1;
  string message = 2;
  int64 timestamp = 3;
}
```

**2. Java gRPC 서버:**

```java
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;
import java.util.HashMap;
import java.util.Map;

public class UserServiceServer {
    private Server server;

    private void start() throws Exception {
        int port = 50051;
        server = ServerBuilder.forPort(port)
                .addService(new UserServiceImpl())
                .build()
                .start();

        System.out.println("gRPC Server started on port " + port);

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("Shutting down gRPC server...");
            UserServiceServer.this.stop();
        }));
    }

    private void stop() {
        if (server != null) {
            server.shutdown();
        }
    }

    private void blockUntilShutdown() throws InterruptedException {
        if (server != null) {
            server.awaitTermination();
        }
    }

    public static void main(String[] args) throws Exception {
        final UserServiceServer server = new UserServiceServer();
        server.start();
        server.blockUntilShutdown();
    }

    static class UserServiceImpl extends UserServiceGrpc.UserServiceImplBase {
        private Map<Integer, User> users = new HashMap<>();

        public UserServiceImpl() {
            users.put(1, User.newBuilder()
                    .setId(1)
                    .setName("Alice")
                    .setEmail("alice@example.com")
                    .setAge(30)
                    .build());
            users.put(2, User.newBuilder()
                    .setId(2)
                    .setName("Bob")
                    .setEmail("bob@example.com")
                    .setAge(25)
                    .build());
        }

        // 1. Unary RPC: 단일 사용자 조회
        @Override
        public void getUser(GetUserRequest request,
                           StreamObserver<User> responseObserver) {
            User user = users.get(request.getId());

            if (user == null) {
                responseObserver.onError(
                    new Exception("User not found")
                );
                return;
            }

            responseObserver.onNext(user);
            responseObserver.onCompleted();
        }

        // 2. 서버 스트리밍 RPC: 사용자 목록
        @Override
        public void listUsers(ListUsersRequest request,
                             StreamObserver<User> responseObserver) {
            for (User user : users.values()) {
                responseObserver.onNext(user);
                try {
                    Thread.sleep(100);  // 스트리밍 시뮬레이션
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            responseObserver.onCompleted();
        }

        // 3. 클라이언트 스트리밍 RPC: 여러 사용자 생성
        @Override
        public StreamObserver<User> createUsers(
                StreamObserver<CreateUsersResponse> responseObserver) {

            return new StreamObserver<User>() {
                int count = 0;

                @Override
                public void onNext(User user) {
                    int userId = users.size() + 1;
                    User newUser = user.toBuilder()
                            .setId(userId)
                            .build();
                    users.put(userId, newUser);
                    count++;
                    System.out.println("Created user: " + newUser.getName());
                }

                @Override
                public void onError(Throwable t) {
                    System.err.println("Error: " + t.getMessage());
                }

                @Override
                public void onCompleted() {
                    CreateUsersResponse response = CreateUsersResponse
                            .newBuilder()
                            .setCount(count)
                            .build();
                    responseObserver.onNext(response);
                    responseObserver.onCompleted();
                }
            };
        }

        // 4. 양방향 스트리밍 RPC: 채팅
        @Override
        public StreamObserver<ChatMessage> chat(
                StreamObserver<ChatMessage> responseObserver) {

            return new StreamObserver<ChatMessage>() {
                @Override
                public void onNext(ChatMessage message) {
                    // 에코 응답
                    ChatMessage response = ChatMessage.newBuilder()
                            .setUserId("server")
                            .setMessage("Echo: " + message.getMessage())
                            .setTimestamp(System.currentTimeMillis())
                            .build();
                    responseObserver.onNext(response);
                }

                @Override
                public void onError(Throwable t) {
                    System.err.println("Error: " + t.getMessage());
                }

                @Override
                public void onCompleted() {
                    responseObserver.onCompleted();
                }
            };
        }
    }
}
```

**3. Java gRPC 클라이언트:**

```java
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.stub.StreamObserver;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

public class UserServiceClient {
    private final ManagedChannel channel;
    private final UserServiceGrpc.UserServiceBlockingStub blockingStub;
    private final UserServiceGrpc.UserServiceStub asyncStub;

    public UserServiceClient(String host, int port) {
        channel = ManagedChannelBuilder
                .forAddress(host, port)
                .usePlaintext()
                .build();

        blockingStub = UserServiceGrpc.newBlockingStub(channel);
        asyncStub = UserServiceGrpc.newStub(channel);
    }

    public void shutdown() throws InterruptedException {
        channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
    }

    // 1. Unary RPC: 단일 사용자 조회
    public void getUser(int userId) {
        System.out.println("=== Unary RPC ===");
        GetUserRequest request = GetUserRequest.newBuilder()
                .setId(userId)
                .build();

        try {
            User user = blockingStub.getUser(request);
            System.out.println("User: " + user.getName() +
                             ", " + user.getEmail());
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
        }
    }

    // 2. 서버 스트리밍 RPC: 사용자 목록
    public void listUsers() {
        System.out.println("\n=== Server Streaming RPC ===");
        ListUsersRequest request = ListUsersRequest.newBuilder()
                .setPage(1)
                .setPageSize(10)
                .build();

        blockingStub.listUsers(request).forEachRemaining(user -> {
            System.out.println("User: " + user.getName() +
                             ", Age: " + user.getAge());
        });
    }

    // 3. 클라이언트 스트리밍 RPC: 여러 사용자 생성
    public void createUsers() throws InterruptedException {
        System.out.println("\n=== Client Streaming RPC ===");
        CountDownLatch latch = new CountDownLatch(1);

        StreamObserver<CreateUsersResponse> responseObserver =
            new StreamObserver<CreateUsersResponse>() {
                @Override
                public void onNext(CreateUsersResponse response) {
                    System.out.println("Created " + response.getCount() +
                                     " users");
                }

                @Override
                public void onError(Throwable t) {
                    System.err.println("Error: " + t.getMessage());
                    latch.countDown();
                }

                @Override
                public void onCompleted() {
                    latch.countDown();
                }
            };

        StreamObserver<User> requestObserver =
            asyncStub.createUsers(responseObserver);

        List<User> newUsers = Arrays.asList(
            User.newBuilder()
                .setName("Charlie")
                .setEmail("charlie@example.com")
                .setAge(28)
                .build(),
            User.newBuilder()
                .setName("David")
                .setEmail("david@example.com")
                .setAge(32)
                .build(),
            User.newBuilder()
                .setName("Eve")
                .setEmail("eve@example.com")
                .setAge(27)
                .build()
        );

        for (User user : newUsers) {
            requestObserver.onNext(user);
            Thread.sleep(100);
        }

        requestObserver.onCompleted();
        latch.await(1, TimeUnit.MINUTES);
    }

    // 4. 양방향 스트리밍 RPC: 채팅
    public void chat() throws InterruptedException {
        System.out.println("\n=== Bidirectional Streaming RPC ===");
        CountDownLatch latch = new CountDownLatch(1);

        StreamObserver<ChatMessage> responseObserver =
            new StreamObserver<ChatMessage>() {
                @Override
                public void onNext(ChatMessage message) {
                    System.out.println("Server: " + message.getMessage());
                }

                @Override
                public void onError(Throwable t) {
                    System.err.println("Error: " + t.getMessage());
                    latch.countDown();
                }

                @Override
                public void onCompleted() {
                    latch.countDown();
                }
            };

        StreamObserver<ChatMessage> requestObserver =
            asyncStub.chat(responseObserver);

        List<String> messages = Arrays.asList(
            "Hello", "How are you?", "Goodbye"
        );

        for (String msg : messages) {
            ChatMessage message = ChatMessage.newBuilder()
                    .setUserId("client1")
                    .setMessage(msg)
                    .setTimestamp(System.currentTimeMillis())
                    .build();
            requestObserver.onNext(message);
            Thread.sleep(500);
        }

        requestObserver.onCompleted();
        latch.await(1, TimeUnit.MINUTES);
    }

    public static void main(String[] args) throws Exception {
        UserServiceClient client = new UserServiceClient("localhost", 50051);

        try {
            client.getUser(1);
            client.listUsers();
            client.createUsers();
            client.chat();
        } finally {
            client.shutdown();
        }
    }
}
```

**gRPC 스트리밍 패턴:**

```protobuf
// 1. Unary (단일 요청-응답)
rpc GetUser(UserRequest) returns (User)

// 2. Server Streaming (서버가 여러 응답)
rpc ListUsers(ListRequest) returns (stream User)
// 예: 대량 데이터 조회, 실시간 업데이트

// 3. Client Streaming (클라이언트가 여러 요청)
rpc UploadFile(stream Chunk) returns (UploadResponse)
// 예: 파일 업로드, 배치 처리

// 4. Bidirectional Streaming (양방향 스트리밍)
rpc Chat(stream Message) returns (stream Message)
// 예: 채팅, 실시간 협업
```

**gRPC 사용 사례:**

- ✓ 마이크로서비스 간 통신
- ✓ IoT 플랫폼
- ✓ 실시간 애플리케이션
- ✓ 모바일 클라이언트 (네트워크 효율성)
- ✓ 낮은 레이턴시가 필요한 서비스
- ✗ 브라우저 직접 통신 (제한적)
- ✗ HTTP/1.1만 지원하는 환경

### gRPC vs REST API

| 특징 | REST | gRPC |
|------|------|------|
| **프로토콜** | HTTP/1.1, HTTP/2 | HTTP/2 |
| **데이터 형식** | JSON, XML (텍스트) | Protocol Buffers (바이너리) |
| **API 스타일** | 리소스 중심 | 메소드/프로시저 중심 |
| **스트리밍** | 제한적 (SSE, Long Polling) | 양방향 스트리밍 지원 |
| **브라우저 지원** | 네이티브 지원 | gRPC-Web 필요 |
| **성능** | 보통 | 높음 (바이너리, 압축) |
| **메시지 크기** | 큼 (텍스트) | 작음 (바이너리) |
| **코드 생성** | 선택적 | 필수 (protoc) |
| **타입 안정성** | 런타임 검증 | 컴파일 타임 검증 |
| **캐싱** | HTTP 캐싱 활용 | 커스텀 캐싱 필요 |
| **가독성** | 높음 (JSON) | 낮음 (바이너리) |
| **학습 곡선** | 낮음 | 높음 |
| **사용 사례** | 공개 API, 웹 앱 | 마이크로서비스, 고성능 통신 |

**REST vs GraphQL vs gRPC 종합 비교:**

| 특징 | REST | GraphQL | gRPC |
|------|------|---------|------|
| **결합도** | 느슨함 (리소스 기반) | 느슨함 (쿼리 기반) | 긴밀함 (RPC 기반) |
| **통신 횟수** | 여러 요청 필요할 수 있음 | 단일 요청으로 필요한 데이터 조회 | 효율적 (Protocol Buffers) |
| **성능** | HTTP 캐싱 의존 | 효율적 데이터 페칭 필요 | 고성능 (HTTP/2, 바이너리) |
| **복잡도** | 간단 (HTTP 메소드) | 학습 곡선 존재 (쿼리 언어) | 높음 (Protocol Buffers, RPC) |
| **캐싱** | HTTP 캐싱 지원 | 커스텀 캐싱 전략 필요 | 커스텀 캐싱 필요 |
| **코드 생성** | 선택적 | 스키마/타입 코드 생성 가능 | 필수 (protoc) |
| **디스커버리** | URL, HTTP 헤더 | 인트로스펙션 지원 | 서비스 계약 (강타입) |
| **버전 관리** | URL 또는 헤더 | 스키마 확장, 필드 deprecated | 서비스 계약 변경, 버전 협상 |

**선택 가이드:**

```
REST 선택:
- 공개 API
- 단순한 CRUD 작업
- 브라우저 직접 호출
- HTTP 캐싱 활용

GraphQL 선택:
- 복잡한 데이터 요구사항
- Over-fetching/Under-fetching 문제
- 프론트엔드 중심 개발
- 유연한 데이터 조회

gRPC 선택:
- 마이크로서비스 간 통신
- 고성능 요구사항
- 실시간 양방향 스트리밍
- 다국어 환경
- 낮은 레이턴시 중요
```

### GraphQL

**GraphQL**은 Facebook이 2012년에 개발하고 2015년에 오픈소스로 공개한 API를 위한 쿼리 언어이자 런타임입니다. 전통적인 REST API 방식의 효율적이고 유연하며 강력한 대안으로 설계되었습니다.

**GraphQL의 핵심 구성 요소:**

1. **쿼리 언어 (Query Language)**
   - 사람이 읽기 쉬운 강타입 쿼리 언어
   - 클라이언트가 필요한 데이터를 정확히 요청
   - 단일 요청으로 중첩되고 관련된 데이터 조회 가능

2. **스키마 (Schema)**
   - 타입, 관계, 연산을 정의
   - Object, Scalar, Enum, Interface, Union 타입 지원
   - 강타입 시스템으로 검증, 성능 최적화, 자동완성 지원

3. **런타임 (Runtime)**
   - 리졸버(Resolver) 함수를 실행하여 쿼리 처리
   - 다양한 데이터 소스(DB, API, 서비스)에서 데이터 가져오기
   - 클라이언트가 요청한 형태로 데이터 반환

**GraphQL의 주요 특징:**

- **유연한 데이터 조회**: Over-fetching, Under-fetching 문제 해결
- **계층적 구조**: 데이터의 계층 구조를 자연스럽게 표현
- **강타입 스키마**: 타입 시스템으로 검증과 도구 지원 향상
- **인트로스펙션**: 런타임에 스키마 탐색 가능
- **실시간 업데이트**: Subscription으로 실시간 데이터 수신

**GraphQL 스키마 예제:**

```graphql
# 타입 정의
type User {
  id: ID!
  name: String!
  email: String!
  age: Int
  posts: [Post!]!
  friends: [User!]!
}

type Post {
  id: ID!
  title: String!
  content: String!
  author: User!
  comments: [Comment!]!
  createdAt: String!
}

type Comment {
  id: ID!
  content: String!
  author: User!
  post: Post!
}

# 쿼리 정의
type Query {
  # 단일 사용자 조회
  user(id: ID!): User

  # 사용자 목록 조회
  users(limit: Int, offset: Int): [User!]!

  # 게시물 조회
  post(id: ID!): Post

  # 게시물 검색
  searchPosts(keyword: String!): [Post!]!
}

# 뮤테이션 정의
type Mutation {
  # 사용자 생성
  createUser(name: String!, email: String!, age: Int): User!

  # 사용자 업데이트
  updateUser(id: ID!, name: String, email: String, age: Int): User!

  # 사용자 삭제
  deleteUser(id: ID!): Boolean!

  # 게시물 생성
  createPost(title: String!, content: String!, authorId: ID!): Post!

  # 댓글 추가
  addComment(postId: ID!, content: String!, authorId: ID!): Comment!
}

# 구독 정의
type Subscription {
  # 새 게시물 알림
  postAdded: Post!

  # 댓글 추가 알림
  commentAdded(postId: ID!): Comment!
}
```

**GraphQL 쿼리 예제:**

```graphql
# 1. 기본 쿼리 - 필요한 필드만 선택
query {
  user(id: "1") {
    id
    name
    email
  }
}

# 2. 중첩 쿼리 - 관련 데이터 함께 조회
query {
  user(id: "1") {
    id
    name
    posts {
      id
      title
      comments {
        id
        content
        author {
          name
        }
      }
    }
  }
}

# 3. 별칭(Alias) 사용
query {
  user1: user(id: "1") {
    name
    email
  }
  user2: user(id: "2") {
    name
    email
  }
}

# 4. 프래그먼트(Fragment) - 재사용 가능한 필드 세트
fragment UserFields on User {
  id
  name
  email
  age
}

query {
  user(id: "1") {
    ...UserFields
    posts {
      title
    }
  }
}

# 5. 변수 사용
query GetUser($userId: ID!, $postLimit: Int) {
  user(id: $userId) {
    name
    posts(limit: $postLimit) {
      title
      content
    }
  }
}

# 6. 뮤테이션
mutation {
  createUser(name: "Alice", email: "alice@example.com", age: 30) {
    id
    name
    email
  }
}

# 7. 구독
subscription {
  postAdded {
    id
    title
    author {
      name
    }
  }
}
```

**Java GraphQL 서버 구현 (Spring Boot + GraphQL Java):**

```java
// 1. Maven 의존성
/*
<dependency>
    <groupId>com.graphql-java</groupId>
    <artifactId>graphql-java</artifactId>
    <version>20.0</version>
</dependency>
<dependency>
    <groupId>com.graphql-java</groupId>
    <artifactId>graphql-java-spring-boot-starter-webmvc</artifactId>
    <version>2.0</version>
</dependency>
*/

// 2. 도메인 모델
public class User {
    private String id;
    private String name;
    private String email;
    private Integer age;
    private List<Post> posts;

    // Constructor, Getters, Setters
    public User(String id, String name, String email, Integer age) {
        this.id = id;
        this.name = name;
        this.email = email;
        this.age = age;
        this.posts = new ArrayList<>();
    }

    // Getters and Setters
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }
    public Integer getAge() { return age; }
    public void setAge(Integer age) { this.age = age; }
    public List<Post> getPosts() { return posts; }
    public void setPosts(List<Post> posts) { this.posts = posts; }
}

public class Post {
    private String id;
    private String title;
    private String content;
    private User author;
    private List<Comment> comments;
    private String createdAt;

    public Post(String id, String title, String content, User author) {
        this.id = id;
        this.title = title;
        this.content = content;
        this.author = author;
        this.comments = new ArrayList<>();
        this.createdAt = Instant.now().toString();
    }

    // Getters and Setters
    public String getId() { return id; }
    public String getTitle() { return title; }
    public String getContent() { return content; }
    public User getAuthor() { return author; }
    public List<Comment> getComments() { return comments; }
    public String getCreatedAt() { return createdAt; }
}

public class Comment {
    private String id;
    private String content;
    private User author;
    private Post post;

    public Comment(String id, String content, User author, Post post) {
        this.id = id;
        this.content = content;
        this.author = author;
        this.post = post;
    }

    // Getters
    public String getId() { return id; }
    public String getContent() { return content; }
    public User getAuthor() { return author; }
    public Post getPost() { return post; }
}

// 3. GraphQL Controller
import org.springframework.graphql.data.method.annotation.Argument;
import org.springframework.graphql.data.method.annotation.MutationMapping;
import org.springframework.graphql.data.method.annotation.QueryMapping;
import org.springframework.graphql.data.method.annotation.SchemaMapping;
import org.springframework.stereotype.Controller;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@Controller
public class GraphQLController {
    private Map<String, User> users = new ConcurrentHashMap<>();
    private Map<String, Post> posts = new ConcurrentHashMap<>();
    private int userIdCounter = 1;
    private int postIdCounter = 1;

    public GraphQLController() {
        // 테스트 데이터 초기화
        User user1 = new User("1", "Alice", "alice@example.com", 30);
        User user2 = new User("2", "Bob", "bob@example.com", 25);
        users.put("1", user1);
        users.put("2", user2);

        Post post1 = new Post("1", "GraphQL Introduction",
                              "GraphQL is awesome!", user1);
        Post post2 = new Post("2", "Java Spring Boot",
                              "Building APIs with Spring Boot", user1);
        posts.put("1", post1);
        posts.put("2", post2);

        user1.getPosts().add(post1);
        user1.getPosts().add(post2);
    }

    // Query Resolvers
    @QueryMapping
    public User user(@Argument String id) {
        return users.get(id);
    }

    @QueryMapping
    public List<User> users(@Argument Integer limit, @Argument Integer offset) {
        List<User> userList = new ArrayList<>(users.values());

        int start = offset != null ? offset : 0;
        int end = limit != null ? Math.min(start + limit, userList.size())
                                : userList.size();

        return userList.subList(start, Math.min(end, userList.size()));
    }

    @QueryMapping
    public Post post(@Argument String id) {
        return posts.get(id);
    }

    @QueryMapping
    public List<Post> searchPosts(@Argument String keyword) {
        return posts.values().stream()
                .filter(post -> post.getTitle().contains(keyword) ||
                               post.getContent().contains(keyword))
                .collect(Collectors.toList());
    }

    // Mutation Resolvers
    @MutationMapping
    public User createUser(@Argument String name,
                          @Argument String email,
                          @Argument Integer age) {
        String id = String.valueOf(userIdCounter++);
        User user = new User(id, name, email, age);
        users.put(id, user);
        return user;
    }

    @MutationMapping
    public User updateUser(@Argument String id,
                          @Argument String name,
                          @Argument String email,
                          @Argument Integer age) {
        User user = users.get(id);
        if (user == null) {
            throw new RuntimeException("User not found: " + id);
        }

        if (name != null) user.setName(name);
        if (email != null) user.setEmail(email);
        if (age != null) user.setAge(age);

        return user;
    }

    @MutationMapping
    public Boolean deleteUser(@Argument String id) {
        return users.remove(id) != null;
    }

    @MutationMapping
    public Post createPost(@Argument String title,
                          @Argument String content,
                          @Argument String authorId) {
        User author = users.get(authorId);
        if (author == null) {
            throw new RuntimeException("Author not found: " + authorId);
        }

        String id = String.valueOf(postIdCounter++);
        Post post = new Post(id, title, content, author);
        posts.put(id, post);
        author.getPosts().add(post);

        return post;
    }

    // Field Resolvers (복잡한 필드 해결)
    @SchemaMapping(typeName = "User", field = "posts")
    public List<Post> userPosts(User user) {
        // User의 posts 필드가 요청될 때 실행
        return posts.values().stream()
                .filter(post -> post.getAuthor().getId().equals(user.getId()))
                .collect(Collectors.toList());
    }
}

// 4. GraphQL Schema 파일 (src/main/resources/graphql/schema.graphqls)
/*
type User {
  id: ID!
  name: String!
  email: String!
  age: Int
  posts: [Post!]!
}

type Post {
  id: ID!
  title: String!
  content: String!
  author: User!
  comments: [Comment!]!
  createdAt: String!
}

type Comment {
  id: ID!
  content: String!
  author: User!
  post: Post!
}

type Query {
  user(id: ID!): User
  users(limit: Int, offset: Int): [User!]!
  post(id: ID!): Post
  searchPosts(keyword: String!): [Post!]!
}

type Mutation {
  createUser(name: String!, email: String!, age: Int): User!
  updateUser(id: ID!, name: String, email: String, age: Int): User!
  deleteUser(id: ID!): Boolean!
  createPost(title: String!, content: String!, authorId: ID!): Post!
}
*/

// 5. Application Properties (application.yml)
/*
spring:
  graphql:
    graphiql:
      enabled: true
      path: /graphiql
*/
```

**Java GraphQL 클라이언트 구현:**

```java
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.URI;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;

public class GraphQLClient {
    private final String endpoint;
    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;

    public GraphQLClient(String endpoint) {
        this.endpoint = endpoint;
        this.httpClient = HttpClient.newHttpClient();
        this.objectMapper = new ObjectMapper();
    }

    public JsonNode executeQuery(String query, Map<String, Object> variables)
            throws Exception {
        // GraphQL 요청 본문 생성
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("query", query);
        if (variables != null) {
            requestBody.put("variables", variables);
        }

        String jsonBody = objectMapper.writeValueAsString(requestBody);

        // HTTP 요청 생성
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(endpoint))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                .build();

        // 요청 실행
        HttpResponse<String> response = httpClient.send(
            request,
            HttpResponse.BodyHandlers.ofString()
        );

        // 응답 파싱
        JsonNode jsonResponse = objectMapper.readTree(response.body());

        if (jsonResponse.has("errors")) {
            throw new RuntimeException("GraphQL errors: " +
                jsonResponse.get("errors").toString());
        }

        return jsonResponse.get("data");
    }

    public static void main(String[] args) throws Exception {
        GraphQLClient client = new GraphQLClient("http://localhost:8080/graphql");

        // 1. 단일 사용자 조회
        String query1 = """
            query {
              user(id: "1") {
                id
                name
                email
                posts {
                  title
                }
              }
            }
            """;

        JsonNode result1 = client.executeQuery(query1, null);
        System.out.println("User: " + result1.toPrettyString());

        // 2. 변수를 사용한 쿼리
        String query2 = """
            query GetUser($userId: ID!) {
              user(id: $userId) {
                name
                email
                age
              }
            }
            """;

        Map<String, Object> variables = new HashMap<>();
        variables.put("userId", "2");

        JsonNode result2 = client.executeQuery(query2, variables);
        System.out.println("User with variables: " + result2.toPrettyString());

        // 3. Mutation - 사용자 생성
        String mutation = """
            mutation CreateUser($name: String!, $email: String!, $age: Int) {
              createUser(name: $name, email: $email, age: $age) {
                id
                name
                email
                age
              }
            }
            """;

        Map<String, Object> mutationVars = new HashMap<>();
        mutationVars.put("name", "Charlie");
        mutationVars.put("email", "charlie@example.com");
        mutationVars.put("age", 28);

        JsonNode result3 = client.executeQuery(mutation, mutationVars);
        System.out.println("Created user: " + result3.toPrettyString());

        // 4. 복잡한 중첩 쿼리
        String complexQuery = """
            query {
              users(limit: 10) {
                id
                name
                posts {
                  id
                  title
                  comments {
                    content
                    author {
                      name
                    }
                  }
                }
              }
            }
            """;

        JsonNode result4 = client.executeQuery(complexQuery, null);
        System.out.println("Complex query: " + result4.toPrettyString());
    }
}
```

**GraphQL vs REST 비교:**

| 특징 | REST | GraphQL |
|------|------|---------|
| **엔드포인트** | 리소스별 여러 엔드포인트 | 단일 엔드포인트 |
| **데이터 조회** | 고정된 구조 | 클라이언트가 필요한 필드 선택 |
| **Over-fetching** | 자주 발생 | 없음 |
| **Under-fetching** | 여러 요청 필요 | 단일 요청으로 해결 |
| **버전 관리** | URL 버전 필요 | 스키마 진화 (필드 deprecated) |
| **캐싱** | HTTP 캐싱 활용 | 커스텀 캐싱 필요 |
| **타입 시스템** | 없음 (문서화 필요) | 강타입 스키마 |
| **도구 지원** | 제한적 | 인트로스펙션, GraphiQL |
| **학습 곡선** | 낮음 | 중간 |
| **성능** | 단순 요청에 유리 | 복잡한 데이터 조회에 유리 |

**GraphQL의 장단점:**

**장점:**
- ✓ 정확히 필요한 데이터만 요청
- ✓ 단일 요청으로 여러 리소스 조회
- ✓ 강타입 스키마로 검증과 문서화 자동화
- ✓ 빠른 프론트엔드 개발 (백엔드 변경 없이 데이터 요구사항 조정)
- ✓ 인트로스펙션으로 API 탐색 용이

**단점:**
- ✗ 단순 API에는 과도한 복잡성
- ✗ 캐싱 전략 복잡
- ✗ 쿼리 복잡도 제어 필요 (깊은 중첩, N+1 문제)
- ✗ 파일 업로드 처리 복잡
- ✗ HTTP 상태 코드 활용 제한

**GraphQL 사용 사례:**

- ✓ 모바일 앱 (네트워크 효율성 중요)
- ✓ 복잡한 데이터 관계
- ✓ 다양한 클라이언트 (웹, 모바일, 데스크톱)
- ✓ 빠른 프로토타이핑
- ✓ 마이크로프론트엔드
- ✗ 단순 CRUD API
- ✗ 실시간 요구사항 (WebSocket이 더 적합할 수 있음)

**GraphQL N+1 문제 해결 (DataLoader):**

```java
import org.dataloader.DataLoader;
import org.dataloader.DataLoaderRegistry;
import java.util.concurrent.CompletableFuture;

public class DataLoaderConfig {

    // BatchLoader로 N+1 문제 해결
    public DataLoaderRegistry buildRegistry() {
        DataLoaderRegistry registry = new DataLoaderRegistry();

        // User의 Posts를 배치로 로드
        DataLoader<String, List<Post>> postsLoader = DataLoader.newDataLoader(
            (List<String> userIds) -> CompletableFuture.supplyAsync(() -> {
                // 한 번의 쿼리로 모든 userId의 posts 조회
                return batchLoadPosts(userIds);
            })
        );

        registry.register("postsLoader", postsLoader);
        return registry;
    }

    private List<List<Post>> batchLoadPosts(List<String> userIds) {
        // DB에서 한 번에 조회
        Map<String, List<Post>> postsByUser =
            postRepository.findByAuthorIdIn(userIds)
                .stream()
                .collect(Collectors.groupingBy(
                    post -> post.getAuthor().getId()
                ));

        // 순서 유지하며 반환
        return userIds.stream()
                .map(id -> postsByUser.getOrDefault(id, new ArrayList<>()))
                .collect(Collectors.toList());
    }
}
```

### Long Polling

**Long Polling**은 서버 푸시가 지원되지 않거나 불가능한 웹 애플리케이션에서 클라이언트와 서버 간의 실시간에 가까운 통신을 시뮬레이션하는 기술입니다. 전통적인 폴링 방식의 확장으로, 클라이언트가 반복적으로 서버에 데이터 업데이트를 요청하는 방식을 개선합니다.

**동작 방식:**

```
클라이언트                              서버
   |                                     |
   |---(1) 요청 전송 -------------------->|
   |                                     |
   |                                     | (2) 새 데이터 대기
   |                                     |     또는 타임아웃까지 유지
   |                                     |
   |<--(3) 응답 (데이터 또는 타임아웃)---|
   |                                     |
   |---(4) 즉시 새 요청 전송 ------------>|
   |                                     |
```

**전통적인 폴링 vs Long Polling:**

| 특징 | 전통적인 폴링 | Long Polling |
|------|--------------|--------------|
| **요청 빈도** | 고정된 간격으로 반복 | 응답 후 즉시 재요청 |
| **서버 부하** | 불필요한 요청 많음 | 필요할 때만 응답 |
| **레이턴시** | 폴링 간격만큼 지연 | 최소화 (거의 실시간) |
| **대역폭** | 비효율적 | 효율적 |

**Java Long Polling 서버 구현 (Spring Boot):**

```java
import org.springframework.web.bind.annotation.*;
import org.springframework.web.context.request.async.DeferredResult;
import org.springframework.stereotype.Service;
import java.util.*;
import java.util.concurrent.*;

@RestController
@RequestMapping("/api")
public class LongPollingController {

    private final NotificationService notificationService;

    public LongPollingController(NotificationService notificationService) {
        this.notificationService = notificationService;
    }

    // Long Polling 엔드포인트
    @GetMapping("/notifications")
    public DeferredResult<List<String>> getNotifications(
            @RequestParam(defaultValue = "0") Long lastEventId) {

        // 30초 타임아웃 설정
        DeferredResult<List<String>> deferredResult =
            new DeferredResult<>(30000L);

        // 타임아웃 시 빈 리스트 반환
        deferredResult.onTimeout(() ->
            deferredResult.setResult(new ArrayList<>())
        );

        // 새 알림 대기
        notificationService.waitForNotifications(lastEventId, deferredResult);

        return deferredResult;
    }

    // 알림 전송 (테스트용)
    @PostMapping("/notifications")
    public Map<String, String> sendNotification(@RequestBody String message) {
        notificationService.addNotification(message);
        return Map.of("status", "sent", "message", message);
    }
}

@Service
class NotificationService {
    private final List<String> notifications = new CopyOnWriteArrayList<>();
    private final Map<Long, DeferredResult<List<String>>> pendingRequests =
        new ConcurrentHashMap<>();
    private long eventIdCounter = 0;

    // 새 알림 추가 및 대기 중인 클라이언트에게 전송
    public void addNotification(String message) {
        notifications.add(message);
        eventIdCounter++;

        // 대기 중인 모든 요청에 응답
        pendingRequests.forEach((id, deferredResult) -> {
            List<String> newNotifications = getNotificationsSince(id);
            if (!newNotifications.isEmpty()) {
                deferredResult.setResult(newNotifications);
                pendingRequests.remove(id);
            }
        });
    }

    // 특정 이벤트 ID 이후의 알림 조회
    public List<String> getNotificationsSince(Long lastEventId) {
        int startIndex = lastEventId.intValue();
        if (startIndex >= notifications.size()) {
            return new ArrayList<>();
        }
        return new ArrayList<>(
            notifications.subList(startIndex, notifications.size())
        );
    }

    // 새 알림 대기
    public void waitForNotifications(Long lastEventId,
                                     DeferredResult<List<String>> deferredResult) {
        // 이미 새 알림이 있는지 확인
        List<String> newNotifications = getNotificationsSince(lastEventId);

        if (!newNotifications.isEmpty()) {
            // 즉시 응답
            deferredResult.setResult(newNotifications);
        } else {
            // 새 알림을 기다림
            pendingRequests.put(lastEventId, deferredResult);

            // 타임아웃 시 정리
            deferredResult.onTimeout(() -> pendingRequests.remove(lastEventId));
            deferredResult.onCompletion(() -> pendingRequests.remove(lastEventId));
        }
    }
}
```

**Java Long Polling 클라이언트 구현:**

```java
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.URI;
import java.time.Duration;
import com.fasterxml.jackson.databind.ObjectMapper;

public class LongPollingClient {
    private final String serverUrl;
    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    private long lastEventId = 0;
    private volatile boolean running = true;

    public LongPollingClient(String serverUrl) {
        this.serverUrl = serverUrl;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
        this.objectMapper = new ObjectMapper();
    }

    // Long Polling 시작
    public void start() {
        Thread pollingThread = new Thread(() -> {
            while (running) {
                try {
                    poll();
                } catch (Exception e) {
                    System.err.println("Polling error: " + e.getMessage());
                    try {
                        Thread.sleep(1000); // 에러 시 1초 대기
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
        });
        pollingThread.start();
    }

    // 단일 폴링 요청
    private void poll() throws Exception {
        String url = serverUrl + "/api/notifications?lastEventId=" + lastEventId;

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .timeout(Duration.ofSeconds(35)) // 서버 타임아웃보다 길게
                .GET()
                .build();

        HttpResponse<String> response = httpClient.send(
            request,
            HttpResponse.BodyHandlers.ofString()
        );

        if (response.statusCode() == 200) {
            String[] notifications = objectMapper.readValue(
                response.body(),
                String[].class
            );

            if (notifications.length > 0) {
                for (String notification : notifications) {
                    System.out.println("New notification: " + notification);
                    lastEventId++;
                }
            }
        }
    }

    public void stop() {
        running = false;
    }

    public static void main(String[] args) throws InterruptedException {
        LongPollingClient client = new LongPollingClient("http://localhost:8080");
        client.start();

        System.out.println("Long polling client started. Press Ctrl+C to stop.");

        // 메인 스레드 유지
        Thread.currentThread().join();
    }
}
```

**Long Polling의 장단점:**

**장점:**
- ✓ 낮은 레이턴시 (전통적인 폴링보다 빠름)
- ✓ 서버 부하 감소 (불필요한 요청 제거)
- ✓ HTTP 기반으로 방화벽/프록시 통과 용이
- ✓ 구현 상대적으로 간단

**단점:**
- ✗ 진정한 실시간 통신 아님
- ✗ 많은 동시 연결 시 서버 리소스 소모
- ✗ 타임아웃 처리 복잡성
- ✗ 헤더 오버헤드 (매 요청마다)

**사용 사례:**
- 알림 시스템
- 채팅 애플리케이션 (낮은 메시지 빈도)
- 대시보드 업데이트
- WebSocket 미지원 환경

### Web Socket

**WebSocket**은 클라이언트와 서버 간의 양방향 실시간 통신을 가능하게 하는 프로토콜입니다. HTTP 핸드셰이크 후 지속적인 TCP 연결을 유지하여 효율적인 실시간 데이터 교환을 제공합니다.

**WebSocket vs HTTP 비교:**

| 특징 | HTTP | WebSocket |
|------|------|-----------|
| **통신 방식** | 요청-응답 (단방향) | 양방향 (Full-duplex) |
| **연결** | 매 요청마다 새 연결 | 지속적인 연결 유지 |
| **오버헤드** | 높음 (헤더 반복) | 낮음 (초기 핸드셰이크만) |
| **레이턴시** | 상대적으로 높음 | 매우 낮음 |
| **프로토콜** | HTTP/HTTPS | ws:// / wss:// |

**WebSocket 연결 과정:**

```
클라이언트                              서버
   |                                     |
   |---(1) HTTP Upgrade 요청 ----------->|
   |     GET /chat HTTP/1.1              |
   |     Upgrade: websocket              |
   |     Connection: Upgrade             |
   |                                     |
   |<--(2) HTTP 101 Switching Protocols -|
   |     HTTP/1.1 101 Switching Protocols|
   |     Upgrade: websocket              |
   |     Connection: Upgrade             |
   |                                     |
   |<===== WebSocket 연결 확립 =========>|
   |                                     |
   |<--(3) 양방향 메시지 교환 ----------->|
   |<----------------------------------->|
```

**Java WebSocket 서버 구현 (Spring Boot):**

```java
// 1. WebSocket 설정
import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.config.annotation.*;

@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {

    @Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry.addHandler(new ChatWebSocketHandler(), "/ws/chat")
                .setAllowedOrigins("*"); // CORS 설정
    }
}

// 2. WebSocket 핸들러
import org.springframework.web.socket.*;
import org.springframework.web.socket.handler.TextWebSocketHandler;
import java.util.*;
import java.util.concurrent.CopyOnWriteArraySet;

public class ChatWebSocketHandler extends TextWebSocketHandler {
    // 연결된 모든 세션 관리
    private static final Set<WebSocketSession> sessions =
        new CopyOnWriteArraySet<>();

    // 연결 성공 시
    @Override
    public void afterConnectionEstablished(WebSocketSession session)
            throws Exception {
        sessions.add(session);
        System.out.println("New connection: " + session.getId());

        // 환영 메시지 전송
        session.sendMessage(new TextMessage(
            "Welcome! Connected as " + session.getId()
        ));

        // 다른 사용자들에게 알림
        broadcast(session.getId() + " joined the chat", session);
    }

    // 메시지 수신 시
    @Override
    protected void handleTextMessage(WebSocketSession session,
                                     TextMessage message) throws Exception {
        String payload = message.getPayload();
        System.out.println("Received from " + session.getId() + ": " + payload);

        // 모든 연결된 클라이언트에게 브로드캐스트
        String formattedMessage = "[" + session.getId() + "]: " + payload;
        broadcast(formattedMessage, null);
    }

    // 연결 종료 시
    @Override
    public void afterConnectionClosed(WebSocketSession session,
                                      CloseStatus status) throws Exception {
        sessions.remove(session);
        System.out.println("Connection closed: " + session.getId());

        // 다른 사용자들에게 알림
        broadcast(session.getId() + " left the chat", null);
    }

    // 에러 처리
    @Override
    public void handleTransportError(WebSocketSession session,
                                     Throwable exception) throws Exception {
        System.err.println("Error for session " + session.getId() +
                          ": " + exception.getMessage());
        session.close();
    }

    // 모든 세션에 메시지 브로드캐스트
    private void broadcast(String message, WebSocketSession excludeSession)
            throws Exception {
        TextMessage textMessage = new TextMessage(message);

        for (WebSocketSession session : sessions) {
            if (session.isOpen() && !session.equals(excludeSession)) {
                session.sendMessage(textMessage);
            }
        }
    }
}

// 3. 채팅방 관리 (여러 방 지원)
import java.util.concurrent.ConcurrentHashMap;

public class ChatRoomManager {
    private final Map<String, Set<WebSocketSession>> chatRooms =
        new ConcurrentHashMap<>();

    // 채팅방 참가
    public void joinRoom(String roomId, WebSocketSession session) {
        chatRooms.computeIfAbsent(roomId,
            k -> new CopyOnWriteArraySet<>()).add(session);
    }

    // 채팅방 나가기
    public void leaveRoom(String roomId, WebSocketSession session) {
        Set<WebSocketSession> room = chatRooms.get(roomId);
        if (room != null) {
            room.remove(session);
            if (room.isEmpty()) {
                chatRooms.remove(roomId);
            }
        }
    }

    // 특정 방에 메시지 전송
    public void sendToRoom(String roomId, String message,
                          WebSocketSession excludeSession) throws Exception {
        Set<WebSocketSession> room = chatRooms.get(roomId);
        if (room != null) {
            TextMessage textMessage = new TextMessage(message);
            for (WebSocketSession session : room) {
                if (session.isOpen() && !session.equals(excludeSession)) {
                    session.sendMessage(textMessage);
                }
            }
        }
    }
}
```

**Java WebSocket 클라이언트 구현:**

```java
import org.java_websocket.client.WebSocketClient;
import org.java_websocket.handshake.ServerHandshake;
import java.net.URI;
import java.util.Scanner;

public class ChatWebSocketClient extends WebSocketClient {

    public ChatWebSocketClient(URI serverUri) {
        super(serverUri);
    }

    // 연결 성공 시
    @Override
    public void onOpen(ServerHandshake handshakedata) {
        System.out.println("Connected to server");
        System.out.println("Status: " + handshakedata.getHttpStatus());
    }

    // 메시지 수신 시
    @Override
    public void onMessage(String message) {
        System.out.println("Received: " + message);
    }

    // 연결 종료 시
    @Override
    public void onClose(int code, String reason, boolean remote) {
        System.out.println("Connection closed");
        System.out.println("Code: " + code + ", Reason: " + reason);
    }

    // 에러 발생 시
    @Override
    public void onError(Exception ex) {
        System.err.println("Error: " + ex.getMessage());
        ex.printStackTrace();
    }

    public static void main(String[] args) {
        try {
            // WebSocket 서버 연결
            URI serverUri = new URI("ws://localhost:8080/ws/chat");
            ChatWebSocketClient client = new ChatWebSocketClient(serverUri);

            // 연결 시도
            client.connect();

            // 연결 대기
            while (!client.isOpen()) {
                Thread.sleep(100);
            }

            System.out.println("Type messages to send (type 'exit' to quit):");

            // 메시지 입력 및 전송
            Scanner scanner = new Scanner(System.in);
            while (true) {
                String message = scanner.nextLine();

                if ("exit".equalsIgnoreCase(message)) {
                    break;
                }

                client.send(message);
            }

            // 연결 종료
            client.close();
            scanner.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

**WebSocket의 장단점:**

**장점:**
- ✓ 진정한 양방향 실시간 통신
- ✓ 낮은 레이턴시
- ✓ 낮은 오버헤드 (헤더 최소화)
- ✓ 높은 처리량
- ✓ 서버 푸시 지원

**단점:**
- ✗ 일부 프록시/방화벽에서 차단 가능
- ✗ 연결 유지로 서버 리소스 소모
- ✗ 수평 확장 복잡 (세션 공유 필요)
- ✗ HTTP 캐싱 불가
- ✗ 구형 브라우저 미지원

**사용 사례:**
- 실시간 채팅
- 멀티플레이어 게임
- 라이브 스포츠 업데이트
- 협업 도구 (공동 편집)
- 금융 거래 플랫폼
- IoT 실시간 모니터링

### Server-Sent Events (SSE)

**Server-Sent Events (SSE)**는 서버가 클라이언트에게 실시간 업데이트를 단방향으로 전송하는 웹 표준입니다. HTTP 프로토콜을 기반으로 하여 간단하고 효율적인 서버 푸시 메커니즘을 제공합니다.

**SSE vs WebSocket vs Long Polling:**

| 특징 | SSE | WebSocket | Long Polling |
|------|-----|-----------|--------------|
| **통신 방향** | 단방향 (서버→클라이언트) | 양방향 | 단방향 |
| **프로토콜** | HTTP | WebSocket (ws://) | HTTP |
| **연결** | 지속적 (Keep-alive) | 지속적 | 요청마다 재연결 |
| **재연결** | 자동 | 수동 구현 필요 | 매 요청마다 |
| **데이터 형식** | 텍스트 (UTF-8) | 바이너리/텍스트 | 모든 형식 |
| **브라우저 지원** | 대부분 (IE 제외) | 현대 브라우저 | 모든 브라우저 |
| **구현 복잡도** | 낮음 | 중간 | 낮음 |

**SSE 메시지 형식:**

```
data: 첫 번째 메시지

data: 두 번째 메시지
id: 1

event: userUpdate
data: {"userId": 123, "name": "Alice"}
id: 2

data: 여러 줄
data: 메시지도
data: 가능합니다
id: 3
```

**Java SSE 서버 구현 (Spring Boot):**

```java
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;
import org.springframework.stereotype.Service;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;

@RestController
@RequestMapping("/api/sse")
public class SseController {

    private final SseService sseService;

    public SseController(SseService sseService) {
        this.sseService = sseService;
    }

    // SSE 연결 엔드포인트
    @GetMapping("/notifications")
    public SseEmitter streamNotifications() {
        // 타임아웃: 30분
        SseEmitter emitter = new SseEmitter(30 * 60 * 1000L);

        sseService.addEmitter(emitter);

        // 연결 성공 메시지
        try {
            emitter.send(SseEmitter.event()
                    .name("connected")
                    .data("Connected to notification stream"));
        } catch (IOException e) {
            emitter.completeWithError(e);
        }

        // 완료/타임아웃/에러 시 정리
        emitter.onCompletion(() -> sseService.removeEmitter(emitter));
        emitter.onTimeout(() -> sseService.removeEmitter(emitter));
        emitter.onError(e -> sseService.removeEmitter(emitter));

        return emitter;
    }

    // 알림 전송 (테스트용)
    @PostMapping("/send")
    public Map<String, String> sendNotification(@RequestBody String message) {
        sseService.sendToAll(message);
        return Map.of("status", "sent", "message", message);
    }

    // 특정 이벤트 타입으로 전송
    @PostMapping("/send/{eventType}")
    public Map<String, String> sendEvent(
            @PathVariable String eventType,
            @RequestBody Map<String, Object> data) {
        sseService.sendEventToAll(eventType, data);
        return Map.of("status", "sent", "eventType", eventType);
    }
}

@Service
class SseService {
    private final Set<SseEmitter> emitters = new CopyOnWriteArraySet<>();
    private final ScheduledExecutorService scheduler =
        Executors.newScheduledThreadPool(1);

    public SseService() {
        // 주기적으로 heartbeat 전송 (연결 유지)
        scheduler.scheduleAtFixedRate(
            this::sendHeartbeat,
            0, 15, TimeUnit.SECONDS
        );
    }

    // 새 emitter 추가
    public void addEmitter(SseEmitter emitter) {
        emitters.add(emitter);
        System.out.println("New SSE connection. Total: " + emitters.size());
    }

    // emitter 제거
    public void removeEmitter(SseEmitter emitter) {
        emitters.remove(emitter);
        System.out.println("SSE connection closed. Total: " + emitters.size());
    }

    // 모든 클라이언트에게 메시지 전송
    public void sendToAll(String message) {
        List<SseEmitter> deadEmitters = new ArrayList<>();

        emitters.forEach(emitter -> {
            try {
                emitter.send(SseEmitter.event()
                        .name("message")
                        .data(message)
                        .id(String.valueOf(System.currentTimeMillis())));
            } catch (IOException e) {
                deadEmitters.add(emitter);
            }
        });

        // 끊어진 연결 제거
        deadEmitters.forEach(this::removeEmitter);
    }

    // 이벤트 타입과 함께 전송
    public void sendEventToAll(String eventType, Object data) {
        List<SseEmitter> deadEmitters = new ArrayList<>();

        emitters.forEach(emitter -> {
            try {
                emitter.send(SseEmitter.event()
                        .name(eventType)
                        .data(data)
                        .id(String.valueOf(System.currentTimeMillis())));
            } catch (IOException e) {
                deadEmitters.add(emitter);
            }
        });

        deadEmitters.forEach(this::removeEmitter);
    }

    // Heartbeat 전송 (연결 유지)
    private void sendHeartbeat() {
        List<SseEmitter> deadEmitters = new ArrayList<>();

        emitters.forEach(emitter -> {
            try {
                emitter.send(SseEmitter.event()
                        .comment("heartbeat"));
            } catch (IOException e) {
                deadEmitters.add(emitter);
            }
        });

        deadEmitters.forEach(this::removeEmitter);
    }
}

// 실시간 대시보드 예제
@RestController
@RequestMapping("/api/dashboard")
class DashboardController {

    private final DashboardService dashboardService;

    public DashboardController(DashboardService dashboardService) {
        this.dashboardService = dashboardService;
    }

    @GetMapping("/metrics")
    public SseEmitter streamMetrics() {
        SseEmitter emitter = new SseEmitter(Long.MAX_VALUE);
        dashboardService.addMetricsSubscriber(emitter);
        return emitter;
    }
}

@Service
class DashboardService {
    private final Set<SseEmitter> metricsSubscribers =
        new CopyOnWriteArraySet<>();
    private final ScheduledExecutorService scheduler =
        Executors.newScheduledThreadPool(1);

    public DashboardService() {
        // 1초마다 메트릭 업데이트 전송
        scheduler.scheduleAtFixedRate(
            this::sendMetrics,
            0, 1, TimeUnit.SECONDS
        );
    }

    public void addMetricsSubscriber(SseEmitter emitter) {
        metricsSubscribers.add(emitter);

        emitter.onCompletion(() -> metricsSubscribers.remove(emitter));
        emitter.onTimeout(() -> metricsSubscribers.remove(emitter));
        emitter.onError(e -> metricsSubscribers.remove(emitter));
    }

    private void sendMetrics() {
        Map<String, Object> metrics = Map.of(
            "cpu", Math.random() * 100,
            "memory", Math.random() * 100,
            "activeUsers", (int) (Math.random() * 1000),
            "timestamp", System.currentTimeMillis()
        );

        List<SseEmitter> deadEmitters = new ArrayList<>();

        metricsSubscribers.forEach(emitter -> {
            try {
                emitter.send(SseEmitter.event()
                        .name("metrics")
                        .data(metrics));
            } catch (IOException e) {
                deadEmitters.add(emitter);
            }
        });

        deadEmitters.forEach(metricsSubscribers::remove);
    }
}
```

**JavaScript SSE 클라이언트 (브라우저):**

```javascript
// EventSource API 사용
const eventSource = new EventSource('http://localhost:8080/api/sse/notifications');

// 연결 성공
eventSource.addEventListener('connected', (event) => {
    console.log('Connected:', event.data);
});

// 일반 메시지 수신
eventSource.addEventListener('message', (event) => {
    console.log('Message:', event.data);
    console.log('ID:', event.lastEventId);
});

// 커스텀 이벤트 수신
eventSource.addEventListener('userUpdate', (event) => {
    const data = JSON.parse(event.data);
    console.log('User update:', data);
});

// 에러 처리
eventSource.onerror = (error) => {
    console.error('SSE error:', error);
    if (eventSource.readyState === EventSource.CLOSED) {
        console.log('Connection closed');
    }
};

// 연결 종료
// eventSource.close();
```

**Java SSE 클라이언트 구현:**

```java
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.URI;
import java.io.BufferedReader;
import java.io.StringReader;

public class SseClient {
    private final String serverUrl;
    private final HttpClient httpClient;
    private volatile boolean running = true;

    public SseClient(String serverUrl) {
        this.serverUrl = serverUrl;
        this.httpClient = HttpClient.newHttpClient();
    }

    public void connect() {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(serverUrl))
                .header("Accept", "text/event-stream")
                .GET()
                .build();

        httpClient.sendAsync(request,
                HttpResponse.BodyHandlers.ofLines())
                .thenAccept(response -> {
                    if (response.statusCode() == 200) {
                        processStream(response);
                    } else {
                        System.err.println("Connection failed: " +
                                         response.statusCode());
                    }
                })
                .exceptionally(e -> {
                    System.err.println("Connection error: " + e.getMessage());
                    return null;
                });
    }

    private void processStream(HttpResponse<Stream<String>> response) {
        String eventType = null;
        String eventId = null;
        StringBuilder data = new StringBuilder();

        response.body().forEach(line -> {
            if (!running) return;

            if (line.isEmpty()) {
                // 빈 줄은 메시지 종료
                if (data.length() > 0) {
                    handleEvent(eventType, eventId, data.toString());
                    data.setLength(0);
                    eventType = null;
                    eventId = null;
                }
            } else if (line.startsWith("event:")) {
                eventType = line.substring(6).trim();
            } else if (line.startsWith("id:")) {
                eventId = line.substring(3).trim();
            } else if (line.startsWith("data:")) {
                if (data.length() > 0) {
                    data.append("\n");
                }
                data.append(line.substring(5).trim());
            } else if (line.startsWith(":")) {
                // 주석 (heartbeat 등)
                System.out.println("Comment: " + line.substring(1));
            }
        });
    }

    private void handleEvent(String eventType, String eventId, String data) {
        System.out.println("Event: " + (eventType != null ? eventType : "message"));
        if (eventId != null) {
            System.out.println("ID: " + eventId);
        }
        System.out.println("Data: " + data);
        System.out.println("---");
    }

    public void stop() {
        running = false;
    }

    public static void main(String[] args) throws InterruptedException {
        SseClient client = new SseClient(
            "http://localhost:8080/api/sse/notifications"
        );
        client.connect();

        System.out.println("SSE client started. Press Ctrl+C to stop.");
        Thread.currentThread().join();
    }
}
```

**SSE의 장단점:**

**장점:**
- ✓ 간단한 구현 (HTTP 기반)
- ✓ 자동 재연결 기능
- ✓ 이벤트 ID로 마지막 이벤트부터 재개 가능
- ✓ 텍스트 기반으로 디버깅 용이
- ✓ 방화벽/프록시 통과 용이

**단점:**
- ✗ 단방향 통신만 가능
- ✗ 바이너리 데이터 미지원
- ✗ IE/Edge(구버전) 미지원
- ✗ HTTP/1.1에서 브라우저당 연결 제한 (6개)

**사용 사례:**
- 실시간 알림
- 뉴스피드 업데이트
- 주식 시세 업데이트
- 실시간 대시보드
- 로그 스트리밍
- 진행률 표시

**Long Polling vs WebSocket vs SSE 비교:**

| 사용 사례 | 추천 기술 | 이유 |
|----------|----------|------|
| 실시간 채팅 | WebSocket | 양방향 통신 필요 |
| 알림 시스템 | SSE | 서버→클라이언트 단방향 |
| 멀티플레이어 게임 | WebSocket | 낮은 레이턴시, 양방향 |
| 주식 시세 | SSE | 서버 푸시, 텍스트 데이터 |
| 레거시 브라우저 지원 | Long Polling | 가장 넓은 호환성 |
| 실시간 협업 도구 | WebSocket | 양방향 + 낮은 레이턴시 |

### Circuit Breaker

* [Circuit Breaker Pattern (Design Patterns for Microservices)](https://medium.com/geekculture/design-patterns-for-microservices-circuit-breaker-pattern-276249ffab33)

Circuit Breaker는 분산 시스템 및 마이크로서비스 아키텍처에서 실패를 감지하고 예방하며, 우아하게 처리할 수 있는 디자인 패턴입니다. 이는 시스템의 회복력을 향상시킵니다. 전기 회로 차단기에서 영감을 받은 회로 차단기 패턴은 소프트웨어 시스템에서 서비스나 구성요소가 실패할 수 있음을 감지하고 보호합니다.

일반적으로 회로 차단기 패턴은 원격 서비스에 대한 서비스 호출이나 요청을 래핑하거나 미들웨어로 구현됩니다. 회로 차단기의 주요 목적은 실패, 느린 서비스에 대한 추가 요청이나 작업을 차단하여 관리할 수 있는 방법을 제공하는 것입니다.

**회로 차단기의 생애 주기**

회로 차단기의 생애 주기에는 세 가지 주요 상태가 있습니다.

- **Closed**: closed 상태에서 회로 차단기는 요청을 서비스로 전달합니다. 요청의 성공 여부를 모니터링하고, 실패 횟수나 응답 시간이 설정된 임계 값보다 큰 경우, 회로 차단기는 "open" 상태로 전환합니다.
- **Open**: open 상태에서 회로 차단기는 실패하거나 느린 서비스로의 추가 요청을 차단하고, 즉시 fallback 상태로 응답합니다. 이 상태는 서비스로의 추가 부하를 방지하고 회복을 허용합니다. 설정된 시간이 지나면, 회로 차단기는 "half-open" 상태로 전환합니다.
- **Half-Open**: half-open 상태에서 회로 차단기는 서비스의 건강 상태를 확인하기 위해 일정 수 또는 비율의 요청을 허용합니다. 요청이 성공하고 서비스가 회복되면, 회로 차단기는 "closed" 상태로 되돌아옵니다. 실패가 계속되면, 회로 차단기는 "open" 상태로 되돌아갑니다.

Circuit breaker 패턴을 구현함으로써, 시스템은 연쇄적인 실패를 방지하고, 느린 또는 실패하는 서비스의 영향을 줄이며, 오류 처리를 위한 fallback 메커니즘을 제공하고, 분산 시스템 및 마이크로서비스 아키텍처의 전반적인 회복력과 내결함성을 향상시킬 수 있습니다.

**Java 구현 예제**

Java에서는 Resilience4j 라이브러리를 사용하여 Circuit Breaker를 쉽게 구현할 수 있습니다.

```java
import io.github.resilience4j.circuitbreaker.CircuitBreaker;
import io.github.resilience4j.circuitbreaker.CircuitBreakerConfig;
import io.github.resilience4j.circuitbreaker.CircuitBreakerRegistry;
import java.time.Duration;
import java.util.function.Supplier;

// Circuit Breaker 설정
public class CircuitBreakerExample {

    public static void main(String[] args) {
        // Circuit Breaker 설정 생성
        CircuitBreakerConfig config = CircuitBreakerConfig.custom()
            .failureRateThreshold(50)                    // 실패율 임계값 50%
            .waitDurationInOpenState(Duration.ofMillis(1000))  // Open 상태 지속 시간
            .slidingWindowSize(10)                       // 슬라이딩 윈도우 크기
            .minimumNumberOfCalls(5)                     // 최소 호출 횟수
            .permittedNumberOfCallsInHalfOpenState(3)    // Half-Open 상태에서 허용되는 호출 수
            .build();

        // Circuit Breaker Registry 생성
        CircuitBreakerRegistry registry = CircuitBreakerRegistry.of(config);

        // Circuit Breaker 인스턴스 생성
        CircuitBreaker circuitBreaker = registry.circuitBreaker("paymentService");

        // 원격 서비스 호출을 Circuit Breaker로 래핑
        Supplier<String> decoratedSupplier = CircuitBreaker
            .decorateSupplier(circuitBreaker, CircuitBreakerExample::callExternalService);

        // 서비스 호출
        for (int i = 0; i < 20; i++) {
            try {
                String result = decoratedSupplier.get();
                System.out.println("호출 성공: " + result);
            } catch (Exception e) {
                System.out.println("호출 실패: " + e.getMessage());
            }

            // Circuit Breaker 상태 출력
            System.out.println("Circuit Breaker 상태: " +
                circuitBreaker.getState());

            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    private static String callExternalService() {
        // 외부 서비스 호출 시뮬레이션
        if (Math.random() > 0.6) {
            throw new RuntimeException("서비스 장애 발생");
        }
        return "성공";
    }
}
```

**Spring Boot에서의 Circuit Breaker 구현**

```java
import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

@Service
public class PaymentService {

    private final RestTemplate restTemplate;

    public PaymentService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    @CircuitBreaker(name = "paymentService", fallbackMethod = "fallbackPayment")
    public PaymentResponse processPayment(PaymentRequest request) {
        // 외부 결제 서비스 호출
        String url = "http://payment-service/api/payments";
        return restTemplate.postForObject(url, request, PaymentResponse.class);
    }

    // Fallback 메서드 - Circuit Breaker가 Open 상태일 때 호출됨
    private PaymentResponse fallbackPayment(PaymentRequest request, Exception ex) {
        System.out.println("Circuit Breaker Open - Fallback 실행: " + ex.getMessage());
        return new PaymentResponse("PENDING", "결제 서비스가 일시적으로 사용 불가능합니다.");
    }
}
```

**Circuit Breaker 장점**

| 장점 | 설명 |
|------|------|
| 연쇄 실패 방지 | 하나의 서비스 실패가 전체 시스템으로 전파되는 것을 방지 |
| 빠른 실패 처리 | Open 상태에서 즉시 Fallback 응답을 반환하여 응답 시간 단축 |
| 자동 복구 | Half-Open 상태를 통해 서비스 복구 여부를 자동으로 확인 |
| 리소스 보호 | 실패하는 서비스에 대한 불필요한 호출을 차단하여 리소스 절약 |
| 시스템 안정성 | 일부 서비스 장애에도 전체 시스템의 가용성 유지 |

### Rate Limiter

* [Rate Limiting Fundamentals | bytebytego](https://blog.bytebytego.com/p/rate-limiting-fundamentals)
* [Rate Limiting](https://www.imperva.com/learn/application-security/rate-limiting/)

컴퓨터 시스템, API, 그리고 네트워크에서의 Rate Limiting은 요청이나 데이터 패킷이 처리되는 속도를 제어하는 기술입니다. 특정 시간 간격 내에서 허용되는 요청, 거래, 데이터의 수에 제한을 두어 리소스의 공정한 사용을 보장하고 시스템 안정성을 유지하며 과도한 부하, 남용, 서비스 거부 공격(DoS)으로부터 서비스를 보호하는 것이 목표입니다.

**Rate Limiting 적용 영역**

다양한 단계 및 시스템의 다른 구성 요소에서 Rate Limiting을 구현할 수 있습니다:

- **API 및 웹 서비스**: 응용 프로그램 수준에서 Rate Limiting을 적용하여 클라이언트가 특정 시간 동안 API 또는 웹 서비스에 보낼 수 있는 요청 수를 제어할 수 있습니다. 초당, 분당, 시간당 요청 수를 제한하는 것이 일반적이며, 클라이언트를 식별하고 추적하기 위해 토큰이나 API 키를 사용합니다.
- **데이터베이스 및 백엔드 서비스**: Rate Limiting을 적용하여 데이터베이스, 메시지 큐 또는 캐싱 시스템과 같은 백엔드 서비스에 의해 소비되는 리소스를 관리하여 가용 용량을 과다하게 로드하거나 소진하는 것을 방지할 수 있습니다.
- **네트워크**: 네트워크 수준에서 Rate Limiting을 구현하여 대역폭 사용률을 제어하고, 네트워크 혼잡을 방지하며, 클라이언트나 장치 간에 네트워크 리소스의 공정한 분배를 보장할 수 있습니다.

**주요 Rate Limiting 알고리즘**

- **Token Bucket**: 이 방법에서는 고정된 속도로 토큰을 버킷에 추가하되 최대 용량까지만 추가합니다. 각 요청이나 패킷은 버킷의 토큰을 소비합니다. 버킷이 비어 있으면 요청이 거부되거나 토큰을 사용할 수 있을 때까지 지연됩니다.
- **Leaky Bucket**: 이 알고리즘은 고정 크기의 버퍼(버킷)을 사용하며, 상수 속도로 버킷의 아이템이 제거됩니다. 공간이 있다면 들어오는 요청이나 패킷이 버퍼에 추가되고, 아니면 거부되거나 지연됩니다.
- **Fixed Window**: 이 알고리즘은 시간을 고정된 크기의 창이나 간격으로 나누어 각 창에서의 요청이나 패킷 수를 추적합니다. 창이 최대 허용된 건수에 도달하면 추가 요청이나 패킷이 거부되거나 다음 창이 시작될 때까지 지연됩니다.
- **Sliding Window**: 이 접근법은 고정 창 알고리즘을 개선하여 요청 타임스탬프에 기반한 점진적 시간 창을 사용함으로써 더 나은 공정성과 부드러운 제한률을 보장합니다.

Rate Limiting을 효과적으로 구현함으로써 컴퓨터 시스템, API, 네트워크의 신뢰성, 성능, 보안을 유지할 수 있고 클라이언트, 사용자, 장치 간에 공정한 사용 정책을 적용할 수 있습니다.

**Java 구현 예제 - Token Bucket 알고리즘**

```java
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class TokenBucketRateLimiter {

    private final int maxTokens;              // 최대 토큰 수
    private final int refillRate;             // 초당 리필되는 토큰 수
    private final AtomicInteger tokens;       // 현재 토큰 수
    private final ScheduledExecutorService scheduler;

    public TokenBucketRateLimiter(int maxTokens, int refillRate) {
        this.maxTokens = maxTokens;
        this.refillRate = refillRate;
        this.tokens = new AtomicInteger(maxTokens);
        this.scheduler = Executors.newScheduledThreadPool(1);

        // 주기적으로 토큰 리필
        scheduler.scheduleAtFixedRate(this::refillTokens, 1, 1, TimeUnit.SECONDS);
    }

    private void refillTokens() {
        int currentTokens = tokens.get();
        if (currentTokens < maxTokens) {
            int newTokens = Math.min(maxTokens, currentTokens + refillRate);
            tokens.set(newTokens);
            System.out.println("토큰 리필: " + newTokens);
        }
    }

    public boolean tryConsume(int tokensToConsume) {
        while (true) {
            int currentTokens = tokens.get();
            if (currentTokens >= tokensToConsume) {
                if (tokens.compareAndSet(currentTokens, currentTokens - tokensToConsume)) {
                    return true;
                }
            } else {
                return false;
            }
        }
    }

    public void shutdown() {
        scheduler.shutdown();
    }

    public static void main(String[] args) throws InterruptedException {
        // 최대 10개 토큰, 초당 2개 리필
        TokenBucketRateLimiter rateLimiter = new TokenBucketRateLimiter(10, 2);

        // 요청 시뮬레이션
        for (int i = 0; i < 20; i++) {
            boolean allowed = rateLimiter.tryConsume(1);
            System.out.println("요청 #" + (i + 1) + ": " +
                (allowed ? "허용됨" : "거부됨 (Rate Limit 초과)"));
            Thread.sleep(300);
        }

        rateLimiter.shutdown();
    }
}
```

**Spring Boot에서의 Rate Limiter 구현 (Guava RateLimiter 사용)**

```java
import com.google.common.util.concurrent.RateLimiter;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class RateLimitedController {

    // 초당 10개의 요청 허용
    private final RateLimiter rateLimiter = RateLimiter.create(10.0);

    @GetMapping("/data")
    public ResponseEntity<String> getData() {
        // Rate Limiter 확인 (타임아웃 없이)
        if (!rateLimiter.tryAcquire()) {
            return ResponseEntity
                .status(HttpStatus.TOO_MANY_REQUESTS)
                .body("Rate limit exceeded. Please try again later.");
        }

        // 정상 처리
        return ResponseEntity.ok("데이터 조회 성공");
    }
}
```

**Bucket4j를 사용한 고급 Rate Limiter 구현**

```java
import io.github.bucket4j.Bandwidth;
import io.github.bucket4j.Bucket;
import io.github.bucket4j.Bucket4j;
import io.github.bucket4j.Refill;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletRequest;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@RestController
@RequestMapping("/api")
public class AdvancedRateLimitedController {

    // 사용자별 Rate Limiter 관리
    private final Map<String, Bucket> cache = new ConcurrentHashMap<>();

    private Bucket resolveBucket(String apiKey) {
        return cache.computeIfAbsent(apiKey, key -> {
            // 1분당 100개 요청 허용
            Bandwidth limit = Bandwidth.classic(100, Refill.intervally(100, Duration.ofMinutes(1)));
            return Bucket4j.builder()
                .addLimit(limit)
                .build();
        });
    }

    @GetMapping("/users")
    public ResponseEntity<String> getUsers(
            @RequestHeader(value = "X-API-Key", required = false) String apiKey,
            HttpServletRequest request) {

        // API Key 검증
        if (apiKey == null || apiKey.isEmpty()) {
            apiKey = request.getRemoteAddr(); // IP 주소를 키로 사용
        }

        Bucket bucket = resolveBucket(apiKey);

        // 토큰 소비 시도
        if (bucket.tryConsume(1)) {
            long availableTokens = bucket.getAvailableTokens();
            return ResponseEntity.ok()
                .header("X-Rate-Limit-Remaining", String.valueOf(availableTokens))
                .body("사용자 목록 조회 성공");
        }

        // Rate Limit 초과
        return ResponseEntity
            .status(HttpStatus.TOO_MANY_REQUESTS)
            .header("X-Rate-Limit-Retry-After-Seconds", "60")
            .body("요청 한도를 초과했습니다. 1분 후에 다시 시도해주세요.");
    }
}
```

**Rate Limiting 알고리즘 비교**

| 알고리즘 | 장점 | 단점 | 사용 사례 |
|---------|------|------|----------|
| Token Bucket | 버스트 트래픽 허용, 유연한 속도 제어 | 복잡한 구현 | API Gateway, 일반적인 Rate Limiting |
| Leaky Bucket | 일정한 출력 속도 보장 | 버스트 트래픽 처리 불가 | 네트워크 트래픽 제어 |
| Fixed Window | 간단한 구현 | 경계에서 버스트 가능 | 간단한 API Rate Limiting |
| Sliding Window | 정확한 Rate Limiting | 메모리 사용량 높음 | 정밀한 제어가 필요한 경우 |

### Asynchronism

<p align="center">
  <img src="http://i.imgur.com/54GYsSx.png"/>
  <br/>
  <i><a href=http://lethain.com/introduction-to-architecting-systems-for-scale/#platform_layer>Source: Intro to architecting systems for scale</a></i>
</p>

비동기 처리(Asynchronism)는 시스템의 확장성과 성능을 향상시키는 핵심 기술입니다. 비동기 처리를 통해 시간이 오래 걸리는 작업을 백그라운드에서 처리하고, 사용자에게 빠른 응답을 제공할 수 있습니다.

**비동기 처리의 주요 구성 요소**

- **Message Queues**: Redis, RabbitMQ, Amazon SQS와 같은 메시지 큐를 사용하여 작업을 비동기적으로 처리
- **Task Queues**: Celery와 같은 작업 큐를 사용하여 분산 작업 처리
- **Back Pressure**: MQ가 바쁘면 client에게 503 Service Unavailable을 줘서 시스템의 성능 저하를 예방합니다. 일종의 Circuit Breaker와 같습니다.

**비동기 처리 패턴**

1. **작업 큐 패턴 (Task Queue Pattern)**
   - 시간이 오래 걸리는 작업을 큐에 넣고 별도의 워커가 처리
   - 사용자는 즉시 응답을 받고, 작업은 백그라운드에서 처리됨

2. **이벤트 기반 패턴 (Event-Driven Pattern)**
   - 이벤트가 발생하면 해당 이벤트를 구독하는 핸들러가 비동기적으로 처리
   - 느슨한 결합(Loose Coupling)과 높은 확장성 제공

**Java에서의 비동기 처리 구현**

**1. CompletableFuture를 사용한 비동기 처리**

```java
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class AsyncExample {

    private static final ExecutorService executor = Executors.newFixedThreadPool(10);

    public static void main(String[] args) {
        System.out.println("메인 스레드 시작: " + Thread.currentThread().getName());

        // 비동기 작업 1: 사용자 정보 조회
        CompletableFuture<String> userFuture = CompletableFuture.supplyAsync(() -> {
            System.out.println("사용자 조회 중: " + Thread.currentThread().getName());
            sleep(2000);
            return "User: John Doe";
        }, executor);

        // 비동기 작업 2: 주문 정보 조회
        CompletableFuture<String> orderFuture = CompletableFuture.supplyAsync(() -> {
            System.out.println("주문 조회 중: " + Thread.currentThread().getName());
            sleep(1500);
            return "Order: #12345";
        }, executor);

        // 비동기 작업 3: 결제 정보 조회
        CompletableFuture<String> paymentFuture = CompletableFuture.supplyAsync(() -> {
            System.out.println("결제 조회 중: " + Thread.currentThread().getName());
            sleep(1000);
            return "Payment: Card ****1234";
        }, executor);

        // 모든 작업 완료 대기 및 결합
        CompletableFuture<String> combinedFuture = userFuture
            .thenCombine(orderFuture, (user, order) -> user + ", " + order)
            .thenCombine(paymentFuture, (prev, payment) -> prev + ", " + payment);

        // 결과 출력
        combinedFuture.thenAccept(result -> {
            System.out.println("\n최종 결과: " + result);
            System.out.println("처리 완료: " + Thread.currentThread().getName());
        });

        // 메인 스레드는 계속 진행
        System.out.println("메인 스레드 계속 진행...\n");

        // 작업 완료 대기
        combinedFuture.join();
        executor.shutdown();
    }

    private static void sleep(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
```

**2. Spring @Async를 사용한 비동기 처리**

```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

import java.util.concurrent.Executor;

@Configuration
@EnableAsync
public class AsyncConfiguration {

    @Bean(name = "taskExecutor")
    public Executor taskExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(5);
        executor.setMaxPoolSize(10);
        executor.setQueueCapacity(100);
        executor.setThreadNamePrefix("async-");
        executor.initialize();
        return executor;
    }
}
```

```java
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.util.concurrent.CompletableFuture;

@Service
public class AsyncService {

    @Async("taskExecutor")
    public CompletableFuture<String> processOrder(String orderId) {
        System.out.println("주문 처리 시작: " + orderId + " on " +
            Thread.currentThread().getName());

        // 시간이 오래 걸리는 작업 시뮬레이션
        try {
            Thread.sleep(3000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        String result = "주문 " + orderId + " 처리 완료";
        System.out.println(result);

        return CompletableFuture.completedFuture(result);
    }

    @Async("taskExecutor")
    public CompletableFuture<String> sendNotification(String userId) {
        System.out.println("알림 전송 시작: " + userId + " on " +
            Thread.currentThread().getName());

        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        String result = "사용자 " + userId + "에게 알림 전송 완료";
        System.out.println(result);

        return CompletableFuture.completedFuture(result);
    }
}
```

```java
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.concurrent.CompletableFuture;

@RestController
@RequestMapping("/api")
public class AsyncController {

    private final AsyncService asyncService;

    public AsyncController(AsyncService asyncService) {
        this.asyncService = asyncService;
    }

    @PostMapping("/orders")
    public ResponseEntity<String> createOrder(@RequestBody OrderRequest request) {
        // 즉시 응답 반환
        asyncService.processOrder(request.getOrderId());
        asyncService.sendNotification(request.getUserId());

        return ResponseEntity.accepted()
            .body("주문이 접수되었습니다. 처리 중입니다.");
    }

    @GetMapping("/orders/{orderId}/status")
    public CompletableFuture<ResponseEntity<String>> getOrderStatus(
            @PathVariable String orderId) {

        return asyncService.processOrder(orderId)
            .thenApply(result -> ResponseEntity.ok(result));
    }
}
```

**3. RabbitMQ를 사용한 비동기 메시징**

```java
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.stereotype.Service;

@Service
public class OrderProducer {

    private final RabbitTemplate rabbitTemplate;
    private static final String EXCHANGE = "order.exchange";
    private static final String ROUTING_KEY = "order.created";

    public OrderProducer(RabbitTemplate rabbitTemplate) {
        this.rabbitTemplate = rabbitTemplate;
    }

    public void sendOrderMessage(Order order) {
        System.out.println("주문 메시지 전송: " + order.getId());
        rabbitTemplate.convertAndSend(EXCHANGE, ROUTING_KEY, order);
        System.out.println("메시지 전송 완료 - 즉시 반환");
    }
}
```

```java
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.stereotype.Component;

@Component
public class OrderConsumer {

    @RabbitListener(queues = "order.queue")
    public void handleOrderMessage(Order order) {
        System.out.println("주문 메시지 수신: " + order.getId() + " on " +
            Thread.currentThread().getName());

        // 시간이 오래 걸리는 처리
        try {
            Thread.sleep(5000);
            processOrder(order);
            System.out.println("주문 처리 완료: " + order.getId());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.err.println("주문 처리 실패: " + order.getId());
        }
    }

    private void processOrder(Order order) {
        // 주문 처리 로직
        System.out.println("주문 재고 확인, 결제 처리, 배송 준비...");
    }
}
```

**비동기 처리의 장단점**

| 장점 | 단점 |
|------|------|
| 사용자 응답 시간 단축 | 디버깅 복잡도 증가 |
| 시스템 처리량 증가 | 에러 처리 복잡 |
| 리소스 효율적 사용 | 작업 순서 보장 어려움 |
| 확장성 향상 | 모니터링 복잡 |
| 시스템 결합도 감소 | 트랜잭션 관리 어려움 |

### Message Queue

- [System Design — Message Queues](https://medium.com/must-know-computer-science/system-design-message-queues-245612428a22)

Message Queue(메시지 큐)를 통해 응용 프로그램은 비동기적으로 통신할 수 있습니다. 큐를 사용하여 서로에게 메시지를 보내는 것으로, 보내는 프로그램과 받는 프로그램 사이의 일시적 저장소를 제공해서 연결되지 않거나 바쁠 때 중단 없이 동작할 수 있도록 합니다. 큐의 기본 구조는 프로듀서(메시지 생성자)가 메시지 큐에 전달할 메시지를 생성하는 몇 가지 클라이언트 애플리케이션과 메시지를 처리하는 소비자 애플리케이션이 있습니다. 큐에 배치된 메시지는 소비자가 검색할 때까지 저장됩니다.

메시지 큐는 마이크로서비스 아키텍처에서도 중요한 역할을 담당합니다. 서로 다른 서비스에서 기능이 분산되며, 전체 소프트웨어 어플리케이션을 구성하기 위해 병합됩니다. 이때 상호 종속성이 발생하며 시스템에 서비스 간 비블록 응답 없이 서로 연결되는 메커니즘이 필요합니다. 메시지 큐는 서비스가 비동기적으로 큐에 메시지를 푸시하고 올바른 목적지에 전달되도록 하는 수단을 제공하여 이 목적을 달성합니다. 서비스 간 메시지 큐를 구현하려면 메시지 브로커(예: RabbitMQ, Kafka)가 필요합니다.

**Message Queue의 주요 특징**

- **비동기 통신**: Producer와 Consumer가 독립적으로 동작
- **느슨한 결합**: 서비스 간 직접적인 의존성 제거
- **부하 분산**: 여러 Consumer가 메시지를 분산 처리
- **내구성**: 메시지를 디스크에 저장하여 시스템 장애 시에도 손실 방지
- **순서 보장**: FIFO(First-In-First-Out) 방식으로 메시지 처리

**Java에서 RabbitMQ를 사용한 Message Queue 구현**

**1. RabbitMQ 설정**

```java
import org.springframework.amqp.core.*;
import org.springframework.amqp.rabbit.connection.ConnectionFactory;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.amqp.support.converter.Jackson2JsonMessageConverter;
import org.springframework.amqp.support.converter.MessageConverter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class RabbitMQConfig {

    public static final String QUEUE_NAME = "order.queue";
    public static final String EXCHANGE_NAME = "order.exchange";
    public static final String ROUTING_KEY = "order.routing.key";

    public static final String DLQ_NAME = "order.dlq";
    public static final String DLX_NAME = "order.dlx";

    @Bean
    public Queue queue() {
        return QueueBuilder.durable(QUEUE_NAME)
            .withArgument("x-dead-letter-exchange", DLX_NAME)
            .withArgument("x-dead-letter-routing-key", "dlq")
            .withArgument("x-message-ttl", 60000) // 메시지 TTL 60초
            .build();
    }

    @Bean
    public Queue deadLetterQueue() {
        return QueueBuilder.durable(DLQ_NAME).build();
    }

    @Bean
    public DirectExchange exchange() {
        return new DirectExchange(EXCHANGE_NAME);
    }

    @Bean
    public DirectExchange deadLetterExchange() {
        return new DirectExchange(DLX_NAME);
    }

    @Bean
    public Binding binding(Queue queue, DirectExchange exchange) {
        return BindingBuilder.bind(queue)
            .to(exchange)
            .with(ROUTING_KEY);
    }

    @Bean
    public Binding deadLetterBinding(Queue deadLetterQueue, DirectExchange deadLetterExchange) {
        return BindingBuilder.bind(deadLetterQueue)
            .to(deadLetterExchange)
            .with("dlq");
    }

    @Bean
    public MessageConverter jsonMessageConverter() {
        return new Jackson2JsonMessageConverter();
    }

    @Bean
    public RabbitTemplate rabbitTemplate(ConnectionFactory connectionFactory) {
        RabbitTemplate template = new RabbitTemplate(connectionFactory);
        template.setMessageConverter(jsonMessageConverter());
        return template;
    }
}
```

**2. Message Producer (메시지 생성자)**

```java
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.UUID;

@Service
public class OrderMessageProducer {

    private final RabbitTemplate rabbitTemplate;

    public OrderMessageProducer(RabbitTemplate rabbitTemplate) {
        this.rabbitTemplate = rabbitTemplate;
    }

    public void sendOrderCreatedMessage(OrderMessage order) {
        System.out.println("주문 메시지 전송: " + order.getOrderId() +
            " at " + LocalDateTime.now());

        rabbitTemplate.convertAndSend(
            RabbitMQConfig.EXCHANGE_NAME,
            RabbitMQConfig.ROUTING_KEY,
            order
        );

        System.out.println("메시지 전송 완료");
    }

    public void sendBulkOrders(int count) {
        for (int i = 0; i < count; i++) {
            OrderMessage order = new OrderMessage(
                UUID.randomUUID().toString(),
                "customer-" + i,
                "Product-" + i,
                100.0 * (i + 1)
            );
            sendOrderCreatedMessage(order);
        }
    }
}
```

**3. Message Consumer (메시지 소비자)**

```java
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.stereotype.Component;

import java.time.LocalDateTime;

@Component
public class OrderMessageConsumer {

    @RabbitListener(queues = RabbitMQConfig.QUEUE_NAME)
    public void handleOrderMessage(OrderMessage order) {
        System.out.println("\n=== 주문 메시지 수신 ===");
        System.out.println("시간: " + LocalDateTime.now());
        System.out.println("주문 ID: " + order.getOrderId());
        System.out.println("고객 ID: " + order.getCustomerId());
        System.out.println("상품: " + order.getProductName());
        System.out.println("금액: $" + order.getAmount());

        try {
            // 주문 처리 시뮬레이션
            processOrder(order);
            System.out.println("주문 처리 완료: " + order.getOrderId());
        } catch (Exception e) {
            System.err.println("주문 처리 실패: " + e.getMessage());
            throw new RuntimeException("주문 처리 중 오류 발생", e);
        }
    }

    private void processOrder(OrderMessage order) throws InterruptedException {
        // 재고 확인
        System.out.println("  1. 재고 확인 중...");
        Thread.sleep(500);

        // 결제 처리
        System.out.println("  2. 결제 처리 중...");
        Thread.sleep(1000);

        // 배송 준비
        System.out.println("  3. 배송 준비 중...");
        Thread.sleep(500);

        System.out.println("  ✓ 모든 처리 완료");
    }
}
```

**4. Dead Letter Queue 처리**

```java
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.stereotype.Component;

@Component
public class DeadLetterQueueConsumer {

    @RabbitListener(queues = RabbitMQConfig.DLQ_NAME)
    public void handleDeadLetterMessage(OrderMessage order) {
        System.err.println("\n=== Dead Letter Queue 메시지 수신 ===");
        System.err.println("실패한 주문 ID: " + order.getOrderId());
        System.err.println("고객 ID: " + order.getCustomerId());

        // 실패 메시지 처리 로직
        // 1. 관리자에게 알림
        // 2. 데이터베이스에 실패 로그 저장
        // 3. 재시도 큐에 추가

        System.err.println("실패 메시지 로깅 완료");
    }
}
```

**5. Message DTO**

```java
import java.io.Serializable;

public class OrderMessage implements Serializable {
    private String orderId;
    private String customerId;
    private String productName;
    private Double amount;

    public OrderMessage() {}

    public OrderMessage(String orderId, String customerId, String productName, Double amount) {
        this.orderId = orderId;
        this.customerId = customerId;
        this.productName = productName;
        this.amount = amount;
    }

    // Getters and Setters
    public String getOrderId() { return orderId; }
    public void setOrderId(String orderId) { this.orderId = orderId; }

    public String getCustomerId() { return customerId; }
    public void setCustomerId(String customerId) { this.customerId = customerId; }

    public String getProductName() { return productName; }
    public void setProductName(String productName) { this.productName = productName; }

    public Double getAmount() { return amount; }
    public void setAmount(Double amount) { this.amount = amount; }
}
```

**Message Queue 사용 사례**

| 사용 사례 | 설명 | 예시 |
|----------|------|------|
| 작업 큐 | 시간이 오래 걸리는 작업을 백그라운드에서 처리 | 이미지 리사이징, 비디오 인코딩 |
| 서비스 간 통신 | 마이크로서비스 간 비동기 통신 | 주문 서비스 → 결제 서비스 → 배송 서비스 |
| 이벤트 브로드캐스팅 | 하나의 이벤트를 여러 서비스에 전달 | 사용자 가입 → 이메일, SMS, 푸시 알림 |
| 부하 평준화 | 트래픽 스파이크를 큐로 완충 | 블랙프라이데이 주문 처리 |
| 로그 수집 | 분산 시스템의 로그를 중앙 집중화 | 애플리케이션 로그 → 로그 분석 시스템 |

**Message Queue의 장점**

- **확장성**: Consumer를 추가하여 수평 확장 가능
- **내결함성**: 메시지 지속성으로 시스템 장애에도 메시지 보존
- **유연성**: Producer와 Consumer의 독립적인 배포 및 업데이트
- **피크 부하 처리**: 큐가 버퍼 역할을 하여 트래픽 스파이크 흡수

### Message Queue VS Event Streaming Platform

Message Queue와 Event Streaming Platform은 모두 비동기 메시징을 제공하지만, 사용 목적과 특성이 다릅니다.

**Event Streaming Platform의 특징**

- **Long data retention**: 메시지를 장기간 보관 (일, 주, 월 단위)
- **Repeated consumption of messages**: 같은 메시지를 여러 번 읽을 수 있음
- **이벤트 로그**: 모든 이벤트를 시간 순서대로 저장
- **실시간 스트림 처리**: 대용량 데이터를 실시간으로 처리

**Event Streaming Platform 종류**

- [Kafka](/kafka/README.md)
- Apache Pulsar

**Message Queue의 특징**

- **Short data retention**: 메시지를 메모리에 짧은 시간만 보관
- **One-time consumption of messages**: 메시지를 한 번 소비하면 큐에서 제거
- **작업 큐**: 처리해야 할 작업을 전달
- **포인트-투-포인트 통신**: 각 메시지는 하나의 Consumer만 처리

**Message Queue 종류**

- [NATS](/nats/README.md)
- RocketMQ
- ActiveMQ
- RabbitMQ
- ZeroMQ

**중요**: Message Queue와 Event Streaming Platform의 경계는 흐릿해지고 있습니다. RabbitMQ 역시 Long data retention, repeated consumption이 가능합니다.

**비교 테이블**

| 특징 | Message Queue | Event Streaming Platform |
|------|--------------|-------------------------|
| 데이터 보존 | 짧음 (초~분) | 길음 (일~월) |
| 메시지 소비 | 1회 (소비 후 삭제) | 여러 번 (offset 기반) |
| 처리 패러다임 | 작업 큐 | 이벤트 로그 |
| 메시지 순서 | 보장 (Queue 단위) | 보장 (Partition 단위) |
| 확장성 | 중간 | 매우 높음 |
| 지연 시간 | 낮음 | 낮음~중간 |
| 복잡도 | 낮음 | 높음 |
| 스토리지 | 메모리 중심 | 디스크 중심 |
| 재처리 | 어려움 | 쉬움 (offset 이동) |
| 사용 사례 | 작업 큐, RPC | 이벤트 소싱, 로그 수집, 실시간 분석 |

**Kafka (Event Streaming Platform) Java 예제**

```java
import org.apache.kafka.clients.producer.*;
import org.apache.kafka.clients.consumer.*;
import org.apache.kafka.common.serialization.StringDeserializer;
import org.apache.kafka.common.serialization.StringSerializer;

import java.time.Duration;
import java.util.Collections;
import java.util.Properties;

// Kafka Producer
public class KafkaProducerExample {

    public static void main(String[] args) {
        Properties props = new Properties();
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        props.put(ProducerConfig.ACKS_CONFIG, "all");  // 모든 replica 확인

        KafkaProducer<String, String> producer = new KafkaProducer<>(props);

        try {
            for (int i = 0; i < 10; i++) {
                String key = "key-" + i;
                String value = "message-" + i;

                ProducerRecord<String, String> record =
                    new ProducerRecord<>("orders", key, value);

                // 비동기 전송
                producer.send(record, (metadata, exception) -> {
                    if (exception == null) {
                        System.out.printf("메시지 전송 성공 - Topic: %s, Partition: %d, Offset: %d%n",
                            metadata.topic(), metadata.partition(), metadata.offset());
                    } else {
                        System.err.println("메시지 전송 실패: " + exception.getMessage());
                    }
                });
            }
        } finally {
            producer.flush();
            producer.close();
        }
    }
}

// Kafka Consumer
public class KafkaConsumerExample {

    public static void main(String[] args) {
        Properties props = new Properties();
        props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        props.put(ConsumerConfig.GROUP_ID_CONFIG, "order-consumer-group");
        props.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        props.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        props.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");  // 처음부터 읽기
        props.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "false");  // 수동 커밋

        KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
        consumer.subscribe(Collections.singletonList("orders"));

        try {
            while (true) {
                ConsumerRecords<String, String> records =
                    consumer.poll(Duration.ofMillis(100));

                for (ConsumerRecord<String, String> record : records) {
                    System.out.printf("메시지 수신 - Key: %s, Value: %s, Partition: %d, Offset: %d%n",
                        record.key(), record.value(), record.partition(), record.offset());

                    // 메시지 처리
                    processMessage(record.value());
                }

                // 수동 커밋
                consumer.commitSync();
            }
        } finally {
            consumer.close();
        }
    }

    private static void processMessage(String message) {
        System.out.println("메시지 처리 중: " + message);
    }
}
```

**선택 가이드**

**Message Queue를 선택해야 하는 경우:**
- 작업 큐가 필요한 경우 (이메일 전송, 이미지 처리 등)
- 메시지를 한 번만 처리하면 되는 경우
- 간단한 포인트-투-포인트 통신이 필요한 경우
- 낮은 지연 시간이 중요한 경우

**Event Streaming Platform을 선택해야 하는 경우:**
- 이벤트 소싱이 필요한 경우
- 데이터를 장기간 보관하고 재처리해야 하는 경우
- 실시간 스트림 처리 및 분석이 필요한 경우
- 여러 Consumer가 같은 데이터를 독립적으로 읽어야 하는 경우
- 매우 높은 처리량이 필요한 경우 (초당 수백만 메시지)

## 위치 기반 서비스 (Location-Based Services)

### Geohashing

**Geohashing**은 지리적 좌표(위도와 경도)를 지구 표면의 정의된 영역이나 그리드 셀을 나타내는 짧은 문자열로 인코딩하는 기술입니다. 2008년 Gustavo Niemeyer가 개발했으며, 계층적 구조를 가지고 있어 접두사가 더 큰 영역을 나타내고 문자를 추가할수록 정밀도가 높아집니다.

**Geohash 구조:**

```
예시: "9q8yy" (샌프란시스코 근처)

길이별 정밀도:
1자: ±2500 km
2자: ±630 km
3자: ±78 km
4자: ±20 km
5자: ±2.4 km
6자: ±610 m
7자: ±76 m
8자: ±19 m
```

**Geohashing의 특징:**

1. **계층적 구조**: 접두사가 같으면 가까운 위치
2. **문자열 기반**: 데이터베이스 인덱싱 용이
3. **근접 검색**: 같은 접두사로 빠른 검색
4. **프라이버시**: 정확한 좌표 노출 없이 대략적 위치 공유

**Java Geohashing 구현:**

```java
public class Geohash {
    private static final String BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz";
    private static final int[] BITS = {16, 8, 4, 2, 1};

    // 위도, 경도를 Geohash로 인코딩
    public static String encode(double latitude, double longitude, int precision) {
        double[] latRange = {-90.0, 90.0};
        double[] lonRange = {-180.0, 180.0};

        StringBuilder geohash = new StringBuilder();
        boolean isEven = true;
        int bit = 0;
        int ch = 0;

        while (geohash.length() < precision) {
            double mid;

            if (isEven) {
                // 경도 처리
                mid = (lonRange[0] + lonRange[1]) / 2;
                if (longitude > mid) {
                    ch |= BITS[bit];
                    lonRange[0] = mid;
                } else {
                    lonRange[1] = mid;
                }
            } else {
                // 위도 처리
                mid = (latRange[0] + latRange[1]) / 2;
                if (latitude > mid) {
                    ch |= BITS[bit];
                    latRange[0] = mid;
                } else {
                    latRange[1] = mid;
                }
            }

            isEven = !isEven;

            if (bit < 4) {
                bit++;
            } else {
                geohash.append(BASE32.charAt(ch));
                bit = 0;
                ch = 0;
            }
        }

        return geohash.toString();
    }

    // Geohash를 위도, 경도로 디코딩
    public static double[] decode(String geohash) {
        double[] latRange = {-90.0, 90.0};
        double[] lonRange = {-180.0, 180.0};
        boolean isEven = true;

        for (int i = 0; i < geohash.length(); i++) {
            int cd = BASE32.indexOf(geohash.charAt(i));

            for (int j = 0; j < 5; j++) {
                int mask = BITS[j];

                if (isEven) {
                    // 경도 처리
                    double mid = (lonRange[0] + lonRange[1]) / 2;
                    if ((cd & mask) != 0) {
                        lonRange[0] = mid;
                    } else {
                        lonRange[1] = mid;
                    }
                } else {
                    // 위도 처리
                    double mid = (latRange[0] + latRange[1]) / 2;
                    if ((cd & mask) != 0) {
                        latRange[0] = mid;
                    } else {
                        latRange[1] = mid;
                    }
                }

                isEven = !isEven;
            }
        }

        double latitude = (latRange[0] + latRange[1]) / 2;
        double longitude = (lonRange[0] + lonRange[1]) / 2;

        return new double[]{latitude, longitude};
    }

    // 인접한 Geohash 계산
    public static String[] getNeighbors(String geohash) {
        return new String[]{
            getNeighbor(geohash, "top"),
            getNeighbor(geohash, "bottom"),
            getNeighbor(geohash, "left"),
            getNeighbor(geohash, "right"),
            getNeighbor(geohash, "top-left"),
            getNeighbor(geohash, "top-right"),
            getNeighbor(geohash, "bottom-left"),
            getNeighbor(geohash, "bottom-right")
        };
    }

    private static String getNeighbor(String geohash, String direction) {
        // 방향별 이웃 계산 로직 (복잡하므로 라이브러리 사용 권장)
        // 여기서는 간단한 구현 예시
        return geohash; // 실제로는 방향에 따라 계산
    }

    public static void main(String[] args) {
        // 인코딩 예제
        double lat = 37.7749;  // 샌프란시스코 위도
        double lon = -122.4194; // 샌프란시스코 경도

        String geohash = encode(lat, lon, 7);
        System.out.println("Geohash: " + geohash); // "9q8yy9m"

        // 디코딩 예제
        double[] coords = decode(geohash);
        System.out.println("Latitude: " + coords[0]);
        System.out.println("Longitude: " + coords[1]);

        // 다양한 정밀도
        System.out.println("Precision 4: " + encode(lat, lon, 4)); // "9q8y"
        System.out.println("Precision 5: " + encode(lat, lon, 5)); // "9q8yy"
        System.out.println("Precision 6: " + encode(lat, lon, 6)); // "9q8yy9"
    }
}
```

**Geohash 기반 근접 검색:**

```java
import java.util.*;

public class LocationService {
    // 위치 데이터 저장 (DB 대신 메모리 사용)
    private Map<String, List<Location>> geohashIndex = new HashMap<>();

    static class Location {
        String id;
        String name;
        double latitude;
        double longitude;
        String geohash;

        public Location(String id, String name, double lat, double lon) {
            this.id = id;
            this.name = name;
            this.latitude = lat;
            this.longitude = lon;
            this.geohash = Geohash.encode(lat, lon, 6);
        }
    }

    // 위치 추가
    public void addLocation(Location location) {
        geohashIndex.computeIfAbsent(location.geohash,
            k -> new ArrayList<>()).add(location);
    }

    // 근처 위치 검색 (Geohash 기반)
    public List<Location> findNearby(double latitude, double longitude,
                                     int radiusKm) {
        String centerGeohash = Geohash.encode(latitude, longitude, 6);
        Set<Location> results = new HashSet<>();

        // 중심 Geohash와 인접한 Geohash 검색
        searchGeohash(centerGeohash, results);

        // 인접한 셀도 검색
        String[] neighbors = Geohash.getNeighbors(centerGeohash);
        for (String neighbor : neighbors) {
            searchGeohash(neighbor, results);
        }

        // 실제 거리로 필터링
        List<Location> filtered = new ArrayList<>();
        for (Location loc : results) {
            double distance = calculateDistance(
                latitude, longitude,
                loc.latitude, loc.longitude
            );
            if (distance <= radiusKm) {
                filtered.add(loc);
            }
        }

        return filtered;
    }

    private void searchGeohash(String geohash, Set<Location> results) {
        // 정확히 일치하는 geohash
        if (geohashIndex.containsKey(geohash)) {
            results.addAll(geohashIndex.get(geohash));
        }

        // 접두사가 일치하는 geohash (더 정밀한 위치)
        for (Map.Entry<String, List<Location>> entry : geohashIndex.entrySet()) {
            if (entry.getKey().startsWith(geohash)) {
                results.addAll(entry.getValue());
            }
        }
    }

    // Haversine 공식으로 거리 계산 (km)
    private double calculateDistance(double lat1, double lon1,
                                    double lat2, double lon2) {
        final int R = 6371; // 지구 반지름 (km)

        double latDistance = Math.toRadians(lat2 - lat1);
        double lonDistance = Math.toRadians(lon2 - lon1);

        double a = Math.sin(latDistance / 2) * Math.sin(latDistance / 2)
                + Math.cos(Math.toRadians(lat1))
                * Math.cos(Math.toRadians(lat2))
                * Math.sin(lonDistance / 2) * Math.sin(lonDistance / 2);

        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

        return R * c;
    }

    public static void main(String[] args) {
        LocationService service = new LocationService();

        // 샘플 위치 추가
        service.addLocation(new Location("1", "Starbucks A",
                                        37.7749, -122.4194));
        service.addLocation(new Location("2", "Starbucks B",
                                        37.7750, -122.4195));
        service.addLocation(new Location("3", "Starbucks C",
                                        37.7850, -122.4294));

        // 근처 위치 검색 (반경 5km)
        List<Location> nearby = service.findNearby(37.7749, -122.4194, 5);

        System.out.println("Found " + nearby.size() + " locations:");
        for (Location loc : nearby) {
            System.out.println("- " + loc.name +
                             " (Geohash: " + loc.geohash + ")");
        }
    }
}
```

**데이터베이스 Geohash 인덱싱 (PostgreSQL):**

```sql
-- 테이블 생성
CREATE TABLE locations (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    geohash VARCHAR(12)
);

-- Geohash 인덱스 생성
CREATE INDEX idx_geohash ON locations(geohash);

-- 데이터 삽입 (Java에서 geohash 계산 후 저장)
INSERT INTO locations (name, latitude, longitude, geohash)
VALUES ('Starbucks A', 37.7749, -122.4194, '9q8yy9');

-- 근처 위치 검색 (Geohash 접두사 매칭)
SELECT * FROM locations
WHERE geohash LIKE '9q8yy%'
ORDER BY geohash;

-- 더 넓은 범위 검색
SELECT * FROM locations
WHERE geohash LIKE '9q8%'
ORDER BY geohash;
```

**Geohashing 활용 사례:**

1. **공간 인덱싱**
   - 데이터베이스에서 효율적인 위치 기반 쿼리
   - 빠른 근접 검색

2. **위치 공유**
   - 정확한 좌표 노출 없이 대략적 위치 공유
   - 프라이버시 보호

3. **클러스터링 및 시각화**
   - 지리 데이터 집계
   - 히트맵 생성

4. **캐싱 키**
   - 위치 기반 캐시 키로 사용
   - 지역별 데이터 캐싱

**Geohashing의 장단점:**

**장점:**
- ✓ 단순하고 효율적인 인코딩
- ✓ 문자열 기반으로 데이터베이스 인덱싱 용이
- ✓ 계층적 구조로 다양한 정밀도 지원
- ✓ 접두사 매칭으로 빠른 근접 검색

**단점:**
- ✗ 경계선 문제 (가까운 점이 다른 셀에 있을 수 있음)
- ✗ 극지방에서 정확도 저하
- ✗ 정확한 거리 계산 필요 시 추가 연산 필요

### Quadtrees

**Quadtree**는 2차원 공간 데이터를 효율적으로 조직화, 검색, 조작하기 위한 트리 자료구조입니다. 각 내부 노드가 정확히 4개의 자식(북서, 북동, 남서, 남동)을 가지며, 공간을 재귀적으로 4등분하여 분할합니다.

**Quadtree 구조:**

```
              Root
         (전체 공간)
              |
      +-------+-------+
      |       |       |
     NW      NE      SW      SE
(북서) (북동) (남서) (남동)
      |
  +---+---+
  |   |   |
 NW  NE  SW  SE
```

**Quadtree의 특징:**

1. **공간 분할**: 영역을 4개의 동일한 사분면으로 재귀적 분할
2. **동적 깊이**: 데이터 밀도에 따라 트리 깊이 결정
3. **효율적 검색**: 공간 쿼리 시 불필요한 영역 건너뛰기
4. **균형 조절**: 데이터 분포에 따라 자동 균형

**Java Quadtree 구현:**

```java
import java.util.*;

public class Quadtree {
    private static final int MAX_CAPACITY = 4;  // 노드당 최대 포인트 수
    private static final int MAX_DEPTH = 8;     // 최대 깊이

    static class Point {
        double x, y;
        String data;

        public Point(double x, double y, String data) {
            this.x = x;
            this.y = y;
            this.data = data;
        }

        @Override
        public String toString() {
            return String.format("Point(%.2f, %.2f, %s)", x, y, data);
        }
    }

    static class Rectangle {
        double x, y;        // 중심점
        double width, height;

        public Rectangle(double x, double y, double width, double height) {
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
        }

        // 점이 사각형 안에 있는지 확인
        public boolean contains(Point point) {
            return (point.x >= x - width &&
                    point.x <= x + width &&
                    point.y >= y - height &&
                    point.y <= y + height);
        }

        // 두 사각형이 교차하는지 확인
        public boolean intersects(Rectangle range) {
            return !(range.x - range.width > x + width ||
                    range.x + range.width < x - width ||
                    range.y - range.height > y + height ||
                    range.y + range.height < y - height);
        }
    }

    static class QuadtreeNode {
        Rectangle boundary;
        List<Point> points;
        int depth;

        // 4개의 자식 노드
        QuadtreeNode northwest;
        QuadtreeNode northeast;
        QuadtreeNode southwest;
        QuadtreeNode southeast;

        boolean divided;

        public QuadtreeNode(Rectangle boundary, int depth) {
            this.boundary = boundary;
            this.points = new ArrayList<>();
            this.depth = depth;
            this.divided = false;
        }

        // 포인트 삽입
        public boolean insert(Point point) {
            // 경계 밖의 점은 무시
            if (!boundary.contains(point)) {
                return false;
            }

            // 용량 이내이고 분할되지 않았으면 추가
            if (points.size() < MAX_CAPACITY && !divided) {
                points.add(point);
                return true;
            }

            // 최대 깊이에 도달했으면 강제로 추가
            if (depth >= MAX_DEPTH) {
                points.add(point);
                return true;
            }

            // 분할이 필요한 경우
            if (!divided) {
                subdivide();
            }

            // 적절한 자식 노드에 삽입
            if (northwest.insert(point)) return true;
            if (northeast.insert(point)) return true;
            if (southwest.insert(point)) return true;
            if (southeast.insert(point)) return true;

            return false;
        }

        // 노드 분할
        private void subdivide() {
            double x = boundary.x;
            double y = boundary.y;
            double w = boundary.width / 2;
            double h = boundary.height / 2;

            northwest = new QuadtreeNode(
                new Rectangle(x - w, y - h, w, h), depth + 1);
            northeast = new QuadtreeNode(
                new Rectangle(x + w, y - h, w, h), depth + 1);
            southwest = new QuadtreeNode(
                new Rectangle(x - w, y + h, w, h), depth + 1);
            southeast = new QuadtreeNode(
                new Rectangle(x + w, y + h, w, h), depth + 1);

            divided = true;

            // 기존 포인트들을 자식 노드로 재분배
            List<Point> oldPoints = new ArrayList<>(points);
            points.clear();

            for (Point point : oldPoints) {
                insert(point);
            }
        }

        // 범위 내 포인트 검색
        public List<Point> query(Rectangle range, List<Point> found) {
            if (found == null) {
                found = new ArrayList<>();
            }

            // 범위와 교차하지 않으면 검색 중단
            if (!boundary.intersects(range)) {
                return found;
            }

            // 현재 노드의 포인트 확인
            for (Point point : points) {
                if (range.contains(point)) {
                    found.add(point);
                }
            }

            // 분할되어 있으면 자식 노드도 검색
            if (divided) {
                northwest.query(range, found);
                northeast.query(range, found);
                southwest.query(range, found);
                southeast.query(range, found);
            }

            return found;
        }

        // 트리 시각화 (디버깅용)
        public void print(String indent) {
            System.out.println(indent + "Node at depth " + depth +
                             ": " + points.size() + " points");
            if (divided) {
                northwest.print(indent + "  ");
                northeast.print(indent + "  ");
                southwest.print(indent + "  ");
                southeast.print(indent + "  ");
            }
        }
    }

    private QuadtreeNode root;

    public Quadtree(double width, double height) {
        this.root = new QuadtreeNode(
            new Rectangle(width / 2, height / 2, width / 2, height / 2), 0);
    }

    public void insert(Point point) {
        root.insert(point);
    }

    public List<Point> query(Rectangle range) {
        return root.query(range, null);
    }

    public void print() {
        root.print("");
    }

    public static void main(String[] args) {
        // 1000x1000 공간에 Quadtree 생성
        Quadtree quadtree = new Quadtree(1000, 1000);

        // 랜덤 포인트 삽입
        Random random = new Random(42);
        int numPoints = 100;

        System.out.println("Inserting " + numPoints + " points...");
        for (int i = 0; i < numPoints; i++) {
            double x = random.nextDouble() * 1000;
            double y = random.nextDouble() * 1000;
            quadtree.insert(new Point(x, y, "Point" + i));
        }

        // 트리 구조 출력
        System.out.println("\nQuadtree structure:");
        quadtree.print();

        // 범위 검색 (중심 500,500, 범위 100x100)
        Rectangle searchRange = new Rectangle(500, 500, 100, 100);
        List<Point> results = quadtree.query(searchRange);

        System.out.println("\nPoints in range (500±100, 500±100):");
        System.out.println("Found " + results.size() + " points:");
        for (Point p : results) {
            System.out.println("  " + p);
        }

        // 성능 비교: Quadtree vs Brute Force
        long startTime = System.nanoTime();
        List<Point> qtResults = quadtree.query(searchRange);
        long qtTime = System.nanoTime() - startTime;

        System.out.println("\nQuadtree search time: " +
                         qtTime / 1000.0 + " μs");
    }
}
```

**Quadtree 활용 사례:**

```java
// 충돌 감지 (게임 개발)
public class CollisionDetection {
    private Quadtree quadtree;

    static class GameObject {
        Quadtree.Point position;
        double radius;

        public GameObject(double x, double y, double radius, String id) {
            this.position = new Quadtree.Point(x, y, id);
            this.radius = radius;
        }
    }

    public CollisionDetection(double width, double height) {
        this.quadtree = new Quadtree(width, height);
    }

    public void addObject(GameObject obj) {
        quadtree.insert(obj.position);
    }

    // 특정 객체와 충돌 가능한 객체 찾기
    public List<Quadtree.Point> findPotentialCollisions(GameObject obj) {
        // 객체 주변 범위 검색
        Quadtree.Rectangle range = new Quadtree.Rectangle(
            obj.position.x, obj.position.y,
            obj.radius * 2, obj.radius * 2
        );

        return quadtree.query(range);
    }
}

// 이미지 압축
public class ImageCompression {
    static class QuadNode {
        int color;  // 평균 색상
        QuadNode[] children;  // 4개의 자식
        boolean isLeaf;

        public QuadNode(int color) {
            this.color = color;
            this.isLeaf = true;
        }
    }

    public static QuadNode compress(int[][] image, int x, int y,
                                   int size, int threshold) {
        // 영역의 평균 색상과 편차 계산
        int avgColor = calculateAverage(image, x, y, size);
        int variance = calculateVariance(image, x, y, size, avgColor);

        QuadNode node = new QuadNode(avgColor);

        // 편차가 임계값보다 크면 분할
        if (variance > threshold && size > 1) {
            node.isLeaf = false;
            node.children = new QuadNode[4];

            int halfSize = size / 2;
            node.children[0] = compress(image, x, y, halfSize, threshold);
            node.children[1] = compress(image, x + halfSize, y,
                                       halfSize, threshold);
            node.children[2] = compress(image, x, y + halfSize,
                                       halfSize, threshold);
            node.children[3] = compress(image, x + halfSize, y + halfSize,
                                       halfSize, threshold);
        }

        return node;
    }

    private static int calculateAverage(int[][] image, int x, int y,
                                       int size) {
        int sum = 0;
        int count = 0;
        for (int i = x; i < x + size; i++) {
            for (int j = y; j < y + size; j++) {
                sum += image[i][j];
                count++;
            }
        }
        return sum / count;
    }

    private static int calculateVariance(int[][] image, int x, int y,
                                        int size, int avg) {
        int variance = 0;
        for (int i = x; i < x + size; i++) {
            for (int j = y; j < y + size; j++) {
                int diff = image[i][j] - avg;
                variance += diff * diff;
            }
        }
        return variance / (size * size);
    }
}
```

**Quadtree의 장단점:**

**장점:**
- ✓ 효율적인 공간 분할 및 검색
- ✓ 동적 데이터에 적합
- ✓ 범위 쿼리 빠름 (O(log n) ~ O(n))
- ✓ 메모리 효율적 (데이터 밀도에 따라 조절)

**단점:**
- ✗ 삽입 순서에 따라 불균형 가능
- ✗ 3차원 이상에는 적합하지 않음
- ✗ 구현 복잡도
- ✗ 경계 영역 처리 복잡

**Geohashing vs Quadtrees 비교:**

| 특징 | Geohashing | Quadtrees |
|------|-----------|-----------|
| **데이터 구조** | 문자열 | 트리 |
| **공간 분할** | 고정 그리드 | 동적 분할 |
| **검색 복잡도** | O(1) ~ O(n) | O(log n) ~ O(n) |
| **삽입 복잡도** | O(1) | O(log n) |
| **메모리** | 작음 | 중간 |
| **데이터베이스 통합** | 쉬움 (문자열 인덱스) | 어려움 |
| **정밀도** | 고정 레벨 | 동적 조절 |
| **사용 사례** | 위치 검색, 캐싱 | 충돌 감지, 이미지 처리 |

**위치 기반 서비스 선택 가이드:**

```
Geohashing 선택:
- 데이터베이스 기반 위치 검색
- 대규모 위치 데이터
- 캐싱 키 생성
- 위치 공유

Quadtrees 선택:
- 게임 개발 (충돌 감지)
- 이미지 처리
- 동적 데이터
- 메모리 기반 검색
- 불균등 데이터 분포
```

## 로드 밸런싱 (Load Balancing)

### 알고리즘 비교

| 알고리즘 | 장점 | 단점 | 사용 사례 |
|---------|------|------|-----------|
| Round Robin | 구현 간단, 균등 분배 | 서버 성능 차이 무시 | 동일 스펙 서버 |
| Least Connections | 실시간 부하 반영 | 오버헤드 존재 | 긴 요청 처리 |
| IP Hash | 세션 유지 용이 | 특정 서버에 부하 집중 가능 | Stateful 앱 |
| Weighted Round Robin | 서버 성능 차이 반영 | 가중치 설정 필요 | 이기종 서버 |

### Nginx 설정 예제

```nginx
# 기본 Round Robin
upstream backend {
    server backend1.example.com;
    server backend2.example.com;
    server backend3.example.com;
}

# Weighted Load Balancing
upstream backend_weighted {
    server backend1.example.com weight=3;
    server backend2.example.com weight=2;
    server backend3.example.com weight=1;
}

# Least Connections
upstream backend_least_conn {
    least_conn;
    server backend1.example.com;
    server backend2.example.com;
}

# IP Hash (Session Affinity)
upstream backend_ip_hash {
    ip_hash;
    server backend1.example.com;
    server backend2.example.com;
}

# Health Check 설정
upstream backend_health {
    server backend1.example.com max_fails=3 fail_timeout=30s;
    server backend2.example.com max_fails=3 fail_timeout=30s;
    server backend3.example.com backup;  # 백업 서버
}

server {
    listen 80;
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Layer 4 vs Layer 7 로드 밸런싱

**Layer 4 (Transport Layer):**
- TCP/UDP 헤더 정보만 사용
- 빠른 처리 속도
- 패킷 내용 검사 불가
- 간단한 분산에 적합

**Layer 7 (Application Layer):**
- HTTP 헤더, URL, 쿠키 등 활용
- 복잡한 라우팅 가능
- SSL 종료 처리 가능
- 더 많은 리소스 필요

## 캐싱 (Caching)

### 캐시 무효화 전략

**1. Time-based Expiration (TTL)**
```python
# Redis 예제
redis_client.setex('user:123', 3600, user_data)  # 1시간 TTL
```

**2. Event-based Invalidation**
```python
def update_user(user_id, data):
    # DB 업데이트
    db.update_user(user_id, data)
    # 캐시 무효화
    cache.delete(f'user:{user_id}')
```

**3. Write-Through Pattern**
```python
def update_user_write_through(user_id, data):
    # 캐시와 DB 동시 업데이트
    cache.set(f'user:{user_id}', data)
    db.update_user(user_id, data)
```

### Cache Stampede 문제

**문제:** 인기 있는 캐시 항목이 만료될 때 여러 요청이 동시에 DB에 접근

**해결책 1: Lock-based Approach**
```python
import redis
from redis.lock import Lock

def get_popular_data(key):
    data = cache.get(key)
    if data is None:
        lock = Lock(redis_client, f"lock:{key}", timeout=10)
        if lock.acquire(blocking=False):
            try:
                # DB에서 데이터 조회
                data = db.get(key)
                cache.set(key, data, ttl=3600)
            finally:
                lock.release()
        else:
            # 다른 프로세스가 데이터를 로드하는 동안 대기
            time.sleep(0.1)
            data = cache.get(key)
    return data
```

**해결책 2: Probabilistic Early Expiration**
```python
import random
import time

def get_with_early_expiration(key, ttl=3600):
    data, timestamp = cache.get_with_timestamp(key)

    if data is None:
        return refresh_cache(key, ttl)

    # 만료 시간의 일정 비율 전에 확률적으로 갱신
    time_left = ttl - (time.time() - timestamp)
    refresh_threshold = ttl * 0.1  # 10% 남았을 때

    if time_left < refresh_threshold:
        if random.random() < (refresh_threshold / time_left):
            return refresh_cache(key, ttl)

    return data

def refresh_cache(key, ttl):
    data = db.get(key)
    cache.set(key, data, ttl)
    return data
```

### Multi-level Caching

```python
class MultiLevelCache:
    def __init__(self):
        self.l1_cache = {}  # 로컬 메모리
        self.l2_cache = redis.Redis()  # Redis

    def get(self, key):
        # L1 캐시 확인
        if key in self.l1_cache:
            return self.l1_cache[key]

        # L2 캐시 확인
        value = self.l2_cache.get(key)
        if value:
            self.l1_cache[key] = value  # L1에 복사
            return value

        # DB에서 조회
        value = db.get(key)
        if value:
            self.l1_cache[key] = value
            self.l2_cache.setex(key, 3600, value)

        return value

    def set(self, key, value):
        self.l1_cache[key] = value
        self.l2_cache.setex(key, 3600, value)
        db.set(key, value)
```

## 데이터베이스 (Database)

### SQL vs NoSQL

> [SQL vs NoSQL: 5 Critical Differences](https://www.integrate.io/blog/the-sql-vs-nosql-difference/)

SQL과 NoSQL은 데이터 저장 및 검색 요구사항에 맞춰 서로 다른 접근 방식을 제공하는 데이터베이스 관리 시스템입니다. 두 시스템의 주요 차이점은 데이터가 구조화되고, 조직화되며, 조회되는 방식에 있습니다.

#### SQL (Structured Query Language)

관계형 데이터베이스 관리 시스템(RDBMS)을 위한 표준화된 프로그래밍 언어입니다.

**특징**:
- 사전에 정의된 고정 스키마와 구조화된 테이블 형식 사용
- 데이터는 행(Row)과 열(Column)로 구성된 테이블에 저장
- ACID(원자성, 일관성, 격리성, 지속성) 속성 완벽 지원
- 복잡한 쿼리, 다중 행 트랜잭션, JOIN 연산에 최적화
- 수직 확장(Scale-up): 단일 서버에 CPU, 메모리 추가

**대표 제품**:
- MySQL, PostgreSQL, Oracle Database, SQL Server, MariaDB

**사용 사례**:
```sql
-- 복잡한 JOIN 쿼리
SELECT
    o.order_id,
    u.username,
    p.product_name,
    o.quantity,
    o.total_price
FROM orders o
INNER JOIN users u ON o.user_id = u.id
INNER JOIN products p ON o.product_id = p.id
WHERE o.created_at >= '2024-01-01'
  AND o.status = 'COMPLETED'
ORDER BY o.total_price DESC;

-- 트랜잭션 예제
BEGIN TRANSACTION;

UPDATE accounts SET balance = balance - 100 WHERE account_id = 1;
UPDATE accounts SET balance = balance + 100 WHERE account_id = 2;

COMMIT; -- 모두 성공하면 커밋
-- 하나라도 실패하면 ROLLBACK
```

#### NoSQL (Not Only SQL)

SQL 쿼리 언어를 필수로 사용하지 않는 비관계형 데이터베이스입니다.

**특징**:
- 유연한 스키마: 키-값, 문서, 컬럼 패밀리, 그래프 등 다양한 데이터 모델 지원
- 확장성, 고가용성, 장애 허용성에 초점
- 최종 일관성(Eventual Consistency) 허용으로 성능과 분산 개선
- 비구조화/반구조화/계층형 데이터 처리에 적합
- 수평 확장(Scale-out): 여러 서버/클러스터로 데이터 분산

**NoSQL 유형별 특징**:

| 유형 | 설명 | 대표 제품 | 사용 사례 |
|------|------|-----------|----------|
| **Key-Value** | 단순한 키-값 쌍 저장 | Redis, DynamoDB, Memcached | 캐싱, 세션 관리, 장바구니 |
| **Document** | JSON/BSON 형식 문서 저장 | MongoDB, CouchDB, Firestore | 콘텐츠 관리, 사용자 프로필 |
| **Column-Family** | 컬럼 기반 저장 | Cassandra, HBase, ScyllaDB | 시계열 데이터, 로그 분석 |
| **Graph** | 노드와 엣지로 관계 표현 | Neo4j, ArangoDB, Amazon Neptune | 소셜 네트워크, 추천 시스템 |

**MongoDB 예제 (Document DB)**:
```javascript
// 유연한 스키마 - 각 문서가 다른 필드를 가질 수 있음
db.users.insertOne({
  username: "john_doe",
  email: "john@example.com",
  profile: {
    age: 30,
    city: "Seoul",
    interests: ["coding", "gaming"]
  },
  created_at: new Date()
});

// 복잡한 중첩 구조도 자연스럽게 표현
db.orders.insertOne({
  order_id: "ORD-12345",
  user_id: "user123",
  items: [
    {
      product_id: "PRD-001",
      name: "Laptop",
      quantity: 1,
      price: 1200000
    },
    {
      product_id: "PRD-002",
      name: "Mouse",
      quantity: 2,
      price: 30000
    }
  ],
  total: 1260000,
  status: "PENDING"
});

// 강력한 쿼리
db.users.find({
  "profile.age": { $gte: 25, $lte: 35 },
  "profile.city": "Seoul"
}).limit(10);
```

**Cassandra 예제 (Column-Family DB)**:
```sql
-- 파티션 키로 데이터 분산
CREATE TABLE sensor_data (
    sensor_id UUID,
    timestamp TIMESTAMP,
    temperature DOUBLE,
    humidity DOUBLE,
    PRIMARY KEY (sensor_id, timestamp)
) WITH CLUSTERING ORDER BY (timestamp DESC);

-- 시계열 데이터 조회
SELECT * FROM sensor_data
WHERE sensor_id = 123e4567-e89b-12d3-a456-426614174000
  AND timestamp >= '2024-01-01'
  AND timestamp < '2024-02-01';
```

#### 상세 비교

| 특성 | SQL (관계형) | NoSQL (비관계형) |
|------|-------------|-----------------|
| **스키마** | 엄격한 고정 스키마 | 유연한 스키마 또는 스키마리스 |
| **데이터 모델** | 테이블(행, 열) | 키-값, 문서, 컬럼, 그래프 |
| **쿼리 언어** | 표준 SQL | 제품별 쿼리 언어 (또는 SQL 유사) |
| **확장성** | 수직 확장 (Scale-up) | 수평 확장 (Scale-out) |
| **트랜잭션** | 강력한 ACID 보장 | BASE (기본 가용성, 유연한 상태, 최종 일관성) |
| **일관성** | 즉시 일관성 (Immediate) | 최종 일관성 (Eventual) |
| **JOIN** | 복잡한 JOIN 지원 | 일반적으로 JOIN 없음 (비정규화) |
| **성능** | 복잡한 쿼리에 강함 | 단순 쿼리에 매우 빠름 |
| **학습 곡선** | 표준화되어 배우기 쉬움 | 제품별로 다름 |
| **사용 사례** | 금융, ERP, 전자상거래 | 빅데이터, 실시간 분석, IoT |

#### ACID vs BASE

**ACID (SQL 데이터베이스)**:
- **Atomicity (원자성)**: 트랜잭션은 모두 성공하거나 모두 실패
- **Consistency (일관성)**: 트랜잭션 전후로 데이터 무결성 유지
- **Isolation (격리성)**: 동시 실행 트랜잭션이 서로 영향 없음
- **Durability (지속성)**: 커밋된 데이터는 영구 저장

**BASE (NoSQL 데이터베이스)**:
- **Basically Available (기본 가용성)**: 부분 장애에도 시스템 작동
- **Soft state (유연한 상태)**: 상태가 시간에 따라 변할 수 있음
- **Eventual consistency (최종 일관성)**: 시간이 지나면 일관성 도달

#### 선택 가이드

**SQL을 선택해야 하는 경우**:

✓ 데이터 구조가 명확하고 변경이 적음
✓ 복잡한 관계와 JOIN이 필요
✓ 강력한 트랜잭션 보장 필수 (금융, 주문)
✓ ACID 준수가 중요
✓ 비즈니스 규칙이 명확하고 일관성 필요
✓ 데이터 무결성이 최우선

**예**: 은행 시스템, 전자상거래 주문, 재고 관리, ERP

**NoSQL을 선택해야 하는 경우**:

✓ 데이터 구조가 유동적이거나 자주 변경
✓ 대용량 데이터 저장 및 빠른 읽기/쓰기
✓ 수평 확장 필요 (트래픽 증가에 유연)
✓ 비정규화된 데이터 모델 허용
✓ 최종 일관성으로 충분
✓ 지리적으로 분산된 데이터

**예**: 소셜 미디어 피드, 실시간 분석, IoT 센서 데이터, 로그 수집, 콘텐츠 관리

#### 하이브리드 접근 (Polyglot Persistence)

현대 애플리케이션은 종종 SQL과 NoSQL을 함께 사용합니다.

```java
@Service
public class EcommerceService {

    @Autowired
    private OrderRepository orderRepository; // PostgreSQL (SQL)

    @Autowired
    private ProductCatalogRepository catalogRepository; // MongoDB (NoSQL)

    @Autowired
    private SessionRepository sessionRepository; // Redis (NoSQL)

    @Autowired
    private RecommendationRepository recommendationRepository; // Neo4j (Graph)

    public Order createOrder(CreateOrderRequest request) {
        // 1. SQL: 주문 데이터는 트랜잭션 보장 필요
        Order order = new Order();
        order.setUserId(request.getUserId());
        order.setStatus(OrderStatus.PENDING);
        order = orderRepository.save(order); // PostgreSQL

        // 2. NoSQL: 상품 정보 조회 (빠른 읽기)
        Product product = catalogRepository.findById(request.getProductId()); // MongoDB

        // 3. NoSQL: 세션 정보 (캐싱)
        sessionRepository.updateCartCount(request.getUserId()); // Redis

        // 4. Graph: 추천 업데이트
        recommendationRepository.recordPurchase(
            request.getUserId(),
            request.getProductId()
        ); // Neo4j

        return order;
    }
}
```

**실전 아키텍처 예시**:

```
┌─────────────────────────────────────────────────────┐
│               애플리케이션 레이어                     │
└─────────────────────┬───────────────────────────────┘
                      │
      ┌───────────────┼───────────────┐
      │               │               │
      ↓               ↓               ↓
┌──────────┐    ┌──────────┐   ┌──────────┐
│PostgreSQL│    │ MongoDB  │   │  Redis   │
│  (SQL)   │    │ (NoSQL)  │   │ (NoSQL)  │
├──────────┤    ├──────────┤   ├──────────┤
│• 주문    │    │• 상품    │   │• 세션    │
│• 결제    │    │• 리뷰    │   │• 캐시    │
│• 사용자  │    │• 로그    │   │• 랭킹    │
└──────────┘    └──────────┘   └──────────┘
```

#### 마이그레이션 고려사항

**SQL → NoSQL 마이그레이션**:
```java
// 정규화된 SQL 스키마
// users 테이블: id, name, email
// orders 테이블: id, user_id, amount
// order_items 테이블: id, order_id, product_id, quantity

// MongoDB로 마이그레이션 (비정규화)
{
  "_id": "user123",
  "name": "John Doe",
  "email": "john@example.com",
  "orders": [
    {
      "order_id": "ORD-001",
      "amount": 150000,
      "items": [
        { "product_id": "PRD-1", "quantity": 2 },
        { "product_id": "PRD-2", "quantity": 1 }
      ]
    }
  ]
}
```

**주의사항**:
- JOIN 로직을 애플리케이션 레벨로 이동
- 데이터 중복 허용 및 관리
- 트랜잭션 범위 재정의
- 최종 일관성 처리 로직 추가

### 인덱싱 전략

- [Database Index](/index/README.md)

**1. B-Tree Index (기본)**
- 범위 검색에 효율적
- 정렬된 데이터 접근
- 대부분의 RDBMS 기본 인덱스

```sql
-- 단일 컬럼 인덱스
CREATE INDEX idx_user_email ON users(email);

-- 복합 인덱스 (순서 중요!)
CREATE INDEX idx_user_age_city ON users(age, city);

-- 이 쿼리는 인덱스 사용
SELECT * FROM users WHERE age = 25 AND city = 'Seoul';

-- 이 쿼리는 인덱스 일부만 사용
SELECT * FROM users WHERE city = 'Seoul';  -- age가 선행 컬럼
```

**2. Hash Index**
- 동등 비교에 최적화
- 범위 검색 불가
- O(1) 조회 성능

```sql
-- PostgreSQL Hash Index
CREATE INDEX idx_user_id_hash ON users USING HASH (user_id);
```

**3. Full-Text Index**
- 텍스트 검색에 특화
- 키워드 검색, 관련도 순위 지원

```sql
-- MySQL Full-Text Index
CREATE FULLTEXT INDEX idx_article_content ON articles(title, content);

SELECT * FROM articles
WHERE MATCH(title, content) AGAINST('system design' IN NATURAL LANGUAGE MODE);
```

**인덱스 사용 시 주의사항:**
- 쓰기 성능 저하 (INSERT, UPDATE, DELETE)
- 저장 공간 추가 필요
- 너무 많은 인덱스는 오히려 성능 저하

### Replication Lag 처리

**문제 상황:**
```python
# 마스터에 쓰기
master_db.execute("INSERT INTO users VALUES (...)")

# 즉시 읽기 시도 - Replication Lag로 인해 데이터 없을 수 있음
user = slave_db.query("SELECT * FROM users WHERE id = 123")
```

**해결책 1: Read Your Writes**
```python
class ReplicationAwareDB:
    def __init__(self):
        self.master = master_db
        self.slave = slave_db
        self.recent_writes = {}  # user_id -> timestamp

    def write(self, user_id, data):
        self.master.execute(data)
        self.recent_writes[user_id] = time.time()

    def read(self, user_id):
        # 최근 쓰기가 있었다면 마스터에서 읽기
        if user_id in self.recent_writes:
            write_time = self.recent_writes[user_id]
            if time.time() - write_time < 5:  # 5초 이내
                return self.master.query(user_id)

        # 그렇지 않으면 슬레이브에서 읽기
        return self.slave.query(user_id)
```

**해결책 2: Monotonic Reads**
```python
# 세션별로 항상 같은 복제본 사용
class SessionAwareDB:
    def __init__(self):
        self.slaves = [slave1, slave2, slave3]

    def get_slave_for_session(self, session_id):
        # 세션 ID에 따라 일관된 슬레이브 선택
        slave_index = hash(session_id) % len(self.slaves)
        return self.slaves[slave_index]
```

**해결책 3: Eventual Consistency 명시**
```python
def get_user_profile(user_id, consistency='eventual'):
    if consistency == 'strong':
        # 마스터에서 읽기 (최신 데이터 보장)
        return master_db.query(user_id)
    else:
        # 슬레이브에서 읽기 (빠르지만 약간 오래된 데이터 가능)
        return slave_db.query(user_id)
```

### Database Connection Pooling

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Connection Pool 설정
engine = create_engine(
    'postgresql://user:pass@localhost/db',
    poolclass=QueuePool,
    pool_size=10,        # 기본 연결 수
    max_overflow=20,     # 추가 생성 가능한 연결 수
    pool_timeout=30,     # 연결 대기 시간
    pool_recycle=3600,   # 연결 재사용 시간 (1시간)
    pool_pre_ping=True   # 연결 유효성 사전 체크
)

# 사용 예제
with engine.connect() as conn:
    result = conn.execute("SELECT * FROM users")
```

### Bloom Filter

- [Bloom Filter](/bloomfilter/README-kr.md)

## 분산 시스템 (Distributed Systems)

### CAP Theorem (일관성, 가용성, 분할 허용)

분산 시스템은 **Consistency(일관성), Availability(가용성), Partition Tolerance(분할 허용)** 중 2가지만 만족할 수 있습니다.

**CAP 이론 설명:**

<p align="center">
  <img src="https://camo.githubusercontent.com/13719b22b169e7d8c1c424b67f29811208877ab77ff8e82d59fa058c7f5e6e3d/68747470733a2f2f6769746875622e636f6d2f646f6e6e656d617274696e2f73797374656d2d64657369676e2d7072696d65722f7261772f6d61737465722f696d616765732f6267506c52746c2e706e67"/>
</p>

- **CP (Consistency + Partition Tolerance)**: 일관성과 분할 허용을 보장하지만 가용성을 희생
  - 예: MongoDB, HBase, Redis
  - 네트워크 분할 시 일부 노드는 응답하지 않음

- **AP (Availability + Partition Tolerance)**: 가용성과 분할 허용을 보장하지만 일관성을 희생
  - 예: Cassandra, CouchDB, DynamoDB
  - 항상 응답하지만 최신 데이터가 아닐 수 있음

- **CA (Consistency + Availability)**: 일관성과 가용성을 보장하지만 분할 허용을 희생
  - 예: RDBMS (PostgreSQL, MySQL)
  - 실제로는 네트워크 분할이 발생하므로 완전한 CA는 불가능

**실전 예제:**

```python
# CP 시스템 - 강한 일관성 보장
class CPDatabase:
    def write(self, key, value):
        # Quorum Write: 과반수 노드가 확인할 때까지 대기
        acks = 0
        required_acks = (len(nodes) // 2) + 1

        for node in nodes:
            try:
                node.write(key, value)
                acks += 1
                if acks >= required_acks:
                    return True
            except NetworkError:
                continue

        raise Exception("Failed to achieve quorum")

    def read(self, key):
        # Quorum Read: 과반수 노드에서 최신 값 읽기
        values = []
        required_reads = (len(nodes) // 2) + 1

        for node in nodes:
            try:
                value = node.read(key)
                values.append(value)
                if len(values) >= required_reads:
                    return self.resolve_conflicts(values)
            except NetworkError:
                continue

        raise Exception("Failed to achieve quorum")

# AP 시스템 - 높은 가용성 보장
class APDatabase:
    def write(self, key, value):
        # Hinted Handoff: 사용 가능한 노드에 쓰기
        written = False
        for node in nodes:
            try:
                node.write(key, value)
                written = True
            except NetworkError:
                # 힌트를 다른 노드에 저장
                hint_node.store_hint(node, key, value)

        return written  # 최소 1개 노드에 쓰기 성공하면 OK

    def read(self, key):
        # Read Repair: 여러 노드에서 읽고 나중에 동기화
        for node in nodes:
            try:
                return node.read(key)  # 첫 번째 응답 반환
            except NetworkError:
                continue

        return None  # 모든 노드 실패 시에만 실패
```

### PACELC Theorem

CAP을 확장한 이론으로, 네트워크 분할(P) 상황과 정상(E) 상황을 모두 고려합니다.

**PACELC = If Partition then (Availability vs Consistency) Else (Latency vs Consistency)**

```
분할 시: A vs C 중 선택
정상 시: L vs C 중 선택
```

**데이터베이스별 PACELC 분류:**

| Database | Partition | Normal | 설명 |
|----------|-----------|--------|------|
| HBase | PC | EC | 항상 일관성 우선 |
| Cassandra | PA | EL | 항상 가용성/성능 우선 |
| MongoDB | PA | EC | 분할 시 가용성, 평소 일관성 |
| DynamoDB | PA | EL | 가용성과 낮은 지연시간 우선 |

```python
class PAELCDatabase:
    """Cassandra 스타일 - PA/EL"""

    def __init__(self, consistency_level='ONE'):
        self.consistency_level = consistency_level

    def write(self, key, value):
        if self.detect_partition():
            # Partition: Availability 우선
            return self.write_to_any_node(key, value)
        else:
            # Normal: Latency 우선 (빠른 응답)
            return self.write_with_async_replication(key, value)

    def write_to_any_node(self, key, value):
        # 사용 가능한 아무 노드나 쓰기
        for node in self.available_nodes():
            try:
                node.write(key, value)
                return True
            except:
                continue
        return False

    def write_with_async_replication(self, key, value):
        # 1개 노드에 쓰고 비동기로 복제
        primary = self.get_primary_node(key)
        primary.write(key, value)

        # 백그라운드로 복제
        self.async_replicate(key, value)
        return True
```

### DNS (Domain Name System)

DNS는 도메인 이름을 IP 주소로 변환하는 분산 계층적 시스템입니다.

**DNS 조회 과정:**

```
User → Browser Cache → OS Cache → Router Cache → ISP DNS → Root DNS → TLD DNS → Authoritative DNS
```

**DNS 서버 계층:**

```python
class DNSResolver:
    def resolve(self, domain):
        # 1. 캐시 확인
        if cached := self.cache.get(domain):
            return cached

        # 2. Root DNS 조회 (.com, .org 등의 TLD 서버 주소)
        tld_server = self.query_root_dns(domain)

        # 3. TLD DNS 조회 (example.com의 Authoritative 서버 주소)
        auth_server = self.query_tld_dns(tld_server, domain)

        # 4. Authoritative DNS 조회 (실제 IP 주소)
        ip = self.query_authoritative_dns(auth_server, domain)

        # 5. 캐시에 저장 (TTL 설정)
        self.cache.set(domain, ip, ttl=3600)

        return ip
```

**DNS 레코드 타입:**

```python
# A Record - 도메인 → IPv4
{
    'type': 'A',
    'name': 'example.com',
    'value': '192.0.2.1',
    'ttl': 3600
}

# AAAA Record - 도메인 → IPv6
{
    'type': 'AAAA',
    'name': 'example.com',
    'value': '2001:0db8:85a3::8a2e:0370:7334',
    'ttl': 3600
}

# CNAME Record - 도메인 → 다른 도메인
{
    'type': 'CNAME',
    'name': 'www.example.com',
    'value': 'example.com',
    'ttl': 3600
}

# MX Record - 메일 서버
{
    'type': 'MX',
    'name': 'example.com',
    'value': 'mail.example.com',
    'priority': 10,
    'ttl': 3600
}
```

### CDN (Content Delivery Network)

CDN은 전 세계에 분산된 서버를 통해 콘텐츠를 사용자와 가까운 위치에서 제공합니다.

**Push CDN vs Pull CDN:**

```python
# Push CDN - 콘텐츠를 CDN에 미리 업로드
class PushCDN:
    def upload_content(self, content, path):
        # 모든 CDN 엣지 서버에 콘텐츠 푸시
        for edge_server in self.edge_servers:
            edge_server.store(path, content)

    def update_content(self, content, path):
        # 업데이트 시 모든 서버에 다시 푸시 필요
        self.invalidate(path)
        self.upload_content(content, path)

    # 장점: 첫 요청부터 빠름, 트래픽 예측 가능
    # 단점: 저장 공간 많이 사용, 업데이트 비용 높음

# Pull CDN - 요청 시 Origin에서 가져와 캐싱
class PullCDN:
    def get_content(self, path):
        # 엣지 서버 캐시 확인
        if cached := self.cache.get(path):
            return cached

        # 캐시 미스 - Origin에서 가져오기
        content = self.origin_server.get(path)

        # 캐시에 저장
        self.cache.set(path, content, ttl=3600)

        return content

    # 장점: 저장 공간 효율적, 자동 업데이트
    # 단점: 첫 요청 느림, Origin 부하 발생 가능
```

**CDN 구현 예제:**

```python
class CDNSystem:
    def __init__(self):
        self.origin = OriginServer()
        self.edge_servers = [
            EdgeServer('us-west'),
            EdgeServer('us-east'),
            EdgeServer('eu-central'),
            EdgeServer('ap-southeast')
        ]

    def route_request(self, user_location, path):
        # 사용자와 가장 가까운 엣지 서버 선택
        edge_server = self.find_nearest_edge(user_location)

        try:
            # 엣지 서버에서 콘텐츠 조회
            return edge_server.get(path)
        except CacheMiss:
            # Origin에서 가져와서 캐시
            content = self.origin.get(path)
            edge_server.cache(path, content)
            return content

    def find_nearest_edge(self, user_location):
        # Geo-IP 기반으로 가장 가까운 서버 선택
        min_latency = float('inf')
        nearest_server = None

        for server in self.edge_servers:
            latency = self.calculate_latency(user_location, server.location)
            if latency < min_latency:
                min_latency = latency
                nearest_server = server

        return nearest_server
```

### Saga Pattern 상세

**문제:** 분산 트랜잭션에서 ACID 보장의 어려움

**Choreography-based Saga (이벤트 기반)**

```python
# 주문 서비스
class OrderService:
    def create_order(self, order_data):
        # 1. 주문 생성
        order = db.create_order(order_data)

        # 2. 이벤트 발행
        event_bus.publish('OrderCreated', {
            'order_id': order.id,
            'user_id': order.user_id,
            'amount': order.amount
        })

        return order

    def handle_payment_failed(self, event):
        # 보상 트랜잭션: 주문 취소
        order_id = event['order_id']
        db.cancel_order(order_id)

# 결제 서비스
class PaymentService:
    def __init__(self):
        event_bus.subscribe('OrderCreated', self.process_payment)

    def process_payment(self, event):
        try:
            # 결제 처리
            payment = db.create_payment(event['order_id'], event['amount'])

            # 성공 이벤트 발행
            event_bus.publish('PaymentCompleted', {
                'order_id': event['order_id'],
                'payment_id': payment.id
            })
        except PaymentError as e:
            # 실패 이벤트 발행
            event_bus.publish('PaymentFailed', {
                'order_id': event['order_id'],
                'reason': str(e)
            })

# 배송 서비스
class ShippingService:
    def __init__(self):
        event_bus.subscribe('PaymentCompleted', self.create_shipment)
        event_bus.subscribe('PaymentFailed', self.handle_payment_failed)

    def create_shipment(self, event):
        # 배송 생성
        shipment = db.create_shipment(event['order_id'])
        event_bus.publish('ShipmentCreated', {'order_id': event['order_id']})
```

**Orchestration-based Saga (중앙 제어)**

```python
class OrderSagaOrchestrator:
    def execute_order_saga(self, order_data):
        saga_state = {
            'order_id': None,
            'payment_id': None,
            'shipment_id': None,
            'step': 0
        }

        try:
            # Step 1: 주문 생성
            saga_state['order_id'] = order_service.create(order_data)
            saga_state['step'] = 1

            # Step 2: 결제 처리
            saga_state['payment_id'] = payment_service.process(
                saga_state['order_id'], order_data['amount']
            )
            saga_state['step'] = 2

            # Step 3: 배송 생성
            saga_state['shipment_id'] = shipping_service.create(
                saga_state['order_id']
            )
            saga_state['step'] = 3

            return {'status': 'success', 'order_id': saga_state['order_id']}

        except Exception as e:
            # 보상 트랜잭션 실행
            self.compensate(saga_state)
            return {'status': 'failed', 'error': str(e)}

    def compensate(self, saga_state):
        # 역순으로 보상 트랜잭션 실행
        if saga_state['step'] >= 2:
            payment_service.refund(saga_state['payment_id'])

        if saga_state['step'] >= 1:
            order_service.cancel(saga_state['order_id'])
```

### Outbox Pattern

**문제:** DB 트랜잭션과 메시지 발행의 원자성 보장

**해결책:**

```python
def create_order_with_outbox(order_data):
    with db.transaction():
        # 1. 주문 생성
        order = db.insert("INSERT INTO orders (...) VALUES (...)")

        # 2. Outbox 테이블에 이벤트 저장
        db.insert("""
            INSERT INTO outbox (
                aggregate_id, event_type, payload, created_at
            ) VALUES (?, ?, ?, ?)
        """, (order.id, 'OrderCreated', json.dumps(order_data), now()))

# 별도 프로세스: Outbox Publisher
class OutboxPublisher:
    def poll_and_publish(self):
        while True:
            # 미발행 이벤트 조회
            events = db.query("""
                SELECT * FROM outbox
                WHERE published = false
                ORDER BY created_at
                LIMIT 100
            """)

            for event in events:
                try:
                    # 메시지 큐에 발행
                    kafka.publish(event.event_type, event.payload)

                    # 발행 완료 표시
                    db.update("""
                        UPDATE outbox
                        SET published = true, published_at = ?
                        WHERE id = ?
                    """, (now(), event.id))
                except Exception as e:
                    logger.error(f"Failed to publish event {event.id}: {e}")

            time.sleep(1)
```

### Distributed Primary Key

* [강대명 <대용량 서버 구축을 위한 Memcached와 Redis>](https://americanopeople.tistory.com/177)

분산 시스템에서 Sharding을 고려한 Primary Key를 효율적으로 설계하는 방법을 알아봅시다. 이메일 시스템을 예시로 User와 Email 테이블의 스키마를 설계해보겠습니다.

**테이블 스키마 설계**

**User 테이블**

| 필드명           | 타입       | 설명                                   |
|-----------------|-----------|----------------------------------------|
| user_id         | Long (8B) | 고유 ID (각 DB 샤드별)                  |
| email           | String    | 이메일 주소                             |
| shard           | Long      | 메일 리스트가 저장된 DB 서버 번호        |
| type            | int       | 사용자 활성화 상태                      |
| created_at      | timestamp | 계정 생성 시간                          |
| last_login_time | timestamp | 마지막 로그인 시간                      |

**Email 테이블**

| 필드명       | 타입            | 설명                        |
|-------------|----------------|-----------------------------|
| mail_id     | Long (8B)      | 고유 ID (각 DB 샤드별)       |
| receiver    | String or Long | 수신자                      |
| sender      | String or Long | 송신자                      |
| subject     | String         | 메일 제목                   |
| received_at | timestamp      | 수신 시간                   |
| eml_id      | String or Long | 메일 본문 저장 ID 또는 URL   |
| is_read     | boolean        | 읽음 여부                   |
| contents    | String         | 미리보기 (내용 일부)         |

**Distributed Primary Key 설계 전략**

이메일 파일은 AWS S3에 저장하며, 적절한 Key 설계가 필요합니다.

**1. 기본 조합 키**
```
{receiver_id}_{mail_id}
```
- `mail_id`는 샤드마다 중복 가능하므로 `receiver_id`와 결합하여 전역 고유성 보장
- `eml_id` 필드가 필요한가? `{receiver_id}_{mail_id}`만으로도 충분할 수 있음

**2. UUID (Universally Unique Identifier)**
```
크기: 16 bytes (128 bits), 36 characters
예시: 550e8400-e29b-41d4-a716-446655440000
```
- **장점**: ID에 시간 정보 포함, 오름차순 정렬 시 시간순 정렬 가능
- **단점**: 크기가 크고 (16 bytes) 인덱싱 비효율적
- 더 작은 바이트로 시간 정보를 저장하는 방법이 필요

**3. 타임스탬프 + 시퀀스**
```
{timestamp: 52 bits}_{sequence: 12 bits} = 8 bytes
```
- 시간 정보를 포함하여 시간순 정렬 가능
- **제약**: 4 bytes 타임스탬프 사용 시 `1970/01/01` ~ `2106/02/07 06:28`까지만 표현 가능

**4. 타임스탬프 + 샤드 ID + 시퀀스**
```
{timestamp: 52 bits}_{shard_id: 12 bits}_{sequence: 12 bits} = 8 bytes
```
- 샤드 정보를 ID에 포함하여 데이터 위치 파악 용이
- 키에 시간을 포함하면 시간에 따라 데이터가 적절히 샤드로 분산됨

**5. Twitter Snowflake ID**
```
{timestamp: 42 bits}_{datacenter_id: 5 bits}_{worker_id: 5 bits}_{sequence: 12 bits} = 8 bytes
```
- Timestamp (42 bits): 밀리초 단위, 약 69년 사용 가능
- Datacenter ID (5 bits): 최대 32개 데이터센터
- Worker ID (5 bits): 데이터센터당 최대 32개 워커
- Sequence (12 bits): 밀리초당 최대 4,096개 ID 생성
- IDC(데이터센터) 정보 포함

**6. Instagram ID**
```
{timestamp: 41 bits}_{logical_shard_id: 13 bits}_{auto_increment/1024: 10 bits} = 8 bytes
```
- 논리적 샤드 ID를 포함하여 데이터 분산 최적화
- Auto Increment를 1024로 나눈 값 사용

**7. MongoDB ObjectId**
```
{timestamp: 4 bytes}_{machine_id: 3 bytes}_{process_id: 2 bytes}_{counter: 3 bytes} = 12 bytes
```
- 타임스탬프 + 머신 식별자 + 프로세스 ID + 카운터
- 분산 환경에서 충돌 없이 고유 ID 생성

**8. 범용 조합**
```
{timestamp}_{shard_id}_{type}_{sequence} = 8 bytes
```
- 타임스탬프, 샤드 ID, 타입, 시퀀스를 모두 포함
- 유연한 확장 가능

**9. Timebound 최적화**
```
{timebound}_{shard_id}_{type}_{sequence} = 8 bytes
```

특정 사용자의 최근 10분간 수신 이메일만 조회하는 경우, Primary Key에 timebound를 도입하면 효율적입니다.

- **Timebound 없는 경우**:
  - 이메일 데이터가 모든 샤드에 골고루 분산
  - 조회 시 모든 샤드를 쿼리해야 함 (비효율적)

- **Timebound 적용**:
  - 특정 시간대(예: 1시간 단위)의 이메일이 하나의 샤드에 저장
  - 특정 유저의 최근 1시간 이메일은 하나의 샤드에만 존재
  - 해당 샤드만 쿼리하면 되어 효율적

**설계 시 고려사항**

| 고려사항 | 설명 |
|---------|------|
| **크기 vs 정보량** | 더 많은 정보를 담을수록 ID 크기 증가 |
| **정렬 가능성** | 시간 기반 정렬이 필요한지 고려 |
| **샤드 정보** | ID에서 데이터 위치를 바로 알 수 있는지 |
| **생성 속도** | 밀리초당 필요한 ID 개수 |
| **충돌 방지** | 분산 환경에서 고유성 보장 방법 |
| **쿼리 패턴** | 시간 범위 쿼리가 빈번한지 확인 |

### Consistent Hashing

- [Consistent Hashing](/consistenthasing/README.md)

### 합의 알고리즘 (Consensus Algorithm)

합의 알고리즘(Consensus Algorithm)은 분산 시스템에서 여러 노드가 하나의 값이나 상태에 대해 합의하도록 하는 알고리즘이다. 분산 환경에서는 네트워크 지연, 노드 장애, 메시지 유실 등이 발생할 수 있기 때문에 모든 노드가 동일한 상태를 유지하는 것이 어렵다. 합의 알고리즘은 이러한 문제를 해결하여 시스템의 일관성과 신뢰성을 보장한다.

**합의 문제의 핵심 요구사항**:

- **Agreement (합의)**: 모든 정상 노드는 동일한 값에 합의해야 한다
- **Validity (유효성)**: 합의된 값은 어떤 노드가 제안한 값이어야 한다
- **Termination (종료성)**: 모든 정상 노드는 결국 어떤 값에 합의해야 한다
- **Fault Tolerance (내결함성)**: 일부 노드에 장애가 발생해도 시스템이 동작해야 한다

**주요 합의 알고리즘 비교**:

| 알고리즘 | 개발자 | 복잡도 | 성능 | 장점 | 사용 사례 |
|---------|--------|--------|------|------|----------|
| **Paxos** | Leslie Lamport | 높음 | 중간 | 이론적으로 검증됨, 강력한 일관성 | Chubby, Spanner |
| **Raft** | Diego Ongaro | 낮음 | 높음 | 이해하기 쉬움, 구현 용이 | etcd, Consul, CockroachDB |
| **Gossip** | - | 낮음 | 높음 | 확장성, 최종 일관성 | Cassandra, DynamoDB |
| **ZAB** | Yahoo | 중간 | 높음 | 순서 보장, 리더 선출 | ZooKeeper |

### Paxos

Paxos는 Leslie Lamport가 1989년에 발표한 분산 합의 알고리즘으로, 비동기 네트워크 환경에서 노드 장애가 발생해도 안전하게 합의에 도달할 수 있는 방법을 제공한다. Paxos는 이론적으로 매우 우수하지만 이해하고 구현하기 어렵다는 단점이 있다.

**Paxos의 역할**:

- **Proposer (제안자)**: 새로운 값을 제안한다
- **Acceptor (수락자)**: 제안을 수락하거나 거부한다 (과반수 이상이 수락해야 함)
- **Learner (학습자)**: 합의된 값을 학습한다

**Paxos 알고리즘 단계**:

```
Phase 1 (Prepare):
1. Proposer가 제안 번호 n을 생성하고 Prepare(n)을 Acceptor들에게 전송
2. Acceptor는 n보다 큰 번호의 Prepare를 받지 않았다면 Promise(n)으로 응답
   - 이전에 수락한 제안이 있다면 함께 반환

Phase 2 (Accept):
3. Proposer가 과반수 Promise를 받으면 Accept(n, v)를 전송 (v는 제안 값)
4. Acceptor는 n보다 큰 번호의 Prepare를 받지 않았다면 Accept(n, v)를 수락
5. 과반수가 수락하면 합의 완료

Phase 3 (Learn):
6. Learner는 합의된 값을 학습하고 전파
```

**Paxos 구현 예제 (간소화 버전)**:

```java
// PaxosNode.java
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.Map;

public class PaxosNode {
    private final String nodeId;
    private final AtomicInteger highestPromisedProposal = new AtomicInteger(0);
    private Integer acceptedProposal = null;
    private String acceptedValue = null;

    public PaxosNode(String nodeId) {
        this.nodeId = nodeId;
    }

    // Phase 1: Prepare
    public synchronized PrepareResponse prepare(int proposalNumber) {
        if (proposalNumber > highestPromisedProposal.get()) {
            highestPromisedProposal.set(proposalNumber);
            return new PrepareResponse(true, acceptedProposal, acceptedValue);
        }
        return new PrepareResponse(false, null, null);
    }

    // Phase 2: Accept
    public synchronized boolean accept(int proposalNumber, String value) {
        if (proposalNumber >= highestPromisedProposal.get()) {
            highestPromisedProposal.set(proposalNumber);
            acceptedProposal = proposalNumber;
            acceptedValue = value;
            return true;
        }
        return false;
    }

    public String getAcceptedValue() {
        return acceptedValue;
    }
}

// PrepareResponse.java
class PrepareResponse {
    private final boolean promised;
    private final Integer previousProposal;
    private final String previousValue;

    public PrepareResponse(boolean promised, Integer previousProposal, String previousValue) {
        this.promised = promised;
        this.previousProposal = previousProposal;
        this.previousValue = previousValue;
    }

    public boolean isPromised() { return promised; }
    public Integer getPreviousProposal() { return previousProposal; }
    public String getPreviousValue() { return previousValue; }
}

// PaxosProposer.java
import java.util.List;
import java.util.ArrayList;

public class PaxosProposer {
    private final List<PaxosNode> acceptors;
    private int proposalNumber = 0;

    public PaxosProposer(List<PaxosNode> acceptors) {
        this.acceptors = acceptors;
    }

    public String propose(String value) {
        proposalNumber++;

        // Phase 1: Prepare
        List<PrepareResponse> promises = new ArrayList<>();
        for (PaxosNode acceptor : acceptors) {
            PrepareResponse response = acceptor.prepare(proposalNumber);
            if (response.isPromised()) {
                promises.add(response);
            }
        }

        // 과반수 확인
        if (promises.size() < (acceptors.size() / 2 + 1)) {
            return null; // 실패
        }

        // 이전에 수락된 값이 있으면 그 값을 사용
        String proposedValue = value;
        int highestPrevProposal = -1;
        for (PrepareResponse promise : promises) {
            if (promise.getPreviousProposal() != null &&
                promise.getPreviousProposal() > highestPrevProposal) {
                highestPrevProposal = promise.getPreviousProposal();
                proposedValue = promise.getPreviousValue();
            }
        }

        // Phase 2: Accept
        int acceptCount = 0;
        for (PaxosNode acceptor : acceptors) {
            if (acceptor.accept(proposalNumber, proposedValue)) {
                acceptCount++;
            }
        }

        // 과반수 수락 확인
        if (acceptCount >= (acceptors.size() / 2 + 1)) {
            return proposedValue; // 합의 성공
        }

        return null; // 실패
    }
}

// PaxosExample.java
public class PaxosExample {
    public static void main(String[] args) {
        // 5개의 Acceptor 노드 생성
        List<PaxosNode> acceptors = List.of(
            new PaxosNode("Node1"),
            new PaxosNode("Node2"),
            new PaxosNode("Node3"),
            new PaxosNode("Node4"),
            new PaxosNode("Node5")
        );

        // Proposer 생성
        PaxosProposer proposer = new PaxosProposer(acceptors);

        // 값 제안
        String result = proposer.propose("Value-A");
        if (result != null) {
            System.out.println("합의 성공: " + result);
        } else {
            System.out.println("합의 실패");
        }
    }
}
```

**Paxos 사용 사례**:

- **Google Chubby**: 분산 락 서비스에서 리더 선출과 메타데이터 일관성 유지
- **Google Spanner**: 전역 분산 데이터베이스에서 트랜잭션 합의
- **Apache Cassandra**: Lightweight Transactions (LWT)에서 선형화 가능한 연산 제공

### Gossip Protocol

- [Gossip Protocol Explained | HighScalability](http://highscalability.com/blog/2023/7/16/gossip-protocol-explained.html)

Gossip Protocol(가십 프로토콜)은 대규모 분산 시스템에서 정보를 전파하는 분산형 P2P 통신 프로토콜이다. 각 노드가 일정 시간마다 무작위로 선택된 다른 노드들에게 메시지를 전송하는 방식으로 동작하며, 마치 사람들이 소문(gossip)을 퍼뜨리듯이 정보가 확률적으로 전체 시스템에 전파된다.

**Gossip Protocol의 동작 방식**:

```
1. 각 노드는 주기적으로 (예: 1초마다) 실행
2. 무작위로 일부 노드를 선택 (예: fanout=3이면 3개 노드)
3. 선택된 노드들에게 자신의 상태 정보를 전송
4. 수신한 노드는 새로운 정보를 자신의 상태에 병합
5. 수신한 노드도 동일한 방식으로 다른 노드들에게 전파
6. 로그 시간 복잡도로 전체 클러스터에 정보 전파 (O(log N))
```

**주요 특징**:

**장점**:
- **확장성**: 수천 개의 노드로 확장 가능, 중앙 조정자 불필요
- **내결함성**: 일부 노드 장애에도 정보 전파 가능
- **강건성**: 네트워크 분할에도 각 파티션 내에서 동작
- **최종 일관성**: 충분한 시간이 지나면 모든 노드가 동일한 상태에 수렴
- **분산성**: 중앙 서버나 코디네이터 없이 완전 분산 방식
- **간단함**: 구현과 이해가 쉬움

**단점**:
- **최종 일관성**: 강한 일관성을 제공하지 않음 (즉각적인 일관성 보장 안 됨)
- **네트워크 대역폭**: 중복 메시지로 인한 높은 대역폭 소비
- **높은 지연 시간**: 정보가 전체 클러스터에 전파되는데 여러 라운드 필요
- **디버깅 어려움**: 확률적 동작으로 인해 문제 재현과 디버깅이 어려움

**Gossip Protocol 구현 예제**:

```java
// NodeState.java
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class NodeState {
    private final String nodeId;
    private final Map<String, NodeInfo> clusterMembers;
    private long version;

    public NodeState(String nodeId) {
        this.nodeId = nodeId;
        this.clusterMembers = new ConcurrentHashMap<>();
        this.version = 0;

        // 자신의 정보 추가
        clusterMembers.put(nodeId, new NodeInfo(nodeId, System.currentTimeMillis(), NodeStatus.ALIVE));
    }

    public synchronized void update(String nodeId, NodeInfo info) {
        NodeInfo existing = clusterMembers.get(nodeId);
        if (existing == null || info.getVersion() > existing.getVersion()) {
            clusterMembers.put(nodeId, info);
            version++;
        }
    }

    public Map<String, NodeInfo> getClusterMembers() {
        return new ConcurrentHashMap<>(clusterMembers);
    }

    public void markNodeAsFailed(String nodeId) {
        NodeInfo info = clusterMembers.get(nodeId);
        if (info != null) {
            clusterMembers.put(nodeId, new NodeInfo(
                nodeId,
                info.getVersion() + 1,
                NodeStatus.FAILED
            ));
        }
    }
}

// NodeInfo.java
enum NodeStatus { ALIVE, SUSPECTED, FAILED }

class NodeInfo {
    private final String nodeId;
    private final long version;
    private final NodeStatus status;
    private final long lastHeartbeat;

    public NodeInfo(String nodeId, long version, NodeStatus status) {
        this.nodeId = nodeId;
        this.version = version;
        this.status = status;
        this.lastHeartbeat = System.currentTimeMillis();
    }

    public long getVersion() { return version; }
    public NodeStatus getStatus() { return status; }
    public long getLastHeartbeat() { return lastHeartbeat; }
    public String getNodeId() { return nodeId; }
}

// GossipNode.java
import java.util.*;
import java.util.concurrent.*;

public class GossipNode {
    private final String nodeId;
    private final NodeState state;
    private final List<String> seedNodes;
    private final int gossipFanout;
    private final long gossipInterval;
    private final ScheduledExecutorService executor;
    private final Random random;

    public GossipNode(String nodeId, List<String> seedNodes, int gossipFanout, long gossipInterval) {
        this.nodeId = nodeId;
        this.state = new NodeState(nodeId);
        this.seedNodes = new ArrayList<>(seedNodes);
        this.gossipFanout = gossipFanout;
        this.gossipInterval = gossipInterval;
        this.executor = Executors.newScheduledThreadPool(2);
        this.random = new Random();
    }

    public void start() {
        // 주기적으로 Gossip 실행
        executor.scheduleAtFixedRate(
            this::gossip,
            0,
            gossipInterval,
            TimeUnit.MILLISECONDS
        );

        // 주기적으로 장애 감지
        executor.scheduleAtFixedRate(
            this::detectFailures,
            gossipInterval * 2,
            gossipInterval * 2,
            TimeUnit.MILLISECONDS
        );
    }

    private void gossip() {
        Map<String, NodeInfo> members = state.getClusterMembers();
        List<String> targets = selectRandomNodes(members.keySet(), gossipFanout);

        for (String target : targets) {
            if (!target.equals(nodeId)) {
                sendGossipMessage(target, members);
            }
        }
    }

    private void sendGossipMessage(String targetNode, Map<String, NodeInfo> members) {
        // 실제 구현에서는 네트워크 통신 수행
        System.out.println(nodeId + " -> " + targetNode + " : Gossip " + members.size() + " members");

        // 여기서는 시뮬레이션: 상대 노드의 receiveGossip 호출
        // receiveGossip(members);
    }

    public void receiveGossip(Map<String, NodeInfo> receivedMembers) {
        for (Map.Entry<String, NodeInfo> entry : receivedMembers.entrySet()) {
            state.update(entry.getKey(), entry.getValue());
        }
    }

    private void detectFailures() {
        long now = System.currentTimeMillis();
        long failureThreshold = gossipInterval * 10; // 10번의 gossip 주기

        Map<String, NodeInfo> members = state.getClusterMembers();
        for (Map.Entry<String, NodeInfo> entry : members.entrySet()) {
            NodeInfo info = entry.getValue();
            if (info.getStatus() == NodeStatus.ALIVE &&
                (now - info.getLastHeartbeat()) > failureThreshold) {
                System.out.println(nodeId + " detected failure: " + entry.getKey());
                state.markNodeAsFailed(entry.getKey());
            }
        }
    }

    private List<String> selectRandomNodes(Set<String> nodes, int count) {
        List<String> nodeList = new ArrayList<>(nodes);
        Collections.shuffle(nodeList, random);
        return nodeList.subList(0, Math.min(count, nodeList.size()));
    }

    public void shutdown() {
        executor.shutdown();
    }

    public void printClusterState() {
        System.out.println("\n=== Node " + nodeId + " Cluster State ===");
        state.getClusterMembers().forEach((id, info) ->
            System.out.println(id + ": " + info.getStatus())
        );
    }
}

// GossipProtocolExample.java
public class GossipProtocolExample {
    public static void main(String[] args) throws InterruptedException {
        // 5개 노드 클러스터 생성
        List<String> seedNodes = List.of("Node1", "Node2", "Node3", "Node4", "Node5");

        GossipNode node1 = new GossipNode("Node1", seedNodes, 3, 1000);
        GossipNode node2 = new GossipNode("Node2", seedNodes, 3, 1000);
        GossipNode node3 = new GossipNode("Node3", seedNodes, 3, 1000);
        GossipNode node4 = new GossipNode("Node4", seedNodes, 3, 1000);
        GossipNode node5 = new GossipNode("Node5", seedNodes, 3, 1000);

        // 모든 노드 시작
        node1.start();
        node2.start();
        node3.start();
        node4.start();
        node5.start();

        // 10초 동안 실행
        Thread.sleep(10000);

        // 클러스터 상태 출력
        node1.printClusterState();
        node3.printClusterState();
        node5.printClusterState();

        // 종료
        node1.shutdown();
        node2.shutdown();
        node3.shutdown();
        node4.shutdown();
        node5.shutdown();
    }
}
```

**Gossip Protocol 사용 사례**:

**Amazon DynamoDB**:
- **클러스터 멤버십 관리**: 각 노드가 클러스터의 다른 노드에 대한 정보를 유지한다. 노드는 일정 간격으로 무작위로 선택한 이웃 노드와 정보를 교환하여 전체 클러스터 상태를 학습한다.
- **장애 감지**: 노드가 주기적으로 이웃 노드와 정보를 교환하면서 응답하지 않는 노드를 식별한다. 장애가 발견되면 메시지가 전체 클러스터로 전파되어 다른 노드들이 적절한 조치를 취할 수 있다 (예: 데이터 복제를 위한 새로운 노드 선택).

**Apache Cassandra**:
- **노드 상태 전파**: Gossip을 통해 각 노드가 클러스터의 다른 모든 노드 상태를 학습
- **스키마 변경 전파**: DDL 변경사항을 전체 클러스터에 전파
- **토큰 범위 정보**: 각 노드가 담당하는 데이터 범위 정보 공유

**HashiCorp Consul**:
- **서비스 디스커버리**: 서비스 인스턴스 추가/제거 정보 전파
- **헬스 체크 결과**: 서비스 상태 정보를 클러스터 전체에 공유

### Raft

Raft는 Diego Ongaro와 John Ousterhout가 2013년에 발표한 합의 알고리즘으로, Paxos보다 이해하기 쉽고 구현하기 쉽도록 설계되었다. Raft는 "Understandability"를 주요 설계 목표로 하여 교육용으로도 널리 사용된다.

**Raft의 핵심 개념**:

- **리더 선출 (Leader Election)**: 클러스터에서 하나의 리더를 선출
- **로그 복제 (Log Replication)**: 리더가 로그 엔트리를 팔로워들에게 복제
- **안전성 (Safety)**: 합의된 값은 변경되지 않음을 보장

**Raft 노드의 세 가지 상태**:

- **Leader (리더)**: 클라이언트 요청을 처리하고 로그를 복제
- **Follower (팔로워)**: 리더의 로그를 수신하고 복제
- **Candidate (후보자)**: 리더 선출을 시작한 노드

**Raft 동작 과정**:

```
1. 리더 선출:
   - 모든 노드는 Follower로 시작
   - Election Timeout이 지나면 Candidate가 되어 투표 요청
   - 과반수 투표를 받으면 Leader가 됨
   - Leader는 주기적으로 Heartbeat 전송

2. 로그 복제:
   - 클라이언트가 Leader에게 명령 전송
   - Leader는 로그 엔트리를 생성하고 Follower들에게 복제
   - 과반수가 로그를 저장하면 Leader가 커밋
   - Leader는 Follower들에게 커밋 정보 전송

3. 안전성 보장:
   - Leader는 이전 Term의 모든 커밋된 엔트리를 포함
   - 로그 일치 규칙: 같은 인덱스와 Term이면 동일한 명령
   - 리더 완전성: 커밋된 엔트리는 미래의 모든 리더에 존재
```

**Raft 구현 예제 (간소화 버전)**:

```java
// RaftNode.java
import java.util.*;
import java.util.concurrent.*;

enum NodeState { FOLLOWER, CANDIDATE, LEADER }

public class RaftNode {
    private final String nodeId;
    private NodeState state;
    private int currentTerm;
    private String votedFor;
    private List<LogEntry> log;
    private int commitIndex;
    private int lastApplied;

    // Leader 전용
    private Map<String, Integer> nextIndex;
    private Map<String, Integer> matchIndex;

    private final List<RaftNode> cluster;
    private final Random random;
    private ScheduledExecutorService executor;
    private long lastHeartbeat;
    private long electionTimeout;

    public RaftNode(String nodeId, List<RaftNode> cluster) {
        this.nodeId = nodeId;
        this.state = NodeState.FOLLOWER;
        this.currentTerm = 0;
        this.votedFor = null;
        this.log = new ArrayList<>();
        this.commitIndex = 0;
        this.lastApplied = 0;
        this.cluster = cluster;
        this.random = new Random();
        this.lastHeartbeat = System.currentTimeMillis();
        this.electionTimeout = 150 + random.nextInt(150); // 150-300ms
        this.executor = Executors.newScheduledThreadPool(2);
    }

    public void start() {
        executor.scheduleAtFixedRate(this::checkElectionTimeout, 50, 50, TimeUnit.MILLISECONDS);
    }

    private void checkElectionTimeout() {
        if (state != NodeState.LEADER &&
            System.currentTimeMillis() - lastHeartbeat > electionTimeout) {
            startElection();
        }
    }

    // 리더 선출 시작
    private synchronized void startElection() {
        state = NodeState.CANDIDATE;
        currentTerm++;
        votedFor = nodeId;
        lastHeartbeat = System.currentTimeMillis();

        System.out.println(nodeId + " starting election for term " + currentTerm);

        int lastLogIndex = log.size() - 1;
        int lastLogTerm = lastLogIndex >= 0 ? log.get(lastLogIndex).getTerm() : 0;

        int votes = 1; // 자신에게 투표
        for (RaftNode node : cluster) {
            if (!node.nodeId.equals(nodeId)) {
                VoteResponse response = node.requestVote(
                    currentTerm, nodeId, lastLogIndex, lastLogTerm
                );
                if (response.isVoteGranted()) {
                    votes++;
                }
            }
        }

        // 과반수 획득 시 리더가 됨
        if (votes > (cluster.size() / 2)) {
            becomeLeader();
        }
    }

    // 투표 요청 처리
    public synchronized VoteResponse requestVote(
        int term, String candidateId, int lastLogIndex, int lastLogTerm) {

        if (term < currentTerm) {
            return new VoteResponse(currentTerm, false);
        }

        if (term > currentTerm) {
            currentTerm = term;
            state = NodeState.FOLLOWER;
            votedFor = null;
        }

        boolean voteGranted = false;
        if ((votedFor == null || votedFor.equals(candidateId)) &&
            isLogUpToDate(lastLogIndex, lastLogTerm)) {
            votedFor = candidateId;
            voteGranted = true;
            lastHeartbeat = System.currentTimeMillis();
        }

        return new VoteResponse(currentTerm, voteGranted);
    }

    private boolean isLogUpToDate(int candidateLastIndex, int candidateLastTerm) {
        int lastIndex = log.size() - 1;
        int lastTerm = lastIndex >= 0 ? log.get(lastIndex).getTerm() : 0;

        if (candidateLastTerm != lastTerm) {
            return candidateLastTerm >= lastTerm;
        }
        return candidateLastIndex >= lastIndex;
    }

    // 리더가 되기
    private synchronized void becomeLeader() {
        state = NodeState.LEADER;
        System.out.println(nodeId + " became LEADER for term " + currentTerm);

        nextIndex = new HashMap<>();
        matchIndex = new HashMap<>();
        for (RaftNode node : cluster) {
            nextIndex.put(node.nodeId, log.size());
            matchIndex.put(node.nodeId, 0);
        }

        // 주기적으로 Heartbeat 전송
        executor.scheduleAtFixedRate(this::sendHeartbeats, 0, 50, TimeUnit.MILLISECONDS);
    }

    // Heartbeat 전송
    private void sendHeartbeats() {
        if (state != NodeState.LEADER) return;

        for (RaftNode node : cluster) {
            if (!node.nodeId.equals(nodeId)) {
                int prevLogIndex = nextIndex.get(node.nodeId) - 1;
                int prevLogTerm = prevLogIndex >= 0 ? log.get(prevLogIndex).getTerm() : 0;

                List<LogEntry> entries = new ArrayList<>();
                if (log.size() > nextIndex.get(node.nodeId)) {
                    entries = log.subList(nextIndex.get(node.nodeId), log.size());
                }

                AppendEntriesResponse response = node.appendEntries(
                    currentTerm, nodeId, prevLogIndex, prevLogTerm,
                    entries, commitIndex
                );

                if (response.isSuccess()) {
                    if (!entries.isEmpty()) {
                        nextIndex.put(node.nodeId, log.size());
                        matchIndex.put(node.nodeId, log.size() - 1);
                    }
                }
            }
        }
    }

    // 로그 엔트리 추가 요청 처리
    public synchronized AppendEntriesResponse appendEntries(
        int term, String leaderId, int prevLogIndex, int prevLogTerm,
        List<LogEntry> entries, int leaderCommit) {

        lastHeartbeat = System.currentTimeMillis();

        if (term < currentTerm) {
            return new AppendEntriesResponse(currentTerm, false);
        }

        if (term > currentTerm) {
            currentTerm = term;
            state = NodeState.FOLLOWER;
            votedFor = null;
        }

        if (state == NodeState.CANDIDATE) {
            state = NodeState.FOLLOWER;
        }

        // 로그 일치 확인
        if (prevLogIndex >= 0 &&
            (prevLogIndex >= log.size() || log.get(prevLogIndex).getTerm() != prevLogTerm)) {
            return new AppendEntriesResponse(currentTerm, false);
        }

        // 로그 엔트리 추가
        for (int i = 0; i < entries.size(); i++) {
            int index = prevLogIndex + 1 + i;
            if (index < log.size()) {
                if (log.get(index).getTerm() != entries.get(i).getTerm()) {
                    log = log.subList(0, index);
                    log.add(entries.get(i));
                }
            } else {
                log.add(entries.get(i));
            }
        }

        if (leaderCommit > commitIndex) {
            commitIndex = Math.min(leaderCommit, log.size() - 1);
        }

        return new AppendEntriesResponse(currentTerm, true);
    }

    // 클라이언트 요청 처리 (리더만 가능)
    public synchronized boolean clientRequest(String command) {
        if (state != NodeState.LEADER) {
            return false;
        }

        LogEntry entry = new LogEntry(currentTerm, command);
        log.add(entry);
        System.out.println(nodeId + " received command: " + command);

        return true;
    }

    public NodeState getState() { return state; }
    public int getCurrentTerm() { return currentTerm; }
}

// LogEntry.java
class LogEntry {
    private final int term;
    private final String command;

    public LogEntry(int term, String command) {
        this.term = term;
        this.command = command;
    }

    public int getTerm() { return term; }
    public String getCommand() { return command; }
}

// VoteResponse.java
class VoteResponse {
    private final int term;
    private final boolean voteGranted;

    public VoteResponse(int term, boolean voteGranted) {
        this.term = term;
        this.voteGranted = voteGranted;
    }

    public int getTerm() { return term; }
    public boolean isVoteGranted() { return voteGranted; }
}

// AppendEntriesResponse.java
class AppendEntriesResponse {
    private final int term;
    private final boolean success;

    public AppendEntriesResponse(int term, boolean success) {
        this.term = term;
        this.success = success;
    }

    public int getTerm() { return term; }
    public boolean isSuccess() { return success; }
}

// RaftExample.java
public class RaftExample {
    public static void main(String[] args) throws InterruptedException {
        // 5개 노드 클러스터 생성
        List<RaftNode> cluster = new ArrayList<>();
        RaftNode node1 = new RaftNode("Node1", cluster);
        RaftNode node2 = new RaftNode("Node2", cluster);
        RaftNode node3 = new RaftNode("Node3", cluster);
        RaftNode node4 = new RaftNode("Node4", cluster);
        RaftNode node5 = new RaftNode("Node5", cluster);

        cluster.addAll(List.of(node1, node2, node3, node4, node5));

        // 모든 노드 시작
        node1.start();
        node2.start();
        node3.start();
        node4.start();
        node5.start();

        // 리더 선출 대기
        Thread.sleep(1000);

        // 리더 찾기
        RaftNode leader = cluster.stream()
            .filter(n -> n.getState() == NodeState.LEADER)
            .findFirst()
            .orElse(null);

        if (leader != null) {
            System.out.println("\n=== Sending client requests ===");
            leader.clientRequest("SET x = 10");
            leader.clientRequest("SET y = 20");
            leader.clientRequest("ADD x y");
        }

        // 5초 동안 실행
        Thread.sleep(5000);
    }
}
```

**Raft 사용 사례**:

- **etcd**: Kubernetes의 핵심 컴포넌트로 클러스터 구성 정보 저장
- **HashiCorp Consul**: 서비스 디스커버리와 구성 관리
- **CockroachDB**: 분산 SQL 데이터베이스의 트랜잭션 합의
- **TiKV**: PingCAP의 분산 Key-Value 스토어

**Raft vs Paxos 비교**:

| 특징 | Raft | Paxos |
|------|------|-------|
| **이해 용이성** | 쉬움 (교육용으로 설계) | 어려움 (이론적으로 복잡) |
| **구현 복잡도** | 낮음 | 높음 |
| **리더 선출** | 명확한 리더 선출 과정 | Multi-Paxos에서 암묵적 리더 |
| **로그 구조** | 강한 리더, 순차적 로그 | 로그 구멍 허용 |
| **성능** | 유사 | 유사 |
| **멤버십 변경** | Joint Consensus | 복잡 |

### Chubby

Chubby는 Google이 개발한 분산 락 서비스(Distributed Lock Service)로, 분산 시스템에서 활동을 조정하고 공유 리소스를 관리하기 위해 사용된다. Chubby는 느슨하게 결합된(coarse-grained) 락 메커니즘을 제공하여 시스템이 클러스터 전체에서 일관성, 동기화, 내결함성을 유지할 수 있도록 한다.

**Chubby의 핵심 개념**:

Chubby는 클라이언트에게 파일 시스템과 유사한 인터페이스를 제공하며, 디렉토리와 파일로 구성된 계층적 네임스페이스를 사용한다. Chubby의 각 파일은 클라이언트가 획득할 수 있는 락을 나타내며, 해당 락과 연관된 공유 리소스에 대한 배타적 접근을 제공한다. Chubby 파일은 또한 소량의 메타데이터를 저장할 수 있어 분산 시스템이 락의 상태를 기반으로 의사 결정을 할 수 있다.

**Chubby 아키텍처**:

```
┌─────────────────────────────────────────────────────┐
│                  Client Applications                 │
│  (GFS, Bigtable, Borg, BigQuery, etc.)              │
└────────────┬────────────────────────────┬───────────┘
             │                            │
             │  Chubby Client Library     │
             │                            │
             v                            v
┌─────────────────────────────────────────────────────┐
│           Chubby Cell (5 replicas)                  │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │  Master  │  │ Replica  │  │ Replica  │ ...     │
│  │(Active)  │  │(Passive) │  │(Passive) │         │
│  └────┬─────┘  └──────────┘  └──────────┘         │
│       │                                             │
│       │ Paxos Consensus                            │
│       v                                             │
│  ┌──────────────────────────────┐                  │
│  │  Persistent Storage (DB)     │                  │
│  └──────────────────────────────┘                  │
└─────────────────────────────────────────────────────┘
```

**Chubby의 주요 특징**:

**1. 일관성과 장애 조치 (Consistency & Failover)**:
- Paxos 합의 알고리즘을 사용하여 서버 간 일관된 상태 유지
- 개별 노드 장애를 처리
- 일반적으로 5대의 서버로 구성된 Chubby Cell 사용
- 단일 Master 서버가 주어진 시간에 클라이언트 요청 처리
- Master 장애 시 Paxos를 통해 새로운 Master 선출

**2. 캐싱 (Caching)**:
- 클라이언트는 Chubby 락을 캐시하고 지정된 기간 동안 임대(lease)
- Chubby 서버와의 빈번한 통신 필요성 최소화
- 성능 향상 및 네트워크 트래픽 감소

**3. 이벤트 알림 (Event Notifications)**:
- Watch 메커니즘을 통해 클라이언트가 락 상태를 모니터링
- 락 소유권이나 내용 변경 감지 가능
- 파일 수정, 삭제, 자식 노드 추가/제거 등의 이벤트 알림

**4. 락 만료 및 Keep-alive (Lock Expiration & Keep-alive)**:
- 락이 무기한 보유되는 것을 방지
- 임대(lease) 기반 만료 메커니즘
- 클라이언트는 주기적으로 갱신하여 소유권 유지
- 장애가 발생한 클라이언트가 보유한 락은 결국 해제됨

**Chubby 구현 예제 (Chubby와 유사한 ZooKeeper 사용)**:

```java
// ChubbyLikeService.java (ZooKeeper 기반)
import org.apache.zookeeper.*;
import org.apache.zookeeper.data.Stat;
import java.util.concurrent.CountDownLatch;

public class ChubbyLikeService {
    private ZooKeeper zooKeeper;
    private final String connectionString;
    private final int sessionTimeout;
    private final CountDownLatch connectedSignal = new CountDownLatch(1);

    public ChubbyLikeService(String connectionString, int sessionTimeout) {
        this.connectionString = connectionString;
        this.sessionTimeout = sessionTimeout;
    }

    public void connect() throws Exception {
        zooKeeper = new ZooKeeper(connectionString, sessionTimeout, event -> {
            if (event.getState() == Watcher.Event.KeeperState.SyncConnected) {
                connectedSignal.countDown();
            }
        });
        connectedSignal.await();
    }

    // 파일/디렉토리 생성 (락 생성)
    public String createNode(String path, byte[] data, boolean ephemeral) throws Exception {
        CreateMode mode = ephemeral ? CreateMode.EPHEMERAL : CreateMode.PERSISTENT;
        return zooKeeper.create(path, data, ZooDefs.Ids.OPEN_ACL_UNSAFE, mode);
    }

    // 락 획득 시도
    public boolean acquireLock(String lockPath, byte[] data) throws Exception {
        try {
            zooKeeper.create(
                lockPath,
                data,
                ZooDefs.Ids.OPEN_ACL_UNSAFE,
                CreateMode.EPHEMERAL
            );
            System.out.println("Lock acquired: " + lockPath);
            return true;
        } catch (KeeperException.NodeExistsException e) {
            System.out.println("Lock already held by another client: " + lockPath);
            return false;
        }
    }

    // 락 해제
    public void releaseLock(String lockPath) throws Exception {
        zooKeeper.delete(lockPath, -1);
        System.out.println("Lock released: " + lockPath);
    }

    // 데이터 읽기
    public byte[] getData(String path) throws Exception {
        Stat stat = new Stat();
        return zooKeeper.getData(path, false, stat);
    }

    // 데이터 쓰기 (메타데이터 업데이트)
    public void setData(String path, byte[] data) throws Exception {
        zooKeeper.setData(path, data, -1);
    }

    // Watch 설정 (이벤트 알림)
    public void watchNode(String path, Watcher watcher) throws Exception {
        zooKeeper.getData(path, watcher, null);
    }

    public void close() throws InterruptedException {
        zooKeeper.close();
    }
}

// DistributedLockExample.java
public class DistributedLockExample {
    public static void main(String[] args) throws Exception {
        String zkConnection = "localhost:2181";
        ChubbyLikeService client1 = new ChubbyLikeService(zkConnection, 5000);
        ChubbyLikeService client2 = new ChubbyLikeService(zkConnection, 5000);

        client1.connect();
        client2.connect();

        String lockPath = "/locks/resource-lock";

        // Client 1이 락 획득 시도
        boolean acquired1 = client1.acquireLock(lockPath, "client1".getBytes());

        // Client 2가 락 획득 시도 (실패)
        boolean acquired2 = client2.acquireLock(lockPath, "client2".getBytes());

        if (acquired1) {
            // Client 1이 임계 영역에서 작업 수행
            System.out.println("Client 1 working in critical section...");
            Thread.sleep(2000);

            // 락 해제
            client1.releaseLock(lockPath);
        }

        // 이제 Client 2가 락 획득 가능
        acquired2 = client2.acquireLock(lockPath, "client2".getBytes());
        if (acquired2) {
            System.out.println("Client 2 working in critical section...");
            Thread.sleep(2000);
            client2.releaseLock(lockPath);
        }

        client1.close();
        client2.close();
    }
}

// LeaderElectionExample.java (리더 선출 시나리오)
public class LeaderElectionExample {
    private ChubbyLikeService chubbyService;
    private final String leaderPath = "/election/leader";
    private final String nodeId;

    public LeaderElectionExample(String nodeId, String zkConnection) throws Exception {
        this.nodeId = nodeId;
        this.chubbyService = new ChubbyLikeService(zkConnection, 5000);
        chubbyService.connect();
    }

    public void electLeader() throws Exception {
        boolean isLeader = chubbyService.acquireLock(
            leaderPath,
            nodeId.getBytes()
        );

        if (isLeader) {
            System.out.println(nodeId + " became the LEADER");
            performLeaderDuties();
        } else {
            System.out.println(nodeId + " is a FOLLOWER");
            watchLeader();
        }
    }

    private void performLeaderDuties() {
        System.out.println(nodeId + " performing leader duties...");
        // 리더 작업 수행
    }

    private void watchLeader() throws Exception {
        // 리더 노드 변경 감지
        chubbyService.watchNode(leaderPath, event -> {
            if (event.getType() == Watcher.Event.EventType.NodeDeleted) {
                System.out.println(nodeId + " detected leader failure, starting new election");
                try {
                    electLeader(); // 새로운 선거 시작
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }

    public static void main(String[] args) throws Exception {
        String zkConnection = "localhost:2181";

        // 3개 노드가 리더 선출에 참여
        LeaderElectionExample node1 = new LeaderElectionExample("Node1", zkConnection);
        LeaderElectionExample node2 = new LeaderElectionExample("Node2", zkConnection);
        LeaderElectionExample node3 = new LeaderElectionExample("Node3", zkConnection);

        node1.electLeader();
        Thread.sleep(500);
        node2.electLeader();
        Thread.sleep(500);
        node3.electLeader();

        // 10초 동안 실행
        Thread.sleep(10000);
    }
}

// ConfigurationManagementExample.java (구성 관리)
public class ConfigurationManagementExample {
    private ChubbyLikeService chubbyService;
    private final String configPath = "/config/app-config";

    public ConfigurationManagementExample(String zkConnection) throws Exception {
        this.chubbyService = new ChubbyLikeService(zkConnection, 5000);
        chubbyService.connect();
    }

    public void setConfiguration(String config) throws Exception {
        try {
            chubbyService.createNode(configPath, config.getBytes(), false);
            System.out.println("Configuration created: " + config);
        } catch (KeeperException.NodeExistsException e) {
            chubbyService.setData(configPath, config.getBytes());
            System.out.println("Configuration updated: " + config);
        }
    }

    public String getConfiguration() throws Exception {
        byte[] data = chubbyService.getData(configPath);
        return new String(data);
    }

    public void watchConfiguration() throws Exception {
        chubbyService.watchNode(configPath, event -> {
            if (event.getType() == Watcher.Event.EventType.NodeDataChanged) {
                try {
                    String newConfig = getConfiguration();
                    System.out.println("Configuration changed: " + newConfig);
                    applyConfiguration(newConfig);

                    // Watch 재설정 (ZooKeeper는 one-time watch)
                    watchConfiguration();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });
    }

    private void applyConfiguration(String config) {
        System.out.println("Applying new configuration: " + config);
        // 애플리케이션에 새로운 구성 적용
    }

    public static void main(String[] args) throws Exception {
        ConfigurationManagementExample configManager =
            new ConfigurationManagementExample("localhost:2181");

        // 초기 구성 설정
        configManager.setConfiguration("max_connections=100,timeout=30");

        // 구성 변경 감지
        configManager.watchConfiguration();

        // 2초 후 구성 변경
        Thread.sleep(2000);
        configManager.setConfiguration("max_connections=200,timeout=60");

        Thread.sleep(5000);
    }
}
```

**Chubby 사용 사례**:

- **Google File System (GFS)**: Master 선출 및 메타데이터 관리
- **Bigtable**: Tablet 서버 간 조정 및 Master 선출
- **Borg**: 클러스터 관리 시스템에서 리소스 할당 조정
- **BigQuery**: 분산 쿼리 실행을 위한 작업 조정

**Chubby vs ZooKeeper vs etcd 비교**:

| 특징 | Chubby | ZooKeeper | etcd |
|------|--------|-----------|------|
| **개발사** | Google (비공개) | Apache (오픈소스) | CoreOS/CNCF (오픈소스) |
| **합의 알고리즘** | Paxos | ZAB (Paxos 변형) | Raft |
| **인터페이스** | 파일시스템 API | 파일시스템 API | HTTP/gRPC API |
| **데이터 모델** | 계층적 네임스페이스 | 계층적 네임스페이스 | Key-Value |
| **캐싱** | 강력한 클라이언트 캐싱 | 제한적 캐싱 | 제한적 캐싱 |
| **Watch** | 지원 | 지원 (one-time) | 지원 (continuous) |
| **언어** | C++ | Java | Go |
| **사용 사례** | Google 내부 서비스 | Hadoop, Kafka, HBase | Kubernetes, Cloud Native |
| **성능** | 높음 | 높음 | 매우 높음 |

Chubby는 Google 내부용으로 공개되지 않았지만, Apache ZooKeeper와 etcd는 Chubby에서 영감을 받아 분산 시스템 조정 기능을 제공하는 오픈소스 대안이다.

### Distributed Locking

- [How to do distributed locking](https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html)

#### Redis Redlock의 한계

Redis의 Redlock 알고리즘은 다음과 같은 단점이 있습니다:

- **Fencing Tokens 미지원**: Redlock은 Fencing Tokens를 생성하지 않아 클라이언트 간 경쟁 조건을 방지할 수 없습니다. 여러 클라이언트가 동시에 같은 리소스에 접근할 때 안전성을 보장하지 못합니다.

- **타이밍 가정의 문제**: Redlock은 동기적 시스템 모델을 가정합니다. 네트워크 지연, 프로세스 일시 중지, 클럭 오류에 대한 정확한 시간을 알 수 있다고 가정하지만, 실제 분산 시스템에서는 이러한 가정이 항상 지켜지지 않습니다.

- **안전성 위반 가능성**: 타이밍 문제가 발생할 경우 Redlock의 일관성 메커니즘이 안전성을 위반할 수 있습니다.

#### Redis Redisson 구현 예제

다음은 Redis와 Redisson을 이용한 분산 락 구현입니다:

```java
import org.redisson.Redisson;
import org.redisson.api.RLock;
import org.redisson.api.RedissonClient;
import org.redisson.config.Config;

public class RedissonExample {

    public static void main(String[] args) {
        // Redisson 설정
        Config config = new Config();
        config.useSingleServer().setAddress("redis://127.0.0.1:6379");

        // Redisson 클라이언트 생성
        RedissonClient redisson = Redisson.create(config);

        // 락을 획득할 리소스 정의
        String resource = "myResource";

        // 리소스에 대한 락 획득
        RLock lock = redisson.getLock(resource);
        lock.lock(); // 블로킹 호출, 락을 획득할 때까지 대기

        try {
            // 락으로 보호되는 작업 수행
            performProtectedActions();
        } finally {
            // 락 해제
            lock.unlock();
        }

        // Redisson 클라이언트 종료
        redisson.shutdown();
    }

    private static void performProtectedActions() {
        // 분산 락이 필요한 코드 작성
    }
}
```

#### ZooKeeper를 사용한 안전한 분산 락

강력한 일관성과 안전성이 필요한 분산 락의 경우, Apache ZooKeeper와 같은 합의 알고리즘 기반 솔루션을 사용하는 것이 좋습니다. ZooKeeper는 분산 락 및 분산 시스템 조정 작업을 안전하고 일관성 있게 처리하도록 설계되었습니다.

**ZooKeeper가 Redis보다 나은 이유:**

- **안정성**: ZooKeeper는 자체 합의 프로토콜(ZAB)을 사용하여 특정 노드가 실패해도 클러스터 내 노드 간 동기화를 유지합니다. 반면 Redlock은 클러스터 작동에 필요한 안정성과 상호 운용성 제공에 한계가 있습니다.

- **강력한 일관성**: ZooKeeper는 분산 락 전용으로 설계되어 높은 가용성과 일관성을 제공합니다. Redis는 원래 캐싱과 메시징용으로 설계되어 일관성보다 가용성에 중점을 두었습니다.

- **Fencing 메커니즘**: ZooKeeper 클라이언트는 Fencing Tokens을 생성하여 분산 락의 격리 수준을 높입니다. 이 토큰을 통해 동시에 여러 클라이언트가 같은 락에 접근하는 것을 방지합니다. Redlock은 이를 자체적으로 지원하지 않습니다.

**선택 가이드:**
- **ZooKeeper**: 금융, 재고 관리 등 강력한 일관성이 필요한 경우
- **Redis (Redisson)**: 일시적 데이터나 메시징 등 상대적으로 덜 중요한 경우

#### ZooKeeper Curator 구현 예제

다음은 ZooKeeper와 Curator를 이용한 분산 락 구현입니다:

```groovy
// Gradle
dependencies {
    implementation 'org.apache.curator:curator-recipes:5.1.0'
    implementation 'org.apache.curator:curator-framework:5.1.0'
    ...
}
```

```java
import org.apache.curator.framework.CuratorFramework;
import org.apache.curator.framework.CuratorFrameworkFactory;
import org.apache.curator.framework.recipes.locks.InterProcessMutex;
import org.apache.curator.retry.ExponentialBackoffRetry;
import java.util.concurrent.TimeUnit;

public class ZookeeperExample {

    public static void main(String[] args) {
        // ZooKeeper 설정
        String zkConnectionString = "localhost:2181";
        int baseSleepTimeMs = 1000;
        int maxRetries = 3;

        // ZooKeeper용 Curator 클라이언트 생성
        CuratorFramework zookeeperClient = CuratorFrameworkFactory.newClient(
                zkConnectionString,
                new ExponentialBackoffRetry(baseSleepTimeMs, maxRetries)
        );
        zookeeperClient.start();

        // 락을 획득할 리소스 정의
        String lockPath = "/locks/myResource";

        // 리소스에 대한 락 획득
        InterProcessMutex lock = new InterProcessMutex(zookeeperClient, lockPath);

        try {
            // 10초 내에 락 획득 시도
            if (lock.acquire(10, TimeUnit.SECONDS)) {
                try {
                    // 락으로 보호되는 작업 수행
                    performProtectedActions();
                } finally {
                    // 락 해제
                    lock.release();
                }
            } else {
                // 락 획득 실패 시 예외 발생 또는 처리
                throw new IllegalStateException("락 획득 실패");
            }
        } catch (Exception e) {
            // 락 획득 또는 보호된 작업 중 예외 처리
            e.printStackTrace();
        } finally {
            // ZooKeeper 클라이언트 종료
            zookeeperClient.close();
        }
    }

    private static void performProtectedActions() {
        // 분산 락이 필요한 코드 작성
    }
}
```

### Distributed Tracing

- [What is distributed tracing, and why is it important?](https://www.dynatrace.com/news/blog/what-is-distributed-tracing/)

#### 개요

분산 추적(Distributed Tracing)은 고유 식별자를 사용하여 분산 클라우드 환경을 통해 전파되는 요청을 관찰하는 방법입니다. 이를 통해 다음을 제공합니다:

- 사용자 경험, 애플리케이션 계층, 인프라에 대한 실시간 가시성
- 애플리케이션 성능 향상
- 서비스 수준 계약(SLA) 준수
- 내부 협업 개선
- 평균 탐지 시간(MTTD) 및 평균 수리 시간(MTTR) 감소

#### 필요성

모놀리식 애플리케이션이 마이크로서비스 아키텍처로 발전하면서, 복잡한 클라우드 네이티브 환경에서는 전통적인 모니터링 도구만으로는 효과적인 관측이 어렵습니다. 분산 추적은 이러한 환경에서 관측 가능성(Observability)을 확보하기 위한 필수 요소가 되었습니다.

## 낙관적 락 vs 비관적 락 (Optimistic Lock vs Pessimistic Lock)

> * [비관적 Lock, 낙관적 Lock 이해하기](https://medium.com/@jinhanchoi1/%EB%B9%84%EA%B4%80%EC%A0%81-lock-%EB%82%99%EA%B4%80%EC%A0%81-lock-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0-1986a399a54)
> * [Optimistic Locking in JPA](https://www.baeldung.com/jpa-optimistic-locking)
> * [Pessimistic Locking in JPA](https://www.baeldung.com/jpa-pessimistic-locking)

### 개념

데이터베이스의 Isolation Level보다 더 융통성 있게 동시성을 제어하는 락킹 방법입니다. 동시에 여러 트랜잭션이 같은 데이터를 수정할 때 데이터 일관성을 보장하기 위해 사용됩니다.

### 비교

| 구분 | 낙관적 락 (Optimistic Lock) | 비관적 락 (Pessimistic Lock) |
|------|---------------------------|----------------------------|
| **기본 전략** | 충돌이 드물다고 가정 | 충돌이 자주 발생한다고 가정 |
| **락 시점** | 커밋 시점에 충돌 검증 | 데이터 읽을 때 락 획득 |
| **성능** | 읽기 작업이 많을 때 유리 | 쓰기 작업이 많을 때 유리 |
| **충돌 처리** | 롤백 후 재시도 | 대기 후 락 획득 |
| **구현 방식** | Version 또는 Timestamp 사용 | SELECT FOR UPDATE 사용 |
| **데드락** | 발생 안 함 | 발생 가능 |
| **적용 분야** | 조회 위주 서비스, CMS | 금융, 재고, 좌석 예약 |

### 낙관적 락 (Optimistic Lock)

**동작 원리**:
1. 테이블에 `version` 또는 `timestamp` 컬럼 추가
2. 데이터를 읽을 때 version 값도 함께 조회
3. 데이터를 수정할 때 version 값을 조건에 포함
4. version이 일치하면 업데이트 성공, 불일치하면 실패 (다른 트랜잭션이 먼저 수정함)

**JPA 구현 예제**:

```java
@Entity
@Table(name = "products")
public class Product {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String name;
    private Integer stock;
    private BigDecimal price;

    // 낙관적 락을 위한 버전 필드
    @Version
    private Long version;

    // Getters and Setters
}

@Service
public class ProductService {

    @Autowired
    private ProductRepository productRepository;

    @Transactional
    public void decreaseStock(Long productId, int quantity) {
        // 1. version을 포함하여 조회 (예: version = 1)
        Product product = productRepository.findById(productId)
            .orElseThrow(() -> new ProductNotFoundException());

        // 2. 재고 확인
        if (product.getStock() < quantity) {
            throw new InsufficientStockException();
        }

        // 3. 재고 감소
        product.setStock(product.getStock() - quantity);

        // 4. 저장 시 자동으로 다음 쿼리 실행:
        // UPDATE products
        // SET stock = ?, version = version + 1
        // WHERE id = ? AND version = 1
        //
        // 만약 다른 트랜잭션이 먼저 수정했다면 version이 2가 되어
        // WHERE 조건이 맞지 않아 업데이트 실패 (0 rows affected)
        // -> OptimisticLockException 발생
        try {
            productRepository.save(product);
        } catch (OptimisticLockException e) {
            throw new ConcurrentModificationException(
                "다른 사용자가 동시에 수정했습니다. 다시 시도해주세요.");
        }
    }

    // 재시도 로직을 포함한 버전
    @Retryable(
        value = OptimisticLockException.class,
        maxAttempts = 3,
        backoff = @Backoff(delay = 100)
    )
    @Transactional
    public void decreaseStockWithRetry(Long productId, int quantity) {
        Product product = productRepository.findById(productId)
            .orElseThrow(() -> new ProductNotFoundException());

        if (product.getStock() < quantity) {
            throw new InsufficientStockException();
        }

        product.setStock(product.getStock() - quantity);
        productRepository.save(product);
    }
}
```

**수동 버전 관리 예제** (JPA 없이):

```java
@Repository
public class ProductRepositoryImpl {

    @Autowired
    private JdbcTemplate jdbcTemplate;

    public boolean updateWithOptimisticLock(Long productId, int newStock, Long expectedVersion) {
        String sql = """
            UPDATE products
            SET stock = ?, version = version + 1, updated_at = NOW()
            WHERE id = ? AND version = ?
            """;

        int rowsAffected = jdbcTemplate.update(sql, newStock, productId, expectedVersion);

        // rowsAffected가 0이면 다른 트랜잭션이 먼저 수정함
        return rowsAffected > 0;
    }
}

@Service
public class ProductService {

    @Autowired
    private ProductRepositoryImpl productRepository;

    public void decreaseStock(Long productId, int quantity) {
        int maxRetries = 3;
        int attempt = 0;

        while (attempt < maxRetries) {
            try {
                // 1. 현재 데이터와 version 조회
                Product product = productRepository.findById(productId);
                Long currentVersion = product.getVersion();

                // 2. 재고 확인
                if (product.getStock() < quantity) {
                    throw new InsufficientStockException();
                }

                // 3. 새 재고 계산
                int newStock = product.getStock() - quantity;

                // 4. Optimistic Lock으로 업데이트 시도
                boolean success = productRepository.updateWithOptimisticLock(
                    productId, newStock, currentVersion);

                if (success) {
                    return; // 성공
                } else {
                    // 실패 - 재시도
                    attempt++;
                    Thread.sleep(50 * attempt); // 백오프
                }

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("재시도 중단", e);
            }
        }

        throw new ConcurrentModificationException("재고 차감 실패: 최대 재시도 횟수 초과");
    }
}
```

### 비관적 락 (Pessimistic Lock)

**동작 원리**:
1. 데이터를 읽는 시점에 데이터베이스 레벨에서 락을 획득
2. 트랜잭션이 완료될 때까지 다른 트랜잭션은 해당 데이터를 수정할 수 없음
3. 락을 기다리는 트랜잭션은 대기 상태가 됨

**JPA 구현 예제**:

```java
@Repository
public interface ProductRepository extends JpaRepository<Product, Long> {

    // Pessimistic Write Lock: 쓰기 락
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT p FROM Product p WHERE p.id = :id")
    Optional<Product> findByIdWithPessimisticLock(@Param("id") Long id);

    // Pessimistic Read Lock: 읽기 락
    @Lock(LockModeType.PESSIMISTIC_READ)
    @Query("SELECT p FROM Product p WHERE p.id = :id")
    Optional<Product> findByIdWithPessimisticReadLock(@Param("id") Long id);
}

@Service
public class ProductService {

    @Autowired
    private ProductRepository productRepository;

    @Transactional
    public void decreaseStock(Long productId, int quantity) {
        // SELECT ... FOR UPDATE 쿼리 실행
        // 다른 트랜잭션은 이 row에 대한 UPDATE를 기다려야 함
        Product product = productRepository
            .findByIdWithPessimisticLock(productId)
            .orElseThrow(() -> new ProductNotFoundException());

        if (product.getStock() < quantity) {
            throw new InsufficientStockException();
        }

        product.setStock(product.getStock() - quantity);
        productRepository.save(product);

        // 트랜잭션 커밋 시 자동으로 락 해제
    }
}
```

**락 타입 비교**:

| Lock Type | SQL | 설명 | 사용 시점 |
|-----------|-----|------|-----------|
| **PESSIMISTIC_WRITE** | `SELECT ... FOR UPDATE` | 배타적 락 (Exclusive Lock) | 데이터를 수정할 예정 |
| **PESSIMISTIC_READ** | `SELECT ... FOR SHARE` | 공유 락 (Shared Lock) | 읽기만 하되 수정 방지 |
| **PESSIMISTIC_FORCE_INCREMENT** | `SELECT ... FOR UPDATE` + version 증가 | 쓰기 락 + 버전 증가 | 비관적+낙관적 혼합 |

**타임아웃 설정**:

```java
@Service
public class ProductService {

    @Autowired
    private EntityManager entityManager;

    @Transactional
    public void decreaseStockWithTimeout(Long productId, int quantity) {
        Map<String, Object> properties = new HashMap<>();
        properties.put("javax.persistence.lock.timeout", 3000); // 3초

        Product product = entityManager.find(
            Product.class,
            productId,
            LockModeType.PESSIMISTIC_WRITE,
            properties
        );

        if (product == null) {
            throw new ProductNotFoundException();
        }

        if (product.getStock() < quantity) {
            throw new InsufficientStockException();
        }

        product.setStock(product.getStock() - quantity);
    }
}
```

**SKIP LOCKED 활용** (작업 큐 패턴):

```java
@Repository
public interface OrderRepository extends JpaRepository<Order, Long> {

    // 락을 획득할 수 없는 row는 건너뛰고 다음 row 선택
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @QueryHints(@QueryHint(name = "javax.persistence.lock.timeout", value = "0"))
    @Query(value = """
        SELECT o FROM Order o
        WHERE o.status = 'PENDING'
        ORDER BY o.createdAt ASC
        """, nativeQuery = false)
    List<Order> findPendingOrdersWithSkipLocked(Pageable pageable);
}

@Service
public class OrderProcessorService {

    @Autowired
    private OrderRepository orderRepository;

    @Scheduled(fixedDelay = 1000)
    @Transactional
    public void processOrders() {
        // 여러 워커가 동시에 실행되어도 각자 다른 주문 처리
        List<Order> orders = orderRepository
            .findPendingOrdersWithSkipLocked(PageRequest.of(0, 10));

        for (Order order : orders) {
            processOrder(order);
        }
    }

    private void processOrder(Order order) {
        // 주문 처리 로직
        order.setStatus(OrderStatus.PROCESSING);
        // ...
    }
}
```

### 실전 시나리오

**1. 전자상거래 재고 관리**

```java
// 낙관적 락 사용 (조회가 많고 구매는 적음)
@Service
public class EcommerceStockService {

    @Autowired
    private ProductRepository productRepository;

    @Retryable(
        value = OptimisticLockException.class,
        maxAttempts = 5,
        backoff = @Backoff(delay = 50, multiplier = 2)
    )
    @Transactional
    public void purchaseProduct(Long productId, int quantity) {
        Product product = productRepository.findById(productId)
            .orElseThrow(() -> new ProductNotFoundException());

        if (product.getStock() < quantity) {
            throw new OutOfStockException();
        }

        product.decreaseStock(quantity);
        productRepository.save(product);

        // 낙관적 락 충돌 시 @Retryable이 자동으로 재시도
    }
}
```

**2. 콘서트 티켓 예매 (좌석 선택)**

```java
// 비관적 락 사용 (동시 예매 충돌 가능성 높음)
@Service
public class TicketReservationService {

    @Autowired
    private SeatRepository seatRepository;

    @Transactional
    public Reservation reserveSeat(Long seatId, Long userId) {
        // SELECT ... FOR UPDATE로 좌석 락 획득
        Seat seat = seatRepository.findByIdWithLock(seatId)
            .orElseThrow(() -> new SeatNotFoundException());

        if (!seat.isAvailable()) {
            throw new SeatAlreadyReservedException();
        }

        seat.setAvailable(false);
        seat.setReservedBy(userId);
        seat.setReservedAt(LocalDateTime.now());

        seatRepository.save(seat);

        return new Reservation(seat, userId);
    }
}
```

**3. 은행 계좌 이체**

```java
// 비관적 락 사용 (정확한 잔액 보장 필수)
@Service
public class BankTransferService {

    @Autowired
    private AccountRepository accountRepository;

    @Transactional
    public void transfer(Long fromAccountId, Long toAccountId, BigDecimal amount) {
        // 계좌 번호 순서대로 락 획득 (데드락 방지)
        List<Long> sortedIds = Stream.of(fromAccountId, toAccountId)
            .sorted()
            .collect(Collectors.toList());

        Account firstAccount = accountRepository
            .findByIdWithLock(sortedIds.get(0))
            .orElseThrow();

        Account secondAccount = accountRepository
            .findByIdWithLock(sortedIds.get(1))
            .orElseThrow();

        // 출금 계좌와 입금 계좌 식별
        Account fromAccount = firstAccount.getId().equals(fromAccountId)
            ? firstAccount : secondAccount;
        Account toAccount = firstAccount.getId().equals(toAccountId)
            ? firstAccount : secondAccount;

        // 잔액 확인 및 이체
        if (fromAccount.getBalance().compareTo(amount) < 0) {
            throw new InsufficientBalanceException();
        }

        fromAccount.setBalance(fromAccount.getBalance().subtract(amount));
        toAccount.setBalance(toAccount.getBalance().add(amount));

        accountRepository.save(fromAccount);
        accountRepository.save(toAccount);
    }
}
```

### 선택 가이드

**낙관적 락을 선택하는 경우**:
- 읽기 작업이 쓰기 작업보다 훨씬 많음 (10:1 이상)
- 동시 수정 충돌이 드물게 발생
- 빠른 응답 속도가 중요
- 재시도 로직 구현 가능
- 예: 게시판, 블로그, CMS, 상품 조회

**비관적 락을 선택하는 경우**:
- 동시 수정 충돌이 자주 발생
- 데이터 정확성이 매우 중요 (금융, 재고)
- 충돌 시 재시도 비용이 높음
- 트랜잭션이 짧고 빠름
- 예: 좌석 예약, 계좌 이체, 재고 차감, 한정 수량 판매

**성능 비교**:

| 시나리오 | 동시 사용자 | 충돌률 | 낙관적 락 성능 | 비관적 락 성능 | 권장 |
|---------|-----------|-------|--------------|--------------|------|
| 블로그 조회/수정 | 1000 | 0.1% | 우수 | 보통 | 낙관적 락 |
| 한정판 구매 | 10000 | 80% | 나쁨 (재시도 폭증) | 우수 | 비관적 락 |
| 일반 쇼핑몰 | 5000 | 5% | 우수 | 보통 | 낙관적 락 |
| 좌석 예매 | 3000 | 40% | 보통 | 우수 | 비관적 락 |

## 분산 트랜잭션 (Distributed Transaction)

> * [distributed transaction | TIL](/distributedtransaction/README.md)
> * [SAGAS - Cornell University](https://www.cs.cornell.edu/andru/cs711/2002fa/reading/sagas.pdf)

### 개념

분산 트랜잭션은 여러 데이터베이스나 서비스에 걸쳐 실행되는 트랜잭션으로, 모든 참여자가 성공하거나 모두 실패하는 원자성(Atomicity)을 보장해야 합니다. 마이크로서비스 아키텍처에서 가장 어려운 문제 중 하나입니다.

**문제 상황**:
```
주문 서비스    → 주문 생성 ✓
결제 서비스    → 결제 처리 ✓
재고 서비스    → 재고 차감 ✗ (실패!)

모든 작업을 롤백해야 함
```

### ACID vs BASE

| 속성 | ACID (전통적 DB) | BASE (분산 시스템) |
|------|------------------|-------------------|
| **A** | Atomicity (원자성) | Basically Available (기본적 가용성) |
| **C** | Consistency (일관성) | Soft state (유연한 상태) |
| **I** | Isolation (격리성) | Eventual consistency (최종 일관성) |
| **D** | Durability (영속성) | - |

### 분산 트랜잭션 패턴

**1. Two-Phase Commit (2PC)**

가장 전통적인 방법이지만 성능과 가용성 문제가 있습니다.

**동작 방식**:

```
Phase 1 (Prepare):
Coordinator → Participant 1: "커밋 준비 가능?"
Coordinator → Participant 2: "커밋 준비 가능?"
Coordinator ← Participant 1: "준비 완료"
Coordinator ← Participant 2: "준비 완료"

Phase 2 (Commit):
Coordinator → Participant 1: "커밋 실행"
Coordinator → Participant 2: "커밋 실행"
Coordinator ← Participant 1: "커밋 완료"
Coordinator ← Participant 2: "커밋 완료"
```

**Java/Spring 구현 (JTA)**:

```java
@Configuration
public class DistributedTransactionConfig {

    @Bean
    public JtaTransactionManager transactionManager() {
        AtomikosJtaTransactionManager tm = new AtomikosJtaTransactionManager();
        tm.setTransactionTimeout(300);
        return tm;
    }

    @Bean
    public DataSource orderDataSource() {
        AtomikosDataSourceBean ds = new AtomikosDataSourceBean();
        ds.setUniqueResourceName("orderDB");
        ds.setXaDataSourceClassName("com.mysql.cj.jdbc.MysqlXADataSource");
        ds.setXaProperties(getProperties("jdbc:mysql://localhost:3306/orders"));
        return ds;
    }

    @Bean
    public DataSource paymentDataSource() {
        AtomikosDataSourceBean ds = new AtomikosDataSourceBean();
        ds.setUniqueResourceName("paymentDB");
        ds.setXaDataSourceClassName("com.mysql.cj.jdbc.MysqlXADataSource");
        ds.setXaProperties(getProperties("jdbc:mysql://localhost:3307/payments"));
        return ds;
    }
}

@Service
public class OrderService {

    @Autowired
    @Qualifier("orderDataSource")
    private DataSource orderDataSource;

    @Autowired
    @Qualifier("paymentDataSource")
    private DataSource paymentDataSource;

    @Transactional(transactionManager = "transactionManager")
    public void createOrder(Order order, Payment payment) {
        // Phase 1: Prepare
        // 1. 주문 생성 (orderDB)
        try (Connection conn = orderDataSource.getConnection()) {
            PreparedStatement stmt = conn.prepareStatement(
                "INSERT INTO orders (id, user_id, amount) VALUES (?, ?, ?)");
            stmt.setLong(1, order.getId());
            stmt.setLong(2, order.getUserId());
            stmt.setBigDecimal(3, order.getAmount());
            stmt.executeUpdate();
        }

        // 2. 결제 처리 (paymentDB)
        try (Connection conn = paymentDataSource.getConnection()) {
            PreparedStatement stmt = conn.prepareStatement(
                "INSERT INTO payments (id, order_id, amount, status) VALUES (?, ?, ?, ?)");
            stmt.setLong(1, payment.getId());
            stmt.setLong(2, order.getId());
            stmt.setBigDecimal(3, payment.getAmount());
            stmt.setString(4, "COMPLETED");
            stmt.executeUpdate();
        }

        // Phase 2: Commit
        // JTA Transaction Manager가 자동으로 모든 참여자에게 커밋 명령
        // 하나라도 실패하면 모두 롤백
    }
}
```

**2PC 문제점**:
- Coordinator가 단일 장애점(SPOF)
- 블로킹 프로토콜 (성능 저하)
- 참여자가 많을수록 실패 확률 증가
- 네트워크 파티션 시 복구 어려움

**2. SAGA Pattern**

장기 실행 트랜잭션(Long Lived Transaction)을 여러 개의 작은 로컬 트랜잭션으로 분할하고, 각각에 대한 보상 트랜잭션(Compensating Transaction)을 정의합니다.

**두 가지 SAGA 방식**:

**A. Choreography (이벤트 기반)**

각 서비스가 이벤트를 발행하고 구독하여 자율적으로 동작합니다.

```java
// 이벤트 정의
@Data
class OrderCreatedEvent {
    private Long orderId;
    private Long userId;
    private BigDecimal amount;
    private LocalDateTime timestamp;
}

@Data
class PaymentCompletedEvent {
    private Long paymentId;
    private Long orderId;
    private LocalDateTime timestamp;
}

@Data
class PaymentFailedEvent {
    private Long orderId;
    private String reason;
    private LocalDateTime timestamp;
}

// 주문 서비스
@Service
public class OrderService {

    @Autowired
    private ApplicationEventPublisher eventPublisher;

    @Autowired
    private OrderRepository orderRepository;

    // 1. 주문 생성
    @Transactional
    public Order createOrder(CreateOrderRequest request) {
        Order order = new Order();
        order.setUserId(request.getUserId());
        order.setAmount(request.getAmount());
        order.setStatus(OrderStatus.PENDING);

        order = orderRepository.save(order);

        // 이벤트 발행
        eventPublisher.publishEvent(new OrderCreatedEvent(
            order.getId(),
            order.getUserId(),
            order.getAmount(),
            LocalDateTime.now()
        ));

        return order;
    }

    // 보상 트랜잭션: 결제 실패 시 주문 취소
    @EventListener
    @Transactional
    public void onPaymentFailed(PaymentFailedEvent event) {
        Order order = orderRepository.findById(event.getOrderId())
            .orElseThrow();

        order.setStatus(OrderStatus.CANCELLED);
        order.setCancelReason(event.getReason());
        orderRepository.save(order);

        log.info("주문 {} 취소됨: {}", order.getId(), event.getReason());
    }

    // 재고 차감 실패 시 주문 취소
    @EventListener
    @Transactional
    public void onInventoryDeductionFailed(InventoryDeductionFailedEvent event) {
        Order order = orderRepository.findById(event.getOrderId())
            .orElseThrow();

        order.setStatus(OrderStatus.CANCELLED);
        orderRepository.save(order);
    }
}

// 결제 서비스
@Service
public class PaymentService {

    @Autowired
    private ApplicationEventPublisher eventPublisher;

    @Autowired
    private PaymentRepository paymentRepository;

    // 주문 생성 이벤트 수신
    @EventListener
    @Transactional
    public void onOrderCreated(OrderCreatedEvent event) {
        try {
            // 2. 결제 처리
            Payment payment = new Payment();
            payment.setOrderId(event.getOrderId());
            payment.setAmount(event.getAmount());
            payment.setStatus(PaymentStatus.PROCESSING);

            // 외부 결제 게이트웨이 호출
            PaymentResult result = paymentGateway.charge(
                event.getUserId(),
                event.getAmount()
            );

            if (result.isSuccess()) {
                payment.setStatus(PaymentStatus.COMPLETED);
                paymentRepository.save(payment);

                // 성공 이벤트 발행
                eventPublisher.publishEvent(new PaymentCompletedEvent(
                    payment.getId(),
                    event.getOrderId(),
                    LocalDateTime.now()
                ));

            } else {
                // 실패 이벤트 발행
                eventPublisher.publishEvent(new PaymentFailedEvent(
                    event.getOrderId(),
                    result.getErrorMessage(),
                    LocalDateTime.now()
                ));
            }

        } catch (Exception e) {
            log.error("결제 처리 실패", e);
            eventPublisher.publishEvent(new PaymentFailedEvent(
                event.getOrderId(),
                e.getMessage(),
                LocalDateTime.now()
            ));
        }
    }
}

// 재고 서비스
@Service
public class InventoryService {

    @Autowired
    private ApplicationEventPublisher eventPublisher;

    @Autowired
    private InventoryRepository inventoryRepository;

    // 결제 완료 이벤트 수신
    @EventListener
    @Transactional
    public void onPaymentCompleted(PaymentCompletedEvent event) {
        try {
            // 3. 재고 차감
            Inventory inventory = inventoryRepository.findByOrderId(event.getOrderId())
                .orElseThrow();

            if (inventory.getQuantity() < 1) {
                // 재고 부족 - 실패 이벤트 발행
                eventPublisher.publishEvent(new InventoryDeductionFailedEvent(
                    event.getOrderId(),
                    "재고 부족",
                    LocalDateTime.now()
                ));
                return;
            }

            inventory.setQuantity(inventory.getQuantity() - 1);
            inventoryRepository.save(inventory);

            // 성공 이벤트 발행
            eventPublisher.publishEvent(new InventoryDeductedEvent(
                event.getOrderId(),
                LocalDateTime.now()
            ));

        } catch (Exception e) {
            log.error("재고 차감 실패", e);
            eventPublisher.publishEvent(new InventoryDeductionFailedEvent(
                event.getOrderId(),
                e.getMessage(),
                LocalDateTime.now()
            ));
        }
    }

    // 보상 트랜잭션: 재고 복구
    @EventListener
    @Transactional
    public void onPaymentFailed(PaymentFailedEvent event) {
        // 이미 차감된 재고가 있다면 복구
        Inventory inventory = inventoryRepository.findByOrderId(event.getOrderId())
            .orElse(null);

        if (inventory != null) {
            inventory.setQuantity(inventory.getQuantity() + 1);
            inventoryRepository.save(inventory);
            log.info("재고 복구: 주문 {}", event.getOrderId());
        }
    }
}
```

**B. Orchestration (중앙 조정자)**

중앙 조정자(Orchestrator)가 모든 트랜잭션을 제어합니다.

```java
// SAGA 상태 정의
public enum SagaStatus {
    STARTED,
    ORDER_CREATED,
    PAYMENT_COMPLETED,
    INVENTORY_DEDUCTED,
    COMPLETED,
    FAILED,
    COMPENSATING,
    COMPENSATED
}

// SAGA 인스턴스
@Entity
@Table(name = "saga_instances")
public class SagaInstance {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String sagaType; // "ORDER_SAGA"
    private Long businessKey; // orderId

    @Enumerated(EnumType.STRING)
    private SagaStatus status;

    private String currentStep;
    private LocalDateTime startedAt;
    private LocalDateTime completedAt;

    @Column(columnDefinition = "JSON")
    private String payload; // JSON으로 저장
}

// SAGA Orchestrator
@Service
public class OrderSagaOrchestrator {

    @Autowired
    private SagaInstanceRepository sagaRepository;

    @Autowired
    private OrderService orderService;

    @Autowired
    private PaymentService paymentService;

    @Autowired
    private InventoryService inventoryService;

    // SAGA 시작
    @Transactional
    public SagaInstance startOrderSaga(CreateOrderRequest request) {
        // 1. SAGA 인스턴스 생성
        SagaInstance saga = new SagaInstance();
        saga.setSagaType("ORDER_SAGA");
        saga.setStatus(SagaStatus.STARTED);
        saga.setStartedAt(LocalDateTime.now());
        saga.setPayload(toJson(request));

        saga = sagaRepository.save(saga);

        // 2. 비동기로 SAGA 실행
        CompletableFuture.runAsync(() -> executeSaga(saga.getId()));

        return saga;
    }

    // SAGA 실행
    public void executeSaga(Long sagaId) {
        SagaInstance saga = sagaRepository.findById(sagaId).orElseThrow();
        CreateOrderRequest request = fromJson(saga.getPayload(), CreateOrderRequest.class);

        try {
            // Step 1: 주문 생성
            saga.setCurrentStep("ORDER_CREATION");
            saga.setStatus(SagaStatus.ORDER_CREATED);
            sagaRepository.save(saga);

            Order order = orderService.createOrder(request);
            saga.setBusinessKey(order.getId());
            sagaRepository.save(saga);

            // Step 2: 결제 처리
            saga.setCurrentStep("PAYMENT_PROCESSING");
            sagaRepository.save(saga);

            PaymentResult paymentResult = paymentService.processPayment(
                order.getId(),
                order.getAmount()
            );

            if (!paymentResult.isSuccess()) {
                throw new PaymentException(paymentResult.getErrorMessage());
            }

            saga.setStatus(SagaStatus.PAYMENT_COMPLETED);
            sagaRepository.save(saga);

            // Step 3: 재고 차감
            saga.setCurrentStep("INVENTORY_DEDUCTION");
            sagaRepository.save(saga);

            boolean inventoryDeducted = inventoryService.deductInventory(
                order.getId(),
                request.getProductId(),
                request.getQuantity()
            );

            if (!inventoryDeducted) {
                throw new InventoryException("재고 부족");
            }

            // 성공
            saga.setStatus(SagaStatus.COMPLETED);
            saga.setCompletedAt(LocalDateTime.now());
            sagaRepository.save(saga);

            log.info("SAGA {} 완료", sagaId);

        } catch (Exception e) {
            log.error("SAGA {} 실패: {}", sagaId, e.getMessage());
            compensate(saga, e);
        }
    }

    // 보상 트랜잭션 실행
    private void compensate(SagaInstance saga, Exception error) {
        saga.setStatus(SagaStatus.COMPENSATING);
        sagaRepository.save(saga);

        try {
            String currentStep = saga.getCurrentStep();

            // 역순으로 보상 실행
            if ("INVENTORY_DEDUCTION".equals(currentStep) ||
                "PAYMENT_PROCESSING".equals(currentStep)) {
                // 재고 복구
                inventoryService.restoreInventory(saga.getBusinessKey());
            }

            if ("PAYMENT_PROCESSING".equals(currentStep)) {
                // 결제 취소
                paymentService.refund(saga.getBusinessKey());
            }

            // 주문 취소
            orderService.cancelOrder(saga.getBusinessKey(), error.getMessage());

            saga.setStatus(SagaStatus.COMPENSATED);
            saga.setCompletedAt(LocalDateTime.now());
            sagaRepository.save(saga);

            log.info("SAGA {} 보상 완료", saga.getId());

        } catch (Exception e) {
            log.error("SAGA {} 보상 실패", saga.getId(), e);
            saga.setStatus(SagaStatus.FAILED);
            sagaRepository.save(saga);

            // 수동 개입 필요 알림
            alertService.sendAlert("SAGA 보상 실패", saga.getId());
        }
    }
}
```

**3. Outbox Pattern**

메시지 발행과 데이터베이스 업데이트를 원자적으로 처리합니다.

```java
// Outbox 테이블
@Entity
@Table(name = "outbox")
public class OutboxEvent {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String aggregateType; // "Order"
    private Long aggregateId;
    private String eventType; // "OrderCreated"

    @Column(columnDefinition = "JSON")
    private String payload;

    private LocalDateTime createdAt;
    private LocalDateTime processedAt;
    private boolean processed;
}

@Service
public class OrderServiceWithOutbox {

    @Autowired
    private OrderRepository orderRepository;

    @Autowired
    private OutboxRepository outboxRepository;

    // 주문 생성 + 이벤트 저장 (단일 트랜잭션)
    @Transactional
    public Order createOrder(CreateOrderRequest request) {
        // 1. 주문 저장
        Order order = new Order();
        order.setUserId(request.getUserId());
        order.setAmount(request.getAmount());
        order.setStatus(OrderStatus.PENDING);

        order = orderRepository.save(order);

        // 2. Outbox에 이벤트 저장 (같은 트랜잭션)
        OutboxEvent event = new OutboxEvent();
        event.setAggregateType("Order");
        event.setAggregateId(order.getId());
        event.setEventType("OrderCreated");
        event.setPayload(toJson(new OrderCreatedEvent(order)));
        event.setCreatedAt(LocalDateTime.now());
        event.setProcessed(false);

        outboxRepository.save(event);

        // 트랜잭션 커밋 시 주문과 이벤트가 함께 저장됨
        return order;
    }
}

// Outbox Polling Publisher
@Service
public class OutboxPublisher {

    @Autowired
    private OutboxRepository outboxRepository;

    @Autowired
    private KafkaTemplate<String, String> kafkaTemplate;

    @Scheduled(fixedDelay = 1000) // 1초마다
    @Transactional
    public void publishEvents() {
        // 미처리 이벤트 조회
        List<OutboxEvent> events = outboxRepository
            .findTop100ByProcessedFalseOrderByCreatedAtAsc();

        for (OutboxEvent event : events) {
            try {
                // Kafka로 발행
                kafkaTemplate.send(
                    event.getEventType(),
                    event.getAggregateId().toString(),
                    event.getPayload()
                ).get(); // 동기 대기

                // 처리 완료 표시
                event.setProcessed(true);
                event.setProcessedAt(LocalDateTime.now());
                outboxRepository.save(event);

            } catch (Exception e) {
                log.error("이벤트 발행 실패: {}", event.getId(), e);
                // 다음 폴링 시 재시도
            }
        }
    }
}
```

### 패턴 비교

| 패턴 | 장점 | 단점 | 사용 시기 |
|------|------|------|----------|
| **2PC** | 강한 일관성 보장 | 성능 저하, SPOF, 복잡도 높음 | 금융, 결제 등 강한 일관성 필수 |
| **SAGA (Choreography)** | 느슨한 결합, 확장성 | 복잡한 플로우 추적 어려움 | 서비스 간 의존성 낮은 경우 |
| **SAGA (Orchestration)** | 중앙 집중 제어, 추적 용이 | Orchestrator가 SPOF | 복잡한 비즈니스 플로우 |
| **Outbox** | At-least-once 보장 | 폴링 오버헤드 | 이벤트 손실 방지 필수 |

### 보상 트랜잭션 설계 원칙

1. **멱등성**: 보상 트랜잭션은 여러 번 실행되어도 안전해야 함
2. **순서**: 정방향의 역순으로 실행
3. **타임아웃**: 적절한 타임아웃 설정
4. **재시도**: 실패 시 자동 재시도 로직
5. **수동 개입**: 최종 실패 시 알림 및 수동 처리

### 분산 트랜잭션 모니터링

```java
@RestController
@RequestMapping("/admin/sagas")
public class SagaMonitoringController {

    @Autowired
    private SagaInstanceRepository sagaRepository;

    // 진행 중인 SAGA 목록
    @GetMapping("/active")
    public List<SagaInstance> getActiveSagas() {
        return sagaRepository.findByStatusIn(
            Arrays.asList(
                SagaStatus.STARTED,
                SagaStatus.ORDER_CREATED,
                SagaStatus.PAYMENT_COMPLETED,
                SagaStatus.COMPENSATING
            )
        );
    }

    // 실패한 SAGA 목록
    @GetMapping("/failed")
    public List<SagaInstance> getFailedSagas() {
        return sagaRepository.findByStatus(SagaStatus.FAILED);
    }

    // SAGA 수동 재시도
    @PostMapping("/{sagaId}/retry")
    public ResponseEntity<Void> retrySaga(@PathVariable Long sagaId) {
        SagaInstance saga = sagaRepository.findById(sagaId).orElseThrow();

        if (saga.getStatus() != SagaStatus.FAILED) {
            return ResponseEntity.badRequest().build();
        }

        // 상태 초기화 후 재시도
        saga.setStatus(SagaStatus.STARTED);
        sagaRepository.save(saga);

        orchestrator.executeSaga(sagaId);

        return ResponseEntity.ok().build();
    }
}
```

### 설계 체크리스트

**SAGA 설계 시**:
- [ ] 각 단계별 보상 트랜잭션 정의
- [ ] 보상 트랜잭션의 멱등성 보장
- [ ] 타임아웃 및 재시도 정책 수립
- [ ] SAGA 인스턴스 상태 추적 및 모니터링
- [ ] 실패 시 알림 및 수동 개입 프로세스

**성능 고려사항**:
- [ ] 불필요한 동기 호출 최소화
- [ ] 비동기 이벤트 처리 활용
- [ ] 데이터베이스 인덱스 최적화
- [ ] 이벤트 중복 제거 메커니즘

## 멱등성 (Idempotency)

### 개념

멱등성(Idempotency)은 동일한 연산을 여러 번 수행해도 한 번 수행한 것과 같은 결과를 생성하는 속성입니다. RESTful API에서 같은 요청을 여러 번 호출해도 동일한 결과를 반환하는 것을 의미합니다.

**핵심 개념**:
- 연산을 여러 번 반복해도 한 번 실행하는 것과 같은 결과 생성
- 네트워크 장애나 타임아웃으로 인한 재시도 시 안전성 보장
- 분산 시스템에서 중복 요청 처리 문제 해결
- 안전한(safe) 메서드는 멱등성이 있지만, 모든 멱등한 메서드가 안전한 것은 아님

### HTTP 메서드 멱등성

| HTTP 메서드 | 멱등성 | 안전성 | 설명 |
|------------|-------|-------|------|
| **GET** | O | O | 리소스 조회, 상태 변경 없음 |
| **HEAD** | O | O | 메타데이터만 조회 |
| **OPTIONS** | O | O | 지원하는 메서드 확인 |
| **TRACE** | O | O | 요청 경로 확인 |
| **PUT** | O | X | 리소스 전체 교체, 같은 결과 |
| **DELETE** | O | X | 리소스 삭제, 여러 번 호출해도 삭제됨 |
| **POST** | X | X | 리소스 생성, 호출마다 새 리소스 |
| **PATCH** | X | X | 리소스 부분 수정 |

### 멱등성 키 (Idempotency Key) 패턴

클라이언트가 고유한 키를 요청에 포함시켜 서버가 중복 요청을 식별하고 처리하는 패턴입니다.

**구현 예제 - Spring Boot + Redis**:

```java
@RestController
@RequestMapping("/api/payments")
public class PaymentController {

    @Autowired
    private PaymentService paymentService;

    @Autowired
    private RedisTemplate<String, String> redisTemplate;

    private static final String IDEMPOTENCY_KEY_PREFIX = "idempotency:";
    private static final long IDEMPOTENCY_KEY_TTL = 24 * 60 * 60; // 24시간

    @PostMapping
    public ResponseEntity<PaymentResponse> processPayment(
            @RequestHeader("Idempotency-Key") String idempotencyKey,
            @RequestBody PaymentRequest request) {

        String redisKey = IDEMPOTENCY_KEY_PREFIX + idempotencyKey;

        // 1. 이미 처리된 요청인지 확인
        String cachedResult = redisTemplate.opsForValue().get(redisKey);
        if (cachedResult != null) {
            PaymentResponse response = parseResponse(cachedResult);
            return ResponseEntity.ok(response);
        }

        // 2. 처리 중인 요청인지 확인 (동시 요청 방지)
        Boolean acquired = redisTemplate.opsForValue()
            .setIfAbsent(redisKey + ":lock", "processing",
                        Duration.ofSeconds(30));

        if (!acquired) {
            return ResponseEntity.status(HttpStatus.CONFLICT)
                .body(new PaymentResponse("요청 처리 중입니다"));
        }

        try {
            // 3. 결제 처리
            PaymentResponse response = paymentService.processPayment(request);

            // 4. 결과 캐싱
            redisTemplate.opsForValue().set(
                redisKey,
                serializeResponse(response),
                Duration.ofSeconds(IDEMPOTENCY_KEY_TTL)
            );

            return ResponseEntity.ok(response);

        } finally {
            // 5. 처리 중 락 해제
            redisTemplate.delete(redisKey + ":lock");
        }
    }
}

@Service
public class PaymentService {

    @Autowired
    private PaymentRepository paymentRepository;

    @Transactional
    public PaymentResponse processPayment(PaymentRequest request) {
        // 데이터베이스 레벨에서도 중복 방지
        Optional<Payment> existing = paymentRepository
            .findByTransactionId(request.getTransactionId());

        if (existing.isPresent()) {
            return PaymentResponse.from(existing.get());
        }

        Payment payment = new Payment();
        payment.setTransactionId(request.getTransactionId());
        payment.setAmount(request.getAmount());
        payment.setStatus(PaymentStatus.COMPLETED);
        payment.setCreatedAt(LocalDateTime.now());

        payment = paymentRepository.save(payment);

        return PaymentResponse.from(payment);
    }
}
```

### 멱등성 보장 전략

**1. 고유 식별자 기반 (Unique Identifier)**

```java
@Entity
@Table(name = "orders",
       uniqueConstraints = @UniqueConstraint(columnNames = "idempotency_key"))
public class Order {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "idempotency_key", nullable = false, unique = true)
    private String idempotencyKey;

    @Column(nullable = false)
    private String userId;

    @Column(nullable = false)
    private BigDecimal totalAmount;

    @Enumerated(EnumType.STRING)
    private OrderStatus status;
}

@Service
public class OrderService {

    @Autowired
    private OrderRepository orderRepository;

    @Transactional
    public Order createOrder(CreateOrderRequest request, String idempotencyKey) {
        try {
            // 중복 키로 인한 예외 처리
            Order order = new Order();
            order.setIdempotencyKey(idempotencyKey);
            order.setUserId(request.getUserId());
            order.setTotalAmount(request.getTotalAmount());
            order.setStatus(OrderStatus.PENDING);

            return orderRepository.save(order);

        } catch (DataIntegrityViolationException e) {
            // 이미 존재하는 주문 반환
            return orderRepository.findByIdempotencyKey(idempotencyKey)
                .orElseThrow(() -> new RuntimeException("주문을 찾을 수 없습니다"));
        }
    }
}
```

**2. 버전 기반 낙관적 락 (Optimistic Locking)**

```java
@Entity
public class Account {
    @Id
    private Long id;

    private BigDecimal balance;

    @Version
    private Long version;
}

@Service
public class AccountService {

    @Autowired
    private AccountRepository accountRepository;

    @Transactional
    public void withdraw(Long accountId, BigDecimal amount) {
        Account account = accountRepository.findById(accountId)
            .orElseThrow(() -> new RuntimeException("계좌를 찾을 수 없습니다"));

        if (account.getBalance().compareTo(amount) < 0) {
            throw new InsufficientBalanceException();
        }

        account.setBalance(account.getBalance().subtract(amount));

        try {
            accountRepository.save(account);
        } catch (OptimisticLockException e) {
            // 재시도 로직
            throw new RetryableException("동시 수정이 감지되었습니다");
        }
    }
}
```

**3. 상태 전이 기반 (State Machine)**

```java
public enum PaymentStatus {
    PENDING,
    PROCESSING,
    COMPLETED,
    FAILED,
    REFUNDED
}

@Service
public class PaymentStateMachine {

    @Transactional
    public Payment updatePaymentStatus(Long paymentId, PaymentStatus newStatus) {
        Payment payment = paymentRepository.findById(paymentId)
            .orElseThrow(() -> new RuntimeException("결제를 찾을 수 없습니다"));

        // 상태 전이 검증
        if (!isValidTransition(payment.getStatus(), newStatus)) {
            throw new InvalidStateTransitionException(
                String.format("%s에서 %s로 전이할 수 없습니다",
                             payment.getStatus(), newStatus)
            );
        }

        payment.setStatus(newStatus);
        payment.setUpdatedAt(LocalDateTime.now());

        return paymentRepository.save(payment);
    }

    private boolean isValidTransition(PaymentStatus current, PaymentStatus next) {
        switch (current) {
            case PENDING:
                return next == PaymentStatus.PROCESSING || next == PaymentStatus.FAILED;
            case PROCESSING:
                return next == PaymentStatus.COMPLETED || next == PaymentStatus.FAILED;
            case COMPLETED:
                return next == PaymentStatus.REFUNDED;
            case FAILED:
            case REFUNDED:
                return false; // 종료 상태
            default:
                return false;
        }
    }
}
```

### 실전 시나리오

**1. 결제 API - 네트워크 타임아웃 대응**

```java
@RestController
public class PaymentApiController {

    @PostMapping("/api/v1/payments")
    public ResponseEntity<PaymentResponse> createPayment(
            @RequestHeader("X-Idempotency-Key") String idempotencyKey,
            @RequestBody PaymentRequest request) {

        // UUID 검증
        try {
            UUID.fromString(idempotencyKey);
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest()
                .body(new PaymentResponse("유효하지 않은 Idempotency-Key"));
        }

        // 멱등성 보장 처리
        PaymentResponse response = paymentService
            .processPaymentIdempotent(idempotencyKey, request);

        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }
}

// 클라이언트 재시도 로직
public class PaymentClient {

    private static final int MAX_RETRIES = 3;
    private static final long RETRY_DELAY_MS = 1000;

    public PaymentResponse createPayment(PaymentRequest request) {
        String idempotencyKey = UUID.randomUUID().toString();

        for (int attempt = 1; attempt <= MAX_RETRIES; attempt++) {
            try {
                return httpClient.post("/api/v1/payments")
                    .header("X-Idempotency-Key", idempotencyKey)
                    .body(request)
                    .execute(PaymentResponse.class);

            } catch (TimeoutException e) {
                if (attempt == MAX_RETRIES) {
                    throw new PaymentException("결제 처리 실패: 타임아웃", e);
                }

                try {
                    Thread.sleep(RETRY_DELAY_MS * attempt);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    throw new PaymentException("재시도 중단", ie);
                }
            }
        }

        throw new PaymentException("결제 처리 실패");
    }
}
```

**2. 이메일 발송 - 중복 발송 방지**

```java
@Service
public class EmailService {

    @Autowired
    private RedisTemplate<String, String> redisTemplate;

    @Autowired
    private EmailSender emailSender;

    public void sendOrderConfirmation(String orderId, String email) {
        String idempotencyKey = "email:order:" + orderId;

        // 이미 발송했는지 확인 (7일 동안 기록 유지)
        Boolean sent = redisTemplate.opsForValue()
            .setIfAbsent(idempotencyKey, "sent", Duration.ofDays(7));

        if (sent == null || !sent) {
            // 이미 발송됨
            log.info("주문 {} 확인 이메일이 이미 발송되었습니다", orderId);
            return;
        }

        try {
            emailSender.send(email, "주문 확인",
                           generateOrderConfirmationBody(orderId));
            log.info("주문 {} 확인 이메일 발송 완료", orderId);

        } catch (Exception e) {
            // 실패 시 키 삭제 (재시도 가능하도록)
            redisTemplate.delete(idempotencyKey);
            throw new EmailSendException("이메일 발송 실패", e);
        }
    }
}
```

### 멱등성 구현 시 고려사항

| 고려사항 | 설명 | 권장사항 |
|---------|------|---------|
| **키 생성** | 클라이언트가 고유 키 생성 | UUID v4 사용 권장 |
| **키 저장 기간** | 중복 감지를 위한 보관 기간 | 24시간 ~ 7일 (비즈니스 요구사항에 따름) |
| **동시 요청** | 같은 키로 동시 요청 발생 | 분산 락 또는 낙관적 락 사용 |
| **응답 캐싱** | 이전 응답 재사용 | 완전히 같은 응답 반환 |
| **부분 실패** | 일부만 성공한 경우 | 트랜잭션 경계 명확히 정의 |
| **모니터링** | 중복 요청 빈도 추적 | 메트릭 수집 및 알림 |

**모니터링 예제**:

```java
@Aspect
@Component
public class IdempotencyMonitoringAspect {

    @Autowired
    private MeterRegistry meterRegistry;

    @Around("@annotation(IdempotentOperation)")
    public Object monitorIdempotency(ProceedingJoinPoint joinPoint)
            throws Throwable {

        String operationType = joinPoint.getSignature().getName();
        Timer.Sample sample = Timer.start(meterRegistry);

        try {
            Object result = joinPoint.proceed();

            sample.stop(Timer.builder("idempotent.operation.duration")
                .tag("operation", operationType)
                .tag("status", "success")
                .register(meterRegistry));

            meterRegistry.counter("idempotent.operation.count",
                "operation", operationType,
                "result", "new")
                .increment();

            return result;

        } catch (DuplicateRequestException e) {
            meterRegistry.counter("idempotent.operation.count",
                "operation", operationType,
                "result", "duplicate")
                .increment();
            throw e;
        }
    }
}
```

## 메시지 큐 (Message Queue)

### 전송 보장 방식

**1. At-most-once (최대 한 번)**
- 메시지 손실 가능
- 중복 없음
- 가장 빠른 성능

```python
# Kafka Producer 예제
producer.send('topic', message, acks=0)  # 응답 대기 안 함
```

**2. At-least-once (최소 한 번)**
- 메시지 손실 없음
- 중복 가능
- 멱등성 처리 필요

```python
# Kafka Producer
producer.send('topic', message, acks=1)  # 리더 확인 대기

# Consumer - 멱등성 보장
class IdempotentConsumer:
    def __init__(self):
        self.processed_ids = set()

    def process(self, message):
        msg_id = message['id']

        # 이미 처리한 메시지 스킵
        if msg_id in self.processed_ids:
            return

        # 메시지 처리
        handle_message(message)

        # 처리 완료 기록
        self.processed_ids.add(msg_id)
```

**3. Exactly-once (정확히 한 번)**
- 메시지 손실 없음
- 중복 없음
- 가장 느린 성능

```python
# Kafka Transactional Producer
producer = KafkaProducer(
    transactional_id='my-transactional-id',
    enable_idempotence=True
)

producer.init_transactions()
try:
    producer.begin_transaction()
    producer.send('topic', message)
    producer.commit_transaction()
except Exception:
    producer.abort_transaction()
```

### Dead Letter Queue

```python
class MessageProcessor:
    def __init__(self):
        self.main_queue = kafka.Consumer('main-topic')
        self.dlq = kafka.Producer('dlq-topic')
        self.max_retries = 3

    def process_message(self, message):
        retry_count = message.get('retry_count', 0)

        try:
            # 메시지 처리 로직
            handle_message(message)
        except Exception as e:
            if retry_count < self.max_retries:
                # 재시도
                message['retry_count'] = retry_count + 1
                message['last_error'] = str(e)
                self.main_queue.send(message, delay=60)  # 1분 후 재시도
            else:
                # DLQ로 이동
                self.dlq.send({
                    'original_message': message,
                    'error': str(e),
                    'retry_count': retry_count,
                    'timestamp': time.time()
                })
```

## Observability

- [Monitoring](essentials/Monitoring.md)

## Load Test

- [Load Test](/loadtest/README.md)

monitoring, logging, tracing, alerting, auditing 등을 말한다.

## Incidenct

- [Incident](/incident/README.md)

## 재해 복구 (Disaster Recovery, DR)

### 개념

재해 복구(Disaster Recovery)는 자연재해, 사이버 공격, 정전, 하드웨어 장애와 같은 예상치 못한 중단 사건 발생 후 조직의 중요한 운영과 서비스를 신속하게 재개할 수 있도록 하는 체계적인 접근 방식입니다. 데이터 백업, 중복성, 장애 조치 메커니즘, 정기적으로 테스트되고 업데이트되는 포괄적인 복구 계획을 통해 다운타임, 데이터 손실, 재정적 영향을 최소화하고 비즈니스 연속성과 고객 신뢰를 유지하는 것이 주요 목표입니다.

### 핵심 지표

**RTO (Recovery Time Objective - 목표 복구 시간)**
- 재해 발생 후 시스템을 복구하는 데 걸리는 최대 허용 시간
- 비즈니스가 감당할 수 있는 최대 다운타임

**RPO (Recovery Point Objective - 목표 복구 시점)**
- 재해 발생 시 손실될 수 있는 최대 데이터의 시간 범위
- 마지막 백업 시점부터 재해 발생 시점까지의 시간

**RLO (Recovery Level Objective - 목표 복구 수준)**
- 복구 후 제공해야 하는 최소 서비스 수준
- 완전 복구 또는 부분 복구 수준 정의

```
┌─────────────────────────────────────────────────────────┐
│  정상 운영                                               │
│  ─────────────────────────────────────────────────────  │
│                      ↓ 재해 발생                         │
│                      │                                   │
│                      │←─── RPO (데이터 손실 범위) ──→│  │
│  마지막 백업 ────────┼───────────────────────────── 재해 │
│                      │                                   │
│                      │←────── RTO (복구 시간) ─────→│   │
│                      │                              복구  │
│  ─────────────────────────────────────────────────────  │
│  정상 운영 재개                                          │
└─────────────────────────────────────────────────────────┘
```

### DR 전략 비교

| 전략 | RTO | RPO | 비용 | 복잡도 | 설명 |
|------|-----|-----|------|--------|------|
| **Backup & Restore** | 시간 ~ 일 | 시간 | 낮음 | 낮음 | 정기적 백업, 재해 시 복원 |
| **Pilot Light** | 10분 ~ 시간 | 분 | 보통 | 보통 | 핵심 시스템만 대기 상태 유지 |
| **Warm Standby** | 분 ~ 10분 | 초 ~ 분 | 높음 | 높음 | 축소 버전 시스템 항상 가동 |
| **Hot Standby (Active-Active)** | 초 ~ 분 | 거의 0 | 매우 높음 | 매우 높음 | 동일한 시스템 동시 운영 |

### DR 전략 구현

**1. Backup & Restore (백업 및 복원)**

가장 기본적이고 비용 효율적인 방법입니다.

```java
@Service
public class BackupService {

    @Autowired
    private DatabaseBackupRepository backupRepository;

    @Autowired
    private S3Client s3Client;

    // 일일 전체 백업 + 시간별 증분 백업
    @Scheduled(cron = "0 0 2 * * *") // 매일 새벽 2시
    public void performFullBackup() {
        String timestamp = LocalDateTime.now()
            .format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));

        String backupFileName = String.format("full_backup_%s.sql", timestamp);

        try {
            // 1. 데이터베이스 전체 백업
            File backupFile = createDatabaseDump(backupFileName);

            // 2. 압축
            File compressedFile = compressFile(backupFile);

            // 3. 암호화
            File encryptedFile = encryptFile(compressedFile);

            // 4. S3에 업로드 (다중 리전에 복제)
            uploadToS3(encryptedFile, "backups/full/" + backupFileName + ".enc");

            // 5. 백업 메타데이터 저장
            BackupMetadata metadata = new BackupMetadata();
            metadata.setBackupType(BackupType.FULL);
            metadata.setFileName(backupFileName);
            metadata.setSize(encryptedFile.length());
            metadata.setCreatedAt(LocalDateTime.now());
            metadata.setStatus(BackupStatus.COMPLETED);

            backupRepository.save(metadata);

            // 6. 이전 백업 정리 (30일 이상 보관)
            cleanupOldBackups(30);

            log.info("전체 백업 완료: {}", backupFileName);

        } catch (Exception e) {
            log.error("백업 실패", e);
            sendAlertToOps("백업 실패: " + e.getMessage());
        }
    }

    @Scheduled(cron = "0 0 * * * *") // 매시간
    public void performIncrementalBackup() {
        String timestamp = LocalDateTime.now()
            .format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));

        String backupFileName = String.format("incremental_backup_%s.sql", timestamp);

        try {
            // 마지막 백업 이후 변경된 데이터만 백업
            LocalDateTime lastBackupTime = getLastBackupTime();
            File incrementalBackup = createIncrementalDump(lastBackupTime, backupFileName);

            File compressed = compressFile(incrementalBackup);
            File encrypted = encryptFile(compressed);

            uploadToS3(encrypted, "backups/incremental/" + backupFileName + ".enc");

            BackupMetadata metadata = new BackupMetadata();
            metadata.setBackupType(BackupType.INCREMENTAL);
            metadata.setFileName(backupFileName);
            metadata.setSize(encrypted.length());
            metadata.setCreatedAt(LocalDateTime.now());
            metadata.setStatus(BackupStatus.COMPLETED);

            backupRepository.save(metadata);

        } catch (Exception e) {
            log.error("증분 백업 실패", e);
        }
    }

    // 복원 프로세스
    public void restoreFromBackup(String backupFileName) {
        try {
            // 1. S3에서 백업 파일 다운로드
            File encryptedFile = downloadFromS3("backups/full/" + backupFileName);

            // 2. 복호화
            File compressedFile = decryptFile(encryptedFile);

            // 3. 압축 해제
            File backupFile = decompressFile(compressedFile);

            // 4. 데이터베이스 복원
            restoreDatabase(backupFile);

            log.info("데이터베이스 복원 완료: {}", backupFileName);

        } catch (Exception e) {
            log.error("복원 실패", e);
            throw new DisasterRecoveryException("복원 실패: " + e.getMessage());
        }
    }
}
```

**2. Pilot Light (파일럿 라이트)**

핵심 시스템만 최소 구성으로 대기 상태를 유지합니다.

```java
@Service
public class PilotLightDRService {

    @Autowired
    private EC2Client ec2Client;

    @Autowired
    private RDSClient rdsClient;

    @Autowired
    private Route53Client route53Client;

    // 재해 발생 시 DR 사이트 활성화
    public void activateDRSite() {
        try {
            log.info("DR 사이트 활성화 시작...");

            // 1. RDS 읽기 전용 복제본을 쓰기 가능한 마스터로 승격
            promoteDRDatabase();

            // 2. 대기 중인 EC2 인스턴스 타입 확장
            scaleUpApplicationServers();

            // 3. Auto Scaling 그룹 활성화
            enableAutoScaling();

            // 4. 로드 밸런서 상태 확인 대상 추가
            addTargetsToLoadBalancer();

            // 5. DNS 장애 조치 (Route 53)
            failoverDNS();

            // 6. CloudFront 오리진 변경
            updateCDNOrigin();

            // 7. 애플리케이션 상태 확인
            verifyApplicationHealth();

            log.info("DR 사이트 활성화 완료 (RTO: 약 10분)");

        } catch (Exception e) {
            log.error("DR 사이트 활성화 실패", e);
            throw new DisasterRecoveryException("DR 활성화 실패", e);
        }
    }

    private void promoteDRDatabase() {
        // RDS 읽기 복제본을 독립적인 DB 인스턴스로 승격
        PromoteReadReplicaRequest request = PromoteReadReplicaRequest.builder()
            .dbInstanceIdentifier("dr-database-replica")
            .backupRetentionPeriod(7)
            .preferredBackupWindow("03:00-04:00")
            .build();

        rdsClient.promoteReadReplica(request);

        // 승격 완료 대기
        waitForDatabaseAvailable("dr-database-replica");
    }

    private void scaleUpApplicationServers() {
        // t3.micro -> t3.large로 확장
        ModifyInstanceAttributeRequest request = ModifyInstanceAttributeRequest.builder()
            .instanceId("i-dr-app-server")
            .instanceType(AttributeValue.builder()
                .value(InstanceType.T3_LARGE.toString())
                .build())
            .build();

        ec2Client.modifyInstanceAttribute(request);
    }

    private void failoverDNS() {
        // Route 53에서 주 리전에서 DR 리전으로 트래픽 전환
        ChangeResourceRecordSetsRequest request = ChangeResourceRecordSetsRequest.builder()
            .hostedZoneId("Z1234567890ABC")
            .changeBatch(ChangeBatch.builder()
                .changes(Change.builder()
                    .action(ChangeAction.UPSERT)
                    .resourceRecordSet(ResourceRecordSet.builder()
                        .name("api.example.com")
                        .type(RRType.A)
                        .setIdentifier("DR-Region")
                        .weight(100L) // DR 리전으로 모든 트래픽 전환
                        .ttl(60L)
                        .resourceRecords(ResourceRecord.builder()
                            .value("203.0.113.1") // DR 로드 밸런서 IP
                            .build())
                        .build())
                    .build())
                .build())
            .build();

        route53Client.changeResourceRecordSets(request);
    }
}
```

**3. Warm Standby (웜 스탠바이)**

축소 버전의 완전한 시스템을 항상 실행 상태로 유지합니다.

```java
@Service
public class WarmStandbyDRService {

    @Autowired
    private AutoScalingClient autoScalingClient;

    @Autowired
    private DatabaseReplicationMonitor replicationMonitor;

    // 데이터베이스 실시간 복제 모니터링
    @Scheduled(fixedDelay = 30000) // 30초마다
    public void monitorReplicationLag() {
        Long replicationLagSeconds = replicationMonitor.getReplicationLag();

        if (replicationLagSeconds > 60) { // 1분 이상 지연
            log.warn("복제 지연 감지: {} 초", replicationLagSeconds);
            sendAlert("DB 복제 지연", replicationLagSeconds);
        }

        // 복제 상태 메트릭 저장
        metricsService.recordReplicationLag(replicationLagSeconds);
    }

    // 재해 발생 시 DR 사이트로 빠른 전환
    public void failoverToDR() {
        try {
            log.info("Warm Standby DR 장애 조치 시작...");

            // 1. Auto Scaling 그룹 용량 증가 (축소 -> 전체 크기)
            scaleUpToFullCapacity();

            // 2. 데이터베이스 복제 확인 및 승격
            verifyReplicationAndPromote();

            // 3. 캐시 웜업
            warmupCache();

            // 4. 트래픽 전환
            switchTraffic();

            log.info("DR 장애 조치 완료 (RTO: 약 5분)");

        } catch (Exception e) {
            log.error("DR 장애 조치 실패", e);
            throw new DisasterRecoveryException("장애 조치 실패", e);
        }
    }

    private void scaleUpToFullCapacity() {
        // 현재: 2대 -> 목표: 10대
        UpdateAutoScalingGroupRequest request = UpdateAutoScalingGroupRequest.builder()
            .autoScalingGroupName("dr-app-asg")
            .minSize(10)
            .maxSize(20)
            .desiredCapacity(10)
            .build();

        autoScalingClient.updateAutoScalingGroup(request);

        // 모든 인스턴스가 healthy 상태가 될 때까지 대기
        waitForHealthyInstances("dr-app-asg", 10);
    }

    private void warmupCache() {
        // 주요 데이터 미리 로드
        List<String> hotKeys = getHotKeys(); // 자주 액세스되는 키 목록

        hotKeys.parallelStream().forEach(key -> {
            try {
                cacheService.preload(key);
            } catch (Exception e) {
                log.warn("캐시 웜업 실패: {}", key, e);
            }
        });
    }
}
```

**4. Hot Standby / Active-Active (액티브-액티브)**

두 개 이상의 리전에서 동시에 운영합니다.

```java
@Service
public class ActiveActiveDRService {

    @Autowired
    private DynamoDBClient dynamoDBClient;

    @Autowired
    private LoadBalancerClient lbClient;

    // 글로벌 테이블 설정 (자동 양방향 복제)
    public void setupGlobalTable() {
        CreateGlobalTableRequest request = CreateGlobalTableRequest.builder()
            .globalTableName("users")
            .replicationGroup(
                Replica.builder().regionName("us-east-1").build(),
                Replica.builder().regionName("eu-west-1").build(),
                Replica.builder().regionName("ap-northeast-2").build()
            )
            .build();

        dynamoDBClient.createGlobalTable(request);
    }

    // 리전 간 헬스 체크 및 자동 장애 조치
    @Scheduled(fixedDelay = 10000) // 10초마다
    public void monitorRegionalHealth() {
        Map<String, HealthStatus> regionHealth = new HashMap<>();

        for (String region : Arrays.asList("us-east-1", "eu-west-1", "ap-northeast-2")) {
            HealthStatus status = checkRegionHealth(region);
            regionHealth.put(region, status);

            if (status == HealthStatus.UNHEALTHY) {
                log.error("리전 {} 비정상 감지", region);
                // 자동으로 트래픽을 다른 리전으로 재분배
                redistributeTraffic(region);
            }
        }
    }

    private void redistributeTraffic(String unhealthyRegion) {
        // 비정상 리전의 가중치를 0으로 설정
        List<String> healthyRegions = getHealthyRegions();

        int weightPerRegion = 100 / healthyRegions.size();

        for (String region : getAllRegions()) {
            int weight = healthyRegions.contains(region) ? weightPerRegion : 0;

            updateRoute53Weight(region, weight);
        }

        log.info("트래픽 재분배 완료. 비정상 리전: {}", unhealthyRegion);
    }

    // 충돌 해결 전략 (Last Write Wins)
    public void resolveConflict(String key, List<VersionedValue> conflictingVersions) {
        // 타임스탬프 기반 최신 값 선택
        VersionedValue latest = conflictingVersions.stream()
            .max(Comparator.comparing(VersionedValue::getTimestamp))
            .orElseThrow();

        // 모든 리전에 최신 값 전파
        propagateToAllRegions(key, latest);
    }
}
```

### DR 테스트 및 검증

```java
@Service
public class DRTestService {

    @Autowired
    private DRActivationService drActivationService;

    @Autowired
    private HealthCheckService healthCheckService;

    // 분기별 DR 훈련
    @Scheduled(cron = "0 0 9 1 */3 *") // 매 분기 첫날 오전 9시
    public void conductDRDrill() {
        log.info("DR 훈련 시작...");

        DRTestReport report = new DRTestReport();
        report.setStartTime(LocalDateTime.now());

        try {
            // 1. DR 사이트 활성화
            long activationStart = System.currentTimeMillis();
            drActivationService.activateDRSite();
            long activationTime = System.currentTimeMillis() - activationStart;

            report.setActualRTO(activationTime / 1000); // 초 단위

            // 2. 애플리케이션 기능 테스트
            List<String> testResults = runFunctionalTests();
            report.setTestResults(testResults);

            // 3. 데이터 일관성 확인
            DataConsistencyCheck consistencyCheck = verifyDataConsistency();
            report.setDataConsistencyCheck(consistencyCheck);

            // 4. 성능 테스트
            PerformanceMetrics metrics = runPerformanceTests();
            report.setPerformanceMetrics(metrics);

            // 5. 주 사이트로 복귀
            drActivationService.failbackToPrimary();

            report.setEndTime(LocalDateTime.now());
            report.setStatus(TestStatus.SUCCESS);

            // 보고서 생성 및 전송
            generateAndSendReport(report);

        } catch (Exception e) {
            report.setStatus(TestStatus.FAILED);
            report.setErrorMessage(e.getMessage());
            log.error("DR 훈련 실패", e);
        }
    }

    private List<String> runFunctionalTests() {
        List<String> results = new ArrayList<>();

        // 핵심 기능 테스트
        results.add(testUserAuthentication() ? "✓ 사용자 인증" : "✗ 사용자 인증");
        results.add(testOrderPlacement() ? "✓ 주문 생성" : "✗ 주문 생성");
        results.add(testPaymentProcessing() ? "✓ 결제 처리" : "✗ 결제 처리");
        results.add(testDataRetrieval() ? "✓ 데이터 조회" : "✗ 데이터 조회");

        return results;
    }
}
```

### DR 체크리스트

**계획 단계**:
- [ ] RTO/RPO 요구사항 정의
- [ ] DR 전략 선택 (비용 vs 복구 시간)
- [ ] 중요 시스템 및 데이터 식별
- [ ] 백업 정책 수립 (빈도, 보관 기간, 위치)
- [ ] DR 사이트 선정 (지리적 분리)

**구현 단계**:
- [ ] 자동화된 백업 시스템 구축
- [ ] 데이터 복제 설정 (실시간 or 주기적)
- [ ] 장애 조치 메커니즘 구현
- [ ] 모니터링 및 알림 시스템 구축
- [ ] DR 런북(Runbook) 작성

**테스트 단계**:
- [ ] 분기별 DR 훈련 실시
- [ ] 백업 복원 테스트
- [ ] 장애 조치 시간 측정 (RTO 검증)
- [ ] 데이터 손실 범위 확인 (RPO 검증)
- [ ] 문제점 식별 및 개선

**유지보수 단계**:
- [ ] DR 계획 정기 검토 및 업데이트
- [ ] 백업 무결성 정기 검증
- [ ] DR 문서 최신화
- [ ] 팀원 교육 및 역할 분담

### 비용 최적화 전략

| 항목 | 비용 절감 방법 |
|------|---------------|
| **백업 스토리지** | S3 Glacier로 장기 보관, 수명 주기 정책 적용 |
| **DR 인스턴스** | 예약 인스턴스 또는 Savings Plan 활용 |
| **데이터 전송** | VPC 피어링, Direct Connect로 전송 비용 절감 |
| **데이터베이스** | 읽기 복제본 활용, Multi-AZ 대신 Cross-Region 복제 |
| **스냅샷** | 증분 스냅샷 활용, 불필요한 스냅샷 자동 삭제 |

### 주요 클라우드 DR 서비스

**AWS**:
- AWS Backup: 중앙 집중식 백업 관리
- Amazon S3 Cross-Region Replication: 자동 복제
- AWS CloudEndure Disaster Recovery: 연속 복제 기반 DR
- Route 53 Health Checks: DNS 장애 조치

**Azure**:
- Azure Site Recovery: 자동화된 DR 오케스트레이션
- Azure Backup: 통합 백업 솔루션
- Geo-redundant Storage: 지역 간 자동 복제

**GCP**:
- Cloud Storage Transfer Service: 데이터 전송 및 백업
- Persistent Disk Snapshots: 증분 스냅샷
- Global Load Balancing: 멀티 리전 트래픽 분산

## 구성 관리 데이터베이스 (Configuration Management Database, CMDB)

CMDB(Configuration Management Database)는 IT 환경 내의 하드웨어 및 소프트웨어 구성 요소(Configuration Items, CIs)와 이들 간의 관계 정보를 저장하는 중앙 저장소다. CMDB는 IT 서비스 관리(ITSM) 프로세스에서 핵심적인 역할을 수행하며, 인시던트 관리, 문제 관리, 변경 관리, 자산 관리 등에 활용된다.

**CMDB의 목적**:

CMDB는 모든 IT 구성 요소에 대한 정확하고 최신의 인벤토리를 유지하여 인프라에 대한 가시성과 이해도를 향상시킨다. CI(Configuration Item) 간의 상호 의존성을 문서화함으로써 다양한 IT 프로세스와 의사 결정을 효율화한다.

**CMDB의 핵심 가치**:

**1. IT 리소스 최적화**:
- 리소스 활용도를 추적하고 분석하여 활용도가 낮거나 높은 구성 요소 식별
- 리소스를 보다 효과적으로 할당하고 재배치
- 비용 절감 및 성능 최적화

**2. 인시던트 및 문제 관리 강화**:
- 영향을 받는 CI 간의 관계와 의존성을 시각화하여 문제를 빠르게 식별
- 장애 발생 시 영향 범위를 신속하게 파악
- 평균 복구 시간(MTTR) 단축

**3. 변경 관리 리스크 완화**:
- 상호 연결된 시스템에 대한 변경의 잠재적 영향 이해
- 변경 구현 전 정보에 기반한 의사 결정 가능
- 변경으로 인한 장애 예방

**4. 효과적인 자산 관리 지원**:
- IT 자산에 대한 최신의 중앙 집중식 저장소 유지
- 라이선스, 구매, 유지보수, 수명 종료(EOL) 계획 지원
- 자산 수명 주기 관리 최적화

**5. 규정 준수 관리 지원**:
- IT 인프라에 대한 포괄적인 가시성 확보
- 규제 요구사항 및 조직 정책 준수 보장
- 감사 및 컴플라이언스 보고 간소화

**CMDB 데이터 모델**:

```
┌─────────────────────────────────────────────────────────┐
│                    CMDB Core Model                      │
└─────────────────────────────────────────────────────────┘

Configuration Items (CIs):
├── Infrastructure CIs
│   ├── Servers (Physical, Virtual)
│   ├── Network Devices (Routers, Switches, Firewalls)
│   ├── Storage Systems
│   └── Data Centers
│
├── Application CIs
│   ├── Applications
│   ├── Microservices
│   ├── Databases
│   └── Middleware
│
├── Service CIs
│   ├── Business Services
│   ├── Technical Services
│   └── APIs
│
└── Supporting CIs
    ├── Documentation
    ├── Licenses
    ├── Contracts
    └── Personnel

Relationships:
- Depends On: CI A depends on CI B
- Hosts: Physical server hosts virtual machine
- Communicates With: Application A calls API B
- Owned By: CI is owned by department/person
- Part Of: Component is part of larger system
```

**CMDB 구현 예제 (Java)**:

```java
// ConfigurationItem.java
import java.time.LocalDateTime;
import java.util.*;

public abstract class ConfigurationItem {
    private String id;
    private String name;
    private String type;
    private String status; // Active, Retired, Planned, Under Maintenance
    private String owner;
    private LocalDateTime createdDate;
    private LocalDateTime lastModified;
    private Map<String, String> attributes;
    private List<Relationship> relationships;

    public ConfigurationItem(String id, String name, String type) {
        this.id = id;
        this.name = name;
        this.type = type;
        this.status = "Active";
        this.createdDate = LocalDateTime.now();
        this.lastModified = LocalDateTime.now();
        this.attributes = new HashMap<>();
        this.relationships = new ArrayList<>();
    }

    public void addRelationship(Relationship relationship) {
        this.relationships.add(relationship);
        this.lastModified = LocalDateTime.now();
    }

    public List<ConfigurationItem> getDependencies() {
        return relationships.stream()
            .filter(r -> r.getType().equals("DEPENDS_ON"))
            .map(Relationship::getTarget)
            .toList();
    }

    // Getters and setters
    public String getId() { return id; }
    public String getName() { return name; }
    public String getType() { return type; }
    public String getStatus() { return status; }
    public List<Relationship> getRelationships() { return relationships; }
    public void setStatus(String status) {
        this.status = status;
        this.lastModified = LocalDateTime.now();
    }
}

// Relationship.java
class Relationship {
    private String type; // DEPENDS_ON, HOSTS, COMMUNICATES_WITH, OWNED_BY, PART_OF
    private ConfigurationItem source;
    private ConfigurationItem target;
    private Map<String, String> properties;

    public Relationship(String type, ConfigurationItem source, ConfigurationItem target) {
        this.type = type;
        this.source = source;
        this.target = target;
        this.properties = new HashMap<>();
    }

    public String getType() { return type; }
    public ConfigurationItem getTarget() { return target; }
}

// ServerCI.java
class ServerCI extends ConfigurationItem {
    private String ipAddress;
    private String osType;
    private int cpu;
    private int memory;
    private String location;

    public ServerCI(String id, String name, String ipAddress, String osType) {
        super(id, name, "Server");
        this.ipAddress = ipAddress;
        this.osType = osType;
    }

    public String getIpAddress() { return ipAddress; }
    public String getOsType() { return osType; }
}

// ApplicationCI.java
class ApplicationCI extends ConfigurationItem {
    private String version;
    private String environment; // Production, Staging, Development
    private List<String> dependencies;

    public ApplicationCI(String id, String name, String version) {
        super(id, name, "Application");
        this.version = version;
        this.dependencies = new ArrayList<>();
    }

    public String getVersion() { return version; }
    public String getEnvironment() { return environment; }
}

// DatabaseCI.java
class DatabaseCI extends ConfigurationItem {
    private String dbType; // MySQL, PostgreSQL, MongoDB, etc.
    private String version;
    private int port;
    private long storageSize;

    public DatabaseCI(String id, String name, String dbType) {
        super(id, name, "Database");
        this.dbType = dbType;
    }

    public String getDbType() { return dbType; }
}

// CMDBService.java
import java.util.*;
import java.util.stream.Collectors;

public class CMDBService {
    private Map<String, ConfigurationItem> configurationItems;

    public CMDBService() {
        this.configurationItems = new HashMap<>();
    }

    // CI 추가
    public void addCI(ConfigurationItem ci) {
        configurationItems.put(ci.getId(), ci);
        System.out.println("Added CI: " + ci.getName() + " (ID: " + ci.getId() + ")");
    }

    // CI 조회
    public ConfigurationItem getCI(String id) {
        return configurationItems.get(id);
    }

    // 관계 생성
    public void createRelationship(String sourceId, String targetId, String relationshipType) {
        ConfigurationItem source = configurationItems.get(sourceId);
        ConfigurationItem target = configurationItems.get(targetId);

        if (source != null && target != null) {
            Relationship relationship = new Relationship(relationshipType, source, target);
            source.addRelationship(relationship);
            System.out.println("Created relationship: " + source.getName() + " " +
                             relationshipType + " " + target.getName());
        }
    }

    // 영향 분석: 특정 CI가 장애 시 영향받는 CI 목록
    public List<ConfigurationItem> getImpactAnalysis(String ciId) {
        Set<ConfigurationItem> impactedCIs = new HashSet<>();
        ConfigurationItem startCI = configurationItems.get(ciId);

        if (startCI == null) {
            return Collections.emptyList();
        }

        // BFS로 의존 그래프 탐색
        Queue<ConfigurationItem> queue = new LinkedList<>();
        queue.offer(startCI);
        Set<String> visited = new HashSet<>();
        visited.add(ciId);

        while (!queue.isEmpty()) {
            ConfigurationItem current = queue.poll();

            // 현재 CI에 의존하는 모든 CI 찾기
            for (ConfigurationItem ci : configurationItems.values()) {
                if (!visited.contains(ci.getId())) {
                    List<ConfigurationItem> dependencies = ci.getDependencies();
                    if (dependencies.contains(current)) {
                        impactedCIs.add(ci);
                        queue.offer(ci);
                        visited.add(ci.getId());
                    }
                }
            }
        }

        return new ArrayList<>(impactedCIs);
    }

    // 특정 타입의 CI 조회
    public List<ConfigurationItem> getCIsByType(String type) {
        return configurationItems.values().stream()
            .filter(ci -> ci.getType().equals(type))
            .collect(Collectors.toList());
    }

    // 특정 상태의 CI 조회
    public List<ConfigurationItem> getCIsByStatus(String status) {
        return configurationItems.values().stream()
            .filter(ci -> ci.getStatus().equals(status))
            .collect(Collectors.toList());
    }

    // CI 의존성 트리 출력
    public void printDependencyTree(String ciId, int depth) {
        ConfigurationItem ci = configurationItems.get(ciId);
        if (ci == null) return;

        String indent = "  ".repeat(depth);
        System.out.println(indent + "- " + ci.getName() + " (" + ci.getType() + ")");

        List<ConfigurationItem> dependencies = ci.getDependencies();
        for (ConfigurationItem dep : dependencies) {
            printDependencyTree(dep.getId(), depth + 1);
        }
    }

    // 변경 영향 평가
    public ChangeImpactReport assessChangeImpact(String ciId) {
        ConfigurationItem ci = configurationItems.get(ciId);
        if (ci == null) {
            return null;
        }

        List<ConfigurationItem> impactedCIs = getImpactAnalysis(ciId);
        List<ConfigurationItem> dependencies = ci.getDependencies();

        return new ChangeImpactReport(ci, impactedCIs, dependencies);
    }
}

// ChangeImpactReport.java
class ChangeImpactReport {
    private ConfigurationItem targetCI;
    private List<ConfigurationItem> impactedCIs;
    private List<ConfigurationItem> dependencies;
    private String riskLevel;

    public ChangeImpactReport(ConfigurationItem targetCI,
                             List<ConfigurationItem> impactedCIs,
                             List<ConfigurationItem> dependencies) {
        this.targetCI = targetCI;
        this.impactedCIs = impactedCIs;
        this.dependencies = dependencies;
        this.riskLevel = calculateRiskLevel();
    }

    private String calculateRiskLevel() {
        int totalImpact = impactedCIs.size();
        if (totalImpact == 0) return "LOW";
        if (totalImpact <= 5) return "MEDIUM";
        if (totalImpact <= 10) return "HIGH";
        return "CRITICAL";
    }

    public void printReport() {
        System.out.println("\n=== Change Impact Report ===");
        System.out.println("Target CI: " + targetCI.getName());
        System.out.println("Risk Level: " + riskLevel);
        System.out.println("\nDependencies (" + dependencies.size() + "):");
        dependencies.forEach(ci -> System.out.println("  - " + ci.getName()));
        System.out.println("\nImpacted CIs (" + impactedCIs.size() + "):");
        impactedCIs.forEach(ci -> System.out.println("  - " + ci.getName()));
    }
}

// CMDBExample.java
public class CMDBExample {
    public static void main(String[] args) {
        CMDBService cmdb = new CMDBService();

        // 서버 CI 생성
        ServerCI webServer = new ServerCI("SRV001", "Web-Server-01", "10.0.1.10", "Ubuntu 22.04");
        ServerCI appServer = new ServerCI("SRV002", "App-Server-01", "10.0.1.20", "CentOS 8");
        ServerCI dbServer = new ServerCI("SRV003", "DB-Server-01", "10.0.1.30", "RedHat 8");

        // 애플리케이션 CI 생성
        ApplicationCI webApp = new ApplicationCI("APP001", "E-Commerce-Web", "2.5.0");
        ApplicationCI apiService = new ApplicationCI("APP002", "Order-API-Service", "1.3.2");

        // 데이터베이스 CI 생성
        DatabaseCI orderDB = new DatabaseCI("DB001", "Order-Database", "PostgreSQL");

        // CMDB에 CI 추가
        cmdb.addCI(webServer);
        cmdb.addCI(appServer);
        cmdb.addCI(dbServer);
        cmdb.addCI(webApp);
        cmdb.addCI(apiService);
        cmdb.addCI(orderDB);

        // 관계 설정
        System.out.println("\n=== Creating Relationships ===");
        cmdb.createRelationship("APP001", "SRV001", "HOSTS"); // Web App hosted on Web Server
        cmdb.createRelationship("APP002", "SRV002", "HOSTS"); // API Service hosted on App Server
        cmdb.createRelationship("DB001", "SRV003", "HOSTS");  // Database hosted on DB Server
        cmdb.createRelationship("APP001", "APP002", "DEPENDS_ON"); // Web App depends on API
        cmdb.createRelationship("APP002", "DB001", "DEPENDS_ON");  // API depends on Database

        // 의존성 트리 출력
        System.out.println("\n=== Dependency Tree for Web Application ===");
        cmdb.printDependencyTree("APP001", 0);

        // 영향 분석: Database가 장애 시 영향받는 CI
        System.out.println("\n=== Impact Analysis: If Database fails ===");
        List<ConfigurationItem> impacted = cmdb.getImpactAnalysis("DB001");
        impacted.forEach(ci -> System.out.println("  - " + ci.getName() + " will be impacted"));

        // 변경 영향 평가
        System.out.println("\n=== Change Impact Assessment ===");
        ChangeImpactReport report = cmdb.assessChangeImpact("DB001");
        if (report != null) {
            report.printReport();
        }

        // 특정 타입의 CI 조회
        System.out.println("\n=== All Server CIs ===");
        List<ConfigurationItem> servers = cmdb.getCIsByType("Server");
        servers.forEach(ci -> System.out.println("  - " + ci.getName()));
    }
}
```

**CMDB 통합 및 자동화**:

```java
// CMDBAutoDiscoveryService.java
import java.util.*;

public class CMDBAutoDiscoveryService {
    private CMDBService cmdb;

    public CMDBAutoDiscoveryService(CMDBService cmdb) {
        this.cmdb = cmdb;
    }

    // 네트워크 스캔을 통한 자동 검색
    public void discoverInfrastructure() {
        System.out.println("Starting infrastructure auto-discovery...");

        // 네트워크 스캔 (실제로는 nmap, Ansible 등 사용)
        List<ServerCI> discoveredServers = scanNetwork();

        for (ServerCI server : discoveredServers) {
            cmdb.addCI(server);
            // 서버에서 실행 중인 애플리케이션 검색
            discoverApplicationsOnServer(server);
        }

        System.out.println("Auto-discovery completed.");
    }

    private List<ServerCI> scanNetwork() {
        // 실제 구현에서는 네트워크 스캔 도구 사용
        List<ServerCI> servers = new ArrayList<>();
        servers.add(new ServerCI("SRV100", "Discovered-Server-1", "10.0.2.10", "Ubuntu"));
        servers.add(new ServerCI("SRV101", "Discovered-Server-2", "10.0.2.11", "CentOS"));
        return servers;
    }

    private void discoverApplicationsOnServer(ServerCI server) {
        // 실제 구현에서는 SSH/WMI를 통해 서버의 프로세스 목록 조회
        System.out.println("Discovering applications on " + server.getName());
    }

    // 모니터링 도구와 통합
    public void syncWithMonitoring() {
        System.out.println("Syncing with monitoring tools (Prometheus, Datadog, etc.)...");
        // 모니터링 도구에서 CI 상태 업데이트
    }

    // APM 도구와 통합하여 애플리케이션 의존성 자동 발견
    public void discoverApplicationDependencies() {
        System.out.println("Discovering application dependencies from APM tools...");
        // APM (Dynatrace, New Relic 등)에서 호출 관계 파악
    }
}

// CMDBIntegrationService.java
public class CMDBIntegrationService {
    private CMDBService cmdb;

    public CMDBIntegrationService(CMDBService cmdb) {
        this.cmdb = cmdb;
    }

    // ITSM 도구와 통합 (ServiceNow, Jira Service Management 등)
    public void syncWithITSM(String incidentId) {
        System.out.println("Fetching incident " + incidentId + " from ITSM...");

        // 인시던트와 관련된 CI 조회
        String affectedCiId = "SRV001"; // ITSM에서 가져온 정보
        ConfigurationItem ci = cmdb.getCI(affectedCiId);

        if (ci != null) {
            // 영향 분석 수행
            List<ConfigurationItem> impacted = cmdb.getImpactAnalysis(affectedCiId);
            System.out.println("Incident affects " + impacted.size() + " CIs");

            // ITSM으로 정보 전송
            sendImpactToITSM(incidentId, impacted);
        }
    }

    private void sendImpactToITSM(String incidentId, List<ConfigurationItem> impacted) {
        // ITSM API를 통해 영향 범위 업데이트
        System.out.println("Updating incident " + incidentId + " with impact analysis");
    }

    // 클라우드 자산 관리 통합 (AWS Config, Azure Resource Graph 등)
    public void syncWithCloudProvider() {
        System.out.println("Syncing with cloud provider APIs...");

        // AWS 예시
        syncAWSResources();

        // Azure 예시
        syncAzureResources();
    }

    private void syncAWSResources() {
        // AWS SDK를 사용하여 EC2, RDS, ELB 등의 리소스 정보 수집
        System.out.println("Fetching AWS resources...");

        // 예시: EC2 인스턴스 정보를 CMDB에 추가
        ServerCI awsInstance = new ServerCI(
            "AWS-EC2-001",
            "prod-web-server",
            "10.0.5.10",
            "Amazon Linux 2"
        );
        cmdb.addCI(awsInstance);
    }

    private void syncAzureResources() {
        // Azure SDK를 사용하여 VM, Database 등의 리소스 정보 수집
        System.out.println("Fetching Azure resources...");
    }
}
```

**CMDB 모범 사례**:

**1. 데이터 품질 유지**:
- 정기적인 데이터 정확성 검증
- 자동화된 검색 도구 활용
- 데이터 소유자 지정 및 책임 명확화

**2. 적절한 세부 수준**:
- 과도하게 상세한 정보는 유지보수 부담 증가
- 비즈니스 가치에 따라 CI 세부 수준 결정
- 중요한 관계에 집중

**3. 프로세스 통합**:
- ITSM 프로세스와 긴밀한 통합
- 변경 관리 워크플로우에 CMDB 활용
- 인시던트 해결 시 영향 분석 자동화

**4. 자동화 및 통합**:
- 자동 검색 도구 구현
- 모니터링 및 APM 도구와 통합
- 클라우드 API를 통한 실시간 동기화
- CI/CD 파이프라인과 연계

**5. 거버넌스**:
- 명확한 데이터 소유권 정의
- 변경 승인 프로세스 수립
- 정기적인 감사 및 검증

**CMDB 구현 시 도전 과제**:

**1. 데이터 정확성 유지**:
- IT 시스템의 동적인 특성으로 인한 데이터 불일치
- 해결책: 자동화된 검색 및 지속적인 동기화

**2. 초기 구축 비용**:
- 모든 CI와 관계 정보 수집에 시간 소요
- 해결책: 점진적 구축, 중요 시스템부터 시작

**3. 조직 문화**:
- 데이터 입력 및 유지보수에 대한 저항
- 해결책: 자동화 최대화, 명확한 가치 제시

**4. 도구 통합**:
- 다양한 IT 관리 도구와의 통합 복잡성
- 해결책: 표준 API 사용, 통합 플랫폼 활용

**주요 CMDB 솔루션**:

| 솔루션 | 타입 | 주요 특징 |
|--------|------|----------|
| **ServiceNow CMDB** | 상용 | ITSM과 완전 통합, 강력한 자동 검색 |
| **BMC Helix CMDB** | 상용 | AI 기반 분석, 클라우드 최적화 |
| **Device42** | 상용 | 자동 검색 특화, 데이터센터 관리 |
| **Ralph** | 오픈소스 | 자산 관리 중심, 데이터센터 관리 |
| **iTop** | 오픈소스 | ITIL 기반, 커스터마이징 용이 |

CMDB는 복잡한 IT 환경을 관리하는 데 매우 유용한 도구지만, 구현과 유지보수에는 지속적인 노력이 필요하다. 자동화된 검색, 구성 및 모니터링 도구와의 통합을 통해 시간이 지나도 CMDB의 정확성과 효과를 유지할 수 있다.

## 보안 (Security)

### Security Overview

- [Public Key Infrastructure](/pki/README.md)
- [Cross Origin Resource Sharing](/cors/README.md)

### Web Application Firewall (WAF)

* [AWS WAF – 웹 애플리케이션 방화벽](https://aws.amazon.com/ko/waf/)
* [웹방화벽이란?](https://www.pentasecurity.co.kr/resource/%EC%9B%B9%EB%B3%B4%EC%95%88/%EC%9B%B9%EB%B0%A9%ED%99%94%EB%B2%BD%EC%9D%B4%EB%9E%80/)

WAF(Web Application Firewall)는 일반적인 방화벽과 달리 웹 애플리케이션의 보안에 특화된 솔루션입니다. 애플리케이션의 가용성에 영향을 주거나, SQL Injection, XSS(Cross Site Scripting)과 같이 보안을 위협하거나, 리소스를 과도하게 사용하는 웹 공격으로부터 웹 애플리케이션을 보호하는 데 도움이 됩니다.

**WAF의 주요 기능**

- **SQL Injection 방어**: 악의적인 SQL 쿼리 차단
- **XSS 공격 방어**: 스크립트 삽입 공격 차단
- **DDoS 공격 완화**: 과도한 트래픽으로부터 보호
- **봇 탐지**: 악성 봇과 크롤러 차단
- **API 보호**: REST/GraphQL API 엔드포인트 보호

**Java에서 WAF 규칙 구현 예제**

```java
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.regex.Pattern;

@Component
public class WAFFilter extends OncePerRequestFilter {

    // SQL Injection 패턴
    private static final Pattern SQL_INJECTION_PATTERN = Pattern.compile(
        ".*(union|select|insert|update|delete|drop|create|alter|exec|script|javascript|eval|expression).*",
        Pattern.CASE_INSENSITIVE
    );

    // XSS 패턴
    private static final Pattern XSS_PATTERN = Pattern.compile(
        ".*(<script|javascript:|onerror=|onload=|eval\\(|expression\\().*",
        Pattern.CASE_INSENSITIVE
    );

    @Override
    protected void doFilterInternal(
            HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain) throws ServletException, IOException {

        String uri = request.getRequestURI();
        String queryString = request.getQueryString();

        // SQL Injection 체크
        if (queryString != null && SQL_INJECTION_PATTERN.matcher(queryString).matches()) {
            response.setStatus(HttpServletResponse.SC_FORBIDDEN);
            response.getWriter().write("Potential SQL Injection detected");
            return;
        }

        // XSS 체크
        if (queryString != null && XSS_PATTERN.matcher(queryString).matches()) {
            response.setStatus(HttpServletResponse.SC_FORBIDDEN);
            response.getWriter().write("Potential XSS attack detected");
            return;
        }

        // 요청 파라미터 검증
        request.getParameterMap().forEach((key, values) -> {
            for (String value : values) {
                if (SQL_INJECTION_PATTERN.matcher(value).matches() ||
                    XSS_PATTERN.matcher(value).matches()) {
                    try {
                        response.setStatus(HttpServletResponse.SC_FORBIDDEN);
                        response.getWriter().write("Malicious content detected in parameter: " + key);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    return;
                }
            }
        });

        filterChain.doFilter(request, response);
    }
}
```

### Cross Site Scripting (XSS)

* [웹 해킹 강좌 ⑦ - XSS(Cross Site Scripting) 공격의 개요와 실습 (Web Hacking Tutorial #07) @ youtube](https://www.youtube.com/watch?v=DoN7bkdQBXU)

XSS(Cross Site Scripting)는 웹 게시판에 JavaScript를 내용으로 삽입해 놓으면 그 게시물을 사용자가 읽을 때 삽입된 스크립트가 실행되는 공격 방법입니다. 공격자는 악의적인 스크립트를 웹 페이지에 삽입하여 다른 사용자의 브라우저에서 실행되도록 합니다.

**XSS 공격 유형**

1. **Stored XSS (저장형)**: 악성 스크립트가 서버의 데이터베이스에 영구적으로 저장됨
2. **Reflected XSS (반사형)**: URL 파라미터에 스크립트를 포함하여 즉시 실행
3. **DOM-based XSS**: 클라이언트 측 스크립트가 DOM을 조작하여 발생

**Java에서 XSS 방어 구현**

```java
import org.springframework.web.util.HtmlUtils;
import org.owasp.encoder.Encode;

public class XSSDefense {

    // Spring의 HtmlUtils 사용
    public String sanitizeWithSpring(String input) {
        if (input == null) {
            return null;
        }
        return HtmlUtils.htmlEscape(input);
    }

    // OWASP Java Encoder 사용 (권장)
    public String sanitizeWithOWASP(String input) {
        if (input == null) {
            return null;
        }
        return Encode.forHtml(input);
    }
}
```

### Cross Site Request Forgery (CSRF)

* [웹 해킹 강좌 ⑩ - CSRF(Cross Site Request Forgery) 공격 기법 (Web Hacking Tutorial #10) @ youtube](https://www.youtube.com/watch?v=nzoUgKPwn_A)

CSRF(Cross Site Request Forgery)는 특정 사용자의 세션을 탈취하는 데에는 실패하였지만 스크립팅 공격이 통할 때 사용할 수 있는 해킹 기법입니다. 피해자가 스크립트를 보는 것과 동시에 자기도 모르게 특정한 사이트에 어떠한 요청(Request) 데이터를 보냅니다.

**Java Spring에서 CSRF 방어**

```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.csrf.CookieCsrfTokenRepository;

@Configuration
public class CSRFSecurityConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .csrf()
                .csrfTokenRepository(CookieCsrfTokenRepository.withHttpOnlyFalse())
            .and()
            .authorizeRequests()
                .antMatchers("/api/public/**").permitAll()
                .anyRequest().authenticated();

        return http.build();
    }
}
```

### XSS vs CSRF

XSS와 CSRF는 모두 웹 애플리케이션의 보안 취약점을 이용한 공격이지만, 공격 대상과 방식이 다릅니다.

**비교 테이블**

| 구분 | XSS | CSRF |
|------|-----|------|
| 공격 대상 | Client (사용자 브라우저) | Server (웹 서버) |
| 공격 방식 | 악성 스크립트 삽입 | 위조된 요청 전송 |
| 목적 | 사용자 정보 탈취, 세션 탈취 | 사용자 권한으로 서버에 요청 실행 |
| 방어 방법 | 입력 검증, 출력 인코딩, CSP | CSRF 토큰, SameSite 쿠키 |

**주요 차이점**

- **XSS**: 공격 대상이 Client이고, 사이트 변조나 백도어를 통해 Client를 공격합니다.
- **CSRF**: 공격 대상이 Server이고, 요청을 위조하여 사용자의 권한을 이용해 Server를 공격합니다.

### Cross Origin Resource Sharing (CORS)

[cors | TIL](/cors/README.md)

CORS(Cross Origin Resource Sharing)는 웹 브라우저에서 실행되는 스크립트가 다른 출처(origin)의 리소스에 접근할 수 있도록 허용하는 메커니즘입니다.

**Java Spring에서 CORS 설정**

```java
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class CorsConfig implements WebMvcConfigurer {

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/api/**")
            .allowedOrigins("http://localhost:3000", "https://myapp.com")
            .allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS")
            .allowedHeaders("*")
            .allowCredentials(true)
            .maxAge(3600);
    }
}
```

### SSL/TLS

> [ssl/tls | TIL](/ssltls/README.md)

**SSL (Secure Sockets Layer)**과 **TLS (Transport Layer Security)**는 컴퓨터 네트워크에서 안전한 통신을 제공하는 암호화 프로토콜입니다. 클라이언트와 서버 간에 전송되는 데이터가 기밀성을 유지하고, 변경되지 않으며, 무단 액세스로부터 보호되도록 보장합니다.

SSL은 1990년대 Netscape에서 개발한 오래된 프로토콜이며, TLS는 IETF에서 개발한 더 현대적이고 안전한 SSL의 후속 프로토콜입니다. 2018년 8월 기준 TLS 버전 1.3이 최신입니다.

**SSL/TLS 핸드셰이크 프로세스**

1. 클라이언트가 연결을 시작하고 지원하는 암호화 알고리즘과 매개변수를 서버에 전송
2. 서버는 선택한 암호화 알고리즘, 디지털 인증서 및 공개 키로 응답
3. 클라이언트는 CA(Certificate Authority)의 도움을 받아 서버의 인증서를 확인하여 중간자 공격 방지
4. 클라이언트는 서버의 공개 키를 사용하여 공유 비밀(세션 키)을 생성하고 암호화하여 전송
5. 서버는 개인 키를 사용하여 세션 키를 해독하고, 이후 양측이 세션 키로 데이터를 암호화/해독

**Java에서 SSL/TLS 구현**

```java
import javax.net.ssl.*;
import java.io.*;
import java.security.KeyStore;

public class SSLServerExample {

    public static void main(String[] args) throws Exception {
        // KeyStore 로드
        KeyStore keyStore = KeyStore.getInstance("JKS");
        FileInputStream keyStoreFile = new FileInputStream("server.keystore");
        keyStore.load(keyStoreFile, "password".toCharArray());

        // KeyManagerFactory 초기화
        KeyManagerFactory keyManagerFactory =
            KeyManagerFactory.getInstance(KeyManagerFactory.getDefaultAlgorithm());
        keyManagerFactory.init(keyStore, "password".toCharArray());

        // SSLContext 생성
        SSLContext sslContext = SSLContext.getInstance("TLS");
        sslContext.init(
            keyManagerFactory.getKeyManagers(),
            null,
            new java.security.SecureRandom()
        );

        // SSLServerSocket 생성
        SSLServerSocketFactory serverSocketFactory = sslContext.getServerSocketFactory();
        SSLServerSocket serverSocket =
            (SSLServerSocket) serverSocketFactory.createServerSocket(8443);

        // TLS 버전 설정
        serverSocket.setEnabledProtocols(new String[]{"TLSv1.2", "TLSv1.3"});

        System.out.println("SSL Server started on port 8443");
    }
}
```

**Spring Boot에서 SSL/TLS 설정**

```properties
# application.properties
server.port=8443
server.ssl.enabled=true
server.ssl.key-store=classpath:keystore.p12
server.ssl.key-store-password=changeit
server.ssl.key-store-type=PKCS12
server.ssl.key-alias=tomcat
server.ssl.enabled-protocols=TLSv1.2,TLSv1.3
```

**SSL/TLS 프로토콜 버전 비교**

| 프로토콜 | 연도 | 상태 | 보안 수준 |
|---------|------|------|----------|
| SSL 2.0 | 1995 | 사용 중단 | 매우 취약 |
| SSL 3.0 | 1996 | 사용 중단 | 취약 |
| TLS 1.0 | 1999 | 사용 중단 | 취약 |
| TLS 1.1 | 2006 | 사용 중단 | 낮음 |
| TLS 1.2 | 2008 | 권장 | 높음 |
| TLS 1.3 | 2018 | 최신/권장 | 매우 높음 |


### JWT 상세 구조

**JWT 구성:**
```
Header.Payload.Signature
```

**1. Header**
```json
{
  "alg": "HS256",
  "typ": "JWT"
}
```

**2. Payload**
```json
{
  "sub": "1234567890",
  "name": "John Doe",
  "iat": 1516239022,
  "exp": 1516242622,
  "roles": ["user", "admin"]
}
```

**3. Signature**
```
HMACSHA256(
  base64UrlEncode(header) + "." +
  base64UrlEncode(payload),
  secret
)
```

**Python 구현:**

```python
import jwt
import time

class JWTManager:
    def __init__(self, secret_key):
        self.secret = secret_key

    def generate_token(self, user_id, roles, expires_in=3600):
        payload = {
            'sub': user_id,
            'roles': roles,
            'iat': int(time.time()),
            'exp': int(time.time()) + expires_in
        }

        token = jwt.encode(payload, self.secret, algorithm='HS256')
        return token

    def verify_token(self, token):
        try:
            payload = jwt.decode(token, self.secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise Exception('Token expired')
        except jwt.InvalidTokenError:
            raise Exception('Invalid token')

# 사용 예제
jwt_manager = JWTManager('your-secret-key')

# 토큰 생성
token = jwt_manager.generate_token(
    user_id='user123',
    roles=['user', 'admin'],
    expires_in=3600
)

# 토큰 검증
try:
    payload = jwt_manager.verify_token(token)
    print(f"User: {payload['sub']}, Roles: {payload['roles']}")
except Exception as e:
    print(f"Validation failed: {e}")
```

### OAuth 2.0 Flow

**Authorization Code Flow (가장 안전)**

```
+----------+
| Resource |
|  Owner   |
|          |
+----------+
     ^
     |
    (B)
+----|-----+          Client Identifier      +---------------+
|         -+----(A)-- & Redirection URI ---->|               |
|  User-   |                                 | Authorization |
|  Agent  -+----(B)-- User authenticates --->|     Server    |
|          |                                 |               |
|         -+----(C)-- Authorization Code ---<|               |
+-|----|---+                                 +---------------+
  |    |                                         ^      v
 (A)  (C)                                        |      |
  |    |                                         |      |
  ^    v                                         |      |
+---------+                                      |      |
|         |>---(D)-- Authorization Code ---------'      |
|  Client |          & Redirection URI                  |
|         |                                             |
|         |<---(E)----- Access Token -------------------'
+---------+       (w/ Optional Refresh Token)
```

**구현 예제:**

```python
from flask import Flask, request, redirect
import requests

app = Flask(__name__)

# OAuth 설정
CLIENT_ID = 'your-client-id'
CLIENT_SECRET = 'your-client-secret'
REDIRECT_URI = 'http://localhost:5000/callback'
AUTH_URL = 'https://oauth-provider.com/authorize'
TOKEN_URL = 'https://oauth-provider.com/token'

# Step 1: 사용자를 인증 서버로 리다이렉트
@app.route('/login')
def login():
    params = {
        'client_id': CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'response_type': 'code',
        'scope': 'read write',
        'state': 'random_state_string'  # CSRF 방지
    }
    auth_url = f"{AUTH_URL}?{urlencode(params)}"
    return redirect(auth_url)

# Step 2: 인증 서버에서 리다이렉트된 콜백 처리
@app.route('/callback')
def callback():
    # Authorization code 받기
    code = request.args.get('code')
    state = request.args.get('state')

    # State 검증 (CSRF 방지)
    if state != 'random_state_string':
        return 'Invalid state', 400

    # Step 3: Authorization code로 Access token 교환
    token_data = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': REDIRECT_URI,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET
    }

    response = requests.post(TOKEN_URL, data=token_data)
    tokens = response.json()

    access_token = tokens['access_token']
    refresh_token = tokens.get('refresh_token')

    # Access token으로 보호된 리소스 접근
    user_data = get_user_data(access_token)

    return f'Logged in as {user_data["name"]}'

def get_user_data(access_token):
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get('https://api.example.com/user', headers=headers)
    return response.json()

# Refresh token으로 새 access token 발급
def refresh_access_token(refresh_token):
    token_data = {
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET
    }
    response = requests.post(TOKEN_URL, data=token_data)
    return response.json()['access_token']
```

**Client Credentials Flow (서버 간 통신)**

```python
def get_server_access_token():
    token_data = {
        'grant_type': 'client_credentials',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'scope': 'api.read api.write'
    }

    response = requests.post(TOKEN_URL, data=token_data)
    return response.json()['access_token']

# 사용 예제
access_token = get_server_access_token()
headers = {'Authorization': f'Bearer {access_token}'}
api_response = requests.get('https://api.example.com/data', headers=headers)
```

### mTLS (Mutual TLS)

**개념:**

mTLS(Mutual TLS, 상호 TLS)는 클라이언트와 서버가 TLS 핸드셰이크 과정에서 서로의 신원을 검증하는 강화된 보안 프로토콜입니다. 일반 TLS는 서버만 클라이언트에게 자신의 신원을 증명하지만, mTLS는 클라이언트도 서버에게 자신의 신원을 증명해야 합니다.

**일반 TLS vs mTLS:**

| 특징 | 일반 TLS | mTLS |
|------|---------|------|
| 서버 인증 | ✓ | ✓ |
| 클라이언트 인증 | ✗ | ✓ |
| 인증서 필요 | 서버만 | 서버 + 클라이언트 |
| 사용 사례 | 웹사이트 (HTTPS) | API, 마이크로서비스 |

**mTLS 핸드셰이크 과정:**

```
Client                                Server
  |                                      |
  |----(1) ClientHello ----------------->|
  |     (지원하는 암호화 알고리즘)          |
  |                                      |
  |<---(2) ServerHello ------------------|
  |     (선택된 암호화 알고리즘)            |
  |     (서버 인증서 + 공개키)             |
  |                                      |
  |----(3) Client 인증서 확인 ------------>|
  |     (CA를 통한 서버 인증서 검증)        |
  |                                      |
  |----(4) ClientCertificate ----------->|
  |     (클라이언트 인증서 + 공개키)        |
  |                                      |
  |<---(5) Server 인증서 확인 ------------|
  |     (CA를 통한 클라이언트 인증서 검증)  |
  |                                      |
  |----(6) Session Key 교환 ------------>|
  |<------------------------------------ |
  |                                      |
  |====== 암호화된 통신 시작 =============|
```

**Python 구현 예제:**

```python
import ssl
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler

# mTLS 서버 설정
class MTLSServer:
    def __init__(self, host='localhost', port=8443):
        self.host = host
        self.port = port

    def create_ssl_context(self):
        # SSL 컨텍스트 생성 (서버)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

        # 서버 인증서와 개인키 로드
        context.load_cert_chain(
            certfile='server.crt',
            keyfile='server.key'
        )

        # 클라이언트 인증 요구
        context.verify_mode = ssl.CERT_REQUIRED

        # 클라이언트 인증서 검증을 위한 CA 인증서 로드
        context.load_verify_locations(cafile='ca.crt')

        return context

    def start(self):
        server = HTTPServer((self.host, self.port), RequestHandler)
        server.socket = self.create_ssl_context().wrap_socket(
            server.socket,
            server_side=True
        )
        print(f"mTLS Server running on https://{self.host}:{self.port}")
        server.serve_forever()

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # 클라이언트 인증서 정보 확인
        client_cert = self.connection.getpeercert()
        client_cn = dict(x[0] for x in client_cert['subject'])['commonName']

        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(f"Authenticated client: {client_cn}".encode())

# mTLS 클라이언트 설정
class MTLSClient:
    def __init__(self, server_host='localhost', server_port=8443):
        self.server_host = server_host
        self.server_port = server_port

    def create_ssl_context(self):
        # SSL 컨텍스트 생성 (클라이언트)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

        # 클라이언트 인증서와 개인키 로드
        context.load_cert_chain(
            certfile='client.crt',
            keyfile='client.key'
        )

        # 서버 인증서 검증을 위한 CA 인증서 로드
        context.load_verify_locations(cafile='ca.crt')

        # 서버 인증서 검증 필수
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED

        return context

    def make_request(self, path='/'):
        context = self.create_ssl_context()

        with socket.create_connection((self.server_host, self.server_port)) as sock:
            with context.wrap_socket(sock, server_hostname=self.server_host) as ssock:
                # 서버 인증서 정보 확인
                server_cert = ssock.getpeercert()
                print(f"Connected to: {server_cert['subject']}")

                # HTTP 요청 전송
                request = f"GET {path} HTTP/1.1\r\nHost: {self.server_host}\r\n\r\n"
                ssock.sendall(request.encode())

                # 응답 수신
                response = ssock.recv(4096).decode()
                return response

# 사용 예제
if __name__ == '__main__':
    # 서버 시작 (별도 프로세스)
    # server = MTLSServer()
    # server.start()

    # 클라이언트 요청
    client = MTLSClient()
    try:
        response = client.make_request()
        print(f"Response: {response}")
    except ssl.SSLError as e:
        print(f"SSL Error: {e}")
```

**인증서 생성 (OpenSSL):**

```bash
# 1. CA (Certificate Authority) 인증서 생성
openssl req -x509 -newkey rsa:4096 -days 365 -nodes \
  -keyout ca.key -out ca.crt \
  -subj "/CN=MyCA/O=MyOrg/C=US"

# 2. 서버 개인키 및 CSR 생성
openssl req -newkey rsa:4096 -nodes \
  -keyout server.key -out server.csr \
  -subj "/CN=localhost/O=MyOrg/C=US"

# 3. 서버 인증서 서명
openssl x509 -req -in server.csr -days 365 \
  -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out server.crt

# 4. 클라이언트 개인키 및 CSR 생성
openssl req -newkey rsa:4096 -nodes \
  -keyout client.key -out client.csr \
  -subj "/CN=client1/O=MyOrg/C=US"

# 5. 클라이언트 인증서 서명
openssl x509 -req -in client.csr -days 365 \
  -CA ca.crt -CAkey ca.key -CAcreateserial \
  -out client.crt
```

**Nginx mTLS 설정:**

```nginx
server {
    listen 443 ssl;
    server_name api.example.com;

    # 서버 인증서 설정
    ssl_certificate /etc/nginx/ssl/server.crt;
    ssl_certificate_key /etc/nginx/ssl/server.key;

    # 클라이언트 인증 활성화
    ssl_client_certificate /etc/nginx/ssl/ca.crt;
    ssl_verify_client on;
    ssl_verify_depth 2;

    # TLS 설정
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        # 클라이언트 인증서 정보를 헤더로 전달
        proxy_set_header X-Client-DN $ssl_client_s_dn;
        proxy_set_header X-Client-Cert $ssl_client_cert;

        proxy_pass http://backend;
    }
}
```

**마이크로서비스에서의 mTLS:**

```python
# Service Mesh (Istio) 없이 mTLS 구현
class MicroserviceClient:
    def __init__(self, service_name):
        self.service_name = service_name
        self.cert_path = f'/etc/certs/{service_name}'

    def call_service(self, target_service, endpoint, data):
        import requests

        # mTLS 설정
        cert = (
            f'{self.cert_path}/client.crt',
            f'{self.cert_path}/client.key'
        )
        verify = '/etc/certs/ca.crt'

        url = f'https://{target_service}:443{endpoint}'

        try:
            response = requests.post(
                url,
                json=data,
                cert=cert,
                verify=verify,
                timeout=5
            )
            return response.json()
        except requests.exceptions.SSLError as e:
            print(f"mTLS authentication failed: {e}")
            raise
        except requests.exceptions.Timeout:
            print(f"Service {target_service} timeout")
            raise

# 사용 예제
user_service = MicroserviceClient('user-service')
try:
    result = user_service.call_service(
        target_service='payment-service',
        endpoint='/api/v1/charge',
        data={'user_id': 123, 'amount': 100.00}
    )
    print(f"Payment result: {result}")
except Exception as e:
    print(f"Service call failed: {e}")
```

**mTLS 사용 사례:**

1. **API 보안:**
   - REST API 엔드포인트 보호
   - 인증된 클라이언트만 접근 허용
   - API 키 대신 인증서 기반 인증

2. **마이크로서비스 통신:**
   - Service-to-Service 통신 암호화
   - Zero Trust 네트워크 구현
   - Service Mesh (Istio, Linkerd)에서 자동 mTLS

3. **IoT 디바이스:**
   - 디바이스 신원 검증
   - 중간자 공격 방지
   - 안전한 펌웨어 업데이트

4. **원격 접근:**
   - VPN 대체
   - 관리자 접근 제어
   - 특권 계정 보호

**mTLS 장단점:**

**장점:**
- 양방향 인증으로 높은 보안성
- 중간자 공격 방지
- 네트워크 레벨 인증
- 암호화된 통신

**단점:**
- 인증서 관리 복잡도 증가
- 인증서 갱신 자동화 필요
- 초기 설정 복잡
- 성능 오버헤드 (핸드셰이크)

**모범 사례:**

1. 인증서 자동 갱신 (cert-manager, Let's Encrypt)
2. 인증서 만료 모니터링
3. 인증서 회전(Rotation) 전략
4. 최소 권한 원칙 적용
5. Service Mesh 활용 고려


### OpenID Connect (OIDC)

> [OpenID Connect (OIDC)](/oidc/README.md)

**개념**

OpenID Connect (OIDC)는 OAuth 2.0 프로토콜 위에 구축된 인증 계층입니다. 웹 애플리케이션, 모바일 앱, API에 대한 사용자 인증 및 Single Sign-On (SSO) 기능을 안전하게 제공하는 간단하고 표준화된 방법입니다. OAuth 2.0이 타사 애플리케이션에 리소스 접근 권한을 부여하는 **인가(Authorization)**에 초점을 맞춘다면, OIDC는 이를 확장하여 사용자 **인증(Authentication)** 및 신원 관리를 제공합니다.

**OAuth 2.0 vs OIDC 비교**:

| 구분 | OAuth 2.0 | OpenID Connect (OIDC) |
|------|-----------|----------------------|
| **목적** | 인가 (Authorization) | 인증 (Authentication) + 인가 |
| **사용 사례** | API 접근 권한 부여 | 사용자 로그인, SSO |
| **토큰** | Access Token | Access Token + **ID Token** |
| **사용자 정보** | 포함 안 됨 | ID Token에 포함 (JWT) |
| **표준 엔드포인트** | /token | /token, **/userinfo**, /.well-known/openid-configuration |

**주요 구성 요소**:

**1. ID Token (신원 토큰)**
- **JSON Web Token (JWT)** 형식
- 사용자 속성(claims) 포함: 사용자 ID, 이메일, 이름 등
- 인증 이벤트 정보 포함: 인증 시간, 인증 방법
- OIDC Identity Provider (IdP)가 발급
- 디지털 서명으로 진위성 및 무결성 보장

**2. UserInfo Endpoint**
- Identity Provider가 제공하는 API
- 사용자 클레임(claims) 반환
- 인구 통계 정보, 이메일 주소 등 사용자 속성 제공

**3. Discovery (자동 검색)**
- Well-known URL을 통한 표준 검색 메커니즘
- 클라이언트 애플리케이션이 구성 및 엔드포인트 정보 자동 검색
- `/.well-known/openid-configuration` 엔드포인트

**4. Standardized Flows (표준화된 흐름)**
- Authorization Code Flow (권장)
- Implicit Flow (레거시)
- Hybrid Flow

### OIDC 구현 예제

**Spring Security + OIDC 통합**:

```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/", "/login**", "/error**").permitAll()
                .anyRequest().authenticated()
            )
            .oauth2Login(oauth2 -> oauth2
                .userInfoEndpoint(userInfo -> userInfo
                    .oidcUserService(oidcUserService())
                )
            );

        return http.build();
    }

    @Bean
    public OidcUserService oidcUserService() {
        return new OidcUserService() {
            @Override
            public OidcUser loadUser(OidcUserRequest userRequest) throws OAuth2AuthenticationException {
                OidcUser oidcUser = super.loadUser(userRequest);

                // ID Token에서 사용자 정보 추출
                OidcIdToken idToken = oidcUser.getIdToken();
                String email = idToken.getEmail();
                String name = idToken.getFullName();
                String subject = idToken.getSubject(); // 고유 사용자 ID

                // 애플리케이션 사용자로 변환/저장
                User user = userRepository.findByEmail(email)
                    .orElseGet(() -> createNewUser(email, name, subject));

                return oidcUser;
            }
        };
    }
}

// application.yml 설정
/*
spring:
  security:
    oauth2:
      client:
        registration:
          google:
            client-id: ${GOOGLE_CLIENT_ID}
            client-secret: ${GOOGLE_CLIENT_SECRET}
            scope:
              - openid
              - profile
              - email
        provider:
          google:
            issuer-uri: https://accounts.google.com
*/
```

**수동 OIDC 구현 (교육 목적)**:

```java
@RestController
@RequestMapping("/auth")
public class OIDCController {

    @Autowired
    private RestTemplate restTemplate;

    @Autowired
    private JwtDecoder jwtDecoder;

    // 1. Authorization Code Flow 시작
    @GetMapping("/login")
    public RedirectView login() {
        String authorizationUrl = String.format(
            "%s?client_id=%s&redirect_uri=%s&response_type=code&scope=openid profile email&state=%s",
            "https://accounts.google.com/o/oauth2/v2/auth",
            clientId,
            URLEncoder.encode(redirectUri, StandardCharsets.UTF_8),
            generateState() // CSRF 방지
        );

        return new RedirectView(authorizationUrl);
    }

    // 2. 콜백 처리 및 토큰 교환
    @GetMapping("/callback")
    public ResponseEntity<AuthResponse> callback(
            @RequestParam String code,
            @RequestParam String state) {

        // State 검증 (CSRF 방지)
        if (!validateState(state)) {
            throw new SecurityException("Invalid state parameter");
        }

        // Authorization Code를 Access Token + ID Token으로 교환
        TokenResponse tokens = exchangeCodeForTokens(code);

        // ID Token 검증 및 디코딩
        Jwt idToken = jwtDecoder.decode(tokens.getIdToken());

        // 사용자 정보 추출
        String userId = idToken.getSubject();
        String email = idToken.getClaim("email");
        String name = idToken.getClaim("name");
        String picture = idToken.getClaim("picture");

        // 애플리케이션 세션 생성
        User user = createOrUpdateUser(userId, email, name, picture);
        String sessionToken = createSessionToken(user);

        return ResponseEntity.ok(new AuthResponse(sessionToken, user));
    }

    private TokenResponse exchangeCodeForTokens(String code) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);

        MultiValueMap<String, String> body = new LinkedMultiValueMap<>();
        body.add("grant_type", "authorization_code");
        body.add("code", code);
        body.add("client_id", clientId);
        body.add("client_secret", clientSecret);
        body.add("redirect_uri", redirectUri);

        HttpEntity<MultiValueMap<String, String>> request = new HttpEntity<>(body, headers);

        ResponseEntity<TokenResponse> response = restTemplate.postForEntity(
            "https://oauth2.googleapis.com/token",
            request,
            TokenResponse.class
        );

        return response.getBody();
    }

    // 3. UserInfo Endpoint 호출 (선택적)
    @GetMapping("/userinfo")
    public ResponseEntity<UserInfo> getUserInfo(@RequestHeader("Authorization") String accessToken) {
        HttpHeaders headers = new HttpHeaders();
        headers.setBearerAuth(accessToken.replace("Bearer ", ""));

        HttpEntity<?> request = new HttpEntity<>(headers);

        ResponseEntity<UserInfo> response = restTemplate.exchange(
            "https://openidconnect.googleapis.com/v1/userinfo",
            HttpMethod.GET,
            request,
            UserInfo.class
        );

        return response;
    }

    // ID Token 검증
    private boolean validateIdToken(String idToken) {
        try {
            Jwt jwt = jwtDecoder.decode(idToken);

            // 발급자 검증
            String issuer = jwt.getIssuer().toString();
            if (!issuer.equals("https://accounts.google.com")) {
                return false;
            }

            // 대상(Audience) 검증
            List<String> audience = jwt.getAudience();
            if (!audience.contains(clientId)) {
                return false;
            }

            // 만료 시간 검증
            Instant expiresAt = jwt.getExpiresAt();
            if (expiresAt.isBefore(Instant.now())) {
                return false;
            }

            // 발급 시간 검증 (너무 오래된 토큰 거부)
            Instant issuedAt = jwt.getIssuedAt();
            if (issuedAt.isBefore(Instant.now().minus(Duration.ofMinutes(5)))) {
                return false;
            }

            return true;

        } catch (Exception e) {
            log.error("ID Token 검증 실패", e);
            return false;
        }
    }
}

@Data
class TokenResponse {
    @JsonProperty("access_token")
    private String accessToken;

    @JsonProperty("id_token")
    private String idToken;

    @JsonProperty("expires_in")
    private Integer expiresIn;

    @JsonProperty("token_type")
    private String tokenType;

    @JsonProperty("refresh_token")
    private String refreshToken;
}

@Data
class UserInfo {
    private String sub; // Subject (고유 사용자 ID)
    private String email;
    private String name;
    private String picture;
    private Boolean emailVerified;
}
```

### Single Sign-On (SSO)

**개념**

Single Sign-On (SSO)은 사용자가 한 번의 로그인으로 조직 내 여러 애플리케이션, 시스템 또는 서비스에 접근할 수 있도록 하는 인증 프로세스입니다. 사용자는 각 애플리케이션마다 별도의 사용자 이름과 비밀번호를 기억하고 입력할 필요가 없어 원활하고 편리한 사용자 경험을 제공합니다.

**SSO 장점**:

| 장점 | 설명 |
|------|------|
| **향상된 사용자 경험** | 시스템 간 이동 시 자격 증명 재입력 불필요 |
| **보안 강화** | 기억할 비밀번호 수 감소 → 더 복잡하고 안전한 비밀번호 사용 가능 |
| **헬프데스크 문의 감소** | 비밀번호 재설정 및 분실 문의 감소 |
| **간소화된 사용자 관리** | 중앙 집중식 인증 관리로 접근 제어 용이 |
| **규정 준수** | 중앙화된 감사 로그 및 접근 제어 |

**SSO 구현 방법**:

1. **SAML (Security Assertion Markup Language)**
   - XML 기반 프로토콜
   - 엔터프라이즈 환경에서 널리 사용
   - IdP와 SP 간 신뢰 관계 필요

2. **OAuth 2.0 + OpenID Connect (OIDC)**
   - 현대적인 웹/모바일 애플리케이션에 적합
   - JSON/REST 기반
   - 클라우드 네이티브 환경에 최적화

3. **Kerberos**
   - 네트워크 인증 프로토콜
   - 주로 Windows Active Directory 환경

### SSO 구현 예제

**SAML 기반 SSO (Spring Security)**:

```java
@Configuration
@EnableWebSecurity
public class SamlSecurityConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/saml/**").permitAll()
                .anyRequest().authenticated()
            )
            .saml2Login(saml2 -> saml2
                .relyingPartyRegistrationRepository(relyingPartyRegistrations())
            );

        return http.build();
    }

    @Bean
    public RelyingPartyRegistrationRepository relyingPartyRegistrations() {
        RelyingPartyRegistration registration = RelyingPartyRegistrations
            .fromMetadataLocation("https://idp.example.com/metadata")
            .registrationId("okta")
            .build();

        return new InMemoryRelyingPartyRegistrationRepository(registration);
    }

    // SAML 응답 처리
    @Bean
    public Converter<OpenSaml4AuthenticationProvider.ResponseToken, Saml2Authentication>
            responseAuthenticationConverter() {
        return responseToken -> {
            Saml2AuthenticatedPrincipal principal =
                (Saml2AuthenticatedPrincipal) responseToken.getAuthentication().getPrincipal();

            // SAML Assertion에서 사용자 정보 추출
            String username = principal.getName();
            String email = principal.getFirstAttribute("email");
            List<String> groups = principal.getAttribute("groups");

            // 권한 매핑
            Set<GrantedAuthority> authorities = groups.stream()
                .map(group -> new SimpleGrantedAuthority("ROLE_" + group))
                .collect(Collectors.toSet());

            return new Saml2Authentication(principal,
                responseToken.getToken().getSaml2Response(), authorities);
        };
    }
}
```

**JWT 기반 SSO (마이크로서비스 환경)**:

```java
@Service
public class SSOTokenService {

    @Value("${jwt.secret}")
    private String jwtSecret;

    @Value("${jwt.expiration}")
    private long jwtExpirationMs;

    // SSO 토큰 생성
    public String generateSSOToken(User user) {
        Date now = new Date();
        Date expiryDate = new Date(now.getTime() + jwtExpirationMs);

        return Jwts.builder()
            .setSubject(user.getId().toString())
            .claim("email", user.getEmail())
            .claim("name", user.getName())
            .claim("roles", user.getRoles())
            .claim("sso_session_id", UUID.randomUUID().toString())
            .setIssuedAt(now)
            .setExpiration(expiryDate)
            .signWith(SignatureAlgorithm.HS512, jwtSecret)
            .compact();
    }

    // SSO 토큰 검증
    public Claims validateSSOToken(String token) {
        try {
            return Jwts.parser()
                .setSigningKey(jwtSecret)
                .parseClaimsJws(token)
                .getBody();
        } catch (JwtException e) {
            throw new SSOAuthenticationException("Invalid SSO token", e);
        }
    }

    // SSO 세션 관리 (Redis)
    @Autowired
    private RedisTemplate<String, String> redisTemplate;

    public void createSSOSession(String sessionId, User user) {
        String key = "sso:session:" + sessionId;

        Map<String, String> sessionData = new HashMap<>();
        sessionData.put("userId", user.getId().toString());
        sessionData.put("email", user.getEmail());
        sessionData.put("loginTime", Instant.now().toString());

        redisTemplate.opsForHash().putAll(key, sessionData);
        redisTemplate.expire(key, Duration.ofHours(8)); // 8시간 세션
    }

    public void invalidateSSOSession(String sessionId) {
        redisTemplate.delete("sso:session:" + sessionId);
    }

    public boolean isValidSSOSession(String sessionId) {
        return redisTemplate.hasKey("sso:session:" + sessionId);
    }
}

// SSO Filter
@Component
public class SSOAuthenticationFilter extends OncePerRequestFilter {

    @Autowired
    private SSOTokenService ssoTokenService;

    @Override
    protected void doFilterInternal(HttpServletRequest request,
                                     HttpServletResponse response,
                                     FilterChain filterChain)
            throws ServletException, IOException {

        String token = extractSSOToken(request);

        if (token != null) {
            try {
                Claims claims = ssoTokenService.validateSSOToken(token);
                String sessionId = claims.get("sso_session_id", String.class);

                // SSO 세션 유효성 확인
                if (ssoTokenService.isValidSSOSession(sessionId)) {
                    // Spring Security 컨텍스트에 인증 정보 설정
                    Authentication authentication = createAuthentication(claims);
                    SecurityContextHolder.getContext().setAuthentication(authentication);
                }

            } catch (Exception e) {
                log.error("SSO 인증 실패", e);
            }
        }

        filterChain.doFilter(request, response);
    }

    private String extractSSOToken(HttpServletRequest request) {
        // Cookie에서 SSO 토큰 추출
        Cookie[] cookies = request.getCookies();
        if (cookies != null) {
            for (Cookie cookie : cookies) {
                if ("SSO_TOKEN".equals(cookie.getName())) {
                    return cookie.getValue();
                }
            }
        }

        // Header에서 추출
        String bearerToken = request.getHeader("Authorization");
        if (bearerToken != null && bearerToken.startsWith("Bearer ")) {
            return bearerToken.substring(7);
        }

        return null;
    }
}
```

**SSO 로그아웃 (Single Logout)**:

```java
@RestController
@RequestMapping("/sso")
public class SSOController {

    @Autowired
    private SSOTokenService ssoTokenService;

    @Autowired
    private RestTemplate restTemplate;

    // 모든 애플리케이션에서 로그아웃
    @PostMapping("/logout")
    public ResponseEntity<Void> logout(
            @CookieValue("SSO_TOKEN") String token,
            HttpServletResponse response) {

        try {
            Claims claims = ssoTokenService.validateSSOToken(token);
            String sessionId = claims.get("sso_session_id", String.class);

            // SSO 세션 무효화
            ssoTokenService.invalidateSSOSession(sessionId);

            // 모든 연결된 애플리케이션에 로그아웃 알림
            List<String> registeredApps = getRegisteredApplications();
            for (String appUrl : registeredApps) {
                notifyLogout(appUrl, sessionId);
            }

            // SSO 쿠키 삭제
            Cookie cookie = new Cookie("SSO_TOKEN", null);
            cookie.setMaxAge(0);
            cookie.setPath("/");
            cookie.setHttpOnly(true);
            cookie.setSecure(true);
            response.addCookie(cookie);

            return ResponseEntity.ok().build();

        } catch (Exception e) {
            log.error("SSO 로그아웃 실패", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }

    private void notifyLogout(String appUrl, String sessionId) {
        try {
            restTemplate.postForEntity(
                appUrl + "/sso/logout-callback",
                new LogoutRequest(sessionId),
                Void.class
            );
        } catch (Exception e) {
            log.warn("앱 {}에 로그아웃 알림 실패", appUrl, e);
        }
    }
}
```

**보안 고려사항**:

| 항목 | 권장사항 |
|------|---------|
| **토큰 전송** | HTTPS만 사용, HttpOnly 쿠키 |
| **세션 관리** | 타임아웃 설정, 유휴 세션 자동 종료 |
| **CSRF 방지** | State 파라미터, CSRF 토큰 |
| **토큰 저장** | 민감 정보 제외, 암호화 권장 |
| **Single Logout** | 모든 앱에서 세션 무효화 |
| **감사 로그** | 모든 인증 이벤트 기록 |

### API 보안 (API Security)

**API 보안**은 애플리케이션 프로그래밍 인터페이스(API)를 무단 접근, 오용, 취약점으로부터 보호하기 위한 관행과 조치를 의미합니다. API는 현대 애플리케이션의 핵심 구성 요소로서 소프트웨어 시스템을 연결하고, 데이터를 공유하며, 기능을 확장하는 방법을 제공하므로 API 보안은 매우 중요합니다.

#### API 보안 핵심 구성 요소

```
┌─────────────────────────────────────┐
│         API 보안 계층                │
├─────────────────────────────────────┤
│  인증 (Authentication)               │
│  • OAuth2, WebAuthn                 │
│  • API Keys (레벨별)                 │
├─────────────────────────────────────┤
│  인가 (Authorization)                │
│  • 역할 기반 접근 제어 (RBAC)         │
│  • 리소스 수준 권한                   │
├─────────────────────────────────────┤
│  통신 보안                            │
│  • HTTPS/TLS 암호화                  │
│  • 인증서 검증                        │
├─────────────────────────────────────┤
│  요청 제어                            │
│  • Rate Limiting                    │
│  • Throttling                       │
│  • IP Whitelisting                  │
├─────────────────────────────────────┤
│  데이터 보호                          │
│  • Input Validation                 │
│  • Output Encoding                  │
│  • Error Handling                   │
└─────────────────────────────────────┘
```

#### 1. HTTPS 사용

**목적**: 클라이언트와 API 서버 간의 통신 보안

**특징**:
- 전송 중인 데이터 암호화
- 데이터 기밀성 및 무결성 보장
- 중간자 공격(MITM) 방지

**구현 예제**:
```java
// Spring Boot에서 HTTPS 강제
@Configuration
public class SecurityConfig {
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http)
            throws Exception {
        http
            .requiresChannel(channel ->
                channel.anyRequest().requiresSecure()  // 모든 요청 HTTPS
            )
            .headers(headers ->
                headers.httpStrictTransportSecurity(hsts ->
                    hsts.maxAgeInSeconds(31536000)  // HSTS 1년
                        .includeSubDomains(true)
                )
            );
        return http.build();
    }
}
```

#### 2. OAuth2 사용

**목적**: 표준화된 인가 프로토콜

**특징**:
- 액세스 토큰 기반 인증
- 제한된 권한 부여 (Scope)
- 토큰 갱신 메커니즘

**구현 예제**:
```java
// OAuth2 Resource Server 설정
@Configuration
@EnableResourceServer
public class OAuth2ResourceServerConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http)
            throws Exception {
        http
            .oauth2ResourceServer(oauth2 ->
                oauth2.jwt(jwt ->
                    jwt.decoder(jwtDecoder())
                )
            )
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/api/public/**").permitAll()
                .requestMatchers("/api/admin/**")
                    .hasAuthority("SCOPE_admin")
                .anyRequest().authenticated()
            );
        return http.build();
    }

    @Bean
    public JwtDecoder jwtDecoder() {
        return JwtDecoders.fromIssuerLocation(
            "https://auth-server.com"
        );
    }
}

// API 엔드포인트
@RestController
@RequestMapping("/api")
public class SecureApiController {

    @GetMapping("/profile")
    public UserProfile getProfile(
            @AuthenticationPrincipal Jwt jwt) {
        String userId = jwt.getSubject();
        List<String> scopes = jwt.getClaimAsStringList("scope");

        return userService.getProfile(userId);
    }
}
```

#### 3. WebAuthn 사용

**목적**: 비밀번호 없는 안전한 인증

**특징**:
- 피싱 공격 방지
- 생체 인증 지원
- 하드웨어 보안 키 사용

**구현 예제**:
```java
// WebAuthn 등록
@RestController
@RequestMapping("/api/webauthn")
public class WebAuthnController {

    @PostMapping("/register/start")
    public PublicKeyCredentialCreationOptions startRegistration(
            @RequestParam String username) {

        return PublicKeyCredentialCreationOptions.builder()
            .rp(RelyingPartyIdentity.builder()
                .name("My App")
                .id("example.com")
                .build())
            .user(UserIdentity.builder()
                .name(username)
                .displayName(username)
                .id(generateUserId())
                .build())
            .challenge(generateChallenge())
            .pubKeyCredParams(Arrays.asList(
                PublicKeyCredentialParameters.ES256,
                PublicKeyCredentialParameters.RS256
            ))
            .timeout(60000L)
            .build();
    }

    @PostMapping("/register/finish")
    public RegistrationResult finishRegistration(
            @RequestBody PublicKeyCredential credential) {
        // 자격 증명 검증 및 저장
        return registrationService.verify(credential);
    }
}
```

#### 4. 레벨별 API 키 사용

**목적**: 사용자 및 사용 사례별 차등 접근 제어

**구현 예제**:
```java
// API Key 레벨 정의
public enum ApiKeyLevel {
    FREE(100, Arrays.asList("read")),           // 시간당 100회
    BASIC(1000, Arrays.asList("read", "write")), // 시간당 1000회
    PREMIUM(10000, Arrays.asList("read", "write", "admin")); // 시간당 10000회

    private final int rateLimit;
    private final List<String> permissions;

    ApiKeyLevel(int rateLimit, List<String> permissions) {
        this.rateLimit = rateLimit;
        this.permissions = permissions;
    }
}

// API Key 검증 필터
@Component
public class ApiKeyFilter extends OncePerRequestFilter {

    @Override
    protected void doFilterInternal(
            HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain) throws ServletException, IOException {

        String apiKey = request.getHeader("X-API-Key");

        if (apiKey == null) {
            response.sendError(HttpServletResponse.SC_UNAUTHORIZED,
                "API Key 누락");
            return;
        }

        // API Key 검증
        ApiKeyInfo keyInfo = apiKeyService.validate(apiKey);
        if (keyInfo == null) {
            response.sendError(HttpServletResponse.SC_UNAUTHORIZED,
                "유효하지 않은 API Key");
            return;
        }

        // Rate Limit 확인
        if (!rateLimiter.tryAcquire(apiKey, keyInfo.getLevel())) {
            response.sendError(HttpServletResponse.SC_TOO_MANY_REQUESTS,
                "Rate Limit 초과");
            return;
        }

        // 요청 컨텍스트에 API Key 정보 저장
        request.setAttribute("apiKeyInfo", keyInfo);

        filterChain.doFilter(request, response);
    }
}
```

#### 5. 인가 (Authorization)

**목적**: 역할 및 권한 기반 리소스 접근 제어

**구현 예제**:
```java
// 역할 기반 접근 제어
@RestController
@RequestMapping("/api/resources")
public class ResourceController {

    @GetMapping("/{id}")
    @PreAuthorize("hasPermission(#id, 'Resource', 'read')")
    public Resource getResource(@PathVariable Long id) {
        return resourceService.findById(id);
    }

    @PutMapping("/{id}")
    @PreAuthorize("hasPermission(#id, 'Resource', 'write')")
    public Resource updateResource(
            @PathVariable Long id,
            @RequestBody Resource resource) {
        return resourceService.update(id, resource);
    }

    @DeleteMapping("/{id}")
    @PreAuthorize("hasRole('ADMIN') or @resourceService.isOwner(#id, principal)")
    public void deleteResource(@PathVariable Long id) {
        resourceService.delete(id);
    }
}

// 커스텀 Permission Evaluator
@Component
public class CustomPermissionEvaluator implements PermissionEvaluator {

    @Override
    public boolean hasPermission(
            Authentication authentication,
            Object targetDomainObject,
            Object permission) {

        if (authentication == null || !(permission instanceof String)) {
            return false;
        }

        String targetType = targetDomainObject.getClass().getSimpleName();
        return hasPrivilege(authentication,
            targetType.toUpperCase(),
            permission.toString().toUpperCase());
    }

    private boolean hasPrivilege(
            Authentication auth,
            String targetType,
            String permission) {
        // 권한 확인 로직
        return auth.getAuthorities().stream()
            .anyMatch(grantedAuth ->
                grantedAuth.getAuthority()
                    .equals(targetType + "_" + permission)
            );
    }
}
```

#### 6. Rate Limiting (속도 제한)

**목적**: API 남용 및 과부하 방지

**구현 예제**:
```java
// Token Bucket 알고리즘을 사용한 Rate Limiter
@Component
public class RateLimiter {
    private final Cache<String, AtomicInteger> requestCounts;

    public RateLimiter() {
        this.requestCounts = Caffeine.newBuilder()
            .expireAfterWrite(1, TimeUnit.HOURS)
            .build();
    }

    public boolean tryAcquire(String apiKey, ApiKeyLevel level) {
        AtomicInteger count = requestCounts.get(apiKey,
            k -> new AtomicInteger(0));

        int currentCount = count.incrementAndGet();
        return currentCount <= level.getRateLimit();
    }
}

// Spring Cloud Gateway Rate Limiter
@Configuration
public class GatewayRateLimitConfig {

    @Bean
    public RouteLocator customRouteLocator(
            RouteLocatorBuilder builder) {
        return builder.routes()
            .route("api_route", r -> r
                .path("/api/**")
                .filters(f -> f
                    .requestRateLimiter(config -> config
                        .setRateLimiter(redisRateLimiter())
                        .setKeyResolver(userKeyResolver())
                    )
                )
                .uri("http://localhost:8080")
            )
            .build();
    }

    @Bean
    public RedisRateLimiter redisRateLimiter() {
        return new RedisRateLimiter(10, 20);  // 초당 10개, 버스트 20개
    }

    @Bean
    public KeyResolver userKeyResolver() {
        return exchange -> Mono.just(
            exchange.getRequest()
                .getHeaders()
                .getFirst("X-API-Key")
        );
    }
}
```

#### 7. API 버전 관리

**목적**: 기존 클라이언트를 손상시키지 않고 API 변경 관리

**구현 예제**:
```java
// URL 기반 버전 관리
@RestController
@RequestMapping("/api/v1/users")
public class UserControllerV1 {

    @GetMapping("/{id}")
    public UserV1 getUser(@PathVariable Long id) {
        return userService.getUserV1(id);
    }
}

@RestController
@RequestMapping("/api/v2/users")
public class UserControllerV2 {

    @GetMapping("/{id}")
    public UserV2 getUser(@PathVariable Long id) {
        // V2에서는 추가 필드 포함
        return userService.getUserV2(id);
    }
}

// 헤더 기반 버전 관리
@RestController
@RequestMapping("/api/users")
public class UserController {

    @GetMapping(value = "/{id}", headers = "API-Version=1")
    public UserV1 getUserV1(@PathVariable Long id) {
        return userService.getUserV1(id);
    }

    @GetMapping(value = "/{id}", headers = "API-Version=2")
    public UserV2 getUserV2(@PathVariable Long id) {
        return userService.getUserV2(id);
    }
}
```

#### 8. 화이트리스트 (Whitelisting)

**목적**: 신뢰할 수 있는 소스에서만 API 접근 허용

**구현 예제**:
```java
// IP 화이트리스트 필터
@Component
public class IpWhitelistFilter extends OncePerRequestFilter {

    @Value("${api.whitelist.ips}")
    private List<String> whitelistedIps;

    @Override
    protected void doFilterInternal(
            HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain) throws ServletException, IOException {

        String clientIp = getClientIp(request);

        if (!isWhitelisted(clientIp)) {
            response.sendError(HttpServletResponse.SC_FORBIDDEN,
                "접근이 거부되었습니다: IP " + clientIp);
            return;
        }

        filterChain.doFilter(request, response);
    }

    private boolean isWhitelisted(String ip) {
        return whitelistedIps.stream()
            .anyMatch(whitelistedIp ->
                matchesIpPattern(ip, whitelistedIp)
            );
    }

    private String getClientIp(HttpServletRequest request) {
        String ip = request.getHeader("X-Forwarded-For");
        if (ip == null || ip.isEmpty()) {
            ip = request.getRemoteAddr();
        }
        return ip;
    }
}

// CORS 화이트리스트
@Configuration
public class CorsConfig {

    @Bean
    public CorsFilter corsFilter() {
        UrlBasedCorsConfigurationSource source =
            new UrlBasedCorsConfigurationSource();

        CorsConfiguration config = new CorsConfiguration();
        config.setAllowCredentials(true);
        config.setAllowedOrigins(Arrays.asList(
            "https://trusted-domain1.com",
            "https://trusted-domain2.com"
        ));
        config.setAllowedMethods(Arrays.asList("GET", "POST", "PUT", "DELETE"));
        config.setAllowedHeaders(Arrays.asList("*"));

        source.registerCorsConfiguration("/api/**", config);
        return new CorsFilter(source);
    }
}
```

#### 9. OWASP API 보안 위험 점검

**OWASP API Security Top 10 (2023)**:

1. **API1:2023 - Broken Object Level Authorization (BOLA)**
   - 객체 수준 인가 취약점

2. **API2:2023 - Broken Authentication**
   - 인증 메커니즘 취약점

3. **API3:2023 - Broken Object Property Level Authorization**
   - 속성 수준 인가 취약점

4. **API4:2023 - Unrestricted Resource Consumption**
   - 무제한 리소스 소비

5. **API5:2023 - Broken Function Level Authorization**
   - 기능 수준 인가 취약점

6. **API6:2023 - Unrestricted Access to Sensitive Business Flows**
   - 민감한 비즈니스 플로우에 대한 무제한 접근

7. **API7:2023 - Server Side Request Forgery (SSRF)**
   - 서버 측 요청 위조

8. **API8:2023 - Security Misconfiguration**
   - 보안 설정 오류

9. **API9:2023 - Improper Inventory Management**
   - 부적절한 인벤토리 관리

10. **API10:2023 - Unsafe Consumption of APIs**
    - 안전하지 않은 API 사용

**점검 예제**:
```java
// BOLA 방어 - 리소스 소유권 확인
@RestController
@RequestMapping("/api/orders")
public class OrderController {

    @GetMapping("/{orderId}")
    public Order getOrder(
            @PathVariable Long orderId,
            @AuthenticationPrincipal User user) {

        Order order = orderService.findById(orderId);

        // BOLA 방어: 소유권 확인
        if (!order.getUserId().equals(user.getId())
                && !user.hasRole("ADMIN")) {
            throw new AccessDeniedException(
                "이 주문에 접근할 권한이 없습니다"
            );
        }

        return order;
    }
}
```

#### 10. API Gateway 사용

**목적**: API에 대한 중앙 집중식 접근 및 관리 지점

**구현 예제**:
```java
// Spring Cloud Gateway
@Configuration
public class ApiGatewayConfig {

    @Bean
    public RouteLocator customRouteLocator(
            RouteLocatorBuilder builder) {
        return builder.routes()
            .route("user_service", r -> r
                .path("/api/users/**")
                .filters(f -> f
                    .circuitBreaker(config -> config
                        .setName("userServiceCircuitBreaker")
                        .setFallbackUri("forward:/fallback/users")
                    )
                    .retry(config -> config
                        .setRetries(3)
                        .setStatuses(HttpStatus.SERVICE_UNAVAILABLE)
                    )
                    .requestRateLimiter(config -> config
                        .setRateLimiter(redisRateLimiter())
                    )
                )
                .uri("lb://user-service")
            )
            .route("order_service", r -> r
                .path("/api/orders/**")
                .filters(f -> f
                    .addRequestHeader("X-Gateway-Timestamp",
                        String.valueOf(System.currentTimeMillis()))
                )
                .uri("lb://order-service")
            )
            .build();
    }
}

// API Gateway 필터
@Component
public class LoggingGlobalFilter implements GlobalFilter, Ordered {

    @Override
    public Mono<Void> filter(
            ServerWebExchange exchange,
            GatewayFilterChain chain) {

        ServerHttpRequest request = exchange.getRequest();

        // 요청 로깅
        log.info("API Gateway 요청: {} {} from {}",
            request.getMethod(),
            request.getPath(),
            request.getRemoteAddress()
        );

        return chain.filter(exchange).then(Mono.fromRunnable(() -> {
            // 응답 로깅
            ServerHttpResponse response = exchange.getResponse();
            log.info("API Gateway 응답: {} with status {}",
                request.getPath(),
                response.getStatusCode()
            );
        }));
    }

    @Override
    public int getOrder() {
        return -1;  // 가장 먼저 실행
    }
}
```

#### 11. 오류 처리 (Error Handling)

**목적**: 민감한 정보를 노출하지 않는 적절한 오류 메시지 반환

**구현 예제**:
```java
// 전역 예외 핸들러
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(ResourceNotFoundException.class)
    public ResponseEntity<ErrorResponse> handleNotFound(
            ResourceNotFoundException ex) {

        ErrorResponse error = ErrorResponse.builder()
            .timestamp(LocalDateTime.now())
            .status(HttpStatus.NOT_FOUND.value())
            .error("Not Found")
            .message("요청한 리소스를 찾을 수 없습니다")
            .path(getCurrentRequest().getRequestURI())
            .build();

        // 민감한 정보(스택 트레이스 등) 제외
        return ResponseEntity.status(HttpStatus.NOT_FOUND).body(error);
    }

    @ExceptionHandler(AccessDeniedException.class)
    public ResponseEntity<ErrorResponse> handleAccessDenied(
            AccessDeniedException ex) {

        ErrorResponse error = ErrorResponse.builder()
            .timestamp(LocalDateTime.now())
            .status(HttpStatus.FORBIDDEN.value())
            .error("Forbidden")
            .message("이 리소스에 접근할 권한이 없습니다")
            .build();

        // 구체적인 권한 정보는 로그에만 기록
        log.warn("Access denied: {}", ex.getMessage());

        return ResponseEntity.status(HttpStatus.FORBIDDEN).body(error);
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleGeneral(Exception ex) {

        // 일반적인 오류 메시지만 반환
        ErrorResponse error = ErrorResponse.builder()
            .timestamp(LocalDateTime.now())
            .status(HttpStatus.INTERNAL_SERVER_ERROR.value())
            .error("Internal Server Error")
            .message("요청을 처리하는 중 오류가 발생했습니다")
            .build();

        // 상세 오류는 서버 로그에만 기록
        log.error("Unexpected error", ex);

        return ResponseEntity
            .status(HttpStatus.INTERNAL_SERVER_ERROR)
            .body(error);
    }
}

@Data
@Builder
class ErrorResponse {
    private LocalDateTime timestamp;
    private int status;
    private String error;
    private String message;
    private String path;
    // 스택 트레이스, 내부 오류 세부사항 등은 포함하지 않음
}
```

#### 12. 입력 검증 (Input Validation)

**목적**: SQL 인젝션, XSS 등의 공격 방지

**구현 예제**:
```java
// Bean Validation을 사용한 입력 검증
@RestController
@RequestMapping("/api/users")
@Validated
public class UserController {

    @PostMapping
    public User createUser(@Valid @RequestBody CreateUserRequest request) {
        return userService.create(request);
    }
}

@Data
class CreateUserRequest {

    @NotBlank(message = "사용자 이름은 필수입니다")
    @Size(min = 3, max = 50, message = "사용자 이름은 3-50자 사이여야 합니다")
    @Pattern(regexp = "^[a-zA-Z0-9_]+$",
        message = "사용자 이름은 영문자, 숫자, 언더스코어만 포함할 수 있습니다")
    private String username;

    @NotBlank(message = "이메일은 필수입니다")
    @Email(message = "유효한 이메일 형식이 아닙니다")
    private String email;

    @NotBlank(message = "비밀번호는 필수입니다")
    @Size(min = 8, max = 100, message = "비밀번호는 8-100자 사이여야 합니다")
    @Pattern(regexp = "^(?=.*[a-z])(?=.*[A-Z])(?=.*\\d)(?=.*[@$!%*?&])[A-Za-z\\d@$!%*?&]+$",
        message = "비밀번호는 대소문자, 숫자, 특수문자를 포함해야 합니다")
    private String password;

    @Min(value = 18, message = "나이는 18세 이상이어야 합니다")
    @Max(value = 150, message = "나이는 150세 이하여야 합니다")
    private Integer age;
}

// 커스텀 검증 애노테이션
@Target({ElementType.FIELD, ElementType.PARAMETER})
@Retention(RetentionPolicy.RUNTIME)
@Constraint(validatedBy = SafeHtmlValidator.class)
public @interface SafeHtml {
    String message() default "안전하지 않은 HTML이 포함되어 있습니다";
    Class<?>[] groups() default {};
    Class<? extends Payload>[] payload() default {};
}

public class SafeHtmlValidator
        implements ConstraintValidator<SafeHtml, String> {

    private final PolicyFactory policy = Sanitizers.FORMATTING
        .and(Sanitizers.LINKS);

    @Override
    public boolean isValid(String value,
            ConstraintValidatorContext context) {
        if (value == null) {
            return true;
        }

        // XSS 방어: HTML 새니타이징
        String sanitized = policy.sanitize(value);
        return value.equals(sanitized);
    }
}

// SQL 인젝션 방어 - Prepared Statement 사용
@Repository
public class UserRepository {

    @Autowired
    private JdbcTemplate jdbcTemplate;

    public User findByUsername(String username) {
        // 안전: Prepared Statement 사용
        String sql = "SELECT * FROM users WHERE username = ?";
        return jdbcTemplate.queryForObject(
            sql,
            new Object[]{username},
            new UserRowMapper()
        );
    }

    // 위험: 문자열 연결 (사용 금지)
    // String sql = "SELECT * FROM users WHERE username = '" + username + "'";
}
```

#### API 보안 체크리스트

| 보안 영역 | 구현 사항 | 우선순위 |
|----------|-----------|----------|
| **전송 보안** | HTTPS 강제, TLS 1.2+ | 🔴 높음 |
| **인증** | OAuth2, JWT, WebAuthn | 🔴 높음 |
| **인가** | RBAC, 리소스 수준 권한 | 🔴 높음 |
| **Rate Limiting** | Token Bucket, Sliding Window | 🟡 중간 |
| **입력 검증** | Bean Validation, Sanitization | 🔴 높음 |
| **API Gateway** | 중앙 집중식 보안 정책 | 🟡 중간 |
| **모니터링** | 로깅, 알림, 감사 추적 | 🟡 중간 |
| **버전 관리** | URL/헤더 기반 버전 관리 | 🟢 낮음 |
| **CORS** | 화이트리스트 기반 제어 | 🟡 중간 |
| **오류 처리** | 민감 정보 노출 방지 | 🔴 높음 |

#### 모니터링 및 감사

```java
// API 감사 로깅
@Aspect
@Component
public class ApiAuditAspect {

    @Around("@annotation(org.springframework.web.bind.annotation.RequestMapping)")
    public Object auditApiCall(ProceedingJoinPoint joinPoint)
            throws Throwable {

        ServletRequestAttributes attributes =
            (ServletRequestAttributes) RequestContextHolder
                .currentRequestAttributes();
        HttpServletRequest request = attributes.getRequest();

        AuditLog auditLog = AuditLog.builder()
            .timestamp(LocalDateTime.now())
            .method(request.getMethod())
            .path(request.getRequestURI())
            .ipAddress(request.getRemoteAddr())
            .userAgent(request.getHeader("User-Agent"))
            .apiKey(request.getHeader("X-API-Key"))
            .build();

        try {
            Object result = joinPoint.proceed();
            auditLog.setStatus("SUCCESS");
            return result;
        } catch (Exception e) {
            auditLog.setStatus("FAILURE");
            auditLog.setErrorMessage(e.getMessage());
            throw e;
        } finally {
            auditLogRepository.save(auditLog);
        }
    }
}
```

#### 결론

API 보안은 다층 방어(Defense in Depth) 전략을 통해 구현되어야 합니다. 각 보안 계층이 독립적으로 작동하면서도 함께 작동하여 강력한 보안 체계를 만들어야 합니다. 정기적인 보안 감사와 OWASP API Security Top 10에 대한 점검을 통해 지속적으로 보안 수준을 유지하고 개선해야 합니다.

## Big Data

### Hadoop

> [hadoop | TIL](/hadoop/README.md)

**개념**

Hadoop은 Apache Software Foundation에서 개발한 오픈소스 **소프트웨어 프레임워크**로, 컴퓨터 클러스터 전체에서 **대용량 데이터의 분산 저장** 및 **처리**를 위해 설계되었습니다. 단일 서버에서 수천 대의 머신으로 수평 확장이 가능하며, 고가용성, 장애 허용, 데이터 중복성을 제공합니다.

Hadoop은 데이터를 작은 청크(chunk)로 분할하고 여러 노드에서 병렬로 처리하는 원리를 기반으로 구축되어, 사용자가 대규모 데이터셋을 빠르고 효율적으로 처리하고 분석할 수 있도록 합니다. MapReduce 프로그래밍 모델을 기반으로 하며, 데이터를 독립적인 작업으로 나누어 처리 노드에 매핑한 후 결과를 집계하여 출력을 생성합니다.

#### Hadoop 핵심 구성요소

**1. HDFS (Hadoop Distributed File System)**

분산되고 확장 가능한 파일 시스템으로, 여러 노드에 데이터를 저장합니다.

**특징**:
- 데이터를 블록으로 분할하여 저장 (기본 128MB 또는 256MB)
- 각 블록을 여러 노드에 복제 (기본 3개 복제본)
- 장애 허용 및 고가용성 보장
- 대규모 순차 읽기/쓰기 작업에 최적화
- Write-Once-Read-Many (WORM) 모델

**아키텍처**:
```
┌──────────────────────────────────────────────────┐
│              NameNode (마스터)                    │
│  • 메타데이터 관리                                │
│  • 파일 시스템 네임스페이스                        │
│  • 블록 위치 정보                                 │
└────────────────┬─────────────────────────────────┘
                 │
     ┌───────────┼───────────┐
     │           │           │
     ↓           ↓           ↓
┌─────────┐ ┌─────────┐ ┌─────────┐
│DataNode1│ │DataNode2│ │DataNode3│
│ Block A │ │ Block A │ │ Block B │
│ Block B │ │ Block C │ │ Block C │
└─────────┘ └─────────┘ └─────────┘
```

**2. MapReduce**

Hadoop의 처리 엔진으로, 대규모 데이터 처리 작업을 관리 가능한 작은 작업으로 변환합니다.

**두 단계 처리**:
- **Map 단계**: 입력 데이터를 처리하고 필터링하여 중간 키-값 쌍 생성
- **Reduce 단계**: Map 출력을 집계하여 최종 결과 생성

**예제 - 단어 개수 세기 (Word Count)**:
```java
public class WordCount {

    // Map 함수
    public static class TokenizerMapper
            extends Mapper<Object, Text, Text, IntWritable> {

        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(Object key, Text value, Context context)
                throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                word.set(itr.nextToken());
                context.write(word, one); // (단어, 1) 출력
            }
        }
    }

    // Reduce 함수
    public static class IntSumReducer
            extends Reducer<Text, IntWritable, Text, IntWritable> {

        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result); // (단어, 총 개수) 출력
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "word count");
        job.setJarByClass(WordCount.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
```

**처리 흐름**:
```
입력: "hello world hello hadoop"

Map 단계:
  hello → 1
  world → 1
  hello → 1
  hadoop → 1

Shuffle & Sort:
  hello → [1, 1]
  world → [1]
  hadoop → [1]

Reduce 단계:
  hello → 2
  world → 1
  hadoop → 1
```

**3. YARN (Yet Another Resource Negotiator)**

클러스터 리소스 관리 계층으로, Hadoop 클러스터에서 실행되는 애플리케이션에 CPU, 메모리 등의 리소스를 할당합니다.

**구성요소**:
- **ResourceManager**: 클러스터 전체 리소스 관리
- **NodeManager**: 각 노드의 리소스 모니터링 및 관리
- **ApplicationMaster**: 개별 애플리케이션의 리소스 요청 및 작업 조정
- **Container**: 리소스의 논리적 묶음 (CPU, 메모리)

**YARN 아키텍처**:
```
┌────────────────────────────────────────┐
│      ResourceManager (마스터)          │
│  • 리소스 스케줄링                      │
│  • ApplicationMaster 관리              │
└──────────────┬─────────────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
    ↓          ↓          ↓
┌──────────┐ ┌──────────┐ ┌──────────┐
│NodeMgr 1 │ │NodeMgr 2 │ │NodeMgr 3 │
│Container │ │Container │ │Container │
└──────────┘ └──────────┘ └──────────┘
```

**4. Hadoop Common**

다른 Hadoop 모듈을 지원하는 공통 라이브러리 및 유틸리티 모음입니다.

#### Hadoop 생태계

Hadoop 주변에는 방대한 도구와 기술 생태계가 있습니다:

| 도구 | 역할 | 설명 |
|------|------|------|
| **Apache Spark** | 데이터 처리 엔진 | 인메모리 처리로 MapReduce보다 100배 빠름 |
| **Apache Hive** | 데이터 웨어하우스 | SQL 유사 쿼리 언어 (HiveQL)로 데이터 분석 |
| **Apache HBase** | NoSQL 데이터베이스 | HDFS 위에 구축된 컬럼 지향 데이터베이스 |
| **Apache Pig** | 데이터 플로우 언어 | 데이터 변환을 위한 고수준 스크립팅 언어 |
| **Apache Sqoop** | 데이터 전송 | RDBMS와 Hadoop 간 데이터 이동 |
| **Apache Flume** | 로그 수집 | 대규모 로그 데이터 수집 및 집계 |
| **Apache Kafka** | 메시징 시스템 | 실시간 데이터 스트리밍 |
| **Apache Oozie** | 워크플로우 스케줄러 | Hadoop 작업 조정 및 스케줄링 |

#### 실전 사용 사례

**1. 로그 분석**

```java
// Apache 웹 서버 로그 분석
public class LogAnalysisMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

    private Text ipAddress = new Text();
    private IntWritable one = new IntWritable(1);

    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
        // 로그 포맷: 192.168.1.1 - - [10/Oct/2023:13:55:36 -0700] "GET /index.html"
        String line = value.toString();
        String[] parts = line.split(" ");

        if (parts.length > 0) {
            ipAddress.set(parts[0]); // IP 주소 추출
            context.write(ipAddress, one);
        }
    }
}

public class LogAnalysisReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

    public void reduce(Text key, Iterable<IntWritable> values, Context context)
            throws IOException, InterruptedException {
        int count = 0;
        for (IntWritable val : values) {
            count += val.get();
        }
        context.write(key, new IntWritable(count)); // IP별 요청 수
    }
}
```

**2. Hive를 이용한 데이터 분석**

```sql
-- Hive 테이블 생성 (HDFS 데이터 매핑)
CREATE EXTERNAL TABLE web_logs (
    ip_address STRING,
    timestamp STRING,
    request STRING,
    status_code INT,
    bytes INT
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
LOCATION '/data/logs';

-- SQL 쿼리로 분석
SELECT
    ip_address,
    COUNT(*) as request_count,
    SUM(bytes) as total_bytes
FROM web_logs
WHERE status_code = 200
GROUP BY ip_address
ORDER BY request_count DESC
LIMIT 10;
```

#### Hadoop 장단점

**장점**:
- 수평 확장성: 노드 추가로 선형적 성능 향상
- 장애 허용: 자동 장애 감지 및 복구
- 비용 효율성: 저렴한 commodity 하드웨어 사용
- 유연성: 정형/비정형 데이터 모두 처리
- 오픈소스: 무료이며 활발한 커뮤니티

**단점**:
- 높은 지연시간: 배치 처리에 적합, 실시간 처리는 부적합
- 복잡성: 설정 및 관리가 복잡
- 소규모 파일 비효율: 많은 작은 파일 처리 시 성능 저하
- HDFS 제약: Random write 불가능

**적용 분야**:
- 검색 엔진 (웹 크롤링 및 인덱싱)
- 소셜 네트워크 (사용자 행동 분석)
- 금융 (사기 탐지, 리스크 분석)
- 빅데이터 분석 (로그 분석, BI)
- 과학 연구 (게놈 분석, 기후 모델링)

### MapReduce

> [MapReduce - Google Research](https://static.googleusercontent.com/media/research.google.com/en//archive/mapreduce-osdi04.pdf)

**개념**

MapReduce는 분산 컴퓨터 클러스터에서 대용량 데이터를 병렬로 처리하고 분석하기 위해 설계된 프로그래밍 모델 및 프레임워크입니다. Google에서 확장 가능하고 장애 허용적인 방식으로 대규모 계산 문제를 해결하기 위해 도입했으며, 빅데이터 처리 분야에서 큰 인기를 얻었고 Hadoop 생태계의 핵심 구성요소입니다.

MapReduce는 대규모 데이터 처리 작업을 작고 관리 가능한 하위 작업으로 나눈 다음, 클러스터의 여러 노드에서 이러한 하위 작업을 병렬로 처리한 후 결과를 집계하여 최종 출력을 생성합니다.

#### MapReduce 두 단계

**1. Map 단계 (매핑)**

입력 데이터를 고정 크기 청크로 나누고 각 청크를 처리합니다.

**동작 과정**:
```
입력 데이터
    ↓
Input Split (분할)
    ↓
Map 함수 적용 (사용자 정의)
    ↓
중간 Key-Value 쌍 생성
```

**특징**:
- 입력 데이터를 Input Split으로 분할하여 클러스터에 분산
- Map 함수가 각 Input Split을 처리하여 중간 키-값 쌍으로 변환
- Map 함수는 사용자 정의이며 해결하려는 특정 문제에 맞춤

**2. Reduce 단계 (집계)**

Map 단계에서 생성된 중간 키-값 쌍을 정렬하고 그룹화합니다.

**동작 과정**:
```
중간 Key-Value 쌍
    ↓
Shuffle & Sort (키별 정렬 및 그룹화)
    ↓
Reduce 함수 적용 (사용자 정의)
    ↓
최종 출력
```

**특징**:
- Map 단계의 중간 키-값 쌍을 키별로 정렬 및 그룹화
- Reduce 함수가 같은 키를 가진 값들을 집계하거나 결합
- Reduce 함수는 사용자 정의이며 문제에 특화

#### 상세 처리 흐름

```
┌─────────────────────────────────────────────────────┐
│               입력 데이터 (HDFS)                     │
└────────────────┬────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────┐
│            Input Splitting (분할)                    │
│  파일1 → Split1, Split2                              │
│  파일2 → Split3, Split4                              │
└────────────┬────────────────────────────────────────┘
             │
             ↓ (병렬 실행)
┌────────────┴────────────┬───────────┬───────────────┐
│                         │           │               │
↓                         ↓           ↓               ↓
┌─────────┐          ┌─────────┐  ┌─────────┐  ┌─────────┐
│ Map 1   │          │ Map 2   │  │ Map 3   │  │ Map 4   │
│(k1, v1) │          │(k2, v2) │  │(k3, v3) │  │(k4, v4) │
└────┬────┘          └────┬────┘  └────┬────┘  └────┬────┘
     │                    │            │            │
     └────────────────────┴────────────┴────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────┐
│         Shuffle & Sort (키별 그룹화)                 │
│  k1 → [v1, v5, v9]                                  │
│  k2 → [v2, v6]                                      │
│  k3 → [v3, v7, v8]                                  │
└────────────┬────────────────────────────────────────┘
             │
             ↓ (병렬 실행)
┌────────────┴───────────┬─────────────┐
│                        │             │
↓                        ↓             ↓
┌──────────┐        ┌──────────┐  ┌──────────┐
│ Reduce 1 │        │ Reduce 2 │  │ Reduce 3 │
│ (k1, r1) │        │ (k2, r2) │  │ (k3, r3) │
└────┬─────┘        └────┬─────┘  └────┬─────┘
     │                   │             │
     └───────────────────┴─────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────┐
│               최종 출력 (HDFS)                       │
└─────────────────────────────────────────────────────┘
```

#### MapReduce 자동 처리 기능

MapReduce 프레임워크가 자동으로 처리하는 복잡한 작업들:

- **데이터 파티셔닝**: 입력 데이터를 적절한 크기로 분할
- **작업 스케줄링**: Map/Reduce 작업을 노드에 할당
- **리소스 관리**: CPU, 메모리 등 클러스터 리소스 관리
- **에러 처리**: 장애 노드 감지 및 작업 재시도
- **노드 간 통신**: Shuffle 단계에서 데이터 전송
- **데이터 지역성**: 데이터가 있는 노드에서 작업 실행 (Network I/O 최소화)

**결과**: 개발자는 Map 및 Reduce 함수 작성에만 집중하면 되며, 병렬 및 분산 컴퓨팅의 복잡성을 관리할 필요가 없습니다.

#### 실전 예제

**예제 1: 판매 데이터 집계**

```java
// 입력: 날짜, 상품, 판매량
// 출력: 상품별 총 판매량

public class SalesMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
        // 입력: "2024-01-01,Laptop,5"
        String[] fields = value.toString().split(",");

        if (fields.length == 3) {
            String product = fields[1];
            int quantity = Integer.parseInt(fields[2]);

            context.write(new Text(product), new IntWritable(quantity));
            // 출력: ("Laptop", 5)
        }
    }
}

public class SalesReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

    public void reduce(Text key, Iterable<IntWritable> values, Context context)
            throws IOException, InterruptedException {
        int totalSales = 0;

        for (IntWritable val : values) {
            totalSales += val.get();
        }

        context.write(key, new IntWritable(totalSales));
        // 출력: ("Laptop", 150)
    }
}
```

**예제 2: 사용자 활동 분석**

```java
// 입력: user_id, action, timestamp
// 출력: 사용자별 액션 유형별 카운트

public class UserActivityMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
        String[] fields = value.toString().split(",");

        if (fields.length == 3) {
            String userId = fields[0];
            String action = fields[1];

            String compositeKey = userId + ":" + action;
            context.write(new Text(compositeKey), new IntWritable(1));
        }
    }
}
```

#### MapReduce 장단점

**장점**:
✓ **확장성**: 노드 추가로 처리 능력 선형 증가
✓ **장애 허용**: 자동 재시도 및 복구
✓ **단순성**: 복잡한 분산 처리를 Map/Reduce로 단순화
✓ **데이터 지역성**: 데이터 근처에서 처리하여 네트워크 I/O 최소화
✓ **비정형 데이터**: 정형/비정형 데이터 모두 처리 가능

**단점**:
✗ **높은 지연시간**: 배치 처리에 적합, 실시간 처리 부적합
✗ **반복 알고리즘 비효율**: 각 단계마다 디스크 I/O 발생
✗ **디스크 I/O**: 중간 결과를 디스크에 저장하여 느림
✗ **작은 데이터 오버헤드**: 소규모 데이터에는 비효율적

#### MapReduce vs Spark

| 특성 | MapReduce | Apache Spark |
|------|-----------|--------------|
| **처리 방식** | 디스크 기반 | 인메모리 기반 |
| **속도** | 느림 | 100배 빠름 (인메모리) |
| **반복 알고리즘** | 비효율적 | 효율적 |
| **실시간 처리** | 지원 안 함 | 지원 (Spark Streaming) |
| **사용성** | 복잡 (Java 코드) | 간단 (Scala, Python, SQL) |
| **적용** | 대규모 배치 처리 | 배치 + 실시간 + ML |

**MapReduce 대안의 등장 이유**:
- 높은 지연시간
- 반복 알고리즘 지원 부족
- 실시간 처리 불가능

→ **Apache Spark, Apache Flink** 등 차세대 프레임워크 등장

#### 언제 MapReduce를 사용할까?

**MapReduce가 적합한 경우**:
- 대규모 배치 처리 (TB/PB 단위)
- 단순한 집계 작업
- 한 번만 데이터를 읽는 작업
- 디스크 I/O가 메모리보다 저렴한 환경

**Spark가 더 나은 경우**:
- 반복적인 머신러닝 알고리즘
- 실시간 스트림 처리
- 대화형 데이터 분석
- 복잡한 데이터 파이프라인


### Data Lake (데이터 레이크)

- [Frequently Asked Questions About the Data Lakehouse | databricks](https://www.databricks.com/blog/2021/08/30/frequently-asked-questions-about-the-data-lakehouse.html)

**데이터 레이크(Data Lake)**는 모든 유형의 데이터를 저장하도록 설계된 저비용, 개방형, 내구성 있는 스토리지 시스템입니다. **테이블 데이터**, **텍스트**, **이미지**, **오디오**, **비디오**, **JSON**, **CSV** 등 다양한 형식의 데이터를 저장할 수 있습니다.

#### 주요 장점

1. **유연한 데이터 저장**
   - **구조화된 데이터**와 **비구조화된 데이터** 모두 저장 가능
   - 일반적으로 **Apache Parquet** 또는 **ORC** 같은 개방형 표준 포맷 사용
   - 대량의 데이터를 저비용으로 저장

2. **개방성 및 생태계**
   - 다양한 도구와 애플리케이션이 직접 데이터에 접근 가능
   - 특정 벤더 종속(Vendor Lock-in)을 피할 수 있음
   - 대규모 데이터 축적 가능

3. **주요 클라우드 제공자 솔루션**
   - **AWS S3** (Amazon Web Services)
   - **Azure Data Lake Storage (ADLS)** (Microsoft Azure)
   - **Google Cloud Storage (GCS)** (Google Cloud)

#### 한계점

데이터 레이크는 역사적으로 다음과 같은 문제점을 가지고 있습니다:
- **보안(Security)** - 세밀한 접근 제어 부족
- **품질(Quality)** - 데이터 일관성 및 신뢰성 문제
- **성능(Performance)** - SQL 쿼리 성능 제한

이러한 한계로 인해 조직은 종종 데이터의 일부를 데이터 웨어하우스로 이동하여 가치를 추출해야 합니다.

### Data Warehouse (데이터 웨어하우스)

- [Frequently Asked Questions About the Data Lakehouse | databricks](https://www.databricks.com/blog/2021/08/30/frequently-asked-questions-about-the-data-lakehouse.html)

**데이터 웨어하우스(Data Warehouse)**는 SQL 기반 분석 및 비즈니스 인텔리전스(BI)를 위해 구조화되거나 반구조화된 데이터를 저장, 관리, 분석하도록 설계된 전용 시스템입니다.

#### 주요 특징

1. **최적화된 성능**
   - 높은 성능, 동시성, 신뢰성에 최적화
   - 빠른 SQL 쿼리 처리
   - 동시 다중 사용자 지원

2. **데이터 타입 지원**
   - 주로 **구조화된 데이터** 지원
   - **비구조화된 데이터**에 대한 제한적 지원
     - **이미지**, **센서 데이터**, **문서**, **비디오** 등

3. **비용 및 제약사항**
   - 데이터 레이크에 비해 높은 비용
   - 오픈소스 라이브러리 및 도구 지원 제한
     - **TensorFlow**, **PyTorch**, **Python 기반 라이브러리** 등
   - 머신러닝 및 데이터 사이언스 사용 사례에 제약

#### 일반적인 사용 패턴

조직은 일반적으로 다음과 같이 데이터를 분리하여 관리합니다:
- **데이터 웨어하우스**: 빠른 동시 SQL 및 BI 사용 사례를 위한 핵심 비즈니스 데이터
- **데이터 레이크**: 비구조화된 데이터를 포함한 대규모 데이터셋

### Data Lakehouse (데이터 레이크하우스)

- [Frequently Asked Questions About the Data Lakehouse | databricks](https://www.databricks.com/blog/2021/08/30/frequently-asked-questions-about-the-data-lakehouse.html)

**데이터 레이크하우스(Data Lakehouse)**는 **데이터 레이크**와 **데이터 웨어하우스**의 장점을 결합한 현대적인 데이터 아키텍처입니다. 데이터 레이크에 저장된 대량의 데이터에 대해 효율적이고 안전한 AI 및 비즈니스 인텔리전스(BI)를 직접 수행할 수 있도록 합니다.

#### 핵심 목표

1. **데이터 레이크의 문제 해결**
   - 보안(Security) 강화
   - 품질(Quality) 개선
   - 성능(Performance) 향상

2. **데이터 웨어하우스의 기능 제공**
   - SQL 쿼리 최적화된 성능
   - BI 스타일 리포팅

#### 주요 기능

```
┌─────────────────────────────────────────┐
│      Data Lakehouse 아키텍처             │
├─────────────────────────────────────────┤
│  AI/ML       BI/Analytics    Real-time  │
│  ┌─────┐     ┌─────┐        ┌─────┐    │
│  │Python│    │ SQL │        │Stream│    │
│  └─────┘     └─────┘        └─────┘    │
├─────────────────────────────────────────┤
│  메타데이터 레이어 (Delta Lake/Iceberg) │
│  - ACID 트랜잭션                        │
│  - 스키마 관리                          │
│  - 타임 트래블                          │
├─────────────────────────────────────────┤
│  스토리지 (S3, ADLS, GCS)               │
│  - Parquet/ORC 파일                     │
│  - 모든 데이터 타입 지원                │
└─────────────────────────────────────────┘
```

**1. 통합 스토리지**
- 모든 데이터 타입을 한 곳에 저장
  - 구조화된 데이터 (Structured)
  - 반구조화된 데이터 (Semi-structured)
  - 비구조화된 데이터 (Unstructured)
- 데이터 이동 불필요

**2. ACID 트랜잭션**
- 데이터 일관성 보장
- 동시 읽기/쓰기 지원
- 데이터 품질 향상

**3. 세밀한 데이터 보안**
- 행/열 수준 접근 제어
- 역할 기반 권한 관리
- 감사 로깅

**4. 저비용 업데이트 및 삭제**
- 효율적인 데이터 수정
- 증분 업데이트 지원
- 스토리지 비용 최적화

**5. SQL 및 데이터 사이언스 네이티브 지원**
- 완전한 SQL 지원
- Python, R, Scala 네이티브 통합
- TensorFlow, PyTorch 등 ML 프레임워크 지원

#### 오픈소스 기술

주요 데이터 레이크하우스 기술:

| 기술 | 특징 | 주요 기능 |
|------|------|-----------|
| **Delta Lake** | Databricks 주도 | - ACID 트랜잭션<br>- 타임 트래블<br>- 스키마 진화 |
| **Apache Hudi** | Uber 개발 | - 증분 처리<br>- 업서트 최적화<br>- 스트리밍 통합 |
| **Apache Iceberg** | Netflix 개발 | - 스냅샷 격리<br>- 숨겨진 파티셔닝<br>- 스키마 진화 |

#### 주요 벤더

- **Databricks** - Delta Lake 기반 통합 플랫폼
- **AWS** - Lake Formation, Redshift Spectrum
- **Dremio** - 데이터 가상화 및 쿼리 가속
- **Starburst** - Trino 기반 분산 쿼리 엔진

#### 장점

1. **단일 통합 시스템**
   - 모든 데이터 타입 커버
   - BI부터 AI까지 광범위한 분석 사용 사례

2. **개방형 표준**
   - 벤더 종속 회피
   - 다양한 도구 생태계 활용

3. **비용 효율성**
   - 데이터 이동 제거
   - 저비용 스토리지 활용
   - 중복 스토리지 불필요

4. **성능 및 확장성**
   - 대규모 데이터 처리
   - SQL 쿼리 최적화
   - 동시 사용자 지원

#### 사용 사례

```java
// Delta Lake 예제 (Spark 기반)
import org.apache.spark.sql.SparkSession;
import io.delta.tables.*;

public class DataLakehouseExample {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
            .appName("Lakehouse Example")
            .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .getOrCreate();

        // 1. 데이터 레이크하우스에 데이터 쓰기 (ACID 보장)
        spark.range(0, 10000)
            .write()
            .format("delta")
            .mode("overwrite")
            .save("s3://my-lakehouse/orders");

        // 2. ACID 트랜잭션으로 업데이트
        DeltaTable orders = DeltaTable.forPath(spark,
            "s3://my-lakehouse/orders");

        orders.update(
            "id < 100",
            Map.of("status", lit("processed"))
        );

        // 3. 타임 트래블 - 과거 버전 조회
        spark.read()
            .format("delta")
            .option("versionAsOf", "2")  // 버전 2로 롤백
            .load("s3://my-lakehouse/orders")
            .show();

        // 4. SQL로 분석 (BI 사용 사례)
        spark.sql("""
            SELECT
                date_trunc('day', order_date) as day,
                count(*) as order_count,
                sum(amount) as total_amount
            FROM delta.`s3://my-lakehouse/orders`
            WHERE status = 'completed'
            GROUP BY day
            ORDER BY day DESC
        """).show();

        // 5. 머신러닝 데이터 준비 (AI 사용 사례)
        Dataset<Row> mlData = spark.read()
            .format("delta")
            .load("s3://my-lakehouse/orders")
            .selectExpr("features", "label");

        // TensorFlow, PyTorch 등과 통합 가능
        spark.stop();
    }
}
```

#### 아키텍처 비교

| 특징 | Data Lake | Data Warehouse | Data Lakehouse |
|------|-----------|----------------|----------------|
| **데이터 타입** | 모든 타입 | 주로 구조화 | 모든 타입 |
| **비용** | 낮음 | 높음 | 낮음~중간 |
| **SQL 성능** | 낮음 | 높음 | 높음 |
| **ACID** | ❌ | ✅ | ✅ |
| **ML/AI 지원** | ✅ | ❌ | ✅ |
| **BI 지원** | 제한적 | ✅ | ✅ |
| **스키마** | Schema-on-read | Schema-on-write | 유연함 |
| **데이터 품질** | 낮음 | 높음 | 높음 |

#### 결론

데이터 레이크하우스는 데이터 레이크의 유연성과 저비용, 데이터 웨어하우스의 성능과 신뢰성을 결합하여 조직이 단일 플랫폼에서 모든 분석 워크로드(BI, AI, 실시간 분석)를 수행할 수 있게 합니다.


## A/B 테스트 (A/B Testing)

A/B 테스트(A/B Testing)는 분할 테스트(Split Testing) 또는 버킷 테스트(Bucket Testing)로도 알려져 있으며, 제품, 기능, 콘텐츠의 둘 이상의 변형(variant)을 비교하여 특정 지표를 기준으로 어느 것이 더 나은 성과를 내는지 판단하는 통제된 실험 방법이다. 웹사이트 및 앱 디자인, 온라인 마케팅, 광고, 사용자 경험 최적화 등에 널리 사용된다.

**A/B 테스트의 작동 원리**:

A/B 테스트에서는 대상 사용자를 무작위로 두 개 이상의 그룹으로 나누고, 각 그룹에 서로 다른 변형(A, B 또는 그 이상)을 제공한다. 변형은 디자인, 헤드라인, 행동 유도 버튼(CTA), 페이지 레이아웃, 마케팅 메시지 등 사용자 행동에 영향을 줄 수 있는 모든 요소가 될 수 있다. 사용자의 행동이나 반응을 측정하여 그룹 간 비교함으로써 전환율, 클릭률, 사용자 참여도, 페이지 체류 시간 등 사전 정의된 지표에서 최상의 결과를 산출하는 변형을 결정한다.

**A/B 테스트 프로세스**:

**1. 목표 정의 (Define the Objective)**:
- 테스트를 통해 개선하고자 하는 구체적인 목표나 지표를 결정
- 예: 전환율 5% 향상, 이탈률 10% 감소

**2. 가설 수립 (Develop Hypotheses)**:
- 사용자 선호도와 행동에 대한 가정과 데이터를 기반으로 대안 버전 생성
- 예: "버튼 색상을 빨간색에서 녹색으로 변경하면 클릭률이 증가할 것이다"

**3. 테스트 변형 생성 (Create Test Variants)**:
- 가설에 따라 요소를 수정하여 테스트할 다양한 버전 생성
- 변형 A (Control): 기존 버전
- 변형 B (Treatment): 새로운 버전

**4. 사용자 무작위 분할 (Randomly Split the Audience)**:
- 사용자를 무작위로 다른 그룹에 할당
- 각 그룹이 하나의 변형을 받도록 보장
- 일반적으로 50/50 또는 다른 비율로 분할

**5. 테스트 실행 (Run the Test)**:
- 사전 정의된 기간 동안 또는 통계적으로 유의미한 샘플 크기에 도달할 때까지 실험 수행
- 외부 변수 최소화, 동일한 조건 유지

**6. 결과 분석 (Analyze the Results)**:
- 정의된 지표를 기반으로 각 변형의 성과 비교
- 통계적 유의성 검증
- 우승 변형 결정

**A/B 테스트 주요 지표**:

| 지표 | 설명 | 계산 방법 |
|------|------|----------|
| **전환율 (Conversion Rate)** | 원하는 행동을 수행한 사용자 비율 | (전환 수 / 총 방문자 수) × 100% |
| **클릭률 (Click-Through Rate, CTR)** | 링크나 버튼을 클릭한 사용자 비율 | (클릭 수 / 노출 수) × 100% |
| **이탈률 (Bounce Rate)** | 한 페이지만 보고 떠난 사용자 비율 | (단일 페이지 세션 / 총 세션) × 100% |
| **평균 세션 시간** | 사용자가 사이트에 머문 평균 시간 | 총 세션 시간 / 총 세션 수 |
| **페이지당 조회수** | 세션당 평균 페이지 조회수 | 총 페이지뷰 / 총 세션 수 |

**A/B 테스트 구현 예제 (Java)**:

```java
// Variant.java
public class Variant {
    private String id;
    private String name;
    private String description;
    private int impressions;
    private int conversions;
    private int clicks;

    public Variant(String id, String name, String description) {
        this.id = id;
        this.name = name;
        this.description = description;
        this.impressions = 0;
        this.conversions = 0;
        this.clicks = 0;
    }

    public synchronized void recordImpression() {
        impressions++;
    }

    public synchronized void recordClick() {
        clicks++;
    }

    public synchronized void recordConversion() {
        conversions++;
    }

    public double getConversionRate() {
        return impressions > 0 ? (double) conversions / impressions : 0.0;
    }

    public double getClickThroughRate() {
        return impressions > 0 ? (double) clicks / impressions : 0.0;
    }

    // Getters
    public String getId() { return id; }
    public String getName() { return name; }
    public int getImpressions() { return impressions; }
    public int getConversions() { return conversions; }
    public int getClicks() { return clicks; }
}

// ABTest.java
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class ABTest {
    private String testId;
    private String testName;
    private List<Variant> variants;
    private Map<String, String> userAssignments; // userId -> variantId
    private Random random;
    private boolean isActive;

    public ABTest(String testId, String testName) {
        this.testId = testId;
        this.testName = testName;
        this.variants = new ArrayList<>();
        this.userAssignments = new ConcurrentHashMap<>();
        this.random = new Random();
        this.isActive = true;
    }

    public void addVariant(Variant variant) {
        variants.add(variant);
    }

    // 사용자를 변형에 할당 (Consistent Hashing 사용)
    public Variant assignUser(String userId) {
        // 이미 할당된 사용자는 동일한 변형 반환
        if (userAssignments.containsKey(userId)) {
            String variantId = userAssignments.get(userId);
            return variants.stream()
                .filter(v -> v.getId().equals(variantId))
                .findFirst()
                .orElse(null);
        }

        // 새로운 사용자는 무작위로 할당
        Variant assignedVariant = variants.get(random.nextInt(variants.size()));
        userAssignments.put(userId, assignedVariant.getId());

        return assignedVariant;
    }

    // 해시 기반 일관된 할당 (동일한 userId는 항상 동일한 변형)
    public Variant assignUserConsistently(String userId) {
        if (userAssignments.containsKey(userId)) {
            String variantId = userAssignments.get(userId);
            return variants.stream()
                .filter(v -> v.getId().equals(variantId))
                .findFirst()
                .orElse(null);
        }

        // userId의 해시값을 사용하여 일관된 할당
        int hash = Math.abs(userId.hashCode());
        int index = hash % variants.size();
        Variant assignedVariant = variants.get(index);
        userAssignments.put(userId, assignedVariant.getId());

        return assignedVariant;
    }

    public void recordImpression(String userId) {
        Variant variant = assignUser(userId);
        if (variant != null && isActive) {
            variant.recordImpression();
        }
    }

    public void recordClick(String userId) {
        String variantId = userAssignments.get(userId);
        if (variantId != null && isActive) {
            variants.stream()
                .filter(v -> v.getId().equals(variantId))
                .findFirst()
                .ifPresent(Variant::recordClick);
        }
    }

    public void recordConversion(String userId) {
        String variantId = userAssignments.get(userId);
        if (variantId != null && isActive) {
            variants.stream()
                .filter(v -> v.getId().equals(variantId))
                .findFirst()
                .ifPresent(Variant::recordConversion);
        }
    }

    public ABTestResult getTestResults() {
        return new ABTestResult(testId, testName, variants);
    }

    public void stopTest() {
        isActive = false;
    }

    public boolean isActive() {
        return isActive;
    }

    public List<Variant> getVariants() {
        return new ArrayList<>(variants);
    }
}

// ABTestResult.java
import java.util.*;

public class ABTestResult {
    private String testId;
    private String testName;
    private List<Variant> variants;

    public ABTestResult(String testId, String testName, List<Variant> variants) {
        this.testId = testId;
        this.testName = testName;
        this.variants = new ArrayList<>(variants);
    }

    // 통계적 유의성 계산 (Z-test)
    public boolean isStatisticallySignificant(Variant variantA, Variant variantB) {
        double p1 = variantA.getConversionRate();
        double p2 = variantB.getConversionRate();
        int n1 = variantA.getImpressions();
        int n2 = variantB.getImpressions();

        // 샘플 크기가 너무 작으면 유의성 검증 불가
        if (n1 < 30 || n2 < 30) {
            return false;
        }

        // 풀링된 비율
        double pooledP = ((double) (variantA.getConversions() + variantB.getConversions())) / (n1 + n2);

        // 표준 오차
        double se = Math.sqrt(pooledP * (1 - pooledP) * (1.0/n1 + 1.0/n2));

        // Z-score 계산
        double zScore = Math.abs(p1 - p2) / se;

        // 95% 신뢰 수준 (Z > 1.96)
        return zScore > 1.96;
    }

    // 우승 변형 결정
    public Variant getWinner() {
        return variants.stream()
            .max(Comparator.comparingDouble(Variant::getConversionRate))
            .orElse(null);
    }

    // 결과 리포트 출력
    public void printReport() {
        System.out.println("\n=== A/B Test Results ===");
        System.out.println("Test ID: " + testId);
        System.out.println("Test Name: " + testName);
        System.out.println("\nVariant Performance:");

        for (Variant variant : variants) {
            System.out.println("\n" + variant.getName() + " (ID: " + variant.getId() + ")");
            System.out.println("  Impressions: " + variant.getImpressions());
            System.out.println("  Clicks: " + variant.getClicks());
            System.out.println("  Conversions: " + variant.getConversions());
            System.out.println("  CTR: " + String.format("%.2f%%", variant.getClickThroughRate() * 100));
            System.out.println("  Conversion Rate: " + String.format("%.2f%%", variant.getConversionRate() * 100));
        }

        if (variants.size() >= 2) {
            boolean significant = isStatisticallySignificant(variants.get(0), variants.get(1));
            System.out.println("\nStatistically Significant: " + (significant ? "YES" : "NO"));

            Variant winner = getWinner();
            if (winner != null && significant) {
                System.out.println("Winner: " + winner.getName());
            }
        }
    }

    // 신뢰 구간 계산
    public double[] getConfidenceInterval(Variant variant, double confidenceLevel) {
        double p = variant.getConversionRate();
        int n = variant.getImpressions();

        // Z-score (95% 신뢰 수준 = 1.96)
        double z = 1.96;
        if (confidenceLevel == 0.99) {
            z = 2.576;
        } else if (confidenceLevel == 0.90) {
            z = 1.645;
        }

        double se = Math.sqrt((p * (1 - p)) / n);
        double marginOfError = z * se;

        return new double[] {
            Math.max(0, p - marginOfError),
            Math.min(1, p + marginOfError)
        };
    }
}

// ABTestService.java
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class ABTestService {
    private Map<String, ABTest> activeTests;

    public ABTestService() {
        this.activeTests = new ConcurrentHashMap<>();
    }

    public ABTest createTest(String testId, String testName) {
        ABTest test = new ABTest(testId, testName);
        activeTests.put(testId, test);
        return test;
    }

    public ABTest getTest(String testId) {
        return activeTests.get(testId);
    }

    public void stopTest(String testId) {
        ABTest test = activeTests.get(testId);
        if (test != null) {
            test.stopTest();
        }
    }

    public List<ABTest> getActiveTests() {
        return activeTests.values().stream()
            .filter(ABTest::isActive)
            .toList();
    }
}

// ABTestExample.java
public class ABTestExample {
    public static void main(String[] args) {
        ABTestService service = new ABTestService();

        // A/B 테스트 생성: 체크아웃 버튼 색상 테스트
        ABTest checkoutButtonTest = service.createTest(
            "checkout-button-color",
            "Checkout Button Color Test"
        );

        // 변형 추가
        Variant variantA = new Variant("variant-a", "Control (Blue Button)", "Original blue checkout button");
        Variant variantB = new Variant("variant-b", "Treatment (Green Button)", "New green checkout button");

        checkoutButtonTest.addVariant(variantA);
        checkoutButtonTest.addVariant(variantB);

        // 사용자 시뮬레이션
        System.out.println("=== Simulating User Interactions ===");
        Random random = new Random();

        for (int i = 1; i <= 1000; i++) {
            String userId = "user-" + i;

            // 사용자에게 페이지 노출
            checkoutButtonTest.recordImpression(userId);

            // 70% 확률로 클릭
            if (random.nextDouble() < 0.7) {
                checkoutButtonTest.recordClick(userId);

                // 변형 B는 10% 더 높은 전환율
                Variant assignedVariant = checkoutButtonTest.assignUser(userId);
                double conversionProbability = assignedVariant.getId().equals("variant-a") ? 0.15 : 0.25;

                if (random.nextDouble() < conversionProbability) {
                    checkoutButtonTest.recordConversion(userId);
                }
            }
        }

        // 결과 분석
        ABTestResult results = checkoutButtonTest.getTestResults();
        results.printReport();

        // 신뢰 구간 출력
        System.out.println("\n=== Confidence Intervals (95%) ===");
        for (Variant variant : checkoutButtonTest.getVariants()) {
            double[] ci = results.getConfidenceInterval(variant, 0.95);
            System.out.println(variant.getName() + ": [" +
                String.format("%.2f%%", ci[0] * 100) + ", " +
                String.format("%.2f%%", ci[1] * 100) + "]");
        }

        // 테스트 중지
        checkoutButtonTest.stopTest();
    }
}
```

**A/B 테스트 고급 기법**:

```java
// MultiVariateTest.java (다변량 테스트)
public class MultiVariateTest {
    private String testId;
    private Map<String, List<String>> factors; // factor -> [option1, option2, ...]
    private Map<String, Variant> variants;

    public MultiVariateTest(String testId) {
        this.testId = testId;
        this.factors = new HashMap<>();
        this.variants = new HashMap<>();
    }

    // 팩터 추가 (예: 버튼 색상, 텍스트, 위치)
    public void addFactor(String factorName, List<String> options) {
        factors.put(factorName, options);
        generateVariants();
    }

    // 모든 조합의 변형 생성
    private void generateVariants() {
        variants.clear();
        List<String> factorNames = new ArrayList<>(factors.keySet());
        generateVariantsRecursive(factorNames, 0, new HashMap<>());
    }

    private void generateVariantsRecursive(List<String> factorNames, int index,
                                          Map<String, String> currentCombination) {
        if (index == factorNames.size()) {
            String variantId = generateVariantId(currentCombination);
            String variantName = generateVariantName(currentCombination);
            variants.put(variantId, new Variant(variantId, variantName, ""));
            return;
        }

        String factorName = factorNames.get(index);
        List<String> options = factors.get(factorName);

        for (String option : options) {
            currentCombination.put(factorName, option);
            generateVariantsRecursive(factorNames, index + 1, new HashMap<>(currentCombination));
        }
    }

    private String generateVariantId(Map<String, String> combination) {
        return String.join("-", combination.values());
    }

    private String generateVariantName(Map<String, String> combination) {
        StringBuilder sb = new StringBuilder();
        combination.forEach((factor, option) ->
            sb.append(factor).append(":").append(option).append(", ")
        );
        return sb.toString();
    }
}

// BanditAlgorithm.java (Multi-Armed Bandit)
public class BanditAlgorithm {
    private List<Variant> variants;
    private double epsilon; // 탐색 비율

    public BanditAlgorithm(List<Variant> variants, double epsilon) {
        this.variants = variants;
        this.epsilon = epsilon;
    }

    // Epsilon-Greedy 알고리즘
    public Variant selectVariant() {
        Random random = new Random();

        // epsilon 확률로 무작위 탐색
        if (random.nextDouble() < epsilon) {
            return variants.get(random.nextInt(variants.size()));
        }

        // (1-epsilon) 확률로 최고 성과 변형 활용
        return variants.stream()
            .max(Comparator.comparingDouble(Variant::getConversionRate))
            .orElse(variants.get(0));
    }

    // Thompson Sampling (베이지안 접근)
    public Variant selectVariantThompsonSampling() {
        Random random = new Random();
        double maxSample = Double.NEGATIVE_INFINITY;
        Variant bestVariant = null;

        for (Variant variant : variants) {
            int successes = variant.getConversions();
            int failures = variant.getImpressions() - variant.getConversions();

            // 베타 분포에서 샘플링
            double sample = sampleBeta(successes + 1, failures + 1, random);

            if (sample > maxSample) {
                maxSample = sample;
                bestVariant = variant;
            }
        }

        return bestVariant;
    }

    private double sampleBeta(int alpha, int beta, Random random) {
        // 간단한 베타 분포 샘플링 (실제로는 Apache Commons Math 사용 권장)
        double x = sampleGamma(alpha, 1, random);
        double y = sampleGamma(beta, 1, random);
        return x / (x + y);
    }

    private double sampleGamma(double shape, double scale, Random random) {
        // 간단한 감마 분포 샘플링
        if (shape < 1) {
            return sampleGamma(shape + 1, scale, random) * Math.pow(random.nextDouble(), 1.0 / shape);
        }

        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.sqrt(9.0 * d);

        while (true) {
            double x, v;
            do {
                x = random.nextGaussian();
                v = 1.0 + c * x;
            } while (v <= 0);

            v = v * v * v;
            x = x * x;

            double u = random.nextDouble();
            if (u < 1 - 0.0331 * x * x) {
                return scale * d * v;
            }

            if (Math.log(u) < 0.5 * x + d * (1 - v + Math.log(v))) {
                return scale * d * v;
            }
        }
    }
}
```

**A/B 테스트 모범 사례**:

**1. 충분한 샘플 크기**:
- 통계적 유의성을 확보하기 위해 최소 1,000명 이상의 사용자
- 전환율이 낮을수록 더 많은 샘플 필요

**2. 테스트 기간**:
- 최소 1-2주 동안 실행하여 요일 효과 제거
- 계절성이나 특별 이벤트 고려

**3. 한 번에 하나씩 테스트**:
- 여러 요소를 동시에 변경하지 말 것
- 명확한 인과관계 파악

**4. 사전 가설 수립**:
- 테스트 전에 명확한 가설 정의
- 데이터 기반 의사결정

**5. 일관된 할당**:
- 동일한 사용자는 항상 동일한 변형 제공
- 쿠키나 사용자 ID 기반 할당

**A/B 테스트 사용 사례**:

| 분야 | 테스트 요소 | 측정 지표 |
|------|------------|----------|
| **이커머스** | 체크아웃 버튼 색상, 제품 이미지 | 전환율, 평균 주문 금액 |
| **SaaS** | 가격 페이지 레이아웃, 무료 체험 기간 | 가입률, 유료 전환율 |
| **콘텐츠** | 헤드라인, 썸네일 이미지 | 클릭률, 페이지 체류 시간 |
| **모바일 앱** | 온보딩 플로우, 푸시 알림 문구 | 활성 사용자, 리텐션 |
| **이메일** | 제목, 발송 시간, CTA 문구 | 오픈율, 클릭률 |

A/B 테스트는 데이터 기반 의사결정, 사용자 경험 최적화, 매출 또는 전환율 증가, 마케팅 전략 개선에 매우 유용하다. 그러나 신중한 계획, 명확한 목표 설정, 적절한 통계 분석이 필요하며, 결과의 정확성과 신뢰성을 보장해야 한다.

## Actor Model

- [Actor Model](/actormodel/README-kr.md)
- Actor Model은 **동시성 컴퓨팅**을 위한 수학적 모델이자 설계 패턴입니다. 1973년 Carl Hewitt에 의해 제안된 이 모델은 "Actor"라는 기본 단위를 통해 병렬 및 분산 시스템을 구축합니다.
- Reactor/Proactor: 저수준 I/O 이벤트 처리 메커니즘
- Actor Model: 고수준 동시성 및 분산 시스템 패턴
- 관계: Actor Model은 내부적으로 Reactor/Proactor를 사용하여 I/O를 처리하지만, 개발자에게는 메시지 기반의 고수준 추상화를 제공
- 장점:
  - Reactor/Proactor의 효율적인 I/O 처리
  - Actor Model의 간단한 동시성 관리
  - 분산 시스템으로의 자연스러운 확장
- Actor Model을 사용하면 Reactor/Proactor의 복잡한 콜백 지옥에서 벗어나 메시지 기반의 직관적인 코드를 작성할 수 있습니다.

## Reactor vs Proactor

**Reactor**와 **Proactor** 패턴은 동시성 시스템에서 비동기 입출력(I/O) 작업을 처리하기 위한 디자인 패턴이다. 이들은 다수의 클라이언트 요청이나 리소스 집약적인 작업과 같은 여러 이벤트와 I/O 작업을 관리하는 데 도움을 준다. 두 패턴의 주요 차이점은 I/O 작업을 처리하는 방식이다.

**패턴 비교**:

| 특징 | Reactor 패턴 | Proactor 패턴 |
|------|-------------|--------------|
| **I/O 모델** | 동기 논블로킹 I/O | 비동기 I/O |
| **이벤트 처리** | I/O 준비 완료 이벤트 | I/O 완료 이벤트 |
| **처리 방식** | 애플리케이션이 직접 I/O 수행 | OS가 I/O 수행, 결과만 전달 |
| **멀티스레딩** | 일반적으로 필요 | 선택적 |
| **구현 복잡도** | 상대적으로 간단 | 상대적으로 복잡 |
| **성능** | 많은 동시 연결 시 제한적 | 높은 동시성 지원 |
| **Java API** | java.nio (NIO) | java.nio.channels (AIO) |

**Reactor 패턴의 동작 방식**:

```
1. 애플리케이션이 I/O 작업 등록
2. Reactor가 이벤트 루프에서 I/O 준비 상태 대기
3. I/O가 준비되면 이벤트 발생
4. 애플리케이션의 핸들러가 직접 I/O 수행
5. 결과 처리
```

**장점**:
- 이해하고 구현하기 쉬움
- 동기식 I/O 작업으로 디버깅 용이
- 널리 사용되는 패턴 (Node.js, Redis, Nginx)

**단점**:
- I/O 작업 수행 시 애플리케이션 코드가 블록됨
- 많은 동시 I/O 작업 처리 시 성능 저하
- 긴 시간이 걸리는 I/O 작업에 취약

**Proactor 패턴의 동작 방식**:

```
1. 애플리케이션이 비동기 I/O 요청 제출
2. OS가 백그라운드에서 I/O 작업 수행
3. I/O 작업 완료 시 완료 이벤트 발생
4. 완료 핸들러(Completion Handler)가 결과 처리
5. 애플리케이션은 블록되지 않음
```

**장점**:
- 완전한 비동기 I/O로 애플리케이션 블록 없음
- 높은 동시성 처리 가능
- 멀티스레딩에 대한 의존도 낮음
- 높은 처리량과 확장성

**단점**:
- 구현이 복잡함
- 비동기 프로그래밍 모델의 학습 곡선
- OS 레벨 지원 필요 (Windows IOCP, Linux AIO)
- 디버깅과 에러 처리가 어려움

**Reactor 패턴 구현 (Java NIO)**:

```java
// ReactorServer.java
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.*;
import java.util.*;

public class ReactorServer {
    private final Selector selector;
    private final ServerSocketChannel serverChannel;

    public ReactorServer(int port) throws IOException {
        selector = Selector.open();
        serverChannel = ServerSocketChannel.open();
        serverChannel.bind(new InetSocketAddress("localhost", port));
        serverChannel.configureBlocking(false);
        serverChannel.register(selector, SelectionKey.OP_ACCEPT);

        System.out.println("Reactor Server started on port " + port);
    }

    public void run() throws IOException {
        while (true) {
            // I/O 준비 완료 이벤트 대기 (블로킹)
            selector.select();

            // 준비된 이벤트 처리
            Set<SelectionKey> selectedKeys = selector.selectedKeys();
            Iterator<SelectionKey> iterator = selectedKeys.iterator();

            while (iterator.hasNext()) {
                SelectionKey key = iterator.next();
                iterator.remove();

                try {
                    if (!key.isValid()) {
                        continue;
                    }

                    if (key.isAcceptable()) {
                        handleAccept(key);
                    } else if (key.isReadable()) {
                        handleRead(key);
                    } else if (key.isWritable()) {
                        handleWrite(key);
                    }
                } catch (IOException e) {
                    System.err.println("Error handling key: " + e.getMessage());
                    key.cancel();
                    key.channel().close();
                }
            }
        }
    }

    // 새 연결 수락
    private void handleAccept(SelectionKey key) throws IOException {
        ServerSocketChannel serverChannel = (ServerSocketChannel) key.channel();
        SocketChannel clientChannel = serverChannel.accept();

        if (clientChannel != null) {
            clientChannel.configureBlocking(false);
            clientChannel.register(selector, SelectionKey.OP_READ);
            System.out.println("Accepted connection from: " + clientChannel.getRemoteAddress());
        }
    }

    // 데이터 읽기 (동기적으로 수행)
    private void handleRead(SelectionKey key) throws IOException {
        SocketChannel clientChannel = (SocketChannel) key.channel();
        ByteBuffer buffer = ByteBuffer.allocate(1024);

        int bytesRead = clientChannel.read(buffer);

        if (bytesRead == -1) {
            System.out.println("Client closed connection: " + clientChannel.getRemoteAddress());
            clientChannel.close();
            key.cancel();
            return;
        }

        if (bytesRead > 0) {
            buffer.flip();
            byte[] data = new byte[buffer.remaining()];
            buffer.get(data);
            String message = new String(data).trim();

            System.out.println("Received: " + message + " from " + clientChannel.getRemoteAddress());

            // 응답 준비
            String response = "Echo: " + message;
            key.attach(ByteBuffer.wrap(response.getBytes()));
            key.interestOps(SelectionKey.OP_WRITE);
        }
    }

    // 데이터 쓰기 (동기적으로 수행)
    private void handleWrite(SelectionKey key) throws IOException {
        SocketChannel clientChannel = (SocketChannel) key.channel();
        ByteBuffer buffer = (ByteBuffer) key.attachment();

        if (buffer != null && buffer.hasRemaining()) {
            clientChannel.write(buffer);
        }

        if (buffer == null || !buffer.hasRemaining()) {
            key.interestOps(SelectionKey.OP_READ);
            key.attach(null);
        }
    }

    public static void main(String[] args) {
        try {
            ReactorServer server = new ReactorServer(8080);
            server.run();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

// ReactorClient.java (테스트용 클라이언트)
import java.io.*;
import java.net.*;

public class ReactorClient {
    public static void main(String[] args) {
        try (Socket socket = new Socket("localhost", 8080);
             PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
             BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()))) {

            // 메시지 전송
            String message = "Hello Reactor!";
            System.out.println("Sending: " + message);
            out.println(message);

            // 응답 수신
            String response = in.readLine();
            System.out.println("Received: " + response);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

**Proactor 패턴 구현 (Java AIO)**:

```java
// ProactorServer.java
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.*;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;

public class ProactorServer {
    private final AsynchronousServerSocketChannel serverChannel;
    private final int port;

    public ProactorServer(int port) throws IOException {
        this.port = port;
        this.serverChannel = AsynchronousServerSocketChannel.open();
        this.serverChannel.bind(new InetSocketAddress("localhost", port));

        System.out.println("Proactor Server started on port " + port);
    }

    public void start() {
        // 비동기로 연결 수락 (논블로킹)
        serverChannel.accept(null, new AcceptCompletionHandler(serverChannel));
    }

    // 연결 수락 완료 핸들러
    private static class AcceptCompletionHandler
            implements CompletionHandler<AsynchronousSocketChannel, Void> {

        private final AsynchronousServerSocketChannel serverChannel;

        public AcceptCompletionHandler(AsynchronousServerSocketChannel serverChannel) {
            this.serverChannel = serverChannel;
        }

        @Override
        public void completed(AsynchronousSocketChannel clientChannel, Void attachment) {
            // 다음 연결을 위해 다시 accept 호출
            serverChannel.accept(null, this);

            try {
                System.out.println("Accepted connection from: " +
                    clientChannel.getRemoteAddress());

                // 비동기 읽기 시작
                ByteBuffer buffer = ByteBuffer.allocate(1024);
                clientChannel.read(buffer, buffer,
                    new ReadCompletionHandler(clientChannel));

            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        @Override
        public void failed(Throwable exc, Void attachment) {
            System.err.println("Failed to accept connection: " + exc.getMessage());
        }
    }

    // 읽기 완료 핸들러
    private static class ReadCompletionHandler
            implements CompletionHandler<Integer, ByteBuffer> {

        private final AsynchronousSocketChannel clientChannel;

        public ReadCompletionHandler(AsynchronousSocketChannel clientChannel) {
            this.clientChannel = clientChannel;
        }

        @Override
        public void completed(Integer bytesRead, ByteBuffer buffer) {
            if (bytesRead == -1) {
                // 연결 종료
                try {
                    System.out.println("Client closed connection: " +
                        clientChannel.getRemoteAddress());
                    clientChannel.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
                return;
            }

            // 읽은 데이터 처리
            buffer.flip();
            byte[] data = new byte[buffer.remaining()];
            buffer.get(data);
            String message = new String(data, StandardCharsets.UTF_8).trim();

            try {
                System.out.println("Received: " + message + " from " +
                    clientChannel.getRemoteAddress());
            } catch (IOException e) {
                e.printStackTrace();
            }

            // 응답 준비 및 비동기 쓰기
            String response = "Echo: " + message;
            ByteBuffer responseBuffer = ByteBuffer.wrap(response.getBytes(StandardCharsets.UTF_8));

            clientChannel.write(responseBuffer, responseBuffer,
                new WriteCompletionHandler(clientChannel));
        }

        @Override
        public void failed(Throwable exc, ByteBuffer buffer) {
            System.err.println("Failed to read: " + exc.getMessage());
            try {
                clientChannel.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    // 쓰기 완료 핸들러
    private static class WriteCompletionHandler
            implements CompletionHandler<Integer, ByteBuffer> {

        private final AsynchronousSocketChannel clientChannel;

        public WriteCompletionHandler(AsynchronousSocketChannel clientChannel) {
            this.clientChannel = clientChannel;
        }

        @Override
        public void completed(Integer bytesWritten, ByteBuffer buffer) {
            if (buffer.hasRemaining()) {
                // 버퍼에 남은 데이터가 있으면 계속 쓰기
                clientChannel.write(buffer, buffer, this);
            } else {
                // 쓰기 완료, 다음 읽기 대기
                ByteBuffer readBuffer = ByteBuffer.allocate(1024);
                clientChannel.read(readBuffer, readBuffer,
                    new ReadCompletionHandler(clientChannel));
            }
        }

        @Override
        public void failed(Throwable exc, ByteBuffer buffer) {
            System.err.println("Failed to write: " + exc.getMessage());
            try {
                clientChannel.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static void main(String[] args) {
        try {
            ProactorServer server = new ProactorServer(8081);
            server.start();

            // 메인 스레드가 종료되지 않도록 대기
            System.out.println("Server is running. Press Ctrl+C to stop.");
            TimeUnit.DAYS.sleep(1);

        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }
}

// ProactorClient.java (테스트용 비동기 클라이언트)
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.*;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.*;

public class ProactorClient {
    public static void main(String[] args) {
        try {
            AsynchronousSocketChannel clientChannel = AsynchronousSocketChannel.open();

            // 비동기 연결
            Future<Void> connectFuture = clientChannel.connect(
                new InetSocketAddress("localhost", 8081));
            connectFuture.get(); // 연결 완료 대기

            System.out.println("Connected to server");

            // 메시지 전송
            String message = "Hello Proactor!";
            ByteBuffer writeBuffer = ByteBuffer.wrap(message.getBytes(StandardCharsets.UTF_8));

            Future<Integer> writeFuture = clientChannel.write(writeBuffer);
            writeFuture.get(); // 쓰기 완료 대기
            System.out.println("Sent: " + message);

            // 응답 수신
            ByteBuffer readBuffer = ByteBuffer.allocate(1024);
            Future<Integer> readFuture = clientChannel.read(readBuffer);
            int bytesRead = readFuture.get(5, TimeUnit.SECONDS); // 5초 타임아웃

            readBuffer.flip();
            byte[] data = new byte[bytesRead];
            readBuffer.get(data);
            String response = new String(data, StandardCharsets.UTF_8);
            System.out.println("Received: " + response);

            clientChannel.close();

        } catch (IOException | InterruptedException | ExecutionException | TimeoutException e) {
            e.printStackTrace();
        }
    }
}
```

**실제 사용 사례**:

**Reactor 패턴 사용**:
- **Node.js**: Event Loop 기반 아키텍처
- **Redis**: 단일 스레드 이벤트 루프
- **Nginx**: 이벤트 기반 웹 서버
- **Netty**: 고성능 네트워크 프레임워크 (Reactor 기반)

**Proactor 패턴 사용**:
- **Windows IOCP (I/O Completion Ports)**: Windows 비동기 I/O
- **Boost.Asio**: C++ 비동기 I/O 라이브러리
- **Proactor Framework**: ACE (Adaptive Communication Environment)

**Reactor vs Proactor 선택 기준**:

| 상황 | 권장 패턴 |
|------|----------|
| **적은 수의 동시 연결** | Reactor (간단한 구현) |
| **많은 동시 연결 (10,000+)** | Proactor (높은 확장성) |
| **빠른 I/O 작업** | Reactor (오버헤드 적음) |
| **긴 시간의 I/O 작업** | Proactor (블로킹 없음) |
| **단순한 프로토콜** | Reactor (디버깅 용이) |
| **복잡한 비동기 흐름** | Proactor (비동기 체인) |
| **Linux/Unix 환경** | Reactor (epoll, kqueue 지원) |
| **Windows 환경** | Proactor (IOCP 지원) |

**요약**:

Reactor 패턴은 동기 이벤트 기반 I/O를 사용하며, Proactor 패턴은 비동기 I/O 처리를 기반으로 한다. 두 패턴 간의 선택은 애플리케이션의 특정 요구사항에 따라 결정되어야 하며, 동시 I/O 작업의 수, 예상 완료 시간, 사용 중인 플랫폼이나 라이브러리 등을 고려해야 한다. 실제 프로덕션 애플리케이션에서는 더 고급 메커니즘을 사용하여 I/O 작업, 에러 처리, 리소스 공유 등을 처리해야 한다.

## 배치 처리 vs 스트림 처리 (Batch Processing vs Stream Processing)

- [7 Best Practices for Data Governance](https://atlan.com/batch-processing-vs-stream-processing/)

**배치 처리(Batch Processing)**와 **스트림 처리(Stream Processing)**는 대규모 데이터를 처리하는 두 가지 주요 패러다임입니다. 각각의 특성과 사용 사례에 따라 적절한 방식을 선택해야 합니다.

### 아키텍처 비교

```
┌─────────────────────────────────────────────────────────┐
│                   배치 처리 (Batch)                      │
├─────────────────────────────────────────────────────────┤
│  데이터 수집 (1시간/1일)                                  │
│       ↓                                                  │
│  [========== 대량 데이터 ==========]                     │
│       ↓                                                  │
│  배치 작업 실행 (MapReduce/Spark)                        │
│       ↓                                                  │
│  결과 저장 (HDFS/S3)                                     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   스트림 처리 (Stream)                    │
├─────────────────────────────────────────────────────────┤
│  데이터 스트림 (실시간)                                   │
│    ↓    ↓    ↓    ↓    ↓    ↓                           │
│  [이벤트1][이벤트2][이벤트3]...                           │
│    ↓    ↓    ↓    ↓    ↓    ↓                           │
│  실시간 처리 (Kafka/Flink)                               │
│    ↓    ↓    ↓    ↓    ↓    ↓                           │
│  즉시 결과 출력                                           │
└─────────────────────────────────────────────────────────┘
```

### 1. 데이터 처리 방식

#### 배치 처리 (Batch Processing)

**정의**: 일정 시간 동안 수집된 데이터를 모아서 한 번에 처리하는 방식

**특징**:
- 데이터를 큰 블록(chunk)으로 묶어서 처리
- 정해진 스케줄에 따라 실행 (예: 매일 자정, 매시간)
- 높은 처리량(Throughput) 중심
- 완료 후 일괄 결과 산출

**예제 - Hadoop MapReduce**:
```java
// 일일 로그 분석 배치 작업
public class DailyLogAnalysisMapper
        extends Mapper<LongWritable, Text, Text, IntWritable> {

    private final static IntWritable one = new IntWritable(1);
    private Text errorType = new Text();

    @Override
    public void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {

        String line = value.toString();
        // 로그 파싱: [2024-01-15 10:23:45] ERROR: Database connection failed
        if (line.contains("ERROR")) {
            String[] parts = line.split(":");
            if (parts.length >= 2) {
                errorType.set(parts[1].trim());
                context.write(errorType, one);
            }
        }
    }
}

public class DailyLogAnalysisReducer
        extends Reducer<Text, IntWritable, Text, IntWritable> {

    @Override
    public void reduce(Text key, Iterable<IntWritable> values,
            Context context) throws IOException, InterruptedException {

        int sum = 0;
        for (IntWritable val : values) {
            sum += val.get();
        }

        context.write(key, new IntWritable(sum));
    }
}

// 실행: 하루에 한 번 자정에 실행
// 결과: Database connection failed: 1523건
//       Network timeout: 342건
//       Out of memory: 89건
```

#### 스트림 처리 (Stream Processing)

**정의**: 데이터가 도착하는 즉시 실시간으로 처리하는 방식

**특징**:
- 연속적인 데이터 스트림 처리
- 밀리초~초 단위의 낮은 지연시간(Latency)
- 이벤트 기반 처리
- 지속적인 결과 산출

**예제 - Apache Kafka Streams**:
```java
// 실시간 주문 처리 스트림
public class RealTimeOrderProcessor {

    public static void main(String[] args) {
        Properties props = new Properties();
        props.put(StreamsConfig.APPLICATION_ID_CONFIG,
            "realtime-order-processor");
        props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG,
            "localhost:9092");

        StreamsBuilder builder = new StreamsBuilder();

        // 주문 스트림 읽기
        KStream<String, Order> orders = builder.stream("orders",
            Consumed.with(Serdes.String(), new OrderSerde()));

        // 실시간 처리: 고액 주문 감지
        KStream<String, Order> highValueOrders = orders
            .filter((orderId, order) ->
                order.getAmount() > 10000
            )
            .peek((orderId, order) ->
                System.out.println("고액 주문 감지: " + orderId +
                    " - $" + order.getAmount())
            );

        // 1분 윈도우로 집계
        KTable<Windowed<String>, Long> orderCounts = orders
            .groupBy((orderId, order) -> order.getCustomerId())
            .windowedBy(TimeWindows.of(Duration.ofMinutes(1)))
            .count();

        // 실시간 사기 감지
        KStream<String, Alert> fraudAlerts = orders
            .filter((orderId, order) -> {
                // 1분 내 동일 고객의 주문 수 확인
                Long count = getCustomerOrderCount(
                    order.getCustomerId()
                );
                return count != null && count > 5;
            })
            .mapValues(order ->
                new Alert(order.getCustomerId(),
                    "의심스러운 활동 감지: 1분 내 5건 이상 주문")
            );

        // 알림 토픽으로 전송
        fraudAlerts.to("fraud-alerts",
            Produced.with(Serdes.String(), new AlertSerde()));

        KafkaStreams streams = new KafkaStreams(
            builder.build(), props
        );
        streams.start();

        // 결과: 주문이 들어오는 즉시 처리 및 알림
        // 지연시간: < 100ms
    }
}
```

### 2. 지연 시간 (Latency)

| 측면 | 배치 처리 | 스트림 처리 |
|------|-----------|-------------|
| **처리 시점** | 스케줄된 시간 | 데이터 도착 즉시 |
| **지연 시간** | 분~시간 단위 | 밀리초~초 단위 |
| **데이터 신선도** | 오래된 데이터 | 최신 데이터 |
| **결과 가용성** | 배치 완료 후 | 실시간 |

**배치 처리 타임라인**:
```
시간축: 0:00 -------- 1:00 -------- 2:00 -------- 3:00
        [데이터 수집1시간] [처리 30분] [결과]
        ↓ 지연시간: 1.5시간
```

**스트림 처리 타임라인**:
```
시간축: 0:00:00.000 -- 0:00:00.100 -- 0:00:00.200
        [이벤트] → [처리] → [결과]
        ↓ 지연시간: < 100ms
```

### 3. 사용 사례

#### 배치 처리 사용 사례

**1. 일일 리포트 생성**
```java
// 매일 자정 실행: 일일 매출 리포트
@Scheduled(cron = "0 0 0 * * *")
public void generateDailySalesReport() {
    LocalDate yesterday = LocalDate.now().minusDays(1);

    // 전날 데이터 조회
    List<Order> orders = orderRepository
        .findByOrderDateBetween(
            yesterday.atStartOfDay(),
            yesterday.plusDays(1).atStartOfDay()
        );

    // 집계
    BigDecimal totalSales = orders.stream()
        .map(Order::getAmount)
        .reduce(BigDecimal.ZERO, BigDecimal::add);

    // 리포트 생성
    DailyReport report = DailyReport.builder()
        .date(yesterday)
        .totalOrders(orders.size())
        .totalSales(totalSales)
        .build();

    reportRepository.save(report);
    emailService.sendDailyReport(report);
}
```

**2. ETL 작업**
```java
// 데이터 웨어하우스로 일괄 로드
public class ETLBatchJob {

    @Scheduled(cron = "0 0 2 * * *")  // 매일 오전 2시
    public void runETL() {
        // Extract: 운영 DB에서 추출
        List<Transaction> transactions =
            extractTransactionsFromOperationalDB();

        // Transform: 데이터 변환
        List<FactTransaction> facts = transactions.stream()
            .map(this::transformToFact)
            .collect(Collectors.toList());

        // Load: 데이터 웨어하우스로 적재
        dataWarehouseRepository.batchInsert(facts);

        // 통계 업데이트
        updateAggregatedTables();
    }
}
```

**3. 월간 급여 처리**
- 즉시 처리 불필요
- 정확성이 속도보다 중요
- 대량의 복잡한 계산

#### 스트림 처리 사용 사례

**1. 실시간 사기 감지**
```java
// 실시간 신용카드 거래 모니터링
public class FraudDetectionStream {

    public void processTransactionStream() {
        StreamsBuilder builder = new StreamsBuilder();

        KStream<String, Transaction> transactions =
            builder.stream("card-transactions");

        // 실시간 패턴 분석
        KStream<String, FraudAlert> frauds = transactions
            .groupByKey()
            .windowedBy(TimeWindows.of(Duration.ofMinutes(5)))
            .aggregate(
                TransactionStats::new,
                (cardId, transaction, stats) -> {
                    stats.addTransaction(transaction);

                    // 의심스러운 패턴 감지
                    if (stats.getTotalAmount() > 5000 ||
                        stats.getTransactionCount() > 10 ||
                        stats.hasInternationalTransaction()) {
                        return stats.markAsFraud();
                    }
                    return stats;
                },
                Materialized.with(Serdes.String(),
                    new TransactionStatsSerde())
            )
            .toStream()
            .filter((windowed, stats) -> stats.isFraud())
            .mapValues(stats ->
                new FraudAlert(stats.getCardId(),
                    "의심스러운 거래 패턴 감지",
                    stats.getTotalAmount())
            );

        // 즉시 카드 차단
        frauds.foreach((cardId, alert) -> {
            cardService.blockCard(alert.getCardId());
            notificationService.sendAlert(alert);
        });

        // 지연시간: < 100ms (거래 후 즉시 차단)
    }
}
```

**2. 실시간 추천 시스템**
```java
// 사용자 행동 기반 실시간 추천
public class RealtimeRecommendationStream {

    public void processUserBehavior() {
        StreamsBuilder builder = new StreamsBuilder();

        KStream<String, UserEvent> events =
            builder.stream("user-events");

        // 실시간 프로필 업데이트
        KTable<String, UserProfile> profiles = events
            .groupBy((userId, event) -> userId)
            .aggregate(
                UserProfile::new,
                (userId, event, profile) -> {
                    profile.updateWithEvent(event);
                    return profile;
                },
                Materialized.with(Serdes.String(),
                    new UserProfileSerde())
            );

        // 실시간 추천 생성
        KStream<String, Recommendation> recommendations = events
            .join(profiles,
                (event, profile) ->
                    recommendationEngine.generate(event, profile),
                JoinWindows.of(Duration.ofSeconds(10))
            );

        // 즉시 사용자에게 푸시
        recommendations.foreach((userId, rec) -> {
            pushService.sendRecommendation(userId, rec);
        });

        // 결과: 사용자가 상품을 클릭하면 즉시 관련 추천 표시
    }
}
```

**3. 실시간 대시보드**
```java
// IoT 센서 데이터 실시간 모니터링
public class IoTMonitoringStream {

    public void processS
ensorData() {
        StreamsBuilder builder = new StreamsBuilder();

        KStream<String, SensorReading> readings =
            builder.stream("sensor-data");

        // 실시간 이상 감지
        KStream<String, Alert> alerts = readings
            .filter((sensorId, reading) ->
                reading.getTemperature() > 80 ||  // 온도 임계값
                reading.getPressure() < 10         // 압력 임계값
            )
            .mapValues(reading ->
                new Alert(reading.getSensorId(),
                    "임계값 초과",
                    reading.getTimestamp())
            );

        // 즉시 알림 전송
        alerts.foreach((sensorId, alert) -> {
            alertService.sendUrgentAlert(alert);
            controlSystem.shutdown(sensorId);  // 자동 차단
        });

        // 1초 윈도우로 집계하여 대시보드 업데이트
        KTable<Windowed<String>, Double> avgReadings = readings
            .groupByKey()
            .windowedBy(TimeWindows.of(Duration.ofSeconds(1)))
            .aggregate(
                () -> new AverageCalculator(),
                (sensorId, reading, avg) -> {
                    avg.add(reading.getValue());
                    return avg;
                },
                Materialized.with(Serdes.String(),
                    new AverageCalculatorSerde())
            )
            .mapValues(avg -> avg.getAverage());

        // WebSocket으로 대시보드 실시간 업데이트
        avgReadings.toStream().foreach((windowed, avg) -> {
            dashboardService.updateMetric(
                windowed.key(),
                avg
            );
        });
    }
}
```

### 4. 장애 허용성 및 신뢰성

#### 배치 처리

**특징**:
- 재시작 가능: 체크포인트에서 재실행
- 멱등성(Idempotency) 보장 용이
- 전체 재처리 가능

**예제**:
```java
// 배치 작업 재시작 메커니즘
@Component
public class ResumableBatchJob {

    @Autowired
    private JobRepository jobRepository;

    public void runBatchWithCheckpoint() {
        String jobId = UUID.randomUUID().toString();
        JobExecution execution = jobRepository.createJobExecution(
            "daily-sales-job", jobId
        );

        try {
            // 마지막 체크포인트 조회
            Long lastProcessedId = execution.getLastProcessedId();

            // 체크포인트 이후 데이터 처리
            List<Order> orders = orderRepository
                .findByIdGreaterThan(lastProcessedId);

            for (int i = 0; i < orders.size(); i += 1000) {
                List<Order> batch = orders.subList(
                    i, Math.min(i + 1000, orders.size())
                );

                processBatch(batch);

                // 체크포인트 저장
                execution.updateCheckpoint(
                    batch.get(batch.size() - 1).getId()
                );
            }

            execution.complete();

        } catch (Exception e) {
            execution.fail(e);
            // 다음 실행 시 마지막 체크포인트부터 재시작
        }
    }
}
```

#### 스트림 처리

**특징**:
- Exactly-once 처리 보장 필요
- 상태 관리 복잡
- 워터마크(Watermark) 처리

**예제**:
```java
// Kafka Streams Exactly-Once 처리
public class ExactlyOnceProcessor {

    public void configureExactlyOnce() {
        Properties props = new Properties();

        // Exactly-Once 시맨틱 활성화
        props.put(StreamsConfig.PROCESSING_GUARANTEE_CONFIG,
            StreamsConfig.EXACTLY_ONCE_V2);

        // 트랜잭션 타임아웃
        props.put(StreamsConfig.COMMIT_INTERVAL_MS_CONFIG, 1000);

        StreamsBuilder builder = new StreamsBuilder();

        KStream<String, Payment> payments =
            builder.stream("payments");

        // 상태 저장소 (장애 복구 가능)
        payments
            .groupByKey()
            .aggregate(
                () -> new AccountBalance(),
                (accountId, payment, balance) -> {
                    balance.add(payment.getAmount());
                    return balance;
                },
                Materialized.<String, AccountBalance>as(
                    Stores.persistentKeyValueStore("balances")
                )
                .withKeySerde(Serdes.String())
                .withValueSerde(new AccountBalanceSerde())
            );

        // 장애 발생 시 자동 복구
        // - 마지막 커밋된 오프셋부터 재처리
        // - 상태 저장소에서 이전 상태 복원
        // - 중복 처리 방지
    }
}
```

### 5. 확장성 및 성능

| 측면 | 배치 처리 | 스트림 처리 |
|------|-----------|-------------|
| **확장 방식** | 수직/수평 모두 가능 | 주로 수평 확장 |
| **처리량** | 매우 높음 (TB~PB) | 높음 (GB~TB/시간) |
| **리소스 활용** | 주기적 피크 | 지속적 사용 |
| **병렬화** | 데이터 파티셔닝 | 파티션/샤딩 |

**배치 처리 확장**:
```java
// Spark를 사용한 대규모 배치 처리
SparkConf conf = new SparkConf()
    .setAppName("Large Scale Batch")
    .set("spark.executor.instances", "100")    // 100개 실행자
    .set("spark.executor.memory", "8g")        // 각 8GB 메모리
    .set("spark.executor.cores", "4");         // 각 4코어

JavaSparkContext sc = new JavaSparkContext(conf);

// 대규모 데이터 처리 (수 TB)
JavaRDD<String> logs = sc.textFile("hdfs://logs/2024/*");

// 400개 파티션으로 병렬 처리
JavaPairRDD<String, Integer> errors = logs
    .filter(line -> line.contains("ERROR"))
    .mapToPair(line -> new Tuple2<>(extractError(line), 1))
    .reduceByKey((a, b) -> a + b, 400);  // 400개 태스크로 분산

errors.saveAsTextFile("hdfs://results/error-counts");

// 처리량: 10TB 데이터를 1시간 내 처리
```

**스트림 처리 확장**:
```java
// Kafka 파티션을 통한 수평 확장
public class ScalableStreamProcessor {

    public void setupScalableStream() {
        Properties props = new Properties();

        // 여러 인스턴스가 파티션 분할 처리
        props.put(StreamsConfig.APPLICATION_ID_CONFIG,
            "scalable-processor");
        props.put(StreamsConfig.NUM_STREAM_THREADS_CONFIG, 4);

        // 입력 토픽: 32개 파티션
        // 인스턴스 8개 × 스레드 4개 = 32개 병렬 처리

        StreamsBuilder builder = new StreamsBuilder();

        KStream<String, Order> orders = builder.stream("orders");

        orders
            .filter((id, order) -> order.getAmount() > 100)
            .to("high-value-orders");

        // 동적 확장: 인스턴스 추가 시 자동 리밸런싱
        // 처리량: 초당 10만 이벤트 이상
    }
}
```

### 6. 복잡성

#### 배치 처리
- 설정 및 운영이 상대적으로 단순
- 디버깅 용이
- 멱등성 보장 쉬움

#### 스트림 처리
- 상태 관리 복잡
- 시간 순서 처리 (Out-of-order events)
- 워터마크 관리
- Windowing 전략

**시간 순서 문제 해결**:
```java
// 늦게 도착하는 이벤트 처리
public class LateArrivalHandler {

    public void handleLateEvents() {
        StreamsBuilder builder = new StreamsBuilder();

        KStream<String, Event> events = builder.stream("events");

        // 워터마크: 5분까지 늦게 도착한 이벤트 허용
        KTable<Windowed<String>, Long> counts = events
            .groupByKey()
            .windowedBy(
                TimeWindows.of(Duration.ofMinutes(1))
                    .grace(Duration.ofMinutes(5))  // 지연 허용
            )
            .count();

        // 늦게 도착한 이벤트도 올바른 윈도우에 포함
    }
}
```

### 7. 도구 및 플랫폼

#### 배치 처리 도구

| 도구 | 특징 | 사용 사례 |
|------|------|-----------|
| **Hadoop MapReduce** | 초기 빅데이터 프레임워크 | 로그 분석, ETL |
| **Apache Spark** | 인메모리 처리, 빠른 속도 | 데이터 웨어하우스, ML |
| **Apache Hive** | SQL 기반 배치 처리 | 데이터 웨어하우징 |

#### 스트림 처리 도구

| 도구 | 특징 | 사용 사례 |
|------|------|-----------|
| **Apache Kafka Streams** | 경량, Kafka 통합 | 이벤트 처리, 마이크로서비스 |
| **Apache Flink** | 진정한 스트리밍, 상태 관리 | 복잡한 이벤트 처리 |
| **Apache Storm** | 초기 스트림 처리 프레임워크 | 실시간 분석 |
| **Spark Streaming** | 마이크로 배치 방식 | 하이브리드 워크로드 |

### 8. 하이브리드 접근: Lambda & Kappa 아키텍처

#### Lambda 아키텍처

```
┌─────────────────────────────────────────┐
│            데이터 소스                   │
└────────┬────────────────┬───────────────┘
         │                │
    ┌────▼─────┐    ┌────▼─────┐
    │  배치 계층  │    │ 속도 계층  │
    │ (Spark)   │    │ (Kafka)  │
    │ 정확한 결과│    │ 빠른 결과 │
    └────┬─────┘    └────┬─────┘
         │                │
         └────────┬───────┘
                  │
           ┌──────▼──────┐
           │   서빙 계층   │
           │  (병합 결과) │
           └─────────────┘
```

**구현 예제**:
```java
// Lambda 아키텍처 - 배치 + 스트림
public class LambdaArchitecture {

    // 배치 계층: 정확한 계산 (1시간마다)
    @Scheduled(cron = "0 0 * * * *")
    public void batchLayer() {
        // 지난 1시간 데이터로 정확한 통계 계산
        Statistics accurate = calculateAccurate(
            lastHourData
        );
        batchResultStore.save(accurate);
    }

    // 속도 계층: 빠른 근사치 (실시간)
    public void speedLayer() {
        KStream<String, Event> stream = ...;

        KTable<String, ApproxStats> approx = stream
            .groupByKey()
            .aggregate(
                ApproxStats::new,
                (key, event, stats) -> {
                    stats.update(event);
                    return stats;
                }
            );

        // 실시간 근사 결과
    }

    // 서빙 계층: 결과 병합
    public Statistics query(String key) {
        // 배치 결과 + 최근 실시간 결과 병합
        Statistics batch = batchResultStore.get(key);
        Statistics speed = speedResultStore.get(key);
        return merge(batch, speed);
    }
}
```

#### Kappa 아키텍처 (단순화)

```
┌─────────────────────────────────────────┐
│            데이터 소스                   │
└────────────────┬───────────────────────┘
                 │
         ┌───────▼────────┐
         │  스트림 계층     │
         │  (Kafka/Flink) │
         │  (재처리 가능)   │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │   서빙 계층      │
         └────────────────┘
```

### 비교 요약

| 기준 | 배치 처리 | 스트림 처리 |
|------|-----------|-------------|
| **지연시간** | 분~시간 | 밀리초~초 |
| **처리량** | 매우 높음 | 높음 |
| **복잡성** | 낮음 | 높음 |
| **정확성** | 매우 높음 | 높음 (근사치 가능) |
| **비용** | 저렴 (주기적 실행) | 비쌈 (지속 실행) |
| **사용 사례** | 리포트, ETL, 분석 | 모니터링, 알림, 대시보드 |
| **장애 복구** | 쉬움 (재실행) | 복잡 (상태 복원) |

### 선택 가이드

**배치 처리를 선택해야 할 때**:
- 실시간 응답이 필요 없는 경우
- 대량의 히스토리 데이터 처리
- 높은 정확성이 요구되는 경우
- 복잡한 집계 및 분석

**스트림 처리를 선택해야 할 때**:
- 밀리초~초 단위 응답 필요
- 실시간 의사결정 (사기 감지, 추천)
- 연속적인 데이터 모니터링
- 이벤트 기반 아키텍처

**하이브리드 (Lambda/Kappa)를 선택해야 할 때**:
- 실시간성과 정확성 모두 중요
- 대규모 데이터 + 실시간 분석
- 재처리 및 수정이 빈번한 경우

## Checksum

- [What Is a Checksum?](https://www.lifewire.com/what-does-checksum-mean-2625825)

-----

### 개념

체크섬(Checksum)은 데이터(주로 파일)에 대해 암호화 해시 함수 알고리즘을 실행한 결과값입니다. 파일의 원본과 사용자 버전을 비교하여 파일의 무결성과 정품 여부를 확인하는 데 사용됩니다.

**다른 이름:**
- 해시합계(Hash Sum)
- 해시 값(Hash Value)
- 해시 코드(Hash Code)
- 해시(Hash)

### 사용 목적

체크섬은 파일이 올바르게 전송되었는지 확인하는 데 사용됩니다:

1. **파일 무결성 검증**
   - 다운로드한 파일의 체크섬 생성
   - 원본 파일의 체크섬과 비교
   - 두 값이 일치하면 파일이 손상되지 않았음을 확인

2. **변조 탐지**
   - 체크섬이 일치하지 않으면 파일이 손상되었거나 변조되었을 가능성
   - 원본 소스에서 다운로드한 파일이 올바른지 확인 가능

### 체크섬 계산기

체크섬 계산기는 체크섬을 생성하는 도구로, 다양한 암호화 해시 함수를 지원합니다:

- **주요 해시 알고리즘**: MD5, SHA-1, SHA-256, CRC32 등
- **용도**: 파일 무결성 검증, 데이터 전송 오류 탐지
- **활용 예**: 소프트웨어 다운로드 검증, 백업 파일 확인, 네트워크 전송 데이터 검증

## MSA (Micro Service Architecture)

[Micro Service Architecture](/msa/README.md)

## Cloud Design Patterns

[Cloud Design Patterns](clouddesignpattern.md)

## Enterprise Integration Patterns

[Enterprise Integration Patterns](/eip/README.md)

## DDD

[DDD](/ddd/README.md)

## Architecture

[Architecture](/architecture/README.md)

# 최신 기술 트렌드

## Serverless Architecture

**Lambda 함수 예제 (Python):**

```python
import json

def lambda_handler(event, context):
    # API Gateway 요청 처리
    body = json.loads(event['body'])

    # 비즈니스 로직
    result = process_data(body)

    # 응답 반환
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json'
        },
        'body': json.dumps(result)
    }

def process_data(data):
    # 데이터 처리 로직
    return {'message': 'processed', 'data': data}
```

**장점:**
- 서버 관리 불필요
- 자동 스케일링
- 사용한 만큼만 비용 지불
- 빠른 배포

**단점:**
- Cold Start 지연
- 실행 시간 제한 (AWS Lambda: 15분)
- 상태 유지 어려움
- 벤더 종속성

**사용 사례:**
- API Gateway
- 이벤트 처리 (S3 업로드, DynamoDB 변경)
- 스케줄된 작업
- 이미지 리사이징, 데이터 변환

## Container Orchestration - Kubernetes

**기본 개념:**

```yaml
# Deployment - 애플리케이션 배포 정의
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  replicas: 3  # 3개의 Pod 실행
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: web
        image: nginx:1.21
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "128Mi"
            cpu: "250m"
          limits:
            memory: "256Mi"
            cpu: "500m"
---
# Service - 네트워크 엔드포인트 노출
apiVersion: v1
kind: Service
metadata:
  name: web-service
spec:
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 80
  type: LoadBalancer
---
# HorizontalPodAutoscaler - 자동 스케일링
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**StatefulSet - 상태 유지 애플리케이션:**

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mysql
spec:
  serviceName: mysql
  replicas: 3
  selector:
    matchLabels:
      app: mysql
  template:
    metadata:
      labels:
        app: mysql
    spec:
      containers:
      - name: mysql
        image: mysql:8.0
        ports:
        - containerPort: 3306
        volumeMounts:
        - name: data
          mountPath: /var/lib/mysql
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
```

## Control Plane vs Data Plane vs Management Plane

> * [Data plane - Wikipedia](https://en.wikipedia.org/wiki/Data_plane)

### 개념

현대 분산 시스템과 네트워크 아키텍처는 역할에 따라 세 가지 주요 평면(Plane)으로 구분됩니다. 이러한 분리는 시스템의 확장성, 유지보수성, 보안성을 향상시킵니다.

**세 가지 평면 비교**:

| 평면 | 역할 | 주요 기능 | 예시 | 성능 요구사항 |
|------|------|----------|------|--------------|
| **Data Plane** | 실제 데이터 처리 | 패킷 전달, 요청 처리, 트래픽 라우팅 | 프록시, 로드 밸런서, 사이드카 | 매우 높음 (낮은 지연시간) |
| **Control Plane** | 설정 및 제어 | 정책 배포, 라우팅 규칙, 구성 관리 | API 서버, 컨트롤러, 오케스트레이터 | 보통 (안정성 중요) |
| **Management Plane** | 운영 및 관리 | 모니터링, 로깅, 분석, UI/CLI | 대시보드, 메트릭 수집기, 알림 시스템 | 낮음 (가용성 중요) |

### 상세 설명

**1. Data Plane (데이터 평면)**

실제 사용자 트래픽과 데이터를 처리하는 계층입니다.

**특징**:
- 모든 실제 데이터 요청 처리
- 초당 수백만 건의 요청 처리 가능해야 함
- 낮은 지연시간(레이턴시)이 매우 중요
- 수평 확장(Scale-out)으로 성능 향상
- 장애 시 서비스 중단 직접 영향

**역할**:
- 패킷/요청 전달 및 라우팅
- 로드 밸런싱
- 서비스 간 통신 프록싱
- TLS 종료 및 암호화
- 트래픽 필터링 및 속도 제한

**2. Control Plane (컨트롤 평면)**

Data Plane을 설정하고 제어하는 계층입니다.

**특징**:
- Data Plane의 동작 방식 결정
- 상대적으로 낮은 요청 빈도
- 높은 안정성과 일관성 필요
- 장애 시 기존 연결은 유지되지만 새 설정 불가

**역할**:
- 라우팅 규칙 및 정책 배포
- 서비스 디스커버리 정보 제공
- 인증서 및 시크릿 관리
- 상태 관리 및 동기화
- Data Plane 구성 업데이트

**3. Management Plane (관리 평면)**

시스템 전체를 모니터링하고 운영하는 계층입니다.

**특징**:
- 사람(운영자)과 시스템 간 인터페이스
- 실시간 성능 요구사항 낮음
- 장애 시에도 기존 서비스는 계속 동작
- 관찰 가능성(Observability) 제공

**역할**:
- 메트릭 수집 및 시각화
- 로그 집계 및 분석
- 알림 및 경보 관리
- 감사 및 컴플라이언스 추적
- UI/CLI를 통한 관리 인터페이스

### 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│                    Management Plane                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Dashboard   │  │  Monitoring  │  │   Logging    │      │
│  │     UI       │  │  (Prometheus)│  │ (ELK Stack)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │ 관리/모니터링
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                     Control Plane                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ API Server   │  │  Scheduler   │  │  Controllers │      │
│  │ (K8s/Istio)  │  │              │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │ 설정/정책
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                      Data Plane                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Envoy      │  │   Envoy      │  │   Envoy      │      │
│  │   Proxy      │  │   Proxy      │  │   Proxy      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         ↕                 ↕                 ↕                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Service A   │  │  Service B   │  │  Service C   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
         ↑                                          ↑
         └──────── 실제 사용자 트래픽 ──────────────┘
```

### 실제 예시

**1. Kubernetes**

```java
// Control Plane 컴포넌트
@Component
public class KubernetesControlPlane {

    // API Server: Control Plane의 핵심
    @RestController
    @RequestMapping("/api/v1")
    public class ApiServer {

        @PostMapping("/namespaces/{namespace}/pods")
        public ResponseEntity<Pod> createPod(
                @PathVariable String namespace,
                @RequestBody Pod pod) {

            // 1. 요청 인증 및 권한 확인
            authenticate(request);
            authorize(request, "create", "pods");

            // 2. 요청 검증
            validate(pod);

            // 3. etcd에 저장
            etcdClient.save(pod);

            // 4. Scheduler에게 알림
            schedulerQueue.add(pod);

            return ResponseEntity.ok(pod);
        }
    }

    // Scheduler: Pod를 Node에 배치
    @Service
    public class Scheduler {

        @EventListener
        public void onPodCreated(PodCreatedEvent event) {
            Pod pod = event.getPod();

            // 1. 적절한 Node 선택
            Node node = selectBestNode(pod);

            // 2. Binding 생성
            Binding binding = new Binding(pod, node);
            apiServer.createBinding(binding);

            // 3. Kubelet(Data Plane)이 실제 컨테이너 시작
        }
    }

    // Controller: 원하는 상태 유지
    @Service
    public class ReplicaSetController {

        @Scheduled(fixedDelay = 5000)
        public void reconcile() {
            List<ReplicaSet> replicaSets = apiServer.listReplicaSets();

            for (ReplicaSet rs : replicaSets) {
                int desired = rs.getSpec().getReplicas();
                int current = countRunningPods(rs);

                if (current < desired) {
                    // Pod 생성
                    createPods(rs, desired - current);
                } else if (current > desired) {
                    // Pod 삭제
                    deletePods(rs, current - desired);
                }
            }
        }
    }
}

// Data Plane 컴포넌트
@Component
public class Kubelet {

    // Node의 에이전트, 실제 컨테이너 실행
    @EventListener
    public void onPodScheduled(PodScheduledEvent event) {
        Pod pod = event.getPod();

        // 1. 컨테이너 이미지 Pull
        containerRuntime.pullImage(pod.getSpec().getContainers());

        // 2. 컨테이너 시작
        for (Container container : pod.getSpec().getContainers()) {
            containerRuntime.startContainer(container);
        }

        // 3. 상태 업데이트 (Control Plane에 보고)
        apiServer.updatePodStatus(pod.getName(), PodStatus.RUNNING);
    }

    // Health Check (Data Plane 역할)
    @Scheduled(fixedDelay = 10000)
    public void healthCheck() {
        List<Pod> pods = getPodsOnThisNode();

        for (Pod pod : pods) {
            boolean healthy = checkHealth(pod);
            if (!healthy) {
                // Control Plane에 보고
                apiServer.updatePodStatus(pod.getName(), PodStatus.UNHEALTHY);
            }
        }
    }
}

// Management Plane 컴포넌트
@RestController
@RequestMapping("/management")
public class KubernetesDashboard {

    // 모니터링 대시보드
    @GetMapping("/metrics")
    public ClusterMetrics getClusterMetrics() {
        return ClusterMetrics.builder()
            .totalNodes(metricsService.getTotalNodes())
            .totalPods(metricsService.getTotalPods())
            .cpuUsage(metricsService.getCpuUsage())
            .memoryUsage(metricsService.getMemoryUsage())
            .build();
    }

    // 로그 조회
    @GetMapping("/logs/{podName}")
    public String getPodLogs(@PathVariable String podName) {
        return logService.getPodLogs(podName);
    }

    // 알림 설정
    @PostMapping("/alerts")
    public void createAlert(@RequestBody AlertRule rule) {
        alertManager.addRule(rule);
    }
}
```

**2. Service Mesh (Istio)**

```java
// Control Plane (Istiod)
@Service
public class IstioControlPlane {

    // Pilot: 서비스 디스커버리 및 트래픽 관리
    @Service
    public class Pilot {

        // VirtualService 설정을 Envoy Config로 변환
        public void onVirtualServiceChanged(VirtualService vs) {
            // 1. VirtualService 규칙 파싱
            List<Route> routes = parseRoutes(vs);

            // 2. Envoy Configuration 생성
            EnvoyConfig config = EnvoyConfig.builder()
                .routes(routes)
                .clusters(getClusters(vs))
                .listeners(getListeners(vs))
                .build();

            // 3. 모든 Envoy Proxy에 푸시 (xDS 프로토콜)
            for (EnvoyProxy proxy : getAllProxies()) {
                xdsService.pushConfig(proxy, config);
            }
        }
    }

    // Citadel: 인증서 관리
    @Service
    public class Citadel {

        @Scheduled(cron = "0 0 * * * *") // 매시간
        public void rotateCertificates() {
            List<Service> services = getAllServices();

            for (Service service : services) {
                // 새 인증서 생성
                Certificate cert = generateCertificate(service);

                // Envoy Proxy에 배포
                distributeCertificate(service, cert);
            }
        }
    }
}

// Data Plane (Envoy Proxy)
public class EnvoyProxy {

    // 실제 트래픽 프록싱
    public Response handleRequest(Request request) {
        // 1. 라우팅 규칙 적용 (Control Plane에서 받은 설정)
        Cluster targetCluster = routeRequest(request);

        // 2. 로드 밸런싱
        Endpoint endpoint = loadBalancer.selectEndpoint(targetCluster);

        // 3. Circuit Breaker 확인
        if (circuitBreaker.isOpen(endpoint)) {
            return Response.serviceUnavailable();
        }

        // 4. mTLS 적용
        Request secureRequest = applyMTLS(request);

        // 5. 실제 요청 전달
        Response response = httpClient.send(endpoint, secureRequest);

        // 6. 텔레메트리 데이터 수집 (Management Plane으로 전송)
        collectMetrics(request, response);

        return response;
    }

    // Control Plane으로부터 설정 수신 (xDS)
    @Streaming
    public void receiveConfiguration(StreamObserver<EnvoyConfig> observer) {
        xdsClient.subscribe(config -> {
            // 런타임에 동적으로 설정 적용
            applyConfiguration(config);
            observer.onNext(ack());
        });
    }
}

// Management Plane (Kiali)
@RestController
@RequestMapping("/kiali/api")
public class KialiManagementPlane {

    // 서비스 그래프 시각화
    @GetMapping("/namespaces/graph")
    public ServiceGraph getServiceGraph(
            @RequestParam String namespace,
            @RequestParam String duration) {

        // Prometheus에서 메트릭 수집
        List<Metric> metrics = prometheusClient.query(
            String.format("istio_requests_total{namespace='%s'}[%s]", namespace, duration)
        );

        // 서비스 간 관계 그래프 생성
        return buildServiceGraph(metrics);
    }

    // 트래픽 분석
    @GetMapping("/namespaces/{namespace}/services/{service}/metrics")
    public ServiceMetrics getServiceMetrics(
            @PathVariable String namespace,
            @PathVariable String service) {

        return ServiceMetrics.builder()
            .requestRate(getRequestRate(namespace, service))
            .errorRate(getErrorRate(namespace, service))
            .p50Latency(getLatency(namespace, service, 0.5))
            .p95Latency(getLatency(namespace, service, 0.95))
            .p99Latency(getLatency(namespace, service, 0.99))
            .build();
    }
}
```

**3. API Gateway**

```java
// Control Plane
@Service
public class GatewayControlPlane {

    @RestController
    @RequestMapping("/admin/api")
    public class AdminAPI {

        // 라우팅 규칙 추가
        @PostMapping("/routes")
        public ResponseEntity<Route> createRoute(@RequestBody RouteConfig config) {
            // 1. 검증
            validateRoute(config);

            // 2. 데이터베이스에 저장
            Route route = routeRepository.save(config);

            // 3. 모든 Gateway 인스턴스에 설정 푸시
            for (GatewayInstance instance : gatewayInstances) {
                configSyncService.pushRoute(instance, route);
            }

            return ResponseEntity.ok(route);
        }

        // Rate Limit 정책 설정
        @PostMapping("/rate-limits")
        public void setRateLimit(@RequestBody RateLimitPolicy policy) {
            rateLimitService.updatePolicy(policy);

            // Data Plane에 즉시 반영
            notifyDataPlane(policy);
        }
    }
}

// Data Plane
@Component
public class GatewayDataPlane {

    @Autowired
    private RateLimiter rateLimiter;

    @Autowired
    private LoadBalancer loadBalancer;

    // 실제 요청 처리
    public Response handleRequest(HttpServletRequest request) {
        String path = request.getRequestURI();

        // 1. Rate Limiting (Control Plane에서 설정된 정책 적용)
        if (!rateLimiter.allowRequest(request)) {
            return Response.tooManyRequests();
        }

        // 2. 라우팅 규칙 매칭
        Route route = routeMatcher.findRoute(path);
        if (route == null) {
            return Response.notFound();
        }

        // 3. 인증/인가
        if (!authenticator.authenticate(request)) {
            return Response.unauthorized();
        }

        // 4. 백엔드 선택 및 요청 전달
        Backend backend = loadBalancer.selectBackend(route);
        Response response = proxyRequest(backend, request);

        // 5. 메트릭 수집 (Management Plane으로 전송)
        metricsCollector.record(request, response);

        return response;
    }
}

// Management Plane
@RestController
@RequestMapping("/management")
public class GatewayManagementPlane {

    // 실시간 트래픽 모니터링
    @GetMapping("/metrics/realtime")
    public RealtimeMetrics getRealtimeMetrics() {
        return RealtimeMetrics.builder()
            .currentRPS(metricsService.getCurrentRPS())
            .activeConnections(metricsService.getActiveConnections())
            .errorRate(metricsService.getErrorRate())
            .avgLatency(metricsService.getAvgLatency())
            .build();
    }

    // 로그 분석
    @GetMapping("/logs/analysis")
    public LogAnalysis analyzeAccessLogs(
            @RequestParam String startTime,
            @RequestParam String endTime) {

        List<AccessLog> logs = logService.getAccessLogs(startTime, endTime);

        return LogAnalysis.builder()
            .topPaths(analyzeTopPaths(logs))
            .topClients(analyzeTopClients(logs))
            .errorSummary(analyzeErrors(logs))
            .build();
    }

    // 알림 대시보드
    @GetMapping("/alerts")
    public List<Alert> getActiveAlerts() {
        return alertService.getActiveAlerts();
    }
}
```

### 평면 간 통신

```java
@Service
public class PlaneInteraction {

    // Control Plane -> Data Plane: 설정 푸시
    public void pushConfiguration() {
        // 1. Control Plane에서 새 설정 생성
        Configuration config = controlPlane.generateConfig();

        // 2. 모든 Data Plane 인스턴스에 전달
        for (DataPlaneInstance instance : dataPlaneInstances) {
            // 비동기로 설정 푸시 (블로킹 방지)
            CompletableFuture.runAsync(() -> {
                instance.applyConfiguration(config);
            });
        }
    }

    // Data Plane -> Management Plane: 메트릭 전송
    @Scheduled(fixedDelay = 10000) // 10초마다
    public void reportMetrics() {
        DataPlaneMetrics metrics = dataPlane.collectMetrics();

        // Management Plane의 메트릭 수집기로 전송
        managementPlane.ingestMetrics(metrics);
    }

    // Data Plane -> Control Plane: 상태 보고
    @Scheduled(fixedDelay = 5000) // 5초마다
    public void reportHealth() {
        HealthStatus status = dataPlane.checkHealth();

        // Control Plane에 보고
        controlPlane.updateInstanceHealth(instanceId, status);
    }

    // Management Plane -> Control Plane: 관리자 명령
    public void applyManualChange(ConfigChange change) {
        // 1. Management UI에서 변경 요청
        managementPlane.validateChange(change);

        // 2. Control Plane을 통해 적용
        controlPlane.applyChange(change);

        // 3. Data Plane에 자동 전파됨
    }
}
```

### 평면 분리의 이점

| 이점 | 설명 |
|------|------|
| **독립적 확장** | Data Plane은 트래픽에 따라, Control Plane은 클러스터 크기에 따라 개별 확장 |
| **장애 격리** | Control Plane 장애 시에도 Data Plane은 기존 설정으로 계속 동작 |
| **보안 강화** | 각 평면에 대한 접근 제어 및 권한 분리 |
| **유지보수 용이** | 각 평면을 독립적으로 업그레이드 및 패치 가능 |
| **성능 최적화** | Data Plane은 성능, Control Plane은 일관성, Management Plane은 사용성에 최적화 |

### 설계 고려사항

**1. Data Plane**:
- [ ] 최소 지연시간 목표 설정 (예: P99 < 10ms)
- [ ] 수평 확장 가능한 아키텍처
- [ ] Circuit Breaker 및 Retry 로직
- [ ] 경량 프록시 사용 (Envoy, NGINX)

**2. Control Plane**:
- [ ] 강력한 일관성 보장 (etcd, Consul)
- [ ] 고가용성 구성 (Multi-master)
- [ ] 버전 관리 및 롤백 지원
- [ ] Rate Limiting으로 과부하 방지

**3. Management Plane**:
- [ ] 직관적인 UI/UX
- [ ] 역할 기반 접근 제어 (RBAC)
- [ ] 감사 로그 기록
- [ ] 장기 메트릭 저장소 (Prometheus + Thanos)

## Service Mesh

**Istio 아키텍처:**

```yaml
# VirtualService - 트래픽 라우팅
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: reviews
spec:
  hosts:
  - reviews
  http:
  - match:
    - headers:
        end-user:
          exact: jason
    route:
    - destination:
        host: reviews
        subset: v2  # jason 사용자는 v2로
  - route:
    - destination:
        host: reviews
        subset: v1  # 나머지는 v1로
      weight: 90
    - destination:
        host: reviews
        subset: v2
      weight: 10  # 10% 트래픽만 v2로 (Canary)
---
# DestinationRule - 로드 밸런싱, Circuit Breaker
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: reviews
spec:
  host: reviews
  trafficPolicy:
    loadBalancer:
      simple: LEAST_REQUEST
    outlierDetection:  # Circuit Breaker
      consecutiveErrors: 5
      interval: 30s
      baseEjectionTime: 30s
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
```

**주요 기능:**
- 트래픽 관리 (A/B Testing, Canary)
- 보안 (mTLS 자동화)
- 관측성 (분산 추적, 메트릭)
- 장애 복구 (Circuit Breaker, Retry)

## Edge Computing

**특징:**
- 데이터 소스에 가까운 곳에서 처리
- 낮은 지연 시간
- 대역폭 절약
- 개인정보 보호

**사용 사례:**
- IoT 디바이스 데이터 처리
- 실시간 비디오 분석
- CDN
- 자율주행차

**예제: Cloudflare Workers**

```javascript
addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
  // Edge에서 실행되는 코드
  const cache = caches.default
  let response = await cache.match(request)

  if (!response) {
    // 캐시 미스 - Origin 서버에서 가져오기
    response = await fetch(request)

    // 캐시에 저장
    event.waitUntil(cache.put(request, response.clone()))
  }

  return response
}
```

# 시스템 디자인 인터뷰

## 인터뷰 진행 방법

**1단계: 요구사항 명확화 (3-10분)**

질문 예시:
- "사용자 수는 얼마나 되나요?"
- "읽기와 쓰기 비율은 어떻게 되나요?"
- "어떤 기능이 가장 중요한가요?"
- "데이터는 얼마나 오래 보관해야 하나요?"
- "가용성과 일관성 중 무엇이 더 중요한가요?"

**2단계: 용량 추정 (5-10분)**

```
사용자: 1억 명 DAU
QPS 계산:
- 평균 QPS = 100M * 10 requests/day / 86,400 ≈ 11,600 QPS
- 피크 QPS = 11,600 * 2 = 23,200 QPS

저장 공간:
- 사용자당 데이터: 1KB
- 일일 증가량: 100M * 1KB = 100GB/일
- 5년 저장: 100GB * 365 * 5 = 183TB

대역폭:
- 쓰기: 100GB / 86,400초 ≈ 1.2 MB/s
- 읽기 (100:1): 120 MB/s
```

**3단계: 상위 레벨 디자인 (10-15분)**

```
[Client] → [Load Balancer] → [API Servers]
                                    ↓
                              [Cache Layer]
                                    ↓
                            [Database (Master)]
                                    ↓
                          [Database (Replicas)]
```

**4단계: 세부 디자인 (10-25분)**

집중할 영역:
- 데이터 모델
- API 설계
- 캐싱 전략
- 확장성
- 병목 지점

**5단계: 마무리 및 질의응답**

논의 주제:
- 잠재적 병목 지점
- 모니터링 전략
- 배포 방법
- 장애 시나리오

## 난이도별 문제

### Easy

**URL Shortener 상세 설계**

**요구사항:**
- URL 단축 (long URL → short URL)
- 리다이렉션 (short URL → long URL)
- 100M URLs/day
- 읽기:쓰기 = 100:1

**1. API 설계**
```
POST /api/v1/shorten
Body: { "long_url": "https://example.com/very/long/url" }
Response: { "short_url": "https://short.ly/abc123" }

GET /{short_code}
Response: 302 Redirect to long URL
```

**2. 데이터 모델**
```sql
CREATE TABLE urls (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    short_code VARCHAR(7) NOT NULL UNIQUE,
    long_url VARCHAR(2048) NOT NULL,
    user_id BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    click_count BIGINT DEFAULT 0,
    INDEX idx_short_code (short_code),
    INDEX idx_user_id (user_id)
);
```

**3. Short Code 생성 전략**

**방법 1: Hash + Base62**
```python
import hashlib
import base62

def generate_short_code(long_url, counter=0):
    # URL + counter로 해시 생성
    hash_input = f"{long_url}{counter}"
    hash_value = hashlib.sha256(hash_input.encode()).hexdigest()

    # 처음 7자리를 base62로 인코딩
    number = int(hash_value[:8], 16)
    short_code = base62.encode(number)[:7]

    return short_code

# 충돌 처리
def create_short_url(long_url):
    counter = 0
    while True:
        short_code = generate_short_code(long_url, counter)

        # DB에 존재하는지 확인
        if not db.exists(short_code):
            db.insert(short_code, long_url)
            return short_code

        counter += 1
```

**방법 2: ID 기반 Base62**
```python
def id_to_short_code(id):
    # Auto-increment ID를 base62로 변환
    return base62.encode(id)

def create_short_url(long_url):
    # DB에 삽입하고 ID 받기
    id = db.insert(long_url)
    short_code = id_to_short_code(id)

    # short_code 업데이트
    db.update(id, short_code=short_code)

    return short_code
```

**4. 캐싱 전략**
```python
class URLShortener:
    def __init__(self):
        self.cache = redis.Redis()
        self.cache_ttl = 86400  # 24시간

    def get_long_url(self, short_code):
        # 캐시 확인
        cache_key = f"url:{short_code}"
        long_url = self.cache.get(cache_key)

        if long_url:
            return long_url

        # DB 조회
        long_url = db.query(
            "SELECT long_url FROM urls WHERE short_code = ?",
            short_code
        )

        if long_url:
            # 캐시에 저장
            self.cache.setex(cache_key, self.cache_ttl, long_url)

        return long_url

    def create_short_url(self, long_url):
        # 이미 존재하는지 확인 (선택사항)
        cache_key = f"long:{hashlib.md5(long_url.encode()).hexdigest()}"
        short_code = self.cache.get(cache_key)

        if short_code:
            return short_code

        # 새로 생성
        short_code = generate_short_code(long_url)
        db.insert(short_code, long_url)

        # 캐시에 저장
        self.cache.setex(cache_key, self.cache_ttl, short_code)

        return short_code
```

**5. 확장성**

- **Database Sharding**: short_code 기준으로 샤딩
```python
def get_shard_id(short_code):
    # 첫 글자 기준으로 샤딩 (62개 샤드)
    return ord(short_code[0]) % NUM_SHARDS

def get_long_url(short_code):
    shard_id = get_shard_id(short_code)
    db = get_db_connection(shard_id)
    return db.query("SELECT long_url FROM urls WHERE short_code = ?", short_code)
```

- **Read Replica**: 읽기 부하 분산
- **CDN**: 인기 있는 URL은 CDN에 캐싱

**6. Analytics (선택사항)**
```python
def track_click(short_code):
    # 비동기로 클릭 카운트 증가
    kafka.produce('click-events', {
        'short_code': short_code,
        'timestamp': time.time(),
        'ip': request.remote_addr,
        'user_agent': request.headers.get('User-Agent')
    })

# 별도 프로세스에서 집계
class ClickAggregator:
    def process_events(self):
        for event in kafka.consume('click-events'):
            # 실시간 카운터 (Redis)
            redis.incr(f"clicks:{event['short_code']}")

            # 상세 분석 데이터 (Cassandra/ClickHouse)
            analytics_db.insert({
                'short_code': event['short_code'],
                'timestamp': event['timestamp'],
                'country': geoip.lookup(event['ip']),
                'device': parse_user_agent(event['user_agent'])
            })
```

### Medium

**News Feed System 상세 설계**

**요구사항:**
- 사용자가 포스트를 작성하면 팔로워의 피드에 표시
- 팔로우/언팔로우 기능
- 피드는 시간 역순으로 정렬
- 1억 DAU, 평균 300명 팔로우

**1. Fan-out 전략 비교**

**Fan-out on Write (Push Model)**
```python
def create_post(user_id, content):
    # 1. 포스트 저장
    post_id = db.insert_post(user_id, content, timestamp=now())

    # 2. 팔로워 목록 조회
    followers = db.get_followers(user_id)

    # 3. 각 팔로워의 피드에 추가 (비동기)
    for follower_id in followers:
        news_feed_cache.lpush(
            f"feed:{follower_id}",
            post_id
        )

    return post_id

def get_feed(user_id, page=1, page_size=20):
    # 캐시에서 바로 조회
    start = (page - 1) * page_size
    end = start + page_size - 1

    post_ids = news_feed_cache.lrange(
        f"feed:{user_id}",
        start,
        end
    )

    # 포스트 상세 정보 조회
    posts = db.get_posts(post_ids)
    return posts
```

**Fan-out on Read (Pull Model)**
```python
def create_post(user_id, content):
    # 포스트만 저장
    post_id = db.insert_post(user_id, content, timestamp=now())
    return post_id

def get_feed(user_id, page=1, page_size=20):
    # 1. 팔로잉 목록 조회
    following = db.get_following(user_id)

    # 2. 각 사용자의 최근 포스트 조회
    all_posts = []
    for followed_user_id in following:
        posts = db.get_recent_posts(followed_user_id, limit=100)
        all_posts.extend(posts)

    # 3. 시간순 정렬
    all_posts.sort(key=lambda p: p.timestamp, reverse=True)

    # 4. 페이지네이션
    start = (page - 1) * page_size
    return all_posts[start:start + page_size]
```

**Hybrid 접근법 (실제 프로덕션)**
```python
def create_post(user_id, content):
    post_id = db.insert_post(user_id, content, timestamp=now())

    # 1. 활성 사용자에게는 Fan-out on Write
    active_followers = db.get_active_followers(user_id, last_active_minutes=30)
    for follower_id in active_followers:
        news_feed_cache.lpush(f"feed:{follower_id}", post_id)

    # 2. 셀럽리티는 Fan-out 하지 않음
    if db.is_celebrity(user_id):
        # 포스트 ID만 타임라인에 저장
        celebrity_posts_cache.zadd(
            f"celebrity:{user_id}",
            {post_id: timestamp}
        )

    return post_id

def get_feed(user_id, page=1, page_size=20):
    # 1. 일반 팔로우 피드 (캐시)
    normal_posts = news_feed_cache.lrange(f"feed:{user_id}", 0, 99)

    # 2. 셀럽리티 포스트 (실시간 조회)
    celebrity_followings = db.get_celebrity_followings(user_id)
    celebrity_posts = []
    for celeb_id in celebrity_followings:
        posts = celebrity_posts_cache.zrevrange(
            f"celebrity:{celeb_id}",
            0, 99,
            withscores=True
        )
        celebrity_posts.extend(posts)

    # 3. 병합 및 정렬
    all_posts = merge_and_sort(normal_posts, celebrity_posts)

    # 4. 페이지네이션
    start = (page - 1) * page_size
    return all_posts[start:start + page_size]
```

**2. Ranking Algorithm**
```python
def calculate_post_score(post):
    """
    점수 = 시간 점수 + 인기도 점수 + 개인화 점수
    """
    # 시간 점수 (최신일수록 높음)
    time_score = 1.0 / (1 + (now() - post.timestamp) / 3600)

    # 인기도 점수
    popularity_score = (
        post.likes * 1.0 +
        post.comments * 2.0 +
        post.shares * 3.0
    )

    # 개인화 점수
    personalization_score = get_user_affinity(
        current_user_id,
        post.author_id
    )

    total_score = (
        time_score * 0.4 +
        popularity_score * 0.3 +
        personalization_score * 0.3
    )

    return total_score

def get_ranked_feed(user_id, page_size=20):
    # 후보 포스트 조회 (캐시)
    candidate_posts = get_candidate_posts(user_id, limit=500)

    # 점수 계산 및 정렬
    scored_posts = [
        (post, calculate_post_score(post))
        for post in candidate_posts
    ]
    scored_posts.sort(key=lambda x: x[1], reverse=True)

    # 상위 N개 반환
    return [post for post, score in scored_posts[:page_size]]
```

### Hard

**분산 Rate Limiter 상세 설계**

**요구사항:**
- 사용자당 API 호출 제한 (예: 100 req/min)
- 분산 환경에서 동작
- 낮은 지연시간 (<10ms)
- 정확도와 성능 균형

**1. Token Bucket Algorithm**
```python
class TokenBucketRateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client

    def is_allowed(self, user_id, max_tokens=100, refill_rate=100):
        """
        max_tokens: 버킷 최대 크기
        refill_rate: 초당 토큰 재충전 속도
        """
        bucket_key = f"rate_limit:{user_id}"
        now = time.time()

        # Lua 스크립트로 원자적 실행
        lua_script = """
        local bucket_key = KEYS[1]
        local max_tokens = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])

        -- 현재 토큰 수와 마지막 업데이트 시간 조회
        local token_data = redis.call('HMGET', bucket_key, 'tokens', 'last_refill')
        local tokens = tonumber(token_data[1]) or max_tokens
        local last_refill = tonumber(token_data[2]) or now

        -- 시간 경과에 따른 토큰 재충전
        local time_passed = now - last_refill
        tokens = math.min(max_tokens, tokens + (time_passed * refill_rate))

        -- 토큰 사용 가능 여부 확인
        if tokens >= 1 then
            tokens = tokens - 1
            redis.call('HMSET', bucket_key, 'tokens', tokens, 'last_refill', now)
            redis.call('EXPIRE', bucket_key, 3600)
            return 1  -- 허용
        else
            return 0  -- 거부
        end
        """

        result = self.redis.eval(
            lua_script,
            1,
            bucket_key,
            max_tokens,
            refill_rate,
            now
        )

        return bool(result)
```

**2. Sliding Window Log**
```python
class SlidingWindowLogRateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client

    def is_allowed(self, user_id, max_requests=100, window_seconds=60):
        """
        window_seconds 동안 max_requests 초과 여부 확인
        """
        log_key = f"rate_limit:log:{user_id}"
        now = time.time()
        window_start = now - window_seconds

        # Lua 스크립트
        lua_script = """
        local log_key = KEYS[1]
        local window_start = tonumber(ARGV[1])
        local now = tonumber(ARGV[2])
        local max_requests = tonumber(ARGV[3])
        local window_seconds = tonumber(ARGV[4])

        -- 윈도우 이전 요청 삭제
        redis.call('ZREMRANGEBYSCORE', log_key, 0, window_start)

        -- 현재 윈도우 내 요청 수 확인
        local current_requests = redis.call('ZCARD', log_key)

        if current_requests < max_requests then
            -- 새 요청 추가
            redis.call('ZADD', log_key, now, now)
            redis.call('EXPIRE', log_key, window_seconds)
            return 1
        else
            return 0
        end
        """

        result = self.redis.eval(
            lua_script,
            1,
            log_key,
            window_start,
            now,
            max_requests,
            window_seconds
        )

        return bool(result)
```

**3. Sliding Window Counter (최적화)**
```python
class SlidingWindowCounterRateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client

    def is_allowed(self, user_id, max_requests=100, window_seconds=60):
        """
        Fixed Window의 메모리 효율 + Sliding Window의 정확도
        """
        now = time.time()
        current_window = int(now / window_seconds)
        previous_window = current_window - 1

        current_key = f"rate_limit:{user_id}:{current_window}"
        previous_key = f"rate_limit:{user_id}:{previous_window}"

        # 현재 윈도우 카운트
        current_count = int(self.redis.get(current_key) or 0)

        # 이전 윈도우 카운트
        previous_count = int(self.redis.get(previous_key) or 0)

        # 윈도우 오버랩 비율 계산
        elapsed_in_window = now % window_seconds
        weight = 1 - (elapsed_in_window / window_seconds)

        # 가중 평균으로 요청 수 계산
        estimated_count = previous_count * weight + current_count

        if estimated_count < max_requests:
            # 카운터 증가
            pipe = self.redis.pipeline()
            pipe.incr(current_key)
            pipe.expire(current_key, window_seconds * 2)
            pipe.execute()
            return True
        else:
            return False
```

**4. 분산 환경에서의 Rate Limiting**

**중앙집중식 (Redis Cluster)**
```python
class DistributedRateLimiter:
    def __init__(self):
        # Redis Cluster 연결
        self.redis_cluster = redis.RedisCluster(
            host='redis-cluster',
            port=6379
        )

    def is_allowed(self, user_id, limit_config):
        # 모든 노드가 같은 Redis Cluster 사용
        return TokenBucketRateLimiter(
            self.redis_cluster
        ).is_allowed(user_id, **limit_config)
```

**분산 카운팅 (로컬 + 동기화)**
```python
class LocalWithSyncRateLimiter:
    def __init__(self, node_id, total_nodes):
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.local_cache = {}
        self.redis = redis.Redis()

    def is_allowed(self, user_id, max_requests=100):
        # 로컬 할당량
        local_quota = max_requests / self.total_nodes

        # 로컬 카운터 확인
        local_key = f"{user_id}:{int(time.time() / 60)}"
        local_count = self.local_cache.get(local_key, 0)

        if local_count < local_quota:
            self.local_cache[local_key] = local_count + 1

            # 비동기로 글로벌 카운터 업데이트
            self.sync_to_redis(user_id)
            return True
        else:
            # 로컬 할당량 초과 - Redis 확인
            global_count = self.redis.get(f"global:{user_id}")
            return global_count < max_requests

    def sync_to_redis(self, user_id):
        # 주기적으로 로컬 카운트를 Redis에 동기화
        pass
```

**5. API Gateway 통합**
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
rate_limiter = DistributedRateLimiter()

def rate_limit_middleware():
    user_id = request.headers.get('X-User-ID')

    # Rate limit 설정 (tier별로 다를 수 있음)
    tier = get_user_tier(user_id)
    limit_config = {
        'free': {'max_requests': 100, 'window_seconds': 60},
        'pro': {'max_requests': 1000, 'window_seconds': 60},
        'enterprise': {'max_requests': 10000, 'window_seconds': 60}
    }[tier]

    if not rate_limiter.is_allowed(user_id, limit_config):
        # Rate limit 초과
        retry_after = 60  # 초
        return jsonify({
            'error': 'Rate limit exceeded',
            'retry_after': retry_after
        }), 429, {'Retry-After': str(retry_after)}

@app.before_request
def before_request():
    response = rate_limit_middleware()
    if response:
        return response

@app.route('/api/data')
def get_data():
    return jsonify({'data': 'some data'})
```

# 실제 시스템 아키텍처 사례

## 주요 기업 아키텍처

### Twitter

**특징:**
- 3억 MAU (Monthly Active Users)
- 6,000 TPS (Tweets Per Second) 평균
- Read:Write = 100:1

**주요 설계:**
```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
┌──────▼──────────────────┐
│   Load Balancer (ELB)   │
└──────┬──────────────────┘
       │
┌──────▼──────────────────┐
│   API Gateway Layer     │
│  (Rate Limiting, Auth)  │
└──────┬──────────────────┘
       │
       ├─────────┬─────────┬──────────┐
       │         │         │          │
┌──────▼─────┐  │  ┌──────▼──────┐   │
│ Tweet      │  │  │ Timeline    │   │
│ Service    │  │  │ Service     │   │
└──────┬─────┘  │  └──────┬──────┘   │
       │         │         │          │
┌──────▼─────┐  │  ┌──────▼──────┐   │
│ Redis      │  │  │ Redis       │   │
│ (Tweets)   │  │  │ (Timeline)  │   │
└────────────┘  │  └─────────────┘   │
                │                     │
         ┌──────▼──────┐      ┌──────▼──────┐
         │ Search      │      │ User        │
         │ Service     │      │ Service     │
         │ (Elastic)   │      │ (MySQL)     │
         └─────────────┘      └─────────────┘
```

**핵심 전략:**
- Fan-out on Write for active users
- Fan-out on Read for celebrities
- Tweet은 Redis에 캐싱 (Hot data)
- Search는 Elasticsearch
- Timeline은 Redis Sorted Set

### Instagram

**규모:**
- 10억+ 사용자
- 하루 9,500만 장의 사진 업로드
- 42억 개의 좋아요/일

**저장 시스템:**
```python
# 사진 저장 전략
class PhotoStorage:
    def save_photo(self, photo_data, user_id):
        # 1. 고유 ID 생성
        photo_id = self.generate_id()

        # 2. 원본 사진 저장 (S3)
        original_key = f"photos/{photo_id}/original.jpg"
        s3.upload(original_key, photo_data)

        # 3. 썸네일 생성 (비동기)
        self.async_create_thumbnails(photo_id, photo_data)

        # 4. 메타데이터 저장 (Cassandra)
        metadata = {
            'photo_id': photo_id,
            'user_id': user_id,
            'created_at': now(),
            's3_key': original_key,
            'cdn_url': f"cdn.instagram.com/{photo_id}"
        }
        cassandra.insert('photos', metadata)

        # 5. 피드에 Fan-out
        self.fanout_to_followers(user_id, photo_id)

        return photo_id

    def async_create_thumbnails(self, photo_id, photo_data):
        sizes = [(150, 150), (320, 320), (640, 640)]
        for width, height in sizes:
            thumbnail = resize_image(photo_data, width, height)
            key = f"photos/{photo_id}/thumb_{width}x{height}.jpg"
            s3.upload(key, thumbnail)
```

### Netflix

**스트리밍 인프라:**
- Open Connect (자체 CDN)
- AWS 기반 마이크로서비스
- Chaos Engineering (Chaos Monkey)

```python
# 비디오 인코딩 파이프라인
class VideoEncodingPipeline:
    def process_video(self, video_file):
        # 1. 원본 업로드
        video_id = self.upload_to_s3(video_file)

        # 2. 여러 해상도로 인코딩 (병렬)
        encoding_jobs = [
            ('4K', '3840x2160', '25mbps'),
            ('1080p', '1920x1080', '8mbps'),
            ('720p', '1280x720', '5mbps'),
            ('480p', '854x480', '2.5mbps'),
        ]

        job_ids = []
        for quality, resolution, bitrate in encoding_jobs:
            job_id = self.submit_encoding_job(
                video_id, quality, resolution, bitrate
            )
            job_ids.append(job_id)

        # 3. 인코딩 완료 대기
        self.wait_for_jobs(job_ids)

        # 4. CDN에 배포
        self.distribute_to_cdn(video_id)

        # 5. 메타데이터 업데이트
        self.update_catalog(video_id, encoding_jobs)

    def submit_encoding_job(self, video_id, quality, resolution, bitrate):
        # AWS Elemental MediaConvert 사용
        job = {
            'input': f's3://raw-videos/{video_id}',
            'output': f's3://encoded-videos/{video_id}/{quality}',
            'settings': {
                'resolution': resolution,
                'bitrate': bitrate,
                'format': 'HLS'  # HTTP Live Streaming
            }
        }
        return media_convert.submit_job(job)
```

### Uber

**실시간 매칭 시스템:**

```python
class RideMatchingSystem:
    def __init__(self):
        self.geo_index = GeospatialIndex()  # Redis Geo
        self.matching_queue = Queue()

    def find_driver(self, rider_location, ride_request):
        # 1. 주변 드라이버 검색 (5km 반경)
        nearby_drivers = self.geo_index.search_radius(
            rider_location,
            radius_km=5
        )

        # 2. 가용한 드라이버 필터링
        available_drivers = [
            d for d in nearby_drivers
            if d.status == 'available' and d.rating >= 4.0
        ]

        if not available_drivers:
            # 반경 확대 재시도
            return self.find_driver_extended_radius(rider_location)

        # 3. 최적 드라이버 선택
        best_driver = self.select_best_driver(
            available_drivers,
            rider_location,
            ride_request
        )

        # 4. 매칭 요청
        match_id = self.create_match(best_driver.id, ride_request.id)

        # 5. 드라이버에게 푸시 알림
        self.notify_driver(best_driver.id, match_id)

        # 6. 15초 타임아웃 대기
        if self.wait_for_acceptance(match_id, timeout=15):
            return best_driver
        else:
            # 거절 시 다음 드라이버에게 요청
            available_drivers.remove(best_driver)
            return self.find_driver(rider_location, ride_request)

    def select_best_driver(self, drivers, rider_location, ride_request):
        # 점수 = ETA * 0.4 + 평점 * 0.3 + 수락률 * 0.3
        scored_drivers = []

        for driver in drivers:
            eta = self.calculate_eta(driver.location, rider_location)
            score = (
                (1.0 / eta) * 0.4 +
                (driver.rating / 5.0) * 0.3 +
                driver.acceptance_rate * 0.3
            )
            scored_drivers.append((driver, score))

        scored_drivers.sort(key=lambda x: x[1], reverse=True)
        return scored_drivers[0][0]
```

# 추가 학습 리소스

## 데이터베이스 특화

### Database Sharding 전략

```python
# Vertical Sharding (수직 분할) - 테이블별로 분할
class VerticalSharding:
    def __init__(self):
        self.user_db = Database('users')      # 사용자 정보
        self.order_db = Database('orders')     # 주문 정보
        self.product_db = Database('products') # 상품 정보

    def get_user_orders(self, user_id):
        # 여러 DB에서 조회 필요
        user = self.user_db.get_user(user_id)
        orders = self.order_db.get_orders_by_user(user_id)
        return user, orders

# Horizontal Sharding (수평 분할) - 레코드별로 분할
class HorizontalSharding:
    def __init__(self, num_shards=16):
        self.shards = [Database(f'shard_{i}') for i in range(num_shards)]
        self.num_shards = num_shards

    def get_shard(self, user_id):
        # Consistent Hashing
        shard_id = hash(user_id) % self.num_shards
        return self.shards[shard_id]

    def save_user(self, user):
        shard = self.get_shard(user.id)
        shard.insert('users', user)

    def get_user(self, user_id):
        shard = self.get_shard(user_id)
        return shard.query('users', user_id)

# Range-based Sharding (범위 기반)
class RangeSharding:
    def __init__(self):
        self.shard_ranges = [
            (0, 1000000, Database('shard_0')),
            (1000001, 2000000, Database('shard_1')),
            (2000001, 3000000, Database('shard_2')),
        ]

    def get_shard(self, user_id):
        for start, end, shard in self.shard_ranges:
            if start <= user_id <= end:
                return shard
        raise Exception('Shard not found')

# Directory-based Sharding (디렉토리 기반)
class DirectorySharding:
    def __init__(self):
        self.lookup_service = ShardLookupService()

    def get_shard(self, user_id):
        # 룩업 테이블에서 샤드 위치 조회
        shard_id = self.lookup_service.get_shard_for_user(user_id)
        return Database(f'shard_{shard_id}')

    def migrate_user(self, user_id, target_shard):
        # 사용자 데이터를 다른 샤드로 이동
        source_shard = self.get_shard(user_id)
        user_data = source_shard.get_user(user_id)

        target_db = Database(f'shard_{target_shard}')
        target_db.insert('users', user_data)
        source_shard.delete('users', user_id)

        # 룩업 테이블 업데이트
        self.lookup_service.update_shard_mapping(user_id, target_shard)
```

## SQL vs NoSQL 선택 가이드

| 기준 | SQL | NoSQL |
|------|-----|-------|
| 데이터 구조 | 정형화된 스키마 | 유연한 스키마 |
| 확장성 | 수직 확장 (Scale-up) | 수평 확장 (Scale-out) |
| 트랜잭션 | ACID 완벽 지원 | BASE (결과적 일관성) |
| 조인 | 복잡한 조인 지원 | 조인 제한적 (비정규화) |
| 사용 사례 | 은행, 전자상거래 | SNS, 실시간 분석, IoT |
| 예시 | PostgreSQL, MySQL | MongoDB, Cassandra, Redis |

```python
# SQL이 적합한 경우
class BankingSystem:
    """
    요구사항:
    - 강한 일관성 (ACID)
    - 복잡한 쿼리 (JOIN)
    - 스키마 변경 드묾
    """
    def transfer_money(self, from_account, to_account, amount):
        with transaction():
            # 원자적 실행 보장
            db.execute("UPDATE accounts SET balance = balance - ? WHERE id = ?",
                      amount, from_account)
            db.execute("UPDATE accounts SET balance = balance + ? WHERE id = ?",
                      amount, to_account)
            db.execute("INSERT INTO transactions (from, to, amount) VALUES (?, ?, ?)",
                      from_account, to_account, amount)

# NoSQL이 적합한 경우
class SocialMediaFeed:
    """
    요구사항:
    - 높은 쓰기 처리량
    - 수평 확장
    - 유연한 스키마
    - 결과적 일관성 허용
    """
    def create_post(self, user_id, content):
        post = {
            'post_id': generate_id(),
            'user_id': user_id,
            'content': content,
            'timestamp': now(),
            'likes': 0,
            'comments': []
        }

        # Cassandra에 저장 (높은 쓰기 성능)
        cassandra.insert('posts', post)

        # 비동기로 팔로워 피드에 추가
        async_fanout_to_followers(user_id, post['post_id'])
```

## 성능 최적화 체크리스트

### Backend 최적화

- [ ] **Database Query 최적화**
  - Explain Plan 분석
  - 적절한 인덱스 생성
  - N+1 쿼리 문제 해결
  - Connection Pooling 설정

- [ ] **Caching 전략**
  - Redis/Memcached 도입
  - Cache-Aside 패턴 적용
  - Cache Stampede 방지
  - TTL 적절히 설정

- [ ] **비동기 처리**
  - Message Queue 도입 (Kafka, RabbitMQ)
  - 백그라운드 작업 분리
  - Event-driven Architecture

- [ ] **API 최적화**
  - Response 압축 (gzip)
  - Pagination 구현
  - Rate Limiting
  - API Gateway 도입

### Frontend 최적화

- [ ] **번들 크기 최적화**
  - Code Splitting
  - Tree Shaking
  - Lazy Loading

- [ ] **이미지 최적화**
  - WebP 포맷 사용
  - Lazy Loading
  - CDN 활용
  - Responsive Images

- [ ] **렌더링 최적화**
  - SSR (Server-Side Rendering)
  - SSG (Static Site Generation)
  - Critical CSS

### Infrastructure 최적화

- [ ] **CDN 활용**
  - 정적 자원 CDN 배포
  - Edge Caching
  - Geo-distribution

- [ ] **Load Balancing**
  - Health Check 설정
  - Auto Scaling 구성
  - Circuit Breaker 패턴

- [ ] **Monitoring**
  - APM 도구 (New Relic, DataDog)
  - 로그 수집 (ELK Stack)
  - 메트릭 모니터링 (Prometheus, Grafana)

## 추가 학습 자료

- **책**: "System Design Interview" by Alex Xu
- **온라인 코스**: [Educative.io System Design](https://www.educative.io/courses/grokking-the-system-design-interview)
- **유튜브**: [System Design Interview Channel](https://www.youtube.com/@SystemDesignInterview)
- **실습**: [GitHub - System Design Examples](https://github.com/donnemartin/system-design-primer)

# System Design Interview

## Easy

* [Designing Consistent Hashing](practices/DesigningConsistentHashing/DesigningConsistentHashing.md)
* [Designing A Uniqe ID Generator In Distributed Systems](practices/DesigningAUniqeIDGeneratorInDistributedSystems/DesigningAUniqeIDGeneratorInDistributedSystems.md)
* [Designing Real-Time Gaming Leaderboard](practices/DesigningReal-TimeGamingLeaderboard/DesigningReal-TimeGamingLeaderboard.md)
* [Designing CDN](practices/DesigningCDN/DesigningCDN.md)
* [Designing Parking Garrage](practices/DesigningParkingGarrage/DesigningParkingGarrage.md)
* [Designing Hotel Reservation System](practices/DesigningHotelReservationSystem/DesigningHotelReservationSystem-kr.md)
* [Designing Ticketmaster](practices/DesigningTicketmaster/DesigningTicketmaster.md)
* [Designing A URL Shortener](practices/DesigningUrlShorteningService/DesigningUrlShorteningService.md)
* [Designing Pastebin](practices/DesigningPastebin/DesigningPastebin.md)
* [Designing Vending Machine](practices/DesigningVendingMachine/DesigningVendingMachine.md)
* [Designing A Key-Value Store](practices/DesigningAKey-ValueStore/DesigningAKey-ValueStore.md)
* [Designing Distributed Cache](practices/DesigningDistributedCache/DesigningDistributedCache.md)
* [Designing Distributed Job Scheduler](practices/DesigningDistributedJobScheduler/DesigningDistributedJobScheduler.md)
* [Designing Authentication System](practices/DesigningAuthenticationSystem/DesigningAuthenticationSystem.md)
* [Designing Unified Payments Interface (UPI)](practices/DesigningUnifiedPaymentsInterface/DesigningUnifiedPaymentsInterface.md)
* [Designing A News Feed System](practices/DesigningFacebooksNewsfeed/DesigningFacebooksNewsfeed.md)
* [Designing Ad Click Event Aggregation](practices/DesigningAdClickEventAggregation/DesigningAdClickEventAggregation.md)
* [Designing Distributed Email Service](practices/DesigningEmailService/DesigningEmailService.md)
* [Designing Twitter Search](practices/DesigningTwitterSearch/DesigningTwitterSearch.md)

## Medium

* [Designing Instagram](practices/DesigningInstagram/DesigningInstagram.md)
* [Designing Tinder](practices/DesigningTinder/DesigningTinder.md)
* [Designing A Chat System](practices/DesigningFacebookMessenger/DesigningFacebookMessenger.md)
* [Designing Facebook](practices/DesigningFacebook/DesigningFacebook.md)
* [Designing Twitter](practices/DesigningTwitter/DesigningTwitter.md)
* [Designing Reddit](practices/DesigningReddit/DesigningReddit.md)
* [Designing Netflix](practices/DesigningNetflix/DesigningNetflix.md)
* [Designing Youtube](practices/DesigningYoutubeorNetflix/DesigningYoutubeorNetflix.md)
* [Designing Google Search](practices/DesigningGoogleSearch/DesigningGoogleSearch.md)
* [Designing Amazon](practices/DesigningAmazon/DesigningAmazon.md)
* [Designing Spotify](practices/DesigningSpotify/DesigningSpotify.md)
* [Designing TikTok](practices/DesigningTikTok/DesigningTikTok.md)
* [Designing Shopify](practices/DesigningShopify/DesigningShopify.md)
* [Designing Airbnb](practices/DesigningAirbnb/DesigningAirbnb.md)
* [Designing A Search Autocomplete System](practices/DesigningTypeaheadSuggestion/DesigningTypeaheadSuggestion.md)
* [Designing A Rate Limiter](practices/DesigningAnApiRateLimiter/DesigningAnApiRateLimiter.md)
* [Designing Distributed Message Queue](practices/DesigningDistributedMessageQueue/DesigningDistributedMessageQueue.md)
* [Designing Flight Booking System](practices/DesigningFlightBookingSystem/DesigningFlightBookingSystem.md)
* [Designing Online Code Editor](practices/DesigningOnlineCodeEditor/DesigningOnlineCodeEditor.md)
* [Designing Stock Exchange System](practices/DesigningStockExchangeSystem/DesigningStockExchangeSystem.md)
* [Designing Metrics Monitoring and Alerting System](practices/DesigningMetricsMonitoringandAlertingSystem/DesigningMetricsMonitoringandAlertingSystem.md)
* [Designing Notification Service](practices/DesigningNotificationService/DesigningNotificationService.md)
* [Designing Payment System](practices/DesigningPaymentSystem/DesigningPaymentSystem.md)

## Hard

* [Designing Slack](practices/DesigningSlack/DesigningSlack.md)
* [Designing Live Comments](practices/DesigningLiveComments/DesigningLiveComments.md)
* [Designing Distributed Counter](practices/DesigningDistributedCounter/DesigningDistributedCounter.md)
* [Designing Proximity Service](practices/DesigningProximityService/DesigningProximityService.md)
* [Designing Nearby Friends](practices/DesigningNearbyFriends/DesigningNearbyFriends.md)
* [Designing Uber Backend](practices/DesigningUberBackend/DesigningUberBackend.md)
* [Designing Food Delivery App like Doordash](practices/DesigningFoodDeliveryApplikeDoordash/DesigningFoodDeliveryApplikeDoordash.md)
* [Designing Google Docs](practices/DesigningGoogleDocs/DesigningGoogleDocs.md)
* [Designing Google Maps](practices/DesigningGoogleMaps/DesigningGoogleMaps.md)
* [Designing Zoom](practices/DesigningZoom/DesigningZoom.md)
* [Designing Dropbox](practices/DesigningDropbox/DesigningDropbox.md)
* [Designing A Web Crawler](practices/DesigningaWebCrawler/DesigningaWebCrawler.md)
* [Designing Ticket Booking System like BookMyShow](practices/DesigningTicketBookingSystemlikeBookMyShow/DesigningTicketBookingSystemlikeBookMyShow.md)
* [Designing Code Deployment System](practices/DesigningCodeDeploymentSystem/DesigningCodeDeploymentSystem.md)
* [Designing Distributed Cloud Storage like S3](practices/DesigningDistributedCloudStoragelikeS3/DesigningDistributedCloudStoragelikeS3.md)
* [Designing Distributed Locking Service](practices/DesigningDistributedLockingService/DesigningDistributedLockingService.md)
* Designing Digital Wallet

# Scalability Articles

- [How Discord stores trillions of messages](https://discord.com/blog/how-discord-stores-trillions-of-messages)
- [Building In-Video Search](https://netflixtechblog.com/building-in-video-search-936766f0017c)
- [How Canva scaled Media uploads from Zero to 50 Million per Day](https://www.canva.dev/blog/engineering/from-zero-to-50-million-uploads-per-day-scaling-media-at-canva/)
- [How Airbnb avoids double payments in a Distributed Payments System](https://medium.com/airbnb-engineering/avoiding-double-payments-in-a-distributed-payments-system-2981f6b070bb)
- [Stripe’s payments APIs - The first 10 years](https://stripe.com/blog/payment-api-design)
- [Real time messaging at Slack](https://slack.engineering/real-time-messaging/)

# Real World Architecture

| Type | System | Reference(s) |
| -- | -- | -- |
| Data processing | **MapReduce** - Distributed data processing from Google | [research.google.com](http://static.googleusercontent.com/media/research.google.com/zh-CN/us/archive/mapreduce-osdi04.pdf) |
|  | **Spark** - Distributed data processing from Databricks | [slideshare.net](http://www.slideshare.net/AGrishchenko/apache-spark-architecture) |
|  | **Storm** - Distributed data processing from Twitter | [slideshare.net](http://www.slideshare.net/previa/storm-16094009) |
| Data store      | **Bigtable** - Distributed column-oriented database from Google | [harvard.edu](http://www.read.seas.harvard.edu/~kohler/class/cs239-w08/chang06bigtable.pdf) |
|       | **HBase** - Open source implementation of Bigtable | [slideshare.net](http://www.slideshare.net/alexbaranau/intro-to-hbase) |
|       | **[Cassandra](/cassandra/README.md)** - Distributed column-oriented database from Facebook | [slideshare.net](http://www.slideshare.net/planetcassandra/cassandra-introduction-features-30103666) |
|       | **[DynamoDB](/dynamodb/README.md)** - Document-oriented database from Amazon | [harvard.edu](http://www.read.seas.harvard.edu/~kohler/class/cs239-w08/decandia07dynamo.pdf) |
|       | **[MongoDB](/mongodb/README.md)** - Document-oriented database | [slideshare.net](http://www.slideshare.net/mdirolf/introduction-to-mongodb) |
|       | **Spanner** - Globally-distributed database from Google | [research.google.com](http://research.google.com/archive/spanner-osdi2012.pdf) |
|       | **[Memcached](/memcached/README.md)** - Distributed memory caching system | [slideshare.net](http://www.slideshare.net/oemebamo/introduction-to-memcached) |
|       | **[Redis](/redis/README.md)** - Distributed memory caching system with persistence and value types | [slideshare.net](http://www.slideshare.net/dvirsky/introduction-to-redis) |
|       | **Couchbase** - an open-source, distributed multi-model NoSQL document-oriented database | [couchbase.com](https://www.couchbase.com/) |
|       | **[Elasticsearch](/elasticsearch/README.md)** | [Elasticsearch @ TIL](/elasticsearch/README.md) |
| File system     | **Google File System (GFS)** - Distributed file system | [research.google.com](http://static.googleusercontent.com/media/research.google.com/zh-CN/us/archive/gfs-sosp2003.pdf) |
|      | **[Hadoop File System (HDFS)](/hadoop/README.md)** - Open source implementation of GFS | [apache.org](https://hadoop.apache.org/docs/r1.2.1/hdfs_design.html) |
|      | **GlusterFS** - Distributed File System | [GlusterFS](/GlusterFS/README.md) |
| Monitoring      | **[Graylog](/graylog/README.md)** | [Graylog @ TIL](/graylog/README.md) |
|       | **Prometheus** | [Prometheus @ TIL](/prometheus/README.md) |
|       | **[Grafana](/grafana/README.md)** | [Grafana @ TIL](/grafana/README.md) |
| CI/CD           | **[Jenkins](/jenkins/README.md)** | [Jenkins @ TIL](/jenkins/README.md) |
| Misc            | **Chubby** - Lock service for loosely-coupled distributed systems from Google | [research.google.com](http://static.googleusercontent.com/external_content/untrusted_dlcp/research.google.com/en/us/archive/chubby-osdi06.pdf) |
|             | **Dapper** - Distributed systems tracing infrastructure | [research.google.com](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36356.pdf) |
|             | **[Kafka](/kafka/README.md)** - Pub/sub message queue from LinkedIn | [slideshare.net](http://www.slideshare.net/mumrah/kafka-talk-tri-hug) |
|             | **[Zookeeper](/zookeeper/README.md)** - Centralized infrastructure and services enabling synchronization | [slideshare.net](http://www.slideshare.net/sauravhaloi/introduction-to-apache-zookeeper) |
|             | **ØMQ** - a high-performance asynchronous messaging library, aimed at use in distributed or concurrent applications. | [zeromq.org](http://zeromq.org/) |
|             | **[etcd](/etcd/README.md)** - A distributed, reliable key-value store for the most critical data of a distributed system. | [etcd docs](https://coreos.com/etcd/docs/latest/) |
|             | **Mosquitto** - An open source MQTT broker. [MQTT](/mqtt/README.md) is a Standard for IoT Messaging |  |
|             | **Netty** - Netty is a NIO client server framework. | |

# Company Architectures

| Company        | Reference(s) |
| -------------- | -- |
| 배달의 민족 | [배달의민족 msa](wooahan_msa.md) |
| Amazon         | [Amazon architecture](http://highscalability.com/amazon-architecture) |
| Cinchcast      | [Producing 1,500 hours of audio every day](http://highscalability.com/blog/2012/7/16/cinchcast-architecture-producing-1500-hours-of-audio-every-d.html) |
| DataSift       | [Realtime datamining At 120,000 tweets per second](http://highscalability.com/blog/2011/11/29/datasift-architecture-realtime-datamining-at-120000-tweets-p.html)|
| DropBox        | [How we've scaled Dropbox](https://www.youtube.com/watch?v=PE4gwstWhmc) |
| ESPN           | [Operating At 100,000 duh nuh nuhs per second](http://highscalability.com/blog/2013/11/4/espns-architecture-at-scale-operating-at-100000-duh-nuh-nuhs.html) |
| Google         | [Google architecture](http://highscalability.com/google-architecture) |
| Instagram      | [14 million users, terabytes of photos](http://highscalability.com/blog/2011/12/6/instagram-architecture-14-million-users-terabytes-of-photos.html)<br/>[What powers Instagram](http://instagram-engineering.tumblr.com/post/13649370142/what-powers-instagram-hundreds-of-instances) |
| Justin.tv      | [Justin.Tv's live video broadcasting architecture](http://highscalability.com/blog/2010/3/16/justintvs-live-video-broadcasting-architecture.html) |
| Facebook       | [Scaling memcached at Facebook](https://cs.uwaterloo.ca/~brecht/courses/854-Emerging-2014/readings/key-value/fb-memcached-nsdi-2013.pdf)<br/>[TAO: Facebook’s distributed data store for the social graph](https://cs.uwaterloo.ca/~brecht/courses/854-Emerging-2014/readings/data-store/tao-facebook-distributed-datastore-atc-2013.pdf)<br/>[Facebook’s photo storage](https://www.usenix.org/legacy/event/osdi10/tech/full_papers/Beaver.pdf) |
| Flickr         | [Flickr architecture](http://highscalability.com/flickr-architecture) |
| Mailbox        | [From 0 to one million users in 6 weeks](http://highscalability.com/blog/2013/6/18/scaling-mailbox-from-0-to-one-million-users-in-6-weeks-and-1.html) |
| Pinterest      | [From 0 To 10s of billions of page views a month](http://highscalability.com/blog/2013/4/15/scaling-pinterest-from-0-to-10s-of-billions-of-page-views-a.html)<br/>[18 million visitors, 10x growth, 12 employees](http://highscalability.com/blog/2012/5/21/pinterest-architecture-update-18-million-visitors-10x-growth.html) |
| Playfish       | [50 million monthly users and growing](http://highscalability.com/blog/2010/9/21/playfishs-social-gaming-architecture-50-million-monthly-user.html) |
| PlentyOfFish   | [PlentyOfFish architecture](http://highscalability.com/plentyoffish-architecture) |
| Salesforce     | [How they handle 1.3 billion transactions a day](http://highscalability.com/blog/2013/9/23/salesforce-architecture-how-they-handle-13-billion-transacti.html) |
| Stack Overflow | [Stack Overflow architecture](http://highscalability.com/blog/2009/8/5/stack-overflow-architecture.html) |
| TripAdvisor    | [40M visitors, 200M dynamic page views, 30TB data](http://highscalability.com/blog/2011/6/27/tripadvisor-architecture-40m-visitors-200m-dynamic-page-view.html) |
| Tumblr         | [15 billion page views a month](http://highscalability.com/blog/2012/2/13/tumblr-architecture-15-billion-page-views-a-month-and-harder.html) |
| Twitter        | [Making Twitter 10000 percent faster](http://highscalability.com/scaling-twitter-making-twitter-10000-percent-faster)<br/>[Storing 250 million tweets a day using MySQL](http://highscalability.com/blog/2011/12/19/how-twitter-stores-250-million-tweets-a-day-using-mysql.html)<br/>[150M active users, 300K QPS, a 22 MB/S firehose](http://highscalability.com/blog/2013/7/8/the-architecture-twitter-uses-to-deal-with-150m-active-users.html)<br/>[Timelines at scale](https://www.infoq.com/presentations/Twitter-Timeline-Scalability)<br/>[Big and small data at Twitter](https://www.youtube.com/watch?v=5cKTP36HVgI)<br/>[Operations at Twitter: scaling beyond 100 million users](https://www.youtube.com/watch?v=z8LU0Cj6BOU) |
| Uber           | [How Uber scales their real-time market platform](http://highscalability.com/blog/2015/9/14/how-uber-scales-their-real-time-market-platform.html) |
| WhatsApp       | [The WhatsApp architecture Facebook bought for $19 billion](http://highscalability.com/blog/2014/2/26/the-whatsapp-architecture-facebook-bought-for-19-billion.html) |
| YouTube        | [YouTube scalability](https://www.youtube.com/watch?v=w5WVu624fY8) |

# Company Engineering Blog

* [Airbnb Engineering](http://nerds.airbnb.com/)
* [Atlassian Developers](https://developer.atlassian.com/blog/)
* [Autodesk Engineering](http://cloudengineering.autodesk.com/blog/)
* [AWS Blog](https://aws.amazon.com/blogs/aws/)
* [Bitly Engineering Blog](http://word.bitly.com/)
* [Box Blogs](https://www.box.com/blog/engineering/)
* [Cloudera Developer Blog](http://blog.cloudera.com/blog/)
* [Dropbox Tech Blog](https://tech.dropbox.com/)
* [Engineering at Quora](http://engineering.quora.com/)
* [Ebay Tech Blog](http://www.ebaytechblog.com/)
* [Evernote Tech Blog](https://blog.evernote.com/tech/)
* [Etsy Code as Craft](http://codeascraft.com/)
* [Facebook Engineering](https://www.facebook.com/Engineering)
* [Flickr Code](http://code.flickr.net/)
* [Foursquare Engineering Blog](http://engineering.foursquare.com/)
* [GitHub Engineering Blog](http://githubengineering.com/)
* [Google Research Blog](http://googleresearch.blogspot.com/)
* [Groupon Engineering Blog](https://engineering.groupon.com/)
* [Heroku Engineering Blog](https://engineering.heroku.com/)
* [Hubspot Engineering Blog](http://product.hubspot.com/blog/topic/engineering)
* [High Scalability](http://highscalability.com/)
* [Instagram Engineering](http://instagram-engineering.tumblr.com/)
* [Intel Software Blog](https://software.intel.com/en-us/blogs/)
* [Jane Street Tech Blog](https://blogs.janestreet.com/category/ocaml/)
* [LinkedIn Engineering](http://engineering.linkedin.com/blog)
* [Microsoft Engineering](https://engineering.microsoft.com/)
* [Microsoft Python Engineering](https://blogs.msdn.microsoft.com/pythonengineering/)
* [Netflix Tech Blog](http://techblog.netflix.com/)
* [Paypal Developer Blog](https://devblog.paypal.com/category/engineering/)
* [Pinterest Engineering Blog](http://engineering.pinterest.com/)
* [Quora Engineering](https://engineering.quora.com/)
* [Reddit Blog](http://www.redditblog.com/)
* [Salesforce Engineering Blog](https://developer.salesforce.com/blogs/engineering/)
* [Slack Engineering Blog](https://slack.engineering/)
* [Spotify Labs](https://labs.spotify.com/)
* [Twilio Engineering Blog](http://www.twilio.com/engineering)
* [Twitter Engineering](https://engineering.twitter.com/)
* [Uber Engineering Blog](http://eng.uber.com/)
* [Yahoo Engineering Blog](http://yahooeng.tumblr.com/)
* [Yelp Engineering Blog](http://engineeringblog.yelp.com/)
* [Zynga Engineering Blog](https://www.zynga.com/blogs/engineering)
