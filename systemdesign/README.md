- [Abstract](#abstract)
- [References](#references)
- [Materials](#materials)
- [Articles](#articles)
- [Design Tools](#design-tools)
- [System Extentions](#system-extentions)
- [Estimations](#estimations)
  - [Major Items](#major-items)
  - [Numbers](#numbers)
  - [Powers of two](#powers-of-two)
  - [Latency numbers every programmer should know](#latency-numbers-every-programmer-should-know)
  - [Availability](#availability)
  - [Time](#time)
- [Fundamentals](#fundamentals)
  - [System Design Interview Process](#system-design-interview-process)
  - [IP (Internet Protocol)](#ip-internet-protocol)
  - [OSI](#osi)
  - [Network](#network)
    - [TCP](#tcp)
    - [UDP](#udp)
    - [RPC](#rpc)
    - [REST (REpresentational State Transfer) API](#rest-representational-state-transfer-api)
    - [REST API DESIGN](#rest-api-design)
    - [RPC VS REST](#rpc-vs-rest)
    - [HTTP 1.x vs HTTP 2.0](#http-1x-vs-http-20)
    - [HTTP Flow](#http-flow)
  - [Domain Name System](#domain-name-system)
  - [Load Balancing](#load-balancing)
  - [Caching](#caching)
  - [Caching Strategies](#caching-strategies)
  - [Distributed Caching](#distributed-caching)
  - [CDN (Content Delivery Network)](#cdn-content-delivery-network)
  - [Proxy](#proxy)
  - [Availability](#availability-1)
    - [Availability Patterns](#availability-patterns)
    - [The Nine's of availability](#the-nines-of-availability)
  - [Failover](#failover)
  - [Fault Tolerance](#fault-tolerance)
  - [Distributed System](#distributed-system)
  - [Software Design Principle](#software-design-principle)
  - [Read Heavy vs Write Heavy](#read-heavy-vs-write-heavy)
  - [Performance vs Scalability](#performance-vs-scalability)
  - [Latency vs Throughput](#latency-vs-throughput)
  - [Availability vs Consistency](#availability-vs-consistency)
    - [CAP (Consistency Availability Partition tolerance) Theorem](#cap-consistency-availability-partition-tolerance-theorem)
    - [PACELC (Partitioning Availability Consistency Else Latency Consistency)](#pacelc-partitioning-availability-consistency-else-latency-consistency)
  - [Consistency Patterns](#consistency-patterns)
  - [Database](#database)
  - [ACID](#acid)
  - [Sharding](#sharding)
  - [Application layer](#application-layer)
  - [Service Mesh](#service-mesh)
  - [Service Discovery](#service-discovery)
  - [Event Driven Architecture](#event-driven-architecture)
  - [Event Sourcing Architecture](#event-sourcing-architecture)
  - [Command and Query Responsibility Segregation (CQRS)](#command-and-query-responsibility-segregation-cqrs)
  - [API Gateway](#api-gateway)
  - [gRPC](#grpc)
  - [GraphQL](#graphql)
  - [REST vs GraphQL vs gRPC](#rest-vs-graphql-vs-grpc)
  - [Long Polling](#long-polling)
  - [Web Socket](#web-socket)
  - [Server-Sent Events (SSE)](#server-sent-events-sse)
  - [Long-Polling vs WebSockets vs Server-Sent Events](#long-polling-vs-websockets-vs-server-sent-events)
  - [Geohashing](#geohashing)
  - [Quadtrees](#quadtrees)
  - [Circuit Breaker](#circuit-breaker)
  - [Rate Limiting](#rate-limiting)
  - [Asynchronism](#asynchronism)
  - [Message Queue](#message-queue)
  - [Message Queue VS Event Streaming Platform](#message-queue-vs-event-streaming-platform)
  - [Security](#security)
    - [WAF (Web Application Fairewall)](#waf-web-application-fairewall)
    - [XSS (Cross Site Scripting)](#xss-cross-site-scripting)
    - [CSRF (Cross Site Request Forgery)](#csrf-cross-site-request-forgery)
    - [XSS vs CSRF](#xss-vs-csrf)
    - [CORS (Cross Origin Resource Sharing)](#cors-cross-origin-resource-sharing)
    - [PKI (Public Key Infrastructure)](#pki-public-key-infrastructure)
  - [SSL/TLS](#ssltls)
  - [mTLS](#mtls)
  - [Distributed Primary Key](#distributed-primary-key)
  - [Idempotency](#idempotency)
  - [80/20 rule](#8020-rule)
  - [70% Capacity model](#70-capacity-model)
  - [SLA, SLO, SLI](#sla-slo-sli)
  - [Optimistic Lock vs Pessimistic Lock](#optimistic-lock-vs-pessimistic-lock)
  - [Disaster Recovery (DR)](#disaster-recovery-dr)
  - [OAuth 2.0](#oauth-20)
  - [OpenID Connect (OIDC)](#openid-connect-oidc)
  - [Single Sign-On (SSO)](#single-sign-on-sso)
  - [Control Plane, Data Plane, Management Plane](#control-plane-data-plane-management-plane)
  - [Distributed Transaction](#distributed-transaction)
  - [Observability](#observability)
  - [Load Test](#load-test)
  - [Incidenct](#incidenct)
  - [Consistent Hashing](#consistent-hashing)
  - [Database Index](#database-index)
  - [SQL vs NoSQL](#sql-vs-nosql)
  - [Hadoop](#hadoop)
  - [MapReduce](#mapreduce)
  - [Consensus Algorithm](#consensus-algorithm)
  - [Paxos](#paxos)
  - [Gossip protocol](#gossip-protocol)
  - [Raft](#raft)
  - [Chubby](#chubby)
  - [Configuration Management Database (CMDB)](#configuration-management-database-cmdb)
  - [A/B Test](#ab-test)
  - [Actor Model](#actor-model)
  - [Reactor vs Proactor](#reactor-vs-proactor)
  - [Data Lake](#data-lake)
  - [Data Warehouse](#data-warehouse)
  - [Data Lakehouse](#data-lakehouse)
  - [API Security](#api-security)
  - [Batch Processing vs Stream Processing](#batch-processing-vs-stream-processing)
  - [HeartBeat](#heartbeat)
  - [Bloom Filter](#bloom-filter)
  - [Distributed Locking](#distributed-locking)
  - [Distributed Tracing](#distributed-tracing)
  - [Checksum](#checksum)
- [System Design Interview](#system-design-interview)
  - [Easy](#easy)
  - [Medium](#medium)
  - [Hard](#hard)
- [Scalability Articles](#scalability-articles)
- [Real World Architecture](#real-world-architecture)
- [Company Architectures](#company-architectures)
- [Company Engineering Blog](#company-engineering-blog)
- [MSA (Micro Service Architecture)](#msa-micro-service-architecture)
- [Cloud Design Patterns](#cloud-design-patterns)
- [Enterprise Integration Patterns](#enterprise-integration-patterns)
- [DDD](#ddd)
- [Architecture](#architecture)

----

# Abstract

Describe system design.

<p align="center">
  <img src="http://i.imgur.com/jj3A5N8.png"/>
  <br/>
</p>

# References

* [system-design-101 | github](https://github.com/ByteByteGoHq/system-design-101)
  * System design one page
* [Low-Latency Engineering Tech Talks | p99conf](https://www.p99conf.io/on-demand/)
  * low-latency, high-performance distributed computing challenges
* [neetcode System Design Courses](https://neetcode.io/courses)
* [system-design | github](https://github.com/karanpratapsingh/system-design)
  * [html](https://www.karanpratapsingh.com/courses/system-design)
  * [ebook](https://leanpub.com/systemdesign)
* [Machine Learning System Design | amazon](https://www.amazon.com/Machine-Learning-System-Design-Interview/dp/1736049127/ref=sr_1_1?crid=1G3JKBVICKM0Q&keywords=machine+learning+system+design&qid=1676028725&sprefix=machine+learning+system+desi%2Caps%2C475&sr=8-1)
  * [links | github](https://github.com/ByteByteGoHq/ml-bytebytego)
* [SwirlAI Newsletter](https://swirlai.substack.com/)
  * Usually about ML Ops
* System Design Interview – An insider's guide by Alex
  * [blog archives](https://blog.bytebytego.com/archive)
  * [oneline ebook](https://bytebytego.com/courses/system-design-interview/)
  * [system design | discord](https://discord.com/channels/805287783107264553/805880778671128586)
  * [System Design Interview – An insider's guide, Second Edition](https://www.amazon.com/System-Design-Interview-insiders-Second/dp/B08CMF2CQF)
    * [links](https://github.com/alex-xu-system/bytecode/blob/main/system_design_links.md)
  * [System Design Interview – An Insider's Guide: Volume 2](https://www.amazon.com/System-Design-Interview-Insiders-Guide/dp/1736049119)
    * [links](https://github.com/alex-xu-system/bytebytego/blob/main/system_design_links_vol2.md)https://blog.bytebytego.com/p/how-does-https-work-episode-6?s=r
* [Design Microservices Architecture with Patterns & Principles | udemy](https://www.udemy.com/course/design-microservices-architecture-with-patterns-principles/)
  * 유료이다. 실용적인 패턴만 정리. distributed transaction 설명
  * [slide @ github](https://github.com/mehmetozkaya/Design-Microservices-Architecture-with-Patterns-Principles)
  * [src @ github](https://github.com/aspnetrun/run-aspnetcore-microservices)
* [A pattern language for microservices](https://microservices.io/patterns/index.html)
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
* [System Deisgn Primer](https://github.com/donnemartin/system-design-primer)
  - [System Design Primer Mindmap of XMind](https://drive.google.com/file/d/190mY7w8ea1jYpoUFJPsd8rk8YgaSvrAP/view?usp=sharing)
* [Grokking the System Design Interview](https://www.educative.io/collection/5668639101419520/5649050225344512)
  - 유료 시스템 디자인 인터뷰
  - [System Design @ blogspot](https://learnsystemdesign.blogspot.com/)
  - [src](https://github.com/Jeevan-kumar-Raj/Grokking-System-Design)
* [System Design The Big Archives](https://blog.bytebytego.com/p/free-system-design-pdf-158-pages?s=r)
  * [pdf](https://bytebyte-go.s3.amazonaws.com/ByteByteGo_LinkedIn_PDF.pdf)
  
# Materials

* [Awesome System Design Resources | github](https://github.com/ashishps1/awesome-system-design-resources)
* [대규모 소프트웨어 패턴 강좌 업데이트](https://architecture101.blog/2023/02/11/welcome_2_pattern_worlds/?fbclid=IwAR2lVvRIidYW_CnAs4tTExStad1T4pq54V2ySCtMMlBS0DTfxD0_NQdGW9Y&mibextid=Zxz2cZ)
  * 아키텍처 전문 강사의 커리큘럼, POSA1, POSA2, POSA3, AOSA, Cloud+MSA
* [System Design Resources | github](https://github.com/InterviewReady/system-design-resources)
* [mobile system design](https://github.com/weeeBox/mobile-system-design)
* [The Software Architecture Chronicles](https://herbertograca.com/2017/07/03/the-software-architecture-chronicles/)
  * Software Architecture 의 역사
  * [src](git@github.com:hgraca/explicit-architecture-php.git)
    * php 
* [DreamOfTheRedChamber/system-design @ github](https://github.com/DreamOfTheRedChamber/system-design)
  * 킹왕짱
* [Grokking the Object Oriented Design Interview](https://www.educative.io/collection/5668639101419520/5692201761767424)
  - 유료 OOD 인터뷰 
  - [Grokking the Object Oriented Design Interview @ github](https://github.com/tssovi/grokking-the-object-oriented-design-interview)
* [Here are some of the favorite posts on HighScalability...](http://highscalability.com/all-time-favorites/)
  * great case studies
* [FullStack cafe](https://www.fullstack.cafe/)
* [AWS @ TIL](/aws/README.md)
* [Mastering Chaos - A Netflix Guide to Microservices](https://www.youtube.com/watch?v=CZ3wIuvmHeM)
* [cracking the coding interview](http://www.crackingthecodinginterview.com/)
* [Designing Data-Intensive Applications](https://dataintensive.net/)
- [Azure Cloud Patterns](https://docs.microsoft.com/en-us/azure/architecture/patterns/)
  - [infographic](https://techcommunity.microsoft.com/t5/educator-developer-blog/getting-started-with-azure-design-patterns-and-azure-arm-quick/ba-p/378609?lightbox-message-images-378609=94372i2134F7E89D8BF452)
- [AWS Architect](https://aws.amazon.com/ko/architecture/)
- [GCP Solutions](https://cloud.google.com/solutions/)

# Articles

* [Spotify System Architecture](https://medium.com/interviewnoodle/spotify-system-architecture-6bb418db6084)

# Design Tools

* [Database diagram](https://databasediagram.com/app)
* [draw.io](https://www.draw.io/)
  * architecture 를 포함한 다양한 diagram tool
* [diagrams.net](https://www.diagrams.net/)
  * draw.io standalone application download
* [Google Drawings](https://docs.google.com/drawings/)
* [cloudcraft](https://cloudcraft.co/app)
  * aws architcture diagram tool
* [webwhiteboard](https://www.webwhiteboard.com/)
  * web white board for system design interview 

# System Extentions

System extention factors:

* RDBMS, NoSQL
* Vertical scaling
* Horizontal scaling
* Load balancing
* Replication
* Cache
* Static assets in CDN
* Stateless Web Servers
* Multiple data centers
* Asynchronism, message queue
* logging, metric
* Sharding, Parititioning
* Micro Services
* CI/CD

# Estimations

## Major Items

QPS, peak QPS, Storage, Cache, Number of servers, etc...

## Numbers

> * [Names of large numbers](https://en.wikipedia.org/wiki/Names_of_large_numbers)
> * [SI 기본 단위](https://ko.wikipedia.org/wiki/SI_%EA%B8%B0%EB%B3%B8_%EB%8B%A8%EC%9C%84)
> * SI 는 System International 의 줄임말로 국제 단위
  
| Value | short-scale | SI-symbol | SI-prefix |
| ----- | ----------- | --------- | --------- |
| 10^3  | Thousand    | K         | Killo-    |
| 10^6  | Million     | M         | Mega-     |
| 10^9  | Billion     | G         | Giga-     |
| 10^12 | Trillion    | T         | Tera-     |
| 10^15 | Quadrillion | P         | Peta-     |
| 10^18 | Quintillion | E         | Exa-      |
| 10^21 | Sextillion  | Z         | Zeta-     |
| 10^24 | Septillion  | Y         | Yota-     |

## Powers of two

| Power | Approximate value | Full name | Short name|
|--|--|--|--|
| 10 | 1 Thousand | 1 Kilobyte | 1 KB |
| 20 | 1 Million | 1 Megabyte | 1 MB |
| 30 | 1 Billion | 1 Gigabyte | 1 GB |
| 40 | 1 Trillion | 1 Terabyte | 1 TB |
| 50 | 1 Quadrillion | 1 Petabyte | 1 PB |

## Latency numbers every programmer should know

> [Latency Numbers Every Programmer Should Know](https://gist.github.com/jboner/2841832)

| Name | Time | |
|--|--|--|
| L1 cache reference | 0.5 ns | |
| Branch mispredict | 5 ns | |
| L2 cache reference | 7 ns |
| Mutex lock/unlock | 100 ns |
| Main memory reference | 100 ns |
| Compress 1K bytes with Zippy | 10,000 ns = 10 us |
| Send 1 KB bytes over 1 Gbps network | 20,000 ns = 20 us |
| Read 1 MB sequentially from memory | 250,000 ns = 250 us |
| Round trip within same datacenter  | 5000,000 ns = 500 us |
| Disk seek | 10,000,000 ns = 10 ms |
| Read 1 MB sequentially from 1 Gbps | 10,000,000 ns = 10 ms | |
| Read 1 MB sequentially from disk | 30,000,000 ns = 30 ms | |
| Send packet CA->Netherlands->CA | 150,000,000 ns = 150 ms | |

```
Notes
-----
1 ns = 10^-9 seconds
1 us = 10^-6 seconds = 1,000 ns
1 ms = 10^-3 seconds = 1,000 us = 1,000,000 ns
```

![](img/latency_numbers_every_programmer_should_know.png)

## Availability

> * [Uptime and downtime with 99.9 % SLA](https://uptime.is/)

| Availability % | Downtime per year |
|---:|---:|
| 99% | 3.65 days |
| 99.9% | 8.77 hours |
| 99.99% | 52.60 mins |
| 99.999% | 5.26 mins |
| 99.9999% | 31.56 secs |

## Time

| years | days | hours |    mins |       secs |
| ----: | ---: | ----: | ------: | ---------: |
|     1 |  365 | 8,760 | 525,600 | 31,536,000 |
|       |    1 |    24 |   1,440 |     86,400 |
|       |      |     1 |      60 |      3,600 |
|       |      |       |       1 |         60 |

# Fundamentals

## System Design Interview Process

* Outline use cases, constraints, and assumptions
* Create a high level design
* Design core components
* Scale the design

## IP (Internet Protocol)

IP is the primary protocol of the network layer (Layer 3) in the OSI (Open
Systems Interconnection) model, and it is responsible for identifying and
routing data packets between devices connected to a network.

**IPv4**(Internet Protocol version 4): It is the most widely used IP version,
which uses a 32-bit addressing scheme, allowing for a maximum of approximately
4.3 billion unique IP addresses. IPv4 addresses are usually represented in a
dotted-decimal notation, such as 192.168.1.1.

[ipv4](/network/README.md#ipv4)

**IPv6**(Internet Protocol version 6): Developed as a replacement for IPv4 due
to the exhaustion of available IP addresses, IPv6 uses a 128-bit addressing
scheme, providing a vastly greater number of unique IP addresses. IPv6 addresses
are typically represented in colon-separated hexadecimal notation, such as
2001:0db8:85a3:0000:0000:8a2e:0370:7334.

[ipv6](/network/README.md#ipv6)

## OSI

The OSI (Open Systems Interconnection) model is a conceptual framework used to
standardize the functions of a telecommunication or computing system, regardless
of its underlying internal structure and technology. It was developed by the
International Organization for Standardization (ISO) in the 1970s and 1980s.

![](/network/Osi-model-7-layers.png)

[osi 7 layer](/network/README.md#osi-7-layer)

## Network

* [Network | TIL](/network/README.md)

----

### TCP

[TCP | TIL](/network/README.md#tcp)

### UDP

[UDP | TIL](/network/README.md#udp)

### RPC

**RPC** (Remote Procedure Call) is a communication protocol that allows one
computer or program to execute and request procedures or functions on another
computer or program, as if they were local. It enables distributed communication
between client and server systems in a network, allowing a client program to
request a service from a server program, usually located on a different machine,
without needing to understand network details.

### REST (REpresentational State Transfer) API 

* [1) Rest API란? @ edwith](https://www.edwith.org/boostcourse-web/lecture/16740/)
* [HTTP 응답코드 메소드 정리 GET, POST, PUT, PATCH, DELETE, TRACE, OPTIONS](https://javaplant.tistory.com/18)

----

2000 년도에 로이 필딩 (Roy Fielding) 의 박사학위 논문에서 최초로 소개되었다.
REST 형식의 API 를 말한다.

로이 필딩은 현재 공개된 REST API 라고 불리우는 것은 대부분 REST API 가
아니다라고 말한다. REST API 는 다음과 같은 것들을 포함해야 한다고 한다.

* client-server
* stateless
* cache
* uniform interface
* layered system
* code-on-demand (optional)

HTTP 를 사용하면 uniform interface 를 제외하고는 모두 만족 한다. uniform
interface 는 다음을 포함한다.

* 리소스가 URI로 식별되야 합니다.
* 리소스를 생성,수정,추가하고자 할 때 HTTP메시지에 표현을 해서 전송해야 합니다.
* 메시지는 스스로 설명할 수 있어야 합니다. (Self-descriptive message)
* 애플리케이션의 상태는 Hyperlink를 이용해 전이되야 합니다.(HATEOAS)

위의 두가지는 이미 만족하지만 나머지 두가지는 HTTP 로 구현하기 어렵다. 예를 들어
HTTP BODY 에 JSON 을 포함했을 때 HTTP message 스스로 body 의 내용을 설명하기란
어렵다. 그리고 웹 게시판을 사용할 때, 리스트 보기를 보면, 상세보기나 글쓰기로
이동할 수 있는 링크가 있습니다. 상세보기에서는 글 수정이나 글 삭제로 갈 수 있는
링크가 있습니다. 이렇게 웹 페이지를 보면, 웹 페이지 자체에 관련된 링크가
있는것을 알 수 있는데 이를 HATEOAS (Hypermedia As The Engine Of Application
State) 라고 한다. HATEOAS 를 API 에서 제공하는 것은 어렵다.

결국 HTTP 는 REST API 의 uniform interface 스타일 중 self-descriptive message,
HATEOAS 를 제외하고 대부분의 특징들이 구현되어 있다고 할 수 있다. 그래서 REST
API 대신 HTTP API 또는 WEB API 라고 한다.

### REST API DESIGN

* [15 fundamental tips on REST API design | medium](https://medium.com/@liams_o/15-fundamental-tips-on-rest-api-design-9a05bcd42920)
* [EP53: Design effective and safe APIs](https://blog.bytebytego.com/p/ep53-design-effective-and-safe-apis)

----

[restapi | TIL](/restapi/README.md)

### RPC VS REST

| Operation                       | RPC       | REST     |
| ------------------------------- | ---------------------------------------------------- | ----------------------------------------- |
| Signup | **POST** /signup    | **POST** /persons                                            |
| Resign  | **POST** /resign<br/>{<br/>"personid": "1234"<br/>}  | **DELETE** /persons/1234  |
| Read a person  | **GET** /readPerson?personid=1234  | **GET** /persons/1234 |
| Read a person’s items list      | **GET** /readUsersItemsList?personid=1234 | **GET** /persons/1234/items  |
| Add an item to a person’s items | **POST** /addItemToUsersItemsList<br/>{<br/>"personid": "1234";<br/>"itemid": "456"<br/>} | **POST** /persons/1234/items<br/>{<br/>"itemid": "456"<br/>} |
| Update an item | **POST** /modifyItem<br/>{<br/>"itemid": "456";<br/>"key": "value"<br/>} | **PUT** /items/456<br/>{<br/>"key": "value"<br/>} |
| Delete an item | **POST** /removeItem<br/>{<br/>"itemid": "456"<br/>} | **DELETE** /items/456 |

### HTTP 1.x vs HTTP 2.0

* [HTTP | TIL](/HTTP/README.md)

### HTTP Flow

* [HTTP Flow | TIL](/HTTP/README.md#http-flow)

## Domain Name System

> - [DNS란 무엇입니까? | DNS 작동 원리 | cloudflare](https://www.cloudflare.com/ko-kr/learning/dns/what-is-dns/)
> - [Domain Name System (DNS)](https://www.karanpratapsingh.com/courses/system-design/domain-name-system)
> - [How does the Domain Name System (DNS) lookup work? | bytebytego](https://blog.bytebytego.com/p/how-does-the-domain-name-system-dns?s=r)

----

DNS(도메인 네임 시스템)은 인터넷 전화번호부로, 웹 브라우저에서 도메인 이름을 IP
주소로 변환하여 인터넷 자원을 로드할 수 있게 합니다. DNS 확인 과정은 호스트
이름을 IP 주소로 변환하는 것이며, DNS 리커서, 루트 이름 서버, TLD 이름 서버,
권한 있는 이름 서버와 같은 다양한 하드웨어 구성 요소가 포함되어 있습니다. DNS
조회 프로세스는 통상 8단계를 거쳐 웹 브라우저가 웹 페이지를 요청할 수 있게 되며,
이 과정에서 DNS 캐싱이 데이터를 임시 저장하여 성능과 신뢰성을 높이는 데 도움이
됩니다. DNS 확인자는 DNS 조회의 첫 번째 중단점으로 기능합니다.

DNS server 의 종류는 다음과 같이 3 가지가 있다.

* **Root Name Server**
  * redirect request to TLD Server. ex) redirect request to TLD server for ".com"
* **TLD (Top Level Domain) Name Server**
  * redirect request to Authoritative Server. ex) redirect request to Authoritative Name Server for "google.com"
* **Authoritative Name Server**
  * redirect request to Real Server. ex) redirect request to Real Name Server for "www.google.com"

DNS lookups on average take almost 20-120 milliseconds.

<p align="center">
  <img src="http://i.imgur.com/IOyLj4i.jpg"/>
  <br/>
  <i><a href=http://www.slideshare.net/srikrupa5/dns-security-presentation-issa>Source: DNS security presentation</a></i>
</p>

다음은 주요 DNS Record 들이다.

* **NS record (name server)** - Specifies the DNS servers for your
  domain/subdomain.
* **MX record (mail exchange)** - Specifies the mail servers for accepting
  messages.
* **A record (address)** - Points a name to an IP address.
* **CNAME (canonical)** - Points a name to another name or `CNAME` (example.com
  to www.example.com) or to an `A` record.

| name      | type  | value      |
| --------- | ----- | ---------- |
| a.foo.com | A     | 192.1.1.15 |
| b.foo.com | CNAME | a.foo.com  |

* [Online DNS Record Viewer](http://dns-record-viewer.online-domain-tools.com/)

## Load Balancing

> - [로드 밸런싱이란 무엇인가요? | aws](https://aws.amazon.com/ko/what-is/load-balancing/)
> - [Introduction to modern network load balancing and proxying](https://blog.envoyproxy.io/introduction-to-modern-network-load-balancing-and-proxying-a57f6ff80236)
> - [What is load balancing](https://avinetworks.com/what-is-load-balancing/)
> - [Introduction to architecting systems](https://lethain.com/introduction-to-architecting-systems-for-scale/)
> - [Load balancing](https://en.wikipedia.org/wiki/Load_balancing_(computing))

<p align="center">
  <img src="http://i.imgur.com/h81n9iK.png"/>
  <br/>
  <i><a href=http://horicky.blogspot.com/2010/10/scalable-system-design-patterns.html>Source: Scalable system design patterns</a></i>
</p>

로드 밸런서(load balancer)는 네트워크 트래픽이나 워크로드를 여러 서버나 리소스에
분산시켜 리소스 사용률을 최적화하고 성능을 향상시키며 신뢰성을 개선하는 네트워킹
장치 또는 소프트웨어입니다. 들어오는 요청을 사용 가능한 리소스에 고르게
분산함으로써 로드 밸런서는 단일 서버의 과부하를 방지하고, 성능 저하, 응답 시간
지연 및 시스템 오류가 발생하는 것을 방지합니다.

로드 밸런서는 고가용성, 확장성 및 장애 허용성이 중요한 환경에서 일반적으로
사용됩니다. 로드 밸런싱에는 여러 가지 유형과 기법이 사용되며, 다음과 같습니다:

- **라운드 로빈(Round Robin)**: 요청이 사용 가능한 서버에 순차적으로 분산되며,
  마지막 서버에 도달한 후에는 다시 첫 번째 서버부터 순환합니다.
- **최소 연결(Least Connections)**: 활성 연결이 가장 적은 서버에 요청이 전송되며,
  이는 로드를 균등하게 분배하는 데 도움이 됩니다.
- **가중 분배(Weighted Distribution)**: 서버에는 용량에 따라 다른 가중치가 부여되며,
  요청은 가중치에 비례하여 분배됩니다.

로드 밸런서는 OSI 모델의 다양한 레벨에서 작동할 수 있습니다:

- **레이어 4 (전송 계층) 로드 밸런싱**: TCP 또는 UDP 헤더를 기반으로 로드
  밸런싱이 수행되며, 패킷의 내용을 검사하지 않고 트래픽을 분배합니다. 이 방법은
  레이어 7 로드 밸런싱보다 빠르지만 유연성이 떨어집니다.
- **레이어 7 (애플리케이션 계층) 로드 밸런싱**: 요청의 내용, 예를 들어 URL,
  쿠키, HTTP 헤더 등을 기반으로 로드 밸런싱이 수행됩니다. 이 방법은 더 많은
  유연성을 제공하며, 필요한 처리에 따라 요청을 특정 서버로 바로 보내는 고급
  라우팅 결정이 가능해집니다.

온프레미스 하드웨어 및 가상 어플라이언스 외에도 Amazon Web Services(AWS), Google
Cloud Platform(GCP), Microsoft Azure와 같은 클라우드 제공업체들은 관리되는 로드
밸런싱 서비스를 제공하여 애플리케이션에 쉽게 통합할 수 있습니다.

## Caching

- [System Design — Caching](https://medium.com/must-know-computer-science/system-design-caching-acbd1b02ca01)

<p align="center">
  <img src="http://i.imgur.com/Q6z24La.png"/>
  <br/>
  <i><a href=http://horicky.blogspot.com/2010/10/scalable-system-design-patterns.html>Source: Scalable system design patterns</a></i>
</p>

**Use-Cases**

* Client caching
* CDN caching
* Web server caching
* Database caching
* Application caching
* Caching at the database query level
* Caching at the object level

## Caching Strategies

- [Top 5 Caching Patterns](https://newsletter.systemdesign.one/p/caching-patterns)
- [Top caching strategies | bytebytego](https://blog.bytebytego.com/p/top-caching-strategies)

**Read**

* Cache-Aside
  * 응용프로그램이 직접 cache를 제어하여 읽는다.
  * `cache` 에 읽기 요청하고 값이 있으면 응답한다. 없으면 `SOR(System Of Record)` 
    로 부터 읽어들인 값으로 응답한다. 그리고 `cache` 를 update 한다.
    ```python
    # reading values
    v = cache.get(k)
    if (v == null) {
      v = sor.get(k)
      cache.put(k, v)
    }
    ```
* Read-Through
  * `cache` 에 읽기 요청하면 `cache` 가 판단해서 자신이 가지고 있는 값 혹은
    `SOR(system of record)` 로 부터 읽어들인 값을 응답으로 전달한다.

**Write**

* Write-Around
  * 응용프로그램이 직접 cache를 제어하여 쓴다.
  * `SOR(System Of Record)` 쓴다. 그리고 `cache` 를 update 한다.
    ```py
    # writing values
    v = newV
    sor.put(k, v)
    cache.put(k, v)
    ```
* Write-Through
  * `cache` 에 쓰기 요청하면 `cache` 가 판단해서 `SOR(system of record)` 에
    쓰고 자신을 갱신한다.

* Write-Behind (Write-Back)
  * `cache` 에 쓰기 요청하면 일단 자신을 갱신하고 요청에 응답한후
    `SOR` 을 갱신한다. `SOR` 갱신이 완료되기 전에 요청에 빠르게 응답한다.

**Cache eviction policies**

* First In First Out (FIFO)
* Last In First Out (LIFO)
* Least Recently Used (LRU)
* Most Recently Used (MRU)
* Least Frequently Used (LFU)
* Random Replacement (RR)    

## Distributed Caching

- [Distributed Caching](https://redis.com/glossary/distributed-caching/)

캐싱은 데이터 요청 속도를 높이기 위해 자주 액세스하는 데이터 또는 계산을
저장하고 검색하는 기술입니다. 캐시에 데이터를 임시로 저장함으로써, 시스템은 원래
소스에서 동일한 데이터를 가져오는 데 필요한 시간과 자원을 줄일 수 있으며, 성능
향상과 지연 시간 감소를 이룰 수 있습니다.

캐싱은 기본적으로 로컬 캐싱과 분산 캐싱 두 가지로 분류할 수 있습니다. 로컬
캐싱은 단일 기계 또는 단일 응용프로그램 내에서 데이터를 저장하는 것을 의미하며,
데이터 검색이 한 대의 기계로 제한되거나 데이터량이 상대적으로 작은 경우에
사용됩니다. 로컬 캐싱의 예로는 브라우저 캐시나 응용 프로그램 수준 캐시가
있습니다. 분산 캐싱은 여러 기계 또는 노드, 종종 네트워크에서 데이터를 저장하는
것으로, 여러 서버 간에 확장되어야 하거나 지리적으로 분산되어 있는 애플리케이션에
필수적입니다. 분산 캐싱은 원래 데이터 소스가 원격이거나 과부하가 있는 경우에도
데이터가 필요한 곳 근처에 사용 가능하도록 합니다.

예를 들어 초당 수천 개의 요청을 받는 전자 상거래 웹 사이트를 생각해 볼 때, 로컬
캐싱에만 의존하는 경우 제품 상세 정보를 웹 사이트가 호스팅되는 서버에 저장할 수
있습니다. 그러나 트래픽이 증가하거나 웹 사이트가 다른 지역에서 접근되는 경우
이러한 접근 방식으로 병목 현상이 발생할 수 있습니다. 반면에 분산 캐싱을
사용하면, 제품 상세 정보가 다른 지역에 위치한 여러 캐시 서버에 저장됩니다.
이렇게 하면 사용자가 웹 사이트에 액세스할 때 가장 가까운 캐시 서버에서 제품
정보를 검색하여 응답 시간을 빠르게 하고 사용자 경험을 향상시킬 수 있습니다.

**분산 캐싱**은 **로컬 캐싱**의 한계를 해결하여 네트워크 내 여러 머신 또는 노드
간에 데이터를 저장함으로써 확장성, 내결함성, 성능 향상을 제공합니다. 분산 캐싱의
적용 예로, 전 세계 소비자들이 멀티 컨틴넌트에서 접속하는 글로벌 온라인 소매
업체가 있습니다.

분산 캐싱 시스템의 주요 구성 요소는 캐시 서버입니다. 캐시 서버는 여러 기계 또는
노드에 걸쳐 임시 데이터를 저장하여 필요한 위치에서 데이터를 사용할 수 있도록
합니다. 각 캐시 서버는 독립적으로 작동할 수 있으며, 서버 장애 발생 시 요청을
다른 서버로 리디렉션하여 고 가용성과 내결함성을 보장할 수 있습니다.

분산 캐싱 솔루션에는 [Redis](/redis/README.md),
[Memcached](/memcached/README.md), Hazelcast, [Apache Ignite](/ignite/README.md)
등이 있습니다. 이러한 솔루션은 각각 고유한 기능과 장점을 제공하며, 다양한
애플리케이션 요구에 맞게 사용됩니다. 분산 캐싱 구현에는 적합한 캐싱 솔루션
선택부터 분산 환경에서의 구성 및 배포에 이르기까지 여러 단계가 포함됩니다. 또한
캐시 효율을 극대화하기 위해 적절한 캐시 삭제 정책 및 데이터 일관성 유지 등의
캐시 관리 방법을 사용하는 것이 중요합니다.

요약하면, 분산 캐싱은 고성능, 확장성 및 실시간 데이터 액세스를 요구하는 현대
애플리케이션에 중요한 솔루션으로 부상했습니다. 분산 캐싱을 통해 자주 액세스하는
데이터를 여러 서버에 저장하여 주 데이터 소스에 가해지는 부담을 줄임으로써 빠른
데이터 검색 및 향상된 사용자 경험을 보장할 수 있습니다.

분산 캐싱(distributed caching)과 리모트 캐싱(remote caching)은 유사한
개념이지만, 약간의 차이가 있습니다.

분산 캐싱은 여러 기계 또는 노드에 데이터를 저장하여 확장성, 고성능 및 내결함성을
제공하는 캐싱 방식입니다. 분산 캐싱은 대규모 및 지리적으로 분산된 애플리케이션에
적합하며, 주로 데이터를 여러 지역 또는 서버에 복제하여 처리 시간을 단축시키고
사용자 경험을 개선합니다.

반면에 리모트 캐싱은 애플리케이션 서버와 다른 위치에 있는 별도의 캐싱 서버에
데이터를 저장하는 캐싱 방법을 지칭합니다. 리모트 캐싱은 로컬 캐싱이나
애플리케이션 서버에 데이터를 저장하는 것보다 효율적인 경우에 사용됩니다. 리모트
캐싱을 사용하면 애플리케이션 서버의 리소스를 절약하고 처리 시간을 단축시킬 수
있습니다.

결론적으로, 분산 캐싱은 데이터를 여러 머신이나 노드에 저장하는 캐싱 방식의
일반적인 용어로 사용되며, 리모트 캐싱은 애플리케이션 서버와 별개의 위치에서
데이터를 캐싱하는 방식에 가까운 개념입니다. 한편, 분산 캐싱은 리모트 캐싱의
확장된 형태로 간주될 수도 있습니다.

## CDN (Content Delivery Network)

- [What is a content delivery network (CDN)? | How do CDNs work? | CloudFlare](https://www.cloudflare.com/ko-kr/learning/cdn/what-is-a-cdn/)

<p align="center">
  <img src="http://i.imgur.com/h9TAuGI.jpg"/>
  <br/>
  <i><a href=https://www.creative-artworks.eu/why-use-a-content-delivery-network-cdn/>Source: Why use a CDN</a></i>
</p>

**콘텐츠 전송 네트워크 (CDN)**는 사용자에게 웹 콘텐츠와 기타 디지털 자산을 보다
효율적으로 전달하기 위해 설계된 분산 서버(엣지 서버 또는 노드라고도 함)의
시스템입니다. CDN은 이미지, 비디오, 웹 페이지, 스크립트 및 스타일시트와 같은
콘텐츠를 다양한 지역에 위치한 여러 서버에서 캐싱 및 저장함으로써 작동합니다.
사용자가 리소스를 요청하면, 콘텐츠는 단일 중앙 집중식 서버가 아니라 가장 가까운
또는 최적의 성능을 가진 서버로부터 전달되어 빠른 로드 시간, 개선된 성능 및
감소된 대역폭 사용이 보장됩니다.

- Push CDN : Push CDN에서는 콘텐츠 소유자가 콘텐츠를 CDN 서버에 업로드 또는
  "푸시" 합니다. 이후 CDN 서버는 콘텐츠의 원본 역할을 하고 사용자에게 직접
  콘텐츠를 전달합니다. 업데이트 또는 새로운 콘텐츠가 있을 경우, 콘텐츠 소유자는
  다시 CDN 서버에 변경 사항을 푸시해야 합니다. 푸시(CDN)은 콘텐츠 업데이트가
  자주 발생하지 않거나 원본 서버에서 엄격하게 제어해야하는 경우에 적합합니다.
- Push CDN의 장점:
  - 콘텐츠 만료 및 버전 관리에 대한 더 나은 제어
  - CDN에 콘텐츠를 한 번만 푸시하기 때문에 원본 서버의 부하 감소
- Push CDN의 단점:
  - CDN에 콘텐츠를 업로드하고 업데이트하는 것에 대한 수동 관리 필요
  - 접근 빈도가 낮은 콘텐츠를 저장함으로써 CDN 서버의 저장 공간을 더 많이 소비할 수 있음
- Pull CDN: Pull CDN에서는 콘텐츠가 원본 서버에 남아 있고, CDN 서버는 사용자가
  처음으로 요청할 때 원본에서 콘텐츠를 "끌어옵니다" (pull). CDN 서버는 콘텐츠를
  캐시하고, 이후 요청에 대해 사용자에게 캐싱된 버전을 제공합니다. 캐싱된
  콘텐츠가 만료되거나 오래된 경우, CDN에 의해 원본 서버에서 다시 자동으로
  끌어옵니다. Pull CDN은 콘텐츠 업데이트가 자주 발생하는 동적 웹사이트 및
  애플리케이션에 적합합니다.
- Pull CDN의 장점:
  - URL 또는 DNS 레코드를 CDN을 가리키도록 변경하는 것만으로 쉽게 설정 가능
  - 원본 서버와 콘텐츠를 자동으로 업데이트하고 동기화
- Pull CDN의 단점:
  - 사용자가 원본 서버에서 콘텐츠를 끌어올 때 처음 요청에 대해 약간 높은 지연 시간을 경험할 수 있음
  - 캐시 미스 또는 콘텐츠 업데이트 중 원본 서버의 부하를 증가시킬 수 있음

## Proxy

- [What is a Proxy Server? How does it work?](https://www.fortinet.com/resources/cyberglossary/proxy-server)
- [Apache2 설치 (Ubuntu 16.04)](https://lng1982.tistory.com/288)
  
-----

<p align="center">
  <img src="http://i.imgur.com/n41Azff.png"/>
  <br/>
  <i><a href=https://upload.wikimedia.org/wikipedia/commons/6/67/Reverse_proxy_h2g2bob.svg>Source: Wikipedia</a></i>
  <br/>
</p>

![](img/foward_reverse_proxy.png)

프록시에는 포워드 프록시 와 리버스 프록시 두 가지가 있다.

**포워드 프록시**

포워드 프록시는 클라이언트와 외부 서버 사이에 위치해 로컬 네트워크에서 동작한다.
클라이언트의 요청을 받아 대신 자원을 요청한 후 응답을 클라이언트에 전달한다.
인터넷 접근 제어, 컨텐츠 필터링, 트래픽 모니터링이 필요한 경우에 주로 사용된다.
포워드 프록시의 일반적인 사용 사례는 다음과 같다.

- 익명성: 클라이언트가 자신의 IP 주소를 드러내지 않고 인터넷 자원에 접근할 수
  있다. 대신 프록시 서버의 IP가 사용된다.
- 보안: 조직은 컨텐츠 필터링, URL 차단, 모니터링을 구현하여 내부 보안 정책을
  준수할 수 있다.
- 캐싱: 포워드 프록시는 자주 접근하는 페이지를 캐싱하여 여러 클라이언트에게
  제공할 수 있어 외부 서버의 부하를 줄이고 응답 시간을 개선한다.

**리버스 프록시**

리버스 프록시는 외부 클라이언트와 내부 서버 사이에 위치하며, 하나 이상의
서버로의 게이트웨이 역할을 한다. 들어오는 요청을 처리하고 적절한 백엔드 서버로
전달한다. 리버스 프록시는 백엔드 서버의 성능, 보안, 로드 밸런싱 향상을 위해
사용된다. 리버스 프록시의 주요 사용 사례로는 다음과 같다.

- 로드 밸런싱: 들어오는 요청을 여러 백엔드 서버간에 분산시켜 부하를 나누어
  효율적인 무중단 서비스를 보장한다. SSL 종료: 리버스 프록시 수준에서 SSL 암호화
  및 복호화를 처리하여 백엔드 서버의 부담을 줄이고 성능을 개선한다.
- 캐싱: 클라이언트에게 정적 컨텐츠를 저장하고 제공하여 백엔드 서버의 부하를
  줄이고 응답 시간을 개선한다.
- 보안: 백엔드 서버를 외부 클라이언트의 직접 노출로부터 보호하여 취약점의 위험을
  줄이고 DDoS 공격을 완화한다.

## Availability

### Availability Patterns

* Fail-over
  * Active-passive
    * LB 가 active 와 passive 를 health check 한다. acitve 에 장애가 발생하면 passive 를 active 시킨다.
  * Active-active
    * active 를 여러개 운용하기 때문에 load 가 분산된다. DNS 가 모든 active 의 IP 를 알아야할 수도 있다.
* Disadvanges of Fail-over 
  * Active 가 passive 에 data 를 replication 하기 전에 장애가 발생하면 일부 data 를 유실할 수 있다.
* Replication
  * Master-slave replication
  * Master-master replication

### The Nine's of availability

* availability 는 uptime 혹은 downtime 을 percent 단위로 표현된다. 예를 들어 99.9% (three 9s) 혹은 99.99% (four 9s) 등으로 표기한다.

| Availability (Percent) | Downtime (Year) | Downtime (Month) | Downtime (Week) |
|--|--|--|--|
| 90% (one nine) | 36.53 days | 72 hours | 16.8 hours |
| 99% (two nines) | 3.65 days | 7.20 hours | 1.68 hours |
| 99.9% (three nines) | 8.77 hours | 43.8 minutes | 10.1 minutes |
| 99.99% (four nines) | 52.6 minutes | 4.32 minutes | 1.01 minutes |

## Failover

- [Failover](https://avinetworks.com/glossary/failover/)

**Failover 정의**

Failover란 신뢰할 수 있는 백업 시스템으로 자동으로 전환되는 능력을 의미합니다.
Failover는 주요 시스템 구성 요소가 실패했을 때 오류를 줄이거나 제거하며
사용자에게 부정적인 영향을 줄이기 위해 예비 운영 모드로 전환해야 합니다.

데이터베이스 서버, 시스템 또는 기타 하드웨어 구성 요소, 서버, 네크워크 등의 중복
또는 예비를 사용하여 이전에 활성화된 버전이 비정상 종료되거나 실패했을 때 대체할
수 있어야 합니다. Failover는 재해 복구에 필수적이기 때문에 예비 컴퓨터 서버
시스템 및 기타 백업 기술 자체도 실패하지 않아야 합니다.

Switchover는 비슷한 작업을 수행하지만, failover와 달리 자동으로 수행되지 않고
인간의 개입이 필요합니다. 대부분의 컴퓨터 시스템은 자동 failover 솔루션에 의해
백업됩니다.

**FAQ**

- Failover란 무엇인가?
  - 서버용 failover 자동화에는 하트비트 케이블을 사용하여 서버 쌍을 연결합니다.
    이차 서버는 하트비트 또는 펄스가 계속되는 한 휴식 상태로 있습니다. 
  - 그러나 기본 실패 서버에서 수신하는 펄스에 변경이 있으면 이차 서버는
    인스턴스를 시작하고 기본 작업을 인수합니다. 또한 데이터 센터 또는 기술자에게
    메시지를 보내 기본 서버를 다시 온라인 상태로 전환하도록 요청합니다.
  - 일부 시스템은 대신 데이터 센터 또는 기술자에게 알리고 한정된 동의 설정인
    보조 서버로 수동 변경을 요청합니다.
- SQL Server Failover 클러스터란 무엇인가?
  - SQL 서버 failover 클러스터는 공유 데이터 스토리지와 NAS(네트워크 결합
    스토리지) 또는 SAN을 통한 다중 네트워크 연결을 포함하여 잠재적 포인트가
    없어야 합니다.
- DHCP Failover란 무엇인가?
  - DHCP failover 설정은 두 개 이상의 DHCP 서버를 사용하여 같은 주소 풀을
    관리하도록 합니다. 이를 통해 네트워크 중단 시 백업 DHCP 서버가 다른 서버를
    지원하고, 해당 풀과 관련된 임대 할당 작업을 모두 공유 할 수 있습니다.
- DNS Failover란 무엇인가?
  - DNS(Domain Name System)은 IP 주소와 사람이 읽을 수 있는 호스트 이름 간의
    변환을 돕는 프로토콜입니다. DNS failover는 네트워크 서비스 또는 웹 사이트가
    사용 중지되는 동안 작동합니다.
- 애플리케이션 서버 Failover란 무엇인가?
  - 응용 프로그램 서버 Failover는 여러 서버가 실행되는 애플리케이션을 보호하는
    전략입니다. 이러한 애플리케이션 서버는 다른 서버에서 실행되어야 하지만
    최소한 고유한 도메인 이름을 가져야 합니다. 애플리케이션 서버 로드 밸런싱은
    종종 failover 클러스터 모범 사례에 따라 단계를 수행합니다.
- Failover 테스트란 무엇인가?
  - Failover 테스트는 서버의 failover 기능을 검증하는 방법입니다. 다시말해
    시스템의 가용 리소스를 실패한 서버에서 복구로 할당할 수 있는 용량을
    테스트합니다.

## Fault Tolerance

- [What is fault tolerance, and how to build fault-tolerant systems](https://www.cockroachlabs.com/blog/what-is-fault-tolerance/)

2020년 11월 25일, AWS의 US-East-1 지역이 중요한 인터넷 서비스 일부가 중단된 큰
문제를 겪었다. 이로 인해 고객들의 제품에 대한 신뢰가 침근해졌다. 이러한 사례들로
인해 fault tolerance(결함 허용)가 현대 애플리케이션 구조의 핵심 부분이 되었다.

결함 허용이란 시스템이 기능상의 손실 없이 오류와 중단을 처리할 수 있는 능력을
말한다. 결함 허용은 여러 가지 방식으로 이루어질 수 있으며, 다음과 같은 방법들이
일반적으로 사용된다: 동일한 작업을 수행할 수 있는 다중 하드웨어 시스템, 다중
인스턴스의 소프트웨어, 백업 전원 공급원 등.

결함 허용과 고가용성은 기술적으로 정확히 같은 것은 아니지만, 실제로 두 가지는
밀접하게 연결되어 있으며, 결함 허용 시스템이 없으면 고가용성을 달성하기 어렵다.

결함 허용 시스템 구축은 보다 복잡하고 비용이 더 많이 든다. 애플리케이션에 필요한
결함 허용 수준을 평가하고 시스템을 그에 따라 구축해야 한다. 결함 허용 시스템
설계시 목표는 정상 작동과 우아한 저하 두 가지로 나눌 수 있다. 정상 작동을
목표로하는 경우, 시스템 구성 요소가 실패하거나 오프라인 상태가 되더라도
애플리케이션이 온라인과 완전한 기능을 유지해야 한다. 우아한 저하를 목표로 하는
경우, 오류 및 중단이 발생하여도 애플리케이션 전체가 완전히 중단되지 않도록
기능이 저하되어 사용자 경험이 영향을 받을 수 있다.

결함 허용 시스템을 구축하는 데 드는 비용도 고려해야 한다. 결함 허용 수준을 높게
설정하려면 더 많은 비용이 발생하지만, 동시에 결함 허용 수준이 높지 않을 때
생기는 비용 역시 고려해야 한다. 손실된 수익, 평판 훼손, 엔지니어링 시간의 손실,
팀의 사기 저하와 인력 유지 및 채용에 대한 비용 등이 그러한 비용에 해당한다.

결함 허용 구조의 예로는 클라우드 기반 다중 지역 구조를 사용하는 것이 있다.
이러한 구조에서 애플리케이션 계층과 데이터베이스 계층 모두에서 노드, 사용 가능
영역 또는 지역 실패가 발생하더라도 애플리케이션의 가용성을 유지할 수 있다.
이렇게 결함 허용기능을 통해 시스템의 다양한 층에서 안정성을 확보할 수 있다.

## Distributed System

[Distributed System | TIL](/distributedsystem/README.md)

## Software Design Principle

[Design Principle | TIL](/designprinciple/README.md)

## Read Heavy vs Write Heavy

* Read Heavy Service
  * Traffic 의 대부분이 Read Operation 이라면 Cache 를 사용한다. Read Replica 도 좋다.
* Write Heavy Service
  * Traffic 의 대부분이 Write Operation 이라면 Distributed Storagae 를 사용한다. 성능과 비용면에서 좋다.
  * [Cassandra](/cassandra/README.md) 혹은 [Scylla](/scylla/README.md) 가 좋다.

## Performance vs Scalability

performance 의 문제가 있다면 single user 가 느린 시스템을 경험할 것이다.
scalability 의 문제가 있다면 single user 가 빠른 시스템을 경험할 지라도 multi
user 는 느린 시스템을 경험할 수 있다???

## Latency vs Throughput

- [처리량과 지연 시간의 차이점은 무엇인가요? | aws](https://aws.amazon.com/ko/compare/the-difference-between-throughput-and-latency/)

Latency 는 어떤 action 을 수행하고 결과를 도출하는데 걸리는 시간이다. Throughput
은 단위 시간당 수행하는 액션 혹은 결과의 수이다.

## Availability vs Consistency

### CAP (Consistency Availability Partition tolerance) Theorem

- [CAP Theorem for Databases: Consistency, Availability & Partition Tolerance](https://www.bmc.com/blogs/cap-theorem/)
- [CAP Theorem @ medium](https://medium.com/system-design-blog/cap-theorem-1455ce5fc0a0)
- [The CAP Theorem](https://teddyma.gitbooks.io/learncassandra/content/about/the_cap_theorem.html)

----

![](/aws/img/1_rxTP-_STj-QRDt1X9fdVlA.jpg)

Brewer's theorem 이라고도 한다. Distributed System 은
**Consistency, Availability, Partition Tolerance** 중 2 가지만 만족할 수 있다. 2
가지를 만족시키기 위해 1 가지를 희생해야 한다는 의미와 같다.

CAP 이론은 분산 시스템에서 다음 중 세 가지 중 두 가지 속성만 선택할 수 있음을 말합니다:

- **Consistency** (일관성): 모든 노드가 데이터의 가장 최신 버전을 갖고 있다.
- **Availability** (가용성): 모든 노드가 항상 데이터를 읽고 쓸 수 있다.
- **Partition tolerance** (분할 허용성): 데이터는 여러 노드에 분할되어 저장되며,
  일부 노드가 실행되지 않거나 손상되더라도 시스템의 전체 기능에 영향을 주지
  않는다.

다음은 2가지를 만족하는 경우의 예이다.

- **CA** (Consistency & Availability)를 만족하는 예: 전통적인 단일 노드
  데이터베이스는 일관성과 가용성을 만족하지만, 예를 들어 하드웨어 장애에 의해
  전체 시스템이 다운되는 문제가 발생하면 분할 허용성에 실패할 수 있습니다.
- **AP** (Availability & Partition tolerance)를 만족하는 예: DynamoDB,
  Cassandra와 같은 분산 데이터베이스는 시스템을 네트워크 파티션이 발생할 때
  가용성이 유지되고 시스템이 계속 작동할 수 있도록 설계되었습니다. 이러한
  시스템에서 데이터는 복수의 노드에 저장되어 일정 시간 동안 일관성이 무시될 수
  있지만, 최종적으로 데이터의 일관성이 보장됩니다.
- **CP** (Consistency & Partition tolerance)를 만족하는 예: Google Spanner,
  ZooKeeper 등의 시스템은 애플리케이션의 데이터 세트를 분할되든 나누어지지 않든
  일관성을 보장하는데 초점을 두고 있습니다. 일관성과 분할 허용성을 유지하면서
  가용성을 희생하는 경우가 발생할 수 있습니다. 예를 들어, 일부 노드에 장애가
  발생하면 성능 손실이나 요청 거부와 같은 여러 가용성 문제가 발생할 수 있습니다.

각 시스템은 적절한 필요요건에 따라 CAP 이론의 특성을 선택하며, CA, AP, CP 속성
중 두 가지를 만족하게 됩니다.

### PACELC (Partitioning Availability Consistency Else Latency Consistency)

* [CAP Theorem, 오해와 진실](http://eincs.com/2013/07/misleading-and-truth-of-cap-theorem/)

----

![](/aws/img/truth-of-cap-theorem-pacelc.jpg)

시스템이 Partitioning 상황 즉 네트워크 장애 상황일 때는 Availability 혹은
Consistency 중 하나를 추구하고 일반적인 상황일 때는 Latency 혹은 Consistency 중
하나를 추구하라는 이론이다. 

이것을 다시 한번 풀어보면 이렇다. 네트워크 장애 상황일 때 클라이언트는 일관성은
떨어져도 좋으니 일단 데이터를 받겠다 혹은 일관성있는 데이터 아니면 에러를
받겠다는 말이다. 네트워크 장애가 아닌 보통의 상황일 때 클라이언트는 일관성은
떨어져도 빨리 받겠다 혹은 일관성있는 데이터 아니면 늦게 받겠다는 말이다.

* HBase 는 PC/EC 이다. 네트워크 장애상황일 때 무조건 일관성있는 데이터를 보내고
  보통의 상황일 때도 무조건 일관성있는 데이터를 보낸다. 한마디로 일관성
  성애자이다.
* [Cassandra](/cassandra/README.md) 는 PA/EL 이다. 일관성은 별로 중요하지 않다. 네트워크 장애상황일 때
  일관성은 떨어져도 데이터를 일단 보낸다. 보통의 상황일 때 역시 일관성은
  떨어져도 좋으니 일단 빨리 데이터를 보낸다.

## Consistency Patterns

- [Consistency Patterns](https://systemdesign.one/consistency-patterns/)
- [Eventual vs Strong Consistency in Distributed Databases](https://hackernoon.com/eventual-vs-strong-consistency-in-distributed-databases-282fdad37cf7)

* Weak consistency
  * write operation 후에 그 값을 read 할 수 있다고 장담할 수 없다.
  * memcached 가 해당된다.
* Eventual consistency
  * write operation 후에 시간이 걸리기는 하지만 그 값을 read 할 수 있다.
  * DNS, email 이 해당된다.
* Strong consistency
  * write operation 후에 그 값을 바로 read 할 수 있다.
  * RDBMS

## Database

<p align="center">
  <img src="http://i.imgur.com/Xkm5CXz.png"/>
  <br/>
  <i><a href=https://www.youtube.com/watch?v=w95murBkYmU>Source: Scaling up to your first 10 million users</a></i>
</p>

* RDBMS
  * ACID - set of properties of relational database transactions
    * Atomicity(원자성) - Each transaction is all or nothing
    * Consistency(일관성) - Any transaction will bring the database from one valid state to another
    * Isolation(고립성) - Executing transactions concurrently has the same results as if the transactions were executed serially
    * Durability(영속성) - Once a transaction has been committed, it will remain so.
  * Master-slave replication
  * Master-Master replication
* Federation
  * 수직분할 이라고도 한다. 테이블별로 partitioning 한다.

<p align="center">
  <img src="http://i.imgur.com/U3qV33e.png"/>
  <br/>
  <i><a href=https://www.youtube.com/watch?v=w95murBkYmU>Source: Scaling up to your first 10 million users</a></i>
</p>

* Sharding
  * 수평분할 이라고도 한다. 하나의 테이블을 레코드별로 partitioning 한다.
  * [consistent hashing | TIL](/consistenthasing/README.md)

<p align="center">
  <img src="http://i.imgur.com/wU8x5Id.png"/>
  <br/>
  <i><a href=http://www.slideshare.net/jboner/scalability-availability-stability-patterns/>Source: Scalability, availability, stability, patterns</a></i>
</p>

* Denormalization
  * [Normalization | TIL](/normalization/README.md)
* SQL Tuning
* NoSQL
  * Key-value store
  * Document store
  * Wide solumn store
  * Graph database
* Schema Design
  * Document DB (embedded data model, normalized data model)
    * Schema Design in MongoDB vs Schema Design in MySQL](https://www.percona.com/blog/2013/08/01/schema-design-in-mongodb-vs-schema-design-in-mysql/)
  * [RDBMS Schema Design](/rdbmsschemadesign/README.md)

## ACID

[ACID](/database/README.md#acid)

## Sharding

- [Database Sharding: Concepts and Examples](https://www.mongodb.com/features/database-sharding-explained#)

애플리케이션의 성장에 따라 데이터베이스는 병목 현상의 원인이 될 수 있다. 이
문제를 해결하기 위해 데이터베이스 샤딩(Database sharding)이라는 방법을 사용할 수
있다. 샤딩은 하나의 데이터셋을 여러 데이터베이스에 분산시키는 방법으로, 이를
통해 더 큰 데이터셋을 작은 덩어리로 나누어 여러 노드에 저장할 수 있다. 이는
시스템의 총체적인 저장 용량을 늘릴 수 있는 것 외에도 여러 시스템에서 데이터를
분산 처리하여 단일 시스템보다 많은 요청을 처리할 수 있다. 지속적으로 성장하는
애플리케이션의 경우 샤딩은 거의 무한한 확장성을 제공하여 큰 데이터와 높은 작업
부하를 처리할 수 있다.

하지만 데이터베이스 샤딩은 구조의 복잡성과 오버헤드를 유발하므로, 샤딩 구현 전
다음과 같은 대안들을 고려해야 한다: 수직 확장(기기 업그레이드), 특수 서비스 또는
데이터베이스의 사용(데이터 부하를 다른 곳으로 이동), 복제(주로 읽기 작업이 많을
경우 적합).

샤딩의 장점은 읽기/쓰기 처리량의 증가, 저장 용량 증가, 신뢰성 확보이다. 그러나
쿼리 오버헤드 발생, 관리의 복잡성, 인프라 비용 증가와 같은 단점도 있다. 샤딩
구현을 결정하기 전에 이러한 장단점을 고려하고 적절한 샤딩 아키텍처를 선택해야
한다. 주요 네 가지 샤딩 방법은 범위/동적 샤딩, 알고리즘/해시 샤딩, 개체/관계
기반 샤딩, 지리 기반 샤딩이다.

샤딩은 대규모 데이터 요구와 높은 읽기/쓰기 작업 부하를 처리해야 하는
애플리케이션에 좋은 해결책이지만, 추가 복잡성 또한 동반한다. 구현 전 이점과
비용을 고려하고 더 간단한 해결책이 있는지 확인해야 한다.

## Application layer

<p align="center">
  <img src="http://i.imgur.com/yB5SYwm.png"/>
  <br/>
  <i><a href=http://lethain.com/introduction-to-architecting-systems-for-scale/#platform_layer>Source: Intro to architecting systems for scale</a></i>
</p>

서비스의 성격에 따라 layer 를 두면 SPOF (Single Point of Failure) 를 해결할 수 있다.

## Service Mesh

* [서비스 메쉬란 무엇일까요?](https://www.redhat.com/ko/topics/microservices/what-is-a-service-mesh)
* [Service Mesh Comparison](https://servicemesh.es/)
* [[아코디언v2 시리즈 웨비나] Series 5. MSA와 Service mesh @ youtube](https://www.youtube.com/watch?v=mvEvPXaey50)
  
----

A service mesh is a dedicated infrastructure layer used in **microservices** or
**distributed systems** architectures to facilitate **communication**,
**management**, and **monitoring** of **inter-service connections**. It provides
a **flexible**, **scalable**, and **resilient** way to handle service-to-service
communication, abstracting complexities such as load balancing, fault tolerance,
routing, authentication, and monitoring from individual services, and
centralizing these concerns into the mesh.

The service mesh is often implemented using a combination of **sidecar proxies**
(also known as **data plane** components) and **control plane** components.

* **Sidecar proxies**: A sidecar proxy is a lightweight network proxy that is
  deployed alongside each service instance, acting as an intermediary for
  service-to-service communication. Sidecar proxies capture and route network
  traffic to and from the service, forming the data plane in the service mesh.
  They handle tasks like load balancing, circuit breaking, routing, and TLS
  termination transparently from the services' perspective.

* **Control plane**: The control plane is a set of components responsible for
  managing, configuring, and monitoring the overall service mesh. It
  communicates with the sidecar proxies to enforce global policies, handle
  service discovery, collect metrics, and provide insights into the network
  traffic and health of the mesh.

Some popular service mesh implementations include:

* **Istio**: An open-source service mesh introduced by Google, IBM, and Lyft,
  designed to work with Kubernetes but also supports other platforms. Istio uses
  the Envoy proxy as the sidecar proxy.
* **Linkerd**: Developed by Buoyant, Linkerd is an open-source service mesh that
  is lightweight and easy to deploy. It is designed primarily for Kubernetes
  environments but can be used in other scenarios too.
* **Consul Connect**: Created by HashiCorp, Consul Connect is a service mesh
  built into the Consul service discovery and configuration platform, providing
  a unified solution for service registration, discovery,

## Service Discovery

Service discovery is a crucial concept in distributed systems and microservices
architectures that enables applications, services, and clients to dynamically
locate and communicate with each other within the system. In these environments,
instances of services are often spread across multiple hosts, containers, or
data centers, and can be frequently added, removed, or modified due to scaling,
failures, or deployments. Service discovery simplifies the process of finding
these services and their addresses, abstracting the underlying complexities of
the system's topology and network configurations.

There are two major approaches to service discovery:

* **Client-side service discovery**: In this approach, clients or applications
  are responsible for locating and communicating with available service
  instances. They query a central registry or service discovery system to obtain
  the location (IP address, hostname, and port) of the desired service instance
  and then make requests to that instance directly. Examples of client-side
  service discovery systems include Netflix Eureka, [Apache
  ZooKeeper](/zookeeper/README.md), and HashiCorp Consul.

* **Server-side service discovery**: In this approach, a server-side component
  (like a load balancer or API gateway) handles service discovery on behalf of
  clients or applications. Clients do not need to be aware of the service
  instances' locations; instead, they send requests to the intermediary
  component, which locates available instances and forwards the requests.
  Examples of server-side service discovery implementations include AWS Elastic
  Load Balancing, Google Cloud Load Balancing, and Kubernetes Ingress.

Service discovery solutions often provide additional features such as:

* **Health checks**: Monitoring the availability and health of service instances
  so that only healthy instances are used for request handling.
* **Load balancing**: Distributing client requests across multiple instances of
  a service based on various algorithms and strategies, ensuring optimal
  resource utilization and fault tolerance.
* **Dynamic registration and deregistration**: Automated registration and
  removal of service instances as they are added or removed from the system.

In summary, service discovery is a key component in modern distributed systems
and microservices architectures, providing dynamic, automated, and scalable
mechanisms to locate and interact with services, improving performance,
reliability, and maintainability.

## Event Driven Architecture

Event-driven architecture (EDA) is a software architecture paradigm that focuses
on the production, detection, and reaction to events or messages, which are
generated by various components and systems within or outside the application.
In this architecture, components communicate asynchronously by emitting events
and listening for events they're interested in, without having direct
dependencies on one another. When an event is triggered, it is processed by the
corresponding event handlers or services that have subscribed to it.

Event-driven architecture offers several benefits:

* **Loose coupling**: Components in an event-driven architecture don't need to
  be aware of or depend on other components' implementation details. They only
  need to know the structure and format of events they're interested in, which
  enables easier system evolution and maintenance.

* **Scalability**: Asynchronous communication and the separation of concerns
  allow systems to scale more effectively, especially in distributed
  environments. Components can be scaled individually and operate independently,
  improving both horizontal and vertical scalability.

* **Resilience**: The decoupling of components in an EDA allows for better fault
  isolation and system resilience. If one component fails, it doesn't
  necessarily mean the entire system will fail.

* **Real-time processing**: Because components communicate asynchronously,
  event-driven architectures can efficiently handle real-time, streaming data
  and support fast or even real-time decision-making processes.

* **Adaptability**: EDAs are highly adaptable to changes in business rules or
  logic since they are based on events instead of rigid workflows. New
  components or event subscribers can be easily added without affecting existing
  components.

* **Improved traceability**: Events provide a natural means of tracking and
  tracing system activities, making auditing and monitoring easier.

Event-driven architectures are common in modern systems, particularly in
microservices, serverless computing, and IoT applications, where components need
to react to events generated by other components, devices, or external services.
However, EDA can also introduce complexity, and developers must carefully manage
and monitor event flows, ensure consistency across distributed systems, and
handle issues like **at-least-once** or **exactly-once** processing.

## Event Sourcing Architecture

Event Sourcing is an architectural pattern that revolves around the concept of
storing and managing the state of a system using a sequence of events rather
than persisting the current state. In this approach, any changes to a system's
state are recorded as immutable, append-only events in an event log or event
store. Instead of updating records, new events that represent the changes are
added to the log. This enables a system to recreate its state at any point in
time by reprocessing the sequence of events, essentially "replaying" the history
of the system.

Event sourcing architecture has several benefits and unique characteristics:

* **Auditability and traceability**: Since each event represents a change in the
  system's state, event sourcing naturally builds a complete audit trail. It
  allows for easy traceability of actions and better understanding of the state
  of the system.

* **Temporal querying**: Event sourcing facilitates the ability to query the
  system's state at any point in time, which is useful for historical analysis
  and debugging.

* **Event-driven**: This architecture style is inherently event-driven, which
  works well with modern, reactive, and distributed systems that rely on
  message-based communication.

* **Scalability**: Event sourcing enables systems to scale horizontally by
  separating concerns, such as reads, writes, and processing of events across
  multiple instances or services.

* **Resilience**: Since events are immutable and each event captures a change
  that occurred to the system, event sourcing can help improve resilience by
  allowing the restoration of a system's state from the event log in case of
  failures or corruption.

Simplifies complex business logic: By decomposing complex business logic into a
series of events, event sourcing can make these processes more understandable
and maintainable.

Event sourcing is often used in conjunction with Command Query Responsibility
Segregation (CQRS) and event-driven architectures. Though it offers many
benefits, event sourcing is not suitable for every scenario and can introduce
complexity in terms of event storage, duplication, and ensuring event-driven
consistency. Designers should carefully evaluate its appropriateness for a given
system or problem domain.

## Command and Query Responsibility Segregation (CQRS)

Command and Query Responsibility Segregation (CQRS) is an architectural pattern
that separates the read operations (queries) from the write operations
(commands) in a system. In this pattern, different models, services, or
components handle the processing of commands and queries, allowing them to be
optimized independently for their respective use cases.

CQRS has several benefits:

* **Scalability**: With separate read and write models, each part of the system
  can be scaled independently, resulting in a more balanced and scalable
  solution.
* **Optimization**: By separating the aspects of a system that deal with writes
  and reads, each side can be optimized for its specific purpose. For instance,
  read models can be denormalized to improve query performance, while write
  models can be focused on protecting data consistency and integrity.
* **Read/write contention reduction**: By handling reads and writes separately,
  CQRS helps reduce contention between read and write operations that can occur
  in traditional CRUD (Create, Read, Update, Delete) systems, where reads and
  writes compete for the same resources.

* **Flexibility**: CQRS allows for different models, data storage solutions, and
  technologies to be used for implementing the **command** and **query** sides,
  providing greater flexibility for designing and evolving the system.

* **Parallel development**: Since the query and command models are separate,
  development teams can work on different parts of the system simultaneously,
  leading to faster development cycles.

**CQRS** is often used in conjunction with **[Domain-Driven Design
(DDD)](/ddd/README.md)** and **Event Sourcing (ES)**, although it can be applied
independently of these patterns. It's important to note that **CQRS** is not
suitable for all systems and use cases. The additional complexity that comes
with separating the read and write models should be carefully evaluated against
the benefits it provides for the specific problem domain.

When implementing **CQRS**, developers need to address challenges such as
managing **eventual consistency** between the command and query models, handling
versioning and schema changes, and ensuring appropriate security measures for
both read and write operations.

## API Gateway

- [API Gateway | nginx](https://www.nginx.com/learn/api-gateway/)

An API Gateway is a server or software component that acts as an entry point and
mediator for API (Application Programming Interface) requests in a microservices
architecture or distributed systems. It serves as a reverse proxy that routes
incoming client requests to the appropriate backend services and handles
functions such as request/response transformation, authentication,
authorization, load balancing, caching, and monitoring.

The API Gateway fulfills several roles and benefits in a system:

* **Abstraction**: By providing a single entry point for client requests, the
  API Gateway hides the underlying microservices' complexity and allows clients
  to access various services through a unified and consistent interface.

* **Routing**: The gateway is responsible for routing requests to the
  appropriate backend services, which can be based on factors like API
  versioning, request paths, and request content.

* **Load balancing**: It distributes incoming requests across multiple instances
  of services, which improves the system's overall performance and resiliency.

* **Authentication and authorization**: The API Gateway can centralize
  authentication and authorization responsibilities, verifying client
  credentials and ensuring appropriate access control before allowing requests
  to reach the backend services.

* **Request and response transformation**: The gateway can transform request
  payloads and headers to comply with backend services' requirements or modify
  response payloads to suit the client's specifications.

* **Caching**: By caching responses from backend services, the API Gateway can
  reduce the load on those services and improve response times for specific
  requests.

* **Monitoring and logging**: The API Gateway serves as a centralized point for
  monitoring and logging, allowing easier management of API usage, performance,
  and error tracking.

* **Security**: The gateway can implement security features such as rate
  limiting, IP whitelisting, and API key management to protect the system from
  malicious clients or abuse.

In summary, the API Gateway is an essential component in microservices
architectures and distributed systems, helping manage, control, and secure the
access to backend services while providing a simplified, consistent experience
for clients. Developers can implement custom API Gateways or leverage existing
solutions like Amazon API Gateway, Kong, and Apigee.

## gRPC

> * [gRPC | TIL](/grpc/README.md)

----

gRPC (gRPC Remote Procedure Calls) is an open-source, high-performance Remote
Procedure Call (RPC) framework initially developed by Google. It is designed to
enable efficient communication between microservices or distributed systems by
providing a robust, reliable, and language-agnostic way for services to call
each other across networks. gRPC is built on top of the HTTP/2 protocol, which
enables bi-directional streaming and reduced latency when compared to its
predecessor, HTTP/1.1.

gRPC has several key features and benefits:

* **Protocol Buffers**: gRPC uses Protocol Buffers (also known as protobuf) as
  the **Interface Definition Language (IDL)** and default serialization format
  for messages exchanged between services. Protocol Buffers are more efficient,
  faster, and smaller in size compared to formats like JSON or XML.

* **Language-agnostic**: gRPC is supported by many programming languages,
  including Java, Go, C++, Python, and Node.js, which allows developers to build
  and interact with services across different languages and platforms.

* **Bi-directional streaming**: gRPC leverages HTTP/2 features to support
  bi-directional streaming, enabling efficient, real-time communication between
  client and server by sending and receiving messages concurrently.

* **Strongly-typed**: By using Protocol Buffers, gRPC enforces strongly-typed
  contracts for service interfaces and messages, improving development
  experience, validation, and error handling.

* **Compression**: gRPC supports compression, reducing network bandwidth usage and
  improving performance, especially in cases where payloads are extensive, and
  network conditions are not optimal.

* **Multiplexing**: With HTTP/2 multiplexing, gRPC can send multiple requests
  and responses concurrently over a single TCP connection, reducing overhead and
  latency.

* **Security**: gRPC provides built-in support for Transport Layer Security
  (TLS), ensuring secure communication between client and server, and allowing
  for authentication with client certificates.

Due to its performance and feature-set, gRPC is especially suited for
low-latency services, high-throughput communication, or when using a wide range
of programming languages across a system. It is commonly used in microservices,
IoT platforms, and real-time applications. However, gRPC may not be suitable for
all use cases, such as browser-facing APIs or scenarios where networks only
support HTTP/1.1.

## GraphQL

> * [Fullstack React GraphQL TypeScript Tutorial](https://www.youtube.com/watch?v=I6ypD7qv3Z8)
>   * [src](https://github.com/benawad/lireddit)
> * [graphql](https://graphql.org/)

----

GraphQL is a query language and runtime for APIs developed by Facebook in 2012,
and released as an open-source project in 2015. It is designed to provide a more
efficient, flexible, and powerful alternative to the traditional RESTful API
approach. GraphQL enables clients to request the specific data they need instead
of over-fetching or under-fetching data with predefined endpoints like in REST
APIs.

GraphQL has three major components:

* **Query language**: GraphQL provides a human-readable and strongly-typed query
  language that allows clients to request specific data, including nested and
  related data from multiple resources, in a single request.
* **Schema**: GraphQL uses a schema to define the types, relationships, and
  operations that are available for clients to query. The schema is based on
  GraphQL's type system, which includes Object types, Scalar types, Enum types,
  Interface types, and Union types. This strongly-typed schema enables improved
  validation, performance, and autocompletion features in client applications.
* **Runtime**: The GraphQL runtime processes queries by executing resolver
  functions that fetch data from various data sources (such as databases, APIs,
  or other services) and return the data in the shape specified by the client's
  query.

Key features of GraphQL include:

* **Flexible data retrieval**: Clients can request the exact data they need,
  reducing over-fetching and under-fetching issues associated with fixed REST
  endpoints.
* **Hierarchical structure**: GraphQL queries naturally follow the hierarchical
  structure of the data, which simplifies requesting data from multiple
  resources and related entities.
* **Strongly-typed schema**: The type system in GraphQL enables better
  validation, performance optimization, and developer tooling.
* **Introspection**: GraphQL supports introspection, allowing clients to
  discover the schema, types, and fields available at runtime, which facilitates
  building dynamic clients and API explorers.
* **Real-time updates**: GraphQL supports real-time updates with subscriptions,
  enabling clients to receive updates when data changes on the server.

Although GraphQL has many advantages, it is essential to consider its
trade-offs, such as **increased complexity** compared to simple REST APIs, the
potential for non-optimal queries, and the need for adequate caching strategies
and performance optimizations.

## REST vs GraphQL vs gRPC

- [RPC와 REST의 차이점은 무엇인가요?](https://aws.amazon.com/ko/compare/the-difference-between-rpc-and-rest/)

| Feature | REST | GraphQL | gRPC |
|-|-|-|-|
| Coupling | Loosely-coupled (resource-based) | Loosely-coupled (flexible query-based) | More tightly-coupled (RPC-based, using procedure calls) |
| Chattiness | Can be chatty (multiple endpoints) | Less chatty (single concise request for data needed) | Can be chatty but supports efficient communication (Protocol Buffers) |
| Performance  | Depends on HTTP, caching, serialization| Efficient data fetching but may need optimization | High performance (HTTP/2, Protocol Buffers, bi-directional streaming) |
| | Limited streaming with HTTP/1.1 | and caching strategies | |
| Complexity | Simple (based on standard HTTP methods)| Learning curve for query language, schema, tooling    | Steeper learning curve, Protocol Buffers, RPC-based communication |
| Caching | Supports HTTP caching mechanisms | Requires custom caching strategies | No built-in caching, requires custom caching mechanisms |
| Codegen | Usually not required (library support) | Codegen tools for schemas and types available | Code generation for message types and service interfaces (Protocol Buffers)|
| Discoverability   | Discoverable through URLs and HTTP headers | Supports introspection for schema discovery | Service contracts help in discoverability (due to strong typing) |
| Versioning | Versioning using different URLs or headers | Extendable schema design, field deprecation | Require changes in service contracts, version negotiation or new services |

## Long Polling

Long polling is a technique used to simulate real-time communication between a
client and a server in web applications where server push is not supported or
feasible. It is an extension of the traditional polling method, which involves
clients repeatedly requesting data updates from the server.

In long polling, the client makes a request to the server for new data, similar
to traditional polling. However, instead of replying immediately with an empty
response if no new data is available, the server holds the client's request
open. As soon as new data becomes available or a predefined timeout is reached,
the server returns the available data or an empty response. Upon receiving the
response, the client immediately initiates a new long polling request, repeating
the process.

Long polling provides some advantages over traditional polling:

* **Reduced latency**: Since the server responds when new data is available,
  long polling minimizes the latency between data becoming available and the
  client receiving it, as compared to traditional polling, where clients wait
  between requests.
* **Lower server load**: Long polling reduces unnecessary polling requests to
  the server when no new data is available, which can help lower the server load
  and save bandwidth.

However, long polling has its limitations:

* **Not real-time**: Although long polling reduces latency, it is not true
  real-time communication and may still introduce some delays due to
  request/response cycle times.
* **Resource consumption**: Holding open multiple long polling connections can
  consume server resources, especially in high traffic applications, which could
  impact performance.
* **Timeout handling**: Properly handling request timeouts and maintaining state
  between long polling requests adds complexity to the implementation.

Modern web technologies like **WebSockets** have addressed many of the
limitations of long polling by providing a real-time, bi-directional, and more
efficient communication channel between clients and servers. However, long
polling may still be a viable solution for scenarios where **WebSockets** or
other real-time communication methods are not supported or feasible.

## Web Socket

- [What are WebSockets?](https://www.pubnub.com/guides/websockets/)

WebSocket은 클라이언트와 서버 간의 양방향 통신을 가능하게 하는 프로토콜입니다.
실시간 채팅, 메시징 및 멀티플레이어 게임과 같이 실시간 업데이트를 필요로 하는
애플리케이션 개발에 사용되며, 기존의 HTTP 요청-응답 모델과 달리 지속적인 연결을
통해 효율성과 성능을 개선할 수 있습니다.

그러나 브라우저 지원, 프록시 및 방화벽 제한, 확장성, 상태 정보 유지, 보안 고려
사항 등의 단점이 있습니다. 이에 따라 WebSocket과 함께 HTTP 스트리밍 또는 롱
폴링과 같은 대체 솔루션도 사용해야 할 수 있습니다.

WebSocket, HTTP, 웹 서버, 폴링 사이의 차이를 이해하는 것은 중요하며, 여러
WebSocket 라이브러리를 사용하여 애플리케이션을 개발할 수 있습니다. 예를 들어
Socket.IO, SignalR, SockJS, ws 및 Django Channels 등이 있습니다.

실시간 통신에 대한 요구가 있는 경우 WebSockets를 고려해볼 만한 이유는 성능 향상,
호환성, 확장성, 다양한 개방형 리소스 및 가이드 등이 있습니다. 그러나 PubNub은
현재 대부분의 경우에는 롱 폴링이 더 나은 선택이라고 생각하며, 신뢰성, 안정성,
확장성을 위해 롱 폴링을 사용하는 것을 권장하고 있습니다.

결론적으로, WebSocket은 실시간 기능을 구현하는 데 유용한 프로토콜이지만 모든
상황에 적합한 것은 아닙니다. WebSocket과 함께 다른 솔루션을 사용하여 더 나은,
확장성 있는 실시간 애플리케이션을 구축할 수 있습니다. PubNub을 사용하면 사용자
경험 개선, 개발 및 유지 관리 비용 절감, 출시 시간 단축 및 엔지니어링 및 웹 개발
팀이 관리해야 하는 복잡성 감소 등의 이점이 있습니다.

## Server-Sent Events (SSE)

서버 전송 이벤트(Server-Sent Events, SSE)는 단일 HTTP 연결을 통해 서버가
클라이언트에게 실시간 업데이트를 전송하는 웹 표준입니다. SSE는 서버에서
클라이언트로 업데이트를 보내는 단방향 통신을 처리하도록 설계되어 있어, 알림,
실시간 업데이트, 이벤트 스트리밍 같은 경우에 유용합니다.

SSE는 메시지에 텍스트 기반 인코딩을 사용하고 표준 HTTP 프로토콜에 의존하기
때문에 WebSocket과 비교할 때 기존 네트워크 인프라와 더 호환성이 좋습니다. SSE의
핵심 구성 요소는 현대 웹 브라우저에 내장되어 있고 SSE 연결을 설정하고 처리하는
간단한 자바스크립트 인터페이스를 제공하는 EventSource API입니다.

SSE의 주요 특징:

- 단방향 통신: SSE는 서버에서 클라이언트로 업데이트를 효율적으로 보내는 것을
  목표로 설계되어 있습니다.
- 실시간 업데이트: SSE는 단일 HTTP 연결을 통해 작동하여 서버가 연결된
  클라이언트에게 낮은 지연 시간으로 실시간 업데이트를 전송할 수 있습니다.
- 텍스트 기반 인코딩: SSE에서는 메시지 페이로드에 대해 일반적으로 UTF-8과 같은
  텍스트 기반 인코딩을 사용하여 처리, 디버깅 및 플랫폼 간 호환성을 단순화합니다.
- 재연결 기능: EventSource API는 연결 손실을 처리하고, 연결이 재개되면 서버에
  다시 연결하여 업데이트를 계속 진행합니다.
- 메시지 구조: SSE 메시지에는 이벤트 유형, 메시지 ID, 데이터 페이로드가 포함될
  수 있습니다. 이 구조는 클라이언트가 별도의 이벤트 리스너를 사용하여 다양한
  유형의 이벤트를 처리하고 연결이 끊어진 경우 이벤트 스트림을 효율적으로
  이어받을 수 있게 합니다.
- 서버 전송 이벤트는 기존의 롱 폴링(long-polling) 기술에 비해 일부 장점을
  제공하지만, 단방향 통신에만 제한됩니다. 양방향 통신이 필요한 경우, WebSocket과
  같은 프로토콜이 더 적합합니다. 또한, 모든 웹 브라우저에서 SSE를 지원하지 않아
  (예: 기본적으로 Internet Explorer에서 지원하지 않음) 이런 경우 폴리필이나 대체
  방법을 사용해야 할 수 있습니다.

## Long-Polling vs WebSockets vs Server-Sent Events

[Long-Polling vs WebSockets vs Server-Sent Events](fundamentals/Long-PollingvsWebSocketsvsServer-SentEvents.md)

## Geohashing

Geohashing is a technique for encoding geographic coordinates (latitude and
longitude) into a short string representing a defined area or grid cell on
Earth's surface. These strings or geohashes are typically composed of a
combination of alphanumeric characters and have a hierarchical structure,
meaning that the prefix of a geohash represents a larger area, and adding
characters increases precision.

Geohashing was developed by Gustavo Niemeyer in 2008 and can be used for various
applications, such as:

* **Spatial indexing**: Geohashing can be used to index spatial data efficiently
  in databases, making it easier and faster to perform location-based queries
  and search for nearby points.
* **Location-sharing**: Geohashes can be used for sharing approximate locations
  without exposing exact coordinates, which provides an added layer of privacy
  and security.
* **Clustering and visualization**: Geohashes can be used to cluster and
  aggregate geospatial data points, which is helpful for visualizing and
  analyzing large data sets with spatial components.

The main advantage of geohashing is that it simplifies the representation and
handling of geographic coordinates, making it easier to store, query, and
manipulate geospatial data. Also, geohashes can be indexed and searched
efficiently using standard string-based algorithms and data structures. However,
one limitation of geohashing is that it can produce errors when working near
boundaries of geohash cells, where nearby points might be located in different
cells and may require more complex search strategies.

Several libraries and tools are available in various programming languages to
implement geohashing, such as Geohash.org, which provides an online interface
for encoding and decoding geohash values, and programming libraries like
pygeohash (Python), Geohash-java (Java), and node-geohash (JavaScript).

## Quadtrees

A quadtree is a tree data structure commonly used in computer graphics,
computational geometry, and geographic information systems (GIS) to efficiently
organize, search, and manipulate two-dimensional spatial data, such as points,
lines, or images. In a quadtree, each node represents a specific area or region
within a two-dimensional space, and each internal node has exactly four children
corresponding to the four quadrants (Northwest, Northeast, Southwest, and
Southeast) formed by subdividing the region. The tree's leaf nodes contain the
actual spatial data points or empty regions.

Quadtrees can be used for various applications, such as:

* **Spatial indexing**: Quadtrees can be utilized to index spatial data
  efficiently in databases, enabling fast and accurate location-based queries,
  such as searching for nearby points or objects within a given area.
* **Image compression**: Quadtrees are often used in image compression
  techniques, such as lossless and lossy compression algorithms. They can
  efficiently represent the hierarchical structure and areas of uniform color or
  intensity within an image.
* **Collision detection**: In computer graphics or game development, quadtrees
  can be used for collision detection by efficiently organizing and searching
  for objects within a given spatial area, significantly improving performance
  compared to brute-force methods.
* **Terrain modeling**: In geospatial applications, quadtrees can be utilized
  for terrain modeling and efficient rendering of large datasets, such as
  digital elevation models and raster data.

The main advantage of quadtrees is their efficiency in handling and querying
spatial data, as they divide the space into finer and finer regions based on the
distribution of objects, allowing for fast and precise search operations.
However, one limitation of quadtrees is their sensitivity to the order of data
insertion, which might lead to unbalanced trees in some cases, impacting
performance. There are variants like the balanced quadtree or compressed
quadtree that address this issue to some extent.

## Circuit Breaker

- [Circuit Breaker Pattern (Design Patterns for Microservices)](https://medium.com/geekculture/design-patterns-for-microservices-circuit-breaker-pattern-276249ffab33)

Circuit breaker는 분산 시스템 및 마이크로서비스 아키텍쳐에서 실패를 감지하고
예방하며, 우아하게 처리할 수 있는 디자인 패턴이다. 이는 시스템의 회복력을
향상시킨다. 전기 회로 차단기에서 영감을 받은 회로 차단기 패턴은 소프트웨어
시스템에서 서비스나 구성요소가 실패할 수 있음을 감지하고 보호한다.

일반적으로 회로 차단기 패턴은 원격 서비스에 대한 서비스 호출이나 요청을
래핑하거나 미들웨어로 구현된다. 회로 차단기의 주요 목적은 실패, 느린 서비스에
대한 추가 요청이나 작업을 차단하여 관리할 수 있는 방법을 제공하는 것이다.

회로 차단기의 생애 주기에는 세 가지 주요 상태가 있다.

- **Closed**: closed 상태에서 회로 차단기는 요청을 서비스로 전달한다. 요청의
  성공여부를 모니터링하고, 실패 횟수나 응답 시간이 설정된 임계 값보다 큰 경우,
  회로 차단기는 "open" 상태로 전환한다.
- **Open**: open 상태에서 회로 차단기는 실패하거나 느린 서비스로의 추가 요청을
  차단하고, 즉시 fallback 상태로 응답한다. 이 상태는 서비스로의 추가 부하를
  방지하고 회복을 허용한다. 설정된 시간이 지나면, 회로 차단기는 "half-open"
  상태로 전환한다.
- **Half-Open**: half-open 상태에서 회로 차단기는 서비스의 건강 상태를 확인하기
  위해 일정 수 또는 비율의 요청을 허용한다. 요청이 성공하고 서비스가 회복되면,
  회로 차단기는 "closed" 상태로 되돌아온다. 실패가 계속되면, 회로 차단기는
  "open" 상태로 되돌아간다.

Circuit breaker 패턴을 구현함으로써, 시스템은 연쇄적인 실패를 방지하고, 느린
또는 실패하는 서비스의 영향을 줄이며, 오류 처리를 위한 fallback 메커니즘을
제공하고, 분산 시스템 및 마이크로서비스 아키텍처의 전반적인 회복력과 내결함성을
향상시킬 수 있다.

## Rate Limiting

* [Rate Limiting Fundamentals | bytebytego](https://blog.bytebytego.com/p/rate-limiting-fundamentals)
* [Rate Limiting](https://www.imperva.com/learn/application-security/rate-limiting/)

-----

컴퓨터 시스템, API, 그리고 네트워크에서의 rate limiting은 요청이나 데이터 패킷이
처리되는 속도를 제어하는 기술입니다. 특정 시간 간격 내에서 허용되는 요청, 거래,
데이터의 수에 제한을 두어 리소스의 공정한 사용을 보장하고 시스템 안정성을
유지하며 과도한 부하, 남용, 서비스 거부 공격(DoS)으로부터 서비스를 보호하는 것이
비롯된 목표입니다.

다양한 단계 및 시스템의 다른 구성 요소에서 rate limiting을 구현할 수 있습니다:

- API 및 웹 서비스: 응용 프로그램 수준에서 rate limiting을 적용하여 클라이언트가
  특정 시간 동안 API 또는 웹 서비스에 보낼 수 있는 요청 수를 제어할 수 있습니다.
  초당, 분당, 시간당 요청 수를 제한하는 것이 일반적이며, 클라이언트를 식별하고
  추적하기 위해 토큰이나 API 키를 사용합니다.
- 데이터베이스 및 백엔드 서비스: rate limiting을 적용하여 데이터베이스, 메시지
  큐 또는 캐싱 시스템과 같은 백엔드 서비스에 의해 소비되는 리소스를 관리하여
  가용 용량을 과다하게 로드하거나 소진하는 것을 방지할 수 있습니다.
- 네트워크: 네트워크 수준에서 rate limiting을 구현하여 대역폭 사용률을 제어하고,
  네트워크 혼잡을 방지하며, 클라이언트나 장치 간에 네트워크 리소스의 공정한
  분배를 보장할 수 있습니다.

주요한 rate-limiting 알고리즘은 다음과 같습니다:

- **Token Bucket**: 이 방법에서는 고정된 속도로 토큰을 버킷에 추가하되 최대
  용량까지만 추가합니다. 각 요청이나 패킷은 버킷의 토큰을 소비합니다. 버킷이
  비어 있으면 요청이 거부되거나 토큰을 사용할 수 있을 때까지 지연됩니다.
- **Leaky Bucket**: 이 알고리즘은 고정 크기의 버퍼(버킷)을 사용하며, 상수 속도로
  버킷의 아이템이 제거됩니다. 공간이 있다면 들어오는 요청이나 패킷이 버퍼에
  추가되고, 아니면 거부되거나 지연됩니다.
- **Fixed Window**: 이 알고리즘은 시간을 고정된 크기의 창이나 간격으로 나누어 각
  창에서의 요청이나 패킷 수를 추적합니다. 창이 최대 허용된 건수에 도달하면 추가
  요청이나 패킷이 거부되거나 다음 창이 시작될 때까지 지연됩니다.
- **Sliding Window**: 이 접근법은 고정 작업 창 알고리즘을 개선하여 요청
  타임스탬프에 기 바른 서적식 시간 창을 사용함으로써 더 나은 공정성과 부드러운
  제한률을 보장합니다.

Rate limiting을 효과적으로 구현함으로써 컴퓨터 시스템, API, 네트워크의 신뢰성,
성능, 보안을 유지할 수 있고 클라이언트, 사용자, 장치 간에 공정한 사용 정책을
적용할 수 있습니다.

## Asynchronism

<p align="center">
  <img src="http://i.imgur.com/54GYsSx.png"/>
  <br/>
  <i><a href=http://lethain.com/introduction-to-architecting-systems-for-scale/#platform_layer>Source: Intro to architecting systems for scale</a></i>
</p>

* Message Queues
  * Redis, RabbitMQ, Amazon SQS
* Task Queues
  * Celery
* Back pressure
  * MQ 가 바쁘면 client 에게 503 Service Unavailable 을 줘서 시스템의 성능저하를 예방한다. 일종의 circuit breaker 같다.

## Message Queue

- [System Design — Message Queues](https://medium.com/must-know-computer-science/system-design-message-queues-245612428a22)

Message Queue(메시지 큐)를 통해 응용 프로그램은 비동기적으로 통신할 수 있습니다.
큐를 사용하여 서로에게 메시지를 보내는 것으로, 보내는 프로그램과 받는 프로그램
사이의 일시적 저장소를 제공해서 연결되지 않거나 바쁠 때 중단 없이 동작할 수
있도록 합니다. 큐의 기본 구조는 프로듀서(메시지 생성자)가 메시지 큐에 전달할
메시지를 생성하는 몇 가지 클라이언트 애플리케이션과 메시지를 처리하는 소비자
애플리케이션이 있습니다. 큐에 배치된 메시지는 소비자가 검색할 때까지 저장됩니다.

메시지 큐는 마이크로서비스 아키텍처에서도 중요한 역할을 담당합니다. 서로 다른
서비스에서 기능이 분산되며, 전체 소프트웨어 어플리케이션을 구성하기 위해
병합됩니다. 이때 상호 종속성이 발생하며 시스템에 서비스 간 비블록 응답 없이 서로
연결되는 메커니즘이 필요합니다. 메시지 큐는 서비스가 비동기적으로 큐에 메시지를
푸시하고 올바른 목적지에 전달되도록 하는 수단을 제공하여 이 목적을 달성합니다.
서비스 간 메시지 큐를 구현하려면 메시지 브로커(예: RabbitMQ, Kafka)가
필요합니다.

## Message Queue VS Event Streaming Platform

Event streaming platform 은 다음과 같은 특징을 갖는다.

* Long data retention
* Repeated consumption of messages

Event streaming platform 의 종류는 다음과 같다.

* [Kafka](/kafka/README.md)
* Pulsa 

Message queue 는 다음과 같은 특징을 갖는다.

* Short data retention (Just on memory)
* Onetime consumption of messages

Message queue 의 종류는 다음과 같다.

* [nats](/nats/README.md)
* RocketMQ
* ActiveMQ
* RabbitMQ
* ZeroMQ

그러나 Message queue, Event streaming platform 경계는 흐릿해지고 있다. Rabbit MQ
역시 Long data retention, repeated consumption 이 가능하다.

## Security

### WAF (Web Application Fairewall)

* [AWS WAF – 웹 애플리케이션 방화벽](https://aws.amazon.com/ko/waf/)
* [웹방화벽이란?](https://www.pentasecurity.co.kr/resource/%EC%9B%B9%EB%B3%B4%EC%95%88/%EC%9B%B9%EB%B0%A9%ED%99%94%EB%B2%BD%EC%9D%B4%EB%9E%80/)

----
  
* 일반적인 방화벽과 달리 웹 애플리케이션의 보안에 특화된 솔루션이다. 
* 애플리케이션의 가용성에 영향을 주거나, SQL Injection, XSS (Cross Site
  Scripting) 과 같이 보안을 위협하거나, 리소스를 과도하게 사용하는 웹
  공격으로부터 웹 애플리케이션을 보호하는 데 도움이 된다.

### XSS (Cross Site Scripting)

* [웹 해킹 강좌 ⑦ - XSS(Cross Site Scripting) 공격의 개요와 실습 (Web Hacking Tutorial #07) @ youtube](https://www.youtube.com/watch?v=DoN7bkdQBXU)

----

* 웹 게시판에 javascript 를 내용으로 삽입해 놓으면 그 게시물을 사용자가 읽을 때 삽입된 스크립트가 실행되는 공격방법

### CSRF (Cross Site Request Forgery)

* [웹 해킹 강좌 ⑩ - CSRF(Cross Site Request Forgery) 공격 기법 (Web Hacking Tutorial #10) @ youtube](https://www.youtube.com/watch?v=nzoUgKPwn_A)

----

* 특정 사용자의 세션을 탈취하는 데에는 실패하였지만 스크립팅 공격이 통할 때
  사용할 수 있는 해킹 기법. 피해자가 스크립트를 보는 것과 동시에 자기도 모르게
  특정한 사이트에 어떠한 요청(Request) 데이터를 보낸다.

### XSS vs CSRF

* XSS 는 공격대상이 Client 이고 CSRF 는 공격대상이 Server 이다.
* XSS 는 사이트변조나 백도어를 통해 Client 를 공격한다.
* CSRF 는 요청을 위조하여 사용자의 권한을 이용해 Server 를 공격한다.

### CORS (Cross Origin Resource Sharing)

[cors | TIL](/cors/README.md)

### PKI (Public Key Infrastructure)

[PKI | TIL](/pki/README.md)

## SSL/TLS

> [ssl/tls | TIL](/ssltls/README.md)

**SSL (Secure Sockets Layer)** and **TLS (Transport Layer Security)** are
cryptographic protocols that provide secure communication over a computer
network, mainly the internet. They ensure that the data transmitted between a
client (such as a web browser) and a server (such as a website) remains
confidential, unaltered, and protected from unauthorized access.

SSL is the older protocol, originally developed by Netscape in the 1990s, while
TLS is the more modern and secure successor to SSL, developed by the Internet
Engineering Task Force (IETF). TLS is an evolution of SSL, currently in version
1.3 as of August 2018. In general conversation, people might still use the term
SSL, but in practice, TLS has mostly replaced SSL.

Both SSL and TLS work by performing a series of steps during a "handshake"
process:

* The client initiates the connection and sends its supported cryptographic
  algorithms and parameters to the server.
* The server responds with its chosen cryptographic algorithm, digital
  certificate, and public key.
* The client verifies the server's certificate, typically with the help of a
  Certificate Authority (CA), to ensure the server's identity. This helps
  prevent **man-in-the-middle attacks**.
* The client generates a shared secret (also known as the session key) using the
  server's public key. It sends the encrypted session key back to the server.
* The server decrypts the session key using its private key. Both the client and
  the server now have the shared secret, which can be used to encrypt and
  decrypt information exchanged during the session.

The use of SSL or TLS is crucial for ensuring secure communication, particularly
for sensitive transactions or when transmitting personal information, such as in
e-commerce, online banking, and email services. To establish an SSL/TLS
connection, websites typically use an HTTPS (Hypertext Transfer Protocol Secure)
connection, denoted by the padlock icon in most web browsers.

## mTLS

mTLS, or mutual TLS (Transport Layer Security), is an enhanced security protocol
that requires both the client and the server to establish and verify each
other's identities during the TLS handshake process. In contrast, traditional
TLS only requires the server to authenticate itself to the client, leaving the
client's identity unverified.

Mutual TLS adds an extra layer of security by authenticating both parties, which
is particularly useful in sensitive or high-security environments where client
authentication is critical.

The mTLS process involves the following steps:

* The client initiates a connection and sends its supported cryptographic
  algorithms and parameters to the server, just like in a standard TLS
  handshake.
* The server responds with its chosen cryptographic algorithm, digital
  certificate, and public key.
* The client verifies the server's certificate with the help of a Certificate
  Authority (CA) to ensure the server's identity.
* In addition to the previous steps in a standard TLS handshake, in mTLS, the
  client also sends its own digital certificate and public key to the server.
* The server verifies the client's certificate, typically with the help of a CA,
  ensuring the client's authenticity.
* The rest of the mutual TLS handshake process follows the standard TLS
  handshake, with the client and the server generating and exchanging the shared
  secret (session key) to enable encrypted communication.

mTLS is widely used in scenarios where secure and verified client-to-server
communication is essential, such as in API security, secure remote access, and
communication between microservices within an organization.

## Distributed Primary Key

* [강대명 <대용량 서버 구축을 위한 Memcached와 Redis>](https://americanopeople.tistory.com/177)

----

Sharding 을 고려하여 Primary Key 를 효율적으로 설계해 보자. 예를 들어 이메일
시스템을 디자인한다고 해보자. User 와 Email 테이블의 스키마는 다음과 같다. 

* `User`

  | field           | type      | description                                |
  | --------------- | --------- | ------------------------------------------ |
  | user_id         | Long (8B) | unique id (각 DB 별)                       |
  | email           | String    | 이메일 주소                                |
  | shard           | Long      | 자신의 메일 리스트를 저장한 DB server 번호 |
  | type            | int       | 활성화 유저인가??                          |
  | created_at      | timestamp | 계정 생성시간                              |
  | last_login_time | timestamp | 마지막 로그인 시간                         |

* `Email`  

  | field       | type           | description              |
  | ----------- | -------------- | ------------------------ |
  | mail_id     | Long (8B)      | unique id (각 DB 별)     |
  | receiver    | String or Long | 수신자                   |
  | sender      | String or Long | 송신자                   |
  | subject     | String         | 메일제목                 |
  | received_at | timestamp      | 수신시간                 |
  | eml_id      | String or Long | 메일 본문 저장 id or url |
  | is_read     | boolean        | 읽었는가??               |
  | contents    | String         | 미리보기 (내용의 일부)   |

email file 은 AWS S3 에 저장하자. email file 의 key 를 마련해야 한다. 

* `{receiver_id}_{mail_id}` 
  * `mail_id` 는 이미 shard 마다 중복해서 존재한다. 따라서 `receiver_id` 와 결합하여 사용하자.
  * 그렇다면 `eml_id` 는 필요할까? `{receiver_id}_{mail_id}` 만으로도 eml file 의 key 로 사용할 수 있기 때문이다. 조금 더 key 를 잘 설계할 수는 없을까???
* key 에 시간을 포함하면 시간에 따라 data 가 적절히 shard 로 분산된다.
* UUID (Universally Unique Identifier)
  * id 에 시간 정보가 반영되어 있다. id 를 오름차순으로 정렬하면 시간순 으로 데이터를 정렬할 수 있다.
  * 16 bytes (128 bit), 36 characters 이다. 너무 크다.
  * 적은 바이트로 시간 정보를 저장할 수 있었으면 좋겠다.
* `{timestamp: 52 bits}_{sequence: 12 bits}` 8 bytes
  * 샤드 아이디도 저장되었으면 좋겠다.
  * timestamp 는 4 bytes 를 모두 사용하면 `1970/01/01` 부터 `2106/02/07 06:28` 까지만 표현 가능하다.  
* `{timestamp: 52 bits}_{shard_id: 12 bits}_{sequence: 12 bits}` 8 bytes 
  * IDC 정보도 반영되었으면 좋겠다.
* `{timestamp: 42 bits}_{datacenter_id: 5 bits}_{worker_id: 5 bits}_{sequence: 12bits}` 8 bytes
  * 이것은 twitter 의 id 이다.
* `{timestamp: 41 bits}_{Logical Shard ID: 13 its}_{Auto Increment/1024: 10 bits}` 8 bytes
  * 이것은 Instagram 의 id 이다.
* `{timetamp: 4 bytes}_{machine_id:3 bytes}_{process_id:2 bytes}_{counter:3 bytes}` 12 bytes
  * 이것은 mongoDB 의 ID 이다. 
* `{timestamp}_{shard_id}_{type}_{sequence}` 8 bytes
* 만약 select 의 형태가 특정 user 의 최근 10 분간 수신된 email data 만 얻어오는
  형태라면 Primary Key 에 timebound 를 도입해 보는 것도 좋은 방법이다. 
  * timebound 가 없다면 email data 는 모든 shard 로 골고루 분산될 것이다.
    Primary Key 를 `{timebound}_{shard_id}_{type}_{sequence}` 를 설정해보자.
    그렇다면 특정 유저의 최근 1 시간동안 수신된 email 은 하나의 shard 에
    저장된다. 따라서 특정유저의 email data 를 얻어올 때 모든 shard 에 query 할
    필요가 없다.

## Idempotency

- [RESTful API](https://lifeisgift.tistory.com/entry/Restful-API-%EA%B0%9C%EC%9A%94)
- [What is Idempotency?](https://blog.dreamfactory.com/what-is-idempotency/)

----

한글로 멱등성이라고 한다. RESTful API 에서 같은 호출을 여러번 해도 동일한 결과를
리턴하는 것을 말한다.

멱등성에 대해 알아야 하는 주요 사실은 다음과 같습니다:

- 멱등성은 연산이나 API 요청의 속성으로, 연산을 여러 번 반복해도 한 번 실행하는
  것과 같은 결과를 생성합니다.
- 안전한(safe) 방법은 멱등성이 있지만, 모든 멱등한 방법이 안전한 것은 아닙니다.
- HTTP 메소드 중 **GET, HEAD, PUT, DELETE, OPTIONS, TRACE**는 멱등성이 있고, **POST**와
  **PATCH**는 일반적으로 멱등성이 없습니다.
- HTTP 메소드의 멱등성을 이해하고 활용하면 더욱 일관성 있고, 신뢰성 있고, 예측
  가능한 웹 애플리케이션과 API를 만들 수 있습니다.
- REST API에서 사용되는 대부분의 HTTP 메소드는 POST를 제외하고 모두 멱등하며,
  REST 원칙에 따라서 멱등한 메소드를 적절하게 사용하게 됩니다.

## 80/20 rule

어떠한 데이터의 20% 만 자주사용한다는 규칙이다. 주로 Cache data size 를 estimate
할 때 사용한다. 예를 들어 total data size 가 100GB 이면 cache data size 는 20GB
로 예측한다. 

## 70% Capacity model

estimated data size 는 total data size 의 70% 라는 규칙이다. 예를 들어 estimated
data size 가 70GB 이면 total data size 는 100GB 이면 충분하다고 예측한다.

```
total data size : estimated data size = 100 : 70
```

## SLA, SLO, SLI

> * [The Difference between SLI, SLO, and SLA](https://enqueuezero.com/the-difference-between-sli-slo-and-sla.html)
> * [Uptime Calculation Web](https://uptime.is/)

-----

* **SLA (Service Level Agreement)** is a contract that the service provider
  promises customers on service availability, performance, etc.
* **SLO (Service Level Objective)** is a goal that service provider wants to
  reach.
* **SLI (Service Level Indicator)** is a measurement the service provider uses
  for the goal.

## Optimistic Lock vs Pessimistic Lock

* [비관적 Lock, 낙관적 Lock 이해하기](https://medium.com/@jinhanchoi1/%EB%B9%84%EA%B4%80%EC%A0%81-lock-%EB%82%99%EA%B4%80%EC%A0%81-lock-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0-1986a399a54)
* [Optimistic Locking in JPA](https://www.baeldung.com/jpa-optimistic-locking)
  * [src](https://github.com/eugenp/tutorials/tree/master/persistence-modules/hibernate-jpa)
* [Pessimistic Locking in JPA](https://www.baeldung.com/jpa-pessimistic-locking)
  * [src](https://github.com/eugenp/tutorials/tree/master/persistence-modules/hibernate-jpa)

-----

Database 의 isolation level 보다 융통성있는 locking 방법

원리는 다음과 같다.

* Table Schema 에 `version` field 가 있어야 한다.
* A Client 가 원하는 record 를 `version` 을 포함하여 읽어온다. `version` 은 `v0` 이라고 하자. 
* 그 record 의 내용을 변경하여 `version` 은 `v0` 으로 다시 저장하자. 실패할 수
  있다. 만약 B client 가 그 record 의 내용을 변경했다면 `version` 은 `v1`
  으로 바뀌어 있을 것이기 때문이다.  

## Disaster Recovery (DR)

Disaster Recovery (DR) is a well-planned and structured approach that involves
policies, procedures, and tools to enable organizations to quickly resume their
critical operations and services after experiencing a disruptive event, such as
a natural disaster, cyberattack, power outage, or hardware failure. The primary
goal of Disaster Recovery is to minimize downtime, data loss, and financial
impact while ensuring business continuity and maintaining customer trust. This
is achieved through a combination of data backups, redundancy, failover
mechanisms, and comprehensive recovery plans that are regularly tested and
updated.

## OAuth 2.0

OAuth 2.0 (Open Authorization) is an open standard authorization framework that
allows users to grant third-party applications or services limited access to
their resources or data on other platforms without sharing their credentials
(username and password). It is widely used for secure communication between web
applications, mobile apps, and APIs, providing a more secure and streamlined
experience for end-users.

In OAuth 2.0, the user provides consent to grant specific access rights or
permissions (called "scopes") to a third-party application. The application then
obtains an access token from the authorization server which it uses to request
resources or data on behalf of the user from the resource server. This process
ensures that the user's confidential information remains protected and that
third-party applications can only access the authorized data or perform
restricted actions as permitted by the user.

OAuth 2.0 simplifies the authentication process, improves security, and
standardizes how applications request access across various systems or
platforms, making it a popular choice for many web services and API providers
such as Google, Facebook, and Twitter.

## OpenID Connect (OIDC)

[OpenID Connect (OIDC)](/oidc/README.md)

OpenID Connect (OIDC) is an identity layer built on top of the OAuth 2.0
protocol. It is a simple and standardized method for securely authenticating
users and providing single sign-on (SSO) functionality for web applications,
mobile apps, and APIs. While OAuth 2.0 focuses on authorization for granting
third-party applications access to resources, OIDC extends this framework to
provide user authentication and identity management.

Key components of OpenID Connect include:

* **ID Token**: A **JSON Web Token (JWT)** containing user attributes (claims),
  such as user ID, email, and name, as well as information about the
  authentication event. It is issued by the OIDC identity provider (IdP) and is
  digitally signed to ensure its authenticity and integrity.
* **UserInfo Endpoint**: It is an API provided by the identity provider that
  returns user claims. These claims may include demographic information, email
  address, and other user attributes.
* **Discovery**: OIDC defines a standard discovery mechanism using well-known
  URLs and documents, allowing client applications to automatically discover
  configuration and endpoint information.
* **Standardized Flows**: OIDC builds upon OAuth 2.0 flows such as the
  Authorization Code Flow and Implicit Flow for obtaining access and ID tokens,
  leveraging market-proven practices for identity management.

In summary, OpenID Connect extends OAuth 2.0 to provide smooth and secure user
authentication and single sign-on functionality for web-based applications,
making it easier for developers to implement and manage trusted user
authentication across a wide range of platforms and services.

## Single Sign-On (SSO)

Single Sign-On (SSO) is an authentication process that allows users to access
multiple applications, systems, or services within an organization by logging in
only once with their credentials. This enables a seamless and convenient user
experience, as users don't have to remember and input separate usernames and
passwords for each application they need to access.

SSO provides several benefits:

* **Improved User Experience**: Users can navigate between different systems or
  applications without the need to re-enter their credentials, simplifying their
  workflow and streamlining access to resources.

* **Enhanced Security**: By reducing the number of passwords that users need to
  remember, SSO encourages the use of more secure, complex passwords, decreasing
  the likelihood of password-related security breaches.

* **Reduced Helpdesk Inquiries**: With fewer passwords to remember, there is a
  decrease in requests for password resets or forgotten credentials, reducing
  the burden on the helpdesk.

* **Simplified User Management**: Centralized management of user authentication
  makes it easier for administrators to control access, manage user accounts,
  and monitor user activity across multiple systems or applications.

SSO can be implemented using various methods, such as **Security Assertion
Markup Language (SAML)**, **OAuth 2.0 with OpenID Connect (OIDC)**, or
proprietary protocols provided by identity providers. When implementing SSO,
it's crucial to ensure that the chosen method complies with security best
practices to avoid potential vulnerabilities.

## Control Plane, Data Plane, Management Plane

* [Data plane](https://en.wikipedia.org/wiki/Data_plane)

----

* The **data plane** is the part of the software that processes the data requests.
* The **control plane** is the part of the software that configures and shuts down the data plane.
* The **management plane** is ???

## Distributed Transaction

* [distributed transaction | TIL](/distributedtransaction/README.md)
* [SAGAS](https://www.cs.cornell.edu/andru/cs711/2002fa/reading/sagas.pdf)

----

SAGAS is a long lived transaction that can be broken up into transactions.

## Observability

* [Monitoring @ TIL](essentials/Monitoring.md)
* [Observability: 로그라고해서 다 같은 로그가 아니다(1/2)](https://netmarble.engineering/observability-logging-a/)
  * [Observability: 로그라고해서 다 같은 로그가 아니다(2/2)](https://netmarble.engineering/observability-logging-b/)

## Load Test

* [Load Test | TIL](/loadtest/README.md)

-------

monitoring, logging, tracing, alerting, auditing 등을 말한다.

## Incidenct

* [Incident](/incident/README.md)

## Consistent Hashing

- [Consistent Hashing | TIL](/consistenthasing/README.md)

## Database Index

[Database Index](/index/README.md)

## SQL vs NoSQL

- [SQL vs NoSQL: 5 Critical Differences](https://www.integrate.io/blog/the-sql-vs-nosql-difference/)

SQL과 NoSQL은 데이터 저장 및 검색 요구 사항에 맞춰 다른 유형의 데이터베이스 관리
시스템입니다. 그들 사이의 주요 차이점은 데이터가 구조화되고, 조직화되며 조회되는
방식에 있습니다.

**SQL**:

- SQL 데이터베이스는 사전에 정의된 고정 스키마와 구조화된 테이블 형식에
- SQL은 구조화된 질의 언어(Structured Query Language)의 약자로, 관계형
  데이터베이스(RDBMS)를 관리하는 데 사용되는 표준화된 프로그래밍 언어입니다. SQL
  데이터베이스의 예로는 SQL Server, Oracle, PostgreSQL이 있습니다. 의존하며,
  데이터는 행과 열에 저장됩니다.
- SQL은 ACID(원자성, 일관성, 고립성 및 지속성) 속성에 대한 강력한 지원을
  제공하여 높은 데이터 일관성과 무결성을 보장합니다.
- SQL 데이터베이스는 복잡한 질의, 다중 행 트랜잭션 및 조인에 적합합니다.
- 일반적으로 단일 서버에 컴퓨팅 리소스(CPU, 메모리)를 추가하여 수직으로
  확장됩니다.

**NoSQL**:

- NoSQL은 "not only SQL"의 약자이며, 표준 SQL 질의 언어를 사용하지 않는 비관계형
  데이터베이스를 가리킵니다. NoSQL 데이터베이스의 예로는 MongoDB, Cassandra,
  Couchbase가 있습니다.
- NoSQL 데이터베이스는 키-값, 문서, 컬럼-패밀리 및 그래프 모델 등 다양한 데이터
  구조를 지원하며, 데이터 표현에 유연성을 제공합니다.
- NoSQL은 확장성, 고 가용성 및 장애 허용을 지원하도록 설계되어 있으며, 일관성,
  성능 및 분산을 개선하기 위해 최종적 일관성(eventual consistency)을 허용하는
  경우가 많습니다.
- NoSQL 데이터베이스는 비구조화, 반구조화 또는 계층형 데이터, 대용량 데이터 및
  간단한 쿼리 처리에 적합합니다.
- 일반적으로 데이터와 작업 부하를 여러 서버나 클러스터로 분산하여 수평으로
  확장됩니다.

SQL과 NoSQL 데이터베이스 사이에서 선택하는 것은 주로 애플리케이션의 특정 요구
사항, 데이터 모델, 예상되는 쿼리 복잡성, 일관성 요구 사항 및 확장 요구에 따라
달라집니다. SQL 데이터베이스는 일반적으로 구조화된 데이터, 복잡한 트랜잭션 및
레코드 간 관계를 처리할 때 선호되는 반면, NoSQL 데이터베이스는 대규모-분산,
스키마를 가지지 않은 데이터 구조를 높은 동시성 환경에서 관리하는데 자주
사용됩니다.

| 항목 | SQL | NoSQL |
|:--------|:----|:------|
| 스키마 | 엄격한 스키마 | 스키마 또는 스키마리스 |
| 쿼리 | SQL | SQL의 일부 또는 UnSQL (비구조화된 SQL) |
| 확장성 | 스케일 업 | 스케일 아웃 |
| 신뢰성 또는 ACID 준수 | 엄격한 ACID | ACID의 일부 |

## Hadoop

> [hadoop | TIL](/hadoop/README.md)

Hadoop is an open-source **software framework** developed by the Apache Software
Foundation for **distributed storage** and **processing of large volumes of data**
across clusters of computers. It is designed to scale horizontally from a single
server to thousands of machines, offering high availability, fault tolerance,
and data redundancy.

Hadoop is built on the principle of splitting data into smaller chunks and
processing them in parallel on multiple nodes, allowing users to process and
analyze large datasets quickly and efficiently. It is based on the MapReduce
programming model, where data is divided into independent tasks that are mapped
to processing nodes and then reduced to generate output.

The core components of the Hadoop ecosystem include:

* **Hadoop Distributed File System (HDFS)**: A distributed and scalable file
  system that stores data across multiple nodes, splitting the data into blocks
  and replicating them to ensure fault tolerance and high availability. HDFS
  supports large-scale data storage and is optimized for large, sequential
  read/write operations.
* **MapReduce**: The processing engine of Hadoop, responsible for converting
  large-scale data processing tasks into smaller, manageable jobs, processing
  them in parallel across the cluster, and then aggregating the results. It
  consists of two main phases: the Map phase, which processes and filters the
  input data; and the Reduce phase, which performs a summary operation on the
  output from the Map phase.
* **YARN (Yet Another Resource Negotiator)**: The cluster management layer that
  is responsible for allocating resources, such as CPU and memory, to
  applications running on the Hadoop cluster. YARN acts as a bridge between HDFS
  and the various processing engines, allowing multiple engines to run
  concurrently on the same data.
* **Hadoop Common**: A set of libraries and utilities that support the other
  Hadoop modules.

Besides these core components, there is a vast ecosystem of related tools and
technologies that work with Hadoop to enhance its capabilities, including data
processing engines like **Spark**, data warehousing solutions like **Hive**, and
large-scale data storage systems like **HBase**.

Hadoop is widely used in many industries for processing and analyzing large
amounts of data, including search engines, social networks, finance, big data
analytics, scientific research, and more. Its scalability, flexibility, and
open-source nature make it an attractive choice for organizations that need to
store and process vast amounts of data efficiently.

## MapReduce

> [MapReduce](https://static.googleusercontent.com/media/research.google.com/en//archive/mapreduce-osdi04.pdf) - Simplified Data Processing on Large Clusters

MapReduce is a programming model and framework designed for processing and
analyzing large volumes of data in parallel across distributed clusters of
computers. It was introduced by Google to solve large-scale computation problems
in a scalable and fault-tolerant manner. MapReduce has gained significant
popularity in the field of big data processing and is a fundamental component of
the Hadoop ecosystem.

MapReduce divides a large data-processing task into smaller, more manageable
subtasks, and then processes these subtasks in parallel across multiple nodes in
the cluster, before aggregating the results and generating the final output.

The MapReduce model consists of two primary phases:

* **Map phase**: In this phase, the input data is divided into fixed-size
  chunks, called input splits, which are distributed across the cluster. A map
  function processes each input split, filtering or transforming it into a set
  of intermediate key-value pairs. This map function is user-defined and is
  tailored to the specific problem to be solved.
* **Reduce phase**: In this phase, the intermediate key-value pairs generated by
  the map phase are sorted and grouped by key. A reduce function is then applied
  to each group of values sharing the same key, which aggregates or combines
  these values to generate a smaller set of key-value pairs as final output. The
  reduce function is also user-defined and problem-specific.

MapReduce takes care of the underlying complexity of distributed processing,
such as data partitioning, task scheduling, resource management, error handling,
and inter-node communication. As a result, the developer can focus on writing
the map and reduce functions to solve the specific problem at hand, without
needing to manage the intricacies of parallel and distributed computation.

MapReduce has been widely adopted in big data processing and analytics due to
its scalability, fault tolerance, and ability to handle unstructured or
semi-structured data. However, it has some limitations, such as high latency and
lack of support for iterative algorithms or real-time processing, which have led
to the development of alternative data-processing frameworks like Apache Spark.

## Consensus Algorithm

[Consensus Problems](/distributedsystem/README.md#consensus-problems)

## Paxos

[Paxos](/distributedsystem/README.md#paxos)

## Gossip protocol

- [Gossip Protocol Explained | HighScalability](http://highscalability.com/blog/2023/7/16/gossip-protocol-explained.html)
 
Gossip Protocol은 대규모 분산 시스템에서 메시지를 전송하는 분산형 P2P 통신
기술로, 각 노드가 일정 시간마다 무작위로 선택된 다른 노드들에게 메시지를 보내는
방식이다. 이 프로토콜은 확률적으로 전체 시스템에 특정 메시지를 받을 수 있는 높은
확률을 제공한다. Gossip Protocol은 일반적으로 노드 멤버십 목록 유지, 합의 도달,
장애 감지 등의 목적으로 사용된다.

주요 장점은 확장성, 내결함성, 강건성, 수렴 일관성, 분산성, 간단함, 결합가능성 및
부하 경계이다. 단점으로는 최종 일관성, 네트워크 분할 인식 불능, 상대적으로 높은
대역폭 소비, 높은 대기 시간, 디버깅 및 테스트의 어려움 등이 있다. 분산
시스템에서 일치성을 선호할 때 Gossip Protocol은 다양한 애플리케이션에서
사용되며, Aamzon Dynamo, Apache Cassandra, Consul 등의 실제 솔루션에서 사용되고
있다.

다음은 Gossip 을 구현한 java code 이다.

```java
// Node.java
import java.util.Random;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.HashSet;

public class Node {
  private final String id;
  private final Set<String> knownMessages;
  private List<Node> neighbors;

  public Node(String id) {
    this.id = id;
    this.knownMessages = new HashSet<>();
    this.neighbors = new ArrayList<>();
  }

  public void addNeighbor(Node node) {
    neighbors.add(node);
  }

  public void receiveMessage(String message) {
    if (!knownMessages.contains(message)) {
      System.out.println("Node " + id + " got a new message: " + message);
      knownMessages.add(message);
      spreadMessageToRandomNeighbors(message);
    }
  }

  private void spreadMessageToRandomNeighbors(String message) {
    int randomNeighborsCount = new Random().nextInt(2) + 1; // 선택할 무작위 이웃 수 (1-2개)
    List<Node> alreadySentTo = new ArrayList<>();
    while (randomNeighborsCount > 0 && alreadySentTo.size() < neighbors.size()) {
      int randomIndex = new Random().nextInt(neighbors.size());
      Node chosenNode = neighbors.get(randomIndex);

      if (!alreadySentTo.contains(chosenNode)) {
        chosenNode.receiveMessage(message);
        alreadySentTo.add(chosenNode);
        randomNeighborsCount--;
      }
    }
  }
}

// GossipProtocolExample.java
public class GossipProtocolExample {
  public static void main(String[] args) {
    Node nodeA = new Node("A");
    Node nodeB = new Node("B");
    Node nodeC = new Node("C");
    Node nodeD = new Node("D");
    Node nodeE = new Node("E");

    // 노드들 간에 이웃 관계 설정
    nodeA.addNeighbor(nodeB);
    nodeA.addNeighbor(nodeC);
    nodeB.addNeighbor(nodeA);
    nodeB.addNeighbor(nodeC);
    nodeB.addNeighbor(nodeD);
    nodeC.addNeighbor(nodeA);
    nodeC.addNeighbor(nodeB);
    nodeC.addNeighbor(nodeD);
    nodeD.addNeighbor(nodeB);
    nodeD.addNeighbor(nodeC);
    nodeD.addNeighbor(nodeE);
    nodeE.addNeighbor(nodeD);

    // 최초 메시지 전달 (이 예제에서는 nodeA가 시작)
    nodeA.receiveMessage("Hello, Gossip Protocol!");
  }
}
```

Gossip Protocol 은 Amazon DynamoDB 에서 다음과 같이 사용된다.

- **클러스터 멤버십 관리**: Gossip Protocol은 DynamoDB가 각 노드가 클러스터의 다른
  노드에 대한 지식을 유지하도록 돕습니다. 각 노드는 일정한 간격으로 무작위로
  선택한 이웃 노드와 정보를 교환합니다. 곧, 모든 노드는 전체 클러스터의 상태에
  대해 알게되며, 최신 정보를 유지할 수 있습니다.
- **장애 감지**: Gossip Protocol은 클러스터에서 노드 장애를 감지하는 데 사용됩니다.
  노드가 주기적으로 이웃 노드와 정보를 교환하면서 올바르게 작동하지 않거나
  응답하지 않는 노드를 식별할 수 있습니다. 장애가 발견되면 메시지가 전체
  클러스터로 전파됩니다. 그러면 다른 노드들은 적절한 조치를 취할 수 있습니다(예:
  데이터 복제를 위한 새로운 노드 선택 등).

Gossip Protocol을 사용함으로써 DynamoDB는 시스템 규모를 확장하고, 성능을
향상시키며, 신속한 장애 감지와 교정을 보장할 수 있습니다. 하지만 DynamoDB는
솔루션 전체에서 Gossip Protocol만 사용하는 것이 아니며 다른 시스템 구성 요소와
통합하여 작동합니다. 따라서 Gossip Protocol은 DynamoDB에서 중요한 기능을
수행하지만 전체 시스템 운영에 있어서는 한 부분에 해당합니다.

## Raft

[Raft](/distributedsystem/README.md#raft)

## Chubby

Chubby is a distributed lock service developed by Google for coordinating
activities and managing shared resources among distributed systems. It provides
a simple, coarse-grained locking mechanism that allows systems to maintain
consistency, achieve synchronization, and ensure fault tolerance across a
cluster of machines.

Chubby exposes a file-system-like interface to clients, using a hierarchical
namespace with directories and files. Each file in Chubby represents a lock that
can be acquired by a client, providing exclusive access to the shared resource
associated with that lock. Chubby files can also store small amounts of metadata
which help the distributed systems make decisions based on the lock's state.

Some key features of Chubby include:

* **Consistency & Failover**: Chubby uses the Paxos consensus algorithm to
  maintain consistent state across its servers and handle the failures of
  individual nodes. A small cluster of Chubby servers ensures high availability,
  with a single master server responsible for handling client requests at any
  given time.
* **Caching**: Clients can cache Chubby locks and lease them for a specified
  duration, minimizing the need for frequent communication with the Chubby
  servers, and improving performance.
* **Event Notifications**: Chubby allows clients to monitor the state of locks
  by providing event notifications in the form of watches, helping clients
  detect changes to lock ownership or content.
* **Lock Expiration & Keep-alive**: To prevent locks from being held
  indefinitely, Chubby locks have an expiration mechanism based on leases, which
  clients must periodically renew to maintain ownership. This ensures that locks
  held by failed clients are eventually released.

Chubby plays a crucial role in several Google services, such as the distributed
file system **Google File System (GFS)**, the distributed database **Bigtable**,
and the cluster management system **Borg**. Although **Chubby** is not available
as a standalone product, similar services like Apache **ZooKeeper** and **etcd**
are inspired by Chubby and provide comparable functionality in coordinating
distributed systems.

## Configuration Management Database (CMDB)

CMDB stands for Configuration Management Database. It is a central repository
that stores information about the hardware and software components, also known
as Configuration Items (CIs), within an IT environment, as well as the
relationships between these components. CMDB plays a critical role in IT Service
Management (ITSM) processes, such as incident management, problem management,
change management, and asset management.

The purpose of a CMDB is to maintain an accurate and up-to-date inventory of all
IT components, enabling better understanding and visibility of the
infrastructure. By documenting the interdependencies between CIs, CMDB helps
streamline various IT processes and decision-making capabilities. This
information allows organizations to:

* **Optimize IT resources**: Track and analyze resource utilization to identify
  underused or overused components and allocate resources more effectively.
* **Enhance incident and problem management**: Identify and troubleshoot issues
  more quickly by visualizing the relationships and dependencies between
  affected CIs.
* **Mitigate risk in change management**: Understand the potential impact of
  proposed changes on the interconnected systems and make informed decisions
  before implementing changes.
* **Support effective asset management**: Maintain an up-to-date, centralized
  repository of IT assets that helps in licensing, procurement, maintenance, and
  end-of-life planning.
* **Enable compliance management**: Ensure adherence to regulatory requirements
  and organizational policies by having a comprehensive view of the IT
  infrastructure.

Properly implemented and maintained CMDB is a valuable tool in managing complex
IT environments. However, implementing a CMDB can be challenging due to the
dynamic nature of IT systems and the need for regular updates to ensure
accuracy. Integrating automated discovery, configuration, and monitoring tools
can help to maintain the accuracy and effectiveness of the CMDB over time.

## A/B Test

A/B testing, also known as split testing or bucket testing, is a controlled
experimental approach used to compare the effectiveness of two or more variants
of a product, feature, or content to determine which one performs better based
on a specific metric. It is widely used in website and app design, online
marketing, advertising, and user experience optimization.

In an A/B test, the target audience is randomly divided into two or more groups,
with each group exposed to a different variant (A, B, or more). The variants can
be different designs, headlines, call-to-action buttons, page layouts, marketing
messages, or any other element that can influence user behavior. Users' actions
or responses are then measured and compared across the groups to determine the
variant that yields the best results in terms of the predefined metric, such as
conversion rate, click-through rate, user engagement, or time spent on the page.

Steps involved in A/B testing process:

* **Define the objective**: Determine the specific goal or metric that the test
  aims to improve.
* **Develop hypotheses**: Create one or more alternative versions of the element
  being tested based on assumptions and data about user preferences and
  behavior.
* **Create test variants**: Modify the element according to the hypotheses to
  create the different versions to be tested.
* **Randomly split the audience**: Assign users to different groups in a random
  manner, ensuring that each group receives one of the variants.
* **Run the test**: Conduct the experiment for a predefined duration or until a
  statistically significant sample size is achieved.
* **Analyze the results**: Compare the performance of each variant based on the
  defined metric and determine the winner.

A/B testing is valuable for making data-driven decisions, optimizing user
experiences, increasing sales or conversions, and refining marketing strategies.
However, it requires careful planning, well-defined objectives, and proper
statistical analysis to ensure the results are accurate and reliable.

## Actor Model

[Actor Model](/actormodel/README.md)

## Reactor vs Proactor

**Reactor** and **Proactor** patterns are design patterns used for handling
asynchronous input/output (I/O) operations in concurrent systems. They help
manage multiple events and I/O operations, such as requests from numerous
clients or resource-intensive tasks. The primary difference between these two
patterns is the way they handle I/O operations.

**Reactor Pattern**:

- The Reactor pattern uses event-driven, synchronous I/O operations. It uses an
  event loop to wait for I/O events, then dispatches these events to
  corresponding application-defined event handlers when they occur.
- When an I/O operation (like reading or writing to a network socket) is
  initiated, the Reactor pattern maintains a non-blocking mode, but it still
  blocks the application code while performing the actual I/O operation.
- Since the I/O operations are synchronous, the Reactor pattern usually requires
  multi-threading to handle multiple tasks simultaneously without blocking the
  entire system.
- The Reactor pattern is typically easier to implement and understand because of
  its synchronous I/O operations, but it can suffer from performance issues when
  dealing with many concurrent I/O operations, especially when some I/O
  operations take a long time to complete.

**Proactor Pattern**:

- The Proactor pattern uses asynchronous I/O operations to initiate potentially
  long-lasting tasks while ensuring the application does not block.
- In the Proactor pattern, the application posts asynchronous I/O requests to an
  I/O subsystem that handles the operation without blocking. A completion event
  is raised when the I/O operation is finished, and a completion handler is
  called to process the results.
- This pattern is well suited for systems that need to manage many concurrent
  I/O operations without blocking or relying heavily on multi-threading.
- However, the Proactor pattern is more complex to implement due to the
  inherently asynchronous nature of its I/O operations.

In summary, the Reactor pattern is based on synchronous event-driven I/O, while
the Proactor pattern is based on asynchronous I/O handling. The choice between
the two will depend on the specific requirements of your application, such as
the number of concurrent I/O operations, their expected completion times, and
the platforms or libraries being used.

In Java, Reactor and Proactor patterns can be implemented using different APIs
and mechanisms. The following are examples of how to implement these patterns in
Java:

**Reactor Pattern in Java**:

The Reactor pattern can be implemented in Java using the java.nio package, which
provides non-blocking I/O support. Here's a simplified example of a Reactor
pattern using a server accepting multiple client connections:

```java
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.*;
import java.net.*;

// A simple Reactor-based server
public class ReactorServer {
  public static void main(String[] args) throws IOException {
    Selector selector = Selector.open();
    ServerSocketChannel serverChannel = ServerSocketChannel.open();
    
    serverChannel.bind(new InetSocketAddress("localhost", 8080));
    serverChannel.configureBlocking(false);
    serverChannel.register(selector, SelectionKey.OP_ACCEPT);

    while (true) {
      selector.select(); // Wait for I/O events
      Iterator<SelectionKey> keys = selector.selectedKeys().iterator();
      
      while (keys.hasNext()) {
        SelectionKey key = keys.next();
        keys.remove();
        
        if (key.isAcceptable()) {
          ServerSocketChannel server = (ServerSocketChannel) key.channel();
          SocketChannel clientChannel = server.accept();
          clientChannel.configureBlocking(false);
          clientChannel.register(selector, SelectionKey.OP_READ);
          System.out.println("Accepted connection from: " + clientChannel);
        } else if (key.isReadable()) {
          SocketChannel clientChannel = (SocketChannel) key.channel();
          ByteBuffer buffer = ByteBuffer.allocate(1024);
          int bytesRead = clientChannel.read(buffer);
          
          if (bytesRead == -1) {
            clientChannel.close();
          } else {
            System.out.println("Read: " + new String(buffer.array(), 0, bytesRead));
            buffer.flip();
            clientChannel.write(buffer);
            buffer.clear();
          }
        }
      }
    }
  }
}
```

**Proactor Pattern in Java**:

The Proactor pattern relies on asynchronous I/O operations. With Java, you can
use the Asynchronous I/O API introduced in Java 7, which is part of the
`java.nio.channels` package. Here's a simple example of a Proactor pattern using
an asynchronous server:

```java
import java.io.IOException;
import java.net.*;
import java.nio.ByteBuffer;
import java.nio.channels.*;
import java.util.concurrent.*;

// A simple Proactor-based server
public class ProactorServer {
  public static void main(String[] args) throws IOException {
    AsynchronousServerSocketChannel serverChannel = AsynchronousServerSocketChannel.open();
    serverChannel.bind(new InetSocketAddress("localhost", 8080));

    CompletionHandler<AsynchronousSocketChannel, Object> handler = new CompletionHandler<>() {

      @Override
      public void completed(AsynchronousSocketChannel clientChannel, Object attachment) {
        serverChannel.accept(null, this); // Accept next connection

        ByteBuffer buffer = ByteBuffer.allocate(1024);
        clientChannel.read(buffer, null, new CompletionHandler<>() {
          @Override
          public void completed(Integer bytesRead, Object attachment) {
            if (bytesRead == -1) {
              try {
                clientChannel.close();
              } catch (IOException e) {
                e.printStackTrace();
              }
            } else {
              System.out.println("Read: " + new String(buffer.array(), 0, bytesRead));
              buffer.flip();
              clientChannel.write(buffer);
              buffer.clear();
            }
          }

          @Override
          public void failed(Throwable exc, Object attachment) {
            exc.printStackTrace();
          }
        });
      }

      @Override
      public void failed(Throwable exc, Object attachment) {
        exc.printStackTrace();
      }
    };

    serverChannel.accept(null, handler);

    try {
      Thread.sleep(Long.MAX_VALUE);
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
  }
}
```

This example demonstrates a simple single-threaded server implementing the
Reactor and Proactor patterns, respectively. In real-world applications, you
would typically have more advanced mechanisms to handle the I/O operations,
error handling, and resource sharing, among other things.

## Data Lake

- [Frequently Asked Questions About the Data Lakehouse | databricks](https://www.databricks.com/blog/2021/08/30/frequently-asked-questions-about-the-data-lakehouse.html)

A **data lake** is a low-cost, open, and durable storage system designed to
store any data type, such as **tabular data**, **text**, **images**, **audio**,
**video**, **JSON**, and **CSV**. The main advantage of **data lakes** is their
ability to store vast amounts of **structured** and **unstructured data** using
open standardized formats, typically **Apache Parquet** or **ORC**. 

This approach allows a large ecosystem of tools and applications to directly
work with the data and helps organizations avoid vendor lock-in while amassing
large quantities of data. 

Major cloud providers like AWS, Azure, and Google Cloud offer data lake
solutions like **AWS S3**, **Azure Data Lake Storage (ADLS)**, and 
**Google Cloud Storage (GCS)**. 

However, data lakes historically suffer from issues related to security,
quality, and performance, often requiring organizations to move subsets of data
into data warehouses to extract value.

## Data Warehouse

- [Frequently Asked Questions About the Data Lakehouse | databricks](https://www.databricks.com/blog/2021/08/30/frequently-asked-questions-about-the-data-lakehouse.html)

A **data warehouse** is a proprietary system designed to store, manage, and
analyze structured or semi-structured data for SQL-based analytics and business
intelligence. These systems are optimized for high performance, concurrency, and
reliability but usually come at a higher cost compared to data lakes. Data
warehouses primarily support **structured data** and have limited support for
**unstructured data** types like **images**, **sensor data**, **documents**, or
**videos**.

As data warehouses are built for SQL-based analytics, they usually do not
support open-source libraries and tools like TensorFlow, PyTorch, and
Python-based libraries natively, particularly for machine learning and data
science use cases. Organizations typically store subsets of their valuable
business data in data warehouses for fast, concurrent SQL and BI use cases,
while keeping larger datasets, which include unstructured data, in data lakes.

## Data Lakehouse

- [Frequently Asked Questions About the Data Lakehouse | databricks](https://www.databricks.com/blog/2021/08/30/frequently-asked-questions-about-the-data-lakehouse.html)

A **Data Lakehouse** is a modern data architecture that combines the best
features of **data lakes** and **data warehouses** to enable efficient and
secure AI and Business Intelligence (BI) directly on vast amounts of data stored
in data lakes. Data Lakehouses aim to address the challenges faced by data
lakes, such as security, quality, and performance, while also providing the
capabilities of data warehouses, such as optimized performance for SQL queries
and BI-style reporting.

**Data Lakehouses** allow organizations to store all data types (structured,
semi-structured, and unstructured) in one place and perform AI and BI directly,
without the need to move data between data lakes and data warehouses. They use
open-source technologies like **Delta Lake**, **Hudi**, and **Iceberg** and
provide features such as ACID transactions, fine-grained data security, low-cost
updates and deletes, first-class SQL support, and native support for data
science and machine learning.

Vendors focusing on Data Lakehouses include **Databricks**, **AWS**, **Dremio**,
and **Starburst**, among others. By using a Data Lakehouse architecture,
organizations benefit from a single, unified system that covers all data types
and a wide range of analytical use cases, from BI to AI, while leveraging open
standards and technologies to avoid vendor lock-in.

## API Security

API security refers to the practices and measures used to protect application
programming interfaces (APIs) from unauthorized access, misuse, or
vulnerabilities. Ensuring API security is crucial as APIs are a key component in
modern applications and provide a way for developers to connect software
systems, share data, and extend functionality. The following list explains some
important components of API security:

- **Use HTTPS**: Always use HTTPS to secure the communication between the client
  and the API server. This ensures data confidentiality and integrity by
  encrypting the data exchanged.
- **Use OAuth2**: OAuth2 is an industry-standard protocol for authorization that
  provides a secure way to manage access and permissions for API users.
- **Use WebAuthn**: WebAuthn is a web standard for secure and passwordless
  authentication, which helps protect against phishing attacks and improve user
  experience.
- **Use Leveled API Keys**: Implement different levels of API keys to provide
  varying access levels and permissions for different users and use cases. This
  helps ensure that users only have access to the API resources they need.
- **Authorization**: Ensure proper authorization mechanisms are in place to
  control user access to specific API resources and data based on their roles
  and permissions.
- **Rate Limiting**: Implement rate limiting to prevent abuse and overload of
  your API. This involves limiting the number of requests a user can make within
  a given period.
- **API Versioning**: Use API versioning to manage changes in your API in a way
  that doesn't break existing clients.
- **Whitelisting**: Permit access to your API only from specific trusted sources
  by maintaining a whitelist of allowed IP addresses, domain names, or
  applications.
- **Check OWASP API Security Risks**: Regularly assess your API against common
  security risks identified by the Open Web Application Security Project (OWASP)
  to ensure it's protected from known vulnerabilities.
- **Use API Gateway**: Implement an API gateway as a central point of access and
  management for your APIs. This allows for easy monitoring, security
  implementation, and traffic control.
- **Error Handling**: Ensure that your API returns appropriate error messages
  that do not reveal sensitive information or expose vulnerabilities.
- **Input Validation**: Validate all input data sent to your API to prevent
  potential attacks, such as SQL injections or cross-site scripting (XSS). This
  includes checking for correct data types, lengths, and allowable characters.

## Batch Processing vs Stream Processing

- [7 Best Practices for Data Governance](https://atlan.com/batch-processing-vs-stream-processing/)

대량 처리와 스트림 처리의 주요 차이점은 다음과 같습니다:

- 데이터 처리의 정의 및 특성
  - 대량 처리: 데이터를 큰 덩어리로 처리하며 일정 시간 동안 수집된 데이터를 모두
    한 번에 처리합니다.
  - 스트림 처리: 데이터를 실시간으로 지속적으로 처리하며, 데이터가 도착하자마자
    처리합니다. 
- 지연 시간 및 처리 시간
  - 대량 처리: 일정한 배치가 완료되거나 특정 스케줄에 의해 처리가 실행될 때까지
    데이터가 즉시 처리되지 않기 때문에 지연 시간이 높습니다.
  - 스트림 처리: 데이터가 시스템에 도착하는 즉시 처리되므로 실시간 분석 또는
    즉각적인 인사이트가 필요한 작업에 적합하며 지연 시간이 낮습니다.
- 사용 사례 및 응용 프로그램
  - 대량 처리: 즉각적인 데이터 처리가 필요하지 않은 상황에서 일반적으로
    사용됩니다. 예를 들어 월간 급여 처리, 일일 보고서 생성, 대규모 데이터 분석
    등이 해당됩니다.
  - 스트림 처리: 금융 분야의 사기 감지, 전자 상거래의 실시간 추천, 실시간
    대시보드 업데이트 등과 같이 데이터를 기반으로 한 즉각적인 조치가 필요한
    상황에서 사용됩니다.
- 오류 허용과 신뢰성
  - 대량 처리: 대량 처리 작업이 실패하면 중단 된 시점에서 다시 시작하거나 전체
    배치를 다시 처리할 수 있습니다.
  - 스트림 처리: 더 정교한 오류 허용 메커니즘이 필요합니다. 데이터 스트림이
    중단되면 데이터가 손실되지 않도록 시스템이 중단을 처리하는 방법이
    필요합니다.
- 확장성 및 성능
  - 대량 처리: 대량의 데이터를 한 번에 처리하기 때문에 처리량이 향상됩니다. 사용
    사례에 따라 수직(강력한 기계 추가) 또는 수평(기계 수 추가)으로 확장할 수
    있습니다.
  - 스트림 처리: 데이터의 다양한 속도를 처리할 수 있는 시스템이 수평으로
    확장되어야 합니다.
- 복잡성 및 설정
  - 대량 처리: 실시간 처리의 복잡성을 처리할 필요가 없기 때문에 설정 및 설계가
    더 간단할 수 있습니다.
  - 스트림 처리: 오류 허용을 보장하고 상태를 관리하며 시간 순서가 지정되지 않은
    데이터 이벤트와 같은 복잡한 설정이 필요합니다.
- 도구 및 플랫폼의 예
  - 대량 처리: Hadoop MapReduce, Apache Hive 및 Apache Spark의 대량 처리 기능이
    대표적입니다.
  - 스트림 처리: Apache Kafka Streams, Apache Flink 및 Apache Storm 등이 예시로
    들 수 있습니다.

## HeartBeat

- [HeartBeat](https://martinfowler.com/articles/patterns-of-distributed-systems/heartbeat.html)

서버의 가용성을 확인하기 위해 주기적으로 다른 서버에 메시지를 보내는 것이다.

**Problem**

여러 서버가 클러스터를 형성할 때, 각 서버는 사용되는 파티셔닝 및 복제 방식에
따라 데이터의 일부를 저장하는 것이다. 서버 실패의 적시 감지는 실패한 서버에 대한
데이터 요청을 처리하도록 다른 서버에 책임을 부여함으로써 수정 조치를 취하는 것이
중요하다.

**Solution**

정기적으로 송신 서버의 살아있음을 나타내는 요청을 다른 모든 서버에 보낸다. 요청
간격은 서버 간의 네트워크 왕복 시간보다 길게 설정해야 한다. 모든 수신 서버는
요청 간격의 배수인 타임아웃 간격 동안 대기한다. 

## Bloom Filter

[Bloom Filter](/bloomfilter/README.md)

## Distributed Locking

- [How to do distributed locking](https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html)

Redis 의 redlock 은 다음과 같은 단점이 있다.

- **Fencing Tokens를 생성하지 않음**: Redlock 알고리즘이 Fencing Tokens를
  생성하지 않으므로, 클라이언트 간의 경쟁 조건을 방지할 수 없습니다. 이로 인해
  여러 클라이언트가 동시에 같은 리소스에 접근하는 상황에서 안전하지 않을 수
  있습니다.
- **타이밍 가정을 사용**: Redlock은 동기적 시스템 모델에커튼 가정을 하고 있으며,
  이는 네트워크 지연, 프로세스 일시 중지, 클럭 오류에 대한 정확한 시간을 알 수
  있다고 가정합니다. 그러나 실패, 네트워크 지연, 클럭 오류 같은 분산
  시스템에서의 현실적인 문제로 인해 이러한 가정은 항상 지켜지지 않습니다.
- **일부 타이밍 문제가 발생할 경우 안전성을 위반할 수 있음**: Redlock의 강력한
  일관성 메커니즘이 완벽한지 확신할 수 없기 때문에, 어떠한 타이밍 문제로 인해
  Redlock이 안전성을 위반할 수 있습니다.

다음은 Redis, Redisson 을 이용하여 distributed lock 을 구현한 java code 이다.

```java
import org.redisson.Redisson;
import org.redisson.api.RLock;
import org.redisson.api.RedissonClient;
import org.redisson.config.Config;

public class RedissonExample {

    public static void main(String[] args) {
        // Redisson configuration
        Config config = new Config();
        config.useSingleServer().setAddress("redis://127.0.0.1:6379");

        // Create the Redisson client instance
        RedissonClient redisson = Redisson.create(config);

        // Define the resource to lock (e.g., some shared file or record)
        String resource = "myResource";

        // Acquire the lock for the resource
        RLock lock = redisson.getLock(resource);
        lock.lock(); // Blocking call, waits until the lock is available

        try {
            // Perform actions protected by the lock
            performProtectedActions();
        } finally {
            // Release the lock
            lock.unlock();
        }

        // Shutdown the Redisson client
        redisson.shutdown();
    }

    private static void performProtectedActions() {
        // Insert code that requires the distributed lock here
    }
}
```

분산 락에 대한 강력한 일관성과 사용 안전성을 필요로 하는 경우, 대신 Apache
ZooKeeper와 같은 합의 알고리즘을 사용하는 것이 좋습니다. ZooKeeper는 분산 락 및
기타 분산 시스템 작업을 처리하기 위해 안전하고 일관성 있게 설계되었습니다. 이런
경우, Redlock 대신 ZooKeeper를 사용하여 더 안전한 환경을 구현할 수 있습니다.

ZooKeeper 가 Redis 보다 좋은 이유는 다음과 같다.

- **안정성**: ZooKeeper는 자체 합의 프로토콜을 사용하여 특정 노드가 실패한
  경우에도 클러스터 내 노드간 동기화를 유지합니다. 이는 클러스터의 전체 상태를
  보다 안정적으로 만듭니다. 반면, Redisson은 Redis의 분산 락 메커니즘인
  Redlock에 의존합니다. 위에서 논의한 바와 같이, Redlock은 클러스터 작동에
  필요한 안정성과 상호 운용성을 제공하는 데 몇 가지 제한이 있습니다.
- **일관성**: ZooKeeper는 분산 락 용도로 설계되었으며, 높은 가용성과 일관성을
  제공하기 위해 설계되었습니다. ZooKeeper 클러스터의 노드들은 자체 합의
  알고리즘을 사용하여 데이터 동기화와 클러스터 일관성을 유지하며, 일반적으로 더
  강력한 일관성 보장이 있는데 도움이 됩니다. 반면, Redisson은 기본적으로 Redis에
  의존합니다. Redis는 원래 캐싱 및 메시징용으로 설계되었으며, Redis 클러스터의
  일관성보다 가용성에 중점을 두었습니다. 이로 인해 ZooKeeper가 분산 락 작업을
  처리하는 데 더 높은 일관성을 제공할 수 있습니다.
- **순찰 기능** (Fencing mechanisms): 위에서 언급한 바와 같이
  ZooKeeper클라이언트는 스스로 Fencing Tokens을 생성하는 것이 가능하고 이러한
  메커니즘은 분산 락에서 보장해야하는 격리 수준을 높여 줍니다. 이 토큰 사용을
  통해 ZooKeeper는 동시에 다른 클라이언트가 같은 락에 액세스하는 것을 방지해
  줍니다. 이와 반면에 Redlock은 이러한 메커니즘을 자체적으로 지원하지 않습니다.

ZooKeeper는 분산 시스템에서 일관성과 락의 안정성을 더 잘 보장합니다. 반면
Redisson은 일시적 데이터나 메시징 등 여타의 사용 사례에 더 적합할 수 있습니다.
작업을 수행하기 전에 구체적인 사용 케이스와 요구 사항을 분석하고, 이에 따라
적절한 도구를 선택해야 합니다.

다음은 ZooKeeper, Curator 를 이용하여 distributed lock 을 구현한 java code 이다.

```groovy
// Gradle
dependencies {
    implementation 'org.apache.curator:curator-recipes:5.1.0'
    implementation 'org.apache.curator:curator-framework:5.1.0'
    ...
}
```

```java
//
import org.apache.curator.framework.CuratorFramework;
import org.apache.curator.framework.CuratorFrameworkFactory;
import org.apache.curator.framework.recipes.locks.InterProcessMutex;
import org.apache.curator.retry.ExponentialBackoffRetry;

public class ZookeeperExample {

    public static void main(String[] args) {
        // ZooKeeper configuration
        String zkConnectionString = "localhost:2181";
        int baseSleepTimeMs = 1000;
        int maxRetries = 3;

        // Create the Curator client for ZooKeeper
        CuratorFramework zookeeperClient = CuratorFrameworkFactory.newClient(
                zkConnectionString,
                new ExponentialBackoffRetry(baseSleepTimeMs, maxRetries)
        );
        zookeeperClient.start();

        // Define the resource to lock (e.g., some shared file or record)
        String lockPath = "/locks/myResource";

        // Acquire the lock for the resource
        InterProcessMutex lock = new InterProcessMutex(zookeeperClient, lockPath);

        try {
            if (lock.acquire(10, TimeUnit.SECONDS)) { // Attempt to acquire the lock within 10 seconds
                try {
                    // Perform actions protected by the lock
                    performProtectedActions();
                } finally {
                    // Release the lock
                    lock.release();
                }
            } else {
                // Failed to acquire the lock, throw an exception or handle it in some way
                throw new IllegalStateException("Failed to acquire lock");
            }
        } catch (Exception e) {
            // Handle exceptions during lock acquisition or protected actions
            e.printStackTrace();
        } finally {
            // Close the ZooKeeper client
            zookeeperClient.close();
        }
    }

    private static void performProtectedActions() {
        // Insert code that requires the distributed lock here
    }
}
```

## Distributed Tracing

- [What is distributed tracing, and why is it important?](https://www.dynatrace.com/news/blog/what-is-distributed-tracing/)

분산 추적은 고유 식별자로 표시하여 분산 클라우드 환경을 통해 전파되는 요청을
관찰하는 방법입니다. 이를 통해 사용자 경험, 애플리케이션 계층 및 인프라에 대한
실시간 가시성을 제공하며, 애플리케이션이 복잡한 랜드스케이프에 걸쳐 더욱
분산되면서 중요해집니다. 분산 추적은 애플리케이션 성능 향상, 서비스 수준 계약
준수 및 내부 협업을 개선하고 탐지 및 수리까지 평균 시간을 줄입니다.

모놀리식 애플리케이션은 보다 휴대성 있는 서비스로 발전하면서 복잡한 클라우드
네이티브 아키텍처에 전통적인 모니터링 도구가 효과적으로 제공할 수 없습니다.
대신, 분산 추적은 이러한 환경에서의 관측 가능성에 대한 필수 요소가 되었습니다.

## Checksum

- [What Is a Checksum?](https://www.lifewire.com/what-does-checksum-mean-2625825)

-----

체크섬은 데이터 조각(보통 하나의 파일)에 대해 암호화 해시 함수라는 알고리즘을
실행한 결과입니다. 체크섬은 파일의 원본과 사용자의 버전을 비교하여 파일이
정품이고 오류가 없는지 확인하는 데 도움이 됩니다. 체크섬은 종종 해시합계, 해시
값, 해시 코드 또는 단순하게 해시라고도 불립니다.

체크섬을 사용하는 것은 파일이 올바르게 받아졌는지 확인하는 데 도움이 됩니다.
예를 들어, 다운로드한 파일에 대한 체크섬을 생성하고 원래 파일의 체크섬과
비교하여 두 파일이 동일한지 확인할 수 있습니다. 체크섬이 일치하지 않으면 여러
가지 원인이 있을 수 있고 원본 소스에서 다운로드한 파일이 올바른 파일인지
확인하는 데 체크섬을 사용할 수 있습니다.

체크섬 계산기는 체크섬을 계산하는 데 사용되는 도구로, 각기 다른 암호화 해시
함수를 지원하는 다양한 종류의 체크섬 계산기가 있습니다. 체크섬 계산기는 파일의
정합성을 검증하는 데 사용할 수 있는 여러 도구 중 하나입니다.

# System Design Interview

## Easy

* [Designing Consistent Hashing](practices/DesigningConsistentHashing/DesigningConsistentHashing.md)
* [Designing A Uniqe ID Generator In Distributed Systems](practices/DesigningAUniqeIDGeneratorInDistributedSystems/DesigningAUniqeIDGeneratorInDistributedSystems.md)
* [Designing Real-Time Gaming Leaderboard](practices/DesigningReal-TimeGamingLeaderboard/DesigningReal-TimeGamingLeaderboard.md)
* [Designing A URL Shortener](practices/DesigningUrlShorteningService/DesigningUrlShorteningService.md)
* [Designing Pastebin](practices/DesigningPastebin/DesigningPastebin.md)
* [Designing CDN](practices/DesigningCDN/DesigningCDN.md)
* [Designing Parking Garrage](practices/DesigningParkingGarrage/DesigningParkingGarrage.md)
* [Designing Hotel Reservation System](practices/DesigningHotelReservationSystem/DesigningHotelReservationSystem.md)
* [Designing Ticketmaster](practices/DesigningTicketmaster/DesigningTicketmaster.md)
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

# MSA (Micro Service Architecture)

[Micro Service Architecture | TIL](/msa/README.md)

# Cloud Design Patterns

[Cloud Design Patterns | TIL](clouddesignpattern.md)

# Enterprise Integration Patterns

[Enterprise Integration Patterns | TIL](/eip/README.md)

# DDD

[DDD | TIL](/ddd/README.md)

# Architecture

[Architecture | TIL](/architecture/README.md)
