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
  - [Cache](#cache)
    - [Use-Cases](#use-cases)
    - [Caching Strategies](#caching-strategies)
  - [Content Delivery Network](#content-delivery-network)
  - [Proxy](#proxy)
  - [Availability](#availability-1)
    - [Availability Patterns](#availability-patterns)
    - [The Nine's of availability](#the-nines-of-availability)
  - [Distributed System](#distributed-system)
  - [Software Design Principle](#software-design-principle)
  - [Read Heavy vs Write Heavy](#read-heavy-vs-write-heavy)
  - [Performance vs Scalability](#performance-vs-scalability)
  - [Latency vs Throughput](#latency-vs-throughput)
  - [Availability vs Consistency](#availability-vs-consistency)
    - [CAP (Consistency Availability Partition tolerance)](#cap-consistency-availability-partition-tolerance)
    - [PACELC (Partitioning Availability Consistency Else Latency Consistency)](#pacelc-partitioning-availability-consistency-else-latency-consistency)
  - [Consistency patterns](#consistency-patterns)
  - [Database](#database)
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
  - [Message Queues VS Event Streaming Platform](#message-queues-vs-event-streaming-platform)
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
  - [Index](#index)
  - [SQL vs NoSQL](#sql-vs-nosql)
  - [Hadoop](#hadoop)
  - [MapReduce](#mapreduce)
  - [Paxos](#paxos)
  - [Gossip protocol](#gossip-protocol)
  - [Chubby](#chubby)
  - [Configuration Management Database (CMDB)](#configuration-management-database-cmdb)
  - [A/B Test](#ab-test)
  - [Actor Model](#actor-model)
  - [Reactor vs Proactor](#reactor-vs-proactor)
  - [Data Lake](#data-lake)
  - [Data Warehouse](#data-warehouse)
  - [Data Lakehouse](#data-lakehouse)
- [System Design Interview](#system-design-interview)
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

> * [Domain Name System (DNS)](https://www.karanpratapsingh.com/courses/system-design/domain-name-system)
> * [How does the Domain Name System (DNS) lookup work? | bytebytego](https://blog.bytebytego.com/p/how-does-the-domain-name-system-dns?s=r)

----

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

사람이 읽을 수 있는 domain name 을 기계가 이해할 수 있는 IP address 로 번역하는 시스템이다.

* **NS record (name server)** - Specifies the DNS servers for your domain/subdomain.
* **MX record (mail exchange)** - Specifies the mail servers for accepting messages.
* **A record (address)** - Points a name to an IP address.
* **CNAME (canonical)** - Points a name to another name or `CNAME` (example.com to www.example.com) or to an `A` record.

| name      | type  | value      |
| --------- | ----- | ---------- |
| a.foo.com | A     | 192.1.1.15 |
| b.foo.com | CNAME | a.foo.com  |

* [Online DNS Record Viewer](http://dns-record-viewer.online-domain-tools.com/)

## Load Balancing

> * [Introduction to modern network load balancing and proxying](https://blog.envoyproxy.io/introduction-to-modern-network-load-balancing-and-proxying-a57f6ff80236)
> * [What is load balancing](https://avinetworks.com/what-is-load-balancing/)
> * [Introduction to architecting systems](https://lethain.com/introduction-to-architecting-systems-for-scale/)
> * [Load balancing](https://en.wikipedia.org/wiki/Load_balancing_(computing))

<p align="center">
  <img src="http://i.imgur.com/h81n9iK.png"/>
  <br/>
  <i><a href=http://horicky.blogspot.com/2010/10/scalable-system-design-patterns.html>Source: Scalable system design patterns</a></i>
</p>

A load balancer is a networking device or software that distributes network
traffic or workload across multiple servers or resources to optimize resource
utilization, enhance performance, and improve reliability. By distributing
incoming requests evenly among the available resources, a load balancer helps
prevent any single server from becoming overwhelmed, which might otherwise lead
to poor performance, slow response times, and even system failures.

Load balancers are commonly used in environments where high availability,
scalability, and fault tolerance are critical. There are several types and
techniques of load balancing, including:

* **Round Robin**: Requests are distributed sequentially to the available
  servers, cycling back to the first server after reaching the last.
* **Least Connections**: Requests are sent to the server with the fewest active
  connections, which helps in more evenly distributing the load.
* **Weighted Distribution**: Servers are assigned different weights based on
  their capacity, and requests are distributed proportionally to their weights.

The load balancer can operate at different levels in the OSI Model:

* **Layer 4 (Transport Layer) Load Balancing**: Load balancing is done based on
  TCP or UDP headers, distributing traffic without inspecting the content of the
  packets. This method is faster but less flexible than Layer 7 load balancing.
* **Layer 7 (Application Layer) Load Balancing**: Load balancing is done based
  on the content of the request, such as the URL, cookies, or HTTP headers. This
  method offers more flexibility and allows for advanced routing decisions, such
  as directing requests to specific servers based on the required processing.

In addition to on-premises hardware and virtual appliances, cloud providers like
Amazon Web Services (AWS), Google Cloud Platform (GCP), and Microsoft Azure
offer managed load balancing services that can be easily integrated into your
applications.

## Cache

<p align="center">
  <img src="http://i.imgur.com/Q6z24La.png"/>
  <br/>
  <i><a href=http://horicky.blogspot.com/2010/10/scalable-system-design-patterns.html>Source: Scalable system design patterns</a></i>
</p>

### Use-Cases

* Client caching
* CDN caching
* Web server caching
* Database caching
* Application caching
* Caching at the database query level
* Caching at the object level

### Caching Strategies

> [Top caching strategies | bytebytego](https://blog.bytebytego.com/p/top-caching-strategies)

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

## Content Delivery Network

<p align="center">
  <img src="http://i.imgur.com/h9TAuGI.jpg"/>
  <br/>
  <i><a href=https://www.creative-artworks.eu/why-use-a-content-delivery-network-cdn/>Source: Why use a CDN</a></i>
</p>

**Content Delivery Network (CDN)** is a system of distributed servers (also called
edge servers or nodes) designed to deliver web content and other digital assets
to users more efficiently. CDNs work by caching and storing content, such as
images, videos, web pages, scripts, and stylesheets, on multiple servers located
across different geographical locations. When a user requests a resource, the
content is delivered from the nearest or best-performing server rather than from
a single centralized server, ensuring faster load times, improved performance,
and reduced bandwidth consumption.

**Push CDN**: In a Push CDN, the content owner uploads or "pushes" the content
to the CDN servers. The CDN servers then serve as the origin for the content,
and they deliver it directly to the users. When there is an update or new
content, the content owner has to push the changes to the CDN servers again.
Push CDNs are typically suitable for scenarios where content updates are
infrequent or need to be strictly controlled by the origin server. 

**Advantages of Push CDN**:

* Better control over content expiration and versioning. 
* Reduced load on the origin server since content is pushed only once to the
  CDN.

**Disadvantages of Push CDN**:

* Requires manual management of uploading and updating content on the CDN. 
* May consume more storage on the CDN servers by storing infrequently accessed
  content.

**Pull CDN**: In a Pull CDN, the content remains on the origin server, and the
CDN servers "pull" the content from the origin when a user requests it for the
first time. The CDN server caches the content, and for subsequent requests, it
serves the cached version to the users. When the cached content expires or
becomes outdated, it is automatically pulled again from the origin server by the
CDN. Pull CDNs are suitable for dynamic websites and applications where content
updates are frequent.

**Advantages of Pull CDN**:

* Easy to set up, as it only requires a change in the URL or DNS records to
  point to the CDN. 
* Automatically updates and synchronizes content with the origin server.

**Disadvantages of Pull CDN**:

* For the first request, the user may experience slightly higher latency when
  the content is pulled from the origin server.
* Can increase the load on the origin server during cache misses or content
  updates.

## Proxy

* [Apache2 설치 (Ubuntu 16.04)](https://lng1982.tistory.com/288)
  
-----

<p align="center">
  <img src="http://i.imgur.com/n41Azff.png"/>
  <br/>
  <i><a href=https://upload.wikimedia.org/wikipedia/commons/6/67/Reverse_proxy_h2g2bob.svg>Source: Wikipedia</a></i>
  <br/>
</p>

![](img/foward_reverse_proxy.png)

**Forward Proxy**: A forward proxy, also known simply as a proxy, sits between
clients and external servers, typically on a local network. It receives requests
from clients, accesses the requested resources on their behalf, and then
forwards the responses back to the clients. Forward proxies are often used in
scenarios where there is a need for internet access control, content filtering,
or traffic monitoring.

Common use cases for forward proxies include:

* **Anonymity**: Clients can access resources on the internet without revealing
  their IP addresses, as the proxy server's IP is used instead.
* **Security**: Organizations can implement content filtering, URL blacklisting,
  or monitoring to ensure compliance with internal security policies.
* **Caching**: Forward proxies can cache frequently accessed pages and serve
  them to multiple clients, reducing load on external servers and improving
  response times.

**Reverse Proxy**: A reverse proxy sits between external clients and internal
servers, acting as a gateway to one or more servers, handling incoming requests
and forwarding them to the appropriate backend server. Reverse proxies are used
to improve the performance, security, and load balancing of backend servers.

Common use cases for reverse proxies include:

* **Load Balancing**: Distributing incoming requests across multiple backend
  servers to balance load and ensure smooth, uninterrupted service.
* **SSL Termination**: Handling SSL encryption and decryption at the reverse
  proxy level, offloading tasks from backend servers for improved performance.
* **Caching**: Storing and serving static content to clients, reducing load on
  backend servers and improving response times.
* **Security**: Protecting backend servers from direct exposure to external
  clients, reducing the risk of vulnerabilities, and mitigating DDoS attacks.

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

## Distributed System

[Distributed System | TIL](/distributedsystem/README.md)

## Software Design Principle

* [Design Principle | TIL](/designprinciple/README.md)

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

Latency 는 어떤 action 을 수행하고 결과를 도출하는데 걸리는 시간이다. Throughput
은 단위 시간당 수행하는 액션 혹은 결과의 수이다.

## Availability vs Consistency

### CAP (Consistency Availability Partition tolerance)

* [CAP Theorem @ medium](https://medium.com/system-design-blog/cap-theorem-1455ce5fc0a0)
* [The CAP Theorem](https://teddyma.gitbooks.io/learncassandra/content/about/the_cap_theorem.html)

----

![](/aws/img/1_rxTP-_STj-QRDt1X9fdVlA.jpg)

Brewer's theorem 이라고도 한다. Distributed System 은
**Consistency,Availability, Partition tolerance** 중 2 가지만 만족할 수 있다. 2
가지를 만족시키기 위해 1 가지를 희생해야 한다는 의미와 같다.

* **Consistency**
  * all nodes see the same data at the same time
  * 모든 node 가 같은 시간에 같은 data 를 갖는다.
* **Availability**
  * a guarantee that every request receives a response about whether it was
    successful or failed
  * 일부 node 에 장애가 발생해도 서비스에 지장이 없다.
* **Partition tolerance**
  * the system continues to operate despite arbitrary message loss or failure of
    part of the system.
  * node 간에 네트워크가 단절되었을 때 서비스에 지장이 없다.

MySQL 은 Distribute System 이 아니다. CAP 를 적용할 수 없다.

따라서 Distributed System 은 다음과 같이 분류할 수 있다.

* **CP (Consistency and Partition Tolerance)**
  * node1, node2, node3 이 있다. node3 이 Network Partition 되었다고 하자.
    Consistency 를 위해 node1, node2 가 동작하지 않는다. Availability 가 떨어진다.
  * banking 과 같이 Consistency 가 중요한 곳에 사용된다.
* **AP (Availabiltity and Paritition Tolerance)**
  * node1, node2, node3 가 있다. node3 가 Network Partition 되었다고 하자.
    node1, node2 가 동작한다. node3 에 write 된 data 가 node1, node2 에 전파되지 않았다.
    Consistency 는 떨어진다. 그러나 서비스의 지장은 없다. 즉, Availability 가 높다. node3 가 
    Network Partition 에서 복구된다면 그 data 는 다시 동기화 된다.
* **CA (Consistency and Partition Tolerance)**
  * 현실세계에서 Consistency, Partition Tolerance 를 둘다 만족하는 것은 어렵다.

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

## Consistency patterns

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

WebSocket is a communication protocol that enables bidirectional and real-time
communication between a client (typically a web browser) and a server over a
single, long-lived connection. It was introduced as part of the HTML5
specification and designed to overcome the limitations of traditional HTTP
communication, which is unidirectional and requires multiple connections for
continuous data exchange.

WebSocket operates over the same ports as HTTP and HTTPS (ports 80 and 443,
respectively). It starts with an HTTP or HTTPS handshake (known as WebSocket
handshake) to establish a connection using the "Upgrade" header. Once the
handshake is successful, the connection is upgraded to a WebSocket connection,
allowing for low-latency communication in both directions (client-to-server and
server-to-client) without the overhead of establishing new connections for each
message.

Key features of WebSocket:

* **Bidirectional communication**: WebSocket enables two-way communication,
  allowing both clients and servers to send messages to each other without
  requiring multiple connections or polling mechanisms.
* **Real-time communication**: As WebSocket operates over a single, long-lived
  connection, it allows for low latency, real-time communication between clients
  and servers.
* **Reduced overhead**: WebSocket uses a lightweight framing system to send and
  receive messages, which reduces the overhead associated with HTTP headers and
  allows for efficient data transfer.
* **Compatibility**: WebSocket is designed to work with existing web
  infrastructure, using the same ports as HTTP and HTTPS (80 and 443) and a
  similar handshake process to establish connections.

WebSocket is commonly used in web applications to implement real-time features
such as notifications, chat systems, live updates, and online gaming. While
WebSocket significantly improves real-time communication in web applications
compared to older techniques like HTTP polling or long polling, it may not be
suitable for all cases, especially when dealing with legacy systems or network
environments that restrict WebSocket traffic.

## Server-Sent Events (SSE)

Server-Sent Events (SSE) is a web standard that enables a server to push
real-time updates to clients over a single HTTP connection. SSE is designed to
handle unidirectional communication, where the server sends updates to clients
without the need for clients to explicitly request those updates. This makes SSE
particularly useful for scenarios where updates originate from the server-side,
such as notifications, live updates, or event streaming.

SSE uses text-based encoding for messages and relies on the standard HTTP
protocol, which makes it more compatible with existing network infrastructure
compared to WebSocket. The central component of SSE is the EventSource API,
which is built into modern web browsers and provides a simple JavaScript
interface for setting up and handling the SSE connection.

Key features of Server-Sent Events:

* **Unidirectional communication**: SSE is designed specifically for
server-to-client updates, allowing the server to push real-time updates to
connected clients efficiently. Real-time updates: As SSE operates over a single
HTTP connection, it allows servers to send real-time updates to connected
clients with low latency.
* **Text-based encoding**: Server-Sent Events use text-based encoding (typically
  UTF-8) for message payloads, which simplifies processing, debugging, and
  compatibility across platforms.
* **Reconnection capability**: The EventSource API handles connection losses,
  automatically attempting to reconnect to the server and resume updates when
  the connection is reestablished.
* **Message structure**: SSE messages can include event types, message IDs, and
  data payloads. This structure allows clients to handle different types of
  events using separate event listeners and resume event streams efficiently if
  the connection is lost.

While Server-Sent Events offer some advantages over traditional **long-polling**
techniques, they are limited to unidirectional communication. If bidirectional
communication is required, protocols such as **WebSocket** would be more
appropriate. Additionally, SSE is not supported in all web browsers (for
example, it's not natively supported in Internet Explorer), requiring the use of
polyfills or alternative methods for such cases.

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

A circuit breaker is a design pattern used in distributed systems and
microservices architectures to detect, prevent, and handle failures in a
graceful manner, improving the system's resilience. The circuit breaker pattern
is inspired by electrical circuit breakers that protect electrical circuits from
damage due to overloads or short circuits. Similarly, in software systems, a
circuit breaker monitors and protects services or components that could fail.

The circuit breaker pattern is typically implemented as a wrapper or middleware
around service calls or requests, especially for remote services. It maintains
the state and monitors the health of the service or component. The circuit
breaker's primary purpose is to prevent further requests or operations on a
failing or slow service to give it time to recover, and it offers fallback
mechanisms to handle such scenarios, such as default responses, error messages,
or cached data.

A circuit breaker has three main states in its life cycle:

* **Closed**: In the closed state, the circuit breaker allows requests to pass
  through to the service. It monitors the success or failure of these requests,
  and if the number of failures or response time exceeds a defined threshold,
  the circuit breaker enters the "open" state.
* **Open**: In the open state, the circuit breaker blocks further requests to
  the failing or slow service, returning fallback responses immediately. This
  state helps protect the service from additional load and allows it to recover.
  After a predefined timeout, the circuit breaker enters the "half-open" state.
* **Half-Open**: In the half-open state, the circuit breaker allows a limited
  number or percentage of requests to pass through to the service, testing its
  health. If these requests are successful and the service has recovered, the
  circuit breaker returns to the "closed" state. If the failures continue, the
  circuit breaker reverts to the "open" state.

By implementing the circuit breaker pattern, systems can prevent cascading
failures, reduce the impact of slow or failing services, provide fallback
mechanisms for handling errors, and improve the overall resilience and
fault-tolerance of distributed systems and microservices architectures.

## Rate Limiting

* [Rate Limiting Fundamentals | bytebytego](https://blog.bytebytego.com/p/rate-limiting-fundamentals)

-----

Rate limiting is a technique used in computer systems, APIs, and networks to
control the rate at which requests or data packets are processed. It enforces a
limit on the number of requests, transactions, or data allowed within a
specified time interval. The primary goal of rate limiting is to ensure fair
resource usage, maintain system stability, and protect services from excessive
load, abuse, or denial-of-service attacks.

Rate limiting can be implemented at various levels and in different components
of a system:

* **APIs and web services**: Rate limiting can be enforced at the application
  level to control the number of requests a client can make to an API or web
  service within a specific time frame. Common strategies include limiting the
  number of requests per second, per minute, or per hour, often using tokens or
  API keys to identify and track clients.
* **Databases and backend services**: Rate limiting can be applied to manage the
  resources consumed by backend services, such as databases, message queues, or
  caching systems, to prevent overloading or exhausting the available capacity.
* **Networks**: Rate limiting can be implemented at the network level to control
  bandwidth usage, prevent network congestion, and ensure fair distribution of
  network resources among clients or devices.

There are several rate-limiting algorithms, such as:

* **Token Bucket**: In this approach, tokens are added to a bucket at a fixed
  rate up to a maximum capacity. Each request or packet consumes a token from
  the bucket. If the bucket is empty, the request is either rejected or delayed
  until tokens are available.
* **Leaky Bucket**: The leaky bucket algorithm uses a fixed-size buffer (the
  bucket) that "leaks" or removes items at a constant rate. Incoming requests or
  packets are added to the buffer if space is available; otherwise, they are
  rejected or delayed.
* **Fixed Window**: This algorithm divides time into fixed-sized windows or
  intervals and tracks the number of requests or packets within each window. If
  a window reaches its maximum allowed count, additional requests or packets are
  rejected or delayed until the next window starts.
* **Sliding Window**: This approach improves upon the fixed window algorithm by
  using a dynamic time window that slides based on request timestamps, ensuring
  better fairness and smoother rate limits.

Implementing rate limiting effectively can help maintain the reliability,
performance, and security of computer systems, APIs, and networks, while also
promoting fair usage policies among clients, users, or devices.

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

## Message Queues VS Event Streaming Platform

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
는 Long data retention, repeated consumption 이 가능하다.

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

* [RESTful API](https://lifeisgift.tistory.com/entry/Restful-API-%EA%B0%9C%EC%9A%94)

----

한글로 멱등성이라고 한다. RESTful API 에서 같은 호출을 여러번 해도 동일한 결과를
리턴하는 것을 말한다.

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

> * [Consistent Hashing | TIL](/consistenthasing/README.md)
> * [Consistent Hashing and Random Trees:](https://www.akamai.com/es/es/multimedia/documents/technical-publication/consistent-hashing-and-random-trees-distributed-caching-protocols-for-relieving-hot-spots-on-the-world-wide-web-technical-publication.pdf) 
>   * Original paper

---

Consistent hashing is a distributed hashing technique used to efficiently
distribute keys across multiple nodes in a distributed system, such as a cluster
of cache servers or databases. It provides a stable and balanced distribution of
keys while minimizing the impact of adding or removing nodes, reducing the
number of key redistributions and rehashing operations required to maintain the
system.

Consistent hashing works by mapping keys and nodes onto a circular, fixed-size
hash ring using a predefined hash function. Each key and node is hashed, and the
resulting value determines its position on the ring. A key is assigned to the
closest node in the clockwise direction.

When a node is added or removed from the system, it only affects the keys that
were directly linked to that node, rather than causing a complete rehashing of
all keys across all nodes. This significantly reduces the amount of data that
needs to be moved and allows the system to scale more efficiently without
causing major disruption.

Some of the benefits of consistent hashing include:

* **Load balancing**: It efficiently distributes keys across the nodes, ensuring
  a balanced load for each node in the system.
* **Scalability**: Consistent hashing simplifies the process of adding or
  removing nodes by minimizing the key redistributions required, leading to less
  downtime and better handling of dynamic system growth.
* **Fault tolerance**: When a node fails, the keys can be easily reassigned to
  other available nodes, minimizing both data loss and service disruption.

Consistent hashing is commonly used in distributed systems, such as distributed
caches (e.g., Amazon's Dynamo or Akamai's content delivery network), distributed
databases, and load balancers.

## Index

> * [chidb | github](https://github.com/iamslash/chidb)
>   * index implementation

An index in a database is a data structure that improves the speed of data
retrieval operations by providing a more efficient way to look up rows/records
based on the values of specific columns. Just like an index in a textbook allows
you to quickly find information without scanning the entire book, a database
index allows for faster data lookups without scanning the entire table.

Indexes can significantly improve the performance of queries by reducing the
amount of data that the database management system (DBMS) needs to examine when
searching for specific values or ranges of values. However, creating and
maintaining indexes come with some overhead, as the indexes have to be updated
whenever the data in the table changes (insertions, deletions, or updates),
which may impact write performance.

In a relational database system, two primary types of indexes are:

* **Single-column index**: An index created on a single column.
* **Multi-column index** (also known as a compound or composite index): An index
  created on multiple columns, used when queries frequently filter or join on
  those columns together. 
  
Additionally, there are several types of indexing techniques used in databases, such as:

* **B-Tree index**: The most commonly used indexing method, suitable for a wide
  variety of queries and various types of data. It is a balanced tree structure
  that keeps the data sorted and provides fast search, insertion, and deletion
  operations.
* **Bitmap index**: A memory-efficient indexing technique that represents index
  data as a series of binary values (bits), ideal for columns with low
  cardinality (limited distinct values) and often used in data warehousing and
  analytical systems.
* **Hash index**: An index based on a hash function, which allows for quick and
  direct access to data records based on their hashed values. It is useful for
  exact match queries but not efficient for range-based queries.

Choosing the right index for a database depends on factors such as the type of
data, the nature of the queries, the frequency of reads versus writes, and the
specific requirements of the application. Proper indexing can lead to
significant performance improvements, but it's important to create and manage
indexes judiciously to avoid unnecessary overhead.

## SQL vs NoSQL

SQL and NoSQL are two types of database management systems that cater to
different data storage and retrieval requirements. The primary difference
between them lies in how the data is structured, organized, and queried.

SQL:

* SQL stands for Structured Query Language, which is a standardized programming
  language used for managing relational databases (RDBMS). Examples of SQL
  databases include SQL Server, Oracle, and PostgreSQL.
* SQL databases rely on a predefined, fixed schema and a structured table
  format, where the data is stored in rows and columns.
* They have strong support for ACID (Atomicity, Consistency, Isolation, and
  Durability) properties, providing high data consistency and integrity.
* SQL databases are well-suited for complex queries, multi-row transactions, and
  joins.
* They typically scale vertically by adding more computational resources (CPU,
  memory) to a single server.

NoSQL:

* NoSQL stands for "not only SQL" and refers to non-relational databases that do
  not use the standard SQL query language. Examples of NoSQL databases include
  **MongoDB**, **Cassandra**, and **Couchbase**.
* NoSQL databases support various data structures, such as key-value, document,
  column-family, and graph models, offering flexibility in data representation.
* They are designed for **scalability**, **high availability**, and **fault
  tolerance**, and often allow for **eventual consistency**, trading off some
  **consistency** for better **performance** and **distribution**.
* NoSQL databases are suitable for handling unstructured, semi-structured, or
  hierarchical data, large volumes of data, and simple queries.

They usually scale horizontally by distributing data and workload across
multiple servers or clusters.

Choosing between SQL and NoSQL databases largely depends on the specific
requirements of the application, the data model, expected query complexity,
consistency needs, and scaling requirements. SQL databases are generally
preferred when dealing with structured data, complex transactions, and
relationships between records, while NoSQL databases are often used for managing
large-scale, distributed, and schema-less data structures in highly concurrent
environments.

| Subject | SQL | NoSQL |
|:--------|:----|:------|
| Schema  | Strict schema | schema or schemaless |
| Querying  | SQL | some of SQL or UnSQL (Unstructured SQL) |
| Scalability | scale-up | scale-out |
| Reliability or ACID Compliancy | Stric ACID | Some of ACID |

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

## Paxos

[Paxos](/distributedsystem/README.md#paxos)

## Gossip protocol

> [Gossip protocol](http://highscalability.com/blog/2011/11/14/using-gossip-protocols-for-failure-detection-monitoring-mess.html)

Gossip protocol, also known as epidemic protocol, is a distributed communication
strategy used for disseminating information and maintaining the state of a
distributed system in a scalable, fault-tolerant, and efficient manner. It is
inspired by the way rumors or gossip spread in human social networks.

In a gossip protocol, each node in the system periodically exchanges information
with a random or selected subset of neighbors. The exchanged information may
include updates, new data, or the state of the system. This process of
information exchange continues in an iterative manner, resulting in a rapid and
robust dissemination of information across the entire network.

Gossip protocols have several key features that make them suitable for
large-scale, distributed systems:

* **Scalability**: The randomized, partial communication strategy of gossip
  protocols allows them to scale well to large numbers of nodes, without
  overwhelming any single node or network link.
* **Fault Tolerance**: Gossip protocols can continue to operate even when some
  nodes fail or become unreachable, as the remaining nodes continue to exchange
  information.
* **Self-healing**: Gossip protocols can automatically adjust to changes in the
  network, compensating for node failures, additions, or removals. The
  eventually consistent nature of the information exchange helps maintain a
  coherent global state over time.
* **Decentralization**: Gossip protocols do not rely on a central coordinator or
  hub, making them more resilient to targeted attacks or single points of
  failure.

Gossip protocols are used in various distributed systems and applications, such
as maintaining consistent state in distributed databases and caches (e.g.,
Amazon's Dynamo), disseminating updates in peer-to-peer networks, monitoring and
cluster membership management in datacenter infrastructures, and consensus
algorithms in distributed ledger technologies like blockchains.

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
java.nio.channels package. Here's a simple example of a Proactor pattern using
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

# System Design Interview

* [Designing A Rate Limiter](practices/DesigningAnApiRateLimiter/DesigningAnApiRateLimiter.md)
* [Designing Consistent Hashing](practices/DesigningConsistentHashing/DesigningConsistentHashing.md)
* [Designing A Key-Value Store](practices/DesigningAKey-ValueStore/DesigningAKey-ValueStore.md)
* [Designing A Uniqe ID Generator In Distributed Systems](practices/DesigningAUniqeIDGeneratorInDistributedSystems/DesigningAUniqeIDGeneratorInDistributedSystems.md)
* [Designing A URL Shortener](practices/DesigningUrlShorteningService/DesigningUrlShorteningService.md)
* [Designing A Web Crawler](practices/DesigningaWebCrawler/DesigningaWebCrawler.md)
* Designing A Notification System
* [Designing A News Feed System](practices/DesigningFacebooksNewsfeed/DesigningFacebooksNewsfeed.md)
* [Designing A Chat System](practices/DesigningFacebookMessenger/DesigningFacebookMessenger.md)
* [Designing A Search Autocomplete System](practices/DesigningTypeaheadSuggestion/DesigningTypeaheadSuggestion.md)
* [Designing Youtube](practices/DesigningYoutubeorNetflix/DesigningYoutubeorNetflix.md)
* Designing Google Drive
* [Designing Proximity Service](practices/DesigningProximityService/DesigningProximityService.md)
* [Designing Nearby Friends](practices/DesigningNearbyFriends/DesigningNearbyFriends.md)
* [Designing Google Maps](practices/DesigningGoogleMaps/DesigningGoogleMaps.md)
* [Designing Distributed Message Queue](practices/DesigningDistributedMessageQueue/DesigningDistributedMessageQueue.md)
* [Designing Metrics Monitoring and Alerting System](practices/DesigningMetricsMonitoringandAlertingSystem/DesigningMetricsMonitoringandAlertingSystem.md)
* [Designing Ad Click Event Aggregation](practices/DesigningAdClickEventAggregation/DesigningAdClickEventAggregation.md)
* [Designing Hotel Reservation System](practices/DesigningHotelReservationSystem/DesigningHotelReservationSystem.md)
* [Designing Distributed Email Service](practices/DesigningEmailService/DesigningEmailService.md)
* Designing S3-like Object Storage 
* [Designing Real-Time Gaming Leaderboard](practices/DesigningReal-TimeGamingLeaderboard/DesigningReal-TimeGamingLeaderboard.md)
* Designing Payment System
* Designing Digital Wallet
* Designing Stock Exchange
* [Designing Ticketmaster](practices/DesigningTicketmaster/DesigningTicketmaster.md)
* [Designing Pastebin](practices/DesigningPastebin/DesigningPastebin.md)
* [Designing Instagram](practices/DesigningInstagram/DesigningInstagram.md)
* [Designing Dropbox](practices/DesigningDropbox/DesigningDropbox.md)
* [Designing Twitter](practices/DesigningTwitter/DesigningTwitter.md)
* [Designing Twitter Search](practices/DesigningTwitterSearch/DesigningTwitterSearch.md)
* [Designing Uber Backend](practices/DesigningUberBackend/DesigningUberBackend.md)

# Real World Architecture

| Type | System | Reference(s) |
| -- | -- | -- |
| Data processing | **MapReduce** - Distributed data processing from Google | [research.google.com](http://static.googleusercontent.com/media/research.google.com/zh-CN/us/archive/mapreduce-osdi04.pdf) |
| Data processing | **Spark** - Distributed data processing from Databricks | [slideshare.net](http://www.slideshare.net/AGrishchenko/apache-spark-architecture) |
| Data processing | **Storm** - Distributed data processing from Twitter | [slideshare.net](http://www.slideshare.net/previa/storm-16094009) |
| Data store      | **Bigtable** - Distributed column-oriented database from Google | [harvard.edu](http://www.read.seas.harvard.edu/~kohler/class/cs239-w08/chang06bigtable.pdf) |
| Data store      | **HBase** - Open source implementation of Bigtable | [slideshare.net](http://www.slideshare.net/alexbaranau/intro-to-hbase) |
| Data store      | **[Cassandra](/cassandra/README.md)** - Distributed column-oriented database from Facebook | [slideshare.net](http://www.slideshare.net/planetcassandra/cassandra-introduction-features-30103666) |
| Data store      | **[DynamoDB](/dynamodb/README.md)** - Document-oriented database from Amazon | [harvard.edu](http://www.read.seas.harvard.edu/~kohler/class/cs239-w08/decandia07dynamo.pdf) |
| Data store      | **[MongoDB](/mongodb/README.md)** - Document-oriented database | [slideshare.net](http://www.slideshare.net/mdirolf/introduction-to-mongodb) |
| Data store      | **Spanner** - Globally-distributed database from Google | [research.google.com](http://research.google.com/archive/spanner-osdi2012.pdf) |
| Data store      | **[Memcached](/memcached/README.md)** - Distributed memory caching system | [slideshare.net](http://www.slideshare.net/oemebamo/introduction-to-memcached) |
| Data store      | **[Redis](/redis/README.md)** - Distributed memory caching system with persistence and value types | [slideshare.net](http://www.slideshare.net/dvirsky/introduction-to-redis) |
| Data store      | **Couchbase** - an open-source, distributed multi-model NoSQL document-oriented database | [couchbase.com](https://www.couchbase.com/) |
| Data store      | **[Elasticsearch](/elasticsearch/README.md)** | [Elasticsearch @ TIL](/elasticsearch/README.md) |
| File system     | **Google File System (GFS)** - Distributed file system | [research.google.com](http://static.googleusercontent.com/media/research.google.com/zh-CN/us/archive/gfs-sosp2003.pdf) |
| File system     | **[Hadoop File System (HDFS)](/hadoop/README.md)** - Open source implementation of GFS | [apache.org](https://hadoop.apache.org/docs/r1.2.1/hdfs_design.html) |
| File system     | **GlusterFS** - Distributed File System | [GlusterFS](/GlusterFS/README.md) |
| Monitoring      | **[Graylog](/graylog/README.md)** | [Graylog @ TIL](/graylog/README.md) |
| Monitoring      | **Prometheus** | [Prometheus @ TIL](/prometheus/README.md) |
| Monitoring      | **[Grafana](/grafana/README.md)** | [Grafana @ TIL](/grafana/README.md) |
| CI/CD           | **[Jenkins](/jenkins/README.md)** | [Jenkins @ TIL](/jenkins/README.md) |
| Misc            | **Chubby** - Lock service for loosely-coupled distributed systems from Google | [research.google.com](http://static.googleusercontent.com/external_content/untrusted_dlcp/research.google.com/en/us/archive/chubby-osdi06.pdf) |
| Misc            | **Dapper** - Distributed systems tracing infrastructure | [research.google.com](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36356.pdf) |
| Misc            | **[Kafka](/kafka/README.md)** - Pub/sub message queue from LinkedIn | [slideshare.net](http://www.slideshare.net/mumrah/kafka-talk-tri-hug) |
| Misc            | **[Zookeeper](/zookeeper/README.md)** - Centralized infrastructure and services enabling synchronization | [slideshare.net](http://www.slideshare.net/sauravhaloi/introduction-to-apache-zookeeper) |
| Misc            | **ØMQ** - a high-performance asynchronous messaging library, aimed at use in distributed or concurrent applications. | [zeromq.org](http://zeromq.org/) |
| Misc            | **[etcd](/etcd/README.md)** - A distributed, reliable key-value store for the most critical data of a distributed system. | [etcd docs](https://coreos.com/etcd/docs/latest/) |
| Misc            | **Mosquitto** - An open source MQTT broker. [MQTT](/mqtt/README.md) is a Standard for IoT Messaging |  |
| Misc            | **Netty** - Netty is a NIO client server framework. | |

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

> * [Micro Service Architecture | TIL](/msa/README.md)
>   * [ftgo monolith src | github](https://github.com/microservices-patterns/ftgo-monolith)
>   * [ftgo msa src | github](https://github.com/microservices-patterns/ftgo-application)
> * [A pattern language for microservices](https://microservices.io/patterns/index.html)

Microservice architecture is a software development approach that structures an
application as a collection of small, modular, and independently deployable
services. Each microservice is designed to perform a single, specific function
and can be developed, deployed, and scaled independently from the rest of the
application. The microservices communicate with each other using lightweight
APIs, standardized protocols, or messaging mechanisms.

This architecture offers a more granular and flexible approach to building
complex applications compared to traditional monolithic architecture, where the
entire application is built as a single, cohesive unit.

Key characteristics of microservice architecture include:

* **Decentralized and modular**: Each microservice is a self-contained unit with
  its own codebase, data storage, and infrastructure, allowing it to be
  developed and managed independently.
* **Domain-driven design**: Microservices are aligned with specific business
  capabilities or domain concepts, promoting a better understanding of the
  application's purpose and functionality.
* **Scalability**: Each microservice can be scaled independently, enabling
  efficient resource utilization and handling variable loads for different parts
  of an application.
* **Flexibility**: Different microservices can be developed using different
  programming languages, frameworks, and tools, allowing teams to choose the
  most suitable technology for their specific service.
* **Resilience**: The failure or issues in one microservice are less likely to
  impact the entire application since each service is isolated and can be
  designed with the necessary fault tolerance and fallback mechanisms.

However, microservice architecture also introduces some challenges, such as
increased complexity in managing **inter-service communication**, **service
discovery**, and **data consistency**. To address these challenges,
organizations often rely on tools and techniques like **containerization**,
**service meshes**, **API gateways**, and **distributed tracing**.

Microservice architecture has become popular among organizations looking to
build large, complex, and highly scalable applications that can be easily
updated and maintained without causing major disruptions to the entire system.
Examples of companies using microservice architecture include Amazon, Netflix,
and Spotify.

# Cloud Design Patterns

[Cloud Design Patterns | TIL](clouddesignpattern.md)

# Enterprise Integration Patterns

[Enterprise Integration Patterns | TIL](/eip/README.md)

# DDD

> [DDD | TIL](/ddd/README.md)

Domain-Driven Design (DDD) is a software development approach that emphasizes
the importance of understanding and modeling the core business domain, which
refers to the subject area or problem scope that the software is intended to
address. DDD focuses on creating software that accurately reflects the needs,
concepts, and rules of the real-world domain by utilizing a common language,
models, and patterns.

DDD requires close collaboration between domain experts, such as business
stakeholders, and software developers to ensure that domain knowledge is
effectively translated into the software design. Key aspects of DDD include:

* **Ubiquitous Language**: A shared and consistent language that is used by both
  domain experts and developers to describe the domain concepts, rules, and
  processes. This language helps eliminate misunderstandings and improves
  communication among team members.
* **Bounded Context**: A logical boundary that encapsulates a specific part of
  the domain, separating it from other parts to maintain focus, reduce
  complexity, and ensure consistency within that context. Bounded Contexts allow
  different teams or parts of an application to evolve independently.
* **Entities, Value Objects, and Aggregates**: These are building blocks for
  creating the domain model. Entities are objects with a distinct identity
  (e.g., customer, product), while value objects are immutable and defined only
  by their attributes (e.g., color, price). Aggregates are clusters of related
  objects, with a single root entity acting as the entry point for interactions.
* **Repositories and Factories**: Repositories are used to store, retrieve, and
  manage instances of entities and aggregates, abstracting away data persistence
  details. Factories are responsible for creating complex domain objects or
  aggregates, encapsulating object instantiation logic.
* **Domain Events**: Events that signify a change in the state of the domain or
  an important occurrence in the business process, which can trigger actions or
  side effects in other parts of the system.

Domain-Driven Design is particularly useful for complex and evolving software
systems where having a clear understanding and accurate representation of the
domain is critical. It helps create maintainable, flexible, and business-focused
software solutions by fostering a deep understanding of the domain, promoting
effective communication among team members, and leveraging well-defined models
and patterns.

# Architecture

* [Architecture | TIL](/architecture/README.md)
