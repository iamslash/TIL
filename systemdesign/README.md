- [Abstract](#abstract)
- [Materials](#materials)
- [Prerequisites](#prerequisites)
- [Principles](#principles)
  - [How to approach a system design interview question](#how-to-approach-a-system-design-interview-question)
  - [Scalability](#scalability)
  - [Performance vs scalability](#performance-vs-scalability)
  - [Latency vs throughput](#latency-vs-throughput)
  - [Availability vs consistency](#availability-vs-consistency)
  - [Consistency patterns](#consistency-patterns)
  - [Availability patterns](#availability-patterns)
  - [Domain name system](#domain-name-system)
  - [Content delivery network](#content-delivery-network)
  - [Load balancer](#load-balancer)
  - [Reverse proxy](#reverse-proxy)
  - [Application layer](#application-layer)
  - [Database](#database)
  - [Cache](#cache)
  - [Asynchronism](#asynchronism)
  - [Communication](#communication)
  - [Security](#security)
  - [Database Primary Key](#database-primary-key)
  - [Coordinator, discovery](#coordinator-discovery)
- [Grokking the System Design Interview Practices](#grokking-the-system-design-interview-practices)
- [System Design Primer Practices](#system-design-primer-practices)
- [Additional System Design Interview Questions](#additional-system-design-interview-questions)
- [Real World Architecture](#real-world-architecture)
- [Company Architectures](#company-architectures)
- [company engineering blog](#company-engineering-blog)
- [System Design Pattern](#system-design-pattern)
  - [aws cloud design pattern](#aws-cloud-design-pattern)
  - [azure cloud design pattern](#azure-cloud-design-pattern)
  - [google cloud design pattern](#google-cloud-design-pattern)
- [Cracking The Coding Interview Quiz](#cracking-the-coding-interview-quiz)

----

# Abstract

- 시스템 디자인에 대해 적어본다. [system deisgn primer](https://github.com/donnemartin/system-design-primer#federation)
  이 너무 잘 정리 되 있어서 기억할 만한 주제들을 열거해 본다.

# Materials

- [A pattern language for microservices](https://microservices.io/patterns/index.html)
  - microservices 의 기본개념
- [cracking the coding interview](http://www.crackingthecodinginterview.com/)
* [Designing Data-Intensive Applications](https://dataintensive.net/)
* [Grokking the System Design Interview](https://www.educative.io/collection/5668639101419520/5649050225344512)
  - 유료 시스템 디자인 인터뷰
* [Grokking the Object Oriented Design Interview](https://www.educative.io/collection/5668639101419520/5692201761767424)
  - 유료 OOD 인터뷰 
- [system deisgn primer](https://github.com/donnemartin/system-design-primer#federation)
  - 킹왕짱 내가 다시 정리할 필요성을 못 느낀다.
- [Azure Cloud Patterns](https://docs.microsoft.com/en-us/azure/architecture/patterns/)
  - [infographic](https://azure.microsoft.com/en-us/resources/infographics/cloud-design-patterns/)
- [AWS Architect](https://aws.amazon.com/ko/architecture/)
- [GCP Solutions](https://cloud.google.com/solutions/)

# Prerequisites

- powers of two table

```
Power           Exact Value         Approx Value        Bytes
---------------------------------------------------------------
7                             128
8                             256
10                           1024   1 thousand           1 KB
16                         65,536                       64 KB
20                      1,048,576   1 million            1 MB
30                  1,073,741,824   1 billion            1 GB
32                  4,294,967,296                        4 GB
40              1,099,511,627,776   1 trillion           1 TB
```

- latency numbers every programmer should know

```
Latency Comparison Numbers
--------------------------
L1 cache reference                           0.5 ns
Branch mispredict                            5   ns
L2 cache reference                           7   ns                      14x L1 cache
Mutex lock/unlock                          100   ns
Main memory reference                      100   ns                      20x L2 cache, 200x L1 cache
Compress 1K bytes with Zippy            10,000   ns       10 us
Send 1 KB bytes over 1 Gbps network     10,000   ns       10 us
Read 4 KB randomly from SSD*           150,000   ns      150 us          ~1GB/sec SSD
Read 1 MB sequentially from memory     250,000   ns      250 us
Round trip within same datacenter      500,000   ns      500 us
Read 1 MB sequentially from SSD*     1,000,000   ns    1,000 us    1 ms  ~1GB/sec SSD, 4X memory
Disk seek                           10,000,000   ns   10,000 us   10 ms  20x datacenter roundtrip
Read 1 MB sequentially from 1 Gbps  10,000,000   ns   10,000 us   10 ms  40x memory, 10X SSD
Read 1 MB sequentially from disk    30,000,000   ns   30,000 us   30 ms 120x memory, 30X SSD
Send packet CA->Netherlands->CA    150,000,000   ns  150,000 us  150 ms

Notes
-----
1 ns = 10^-9 seconds
1 us = 10^-6 seconds = 1,000 ns
1 ms = 10^-3 seconds = 1,000 us = 1,000,000 ns
```

![](img/latency_numbers_every_programmer_should_know.png)

- time

| years | days | hours | mins | secs |
|------:|-----:|------:|-----:|-----:|
| 1     | 365  | 8,760  | 525,600 | 31,536,000 |
|       | 1    | 24    | 1,440 | 86,400 |
|       |      | 1    | 60 | 3,600 |
|       |      |      | 1 | 60 |

- terms
  - nas, san, das 
  - saas paas iaas 
  - waf firewall
  - osi 7 layer
  - how to make onpremise vpc
  - restfull api advantages, disadvantages
  - msa
  - hdfs 
  - virtualization 3 types
  - devops

# Principles

## How to approach a system design interview question

* Outline use cases, constraints, and assumptions
* Create a high level design
* Design core components
* Scale the design

## Scalability

- vertical scaling
- horizontal scaling
- caching
- load balancing
- database replication
- database partitioning
- asynchronism

## Performance vs scalability

## Latency vs throughput

## Availability vs consistency

* [CAP Theorem, 오해와 진실](http://eincs.com/2013/07/misleading-and-truth-of-cap-theorem/)
  * CAP이론은 논란이 많다. CAP보다 PACELC를 이용하자.
  * Partition(장애)상활일때 A(Availability) 혹은 C(Consistency)가
    중요하냐 Else(정상)상황일때 L(Latency) 혹은 C(Consistency)가
    중요하냐
  * HBase는 PC/EC이다. 장애 상황일때 C를 위해 A를 희생한다. 정상
    상황일때 C를 위해 L를 희생한다.
  * Cassandra는 PA/EL이다. 장애 상황일때 A를 위해 C를 희생한다. 즉
    Eventual Consistency의 특성을 갖는다. 정상 상황일때 L을 위해 C를
    희생한다. 즉 모든 노드에 데이터를 반영하지는 않는다.

## Consistency patterns

* Weak consistency
* Eventual consistency
* Strong consistency

## Availability patterns

* Fail-over
* Replication

## Domain name system

## Content delivery network

* Push CDNs
* Pull CDNs

## Load balancer

* Active-passive
* Active-active
* Layer 4 load balancing
* Layer 7 load balancing
* Horizontal scaling

## Reverse proxy

* [Apache2 설치 (Ubuntu 16.04)](https://lng1982.tistory.com/288)
  
-----

![](img/foward_reverse_proxy.png)

forward proxy 는 HTTP req 를 인터넷에 전달한다. reverse proxy 는 HTTP 를 요청을 인터넷으로부터 HTTP req 를 받아서 back-end 서버들에게 전달한다. L4, L7 스위치도 reverse proxy 라고 할 수 있다. reverse 라는 말은 왜 사용되었을까???

`reverse proxy` 는 `load balaning` 혹은 `SPOF (single point failure)` 를 위해 사용된다.

## Application layer

* Microservices
* Service discovery

## Database

* RDBMS
  * ACID - set of properties of relational database transactions
    * Atomicity(원자성) - Each transaction is all or nothing
    * Consistency(일관성) - Any transaction will bring the database from one valid state to another
    * Isolation(고립성) - Executing transactions concurrently has the same results as if the transactions were executed serially
    * Durability(영속성) - Once a transaction has been committed, it will remain so.
  * Master-slave replication
  * Master-Master replication
* Federation
* Sharding
  * [consistent hashing](/consistent_hasing/README.md)
* Denormalization
* SQL Tuning
* NoSQL
  * Key-value store
  * Document store
  * Wide solumn store
  * Graph database

## Cache

* Client caching
* CDN caching
* Web server caching
* Database caching
* Application caching
* Caching at the database query level
* CAching at the object level
* When to update the cache
  * Cache-Aside
    * 응용프로그램이 직접 cache를 제어한다.

```python
# reading values
v = cache.get(k)
if (v == null) {
  v = sor.get(k)
  cache.put(k, v)
}

# writing values
v = newV
sor.put(k, v)
cache.put(k, v)
```
  * Read-through
    * `cache` 에 읽기 요청하면 `cache` 가 판단해서 자신이 가지고 있는 값 혹은
      `SOR(system of record)` 로 부터 읽어들인 값을 응답으로 전달한다.
  
  * Write-through
    * `cache` 에 쓰기 요청하면 `cache` 가 판단해서 `SOR(system of record)` 에
      쓰고 자신을 갱신한다.

  * Write-behind
    * `cache` 에 쓰기 요청하면 일단 자신을 갱신하고 요청에 응답한후
      `SOR` 을 갱신한다. `SOR` 갱신이 완료되기 전에 요청에 빠르게 응답한다.

## Asynchronism

* Message Queues
* Task Queues
* Back pressure

## Communication

* TCP
* UDP
* RPC
* REST

## Security

## Database Primary Key

* [강대명 <대용량 서버 구축을 위한 Memcached와 Redis>](https://americanopeople.tistory.com/177)

----

예를 들어 이메일 시스템을 디자인한다고 해보자. User 와 Email 테이블의 스키마는 다음과 같다. 

| field | type | description |
|-------|------|-------------|
| user_id | Long (8B) | unique id (각 DB 별) |
| email | String | 이메일 주소 |
| shard | Long | 자신의 메일 리스트를 저장한 DB server 번호 |
| type | int | 활성화 유저인가?? |
| created_at | timestamp | 계정 생성시간 |
| last_login_time | timestamp | 마지막 로그인 시간 |

| field | type | description |
|-------|------|-------------|
| mail_id | Long (8B) | unique id (각 DB 별) |
| receiver | String or Long | 수신자 |
| sender | String or Long | 송신자 |
| subject | String | 메일제목 |
| received_at | timestamp | 수신시간 |
| eml_id | String or Long | 메일 본문 저장 id or url |
| is_read | boolean | 읽었는가?? |
| contents | String | 미리보기 (내용의 일부) |

eml 은 AWS S3 에 저장하자. eml file 의 key 를 마련해야 한다. 

* `{receiver_id}_{mail_id}` 
  * `mail_id` 는 이미 shard 마다 중복해서 존재한다. 따라서 `receiver_id` 와 결합하여 사용하자.
  * 그렇다면 `eml_id` 는 필요할까? `{receiver_id}_{mail_id}` 만으로도 eml file 의 key 로 사용할 수 있기 때문이다. 조금 더 key 를 잘 설계할 수는 없을까???
* UUID (Universally Unique Identifier)
  * id 에 시간 정보가 반영되어 있다. id 를 오름차순으로 정렬하면 시간순 으로 데이터를 정렬할 수 있다.
  * 16B (128b), 36 characters 이다. 너무 크다.
  * 적은 바이트로 시간 정보를 저장할 수 있었으면 좋겠다.
* `{timestamp:52bit}_{sequence:12bit}` 8 bytes
  * 샤드 아이디도 저장되었으면 좋겠다.
  * timestamp 는 4 bytes 이다. 단, `1970/01/01` 부터 `2016/02/07/06/28` 까지만 표현 가능하다.  
* `{timestamp:52bit}_{shard_id:12bit}_{sequence:12bit}` 8 bytes 
  * IDC 정보도 반영되었으면 좋겠다.
* `{timestamp:42bits}_{datacenter_id:5bits}_{worker_id:5bits}_{sequence:12bits}` 8 bytes
  * 이것은 twitter 의 id 이다.
* `{timetamp:4B}_{machine_id:3B}_{process_id:2B}_{counter:3B}` 12 bytes
  * 이것은 mongoDB 의 ID 이다. 
* `{timestamp}_{shard_id}_{type}_{sequence}` 8 bytes

## Coordinator, discovery

service 들의 목록을 저장하고 살아있는지 검증한다. 변경된 사항은 등록된 service 들에게 공지한다. zookeeper, etcd, consul, eureka 가 해당된다.

# Grokking the System Design Interview Practices

| Question | |
|---|---|
| [Designing a URL Shortening service like TinyURL](Designing_a_URL_Shortening_service_like_TinyURL.md) |
| [Designing Pastebin](DesigningPastebin.md) |
| [Designing Instagram]() |
| [Designing Dropbox]() |
| [Designing Facebook Messenger]() |
| [Designing Twitter](DesigningTwitter.md) |
| [Designing Youtube or Netflix]() |
| [Designing Typeahead Suggestion]() |
| [Designing an API Rate Limiter]() |
| [Designing Twitter Search](DesigningTwitterSearch.md) |
| [Designing a Web Crawler]() |
| [Designing Facebook’s Newsfeed](DesigningFacebooksNewsfeed.md) |
| [Designing Yelp or Nearby Friends]() |
| [Designing Uber backend]() |
| [Design Ticketmaster]() |
| [Dynamo](https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf) - Highly Available Key-value Store |
| [Kafka](http://notes.stephenholiday.com/Kafka.pdf) - A Distributed Messaging System for Log Processing |
| [Consistent Hashing](https://www.akamai.com/es/es/multimedia/documents/technical-publication/consistent-hashing-and-random-trees-distributed-caching-protocols-for-relieving-hot-spots-on-the-world-wide-web-technical-publication.pdf) - Original paper |
| [Paxos](https://www.microsoft.com/en-us/research/uploads/prod/2016/12/paxos-simple-Copy.pdf) - Protocol for distributed consensus |
| [Concurrency Controls](http://sites.fas.harvard.edu/~cs265/papers/kung-1981.pdf) - Optimistic methods for concurrency controls |
| [Gossip protocol](http://highscalability.com/blog/2011/11/14/using-gossip-protocols-for-failure-detection-monitoring-mess.html) - For failure detection and more. |
| [Chubby](http://static.googleusercontent.com/media/research.google.com/en/us/archive/chubby-osdi06.pdf) - Lock service for loosely-coupled distributed systems |
| [ZooKeeper](https://www.usenix.org/legacy/event/usenix10/tech/full_papers/Hunt.pdf) - Wait-free coordination for Internet-scale systems |
| [MapReduce](https://static.googleusercontent.com/media/research.google.com/en//archive/mapreduce-osdi04.pdf) - Simplified Data Processing on Large Clusters |
| [Hadoop](http://storageconference.us/2010/Papers/MSST/Shvachko.pdf) - A Distributed File System [hadoop @ TIL](/hadoop/README.md) |
| [Key Characteristics of Distributed Systems](Key_Characteristics_of_Distributed_Systems.md) |
| [Load Balancing](LoadBalancing.md) |
| [Caching](Caching.md) |
| [Data Partitioning](DataPartitioning.md) |
| [Indexes](Indexes.md) |
| [Proxies](Proxies.md) |
| [Redundancy and Replication](RedundancyandReplication.md) |
| [SQL vs. NoSQL](SQLvsNoSQL.md) |
| [CAP Theorem](CAPTheorem.md) |
| [Consistent Hashing](ConsistentHashing.md) |
| [Long-Polling vs WebSockets vs Server-Sent Events](Long-PollingvsWebSocketsvsServer-SentEvents.md) |

# System Design Primer Practices

| Question | |
|---|---|
| Design Pastebin.com (or Bit.ly) | [Solution](https://github.com/donnemartin/system-design-primer/blob/master/solutions/system_design/pastebin/README.md) |
| Design the Twitter timeline (or Facebook feed)<br/>Design Twitter search (or Facebook search) | [Solution](https://github.com/donnemartin/system-design-primer/blob/master/solutions/system_design/twitter/README.md) |
| Design a web crawler | [Solution](https://github.com/donnemartin/system-design-primer/blob/master/solutions/system_design/web_crawler/README.md) |
| Design Mint.com | [Solution](https://github.com/donnemartin/system-design-primer/blob/master/solutions/system_design/mint/README.md) |
| Design the data structures for a social network | [Solution](https://github.com/donnemartin/system-design-primer/blob/master/solutions/system_design/social_graph/README.md) |
| Design a key-value store for a search engine | [Solution](https://github.com/donnemartin/system-design-primer/blob/master/solutions/system_design/query_cache/README.md) |
| Design Amazon's sales ranking by category feature | [Solution](https://github.com/donnemartin/system-design-primer/blob/master/solutions/system_design/sales_rank/README.md) |
| Design a system that scales to millions of users on AWS | [Solution](https://github.com/donnemartin/system-design-primer/blob/master/solutions/system_design/scaling_aws/README.md) |

# Additional System Design Interview Questions

| Question | Reference(s) |
|---|---|
| Design a file sync service like Dropbox | [youtube.com](https://www.youtube.com/watch?v=PE4gwstWhmc) |
| Design a search engine like Google | [queue.acm.org](http://queue.acm.org/detail.cfm?id=988407)<br/>[stackexchange.com](http://programmers.stackexchange.com/questions/38324/interview-question-how-would-you-implement-google-search)<br/>[ardendertat.com](http://www.ardendertat.com/2012/01/11/implementing-search-engines/)<br>[stanford.edu](http://infolab.stanford.edu/~backrub/google.html) |
| Design a scalable web crawler like Google | [quora.com](https://www.quora.com/How-can-I-build-a-web-crawler-from-scratch) |
| Design Google docs | [code.google.com](https://code.google.com/p/google-mobwrite/)<br/>[neil.fraser.name](https://neil.fraser.name/writing/sync/) |
| Design a key-value store like Redis | [slideshare.net](http://www.slideshare.net/dvirsky/introduction-to-redis) |
| Design a cache system like Memcached | [slideshare.net](http://www.slideshare.net/oemebamo/introduction-to-memcached) |
| Design a recommendation system like Amazon's | [hulu.com](http://tech.hulu.com/blog/2011/09/19/recommendation-system.html)<br/>[ijcai13.org](http://ijcai13.org/files/tutorial_slides/td3.pdf) |
| Design a tinyurl system like Bitly | [n00tc0d3r.blogspot.com](http://n00tc0d3r.blogspot.com/) |
| Design a chat app like WhatsApp | [highscalability.com](http://highscalability.com/blog/2014/2/26/the-whatsapp-architecture-facebook-bought-for-19-billion.html)
| Design a picture sharing system like Instagram | [highscalability.com](http://highscalability.com/flickr-architecture)<br/>[highscalability.com](http://highscalability.com/blog/2011/12/6/instagram-architecture-14-million-users-terabytes-of-photos.html) |
| Design the Facebook news feed function | [quora.com](http://www.quora.com/What-are-best-practices-for-building-something-like-a-News-Feed)<br/>[quora.com](http://www.quora.com/Activity-Streams/What-are-the-scaling-issues-to-keep-in-mind-while-developing-a-social-network-feed)<br/>[slideshare.net](http://www.slideshare.net/danmckinley/etsy-activity-feeds-architecture) |
| Design the Facebook timeline function | [facebook.com](https://www.facebook.com/note.php?note_id=10150468255628920)<br/>[highscalability.com](http://highscalability.com/blog/2012/1/23/facebook-timeline-brought-to-you-by-the-power-of-denormaliza.html) |
| Design the Facebook chat function | [erlang-factory.com](http://www.erlang-factory.com/upload/presentations/31/EugeneLetuchy-ErlangatFacebook.pdf)<br/>[facebook.com](https://www.facebook.com/note.php?note_id=14218138919&id=9445547199&index=0) |
| Design a graph search function like Facebook's | [facebook.com](https://www.facebook.com/notes/facebook-engineering/under-the-hood-building-out-the-infrastructure-for-graph-search/10151347573598920)<br/>[facebook.com](https://www.facebook.com/notes/facebook-engineering/under-the-hood-indexing-and-ranking-in-graph-search/10151361720763920)<br/>[facebook.com](https://www.facebook.com/notes/facebook-engineering/under-the-hood-the-natural-language-interface-of-graph-search/10151432733048920) |
| Design a content delivery network like CloudFlare | [cmu.edu](http://repository.cmu.edu/cgi/viewcontent.cgi?article=2112&context=compsci) |
| Design a trending topic system like Twitter's | [michael-noll.com](http://www.michael-noll.com/blog/2013/01/18/implementing-real-time-trending-topics-in-storm/)<br/>[snikolov .wordpress.com](http://snikolov.wordpress.com/2012/11/14/early-detection-of-twitter-trends/) |
| Design a random ID generation system | [blog.twitter.com](https://blog.twitter.com/2010/announcing-snowflake)<br/>[github.com](https://github.com/twitter/snowflake/) |
| Return the top k requests during a time interval | [ucsb.edu](https://icmi.cs.ucsb.edu/research/tech_reports/reports/2005-23.pdf)<br/>[wpi.edu](http://davis.wpi.edu/xmdv/docs/EDBT11-diyang.pdf) |
| Design a system that serves data from multiple data centers | [highscalability.com](http://highscalability.com/blog/2009/8/24/how-google-serves-data-from-multiple-datacenters.html) |
| Design an online multiplayer card game | [indieflashblog.com](http://www.indieflashblog.com/how-to-create-an-asynchronous-multiplayer-game.html)<br/>[buildnewgames.com](http://buildnewgames.com/real-time-multiplayer/) |
| Design a garbage collection system | [stuffwithstuff.com](http://journal.stuffwithstuff.com/2013/12/08/babys-first-garbage-collector/)<br/>[washington.edu](http://courses.cs.washington.edu/courses/csep521/07wi/prj/rick.pdf) |
| Design an API rate limiter | [https://stripe.com/blog/](https://stripe.com/blog/rate-limiters) |
| Add a system design question | [Contribute](#contributing) |

# Real World Architecture

|Type | System | Reference(s) |
|---|---|---|
| Data processing | **MapReduce** - Distributed data processing from Google | [research.google.com](http://static.googleusercontent.com/media/research.google.com/zh-CN/us/archive/mapreduce-osdi04.pdf) |
| Data processing | **Spark** - Distributed data processing from Databricks | [slideshare.net](http://www.slideshare.net/AGrishchenko/apache-spark-architecture) |
| Data processing | **Storm** - Distributed data processing from Twitter | [slideshare.net](http://www.slideshare.net/previa/storm-16094009) |
| | | |
| Data store | **Bigtable** - Distributed column-oriented database from Google | [harvard.edu](http://www.read.seas.harvard.edu/~kohler/class/cs239-w08/chang06bigtable.pdf) |
| Data store | **HBase** - Open source implementation of Bigtable | [slideshare.net](http://www.slideshare.net/alexbaranau/intro-to-hbase) |
| Data store | **Cassandra** - Distributed column-oriented database from Facebook | [slideshare.net](http://www.slideshare.net/planetcassandra/cassandra-introduction-features-30103666)
| Data store | **DynamoDB** - Document-oriented database from Amazon | [harvard.edu](http://www.read.seas.harvard.edu/~kohler/class/cs239-w08/decandia07dynamo.pdf) |
| Data store | **MongoDB** - Document-oriented database | [slideshare.net](http://www.slideshare.net/mdirolf/introduction-to-mongodb) |
| Data store | **Spanner** - Globally-distributed database from Google | [research.google.com](http://research.google.com/archive/spanner-osdi2012.pdf) |
| Data store | **Memcached** - Distributed memory caching system | [slideshare.net](http://www.slideshare.net/oemebamo/introduction-to-memcached) |
| Data store | **Redis** - Distributed memory caching system with persistence and value types | [slideshare.net](http://www.slideshare.net/dvirsky/introduction-to-redis) |
| Data store | **Couchbase** - an open-source, distributed multi-model NoSQL document-oriented database | [couchbase.com](https://www.couchbase.com/) |
| | | |
| File system | **Google File System (GFS)** - Distributed file system | [research.google.com](http://static.googleusercontent.com/media/research.google.com/zh-CN/us/archive/gfs-sosp2003.pdf) |
| File system | **Hadoop File System (HDFS)** - Open source implementation of GFS | [apache.org](https://hadoop.apache.org/docs/r1.2.1/hdfs_design.html) |
| | | |
| Misc | **Chubby** - Lock service for loosely-coupled distributed systems from Google | [research.google.com](http://static.googleusercontent.com/external_content/untrusted_dlcp/research.google.com/en/us/archive/chubby-osdi06.pdf) |
| Misc | **Dapper** - Distributed systems tracing infrastructure | [research.google.com](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36356.pdf)
| Misc | **Kafka** - Pub/sub message queue from LinkedIn | [slideshare.net](http://www.slideshare.net/mumrah/kafka-talk-tri-hug) |
| Misc | **Zookeeper** - Centralized infrastructure and services enabling synchronization | [slideshare.net](http://www.slideshare.net/sauravhaloi/introduction-to-apache-zookeeper) |
| Misc | **ØMQ** - a high-performance asynchronous messaging library, aimed at use in distributed or concurrent applications. | [zeromq.org](http://zeromq.org/) |
| Misc | **etcd** - A distributed, reliable key-value store for the most critical data of a distributed system. | [etcd docs](https://coreos.com/etcd/docs/latest/) |

# Company Architectures

| Company | Reference(s) |
|---|---|
| Amazon | [Amazon architecture](http://highscalability.com/amazon-architecture) |
| Cinchcast | [Producing 1,500 hours of audio every day](http://highscalability.com/blog/2012/7/16/cinchcast-architecture-producing-1500-hours-of-audio-every-d.html) |
| DataSift | [Realtime datamining At 120,000 tweets per second](http://highscalability.com/blog/2011/11/29/datasift-architecture-realtime-datamining-at-120000-tweets-p.html) |
| DropBox | [How we've scaled Dropbox](https://www.youtube.com/watch?v=PE4gwstWhmc) |
| ESPN | [Operating At 100,000 duh nuh nuhs per second](http://highscalability.com/blog/2013/11/4/espns-architecture-at-scale-operating-at-100000-duh-nuh-nuhs.html) |
| Google | [Google architecture](http://highscalability.com/google-architecture) |
| Instagram | [14 million users, terabytes of photos](http://highscalability.com/blog/2011/12/6/instagram-architecture-14-million-users-terabytes-of-photos.html)<br/>[What powers Instagram](http://instagram-engineering.tumblr.com/post/13649370142/what-powers-instagram-hundreds-of-instances) |
| Justin.tv | [Justin.Tv's live video broadcasting architecture](http://highscalability.com/blog/2010/3/16/justintvs-live-video-broadcasting-architecture.html) |
| Facebook | [Scaling memcached at Facebook](https://cs.uwaterloo.ca/~brecht/courses/854-Emerging-2014/readings/key-value/fb-memcached-nsdi-2013.pdf)<br/>[TAO: Facebook’s distributed data store for the social graph](https://cs.uwaterloo.ca/~brecht/courses/854-Emerging-2014/readings/data-store/tao-facebook-distributed-datastore-atc-2013.pdf)<br/>[Facebook’s photo storage](https://www.usenix.org/legacy/event/osdi10/tech/full_papers/Beaver.pdf) |
| Flickr | [Flickr architecture](http://highscalability.com/flickr-architecture) |
| Mailbox | [From 0 to one million users in 6 weeks](http://highscalability.com/blog/2013/6/18/scaling-mailbox-from-0-to-one-million-users-in-6-weeks-and-1.html) |
| Pinterest | [From 0 To 10s of billions of page views a month](http://highscalability.com/blog/2013/4/15/scaling-pinterest-from-0-to-10s-of-billions-of-page-views-a.html)<br/>[18 million visitors, 10x growth, 12 employees](http://highscalability.com/blog/2012/5/21/pinterest-architecture-update-18-million-visitors-10x-growth.html) |
| Playfish | [50 million monthly users and growing](http://highscalability.com/blog/2010/9/21/playfishs-social-gaming-architecture-50-million-monthly-user.html) |
| PlentyOfFish | [PlentyOfFish architecture](http://highscalability.com/plentyoffish-architecture) |
| Salesforce | [How they handle 1.3 billion transactions a day](http://highscalability.com/blog/2013/9/23/salesforce-architecture-how-they-handle-13-billion-transacti.html) |
| Stack Overflow | [Stack Overflow architecture](http://highscalability.com/blog/2009/8/5/stack-overflow-architecture.html) |
| TripAdvisor | [40M visitors, 200M dynamic page views, 30TB data](http://highscalability.com/blog/2011/6/27/tripadvisor-architecture-40m-visitors-200m-dynamic-page-view.html) |
| Tumblr | [15 billion page views a month](http://highscalability.com/blog/2012/2/13/tumblr-architecture-15-billion-page-views-a-month-and-harder.html) |
| Twitter | [Making Twitter 10000 percent faster](http://highscalability.com/scaling-twitter-making-twitter-10000-percent-faster)<br/>[Storing 250 million tweets a day using MySQL](http://highscalability.com/blog/2011/12/19/how-twitter-stores-250-million-tweets-a-day-using-mysql.html)<br/>[150M active users, 300K QPS, a 22 MB/S firehose](http://highscalability.com/blog/2013/7/8/the-architecture-twitter-uses-to-deal-with-150m-active-users.html)<br/>[Timelines at scale](https://www.infoq.com/presentations/Twitter-Timeline-Scalability)<br/>[Big and small data at Twitter](https://www.youtube.com/watch?v=5cKTP36HVgI)<br/>[Operations at Twitter: scaling beyond 100 million users](https://www.youtube.com/watch?v=z8LU0Cj6BOU) |
| Uber | [How Uber scales their real-time market platform](http://highscalability.com/blog/2015/9/14/how-uber-scales-their-real-time-market-platform.html) |
| WhatsApp | [The WhatsApp architecture Facebook bought for $19 billion](http://highscalability.com/blog/2014/2/26/the-whatsapp-architecture-facebook-bought-for-19-billion.html) |
| YouTube | [YouTube scalability](https://www.youtube.com/watch?v=w5WVu624fY8)<br/>[YouTube architecture]

# company engineering blog

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

# System Design Pattern

## aws cloud design pattern

## azure cloud design pattern

## google cloud design pattern

# Cracking The Coding Interview Quiz

* Stock Data
* Social Network
* Web Crawler
* Duplicate URLs
* Cache
* Sales Rank
* Personal Financial Manager
* Pastebin