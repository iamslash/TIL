- [Requirements](#requirements)
- [Some Design Considerations](#some-design-considerations)
- [Capacity Estimation and Constraints](#capacity-estimation-and-constraints)
- [System APIs](#system-apis)
- [Database Design](#database-design)
- [High Level Design](#high-level-design)
- [Component Design](#component-design)
  - [Application Layer](#application-layer)
  - [Datastore layer](#datastore-layer)
- [Purging or DB Cleanup](#purging-or-db-cleanup)
- [Data Partitioning and Replication](#data-partitioning-and-replication)
- [Cache and Load Balancer](#cache-and-load-balancer)
- [Security and Permissions](#security-and-permissions)

----

# Requirements

* Functional Requirements
  * 유저는 데이터를 업로드하고 url 을 얻을 수 있다.
  * 유저는 텍스트 데이터만 업로드할 수 있다.
  * 데이터와 url 은 일정시간이 지나면 소멸한다. 유저는 그 유지시간을 정할 수 있다.
  * 유저는 업로드한 데이터의 custom alias 를 만들 수 있다.
* Non-Functional Requirements
  * 시스템의 신뢰도는 높고 업로드된 데이터는 유실 될 수 없다.
  * 시스템의 가용성은 높다.
  * latency 는 매우 낮다.
  * 데이터의 url 은 예측하기 힘들다.
* Extended Requirements
  * Analytics
  * provides REST APIs.

# Some Design Considerations

* [Designing a URL Shortening service like TinyURL](Designing_a_URL_Shortening_service_like_TinyURL.md) 과 비슷하다.
* What should be the limit on the amount of text user can paste at a time?
  * 10 MB 이상은 업로드 불가
* Should we impose size limits on custom URLs?
  * URL database  를 위해 필요하다.

# Capacity Estimation and Constraints

* very read-heavy. 5:1 ratio between read and write.
* Traffic estimates
  * write `1 M pastes / day` read `5 M pastes day`
  * write pastes `1 M / (24 hrs * 3600 sec) = 12 writes / sec`
  * read pastes `5 M / (24 hrs * 3600 sec) = 58 reads / sec`
* Storage estimates
  * upload `10 MB` data. `10 KB texts / pastes`
  * `1 M * 10 KB = 10 GB / day`
  * `10 GB / day * 365 days * 10 years = 36 TB / 10 years`
  * base64 encoding ([A-Za-z0-9.-]) url
  * `64 ^ 6 = 68.7 B unique strings`
  * `3.6 B * 6 = 22 GB`
* Bandwidth estimates
* Memory estimates

# System APIs

```c
addPaste(api_dev_key, 
  paste_data, 
  custom_url=None,
  user_name=None, 
  paste_name=None, 
  expire_date=None)

parameters:
returns: (string)  

getPaste(api_dev_key, api_paste_key)

deletePaste(api_dev_key, api_paste_key)
```

# Database Design

# High Level Design

# Component Design

## Application Layer

* How to handle a write request?
* Isn’t KGS a single point of failure? 
* Can each app server cache some keys from key-DB? 
* How does it handle a paste read request? 

## Datastore layer

* Metadata database: MySQL or Dynamo or Cassandra
* Object storage: amazon s3

# Purging or DB Cleanup

* [Designing a URL Shortening service like TinyURL](Designing_a_URL_Shortening_service_like_TinyURL.md)

# Data Partitioning and Replication

* [Designing a URL Shortening service like TinyURL](Designing_a_URL_Shortening_service_like_TinyURL.md)

# Cache and Load Balancer

* [Designing a URL Shortening service like TinyURL](Designing_a_URL_Shortening_service_like_TinyURL.md)

# Security and Permissions

* [Designing a URL Shortening service like TinyURL](Designing_a_URL_Shortening_service_like_TinyURL.md)
