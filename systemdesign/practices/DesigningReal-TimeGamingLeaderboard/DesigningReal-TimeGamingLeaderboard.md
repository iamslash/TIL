- [Requirements](#requirements)
  - [Functional Requirement](#functional-requirement)
  - [Non-Functional Requirement](#non-functional-requirement)
  - [Estimation](#estimation)
- [High Level Design](#high-level-design)
  - [API Design](#api-design)
  - [High-Level Design](#high-level-design-1)
  - [Data Model](#data-model)
- [Low Level Design](#low-level-design)
  - [Build on the cloud or not](#build-on-the-cloud-or-not)
  - [Scale Redis](#scale-redis)
  - [Alternative Solution: NoSQL](#alternative-solution-nosql)
- [System Extension](#system-extension)
  - [Faster Retrieval and Breaking Tie](#faster-retrieval-and-breaking-tie)
  - [System Failure Recovery](#system-failure-recovery)
- [Q&A](#qa)
  - [How To Deal with 500M users](#how-to-deal-with-500m-users)
- [References](#references)

----

# Requirements

## Functional Requirement

## Non-Functional Requirement

## Estimation

# High Level Design

## API Design

## High-Level Design

## Data Model

# Low Level Design

## Build on the cloud or not

## Scale Redis

## Alternative Solution: NoSQL

# System Extension

## Faster Retrieval and Breaking Tie

## System Failure Recovery

# Q&A

## How To Deal with 500M users

`5 Billian users` 를 위해 `5 Billian * 10 Byte = 50 GB` 가 필요하다. 
Redis 의 sorted set 은 sharding 이 필요하다. sorted set 을 sharding 
하는 것은 system 의 complexity 를 높여준다. 

rank 에 관심있는 user 들은 상위 user 들이다. 상위 user 들만 sorted 
set 에 저장한다. 하위 user 들은 어림잡아 계산한다.

# References
