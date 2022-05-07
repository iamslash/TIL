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
- [Extension](#extension)
  - [Faster Retrieval and Breaking Tie](#faster-retrieval-and-breaking-tie)
  - [System Failure Recovery](#system-failure-recovery)
- [Q&A](#qa)
  - [How To Deal with 5B users](#how-to-deal-with-5b-users)
- [References](#references)

----

# Requirements

## Functional Requirement

## Non-Functional Requirement

# Estimation

# High Level Design

## API Design

## High-Level Design

## Data Model

# Low Level Design

## Build on the cloud or not

## Scale Redis

## Alternative Solution: NoSQL

# Extension

## Faster Retrieval and Breaking Tie

## System Failure Recovery

# Q&A

## How To Deal with 5B users

매우 많은 수의 user 가 있다면 어떻게 해결할 수 있을까?

`5 Billian users` 를 위해 `5 Billian * 10 Byte = 50 GB` 가 필요하다. Redis 의
sorted set 은 sharding 이 필요하다. 다음과 같은 2 가지 방법을 생각할 수 있다.

첫째, shard key 를 user 의 순위로 하자. 즉, 순위를 구간별로 나누고 한 구간을
하나의 shard 에 배정한다. top 10 users 를 쉽게 가져올 수 있다. 그러나 특정 user
의 순위를 쉽게가져 올 수는 없다. user 의 순위 구간이 변경되면 전체 user 의
순위가 조정되야 한다. system 의 complexity 가 높다.

두번째, shard key 를 user 의 id 로 하자. 즉, 각 shard 마다 독자적인 순위를
갖는다. client 가 top 10 users 의 정보를 얻으려면 각 shard 의 top 10 users 를
모두 끌어모아서 정렬하고 다시 top 10 users 를 추출한다. 특정 user 의 순위 역시
알아내기가 쉽지 않다. system 의 complexity 가 높다. client 는 필요한 정보보다
shard count 배 만큼 정보를 조회해야 한다. 불필요하다. 

sorted set 을 sharding 하는 것은 system 의 complexity 를 높인다.

다음과 같은 해결책을 생각해 볼 수 있다.

rank 에 관심있는 user 들은 상위권 user 들이다. 상위권 user 들만 Single Instance
Redis 의 sorted set 에 저장한다. top 10 users 는 쉽게 가져올 수 있다.

하위권 user 들은 어림잡아 계산한다. 정확하지 않아도 된다. 예를 들어 점수를
구간별로 나누고 구간에 얼만큼 user 들이 존재하는지 하루에 한번 계산해 Sing
Instance Redis 에 저장해 둔다. 특정 user 의 점수를 보고 상위 구간에 얼만큼 user
들이 있는지 알 수 있다.

# References
