- [Requirements](#requirements)
- [High Level Design](#high-level-design)
- [Low Level Design](#low-level-design)
  - [Major subjects](#major-subjects)
  - [CAP theorem](#cap-theorem)
  - [Data Partition](#data-partition)
  - [Data Replication](#data-replication)
  - [Consistency](#consistency)
  - [Inconsistency Resolution](#inconsistency-resolution)
  - [Handling Failures](#handling-failures)
  - [System Arhictecture Diagram](#system-arhictecture-diagram)
  - [Write Path](#write-path)
  - [Read Path](#read-path)
- [Extentions](#extentions)
- [Q\&A](#qa)
- [Implementation](#implementation)
- [References](#references)

-----

# Requirements

* The system supports `put(key, val), get(key)`
* The size of key-value pair is less than 10KB.
* The system should handle big data.
* The system should be highly reliable, available, scalable.
* The system should support tunable consistency.
* The system should has low latency.

# High Level Design

# Low Level Design

## Major subjects

다음과 같은 주제들을 생각해 보자.

* CAP theorem
  * CAP 가 무엇인가? 우리는 어떤 것을 반영할 것인가?
* Data partition
  * Availability 를 위해 data 를 분리해서 여러 server 에 배치한다.
* Data replication
  * Reliability 를 위해 data 를 복제한다.
* Consistency
* Inconsistency resolution
* Handling failures
* System architecture diagram
* Write path
* Read path

## CAP theorem

* [CAP](/systemdesign/README.md#cap-consistency-availability-partition-tolerance)

We need to design AP (Availability and Partition Tolerance) system.

## Data Partition

[consistent hashing](/consistenthasing/README.md) 으로 해결한다???

## Data Replication

## Consistency

다음과 같이 `N, W, R` 을 정의하자.

```
N: The number of replicas
W: A write quorum of size W
R: A read quorum of size R
```

## Inconsistency Resolution

Vector Clock 을 이용한다???

## Handling Failures

gossip protocol 로 problematic server 를 marking 한다???

일시적인 장애는 sloppy quorum, hinted handoff 으로 해결한다???

영구적인 장애는 merckle tree 로 해결한다???

data center 장애는 system 복제로 해결한다.

## System Arhictecture Diagram

## Write Path

## Read Path

# Extentions

# Q&A

# Implementation

# References

- [System Design : Distributed Database System Key Value Store | youtube](https://www.youtube.com/watch?v=rnZmdmlR-2M)
