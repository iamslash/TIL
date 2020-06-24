# Materials

* [How To Install Apache Kafka on Ubuntu 18.04](https://www.digitalocean.com/community/tutorials/how-to-install-apache-kafka-on-ubuntu-18-04)
* [kafka @ joinc](https://www.joinc.co.kr/w/man/12/Kafka)
* [Core Concepts](https://kafka.apache.org/0110/documentation/streams/core-concepts)
* [Kafka 이해하기](https://medium.com/@umanking/%EC%B9%B4%ED%94%84%EC%B9%B4%EC%97%90-%EB%8C%80%ED%95%B4%EC%84%9C-%EC%9D%B4%EC%95%BC%EA%B8%B0-%ED%95%98%EA%B8%B0%EC%A0%84%EC%97%90-%EB%A8%BC%EC%A0%80-data%EC%97%90-%EB%8C%80%ED%95%B4%EC%84%9C-%EC%9D%B4%EC%95%BC%EA%B8%B0%ED%95%B4%EB%B3%B4%EC%9E%90-d2e3ca2f3c2)

# Install

## Install on Ubuntu

* [](https://www.digitalocean.com/community/tutorials/how-to-install-apache-kafka-on-ubuntu-18-04)

## Install with docker

* [kafka-stack-docker-compose](https://github.com/simplesteph/kafka-stack-docker-compose)
  * zookeeper, kafaka cluter

```console
```

# Feature

Queue 와 Pub/Sub 을 지원하는 Message Queue 이다. kafka 는 disk 에서 데이터를 caching 한다.
따라서 저렴한 비용으로 대량의 데이터를 보관할 수 있다. 실제로 disk 에 random access 는 100 KB/sec 이지만
linear writing 은 600 MB/sec 이다. 6000 배이다. 따라서 random access 보다 linear writing 을 많이 한다면 disk 를 이용해도 좋다.

![](http://deliveryimages.acm.org/10.1145/1570000/1563874/jacobs3.jpg)

# Basic

## Zero Copy

![](img/zerocopy_1.gif)

데이터를 읽어서 네트워크로 전송할 때 kernel mode -> user mode -> kernel mode 순서로 OS 의 mode 변환이 필요하다.

![](img/zerocopy_2.gif)

이때 user mode 변환 없이 데이터를 네트워크로 전송하는 것을 zero copy 라고 한다.

## Partition

하나의 Topic 은 여러개의 Partition 으로 구성한다. Producer 는 여러개의 Partition 에 병렬로 메시지를 전송할 수 있다. Consumer 입장에서 Message 순서가 보장될 수 없다.

Message 의 순서가 중요하다면 하나의 Topic 은 하나의 Partition 으로 구성한다.

## ACK

Kafka 는 하나의 leader 와 여러개의 follower 들로 구성된다. leader 가 Producer 로 부터 Message 를 넘겨 받으면 follower 에게 전송한다. 

Producer config 에서 `ack=1` 을 설정하면 leader 및 follower 에게 모두 Message 가 전송되었음을 보장한다. Producer 입장에서 느리다. 보통은 ack 를 default 로 설정해서 leader 에게만 Message 가 전송되었음을 보장한다.

## Consumer Group

???

## Exactly once

* configuration file 에서 `processing.guarantee=exactly_once` 로 설정하면 된다.
  * [PROCESSING GUARANTEES @ manual](https://kafka.apache.org/0110/documentation/streams/core-concepts)
  * In order to achieve exactly-once semantics when running Kafka Streams applications, users can simply set the processing.guarantee config value to exactly_once (default value is at_least_once). More details can be found in the Kafka Streams Configs section.

# Advanced

## Gurant order of messages, no duplicates

* 하나의 Topic 에 하나의 Partition 을 구성한다.
* Kafka 가 죽었다가 살아날 때 중복 소비를 방지하기 위해 `processing.guarantee=exactly_once` 를 설정한다.
