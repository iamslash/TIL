- [Materials](#materials)
- [Install](#install)
  - [Install on Ubuntu](#install-on-ubuntu)
  - [Install with docker](#install-with-docker)
    - [Single Zookeeper / Single Kafka](#single-zookeeper--single-kafka)
    - [Single Zookeeper / Multiple Kafka](#single-zookeeper--multiple-kafka)
    - [Multiple Zookeeper / Single Kafka](#multiple-zookeeper--single-kafka)
    - [Multiple Zookeeper / Multiple Kafka](#multiple-zookeeper--multiple-kafka)
    - [Full stack](#full-stack)
- [Feature](#feature)
  - [Overview](#overview)
  - [Zero Copy](#zero-copy)
  - [Zookeeper](#zookeeper)
  - [Topic](#topic)
  - [Partition](#partition)
  - [Rebalance](#rebalance)
  - [Consumer Group](#consumer-group)
  - [ACK](#ack)
  - [Exactly once](#exactly-once)
- [Basic](#basic)
  - [Useful Commands](#useful-commands)
- [Advanced](#advanced)
  - [Gurantee order of messages, no duplicates](#gurantee-order-of-messages-no-duplicates)

-----

# Materials

* [kafka 조금 아는 척하기 1 (개발자용) @ youtube](https://www.youtube.com/watch?v=0Ssx7jJJADI)
  * [kafka 조금 아는 척하기 2 (개발자용) @ youtube](https://www.youtube.com/watch?v=geMtm17ofPY)
  * [kafka 조금 아는 척하기 3 (개발자용) @ youtube](https://www.youtube.com/watch?v=xqrIDHbGjOY)
* [아파치 카프카 입문 @ Tacademy](https://tacademy.skplanet.com/live/player/onlineLectureDetail.action?seq=183)
  * [src](https://github.com/AndersonChoi/tacademy-kafka)
  * [토크ON 77차. 아파치 카프카 입문 1강 - Kafka 기본개념 및 생태계 | T아카데미 @ youtube](https://www.youtube.com/watch?v=VJKZvOASvUA)
  * [유튜브 ™ 를위한 애드 블록 에 의해 청소 Share 토크ON 77차. 아파치 카프카 입문 2강 - Kafka 설치, 실행, CLI | T아카데미 @ youtube](https://www.youtube.com/watch?v=iUX6d14bvj0)
  * [토크ON 77차. 아파치 카프카 입문 3강 - Kafka Producer application @ youtube](https://www.youtube.com/watch?v=dubFjEXuK6w)
  * [토크ON 77차. 아파치 카프카 입문 4강 - Kafka Consumer application @ youtube](https://www.youtube.com/watch?v=oyNjiQ2q2CE)
  * [토크ON 77차. 아파치 카프카 입문 5강 - Kafka 활용 실습 @ youtube](https://www.youtube.com/watch?v=3OPZ7_sHtWo)
* [How To Install Apache Kafka on Ubuntu 18.04](https://www.digitalocean.com/community/tutorials/how-to-install-apache-kafka-on-ubuntu-18-04)
* [kafka @ joinc](https://www.joinc.co.kr/w/man/12/Kafka)
* [Core Concepts](https://kafka.apache.org/0110/documentation/streams/core-concepts)
* [Kafka 이해하기](https://medium.com/@umanking/%EC%B9%B4%ED%94%84%EC%B9%B4%EC%97%90-%EB%8C%80%ED%95%B4%EC%84%9C-%EC%9D%B4%EC%95%BC%EA%B8%B0-%ED%95%98%EA%B8%B0%EC%A0%84%EC%97%90-%EB%A8%BC%EC%A0%80-data%EC%97%90-%EB%8C%80%ED%95%B4%EC%84%9C-%EC%9D%B4%EC%95%BC%EA%B8%B0%ED%95%B4%EB%B3%B4%EC%9E%90-d2e3ca2f3c2)
* [Kafka 운영자가 말하는 처음 접하는 Kafka](https://www.popit.kr/kafka-%EC%9A%B4%EC%98%81%EC%9E%90%EA%B0%80-%EB%A7%90%ED%95%98%EB%8A%94-%EC%B2%98%EC%9D%8C-%EC%A0%91%ED%95%98%EB%8A%94-kafka/)
  * [Kafka 운영자가 말하는 Topic Replication](https://www.popit.kr/kafka-%EC%9A%B4%EC%98%81%EC%9E%90%EA%B0%80-%EB%A7%90%ED%95%98%EB%8A%94-topic-replication/) 
  * [Kafka 운영자가 말하는 TIP](https://www.popit.kr/kafka-%EC%9A%B4%EC%98%81%EC%9E%90%EA%B0%80-%EB%A7%90%ED%95%98%EB%8A%94-tip/) 
  * [Kafka 운영자가 말하는 Producer ACKS](https://www.popit.kr/kafka-%EC%9A%B4%EC%98%81%EC%9E%90%EA%B0%80-%EB%A7%90%ED%95%98%EB%8A%94-producer-acks/)

# Install

## Install on Ubuntu

* [How To Install Apache Kafka on Ubuntu 18.04](https://www.digitalocean.com/community/tutorials/how-to-install-apache-kafka-on-ubuntu-18-04)

## Install with docker

* [kafka-stack-docker-compose](https://github.com/simplesteph/kafka-stack-docker-compose)
  * zookeeper, kafka cluter

```console
$ git clone git@github.com:simplesteph/kafka-stack-docker-compose.git
```

### Single Zookeeper / Single Kafka

```bash
# If you didn't remove this directory It might fail to start
$ rm -rf zk-single-kafka-single
$ docker-compose -f zk-single-kafka-single.yml up
$ docker-compose -f zk-single-kafka-single.yml down
```

### Single Zookeeper / Multiple Kafka

```console
$ docker-compose -f zk-single-kafka-multiple.yml up
$ docker-compose -f zk-single-kafka-multiple.yml down
```

### Multiple Zookeeper / Single Kafka

```console
$ docker-compose -f zk-multiple-kafka-single.yml up
$ docker-compose -f zk-multiple-kafka-single.yml down
```

### Multiple Zookeeper / Multiple Kafka

```console
$ docker-compose -f zk-multiple-kafka-multiple.yml up
$ docker-compose -f zk-multiple-kafka-multiple.yml down
```

### Full stack

* Single Zookeeper: $DOCKER_HOST_IP:2181
* Single Kafka: $DOCKER_HOST_IP:9092
* Kafka Schema Registry: $DOCKER_HOST_IP:8081
* Kafka Schema Registry UI: $DOCKER_HOST_IP:8001
* Kafka Rest Proxy: $DOCKER_HOST_IP:8082
* Kafka Topics UI: $DOCKER_HOST_IP:8000
* Kafka Connect: $DOCKER_HOST_IP:8083
* Kafka Connect UI: $DOCKER_HOST_IP:8003
* KSQL Server: $DOCKER_HOST_IP:8088
* Zoonavigator Web: $DOCKER_HOST_IP:8004

```console
$ docker-compose -f full-stack.yml up
$ docker-compose -f full-stack.yml down
```

# Feature

## Overview

Queue 와 Pub/Sub 을 지원하는 Message Queue 이다. scala 로 만들어 졌다. kafka 는 disk 에서 데이터를 caching 한다.
따라서 저렴한 비용으로 대량의 데이터를 보관할 수 있다. 실제로 disk 에 random access 는 100 KB/sec 이지만
linear writing 은 600 MB/sec 이다. 6000 배이다. 따라서 random access 보다 linear writing 을 많이 한다면 disk 를 이용해도 좋다.

![](http://deliveryimages.acm.org/10.1145/1570000/1563874/jacobs3.jpg)


## Zero Copy

![](img/zerocopy_1.gif)

데이터를 읽어서 네트워크로 전송할 때 kernel mode -> user mode -> kernel mode 순서로 OS 의 mode 변환이 필요하다.

![](img/zerocopy_2.gif)

이때 user mode 변환 없이 데이터를 네트워크로 전송하는 것을 zero copy 라고 한다.

## Zookeeper

zookeeper 는 kafka node 를 관리하고 topic 의 offset 을 저장한다.

## Topic

topic 은 RDBMS 의 Table 과 같다. durability 를 위해 replication 개수를 정할 수 있고 partition 을 통해서 totpic 을 나눌 수 있다. `consumer_offsets` totpic 은 자동으로 생성되는 topic 이다.

```bash
## Create the topic
$ /usr/bin/kafka-topics --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic my-topic

## Show topic list
$ /usr/bin/kafka-topics --list --zookeeper zoo1:2181
__confluent.support.metrics
__consumer_offsets
my-topic
```

다음과 같이 메시지를 전송할 수 있다.

```console
$ /usr/bin/kafka-console-producer --broker-list localhost:9092 --topic my-topic
> Hello
> World
```

다음과 같이 메시지를 가져올 수 있다.

```console
$ /usr/bin/kafka-console-consumer --bootstrap-server localhost:9092 --from-beginning --topic my-topic
```

## Partition

하나의 Topic 은 여러개의 Partition 으로 구성한다. Partition 은 message 를
저장하는 file 과 같다. 이것을 append only file 이라고 한다. Partition 의 message
는 일정시간이 지나면 지워진다. 일정한 기간동안 보관된다.

Producer 는 여러개의 Partition 에 병렬로 메시지를 전송할 수 있다. Consumer
입장에서 Message 순서가 보장될 수 없다. Partition 의 개수는 한번 늘리면 줄일 수
없다. partition 과 consumer group 을 사용하면 topic 을 parallel 하게 처리하여
수행성능을 높일 수 있다.  

Message 의 순서가 중요하다면 하나의 Topic 은 하나의 Partition 으로 구성한다.

Message 순서에 대해 Deep Dive 해보자. 다음과 같이 8 개의 partition 에 my-topic-8
을 만들어 보자.

```console
$ /usr/bin/kafka-topics --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 8 --topic my-topic-8
```

이제 Message 를 보내자. 

```console
$ /usr/bin/kafka-console-producer --broker-list localhost:9092 --topic my-topic-8
>1
>2
>3
>4
>5
>6
>7
>8
```

이제 가져와보자. 순서가 보장되지 않는다. 

```console
$ /usr/bin/kafka-console-consumer --bootstrap-server localhost:9092 --from-beginning --topic my-topic-8
4
5
6
7
1
2
3
8
9
0
```

## Rebalance

Kafka 가 partition 을 Consumer Instance 에 다시 할당하는 과정이다. 예를 들어 Consumer Group 의 특정 Consumer Instance 가 장애가 발생했다면 Kafka 는 잠깐 시간을 내어 살아있는 Consumer Instance 들에게 Partition 을 다시 할당한다.

## Consumer Group

* [Kafka 운영자가 말하는 Kafka Consumer Group](https://www.popit.kr/kafka-consumer-group/)

하나의 Consumer Group 은 여러개의 Consumer Instance 들로 구성된다. 하나의
Consumer Instance 는 하나의 Partition 하고만 연결할 수 있다. 즉, 하나의 Consumer
Instance 는 특정한 Topic 의 특정 partition 에서 message 를 가져온다.  

하나의 partition 을 두개의 Consumer Instance 가 consuming 할 수는 없다. 하나의 Consumer Instance 가 두개의 partition 을 consuming 할 수는 있다.

Consumer Group 은 Consumer Instance 의 High Availability 를 위해 필요하다. 예를 들어 하나의 Consumer Group `Hello` 는 4 개의 Consumer Instance 로 구성되어 있다. `Hello` 는 `world-topic` 에서 message 를 가져온다. Consumer Instance 하나가 장애가 발생해도 서비스의 지장은 없다.

## ACK

Kafka 는 하나의 leader 와 여러개의 follower 들로 구성된다. leader 가 Producer 로 부터 Message 를 넘겨 받으면 follower 에게 전송한다. 

Producer config 의 ack 설정은 다음과 같다.

* `ack=0`: producer 는 message 의 ack 를 필요로 하지 않는다.
* `ack=1`: producer 는 leader 에게 Message 가 전송되었음을 보장한다. 
* `ack=all(-1)`: producer 는 leader 및 follower 에게 모두 Message 가 전송되었음을 보장한다. 

## Exactly once

* configuration file 에서 `processing.guarantee=exactly_once` 로 설정하면 된다.
  * [PROCESSING GUARANTEES @ manual](https://kafka.apache.org/0110/documentation/streams/core-concepts)
  * In order to achieve exactly-once semantics when running Kafka Streams applications, users can simply set the processing.guarantee config value to exactly_once (default value is at_least_once). More details can be found in the Kafka Streams Configs section.

# Basic

## Useful Commands

* [Apache Kafka CLI commands cheat sheet](https://medium.com/@TimvanBaarsen/apache-kafka-cli-commands-cheat-sheet-a6f06eac01b#09e8)

----

```bash
## Start zookeeper, kafka server
## But You don't need this when you use docker-compose
$ /usr/bin/zookeeper-server-start /etc/kafka/config/zookeeper.properties
$ /usr/bin/kafka-server-start /etc/kafka/config/server.properties

## Connect kafka docker container
$ docker exec -it kafka-stack-docker-compose_kafka1_1 bash

## Show topic list
$ /usr/bin/kafka-topics --list --zookeeper zoo1:2181

## Create the topic
$ /usr/bin/kafka-topics --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic my-topic

## Pull message from topic
$ /usr/bin/kafka-console-consumer --bootstrap-server localhost:9092 --from-beginning --topic my-topic

## Send message to topic
$ /usr/bin/kafka-console-producer --broker-list localhost:9092 --topic my-topic
> Hello
> World

## Pull message from topic with partition 1
$ /usr/bin/kafka-console-consumer --bootstrap-server localhost:9092 --from-beginning --partition 1 --topic my-topic

## Delete the topic
$ /usr/bin/kafka-topics --zookeeper zoo1:2181 --delete --topic my-topic

## Stop kafka, zookeeper
## But You don't need this when you use docker-compose
$ /usr/bin/zookeeper-server-stop /etc/kafka/config/zookeeper.properties
$ /usr/bin/kafka-server-stop /etc/kafka/config/server.properties

## Show consumer groups
$ /usr/bin/kafka-consumer-groups --bootstrap-server localhost:9092 --list

## Describe consumer group
$ /usr/bin/kafka-consumer-groups --bootstrap-server localhost:9092 --group <group-name> --describe

## Delete consumer group
$ /usr/bin/kafka-consumer-groups --zookeeper zoo1:2181 --delete --group <group-name>

## Check topic leader follower
$ /usr/bin/kafka-topics --zookeeper zoo1:2181 --topic my-topic --describe

## server log check ???
$ cat /usr/local/bin/kafka/logs/server.log 
```

# Advanced

## Gurantee order of messages, no duplicates

* 하나의 Topic 에 하나의 Partition 을 구성한다.
* Kafka 가 죽었다가 살아날 때 중복 소비를 방지하기 위해 `processing.guarantee=exactly_once` 를 설정한다.
