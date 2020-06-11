# Materials

* [How To Install Apache Kafka on Ubuntu 18.04](https://www.digitalocean.com/community/tutorials/how-to-install-apache-kafka-on-ubuntu-18-04)
* [kafka @ joinc](https://www.joinc.co.kr/w/man/12/Kafka)
* [Core Concepts](https://kafka.apache.org/0110/documentation/streams/core-concepts)

# Features

Queue 와 Pub/Sub 을 지원하는 Message Queue 이다. kafka 는 disk 에서 데이터를 caching 한다.
따라서 저렴한 비용으로 대량의 데이터를 보관할 수 있다. 실제로 disk 에 random access 는 100 KB/sec 이지만
 linear writing 은 600 MB/sec 이다. 6000 배이다.

![](http://deliveryimages.acm.org/10.1145/1570000/1563874/jacobs3.jpg)

## Zero Copy

![](img/zerocopy_1.gif)

데이터를 읽어서 네트워크로 전송할 때 kernel mode -> user mode -> kernel mode 순서로 OS 의 mode 변환이 필요하다.

![](img/zerocopy_2.gif)

이때 user mode 변환 없이 데이터를 네트워크로 전송하는 것을 zero copy 라고 한다.

# Install with docker

* [kafka-elk-docker-compose @ github](https://github.com/sermilrod/kafka-elk-docker-compose)
  * filebeat, kafka, elk 를 docker compose 로 실행해 보자.

# Exactly once

* configuration file 에서 `processing.guarantee=exactly_once` 로 설정하면 된다.
  * [PROCESSING GUARANTEES @ manual](https://kafka.apache.org/0110/documentation/streams/core-concepts)
  * In order to achieve exactly-once semantics when running Kafka Streams applications, users can simply set the processing.guarantee config value to exactly_once (default value is at_least_once). More details can be found in the Kafka Streams Configs section.
