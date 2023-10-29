- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
  - [Install](#install)
  - [Hello World](#hello-world)
  - [Water Mark](#water-mark)

----

# Abstract

Apache Flink is a framework and distributed processing engine for stateful computations over unbounded and bounded data streams

# Materials

* [스트림 프로세싱의 긴 여정을 위한 이정표 (w. Apache Flink) | medium](https://medium.com/rate-labs/%EC%8A%A4%ED%8A%B8%EB%A6%BC-%ED%94%84%EB%A1%9C%EC%84%B8%EC%8B%B1%EC%9D%98-%EA%B8%B4-%EC%97%AC%EC%A0%95%EC%9D%84-%EC%9C%84%ED%95%9C-%EC%9D%B4%EC%A0%95%ED%91%9C-with-flink-8e3953f97986)
  * Stream framework 의 jardon 을 포함하여 설명
* [Apache Flink Training Excercises | github](https://github.com/apache/flink-training)
* [Stream Processing with Apache Flink: Fundamentals, Implementation, and Operation of Streaming Applications | amazon](https://www.amazon.com/Stream-Processing-Apache-Flink-Implementation-ebook/dp/B07QM3DSB7)
  * [src-java](https://github.com/streaming-with-flink/examples-java)
  * [src-scala](https://github.com/streaming-with-flink/examples-scala)
  * [src-kotlin](https://github.com/rockmkd/flink-examples-kotlin)
* [Demystifying Flink Memory Allocation and tuning - Roshan Naik | youtube](https://www.youtube.com/watch?v=aq1Whga-RJ4)
  * [Flink Memory Tuning Calculator | googledoc](https://docs.google.com/spreadsheets/d/1DMUnHXNdoK1BR9TpTTpqeZvbNqvXGO7PlNmTojtaStU/edit#gid=0)

# Basic

## Install

Download and decompress files

* [flink downloads](https://flink.apache.org/downloads.html)

## Hello World

* [First steps | flink](https://nightlies.apache.org/flink/flink-docs-release-1.16/docs/try-flink/local_installation/)

JDK 11 이어야 한다. JDK 17 안된다.

```bash
$ sdk list java

# Use jdk 1.11
$ sdk use java 11.0.17-amzn

# Download and decompress flink-1.16.0
$ cd flink-1.16.0

# Start the cluster
$ bin/start-cluster.sh

# Submit job
$ bin/flink run examples/streaming/WordCount.jar

# Browse the dash board, http://localhost:8081

# Check logs
$ tail -f log/flink-*-taskexecutor-*.out

# Stop the cluster
$ bin/stop-cluster.sh
```

## Water Mark

* [이벤트 시간 처리(Event Time Processing)와 워터마크(Watermark) | tistory](https://seamless.tistory.com/99)

Watermark 는 Event Time Processing 에서 지연된 message 처리를 위한 것이다.
