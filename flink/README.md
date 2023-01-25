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

* [Apache Flink Training Excercises | github](https://github.com/apache/flink-training)
* [Stream Processing with Apache Flink: Fundamentals, Implementation, and Operation of Streaming Applications | amazon](https://www.amazon.com/Stream-Processing-Apache-Flink-Implementation-ebook/dp/B07QM3DSB7)
  * [src-java](https://github.com/streaming-with-flink/examples-java)
  * [src-scala](https://github.com/streaming-with-flink/examples-scala)
  * [src-kotlin](https://github.com/rockmkd/flink-examples-kotlin)

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
