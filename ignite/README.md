# Abstract

Apache Ignite 에 대해 정리한다.

# Materials

* [Learn Apache Ignite Through Coding Examples @ youtube](https://www.youtube.com/watch?v=uRdSKhFqeaU)
  * [src](https://github.com/GridGain-Demos/ignite-learning-by-examples)

# Basic

## Run

다음과 같은 방법으로 Ignite, GridGain Control Center 를 실행한다.

* [Learning Apache Ignite Through Examples | github](https://github.com/GridGain-Demos/ignite-learning-by-examples) 를 clone 한다.
* Download [Apache Ignite 2.8.1 or later](https://ignite.apache.org/download.cgi)
  * Binary Releases 를 다운받는다.
* Download [GridGain Control Center agent](https://www.gridgain.com/tryfree#controlcenteragent) and put it into the Ignite libs folder.
  * GridGain Control Center agent 의 압축을 풀고 `bin, libs` 디렉토리를 `apache-ignite-2.12.0-bin/` 에 복사한다.

```bash
# Start a 2-nodes cluster 
# using `{root_of_this_project}/complete/cfg/ignite-config.xml`.
#
# Run node 1 on terminal 1
$ ./ignite.sh ~/my/java/ignite-learning-by-examples/complete/cfg/ignite-config.xml
# Run node 2 on terminal 2
$ ./ignite.sh ~/my/java/ignite-learning-by-examples/complete/cfg/ignite-config.xml
# stdout 의 log 를 살펴보자. Control Center 의 link 를 클릭하여 
# Control Center 에 접속한다. Cluter 를 Activate 한다.

# Run sql client
$ ./sqlline.sh --verbose=true -u jdbc:ignite:thin://127.0.0.1/

# Restore sql script
sqlline> !run ~/my/java/ignite-learning-by-examples/complete/scripts/ignite_world.sql
sqlline> SELECT * FROM COUNTRY LIMIT 100;
```

## Use Cases

다음과 같은 용도로 Ignite 를 사용한다.

* In-Memory Cache
* In-Memory Data Grid
* In-Memory Database
* Key-Value Store
* High-Performance Computing
* Digital Integration Hub
* Spark Acceleration
* Hadoop Acceleration

## Data Partitioning

* [Data Partitioning](https://ignite.apache.org/docs/latest/data-modeling/data-partitioning)
* [Data Distribution in Apache Ignite](https://www.gridgain.com/resources/blog/data-distribution-in-apache-ignite)
