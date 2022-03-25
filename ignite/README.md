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

## Data Modeling

> [Data Modeling](https://ignite.apache.org/docs/latest/data-modeling/data-modeling)

Physical level 에서 Data Entry (either **cache entry** or **table row**) 는
binary object 형태로 저장된다. 여러 Data Entry 가 모여있는 Data Set 은 다시 여러
Partition 으로 나눠진다. 각 Partition 은 여러 Node 들로 분산되어 배치된다. 또한
각 Partition 은 Replication Factor 만큼 복제된다. [Kafka](/kafka/README.md) 와 똑같다.

Logical level 에서 Data Set 은 **Key-Value Cache** 혹은 **SQL Tables** 로 표현된다.
표현만 다를 뿐이지 Physical level 에서 같다. 아래와 같이 Country Table 은 key 가 CODE 이고
나머지 값들이 value 인 **Key-Value Cache** 와 같다.

![](img/cache_table.png)

## Data Partitioning

* [Data Partitioning](https://ignite.apache.org/docs/latest/data-modeling/data-partitioning)
* [Data Distribution in Apache Ignite](https://www.gridgain.com/resources/blog/data-distribution-in-apache-ignite)
  * [pdf](https://go.gridgain.com/rs/491-TWR-806/images/2019-03-12-AI-meetup-Affinity.pdf)

## Distributed Joins

> [Distributed Joins](https://ignite.apache.org/docs/latest/SQL/distributed-joins)

Distributed Joins 는 **Colocated Joins** 와 **Non-colocated Joins** 와 같이 2 가지가 있다.

If the tables are joined on the partitioning column (affinity key), the join is called a colocated join. Otherwise, it is called a non-colocated join.

affinity key 를 설정해 두면 가까운 Data Entry 들은 같은 Partition 으로 배치된다.

**Colocated Joins**

Client 는 Ignite Cluster Node 들에게 Query 를 보낸다. 각 Node 는 Query 를 실행하고 결과를 Client 에게 돌려준다. Client 는 결과를 모은다.

![](img/collocated_joins.png)

**Non-colocated Joins**

Client 는 Ignite Cluster Node 들에게 Query 를 보낸다. 각 Node 는 broad-cast, uni-cast 를 통해서 missing data 를 주고 받는다. 각 Node 는 결과를 Client 에게 돌려준다. Client 는 결과를 모은다.

![](img/non_collocated_joins.png)

## Affinity Colocation

서로 다른 Table 의 record 라도 affinity 설정을 해 놓으면 같은 partition 에 배치된다.

## Ignite vs Redis

> [GridGain In-Memory Computing Platform
Feature Comparison: Redis](https://go.gridgain.com/rs/491-TWR-806/images/GridGain-Feature-Comparison-Redis-Final.pdf)

## Data Rebalancing

> [Data Rebalancing](https://ignite.apache.org/docs/latest/data-rebalancing)

Ignite Cluster 에 새로운 Node 가 참여하거나 기존의 Node 가 빠져나갔을 때 Partition 이 재배치 되는 것을 말한다.

## Performing Transactions

> [Performing Transactions](https://ignite.apache.org/docs/latest/key-value-api/transactions)

## Baseline Topology

> [Baseline Topology](https://ignite.apache.org/docs/latest/clustering/baseline-topology)

Baseline Topology 가 바뀌면 Data Rebalancing 이 일어난다. 

유지보수를 위해 Node 가 잠깐 빠져나갔다가 다시 들어온다면 Baseline Toplogy 를 바꾸지 말자.

## Partition Awareness

> [Partition Awareness](https://ignite.apache.org/docs/latest/thin-clients/getting-started-with-thin-clients#partition-awareness)

Partition Awareness 가 없는 Thin Client 는 모든 Query 를 하나의 Proxy Node 를 통해 보낸다. 병목현상이 있다.

![](img/partitionawareness01.png)

Partition Areness 가 있는 Thin Client 는 모든 Query 를 Partition 이 배치된 Node 들에게 나눠 보낸다. 병목현상이 없다.

![](img/partitionawareness02.png)
