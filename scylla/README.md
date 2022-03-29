- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
  - [Architecture](#architecture)
    - [Scylla Architecture](#scylla-architecture)
    - [Shard-per-Core Architecture](#shard-per-core-architecture)
    - [High Availability](#high-availability)
    - [Networking in Scylla](#networking-in-scylla)
    - [Memory Management](#memory-management)
  - [Consistency Level](#consistency-level)

----

# Abstract

c++ 로 제작됬다. [Cassandra](/cassandra/README.md) 보다 빠르다.

# Materials

* [Getting Started](https://docs.scylladb.com/getting-started/)
  * [Lightweight Transactions](https://docs.scylladb.com/using-scylla/lwt/)
  * [Compaction](https://docs.scylladb.com/kb/compaction/)
* [Best Practices for Data Modeling](https://www.scylladb.com/2019/08/20/best-practices-for-data-modeling/)
* [Scylla University](https://university.scylladb.com/)
  * [S101: Scylla Essentials](https://university.scylladb.com/courses/scylla-operations/)
  * [S301: Operations](https://university.scylladb.com/courses/scylla-operations/)
  * [S201: Data Modeling](https://university.scylladb.com/courses/data-modeling/)
  * [S110: Mutant Monitoring System (MMS)](https://university.scylladb.com/courses/the-mutant-monitoring-system-training-course/)
  * [S210: Using Scylla Drivers](https://university.scylladb.com/courses/using-scylla-drivers/)
  * [S310: Scylla Alternator](https://university.scylladb.com/courses/scylla-alternator/)

# Basic

## Architecture

> [Scylla Architecture](https://www.scylladb.com/product/technology/)

다음과 같은 특징들을 가지고 있다. `vNode, Shard, Partition` 은 무슨 차이가 있는가???

### Scylla Architecture

* Compatibilities and Differernces
  * [Cassandra](/cassandra/README.md), [DynamoDB](/dynamodb/README.md) 의 API 를 지원한다.
* Scylla Server Architecture
  * `Cluster`
  * `Node`
  * `Shard`
    * Data 는 Shard 로 나누어 진다.
* Data Architecture
  * `Keyspace`
  * `Table`
  * `Partition`
    * Table 의 data 는 Partition 으로 나누어 진다. 여러 Shard 에 배치된다.
  * `Columns`
* Ring Architecture  
  * `Ring`
  * `vNode`
    * Virtual Node 이다. Replication Factor 만큼 Replication 되어 Physical Node 에 배치된다.
* Storage Architecture
  * `Memtable`
    * Mem 에 있다.
  * `Commitlog`
    * Durability 를 위해 Write Operation 을 기록한 log
  * `SSTables`
    * Disk 에 있다.
  * `Tombstones`
  * `Compactions`
  * `Compaction Strategy`
* Client-Server Architecture
  * `Drivers`

### Shard-per-Core Architecture

하나의 Shard 는 하나의 Core 가 처리한다. Locks Free 해서 Performance 가 좋다.

### High Availability

* Peer-to-Peer Architecture
* Automatic Data Replication
* Scylla and the CAP Theorem
  * **CP** or **AP**
* Tunalbe Consistency
  * Consistency Level 을 조정할 수 있다.
* Achieving Zero Downtime
  * Rack and Datacenter Awareness
  * Multi-Datacenter Replication
* Anti-Entropy
  * Hinted Handoffs
    * 일시적으로 장애가 생긴 node 는 cluster 에서 빠져나갔을 때 발생한
      transaction 들을 모아둔다. 그 node 가 다시 cluster 에 참여했을 때 catch up
      할 수 있다.
  * Row-level Repair
    * `nodetool repair` command 를 통해 backup 으로 부터 data 를 restore 할 수
      있다.

### Networking in Scylla

* Scylla supports multiple networking protocols as part of our client-server networking with RPC Streaming.
  * Cassandra Query Language (CQL)
  * Apache Thrift
  * HTTP/HTTPS RESTful API
* Scylla Shard-Aware Drivers for CQL
* Server-to-Server Networking with RPC Streaming
* Secure Networking

### Memory Management

* How Scylla Maximizes the Usage of Memory
* **Memtable** and **Row-Based Cache**
* In-Memory Tables
  * **Scylla Enterprise** and **Scylla Cloud** support this.

## Consistency Level

Consistency Level 을 조정할 수 있다. 

* CL of 1: Wait for a response from one replica node
  * Client 의 read, write operation 에 대해 하나의 replica node 만 확인하고 응답한다.
* CL of ALL:  Wait for a response from all replica nodes
* CL LOCAL_QUORUM:  Wait for floor((#dc_replicas/2)+1), meaning that if a DC has 3 nodes replica in the cluster, the application will wait for a response from 2 replica nodes
* CL EACH_QUORUM: For multi DC, each DC must have a LOCAL_QUORUM. This is unsupported for reads.
* CL ALL: All replica nodes must respond. Provides the highest consistency but lowest availability.
