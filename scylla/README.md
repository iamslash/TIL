- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
  - [Consistency Level](#consistency-level)

----

# Abstract

c++ 로 제작됬다. cassandra 보다 빠르다.

# Materials

* [Getting Started](https://docs.scylladb.com/getting-started/)
  * [Lightweight Transactions](https://docs.scylladb.com/using-scylla/lwt/)
  * [Compaction](https://docs.scylladb.com/kb/compaction/)
* [Best Practices for Data Modeling](https://www.scylladb.com/2019/08/20/best-practices-for-data-modeling/)

# Basic

## Consistency Level

Consistency Level 을 조정할 수 있다. 

* CL of 1: Wait for a response from one replica node
  * Client 의 read, write operation 에 대해 하나의 replica node 만 확인하고 응답한다.
* CL of ALL:  Wait for a response from all replica nodes
* CL LOCAL_QUORUM:  Wait for floor((#dc_replicas/2)+1), meaning that if a DC has 3 nodes replica in the cluster, the application will wait for a response from 2 replica nodes
* CL EACH_QUORUM: For multi DC, each DC must have a LOCAL_QUORUM. This is unsupported for reads.
* CL ALL: All replica nodes must respond. Provides the highest consistency but lowest availability.
