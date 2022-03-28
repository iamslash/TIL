- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
  - [Redis Cluster 101](#redis-cluster-101)
    - [Redis Cluster TCP ports](#redis-cluster-tcp-ports)
    - [Redis Cluster and Docker](#redis-cluster-and-docker)
    - [Redis Cluster data sharding](#redis-cluster-data-sharding)
    - [Redis Cluster master-replica model](#redis-cluster-master-replica-model)
    - [Redis Cluster consistency guarantees](#redis-cluster-consistency-guarantees)
  - [Redis Cluster configuration parameters](#redis-cluster-configuration-parameters)
  - [Creating and using a Redis Cluster](#creating-and-using-a-redis-cluster)
    - [Initializing the cluster](#initializing-the-cluster)
    - [Creating a Redis Cluster using the create-cluster script](#creating-a-redis-cluster-using-the-create-cluster-script)
    - [Interacting with the cluster](#interacting-with-the-cluster)
    - [Writing an example app with redis-rb-cluster](#writing-an-example-app-with-redis-rb-cluster)
    - [Resharding the cluster](#resharding-the-cluster)
    - [Scripting a resharding operation](#scripting-a-resharding-operation)
    - [A more interesting example application](#a-more-interesting-example-application)
  - [Testing the failover](#testing-the-failover)
    - [Manual failover](#manual-failover)
    - [Adding a new node](#adding-a-new-node)
    - [Adding a new node as a replica](#adding-a-new-node-as-a-replica)
    - [Replica migration](#replica-migration)
  - [Upgrading nodes in a Redis Cluster](#upgrading-nodes-in-a-redis-cluster)
  - [Migrating to Redis Cluster](#migrating-to-redis-cluster)

---

# Abstract

Redis 는 Cluster 기능을 제공한다. Redis 는 [consistenthashing](/consistenthasing/) 을 사용하지 않는다. 16384 개의 slot 을 사용한다. 이것을 Primary Node 들이 나눠 갖는다. Primary Node 들은 Replication Node 로
Replication 한다. Availability 가 보장된다. Primary Node 에서 Replica Node 로
data 가 aync 하게 이동하기 때문에 완벽한 Consistency 는 보장하지 않는다. 

# Materials

* [Docker기반 Redis 구축하기 - (10) Redis Cluster Mode 설정하기](https://jaehun2841.github.io/2018/12/03/2018-12-03-docker-10/#docker-entrypointsh)
* [vishnunair/docker-redis-cluster](https://hub.docker.com/r/vishnunair/docker-redis-cluster/)
* [레디스 클러스터 소개](http://redisgate.kr/redis/cluster/cluster_introduction.php)
* [레디스 클러스터 구성](http://redisgate.kr/redis/cluster/cluster_configuration.php)
* [[Redis Documentation #2] 레디스 클러스터 튜토리얼](https://medium.com/garimoo/redis-documentation-2-%EB%A0%88%EB%94%94%EC%8A%A4-%ED%81%B4%EB%9F%AC%EC%8A%A4%ED%84%B0-%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC-911ba145e63)
* [Redis - Cluster 설정](https://daddyprogrammer.org/post/1601/redis-cluster/)
* [Scaling with Redis Cluster](https://redis.io/docs/manual/scaling/#creating-and-using-a-redis-cluster)
  * [[Redis Documentation #2] 레디스 클러스터 튜토리얼](https://medium.com/garimoo/redis-documentation-2-%EB%A0%88%EB%94%94%EC%8A%A4-%ED%81%B4%EB%9F%AC%EC%8A%A4%ED%84%B0-%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC-911ba145e63)
  * [Redis cluster specification](https://redis.io/docs/reference/cluster-spec/)
  
# Basic

## Redis Cluster 101

Redis Cluster 는 다음과 같은 기능을 제공한다.

* Data Sets 을 여러 node 에 분산하여 저장한다. Auto Sharding.
* node 의 일부에 장애가 발생해도 service 가 가능하다. High Availability.

### Redis Cluster TCP ports

다음과 같이 2 가지 TCP port 가 필요하다.

* `6379`: Client to Server for commands
* `16379`: Server to Server for cluster bus

### Redis Cluster and Docker

Redis Cluster 는 NATted environments 를 지원하지 않는다. Docker 로 실행하길
원한다면 host networking mode 를 사용하라.

### Redis Cluster data sharding

Redis Cluster 는 [consistenthashing](/consistenthasing/) 을 사용하지 않는다.
16384 개의 hash slot 을 사용한다. `key % 16384` slot 이 할당된 node 에 data 를
저장한다.

모든 node 는 hash slot range 를 갖고 있다. 예를 들어 3 개의 node 가 있다고 하자.
각 node 는 다음과 같이 hash slot range 를 갖는다.

* Node A contains hash slots from 0 to 5500.
* Node B contains hash slots from 5501 to 11000.
* Node C contains hash slots from 11001 to 16383.

### Redis Cluster master-replica model

Primary Node 는 N 개의 Replica Node 를 갖을 수 있다. Primary Node 에 장애가
발생하면 Replica Node 중 하나가 Primary Node 로 promotion 된다.

### Redis Cluster consistency guarantees

Redis Cluster 는 strong consistency 를 보장하지 못한다. 예를 들어 다음과 같은 경우를
살펴보자.

* Your client writes to the master B.
* The master B replies OK to your client.
* The master B propagates the write to its replicas B1, B2 and B3.

B 는 write operation 에 대해 B1, B2, B3 로 부터 응답을 기다리지 않고 Client 에게
응답을 보낸다. B1, B2, B3 로 write operation 이 전파되기 전에 B 에 장애가
발생했다고 해보자. B1 이 Primary node 로 promotion 되면 B 에 수행했던 write
operation 은 유실된다.

## Redis Cluster configuration parameters

* `cluster-enabled <yes/no>`
* `cluster-config-file <filename>`
* `cluster-node-timeout <milliseconds>`
* `cluster-slave-validity-factor <factor>`
* `cluster-migration-barrier <count>`
* `cluster-require-full-coverage <yes/no>`
* `cluster-allow-reads-when-down <yes/no>`

## Creating and using a Redis Cluster

### Initializing the cluster

### Creating a Redis Cluster using the create-cluster script

### Interacting with the cluster

### Writing an example app with redis-rb-cluster

### Resharding the cluster

### Scripting a resharding operation

### A more interesting example application

## Testing the failover

### Manual failover

### Adding a new node

### Adding a new node as a replica

### Replica migration

## Upgrading nodes in a Redis Cluster

## Migrating to Redis Cluster
