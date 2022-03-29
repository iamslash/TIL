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
  - [Removing a node](#removing-a-node)
    - [Replica migration](#replica-migration)
  - [Upgrading nodes in a Redis Cluster](#upgrading-nodes-in-a-redis-cluster)
  - [Migrating to Redis Cluster](#migrating-to-redis-cluster)

---

# Abstract

Redis 는 Cluster 기능을 제공한다. Redis 는 [consistent hashing](/consistenthasing/) 을 사용하지 않는다. 16384 개의 slot 을 사용한다. 이것을 Primary Node 들이 나눠 갖는다. Primary Node 들은 Replication Node 로
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

6 개의 node 로 구성된 Redis Cluster 를 만들어 보자.s

6 개의 node 를 위한 directory 를 생성한다.

```bash
$ mkdir cluster-test
$ cd cluster-test
$ mkdir 7000 7001 7002 7003 7004 7005
$ vim 7000/redis.conf
```

`redis.conf` 는 다음과 같다. node 별로 port 값을 다르게 설정한다.

```
port 7000
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000
appendonly yes
```

node 별로 다음과 같이 실행한다.

```bash
$ cd 7000
$ redis-server ./redis.conf
[82462] 26 Nov 11:56:55.329 * No cluster configuration found, I'm 
```

### Initializing the cluster

```bash
$ redis-cli --cluster create 127.0.0.1:7000 127.0.0.1:7001 \
    127.0.0.1:7002 127.0.0.1:7003 127.0.0.1:7004 127.0.0.1:7005 \
    --cluster-replicas 1
[OK] All 16384 slots covered
```

### Creating a Redis Cluster using the create-cluster script

`~/utils/create-cluster` script 를 이용하여 Redis Cluster 를 생성하고 실행할 수 있다.
`~/utils/create-cluster/README` 를 읽어라.

```
$ create-cluster start
$ create-cluster create
$ create-cluster stop
```

### Interacting with the cluster

Client 의 요청을 Server 가 받아서 해당 slot 을 갖는 Node 에게 redirection 한다.

```bash
$ redis-cli -c -p 7000
redis 127.0.0.1:7000> set foo bar
-> Redirected to slot [12182] located at 127.0.0.1:7002
OK
redis 127.0.0.1:7002> set hello world
-> Redirected to slot [866] located at 127.0.0.1:7000
OK
redis 127.0.0.1:7000> get foo
-> Redirected to slot [12182] located at 127.0.0.1:7002
"bar"
redis 127.0.0.1:7002> get hello
-> Redirected to slot [866] located at 127.0.0.1:7000
"world"
```

### Writing an example app with redis-rb-cluster

[Writing an example app with redis-rb-cluster](https://redis.io/docs/manual/scaling/#creating-and-using-a-redis-cluster)

### Resharding the cluster

```bash
# Get a node id
$ redis-cli -p 7000 cluster nodes | grep myself
97a3a64667477371c4479320d683e4c8db5858b1 :0 myself,master - 0 

$ redis-cli --cluster reshard 127.0.0.1:7000
How many slots do you want to move (from 1 to 16384)?

$ redis-cli --cluster check 127.0.0.1:7000
```

### Scripting a resharding operation

```bash
$ redis-cli --cluster reshard <host>:<port> \
  --cluster-from <node-id> --cluster-to <node-id> \
  --cluster-slots <number of slots> --cluster-yes
```

### A more interesting example application

[A more interesting example application](https://redis.io/docs/manual/scaling/#a-more-interesting-example-application)

## Testing the failover

```bash
$ redis-cli -p 7000 cluster nodes | grep master
3e3a6cb0d9a9a87168e266b0a0b24026c0aae3f0 127.0.0.1:7001 master - 0 1385482984082 0 connected 5960-10921
2938205e12de373867bf38f1ca29d31d0ddb3e46 127.0.0.1:7002 master - 0 1385482983582 0 connected 11423-16383
97a3a64667477371c4479320d683e4c8db5858b1 :0 myself,master - 0 0 0 connected 0-5959 10922-11422

$ redis-cli -p 7002 debug segfault
Error: Server closed the connection

# Error ouputs
18849 R (0 err) | 18849 W (0 err) |
23151 R (0 err) | 23151 W (0 err) |
27302 R (0 err) | 27302 W (0 err) |

... many error warnings here ...

29659 R (578 err) | 29660 W (577 err) |
33749 R (578 err) | 33750 W (577 err) |
37918 R (578 err) | 37919 W (577 err) |
42077 R (578 err) | 42078 W (577 err) |

$ redis-cli -p 7000 cluster nodes
3fc783611028b1707fd65345e763befb36454d73 127.0.0.1:7004 slave 3e3a6cb0d9a9a87168e266b0a0b24026c0aae3f0 0 1385503418521 0 connected
a211e242fc6b22a9427fed61285e85892fa04e08 127.0.0.1:7003 slave 97a3a64667477371c4479320d683e4c8db5858b1 0 1385503419023 0 connected
97a3a64667477371c4479320d683e4c8db5858b1 :0 myself,master - 0 0 0 connected 0-5959 10922-11422
3c3a0c74aae0b56170ccb03a76b60cfe7dc1912e 127.0.0.1:7005 master - 0 1385503419023 3 connected 11423-16383
3e3a6cb0d9a9a87168e266b0a0b24026c0aae3f0 127.0.0.1:7001 master - 0 1385503417005 0 connected 5960-10921
2938205e12de373867bf38f1ca29d31d0ddb3e46 127.0.0.1:7002 slave 3c3a0c74aae0b56170ccb03a76b60cfe7dc1912e 0 1385503418016 3 connected
```

### Manual failover

`CLUSTER FAILOVER` command 를 통해서 특정 node 를 failover 할 수 있다. 주로
rolling upgrade 할 때 사용한다. 반드시 Replica node 에 사용해야 한다.

### Adding a new node

다음과 같이 master node 를 추가한다.

```bash
$ redis-cli --cluster add-node 127.0.0.1:7006 127.0.0.1:7000

redis 127.0.0.1:7006> cluster nodes
3e3a6cb0d9a9a87168e266b0a0b24026c0aae3f0 127.0.0.1:7001 master - 0 1385543178575 0 connected 5960-10921
3fc783611028b1707fd65345e763befb36454d73 127.0.0.1:7004 slave 3e3a6cb0d9a9a87168e266b0a0b24026c0aae3f0 0 1385543179583 0 connected
f093c80dde814da99c5cf72a7dd01590792b783b :0 myself,master - 0 0 0 connected
2938205e12de373867bf38f1ca29d31d0ddb3e46 127.0.0.1:7002 slave 3c3a0c74aae0b56170ccb03a76b60cfe7dc1912e 0 1385543178072 3 connected
a211e242fc6b22a9427fed61285e85892fa04e08 127.0.0.1:7003 slave 97a3a64667477371c4479320d683e4c8db5858b1 0 1385543178575 0 connected
97a3a64667477371c4479320d683e4c8db5858b1 127.0.0.1:7000 master - 0 1385543179080 0 connected 0-5959 10922-11422
3c3a0c74aae0b56170ccb03a76b60cfe7dc1912e 127.0.0.1:7005 master - 0 1385543177568 3 connected 11423-16383
```

새로 추가된 node 는 master node 이다. redirection 은 수행한다. 그러나 할당된 hash slot 이 없다.

### Adding a new node as a replica

다음과 같이 slave node 를 추가한다.

```bash
$ redis-cli --cluster add-node 127.0.0.1:7006 127.0.0.1:7000 --cluster-slave

# Start to replicate
$ redis 127.0.0.1:7006> cluster replicate 3c3a0c74aae0b56170ccb03a76b60cfe7dc1912e

# The node 3c3a0c… now has two replicas, 
# running on ports 7002 (the existing one) and 7006 (the new one).
$ redis-cli -p 7000 cluster nodes | grep slave | grep 3c3a0c74aae0b56170ccb03a76b60cfe7dc1912e
f093c80dde814da99c5cf72a7dd01590792b783b 127.0.0.1:7006 slave 3c3a0c74aae0b56170ccb03a76b60cfe7dc1912e 0 1385543617702 3 connected
2938205e12de373867bf38f1ca29d31d0ddb3e46 127.0.0.1:7002 slave 3c3a0c74aae0b56170ccb03a76b60cfe7dc1912e 0 1385543617198 3 connected
```


## Removing a node

node 를 제거하기 전에 반드시 그 node 는 비어 있어야 한다. 그렇지 않다면 resharding 
을 해서 비워야 한다.

```bash
$ redis-cli --cluster del-node 127.0.0.1:7000 `<node-id>`
```

### Replica migration

Replica node 가 바라보는 master node 를 바꿀 수 있다???

```bash
> CLUSTER REPLICATE <master-node-id>
```

## Upgrading nodes in a Redis Cluster

Replica node 의 upgrade 는 쉽다. failover 되도 서비스에 지장이 없기 때문이다.

Master node 의 upgrade 는 다음과 같이 한다.

* Use `CLUSTER FAILOVER` to trigger a manual failover of the master to one of its replicas. (See the Manual failover section in this document.)
* Wait for the master to turn into a replica.
* Finally upgrade the node as you do for replicas.
* If you want the master to be the node you just upgraded, trigger a new manual failover in order to turn back the upgraded node into a master.

## Migrating to Redis Cluster

???
