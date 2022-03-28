# Abstract

Redis 는 Cluster 기능을 제공한다. slot 은 16384 이다. 이것을 Primary Node 들이
나눠 갖는다. Primary Node 들은 Replication Node 로 Replication 한다.
Availability 가 보장된다. Primary Node 에서 Replica Node 로 data 가 aync 하게
이동하기 때문에 완벽한 Consistency 는 보장하지 않는다. 

# Materials

* [Docker기반 Redis 구축하기 - (10) Redis Cluster Mode 설정하기](https://jaehun2841.github.io/2018/12/03/2018-12-03-docker-10/#docker-entrypointsh)
* [vishnunair/docker-redis-cluster](https://hub.docker.com/r/vishnunair/docker-redis-cluster/)
* [레디스 클러스터 소개](http://redisgate.kr/redis/cluster/cluster_introduction.php)
* [레디스 클러스터 구성](http://redisgate.kr/redis/cluster/cluster_configuration.php)
* [[Redis Documentation #2] 레디스 클러스터 튜토리얼](https://medium.com/garimoo/redis-documentation-2-%EB%A0%88%EB%94%94%EC%8A%A4-%ED%81%B4%EB%9F%AC%EC%8A%A4%ED%84%B0-%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC-911ba145e63)
* [Redis - Cluster 설정](https://daddyprogrammer.org/post/1601/redis-cluster/)
* [[Redis Documentation #2] 레디스 클러스터 튜토리얼](https://medium.com/garimoo/redis-documentation-2-%EB%A0%88%EB%94%94%EC%8A%A4-%ED%81%B4%EB%9F%AC%EC%8A%A4%ED%84%B0-%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC-911ba145e63)

# Basic

## Overview

* Redis 3 부터 cluster mode 를 지원한다.
* Cluster Mode 에서는 Redis Sentinel 의 도움없이 Cluster 자체적으로 Failover 를
  진행한다.
* Cluster Mode 에서는 Master-Slave 노드 구조를 가질 수 있고, 노드 간 Replication
  을 지원한다.
* Cluster Mode 에서는 redis key 의 HashCode 에 대해 `CRC16` 의 `key % 16384`
  연산을 실행 Auto Sharding을 지원한다.
* Application Sharding 이 필요없기 때문에, `Spring-Data-Redis` 사용이 가능하다.

```bash
$ docker pull vishnunair/docker-redis-cluster:latest
$ docker run --rm -d -p 6000:6379 -p 6001:6380 -p 6002:6381 -p 6003:6382 -p 6004:6383 -p 6005:6384 --name my-redis-cluster vishnunair/docker-redis-cluster
$ docker exec -it my-redis-cluster redis-cli
# 127.0.0.1:6379> SET helloworld 1
# OK
# 127.0.0.1:6379> SET helloworld 2
# OK
# 127.0.0.1:6379> GET helloworld
# "2"
```
