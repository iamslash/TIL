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

## Features

Redis Cluster 는 다음과 같은 기능을 제공한다.

* Data Sets 을 여러 node 에 분산하여 저장한다. Auto Sharding.
* node 의 일부에 장애가 발생해도 service 가 가능하다. High Availability.
