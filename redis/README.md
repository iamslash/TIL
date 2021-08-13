- [Abstract](#abstract)
- [Materials](#materials)
- [Features](#features)
- [Install](#install)
  - [Install with docker](#install-with-docker)
- [Sentinel](#sentinel)
- [Cluster](#cluster)
- [Commands](#commands)
  - [Client/Server](#clientserver)
  - [Database](#database)
  - [Scripts](#scripts)
  - [HyperL­ogLogs](#hyperloglogs)
  - [Strings](#strings)
  - [Lists](#lists)
  - [Sets](#sets)
  - [Sorted Sets](#sorted-sets)
  - [Hashes](#hashes)
  - [Common](#common)
  - [Geo](#geo)
  - [Pub/Sub](#pubsub)
  - [Streams](#streams)
- [Advanced](#advanced)
  - [How to debug](#how-to-debug)

----

# Abstract

redis 에 대해 정리한다.

# Materials

* [[우아한테크세미나] 191121 우아한레디스 by 강대명님](https://www.youtube.com/watch?v=mPB2CZiAkKM) 
* [강대명 <대용량 서버 구축을 위한 Memcached와 Redis>](https://americanopeople.tistory.com/177)
* [대용량 서버 구축을 위한 Memcached와 Redis](http://www.hanbit.co.kr/store/books/look.php?p_code=E1904063627)
* [redis Introduction](https://bcho.tistory.com/654)
* [redis cheatsheet](https://www.cheatography.com/tasjaevan/cheat-sheets/redis/)
* [redis @ github](https://github.com/antirez/redis)
* [try redis](https://try.redis.io/)
  * web 에서 redis 를 실습할 수 있다.
* [redisgate](http://redisgate.kr/redis/introduction/redis_intro.php)
  * 한글 redis 강좌
* [hiredis](https://github.com/redis/hiredis)
  * minimal redis client

# Features

redis 는 REmote dIctionary System 의 약자이다. 

redis 는 disk 에 데이터를 저장할 수 있다. RDB (snapshot), AOF (append olny file) 의 방법이 있다. RDB 는 한번에 메모리의 데이터를 디스크로 저장하는 방법이다. AOF 는 조금씩 디스크에 저장하는 방법이다. 두가지 방법을 적절히 혼합하여 사용하면 좋다. [참고](http://redisgate.kr/redis/configuration/redis_overview.php)

string, set, sorted set, hash, list 등의 datatype 을 지원한다.

Sentinel 은 redis monitoring tool 이다. redis 의 master, slave 들을 지켜보고 있다가 장애처리를 해준다. `> redis-sentinel sentinel.conf` 와 같이 실행한다.

redis 3.0 부터 cluster 기능을 지원한다.

master 와 여러개의 slave 들로 read replica 구성을 할 수 있다.

# Install

## Install with docker

```bash
$ docker pull redis
$ docker run --rm -p 6379:6379 --name my-redis -d redis
## link volumf of local host
# docker run --name my-redis -d -v /your/dir:/data redis redis-server --appendonly yes
## link volume of other container
# docker run --name my-redis -d --volumes-from some-volume-container redis redis-server --appendonly yes
$ docker exec -it my-redis /bin/bash
> redis-cli

```

# Sentinel

* [sentinel](http://redisgate.kr/redis/sentinel/sentinel.php)
* [twemproxy를 이용한 redis failover @ youtube](https://www.youtube.com/watch?v=xMSVlUnBy6c)
  * twemproxy 와 sentinel 을 이용한 failover 방법을 설명한다. 

----

![](http://redisgate.kr/images/sentinel/sentinel-01.png)

sentinel 은 Master 와 Redis Slave 를 fail over 처리 한다.
sentinel 은 twemproxy 와 같은 machine 에서 실행해야 한다.
만약 Redis Mater 가 죽으면 twemproxy 의 설정파일을 수정하여
Redis Slave 의 주소를 Redis Master 의 주소로 교체한다.

# Cluster

* [Docker기반 Redis 구축하기 - (10) Redis Cluster Mode 설정하기](https://jaehun2841.github.io/2018/12/03/2018-12-03-docker-10/#docker-entrypointsh)
* [vishnunair/docker-redis-cluster](https://hub.docker.com/r/vishnunair/docker-redis-cluster/)
* [레디스 클러스터 소개](http://redisgate.kr/redis/cluster/cluster_introduction.php)
* [레디스 클러스터 구성](http://redisgate.kr/redis/cluster/cluster_configuration.php)
* [[Redis Documentation #2] 레디스 클러스터 튜토리얼](https://medium.com/garimoo/redis-documentation-2-%EB%A0%88%EB%94%94%EC%8A%A4-%ED%81%B4%EB%9F%AC%EC%8A%A4%ED%84%B0-%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC-911ba145e63)
* [Redis - Cluster 설정](https://daddyprogrammer.org/post/1601/redis-cluster/)

----

* Redis 3 부터 cluster mode 를 지원한다.
* Cluster Mode 에서는 Redis Sentinel 의 도움없이 Cluster 자체적으로 Failover 를
  진행한다.
* Cluster Mode 에서는 Master-Slave 노드 구조를 가질 수 있고, 노드 간 Replication
  을 지원한다.
* Cluster Mode 에서는 redis key 의 HashCode 에 대해 CRC16 의 16384 modules (key
  % 16384) 연산을 실행 Auto Sharding을 지원한다.
* Application Sharding 이 필요없기 때문에, Spring-Data-Redis 사용이 가능하다.

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

# Commands

## Client/Server

| Command | Description | Exapmle |
|---------|-------------|---------|
| `SELECT`  | Set current database by index | `> SELECT 8` |

## Database

| Command | Description | Exapmle |
|---------|-------------|---------|
| `DUMP` | Serialise item | |
| `RESTORE` | Deseri­alise | |

## Scripts

| Command | Description | Exapmle |
|---------|-------------|---------|
| `EVAL` | Run | |
| `EVALSHA` | Run cached | |

## HyperL­ogLogs

HyperLogLog 는 집합의 원소의 개수를 추정하는 방법으로 2.8.9 에 추가되었다.

| Command | Description | Exapmle |
|---------|-------------|---------|
| `PFADD` | add elements | `> PFADD k1 e1 e2` |
| `PFCOUNT` | get counts of key | `> PFCOUNT k1` |
| `PFMERGE` | merge keys | `> PFMERGE dstkey k1 k2` |

## Strings

key 와 value 가 일 대 일 관계이다. 한편, Lists, Sets, Sorted Sets, Hashes 는 일 대 다 관계이다.

* SET: `SET, SETNX, SETEX, SETPEX, MSET, MSETNX, APPEND, SETRANGE`
* GET: `GET, MGET, GETRANGE, STRLEN`
* INCR: `INCR, DECR, INCRBY, DECRBY, INCRBYFLOAT`

## Lists

* SET (PUSH): `LPUSH, RPUSH, LPUSHX, RPUSHX, LSET, LINSERT, RPOPLPUSH`
* GET: `LRANGE, LINDEX, LLEN`
* POP: `LPOP, RPOP, BLPOP, BRPOP`
* REM: `LREM, LTRIM`
* BLOCK: `BLPOP, BRPOP, BRPOPLPUSH`

## Sets

* SET: `SADD, SMOVE`
* GET: `SMEMBERS, SCARD, SRANDMEMBER, SISMEMBER, SSCAN`
* POP: `SPOP`
* REM: `SREM`
* 집합연산: `SUNION, SINTER, SDIFF, SUNIONSTORE, SINTERSTORE, SDIFFSTORE`

## Sorted Sets

* [Redis Sorted Set](https://jupiny.com/2020/03/28/redis-sorted-set/)

------

* SET: `ZADD`
* GET: `ZRANGE, ZRANGEBYSCORE, ZRANGEBYLEX, ZREVRANGE, ZREVRANGEBYSCORE, ZREVRANGEBYLEX, ZRANK, ZREVRANK, ZSCORE, ZCARD, ZCOUNT, ZLEXCOUNT, ZSCAN`
* POP: `ZPOPMIN, ZPOPMAX`
* REM: `ZREM, ZREMRANGEBYRANK, ZREMRANGEBYSCORE, ZREMRANGEBYLEX`
* INCR: `ZINCRBY`
* 집합연산: `ZUNIONSTORE, ZINTERSTORE`

## Hashes

* SET: `HSET, HMSET, HSETNX`
* GET: `HGET, HMGET, HLEN, HKEYS, HVALS, HGETALL, HSTRLEN, HSCAN, HEXISTS`
* REM: `HDEL`
* INCR: `HINCRBY, HINCRBYFLOAT`

## Common

5 가지 Data type 에 관계없이 모든 Key 적용되는 명령이다.

* Key 확인, 조회: `EXISTS, KEYS, SCAN, SORT`
* Key 이름 변경: `RENAME, RENAMENX`
* Key 자동 소멸 관련: `EXPIRE, EXPIREAT, TTL, PEXPIRE, EXPIREAT, PTTL, PERSIST`
* 정보 확인: `TYPE, OBJECT`
* 샘플링: `RANDOMKEY`
* Data 이동: `MOVE, DUMP, RESTORE, MIGRATE`

## Geo

3.2 에 도입된 기능이다. 두 지점/도시의 경도(세로선/longitude)와 위도(가로선/latitude)를 입력해서 두 지점의 거리를 구한다.

* 경도/위도 입력: `GEOADD`
* 경도/위도 조회: `GEOPOS`
* 거리 조회: `GEODIST`
* 주변 지점 조회: `GEORADIUSBYMEMBER, GEORADIUS`
* 해시값 조회: `GEOHASH`
* 범위 조회: `ZRANGE`
* 삭제 조회: `ZREM`
* 개수 조회: `ZCARD`

## Pub/Sub

Pub 으로 message 를 보내고 Sub 으로 message 를 받는다.

## Streams

로그 데이터를 처리하기 위해서 5.0 에 도입된 데이터 타입이다.

# Advanced

## How to debug

* [redis debugging in vscode](https://github.com/wenfh2020/youtobe/blob/master/redis-debug.md)

