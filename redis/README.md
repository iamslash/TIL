# Abstract

redis 에 대해 정리한다.

# Materials

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

# Commands

## Client/Server

| Command | Description | Exapmle |
|---------|-------------|---------|
| `SELECT`  | Set current database by index | `> SELECT 8` |
| | |

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

HyperLogLog 는 집합의 원소의 개수를 추정하는 방법으로 2.8.9 추가되었다.

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

* SET: `ZADD`
* GET: `ZRANGE, ZRANGEBYSCORE, ZRANGEBYLEX, ZREVRANGE, ZREVRANGEBYSCORE, ZREVRANGEBYLEX, ZRANK, ZREVRANK, ZSCORE, ZCARD, ZCOUNT, ZLEXCOUNT, ZSCAN`
* POP: `ZPOPMIN, ZPOPMAX`
* REM: `ZREM, ZREMRANGEBYRANK, ZREMRANGEBYSCORE, ZREMRANGEBYLEX
INCR: ZINCRBY`
* 집합연산: `ZUNIONSTORE, ZINTERSTORE`

| Command | Description | Exapmle |
|---------|-------------|---------|
| | | |

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
