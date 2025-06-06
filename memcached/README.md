- [Materials](#materials)
- [Features](#features)
- [Install with docker](#install-with-docker)
- [Architecture](#architecture)
- [Tutorial](#tutorial)
- [Python3 Examples](#python3-examples)
- [Consistency Hashing in Memcached](#consistency-hashing-in-memcached)
- [How to solve SPOF](#how-to-solve-spof)
  - [moxi](#moxi)
  - [zookeeper](#zookeeper)

-----

# Materials

* [memcached @ github](https://github.com/memcached/memcached)
* [How To Install and Secure Memcached on Ubuntu 16.04 @ digitalocean](https://www.digitalocean.com/community/tutorials/how-to-install-and-secure-memcached-on-ubuntu-16-04)
* [Memcached의 확장성 개선 @ D2](https://d2.naver.com/helloworld/151047)
  * memcached 는 코어가 4 개이상인 머신에서 느리다. 이것을 개선한 것에 대한 글이다.
* [memcached wiki](https://github.com/memcached/memcached/wiki)

# Features

memcached 의 확장성 있는 아키텍처 (scale-out) 는 memcached 서버를 간단하게 추가만 하여 처리량을 높일 수 있다. 
그러나 코어 수가 4개를 넘으면 성능 저하가 발생하기 때문에 수직 scalability (scale-up) 에는 문제가 있다.

# Install with docker

```bash
docker pull memcached
# docker run --name my-memcache -d memcached
# docker run --name my-memcache -d memcached memcached -m 64
docker run -p 11211:11211 --name my-memcached -d memcached
docker exec -it my-memcached /bin/bash
telent localhost 11211
```

# Architecture

* Hash Table Array
  * > ![](https://d2.naver.com/content/images/2015/06/helloworld-151047-3.png)   
* LRU
  * > ![](https://d2.naver.com/content/images/2015/06/helloworld-151047-4.png)

# Tutorial

* [Memcached Tutorial @ tutorialpoint](https://www.tutorialspoint.com/memcached/index.htm)

-----

```bash
stat
# STAT pid 1
# STAT uptime 149
# STAT time 1567576140
# STAT version 1.5.17
# STAT libevent 2.1.8-stable
# STAT pointer_size 64
# STAT rusage_user 0.010000
# STAT rusage_system 0.020000
# STAT max_connections 1024
# STAT curr_connections 2
# STAT total_connections 3
# STAT rejected_connections 0
# STAT connection_structures 3
# STAT reserved_fds 20
# STAT cmd_get 0
# STAT cmd_set 0
# STAT cmd_flush 0
# STAT cmd_touch 0
# STAT get_hits 0
# STAT get_misses 0
# STAT get_expired 0
# STAT get_flushed 0
# STAT delete_misses 0
# STAT delete_hits 0
# STAT incr_misses 0
# STAT incr_hits 0
# STAT decr_misses 0
# STAT decr_hits 0
# STAT cas_misses 0
# STAT cas_hits 0
# STAT cas_badval 0
# STAT touch_hits 0
# STAT touch_misses 0
# STAT auth_cmds 0
# STAT auth_errors 0
# STAT bytes_read 7
# STAT bytes_written 0
# STAT limit_maxbytes 67108864
# STAT accepting_conns 1
# STAT listen_disabled_num 0
# STAT time_in_listen_disabled_us 0
# STAT threads 4
# STAT conn_yields 0
# STAT hash_power_level 16
# STAT hash_bytes 524288
# STAT hash_is_expanding 0
# STAT slab_reassign_rescues 0
# STAT slab_reassign_chunk_rescues 0
# STAT slab_reassign_evictions_nomem 0
# STAT slab_reassign_inline_reclaim 0
# STAT slab_reassign_busy_items 0
# STAT slab_reassign_busy_deletes 0
# STAT slab_reassign_running 0
# STAT slabs_moved 0
# STAT lru_crawler_running 0
# STAT lru_crawler_starts 510
# STAT lru_maintainer_juggles 199
# STAT malloc_fails 0
# STAT log_worker_dropped 0
# STAT log_worker_written 0
# STAT log_watcher_skipped 0
# STAT log_watcher_sent 0
# STAT bytes 0
# STAT curr_items 0
# STAT total_items 0
# STAT slab_global_page_pool 0
# STAT expired_unfetched 0
# STAT evicted_unfetched 0
# STAT evicted_active 0
# STAT evictions 0
# STAT reclaimed 0
# STAT crawler_reclaimed 0
# STAT crawler_items_checked 0
# STAT lrutail_reflocked 0
# STAT moves_to_cold 0
# STAT moves_to_warm 0
# STAT moves_within_lru 0
# STAT direct_reclaims 0
# STAT lru_bumps_dropped 0
# END

# set key flags exptime bytes [noreply] 
# value 
set A 0 900 9
memcached
# STORED | ERROR
STORED
# get key
get A
VALUE A 0 9
memcached
END

# key − key of datum.
# flags − metadata???
# exptime − expiration time in seconds.
# bytes − It is the number of bytes in the data block.
# noreply (optional) - It is a parameter that informs the server not to send any reply.
# value − value of datum.
```

# Python3 Examples

* [Python + Memcached: Efficient Caching in Distributed Applications](https://realpython.com/python-memcache-efficient-caching/)

```bash
pip install pymemcache
>>> from pymemcache.client import base
# Don't forget to run `memcached' before running this next line:
>>> client = base.Client(('localhost', 11211))
# Once the client is instantiated, you can access the cache:
>>> client.set('some_key', 'some value')
# Retrieve previously set data again:
>>> client.get('some_key')
'some value'
```

# Consistency Hashing in Memcached

Memcached 는 consistency hashing 을 이용하여 data 를 분배한다. [consistency hashing](/consistenthasing/README.md)

memcached 를 4 대 (`-p 20001, -p 20002, -p 20003, -p 20004`) 를 띄워보자. 그리고 다음과 같이 set.py 를 제작하여 실행한다.

```py
import memcache
ServerList = ['127.0.0.1:20001', '127.0.0.1:20002', '127.0.0.1:20003', '127.0.0.1:20004']
if __name__ == '__main__':
  mc = memcache.Client(ServerList)
  for idx in range(1, 10):
    mc.set(str(idx), str(idx))
```

이제 10 개의 값이 4 개의 서버에 분배되었다. 각 서버에 들어가서 값을 확인해 보자.

```console
$ telnet 127.0.0.1 20001
Trying 127.0.0.1...
connected to 127.0.0.1.
Escape character is '^]'.
get 1
VALUE 1 0 1
1
END

$ telnet 127.0.0.1 20002
Trying 127.0.0.1...
connected to 127.0.0.1.
Escape character is '^]'.
get 1
END
get 2
VALUE 2 0 1
2
END
```

[libmemcached @ github](https://github.com/libmemcached/libmemcached/blob/eda2becbec24363f56115fa5d16d38a2d1f54775/libmemcached/hash.cc) 의 코드는 다음과 같다. dispatch_host 가 consistent hashing 을 이용하여 분배할 서버의 hash 값을 리턴한다.

```cpp
uint32_t memcached_generate_hash_with_redistribution(memcached_st *ptr, const char *key, size_t key_length)
{
  uint32_t hash= _generate_hash_wrapper(ptr, key, key_length);

  _regen_for_auto_eject(ptr);

  return dispatch_host(ptr, hash);
}
```

# How to solve SPOF

## moxi

* moxi @ TIL

memcached 의 proxy 이다. SPOF 를 해결하기 위해 Load Balancer 뒤에
moxi 를 여러대 운용한다. 

## zookeeper

zookeepr 를 이용해서 memcached 를 service dicovery 하여 SPOF 를
해결할 수도 있다.
