# Application server cache
# Content Distribution Network (CDN)
# Cache Invalidation

* Write-through cache : cache, storage 에 동시에 기록한다. 동시에 기록하기 때문에 latency 가 높다.

* Write-around cache : storage 에 기록한다.

* Write-back cache : cache 에 기록한다. cache miss 일 때 storage 에 기록하고 cache 에 기록한다. latency 는 낮지만 cache server 가 장애가 발생했을 때 데이터를 유실할 수 있다.

# Cache eviction policies

* First In First Out (FIFO)
* Last In First Out (LIFO)
* Least Recently Used (LRU)
* Most Recently Used (MRU)
* Least Frequently Used (LFU)
* Random Replacement (RR)

# References

* [Cache](https://en.wikipedia.org/wiki/Cache_(computing))
* [Introduction to architecting systems](https://lethain.com/introduction-to-architecting-systems-for-scale/)