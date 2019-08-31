# Abstract

key 가 주어졌을 때 hash ring 에 적절히 hash 가 되도록 하는 방법이다. hash ring 에 5 개의 서버 A, B, C, D, E 가 있을 때 임의의 서버가 장애가 발생하고 복구되더라도 데이터의 이동을 최소화 할 수 있다. sharding 과 같은 partitioning 전략을 도입할 때 필수이다.

# Materials

* [A Fast, Minimal Memory, Consistent Hash Algorithm @ arxiv ](https://arxiv.org/pdf/1406.2294.pdf)
  * consistent hash 를 5 줄로 구현한 논문 

# Basic

![](basic.png)

위의 그림과 같이 server A, B, C 가 있다. 그리고 키 1, 2, 3, 4, 5 를 server 에 배치해 보자. A 는 3, 4 가 할당되고 B 는 1 이 할당되고 C 는 2, 5 가 할당된다.

만약 B 가 장애가 발생하면 B 가 가지고 있던 1 은 유실된다. 그리고 클라이언트가 1 을 요청했을 때 C 에 1 이 할당된다. 따라서 C 는 1, 2, 5 를 가지고 있다.

B 가 복구되고 클라이언트가 1 을 요청하면 B 에 할당된다. 이 때 C 역시 예전의 1 을 가지고 있다는 것을 유의해야 한다. 이것은 expire time 을 이용하여 예전의 1 이 소멸되도록 해야 한다.

이렇게 consistent hash 를 이용하여 key 를 할당하면 일부 서버가 장애가 발생하더라도 데이터의 이동을 최소화할 수 있다.

마지막으로 hash ring 을 더욱 최적화 할 수 있다. A 의 영역을 A, A+1, A+2, A+3 와 같이 이곳 저곳에 배치하는 것이다. 이렇게 하면 key 들이 여러 서버에 적절히 분산될 수 있도록 할 수 있다.

# Implementation

```c
int32_t JumpConsistentHash(uint64_t key, int32_t num_buckets) {
   int64_t b = 1,   j = 0;
   while (j < num_buckets) {
        b = j;
        key = key * 2862933555777941757ULL + 1;
        j = (b + 1) * (double(1LL << 31) / double((key >> 33) + 1));
    }
    return b;
}
```