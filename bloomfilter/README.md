# Abstract

Bloom Filter 는 확률형 자료구조이다. data 가 있는지 빠르게 검사하고 싶을 때
사용한다. data 가 있다면 반드시 있다는 얘기다. 그러나 없는데 있다고 할 수 있다.

# Materials

* [알아두면 좋은 자료 구조, Bloom Filter](https://steemit.com/kr-dev/@heejin/bloom-filter)
* [Bloom Filters by Example](https://llimllib.github.io/bloomfilter-tutorial/)

# Basic

예를 들어 "Hello" 를 hashing 한 것과 "World" 를 hashing 한 것을 더해서 Bloom
Filter 에 저장한다.

그리고 "Bye" 가 존재하는지 확인해보자. "Bye" 를 hashing 해서 Bloom Filter 에
masking 되어 있는지 확인해 보자. 

"Bye"
가 있다면 "Bye" 의 hash 가 masking 되어 있을 것이다. 그러나 "Bye" 가 없을지라도 "Bye" 의 hash 가 masking 되어 있을 수 있다. 
