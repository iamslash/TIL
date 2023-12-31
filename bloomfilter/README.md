- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)

----

# Abstract

Bloom Filter 는 확률형 자료구조이다. data 가 있는지 빠르게 검사하고 싶을 때
사용한다. data 가 있다면 반드시 있다는 얘기다. 그러나 없는데 있다고 할 수 있다.
(False Positive)

# Materials

* [Bloom Filter Data Structure: Implementation and Application](https://www.enjoyalgorithms.com/blog/bloom-filter)
* [How to avoid crawling duplicate URLs at Google scale | bytebytego](https://blog.bytebytego.com/p/how-to-avoid-crawling-duplicate-urls?s=r)
* [알아두면 좋은 자료 구조, Bloom Filter](https://steemit.com/kr-dev/@heejin/bloom-filter)
* [Bloom Filters by Example](https://llimllib.github.io/bloomfilter-tutorial/)

# Basic

예를 들어 "Hello" 를 hashing 한 것과 "World" 를 hashing 한 것을 더해서 Bloom
Filter 에 저장한다.

그리고 "Bye" 가 존재하는지 확인해보자. "Bye" 를 hashing 해서 Bloom Filter 에
masking 되어 있는지 확인해 보자. 

"Bye"
가 있다면 "Bye" 의 hash 가 masking 되어 있을 것이다. 그러나 "Bye" 가 없을지라도 "Bye" 의 hash 가 masking 되어 있을 수 있다. 

다음은 bloom filter 를 구현한 java code 이다.

```java
import java.util.BitSet;
import java.util.HashSet;
import java.util.Set;

public class BloomFilter {
  
    private static final int DEFAULT_SIZE = 1000; // Bloom Filter의 기본 크기입니다.
    private BitSet bitSet;
    private int size;
    // 원본 데이터를 저장하기 위한 셋입니다. 실제 Bloom Filter 에서는 사용되지 않습니다.
    private Set<String> actualData; 
    

    public BloomFilter() {
        this(DEFAULT_SIZE);
    }

    public BloomFilter(int size) {
        this.size = size;
        this.bitSet = new BitSet(size);
        this.actualData = new HashSet<>();
    }

    // 값 추가 함수
    public void add(String value) {
        actualData.add(value);

        int hash1 = hash1(value);
        int hash2 = hash2(value);
        int hash3 = hash3(value);

        bitSet.set(hash1 % size);
        bitSet.set(hash2 % size);
        bitSet.set(hash3 % size);
    }

    // 값 확인 함수
    public boolean contains(String value) {
        int hash1 = hash1(value);
        int hash2 = hash2(value);
        int hash3 = hash3(value);

        return bitSet.get(hash1 % size) && bitSet.get(hash2 % size) && bitSet.get(hash3 % size);
    }

    // 1번째 해시 함수
    private int hash1(String value) {
        return value.hashCode();
    }

    // 2번째 해시 함수
    private int hash2(String value) {
        int hash = 7;
        for (int i = 0; i < value.length(); i++) {
            hash = hash * 31 + value.charAt(i);
        }
        return hash;
    }

    // 3번째 해시 함수
    private int hash3(String value) {
        int hash = 0;
        for (int i = 0; i < value.length(); i++) {
            hash = 37 * hash + value.charAt(i);
        }
        return hash;
    }

    public static void main(String[] args) {
        BloomFilter bloomFilter = new BloomFilter();

        // 값을 추가합니다.
        bloomFilter.add("apple");
        bloomFilter.add("banana");
        bloomFilter.add("orange");

        // 값이 포함되어 있는지 확인합니다.
        System.out.println("Is 'apple' in Bloom Filter? " + bloomFilter.contains("apple")); // true
        System.out.println("Is 'banana' in Bloom Filter? " + bloomFilter.contains("banana")); // true
        System.out.println("Is 'orange' in Bloom Filter? " + bloomFilter.contains("orange")); // true
        System.out.println("Is 'grape' in Bloom Filter? " + bloomFilter.contains("grape")); // false
    }
}
```
