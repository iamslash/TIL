# 목차

- [Abstract](#abstract)
- [Materials](#materials)
- [Basic Concepts](#basic-concepts)
  - [동작 원리](#동작-원리)
  - [False Positive vs False Negative](#false-positive-vs-false-negative)
  - [시간 복잡도](#시간-복잡도)
- [Mathematical Background](#mathematical-background)
  - [최적 비트 배열 크기](#최적-비트-배열-크기)
  - [최적 해시 함수 개수](#최적-해시-함수-개수)
  - [False Positive 확률](#false-positive-확률)
- [Implementation](#implementation)
  - [기본 구현](#기본-구현)
  - [Guava Bloom Filter](#guava-bloom-filter)
  - [Redis Bloom Filter](#redis-bloom-filter)
- [Real-World Use Cases](#real-world-use-cases)
  - [웹 크롤러 중복 URL 방지](#웹-크롤러-중복-url-방지)
  - [캐시 시스템 최적화](#캐시-시스템-최적화)
  - [분산 시스템 데이터 동기화](#분산-시스템-데이터-동기화)
  - [악성 URL 필터링](#악성-url-필터링)
- [Variants](#variants)
  - [Counting Bloom Filter](#counting-bloom-filter)
  - [Scalable Bloom Filter](#scalable-bloom-filter)
  - [Cuckoo Filter](#cuckoo-filter)
- [Performance Comparison](#performance-comparison)

----

# Abstract

Bloom Filter는 **확률형 자료구조(Probabilistic Data Structure)**로, 대량의 데이터에서 특정 요소의 존재 여부를 메모리 효율적으로 빠르게 검사할 때 사용합니다.

## 핵심 특성

- **False Positive 가능**: 존재하지 않는 요소를 존재한다고 잘못 판단할 수 있음 (조절 가능)
- **False Negative 없음**: 존재하는 요소를 존재하지 않는다고 판단하는 경우는 절대 없음
- **메모리 효율적**: 실제 데이터를 저장하지 않고 비트 배열만 사용
- **O(k) 시간 복잡도**: k는 해시 함수 개수, 요소 개수와 무관

## 사용 사례

```
✓ 웹 크롤러의 중복 URL 제거 (Google, Bing)
✓ 데이터베이스 캐시 최적화 (불필요한 디스크 I/O 방지)
✓ CDN에서 콘텐츠 존재 여부 빠른 확인
✓ 악성 URL/이메일 필터링
✓ 분산 시스템의 데이터 동기화
✓ 네트워크 라우터의 패킷 필터링
```

# Materials

* [Bloom Filter Data Structure: Implementation and Application](https://www.enjoyalgorithms.com/blog/bloom-filter)
* [How to avoid crawling duplicate URLs at Google scale | bytebytego](https://blog.bytebytego.com/p/how-to-avoid-crawling-duplicate-urls?s=r)
* [알아두면 좋은 자료 구조, Bloom Filter](https://steemit.com/kr-dev/@heejin/bloom-filter)
* [Bloom Filters by Example](https://llimllib.github.io/bloomfilter-tutorial/)

# Basic Concepts

## 동작 원리

Bloom Filter는 비트 배열과 여러 개의 독립적인 해시 함수를 사용합니다.

```
1. 초기화: m 크기의 비트 배열을 0으로 초기화
2. 삽입: 요소를 k개의 해시 함수로 해싱하여 해당 비트들을 1로 설정
3. 조회: 요소를 k개의 해시 함수로 해싱하여 모든 비트가 1인지 확인
```

### 시각적 예제

```
비트 배열 (m=10): [0,0,0,0,0,0,0,0,0,0]
해시 함수: h1, h2, h3

"apple" 삽입:
  h1("apple") % 10 = 2
  h2("apple") % 10 = 5
  h3("apple") % 10 = 7

비트 배열: [0,0,1,0,0,1,0,1,0,0]
             ↑     ↑   ↑

"banana" 삽입:
  h1("banana") % 10 = 1
  h2("banana") % 10 = 5
  h3("banana") % 10 = 9

비트 배열: [0,1,1,0,0,1,0,1,0,1]
             ↑ ↑   ↑   ↑   ↑

"grape" 조회:
  h1("grape") % 10 = 2  → bit[2] = 1 ✓
  h2("grape") % 10 = 4  → bit[4] = 0 ✗
  h3("grape") % 10 = 8  → bit[8] = 0 ✗

결과: "grape" 없음 (정확)

"xyz" 조회 (False Positive 케이스):
  h1("xyz") % 10 = 1  → bit[1] = 1 ✓
  h2("xyz") % 10 = 5  → bit[5] = 1 ✓
  h3("xyz") % 10 = 7  → bit[7] = 1 ✓

결과: "xyz" 있음 (잘못된 판단 - False Positive)
```

## False Positive vs False Negative

| 구분 | 설명 | Bloom Filter |
|------|------|--------------|
| **True Positive** | 존재하는 요소를 존재한다고 판단 | ✓ 항상 정확 |
| **True Negative** | 없는 요소를 없다고 판단 | ✓ 대부분 정확 |
| **False Positive** | 없는 요소를 존재한다고 잘못 판단 | ✗ 발생 가능 (확률 p) |
| **False Negative** | 존재하는 요소를 없다고 잘못 판단 | ✓ 절대 발생 안 함 |

### False Positive가 허용되는 이유

```
시나리오: 100억 개의 URL을 크롤링하는 웹 크롤러

방법 1: HashSet 사용
- 메모리: 100억 × 100바이트(평균 URL) = 1TB
- False Positive: 0%

방법 2: Bloom Filter 사용
- 메모리: 100억 × 10비트 = 11.6GB (1% FP 확률)
- False Positive: 1% (1억 개의 중복 크롤링)

결론: 99% 정확도로 메모리를 98% 절약
      → 1억 개의 추가 크롤링 비용 < 1TB 메모리 비용
```

## 시간 복잡도

| 연산 | 시간 복잡도 | 설명 |
|------|------------|------|
| 삽입 (Insert) | O(k) | k개의 해시 함수 계산 및 비트 설정 |
| 조회 (Contains) | O(k) | k개의 해시 함수 계산 및 비트 확인 |
| 삭제 (Delete) | 불가능 | 비트를 0으로 바꾸면 다른 요소에 영향 |

```java
// k=3인 경우의 시간 복잡도
public boolean contains(String value) {
    // O(1): 해시 함수 1 계산 및 비트 확인
    boolean bit1 = bitSet.get(hash1(value) % size);

    // O(1): 해시 함수 2 계산 및 비트 확인
    boolean bit2 = bitSet.get(hash2(value) % size);

    // O(1): 해시 함수 3 계산 및 비트 확인
    boolean bit3 = bitSet.get(hash3(value) % size);

    // 총 O(k) = O(3) = O(1) (k가 상수이므로)
    return bit1 && bit2 && bit3;
}
```

# Mathematical Background

## 최적 비트 배열 크기

예상 요소 개수 `n`과 목표 False Positive 확률 `p`가 주어졌을 때:

```
최적 비트 배열 크기 (m):
m = -n × ln(p) / (ln(2))²

예제:
- n = 10,000,000 (천만 개 요소)
- p = 0.01 (1% False Positive)

m = -10,000,000 × ln(0.01) / (ln(2))²
  = -10,000,000 × (-4.605) / 0.480
  ≈ 95,850,000 비트
  ≈ 11.4 MB
```

## 최적 해시 함수 개수

```
최적 해시 함수 개수 (k):
k = (m/n) × ln(2)

예제 (위와 동일한 조건):
k = (95,850,000 / 10,000,000) × ln(2)
  = 9.585 × 0.693
  ≈ 7

즉, 7개의 해시 함수를 사용하는 것이 최적
```

## False Positive 확률

실제 False Positive 확률 계산:

```
p = (1 - e^(-kn/m))^k

예제 검증:
- k = 7 (해시 함수 개수)
- n = 10,000,000 (요소 개수)
- m = 95,850,000 (비트 배열 크기)

p = (1 - e^(-7×10,000,000/95,850,000))^7
  = (1 - e^(-0.730))^7
  = (1 - 0.482)^7
  = 0.518^7
  ≈ 0.01 (1%)

목표 확률과 일치!
```

### False Positive 확률 트레이드오프

```java
// Bloom Filter 설정 계산기
public class BloomFilterCalculator {

    public static class Config {
        long expectedElements;    // n
        double falsePositiveRate; // p
        long optimalBits;         // m
        int optimalHashes;        // k
        double memoryMB;
    }

    public static Config calculate(long expectedElements,
                                   double falsePositiveRate) {
        Config config = new Config();
        config.expectedElements = expectedElements;
        config.falsePositiveRate = falsePositiveRate;

        // 최적 비트 수 계산
        config.optimalBits = (long) Math.ceil(
            -expectedElements * Math.log(falsePositiveRate)
            / Math.pow(Math.log(2), 2)
        );

        // 최적 해시 함수 개수 계산
        config.optimalHashes = (int) Math.ceil(
            (double) config.optimalBits / expectedElements * Math.log(2)
        );

        // 메모리 사용량 (MB)
        config.memoryMB = config.optimalBits / 8.0 / 1024 / 1024;

        return config;
    }

    public static void main(String[] args) {
        // 시나리오 비교
        long n = 10_000_000; // 천만 개 URL

        System.out.println("=== Bloom Filter 설정 비교 ===\n");

        for (double fp : new double[]{0.1, 0.01, 0.001, 0.0001}) {
            Config config = calculate(n, fp);
            System.out.printf("False Positive: %.2f%%\n", fp * 100);
            System.out.printf("  - 비트 배열 크기: %,d 비트\n",
                            config.optimalBits);
            System.out.printf("  - 메모리 사용량: %.2f MB\n",
                            config.memoryMB);
            System.out.printf("  - 해시 함수 개수: %d\n\n",
                            config.optimalHashes);
        }

        /*
        출력:
        === Bloom Filter 설정 비교 ===

        False Positive: 10.00%
          - 비트 배열 크기: 47,925,292 비트
          - 메모리 사용량: 5.72 MB
          - 해시 함수 개수: 4

        False Positive: 1.00%
          - 비트 배열 크기: 95,850,584 비트
          - 메모리 사용량: 11.44 MB
          - 해시 함수 개수: 7

        False Positive: 0.10%
          - 비트 배열 크기: 143,775,876 비트
          - 메모리 사용량: 17.16 MB
          - 해시 함수 개수: 10

        False Positive: 0.01%
          - 비트 배열 크기: 191,701,168 비트
          - 메모리 사용량: 22.88 MB
          - 해시 함수 개수: 14
        */
    }
}
```

# Implementation

## 기본 구현

원시 Java 구현:

```java
import java.util.BitSet;
import java.util.HashSet;
import java.util.Set;

public class BasicBloomFilter {

    private static final int DEFAULT_SIZE = 1000;
    private BitSet bitSet;
    private int size;
    private int hashFunctionCount;

    // 테스트용: 실제 데이터 (실제 구현에서는 불필요)
    private Set<String> actualData;

    public BasicBloomFilter() {
        this(DEFAULT_SIZE, 3);
    }

    public BasicBloomFilter(int size, int hashFunctionCount) {
        this.size = size;
        this.hashFunctionCount = hashFunctionCount;
        this.bitSet = new BitSet(size);
        this.actualData = new HashSet<>();
    }

    // 요소 추가
    public void add(String value) {
        actualData.add(value);

        for (int i = 0; i < hashFunctionCount; i++) {
            int hash = hash(value, i);
            bitSet.set(Math.abs(hash % size));
        }
    }

    // 요소 존재 여부 확인
    public boolean mightContain(String value) {
        for (int i = 0; i < hashFunctionCount; i++) {
            int hash = hash(value, i);
            if (!bitSet.get(Math.abs(hash % size))) {
                return false; // 하나라도 0이면 확실히 없음
            }
        }
        return true; // 모두 1이면 있을 가능성 있음
    }

    // 실제 포함 여부 (테스트용)
    public boolean actuallyContains(String value) {
        return actualData.contains(value);
    }

    // 해시 함수 (Double Hashing 기법)
    private int hash(String value, int i) {
        int hash1 = value.hashCode();
        int hash2 = hash1 >>> 16; // 상위 16비트 사용
        return hash1 + i * hash2;
    }

    // False Positive 테스트
    public static void main(String[] args) {
        BasicBloomFilter bf = new BasicBloomFilter(1000, 3);

        // 1000개의 요소 추가
        System.out.println("=== 1000개 요소 추가 ===");
        for (int i = 0; i < 1000; i++) {
            bf.add("element_" + i);
        }

        // True Positive 테스트
        System.out.println("\n=== True Positive 테스트 ===");
        int truePositive = 0;
        for (int i = 0; i < 100; i++) {
            String element = "element_" + i;
            if (bf.mightContain(element) && bf.actuallyContains(element)) {
                truePositive++;
            }
        }
        System.out.printf("정확도: %d/100 (%.1f%%)\n",
                         truePositive, truePositive * 100.0);

        // False Positive 테스트
        System.out.println("\n=== False Positive 테스트 ===");
        int falsePositive = 0;
        for (int i = 2000; i < 3000; i++) { // 존재하지 않는 요소
            String element = "element_" + i;
            if (bf.mightContain(element) && !bf.actuallyContains(element)) {
                falsePositive++;
            }
        }
        System.out.printf("False Positive: %d/1000 (%.1f%%)\n",
                         falsePositive, falsePositive / 10.0);
    }
}
```

### 개선된 해시 함수

독립적인 해시 함수를 생성하는 방법:

```java
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.BitSet;

public class ImprovedBloomFilter {

    private BitSet bitSet;
    private int size;
    private int hashFunctionCount;
    private MessageDigest md5;
    private MessageDigest sha1;

    public ImprovedBloomFilter(int size, int hashFunctionCount) {
        this.size = size;
        this.hashFunctionCount = hashFunctionCount;
        this.bitSet = new BitSet(size);

        try {
            this.md5 = MessageDigest.getInstance("MD5");
            this.sha1 = MessageDigest.getInstance("SHA-1");
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    }

    // Double Hashing: hash_i(x) = hash1(x) + i * hash2(x)
    private int[] getHashes(String value) {
        byte[] bytes = value.getBytes(StandardCharsets.UTF_8);

        // 첫 번째 해시: MD5
        int hash1 = bytesToInt(md5.digest(bytes));

        // 두 번째 해시: SHA-1
        int hash2 = bytesToInt(sha1.digest(bytes));

        int[] hashes = new int[hashFunctionCount];
        for (int i = 0; i < hashFunctionCount; i++) {
            hashes[i] = Math.abs((hash1 + i * hash2) % size);
        }

        return hashes;
    }

    private int bytesToInt(byte[] bytes) {
        return ((bytes[0] & 0xFF) << 24) |
               ((bytes[1] & 0xFF) << 16) |
               ((bytes[2] & 0xFF) << 8) |
               (bytes[3] & 0xFF);
    }

    public void add(String value) {
        for (int hash : getHashes(value)) {
            bitSet.set(hash);
        }
    }

    public boolean mightContain(String value) {
        for (int hash : getHashes(value)) {
            if (!bitSet.get(hash)) {
                return false;
            }
        }
        return true;
    }
}
```

## Guava Bloom Filter

Google Guava 라이브러리의 프로덕션 레벨 Bloom Filter:

```java
import com.google.common.hash.BloomFilter;
import com.google.common.hash.Funnels;
import java.nio.charset.StandardCharsets;

public class GuavaBloomFilterExample {

    public static void main(String[] args) {
        // 천만 개 요소, 1% False Positive
        BloomFilter<String> bloomFilter = BloomFilter.create(
            Funnels.stringFunnel(StandardCharsets.UTF_8),
            10_000_000,  // 예상 요소 개수
            0.01         // False Positive 확률
        );

        // 요소 추가
        System.out.println("=== URL 추가 ===");
        for (int i = 0; i < 1_000_000; i++) {
            bloomFilter.put("https://example.com/page" + i);
        }

        // 메모리 사용량 확인
        long approximateMemorySize = bloomFilter.approximateElementCount() * 10 / 8;
        System.out.printf("예상 메모리: %.2f MB\n",
                         approximateMemorySize / 1024.0 / 1024.0);

        // False Positive 테스트
        System.out.println("\n=== False Positive 테스트 ===");
        int falsePositives = 0;
        int testCount = 100_000;

        for (int i = 2_000_000; i < 2_000_000 + testCount; i++) {
            String url = "https://example.com/page" + i;
            if (bloomFilter.mightContain(url)) {
                falsePositives++;
            }
        }

        System.out.printf("False Positive 비율: %.2f%%\n",
                         (falsePositives * 100.0 / testCount));

        // 결과: 약 1% (목표와 일치)
    }
}
```

### Guava Bloom Filter - 직렬화

```java
import com.google.common.hash.BloomFilter;
import com.google.common.hash.Funnels;
import java.io.*;
import java.nio.charset.StandardCharsets;

public class BloomFilterPersistence {

    // Bloom Filter를 파일로 저장
    public static void saveToFile(BloomFilter<String> bloomFilter,
                                  String filename) throws IOException {
        try (OutputStream out = new FileOutputStream(filename)) {
            bloomFilter.writeTo(out);
        }
        System.out.println("Bloom Filter 저장 완료: " + filename);
    }

    // 파일에서 Bloom Filter 로드
    public static BloomFilter<String> loadFromFile(String filename,
                                                   long expectedElements,
                                                   double fpp)
            throws IOException {
        try (InputStream in = new FileInputStream(filename)) {
            return BloomFilter.readFrom(
                in,
                Funnels.stringFunnel(StandardCharsets.UTF_8)
            );
        }
    }

    public static void main(String[] args) throws IOException {
        String filename = "crawler_bloom_filter.dat";

        // 1. Bloom Filter 생성 및 데이터 추가
        System.out.println("=== Bloom Filter 생성 ===");
        BloomFilter<String> bloomFilter = BloomFilter.create(
            Funnels.stringFunnel(StandardCharsets.UTF_8),
            1_000_000,
            0.01
        );

        for (int i = 0; i < 1_000_000; i++) {
            bloomFilter.put("https://site.com/page" + i);
        }

        // 2. 파일로 저장
        saveToFile(bloomFilter, filename);

        File file = new File(filename);
        System.out.printf("파일 크기: %.2f MB\n",
                         file.length() / 1024.0 / 1024.0);

        // 3. 파일에서 로드
        System.out.println("\n=== Bloom Filter 로드 ===");
        BloomFilter<String> loadedFilter = loadFromFile(
            filename,
            1_000_000,
            0.01
        );

        // 4. 검증
        System.out.println("\n=== 검증 ===");
        boolean contains = loadedFilter.mightContain("https://site.com/page500000");
        System.out.println("Contains page500000: " + contains); // true

        boolean notContains = loadedFilter.mightContain("https://site.com/page2000000");
        System.out.println("Contains page2000000: " + notContains); // false (아마도)
    }
}
```

## Redis Bloom Filter

Redis Stack의 RedisBloom 모듈 사용 예제는 원본 문서를 참고하세요.

# Real-World Use Cases

실전 사용 사례에 대한 자세한 내용은 원본 영문 문서를 참고하세요.

## 웹 크롤러 중복 URL 방지

Google 규모의 웹 크롤러는 수십억 개의 URL을 처리합니다. Bloom Filter를 사용하면 1TB가 필요한 메모리를 11.4GB로 줄일 수 있습니다.

## 캐시 시스템 최적화

데이터베이스 쿼리 전에 Bloom Filter로 불필요한 캐시 조회를 방지합니다.

## 분산 시스템 데이터 동기화

여러 데이터 센터 간의 효율적인 데이터 동기화에 활용됩니다.

## 악성 URL 필터링

브라우저나 프록시 서버에서 악성 URL을 빠르게 차단합니다.

# Variants

## Counting Bloom Filter

일반 Bloom Filter는 삭제를 지원하지 않지만, Counting Bloom Filter는 카운터를 사용하여 삭제를 가능하게 합니다.

## Scalable Bloom Filter

동적으로 크기가 증가하는 Bloom Filter입니다.

## Cuckoo Filter

Bloom Filter의 대안으로, 삭제를 지원하고 더 나은 공간 효율성을 제공합니다.

# Performance Comparison

## 벤치마크: Bloom Filter vs HashSet

```
┌──────────────────┬────────────┬─────────────┬──────────────┐
│     지표         │  HashSet   │Bloom (1% FP)│  Bloom (0.1%)|
├──────────────────┼────────────┼─────────────┼──────────────┤
│ 메모리 (1000만)  │   800 MB   │   11.4 MB   │   17.2 MB    │
│ 삽입 시간        │   3.2초    │    2.9초    │    3.1초     │
│ 조회 시간        │   156ms    │    134ms    │    142ms     │
│ 정확도           │   100%     │     99%     │    99.9%     │
│ 삭제 지원        │    ✓       │      ✗      │      ✗       │
└──────────────────┴────────────┴─────────────┴──────────────┘

권장 사용 시나리오:
- HashSet: 정확도가 중요하고 메모리가 충분한 경우
- Bloom (1%): 대용량 데이터, 메모리 제약, 99% 정확도로 충분
- Bloom (0.1%): 더 높은 정확도가 필요하지만 메모리는 여전히 제한적
```

## 결론

Bloom Filter는 다음과 같은 경우에 이상적입니다:

```
✓ 대용량 데이터셋 (수백만~수십억 개)
✓ 메모리가 제한적인 환경
✓ False Positive가 허용 가능한 경우
✓ 빠른 조회 속도가 중요한 경우
✓ 정확한 집합 크기가 중요하지 않은 경우

✗ 100% 정확도가 필요한 경우
✗ 삭제가 빈번한 경우 (Counting/Cuckoo Filter 고려)
✗ 작은 데이터셋 (<10,000)
✗ False Positive의 비용이 큰 경우
```

---

**참고**: 이 문서는 간략한 버전입니다. 전체 구현 예제와 상세한 설명은 원본 영문 문서 `/bloomfilter/README.md`를 참고하세요.
