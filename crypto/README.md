- [Abstract](#abstract)
- [Materials](#materials)
- [대칭키 암호화 (Symmetric-key algorithm)](#대칭키-암호화-symmetric-key-algorithm)
- [비대칭키 암호화 (Asymmetric-key algorithm)](#비대칭키-암호화-asymmetric-key-algorithm)
- [RSA](#rsa)
- [How to save password safely](#how-to-save-password-safely)

----

# Abstract

암호화에 대해 적는다.

# Materials

* [RSA 인증서 (Certification) 와 전자서명 (Digital Sign)의 원리](https://rsec.kr/?p=426)
* [HTTPS와 SSL 인증서](https://opentutorials.org/course/228/4894)
* [SSL이란 무엇이며 인증서(Certificate)란 무엇인가?](https://wiki.kldp.org/HOWTO/html/SSL-Certificates-HOWTO/x70.html)
* [[Windows] 디지털서명의 구조와 원리](https://code13.tistory.com/165)
* [네이버 애플리케이션의 전자 서명 원리](https://d2.naver.com/helloworld/744920)
* [안전한 패스워드 저장](https://d2.naver.com/helloworld/318732)

# 대칭키 암호화 (Symmetric-key algorithm)

암호화와 복호화에 같은 암호키를 쓰는 방식이다. 

다음과 같은 알고리즘들이 있다.

* DES
* AES (Advanced Encryption Standard)
* ARIA
* Twofish
* SEED

# 비대칭키 암호화 (Asymmetric-key algorithm)

개인키, 공개키 한쌍을 만들어서 개인키로 암호화 하고
공개키로 복호화 한다. 또한 공개키로 암호화 하고
개인키로 복호화 한다.

예를 들어 A 는 개인키 공개키를 한쌍 만들고 이것을 
prvA, pubA 라고 하자. B 는 개인키 공개키를 한쌍 만들고
이것을 prvB, pubB 라고 하자. A 는 B 에게
message 를 pubB 로 암호화 해서 보낸다. B 는
prvB 로 message 를 복호화할 수 있다. 또한
B 는 A 에게 message 를 보낼 때 pubA 로
암호화해서 보낸다. A 는 prvA 로 message 를
복호화할 수 있다.

개인키, 공개키를 사용하여 암호화 복호화 하는 것은 
CPU bound job 이다. 따라서 먼저 대칭키 암호화에 사용할
대칭키를 비대칭키 암호화의 대상으로 하고 이후 대칭키를
이용하여 대칭키 암호화를 한다.

다음과 같은 알고리즘들이 있다.

* RSA
* 타원 곡선 암호화

# RSA 

* [RSA 암호화 - ICPA(인천포스코고등학교) @ youtube](https://www.youtube.com/watch?v=kGUlfVpIfaQ)
* [RSA @ 나무위키](https://namu.wiki/w/RSA%20%EC%95%94%ED%98%B8%ED%99%94)
* [RSA암호의 원리를 이해해보자.](https://blog.naver.com/at3650/40207488914)
  
----

1977년 Ron Rivest, Adi Shamir, Leonard Adleman 세 사람이 개발했고 이들의 성을 따서 RSA 라고 이름을 지었다. 큰 정수의 소인수 분해 및 나머지 역연산이 어렵다는 점을 이용한다.

다음은 RSA 알고리즘이다.

* RSA 를 위해서는 public key `(e, n)`, private key `(d, n)` 이 필요하다. `e, d, n` 은 모두 정수이다.
  * 임의의 소수 `p` 를 구한다. `p` 를 2 씩 더하면서 소수 `q` 를 구한다.
  * `n = p * q` 이다.
  * `Φ(n) = (p-1)(q-1)` 이다.
  * `1 < e < Φ(n)` 이고 `Φ(n)` 과 서로소인 `e` 를 선택한다.
  * `(e * d) mod Φ(n) = 1` 인 `d` 를 구한다.
* [페르마의 소정리](/numbertheory/README.md) 에 의해 다음의 두식이 성립한다. `M` 은 평문 `C` 는 암호문이라고 했을 때 다음의 두식으로 암호화 및 복호화가 가능하다.
  * `C = M^e mod n`
  * `M = C^d mod n`

다음은 RSA 알고리즘의 예이다.

```
p    = 2
q    = 7
n    = 14
Φ(n) = 6
e    = 5
d    = 11 ((e * d) mod Φ(n) = 1)

M    = 3
C    = 5 (3^5 mod 14)
M    = 3 (5^11 mod 14)
```

그렇다면 공개키를 획득한 크래커가 private key 를 알아낼 수 있는지 살펴보자.

* 크래커는 public key `(e, n)` 을 알고 있다.
* `n = p * q` 이므로 `p, q` 를 구한다. 그러나 어려울 것이다. `n` 이 `10, 65` 와 같이 작은 수라면 소인수 분해가 쉽다. 그러나 `1921713123` 과 같이 큰 수라면 소인수 분해가 거의 불가능하다.
* `p, q` 를 구하면 `Φ(n) = (p-1)(q-1)` 이므로 `Φ(n)` 을 구할 수 있다.
* `(e * d) mod Φ(n) = 1` 인 `d` 를 구한다. 그러나 어려울 것이다. 예를 들어 `a mod 6 = 1` 인 `a` 를 구해보자. 무수히 많다. 따라서 `d` 를 구하는 것은 거의 불가능하다.

이와 같이 RSA 는 큰 정수의 소인수 분해 및 나머지 역연산이 어렵다는 점을 이용한다.

이제 RSA 는 어떻게 페르마의 소정리를 이용하는지 알아보자. 다음은 페르마의 소정리이다.

```
임의의 정수 a 와 N 에 대해 항상 a ^ φ(N) ≡ 1 (mod N) 이다.

φ(N) 은 오일러-phi 함수로, 1부터 N 까지의 수들 중에서 N 과 서로소인 수들의 개수이다.

따라서 두 소수 p, q 가 있고 N = p * q 라면 φ(N) = (p-1) * (q - 1) 이다.
```

다음은 페르마의 소정리를 이용하여 `M == C` 인 것을 증명한 것이다.

TODO

# How to save password safely

* [안전한 패스워드 저장](https://d2.naver.com/helloworld/318732)
* [How to store passwords safely in the database and how to validate a password?](https://blog.bytebytego.com/p/how-does-https-work-episode-6?s=r)

----

보통 단방향 해쉬함수를 사용한다. 그러나 단방향 해쉬함수는
인식가능성(Recognizability) 와 속도 (Speed) 의 문제점이 있다. 그래서 Salting 과
Key stretching 으로 문제를 해결할 수 있다.

Adaptive key derivation function은 다이제스트를 생성할 때 솔팅과 키 스트레칭을
반복한다.

adaptive key derivation function 중 주요한 key derivation function은 `PBKDF2,
bcrypt, scrypt` 등이 있다.

PBKDF2(Password-Based Key Derivation Function) 는 NIST(National Institute of Standards and Technology, 미국표준기술연구소) 에서 승인되었고 가장 많이 사용된다. 다음은 PBKDF2 의 prototype 이다.

```c
DIGEST = PBKDF2(PRF, Password, Salt, c, DLen)  
```

* `PRF`: 난수(예: HMAC)
* `Password`: 패스워드
* `Salt`: 암호학 솔트
* `c`: 원하는 iteration 반복 수
* `DLen`: 원하는 다이제스트 길이

결론적으로 PBKDF2-HMAC-SHA-256/SHA-512 을 사용하면 된다.
