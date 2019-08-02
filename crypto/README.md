# Abstract

암호화에 대해 적는다.

# Materials

* [RSA 인증서 (Certification) 와 전자서명 (Digital Sign)의 원리](https://rsec.kr/?p=426)
* [HTTPS와 SSL 인증서](https://opentutorials.org/course/228/4894)
* [SSL이란 무엇이며 인증서(Certificate)란 무엇인가?](https://wiki.kldp.org/HOWTO/html/SSL-Certificates-HOWTO/x70.html)
* [[Windows] 디지털서명의 구조와 원리](https://code13.tistory.com/165)
* [네이버 애플리케이션의 전자 서명 원리](https://d2.naver.com/helloworld/744920)

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

개인키, 공개키를 사용하여 암호화 복호화 하는 것은 
CPU bound job 이다. 따라서 먼저 대칭키 암호화에 사용할
대칭키를 비대칭키 암호화의 대상으로 하고 이후 대칭키를
이용하여 대칭키 암호화를 한다.

다음과 같은 알고리즘들이 있다.

* RSA
* 타원 곡선 암호화

# 인증서

인증서는 인증기관 (CA, Certificate Authority) 에서 발행한다.

다음은 인증서에 포함된 정보들이다.

* 인증서 소유자의 e-mail 주소
* 소유자의 이름
* 인증서의 용도
* 인증서 유효기간
* 발행 장소
* Distinguished Name (DN)
  * Common Name (CN)
  * 인증서 정보에 대해 서명한 사람의 디지털 ID
* Public Key
* 해쉬(Hash)

https://wiki.kldp.org/HOWTO/html/SSL-Certificates-HOWTO/x70.html

다음은 크롬브라우저에서 구글에 접속했을 때 사용한 인증서이다.

![](certificate_sample.png)

# Digital Signing

![](digital_signing.png)

다음은 데이터 B1 을 디지털 서명하는 과정이다. C1 은 디지털 서명된 데이터이다. C1 은 원래의 데이터 B1 과 서명 S 그리고 공개키 KeyD 로 구성된다.

```c
Hash(B1) => H1;                 // B1 을 해쉬한 값을 H1 에 저장한다. H1 을 지문이라 하자.
Encrypt(KeyE, H1) => S;         // H1 을 개인키 KeyE 로 암호화하여 서명 S 를 얻는다.  
C1 = {B1, S, KeyD}              // B1, S, 공개키 KeyD 를 묶어서 디지털 서명된 데이터 C1 생성한다.
```

다음은 디지털 서명된 데이터 C1 을 받아서 검증하는 절차이다.

```c
C1 => B1, S, KeyD;                            // C1 에서 B1, S, KeyD 를 추출한다.  
Decrypt(KeyD, S) => H1;                       // 서명 S 를 공개키 KeyD 로 복호화하여 지문 H1 을 얻는다.  
Hash(B1) => H1;                               // B1 을 해시 해쉬한 값 H1 을 얻는다.  
Because H1 == H1, Execute C1 => Very Good!    // 두 지문이 일치하는지 검증한다.
```

과연 `C1` 의 `S` 는 믿을만 한가?

예를 들어 크래커가 다음과 같이 `C1` 을 조작하면 `C2` 를
배포할 수 있다.

```c
C1 => B1, S, KeyD;                  // 해커는 C1을 획득하여 B1, S, KeyD로 분리  
B1 => B2                            // B1을 B2로 변조  
Hash(B2) => H2;                     // B2를 해시 함수에 입력하여 지문 H2를 얻음  
Encrypt(FKeyE, H2) => S2;           // 지문 H2를 개인키 FKeyE로 암호화하여 서명 S2를 얻음  
C2 = {B2, S2, FKeyD}                // B2, S2, 공개키 FKeyD를 묶어서 코드사인 바이너리 C2 생성  
C1 => C2                            // 해커는 네이버 배포 서버의 C1을 C2로 바꿔치기함  
```

다음은 `C2` 를 받아서 검증하는 절차이다.

```c
C2 => B2, S2, FKeyD;                           // 다운로드한 C2를 B2, S2, FKeyD로 분리  
Decrypt(FKeyD, S2) => H2;                      // 서명 S2를 공개키 FKeyD로 복호화하여 지문 H2을 얻음  
Hash(B2) => H2;                                // B2를 해시 함수에 입력하여 지문 H2를 얻음  
Because H2 == H2, Execute C2 => Oh, my god!    // 두 지문이 일치하는 것을 확인 
```

인증서를 이용하여 `C1 = {B1, S, KeyD}` 대신 `C1 = {B1, S, 인증서}` 형태로 C1 을 제작하면
위의 문제를 해결할 수 있다. `KeyD` 는 인증서에 포함된다.