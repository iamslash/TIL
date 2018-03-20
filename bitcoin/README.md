# Abstract

bitcoin에 대해 정리한다.

# Materials

* [Ever wonder how Bitcoin (and other cryptocurrencies) actually work?](https://www.youtube.com/watch?v=bBC-nXj3Ng4)
  * bitcoin internals 30분 요약 
* [Bitcoin: A Peer-to-Peer Electronic Cash System - Satoshi Nakamoto](https://bitcoin.org/bitcoin.pdf)
* [Mastering Bitcoin](https://github.com/bitcoinbook/bitcoinbook/blob/develop/book.asciidoc)
  * 최고의 책
* [bitcoin guide](https://bitcoin.org/en/developer-guide#p2p-network)
  * 최고의 가이드 메뉴얼
* [gitcoin reference](https://bitcoin.org/en/developer-reference)
  * 최고의 레퍼런스 메뉴얼
  
# Fundamentals

## Digital Signatures

A 유저는 개인키(secret key)와 공개키(public key)를 생성한다. A 유저는
자신이 거래한 내역을 장부에 적을 때 거래의 내용(message)과 함께
서명(signature)을 기재한다. 서명은 다음과 같이 거래의 내용과 개인키를
이용하여 생성한다.

```
Sign(message, secret key) = signature
```

A 유저의 거래 내역이 참인지 거짓인지 검증할 때는 다음과 같이 거래의
내역(message), 서명(signature), 공개키(public key)를 이용해서 결과가
True인지 확인한다.

```
Verify(message, signature, public key) = True / False
```

## Peer Discovery

bitcoin node들의 리스트는 DNS seed에게서 얻는다. DNS seed는 bitcoin
core에 hard code되어 있다.

[CConnman::ThreadDNSAddressSeed](https://github.com/bitcoin/bitcoin/blob/master/src/net.cpp#L1592)

```cpp
void CConnman::ThreadDNSAddressSeed()
{
...
}
```

## Consensus algorithm

길이가 제일 긴 블록체인을 따라 블록을 추가하는 것은 모든 노드들의 51%
동의를 얻는 것과 같은 의미 일까?

## Simplified payment verification

커피숍에서 커피를 한잔 먹고 BTC로 결제하면 그 결제 트랜잭션은 어떤 방법으로
보장될까?

## vulnerability

### 51% attack

51%에 해당하는 노드들이 합의 한다면???


# References

* [block explorer](https://blockexplorer.com/)
  * bitcoin의 block을 구경할 수 있다.
* [bitcoin reference implementation](https://github.com/bitcoin/bitcoin)
  * bitcoin 구현체
