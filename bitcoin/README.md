<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Abstract](#abstract)
- [References](#references)
- [Materials](#materials)
- [Fundamentals](#fundamentals)
  - [Overview](#overview)
  - [Block Structure](#block-structure)
  - [Transactions](#transactions)
  - [UTXO (Unspent Transaction Outputs)](#utxo-unspent-transaction-outputs)
  - [Digital Signatures](#digital-signatures)
  - [Peer Discovery](#peer-discovery)
  - [POW (Proof of Work)](#pow-proof-of-work)
  - [Consensus algorithm](#consensus-algorithm)
  - [Simplified payment verification](#simplified-payment-verification)
  - [vulnerability](#vulnerability)
    - [51% attack](#51-attack)
- [Code Tour](#code-tour)

<!-- markdown-toc end -->

-------------------------------------------------------------------------------

# Abstract

bitcoin에 대해 정리한다.

# References

* [How does bitcoin work (internals)](https://medium.com/ascentic-technology/how-does-bitcoin-work-internals-c2450793a0db)
  * bitcoin 의 under the hood 를 잘 설명한 글
* [block explorer](https://blockexplorer.com/)
  * bitcoin의 block을 구경할 수 있다.
* [bitcoin reference implementation](https://github.com/bitcoin/bitcoin)
  * bitcoin 구현체

# Materials

* [awsome blockchain @ github](https://github.com/yunho0130/awesome-blockchain-kor/tree/master/whitepapers)
  * 모두를 위한 블록체인
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

## Overview

Block Chain 은 Block 들이 서로 연결된 Chain 을 말한다. 최초 Block 은 Genesis Block 이라고 한다. Block 은 Header 가 있고 다음과 같은 것들을 포함한다. [Block Chain @ bitcoin](https://developer.bitcoin.org/reference/block_chain.html)

| Bytes | Name | Data Type |
|-------:|:---|:---|
| 4 | version | `int32_t` |
| 32 | prev block header hash | `char[32]` |
| 32 | merkle root hash | `char[32]` |
| 4 | time | `uint32_t` |
| 4 | nBits | `uint32_t` |
| 4 | nonce | `uint32_t` |

[Block 540617](https://www.blockchain.com/btc/block/0000000000000000000ebb857fd5389cb27b963c6d4e7852efb5a11ceac62ee8) 는 Height 가 `540617` 인 Block 의 내용이다. Height 는 Genesis Block 으로 부터의 거리이다. 

Miner 는 Mining Application 을 다운로드하여 실행하는 사람을 말한다. Minig Application 은 지금까지 만들어진 Block Chain 에 새로운 Block 을 생성하여 연결하려는 노력을 한다. 새롭게 만들어진 Block 이 다른 Miner 들에 의해 인정되면 bitcoin 으로 보상을 받는다. 보통 Block 은 10 분마다 하나씩 생성된다고 한다.

A 유저가 B 유저에게 송금하면 Transaction 이 하나 발생한다. 그리고 그 것은 unconfirmed transaction pool 에 보내진다. 이것을 [mem pool](https://www.blockchain.com/btc/unconfirmed-transactions) 이라고도 한다. Mining Application 은 [mem pool](https://www.blockchain.com/btc/unconfirmed-transactions) 에서 Transactions 을 가져와 새로운 Block 제작을 시도한다. 

새로운 Block 을 생성하려면 Block 의 Nonce 를 제대로 생성해야 한다. Block Header 의 Hash 가 nBits 보다 작거나 같아야 한다. Block Header Hash 는 Block 의 Header 를 기반으로 생성한다. 이때 Block Header 에서 nonce 를 제외하고 다른 값들은 모두 고정이다. 따라서 nonce 를 바꿔가면서 제대로 된 Block Header Hash 를 생성해야 한다. 이것은 CPU Bound Job 이고 전기를 많이 소모한다.

nBits 는 어떻게 정해지는 거지???

## Block Structure

WIP...

## Transactions

WIP...

## UTXO (Unspent Transaction Outputs)

WIP...

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

## POW (Proof of Work)

제대로된 nonce 를 생성하는 과정이다. 즉, Block Header Hash 가 nBits 보다 작거나 같을 때의 nonce 를 찾는 일을 말한다. CPU bound job 이다.

## Consensus algorithm

길이가 제일 긴 블록체인을 따라 새로운 블록을 추가하는 것은 모든
노드들의 51% 동의를 얻는 것과 같은 의미 일까?

## Simplified payment verification

커피숍에서 커피를 한잔 먹고 BTC로 결제하면 그 결제 트랜잭션은 어떤 방법으로
보장될까?

## vulnerability

### 51% attack

51%에 해당하는 노드들이 합의 한다면???

# Code Tour

* [bitcoin code tour](bitcoin_codetour.md)
