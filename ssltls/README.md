- [Abstract](#abstract)
- [Materials](#materials)
- [SSL Handshake](#ssl-handshake)
- [TCP Handshake VS SSL Handshake](#tcp-handshake-vs-ssl-handshake)
- [SSL vs TLS](#ssl-vs-tls)
- [mTLS (Mutual TLS)](#mtls-mutual-tls)

----

# Abstract

SSL/TLS 에 대해 적는다.

# Materials

> * [SSL (1) - SSL에 대한 소개](https://m.blog.naver.com/nttkak/20130246203)
> * [SSL (2) - SSL 프로토콜 알기 @ naverblog](https://m.blog.naver.com/nttkak/20130246501)
> * [SSL (3) - 기본 적인 SSL 통신 하기](https://m.blog.naver.com/nttkak/20130246586)
> * [SSL (4) - 상대방 인증 하면서 SSL 통신 하기](https://m.blog.naver.com/nttkak/20130246706)

# SSL Handshake

> * [K15292: Troubleshooting SSL/TLS handshake failures](https://support.f5.com/csp/article/K15292)
> * [Breaking Down the TLS Handshake](https://www.youtube.com/watch?v=cuR05y_2Gxc)

----

![](img/SSL_flow2_10_08_18_updated.png)

# TCP Handshake VS SSL Handshake

* [Nuts and Bolts of Transport Layer Security (TLS)](https://medium.facilelogin.com/nuts-and-bolts-of-transport-layer-security-tls-2c5af298c4be)

# SSL vs TLS

> * [SSL vs. TLS – What are differences?](https://www.ssl2buy.com/wiki/ssl-vs-tls)
> * [SSL vs. TLS - 차이점은 무엇인가?](https://smartits.tistory.com/209)

SSL refers to Secure Sockets Layer whereas TLS refers to Transport Layer Security.  Basically, they are one and the same, but, entirely different.

How similar both are? SSL and TLS are cryptographic protocols that authenticate data transfer between servers, systems, applications and users. For example, a cryptographic protocol encrypts the data that is exchanged between a web server and a user.

SSL was a first of its kind of cryptographic protocol. TLS on the other hand, was a recent upgraded version of SSL.

# mTLS (Mutual TLS)

> * [Mutual TLS](https://www.jacobbaek.com/1040)

TLS 는 Server 의 Ceritificate 를 Client 에게 보내준다. Client 는 제대로된 Server
인지 검증한다. Server 는 Client 가 제대로된 Client 인지 Application Layer 에서
검증한다. 즉, code 로 검증한다.

mTLS 는 Server 의 Certificate 를 Client 에게 보내주고 Client 의 Certificate 를
Server 에게 보내준다. 제대로 된 Server 인지, 제대로 된 Client 인지 Certificate 를
가지고 검증한다.
