- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
- [HTTP 1.1](#http-11)
- [HTTP 2](#http-2)
- [HTTP Flow](#http-flow)

----

# Abstract

HTTP 1.1 은 1999 년에 출시되었다. 하나의 TCP 연결에 하나의 Request, Response 를 처리한다. 속도와 성능 이슈를 가지고 있다. 따라서 HOL (Head of Line) Blocking - 특정 응답 지연, RTT (Round Trip Time) 증가, 무거운 Header 구조 (Big Cookies) 라는 문제점을 가지고 있었다. 또한 이런 문제들을 해결하기 위해 개발자들은 image sprite, domain sharding, CSS/JavaScript compression, Data URI 등을 이용하였다. 또한 google 은 SPDY 라는 프로토콜을 만들어서 HTTP 1.1 의 제약사항을 극복하려 했지만 HTTP 2 의 등장과 함께 Deprecate 되었다.

# Materials

* [HTTP/2 알아보기 - 1편 @ whatap](https://www.whatap.io/ko/blog/38/)
* [[초보개발자 일지] HTTP 프로토콜의 이해 — 1 (HTTP 정의, HTTP/1.1)](https://medium.com/@shaul1991/%EC%B4%88%EB%B3%B4%EA%B0%9C%EB%B0%9C%EC%9E%90-%EC%9D%BC%EC%A7%80-http-%ED%94%84%EB%A1%9C%ED%86%A0%EC%BD%9C%EC%9D%98-%EC%9D%B4%ED%95%B4-1-b9005a77e5fd)
* [Hypertext Transfer Protocol @ wikipedia](https://en.wikipedia.org/wiki/Hypertext_Transfer_Protocol)
* [HTTP1.1 vs HTTP2.0 차이점 간단히 살펴보기](https://medium.com/@shlee1353/http1-1-vs-http2-0-%EC%B0%A8%EC%9D%B4%EC%A0%90-%EA%B0%84%EB%8B%A8%ED%9E%88-%EC%82%B4%ED%8E%B4%EB%B3%B4%EA%B8%B0-5727b7499b78)
* [SPDY는 무엇인가?](https://d2.naver.com/helloworld/140351)
* [Hypertext Transfer Protocol Version 2 (HTTP/2) @ RFC](https://tools.ietf.org/html/rfc7540)
* [What is HTTP/2 – The Ultimate Guide](https://kinsta.com/learn/what-is-http2/)
* [HTTP 응답코드 메소드 정리 GET, POST, PUT, PATCH, DELETE, TRACE, OPTIONS](https://javaplant.tistory.com/18)

# Basic

History of HTTP 

| Year |	HTTP Version |
|--|--|
| 1991	| 0.9 |
| 1996	| 1.0 |
| 1997	| 1.1 |
| 2015	| 2.0 |
| Draft (2020)	| 3.0 |

# HTTP 1.1

The request message is consisted of the following

* Request line
* Request header fields
* Empty line
* Optional message body

```http
GET / HTTP/1.1
Host: www.example.com
```

This is request methods

----

| HTTP method | Description |
|--|--|
| GET | 읽어온다 |
| HEAD | GET방식과 동일하지만, 응답에 BODY가 없고 응답코드와 HEAD만 응답한다. |
| POST | 생성한다 |
| PUT | 업데이트한다 |
| PATCH | 부분 업데이트한다 |
| DELETE | 삭제한다 |
| CONNECT | 동적으로 터널 모드를 교환, 프락시 기능을 요청시 사용. |
| TRACE | 원격지 서버에 루프백 메시지 호출하기 위해 테스트용으로 사용. |
| OPTIONS | 웹서버에서 지원되는 메소드의 종류를 확인할 경우 사용. |

The reponse message is consisted of the following

* Status line which includes the status code and reason message (e.g., HTTP/1.1 200 OK)
* Response header fields (e.g., Content-Type: text/html)
* Empty line
* Optional message body

```http
HTTP/1.1 200 OK
Date: Mon, 23 May 2005 22:38:34 GMT
Content-Type: text/html; charset=UTF-8
Content-Length: 155
Last-Modified: Wed, 08 Jan 2003 23:11:55 GMT
Server: Apache/1.3.3.7 (Unix) (Red-Hat/Linux)
ETag: "3f80f-1b6-3e1cb03b"
Accept-Ranges: bytes
Connection: close

<html>
  <head>
    <title>An Example Page</title>
  </head>
  <body>
    <p>Hello World, this is a very simple HTML document.</p>
  </body>
</html>
```

These are status codes.

| Category | codes |
|--|--|
| Informational | 1XX |
| Successful | 2XX |
| Redirection | 3XX |
| Client Error | 4XX |
| Server Error | 5XX |

# HTTP 2

HTTP 2 는 속도와 성능이 개선되었다. 

| Feature | description|
|---|---|
| **Multiplexed Streams** | HTTP 1.1 은 한번에 하나의 파일밖에 전송을 못한다. 이것을 극복하기 위해 여러개의 TCP connection 을 맺어서 여러개의 파일을 전송했다. 그러나 HTTP 2 는 하나의 connection 으로 여러개의 Request, Response 를 전송한다. | 
| **Dedupe Headers** | 중복된 헤더를 제거한다. |
| **Header Compression** | Header 를 HPACK 으로 압축한다. |
| **Server Push** | Server 에서 JavaScript, CSS, Font, Image 등을 Client 으로 push 한다. |
| **Stream Prioritization** | 웹페이지를 구성하는 파일들의 우선순위를 표기할 수 있다. |

# HTTP Flow

* [What happens when... @ github](https://github.com/alex/what-happens-when)
  * [...하면 생기는 일 @ github](https://github.com/SantonyChoi/what-happens-when-KR)

----

**The "g" key is pressed**

**The "enter" key bottoms out**

**Interrupt fires [NOT for USB keyboards]**

* "g" 키를 누르면 인터럽트가 발생합니다. 그리고 kernel 이 인터럽트 핸들러를 호출합니다.

**(On Windows) A WM_KEYDOWN message is sent to the app**

* kernel 이 WM_KEYDOWN 메시지를 Browser application 으로 전달합니다.

**Parse URL**

**Is it a URL or a search term?**

* Browser application 은 URL 을 parsing 합니다. 그리고 protocol 혹은 valid domain name 이 아니면 default search engine 에게 HTTP Request 를 전송합니다.

**Convert non-ASCII Unicode characters in hostname**

* Browser application 은 URL 의 host name 에 `a-z, A-Z, 0-9, -, .` 아닌 문자열이 있는지 확인합니다. Unicode 가 있을 때는 [Punycode encoding](https://en.wikipedia.org/wiki/Punycode) 을 하기도 한다.

**Check HSTS list**

* Browser application 는 `HSTS (HTTP Strict Transport Security)` 에 URL 이 있는지 검사합니다. 있다면 HTTP 대신 HTTPS Request 를 해야합니다.

**DNS lookup**

* Browser application 은 Domain Cache 에 host name 이 있는지 검사합니다.
  * Chrome 의 경우 `chrome://net-internals/#dns` 에서 DNS cache 를 확인할 수 있다.
* DNS cache miss 가 발생하면 `gethostbyname` 을 호출합니다.
* `gethostbyname` 은 `/etc/hosts` 를 검색합니다.
* `/etc/hosts` 에 없다면 `gethostbyname` 은 DNS query 를 합니다.

**ARP process**

**Opening of a socket**

**TLS handshake**

**If a packet is dropped**

**HTTP protocol**

**HTTP Server Request Handle**

**Behind the scenes of the Browser**

**Browser**

**HTML parsing**

**CSS interpretation**

**Page Rendering**

**GPU Rendering**

**Post-rendering and user-induced execution**
