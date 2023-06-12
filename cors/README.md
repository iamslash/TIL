# Materials

* [CORS, 기본 동작 원리와 이슈 해결 방법 | velog](https://velog.io/@nemo/cors)

# Abstract

Client 의 XMLHttpRequest 가 cross-domain 을 요청할 수 있도록하는 방법이다.
request 를 수신하는 Web Server 에서 설정해야 한다.

Client 는 `Origin` 을 HTTP Request Header 에 담아 보낸다.

Server 는 `Access-Control-Allow-Origin` 를 HTTP Response Header 에 담아 보낸다.
