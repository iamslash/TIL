# Abstract

WebSocket 은 HTTP 와 같은 Application protocol 이다. 양방향 통신이 가능하다.
HTTP 를 이용해서 WebSocket handshake 를 한다. 접속된 이후는 WebSocket protocol
로 양방향 통신한다.

# Materials

* [Websocket Connection Handsake Under The Hood](https://medium.com/easyread/websocket-connection-handsake-under-the-hood-560ab1ceaff5)

# Basic

## Overview

Client 는 WebSocket Server 에게 다음과 같은 HTTP packet 을 보낸다.

```c
GET /chat HTTP/1.1
Host: server.example.com
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==
Sec-WebSocket-Version: 13
```

WebSocket Server 는 다음과 같은 HTTP response 를 보낸다.

```c
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: HSmrc0sMlYUkAGmm5OPpG2HaGWk=
```

WebSocket Handshake 는 끝났다.

# Advanced

## Horizontal scaling with WebSocket

* [Horizontal scaling with WebSocket – tutorial](https://tsh.io/blog/how-to-scale-websocket/)

----

WebSocket 을 이용한 Scale out 방법은 Sticky Session 과 Pub/Sub 을 생각해 볼 수 있다.

다음의 Architecture 에서 Scale Out 을 고민해 보자.

```
ClientA - ALB |- serverA
              |- serverB
              |- serverC
```

ClientA 는 재접속 하더라도 `serverA` 에 접속하길 원한다면 **Sticky Session** 을 이용하자. HTTP Request Header 에 `serverA` 의 정보가 삽입될 것이다. ALB 는 `serverA` 에게 HTTP request 를 routing 해준다.

ClientA 가 serverA 를 통해 `roomA` 에 입장했다. ClientB 가 serverB 를 통해 `roomA` 에 입장했다. `roomA` 에 속하는 Client 들에게 message 를 보내고 싶다면 **Pub/Sub** 를 이용한다.

예를 들어 Message Broker 로 Redis 를 이용한다면 다음과 같은 Architecture 를 생각해 볼 수 있다. 

```
clientA -| ALB |- serverA -| Redis
clientB -|     |- serverB -|
               |- serverC -|
```
