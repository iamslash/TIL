# Abstract

WebSocket 은 HTTP 와 같은 Application protocol 이다. 양방향 통신이 가능하다.
HTTP 를 이용해서 WebSocket handshake 를 한다. 접속된 이후는 WebSocket protocol
로 양방향 통신한다.

# Materials

* [Websocket Connection Handsake Under The Hood](https://medium.com/easyread/websocket-connection-handsake-under-the-hood-560ab1ceaff5)

# Overview

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
