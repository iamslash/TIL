# Abstract

RTP 는 1996 년 1월에 [RFC 1889](https://datatracker.ietf.org/doc/html/rfc1889) 로 정의되었다. [Ron Frederick](https://github.com/ronf) 는 제작자중 한명이다. [nv | github](https://github.com/ronf/nv) 는 [Ron Frederick](https://github.com/ronf) 이 제작한 최초의 video conference tool 이다. 

# Materials

> [28장. RTP의 이해 ](https://brunch.co.kr/@linecard/154)

# Basic

## RTP Header

> [RTP Packet Format](https://www.cl.cam.ac.uk/~jac22/books/mm/book/node159.html)

```
    0                   1                   2                   3
    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |V=2|P|X|  CC   |M|     PT      |       sequence number         |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |                           timestamp                           |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |           synchronization source (SSRC) identifier            |
   +=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
   |            contributing source (CSRC) identifiers             |
   |                             ....                              |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

## RTP Example

음성을 RTP 로 전송한다고 해보자. Packet 은 Network Layer 별로 다음과 같이
구성된다.

| Ethernet |    IP | UDP | RTP | Voice |
|--|--|--|--|--|
| 18 byte | 20 byte | 8 byte | 12 byte | ??? byte | 

Payload 의 크기가 다르더라도 IP/UDP/RTP 는 40 byte 로 일정하다.

| Ethernet |    IP | UDP | RTP | G.711 10ms |
|--|--|--|--|--|
| 18 byte | 20 byte | 8 byte | 12 byte | 160 byte | 

| Ethernet |    IP | UDP | RTP | G.711 20ms |
|--|--|--|--|--|
| 18 byte | 20 byte | 8 byte | 12 byte | 320 byte | 

| Ethernet |    IP | UDP | RTP | G.729 10ms |
|--|--|--|--|--|
| 18 byte | 20 byte | 8 byte | 12 byte | 20 byte | 
