- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
  - [SDP Overview](#sdp-overview)

----

# Abstract

SDP 는 multimedia 를 주고받기 위한 session 을 기술하는 protocol 이다. 

The IETF published the original specification as a Proposed Standard in April 1998 (RFC2327). Revised specifications were released in 2006 (RFC 4566) and in 2021 (RFC 8866).

# Materials

* [Session Description Protocol @ wikipedia](https://en.wikipedia.org/wiki/Session_Description_Protocol)
* [WebRTC SDP](https://cryingnavi.github.io/webrtc/2016/12/30/WebRTC-SDP.html)

# Basic

## SDP Overview

* [Session Description Protocol @ wikipedia](https://en.wikipedia.org/wiki/Session_Description_Protocol)

----

다음은 SDP 의 Format 이다. multiline text 로 만들어 진다. 각 줄의 형식은 `<character>=<value><CR><LF>` 과 같다. 

time description 은 한개 이상일 수 있다. media description 은 0 개 이상일 수 있다. 

```
    v=  (protocol version number, currently only 0)
    o=  (originator and session identifier : username, id, version number, network address)
    s=  (session name : mandatory with at least one UTF-8-encoded character)
    i=* (session title or short information)
    u=* (URI of description)
    e=* (zero or more email address with optional name of contacts)
    p=* (zero or more phone number with optional name of contacts)
    c=* (connection information—not required if included in all media)
    b=* (zero or more bandwidth information lines)
    One or more time descriptions ("t=" and "r=" lines; see below)
    z=* (time zone adjustments)
    k=* (encryption key)
    a=* (zero or more session attribute lines)
    Zero or more Media descriptions (each one starting by an "m=" line; see below)
```

다음은 time description 의 format 이다.

```
    t=  (time the session is active)
    r=* (zero or more repeat times)
```

다음은 media description 의 format 이다. media 별로 address 가 포함될 수도 있다.

```
    m=  (media name and transport address)
    i=* (media title or information field)
    c=* (connection information — optional if included at session level)
    b=* (zero or more bandwidth information lines)
    k=* (encryption key)
    a=* (zero or more media attribute lines — overriding the Session attribute lines)
```

다음은 SDP 의 예이다.

```
    v=0
    o=jdoe 2890844526 2890842807 IN IP4 10.47.16.5
    s=SDP Seminar
    i=A Seminar on the session description protocol
    u=http://www.example.com/seminars/sdp.pdf
    e=j.doe@example.com (Jane Doe)
    c=IN IP4 224.2.17.12/127
    t=2873397496 2873404696
    a=recvonly
    m=audio 49170 RTP/AVP 0
    m=video 51372 RTP/AVP 99
    a=rtpmap:99 h263-1998/90000
```

* jode 가 IPv4 10.47.16.5 에서 session 을 만들었다.
* session 의 이름은 "SDP Seminar" 이다.
* session 의 추가 정보는 "A Seminar on the session description protocol" 이다.
* session 의 추가 정보 link 는 "http://www.example.com/seminars/sdp.pdf" 이다.
* session 의 책임자 이메일은 "j.doe@example.com" 이다.
* clients 가 connect 해야할 주소는 IPv4 224.2.17.12 이고 TTL 은 127 이다. 만약 address 가 multicast address 이라면 그 주소는 clients 가 subscribe 해야할 주소이다. connect 와 subscribe 의 차이는???
* session 의 유효시간은 2 시간 이다. (2873404696 - 2873397496 = 7200)
* 이 session 의 수신자들은 audio, video data 를 수신하기만 한다. 송신은 하지 않는다.
* 첫번째 media 는 audio 이다. 49170 port 에서 RTP/AVP payload type 0 (defined by RFC 3551 as PCMU) 를 이용하여 만들어 진다.
* 두번째 emdia 는 video 이다. 51372 port 에서 RTP/AVP payload type 99 (defined as "dynamic") 을 이용하여 만들어 진다.
* RTP/AVP payload type 99 는 h263-1998 (90KHz) 로 mapping 된다.
* RTCP ports 는 49171, 51373 이다. 49170, 51372 에서 하나씩 더한 숫자이다.
