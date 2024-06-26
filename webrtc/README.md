- [Abstract](#abstract)
- [References](#references)
- [Meterials](#meterials)
- [Basic](#basic)
  - [Terms](#terms)
  - [WebRTc Overview](#webrtc-overview)
  - [WebRTC Example](#webrtc-example)
  - [WebRTC Sequence](#webrtc-sequence)
  - [Signaling](#signaling)
  - [Connecting](#connecting)
  - [Securing](#securing)
  - [Media Communication](#media-communication)
  - [Data Communication](#data-communication)
  - [WebRTC Topologies](#webrtc-topologies)
    - [Mesh](#mesh)
    - [SFU (Selective Forwarding Unit)](#sfu-selective-forwarding-unit)
    - [MCU (Multi Control Unit)](#mcu-multi-control-unit)
  - [Debugging](#debugging)
- [Advanced](#advanced)
  - [WebRTC Implementations](#webrtc-implementations)

---

# Abstract

- WebRTC (web real-time communication)
- browser 에서도 skype 처럼 실시간으로 video, voice, data 들을 주고 받아 보자.
- browser 에서 화상 채팅등을 할 수 있다.

# References

* [webrtc | googlesource](https://webrtc.googlesource.com/src)
  * google 에서 운영하는 webrtc repo

# Meterials

- [WebRTC for the Curious](https://webrtcforthecurious.com/)
  - 종결자
- [WebRTC 이론 정리하기](https://millo-l.github.io/WebRTC-%EC%9D%B4%EB%A1%A0-%EC%A0%95%EB%A6%AC%ED%95%98%EA%B8%B0/)
  - [WebRTC 구현 방식(Mesh/P2P, SFU, MCU)](https://millo-l.github.io/WebRTC-%EA%B5%AC%ED%98%84-%EB%B0%A9%EC%8B%9D-Mesh-SFU-MCU/)
  - [WebRTC 구현하기(1:1 P2P)](https://millo-l.github.io/WebRTC-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0-1-1-P2P/)
    - [src](https://github.com/millo-L/Typescript-ReactJS-WebRTC-1-1-P2P)
  - [WebRTC 구현하기(1:N P2P)](https://millo-l.github.io/WebRTC-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0-1-N-P2P/)
    - [src](https://github.com/millo-L/Typescript-ReactJS-WebRTC-1-N-P2P)
  - [WebRTC 구현하기(1:N SFU)](https://millo-l.github.io/WebRTC-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0-1-N-SFU/)
    - [src](https://github.com/millo-L/Typescript-ReactJS-WebRTC-1-N-SFU)
  - [WebRTC 성능 비교(P2P vs SFU)](https://millo-l.github.io/WebRTC-%EC%84%B1%EB%8A%A5%EB%B9%84%EA%B5%90-P2P-vs-SFU/)
- [WebRTC: The secret power you didn't know Go has | Sean DuBois | Conf42: Golang 2021 @ youtube](https://www.youtube.com/watch?v=4kdU9_a-gII)
  - pion contributor 가 알려주는 WebRTC API from pion
- [Getting started with WebRTC](https://webrtc.org/getting-started/overview)
  - 킹왕짱
  - [samples](https://webrtc.github.io/samples/)
- [Examples WebRTC Applications @ github](https://github.com/pion/example-webrtc-applications)
  - go examples using pion webrtc
- [Guide to WebRTC @ baeldung](https://www.baeldung.com/webrtc)  
- [STUNTMAN](http://www.stunprotocol.org/)
  - c++로 제작된 STUN, TURN server, client
- [go-stun](https://github.com/pixelbender/go-stun)
  - golang으로 STUN, TURN, ICE protocol을 구현했다.
- [NAT Traversal의 종결자](http://www.nexpert.net/424)
  - WebRTC의 STUN, TUN, ICE를 이해할 수 있었다.

# Basic

## Terms

| Feature | Description|
|---|---|
| STUN | Session Traversal Utilities for NAT |
| TURN | Traversal Using Relay NAT |
| ICE (Internet Connectivity Establishment) candidate | WebRTC 로 연결할 peer 이다. |
| ICE Servers | Ice candidate 끼리 communication 이 가능하도록 도와주는 server 들. STUN, TURN 서버를 말한다. |
| Signaling | Ice candidates 및 SDP 를 교환하는 것. 전화로 따지면 상대의 번호를 누르고 기다리는 행위이다. |
| Trickle ICE | ICE candidate 가 발견될 때 마다 ICE candidate 을 보낸다. 수도꼭지에서 물 방울이 조금씩 흐르는 것처럼 ICE candidate 를 조금씩 흘려보낸다는 의미이다. |
| [SDP](/sdp/README.md) | Session Description Protocol |

## WebRTc Overview

> [A Study of WebRTC Security](https://webrtc-security.github.io/)

WebRTC 는 다음과 같이 여러기술들이 모여서 만들어 졌다.

| Category | Protocol |
|---|---|
| Signaling | [SDP](/sdp/README.md) |
| Connection | STUN, TURN, ICE |
| Security | DTLS, SRTP |
| Network | [RTP](/rtp/README.md), SRTP, RTCP |

* DTLS (Datagram Transport Layer Security)
* SRTP (Secure Real-time Transport Protocol)
* RTCP (RTP control Protocol)
* SCTP (Stream Control Transmission Protocol)
  * Reliable UDP

![](img/diagram_2_en.png)

## WebRTC Example

- [Real-time communication with WebRTC: Google I/O 2013](https://www.youtube.com/watch?v=p2HzZkd2A40) 을 열심히 보자.
  - webRTC에 대한 전반적인 설명과 함께 예제 코드가 좋다.
- [Real time communication with WebRTC](https://codelabs.developers.google.com/codelabs/webrtc-web/#0)
  - 10단계로 설명하는 WebRTC
- Three main JavaScript APIs.
  - MediaStream(getUserMedia), RTCPeerConnection, RTCDataChannel
- [ascii-camera](https://idevelop.ro/ascii-camera/)는 camera의 데이터를 ascii데이터로 렌더링해서 보여준다.
- MediaStream(getUserMedia)

----

Alice 와 Bob 은 ICE candidate 이다. Alice 가 Bob 에게 Signaling 하는 경우를 생각해 보자. 
Signaling 은 WebRTC Specification 에 포함되있지 않다. HTTP 를 이용해도 좋다.
Alice 는 다음과 같이 Bob 에게 "Hello!" 를 보낸다. Bob 은 Alice 로 부터 "Hello!"
를 받아서 처리한다. "Hello!" 는 Signaling 을 위한 Application Protocol 이다.

```js
// Set up an asynchronous communication channel that will be
// used during the peer connection setup
const signalingChannel = new SignalingChannel(remoteClientId);
signalingChannel.addEventListener('message', message => {
    // New message from remote client received
});

// Send an asynchronous message to the remote client
signalingChannel.send('Hello!');
```

Alice 는 Bob 에게 SDP 를 보낸다. 먼저 RTCPeerConnection object 를 만들고
이것을 pc 라고 하자. SDP 를 하나 만들어 offer 라고 하자. 이것을 Bob 에 보낸다. 
또한 `pc.setLocalDescription(offer)` 를 수행한다. Bob 은 offer 를 받아서
SDP 를 하나 만들고 이것을 answer 라고 하자. 그리고 Alice 에게 answer 를 보낸다.
Alice 는 `pc.setRemoteDescription(answer)` 를 수행한다.

다음은 offer 를 보내고 answer 를 받았을 때 처리하는 code 이다.

```js
async function makeCall() {
    const configuration = {'iceServers': [{'urls': 'stun:stun.l.google.com:19302'}]}
    const peerConnection = new RTCPeerConnection(configuration);
    signalingChannel.addEventListener('message', async message => {
        if (message.answer) {
            const remoteDesc = new RTCSessionDescription(message.answer);
            await peerConnection.setRemoteDescription(remoteDesc);
        }
    });
    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);
    signalingChannel.send({'offer': offer});
}
```

다음은 offer 를 받았을 때 처리하는 code 이다.

```js
const peerConnection = new RTCPeerConnection(configuration);
signalingChannel.addEventListener('message', async message => {
    if (message.offer) {
        peerConnection.setRemoteDescription(new RTCSessionDescription(message.offer));
        const answer = await peerConnection.createAnswer();
        await peerConnection.setLocalDescription(answer);
        signalingChannel.send({'answer': answer});
    }
});
```

```js
// Listen for local ICE candidates on the local RTCPeerConnection
peerConnection.addEventListener('icecandidate', event => {
    if (event.candidate) {
        signalingChannel.send({'new-ice-candidate': event.candidate});
    }
});

// Listen for remote ICE candidates and add them to the local RTCPeerConnection
signalingChannel.addEventListener('message', async message => {
    if (message.iceCandidate) {
        try {
            await peerConnection.addIceCandidate(message.iceCandidate);
        } catch (e) {
            console.error('Error adding received ice candidate', e);
        }
    }
});
```

다음은 peer connection 의 상태를 확인하는 code 이다.

```js
// Listen for connectionstatechange on the local RTCPeerConnection
peerConnection.addEventListener('connectionstatechange', event => {
    if (peerConnection.connectionState === 'connected') {
        // Peers connected!
    }
});
```

Alice 는 자신의 browser 에 자신의 얼굴을 보여주는 video element 를
가지고 있다고 하자. video element 의 Audio, Video track 을 
peer connection 에 등록한다. 이후 Bob 에게 track 을 추가해 달라는
요청을 한다.

```js
const localStream = await getUserMedia({vide: true, audio: true});
const peerConnection = new RTCPeerConnection(iceConfig);
localStream.getTracks().forEach(track => {
    peerConnection.addTrack(track, localStream);
});
```

Alice 로 부터 track 을 추가해 달라는 요청이 Bob 에게 왔다고 하자.
Bob 는 자신의 browser 에 Alice 의 얼굴을 보여주는 video element source 에 
remote track 을 등록한다. 다음은 remote track 을 video element 의 
source 에 등록하는 code 이다.

```js
const remoteVideo = document.querySelector('#remoteVideo');

peerConnection.addEventListener('track', async (event) => {
    const [remoteStream] = event.streams;
    remoteVideo.srcObject = remoteStream;
});
```

## WebRTC Sequence

* [sfu-ws @ github](https://github.com/pion/example-webrtc-applications/tree/master/sfu-ws)
  * WebRTC many to many exqmples made by go

----

code 를 중심으로 흐름을 이해하자.

## Signaling

WebRTC Connection 을 위해 Peer 들끼리 Meta Data 를 주고 받는다. 예를 들어 [SDP (Session Description Protocol)](/sdp/README.md), ICE Candidate 을 주고 받는다.

## Connecting

WebRTC Connection 을 위해 [ICE (Interactive Connectivity Establishment)](https://datatracker.ietf.org/doc/html/rfc8445), [STUN (Session Traversal Utilities for NAT)](https://datatracker.ietf.org/doc/html/rfc8489), [TURN (Traversal Using Relays around NAT)](https://datatracker.ietf.org/doc/html/rfc8656) 을 사용한다.

## Securing

Securing 을 위해 DTLS (Datagram Transport Layer Security), SRTP (Secure Real-time Transport Protocol) 를 사용한다.

DTLS 로 대칭 key 를 교환하고 SRTP 로 암호화된 RTP 를 전송한다.

## Media Communication

> [Media Communication](https://webrtcforthecurious.com/docs/06-media-communication/)

Media Communication 을 위해 [RTP (Real-time Transport Protocol)](/rtp/README.md), [RTCP (RTP Control Protocol)](/rtcp/README.md) 를 사용한다.

## Data Communication

> [Data Communication](https://webrtcforthecurious.com/docs/07-data-communication/)

Data Channel 을 위해 SCTP (Stream Control Transmission Protocol), DCEP (Data Channel Establishment Protocol) 을 사용한다.

SCTP 는 패킷의 유실을 막고 순서를 보장해준다. [rfc4960](https://datatracker.ietf.org/doc/html/rfc4960) 에 정의되어 있다.


DCEP 는 SCTP 가 지원하지 않는 기능을 위해 사용한다. Data Channel 에 Label 을 부여할 수 있다. [rfc8832](https://datatracker.ietf.org/doc/html/rfc8832) 에 정의되어 있다.

## WebRTC Topologies

* [WebRTC-Mesh, MCU, SFU architecture](https://www.programmersought.com/article/40593543499/)
* [WebRTC Topologies #](https://webrtcforthecurious.com/docs/08-applied-webrtc/#webrtc-topologies)

-----

WebRTC 의 Architecture 는 구현방법에 따라 Mesh, SFU, MCU 로 구분할 수 있다. network bandwith 만 문제 없다면 SFU 가 가장 적당하다.

### Mesh

![](img/webrtc_mesh.png)

uplink, downlink pressure

### SFU (Selective Forwarding Unit)

![](img/webrtc_mesh.png)

downlink pressure

### MCU (Multi Control Unit)

![](img/webrtc_mesh.png)

CPU pressure for encoding, decoding, mixing

## Debugging

> [Debugging](https://webrtcforthecurious.com/docs/09-debugging/)

WebRTC 의 Debugging 은 어렵다. 영역별로 구분해서 debugging 해야 한다. 다음과 같이 Category 를 나눈다.

* Signaling Failure
* Network Failure
  * STUN server 로 request 해보기
* Security Failure
* Media Failure
* Data Failure

다음과 같은 tool 을 이용한다.

* [nc]([/linux/README.md#commands](https://en.wikipedia.org/wiki/Netcat))
* [tcpdump](https://en.wikipedia.org/wiki/Tcpdump)
* [tshark](/tshark/README.md)
* [webrtc-internals | chrome](chrome://webrtc-internals/)

# Advanced

## WebRTC Implementations

> * [pion | github](https://github.com/pion)

go 로 만든 WebRTC Implementation 이다.

> * [ion-sfu | github](https://github.com/pion/ion-sfu)
>   * [ion-sfu examples | github](https://github.com/pion/ion-sfu/tree/master/examples)

go 로 만든 WebRTC SFU Server 이다. pion, ion 을 사용한다. gRPC, json-rpc 을 지원한다.

> * [ion-sdk-go](https://github.com/pion/ion-sdk-go)
>   * [ion-sdk-go/example](https://github.com/pion/ion-sdk-go/tree/master/example)

go 로 만든 [ion-sfu](https://github.com/pion/ion-sfu) Client SDK 이다.

> * [ion | github](https://github.com/pion/ion)
>   * [ion doc](https://pionion.github.io/docs/intro)

Real-Distributed RTC System by pure Go and Flutter.
It uses ion-sfu

> * [LiveKit](https://livekit.io/)
>   * [LiveKit doc](https://docs.livekit.io/)

go 로 만든 WebRTC Infrastructure 이다. pion, ion 을 사용한다.

> * [janus-gateway](https://github.com/meetecho/janus-gateway)

c 로 만든 WebRTC Server 이다.
