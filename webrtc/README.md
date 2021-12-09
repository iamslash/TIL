- [Abstract](#abstract)
- [Meterials](#meterials)
- [Basic](#basic)
  - [Features](#features)
  - [WebRTC Overview](#webrtc-overview)
  - [WebRTC Sequence](#webrtc-sequence)

---

# Abstract

- WebRTC (web real-time communication)
- browser 에서도 skype 처럼 실시간으로 video, voice, data 들을 주고 받아 보자.
- browser 에서 화상 채팅등을 할 수 있다.

# Meterials

- [WebRTC 이론 정리하기](https://millo-l.github.io/WebRTC-%EC%9D%B4%EB%A1%A0-%EC%A0%95%EB%A6%AC%ED%95%98%EA%B8%B0/)
  - [WebRTC 구현 방식(Mesh/P2P, SFU, MCU)](https://millo-l.github.io/WebRTC-%EA%B5%AC%ED%98%84-%EB%B0%A9%EC%8B%9D-Mesh-SFU-MCU/)
  - [WebRTC 구현하기(1:1 P2P)](https://millo-l.github.io/WebRTC-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0-1-1-P2P/)
  - [WebRTC 구현하기(1:N P2P)](https://millo-l.github.io/WebRTC-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0-1-N-P2P/)
  - [WebRTC 구현하기(1:N SFU)](https://millo-l.github.io/WebRTC-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0-1-N-SFU/)
  - [WebRTC 성능 비교(P2P vs SFU)](https://millo-l.github.io/WebRTC-%EC%84%B1%EB%8A%A5%EB%B9%84%EA%B5%90-P2P-vs-SFU/)
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

## Features

| Feature | Description|
|---|---|
| STUN | Session Traversal Utilities for NAT |
| TURN | Traversal Using Relay NAT |
| ICE (Internet Connectivity Establishment) candidate | WebRTC 로 연결할 peer 이다. |
| ICE Servers | Ice candidate 끼리 communication 이 가능하도록 도와주는 server 들. STUN, TURN 서버를 말한다. |
| Signaling | Ice candidates 및 SDP 를 교환하는 것. 전화로 따지면 상대의 번호를 누르고 기다리는 행위이다. |
| Trickle ICE | ICE candidate 가 발견될 때 마다 ICE candidate 을 보낸다. 수도꼭지에서 물 방울이 조금씩 흐르는 것처럼 ICE candidate 를 조금씩 흘려보낸다는 의미이다. |
| [SDP](/sdp/README.md) | Session Description Protocol |

## WebRTC Overview

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
