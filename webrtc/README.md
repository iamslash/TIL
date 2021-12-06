- [Abstract](#abstract)
- [Meterials](#meterials)
- [Basic](#basic)
  - [WebRTC Overview](#webrtc-overview)
  - [WebRTC Sequence](#webrtc-sequence)

---

# Abstract

- WebRTC (web real-time communication)
- browser 에서도 skype 처럼 실시간으로 video, voice, data 들을 주고 받아 보자.
- browser 에서 화상 채팅등을 할 수 있다.

# Meterials

- [Guide to WebRTC @ baeldung](https://www.baeldung.com/webrtc)  
- [STUNTMAN](http://www.stunprotocol.org/)
  - c++로 제작된 STUN, TURN server, client
- [go-stun](https://github.com/pixelbender/go-stun)
  - golang으로 STUN, TURN, ICE protocol을 구현했다.
- [NAT Traversal의 종결자](http://www.nexpert.net/424)
  - WebRTC의 STUN, TUN, ICE를 이해할 수 있었다.

# Basic

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

```javascript
var constrains = {video: true};

function successCallback(stream) {
    var video = document.querySelector("video");
    video.src = window.URL.createObjectURL(stream);
}

function errorCallback(error) {
    console.log("navigator.getUserMedia error: ", error);
}
navigator.getUserMedia(constrains, successCallback, errorCallback);
```

- Constraints

```javascript
video : {
  mandatory: {
    minWidth: 640,
    minHeight: 360
  },
  optional [{
    minWidgth: 1280,
    minHeight: 720
  }]
}
```

- RTCPeerConnection

```javascript
pc = new RTCPeerConnection(null);
pc.onaddstream = gotRemoteStreaml
pc.addStream(localStream);
pc.createOffer(gotOffer);

function gotOffer(desc) {
    pc.setLocalDescription(desc);
    sendOffer(desc);
}

function gotAnswer(desc) {
    pc.setRemoteDescription(desc);
}

function gotRemoteStream(e) {
    attachMediaStream(remoteVideo, e.stream);
}
```

- RTCDataChannel

```javascript
var pc = new webkitRTCPeerConnection(servers,
  {optional: [{RtpDataChannels: true}]});
  
pc.ondatachannel = function(event) {
  receiveChannel = event.channel;
  receiveChannel.onmessage = function(event) {
    document.querySelect("div#receive").innerHTML = event.data;
  };
};
  
sendChannel = pc.createDataChannel("sendDataChannel", {reliable: false});
  
document.querySelect("button#send").onclick = function() {
  var data = document.querySelect("textarea#send").value;
  sendChannel.send(data);
}
```

- Abstract Signaling
  - peer와 peer간에 session description을 교환하는 것. 

## WebRTC Sequence

* [sfu-ws @ github](https://github.com/pion/example-webrtc-applications/tree/master/sfu-ws)
  * WebRTC many to many exqmples made by go

----

code 를 중심으로 흐름을 이해하자.
