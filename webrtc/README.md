# intro

- WebRTC (web real-time communication)
- browser에서도 skype처럼 실시간으로 video, voice, data들을 주고 받아 보자.
- browser에서 화상 채팅등을 할 수 있다.

# usage

## NAT

- network address translation 을 의미한다.
- private side, public side 두 열의 정보를 이용하여 인터넷 주소를 변환한다.
- ipv4에서 192.168.x.x, 10.x.x.x는 private 주소로 예약되어 있다.
- 예를 들어서 A(192.168.1.3:42301)에서 N(192.168.1.1, 12.13.14.15)을 거쳐
  B(40.30.20.10:80)으로 패킷을 보내자.
- NAT forwarding table에 다음과 같은 정보가 저장된다.

| Private side  | Public side |
|:---:|:--:|
| 192.168.1.3:42301 | 12.13.14.15:24604 |

- N은 A혹은 B에서 패킷을 수신할때마다 주소를 변환하여 전달한다.

## UDP hole punching

- NAT는 크게 Cone NAT와 Symmetric NAT로 분류할 수 있다.
- Cone NAT는 Full Cone, Restricted Cone, Port Restricted Cone으로 분류할 수 있다.
- 특정 port를 bind(192.168.1.3:42301)하여 소켓을 제작한 후에 udp
  패킷을 하나 NAT를 통해 remote machine에 보내면 NAT forwarding
  table에 항목이 추가되며 udp hole이 만들어진다. 추가된 내용중 public
  side(12.13.14.15:24604)로 NAT외부에서 udp packet을 보내면 앞서
  bind한 socket으로 패킷을 수신 할 수 있다. NAT에 기록된 udp hole은
  NAT종류에 따라 유지되는 시간이 다양하다.
- N1 NAT에 속한 C1과 N2 NAT에 속한 C2가 있다고 가정하자. 둘다 S에 udp
  패킷을 하나 보내면 N1, N2의 forwarding table에 S와의 관계가
  기록되면서 udp hole이 만들어 진다.  이것은 N1, N2가 S와 패킷을 주고
  받을 수 있는 hole이다. 아직 N1는 C2와 udp hole이 없기 때문에 C1은
  C2와 패킷을 주고 받을 수 없다.
- C1이 C2에 udp packet을 하나 보내면 N1의 forwarding table에 N2와의
  관계가 기록되면서 udp hole이 만들어 진다. C2는 N2를 통해 C1으로 udp
  packet을 보낼 수 있다.  C2역시 비슷한 과정을 통해서 C1에 N2의 udp
  hole을 이용하여 udp packet을 보낼 수 있다.
- [udp hole punching at youtube](https://www.youtube.com/watch?v=s_-UCmuiYW8)
  - nc, hping3를 이용해서 udp hole punching을 하는 방법을 설명한다.
  - local computer의 public ip(l.l.l.l)를 얻어오자.
    - curl ipecho.net/plain && echo
  - local computer(l.l.l.l)에서 nc를 이용해서 수신된 udp패킷을 출력하자.
    - nc -u -l -p 12001
  - local computer에서 hping3를 이용해서 udp hole을 만들자.
    - hping3 -c 1 -2 -s 12001 -p 12003 r.r.r.r
  - remote computer에서 nc를 이용해서 udp 패킷을 송신하자.
    - local computer의 포트가 12003이란 것은 어떻게 알아내지???
    - echo "udp hole" | nc -p 12003 -u l.l.l.l
  - 한번 만들어진 udp hole은 패킷왕래가 일어나지 않으면 닫혀진다.

## WebRtc

- [Real-time communication with WebRTC: Google I/O 2013](https://www.youtube.com/watch?v=p2HzZkd2A40)을 열심히 보자.
  - webRTC에 대한 전반적인 설명과 함께 예제 코드가 좋다.
- [Real time communication with WebRTC](https://codelabs.developers.google.com/codelabs/webrtc-web/#0)
  - 10단계로 설명하는 WebRTC
- Three main JavaScript APIs.
  - MediaStream(getUserMedia), RTCPeerConnection, RTCDataChannel
- [ascii-camera](https://idevelop.ro/ascii-camera/)는 camera의 데이터를 ascii데이터로 렌더링해서 보여준다.
- MediaStream(getUserMedia)

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

## STUN

- Session Traversal Utilities for NAT
- local computer 에서 NAT바깥의 STUN server에게 자신의 public ip를 얻어 내고 
  p2p session이 가능한지 확인한다.

## TURN

- Traversal Using Relays around NAT
- STUN server를 통해서 얻은 public ip를 이용하여 p2p session을 획득하는데 실패 했다면
  TURN server를 통해서 packet을 relay하자.

## ICE

- STUN, TURN framework

# reference

- [go-stun](https://github.com/pixelbender/go-stun)
  - golang으로 STUN, TURN, ICE protocol을 구현했다.
- [NAT Traversal의 종결자](http://www.nexpert.net/424)
  - WebRTC의 STUN, TUN, ICE를 이해할 수 있었다.
