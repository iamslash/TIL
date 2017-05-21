# intro

- WebRTC (web real-time communication)
- browser에서도 skype처럼 실시간으로 video, voice, data들을 주고 받아 보자.
- browser에서 화상 채팅등을 할 수 있다.

# usage

## UDP hole punching

- [udp hole punching at youtube](https://www.youtube.com/watch?v=s_-UCmuiYW8)
  - nc, hping3를 이용해서 udp hole punching을 하는 방법
  - local computer(l.l.l.l)에서 nc를 이용해서 수신된 udp패킷을 출력하자.
    - nc -u -l -p 12001
  - hping3를 이용해서 udp hole을 만들자.
    - hping3 -c 1 -2 -s 12001 -p 12003 r.r.r.r
    - r.r.r.r의 NAT의 port 12003 
  - nc를 이용해서 remote computer에서 udp 패킷을 송신하자.
    - echo "udp hole" | nc -p 12003 -u l.l.l.l
  - 한번 만들어진 udp hole은 패킷왕래가 일어나지 않으면 닫혀진다.

## STUN
## TUN
## ICE

# reference

- [NAT Traversal의 종결자](http://www.nexpert.net/424)
  - WebRTC의 STUN, TUN, ICE를 이해할 수 있었다.
