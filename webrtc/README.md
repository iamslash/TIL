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

- [udp hole punching at youtube](https://www.youtube.com/watch?v=s_-UCmuiYW8)
  - nc, hping3를 이용해서 udp hole punching을 하는 방법을 설명한다.
  - local computer의 public ip(l.l.l.l)를 얻어오자.
    - curl ipecho.net/plain && echo
  - local computer(l.l.l.l)에서 nc를 이용해서 수신된 udp패킷을 출력하자.
    - nc -u -l -p 12001
  - local computer에서 hping3를 이용해서 udp hole을 만들자.
    - hping3 -c 1 -2 -s 12001 -p 12003 r.r.r.r
  - remote computer에서 nc를 이용해서 udp 패킷을 송신하자.
    - echo "udp hole" | nc -p 12003 -u l.l.l.l
  - 한번 만들어진 udp hole은 패킷왕래가 일어나지 않으면 닫혀진다.

## STUN

- 

## TUN

-

## ICE

-

# reference

- [NAT Traversal의 종결자](http://www.nexpert.net/424)
  - WebRTC의 STUN, TUN, ICE를 이해할 수 있었다.
