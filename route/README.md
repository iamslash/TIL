# Materials

* [라우팅 테이블 다루기](https://thebook.io/006718/part01/ch03/06/02/)
* [Ubnutu route 설정](https://xmlangel.github.io/ubuntu-route/)

# How to reoute packets

현재 route table 목록을 조회한다.

```
$ route
Destination   Gateway   Genmask        Flags Metric Ref   Use Iface
192.168.0.0   *         255.255.255.0  U     0      0       0 eth0
```

현재 MAchine 의 ip 는 192.168.0.195, 목적지 Machine 의 ip 는 192.168.0.196 이라고 하자. 목적지 machine 까지 packet 이 이동하는 절차는 다음과 같다.

* route table 의 모든 항목을 순회한다. 하나의 아이템에 대해 다음을 반복한다.
  * `255.255.255.0` (Genmask) & `192.168.0.196` = `192.168.0.0`
  * `192.168.0.0` 이 Destination 과 같다면 Iface 로 packet 을 보낸다. Gateway 를 통해서 감?
  * 그렇지 않다면 다음 아이템으로 넘어간다.
