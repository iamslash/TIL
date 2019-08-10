# Materials

* [How To Install Apache Kafka on Ubuntu 18.04](https://www.digitalocean.com/community/tutorials/how-to-install-apache-kafka-on-ubuntu-18-04)
* [kafka @ joinc](https://www.joinc.co.kr/w/man/12/Kafka)

# Zero Copy

![](img/zerocopy_1.gif)

데이터를 읽어서 네트워크로 전송할 때 kernel mode -> user mode -> kernel mode 순서로 OS 의 mode 변환이 필요하다.

![](img/zerocopy_2.gif)

이때 user mode 변환 없이 데이터를 네트워크로 전송하는 것을 zero copy 라고 한다.