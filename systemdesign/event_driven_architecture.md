# Abstract

Event Driven Architecture 에 대해 정리한다.

# Materials

* [회원시스템 이벤트기반 아키텍처 구축하기 | wooahan](https://techblog.woowahan.com/7835/)
* [[우아콘2020] 배달의민족 마이크로서비스 여행기 | youtube](https://www.youtube.com/watch?v=BnS6343GTkY)

# Basic

[Kafka](/kafka/README.md) 와 같은 Message Broker 들을 두고 여러
[msa](/systemdesign/msa.md) Application 들이 통신하는 구조이다.

모두 모여 Event 를 정의해야 한다. [Event Storming](/ddd/README.md#event-storming) 를 참고하자.

Message Broker 로 Event 가 전달되었음을 보장하기 위해 [transactional outbox pattern](https://microservices.io/patterns/data/transactional-outbox.html) 을 도입해야 한다.

경우에 따라서는 payload data 를 Event 에 포함하지 않는 경우도 있다. 이것을
zero-payload 라고 한다. Consumer 는 해당 Event 를 받으면 Producer 에게 api 를
호출하여 payload 를 얻어온다. Strong consistency 혹은 big payload Event 에
유용하다. payload 가 zero 이기 때문에 zero-payload 라고 한다. 

과도한 트래픽과 충분한 개발인력이 있다면 반드시 도입해야 한다.
