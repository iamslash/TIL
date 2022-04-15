# Abstract

* RDBMS 의 outbox table 을 사용하여 message 를 message broker 에 전송하는 것을 local transaction 에 포함하는 pattern. outbox table 에 message 가 저장되면 message relay component 가 그것을 polling 하고 있다가 message broker 에게 전송한다. 때로는 message table 을 polling 하지 않고 DB transaction log 를 tailing 하다가 message 를 전송할 수도 있다. 이것을 **Transaction log tailing** 이라고 한다.
* message 전송과 business logic 을 하나의 transaction 으로 관리할 수 있다.

# Materials

* [Sending Reliable Event Notifications with Transactional Outbox Pattern](https://medium.com/event-driven-utopia/sending-reliable-event-notifications-with-transactional-outbox-pattern-7a7c69158d1b)
  * [debezium-examples/outbox/ | github](https://github.com/debezium/debezium-examples/tree/main/outbox)

# Basic

[debezium](https://debezium.io/) 으로 쉽게 구현할 수 있는 것 같다.

[Eventuate Tram](https://eventuate.io/abouteventuatetram.html) 은 Transactional Outbox 를 제공한다.
