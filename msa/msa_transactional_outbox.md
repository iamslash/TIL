# Abstract

* business logic 과 message 전송을 하나의 local transaction 에 포함하는 pattern
  이다. 반드시 message 가 전송될 것을 보장한다.
* business logic 을 수행하고 RDBMS 에 data 를 저장함과 동시에 outbox table 에
  message 를 저장한다. outbox table 에 message 가 저장되면 message relay
  component 가 그것을 polling 하고 있다가 message broker 에게 전송한다. 때로는
  message table 을 polling 하지 않고 DB transaction log 를 tailing 하다가
  message 를 전송할 수도 있다. 이것을 **Transaction log tailing** 이라고 한다.
* message relay component 가 message 를 가져가다 실패하면 재전송할 수 있도록
  구현해야 한다.

# Materials

* [Sending Reliable Event Notifications with Transactional Outbox Pattern](https://medium.com/event-driven-utopia/sending-reliable-event-notifications-with-transactional-outbox-pattern-7a7c69158d1b)
  * [debezium-examples/outbox/ | github](https://github.com/debezium/debezium-examples/tree/main/outbox)
* [Pattern: Transactional outbox](https://microservices.io/patterns/data/transactional-outbox.html)

# Basic

> [Implementing the Outbox Pattern](https://dzone.com/articles/implementing-the-outbox-pattern)

**Transaction log tailing** can be implemented in a very elegant and efficient way using Change Data Capture (CDC) with [Debezium](https://debezium.io/) and Kafka-Connect.
