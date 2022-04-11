- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
  - [Integration Styles](#integration-styles)
    - [File Transfer](#file-transfer)
    - [Shared Database](#shared-database)
    - [Remote Procedure Invocation](#remote-procedure-invocation)
    - [Messaging](#messaging)
  - [Messaging Systems](#messaging-systems)
    - [Message Channel](#message-channel)
    - [Message](#message)
    - [Pipes and Filters](#pipes-and-filters)
    - [Message Router](#message-router)
    - [Message Translator](#message-translator)
    - [Message Endpoint](#message-endpoint)
  - [Messaging Channels](#messaging-channels)
    - [Point-to-Point Channel](#point-to-point-channel)
    - [Publish-Subscribe Channel](#publish-subscribe-channel)
    - [Datatype Channel](#datatype-channel)
    - [Invalid Message Channel](#invalid-message-channel)
    - [Dead Letter Channel](#dead-letter-channel)
    - [Guaranteed Delivery](#guaranteed-delivery)
    - [Channel Adapter](#channel-adapter)
    - [Messaging Bridge](#messaging-bridge)
    - [Message Bus](#message-bus)
  - [Message Construction](#message-construction)
    - [Command Message](#command-message)
    - [Document Message](#document-message)
    - [Event Message](#event-message)
    - [Request-Reply](#request-reply)
    - [Return Address](#return-address)
    - [Correlation Identifier](#correlation-identifier)
    - [Message Sequence](#message-sequence)
    - [Message Expiration](#message-expiration)
    - [Format Indicator](#format-indicator)
  - [Message Routing](#message-routing)
    - [Content-Based Router](#content-based-router)
    - [Message Filter](#message-filter)
    - [Dynamic Router](#dynamic-router)
    - [Recipient List](#recipient-list)
    - [Splitter](#splitter)
    - [Aggregator](#aggregator)
    - [Resequencer](#resequencer)
    - [Composed Msg. Processor](#composed-msg-processor)
    - [Scatter-Gather](#scatter-gather)
    - [Routing Slip](#routing-slip)
    - [Process Manager](#process-manager)
    - [Message Broker](#message-broker)
  - [Message Transformation](#message-transformation)
    - [Envelope Wrapper](#envelope-wrapper)
    - [Content Enricher](#content-enricher)
    - [Content Filter](#content-filter)
    - [Claim Check](#claim-check)
    - [Normalizer](#normalizer)
    - [Canonical Data Model](#canonical-data-model)
  - [Messaging Endpoints](#messaging-endpoints)
    - [Messaging Gateway](#messaging-gateway)
    - [Messaging Mapper](#messaging-mapper)
    - [Transactional Client](#transactional-client)
    - [Polling Consumer](#polling-consumer)
    - [Event-Driven Consumer](#event-driven-consumer)
    - [Competing Consumers](#competing-consumers)
    - [Message Dispatcher](#message-dispatcher)
    - [Selective Consumer](#selective-consumer)
    - [Durable Subscriber](#durable-subscriber)
    - [Idempotent Receiver](#idempotent-receiver)
    - [Service Activator](#service-activator)
  - [System Management](#system-management)
    - [Control Bus](#control-bus)
    - [Detour](#detour)
    - [Wire Tap](#wire-tap)
    - [Message History](#message-history)
    - [Message Store](#message-store)
    - [Smart Proxy](#smart-proxy)
    - [Test Message](#test-message)
    - [Channel Purger](#channel-purger)

----

# Abstract

Enterpise Integration Pattern 에 대해 정리한다. [Patterns of Enterprise Application Architecture](https://www.amazon.com/o/asin/0321200683/ref=nosim/enterpriseint-20) 을 요약한 것이다. 대부분 [Kafka](/kafka/README.md) 으로 구현 가능하다.

Pattern Leanguage 는 책마다 모두 다르다. 주로 Alexandrian form 을 사용한다. Alexandrian form 은 Smalltalk Best Practice Patterns by Kent Beck 에서 처음 소개되었다.

# Materials

* [65 messaging patterns](https://www.enterpriseintegrationpatterns.com/patterns/messaging/toc.html)

# Basic

## Integration Styles
### File Transfer
### Shared Database
### Remote Procedure Invocation
### Messaging

## Messaging Systems
### Message Channel
### Message
### Pipes and Filters
### Message Router
### Message Translator
### Message Endpoint

## Messaging Channels
### Point-to-Point Channel
### Publish-Subscribe Channel
### Datatype Channel
### Invalid Message Channel
### Dead Letter Channel
### Guaranteed Delivery
### Channel Adapter
### Messaging Bridge
### Message Bus

## Message Construction
### Command Message
### Document Message
### Event Message
### Request-Reply
### Return Address
### Correlation Identifier
### Message Sequence
### Message Expiration
### Format Indicator

## Message Routing
### Content-Based Router
### Message Filter
### Dynamic Router
### Recipient List
### Splitter
### Aggregator
### Resequencer
### Composed Msg. Processor
### Scatter-Gather
### Routing Slip
### Process Manager
### Message Broker

## Message Transformation
### Envelope Wrapper
### Content Enricher
### Content Filter
### Claim Check
### Normalizer
### Canonical Data Model

## Messaging Endpoints
### Messaging Gateway
### Messaging Mapper
### Transactional Client
### Polling Consumer
### Event-Driven Consumer
### Competing Consumers
### Message Dispatcher
### Selective Consumer
### Durable Subscriber
### Idempotent Receiver
### Service Activator

## System Management
### Control Bus
### Detour
### Wire Tap
### Message History
### Message Store
### Smart Proxy
### Test Message
### Channel Purger
