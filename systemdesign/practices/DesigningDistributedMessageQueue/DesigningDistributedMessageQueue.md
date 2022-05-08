- [Abstract](#abstract)
- [Requirements](#requirements)
  - [Functional Requirement](#functional-requirement)
  - [Non-Functional Requirement](#non-functional-requirement)
- [Estimation](#estimation)
- [High Level Design](#high-level-design)
  - [Message Models](#message-models)
  - [Topics, Partitions, Brokers](#topics-partitions-brokers)
  - [Consumer Group](#consumer-group)
  - [High Level Arhictecture](#high-level-arhictecture)
- [Low Level Design](#low-level-design)
  - [Data Storage](#data-storage)
  - [Message Data Structure](#message-data-structure)
  - [Batching](#batching)
  - [Producer Flow](#producer-flow)
  - [Consumer Flow](#consumer-flow)
  - [Consumer Rebalancing](#consumer-rebalancing)
  - [State Storage, Metadata Storage](#state-storage-metadata-storage)
  - [Replication](#replication)
  - [Scalability](#scalability)
  - [Deliver Semantics](#deliver-semantics)
- [Extentions](#extentions)
- [Q&A](#qa)
- [References](#references)

----

# Abstract

[Kafka](/kafka/README.md) 와 비슷한 message broker 를 디자인한다.

# Requirements

## Functional Requirement

* Producers send messages to msg queue
* consumers consume msgs from the msg queue
* messages can be repeatedly consumed
* The system should provide message ordering

## Non-Functional Requirement

* The system should be configurable throughtput and latency
* The system should be scalable
* The system should be persistent and durable

# Estimation

# High Level Design

## Message Models

## Topics, Partitions, Brokers

## Consumer Group

## High Level Arhictecture

# Low Level Design

## Data Storage

## Message Data Structure

## Batching

## Producer Flow

## Consumer Flow

## Consumer Rebalancing

## State Storage, Metadata Storage

## Replication

## Scalability

## Deliver Semantics

* at-most once
* at-least once
* exactly once

# Extentions

# Q&A

# References
