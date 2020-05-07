- [Materials](#materials)
- [History](#history)
- [Spring Cloud VS Kubernetes](#spring-cloud-vs-kubernetes)
- [Pattern](#pattern)
  - [Application architecture patterns](#application-architecture-patterns)
  - [Decomposition](#decomposition)
  - [Refactoring to microservices](#refactoring-to-microservices)
  - [Data management](#data-management)
  - [Transactional messaging](#transactional-messaging)
  - [Testing](#testing)
  - [Deployment patterns](#deployment-patterns)
  - [Cross cutting concerns](#cross-cutting-concerns)
  - [Communication style](#communication-style)
  - [External API](#external-api)
  - [Service discovery](#service-discovery)
  - [Reliability](#reliability)
  - [Security](#security)
  - [Observability](#observability)
  - [UI patterns](#ui-patterns)

----

# Materials

* [MSA 제대로 이해하기 -(1) MSA의 기본 개념](https://velog.io/@tedigom/MSA-%EC%A0%9C%EB%8C%80%EB%A1%9C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0-1-MSA%EC%9D%98-%EA%B8%B0%EB%B3%B8-%EA%B0%9C%EB%85%90-3sk28yrv0e)
* [Microservices @ wikipedia](https://en.wikipedia.org/wiki/Microservices)
* [A pattern language for microservices](https://microservices.io/patterns/index.html)
  - microservices 의 기본개념
  - [src](https://github.com/gilbutITbook/007035)
  
# History

As early as 2005, Peter Rodgers introduced the term "Micro-Web-Services" during a presentation at the Web Services Edge conference.

# Spring Cloud VS Kubernetes

Spring Cloud 는 Microservices 를 위해 탄생한 framework 이다.
microservices 의 대표적인 implementation 중 Spring Cloud 와 Kubernetes 를 MSA 관점에 따라 비교해 본다.

| Concern                               | Spring Cloud                          | Kubernetes                                           |
| ------------------------------------- | ------------------------------------- | ---------------------------------------------------- |
| Configuration Management              | Spring Config Server                  | Kubernetes ConfigMaps                                |
| Service Discovery                     | Spring Cloud Eureka                   | Kubernetes Services                                  |
| Load Balancing                        | Spring Cloud Ribbon                   | Kubernetes Services                                  |
| API gateway                           | Spring Cloud Zuul                     | Kubernetes Services, Ingress                         |
| Security                              | Spring Cloud Security                 | Service meshes (ex. Istio)                           |
| Centralized logging                   | ELK (Elasticsearch, LogStash, Kibana) | EFK (Eleasticsearch, Fluentd, Kibana)                |
| Centralized metrics                   | Spring Spectator & Atlas              | Heapster, Prometheus & Grafana                       |
| Distributed tracing                   | Spring Cloud Sleuth                   | Hawkular, Jaeger                                     |
| Resilience and fault tolerance        | Spring Hystrix, Turbine & Ribbon      | Health check, Service meshes (ex. Istio)             |
| Autoscaling and self-healing          | Spring hystrix, Turbine, & Ribbon     | Health check, service meshes (ex. Istio)             |
| Packaging, deployment, and scheduling | Spring Boot, Apache Maven             | Docker, Rkt, Kubernetes Scheduler & Deployment, Helm |
| Job management                        | Spring Batch                          | Kubernetes Pods                                      |
| Singleton application                 | Spring Cloud Cluster                  | Kubernetes Pods                                      |
 
# Pattern

## Application architecture patterns

Monolithic architecture
Microservice architecture

## Decomposition

Decompose by business capability
Decompose by subdomain
Self-contained Service
Service per team

## Refactoring to microservices

Strangler Application
Anti-corruption layer

## Data management

Database per Service
Shared database
Saga
API Composition
CQRS
Domain event
Event sourcing

## Transactional messaging

Transactional outbox
Transaction log tailing
Polling publisher

## Testing

Service Component Test
Consumer-driven contract test
Consumer-side contract test

## Deployment patterns

Multiple service instances per host
Service instance per host
Service instance per VM
Service instance per Container
Serverless deployment
Service deployment platform

## Cross cutting concerns

Microservice chassis
Externalized configuration

## Communication style

Remote Procedure Invocation
Messaging
Domain-specific protocol
Idempotent Consumer

## External API

API gateway
Backend for front-end

## Service discovery

Client-side discovery
Server-side discovery
Service registry
Self registration
3rd party registration

## Reliability

Circuit Breaker

## Security

Access Token

## Observability

Log aggregation
Application metrics
Audit logging
Distributed tracing
Exception tracking
Health check API
Log deployments and changes

## UI patterns

Server-side page fragment composition
Client-side UI composition
