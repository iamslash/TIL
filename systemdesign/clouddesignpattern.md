- [Materials](#materials)
- [Patterns](#patterns)
  - [Ambassador](#ambassador)
  - [Anti-Corruption Layer](#anti-corruption-layer)
  - [Asynchronous Request-Reply](#asynchronous-request-reply)
  - [Backends for Frontends](#backends-for-frontends)
  - [Bulkhead](#bulkhead)
  - [Cache-Aside](#cache-aside)
  - [Choreography](#choreography)
  - [Circuit Breaker](#circuit-breaker)
  - [Claim Check](#claim-check)
  - [Compensating Transaction](#compensating-transaction)
  - [Competing Consumers](#competing-consumers)
  - [Compute Resource Consolidation](#compute-resource-consolidation)
  - [CQRS](#cqrs)
  - [Deployment Stamps](#deployment-stamps)
  - [Event Sourcing](#event-sourcing)
  - [External Configuration Store](#external-configuration-store)
  - [Federated Identity](#federated-identity)
  - [Gatekeeper](#gatekeeper)
  - [Gateway Aggregation](#gateway-aggregation)
  - [Gateway Offloading](#gateway-offloading)
  - [Gateway Routing](#gateway-routing)
  - [Geodes](#geodes)
  - [Health Endpoint Monitoring](#health-endpoint-monitoring)
  - [Index Table](#index-table)
  - [Leader Election](#leader-election)
  - [Materialized View](#materialized-view)
  - [Pipes and Filters](#pipes-and-filters)
  - [Priority Queue](#priority-queue)
  - [Publisher-Subscriber](#publisher-subscriber)
  - [Queue-Based Load Leveling](#queue-based-load-leveling)
  - [Retry](#retry)
  - [Scheduler Agent Supervisor](#scheduler-agent-supervisor)
  - [Sequential Convoy](#sequential-convoy)
  - [Sharding](#sharding)
  - [Sidecar](#sidecar)
  - [Static Content Hosting](#static-content-hosting)
  - [Strangler](#strangler)
  - [Throttling](#throttling)
  - [Valet Key](#valet-key)

----

# Materials

* [Cloud Design Patterns](https://docs.microsoft.com/en-us/azure/architecture/patterns/)

# Patterns

## Ambassador
* Create helper services that send network requests on behalf of a consumer service or application.
* [The Ambassador Pattern](https://blog.davemdavis.net/2020/03/17/the-ambassador-pattern/)
* [Ambassador pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/ambassador)

## Anti-Corruption Layer
* Implement a façade or adapter layer between a modern application and a legacy system.

## Asynchronous Request-Reply
* Decouple backend processing from a frontend host, where backend processing needs to be asynchronous, but the frontend still needs a clear response.

## Backends for Frontends
* Create separate backend services to be consumed by specific frontend applications or interfaces.

## Bulkhead
* Isolate elements of an application into pools so that if one fails, the others will continue to function.

## Cache-Aside
* Load data on demand into a cache from a data store
* [Cache-Aside pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/cache-aside)

## Choreography
* Let each service decide when and how a business operation is processed, instead of depending on a central orchestrator.

## Circuit Breaker

* Handle faults that might take a variable amount of time to fix when connecting to a remote service or resource.
* [Circuit Breaker pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker)

## Claim Check
* Split a large message into a claim check and a payload to avoid overwhelming a message bus.

## Compensating Transaction

* Undo the work performed by a series of steps, which together define an eventually consistent operation.

## Competing Consumers
* Enable multiple concurrent consumers to process messages received on the same messaging channel.

## Compute Resource Consolidation
* Consolidate multiple tasks or operations into a single computational unit

## CQRS
* Segregate operations that read data from operations that update data by using separate interfaces.

## Deployment Stamps
* Deploy multiple independent copies of application components, including data stores.

## Event Sourcing
* Use an append-only store to record the full series of events that describe actions taken on data in a domain.

## External Configuration Store
* Move configuration information out of the application deployment package to a centralized location.

## Federated Identity
* Delegate authentication to an external identity provider.

## Gatekeeper
* Protect applications and services by using a dedicated host instance that acts as a broker between clients and the application or service, validates and sanitizes requests, and passes requests and data between them.

## Gateway Aggregation
* Use a gateway to aggregate multiple individual requests into a single request.

## Gateway Offloading
* Offload shared or specialized service functionality to a gateway proxy.

## Gateway Routing
* Route requests to multiple services using a single endpoint.

## Geodes
* Deploy backend services into a set of geographical nodes, each of which can service any client request in any region.

## Health Endpoint Monitoring
* Implement functional checks in an application that external tools can access through exposed endpoints at regular intervals.
* [Health Endpoint Monitoring pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/health-endpoint-monitoring)


## Index Table
* Create indexes over the fields in data stores that are frequently referenced by queries.

## Leader Election
* Coordinate the actions performed by a collection of collaborating task instances in a distributed application by electing one instance as the leader that assumes responsibility for managing the other instances.

## Materialized View
* Generate prepopulated views over the data in one or more data stores when the data isn't ideally formatted for required query operations.

## Pipes and Filters
* Break down a task that performs complex processing into a series of separate elements that can be reused.

## Priority Queue
* Prioritize requests sent to services so that requests with a higher priority are received and processed more quickly than those with a lower priority.

## Publisher-Subscriber

* Enable an application to announce events to multiple interested consumers asynchronously, without coupling the senders to the receivers.
* [Publisher-Subscriber pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/publisher-subscriber)

## Queue-Based Load Leveling

* Use a queue that acts as a buffer between a task and a service that it invokes in order to smooth intermittent heavy loads.

## Retry
* Enable an application to handle anticipated, temporary failures when it tries to connect to a service or network resource by transparently retrying an operation that's previously failed.
* [Retry pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/retry)

## Scheduler Agent Supervisor

* Coordinate a set of actions across a distributed set of services and other remote resources.

## Sequential Convoy

* Process a set of related messages in a defined order, without blocking processing of other groups of messages.

## Sharding

* Divide a data store into a set of horizontal partitions or shards.
* [Sharding pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/sharding)

## Sidecar
* Deploy components of an application into a separate process or container to provide isolation and encapsulation.
* [The Sidecar Pattern](https://blog.davemdavis.net/2018/03/13/the-sidecar-pattern/)
* [Sidecar pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/sidecar)

## Static Content Hosting
* Deploy static content to a cloud-based storage service that can deliver them directly to the client.

## Strangler
* Incrementally migrate a legacy system by gradually replacing specific pieces of functionality with new applications and services.

## Throttling
* Control the consumption of resources used by an instance of an application, an individual tenant, or an entire service.

## Valet Key
* Use a token or key that provides clients with restricted direct access to a specific resource or service.
