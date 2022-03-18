- [Abstract](#abstract)
- [Materials](#materials)
- [Java Design Patterns](#java-design-patterns)
  - [Architectural](#architectural)
    - [API Gateway](#api-gateway)
    - [Aggregator Microservices](#aggregator-microservices)
    - [CQRS (command query responsibility segregation)](#cqrs-command-query-responsibility-segregation)
    - [Data Bus](#data-bus)
    - [Data Transfer Object](#data-transfer-object)
    - [Event Driven Architecture](#event-driven-architecture)
    - [Event Sourcing](#event-sourcing)
    - [Hexagonal Architecture](#hexagonal-architecture)
    - [Layers](#layers)
    - [Naked Objects](#naked-objects)
    - [Partial Response](#partial-response)
    - [Service Layer](#service-layer)
  - [Behavioral](#behavioral)
    - [Acyclic Visitor](#acyclic-visitor)
    - [Bytecode](#bytecode)
    - [Caching](#caching)
    - [Chain of responsibility](#chain-of-responsibility)
    - [Circuit Breaker](#circuit-breaker)
    - [Command](#command)
    - [Data Locality](#data-locality)
    - [Dirty Flag](#dirty-flag)
    - [Double Buffer](#double-buffer)
    - [Extension objects](#extension-objects)
    - [Feature Toggle](#feature-toggle)
    - [Game Loop](#game-loop)
    - [Intercepting Filter](#intercepting-filter)
    - [Interpreter](#interpreter)
    - [Iterator](#iterator)
    - [Leader Election](#leader-election)
    - [Mediator](#mediator)
    - [Memento](#memento)
    - [Null Object](#null-object)
    - [Observer](#observer)
    - [Parameter Object](#parameter-object)
    - [Partial Response](#partial-response-1)
    - [Pipeline](#pipeline)
    - [Poison Pill](#poison-pill)
    - [Presentation](#presentation)
    - [Priority Queue Pattern](#priority-queue-pattern)
    - [Retry](#retry)
    - [Servant](#servant)
    - [Sharding](#sharding)
    - [Spatial Partition](#spatial-partition)
    - [Special Case](#special-case)
    - [Specification](#specification)
    - [State](#state)
    - [Strategy](#strategy)
    - [Subclass Sandbox](#subclass-sandbox)
    - [Template method](#template-method)
    - [Throttling](#throttling)
    - [Trampoline](#trampoline)
    - [Transaction Script](#transaction-script)
    - [Type-Object](#type-object)
    - [Update Method](#update-method)
    - [Visitor](#visitor)
  - [Business Tier](#business-tier)
    - [Business Delegate](#business-delegate)
  - [Concurrency](#concurrency)
    - [Async Method Invocation](#async-method-invocation)
    - [Balking](#balking)
    - [Double Checked Locking](#double-checked-locking)
    - [Event Queue](#event-queue)
    - [Event-based Asynchronous](#event-based-asynchronous)
    - [Guarded Suspension](#guarded-suspension)
    - [Half-Sync/Half-Async](#half-synchalf-async)
    - [Mutex](#mutex)
    - [Producer Consumer](#producer-consumer)
    - [Promise](#promise)
    - [Reactor](#reactor)
    - [Reader Writer Lock](#reader-writer-lock)
    - [Semaphore](#semaphore)
    - [Thread Local Storage](#thread-local-storage)
    - [Thread Pool](#thread-pool)
  - [Creational](#creational)
    - [Abstract Factory](#abstract-factory)
    - [Builder](#builder)
    - [Converter](#converter)
    - [Dependency Injection](#dependency-injection)
    - [Factory](#factory)
    - [Factory Kit](#factory-kit)
    - [Factory Method](#factory-method)
    - [MonoState](#monostate)
    - [Multiton](#multiton)
    - [Object Mother](#object-mother)
    - [Object Pool](#object-pool)
    - [Property](#property)
    - [Prototype](#prototype)
    - [Registry](#registry)
    - [Singleton](#singleton)
    - [Step Builder](#step-builder)
    - [Value Object](#value-object)
  - [Integration](#integration)
    - [Message Channel](#message-channel)
    - [Publish Subscribe](#publish-subscribe)
    - [Tolerant Reader](#tolerant-reader)
  - [Persistence Tier](#persistence-tier)
    - [Data Access Object](#data-access-object)
    - [Data Mapper](#data-mapper)
    - [Repository](#repository)
  - [Presentation Tier](#presentation-tier)
    - [Flux](#flux)
    - [Front Controller](#front-controller)
    - [Model-View-Controller](#model-view-controller)
    - [Model-View-Presenter](#model-view-presenter)
  - [Structural](#structural)
    - [Abstract Document](#abstract-document)
    - [Adapter](#adapter)
    - [Ambassador](#ambassador)
    - [Bridge](#bridge)
    - [Business Delegate](#business-delegate-1)
    - [Composite](#composite)
    - [Composite Entity](#composite-entity)
    - [Composite View](#composite-view)
    - [Decorator](#decorator)
    - [Delegation](#delegation)
    - [Event Aggregator](#event-aggregator)
    - [Facade](#facade)
    - [Flux](#flux-1)
    - [Flyweight](#flyweight)
    - [Front Controller](#front-controller-1)
    - [Marker Interface](#marker-interface)
    - [Module](#module)
    - [Page Object](#page-object)
    - [Proxy](#proxy)
    - [Role Object](#role-object)
    - [Separated Interface](#separated-interface)
    - [Strangler](#strangler)
    - [Table Module](#table-module)
    - [Twin](#twin)
  - [Testing](#testing)
    - [Page Object](#page-object-1)
  - [Other](#other)
    - [Caching](#caching-1)
    - [Callback](#callback)
    - [Double Dispatch](#double-dispatch)
    - [Execute Around](#execute-around)
    - [Fluent Interface](#fluent-interface)
    - [Lazy Loading](#lazy-loading)
    - [Monad](#monad)
    - [Mute Idiom](#mute-idiom)
    - [Poison Pill](#poison-pill-1)
    - [Private Class Data](#private-class-data)
    - [Queue based load leveling](#queue-based-load-leveling)
    - [Resource Acquisition is Initialization](#resource-acquisition-is-initialization)

----

# Abstract

[Java Design Patterns](http://java-design-patterns.com/) 을 정리한다. 대부분의 pattern 들이 예제와 함께 제공된다.

# Materials

* [Java Design Patterns](https://java-design-patterns.com/)
  * [src](https://github.com/iluwatar/java-design-patterns.git)

# Java Design Patterns

## Architectural

### API Gateway
  - Aggregate calls to microservices in a single location: the API
    Gateway. The user makes a single call to the API Gateway, and the
    API Gateway then calls each relevant microservice.
  - similar to Aggregator Microservices

### Aggregator Microservices
  - The user makes a single call to the Aggregator, and the aggregator
    then calls each relevant microservice and collects the data, apply
    business logic to it, and further publish is as a REST
    Endpoint. More variations of the aggregator are: - Proxy
    Microservice Design Pattern: A different microservice is called
    upon the business need. - Chained Microservice Design Pattern: In
    this case each microservice is dependent/ chained to a series of
    other microservices.
  - similar to APIGateway

### CQRS (command query responsibility segregation)
  - CQRS Command Query Responsibility Segregation - Separate the query
    side from the command side.

### Data Bus
  - Allows send of messages/events between components of an
    application without them needing to know about each other. They
    only need to know about the type of the message/event being sent.
  - similar to mediator, observer, publish/subscribe pattern

### Data Transfer Object

  - Pass data with multiple attributes in one shot from client to
    server, to avoid multiple calls to remote server.

### Event Driven Architecture

  - Send and notify state changes of your objects to other
    applications using an Event-driven Architecture.
  - [What is an Event-Driven Architecture? @ amazon](https://aws.amazon.com/es/event-driven-architecture/)
  - [How to Use Amazon EventBridge to Build Decoupled, Event-Driven Architectures @ amazon](https://pages.awscloud.com/AWS-Learning-Path-How-to-Use-Amazon-EventBridge-to-Build-Decoupled-Event-Driven-Architectures_2020_LP_0001-SRV.html?&trk=ps_a134p000003yBd8AAE&trkCampaign=FY20_2Q_eventbridge_learning_path&sc_channel=ps&sc_campaign=FY20_2Q_EDAPage_eventbridge_learning_path&sc_outcome=PaaS_Digital_Marketing&sc_publisher=Google)

### Event Sourcing

  - Instead of storing just the current state of the data in a domain,
    use an append-only store to record the full series of actions
    taken on that data. The store acts as the system of record and can
    be used to materialize the domain objects. This can simplify tasks
    in complex domains, by avoiding the need to synchronize the data
    model and the business domain, while improving performance,
    scalability, and responsiveness. It can also provide consistency
    for transactional data, and maintain full audit trails and history
    that can enable compensating actions.
  - [eventsourcing & cqrs demo project for springcamp 2017](https://github.com/jaceshim/springcamp2017)
  - [이벤트 소싱 원리와 구현 @ youtube](https://www.youtube.com/watch?v=Yd7TXUdcaUQ)
  - [스프링캠프 2017 [Day2 A2] : 이벤트 소싱 소개 (이론부) @ youtube](https://www.youtube.com/watch?v=TDhknOIYvw4)
    - [스프링캠프 2017 [Day2 A3] : Implementing EventSourcing & CQRS (구현부) @ youtube](https://www.youtube.com/watch?v=12EGxMB8SR8)

### Hexagonal Architecture

  - Allow an application to equally be driven by users, programs,
    automated test or batch scripts, and to be developed and tested in
    isolation from its eventual run-time devices and databases.
  - [Hexagonal Architecture @ TIL](/hexagonalarchitecture/README.md)

### Layers

  - Layers is an architectural style where software responsibilities
    are divided among the different layers of the application.

### Naked Objects

  - The Naked Objects architectural pattern is well suited for rapid
    prototyping. Using the pattern, you only need to write the domain
    objects, everything else is autogenerated by the framework.

### Partial Response

  - Send partial response from server to client on need basis. Client
    will specify the the fields that it need to server, instead of
    serving all details for resource.

### Service Layer

  - Service Layer is an abstraction over domain logic. Typically
    applications require multiple kinds of interfaces to the data they
    store and logic they implement: data loaders, user interfaces,
    integration gateways, and others. Despite their different
    purposes, these interfaces often need common interactions with the
    application to access and manipulate its data and invoke its
    business logic. The Service Layer fulfills this role.

## Behavioral

### Acyclic Visitor

* Allow new functions to be added to existing class hierarchies without affecting those hierarchies, and without creating the troublesome dependency cycles that are inherent to the GoF Visitor Pattern.

### Bytecode

* Allows encoding behavior as instructions for a virtual machine.

### Caching

* The caching pattern avoids expensive re-acquisition of resources by not releasing them immediately after use. The resources retain their identity, are kept in some fast-access storage, and are re-used to avoid having to acquire them again.

### [Chain of responsibility](chainofresp/chainofresp.md)

* Avoid coupling the sender of a request to its receiver by giving more than one object a chance to handle the request. Chain the receiving objects and pass the request along the chain until an object handles it.

### Circuit Breaker

* Handle costly remote service calls in such a way that the failure of a single service/component cannot bring the whole application down, and we can reconnect to the service as soon as possible.

### [Command](command/command.md) 

* Encapsulate a request as an object, thereby letting you parameterize clients with different requests, queue or log requests, and support undoable operations.

### Data Locality

* Accelerate memory access by arranging data to take advantage of CPU caching.

### Dirty Flag

* To avoid expensive re-acquisition of resources. The resources retain their identity, are kept in some fast-access storage, and are re-used to avoid having to acquire them again.

### Double Buffer

* Double buffering is a term used to describe a device that has two buffers. The
  usage of multiple buffers increases the overall throughput of a device and
  helps prevents bottlenecks. This example shows using double buffer pattern on
  graphics. It is used to show one image or frame while a separate frame is
  being buffered to be shown next. This method makes animations and games look
  more realistic than the same done in a single buffer mode.

### Extension objects 

* Anticipate that an object’s interface needs to be extended in the future.
  Additional interfaces are defined by extension objects.

### Feature Toggle 
### Game Loop
### Intercepting Filter 
### [Interpreter](interpreter/interpreter.md) 
### [Iterator](iterator/iterator.md) 
### Leader Election
### [Mediator](mediator/mediator.md) 
### [Memento](memento/memento.md) 
### Null Object 
### [Observer](observer/observer.md)
### Parameter Object 
### Partial Response 
### Pipeline 
### Poison Pill
### Presentation 
### Priority Queue Pattern
### Retry
### Servant 
### Sharding
### Spatial Partition
### Special Case
### Specification
### [State](state/state.md)
### [Strategy](strategy/strategy.md)
### Subclass Sandbox
### [Template method](template/template.md)
### Throttling  
### Trampoline 
### Transaction Script
### Type-Object
### Update Method
### [Visitor](visitor/visitor.md)

## Business Tier

### Business Delegate

  - The Business Delegate pattern adds an abstraction layer between
    presentation and business tiers. By using the pattern we gain
    loose coupling between the tiers and encapsulate knowledge about
    how to locate, connect to, and interact with the business objects
    that make up the application.

## Concurrency

### Async Method Invocation

  - Asynchronous method invocation is pattern where the calling thread
    is not blocked while waiting results of tasks. The pattern
    provides parallel processing of multiple independent tasks and
    retrieving the results via callbacks or waiting until everything
    is done.

### Balking

  - Balking Pattern is used to prevent an object from executing
    certain code if it is an incomplete or inappropriate state

### Double Checked Locking

  - Reduce the overhead of acquiring a lock by first testing the
    locking criterion (the "lock hint") without actually acquiring the
    lock. Only if the locking criterion check indicates that locking
    is required does the actual locking logic proceed.

### Event Queue

  - Event Queue is a good pattern if You have a limited accessibility
    resource (for example: Audio or Database), but You need to handle
    all the requests that want to use that. It puts all the requests
    in a queue and process them asynchronously. Gives the resource for
    the event when it is the next in the queue and in same time
    removes it from the queue.

### Event-based Asynchronous

  - The Event-based Asynchronous Pattern makes available the
    advantages of multithreaded applications while hiding many of the
    complex issues inherent in multithreaded design. Using a class
    that supports this pattern can allow you to:

    - Perform time-consuming tasks, such as downloads and database operations, "in the background," without interrupting your application.
    - Execute multiple operations simultaneously, receiving notifications when each completes.
    - Wait for resources to become available without stopping ("hanging") your application.
    - Communicate with pending asynchronous operations using the familiar events-and-delegates model.

### Guarded Suspension

  - Use Guarded suspension pattern to handle a situation when you want
    to execute a method on object which is not in a proper state.

### Half-Sync/Half-Async

  - The Half-Sync/Half-Async pattern decouples synchronous I/O from
    asynchronous I/O in a system to simplify concurrent programming
    effort without degrading execution efficiency.

### Mutex

  - Mutual Exclusion Lock Binary Semaphore

### Producer Consumer

  - Producer Consumer Design pattern is a classic concurrency pattern
    which reduces coupling between Producer and Consumer by separating
    Identification of work with Execution of Work.

### Promise

  - A Promise represents a proxy for a value not necessarily known
    when the promise is created. It allows you to associate dependent
    promises to an asynchronous action's eventual success value or
    failure reason. Promises are a way to write async code that still
    appears as though it is executing in a synchronous way.

### Reactor

  - The Reactor design pattern handles service requests that are
    delivered concurrently to an application by one or more
    clients. The application can register specific handlers for
    processing which are called by reactor on specific
    events. Dispatching of event handlers is performed by an
    initiation dispatcher, which manages the registered event
    handlers. Demultiplexing of service requests is performed by a
    synchronous event demultiplexer.

### Reader Writer Lock

  - Suppose we have a shared memory area with the basic constraints
    detailed above. It is possible to protect the shared data behind a
    mutual exclusion mutex, in which case no two threads can access
    the data at the same time. However, this solution is suboptimal,
    because it is possible that a reader R1 might have the lock, and
    then another reader R2 requests access. It would be foolish for R2
    to wait until R1 was done before starting its own read operation;
    instead, R2 should start right away. This is the motivation for
    the Reader Writer Lock pattern.

### Semaphore

  - Create a lock which mediates access to a pool of resources. Only a
    limited number of threads, specified at the creation of the
    semaphore, can access the resources at any given time. A semaphore
    which only allows one concurrent access to a resource is called a
    binary semaphore.

### Thread Local Storage

  - Securing variables global to a thread against being spoiled by
    other threads. That is needed if you use class variables or static
    variables in your Callable object or Runnable object that are not
    read-only.

### Thread Pool

  - It is often the case that tasks to be executed are short-lived and
    the number of tasks is large. Creating a new thread for each task
    would make the system spend more time creating and destroying the
    threads than executing the actual tasks. Thread Pool solves this
    problem by reusing existing threads and eliminating the latency of
    creating new threads.

## Creational

### [Abstract Factory](abstractfactory/abstractfactory.md)
### [Builder](builder/builder.md) 
### Converter 
### Dependency Injection 
### Factory 
### Factory Kit
### [Factory Method](factorymethod/factorymethod.md)
### MonoState 
### Multiton 
### Object Mother 
### Object Pool
### Property 
### [Prototype](prototype/prototype.md) 
### Registry 
### [Singleton](singleton/singleton.md) 
### Step Builder 
### Value Object 

## Integration

### Message Channel

  - When two applications communicate using a messaging system they do
   it by using logical addresses of the system, so called Message
   Channels.
 
### Publish Subscribe

  - Broadcast messages from sender to all the interested receivers.
  - similar to observer pattern

### Tolerant Reader

  - Tolerant Reader is an integration pattern that helps creating
    robust communication systems. The idea is to be as tolerant as
    possible when reading data from another service. This way, when
    the communication schema changes, the readers must not break.

## Persistence Tier

### Data Access Object

  - Object provides an abstract interface to some type of database or
    other persistence mechanism.

### Data Mapper

  - A layer of mappers that moves data between objects and a database
    while keeping them independent of each other and the mapper itself

### Repository

  - Repository layer is added between the domain and data mapping
    layers to isolate domain objects from details of the database
    access code and to minimize scattering and duplication of query
    code. The Repository pattern is especially useful in systems where
    number of domain classes is large or heavy querying is utilized.

## Presentation Tier

### Flux

* Flux eschews MVC in favor of a unidirectional data flow. When a user interacts
  with a view, the view propagates an action through a central dispatcher, to
  the various stores that hold the application's data and business logic, which
  updates all of the views that are affected.
* 데이터가 변경되면 단방향으로 뷰를 업데이트한다.

### Front Controller

  - Introduce a common handler for all requests for a web site. This
    way we can encapsulate common functionality such as security,
    internationalization, routing and logging in a single place.

### Model-View-Controller

  - Separate the user interface into three interconnected components:
    the model, the view and the controller. Let the model manage the
    data, the view display the data and the controller mediate
    updating the data and redrawing the display.

### Model-View-Presenter

  - Apply a "Separation of Concerns" principle in a way that allows
    developers to build and test user interfaces.
  - [안드로이드의 MVC, MVP, MVVM 종합 안내서](https://academy.realm.io/kr/posts/eric-maxwell-mvc-mvp-and-mvvm-on-android/)

## Structural

### Abstract Document 
### [Adapter](adapter/adapter.md) 
### Ambassador
### [Bridge](bridge/bridge.md) 
### Business Delegate 
### [Composite](composite/composite.md) 
### Composite Entity
### Composite View
### [Decorator](decorator/decorator.md)
### Delegation 
### Event Aggregator
### [Facade](facade/facade.md) 
### Flux 
### [Flyweight](flyweight/flyweight.md)
### Front Controller 
### Marker Interface 
### Module 
### Page Object 
### [Proxy](proxy/proxy.md) 
### Role Object 
### Separated Interface 
### Strangler
### Table Module
### Twin 

## Testing

### Page Object

  - Page Object encapsulates the UI, hiding the underlying UI widgetry
    of an application (commonly a web application) and providing an
    application-specific API to allow the manipulation of UI
    components required for tests.
  
## Other

### Caching

  - To avoid expensive re-acquisition of resources by not releasing
    the resources immediately after their use. The resources retain
    their identity, are kept in some fast-access storage, and are
    re-used to avoid having to acquire them again.

### Callback

  - Callback is a piece of executable code that is passed as an
    argument to other code, which is expected to call back (execute)
    the argument at some convenient time.

### Double Dispatch

  - Double Dispatch pattern is a way to create maintainable dynamic
    behavior based on receiver and parameter types.

### Execute Around

  - Execute Around idiom frees the user from certain actions that
    should always be executed before and after the business method. A
    good example of this is resource allocation and deallocation
    leaving the user to specify only what to do with the resource.

### Fluent Interface

  - A fluent interface provides an easy-readable, flowing interface,
    that often mimics a domain specific language. Using this pattern
    results in code that can be read nearly as human language.

### Lazy Loading

  - Lazy loading is a design pattern commonly used to defer
    initialization of an object until the point at which it is
    needed. It can contribute to efficiency in the program's operation
    if properly and appropriately used.

### Monad

  - Monad pattern based on monad from linear algebra represents the
    way of chaining operations together step by step. Binding
    functions can be described as passing one's output to another's
    input basing on the 'same type' contract. Formally, monad consists
    of a type constructor M and two operations: bind - that takes
    monadic object and a function from plain object to monadic value
    and returns monadic value return - that takes plain type object
    and returns this object wrapped in a monadic value.

### Mute Idiom

  - Provide a template to suppress any exceptions that either are
    declared but cannot occur or should only be logged; while
    executing some business logic. The template removes the need to
    write repeated try-catch blocks.

### Poison Pill

  - Poison Pill is known predefined data item that allows to provide
    graceful shutdown for separate distributed consumption process.

### Private Class Data

  - Private Class Data design pattern seeks to reduce exposure of
    attributes by limiting their visibility. It reduces the number of
    class attributes by encapsulating them in single Data object.

### Queue based load leveling

  - Use a queue that acts as a buffer between a task and a service
    that it invokes in order to smooth intermittent heavy loads that
    may otherwise cause the service to fail or the task to time
    out. This pattern can help to minimize the impact of peaks in
    demand on availability and responsiveness for both the task and
    the service.

### Resource Acquisition is Initialization

  - Resource Acquisition Is Initialization pattern can be used to
    implement exception safe resource management.
  - similar to execute around pattern
