- [Abstract](#abstract)
- [Materials](#materials)
- [Java Design Patterns](#java-design-patterns)
  - [Architectural](#architectural)
    - [API Gateway](#api-gateway)
    - [Aggregator Microservices](#aggregator-microservices)
    - [CQRS (command query responsibility segregation)](#cqrs-command-query-responsibility-segregation)
    - [Data Access Object](#data-access-object)
    - [Data Bus](#data-bus)
    - [Data Mapper](#data-mapper)
    - [Data Transfer Object](#data-transfer-object)
    - [Domain Model](#domain-model)
    - [Event Driven Architecture](#event-driven-architecture)
    - [Event Sourcing](#event-sourcing)
    - [Hexagonal Architecture](#hexagonal-architecture)
    - [Layers](#layers)
    - [Metadata Mapping](#metadata-mapping)
    - [Model-View-Controller](#model-view-controller)
    - [Model-View-Presenter](#model-view-presenter)
    - [Model-View-ViewModel](#model-view-viewmodel)
    - [Naked Objects](#naked-objects)
    - [Repository](#repository)
    - [Serverless](#serverless)
    - [Service Layer](#service-layer)
    - [Service Locator](#service-locator)
    - [Unit Of Work](#unit-of-work)
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
    - [Partial Response](#partial-response)
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
  - [Cloud](#cloud)
    - [Claim Check Pattern](#claim-check-pattern)
    - [Static Content Hosting](#static-content-hosting)
  - [Concurrency](#concurrency)
    - [Active Object](#active-object)
    - [Async Method Invocation](#async-method-invocation)
    - [Balking](#balking)
    - [Commander](#commander)
    - [Event Queue](#event-queue)
    - [Event-based Asynchronous](#event-based-asynchronous)
    - [Guarded Suspension](#guarded-suspension)
    - [Half-Sync/Half-Async](#half-synchalf-async)
    - [Leader/Followers](#leaderfollowers)
    - [Lockable Object](#lockable-object)
    - [Master-Worker](#master-worker)
    - [Monitors](#monitors)
    - [Producer Consumer](#producer-consumer)
    - [Promise](#promise)
    - [Queue based load leveling](#queue-based-load-leveling)
    - [Reactor](#reactor)
    - [Reader Writer Lock](#reader-writer-lock)
    - [Saga](#saga)
    - [Thread Pool](#thread-pool)
    - [Version Number](#version-number)
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
  - [Functional](#functional)
    - [Collection Pipeline](#collection-pipeline)
    - [Filterer](#filterer)
    - [Fluent Interface](#fluent-interface)
    - [Monad](#monad)
  - [Idiom](#idiom)
    - [Arrange/Act/Assert](#arrangeactassert)
    - [Callback](#callback)
    - [Combinator](#combinator)
    - [Double Checked Locking](#double-checked-locking)
    - [Double Dispatch](#double-dispatch)
    - [Execute Around](#execute-around)
    - [Lazy Loading](#lazy-loading)
    - [Mute Idiom](#mute-idiom)
    - [Private Class Data](#private-class-data)
    - [Resource Acquisition Is Initialization](#resource-acquisition-is-initialization)
    - [Thread Local Storage](#thread-local-storage)
  - [Integration](#integration)
    - [EIP Aggregator](#eip-aggregator)
    - [EIP Message Channel](#eip-message-channel)
    - [EIP Publish Subscribe](#eip-publish-subscribe)
    - [EIP Splitter](#eip-splitter)
    - [EIP Wire Tap](#eip-wire-tap)
    - [Fan-Out/Fan-In](#fan-outfan-in)
    - [Tolerant Reader](#tolerant-reader)
  - [Structural](#structural)
    - [Abstract Document](#abstract-document)
    - [Adapter](#adapter)
    - [Ambassador](#ambassador)
    - [Bridge](#bridge)
    - [Business Delegate](#business-delegate)
    - [Composite](#composite)
    - [Composite Entity](#composite-entity)
    - [Composite View](#composite-view)
    - [Decorator](#decorator)
    - [Delegation](#delegation)
    - [Event Aggregator](#event-aggregator)
    - [Facade](#facade)
    - [Flux](#flux)
    - [Flyweight](#flyweight)
    - [Front Controller](#front-controller)
    - [Marker Interface](#marker-interface)
    - [Module](#module)
    - [Page Object](#page-object)
    - [Proxy](#proxy)
    - [Role Object](#role-object)
    - [Separated Interface](#separated-interface)
    - [Strangler](#strangler)
    - [Table Module](#table-module)
    - [Twin](#twin)

----

# Abstract

[Java Design Patterns](http://java-design-patterns.com/) 을 정리한다. 대부분의 pattern 들이 예제와 함께 제공된다. 용어를 정리하는 정도로 해두자. 이해가 가지 않는 부분은 예제코드를 분석한다.

# Materials

* [Java Design Patterns](https://java-design-patterns.com/)
  * [src](https://github.com/iluwatar/java-design-patterns.git)

# Java Design Patterns

## Architectural

### API Gateway
* Aggregate calls to microservices in a single location: the API
  Gateway. The user makes a single call to the API Gateway, and the
  API Gateway then calls each relevant microservice.
* 마이크로서비스들의 요청을 한 곳에서 처리한다.

### Aggregator Microservices
* The user makes a single call to the Aggregator, and the aggregator
  then calls each relevant microservice and collects the data, apply
  business logic to it, and further publish is as a REST
  Endpoint. More variations of the aggregator are: - Proxy
  Microservice Design Pattern: A different microservice is called
  upon the business need. - Chained Microservice Design Pattern: In
  this case each microservice is dependent/ chained to a series of
  other microservices.
* 여러 마이크로서비스들에 요청을 하고 응답을 모아서 넘겨준다.

### CQRS (command query responsibility segregation)

* CQRS Command Query Responsibility Segregation - Separate the query
  side from the command side.
* CUD 와 R 을 구분해서 구현한다.

### Data Access Object

* Object provides an abstract interface to some type of database or
  other persistence mechanism.
* 데이터베이스에 접근하는 것을 추상화해 준다.

### Data Bus

* Allows send of messages/events between components of an
  application without them needing to know about each other. They
  only need to know about the type of the message/event being sent.
* 여러 마이크로서비스과 주고 받는 메시지 혹은 이벤트를 중심으로 데이터를 주고 받는다.

### Data Mapper

* A layer of mappers that moves data between objects and a database while
  keeping them independent of each other and the mapper itself
* 데이터베이스의 데이터와 객체를 변환해 준다.

### Data Transfer Object

* Pass data with multiple attributes in one shot from client to
  server, to avoid multiple calls to remote server.
* 다양한 데이터를 하나의 객체로 모아서 전달한다. 

### Domain Model

* Domain model pattern provides an object-oriented way of dealing with complicated logic. Instead of having one procedure that handles all business logic for a user action there are multiple objects and each of them handles a slice of domain logic that is relevant to it.
* 복잡한 요구사항을 객체지향으로 정규화하여 구현한다.
* [Domain Driven Design](/ddd/README.md)

### Event Driven Architecture

* Event 를 중심으로 서비스하는 아키텍처이다. [msa](/systemdesign/msa.md) 의 경우
  Application 별로 Event 가 정의되야 한다. Producer 는 Event 를
  [kafka](/kafka/README.md) 에 publish 한다. Consumer 는 관심있는 Event 를
  처리한다. Producer 와 Consumer 가 Lossely Coupled 된다. Consumer 에 장애가
  발생해도 Producer 는 Service 에 지장이 없다. Consumer 가 복구되면 밀린 Event
  를 처리하기 때문에 Resilient 하다.
* [What is an Event-Driven Architecture? @ amazon](https://aws.amazon.com/es/event-driven-architecture/)
* [How to Use Amazon EventBridge to Build Decoupled, Event-Driven Architectures @ amazon](https://pages.awscloud.com/AWS-Learning-Path-How-to-Use-Amazon-EventBridge-to-Build-Decoupled-Event-Driven-Architectures_2020_LP_0001-SRV.html?&trk=ps_a134p000003yBd8AAE&trkCampaign=FY20_2Q_eventbridge_learning_path&sc_channel=ps&sc_campaign=FY20_2Q_EDAPage_eventbridge_learning_path&sc_outcome=PaaS_Digital_Marketing&sc_publisher=Google)
* [회원시스템 이벤트기반 아키텍처 구축하기 | woowahan](https://techblog.woowahan.com/7835/)

### Event Sourcing

* Instead of storing just the current state of the data in a domain,
  use an append-only store to record the full series of actions
  taken on that data. The store acts as the system of record and can
  be used to materialize the domain objects. This can simplify tasks
  in complex domains, by avoiding the need to synchronize the data
  model and the business domain, while improving performance,
  scalability, and responsiveness. It can also provide consistency
  for transactional data, and maintain full audit trails and history
  that can enable compensating actions.
* 현재 상태만을 저장하는 것이 아니고 상태가 변경할 때 마다 저장한다.
- [eventsourcing & cqrs demo project for springcamp 2017](https://github.com/jaceshim/springcamp2017)
- [이벤트 소싱 원리와 구현 @ youtube](https://www.youtube.com/watch?v=Yd7TXUdcaUQ)
- [스프링캠프 2017 [Day2 A2] : 이벤트 소싱 소개 (이론부) @ youtube](https://www.youtube.com/watch?v=TDhknOIYvw4)
  - [스프링캠프 2017 [Day2 A3] : Implementing EventSourcing & CQRS (구현부) @ youtube](https://www.youtube.com/watch?v=12EGxMB8SR8)

### Hexagonal Architecture

* Allow an application to equally be driven by users, programs,
  automated test or batch scripts, and to be developed and tested in
  isolation from its eventual run-time devices and databases.
* Port 와 Adapter 로 나누어서 구현한다.
* [Hexagonal Architecture @ TIL](/hexagonalarchitecture/README.md)

### Layers

* Layers is an architectural style where software responsibilities
  are divided among the different layers of the application.
* 여러 레이어로 구분해서 구현한다.
* [Layered Architecture # TIL](/architecture/README.md))

### Metadata Mapping

* Holds details of object-relational mapping in the metadata.
* 객체의 매핑정보를 별도의 파일에 저장한다. annotation 이 더 편하다.

### Model-View-Controller

* Separate the user interface into three interconnected components:
  the model, the view and the controller. Let the model manage the
  data, the view display the data and the controller mediate
  updating the data and redrawing the display.
* 시스템을 model, view, controller 로 구분해서 구현한다.

### Model-View-Presenter

* Apply a "Separation of Concerns" principle in a way that allows
  developers to build and test user interfaces.
* 시스템을 model, view, presenter 로 구분해서 구현한다.
* [안드로이드의 MVC, MVP, MVVM 종합 안내서](https://academy.realm.io/kr/posts/eric-maxwell-mvc-mvp-and-mvvm-on-android/)

### Model-View-ViewModel

* To apply "Separation of Concerns" to separate the logic from the UI components and allow developers to work on UI without affecting the logic and vice versa.
* 시스템을 model, view, viewmodel 로 구분해서 구현한다.
* [[디자인패턴] MVC, MVP, MVVM 비교](https://beomy.tistory.com/43)
  
### Naked Objects

* The Naked Objects architectural pattern is well suited for rapid
  prototyping. Using the pattern, you only need to write the domain
  objects, everything else is autogenerated by the framework.
* 도메인 오브젝트만 제공하면 나머지는 프레임워크가 생성해 준다. 프로토타이핑할 때 좋다.
* [isis](https://github.com/apache/isis)

### Repository

* Repository layer is added between the domain and data mapping
  layers to isolate domain objects from details of the database
  access code and to minimize scattering and duplication of query
  code. The Repository pattern is especially useful in systems where
  number of domain classes is large or heavy querying is utilized.
* 도메인 객체와 데이터 사이의 변환을 담당한다.

### Serverless

* Serverless eliminates the need to plan for infrastructure and let's you focus on your application.
* 인프라스트럭처는 시스템에서 제공해 준다. 비지니스 로직만 구현한다.

### Service Layer

* Service Layer is an abstraction over domain logic. Typically
  applications require multiple kinds of interfaces to the data they
  store and logic they implement: data loaders, user interfaces,
  integration gateways, and others. Despite their different
  purposes, these interfaces often need common interactions with the
  application to access and manipulate its data and invoke its
  business logic. The Service Layer fulfills this role.
* 도메인 로직을 담당하는 레이어이다.

### Service Locator

* Encapsulate the processes involved in obtaining a service with a strong abstraction layer.
* 서비스의 위치를 추상화 한다.

### Unit Of Work

* When a business transaction is completed, all the updates are sent as one big unit of work to be persisted in one go to minimize database round-trips.
* 데이터베이스와 여러번 통신하지 않기 위해 변경된 내역을 하나의 단위로 모은다. 

## Behavioral

### Acyclic Visitor

* Allow new functions to be added to existing class hierarchies without affecting those hierarchies, and without creating the troublesome dependency cycles that are inherent to the GoF Visitor Pattern.
* [GoF Visitor](/designpattern/visitor/visitor.md) 에서 cycle dependency 를 제거한 것이다???

### Bytecode

* Allows encoding behavior as instructions for a virtual machine.
* 인스트럭션을 바이트데이터로 구현한다.

### Caching

* The caching pattern avoids expensive re-acquisition of resources by not releasing them immediately after use. The resources retain their identity, are kept in some fast-access storage, and are re-used to avoid having to acquire them again.
* 한번 읽은 것을 다시 사용한다.

### [Chain of responsibility](/designpattern/chainofresp/chainofresp.md)

* Avoid coupling the sender of a request to its receiver by giving more than one object a chance to handle the request. Chain the receiving objects and pass the request along the chain until an object handles it.
* 형제 클래스에게 임무를 전달한다.

### Circuit Breaker

* Handle costly remote service calls in such a way that the failure of a single service/component cannot bring the whole application down, and we can reconnect to the service as soon as possible.
* 원격 마이크로 서비스에 장애가 발생하면 요청을 보내지 않는다.

### [Command](/designpattern/command/command.md) 

* Encapsulate a request as an object, thereby letting you parameterize clients with different requests, queue or log requests, and support undoable operations.
* 명령을 추상화한다.

### Data Locality

* Accelerate memory access by arranging data to take advantage of CPU caching.
* 자주 사용하는 데이터는 CPU 가 캐싱하기 좋게 모아놓는다.

### Dirty Flag

* To avoid expensive re-acquisition of resources. The resources retain their identity, are kept in some fast-access storage, and are re-used to avoid having to acquire them again.
* 변경이 되었는지 구분할 플래그를 두자. 변경된 경우만 스토리지에 접근한다. 

### Double Buffer

* Double buffering is a term used to describe a device that has two buffers. The
  usage of multiple buffers increases the overall throughput of a device and
  helps prevents bottlenecks. This example shows using double buffer pattern on
  graphics. It is used to show one image or frame while a separate frame is
  being buffered to be shown next. This method makes animations and games look
  more realistic than the same done in a single buffer mode.
* 버퍼를 두개 두어 깜빡임 현상을 제거한다.

### Extension objects 

* Anticipate that an object’s interface needs to be extended in the future.
  Additional interfaces are defined by extension objects.
* 앞으로 추가될 부분을 extension object 로 미리 자리잡아 놓는다.

### Feature Toggle 

* Feature Flag
* 기능을 켜고 끈다.

### Game Loop

* A game loop runs continuously during gameplay. Each turn of the loop, it processes user input without blocking, updates the game state, and renders the game. It tracks the passage of time to control the rate of gameplay.
* 게임의 화면 업데이트를 루프에서 처리한다.

### Intercepting Filter 

* Provide pluggable filters to conduct necessary pre-processing and post-processing to requests from a client to a target
* 요청을 필터를 삽입하여 가로채서 처리한다.

### [Interpreter](/designpattern/interpreter/interpreter.md) 

* Given a language, define a representation for its grammar along with an interpreter that uses the representation to interpret sentences in the language.
* 해석을 추상화한다.

### [Iterator](iterator/iterator.md) 

* Provide a way to access the elements of an aggregate object sequentially without exposing its underlying representation.
* 순회를 추상화한다.

### Leader Election

* Leader Election pattern is commonly used in cloud system design. It can help to ensure that task instances select the leader instance correctly and do not conflict with each other, cause contention for shared resources, or inadvertently interfere with the work that other task instances are performing.
* 컴포넌트들을 리더와 팔로어로 나눈다.

### [Mediator](/designpattern/mediator/mediator.md) 

* Define an object that encapsulates how a set of objects interact. Mediator promotes loose coupling by keeping objects from referring to each other explicitly, and it lets you vary their interaction independently.
* 중재를 추상화한다.

### [Memento](/designpattern/memento/memento.md) 

* Without violating encapsulation, capture and externalize an object's internal state so that the object can be restored to this state later.
* 객체의 상태를 저장하고 불러온다.

### Null Object 

* n most object-oriented languages, such as Java or C#, references may be null. These references need to be checked to ensure they are not null before invoking any methods, because methods typically cannot be invoked on null references. Instead of using a null reference to convey absence of an object (for instance, a non-existent customer), one uses an object which implements the expected interface, but whose method body is empty. The advantage of this approach over a working default implementation is that a Null Object is very predictable and has no side effects: it does nothing.
* 개체가 아직 만들어지지 않은 상태를 표현한다.

### [Observer](observer/observer.md)

* Define a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.
* 구경꾼을 추상화한다.

### Parameter Object 

* The syntax of Java language doesn’t allow you to declare a method with a predefined value for a parameter. Probably the best option to achieve default method parameters in Java is by using the method overloading. Method overloading allows you to declare several methods with the same name but with a different number of parameters. But the main problem with method overloading as a solution for default parameter values reveals itself when a method accepts multiple parameters. Creating an overloaded method for each possible combination of parameters might be cumbersome. To deal with this issue, the Parameter Object pattern is used.
* argument 를 객체로 감싸서 전달한다.

### Partial Response

* Send partial response from server to client on need basis. Client
  will specify the the fields that it need to server, instead of
  serving all details for resource.
* 클라이언트가 일부만 요청하고 서버는 그만큼만 응답으로 준다. 페이징이 이것에 해당한다.

### Pipeline 

* Allows processing of data in a series of stages by giving in an initial input and passing the processed output to be used by the next stages.
* 여러 단계로 구분하여 구현한다.

### Poison Pill

* Poison Pill is known predefined data item that allows to provide graceful shutdown for separate distributed consumption process.
* 신호를 주면 모두 셧다운 한다.

### Presentation 

* Presentation Model pulls the state and behavior of the view out into a model class that is part of the presentation.
* 유저에게 보여주는 레이어이다.

### Priority Queue Pattern

* Prioritize requests sent to services so that requests with a higher priority are received and processed more quickly than those of a lower priority. This pattern is useful in applications that offer different service level guarantees to individual clients.
* 요청을 받으면 우선순위를 고려하여 처리한다.

### Retry

* Transparently retry certain operations that involve communication with external resources, particularly over the network, isolating calling code from the retry implementation details.
* 한번 요청해보고 실패하면 다시 요청한다. 요청을 하는 부분과 반복해서 요청하는 부분은 분리해서 구현한다.

### Servant 

* Servant is used for providing some behavior to a group of classes. Instead of defining that behavior in each class - or when we cannot factor out this behavior in the common parent class - it is defined once in the Servant.
* 클래스단위가 아니고 클래스 그룹단위로 비지니스 로직을 제공한다.

### Sharding

* Sharding pattern means divide the data store into horizontal partitions or shards. Each shard has the same schema, but holds its own distinct subset of the data. A shard is a data store in its own right (it can contain the data for many entities of different types), running on a server acting as a storage node.
* 데이터를 수평적으로 나누어 구현한다.

### Spatial Partition

* As explained in the book Game Programming Patterns by Bob Nystrom, spatial partition pattern helps to efficiently locate objects by storing them in a data structure organized by their positions.
* 지역을 나누어서 관리한다. 쿼드트리가 이것에 해당한다.

### Special Case

* Define some special cases, and encapsulates them into subclasses that provide different special behaviors.
* 자식 클래스를 정의하여 특별한 행동을 부여한다.

### Specification

* Specification pattern separates the statement of how to match a candidate, from the candidate object that it is matched against. As well as its usefulness in selection, it is also valuable for validation and for building to order.
* ???

### [State](/designpattern/state/state.md)

* Allow an object to alter its behavior when its internal state changes. The object will appear to change its class.
* 상태를 추상화한다.

### [Strategy](/designpattern/strategy/strategy.md)

* Define a family of algorithms, encapsulate each one, and make them interchangeable. Strategy lets the algorithm vary independently from the clients that use it.
* 전략을 추상화한다.

### Subclass Sandbox

* The subclass sandbox pattern describes a basic idea, while not having a lot of detailed mechanics. You will need the pattern when you have several similar subclasses. If you have to make a tiny change, then change the base class, while all subclasses shouldn't have to be touched. So the base class has to be able to provide all of the operations a derived class needs to perform.
* 기능을 추가할 때 subclass 는 수정하지 않고 baseclass 만 수정한다.

### [Template method](/designpattern/template/template.md)

* Define the skeleton of an algorithm in an operation, deferring some steps to subclasses. Template Method lets subclasses redefine certain steps of an algorithm without changing the algorithm's structure.
* 추상 클래스의 함수에서 구현 클래스의 함수들을 절차에 맞게 호출한다. 구현 클래스에서 단위 절차에 해당하는 함수들을 구현한다. 절차가 수정되면 추상 클래스만 수정하면 된다.

### Throttling  

* Ensure that a given client is not able to access service resources more than the assigned limit.
* 제한된 요청의 횟수만 처리한다. 서비스의 장애전파를 막을 수 있다.

### Trampoline 

* Trampoline pattern is used for implementing algorithms recursively in Java without blowing the stack and to interleave the execution of functions without hard coding them together.
* 재귀 호출을 콜 스택을 넘치지 않게 구현한다??? recursive 를 iterative 하게 구현한다는 의미???

### Transaction Script

* Transaction Script organizes business logic by procedures where each procedure handles a single request from the presentation.
* ???

### Type-Object

* As explained in the book Game Programming Patterns by Robert Nystrom, type object pattern helps in
* ???

### Update Method

* Update method pattern simulates a collection of independent objects by telling each to process one frame of behavior at a time.
* 하나의 프레임이 변경되면 모든 오브젝트의 `Update()` 를 호출한다.

### [Visitor](/designpattern/visitor/visitor.md)

* Represent an operation to be performed on the elements of an object structure. Visitor lets you define a new operation without changing the classes of the elements on which it operates.
* receiver 와 argument 를 보고 수행할 비지니스 로직을 결정한다.

## Cloud

### Claim Check Pattern

* Reduce the load of data transfer through the Internet. Instead of sending actual data directly, just send the message reference.
* Improve data security. As only message reference is shared, no data is exposed to the Internet.
* 데이터의 레퍼런스를 교환한다. 보안이 강화된다.

### Static Content Hosting

* Deploy static content to a cloud-based storage service that can deliver them directly to the client. This can reduce the need for potentially expensive compute instances.
* 정적 컨텐트는 CDN 을 통해 제공하자.

## Concurrency

### Active Object

* The active object design pattern decouples method execution from method invocation for objects that each reside in their thread of control. The goal is to introduce concurrency, by using asynchronous method invocation, and a scheduler for handling requests.
* 쓰레드와 큐를 갖는 객체이다. 객체에게 요청을 보내면 큐에 저장되고 객체의 쓰레드가 하나씩 처리한다.
* [go](/golang/README.md) 로 구현한다면 struct 에 channel 과 그 channel 에서 요청을 하나씩 처리하는 go routine 을 추가한다. struct instance 가 곧 Actibe Object 이다.

### Async Method Invocation

* Asynchronous method invocation is pattern where the calling thread
  is not blocked while waiting results of tasks. The pattern
  provides parallel processing of multiple independent tasks and
  retrieving the results via callbacks or waiting until everything
  is done.
* 비동기로 함수를 호출한다.

### Balking

* Balking Pattern is used to prevent an object from executing
  certain code if it is an incomplete or inappropriate state
* 객체의 상태가 부적절하면 특정 영역을 수행하지 않는다.

### Commander

* Used to handle all problems that can be encountered when doing distributed transactions.
* [distributed transactions](/distributedsystem/README.md) 처리를 하다가 만나는 문제들을 해결한다.

### Event Queue

* Event Queue is a good pattern if You have a limited accessibility resource (for example: Audio or Database), but You need to handle all the requests that want to use that. It puts all the requests in a queue and process them asynchronously. Gives the resource for the event when it is the next in the queue and in same time removes it from the queue.
* 큐를 하나 두고 이벤트를 보낸다. 큐에서 이벤트를 꺼내거 처리한다. 큐를 통해 쓰로틀링된다. 제한된 접근을 제공하는 리소스들을 처리할 수 있다.

### Event-based Asynchronous

* The Event-based Asynchronous Pattern makes available the advantages of multithreaded applications while hiding many of the complex issues inherent in multithreaded design.
* 멀티 쓰레드 애플리케이션에서 서로 다른 쓰레드가 이벤트를 주고 받으면서 처리한다.

### Guarded Suspension

* Use Guarded suspension pattern to handle a situation when you want
  to execute a method on object which is not in a proper state.
* 객체가 적절한 상태가 아니라면 객체의 메쏘드를 호출하지 않는다.

### Half-Sync/Half-Async

* The Half-Sync/Half-Async pattern decouples synchronous I/O from
  asynchronous I/O in a system to simplify concurrent programming
  effort without degrading execution efficiency.
* 비지니스 로직을 동기화, 비동기화를 구분해서 구현한다.

### Leader/Followers

* The Leader/Followers pattern provides a concurrency model where multiple threads can efficiently de-multiplex events and dispatch event handlers that process I/O handles shared by the threads.
* ???

### Lockable Object

* The lockable object design pattern ensures that there is only one user using the target object. Compared to the built-in synchronization mechanisms such as using the synchronized keyword, this pattern can lock objects for an undetermined time and is not tied to the duration of the request.
* 쓰레드 안정적인 객체.

### Master-Worker

* Used for centralised parallel processing.
* 마스터가 일을 워커에게 나누어준다.

### Monitors

* Monitor pattern is used to create thread-safe objects and prevent conflicts between threads in concurrent applications.
* 쓰레드 안정적인 객체. Lockable object 와 무슨차이???

### Producer Consumer

* Producer Consumer Design pattern is a classic concurrency pattern
  which reduces coupling between Producer and Consumer by separating
  Identification of work with Execution of Work.
* 생산자는 이벤트를 생성하여 큐에 넣고 소비자는 큐에서 이벤트를 소비한다.

### Promise

* A Promise represents a proxy for a value not necessarily known
  when the promise is created. It allows you to associate dependent
  promises to an asynchronous action's eventual success value or
  failure reason. Promises are a way to write async code that still
  appears as though it is executing in a synchronous way.
* 결과가 정해지면 비지니스를 수행하는 비동기 객체.

### Queue based load leveling

* Use a queue that acts as a buffer between a task and a service that it invokes in order to smooth intermittent heavy loads that may otherwise cause the service to fail or the task to time out. This pattern can help to minimize the impact of peaks in demand on availability and responsiveness for both the task and the service.
* ???

### Reactor

* The Reactor design pattern handles service requests that are
  delivered concurrently to an application by one or more
  clients. The application can register specific handlers for
  processing which are called by reactor on specific
  events. Dispatching of event handlers is performed by an
  initiation dispatcher, which manages the registered event
  handlers. Demultiplexing of service requests is performed by a
  synchronous event demultiplexer.
* 이벤트가 도착하면 등록된 이벤트핸들러가 처리한다.

### Reader Writer Lock

* Suppose we have a shared memory area with the basic constraints
  detailed above. It is possible to protect the shared data behind a
  mutual exclusion mutex, in which case no two threads can access
  the data at the same time. However, this solution is suboptimal,
  because it is possible that a reader R1 might have the lock, and
  then another reader R2 requests access. It would be foolish for R2
  to wait until R1 was done before starting its own read operation;
  instead, R2 should start right away. This is the motivation for
  the Reader Writer Lock pattern.
* 쓰레드 안정적으로 읽고 쓸 수 있는 잠금 객체.

### Saga

* This pattern is used in distributed services to perform a group of operations atomically. This is an analog of transaction in a database but in terms of microservices architecture this is executed in a distributed environment
* [distributed transaction](/distributedsystem/README.md) 을 처리한다.

### Thread Pool

* It is often the case that tasks to be executed are short-lived and
  the number of tasks is large. Creating a new thread for each task
  would make the system spend more time creating and destroying the
  threads than executing the actual tasks. Thread Pool solves this
  problem by reusing existing threads and eliminating the latency of
  creating new threads.
* 쓰레드들을 풀에 담아두고 재활용한다.

### Version Number

* Resolve concurrency conflicts when multiple clients are trying to update same entity simultaneously.
* 데이터마다 버전을 부여하고 여러 쓰레드가 안정적으로 접근할 수 있다.
* 낙관적 잠금이라고도 한다.
* [systemdesign - Optimistic Lock vs Pessimistic Lock](/systemdesign/README.md#optimistic-lock-vs-pessimistic-lock)

## Creational

### [Abstract Factory](/designpattern/abstractfactory/abstractfactory.md)

* Provide an interface for creating families of related or dependent objects without specifying their concrete classes.
* 팩토리 메쏘드를 갖는 팩토리를 생성한다.

### [Builder](/designpattern/builder/builder.md) 

* Separate the construction of a complex object from its representation so that the same construction process can create different representations.
* 객체 생성 과정을 추상화 한다.

### Converter 

* The purpose of the Converter pattern is to provide a generic, common way of bidirectional conversion between corresponding types, allowing a clean implementation in which the types do not need to be aware of each other. Moreover, the Converter pattern introduces bidirectional collection mapping, reducing a boilerplate code to minimum.
* 두 객체 사이를 변환한다. DTO 를 Entity 로 변환하고 Entity 를 DTO 로 변환한다.

### Dependency Injection 

* Dependency Injection is a software design pattern in which one or more dependencies (or services) are injected, or passed by reference, into a dependent object (or client) and are made part of the client's state. The pattern separates the creation of a client's dependencies from its own behavior, which allows program designs to be loosely coupled and to follow the inversion of control and single responsibility principles.
* 객체를 생성해서 주입한다.

### Factory 

* Providing a static method encapsulated in a class called the factory, to hide the implementation logic and make client code focus on usage rather than initializing new objects.
* 객체를 생성하는 클래스.

### Factory Kit

* Define a factory of immutable content with separated builder and factory interfaces.
* ???

### [Factory Method](/designpattern/factorymethod/factorymethod.md)

* Define an interface for creating an object, but let subclasses decide which class to instantiate. Factory Method lets a class defer instantiation to subclasses.
* 객체 생성을 추상화한 메쏘드.

### MonoState 

* Enforces a behaviour like sharing the same state amongst all instances.
* 모든 객체들이 하나의 상태를 공유한다.

### Multiton 

* Ensure a class only has a limited number of instances and provide a global point of access to them.
* 제한된 숫자의 객체를 제공한다.

### Object Mother 

* Define a factory of immutable content with separated builder and factory interfaces.
* ???

### Object Pool

* When objects are expensive to create and they are needed only for short periods of time it is advantageous to utilize the Object Pool pattern. The Object Pool provides a cache for instantiated objects tracking which ones are in use and which are available.
* 객체를 풀에 담아두고 재활용한다.

### Property 

* Create hierarchy of objects and new objects using already existing objects as parents.
* ???

### [Prototype](/designpattern/prototype/prototype.md) 

* Specify the kinds of objects to create using a prototypical instance, and create new objects by copying this prototype.
* 복사에 의해 객체를 생성한다.

### Registry 

* Stores the objects of a single class and provide a global point of access to them. Similar to Multiton pattern, only difference is that in a registry there is no restriction on the number of objects.
* 객체를 하나 등록해 놓고 사용한다. 복사하여 사용한다???

### [Singleton](singleton/singleton.md) 

* Ensure a class only has one instance, and provide a global point of access to it.
* 객체는 하나만 존재한다.

### Step Builder 

* An extension of the Builder pattern that fully guides the user through the creation of the object with no chances of confusion. The user experience will be much more improved by the fact that he will only see the next step methods available, NO build method until is the right time to build the object.
* ???

### Value Object 

* Provide objects which follow value semantics rather than reference semantics. This means value objects' equality is not based on identity. Two value objects are equal when they have the same value, not necessarily being the same object.
* 값만 가지고 있는 객체를 말한다. 레퍼런스는 가지고 있지 않다. 소유한 값들이 같으면 두 객체는 같다.  

## Functional

### Collection Pipeline

* Collection Pipeline introduces Function Composition and Collection Pipeline, two functional-style patterns that you can combine to iterate collections in your code. In functional programming, it's common to sequence complex operations through a series of smaller modular functions or operations. The series is called a composition of functions, or a function composition. When a collection of data flows through a function composition, it becomes a collection pipeline. Function Composition and Collection Pipeline are two design patterns frequently used in functional-style programming.
* ???

### Filterer

* The intent of this design pattern is to introduce a functional interface that will add a functionality for container-like objects to easily return filtered versions of themselves.
* ???

### Fluent Interface

* A fluent interface provides an easy-readable, flowing interface, that often mimics a domain specific language. Using this pattern results in code that can be read nearly as human language.
* ???

### Monad

* Monad pattern based on monad from linear algebra represents the
  way of chaining operations together step by step. Binding
  functions can be described as passing one's output to another's
  input basing on the 'same type' contract. Formally, monad consists
  of a type constructor M and two operations: bind - that takes
  monadic object and a function from plain object to monadic value
  and returns monadic value return - that takes plain type object
  and returns this object wrapped in a monadic value.
* ???

## Idiom

### Arrange/Act/Assert

* `Arrange/Act/Assert (AAA)` is a pattern for organizing unit tests. It breaks tests down into three clear and distinct steps:

  * Arrange: Perform the setup and initialization required for the test.
  * Act: Take action(s) required for the test.
  * Assert: Verify the outcome(s) of the test.
* 테스트를 구현할 때 `Arrange/Act/Assert` 로 구분해서 구현한다.

### Callback

* Callback is a piece of executable code that is passed as an argument to other code, which is expected to call back (execute) the argument at some convenient time.
* 함수를 인자로 다른 함수에게 넘겨준다.

### Combinator

* The functional pattern representing a style of organizing libraries centered around the idea of combining functions.
Putting it simply, there is some type T, some functions for constructing "primitive" values of type T, and some "combinators" which can combine values of type T in various ways to build up more complex values of type T.
* 함수를 합성한다.

### Double Checked Locking

* Reduce the overhead of acquiring a lock by first testing the
  locking criterion (the "lock hint") without actually acquiring the
  lock. Only if the locking criterion check indicates that locking
  is required does the actual locking logic proceed.
* 락을 획득하기 전에 잠겼는지 살펴본다.

### Double Dispatch

* Double Dispatch pattern is a way to create maintainable dynamic behavior based on receiver and parameter types.
* receiver, parameter type 을 보고 수행할 비지니스 로직을 선택한다.
* [visitor](/designpattern/visitor/visitor.md)

### Execute Around

* Execute Around idiom frees the user from certain actions that should always be executed before and after the business method. A good example of this is resource allocation and deallocation leaving the user to specify only what to do with the resource.
* 특정 영역을 수행하기 전에 수행하고 수행하고 나서 수행한다.

### Lazy Loading

* Lazy loading is a design pattern commonly used to defer initialization of an object until the point at which it is needed. It can contribute to efficiency in the program's operation if properly and appropriately used.
* 꼭 필요할 때 로딩한다.

### Mute Idiom

* Provide a template to suppress any exceptions that either are declared but cannot occur or should only be logged; while executing some business logic. The template removes the need to write repeated try-catch blocks.
* ???

### Private Class Data

* Private Class Data design pattern seeks to reduce exposure of attributes by limiting their visibility. It reduces the number of class attributes by encapsulating them in single Data object.
* 멤버 변수를 노출하지 않는다.

### Resource Acquisition Is Initialization

* Resource Acquisition Is Initialization pattern can be used to implement exception safe resource management.
* 리소스를 블록에서 할당 했다면 블록에서 벗어날 때 해제한다. 

### Thread Local Storage

* Securing variables global to a thread against being spoiled by
  other threads. That is needed if you use class variables or static
  variables in your Callable object or Runnable object that are not
  read-only.
* 쓰레드별로 스토리지를 소유한다.

## Integration

### EIP Aggregator

* Sometimes in enterprise systems there is a need to group incoming data in order to process it as a whole. For example you may need to gather offers and after defined number of offers has been received you would like to choose the one with the best parameters.
* EIP stands for Enterprise Integration Pattern
* 요청을 모아서 분류별로 나눈다.

### EIP Message Channel

* When two applications communicate using a messaging system they do it by using logical addresses of the system, so called Message Channels.
* 메시지를 주고 받는 통로이다.

### EIP Publish Subscribe

* Broadcast messages from sender to all the interested receivers.
* 메시지를 구독자들에게 발송한다.

### EIP Splitter

* It is very common in integration systems that incoming messages consists of
  many items bundled together. For example an invoice document contains multiple
  invoice lines describing transaction (quantity, name of provided service/sold
  goods, price etc.). Such bundled messages may not be accepted by other
  systems. This is where splitter pattern comes in handy. It will take the whole
  document, split it based on given criteria and send individual items to the
  endpoint.
* 하나로 모여서 도착한 요청을 잘게 나눈다.

### EIP Wire Tap

* In most integration cases there is a need to monitor the messages flowing through the system. It is usually achieved by intercepting the message and redirecting it to a different location like console, filesystem or the database. It is important that such functionality should not modify the original message and influence the processing path.
* 메시지를 중간에 가로채기 해서 검사한후 다시 보낸다.

### Fan-Out/Fan-In

* The pattern is used when a source system needs to run one or more long-running processes that will fetch some data. The source will not block itself waiting for the reply.
* 하나의 루틴에서 여러 루틴으로 나누어 진다. 여러 루틴에서 하나의 루틴으로 모아진다.

### Tolerant Reader

* Tolerant Reader is an integration pattern that helps creating robust communication systems. The idea is to be as tolerant as possible when reading data from another service. This way, when the communication schema changes, the readers must not break.
* 하나의 서비스가 다른 서비스와 데이터를 주고 받을 때 프로토콜을 하위 호환성을 지키며 변경한다.

## Structural

### Abstract Document

* Use dynamic properties and achieve flexibility of untyped languages while keeping type-safety.
* ???

### [Adapter](/designpattern/adapter/adapter.md)

* Convert the interface of a class into another interface the clients expect. Adapter lets classes work together that couldn't otherwise because of incompatible interfaces.
* 맞지 않는 부분을 맞추어 준다.

### Ambassador

* Provide a helper service instance on a client and offload common functionality away from a shared resource.
* ???

### [Bridge](/designpattern/bridge/bridge.md) 

* Decouple an abstraction from its implementation so that the two can vary independently.
* 구현과 추상을 분리한다.
* `Handle/body`

### Business Delegate

* The Business Delegate pattern adds an abstraction layer between
  presentation and business tiers. By using the pattern we gain
  loose coupling between the tiers and encapsulate knowledge about
  how to locate, connect to, 
  and interact with the business objects
  that make up the application.
* ???

### [Composite](/designpattern/composite/composite.md)

* Compose objects into tree structures to represent part-whole hierarchies. Composite lets clients treat individual objects and compositions of objects uniformly.
* 트리구조를 추상화한다.

### Composite Entity

* It is used to model, represent, and manage a set of persistent objects that are interrelated, rather than representing them as individual fine-grained entities.
* ???

### Composite View

* The purpose of the Composite View Pattern is to increase re-usability and flexibility when creating views for websites/webapps. This pattern seeks to decouple the content of the page from its layout, allowing changes to be made to either the content or layout of the page without impacting the other. This pattern also allows content to be easily reused across different views easily.
* ???

### [Decorator](/designpattern/decorator/decorator.md)

* Attach additional responsibilities to an object dynamically. Decorators provide a flexible alternative to subclassing for extending functionality.
* 서브클래싱 없이 기능을 추가한다.

### Delegation 

* It is a technique where an object expresses certain behavior to the outside but in reality delegates responsibility for implementing that behaviour to an associated object.
* 요청을 특정 객체에게 전달한다. [proxy](/designpattern/proxy/proxy.md) 와 비슷한다.

### Event Aggregator

* A system with lots of objects can lead to complexities when a client wants to subscribe to events. The client has to find and register for each object individually, if each object has multiple events then each event requires a separate subscription. An Event Aggregator acts as a single source of events for many objects. It registers for all the events of the many objects allowing clients to register with just the aggregator.
* 다양한 이벤트를 처리하고 하나로 모아 구독자에게 전달한다. 

### [Facade](/designpattern/facade/facade.md) 

* Provide a unified interface to a set of interfaces in a subsystem. Facade defines a higher-level interface that makes the subsystem easier to use.
* 복잡한 것들을 단순하게 제공한다.

### Flux

* Flux eschews MVC in favor of a unidirectional data flow. When a user interacts
  with a view, the view propagates an action through a central dispatcher, to
  the various stores that hold the application's data and business logic, which
  updates all of the views that are affected.
* 데이터가 변경되면 단방향으로 뷰를 업데이트한다.

### [Flyweight](/designpattern/flyweight/flyweight.md)

* Use sharing to support large numbers of fine-grained objects efficiently.
* 객체를 많이 만들어놓고 공유한다.

### Front Controller

* Introduce a common handler for all requests for a web site. This
  way we can encapsulate common functionality such as security,
  internationalization, routing and logging in a single place.
* ???

### Marker Interface 

* Using empty interfaces as markers to distinguish special treated objects.
* 깡통 인터페이스.

### Module 

* Module pattern is used to implement the concept of software modules, defined by modular programming, in a programming language with incomplete direct support for the concept.
* 모듈화 한다.

### Page Object 

* Page Object encapsulates the UI, hiding the underlying UI widgetry of an application (commonly a web application) and providing an application-specific API to allow the manipulation of UI components required for tests. In doing so, it allows the test class itself to focus on the test logic instead.
* 페이지를 구현한 객체.

### [Proxy](/designpattern/proxy/proxy.md)

* Provide a surrogate or placeholder for another object to control access to it.
* 노출되지 않은 형제 객체에게 요청을 전달한다.

### Role Object 

* Adapt an object to different client’s needs through transparently attached role objects, each one representing a role the object has to play in that client’s context. The object manages its role set dynamically. By representing roles as individual objects, different contexts are kept separate and system configuration is simplified.
* ???

### Separated Interface 

* Separate the interface definition and implementation in different packages. This allows the client to be completely unaware of the implementation.
* 인터페이스의 정의와 구현을 각각 다른 패키지로 관리한다.

### Strangler

* Incrementally migrate a legacy system by gradually replacing specific pieces of functionality with new applications and services. As features from the legacy system are replaced, the new system eventually covers all the old system's features and may has its own new features, then strangling the old system and allowing you to decommission it.
* 조금씩 마이그레이션 한다.

### Table Module

* Table Module organizes domain logic with one class per table in the database, and a single instance of a class contains the various procedures that will act on the data.
* 하나의 테이블을 하나의 클래스로 구현한다.

### Twin 

* Twin pattern is a design pattern which provides a standard solution to simulate multiple inheritance in java
* ???
