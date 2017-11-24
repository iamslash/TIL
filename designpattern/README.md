# Abstract

프로그래밍에 관련된 패턴들에 대해 정리한다.

# Materials

* [패턴으로 가는길, 손영수](http://www.devpia.com/MAEUL/Contents/Detail.aspx?BoardID=70&MAEULNO=28&no=187&page=1)
  *  패턴 공부 로드맵
* [GOF pattern]()
* [POSA]()
* [PLoPD, pattern language of program design](http://wiki.c2.com/?PatternLanguagesOfProgramDesign)
  * 패턴에 관련된 컨퍼런스인 PLoP의 논문들을 정리한 책
* [Pattern-Oriented Analysis and Design: Composing Patterns to Design Software Systems](http://www.kangcom.com/sub/view.asp?sku=200309010011)
* [Remoting Patterns: Foundations of Enterprise, Internet and Realtime Distributed Object Middleware](http://www.kangcom.com/sub/view.asp?sku=200410040309)
* [Xunit Test Patterns: Refactoring Test Code](http://www.kangcom.com/sub/view.asp?sku=200612280010)
* [Patterns for Fault Tolerant Software](http://www.kangcom.com/sub/view.asp?sku=200712160009)
* [Patterns of Enterprise Application Architecture, Martin Fowler](http://www.kangcom.com/sub/view.asp?sku=200212100028)
* [Enterprise Integration Patterns : Designing, Building, and Deploying Messaging Solutions](http://www.kangcom.com/sub/view.asp?sku=200310160006)
* [Real-Time Design Patterns: Robust Scalable Architecture for Real-Time Systems](http://www.kangcom.com/sub/view.asp?sku=200403300020)
* [Refactoring to Patterns](http://www.kangcom.com/sub/view.asp?sku=200406140003)
* [Architecting Enterprise Solutions: Patterns for High-Capability Internet-Based Systems](http://www.kangcom.com/sub/view.asp?sku=200410040307)
* [PLoP papers](http://www.hillside.net/index.php/past-plop-conferences)
* [클로저 디자인 패턴](http://clojure.or.kr/docs/clojure-and-gof-design-patterns.html)
  * java로 표현한 디자인 패턴은 clojure로 이렇게 간단히 된다.

# References

* [Design patterns implemented in Java](http://java-design-patterns.com/)
* [.NET Design Patterns](http://dofactory.com/net/design-patterns)

---

# [GOF Pattern](http://www.dofactory.com/net/design-patterns)
  
- Creational Pattern
  - Abstract Factory	(Creates an instance of several families of classes)
    - strategy of related creating object
  - Builder	(Separates object construction from its representation)
    - selective arguments
  - Factory Method	(Creates an instance of several derived classes)
    - strategy of creating object
  - Prototype	(A fully initialized instance to be copied or cloned)
  - Singleton	(A class of which only a single instance can exist)    

- Structural Pattern
  - Adapter	(Match interfaces of different classes)
    - wrapper, various type, same feature
  - Bridge	(Separates an object’s interface from its implementation)
  - Composite	(A tree structure of simple and composite objects)
    - tree
  - Decorator	(Add responsibilities to objects dynamically)
    - wrapper with same type, new feature
  - Facade	(A single class that represents an entire subsystem)
  - Flyweight	(A fine-grained instance used for efficient sharing)
    - cache
  - Proxy	(An object representing another object)
    - wrapper, composition of function

- Behaviorial Pattern
  - Chain of Resp.	(A way of passing a request between a chain of objects)
    - composition of function
  - Command	(Encapsulate a command request as an object)
    - function
  - Interpreter	(A way to include language elements in a program)
    - functions which process tree
  - Iterator	(Sequentially access the elements of a collection)
    - sequence
  - Mediator	(Defines simplified communication between classes)
  - Memento	(Capture and restore an object's internal state)
    - save and restore
  - Observer	(A way of notifying change to a number of classes)
    - a function which is called after other function is called
  - State	(Alter an object's behavior when its state changes)
    - Strategy pattern which depends on states
  - Strategy	(Encapsulates an algorithm inside a class)
    - a function which receives arguments
  - Template (Method	Defer the exact steps of an algorithm to a subclass)
    - Stategy pattern which includes default values
  - Visitor	(Defines a new operation to a class without change)
    - multi dispath
    
---

# [Game Programming Pattern](http://gameprogrammingpatterns.com/contents.html)

- Sequencing Pattern
  - Double Buffer
  - Game Loop
  - Update Method
- Behavioral Pattern
  - Bytecode
  - Subclass Sandbox
  - Type Object
- Decoupling Pattern
  - Component
  - Event Queue
  - Service Locator
- Optimization Pattern
  - Data Locality
  - Dirty Flag
  - Object Pool
  - Spatial Partition

---

# [Design patterns implemented in Java](http://java-design-patterns.com/)

## Architectural

- API Gateway
  - Aggregate calls to microservices in a single location: the API
    Gateway. The user makes a single call to the API Gateway, and the
    API Gateway then calls each relevant microservice.
  - similar to Aggregator Microservices

- Aggregator Microservices
  - The user makes a single call to the Aggregator, and the aggregator
    then calls each relevant microservice and collects the data, apply
    business logic to it, and further publish is as a REST
    Endpoint. More variations of the aggregator are: - Proxy
    Microservice Design Pattern: A different microservice is called
    upon the business need. - Chained Microservice Design Pattern: In
    this case each microservice is dependent/ chained to a series of
    other microservices.
  - similar to APIGateway

- CQRS (command query responsibility segregation
  - CQRS Command Query Responsibility Segregation - Separate the query
    side from the command side.

- Data Bus
  - Allows send of messages/events between components of an
    application without them needing to know about each other. They
    only need to know about the type of the message/event being sent.
  - similar to mediator, observer, publish/subscribe pattern

- Data Transfer Object

  - Pass data with multiple attributes in one shot from client to
    server, to avoid multiple calls to remote server.

- Event Driven Architecture

  - Send and notify state changes of your objects to other
    applications using an Event-driven Architecture.

- Event Sourcing

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

- Hexagonal Architecture

  - Allow an application to equally be driven by users, programs,
    automated test or batch scripts, and to be developed and tested in
    isolation from its eventual run-time devices and databases.

- Layers

- Naked Objects

- Partial Response

- Service Layer

## Behavioral

## Business Tier

## Concurrency

## Creational

## Integration

## Persistence Tier

## Presentation Tier

## Structural

## Testing

- Page Object
  - Page Object encapsulates the UI, hiding the underlying UI widgetry
    of an application (commonly a web application) and providing an
    application-specific API to allow the manipulation of UI
    components required for tests.
  
## Other

---

# POSA

---

# PLoPD

---

# PLoP
