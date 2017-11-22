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
  - Builder	(Separates object construction from its representation)
  - Factory Method	(Creates an instance of several derived classes)
  - Prototype	(A fully initialized instance to be copied or cloned)
  - Singleton	(A class of which only a single instance can exist)    

- Structural Pattern
  - Adapter	(Match interfaces of different classes)
  - Bridge	(Separates an object’s interface from its implementation)
  - Composite	(A tree structure of simple and composite objects)
  - Decorator	(Add responsibilities to objects dynamically)
  - Facade	(A single class that represents an entire subsystem)
  - Flyweight	(A fine-grained instance used for efficient sharing)
  - Proxy	(An object representing another object)

- Behaviorial Pattern
  - Chain of Resp.	(A way of passing a request between a chain of objects)
  - Command	(Encapsulate a command request as an object)
  - Interpreter	(A way to include language elements in a program)
  - Iterator	(Sequentially access the elements of a collection)
  - Mediator	(Defines simplified communication between classes)
  - Memento	(Capture and restore an object's internal state)
  - Observer	(A way of notifying change to a number of classes)
  - State	(Alter an object's behavior when its state changes)
  - Strategy	(Encapsulates an algorithm inside a class)
  - Template (Method	Defer the exact steps of an algorithm to a subclass)
  - Visitor	(Defines a new operation to a class without change)
    
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
  - 마이크로 서비스들에 대한 모든 요청들을 모두 Gateway에서 처리하자.
  - Gateway는 요청들의 성격에 따라 관련된 마이크로 서비스들에게 전달하자.

- Aggregator Microservices

- CQRS
  - 읽는 행위와 쓰는 행위를 분리하자.

- Data Bus

- Data Transfer Object

- Event Driven Architecture

- Event Sourcing
  - 사용자의 행위에 해당하는 이벤트들을 모두 저장한다.
  - 예를 들어 장바구니에 품목을 담았을때를 저장하고 품목을 제거했을때를 저장하자.
  - 장바구니의 최종 상태를 알려면 저장된 모든 이벤트를 재생해야 한다. 이벤트가 많아지면 수행속도가 늦을 수 있으니 이벤트가 일정 개수마다 쌓이면 스냅샷을 저장하자.
  - [eventsourcing & cqrs demo project for springcamp 2017](https://github.com/jaceshim/springcamp2017)

- Hexagonal Architecture

- Layers

- Naked Objects

- Partial Response

- Service Layer

## Business Tier

## Concurrency

## Integration

## Persistence Tier

## Presentation Tier

## Testing

- Page Object
  - HTML 을 테스트 하기 위해 HTML page를 추상화 한 것 

## Other

---

# POSA

---

# PLoPD

---

# PLoP
