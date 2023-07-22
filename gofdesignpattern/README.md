- [Materials](#materials)
- [Creational Pattern](#creational-pattern)
    - [Abstract Factory](#abstract-factory)
    - [Builder](#builder)
    - [Factory Method](#factory-method)
    - [Prototype](#prototype)
    - [Singleton](#singleton)
- [Structural Pattern](#structural-pattern)
    - [Adapter](#adapter)
    - [Bridge](#bridge)
    - [Composite](#composite)
    - [Decorator](#decorator)
    - [Facade](#facade)
    - [Flyweight](#flyweight)
    - [Proxy](#proxy)
- [Behaviorial Pattern](#behaviorial-pattern)
    - [Chain of Resp.](#chain-of-resp)
    - [Command](#command)
    - [Interpreter](#interpreter)
    - [Iterator](#iterator)
    - [Mediator](#mediator)
    - [Memento](#memento)
    - [Observer](#observer)
    - [State](#state)
    - [Strategy](#strategy)
    - [Template](#template)
    - [Visitor](#visitor)
- [Q\&A](#qa)

----

# Materials

* [GOF Pattern](http://www.dofactory.com/net/design-patterns)

# Creational Pattern

### [Abstract Factory](abstractfactory/abstractfactory.md)
### [Builder](builder/builder.md)
### [Factory Method](factorymethod/factorymethod.md)
### [Prototype](prototype/prototype.md)
### [Singleton](singleton/singleton.md)

# Structural Pattern

### [Adapter](adapter/adapter.md)
### [Bridge](bridge/bridge.md)
### [Composite](composite/composite.md)
### [Decorator](decorator/decorator.md)
### [Facade](facade/facade.md)
### [Flyweight](flyweight/flyweight.md)
### [Proxy](proxy/proxy.md)

# Behaviorial Pattern

### [Chain of Resp.](chainofresp/chainofresp.md)
### [Command](command/command.md)
### [Interpreter](interpreter/interpreter.md)
### [Iterator](iterator/iterator.md)
### [Mediator](mediator/mediator.md)
### [Memento](memento/memento.md)
### [Observer](observer/observer.md)
### [State](state/state.md)
### [Strategy](strategy/strategy.md)
### [Template](template/template.md)
### [Visitor](visitor/visitor.md)

# Q&A

- **Factory Method vs Abstract Factory difference???**
  - Factory Method 는 동일한 분류의 객체를 생성할 때 사용한다. Abstract Factory
    는 다양한 분류의 객체를 생성할 때 사용한다. Abstract Factory 는 두개 이상의
    Factory Method 를 소유한다.
- **Proxy vs Adapter difference???**
  - Proxy class 는 wrapping 하고 싶은 class 와 형제관계이다. Adapter class 는
    wrapping 하고 싶은 class 와 형제관계가 아니다.
- **Decorator is better than Subclassing???**
  - [decorator @ TIL](decorator.md)
  - IFruit 를 상속받은 Apple, Orange 가 있다고 해보자. 과일판매기능을 추가하고
    싶다. Subclassing 을 이용한다면 AppleSellable, OrangeSellable 를 각각 구현해
    주어야 한다. 그러나 Decorator 를 이용하면 하나의 ConcreteDecorator Sellable
    를 추가하여 더욱 간단히 구현할 수 있다.
- **Bridge vs Strategy difference???**
  - Strategy 는 behavioral pattern 이다. 인스턴스 교체를 통해서 runtime 에
    동작이 달라진다. Bridge pattern 은 structural pattern 이다. 추상과 구현이
    분리된 구조이다. Strategy 는 추상과 구현이 분리되어 있지 않다. 따라서
    Strategy 의 추상과 구현의 결합도가 Bridge 보다 높다.
- **Strategy vs Visitor difference???**
  - Strategy 는 `1:many` 관계를 추상화한다. Visitor 는 `many:many` 관계를
    추상화한다. 
  - Strategy 는 Single Dispatch 를 이용한 것이고 Visitor 는 Double Dispatch 를
    이용한 것이다.
  - 예를 들어 Strategy class `Video` 를 상속받은 Concrete Strategy class
    `MpegCompression, AviCompression, QuickTimeCompression` 가 있다고 해보자. 
  - 시간이 지나 Audio 압축을 지원하고자 한다. `MpegVideoCompression,
    MpegAudioCompression` 과 같이 중복해서 Concrete Strategy class 를 추가하는
    것보다는 Visitor interface `IVisitor`  를 상속받은 Concrete Visitor
    `MpegCompression` 을 이용하는 것이 더욱 간단하다. 다음은 `MpegCompression`
    의 vistor method 이다.

    ```java
    MpegCompression::compressVideo(Video object)
    MpegCompression::compressAudio(Audio object)
    ```     
- **Strategy vs State difference???**
  - Strategy 와 State 는 매우 비슷하다. State 는 Concrete State 들 끼리 서로
    변신하는 상태전환의 개념을 포함한다. Strategy 는 Concrete Strategy 들 끼리
    서로 관계가 없다.
  - The strategy pattern focuses on selecting an algorithm or a strategy to
    perform a task, while the state pattern focuses on changing the behavior of
    an object based on its internal state.
