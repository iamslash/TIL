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

----

# Materials

* [Swift Design Pattern | refactoring.guru](https://refactoring.guru/design-patterns/swift)
* [Design Patterns implemented in Swift 5.0 | github](https://github.com/ochococo/Design-Patterns-In-Swift)

# Creational Pattern

## Abstract Factory

Abstract Factory is a design pattern that provides an interface for creating
families of related or dependent objects without specifying their concrete
classes.

```swift
import Foundation

protocol Plant {}

class OrangePlant: Plant {}

class ApplePlant: Plant {}

protocol PlantFactory {
    func makePlant() -> Plant
}

class AppleFactory: PlantFactory {
    func makePlant() -> Plant {
        return ApplePlant()
    }
}

class OrangeFactory: PlantFactory {
    func makePlant() -> Plant {
        return OrangePlant()
    }
}

enum PlantFactoryType {
    case apple
    case orange

    func createFactory() -> PlantFactory {
        switch self {
        case .apple:
            return AppleFactory()
        case .orange:
            return OrangeFactory()
        }
    }
}

func main() {
    let plantFactory = PlantFactoryType.orange.createFactory()
    let plant = plantFactory.makePlant()
    print("Created plant: \(plant)")
}

main()
```

## Builder
## Factory Method
## Prototype
## Singleton
# Structural Pattern
## Adapter
## Bridge
## Composite
## Decorator
## Facade
## Flyweight
## Proxy
# Behaviorial Pattern
## Chain of Resp.
## Command
## Interpreter
## Iterator
## Mediator
## Memento
## Observer
## State
## Strategy
## Template
## Visitor
