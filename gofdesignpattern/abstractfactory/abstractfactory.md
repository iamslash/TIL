# Abstract
  
Abstract Factory 는 Factory 를 생성하고 그 Factory 는 다시 Product 를 생성한다.

최종 Product 종류의 개수는 `Concrete Factory 개수 * Abstract Product 개수` 이다.
예를 들어 `Concrete Factory` 가 `NikeFactory, AdidasFactory` 와 같이 2 개 있다고
해보자. 그리고 `Abstract Product` 가 `Shoe, Shirt, Ball` 과 같이 3 개 있다고
해보자. 최종 Product 종류의 개수는 `NikeShoe, NikeShirt, NikeBall, AdidasShoe,
AdidasShirt, AdidasBall` 과 같이 6 개이다.

# Materials

* [Abstract Factory @ dofactory](https://www.dofactory.com/net/abstract-factory-design-pattern)
* [Abstract Factory @ refactoring.guru](https://refactoring.guru/design-patterns/abstract-factory)

# UML Class Diagram

![](abstract_factory.drawio.png)

# Examples

* [Abstract Factory in C++](/cpp/cpp_gof_designpattern.md#abstract-factory)
* [Abstract Factory in Java](/java/java_gof_designpattern.md#abstract-factory)
* [Abstract Factory in Kotlin](/kotlin/kotlin_gof_design_pattern.md#abstract-factory)
* [Abstract Factory in Python](/python/python_gof_designpattern.md#abstract-factory)
* [Abstract Factory in Go](/go/go_gof_design_pattern.md#abstract-factory)
* [Abstract Factory in Swift](/swift/swift_gof_designpattern.md#abstract-factory)
