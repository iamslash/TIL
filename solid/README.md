# Abstract

로버트 마틴이 정리한 객체 지향 프로그래밍 및 설계의 다섯 가지 기본 원칙을 마이클 페더스가 두문자어 기억술로 소개한 것이다.

# Materials

* [SOLID](https://johngrib.github.io/wiki/SOLID/)
* [SOLID 원칙](https://dev-momo.tistory.com/entry/SOLID-%EC%9B%90%EC%B9%99)
* [SOLID & IoC Principles](https://www.slideshare.net/PavloHodysh/solid-ioc-principles-66628530)

# Basic

| | | Title  | Description |
|--|--|--|--|
| S | SRP | 단일 책임 원칙 (Single responsibility principle) |  한 클래스는 하나의 책임만 가져야 한다. |
| O | OCP | 개방-폐쇄 원칙 (Open/closed principle) | 소프트웨어 요소는 확장에는 열려 있으나 변경에는 닫혀 있어야 한다. |
| L | LSP | 리스코프 치환 원칙 (Liskov substitution principle) | 프로그램의 객체는 프로그램의 정확성을 깨뜨리지 않으면서 하위 타입의 인스턴스로 바꿀 수 있어야 한다. |
| I | ISP | 인터페이스 분리 원칙 (Interface segregation principle) | 특정 클라이언트를 위한 인터페이스 여러 개가 범용 인터페이스 하나보다 낫다. |
| D | DIP | 의존관계 역전 원칙 (Dependency inversion principle) | 추상화에 의존해야지, 구체화에 의존하면 안된다. |

# SRP

A class should have only **one reason** to change. by Robert C. Martin.

# OCP

Objects or entitles should be open for **extension**, but closed for modification. Bertrand Meyer.

# LSP

Let f(x) be a property of objects X of type T. Then f(y) should be true for objects Y of type S where S is a subtype of T. Barbara Liskov.

"is-a" 관계가 성립되지 않으면 `S` 는 `T` 의 자식이 아니다.

# ISP

**Many** client-specific **interfaces** are better than one general-purpose interface. Rober C. Martin.

# DIP

One should depend upon **abstractions**, not on conretions. Rober C. Martin.
