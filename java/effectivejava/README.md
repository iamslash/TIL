- [Materials](#materials)
- [Creating and Destroying Objects](#creating-and-destroying-objects)
- [Methods Common to All Objects](#methods-common-to-all-objects)
- [Classes and Interfaces](#classes-and-interfaces)
  - [Item 18: Prefer interfaces to abstract classes](#item-18-prefer-interfaces-to-abstract-classes)
- [Generics](#generics)
  - [Item 33: Consider typesafe heterogeneous containers](#item-33-consider-typesafe-heterogeneous-containers)
- [Enums and Annotations](#enums-and-annotations)
  - [Item 34: Use enum instead of int constants](#item-34-use-enum-instead-of-int-constants)
- [Lambdas and Streams](#lambdas-and-streams)
- [Methods](#methods)
- [General Programming](#general-programming)
- [Exceptions](#exceptions)
- [Concurrency](#concurrency)
- [Serialization](#serialization)

-----

# Materials

* [Effective Java 3/E Study @ github](https://github.com/19F-Study/effective-java)
* [[책] Joshua Bloch, Effective Java 3rd Edition, 2018, Addison-Wesley (1) @ medium](https://myeongjae.kim/blog/2020/06/28/effective-java-3rd-1)
  * [[책] Joshua Bloch, Effective Java 3rd Edition, 2018, Addison-Wesley (2) @ medium](https://myeongjae.kim/blog/2020/06/28/effective-java-3rd-2)
* [Effective Java 3/E 정리](https://medium.com/@ddt1984/effective-java-3-e-%EC%A0%95%EB%A6%AC-c3fb43eec9d2)
* [『이펙티브 자바 3판』(원서: Effective Java 3rd Edition) @ github](https://github.com/WegraLee/effective-java-3e-source-code)
  * [Effective Java, Third Edition](https://github.com/jbloch/effective-java-3e-source-code) 

# Creating and Destroying Objects

# Methods Common to All Objects

# Classes and Interfaces

## Item 18: Prefer interfaces to abstract classes

Inheritance 보다는 Composition 을 사용해야 [SOLID](/solid/README.md) 의 OCP (Open Closed Principal) 을 지킬 수 있다.

[decorate pattern @ TIL](/designpattern/decorator.md) 은 Item 18 을 지키는 아주 좋은 예이다.

# Generics

## Item 33: Consider typesafe heterogeneous containers

> [item33 @ github](https://github.com/jbloch/effective-java-3e-source-code/tree/master/src/effectivejava/chapter5/item33)

# Enums and Annotations

## Item 34: Use enum instead of int constants

> [item34 @ github](https://github.com/jbloch/effective-java-3e-source-code/tree/master/src/effectivejava/chapter6/item34)

# Lambdas and Streams

# Methods

# General Programming

# Exceptions

# Concurrency

# Serialization
