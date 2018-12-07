
<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Abstract](#abstract)
- [Materials](#materials)
- [Usage](#usage)
    - [Collections compared c++ container](#collections-compared-c-container)
    - [Collections](#collections)
    - [Tour](#tour)
    - [Basics](#basics)
    - [Strings and Characters](#strings-and-characters)
    - [Control Flow](#control-flow)
    - [Functions](#functions)
    - [Closures](#closures)
    - [Enumerations](#enumerations)
    - [Structures and Classes](#structures-and-classes)
    - [Properties](#properties)
    - [Methods](#methods)
    - [Subscripts](#subscripts)
    - [Inheritance](#inheritance)
    - [Initialization](#initialization)
    - [Deinitialization](#deinitialization)
    - [Optional Chaining](#optional-chaining)
    - [Error Handling](#error-handling)
    - [Type Casting](#type-casting)
    - [Nested Types](#nested-types)
    - [Extensions](#extensions)
    - [Protocols](#protocols)
    - [Generics](#generics)
    - [Automatic Reference Counting](#automatic-reference-counting)
    - [Memory Safety](#memory-safety)
    - [Access Control](#access-control)
    - [Advanced Operators](#advanced-operators)

<!-- markdown-toc end -->

-------------------------------------------------------------------------------

# Abstract

swift에 대해 정리한다.

# Materials

* [Introduction to Swift (for C#.NET developers)](https://www.jbssolutions.com/introduction-swift-c-net-developers/)
* [swift basic](http://minsone.github.io/mac/ios/swift-the-basic-summary)
* [the swift programming language
swift 4.2](https://docs.swift.org/swift-book/LanguageGuide/TheBasics.html)
* [야곰의 스위프트 기본 문법 강좌](https://www.inflearn.com/course/%EC%8A%A4%EC%9C%84%ED%94%84%ED%8A%B8-%EA%B8%B0%EB%B3%B8-%EB%AC%B8%EB%B2%95/)
  * swift 3 기본 문법

# Usage

## Collections compared c++ container

[Collection Types](https://docs.swift.org/swift-book/LanguageGuide/CollectionTypes.html) 에 의하면 swift 의 collection 은 `Array, Set, Dictionary` 가 있다.

| c++                  | swift                  |
|:---------------------|:----------------------|
| `if, else`           | `if, else`            |
| `for, while`         | `for, while`          |
| `array`              | `Array`               |
| `vector`             | `Array`               |
| `deque`              | ``                    |
| `forward_list`       | ``                    |
| `list`               | ``                    |
| `stack`              | ``                    |
| `queue`              | ``                    |
| `priority_queue`     | ``                    |
| `set`                | ``                    |
| `multiset`           | ``                    |
| `map`                | ``                    |
| `multimap`           | ``                    |
| `unordered_set`      | `Set`                 |
| `unordered_multiset` | ``        |
| `unordered_map`      | `Dictionary`          |
| `unordered_multimap` | ``                    |

## Collections

* Array

* Set

* Dictionary

## Tour

## Basics

## Strings and Characters

## Control Flow

## Functions

## Closures

## Enumerations

## Structures and Classes

## Properties

## Methods

## Subscripts

## Inheritance

* Class

```csharp
// c#
public class HelloWorld {
    private string _message;
     
    HelloWorld(string message) {
        this._message = message;
    }
     
    void display() {
        System.Diagnostics.Debug.WriteLine(_message);
    }
     
    public string Message {
        get {
            return _message;
        }
        set {
            _message = value;
        }
    }
}

// C#
HelloWorld a;
var b = new HelloWorld("hello");
```

```swift
// Swift
public class HelloWorld {
    private var _message: String
     
    init(message: String) {
        self._message = message
    }
     
    func display() {
        debugPrint(_message)
    }
     
    public var Message: String {
        get {
            return _message
        }
        set {
            _message = newValue
        }
    }
}

// Swift
var a : HelloWorld
var b = HelloWorld(message: "hello")
```

## Initialization

## Deinitialization

## Optional Chaining

## Error Handling

## Type Casting

## Nested Types

## Extensions

## Protocols

## Generics

## Automatic Reference Counting

## Memory Safety

## Access Control

## Advanced Operators
