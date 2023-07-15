- [Materials](#materials)
- [Basic](#basic)
  - [Code Smells](#code-smells)
    - [Bloaters](#bloaters)
    - [Object-Orientation Abusers](#object-orientation-abusers)
    - [Change Preventers](#change-preventers)
    - [Dispensables](#dispensables)
    - [Couplers](#couplers)
  - [Refactoring Techniques](#refactoring-techniques)
    - [Composing Methods](#composing-methods)
    - [Moving Features between Objects](#moving-features-between-objects)
    - [Organizing Data](#organizing-data)
    - [Simplifying Conditional Expressions](#simplifying-conditional-expressions)
    - [Simplifying Method Calls](#simplifying-method-calls)
    - [Dealing with Generalization](#dealing-with-generalization)
- [Examples](#examples)

---

# Materials

* [Catalog of Refactoring](https://refactoring.guru/refactoring/catalog)
* [Five Lines of Code | manning](https://www.manning.com/books/five-lines-of-code)
  * Refacotring of [TypeScript](/typescript/README.md)
  * [src](https://github.com/wikibook/five-lines)
  * [homework](https://github.com/wikibook/bomb-guy)

# Basic

## Code Smells

### Bloaters

* Long Method
* Large Class
* Primitive Obsession
* Long Parameter List
* Data Clumps

### Object-Orientation Abusers

* Alternative Classes with Different Interfaces
* Refused Bequest
* Switch Statements
* Temporary Field

### Change Preventers

* Divergent Change
* Parallel Inheritance Hierarchies
* Shotgun Surgery

### Dispensables

* Comments
* Duplicate Code
* Data Class
* Dead Code
* Lazy Class
* Speculative Generality

### Couplers

* Feature Envy
* Inappropriate Intimacy
* Incomplete Library Class
* Message Chains
* Middle Man

## Refactoring Techniques

### Composing Methods

* Extract Method
* Inline Method
* Extract Variable
* Inline Temp
* Replace Temp with Query
* Split Temporary Variable
* Remove Assignments to Parameters
* Replace Method with Method Object
* Substitute Algorithm

### Moving Features between Objects

* Move Method
* Move Field
* Extract Class
* Inline Class
* Hide Delegate
* Remove Middle Man
* Introduce Foreign Method
* Introduce Local Extension

### Organizing Data

* Change Value to Reference
* Change Reference to Value
* Duplicate Observed Data
* Self Encapsulate Field
* Replace Data Value with Object
* Replace Array with Object
* Change Unidirectional Association to Bidirectional
* Change Bidirectional Association to Unidirectional
* Encapsulate Field
* Encapsulate Collection
* Replace Magic Number with Symbolic Constant
* Replace Type Code with Class
* Replace Type Code with Subclasses
* Replace Type Code with State/Strategy
* Replace Subclass with Fields

### Simplifying Conditional Expressions

* Consolidate Conditional Expression
* Consolidate Duplicate Conditional Fragments
* Decompose Conditional
* Replace Conditional with Polymorphism
* Remove Control Flag
* Replace Nested Conditional with Guard Clauses
* Introduce Null Object
* Introduce Assertion

### Simplifying Method Calls

* Add Parameter
* Remove Parameter
* Rename Method
* Separate Query from Modifier
* Parameterize Method
* Introduce Parameter Object
* Preserve Whole Object
* Remove Setting Method
* Replace Parameter with Explicit Methods
* Replace Parameter with Method Call
* Hide Method
  * class 의 method 가 외부에서 사용되지 않는다면 protected/private 으로 제한하라.
* Replace Constructor with Factory Method
* Replace Error Code with Exception
* Replace Exception with Test

### Dealing with Generalization

* Pull Up Field
  * 두 sub class 가 같은 field 를 가지고 있을 때 super class 로 그 field 를 이동하라.
* Pull Up Method
  * 두 sub class 가 같은 method 를 가지고 있을 때 super class 로 그 method 를 이동하라.
* [Pull Up Constructor Body](PullUpConstructorBody.md)
  * sub class constructor 의 일부를 super class constructor 호출로 교체하라.
* Push Down Field
* Push Down Method
* Extract Subclass
* Extract Superclass
* Extract Interface
* Collapse Hierarchy
* Form Template Method
* Replace Inheritance with Delegation
* Replace Delegation with Inheritance

# Examples

* [Refactoring TypeScript](/typescript/refactoring_typescript.md)
* [Refactoring Python](/python/refactoring_python.md)
