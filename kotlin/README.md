- [Abstract](#abstract)
- [Materials](#materials)
- [Install on windows 10](#install-on-windows-10)
- [Getting Started](#getting-started)
  - [Basic Syntax](#basic-syntax)
  - [Idioms](#idioms)
  - [Formatted String](#formatted-string)
- [Basics](#basics)
  - [Basic types](#basic-types)
  - [Packages and Imports](#packages-and-imports)
  - [Control Flow](#control-flow)
  - [Returns and Jumps](#returns-and-jumps)
- [Classes and Objects](#classes-and-objects)
  - [Classes and Inheritance](#classes-and-inheritance)
  - [Properties and Fields](#properties-and-fields)
  - [Interfaces](#interfaces)
  - [Functional (SAM) Interfaces](#functional-sam-interfaces)
  - [Extensions](#extensions)
  - [Data Classes](#data-classes)
  - [Sealed Classes](#sealed-classes)
  - [Generics](#generics)
  - [Nested Classes](#nested-classes)
  - [Enum Classes](#enum-classes)
  - [Objects](#objects)
  - [Type Aliases](#type-aliases)
  - [Delegation](#delegation)
  - [Delegated Properties](#delegated-properties)
- [Functions and Lambdas](#functions-and-lambdas)
  - [Functions](#functions)
  - [Lambdas](#lambdas)
  - [inline Functions](#inline-functions)
- [Collections](#collections)
  - [Colections Overview](#colections-overview)
- [Advanced](#advanced)
  - [Passing trailing lambdas](#passing-trailing-lambdas)
  - [map vs flatmap](#map-vs-flatmap)
  - [fold](#fold)

----

# Abstract

kotlin 에 대해 정리한다. kotlin 은 종합백화점같다. 없는게 없다. 문서의 완성도가 너무 높아서 대부분의 내용은 link 로 대신한다.

# Materials

* [Kotlin Playground](https://play.kotlinlang.org/#eyJ2ZXJzaW9uIjoiMS40LjEwIiwicGxhdGZvcm0iOiJqYXZhIiwiYXJncyI6IiIsImpzQ29kZSI6IiIsIm5vbmVNYXJrZXJzIjp0cnVlLCJ0aGVtZSI6ImlkZWEiLCJjb2RlIjoiLyoqXG4gKiBZb3UgY2FuIGVkaXQsIHJ1biwgYW5kIHNoYXJlIHRoaXMgY29kZS4gXG4gKiBwbGF5LmtvdGxpbmxhbmcub3JnIFxuICovXG5cbmZ1biBtYWluKCkge1xuICAgIHByaW50bG4oXCJIZWxsbywgd29ybGQhISFcIilcbn0ifQ==)
* [Kotlin Reference](https://kotlinlang.org/docs/reference/)
* [Learning materials overview](https://kotlinlang.org/docs/learning-materials-overview.html)
  * [Basic Syntax @ Kotlin](https://kotlinlang.org/docs/reference/basic-syntax.html)
  * [Idioms @ kotlin](https://kotlinlang.org/docs/idioms.html)
  * [Kotlin Koans @ kotlin](https://kotlinlang.org/docs/koans.html)
  * [Kotlin by example @ kotlin](https://play.kotlinlang.org/byExample/overview?_gl=1*1ch7m8k*_ga*MTU0MzU1NjQ4My4xNjIyNTAwNzUy*_ga_J6T75801PF*MTYyMjUzMDg0OC4yLjEuMTYyMjUzMTg2NS4zMg..&_ga=2.220493660.593975675.1622500752-1543556483.1622500752)
  * [Kotlin books @ kotlin](https://kotlinlang.org/docs/books.html)
  * [Kotlin hands-on tutorials @ kotline](https://play.kotlinlang.org/hands-on/overview?_gl=1*1ch7m8k*_ga*MTU0MzU1NjQ4My4xNjIyNTAwNzUy*_ga_J6T75801PF*MTYyMjUzMDg0OC4yLjEuMTYyMjUzMTg2NS4zMg..&_ga=2.220493660.593975675.1622500752-1543556483.1622500752)

# Install on windows 10

* Install JDK 1.8.
* Install IntelliJ Community Edition.
* Register `C:\Program Files\JetBrains\IntelliJ IDEA Community Edition 2019.2.2\plugins\Kotlin\kotlinc\bin` to `Path`.

```bash
$ kotlinc a.kt -include-runtime -d a.jar
$ java -jar a.jar
```

# Getting Started

## Basic Syntax

* [Basic syntax @ kotlin](https://kotlinlang.org/docs/basic-syntax.html)

## Idioms

* [Idioms @ kotlin](https://kotlinlang.org/docs/idioms.html)

## Formatted String

```kt
val pi = 3.14159265358979323
val fi = "pi = %.2f".format(pi)

println("pi is ${pi}")
println("fi is ${fi}")
```

# Basics

## Basic types

## Packages and Imports

## Control Flow

## Returns and Jumps

# Classes and Objects 

## Classes and Inheritance

## Properties and Fields

## Interfaces

## Functional (SAM) Interfaces

## Extensions

## Data Classes

## Sealed Classes

## Generics

## Nested Classes

## Enum Classes

## Objects

## Type Aliases

## Delegation

## Delegated Properties

# Functions and Lambdas

## Functions

## Lambdas

## inline Functions

# Collections

## Colections Overview

# Advanced

## Passing trailing lambdas

if the last parameter of a function is a function, then a lambda expression passed as the corresponding argument can be placed outside the parentheses:

```kt
val product = items.fold(1) { acc, e -> acc * e }
```

Such syntax is also known as trailing lambda.

If the lambda is the only argument to that call, the parentheses can be omitted entirely:

```kt
run { println("...") }
```

## map vs flatmap

flatMap 의 argument 인 lambda 는 return value 가 iteratable 해야 한다.

```kt
val A  = listOf("A", "B", "C")
val AA = A.map{ "$it!" }
println(AA) // [A!, B!, C!]

val A  = listOf("A", "B", "C")
val AA = A.flatMap{ "$it!".toList() }
println(AA) // [A, !, B, !, C, !]
```

## fold

fold is similar with accumulate from cpp.

```kt
package com.bezkoder.kotlin.fold

fun main(args: Array<String>) {

  println(listOf(1, 2, 3, 4, 5).fold(0) { total, item -> total + item })
  // 15

  println(listOf(1, 2, 3, 4, 5).foldRight(0) { item, total -> total + item })
  // 15

  println(listOf(1, 2, 3, 4, 5).fold(1) { mul, item -> mul * item })
  // 120

  println(listOf(1, 2, 3, 4, 5).foldRight(1) { item, mul -> mul * item })
  // 120

  println(listOf(0, 1, 2, 3, 4, 5)
          .foldIndexed(0) { index, total, item -> if (index % 2 == 0) (total + item) else total })
  // 6

  println(listOf(0, 1, 2, 3, 4, 5)
          .foldRightIndexed(0) { index, item, total -> if (index % 2 == 0) (total + item) else total })
  // 6
}
```
