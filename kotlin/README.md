- [Materials](#materials)
- [Getting Started](#getting-started)
  - [Basic Syntax](#basic-syntax)
  - [Idioms](#idioms)
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

# Materials

* [Kotlin Playground](https://play.kotlinlang.org/#eyJ2ZXJzaW9uIjoiMS40LjEwIiwicGxhdGZvcm0iOiJqYXZhIiwiYXJncyI6IiIsImpzQ29kZSI6IiIsIm5vbmVNYXJrZXJzIjp0cnVlLCJ0aGVtZSI6ImlkZWEiLCJjb2RlIjoiLyoqXG4gKiBZb3UgY2FuIGVkaXQsIHJ1biwgYW5kIHNoYXJlIHRoaXMgY29kZS4gXG4gKiBwbGF5LmtvdGxpbmxhbmcub3JnIFxuICovXG5cbmZ1biBtYWluKCkge1xuICAgIHByaW50bG4oXCJIZWxsbywgd29ybGQhISFcIilcbn0ifQ==)
* [Kotlin Reference](https://kotlinlang.org/docs/reference/)
* [Basic Syntax @ Kotlin](https://kotlinlang.org/docs/reference/basic-syntax.html)

# Getting Started

## Basic Syntax

```kt
///////////////////////////////////////////////////////
// Pckage definition and imports
package com.iamslash.demo
import kotlin.text.*

///////////////////////////////////////////////////////
// Program entry point
fun main() {
  println("Hello Wrold")
}

///////////////////////////////////////////////////////
// Functions
fun sum(a: Int, b: Int) Int {
  return a + b
}
// functions with an expression body and inferred return type
fun sum(a: Int, b: Int) = a + b
fun printSum(a: Int, b: Int): Unit {
  println("sum of $a and $b is ${a + b}")
}
// Unit return type can be omitted
fun printSum(a: Int, b: Int) {
  println("sum of $a and $b is ${a + b}")
}

///////////////////////////////////////////////////////
// Variables
val a: Int = 1
val b = 2
val c: Int
c = 3

var x = 5
x += 1

val PI = 3.14
var x = 0
fun incrementX() {
  x += 1
}

///////////////////////////////////////////////////////
// Comments
// This is an end-of-line comment

/* This is a block comment
   on multiple lines. */
// Block commnets in Kotlin can be nested   
/* The comment starts here
/* contains a nested comment */     
and ends here. */

///////////////////////////////////////////////////////
// String templates
var a = 1
val s1 = "a is $a"
a = 2
val s2 = "${s1.replace("is", "was")}, bu tnow is $a"

///////////////////////////////////////////////////////
// Conditional expressions
fun maxOf(a: Int, b: Int): Int {
  if (a > b) {
    return a
  } else {
    return b
  }
}
// if can also be as an expression
fun maxOf(a: Int, b: Int) = if (a > b) a else b

///////////////////////////////////////////////////////
// Nullable value and null checks
fun parseInt(str: String): Int? {
  //...
}
fun printProduct(arg1: String, arg2: String) {
  val x = parseInt(arg1)
  val y = parseInt(arg2)
  if (x != null && y != null) {
    println(x * y)
  } else {
    println("'$arg1' or '$arg2' is not a number")
  }
}
//...
if (x == null) {
  println("Wrong number format in arg1: '$arg1'")
  return
}
if (y == null) {
  println("Wrong number format in arg2: '$arg2'")
  return
}
println(x * y)

///////////////////////////////////////////////////////
// Type checks and automatic casts
fun getStringLength(obj: Any): Int? {
  if (obj is String) {
    // 'obj' is automatically cast to 'String' 
    return obj.length
  }
  return null
}

func getStringLength(obj: Any): Int? {
  if (obj !is String)
    return null
  return obj.length
}

fun getStringLength(obj: Any): Int? {
  if (obj is String && obj.length > 0) {
    return obj.length
  }
  return null
}

///////////////////////////////////////////////////////
// for loop
val items = listOf("apple", "banana", "kiwifruit")
for (item in items) {
  println(item)
}

///////////////////////////////////////////////////////
// while loop
val items = listOf("apple", "banana", "kiwifruit")
var index = 0
while (index < items.size) {
  print("item at $index is ${items[index]}")
  index++
}

///////////////////////////////////////////////////////
// when expression
fun describe(obj: Any): String =
  when (obj) {
    1          -> "One"
    "Hello"    -> "Greeting"
    is Long    -> "Long"
    !is String -> "Not a string"
    else       -> "Unknown"
  }

///////////////////////////////////////////////////////
// Ranges
val x = 10
val y = 9
if (x in 1..y+1) {
  println("fits in range")
}

val list = listOf("a", "b", "c")
if (-1 !in 0..list.lastIndex) {
  println("-1 is out of range")
}
if (list.size !in list.indices) {
  println("list size is out of valid list indices range, too")
}

for (x in 1..5) {
  print(x)
}

for (x in 1..10 step 2) {
  print(x)
}
println()
for (x in 9 downTo 0 step 3) {
  print(x)
}

///////////////////////////////////////////////////////
// Collections
for (item in items) {
  println(item)
}
when {
  "orange" in items -> println("juicy")
  "apple" in items -> println("apple is fine too")
}
val fruits = listOf("banana", "avocado", "apple", "kiwifruit")
fruits
  .filter { it.startsWith("a") }
  .sortedBy { it }
  .map { it.toUpperCase() }
  .forEach { println(it) }

///////////////////////////////////////////////////////
// Creating basic classes and their intances
val rectangle = Rectangle(5.0, 2.0)
val triangle = Triangle(3.0, 4.0, 5.0)
```

## Idioms

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
