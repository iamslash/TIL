- [Abstract](#abstract)
- [Materials](#materials)
- [Install on windows 10](#install-on-windows-10)
- [Basic](#basic)
  - [Basic Syntax](#basic-syntax)
  - [Idioms](#idioms)
  - [Collections compared to c++](#collections-compared-to-c)
  - [Collections](#collections)
  - [Collection Conversions](#collection-conversions)
  - [Sort](#sort)
  - [min max](#min-max)
  - [Formatted String](#formatted-string)
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

# Basic

## Basic Syntax

* [Basic syntax @ kotlin](https://kotlinlang.org/docs/basic-syntax.html)

## Idioms

* [Idioms @ kotlin](https://kotlinlang.org/docs/idioms.html)

## Collections compared to c++

WIP...

## Collections

* array

* list

* deque

* stack
  
* queue

* set

* map

```kt
// immutable map
val map = mapOf("Vanilla" to 24)
assertEquals(24, map.get("Vanilla"))
assertEquals(24, map["Vanilla"])

// mutable map
val iceCreamSales = mutableMapOf<String, Int>()
iceCreamSales.put("Chocolate", 1)
iceCreamSales["Vanilla"] = 2
iceCreamSales.putAll(setOf("Strawberry" to 3, "Rocky Road" to 2))
iceCreamSales += mapOf("Maple Walnut" to 1, "Mint Chocolate" to 4)
val iceCreamSales = mutableMapOf("Chocolate" to 2)
iceCreamSales.merge("Chocolate", 1, Int::plus)
assertEquals(3, iceCreamSales["Chocolate"])

// Remove entries
val map = mutableMapOf("Chocolate" to 14, "Strawberry" to 9)
map.remove("Strawberry")
map -= "Chocolate"
assertNull(map["Strawberry"])
assertNull(map["Chocolate"])

// Filter
val inventory = mutableMapOf(
  "Vanilla" to 24,
  "Chocolate" to 14,
  "Strawberry" to 9,
)
val lotsLeft = inventory.filterValues { qty -> qty > 10 }
assertEquals(setOf("Vanilla", "Chocolate"), lotsLeft.keys)

// Mapping
val asStrings = inventory.map { (flavor, qty) -> "$qty tubs of $flavor" }
assertTrue(asStrings.containsAll(setOf("24 tubs of Vanilla", "14 tubs of Chocolate", "9 tubs of Strawberry")))
assertEquals(3, asStrings.size)

// forEach
val sales = mapOf("Vanilla" to 7, "Chocolate" to 4, "Strawberry" to 5)
val shipments = mapOf("Chocolate" to 3, "Strawberry" to 7, "Rocky Road" to 5)
with(inventory) {
    sales.forEach { merge(it.key, it.value, Int::minus) }
    shipments.forEach { merge(it.key, it.value, Int::plus) }
}
assertEquals(17, inventory["Vanilla"]) // 24 - 7 + 0
assertEquals(13, inventory["Chocolate"]) // 14 - 4 + 3
assertEquals(11, inventory["Strawberry"]) // 9 - 5 + 7
assertEquals(5, inventory["Rocky Road"]) // 0 - 0 + 5
```

## Collection Conversions

## Sort

* [Guide to Sorting in Kotlin @ baeldung](https://www.baeldung.com/kotlin/sort)

## min max

* [Kotlin pi, 절댓값, 대소 비교 - PI, abs, max, min](https://notepad96.tistory.com/entry/Kotlin-pi-%EC%A0%88%EB%8C%93%EA%B0%92-%EB%8C%80%EC%86%8C-%EB%B9%84%EA%B5%90-PI-abs-max-min)

----

```kt
import kotlin.math.*

fun main(args : Array<String>) {
    var a = 10
    var b = 30
    var c = 40
    println("max val is ${max(a, b)}")    // kotlin.math.max
    println("min val is ${min(c, b)}")   // kotlin.math.min
}
```

## Formatted String

```kt
val pi = 3.14159265358979323
val fi = "pi = %.2f".format(pi)

println("pi is ${pi}")
println("fi is ${fi}")
```

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
