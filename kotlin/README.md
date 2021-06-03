- [Abstract](#abstract)
- [Materials](#materials)
- [Install on windows 10](#install-on-windows-10)
- [Basic](#basic)
  - [Basic Syntax](#basic-syntax)
  - [Idioms](#idioms)
  - [Collections compared to c++](#collections-compared-to-c)
  - [Collections](#collections)
  - [Collection APIs](#collection-apis)
  - [Collection Conversions](#collection-conversions)
  - [Init Array](#init-array)
  - [Lazy Initialization](#lazy-initialization)
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
* [Kotlin @ baeldung](https://www.baeldung.com/kotlin/)
  * [Kotlin Basics @ baeldung](https://www.baeldung.com/kotlin/category/kotlin)
  * [Collections @ baeldung](https://www.baeldung.com/kotlin/category/kotlin)
  * [Patterns @ baeldung](https://www.baeldung.com/kotlin/category/patterns)

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

| c++                  | java                                   |
| :------------------- | :------------------------------------- |
| `if, else`           | `if, else`                             |
| `for, while`         | `for, while`                           |
| `array`              | `Collections.unmodifiableList`         |
| `vector`             | `java.util.Vector, java.util.ArrayList` |
| `deque`              | `java.util.Deque, java.util.ArrayDeque` |
| `forward_list`       | ``                                     |
| `list`               | `List, MutableList, java.util.LinkedList` |
| `stack`              | `java.util.Stack, java.util.Deque, java.util.ArrayDeque, java.util.LinkedList` |
| `queue`              | `java.util.Queue, java.util.LinkedList` |
| `priority_queue`     | `java.util.Queue, java.util.PriorityQueue` |
| `set`                | `java.util.SortedSet, java.util.TreeSet` |
| `multiset`           | ``                                     |
| `map`                | `java.util.SortedMap, java.util.TreeMap` |
| `multimap`           | ``                                     |
| `unordered_set`      | `Set, MutableSet, java.util.HashSet`  |
| `unordered_multiset` | ``                                     |
| `unordered_map`      | `Map, MutableMap, java.util.HashMap` |
| `unordered_multimap` | ``                                     |

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

## Collection APIs

* [Collections @ kotlin Koans](https://play.kotlinlang.org/koans/Collections/Introduction/Task.kt)

----

kotlin 은 [functional programming](/fp/REAME.md) 의 function combinator 를 통해 간결한 coding 을 할 수 있다.

```kt
// Shop.kt
data class Shop(val name: String, val customers: List<Customer>)
data class Customer(val name: String, val city: City, val orders: List<Order>) {
  override fun toString() = "${name} from ${city.name}"
}
data class Order(val products: List<Product>, val isDelivered: Boolean)
data class Product(val name: String, val price: Double) {
  override fun toString() = "'${name}' for ${price}"
}
data class city(val name: String) {
  override fun toString() = name
}

// Make set
fun Shop.getSetOfCustomers(): Set<Customer> = customers.toSet()

// Sort
fun Shop.getCustomersSortedByNumberOfOrders(): List<Customer> = customers.sortedBy { it.orders.size }

// Filter  map
fun Shop.getCitiesCustomersAreFrom(): Set<City> = customers.map { it.city }.toSet()
fun Shop.getCustomersFrom(city: City): List<Customer> = cusotmers.filter { it.city == city }

// All Any and other predicates

// Max min

// Sum

// Associate

// GroupBy

// Partition

// FlatMap

// Fold

// Compound tasks

// Sequences

// Getting used to new style

```

## Collection Conversions

WIP

## Init Array

* [Initializing Arrays in Kotlin](https://www.baeldung.com/kotlin/initialize-array)

----

```kt
// String array
val strings = arrayOf("January", "February", "March")
// Primitive array
val integers = intArrayOf(1, 2, 3, 4)
// Late Initialize. the array is initialized with nulls
val array = arrayOfNulls<Number>(5)
for (i in array.indices) {
    array[i] = i * i
}
// Generate values with indices
val generatedArray = IntArray(10) { i -> i * i }
val generatedStringArray = Array(10) { i -> "Number of index: $i"  }
```

## Lazy Initialization

```kt
// Init later
lateinit var p: String
p = "Hello"

// Init just once
val q: String by lazy {
    "World"
}
println(p)
println(q)
```

## Sort

* [Guide to Sorting in Kotlin @ baeldung](https://www.baeldung.com/kotlin/sort)
* [Kotlin sortedWith syntax and lambda examples](https://alvinalexander.com/source-code-kotlin-sortedWith-syntax-lambda-examples/)

-----

```kt
// Sort primitive array
val integers = intArrayOf(1, 2, 3, 4)
integers.sort()
integers.sortDescending()

// Sort with comparator
val list = listOf(7,3,5,9,1,3)
list.sortedWith(Comparator<Int>{ a, b ->
    when {
        a > b -> 1
        a < b -> -1
        else -> 0
    }
})
// list: [1, 3, 3, 5, 7, 9]

// Sort string
val names = listOf("kim", "julia", "jim", "hala")
names.sortedWith(Comparator<String>{ a, b ->
    when {
        a > b -> 1
        a < b -> -1
        else -> 0
    }
})
// names: [hala, jim, julia, kim]
// Sort string by length
names.sortedWith(Comparator<String>{ a, b ->
    when {
        a.length > b.length -> 1
        a.length < b.length -> -1
        else -> 0
    }
})
// names: [kim, jim, hala, julia]
```

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
