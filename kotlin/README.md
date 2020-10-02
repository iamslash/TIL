- [Materials](#materials)
- [Basic](#basic)
  - [map vs flatmap](#map-vs-flatmap)
  - [fold](#fold)

----

# Materials

* [kotlin tutorials](https://kotlinlang.org/docs/tutorials/)
  * [Koans](https://play.kotlinlang.org/koans/Introduction/Hello,%20world!/Task.kt)
* [Kotlin Reference](https://kotlinlang.org/docs/reference/)

# Basic

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
