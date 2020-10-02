- [Materials](#materials)
- [Basic](#basic)
  - [map vs flatmap](#map-vs-flatmap)

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
