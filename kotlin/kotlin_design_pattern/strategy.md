# Real World Example

```kotlin
package behavioral

import java.util.Locale

class Printer(private val stringFormatterStrategy: (String) -> String) {
    fun printString(string: String) {
        println(stringFormatterStrategy(string))
    }
}

val lowerCaseFormatter: (String) -> String = { it.lowercase(Locale.getDefault()) }
val upperCaseFormatter: (String) -> String = { it.uppercase(Locale.getDefault()) }

fun main() {
    val inputString = "Hello World"

    val lowerCasePrinter = Printer(lowerCaseFormatter)
    lowerCasePrinter.printString(inputString)

    val upperCasePrinter = Printer(upperCaseFormatter)
    upperCasePrinter.printString(inputString)

    val prefixPrinter = Printer { "Prefix: $it" }
    prefixPrinter.printString(inputString)
}
```
