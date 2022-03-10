```kotlin
package creational

object PrinterDriver {
    init {
        println("Initializing with object: $this")
    }

    fun print() = println("Printing with object: $this")
}

fun main() {
    println("Start")
    PrinterDriver.print()
    PrinterDriver.print()
// Output:
//    Initializing with object: creational.PrinterDriver@7f690630
//    Printing with object: creational.PrinterDriver@7f690630
//    Printing with object: creational.PrinterDriver@7f690630
}
```
