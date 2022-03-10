# Real World Example

```kotlin
package behavioral

interface OrderCommand {
    fun execute()
}

class OrderAddCommand(val id: Long): OrderCommand {
    override fun execute() {
        println("Adding order with id: $id")
    }
}

class OrderPaycommand(val id: Long): OrderCommand {
    override fun execute() {
        println("Paying for order with id: $id")
    }
}

class CommandProcessor {
    private val queue = ArrayList<OrderCommand>()

    fun addToQueue(orderCommand: OrderCommand): CommandProcessor {
        return apply {
            queue.add(orderCommand)
        }
    }

    fun processCommands(): CommandProcessor {
        return apply {
            queue.forEach { it.execute() }
            queue.clear()
        }
    }
}

fun main() {
    CommandProcessor()
        .addToQueue(OrderAddCommand(1L))
        .addToQueue(OrderAddCommand(2L))
        .addToQueue(OrderPaycommand(2L))
        .addToQueue(OrderPaycommand(1L))
        .processCommands()
// Output:
// Adding order with id: 1
// Adding order with id: 2
// Paying for order with id: 2
// Paying for order with id: 1
}
```
