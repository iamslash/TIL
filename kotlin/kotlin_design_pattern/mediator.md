# Real World Example

```kotlin
package behavioral

class ChatUser(
    private val mediator: ChatMediator,
    private val name: String,
) {

    fun send(msg: String) {
        println("${String.format("%10s", name)}: Sending Message= $msg")
        mediator.sendMessage(msg, this)
    }

    fun receive(msg: String) {
        println("${String.format("%10s", name)}: Message received: $msg")
    }
}

class ChatMediator {

    private val users: MutableList<ChatUser> = ArrayList()

    fun sendMessage(msg: String, user: ChatUser) {
        users
            .filter { it != user }
            .forEach {
                it.receive(msg)
            }
    }

    fun addUser(user: ChatUser): ChatMediator =
        apply { users.add(user) }
}

fun main() {
    val mediator = ChatMediator()
    val john = ChatUser(mediator, "John")

    mediator
        .addUser(ChatUser(mediator, "Alice"))
        .addUser(ChatUser(mediator, "Bob"))
        .addUser(john)
    john.send("Hello World")
// Output:
//    John: Sending Message= Hello World
//    Alice: Message received: Hello World
//    Bob: Message received: Hello World
}
```
