# Abstract

Kotlin 의 경우 builder pattern, DSL pattern 을 이용하여 객체를 생성하는 것보다
primary constructor 를 이용하는 편이 좋다. [Item 34: Consider a primary constructor with named optional arguments](#item-34-consider-a-primary-constructor-with-named-optional-arguments)

# Real World Example

```kotlin
package creational

import java.io.File

class Dialog {

    fun showTitle() = println("showing title")

    fun setTitle(text: String) = println("setting title text $text")

    fun setTitleColor(color: String) = println("setting title color $color")

    fun showMessage() = println("showing message")

    fun setMessage(text: String) = println("setting message $text")

    fun setMessageColor(color: String) = println("setting message color $color")

    fun showImage(bitmapBytes: ByteArray) = println("showing image with size ${bitmapBytes.size}")

    fun show() = println("showing dialog $this")
}

//Builder:
class DialogBuilder() {
    constructor(init: DialogBuilder.() -> Unit) : this() {
        init()
    }

    private var titleHolder: TextView? = null
    private var messageHolder: TextView? = null
    private var imageHolder: File? = null

    fun title(init: TextView.() -> Unit) {
        titleHolder = TextView().apply { init() }
    }

    fun message(init: TextView.() -> Unit) {
        messageHolder = TextView().apply { init() }
    }

    fun image(init: () -> File) {
        imageHolder = init()
    }

    fun build(): Dialog {
        val dialog = Dialog()

        titleHolder?.apply {
            dialog.setTitle(text)
            dialog.setTitleColor(color)
            dialog.showTitle()
        }

        messageHolder?.apply {
            dialog.setMessage(text)
            dialog.setMessageColor(color)
            dialog.showMessage()
        }

        imageHolder?.apply {
            dialog.showImage(readBytes())
        }

        return dialog
    }

    class TextView {
        var text: String = ""
        var color: String = "#00000"
    }
}

fun main() {
    // Function that creates dialog builder and builds Dialog
    fun dialog(init: DialogBuilder.() -> Unit): Dialog {
        return DialogBuilder(init).build()
    }

    val dialog: Dialog = dialog {
        title {
            text = "Dialog Title"
        }
        message {
            text = "Dialog Message"
            color = "#333333"
        }
        image {
            File.createTempFile("image", "jpg")
        }
    }

    dialog.show()
}
```

# Kotlin Builder class

```kotlin
data class Person private constructor(val builder: Builder) {
    val id: String = builder.id
    val pw: String = builder.pw
    val name: String? 
    val address: String?
    val email: String?

    init {
        name = builder.name
        address = builder.address
        email = builder.email
    }

    class Builder(val id: String, val pw: String) {
        var name: String? = null
        var address: String? = null
        var email: String? = null

        fun build(): Person {
            return Person(this)
        }

        fun name(name: String?): Builder {
            this.name = name
            return this
        }

        fun address(address: String?): Builder {
            this.address = address
            return this
        }

        fun email(email: String?): Builder {
            this.email = email
            return this
        }
    }
}

fun main() {
    val person = Person
        .Builder("AABBCCDD", "123456")
        .name("iamslash")
        .address("Irving Ave")
        .email("iamslash@gmail.com")
        .build()
    println(person)
}
```

# Kotlin @JvmOverloads

* [@JvmOverloads](/kotiln/README.md#jvmoverloads)

```kotlin
data class Person @JvmOverloads constructor(
    val id: String,
    val pw: String,
    var name: String? = "",
    var address: String? = "",
    var email: String? = "",
)

fun main() {
    val person = Person(
        id = "AABBCCDD", 
        pw = "123456",
        name = "iamslash",
        address = "Irving Ave",
        email = "iamslash@gmail.com",
    )
    println(person)
}    
```
