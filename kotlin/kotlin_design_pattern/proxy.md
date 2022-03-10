```kotlin
package structural

interface File {
    fun read(name: String)
}

class NormalFile : File {
    override fun read(name: String) = println("Reading file: $name")
}

//Proxy:
class SecuredFile : File {
    val normalFile = NormalFile()
    var password: String = ""

    override fun read(name: String) {
        if (password == "secret") {
            println("Password is correct: $password")
            normalFile.read(name)
        } else {
            println("Incorrect password. Access denied!")
        }
    }
}

fun main() {
    val securedFile = SecuredFile()
    securedFile.read("readme.md")

    securedFile.password = "secret"
    securedFile.read("readme.md")
// Output:
//    Incorrect password. Access denied!
//    Password is correct: secret
//    Reading file: readme.md
}

```
