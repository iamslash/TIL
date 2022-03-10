# Real World Example

```kotlin
package behavioral

interface HeadersChain {
    fun addHeader(inputHeader: String): String
}

class AuthenticationHeader(
    private val token: String?,
    var next: HeadersChain? = null,
): HeadersChain {

    override fun addHeader(inputHeader: String): String {
        token ?: throw IllegalStateException("Token should be not null")
        return inputHeader + "Authorization: Bearer $token\n"
            .let { next?.addHeader(it) ?: it }
    }
}

class ContentTypeHeader(
    private val contentType: String,
    var next: HeadersChain? = null,
): HeadersChain {

    override fun addHeader(inputHeader: String): String {
        return inputHeader + "ContentType: $contentType\n"
            .let { next?.addHeader(it) ?: it }
    }
}

class BodyPayload(
    val body: String,
    var next: HeadersChain? = null
): HeadersChain {

    override fun addHeader(inputHeader: String): String {
        return inputHeader + "$body"
            .let { next?.addHeader(it) ?: it }
    }
}

fun main() {
    // Create chain elements
    val authenticationHeader = AuthenticationHeader("123456")
    val contentTypeHeader = ContentTypeHeader("json")
    val messageBody = BodyPayload("Body:\n{\n\"username\"=\"iamslash\"\n}")

    // Construct chain
    authenticationHeader.next = contentTypeHeader
    contentTypeHeader.next = messageBody

    // Execute chain
    val messageWithAuthentication =
        authenticationHeader.addHeader("Headers with Authentication:\n")
    println(messageWithAuthentication)

    val messageWithoutAuth =
        contentTypeHeader.addHeader("Headers:\n")
    println(messageWithoutAuth)

// Output:
//    Headers with Authentication:
//    Authorization: Bearer 123456
//    ContentType: json
//    Body:
//    {
//        "username"="iamslash"
//    }
//    Headers:
//    ContentType: json
//    Body:
//    {
//        "username"="iamslash"
//    }
}
```
