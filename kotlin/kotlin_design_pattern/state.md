# Real World Example

```kotlin
package behavioral

sealed class AuthorizationState()

object Unauthorized: AuthorizationState()

class Authorized(val userName: String): AuthorizationState()

class AuthorizationPresenter {
    private var state: AuthorizationState = Unauthorized

    val isAuthorized: Boolean
        get() = when (state) {
            is Authorized -> true
            is Unauthorized -> false
        }

    val userName: String
        get() {
            val state = this.state
            return when (state) {
                is Authorized -> state.userName
                is Unauthorized -> "Unknown"
            }
        }

    fun loginUser(userName: String) {
        state = Authorized(userName)
    }

    fun logoutUser() {
        state = Unauthorized
    }

    override fun toString() = "User '$userName' is logged in: $isAuthorized"
}

fun main() {
    val authorizationPresenter = AuthorizationPresenter()

    authorizationPresenter.loginUser("admin")
    println(authorizationPresenter)

    authorizationPresenter.logoutUser()
    println(authorizationPresenter)
// Output:
// User 'admin' is logged in: true
// User 'Unknown' is logged in: false
}
```
