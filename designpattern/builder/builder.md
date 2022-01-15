- [Abstract](#abstract)
- [Materials](#materials)
- [Concept Class Diagram](#concept-class-diagram)
- [Real World Examples](#real-world-examples)
  - [Go](#go)
  - [Java](#java)
    - [Java Buildable interface](#java-buildable-interface)
    - [Java Lombok](#java-lombok)
  - [Kotlin](#kotlin)
    - [Kotlin Builder class](#kotlin-builder-class)
    - [Kotlin @JvmOverloads](#kotlin-jvmoverloads)

---

# Abstract

* 생성 절차를 다양하게 하여 타겟 오브젝트 인스턴스를 생성한다.
* Director 는 생성절차를 다양하게 호출할 수 있다. 생성절차를 Builder 안으로 포함한다면 Factory Method 와 다를게 없다.

constructor 만으로 object 를 생성하는 것은 다음과 같은 이유 때문에 불편하다. 그러나 Builder pattern 을 사용하면 이러한 불편함을 해결할 수 있다. 

* constructor 의 argument 가 너무 많을 때 readibility 가 떨어진다.
* constructor 의 argument 중 일부만 사용할 때 불편하다. 예를 들면 사용을 원치않는 argument 는 null 을 전달해야 한다. 

# Materials

* [Builder @ dofactory](https://www.dofactory.com/net/builder-design-pattern)
* [[Design Pattern] Builder Pattern](https://beomseok95.tistory.com/240)
* [Builder @ refactoringguru](https://refactoring.guru/design-patterns/builder)

# Concept Class Diagram

![](builder.png)

# Real World Examples

## Go

* [Builder by go](/golang/designpattern/builder.md)

## Java

### Java Buildable interface

```java
public interface Buildable {
    T build();
}

public class Person {

    private String id;
    private String pw;
    private String name;
    private String address;
    private String email;

    private Person(Builder builder) {
        this.id = builder.id;
        this.pw = builder.pw;
        this.name = builder.name;
        this.address = builder.address;
        this.email = builder.email;
    }

    public String getId() {
        return id;
    }

    public String getPw() {
        return pw;
    }

    public String getName() {
        return name;
    }

    public String getAddress() {
        return address;
    }

    public String getEmail() {
        return email;
    }


    public static class Builder implements Buildable {
        private final String id;
        private final String pw;
        private String name;
        private String address;
        private String email;

        @Override
        public Person build() {
            return new Person(this);
        }

        public Builder(String id, String pw) {
            this.id = id;
            this.pw = pw;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder address(String address) {
            this.address = address;
            return this;
        }

        public Builder email(String email) {
            this.email = email;
            return this;
        }
    }
}

Person person = new Person.Builder("AABBCCDD", "123456")
                          .name("iamslash")
                          .address("Irving Ave")
                          .email("iamslash@gmail.com")
                          .build();
```

### Java Lombok

```java
@Builder
public class Person {

    private String id;
    private String pw;
    private String name;
    private String address;
    private String email;

}

Person person = new Person.builder("AABBCCDD", "123456")
                          .name("iamslash")
                          .address("Irving Ave")
                          .email("iamslash@gmail.com")
                          .build();
```

## Kotlin

### Kotlin Builder class

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

### Kotlin @JvmOverloads

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
