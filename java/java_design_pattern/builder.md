
# Java Buildable interface

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

# Java Lombok

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
