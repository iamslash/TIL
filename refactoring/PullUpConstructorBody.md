다음과 같이 자식 class 생성자의 일부분이 부모 class 와 중복된다면 
부모 class 로 자식 class 생성자의 일부분을 추출하라. 

```java
class Person extends Animal {
    public Person(String name, String id, String grade) {
        this.name = name;
        this.id = id;
        this.grade = grade;
    }
}
```

```java
class Person extends Animal {
    public Person(String name, String id, String grade) {
        super(name, id);
        this.grade = grade;
    }
}
```
