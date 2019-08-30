# Abstract

Use Case, Class Diagram, Sequence Diagram, Activity Diagram 정도를 알아두자.

# Materials

* [PlantUML 간단 요약](http://plantuml.com/ko/)

# Use Case Diagram

* [모델링과 UML 유스케이스 다이어그램](https://m.blog.naver.com/PostView.nhn?blogId=ljh0326s&logNo=221001892737&proxyReferer=https%3A%2F%2Fwww.google.com%2F)
* [유즈 케이스 다이어그램](http://plantuml.com/ko/use-case-diagram)

----

* Scope
* Usecase
* Actor
  * Primary Actor
  * Secondary Actor `<<actor>>`
* Relationship
  * include `<<includes>>` 포함 관계
  * generalize 일반화 관계
  * extend `<<extend>>` 선택적 관계

# Class Diagram 

![](/designpattern/Uml_class_relation_arrows_en.svg.png)

## Inheritance

B 클래스가 A 클래스를 상속할 때 둘의 관계는 Inheritance 이다. 

## Realization

B 클래스가 A 인터페이스를 구현할 때 둘의 관계는 Realization 이다.

## Dependency

A 클래스가 B 클래스를 함수의 인자 혹은 리턴값 으로 사용할 때 둘의 관계는 Dependency 이다.

```cs
public class A {
    public void enroll(B b){}
}
```

## Association

A 클래스가 B 클래스를 소유할 때 둘의 관계는 Association 이다. (has-a)

```cs
public class A {
    private B b;
}
```

## Aggregation

A 클래스가 B 클래스를 소유하고 B 클래스는 A 클래스를 구성하는 부분일 때 둘의 관계는 Aggregation 이다. (has-a, whole-part)

```cs
public class A {
    private List<B> b;
}
```

## Composition

A 클래스가 B 클래스를 소유하고 B 클래스는 A 클래스를 구성하는 부분이며 A 클래스가 파괴되면 B 클래스 역시 파괴될 때 둘의 관계는 Aggregation 이다. (has-a, whole-part, ownership)

```cs
public class A {
    private B b;
    public A() {
       b = new B();
    }
}
```

## Aggregation vs Composition

호수 클래스와 오리 클래스가 있다고 하자. 호수위에 오리가 떠있다. 그리고 오리들은 농부의 소유물이다. 호수가 사라진다고 해서 오리가 사라지지는 않는다. 호수 클래스와 오리 클래스는 Aggregation 관계이다.

자동차와 클래스와 카뷰레터 클래스가 있다고 하자. 카뷰레터는 자동차의 부품이다. 자동차가 파괴되면 카뷰레터 역시 파괴된다. 자동차 클래스와 카뷰레터 클래스는 Composition 관계이다.


# Activity diagram

* [Activity Diagram 액티비티 다이어그램 [소프트웨어 설계]](http://blog.naver.com/PostView.nhn?blogId=tlsdlf5&logNo=120116742269)

----

* activity state
* initial state
* final state
* decision
* transition
* swim lane (actor 별 구분)
* 

# Sequence diagram

* [[홀인원 3.06.07] 시퀀스 다이어그램 @ youtube](https://www.youtube.com/watch?v=YwBL-fTTqZQ)

----

```uml
Alice -> Bob: Authentication Request
Bob --> Alice: Authentication Response

Alice -> Bob: Another authentication Request
Alice <-- Bob: another authentication Response
```
