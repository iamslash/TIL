- [Abstract](#abstract)
- [Materials](#materials)
- [Hexagonal Archiecture Over View](#hexagonal-archiecture-over-view)
- [Hexagonal Architecture Example](#hexagonal-architecture-example)

----

# Abstract

Hexagonal Architecture 는 [Alistair Cockburn](https://en.wikipedia.org/wiki/Alistair_Cockburn) 에 의해 발명되었다. 그것은 Layer 들의 의존성과 User Interface 에 Business Logic 이 섞이는 것을 해결해 준다. 

또한 Hexagonal Architecture 는 Ports and Adapters Architecture 라고 불리기도 한다. Port 는 Interface 를 의미하고 Adapter 는 Design Pattern 의 Adapter 와 같다.

Hexagonal 이라는 용어 때문에 6 가지 Port 를 써야할 거 같지만 그렇지 않다. 6 이라는 숫자보다는 충분한 Port 의 개수를 의미한다고 하는 것이 맞다.

# Materials

* [지속 가능한 소프트웨어 설계 패턴: 포트와 어댑터 아키텍처 적용하기 @ line](https://engineering.linecorp.com/ko/blog/port-and-adapter-architecture/?fbclid=IwAR2GLZMhXkX4Weri0qHQaLkwhlaBEJgFZ0yEQ5ilQO_cDJgvb2AP4TCqRu0)
* [DDD, Hexagonal, Onion, Clean, CQRS, … How I put it all together](https://herbertograca.com/2017/11/16/explicit-architecture-01-ddd-hexagonal-onion-clean-cqrs-how-i-put-it-all-together/)

# Hexagonal Archiecture Over View

다음은 Hexagonal Architecture 를 나타낸 그림이다. Secondary Port 는 Domain Layer 에 걸쳐있는게 더 맞을 것 같다.

![](img/hexagonal_architecture.png)

네모는 Adapter 이고 동그라미는 Port 를 의미한다. Port 는 Interface 이다. Adapter 는 주로 class 이다.

Application Layer 를 기준으로 왼쪽의 Port, Adapter 를 Primary Port, Primary Adapter 라고 한다. Primary Adapter 에서 Primary Port 를 호출한다. 즉, 실행의 흐름은 `Primary Adapter -> Primary Port` 이다.

예를 들어 `@Controller` class 에서 Interface 를 호출한다. 그리고 Interface 를 implement 한 Application Layer 의 `@Service` class 호출된다.

Application Layer 를 기준으로 오른쪽의 Port, Adapter 를 Secondary Port, Secondary Adapter 라고 한다. Secondary Port 를 Secondary Adapter 가 구현한다. 즉, 실행의 흐름은 `Secondary Port -> Secondary Adapter` 이다.

예를 들어 `@Repository` 가 부착된 Interface 를 호출한다. Spring Data JPA 는 그것을 implement 하는 Class 를 미리만들어 둔다. 그리고 그 Adapter Class 를 호출한다.

# Hexagonal Architecture Example


