- [Abstract](#abstract)
- [Materials](#materials)
- [Conceptual ERD Symbols and Notations](#conceptual-erd-symbols-and-notations)
- [Physical ERD Symbols and Notations](#physical-erd-symbols-and-notations)

----

# Abstract

Data Modeling 할 때 유용한 Diagram 이다.

Data Modeling 은 다음과 같은 순서로 진행한다. [관계형 데이터 모델링 @ 생활코딩](https://opentutorials.org/course/3883)

* 업무파악
* 개념적 모델링
* 논리적 모델링 - PK, FK, 정규화
* 물리적 모델링 - 역정규화, 인덱스

Conceptual ERD 와 Physical ERD 가 사용하는 Symbol 은 다르다.

# Materials

* [ERD Editor](https://marketplace.visualstudio.com/items?itemName=dineug.vuerd-vscode)
  * useful erd editor

* [What is Entity Relationship Diagram (ERD)?](https://www.visual-paradigm.com/guide/data-modeling/what-is-entity-relationship-diagram/)
* [Entity-Relationship Diagram Symbols and Notation](https://www.lucidchart.com/pages/ER-diagram-symbols-and-meaning)
  * [Entity Relationship Diagram (ERD) Tutorial - Part 1 @ youtube](https://www.youtube.com/watch?v=QpdhBUYk7Kk)
  * [Entity Relationship Diagram (ERD) Tutorial - Part 2 @ youtube](https://www.youtube.com/watch?v=-CuY5ADwn24)
* [https://gngsn.tistory.com/48](https://gngsn.tistory.com/48)

# Conceptual ERD Symbols and Notations

# Physical ERD Symbols and Notations

* **Cardinality** : Cardinality refers to the number of unique items in a set. Cardinality defines the possible number of occurrences in one entity which is associated with the number of occurrences in another. ex) one-to-many, many-to-many.
  * ![](crowfeet.png)

* **실선** : 엔터티의 PK 가 다른 엔터티의 PK 중 일부이다.
* **점선** : 엔터티의 PK 가 다른 엔터티의 PK 가 아니다. FK 혹은 일반 attribute 이다.

* **Bridge Entity** : 두개의 엔터티의 cardinality 가 `n : n` 일 때 엔터티를 하나 추가해서 cardinality 를 `1 : n, n: 1` 로 해결해 준다. 이때 추가한 entity 를 bridge entity 라고 한다.
  