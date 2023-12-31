- [Abstract](#abstract)
- [Materials](#materials)
- [Overview](#overview)
- [Basic Usages](#basic-usages)
  - [SELECT](#select)
  - [PROJECT](#project)
  - [Union](#union)
  - [Difference](#difference)
  - [Intersection](#intersection)
  - [Cartesian Product](#cartesian-product)
  - [Join](#join)
  - [Theta Join](#theta-join)
  - [Equi Join](#equi-join)
  - [Natural Join](#natural-join)
  - [Left Outer Join](#left-outer-join)
  - [Right Outer Join](#right-outer-join)
  - [Full Outer Join](#full-outer-join)
  - [DIVISION](#division)

-----

# Abstract

Relational Algebra 는 relation data 를 다루는 Procedural Query Language (절차
질의 언어) 이다. Relational Algebra Operation 은 릴레이션을 입력으로 받아서
릴레이션을 출력으로 한다. 주요 operation 들을 정리해 본다.

# Materials

* [정보처리 실기_데이터베이스08강_관계 데이터 연산 | youtube](https://www.youtube.com/watch?v=h7AZAWfYH8k&list=PLimVTOIIZt2aP6msQIw0011mfVP-oJGab&index=9&t=0s)
  * 한글 강좌
* [Relational Algebra in DBMS with Examples](https://www.guru99.com/relational-algebra-dbms.html#14)

# Overview

Relational Operations 은 다음과 같이 분류할 수 있다.

* Unary Relational Operations

```cpp
SELECT (symbol: σ)
PROJECT (symbol: π)
RENAME (symbol: )
```

* Relational Algebra Operations From Set Theory

```cpp
UNION (υ)
INTERSECTION ( ),
DIFFERENCE (-)
CARTESIAN PRODUCT ( x )
```

* Binary Relational Operations

```cpp
JOIN
DIVISION
```

# Basic Usages

## SELECT

* Tutorials 을 입력으로 하고 Tutorials 릴레이션에서 topic 이 "Database" 인 릴레이션을 
  출력으로 하라.

```cpp
σ topic = "Database" (Tutorials)
```

## PROJECT

* Customers 릴레이션을 입력으로 하고 CustomerName, Status 속성만을 갖는 릴레이션을 출력으로 하라.

```c
Π CustomerName, Status (Customers)
```

## Union

* A 릴레이션과 B 릴레이션을 입력으로 하고 두 릴레이션의 합집합 릴레이션을 출력으로 하라.

```c
A ∪ B
```

## Difference

* A 릴레이션과 B 릴레이션을 입력으로 하고 두 릴레이션의 차집합 릴레이션을 출력으로 하라.

```c
A - B
```

## Intersection

* A 릴레이션과 B 릴레이션을 입력으로 하고 두 릴레이션의 교집합 릴레이션을 출력으로 하라.

```c
A ∩ B
```

## Cartesian Product

* A 릴레이션과 B 릴레이션의 Cartesian Product 를 한 릴레이션에서 column2 가 `1` 과 같은 릴레이션을 출력한다.

```c
σ column2 = '1' (A X B)
```

## Join

Join 은 두개 이상의 릴레이션을 cartesian product 하고 그 결과 릴레이션에서 일부를 선택하는 operation 이다. Join 은 다음과 같이 분류할 수 있다.

* Inner Join
  * Theta Join
  * Equi Join
  * Natural Join

* Outer Join
  * Left Outer Join
  * Right Outer Join
  * Full Outer Join

## Theta Join

* theta join 은 조건을 갖춘 일반적인 join 이다. 주로 다음과 같이 사용한다. θ 가 조건이다.

```c
A ⋈θ B
```

* A, B 두 릴레이션을 입력으로 하고 cartesian product 한다. 그리고 A.column2, B.column2 인 릴레이션을 출력한다.

```
A ⋈ A.column2 >  B.column2 (B)
```

## Equi Join

* theta join 에서 equivalence condition 만 적용한 것을 equi join 이라고 한다.

```c
A ⋈ A.column2 =  B.column2 (B)
```

* C, D 릴레이션을 입력으로 하고 cartesian product 한다. 두 릴레이션의 같은 속성이 있을 때 그 속성값이 같은 릴레이션을 출력한다.

```c
C ⋈ D
```

* C 

| Num | Square |
|:----|:-------|
| 2   | 4      |
| 3   | 9      |

* D

| Num | Cube |
|:----|:-------|
| 2   | 8      |
| 3   | 18     |

* C X D

| Num | Square | Num | Cube |
|:----|:-------|:----|:-----|
| 2   | 4      | 2   | 8    |
| 2   | 4      | 3   | 18   |
| 3   | 9      | 2   | 8    |
| 3   | 9      | 3   | 18   |

* C ⋈ D

| Num | Square | Num | Cube |
|:----|:-------|:----|:-----|
| 2   | 4      | 2   | 8    |
| 3   | 9      | 3   | 18   |

## Natural Join

* Equi Join 한 결과에서 공통 속성을 한번만 출력하는 것

```c
C ⋈ D
```

* C 

| Num | Square |
|:----|:-------|
| 2   | 4      |
| 3   | 9      |

* D

| Num | Cube |
|:----|:-------|
| 2   | 8      |
| 3   | 18     |

* C X D

| Num | Square | Num | Cube |
|:----|:-------|:----|:-----|
| 2   | 4      | 2   | 8    |
| 2   | 4      | 3   | 18   |
| 3   | 9      | 2   | 8    |
| 3   | 9      | 3   | 18   |

* C ⋈ D

| Num | Square | Cube |
|:----|:-------|:-----|
| 2   | 4      | 4    |
| 3   | 9      | 18   |

## Left Outer Join

* A, B 릴레이션을 입력으로 하고 cartesian product 한다. 두 릴레이션의 공통속성을 비교해 본다. 그 값이 같은 튜플과 A 에는 그 값이 있지만 B 에는 값이 없는 튜들을 합한 릴레이션을 출력한다. 
  * A ![](left_outer_join_op.png) B

* A 

| Num | Square |
|:----|:-------|
| 2   | 4      |
| 3   | 9      |
| 4   | 16     |

* B

| Num | Cube |
|:----|:-------|
| 2   | 8      |
| 3   | 18     |
| 5   | 75     |

* A X B

| Num | Square | Num | Cube |
|:----|:-------|:----|:-----|
| 2   | 4      | 2   | 8    |
| 2   | 4      | 3   | 18   |
| 2   | 4      | 5   | 75   |
| 3   | 9      | 2   | 8    |
| 3   | 9      | 3   | 18   |
| 3   | 9      | 5   | 75   |
| 4   | 16     | 2   | 8    |
| 4   | 16     | 3   | 18   |
| 4   | 16     | 5   | 75   |

* matching

| Num | Square | Num | Cube |
|:----|:-------|:----|:-----|
| 2   | 4      | 2   | 8    |
| 3   | 9      | 3   | 18   |
| 4   | 16     | NUL | NUL  |
| NUL | NUL    | 5   | 75   |

* A ![](left_outer_join_op.png) B

| Num | Square | Cube |
|:----|:-------|:-----|
| 2   | 4      | 8    |
| 3   | 9      | 18   |
| 4   | 16     | NUL  |

## Right Outer Join

* A, B 릴레이션을 입력으로 하고 cartesian product 한다. 두 릴레이션의 공통속성을 비교해 본다. 그 값이 같은 튜플과 B 에는 그 값이 있지만 A 에는 값이 없는 튜들을 합한 릴레이션을 출력한다. 
  * A ![](left_outer_join_op.png) B

* A 

| Num | Square |
|:----|:-------|
| 2   | 4      |
| 3   | 9      |
| 4   | 16     |

* B

| Num | Cube |
|:----|:-------|
| 2   | 8      |
| 3   | 18     |
| 5   | 75     |

* A X B

| Num | Square | Num | Cube |
|:----|:-------|:----|:-----|
| 2   | 4      | 2   | 8    |
| 2   | 4      | 3   | 18   |
| 2   | 4      | 5   | 75   |
| 3   | 9      | 2   | 8    |
| 3   | 9      | 3   | 18   |
| 3   | 9      | 5   | 75   |
| 4   | 16     | 2   | 8    |
| 4   | 16     | 3   | 18   |
| 4   | 16     | 5   | 75   |

* matching

| Num | Square | Num | Cube |
|:----|:-------|:----|:-----|
| 2   | 4      | 2   | 8    |
| 3   | 9      | 3   | 18   |
| 4   | 16     | NUL | NUL  |
| NUL | NUL    | 5   | 75   |

* A ![](left_outer_join_op.png) B

| Num | Square | Cube |
|:----|:-------|:-----|
| 2   | 4      | 8    |
| 3   | 9      | 18   |
| 5   | NUL    | 75   |

## Full Outer Join

* A, B 릴레이션을 입력으로 하고 cartesian product 한다. 두 릴레이션의 공통속성을 비교해 본다. 그 값이 같은 튜플과 A 에 그 값이 있거나 혹은 B 에 그 값이 있는 튜플들을 합한 것을 출력한다.
  * A ![](left_outer_join_op.png) B

* A 

| Num | Square |
|:----|:-------|
| 2   | 4      |
| 3   | 9      |
| 4   | 16     |

* B

| Num | Cube |
|:----|:-------|
| 2   | 8      |
| 3   | 18     |
| 5   | 75     |

* A X B

| Num | Square | Num | Cube |
|:----|:-------|:----|:-----|
| 2   | 4      | 2   | 8    |
| 2   | 4      | 3   | 18   |
| 2   | 4      | 5   | 75   |
| 3   | 9      | 2   | 8    |
| 3   | 9      | 3   | 18   |
| 3   | 9      | 5   | 75   |
| 4   | 16     | 2   | 8    |
| 4   | 16     | 3   | 18   |
| 4   | 16     | 5   | 75   |

* matching

| Num | Square | Num | Cube |
|:----|:-------|:----|:-----|
| 2   | 4      | 2   | 8    |
| 3   | 9      | 3   | 18   |
| 4   | 16     | NUL | NUL  |
| NUL | NUL    | 5   | 75   |

* A ![](left_outer_join_op.png) B

| Num | Square | Cube |
|:----|:-------|:-----|
| 2   | 4      | 8    |
| 3   | 9      | 18   |
| 4   | 16     | NUL  |
| 5   | NUL    | 75   |

## DIVISION

* A, B 릴레이션을 입력으로 한다. A 튜플들중 B 의 모든 과목코드를 값으로 하는 것들을 합하여 출력한다.

* A

| 학번 | 과목 | 성적 | 학과 |
|:----|:-------|:---|:--|
| 95   | C001  | A | 전기 |
| 96   | C001  | A | 기계 |
| 96   | C002  | B | 기계 |
| 97   | C001  | B | 컴퓨터 |
| 97   | C003  | C | 컴퓨터 |

* B

| 과목코드 | 과목명 | 학점 |
|:-------|:---|:--|
| C001  | 데이터베이스 | 3 |
| C002  | 운영체제 | 3 |

* A ÷ B

| 학번 | 과목 | 성적 | 학과 |
|:----|:-------|:---|:--|
| 96   | C001  | A | 기계 |
| 96   | C002  | B | 기계 |
