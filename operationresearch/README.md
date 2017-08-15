# abstract

- operation research(운용과학)은 시스템의 경영에 관련된 의사결정을
  체계적이고 합리적으로 하기 위한 방법을 연구하는 학문이다. Management
  Science라고 불리기도 한다.
- [참고](http://secom.hanbat.ac.kr/or/ch02/right04.html)

# mathematical programming(수리계획법)

- 현실에서 부딛히는 의사 결정 상황을 수학적 모양(수리계획모형)으로 작성하여
  그 해를 구함으로써 최적 의사결정을 도모하는 방법
- 수리계획모형은 다음과 같이 세가지 구성요소를 가지고 있다.
  - decision variable(의사결정변수)
  - objective function(목적함수)
  - constraints(제약조건)
- mathematical programming의 종류
  - LP(Linear Programming)(선형계획법) : 목적 함수와 제약 조건이 모두 1차식으로 표현
  - NLP(Non-Linear Programming)(비선형계획법) : 1차식으로만 표현되지 않는 모형
  - IP(Integer Programming)(정수계획법) : 의사결정변수가 정수값만을 가지는 특수한 경우
  - GP(Goal Programming)(목표계획법) : 목적함수가 여러개의 목표를 포함하고 있는 경우
  - DP(Dynamic Programming)(동적계획법) : 여러 단계에 걸쳐 변수의 값을 결정하는 모형

# ex

## problem

- 각 제품의 원료 사용량과 단위당 이익은 다음과 같다.

| 제품  | 원료A | 원료B | 원료C | 단위당 이익(만원) |
| :----| :----| :----| :----| :----: |
| 가   | 4     | 0    | 6    | 40     |
| 나   | 5     | 2    | 3    | 30     |

- 각 원료의 주간 사용가능량 : 200톤, 50톤, 210톤
- 총이익을 최대로 하는 생산 계획을 수립하는 것이 목표이다.

## solution

- decision variable
  - X1: 제품 가의 주간 생산량
  - X2: 제품 나의 주간 생산량

- objective function
  - MAXIMIZE(최대화) Z = 40X1 + 30X2
  
- Constraints
  - 4X1 + 5X2 <= 200 (원료A의 제약)
  -       2X2 <= 50  (원료B의 제약)
  - 6X1 + 3X2 <= 210 (원료C의 제약)
  - X1, X2 >= 0      (의사결정변수의 비음 조건)

# Simplex Method

- LP를 해결하는 방법으로 1947년 G. Danzig에 의해 처음 개발되었다.
- [단치히@위키피디아](https://ko.wikipedia.org/wiki/%EB%8B%A8%EC%B2%B4%EB%B2%95_(%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98))

# Integer Programming

- decision variable이 정수인 것을 포함하는 mathematical
  programming이다.
- objective function와 contraints가 모두 1차식인 경우 ILP(Integer
  Linear Programming)이라고 한다.

## 종류

- 순수정수계획모형(pure integer programming model) : 모든 변수가 정수인 모형
- 혼합정수계획모형(mixed integer programming model) : 일부가 정수인 모형
- 0-1 정수계획모형(0-1 integer programming model) : 모든 변수가 0 또는 1인 모형

## 해법

- 열거법
- 선형계획법의 해를 이용한 근사법
- cutting plane method(절단평면법)
- branch and bound method(분단탐색법)
