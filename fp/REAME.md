# Abstract

functional programming 은 함수의 동작에 의한 변수의 부수적인 값 변경을 원천 배제하는 프로그래밍 방법이다. 부수효과에 의해 발생할 수 있는 오류를 방지한다. 문제가 발생할 여지가 있는 일은 하지 않는 코딩 방식이다. 

외부변수를 사용하더라도 그 것에 접근해서 변경하는 것이 아니라 외부변수를 인자로 넣어서 사본으로 복사하여 작업한다. 부수효과는 발생하지 않는다.

그러나 모든 것을 함수형 프로그래밍으로 작성할 수는 없다. 일부 상태변화는 필요할 수 있다. 부수효과 없이 안정적이고 예측가능한 프로그램을 만드는 것이 목표이다.

정리하면 함수형 프로그래밍은 immutability 를 선호하고, 순수 함수를 사용하는 경우에 동시성을 쉽게 구현할 수 있으며, 반복보다는 transformation 을 사용하고, 조건문보다는 필터를 사용하는 프로그래밍 방법이다.

# Material

* [Level up with functional programming](https://grokkingsimplicity.com/)
  * [src](https://www.manning.com/books/grokking-functional-programming)
  * Functional Programming 을 계산, 액션, 데이터 로 구분해서 Programming 하는 것이라고 요약 
* [함수형 프로그래밍이 뭔가요? @ youtube](https://www.youtube.com/watch?v=jVG5jvOzu9Y)
  * 킹왕짱 설명
* [[10분 테코톡] 🍩도넛의 함수형 프로그래밍 @ youtube](https://www.youtube.com/watch?v=ii5hnSCE6No)
  * monad 를 Category theory 와 함께 설명함.
* [함수형 자바스크립트 프로그래밍 - 유인동](http://www.yes24.com/24/Goods/56885507?Acode=101)
  * 비함수형 언어인 자바스크립트를 이용하여 함수형 프로그래밍을 시도한다.
  * [src](https://github.com/indongyoo/functional-javascript)
  
# Functional Programming Features

* 함수형 프로그래밍은 **선언형** 이다.
  * ???
* 함수도 **값**이다. first class function
  * 함수를 값으로 전달할 수 있다. 변수에 담을 수도 있고 다른 함수의 인자로 전달할 수 있다.
    ```js
    // Original
    function foo(given) {
      console.log(given);
    }
    // Assign function to a variable
    var foo = function(given) {
      console.log(given);
    }
    // Assign function to a variable with lambda
    const foo = (given) => console.log(given);
    ```
* **고계함수**. higher order function
  * 함수를 인자로 전달받는 함수.
  * calc 는 higher order function 이다.
    ```js
    // calc is a higher order function because op is a function.
    var calc = function(num1, num2, op) {
      return op(num1, num2);
    }
    // Lambda version
    const calc = (num1, num2, op) => op(num1, num2);
    ```
* **커링**. currying
  * 여러 인자를 받는 함수에 일부 인자를 넣어서 나머지 인자를 받는 다른 함수를 만들어 낼 수 있는 functional programming 의 기법
  * calcWith2 는 currying 으로 만들어진 함수이다.
    ```js
    def calc(num1: Int, num2: Int, op: (Int, Int) => Int): Int
      = op(num1, num2)

    def add(num1: Int, num2: Int): Int = num1 + num2
    def multiply(num1: Int, num2: Int): Int = num1 * num2
    def power(num1: Int, num2: Int): Int
      = scala.math.pow(num1, num2).toInt

    def calcWith2(op: (Int, Int) => Int): (Int) => Int
      = (num: Int) => op(2, num)
    ```
* **함수 컴비네이터**
  * 다음과 같이 여러 함수들의 조합으로 구현을 간결하게 한다.
  * 다음은 loop statements 없이 함수들의 조합으로 구현한 예이다.
    ```js
    // as-is
    val students = List(
      new Student("Foo", "male", "25");
      new Student("Bar", "female", "26");
      new Student("Baz", "male", "27");
    )
    println(
      students
        .filter(i => i.sex == "male")
        .take(3)
        .map(i => s"${i.name}(${i.age})")
        .foldLeft("")((i, j) => i + " " + j)
    )
    ```

# Functional Programming Keywords

* each, map, filter, reject, find, findIndex, some, every, reduce, values, keys, rest
* callback, predicate, iteratee
* bind, curry, partial
