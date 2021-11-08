- [Abstract](#abstract)
- [Materials](#materials)
- [Basics](#basics)
  - [What is monad?](#what-is-monad)
  - [Functor](#functor)
  - [Monad](#monad)

----

# Abstract

Monad 에 대해 정리한다.

# Materials

* [[10분 테코톡] 🍩도넛의 함수형 프로그래밍 @ youtube](https://www.youtube.com/watch?v=ii5hnSCE6No)
  * monad 를 Category theory 와 함께 설명함.
* [Monad란 무엇인가? @ youtube](https://www.youtube.com/watch?v=jI4aMyqvpfQ)
* [모나드와 함수형 아키텍처 1장. 시작하기](https://blog.burt.pe.kr/series/monad-and-functional-architecture-part-1/)
  * [모나드와 함수형 아키텍처 2장. 프로그래밍 패러다임](https://blog.burt.pe.kr/series/monad-and-functional-architecture-part-2/)
  * [모나드와 함수형 아키텍처 3장. 모나드](https://blog.burt.pe.kr/series/monad-and-functional-architecture-part-3/)
  * [모나드와 함수형 아키텍처 4장. Monad 실전 예제](https://blog.burt.pe.kr/series/monad-and-functional-architecture-part-4/)
  * [모나드와 함수형 아키텍처 5장. 함수형 아키텍처](https://blog.burt.pe.kr/series/monad-and-functional-architecture-part-5/)
  * [모나드와 함수형 아키텍처 6장. 부록](https://blog.burt.pe.kr/series/monad-and-functional-architecture-part-6/)
* [Monad Programming with Scala Future](https://leadsoftkorea.github.io/2016/03/03/monad-programming-with-scala-future/)

# Basics

## What is monad?

Monad 의 정의는 다음과 같다.

* Monad 는 값을 담는 컨테이너의 일종이다.
* Functor 를 기반으로 구현되었다.
* flatMap() method 를 제공한다.
* Monad Laws 를 만족시키는 구현체를 말한다.

## Functor

Functor 는 다음과 같은 interface 를 말한다. 다음과 같은 특징을 갖는다.

* Functor 는 함수를 인자로 받는 map method 만 갖는다.
* Type argument `<T>` 를 갖는다.
* `f` argument 는 `<T>` type value 를 받아서 `<R>` type value 를 return 하는 함수이다.
* `<T>` type 의 Functor 는 map 함수가 호출되면 `<R>` type 의 Functor 를 return 한다.

```java
import java.util.function.Function;

interface Functor<T> {
  <R> Functor<R> map(Function<T, R> f);
}
```

예를 들어 다음과 같은 `Functor` 와 `f` 가 있다고 해보자.

* `Functor`: `Functor<String>`
* `Function<String, Int> f`: `Int stringToInt(String args)`

`f` 를 `Functor` 의 `map()` 에 전달해보자. `Functor<String>` 을 전달받아서 `Functor<Int>` 가 return 된다. 

`map()` 의 참맛은 collection 의 원소를 순회하는 방법이 아니고 `<T>` type 의 Functor 를 `<R>` type 의 Functor 로 바꾸는 것이다.

Functor 를 이용하면 일반적으로 모델링할 수 없는 상황을 모델링할 수 있다. 즉 값이 없는 경우 (Optional) 혹은 값이 미래에 준비될 것으로 예상되는 경우 (Promise) 를 구현할 수 있다.

또한 Functor 를 이용하면 함수들을 쉽게 합성할 수 있다.

다음은 `Optional` Functor 를 구현한 예이다.

```java
FOptional<T> implements Functor<T, FOptional<?>> {
  private final T valueOrNull;
  private FOptional(T valueOrNull) {
    this.valueOrNull = valueOrNull;
  }
  public <R> FOptional<R> map(Function<T, R> f) {
    if (valueOrNull == null) {
      return empty();
    } else {
      return of(f.apply(valueOrNull));
    }
  }
  public static <T> FOptional<T> of(T a) {
    return new FOptional<T>(a);
  }
  public static <T> FOptional<T> empty() {
    return new FOptional<T>(null);
  }
}
```

다음은 `FOptional` 을 사용하는 예이다. 값이 null 인 경우 not null 인 경우 모두 동일하게 구현할 수 있다. type safety 를 유지하면 null 을 인코딩할 수 있다.

```java
FOptional<String> optionStr = FOptional(null);
FOptional<Integer> optionInt = optionStr.map(Integer::parseInt);

FOptional<String> optionStr = FOptional("1");
FOptional<Integer> optionInt = optionStr.map(Integer::parseInt);
```

다음은 `Promise` Functor 를 사용하는 예이다. `map()` 은 `Promise<T>` 를 반환하기 때문에 비동기 연산들의 합성이 가능하다???

```java
Promise<Customer> customer = //...
Promise<byte[]> bytes = customer.map(Customer::getAddress)  // return Promise<Address>
                                .map(Address::Street)       // return Promise<String>
                                .map((String s) -> s.substring(0, 3))       // return Promise<String>
                                .map(String::toLowerCase)       // return Promise<String>
                                .map(String::getBytes)       // return Promise<byte[]>
```

이번에는 `List<T>` 를 살펴보자. `List<T>` 역시 Functor 이다.

```java
class FList<T> implements Functor<T, FList<?>> {
  private final ImmutableList<T> list;
  FList(Iterable<T> value) {
    this.list = ImmutableList.copyOf(value);
  }
  @Override
  public <R> FList<?> map(Function <T, R> f) {
    ArrayList<R> result = new ArrayList<R>(list.size());
    for (T t : list) {
      result.add(f.apply(t));
    }
    return new FList<>(result);
  }
}
```

## Monad

Monad 는 Functor 에 `flatMap()` 을 추가한 것이다.

다음과 같이 `FOptional` Functor 의 `map()` 에  `tryParse()` 를 전달해 보자.

```java
FOptional<Integer> tryParse(String s) {
  try {
    final int i = Integer.parseInt(s);
    return FOptional.of(i);
  } catch (NumberFormatException e) {
    return FOptional.empty();
  }
}
```

`FOption<String>` type 의 Functor 가 `FOptional<FOptional<Integer>>` type 의 Functor 로 변환되어야 한다. 이것은 다음과 같이 compile error 를 발생시킨다.

```java
FOptional<Integer> num1 = //...
FOptional<FOptional<Integer>> num2 = //...

FOptional<Data> date1 = num1.map(t -> new Date(t));
FOptional<Data> date1 = num2.map(t -> new Date(t)); // compile error
```

Functor 가 두번 감싸져 있기 때문에 그 기능을 제대로 하지 못 한다. 이것을 해결하기 위해 `flatMap()` 이 만들어 졌다. 다음은 `flatMap()` 의 정의이다. 

```java
interface Monad<T, M extends Monad<?, ?>> extends Functor<T, M> {
  M flatMap(Function<T, M> f);
}
```

보다 쉽게 이해하기 위해 Functor 의 map() 과 비교해 보자.

```java
interface Functor<T> {
  <R> Functor<R> map(Function<T, R> f);
}
```

Monad 의 `flatMap()` 에서 `f` argument 는 `<T>` type value 를 받아서 `<M>` type value 를 return 하는 함수이다. 그리고 `flatMap()` 은 `<M>` type value 를 return 한다.

다음은 `flatMap()` 을 적용한 예이다. 이제 `FOptional<String>` 은 Monad 이다.

```java
FOptional<String> num = FOptional.of("42")
// tryParse 는 FOptional<Integer> 를 return 한다.
FOptional<Integer> ans = num.flatMap(this::tryParse);
FOptional<Date> date = ans.map(t -> new Date(t)) // 합성이 가능하다.

num.flatMap(this::tryParse)
   .map(t -> new Date(t))  // 합성이 가능하다.
```

Monad 는 합수를 합성이 가능하도록 해준다. 또한 값이 없는 경우 혹은 미래에 사용되는 경우와 같이 일반적으로 구현할 수 없는 상황을 모델링할 수 있다. 그리고 asynchronous logic 을 synchronous logic 을 구현하는 것과 같은 형태로 구현하면서도 함수의 합성 및 non-blocking pipeline 을 구현할 수 있다???
