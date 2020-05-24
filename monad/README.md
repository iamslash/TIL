# Materials

* [3분 모나드](https://overcurried.com/3%EB%B6%84%20%EB%AA%A8%EB%82%98%EB%93%9C/)

# Basic

monad 는 합성할 수 있는 연산을 말한다.

어떤 타입 `M` 에 대해 두함수 `pure`, `compose` 가 존재할 때 `M` 은 monad 이다.

```haskell
type Pure = <A>(a: A) => M<A>;
type Compose = <A, B, C>(f: (a: A) => M<B>, g: (a: B) => M<C>) => (a: A) => M<C>;
```

다음은 `Maybe` 라는 monad 이고 간단한 monad 중 하나이다.

```haskell
type Maybe<A> = A | null;

function pure<A>(value: A): Maybe<A> {
  return value;
}

function compose<A, B, C>(f: (a: A) => Maybe<B>, g: (a: B) => Maybe<C>): (a: A) => Maybe<C> {
  return (a: A): Maybe<C> => {
    const ma = f(a);
  
    if (ma === null) return null;
    else g(ma);
  }
}
```

범주 의미론 (categorical semantic) 은 카테고리 이론 (category theory, a.k.a 범주론) 의 개념들로 연산을 정의하는 방식이다. category theory 는 집합론의 반대에 해당하는 개념이다. `요소` 대신 `요소들 간의 관계` 에 주목하여 추상적인 개념들을 가루는 수학 이론이다. 유지니오 모기 (Eugenio Moggi) 는 categorical semantic 에서 모나드로 연산을 정의하고 추상화할 수 있다는 것을 발견했다.

monad 는 programming 에서 연산을 정의하고 추상화하기 위해 사용된다.

다음은 generic programming 의 예이다. generic 은 data 를 추상화하고 있다.

```java
class List<T> {
  public readonly head: T;
  public readonly tail?: List<T>;

  constructor(head: T, tail?: List<T>) {
    this.head = head;
    this.tail = tail;
  }
}
```

data 뿐만 아니라 연산을 추상화 한다면 코드의 재사용성이 더욱 늘어날 것이다. 이것이 연산을 추상화하는 이유이다. 어떤 연산이 categorical semantic 에서 monad 로 추상화된다면 그 연산은 `합칠 수 있음` 이 보장된다. 수학적으로 monad 의 성질이 `합칠 수 있음` 이기 때문이다.

모든 프로그램은 곧 연산이다. monad 로 정의되는 연산은 합쳐질 수 있다. 이것이 monad 의 매력이다.

어떤 것이 monad 라는 것은 그것이 합성될 수 있는 연산이라는 것이다.
