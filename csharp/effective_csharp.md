
# Effective CSharp

## volatile을 사용하자.

volatile을 사용하지 않으면 CACHE에서 읽어올 수 있다. multi thread
programming의 경우 A thread가 특정 변수를 수정했을때 B thread가 반드시
언급한 변수의 값을 메모리에서 읽어 오게 하고 싶을때 사용한다.

## var를 잘 사용하자.

var는 compile time에 type deuction할 수 있다. primary type을
제외한 경우 var를 주로 이용하자.

## const 보다 readonly가 좋다.

상수를 선언하는 방법은 총 두가지이다. 컴파일 타임 상수는
const를 이용하고 런타임 상수는 readonly를 사용한다.

```cs
// compile time literal
public onst in a = 2000;
// run time literal
public static readonly int b = 2004;
```

DateTimpe과 같이 rimary type이 아닌 경우 const를 이용하여 상수를 정의
할 수 없다.

```cs
// compile error, use readonly
private const DateTime classCreation = new DateTime(2000, 1, 1, 0, 0, 0);
```

## 캐스트보다 is, as가 더 좋다.

다음과 같은 경우 as를 사용했기 때문에 캐스팅 실패시
InvalidCastException 대신 t에 null이 저장된다.

```cs
object o = Factory.GetObject();
MyType t = o as MyType;
if (t != null) {
  // use t
} else {
  // error
}
```

다음과 같이 형변환의 대상이 value type인 경우 as는 사용할 수 없다.

```cs
object o = Factory.GetObject();
int i = o as int; // compile error
```

위의 코드를 다음과 같이 nullable value type을 사용하면 compile
가능하다.

```cs
object o = Factory.GetObject();
int i = o as int ?;
if (i != null)
  Console.WriteLine(i.Value);
```

## [c#6.0] string.Format()을 보간 문자열로 대체하자. 

## IEnumerable<T>와 IQueryable<T>를 구분해서 사용하자.

IEnumerable<T>를 이용한 경우 해당 테이블의 데이터를 모두 받아온 후
로컬에서 Query를 수행한다. IQuerytable<T>를 이용한 경우 SQL로 전환되어
서버에서 Query를 수행하고 해당 record들을 받아온다.

```cs
var q = (from c in dbContext.Customers
         where c.City == "London"
         select c).AsEnumerable();
var finalAnswer = from c in q
                    orderby c.Name
                    select c;
```

```cs
var q = from c in dbContext.Customers 
        where c.City == "London"
        select c;
var finalAnswer = from c in q
                    orderby c.Name
                    select c;
```
