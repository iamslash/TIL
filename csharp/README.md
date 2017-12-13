# Abstract

c#에 대해 정리한다.

# Material

* [effective c#](http://www.aladin.co.kr/shop/wproduct.aspx?ItemId=873761)
  * 실력자 bill wagner의 번역서

# References

# Uaage

## var를 잘 사용하자.

var는 compile time에 type deuction할 수 있다. primary type을
제외한 경우 var를 주로 이용하자.

## IEnumerable<T>와 IQueryable<T>를 구분해서 사용하자.

IEnumerable<T>를 이용한 경우 해당 테이블의 데이터를 모두 받아온 후
로컬에서 Query를 수행한다. IQuerytable<T>를 이용한 경우 SQL로 전환되어
서버에서 Query를 수행하고 해당 record들을 받아온다.

```cs
var q = ().AsEnumerable();
```

```cs
var q = from c in dbContext.Customers 
        where c.City == "London"
        select c;
var finalAnswer = from c in q
                    orderby c.Name
                    select c;
```
