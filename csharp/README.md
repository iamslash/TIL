- [Abstract](#abstract)
- [Material](#material)
- [References](#references)
- [Basic Usages](#basic-usages)
  - [Hello World](#hello-world)
  - [Collections Compared to c++ containers](#collections-compared-to-c-containers)
  - [Collections](#collections)
- [Advanced Usages](#advanced-usages)
- [Tips](#tips)
  - [volatile을 사용하자.](#volatile%EC%9D%84-%EC%82%AC%EC%9A%A9%ED%95%98%EC%9E%90)
  - [var를 잘 사용하자.](#var%EB%A5%BC-%EC%9E%98-%EC%82%AC%EC%9A%A9%ED%95%98%EC%9E%90)
  - [const 보다 readonly가 좋다.](#const-%EB%B3%B4%EB%8B%A4-readonly%EA%B0%80-%EC%A2%8B%EB%8B%A4)
  - [캐스트보다 is, as가 더 좋다.](#%EC%BA%90%EC%8A%A4%ED%8A%B8%EB%B3%B4%EB%8B%A4-is-as%EA%B0%80-%EB%8D%94-%EC%A2%8B%EB%8B%A4)
  - [[c#6.0] string.Format()을 보간 문자열로 대체하자.](#c60-stringformat%EC%9D%84-%EB%B3%B4%EA%B0%84-%EB%AC%B8%EC%9E%90%EC%97%B4%EB%A1%9C-%EB%8C%80%EC%B2%B4%ED%95%98%EC%9E%90)
  - [IEnumerable<T>와 IQueryable<T>를 구분해서 사용하자.](#ienumerablet%EC%99%80-iqueryablet%EB%A5%BC-%EA%B5%AC%EB%B6%84%ED%95%B4%EC%84%9C-%EC%82%AC%EC%9A%A9%ED%95%98%EC%9E%90)

------

# Abstract

c#에 대해 정리한다.

# Material

* [effective c#](http://www.aladin.co.kr/shop/wproduct.aspx?ItemId=873761)
  * 실력자 bill wagner의 번역서

# References

* [coreclr @ github](https://github.com/dotnet/coreclr)
* [mono @ github](https://github.com/mono/mono)

# Basic Usages

## Hello World

```cs
using System;

namespace HelloWorldApplication {
   class HelloWorld {
      static void Main(string[] args) {
         /* my first program in C# */
         Console.WriteLine("Hello World");
         Console.ReadKey();
      }
   }
}
// mcs 
```

## Collections Compared to c++ containers

| c++                  | c#                   | 
|:---------------------|:---------------------|
| `if, else`           | `if, else`           |
| `for, while`         | `for, while, foreach`|
| `array`              | ``                   |
| `vector`             | `List`               |
| `deque`              | ``                   |
| `forward_list`       | `List`               |
| `list`               | `LinkedList`         |
| `stack`              | `Stack`              |
| `queue`              | `Queue`              |
| `priority_queue`     | ``                   |
| `set`                | `SortedSet`          |
| `multiset`           | ``                   |
| `map`                | `SortedDictionary`   |
| `multimap`           | ``                   |
| `unordered_set`      | `HashSet`            |
| `unordered_multiset` | ``                   |
| `unordered_map`      | `Dictionary`         |
| `unordered_multimap` | ``                   |

## Collections

* List

```cs
List<int> list = new List<int>();
        list.Add(2);
        list.Add(3);
        list.Add(7);
```

* LinkedList

```cs
        LinkedList<string> linked = new LinkedList<string>();
        linked.AddLast("cat");
        linked.AddLast("dog");
        linked.AddLast("man");
        linked.AddFirst("first");
        foreach (var item in linked)
        {
            Console.WriteLine(item);
        }

        LinkedListNode<string> node = linked.Find("one");
        linked.AddAfter(node, "inserted");        
```

* Stack

```cs
        Stack<int> stack = new Stack<int>();
        stack.Push(100);
        stack.Push(1000);
        stack.Push(10000);
        int pop = stack.Pop();
```

* Queue

```cs
        Queue<int> q = new Queue<int>();

        q.Enqueue(5);   // Add 5 to the end of the Queue.
        q.Enqueue(10);  // Then add 10. 5 is at the start.
        q.Enqueue(15);  // Then add 15.
        q.Enqueue(20);  // Then add 20.
        q.Dequeue();
```

* SortedSet

```cs
        SortedSet<string> set = new SortedSet<string>();

        set.Add("perls");
        set.Add("net");
        set.Add("dot");
        set.Add("sam");

        set.Remove("sam");

        foreach (string val in set)
        {
            Console.WriteLine(val);
        }
```

* SortedDictionary

```cs
        SortedDictionary<string, int> sort =
            new SortedDictionary<string, int>();

        sort.Add("zebra", 5);
        sort.Add("cat", 2);
        sort.Add("dog", 9);
        sort.Add("mouse", 4);
        sort.Add("programmer", 100);

        if (sort.ContainsKey("dog"))
        {
            Console.WriteLine(true);
        }

        if (sort.ContainsKey("zebra"))
        {
            Console.WriteLine(true);
        }

        Console.WriteLine(sort.ContainsKey("ape"));

        int v;
        if (sort.TryGetValue("programmer", out v))
        {
            Console.WriteLine(v);
        }

        foreach (KeyValuePair<string, int> p in sort)
        {
            Console.WriteLine("{0} = {1}",
                p.Key,
                p.Value);
        }
```

* HashSet

```cs
        string[] array1 =
        {
            "cat",
            "dog",
            "cat",
            "leopard",
            "tiger",
            "cat"
        };

        Console.WriteLine(string.Join(",", array1));

        var hash = new HashSet<string>(array1);

        string[] array2 = hash.ToArray();

        Console.WriteLine(string.Join(",", array2));

```

* Dictionary

```cs
        Dictionary<string, int> dictionary = new Dictionary<string, int>();

        dictionary.Add("cat", 2);
        dictionary.Add("dog", 1);
        dictionary.Add("llama", 0);
        dictionary.Add("iguana", -1);

        if (dictionary.ContainsKey("apple"))
        {
            int value = dictionary["apple"];
            Console.WriteLine(value);
        }

        string test;
        if (values.TryGetValue("cat", out test)) // Returns true.
        {
            Console.WriteLine(test); // This is the value at cat.
        }
```

# Advanced Usages



# Tips

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
