- [Abstract](#abstract)
- [Material](#material)
- [References](#references)
- [Basic Usages](#basic-usages)
  - [Hello World](#hello-world)
  - [Reserved Words](#reserved-words)
  - [Contextual Keywords](#contextual-keywords)
  - [Collections Compared to c++ containers](#collections-compared-to-c-containers)
  - [Collections](#collections)
  - [Datatypes](#datatypes)
  - [Constants](#constants)
  - [Preprocessor Directives](#preprocessor-directives)
  - [Concurrencies](#concurrencies)
- [Advanced Usages](#advanced-usages)
  - [Atrributes](#atrributes)
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

* [예제로 배우는 csharp 프로그래밍](http://www.csharpstudy.com/)
* [effective c#](http://www.aladin.co.kr/shop/wproduct.aspx?ItemId=873761)
  * 실력자 bill wagner의 번역서
* [An advanced introduction to C# @ codeproject](https://www.codeproject.com/Articles/1094079/An-advanced-introduction-to-Csharp-Lecture-Notes-P)
  * 기초부터 고급까지 정리된 글

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
// > mcs -out:a.exe a.cs
// > mono a.exe
```

## Reserved Words

```cs
abstract as base bool break byte case
catch char checked class const continue decimal
default delegatre do double else enum event
explicit extern false finally fixed float for
foreach goto if implicit in in(generic modifier) int
interface internal is lock long namespace new
null object operator out out(generic modifier) override params
private protected public readonly ref return sbyte
sealed short sizeof stackalloc static string struct
switch this throw true try typeof uint
ulong unchedked unsafe ushort using virtual void
volatile while
```

## Contextual Keywords

```cs
add alias ascending descending dynamic from get
global group into join let orderby partial(type)
partial(method) remove select set
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

## Datatypes

```cs
// value types
bool byte char decimal double float int long
sbyte short uint ulong ushort
// reference types
object, dynamic, string
// pointer types
char* cptr;
int* iptr;
```

## Constants

```cs
//// Integer Literals
212         /* Legal */
215u        /* Legal */
0xFeeL      /* Legal */
85         /* decimal */
0x4b       /* hexadecimal */
30         /* int */
30u        /* unsigned int */
30l        /* long */
30ul       /* unsigned long */

//// Floating-point Literals
3.14159       /* Legal */
314159E-5F    /* Legal */
510E          /* Illegal: incomplete exponent */
210f          /* Illegal: no decimal or exponent */
.e55          /* Illegal: missing integer or fraction */

//// String Literals
"hello, dear"
"hello, \
dear"
"hello, " "d" "ear"
@"hello dear"
```

## Preprocessor Directives

```cs
#define
#undef
#if
#else
#elif
#endif
#line
#error
#warning
#region
#endregion
```

## Concurrencies

* [C# 멀티쓰레딩](http://www.csharpstudy.com/Threads/thread.aspx)

thread 는 다음과 같이 `Thread, ThreadStart` 를 생성하여 사용한다.

```cs
    using System;
    using System.Threading;

    class Program
    {
        static void Main(string[] args)
        {
            new Program().DoTest();
        }

        void DoTest()
        {
            Thread t1 = new Thread(new ThreadStart(Run));
            t1.Start();
            Run();         
        }

        // 출력
        // Thread#1: Begin
        // Thread#3: Begin
        // Thread#1: End
        // Thread#3: End

        void Run()
        {
            Console.WriteLine("Thread#{0}: Begin", Thread.CurrentThread.ManagedThreadId);
            // Do Something
            Thread.Sleep(3000);
            Console.WriteLine("Thread#{0}: End", Thread.CurrentThread.ManagedThreadId);
        }
    }
```

thread pool 을 사용하면 thread 를 효과적으로 관리할 수 있다. `ThreadPool` 보다 `Task` 를 이용하여 thread pooling 하자.

```cs
    using System;    
    using System.Threading.Tasks;    

    class Program
    {
        static void Main(string[] args)
        {
            Task.Factory.StartNew(new Action<object>(Run), null);
            Task.Factory.StartNew(new Action<object>(Run), "1st");
            Task.Factory.StartNew(Run, "2nd");

            Console.Read();
        }

        static void Run(object data)
        {            
            Console.WriteLine(data == null ? "NULL" : data);
        }
    }
```

# Advanced Usages

## Atrributes

classes, methods, structures, enumerators, assemblies 등에 런타임에 확인할 수 있는 추가정보를 제공한다. 주로 다음과 같이 사용한다.

```cs
[attribute(positional_parameters, name_parameter = value, ...)]
element
```

.net framework 는 AttributeUSage, Conditional, Obsolete 를 기본 attribute 로 제공한다.

```cs
//// AttributeUsage
[AttributeUsage(
   AttributeTargets.Class |
   AttributeTargets.Constructor |
   AttributeTargets.Field |
   AttributeTargets.Method |
   AttributeTargets.Property,
   AllowMultiple = true)]
public class DeBugInfo : System.Attribute

//// Conditional
#define DEBUG
using System;
using System.Diagnostics;

public class Myclass {
   [Conditional("DEBUG")]
   
   public static void Message(string msg) {
      Console.WriteLine(msg);
   }
}
class Test {
   static void function1() {
      Myclass.Message("In Function 1.");
      function2();
   }
   static void function2() {
      Myclass.Message("In Function 2.");
   }
   public static void Main() {
      Myclass.Message("In Main function.");
      function1();
      Console.ReadKey();
   }
}

//// Obsolete Attribute
using System;

public class MyClass {
   [Obsolete("Don't use OldMethod, use NewMethod instead", true)]
   
   static void OldMethod() {
      Console.WriteLine("It is the old method");
   }
   static void NewMethod() {
      Console.WriteLine("It is the new method"); 
   }
   public static void Main() {
      OldMethod();
   }
}
```

custom attribute 를 제작하기 위해서는 다음과 같이 4 가지를 수행해야 한다.

* Declaring a custom attribute
* Constructing the custom attribute
* Apply the custom attribute on a target program element
* Accessing Attributes Through Reflection

```cs

// * Declaring a custom attribute
//a custom attribute BugFix to be assigned to a class and its members
[AttributeUsage(
   AttributeTargets.Class |
   AttributeTargets.Constructor |
   AttributeTargets.Field |
   AttributeTargets.Method |
   AttributeTargets.Property,
   AllowMultiple = true)]

public class DeBugInfo : System.Attribute

// * Constructing the custom attribute
//a custom attribute BugFix to be assigned to a class and its members
[AttributeUsage(
   AttributeTargets.Class |
   AttributeTargets.Constructor |
   AttributeTargets.Field |
   AttributeTargets.Method |
   AttributeTargets.Property,
   AllowMultiple = true)]

public class DeBugInfo : System.Attribute {
   private int bugNo;
   private string developer;
   private string lastReview;
   public string message;
   
   public DeBugInfo(int bg, string dev, string d) {
      this.bugNo = bg;
      this.developer = dev;
      this.lastReview = d;
   }
   public int BugNo {
      get {
         return bugNo;
      }
   }
   public string Developer {
      get {
         return developer;
      }
   }
   public string LastReview {
      get {
         return lastReview;
      }
   }
   public string Message {
      get {
         return message;
      }
      set {
         message = value;
      }
   }
}

// * Apply the custom attribute on a target program element
[DeBugInfo(45, "Zara Ali", "12/8/2012", Message = "Return type mismatch")]
[DeBugInfo(49, "Nuha Ali", "10/10/2012", Message = "Unused variable")]
class Rectangle {
   //member variables
   protected double length;
   protected double width;
   public Rectangle(double l, double w) {
      length = l;
      width = w;
   }
   [DeBugInfo(55, "Zara Ali", "19/10/2012", Message = "Return type mismatch")]
   
   public double GetArea() {
      return length * width;
   }
   [DeBugInfo(56, "Zara Ali", "19/10/2012")]
   
   public void Display() {
      Console.WriteLine("Length: {0}", length);
      Console.WriteLine("Width: {0}", width);
      Console.WriteLine("Area: {0}", GetArea());
   }
}

// * Accessing Attributes Through Reflection
using System;
using System.Reflection;

namespace BugFixApplication {
   //a custom attribute BugFix to be assigned to a class and its members
   [AttributeUsage(
      AttributeTargets.Class |
      AttributeTargets.Constructor |
      AttributeTargets.Field |
      AttributeTargets.Method |
      AttributeTargets.Property,
      AllowMultiple = true)]

   public class DeBugInfo : System.Attribute {
      private int bugNo;
      private string developer;
      private string lastReview;
      public string message;
      
      public DeBugInfo(int bg, string dev, string d) {
         this.bugNo = bg;
         this.developer = dev;
         this.lastReview = d;
      }
      public int BugNo {
         get {
            return bugNo;
         }
      }
      public string Developer {
         get {
            return developer;
         }
      }
      public string LastReview {
         get {
            return lastReview;
         }
      }
      public string Message {
         get {
            return message;
         }
         set {
            message = value;
         }
      }
   }
   [DeBugInfo(45, "Zara Ali", "12/8/2012", Message = "Return type mismatch")]
   [DeBugInfo(49, "Nuha Ali", "10/10/2012", Message = "Unused variable")]
   
   class Rectangle {
      //member variables
      protected double length;
      protected double width;
      
      public Rectangle(double l, double w) {
         length = l;
         width = w;
      }
      [DeBugInfo(55, "Zara Ali", "19/10/2012", Message = "Return type mismatch")]
      public double GetArea() {
         return length * width;
      }
      [DeBugInfo(56, "Zara Ali", "19/10/2012")]
      public void Display() {
         Console.WriteLine("Length: {0}", length);
         Console.WriteLine("Width: {0}", width);
         Console.WriteLine("Area: {0}", GetArea());
      }
   }//end class Rectangle
   
   class ExecuteRectangle {
      static void Main(string[] args) {
         Rectangle r = new Rectangle(4.5, 7.5);
         r.Display();
         Type type = typeof(Rectangle);
         
         //iterating through the attribtues of the Rectangle class
         foreach (Object attributes in type.GetCustomAttributes(false)) {
            DeBugInfo dbi = (DeBugInfo)attributes;
            
            if (null != dbi) {
               Console.WriteLine("Bug no: {0}", dbi.BugNo);
               Console.WriteLine("Developer: {0}", dbi.Developer);
               Console.WriteLine("Last Reviewed: {0}", dbi.LastReview);
               Console.WriteLine("Remarks: {0}", dbi.Message);
            }
         }
         
         //iterating through the method attribtues
         foreach (MethodInfo m in type.GetMethods()) {
            
            foreach (Attribute a in m.GetCustomAttributes(true)) {
               DeBugInfo dbi = (DeBugInfo)a;
               
               if (null != dbi) {
                  Console.WriteLine("Bug no: {0}, for Method: {1}", dbi.BugNo, m.Name);
                  Console.WriteLine("Developer: {0}", dbi.Developer);
                  Console.WriteLine("Last Reviewed: {0}", dbi.LastReview);
                  Console.WriteLine("Remarks: {0}", dbi.Message);
               }
            }
         }
         Console.ReadLine();
      }
   }
}
```

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
