- [Abstract](#abstract)
- [References](#references)
- [Materials](#materials)
- [Basic Usages](#basic-usages)
  - [Compile, Execution](#compile-execution)
  - [Reserved Words](#reserved-words)
  - [Useful Keywords](#useful-keywords)
  - [Bit Manipulation](#bit-manipulation)
  - [String](#string)
  - [Collections compared c++ container](#collections-compared-c-container)
  - [Collection Examples](#collection-examples)
  - [Multi dimensional array](#multi-dimensional-array)
  - [Collection Framework](#collection-framework)
  - [Collection Implementations](#collection-implementations)
  - [Data Types](#data-types)
  - [Decision Making](#decision-making)
  - [Loops](#loops)
  - [Inner Classes](#inner-classes)
  - [java 8 Interface Changes](#java-8-interface-changes)
  - [Marker Interfaces](#marker-interfaces)
  - [Functional Interfaces](#functional-interfaces)
  - [Anonymous Classes](#anonymous-classes)
  - [Enum](#enum)
  - [Annotation](#annotation)
  - [Generics](#generics)
  - [Concurrency](#concurrency)
    - [ReentrantLock](#reentrantlock)
    - [Semaphore](#semaphore)
    - [CountDownLatch](#countdownlatch)
  - [Static Class](#static-class)
  - [Test](#test)
  - [What's new Java8](#whats-new-java8)
    - [Interface Default and Static Methods](#interface-default-and-static-methods)
    - [Functional Interfaces](#functional-interfaces-1)
    - [Method References](#method-references)
    - [Optional<T>](#optionalt)
- [Advanced Usage](#advanced-usage)
  - [ArrayList vs CopyOnWriteArrayList](#arraylist-vs-copyonwritearraylist)
  - [jvm architecture](#jvm-architecture)
  - [jvm garbage collector](#jvm-garbage-collector)
  - [Stream](#stream)
  - [Java code coverage library](#java-code-coverage-library)
  - [Java Byte Code Manipulation library](#java-byte-code-manipulation-library)
  - [Java Lombok](#java-lombok)
  - [JVM Options](#jvm-options)
  - [Thread Dump, Heap Dump](#thread-dump-heap-dump)
- [Quiz](#quiz)

-------------------------------------------------------------------------------

# Abstract

java 를 정리한다.

# References

* [Java magazine](https://blogs.oracle.com/javamagazine/issue-archives)
  * [Java magazine - Reactive programming](file:///Users/davidsun/Documents/java-magazine-jan-feb-2018.pdf) 
* [java-examples](https://github.com/iamslash/java-examples)

# Materials

* [더 자바, 코드를 조작하는 다양한 방법 by 백기선](https://www.inflearn.com/course/the-java-code-manipulation/dashboard)
* [더 자바, 애플리케이션을 테스트하는 다양한 방법 by 백기선](https://www.inflearn.com/course/the-java-application-test)
* [Parallel, Concurrent, and Distributed Programming in Java Specialization @ coursera](https://www.coursera.org/specializations/pcdp)
  * [Parallel Programming in Java](https://www.coursera.org/learn/parallel-programming-in-java)
  * [Concurrent Programming in Java](https://www.coursera.org/learn/concurrent-programming-in-java#syllabus)
  * [Distributed Programming in Java](https://www.coursera.org/learn/distributed-programming-in-java)
* [자바 프로그래밍 언어 코딩 규칙](https://myeonguni.tistory.com/1596)
* [Advanced Java](http://enos.itcollege.ee/~jpoial/allalaadimised/reading/Advanced-java.pdf)
  * java 의 고급내용
* [learn java in Y minutes](https://learnxinyminutes.com/docs/java/)
* [java @ tutorialspoint](https://www.tutorialspoint.com/java/)
* [초보자를 위한 Java 200제 (2판)](http://www.infopub.co.kr/index.asp)
  * [목차](http://www.infopub.co.kr/common/book_contents/05000262.html)
  * [src](http://www.infopub.co.kr/index.asp)
* [JAVA 의 정석](http://www.yes24.com/Product/goods/24259565)
  * [video](https://www.youtube.com/watch?v=xRkCbqR0v84&list=PLW2UjW795-f5LNeTO6VQB1ZIeZJ_kwEG1)
  * [src](https://github.com/castello/javajungsuk3)
  * [blog](https://codechobo.tistory.com/1?category=645496)
  
# Basic Usages

## Compile, Execution

```bash
> javac A.java
> java A
```

## Reserved Words

```java
abstract    assert      boolean     break 
byte        case        catch       char
class       const       continue    default
do          double      else        enum
extends     final       finally     float
for         goto        if          implements
import      instanceof  int         interface
long        native      new         package
private     protected   public      return
short       static      strictfp    super
switch      synchronized    this    throw
throws      transient   try         void
volatile    while
```

## Useful Keywords

- volatile
  - 데이터를 읽을 때 cahe 에서 읽지 않고 memory 에서 읽는다. 그리고 데이터를 쓸 때 cache 에 쓰지 않고 memory 에 쓴다.
  - thread 들이 여러개의 cache 때문에 데이터의 원자성이 보장되지 않을 때 사용한다.

```java
public class SharedFoo {
    public volatile int counter = 0;
}
```

- strictfp
  - JVM 은 host platform 에 따라 부동 소수점 표현방식이 다양할 수 있다. IEEE 754 로 표준화 하기 위해 필요하다.

```java
strictfp class Example {
  public static void main(String[] args) {
    double d = Double.MAX_VALUE;
    System.out.println(d*1.1);
  }
}
strictfp class A {...} 
strictfp interface B {...} 
strictfp void method() {...} 
```

- native
  - [참고](https://www.baeldung.com/java-native)
  - java 에서 c/cpp library 와 같은 platform dependent api 를 이용할 때 선언한다.

```java
public class DateTimeUtils {
    public native String getSystemTime();
 
    static {
        System.loadLibrary("nativedatetimeutils");
    }
}

public class DateTimeUtilsManualTest {
 
   @BeforeClass
    public static void setUpClass() {
        // .. load other dependent libraries  
        System.loadLibrary("nativedatetimeutils");
    }
 
    @Test
    public void givenNativeLibsLoaded_thenNativeMethodIsAccessible() {
        DateTimeUtils dateTimeUtils = new DateTimeUtils();
        LOG.info("System time is : " + dateTimeUtils.getSystemTime());
        assertNotNull(dateTimeUtils.getSystemTime());
    }
}
```

- transient
  - serialize 의 대상이 되지 않도록 한다.

```java
class Person implements Serializable {
    private transient String name; // thi shoul be null
    private String email;
    private int age;

    public Member(String name, String email, int age) {
        this.name = name;
        this.email = email;
        this.age = age;
    }
    @Override
    public String toString() {
        return String.format("Person{name='%s', email='%s', age='%s'}", name, email, age);
    }
}
...
  public static void main(String[] args) throws IOException, ClassNotFoundException {
        Person p = new Person("iamslash", "iamslash@gmail.com", 28); 
        String s = serializeTest(p);
        deSerializeTest(s);
    }
```

## Bit Manipulation

* [Difference between >>> and >>](https://stackoverflow.com/questions/2811319/difference-between-and)

----

`>>` arithmatic bit shift vs `>>>` logical bit shift

```java
int a = -2; // 1111 1110

// Arithmatic bit shift
int b = a >> 1; // 1111 1111

// Logical bit shift
int c = a >>> 1; // 0111 1111
```

## String

```java
s = s.substring(0, s.length() - 1);
StringBuilder sb = new StringBuilder();
sb.append("a");
sb.deleteCharAt(sb.length()-1);
```

## Collections compared c++ container

| c++                  | java                                   |
| :------------------- | :------------------------------------- |
| `if, else`           | `if, else`                             |
| `for, while`         | `for, while`                           |
| `array`              | `Collections.unmodifiableList`         |
| `vector`             | `Vector, ArrayList`                    |
| `deque`              | `Deque, ArrayDeque`                    |
| `forward_list`       | ``                                     |
| `list`               | `List, LinkedList`                     |
| `stack`              | `Stack, Deque, ArrayDeque, LinkedList` |
| `queue`              | `Queue, LinkedList`                    |
| `priority_queue`     | `Queue, PriorityQueue`                 |
| `set`                | `SortedSet, TreeSet`                   |
| `multiset`           | ``                                     |
| `map`                | `SortedMap, TreeMap`                   |
| `multimap`           | ``                                     |
| `unordered_set`      | `Set, HashSet`                         |
| `unordered_multiset` | ``                                     |
| `unordered_map`      | `Map, HashMap`                         |
| `unordered_multimap` | ``                                     |

* core interfaces in Collections

```
             Iterable
                |
            Collection        Map
         /    |    |           |
       Set List  Queue     SortedMap 
        |          | 
       SortedSet Deque
```

* core classes in Collections
  * [Collections in java @ geeksforgeeks](https://www.geeksforgeeks.org/collections-in-java-2/)
  * [Collections in Java @ javapoint](https://www.javatpoint.com/collections-in-java)

* Legacy classes
  * Collection 이 개발되기 전에 container 들이다. 사용을 추천하지 않는다.
  * Vector, Dictionary, HashTable, Properties, Stack 은 모두 lgacy class 이다.
* Vector vs ArrayList
  * Vector 는 legacy class 이다. ArrayList 는 새로 개발된 Collection 이다.
  * Vector 는 thread safe 하다. ArrayList 그렇지 않다. 그래서 ArrayList 가 더 빠르다.
* Stack vs Deque
  * Stack 은 legacy class 이다. Degue 는 새로 개발된 Collection 이다.
  * Deque 은 LIFO queue 를 지원한다. Deque 의 사용을 추천한다. 

## Collection Examples

* unmodifiableList

```java
   // create array list
   List<Character> list = new ArrayList<Character>();

   // populate the list
   list.add('X');
   list.add('Y');

   System.out.println("Initial list: "+ list);

   // make the list unmodifiable
   List<Character> immutablelist = Collections.unmodifiableList(list);
```

* Vector

```java
   Vector<Integer> D = new Vector<Integer>(Collections.nCopies(N,  1));
   Integer a = D.get(0);
   D.set(0, 1);
   D.add(1);
```

* List, ArrayList

```java
   List<Integer> D = new ArrayList<Integer>(Collections.nCopies(N,  1));
   Integer a = D.get(0);
   D.set(0, 1);
   D.add(1);
```

* Deque, ArrayDeque
  
| Queue Method | Equivalent Deque Method | Throws Exception |
| ------------ | ----------------------- | ---------------- |
| add(e)       | addLast(e)              |                  |
| offer(e)     | offerLast(e)            |                  |
| remove()     | removeFirst()           |                  |
| poll()       | pollFirst()             | x                |
| element()    | getfirst()              |                  |
| peek()       | peekFirst()             | x                |

| Stack Method | Equivalent Deque Method | Throws Exception |
| ------------ | ----------------------- | ---------------- |
| push(e)      | addFirst(e)             |                  |
| pop(e)       | removeFirst(e)          |                  |
| peek(e)      | peekFirst(e)            | x                |

```java
   Deque<Integer> deque = new ArrayDeque<>();
   int n     = deque.size();
   int first = deque.getFirst();
   int last  = deque.getLast();
   deque.addFirst(3);
   deque.addLast(4);
   deque.pollFirst();
   deque.pollLast();
```

* List, LinkedList

```java
   List<Integer> D = new LinkedList<Integer>(Collections.nCopies(N,  1));
   Integer a = D.get(0);
   D.set(0, 1);
   D.add(1);
```

* Stack

```java
   Stack<String> stack = new Stack<>();
   stack.push("fly");
   stack.push("worm");
   stack.push("butterfly");
   String peekResult = stack.peek();
   String popResult = stack.pop();
   popResult = stack.pop();
```

* Queue, ArrayDeque

```java
   Queue<Integer> queue = new ArrayDeque<>();
   boolean bOffer = queue.offer(2); // Add element.
   boolean bAdd = queue.add(1); // Add element and throw IllegalStateException if no space available.
   int peeked = queue.peek(); // Return head of queue or null or 0 if empty.
   int a = queue.element(); // Return head of queue. it throws an exception if this queue is empty.
   while (queue.size() > 0) {
      int polled = queue.poll(); // Retrieves and remove or return null or 0 if queue is empty.
      int remove = queue.remove(); // Retrieves and remove. only in that it throws an exception if this queue is empty.
      System.out.println(polled);
   }
```

* Queue, PriorityQueue

```java
   // Descending order
   Queue<Integer> queue = new PriorityQueue<>((a,b) -> b - a);
   queue.add(1);
   queue.add(100);
   queue.add(0);
   queue.add(1000);
   int peeked = queue.peek();
   while (queue.size() > 0) {
      int polled = queue.poll();
      System.out.println(polled);
   }
```

* SortedSet, TreeSet

```java
   SortedSet<String> set = new TreeSet<String>();
   set.add("perls");
   set.add("net");
   set.add("dot");
   set.add("sam");
   set.remove("sam");
   for (String val : set) // ... Alphabetical order.
   {
      System.out.println(val);
   }    
```

* SortedMap, TreeMap
  * [SortedMap Interface in Java with Examples](https://www.geeksforgeeks.org/sortedmap-java-examples/)

```java
   SortedMap<String, String> map = new TreeMap<>();
   map.put(".com", "International");
   map.put(".us", "United States");
   System.out.println(map.get(".au"));
   map.remove(".us");
   System.out.println(map.containsKey(".au"));
   System.out.println(map.containsValue("International"));
   System.out.println(map);

   SortedMap<String, String> map2 = new TreeMap<>();
   map2.put(".net", "iamslash");
   map.putAll(map2);
   Set<String> keySet = map.keySet();

   String firstKey = map.firstKey();
   String lastKey = map.lastKey();
   List<String> values = new ArrayList<>(map.values());

   Comparator comp = sotreemap.comparator();
   System.out.println(comp); // null
   SortedMap<String, String> map2 = new TreeMap<Integer, String>(Collections.reverseOrder()); 
   comp = sotreemap.comparator();   
   System.out.println(comp); // 

   // Constructor with lambda
   SortedMap<Integer, Integer> cnts =
      new TreeMap<Integer, Integer>(
            (a, b) -> {
               return a - b;
            });

   // Loops
   // import java.util.Map.Entry
   Set<Entry<Integer,Integer>> sss = map.entrySet();
   Set sst = map.entrySet();
   Iterator it = sst.iterator();
   while (it.hasNext()) {
      Map.Entry m = (Map.Entry)it.next();
      String key = (String)m.getKey();
      String val = (String)m.getValue();
   }
```

* Set, HashSet

```java
   Set<String> set = new HashSet<String>();
   set.add("perls");
   System.out.println(set.add("perls")); // return false for same key
   System.out.println(set.contains("perls")); // true
   System.out.println(set.isEmpty()); // false 
   set.add("net");
   set.add("dot");
   set.add("sam");
   set.remove("sam"); // true
   set.remove("Alice"); // false
   for (String val : set) {  // ... Alphabetical order.
      System.out.println(val);
   }    
```

* Map, HashMap
  * [자바 HashMap을 효과적으로 사용하는 방법](http://tech.javacafe.io/2018/12/03/HashMap/)

```java
   Map<String, String> map = new HashMap<>();
   map.put(".com", "International");
   map.putIfAbsent("aaa", "bbb");
   map.computeIfAbset("aaa", key -> "bbb");
   map.put(".au", map.getOrDefault(".au", "Australia"));
   System.out.println(map);
   System.out.println(map.get(".au"));
   System.out.println(map.containsKey(".au"));
   map.forEach(key -> System.out.println(key));
   Collection<String> values = map.values();
   values.forEach(val -> System.out.println(val));  
   map.forEach((key, val) -> {
      System.out.print("key: "+ key);
      System.out.println(", val: "+ val);
   });    
```

* Set, LinkedHashSet

LinkedList 처럼 입력된 순서대로 저장

```java
   Set<String> set = new LinkedHashSet<String>();
   set.add("perls");
   set.add("net");
   set.add("dot");
   set.add("sam");
   set.remove("sam");
   for (String val : set)
   {
      System.out.println(val);
   }
```

* Map, LinkedHashMap

LinkedList 처럼 입력된 순서대로 저장

```java
   Map<String, String> map = new LinkedHashMap<>();
   map.put(".com", "International");
   map.put(".us", "United States");
   map.put(".uk", "United Kingdom");
   map.put(".jp", "Japan");
   map.put(".au", "Australia");
   System.out.println(map.get(".au"));
```

* sort

```java
// sort array ascending
// It's not possible to sort descending with Arrays.sort 
// use ArrayList instead of Arrays.sort
int[] A = new int[]{5, 4, 3, 2, 1};
// int[] A = {5, 4, 3, 2, 1};
Arrays.sort(A);

// Create a list of strings 
List<String> al = new ArrayList<String>(); 
al.add("Geeks For Geeks"); 
al.add("Friends"); 
al.add("Dear"); 
al.add("Is"); 
al.add("Superb"); 
Collections.sort(al); 
Collections.sort(al, Collections.reverseOrder()); 
// sort by length ascending
Collections.sort(a1, (a, b) -> Integer.valueOf(a.length()).compareTo(b.length()));

// comparator class
class CompDsc implements Comparator<Integer> 
{ 
   public int compare(Integer a, Integer b) 
   { 
      return b - a;
   } 
}
Collections.sort(al, new CompDsc()); 

// anonymous comparator class
Collections.sort(rec, new Comparator() {
   public int compare(Object o1, Object o2) {
      Integer a = (Integer)o1;
      Integer b = (Integer)o2;
      return a.compareTo(b);
   }
});
```

* Arrays

```java
int[] A = new int[10];
Arrays.fill(A, 0);
Arrays.sort(A);
int a = Arrays.binarySearch(A, 437);
int[] b = Arrays.copyOf(A, 3);
int[] c = Arrays.copyOfRange(A, 2, 4);
```

* Collections

```java
List<String> A = new ArrayList<String>(); 
Collections.fill(A, "Foo");
Collections.sort(A); 
Collections.sort(A, Collections.reverseOrder()); 
Collections.sort(A, (a, b) -> String.compareTo(a, b));
Collections.reverse(A);
String s = Collections.min(A);
```

## Multi dimensional array

* [Multi Dimensional ArrayList in Java](https://www.baeldung.com/java-multi-dimensional-arraylist)

----

```java
// 2d Integer
int vertexCount = 3;
ArrayList<ArrayList<Integer>> graph = new ArrayList<>(vertexCount);
for(int i=0; i < vertexCount; i++) {
    graph.add(new ArrayList());
}
graph.get(0).add(1);
graph.get(1).add(2);
graph.get(2).add(0);
graph.get(1).add(0);
graph.get(2).add(1);
graph.get(0).add(2);
int vertexCount = graph.size();
for (int i = 0; i < vertexCount; i++) {
    int edgeCount = graph.get(i).size();
    for (int j = 0; j < edgeCount; j++) {
        Integer startVertex = i;
        Integer endVertex = graph.get(i).get(j);
        System.out.printf("Vertex %d is connected to vertex %d%n", startVertex, endVertex);
    }
}

// 3d String
int x_axis_length = 2;
int y_axis_length = 2;
int z_axis_length = 2;  
ArrayList<ArrayList<ArrayList<String>>> space = new ArrayList<>(x_axis_length);
for (int i = 0; i < x_axis_length; i++) {
    space.add(new ArrayList<ArrayList<String>>(y_axis_length));
    for (int j = 0; j < y_axis_length; j++) {
        space.get(i).add(new ArrayList<String>(z_axis_length));
    }
}
space.get(0).get(0).add(0,"Red");
space.get(0).get(0).add(1,"Red");
space.get(0).get(1).add(0,"Blue");
space.get(0).get(1).add(1,"Blue");
space.get(i).get(j).get(k)

// 2d string
String a = "abac";
String b = "cab";

int m = a.length(), n = b.length();
String[][] C = new String[m+1][n+1];
for (int i = 0; i < m; ++i) {
   for (int j = 0; j < n; ++j) {
      C[i][j] = "";
   }
}
for (int i = 0; i < m; ++i) {
   for (int j = 0; j < n; ++j) {
      System.out.print(String.format("%s ", C[i][j]));
   }
   System.out.println("");
}
```

## Collection Framework

- [Java Collection Framework Technote](https://docs.oracle.com/javase/8/docs/technotes/guides/collections/)
- [Outline of the Collections Framework](http://docs.oracle.com/javase/8/docs/technotes/guides/collections/reference.html)
- [Collection Framework](https://upload.wikimedia.org/wikibooks/en/thumb/c/ca/Java_collection_implementation.jpg/700px-Java_collection_implementation.jpg)
- [Collections in Java](https://www.javatpoint.com/collections-in-java)

![](img/java-collection-hierarchy.png)

## Collection Implementations

| Interface | Hash Table | Resizable Array | Balanced Tree | Linked List | Hash Table + Linked List |
| :-------- | :--------- | :-------------: | :------------ | :---------- | :----------------------: |
| Set       | HashSet    |                 | TreeSet       |             |      LinkedHashSet       |
| List      |            |    ArrayList    |               | LinkedList  |                          |
| Deque     |            |   ArrayDeque    |               | LinkedList  |                          |
| Map       | HashMap    |                 | TreeMap       |             |      LinkedHashMap       |


## Data Types

```java
// 
byte short int long float double
boolean char

// Literals
byte a = 68;
char a = 'A';
int decimal = 100;
int octal = 0144;
int hexa  = 0x64;
char a = '\u001';
String a = "\u001";
```

## Decision Making

```java
//// if
int x = 30;
if (x == 10) {
   System.out.print("Value of X is 10");
} else if ( x == 20 ) {
   System.out.print("Value of X is 20");
} else if ( x == 30 ) {
   System.out.print("Value of X is 30");
} else {
   System.out.print("This is else statement");
}
//// switch
// char grade = args[0].charAt(0);
char grade = 'C';
switch (grade) {
   case 'A' :
      System.out.println("Excellent!"); 
      break;
   case 'B' :
   case 'C' :
      System.out.println("Well done");
      break;
   case 'D' :
      System.out.println("You passed");
   case 'F' :
      System.out.println("Better try again");
      break;
   default :
      System.out.println("Invalid grade");
}
System.out.println("Your grade is " + grade);
```

## Loops

```java
//// while
int x = 10;
while (x < 20) {
   System.out.print("value of x : " + x );
   x++;
   System.out.print("\n");
}
//// for
for (int x = 10; x < 20; x = x + 1) {
   System.out.print("value of x : " + x );
   System.out.print("\n");
}
//// do while 
int x = 10;
do {
   System.out.print("value of x : " + x );
   x++;
   System.out.print("\n");
} while ( x < 20 );
//// break
int [] numbers = {10, 20, 30, 40, 50};
for (int x : numbers) {
   if( x == 30 ) {
      break;
   }
   System.out.print( x );
   System.out.print("\n");
}
//// continue
int [] numbers = {10, 20, 30, 40, 50};
for (int x : numbers) {
   if( x == 30 ) {
      continue;
   }
   System.out.print( x );
   System.out.print("\n");
}
//// range based for
int [] numbers = {10, 20, 30, 40, 50};
for(int x : numbers) {
   System.out.print( x );
   System.out.print(",");
}
System.out.print("\n");
String [] names = {"James", "Larry", "Tom", "Lacy"};
for(String name : names) {
   System.out.print( name );
   System.out.print(",");
}
```

## Inner Classes

```java
//// Anonymous Inner Class
// AnonymousInner an_inner = new AnonymousInner() {
//    public void my_method() {
//       ........
//       ........
//    }   
// };
abstract class AnonymousInner {
   public abstract void mymethod();
}

public class Outer_class {
   public static void main(String args[]) {
      AnonymousInner inner = new AnonymousInner() {
         public void mymethod() {
            System.out.println("This is an example of anonymous inner class");
         }
      };
      inner.mymethod();	
   }
}

//// Anonymous Inner Class as Argument
// obj.my_Method(new My_Class() {
//    public void Do() {
//       .....
//       .....
//    }
// });
interface Message {
   String greet();
}

public class My_class {
   // method which accepts the object of interface Message
   public void displayMessage(Message m) {
      System.out.println(m.greet() +
         ", This is an example of anonymous inner class as an argument");  
   }

   public static void main(String args[]) {
      // Instantiating the class
      My_class obj = new My_class();

      // Passing an anonymous inner class as an argument
      obj.displayMessage(new Message() {
         public String greet() {
            return "Hello";
         }
      });
   }
}
```

## java 8 Interface Changes

java 8 의 interface 는 `default methods, static methods` 가 가능하다.

```java
public interface IA {

	void foo(String str);

	default void bar(String str){
	}
   
   static void baz(String str) {      
   }
}
```

## Marker Interfaces

다음과 같이 몸체가 없는 interface 를 marker interface 라고 한다. 이것을 상속받으면 단지 상속받았다는 표시만 하기 때문에 marker interface 라고 한다.

```java
public interface Cloneable {
}
public interface Serializable {   
}
```

## Functional Interfaces

single abstract method 를 갖는 interface 를 특별히 functional interface 라고 한다. 다음과 같이 `@FunctionalInterface` 를 사용하면 compiler 에게 functional interface 라는 힌트를 줄 수 있다.

```java
@FunctionalInterface
public interface Runnable {
  void run();
}
```

java 8 부터 다음과 같이 functional interface 를 구현한 anonymous class instance 를 lambda expression 으로 생성할 수 있다.

```java
public void runMe(final Runnable r) {
  r.run();
}
...
runMe(() -> System.out.println( "Run!" ));
```

## Anonymous Classes

anonymous function 처럼 interface 를 구현한 class 의 instance 를 이름없이 생성할 수 있다. 다음은 `Runnable` interface 를 상속받는 class 의 instance 를 이름없이 생성하는 예이다.

```java
public class AnonymousClass {
  public static void main( String[] args ) {
    new Thread(
      new Runnable() {
        @Override
        public void run() {
          // Implementation here
        }
      }
    ).start();
  }
}
```

## Enum

다음은 요일을 표현한 class 이다.

```java
public class DaysOfTheWeekConstants {
   public static final int MONDAY = 0;
   public static final int TUESDAY = 1;
   public static final int WEDNESDAY = 2;
   public static final int THURSDAY = 3;
   public static final int FRIDAY = 4;
   public static final int SATURDAY = 5;
   public static final int SUNDAY = 6;
}

public boolean isWeekend(int day) {
   return( day == SATURDAY || day == SUNDAY );
}
```

위의 예를 enum 을 사용하여 다음과 같이 간단히 구현할 수 있다. 

```java
public enum DaysOfTheWeek {
   MONDAY,
   TUESDAY,
   WEDNESDAY,
   THURSDAY,
   FRIDAY,
   SATURDAY,
   SUNDAY
}

public boolean isWeekend(DaysOfTheWeek day) {
   return(day == SATURDAY || day == SUNDAY);
}
```

enum 은 특수한 class 이다. 다음과 같이 instance field, constructor, method 등을 갖을 수 있다.

```java
public enum DaysOfTheWeekFields {
   
   MONDAY(false),
   TUESDAY(false),
   WEDNESDAY(false),
   THURSDAY(false),
   FRIDAY(false),
   SATURDAY(true),
   SUNDAY(true);

   private final boolean isWeekend;

   private DaysOfTheWeekFields(final boolean isWeekend) {
      this.isWeekend = isWeekend;
   }

   public boolean isWeekend() {
      return isWeekend;
   }
}

public boolean isWeekend(DaysOfTheWeek day) {
   return day.isWeekend();
}
```

enum 은 class 이기 때문에 다음과 같이 interface 를 구현할 수도 있다.

```java
interface DayOfWeek {
   boolean isWeekend();
}

public enum DaysOfTheWeekInterfaces implements DayOfWeek {
   MONDAY() {
      @Override
      public boolean isWeekend() {
         return false;
      }
   },
   TUESDAY() {
      @Override
      public boolean isWeekend() {
         return false;
      }
   },
   WEDNESDAY() {
      @Override
      public boolean isWeekend() {
         return false;
      }
   },
   THURSDAY() {
   @Override
      public boolean isWeekend() {
         return false;
      }
   },
   FRIDAY() {
      @Override
      public boolean isWeekend() {
         return false;
      }
   },
   SATURDAY() {
      @Override
      public boolean isWeekend() {
         return true;
      }
   },
   SUNDAY() {
      @Override
      public boolean isWeekend() {
         return true;
      }
   };
}
```

위의 예를 `@Override` 를 하나 사용하여 더욱 간략히 구현할 수도 있다.

```java
public enum DaysOfTheWeekFieldsInterfaces implements DayOfWeek {
   MONDAY( false ),
   TUESDAY( false ),
   WEDNESDAY( false ),
   THURSDAY( false ),
   FRIDAY( false ),
   SATURDAY( true ),
   SUNDAY( true );
   private final boolean isWeekend;
   private DaysOfTheWeekFieldsInterfaces(final boolean isWeekend) {
      this.isWeekend = isWeekend;
   }
   @Override
   public boolean isWeekend() {
      return isWeekend;
   }
}
```

컴파일러는 enum 을 다음과 같이 변환한다. 즉 `Enum<...>` generic 을 상속받는 class 로 변환된다.

```java
public class DaysOfTheWeek extends Enum<DaysOfTheWeek> {
   // Other declarations here
}
```

## Annotation

* [Java Annotation](javaannotation.md)

## Generics

다음은 generic interface 의 예이다. actual type 은 generic interface 를 구현한 class 를 작성할 때 사용한다.

```java
public interface GenericInterfaceOneType<T> {
   void performAction(final T action);
}

public interface GenericInterfaceSeveralTypes<T, R> {
   R performAction(final T action);
}

public class ClassImplementingGenericInterface implements GenericInterfaceOneType<String> {
   @Override
   public void performAction(final String action) {
      // Implementation here
   }
}
```

다음은 generic class 의 예이다. actual type 은 generic class 의 instance 를 생성하거나 class 에서 상속받을 때 사용한다.

```java
public class GenericClassOneType<T> {
   public void performAction(final T action) {
      // Implementation here
   }
}

public class GenericClassImplementingGenericInterface<T> implements GenericInterfaceOneType<T> {
   @Override
   public void performAction(final T action) {
      // Implementation here
   }
}
```

다음은 generic method 의 예이다. actual type 은 generic method 를
호출할 때 사용한다.

```java
public<T, R> R performAction(final T action) {
   final R result = ...;
   // Implementation here
   return result;
}

protected abstract<T, R> R performAction(final T action);
static<T, R> R performActionOn(final Collection<T> action) {
   final R result = ...;
   // Implementation here
   return result;
}  

public class GenericMethods<T> {
   public<R> R performAction(final T action) {
      final R result = ...;
      // Implementation here
      return result;
   }
   public<U, R> R performAnotherAction(final U action) {
      final R result = ...;
      // Implementation here
      return result;
   }
}

public class GenericMethods<T> {
   public GenericMethods(final T initialAction) {
      // Implementation here
   }
   public<J> GenericMethods(final T initialAction, final J nextAction) {
      // Implementation here
   }
}
```

generic 의 type 에 primitive type 은 사용할 수 없다. primitive Wrapper type 를 사용해야 한다. generic method 의 경우 argument 로 primitive type 이 전달될 때 primitive wrapper type 으로 형변환 된다. 이것을 boxing 이라고 한다.

```java
final List<Long> longs = new ArrayList<>();
final Set<Integer> integers = new HashSet<>();

final List<Long> longs = new ArrayList<>();
longs.add(0L); // ’long’ is boxed to ’Long’
long value = longs.get(0); // ’Long’ is unboxed to ’long’
// Do something with value
```

generic 은 compile time 에 type erasure 를 한다. 즉 type 결정을 runtime 에 하기 위해 compile time 에 generic type 을 지운다. 따라서 다음과 같은 코드는 compile 할 때 method 가 중복 선언되었다는 오류를 발생한다.

```java
void sort(Collection<String> strings) {
   // Some implementation over strings heres
}
void sort(Collection<Number> numbers) {
   // Some implementation over numbers here
}
```
generic 의 array 는 만들 수 없다.

```java
public<T> void performAction(final T action) {
   T[] actions = new T[0];
}
```

generic type 을 `extends` 를 사용하여 자신 혹은 후손으로 제한할 수 있다.

```java
public<T extends InputStream> void read(final T stream) {
   // Some implementation here
}
public<T extends Serializable> void store(final T object) {
   // Some implementation here
}  
public<T, J extends T> void action(final T initial, final J next) {
   // Some implementation here
}
public<T extends InputStream & Serializable> 
void storeToRead(final T stream) {
   // Some implementation here
}
public<T extends Serializable & Externalizable & Cloneable> 
void persist(final T object) {
   // Some implementation here
}
```

다음은 method 의 argument 에 wildcard declaration 과 함께 generic type 을 사용한 예이다.

```java
// objects's type is a Collection<T> and T is Serializable or T extends Serializable.
public void store(final Collection<? extends Serializable> objects) {
   // Some implementation here
}
// objects's type is a Collection<T> and T is anything.
public void store( final Collection<?> objects) {
   // Some implementation here
}
// objects's type is a Collection<T> and T is Integer or Integer extends T.
public void interate( final Collection<? super Integer> objects) {
   // Some implementation here
}
```

Wildcard declaration 은 [Difference between <? super T> and <? extends T> in Java @ stackoverflow](https://stackoverflow.com/questions/4343202/difference-between-super-t-and-extends-t-in-java) 을 참고하여 이해하자. 항상 PECS: "Producer Extends, Consumer Super" 를 기억해야 함.

```java
public class Collections { 
  public static <T> void copy(List<? super T> dest, List<? extends T> src) {
      for (int i = 0; i < src.size(); i++) 
        dest.set(i, src.get(i)); 
  } 
}
```

예를 들어 다음과 같이 Zoo 및 Zoo 의 Actual Type 으로 사용할 수 있는 class 들의 diagram 이 있다고 하자. [[Java] 제네릭(generic) - 제한된 타입 파라미터, 와일드카드 타입](https://palpit.tistory.com/668)

```java
public class Zoo<T> {
  private String name;
  private T[] animals;
  ...
}
```

```
        Animal
          |
        ------
       |      |
    Predator Herbivore
              |
             Rabbit
```

* `Zoo<?>` 는 Animal, Predator, Herbivore, Rabbit, Dog 이 가능하다. 아무거나 가능하다. 
* `Zoo<? extends Herbivore>` 는 Herbivore, Rabbit 만 가능하다.
* `Zoo<? super Predator` 는 Predator, Animal 만 가능하다.

## Concurrency

* [동시성](https://github.com/Yooii-Studios/Clean-Code/blob/master/Chapter%2013%20-%20%EB%8F%99%EC%8B%9C%EC%84%B1.md)
  * [blog](https://nesoy.github.io/articles/2018-04/CleanCode-ConCurrency)
* [Java Concurrency and Multithreading Tutorial](http://tutorials.jenkov.com/java-concurrency)
  * Java 의 Concurrency 에 대해 기가 막히게 설명한 글

----

### ReentrantLock

WIP

### Semaphore

WIP

### CountDownLatch

```java
import java.util.concurrent.CountDownLatch;

class Foo {
	private CountDownLatch lock2 = new CountDownLatch(1);
	private CountDownLatch lock3 = new CountDownLatch(1);

	public Foo() {
        
	}

	public void first(Runnable printFirst) throws InterruptedException {
        
		printFirst.run();
		lock2.countDown();
	}

	public void second(Runnable printSecond) throws InterruptedException {
		lock2.await();
		printSecond.run();
		lock3.countDown();
	}

	public void third(Runnable printThird) throws InterruptedException {
		lock3.await();
		printThird.run();
	}
}
```

## Static Class

* [Static class in Java](https://www.geeksforgeeks.org/static-class-in-java/)

Nested Class 는 static 혹은 non-static class 일 수 있다. non-static class 를 특별히 Inner class 라고도 한다. Nested class 를 품고 있는 것을 Outer class 라고 하자. static Nested class 의 instance 는 Outer class instance 없이 생성할 수 있다. 그러나 non-static Nested class 의 instance 는 반드시 Outer class 의 instance 를 생성한 후에 만들 수 있다.

```java
// Java program to demonstrate how to 
// implement static and non-static 
// classes in a Java program. 
class OuterClass { 
	private static String msg = "GeeksForGeeks"; 

	// Static nested class 
	public static class NestedStaticClass { 

		// Only static members of Outer class 
		// is directly accessible in nested 
		// static class 
		public void printMessage() 
		{ 

			// Try making 'message' a non-static 
			// variable, there will be compiler error 
			System.out.println( 
				"Message from nested static class: "
				+ msg); 
		} 
	} 

	// Non-static nested class - 
	// also called Inner class 
	public class InnerClass { 

		// Both static and non-static members 
		// of Outer class are accessible in 
		// this Inner class 
		public void display() 
		{ 
			System.out.println( 
				"Message from non-static nested class: "
				+ msg); 
		} 
	} 
} 
class Main { 
	// How to create instance of static 
	// and non static nested class? 
	public static void main(String args[]) 
	{ 

		// Create instance of nested Static class 
		OuterClass.NestedStaticClass printer 
			= new OuterClass.NestedStaticClass(); 

		// Call non static method of nested 
		// static class 
		printer.printMessage(); 

		// In order to create instance of 
		// Inner class we need an Outer class 
		// instance. Let us create Outer class 
		// instance for creating 
		// non-static nested class 
		OuterClass outer = new OuterClass(); 
		OuterClass.InnerClass inner 
			= outer.new InnerClass(); 

		// Calling non-static method of Inner class 
		inner.display(); 

		// We can also combine above steps in one 
		// step to create instance of Inner class 
		OuterClass.InnerClass innerObject 
			= new OuterClass().new InnerClass(); 

		// Similarly we can now call Inner class method 
		innerObject.display(); 
	} 
} 
```

## Test

* [java test](javatest.md)

## What's new Java8

* [What's New in JDK 8 @ oracle](https://www.oracle.com/technetwork/java/javase/8-whats-new-2157071.html)
* [New Features in Java 8](https://www.baeldung.com/java-8-new-features)

----

### Interface Default and Static Methods

```java
//// static method
interface Vehicle {
  static String producer() {
    return "N&F Vehicles";
  }
}

class A {
  public static void main(String[] args) {
    String producer = Vehicle.producer();
  }
}

//// default method
interface Vehicle {
  static String producer() {
    return "N&F Vehicles";
  }
  default String getOverview() {
    return "ATV made by " + producer();
  }
}

class VehicleImpl implements Vehicle {
}

class A {
  public static void main(String[] args) {
    Vehicle v = new VehicleImpl();
    String overview = v.getOverview();
  }
}
```

### Functional Interfaces

* [Java8#02. 함수형 인터페이스(Functional Interface)](https://multifrontgarden.tistory.com/125?category=471239)

```java
// Runnable
// arguments void return void
Runnable r = () -> System.out.println("Hello World");
r.run();

// Supplier<T>
// arguments void return T
Supplier<String> s = () -> "hello supplier";
String result = s.get();

// Consumer<T>
// arguments T return void
Consumer<String> c = str -> System.out.println(str);
c.accept("hello consumer");

// Function<T, R>
// arguments T return R
Function<String, Integer> f = str -> Integer.parseInt(str);
Integer result = f.apply("1");

// Predicate<T>
// arguments T return boolean
Predicate<String> p = str -> str.isEmpty();
boolean result = p.test("hello");

// UnaryOperator<T>
// arguments T return T
UnaryOperator<String> u = str -> str + " operator";
String result = u.apply("hello unary");

// BinaryOperator<T>
// arguments T, T return T
BinaryOperator<String> b = (str1, str2) -> str1 + " " + str2;
String result = b.apply("hello", "binary");

// BiPredicate<T, U>
// arguments T, U return boolean
BiPredicate<String, Integer> bp = (str, num) -> str.equals(Integer.toString(num));
boolean result = bp.test("1", 1);

// BiConsumer<T, U>
// arguments T, U return void
BiConsumer<String, Integer> bc = (str, num) -> System.out.println(str + " :: " + num);
bc.accept("숫자", 5);

// BiFunction<T, U, R> 
// arguments T, U return R
BiFunction<Integer, String, String> bf = (num, str) -> String.valueOf(num) + str;
String result = bf.apply(5, "999");

// Comparator<T>
Comparator<String> c = (str1, str2) -> str1.compareTo(str2);
int result = c.compare("foo", "bar");
```

### Method References

If you use method references you can reduce the size of code.

* [Java8#03. 메서드 레퍼런스(Method Reference)](https://multifrontgarden.tistory.com/126?category=471239)

```java
//// Reference to a Static Method
// boolean isReal = list.stream().anyMatch(u -> User.isRealUser(u));
boolean isReal = list.stream().anyMatch(User::isRealUser);

//// Reference to an Instance Method
User user = new User();
boolean isLegalName = list.stream().anyMatch(user::isLegalName);

//// Reference to an Instance Method of an Object of a Particular Type
long count = list.stream().filter(String::isEmpty).count();

//// Reference to a Constructor
Stream<User> stream = list.stream().map(User::new);

// Reference to a lambda method
// You will make it simpler than before with method refereces.
Function<String, Integer> f = str -> Integer.parseInt(str);

// Reference to a instance method of an object of a particular type
Function<String, Integer> f = Integer::parseInt;
Integer result = f.apply("123");

// Reference to a instance method of an object of a particular type
Function<String, Boolean> f = String::isEmpty;
Boolean result = f.apply("123");

// Reference to a static method
Function<String, Integer> f = Integer::parseInt;
Integer result = f.apply("123");

// Reference to a constructor
Supplier<String> s = String::new;

// Reference to a instance method
String str = "hello";
Predicate<String> p = str::equals;
p.test("world");

// Reference to a instance method of an object of a particular type
Comparator<String> c = String::compareTo;
```

### Optional<T>

```java
//// Creation of the Optional<T>
Optional<String> optional = Optional.empty();
String str = "value";
Optional<String> optional = Optional.of(str);
Optional<String> optional = Optional.ofNullable(getString());

//// Optional<T> usage
// List<String> list = getList();
// List<String> listOpt = list != null ? list : new ArrayList<>();
List<String> listOpt = getList().orElseGet(() -> new ArrayList<>());

// User user = getUser();
// if (user != null) {
//     Address address = user.getAddress();
//     if (address != null) {
//         String street = address.getStreet();
//         if (street != null) {
//             return street;
//         }
//     }
// }
// return "not specified";
Optional<User> user = Optional.ofNullable(getUser());
String result = user
  .map(User::getAddress)
  .map(Address::getStreet)
  .orElse("not specified");

// Optional<T>
Optional<OptionalUser> optionalUser = Optional.ofNullable(getOptionalUser());
String result = optionalUser
  .flatMap(OptionalUser::getAddress)
  .flatMap(OptionalAddress::getStreet)
  .orElse("not specified");

// handling NPE
// String value = null;
// String result = "";
// try {
//     result = value.toUpperCase();
// } catch (NullPointerException exception) {
//     throw new CustomException();
// }
String value = null;
Optional<String> valueOpt = Optional.ofNullable(value);
String result = valueOpt.orElseThrow(CustomException::new).toUpperCase();

```

# Advanced Usage

## ArrayList vs CopyOnWriteArrayList

* [Java List 인터페이스 중 CopyOnWriteArrayList 소개](https://wedul.site/350)

----

ArrayList 와 CopyOnWriteArrayList 는 둘다 List interface 를 implement 한다.
CopyOnWriteArrayList 는 내용이 update 되면 instance 를 복제하여 update 한다.
따라서 thread 에 argument 로 전달될때 복제본이 전달되기 때문에 thread safety 가
해결된다.

## jvm architecture

* [jvm @ TIL](/jvm/README.md)

## jvm garbage collector

* [Java Garbage Collection Basics](https://www.oracle.com/webfolder/technetwork/tutorials/obe/java/gc01/index.html)
  * garbage collector 의 기본 원리에 대해 알 수 있다.

![](https://www.oracle.com/webfolder/technetwork/tutorials/obe/java/gc01/images/gcslides/Slide5.png)

jvm 의 gc 는 크게 `Young Generation, Old Generation, Permanent Generation` 으로 나누어 진다. 영원히 보존해야 되는 것들은 `Permanent Generation` 으로 격리된다. `Young Generation` 에서 `minor collect` 할때 마다 살아남은 녀석들중 나이가 많은 녀석들은 `Old Generation` 으로 격리된다. 

`Young Generation` 은 다시 `eden, S0, S1` 으로 나누어 진다. `eden` 이 꽉 차면 `minor collect` 이벤트가 발생하고 `eden, S0` 혹은 `eden, S1` 의 `unreferenced object` 는 소멸되고 `referenced object` 는 나이가 하나 증가하여 `S1` 혹은 `S0` 으로 옮겨진다. 나이가 많은 녀석들은 `Old Genration` 으로 옮겨진다. `eden, S0` 과 `eden, S1` 이 교대로 사용된다.

## Stream

* [Guide to Java 8’s Collectors](https://www.baeldung.com/java-8-collectors)
* [자바 스트림(Stream) API 정리, 스트림을 이용한 가독성 좋은 코드 만들기(feat. 자바 람다, 함수형 프로그래밍, 자바8)](https://jeong-pro.tistory.com/165)
* [자바8 스트림 API 소개](https://www.slideshare.net/madvirus/8-api)
* [Java 8 Parallel Streams Examples](https://mkyong.com/java8/java-8-parallel-streams-examples/)

----

다음은 legacy code 를 stream 을 사용하여 compact 하게 만든 예이다.

```java
// Legacy
List<String> names = Arrays.asList("foo", "bar", "baz");
long cnt = 0;
for (String name : names) {
   if (name.contains("f")) {
      cnt++;
   }
}
System.out.println("Count is " + cnt);
// Stream
cnt = 0;
cnt = names.stream().filter(x -> x.contains("f")).count();
System.out.println("Count is " + cnt);
```

다음은 stream 을 생성하는 방법의 예이다.

```java
// Stream 은 주로 Collection, Arrays 에서 생성한다.
// I/o resources(ex, File), Generators, Stream ranges, Pattern 등에서도 생성할 수 있다.
List<string> names = Arrays.asList("foo", "bar", "baz");
names.stream();

// Array 에서 stream 생성
Double[] d = {3.1, 3.2, 3.3};
Arrays.stream(d);

// 직접 stream 생성
Stream<Intger> str = Stream.of(1, 2);
```

Stream 은 스트림생성, 중개연산, 최종연산 과 같이 3 가지로 구분한다. 마치 `스트림생성().중개연산().최종연산()` 과 같은 모양이다.

다음은 중개연산의 예이다.

```java
// Filter
List<string> names = Arrays.asList("foo", "bar", "baz");
Stream<String> a = names.stream().filter(x -> x.contains("f"));

// Map
names.parallelStream().map(x -> return x.concat("s")).forEach(x -> System.out.println(x));

// Peek
names.stream().peek(System.out::println);

// Sorted
names.stream().sorted().peek(System.out::println);

// Limit
names.stream().filter(x -> return x.contains("f")).limit(1);

// Distinct
names.stream().distinct().peek(System.out::println);

// Skip
names.stream().skip(1).peek(System.out::println);

// mapToInt, mapToLong, mapToDouble
List<string> nums = Arrays.asList("1", "2", "3");
nums.stream().mapToInt().peek(System.out::println); 
```

다음은 최종연산의 예이다.

```java
// count, min, max, sum, average
List<Integer> nums = Arrays.asList(1, 2, 3);
System.out.println(nums.stream().count());
System.out.println(nums.stream().min());
System.out.println(nums.stream().max());
System.out.println(nums.stream().sum());
System.out.println(nums.stream().average());

// reduce
System.out.println(nums.stream().reduce());

// forEach
nums.stream().forEach(x -> System.out.println(x));

// collect
Set<Integer> set = nums.stream().collect(Collectors.toSet());

// iterator
Iterator<String> it = nums.stream().iterator();
while(it.hasNext()) {
   System.out.println(iter.next());
}

// noneMatch, anyMatch, allMatch
System.out.println(nums.stream().noneMatch(x -> x > 10)); //false
System.out.println(nums.stream().anyMatch(x -> x > 10)); //false
System.out.println(nums.stream().allMatch(x -> x > 10)); //false
```

`Stream::parallelStream` 을 이용하면 병렬연산을 쉽게 할 수 있다.

```java
public class Solution {
	private Stream<String> crawlMulti(String startUrl, HtmlParser htmlParser,
													String hostname, Set<String> seen) {
		Stream<String> stream = htmlParser.getUrls(startUrl)
				.parallelStream()
				.filter(url -> url.substring(7, url.length()).startsWith(hostname))
				.filter(url -> seen.add(url))
				.flatMap(url -> crawlMulti(url, htmlParser, hostname, seen));
		return Stream.concat(Stream.of(startUrl), stream);
	}
	private String getHostname(String url) {
		return url.substring(7).split("/")[0];
	}
	public List<String> crawl(String startUrl, HtmlParser htmlParser) {
		String hostname = getHostname(startUrl);
		Set<String> seen = ConcurrentHashMap.newKeySet();
		seen.add(startUrl);
		return crawlMulti(startUrl, htmlParser, hostname, seen)
				.collect(Collectors.toList());
	}	
}
```

## Java code coverage library

test code 가 어느정도 코드를 covering 하고 있는지 보여주는 library 이다.

* [JaCoCo Java Code Coverage Library](https://www.eclemma.org/jacoco/)

## Java Byte Code Manipulation library

java byte code 를 수정할 수 있는 library 이다. 

* [Byte Buddy](https://bytebuddy.net)
  * Byte Buddy 는 [ASM](https://asm.ow2.io/) 보다 쉽다.

## Java Lombok

* [Project Lombok](https://projectlombok.org/)
* [lombok을 잘 써보자! (1)](http://wonwoo.ml/index.php/post/1607)

Class 에 Annotation 을 추가하면 getter, setter, toString, hasCode, equals, constructor 를 생성한다.

* `@Data` 
  * `staticConstructor` 속성을 이용하면 static constructor 를 생성해 준다.
  * `canEuqal` 도 생성해준다. 이것은 instanceof 와 같다.  

```java
@Data(staticConstructor = "of")
public class DataObject {
  private final Long id;
  private String name;
}
```

다음과 같이 사용할 수 있다.

```java
DataObject dataObject = DataObject.of(1L);
```

그러나 다음과 같이 생성할 수는 없다.

```java
DataObject dataObject = new DataObject();
```

* `XXXXArgsConstroctor`
  * `NoArgsConstructor`
    * Default constructor 를 생성해 준다.
  * `AllArgsConstructor`
    * 모든 필드의 생성자를 생성해 준다. 
  * `RequiredArgsConstructor`
    * Required constructor (필수생성자) 를 생성해 준다.
    * final field 의 constructor 를 만들어 준다???

  * `staticName` 속성은 static constructor 를 만들어 준다.
  * `access` 속성은 접근제한을 할 수 있다.
  * ``onConstructor` constructor 에 annotation 을 달아준다.
  
```java
@RequiredArgsConstructor(staticName = "of", onConstructor = @__(@Inject))
public class Foo {
  private final Long id;
  private final String name;
}
```

이 코드는 Annotation process 가 수행되면 다음과 같은 코드가 만들어 진다.

```java
class Foo {
  private final Long id;
  private final String name;

  @Inject
  private Foo(Long id, String name) {
    this.id = id;
    this.name = name;
  }
  public static Foo of(Long id, String name) {
    return new Foo(id, name);
  }
}
```

* `@Getter, @Setter`
  * getter, setter 를 만들어 준다.
  * `value` 속성은 접근을 제한할 수 있다.
  * `onMethod` 속성은 method 에 annotation 을 달 수 있다.

```java
public class Foo {
  @Getter(value = AccessLevel.PACKAGE, onMethod = @__({@NonNull, @Id}))
  private Long id;
}
```

의 코드는 Annotatoin process 가 수행되면 다음과 같은 코드를 생성한다.

```java
class Foo {
  private Long id;

  @Id
  @NonNull
  Long getId() {
    return id;
  }
}
```

`@Getter` 는 `lazy` 속성을 갖는다. `@Setter` 는 `onParam` 속성을 갖는다.

```java
@Getter(value = AccessLevel.PUBLIC, lazy = true)
private final String name = bar();

private String bar() {
  return "BAR";
}
```

final field 에만 `lazy=true` 를 적용할 수 있다. `lazy=false` 이면 Object 를 생성할 때 `bar()` 를 호출한다. 그러나 `lazy=true` 이면 `getName()` 를 호출할 때 `bar()` 를 호출한다.

다음은 `@Setter` 의 `onParam` 이다.

```java
@Setter(onParam = @__(@NotNull))
private Long id;
```

Annotation process 가 수행되면 다음과 같은 코드가 만들어 진다.

```java
class Baz {
  private Long id;

  public void setId(@NotNull Long id) {
    this.id = id;
  }
}
```

Parameter 에 Annotaion 을 달아준다.

* `@EqualsAndHashCode, @ToString`
  * `@EqualsAndHashCode` 는 `hashcode, equals` 를 생성한다.
  * `@ToString` 은 `toString()` 을 생성한다.
  * `exclude` 속성은 특정 field 를 제외한다.
  * `of` 속성은 특정 field 를 포함한다.
  * `callSuper` 속성은 상위 클래스의 호출여부를 정한다.
  * `doNotUseGetters` 속성은 getter 사용여부를 정한다.

다음의 코드는 `id` 만 `hashCode, equals, toString` 을 생성한다. 

```java
@EqualsAndHashCode(of = "id")
@ToString(exclude = "name")
public class Foo {
  private Long id;
  private String name;
}
```

## JVM Options

* `-XX:+UseG1GC`
  * [JVM 메모리 구조와 GC](https://johngrib.github.io/wiki/jvm-memory/)

* `-XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=${BASEPATH}/heapdump_$(date '+%Y%m%d%H%M').hprof"`
  * JVM 이 OutOfMemoryError 로 종료되었을 때 지정된 경로로 heap dump 가 생성된다.
  * [Java Memory Analysis](https://kwonnam.pe.kr/wiki/java/memory)

* `-Xms1024m -Xmx1024m`
  * memory start size and memory max size
  * [JVM 메모리 관련 설정](https://epthffh.tistory.com/entry/JVM-%EB%A9%94%EB%AA%A8%EB%A6%AC-%EA%B4%80%EB%A0%A8-%EC%84%A4%EC%A0%95)

## Thread Dump, Heap Dump

* [스레드 덤프 분석하기](https://d2.naver.com/helloworld/10963)
* [THREAD DUMP, HEAP DUMP 생성 및 분석 방법](https://yenaworldblog.wordpress.com/2018/05/09/thread-dump-%EC%83%9D%EC%84%B1-%EB%B0%8F-%EB%B6%84%EC%84%9D-%EB%B0%A9%EB%B2%95/)

----

Java application 이 느려졌다면 thread 들이 dead lock 인지 waiting 상태인지 확인해 보아야 한다.
thread dump 를 확인해 보자. [Visual VM](https://visualvm.github.io/) 을 이용하면 JMX 를 이용하여
remote application 의 Thread Dump 를 실시간으로 확인 가능하다. 물론 snapshot 도 가능하다.

다음은 jcmd 로 thread dump 를 확인하는 방법이다.

```bash
$ sudo ps -aux | grep java
$ jcmd ${pid} Thread.print > a.txt
```

Java application 의 Heap 상태를 보고 싶다면 heap dump 를 확인해 보자. 역시 [Visual VM](https://visualvm.github.io/)
를 이용하면 JMX 를 이용하여 Heap Dump 의 snapshot 을 확인할 수 있다.

다음은 jamp 으로 heap dump file 을 생성하는 방법이다. [7 Ways to Capture Java Heap Dumps](https://dzone.com/articles/how-to-capture-java-heap-dumps-7-options)

```bash
# jmap -dump:format=b,file=<file-path> <pid>
$ jmap -dump:format=b,file=/home/iamslash/tmp/heapdump.bin 1111
```

# Quiz

* Private Constructor
* Return from Finally
* Final, etc.
* Generics vs. Templates
* TreeMap, HashMap, LinkedHashMap
* Object Reflection
* Lambda Expressions
* Lambda Random
