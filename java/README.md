- [Abstract](#abstract)
- [Materials](#materials)
- [Basic Usages](#basic-usages)
  - [Compile, Execution](#compile-execution)
  - [Reserved Words](#reserved-words)
  - [Useful Keywords](#useful-keywords)
  - [Collections compared c++ container](#collections-compared-c-container)
  - [Collection Examples](#collection-examples)
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
  - [What's new Java8](#whats-new-java8)
- [Advanced Usage](#advanced-usage)
  - [jvm architecture](#jvm-architecture)
  - [jvm garbage collector](#jvm-garbage-collector)
- [Quiz](#quiz)

-------------------------------------------------------------------------------

# Abstract

java를 정리한다.

# Materials

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

## Collections compared c++ container

| c++                  | java                            |
|:---------------------|:--------------------------------|
| `if, else`           | `if, else`                      |
| `for, while`         | `for, while`                    |
| `array`              | `Collections.unmodifiableList`  |
| `vector`             | `ArrayList`                     |
| `deque`              | `Deque, ArrayDeque`             |
| `forward_list`       | ``                              |
| `list`               | `List, LinkedList`              |
| `stack`              | `Stack, LinkedList`             |
| `queue`              | `Queue, LinkedList`             |
| `priority_queue`     | `Queue, PriorityQueue`          |
| `set`                | `SortedSet, TreeSet`            |
| `multiset`           | ``                              |
| `map`                | `SortedMap, TreeMap`            |
| `multimap`           | ``                              |
| `unordered_set`      | `Set, HashSet`                  |
| `unordered_multiset` | ``                              |
| `unordered_map`      | `Map, HashMap`                  |
| `unordered_multimap` | ``                              |

* core interfaces in Collections

```
            Collection                Map
         /    |    |    \              |
       Set List  Queue Deque       SortedMap 
        |
       SortedSet
```

* core classes in Collections
  * [Collections in java @ geeksforgeeks](https://www.geeksforgeeks.org/collections-in-java-2/)
  * [Collections in Java @ javapoint](https://www.javatpoint.com/collections-in-java)


```
```

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

* ArrayList
* Deque, ArrayDeque
* List, LinkedList
* Stack, LinkedList
* Queue, LinkedList
* Queue, PriorityQueue
* SortedSet, TreeSet
* SortedMap, TreeMap
* Set, HashSet
* Map, HashMap

## Collection Framework

- [Java Collection Framework Technote](https://docs.oracle.com/javase/8/docs/technotes/guides/collections/)
- [Outline of the Collections Framework](http://docs.oracle.com/javase/8/docs/technotes/guides/collections/reference.html)
- [Collection Framework](https://upload.wikimedia.org/wikibooks/en/thumb/c/ca/Java_collection_implementation.jpg/700px-Java_collection_implementation.jpg)

## Collection Implementations

| Interface | Hash Table         | Resizable Array                 | Balanced Tree | Linked List        | Hash Table + Linked List        |
| :-------- | :----------------- | :-----------------------------: | :--------     | :----------------- | :-----------------------------: |
| Set       | HashSset           |                                 | TreeSet       |                    | LinkedHashSet                   |
| List      |                    | ArrayList                       |               | LinkedList         |                                 |
| Deque     |                    | ArrayDeque                      |               | LinkedList         |                                 |
| Map       | HashMap            |                                 | TreeMap       |                    | LinkedHashMap                   |


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
      if( x == 10 ) {
         System.out.print("Value of X is 10");
      }else if( x == 20 ) {
         System.out.print("Value of X is 20");
      }else if( x == 30 ) {
         System.out.print("Value of X is 30");
      }else {
         System.out.print("This is else statement");
      }
//// switch
      // char grade = args[0].charAt(0);
      char grade = 'C';
      switch(grade) {
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
      while( x < 20 ) {
         System.out.print("value of x : " + x );
         x++;
         System.out.print("\n");
      }
//// for
      for(int x = 10; x < 20; x = x + 1) {
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
      for (int x : numbers ) {
         if( x == 30 ) {
            break;
         }
         System.out.print( x );
         System.out.print("\n");
      }
//// continue
      int [] numbers = {10, 20, 30, 40, 50};
      for (int x : numbers ) {
         if( x == 30 ) {
            continue;
         }
         System.out.print( x );
         System.out.print("\n");
      }
//// range based for
      int [] numbers = {10, 20, 30, 40, 50};
      for(int x : numbers ) {
         System.out.print( x );
         System.out.print(",");
      }
      System.out.print("\n");
      String [] names = {"James", "Larry", "Tom", "Lacy"};
      for( String name : names ) {
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

anonymous function 처럼 interface 를 구현한 클래스를 이름없이 생성할 수 있다. 다음은 `Runnable` 인터페이스를 상속받는 클래스를 이름없이 생성하는 예이다.

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

public boolean isWeekend( int day ) {
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

public boolean isWeekend( DaysOfTheWeek day ) {
   return( day == SATURDAY || day == SUNDAY );
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
   private DaysOfTheWeekFieldsInterfaces( final boolean isWeekend ) {
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

annotation 은 특수한 interface 이다. `@interface` 를 이용하여 다음과 같이 선언한다. class, method, field 등등에 `key=value` 형태의 추가정보를 주입할 수 있다.

```java
public @interface SimpleAnnotation {
}

public @interface SimpleAnnotationWithAttributes {
   String name();
   int order() default 0;
}
```

예를 들어 다음과 같이 `SimpleAnnotationWithValue` 를 선언하고 field `aaa` 에 사용하면 `aaa` 에 `value=new annotation` 가 추가된다.

```java
public @interface SimpleAnnotationWithValue {
   String value();
}

@SimpleAnnotationWithValue( "new annotation" )
public int aaa;
```

다음은 builtin annotation 들이다.

| name | desc | type |
|:-----|:-----|:-----|
| `@Deprecated` | something deprecated | |
| `@Override` | overrided method | |
| `@SuppressWarnings` | suppress compile warnings | |
| `@SafeVarargs` | suppress variable arguments warning | |
| `@Retention` | retention of annotation | `SOURCE, CLASS, RUNTIME` |
| `@Target` | target of annotation | `ANNOTATION_TYPE, CONSTRUCTOR, FIELD, LOCAL_VARIABLE, METHOD, PACKAGE, PARAMETER, TYPE, TYPE_PARAMETER, TYPE_USE` |
| `@Documented` | documented in javadoc | |
| `@Inherited` | this annotation will be inherited | |
| `@FunctionalInterface` | for functional interface | |
| `@Repeatable` | repeatable annotation | |

`@Target` 를 사용하면 다음과 같이 annotation 의 대상이 되는 자료형을 정할 수 있다.

```java
import java.lang.annotation.ElementType;
import java.lang.annotation.Target;
@Target( { ElementType.FIELD, ElementType.METHOD } )
public @interface AnnotationWithTarget {
}
```

annotation 은 기본적으로 상속되지 않는다. 그러나 `@Inherited` 를 사용하면 상속된다.

```java
@Target( { ElementType.TYPE } )
@Retention( RetentionPolicy.RUNTIME )
@Inherited
@interface InheritableAnnotation {
}

@InheritableAnnotation
public class Parent {
}

public class Child extends Parent {
}
```

다음은 `@Repeatable` 를 사용한 예이다.

```java
@Target( ElementType.METHOD )
@Retention( RetentionPolicy.RUNTIME )
public @interface RepeatableAnnotations {
   RepeatableAnnotation[] value();
}
@Target( ElementType.METHOD )
@Retention( RetentionPolicy.RUNTIME )
@Repeatable( RepeatableAnnotations.class )
public @interface RepeatableAnnotation {
   String value();
};
@RepeatableAnnotation( "repeatition 1" )
@RepeatableAnnotation( "repeatition 2" )
public void performAction() {
   // Some code here
}
```

## Generics

다음은 generic interface 의 예이다. actual type 은 generic interface 를 구현한 class 를 작성할 때 사용한다.

```java
public interface GenericInterfaceOneType< T > {
   void performAction( final T action );
}

public interface GenericInterfaceSeveralTypes< T, R > {
   R performAction( final T action );
}

public class ClassImplementingGenericInterface implements GenericInterfaceOneType< String > {
   @Override
   public void performAction( final String action ) {
      // Implementation here
   }
}
```

다음은 generic class 의 예이다. actual type 은 generic class 의 instance 를 생성하거나 class 에서 상속받을 때 사용한다.

```java
public class GenericClassOneType< T > {
   public void performAction( final T action ) {
      // Implementation here
   }
}

public class GenericClassImplementingGenericInterface< T > implements GenericInterfaceOneType< T > {
   @Override
   public void performAction( final T action ) {
      // Implementation here
   }
}
```

다음은 generic method 의 예이다. actual type 은 generic method 를
호출할 때 사용한다.

```java
public< T, R > R performAction( final T action ) {
   final R result = ...;
   // Implementation here
   return result;
}

protected abstract< T, R > R performAction( final T action );
static< T, R > R performActionOn( final Collection< T > action ) {
   final R result = ...;
   // Implementation here
   return result;
}  

public class GenericMethods< T > {
   public< R > R performAction( final T action ) {
      final R result = ...;
      // Implementation here
      return result;
   }
   public< U, R > R performAnotherAction( final U action ) {
      final R result = ...;
      // Implementation here
      return result;
   }
}

public class GenericMethods< T > {
   public GenericMethods( final T initialAction ) {
      // Implementation here
   }
   public< J > GenericMethods( final T initialAction, final J nextAction ) {
      // Implementation here
   }
}
```

generic 의 type 에 primitive type 은 사용할 수 없다. primitive Wrapper type 를 사용해야 한다. generic method 의 경우 argument 로 primitive type 이 전달될 때 primitive wrapper type 으로 형변환 된다. 이것을 boxing 이라고 한다.

```java
final List< Long > longs = new ArrayList<>();
final Set< Integer > integers = new HashSet<>();

final List< Long > longs = new ArrayList<>();
longs.add( 0L ); // ’long’ is boxed to ’Long’
long value = longs.get( 0 ); // ’Long’ is unboxed to ’long’
// Do something with value
```

generic 은 compile time 에 type erasure 를 한다. 즉 type 결정을 runtime 에 하기 위해 compile time 에 generic type 을 지운다. 따라서 다음과 같은 코드는 compile 할 때 method 가 중복 선언되었다는 오류를 발생한다.

```java
void sort( Collection< String > strings ) {
   // Some implementation over strings heres
}
void sort( Collection< Number > numbers ) {
   // Some implementation over numbers here
}
```
generic 의 array 는 만들 수 없다.

```java
public< T > void performAction( final T action ) {
   T[] actions = new T[0];
}
```

generic type 을 `extends` 를 사용하여 자신 혹은 후손으로 제한할 수 있다.

```java
public< T extends InputStream > void read( final T stream ) {
   // Some implementation here
}
public< T extends Serializable > void store( final T object ) {
   // Some implementation here
}  
public< T, J extends T > void action( final T initial, final J next ) {
   // Some implementation here
}
public< T extends InputStream &amp; Serializable > 
void storeToRead( final T stream ) {
   // Some implementation here
}
public< T extends Serializable &amp; Externalizable &amp; Cloneable > 
void persist(final T object ) {
   // Some implementation here
}
```

다음은 method 의 argument 에 generic type 을 사용한 예이다.

```java
public void store( final Collection< ? extends Serializable > objects ) {
   // Some implementation here
}
public void store( final Collection< ? > objects ) {
   // Some implementation here
}
public void interate( final Collection< ? super Integer > objects ) {
   // Some implementation here
}
```

## Concurrency

* [Java Concurrency and Multithreading Tutorial](http://tutorials.jenkov.com/java-concurrency)
  * Java 의 Concurrency 에 대해 기가 막히게 설명한 글

----

TODO

## What's new Java8

TODO

# Advanced Usage

## jvm architecture

* [jvm @ TIL](/jvm/README.md)

## jvm garbage collector

* [Java Garbage Collection Basics](https://www.oracle.com/webfolder/technetwork/tutorials/obe/java/gc01/index.html)
  * garbage collector 의 기본 원리에 대해 알 수 있다.

![](https://www.oracle.com/webfolder/technetwork/tutorials/obe/java/gc01/images/gcslides/Slide5.png)

jvm 의 gc 는 크게 `Young Generation, Old Generation, Permanent Generation` 으로 나누어 진다. 영원히 보존해야 되는 것들은 `Permanent Generation` 으로 격리된다. `Young Generation` 에서 `minor collect` 할때 마다 살아남은 녀석들중 나이가 많은 녀석들은 `Old Generation` 으로 격리된다. 

`Young Generation` 은 다시 `eden, S0, S1` 으로 나누어 진다. `eden` 이 꽉 차면 `minor collect` 이벤트가 발생하고 `eden, S0` 혹은 `eden, S1` 의 `unreferenced object` 는 소멸되고 `referenced object` 는 나이가 하나 증가하여 `S1` 혹은 `S0` 으로 옮겨진다. 나이가 많은 녀석들은 `Old Genration` 으로 옮겨진다. `eden, S0` 과 `eden, S1` 이 교대로 사용된다.

# Quiz

* Private Constructor
* Return from Finally
* Final, etc.
* Generics vs. Templates
* TreeMap, HashMap, LinkedHashMap
* Object Reflection
* Lambda Expressions
* Lambda Random