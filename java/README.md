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
- [Quiz](#quiz)

-------------------------------------------------------------------------------

# Abstract

java를 정리한다.

# Materials

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
- strictfp
- native
- transient

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

# Quiz

* Private Constructor
* Return from Finally
* Final, etc.
* Generics vs. Templates
* TreeMap, HashMap, LinkedHashMap
* Object Reflection
* Lambda Expressions
* Lambda Random