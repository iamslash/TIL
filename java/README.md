- [Abstract](#abstract)
- [Usage](#usage)
    - [Keywords](#keywords)
    - [Collections compared c++ container](#collections-compared-c-container)
    - [Collection Examples](#collection-examples)
    - [Collection Framework](#collection-framework)
    - [Collection Implementations](#collection-implementations)

-------------------------------------------------------------------------------
# Abstract

java를 정리한다.

# Usage

## Keywords

- volatile
- strictfp
- native
- transient

## Collections compared c++ container


| c++                  | java                  |
|:---------------------|:--------------------------------|
| `if, else`           | `if, else`                      |
| `for, while`         | `for, while`                    |
| `array`              | `Collections.unmodifiableList`  |
| `vector`             | `ArrayList`                     |
| `deque`              | `Deque, ArrayDeque` |
| `forward_list`       | ``                              |
| `list`               | `List, LinkedList`              |
| `stack`              | `Stack, LinkedList`             |
| `queue`              | `Queue, LinkedList`             |
| `priority_queue`     | `Queue, PriorityQueue`          |
| `set`                | `SortedSet, TreeSet`            |
| `multiset`           | ``                              |
| `map`                | `SortedMap, TreeMap`                  |
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
