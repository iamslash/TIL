<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Abstract](#abstract)
- [Usage](#usage)
    - [Keywords](#keywords)
    - [Collections compared c++ container](#collections-compared-c-container)
    - [Collections](#collections)
    - [Collection Framework](#collection-framework)
    - [Collection Implementations](#collection-implementations)

<!-- markdown-toc end -->


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

## Collections

## Collection Framework

- [Java Collection Framework Technote](https://docs.oracle.com/javase/8/docs/technotes/guides/collections/)
- [Outline of the Collections Framework](http://docs.oracle.com/javase/8/docs/technotes/guides/collections/reference.html)
- ![Collection Framework](https://upload.wikimedia.org/wikibooks/en/thumb/c/ca/Java_collection_implementation.jpg/700px-Java_collection_implementation.jpg)

## Collection Implementations

| Interface | Hash Table         | Resizable Array                 | Balanced Tree | Linked List        | Hash Table + Linked List        |
| :-------- | :----------------- | :-----------------------------: | :--------     | :----------------- | :-----------------------------: |
| Set       | HashSset           |                                 | TreeSet       |                    | LinkedHashSet                   |
| List      |                    | ArrayList                       |               | LinkedList         |                                 |
| Deque     |                    | ArrayDeque                      |               | LinkedList         |                                 |
| Map       | HashMap            |                                 | TreeMap       |                    | LinkedHashMap                   |
