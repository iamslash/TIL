﻿- [Abstract](#abstract)
- [Essentials](#essentials)
- [Materials](#materials)
- [Basic](#basic)
  - [Build And Run](#build-and-run)
  - [Hello World](#hello-world)
  - [Reserved Words](#reserved-words)
  - [min, max values](#min-max-values)
  - [abs, fabs](#abs-fabs)
  - [Bit Manipulation](#bit-manipulation)
  - [String](#string)
  - [Random](#random)
  - [Formatted Strings](#formatted-strings)
  - [Object Introspection](#object-introspection)
  - [Data Types](#data-types)
  - [Control Flow Statements](#control-flow-statements)
    - [Decision Making Statements](#decision-making-statements)
    - [Looping Statements](#looping-statements)
    - [for and if oneline](#for-and-if-oneline)
  - [Collections Compared to c++ containers](#collections-compared-to-c-containers)
  - [Collections](#collections)
    - [tuple](#tuple)
    - [list](#list)
    - [deque](#deque)
    - [PriorityQueue](#priorityqueue)
    - [heapq](#heapq)
    - [set](#set)
    - [dict](#dict)
    - [collections.Counter](#collectionscounter)
    - [collections.defaultdict](#collectionsdefaultdict)
    - [collections.namedtuple](#collectionsnamedtuple)
    - [collections.ChainMap](#collectionschainmap)
    - [collections.OrderedDict](#collectionsordereddict)
    - [enum (python 3.4+)](#enum-python-34)
  - [Sort](#sort)
  - [Search](#search)
  - [Multidimensional Array](#multidimensional-array)
  - [Built-in Functions](#built-in-functions)
  - [Round](#round)
  - [Classes](#classes)
  - [lambdas](#lambdas)
  - [slice](#slice)
  - [Underscore](#underscore)
  - [Args, Kwargs](#args-kwargs)
  - [Comprehension](#comprehension)
  - [Generator](#generator)
  - [Map, Filter, Reduce](#map-filter-reduce)
  - [Set](#set-1)
  - [Ternary Operators](#ternary-operators)
  - [Multiple Return](#multiple-return)
  - [Exception](#exception)
  - [Dunder methods (Magic methods)](#dunder-methods-magic-methods)
  - [pdb](#pdb)
  - [Async, Await](#async-await)
  - [Performance Check](#performance-check)
  - [`__slots__`](#__slots__)
  - [Enumerate](#enumerate)
  - [Metaclass](#metaclass)
  - [Weakref](#weakref)
  - [Memory Leak](#memory-leak)
  - [gc](#gc)
  - [dependencies](#dependencies)
- [Advanced](#advanced)
  - [walrus operator](#walrus-operator)
  - [Pythonic Way](#pythonic-way)
  - [Open Function](#open-function)
  - [Decorator](#decorator)
  - [Virtual Environment](#virtual-environment)
  - [one-liners](#one-liners)
  - [C Extension](#c-extension)
  - [python 2+3](#python-23)
  - [Coroutine](#coroutine)
  - [Function Caches](#function-caches)
  - [Context Managers](#context-managers)
  - [typing — Support for type hints @ python3](#typing--support-for-type-hints--python3)
  - [itertools — Functions creating iterators for efficient looping](#itertools--functions-creating-iterators-for-efficient-looping)
  - [functools — Higher-order functions and operations on callable objects](#functools--higher-order-functions-and-operations-on-callable-objects)
  - [Tracing Memory](#tracing-memory)
- [Libraries](#libraries)
  - [regex](#regex)
  - [click](#click)
  - [unittest](#unittest)
  - [pytest](#pytest)
  - [urwid](#urwid)
  - [fabric](#fabric)
  - [flake8](#flake8)
  - [objgraph](#objgraph)
  - [MySQL-python 1.2.5 on python2.7 build error, library not found for -lssl](#mysql-python-125-on-python27-build-error-library-not-found-for--lssl)
- [Style Guide](#style-guide)
- [Refactoring](#refactoring)
- [Effective Python](#effective-python)
- [GOF Design Pattern](#gof-design-pattern)
- [Architecture](#architecture)
  
----

# Abstract

python3 에 대해 정리한다.

# Essentials

* [Python Tips](http://book.pythontips.com/en/latest/)
  * intermediate python
  * [번역](https://ddanggle.gitbooks.io/interpy-kr/content/)
* [python 3 doc](https://docs.python.org/3/contents.html)
  * 목차분석이 필요함
* [James Powell: So you want to be a Python expert? | PyData Seattle 2017 @ youtube](https://www.youtube.com/watch?v=cKPlPJyQrt4&list=WL&index=5&t=0s)
  * dunder method, metaclass, decorator, context manager, generator
* [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
* [Practical Python](https://github.com/dabeaz-course/practical-python)
  * [Practical Python Programming Contents](https://github.com/dabeaz-course/practical-python/blob/master/Notes/Contents.md)
* [Python Mastery](https://github.com/dabeaz-course/python-mastery)
  * [Advanced Python Mastery Exercises Index](https://github.com/dabeaz-course/python-mastery/blob/main/Exercises/index.md)
  * [pdf](https://github.com/dabeaz-course/python-mastery/blob/main/PythonMastery.pdf)
  
# Materials

* [초보자를 위한 파이썬 200제](http://www.infopub.co.kr/new/include/detail.asp?sku=06000238)
  * [목차](http://www.infopub.co.kr/common/book_contents/06000238.html)
  * [src](http://www.infopub.co.kr/new/include/detail.asp?sku=06000238#)
* [James Powell - Advanced Metaphors in Coding with Python @ youtube](https://www.youtube.com/watch?v=R2ipPgrWypI)
* [파이썬 생존 안내서](https://www.slideshare.net/sublee/ss-67589513)
  * 듀랑고를 제작한 왓스튜디오의 이흥섭 PT
* [awesome-python](https://awesome-python.com/)
  * A curated list of awesome python things
* [python package index](https://pypi.python.org/pypi)
  * library검색이 용이하다.

# Basic

## Build And Run

```bash
$ python3 a.py
```

## Hello World

```py
print('Hello World')
```

## Reserved Words

```py
and
except
lambda
with
as
finally
nonlocal
while
assert
false
None
yield
break
for
not
class
from
or
continue
global
pass
def
if
raise
del
import
return
elif
in
True
else
is
try
```

## min, max values

```py
# python3 int is unbounded but it depends on wordsize
print(sys.maxsize) # for signed int
print(sys.maxsize * 2 + 1) # for unsigned int
# python3 float
print(float("inf"))
import sys
print(sys.float_info)
print(sys.float_info.max)

# python2
print(sys.maxint)
print(-sys.maxint - 1)
```

## abs, fabs

* [Python | fabs() vs abs()](https://www.geeksforgeeks.org/python-fabs-vs-abs/)

```py
# abs will return int or float depending upon the argument
abs(-12) # 12
abs(12)  # 12
abs(-12.0) # 12.0
abs(12.0)  # 12.0
import math
math.fabs(-12.0) # 12.0
math.fabs(12.0)  # 12.0
```

## Bit Manipulation

```py
# https://www.tutorialspoint.com/python/bitwise_operators_example.htm
a = 60            # 60 = 0011 1100 
b = 13            # 13 = 0000 1101 
c = 0

c = a & b;        # 12 = 0000 1100
c = a | b;        # 61 = 0011 1101 
c = a ^ b;        # 49 = 0011 0001
c = ~a;           # -61 = 1100 0011
c = a << 2;       # 240 = 1111 0000
c = a >> 2;       # 15 = 0000 1111
```

## String

> * [Built-in Types | String Methods @ python](https://docs.python.org/3/library/stdtypes.html#string-methods)

```py
# Sub string
s = 'Hello World'
print(s[0:5])

# Convert string, int
int('57')
float('57')
str(57)
str(57.0)

# Split
>>> 'ab c\n\nde fg\rkl\r\n'.splitlines()
['ab c', '', 'de fg', 'kl']
>>> 'ab c\n\nde fg\rkl\r\n'.splitlines(keepends=True)
['ab c\n', '\n', 'de fg\r', 'kl\r\n']
# splitlines vs split
#   splitlines returns an empty list for the empty string, and a terminal line break does not result in an extra line:
>>> "".splitlines()
[]
>>> "One line\n".splitlines()
['One line']
>>> ''.split('\n')
['']
>>> 'Two lines\n'.split('\n')
['Two lines', '']
```

## Random

```py
>>> import random
# random
>>> random.random()
# randrange
>>> random.randrange(1,7)
>>> random.randrange(1,7)
# shuffle
>>> abc = ['a', 'b', 'c', 'd', 'e']
>>> random.shuffle(abc)
>>> abc
['a', 'd', 'e', 'b', 'c']
>>> random.shuffle(abc)
>>> abc
['e', 'd', 'a', 'c', 'b']
# choice
>>> random.choice(abc)
'a'
>>> random.choice(abc)
'd'
```

## Formatted Strings

```py
print('Hello World')
a, b = 1, 2
# f string, literal string interpolation
print(f'a: {a}, b{b}')

# Recommend f-string which is the fastest.
# https://brownbears.tistory.com/421

# % operator
import timeit 
timeit.timeit("""name = "Eric" ... age = 74 ... '%s is %s.' % (name, age)""", number = 10000)
# 0.003324444866599663

# str.format()
import timeit 
timeit.timeit("""name = "Eric" ... age = 74 ... '{} is {}.'.format(name, age)""", number = 10000) 
# 0.004242089427570761

# f-string
import timeit 
timeit.timeit("""name = "Eric" ... age = 74 ... f'{name} is {age}.'""", number = 10000) 
# 0.0024820892040722242
```

## Object Introspection

* `dir`

```py
my_list = [1, 2, 3]
dir(my_list)
# Output: ['__add__', '__class__', '__contains__', '__delattr__', '__delitem__',
# '__delslice__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__',
# '__getitem__', '__getslice__', '__gt__', '__hash__', '__iadd__', '__imul__',
# '__init__', '__iter__', '__le__', '__len__', '__lt__', '__mul__', '__ne__',
# '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rmul__',
# '__setattr__', '__setitem__', '__setslice__', '__sizeof__', '__str__',
# '__subclasshook__', 'append', 'count', 'extend', 'index', 'insert', 'pop',
# 'remove', 'reverse', 'sort']
```

* `type, id`

```py
print(type(''))
# Output: <type 'str'>

print(type([]))
# Output: <type 'list'>

print(type({}))
# Output: <type 'dict'>

print(type(dict))
# Output: <type 'type'>

print(type(3))
# Output: <type 'int'>

name = "Yasoob"
print(id(name))
# Output: 139972439030304
```

* inspect module

```py
import inspect
print(inspect.getmembers(str))
# Output: [('__add__', <slot wrapper '__add__' of ... ...
```

## Data Types

> [Built-in Types @ python3](https://docs.python.org/3/library/stdtypes.html)

Numeric Type 은 `int, float, complex` 가 있다.
String Type 은 `str` 이 있다. 

```py
>>> type(3)
<class 'int'>
>>> type(3.3)
<class 'float'>
>>> type(complex(2, 3))
<class 'complex'>
>>> type("Hello")
<class 'str'>
```

Data Type 의 크기는 다음과 같다. [Python의 데이터 타입 크기 및 기억 범위](https://ilikemediumrare.tistory.com/5)

| Category | Data Type | Data Limit | Data Range |
|---|---|---|---|
| String | `str` | Infinite | Infinite |
| Integer | `int` | Infinite | Infinite |
| Real | `float` | 8 byte | `4.9×10^-324~1.8×10^308` |
|      | `complex` | 16 byte | `4.9×10^-324~1.8×10^308` |

## Control Flow Statements

### Decision Making Statements

* [4. More Control Flow Tools @ python3](https://docs.python.org/3/tutorial/controlflow.html)

```py
if x < 0:
    x = 0
    print('Negative changed to zero')
elif x == 0:
    print('Zero')
elif x == 1:
    print('Single')
else:
    print('More')
```

### Looping Statements

> [Python For Loops @ w3schools](https://www.w3schools.com/python/python_for_loops.asp)

```py
# basic
fruits = ["apple", "banana", "cherry"]
for x in fruits:
  print(x)

# Looping Through a String
for x in "banana":
  print(x)

# The break Statement
fruits = ["apple", "banana", "cherry"]
for x in fruits:
  print(x)
  if x == "banana":
    break

# The continue Statement
fruits = ["apple", "banana", "cherry"]
for x in fruits:
  if x == "banana":
    continue
  print(x)

# range(start, exclusive end, step), [0..5]
for x in range(6):
  print(x)
# [2..5]
for x in range(2, 6):
  print(x)
# [2, 5]
for x in range(2, 6, 3):
  print(x)
# reverse range(), [5..0]
for x in range(5, -1, -1):
  print(x)
else:
  print("Finally finished!")

# Else in For Loop. print a message when the loop has ended.
for x in range(6):
  print(x)
else:
  print("Finally finished!")

# Nested Loops
adj = ["red", "big", "tasty"]
fruits = ["apple", "banana", "cherry"]
for x in adj:
  for y in fruits:
    print(x, y)

# The pass Statement
for x in [0, 1, 2]:
  pass
```

### for and if oneline

> * [[python] for문, if문 한 줄로 코딩하기 (for and if in one line)](https://leedakyeong.tistory.com/entry/python-for%EB%AC%B8-if%EB%AC%B8-%ED%95%9C-%EC%A4%84%EB%A1%9C-%EC%BD%94%EB%94%A9%ED%95%98%EA%B8%B0)

```py
# print one dimension list
for a in A:
  print(a)
# one line
>>> [a for a in A]
print(" ".join(str(i) for a in A))

# print two dimension list
>>> A = [list(range(10), [10,11,12]]
for a in A:
  for b in a:
    print(b)
# one line
>>> [b for a in A for b in a]

# one condition
if a < 5:
  print(a)
# one line
a = 3
if a < 5: print(a)
# one line
a = 3
print(a if a < 5 else 1)

# more than one condition
if a < 5:
  print(0)
elif a < 10:
  print(1)
else:
  print(2)  
# one line
a = 3
print(0 if a < 5 else 1 if a < 10 else 2)

# for and if
A = list(range(10, 20))
print(A)
# legacy way
for a in A:
  if a == 0:
    print(a)
# oneline
[a for a in A if a == 0]    

# for and if and else
# legacy
for a in A:
  if a == 0:
    print(a)
  else:
    print("P")    
# oneline
[a if a == 0 else "P" for a in A]    
# oneline error
>>> [a if a > 0 for a in A]
  File "<stdin>", line 1
    [a if a > 0 for a in A]
                  ^
SyntaxError: invalid syntax
```

## Collections Compared to c++ containers

| c++                  | python                |
| :------------------- | :-------------------- |
| `if, else`           | `if, elif, else`      |
| `for, while`         | `for, while`          |
| `array`              | `tuple`               |
| `vector`             | `list`                |
| `deque`              | `deque`               |
| `forward_list`       | `list`                |
| `list`               | `deque`               |
| `stack`              | `list`                |
| `queue`              | `deque`               |
| `priority_queue`     | `PriorityQueue, heapq` |
| `set`                | ``                    |
| `multiset`           | ``                    |
| `map`                | ``                    |
| `multimap`           | ``                    |
| `unordered_set`      | `set`                 |
| `unordered_multiset` | `collections.Counter` |
| `unordered_map`      | `dict`                |
| `unordered_multimap` | `collections.defaultdict(list)`   |

## Collections

### tuple

tuple은 list와 비슷하지만 원소를 추가, 갱신, 삭제가 불가한 immutable type 이다.

```python
>>> t = ("A", 1, False)
>>> t
('A', 1, False)
>>> type(t)
<class 'tuple'>
>>> t[1]
1
>>> t[-1]
False
>>> t[1:2]
(1,)
>>> t[1:]
(1, False)
>>> a = (1, 2)
>>> b = (3, 4, 5)
>>> c = a + b
>>> c
(1, 2, 3, 4, 5)
>>> d = a * 3
>>> c[0] = 0
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'tuple' object does not support item assignment
>>> f, l = ("John", "Kim")
>>> f
'John'
>>> l
'Kim'
```

### list

```python
# append, clear, copy, count, extend, index, insert, 
# pop, remove, reverse, sort
>>> a = []
>>> a = [1, True, "Hello"]
>>> x = a[1]
>>> a[1] = False
>>> y = a[-1]
>>> y
'Hello'
>>> a = [1, 3, 5, 7, 9]
>>> x = a[1:3]
>>> x = a[:2]
>>> x = a[3:]
>>> x
[7, 9]
>>> a.append(11)
>>> a.pop()
11
>>> a
[1, 3, 5, 7, 9]
>>> a.append(11)
>>> a
[1, 3, 5, 7, 9, 11]
>>> a[1] = 33
>>> a
[1, 33, 5, 7, 9, 11]
>>> del a[2]
>>> a
[1, 33, 7, 9, 11]
>>> b = [21, 23]
>>> c = a + b
>>> c
[1, 33, 7, 9, 11, 21, 23]
>>> d = c * 3
>>> c
[1, 33, 7, 9, 11, 21, 23]
>>> d
[1, 33, 7, 9, 11, 21, 23, 1, 33, 7, 9, 11, 21, 23, 1, 33, 7, 9, 11, 21, 23]
>>> a.index(1)
0
>>> a.count(3)
0
>>> a.count(33)
1
>>> [n ** 2 for n in range(10) if n % 3 == 0]
[0, 9, 36, 81]
```

### deque

list와 유사하다. 왼쪽 오른쪽으로 원소를 추가 삭제할 수 있다.

```python
>>> from collections import deque
>>> dq = deque('ghi')
>>> for elem in d: print(elem.upper())
...
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'd' is not defined
>>> for elem in dq: print(elem.upper())
...
G
H
I
>>> dq.append('j')
>>> dq.appendleft('f')
>>> dq
deque(['f', 'g', 'h', 'i', 'j'])
>>> dq.pop()
'j'
>>> dq.popleft()
'f'
>>> list(dq)
['g', 'h', 'i']
>>> dq[0]
'g'
>>> dq[-1]
'i'
>>> list(reversed(dq))
['i', 'h', 'g']
>>> 'h' in dq
True
>>> dq.extend('jkl')
>>> dq
deque(['g', 'h', 'i', 'j', 'k', 'l'])
>>> dq.rotate()
>>> dq.rotate(1)
>>> dq
deque(['k', 'l', 'g', 'h', 'i', 'j'])
>>> dq.clear()
>>> dq.pop()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: pop from an empty deque
>>> dq.extendleft('abc')
>>> dq
deque(['c', 'b', 'a'])
```

### PriorityQueue

thread-safe 한 priority queue 이다. heapq 로 구현되어 있다. `put(), get()` 을 기억하자.

다음은 꼭대기 값이 작은 PriorityQueue 이다. 

```python
>>> from queue import PriorityQueue
>>> pq = PriorityQueue()
>>> pq.put(1)
>>> pq.put(3)
>>> pq.put(5)
>>> pq.get()
1
>>> pq.get()
3
>>> pq.get()
5
```

다음은 꼭대기 값이 큰 `PriorityQueue` 이다. element 로 tuple 을
사용한다. tuple 의 첫번째 값은 우선순위를 담당한다. 음수로 하자.

```python
>>> from queue import PriorityQueue
>>> pq = PriorityQueue()
>>> pq.put((-1, 1))
>>> pq.put((-3, 3))
>>> pq.put((-5, 5))
>>> pq.get()[1]
5
>>> pq.get()[1]
3
>>> pq.get()[1]
1
```

### heapq

> * [heapq @ python3](https://docs.python.org/3.10/library/heapq.html)

thread-safe 하지 않은 priority queue 이다.

`heapq.heapify(), heapq.heappush(), heapq.heappop()` 을 기억하자.

다음은 꼭대기 값이 작은 heapq 이다.

```python
>>> import heapq

>>> a = [9, 7, 5, 3, 1]
>>> import heapq

# make an heap with a legacy list in linear time
>>> heapq.heapify(a)
>>> a
[1, 3, 5, 9, 7]

# Push to heap
>>> heapq.heappush(a, 12)
>>> a
[1, 3, 5, 9, 7, 12]

# Pop from heap
>>> heapq.heappop(a)
1
>>> a
[3, 7, 5, 9, 12]

# Pop and return the smallest item from the heap, 
# and also push the new item. 
>>> heapq.heapreplace(a, 6)
3
>>> a
[5, 7, 6, 9, 12]

# Return a list with the n largest elements
>>> heapq.nlargest(2, a)
[12, 9]
>>> a
[5, 7, 6, 9, 12]

# Return a list with the n smallest elements
>>> heapq.nsmallest(2, a)
[5, 6]
>>> a
[5, 7, 6, 9, 12]

# Merge multiple sorted inputs into a single sorted output
>>> heapq.merge(a)
<generator object merge at 0x000001176B5FBBA0>
>>> list(heapq.merge(a))
[5, 7, 6, 9, 12]
```

다음은 꼭대기 값이 큰 heapq 이다. element 로 tuple 을
사용한다. tuple 의 첫번째 값은 우선순위를 담당한다. 음수로 하자.

```python
>>> import heapq
>>> a = []
>>> heapq.heapify(a)
>>> heapq.heappush(a, (-1, 1))
>>> heapq.heappush(a, (-3, 3))
>>> heapq.heappush(a, (-5, 5))
>>> heapq.heappop(a)[1]
5
>>> heapq.heappop(a)[1]
3
>>> heapq.heappop(a)[1]
1
```

### set

```python
>>> s = {1, 1, 1, 1, 2}
>>> s
{1, 2}
>>> s = set([1, 1, 1, 1, 2])
>>> s
{1, 2}
>>> s.add(7)
>>> s
{1, 2, 7}
>>> s.update({4, 2, 10})
>>> s
{1, 2, 4, 7, 10}
>>> s.remove(1)
>>> s
{2, 4, 7, 10}
>>> s.clear()
>>> s
set()
>>> a = {1, 2, 3}
>>> b = {3, 5, 6}
>>> a & b
{3}
>>> a | b
{1, 2, 3, 5, 6}
>>> a - b
{1, 2}
# all
>>> t = '3344'
>>> all(x in t for x in t)
True
>>> all(x in t for x in '2244')
False
>>> all(x in t for x in '3344')
True 
```

### dict

```python
>>> s = {"a": 1, "b": 2, "c": 3}
>>> type(s)
<class 'dict'>
>>> s["a"] = 100
>>> s
{'a': 100, 'b': 2, 'c': 3}
>>> dict([('a', 2), ('b', 4), ('c', 5)])
{'a': 2, 'b': 4, 'c': 5}
>>> dict(a=2, b=4, c=5)
{'a': 2, 'b': 4, 'c': 5}
>>> for key in s:
...     s[key]
...
100
2
3
>>> s.update({'a':-1, 'b':-1})
>>> s
{'a': -1, 'b': -1, 'c': 3}
```

### collections.Counter

dict 의 subclass 이다. 리스트 입력 데이터로 부터 값과 출현횟수를 각각
key 와 value 로 하는 dict 이다.

```python
>>> from collections import Counter
>>> a = Counter([1,2,3,1,2,1,2,1])
>>> a
Counter({1: 4, 2: 3, 3: 1})
>>> b = Counter([1,2,3,2,2,2,2])
>>> b
Counter({2: 5, 1: 1, 3: 1})
>>> a + b
Counter({2: 8, 1: 5, 3: 2})
>>> a - b
Counter({1: 3})
>>> a & b
Counter({2: 3, 1: 1, 3: 1})
>>> a | b
Counter({2: 5, 1: 4, 3: 1})
>>> a.pop(1)
4
>>> a
Counter({2: 3, 3: 1})
```

```py
from collections import Counter

colours = (
    ('Yasoob', 'Yellow'),
    ('Ali', 'Blue'),
    ('Arham', 'Green'),
    ('Ali', 'Black'),
    ('Yasoob', 'Red'),
    ('Ahmed', 'Silver'),
)

favs = Counter(name for name, colour in colours)
print(favs)
# Output: Counter({
#    'Yasoob': 2,
#    'Ali': 2,
#    'Arham': 1,
#    'Ahmed': 1
# })

with open('filename', 'rb') as f:
    line_count = Counter(f)
print(line_count)
```

### collections.defaultdict

dict 의 subclass 이다. 기본값을 지정할 수 있는 dict 이다.  기본값은
callable 하거나 None 이어야 한다.

```python
>>> from collections import defaultdict
>>> d = defaultdict("default-value", a=10)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: first argument must be callable or None
>>> d = defaultdict(lambda:"default-value", a=10)
>>> d
defaultdict(<function <lambda> at 0x0000025C0DA03E18>, {'a': 10})
>>>
>>> d[a]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'a' is not defined
>>> d['a']
10
>>> d['b']
'default-value'
>>> d
defaultdict(<function <lambda> at 0x0000025C0DA03E18>, {'a': 10, 'b': 'default-value'})
>>> d['c'] = 3
>>> d
defaultdict(<function <lambda> at 0x0000025C0DA03E18>, {'a': 10, 'b': 'default-value', 'c': 3})
```

다음과 같이 재귀적으로 사용할 수도 있다.

```py
some_dict = {}
some_dict['colours']['favourite'] = "yellow"
# Raises KeyError: 'colours'

from collections import defaultdict
tree = lambda: defaultdict(tree)
some_dict = tree()
some_dict['colours']['favourite'] = "yellow"
# Works fine

import json
print(json.dumps(some_dict))
# Output: {"colours": {"favourite": "yellow"}}
```

### collections.namedtuple

tuple 의 subclass 이다. tuple 은 index 로만 접근 가능하지만
namedtuple 은 index, name 으로 접근 가능하다.

```python
>>> import collections
>>> p = collections.namedtuple('Vector', ['x', 'y'])
>>> issubclass(p, tuple)
True
>>> Vector =  collections.namedtuple('Vector', ['x', 'y'])
>>> v = Vector(11, y=22)
>>> v
Vector(x=11, y=22)
>>> print(v[0], v[1])
11 22
>>> print(v.x, v.y)
11 22
```

### collections.ChainMap

dict 의 subclass 이다. 여러 개의 dict 를 모아서 하나의 dict 처럼 사용한다.
여러개의 dict 를 이용하여 하나의 dict 를 생성하거나 update 를 사용하는
것보다 훨씬 효율적이다.

```python
>>> from collections import ChainMap
>>> c1 = {'a': 1}
>>> c2 = {'b': 2}
>>> c = ChainMap(c1, c2)
>>> c
ChainMap({'a': 1}, {'b': 2})
>>> c['a']
1
>>> c['b']
2
>>> c['c'] = 3
>>> c
ChainMap({'a': 1, 'c': 3}, {'b': 2})
>>> c.clear()
>>> c
ChainMap({}, {'b': 2})
>>> c['b']
2
```

### collections.OrderedDict

dict 의 subclass 이다. 순서가 보장되는 dict 이다.

```python
>>> from collections import OrderedDict
>>> d = OrderedDict.fromkeys('abcde')
>>> d
OrderedDict([('a', None), ('b', None), ('c', None), ('d', None), ('e', None)])
>>> d.move_to_end('b')
>>> d.keys()
odict_keys(['a', 'c', 'd', 'e', 'b'])
>>> d.move_to_end('b', last=False)
>>> d
OrderedDict([('b', None), ('a', None), ('c', None), ('d', None), ('e', None)])
```

### enum (python 3.4+)

c# 의 enum 과 같다.

```py
from collections import namedtuple
from enum import Enum

class Species(Enum):
    cat = 1
    dog = 2
    horse = 3
    aardvark = 4
    butterfly = 5
    owl = 6
    platypus = 7
    dragon = 8
    unicorn = 9
    # The list goes on and on...

    # But we don't really care about age, so we can use an alias.
    kitten = 1
    puppy = 2

Animal = namedtuple('Animal', 'name age type')
perry = Animal(name="Perry", age=31, type=Species.cat)
drogon = Animal(name="Drogon", age=4, type=Species.dragon)
tom = Animal(name="Tom", age=75, type=Species.cat)
charlie = Animal(name="Charlie", age=2, type=Species.kitten)

# And now, some tests.
>>> charlie.type == tom.type
True
>>> charlie.type
<Species.cat: 1>
```

다음은 enum 의 member 를 접근하는 방법이다. 모두 `cat` 을 리턴한다.

```py
Species(1)
Species['cat']
Species.cat
```

## Sort

> sort
  
```py
> l = [5, -4, 3, -2, 1]
> l.sort()
> print(l)
[-4, -2, 1, 3, 5]
> l.sort(reverse=True)
> print(l)
[5, 3, 1, -2, -4]
> l.sort(key=lambda x: x*x)
> print(l)
[1, -2, 3, -4, 5]
> l.sort(key=lambda x: x*x, reverse=True)
> print(l)
[5, -4, 3, -2, 1]
```

> sorted

```py
>>> l = [5, 4, 3, 2, 1]
>>> sorted(l)
[1, 2, 3, 4, 5]
>>> sorted(l, key=lambda x: x*x)
[1, 2, 3, 4, 5]
>>> sorted(l, key=lambda x: x*x, reverse=True)
[5, 4, 3, 2, 1]
```

> sort vs sorted

sort() modify the object but sorted() create the new one.

> bisect

```py
# The list should be sorted for bisect
>>> l.sort()
>>> l = [1, 3, 5, 7, 9]
# insort_left: bisect_left, insert
>>> bisect.insort_left(l, 2)
>>> l
[1, 2, 3, 5, 7, 9]
# insort_right: bisect_right, insert
>>> bisect.insort_right(l, 4)
>>> l
[1, 2, 3, 4, 5, 7, 9]
>>> bisect.insort(l, 6)
>>> l
[1, 2, 3, 4, 5, 6, 7, 9]
```

## Search

> * `bisect_left`: Locate the left most insertion point for x in a to maintain sorted order.
> * `bisec_right`: Locate the right most insertion point for x in a to maintain sorted order.

```py
# binary search
from bisect import *

# bisect.bisect_left(a, x, lo=0, hi=len(a))
# a는 검색할 정렬된 배열입니다.
# x는 배열에 삽입하려는 값입니다.
# lo는 검색을 시작할 최소 인덱스이며 기본값은 0입니다.
# hi는 검색을 종료할 최대 인덱스이며 기본값은 배열의 길이입니다.

a = [2, 4, 6, 8]
>>> bisect_left(a, 4)
1
>>> bisect_left(a, 3)
1
>>> bisect_left(a, 2)
0
>>> bisect_left(a, 1)
0
>>> bisect_left(a, 8)
3
>>> bisect_left(a, 8, hi=2)
2
>>> bisect_right(a, 4)
2
>>> bisect_right(a, 5)
2
>>> bisect_right(a, 6)
3
>>> bisect_right(a, 6, hi=2)
2
# bisect is same with bisect_right
```

## Multidimensional Array

* [Python | Using 2D arrays/lists the right way](https://www.geeksforgeeks.org/python-using-2d-arrays-lists-the-right-way/)

----

```py
# 1D
N = 5
arr = [0]*N 
# arr = [0 for i in range(N)] 
print(arr)
# [0, 0, 0, 0, 0]

# 2D
h, w = (3, 3) 
A = [[0] * w] * h
print(A) 
# [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
A[0][0] = 1
print(A) 
# [[1, 0, 0], [1, 0, 0], [1, 0, 0]]

A = [[0] * w for _ in range(h)] 
print(A)
# [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
A[0][0] = 1
print(A)
# [[1, 0, 0], [0, 0, 0], [0, 0, 0]]

# Init 2D with 0
h, w = 10, 10
C = [[0] * (w + 1) for _ in range(y + 1)]
print(C)
```

## Built-in Functions

* [Built-in Functions](builtinfunctions.md)

## Round

* [Rounding numbers in Python 2 and Python 3](https://kingant.net/2019/01/rounding-numbers-in-python-2-and-python-3/)

----

python3 는 `ROUND_HALF_EVEN` 반올림 방식을 사용한다. `x.5` 일때 짝수쪽으로 변환한다. 즉, `round(0.5) == 0, round(0.6) == 1` 이다.

```py
print(round(0.5)) # 0
print(round(0.6)) # 1.0
print(round(1.5)) # 2.0 
print(round(1.6)) # 2.0 
print(round(2.5)) # 2.0 
print(round(2.6)) # 3.0
```

`round(0.5) == 1` 을 위해서는 `ROUND_HALF_UP` 반올림 방식을 사용해야 한다.

```py
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
>>> int(Decimal(0.5).quantize(Decimal(0), rounding=ROUND_HALF_UP))
1
>>> int(Decimal(0.6).quantize(Decimal(0), rounding=ROUND_HALF_UP))
1
>>> int(Decimal(1.5).quantize(Decimal(0), rounding=ROUND_HALF_UP))
2
>>> int(Decimal(2.5).quantize(Decimal(0), rounding=ROUND_HALF_UP))
3
```

## Classes

* class, instance variables

```py
class Cal(object):
    # pi is a class variable
    pi = 3.142

    def __init__(self, radius):
        # self.radius is an instance variable
        self.radius = radius

    def area(self):
        return self.pi * (self.radius ** 2)

a = Cal(32)
a.area()
# Output: 3217.408
a.pi
# Output: 3.142
a.pi = 43
a.pi
# Output: 43

b = Cal(44)
b.area()
# Output: 6082.912
b.pi
# Output: 3.142
b.pi = 50
b.pi
# Output: 50
```

* old style vs new style classes

new style classes 는 object 를 상속받는다. 따라서 `__slots__` 등과 같은 magic
mathod 를 사용할 수 있다. python 3 는 명시적으로 object 를 상속받지 않아도 new
style classes 로 선언된다.

```py
class OldClass():
    def __init__(self):
        print('I am an old class')

class NewClass(object):
    def __init__(self):
        print('I am a jazzy new class')

old = OldClass()
# Output: I am an old class

new = NewClass()
# Output: I am a jazzy new class
```

* `__init__`

```py
class GetTest(object):
    def __init__(self):
        print('Greetings!!')
    def another_method(self):
        print('I am another method which is not'
              ' automatically called')

a = GetTest()
# Output: Greetings!!

a.another_method()
# Output: I am another method which is not automatically
# called

class GetTest(object):
    def __init__(self, name):
        print('Greetings!! {0}'.format(name))
    def another_method(self):
        print('I am another method which is not'
              ' automatically called')

a = GetTest('yasoob')
# Output: Greetings!! yasoob

# Try creating an instance without the name arguments
b = GetTest()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: __init__() takes exactly 2 arguments (1 given)
```

* `__getitem__`

```py
class GetTest(object):
    def __init__(self):
        self.info = {
            'name':'Yasoob',
            'country':'Pakistan',
            'number':12345812
        }

    def __getitem__(self,i):
        return self.info[i]

foo = GetTest()

foo['name']
# Output: 'Yasoob'

foo['number']
# Output: 12345812
```

## lambdas

`lambda argument: manipulate(argument)` 과 같은 형식으로 사용하는 한줄 함수이다.

```py
add = lambda x, y: x + y

print(add(3, 5))
# Output: 8

a = [(1, 2), (4, 1), (9, 10), (13, -3)]
a.sort(key=lambda x: x[1])

print(a)
# Output: [(13, -3), (4, 1), (1, 2), (9, 10)]

data = zip(list1, list2)
data.sort()
list1, list2 = map(lambda t: list(t), zip(*data))
```

## slice

`list, tuple` 를 `[start:end:step]` 의 문법을 사용하여 새로운 객체를
만들어 내는 방법이다.

```py
>>> a = ['a', 'b', 'c', 'd', 'e']
>>> #     0    1    2    3    4 : positive index
... #    -5   -4   -3   -2   -1 : negative index
... 
>>> a[1:]
['b', 'c', 'd', 'e']
>>> a[-4:]
['b', 'c', 'd', 'e']
>>> a[:2]
['a', 'b']
>>> a[:-3]
['a', 'b']
>>> a[2:4]
['c', 'd']
>>> a[-3:-1]
['c', 'd']
>>> a[0:5:2]
['a', 'c', 'e']
>>> a[4::-2]
['e', 'c', 'a']
>>> a[4::-1]
['e', 'd', 'c', 'b', 'a']
```

## Underscore

> [참고](https://dbader.org/blog/meaning-of-underscores-in-python)

파이썬은 다음과 같이 5 가지 underscore 문법을 가지고 있다.

| pattern | example | desc |
| :-- | :-- | :--- |
| Single Leading Underscore | `_foo` | 개발자끼리 약속한 internal use |
| Single Trailing Underscore | `foo_` | 파이썬 키워드와 이름충돌을 피하기 위함 |
| Double Leading Underscore | `__foo` | 파이썬 인터프리터가 강제로 이름을 바꿔 버린다. `dir(a)` 이용하여 확인할 수 있다. |
| Double Leading and Trailing Underscore | `__foo__` | 파이썬 인터프리터가 내부적으로 사용하는 이름들. dunder members |
| Single Underscore | `_` | 신경쓰지 않아도되는 오브젝트들 |

```bash
>>> class A:
...   def __init__(self):
...     self.foo = 1
...     self._foo = 2
...     self.__foo = 3
...
>>> a = A()
>>> dir(a)
['_A__foo', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_foo', 'foo']
# 파이썬 인터프리터에 의해 '__foo' 가 `_A__foo` 로 mangling 되었다.
```

## Args, Kwargs

arguments, keyword arguments 를 의미한다. 다음과 같이 함수의 인자를 튜플,
딕셔너리로 받아낼 수 있다.

```py
def foo(*args, **kwargs):
  for a in args:
    print(f'{a}')
  print('-'*8)
  for k, v in kwargs.items():
    print(f'{k} : {v}')
foo(1, 2, 3, 4, i=0, j=1, k=2)
```

## Comprehension

list, set, dict, generator 를 간략히 표현할 수 있다.

* list comprehension

`variable = [out_exp for out_exp in input_list if out_exp == 2]` 와 같은 형식으로 사용한다. `[..for..in..if..]` 형식이라고 기억하자.

```py
l = [x*2 for x in range(10)]
l = [x*2 for x in range(10) if x%2 == 0]
l = [(a,b) for a in range(0,5) for b in range(5,10)]
l = [(a,b) for a in range(0,5) for b in range(5,10) if a%2==0 and b%2==0]

# origin
squared = []
for x in range(10):
    squared.append(x**2)
# list comprehension
squared = [x**2 for x in range(10)]
```

* set comprehension

`variable = {out_exp for out_exp in input_list if out_exp == 2}` 와 같은 형식으로 사용한다. `{..for..in..if..}` 형식이라고 기억하자.


```py
s = {j for i in range(2, 9) for j in range(i*2, 50, i)}

squared = {x**2 for x in [1, 1, 2]}
print(squared)
# Output: {1, 4}
```

* dict comprehension

`{v: k for k, v in some_dict.items()}` 와 같은 형식으로 사용한다.

```py
d = {key: val for key, val in zip(range(0,5), range(5, 10))}

mcase = {'a': 10, 'b': 34, 'A': 7, 'Z': 3}

mcase_frequency = {
    k.lower(): mcase.get(k.lower(), 0) + mcase.get(k.upper(), 0)
    for k in mcase.keys()
}

# mcase_frequency == {'a': 17, 'z': 3, 'b': 34}
```

* generator expression

`variable = (out_exp for out_exp in input_list if out_exp == 2)` 와 같은 형식으로 사용한다. `(..for..in..if..)` 형식이라고 기억하자. list comprehension 과 유사하지만 메모리를 덜 사용한다.

```py
g = (x**2 for x in range(10))
print(next(g))
print(next(g))

multiples_gen = (i for i in range(30) if i % 3 == 0)
print(multiples_gen)
# Output: <generator object <genexpr> at 0x7fdaa8e407d8>
for x in multiples_gen:
  print(x)
  # Outputs numbers
```

## Generator

python 의 반복과 관련된 3 가지 키워드 `iterable, iterator, iteration` 이 중요하다.

* `iterable`
  * `__iter__, __getitem__` 함수를 소유한 오브젝트이다. `iterator` 를 만들어줄 수 있다???
* `iterator`
  * `next (python2), __next__ (python3)` 함수를 소유한 오브젝트이다.
* `iteration`
  * 반복을 처리하는 절차를 의미한다.

generator 는 곧 iterator 이다. 다음과 같이 generator 를 리턴하는 함수를 정의하고 `for..in..` 에서 사용해 보자. `in` 에서 `StopIteration` Error 가 발생할 때까지 매번 `next(generator)` 를 수행한다.

```py
def generator_function():
    for i in range(10):
        yield i
for item in generator_function():
    print(item)
# Output: 0
# 1
# 2
# 3
# 4
# 5
# 6
# 7
# 8
# 9
```

다음은 fibonacci 수열을 generator 를 이용하여 구현한 것이다.

```py
# generator version
def fibon(n):
    a = b = 1
    for i in range(n):
        yield a
        a, b = b, a + b
for x in fibon(1000000):
    print(x)        
```

다음은 fibonacci 수열을 generator 없이 구현한 것이다. 매우 많은 메모리가 필요할 것이다.

```py
def fibon(n):
    a = b = 1
    result = []
    for i in range(n):
        result.append(a)
        a, b = b, a + b
    return result
for x in fibon(1000000):
    print(x)        
```

다음은 `next(generator)` 를 직접 호출한 예이다. 마지막에 `StopIteration` Error 가 발생했다.

```py
def generator_function():
    for i in range(3):
        yield i

gen = generator_function()
print(next(gen))
# Output: 0
print(next(gen))
# Output: 1
print(next(gen))
# Output: 2
print(next(gen))
# Output: Traceback (most recent call last):
#            File "<stdin>", line 1, in <module>
#         StopIteration
```

다음은 문자열을 generator 처럼 사용해 본 예제이다. 문자열은 iterator 가 아니기 때문에 에러가 발생한다.

```py
my_string = "Yasoob"
next(my_string)
# Output: Traceback (most recent call last):
#      File "<stdin>", line 1, in <module>
#    TypeError: str object is not an iterator
```

그러나 다음과 같이 `iter` 함수를 이용하면 iterator 를 만들어 낼 수 있다.

```py
int_var = 1779
iter(int_var)
# Output: Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# TypeError: 'int' object is not iterable
# This is because int is not iterable

my_string = "Yasoob"
my_iter = iter(my_string)
print(next(my_iter))
# Output: 'Y'
```

다음은 `()` 를 이용하여 간단히 generator 를 만들어내는 예이다. [PEP 289 -- Generator Expressions](https://www.python.org/dev/peps/pep-0289/)

```py
>>> (i for i in range(50))
<generator object <genexpr> at 0x000001B42CD5B888>
```

## Map, Filter, Reduce

`map` 은 `map(function_to_apply, list_of_inputs)` 와 같이 사용한다. `map` 은 `map object` 를 리턴하기 때문에 `list` 가 필요할 때가 있다.

```py
items = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, items))

def multiply(x):
    return (x*x)
def add(x):
    return (x+x)

funcs = [multiply, add]
for i in range(5):
    value = list(map(lambda x: x(i), funcs))
    print(value)

# Output:
# [0, 0]
# [1, 2]
# [4, 4]
# [9, 6]
# [16, 8]
```

`filter` 는 `filter(function_to_apply, list_of_inputs)` 와 같이 사용한다. `filter` 은 `filter object` 를 리턴하기 때문에 `list` 가 필요할 때가 있다.

```py
number_list = range(-5, 5)
less_than_zero = list(filter(lambda x: x < 0, number_list))
print(less_than_zero)

# Output: [-5, -4, -3, -2, -1]
```

`reduce` 는 `reduce(function_to_apply, list_of_inputs)` 와 같이 사용한다.

```py
from functools import reduce
product = reduce((lambda x, y: x * y), [1, 2, 3, 4])

# Output: 24
```

## Set

다음은 일반적인 예이다.

```py
some_list = ['a', 'b', 'c', 'b', 'd', 'm', 'n', 'n']
duplicates = set([x for x in some_list if some_list.count(x) > 1])
print(duplicates)
# Output: set(['b', 'n'])

a_set = {'red', 'blue', 'green'}
print(type(a_set))
# Output: <type 'set'>
```

다음은 교집합을 구하는 예이다.

```py
valid = set(['yellow', 'red', 'blue', 'green', 'black'])
input_set = set(['red', 'brown'])
print(input_set.intersection(valid))
# Output: set(['red'])
```

다음은 합집합을 구하는 예이다.

```py
valid = set(['yellow', 'red', 'blue', 'green', 'black'])
input_set = set(['red', 'brown'])
print(input_set.difference(valid))
# Output: set(['brown'])
```

## Ternary Operators

삼항 연산자는 `condition_if_true if condition else condition_if_false` 와 같이 사용한다.

```py
is_nice = True
state = "nice" if is_nice else "not nice"
```

튜플 삼항 연산자는 `(if_test_is_false, if_test_is_true)[test]` 와 같이 사용한다. 그러나 튜플의 내용이 evaluation 되기 때문에 자주 사용되지는 않는다.

```py
nice = True
personality = ("mean", "nice")[nice]
print("The cat is ", personality)
# Output: The cat is nice

condition = True
print(2 if condition else 1/0)
#Output is 2

print((1/0, 2)[condition])
#ZeroDivisionError is raised
```

다음은 shorthand ternary 이다.

```py
>>> True or "Some"
True
>>>
>>> False or "Some"
'Some'
>>> output = None
>>> msg = output or "No data returned"
>>> print(msg)
No data returned
```

## Multiple Return

`global` 을 사용하면 multiple return 이 가능하다.

```py
def profile():
    global name
    global age
    name = "Danny"
    age = 30

profile()
print(name)
# Output: Danny

print(age)
# Output: 30
```

tuple 을 이용하여 구현할 수도 있다.

```py
def profile():
    name = "Danny"
    age = 30
    return (name, age)

profile_data = profile()
print(profile_data[0])
# Output: Danny

print(profile_data[1])
# Output: 30
```

다음과 같이 unpacked tuple 로 구현할 수도 있다.

```py
def profile():
    name = "Danny"
    age = 30
    return name, age

profile_name, profile_age = profile()
print(profile_name)
# Output: Danny
print(profile_age)
# Output: 30
```

다음과 같이 named tuple 로 구현할 수도 있다.

```py
from collections import namedtuple
def profile():
    Person = namedtuple('Person', 'name age')
    return Person(name="Danny", age=31)

# Use as namedtuple
p = profile()
print(p, type(p))
# Person(name='Danny', age=31) <class '__main__.Person'>
print(p.name)
# Danny
print(p.age)
#31

# Use as plain tuple
p = profile()
print(p[0])
# Danny
print(p[1])
#31

# Unpack it immediatly
name, age = profile()
print(name)
# Danny
print(age)
#31
```

## Exception

```py
# single exception
try:
    file = open('test.txt', 'rb')
except IOError as e:
    print('An IOError occurred. {}'.format(e.args[-1]))

# multiple oneline exception
try:
    file = open('test.txt', 'rb')
except (IOError, EOFError) as e:
    print("An error occurred. {}".format(e.args[-1]))

# multiple multiline exception
try:
    file = open('test.txt', 'rb')
except EOFError as e:
    print("An EOF error occurred.")
    raise e
except IOError as e:
    print("An error occurred.")
    raise e    

# all exceptiontry:
    file = open('test.txt', 'rb')
except Exception as e:
    # Some logging if you want
    raise e

# finally
try:
    file = open('test.txt', 'rb')
except IOError as e:
    print('An IOError occurred. {}'.format(e.args[-1]))
finally:
    print("This would be printed whether or not an exception occurred!")

# Output: An IOError occurred. No such file or directory
# This would be printed whether or not an exception occurred!

# try, else
try:
    print('I am sure no exception is going to occur!')
except Exception:
    print('exception')
else:
    # any code that should only run if no exception occurs in the try,
    # but for which exceptions should NOT be caught
    print('This would only run if no exception occurs. And an error here '
          'would NOT be caught.')
finally:
    print('This would be printed in every case.')

# Output: I am sure no exception is going to occur!
# This would only run if no exception occurs. And an error here would NOT be caught
# This would be printed in every case.
```

## Dunder methods (Magic methods)

파이썬 인터프리터가 사용하는 내부적인 함수들이다.

```py
class A:
  def __init__(self, n):
    self.a = n
  def __repr__(self):
    return f'a:{self.a}'
  def __add__(self, other):
    return self.a + other

if __name__ == '__main__':
  a1 = A(1)
  a2 = A(2)
  print(a1)    
  print(a1 + a2)
```

## pdb

다음과 같이 commandline 으로 pdb 를 사용할 수 있다.

```bash
$ python -m pdb foo.py
```

또한 다음과 같이 code 중간에 break point 를 설정하여 pdb 를 사용할 수 있다.

```python
import pdb; pdb.set_trace()
```

다음은 주로 사용하는 pdb command 이다.

```
(Pdb) l         실행중인 코드 위치 확인
(Pdb) pdb.pm()  마지막 에러가 발생했던 곳을 디버깅
(Pdb) c         continue execution
(Pdb) w         shows the context of the current line it is executing.
(Pdb) a         print the argument list of the current function
(Pdb) s         Execute the current line and stop at the first possible occasion.
(Pdb) n         Continue execution until the next line in the current function is reached or it returns.
```

더 읽을 거리

* [visual studio python](https://www.visualstudio.com/ko/vs/python/) 도 멋지다.
* [python in visual studio code](https://code.visualstudio.com/docs/languages/python) 도 멋지다.

## Async, Await

`sync, await` 은 unity c# 의 `IEnumerator, yield` 와 비슷하다.

```python
import aiohttp
import asyncio
import async_timeout

async def fetch(session, url):
    with async_timeout.timeout(10):
        async with session.get(url) as response:
            return await response.text()

async def main():
    async with aiohttp.ClientSession() as session:
        a = await fetch(session, 'http://python.org')
        b = await fetch(session, 'http://daum.net')
        c = await fetch(session, 'http://google.co.kr')
        print([f for f in [a, b, c]])

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
```

## Performance Check

* timeit
  * 두가지의 코드를 각각 c1, c2라고 하자. c1은 한번 실행되는
    코드이다. c2는 반복실행 될 코드이다. c2의 평균 실행시간을
    알려준다.

```python
>>> python -m timeit -s 'import math' 'math.log10(99999999)'
```

* profile
  * 호출 횟수, 고유 실행시간, 하위 콜스택 포함 실행시간 등을 알 수 있다.

```python
>>> python -m profile a.py
```

* cProcfile
  * profile 보다 최적화된 버전
  
```python
>>> python -m cProfile a.py
```

* yappi
  * 호출 횟수, 고유 실행시간, 하위 콜스택 포함 실행시간 등을 쓰레드별로 알 수 있다.

```python
>>> python -m yappi a.py
```

* [profiling](https://github.com/what-studio/profiling)
  * 왓스튜디오에서 제작한 text ui 기반의 profiler이다. 콜스택, 샘플링,
    실시간, 대화형 TUI, 원격 등의 기능을 지원한다.
  
  
## `__slots__`

임의의 class 는 attributes 를 dict 를 이용하여 저장한다. dict 는 메모리를 많이
소모한다.  `__slots__` 를 이용하면 `dict` 를 사용하지 않고 attributes 를 저장할
수 있기에 메모리를 적게 사용한다. 그러나 동적으로 attributes 를 정의 할 수 없다.
특정 class 의 attributes 는 생성과 동시에 정해지고 runtime 에 추가될 일이 없다면
`__slots__` 를 이용하자. 40% 혹은 50% 까지 성능향상 할 수 있다고 함.
[참고](http://book.pythontips.com/en/latest/__slots__magic.html)

```py
# members with dict
class MyClass(object):
    def __init__(self, name, identifier):
        self.name = name
        self.identifier = identifier
        self.set_up()
    # ...

# members with __slots__
class MyClass(object):
    __slots__ = ['name', 'identifier']
    def __init__(self, name, identifier):
        self.name = name
        self.identifier = identifier
        self.set_up()
    # ...
```

## Enumerate

무언가에 대해 반복할 때 auto increment number 와 함께 순회할 수 있다.

```py
for counter, value in enumerate(some_list):
    print(counter, value)

my_list = ['apple', 'banana', 'grapes', 'pear']
for c, value in enumerate(my_list, 1):
    print(c, value)

# Output:
# 1 apple
# 2 banana
# 3 grapes
# 4 pear
```

auto increment number 의 초기값을 제어할 수 있다.

```py
my_list = ['apple', 'banana', 'grapes', 'pear']
counter_list = list(enumerate(my_list, 1))
print(counter_list)
# Output: [(1, 'apple'), (2, 'banana'), (3, 'grapes'), (4, 'pear')]
```

## Metaclass

* metaclass 는 runtime 에 class 를 생성할 수 있다. 한편 class 는 runtime 에
  instance 를 생성할 수 있다. 또한 클래스가 정의될 때를 decorating 할 수 있다. 
* `__metaclass__` 는 python 3 에서 더이상 사용하지 않는다. 클래스를 정의할때
  인자로 `metaclass` 를 사용하자.
* class 는 class instance 의 type 이고 metaclass 는 class 의 type 이다. 
  
```python
>>> class Foo(object): pass
...
>>> f = Foo()
>>> type(f)
<class '__main__.Foo'>
>>> type(Foo)
<class 'type'>
```

* [type](https://docs.python.org/3/library/functions.html#type)
  은 name, bases, dict를 인자로 받아 runtime에 class를 정의할 수 있다.
  
```python
>>> X = type('X', (object,), dict(a=1))
>>> class X(object): a = 1
...
>>> x = X()
>>> x.a
1
>>> X = type('X', (object,), dict(a=1))
>>> x = X()
>>> x.a
1
```

```python
>>> a = 1
>>> a.__class__
<class 'int'>
>>> b = 'foo'
>>> b.__class__
<class 'str'>
>>> def foo(): pass
...
>>> foo.__class__
<class 'function'>
>>> class Foo(object): pass
...
>>> f = Foo()
>>> f.__class__
<class '__main__.Foo'>
>>> Bar = type('Bar', (), {})
>>> Bar.__class__
<class 'type'>
```

* metaclass 를 지정하면 class 가 정의될때 decorating 할 수 있다.

```python
>>> def MyMetaClass(name, bases, attrs):
...     print('MyMetaClass is called.')
...     return type(name, bases, attrs)
...
>>> class Foo(metaclass=MyMetaClass): pass
...
MyMetaClass is called.
>>> f = Foo()
```

## Weakref

다음과 같이 class `Foo`를 정의하여 reference count가 0이 될 때
`__del__`이 호출 되는 것을 확인하자.

```python
>>> class Foo:
...   def __init__(self):
...     self.v = 1
...     self.friend = None
...   def __del__(self):
...     self.friend = None
...     print('{0:x} is destroyed'.format(id(self)))
...
>>> a, b = Foo(), Foo()
>>> a.friend = b
>>> b = None
>>> a = None
7f18548fe710 is destroyed
7f18548fe6a0 is destroyed
```

다음과 같은 경우 a, b, c는 순환 참조 되고 있어서
`__del__` 이 호출되지 않는다.

```python
>>> a, b, c = Foo(), Foo(), Foo()
>>> a.friend = b
>>> b.friend = a
>>> b = None
>>> a = None
>>> c.friend = c
>>> c = None
```

weakref는 약하게 참조한다는 의미로 reference count를 증가하지 않는
참조다. 순환 참조를 막기 위해 과감히 사용하자.

```python
>>> a = Foo(); b = Foo();
>>> a.friend = weakref.ref(b)
>>> b.friend = weakref.ref(a)
>>> b = None
7f1850f0c2b0 is destroyed
>>> a.friend
<weakref at 0x7f1850f06a48; dead>
>>> a = None
7f1850f0c358 is destroyed
>>> c = Foo()
>>> c.friend
>>> c.friend()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: 'NoneType' object is not callable
>>> c.friend = weakref.ref(c)
>>> c.friend
<weakref at 0x7f185499aa98; to 'Foo' at 0x7f1850f0c358>
>>> c = None
7f1850f0c358 is destroyed
```

## Memory Leak

gc가 순환 참조들을 수거하긴 하지만 많은 비용이 필요한 작업이다. reference
count를 잘 관리해서 gc의 도움 없이 메모리 누수를 방지하도록 하자.

* with, finally 를 이용하여 리소스를 꼭 정리해주자.

```python
def love_with_foo(self):
  self.target = load_foo()
  try:
    self.love()
  finally:
    del self.target
```

```python
@contextmanager
def foo_target(self):
  self.target = load_foo()
  try:
    yield
  finally:
    del self.target

def love_with_foo(self):
  with self.foo_target()
    self.love()
```

* weakref를 사용하여 순환참조를 막는다.

* objgraph를 이용하여 디버깅하자.

## gc

cpython은 gc와 reference counting이 둘 다 사용되지만
PyPy는 gc만 사용된다. gc.collect를 수행하면 순환참조를
수거할 수 있지만 많은 비용이 필요하다.

```
>>> import gc
>>> len(gc.get_objects())
4671
>>> gc.collect()
0
>>> len(gc.get_objects())
4598
```

## dependencies

* setup.py 로 대부분 가능하다.

```
from setuptools import setup
setup(
    name='foobar',
    install_requires=[
        'profiling',
        'trueskill',
        'tossi==0.0.1',
    ],
)
```

* 조금 더 복잡한 의존성 해결을 위해 requirements.txt도 고민해 보자.
  requirements.txt는 `pip install` 명령어에 넘길 인자를 기록한
  파일이다.

```
profiling
trueskill
tossi==0.0.1
-r lib/baz/requirements.txt
-e git+https://github.com/sublee/hangulize
```

```
> pip install -r requirements.txt
```

# Advanced 

## walrus operator

> [The Walrus Operator: Python 3.8 Assignment Expressions | realpython](https://realpython.com/python-walrus-operator/)

```py
# as-is
a = [1, 2, 3, 4]
n = len(a)
if n > 5:
    print(f"List is too long ({n} elements, expected <= 5)")

# to-be
a = [1, 2, 3, 4]
if (n := len(a)) > 5:
    print(f"List is too long ({n} elements, expected <= 5)")
```

## Pythonic Way

* [Code Style¶](https://docs.python-guide.org/writing/style/)

## Open Function

다음의 코드는 3 가지 문제점이 있다.

```py
f = open('photo.jpg', 'r+')
jpgdata = f.read()
f.close()
```

첫째, `open` 이후 exception 이 발생해도 `close` 가 호출되지 않는다. 다음과 같이
`with` 를 사용하자.

```py
with open('photo.jpg', 'r+') as f:
    jpgdata = f.read()
```

둘째, 파일이 바이너리인지 텍스트인지를 읽기모드 옵션으로 제공해야 한다.

셋째, 파일의 인코딩형식을 제공해야 한다.

```py
import io

with open('photo.jpg', 'rb') as inf:
    jpgdata = inf.read()

if jpgdata.startswith(b'\xff\xd8'):
    text = u'This is a JPEG file (%d bytes long)\n'
else:
    text = u'This is a random file (%d bytes long)\n'

with io.open('summary.txt', 'w', encoding='utf-8') as outf:
    outf.write(text % len(jpgdata))
```

## Decorator

어떤 함수의 앞, 뒤에 수행될 내용을 장식해 주는 기능을 한다. 

다음은 `a_function_requiring_decoration()` 의 앞, 뒤에 수행될 내용을 장식하는
단순한 방법이다.

```py
def a_new_decorator(a_func):

    def wrapTheFunction():
        print("I am doing some boring work before executing a_func()")

        a_func()

        print("I am doing some boring work after executing a_func()")

    return wrapTheFunction

def a_function_requiring_decoration():
    print("I am the function which needs some decoration to remove my foul smell")

a_function_requiring_decoration()
#outputs: "I am the function which needs some decoration to remove my foul smell"

a_function_requiring_decoration = a_new_decorator(a_function_requiring_decoration)
#now a_function_requiring_decoration is wrapped by wrapTheFunction()

a_function_requiring_decoration()
#outputs:I am doing some boring work before executing a_func()
#        I am the function which needs some decoration to remove my foul smell
#        I am doing some boring work after executing a_func()
```

다음은 위의 예를 `@` 를 이용하여 간단히 구현한 것이다.

```py
@a_new_decorator
def a_function_requiring_decoration():
    """Hey you! Decorate me!"""
    print("I am the function which needs some decoration to "
          "remove my foul smell")

a_function_requiring_decoration()
#outputs: I am doing some boring work before executing a_func()
#         I am the function which needs some decoration to remove my foul smell
#         I am doing some boring work after executing a_func()

#the @a_new_decorator is just a short way of saying:
a_function_requiring_decoration = a_new_decorator(a_function_requiring_decoration)
```

그러나 다음과 같이 `decorator` 의 이름이 의도와 다르게 출력된다.

```py
print(a_function_requiring_decoration.__name__)
# Output: wrapTheFunction
```

다음과 같이 수정하여 `decorator` 의 이름이 올바르게 출력되도록 하자.

```py
from functools import wraps

def a_new_decorator(a_func):
    @wraps(a_func)
    def wrapTheFunction():
        print("I am doing some boring work before executing a_func()")
        a_func()
        print("I am doing some boring work after executing a_func()")
    return wrapTheFunction

@a_new_decorator
def a_function_requiring_decoration():
    """Hey yo! Decorate me!"""
    print("I am the function which needs some decoration to "
          "remove my foul smell")

print(a_function_requiring_decoration.__name__)
# Output: a_function_requiring_decoration
```

정리하면 `decorator` 는 다음과 같이 사용할 수 있다.

```py
from functools import wraps
def decorator_name(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not can_run:
            return "Function will not run"
        return f(*args, **kwargs)
    return decorated

@decorator_name
def func():
    return("Function is running")

can_run = True
print(func())
# Output: Function is running

can_run = False
print(func())
# Output: Function will not run
```

다음은 `method decorator` 의 예이다.

```py
import datetime

def datetime_decorator(f):
  def decorated():
    print(datetime.datetime.now())
    f()
    print(datetime.datetime.now())
  return decorated

@datetime_decorator
def foo():
  print('foo')

@datetime_decorator
def bar():
  print('bar')

@datetime_decorator
def baz():
  print('baz')
```

다음은 `class decorator` 의 예이다.

```py
import datetime

class DatetimeDecorator:
  def __init__(self, f):
    self.f = f
  def __call__(self, *args, **kwargs):
    print(datetime.datetime.now())
    self.f(*args, **kwargs)
    print(datetime.datetime.now())

@DatetimeDecorator
def foo():
  print('foo')

@DatetimeDecorator
def bar():
  print('bar')

@DatetimeDecorator
def baz():
  print('baz')    
```

## Virtual Environment

```bash
# 🐍 Python venv 실전 명령어 모음
# 프로젝트별 가상 환경을 만들고 관리하기 위한 기본 커맨드 정리

# 1. 가상 환경 생성 (현재 디렉터리에 `.venv` 폴더 생성)
python3 -m venv .venv

# 2. 가상 환경 활성화
# macOS / Linux
source .venv/bin/activate

# Windows (cmd)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3. 가상 환경 비활성화
deactivate

# 4. 패키지 설치 (예: requests)
pip install requests

# 5. 현재 가상환경의 패키지 목록 저장
pip freeze > requirements.txt

# 6. 다른 사람이 동일한 환경을 만들 수 있게 requirements.txt로 설치
pip install -r requirements.txt

# 7. 설치된 패키지 목록 확인
pip list

# 8. 특정 패키지 정보 보기
pip show flask

# 9. 패키지 제거
pip uninstall flask

# 10. 현재 Python 버전 확인
python --version

# 11. 현재 사용 중인 Python 실행 파일 경로 확인
which python        # macOS / Linux
where python        # Windows

# 12. 가상 환경 삭제
# (비활성화 후 폴더 자체를 삭제)
deactivate
rm -rf .venv

# 13. .gitignore 예시 (.venv 폴더는 Git에 커밋하지 않음)
# .gitignore 파일에 다음 줄을 추가:
# .venv/

# 14. Makefile 예시 (선택: 자동화용)
# 아래 내용을 Makefile로 저장하고 `make setup`으로 실행
# setup:
# 	python3 -m venv .venv
# 	source .venv/bin/activate && pip install -r requirements.txt
```

## one-liners

```bash
# Python 2
python -m SimpleHTTPServer

# Python 3
python -m http.server

# pretty print
cat file.json | python -m json.tool

# profile
python -m cProfile my_script.py

# json dump
python -c "import csv,json;print json.dumps(list(csv.reader(open('csv_file.csv'))))"
```

```py
# list flattening
a_list = [[1, 2], [3, 4], [5, 6]]
print(list(itertools.chain.from_iterable(a_list)))
# Output: [1, 2, 3, 4, 5, 6]

# or
print(list(itertools.chain(*a_list)))
# Output: [1, 2, 3, 4, 5, 6]

# one line construct
class A(object):
    def __init__(self, a, b, c, d, e, f):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})
```

## C Extension

* Ctypes

다음과 같이 `add.c` 를 작성한다.

```cpp
//sample C file to add 2 numbers - int and floats

#include <stdio.h>

int add_int(int, int);
float add_float(float, float);

int add_int(int num1, int num2){
    return num1 + num2;
}

float add_float(float num1, float num2){
    return num1 + num2;
}
```

다음과 같이 `add.c` 파일을 빌드하여 `adder.so` 를 만들자.

```bash
#For Linux
$  gcc -shared -Wl,-soname,adder -o adder.so -fPIC add.c

#For Mac
$ gcc -shared -Wl,-install_name,adder.so -o adder.so -fPIC add.c
```

다음과 같이 python 에서 `adder.so` 를 이용한다.

```py
from ctypes import *

#load the shared object file
adder = CDLL('./adder.so')

#Find sum of integers
res_int = adder.add_int(4,5)
print "Sum of 4 and 5 = " + str(res_int)

#Find sum of floats
a = c_float(5.5)
b = c_float(4.1)

add_float = adder.add_float
add_float.restype = c_float
print "Sum of 5.5 and 4.1 = ", str(add_float(a, b))
```

출력은 다음과 같다.

```
Sum of 4 and 5 = 9
Sum of 5.5 and 4.1 =  9.60000038147
```

* SWIG (Simplified Wrapper and Interface Generator)

다음과 같이 `example.c` 를 작성한다.

```cpp
#include <time.h>
double My_variable = 3.0;

int fact(int n) {
    if (n <= 1) return 1;
    else return n*fact(n-1);
}

int my_mod(int x, int y) {
    return (x%y);
}

char *get_time()
{
    time_t ltime;
    time(&ltime);
    return ctime(&ltime);
}
```

다음과 같이 `example.i` 를 작성한다.

```c
/* example.i */
 %module example
 %{
 /* Put header files here or function declarations like below */
 extern double My_variable;
 extern int fact(int n);
 extern int my_mod(int x, int y);
 extern char *get_time();
 %}

 extern double My_variable;
 extern int fact(int n);
 extern int my_mod(int x, int y);
 extern char *get_time();
```

다음과 같이 `example.c` 를 빌드하여 `example.so` 를 만들자.

```bash
unix % swig -python example.i
unix % gcc -c example.c example_wrap.c \
        -I/usr/local/include/python2.1
unix % ld -shared example.o example_wrap.o -o _example.so
```

다음과 같이 python 에서 `example.so` 를 이용하자. c library 를 다른 모듈과 같이 `import` 하여 사용할 수 있다.

```py
>>> import example
>>> example.fact(5)
120
>>> example.my_mod(7,3)
1
>>> example.get_time()
'Sun Feb 11 23:01:07 1996'
>>>
```

* python/c api

가장 많이 사용하는 방법이다. python/c api 를 사용하여 `adder.c` 를 작성한다.

```c
//Python.h has all the required function definitions to manipulate the Python objects
#include <Python.h>

 //This is the function that is called from your python code
static PyObject* addList_add(PyObject* self, PyObject* args){

  PyObject * listObj;

  //The input arguments come as a tuple, we parse the args to get the various variables
  //In this case it's only one list variable, which will now be referenced by listObj
  if (! PyArg_ParseTuple( args, "O", &listObj))
    return NULL;

  //length of the list
  long length = PyList_Size(listObj);

  //iterate over all the elements
  long i, sum =0;
  for(i = 0; i < length; i++){
    //get an element out of the list - the element is also a python objects
    PyObject* temp = PyList_GetItem(listObj, i);
    //we know that object represents an integer - so convert it into C long
    long elem = PyInt_AsLong(temp);
    sum += elem;
  }

  //value returned back to python code - another python object
  //build value here converts the C long to a python integer
  return Py_BuildValue("i", sum);
}

//This is the docstring that corresponds to our 'add' function.
static char addList_docs[] =
    "add( ): add all elements of the list\n";

/* This table contains the relavent info mapping -
  <function-name in python module>, <actual-function>,
  <type-of-args the function expects>, <docstring associated with the function>
*/
static PyMethodDef addList_funcs[] = {
    {"add", (PyCFunction)addList_add, METH_VARARGS, addList_docs},
    {NULL, NULL, 0, NULL}
};

/*
addList is the module name, and this is the initialization block of the module.
<desired module name>, <the-info-table>, <module's-docstring>
*/
PyMODINIT_FUNC initaddList(void){
    Py_InitModule3("addList", addList_funcs,
                   "Add all ze lists");
}
```

다음과 같이 module 설치 스크립트 `setpu.py` 를 작성한다.

```py
#build the modules

from distutils.core import setup, Extension

setup(name='addList', version='1.0',  \
      ext_modules=[Extension('addList', ['adder.c'])])
```

`python setup.py install` 와 같이 c module 을 설치한다.

다음과 같이 python 에서 사용한다.

```py
#module that talks to the C code
import addList

l = [1,2,3,4,5]
print "Sum of List - " + str(l) + " = " +  str(addList.add(l))
```

## python 2+3

python 2 와 python 3 에서 실행될 수 있는 코드를 작성하자. 

```py
# python 2 에서 `with` 를 사용할 수 있다.
from __future__ import with_statement

# pythno 2 에서 python 3 print() 를 사용할 수 있다.
print
# Output:

from __future__ import print_function
print(print)
# Output: <built-in function print>

#
import foo as foo

# try exception 을 활용하여 version 별로 다른 module 을 import 할 수 있다.
try:
    import urllib.request as urllib_request  # for Python 3
except ImportError:
    import urllib2 as urllib_request  # for Python 2

# python 2 에서 deprecated 12 functions 들을 사용하지 않도록 설정할 수 있다. 사용하면 NameError 가 발생한다.
from future.builtins.disabled import *
```

## Coroutine

generator 는 data producer 이고 coroutine 은 data consumer 인 것을 제외하면
generator 와 coroutine 은 유사하다. coroutine 은 `send()` 를 이용하여 데이터를
외부에서 제공해야 한다. 

```py
# generator
def fib():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a+b
for i in fib():
    print(i)

# coroutine
def grep(pattern):
    print("Searching for", pattern)
    while True:
        line = (yield)
        if pattern in line:
            print(line)
search = grep('coroutine')
next(search)
# Output: Searching for coroutine
search.send("I love you")
search.send("Don't you love me?")
search.send("I love coroutines instead!")
# Output: I love coroutines instead!
search = grep('coroutine')
# ...
search.close()
```

## Function Caches

함수의 `인자:리턴` 을 캐시에 저장한다.

```py
from functools import lru_cache

@lru_cache(maxsize=32)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

>>> print([fib(n) for n in range(10)])
# Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
>>> fib.cache_clear()
```

## Context Managers

context managers 는 resource 할당과 해제를 자동으로 해준다. 주로 `with` 와 함께
사용한다. 아래의 두블록은 같다.

```py
# 
with open('some_file', 'w') as opened_file:
    opened_file.write('Hola!')
# 
file = open('some_file', 'w')
try:
    file.write('Hola!')
finally:
    file.close()
```

class 에 `__enter__, __exit__` 를 정의하여 context amangers 로 만들어 보자.
그리고 `with` 에서 사용해 보자. exception 은 `__exit__` 에서 `traceback` 을
참고하여 handle 한다. 이상없으면 `True` 를 리턴한다.

```py
class File(object):
    def __init__(self, file_name, method):
        self.file_obj = open(file_name, method)
    def __enter__(self):
        return self.file_obj
    def __exit__(self, type, value, traceback):
        print("Exception has been handled")
        self.file_obj.close()
        return True
with File('demo.txt', 'w') as opened_file:
    opened_file.write('Hola!')
```

generator 를 이용하여 context manager 를 만들자.

```py
from contextlib import contextmanager

@contextmanager
def open_file(name):
    f = open(name, 'w')
    yield f
    f.close()
with open_file('foo') as f:
    f.write('hola')
```

## typing — Support for type hints @ python3

* [typing — Support for type hints @ python3](https://docs.python.org/3/library/typing.html)

----

```py
# List type hints
from typing import List
Vector = List[float]

def scale(scalar: float, vector: Vector) -> Vector:
    return [scalar * num for num in vector]

# typechecks; a list of floats qualifies as a Vector.
new_vector = scale(2.0, [1.0, -4.2, 5.4])

# class type hints
class Foo:
    x: int
    y: int
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __str__(self):
        return '({}, {})'.format(self.x, self.y)

if __name__ == "__main__":
    foo = Foo(1, 2)
    print(foo)
```

## itertools — Functions creating iterators for efficient looping

* [itertools](itertools.md)

## functools — Higher-order functions and operations on callable objects

* [functools](functools.md)

## Tracing Memory

```py
>>> import tracemalloc
>>> tracemalloc.get_traced_memory()
(63621, 252994)
>>> a = []
>>> tracemalloc.get_traced_memory()
(89997, 252994)
>>> tracemalloc.get_traced_memory.__doc__
'Get the current size and peak size of memory blocks traced by tracemalloc.\n\nReturns a tuple: (current: int, peak: int).'
```

# Libraries

## regex

* [11.2 텍스트 처리 @ 파이썬 프로그래밍 입문서 (가제)](https://python.bakyeono.net/chapter-11-2.html#1122-%ED%85%8D%EC%8A%A4%ED%8A%B8-%ED%8C%A8%ED%84%B4)

----

```python
import re
p = re.compile(r'(?P<word>\b\w*\b)')
m = p.search('(((( Lots of punctuation )))')
print(m.group('word'))
print(m.group(0))
print(m.group(1))
```

## click

[click](http://click.pocoo.org/5/)은 콘솔 라이브러리이다.

```python
import click

@click.command()
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name', help='The person to greet.')

def main(count, name):
    for x in range(count):
        click.echo('Hello %s!' % name)

if __name__ == "__main__":
    main()
```

## unittest

* [unittest와 함께하는 파이썬 테스트](https://www.holaxprogramming.com/2017/06/17/python-with-test/)

----

Create a file class TestArray which extends unitest.TestCase in `tests/test_a.py`.

```py
import unittest
from algorithms import array

class TestArray(unittest.TestCase):
    """
    Test that the result sum of all numbers
    """
    def test_sum(self):
        instance = array.Array()
        result = instance.sum(6, '1 2 3 4 10 11')
        self.assertEqual(result, 31)
```

Let's test it.

```bash
# test all
$ cd tests
$ python -m unitest 

# test specific module
$ python -m unitest tests/test_a.py
```

## pytest

[pytest](https://docs.pytest.org/en/latest/)는 test framework이다. unittest보다 편하다.

```python
def inc(x):
    return x + 1

def test_answer():
    assert inc(3) == 5
```

```bash
> pytest a.py
> pytest a.py --cov=a
```

## urwid

[urwid](http://urwid.org/)는 text gui 라이브러이다.

## fabric

[fabric](https://github.com/mathiasertl/fabric/)은 응용프로그램 배포툴이다.

## flake8

[flake8](http://flake8.pycqa.org/en/latest/)는 파이썬 코딩 스타일 가이드 체크 프로그램이다.

## objgraph

[objgraph](https://mg.pov.lt/objgraph/)는 메모리누수를 발견할 수 있는 라이브러리이다.

* forward references

```python
>>> x = []
>>> y = [x, [x], dict(x=x)]
>>> import objgraph
>>> objgraph.show_refs([y], filename='sample-graph.png')
Graph written to ....dot (... nodes)
Image generated as sample-graph.png
```

* backward references

```python
>>> objgraph.show_backrefs([x], filename='sample-backref-graph.png')
... 
Graph written to ....dot (8 nodes)
Image generated as sample-backref-graph.png
```

* show stats

```python
>>> objgraph.show_most_common_types() 
tuple                      5224
function                   1329
wrapper_descriptor         967
dict                       790
builtin_function_or_method 658
method_descriptor          340
weakref                    322
list                       168
member_descriptor          167
type                       163
>>> xs = objgraph.by_type('Foo')
>>> len(xs)
99
```

## MySQL-python 1.2.5 on python2.7 build error, library not found for -lssl

Use `--global-option`.

```bash
$ pip install MySQL-python==1.2.5
...
    gcc -bundle -undefined dynamic_lookup -arch x86_64 -g build/temp.macosx-10.9-x86_64-2.7/_mysql.o -L/usr/local/opt/mysql-client@5.7/lib -lmysqlclient -lssl -lcrypto -o build/lib.macosx-10.9-x86_64-2.7/_mysql.so
    ld: library not found for -lssl
...

$ pip install --global-option=build_ext --global-option="-I/usr/local/opt/openssl@1.1/include" --global-option="-L/usr/local/opt/openssl@1.1/lib" MySQL-python==1.2.5
```

# Style Guide

* [Python PEP 8 Style Guide](python_pep8_style_guide.md)
* [Python Google Style Guide](python_google_style_guide.md)

# Refactoring

[Refactoring Python](refactoring_python.md)

# Effective Python

[Effective Python](effective_python.md)

# GOF Design Pattern

[Python GOF Design Pattern](python_gof_designpattern.md)

# Architecture

* [Architecture Patterns with Python | amazon](https://www.amazon.com/Architecture-Patterns-Python-Domain-Driven-Microservices/dp/1492052205)
* [tructuring Your Project | Hitchhiker's Guide](https://docs.python-guide.org/writing/structure/)
