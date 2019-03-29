
- [Abstract](#abstract)
- [Essentials](#essentials)
- [Materials](#materials)
- [Basic Usages](#basic-usages)
  - [Collections Compared to c++ containers](#collections-compared-to-c-containers)
  - [Collections](#collections)
  - [underscore](#underscore)
  - [generator](#generator)
  - [pdb](#pdb)
  - [async, await](#async-await)
  - [performance check](#performance-check)
  - [`__slots__`](#slots)
  - [`__metaclass__`](#metaclass)
  - [weakref](#weakref)
  - [memory leak](#memory-leak)
  - [gc](#gc)
  - [dependencies](#dependencies)
- [Advanced Usages](#advanced-usages)
  - [Builtin functions](#builtin-functions)
- [Library](#library)
  - [regex](#regex)
  - [numpy](#numpy)
  - [pandas](#pandas)
  - [click](#click)
  - [pytest](#pytest)
  - [urwid](#urwid)
  - [fabric](#fabric)
  - [flake8](#flake8)
  - [objgraph](#objgraph)
  
----

# Abstract

python3에 대해 정리한다.

# Essentials

* [python 3 doc](https://docs.python.org/3/contents.html)
  * 목차분석이 필요함
* [James Powell: So you want to be a Python expert? | PyData Seattle 2017 @ youtube](https://www.youtube.com/watch?v=cKPlPJyQrt4&list=WL&index=5&t=0s)
  * dunder method, metaclass, decorator, context manager, generator
  
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

# Basic Usages

## Collections Compared to c++ containers

| c++                  | python             | 
|:---------------------|:-------------------|
| `if, else`           | `if, elif, else`   |
| `for, while`         | `for, while`       |
| `array`              | `tuple`            |
| `vector`             | `list`             |
| `deque`              | `deque`            |
| `forward_list`       | `list`             |
| `list`               | `deque`            |
| `stack`              | `list`             |
| `queue`              | `deque`            |
| `priority_queue`     | `heapq`            |
| `set`                | ``                 |
| `multiset`           | ``                 |
| `map`                | ``                 |
| `multimap`           | ``                 |
| `unordered_set`      | `set`              |
| `unordered_multiset` | `Counter`          |
| `unordered_map`      | `dict`             |
| `unordered_multimap` | `defaultdict(list)`|

## Collections

* tuple

tuple은 list와 비슷하지만 원소를 추가, 갱신, 삭제가 불가한 immutable
type이다.

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

* list

```python
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

* deque

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

* heapq

```python
>>> import heapq
>>> a = []
>>> heapq.heappush(a, (5, 'write code'))
>>> heapq.heappush(a, (7, 'release product'))
>>> heapq.heappush(a, (1, 'write spec'))
>>> heapq.heappush(a, (3, 'create tests'))
>>> heapq.heappop(a)
(1, 'write spec')
>>> heapq.heappop(a)
(3, 'create tests')
>>> heapq.heappop(a)
(5, 'write code')
>>> heapq.heappop(a)
(7, 'release product')
```

* set

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
```

* Counter

dict의 subclass이다. 리스트 입력 데이터로 부터 값과 출현횟수를 각각
key와 value로 하는 dict이다.

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
```

* dict

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

* defaultdict

dict의 subclass이다. 기본값을 지정할 수 있는 dict이다.  기본값은
callable하거나 None이어야 한다.

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

* namedtuple

tuple의 subclass이다. tuple은 index로만 접근 가능하지만
namedtuple은 index, name으로 접근 가능하다.

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

* ChainMap

dict의 subclass이다. 여러 개의 dict를 모아서 하나의 dict처럼 사용한다.
여러개의 dict를 이용하여 하나의 dict를 생성하거나 update를 사용하는
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

* OrderedDict

dict의 subclass이다. 순서가 보장되는 dict이다.

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

## underscore

[참고](https://dbader.org/blog/meaning-of-underscores-in-python)

파이썬은 다음과 같이 5 가지 underscore 문법을 가지고 있다.

| pattern | example | desc |
|:--------|:--------|:-----|
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

## generator

`iterator` 를 리턴하는 `function` 이다. `yield` 를 이용하여 `iterator` 를 리턴한다. `yield` 를 사용한 `function` 이라고 할 수도 있다.

```python
def only_odds(nums):
    for n in nums:
        if n % 2 == 1:
            yield n
>>> odds = only_odds(range(100))
>>> next(odds)
1
>>> next(odds)
3
```

## pdb

pdb로 break point하고 싶은 라인에 다음을 복사하고 실행하자.

```python
import pdb; pdb.set_trace()
```

```
(Pdb) l         실행중인 코드 위치 확인
(Pdb) pdb.pm()  마지막 에러가 발생했던 곳을 디버깅

```

[visual studio python](https://www.visualstudio.com/ko/vs/python/)도 멋지다.
[python in visual studio code](https://code.visualstudio.com/docs/languages/python)도 멋지다.

## async, await

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

## performance check

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

임의의 class는 attributes를 dict를 이용하여 저장한다. dict는 메모리를
많이 소모한다.  __slots__를 이용하면 dict를 사용하지 않고 attributes를
저장할 수 있기에 메모리를 적게 사용한다. 그러나 동적으로 attributes를
정의 할 수 없다. 특정 class의 attributes는 생성과 동시에 정해지고
runtime에 추가될 일이 없다면 __slots__를 이용하자.

## `__metaclass__`

* class는 class instance의 type이고 metaclass는 class의
  type이다. 
  
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
>>> b = type('Bar', (), {})
>>> b.__class__
<class 'type'>
```

* metaclass를 정의 하면 metaclass를 소유한 class를 새롭게 정의할 수 있다.

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

## weakref

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

## memory leak

gc가 순환 참조들을 수거하긴 하지만 많은 비용이 필요한 작업이다.
reference count를 잘 관리해서 gc의 도움 없이 메모리 누수를 방지하도록
하자.

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

# Advanced Usages

## Builtin functions

[builtin functions](https://docs.python.org/ko/3/library/functions.html)

* `all(iterable)`

```py
def all(iterable):
    for element in iterable:
        if not element:
            return False
    return True

# all values true
l = [1, 3, 4, 5]
print(all(l))

# all values false
l = [0, False]
print(all(l))

# one false value
l = [1, 3, 4, 0]
print(all(l))

# one true value
l = [0, False, 5]
print(all(l))

# empty iterable
l = []
print(all(l))
```

* `any(iterable)`

```py
def any(iterable):
    for element in iterable:
        if element:
            return True
    return False

l = [1, 3, 4, 0]
print(any(l))

l = [0, False]
print(any(l))

l = [0, False, 5]
print(any(l))

l = []
print(any(l))
```

* `zip(iterable)`

```py
def zip(*iterables):
    # zip('ABCD', 'xy') --> Ax By
    sentinel = object()
    iterators = [iter(it) for it in iterables]
    while iterators:
        result = []
        for it in iterators:
            elem = next(it, sentinel)
            if elem is sentinel:
                return
            result.append(elem)
        yield tuple(result)
```

```bash
>>> x = [1, 2, 3]
>>> y = [4, 5, 6]
>>> zipped = zip(x, y)
>>> list(zipped)
[(1, 4), (2, 5), (3, 6)]
# zip()을 * 연산자와 함께 쓰면 리스트를 unzip 할 수 있습니다:
>>> x2, y2 = zip(*zip(x, y))
>>> x == list(x2) and y == list(y2)
True
```

# Library

## regex

```python
import re
p = re.compile(r'(?P<word>\b\w*\b)')
m = p.search('(((( Lots of punctuation )))')
print(m.group('word'))
print(m.group(0))
print(m.group(1))
```

## numpy

## pandas

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

