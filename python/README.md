<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Abstract](#abstract)
- [Usage](#usage)
    - [container](#container)
    - [generator](#generator)
    - [pdb](#pdb)
    - [async, await](#async-await)
    - [performance check](#performance-check)
    - [__slots__](#slots)
    - [__metaclass__](#metaclass)
    - [weakref](#weakref)
    - [memory leak](#memory-leak)
    - [gc](#gc)
    - [dependencies](#dependencies)
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
- [References](#references)

<!-- markdown-toc end -->


# Abstract

python3에 대해 정리한다.

# Usage

## container

* dict, list, set, tuple
* namedtuple, deque, ChainMap, Counter, OrderedDict, defaultdict

```python
>>> from collections import deque
>>> d = deque('abc')
>> d.appendleft('1')
>>> for elem in d:
...   print(elem.upper())
1
a
b
c
>>> sub = OrderedDict([('S', 1), ('U', 2), ('B', 3)])
>>> sub['S']
1
>>> sub.keys()
['S', 'U', 'B']
```

## generator

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
  
  
## __slots__

임의의 class는 attributes를 dict를 이용하여 저장한다. dict는 메모리를
많이 소모한다.  __slots__를 이용하면 dict를 사용하지 않고 attributes를
저장할 수 있기에 메모리를 적게 사용한다. 그러나 동적으로 attributes를
정의 할 수 없다. 특정 class의 attributes는 생성과 동시에 정해지고
runtime에 추가될 일이 없다면 __slots__를 이용하자.

## __metaclass__

* type를 이용하면 class를 runtime에 정의할 수 있다.

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



## weakref



## memory leak

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
탐색하여 쓰레기를 수집한다.

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

# References

* [파이썬 생존 안내서](https://www.slideshare.net/sublee/ss-67589513)
  * 듀랑고를 제작한 왓스튜디오의 이흥섭 PT
* [awesome-python](https://awesome-python.com/)
  * A curated list of awesome python things
* [python package index](https://pypi.python.org/pypi)
  * library검색이 용이하다.
