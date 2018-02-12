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
  
  
## memory optimization

## memory leak

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

# References

* [파이썬 생존 안내서](https://www.slideshare.net/sublee/ss-67589513)
  * 듀랑고를 제작한 왓스튜디오의 이흥섭 PT
* [awesome-python](https://awesome-python.com/)
  * A curated list of awesome python things
* [python package index](https://pypi.python.org/pypi)
  * library검색이 용이하다.
