- [Materials](#materials)
- [Basic](#basic)
  - [all(iterable)](#alliterable)
  - [any(iterable)](#anyiterable)
  - [zip(iterable)](#zipiterable)
  - [map](#map)

----

# Materials

* [Built-in Functions @ python3](https://docs.python.org/3/library/functions.html)

# Basic

## all(iterable)

```py
def all(iterable):
    for element in iterable:
        if not element:
            return False
    return True

# all values true
>>> l = [1, 3, 4, 5]
>>> print(all(l))
True
# all values false
>>> l = [0, False]
>>> print(all(l))
False
# one false value
>>> l = [1, 3, 4, 0]
>>> print(all(l))
False
# one true value
>>> l = [0, False, 5]
>>> print(all(l))
False
# empty iterable
>>> l = []
>>> print(all(l))
True
```

## any(iterable)

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

## zip(iterable)

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

## map

```py
# map(function, iterable, ...)
def addition(n):
  return n + n
numbers = (1, 2, 3, 4)  
result = map(addition, numbers)
print(list(result)) # [2, 4, 6, 8]

result = map(lambda x: x + x, numbers)
print(list(result)) # [2, 4, 6, 8]

numbers1 = [1, 2, 3]
numbers2 = [4, 5, 6]
result = map(lambda x, y: x + y, numbers1, numbers2)
print(list(result)) # [5, 7, 9]

l = ['sat', 'bat', 'cat', 'mat'] 
test = list(map(list, l))
print(test) # [['s', 'a', 't'], ['b', 'a', 't'], ['c', 'a', 't'], ['m', 'a', 't']]
```
