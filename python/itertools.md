- [Materials](#materials)
- [Basic](#basic)
  - [Infinite iterators](#infinite-iterators)
    - [count](#count)
    - [cycle](#cycle)
    - [repeat](#repeat)
  - [Iterators terminating on the shortest input sequence](#iterators-terminating-on-the-shortest-input-sequence)
    - [accumulate()](#accumulate)
    - [chain()](#chain)
    - [chain.from_iterable()](#chainfrom_iterable)
    - [compress()](#compress)
    - [dropwhile()](#dropwhile)
    - [filterfalse()](#filterfalse)
    - [groupby()](#groupby)
    - [islice()](#islice)
    - [starmap()](#starmap)
    - [takewhile()](#takewhile)
    - [tee()](#tee)
    - [zip_longest()](#zip_longest)
  - [Combinatoric iterators](#combinatoric-iterators)
    - [product()](#product)
    - [permutations()](#permutations)
    - [comibnations()](#comibnations)
    - [combinations_with_replacement()](#combinations_with_replacement)

----

# Materials

* [itertools — Functions creating iterators for efficient looping @ python3](https://docs.python.org/3/library/itertools.html)

# Basic

## Infinite iterators

### count

```py
# itertools.count(start=0, step=1)
def count(start=0, step=1):
  n = start
  while True:
    yield n
    n += step   
for i in count(5):
  input('')
  print(i)
```

### cycle

```py
# itertools.cycle(iterable)  
def cycle(iterable):
    # cycle('ABCD') --> A B C D A B C D A B C D ...
    saved = []
    for element in iterable:
        yield element
        saved.append(element)
    while saved:
        for element in saved:
              yield element
for i in cycle('ABCD'):
  input('')
  print(i)
```

### repeat

```py
# itertools.repeat(object[, times])
def repeat(object, times=None):
    # repeat(10, 3) --> 10 10 10
    if times is None:
        while True:
            yield object
    else:
        for i in range(times):
            yield object
list(map(pow, range(10), repeat(2))) # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

## Iterators terminating on the shortest input sequence

### accumulate()

### chain()

### chain.from_iterable()

### compress()

### dropwhile()

### filterfalse()

### groupby()

### islice()

### starmap()

### takewhile()

### tee()

### zip_longest()

## Combinatoric iterators

### product()

### permutations()

### comibnations()

### combinations_with_replacement()
