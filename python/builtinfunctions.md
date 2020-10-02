- [Materials](#materials)
- [Basic](#basic)
  - [map](#map)

----

# Materials

* [Built-in Functions @ python3](https://docs.python.org/3/library/functions.html)

# Basic

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
