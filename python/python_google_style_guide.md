- [Materials](#materials)
- [Python Language Rules](#python-language-rules)
  - [Lint](#lint)
  - [Imports](#imports)
  - [Packages](#packages)
  - [Exceptions](#exceptions)
  - [Mutable Global State](#mutable-global-state)
  - [Nested/Local/Inner Classes and Functions](#nestedlocalinner-classes-and-functions)
  - [Comprehensions \& Generator Expressions](#comprehensions--generator-expressions)
  - [Default Iterators and Operators](#default-iterators-and-operators)
  - [Generators](#generators)
  - [Lambda Functions](#lambda-functions)
  - [Conditional Expressions](#conditional-expressions)
  - [Default Argument Values](#default-argument-values)
  - [Properties](#properties)
  - [True/False Evaluations](#truefalse-evaluations)
  - [Lexical Scoping](#lexical-scoping)
  - [Function and Method Decorators](#function-and-method-decorators)
  - [Threading](#threading)
  - [Power Features](#power-features)
  - [Modern Python: from __future__ imports](#modern-python-from-future-imports)
  - [Type Annotated Code](#type-annotated-code)
- [Python Style Rules](#python-style-rules)
  - [Semicolons](#semicolons)
  - [Line length](#line-length)
  - [Parentheses](#parentheses)
  - [Indentation](#indentation)
    - [Trailing commas in sequences of items?](#trailing-commas-in-sequences-of-items)
  - [Blank Lines](#blank-lines)
  - [Whitespace](#whitespace)
  - [Shebang Line](#shebang-line)
  - [Comments and Docstrings](#comments-and-docstrings)
    - [Docstrings](#docstrings)
    - [Modules](#modules)
      - [Test modules](#test-modules)
    - [Functions and Methods](#functions-and-methods)
    - [Classes](#classes)
    - [Block and Inline Comments](#block-and-inline-comments)
    - [Punctuation, Spelling, and Grammar](#punctuation-spelling-and-grammar)
  - [Strings](#strings)
    - [Logging](#logging)
    - [Error Messages](#error-messages)
  - [Files, Sockets, and similar Stateful Resources](#files-sockets-and-similar-stateful-resources)
  - [TODO Comments](#todo-comments)
  - [Imports formatting](#imports-formatting)
  - [Statements](#statements)
  - [Accessors](#accessors)
  - [Naming](#naming)
    - [Names to Avoid](#names-to-avoid)
    - [Naming Conventions](#naming-conventions)
    - [File Naming](#file-naming)
    - [Guidelines derived from Guido’s Recommendations](#guidelines-derived-from-guidos-recommendations)
  - [Main](#main)
  - [Function length](#function-length)
  - [Type Annotations](#type-annotations)
    - [General Rules](#general-rules)
    - [Line Breaking](#line-breaking)
    - [Forward Declarations](#forward-declarations)
    - [Default Values](#default-values)
    - [NoneType](#nonetype)
    - [Type Aliases](#type-aliases)
    - [Ignoring Types](#ignoring-types)
    - [Typing Variables](#typing-variables)
    - [Tuples vs Lists](#tuples-vs-lists)
    - [Type variables](#type-variables)
    - [String types](#string-types)
    - [Imports For Typing](#imports-for-typing)
    - [Conditional Imports](#conditional-imports)
    - [Circular Dependencies](#circular-dependencies)
    - [Generics](#generics)

----

# Materials

* [Python Style Guide | google](https://google.github.io/styleguide/pyguide.html)

# Python Language Rules

## Lint

Linting refers to checking Python code for coding standards, potential errors,
and stylistic errors using a linter tool like pylint or flake8. Lint tools help
to ensure consistency and improve code readability.

```py
# bad
def add(a,b): return a+b
# good
def add(a, b):
    return a + b
```

## Imports

Imports are statements that bring modules, functions, classes, and variables
from other files into the current scope. They should be placed at the top of the
file and separated into three groups: standard libraries, third-party libraries,
and local application imports.

```py
# good
import os
import sys

import requests

import my_local_module
```

## Packages

Packages are a way of organizing related modules in a directory hierarchy. They
should have an `__init__.py` file and follow the
[PEP8](python_pep8_style_guide.md) naming conventions.

```py
my_package/
    __init__.py
    module1.py
    module2.py
    submodule/
    __init__.py
    module3.py
```

## Exceptions

Exceptions are used to handle errors that may occur during the execution of a
program. They should be raised with a descriptive error message and caught using
try-except blocks.

```py
def divide(a, b):
    if b == 0:
        raise ValueError("division by zero")
    return a / b

try:
    result = divide(10, 0)
except ValueError as e:
    print(f"An error occurred: {e}")
```

## Mutable Global State

Mutable global state refers to global variables that can be modified. They
should be avoided to prevent unexpected behavior and improve code
maintainability.

```py
# bad
global_counter = 0

def increment_counter():
    global global_counter
    global_counter += 1
```

## Nested/Local/Inner Classes and Functions

Nested classes and functions are defined inside another class or function. They
should be used when the nested component is only used within the containing
function or class and when it improves code readability.

```py
class OuterClass:
    class InnerClass:
        pass

def outer_function():
    def inner_function():
        pass
```

## Comprehensions & Generator Expressions

Comprehensions (list, set, and dictionary) and generator expressions provide a
concise way to create or transform iterables.

```py
squared_numbers = [x * x for x in range(10)]

even_numbers = {x for x in range(10) if x % 2 == 0}

word_lengths = {word: len(word) for word in ["apple", "banana", "cherry"]}
```

## Default Iterators and Operators

Default iterators and operators should be used over manual loop implementation
for better performance and readability.

```py
# bad
for i in range(len(my_list)):
    print(my_list[i])
# good
for element in my_list:
    print(element)
```

## Generators

Generators are functions that use the `yield` keyword to produce a sequence of
values when iterated. They should be used for large or infinite sequences to
save memory and improve performance.

```py
def fibonnaci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b
```

## Lambda Functions

Lambda functions are anonymous, single-expression functions that can be defined
inline. They should be used when a short, simple function is needed.

```py
sorted_list = sorted(my_list, key=lambda x: x[1])
```

## Conditional Expressions

These are short and concise expressions used to assign a value based on a
condition.

```py
x = 5
result = "even" if x % 2 == 0 else "odd"
```

## Default Argument Values

Default argument values can be used in function definitions to provide default
values for optional parameters.

```py
def greet(name, greeting="Hello"):
    print(f"{greeting}, {name}")
```

## Properties

Properties are a way to add logic to attribute access and assignment in classes
using `@property` and `@<attribute>`.setter decorators.

```py
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, radius):
        if radius < 0:
            raise ValueError("Radius cannot be negative.")
        self._radius = radius
```

## True/False Evaluations

Objects in Python can be evaluated in a boolean context (e.g., in conditions)
using truthiness values.

```py
my_list = []
if not my_list:
    print("List is empty.")
```

## Lexical Scoping

Lexical scoping is the way variables are resolved based on their location in the
source code, not on the calling context. Python supports lexical scoping by
default.

```py
def outer_function():
    outer_variable = "Hello from outer_function"
    
    def inner_function():
        print(outer_variable)
    
    inner_function()

outer_function()
```

## Function and Method Decorators

Decorators are functions that modify the behavior or properties of other
functions or methods.

```py
def log_calls(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@log_calls
def greet(name):
    print(f"Hello, {name}")
```

## Threading

Threading is the use of multiple threads in a program for concurrent execution.
The `threading` module provides tools for thread management and synchronization.

```py
import threading

def print_numbers():
    for i in range(5):
        print(i)

thread = threading.Thread(target=print_numbers)
thread.start()
thread.join()
```

## Power Features

Power features, such as metaclasses and dynamic typing, can be used to solve
complex problems but should be used sparingly and with caution due to their
complexity.

```py
class Meta(type):
    def __new__(cls, name, bases, attrs):
        new_cls = super().__new__(cls, name, bases, attrs)
        # Custom logic here
        return new_cls

class MyClass(metaclass=Meta):
    pass
```

## Modern Python: from __future__ imports

The `__future__` module provides features that are planned to be included in
future versions of Python. These features should be used with caution to ensure
compatibility with future Python releases.

```py
from __future__ import annotations

def greet(name: str) -> str:
    return f"Hello, {name}"
```

## Type Annotated Code

Type annotations are a way to specify the expected types of variables, function
arguments, and return values. They can be used to improve code readability and
enable type checking with tools like mypy.

```py
def add(a: int, b: int) -> int:
    return a + b

result: float = add(1, 2)
```

# Python Style Rules

## Semicolons

Avoid using semicolons (`;`) to separate multiple statements on a single line.

```py
# Yes:
x = 1
y = 2

# No:
x = 1; y = 2
```

## Line length

Maximum line length is 80 characters.

```py
# Example of exceeding 80 characters:
long_function_name(argument_one, argument_two, argument_three, argument_four)
```

## Parentheses

Use parentheses sparingly and for clarity.

```py
# Yes:
if (x > 0 and y > 0) or z > 0:
    pass

# No:
if x > 0 and y > 0 or z > 0:
    pass
```

## Indentation

Use 4 spaces per indentation level.

### Trailing commas in sequences of items?

Include trailing commas in sequences of items.

```py
names = [
    "Alice",
    "Bob",
    "Charlie",
]
```

## Blank Lines

Two blank lines between top-level functions or classes.

```py
def foo():
    pass


def bar():
    pass
```

## Whitespace

Avoid unnecessary whitespace.

```py
# Yes:
x = 1
y = [1, 2, 3]

# No:
x=1
y = [ 1, 2, 3 ]
```

## Shebang Line

Include a shebang line for executable scripts.

```py
#!/usr/bin/env python3
```

## Comments and Docstrings

### Docstrings

Use proper docstring conventions.

```py
def foo():
    """This function does something."""
    pass
```

### Modules

Document modules with a short description.

#### Test modules

Provide a description and examples of how to run the tests if applicable.

```py
"""This blaze test uses golden files.

You can update those files by running
`blaze run //foo/bar:foo_test -- --update_golden_files` from the `google3`
directory.
"""
```

### Functions and Methods

Supply a clear description of the function or method's purpose, arguments,
return values, and any exceptions raised.

### Classes

Include a docstring in the class definition, explaining the purpose of the class
and its attributes and methods.

```py
class SampleClass:
    """Summary of class here.

    Longer class information...
    Longer class information...

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """

    def __init__(self, likes_spam: bool = False):
        """Initializes the instance based on spam preference.

        Args:
          likes_spam: Defines if instance exhibits this preference.
        """
        self.likes_spam = likes_spam
        self.eggs = 0

    def public_method(self):
        """Performs operation blah."""
```

The class docstring should not repeat unnecessary information, such as that the
class is a class

```py
# Yes:
class CheeseShopAddress:
  """The address of a cheese shop.

  ...
  """

class OutOfCheeseError(Exception):
  """No more cheese is available."""
# No:
class CheeseShopAddress:
  """Class that describes the address of a cheese shop.

  ...
  """

class OutOfCheeseError(Exception):
  """Raised when no more cheese is available."""
```

### Block and Inline Comments

Use block and inline comments to clarify code when necessary.

```py
# This is a block comment.

x = 1  # This is an inline comment.
```

### Punctuation, Spelling, and Grammar

Ensure proper punctuation, spelling, and grammar in comments and docstrings.

## Strings

### Logging

Use the logging library instead of print statements for debugging and logging
purposes, as it provides a more flexible way of handling messages.

```py
import logging

logging.basicConfig(level=logging.DEBUG)

logging.debug('This is a debug message')
logging.info('This is an info message')
logging.warning('This is a warning message')
logging.error('This is an error message')
logging.critical('This is a critical message')
```

### Error Messages

When raising exceptions, provide a clear and descriptive error message.

```py
if x < 0:
    raise ValueError('x must be non-negative, but got {}'.format(x))
```

## Files, Sockets, and similar Stateful Resources

Use the `with` statement for managing resources like files and sockets. This
ensures that the resources are properly closed when the block is exited.

```py
with open('example.txt', 'r') as file:
    content = file.read()
```

## TODO Comments

Mark work to be done using TODO comments.

```py
# TODO: Implement this feature later.
```

## Imports formatting

Separate imports into three groups: standard library, third-party library, and
local modules. Sort imports alphabetically within each group.

```py
import os
import sys

import numpy as np

import my_module
```

## Statements

Each statement should be on a separate line. Avoid putting multiple statements
on a single line.

```py
x = 1
y = 2
z = x + y
```

## Accessors

Use the properties provided by classes to access or modify an object's
attributes, rather than accessing them directly.

```py
class Person:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

person = Person('John')
print(person.name)  # Access the name using the property
person.name = 'Jane'  # Modify the name using the property
```

## Naming

### Names to Avoid

Avoid using single-letter names or names that are too long and difficult to
understand.

```py
# bad
a = 42
# good
answer = 42
```

### Naming Conventions

In Python, it's essential to follow certain naming conventions to keep the code
clean and readable. Here are some common naming conventions:

* Class names: Use CamelCase notation. Example: `class ShoppingCart`:
* Function and variable names: Use lowercase words separated by underscores.
  Example: `def cart_total()`:
* Constants: Use uppercase letters with words separated by underscores. Example:
  `SALES_TAX_RATE = 0.08`

### File Naming

When naming Python files, use lowercase letters and separate words with
underscores. The file should have the `.py` extension.

`shopping_cart.py`

### Guidelines derived from Guido’s Recommendations

* Use 4 spaces for indentation, not tabs.
* Limit all lines to a maximum of 79 characters.
* Use blank lines to separate functions and classes, and larger blocks of code
  within functions.
* When possible, put comments on a line of their own.
* Use docstrings on functions and classes to explain their purpose and usage.

## Main

the starting point of a program is the main function. It's good practice to
include a main function in your scripts to provide an entry point for your code.

```py
def main():
    print("Welcome to the Python program!")
    
if __name__ == "__main__":
    main()
```

## Function length

A function should not be too long; it should only perform one task. Ideally, a
function should fit into the column size of 80 characters at most. If a function
exceeds this limit, consider breaking it into smaller functions.

```py
def compute_tax(income, tax_rate):
    return income * tax_rate


def calculate_total_income(base_income, bonus):
    return base_income + bonus


def main():
    income = calculate_total_income(50000, 10000)
    tax = compute_tax(income, 0.2)
    print(f"The tax to be paid is: {tax}")

if __name__ == "__main__":
    main()
```

## Type Annotations

### General Rules

Use the typing module from Python's standard library for type hinting. Specify
the types of function arguments and return values using the "->" syntax.

```py
from typing import List, Tuple, Dict, Any, Union


def greet(name: str) -> str:
    return f"Hello, {name}!"


def add_numbers(a: int, b: int) -> int:
    return a + b


def multiply_numbers(numbers: List[int]) -> int:
    result = 1
    for number in numbers:
        result *= number
    return result
```

### Line Breaking

When breaking a type hint that exceeds 80 characters across multiple lines, use
parentheses and align the continued lines with the first character after the
opening parenthesis.

```py
def my_long_function_name(
    my_long_parameter_name: Tuple[int, float, Union[Dict[str, Any], List[int]]]
) -> Dict[str, Any]:
    pass
```

### Forward Declarations

Use forward references (as a string) for types that are not yet defined or when
referencing a type within its own definition.

```py
class MyClass:

    def set_value(self, value: "MyClass") -> None:
        self.value = value

    def get_value(self) -> "MyClass":
        return self.value
```

### Default Values

If a function's argument has a default value, you should still provide a type
annotation for that argument.

```python
def greet(name: str = "World") -> str:
    return f"Hello, {name}!"
```

### NoneType

In Python 3.5, you can use "None" as a type hint for a variable that is expected
to have the value None.

```py
def do_nothing() -> None:
    pass
```

### Type Aliases

Type aliases can be used to define shorthand or more descriptive names for
complex types.

```py
from typing import List, Tuple

Coordinates = List[Tuple[float, float]]

def calculate_distance(points: Coordinates) -> float:
    pass
```

### Ignoring Types

To signal that a type hint should be ignored, use the "Any" type from the typing
module. Use sparingly and only when necessary.

```py
from typing import Any

def messy_function(argument: Any) -> Any:
    pass
```

### Typing Variables

You can also provide type annotations for variables within a function or class.

```py
def calculate_sum(numbers: List[int]) -> int:
    total: int = 0
    for num in numbers:
        total += num
    return total
```

### Tuples vs Lists

Use "Tuple" for fixed-length sequences and "List" for variable-length sequences.

```py
from typing import List, Tuple

def split_name(name: str) -> Tuple[str, str]:
    first, last = name.split(" ")
    return first, last

def gather_names() -> List[str]:
    return ["Alice", "Bob", "Charlie"]
```

### Type variables

Type variables can be used as placeholders for types in generic functions or
classes.

```py
from typing import TypeVar, List

T = TypeVar("T")

def first_item(items: List[T]) -> T:
    return items[0]
```

### String types

Use "str" for Unicode strings and "bytes" for binary data.

```py
def write_file(filename: str, content: str) -> None:
    with open(filename, "w") as f:
        f.write(content)

def read_file(filename: str) -> bytes:
    with open(filename, "rb") as f:
        return f.read()
```

### Imports For Typing

Keep imports related to typing separate from other imports. Preferably place
them after other imports or at the bottom of the import list.

```py
import math

from typing import List, Tuple
```

### Conditional Imports

Conditional imports can be used when a type is only available or necessary in
certain situations.

```py
import sys
from typing import List

if sys.version_info[0] >= 3:
    from typing import Text
else:
    Text = str

def split_lines(text: Text) -> List[str]:
    return text.splitlines()
```

### Circular Dependencies

In cases of circular dependencies, use the `TYPE_CHECKING` constant from the
typing module in a conditional import.

```py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from my_other_module import OtherClass

class MyClass:
    def do_something(self, other: "OtherClass") -> None:
        pass
```

### Generics

Use generics like "Iterable", "Mapping", and "Sequence" when you want to express
that a function works with a collection of a certain type but doesn't care about
the specific implementation (list, dict, etc.).

```py
from typing import Iterable, Sequence, Mapping

def sum_numbers(numbers: Iterable[int]) -> int:
    return sum(numbers)

def get_keys(mapping: Mapping[str, int]) -> Sequence[str]:
    return list(mapping.keys())
```
