- [Materials](#materials)
- [Code Lay-out](#code-lay-out)
  - [Indentation](#indentation)
  - [Tabs or Spaces?](#tabs-or-spaces)
  - [Maximum Line Length](#maximum-line-length)
  - [Should a Line Break Before or After a Binary Operator?](#should-a-line-break-before-or-after-a-binary-operator)
  - [Blank Lines](#blank-lines)
  - [Source File Encoding](#source-file-encoding)
  - [Imports](#imports)
  - [Module Level Dunder Names](#module-level-dunder-names)
- [String Quotes](#string-quotes)
- [Whitespace in Expressions and Statements](#whitespace-in-expressions-and-statements)
  - [Pet Peeves](#pet-peeves)
  - [Other Recommendations](#other-recommendations)
- [When to Use Trailing Commas](#when-to-use-trailing-commas)
- [Comments](#comments)
  - [Block Comments](#block-comments)
  - [Inline Comments](#inline-comments)
  - [Documentation Strings](#documentation-strings)
- [Naming Conventions](#naming-conventions)
  - [Overriding Principle](#overriding-principle)
  - [Descriptive: Naming Styles](#descriptive-naming-styles)
  - [Prescriptive: Naming Conventions](#prescriptive-naming-conventions)
    - [Names to Avoid](#names-to-avoid)
    - [ASCII Compatibility](#ascii-compatibility)
    - [Package and Module Names](#package-and-module-names)
    - [Class Names](#class-names)
    - [Type Variable Names](#type-variable-names)
    - [Exception Names](#exception-names)
    - [Global Variable Names](#global-variable-names)
    - [Function and Variable Names](#function-and-variable-names)
    - [Function and Method Arguments](#function-and-method-arguments)
    - [Method Names and Instance Variables](#method-names-and-instance-variables)
    - [Constants](#constants)
    - [Designing for Inheritance](#designing-for-inheritance)
  - [Public and Internal Interfaces](#public-and-internal-interfaces)
- [Programming Recommendations](#programming-recommendations)
  - [Function Annotations](#function-annotations)
  - [Variable Annotations](#variable-annotations)

----

# Materials

* [PEP 8 Style Guide](https://peps.python.org/pep-0008/)

# Code Lay-out

## Indentation

Use 4 spaces per indentation level. Example:

```py
def my_function():
    if condition:
        do_something()
    else:
        do_something_else()
```

## Tabs or Spaces?

Use spaces exclusively, as Python 3 disallows mixing of tabs and spaces.

```py
def my_function():
    do_something()  # Indented with spaces
```

## Maximum Line Length

Limit all lines to a maximum of 79 characters. 

```py
# as-is
def my_function(some_variable, another_variable, yet_another_variable, and_one_more_variable, last_variable):
    pass

# to-be
def my_function(some_variable, another_variable, yet_another_variable,
                and_one_more_variable, last_variable):
    pass
```

## Should a Line Break Before or After a Binary Operator?

Break lines before binary operators.

```py
total = (first_variable
         + second_variable
         - third_variable)
```

## Blank Lines

Use two blank lines to separate top-level functions and classes, and one blank
line between class methods or inside functions to separate sections. 

```py
def first_function():
    pass


def second_function():
    pass


class MyClass:
 
    def method_one(self):
        pass
 
    def method_two(self):
        pass

```

## Source File Encoding

Use UTF-8 as the default source file encoding.

## Imports

* Import one module per line.
* Group imports in the following order: standard library, third-party libraries,
  and local application/library-specific imports.
* Add a blank line between each group of imports. 

```py
import os
import sys

import numpy as np

import my_local_module
```

## Module Level Dunder Names

Place "dunder" names (e.g., `__author__`, `__version__`) after the module docstring
but before any imports or other code. 

```py
"""My Module to demonstrate PEP 8 guidelines."""

__author__ = 'My Name'
__version__ = '1.0.0'

import os
import sys
...
```

# String Quotes

Use single or double quotes consistently throughout the project. Pick one
convention and stick to it. 

```py
string1 = 'Single quotes example'
string2 = 'Another single quotes example'
```

# Whitespace in Expressions and Statements

## Pet Peeves

Avoid using extraneous whitespace in the following situations:

* Immediately inside parentheses, brackets, or braces
* Between a trailing comma and a closing parenthesis
* Immediately before a comma, semicolon, or colon
* Immediately before an open parenthesis that starts a function call or indexing

```py
# Bad
my_list = [ 'a', 'b', 'c' ]
my_dict = { 'a': 1, 'b': 2 }
do_something(a = 1, b = 2)

# Good
my_list = ['a', 'b', 'c']
my_dict = {'a': 1, 'b': 2}
do_something(a=1, b=2)
```

## Other Recommendations

Always surround these binary operators with a single space on either side:
assignment (`=`), augmented assignment (`+=`, `-=`, etc.), comparisons (`==`,
`<`, `>`, `!=`, etc.), and Booleans (`and`, `or`, `not`).

Don't use spaces around the equals sign when indicating default values for
function arguments.

# When to Use Trailing Commas

Use trailing commas when making a multiline collection or sequence of items. It
helps with version control and makes it easier to add or remove items later. 

```py
my_list = [
    'item1',
    'item2',
    'item3',
]
```

# Comments

## Block Comments

* Indent block comments in the same level as the code they describe.
* Start each line with a `#` followed by a single space.
* Separate paragraphs by a line containing a single

```py
# This is a block comment. It explains
# the purpose of the function or a section
# of the code.
#
# The second paragraph provides more details
# about the implementation.
```

## Inline Comments

* Use sparingly.
* Use a single space after the code and start with a `#` followed by another
  space.

```py
x = x + 1  # Increment x by 1
```

## Documentation Strings

* Write a docstring for all public modules, functions, classes, and methods.
  Docstrings are enclosed in triple quotes.
* Use the triple-double-quote style `"""` for consistency. 

```py
def my_function():
    """
    This function does something important.

    It returns an integer value representing its result.
    """
    pass
```

# Naming Conventions

## Overriding Principle

Names should be descriptive and not too short. They should convey their purpose,
and use underscores to improve readability if needed.

## Descriptive: Naming Styles

* lowercase: function_and_variable_names
* lower_case_with_underscores: this_is_a_variable
* UPPERCASE: CONSTANT_VALUE
* UPPER_CASE_WITH_UNDERSCORES: ENUM_OPTIONS
* CapitalizedWords (CamelCase): ClassName
* mixedCase: variableName
* Camel_Case_With_Underscores: Function_Name

## Prescriptive: Naming Conventions

### Names to Avoid

Avoid using single-character names (except for simple loop indices), names that
are not descriptive, or names that can be confused with the names of other
variables or reserved words.

### ASCII Compatibility

Names should be restricted to ASCII characters to ensure compatibility across
different languages and systems.

### Package and Module Names

Packages and modules should have short, all-lowercase names, without underscores
if possible.

```py
import requests
import numpy
```

### Class Names

Class names should use CamelCase (usually CapitalizedWords) convention and
should not contain underscores.

```py
class MyClass:
    pass
```

### Type Variable Names

Type variables should use CamelCase with the prefix `T`.

```py
from typing import TypeVar
TElement = TypeVar('TElement')
```

### Exception Names

Exception names should be based on the class names, but with the Error suffix.

```py
class MyError(Exception):
    pass
```

### Global Variable Names

Global variables should follow the same conventions as function and variable
names - lowercase with underscores.

```py
global_variable = 42
```

### Function and Variable Names

Function and variable names should be lowercase, with words separated by
underscores. They should also be descriptive and not too short.

```py
def my_function():
    variable_name = "John"
```

### Function and Method Arguments

Instance methods should have their first argument named `self`. Class methods
should have their first argument named `cls`.

```py
class MyClass:
    def instance_method(self, arg1):
        pass

    @classmethod
    def class_method(cls, arg1):
        pass
```

### Method Names and Instance Variables

Method names and instance variables should use the same naming conventions as
functions and variable names - lowercase with underscores.

```py
class MyClass:
    instance_variable = 42

    def my_method(self):
        pass
```

### Constants

Constant values should be written in uppercase, with words separated by
underscores.

```py
PI = 3.14159
MAX_VALUE = 1000
```

### Designing for Inheritance

Always decide whether a class's methods and instance variables (collectively,
its "attributes") should be public or non-public. If they need not be public,
use a double underscore prefix (`__`). Public attributes should be those that are
needed for the class to work properly, while non-public attributes should be
implementation details.

## Public and Internal Interfaces

Modules should explicitly define which names are part of their public API by
using the `__all__` attribute. Any name not included in this list should be
considered non-public and subject to change without notice.

```py
__all__ = ['my_function', 'MyClass']
```

# Programming Recommendations

## Function Annotations

Function annotations are a way to provide hints to the user about the expected
types of the function's arguments and return values. They are not enforced by
the Python runtime, but can be used by linters, and documentation tools.
Function annotations are introduced using the `->` syntax, and argument
annotations are provided after the argument name, separated by a colon.

```py
def greet(name: str) -> str:
    """
    Greet the user by their name.
    :param name: The name of the user
    :return: A greeting message string
    """
    return f"Hello, {name}!"

def add_numbers(x: int, y: int) -> int:
    """
    Add two numbers together.
    :param x: The first number
    :param y: The second number
    :return: The sum of the two numbers
    """
    return x + y

from typing import List

def find_max(numbers: List[int]) -> int:
    """
    Find the maximum number in a list.
    :param numbers: A list of integers
    :return: The maximum integer in the list
    """
    return max(numbers)

```

## Variable Annotations

Variable annotations allow you to provide hints to the users about the expected
type of a variable. This can improve readability and maintainability, especially
when using linters or type checkers. Variable annotations are introduced with a
colon followed by the type hint. They can be used for class attributes, instance
attributes, and local variables.

```py
from typing import Dict

# Class attribute variable annotation
class MyClass:
    some_attribute: int

# Instance attribute variable annotation
class Person:
    def __init__(self, name: str, age: int) -> None:
        self.name: str = name
        self.age: int = age

# Local variable annotation
def process_data(data: Dict[str, int]) -> str:
    result: str = ""
    for key, value in data.items():
        result += f"{key}: {value}\n"
    return result

```
