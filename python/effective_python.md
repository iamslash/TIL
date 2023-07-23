- [Materials](#materials)
- [Chapter 1: Pythonic Thinking](#chapter-1-pythonic-thinking)
  - [Know Which Version of Python You’re Using](#know-which-version-of-python-youre-using)
  - [Follow the PEP 8 Style Guide](#follow-the-pep-8-style-guide)
  - [Know the Differences Between bytes and str](#know-the-differences-between-bytes-and-str)
  - [Prefer Interpolated F-Strings Over C-style Format Strings and str.format](#prefer-interpolated-f-strings-over-c-style-format-strings-and-strformat)
  - [Write Helper Functions Instead of Complex Expressions](#write-helper-functions-instead-of-complex-expressions)
  - [Prefer Multiple Assignment Unpacking Over Indexing](#prefer-multiple-assignment-unpacking-over-indexing)
  - [Prefer enumerate Over range](#prefer-enumerate-over-range)
  - [Use zip to Process Iterators in Parallel](#use-zip-to-process-iterators-in-parallel)
  - [Avoid else Blocks After for and while Loops](#avoid-else-blocks-after-for-and-while-loops)
  - [Prevent Repetition with Assignment Expressions](#prevent-repetition-with-assignment-expressions)
- [Chapter 2: Lists and Dictionaries](#chapter-2-lists-and-dictionaries)
  - [Know How to Slice Sequences](#know-how-to-slice-sequences)
  - [Avoid Striding and Slicing in a Single Expression](#avoid-striding-and-slicing-in-a-single-expression)
  - [Prefer Catch-All Unpacking Over Slicing](#prefer-catch-all-unpacking-over-slicing)
  - [Sort by Complex Criteria Using the key Parameter](#sort-by-complex-criteria-using-the-key-parameter)
  - [Be Cautious When Relying on dict Insertion Ordering](#be-cautious-when-relying-on-dict-insertion-ordering)
  - [Prefer get Over in and KeyError to Handle Missing Dictionary Keys](#prefer-get-over-in-and-keyerror-to-handle-missing-dictionary-keys)
  - [Prefer defaultdict Over setdefault to Handle Missing Items in Internal State](#prefer-defaultdict-over-setdefault-to-handle-missing-items-in-internal-state)
  - [Know How to Construct Key-Dependent Default Values with __missing__](#know-how-to-construct-key-dependent-default-values-with-missing)
- [Chapter 3: Functions](#chapter-3-functions)
  - [Never Unpack More Than Three Variables When Functions Return Multiple Values](#never-unpack-more-than-three-variables-when-functions-return-multiple-values)
  - [Prefer Raising Exceptions to Returning None](#prefer-raising-exceptions-to-returning-none)
  - [Know How Closures Interact with Variable Scope](#know-how-closures-interact-with-variable-scope)
  - [Reduce Visual Noise with Variable Positional Arguments](#reduce-visual-noise-with-variable-positional-arguments)
  - [Provide Optional Behavior with Keyword Arguments](#provide-optional-behavior-with-keyword-arguments)
  - [Use None and Docstrings to Specify Dynamic Default Arguments](#use-none-and-docstrings-to-specify-dynamic-default-arguments)
  - [Enforce Clarity with Keyword-Only and Position-Only Arguments](#enforce-clarity-with-keyword-only-and-position-only-arguments)
  - [Define Function Decorators with functools.wraps](#define-function-decorators-with-functoolswraps)
- [Chapter 4: Comprehensions and Generators](#chapter-4-comprehensions-and-generators)
  - [Use Comprehensions Instead of map and filter](#use-comprehensions-instead-of-map-and-filter)
  - [Avoid More Than Two Control Subexpressions in Comprehensions](#avoid-more-than-two-control-subexpressions-in-comprehensions)
  - [Avoid Repeated Work in Comprehensions by Using Assignment Expressions](#avoid-repeated-work-in-comprehensions-by-using-assignment-expressions)
  - [Consider Generators Instead of Returning Lists](#consider-generators-instead-of-returning-lists)
  - [Be Defensive When Iterating Over Arguments](#be-defensive-when-iterating-over-arguments)
  - [Consider Generator Expressions for Large List Comprehensions](#consider-generator-expressions-for-large-list-comprehensions)
  - [Compose Multiple Generators with yield from](#compose-multiple-generators-with-yield-from)
  - [Avoid Injecting Data into Generators with send](#avoid-injecting-data-into-generators-with-send)
  - [Avoid Causing State Transitions in Generators with throw](#avoid-causing-state-transitions-in-generators-with-throw)
  - [Consider itertools for Working with Iterators and Generators](#consider-itertools-for-working-with-iterators-and-generators)
- [Chapter 5: Classes and Interfaces](#chapter-5-classes-and-interfaces)
  - [Compose Classes Instead of Nesting Many Levels of Built-in Types](#compose-classes-instead-of-nesting-many-levels-of-built-in-types)
  - [Accept Functions Instead of Classes for Simple Interfaces](#accept-functions-instead-of-classes-for-simple-interfaces)
  - [Use @classmethod Polymorphism to Construct Objects Generically](#use-classmethod-polymorphism-to-construct-objects-generically)
  - [Initialize Parent Classes with super](#initialize-parent-classes-with-super)
  - [Consider Composing Functionality with Mix-in Classes](#consider-composing-functionality-with-mix-in-classes)
  - [Prefer Public Attributes Over Private Ones](#prefer-public-attributes-over-private-ones)
  - [Inherit from collections.abc for Custom Container Types](#inherit-from-collectionsabc-for-custom-container-types)
- [Chapter 6: Metaclasses and Attributes](#chapter-6-metaclasses-and-attributes)
  - [Use Plain Attributes Instead of Setter and Getter Methods](#use-plain-attributes-instead-of-setter-and-getter-methods)
  - [Consider @property Instead of Refactoring Attributes](#consider-property-instead-of-refactoring-attributes)
  - [Use Descriptors for Reusable @property Methods](#use-descriptors-for-reusable-property-methods)
  - [Use __getattr__, __getattribute__, and __setattr__ for Lazy Attributes](#use-getattr-getattribute-and-setattr-for-lazy-attributes)
  - [Validate Subclasses with __init\_subclass__](#validate-subclasses-with-init_subclass)
  - [Register Class Existence with __init\_subclass__](#register-class-existence-with-init_subclass)
  - [Annotate Class Attributes with __set\_name__](#annotate-class-attributes-with-set_name)
  - [Prefer Class Decorators Over Metaclasses for Composable Class Extensions](#prefer-class-decorators-over-metaclasses-for-composable-class-extensions)
- [Chapter 7: Concurrency and Parallelism](#chapter-7-concurrency-and-parallelism)
  - [Use subprocess to Manage Child Processes](#use-subprocess-to-manage-child-processes)
  - [Use Threads for Blocking I/O, Avoid for Parallelism](#use-threads-for-blocking-io-avoid-for-parallelism)
  - [Use Lock to Prevent Data Races in Threads](#use-lock-to-prevent-data-races-in-threads)
  - [Use Queue to Coordinate Work Between Threads](#use-queue-to-coordinate-work-between-threads)
  - [Know How to Recognize When Concurrency Is Necessary](#know-how-to-recognize-when-concurrency-is-necessary)
  - [Avoid Creating New Thread Instances for On-demand Fan-out](#avoid-creating-new-thread-instances-for-on-demand-fan-out)
  - [Understand How Using Queue for Concurrency Requires Refactoring](#understand-how-using-queue-for-concurrency-requires-refactoring)
  - [Consider ThreadPoolExecutor When Threads Are Necessary for Concurrency](#consider-threadpoolexecutor-when-threads-are-necessary-for-concurrency)
  - [Achieve Highly Concurrent I/O with Coroutines](#achieve-highly-concurrent-io-with-coroutines)
  - [Know How to Port Threaded I/O to asyncio](#know-how-to-port-threaded-io-to-asyncio)
  - [Mix Threads and Coroutines to Ease the Transition to asyncio](#mix-threads-and-coroutines-to-ease-the-transition-to-asyncio)
  - [Avoid Blocking the asyncio Event Loop to Maximize Responsiveness](#avoid-blocking-the-asyncio-event-loop-to-maximize-responsiveness)
  - [Consider concurrent.futures for True Parallelism](#consider-concurrentfutures-for-true-parallelism)
- [Chapter 8: Robustness and Performance](#chapter-8-robustness-and-performance)
  - [Take Advantage of Each Block in try/except/else/finally](#take-advantage-of-each-block-in-tryexceptelsefinally)
  - [Consider contextlib and with Statements for Reusable try/finally Behavior](#consider-contextlib-and-with-statements-for-reusable-tryfinally-behavior)
  - [Use datetime Instead of time for Local Clocks](#use-datetime-instead-of-time-for-local-clocks)
  - [Make pickle Reliable with copyreg](#make-pickle-reliable-with-copyreg)
  - [Use decimal When Precision Is Paramount](#use-decimal-when-precision-is-paramount)
  - [Profile Before Optimizing](#profile-before-optimizing)
  - [Prefer deque for Producer–Consumer Queues](#prefer-deque-for-producerconsumer-queues)
  - [Consider Searching Sorted Sequences with bisect](#consider-searching-sorted-sequences-with-bisect)
  - [Know How to Use heapq for Priority Queues](#know-how-to-use-heapq-for-priority-queues)
  - [Consider memoryview and bytearray for Zero-Copy Interactions with bytes](#consider-memoryview-and-bytearray-for-zero-copy-interactions-with-bytes)
- [Chapter 9: Testing and Debugging](#chapter-9-testing-and-debugging)
  - [Use repr Strings for Debugging Output](#use-repr-strings-for-debugging-output)
  - [Verify Related Behaviors in TestCase Subclasses](#verify-related-behaviors-in-testcase-subclasses)
  - [Isolate Tests from Each Other with setUp, tearDown, setUpModule, and tearDownModule](#isolate-tests-from-each-other-with-setup-teardown-setupmodule-and-teardownmodule)
  - [Use Mocks to Test Code with Complex Dependencies](#use-mocks-to-test-code-with-complex-dependencies)
  - [Encapsulate Dependencies to Facilitate Mocking and Testing](#encapsulate-dependencies-to-facilitate-mocking-and-testing)
  - [Consider Interactive Debugging with pdb](#consider-interactive-debugging-with-pdb)
  - [Use tracemalloc to Understand Memory Usage and Leaks](#use-tracemalloc-to-understand-memory-usage-and-leaks)
- [Chapter 10: Collaboration](#chapter-10-collaboration)
  - [Know Where to Find Community-Built Modules](#know-where-to-find-community-built-modules)
  - [Use Virtual Environments for Isolated and Reproducible Dependencies](#use-virtual-environments-for-isolated-and-reproducible-dependencies)
  - [Write Docstrings for Every Function, Class, and Module](#write-docstrings-for-every-function-class-and-module)
  - [Use Packages to Organize Modules and Provide Stable APIs](#use-packages-to-organize-modules-and-provide-stable-apis)
  - [Consider Module-Scoped Code to Configure Deployment Environments](#consider-module-scoped-code-to-configure-deployment-environments)
  - [Define a Root Exception to Insulate Callers from APIs](#define-a-root-exception-to-insulate-callers-from-apis)
  - [Know How to Break Circular Dependencies](#know-how-to-break-circular-dependencies)
  - [Consider warnings to Refactor and Migrate Usage](#consider-warnings-to-refactor-and-migrate-usage)
  - [Consider Static Analysis via typing to Obviate Bugs](#consider-static-analysis-via-typing-to-obviate-bugs)

---

# Materials

* [Effective Python](https://effectivepython.com/)
  * [src](https://github.com/bslatkin/effectivepython)

# Chapter 1: Pythonic Thinking

## Know Which Version of Python You’re Using

Ensure compatibility by verifying the Python version used in your project. Use
these commands to check:

```py
import sys
print(sys.version_info)
print(sys.version)
```

## Follow the PEP 8 Style Guide

[PEP 8](python_pep8_style_guide.md) is the official style guide for Python. It
makes your code more readable and maintainable. Key points are:

* Use 4 spaces for indentation
* Keep lines under 79 characters
* Use blank lines to separate functions and classes
* Use lowercase_with_underscores for variable and function names
* Use CapWords for class names

## Know the Differences Between bytes and str

`bytes` represents raw binary data while `str` represents Unicode characters. To
convert between them, use `encode()` and `decode()` methods.

```py
string = "Hello, World!"
binary = string.encode('utf-8')
decoded = binary.decode('utf-8')
print(string, binary, decoded)
```

## Prefer Interpolated F-Strings Over C-style Format Strings and str.format

Use f-strings to embed expressions in a string easily and improve readability.

```py
name = "Alice"
age = 25
print(f"{name} is {age} years old.")
```

## Write Helper Functions Instead of Complex Expressions

Break complex expressions into smaller helper function to improve readability
and maintainability.

```py
def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

normalized_value = normalize(3, 0, 10)
```

## Prefer Multiple Assignment Unpacking Over Indexing

Use tuple unpacking to assign multiple variables from an iterable, improving
code readability.

```py
data = ["one", "two", "three"]
first, second, third = data
print(first, second, third)
```

## Prefer enumerate Over range

Use `enumerate()` to get an index and item from an iterable
instead of using `range()` with indexing.

```py
names = ["Alice", "Bob", "Charlie"]
for idx, name in enumerate(names):
    print(idx, name)
```

## Use zip to Process Iterators in Parallel

`zip()` allows you to iterate through multiple lists in parallel, returning a
tuple of elements from each iterable.

```py
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
for name, age in zip(names, ages):
    print(name, age)
```

## Avoid else Blocks After for and while Loops

The `else` block after a loop is often misunderstood. To prevent confusion, use
other constructs like a `break` with a helper function instead.

```py
def find_target_value(data, target):
    for item in data:
        if item == target:
            return True
    return False
```

## Prevent Repetition with Assignment Expressions

Use the walrus operator (`:=`) to avoid repeating expressions
and improve code readability.

```py
def count_odd_numbers(numbers):
    count = 0
    for n in numbers:
        if (remainder := n % 2) == 1:
            count += 1
    return count
```

# Chapter 2: Lists and Dictionaries

## Know How to Slice Sequences

Slicing is a mechanism in Python that allows you to extract a portion of a
sequence. Slicing works with strings, lists, and other sequence types.

`sequence[start:end]`, where `start` is inclusive and `end` is exclusive.

```py
colors = ['red', 'green', 'blue', 'yellow', 'black']
colors[1:3]  # Output: ['green', 'blue']
```

## Avoid Striding and Slicing in a Single Expression

Striding allows you to access elements from a sequence at regular intervals.
Though it's possible to use striding and slicing together, it's better to avoid
combining them as it may lead to difficult-to-read code.

```py
numbers = list(range(10))
even_numbers = numbers[::2]  # Output: [0, 2, 4, 6, 8]
```

## Prefer Catch-All Unpacking Over Slicing

Catch-all unpacking, introduced in Python 3, improves the readability of code
when extracting optional elements from a sequence.

```py
first, *middle, last = [1, 2, 3, 4, 5]
middle  # Output: [2, 3, 4]
```

## Sort by Complex Criteria Using the key Parameter

The `key` parameter allows you to sort a list according to the value returned by
a custom function or a lambda function.

```py
students = [("Alice", 90), ("Bob", 85), ("Eve", 92)]
students.sort(key=lambda x: x[1])  # Output: [("Bob", 85), ("Alice", 90), ("Eve", 92)]
```

## Be Cautious When Relying on dict Insertion Ordering

As of Python 3.7, dictionaries retain their insertion order. However, relying
too much on this behavior can lead to brittle code that can break in earlier
versions of Python.

```py
capitals = {"USA": "Washington, D.C.", "France": "Paris", "Japan": "Tokyo"}
# This code is OK in Python 3.7+, but not guaranteed to work in earlier versions.
for _, city in capitals.items():
    print(city)
```

## Prefer get Over in and KeyError to Handle Missing Dictionary Keys

Using `get` method to access dictionary keys ensures that missing keys don't
raise an exception. It's a cleaner and more readable approach compared to using
`in` operator or catching `KeyError`.

```py
user_preferences = {"color": "blue", "font_size": 12}
theme_color = user_preferences.get("color", "default")  # Output: "blue"
```

## Prefer defaultdict Over setdefault to Handle Missing Items in Internal State

`defaultdict` from the `collections` module simplifies the handling of missing
keys by automatically initializing the values with a default factory function.

```py
from collections import defaultdict
word_counts = defaultdict(int)
word_counts["hello"] += 1  # Output: 1
```

## Know How to Construct Key-Dependent Default Values with __missing__

When you need to handle missing keys with default values that depend on the key,
you can create a custom dictionary class that implements the `__missing__` method.

```py
class CustomDict(dict):
    def __missing__(self, key):
        value = len(key)
        self[key] = value
        return value

d = CustomDict()
value = d["python"]  # Output: 6
```

# Chapter 3: Functions

## Never Unpack More Than Three Variables When Functions Return Multiple Values

When returning multiple values from a function, use at most three variables to
unpack the values. This improves readability and reduces the chances of errors.

```py
def get_name_and_age():
    return "Alice", 30

name, age = get_name_and_age()
```

## Prefer Raising Exceptions to Returning None

Raise exceptions instead of returning `None` in case of errors, as this is more
explicit and makes error handling easier.

```py
def divide(a, b):
    if b == 0:
        raise ValueError("Division by zero")
    return a / b
```

## Know How Closures Interact with Variable Scope

Understand how closures can capture variables from enclosing scopes and how to
modify them using `nonlocal` keyword.

```py
def outer_function(x):
    def inner_function(y):
        nonlocal x
        x += y
        return x
    return inner_function

closure = outer_function(10)
print(closure(5))  # 15
```

## Reduce Visual Noise with Variable Positional Arguments

Use variable **positional arguments** (`*args`) to accept an arbitrary number of
**positional arguments**, reducing visual noise and improving readability.

```py
def sum_numbers(*args):
    return sum(args)

result = sum_numbers(1, 2, 3, 4, 5)
```

## Provide Optional Behavior with Keyword Arguments

Use **keyword arguments** to provide optional behavior and make your functions more
flexible and expressive.

```py
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

message = greet("John", greeting="Hi")
```

## Use None and Docstrings to Specify Dynamic Default Arguments

Use `None` and docstrings to clearly indicate that a default argument may be
dynamic (e.g., generated during function execution).

```py
def get_current_time(timezone=None):
    """
    Returns the current time in the specified timezone. 
    If no timezone is provided, UTC is used.
    """
    if timezone is None:
        timezone = "UTC"
    # Get current time in the specified timezone
```

## Enforce Clarity with Keyword-Only and Position-Only Arguments

Use keyword-only arguments (`*,`) or position-only arguments (`/`) to enforce
clarity and avoid confusion when calling functions.

```py
def safe_divide(a, *, b):
    if b == 0:
        raise ValueError("Division by zero")
    return a / b

result = safe_divide(10, b=2)
```

## Define Function Decorators with functools.wraps

# Chapter 4: Comprehensions and Generators

## Use Comprehensions Instead of map and filter

Comprehensions provide a more concise and readable way to apply operations or
filter elements from a collection.

```py
squares = [x**2 for x in range(1, 11)]
```

## Avoid More Than Two Control Subexpressions in Comprehensions

Limit control subexpressions in comprehensions to two (conditions and loops) for
better readability.

```py
even_squares = [x**2 for x in range(1, 11) if x % 2 == 0]
```

## Avoid Repeated Work in Comprehensions by Using Assignment Expressions

Use assignment expressions (`:=`) to avoid repeated work and improve performance
in comprehensions.

```py
squares = [(x:=i**2) for i in range(1, 11)]
```

## Consider Generators Instead of Returning Lists

Use generator functions to create iterators that yield elements on-the-fly,
reducing memory consumption for large sequences.

```py
def even_numbers(n):
    for i in range(n):
        if i % 2 == 0:
            yield i

evens = even_numbers(10)
```

## Be Defensive When Iterating Over Arguments

Be cautious when iterating over arguments that could be iterables or single
values. Use a helper function to ensure correct behavior.

```py
def flatten(items):
    for item in items:
        if isinstance(item, (list, tuple)):
            yield from item
        else:
            yield item

result = list(flatten([1, 2, [3, 4], (5, 6)]))
```

## Consider Generator Expressions for Large List Comprehensions

Use generator expressions (similar to list comprehensions) for large
collections, as they save memory by yielding elements on-the-fly.

```py
cubes = (x**3 for x in range(1, 1000000))  # Generator expression
```

## Compose Multiple Generators with yield from

Use `yield from` to compose multiple generators together, simplifying the code
and improving readability.

```py
def odds_and_evens(n):
    def odds(n): 
        for i in range(1, n, 2):
            yield i
    def evens(n): 
        for i in range(0, n, 2):
            yield i
    
    yield from odds(n)
    yield from evens(n)

result = list(odds_and_evens(10))
```

## Avoid Injecting Data into Generators with send

Use `send` method cautiously to interact with generator expressions, as it is
less intuitive and prone to errors.

```py
def generator_with_send():
    message = yield "Ready"
    while message:
        message = yield f"Received: {message}"

g = generator_with_send()
print(g.send(None))  # "Ready"
```

## Avoid Causing State Transitions in Generators with throw

Use `throw` method cautiously to raise exceptions within generator expressions,
as it can be difficult to reason about state transitions.

```py
def generator_with_throw():
    yield 1
    yield 2
    yield 3

g = generator_with_throw()
next(g)
g.throw(RuntimeError("An error occurred"))
```

## Consider itertools for Working with Iterators and Generators

Use `itertools` module to work with iterators and generators effectively, as it
provides many useful functions and recipes.

```py
import itertools

result = list(itertools.islice(itertools.count(10), 5))  # [10, 11, 12, 13, 14]
```

# Chapter 5: Classes and Interfaces

## Compose Classes Instead of Nesting Many Levels of Built-in Types

It is better to use classes to encapsulate complex structures instead of nesting
dictionaries, lists, and sets within each other.

```py
class AddressBookEntry:
    def __init__(self, name, email):
        self.name = name
        self.email = email

class AddressBook:
    def __init__(self, entries=None):
        self.entries = entries if entries else []

    def add_entry(self, entry):
        self.entries.append(entry)
```

## Accept Functions Instead of Classes for Simple Interfaces

Use simple functions as parameters instead of creating classes for small
interfaces.

```py
def sort_by_length(words):
    return sorted(words, key=len)

words = ["apple", "banana", "cherry"]
sorted_words = sort_by_length(words)
```

## Use @classmethod Polymorphism to Construct Objects Generically

With `@classmethod`, you can create alternative constructors within a class
hierarchy. This allows for more generic and flexible code.

```py
class Animal:
    
    @classmethod
    def from_string(cls, string):
        return cls(string)

class Dog(Animal):
    def __init__(self, name):
        self.name = name

dog = Dog.from_string("Snoopy")
```

## Initialize Parent Classes with super

Use `super()` in the child class when overriding the `__init__` method to ensure
that the parent class's `__init__` method is called.

```py
class Animal:
    def __init__(self, name):
        self.name = name

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)
        self.breed = breed

dog = Dog("Snoopy", "Beagle")
```

## Consider Composing Functionality with Mix-in Classes

Use mix-in classes to add reusable functionality to a class hierarchy.

```py
class SerializableMixin:
    def to_json(self):
        pass

class Dog(SerializableMixin):
    def __init__(self, name):
        self.name = name
        
dog = Dog("Snoopy")
dog.to_json()
```

## Prefer Public Attributes Over Private Ones

In Python, use public attributes (`self.attribute`) instead of private ones
(`self.__attribute`) when possible for simplicity.

## Inherit from collections.abc for Custom Container Types

Inherit from appropriate classes in `collections.abc` to create custom container
types.

```py
from collections.abc import Sequence

class MyList(Sequence):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
```

# Chapter 6: Metaclasses and Attributes

## Use Plain Attributes Instead of Setter and Getter Methods

In Python, use plain attributes instead of creating getter and setter methods
for simplicity.

```py
class Dog:
    def __init__(self, name):
        self.name = name
```

## Consider @property Instead of Refactoring Attributes

Use the `@property` decorator to create read-only properties or to enforce
validation rules without changing the API.

```py
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @property
    def diameter(self):
        return self._radius * 2
```

## Use Descriptors for Reusable @property Methods

Use descriptors to create reusable `@property` methods.

```py
class PositiveNumber:
    def __get__(self, instance, owner):
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if value <= 0:
            raise ValueError("Value must be positive.")
        instance.__dict__[self.name] = value

    def __set_name__(self, owner, name):
        self.name = name

class Circle:
    radius = PositiveNumber()

    def __init__(self, radius):
        self.radius = radius

    @property
    def diameter(self):
        return self.radius * 2
```

## Use __getattr__, __getattribute__, and __setattr__ for Lazy Attributes

Use these methods to control attribute access and implement lazy attributes.

```py
class LazyAttributes:
    def __getattr__(self, name):
        if name not in self.__dict__:
            value = f"Lazy attribute '{name}'"
            setattr(self, name, value)
        return self.__dict__[name]

lazy = LazyAttributes()
print(lazy.lazy_attr)  # "Lazy attribute 'lazy_attr'"
```

## Validate Subclasses with __init_subclass__

Use `__init_subclass__` to customize subclass initialization and validation.

```py
class Animal:
    def __init_subclass__(cls, **kwargs):
        if not hasattr(cls, "sound"):
            raise NotImplementedError("Subclass must have 'sound' attribute")

class Dog(Animal):
    sound = "woof"

dog = Dog()
print(dog.sound)  # "woof"
```

## Register Class Existence with __init_subclass__

Create a registry of classes by using `__init_subclass__`.

```py
class PluginBase:
    subclasses = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses.append(cls)

class Plugin1(PluginBase):
    pass

class Plugin2(PluginBase):
    pass

print(PluginBase.subclasses)  # [<class '__main__.Plugin1'>, <class '__main__.Plugin2'>]
```

## Annotate Class Attributes with __set_name__

Use `__set_name__` to store the attribute name.

```py
class AnnotatedAttribute:
    def __set_name__(self, owner, name):
        self.name = name

class MyClass:
    attr = AnnotatedAttribute()

print(MyClass.attr.name)  # "attr"
```

## Prefer Class Decorators Over Metaclasses for Composable Class Extensions

Use class decorators instead of metaclasses for extending classes in a
composable way.

```py
def class_decorator(cls):
    def wrapped():
        print("Before creating instance")
        instance = cls()
        print("After creating instance")
        return instance

    return wrapped

@class_decorator
class MyClass:
    pass

obj = MyClass()  # "Before creating instance" and "After creating instance"
```

# Chapter 7: Concurrency and Parallelism

## Use subprocess to Manage Child Processes
## Use Threads for Blocking I/O, Avoid for Parallelism
## Use Lock to Prevent Data Races in Threads
## Use Queue to Coordinate Work Between Threads
## Know How to Recognize When Concurrency Is Necessary
## Avoid Creating New Thread Instances for On-demand Fan-out
## Understand How Using Queue for Concurrency Requires Refactoring
## Consider ThreadPoolExecutor When Threads Are Necessary for Concurrency
## Achieve Highly Concurrent I/O with Coroutines
## Know How to Port Threaded I/O to asyncio
## Mix Threads and Coroutines to Ease the Transition to asyncio
## Avoid Blocking the asyncio Event Loop to Maximize Responsiveness
## Consider concurrent.futures for True Parallelism

# Chapter 8: Robustness and Performance

## Take Advantage of Each Block in try/except/else/finally
## Consider contextlib and with Statements for Reusable try/finally Behavior
## Use datetime Instead of time for Local Clocks
## Make pickle Reliable with copyreg
## Use decimal When Precision Is Paramount
## Profile Before Optimizing
## Prefer deque for Producer–Consumer Queues
## Consider Searching Sorted Sequences with bisect
## Know How to Use heapq for Priority Queues
## Consider memoryview and bytearray for Zero-Copy Interactions with bytes

# Chapter 9: Testing and Debugging

## Use repr Strings for Debugging Output
## Verify Related Behaviors in TestCase Subclasses
## Isolate Tests from Each Other with setUp, tearDown, setUpModule, and tearDownModule
## Use Mocks to Test Code with Complex Dependencies
## Encapsulate Dependencies to Facilitate Mocking and Testing
## Consider Interactive Debugging with pdb
## Use tracemalloc to Understand Memory Usage and Leaks

# Chapter 10: Collaboration

## Know Where to Find Community-Built Modules
## Use Virtual Environments for Isolated and Reproducible Dependencies
## Write Docstrings for Every Function, Class, and Module
## Use Packages to Organize Modules and Provide Stable APIs
## Consider Module-Scoped Code to Configure Deployment Environments
## Define a Root Exception to Insulate Callers from APIs
## Know How to Break Circular Dependencies
## Consider warnings to Refactor and Migrate Usage
## Consider Static Analysis via typing to Obviate Bugs
