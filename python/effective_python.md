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
## Follow the PEP 8 Style Guide
## Know the Differences Between bytes and str
## Prefer Interpolated F-Strings Over C-style Format Strings and str.format
## Write Helper Functions Instead of Complex Expressions
## Prefer Multiple Assignment Unpacking Over Indexing
## Prefer enumerate Over range
## Use zip to Process Iterators in Parallel
## Avoid else Blocks After for and while Loops
## Prevent Repetition with Assignment Expressions

# Chapter 2: Lists and Dictionaries

## Know How to Slice Sequences
## Avoid Striding and Slicing in a Single Expression
## Prefer Catch-All Unpacking Over Slicing
## Sort by Complex Criteria Using the key Parameter
## Be Cautious When Relying on dict Insertion Ordering
## Prefer get Over in and KeyError to Handle Missing Dictionary Keys
## Prefer defaultdict Over setdefault to Handle Missing Items in Internal State
## Know How to Construct Key-Dependent Default Values with __missing__

# Chapter 3: Functions

## Never Unpack More Than Three Variables When Functions Return Multiple Values
## Prefer Raising Exceptions to Returning None
## Know How Closures Interact with Variable Scope
## Reduce Visual Noise with Variable Positional Arguments
## Provide Optional Behavior with Keyword Arguments
## Use None and Docstrings to Specify Dynamic Default Arguments
## Enforce Clarity with Keyword-Only and Position-Only Arguments
## Define Function Decorators with functools.wraps

# Chapter 4: Comprehensions and Generators

## Use Comprehensions Instead of map and filter
## Avoid More Than Two Control Subexpressions in Comprehensions
## Avoid Repeated Work in Comprehensions by Using Assignment Expressions
## Consider Generators Instead of Returning Lists
## Be Defensive When Iterating Over Arguments
## Consider Generator Expressions for Large List Comprehensions
## Compose Multiple Generators with yield from
## Avoid Injecting Data into Generators with send
## Avoid Causing State Transitions in Generators with throw
## Consider itertools for Working with Iterators and Generators

# Chapter 5: Classes and Interfaces

## Compose Classes Instead of Nesting Many Levels of Built-in Types
## Accept Functions Instead of Classes for Simple Interfaces
## Use @classmethod Polymorphism to Construct Objects Generically
## Initialize Parent Classes with super
## Consider Composing Functionality with Mix-in Classes
## Prefer Public Attributes Over Private Ones
## Inherit from collections.abc for Custom Container Types

# Chapter 6: Metaclasses and Attributes

## Use Plain Attributes Instead of Setter and Getter Methods
## Consider @property Instead of Refactoring Attributes
## Use Descriptors for Reusable @property Methods
## Use __getattr__, __getattribute__, and __setattr__ for Lazy Attributes
## Validate Subclasses with __init_subclass__
## Register Class Existence with __init_subclass__
## Annotate Class Attributes with __set_name__
## Prefer Class Decorators Over Metaclasses for Composable Class Extensions

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
