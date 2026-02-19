- [Abstract](#abstract)
- [References](#references)
- [Materials](#materials)
- [Basic](#basic)
  - [Features](#features)
  - [Install](#install)
  - [Tools](#tools)
  - [Build and Run](#build-and-run)
  - [Hello World](#hello-world)
  - [Reserved Words](#reserved-words)
  - [Data Types](#data-types)
  - [Variables and Mutability](#variables-and-mutability)
  - [String Types](#string-types)
  - [String Conversions](#string-conversions)
  - [String Loops](#string-loops)
  - [Formatted Print](#formatted-print)
  - [Control Flows](#control-flows)
  - [Pattern Matching](#pattern-matching)
  - [Collections](#collections)
    - [Vec](#vec)
    - [HashMap](#hashmap)
    - [HashSet](#hashset)
    - [BTreeMap / BTreeSet](#btreemap--btreeset)
    - [VecDeque](#vecdeque)
  - [Collection Conversions](#collection-conversions)
  - [Sort](#sort)
  - [Search](#search)
  - [Multi Dimensional Array](#multi-dimensional-array)
  - [Tuple](#tuple)
  - [Array](#array)
  - [Struct](#struct)
  - [Enum](#enum)
  - [Functions](#functions)
  - [Closures](#closures)
  - [Generic](#generic)
  - [Module System](#module-system)
  - [Crate](#crate)
  - [Attributes](#attributes)
  - [Environment Variables](#environment-variables)
- [Core Concepts](#core-concepts)
  - [Ownership and Borrowing](#ownership-and-borrowing)
  - [Copy vs Clone](#copy-vs-clone)
  - [Lifetime](#lifetime)
  - [References and Pointers](#references-and-pointers)
  - [Trait](#trait)
  - [Box and Smart Pointers](#box-and-smart-pointers)
  - [Error Handling](#error-handling)
- [Advanced](#advanced)
  - [Dispatch](#dispatch)
  - [Macros](#macros)
  - [Unsafe Rust](#unsafe-rust)
  - [Concurrency](#concurrency)
  - [Async Programming](#async-programming)
  - [Tokio](#tokio)
  - [Memory Management](#memory-management)
  - [Performance Optimization](#performance-optimization)
- [Style Guide](#style-guide)
- [Refactoring](#refactoring)
- [Effective Rust](#effective-rust)
- [Rust Design Patterns](#rust-design-patterns)
- [Rust Architecture](#rust-architecture)

----

# Abstract

Rust is a systems programming language focused on safety, speed, and concurrency. It achieves memory safety without a garbage collector through its unique ownership system. This document provides a comprehensive guide to the features, patterns, and best practices of the Rust programming language.

**Korean version**: See [README-kr.md](README-kr.md) for the Korean documentation.

# References

* [Command-Line Rust | oreilly](https://www.oreilly.com/library/view/command-line-rust/9781098109424/)
  * Clone-codes command-line tools like `echo, cat, head, wc, uniq, find, cut, grep, comm, tail, fortune, cal, ls`.
  * [src](https://github.com/kyclark/command-line-rust)
* [Comprehensive Rust](https://google.github.io/comprehensive-rust/welcome.html)
  * [Korean](https://google.github.io/comprehensive-rust/ko/welcome.html)
* [Awesome Rust @ github](https://github.com/rust-unofficial/awesome-rust)
* [Rust](https://www.rust-lang.org/learn)
  * [Crate std reference](https://doc.rust-lang.org/stable/std/index.html)
* [The Rust Programming Language](https://doc.rust-lang.org/book/index.html)
  * [Korean](https://rinthel.github.io/rust-lang-book-ko/)
* [Learn Rust by Building Real Applications](https://www.udemy.com/course/rust-fundamentals/)
  * [src](https://github.com/gavadinov/Learn-Rust-by-Building-Real-Applications)

# Materials

* [Rust Language Tutorial (Korean)](http://sarojaba.github.io/rust-doc-korean/doc/tutorial.html)
* [Easy Rust](https://dhghomon.github.io/easy_rust/Chapter_0.html)
  * [video](https://www.youtube.com/playlist?list=PLfllocyHVgsRwLkTAhG0E-2QxCf-ozBkk)
* [Asynchronous Programming in Rust](https://rust-lang.github.io/async-book/)
* [Rust By Example (Korean)](https://doc.rust-kr.org/rust-by-example-ko/)
* [The Rust Programming Language Official Guide (Korean book)](http://www.yes24.com/Product/Goods/83075894)

# Basic

## Features

Rust is a language defined by **what it doesn't have**:

| What's Missing | What's There Instead | Why? |
|----------------|---------------------|------|
| Exception | `Result<T, E>`, `Option<T>` | Forces you to handle errors explicitly |
| OOP (class inheritance) | `struct` + `trait` | Composition over inheritance |
| Garbage Collector | Ownership system | Memory deallocation decided at compile time |
| Dangling Reference | Lifetime checking | Compiler prevents dead references at the source |

> Coming from TypeScript/Java, the most shocking point: **when you pass a value to another variable, the original variable can no longer be used** (ownership move).

## Install

```bash
# Install rust on macOS
$ brew install rust

# Install rustup, rustc, cargo on macOS
$ curl https://sh.rustup.rs -sSf | sh
```

## Tools

* **rustc** : Rust compiler
* **cargo** : Rust package manager and build tool
* **rustup** : Rust toolchain (rustc, cargo) manager

## Build and Run

```bash
# Build and run with rustc
$ rustc a.rs -o a.out
$ ./a.out

# Build and run with cargo
$ cargo new a
$ cd a
$ cargo run

# Cargo commands
# Build current directory
$ cargo build
# Run current directory
$ cargo run
# Test current directory
$ cargo test
# Install a crate
$ cargo install cargo-expand
# Use a crate
$ cargo expand
```

## Hello World

```rs
// a.rs
fn main() {
     println!("Hello World")
}
```

## Reserved Words

```rs
as - perform primitive type casting, disambiguate specific trait items, rename items in use and extern crate statements
async - return a Future instead of blocking the current thread
await - suspend execution until the result of a Future is ready
break - exit a loop immediately
const - define constant items or constant raw pointers
continue - continue to the next loop iteration
crate - link an external crate or a macro variable representing the crate where the macro is defined
dyn - dynamic dispatch to a trait object
else - fallback for if and if let control flow constructs
enum - define an enumeration
extern - link an external crate, function, or variable
false - Boolean false literal
fn - define a function or function pointer type
for - loop over items from an iterator, implement a trait, or specify a higher-ranked lifetime
if - branch based on the result of a conditional expression
impl - implement inherent or trait functionality
in - part of for loop syntax
let - bind a variable
loop - loop unconditionally
match - match a value to patterns
mod - define a module
move - make a closure take ownership of all its captures
mut - denote mutability in references, raw pointers, or pattern bindings
pub - denote public visibility in struct fields, impl blocks, or modules
ref - bind by reference
return - return from a function
Self - a type alias for the type being defined or implemented
self - method subject or current module
static - global variable or lifetime lasting the entire program execution
struct - define a structure
super - parent module of the current module
trait - define a trait
true - Boolean true literal
type - define a type alias or associated type
union - define a union; only a keyword when used in a union declaration
unsafe - denote unsafe code, functions, traits, or implementations
use - bring symbols into scope
where - denote clauses that constrain a type
while - loop conditionally based on the result of an expression
```

## Data Types

Rust is a statically typed language. The main data types are:

### Scalar Types

* **Integer**: `i8`, `i16`, `i32`, `i64`, `i128`, `isize` (signed)
* **Unsigned integer**: `u8`, `u16`, `u32`, `u64`, `u128`, `usize`
* **Floating point**: `f32`, `f64`
* **Boolean**: `bool` (true, false)
* **Character**: `char` (Unicode scalar value)

### Compound Types

* **Tuple**: `(i32, f64, u8)`
* **Array**: `[i32; 5]` (fixed size)

## Variables and Mutability

Rust is **immutable by default**. Think of it as TypeScript's `const` being the default.

```rs
let x = 5;          // Immutable (default) - cannot change the value
let mut y = 10;     // Must add mut to allow changes
y = 15;             // OK

const MAX_POINTS: u32 = 100_000;  // Constant - determined at compile time, type required

// Shadowing - create a new variable with the same name (can even change type)
let x = 5;
let x = x + 1;      // x = 6, new variable
let x = "hello";    // Can even change the type!
```

| | `let` | `let mut` | `const` |
|---|---|---|---|
| Mutable | No | Yes | No |
| Shadowable | Yes | Yes | No |
| Type inference | Yes | Yes | No (explicit required) |

## String Types

One of the most confusing parts of Rust. Just remember these two:

| | `String` | `&str` |
|---|---|---|
| Ownership | **Owned** (heap allocated) | **Borrowed** (reference) |
| Mutable | Yes if `let mut s` | No, always immutable |
| Creation | `String::from("hello")` | `"hello"` (literal) |
| Function param | When ownership is needed | **Use this most of the time** |

```rs
let s1: String = String::from("hello");  // Owned on the heap
let s2: &str = "world";                  // Reference on the stack
let s3: &str = &s1[0..2];               // Slice of String -> &str
```

> **Use `&str` for function parameters.** It can accept both `String` and `&str`.

## String Conversions

```rs
// &str -> String
let s = "hello".to_string();
let s = String::from("hello");

// String -> &str
let s = String::from("hello");
let slice: &str = &s;

// Integer -> String
let num = 42;
let s = num.to_string();

// String -> Integer
let s = "42";
let num: i32 = s.parse().unwrap();
```

## String Loops

```rs
// Iterate over characters
for c in "hello".chars() {
    println!("{}", c);
}

// Iterate over bytes
for b in "hello".bytes() {
    println!("{}", b);
}

// With enumerate
for (i, c) in "hello".chars().enumerate() {
    println!("{}: {}", i, c);
}
```

## Formatted Print

```rs
println!("Hello, {}!", "world");
println!("{0} {1} {0}", "a", "b");  // a b a
println!("{name} {age}", name="John", age=30);
println!("{:?}", vec![1, 2, 3]);  // Debug output
println!("{:#?}", vec![1, 2, 3]);  // Pretty Debug
println!("{:b}", 10);  // Binary: 1010
println!("{:x}", 255);  // Hex: ff
println!("{:.2}", 3.14159);  // 3.14
```

## Control Flows

A unique aspect of Rust control flow: **`if` is an expression** that returns a value (instead of having a ternary operator).

```rs
// if expression - returns a value
let x = if number < 10 { "smaller" } else { "bigger" };

// 4 types of loops
for i in 0..5 { println!("{}", i); }       // Range iteration
for item in vec { println!("{}", item); }  // Collection iteration
while count < 10 { count += 1; }           // Conditional loop
loop { break; }                            // Infinite loop (exit with break)

// match - must handle all cases (exhaustive)
match number {
    1 => println!("one"),
    2 | 3 => println!("two or three"),   // OR pattern
    4..=10 => println!("four to ten"),   // Range pattern
    _ => println!("something else"),     // Catch-all (default)
}

// if let - concise alternative to match when handling just one pattern from Option or enum
if let Some(value) = optional_value {
    println!("{}", value);
}
```

| Loop | Use Case | TypeScript Equivalent |
|------|----------|----------------------|
| `for i in 0..5` | Range iteration | `for (let i=0; i<5; i++)` |
| `for item in vec` | Collection iteration | `for (const item of arr)` |
| `while cond` | Conditional loop | `while (cond)` |
| `loop` | Infinite loop (exit with break) | `while (true)` |

## Pattern Matching

`match` goes beyond simple value comparison -- it can **destructure structures**. In TypeScript you'd use `never` for exhaustive checking, but in Rust it's the **default behavior**.

```rs
enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter,
}

fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => 1,
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter => 25,
    }
    // All variants must be handled! Missing even one causes a compile error
}
```

## Collections

Rust's 3 main collections:

| Collection | Type | TypeScript Equivalent | Use Case |
|------------|------|----------------------|----------|
| **Vector** | `Vec<T>` | `Array` | Dynamic array |
| **String** | `String` | `string` | UTF-8 string |
| **HashMap** | `HashMap<K, V>` | `Map` | Key-value store |

### Vec

```rs
let mut v: Vec<i32> = Vec::new();
v.push(1);
v.push(2);
let vec = vec![1, 2, 3];       // Initialize with macro

// Access
let first = &vec[0];            // Can panic
let first = vec.get(0);         // Returns Option (safe)

// Iteration
for val in &vec {
    println!("{}", val);
}

// Useful methods
vec.len();                       // Length
vec.is_empty();                  // Check if empty
vec.contains(&2);                // Check if contains
vec.iter().position(|&x| x == 2); // Search for index
vec.retain(|&x| x > 1);         // Keep only matching elements
vec.dedup();                     // Remove consecutive duplicates

// Slicing
let slice = &vec[1..3];         // &[i32]
```

### HashMap

```rs
use std::collections::HashMap;

let mut map = HashMap::new();
map.insert("key", 1);

// Access
let val = map.get("key");          // Option<&V>
let val = map["key"];               // Can panic

// Insert if not present
map.entry("key").or_insert(0);
*map.entry("key").or_insert(0) += 1;  // Counter pattern

// Iteration
for (key, val) in &map {
    println!("{}: {}", key, val);
}

// Useful methods
map.len();
map.contains_key("key");
map.remove("key");
```

### HashSet

```rs
use std::collections::HashSet;

let mut set = HashSet::new();
set.insert(1);
set.insert(2);
set.insert(2);                     // Duplicate ignored
println!("{}", set.len());          // 2

set.contains(&1);                   // true
set.remove(&1);

// Set operations
let a: HashSet<_> = [1, 2, 3].iter().collect();
let b: HashSet<_> = [2, 3, 4].iter().collect();
let union: HashSet<_> = a.union(&b).collect();
let intersection: HashSet<_> = a.intersection(&b).collect();
let diff: HashSet<_> = a.difference(&b).collect();
```

### BTreeMap / BTreeSet

Collections that maintain sorted order. Equivalent to Java's `TreeMap`/`TreeSet`.

```rs
use std::collections::BTreeMap;

let mut map = BTreeMap::new();
map.insert(3, "c");
map.insert(1, "a");
map.insert(2, "b");

// Always iterates in key order
for (k, v) in &map {
    println!("{}: {}", k, v);  // 1: a, 2: b, 3: c
}
```

### VecDeque

A double-ended queue (deque) that supports adding/removing from both ends. Equivalent to Java's `ArrayDeque`.

```rs
use std::collections::VecDeque;

let mut deque = VecDeque::new();
deque.push_back(1);      // Add to back
deque.push_front(0);     // Add to front
deque.pop_back();        // Remove from back
deque.pop_front();       // Remove from front
```

| Rust Collection | Java Equivalent | TypeScript Equivalent | Order Guarantee |
|-----------------|----------------|----------------------|-----------------|
| `Vec<T>` | `ArrayList` | `Array` | Insertion order |
| `HashMap<K,V>` | `HashMap` | `Map` | None |
| `HashSet<T>` | `HashSet` | `Set` | None |
| `BTreeMap<K,V>` | `TreeMap` | - | Key sorted |
| `BTreeSet<T>` | `TreeSet` | - | Value sorted |
| `VecDeque<T>` | `ArrayDeque` | - | Insertion order |

## Collection Conversions

```rs
// Vec -> HashSet
let vec = vec![1, 2, 2, 3];
let set: HashSet<_> = vec.into_iter().collect();

// HashSet -> Vec
let set: HashSet<i32> = [1, 2, 3].iter().cloned().collect();
let vec: Vec<_> = set.into_iter().collect();

// Vec<&str> -> Vec<String>
let strs = vec!["hello", "world"];
let strings: Vec<String> = strs.iter().map(|s| s.to_string()).collect();

// Iterator -> Vec
let vec: Vec<i32> = (0..5).collect();                    // [0, 1, 2, 3, 4]
let vec: Vec<i32> = (0..5).filter(|x| x % 2 == 0).collect(); // [0, 2, 4]
```

> `collect()` is Rust's universal converter. Most collection conversions follow the pattern `.iter()` + transformation + `.collect()`.

## Sort

```rs
// Sort Vec
let mut vec = vec![3, 1, 4, 1, 5];
vec.sort();                          // [1, 1, 3, 4, 5] - ascending
vec.sort_by(|a, b| b.cmp(a));       // [5, 4, 3, 1, 1] - descending

// Sort by key
let mut people = vec![("Bob", 30), ("Alice", 25), ("Charlie", 35)];
people.sort_by_key(|p| p.1);        // Ascending by age

// Stable sort vs unstable sort
vec.sort();             // Stable sort (preserves order of equal elements)
vec.sort_unstable();    // Unstable sort (faster)

// Sorting f64 (f64 does not implement Ord due to NaN)
let mut floats = vec![3.1, 1.2, 4.5];
floats.sort_by(|a, b| a.partial_cmp(b).unwrap());
```

> `sort()` **mutates the original**. To preserve the original: `let sorted = { let mut v = vec.clone(); v.sort(); v };`

## Search

```rs
let vec = vec![1, 2, 3, 4, 5];

// Linear search
vec.contains(&3);                       // true
vec.iter().find(|&&x| x > 3);          // Some(&4)
vec.iter().position(|&x| x > 3);       // Some(3) - index

// Binary search (on sorted array)
let sorted = vec![1, 2, 3, 4, 5];
sorted.binary_search(&3);              // Ok(2)  - index
sorted.binary_search(&6);              // Err(5) - insertion point
```

| Method | Return Type | Use Case |
|--------|-------------|----------|
| `contains(&val)` | `bool` | Existence check only |
| `iter().find(fn)` | `Option<&T>` | Search by condition |
| `iter().position(fn)` | `Option<usize>` | Index by condition |
| `binary_search(&val)` | `Result<usize, usize>` | Binary search on sorted array |

## Multi Dimensional Array

```rs
// Initialize 2D Vec
let rows = 3;
let cols = 4;
let grid = vec![vec![0; cols]; rows];  // 3x4 array filled with 0

// Iteration
for i in 0..rows {
    for j in 0..cols {
        print!("{} ", grid[i][j]);
    }
    println!();
}

// Initialize 2D with values
let matrix = vec![
    vec![1, 2, 3],
    vec![4, 5, 6],
];
```

> Rust's `vec![vec![0; cols]; rows]` is like TypeScript's `Array.from()` pattern -- each row is an **independent Vec** (no shared reference issues).

## Tuple

```rs
let tup: (i32, f64, u8) = (500, 6.4, 1);
let (x, y, z) = tup;  // Destructuring
let first = tup.0;    // Index access
```

## Array

```rs
// [u8] is a u8 array of unknown size
fn foo(a: [u8]) {}

// [u8; 5] is a u8 array of size 5
fn foo(a: [u8; 5]) {}

// &[u8] is a reference to a u8 array of unknown size
fn foo(a: &[u8]) {}

// An array reference is a slice
let a = [1, 2, 3, 4, 5];
foo(&a[1..3]);

// 1024-byte array initialized with value 0
let mut buf = [0; 1024];
```

## Struct

A structure that **groups** data together (AND). Defined with `struct` and methods are implemented with `impl`. **Instance methods** have `self` as the first parameter. **Associated functions** (static functions) do not have `self` as the first parameter.

```rs
pub struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    // Associated function
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    // Instance method
    pub fn area(&self) -> u32 {
        self.width * self.height
    }
}

fn main() {
    let rect = Rectangle::new(10, 20);
    println!("Area: {}", rect.area());
}
```

## Enum

Represents **one** of several possibilities (OR). Members of an **enum** are called **variants**. Unlike TypeScript's `enum`, Rust's `enum` can **hold different data per variant**, making it much more powerful.

| | `struct` | `enum` |
|---|---|---|
| Role | **Group** data (AND) | **One of** several possibilities (OR) |
| TypeScript equivalent | `interface` | `\| (union type)` |
| Methods | Add with `impl` | Add with `impl` |

```rs
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}

impl Message {
    fn process(&self) {
        match self {
            Message::Quit => println!("Quit"),
            Message::Move { x, y } => println!("Move to {}, {}", x, y),
            Message::Write(text) => println!("Write: {}", text),
            Message::ChangeColor(r, g, b) => println!("Color: {}, {}, {}", r, g, b),
        }
    }
}
```

Declaring `use Message::*;` lets you skip typing `Message::`.

## Functions

```rs
fn add(x: i32, y: i32) -> i32 {
    x + y       // Last expression is the return value (no semicolon!)
}

// Generic function
fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    for item in list {
        if item > largest {
            largest = item;
        }
    }
    largest
}
```

## Closures

| | Function `fn` | Closure `\|\|` |
|---|---|---|
| Type annotation | Required | Can be inferred |
| Capture environment | No | Yes, captures surrounding variables |
| `move` | N/A | Can move ownership |

```rs
// Closure syntax
let add_one = |x| x + 1;
let add_typed = |x: i32, y: i32| -> i32 { x + y };

// Environment capture - the key difference from functions
let x = 10;
let equal_to_x = |z| z == x;   // Captures x
println!("{}", equal_to_x(10)); // true

// move - moves ownership into the closure (required for threads)
let s = String::from("hello");
let closure = move || println!("{}", s);
// println!("{}", s);  // Error: s has been moved into the closure
```

## Generic

```rs
struct Point<T> {
    x: T,
    y: T,
}

impl<T> Point<T> {
    fn x(&self) -> &T {
        &self.x
    }
}

// Multiple generic types
struct Point2<T, U> {
    x: T,
    y: U,
}
```

## Module System

A module is a logical unit of code. It groups related code together. Use `mod` to define modules and `use` to import items from other modules. Module functions are private by default. Use `pub` to make them public.

```rs
mod sound {
    pub mod instrument {
        pub fn clarinet() {
            println!("clarinet");
        }
    }
}

fn main() {
    // Absolute path
    crate::sound::instrument::clarinet();

    // Import with use
    use crate::sound::instrument;
    instrument::clarinet();
}
```

Modules can be separated into different files:

```bash
.
├── lib.rs
└── sound/
    ├── mod.rs
    └── instrument.rs
```

## Crate

A crate is a unit of compiled binary. It can be either an executable or a library.

* **Binary crate**: Produces an executable (`main.rs`)
* **Library crate**: Produces a library (`lib.rs`)

## Attributes

`#[]` is called an **Attribute**. It serves a similar role to Java's `@Annotation` or TypeScript's decorator (`@decorator`).

| Syntax | Name | Scope |
|--------|------|-------|
| `#[...]` | Outer Attribute | Applies to the item **below** |
| `#![...]` | Inner Attribute | Applies to the **enclosing** module/crate |

```rs
// #[derive(...)] - auto-implement functionality for structs (most commonly used)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Point {
    x: i32,
    y: i32,
}

// Conditional compilation
#[cfg(target_os = "linux")]
fn linux_only() {}

// Testing
#[test]
fn test_add() {
    assert_eq!(2 + 2, 4);
}

// Deprecated
#[deprecated(since = "1.0.0", note = "Use new_function instead")]
fn old_function() {}

// Inner Attribute - applies to the entire crate
#![allow(unused)]
```

**Commonly used derives:**

| derive | What it provides | When needed? |
|--------|-----------------|-------------|
| `Debug` | `{:?}` output | Debugging, logging |
| `Clone` | `.clone()` copy | When you need to duplicate a value |
| `Copy` | Auto copy (without clone) | Small value types |
| `PartialEq` | `==`, `!=` comparison | When comparing values |
| `Eq` | Full equality | When using as `HashMap` key |
| `Hash` | Hash value computation | When putting in `HashMap`/`HashSet` |
| `Default` | `::default()` default value | When a default is needed |

> In TypeScript, `console.log`, `===`, `{...obj}` just work. In Rust, **structs can do nothing by default** -- printing, comparing, copying all require explicit derives.

## Environment Variables

```rs
use std::env;

// Get environment variable at compile time
// Build fails if the variable doesn't exist
let cargo_dir = env!("CARGO_MANIFEST_DIR");

// Get environment variable at runtime
match env::var("HOME") {
    Ok(val) => println!("HOME: {}", val),
    Err(e) => println!("Error: {}", e),
}

// With default value
let path = env::var("PATH").unwrap_or_else(|_| String::from("/usr/bin"));
```

# Core Concepts

These are concepts unique to Rust. They don't exist in other languages, which makes them challenging at first, but once you understand them the rest becomes easy.

## Ownership and Borrowing

**The most fundamental concept in Rust.** It doesn't exist in other languages, so it's difficult at first.

### Ownership Rules (just remember 3)

1. Every value has exactly **one owner** (variable)
2. When the owner goes out of scope, the value is **automatically dropped**
3. When you pass a value to another variable, **ownership moves** and the original variable can no longer be used

```rs
let s1 = String::from("hello");
let s2 = s1;          // Ownership moves to s2
// println!("{}", s1);  // Error! s1 is no longer valid
println!("{}", s2);    // OK
```

### Borrowing -- lending without transferring ownership

```rs
let s = String::from("hello");

// Immutable borrow -- &s = "I'll lend it to you, read only"
let len = calculate_length(&s);
println!("{}", s);                // Still usable

// Mutable borrow -- &mut s2 = "I'll lend it to you, you can modify it"
let mut s2 = String::from("hello");
change(&mut s2);

fn calculate_length(s: &String) -> usize {
    s.len()
}

fn change(s: &mut String) {
    s.push_str(", world");
}
```

| | Immutable ref `&T` | Mutable ref `&mut T` |
|---|---|---|
| How many at once? | **Multiple** OK | **Only one** |
| Read | Yes | Yes |
| Write | No | Yes |
| Immutable + mutable simultaneously | Not allowed | Not allowed |

> In TypeScript, `const obj = {a: 1}; obj.a = 2;` works. In Rust, if it's `let`, inner values can't be changed either. You need `let mut`.

## Copy vs Clone

```rs
// Copy -- small values on the stack are automatically copied (integers, bool, char, etc.)
let x = 5;
let y = x;        // x is copied, both can be used
println!("{} {}", x, y);  // OK

// Clone -- heap data must be explicitly deep-copied
let s1 = String::from("hello");
// let s2 = s1;         // Ownership moves! s1 can no longer be used
let s2 = s1.clone();    // Explicit copy
println!("{} {}", s1, s2);  // Both OK
```

| | Copy | Clone |
|---|---|---|
| Method | Automatic (implicit) | Explicit `.clone()` |
| Cost | Cheap (stack copy) | Expensive (may copy heap) |
| Targets | `i32, bool, char, f64`, etc. | `String, Vec`, etc. |

Types that implement Copy:

* All integer types: `u32`, etc.
* Boolean type: `bool`
* All floating point types: `f64`, etc.
* Character type: `char`
* **Tuples**: Only if they contain only Copy types. `(i32, i32)` is Copy, `(i32, String)` is not

## Lifetime

Tells the compiler **"how long this reference is valid."** The purpose is to prevent dangling references.

```rs
// 'a = "the return value is valid for the shorter lifetime of x and y"
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}
```

> In most cases, the compiler infers lifetimes automatically. You only need to write them explicitly when "a function takes references and returns a reference."

## References and Pointers

* **References**: `&T`, `&mut T`
* **Raw pointers**: `*const T`, `*mut T` (used in unsafe blocks)
* **Smart pointers**: `Box<T>`, `Rc<T>`, `Arc<T>`, `RefCell<T>`

## Trait

Similar to TypeScript's `interface`, but **can be implemented on existing types after the fact**. `PartialOrd + Copy` is a **trait bound**.

| | TypeScript `interface` | Rust `trait` |
|---|---|---|
| Add impl to existing type | No | Yes |
| Default implementation | No | Yes |
| Generic constraint | `<T extends I>` | `<T: Trait>` |

```rs
trait Summary {
    fn summarize(&self) -> String;

    // Default implementation
    fn summarize_author(&self) -> String {
        String::from("Anonymous")
    }
}

struct Article {
    title: String,
    content: String,
}

impl Summary for Article {
    fn summarize(&self) -> String {
        format!("{}: {}", self.title, self.content)
    }
}

// Trait bound
fn notify<T: Summary>(item: &T) {
    println!("{}", item.summarize());
}

// where clause
fn notify<T>(item: &T)
where
    T: Summary + Display
{
    println!("{}", item.summarize());
}
```

## Box and Smart Pointers

Box is used when you want to store specific data on the heap. If large data is stored on the stack, ownership transfer can be slow due to data movement time.

```rs
fn main() {
    let b = Box::new(5);
    println!("b = {}", b);
}

// Recursive type
enum List {
    Cons(i32, Box<List>),
    Nil,
}
```

**Key smart pointers**:

* **Box<T>**: Allocate data on the heap
* **Rc<T>**: Reference counting (single-threaded)
* **Arc<T>**: Atomic reference counting (multi-threaded)
* **RefCell<T>**: Runtime borrow rule checking
* **Mutex<T>**: Data protection through mutual exclusion

## Error Handling

Rust has **no `try/catch`**. Instead, it uses `Result` and `Option`.

| Type | Meaning | TypeScript Equivalent |
|------|---------|----------------------|
| `Result<T, E>` | Success(`Ok(T)`) or Failure(`Err(E)`) | `Promise<T>` (resolve/reject) |
| `Option<T>` | Has value(`Some(T)`) or None(`None`) | `T \| undefined` |

> `?` is shorthand for "if error, immediately return Err; if success, extract the value." Don't use `unwrap()` in production (causes panic).

```rs
// Recoverable error handling with Result
fn read_file(path: &str) -> Result<String, std::io::Error> {
    std::fs::read_to_string(path)
}

// Error propagation with ? operator
fn process_file() -> Result<(), std::io::Error> {
    let content = read_file("file.txt")?;
    println!("{}", content);
    Ok(())
}

// Unrecoverable error with panic!
fn divide(a: i32, b: i32) -> i32 {
    if b == 0 {
        panic!("Division by zero!");
    }
    a / b
}

// Option type
fn find_user(id: u32) -> Option<User> {
    // Returns Some(user) or None
}
```

# Advanced

## Dispatch

The question of **"when do we decide which implementation to execute"** when calling a function.

```rs
// Static dispatch - decided at compile time (fast, larger binary)
fn process(item: &impl Summary) {
    println!("{}", item.summarize());
}

// Dynamic dispatch - decided at runtime (flexible, slightly slower)
fn process(item: &dyn Summary) {
    println!("{}", item.summarize());
}
```

| | Static `impl Trait` | Dynamic `dyn Trait` |
|---|---|---|
| Decision time | **Compile time** | **Runtime** |
| Speed | Fast (can be inlined) | vtable lookup overhead |
| Binary size | Larger (code duplicated per type) | Smaller |
| Flexibility | Type is fixed | Can mix multiple types |

> **Default to `impl Trait` (static)**, and only use `dyn` when you need to store multiple types in one collection like `Vec<Box<dyn Trait>>`.

## Macros

"Code that generates code." `vec![1,2,3]` is a classic example of a macro.

| Type | Syntax | Use Case |
|------|--------|----------|
| Declarative | `macro_rules!` | Generate code for repetitive patterns |
| Derive | `#[derive(...)]` | Auto-implement traits |
| Attribute | `#[my_macro]` | Custom attributes |
| Function-like | `my_macro!(...)` | Custom DSL |

```rs
// Declarative macro (macro_rules!) - generate code via pattern matching
macro_rules! say_hello {
    () => { println!("Hello!") };
    ($name:expr) => { println!("Hello, {}!", $name) };
}
say_hello!();          // Hello!
say_hello!("David");   // Hello, David!

// Internal implementation of the vec! macro
macro_rules! vec {
    ( $( $x:expr ),* ) => {
        {
            let mut temp_vec = Vec::new();
            $(
                temp_vec.push($x);
            )*
            temp_vec
        }
    };
}

// Procedural macro (derive) - the most commonly used form
#[derive(Debug, Clone, PartialEq)]
struct Point { x: f64, y: f64 }
```

> At first, just knowing `#[derive(Debug, Clone)]` is enough. Learn `macro_rules!` later when you need it.

## Unsafe Rust

**Temporarily disabling** Rust's safety guarantees. Only 4 things are allowed inside an `unsafe` block:

1. Dereference raw pointers
2. Call unsafe functions
3. Access mutable global variables
4. Implement unsafe traits

```rs
unsafe {
    // Dereference raw pointer
    let mut num = 5;
    let r1 = &num as *const i32;
    let r2 = &mut num as *mut i32;
    println!("r1: {}", *r1);

    // Call unsafe function
    unsafe fn dangerous() {}
    dangerous();
}
```

> **99% of Rust code doesn't need `unsafe`.** It's only used for FFI (calling C libraries) or extreme performance optimization.

## Concurrency

Rust's concurrency motto is **"fearless concurrency."** The ownership system prevents data races at compile time.

| Tool | Use Case | TypeScript Equivalent |
|------|----------|----------------------|
| `thread::spawn` | Create thread | `new Worker()` |
| `Arc<T>` | Share ownership across threads | N/A |
| `Mutex<T>` | Mutual exclusion lock | N/A (JS is single-threaded) |
| `channel` | Message passing between threads | `postMessage` |

```rs
use std::thread;
use std::sync::{Arc, Mutex};

fn main() {
    // Create thread
    let handle = thread::spawn(|| {
        println!("Hello from thread!");
    });

    handle.join().unwrap();

    // Shared state
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Result: {}", *counter.lock().unwrap());
}
```

## Async Programming

The **syntax is similar** to TypeScript's `async/await`, but in Rust the **runtime is separate**.

| | TypeScript | Rust |
|---|---|---|
| Runtime | Built into V8 | **Installed separately** (Tokio, async-std) |
| `async fn` | Returns Promise | Returns Future |
| `.await` | Same | Same |
| Event loop | Automatic | `#[tokio::main]` required |

```rs
async fn fetch_data() -> Result<String, Error> {
    // Async operation
    Ok(String::from("data"))
}

#[tokio::main]
async fn main() {
    let result = fetch_data().await;
    match result {
        Ok(data) => println!("{}", data),
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

## Tokio

* [tokio @ github](https://github.com/tokio-rs/tokio)
* [Making the Tokio scheduler 10x faster](https://tokio.rs/blog/2019-10-scheduler)

Tokio is Rust's asynchronous runtime framework. It provides building blocks for writing asynchronous applications using async/await syntax.

```rs
use tokio::net::TcpListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind("127.0.0.1:8080").await?;

    loop {
        let (mut socket, _) = listener.accept().await?;

        tokio::spawn(async move {
            let mut buf = [0; 1024];

            loop {
                let n = match socket.read(&mut buf).await {
                    Ok(n) if n == 0 => return,
                    Ok(n) => n,
                    Err(_) => return,
                };

                if socket.write_all(&buf[0..n]).await.is_err() {
                    return;
                }
            }
        });
    }
}
```

## Memory Management

Rust guarantees memory safety at compile time through the ownership system. There's no GC, but the **ownership system automatically calls `drop()` when a scope ends**.

| | Stack | Heap |
|---|---|---|
| Allocation/Deallocation | Automatic, very fast | Relatively slow |
| Size | Fixed at compile time | Dynamic at runtime |
| Targets | `i32`, `bool`, `[u8; 5]` | `String`, `Vec`, `Box` |
| Deallocation timing | When scope ends | When owner goes out of scope |

```rs
fn main() {
    // Stack allocation
    let x = 5;

    // Heap allocation
    let s = String::from("hello");
    let v = vec![1, 2, 3];
    let b = Box::new(5);
}  // Memory is automatically freed when scope ends
```

## Performance Optimization

| Tip | Description |
|-----|-------------|
| Prefer iterators | `.iter().map().filter()` chains optimize better than `for` loops |
| Use `&str` | Use instead of `String` when possible -- avoids heap allocation |
| `Vec::with_capacity` | Pre-allocate when size is known -- avoids reallocations |
| `#[inline]` | Inline hint for hot path functions |
| Profile first | Measure with `cargo flamegraph`, `criterion` before optimizing |

> **"Don't guess, measure."** Find bottlenecks with `cargo flamegraph` first, then optimize.

```rs
// Iterator chain (optimized)
let sum: i32 = (0..100)
    .filter(|x| x % 2 == 0)
    .map(|x| x * 2)
    .sum();

// Benchmark
#[bench]
fn bench_function(b: &mut Bencher) {
    b.iter(|| {
        // Code to benchmark
    });
}
```

# Style Guide

[Rust Style Guide](rust_style_guide.md)

# Refactoring

[Rust Refactoring](rust_refactoring.md)

# Effective Rust

[Effective Rust](effective_rust.md)

# Rust Design Patterns

[Rust Design Patterns](rust_design_pattern.md)

# Rust Architecture

[Rust Architecture](rust_architecture.md)
