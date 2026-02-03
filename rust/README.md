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
  - [Ownership and Borrowing](#ownership-and-borrowing)
  - [Copy vs Clone](#copy-vs-clone)
  - [Lifetime](#lifetime)
  - [References and Pointers](#references-and-pointers)
  - [Control Flows](#control-flows)
  - [Pattern Matching](#pattern-matching)
  - [Collections](#collections)
  - [String Types](#string-types)
  - [String Conversions](#string-conversions)
  - [String Loops](#string-loops)
  - [Formatted Print](#formatted-print)
  - [Struct](#struct)
  - [Enum](#enum)
  - [Tuple](#tuple)
  - [Array](#array)
  - [Functions](#functions)
  - [Closures](#closures)
  - [Generic](#generic)
  - [Trait](#trait)
  - [Box and Smart Pointers](#box-and-smart-pointers)
  - [Module System](#module-system)
  - [Crate](#crate)
  - [Error Handling](#error-handling)
  - [Attributes](#attributes)
  - [Environment Variables](#environment-variables)
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
  - [Rust Naming Conventions](#rust-naming-conventions)
  - [Code Organization](#code-organization)
  - [Best Practices](#best-practices)
- [Refactoring](#refactoring)
  - [Common Refactoring Patterns](#common-refactoring-patterns)
  - [Refactoring Tools](#refactoring-tools)
- [Effective Rust](#effective-rust)
  - [Ownership and Borrowing](#ownership-and-borrowing)
  - [Error Handling](#error-handling)
  - [Performance](#performance)
  - [API Design](#api-design)
  - [Testing](#testing)
- [Rust Design Patterns](#rust-design-patterns)
  - [Creational Patterns](#creational-patterns)
  - [Structural Patterns](#structural-patterns)
  - [Behavioral Patterns](#behavioral-patterns)
- [Rust Architecture](#rust-architecture)
  - [Project Structure](#project-structure)
  - [Layered Architecture](#layered-architecture)
  - [Hexagonal Architecture](#hexagonal-architecture-ports--adapters)
  - [Async Architecture](#async-architecture)
  - [Microservices](#microservices)

----

# Abstract

Rust is a systems programming language that focuses on safety, speed, and concurrency. It achieves memory safety without garbage collection through its unique ownership system. This document provides a comprehensive guide to Rust programming language features, patterns, and best practices.

**Korean version**: See [README-kr.md](README-kr.md) for Korean language documentation.

# References

* [Command-Line Rust | oreilly](https://www.oreilly.com/library/view/command-line-rust/9781098109424/)
  * clone coding for command lines such as `echo, cat, head, wc, uniq, find, cut, grep, comm, tail, fortune, cal, ls`.
  * [src](https://github.com/kyclark/command-line-rust)
* [Comprehensive Rust](https://google.github.io/comprehensive-rust/welcome.html)
  * [kor](https://google.github.io/comprehensive-rust/ko/welcome.html)
* [Awesome Rust @ github](https://github.com/rust-unofficial/awesome-rust)
* [Rust](https://www.rust-lang.org/learn)
  * [Crate std reference](https://doc.rust-lang.org/stable/std/index.html)
* [The Rust Programming Language](https://doc.rust-lang.org/book/index.html) 
  * [한글](https://rinthel.github.io/rust-lang-book-ko/)
* [Learn Rust by Building Real Applications](https://www.udemy.com/course/rust-fundamentals/)
  * [src](https://github.com/gavadinov/Learn-Rust-by-Building-Real-Applications)
  
# Materials

* [Rust 언어 튜토리얼](http://sarojaba.github.io/rust-doc-korean/doc/tutorial.html)
* [Easy Rust](https://dhghomon.github.io/easy_rust/Chapter_0.html)
  * [video](https://www.youtube.com/playlist?list=PLfllocyHVgsRwLkTAhG0E-2QxCf-ozBkk)
* [Asynchronous Programming in Rust](https://rust-lang.github.io/async-book/)
  
# Basic

## Features

* No Exception
* No OOP
* No Garbage Collector
* No Dangling Reference

## Install

```bash
# Install rust on macosx
$ brew install rust

# Install rustup, rustc, cargo on macosx
$ curl https://sh.rustup.rs -sSf | sh
```

## Tools

* **rustc** : rust compiler
* **cargo** : rust package manager, build tool
* **rustup** : rust toolchain (rustc, cargo) manager

## Build and Run

```bash
# Build and run with rustc
$ rustc a.rs -o a.out
$ ./a.out

# Build and run with cargo
$ cargo new a
$ cd a
$ cargo run

# cargo commands
# Build current directory
$ cargo build
# Run current directory
$ cargo run
# Test current directory
$ cargo test
# Install crate
$ cargo install cargo-expand
# Use the crate
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
as - perform primitive casting, disambiguate the specific trait containing an item, or rename items in use and extern crate statements
async - return a Future instead of blocking the current thread
await - suspend execution until the result of a Future is ready
break - exit a loop immediately
const - define constant items or constant raw pointers
continue - continue to the next loop iteration
crate - link an external crate or a macro variable representing the crate in which the macro is defined
dyn - dynamic dispatch to a trait object
else - fallback for if and if let control flow constructs
enum - define an enumeration
extern - link an external crate, function, or variable
false - Boolean false literal
fn - define a function or the function pointer type
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
return - return from function
Self - a type alias for the type we are defining or implementing
self - method subject or current module
static - global variable or lifetime lasting the entire program execution
struct - define a structure
super - parent module of the current module
trait - define a trait
true - Boolean true literal
type - define a type alias or associated type
union - define a union and is only a keyword when used in a union declaration
unsafe - denote unsafe code, functions, traits, or implementations
use - bring symbols into scope
where - denote clauses that constrain a type
while - loop conditionally based on the result of an expression
```

## Ownership

**Ownership Rules**

* Each value in Rust has a variable that’s called its owner.
* There can only be one owner at a time.
* When the owner goes out of scope, the value will be dropped.

## Copy vs Clone

**Copy trait** means shallow copy, **Clone trait** means deep copy.

Here are some of the types that implement Copy:

* All the integer types, such as **u32**.
* The Boolean type, bool, with values true and **false**.
* All the floating point types, such as **f64**.
* The character type, **char**.
* **Tuples**, if they only contain types that also implement Copy. For example,
  `(i32, i32)` implements Copy, but `(i32, String)` does not.

## Crate

A crate is the unit of a built binary. It can be classified as either an executable or a library.

## Module System

A module is a logical unit of code. It's a collection of related code grouped together. You define modules using `mod` and import functions from other modules using `use`. Module functions are private by default, but you can make them public using `pub`.

```rs
mod print_things {
    use std::fmt::Display;

    fn prints_one_thing<T: Display>(input: T) { // Print anything that implements Display
        println!("{}", input)
    }
}

fn main() {}
```

Modules can be separated into different files. For example, let's create an **http** module. The separated module directory must have a `mod.rs` file. In `mod.rs`, you need to declare the names of files to include in the module tree using `mod`. This way, those files are included in the module tree and compiled together when `mod.rs` is compiled.

Additionally, the http module must be declared in `main.rs` as `mod http;`. This includes the http module's files in the root module tree, and rustc will compile the **http** module when compiling main.rs. 

```bash
.
├── http
│   ├── method.rs
│   └── mod.rs
├── main.rs
└── server.rs
```

```rs
//////////////////////////////////////////////////
// http/method.rs
use std::str::FromStr;

pub struct MethodError;

#[derive(Debug)]
pub enum Method {
    GET,
    DELETE,
    POST,
    PUT,
    HEAD,
    CONNECT,
    OPTIONS,
    TRACE,
    PATCH,
}

impl FromStr for Method {
    type Err = MethodError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "GET" => Ok(Self::GET),
            "DELETE" => Ok(Self::DELETE),
            "POST" => Ok(Self::POST),
            "PUT" => Ok(Self::PUT),
            "HEAD" => Ok(Self::HEAD),
            "CONNECT" => Ok(Self::CONNECT),
            "OPTIONS" => Ok(Self::OPTIONS),
            "TRACE" => Ok(Self::TRACE),
            "PATCH" => Ok(Self::PATCH),
            _ => Err(MethodError),
        }
    }
}

//////////////////////////////////////////////////
// http/mod.rs
pub mod method;

//////////////////////////////////////////////////
// main.rs
mod server;
mod http;

use std::env;
use crate::server::Server;

fn main() {
    let def_pub_path = format!("{}/public", env!("CARGO_MANIFEST_DIR"));
    let pub_path = env::var("PUBLIC_PATH")
        .unwrap_or(def_pub_path);
    println!("public path: {}", pub_path);
    let server = Server::new("0.0.0.0:8080".to_string());
    server.run();
}
```

`use crate::server::Server;` - here `crate` refers to the root module.

## Control Flows

if, let-if, loop, while, for

```rs
// if

// let-if

// loop

// while

// for
```

## Struct

Similar to structs in Go. Define with `struct` and implement with `impl`. **Instance methods** take `self` as the first argument. **Associated functions** (static functions) do not take `self` as the first argument. 

```rs
// server.rs
pub struct Server {
    addr: String,
}

impl Server {
    pub fn new(addr: String) -> Self {
        Self { addr }
    }

    pub fn run(self) {
        println!("Listening on {}...", self.addr);
    }
}
```

## Enum

* [Enums @ easy-rust](https://dhghomon.github.io/easy_rust/Chapter_25.html)

----

Enumerations are defined with `enum`. Members of an **enum** are called **variants**. Here is a simple **enum** example.

```rs
// main.rs
enum Job {
   Programmer,
   Teacher,
   Warrior,
}
fn main() {}
```

Variants can hold data as shown below.

```rs
// main.rs
enum Job {
   Programmer(String),
   Teacher(String),
   Warrior(String),
}
fn create_job(job_code: i32) {
   match job_code {
      1..=10 => Job::Programmer(String::from("I am a programmer")),
      11 => Job::Programmer(String::from("I am a teacher")),
      _ => Job::Programmer(String::from("I am a warrior")),
   }
}
fn check_job(job: &Job) {
   match job {
      Job::Programmer(description) => println("{}", description),
      Job::Teacher(a) => println("{}", a),
   }
}
fn main() {
   let job_code = 8;
   let job = create_job(job_code);
   check_job(job);
}
```

By declaring `use Job::*;`, you don't need to type `Job::`.

```rs
// main.rs
enum Job {
   Programmer(String),
   Teacher(String),
   Warrior(String),
}
fn create_job(job_code: i32) {
   use Job::*;
   match job_code {
      1..=10 => Programmer(String::from("I am a programmer")),
      11 => Programmer(String::from("I am a teacher")),
      _ => Programmer(String::from("I am a warrior")),
   }
}
fn check_job(job: &Job) {
   use Job::*;
   match job {
      Programmer(description) => println("{}", description),
      Teacher(a) => println("{}", a),
   }
}
fn main() {
   let job_code = 8;
   let job = create_job(job_code);
   check_job(job);
}
```

## Box and Smart Pointers

* [Box<T> Points to Data on the Heap and Has a Known Size](https://doc.rust-lang.org/book/ch15-01-box.html)

Box is used when you want to store data on the heap. When large data is stored on the stack, moving data during ownership transfer can be time-consuming.

```rs
fn main() {
    let b = Box::new(5);
    println!("b = {}", b);
}
```

## Generic

## Trait

Something like interface in Java. `PartialOrd + Copy` is a **trait bound**.

```rs
use std::cmp::PartialOrd;

fn largest<T: PartialOrd + Copy>(list: &[T]) -> T {
    let mut largest = list[0];
    for &item in list.iter() {
        if item > largest {
            largest = item;
        }
    }
    largest
}

fn main() {
    let numbers = vec![34, 50, 25, 100, 65];
    let result = largest(&numbers);
    println!("The largest number is {}", result);
    let chars = vec!['y', 'm', 'a', 'q'];
    let result = largest(&chars);
    println!("The largest char is {}", result);
}
```

## Tuple

```rs
let t = (1, "abc", 2.5);
```

## Match

Something like switch statement in C.

## Array

```rs
// [u8] is array of u8 size unknown
fn foo(a: [u8]) {}

// [u8] is array of u8 size 5
fn foo(a: [u8; 5]) {}

// &[u8] is array reference of u8 size unknown
fn foo(a: &[u8]) {}

// array reference is a slice
let a = p[1, 2, 3, 4, 5];
foo(&a[1..3]);

// array with value 0 size 1024 bytes
let mut buf = [0; 1024];
```

## Formatted Print

## Custom Errors

## Advanced Error Handling

```rs
// ParseError
enum ParseError {
   InvalidEncoding;
}

// or
match str::from_utf8(buf).or(Err(ParseError::InvalidEncoding)) {
   Ok(request) => {}
   Err(e) => return Err(e),
}

// ? means ...
let req = str::from_utf8(buf).or(Err(ParseError::InvalidEncoding))?;

//
impl From<Utf8Error> fro ParseError {

}
```

## String Conversions

* [string-conversion.rs](https://gist.github.com/jimmychu0807/9a89355e642afad0d2aeda52e6ad2424)

----

```rust
  // -- FROM: vec of chars --
  let src1: Vec<char> = vec!['j','{','"','i','m','m','y','"','}'];
  // to String
  let string1: String = src1.iter().collect::<String>();
  // to str
  let str1: &str = &src1.iter().collect::<String>();
  // to vec of byte
  let byte1: Vec<u8> = src1.iter().map(|c| *c as u8).collect::<Vec<_>>();
  println!("Vec<char>:{:?} | String:{:?}, str:{:?}, Vec<u8>:{:?}", src1, string1, str1, byte1);

  // -- FROM: vec of bytes --
  // in rust, this is a slice
  // b - byte, r - raw string, br - byte of raw string
  let src2: Vec<u8> = br#"e{"ddie"}"#.to_vec();
  // to String
  // from_utf8 consume the vector of bytes
  let string2: String = String::from_utf8(src2.clone()).unwrap();
  // to str
  let str2: &str = str::from_utf8(&src2).unwrap();
  // to vec of chars
  let char2: Vec<char> = src2.iter().map(|b| *b as char).collect::<Vec<_>>();
  println!("Vec<u8>:{:?} | String:{:?}, str:{:?}, Vec<char>:{:?}", src2, string2, str2, char2);

  // -- FROM: String --
  let src3: String = String::from(r#"o{"livia"}"#);
  let str3: &str = &src3;
  let char3: Vec<char> = src3.chars().collect::<Vec<_>>();
  let byte3: Vec<u8> = src3.as_bytes().to_vec();
  println!("String:{:?} | str:{:?}, Vec<char>:{:?}, Vec<u8>:{:?}", src3, str3, char3, byte3);

  // -- FROM: str --
  let src4: &str = r#"g{'race'}"#;
  let string4 = String::from(src4);
  let char4: Vec<char> = src4.chars().collect();
  let byte4: Vec<u8> = src4.as_bytes().to_vec();
  println!("str:{:?} | String:{:?}, Vec<char>:{:?}, Vec<u8>:{:?}", src4, string4, char4, byte4);
```

## String Loops

```rs
// loop for string traversal
fn get_next_word(req: &str) -> Option<(&str, &str)> {
   let mut iter = req.chars();
   loop {
      let item = iter.next();
      match item {
         Some(c) => {},
         None => break,
      }
   }
}

// for iterator for string traversal
fn get_next_word(req: &str) -> Option<(&str, &str)> {
   for c in req.chars() {

   }
}

// for enumerate for string traversal
fn get_next_word(req: &str) -> Option<(&str, &str)> {
   for (i, c) in req.chars().enumerate() {
      if c == ' ' {
         return Some((&req[..i], &req[i+1..]));
      }
   }
   None
}
```

## if let

```rs
if let Some(i) = path.find('?') {
   query_string = Some(&path[i+1...]);
   path = &path[..i];
}
```

## Lifetime

The purpose of lifetimes is to prevent dangling references.

## Attributes

```rs
#[derive(Copy, Clone, Debug)]
```

## Copy, Clone

## Dispatch

There are 2 kinds of dispatch such as dynamic, static.

```rs
```

We use `dyn` for dynamic dispatch and `impl` for static dispatch. static dispatch is slow and makes larger binary.

## Environment Variables

```rs
// env! get env variable value at compile time.
//   If there is no env variable build will be failed. 
let default_path = format!("{}/public", env!("CARGO_MANIFEST_DIR"));


```

# Advanced

## Tokio

* [tokio @ github](https://github.com/tokio-rs/tokio)
* [Making the Tokio scheduler 10x faster](https://tokio.rs/blog/2019-10-scheduler)

Tokio is Rust's asynchronous runtime framework. It provides the building blocks for writing asynchronous applications with async/await syntax.

# Style Guide

## Rust Naming Conventions

* **Modules**: Use snake_case (e.g., `my_module`)
* **Types**: Use UpperCamelCase (e.g., `MyStruct`)
* **Functions**: Use snake_case (e.g., `my_function`)
* **Constants**: Use SCREAMING_SNAKE_CASE (e.g., `MY_CONSTANT`)
* **Lifetimes**: Use short lowercase names (e.g., `'a`, `'b`)

## Code Organization

* Keep modules focused and cohesive
* Use `mod.rs` or module files for organization
* Export public APIs carefully with `pub`
* Document public APIs with `///` doc comments

## Best Practices

* Prefer immutability by default
* Use `Result` and `Option` instead of panicking
* Handle errors explicitly with `?` operator
* Use iterators and functional patterns where appropriate
* Avoid unnecessary clones

# Refactoring

## Common Refactoring Patterns

### Extract Method
Break down large functions into smaller, focused ones.

### Replace Clone with References
Use borrowing instead of cloning when possible.

### Introduce Type Parameter
Replace concrete types with generics for reusability.

### Extract Trait
Create traits to abstract common behavior.

### Replace Error Handling
Migrate from `unwrap()` to proper error propagation with `?`.

## Refactoring Tools

* **rustfmt**: Automatic code formatting
* **clippy**: Linting and suggestions
* **rust-analyzer**: IDE support with refactoring actions

# Effective Rust

## Ownership and Borrowing

1. **Prefer borrowing over ownership transfer** when you don't need ownership
2. **Use lifetimes explicitly** when references are involved
3. **Avoid unnecessary clones** - use references and slices
4. **Return owned values** from functions when necessary

## Error Handling

1. **Use `Result<T, E>`** for recoverable errors
2. **Use `panic!`** only for unrecoverable errors
3. **Implement `From` trait** for error conversions
4. **Use `thiserror`** or `anyhow`** crates for better error handling

## Performance

1. **Prefer iterators** over loops for better optimization
2. **Use `&str` instead of `String`** when possible
3. **Avoid allocations** in hot paths
4. **Use `Vec::with_capacity`** when size is known
5. **Profile before optimizing** with tools like `cargo flamegraph`

## API Design

1. **Return `impl Trait`** for flexibility
2. **Use builder pattern** for complex constructors
3. **Implement standard traits** (`Debug`, `Display`, `Clone`, etc.)
4. **Use `Cow<str>`** for flexible string APIs

## Testing

1. **Write unit tests** with `#[test]`
2. **Use `#[cfg(test)]`** modules for test code
3. **Write integration tests** in `tests/` directory
4. **Use `cargo test`** for running tests
5. **Document examples** that also serve as tests

# Rust Design Patterns

## Creational Patterns

### Builder Pattern
```rs
pub struct Config {
    host: String,
    port: u16,
    timeout: u64,
}

impl Config {
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder::default()
    }
}

pub struct ConfigBuilder {
    host: String,
    port: u16,
    timeout: u64,
}

impl ConfigBuilder {
    pub fn host(mut self, host: String) -> Self {
        self.host = host;
        self
    }

    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    pub fn build(self) -> Config {
        Config {
            host: self.host,
            port: self.port,
            timeout: self.timeout,
        }
    }
}
```

### Factory Pattern
Use associated functions (static methods) as factories.

```rs
impl Widget {
    pub fn new_button() -> Self { /* ... */ }
    pub fn new_label() -> Self { /* ... */ }
}
```

## Structural Patterns

### Newtype Pattern
Wrap existing types to add type safety.

```rs
struct UserId(u64);
struct ProductId(u64);
```

### Type State Pattern
Use types to represent states.

```rs
struct Locked;
struct Unlocked;

struct Door<State> {
    state: PhantomData<State>,
}

impl Door<Locked> {
    fn unlock(self) -> Door<Unlocked> { /* ... */ }
}

impl Door<Unlocked> {
    fn lock(self) -> Door<Locked> { /* ... */ }
}
```

## Behavioral Patterns

### Strategy Pattern
Use trait objects or generics for different strategies.

```rs
trait CompressionStrategy {
    fn compress(&self, data: &[u8]) -> Vec<u8>;
}

struct Gzip;
impl CompressionStrategy for Gzip { /* ... */ }

struct Zlib;
impl CompressionStrategy for Zlib { /* ... */ }
```

### Visitor Pattern
Use traits to separate algorithms from data structures.

### RAII Pattern
Resource Acquisition Is Initialization - automatic cleanup.

```rs
struct File {
    handle: FileHandle,
}

impl Drop for File {
    fn drop(&mut self) {
        // Cleanup happens automatically
        self.handle.close();
    }
}
```

# Rust Architecture

## Project Structure

```
my_project/
├── Cargo.toml          # Package manifest
├── Cargo.lock          # Dependency lock file
├── src/
│   ├── main.rs         # Binary entry point
│   ├── lib.rs          # Library entry point
│   ├── module1.rs      # Module file
│   └── module2/        # Module directory
│       ├── mod.rs      # Module declaration
│       └── submod.rs   # Submodule
├── tests/              # Integration tests
│   └── integration_test.rs
├── benches/            # Benchmarks
│   └── benchmark.rs
└── examples/           # Example binaries
    └── example.rs
```

## Layered Architecture

### Domain Layer
Core business logic with no external dependencies.

```rs
// domain/user.rs
pub struct User {
    id: UserId,
    email: Email,
}
```

### Application Layer
Use cases and application services.

```rs
// application/user_service.rs
pub struct UserService {
    repo: Box<dyn UserRepository>,
}
```

### Infrastructure Layer
External dependencies (database, HTTP, etc.).

```rs
// infrastructure/postgres_user_repo.rs
pub struct PostgresUserRepository {
    pool: PgPool,
}
```

## Hexagonal Architecture (Ports & Adapters)

Define traits (ports) for external dependencies and implement adapters.

```rs
// ports
trait UserRepository {
    fn find(&self, id: UserId) -> Result<User>;
    fn save(&self, user: &User) -> Result<()>;
}

// adapters
struct PostgresUserRepository;
impl UserRepository for PostgresUserRepository { /* ... */ }

struct InMemoryUserRepository;
impl UserRepository for InMemoryUserRepository { /* ... */ }
```

## Async Architecture

Use Tokio or async-std for asynchronous applications.

```rs
#[tokio::main]
async fn main() {
    let server = Server::new();
    server.run().await;
}
```

## Microservices

* Use **tonic** for gRPC services
* Use **actix-web** or **axum** for REST APIs
* Use **rdkafka** for Kafka integration
* Use **redis** for caching
