- [Abstract](#abstract)
- [References](#references)
- [Materials](#materials)
- [Basic](#basic)
  - [Features](#features)
  - [Install](#install)
  - [Tools](#tools)
  - [Build and Run](#build-and-run)
  - [Hello World](#hello-world)
  - [Useful Keywords](#useful-keywords)
  - [Ownership](#ownership)
  - [Copy vs Clone](#copy-vs-clone)
  - [Crate](#crate)
  - [Module](#module)
  - [Control Flows](#control-flows)
  - [Struct](#struct)
  - [Enum](#enum)
  - [Box](#box)
  - [Generic](#generic)
  - [Trait](#trait)
  - [Tuple](#tuple)
  - [Match](#match)
  - [Array](#array)
  - [Formatted Print](#formatted-print)
  - [Custom Errors](#custom-errors)
  - [Advanced Error Handling](#advanced-error-handling)
  - [String Conversions](#string-conversions)
  - [String Loops](#string-loops)
  - [if let](#if-let)
  - [Lifetime](#lifetime)
  - [Attributes](#attributes)
  - [Copy, Clone](#copy-clone)
  - [Dispatch](#dispatch)
  - [Evnironment Variables](#evnironment-variables)
- [Advanced](#advanced)
  - [Tokio](#tokio)

----

# Abstract

Rust 에 대해 정리한다.

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

## Useful Keywords

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

create 은 build 된 binary 의 단위이다. 실행파일 혹은 라이브러리로 구분할 수
있다. 

## Module

module 은 code 의 logical unit 이다. 관련된 code 들이 한덩이 모여있는 것이다.
다음과 같이 `mod` 를 사용하여 module 을 정의한다. 또한 `use` 를 사용하여 다른
module 의 function 등을 import 한다. module function 은 기본적으로 prviate 이다.
`pub` 을 사용하여 public 으로 바꿀 수 있다.

```rs
mod print_things {
    use std::fmt::Display;

    fn prints_one_thing<T: Display>(input: T) { // Print anything that implements Display
        println!("{}", input)
    }
}

fn main() {}
```

또한 module 을 별도의 파일로 분리할 수 있다. 예를 들어 다음과 같이 **http**
module 을 만들어 보자. 반드시 분리된 module directory 에 `mod.rs` 가 있어야
한다. `mod.rs` 에 module tree 에 포함할 파일들의 이름을 `mod` 를 이용하여
선언해야 한다. 그래야 해당파일들이 module tree 에 포함되고 `mod.rs` 가 compile
될 때 함께 compile 된다.

또한 http module 은 `main.rs` 에 `mod http;` 로 선언되야 한다. 그래야 http
module 의 file 들이 root module tree 에 포함된다. 그리고 rustc 가 main.rs 를
compile 할 때 **http** module 을 compile 할 것이다. 

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

`use crate::server::Server;` 의 `crate` 는 root module 을 말한다.

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

golang 의 struct 와 같다. `struct` 로 정의하고 `impl` 로 구현한다. **instant
function** 은 첫번째 arguement 가 self 이다. **associate function** (static
function) 은 첫번째 arguement 가 self 가 아니다. 

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

enumrations 을 `enum` 으로 정의한다. **enum** 의 member 를 **variant** 라고
한다. 다음은 간단한 **enum** 의 예이다.

```rs
// main.rs
enum Job {
   Programmer,
   Teacher,
   Warrior,
}
fn main() {}
```

varriant 는 다음과 같이 data 를 가질 수 있다.

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

`use Job::*;` 를 선언하면 `Job::` 은 typing 하지 않아도 된다.

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

## Box

* [Box<T>는 힙에 있는 데이터를 가리키고 알려진 크기를 갖습니다](https://rinthel.github.io/rust-lang-book-ko/ch15-01-box.html)

----

Box 는 특정 데이터를 heap 에 보관하고 싶을 때 쓴다. 크기가 큰 데이터가 stack 에
보관되는 경우 ownership 이 옮겨질 때 데이터의 이동시간이 길 수 있다.

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

Lifetime 의 목적은 dangling reference 를 방지하는 것이다.

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

## Evnironment Variables

```rs
// env! get env variable value at compile time.
//   If there is no env variable build will be failed. 
let default_path = format!("{}/public", env!("CARGO_MANIFEST_DIR"));


```

# Advanced

## Tokio

* [tokio @ github](https://github.com/tokio-rs/tokio)
* [Making the Tokio scheduler 10x faster](https://tokio.rs/blog/2019-10-scheduler)

-----

Tokio 는 Rust 의 Asynchronous Framework 이다.
