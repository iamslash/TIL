# Abstract

Rust 에 대해 정리한다.

# References

* [Awesome Rust @ github](https://github.com/rust-unofficial/awesome-rust)

# Materials

* [Rust 언어 튜토리얼](http://sarojaba.github.io/rust-doc-korean/doc/tutorial.html)
* [Easy Rust](https://dhghomon.github.io/easy_rust/Chapter_0.html)
  * [video](https://www.youtube.com/playlist?list=PLfllocyHVgsRwLkTAhG0E-2QxCf-ozBkk)
* [The Rust Programming Language](https://doc.rust-lang.org/book/index.html)
* [Asynchronous Programming in Rust](https://rust-lang.github.io/async-book/)
  
# Basic

## Install

```bash
$ brew install rust
```

## Build and Run

```bash
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

## Useful Keywords

## RAII 

* Rust 의 모든 value 는 variable 가 ownership 을 가지고 있다.
* owner 가 scope 을 벗어나면 value 는 deallocate 된다.
* owner 는 항상 하나이다.

```rs

```

## Ownership

```rs
    // Immutable Borrow, Immutable Borrow is ok
    let mut a = String::new();
    a.push_str("Hello World");

    let b = &a; // Immutable Borrow
    let c = &a; // Immutable Borrow

    println!("b : {}, c : {}", b, c);

    // Mutable Borrow, Mutable Borrow is error
    let mut a = String::new();
    a.push_str("Hello World");

    let b = &mut a; // Mutable Borrow
    let c = &a; // Immutable Borrow

    println!("b : {}, c : {}", b, c);
error[E0502]: cannot borrow `a` as immutable because it is also borrowed as mutable
  --> src/main.rs:34:13
   |
33 |     let b = &mut a; // Mutable Borrow
   |             ------ mutable borrow occurs here
34 |     let c = &a; // Immutable Borrow
   |             ^^ immutable borrow occurs here
35 | 
36 |     println!("b : {}, c : {}", b, c);
   |                                - mutable borrow later used here
```


