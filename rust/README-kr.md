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
- [Core Concepts (핵심 개념)](#core-concepts-핵심-개념)
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
  - [Rust 명명 규칙](#rust-명명-규칙)
  - [코드 구성](#코드-구성)
  - [Best Practices](#best-practices)
- [Refactoring](#refactoring)
  - [일반적인 리팩토링 패턴](#일반적인-리팩토링-패턴)
  - [리팩토링 도구](#리팩토링-도구)
- [Effective Rust](#effective-rust)
  - [Ownership과 Borrowing](#ownership과-borrowing)
  - [에러 처리](#에러-처리)
  - [성능](#성능)
  - [API 설계](#api-설계)
  - [테스팅](#테스팅)
- [Rust Design Patterns](#rust-design-patterns)
  - [생성 패턴](#생성-패턴)
  - [구조 패턴](#구조-패턴)
  - [행동 패턴](#행동-패턴)
- [Rust Architecture](#rust-architecture)
  - [프로젝트 구조](#프로젝트-구조)
  - [계층형 아키텍처](#계층형-아키텍처)
  - [헥사고날 아키텍처](#헥사고날-아키텍처-ports--adapters)
  - [비동기 아키텍처](#비동기-아키텍처)
  - [마이크로서비스](#마이크로서비스)

----

# Abstract

Rust는 안전성, 속도, 동시성에 초점을 맞춘 시스템 프로그래밍 언어입니다. 고유한 소유권 시스템을 통해 가비지 컬렉션 없이 메모리 안전성을 달성합니다. 이 문서는 Rust 프로그래밍 언어의 기능, 패턴 및 모범 사례에 대한 포괄적인 가이드를 제공합니다.

**영문 버전**: 영문 문서는 [README.md](README.md)를 참조하세요.

# References

* [Command-Line Rust | oreilly](https://www.oreilly.com/library/view/command-line-rust/9781098109424/)
  * `echo, cat, head, wc, uniq, find, cut, grep, comm, tail, fortune, cal, ls` 같은 커맨드 라인 도구를 클론 코딩합니다.
  * [src](https://github.com/kyclark/command-line-rust)
* [Comprehensive Rust](https://google.github.io/comprehensive-rust/welcome.html)
  * [한글](https://google.github.io/comprehensive-rust/ko/welcome.html)
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
* [Rust By Example 한글](https://doc.rust-kr.org/rust-by-example-ko/)
* [러스트 프로그래밍 공식 가이드](http://www.yes24.com/Product/Goods/83075894)

# Basic

## Features

Rust는 **"없는 것"**으로 정의되는 언어입니다:

| 없는 것 | 대신 있는 것 | 왜? |
|---------|-------------|-----|
| Exception | `Result<T, E>`, `Option<T>` | 에러를 무시할 수 없게 강제 |
| OOP (클래스 상속) | `struct` + `trait` | 상속 대신 합성(composition) |
| Garbage Collector | 소유권 시스템 | 컴파일 타임에 메모리 해제 결정 |
| Dangling Reference | 라이프타임 검사 | 컴파일러가 죽은 참조를 원천 차단 |

> TypeScript/Java에서 오면 가장 충격적인 점: **변수에 값을 넘기면 원래 변수를 더 이상 쓸 수 없습니다** (소유권 이동).

## Install

```bash
# macOS에서 rust 설치
$ brew install rust

# macOS에서 rustup, rustc, cargo 설치
$ curl https://sh.rustup.rs -sSf | sh
```

## Tools

* **rustc** : rust 컴파일러
* **cargo** : rust 패키지 관리자, 빌드 도구
* **rustup** : rust 툴체인(rustc, cargo) 관리자

## Build and Run

```bash
# rustc로 빌드 및 실행
$ rustc a.rs -o a.out
$ ./a.out

# cargo로 빌드 및 실행
$ cargo new a
$ cd a
$ cargo run

# cargo 명령어
# 현재 디렉토리 빌드
$ cargo build
# 현재 디렉토리 실행
$ cargo run
# 현재 디렉토리 테스트
$ cargo test
# crate 설치
$ cargo install cargo-expand
# crate 사용
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
as - 기본 타입 캐스팅 수행, 특정 trait의 아이템 명확화, use 및 extern crate 문에서 아이템 이름 변경
async - 현재 스레드를 차단하는 대신 Future를 반환
await - Future의 결과가 준비될 때까지 실행 중단
break - 루프를 즉시 종료
const - 상수 아이템 또는 상수 raw 포인터 정의
continue - 다음 루프 반복으로 계속
crate - 외부 crate를 링크하거나 매크로가 정의된 crate를 나타내는 매크로 변수
dyn - trait 객체에 대한 동적 디스패치
else - if 및 if let 제어 흐름 구문의 대체
enum - 열거형 정의
extern - 외부 crate, 함수 또는 변수 링크
false - Boolean false 리터럴
fn - 함수 또는 함수 포인터 타입 정의
for - 반복자의 아이템에 대해 루프, trait 구현 또는 higher-ranked lifetime 지정
if - 조건식의 결과에 따라 분기
impl - 고유 또는 trait 기능 구현
in - for 루프 구문의 일부
let - 변수 바인딩
loop - 무조건 루프
match - 값을 패턴에 매치
mod - 모듈 정의
move - 클로저가 모든 캡처의 소유권을 갖도록 함
mut - 참조, raw 포인터 또는 패턴 바인딩에서 가변성 표시
pub - 구조체 필드, impl 블록 또는 모듈에서 공개 가시성 표시
ref - 참조로 바인딩
return - 함수에서 반환
Self - 정의하거나 구현하는 타입에 대한 타입 별칭
self - 메서드 주체 또는 현재 모듈
static - 전역 변수 또는 프로그램 실행 전체에 걸친 라이프타임
struct - 구조체 정의
super - 현재 모듈의 부모 모듈
trait - trait 정의
true - Boolean true 리터럴
type - 타입 별칭 또는 연관 타입 정의
union - 유니온 정의, union 선언에 사용될 때만 키워드
unsafe - 안전하지 않은 코드, 함수, trait 또는 구현 표시
use - 심볼을 스코프로 가져오기
where - 타입을 제약하는 절 표시
while - 표현식의 결과에 따라 조건부 루프
```

## Data Types

Rust는 정적 타입 언어입니다. 주요 데이터 타입은 다음과 같습니다:

### 스칼라 타입

* **정수형**: `i8`, `i16`, `i32`, `i64`, `i128`, `isize` (부호 있음)
* **부호 없는 정수형**: `u8`, `u16`, `u32`, `u64`, `u128`, `usize`
* **부동 소수점**: `f32`, `f64`
* **Boolean**: `bool` (true, false)
* **문자**: `char` (유니코드 스칼라 값)

### 복합 타입

* **튜플**: `(i32, f64, u8)`
* **배열**: `[i32; 5]` (고정 크기)

## Variables and Mutability

Rust는 **기본이 불변**입니다. TypeScript의 `const`가 기본인 셈입니다.

```rs
let x = 5;          // 불변 (기본값) — 값 변경 불가
let mut y = 10;     // mut 붙여야 변경 가능
y = 15;             // ✅ OK

const MAX_POINTS: u32 = 100_000;  // 상수 — 컴파일 타임에 확정, 타입 필수

// 섀도잉 — 같은 이름으로 새 변수 생성 (타입도 바꿀 수 있음)
let x = 5;
let x = x + 1;      // x = 6, 새 변수
let x = "hello";    // 타입까지 변경 가능!
```

| | `let` | `let mut` | `const` |
|---|---|---|---|
| 변경 가능 | ❌ | ✅ | ❌ |
| 섀도잉 가능 | ✅ | ✅ | ❌ |
| 타입 추론 | ✅ | ✅ | ❌ (명시 필수) |

## String Types

Rust에서 가장 혼란스러운 부분 중 하나입니다. 두 가지만 기억하세요:

| | `String` | `&str` |
|---|---|---|
| 소유권 | **소유** (힙에 할당) | **빌림** (참조) |
| 가변 | `let mut s` 이면 ✅ | ❌ 항상 불변 |
| 생성 | `String::from("hello")` | `"hello"` (리터럴) |
| 함수 매개변수 | 소유권 필요할 때 | **대부분 이걸 쓰세요** |

```rs
let s1: String = String::from("hello");  // 힙에 소유
let s2: &str = "world";                  // 스택의 참조
let s3: &str = &s1[0..2];               // String의 슬라이스 → &str
```

> **함수 매개변수에는 `&str`을 쓰세요.** `String`도 `&str`도 둘 다 받을 수 있습니다.

## String Conversions

```rs
// &str → String
let s = "hello".to_string();
let s = String::from("hello");

// String → &str
let s = String::from("hello");
let slice: &str = &s;

// 정수 → String
let num = 42;
let s = num.to_string();

// String → 정수
let s = "42";
let num: i32 = s.parse().unwrap();
```

## String Loops

```rs
// 문자 순회
for c in "안녕하세요".chars() {
    println!("{}", c);
}

// 바이트 순회
for b in "hello".bytes() {
    println!("{}", b);
}

// enumerate와 함께
for (i, c) in "hello".chars().enumerate() {
    println!("{}: {}", i, c);
}
```

## Formatted Print

```rs
println!("Hello, {}!", "world");
println!("{0} {1} {0}", "a", "b");  // a b a
println!("{name} {age}", name="John", age=30);
println!("{:?}", vec![1, 2, 3]);  // Debug 출력
println!("{:#?}", vec![1, 2, 3]);  // Pretty Debug
println!("{:b}", 10);  // 2진수: 1010
println!("{:x}", 255);  // 16진수: ff
println!("{:.2}", 3.14159);  // 3.14
```

## Control Flows

Rust의 제어 흐름에서 특이한 점: **`if`는 표현식**이라 값을 반환합니다 (삼항 연산자가 없는 대신).

```rs
// if 표현식 — 값을 반환
let x = if number < 10 { "smaller" } else { "bigger" };

// 루프 4종류
for i in 0..5 { println!("{}", i); }       // 범위 순회
for item in vec { println!("{}", item); }  // 컬렉션 순회
while count < 10 { count += 1; }           // 조건 반복
loop { break; }                            // 무한 반복 (break로 탈출)

// match — 모든 케이스를 처리해야 함 (exhaustive)
match number {
    1 => println!("one"),
    2 | 3 => println!("two or three"),   // OR 패턴
    4..=10 => println!("four to ten"),   // 범위 패턴
    _ => println!("something else"),     // 나머지 (default)
}

// if let — Option이나 enum에서 하나의 패턴만 처리할 때 match 대신 간결하게
if let Some(value) = optional_value {
    println!("{}", value);
}
```

| 루프 | 용도 | TypeScript 대응 |
|------|------|----------------|
| `for i in 0..5` | 범위 순회 | `for (let i=0; i<5; i++)` |
| `for item in vec` | 컬렉션 순회 | `for (const item of arr)` |
| `while cond` | 조건 반복 | `while (cond)` |
| `loop` | 무한 반복 (break로 탈출) | `while (true)` |

## Pattern Matching

`match`는 단순 값 비교를 넘어 **구조를 분해**할 수 있습니다. TypeScript의 `never`로 exhaustive check하던 것이 Rust에서는 **기본 동작**입니다.

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
    // 모든 variant를 처리해야 함! 하나라도 빠지면 컴파일 에러
}
```

## Collections

Rust의 3대 컬렉션:

| 컬렉션 | 타입 | TypeScript 대응 | 용도 |
|--------|------|----------------|------|
| **Vector** | `Vec<T>` | `Array` | 동적 배열 |
| **String** | `String` | `string` | UTF-8 문자열 |
| **HashMap** | `HashMap<K, V>` | `Map` | 키-값 저장 |

### Vec

```rs
let mut v: Vec<i32> = Vec::new();
v.push(1);
v.push(2);
let vec = vec![1, 2, 3];       // 매크로로 초기화

// 접근
let first = &vec[0];            // 패닉 가능
let first = vec.get(0);         // Option 반환 (안전)

// 순회
for val in &vec {
    println!("{}", val);
}

// 유용한 메서드
vec.len();                       // 길이
vec.is_empty();                  // 비어있는지
vec.contains(&2);                // 포함 여부
vec.iter().position(|&x| x == 2); // 인덱스 검색
vec.retain(|&x| x > 1);         // 조건에 맞는 것만 유지
vec.dedup();                     // 연속 중복 제거

// 슬라이스
let slice = &vec[1..3];         // &[i32]
```

### HashMap

```rs
use std::collections::HashMap;

let mut map = HashMap::new();
map.insert("key", 1);

// 접근
let val = map.get("key");          // Option<&V>
let val = map["key"];               // 패닉 가능

// 없으면 삽입
map.entry("key").or_insert(0);
*map.entry("key").or_insert(0) += 1;  // 카운터 패턴

// 순회
for (key, val) in &map {
    println!("{}: {}", key, val);
}

// 유용한 메서드
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
set.insert(2);                     // 중복 무시
println!("{}", set.len());          // 2

set.contains(&1);                   // true
set.remove(&1);

// 집합 연산
let a: HashSet<_> = [1, 2, 3].iter().collect();
let b: HashSet<_> = [2, 3, 4].iter().collect();
let union: HashSet<_> = a.union(&b).collect();
let intersection: HashSet<_> = a.intersection(&b).collect();
let diff: HashSet<_> = a.difference(&b).collect();
```

### BTreeMap / BTreeSet

정렬된 순서를 유지하는 컬렉션입니다. Java의 `TreeMap`/`TreeSet`에 대응합니다.

```rs
use std::collections::BTreeMap;

let mut map = BTreeMap::new();
map.insert(3, "c");
map.insert(1, "a");
map.insert(2, "b");

// 항상 키 순서대로 순회
for (k, v) in &map {
    println!("{}: {}", k, v);  // 1: a, 2: b, 3: c
}
```

### VecDeque

양쪽 끝에서 추가/제거할 수 있는 덱(deque)입니다. Java의 `ArrayDeque`에 대응합니다.

```rs
use std::collections::VecDeque;

let mut deque = VecDeque::new();
deque.push_back(1);      // 뒤에 추가
deque.push_front(0);     // 앞에 추가
deque.pop_back();        // 뒤에서 제거
deque.pop_front();       // 앞에서 제거
```

| Rust 컬렉션 | Java 대응 | TypeScript 대응 | 순서 보장 |
|-------------|----------|----------------|----------|
| `Vec<T>` | `ArrayList` | `Array` | 삽입 순서 |
| `HashMap<K,V>` | `HashMap` | `Map` | ❌ |
| `HashSet<T>` | `HashSet` | `Set` | ❌ |
| `BTreeMap<K,V>` | `TreeMap` | - | 키 정렬 |
| `BTreeSet<T>` | `TreeSet` | - | 값 정렬 |
| `VecDeque<T>` | `ArrayDeque` | - | 삽입 순서 |

## Collection Conversions

```rs
// Vec → HashSet
let vec = vec![1, 2, 2, 3];
let set: HashSet<_> = vec.into_iter().collect();

// HashSet → Vec
let set: HashSet<i32> = [1, 2, 3].iter().cloned().collect();
let vec: Vec<_> = set.into_iter().collect();

// Vec<&str> → Vec<String>
let strs = vec!["hello", "world"];
let strings: Vec<String> = strs.iter().map(|s| s.to_string()).collect();

// Iterator → Vec
let vec: Vec<i32> = (0..5).collect();                    // [0, 1, 2, 3, 4]
let vec: Vec<i32> = (0..5).filter(|x| x % 2 == 0).collect(); // [0, 2, 4]
```

> `collect()`는 Rust의 만능 변환기입니다. 대부분의 컬렉션 변환은 `.iter()` + 변환 + `.collect()`로 처리합니다.

## Sort

```rs
// Vec 정렬
let mut vec = vec![3, 1, 4, 1, 5];
vec.sort();                          // [1, 1, 3, 4, 5] — 오름차순
vec.sort_by(|a, b| b.cmp(a));       // [5, 4, 3, 1, 1] — 내림차순

// 키로 정렬
let mut people = vec![("Bob", 30), ("Alice", 25), ("Charlie", 35)];
people.sort_by_key(|p| p.1);        // 나이 오름차순

// 안정 정렬 vs 불안정 정렬
vec.sort();             // 안정 정렬 (같은 값의 순서 유지)
vec.sort_unstable();    // 불안정 정렬 (더 빠름)

// f64 정렬 (f64는 Ord가 아님 — NaN 때문)
let mut floats = vec![3.1, 1.2, 4.5];
floats.sort_by(|a, b| a.partial_cmp(b).unwrap());
```

> `sort()`는 **원본을 변경**합니다. 원본 유지하려면 `let sorted = { let mut v = vec.clone(); v.sort(); v };`

## Search

```rs
let vec = vec![1, 2, 3, 4, 5];

// 선형 검색
vec.contains(&3);                       // true
vec.iter().find(|&&x| x > 3);          // Some(&4)
vec.iter().position(|&x| x > 3);       // Some(3) — 인덱스

// 이진 검색 (정렬된 배열에서)
let sorted = vec![1, 2, 3, 4, 5];
sorted.binary_search(&3);              // Ok(2)  — 인덱스
sorted.binary_search(&6);              // Err(5) — 삽입 위치
```

| 메서드 | 반환값 | 용도 |
|--------|--------|------|
| `contains(&val)` | `bool` | 존재 여부만 |
| `iter().find(fn)` | `Option<&T>` | 조건으로 검색 |
| `iter().position(fn)` | `Option<usize>` | 조건으로 인덱스 |
| `binary_search(&val)` | `Result<usize, usize>` | 정렬된 배열에서 이진 검색 |

## Multi Dimensional Array

```rs
// 2D Vec 초기화
let rows = 3;
let cols = 4;
let grid = vec![vec![0; cols]; rows];  // 3x4 배열, 0으로 채움

// 순회
for i in 0..rows {
    for j in 0..cols {
        print!("{} ", grid[i][j]);
    }
    println!();
}

// 값이 있는 2D 초기화
let matrix = vec![
    vec![1, 2, 3],
    vec![4, 5, 6],
];
```

> Rust의 `vec![vec![0; cols]; rows]`는 TypeScript의 `Array.from()` 패턴처럼 각 행이 **독립적인 Vec**입니다 (참조 공유 문제 없음).

## Tuple

```rs
let tup: (i32, f64, u8) = (500, 6.4, 1);
let (x, y, z) = tup;  // 구조 분해
let first = tup.0;    // 인덱스 접근
```

## Array

```rs
// [u8]는 크기를 알 수 없는 u8 배열
fn foo(a: [u8]) {}

// [u8; 5]는 크기가 5인 u8 배열
fn foo(a: [u8; 5]) {}

// &[u8]는 크기를 알 수 없는 u8 배열 참조
fn foo(a: &[u8]) {}

// 배열 참조는 슬라이스
let a = [1, 2, 3, 4, 5];
foo(&a[1..3]);

// 값 0으로 1024 바이트 배열
let mut buf = [0; 1024];
```

## Struct

데이터를 **묶는** 구조입니다 (AND). `struct`로 정의하고 `impl`로 메서드를 구현합니다. **인스턴스 메서드**는 첫 번째 인자가 `self`입니다. **연관 함수**(정적 함수)는 첫 번째 인자가 `self`가 아닙니다.

```rs
pub struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    // 연관 함수
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    // 인스턴스 메서드
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

여러 가능성 중 **하나**를 나타냅니다 (OR). **enum**의 멤버를 **variant**라고 합니다. TypeScript의 `enum`과 달리, Rust의 `enum`은 **variant마다 다른 데이터**를 담을 수 있어 훨씬 강력합니다.

| | `struct` | `enum` |
|---|---|---|
| 역할 | 데이터 **묶기** (AND) | 여러 가능성 중 **하나** (OR) |
| TypeScript 대응 | `interface` | `\| (union type)` |
| 메서드 | `impl`로 추가 | `impl`로 추가 |

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

`use Message::*;`를 선언하면 `Message::`를 타이핑하지 않아도 됩니다.

## Functions

```rs
fn add(x: i32, y: i32) -> i32 {
    x + y       // 마지막 표현식이 반환값 (세미콜론 없음!)
}

// 제네릭 함수
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

| | 함수 `fn` | 클로저 `\|\|` |
|---|---|---|
| 타입 명시 | 필수 | 추론 가능 |
| 환경 캡처 | ❌ | ✅ 주변 변수 캡처 |
| `move` | 해당 없음 | 소유권 이동 가능 |

```rs
// 클로저 문법
let add_one = |x| x + 1;
let add_typed = |x: i32, y: i32| -> i32 { x + y };

// 환경 캡처 — 함수와의 핵심 차이
let x = 10;
let equal_to_x = |z| z == x;   // x를 캡처
println!("{}", equal_to_x(10)); // true

// move — 소유권을 클로저 안으로 이동 (스레드에서 필수)
let s = String::from("hello");
let closure = move || println!("{}", s);
// println!("{}", s);  // ❌ s는 클로저로 이동됨
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

// 여러 제네릭 타입
struct Point2<T, U> {
    x: T,
    y: U,
}
```

## Module System

모듈은 코드의 논리적 단위입니다. 관련된 코드들이 한 덩이로 모여있는 것입니다. `mod`를 사용하여 모듈을 정의하고, `use`를 사용하여 다른 모듈의 함수 등을 import합니다. 모듈 함수는 기본적으로 private입니다. `pub`을 사용하여 public으로 바꿀 수 있습니다.

```rs
mod sound {
    pub mod instrument {
        pub fn clarinet() {
            println!("clarinet");
        }
    }
}

fn main() {
    // 절대 경로
    crate::sound::instrument::clarinet();

    // use로 가져오기
    use crate::sound::instrument;
    instrument::clarinet();
}
```

모듈을 별도의 파일로 분리할 수 있습니다:

```bash
.
├── lib.rs
└── sound/
    ├── mod.rs
    └── instrument.rs
```

## Crate

Crate는 빌드된 바이너리의 단위입니다. 실행 파일 또는 라이브러리로 구분할 수 있습니다.

* **바이너리 crate**: 실행 파일을 생성 (`main.rs`)
* **라이브러리 crate**: 라이브러리를 생성 (`lib.rs`)

## Attributes

```rs
// Derive 속성
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Point {
    x: i32,
    y: i32,
}

// 조건부 컴파일
#[cfg(target_os = "linux")]
fn linux_only() {}

// 테스트
#[test]
fn test_add() {
    assert_eq!(2 + 2, 4);
}

// Deprecated
#[deprecated(since = "1.0.0", note = "Use new_function instead")]
fn old_function() {}
```

## Environment Variables

```rs
use std::env;

// 컴파일 타임에 환경 변수 가져오기
// 환경 변수가 없으면 빌드 실패
let cargo_dir = env!("CARGO_MANIFEST_DIR");

// 런타임에 환경 변수 가져오기
match env::var("HOME") {
    Ok(val) => println!("HOME: {}", val),
    Err(e) => println!("Error: {}", e),
}

// 기본값과 함께
let path = env::var("PATH").unwrap_or_else(|_| String::from("/usr/bin"));
```

# Core Concepts (핵심 개념)

Rust만의 고유한 개념들입니다. 다른 언어에는 없는 것들이라 처음에 어렵지만, 이것들을 이해하면 나머지는 수월합니다.

## Ownership and Borrowing

**Rust의 가장 핵심 개념**입니다. 다른 언어에는 없는 개념이라 처음에 어렵습니다.

### 소유권 규칙 (3가지만 기억)

1. 모든 값에는 **소유자**(변수)가 딱 하나
2. 소유자가 스코프를 벗어나면 값이 **자동 삭제**
3. 값을 다른 변수에 넘기면 **소유권이 이동** → 원래 변수 사용 불가

```rs
let s1 = String::from("hello");
let s2 = s1;          // 소유권이 s2로 이동
// println!("{}", s1);  // ❌ 에러! s1은 이미 죽었음
println!("{}", s2);    // ✅ OK
```

### 차용 (Borrowing) — 소유권 안 넘기고 빌려주기

```rs
let s = String::from("hello");

// 불변 차용 — &s = "빌려줄게, 읽기만 해"
let len = calculate_length(&s);
println!("{}", s);                // ✅ 아직 쓸 수 있음

// 가변 차용 — &mut s2 = "빌려줄게, 수정해도 돼"
let mut s2 = String::from("hello");
change(&mut s2);

fn calculate_length(s: &String) -> usize {
    s.len()
}

fn change(s: &mut String) {
    s.push_str(", world");
}
```

| | 불변 참조 `&T` | 가변 참조 `&mut T` |
|---|---|---|
| 동시에 몇 개? | **여러 개** OK | **하나만** |
| 읽기 | ✅ | ✅ |
| 쓰기 | ❌ | ✅ |
| 불변+가변 동시 | ❌ 불가 | ❌ 불가 |

> TypeScript에서는 `const obj = {a: 1}; obj.a = 2;`가 됩니다. Rust에서는 `let`이면 내부 값도 변경 불가. `let mut`이어야 합니다.

## Copy vs Clone

```rs
// Copy — 스택에 있는 작은 값은 자동 복사 (정수, bool, char 등)
let x = 5;
let y = x;        // x가 복사됨, 둘 다 사용 가능
println!("{} {}", x, y);  // ✅ OK

// Clone — 힙 데이터는 명시적으로 깊은 복사해야 함
let s1 = String::from("hello");
// let s2 = s1;         // 소유권 이동! s1 사용 불가
let s2 = s1.clone();    // 명시적 복사
println!("{} {}", s1, s2);  // ✅ 둘 다 OK
```

| | Copy | Clone |
|---|---|---|
| 방식 | 자동 (암묵적) | 명시적 `.clone()` |
| 비용 | 저렴 (스택 복사) | 비쌈 (힙 복사 가능) |
| 대상 | `i32, bool, char, f64` 등 | `String, Vec` 등 |

Copy를 구현하는 타입들:

* 모든 정수 타입: `u32` 등
* Boolean 타입: `bool`
* 모든 부동 소수점 타입: `f64` 등
* 문자 타입: `char`
* **튜플**: Copy 타입만 포함하는 경우. `(i32, i32)`는 Copy, `(i32, String)`은 아님

## Lifetime

"이 참조가 **얼마나 오래 유효한지**" 컴파일러에게 알려주는 것입니다. 목적은 댕글링 참조를 방지하는 것입니다.

```rs
// 'a = "x와 y 중 짧은 쪽의 수명만큼 반환값이 유효하다"
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}
```

> 대부분의 경우 컴파일러가 자동 추론합니다. 직접 쓸 일은 "함수가 참조를 받아서 참조를 반환할 때"뿐입니다.

## References and Pointers

* **참조**: `&T`, `&mut T`
* **Raw 포인터**: `*const T`, `*mut T` (unsafe 블록에서 사용)
* **스마트 포인터**: `Box<T>`, `Rc<T>`, `Arc<T>`, `RefCell<T>`

## Trait

TypeScript의 `interface`와 비슷하지만, **기존 타입에 나중에 구현 가능**합니다. `PartialOrd + Copy`는 **trait bound**입니다.

| | TypeScript `interface` | Rust `trait` |
|---|---|---|
| 기존 타입에 추가 구현 | ❌ | ✅ |
| 기본 구현 | ❌ | ✅ |
| 제네릭 제약 | `<T extends I>` | `<T: Trait>` |

```rs
trait Summary {
    fn summarize(&self) -> String;

    // 기본 구현
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

// where 절
fn notify<T>(item: &T)
where
    T: Summary + Display
{
    println!("{}", item.summarize());
}
```

## Box and Smart Pointers

Box는 특정 데이터를 힙에 보관하고 싶을 때 사용합니다. 크기가 큰 데이터가 스택에 보관되는 경우 소유권이 옮겨질 때 데이터 이동 시간이 길 수 있습니다.

```rs
fn main() {
    let b = Box::new(5);
    println!("b = {}", b);
}

// 재귀 타입
enum List {
    Cons(i32, Box<List>),
    Nil,
}
```

**주요 스마트 포인터**:

* **Box<T>**: 힙에 데이터 할당
* **Rc<T>**: 참조 카운팅 (단일 스레드)
* **Arc<T>**: 원자적 참조 카운팅 (멀티 스레드)
* **RefCell<T>**: 런타임 차용 규칙 검사
* **Mutex<T>**: 뮤텍스를 통한 데이터 보호

## Error Handling

Rust에는 `try/catch`가 **없습니다**. 대신 `Result`와 `Option`을 씁니다.

| 타입 | 의미 | TypeScript 대응 |
|------|------|----------------|
| `Result<T, E>` | 성공(`Ok(T)`) 또는 실패(`Err(E)`) | `Promise<T>` (resolve/reject) |
| `Option<T>` | 값 있음(`Some(T)`) 또는 없음(`None`) | `T \| undefined` |

> `?`는 "에러면 바로 return Err, 성공이면 값을 꺼내라"의 축약입니다. `unwrap()`은 프로덕션에서 쓰지 마세요 (패닉 발생).

```rs
// Result로 복구 가능한 에러 처리
fn read_file(path: &str) -> Result<String, std::io::Error> {
    std::fs::read_to_string(path)
}

// ? 연산자로 에러 전파
fn process_file() -> Result<(), std::io::Error> {
    let content = read_file("file.txt")?;
    println!("{}", content);
    Ok(())
}

// panic!으로 복구 불가능한 에러
fn divide(a: i32, b: i32) -> i32 {
    if b == 0 {
        panic!("Division by zero!");
    }
    a / b
}

// Option 타입
fn find_user(id: u32) -> Option<User> {
    // Some(user) 또는 None 반환
}
```

# Advanced

## Dispatch

디스패치에는 동적(dynamic)과 정적(static) 두 가지 종류가 있습니다.

```rs
// 동적 디스패치 (dyn 키워드)
fn process(item: &dyn Summary) {
    println!("{}", item.summarize());
}

// 정적 디스패치 (impl 키워드)
fn process(item: &impl Summary) {
    println!("{}", item.summarize());
}
```

정적 디스패치는 컴파일 타임에 구체적인 타입이 결정되어 더 빠르지만 바이너리 크기가 커집니다. 동적 디스패치는 런타임에 결정되어 유연하지만 약간 느립니다.

## Macros

```rs
// 선언적 매크로
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

// 절차적 매크로 (derive)
#[derive(Debug)]
struct MyStruct;
```

## Unsafe Rust

```rs
unsafe {
    // Raw 포인터 역참조
    let mut num = 5;
    let r1 = &num as *const i32;
    let r2 = &mut num as *mut i32;

    println!("r1: {}", *r1);

    // Unsafe 함수 호출
    unsafe fn dangerous() {}
    dangerous();
}
```

## Concurrency

```rs
use std::thread;
use std::sync::{Arc, Mutex};

fn main() {
    // 스레드 생성
    let handle = thread::spawn(|| {
        println!("Hello from thread!");
    });

    handle.join().unwrap();

    // 공유 상태
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

```rs
async fn fetch_data() -> Result<String, Error> {
    // 비동기 작업
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

Tokio는 Rust의 비동기 런타임 프레임워크입니다. async/await 구문으로 비동기 애플리케이션을 작성하기 위한 빌딩 블록을 제공합니다.

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

Rust는 소유권 시스템을 통해 컴파일 타임에 메모리 안전성을 보장합니다.

* **스택**: 고정 크기 데이터 (정수, 부울, 고정 배열 등)
* **힙**: 동적 크기 데이터 (String, Vec, Box 등)

```rs
fn main() {
    // 스택 할당
    let x = 5;

    // 힙 할당
    let s = String::from("hello");
    let v = vec![1, 2, 3];
    let b = Box::new(5);
}  // 스코프 종료 시 자동으로 메모리 해제
```

## Performance Optimization

* **제로 비용 추상화**: 추상화가 런타임 비용을 추가하지 않음
* **반복자 사용**: 루프보다 반복자가 더 잘 최적화됨
* **인라인화**: `#[inline]` 속성 사용
* **프로파일링**: `cargo flamegraph`, `perf` 등 사용
* **벤치마킹**: `criterion` crate 사용

```rs
// 반복자 체인 (최적화됨)
let sum: i32 = (0..100)
    .filter(|x| x % 2 == 0)
    .map(|x| x * 2)
    .sum();

// 벤치마크
#[bench]
fn bench_function(b: &mut Bencher) {
    b.iter(|| {
        // 벤치마크할 코드
    });
}
```

# Style Guide

[Rust 스타일 가이드 (한국어)](rust_style_guide-kr.md)

# Refactoring

[Rust 리팩토링 (한국어)](rust_refactoring-kr.md)

# Effective Rust

[Effective Rust (한국어)](effective_rust-kr.md)

# Rust Design Patterns

[Rust 디자인 패턴 (한국어)](rust_design_pattern-kr.md)

# Rust Architecture

[Rust 아키텍처 (한국어)](rust_architecture-kr.md)
