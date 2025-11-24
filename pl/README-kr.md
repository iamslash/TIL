- [Abstract](#abstract)
- [Material](#material)
- [Programming Language Features](#programming-language-features)
  - [Install](#install)
  - [Hello World](#hello-world)
  - [Build, Run](#build-run)
  - [Reserved Words](#reserved-words)
  - [Data Types](#data-types)
  - [Min Max Values](#min-max-values)
  - [Min Max](#min-max)
  - [Abs FAbs](#abs-fabs)
  - [Bit Manipulation](#bit-manipulation)
  - [Random](#random)
  - [String](#string)
  - [Formatted String](#formatted-string)
  - [Enumerations](#enumerations)
  - [Multidimensional Array](#multidimensional-array)
  - [Control Flows](#control-flows)
  - [Loops](#loops)
  - [Operators](#operators)
  - [Collections compared to c++ containers](#collections-compared-to-c-containers)
  - [Collections](#collections)
  - [Functions](#functions)
  - [Type Conversions](#type-conversions)
  - [Sort](#sort)
  - [Search](#search)
  - [Struct, Class, Interface, AbstractClass](#struct-class-interface-abstractclass)
  - [Closure](#closure)
  - [Lambda](#lambda)
  - [Exception](#exception)
  - [Concurrency](#concurrency)
  - [Memory Layout](#memory-layout)
  - [Runtime](#runtime)
  - [Style Guide](#style-guide)
  - [Refactoring](#refactoring)
  - [Effective](#effective)
  - [Design Pattern](#design-pattern)
  - [Structure of Project (Architecture)](#structure-of-project-architecture)

----

# Abstract

프로그래밍 언어를 새로 배울 때 학습해야할 내용을 정리한다.

# Material

* [프로그래밍언어 by 이광근](https://ropas.snu.ac.kr/~kwang/4190.310/mooc/)
  * 서울대학교 프로그래밍 언어 동영상 강좌
* [streem @ github](https://github.com/matz/streem)
  * ruby 제작자 마츠모토가 만든 streem 언어 
* [lex & yacc](https://wiki.kldp.org/KoreanDoc/html/Lex_Yacc-KLDP/Lex_Yacc-KLDP.html#toc1)
  * programming language 제작도구인 lex, yacc 튜토리얼

# Programming Language Features

## Install

  * Install

## Hello World

  * Print out "Hello World"

## Build, Run

  * 컴파일하고 실행하는 방법
  * 빌드 도구 (make, cmake, gradle, maven 등)
  * 패키지 매니저 (npm, pip, cargo 등)
  * IDE 설정 및 실행 환경

## Reserved Words

  * 예약어
  * 키워드 목록
  * 문법 규칙

## Data Types

  * 기본 타입 (int, float, string, boolean)
  * 복합 타입 (array, struct, class)
  * 포인터와 참조
  * 타입 시스템 (정적/동적 타입)
  * 타입 추론
  * 제네릭/템플릿

## Min Max Values

## Min Max

## Abs FAbs

## Bit Manipulation

## Random

## String

## Formatted String

  * 문자열 처리
  * 정규표현식
  * 텍스트 파싱
  * 인코딩 (UTF-8, ASCII 등)

## Enumerations

## Multidimensional Array

  * 2d array
  * 3d array

## Control Flows

  * 조건문 (if-else, switch-case)
  * 분기문 (break, continue, return)
  * 예외 처리 (try-catch, throw)

## Loops

  * 반복문 (for, while, do-while)
  * 반복자 (iterator, foreach)
  * 재귀 함수

## Operators

  * 산술 연산자 (+, -, *, /, %)
  * 비교 연산자 (==, !=, <, >, <=, >=)
  * 논리 연산자 (&&, ||, !)
  * 비트 연산자 (&, |, ^, ~, <<, >>)
  * 할당 연산자 (=, +=, -= 등)
  * 연산자 우선순위

## Collections compared to c++ containers

| c++                  | java                            |
|:---------------------|:--------------------------------|
| `if, else`           | `if, else`                      |
| `for, while`         | `for, while`                    |
| `array`              | `Collections.unmodifiableList`  |
| `vector`             | `Vector, ArrayList`             |
| `deque`              | `Deque, ArrayDeque`             |
| `forward_list`       | `LinkedList`                    |
| `list`               | `List, LinkedList`              |
| `stack`              | `Stack, Deque`                  |
| `queue`              | `Queue, LinkedList`             |
| `priority_queue`     | `Queue, PriorityQueue`          |
| `set`                | `SortedSet, TreeSet`            |
| `multiset`           | `Multiset (Guava)`              |
| `map`                | `SortedMap, TreeMap`            |
| `multimap`           | `Multimap (Guava)`              |
| `unordered_set`      | `Set, HashSet`                  |
| `unordered_multiset` | `Multiset (Guava)`              |
| `unordered_map`      | `Map, HashMap`                  |
| `unordered_multimap` | `Multimap (Guava)`              |

## Collections

  * 배열과 리스트
  * 스택과 큐
  * 해시 테이블
  * 트리와 그래프
  * 컬렉션 프레임워크
  * 반복자 패턴

## Functions

  * 함수 정의와 호출
  * 매개변수 (값 전달, 참조 전달)
  * 반환값
  * 함수 오버로딩
  * 가변 인수
  * 고차 함수

## Type Conversions

  * integer to string
  * string to integer
  * integer to float

## Sort

## Search

## Struct, Class, Interface, AbstractClass

  * 구조체와 클래스
  * 상속과 다형성
  * 인터페이스와 추상 클래스
  * 캡슐화, 상속, 다형성
  * 접근 제어자
  * 생성자와 소멸자

## Closure

A closure is a function value that references variables from outside its body.

  * 렉시컬 스코프
  * 자유 변수
  * 클로저의 활용
  * 메모리 관리

## Lambda

  * 익명 함수
  * 함수형 프로그래밍
  * 람다 표현식
  * 고차 함수와의 연동

## Exception

  * 예외 처리 메커니즘
  * try-catch-finally
  * 예외 타입과 계층
  * 커스텀 예외
  * 예외 전파

## Concurrency

  * [스레드와 프로세스](pl-thread-process-kr.md)
  * [동기화 (mutex, semaphore)](pl-synchronization-kr.md)
  * [데드락과 경쟁 상태](pl-deadlock-racecondition-kr.md)
  * [비동기 프로그래밍](pl-async-programming-kr.md)
  * [coroutine](pl-coroutine-kr.md)
  * [동시성 제어 패턴](pl-concurrency-patterns-kr.md)

## Memory Layout

  * os memory layout
  * application memory layout

## Runtime

  * runtime logic

## Style Guide

  * 코딩 컨벤션
  * 네이밍 규칙
  * 주석 작성법
  * 코드 포맷팅
  * 린터와 포맷터

## Refactoring

  * 코드 리팩토링 기법
  * 리팩토링 도구
  * 코드 품질 개선
  * 기술 부채 관리

## Effective

  * 효율적인 프로그래밍 기법
  * 성능 최적화
  * 메모리 관리
  * 알고리즘과 자료구조
  * 디버깅과 프로파일링

## Design Pattern

  * 생성 패턴 (Singleton, Factory 등)
  * 구조 패턴 (Adapter, Bridge 등)
  * 행동 패턴 (Observer, Strategy 등)
  * 아키텍처 패턴 (MVC, MVP 등)
  * 안티 패턴과 해결책

## Structure of Project (Architecture)

  * 프로젝트 구조
  * 모듈과 패키지
  * 의존성 관리
  * 빌드 시스템
  * 테스트 구조
  