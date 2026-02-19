# TypeScript (한국어)

**영문 버전**: 영문 문서는 [README.md](README.md)를 참조하세요.

- [학습 자료 (Resources)](#학습-자료-resources)
- [기초 (Basics)](#기초-basics)
  - [빌드 및 실행 (Build & Run)](#빌드-및-실행-build--run)
  - [출력하기 (Print)](#출력하기-print)
  - [예약어 (Reserved Words)](#예약어-reserved-words)
  - [최소값, 최대값 (Min, Max)](#최소값-최대값-min-max)
  - [abs vs fabs](#abs-vs-fabs)
  - [비트 연산 (Bitwise Operations)](#비트-연산-bitwise-operations)
  - [문자열 (String)](#문자열-string)
  - [난수 생성 (Random)](#난수-생성-random)
  - [포맷된 문자열 (Formatted String)](#포맷된-문자열-formatted-string)
  - [검사하기 (Inspect)](#검사하기-inspect)
  - [데이터 타입 (Data Types)](#데이터-타입-data-types)
    - [undefined vs unknown vs any vs never 비교](#undefined-vs-unknown-vs-any-vs-never-비교)
  - [제어 흐름문 (Control Flow)](#제어-흐름문-control-flow)
    - [조건문 (Conditionals)](#조건문-conditionals)
    - [반복문 (Loops)](#반복문-loops)
    - [for...of vs for...in](#forof-vs-forin)
    - [var vs let 스코프 (Scope)](#var-vs-let-스코프-scope)
  - [컬렉션 (Collections)](#컬렉션-collections)
    - [튜플 (Tuple)](#튜플-tuple)
    - [배열 (Array)](#배열-array)
    - [집합 (Set)](#집합-set)
    - [맵 (Map)](#맵-map)
  - [컬렉션 변환 (Collection Conversion)](#컬렉션-변환-collection-conversion)
  - [정렬 (Sort)](#정렬-sort)
  - [검색 (Search)](#검색-search)
  - [다차원 배열 (Multidimensional Array)](#다차원-배열-multidimensional-array)
  - [열거형 (Enum)](#열거형-enum)
  - [제네릭 (Generics)](#제네릭-generics)
  - [같은 줄에 여러 변수 정의하기 (Multiple Variables)](#같은-줄에-여러-변수-정의하기-multiple-variables)
- [고급 (Advanced)](#고급-advanced)
  - [Map vs Record](#map-vs-record)
  - [유틸리티 타입 (Utility Types)](#유틸리티-타입-utility-types)
  - [삼중 점 연산자 (Spread/Rest Operator)](#삼중-점-연산자-spreadrest-operator)
  - [널 병합 연산자 (||), 이중 물음표 (??) (Nullish Coalescing)](#널-병합-연산자--이중-물음표--nullish-coalescing)
  - [export와 import (Export & Import)](#export와-import-export--import)
  - [`declare`](#declare)
  - [인터페이스를 사용한 함수 정의 (Function Types with Interface)](#인터페이스를-사용한-함수-정의-function-types-with-interface)
  - [Interface vs Type](#interface-vs-type)
  - [Optional (선택적 매개변수와 속성)](#optional-선택적-매개변수와-속성)
- [스타일 가이드 (Style Guide)](#스타일-가이드-style-guide)
- [리팩토링 (Refactoring)](#리팩토링-refactoring)
- [효율적인 TypeScript (Effective TypeScript)](#효율적인-typescript-effective-typescript)
- [디자인 패턴 (Design Patterns)](#디자인-패턴-design-patterns)
- [아키텍처 (Architecture)](#아키텍처-architecture)

----

# 학습 자료 (Resources)

* [한눈에 보는 타입스크립트(updated)](https://heropy.blog/2020/01/27/typescript/)
* [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html)
  * 반드시 읽어야 할 공식 문서
  * [TypeScript Handbook 한글](https://typescript-kr.github.io/)
* [8장. 리액트 프로젝트에서 타입스크립트 사용하기](https://react.vlpt.us/using-typescript/)
* [TypeScript 환경에서 Redux를 프로처럼 사용하기 @ velog](https://velog.io/@velopert/use-typescript-and-redux-like-a-pro)
* [playcode](https://playcode.io/)
  * TypeScript 플레이그라운드
* [TypeScript의 소개와 개발 환경 구축](https://poiemaweb.com/typescript-introduction)

# 기초 (Basics)

## 빌드 및 실행 (Build & Run)

```bash
# tsc 설치
$ npm install -g typescript
# tsc로 빌드
$ tsc a.ts
# node로 실행
$ node a.js

# ts-node 설치
$ npm install -g ts-node
# ts-node로 실행
$ ts-node a.ts

# REPL
$ ts-node
> let a = 2
> a
2
```

## 출력하기 (Print)

```typescript
function greet(person: string, date: Date) {
  console.log(`Hello ${person}, today is ${date.toDateString()}!`);
}

greet("Maddison", new Date());
```

## 예약어 (Reserved Words)

* [types.ts | github](https://github.com/Microsoft/TypeScript/blob/fad889283e710ee947e8412e173d2c050107a3c1/src/compiler/types.ts#L87)

```ts
// 예약어
break case catch class const
continue debugger default delete
do else enum export extends false
finally for function if import in
instanceof new null return super
switch this throw true try typeof
var void while with

// 엄격 모드 예약어
as implements interface let package
private protected public static yield

// 문맥 키워드
any boolean constructor declare get
module require number set string symbol
type from of
```

## 최소값, 최대값 (Min, Max)

```ts
console.log(Number.MAX_SAFE_INTEGER);   // 9007199254740991
console.log(Number.MIN_SAFE_INTEGER);   // -9007199254740991
console.log(Number.MAX_VALUE);  // 1.7976931348623157e+308
console.log(Number.MIN_VALUE);  // 5e-324

console.log(Number.MAX_SAFE_INTEGER + 1);   // 9007199254740992
console.log(Number.MAX_SAFE_INTEGER + 2);   // 9007199254740992
console.log(Number.MAX_SAFE_INTEGER + 3);   // 9007199254740994
```

## abs vs fabs

```ts
function difference(a, b) {
  return Math.abs(a - b);
}
console.log(difference(3, 5));
// 예상 출력: 2
console.log(difference(5, 3));
// 예상 출력: 2
console.log(difference(1.23456, 7.89012));
// 예상 출력: 6.6555599999999995
```

## 비트 연산 (Bitwise Operations)

```ts
var a: number = 2;  // 10
var b: number = 3;  // 11
var result;

// (a & b) =>  2
console.log("(a & b) => ", a & b);

// (a | b) =>  3
console.log("(a | b) => ", a | b);

// (a ^ b) =>  1
console.log("(a ^ b) => ", a ^ b);

// (~b) =>  -4
console.log("(~b) => ", ~b);

// (a << b) =>  16
console.log("(a << b) => ", a << b);

// (a >> b) =>  0
console.log("(a >> b) => ", a >> b);
```

## 문자열 (String)

```ts
// 보간된 문자열
type World = "world";
type Greeting = `hello ${World}`;
console.log(Greeting);  // hello world

// 문자열 순회
let s = "hello world"
let n = s.length
for (let i = 0; i < n; ++i) {
    let c = s.charAt(i);
    console.log(c);
}

// 문자를 숫자로, 숫자를 문자로 변환
// https://stackoverflow.com/questions/22624379/how-to-convert-letters-to-numbers-with-javascript
let s = "abcdefg";
let c = s.charAt(0);   // c는 문자열, "a"
// 97은 "a"를 의미
let i = c.charCodeAt(0) - 97;  // 0
let d = String.fromCharCode(97 + i) // "a"
```

## 난수 생성 (Random)

* [Math.random() | mdn](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Math/random)

`[0..1]` 범위의 값을 생성합니다.

```ts
console.log(Math.random());
console.log(Math.random());
```

## 포맷된 문자열 (Formatted String)

* [util.format | node.js](https://nodejs.org/api/util.html#utilformatformat-args)

TypeScript의 기본 타입은 `boolean, number, string`임을 기억하세요.

```ts
import util from "util";

console.log(util.format('%s %d %s', true, 4, 'Hello World'));

// 줄바꿈 없이 출력
process.stdout.write(`i: ${i}, diff: ${diff} has: ${numMap.has(diff)} `);
console.log(numMap);
```

## 검사하기 (Inspect)

* [util.inspect | node.js](https://nodejs.org/api/util.html#utilinspectobject-options)

```ts
import util from "util";

class Foo {
  get [Symbol.toStringTag]() {
    return 'bar';
  }
}
class Bar {}
const baz = Object.create(null, { [Symbol.toStringTag]: { value: 'foo' } });
console.log(util.inspect(new Foo())); // 'Foo [bar] {}'
console.log(util.inspect(new Bar())); // 'Bar {}'
console.log(util.inspect(baz));       // '[foo] {}'
```

```console
$ ts-node
> console. <TAB>
```

## 데이터 타입 (Data Types)

```ts
boolean
number
string
array
tuple
enum
unkown
any
void
null
undefined
never
object

// boolean
let isDone: boolean = false;

// number
let decimal: number = 6;
let hex: number = 0xf00d;
let binary: number = 0b1010;
let octal: number = 0o744;
let big: bigint = 100n;

// string
let color: string = "blue";
color = 'red';

// Array
// 값의 배열
let list1: number[] = [1, 2, 3];
// 제네릭 배열
let list2: Array<number> = [1, 2, 3];

// Tuple
let x: [string, number];
x = ["hello", 10];

// Enum
enum Color {
    Red, Green, Blue,
}
let c: Color = Color.Green;

// Unknown
let notSure: unknown = 4;
notSure = "Maybe a string instead";
notSure = false;

// Any
declare function getValue(key: string): any;
const std: string = getValue("David");

// Void
function warnUser(): void {
    console.log("Hello World");
}

// Null과 Undefined
let u: undefined = undefined;
let n: null = null;

// Never
function error(msg: string): never {
    throw new Error(msg);
}
function fail() {
    return error("something failed");
}
function infiniteLoop(): never {
    while (true) {}
}

// Object
declare function create(o: object | null): void;
create({ prop: 0 });
create(null);
create(undefined);

// 타입 단언
let someValue: unknown = "This is a string";
let strLength: number = (someValue as string).length;
let strLength2: number = (<string>someValue).length;
```

### undefined vs unknown vs any vs never 비교

이 네 가지 특수 타입은 역할이 명확히 다릅니다.

| 타입 | 한 줄 요약 | 핵심 |
|------|-----------|------|
| `undefined` | 값이 **아직 없다** | 빈 상태를 나타내는 JavaScript 기본값 |
| `unknown` | 값이 **뭔지 모른다** (확인하고 써라) | `any`의 안전한 버전 |
| `any` | 값이 **뭐든 상관없다** (검사 포기) | 타입 안전성 없음, 비추천 |
| `never` | 값이 **존재할 수 없다** | 함수가 절대 정상 반환하지 않음 |

#### undefined — "아직 안 넣었어"

```ts
let name: string;
console.log(name);          // undefined — 값을 안 넣었으니까

function greet(name?: string) {
    console.log(name);      // 안 넘기면 undefined
}
greet();                    // undefined

const arr = [1, 2, 3];
console.log(arr[10]);       // undefined — 범위 밖
```

#### unknown — "뭔지 모르니까 확인하고 써라"

`any`처럼 아무 값이나 담을 수 있지만, **타입 확인 전에는 사용 불가**합니다.
외부 API 응답, `JSON.parse`, `catch`의 error 처리에 적합합니다.

```ts
let value: unknown = "hello";

value.toUpperCase();            // ❌ 컴파일 에러 — 바로 못 씀
(value as string).toUpperCase(); // ✅ 타입 단언 후 OK

if (typeof value === "string") {
    value.toUpperCase();        // ✅ typeof 확인 후 자동 OK
}

// 실전: catch에서 error 처리
try {
    something();
} catch (err: unknown) {
    // err.message;              // ❌ 바로 못 씀
    if (err instanceof Error) {
        console.log(err.message); // ✅ 확인 후 사용
    }
}
```

#### any vs unknown

```ts
// any: 아무거나 해도 에러 안 남 (위험!)
let a: any = "hello";
a.foo.bar.baz;          // ✅ 컴파일 통과 — 런타임에 터짐 💥

// unknown: 확인 전엔 아무것도 못 함 (안전!)
let b: unknown = "hello";
b.foo.bar.baz;          // ❌ 컴파일 에러 — 런타임 전에 잡아줌
```

> **`any`를 쓰고 싶다면 `unknown`을 쓰세요.** `any`는 타입 검사를 완전히 무력화합니다.

#### never — "이런 상황은 절대 발생하지 않는다"

함수가 절대 정상 반환하지 않거나, 모든 케이스를 처리했는지 검증할 때 사용합니다.

```ts
// 1. 항상 예외를 던지는 함수
function fail(msg: string): never {
    throw new Error(msg);
}

// 2. Exhaustive check — 케이스 빠뜨림을 컴파일 타임에 방지
type Shape = "circle" | "square" | "triangle";

function getArea(shape: Shape): number {
    switch (shape) {
        case "circle":   return 3.14 * 10 * 10;
        case "square":   return 10 * 10;
        case "triangle": return (10 * 5) / 2;
        default:
            const _exhaustive: never = shape;  // 모든 케이스 처리 시 여기 도달 불가
            throw new Error(`Unknown shape: ${_exhaustive}`);
    }
}
// 나중에 "pentagon"을 Shape에 추가하면, case를 안 넣으면 컴파일 에러 발생!
```

## 제어 흐름문 (Control Flow)

### 조건문 (Conditionals)

```ts
// if ... else if ... else
let num: number = 1;
if (num > 0) {
    console.log("positive");
} else if (num < 0) {
    console.log("negative");
} else {
    console.log("zero");
}

// switch
let grade: string = "A";
switch (grade) {
    case "A": {
        console.log("Excellent");
        break;
    }
    case "B": {
        console.log("Good");
        break;
    }
    default: {
        console.log("Invalid choice");
        break;
    }
}
```

### 반복문 (Loops)

```ts
// for
for (let i = 0; i < 3; i++) {
    console.log(i);  // 0, 1, 2
}

// while
let i = 0;
while (i < 3) {
    console.log(i);  // 0, 1, 2
    i++;
}
```

### for...of vs for...in

가장 많이 헷갈리는 부분입니다.

```ts
let arr = [10, 20, 30];

// for...of → "값"을 꺼냄
for (let val of arr) {
    console.log(val);   // 10, 20, 30
}

// for...in → "인덱스(키)"를 꺼냄
for (let idx in arr) {
    console.log(idx);   // "0", "1", "2"  (문자열!)
}

// 문자열에도 for...of 사용 가능
for (let chr of "Hello") {
    console.log(chr);   // H, e, l, l, o
}
```

| | `for...of` | `for...in` |
|---|---|---|
| 꺼내는 것 | **값** | **키(인덱스)** |
| 배열에 쓰면 | `10, 20, 30` | `"0", "1", "2"` |
| 타입 | 원래 타입 | **항상 string** |

> 배열 순회에는 **`for...of`를 쓰세요.** `for...in`은 객체의 키를 순회할 때 씁니다.

### var vs let 스코프 (Scope)

```ts
// var: 루프 밖에서도 살아있음 (function scope)
for (var i in [1, 2, 3]) {}
console.log(i);   // "2" — 접근 가능!

// let: 루프 안에서만 존재 (block scope)
for (let j in [1, 2, 3]) {}
console.log(j);   // ❌ 에러 — 접근 불가
```

> **항상 `let`을 쓰세요.** `var`는 의도치 않게 변수가 살아남아 버그를 만듭니다.

## 컬렉션 (Collections)

### 튜플 (Tuple)

배열처럼 생겼지만 **각 위치의 타입이 고정**됩니다.

```ts
let employee: [number, string] = [1, 'David'];
employee[0] = "hello";  // ❌ 에러 — 0번은 number여야 함
employee[1] = 42;        // ❌ 에러 — 1번은 string이어야 함

let person: [number, string, boolean] = [1, 'David', true];
let employees: [number, string][] = [[1, 'David'], [2, 'Tom']];

// push/pop도 가능
employee.push(2, 'John');
console.log(employee);  // [1, 'David', 2, 'John']
```

**언제 쓰나?** 함수에서 여러 값을 반환할 때 유용합니다:

```ts
function getUser(): [number, string] {
    return [1, "David"];
}
const [id, name] = getUser();  // 구조 분해로 깔끔하게 받기
```

> 3개 이상이면 튜플보다 **인터페이스/객체**가 읽기 좋습니다.

**튜플 vs 배열 — 값은 똑같고, 타입 선언만 다르다**

값 자체는 완전히 똑같이 생겼습니다. 런타임(JavaScript)에서는 **둘 다 그냥 배열**입니다. 차이는 오직 TypeScript 컴파일러가 타입을 검사할 때만 존재합니다.

```ts
// 배열 — 같은 타입의 "임의 개수"
let arr: number[]         = [1, 2];       // number가 몇 개든 OK
arr.push(3);       // ✅ OK
arr = [1];         // ✅ OK

// 튜플 — 각 위치의 타입과 "개수가 고정"
let tup: [number, string] = [1, "David"]; // 정확히 number, string 순서
tup = [1];         // ❌ 에러 — string이 빠짐
tup = [1, 2];      // ❌ 에러 — 두 번째는 string이어야 함

// 런타임에서는 완전히 동일
console.log(Array.isArray(tup));  // true — 튜플도 배열!
```

| | 타입 선언 | 값 모습 | 런타임 |
|---|---|---|---|
| 배열 | `number[]` | `[1, 2, 3]` | Array |
| 튜플 | `[number, string]` | `[1, "David"]` | Array (동일!) |

> 튜플은 "위치별 타입을 컴파일러가 강제하는 배열"이라고 생각하면 됩니다.

### 배열 (Array)

```ts
// 선언
let fruits: Array<string> = ['Apple', 'Orange', 'Banana'];
let numbers: number[] = [1, 2, 3, 4];
let vals: (string | number)[] = ['Apple', 2, 'Orange', 3];

// 초기화
let filled = new Array<number>(5).fill(-1);  // [-1, -1, -1, -1, -1]
```

배열 메서드를 용도별로 정리하면 외우기 쉽습니다:

| 목적 | 메서드 | 원본 변경? |
|------|--------|-----------|
| **변환** | `map`, `flatMap` | No (새 배열) |
| **필터** | `filter` | No (새 배열) |
| **축약** | `reduce` | No (단일 값) |
| **검색** | `find`, `findIndex`, `indexOf`, `includes` | No |
| **검증** | `every`, `some` | No |
| **정렬** | `sort`, `reverse` | **Yes (원본 변경!)** |
| **추가/제거 (뒤)** | `push`, `pop` | Yes |
| **추가/제거 (앞)** | `unshift`, `shift` | Yes |
| **잘라내기** | `splice` | Yes |
| **복사해서 자르기** | `slice` | No (새 배열) |

```ts
const numbers = [1, 2, 3, 4];

// 변환: map — 각 요소를 변환해서 새 배열 반환
numbers.map(n => n * 2);                    // [2, 4, 6, 8]

// 필터: filter — 조건에 맞는 요소만 모아 새 배열 반환
numbers.filter(n => n % 2 === 0);           // [2, 4]

// 축약: reduce — 배열을 단일 값으로 축약
numbers.reduce((acc, cur) => acc + cur, 0); // 10

// 검색: find, findIndex, includes
numbers.find(n => n > 2);                   // 3
numbers.findIndex(n => n > 2);              // 2
numbers.includes(3);                        // true

// 검증: every, some
numbers.every(n => n % 2 === 0);            // false (전부 짝수?)
numbers.some(n => n % 2 !== 0);             // true  (홀수 있나?)

// flat, flatMap — 다차원 배열 평탄화
[1, [2, 3], [4, [5]]].flat(2);             // [1, 2, 3, 4, 5]
["hello", "world"].flatMap(s => s.split('')); // ['h','e','l','l','o','w','o','r','l','d']
```

**헷갈리기 쉬운 메서드:**

```ts
// splice vs slice
const arr = [1, 2, 3, 4, 5];
arr.slice(1, 3);     // [2, 3]        — 원본 그대로, 복사본 반환
arr.splice(1, 2);    // [2, 3] 제거됨  — 원본이 [1, 4, 5]로 변경!

// sort 주의! 기본은 "사전순"
[1, 10, 2].sort();                // [1, 10, 2] — 문자열 비교!
[1, 10, 2].sort((a, b) => a - b); // [1, 2, 10] — 숫자 비교

// push/pop (뒤) vs unshift/shift (앞)
const stack = [1, 2];
stack.push(3);    // [1, 2, 3]  — 뒤에 추가
stack.pop();      // [1, 2]     — 뒤에서 제거
stack.unshift(0); // [0, 1, 2]  — 앞에 추가
stack.shift();    // [1, 2]     — 앞에서 제거

// reverse — 원본 변경! 복사하려면 spread
const reversed = [...stack].reverse();  // stack은 그대로
```

### 집합 (Set)

**중복 없는** 값의 모음입니다. `has()` 검색이 `O(1)`로 배열의 `includes()`보다 빠릅니다.

```ts
let dirs = new Set<string>(['east', 'west']);
dirs.add('north');
dirs.add('east');          // 중복 무시
console.log(dirs.size);    // 3 (east, west, north)
console.log(dirs.has('east'));   // true
dirs.delete('east');       // true
dirs.clear();              // 전부 삭제

// 순회
for (let dir of dirs) {
    console.log(dir);
}
```

**실전 패턴: 배열 중복 제거**

```ts
const arr = [1, 2, 2, 3, 3, 3];
const unique = [...new Set(arr)];  // [1, 2, 3]
```

> **배열 vs Set 판단 기준:** "이 값이 있나?" 검색이 잦으면 **Set**, 순서/인덱스가 중요하면 **배열**.

### 맵 (Map)

키-값 쌍 저장. 일반 객체 `{}`와 비슷하지만 차이가 있습니다.

```ts
let map = new Map<string, number>();
map.set('David', 10);
map.set('John', 20);

console.log(map.get('David'));      // 10
console.log(map.get('Tom'));        // undefined
console.log(map.get('Tom') || 0);   // 0 (기본값 패턴)
console.log(map.has('David'));      // true
console.log(map.size);              // 2
map.delete('John');

// 초기화와 동시에 생성
let config = new Map<string, string>([
    ['host', 'localhost'],
    ['port', '3000']
]);

// 순회
for (let [key, val] of map) {
    console.log(key, val);
}
// for...of로 keys(), values(), entries() 도 사용 가능
```

**Map vs 일반 객체 `{}`:**

| | `Map` | `{}` (객체) |
|---|---|---|
| 키 타입 | **아무 타입** (객체, 함수도 OK) | string / symbol만 |
| 순서 보장 | 삽입 순서 보장 | ES2015+ 부분 보장 |
| 크기 확인 | `map.size` | `Object.keys(obj).length` |
| 성능 | 추가/삭제 잦으면 **빠름** | 고정 구조면 빠름 |

> **판단 기준:** 키가 동적으로 바뀌면 **Map**, 구조가 고정이면 **객체/인터페이스**.

## 컬렉션 변환 (Collection Conversion)

```ts
// 튜플을 집합으로
let arr = [11, 22, 33];
let set = new Set(arr);
console.log(set);  // Set(3) { 11, 22, 33 }
```

## 정렬 (Sort)

```ts
let arr: number[] = [1, 10, 2, 5, 3];
console.log(arr);  // [1, 10, 2, 5, 3]

// 사전순 정렬
arr.sort();
console.log(arr);  // [1, 10, 2, 3, 5]

// 오름차순 정렬
arr.sort((a: number, b: number) => a - b);
console.log(arr);  // [1, 2, 3, 5, 10]

// 내림차순 정렬
arr.sort((a: number, b: number) => b - a);
console.log(arr);  // [10, 5, 3, 2, 1]
```

## 검색 (Search)

내장 이진 검색 함수는 없습니다.

```ts
let arr = [1, 2, 3, 4, 5];
console.log(arr.find(a => a > 3));  // 4
console.log(arr.indexOf(2));        // 1
```

## 다차원 배열 (Multidimensional Array)

```ts
let aa: number[][] = [[1, 2, 3],[23, 24, 25]]
for (let i = 0; i < aa.length; i++) {
    for (let j = 0; j < aa[0].length; j++) {
        console.log(aa[i][j]);
    }
}
// 출력:
// 1
// 2
// 3
// 23
// 24
// 25
```

## 열거형 (Enum)

* [Enum | typescript](https://www.typescriptlang.org/docs/handbook/enums.html#handbook-content)

```ts
// 숫자 열거형
enum Direction {
  Up = 1,
  Down,
  Left,
  Right,
}

// 문자열 열거형
enum Direction {
  Up = "UP",
  Down = "DOWN",
  Left = "LEFT",
  Right = "RIGHT",
}

// 이종 열거형
enum BooleanLikeHeterogeneousEnum {
  No = 0,
  Yes = "YES",
}

let dir: Direction = Direction.Up;
let foo: BooleanLikeHeterogeneousEnum.No;
```

## 제네릭 (Generics)

* [Generics](ts_handbook.md#generics)

```ts
// 제네릭 함수
function identity<Type>(arg: Type): Type {
    return arg;
}

// 제네릭 클래스
class GenericNumber<NumType> {
    zeroValue: NumType;
    add: (x: NumType, y: NumType) => NumType;
}
let a = new GenericNumber<number>();
a.zeroValue = 0;
a.add = function(x, y) {
    return x + y;
}
```

## 같은 줄에 여러 변수 정의하기 (Multiple Variables)

```ts
let i = 0, j = 0, n = s.length
```

# 고급 (Advanced)

## Map vs Record

* [map vs object | TIL](/js/README.md#map-vs-object)

Map vs Object와 같습니다.

## 유틸리티 타입 (Utility Types)

> * [Utility Types | typescript](https://www.typescriptlang.org/ko/docs/handbook/utility-types.html)
> * [[Typescript] 유틸리티 타입 - Parameters, ReturnType, Required](https://www.morolog.dev/entry/Typscript-%EC%9C%A0%ED%8B%B8%EB%A6%AC%ED%8B%B0-%ED%83%80%EC%9E%85-Parameters-ReturnType-Required)

```ts
// ReturnType<T>
// 함수의 반환 타입을 생성합니다.
declare function foo(): Foo
type fooResult = ReturnType<typeof foo>;

type F = (...p: any[]) => any
function debounce(fn: F, t: number): F {
    return function(...args) {
        let timeout: ReturnType<typeof setTimeout>
        return (...args) => {
            clearTimeout(timeout);
            timeout = setTimeout(() => fn(...args), t);
        }
    }
};

// Parameters<T>
// 함수의 매개변수 타입을 생성합니다.
declare function foo(foo: {name: string, mobile: number}): void
type fooParams = Parameters<typeof foo>;

// Required<T>
// 모든 필드를 필수로 만드는 타입을 생성합니다.
interface Props {
  a?: number;
  b?: string;
}

// OK
const obj: Props = { a: 5 };
// 에러: Property 'b' is missing in type '{ a: number; }'
// but required in type 'Required<Props>'.
const obj2: Required<Props> = { a: 5 };

// Record<Keys, Type>
// 프로퍼티 키가 Keys이고 프로퍼티 값이 Type인 객체 타입을 생성합니다.
// https://developer-talk.tistory.com/296

// personType이라는 객체 타입을 정의합니다.
// 인덱스 시그니처를 사용합니다.
type personType = {
    [name: string]: number
}
let person: personType = {
    'foo': 10,
    'bar': 20,
    'baz': 30
}
// 이것을 Record 타입으로 바꿔봅시다.
// 장점이 뭘까요?
type personType = Record<string, number>;
let person: personType = {
    'foo': 10,
    'bar': 20,
    'baz': 30
}
// 인덱스 시그니처로 해결되지 않는 경우가 있습니다.
// 에러:
// An index signature parameter type cannot be a
// literal type or generic type. Consider using a mapped
// object type instead.
type personType = {
    [name: 'foo' | 'bar' | 'baz']: number
}
// 이렇게 해결합니다.
type names = 'foo' | 'bar' | 'baz';
type personType = Record<names, number>;
let person: personType = {
    'foo': 10,
    'bar': 20,
    'baz': 30
}

// keyof
// keyof 연산자는 객체 타입을 받아서 그 키들의
// 문자열 또는 숫자 리터럴 유니온을 생성합니다.
type Point = { x: number; y: number };
type P = keyof Point;

// keyof, Record 타입
type personType = {
    name: string;
    age: number;
    addr: string;
}
type personRecordType = Record<keyof personType, string>
let person: personRecordType = {
    name: "iamslash",
    age: "18",
    addr: "USA"
}
```

## 삼중 점 연산자 (Spread/Rest Operator)

> * [JavaScript | JS에서 점점점(…)은 무엇일까?](https://dinn.github.io/javascript/js-dotdotdot/)

삼중 점은 다음 중 하나입니다.

* rest parameter
* spread operator
* rest property
* spread property

```ts
// Rest parameter
function hello(a, b, ...args) {
    console.log(a);
    console.log(b);
    console.log(args);
}
hello(1, 2, 3, 4, 5)
// 1
// 2
// [3, 4, 5]
let arr = [1, 2, 3];
let [one, two, three] = arr;
console.log(one, two, three)  // 1 2 3
function foo(...[a, b, c]) {
    console.log(a, b, c);
}
foo(1, 2, 3);  // 1 2 3

// Spread operator
let arr = [3, 4, 5]
let foo = [...arr];
console.log(arr)  // [3, 4, 5]
console.log(foo)  // [3, 4, 5]

// Rest property
let foo = {
    a: 1,
    b: 2,
    x: 3,
    y: 4
}
let {a, b, ...c} = foo;
console.log(a);  // 1
console.log(b);  // 2
console.log(c);  // {x: 3, y: 4}
let {a, ...c, y} = foo;  // 에러: Uncaught SyntaxError: Rest element must be last element

// Spread property
let a = 1;
let b = 2;
let c = {x: 3, y: 4};
let foo = {a, b, ...c};
console.log(foo);  // {a: 1, b: 2, x: 3, y: 4}

let foo = {a: 1, b: 2};
let bar = {c: 3, d: 4};
let assignedObj = Object.assign({}, foo, bar);  // {a: 1, b: 2, c: 3, d: 4}
let spreadObj = {...foo, ...bar};               // {a: 1, b: 2, c: 3, d: 4}
console.log(JSON.stringify(assignedObj) === JSON.stringify(spreadObj);) // true
```

## 널 병합 연산자 (||), 이중 물음표 (??) (Nullish Coalescing)

> * [null 값을 처리하는 명령의 비교(How To Use Double Question Marks: ??)](https://ksrae.github.io/angular/double-question-marks/)

```ts
// undefined 또는 null 체크를 위한 if문
if (val !== undefined || val != null) {
    console.log("ok");
}
// undefined 또는 null 체크를 위한 삼항 연산자
val = val !== undefined || val !== null ? val : '';
// undefined 또는 null 체크를 위한 널 병합 연산자
val = val || '';
// 널 병합 연산자는 falsy 값을 체크할 수 있습니다
console.log(undefined || "falsy");  // falsy
console.log(null || "falsy");       // falsy
console.log(false || "falsy");      // falsy
console.log(0 || "falsy");          // falsy
console.log('' || "falsy");         // falsy

// TypeScript 3.7 이후의 이중 물음표
console.log(undefined ?? "falsy");  // falsy
console.log(null ?? "falsy");       // falsy
console.log(false ?? "falsy");      // false
console.log(0 ?? "falsy");          // 0
console.log('' ?? "falsy");         //
```

## export와 import (Export & Import)

* [한눈에 보는 타입스크립트(updated) - 내보내기(export)와 가져오기(import)](https://heropy.blog/2020/01/27/typescript/)

```ts
// foo.ts
// 인터페이스 내보내기
export interface UserType {
    name: string,
    mobile: number
}
// 타입 내보내기
export type UserIDType = string | number;

// bar.ts
// 인터페이스, 타입 가져오기
import { UserType, UserIDType } from './foo';
const user: UserType = {
    name: 'David',
    mobile: 333
}
const userid: UserIDType = "111";
```

TypeScript는 `CommonJS/AMD/UMD` 모듈을 위해 `export = bar;`, `export bar = require('bar');`를 지원합니다. 이것은 ES6의 하나의 모듈에서 하나의 객체를 내보내는 `export default`와 같습니다.

```ts
// bar CommonJS/AMD/UMD 모듈에서 가져오기
import bar = require('bar');
// 또는
import * as bar from 'bar';
// 또는 "esModuleInterop": true
import bar from 'bar';
```

## `declare`

* [Purpose of declare keyword in TypeScript | stackoverflow](https://stackoverflow.com/questions/43335962/purpose-of-declare-keyword-in-typescript)
  * [한글](https://jjnooys.medium.com/typescript-declare-cd163acb9f)

declare로 선언한 타입은 컴파일의 대상이 아닙니다. 컴파일 타임에 이렇게 생겼으니 믿고 넘어가주세요라는 의미입니다.

```ts
        type Callback = (err: Error | String, data: Array<CalledBackData>) => void;
declare type Callback = (err: Error | String, data: Array<CalledBackData>) => void;
```

## 인터페이스를 사용한 함수 정의 (Function Types with Interface)

* [TypeScript Interface](https://www.softwaretestinghelp.com/typescript-interface/)

TypeScript 인터페이스는 함수 시그니처를 보장하여 함수 타입을 정의하는 데 사용할 수 있습니다. 프로퍼티 이름 앞에 물음표를 사용하여 선택적 프로퍼티를 사용합니다.

```ts
{
    interface FunctionComponent {
        (): string;
        displayName?: string;
    }
    const foo: FunctionComponent = () => "Hello Foo";
    foo.displayName = "Hello Foo";
    console.log(foo);

    const bar = () => "Hello Bar";
    bar.displayName = "Hello Bar";
    console.log(bar);
}
```

## Interface vs Type

* [typescript type과 interface의 차이 | tistory](https://bny9164.tistory.com/48)

---

`type`보다는 `interface`를 추천합니다. type은 런타임에 재귀적으로 트랜스파일됩니다. 컴파일 타임이 오래 걸리기 때문에 성능이 좋지 않습니다.

`type`은 `interface`에 비해 아래와 같은 단점들이 있습니다.

```ts
//////////////////////////////////////////////////////////////////////
// Interfaces vs. Intersections
// extends
{
    interface Point {
        x: number;
        y: number;
    }
    interface PointColor extends Point {
        c: number;
    }
    const pointColor = {
        x: 3,
        y: 3,
        c: 3,
    }
    console.log(pointColor);
}
{
    type Point = {
        x: number;
        y: number;
    }
    interface PointColor extends Point {
        c: number;
    }
    const pointColor: PointColor = { x: 3, y: 3, c: 3 };
    console.log(pointColor);
}
{
    // extends는 type에서 작동하지 않습니다
    type Point = {
        x: number;
        y: number;
    }
    // // 에러: type에서 extends를 사용할 수 없습니다
    // type PointColor extends Point {
    //     c: number;
    // }
}
// 병합된 선언
{
    // 병합된 선언은 interface에서 작동합니다
    interface PointColor {
        x: number;
        y: number;
    }
    interface PointColor {
        c: number;
    }
    const pointColor: PointColor = { x: 3, y: 3, c: 3 };
    console.log(pointColor);
}
{
    // // 에러: 병합된 선언은 type에서 작동하지 않습니다
    // type PointColor = {
    //     x: number;
    //     y: number;
    // }
    // type PointColor = {
    //     c: number;
    // }
}
// 계산된 값
{
    // 계산된 값은 interface에서 작동하지 않습니다
    type coords = 'x' | 'y';
    interface CoordTypes {
        [key in coords]: string
    }
}
{
    // 계산된 값은 type에서 작동합니다
    type coords = 'x' | 'y';
    type CoordTypes = {
        [CoordTypes in coords]: string;
    }
    const point: CoordTypes = { x: '3', y: '3' };
    console.log(point);
}
// type은 never 타입으로 해석될 수 있습니다
// 주의해야 합니다
{
    type goodType = { a: 1 } & { b: 2 } // 좋음
    type neverType = { a: 1; b: 2 } & { b: 3 } // `never`로 해석됨

    const foo: goodType = { a: 1, b: 2 } // 좋음
    // 에러: Type 'number' is not assignable to type 'never'.(2322)
    const bar: neverType = { a: 1, b: 3 }
    // 에러: Type 'number' is not assignable to type 'never'.(2322)
    const baz: neverType = { a: 1, b: 2 }
}
{
    type t1 = {
        a: number
    }
    type t2 = t1 & {
        b: string
    }

    const foo: t2 = { a: 1, b: 2 } // 에러
}
```

## Optional (선택적 매개변수와 속성)

TypeScript에서 `?`를 이름 뒤에 붙이면 **"있어도 되고 없어도 된다"** 는 의미입니다. 크게 3가지 용법이 있습니다.

### Optional Parameter (선택적 매개변수)

함수의 매개변수에 `?`를 붙이면 호출할 때 생략할 수 있습니다.

```ts
// name은 필수, greeting은 선택
function greet(name: string, greeting?: string): string {
  return `${greeting ?? "Hello"}, ${name}!`;
}

console.log(greet("David"));            // "Hello, David!"
console.log(greet("David", "Hi"));      // "Hi, David!"
```

옵션 객체 패턴에서도 많이 사용합니다.

```ts
// 두 번째 매개변수 자체가 선택적
async function fetchData(url: string, options?: { timeout?: number; retries?: number }) {
  const timeout = options?.timeout ?? 3000;
  const retries = options?.retries ?? 1;
  console.log(`url=${url}, timeout=${timeout}, retries=${retries}`);
}

await fetchData("/api/users");                          // options 생략 OK
await fetchData("/api/users", { timeout: 5000 });       // retries 생략 OK
await fetchData("/api/users", { timeout: 5000, retries: 3 }); // 둘 다 전달 OK
```

### Optional Property (선택적 속성)

인터페이스나 타입의 속성에 `?`를 붙이면 그 속성은 없어도 됩니다.

```ts
interface User {
  name: string;       // 필수
  age?: number;       // 선택
  email?: string;     // 선택
}

const user1: User = { name: "David" };                     // OK
const user2: User = { name: "David", age: 30 };            // OK
const user3: User = { name: "David", age: 30, email: "a@b.com" }; // OK
// const user4: User = { age: 30 };                        // ERROR: name이 없음
```

### Optional Chaining (선택적 체이닝)

객체의 속성에 접근할 때 `?.`를 사용하면, 값이 `null` 또는 `undefined`일 때 에러 없이 `undefined`를 반환합니다.

```ts
interface Company {
  name: string;
  address?: {
    city?: string;
    zipCode?: string;
  };
}

const company: Company = { name: "Foo Inc." };

// ?. 없이 접근하면 런타임 에러 발생
// console.log(company.address.city);   // ERROR: Cannot read property 'city' of undefined

// ?. 를 사용하면 안전하게 접근
console.log(company.address?.city);     // undefined (에러 없음)
console.log(company.address?.zipCode);  // undefined (에러 없음)
```

### 정리

| 용법 | 문법 | 의미 |
|------|------|------|
| Optional Parameter | `function foo(x?: string)` | 매개변수를 안 넘겨도 됨 |
| Optional Property | `{ name?: string }` | 속성이 없어도 됨 |
| Optional Chaining | `obj?.prop` | null/undefined면 에러 대신 undefined 반환 |

# 스타일 가이드 (Style Guide)

[TypeScript Google Style Guide](ts_google_style_guide.md)

# 리팩토링 (Refactoring)

[Refactoring TypeScript](refactoring_ts.md)

# 효율적인 TypeScript (Effective TypeScript)

[Effective TypeScript](effective_ts.md)

# 디자인 패턴 (Design Patterns)

[TypeScript Design Pattern](ts_gof_design_pattern.md)

# 아키텍처 (Architecture)

* [Typescript Clean Architecture | github](https://github.com/pvarentsov/typescript-clean-architecture)
  * Java의 Clean Architecture와는 조금 다릅니다
* [A TypeScript Stab at Clean Architecture](https://www.freecodecamp.org/news/a-typescript-stab-at-clean-architecture-b51fbb16a304/)
* [Evolution of a React folder structure and why to group by features right away](https://profy.dev/article/react-folder-structure)
* [React Folder Structure in 5 Steps [2022]](https://www.robinwieruch.de/react-folder-structure/)
  * 단순한 구조부터 복잡한 구조까지 단계별로 설명
* [bulletproof-react/docs/project-structure.md](https://github.com/alan2207/bulletproof-react/blob/master/docs/project-structure.md)
* [4 folder structures to organize your React & React Native project](https://reboot.studio/blog/folder-structures-to-organize-react-project/)
* [Project structure | Frontend Handbook](https://infinum.com/handbook/frontend/react/project-structure)
