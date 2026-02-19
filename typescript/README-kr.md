# TypeScript (한국어)

**영문 버전**: 영문 문서는 [README.md](README.md)를 참조하세요.

- [학습 자료](#학습-자료)
- [기초](#기초)
  - [빌드 및 실행](#빌드-및-실행)
  - [출력하기](#출력하기)
  - [예약어](#예약어)
  - [최소값, 최대값](#최소값-최대값)
  - [abs vs fabs](#abs-vs-fabs)
  - [비트 연산](#비트-연산)
  - [문자열](#문자열)
  - [난수 생성](#난수-생성)
  - [포맷된 문자열](#포맷된-문자열)
  - [검사하기](#검사하기)
  - [데이터 타입](#데이터-타입)
    - [undefined vs unknown vs any vs never 비교](#undefined-vs-unknown-vs-any-vs-never-비교)
  - [제어 흐름문](#제어-흐름문)
    - [조건문](#조건문)
    - [반복문](#반복문)
  - [컬렉션](#컬렉션)
    - [튜플](#튜플)
    - [배열](#배열)
    - [집합](#집합)
    - [맵](#맵)
  - [컬렉션 변환](#컬렉션-변환)
  - [정렬](#정렬)
  - [검색](#검색)
  - [다차원 배열](#다차원-배열)
  - [열거형](#열거형)
  - [제네릭](#제네릭)
  - [같은 줄에 여러 변수 정의하기](#같은-줄에-여러-변수-정의하기)
- [고급](#고급)
  - [Map vs Record](#map-vs-record)
  - [유틸리티 타입](#유틸리티-타입)
  - [삼중 점 연산자](#삼중-점-연산자)
  - [널 병합 연산자 (||), 이중 물음표 (??)](#널-병합-연산자--이중-물음표-)
  - [export와 import](#export와-import)
  - [`declare`](#declare)
  - [인터페이스를 사용한 함수 정의](#인터페이스를-사용한-함수-정의)
  - [Interface vs Type](#interface-vs-type)
  - [Optional (선택적 매개변수와 속성)](#optional-선택적-매개변수와-속성)
- [스타일 가이드](#스타일-가이드)
- [리팩토링](#리팩토링)
- [효율적인 TypeScript](#효율적인-typescript)
- [디자인 패턴](#디자인-패턴)
- [아키텍처](#아키텍처)

----

# 학습 자료

* [한눈에 보는 타입스크립트(updated)](https://heropy.blog/2020/01/27/typescript/)
* [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html)
  * 반드시 읽어야 할 공식 문서
  * [TypeScript Handbook 한글](https://typescript-kr.github.io/)
* [8장. 리액트 프로젝트에서 타입스크립트 사용하기](https://react.vlpt.us/using-typescript/)
* [TypeScript 환경에서 Redux를 프로처럼 사용하기 @ velog](https://velog.io/@velopert/use-typescript-and-redux-like-a-pro)
* [playcode](https://playcode.io/)
  * TypeScript 플레이그라운드
* [TypeScript의 소개와 개발 환경 구축](https://poiemaweb.com/typescript-introduction)

# 기초

## 빌드 및 실행

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

## 출력하기

```typescript
function greet(person: string, date: Date) {
  console.log(`Hello ${person}, today is ${date.toDateString()}!`);
}

greet("Maddison", new Date());
```

## 예약어

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

## 최소값, 최대값

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

## 비트 연산

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

## 문자열

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

## 난수 생성

* [Math.random() | mdn](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Math/random)

`[0..1]` 범위의 값을 생성합니다.

```ts
console.log(Math.random());
console.log(Math.random());
```

## 포맷된 문자열

* [util.format | node.js](https://nodejs.org/api/util.html#utilformatformat-args)

TypeScript의 기본 타입은 `boolean, number, string`임을 기억하세요.

```ts
import util from "util";

console.log(util.format('%s %d %s', true, 4, 'Hello World'));

// 줄바꿈 없이 출력
process.stdout.write(`i: ${i}, diff: ${diff} has: ${numMap.has(diff)} `);
console.log(numMap);
```

## 검사하기

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

## 데이터 타입

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

## 제어 흐름문

### 조건문

```ts
// if ... else if ... else
var num: number = 1
if (num > 0) {
    console.log("positive");
} else if (num < 0) {
    console.log("negative");
} else {
    console.log("zero");
}

var grade: string = "A";
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

### 반복문

```ts
// for
for (let i = 0; i < 3; i++) {
    console.log(i);
}
// 출력:
// 0
// 1
// 2

// for ... of는 값을 반환
let arr = [1, 2, 3];
for (var val of arr) {
    console.log(val);
}
// 출력:
// 1
// 2
// 3
let str = "Hello";
for (var chr of str) {
    console.log(chr);
}
// 출력:
// H
// e
// l
// l
// o

// for ... in은 인덱스를 반환
let arr = [1, 2, 3];
for (var idx in arr) {
    console.log(idx);
}
// 출력:
// 0
// 1
// 2

// for 루프에서 var vs let
// var는 루프 밖에서 접근 가능
let arr = [1, 2, 3];
for (var idx1 in arr) {
    console.log(idx1);
}
console.log(idx1);
for (let idx2 in arr) {
    console.log(idx2);
}
console.log(idx2);  // 에러
```

## 컬렉션

### 튜플

```ts
var employee: [number, string] = [1, 'David'];
var person: [number, string, boolean] = [1, 'David', true];
var user: [number, string, boolean, number, string];
user = [1, 'David', true, 10, 'admin'];
var employees: [number, string][];
employees = [[1, 'David'], [2, 'Tom']];

console.log(employee[0], employee[1]);  // 1 'David'

employee.push(2, 'John');
console.log(employee);  // [1, 'David', 2, 'John']
employee.pop();
console.log(employee);  // [1, 'David', 2]
employee[1] = employee[1].concat(' Sun');
console.log(employee);  // [1, 'David Sun', 2]
```

### 배열

```ts
let fruits: Array<string>;
fruits = ['Apple', 'Orange', 'Banana'];
let vals: (string | number)[] = ['Apple', 2, 'Orange', 3, 4, 'Banana'];
let vals: Array<string | number> = ['Apple', 2, 'Orange', 3, 4, 'Banana'];
console.log(vals[0]);  // 'Apple'

for (var idx in fruits) {
    console.log(fruits[idx]);
}
// 출력:
// 0
// 1
// 2
for (var i = 0; i < fruits.length; i++) {
    console.log(fruits[i]);
}
// 출력:
// 'Apple'
// 'Orange'
// 'Banana'

fruits.sort();
console.log(fruits);  // ['Apple', 'Banana', 'Orange']
console.log(fruits.pop()); // 'Orange'
fruits.push('Papaya');
console.log(fruits);  // ['Apple', 'Banana', 'Papaya']
fruits = fruits.concat(['Fig', 'Mango']);
console.log(fruits); // 출력: ['Apple', 'Banana', 'Papaya', 'Fig', 'Mango']
console.log(fruits.indexOf('Papaya'));  // 2
console.log(fruits.slice(1, 2));  // ['Banana']

// 하나의 값으로 배열 초기화
let prevIdxs = new Array<Number>().fill(-1);

// map
// `map`은 배열의 각 요소에 대해 변환 작업을 수행하고 새로운 배열을 반환합니다.
const numbers = [1, 2, 3, 4];
const doubled = numbers.map(num => num * 2);
console.log(doubled); // [2, 4, 6, 8]

// reduce
// `reduce`는 배열을 순회하며 단일 값으로 축약합니다.
const sum = numbers.reduce((acc, cur) => acc + cur, 0);
console.log(sum); // 10

// filter
// `filter`는 조건에 맞는 요소만 걸러서 새로운 배열을 반환합니다.
const evenNumbers = numbers.filter(num => num % 2 === 0);
console.log(evenNumbers); // [2, 4]

// sort
// 배열의 요소를 정렬합니다.
const unsortedNumbers = [3, 1, 4, 2];
unsortedNumbers.sort((a, b) => a - b);
console.log(unsortedNumbers); // [1, 2, 3, 4]

// every와 some
// - `every`: 배열의 모든 요소가 조건을 만족하면 `true`를 반환.
// - `some`: 배열의 일부 요소가 조건을 만족하면 `true`를 반환.
const allEven = numbers.every(num => num % 2 === 0); // false
const hasOdd = numbers.some(num => num % 2 !== 0); // true
console.log(allEven, hasOdd); // false, true

// find와 findIndex
// - `find`: 조건에 맞는 첫 번째 요소를 반환.
// - `findIndex`: 조건에 맞는 첫 번째 요소의 인덱스를 반환.
const firstEven = numbers.find(num => num % 2 === 0); // 2
const indexOfFirstEven = numbers.findIndex(num => num % 2 === 0); // 1
console.log(firstEven, indexOfFirstEven); // 2, 1

// includes
// 특정 값이 배열에 포함되어 있는지 확인합니다.
const hasThree = numbers.includes(3); // true
console.log(hasThree); // true

// flat과 flatMap
// - `flat`: 다차원 배열을 평탄화합니다.
// - `flatMap`: `map` 후 평탄화를 동시에 수행합니다.
const nested = [1, [2, 3], [4, [5]]];
const flatArray = nested.flat(2);
console.log(flatArray); // [1, 2, 3, 4, 5]

const strings = ["hello", "world"];
const charArray = strings.flatMap(str => str.split(''));
console.log(charArray); // ['h', 'e', 'l', 'l', 'o', 'w', 'o', 'r', 'l', 'd']

// splice
// 배열에서 요소를 추가/삭제/대체합니다.
const mutableNumbers = [1, 2, 3, 4];
mutableNumbers.splice(1, 2, 99); // 1번 인덱스부터 2개 제거 후 99 추가
console.log(mutableNumbers); // [1, 99, 4]

// reverse
// 배열 요소의 순서를 뒤집습니다.
const reversedNumbers = [...mutableNumbers].reverse();
console.log(reversedNumbers); // [4, 99, 1]

// push와 pop
// 배열 끝에 요소를 추가하거나 제거합니다.
const stack = [1, 2];
stack.push(3); // [1, 2, 3]
stack.pop(); // [1, 2]
console.log(stack); // [1, 2]

// shift와 unshift
// 배열 시작에 요소를 추가하거나 제거합니다.
const queue = [1, 2];
queue.unshift(0); // [0, 1, 2]
queue.shift(); // [1, 2]
console.log(queue); // [1, 2]
```

### 집합

```ts
let dirs = new Set<string>();
let dirs = new Set<string>(['east', 'west']);
dirs.add('east');
dirs.add('north');
dirs.add('south');
console.log(dirs);      // Set(4) { 'east', 'west', 'north', 'south' }
console.log(dirs.has('east'));  // true
console.log(dirs.size);         // 4

console.log(dirs.delete('east')); // true
console.log(dirs);                // Set(3) { 'west', 'north', 'south' }
for (let dir of dirs) {
    console.log(dir);
}
// 출력:
// west
// north
// south

console.log(dirs.clear()); // undefined
console.log(dirs);         // Set(0) {}
```

### 맵

```ts
let fooMap = new Map<string, number>();
let barMap = new Map<string, string>([
    ['key1', 'val1'],
    ['key2', 'val2']
]);

fooMap.set('David', 10);
fooMap.set('John', 20);
fooMap.set('Raj', 30);
console.log(fooMap.get('David'));   // 10
// 기본값
console.log(fooMap.get('Tom') || 0) // 0
console.log(fooMap.has('David'));  // true
console.log(fooMap.has('Tom'));    // false
console.log(fooMap.size);          // 3
console.log(fooMap.delete('Raj')); // true

for (let key of fooMap.keys()) {
    console.log(key);
}
// 출력:
// David
// John
for (let val of fooMap.values()) {
    console.log(val);
}
// 출력:
// 10
// 20
for (let entry of fooMap.entries()) {
    console.log(entry[0], entry[1]);
}
// 출력:
// "David" 10
// "John" 20
for (let [key, val] of fooMap) {
    console.log(key, val);
}

fooMap.clear();
```

## 컬렉션 변환

```ts
// 튜플을 집합으로
let arr = [11, 22, 33];
let set = new Set(arr);
console.log(set);  // Set(3) { 11, 22, 33 }
```

## 정렬

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

## 검색

내장 이진 검색 함수는 없습니다.

```ts
let arr = [1, 2, 3, 4, 5];
console.log(arr.find(a => a > 3));  // 4
console.log(arr.indexOf(2));        // 1
```

## 다차원 배열

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

## 열거형

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

## 제네릭

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

## 같은 줄에 여러 변수 정의하기

```ts
let i = 0, j = 0, n = s.length
```

# 고급

## Map vs Record

* [map vs object | TIL](/js/README.md#map-vs-object)

Map vs Object와 같습니다.

## 유틸리티 타입

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

## 삼중 점 연산자

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

## 널 병합 연산자 (||), 이중 물음표 (??)

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

## export와 import

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

## 인터페이스를 사용한 함수 정의

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

# 스타일 가이드

[TypeScript Google Style Guide](ts_google_style_guide.md)

# 리팩토링

[Refactoring TypeScript](refactoring_ts.md)

# 효율적인 TypeScript

[Effective TypeScript](effective_ts.md)

# 디자인 패턴

[TypeScript Design Pattern](ts_gof_design_pattern.md)

# 아키텍처

* [Typescript Clean Architecture | github](https://github.com/pvarentsov/typescript-clean-architecture)
  * Java의 Clean Architecture와는 조금 다릅니다
* [A TypeScript Stab at Clean Architecture](https://www.freecodecamp.org/news/a-typescript-stab-at-clean-architecture-b51fbb16a304/)
* [Evolution of a React folder structure and why to group by features right away](https://profy.dev/article/react-folder-structure)
* [React Folder Structure in 5 Steps [2022]](https://www.robinwieruch.de/react-folder-structure/)
  * 단순한 구조부터 복잡한 구조까지 단계별로 설명
* [bulletproof-react/docs/project-structure.md](https://github.com/alan2207/bulletproof-react/blob/master/docs/project-structure.md)
* [4 folder structures to organize your React & React Native project](https://reboot.studio/blog/folder-structures-to-organize-react-project/)
* [Project structure | Frontend Handbook](https://infinum.com/handbook/frontend/react/project-structure)
