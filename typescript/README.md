- [Materials](#materials)
- [Basic](#basic)
  - [Build \& Run](#build--run)
  - [Hello World](#hello-world)
  - [Reserved Words](#reserved-words)
  - [min, max values](#min-max-values)
  - [abs vs fabs](#abs-vs-fabs)
  - [Bit Manipulation](#bit-manipulation)
  - [String](#string)
  - [Random](#random)
  - [Formatted Strings](#formatted-strings)
  - [Inspecting](#inspecting)
  - [Data Types](#data-types)
  - [Control Flow Statements](#control-flow-statements)
    - [Decision Making Statements](#decision-making-statements)
    - [Looping Statements](#looping-statements)
  - [Collections](#collections)
    - [tutple](#tutple)
    - [array](#array)
    - [set](#set)
    - [map](#map)
  - [Collection Conversions](#collection-conversions)
  - [Sort](#sort)
  - [Search](#search)
  - [Multi Dimensional Arrays](#multi-dimensional-arrays)
  - [Enum](#enum)
  - [Generics](#generics)
  - [Define Multiple Variables On The Same Line](#define-multiple-variables-on-the-same-line)
- [Advanced](#advanced)
  - [Map vs Record](#map-vs-record)
  - [Utility Types](#utility-types)
  - [Triple Dots](#triple-dots)
  - [Nullish Coalescing Operator (||), Double Question Marks (??)](#nullish-coalescing-operator--double-question-marks-)
  - [export and import](#export-and-import)
  - [`declare`](#declare)
  - [Function Definition With Interfaces](#function-definition-with-interfaces)
  - [Interface vs Type](#interface-vs-type)
- [Style Guide](#style-guide)
- [Refactoring](#refactoring)
- [Effective](#effective)
- [Design Pattern](#design-pattern)
- [Architecture](#architecture)

----

# Materials

* [한눈에 보는 타입스크립트(updated)](https://heropy.blog/2020/01/27/typescript/)
* [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html)
  * Should read this.
  * [TypeScript Handbook kor](https://typescript-kr.github.io/)
* [8장. 리액트 프로젝트에서 타입스크립트 사용하기](https://react.vlpt.us/using-typescript/)
* [TypeScript 환경에서 Redux를 프로처럼 사용하기 @ velog](https://velog.io/@velopert/use-typescript-and-redux-like-a-pro) 
* [playcode](https://playcode.io/)
  * typescript play ground
* [TypeScript의 소개와 개발 환경 구축](https://poiemaweb.com/typescript-introduction)

# Basic

## Build & Run

```bash
# Install tsc
$ npm install -g typescript
# Build with tsc
$ tsc a.ts
# Run with node
$ node a.js

# Install ts-node 
$ npm install -g ts-node
# Run with ts-node
$ ts-node a.ts

# REPL
$ ts-node
> let a = 2
> a
2
```

## Hello World

```typescript
function greet(person: string, date: Date) {
  console.log(`Hello ${person}, today is ${date.toDateString()}!`);
}
 
greet("Maddison", new Date());
```

## Reserved Words

* [types.ts | github](https://github.com/Microsoft/TypeScript/blob/fad889283e710ee947e8412e173d2c050107a3c1/src/compiler/types.ts#L87)

```ts
// Reserved words
break case catch class const
continue debugger default delete
do else enum export extends false
finally for function if import in
instanceof new null return super
switch this throw true try typeof
var void while with

// Strict mode reserved words
as implements interface let package
private protected public static yield

// Contextual keywords
any boolean constructor declare get
module require number set string symbol
type from of
```

## min, max values

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
// expected output: 2
console.log(difference(5, 3));
// expected output: 2
console.log(difference(1.23456, 7.89012));
// expected output: 6.6555599999999995
```

## Bit Manipulation

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

## String

```ts
// Interpolated string
type World = "world";
type Greeting = `hello ${World}`;
console.log(Greeting);  // hello world

// Loop string
let s = "hello world'
let n = s.length
for (let i = 0; i < n; ++i) {
    let c = s.charAt(i);
    console.log(c);
}

// Convert letters, number in string
// https://stackoverflow.com/questions/22624379/how-to-convert-letters-to-numbers-with-javascript
let s = "abcdefg";
let c = s.charAt(0);   // c is string, "a"
// 97 means "a"
let i = c.charCodeAt(0) - 97;  // 0
let d = string.fromCharCode(97 + i) // "a"
```

## Random

* [Math.random() | mdn](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Math/random)

`[0..1]` 의 값을 생성한다.

```ts
console.log(Math.random());
console.log(Math.random());
```

## Formatted Strings

* [util.format | node.js](https://nodejs.org/api/util.html#utilformatformat-args)

typescript 의 primary type 은 `boolean, number, string` 임을 기억하자.

```ts
import util from "util";

console.log(util.format('%s %d %s', true, 4, 'Hello World'));

// Print without newline
process.stdout.write(`i: ${i}, diff: ${diff} has: ${numMap.has(diff)} `);
console.log(numMap);
```

## Inspecting 

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

## Data Types

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
// Arrays of values
let list1: number[] = [1, 2, 3];
// Generic Array
let list2: Array<number> = [1, 2, 3];

// Tuple
let x: [string, number];
x = ["hello", 10];

// Enum
enum Color {
    Red, Green, Blue,
}
let c: Color = Color.Green;

// Unkonwn
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

// Null and Undefined
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

// Type assertions
let someValue: unknown = "This is a string";
let strLength: number = (someValue as string).length;
let strLength2: number = (<string>someValue).length;
```

## Control Flow Statements

### Decision Making Statements

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

### Looping Statements

```ts
// for
for (let i = 0; i < 3; i++) {
    console.log(i);
}
// Output:
// 0
// 1
// 2

// for ... of return value
let arr = [1, 2, 3];
for (var val of arr) {
    console.log(val);
}
// Output:
// 1
// 2
// 3
let srr = "Hello";
for (var chr of str) {
    console.log(chr);
}
// Output:
// H
// e
// l
// l
// o

// for ... of return index
let arr = [1, 2, 3];
for (var idx in arr) {
    console.log(idx);
}
// Output:
// 0
// 1
// 2

// var vs let in for loop
// var can be accessed outside loop
let arr = [1, 2, 3];
for (var idx1 in arr) {
    console.log(idx1);
}
console.log(idx1);
for (let idx2 in arr) {
    console.log(idx2);
}
console.log(idx2);  // ERROR
```

## Collections

### tutple

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

### array

```ts
let fruits: Array<string>;
fruits = ['Apple', 'Orange', 'Banana'];
let vals: (string | number)[] = ['Apple', 2, 'Orange', 3, 4, 'Banana']; 
let vals: Array<string | number> = ['Apple', 2, 'Orange', 3, 4, 'Banana']; 
console.log(vals[0]);  // 'Apple'

for (var idx in fruits) {
    console.log(fruits[idx]);
}
// Output:
// 0
// 1
// 2
for (var i = 0; i < fruits.length; i++) {
    console.log(fruits[i]);
}
// Output:
// 'Apple'
// 'Orange'
// 'Banana'

fruits.sort();
console.log(fruits);  // ['Apple', 'Banana', 'Orange']
console.log(fruits.pop()); // 'Orange'
fruits.push('Papaya');  
console.log(fruits);  // ['Apple', 'Banana', 'Papaya']
fruits = fruits.concat(['Fig', 'Mango']); 
console.log(fruits); //output: ['Apple', 'Banana', 'Papaya', 'Fig', 'Mango'] 
console.log(fruits.indexOf('Papaya'));  // 2
console.log(fruits.slice(1, 2));  // ['Banana']

// Init array with one value
let prevIdxs = new Array<Number>().fill(-1);
```

### set

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
// Output:
// west
// north
// south

console.log(dirs.clear()); // undefined
console.log(dirs);         // Set(0) {}
```

### map

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
// default value
console.log(fooMap.get('Tom') || 0) // 0
console.log(fooMap.has('David'));  // true
console.log(fooMap.has('Tom'));    // false
console.log(fooMap.size);          // 3
console.log(fooMap.delete('Raj')); // true

for (let key of fooMap.keys()) {
    console.log(key);
}
// Output:
// David
// John
for (let val of fooMap.values()) {
    console.log(val);
}
// Output:
// 10
// 20
for (let entry of fooMap.entries()) {
    console.log(entry[0], entry[1]);
}
// Output:
// "David" 10
// "John" 20
for (let [key, val] of fooMap) {
    console.log(key, val);
}

fooMap.celar();
```

## Collection Conversions

```ts
// tuple to set
let arr = [11, 22, 33];
let set = new Set(arr);
console.log(set);  // Set(3) { 11, 22, 33 }
```

## Sort

```ts
let arr: number[] = [1, 10, 2, 5, 3];
console.log(arr);  // [1, 10, 2, 5, 3]

// sort lexicographically
arr.sort();
console.log(arr);  // [1, 10, 2, 3, 5]

// sort asencding
arr.sort((a: number, b: number) => a - b);
console.log(arr);  // [1, 2, 3, 5, 10]

// sort descending
arr.sort((a: number, b: number) => b - a);
console.log(arr);  // [10, 5, 3, 2, 1]
```

## Search

built-in binary search function 없는 건

```ts
let arr = [1, 2, 3, 4, 5];
console.log(arr.find(a => a > 3));  // 4
console.log(arr.indexOf(2));        // 1 
```

## Multi Dimensional Arrays

```ts
let aa: number[][] = [[1, 2, 3],[23, 24, 25]]  
for (let i = 0; i < aa.length; i++) {
    for (let j = 0; j < aa[0].length; j++) {
        console.log(aa[i][j]);
    }
}
// Output:
// 1
// 2
// 3
// 23
// 24
// 25
```

## Enum

* [Enum | typescript](https://www.typescriptlang.org/docs/handbook/enums.html#handbook-content)

```ts
// numeric enums
enum Direction {
  Up = 1,
  Down,
  Left,
  Right,
}

// string enums
enum Direction {
  Up = "UP",
  Down = "DOWN",
  Left = "LEFT",
  Right = "RIGHT",
}

// heterogeneous enums
enum BooleanLikeHeterogeneousEnum {
  No = 0,
  Yes = "YES",
}

let dir: Direction = Direction.Up;
let foo: BooleanLikeHeterogeneousEnum.No;
```

## Generics

* [Generics](typescript_handbook.md#generics)

```ts
// generic functions
function identity<type>(arg: Type): Type {
    return arg;
}

// generic classes
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

## Define Multiple Variables On The Same Line

```ts
let i = 0, j = 0, n = s.length
```

# Advanced

## Map vs Record

* [map vs object | TIL](/js/README.md#map-vs-object)

Map vs Object 와 같다.

## Utility Types

> * [Utility Types | typescript](https://www.typescriptlang.org/ko/docs/handbook/utility-types.html)
> * [[Typescript] 유틸리티 타입 - Parameters, ReturnType, Required](https://www.morolog.dev/entry/Typscript-%EC%9C%A0%ED%8B%B8%EB%A6%AC%ED%8B%B0-%ED%83%80%EC%9E%85-Parameters-ReturnType-Required)

```ts
// ReturnType<T>
// It creates a type return of the function.
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

// Paremeters<T>
// It creates a type parameter of the function.
declare function foo(foo: {name: string, mobile: number}): void
type fooParams = Parameters<typeof foo>;

// Required<T>
// It creates a type every fields are required. 
interface Props {
  a?: number;
  b?: string;
}

// OK
const obj: Props = { a: 5 };
// ERROR: Property 'b' is missing in type '{ a: number; }' 
// but required in type 'Required<Props>'.
const obj2: Required<Props> = { a: 5 };  

// Record<Keys, Type>
// Constructs an object type whose property keys are Keys and 
// whose property values are Type.  
// https://developer-talk.tistory.com/296

// personType 이라는 object type 을 정의하자.
// index signature 를 사용함.
type personType = {
    [name: string]: number
}
let person: personType = {
    'foo': 10,
    'bar': 20,
    'baz': 30
}
// 이 것을 Record type 으로 바꾸어 보자.
// 장점이 뭐냐?
type personType = Record<string, number>;
let person: personType = {
    'foo': 10,
    'bar': 20,
    'baz': 30
}
// index signature 로 해결안되는 경우가 있다.
// ERROR:
// An index signature parameter type cannot be a
// literal type or generic type. Consider using a mapped
// object type intead.
type personType = {
    [name: 'foo' | 'bar' | 'baz']: number
}
// 이렇게 해결하자.
type = names = 'foo' | 'bar' | 'bar';
type personType = Record<names, number>;
let person: personType = {
    'foo': 10,
    'bar': 20,
    'baz': 30
}

// keyof
// The keyof operator takes an object type and 
// produces a string or numeric literal union of its keys.
type Point = { x: number; y: number };
type P = keyof Point;

// keyof, Record type
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

## Triple Dots

> * [JavaScript | JS에서 점점점(…)은 무엇일까?](https://dinn.github.io/javascript/js-dotdotdot/)
>

Triple dot are one of these.

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
let {a, ...c, y} = foo;  // ERROR: Uncaught SyntaxError: Rest element must be last element

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

## Nullish Coalescing Operator (||), Double Question Marks (??)

> * [null 값을 처리하는 명령의 비교(How To Use Double Question Marks: ??)](https://ksrae.github.io/angular/double-question-marks/)

```ts
// if for checking undefined or null
if (val !== undefined || val != null) {
    console.log("ok");
}
// tri operator for checking undefined or null
val = val !== undefined || val !== null ? val : '';
// Nullish Coalescing Operator for checking undefined or null
val = val || '';
// Nullish Coalescing Operator can check falsy
console.log(undefined || "falsy");  // falsy
console.log(null || "falsy");       // falsy
console.log(false || "falsy");      // falsy
console.log(0 || "falsy");          // falsy
console.log('' || "falsy");         // falsy

// Double Question Marks since typescript 3.7
console.log(undefined ?? "falsy");  // falsy
console.log(null ?? "falsy");       // falsy
console.log(false ?? "falsy");      // false
console.log(0 ?? "falsy");          // 0
console.log('' ?? "falsy");         // 
```

## export and import

* [한눈에 보는 타입스크립트(updated) - 내보내기(export)와 가져오기(import)](https://heropy.blog/2020/01/27/typescript/)

```ts
// foo.ts
// export interface
export interface UserType {
    name: string,
    mobile: number
}
// export type
export type UserIDType = string | number;

// bar.ts
// import interface, type
import { UserType, UserIDType } from './foo';
const user: UserType = {
    name: 'David',
    mobile: 333
}
const userid: UserIDType = "111";
```

typescript supports `export = bar;`, `export bar = require('bar');` for
`CommonJS/AMD/UMD` modules. This is same with `export default` which exports one
object in one module from ES6.

```ts
// import from bar CommonJS/AMD/UMD module
import bar = require('bar');
// or
import * as bar from 'bar';
// or "esModuleInterop": true
import bar from 'bar';
```

## `declare`

* [Purpose of declare keyword in TypeScript | stackoverflow](https://stackoverflow.com/questions/43335962/purpose-of-declare-keyword-in-typescript)
  * [kor](https://jjnooys.medium.com/typescript-declare-cd163acb9f)

declare 로 선언한 type 은 compile 의 대상이 아니다. compile time 에 이렇게 생겼으니 믿고 넘어가주세요 라는 의미이다.

```ts
        type Callback = (err: Error | String, data: Array<CalledBackData>) => void;
declare type Callback = (err: Error | String, data: Array<CalledBackData>) => void;
```

## Function Definition With Interfaces 

* [](https://www.softwaretestinghelp.com/typescript-interface/)

TypeScript Interface can be used to define a function type by ensuring a
function signature. We use the optional property using a question mark before
the property name colon.

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

`type` 보다는 `interface` 를 추천한다. type 은 runtime 에 recursive 하게 transpile 한다. compile time 오래 걸리기 때문에 performance 가 좋지 않다.

`type` 은 `interface` 에 비해 아래와 같은 단점들이 있다.

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
    // extends does not work for type
    type Point = {
        x: number;
        y: number;
    }
    // // ERROR: Could not use type with extends
    // type PointColor extends Point {
    //     c: number;
    // }
}
// merged declaration
{
    // merged declaration works for interface
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
    // // ERROR: mergedd declaration does not work for type
    // type PointColor = {
    //     x: number;
    //     y: number;
    // }
    // type PointColor = {
    //     c: number;
    // }
}
// computed value
{
    // computed value does not work for interface
    type coords = 'x' | 'y';
    interface CoordTypes {
        [key in coords]: string
    }
}
{
    // computed value works for type
    type coords = 'x' | 'y';
    type CoordTypes = {
        [CoordTypes in coords]: string;
    }
    const point: CoordTypes = { x: '3', y: '3' };
    console.log(point);
}
// type could be resolved to never type
// You should be careful
{
    type goodType = { a: 1 } & { b: 2 } // good
    type neverType = { a: 1; b: 2 } & { b: 3 } // resolved to `never`

    const foo: goodType = { a: 1, b: 2 } // good
    // ERROR: Type 'number' is not assignable to type 'never'.(2322)
    const bar: neverType = { a: 1, b: 3 } 
    // ERROR: Type 'number' is not assignable to type 'never'.(2322)
    const baz: neverType = { a: 1, b: 2 } 
}
{
    type t1 = {
        a: number
    }
    type t2 = t1 & {
        b: string
    }
    
    const foo: t2 = { a: 1, b: 2 } // ERROR
}
```

# Style Guide

* [TypeScript Style Guide | google](https://google.github.io/styleguide/tsguide.html)

# Refactoring

[Refactoring TypeScript](refactoring_typescript.md)

# Effective

[Effective TypeScript](effective_typescript.md)

# Design Pattern

[TypeScript Design Pattern](typescript_designpattern.md)

# Architecture

* [Typescript Clean Architecture | github](https://github.com/pvarentsov/typescript-clean-architecture)
  * java 의 Clean Architecture 와는 조금 다른 듯?
* [A TypeScript Stab at Clean Architecture](https://www.freecodecamp.org/news/a-typescript-stab-at-clean-architecture-b51fbb16a304/)
* [Evolution of a React folder structure and why to group by features right away](https://profy.dev/article/react-folder-structure)
* [React Folder Structure in 5 Steps [2022]](https://www.robinwieruch.de/react-folder-structure/)
  * 단순한 구조부터 복잡한 구조까지 단계별로 설명
* [bulletproof-react/docs/project-structure.md](https://github.com/alan2207/bulletproof-react/blob/master/docs/project-structure.md)
* [4 folder structures to organize your React & React Native project](https://reboot.studio/blog/folder-structures-to-organize-react-project/)
* [Project structure | Frontend Handbook](https://infinum.com/handbook/frontend/react/project-structure)
