- [Materials](#materials)
- [Basic](#basic)
  - [Build & Run](#build--run)
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
  - [Decision Making Statements](#decision-making-statements)
  - [Looping Statements](#looping-statements)
  - [Branching Statements](#branching-statements)
  - [Collections](#collections)
  - [Collection Conversions](#collection-conversions)
  - [Sort](#sort)
  - [Search](#search)
  - [Multi Dimensional Arrays](#multi-dimensional-arrays)
  - [Enum](#enum)
  - [Generics](#generics)
- [Advanced](#advanced)
  - [`declare`](#declare)
  - [Function Definition With Interfaces](#function-definition-with-interfaces)
  - [Interface vs Type](#interface-vs-type)

----

# Materials

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
# tsc
$ npm install -g typescript
$ tsc a.ts
$ node a.js
```

```bash
# ts-node
$ npm install -g ts-node
$ ts-node a.ts
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
type World = "world";
type Greeting = `hello ${World}`;
console.log(Greeting)  // hello world
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
import * as util from "util";

console.log(util.format('%s %d %s', true, 4, 'Hello World'));
```

## Inspecting 

* [util.inspect | node.js](https://nodejs.org/api/util.html#utilinspectobject-options)

```ts
import * as util from "util";

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

## Decision Making Statements

if, if-else, switch

## Looping Statements

for, while, do-while

## Branching Statements

break, continue, return

## Collections

## Collection Conversions

## Sort

## Search

## Multi Dimensional Arrays

## Enum

## Generics

# Advanced

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
