- [Materials](#materials)
- [Basic](#basic)
  - [Build & Run](#build--run)
  - [Hello World](#hello-world)
  - [The Basics (Handbook)](#the-basics-handbook)
  - [Everyday Types (Handbook)](#everyday-types-handbook)
  - [Narrowing (Handbook)](#narrowing-handbook)
  - [More on Functions (Handbook)](#more-on-functions-handbook)
  - [Object Types (Handbook)](#object-types-handbook)
  - [Type Manipulation (Handbook)](#type-manipulation-handbook)
    - [Creating Types from Types](#creating-types-from-types)
    - [Generics](#generics)
    - [Keyof Type Operator](#keyof-type-operator)
    - [Typeof Type Operator](#typeof-type-operator)
    - [Indexed Access Types](#indexed-access-types)
    - [Conditional Types](#conditional-types)
    - [Mapped Types](#mapped-types)
    - [Template Literal Types](#template-literal-types)
  - [Classes (Handbook)](#classes-handbook)
  - [Modules (Handbook)](#modules-handbook)
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

## The Basics (Handbook)

```typescript
//////////////////////////////////////////////////////////////////////
// Boolean
{
    let isDone: Boolean = false;
}

//////////////////////////////////////////////////////////////////////
// Number
{
    let decimal: number = 6;
    let hex: number = 0xf00d;       // 0x
    let binary: number = 0b1010;    // 0b
    let octal: Number = 0o744;      // 0o
}

//////////////////////////////////////////////////////////////////////
// String
{
    let color: String = "blue";
    color = "red";

    let fullName: String = `David Sun`;
    let age: number = 18;
    let sentence: String = `Hello, my name is ${ fullName }.
    I'll be ${ age + 1 } years old next month.`;
}

//////////////////////////////////////////////////////////////////////
// Array
{
    let list: number[] = [1, 2, 3];
    let listAgain: Array<number> = [1, 2, 3];
}

//////////////////////////////////////////////////////////////////////
// Tuple
{
    let x: [string, number];
    x = ["Hello", 10];
    // x = [10, "Hello"]; // ERROR
    
    console.log(x[0].substring(1));
    // console.log(x[1].substring(1)); // ERROR

    // x[3] = "world" // ERROR
    // console.log(x[5].toString()); // ERROR
}

//////////////////////////////////////////////////////////////////////
// Enum
{
    {
        enum Color {
            Red,
            Green,
            Blue,
        }
        let c: Color = Color.Green;
    }
    {
        enum Color {
            Red = 1,
            Green,
            Blue,
        }
        let c: Color = Color.Green;
    }
    {
        enum Color {Red = 1, Green, Blue}
        let colorName: string = Color[2];
        console.log(colorName); // Green
    }
}

//////////////////////////////////////////////////////////////////////
// Any
{
    {
        let notSure: any = 4;
        notSure = "Maybe a string instead";
        notSure = false;
    }
    {
        let notSure: any = 4;
        notSure.ifItExists;
        notSure.toFixed();

        let prettySure: Object = 4;
        // prettySure.toFixed();  // ERROR

        let list: any[] = [1, true, "free"];
        list[1] = 100;
        console.log(list);
    }
}

//////////////////////////////////////////////////////////////////////
// Void
{
    function warnUser(): void {
        console.log("This is my warnUser.");
    }
    
    let unusable: void = undefined;
    // unusable = null;  // ERROR, just OK without '--strictNullChecks'
}

//////////////////////////////////////////////////////////////////////
// Null and Undefined
{
    let u: undefined = undefined;
    let n: null = null;
}

//////////////////////////////////////////////////////////////////////
// Never
{
    // Cannot reach end of function with never return.
    function error(message: string): never {
        throw new Error(message);
    }
    // Infer never return
    function fail() {
        return error("Something failed!!!");
    }
    function infiniteLoop(): never {
        while (true) {            
        }
    }
}

//////////////////////////////////////////////////////////////////////
// Object
declare function create(o: object | null): void;
{
    create({ prop: 0 });
    create(null);
    // create(42);         // ERROR
    // create("string");   // ERROR
    // create(false);      // ERROR
    // create(undefined);  // ERROR    
}

//////////////////////////////////////////////////////////////////////
// Type assertions (Type casting)
// Trust me I know what I am doing.
{
    let someValue: any = "This is a string.";
    // 0. angle-bracket
    let strLen: number = (<string>someValue).length;
    // 1. as
    let strLen2: number = (someValue as string).length;
}

//////////////////////////////////////////////////////////////////////
// let
// Prefer let than var
{
    let a: any = 3;
    console.log(a);
}
```

## Everyday Types (Handbook)

## Narrowing (Handbook)

## More on Functions (Handbook)

## Object Types (Handbook)

## Type Manipulation (Handbook)

### Creating Types from Types

### Generics

### Keyof Type Operator

### Typeof Type Operator

### Indexed Access Types

### Conditional Types

### Mapped Types

### Template Literal Types

## Classes (Handbook)

## Modules (Handbook)

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
