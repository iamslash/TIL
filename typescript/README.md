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
