- [Materials](#materials)
- [Basic](#basic)
  - [Build & Run](#build--run)
  - [Hello World](#hello-world)
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
