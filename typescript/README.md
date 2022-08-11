- [Materials](#materials)
- [Basic](#basic)
  - [Build & Run](#build--run)
  - [Hello World](#hello-world)

----

# Materials

* [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html)
  * Should read this.
* [TypeScript 환경에서 Redux를 프로처럼 사용하기 @ velog](https://velog.io/@velopert/use-typescript-and-redux-like-a-pro) 
* [TypeScript-Handbook 한글 문서](https://typescript-kr.github.io/)
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
