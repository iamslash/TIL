- [Abstract](#abstract)
- [Essentials](#essentials)
- [Materials](#materials)
- [Basic](#basic)
  - [Editor Configuration](#editor-configuration)
    - [vscode](#vscode)
  - [Javascript Runtime Architecture](#javascript-runtime-architecture)
  - [JavaScript Engine How to work](#javascript-engine-how-to-work)
  - [Build and Run](#build-and-run)
  - [Print Out](#print-out)
  - [Keywords](#keywords)
  - [Min, Max Values](#min-max-values)
  - [Abs, Fabs](#abs-fabs)
  - [Bit Manipulation](#bit-manipulation)
  - [String](#string)
  - [Random](#random)
  - [Integer Division](#integer-division)
  - [Formatted String](#formatted-string)
  - [Inspecting Types](#inspecting-types)
  - [Inspecting Functions](#inspecting-functions)
  - [Data types](#data-types)
  - [Truthy Falthy](#truthy-falthy)
  - [Standard built-in objects (global objects) by category](#standard-built-in-objects-global-objects-by-category)
    - [Value properties](#value-properties)
    - [Function properties](#function-properties)
    - [Fundamental objects](#fundamental-objects)
    - [Error objects](#error-objects)
    - [Numbers and dates](#numbers-and-dates)
    - [Text processing](#text-processing)
    - [Indexed collections](#indexed-collections)
    - [Keyed collections](#keyed-collections)
    - [Structured Data](#structured-data)
    - [Control abstraction objects](#control-abstraction-objects)
    - [Reflection](#reflection)
    - [Internationalization](#internationalization)
    - [WebAssembly](#webassembly)
    - [Other](#other)
  - [Collections compared c++ container](#collections-compared-c-container)
  - [Collections](#collections)
    - [Object](#object)
    - [Array](#array)
    - [TypedArray](#typedarray)
    - [Set](#set)
    - [Map](#map)
    - [WeakSet](#weakset)
    - [WeakMap](#weakmap)
  - [Multidimensional Array](#multidimensional-array)
  - [template literals (template strings in ECMA 2015)](#template-literals-template-strings-in-ecma-2015)
  - [Tagged Template Literals (ES6)](#tagged-template-literals-es6)
  - [Sort](#sort)
  - [Variables](#variables)
  - [Operators](#operators)
  - [Control Flow](#control-flow)
  - [Loops](#loops)
  - [Functions](#functions)
    - [curly braces](#curly-braces)
    - [parenthese](#parenthese)
    - [anonymous function](#anonymous-function)
    - [named function](#named-function)
    - [closure](#closure)
    - [IIFE(Immediately-invoked function expression)](#iifeimmediately-invoked-function-expression)
    - [arrow function (ES6)](#arrow-function-es6)
  - [Method](#method)
  - [Exception](#exception)
  - [scope](#scope)
  - [Hoisting](#hoisting)
  - [TDZ (Temporal Dead Zone)](#tdz-temporal-dead-zone)
  - [this](#this)
  - [execution context](#execution-context)
  - [Prototype](#prototype)
  - [Class](#class)
  - [Expressions \& Operators](#expressions--operators)
  - [event loop](#event-loop)
  - [background](#background)
  - [task queue](#task-queue)
  - [micro task](#micro-task)
  - [Cookies](#cookies)
  - [HTML DOM](#html-dom)
- [Advanced](#advanced)
  - [Destructuring assignment](#destructuring-assignment)
  - [apply vs call](#apply-vs-call)
  - [Map vs Object](#map-vs-object)
  - [Computed property names](#computed-property-names)
  - [var, let, const](#var-let-const)
  - [Async, Await](#async-await)
  - [Generator function](#generator-function)
  - [Async Generator Function](#async-generator-function)
  - [Async vs Generator Function](#async-vs-generator-function)
  - [Iteration Protocols](#iteration-protocols)
  - [import from](#import-from)
  - [Shorthand property names](#shorthand-property-names)
  - [Duplicate Property Names](#duplicate-property-names)
  - [Decorator](#decorator)
- [Style Guide](#style-guide)
- [Refactoring](#refactoring)
- [Effective JavaScript](#effective-javascript)
- [Design Pattern](#design-pattern)
- [Architecutre](#architecutre)

-------------------------------------------------------------------------------

# Abstract

java script 에 대해 정리한다.

# Essentials

* [The modern javascript tutorial](https://javascript.info/)
  * mdn 보기 전에 꼭
* [JavaScript @ MDN](https://developer.mozilla.org/ko/docs/Web/JavaScript)
* [Javascript 핵심 개념 알아보기 - JS Flow](https://www.inflearn.com/course/%ED%95%B5%EC%8B%AC%EA%B0%9C%EB%85%90-javascript-flow/dashboard)
  * inflearn 유료 강좌 흐름
* [Javascript ES6+ 제대로 알아보기 - 초급](https://www.inflearn.com/course/ecmascript-6-flow/dashboard)
  * inflearn 유료 강좌 기초
* [Javascript ES6+ 제대로 알아보기 - 중급](https://www.inflearn.com/course/es6-2#)
  * inflearn 유료 강좌 중급
* [JavaScript 재입문하기 (JS ​튜토리얼)](https://developer.mozilla.org/ko/docs/A_re-introduction_to_JavaScript)
  * Custom Objects (new 다음에 function) 이 어떻게 동작하는지 알 수 있다.  
* [javascript @ tutorialspoint](https://www.tutorialspoint.com/javascript/)
* [JavaScript @ opentutorials](https://opentutorials.org/module/532)
  * 킹왕짱 JavaScript 기본문법
  * [video](http://www.youtube.com/playlist?list=PLuHgQVnccGMA4uSig3hCjl7wTDeyIeZVU)
* [JavaScript on Web @ opentutorials](https://opentutorials.org/course/838)
  * 웹개발을 위한 JavaScript 
* [초보자를 위한 JavaScript 200제](http://www.infopub.co.kr/new/include/detail.asp?sku=05000265)
  * [목차](http://www.infopub.co.kr/common/book_contents/05000265.html)
  * [src](http://www.infopub.co.kr/new/include/detail.asp?sku=05000265) 

# Materials

* [collections.js](http://www.collectionsjs.com/)
  * [src](https://github.com/montagejs/collections)
  * javascript collections library
* [How JavaScript works](https://blog.sessionstack.com/how-does-javascript-actually-work-part-1-b0bacc073cf)
  * JavaScript 가 어떻게 동작하는지 기술한 시리즈 게시글중 첫번째
  * [번역](https://engineering.huiseoul.com/%EC%9E%90%EB%B0%94%EC%8A%A4%ED%81%AC%EB%A6%BD%ED%8A%B8%EB%8A%94-%EC%96%B4%EB%96%BB%EA%B2%8C-%EC%9E%91%EB%8F%99%ED%95%98%EB%8A%94%EA%B0%80-%EC%97%94%EC%A7%84-%EB%9F%B0%ED%83%80%EC%9E%84-%EC%BD%9C%EC%8A%A4%ED%83%9D-%EA%B0%9C%EA%B4%80-ea47917c8442)
* [Secrets of the JavaScript Ninja - John Resig and Bear Bibeault](https://www.manning.com/books/secrets-of-the-javascript-ninja)
* [함수형 자바스크립트 프로그래밍 - 유인동](http://www.yes24.com/24/Goods/56885507?Acode=101)
  * 비함수형 언어인 자바스크립트를 이용하여 함수형 프로그래밍을 시도한다.
  * [src](https://github.com/indongyoo/functional-javascript)
* [john resig blog](https://johnresig.com/apps/learn/)
* [실행 컨텍스트](https://www.zerocho.com/category/Javascript/post/5741d96d094da4986bc950a0)
* [Underscore.js](http://underscorejs.org/)
  * functional programming helper

# Basic

## Editor Configuration

### vscode

EditorConfig for vscode [참고](https://suyeon96.tistory.com/33)

## Javascript Runtime Architecture

* [이벤트 루프와 매크로태스크, 마이크로태스크 | javascript.info](https://ko.javascript.info/event-loop)
* [어쨌든 이벤트 루프는 무엇입니까? | Philip Roberts | JSConf EU @ youtube](https://www.youtube.com/watch?v=8aGhZQkoFbQ)
  * [loupe](latentflip.com/loupe)
    * js even-loop
  * [loupe @ github](https://github.com/latentflip/loupe)
* [[JS] Event Loop : 마이크로태스크(Microtask)와 매크로태스크(Macrotask) 알아보기 @ velog](https://velog.io/@dami/JS-Microtask-Queue)
* [Tasks, microtasks, queues and schedules](https://jakearchibald.com/2015/tasks-microtasks-queues-and-schedules/)
  * callback queue 가 micro-task-queue, task-queue, animation-frame-queue 와 같이 3 가지가 있다는 것을 알 수 있다.
  
----

Js runtime engine 는 single threaded 이다. 기본적으로 web-browser 의 main thread
에서 js 가 실행된다. 따라서 js code 가 blocking 되면 ui 는 rendering 될 수 없다.
그렇다면 어떻게 aync 를 지원하는지 알아보자.

web-browser 는 js engine 이 제공하지 못하는 WebApi 를 제공한다. WebApi 를
이용하면 aync 구현이 가능하다. [JavaScript Visualized: Event
Loop](https://dev.to/lydiahallie/javascript-visualized-event-loop-3dif)

![](img/js-event-loop.gif)

위의 그림은 `v8` 과 같은 interpreter engine 의 구조이다. `micro-task-queue` 는
`macro-task-queue` 보다 우선순위가 높다.

`process.nextTickm, Promise, Object.observe, MutationObserver` 는 argument 로
전달된 call back 을 `micro-task-queue` 에 삽입한다. 

`setTimeout, setInterval, setImmediate, requestAnimationFrame, I/O, UI 렌더링`
는 argument 로 전달된 call back 을 `macro-task-queue` 에 삽입한다.

아래 그림의 회색 박스는 `v8` 와 같은 interpreter engine 이고 나머지는 `chrome`
과 같은 browser 라고 생각하자. callback queue 가 하나로 표현되어 있음을 주의
하자.

![](https://cdn-images-1.medium.com/max/800/1*4lHHyfEhVB0LnQ3HlhSs8g.png)

event-loop 알고리즘은 다음과 같다. 

1. `macro-task-queue` 에서 가장 오래된 태스크를 꺼내 실행한다.
2. `micro-taskqueue` 를 비울때까지 실행한다.
3. 렌더링한다.
4. `macro-task-queue` 가 비어있으면 기다린다.

예를 다음과 같은 code 를 살펴보자.

```js
setTimeout(() => alert("timeout"));

Promise.resolve()
  .then(() => alert("promise"));

alert("code");
// code
// promise
// timeout
```

## JavaScript Engine How to work

* [JavaScript Visualized: the JavaScript Engine](https://dev.to/lydiahallie/javascript-visualized-the-javascript-engine-4cdf)


## Build and Run

```js
$ node a.js
```

## Print Out

```js
console.log("Hello World");
```

## Keywords

> * [JavaScript Reserved Words](https://www.w3schools.com/js/js_reserved.asp)
> * [Reserved keywords in ES6 with example usage](https://medium.com/@wlodarczyk_j/reserved-keywords-in-es6-with-example-usage-ea0036f63fab)

```js
abstract   else        instanceof   switch
boolean    enum        int          synchronized
break      export      interface    this
byte       extends     long         throw
case       false       native       throws
catch      final       new          transient
char       finally     null         true
class      float       package      try
const      for         private      typeof
continue   function    protected    var
debugger   goto        public       void 
default    if          return       volatile
delete     implements  short        while
do         import      static       with
double     in          super
```

## Min, Max Values

```js
console.log(Number.MAX_SAFE_INTEGER); //  9007199254740991
console.log(Number.MIN_SAFE_INTEGER); // -9007199254740991

console.log(Number.MAX_VALUE) // 1.7976931348623157e+308
console.log(Number.MIN_VALUE) // 5e-324
```

## Abs, Fabs

```js
console.log(Math.abs(-23));     // 23
console.log(Math.abs('-23'));   // 23
console.log(Math.abs(-30 * 2)); // 60
console.log(Math.abs(-2.3));    // 2.3
```

## Bit Manipulation

```js
function dec2bin(dec) {
  return (dec >>> 0).toString(2);
}

a = 5;  // 0000 0101
b = 7;  // 0000 0111
a & b;  // 0000 0101
a | b;  // 0000 0111
a ^ b;  // 0000 0010
~a;     // 1111 1010
```

## String

* [Javascript에서 String을 Number타입으로 바꾸기](https://blog.outsider.ne.kr/361)

-----

```js
// charAt
const s = 'Hello World  ';
console.log(s.charAt(1));         // e
console.log(s.at(1));             // e
console.log(s.charAt(1) === s.at(1)) // true
console.log(s.charCodeAt(1));     // 101
console.log(s.startsWith('H'));   // true
console.log(s.endsWith("  "));    // true
console.log(s.indexOf('e'));      // 1
console.log(s.lastIndexOf('l'));  // 9
console.log(s.trim());            // Hello World
console.log(s.substring(0, 5))    // Hello
console.log(s.split('')[1]);      // e
console.log(s.split(' ')[1]);     // World
console.log(s.search('e'));       // 1
console.log(s.search('[o-z]'));   // 4
console.log(s.replace("Hello", "Bye"));  // Bye World
console.log(s.replaceAll("l", "k"));  // Hekko Workd
console.log(String.fromCharCode(189, 43, 190, 61)); // ½+¾=
console.log(s.concat(' ', "Really"));  // Hello World   Really
console.log(typeof(s.at(1)));

// Convert Number to String
var tt = 2
tt += "";
alert(typeof tt);   // Result : string
console.log(tt)

var tt = 2
alert(typeof tt);    // Result : number
tt = String(tt);
alert(typeof tt);    // Result : string

// Convert String to Number
tt = "2"
tt *= 1;
alert(typeof tt);    // Result : number
console.log(tt)

tt = "2"
alert(typeof tt);    // Result : string
tt = Number(tt);
alert(typeof tt);    // Result : number

// parseInt, parseFloat
var tt = "2"
alert(typeof tt);    // Result : string
tt = parseInt(tt);
alert(typeof tt);    // Result : number
            
tt = "2"
alert(typeof tt);    // Result : string
tt = parseFloat(tt);
alert(typeof tt);    // Result : number

parseInt("123.456");        // 123
parseInt("100mile");        // 100
parseInt("w25");               // NaN
parseInt("05");                  // 5
parseInt("09");                  // 0
parseInt("0x35");              // 53
parseInt("1101", 2);         // 13
parseInt("09", 10);            // 9
parseInt("10", 8);              // 8

parseFloat("123.456");       // 123.456
parseFloat("100.5mile");    // 100.5
parseFloat("w25");               // NaN
parseFloat("05");                  // 5
parseFloat("09");                  // 9
parseFloat("0x35");              // 0
```

## Random

```js
// get random float between 0 and 1
console.log(Math.random())

// get random float from min, max
function getRandomArbitrary(min, max) {
  return Math.random() * (max - min) + min;
}
console.log(getRandomArbitrary(1, 10))

// get random int from min, max
function getRandomInt(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min)) + min; 
}
console.log(getRandomInt(1, 10))
```

## Integer Division

```js
console.log(3 / 2);             // 1.5
consoel.log(Math.floor(3 / 2))  // 1
```

## Formatted String

> * [Text Formatting | mdn](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Text_formatting)

```js
console.log('Hello World')
// from ES6. focus on ` not '
// string interpolation
let a = 10;
console.log(`This is ${a}.`);

// format function
// https://coderwall.com/p/flonoa/simple-string-format-in-javascript
String.prototype.format = function() {
  a = this;
  for (k in arguments) {
    a = a.replace("{" + k + "}", arguments[k])
  }
  return a
}
console.log("Hello, {0}!".format("World"))

// Rounding floating-point numbers
(12.345).toFixed(2); // returns "12.35" (rounding!)
(12.3).toFixed(2); // returns "12.30" (zero padding)

// Exponential form
(33333).toExponential(2); // "3.33e+4"

// Hexadecimal and other bases
(3735928559).toString(16); // to base 16: "deadbeef"
parseInt("deadbeef", 16); // from base 16: 3735928559
```

## Inspecting Types

> [typeof | mdn](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/typeof)

`typeof` returns these.

| Type | Result |
|--|--|
| `Undefined` | `undefined` |
| `Null` | `object` |
| `Boolean` | `boolean` |
| `Number` | `number` |
| `BigInt` | `bigint` |
| `String` | `string` |
| `Symbol` | `symbol` |
| `Function` | `function` |
| Any other object | `object` |

```js
typeof({})    // object
typeof([])    // object
typeof(null)  // object
typeof(undefined) // undefined
```

`typeof(null)` is not `null` but `object`.
[reason](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/typeof#typeof_null) 

According to [The history of “typeof
null”](https://2ality.com/2013/10/typeof-null.html), It is a historical bug.

## Inspecting Functions

`node` 를 실행하고 `TAB` 을 누르면 리스트를 확인할 수 있다.

```console
> node
> console. <TAB>
```

## Data types

javascript 는 다음과 같은 타입들을 가지고 있다.

* Number
* String
* Boolean
* Symbol
* Object
  * Function
  * Array
  * Date
  * RegExp
* Null
* Undefined

```js
// Boolean
var a = true; 

// Null
var b = null;

// Undefined
// 값을 할당하지 않은 변수
var c = undefined

// Numbers
var n = 123;
var f = 120.50;

// Strings
var s = "hello world";

// Symbol
// Symbol 은 ECMAScript 6 에서 추가되었다. Symbol은 유일하고 변경 불가능한 (immutable) 기본값 (primitive value) 이다.
const symbol1 = Symbol();
const symbol2 = Symbol(42);
const symbol3 = Symbol('foo');

console.log(typeof symbol1);
// expected output: "symbol"
console.log(symbol3.toString());
// expected output: "Symbol(foo)"
console.log(Symbol('foo') === Symbol('foo'));
// expected output: false

// parseInt to make int
console.log(3/4);  // 0.75
console.log(parseInt(3/4)) // 0
console.log(parseInt(100, 2)) // 4, number whose base is 2
```

## Truthy Falthy

> [Falsy | mdn](https://developer.mozilla.org/en-US/docs/Glossary/Falsy)

A falsy (sometimes written falsey) value is a value that is considered false when encountered in a Boolean context.

```js
// The logical AND operator, &&
// Returns the first object when it is falsy.
console.log(false && "dog");  // false
console.log(0 && "dog");  // 0

// false in condition 
Boolean(false)        // Boolean
Boolean(null)         // Null
Boolean(undefined)    // Undefined
Boolean(0)            // Number
Boolean(-0)           // Number
Boolean(NaN)          // Number
Boolean(0n)           // BigInt
Boolean('')           // String
Boolean(document.all) // Object

// Comparisons of falsy values
false == 0            // true
false == -0           // true
false == 0n           // true
false == ''           // true
false == null         // false
false == undefined    // false
false == NaN          // false
false === 0           // false
false === ''          // false
false === -0          // false
false === 0n          // false

null == undefined // true
null == 0         // false
null == -0        // false
null == 0n        // false
null == ''        // false
null == NaN       // false
```

A trusy value is a value which is not falsy.

## Standard built-in objects (global objects) by category

* [Standard built-in objects @ MDN](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects)

The term "global objects" (or standard built-in objects) here is not to be
confused with the global object. Here, "global objects" refer to objects in the
global scope.

----

### Value properties

* Infinity
* NaN
* undefined
* globalThis

### Function properties

* eval()
* uneval() 
* isFinite()
* isNaN()
* parseFloat()
* parseInt()
* encodeURI()
* encodeURIComponent()
* decodeURI()
* decodeURIComponent()

### Fundamental objects

* Object
* [Function](standard_built_in_objs/function.md)
* Boolean
* Symbol

### Error objects

* Error
* AggregateError 
* EvalError
* InternalError
* RangeError
* ReferenceError
* SyntaxError
* TypeError
* URIError

### Numbers and dates

* Number
* BigInt
* Math
* [Date](standard_built_in_objs/date.md)

### Text processing

* String
* RegExp

### Indexed collections

* Array
* Int8Array
* Uint8Array
* Uint8ClampedArray
* Int16Array
* Uint16Array
* Int32Array
* Uint32Array
* Float32Array
* Float64Array
* BigInt64Array
* BigUint64Array

### Keyed collections

* Map
* Set
* WeakMap
* WeakSet

### Structured Data

* ArrayBuffer
* SharedArrayBuffer
* Atomics
* DataView
* JSON

### Control abstraction objects

* [Promise](standard_built_in_objs/promise.md)
* Generator
* GeneratorFunction
* AsyncFunction
* AsyncGenerator
* AsyncGeneratorFunction

### Reflection

* [Reflect](standard_built_in_objs/reflect.md)
* [Proxy](standard_built_in_objs/proxy.md)

### Internationalization

* Intl
* Intl.PluralRules
* Intl.collator
* Intl.RelativeTimeFormat
* Intl.DateTimeFormat
* Intl.Locale
* Intl.ListFormat
* Intl.NumberFormat

### WebAssembly

* WebAssembly
* WebAssembly.CompileError
* WebAssembly.Module
* WebAssembly.LinkError
* WebAssembly.Instance 
* WebAssembly.RuntimeError 
* WebAssembly.Memory 
* WebAssembly.Table 

### Other

* arguments

## Collections compared c++ container

| c++                  | js           |
|:---------------------|:-------------|
| `if, else`           | `if, else`   |
| `for, while`         | `for, while` |
| `array`              | ``           |
| `vector`             | `Array`      |
| `deque`              | ``           |
| `forward_list`       | ``           |
| `list`               | ``           |
| `stack`              | ``           |
| `queue`              | ``           |
| `priority_queue`     | ``           |
| `set`                | ``           |
| `multiset`           | ``           |
| `map`                | ``           |
| `multimap`           | ``           |
| `unordered_set`      | `Set`        |
| `unordered_multiset` | ``           |
| `unordered_map`      | `Map`        |
| `unordered_multimap` | ``           |


## Collections

* [JavaScript Collections](https://velog.io/@yesdoing/JavaScript-Collections)
* Object, Array, Typed Array, Set, Map, WeakSet, WeakMap
  
----

### Object

```js
a = new Object()
a.foo = 1; a.bar = 2; a.baz = 3

console.log(a)  
// { foo: 1, bar: 2, baz: 3 }

console.log(Object.entries(a));
// [ [ 'foo', 1 ], [ 'bar', 2 ], [ 'baz', 3 ] ]

console.log(Object.values(a));
// [ 1, 2, 3 ]
```

### Array

> * [Array @ MDN](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array)
> * [Array.from() @ MDN](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Array/from)

```js
// Create an Array
var fruits = ['사과', '바나나'];
console.log(fruits.length); // 2
// Create 10 undefined elements
var a = new Array(10);
console.log(typeof(a[0]))
// Update 10 elements with 1
a.fill(1);
// Loop over an Array
fruits.forEach(function(item, index, array) {
  console.log(item, index);
});
// Add an item to the end of an Array
let b = fruits.push('Orange');
// Remove an item from the end of an Array
let last = fruits.pop();
// Remove an item from the beginning of an Array
let first = fruits.shift();  
// Add an item to the beginning of an Array
let newLength = fruits.unshift('Strawberry');  
// Find the index of an item in the Array
fruits.push('Mango');
let pos = fruits.indexOf('Strawberry');
// Remove an item by index position
let removedItem = fruits.splice(pos, 1);
// Remove items from an index position
let vegetables = ['Cabbage', 'Turnip', 'Radish', 'Carrot'];
console.log(vegetables);
let pos = 1, let n = 2;
let removedItems = vegetables.splice(pos, n);
console.log(vegetables);
console.log(removeditems);
// Copy an Array
let shallowCopy = fruits.slice(0;)

// Array.from
console.log(Array.from('foo'));
// expected output: Array ["f", "o", "o"]
console.log(Array.from([1, 2, 3], x => x + x));
// expected output: Array [2, 4, 6]
let a = Array.from(Array(10), x => Array(10))
a[3][3] = 1
console.log(a)
```

### TypedArray

```js
Int8Array();
Uint8Array();
Uint8ClampedArray();
Int16Array();
Uint16Array();
Int32Array();
Uint32Array();
Float32Array();
Float64Array();
```

### Set

```js
var s = new Set();
s.add(1);
s.add("some text");
s.add("foo");

s.has(1); // true
s.delete("foo");
s.size; // 2

for (let e of s) console.log(e);
// 1
// "some text"
```

### Map

```js
var m = new Map();
m.set("dog", "woof");
m.set("cat", "meow");
m.set("elephant", "toot");
m.size; // 3
m.get("fox"); // undefined
m.has("bird"); // false
m.delete("dog");

for (var [key, val] of m) {
  console.log(key + " goes " + val);
}
// "cat goes meow"
// "elephant goes toot"
```

### WeakSet

```js
const ws = new WeakSet(); 
const age = {}; 
ws.add(age);
ws.has(age); // True
ws.delete(age)
```

### WeakMap

```js
const wm = new WeakMap(); 
const age = {}; 
const job = {}; 

wm.set(age, 11111); 
wm.set(job, 'air'); 
wm.has(job); // True
wm.delete(job)  
```

## Multidimensional Array

* [How to create multidimensional array](https://stackoverflow.com/questions/7545641/how-to-create-multidimensional-array)

----

```js
let a = Array.from(Array(2), x => Array(2).fill(0))
console.log(a)
// [ [ 0, 0 ], [ 0, 0 ] ]
a[1][1] = 1
console.log(a)
// [ [ 0, 0 ], [ 0, 1 ] ]
```

## template literals (template strings in ECMA 2015)

> * [Template literals](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Template_literals)
> * [템플릿 리터럴](https://poiemaweb.com/es6-template-literals)

----

Template literals are useful with backquote.

```js
// You can mix single, double quotes
const template = `You can mix 'single quotes' "double quotes".`;
console.log(template);

// You can include carrage return in string
const template = `<ul class="nav-items">
  <li><a href="#home">Home</a></li>
  <li><a href="#news">News</a></li>
  <li><a href="#contact">Contact</a></li>
  <li><a href="#about">About</a></li>
</ul>`;
console.log(template);

// String interpolation is convenient
const first = 'Ung-mo';
const last = 'Lee';
// ES5: 문자열 연결
console.log('My name is ' + first + ' ' + last + '.');
// "My name is Ung-mo Lee."
// ES6: String Interpolation
console.log(`My name is ${first} ${last}.`);
// "My name is Ung-mo Lee."

// ${} will be converted to string
console.log(`1 + 1 = ${1 + 1}`); // "1 + 1 = 2"
```

## Tagged Template Literals (ES6)

* [Tagged Template | MSDN](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Template_literals#tagged_templates)

Template Literals 앞에 Tag 를 부착할 수 있다. Tag 는 곧 function 이다. Template Literals 는
Tag 의 argument 로 전달된다.

```js
const person = 'Mike';
const age = 28;
const score = 10;

function myTag(strings, personExp, ageExp, aaaExp) {
  const str0 = strings[0]; // "That "
  const str1 = strings[1]; // " is a "
  const str2 = strings[2]; // "."

  const ageStr = ageExp > 99 ? 'centenarian' : 'youngster';

  // We can even return a string built using a template literal
  return `${str0}${personExp}${str1}${ageStr}${str2}${aaaExp}`;
}

const thatOutput = myTag`That ${person} is a ${age}. ${score}`;
const thisOutput = myTag`This ${person} is a ${age}. ${score}`;

console.log(thatOutput);
// That Mike is a youngster. 10
console.log(thisOutput);
// This Mike is a youngster. 10
```

## Sort

```js
//// sort ascending
a = [5, 1, 4, 2, 3]
a.sort();
a.sort((a, b) => (a < b ? -1 : 1))
a.sort((a, b) => {a < b ? -1 : 1})
a.sort((a, b) => a < b ? -1 : 1)

//// sort descending
// obj.sort().reverse() is the best for the performance.
//  
a.sort().reverse();
a.sort((a, b) => (a > b ? -1 : 1))
a.sort((a, b) => b.localeCompare(a))
```

## Variables

JavaScript에서 새로운 변수는 `let, const, var` 키워드로 선언한다. 변수에 값을 지정하지 않고 선언한 경우 타입은 `undefined` 이다.

```js
//// let
// 블록 유효 범위 변수를 선언. 선언 된 변수는 변수가 포함 된 함수 블록에서 사용할 수 있다.
// i 는 여기에서 보이지 않는다.
for (let i = 0; i < 5; i++) {
  // i 는 여기서 유효하다.
}
// i 는 여기에서 보이지 않는다.

//// const
// 블록 유효 범위 변수를 선언. 그러나 선언이후 수정할 수 없다.
const Pi = 3.14; // 변수 Pi 설정 
Pi = 1; // 상수로 설정된 변수는 변경 할 수 없기 때문에 애러 발생.

//// var
// 변수가 선언 된 함수 블록에서 사용가능.
// i 는 여기에서 접근가능.
for (var i = 0; i < 5; i++) {
  // i 는 여기서 유효하다.
}
// i 는 여기에서 접근가능.
```

## Operators

```js
//// Arithmetic Operators
var a = 33;
var b = 10;
var c = "Test";

console.log(`a + b = ${a + b}`); // 43
console.log(`a - b = ${a + b}`); // 23
console.log(`a / b = ${a + b}`); // 3.3
console.log(`a % b = ${a + b}`); // 3
console.log(`a + b + c = ${a + b + c}`); // 43Test
console.log(`++a = ${++a}`); // 34
console.log(`++a = ${--b}`); // 9

//// comparison operator with value, type
//
// undefined == null => true
// undefined === null => false

//// Comparison Operators
var a = 10;
var b = 20;

console.log(`(a == b) => ${a == b}`); // false
console.log(`(a < b) => ${a < b}`); // true
console.log(`(a > b) => ${a > b}`); // false
console.log(`(a != b) => ${a != b}`); // true
console.log(`(a >= b) => ${a >= b}`); // false
console.log(`(a <= b) => ${a <>= b}`); // true

//// Logical Operators
var a = true;
var b = false;

console.log(`(a && b) => ${a && b}`); // false
console.log(`(a || b) => ${a || b}`); // true
console.log(`!(a && b) => ${!(a && b)}`); // true

//// Bitwise Operators
var a = 2; // Bit presentation 10
var b = 3; // Bit presentation 11

console.log(`(a & b) => ${a & b}`); // 2
console.log(`(a | b) => ${a | b}`); // 3
console.log(`(a ^ b) => ${a ^ b}`); // 1
console.log(`(~b) => ${~b}`); // -4
console.log(`(a << b) => ${a << b}`); // 16
console.log(`(a >> b) => ${a >> b}`); // 0

//// Assignment Operators
var a = 33;
var b = 10;

console.log(`(a = b) => ${a = b}`); // 10
console.log(`(a += b) => ${a += b}`); // 20
console.log(`(a -= b) => ${a -= b}`); // 10
console.log(`(a *= b) => ${a *= b}`); // 100
console.log(`(a /= b) => ${a /= b}`); // 10
console.log(`(a %= b) => ${a %= b}`); // 0

//// Miscellaneous Operators
var a = 10;
var b = 20;

console.log(`(a > b) ? 100 : 200) => ${(a > b) ? 100 : 200)}`); // 200
console.log(`(a < b) ? 100 : 200) => ${(a < b) ? 100 : 200)}`); // 100

//// typeof Operators
var a = 10;
var b = "String";

console.log(`(typeof a => ${typeof a == "string" ? "String" : "Numeric")}`); // Numeric
console.log(`(typeof b => ${typeof b == "string" ? "String" : "Numeric")}`); // String

//// !!
// !! 연산자는 JavaScript에서 불리언 형식으로 값을 변환하는 데 사용됩니다. 
// 첫 번째 !는 값의 논리 부정을 수행합니다. 즉, truthy 값을 false로, falsy 값을 true로 변환합니다.
// 두 번째 !는 다시 그 값을 부정하여 원래의 불리언 값으로 되돌립니다.
const value = "hello";
const booleanValue = !!value; // true
```

## Control Flow

```js
//// if else if
var book = "maths";
if( book == "history" ) {
   document.write("<b>History Book</b>");
} else if( book == "maths" ) {
   document.write("<b>Maths Book</b>");
} else if( book == "economics" ) {
   document.write("<b>Economics Book</b>");
} else {
   document.write("<b>Unknown Book</b>");
}
//// switch
var grade = 'A';
document.write("Entering switch block<br />");
switch (grade) {
  case 'A': document.write("Good job<br />");
  break;

  case 'B': document.write("Pretty good<br />");
  break;

  case 'C': document.write("Passed<br />");
  break;

  case 'D': document.write("Not so good<br />");
  break;

  case 'F': document.write("Failed<br />");
  break;

  default:  document.write("Unknown grade<br />")
}
document.write("Exiting switch block");
// Entering switch block
// Good job
// Exiting switch block         
```

## Loops

```js
//// while
var count = 0;
document.write("Starting Loop ");

while (count < 10) {
  document.write("Current Count : " + count + "<br />");
  count++;
}

document.write("Loop stopped!");
// Starting Loop
// Current Count : 0
// Current Count : 1
// Current Count : 2
// Current Count : 3
// Current Count : 4
// Current Count : 5
// Current Count : 6
// Current Count : 7
// Current Count : 8
// Current Count : 9
// Loop stopped!

//// do while
var count = 0;

document.write("Starting Loop" + "<br />");
do {
   document.write("Current Count : " + count + "<br />");
   count++;
}

while (count < 5);
document.write ("Loop stopped!");
// Starting Loop
// Current Count : 0 
// Current Count : 1 
// Current Count : 2 
// Current Count : 3 
// Current Count : 4
// Loop Stopped!       

//// for 
var count;
document.write("Starting Loop" + "<br />");

for(count = 0; count < 10; count++){
   document.write("Current Count : " + count );
   document.write("<br />");
}

document.write("Loop stopped!");

//// for in
var aProperty;
document.write("Navigator Object Properties<br /> ");        
for (aProperty in navigator) {
   document.write(aProperty);
   document.write("<br />");
}
document.write ("Exiting from the loop!");
```

## Functions

### curly braces

코드가 실행된다.

```js
var a = function(a, b) {
    return a + b;
}
```

### parenthese

함수를 정의한다.

```js
var a = function(a, b) {
    return a + b;
}
```

함수를 실행한다. execution context 가 만들어 진다.

```js
(function(a, b) {
    return a + b;
})(1, 2);
```

코드가 실행된다.

```js
var a;
if (a = 5)
    console.log(a);
```

### anonymous function

```js
var a = function(a, b) {
    return a + b;
}
```

### named function


```js
var a = function f(a, b) {
    return a + b;
}
```

### closure

함수와 함수가 선언된 환경의 조합

A closure is the combination of a function and the lexical environment within which that function was declared.

```js
var counter = function() {
  var count = 0;
  function changeCount(number) {
    count += number;
  }
  return {
    increase: function() {
      changeCount(1);
    },
    decrease: function() {
      changeCount(-1);
    },
    show: function() {
      console.log(count);
    }
  }
};
var counterClosure = counter();
counterClosure.increase();
counterClosure.show(); // 1
counterClosure.decrease();
counterClosure.show(); // 0
```

### IIFE(Immediately-invoked function expression)

```js
(function(a, b) {
    return a + b;
})(1, 2);
```

### arrow function (ES6)

```js
var a = (a, b) => a + b;
var b = (a, b) => {
    var result = a + b;
    return result;
}
```

`=> ()` is same with `=> {return()}`. [Arrow functions and the use of parentheses () or {} or ({})](https://stackoverflow.com/questions/49425755/arrow-functions-and-the-use-of-parentheses-or-or)

```js
const FilterLink = ({ filter, children }) => ( // <-- implicit return 
  <NavLink
    to={filter === 'SHOW_ALL' ? '/' : `/${ filter }`}
    activeStyle={ {
      textDecoration: 'none',
      color: 'black'
    }}
  >
    {children}
  </NavLink>
)

const FilterLink = ({ filter, children }) => {
   return (
      <NavLink
        to={filter === 'SHOW_ALL' ? '/' : `/${ filter }`}
        activeStyle={ {
          textDecoration: 'none',
          color: 'black'
        }}
      >
        {children}
      </NavLink>
    )
}
```

## Method

`.` 뒤의 함수는 method 이다.

```js
var obj = {
  a: 1,
  b: function foo() {
    console.log(this);
  };
  c: function() {
    console.log(this.a);
  }
}
obj.b(); // b is a method
obj.c(); // c is a method
```

## Exception

> [throw | mdn](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Statements/throw)

```js
function getRectArea(width, height) {
  if (isNaN(width) || isNaN(height)) {
    throw new Error('Parameter is not a number!');
  }
}

try {
  getRectArea(3, 'A');
} catch (e) {
  console.error(e);
  // Expected output: Error: Parameter is not a number!
}
```

## scope

변수의 유효범위를 말한다. 함수가 정의 되면 scope 가 만들어진다. ES6 부터는 block 역시 scope 를 생성한다.

```js
var name = 'zero';
function log() {
  console.log(name);
}

function wrapper() {
  name = 'nero';
  log();
}
wrapper(); // nero
```

```js
var name = 'zero';
function log() {
  console.log(name);
}

function wrapper() {
  var name = 'nero';
  log();
}
wrapper(); // zero
```


## Hoisting

* [JavaScript Visualized: Hoisting](https://dev.to/lydiahallie/javascript-visualized-hoisting-478h)
* [Hoisting | mdn](https://developer.mozilla.org/en-US/docs/Glossary/Hoisting)

-----

declaration of **functions, variables or classes** 를 interpreter 가 실행하기
전에 해당 scope 의 꼭대기로 끌어올리는 것을 말한다. 즉, 선언하기 전에
ReferenceError 발생 없이 사용할 수 있다는 의미이다.
[HoistableDeclaration](https://tc39.es/ecma262/#prod-HoistableDeclaration) 이
ECMA 에 정의되어 있긴 하지만 Hoisting 이 official term 은 아닌 것 같다.

Hoisting 은 다음과 같이 3 가지 종류가 있다.

* Hoisted variable 를 declaration 실행 이전에 사용할 수 있다. (**Value Hoisting**)
  * `function, function*, async function, async function*`
* Hoisted variable 를 declaration 실행 이전에 사용할 수 있지만 그 값은
  `undefined` 이다. (**Declaration Hoisting**)
  * `var`
* declaration 실행 이전에 사용하면 ReferenceError 를 발생한다. (**Non-Hoisted**)
  * `let, const, class`
  * [TDZ](#tdz-temporal-dead-zone) 참고.

```js
const x = 1;
{
  console.log(x); // ReferenceError
  const x = 2;
}
```

## TDZ (Temporal Dead Zone)

`let, const` 로 변수를 선언하면 실제 선언되기 전에 사용할 수 없다. 즉, TDZ 에서
변수를 사용하면 Reference 가 발생한다. `let, const` 가 선언된 block 의 시작부터
선언된 행까지를 TDZ (Temporal Dead Zone) 이라고 한다.

```js
{
  // TDZ starts at beginning of scope
  console.log(bar); // undefined
  console.log(foo); // ReferenceError
  var bar = 1;
  let foo = 2; // End of TDZ (for foo)
}
```

The TDZ and typeof

```js
// results in a 'ReferenceError'
console.log(typeof i);
let i = 10;
```

TDZ combined with lexical scoping


```js
function test() {
  var foo = 33;
  if (foo) {
    let foo = foo + 55; // ReferenceError
  }
}
test();
```

```js
// "n.a" is "a" of "let n"
function go(n) {
  // n here is defined!
  console.log(n); // { a: [1, 2, 3] }

  for (let n of n.a) {
    //          ^ ReferenceError
    console.log(n);
  }
}
go({ a: [1, 2, 3] });
```

## this

* 전역공간에서 `this` 는 `window/global` 이다. 
* 함수내부에서 `this` 는 `window/global` 이다.
* 메소드 호출시 `this` 는 `.` 앞의 객체이다.
* callback 에서 `this` 는 함수내부에서와 같다.

```js
function a(x, y, z) {
  console.log(this, x, y, z);
}
var b = {
  c: 'eee'
}
a.call(b, 1, 2, 3)        // { c: 'eee' } 1 2 3
a.apply(b, [1, 2, 3])     // { c: 'eee' } 1 2 3
var c = a.bind(b);        
c(1, 2, 3);               // { c: 'eee' } 1 2 3
var d = a.bind(b, 1, 2);
d(3);                     // { c: 'eee' } 1 2 3
```

* 생성자함수에서 `this` 는 인스턴스이다.

```js
function Person(n, a) {
  this.name = n;
  this.age = a;
}
var foo = new Person('Foo', 30);
console.log(foo);  // Person { name: 'Foo', age: 30 }
```

## execution context

* [JavaScript Visualized: Scope (Chain)](https://dev.to/lydiahallie/javascript-visualized-scope-chain-13pd)

-----

`global context` 생성후 함수 호출 할때 마다 `execution context` 가 생성된다.
`execution context` 는 `arguments, variable, scope chain, this` 가 저장된다.
함수가 실행될때 그 함수의 `execution context` 에 변수가 없다면 `scope chain` 을
따라 올라가며 검색한다. 함수 실행이 종료되면 `execution context` 는 사라진다.

## Prototype

* [JavaScript Visualized: Prototypal Inheritance](https://dev.to/lydiahallie/javascript-visualized-prototypal-inheritance-47co)

-----

Prototypes are the mechanism by which JavaScript objects inherit features from
one another. 

Contructor function 이 있을 때 new 연산자를 사용하여 instance 를 생성했다고
해보자. 이때 Constructor function 의 `prototype` 과 instance 의 `__prototype__` 은
같은 객체를 가리킨다. 

`Array` constructor 로 new 연산자를 사용하여 `[1, 2, 3]` 을 생성했다고 해보자.
이때, `Array` 는 `from(), isArray(), of(), arguments, length, name, prototype`
의 property 를 갖고 있다. 이때 `Array.prototype` 은 `[1, 2, 3].__prototype__` 과
같다. `Array.prototype` 은 다시 `concat(), filter(), forEach(), map(), push(),
pop()` 등의 property 를 갖는다.

또한 `__prototype__` 은 생략 가능하다. 따라서 다음의 표현은 모두 같다.

```js
[construct].prototype
[instance].__proto__
[instance]
Object.getPrototypeOf([instance])
```

또한 생성자 함수는 다음과 같은 방법으로 접근 가능하다.

```js
[CONSTRUCTOR]
[CONSTRUCTOR].prototype.constructor
(Object.getPrototypeOf([instance])).constructor
[instance].__proto__.constructor
[instance].constructor
```

`prototype` 을 이용하여 메소드를 상속해 보자. 다음과 같이 Person constructor 를 선언하자.

```js
function Person(n, a) {
  this.name = n;
  this.age = a;
}
var foo = new Person('Foo', 30);
var bar = new Person('Bar', 25);
foo.setOlder = function() {
  this.age += 1;
}
foo.getAge = function() {
  return this.age;
}
bar.setOlder = function() {
  this.age += 1;
}
bar.getAge = function() {
  return this.age;
}
```

`prototype` 을 이용하여 setOlder, getAge 와 같은 duplicates 를 제거해보자.

```js
function Person(n, a) {
  this.name = n;
  this.age = a;
}
Person.prototype.setOlder = function() {
  this.age += 1;
}
Person.prototype.getAge = function() {
  return this.age;
}
var foo = new Person('Foo', 30);
var bar = new Person('Bar', 25);
```

`foo.__proto__.setOlder(), foo.__proto__.getAge()` 는 NaN 이다. this 가
`__proto__` 이기 때문이다. `foo.setOlder(); foo.getOlder();` 는 정상이다.

이번에는 prototype chaining 을 활용하는 예를 살펴보자.

```js
var arr = [1, 2, 3];
console.log(arr.toString()); 
// 1,2,3
```

출력의 형태를 바꾸기 위해 다음과 같이 수정한다.

```js
var arr = [1, 2, 3];
arr.toString = function() {
  return this.join('_');
}
console.log(arr.toString());
// 1_2_3
```

call 함수를 사용하여 this 를 바꾸어 실행해보자.

```js
var arr = [1, 2, 3];
arr.toString = function() {
  return this.join('_');
}
console.log(arr.toString());
// 1_2_3
console.log(arr.__proto__.toString.call(arr));
// 1,2,3
console.log(arr.__proto__.__proto__.toString.call(arr));
// [object Array]
```

이번에는 Array.prototype 에 toString 을 정의하자.

```js
var arr = [1, 2, 3];
Array.prototype.toString = function() {
  return '[' + this.join(', ') + ']';
}
console.log(arr.toString());
// [1, 2, 3]
console.log(arr.__proto__.toString.call(arr));
// [1, 2, 3]
console.log(arr.__proto__.__proto__.toString.call(arr));
// [object Array]
```

## Class

Array 는 class 이다. Array 를 new 연산자를 이용하여 생성한 `[1, 2, 3]` 은
instance 이다. Array 는 다음과 같이 static methods, static properties, methods
로 구성된다.

```
Array.from()                -- static methods
     .isArray()             -- static methods
     .of()                  -- static methods

     .arguments             -- static properties
     .length                -- static properties
     .name                  -- static properties
     
     .prototype.concat()    -- prototype methods
               .filter()    -- prototype methods
               .forEach()   -- prototype methods
               .map()       -- prototype methods
               .push()      -- prototype methods
               .pop()       -- prototype methods
```

다음과 같이 Person 을 정의해 보자.

```js
function Person(n, a) {
  this._name = n;
  this._age = a;
}
// static methods
Person.getInformations = function(instance) {
  return {
    name: instance._name;
    age: instance._age;
  }
}
// prototype method
Person.prototype.getName = function() {
  return this._name;
}
// prototype method
Person.prototype.getAge = function() {
  return this._age;
}
```

다음은 prototype chaining 을 활용하여 Person, Employee  의 상속을 구현한 것이다.

```js
function Person(n, a) {
  this.name = n || 'noname';
  this.age = a || 'unknown';
}
Person.prototype.getName = function() {
  return this.name;
}
Person.prototype.getAge = function() {
  return this.age;
}
function Employee(n, a, p) {
  this.name = n || 'noname';
  this.age = a || 'unknown';
  this.position = p || 'unknown';
}
// 다음은 상속 구현의 핵심이다.
Employee.prototype = new Person();
Employee.prototype.constructor = Employee;
Employee.prototype.getPosition = function() {
  return this.position;
}

var foo = new Employee('Foo', 30, 'CEO');
console.dir(foo);
```

그러나 `Person.prototype` 은 `name, age` prototype 을 갖고 있다. 이것을 `Bridge` constructor 를 만들어서 해결해보자.

```js
function Person(n, a) {
  this.name = n || 'noname';
  this.age = a || 'unknown';
}
Person.prototype.getName = function() {
  return this.name;
}
Person.prototype.getAge = function() {
  return this.age;
}
function Employee(n, a, p) {
  this.name = n || 'noname';
  this.age = a || 'unknown';
  this.position = p || 'unknown';
}
function Bridge() {}
Bridge.prototype = Person.prototype;
Employee.prototype = new Bridge();
Employee.prototype.constructor = Employee;
Employee.prototype.getPosition = function() {
  return this.position;
}
```

위의 방법은 ES5 에서 자주 등장한다. 더글라스는 extendClass 함수를 사용하여 재활용할 것을 제안했다.

```js
var extendClass = (function() {
  function Bridge(){}
  return function(Parent, Child) {
    Bridge.prototype = Parent.prototype;
    Child.protype = new Bridge();
    Child.prototype.constructor = Child;
  }
})();
extendClass(Person, Employee);
Employee.prototype.getPosition = function() {
  return this.position;
}
```

이번에는 superClass 를 이용하여 `name, age` 의 duplicates 를 해결해 보자.

```js
function Person(n, a) {
  this.name = n || 'noname';
  this.age = a || 'unknown';
}
Person.prototype.getName = function() {
  return this.name;
}
Person.prototype.getAge = function() {
  return this.age;
}
function Employee(n, a, p) {
  this.superClass(n, a); // ***
  this.position = p || 'unknown';
}
function Bridge() {}
Bridge.prototype = Person.prototype;
Employee.prototype = new Bridge();
Employee.prototype.constructor = Employee;
Employee.prototype.getPosition = function() {
  return this.position;
}
var extendClass = (function() {
  function Bridge(){}
  return function(Parent, Child) {
    Bridge.prototype = Parent.prototype;
    Child.protype = new Bridge();
    Child.prototype.constructor = Child;
    Child.prototype.superClass = Parent; // ***
  }
})();
extendClass(Person, Employee);
Employee.prototype.getPosition = function() {
  return this.position;
}
```

그러나 ECMA6 에서는 extends 를 활용하여 더욱 간단히 상속을 구현할 수 있다.

```js
class Person {
  constuctor(n, a) {
    this.name = n || 'noname';
    this.age = a || 'unknown';
  }
  getName() {
    return this.name;
  }
  getAge() {
    return this.age;
  }
}
class Employee extends Person {
  constructor(n, a, p) {
    super(n, a);
    this.position = p || 'noposition';
  }
  getPosition() {
    return this.position;
  }
}
```

## Expressions & Operators

```js
// unsigned right shift (>>>)
console.log(`${(5).toString(2)}`); // 0000 0101
console.log(`${(5 >>> 2).toString(2)}`); // 0000 0001
console.log(`${(-5).toString(2)}`); // -101
console.log(`${(-5 >>> 0).toString(2)}`); // 1111 1101

// Optional chaining (?.)
// Returns undefined when target object is undefined or null.
const adventurer = {
  name: 'Alice',
  cat: {
    name: 'Dinah'
  }
};
const dogName = adventurer.dog?.name;
console.log(dogName);                               // undefined
console.log(adventurer.someNonExistentMethod?.());  // undefined

// Strict equality (===)
console.log(1 === 1);             // true
console.log('hello' === 'hello'); // true
console.log('1' ===  1);          // false
console.log(0 === false);         // false

// Spread syntax (...)
function sum(x, y, z) {
  return x + y + z;
}
const numbers = [1, 2, 3];
console.log(sum(...numbers));           // 6
console.log(sum.apply(null, numbers));  // 6

// Logical AND assignment (&&=)
// save when x is truthy.
let a = 1, b = 0;
a &&= 2;  // a = a && 2
console.log(a);  // 2
b &&= 2;  // b = b && 2
console.log(b);  // 0

// Logical OR assignment (||=)
// save when x is falthy.
const a = { duration: 50, title: '' };
a.duration ||= 10;
console.log(a.duration);  // 50
a.title ||= 'title is empty.';
console.log(a.title);  // "title is empty"

// Nullish coalescing operator (??)
const foo = null ?? 'default string';
console.log(foo);  // "default string"
const baz = 0 ?? 42;
console.log(baz);  // 0
```

## event loop

```js
function run() {
  console.log('동작');
}
console.log('시작');
setTimeout(run, 0);
console.log('끝');
```

[Loupe is a little visualisation to help you understand how JavaScript's call stack/event loop/callback queue interact with each other.](http://latentflip.com/loupe/)

## background

WIP...

## task queue

WIP...

## micro task

WIP...

## Cookies

쿠키는 `Expires, Domain, Path, Secure, Name=Value` 와 같이 5가지로 구성되어 있다. 다음은 쿠키를 세팅하는 예이다.

```js
document.cookie = "key1 = value1;key2 = value2;expires = date";
```

다음은 쿠키에 고객이름을 세팅하는 예이다.

```html
<html>
   <head>   
      <script type = "text/javascript">
         <!--
            function WriteCookie() {
               if( document.myform.customer.value == "" ) {
                  alert("Enter some value!");
                  return;
               }
               cookievalue = escape(document.myform.customer.value) + ";";
               document.cookie = "name=" + cookievalue;
               document.write ("Setting Cookies : " + "name=" + cookievalue );
            }
         //-->
      </script>      
   </head>
   
   <body>      
      <form name = "myform" action = "">
         Enter name: <input type = "text" name = "customer"/>
         <input type = "button" value = "Set Cookie" onclick = "WriteCookie();"/>
      </form>   
   </body>
</html>
```

다음은 쿠키를 읽어오는 예이다.

```html
<html>
   <head>   
      <script type = "text/javascript">
         <!--
            function ReadCookie() {
               var allcookies = document.cookie;
               document.write ("All Cookies : " + allcookies );
               
               // Get all the cookies pairs in an array
               cookiearray = allcookies.split(';');
               
               // Now take key value pair out of this array
               for(var i=0; i<cookiearray.length; i++) {
                  name = cookiearray[i].split('=')[0];
                  value = cookiearray[i].split('=')[1];
                  document.write ("Key is : " + name + " and Value is : " + value);
               }
            }
         //-->
      </script>      
   </head>
   
   <body>     
      <form name = "myform" action = "">
         <p> click the following button and see the result:</p>
         <input type = "button" value = "Get Cookie" onclick = "ReadCookie()"/>
      </form>      
   </body>
</html>
```

다음은 쿠키의 유효기간을 설정하는 에이다.

```html
<html>
   <head>   
      <script type = "text/javascript">
         <!--
            function WriteCookie() {
               var now = new Date();
               now.setMonth( now.getMonth() + 1 );
               cookievalue = escape(document.myform.customer.value) + ";"
               
               document.cookie = "name=" + cookievalue;
               document.cookie = "expires=" + now.toUTCString() + ";"
               document.write ("Setting Cookies : " + "name=" + cookievalue );
            }
         //-->
      </script>      
   </head>
   
   <body>
      <form name = "myform" action = "">
         Enter name: <input type = "text" name = "customer"/>
         <input type = "button" value = "Set Cookie" onclick = "WriteCookie()"/>
      </form>      
   </body>
</html>
```

다음은 쿠키를 삭제하는 예이다. expires 를 현재시간을 설정한다.

```html
<html>
   <head>   
      <script type = "text/javascript">
         <!--
            function WriteCookie() {
               var now = new Date();
               now.setMonth( now.getMonth() - 1 );
               cookievalue = escape(document.myform.customer.value) + ";"
               
               document.cookie = "name=" + cookievalue;
               document.cookie = "expires=" + now.toUTCString() + ";"
               document.write("Setting Cookies : " + "name=" + cookievalue );
            }
          //-->
      </script>      
   </head>
   
   <body>
      <form name = "myform" action = "">
         Enter name: <input type = "text" name = "customer"/>
         <input type = "button" value = "Set Cookie" onclick = "WriteCookie()"/>
      </form>      
   </body>
</html>
```

## HTML DOM

다음은 DOM 의 구조를 표현한 그림이다.

![](https://www.tutorialspoint.com/javascript/images/html-dom.jpg)

# Advanced

## Destructuring assignment

```js
// Array Destructuring
const numbers = [1, 2, 3];
const [first, second, third] = numbers;
console.log(first, second, third);  // 1 2 3

// Object Destructuring
const person = { name: 'John', age: 30 };
const { name, age } = person;
console.log(name, age);  // John 30

// Default values
const [a = 10, b = 5] = [1];
console.log(a);  // 1
console.log(b);  // 5

// Nested Destructuring
const nested = { foo: { bar: 42 } };
const { foo: { bar } } = nested;
console.log(bar);  // 42

// Rest syntax
const [x, ...rest] = [10, 20, 30, 40];
console.log(x);  // 10
console.log(rest);  // [20, 30, 40]

// Swapping variables
let p = 1, q = 2;
[p, q] = [q, p];
console.log(p);  // 2
console.log(q);  // 1

// Function parameter destructuring
const user = { id: 42, isVerified: true };
function printUserInfo({ id, isVerified }) {
  console.log(`ID: ${id}, Verified: ${isVerified}`);
}
printUserInfo(user);  // ID: 42, Verified: true
```

## apply vs call

둘다 [Function](standard_built_in_objs/function.md) 의 methods 이다. apply 는 args 가 array 이고 call 은 unwinded array 이다.

```js
function a(x, y, z) {
  console.log(this, x, y, z);
}
var b = {
  c: 'eee'
}
a.call(b, 1, 2, 3)        // { c: 'eee' } 1 2 3
a.apply(b, [1, 2, 3])     // { c: 'eee' } 1 2 3
```

## Map vs Object

> * [Objects vs maps | msdn](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map#objects_vs._maps)

Object 와 map 는 둘다 key, value 를 저장할 수 있다. 주요 차이는 다음과 같다.

* Object 는 prototype 이 있다. Map 은 없다.
* Map 은 size 를 쉽게 얻을 수 있다. Object 는 size 를 수동으로 저장해야 한다.

## Computed property names

> [Computed property names](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Object_initializer#computed_property_names)

Object 의 property name 을 runtime 에 수정하여 property value 를 저장할 수 있다.

```js
// Computed property names (ES2015)
let i = 0
let a = {
  ['foo' + ++i]: i,
  ['foo' + ++i]: i,
  ['foo' + ++i]: i
}

console.log(a.foo1) // 1
console.log(a.foo2) // 2
console.log(a.foo3) // 3
```

## var, let, const

* [let | mdn](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/let)
* [var, let, const 차이점은?](https://gist.github.com/LeoHeo/7c2a2a6dbcf80becaaa1e61e90091e5d)

----

* **var** 는 global-scoped 혹은 function-scoped 이다.
  * 함수 밖에서 선언했을 때 global-scoped 이다.
  * 함수 안에서 선언했을 때 function-scoped 이다.
* **let, const** 는 block-scoped 이다.
* **var** 는 hoisted 이다.
* **let, const** 는 non-hoisted 이다.

```js
// i is hoisted.
for (var i = 0; i < 10; i++) {
  console.log('i', i)
}
console.log('after loop i is ', i) // after loop i is 10

// i is hoisted in a function. So error happens.
function counter () {
  for(var i = 0; i < 10; i++) {
    console.log('i', i)
  }
}
counter()
console.log('after loop i is', i) // ReferenceError: i is not defined
```

* IIFE 를 사용하면 var 를 선언한 것처럼 hoisting 된다.

```js
(function() {
  // i is hoisted in a function.
  for(var i = 0; i < 10; i++) {
    console.log('i', i)
  }
})()
console.log('after loop i is', i) // ReferenceError: i is not defined
```

```js
// This is ok. because i is hoisted in global.
(function() {
  for(i = 0; i < 10; i++) {
    console.log('i', i)
  }
})()
console.log('after loop i is', i) // after loop i is 10
```

This is same as follow.

```js
var i
(function() {
  for(i = 0; i < 10; i++) {
    console.log('i', i)
  }
})()
console.log('after loop i is', i) // after loop i is 10
```

To prevent hoisting in IIFE, just use strict

```js
// Error will happen.
(function() {
  'use strict'
  for(i = 0; i < 10; i++) {
    console.log('i', i)
  }
})()
console.log('after loop i is', i) // ReferenceError: i is not defined
```

* let, const 는 변수 재선언이 불가능하다. const 는 immutable 하다.

```js
let a = 'hello'
let a = 'world' // 안된다.
a = 'world'; // 잘된다.

const b = 'hello';
const b; // ERROR
b = 'world'; // ERROR
```

## Async, Await

* [async function | mdn](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/async_function)
* [JavaScript Visualized: Promises & Async/Await](https://dev.to/lydiahallie/javascript-visualized-promises-async-await-5gke)
* [JavaScript async and await](https://zellwk.com/blog/async-await/)
* [JavaScript async and await in loops](https://zellwk.com/blog/async-await-in-loops/)

----

Asynchronous programming 을 구현하는 새로운 방법이다. 콜백지옥을 탈출 할 수
있다. 다음은 `async` function 의 특징이다.

* `async` 로 함수를 정의하면 그 함수는 `Promise` object 를 return 한다.
* `async` 함수안에서 `await` 으로 `Promise` object 가 실행을 마칠 때 까지
  기다린다. 

not `Promise` object 를 return 하면 resolved `Promise` object 를 return 한다.

```js
// This will return resolved Promise object.
async function f() {
  return 1;
}
f().then(alert); // 1
// This is same with return 1;
async function f() {
  return Promise.resolve(1);
}
f().then(alert); // 1

// sleep
const sleep = ms => {
  return new Promise(resolve => setTimeout(resolve, ms))
}

// It works well, sleep, async, await in a loop
(async _ => {
  for (i = 0; i < 5; ++i) {
    await sleep(1000);
    console.log(i)
  }
})().then(() => { console.log("done") })

// It works welll, sleep, async, await in a loop
(async _ => {
  for (i = 0; i < 5; ++i) {
    await sleep(1000).then(() => {});
    console.log(i)
  }
})().then(() => { console.log("done") })

// It works well. sleep, async, await in a loop with await return
(async _ => {
  for (i = 0; i < 5; ++i) {
    r = await sleep(1000).then(() => i);
    console.log(r)
  }
})().then(() => { console.log("done") })
```

다음은 nested async function 의 예이다. `await bazAsync()` 는 `basZsync()` 를
모두 마치고 종료한다. 이때 index `i, j, k` 를 구분해서 사용해야 한다. 같은 index
를 사용하면 동작하지 않는다. async, await 를 사용해서 callback hell 을 피했다.

```js
const sleep = ms => {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function bazAsync() {
  for (i = 0; i < 3; ++i) {
    await sleep(1000);
    // do something
    console.log(`\t\t${i}`);
  }
}
//await bazAsync().then(() => console.log("\t\tdone"));

async function barAsync() {
  for (j = 0; j < 3; ++j) {
    await bazAsync();
    // do something
    console.log(`\t${j}`);
  }
}
//await barAsync().then(() => console.log("\tdone"));

async function fooAsync() {
  for (k = 0; k < 3; ++k) {
    await barAsync();
    // do something
    console.log(`${k}`);
  }
}
await fooAsync().then(() => console.log("done"));
```

`await` 는 `thenable` object 앞에 사용할 수 있다. `thenable` object 는
`then()` 를 갖는 object 를 말한다.

```js
class Thenable {
  constructor(num) {
    this.num = num;
  }
  then(resolve, reject) {
    alert(resolve);
    setTimeout(() => resolve(this.num * 2), 1000);
  }
};

async function f() {
  let result = await new Thenable(1);
  // -- 1s later
  alert(result);
  // 2
}
f();
```

## Generator function

* [JavaScript Visualized: Generators and Iterators](https://dev.to/lydiahallie/javascript-visualized-generators-and-iterators-e36)
* [function*](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Statements/function*)

----

generator function 은 `Generator` object 를 return 하는 함수이다. `function*` 을
이용하여 정의할 수 있다. generator function 으로 asyncronous programming 을 할 수 있다.

`Generator` object 의 `next()` 를 호출하면 generator function 의 처음 혹은
이전에 멈춘 `yield` 부터 다음 `yield` 혹은 함수의 끝까지 실행된다. `next()` 는
`iterator protocol` 의 함수이다.

`next()` 는 `value, done` key 가 포함된 object 를 return 한다. ex) `{value: undefined, done: true}`

```js
// Define using function*
function* foo(i) {
  yield i;
  yield i + 10;
}
const gen = foo(10);     // Return Generator object
console.log(gen.next().value); // 10
console.log(gen.next().value); // 20
```

`yield*` 를 사용하여 다른 generator function 을 호출할 수 있다.

```js
function* anotherGenerator(i) {
  yield i + 1;
  yield i + 2;
  yield i + 3;
}

function* generator(i){
  yield i;
  yield* anotherGenerator(i);
  yield i + 10;
}

var gen = generator(10);

console.log(gen.next().value); // 10
console.log(gen.next().value); // 11
console.log(gen.next().value); // 12
console.log(gen.next().value); // 13
console.log(gen.next().value); // 20
```

`next()` 에 argument 를 넘길 수도 있다.

```js
function* logGenerator() {
  console.log(yield);  // yield return 'pretzel'
  console.log(yield);  // yield return 'california'
  console.log(yield);  // yield return 'mayonnaise'
}

var gen = logGenerator();

// the first call of next executes from the start of the function
// until the first yield statement
gen.next();
gen.next('pretzel'); // pretzel
gen.next('california'); // california
gen.next('mayonnaise'); // mayonnaise
```

## Async Generator Function

`async function*` 를 이용하여 정의한다. `AsyncGenerator` object 를 리턴한다.
`yield await` 로 `Promise` object 가 실행을 끝낼 때까지 기다리고 `resolve()` 로
전달한 argument 를 yield 한다.

`foo()` 가 리턴한 `AsyncGenerator` object 는 `await` 로 기다려야 한다.

```js
async function* foo() {
  yield await Promise.resolve('a');
  yield await Promise.resolve('b');
  yield await Promise.resolve('c');
}

let str = '';

async function generate() {
  for await (const val of foo()) {
    str = str + val;
  }
  console.log(str);
}

generate();  // "abc"
```

## Async vs Generator Function

`async function` 와 `generator function` 을 이용하여 asynchronous programming 을
구현할 수 있다. 

`generator function` 는 실행 도중 잠깐 멈추고 다음에 멈춘 부분부터 다시 실행할
수 있다. 그러나 `async function` 은 실행도중 멈출 수 없다.

## Iteration Protocols

> [Iteration protocols | mdn](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Iteration_protocols)

protocol 은 interface 를 말하는 건가? Iteration protocol 은 다음과 같이
두가지가 있다.

* iterable protocol
  * iterable protocol object 를 `for...of` 에 사용하여 반복할 수 있다.
* iterator protocol
  * iterator protocol object 의 `next()` 함수를 호출하여 반복할 수 있다.
  * `value, done` key 를 포함한 object 를 리턴한다. ex) `{value: 3, done: true}`

```js
// Custom iterable, difficult to understand
const myIterable = {
  *[Symbol.iterator]() {
    yield 1;
    yield 2;
    yield 3;
  },
};

console.log([...myIterable]); // [1, 2, 3]
```

## import from

* [When should I use curly braces for ES6 import? | stackoverflow](https://stackoverflow.com/questions/36795819/when-should-i-use-curly-braces-for-es6-import)
* [ES6 export default with multiple functions referring to each other | stackoverflow ](https://stackoverflow.com/questions/33178843/es6-export-default-with-multiple-functions-referring-to-each-other)

----

**default export**

```js
// B.js, This is default import.
import A from './A'
// A.js, A.js should has default export.
export default 42

// B.js, This doen't matter what name you assign to.
import A from './A'
import MyA from './A'
import Something from './A'
```

**single named export**

```js
// B.js, This is named import called A.
import { A } from './A'
// A.js, This should has named export called A.
export const A = 42

// B.js, You should use specific names when import from A.
import { A } from './A'
import { myA } from './A' // Doesn't work!
import { Something } from './A' // Doesn't work!
// A.js, If you use those names, you have to export corresponding names.
export const A = 42
export const myA = 43
export const Something = 44
```

**multiple named import**

```js
// B.js, A module can only have one default export and
// as many named exports as you'd like. You can import them all together:
import A, { myA, Something } from './A'
// A.js
export default 42
export const myA = 43
export const Something = 44

// B.js, We can assign them all different names.
import X, { myA as myX, Something as XSomething } from './A'
```

**multiple named function import**

```js
// util.js
export function foo() { console.log('foo') }
export function bar() { console.log('bar') }
export function baz() { foo(); bar() }

export default {foo, bar, baz}

// A.js, using default export
import util from './util'
util.foo()

// B.js, using named exports
import {bar} from './util'
bar()
```

## Shorthand property names

* [Object initializer @ MDN](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Object_initializer)

```js
let a = 'foo', 
    b = 42,
    c = {};

let o = { 
  a: a,
  b: b,
  c: c
}
```

ECMAScript 2015 supports shorter notation like this.

```js
let a = 'foo', 
    b = 42, 
    c = {};

// Shorthand property names (ES2015)
let o = {a, b, c}

// In other words,
console.log((o.a === {a}.a)) // true
```

## Duplicate Property Names

* [Object initializer @ MDN](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Object_initializer)

When using the same name for your properties, the second property will overwrite the first.

```js
let a = {x: 1, x: 2}
console.log(a) // {x: 2}
```

## Decorator

* [The @ symbol is in fact a JavaScript expression currently proposed to signify decorators:](https://stackoverflow.com/questions/32646920/whats-the-at-symbol-in-the-redux-connect-decorator)

It's only used with Babel.

Without decorators.

```js
import React from 'react';
import * as actionCreators from './actionCreators';
import { bindActionCreators } from 'redux';
import { connect } from 'react-redux';

function mapStateToProps(state) {
  return { todos: state.todos };
}

function mapDispatchToProps(dispatch) {
  return { actions: bindActionCreators(actionCreators, dispatch) };
}

class MyApp extends React.Component {
  // ...define your main app here
}

export default connect(mapStateToProps, mapDispatchToProps)(MyApp);
```

With decorators.

```js
import React from 'react';
import * as actionCreators from './actionCreators';
import { bindActionCreators } from 'redux';
import { connect } from 'react-redux';

function mapStateToProps(state) {
  return { todos: state.todos };
}

function mapDispatchToProps(dispatch) {
  return { actions: bindActionCreators(actionCreators, dispatch) };
}

@connect(mapStateToProps, mapDispatchToProps)
export default class MyApp extends React.Component {
  // ...define your main app here
}
```

# Style Guide

* [JavaScript Standard Style](js_standard_style_guide.md)
* [JavaScript Google Style Guide](js_google_style_guide.md)

# Refactoring

[Refactoring JavaScript](refactoring_js.md)

# Effective JavaScript

[Effective JavaScript](effective_js.md)

# Design Pattern

[JavaScript GOF Design Pattern](js_gof_design_pattern.md)

# Architecutre

JavaScript 는 주로 [React.js](/react.js/README.md) 와 함께 사용한다. [React.js
Architecture](/React.js/README.md#architectures) 를 참고하자.
