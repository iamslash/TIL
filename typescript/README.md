# TypeScript

**Korean version**: See [README-kr.md](README-kr.md) for Korean language documentation.

- [Resources](#resources)
- [Basics](#basics)
  - [Build & Run](#build--run)
  - [Print](#print)
  - [Reserved Words](#reserved-words)
  - [Min, Max](#min-max)
  - [abs vs fabs](#abs-vs-fabs)
  - [Bitwise Operations](#bitwise-operations)
  - [String](#string)
  - [Random](#random)
  - [Formatted String](#formatted-string)
  - [Inspect](#inspect)
  - [Data Types](#data-types)
  - [Control Flow](#control-flow)
  - [Collections](#collections)
  - [Collection Conversion](#collection-conversion)
  - [Sort](#sort)
  - [Search](#search)
  - [Multidimensional Array](#multidimensional-array)
  - [Enum](#enum)
  - [Multiple Variables](#multiple-variables)
  - [Spread/Rest Operator](#spreadrest-operator)
  - [Nullish Coalescing (||) and Double Question Mark (??)](#nullish-coalescing--and-double-question-mark-)
  - [Export & Import](#export--import)
- [Core Concepts](#core-concepts)
  - [undefined vs unknown vs any vs never Comparison](#undefined-vs-unknown-vs-any-vs-never-comparison)
  - [Generics](#generics)
  - [Utility Types](#utility-types)
  - [Interface vs Type](#interface-vs-type)
  - [Optional (Optional Parameters and Properties)](#optional-optional-parameters-and-properties)
  - [`declare`](#declare)
  - [Function Types with Interface](#function-types-with-interface)
- [Advanced](#advanced)
  - [Map vs Record](#map-vs-record)
- [Style Guide](#style-guide)
- [Refactoring](#refactoring)
- [Effective TypeScript](#effective-typescript)
- [Design Patterns](#design-patterns)
- [Architecture](#architecture)

----

# Resources

* [TypeScript at a Glance (updated)](https://heropy.blog/2020/01/27/typescript/)
* [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html)
  * Essential official documentation
  * [TypeScript Handbook Korean](https://typescript-kr.github.io/)
* [Chapter 8. Using TypeScript in React Projects](https://react.vlpt.us/using-typescript/)
* [Using TypeScript and Redux Like a Pro @ velog](https://velog.io/@velopert/use-typescript-and-redux-like-a-pro)
* [playcode](https://playcode.io/)
  * TypeScript playground
* [Introduction to TypeScript and Setting Up the Development Environment](https://poiemaweb.com/typescript-introduction)

# Basics

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

## Print

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

## Min, Max

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
// Expected output: 2
console.log(difference(5, 3));
// Expected output: 2
console.log(difference(1.23456, 7.89012));
// Expected output: 6.6555599999999995
```

## Bitwise Operations

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

// String iteration
let s = "hello world"
let n = s.length
for (let i = 0; i < n; ++i) {
    let c = s.charAt(i);
    console.log(c);
}

// Convert character to number and number to character
// https://stackoverflow.com/questions/22624379/how-to-convert-letters-to-numbers-with-javascript
let s = "abcdefg";
let c = s.charAt(0);   // c is a string, "a"
// 97 represents "a"
let i = c.charCodeAt(0) - 97;  // 0
let d = String.fromCharCode(97 + i) // "a"
```

## Random

* [Math.random() | mdn](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Math/random)

Generates a value in the range `[0..1]`.

```ts
console.log(Math.random());
console.log(Math.random());
```

## Formatted String

* [util.format | node.js](https://nodejs.org/api/util.html#utilformatformat-args)

Remember that the basic types in TypeScript are `boolean, number, string`.

```ts
import util from "util";

console.log(util.format('%s %d %s', true, 4, 'Hello World'));

// Print without newline
process.stdout.write(`i: ${i}, diff: ${diff} has: ${numMap.has(diff)} `);
console.log(numMap);
```

## Inspect

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
// Array of values
let list1: number[] = [1, 2, 3];
// Generic array
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

// Type Assertion
let someValue: unknown = "This is a string";
let strLength: number = (someValue as string).length;
let strLength2: number = (<string>someValue).length;
```

## Control Flow

### Conditionals

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

### Loops

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

This is one of the most commonly confused topics.

```ts
let arr = [10, 20, 30];

// for...of -> extracts "values"
for (let val of arr) {
    console.log(val);   // 10, 20, 30
}

// for...in -> extracts "indices (keys)"
for (let idx in arr) {
    console.log(idx);   // "0", "1", "2"  (strings!)
}

// for...of also works with strings
for (let chr of "Hello") {
    console.log(chr);   // H, e, l, l, o
}
```

| | `for...of` | `for...in` |
|---|---|---|
| Extracts | **Values** | **Keys (indices)** |
| With arrays | `10, 20, 30` | `"0", "1", "2"` |
| Type | Original type | **Always string** |

> For array iteration, **use `for...of`.** Use `for...in` when iterating over object keys.

### var vs let Scope

```ts
// var: survives outside the loop (function scope)
for (var i in [1, 2, 3]) {}
console.log(i);   // "2" -- accessible!

// let: exists only inside the loop (block scope)
for (let j in [1, 2, 3]) {}
console.log(j);   // Error -- not accessible
```

> **Always use `let`.** `var` can cause bugs by keeping variables alive unintentionally.

## Collections

### Tuple

Looks like an array, but **each position has a fixed type**.

```ts
let employee: [number, string] = [1, 'David'];
employee[0] = "hello";  // Error -- index 0 must be number
employee[1] = 42;        // Error -- index 1 must be string

let person: [number, string, boolean] = [1, 'David', true];
let employees: [number, string][] = [[1, 'David'], [2, 'Tom']];

// push/pop also work
employee.push(2, 'John');
console.log(employee);  // [1, 'David', 2, 'John']
```

**When to use?** Useful when returning multiple values from a function:

```ts
function getUser(): [number, string] {
    return [1, "David"];
}
const [id, name] = getUser();  // Clean destructuring
```

> For 3 or more elements, **interfaces/objects** are more readable than tuples.

**Tuple vs Array -- Same values, different type declarations**

The values themselves look exactly the same. At runtime (JavaScript), **both are just arrays**. The difference only exists when the TypeScript compiler checks types.

```ts
// Array -- "any number" of the same type
let arr: number[]         = [1, 2];       // Any number of numbers OK
arr.push(3);       // OK
arr = [1];         // OK

// Tuple -- fixed type and "count" for each position
let tup: [number, string] = [1, "David"]; // Exactly number, string in order
tup = [1];         // Error -- string is missing
tup = [1, 2];      // Error -- second element must be string

// At runtime, they are completely identical
console.log(Array.isArray(tup));  // true -- tuple is also an array!
```

| | Type Declaration | Value Appearance | Runtime |
|---|---|---|---|
| Array | `number[]` | `[1, 2, 3]` | Array |
| Tuple | `[number, string]` | `[1, "David"]` | Array (same!) |

> Think of a tuple as "an array where the compiler enforces types per position."

### Array

```ts
// Declaration
let fruits: Array<string> = ['Apple', 'Orange', 'Banana'];
let numbers: number[] = [1, 2, 3, 4];
let vals: (string | number)[] = ['Apple', 2, 'Orange', 3];

// Initialization
let filled = new Array<number>(5).fill(-1);  // [-1, -1, -1, -1, -1]
```

Array methods are easier to remember when organized by purpose:

| Purpose | Method | Mutates Original? |
|---------|--------|-------------------|
| **Transform** | `map`, `flatMap` | No (new array) |
| **Filter** | `filter` | No (new array) |
| **Reduce** | `reduce` | No (single value) |
| **Search** | `find`, `findIndex`, `indexOf`, `includes` | No |
| **Validate** | `every`, `some` | No |
| **Sort** | `sort`, `reverse` | **Yes (mutates!)** |
| **Add/Remove (end)** | `push`, `pop` | Yes |
| **Add/Remove (front)** | `unshift`, `shift` | Yes |
| **Cut** | `splice` | Yes |
| **Copy & Cut** | `slice` | No (new array) |

```ts
const numbers = [1, 2, 3, 4];

// Transform: map -- transforms each element and returns a new array
numbers.map(n => n * 2);                    // [2, 4, 6, 8]

// Filter: filter -- collects only elements matching the condition into a new array
numbers.filter(n => n % 2 === 0);           // [2, 4]

// Reduce: reduce -- reduces the array to a single value
numbers.reduce((acc, cur) => acc + cur, 0); // 10

// Search: find, findIndex, includes
numbers.find(n => n > 2);                   // 3
numbers.findIndex(n => n > 2);              // 2
numbers.includes(3);                        // true

// Validate: every, some
numbers.every(n => n % 2 === 0);            // false (all even?)
numbers.some(n => n % 2 !== 0);             // true  (any odd?)

// flat, flatMap -- flatten multidimensional arrays
[1, [2, 3], [4, [5]]].flat(2);             // [1, 2, 3, 4, 5]
["hello", "world"].flatMap(s => s.split('')); // ['h','e','l','l','o','w','o','r','l','d']
```

**Commonly confused methods:**

```ts
// splice vs slice
const arr = [1, 2, 3, 4, 5];
arr.slice(1, 3);     // [2, 3]        -- original unchanged, returns copy
arr.splice(1, 2);    // [2, 3] removed -- original becomes [1, 4, 5]!

// sort caution! Default is "lexicographic"
[1, 10, 2].sort();                // [1, 10, 2] -- string comparison!
[1, 10, 2].sort((a, b) => a - b); // [1, 2, 10] -- numeric comparison

// push/pop (end) vs unshift/shift (front)
const stack = [1, 2];
stack.push(3);    // [1, 2, 3]  -- add to end
stack.pop();      // [1, 2]     -- remove from end
stack.unshift(0); // [0, 1, 2]  -- add to front
stack.shift();    // [1, 2]     -- remove from front

// reverse -- mutates original! Use spread to copy
const reversed = [...stack].reverse();  // stack remains unchanged
```

### Set

A collection of **unique** values. `has()` lookup is `O(1)`, faster than array's `includes()`.

```ts
let dirs = new Set<string>(['east', 'west']);
dirs.add('north');
dirs.add('east');          // Duplicate ignored
console.log(dirs.size);    // 3 (east, west, north)
console.log(dirs.has('east'));   // true
dirs.delete('east');       // true
dirs.clear();              // Delete all

// Iteration
for (let dir of dirs) {
    console.log(dir);
}
```

**Practical pattern: Remove duplicates from array**

```ts
const arr = [1, 2, 2, 3, 3, 3];
const unique = [...new Set(arr)];  // [1, 2, 3]
```

> **Array vs Set decision:** Use **Set** when frequently checking "does this value exist?", use **Array** when order/index matters.

### Map

Stores key-value pairs. Similar to plain objects `{}` but with differences.

```ts
let map = new Map<string, number>();
map.set('David', 10);
map.set('John', 20);

console.log(map.get('David'));      // 10
console.log(map.get('Tom'));        // undefined
console.log(map.get('Tom') || 0);   // 0 (default value pattern)
console.log(map.has('David'));      // true
console.log(map.size);              // 2
map.delete('John');

// Create with initialization
let config = new Map<string, string>([
    ['host', 'localhost'],
    ['port', '3000']
]);

// Iteration
for (let [key, val] of map) {
    console.log(key, val);
}
// for...of with keys(), values(), entries() is also available
```

**Map vs Plain Object `{}`:**

| | `Map` | `{}` (Object) |
|---|---|---|
| Key type | **Any type** (objects, functions OK) | string / symbol only |
| Order guaranteed | Insertion order guaranteed | Partially guaranteed in ES2015+ |
| Size check | `map.size` | `Object.keys(obj).length` |
| Performance | **Faster** with frequent add/delete | Faster with fixed structure |

> **Decision:** Use **Map** when keys change dynamically, use **object/interface** when the structure is fixed.

## Collection Conversion

```ts
// Tuple to Set
let arr = [11, 22, 33];
let set = new Set(arr);
console.log(set);  // Set(3) { 11, 22, 33 }
```

## Sort

```ts
let arr: number[] = [1, 10, 2, 5, 3];

// Warning: sort() without a comparator sorts "lexicographically"!
arr.sort();
console.log(arr);  // [1, 10, 2, 3, 5] -- 10 comes before 2!

// Ascending -- always provide a comparator for number arrays
arr.sort((a, b) => a - b);
console.log(arr);  // [1, 2, 3, 5, 10]

// Descending
arr.sort((a, b) => b - a);
console.log(arr);  // [10, 5, 3, 2, 1]

// String arrays are safe with default sort()
let fruits = ['Banana', 'Apple', 'Cherry'];
fruits.sort();
console.log(fruits);  // ['Apple', 'Banana', 'Cherry']
```

> **`sort()` mutates the original array.** To preserve the original, use `[...arr].sort()`.

## Search

TypeScript has **no built-in binary search.** Only linear search is provided.

```ts
let arr = [1, 2, 3, 4, 5];

arr.find(a => a > 3);       // 4         -- first "value" matching the condition
arr.findIndex(a => a > 3);  // 3         -- first "index" matching the condition
arr.indexOf(2);             // 1         -- "index" of the exact value
arr.includes(3);            // true      -- existence check only
```

| Method | Return Value | Use Case |
|--------|-------------|----------|
| `find(fn)` | Value or `undefined` | Search by condition |
| `findIndex(fn)` | Index or `-1` | Find position by condition |
| `indexOf(val)` | Index or `-1` | Find position by exact value |
| `includes(val)` | `boolean` | Check existence only |

> If you need binary search, implement it yourself or use a library.

## Multidimensional Array

```ts
// Declaration and iteration
let grid: number[][] = [
    [1, 2, 3],
    [4, 5, 6]
];

for (let i = 0; i < grid.length; i++) {
    for (let j = 0; j < grid[0].length; j++) {
        console.log(grid[i][j]);
    }
}
// Output: 1, 2, 3, 4, 5, 6

// Initialization: fill a 3x4 array with 0
let matrix: number[][] = Array.from({length: 3}, () => new Array(4).fill(0));
```

> **Pitfall:** `new Array(3).fill(new Array(4).fill(0))` makes all rows **share the same reference**. Changing `matrix[0][0] = 1` will change index 0 of every row. Always use `Array.from()`.

## Enum

* [Enum | typescript](https://www.typescriptlang.org/docs/handbook/enums.html#handbook-content)

Used to group related constants together. There are 3 kinds.

```ts
// 1. Numeric enum -- auto-incrementing
enum Direction {
  Up = 1,    // 1
  Down,      // 2 (auto)
  Left,      // 3
  Right,     // 4
}

// 2. String enum -- values must be explicit
enum Status {
  Active = "ACTIVE",
  Inactive = "INACTIVE",
  Pending = "PENDING",
}

// 3. Heterogeneous enum -- mixed number+string (not recommended)
enum BooleanLikeHeterogeneousEnum {
  No = 0,
  Yes = "YES",
}

let dir: Direction = Direction.Up;
```

**When to use?** When replacing magic strings with type-safe values:

```ts
// before -- no error even with typos
if (user.status === "ACTVE") { ... }  // Typo! Bug at runtime

// after -- compile error on typo with enum
if (user.status === Status.Active) { ... }  // Safe
```

> Recently, **union types** are preferred over enums:
> ```ts
> type Direction = "UP" | "DOWN" | "LEFT" | "RIGHT";  // More concise
> ```

## Multiple Variables

```ts
let i = 0, j = 0, n = s.length
```

## Spread/Rest Operator

> * [JavaScript | What is ... (spread/rest) in JS?](https://dinn.github.io/javascript/js-dotdotdot/)

`...` serves two roles: **Spread** and **Rest**. Distinguish them by **position**:

| Position | Role | Name |
|----------|------|------|
| **Receiving side** (parameter, destructuring) | **Gathers** the rest into an array/object | Rest |
| **Sending side** (function call, literal) | **Spreads** array/object into individual values | Spread |

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
let {a, ...c, y} = foo;  // Error: Uncaught SyntaxError: Rest element must be last element

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

## Nullish Coalescing (||) and Double Question Mark (??)

> * [Comparison of Null Handling Operators (How To Use Double Question Marks: ??)](https://ksrae.github.io/angular/double-question-marks/)

Both are used for "providing default values", but they differ in **what they consider falsy**.

| | `\|\|` | `??` |
|---|---|---|
| Filtered values | `undefined, null, false, 0, ""` | `undefined, null` only |
| Risk | `0` or `""` disappear when they are valid values | None |

> **Use `??`.** `||` can cause bugs when `0` or empty string are valid values.

```ts
// if statement for undefined or null check
if (val !== undefined || val != null) {
    console.log("ok");
}
// Ternary operator for undefined or null check
val = val !== undefined || val !== null ? val : '';
// Logical OR for undefined or null check
val = val || '';
// Logical OR can check falsy values
console.log(undefined || "falsy");  // falsy
console.log(null || "falsy");       // falsy
console.log(false || "falsy");      // falsy
console.log(0 || "falsy");          // falsy
console.log('' || "falsy");         // falsy

// Double question mark since TypeScript 3.7
console.log(undefined ?? "falsy");  // falsy
console.log(null ?? "falsy");       // falsy
console.log(false ?? "falsy");      // false
console.log(0 ?? "falsy");          // 0
console.log('' ?? "falsy");         //
```

## Export & Import

* [TypeScript at a Glance (updated) - Export and Import](https://heropy.blog/2020/01/27/typescript/)

This is the module system. It shares types/functions/variables between files.

| | Named Export | Default Export |
|---|---|---|
| Syntax | `export { A, B }` | `export default A` |
| Import | `import { A, B }` (name must match) | `import Anything` (any name) |
| Per file | Multiple | One only |

```ts
// foo.ts -- Named Export (export multiple)
export interface UserType {
    name: string,
    mobile: number
}
export type UserIDType = string | number;

// bar.ts -- Named Import (name must match)
import { UserType, UserIDType } from './foo';
const user: UserType = {
    name: 'David',
    mobile: 333
}
const userid: UserIDType = "111";
```

```ts
// Button.ts -- Default Export (one representative export per file)
export default function Button() { ... }

// App.ts -- Default Import (any name, no curly braces)
import Button from './Button';
import MyButton from './Button';  // Different name also OK
```

TypeScript supports `export = bar;` and `export bar = require('bar');` for `CommonJS/AMD/UMD` modules. This is equivalent to ES6's `export default`.

```ts
// Import from CommonJS/AMD/UMD module bar
import bar = require('bar');
// or
import * as bar from 'bar';
// or with "esModuleInterop": true
import bar from 'bar';
```

# Core Concepts

These are type system concepts unique to TypeScript. They do not exist in JavaScript, and you must understand them to use TypeScript effectively.

## undefined vs unknown vs any vs never Comparison

These four special types have clearly different roles.

| Type | One-line Summary | Key Point |
|------|-----------------|-----------|
| `undefined` | Value **not yet assigned** | JavaScript default for empty state |
| `unknown` | Value is **unknown** (check before using) | Safe version of `any` |
| `any` | Value can be **anything** (skip checking) | No type safety, not recommended |
| `never` | Value **cannot exist** | Function never returns normally |

### undefined -- "Not assigned yet"

```ts
let name: string;
console.log(name);          // undefined -- no value assigned

function greet(name?: string) {
    console.log(name);      // undefined if not passed
}
greet();                    // undefined

const arr = [1, 2, 3];
console.log(arr[10]);       // undefined -- out of bounds
```

### unknown -- "Unknown, so check before using"

Like `any`, it can hold any value, but **cannot be used without type checking**.
Suitable for external API responses, `JSON.parse`, and `catch` error handling.

```ts
let value: unknown = "hello";

value.toUpperCase();            // Compile error -- cannot use directly
(value as string).toUpperCase(); // OK after type assertion

if (typeof value === "string") {
    value.toUpperCase();        // OK after typeof check (auto-narrowed)
}

// Practical: error handling in catch
try {
    something();
} catch (err: unknown) {
    // err.message;              // Cannot use directly
    if (err instanceof Error) {
        console.log(err.message); // OK after check
    }
}
```

### any vs unknown

```ts
// any: anything goes without error (dangerous!)
let a: any = "hello";
a.foo.bar.baz;          // Compiles fine -- crashes at runtime

// unknown: nothing allowed without checking (safe!)
let b: unknown = "hello";
b.foo.bar.baz;          // Compile error -- caught before runtime
```

> **If you want to use `any`, use `unknown` instead.** `any` completely disables type checking.

### never -- "This situation can never occur"

Used when a function never returns normally, or to verify all cases are handled.

```ts
// 1. Function that always throws
function fail(msg: string): never {
    throw new Error(msg);
}

// 2. Exhaustive check -- prevent missing cases at compile time
type Shape = "circle" | "square" | "triangle";

function getArea(shape: Shape): number {
    switch (shape) {
        case "circle":   return 3.14 * 10 * 10;
        case "square":   return 10 * 10;
        case "triangle": return (10 * 5) / 2;
        default:
            const _exhaustive: never = shape;  // Unreachable if all cases handled
            throw new Error(`Unknown shape: ${_exhaustive}`);
    }
}
// If you later add "pentagon" to Shape without adding a case, you get a compile error!
```

## Generics

* [Generics](ts_handbook.md#generics)

**Passing types as parameters.** The key idea is "accept any type, but maintain consistency."

```ts
// Without generics -- using any loses type information
function identity(arg: any): any {
    return arg;
}
let result = identity("hello");  // result type: any (not string!)

// With generics -- type information preserved
function identity<T>(arg: T): T {
    return arg;
}
let result = identity("hello");  // result type: string
let num = identity(42);          // num type: number
```

**Practical patterns:**

```ts
// Generic function
function firstElement<T>(arr: T[]): T | undefined {
    return arr[0];
}
firstElement([1, 2, 3]);      // number
firstElement(["a", "b"]);     // string

// Generic class
class Box<T> {
    content: T;
    constructor(value: T) { this.content = value; }
}
let numBox = new Box(42);       // Box<number>
let strBox = new Box("hello");  // Box<string>
```

> `<T>` in generics is a **placeholder** for "a type to be specified later." Unlike `any`, it maintains type safety.

## Utility Types

> * [Utility Types | typescript](https://www.typescriptlang.org/ko/docs/handbook/utility-types.html)
> * [Typescript Utility Types - Parameters, ReturnType, Required](https://www.morolog.dev/entry/Typscript-%EC%9C%A0%ED%8B%B8%EB%A6%AC%ED%8B%B0-%ED%83%80%EC%9E%85-Parameters-ReturnType-Required)

Built-in tools that **transform** existing types to create new ones. The most commonly used:

| Utility | What it does | Example |
|---------|-------------|---------|
| `ReturnType<T>` | Extract function **return** type | `ReturnType<typeof getUser>` |
| `Parameters<T>` | Extract function **parameter** types | `Parameters<typeof login>` |
| `Required<T>` | Remove all `?` -> make required | `Required<Props>` |
| `Partial<T>` | Add `?` to all properties -> make optional | `Partial<User>` |
| `Record<K,V>` | Define key-value type | `Record<string, number>` |
| `Pick<T,K>` | **Pick** specific properties only | `Pick<User, 'name'>` |
| `Omit<T,K>` | **Exclude** specific properties | `Omit<User, 'password'>` |
| `keyof T` | Keys as union type | `keyof Point` -> `"x" \| "y"` |

```ts
// ReturnType<T>
// Creates the return type of a function.
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
// Creates the parameter types of a function.
declare function foo(foo: {name: string, mobile: number}): void
type fooParams = Parameters<typeof foo>;

// Required<T>
// Creates a type with all fields required.
interface Props {
  a?: number;
  b?: string;
}

// OK
const obj: Props = { a: 5 };
// Error: Property 'b' is missing in type '{ a: number; }'
// but required in type 'Required<Props>'.
const obj2: Required<Props> = { a: 5 };

// Record<Keys, Type>
// Creates an object type whose property keys are Keys and property values are Type.
// https://developer-talk.tistory.com/296

// Define an object type called personType.
// Using index signature.
type personType = {
    [name: string]: number
}
let person: personType = {
    'foo': 10,
    'bar': 20,
    'baz': 30
}
// Let's convert this to a Record type.
// What are the advantages?
type personType = Record<string, number>;
let person: personType = {
    'foo': 10,
    'bar': 20,
    'baz': 30
}
// There are cases where index signatures don't work.
// Error:
// An index signature parameter type cannot be a
// literal type or generic type. Consider using a mapped
// object type instead.
type personType = {
    [name: 'foo' | 'bar' | 'baz']: number
}
// Solve it like this.
type names = 'foo' | 'bar' | 'baz';
type personType = Record<names, number>;
let person: personType = {
    'foo': 10,
    'bar': 20,
    'baz': 30
}

// keyof
// The keyof operator takes an object type and produces
// a string or numeric literal union of its keys.
type Point = { x: number; y: number };
type P = keyof Point;

// keyof with Record type
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

## Interface vs Type

* [Difference between TypeScript type and interface | tistory](https://bny9164.tistory.com/48)

---

`interface` is recommended over `type`. Types are recursively transpiled at runtime. This results in poor performance due to longer compile times.

| | `interface` | `type` |
|---|---|---|
| `extends` (inheritance) | Yes | No (use `&` instead) |
| Declaration merging (same name declared twice) | Yes | No, error |
| Computed keys `[key in ...]` | No | Yes |
| Union type `string \| number` | No | Yes |
| Performance | **Fast** | Slow (recursive transpilation) |

> **Default to `interface`; use `type` when you need unions or computed keys.**

`type` has the following disadvantages compared to `interface`:

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
    // extends does not work with type
    type Point = {
        x: number;
        y: number;
    }
    // // Error: Cannot use extends with type
    // type PointColor extends Point {
    //     c: number;
    // }
}
// Merged declarations
{
    // Merged declarations work with interface
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
    // // Error: Merged declarations do not work with type
    // type PointColor = {
    //     x: number;
    //     y: number;
    // }
    // type PointColor = {
    //     c: number;
    // }
}
// Computed values
{
    // Computed values do not work with interface
    type coords = 'x' | 'y';
    interface CoordTypes {
        [key in coords]: string
    }
}
{
    // Computed values work with type
    type coords = 'x' | 'y';
    type CoordTypes = {
        [CoordTypes in coords]: string;
    }
    const point: CoordTypes = { x: '3', y: '3' };
    console.log(point);
}
// type can be resolved to never type
// Be careful
{
    type goodType = { a: 1 } & { b: 2 } // Good
    type neverType = { a: 1; b: 2 } & { b: 3 } // Resolved to `never`

    const foo: goodType = { a: 1, b: 2 } // Good
    // Error: Type 'number' is not assignable to type 'never'.(2322)
    const bar: neverType = { a: 1, b: 3 }
    // Error: Type 'number' is not assignable to type 'never'.(2322)
    const baz: neverType = { a: 1, b: 2 }
}
{
    type t1 = {
        a: number
    }
    type t2 = t1 & {
        b: string
    }

    const foo: t2 = { a: 1, b: 2 } // Error
}
```

## Optional (Optional Parameters and Properties)

In TypeScript, adding `?` after a name means **"it can be present or absent."** There are 3 main usages.

### Optional Parameter

Adding `?` to a function parameter allows it to be omitted when calling.

```ts
// name is required, greeting is optional
function greet(name: string, greeting?: string): string {
  return `${greeting ?? "Hello"}, ${name}!`;
}

console.log(greet("David"));            // "Hello, David!"
console.log(greet("David", "Hi"));      // "Hi, David!"
```

Also commonly used with the options object pattern.

```ts
// The second parameter itself is optional
async function fetchData(url: string, options?: { timeout?: number; retries?: number }) {
  const timeout = options?.timeout ?? 3000;
  const retries = options?.retries ?? 1;
  console.log(`url=${url}, timeout=${timeout}, retries=${retries}`);
}

await fetchData("/api/users");                          // options omitted OK
await fetchData("/api/users", { timeout: 5000 });       // retries omitted OK
await fetchData("/api/users", { timeout: 5000, retries: 3 }); // Both provided OK
```

### Optional Property

Adding `?` to an interface or type property makes that property optional.

```ts
interface User {
  name: string;       // Required
  age?: number;       // Optional
  email?: string;     // Optional
}

const user1: User = { name: "David" };                     // OK
const user2: User = { name: "David", age: 30 };            // OK
const user3: User = { name: "David", age: 30, email: "a@b.com" }; // OK
// const user4: User = { age: 30 };                        // ERROR: name is missing
```

### Optional Chaining

Using `?.` when accessing object properties safely returns `undefined` instead of throwing an error when the value is `null` or `undefined`.

```ts
interface Company {
  name: string;
  address?: {
    city?: string;
    zipCode?: string;
  };
}

const company: Company = { name: "Foo Inc." };

// Accessing without ?. causes a runtime error
// console.log(company.address.city);   // ERROR: Cannot read property 'city' of undefined

// Using ?. for safe access
console.log(company.address?.city);     // undefined (no error)
console.log(company.address?.zipCode);  // undefined (no error)
```

### Summary

| Usage | Syntax | Meaning |
|-------|--------|---------|
| Optional Parameter | `function foo(x?: string)` | Parameter can be omitted |
| Optional Property | `{ name?: string }` | Property can be absent |
| Optional Chaining | `obj?.prop` | Returns undefined instead of error for null/undefined |

## `declare`

* [Purpose of declare keyword in TypeScript | stackoverflow](https://stackoverflow.com/questions/43335962/purpose-of-declare-keyword-in-typescript)
  * [Korean](https://jjnooys.medium.com/typescript-declare-cd163acb9f)

It means "this type/variable **already exists elsewhere**, so trust me, compiler." It is not converted to JavaScript.

```ts
// jQuery is already loaded via <script>
declare var $: any;
$(".btn").click();  // No compile error

// Without declare vs with declare
        type Callback = (err: Error | String, data: Array<CalledBackData>) => void;
declare type Callback = (err: Error | String, data: Array<CalledBackData>) => void;
```

> `.d.ts` files are entirely collections of `declare` statements. They provide only type information without actual code.

## Function Types with Interface

* [TypeScript Interface](https://www.softwaretestinghelp.com/typescript-interface/)

A pattern that allows **adding properties to functions**. React's `FunctionComponent` is a representative example. Use a question mark before the property name for optional properties.

> Use this when you need an object that is both a function and has properties. For regular function types, arrow syntax (`(x: string) => void`) is sufficient.

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

# Advanced

## Map vs Record

* [map vs object | TIL](/js/README.md#map-vs-object)

```ts
// Record -- object type with "fixed" keys (compile time)
type UserScores = Record<string, number>;
let scores: UserScores = { David: 100, John: 85 };

// Map -- collection with "dynamic" keys (runtime object)
let scoreMap = new Map<string, number>();
scoreMap.set("David", 100);
```

| | `Record<K, V>` | `Map<K, V>` |
|---|---|---|
| Nature | **Type** (compile time) | **Class** (runtime object) |
| Key type | string / number / symbol | **Any type** |
| Use case | Define object shape | Dynamic key-value storage |
| Iteration | `Object.keys()` | `for...of`, `forEach` |
| Size | `Object.keys(obj).length` | `map.size` |

> **Decision:** Use `Record` when the structure is predetermined, use `Map` when keys are added/removed at runtime.

# Style Guide

[TypeScript Style Guide](ts_google_style_guide.md)

# Refactoring

[Refactoring TypeScript](refactoring_ts.md)

# Effective TypeScript

[Effective TypeScript](effective_ts.md)

# Design Patterns

[TypeScript Design Pattern](ts_gof_design_pattern.md)

# Architecture

* [Typescript Clean Architecture | github](https://github.com/pvarentsov/typescript-clean-architecture)
  * Slightly different from Java's Clean Architecture
* [A TypeScript Stab at Clean Architecture](https://www.freecodecamp.org/news/a-typescript-stab-at-clean-architecture-b51fbb16a304/)
* [Evolution of a React folder structure and why to group by features right away](https://profy.dev/article/react-folder-structure)
* [React Folder Structure in 5 Steps [2022]](https://www.robinwieruch.de/react-folder-structure/)
  * Step-by-step explanation from simple to complex structures
* [bulletproof-react/docs/project-structure.md](https://github.com/alan2207/bulletproof-react/blob/master/docs/project-structure.md)
* [4 folder structures to organize your React & React Native project](https://reboot.studio/blog/folder-structures-to-organize-react-project/)
* [Project structure | Frontend Handbook](https://infinum.com/handbook/frontend/react/project-structure)
