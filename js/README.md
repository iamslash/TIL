- [Abstract](#abstract)
- [Essentials](#essentials)
- [Materials](#materials)
- [Basic Usages](#basic-usages)
  - [Collections compared c++ container](#collections-compared-c-container)
  - [Collection by Examples](#collection-by-examples)
  - [Multidimensional Array](#multidimensional-array)
  - [Sort](#sort)
  - [Data types](#data-types)
  - [Reserved Words](#reserved-words)
  - [Operators](#operators)
  - [Decision Making](#decision-making)
  - [Loops](#loops)
  - [Javascript Runtime Architecture](#javascript-runtime-architecture)
  - [Functions](#functions)
    - [curly braces](#curly-braces)
    - [parenthese](#parenthese)
    - [anonymous function](#anonymous-function)
    - [named function](#named-function)
    - [closure](#closure)
    - [IIFE(Immediately-invoked function expression)](#iifeimmediately-invoked-function-expression)
    - [arrow function (ES6)](#arrow-function-es6)
  - [hoisting](#hoisting)
  - [scope](#scope)
  - [execution context](#execution-context)
  - [event loop](#event-loop)
  - [background](#background)
  - [task queue](#task-queue)
  - [micro task](#micro-task)
  - [Cookies](#cookies)
  - [Objects](#objects)
  - [Number](#number)
  - [Boolean](#boolean)
  - [Strings](#strings)
  - [Arrays](#arrays)
  - [Date](#date)
  - [Math](#math)
  - [RegExp](#regexp)
  - [HTML DOM](#html-dom)
- [Advanced Usages](#advanced-usages)
  - [var, let, const](#var-let-const)
  - [promise](#promise)
  - [async, await](#async-await)

-------------------------------------------------------------------------------

# Abstract

java script에 대해 정리한다.

# Essentials

* [learn javascript in Y minutes](https://learnxinyminutes.com/docs/javascript/)
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
* [The modern javascript tutorial](https://javascript.info/)
  * 가장 자세한 tutorial
* [Secrets of the JavaScript Ninja - John Resig and Bear Bibeault](https://www.manning.com/books/secrets-of-the-javascript-ninja)
* [함수형 자바스크립트 프로그래밍 - 유인동](http://www.yes24.com/24/Goods/56885507?Acode=101)
  * 비함수형 언어인 자바스크립트를 이용하여 함수형 프로그래밍을 시도한다.
  * [src](https://github.com/indongyoo/functional-javascript)
* [john resig blog](https://johnresig.com/apps/learn/)
* [실행 컨텍스트](https://www.zerocho.com/category/Javascript/post/5741d96d094da4986bc950a0)
* [Underscore.js](http://underscorejs.org/)
  * functional programming helper

# Basic Usages

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


## Collection by Examples

* Array

```js
var fruits = ['사과', '바나나'];
console.log(fruits.length);
// 2
```

* Set

```js
var mySet = new Set();
mySet.add(1);
mySet.add("some text");
mySet.add("foo");

mySet.has(1); // true
mySet.delete("foo");
mySet.size; // 2

for (let item of mySet) console.log(item);
// 1
// "some text"
```

* Map

```js
var sayings = new Map();
sayings.set("dog", "woof");
sayings.set("cat", "meow");
sayings.set("elephant", "toot");
sayings.size; // 3
sayings.get("fox"); // undefined
sayings.has("bird"); // false
sayings.delete("dog");

for (var [key, value] of sayings) {
  console.log(key + " goes " + value);
}
// "cat goes meow"
// "elephant goes toot"
```

## Multidimensional Array

* [How to create multidimensional array](https://stackoverflow.com/questions/7545641/how-to-create-multidimensional-array)

----

```js
Array.matrix = function(numrows, numcols, initial) {
   var arr = [];
   for (var i = 0; i < numrows; ++i) {
      var columns = [];
      for (var j = 0; j < numcols; ++j) {
         columns[j] = initial;
      }
      arr[i] = columns;
   }
   return arr;
}
var nums = Array.matrix(5, 5, 0);
print(nums[1][1]); // displays 0
var names = Array.matrix(3, 3, "");
names[1][2] = "Joe";
print(names[1][2]); // display "Joe"
var grades = [[89, 77, 78],[76, 82, 81],[91, 94, 89]];
print(grades[2][2]); // displays 89
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

## Data types

```js
// Numbers
var n = 123;
var f = 120.50;

// Strings
var s = "hello world";

// Boolean
var b = true;
```

## Reserved Words

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

## Operators

```js
// Arithmetic Operators
            var a = 33;
            var b = 10;
            var c = "Test";
            var linebreak = "<br />";
         
            document.write("a + b = ");
            result = a + b;
            document.write(result);
            document.write(linebreak);
         
            document.write("a - b = ");
            result = a - b;
            document.write(result);
            document.write(linebreak);
         
            document.write("a / b = ");
            result = a / b;
            document.write(result);
            document.write(linebreak);
         
            document.write("a % b = ");
            result = a % b;
            document.write(result);
            document.write(linebreak);
         
            document.write("a + b + c = ");
            result = a + b + c;
            document.write(result);
            document.write(linebreak);
         
            a = ++a;
            document.write("++a = ");
            result = ++a;
            document.write(result);
            document.write(linebreak);
         
            b = --b;
            document.write("--b = ");
            result = --b;
            document.write(result);
            document.write(linebreak);
// a + b = 43
// a - b = 23
// a / b = 3.3
// a % b = 3
// a + b + c = 43Test
// ++a = 35
// --b = 8

// === comparison operator with value, type
//
// undefined == null => true
// undefined === null => false

// Comparison Operators
            var a = 10;
            var b = 20;
            var linebreak = "<br />";
      
            document.write("(a == b) => ");
            result = (a == b);
            document.write(result);
            document.write(linebreak);
         
            document.write("(a < b) => ");
            result = (a < b);
            document.write(result);
            document.write(linebreak);
         
            document.write("(a > b) => ");
            result = (a > b);
            document.write(result);
            document.write(linebreak);
         
            document.write("(a != b) => ");
            result = (a != b);
            document.write(result);
            document.write(linebreak);
         
            document.write("(a >= b) => ");
            result = (a >= b);
            document.write(result);
            document.write(linebreak);
         
            document.write("(a <= b) => ");
            result = (a <= b);
            document.write(result);
            document.write(linebreak);
// (a == b) => false 
// (a < b) => true 
// (a > b) => false 
// (a != b) => true 
// (a >= b) => false 
// a <= b) => true  

// Logical Operators
            var a = true;
            var b = false;
            var linebreak = "<br />";
      
            document.write("(a && b) => ");
            result = (a && b);
            document.write(result);
            document.write(linebreak);
         
            document.write("(a || b) => ");
            result = (a || b);
            document.write(result);
            document.write(linebreak);
         
            document.write("!(a && b) => ");
            result = (!(a && b));
            document.write(result);
            document.write(linebreak);
// (a && b) => false 
// (a || b) => true 
// !(a && b) => true

// Bitwise Operators
            var a = 2; // Bit presentation 10
            var b = 3; // Bit presentation 11
            var linebreak = "<br />";
         
            document.write("(a & b) => ");
            result = (a & b);
            document.write(result);
            document.write(linebreak);
         
            document.write("(a | b) => ");
            result = (a | b);
            document.write(result);
            document.write(linebreak);
         
            document.write("(a ^ b) => ");
            result = (a ^ b);
            document.write(result);
            document.write(linebreak);
         
            document.write("(~b) => ");
            result = (~b);
            document.write(result);
            document.write(linebreak);
         
            document.write("(a << b) => ");
            result = (a << b);
            document.write(result);
            document.write(linebreak);
         
            document.write("(a >> b) => ");
            result = (a >> b);
            document.write(result);
            document.write(linebreak);
// (a & b) => 2 
// (a | b) => 3 
// (a ^ b) => 1 
// (~b) => -4 
// (a << b) => 16 
// (a >> b) => 0

// Assignment Operators
            var a = 33;
            var b = 10;
            var linebreak = "<br />";
         
            document.write("Value of a => (a = b) => ");
            result = (a = b);
            document.write(result);
            document.write(linebreak);
         
            document.write("Value of a => (a += b) => ");
            result = (a += b);
            document.write(result);
            document.write(linebreak);
         
            document.write("Value of a => (a -= b) => ");
            result = (a -= b);
            document.write(result);
            document.write(linebreak);
         
            document.write("Value of a => (a *= b) => ");
            result = (a *= b);
            document.write(result);
            document.write(linebreak);
         
            document.write("Value of a => (a /= b) => ");
            result = (a /= b);
            document.write(result);
            document.write(linebreak);
         
            document.write("Value of a => (a %= b) => ");
            result = (a %= b);
            document.write(result);
            document.write(linebreak);
// Value of a => (a = b) => 10
// Value of a => (a += b) => 20 
// Value of a => (a -= b) => 10 
// Value of a => (a *= b) => 100 
// Value of a => (a /= b) => 10
// Value of a => (a %= b) => 0

// Miscellaneous Operators
            var a = 10;
            var b = 20;
            var linebreak = "<br />";
         
            document.write ("((a > b) ? 100 : 200) => ");
            result = (a > b) ? 100 : 200;
            document.write(result);
            document.write(linebreak);
         
            document.write ("((a < b) ? 100 : 200) => ");
            result = (a < b) ? 100 : 200;
            document.write(result);
            document.write(linebreak);
// ((a > b) ? 100 : 200) => 200 
// ((a < b) ? 100 : 200) => 100

// typeof Operators
            var a = 10;
            var b = "String";
            var linebreak = "<br />";
         
            result = (typeof b == "string" ? "B is String" : "B is Numeric");
            document.write("Result => ");
            document.write(result);
            document.write(linebreak);
         
            result = (typeof a == "string" ? "A is String" : "A is Numeric");
            document.write("Result => ");
            document.write(result);
            document.write(linebreak);
// Result => B is String 
// Result => A is Numeric
```

## Decision Making

```js
// if else if
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
// switch
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
// while
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

// do while
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

// for 
            var count;
            document.write("Starting Loop" + "<br />");
         
            for(count = 0; count < 10; count++){
               document.write("Current Count : " + count );
               document.write("<br />");
            }
         
            document.write("Loop stopped!");

// for in
            var aProperty;
            document.write("Navigator Object Properties<br /> ");        
            for (aProperty in navigator) {
               document.write(aProperty);
               document.write("<br />");
            }
            document.write ("Exiting from the loop!");
```

## Javascript Runtime Architecture

아래 그림의 회색 박스는 `v8` 와 같은 interpreter engine 이고 나머지는 `chrome` 과 같은 browser 라고 생각하자.

![](https://cdn-images-1.medium.com/max/800/1*4lHHyfEhVB0LnQ3HlhSs8g.png)

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

함수를 실행한다. execution context가 만들어 진다.

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

자신이 생성될때의 스코프에서 알 수 있었던 변수를 기억하는 함수

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

## hoisting

변수를 선언하고 초기화했을때 선언부분이 최상단으로 끌어올려지는 현상

```js
console.log(zero); // 에러가 아니라 undefined
sayWow(); // 정상적으로 wow
function sayWow() {
  console.log('wow');
}
var zero = 'zero';
```

위의 코드는 아래와 같다.

```js
function sayWow() {
  console.log('wow');
}
var zero;
console.log(zero);
sayWow();
zero = 'zero';
```

함수를 표현식으로 선언한 경우는 에러가 발생한다.

```js
sayWow(); // (3)
sayYeah(); // (5) 여기서 대입되기 전에 호출해서 에러
var sayYeah = function() { // (1) 선언 (6) 대입
  console.log('yeah');
}
function sayWow() { // (2) 선언과 동시에 초기화(호이스팅)
  console.log('wow'); // (4)
}
```

## scope

lexical scoping

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

## execution context

global context 생성후 함수 호출 할때 마다 execution context가
생성된다.

execution context는 arguments, variable, scope chain, this가 저장된다.

함수가 실행될때 그 함수의 execution context에 변수가 없다면 scope
chain을 따라 올라가며 검색한다.

함수 실행이 종료되면 execution context는 사라진다.

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

## task queue

## micro task

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

## Objects

## Number

## Boolean

## Strings

## Arrays

## Date

## Math

## RegExp

## HTML DOM

다음은 DOM 의 구조를 표현한 그림이다.

![](https://www.tutorialspoint.com/javascript/images/html-dom.jpg)

# Advanced Usages

## var, let, const

* var 는 function-scoped 이고 let, const 는 block-scoped 이다.
* var 는 함수를 기준으로 hoisting 이 발생한다.

```js
// 잘된다.
for (var i = 0; i < 5; ++i)
  console.log(i);
console.log(i);

// 안된다.
for (j = 0; j < 5; ++j)
  console.log(j);
console.log(j);
//Thrown:
//ReferenceError: k is not defined
```

* IIFE 를 사용하면 var 를 선언한 것처럼 hoisting 된다.

```js
// "var i" 가 선언된 것 같다.
(function() {
  for(i = 0; i < 5; i++) {
    console.log(i)
  }
})()
console.log(i) 

// 그러나 use strict 를 사용하면 불가하다.
(function() {
  'use strict'
  for(i = 0; i < 5; i++) {
    console.log(i)
  }
})()
console.log(i) 
```

* let, const 는 변수재선언이 불가능하다. const 는 immutable 하다.

```js
let a = 'hello'
let a = 'world' // 안된다.
a = 'world'; // 잘된다.

const b = 'hello';
const b; // ERROR
b = 'world'; // ERROR
```

## promise


비동기를 구현하기 위한 object 이다. `promise` 는 `pending, fullfilled, rejected` 와 같이 3 가지 상태를 갖는다. `then()` 에서 해결하고 `catch()` 에서 오류처리를 한다. 

```js
function promiseFoo(b) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      console.log('Foo');
      if (b) {
         resolve({result: 'RESOLVE'});
      } else {
         reject(new Error('REJECT'));
      }
    }, 5000);
  });
}

const promiseA = promiseFoo(true);
console.log('promiseA created', promiseA);

const promiseB = promiseFoo(false);
console.log('promiseB created', promiseB);

promiseA.then(a => console.log(a));
promiseB
  .then(a => console.log(a))
  .catch(e => console.error(e));
```

다음은 `promise chaining` 의 예이다. `then` 에서
다시 `promise` 를 리턴한다. 그 `promise` 가 `fullfilled` 상태로 전환되면 다음 `then` 이 호출되고 `rejected` 상태로 전환되면 `catch` 가 호출된다.

```js
function promiseBar(name, stuff) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (stuff.energy > 50) {
         resolve({result: '${name} alive', loss: 10});
      } else {
         reject(new Error('${name} died'));
      }
    }, 3000);
  });
}
const bar = { energy: 70 };
promiseBar('jane', bar)
  .then(a => {
    console.log(a.result);
    bar.energy -= a.loss;
    return promiseBar('john', bar);
  })
  .then(a => {
     console.log(a.result);
     bar.energy -= a.loss;
     return promiseBar('paul', bar);
  })
  .then(a => {
     console.log(a.result);
     bar.energy -= a.loss;
     return promiseBar('sam', bar);
  })
  .catch(e => console.error(e));
```

## async, await

비동기를 구현하는 새로운 방법이다. 콜백지옥을 탈출 할 수 있다. `async` 로 함수선언을 하고 함수안에서 `promise` 앞에 `await` 로 기다린다. 이것은 `c#` 의 `IEnumerator, yield` 와 유사하다.

```js
function promiseBaz(name, stuff) {
   return new Promise((resolve, reject) => {
      setTimeout(() => {
         if (stuff.energy > 50) {
            stuff.energy -= 10;
            resolve({ result: `${name} alive` });
         } else {
            reject(new Error(`${name} dead`));
         }
      }, 1000);
   })
}
const bar = { energy: 70 };
const f = async function() {
   try {
      let a = await promiseBaz('jane', bar);
      console.log(a.result);
      a = await promiseBaz('john', bar);
      console.log(a.result);
      a = await promiseBaz('paul', bar);
      console.log(a.result);
      a = await promiseBaz('sam', bar);
      console.log(a.result);
   } catch (e) {
      console.log(e);
   }
}
```