# Abstract

java script에 대해 정리한다.

# Materials

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

# Tips

* curly braces

코드가 실행된다.

```js
var a = function(a, b) {
    return a + b;
}
```

* parenthese

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

* anonymous function

```js
var a = function(a, b) {
    return a + b;
}
```

* named function


```js
var a = function f(a, b) {
    return a + b;
}
```

* closure

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

* IIFE(Immediately-invoked function expression)

```js
(function(a, b) {
    return a + b;
})(1, 2);
```

* arrow function (ES6)

```js
var a = (a, b) => a + b;
var b = (a, b) => {
    var result = a + b;
    return result;
}
```

* hosting

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

* scope

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

* execution context

global context 생성후 함수 호출 할때 마다 execution context가
생성된다.

execution context는 arguments, variable, scope chain, this가 저장된다.

함수가 실행될때 그 함수의 execution context에 변수가 없다면 scope
chain을 따라 올라가며 검색한다.

함수 실행이 종료되면 execution context는 사라진다.

* event loop

```js
function run() {
  console.log('동작');
}
console.log('시작');
setTimeout(run, 0);
console.log('끝');
```

[Loupe is a little visualisation to help you understand how JavaScript's call stack/event loop/callback queue interact with each other.](http://latentflip.com/loupe/)

* background

* task queue

* micro task

