# Materials

* [Using promise | mdn](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Using_promises)
* [JavaScript Visualized: Promises & Async/Await](https://dev.to/lydiahallie/javascript-visualized-promises-async-await-5gke)
* [Promises in JavaScript](https://zellwk.com/blog/js-promises/)

----

비동기를 구현하기 위한 object 이다. `promise` 는 `pending, resolved, rejected`
와 같이 3 가지 상태를 갖는다.

`Promise` object 의 `then()` 혹은 `catch()`을 호출하면 `Promise` object 는
resolved 상태로 된다.

`Promise` `constructor` 의 arg 는 `Function` 이다. `resolveFn, rejectFn` 을
arg 로 한다. 모두 `Function` 이다.

`resolveFn()` 를 호출하면 나중에 `then()` 이 호출되었을 때 `resolveFn()` 로 넘어간
arg 가 `then()` 의 arg 로 넘어온다.

`rejectFn()` 를 호출하면 나중에 `catch()` 가 호출되었을 때 `rejectFn()` 로 넘어간
arg 가 `catch()` 의 arg 로 넘어온다.

`Promise` 를 `then()` 에서 정상처리를 하고 `catch()` 에서 오류처리를 한다고 생각하자.

```js
// simple promise
p = new Promise(() => {}) // p is pending
p = new Promise((resolve, reject) => {}) // p is pending

// promise with calling resolve 
p = new Promise((resolve, reject) => {
  return resolve(7) // 
})
p.then(num => console.log(num)) // 7, p is resolved

// promise with calling reject
p = new Promise((resolve, reject) => {
   //return reject(7)
   return reject(new Error('7'))
})
p.then(num => console.log(num))
.catch(num => console.error(num)) // Error 7, p is resolved

// sleep with setTimeout
setTimeout(() => console.log("done"), 1000)

// promise with setTimeout
p = new Promise((resolve, reject) => {
  return setTimeout(resolve, 1000);
})
p.then(() => console.log("done"))

// sleep function with promise
const sleep = ms => {
  //return new Promise(resolve => setTimeout(resolve, ms))
  return new Promise((resolve, reject) => setTimeout(resolve, ms))
}
sleep(1000).then(() => console.log("done"))

// await sleep function with promise
r = await sleep(1000).then(() => "done")
```

```js
function promiseFoo(b) {
  return new Promise((resolveFn, rejectFn) => {
    setTimeout(() => {
      console.log('Foo');
      if (b) {
         resolveFn({result: 'RESOLVE'});
      } else {
         rejectFn(new Error('REJECT'));
      }
    }, 5000);
  });
}

const promiseA = promiseFoo(true);
console.log('promiseA created', promiseA);
// promiseA created Promise {
//   <pending>,
//   [Symbol(async_id_symbol)]: 41,
//   [Symbol(trigger_async_id_symbol)]: 5
// }

const promiseB = promiseFoo(false);
console.log('promiseB created', promiseB);
// promiseB created Promise {
//   <pending>,
//   [Symbol(async_id_symbol)]: 55,
//   [Symbol(trigger_async_id_symbol)]: 5
// }

promiseA.then(a => console.log(a)); 
// { result: 'RESOLVE' }
promiseB
  .then(a => console.log(a))
  .catch(e => console.error(e));
// [Error: REJECT]
```

다음은 `promise chaining` 의 예이다. `then` 에서 다시 `promise` 를 리턴한다. 그
`promise` 가 `resolved` 상태로 전환되면 다음 `then` 이 호출되고 `rejected`
상태로 전환되면 `catch` 가 호출된다.

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

`Promise.all()` 는 promise 배열을 받아서 하나의 promise 를 리턴한다.

```js
const promise1 = Promise.resolve(3);
const promise2 = 42;
const promise3 = new Promise((resolve, reject) => {
  setTimeout(resolve, 100, 'foo');
});

Promise.all([promise1, promise2, promise3]).then((values) => {
  console.log(values);
});
// Array [3, 42, "foo"]
```
