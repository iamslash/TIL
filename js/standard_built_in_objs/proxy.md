
Intercept and redefine operations for the object.

```js
// Do nothing
const target = {
  msg1: "hello",
  msg2: "everyone",
}
const handler1 = {}
const proxy1 = new Proxy(target, handler1)
console.log(proxy1.message1); // hello
console.log(proxy1.message2); // everyone

// Do something
const target = {
  message1: "hello",
  message2: "everyone",
};
const handler2 = {
  get(target, prop, receiver) {
    return "world";
  },
};
const proxy2 = new Proxy(target, handler2);
console.log(proxy2.message1); // world
console.log(proxy2.message2); // world

// Do something in specific condition
// with Reflect
const target = {
  message1: "hello",
  message2: "everyone"
};
const handler3 = {
  get(target, prop, receiver) {
    if (prop === "message2") {
      return "world";
    }
    return Reflect.get(...arguments);
  },
};
const proxy3 = new Proxy(target, handler3);
console.log(proxy3.message1); // hello
console.log(proxy3.message2); // world
```
