
Provide static methods to be interceptable for the object. `Reflect` is not a
constructor. We can not use it with the new operator or invoke the `Reflect`
object as a function. `Reflect` has same method names with `Proxy`.

```js
const object1 = {
  x: 1,
  y: 2
};
console.log(Reflect.get(object1, 'x'));   // 1

const array1 = ['zero', 'one'];
console.log(Reflect.get(array1, 1));      // "one"
```
