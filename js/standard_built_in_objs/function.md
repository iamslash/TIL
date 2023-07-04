[Function | mdn](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Function)

The Function object provides methods for functions. In JavaScript, every function is actually a Function object.

```js
// Function constructor
const sum = new Function('a', 'b', 'return a + b');
console.log(sum(2, 6));  // 8

// apply(given this, array arg) method
const numbers = [5, 6, 2, 3, 7];
const max = Math.max.apply(null, numbers);
console.log(max);  // 7
const min = Math.min.apply(null, numbers);
console.log(min);  // 2

// bind method
const module = {
  x: 42,
  getX: function() {
    return this.x;
  }
};
const unboundGetX = module.getX;
// The function gets invoked at the global scope 
console.log(unboundGetX());  // undefined

const boundGetX = unboundGetX.bind(module);
console.log(boundGetX());  // 42

// call(given this, args) method
function Product(name, price) {
  this.name = name;
  this.price = price;
}
function Food(name, price) {
  Product.call(this, name, price);
  this.category = 'food';
}
console.log(new Food('cheese', 5).name);  // "cheese"
```
