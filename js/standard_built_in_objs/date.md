[Date | mdn](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Date)

```ts
const now = new Date();
console.log(now);                   // 2023-06-29T22:31:47.557Z

// now.getYear() was deprecated.
console.log(now.getFullYear());     // 2023

// now.getMonth() is 0-based.
console.log(now.getMonth() + 1);    // 6
console.log(now.getDate());         // 29

// Increase one day.
now.setDate(now.getDate() + 1);
console.log(now);                   // 2023-06-30T22:31:47.557Z
```
