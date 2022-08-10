# Abstract

clsx 에 대해 적는다.

# Materials

* [clsx | velog](https://velog.io/@robinyeon/clsx)
* [clsx | github](https://github.com/lukeed/clsx)

# Basic

짧은 line 으로 runtime 에 css string 을 merge 한다.

```js
import clsx from 'clsx';

// Strings (variadic)
const A1 = clsx('foo', true && 'bar', 'baz');
//=> 'foo bar baz'

// // Objects
// const A2 = clsx({ foo:true, bar:false, baz:isTrue() });
// //=> 'foo baz'

// Objects (variadic)
const A3 = clsx({ foo:true }, { bar:false }, null, { '--foobar':'hello' });
//=> 'foo --foobar'

// Arrays
const A4 = clsx(['foo', 0, false, 'bar']);
//=> 'foo bar'

// Arrays (variadic)
const A5 = clsx(['foo'], ['', 0, false, 'bar'], [['baz', [['hello'], 'there']]]);
//=> 'foo bar baz hello there'

// Kitchen sink (with nesting)
const A6 = clsx('foo', [1 && 'bar', { baz:false, bat:null }, ['hello', ['world']]], 'cya');
//=> 'foo bar hello world cya'


function App() {
  return (
    <div>
      {A1}<br/>
      {/* {A2}<br/> */}
      {A3}<br/>
      {A4}<br/>
      {A5}<br/>
      {A6}<br/>
    </div>
  );
}

export default App;
```
