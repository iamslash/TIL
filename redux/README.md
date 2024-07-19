
- [Abstract](#abstract)
- [Basic](#basic)
  - [Redux](#redux)
  - [To Do List with redux](#to-do-list-with-redux)
  - [react-redux](#react-redux)
  - [redux-toolkit](#redux-toolkit)
  - [redux](#redux-1)
    - [`function combineReducers<S>(reducers: ReducersMapObject): Reducer<S>`](#function-combinereducerssreducers-reducersmapobject-reducers)
    - [`function applyMiddleware(...middlewares: Middleware[]): GenericStoreEnhancer`](#function-applymiddlewaremiddlewares-middleware-genericstoreenhancer)
    - [`createStore()`](#createstore)
  - [react-redux](#react-redux-1)
  - [redux-actions](#redux-actions)
    - [`function createActions(actionsMap)`](#function-createactionsactionsmap)
    - [`function combineActions(...types)`](#function-combineactionstypes)
    - [`function handleActions(handlers, defaultState)`](#function-handleactionshandlers-defaultstate)
  - [react-router](#react-router)
  - [redux-saga](#redux-saga)
  - [Redux Debugger in Chrome](#redux-debugger-in-chrome)

-----

# Abstract

Redux 를 정리한다. 

# Basic

## Redux

* [초보자를 위한 리덕스 101](https://academy.nomadcoders.co/courses/235420/lectures/13817530)
  * [src](https://github.com/nomadcoders/vanilla-redux)

Redux 는 state 를 관리하기 위한 거대한 event loop 이다. 여기서 말하는 state 는
redux state 혹은 global state 이다. React Component 의 component state 혹은
local state 와는 다르다.

Action 은 event 를 말하고 Reducer 는 event handler 이다. 즉, Reducer 는 함수이고
변경된 redux state 를 return 한다. 변경된 redux state 가 return 되면 react
component 에게 props 형태로 전달되고 react component 의 render() 가 호출된다.
즉, rendering 된다. Reducer 의 첫번째 argument 는 state 이고 두번째 argument 는
action 이다.

Store 는 Application 의 state 이다. Store 를 생성하기 위해서는 Reducer 가
필요하다. Store instance 의 `getState()` 를 호출하면 현재 state 를 얻어올 수
있다. Store instance 의 `dispatch()` 를 특정 `action` 과 함께 호출하면 Store
instance 에 등록된 Reducer 가 그 action 을 두번째 argument 로 호출된다. 

또한 Store instance 의 `subscribe()` 를 함수와 함께 호출하면 Store 가 변경될 때
마다 그 함수가 호출된다. 그 함수에서 Store instance 의 `getState()` 를 호출하면
변경된 state 를 얻어올 수 있다.

[리덕스(Redux)란 무엇인가?](https://voidsatisfaction.github.io/2017/02/24/what-is-redux/)

Redux 는 다음과 같은 흐름으로 처리된다.

* Store 에 Component 를 subscribe 한다.
* User 가 button 을 click 하면 Store 의 dispatch 함수가 특정 action 을 argument
  로 호출된다.
* Store 에 등록된 Reducer 가 호출된다. 
* Reducer 는 redux state 를 변경하여 return 한다.
* Store 에 미리 subscribe 된 Component 에게 변경된 redux state 가 props 형태로
  전달된다.
* Component 는 props 에서 변경된 redux state 를 이용하여 rendering 한다.

```js
import { createStore } from "redux";

const add = document.getElementById("add");
const minus = document.getElementById("minus");
const number = document.querySelector("span");

number.innerText = 0;

const ADD = "ADD";
const MINUS = "MINUS";

const countModifier = (count = 0, action) => {
  switch (action.type) {
    case ADD:
      return count + 1;
    case MINUS:
      return count - 1;
    default:
      return count;
  }
};

const countStore = createStore(countModifier);

const onChange = () => {
  number.innerText = countStore.getState();
};

countStore.subscribe(onChange);

const handleAdd = () => {
  countStore.dispatch({ type: ADD });
};

const handleMinus = () => {
  countStore.dispatch({ type: MINUS });
};

add.addEventListener("click", handleAdd);
minus.addEventListener("click", handleMinus);
```

## To Do List with redux

Redux 를 이용하여 간단한 To Do list 를 구현해 본다.

* [Vanilla-redux To Do List by nomad coders @ src](https://github.com/nomadcoders/vanilla-redux/blob/794f2a3eb7d169de7ca240b163e481a22653f7bd/src/index.js)

```js
import { createStore } from "redux";
const form = document.querySelector("form");
const input = document.querySelector("input");
const ul = document.querySelector("ul");

const ADD_TODO = "ADD_TODO";
const DELETE_TODO = "DELETE_TODO";

const addToDo = text => {
  return {
    type: ADD_TODO,
    text
  };
};

const deleteToDo = id => {
  return {
    type: DELETE_TODO,
    id
  };
};

const reducer = (state = [], action) => {
  switch (action.type) {
    case ADD_TODO:
      const newToDoObj = { text: action.text, id: Date.now() };
      return [newToDoObj, ...state];
    case DELETE_TODO:
      const cleaned = state.filter(toDo => toDo.id !== action.id);
      return cleaned;
    default:
      return state;
  }
};

const store = createStore(reducer);

store.subscribe(() => console.log(store.getState()));

const dispatchAddToDo = text => {
  store.dispatch(addToDo(text));
};

const dispatchDeleteToDo = e => {
  const id = parseInt(e.target.parentNode.id);
  store.dispatch(deleteToDo(id));
};

const paintToDos = () => {
  const toDos = store.getState();
  ul.innerHTML = "";
  toDos.forEach(toDo => {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.innerText = "DEL";
    btn.addEventListener("click", dispatchDeleteToDo);
    li.id = toDo.id;
    li.innerText = toDo.text;
    li.appendChild(btn);
    ul.appendChild(li);
  });
};

store.subscribe(paintToDos);

const onSubmit = e => {
  e.preventDefault();
  const toDo = input.value;
  input.value = "";
  dispatchAddToDo(toDo);
};

form.addEventListener("submit", onSubmit);
```

## react-redux

앞서 제작한 To Do List App 을 React Redux 를 사용하여 더욱 효율적으로 구현해보자.

* [Vanilla-redux react-redux by nomad coders @ src](https://github.com/nomadcoders/vanilla-redux/tree/ccaa1acd081f27239f2cc8ad3c571bd0a9923f73/src)

React-Router 는 url path 에 따라 routing 기능을 지원하는 library 이다. url `/` 은 `Home` component, `/:id` 는 `Detail` component 로 routing 된다.

```js
import React from "react";
import { HashRouter as Router, Route } from "react-router-dom";
import Home from "../routes/Home";
import Detail from "../routes/Detail";

function App() {
  return (
    <Router>
      <Route path="/" exact component={Home}></Route>
      <Route path="/:id" component={Detail}></Route>
    </Router>
  );
}

export default App;
```

react-redux 의 Provider 는 state 변경사항을 자손에게 전달할 수 있게 해준다.

```js
import React from "react";
import ReactDOM from "react-dom";
import App from "./components/App";
import { Provider } from "react-redux";
import store from "./store";

ReactDOM.render(
  <Provider store={store}>
    <App />
  </Provider>,
  document.getElementById("root")
);
```

`connect` 를 호출하여 `mapStateToProps` 와 함께 특정 Component 를 연결할 수
있다. `mapStateToProps` 는 state 가 주어지면 state 를 포함한 props 를 리턴하는
함수이다. 즉, state 가 변경되면 `mapStateToProps` 가 호출된다. 그리고
`mapStateToProps` 는 state 가 포함된 props 를 리턴한다. 리턴된 props 는
component 의 render 함수로 전달된다. 렌더 함수안에서 props 를 통해서 변경된
state 를 읽어올 수 있다. 

```js
import React, { useState } from "react";
import { connect } from "react-redux";

function Home({ toDos }) {
  const [text, setText] = useState("");
  function onChange(e) {
    setText(e.target.value);
  }
  function onSubmit(e) {
    e.preventDefault();
    setText("");
  }
  return (
    <>
      <h1>To Do</h1>
      <form onSubmit={onSubmit}>
        <input type="text" value={text} onChange={onChange} />
        <button>Add</button>
      </form>
      <ul>{JSON.stringify(toDos)}</ul>
    </>
  );
}

function mapStateToProps(state) {
  return { toDos: state };
}

export default connect(mapStateToProps)(Home);
```

`connect` 를 호출하여 `mapDispatchToProps` 와 함께 특정 Component 를 연결할 수
있다. `mapDispatchToProps` 는 dispatch function 이 주어지면 dispatch function 을
포함한 props 를 리턴하는 함수이다. 리턴된 props 는 `connect` 에 연결된 component
의 render 함수로 전달된다. render 함수안에서 props 를 통해 dispatch function 을
읽어올 수 있다. 특정 dispatch function 을 호출하면 특정 reducer 를 호출할 수
있다.

```js
import React, { useState } from "react";
import { connect } from "react-redux";
import { actionCreators } from "../store";

function Home({ toDos, addToDo }) {
  const [text, setText] = useState("");
  function onChange(e) {
    setText(e.target.value);
  }
  function onSubmit(e) {
    e.preventDefault();
    addToDo(text);
    setText("");
  }
  return (
    <>
      <h1>To Do</h1>
      <form onSubmit={onSubmit}>
        <input type="text" value={text} onChange={onChange} />
        <button>Add</button>
      </form>
      <ul>{JSON.stringify(toDos)}</ul>
    </>
  );
}

function mapStateToProps(state) {
  return { toDos: state };
}

function mapDispatchToProps(dispatch) {
  return {
    addToDo: text => dispatch(actionCreators.addToDo(text))
  };
}

export default connect(mapStateToProps, mapDispatchToProps)(Home);
```

다음은 `connect` 에 두번째 argument 로 `mapDispatchToProps` 를 전달하고 `ToDo`
component 와 연결한다. `ToDo` component 의 button 이 click 되면 `ToDo` component
에 전달된 props 의 두번째 요소인 dispatch function 이 호출된다. dispatch
function 에 해당하는 `onBtnClick` 이 호출되면 `DELETE` action 이 발생하고
Reducer 가 호출된다. reducer 는 변경된 state 를 리턴하고 `ToDo` component 의
render function 이 변경된 state argument 와 함께 호출된다.

```js
import React from "react";
import { connect } from "react-redux";
import { actionCreators } from "../store";

function ToDo({ text, onBtnClick }) {
  return (
    <li>
      {text} <button onClick={onBtnClick}>DEL</button>
    </li>
  );
}

function mapDispatchToProps(dispatch, ownProps) {
  return {
    onBtnClick: () => dispatch(actionCreators.deleteToDo(ownProps.id))
  };
}

export default connect(null, mapDispatchToProps)(ToDo);
```

`Reducer` 에서 `DELETE` 을 처리한다. `filter` 를 이용하여 특정 id 를 제거한 목록을 변경된 state 로 리턴한다.

```js
import { createStore } from "redux";

const ADD = "ADD";
const DELETE = "DELETE";

const addToDo = text => {
  return {
    type: ADD,
    text
  };
};

const deleteToDo = id => {
  return {
    type: DELETE,
    id: parseInt(id)
  };
};

const reducer = (state = [], action) => {
  switch (action.type) {
    case ADD:
      return [{ text: action.text, id: Date.now() }, ...state];
    case DELETE:
      return state.filter(toDo => toDo.id !== action.id);
    default:
      return state;
  }
};

const store = createStore(reducer);

export const actionCreators = {
  addToDo,
  deleteToDo
};

export default store;
```


## redux-toolkit

* [createAction @ redux](https://redux-toolkit.js.org/api/createAction)
* [createReducer @ redux](https://redux-toolkit.js.org/api/createReducer)
* [configureStore @ redux](https://redux-toolkit.js.org/api/configureStore)
* [createSlice @ redux](https://redux-toolkit.js.org/api/createSlice)

## redux

### `function combineReducers<S>(reducers: ReducersMapObject): Reducer<S>`

* [combineReducers(reducers) @ redux](https://redux.js.org/api/combinereducers)

----

The combineReducers helper function turns an object whose values are different
reducing functions into a single reducing function you can pass to createStore.

### `function applyMiddleware(...middlewares: Middleware[]): GenericStoreEnhancer`

* [applyMiddleware(...middleware)](https://redux.js.org/api/applymiddleware)

----

Middleware is the suggested way to extend Redux with custom functionality.
Middleware lets you wrap the store's dispatch method for fun and profit. 

This is an example of custom log middleware.

```js
import { createStore, applyMiddleware } from 'redux'
import todos from './reducers'

function logger({ getState }) {
  return next => action => {
    console.log('will dispatch', action)

    // Call the next dispatch method in the middleware chain.
    const returnValue = next(action)

    console.log('state after dispatch', getState())

    // This will likely be the action itself, unless
    // a middleware further in chain changed it.
    return returnValue
  }
}

const store = createStore(todos, ['Use Redux'], applyMiddleware(logger))

store.dispatch({
  type: 'ADD_TODO',
  text: 'Understand the middleware'
})
// (These lines will be logged by the middleware:)
// will dispatch: { type: 'ADD_TODO', text: 'Understand the middleware' }
// state after dispatch: [ 'Use Redux', 'Understand the middleware' ]
```

### `createStore()`

* [createStore(reducer, [preloadedState], [enhancer])](https://redux.js.org/api/createstore)

----

Creates a Redux store that holds the complete state tree of your app. There
should only be a single store in your app.

## react-redux

* [What's the '@' (at symbol) in the Redux @connect decorator?](https://stackoverflow.com/questions/32646920/whats-the-at-symbol-in-the-redux-connect-decorator)

-----

`@connect` 는 decorator 이다. 다음의 두 코드는 같다.

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

## redux-actions

### `function createActions(actionsMap)`

* [createAction(s) @ redux-actions](https://redux-actions.js.org/api/createaction) 

----

Returns an object mapping action types to action creators. 

### `function combineActions(...types)`

* [createAction(s) @ redux-actions](https://redux-actions.js.org/api/createaction) 

----

Combine any number of action types or action creators. 

### `function handleActions(handlers, defaultState)`

* [handleAction(s) @ redux-actions](https://redux-actions.js.org/api/handleaction) 

----

Creates multiple reducers using handleAction() and combines them into a single reducer that handles multiple actions.

## react-router

* [REACT ROUTER](https://reactrouter.com/)
  * [React Router Introduction @ youtube](https://www.youtube.com/watch?time_continue=542&v=cKnc8gXn80Q&feature=emb_logo)
-----

navigational components


## redux-saga

* [redux-saga](https://github.com/redux-saga/redux-saga)
  * [한글](https://mskims.github.io/redux-saga-in-korean/)
* [redux-saga에서 비동기 처리 싸움](https://qiita.com/kuy/items/716affc808ebb3e1e8ac)

## Redux Debugger in Chrome

* [React Redux Tutorials - 24 - Redux Devtool Extension @ youtube](https://www.youtube.com/watch?v=IlM7497j6LY)
* [#4.3 configureStore @ nomad](https://academy.nomadcoders.co/courses/235420/lectures/14735315)

----

```js
import { composeWithDevTools } from 'redux-devtools-extension';
...
const store = createStore(
  rootReducer,
  composeWithDevTools(applyMiddleware(logger, ReduxThunk, sagaMiddleware))
);
```
