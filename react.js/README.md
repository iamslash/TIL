- [Abstract](#abstract)
- [Templates](#templates)
- [Materials](#materials)
- [Movie Tutorial](#movie-tutorial)
  - [Create React App](#create-react-app)
  - [Create React Components with JSX](#create-react-components-with-jsx)
  - [Data flow with Props](#data-flow-with-props)
  - [Lists With `.maps`](#lists-with-maps)
  - [Validating Props With Prop Types](#validating-props-with-prop-types)
  - [Component Lifecycle](#component-lifecycle)
  - [Thinking in React Component State](#thinking-in-react-component-state)
  - [Practicing this setState](#practicing-this-setstate)
  - [SetState Caveats](#setstate-caveats)
  - [Loading States](#loading-states)
  - [AJAX on React](#ajax-on-react)
  - [Promises](#promises)
  - [Async Await](#async-await)
  - [Updating Movie](#updating-movie)
  - [CSS for Movie](#css-for-movie)
  - [Building for Production](#building-for-production)
- [Advanced](#advanced)
  - [Redux](#redux)
  - [To Do List with redux](#to-do-list-with-redux)
  - [react-redux](#react-redux)
  - [rendering Sequences](#rendering-sequences)
  - [Smart vs Dumb](#smart-vs-dumb)
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
  - [Ant Design](#ant-design)
  - [redux-saga](#redux-saga)
  - [Redux Debugger in Chrome](#redux-debugger-in-chrome)
  - [Don't Use `useEffect`](#dont-use-useeffect)
- [Architectures](#architectures)

----

# Abstract

react.js 는 view library 이다. redux 는 state management library 이다. 

props 는 function parameter 와 유사하다. immutable 이다. state 는 local variable of function 과 같다. mutable 이다. [[React] Props와  State의 차이](https://singa-korean.tistory.com/37) 참고.

react.js 의 문서는 완성도가 높다. 모두 읽어봐야 한다. [https://reactjs.org/docs/getting-started.html](https://reactjs.org/docs/getting-started.html) 

# Templates

* [gogo-react](https://gogo-react-docs.coloredstrategies.com/docs/gettingstarted/introduction)
* [react-admin](https://github.com/marmelab/react-admin)
  * An enterprise-class UI design language and React UI library
* [ant-design](https://github.com/ant-design/ant-design)
* [coreui](https://coreui.io/react/)

# Materials

* [리액트를 다루는 기술(개정판) -16.8버전](https://www.gilbut.co.kr/book/view?bookcode=BN002496)
  * [src](https://github.com/velopert/learning-react)
  * [벨로퍼트와 함께하는 모던 리액트](https://react.vlpt.us/)
* [한 번에 끝내는 React의 모든 것 초격차 패키지 | FastCampus](https://fastcampus.co.kr/dev_online_react/)
  * 유료이지만 그럴듯 함.
* [React 적용 가이드 - React와 Redux @ NaverD2](https://d2.naver.com/helloworld/1848131)
  * [src](https://github.com/naver/react-sample-code)
  * [React 적용 가이드 - React 작동 방법 @ NaverD2](https://d2.naver.com/helloworld/9297403)
  * [React 적용 가이드 - 네이버 메일 모바일 웹 적용기 @ NaverD2](https://d2.naver.com/helloworld/4966453)
* [Redux 문서(한글)](https://ko.redux.js.org/)
* [create react app](https://github.com/facebook/create-react-app)
  * react app wizard
* [nomad academy](https://academy.nomadcoders.co/courses/category/KR)
  * react class
* [reactjs @ inflearn](https://www.inflearn.com/course/reactjs-web/)
  * react class for beginner
  * [src movie app](https://github.com/nomadcoders/movie_app)
* [ReactJS로 웹 서비스 만들기](https://academy.nomadcoders.co/p/reactjs-fundamentals)
  * [src movie app 2019 update](https://github.com/nomadcoders/movie_app_2019)
* [초보자를 위한 리덕스 101](https://academy.nomadcoders.co/courses/235420/lectures/13817530)
  * [src](https://github.com/nomadcoders/vanilla-redux)
* [Create-React-App: A Closer Look](https://github.com/nitishdayal/cra_closer_look)
  * stuffs in create-react-app in detail
* [아마 이게 제일 이해하기 쉬울걸요? React + Redux 플로우의 이해](https://medium.com/@ca3rot/%EC%95%84%EB%A7%88-%EC%9D%B4%EA%B2%8C-%EC%A0%9C%EC%9D%BC-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0-%EC%89%AC%EC%9A%B8%EA%B1%B8%EC%9A%94-react-redux-%ED%94%8C%EB%A1%9C%EC%9A%B0%EC%9D%98-%EC%9D%B4%ED%95%B4-1585e911a0a6)
* [Redux 정복하기](https://velopert.com/3365)
  * [1. 카운터 만들기](https://velopert.com/3346)
  * [2. 멀티 카운터 만들기](https://velopert.com/3352)
  * [3. Immutable.js 익히기](https://velopert.com/3354)
  * [4. Ducks 구조와 react-actions 익히기](https://velopert.com/3358)
  * [5. 주소록에 Redux 끼얹기](https://velopert.com/3360)
* [redux @ github](https://github.com/reduxjs/redux)
  * [Getting Started with Redux](https://egghead.io/courses/getting-started-with-redux)
  * [Building React Applications with Idiomatic Redux](https://egghead.io/courses/building-react-applications-with-idiomatic-redux)

# Movie Tutorial

## Create React App

React 애플리케이션을 빠르게 시작하기 위해 Create React App 도구를 사용합니다. 다음 명령어를 통해 설치 및 프로젝트 생성이 가능합니다:

```bash
# nvm 설치 (이미 설치된 경우 생략 가능)
$ brew install nvm

# nvm 설정
$ mkdir ~/.nvm
$ export NVM_DIR="$HOME/.nvm"
$ [ -s "/usr/local/opt/nvm/nvm.sh" ] && \. "/usr/local/opt/nvm/nvm.sh"  # This loads nvm
$ [ -s "/usr/local/opt/nvm/etc/bash_completion" ] && \. "/usr/local/opt/nvm/etc/bash_completion"  # This loads nvm bash_completion

# nvm을 사용하여 Node.js 17.0.0 설치 및 사용
$ nvm install 17.0.0
$ nvm use 17.0.0

# yarn 설치
$ npm install -g yarn

# create-react-app 설치
$ yarn global add create-react-app

# React 애플리케이션 생성
$ create-react-app my-app

# 생성한 애플리케이션 디렉토리로 이동
$ cd my-app

# 애플리케이션 실행
$ yarn start
```

## Create React Components with JSX

```js
////////////////////////////////////////////////////////////////////////////////
// public/index.html:

<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>React App</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>

////////////////////////////////////////////////////////////////////////////////
// src/index.js:
import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import './index.css';

ReactDOM.render(<App />, document.getElementById('root'));

////////////////////////////////////////////////////////////////////////////////
// src/App.js:
import React from 'react';
import './App.css';
import Movie from './Movie';

// App 컴포넌트는 Movie 컴포넌트를 렌더링합니다.
function App() {
  return (
    <div className="App">
      <Movie />
      <Movie />
    </div>
  );
}

export default App;

////////////////////////////////////////////////////////////////////////////////
// src/Movie.js:
import React from 'react';
import './Movie.css';

// Movie 컴포넌트는 MoviePoster 컴포넌트를 렌더링합니다.
function Movie() {
  return (
    <div>
      <MoviePoster />
      <h1>Hello This is a movie</h1>
    </div>
  );
}

// MoviePoster 컴포넌트는 이미지를 렌더링합니다.
function MoviePoster() {
  return (
    <img src='http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778581_STD.jpg' alt="Movie Poster" />
  );
}

export default Movie;
```

## Data flow with Props

![](https://miro.medium.com/max/1540/0*NLC2HyJRjh0_3r0e.)

data 를 `src/App.js` 에 선언해 보자. 그리고 `Movie` component 에게 props 형태로 전달한다.

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.js:
import React from 'react';
import Movie from './Movie';

const movieTitles = [
  "Matrix",
  "Full Metal Jacket",
  "Oldboy",
  "Star wars"
]

const movieImages = [
  'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778581_STD.jpg',
  'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778470_STD.jpg',
  'https://cdn1.thr.com/sites/default/files/imagecache/768x433/2017/06/143289-1496932680-mm_2012_047_italy_57_-_h_2017.jpg',
  'https://cdn1.thr.com/sites/default/files/2017/06/143226-1496932903-mm_2012_047_italy_11_-_embed_2018.jpg',
]

function App() {
  return (
    <div className="App">
      <Movie title={movieTitles[0]} poster={movieImages[0]} />
      <Movie title={movieTitles[1]} poster={movieImages[1]} />
      <Movie title={movieTitles[2]} poster={movieImages[2]} />
      <Movie title={movieTitles[3]} poster={movieImages[3]} />
    </div>
  );
}

export default App;

////////////////////////////////////////////////////////////////////////////////
// src/Movie.js:
import React from 'react';
import PropTypes from 'prop-types';
import './Movie.css';

function Movie({ title, poster }) {
  return (
    <div>
      <MoviePoster poster={poster} />
      <h1>{title}</h1>
    </div>
  );
}

function MoviePoster({ poster }) {
  return (
    <img src={poster} alt="Movie Poster" />
  );
}

Movie.propTypes = {
  title: PropTypes.string.isRequired,
  poster: PropTypes.string.isRequired
};

MoviePoster.propTypes = {
  poster: PropTypes.string.isRequired
};

export default Movie;
```

## Lists With `.maps`

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.js:
import React from 'react';
import Movie from './Movie';

const movies = [
  {
    title: "Matrix",
    poster: 'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778581_STD.jpg',
  },
  {
    title: "Full Metal Jacket",
    poster: 'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778470_STD.jpg',
  },
  {
    title: "Oldboy",
    poster: 'https://cdn1.thr.com/sites/default/files/imagecache/768x433/2017/06/143289-1496932680-mm_2012_047_italy_57_-_h_2017.jpg',
  },
  {
    title: "Star wars",
    poster: 'https://cdn1.thr.com/sites/default/files/2017/06/143226-1496932903-mm_2012_047_italy_11_-_embed_2018.jpg',
  },
]

function App() {
  return (
    <div className="App">
      {movies.map(movie => (
        <Movie title={movie.title} poster={movie.poster} />
      ))}
    </div>
  );
}

export default App;
```

## Validating Props With Prop Types

`static propTypes` 를 선언하여 props 의 값을 제어할 수 있다. 이때 PropTypes module 이 설치되어야 한다. `yarn add PropTypes`

```js
////////////////////////////////////////////////////////////////////////////////
// src/Movie.js:
import React from 'react';
import PropTypes from 'prop-types';
import './Movie.css';

function Movie({ title, poster }) {
  return (
    <div>
      <MoviePoster poster={poster} />
      <h1>{title}</h1>
    </div>
  );
}

function MoviePoster({ poster }) {
  return (
    <img src={poster} alt="Movie Poster" />
  );
}

Movie.propTypes = {
  title: PropTypes.string.isRequired,
  poster: PropTypes.string.isRequired
};

MoviePoster.propTypes = {
  poster: PropTypes.string.isRequired
};

export default Movie;
```

## Component Lifecycle

하나의 component 는 다음과 같은 순서로 `Render, Update` 가 수행된다. Override
function 의 순서를 주의하자.

```js
  // Render: componentWillMount() -> render() -> componentDidMount()
  //
  // Update: componentWillReceiveProps() -> shouldComponentUpdate() -> 
  //         componentWillUpate() -> render() -> componentDidUpdate()
```

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.js:
import React, { useEffect } from 'react';
import Movie from './Movie';

const movies = [
  {
    title: "Matrix",
    poster: 'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778581_STD.jpg',
  },
  {
    title: "Full Metal Jacket",
    poster: 'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778470_STD.jpg',
  },
  {
    title: "Oldboy",
    poster: 'https://cdn1.thr.com/sites/default/files/imagecache/768x433/2017/06/143289-1496932680-mm_2012_047_italy_57_-_h_2017.jpg',
  },
  {
    title: "Star wars",
    poster: 'https://cdn1.thr.com/sites/default/files/2017/06/143226-1496932903-mm_2012_047_italy_11_-_embed_2018.jpg',
  },
]

const App = () => {
  useEffect(() => {
    console.log("componentWillMount");
    console.log("componentDidMount");
  }, []);

  console.log("render");

  return (
    <div className="App">
      {movies.map((movie, index) => (
        <Movie title={movie.title} poster={movie.poster} key={index} />
      ))}
    </div>
  );
}

export default App;
```

## Thinking in React Component State

`App` component 에 `state` 를 선언하고 `componentDidMount()` 에서 바꿔보자. `this.setState()` 함수를 호출하면 `render()` 가 호출된다. `state` 를 바꾸고 `this.setState()` 를 호출하여 화면을 업데이트한다. 여기서 언급한 `state` 는 redux 의 `state` 과는 다르다는 것을 주의하자. [React State vs. Redux State: When and Why?](https://spin.atomicobject.com/2017/06/07/react-state-vs-redux-state/)

다음은 smart component 예이다.

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.js:
import React, { Component } from 'react';
import Movie from './Movie';

const movies = [
  {
    title: "Matrix",
    poster: 'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778581_STD.jpg',
  },
  {
    title: "Full Metal Jacket",
    poster: 'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778470_STD.jpg',
  },
  {
    title: "Oldboy",
    poster: 'https://cdn1.thr.com/sites/default/files/imagecache/768x433/2017/06/143289-1496932680-mm_2012_047_italy_57_-_h_2017.jpg',
  },
  {
    title: "Star wars",
    poster: 'https://cdn1.thr.com/sites/default/files/2017/06/143226-1496932903-mm_2012_047_italy_11_-_embed_2018.jpg',
  },
]

class App extends Component {
  state = {
    greeting: 'Hello'
  }
  componentDidMount() {
    setTimeout(() => {
      this.setState({
        greeting: 'Hello Again'
      })
    }, 2000)
  }
  render() {
    console.log("render");
    return (
      <div className="App">
        {this.state.greeting}
        {movies.map((movie, index) => (
          <Movie title={movie.title} poster={movie.poster} key={index} />
        ))}
      </div>
    );
  }
}

export default App;
```

다음은 functional component 예이다.

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.js:
import React, { useEffect } from 'react';
import Movie from './Movie';

const movies = [
  {
    title: "Matrix",
    poster: 'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778581_STD.jpg',
  },
  {
    title: "Full Metal Jacket",
    poster: 'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778470_STD.jpg',
  },
  {
    title: "Oldboy",
    poster: 'https://cdn1.thr.com/sites/default/files/imagecache/768x433/2017/06/143289-1496932680-mm_2012_047_italy_57_-_h_2017.jpg',
  },
  {
    title: "Star wars",
    poster: 'https://cdn1.thr.com/sites/default/files/2017/06/143226-1496932903-mm_2012_047_italy_11_-_embed_2018.jpg',
  },
]

// 함수형 컴포넌트로 App 정의
const App = () => {
  // useEffect 훅을 사용하여 componentWillMount 및 componentDidMount 대체
  useEffect(() => {
    console.log("componentWillMount");
    console.log("componentDidMount");
  }, []); // 빈 배열을 두 번째 인수로 주어 componentDidMount와 동일한 효과

  console.log("render"); // 렌더링 시마다 출력

  return (
    <div className="App">
      {movies.map((movie, index) => (
        // Movie 컴포넌트에 title과 poster props 전달
        <Movie title={movie.title} poster={movie.poster} key={index} />
      ))}
    </div>
  );
}

export default App;
```

## Practicing this setState

`App` component 의 `state` 으로 title, poster 를 옮기자. 그리고 일정 시간 이후에 `state` 을 변경해 보자. `...this.state.movies` 를 이용하면 기존의 array 에 `this.state.movies` 를 unwind 해서 추가할 수 있다. 이 방법을 이용하면 스크롤을 아래로 내렸을 때 infinite scroll 을 구현할 수 있다.

다음은 smart component 의 예이다.

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.js:
import React, { Component } from 'react';
import Movie from './Movie';

class App extends Component {
  state = {
    greeting: 'Hello World',
    movies: [
      {
        title: "Matrix",
        poster: 'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778581_STD.jpg',
      },
      {
        title: "Full Metal Jacket",
        poster: 'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778470_STD.jpg',
      },
      {
        title: "Oldboy",
        poster: 'https://cdn1.thr.com/sites/default/files/imagecache/768x433/2017/06/143289-1496932680-mm_2012_047_italy_57_-_h_2017.jpg',
      },
      {
        title: "Star wars",
        poster: 'https://cdn1.thr.com/sites/default/files/2017/06/143226-1496932903-mm_2012_047_italy_11_-_embed_2018.jpg',
      },
    ]    
  }
  componentDidMount() {
    setTimeout(() => {
      this.setState({
        movies: [
          ...this.state.movies,
          {
            title: "Trainspotting",
            poster: 'https://cdn1.thr.com/sites/default/files/2017/06/143226-1496932903-mm_2012_047_italy_11_-_embed_2018.jpg',
          }
        ]
      })
    }, 2000)
  }
  render() {
    console.log("render");
    return (
      <div className="App">
        {this.state.greeting}
        {this.state.movies.map((movie, index) => (
          <Movie title={movie.title} poster={movie.poster} key={index} />
        ))}
      </div>
    );
  }
}

export default App;
```

다음은 functional component 예이다.

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.js:
import React, { useState, useEffect } from 'react';
import Movie from './Movie';

const App = () => {
  // greeting 상태 정의
  const [greeting, setGreeting] = useState('Hello World');
  // movies 상태 정의
  const [movies, setMovies] = useState([
    {
      title: "Matrix",
      poster: 'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778581_STD.jpg',
    },
    {
      title: "Full Metal Jacket",
      poster: 'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778470_STD.jpg',
    },
    {
      title: "Oldboy",
      poster: 'https://cdn1.thr.com/sites/default/files/imagecache/768x433/2017/06/143289-1496932680-mm_2012_047_italy_57_-_h_2017.jpg',
    },
    {
      title: "Star wars",
      poster: 'https://cdn1.thr.com/sites/default/files/2017/06/143226-1496932903-mm_2012_047_italy_11_-_embed_2018.jpg',
    },
  ]);

  // useEffect 훅을 사용하여 componentDidMount 대체
  useEffect(() => {
    setTimeout(() => {
      // 이전 상태를 복사하여 새로운 영화 데이터를 추가
      setMovies((prevMovies) => [
        ...prevMovies,
        {
          title: "Trainspotting",
          poster: 'https://cdn1.thr.com/sites/default/files/2017/06/143226-1496932903-mm_2012_047_italy_11_-_embed_2018.jpg',
        }
      ]);
    }, 2000);
  }, []); // 빈 배열을 두 번째 인수로 주어 componentDidMount와 동일한 효과

  console.log("render"); // 렌더링 시마다 출력

  return (
    <div className="App">
      {greeting}
      {movies.map((movie, index) => (
        // Movie 컴포넌트에 title과 poster props 전달
        <Movie title={movie.title} poster={movie.poster} key={index} />
      ))}
    </div>
  );
}

export default App;
```

## SetState Caveats

```js
  // Render:  -> render() -> componentDidMount()
  //
  // Update: componentWillReceiveProps() -> shouldComponentUpdate() -> 
  //         componentWillUpate() -> render() -> componentDidUpdate()
```

`setState` 를 component lifecycle event handler (`render, componentWillMount, componentWillReceiveProps, shouldComponentUpdate, componentWillUpate, componentDidUpdate`) 에서 호출할 때 주의해야 한다.

* [React setState usage and gotchas](https://itnext.io/react-setstate-usage-and-gotchas-ac10b4e03d60)
  * [Boost your React with State Machines](https://www.freecodecamp.org/news/boost-your-react-with-state-machines-1e9641b0aa43/)

다음은 smart component 예이다.

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.js:
import React, { Component } from 'react';
import Movie from './Movie';

class App extends Component {
  state = {
    movies: []
  }
  componentDidMount() {
    setTimeout(() => {
      this.setState({
        movies: [
          {
            title: "Matrix",
            poster: 'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778581_STD.jpg',
          },
          {
            title: "Full Metal Jacket",
            poster: 'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778470_STD.jpg',
          },
          {
            title: "Oldboy",
            poster: 'https://cdn1.thr.com/sites/default/files/imagecache/768x433/2017/06/143289-1496932680-mm_2012_047_italy_57_-_h_2017.jpg',
          },
          {
            title: "Star wars",
            poster: 'https://cdn1.thr.com/sites/default/files/2017/06/143226-1496932903-mm_2012_047_italy_11_-_embed_2018.jpg',
          },
        ]
      })
    }, 2000)
  }
  render() {
    console.log("render");
    return (
      <div className="App">
        {this.state.movies.length > 0 ? (
          this.state.movies.map((movie, index) => (
            <Movie title={movie.title} poster={movie.poster} key={index} />
          ))
        ) : (
          'Loading...'
        )}
      </div>
    );
  }
}

export default App;
```

다음은 functional component 예이다.

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.js:
import React, { useState, useEffect } from 'react';
import Movie from './Movie';

const App = () => {
  // movies 상태 정의, 초기값은 빈 배열
  const [movies, setMovies] = useState([]);

  // useEffect 훅을 사용하여 componentDidMount 대체
  useEffect(() => {
    setTimeout(() => {
      // setMovies 함수를 사용하여 movies 상태 업데이트
      setMovies([
        {
          title: "Matrix",
          poster: 'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778581_STD.jpg',
        },
        {
          title: "Full Metal Jacket",
          poster: 'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778470_STD.jpg',
        },
        {
          title: "Oldboy",
          poster: 'https://cdn1.thr.com/sites/default/files/imagecache/768x433/2017/06/143289-1496932680-mm_2012_047_italy_57_-_h_2017.jpg',
        },
        {
          title: "Star wars",
          poster: 'https://cdn1.thr.com/sites/default/files/2017/06/143226-1496932903-mm_2012_047_italy_11_-_embed_2018.jpg',
        },
      ]);
    }, 2000);
  }, []); // 빈 배열을 두 번째 인수로 주어 componentDidMount와 동일한 효과

  console.log("render"); // 렌더링 시마다 출력

  return (
    <div className="App">
      {movies.length > 0 ? (
        movies.map((movie, index) => (
          // Movie 컴포넌트에 title과 poster props 전달
          <Movie title={movie.title} poster={movie.poster} key={index} />
        ))
      ) : (
        'Loading...' // movies 배열이 비어 있을 때 표시할 로딩 메시지
      )}
    </div>
  );
}

export default App;
```

## Loading States

loading screen 을 구현해 보자. `App` component 에 rendering 을 시작하자 마자 `Loading...` 을 출력하고 일정 시간이 지나면 state 을 업데이트하여 movies 가 rendering 되도록 해보자.

다음은 smart component 예이다.

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.js:
import React, { Component } from 'react';
import Movie from './Movie';

class App extends Component {
  state = {
    greeting: 'Hello World',
    movies: null,
  }
  componentDidMount() {
    setTimeout(() => {
      this.setState({
        movies: [
          {
            title: "Matrix",
            poster: 'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778581_STD.jpg',
          },
          {
            title: "Full Metal Jacket",
            poster: 'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778470_STD.jpg',
          },
          {
            title: "Oldboy",
            poster: 'https://cdn1.thr.com/sites/default/files/imagecache/768x433/2017/06/143289-1496932680-mm_2012_047_italy_57_-_h_2017.jpg',
          },
          {
            title: "Star wars",
            poster: 'https://cdn1.thr.com/sites/default/files/2017/06/143226-1496932903-mm_2012_047_italy_11_-_embed_2018.jpg',
          },
        ]    
      })
    }, 2000)
  }
  _renderMovies = () => {
    const movies = this.state.movies.map((movie, index) => (
      <Movie title={movie.title} poster={movie.poster} key={index} />
    ));
    return movies;
  }
  render() {
    console.log("render");
    return (
      <div className="App">
        {this.state.movies ? this._renderMovies() : 'Loading...'}
      </div>
    );
  }
}

export default App;
```

다음은 functional component 예이다.

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.js:
import React, { useState, useEffect } from 'react';
import Movie from './Movie';

const App = () => {
  // greeting 상태 정의, 초기값은 'Hello World'
  const [greeting, setGreeting] = useState('Hello World');
  // movies 상태 정의, 초기값은 null
  const [movies, setMovies] = useState(null);

  // useEffect 훅을 사용하여 componentDidMount 대체
  useEffect(() => {
    setTimeout(() => {
      // setMovies 함수를 사용하여 movies 상태 업데이트
      setMovies([
        {
          title: "Matrix",
          poster: 'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778581_STD.jpg',
        },
        {
          title: "Full Metal Jacket",
          poster: 'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778470_STD.jpg',
        },
        {
          title: "Oldboy",
          poster: 'https://cdn1.thr.com/sites/default/files/imagecache/768x433/2017/06/143289-1496932680-mm_2012_047_italy_57_-_h_2017.jpg',
        },
        {
          title: "Star wars",
          poster: 'https://cdn1.thr.com/sites/default/files/2017/06/143226-1496932903-mm_2012_047_italy_11_-_embed_2018.jpg',
        },
      ]);
    }, 2000);
  }, []); // 빈 배열을 두 번째 인수로 주어 componentDidMount와 동일한 효과

  // _renderMovies 메서드를 함수형 컴포넌트 내에 정의
  const renderMovies = () => {
    return movies.map((movie, index) => (
      // Movie 컴포넌트에 title과 poster props 전달
      <Movie title={movie.title} poster={movie.poster} key={index} />
    ));
  };

  console.log("render"); // 렌더링 시마다 출력

  return (
    <div className="App">
      {movies ? renderMovies() : 'Loading...'} {/* movies 상태가 null이 아니면 영화 목록 렌더링, 그렇지 않으면 'Loading...' 표시 */}
    </div>
  );
};

export default App;
```

## AJAX on React

* [Added CORS to Fetch Request @ src](https://github.com/nomadcoders/movie_app/commit/a5e045ecee069b2cb332892cb60061e8b5b22bd5)

----

AJAX 는 **Asynchrous JavaScript and XML** 의 약자이다. 그러나 XML 은 사용하지 않고 JSON 을 사용한다. AJAJ 로 바뀌어야 한다??? 다음은 fetch 함수를 이용하여 XHR (XML HTTP Request) 를 실행한 것이다.

다음은 smart component 예이다.

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.js:
import React, { Component } from 'react';
import Movie from './Movie';

class App extends Component {
  state = {
    movies: null
  };
  componentDidMount() {
    fetch('https://yts.ag/api/v2/list_movies.json?sort_by=rating')
      .then(potato => potato.json())
      .then(json => {
        this.setState({
          movies: json.data.movies
        });
      });
  }
  _renderMovies = () => {
    const movies = this.state.movies.map((movie, index) => (
      <Movie title={movie.title} poster={movie.medium_cover_image} key={index} />
    ));
    return movies;
  }
  render() {
    console.log("render");
    return (
      <div className="App">
        {this.state.movies ? this._renderMovies() : 'Loading...'}
      </div>
    );
  }
}

export default App;
```

다음은 functional component 예이다.

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.js:
import React, { useState, useEffect } from 'react';
import Movie from './Movie';

const App = () => {
  // movies 상태 정의, 초기값은 null
  const [movies, setMovies] = useState(null);

  // useEffect 훅을 사용하여 componentDidMount 대체
  useEffect(() => {
    fetch('https://yts.ag/api/v2/list_movies.json?sort_by=rating')
      .then(potato => potato.json())
      .then(json => {
        setMovies(json.data.movies);
      });
  }, []); // 빈 배열을 두 번째 인수로 주어 componentDidMount와 동일한 효과

  // _renderMovies 메서드를 함수형 컴포넌트 내에 정의
  const renderMovies = () => {
    return movies.map((movie, index) => (
      // Movie 컴포넌트에 title과 poster props 전달
      <Movie title={movie.title} poster={movie.medium_cover_image} key={index} />
    ));
  };

  console.log("render"); // 렌더링 시마다 출력

  return (
    <div className="App">
      {movies ? renderMovies() : 'Loading...'} {/* movies 상태가 null이 아니면 영화 목록 렌더링, 그렇지 않으면 'Loading...' 표시 */}
    </div>
  );
};

export default App;
```

## Promises

[javascript proxy](/js/README.md#promise) 를 참고하여 이해하자. 다음은 앞서 작성한 XHR 의 handler 를 추가한 것이다.

```js
////////////////////////////////////////////////////////////////////////////////
// Smart Component
// src/App.js:
import React, { Component } from 'react';
import Movie from './Movie';

class App extends Component {
  state = {
    movies: null
  };
  componentDidMount() {
    fetch('https://yts.ag/api/v2/list_movies.json?sort_by=rating')
      .then(potato => potato.json())
      .then(json => this.setState({ movies: json.data.movies }))
      .catch(err => console.log(err));
  }
  _renderMovies = () => {
    const movies = this.state.movies.map((movie, index) => (
      <Movie title={movie.title} poster={movie.medium_cover_image} key={index} />
    ));
    return movies;
  }
  render() {
    console.log("render");
    return (
      <div className="App">
        {this.state.movies ? this._renderMovies() : 'Loading...'}
      </div>
    );
  }
}

export default App;

////////////////////////////////////////////////////////////////////////////////
// Functional Component
// src/App.js:
import React, { useState, useEffect } from 'react';
import Movie from './Movie';

const App = () => {
  // movies 상태 정의, 초기값은 null
  const [movies, setMovies] = useState(null);

  // useEffect 훅을 사용하여 componentDidMount 대체
  useEffect(() => {
    fetch('https://yts.ag/api/v2/list_movies.json?sort_by=rating')
      .then(potato => potato.json())
      .then(json => setMovies(json.data.movies))
      .catch(err => console.log(err)); // 에러 처리
  }, []); // 빈 배열을 두 번째 인수로 주어 componentDidMount와 동일한 효과

  // _renderMovies 메서드를 함수형 컴포넌트 내에 정의
  const renderMovies = () => {
    return movies.map((movie, index) => (
      // Movie 컴포넌트에 title과 poster props 전달
      <Movie title={movie.title} poster={movie.medium_cover_image} key={index} />
    ));
  };

  console.log("render"); // 렌더링 시마다 출력

  return (
    <div className="App">
      {movies ? renderMovies() : 'Loading...'} {/* movies 상태가 null이 아니면 영화 목록 렌더링, 그렇지 않으면 'Loading...' 표시 */}
    </div>
  );
};

export default App;
```

그러나 CORS 설정이 되어 있지 않아서 error 가 발생한다. 다음과 같이 proxy 를 설정하면 해결할 수 있다.

```js
////////////////////////////////////////////////////////////////////////////////
// Smart Component
// src/App.js:
class App extends Component {
  state = {};
  componentDidMount() {
    const proxyurl = "https://cors-anywhere.herokuapp.com/";
    const url = 'https://yts.ag/api/v2/list_movies.json?sort_by=rating'; // site that doesn’t send Access-Control-*
    fetch(proxyurl + url)
    .then((resp) => resp.json())
    .then(json => console.log(json))
    .catch(err => console.log(err));
  }
  _renderMovies = () => {
    const movies = this.state.movies.map((movie, index) => {
      return <Movie title={movie.title} poster={movie.poster} key={index} />
    })
    return movies;
  }
  render() {
    console.log("render");
    return (
      <div className="App">
        {this.state.movies ? this._renderMovies() : 'Loading...'}
      </div>
    );
  }
}

////////////////////////////////////////////////////////////////////////////////
// Functional Component
// src/App.js:
import React, { useState, useEffect } from 'react';
import Movie from './Movie';

const App = () => {
  // movies 상태 정의, 초기값은 null
  const [movies, setMovies] = useState(null);

  // useEffect 훅을 사용하여 componentDidMount 대체
  useEffect(() => {
    const proxyurl = "https://cors-anywhere.herokuapp.com/";
    const url = 'https://yts.ag/api/v2/list_movies.json?sort_by=rating'; // site that doesn’t send Access-Control-*
    fetch(proxyurl + url)
      .then((resp) => resp.json())
      .then(json => {
        console.log(json); // 데이터 확인
        setMovies(json.data.movies); // movies 상태 업데이트
      })
      .catch(err => console.log(err)); // 에러 처리
  }, []); // 빈 배열을 두 번째 인수로 주어 componentDidMount와 동일한 효과

  // _renderMovies 메서드를 함수형 컴포넌트 내에 정의
  const renderMovies = () => {
    return movies.map((movie, index) => (
      // Movie 컴포넌트에 title과 poster props 전달
      <Movie title={movie.title} poster={movie.poster} key={index} />
    ));
  };

  console.log("render"); // 렌더링 시마다 출력

  return (
    <div className="App">
      {movies ? renderMovies() : 'Loading...'} {/* movies 상태가 null이 아니면 영화 목록 렌더링, 그렇지 않으면 'Loading...' 표시 */}
    </div>
  );
};

export default App;
```

그러나 promise 는 `then()` 을 무수히 만들어 낸다. 이것을 **Callback Hell** 이라고 한다. `Async Await` 을 이용하면 **Callback Hell** 을 탈출할 수 있다.

## Async Await

[JavaScript async](/js/README.md#async) 를 참고하여 이해하자. promise 는 then() 의 남용으로 Call Back 함수가 많아져서 code 의 readability 를 떨어뜨린다. `async, await` 을 이용하면 call back functions 을 줄일 수 있고 code 의 readability 를 끌어 올릴 수 있다.

`async` 로 function 을 선언하면 function 안에서 `await` 로 기다릴 수 있다. `await` 로 기다리는 것은 `promise` 이다.

```js
////////////////////////////////////////////////////////////////////////////////
// Smart Component
// src/App.js:
class App extends Component {
  state = {};
  componentDidMount() {
    this._getMovies();
  }
  _renderMovies = () => {
    const movies = this.state.movies.map((movie, index) => {
      return <Movie title={movie.title} poster={movie.large_cover_image} key={index} />
    })
    return movies;
  }
  _getMovies = async () => {
    const movies = await this._callApi();
    this.setState({
      movies
    });
  }
  _callApi = () => {
    const proxyurl = "https://cors-anywhere.herokuapp.com/";
    const url = 'https://yts.ag/api/v2/list_movies.json?sort_by=rating'; // site that doesn’t send Access-Control-*
    return fetch(proxyurl + url)
    .then((resp) => resp.json())
    .then(json => json.data.movies)
    .catch(err => console.log(err));
  }
  render() {
    console.log("render");
    return (
      <div className="App">
        {this.state.movies ? this._renderMovies() : 'Loading...'}
      </div>
    );
  }
}

////////////////////////////////////////////////////////////////////////////////
// Functional Component
// src/App.js:
import React, { useState, useEffect } from 'react';
import Movie from './Movie';

const App = () => {
  // movies 상태 정의, 초기값은 null
  const [movies, setMovies] = useState(null);

  // useEffect 훅을 사용하여 componentDidMount 대체
  useEffect(() => {
    getMovies();
  }, []); // 빈 배열을 두 번째 인수로 주어 componentDidMount와 동일한 효과

  // 영화 데이터를 가져오는 비동기 함수 정의
  const getMovies = async () => {
    const movies = await callApi();
    setMovies(movies); // movies 상태 업데이트
  }

  // API 호출 함수 정의
  const callApi = () => {
    const proxyurl = "https://cors-anywhere.herokuapp.com/";
    const url = 'https://yts.ag/api/v2/list_movies.json?sort_by=rating'; // site that doesn’t send Access-Control-*
    return fetch(proxyurl + url)
      .then((resp) => resp.json())
      .then(json => json.data.movies)
      .catch(err => console.log(err)); // 에러 처리
  }

  // renderMovies 함수 정의
  const renderMovies = () => {
    return movies.map((movie, index) => (
      // Movie 컴포넌트에 title과 poster props 전달
      <Movie title={movie.title} poster={movie.large_cover_image} key={index} />
    ));
  };

  console.log("render"); // 렌더링 시마다 출력

  return (
    <div className="App">
      {movies ? renderMovies() : 'Loading...'} {/* movies 상태가 null이 아니면 영화 목록 렌더링, 그렇지 않으면 'Loading...' 표시 */}
    </div>
  );
};

export default App;
```

## Updating Movie

이제 XHR 을 통해 얻어온 json 데이터를 화면에서 업데이트해 보자.

```js
////////////////////////////////////////////////////////////////////////////////
// Smart Component
// src/App.js:
class App extends Component {
  state = {};
  componentDidMount() {
    this._getMovies();
  }
  _renderMovies = () => {
    const movies = this.state.movies.map((movie, index) => {
      return <Movie title={movie.title_english} 
      poster={movie.medium_cover_image}       
      key={index} 
      genres={movie.generes}
      synopsis={movie.synopsis}
      />
    })
    return movies;
  }
  _getMovies = async () => {
    const movies = await this._callApi();
    this.setState({
      movies
    });
  }
  _callApi = () => {
    const proxyurl = "https://cors-anywhere.herokuapp.com/";
    const url = 'https://yts.ag/api/v2/list_movies.json?sort_by=rating'; // site that doesn’t send Access-Control-*
    return fetch(proxyurl + url)
    .then((resp) => resp.json())
    .then(json => json.data.movies)
    .catch(err => console.log(err));
  }
  render() {
    console.log("render");
    return (
      <div className="App">
        {this.state.movies ? this._renderMovies() : 'Loading...'}
      </div>
    );
  }
}

////////////////////////////////////////////////////////////////////////////////
// Smart Component
// src/Movie.js:
import React, { Component } from 'react';
import PropTypes from 'prop-types';
import './Movie.css';

// Movie 컴포넌트는 MoviePoster 컴포넌트를 렌더링합니다.
class Movie extends Component {
  render() {
    const { title, poster, genres, synopsis } = this.props;
    return (
      <div className="Movie">
        <div className="Movie__Columns">
          <MoviePoster poster={poster} alt={title} />
        </div>
        <div className="Movie__Columns">
          <h1>{title}</h1>
          <div className="Movie__Genre">
            {genres.map((genre, index) => (
              <MovieGenre genre={genre} key={index} />
            ))}
          </div>
          <p className="Movie__Synopsis">{synopsis}</p>
        </div>
      </div>
    );
  }
}

class MoviePoster extends Component {
  render() {
    const { poster, alt } = this.props;
    return <img src={poster} alt={alt} title={alt} className="Movie__Poster" />;
  }
}

class MovieGenre extends Component {
  render() {
    const { genre } = this.props;
    return <span className="Movie__Genre">{genre}</span>;
  }
}

Movie.propTypes = {
  title: PropTypes.string.isRequired,
  poster: PropTypes.string.isRequired,
  genres: PropTypes.arrayOf(PropTypes.string).isRequired,
  synopsis: PropTypes.string.isRequired,
};

MoviePoster.propTypes = {
  poster: PropTypes.string.isRequired,
  alt: PropTypes.string.isRequired,
};

MovieGenre.propTypes = {
  genre: PropTypes.string.isRequired,
};

export default Movie;

////////////////////////////////////////////////////////////////////////////////
// Functional Component
// src/App.js:
import React, { useState, useEffect } from 'react';
import Movie from './Movie';

const App = () => {
  // movies 상태 정의, 초기값은 null
  const [movies, setMovies] = useState(null);

  // useEffect 훅을 사용하여 componentDidMount 대체
  useEffect(() => {
    getMovies();
  }, []); // 빈 배열을 두 번째 인수로 주어 componentDidMount와 동일한 효과

  // 영화 데이터를 가져오는 비동기 함수 정의
  const getMovies = async () => {
    const movies = await callApi();
    setMovies(movies); // movies 상태 업데이트
  }

  // API 호출 함수 정의
  const callApi = () => {
    const proxyurl = "https://cors-anywhere.herokuapp.com/";
    const url = 'https://yts.ag/api/v2/list_movies.json?sort_by=rating'; // site that doesn’t send Access-Control-*
    return fetch(proxyurl + url)
      .then((resp) => resp.json())
      .then(json => json.data.movies)
      .catch(err => console.log(err)); // 에러 처리
  }

  // renderMovies 함수 정의
  const renderMovies = () => {
    return movies.map((movie, index) => (
      // Movie 컴포넌트에 title, poster, genres, synopsis props 전달
      <Movie 
        title={movie.title_english} 
        poster={movie.medium_cover_image}       
        key={index} 
        genres={movie.genres}
        synopsis={movie.synopsis}
      />
    ));
  };

  console.log("render"); // 렌더링 시마다 출력

  return (
    <div className="App">
      {movies ? renderMovies() : 'Loading...'} {/* movies 상태가 null이 아니면 영화 목록 렌더링, 그렇지 않으면 'Loading...' 표시 */}
    </div>
  );
};

export default App;

////////////////////////////////////////////////////////////////////////////////
// Functional Component
// src/Movie.js:
import React from 'react';
import PropTypes from 'prop-types';
import './Movie.css';

// Movie 컴포넌트는 MoviePoster 컴포넌트를 렌더링합니다.
function Movie({ title, poster, genres, synopsis }) {
  return (
    <div className="Movie">
      <div className="Movie__Columns">
        <MoviePoster poster={poster} alt={title} />
      </div>
      <div className="Movie__Columns">
        <h1>{title}</h1>
        <div className="Movie__Genre">
          {genres.map((genre, index) => <MovieGenre genre={genre} key={index} />)}
        </div>
        <p className="Movie__Synopsis">
          {synopsis}
        </p>
      </div>
    </div>
  );
}

function MoviePoster({ poster, alt }) {
  return (
    <img src={poster} alt={alt} title={alt} className="Movie__Poster" />
  );
}

function MovieGenre({ genre }) {
  return (
    <span className="Movie__Genre">{genre}</span>
  );
}

Movie.propTypes = {
  title: PropTypes.string.isRequired,
  poster: PropTypes.string.isRequired,
  genres: PropTypes.arrayOf(PropTypes.string).isRequired,
  synopsis: PropTypes.string.isRequired,
}

MoviePoster.propTypes = {
  poster: PropTypes.string.isRequired,
  alt: PropTypes.string.isRequired,
}

MovieGenre.propTypes = {
  genre: PropTypes.string.isRequired,
}

export default Movie;
```

## CSS for Movie

[react.js fundamentals src 2019 update](https://github.com/nomadcoders/movie_app_2019)

[kakao-clone-v2 | github](git@github.com:nomadcoders/kakao-clone-v2.git) 를 통해 css 를 더욱 배울 수 있다.

## Building for Production

지금까지 제작한 react.js app 을 `github` 에 publishing 해보자.

`npm run build` 혹은 `yarn build` 를 수행하면 `build` 디렉토리가 만들어진다. `build` 에는 압축된 static files 들이 만들어 진다.

`gh-pages` module 을 추가해 보자.

```bash
$ yarn add --dev gh-pages
```

`package.json` 의 "`homepage, predploy, deploy`" 을 다음과 같이 수정한다.

```js
  ...
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject",
    "predeploy": "npm run build",
    "deploy": "gh-pages -d build -r git@github.com:iamslash/examplesofweb.git"
  },
  ...
  "homepage": "https://iamslash.github.io/examplesofweb",
  "devDependencies": {
    "gh-pages": "^2.2.0"
  }
  ...
```

이제 배포해 본다.

```bash
$ yarn run deploy
```

# Advanced

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

## rendering Sequences

Component 가 rendering 되는 경우들을 생각해 보자. 

먼저 부모 Component 가 rendering 될때 자식 Component 의 render function 이 props 와 함께 호출되는 경우가 있다.

또한  Component 의 user event 혹은 timer event 에 의해 dispatch function 이 호출된다. reducer 는 변경된 state 를 리턴한다. 그리고 그 component 의 render function 이 호출된다. redner function 에서 props 를 통해 state 를 접근할 수 있다.

## Smart vs Dumb

* [Loading states + Smart vs Dumb Components @ src](https://github.com/nomadcoders/movie_app/commit/9cc9cf90d3c21dfea9c04c455f59aab7440018c4)

-------

Smart component 는 state 이 있는 component 이다. Dumb component 는 state 이 없는 component 이다. props 만 있다.
Dumb component 를 stateless 혹은 functional component 라고도 한다. Dumb component 는 state 가 필요없을 때 간결한 코드를 만들기 위해 사용한다.
그러나 state 가 없기도 하고 componentDidMount, componentWillMount 와 같은 함수들을 사용할 수 없다.

다음은 `Movie, MoviePoster` component 를 Dumb component 로 수정한 것이다.

```js
function Movie({title, poster}) {
  return (
    <div>
      <MoviePoster poster={poster}/>
      <h1>{title}</h1>
    </div>
  );
}
Movie.propTypes = {
  title: PropTypes.string.isRequired,
  poster: PropTypes.string.isRequired
}

// class MoviePoster extends Component {
//   static propTypes = {
//     poster: PropTypes.string.isRequired
//   }
//   render() {
//     return (
//       <img src={this.props.poster} alt="Movie Poster"/>
//     )
//   }
// }
function MoviePoster({poster}) {
  return (
    <img src={poster} alt="Movie Poster"/>
  );
}
MoviePoster.propTypes = {
  poster: PropTypes.string.isRequired
}
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

## Ant Design

* [Ant Design](https://ant.design/components/overview/)
  * [src](https://github.com/ant-design/ant-design)

-----

React ui library

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

## Don't Use `useEffect`

- [You Might Not Need an Effect | react.js](https://react.dev/learn/you-might-not-need-an-effect)

# Architectures

* [Optimize React apps using a multi-layered structure](https://blog.logrocket.com/optimize-react-apps-using-a-multi-layered-structure/)
