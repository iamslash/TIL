- [Abstract](#abstract)
- [Materials](#materials)
- [Templates](#templates)
- [Tech Stacks](#tech-stacks)
- [Movie Tutorial](#movie-tutorial)
  - [Overview](#overview)
  - [Debugging tools](#debugging-tools)
  - [Create React App](#create-react-app)
  - [npx vs npm](#npx-vs-npm)
  - [Create React Components with JSX](#create-react-components-with-jsx)
  - [Propagate Props](#propagate-props)
  - [Show The List With `.map()`](#show-the-list-with-map)
  - [Component Lifecycle](#component-lifecycle)
  - [Thinking In React Component State](#thinking-in-react-component-state)
  - [Practicing `this.setState`](#practicing-thissetstate)
  - [`setState` Caveats](#setstate-caveats)
  - [Loading States](#loading-states)
  - [AJAX on React](#ajax-on-react)
  - [CORS](#cors)
  - [Async Await](#async-await)
  - [Updating Movie](#updating-movie)
  - [CSS for Movie](#css-for-movie)
  - [Server State Management (react-query)](#server-state-management-react-query)
  - [Form Input](#form-input)
  - [Optimize Callback Definition With `useCallback()`](#optimize-callback-definition-with-usecallback)
  - [Form Validation](#form-validation)
  - [Form Management With react-hook-form, zod](#form-management-with-react-hook-form-zod)
  - [Routes](#routes)
  - [Handling Errors](#handling-errors)
  - [API Mocking (msw)](#api-mocking-msw)
  - [Profiles](#profiles)
  - [Client State Management (Zustand)](#client-state-management-zustand)
  - [Unit Test](#unit-test)
  - [E2E Test](#e2e-test)
  - [Directory Structures](#directory-structures)
  - [Build, Deploy To GitHub Pages](#build-deploy-to-github-pages)
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

----

# Abstract

최신 [Tech Stack](#tech-stacks) 을 참고해서 react.js tutorial 을 업데이트한다. react.js 의 문서는 완성도가 높다. 모두 읽어봐야 한다. [https://reactjs.org/docs/getting-started.html](https://reactjs.org/docs/getting-started.html) 

react.js 는 view library 이다. redux 는 state management library 이다. props 는 function parameter 와 유사하다. immutable 이다. state 는 local variable of function 과 같다. mutable 이다. [[React] Props와  State의 차이](https://singa-korean.tistory.com/37) 참고.

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

# Templates

* [gogo-react](https://gogo-react-docs.coloredstrategies.com/docs/gettingstarted/introduction)
* [react-admin](https://github.com/marmelab/react-admin)
  * An enterprise-class UI design language and React UI library
* [ant-design](https://github.com/ant-design/ant-design)
* [coreui](https://coreui.io/react/)

# Tech Stacks

- Essential
    - react
    - react-dom
    - react-router, react-router-dom
    - typescript
- Formatting
    - eslint (airbnb)
    - prettier
- Styling
    - @emotion/react
    - @emotion/styled
    - MUI (Material UI)
- Http Client
    - axios
- Client State Management
    - Zustand 
    - Jotai
    - Redux, Redux-saga
    - Recoil
- Server State Management
    - @tanstack/react-query
- Form State Management & Validation
    - react-hook-form
    - zod
- Unit & Integration testing
    - jest
    - @testing-library/jest-dom
    - @testing-library/react
    - @testing-library/user-event
- Api Mocking
    - msw
- Authentication
    - @okta/okta-auth-js
    - @okta/okta-react
- e2e testing
    - cypress

# Movie Tutorial

## Overview

[movie_app](https://github.com/nomadcoders/movie_app/tree) 를 from the scratch 해보자.

## Debugging tools

- [React Developer Tools | chromewebstore](https://chromewebstore.google.com/detail/react-developer-tools/fmkadmapgofadopljbjfkapdkoienihi)
  - Components
  - Profiler

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
$ create-react-app movie-app --template typescript

# 생성한 애플리케이션 디렉토리로 이동
$ cd movie-app

# 애플리케이션 실행
$ yarn start

$ tree movie-app -l 2 --gitignore node_module
movie-app
├── README.md
├── package-lock.json
├── package.json
├── public
│   ├── favicon.ico
│   ├── index.html
│   ├── logo192.png
│   ├── logo512.png
│   ├── manifest.json
│   └── robots.txt
├── src
│   ├── App.css
│   ├── App.test.tsx
│   ├── App.tsx
│   ├── index.css
│   ├── index.tsx
│   ├── logo.svg
│   ├── react-app-env.d.ts
│   ├── reportWebVitals.ts
│   └── setupTests.ts
└── tsconfig.json
```

## npx vs npm

- npm (Node Package Manager)
  - 주된 기능: Node.js 패키지를 설치, 관리, 삭제하는 도구.
  - 사용법: 일반적으로 `npm install <패키지명>` 명령어를 사용하여 패키지를 설치합니다.
  - 설치 위치: 패키지를 전역(-g 옵션) 또는 프로젝트 로컬(node_modules 디렉토리)에 설치할 수 있습니다.
  - 스크립트 실행: 프로젝트 내의 package.json 파일에 정의된 스크립트를 실행하는 데 사용됩니다. 예를 들어, `npm run start`는 package.json에 정의된 start 스크립트를 실행합니다.

- npx (Node Package Execute)
  - 주된 기능: Node.js 패키지를 일시적으로 실행하는 도구.
  - 사용법: `npx <명령어>` 형식으로 사용하며, 패키지가 설치되어 있지 않으면 자동으로 설치한 후 실행합니다.
  - 즉시 실행: npx는 명령어를 실행할 때마다 필요한 패키지를 일시적으로 다운로드하고 실행합니다. 따라서 특정 패키지를 한 번만 사용해야 할 때 유용합니다.
  - 편리함: 전역 설치 없이도 명령어를 실행할 수 있기 때문에 환경 설정이 간편해집니다.

- 사용 예시
  - npm: `npm install -g typescript`로 TypeScript를 전역 설치하고 tsc 명령어를 사용합니다.
  - npx: `npx create-react-app my-app`으로 React 앱을 생성합니다. 이 경우 `create-react-app` 패키지가 자동으로 설치되고 실행된 후, 설치된 패키지는 삭제됩니다.

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
// src/index.tsx:
import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import './index.css';

ReactDOM.render(<App />, document.getElementById('root') as HTMLElement);

////////////////////////////////////////////////////////////////////////////////
// src/App.tsx:
import React from 'react';
import './App.css';
import Movie from './Movie';

const App: React.FC = () => {
  return (
    <div className="App">
      <Movie />
      <Movie />
    </div>
  );
};

export default App;

////////////////////////////////////////////////////////////////////////////////
// src/Movie.tsx:
import React from 'react';
import './Movie.css';

const Movie: React.FC = () => {
  return (
    <div>
      <MoviePoster />
      <h1>Hello This is a movie</h1>
    </div>
  );
};

const MoviePoster: React.FC = () => {
  return (
    <img src='http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778581_STD.jpg' alt="Movie Poster" />
  );
};

export default Movie;
```

## Propagate Props

![](https://miro.medium.com/max/1540/0*NLC2HyJRjh0_3r0e.)

PropTypes를 사용하여 props가 필수인지 확인하는 것은 JavaScript에서 컴포넌트의 props 타입을 검사하는 일반적인 방법입니다. 그러나 TypeScript를 사용하면 더 강력하고 유지보수하기 쉬운 타입 시스템을 활용할 수 있습니다.

이 접근 방식의 장점:

- 정적 타입 검사: 컴파일 타임에 타입을 검사하여 런타임 오류를 줄일 수 있습니다.
- IDE 지원: TypeScript는 많은 IDE에서 더 나은 자동 완성, 리팩토링 지원, 코드 네비게이션을 제공합니다.
- 유지보수 용이성: 타입이 명시적으로 정의되어 있어 코드의 의도를 더 쉽게 이해할 수 있습니다.

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.tsx:
import React from 'react';
import Movie from './Movie';

const movieTitles = [
  "Matrix",
  "Full Metal Jacket",
  "Oldboy",
  "Star wars"
];

const movieImages = [
  'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778581_STD.jpg',
  'http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778470_STD.jpg',
  'https://cdn1.thr.com/sites/default/files/imagecache/768x433/2017/06/143289-1496932680-mm_2012_047_italy_57_-_h_2017.jpg',
  'https://cdn1.thr.com/sites/default/files/2017/06/143226-1496932903-mm_2012_047_italy_11_-_embed_2018.jpg',
];

const App: React.FC = () => {
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
// src/Movie.tsx:
import React from 'react';
import './Movie.css';

interface MovieProps {
  title: string;
  poster: string;
}

interface MoviePosterProps {
  poster: string;
}

const Movie: React.FC<MovieProps> = ({ title, poster }) => {
  return (
    <div>
      <MoviePoster poster={poster} />
      <h1>{title}</h1>
    </div>
  );
};

const MoviePoster: React.FC<MoviePosterProps> = ({ poster }) => {
  return (
    <img src={poster} alt="Movie Poster" />
  );
};

export default Movie;
```

## Show The List With `.map()`

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.tsx:
import React from 'react';
import Movie from './Movie';

interface MovieType {
  title: string;
  poster: string;
}

const movies: MovieType[] = [
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
];

const App: React.FC = () => {
  return (
    <div className="App">
      {movies.map((movie, index) => (
        <Movie key={index} title={movie.title} poster={movie.poster} />
      ))}
    </div>
  );
};

export default App;
```

## Component Lifecycle

Functional Component Rendering and Lifecycle:

- Initial Render:
  - `render()`: Functional component의 본문이 실행됩니다.
  - `useEffect(() => { ... }, [])`: 이 hook은 마운트 후에 실행됩니다. 클래스형 컴포넌트의 componentDidMount에 해당합니다.
- Update:
  - `render()`: State나 props가 변경되면 컴포넌트가 다시 렌더링됩니다.
  - `useEffect(() => { ... })`: 이 hook은 디펜던시 배열이 변경될 때마다 실행됩니다. 클래스형 컴포넌트의 componentDidUpdate에 해당합니다.
  - `useEffect(() => { return () => { ... } }, [])`: 이 훅의 클린업 함수는 컴포넌트가 언마운트되기 직전에 실행됩니다. 클래스형 컴포넌트의 componentWillUnmount에 해당합니다.
- Unmount:
  - `useEffect(() => { return () => { ... } }, [])`: 컴포넌트가 언마운트될 때 클린업 함수가 호출됩니다.

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.tsx:
// Initial Render: render() -> useEffect(() => { /* componentDidMount */ }, [])
// Update: render() -> useEffect(() => { /* componentDidUpdate */ })
//
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
];

const App: React.FC = () => {
  useEffect(() => {
    // componentDidMount
    console.log("componentDidMount");

    return () => {
      // componentWillUnmount
      console.log("componentWillUnmount");
    };
  }, []);

  // render
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

## Thinking In React Component State

`App` 컴포넌트에 `state`를 선언하고 `useEffect` 훅에서 상태를 변경한다. 상태를 변경하기 위해 `setMovies` 함수를 호출하면 컴포넌트는 다시 렌더링된다. 이 코드는 상태를 변경하고 렌더링을 업데이트하는 과정을 보여준다. React의 `state`와 Redux의 `state`는 서로 다르며, 각각의 사용 목적과 상황에 맞게 사용해야 한다. [React State vs. Redux State: When and Why?](https://spin.atomicobject.com/2017/06/07/react-state-vs-redux-state/)에서 더 자세히 알아볼 수 있다.

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.tsx:
import React, { useState, useEffect } from 'react';
import Movie from './Movie';

interface MovieType {
  title: string;
  poster: string;
}

const initialMovies: MovieType[] = [
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
];

// 함수형 컴포넌트로 App 정의
const App: React.FC = () => {
  const [movies, setMovies] = useState<MovieType[]>([]);

  // useEffect 훅을 사용하여 componentDidMount 대체
  useEffect(() => {
    console.log("componentDidMount");
    setMovies(initialMovies); // state 변경
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

////////////////////////////////////////////////////////////////////////////////
// src/Movie.tsx:
import React from 'react';
import './Movie.css';

interface MovieProps {
  title: string;
  poster: string;
}

const Movie: React.FC<MovieProps> = ({ title, poster }) => {
  return (
    <div>
      <MoviePoster poster={poster} />
      <h1>{title}</h1>
    </div>
  );
};

interface MoviePosterProps {
  poster: string;
}

const MoviePoster: React.FC<MoviePosterProps> = ({ poster }) => {
  return (
    <img src={poster} alt="Movie Poster" />
  );
};

export default Movie;
```

## Practicing `this.setState`

App 컴포넌트의 `state`로 `title`과 `poster`를 관리하고, 일정 시간 후에 새로운 영화를 추가하여 상태를 변경합니다. `setMovies` 함수를 사용하여 이전 상태를 복사하고 새로운 영화를 추가합니다. 이를 통해 컴포넌트가 다시 렌더링되어 업데이트된 상태를 반영합니다. 이 방법은 무한 스크롤과 같은 기능을 구현할 때 유용합니다. `...prevMovies`를 사용하면 기존 배열을 펼쳐서 새로운 요소를 추가할 수 있습니다. 

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.tsx:
import React, { useState, useEffect } from 'react';
import Movie from './Movie';

interface MovieType {
  title: string;
  poster: string;
}

const App: React.FC = () => {
  // greeting 상태 정의
  const [greeting, setGreeting] = useState<string>('Hello World');
  // movies 상태 정의
  const [movies, setMovies] = useState<MovieType[]>([
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

## `setState` Caveats

* [React setState usage and gotchas](https://itnext.io/react-setstate-usage-and-gotchas-ac10b4e03d60)
  * [Boost your React with State Machines](https://www.freecodecamp.org/news/boost-your-react-with-state-machines-1e9641b0aa43/)

`useEffect`에서 `setState`를 호출할 때 주의사항:

- 의존성 배열 사용: `useEffect` 훅의 두 번째 인수로 의존성 배열을 전달하여 `componentDidMount`와 동일한 효과를 낼 수 있다. 만약 의존성 배열을 빈 배열로 지정하면, 해당 useEffect 훅은 컴포넌트가 마운트될 때 한 번만 실행된다.
- 클린업 함수 사용: `useEffect` 훅 내에서 반환되는 함수는 컴포넌트가 언마운트될 때 호출된다. 이를 통해 타이머나 이벤트 리스너 등을 정리할 수 있다. 예를 들어, setTimeout을 사용한 경우, 컴포넌트가 언마운트되기 전에 타이머를 정리하여 메모리 누수를 방지할 수 있다.
- 비동기 작업 주의: 비동기 작업이 완료된 후 `setState`를 호출하는 경우, 컴포넌트가 이미 언마운트된 상태일 수 있으므로, 이를 방지하기 위해 컴포넌트가 여전히 마운트된 상태인지 확인해야 한다. 이는 클린업 함수를 통해 해결할 수 있다.
- 불필요한 상태 업데이트 방지: `setState`를 호출하면 컴포넌트가 다시 렌더링되므로, 불필요한 상태 업데이트를 피해야 한다. 필요한 경우에만 상태를 업데이트하도록 조건을 설정할 수 있다.

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.tsx:
import React, { useState, useEffect } from 'react';
import Movie from './Movie';

interface MovieType {
  title: string;
  poster: string;
}

const App: React.FC = () => {
  // movies 상태 정의, 초기값은 빈 배열
  const [movies, setMovies] = useState<MovieType[]>([]);

  // useEffect 훅을 사용하여 componentDidMount 대체
  useEffect(() => {
    const timer = setTimeout(() => {
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

    // Cleanup function to clear the timeout if the component unmounts
    return () => clearTimeout(timer);
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

이 코드에서는 `useState`를 사용하여 `movies` 상태를 관리합니다. 초기값은 `null`로 설정되어 있으며, `useEffect`를 사용하여 컴포넌트가 마운트될 때 2초 후에 `movies` 상태를 업데이트합니다. 상태가 `null`인 동안에는 'Loading...' 메시지를 표시하고, 상태가 업데이트되면 영화 목록을 렌더링합니다.

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.js:
import React, { useState, useEffect } from 'react';
import Movie from './Movie';

interface MovieType {
  title: string;
  poster: string;
}

const App: React.FC = () => {
  // greeting 상태 정의, 초기값은 'Hello World'
  const [greeting, setGreeting] = useState<string>('Hello World');
  // movies 상태 정의, 초기값은 null
  const [movies, setMovies] = useState<MovieType[] | null>(null);

  // useEffect 훅을 사용하여 componentDidMount 대체
  useEffect(() => {
    const timer = setTimeout(() => {
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

    // Cleanup function to clear the timeout if the component unmounts
    return () => clearTimeout(timer);
  }, []); // 빈 배열을 두 번째 인수로 주어 componentDidMount와 동일한 효과

  // renderMovies 메서드를 함수형 컴포넌트 내에 정의
  const renderMovies = () => {
    return movies!.map((movie, index) => (
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

AJAX 는 **Asynchrous JavaScript and XML** 의 약자이다. 그러나 XML 은 사용하지 않고 JSON 을 사용한다. AJAJ 로 바뀌어야 한다??? 다음은 fetch 함수를 이용하여 XHR (XML HTTP Request) 를 실행한 것이다.

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.js:
import React, { useState, useEffect } from 'react';
import Movie from './Movie';

interface MovieType {
  title: string;
  medium_cover_image: string;
}

const App: React.FC = () => {
  // movies 상태 정의, 초기값은 null
  const [movies, setMovies] = useState<MovieType[] | null>(null);

  // useEffect 훅을 사용하여 componentDidMount 대체
  useEffect(() => {
    fetch('https://yts.mx/api/v2/list_movies.json?sort_by=rating')
      .then(response => response.json())
      .then(json => { setMovies(json.data.movies) })
      .catch(error => { console.error('Error fetching movies:', error) });
  }, []); // 빈 배열을 두 번째 인수로 주어 componentDidMount와 동일한 효과

  // renderMovies 메서드를 함수형 컴포넌트 내에 정의
  const renderMovies = () => {
    return movies!.map((movie, index) => (
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

Promise 는 [JavaScript Promise](/js/README.md#async-await) 를 참고하여 이해하자. 그러나 promise 는 `then()` 을 무수히 만들어 낸다. 이것을 **Callback Hell** 이라고 한다. `Async Await` 을 이용하면 **Callback Hell** 을 탈출할 수 있다.

## CORS

[CORS](/cors/README.md) 를 참고하자.

만약 CORS 설정이 되어 있지 않아서 error 가 발생한다면 다음과 같이 proxy 를 설정하여 해결한다. 그런데 잘 안된다. (2024.07.12)

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.tsx:
import React, { useState, useEffect } from 'react';
import Movie from './Movie';

interface MovieType {
  title: string;
  poster: string;
}

const App: React.FC = () => {
  // movies 상태 정의, 초기값은 null
  const [movies, setMovies] = useState<MovieType[] | null>(null);

  // useEffect 훅을 사용하여 componentDidMount 대체
  useEffect(() => {
    const proxyurl = "https://cors-anywhere.herokuapp.com/";
    const url = 'https://yts.mx/api/v2/list_movies.json?sort_by=rating'; // site that doesn’t send Access-Control-*
    fetch(proxyurl + url)
      .then((resp) => resp.json())
      .then(json => {
        console.log(json); // 데이터 확인
        setMovies(json.data.movies); // movies 상태 업데이트
      })
      .catch(err => console.error('Error fetching movies:', err)); // 에러 처리
  }, []); // 빈 배열을 두 번째 인수로 주어 componentDidMount와 동일한 효과

  // renderMovies 메서드를 함수형 컴포넌트 내에 정의
  const renderMovies = () => {
    return movies!.map((movie, index) => (
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

## Async Await

[JavaScript Async & Await](/js/README.md#async-await) 를 참고하여 이해하자. Promise 는 `then()` 의 남용으로 Call Back 함수가 많아져서 code 의 readability 를 떨어뜨린다. `async, await` 을 이용하면 call back functions 을 줄일 수 있고 code 의 readability 를 끌어 올릴 수 있다.

`async` 로 function 을 선언하면 function 안에서 `await` 로 기다릴 수 있다. `await` 로 기다리는 것은 `promise` 이다.

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.tsx:
import React, { useState, useEffect } from 'react';
import Movie from './Movie';

interface MovieType {
  title: string;
  large_cover_image: string;
}

const App: React.FC = () => {
  // movies 상태 정의, 초기값은 null
  const [movies, setMovies] = useState<MovieType[] | null>(null);

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
  const callApi = async (): Promise<MovieType[]> => {
    const url = 'https://yts.mx/api/v2/list_movies.json?sort_by=rating'; // site that doesn’t send Access-Control-*
    try {
      const response = await fetch(url);
      const json = await response.json();
      return json.data.movies;
    } catch (err) {
      console.error('Error fetching movies:', err);
      return [];
    }
  }

  // renderMovies 함수 정의
  const renderMovies = () => {
    return movies!.map((movie, index) => (
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
// src/App.tsx:
import React, { useState, useEffect } from 'react';
import Movie from './Movie';

interface MovieType {
  title_english: string;
  medium_cover_image: string;
  genres: string[];
  synopsis: string;
}

const App: React.FC = () => {
  // movies 상태 정의, 초기값은 null
  const [movies, setMovies] = useState<MovieType[] | null>(null);

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
  const callApi = async (): Promise<MovieType[]> => {
    const url = 'https://yts.mx/api/v2/list_movies.json?sort_by=rating'; // site that doesn’t send Access-Control-*
    try {
      const response = await fetch(url);
      const json = await response.json();
      return json.data.movies;
    } catch (err) {
      console.error('Error fetching movies:', err);
      return [];
    }
  }

  // renderMovies 함수 정의
  const renderMovies = () => {
    return movies!.map((movie, index) => (
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
// src/Movie.tsx:
import React from 'react';
import './Movie.css';

interface MovieProps {
  title: string;
  poster: string;
  genres: string[];
  synopsis: string;
}

const Movie: React.FC<MovieProps> = ({ title, poster, genres, synopsis }) => {
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
};

interface MoviePosterProps {
  poster: string;
  alt: string;
}

const MoviePoster: React.FC<MoviePosterProps> = ({ poster, alt }) => {
  return (
    <img src={poster} alt={alt} title={alt} className="Movie__Poster" />
  );
};

interface MovieGenreProps {
  genre: string;
}

const MovieGenre: React.FC<MovieGenreProps> = ({ genre }) => {
  return (
    <span className="Movie__Genre">{genre}</span>
  );
};

export default Movie;
```

## CSS for Movie

[CSS](/css/README.md), [react.js fundamentals src 2019 update](https://github.com/nomadcoders/movie_app_2019), [kakao-clone-v2 | github](https://github.com/nomadcoders/kakao-clone-v2) 를 통해 css 를 더욱 배울 수 있다.

```css
.Movie{
    background-color:white;
    width:40%;
    display: flex;
    justify-content: space-between;
    align-items:flex-start;
    flex-wrap:wrap;
    margin-bottom:50px;
    text-overflow: ellipsis;
    padding:0 20px;
    box-shadow: 0 8px 38px rgba(133, 133, 133, 0.3), 0 5px 12px rgba(133, 133, 133,0.22);
}

.Movie__Column{
    width:30%;
    box-sizing:border-box;
    text-overflow: ellipsis;
}

.Movie__Column:last-child{
    padding:20px 0;
    width:60%;
}

.Movie h1{
    font-size:20px;
    font-weight: 600;
}

.Movie .Movie__Genres{
    display: flex;
    flex-wrap:wrap;
    margin-bottom:20px;
}

.Movie__Genres .Movie__Genre{
    margin-right:10px;
    color:#B4B5BD;
}

.Movie .Movie__Synopsis {
    text-overflow: ellipsis;
    color:#B4B5BD;
    overflow: hidden;
}

.Movie .Movie__Poster{
    max-width: 100%;
    position: relative;
    top:-20px;
    box-shadow: -10px 19px 38px rgba(83, 83, 83, 0.3), 10px 15px 12px rgba(80,80,80,0.22);
}

@media screen and (min-width:320px) and (max-width:667px){
    .Movie{
        width:100%;
    }
}

@media screen and (min-width:320px) and (max-width:667px) and (orientation: portrait){
    .Movie{
        width:100%;
        flex-direction: column;
    }
    .Movie__Poster{
        top:0;
        left:0;
        width:100%;
    }
    .Movie__Column{
        width:100%!important;
    }
}
```

## Server State Management (react-query)

- react-query 사용:
  - useQuery 훅을 사용하여 영화 데이터를 비동기적으로 가져옵니다. fetchMovies 함수는 fetch를 사용하여 데이터를 가져오고, useQuery는 이 함수를 호출하여 데이터를 캐싱하고 상태를 관리합니다.
- query key:
  - useQuery 훅의 첫 번째 인수로 `['movies', downloadUrl]` 배열을 사용합니다. 이 배열은 쿼리의 키 역할을 하며, downloadUrl이 변경될 때마다 새로운 쿼리를 트리거합니다.
- 조건부 실행:
  - enabled 옵션을 사용하여 downloadUrl이 설정되었을 때만 쿼리가 실행되도록 합니다.
- 상태 관리:
  - isLoading과 error 상태를 사용하여 로딩 상태와 에러 상태를 처리합니다.
- MovieForm 컴포넌트:
  - react-hook-form과 Zod를 사용하여 폼 데이터를 관리하고 URL 검증을 수행합니다. 폼 제출 시 onSubmit 함수를 호출하여 부모 컴포넌트에 URL을 전달합니다.

```js
////////////////////////////////////////////////////////////////////////////////
// src/index.tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import './index.css';
import App from './App';

const queryClient = new QueryClient();

const root = ReactDOM.createRoot(
    document.getElementById('root') as HTMLElement
);
root.render(
    <React.StrictMode>
        <QueryClientProvider client={queryClient}>
            <App />
        </QueryClientProvider>
    </React.StrictMode>
);

////////////////////////////////////////////////////////////////////////////////
// src/App.tsx
import React from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import Movie from './Movie';

interface MovieType {
    title_english: string;
    medium_cover_image: string;
    genres: string[];
    synopsis: string;
}

const fetchMovies = async (url: string): Promise<MovieType[]> => {
    const response = await axios.get(url);
    // console.log(response);
    return response.data.data.movies;
};

const App: React.FC = () => {
    const downloadUrl = 'https://yts.mx/api/v2/list_movies.json?sort_by=rating';

    const { data: movies, error, isLoading } = useQuery({
        queryKey: ['movies', downloadUrl],
        queryFn: () => fetchMovies(downloadUrl),
        enabled: !!downloadUrl, // Only run the query if downloadUrl is not empty
    });

    const renderMovies = () => {
        return movies?.map((movie, index) => (
            <Movie
                title={movie.title_english}
                poster={movie.medium_cover_image}
                key={index}
                genres={movie.genres}
                synopsis={movie.synopsis}
            />
        ));
    };

    if (isLoading) {
        return <div>Loading...</div>;
    }
    
    if (error) {
        return <div>Error fetching movies</div>;
    }

    return (
        <div className="App">
            <div style={{ marginTop: '20px' }}>
                {movies ? renderMovies() : 'No movies available'}
            </div>
        </div>
    );
};

export default App;
```

## Form Input

`url` 을 form 으로 입력받고 영화를 다운로드 하자.

- `preventDefault()`가 필요한 이유:
  - 페이지 새로 고침 방지: 폼 제출 시 페이지가 새로 고침되는 것을 방지합니다. SPA에서는 서버로 전송하지 않고, 클라이언트 측에서 데이터를 처리하고 필요한 작업을 수행합니다.
  - 비동기 처리: 폼 제출을 비동기적으로 처리하여 서버와의 통신을 통해 페이지를 다시 로드하지 않고 데이터를 업데이트합니다.
  - 커스텀 동작: 폼이나 다른 요소에 대해 기본 동작 대신 사용자 정의 동작을 지정할 수 있습니다. 예를 들어, 클릭 이벤트에서 링크의 기본 동작을 막고 다른 작업을 수행할 수 있습니다.

```js
////////////////////////////////////////////////////////////////////////////////
// App.tsx
import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import Movie from './Movie';
import MovieForm from './MovieForm';

interface MovieType {
    title_english: string;
    medium_cover_image: string;
    genres: string[];
    synopsis: string;
}

const fetchMovies = async (url: string): Promise<MovieType[]> => {
    const response = await axios.get(url);
    return response.data.data.movies;
};

const App: React.FC = () => {
    const [downloadUrl, setDownloadUrl] = useState<string>('');

    const { data: movies, error, isLoading, refetch } = useQuery({
        queryKey: ['movies', downloadUrl],
        queryFn: () => fetchMovies(downloadUrl),
        enabled: !!downloadUrl, // Do not run the query automatically
        retry: false, // Don't retry automatically
        refetchOnWindowFocus: false, // Don't refetch on window focus
    });

    const handleUrlSubmit = (url: string) => {
        console.log('handleUrlSubmit: ', url);
        setDownloadUrl(url);
        refetch();
    };

    const renderMovies = () => {
        return movies?.map((movie, index) => (
            <Movie
                title={movie.title_english}
                poster={movie.medium_cover_image}
                key={index}
                genres={movie.genres}
                synopsis={movie.synopsis}
            />
        ));
    };

    return (
        <div className="App">
            <div>
                <MovieForm onSubmit={handleUrlSubmit} />
            </div>
            <div style={{ marginTop: '20px' }}>
                {isLoading && <div>Loading...</div>}
                {error && <div>Error fetching movies</div>}
                {movies ? renderMovies() : 'No movies available'}
            </div>
        </div>
    );
};

export default App;

////////////////////////////////////////////////////////////////////////////////
// MovieForm.tsx
import React, { useState } from 'react';

interface MovieFormProps {
    url: string;
    onSubmit: (url: string) => void;
}

const MovieForm: React.FC<MovieFormProps> = ({ url, onSubmit }) => {
    const [inputUrl, setInputUrl] = useState<string>(url);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setInputUrl(e.target.value);
    };

    const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        onSubmit(inputUrl);
    };

    return (
        <form onSubmit={handleSubmit}>
            <input
                type="text"
                placeholder="Enter download URL"
                value={inputUrl}
                onChange={handleChange}
            />
            <button type="submit">Download Movies</button>
        </form>
    );
};

export default MovieForm;
```

## Optimize Callback Definition With `useCallback()`

`useCallback()` 을 사용하면 `handleChange`와 `handleSubmit` 함수가 렌더링될 때마다 새로 정의되지 않으므로, 성능을 최적화할 수 있습니다. `handleChange`와 `handleSubmit` 함수를 `useCallback` 훅으로 감싸서 메모이제이션합니다. 이는 함수가 의존성 배열의 값이 변경되지 않는 한 동일한 참조를 유지하게 합니다.

- 의존성 배열:
  - `handleChange` 함수는 의존성이 없으므로 빈 배열을 사용합니다.
  - `handleSubmit` 함수는 `inputUrl`과 `onSubmit`에 의존합니다. 따라서 이 값들이 변경될 때만 새로운 함수가 생성됩니다.
- 성능 최적화의 이점:
  - 함수 재사용: `useCallback`을 사용하여 함수를 메모이제이션하면, 컴포넌트가 다시 렌더링될 때마다 새로운 함수를 생성하지 않고 동일한 함수를 재사용합니다. 이는 특히 함수가 자주 재정의되는 경우 성능에 긍정적인 영향을 미칩니다.
  - 불필요한 렌더링 방지: 메모이제이션된 함수는 자식 컴포넌트에 전달될 때 참조가 변경되지 않으므로, 자식 컴포넌트가 불필요하게 다시 렌더링되는 것을 방지할 수 있습니다.

```js
////////////////////////////////////////////////////////////////////////////////
// MovieForm.tsx
import React, { useState, useCallback } from 'react';

interface MovieFormProps {
    url: string;
    onSubmit: (url: string) => void;
}

const MovieForm: React.FC<MovieFormProps> = ({ url, onSubmit }) => {
    const [inputUrl, setInputUrl] = useState<string>(url);

    const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        setInputUrl(e.target.value);
    }, []);

    const handleSubmit = useCallback((e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        onSubmit(inputUrl);
    }, [inputUrl, onSubmit]);

    return (
        <form onSubmit={handleSubmit}>
            <input
                type="text"
                placeholder="Enter download URL"
                value={inputUrl}
                onChange={handleChange}
            />
            <button type="submit">Download Movies</button>
        </form>
    );
};

export default MovieForm;
```

## Form Validation

- URL 검증 함수 추가:
  - isValidUrl 함수는 정규 표현식을 사용하여 입력된 URL이 유효한지 검증합니다. https, http, 및 chrome로 시작하는 URL을 허용합니다.
- 상태 관리:
  - inputUrl 상태를 관리하여 사용자가 입력한 URL을 저장합니다.
  - errorMessage 상태를 추가하여 URL이 유효하지 않은 경우 에러 메시지를 표시합니다.
- handleSubmit 함수 수정:
  - handleSubmit 함수는 URL이 유효한지 검증하고, 유효한 경우에만 onSubmit 함수를 호출합니다.
  - 유효하지 않은 URL이 입력되면 에러 메시지를 설정하여 사용자에게 표시합니다.
- 에러 메시지 표시:
  - 유효하지 않은 URL이 입력된 경우, 에러 메시지를 빨간색 텍스트로 표시합니다.

```js
////////////////////////////////////////////////////////////////////////////////
// MovieForm
import React, { useState } from 'react';

interface MovieFormProps {
    onSubmit: (url: string) => void;
}

const MovieForm: React.FC<MovieFormProps> = ({ onSubmit }) => {
    const [inputUrl, setInputUrl] = useState<string>('https://yts.mx/api/v2/list_movies.json?sort_by=rating');
    const [errorMessage, setErrorMessage] = useState<string>('');

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setInputUrl(e.target.value);
    };

    const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        if (isValidUrl(inputUrl)) {
            setErrorMessage('');
            onSubmit(inputUrl);
        } else {
            setErrorMessage('Please enter a valid URL.');
        }
    };

    const isValidUrl = (url: string) => {
        const urlRegex = /^(https?|chrome):\/\/[^\s$.?#].[^\s]*$/;
        return urlRegex.test(url);
    };

    return (
        <form onSubmit={handleSubmit}>
            <input
                type="text"
                placeholder="Enter download URL"
                value={inputUrl}
                onChange={handleChange}
            />
            <button type="submit">Download Movies</button>
            {errorMessage && <p style={{ color: 'red' }}>{errorMessage}</p>}
        </form>
    );
};

export default MovieForm;
```

## Form Management With react-hook-form, zod

react-hook-form, zod 를 이용해 form management 를 간단히 구현한다.

```js
////////////////////////////////////////////////////////////////////////////////
// src/MovieForm.tsx
import React from 'react';
import { useForm, SubmitHandler } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';

const urlSchema = z.object({
    url: z.string().url({ message: "Please enter a valid URL" }),
});

type UrlFormInputs = z.infer<typeof urlSchema>;

interface MovieFormProps {
    onSubmit: (url: string) => void;
}

const MovieForm: React.FC<MovieFormProps> = ({ onSubmit }) => {
    const { register, handleSubmit, formState: { errors } } = useForm<UrlFormInputs>({
        resolver: zodResolver(urlSchema),
        defaultValues: {
            url: 'https://yts.mx/api/v2/list_movies.json?sort_by=rating',
        },
    });

    const onSubmitHandler: SubmitHandler<UrlFormInputs> = (data) => {
        onSubmit(data.url);
    };

    return (
        <form onSubmit={handleSubmit(onSubmitHandler)}>
            <input
                type="text"
                placeholder="Enter download URL"
                {...register('url')}
            />
            <button type="submit">Download Movies</button>
            {errors.url && <p style={{ color: 'red' }}>{errors.url.message}</p>}
        </form>
    );
};

export default MovieForm;
```

## Routes

## Handling Errors

## API Mocking (msw)

## Profiles

## Client State Management (Zustand)

## Unit Test

## E2E Test

## Directory Structures

* [Optimize React apps using a multi-layered structure](https://blog.logrocket.com/optimize-react-apps-using-a-multi-layered-structure/)

```py
public/
src/
	__mocks__/ # mocking dir
		api/
			[dir]/
				fixtures/ # dir to store json
				handlers/ # dir to handle api
	__tests__/ # test dir
	pages/ # pages dir
		[dir]/
			index.tsx 
	features/ # dir to be used in pages
		[dir]/
			api/ # dir to manage apis
			components/ # dir to manage components
			constants/ # dir to manage constants
			hooks/ # dir to manage hooks
			queries/ # dir to manage react-query's useQuery hooks
			mutations/ # dir to manage react-query's useMutation hooks
			types/ # dir to manage types
			utils/ # dir to manage utility functions
	openapi/ # dir to manage openapi
	shared/ # dir to manage project-wide shared items
		components/
		constants/
		hooks/
		utils/
		types/
```

- `public/`:
  - 정적 파일들을 저장하는 폴더입니다. 예를 들어, index.html, 이미지 파일 등이 여기에 포함될 수 있습니다.
- `src/`:
  - 소스 코드의 루트 디렉토리입니다.
- `__mocks__/`:
  - 테스트를 위한 mocking 데이터를 저장하는 폴더입니다.
- `fixtures/`: 테스트용 JSON 데이터를 저장합니다.
- `handlers/`: Mock API 처리 로직을 관리합니다.
- `__tests__/`:
  - 모든 테스트 파일을 저장하는 폴더입니다. 테스트 파일이 각 기능별로 분산되지 않고 한 곳에 모아져 있어 관리하기 쉽습니다.
- `pages/`:
  - 각 페이지별 컴포넌트를 관리하는 폴더입니다.
  - `[dir]/index.tsx`: 각 페이지의 진입점 파일입니다.
- `features/`:
  - 페이지에서 사용할 기능별 모듈을 관리하는 폴더입니다.
  - `api/`: API 호출 로직을 관리합니다.
  - `components/`: 해당 기능에서 사용하는 컴포넌트를 관리합니다.
  - `constants/`: 기능별 상수를 관리합니다.
  - `hooks/`: 기능별 훅을 관리합니다.
  - `queries/`: React Query의 useQuery 훅을 관리합니다.
  - `mutations/`: React Query의 useMutation 훅을 관리합니다.
  - `types/`: 타입스크립트 타입 정의를 관리합니다.
  - `utils/`: 유틸리티 함수들을 관리합니다.
- `openapi/`:
  - OpenAPI 스펙을 관리하는 폴더입니다. API 문서화 및 클라이언트 코드 생성을 위해 사용됩니다.
- `shared/`:
  - 프로젝트 전반적으로 사용되는 공통 모듈을 관리하는 폴더입니다.
  - `components/`: 공통 컴포넌트를 관리합니다.
  - `constants/`: 전역 상수를 관리합니다.
  - `hooks/`: 공통 훅을 관리합니다.
  - `utils/`: 공통 유틸리티 함수들을 관리합니다.
  - `types/`: 전역 타입 정의를 관리합니다.

## Build, Deploy To GitHub Pages

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
