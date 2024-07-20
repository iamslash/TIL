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
  - [Routes, Pages](#routes-pages)
  - [Layouts](#layouts)
  - [Dark, Light mode](#dark-light-mode)
  - [Client State Management (Zustand)](#client-state-management-zustand)
  - [Props Drilling](#props-drilling)
  - [Profiles](#profiles)
  - [Unit Test (jest)](#unit-test-jest)
  - [Integration Test (msw)](#integration-test-msw)
  - [E2E Test (cypress)](#e2e-test-cypress)
  - [eslint, prettier](#eslint-prettier)
  - [Directory Structures](#directory-structures)
  - [Build, Deploy To GitHub Pages](#build-deploy-to-github-pages)
- [Advanced](#advanced)
  - [Smart vs Dumb Components](#smart-vs-dumb-components)
  - [Ant Design](#ant-design)
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
  - useQuery 훅을 사용하여 영화 데이터를 비동기적으로 가져옵니다. fetchMovies 함수는 fetch를 사용하여 데이터를 가져오고, useQuery는 이 함수를 호출하여 데이터를 캐싱하고 상태를 관리합니다. `useQuery()` 가 돌려준 값중 일부를 `movies` 에 저장하고 `movies` 를 JSX 에서 사용합니다. `movies` 가 변경되면 re-rendering 됩니다. react-query 가 상태관리를 해줍니다.
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

- `url` 을 form 으로 입력받고 영화를 다운로드 하자. `downloadUrl` 가 변경되면 re-rendering 이 되어야 한다. `useState()` 로 상태관리를 한다.
- `MoveForm` 은 `onSubmit` 을 부모 Component `App` 로 부터 prop 으로 전달 받는다. `onSubmit` 을 호출하면 부모 Component `App` 에 data 를 전달할 수 있다.
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

`useCallback()` 을 사용하면 `handleChange`와 `handleSubmit` 함수가 렌더링될 때마다 새로 정의되지 않으므로, 성능을 최적화할 수 있습니다. `handleChange`와 `handleSubmit` 함수를 `useCallback` 훅으로 감싸서 메모이제이션합니다. 이는 함수가 의존성 배열의 값이 변경되지 않는 한 동일한 참조를 유지하게 합니다. 그런데 실효성이 떨어진다고 한다. 굳이 사용할 필요가 없다?

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

- useForm Hook: useForm 훅을 사용하여 폼을 설정합니다. register를 통해 입력 필드를 등록하고, handleSubmit을 통해 폼 제출을 처리합니다.
- Validation: register 함수에 유효성 검사 규칙을 정의할 수 있습니다. 여기서는 URL 형식을 검사하고, 유효하지 않으면 오류 메시지를 표시합니다.
- Error Handling: formState.errors를 사용하여 유효성 검사 오류를 처리하고, 오류가 있을 경우 사용자에게 피드백을 제공합니다.

React Hook Form을 사용하면 폼 상태와 유효성 검사를 더욱 간결하고 효율적으로 관리할 수 있습니다. useState를 사용하지 않고도 폼 필드의 상태를 관리할 수 있으며, 성능 최적화에도 도움이 됩니다.

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

## Routes, Pages

```
npm install react-router-dom
```

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.tsx
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import HomePage from './pages/HomePage';
import MoviePage from './pages/MoviePage';
import NotFoundPage from './pages/NotFoundPage';

const App = () => {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/movies" element={<MoviePage />} />
                <Route path="*" element={<NotFoundPage />} />
            </Routes>
        </Router>
    );
};

export default App;

////////////////////////////////////////////////////////////////////////////////
// src/pages/HomePage.tsx
import React from 'react';

const HomePage = () => {
    return (
        <div>
            <h1>Home Page</h1>
            <p>Welcome to the Home Page!</p>
        </div>
    );
};

export default HomePage;

////////////////////////////////////////////////////////////////////////////////
// src/pages/MoviePage.tsx
import React, { useState } from 'react';
import axios from 'axios';
import { useQuery } from '@tanstack/react-query';
import Movie from '../Movie';
import MovieForm from '../MovieForm';

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

const MoviePage: React.FC = () => {
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

export default MoviePage;

////////////////////////////////////////////////////////////////////////////////
// src/pages/NotFoundPage.tsx
import React from 'react';

const NotFoundPage: React.FC = () => {
    return (
        <div>
            <h1>404 - Page Not Found</h1>
            <p>The page you are looking for does not exist.</p>
        </div>
    );
};

export default NotFoundPage;

```

## Layouts

- `layouts` 디렉토리를 만들고 `TopBar, SideBar, MainLayout` 컴포넌트를 구현했습니다.
- `MainLayout` 컴포넌트는 `TopBar`와 `SideBar`를 항상 렌더링하고, `<Outlet>`을 통해 변경되는 메인 콘텐츠를 표시합니다.
- `App.tsx`와 `index.tsx` 파일을 수정하여 라우팅과 레이아웃을 구성했습니다.
- `TopBar`와 `SideBar`가 항상 표시되고, `HomePage, MoviePage, NotFoundPage` 등의 메인 콘텐츠가 경로에 따라 변경되는 구조를 구현할 수 있습니다.

```js
////////////////////////////////////////////////////////////////////////////////
// src/laouts/TopBar.tsx
// src/layouts/TopBar.tsx
import React from 'react';
import { Link } from 'react-router-dom';

const TopBar: React.FC = () => {
    return (
        <div style={{ height: '50px', background: '#333', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'space-around' }}>
            <h1>TopBar</h1>
            <nav>
                <Link to="/" style={{ color: '#fff', textDecoration: 'none', margin: '0 10px' }}>Home</Link>
                <Link to="/movies" style={{ color: '#fff', textDecoration: 'none', margin: '0 10px' }}>Movies</Link>
            </nav>
        </div>
    );
};

export default TopBar;

////////////////////////////////////////////////////////////////////////////////
// src/laouts/SideBar.tsx
// src/layouts/SideBar.tsx
import React from 'react';
import { Link } from 'react-router-dom';

const SideBar: React.FC = () => {
    return (
        <div style={{ width: '200px', background: '#444', color: '#fff', height: '100vh', padding: '20px' }}>
            <h2>SideBar</h2>
            <nav>
                <ul style={{ listStyle: 'none', padding: 0 }}>
                    <li style={{ margin: '10px 0' }}>
                        <Link to="/" style={{ color: '#fff', textDecoration: 'none' }}>Home</Link>
                    </li>
                    <li style={{ margin: '10px 0' }}>
                        <Link to="/movies" style={{ color: '#fff', textDecoration: 'none' }}>Movies</Link>
                    </li>
                </ul>
            </nav>
        </div>
    );
};

export default SideBar;

////////////////////////////////////////////////////////////////////////////////
// src/laouts/MainLayout.tsx
import React from 'react';
import { Outlet } from 'react-router-dom';
import TopBar from './TopBar';
import SideBar from './SideBar';

const MainLayout: React.FC = () => {
    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
            <TopBar />
            <div style={{ display: 'flex', flexGrow: 1 }}>
                <SideBar />
                <div style={{ flexGrow: 1, padding: '20px' }}>
                    <Outlet />
                </div>
            </div>
        </div>
    );
};

export default MainLayout;

////////////////////////////////////////////////////////////////////////////////
// src/laouts/App.tsx
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import HomePage from './pages/HomePage';
import MoviePage from './pages/MoviePage';
import NotFoundPage from './pages/NotFoundPage';
import MainLayout from './layouts/MainLayout';

const App = () => {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<MainLayout />}>
                    <Route index element={<HomePage />} />
                    <Route path="movies" element={<MoviePage />} />
                    <Route path="*" element={<NotFoundPage />} />
                </Route>
            </Routes>
        </Router>
    );
};

export default App;

////////////////////////////////////////////////////////////////////////////////
// src/laouts/Index.tsx
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
```

## Dark, Light mode

- `App.tsx`에서 `theme` 상태를 관리하고 이를 `MainLayout`에 전달합니다.
- `MainLayout`에서 `theme`과 `setTheme`을 받아 `TopBar`에 전달합니다.
- `TopBar`에서 선택 상자를 통해 테마를 변경할 수 있도록 합니다.
- `index.css`에 다크 모드와 라이트 모드 스타일을 추가합니다.
- `TopBar`에서 선택 상자를 통해 다크 모드와 라이트 모드를 변경할 수 있습니다.

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.tsx
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import HomePage from './pages/HomePage';
import MoviePage from './pages/MoviePage';
import NotFoundPage from './pages/NotFoundPage';
import MainLayout from './layouts/MainLayout';

const App: React.FC = () => {
    const [theme, setTheme] = useState('light');

    useEffect(() => {
        document.body.className = theme;
    }, [theme]);

    return (
        <Router>
            <Routes>
                <Route path="/" element={<MainLayout theme={theme} setTheme={setTheme} />}>
                    <Route index element={<HomePage />} />
                    <Route path="movies" element={<MoviePage />} />
                    <Route path="*" element={<NotFoundPage />} />
                </Route>
            </Routes>
        </Router>
    );
};

export default App;

////////////////////////////////////////////////////////////////////////////////
// src/laouts/MainLayout.tsx
import React from 'react';
import { Outlet } from 'react-router-dom';
import TopBar from './TopBar';
import SideBar from './SideBar';

interface MainLayoutProps {
    theme: string;
    setTheme: React.Dispatch<React.SetStateAction<string>>;
}

const MainLayout: React.FC<MainLayoutProps> = ({ theme, setTheme }) => {
    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
            <TopBar theme={theme} setTheme={setTheme} />
            <div style={{ display: 'flex', flexGrow: 1 }}>
                <SideBar />
                <div style={{ flexGrow: 1, padding: '20px' }}>
                    <Outlet />
                </div>
            </div>
        </div>
    );
};

export default MainLayout;

////////////////////////////////////////////////////////////////////////////////
// src/laouts/TopBar.tsx
import React from 'react';
import { Link } from 'react-router-dom';

interface TopBarProps {
    theme: string;
    setTheme: React.Dispatch<React.SetStateAction<string>>;
}

const TopBar: React.FC<TopBarProps> = ({ theme, setTheme }) => {
    const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        setTheme(event.target.value);
    };

    return (
        <div style={{ height: '50px', background: '#333', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'space-around' }}>
            <h1>TopBar</h1>
            <nav>
                <Link to="/" style={{ color: '#fff', textDecoration: 'none', margin: '0 10px' }}>Home</Link>
                <Link to="/movies" style={{ color: '#fff', textDecoration: 'none', margin: '0 10px' }}>Movies</Link>
            </nav>
            <select value={theme} onChange={handleChange} style={{ marginLeft: '20px' }}>
                <option value="light">Light Mode</option>
                <option value="dark">Dark Mode</option>
            </select>
        </div>
    );
};

export default TopBar;

////////////////////////////////////////////////////////////////////////////////
// src/index.css
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}

body.light {
  background-color: white;
  color: black;
}

body.dark {
  background-color: black;
  color: white;
}
```

## Client State Management (Zustand)

- `zustand`를 설치하고 상태 관리를 위한 스토어를 생성했습니다.
- `App.tsx`에서 `zustand` 스토어를 사용하여 테마 상태를 관리하고 이를 반영했습니다.
- `MainLayout`과 `TopBar` 컴포넌트를 수정하여 `zustand` 스토어에서 테마 상태를 가져오고 변경할 수 있도록 했습니다.
- `index.css`에 다크 모드와 라이트 모드 스타일을 추가했습니다.
- 이렇게 하면 `TopBar`에서 선택 상자를 통해 다크 모드와 라이트 모드를 변경할 수 있습니다.

```bash
$ npm install zustand
```

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.tsx
import React, { useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import HomePage from './pages/HomePage';
import MoviePage from './pages/MoviePage';
import NotFoundPage from './pages/NotFoundPage';
import MainLayout from './layouts/MainLayout';
import { useThemeStore } from './store/useThemeStore';

const App: React.FC = () => {
    const theme = useThemeStore((state: { theme: string; }) => state.theme);

    useEffect(() => {
        document.body.className = theme;
    }, [theme]);

    return (
        <Router>
            <Routes>
                <Route path="/" element={<MainLayout />}>
                    <Route index element={<HomePage />} />
                    <Route path="movies" element={<MoviePage />} />
                    <Route path="*" element={<NotFoundPage />} />
                </Route>
            </Routes>
        </Router>
    );
};

export default App;

////////////////////////////////////////////////////////////////////////////////
// src/layouts/MainLayout.tsx
import React from 'react';
import { Outlet } from 'react-router-dom';
import TopBar from './TopBar';
import SideBar from './SideBar';

const MainLayout: React.FC = () => {
    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
            <TopBar />
            <div style={{ display: 'flex', flexGrow: 1 }}>
                <SideBar />
                <div style={{ flexGrow: 1, padding: '20px' }}>
                    <Outlet />
                </div>
            </div>
        </div>
    );
};

export default MainLayout;

////////////////////////////////////////////////////////////////////////////////
// src/layouts/TopBar.tsx
import React from 'react';
import { Link } from 'react-router-dom';
import { useThemeStore } from '../store/useThemeStore';

const TopBar: React.FC = () => {
    const theme = useThemeStore((state) => state.theme);
    const setTheme = useThemeStore((state) => state.setTheme);

    const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
        setTheme(event.target.value);
    };

    return (
        <div style={{ height: '50px', background: '#333', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'space-around' }}>
            <h1>TopBar</h1>
            <nav>
                <Link to="/" style={{ color: '#fff', textDecoration: 'none', margin: '0 10px' }}>Home</Link>
                <Link to="/movies" style={{ color: '#fff', textDecoration: 'none', margin: '0 10px' }}>Movies</Link>
            </nav>
            <select value={theme} onChange={handleChange} style={{ marginLeft: '20px' }}>
                <option value="light">Light Mode</option>
                <option value="dark">Dark Mode</option>
            </select>
        </div>
    );
};

export default TopBar;


////////////////////////////////////////////////////////////////////////////////
// src/store/useThemStore.ts
import create from 'zustand';

interface ThemeStore {
    theme: string;
    setTheme: (theme: string) => void;
}

export const useThemeStore = create<ThemeStore>((set) => ({
    theme: 'light',
    setTheme: (theme) => set({ theme }),
}));
```

## Props Drilling

Props Drilling은 React 컴포넌트 계층 구조에서, 상위 컴포넌트가 하위 컴포넌트에 데이터를 전달하기 위해 중간에 있는 모든 컴포넌트를 거쳐 props를 전달하는 과정을 말합니다. 이는 계층 구조가 깊어질수록 관리가 어려워지고 코드가 복잡해질 수 있습니다.

다음과 같은 방법으로 해결하자.

- Composition (children 사용)
- Context API 사용
- 전역 상태관리 라이브러리 사용 (redux, zustand)

**Composition (children 사용)**

Composition은 props를 통해 데이터를 직접 전달하는 대신, 컴포넌트의 자식(children)으로 다른 컴포넌트를 전달하여 데이터를 공유하는 방식입니다. 이를 통해 코드의 재사용성을 높이고, 중간 컴포넌트의 개입을 최소화할 수 있습니다.

```jsx
const Parent = ({ children }) => {
  return <div>{children}</div>;
};

const Child = ({ message }) => {
  return <p>{message}</p>;
};

// 사용 예
<Parent>
  <Child message="Hello, World!" />
</Parent>
```

**Context API 사용**

Context API는 React에서 전역 상태를 쉽게 관리할 수 있도록 도와주는 도구입니다. Provider를 통해 데이터를 전역으로 제공하고, Consumer를 통해 필요한 곳에서만 데이터를 사용할 수 있습니다. 이를 통해 props drilling 문제를 해결할 수 있습니다.

```jsx
import React, { createContext, useContext } from 'react';

const ThemeContext = createContext();

const Parent = () => {
  const theme = 'dark';
  return (
    <ThemeContext.Provider value={theme}>
      <Child />
    </ThemeContext.Provider>
  );
};

const Child = () => {
  const theme = useContext(ThemeContext);
  return <p>Current theme: {theme}</p>;
};
```

**전역 상태관리 라이브러리 사용 (redux, mobx, recoil, jotai, zustand)**

전역 상태관리 라이브러리는 애플리케이션의 상태를 전역적으로 관리하기 위해 사용됩니다. 이러한 라이브러리는 복잡한 상태 관리를 단순화하고, 컴포넌트 간의 상태 공유를 쉽게 만들어 props drilling 문제를 해결합니다.

Redux 예제:

```jsx
import { createStore } from 'redux';
import { Provider, useSelector, useDispatch } from 'react-redux';

const initialState = { theme: 'light' };

const reducer = (state = initialState, action) => {
  switch (action.type) {
    case 'SET_THEME':
      return { ...state, theme: action.theme };
    default:
      return state;
  }
};

const store = createStore(reducer);

const Parent = () => (
  <Provider store={store}>
    <Child />
  </Provider>
);

const Child = () => {
  const theme = useSelector((state) => state.theme);
  const dispatch = useDispatch();

  return (
    <div>
      <p>Current theme: {theme}</p>
      <button onClick={() => dispatch({ type: 'SET_THEME', theme: 'dark' })}>
        Set Dark Theme
      </button>
    </div>
  );
};
```

## Profiles

- `MovieForm.tsx`에서 현재 프로파일에 따라 기본 URL을 설정합니다.
- `.env.sample` 을 참고하여 `.env.qa, .env.stage, .env.prod` 파일을 제작한다.
- profile 별로 실행할 수 있다.
    ```bash
    $ npm run start:prod
    ```
- 각 프로파일별로 다른 URL을 사용하여 영화 데이터를 다운로드할 수 있고, 사용자도 입력한 URL을 통해 데이터를 가져올 수 있습니다.

```
$ touch .env.sample .env.qa .env.stage .env.prod
$ npm install env-cmd --save-dev
```

```bash
# .env.sample
REACT_APP_MOVIE_URL=https://yts.mx/api/v2/list_movies.json?sort_by=rating
```

```jsx
////////////////////////////////////////////////////////////////////////////////
// package.json
{
  "name": "movie-app",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "@hookform/resolvers": "^3.9.0",
    "@tanstack/react-query": "^5.51.1",
    "@testing-library/jest-dom": "^5.17.0",
    "@testing-library/react": "^13.4.0",
    "@testing-library/user-event": "^13.5.0",
    "@types/jest": "^27.5.2",
    "@types/node": "^16.18.101",
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "axios": "^1.7.2",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-hook-form": "^7.52.1",
    "react-router-dom": "^6.24.1",
    "react-scripts": "^5.0.1",
    "typescript": "^4.9.5",
    "web-vitals": "^2.1.4",
    "zod": "^3.23.8",
    "zustand": "^4.5.4"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject",
    "start:qa": "env-cmd -f .env.qa react-scripts start",
    "start:stage": "env-cmd -f .env.stage react-scripts start",
    "start:prod": "env-cmd -f .env.prod react-scripts start"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  },
  "devDependencies": {
    "@typescript-eslint/eslint-plugin": "^7.16.1",
    "@typescript-eslint/parser": "^7.16.1",
    "env-cmd": "^10.1.0"
  }
}

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
    const defaultUrl = process.env.REACT_APP_MOVIE_URL || '';

    const { register, handleSubmit, formState: { errors } } = useForm<UrlFormInputs>({
        resolver: zodResolver(urlSchema),
        defaultValues: {
            url: defaultUrl,
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

## Unit Test (jest)

jest 로 다음과 같은 것들을 할 수 있다.

- 유닛 테스트 (Unit Testing):
  - 개별 함수나 모듈을 테스트하여 올바르게 동작하는지 확인합니다.
  - 빠르고 독립적인 테스트가 가능하며, 주로 작은 코드 단위를 검증합니다.
- 통합 테스트 (Integration Testing):
  - 여러 모듈이 함께 작동하는 방식을 테스트합니다.
  - 데이터베이스와의 상호작용, API 호출 등 외부 시스템과의 통합을 검증합니다.
  - msw와 같은 라이브러리를 사용하여 API를 모킹할 수 있습니다.
- 엔드 투 엔드 테스트 (End-to-End Testing):
  - 애플리케이션의 전체 플로우를 테스트합니다.
  - 브라우저 환경에서 실제 사용자와 유사한 상호작용을 테스트합니다.
  - 일반적으로 Cypress나 Puppeteer와 같은 도구를 사용하여 수행됩니다.
- 스냅샷 테스트 (Snapshot Testing):
  - 컴포넌트의 렌더링 결과를 스냅샷으로 저장하고, 이후 테스트에서 변경된 부분이 있는지 확인합니다.
  - 주로 React 컴포넌트의 UI 변화를 감지하는 데 사용됩니다.
- 모킹 (Mocking):
  - 종속성을 모킹하여 테스트 환경을 제어할 수 있습니다.
  - jest.mock을 사용하여 모듈을 모킹하거나, msw를 사용하여 API 호출을 모킹할 수 있습니다.
- 비동기 코드 테스트:
  - 비동기 함수를 테스트하고, 프로미스가 올바르게 해결되거나 거부되는지 확인합니다.
  - async/await 또는 done 콜백을 사용하여 비동기 코드를 테스트할 수 있습니다.
- 타임 트래블 (Time Travel):
  - 타이머 함수를 모킹하고, 특정 시간에 대한 테스트를 수행할 수 있습니다.
  - jest.useFakeTimers와 jest.advanceTimersByTime을 사용하여 타이머를 제어할 수 있습니다.
- 코드 커버리지 (Code Coverage):
  - 테스트된 코드의 비율을 측정하고, 커버리지 리포트를 생성합니다.
  - `jest --coverage` 명령을 사용하여 코드 커버리지 정보를 확인할 수 있습니다.

다음은 `typescript, jest` 를 사용할 수 있는 `package.json` 이다.

```js
{
  "name": "movie-app",
  "version": "0.1.0",
  "private": true,
  "dependencies": {
    "@babel/plugin-proposal-private-property-in-object": "^7.21.11",
    "@hookform/resolvers": "^3.9.0",
    "@tanstack/react-query": "^5.51.1",
    "@types/node": "^16.18.101",
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "axios": "^1.7.2",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-hook-form": "^7.52.1",
    "react-router-dom": "^6.24.1",
    "react-scripts": "^5.0.1",
    "typescript": "^4.9.5",
    "web-vitals": "^2.1.4",
    "zod": "^3.23.8",
    "zustand": "^4.5.4"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "jest",
    "eject": "react-scripts eject",
    "start:qa": "env-cmd -f .env.qa react-scripts start",
    "start:stage": "env-cmd -f .env.stage react-scripts start",
    "start:prod": "env-cmd -f .env.prod react-scripts start"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.1.3",
    "@testing-library/react": "^14.0.0",
    "@testing-library/user-event": "^14.4.3",
    "@types/jest": "^29.5.4",
    "@typescript-eslint/eslint-plugin": "^7.16.1",
    "@typescript-eslint/parser": "^7.16.1",
    "env-cmd": "^10.1.0",
    "identity-obj-proxy": "^3.0.0",
    "jest": "^29.6.4",
    "jest-environment-jsdom": "^29.6.4",
    "jest-stare": "^2.5.1",
    "ts-jest": "^29.1.1",
    "ts-node": "^10.9.1",
    "typescript": "5.2.2"
  }
}
```

다음은 jest configuration 을 작성한 `jest.config.ts, jest.setup.ts` 이다.

```ts
////////////////////////////////////////////////////////////////////////////////
// jest.config.ts
import type { Config } from 'jest';

const config: Config = {
    /**
     * 테스트 실행 결과를 출력할 때, 상세한 정보를 보여준다.
     */
    verbose: true,

    /**
     * 테스트를 실행할 파일을 지정한다.
     */
    testMatch: ['<rootDir>/src/**/?(*.)+(spec|test).[jt]s?(x)'],

    /**
     * transform
     * Jest가 테스트 파일을 실행하기 전에 TypeScript를 JavaScript로 변환한다.
     */
    transform: {
        // 이 설정은 .ts와 .tsx 파일을 변환합니다.
        '^.+\\.tsx?$': ['ts-jest', { tsconfig: 'tsconfig.test.json' }],
    },

    transformIgnorePatterns: ['node_modules'],

    /**
     * 테스트 코드 커버리지를 수집한다.
     * 코드의 어느 부분이 테스트되지 않았는지 확인할 수 있다.
     */
    collectCoverage: true,

    /**
     * 테스트 코드 커버리지를 수집할 파일을 지정한다.
     */
    collectCoverageFrom: ['src/**/*.[jt]s?(x)'],

    /**
     * 테스트 코드 커버리지를 수집한 결과를 저장할 디렉토리를 지정한다.
     */
    coverageDirectory: 'coverage',

    /**
     * Jest가 TypeScript를 이해할 수 있도록 설정한다.
     */
    preset: 'ts-jest',

    /**
     * Jest가 테스트를 실행할 환경을 설정한다.
     */
    testEnvironment: 'jsdom',

    /**
     * Jest가 테스트를 실행하기 전에 실행할 코드를 설정한다.
     */
    setupFilesAfterEnv: ['<rootDir>/jest.setup.ts'],

    /**
     * 모듈 이름 매핑
     * CSS, Less, SCSS 파일을 모킹하도록 설정
     */
    moduleNameMapper: {
        '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    },
};

export default config;

////////////////////////////////////////////////////////////////////////////////
// jest.setup.ts
import '@testing-library/jest-dom';
```

다음과 같은 commandline 으로 `jest` 를 실행할 수 있다.

```
$ jest
```

jest 는 기본적으로 `src/__tests__` 의 파일들을 테스트한다. 그러나 설명의 편의를 위해 디렉토리 이동없이 `App.test.tsx, calculator.test.ts` 를 만들어 jest 를 실행해 보자.

```js
////////////////////////////////////////////////////////////////////////////////
// src/App.test.tsx
import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

test('renders TopBar text', () => {
  render(<App />);
  const topBarElement = screen.getByText(/TopBar/i);
  expect(topBarElement).toBeInTheDocument();
});

////////////////////////////////////////////////////////////////////////////////
// src/shared/utils/calculator.ts
// shared/utils/calculator.ts

/**
 * 두 숫자를 더하는 함수
 * @param a 첫 번째 숫자
 * @param b 두 번째 숫자
 * @returns 두 숫자의 합
 */
export const add = (a: number, b: number): number => {
    return a + b;
};

/**
 * 두 숫자를 빼는 함수
 * @param a 첫 번째 숫자
 * @param b 두 번째 숫자
 * @returns 두 숫자의 차
 */
export const subtract = (a: number, b: number): number => {
    return a - b;
};

/**
 * 두 숫자를 곱하는 함수
 * @param a 첫 번째 숫자
 * @param b 두 번째 숫자
 * @returns 두 숫자의 곱
 */
export const multiply = (a: number, b: number): number => {
    return a * b;
};

/**
 * 두 숫자를 나누는 함수
 * @param a 첫 번째 숫자
 * @param b 두 번째 숫자
 * @returns 두 숫자의 몫
 */
export const divide = (a: number, b: number): number => {
    if (b === 0) {
        throw new Error("0으로 나눌 수 없습니다.");
    }
    return a / b;
};

/**
 * 두 숫자의 나머지를 구하는 함수
 * @param a 첫 번째 숫자
 * @param b 두 번째 숫자
 * @returns 두 숫자의 나머지
 */
export const modulo = (a: number, b: number): number => {
    if (b === 0) {
        throw new Error("0으로 나눌 수 없습니다.");
    }
    return a % b;
};

////////////////////////////////////////////////////////////////////////////////
// src/shared/utils/calculator.test.ts
import { add, subtract, multiply, divide, modulo } from './calculator';

describe('calculator 함수 테스트', () => {
    test('덧셈 테스트', () => {
        expect(add(1, 2)).toBe(3);
        expect(add(-1, -2)).toBe(-3);
        expect(add(1, -2)).toBe(-1);
        expect(add(0, 0)).toBe(0);
    });

    test('뺄셈 테스트', () => {
        expect(subtract(5, 3)).toBe(2);
        expect(subtract(-1, -2)).toBe(1);
        expect(subtract(1, -2)).toBe(3);
        expect(subtract(0, 0)).toBe(0);
    });

    test('곱셈 테스트', () => {
        expect(multiply(2, 3)).toBe(6);
        expect(multiply(-1, -2)).toBe(2);
        expect(multiply(1, -2)).toBe(-2);
        expect(multiply(0, 5)).toBe(0);
    });

    test('나눗셈 테스트', () => {
        expect(divide(6, 3)).toBe(2);
        expect(divide(-6, -2)).toBe(3);
        expect(divide(6, -2)).toBe(-3);
        expect(() => divide(6, 0)).toThrow("0으로 나눌 수 없습니다.");
    });

    test('나머지 테스트', () => {
        expect(modulo(5, 2)).toBe(1);
        expect(modulo(-5, -2)).toBe(-1);
        expect(modulo(5, -2)).toBe(1);
        expect(() => modulo(5, 0)).toThrow("0으로 나눌 수 없습니다.");
    });
});
```

## Integration Test (msw)

WIP...

## E2E Test (cypress)

WIP...

## eslint, prettier

WIP...

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
    layouts/ # layout dir     
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
        stores/ # client state management   
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

## Smart vs Dumb Components

* [Loading states + Smart vs Dumb Components @ src](https://github.com/nomadcoders/movie_app/commit/9cc9cf90d3c21dfea9c04c455f59aab7440018c4)

-------

Smart component 는 state 이 있는 component 이다. Dumb component 는 state 이 없는 component 이다. props 만 있다. Dumb component 를 stateless 혹은 functional component 라고도 한다. Dumb component 는 state 가 필요없을 때 간결한 코드를 만들기 위해 사용한다. 그러나 state 가 없기도 하고 componentDidMount, componentWillMount 와 같은 함수들을 사용할 수 없다.

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

## Ant Design

* [Ant Design](https://ant.design/components/overview/)
  * [src](https://github.com/ant-design/ant-design)

-----

React ui library

## Don't Use `useEffect`

- [You Might Not Need an Effect | react.js](https://react.dev/learn/you-might-not-need-an-effect)
