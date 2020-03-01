# Materials

* [create react app](https://github.com/facebook/create-react-app)
  * react app wizard
* [nomad academy](https://academy.nomadcoders.co/courses/category/KR)
  * react class
* [reactjs @ inflearn](https://www.inflearn.com/course/reactjs-web/)
  * react class for beginner
* [Create-React-App: A Closer Look](https://github.com/nitishdayal/cra_closer_look)
  * stuffs in create-react-app in detail

# Basic

## Create React App

[Webpack](https://webpack.js.org/) 은 ES6 를 browser 가 이해할 수 있는 code 로 transpile 한다.

create-react-app 으로 startup-repo 를 생성할 수 있다. create-react-app 은 Webpack 을 포함한다.

```bash
$ brew install node.js
$ npm install -g create-react-app
$ create-react-app my-app
$ cd my-app
$ npm start
```

## Create React Components with JSX

`public/index.html` 에 `id="root"` 인 `div` 가 있다.

```html
<div id="root"></div>
```

`src/index.js` 는 `App` component 를 `id="root"` 인 `div` 에 rendering 하고 있다.

```js
ReactDOM.render(<App />, document.getElementById('root'));
```

`src/App.js` 는 `App` component 가 정의되어 있다.

```js
import React, {Component} from 'react';
import './App.css';
import Movie from './Movie';

class App extends Component {
  render() {
    return (
      <div className="App">
        <Movie />
        <Movie />
      </div>
    );
  }
}

export default App;
```

`src/App.js` 는 `Movie` component 를 rednering 하고 있다. `Movie` component 는 다시 `MoviePoser` component 를 rendering 하고 있다.

`src/Movie.js` 는 `Movie, MoviePoster` component 가 정의되어 있다.

```js
import React, {Component} from 'react';
import './Movie.css';

class Movie extends Component {
  render() {
    return (
      <div>
        <MoviePoster />
        <h1>Hello This is a movie</h1>
      </div>
    )
  }
}

class MoviePoster extends Component {
  render() {
    return (
      <img src='http://ojsfile.ohmynews.com/STD_IMG_FILE/2014/1202/IE001778581_STD.jpg'/>
    )
  }
}

export default Movie;
```

## Data flow with Props

data 를 `src/App.js` 에 선언해 보자. 그리고 `Movie` component 에게 props 형태로 전달한다.

```js
...
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

class App extends Component {
  render() {
    return (
      <div className="App">
        <Movie title={movieTitles[0]} poster={movieImages[0]}/>
        <Movie title={movieTitles[1]} poster={movieImages[1]}/>
        <Movie title={movieTitles[2]} poster={movieImages[2]}/>
        <Movie title={movieTitles[3]} poster={movieImages[3]}/>
      </div>
    );
  }
}
...
```

`Movie` component 는 props 의 일부를 다시 `MoviePoster` component 에게 전달한다.

```js
...
class Movie extends Component {
  render() {
    console.log(this.props);
    return (
      <div>
        <MoviePoster poster={this.props.poster}/>
        <h1>{this.props.title}</h1>
      </div>
    )
  }
}

class MoviePoster extends Component {
  render() {
    return (
      <img src={this.props.poster}/>
    )
  }
}
...
```

JSX 를 이용하여 각 component 의 props 를 rendering 한다.

## Lists with .maps

`movies` 배열을 만들고 `App` component 를 `.map` function 을 활용하여 구현해보자.

```js
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
  render() {
    return (
      <div className="App">
        {movies.map(movie => {
          return <Movie title={movie.title} poster={movie.poster} />
        })}
      </div>
    );
  }
}
```

## Validating Props with Prop Types

`static propTypes` 를 선언하여 props 의 값을 제어할 수 있다. 이때 PropTypes module 이 설치되어야 한다. `yarn add PropTypes`

```js
import React, {Component} from 'react';
import PropTypes from 'prop-types';
import './Movie.css';

class Movie extends Component {
  static propTypes = {
    title: PropTypes.string.isRequired,
    poster: PropTypes.string.isRequired
  }
  render() {
    console.log(this.props);
    return (
      <div>
        <MoviePoster poster={this.props.poster}/>
        <h1>{this.props.title}</h1>
      </div>
    )
  }
}

class MoviePoster extends Component {
  static propTypes = {
    poster: PropTypes.string.isRequired
  }
  render() {
    return (
      <img src={this.props.poster}/>
    )
  }
}

export default Movie;
```

## Component Lifecycle

하나의 component 는 다음과 같은 순서로 `Render, Update` 가 수행된다. Override function 의 순서를 주의하자.

```js
class App extends Component {
  // Render: componentWillMount() -> render() -> componentDidMount()
  //
  // Update: componentWillReceiveProps() -> shouldComponentUpdate() -> 
  // componentWillUpate() -> render() -> componentDidUpdate()
  componentWillMount() {
    console.log("componentWillMount");
  }
  componentDidMount() {
    console.log("componentDidMount");
  }
  render() {
    console.log("render");
    return (
      <div className="App">
        {movies.map((movie, index) => {
          return <Movie title={movie.title} poster={movie.poster} key={index} />
        })}
      </div>
    );
  }
}
```

## Thinking in React Component State

`App` component 에 state 를 선언하고 `render()` 에서 바꿔보자. 반드시 `this.setState()` 함수를 이용해야 한다.

```js
class App extends Component {
  state = {
    greeting: 'Hello'
  }
  componentDidMount() {
    setTimeout(() => {
      //this.state.greeting = 'something'
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
        {movies.map((movie, index) => {
          return <Movie title={movie.title} poster={movie.poster} key={index} />
        })}
      </div>
    );
  }
}
```

## Practicing this setState

## Loading states

## Smart vs Dumb

## AJAX on React

## Promises

## Async Await

## Updating Movie

## CSS for Movie

## Building for Production

