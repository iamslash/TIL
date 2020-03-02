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

`App` component 의 `state` 으로 title, poster 를 옮기자. 그리고 일정 시간 이후에 `state` 을 변경해 보자. `...this.state.movies` 를 이용하면 기존의 array 에 새로운 원소들을 추가할 수 있다.

```js
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
      //this.state.greeting = 'something'
      this.setState({
        movies: [
        ...this.state.movies,
        {
          tilte: "Trainspotting",
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
        {this.state.movies.map((movie, index) => {
          return <Movie title={movie.title} poster={movie.poster} key={index} />
        })}
      </div>
    );
  }
}
```

이 방법을 이용하면 스크롤을 아래로 내렸을 때 infinite scroll 을 구현할 수 있다.

## Loading states

loading screen 을 구현해 보자. `App` component 에 rendering 을 시작하자 마자 `Loading...` 을 출력하고 일정 시간이 지나면 state 을 업데이트하여 movies 가 rendering 되도록 해보자.

```js
class App extends Component {
  state = {
    greeting: 'Hello World',
  }  
  componentDidMount() {
    setTimeout(() => {
      console.log("change state...")
      //this.state.greeting = 'something'
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
```

## Smart vs Dumb

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

## AJAX on React

AJAX 는 Asynchrous JavaScript and XML 이다. 그러나 XML 은 사용하지 않고 JSON 을 사용한다. AJAJ 로 바뀌어야 한다???
다음은 fetch 함수를 이용하여 XHR (XML HTTP Request) 를 실행한 것이다.

```js
class App extends Component {
  state = {};
  componentDidMount() {
    fetch('https://yts.ag/api/v2/list_movies.json?sort_by=rating')
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
```

## Promises

[javascript proxy](/js/README.md#promise) 를 참고하여 이해하자. 다음은 앞서 작성한 XHR 의 handler 를 추가한 것이다.

```js
class App extends Component {
  state = {};
  componentDidMount() {
    fetch('https://yts.ag/api/v2/list_movies.json?sort_by=rating')
    .then(potato => potato.json())
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
```

그러나 CORS 설정이 되어 있지 않아서 error 가 발생한다. 다음과 같이 proxy 를 설정하면 해결할 수 있다.

```js

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
```

그러나 promise 는 then() 을 무수히 만들어 낸다. 이것을 Callback Hell 이라고 한다. Async Await 을 이용하면 Callback Hell 을 탈출할 수 있다.

## Async Await

[JavaScript async](/js/README.md#async) 를 참고하여 이해하자. promise 는 then() 의 남용으로 Call Back 함수가 많아져서 code 의 readability 를 떨어뜨린다. async, await 을 이용하면 call back functions 을 줄일 수 있고 code 의 readability 를 끌어 올릴 수 있다.

`async` 로 function 을 선언하면 function 안에서 `await` 로 기다릴 수 있다. `await` 로 기다리는 것은 `promise` 이다.

```js
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
```

## Updating Movie

이제 XHR 을 통해 얻어온 json 데이터를 화면에서 업데이트해 보자.

```js
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
...
```

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
  poster: PropTypes.string.isRequired,
  key: PropTypes.string.isRequired,
  genres: PropTypes.string.isRequired,
  synopsis: PropTypes.string.isRequired,
}
```

이제 `Movie` component 에 XHR 을 통하여 얻은 데이터 `title, poster, key, genres, synopsis` 를 표시해 보자.

```js
import React, {Component} from 'react';
import PropTypes from 'prop-types';
import './Movie.css';

function Movie({title, poster, genres, synopsis}) {
  return (
    <div clasName="Movie">
      <div className="Movie__Columns">
      <MoviePoster poster={poster} alt={title}/>
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
function MoviePoster({poster, alt}) {
  return (
    <img src={poster} alt={alt} title={alt} className="Movie__Poster" />
  );
}
function MovieGenre({genre}) {
  return (
    <span className="Movie__Genre">{genre}</span>
  );
}
Movie.propTypes = {
  title: PropTypes.string.isRequired,
  poster: PropTypes.string.isRequired,
  key: PropTypes.string.isRequired,
  genres: PropTypes.string.isRequired,
  synopsis: PropTypes.string.isRequired,
}
MoviePoster.propTypes = {
  poster: PropTypes.string.isRequired,
  alt: PropTypes.string.isRequired,
}
MovieGenre.proptype = {
  genre: PropTypes.string.isRequired,
}

export default Movie;
```

## CSS for Movie

## Building for Production

