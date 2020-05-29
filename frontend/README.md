# Abstract

HTML, CSS, JavaScript 에 대해 정리한다.

# Essentials

* [CSS Tutorial](https://www.w3schools.com/css/default.asp)
  * 최고의 CSS 튜토리얼
* [W3Schools How To](https://www.w3schools.com/howto/default.asp)
  * html, css, javascript examples
* [초보자를 위한 JavaScript 200제](http://www.infopub.co.kr/new/include/detail.asp?sku=05000265)
  * [src](http://www.infopub.co.kr/new/include/detail.asp?sku=05000265)


# Materials

* 프론트엔드 개발환경의 이해
  * 최고의 front-end blog
  * [프론트엔드 개발환경의 이해: NPM](http://jeonghwan-kim.github.io/series/2019/12/09/frontend-dev-env-npm.html)
  * [프론트엔드 개발환경의 이해: 웹팩(기본)](http://jeonghwan-kim.github.io/series/2019/12/10/frontend-dev-env-webpack-basic.html)
  * [프론트엔드 개발환경의 이해: Babel](http://jeonghwan-kim.github.io/series/2019/12/22/frontend-dev-env-babel.html)
  * [프론트엔드 개발환경의 이해: 린트](http://jeonghwan-kim.github.io/series/2019/12/30/frontend-dev-env-lint.html)
  * [프론트엔드 개발환경의 이해: 웹팩(심화)](http://jeonghwan-kim.github.io/series/2020/01/02/frontend-dev-env-webpack-intermediate.html)
* [30 Seconds of CSS](https://30-seconds.github.io/30-seconds-of-css/)
  * 유용한 CSS 팁모음
* [learn query](https://github.com/infinum/learnQuery)
  * let's make jQuery from scratch 
* [반응형 자기소개 웹사이트 따라 만들기 | 티티하우스 | 빔캠프 @ youtube](https://www.youtube.com/watch?v=KYo62fhaR7M)
  * [src](http://t.veam.me/aboutme/)
* [jsbin](https://jsbin.com/)
  * frontend web ide

# References

* [jQuery api documentation](http://api.jquery.com/category/selectors/)
* [google fonts](https://fonts.google.com/)
  * 글자를 입력하고 폰트미리보기를 할 수 있다.

# Basic

## NPM

```bash
$ mkdir basic

$ cd basic

$ npm init

package name:
version:
description:
entry point:
test command:
git repository:
keywords:
author:
license:

# You can skip with -y option
# This will generate package.json
$ npm init -y

$ npm test
```

Let's update pakcage.json like this

```js
{
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1"
  }
}
```

```bash
$ npm test
```

These are usual npm commands.

* `start`: 어플리케이션 실행
* `test`: 테스트
* `install`: 패키지 설치
* `uninstall`: 패키지 삭제

Let's update package.json again like this.

```js
{
  "scripts": {
    "build": "여기에 빌드 스크립트를 등록한다"
  }
}
```

And run it.

```console
$ npm run build
```

If you want to use js library in html, please embed this.

```html
<script src="https://unpkg.com/react@16/umd/react.development.js"></script>
```

If you wnat to download js library in loca, please run this command.

```console
$ npm install react
```

You can find out package.json has been changed.

```js
{
  "dependencies": {
    "react": "^16.12.0"
  }
}
```

The type of version are like these.

```js
// exact version
1.2.3

// greater than, greater than equal
// lesser than, lesser than equal
>1.2.3
>=1.2.3
<1.2.3
<=1.2.3

// If there is a minor version, upgrading patch version is ok.
~1.2.3

// If there is a major version, upgrading minor, patch is ok.
^1.2.3
```

## Webpack

js 는 ES2015 에서 표준화된 module import, export 를 내놓는다. 예를 들어 다음과 같이 app.js, math.js 는 ES2015 의 방법대로 math.js module 을 import, export 한다.

* math.js

```js
export function sum(a, b) { return a + b; }
```

* app.js

```js
import * as math from './math.js';
math.sum(1, 2); // 3
```

모든 browser 에서 module system 을 지원하지 않는다. webpack 으로 transpile 해야 한다.

webpack-cli 를 설치 및 실행해 보자.

```console
$ npm install -D webpack webpack-cli

$ ./node_module/bin/webpack
```

이제 webpack.config.js 를 작성하고 package.json, index.html 및 src/main.js, src/Utils.js, src/style.scss 를 다음과 같이 작성한다. 

* webpack.config.js

```js
const path = require('path')
const webpack = require('webpack')
const ExtractTextPlugin = require('extract-text-webpack-plugin')

module.exports = {
  entry: {
    main: './src/main.js'
  },
  output: {
    filename: 'bundle.js',
    path: path.join(__dirname, './dist')
  },
  module: {
    rules: [{
      test: /\.js$/,
      exclude: /node_modules/,
      use: {
        loader: 'babel-loader',
        options: {
          presets: ['env']
        }
      }
    }, {
      test: /\.scss$/,
      use: ExtractTextPlugin.extract({
        fallback: 'style-loader',
        use: ['css-loader', 'sass-loader']
      })
    }]
  },
  plugins: [
    new webpack.optimize.UglifyJsPlugin(),
    new ExtractTextPlugin('style.css')
  ]
};
```

* package.json

```js
{
  "name": "basic",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1",
    "build": "webpack"
  },
  "author": "",
  "license": "ISC",
  "devDependencies": {
    "babel-core": "^6.24.1",
    "babel-loader": "^7.0.0",
    "babel-preset-env": "^1.4.0",
    "css-loader": "^0.28.1",
    "extract-text-webpack-plugin": "^2.1.0",
    "node-sass": "^4.5.2",
    "sass-loader": "^6.0.5",
    "style-loader": "^0.17.0",
    "webpack": "^2.7.0",
    "webpack-cli": "^3.3.11"
  }
}
```

* index.html

```html
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <title>webpack study</title>
    <!-- <link rel="stylesheet" href="./dist/style.css"> -->
  </head>
  <body>
    <h1>webpack study</h1>

    <script src="./dist/bundle.js"></script>
  </body>
</html>
```

* src/main.js

```js
import Utils from './Utils'
require('./style.scss')

Utils.log('Hello World')
```

* src/style.js

```css
$bg-color: green;

body {
  background-color: $bg-color;
}
```

* src/Utils.js

```js
export default class Utils {
  static log(msg) {
    if (!msg) return;
    console.log('[LOG] ' + msg);
  }
}
```

다음과 같이 webpack 을 실행한다.

```console
$ npm run build
```

webpack.config.js 는 entry, output, module, plugins 으로 구성된다.

webpack 의 loader 는 js, css, image, font 등을 packing 한다. 다음과 같이 myloader.js 를 만들어서 작동원리를 이해해 보자.

* myloader.js

```js
module.exports = function myloader (content) {
  console.log('myloader is working');
  return content;
}
```

myloader 를 사용하기 위해 `webpack.config.js` 의 module 에 다음과 같이 설정한다.

```js
module: {
  rule: [{
    test: /\.js$/, // all files which end with .js
    use: [path.resolve('./myloader.js')]
  }]
}
```

그리고 다음과 같이 실행한다.

```console
$ npm run build
...
myloader is working
myloader is wokring
```

다음과 같이 myloader.js 를 수정하면 myloader.js 의 내용을 교체할 수 있다.

```js
module.exports = function myloader(content) {
  console.log('myloader is working');
  return content.replace('console.log(', 'alert(');
}
```

자주 사용하는 loader 는 css-loader, style-loader, file-loader, url-loader 등이 있다.

css-loader 는 css 파일을 import 로 불러올 수 있다.

먼저 css-loader 를 설치한다.

```console
$ npm install -D css-loader
```

이제 `app.js, style.css, webpack.config.js` 를 작성하자.

* /app.js

```js
import './style.css'
```

* /style.css

```js
body {
  background-color: green;
}
```

* webpack.config.js

```js
module.exports = {
  module: {
    rules: [{
      test: /\.css$/, // .css 확장자로 끝나는 모든 파일 
      use: ['css-loader'], // css-loader를 적용한다 
    }]
  }
}
```

그리고 다음과 같이 실행한다.

```console
$ npm run build
```

css code 가 js 로 transpile 되었다.

style-loader 는 js 로 transpile 된 style 을 runtime 에 DOM 에 추가한다. css 를 bundling 하기 위해서는 css-loader, style-loader 를 함께 사용해야 한다.

먼저 style-loader 를 설치한다.

```console
$ npm install -D style-loader
```

그리고 webpack.config.js 에 다음을 추가한다.

```js
module.exports = {
  module: {
    rules: [{
      test: /\.css$/,
      use: ['style-loader', 'css-loader'], // style-loader를 앞에 추가한다 
    }]
  }
}
```

file-loader 는 모든 file 을 js module 로 transpile 한다.

다음과 같이 `style.css` 를 작성한다.

```css
body {
  background-image: url(bg.png);
}
```

다음과 같이 `webconfig.config.js` 를 작성한다.

```js
module.exports = {
  module: {
    rules: [{
      test: /\.png$/, // .png 확장자로 마치는 모든 파일
      loader: 'file-loader', // 파일 로더를 적용한다
    }]
  }
}
```

다음과 같이 `webconfig.config.js` 의 경로를 수정해야 png 가 제대로 로딩된다.

```js
module.exports = {
  module: {
    rules: [{
      test: /\.png$/, // .png 확장자로 마치는 모든 파일
      loader: 'file-loader',
      options: {
        publicPath: './dist/', // prefix를 아웃풋 경로로 지정 
        name: '[name].[ext]?[hash]', // 파일명 형식 
      }
    }]
  }
}
```

url-loader 는 image 를 Base64 로 인코딩하여 js 에 packing 한다. 다음과 같이 url-loader 를 설치한다.

```console
$ npm install -D url-loader
```

다음과 같이 `webpack.config.js` 에 url-loader 를 추가한다.

```js
{
  test: /\.png$/,
  use: {
    loader: 'url-loader', // url 로더를 설정한다
    options: {
      publicPath: './dist/', // file-loader와 동일
      name: '[name].[ext]?[hash]', // file-loader와 동일
      limit: 5000 // 5kb 미만 파일만 data url로 처리 
    }
  }
}
```

plugin 은 bundle 단위로 처리한다. 한편 loader 는 file 단위로 처리를 한다. 예를 들어 obfucation 은 plugin 으로 할만 하다.

plugin 의 이해를 위해 다음과 같이 myplugin.js 를 만들어 보자.

```js
class MyPlugin {
  apply(compiler) {
    compiler.hooks.done.tap('My Plugin', stats => {
      console.log('MyPlugin: done');
    })
  }
}

module.exports = MyPlugin;
```

plugin 은 loader 와 다르게 class 로 제작한다.

다음은 webpack.config.js 에 myplugin 을 설정한다.

```js
const MyPlugin = require('./myplugin');

module.exports = {
  plugins: [
    new MyPlugin(),
  ]
}
```

그리고 build 해본다.

```console
$ npm run build
...
MyPlugin: done
```

myplugin 에서 bundle 안의 main.js 를 접근해 보자. 다음과 같이 myplugin.js 을 수정해보자. `compilation.assets['main.js'].source()` 와 같이 접근한다.

```js
class MyPlugin {
  apply(compiler) {
    compiler.hooks.done.tap('My Plugin', stats => {
      console.log('MyPlugin: done');
    })

    // compiler.plugin() 함수로 후처리한다
    compiler.plugin('emit', (compilation, callback) => { 
      const source = compilation.assets['main.js'].source();
      console.log(source);
      callback();
    })
  }
}
```

위와 같은 방법을 이용하여 banner 를 추가하도록 myplugin.js 를 수정해 보자.

```js
class MyPlugin {
  apply(compiler) {
    compiler.plugin('emit', (compilation, callback) => {
      const source = compilation.assets['main.js'].source();
      compilation.assets['main.js'].source = () => {
        const banner = [
          '/**',
          ' * 이것은 BannerPlugin이 처리한 결과입니다.',
          ' * Build Date: 2019-10-10',
          ' */'
          ''
        ].join('\n');
        return banner + '\n' + source;
      }
 
      callback();
    })
  }
}
```

자주 사용하는 plugin 은 BannerPlugin, DefinePlugin, HtmlWebpackPlugin, CleanWebpackPlugin, MiniCssExtractPlugin 등이 있다.

BannerPlugin 은 bundle 에 임의의 정보를 기록한다.

다음과 같이 `webpack.config.js` 를 작성한다.

```js
const webpack = require('webpack');

module.exports = {
  plugins: [
    new webpack.BannerPlugin({
      banner: '이것은 배너 입니다',
    })
  ]
```

다음과 같이 build date 을 삽입할 수도 있다.

```js
const webpack = require('webpack');

module.exports = {
  plugins: [
    new webpack.BannerPlugin({
      banner: () => `빌드 날짜: ${new Date().toLocaleString()}`
    })
  ]
```

별도의 file 에서 banner 를 읽어올 수도 있다.

```js
const webpack = require('webpack');
const banner = require('./banner.js');

module.exports = {
  plugins: [
    new webpack.BannerPlugin(banner);
  ]
```

banner.js 는 다음과 같이 작성한다.

```js
const childProcess = require('child_process');

module.exports = function banner() {
  const commit = childProcess.execSync('git rev-parse --short HEAD')
  const user = childProcess.execSync('git config user.name')
  const date = new Date().toLocaleString();
  
  return (
    `commitVersion: ${commit}` +
    `Build Date: ${date}\n` +
    `Author: ${user}`
  );
}
```

DefinePlugin 은 environment 에 따라 deployment 할 수 있도록 도와준다.

다음과 같이 webpack.config.js 를 작성한다. 

```js
const webpack = require('webpack');

export default {
  plugins: [
    new webpack.DefinePlugin({}),
  ]
}
```

DefinePlugin 에 environment 를 argument 로 넘겨준다.

```js
const webpack = require('webpack');

export default {
  plugins: [
    new webpack.DefinePlugin({
      VERSION: JSON.stringify('v.1.2.3'),
      PRODUCTION: JSON.stringify(false),
      MAX_COUNT: JSON.stringify(999),
      'api.domain': JSON.stringify('http://dev.api.domain.com'),      
    }),
  ]
}
```

HtmlWebpackPlugin 은 HTML post processing 을 도와 준다. 다음과 같이 설치한다.

```console
$ npm install -D html-webpack-plugin
```

그리고 다음과 같이 src/index.html 을 작성한다.

```html
<!DOCTYPE html>
<html>
  <head>
    <title>타이틀<%= env %></title>
  </head>
  <body>
    <!-- 로딩 스크립트 제거 -->
    <!-- <script src="dist/main.js"></script> -->
  </body>
</html>
```

HtmlWebpackPlugin 은 build time 에 src/index.html 에 env 를 주입시키고 html 을 generate 한다.

```js
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports {
  plugins: [
    new HtmlWebpackPlugin({
      template: './src/index.html', // 템플릿 경로를 지정 
      templateParameters: { // 템플릿에 주입할 파라매터 변수 지정
        env: process.env.NODE_ENV === 'development' ? '(개발용)' : '', 
      },
    })
  ]
}
```

다음은 HtmlWebpackPlugin 을 이용하여 파일을 압축하고 불필요한 주석을 제거하는 예이다.

```js

const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports {
  plugins: [
    new HtmlWebpackPlugin({
      minify: process.env.NODE_ENV === 'production' ? { 
        collapseWhitespace: true, // 빈칸 제거 
        removeComments: true, // 주석 제거 
      } : false,
    }
  ]
}
```

static resource 의 경우 browser cache 때문에 browser 에 반영되지 않는 경우가 있다. cache miss 를 위해 url 에 hash 를 추가한다.

```js
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports {
  plugins: [
    new HtmlWebpackPlugin({
      minify: process.env.NODE_ENV === 'production' ? { 
        collapseWhitespace: true, // 빈칸 제거 
        removeComments: true, // 주석 제거 
      } : false,
      hash: true,
    }
  ]
}
```

CleanWebpackPlugin 은 build 이전 결과물을 제거한다. 먼저 다음과 같이 CleanWebpackPlugin 을 설치하자.

```console
$ npm install -D clean-webpack-plugin
```

그리고 다음과 같이 webpack.config.js 를 설정한다.

```js
const { CleanWebpackPlugin } = require('clean-webpack-plugin');

module.exports = {
  plugins: [
    new CleanWebpackPlugin(),
  ]
}
```

`$ npm run build` 할 때 마다 `./dist` 가 깨끗하게 지워지고 생성된다.

MiniCssExtractPlugin 은 CSS 를 각각의 파일로 추출하도록 한다. 하나의 큰 파일보다는 여러개의 작은 파일을 서비스하는 것이 성능향상을 도모할 수 있다. 먼저 다음과 같이 MiniCssExtractPlugin 을 설치한다.

```console
$ npm install -D mini-css-extract-plugin
```

다음과 같이 webpack.config.js 를 작성한다.

```js
const MiniCssExtractPlugin = require('mini-css-extract-plugin');

module.exports = {
  plugins: [
    ...(
      process.env.NODE_ENV === 'production' 
      ? [ new MiniCssExtractPlugin({filename: `[name].css`}) ]
      : []
    ),
  ],
}
```

또한 다음과 같이 environment 에 따라 loader 를 달리할 수 도 있다.

```js
const MiniCssExtractPlugin = require('mini-css-extract-plugin');

module.exports = {
  module: {
    rules: [{
      test: /\.css$/,
      use: [
        process.env.NODE_ENV === 'production' 
        ? MiniCssExtractPlugin.loader  // 프로덕션 환경
        : 'style-loader',  // 개발 환경
        'css-loader'
      ],
    }]
  }
}
```

다음과 같이 production 환경으로 build 한다.

```console
$ NODE_ENV=production npm run build
```

## Webpack Deep Dive

## Babel

ToDo

## Lint

ToDo
