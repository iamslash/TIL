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

* [프론트엔드 개발환경의 이해: NPM](http://jeonghwan-kim.github.io/series/2019/12/09/frontend-dev-env-npm.html)
  * 최고의 front-end blog
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




## Babel

## Lint

## Webpack Deep Dive


