- [Materials](#materials)
- [Basics](#basics)
  - [Overview](#overview)
- [Advanced](#advanced)
  - [What is package-lock.json](#what-is-package-lockjson)

---

# Materials

* [npm](https://www.npmjs.com/)

# Basics

## Overview

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

# Advanced

## What is package-lock.json

* [package-lock.json은 왜 필요할까?](https://hyunjun19.github.io/2018/03/23/package-lock-why-need/)

----

node_modules 혹은 package.json 수정되는 경우 업데이트된다. SCM 으로 tracking 해야 동일한 개발환경을 구성할 수 있다.
