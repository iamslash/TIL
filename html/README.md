- [Materials](#materials)
- [Tutorial](#tutorial)
  - [단계 1: 기본 HTML 구조 설정](#단계-1-기본-html-구조-설정)
  - [단계2: 전체 CSS 스타일 초기화](#단계2-전체-css-스타일-초기화)
  - [단계3: body의 기본 스타일 설정](#단계3-body의-기본-스타일-설정)
  - [단계4: header 요소 추가](#단계4-header-요소-추가)
  - [단계5: header 스타일 설정](#단계5-header-스타일-설정)
  - [단계6: header의 center와 right 스타일 설정](#단계6-header의-center와-right-스타일-설정)
  - [단계7: aside 요소 추가 및 스타일 설정](#단계7-aside-요소-추가-및-스타일-설정)
  - [단계8: main 요소 추가](#단계8-main-요소-추가)
  - [단계9: 첫 번째 section 요소 추가](#단계9-첫-번째-section-요소-추가)
  - [단계10: 첫 번째 section의 내부 스타일 추가](#단계10-첫-번째-section의-내부-스타일-추가)
  - [단계11: 두 번째 section 요소 추가](#단계11-두-번째-section-요소-추가)
  - [단계12: 세 번째 section 요소 추가](#단계12-세-번째-section-요소-추가)
  - [단계13: 세 번째 section 스타일 추가](#단계13-세-번째-section-스타일-추가)
  - [단계14: 추가 CSS 스타일 설정](#단계14-추가-css-스타일-설정)
  - [단계15: 최종 HTML 구조 확인 및 정리](#단계15-최종-html-구조-확인-및-정리)

-----

# Materials

- [ 제대로 파는 HTML & CSS (무료 파트) | yalco](https://www.yalco.kr/lectures/html-css/)

# Tutorial

## 단계 1: 기본 HTML 구조 설정

```js
<!-- 단계 1: 기본 HTML 구조 설정 -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
<body>
</body>
</html>
```

## 단계2: 전체 CSS 스타일 초기화

```js
<!-- 단계 2: 전체 CSS 스타일 초기화 -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <style>
        /* 모든 요소의 기본 여백과 패딩을 제거하고, 박스 크기 설정을 변경합니다. */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
    </style>
</head>
<body>
</body>
</html>
```

## 단계3: body의 기본 스타일 설정
```js
<!-- 단계 3: body의 기본 스타일 설정 -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <style>
        /* 모든 요소의 기본 여백과 패딩을 제거하고, 박스 크기 설정을 변경합니다. */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        /* body 요소의 배경 색상을 설정합니다. */
        body {
            background-color: #f5f5f5;
        }
    </style>
</head>
<body>
</body>
</html>
```

## 단계4: header 요소 추가

```js
<!-- 단계 4: header 요소 추가 -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <style>
        /* 모든 요소의 기본 여백과 패딩을 제거하고, 박스 크기 설정을 변경합니다. */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        /* body 요소의 배경 색상을 설정합니다. */
        body {
            background-color: #f5f5f5;
        }
    </style>
</head>
<body>
    <!-- header 요소: 페이지 상단의 고정 헤더입니다. -->
    <header>
        <div class="left">
            <div class="logo">logo</div>
            <input type="text">
        </div>
        <div class="center">
            <div class="icons">
                <div class="icon">home</div>
                <div class="icon">friends</div>
                <div class="icon">camera</div>
                <div class="icon">game</div>
            </div>
        </div>
        <div class="right">
            <div class="profile">profile</div>
        </div>
    </header>
</body>
</html>
```

## 단계5: header 스타일 설정

```js
<!-- 단계 5: header 스타일 설정 -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <style>
        /* 모든 요소의 기본 여백과 패딩을 제거하고, 박스 크기 설정을 변경합니다. */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        /* body 요소의 배경 색상을 설정합니다. */
        body {
            background-color: #f5f5f5;
        }

        /* header 요소의 스타일을 설정합니다. */
        header {
            width: 100%;
            height: 90px;
            background-color: #fff;
            border-bottom: 2px solid #ddd;
            position: fixed;
            top: 0;
            left: 0;
            display: flex;
            justify-content: space-between;
            padding: 16px;
        }

        /* header 안의 left 클래스에 대한 스타일입니다. */
        header .left {
            display: flex;
            align-self: center;
        }

        /* header 안의 left 클래스 안의 input 요소에 대한 스타일입니다. */
        header .left input {
            height: 40px;
            margin-left: 20px;
        }
    </style>
</head>
<body>
    <!-- header 요소: 페이지 상단의 고정 헤더입니다. -->
    <header>
        <div class="left">
            <div class="logo">logo</div>
            <input type="text">
        </div>
        <div class="center">
            <div class="icons">
                <div class="icon">home</div>
                <div class="icon">friends</div>
                <div class="icon">camera</div>
                <div class="icon">game</div>
            </div>
        </div>
        <div class="right">
            <div class="profile">profile</div>
        </div>
    </header>
</body>
</html>
```

## 단계6: header의 center와 right 스타일 설정

```js
<!-- 단계 6: header의 center와 right 스타일 설정 -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <style>
        /* 모든 요소의 기본 여백과 패딩을 제거하고, 박스 크기 설정을 변경합니다. */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        /* body 요소의 배경 색상을 설정합니다. */
        body {
            background-color: #f5f5f5;
        }

        /* header 요소의 스타일을 설정합니다. */
        header {
            width: 100%;
            height: 90px;
            background-color: #fff;
            border-bottom: 2px solid #ddd;
            position: fixed;
            top: 0;
            left: 0;
            display: flex;
            justify-content: space-between;
            padding: 16px;
        }

        /* header 안의 left 클래스에 대한 스타일입니다. */
        header .left {
            display: flex;
            align-self: center;
        }

        /* header 안의 left 클래스 안의 input 요소에 대한 스타일입니다. */
        header .left input {
            height: 40px;
            margin-left: 20px;
        }

        /* header 안의 center 클래스에 대한 스타일입니다. */
        header .center {
            display: flex;
            align-items: center;
        }

        /* header 안의 center 클래스 안의 icons 클래스에 대한 스타일입니다. */
        header .center .icons {
            display: flex;
        }

        /* header 안의 center 클래스 안의 icons 클래스 안의 icon 클래스에 대한 스타일입니다. */
        header .center .icons .icon {
            padding: 0 16px;
        }

        /* header 안의 right 클래스에 대한 스타일입니다. */
        header .right {
            display: flex;
            align-items: center;
        }
    </style>
</head>
<body>
    <!-- header 요소: 페이지 상단의 고정 헤더입니다. -->
    <header>
        <div class="left">
            <div class="logo">logo</div>
            <input type="text">
        </div>
        <div class="center">
            <div class="icons">
                <div class="icon">home</div>
                <div class="icon">friends</div>
                <div class="icon">camera</div>
                <div class="icon">game</div>
            </div>
        </div>
        <div class="right">
            <div class="profile">profile</div>
        </div>
    </header>
</body>
</html>
```

## 단계7: aside 요소 추가 및 스타일 설정

```js
<!-- 단계 7: aside 요소 추가 및 스타일 설정 -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <style>
        /* 모든 요소의 기본 여백과 패딩을 제거하고, 박스 크기 설정을 변경합니다. */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        /* body 요소의 배경 색상을 설정합니다. */
        body {
            background-color: #f5f5f5;
        }

        /* header 요소의 스타일을 설정합니다. */
        header {
            width: 100%;
            height: 90px;
            background-color: #fff;
            border-bottom: 2px solid #ddd;
            position: fixed;
            top: 0;
            left: 0;
            display: flex;
            justify-content: space-between;
            padding: 16px;
        }

        /* header 안의 left 클래스에 대한 스타일입니다. */
        header .left {
            display: flex;
            align-self: center;
        }

        /* header 안의 left 클래스 안의 input 요소에 대한 스타일입니다. */
        header .left input {
            height: 40px;
            margin-left: 20px;
        }

        /* header 안의 center 클래스에 대한 스타일입니다. */
        header .center {
            display: flex;
            align-items: center;
        }

        /* header 안의 center 클래스 안의 icons 클래스에 대한 스타일입니다. */
        header .center .icons {
            display: flex;
        }

        /* header 안의 center 클래스 안의 icons 클래스 안의 icon 클래스에 대한 스타일입니다. */
        header .center .icons .icon {
            padding: 0 16px;
        }

        /* header 안의 right 클래스에 대한 스타일입니다. */
        header .right {
            display: flex;
            align-items: center;
        }

        /* aside 요소의 스타일을 설정합니다. */
        aside {
            width: 240px;
            height: 100%;
            position: fixed;
            background-color: red;
            top: 90px;
            left: 0;
            border-right: 2px solid #ddd;
            opacity: 0.5;
        }
    </style>
</head>
<body>
    <!-- header 요소: 페이지 상단의 고정 헤더입니다. -->
    <header>
        <div class="left">
            <div class="logo">logo</div>
            <input type="text">
        </div>
        <div class="center">
            <div class="icons">
                <div class="icon">home</div>
                <div class="icon">friends</div>
                <div class="icon">camera</div>
                <div class="icon">game</div>
            </div>
        </div>
        <div class="right">
            <div class="profile">profile</div>
        </div>
    </header>

    <!-- aside 요소: 페이지의 사이드바입니다. -->
    <aside></aside>
</body>
</html>
```

## 단계8: main 요소 추가

```js
<!-- 단계 8: main 요소 추가 -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <style>
        /* 모든 요소의 기본 여백과 패딩을 제거하고, 박스 크기 설정을 변경합니다. */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        /* body 요소의 배경 색상을 설정합니다. */
        body {
            background-color: #f5f5f5;
        }

        /* header 요소의 스타일을 설정합니다. */
        header {
            width: 100%;
            height: 90px;
            background-color: #fff;
            border-bottom: 2px solid #ddd;
            position: fixed;
            top: 0;
            left: 0;
            display: flex;
            justify-content: space-between;
            padding: 16px;
        }

        /* header 안의 left 클래스에 대한 스타일입니다. */
        header .left {
            display: flex;
            align-self: center;
        }

        /* header 안의 left 클래스 안의 input 요소에 대한 스타일입니다. */
        header .left input {
            height: 40px;
            margin-left: 20px;
        }

        /* header 안의 center 클래스에 대한 스타일입니다. */
        header .center {
            display: flex;
            align-items: center;
        }

        /* header 안의 center 클래스 안의 icons 클래스에 대한 스타일입니다. */
        header .center .icons {
            display: flex;
        }

        /* header 안의 center 클래스 안의 icons 클래스 안의 icon 클래스에 대한 스타일입니다. */
        header .center .icons .icon {
            padding: 0 16px;
        }

        /* header 안의 right 클래스에 대한 스타일입니다. */
        header .right {
            display: flex;
            align-items: center;
        }

        /* aside 요소의 스타일을 설정합니다. */
        aside {
            width: 240px;
            height: 100%;
            position: fixed;
            background-color: red;
            top: 90px;
            left: 0;
            border-right: 2px solid #ddd;
            opacity: 0.5;
        }

        /* main 요소의 스타일을 설정합니다. */
        main {
            width: 100%;
            padding-top: 122px;
            padding-bottom: 32px;
            padding-left: 272px;
            padding-right: 32px;
        }
    </style>
</head>
<body>
    <!-- header 요소: 페이지 상단의 고정 헤더입니다. -->
    <header>
        <div class="left">
            <div class="logo">logo</div>
            <input type="text">
        </div>
        <div class="center">
            <div class="icons">
                <div class="icon">home</div>
                <div class="icon">friends</div>
                <div class="icon">camera</div>
                <div class="icon">game</div>
            </div>
        </div>
        <div class="right">
            <div class="profile">profile</div>
        </div>
    </header>

    <!-- aside 요소: 페이지의 사이드바입니다. -->
    <aside></aside>

    <!-- main 요소: 페이지의 주요 콘텐츠 영역입니다. -->
    <main></main>
</body>
</html>
```

## 단계9: 첫 번째 section 요소 추가

```js
<!-- 단계 9: 첫 번째 section 요소 추가 -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <style>
        /* 모든 요소의 기본 여백과 패딩을 제거하고, 박스 크기 설정을 변경합니다. */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        /* body 요소의 배경 색상을 설정합니다. */
        body {
            background-color: #f5f5f5;
        }

        /* header 요소의 스타일을 설정합니다. */
        header {
            width: 100%;
            height: 90px;
            background-color: #fff;
            border-bottom: 2px solid #ddd;
            position: fixed;
            top: 0;
            left: 0;
            display: flex;
            justify-content: space-between;
            padding: 16px;
        }

        /* header 안의 left 클래스에 대한 스타일입니다. */
        header .left {
            display: flex;
            align-self: center;
        }

        /* header 안의 left 클래스 안의 input 요소에 대한 스타일입니다. */
        header .left input {
            height: 40px;
            margin-left: 20px;
        }

        /* header 안의 center 클래스에 대한 스타일입니다. */
        header .center {
            display: flex;
            align-items: center;
        }

        /* header 안의 center 클래스 안의 icons 클래스에 대한 스타일입니다. */
        header .center .icons {
            display: flex;
        }

        /* header 안의 center 클래스 안의 icons 클래스 안의 icon 클래스에 대한 스타일입니다. */
        header .center .icons .icon {
            padding: 0 16px;
        }

        /* header 안의 right 클래스에 대한 스타일입니다. */
        header .right {
            display: flex;
            align-items: center;
        }

        /* aside 요소의 스타일을 설정합니다. */
        aside {
            width: 240px;
            height: 100%;
            position: fixed;
            background-color: red;
            top: 90px;
            left: 0;
            border-right: 2px solid #ddd;
            opacity: 0.5;
        }

        /* main 요소의 스타일을 설정합니다. */
        main {
            width: 100%;
            padding-top: 122px;
            padding-bottom: 32px;
            padding-left: 272px;
            padding-right: 32px;
        }

        /* section 요소의 기본 스타일을 설정합니다. */
        section {
            padding: 32px;
            background-color: #fff;
        }
    </style>
</head>
<body>
    <!-- header 요소: 페이지 상단의 고정 헤더입니다. -->
    <header>
        <div class="left">
            <div class="logo">logo</div>
            <input type="text">
        </div>
        <div class="center">
            <div class="icons">
                <div class="icon">home</div>
                <div class="icon">friends</div>
                <div class="icon">camera</div>
                <div class="icon">game</div>
            </div>
        </div>
        <div class="right">
            <div class="profile">profile</div>
        </div>
    </header>

    <!-- aside 요소: 페이지의 사이드바입니다. -->
    <aside></aside>

    <!-- main 요소: 페이지의 주요 콘텐츠 영역입니다. -->
    <main>
        <!-- 첫 번째 section: 이야기 섹션입니다. -->
        <section id="story-section">
            <h2>Stories</h2>
        </section>
    </main>
</body>
</html>
```

## 단계10: 첫 번째 section의 내부 스타일 추가

```js
<!-- 단계 10: 첫 번째 section의 내부 스타일 추가 -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <style>
        /* 모든 요소의 기본 여백과 패딩을 제거하고, 박스 크기 설정을 변경합니다. */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        /* body 요소의 배경 색상을 설정합니다. */
        body {
            background-color: #f5f5f5;
        }

        /* header 요소의 스타일을 설정합니다. */
        header {
            width: 100%;
            height: 90px;
            background-color: #fff;
            border-bottom: 2px solid #ddd;
            position: fixed;
            top: 0;
            left: 0;
            display: flex;
            justify-content: space-between;
            padding: 16px;
        }

        /* header 안의 left 클래스에 대한 스타일입니다. */
        header .left {
            display: flex;
            align-self: center;
        }

        /* header 안의 left 클래스 안의 input 요소에 대한 스타일입니다. */
        header .left input {
            height: 40px;
            margin-left: 20px;
        }

        /* header 안의 center 클래스에 대한 스타일입니다. */
        header .center {
            display: flex;
            align-items: center;
        }

        /* header 안의 center 클래스 안의 icons 클래스에 대한 스타일입니다. */
        header .center .icons {
            display: flex;
        }

        /* header 안의 center 클래스 안의 icons 클래스 안의 icon 클래스에 대한 스타일입니다. */
        header .center .icons .icon {
            padding: 0 16px;
        }

        /* header 안의 right 클래스에 대한 스타일입니다. */
        header .right {
            display: flex;
            align-items: center;
        }

        /* aside 요소의 스타일을 설정합니다. */
        aside {
            width: 240px;
            height: 100%;
            position: fixed;
            background-color: red;
            top: 90px;
            left: 0;
            border-right: 2px solid #ddd;
            opacity: 0.5;
        }

        /* main 요소의 스타일을 설정합니다. */
        main {
            width: 100%;
            padding-top: 122px;
            padding-bottom: 32px;
            padding-left: 272px;
            padding-right: 32px;
        }

        /* section 요소의 기본 스타일을 설정합니다. */
        section {
            padding: 32px;
            background-color: #fff;
        }

        /* #story-section 내의 h2 요소 스타일을 설정합니다. */
        #story-section h2 {
            margin-bottom: 4rem;
        }

        /* story-list 클래스의 스타일을 설정합니다. */
        .story-list {
            display: flex;
            gap: 1rem;
        }

        /* story 클래스의 스타일을 설정합니다. */
        .story {
            width: calc(100% / 3);
            height: 10rem;
            position: relative;
        }

        /* story-profile 클래스의 스타일을 설정합니다. */
        .story-profile {
            width: 4rem;
            height: 4rem;
            background-color: #1659bd;
            position: absolute;
            left: 50%;
            top: -2rem;
            transform: translateX(-50%);
            border-radius: 50%;
        }

        /* story-thumbnail 클래스의 스타일을 설정합니다. */
        .story-thumbnail {
            width: 100%;
            height: 100%;
            background-color: #eee;
            border-radius: 5px;
        }

        /* story-description 클래스의 스타일을 설정합니다. */
        .story-description {
            position: absolute;
            bottom: 0;
            width: 100%;
            padding: 16px;
        }

        /* story-description 클래스 안의 p 요소 스타일을 설정합니다. */
        .story-description p {
            width: 100%;
            display: -webkit-box;
            -webkit-line-clamp: 1;
            -webkit-box-orient: vertical;
            text-overflow: ellipsis;
            overflow: hidden;
        }
    </style>
</head>
<body>
    <!-- header 요소: 페이지 상단의 고정 헤더입니다. -->
    <header>
        <div class="left">
            <div class="logo">logo</div>
            <input type="text">
        </div>
        <div class="center">
            <div class="icons">
                <div class="icon">home</div>
                <div class="icon">friends</div>
                <div class="icon">camera</div>
                <div class="icon">game</div>
            </div>
        </div>
        <div class="right">
            <div class="profile">profile</div>
        </div>
    </header>

    <!-- aside 요소: 페이지의 사이드바입니다. -->
    <aside></aside>

    <!-- main 요소: 페이지의 주요 콘텐츠 영역입니다. -->
    <main>
        <!-- 첫 번째 section: 이야기 섹션입니다. -->
        <section id="story-section">
            <h2>Stories</h2>
            <div class="story-list">
                <!-- story 클래스: 각각의 이야기 카드입니다. -->
                <div class="story">
                    <div class="story-profile"></div>
                    <div class="story-thumbnail"></div>
                    <div class="story-description">
                        <p>
                            Lorem Ipsum Dolor asdfjkasdjflkads
                            loremasdjfajsdlfajds
                        </p>
                    </div>
                </div>

                <div class="story">
                    <div class="story-profile"></div>
                    <div class="story-thumbnail"></div>
                    <div class="story-description">
                        Lorem Ipsum Dolor
                    </div>
                </div>

                <div class="story">
                    <div class="story-profile"></div>
                    <div class="story-thumbnail"></div>
                    <div class="story-description">
                        Lorem Ipsum Dolor
                    </div>
                </div>
            </div>
        </section>
    </main>
</body>
</html>
```

## 단계11: 두 번째 section 요소 추가

```js
<!-- 단계 11: 두 번째 section 요소 추가 -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <style>
        /* 모든 요소의 기본 여백과 패딩을 제거하고, 박스 크기 설정을 변경합니다. */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        /* body 요소의 배경 색상을 설정합니다. */
        body {
            background-color: #f5f5f5;
        }

        /* header 요소의 스타일을 설정합니다. */
        header {
            width: 100%;
            height: 90px;
            background-color: #fff;
            border-bottom: 2px solid #ddd;
            position: fixed;
            top: 0;
            left: 0;
            display: flex;
            justify-content: space-between;
            padding: 16px;
        }

        /* header 안의 left 클래스에 대한 스타일입니다. */
        header .left {
            display: flex;
            align-self: center;
        }

        /* header 안의 left 클래스 안의 input 요소에 대한 스타일입니다. */
        header .left input {
            height: 40px;
            margin-left: 20px;
        }

        /* header 안의 center 클래스에 대한 스타일입니다. */
        header .center {
            display: flex;
            align-items: center;
        }

        /* header 안의 center 클래스 안의 icons 클래스에 대한 스타일입니다. */
        header .center .icons {
            display: flex;
        }

        /* header 안의 center 클래스 안의 icons 클래스 안의 icon 클래스에 대한 스타일입니다. */
        header .center .icons .icon {
            padding: 0 16px;
        }

        /* header 안의 right 클래스에 대한 스타일입니다. */
        header .right {
            display: flex;
            align-items: center;
        }

        /* aside 요소의 스타일을 설정합니다. */
        aside {
            width: 240px;
            height: 100%;
            position: fixed;
            background-color: red;
            top: 90px;
            left: 0;
            border-right: 2px solid #ddd;
            opacity: 0.5;
        }

        /* main 요소의 스타일을 설정합니다. */
        main {
            width: 100%;
            padding-top: 122px;
            padding-bottom: 32px;
            padding-left: 272px;
            padding-right: 32px;
        }

        /* section 요소의 기본 스타일을 설정합니다. */
        section {
            padding: 32px;
            background-color: #fff;
        }

        /* #story-section 내의 h2 요소 스타일을 설정합니다. */
        #story-section h2 {
            margin-bottom: 4rem;
        }

        /* story-list 클래스의 스타일을 설정합니다. */
        .story-list {
            display: flex;
            gap: 1rem;
        }

        /* story 클래스의 스타일을 설정합니다. */
        .story {
            width: calc(100% / 3);
            height: 10rem;
            position: relative;
        }

        /* story-profile 클래스의 스타일을 설정합니다. */
        .story-profile {
            width: 4rem;
            height: 4rem;
            background-color: #1659bd;
            position: absolute;
            left: 50%;
            top: -2rem;
            transform: translateX(-50%);
            border-radius: 50%;
        }

        /* story-thumbnail 클래스의 스타일을 설정합니다. */
        .story-thumbnail {
            width: 100%;
            height: 100%;
            background-color: #eee;
            border-radius: 5px;
        }

        /* story-description 클래스의 스타일을 설정합니다. */
        .story-description {
            position: absolute;
            bottom: 0;
            width: 100%;
            padding: 16px;
        }

        /* story-description 클래스 안의 p 요소 스타일을 설정합니다. */
        .story-description p {
            width: 100%;
            display: -webkit-box;
            -webkit-line-clamp: 1;
            -webkit-box-orient: vertical;
            text-overflow: ellipsis;
            overflow: hidden;
        }

        /* 첫 번째가 아닌 모든 section 요소에 대한 스타일입니다. */
        section:not(:first-child) {
            margin-top: 32px;
        }
    </style>
</head>
<body>
    <!-- header 요소: 페이지 상단의 고정 헤더입니다. -->
    <header>
        <div class="left">
            <div class="logo">logo</div>
            <input type="text">
        </div>
        <div class="center">
            <div class="icons">
                <div class="icon">home</div>
                <div class="icon">friends</div>
                <div class="icon">camera</div>
                <div class="icon">game</div>
            </div>
        </div>
        <div class="right">
            <div class="profile">profile</div>
        </div>
    </header>

    <!-- aside 요소: 페이지의 사이드바입니다. -->
    <aside></aside>

    <!-- main 요소: 페이지의 주요 콘텐츠 영역입니다. -->
    <main>
        <!-- 첫 번째 section: 이야기 섹션입니다. -->
        <section id="story-section">
            <h2>Stories</h2>
            <div class="story-list">
                <!-- story 클래스: 각각의 이야기 카드입니다. -->
                <div class="story">
                    <div class="story-profile"></div>
                    <div class="story-thumbnail"></div>
                    <div class="story-description">
                        <p>
                            Lorem Ipsum Dolor asdfjkasdjflkads
                            loremasdjfajsdlfajds
                        </p>
                    </div>
                </div>

                <div class="story">
                    <div class="story-profile"></div>
                    <div class="story-thumbnail"></div>
                    <div class="story-description">
                        Lorem Ipsum Dolor
                    </div>
                </div>

                <div class="story">
                    <div class="story-profile"></div>
                    <div class="story-thumbnail"></div>
                    <div class="story-description">
                        Lorem Ipsum Dolor
                    </div>
                </div>
            </div>
        </section>

        <!-- 두 번째 section: 포스트 폼 섹션입니다. -->
        <section id="post-form-section">
            b
        </section>
    </main>
</body>
</html>
```

## 단계12: 세 번째 section 요소 추가

```js
<!-- 단계 12: 세 번째 section 요소 추가 -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <style>
        /* 모든 요소의 기본 여백과 패딩을 제거하고, 박스 크기 설정을 변경합니다. */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        /* body 요소의 배경 색상을 설정합니다. */
        body {
            background-color: #f5f5f5;
        }

        /* header 요소의 스타일을 설정합니다. */
        header {
            width: 100%;
            height: 90px;
            background-color: #fff;
            border-bottom: 2px solid #ddd;
            position: fixed;
            top: 0;
            left: 0;
            display: flex;
            justify-content: space-between;
            padding: 16px;
        }

        /* header 안의 left 클래스에 대한 스타일입니다. */
        header .left {
            display: flex;
            align-self: center;
        }

        /* header 안의 left 클래스 안의 input 요소에 대한 스타일입니다. */
        header .left input {
            height: 40px;
            margin-left: 20px;
        }

        /* header 안의 center 클래스에 대한 스타일입니다. */
        header .center {
            display: flex;
            align-items: center;
        }

        /* header 안의 center 클래스 안의 icons 클래스에 대한 스타일입니다. */
        header .center .icons {
            display: flex;
        }

        /* header 안의 center 클래스 안의 icons 클래스 안의 icon 클래스에 대한 스타일입니다. */
        header .center .icons .icon {
            padding: 0 16px;
        }

        /* header 안의 right 클래스에 대한 스타일입니다. */
        header .right {
            display: flex;
            align-items: center;
        }

        /* aside 요소의 스타일을 설정합니다. */
        aside {
            width: 240px;
            height: 100%;
            position: fixed;
            background-color: red;
            top: 90px;
            left: 0;
            border-right: 2px solid #ddd;
            opacity: 0.5;
        }

        /* main 요소의 스타일을 설정합니다. */
        main {
            width: 100%;
            padding-top: 122px;
            padding-bottom: 32px;
            padding-left: 272px;
            padding-right: 32px;
        }

        /* section 요소의 기본 스타일을 설정합니다. */
        section {
            padding: 32px;
            background-color: #fff;
        }

        /* #story-section 내의 h2 요소 스타일을 설정합니다. */
        #story-section h2 {
            margin-bottom: 4rem;
        }

        /* story-list 클래스의 스타일을 설정합니다. */
        .story-list {
            display: flex;
            gap: 1rem;
        }

        /* story 클래스의 스타일을 설정합니다. */
        .story {
            width: calc(100% / 3);
            height: 10rem;
            position: relative;
        }

        /* story-profile 클래스의 스타일을 설정합니다. */
        .story-profile {
            width: 4rem;
            height: 4rem;
            background-color: #1659bd;
            position: absolute;
            left: 50%;
            top: -2rem;
            transform: translateX(-50%);
            border-radius: 50%;
        }

        /* story-thumbnail 클래스의 스타일을 설정합니다. */
        .story-thumbnail {
            width: 100%;
            height: 100%;
            background-color: #eee;
            border-radius: 5px;
        }

        /* story-description 클래스의 스타일을 설정합니다. */
        .story-description {
            position: absolute;
            bottom: 0;
            width: 100%;
            padding: 16px;
        }

        /* story-description 클래스 안의 p 요소 스타일을 설정합니다. */
        .story-description p {
            width: 100%;
            display: -webkit-box;
            -webkit-line-clamp: 1;
            -webkit-box-orient: vertical;
            text-overflow: ellipsis;
            overflow: hidden;
        }

        /* 첫 번째가 아닌 모든 section 요소에 대한 스타일입니다. */
        section:not(:first-child) {
            margin-top: 32px;
        }
    </style>
</head>
<body>
    <!-- header 요소: 페이지 상단의 고정 헤더입니다. -->
    <header>
        <div class="left">
            <div class="logo">logo</div>
            <input type="text">
        </div>
        <div class="center">
            <div class="icons">
                <div class="icon">home</div>
                <div class="icon">friends</div>
                <div class="icon">camera</div>
                <div class="icon">game</div>
            </div>
        </div>
        <div class="right">
            <div class="profile">profile</div>
        </div>
    </header>

    <!-- aside 요소: 페이지의 사이드바입니다. -->
    <aside></aside>

    <!-- main 요소: 페이지의 주요 콘텐츠 영역입니다. -->
    <main>
        <!-- 첫 번째 section: 이야기 섹션입니다. -->
        <section id="story-section">
            <h2>Stories</h2>
            <div class="story-list">
                <!-- story 클래스: 각각의 이야기 카드입니다. -->
                <div class="story">
                    <div class="story-profile"></div>
                    <div class="story-thumbnail"></div>
                    <div class="story-description">
                        <p>
                            Lorem Ipsum Dolor asdfjkasdjflkads
                            loremasdjfajsdlfajds
                        </p>
                    </div>
                </div>

                <div class="story">
                    <div class="story-profile"></div>
                    <div class="story-thumbnail"></div>
                    <div class="story-description">
                        Lorem Ipsum Dolor
                    </div>
                </div>

                <div class="story">
                    <div class="story-profile"></div>
                    <div class="story-thumbnail"></div>
                    <div class="story-description">
                        Lorem Ipsum Dolor
                    </div>
                </div>
            </div>
        </section>

        <!-- 두 번째 section: 포스트 폼 섹션입니다. -->
        <section id="post-form-section">
            b
        </section>

        <!-- 세 번째 section: 기사 섹션입니다. -->
        <section id="article-section">
            c
        </section>
    </main>
</body>
</html>
```

## 단계13: 세 번째 section 스타일 추가

```js
<!-- 단계 13: 세 번째 section 스타일 추가 -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <style>
        /* 모든 요소의 기본 여백과 패딩을 제거하고, 박스 크기 설정을 변경합니다. */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        /* body 요소의 배경 색상을 설정합니다. */
        body {
            background-color: #f5f5f5;
        }

        /* header 요소의 스타일을 설정합니다. */
        header {
            width: 100%;
            height: 90px;
            background-color: #fff;
            border-bottom: 2px solid #ddd;
            position: fixed;
            top: 0;
            left: 0;
            display: flex;
            justify-content: space-between;
            padding: 16px;
        }

        /* header 안의 left 클래스에 대한 스타일입니다. */
        header .left {
            display: flex;
            align-self: center;
        }

        /* header 안의 left 클래스 안의 input 요소에 대한 스타일입니다. */
        header .left input {
            height: 40px;
            margin-left: 20px;
        }

        /* header 안의 center 클래스에 대한 스타일입니다. */
        header .center {
            display: flex;
            align-items: center;
        }

        /* header 안의 center 클래스 안의 icons 클래스에 대한 스타일입니다. */
        header .center .icons {
            display: flex;
        }

        /* header 안의 center 클래스 안의 icons 클래스 안의 icon 클래스에 대한 스타일입니다. */
        header .center .icons .icon {
            padding: 0 16px;
        }

        /* header 안의 right 클래스에 대한 스타일입니다. */
        header .right {
            display: flex;
            align-items: center;
        }

        /* aside 요소의 스타일을 설정합니다. */
        aside {
            width: 240px;
            height: 100%;
            position: fixed;
            background-color: red;
            top: 90px;
            left: 0;
            border-right: 2px solid #ddd;
            opacity: 0.5;
        }

        /* main 요소의 스타일을 설정합니다. */
        main {
            width: 100%;
            padding-top: 122px;
            padding-bottom: 32px;
            padding-left: 272px;
            padding-right: 32px;
        }

        /* section 요소의 기본 스타일을 설정합니다. */
        section {
            padding: 32px;
            background-color: #fff;
        }

        /* #story-section 내의 h2 요소 스타일을 설정합니다. */
        #story-section h2 {
            margin-bottom: 4rem;
        }

        /* story-list 클래스의 스타일을 설정합니다. */
        .story-list {
            display: flex;
            gap: 1rem;
        }

        /* story 클래스의 스타일을 설정합니다. */
        .story {
            width: calc(100% / 3);
            height: 10rem;
            position: relative;
        }

        /* story-profile 클래스의 스타일을 설정합니다. */
        .story-profile {
            width: 4rem;
            height: 4rem;
            background-color: #1659bd;
            position: absolute;
            left: 50%;
            top: -2rem;
            transform: translateX(-50%);
            border-radius: 50%;
        }

        /* story-thumbnail 클래스의 스타일을 설정합니다. */
        .story-thumbnail {
            width: 100%;
            height: 100%;
            background-color: #eee;
            border-radius: 5px;
        }

        /* story-description 클래스의 스타일을 설정합니다. */
        .story-description {
            position: absolute;
            bottom: 0;
            width: 100%;
            padding: 16px;
        }

        /* story-description 클래스 안의 p 요소 스타일을 설정합니다. */
        .story-description p {
            width: 100%;
            display: -webkit-box;
            -webkit-line-clamp: 1;
            -webkit-box-orient: vertical;
            text-overflow: ellipsis;
            overflow: hidden;
        }

        /* 첫 번째가 아닌 모든 section 요소에 대한 스타일입니다. */
        section:not(:first-child) {
            margin-top: 32px;
        }

        /* 추가된 세 번째 section 요소의 스타일을 설정합니다. */
        #article-section {
            background-color: #f0f0f0;
        }
    </style>
</head>
<body>
    <!-- header 요소: 페이지 상단의 고정 헤더입니다. -->
    <header>
        <div class="left">
            <div class="logo">logo</div>
            <input type="text">
        </div>
        <div class="center">
            <div class="icons">
                <div class="icon">home</div>
                <div class="icon">friends</div>
                <div class="icon">camera</div>
                <div class="icon">game</div>
            </div>
        </div>
        <div class="right">
            <div class="profile">profile</div>
        </div>
    </header>

    <!-- aside 요소: 페이지의 사이드바입니다. -->
    <aside></aside>

    <!-- main 요소: 페이지의 주요 콘텐츠 영역입니다. -->
    <main>
        <!-- 첫 번째 section: 이야기 섹션입니다. -->
        <section id="story-section">
            <h2>Stories</h2>
            <div class="story-list">
                <!-- story 클래스: 각각의 이야기 카드입니다. -->
                <div class="story">
                    <div class="story-profile"></div>
                    <div class="story-thumbnail"></div>
                    <div class="story-description">
                        <p>
                            Lorem Ipsum Dolor asdfjkasdjflkads
                            loremasdjfajsdlfajds
                        </p>
                    </div>
                </div>

                <div class="story">
                    <div class="story-profile"></div>
                    <div class="story-thumbnail"></div>
                    <div class="story-description">
                        Lorem Ipsum Dolor
                    </div>
                </div>

                <div class="story">
                    <div class="story-profile"></div>
                    <div class="story-thumbnail"></div>
                    <div class="story-description">
                        Lorem Ipsum Dolor
                    </div>
                </div>
            </div>
        </section>

        <!-- 두 번째 section: 포스트 폼 섹션입니다. -->
        <section id="post-form-section">
            b
        </section>

        <!-- 세 번째 section: 기사 섹션입니다. -->
        <section id="article-section">
            c
        </section>
    </main>
</body>
</html>
```

## 단계14: 추가 CSS 스타일 설정

```js
<!-- 단계 14: 추가 CSS 스타일 설정 -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <style>
        /* 모든 요소의 기본 여백과 패딩을 제거하고, 박스 크기 설정을 변경합니다. */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        /* body 요소의 배경 색상을 설정합니다. */
        body {
            background-color: #f5f5f5;
        }

        /* header 요소의 스타일을 설정합니다. */
        header {
            width: 100%;
            height: 90px;
            background-color: #fff;
            border-bottom: 2px solid #ddd;
            position: fixed;
            top: 0;
            left: 0;
            display: flex;
            justify-content: space-between;
            padding: 16px;
        }

        /* header 안의 left 클래스에 대한 스타일입니다. */
        header .left {
            display: flex;
            align-self: center;
        }

        /* header 안의 left 클래스 안의 input 요소에 대한 스타일입니다. */
        header .left input {
            height: 40px;
            margin-left: 20px;
        }

        /* header 안의 center 클래스에 대한 스타일입니다. */
        header .center {
            display: flex;
            align-items: center;
        }

        /* header 안의 center 클래스 안의 icons 클래스에 대한 스타일입니다. */
        header .center .icons {
            display: flex;
        }

        /* header 안의 center 클래스 안의 icons 클래스 안의 icon 클래스에 대한 스타일입니다. */
        header .center .icons .icon {
            padding: 0 16px;
        }

        /* header 안의 right 클래스에 대한 스타일입니다. */
        header .right {
            display: flex;
            align-items: center;
        }

        /* aside 요소의 스타일을 설정합니다. */
        aside {
            width: 240px;
            height: 100%;
            position: fixed;
            background-color: red;
            top: 90px;
            left: 0;
            border-right: 2px solid #ddd;
            opacity: 0.5;
        }

        /* main 요소의 스타일을 설정합니다. */
        main {
            width: 100%;
            padding-top: 122px;
            padding-bottom: 32px;
            padding-left: 272px;
            padding-right: 32px;
        }

        /* section 요소의 기본 스타일을 설정합니다. */
        section {
            padding: 32px;
            background-color: #fff;
        }

        /* #story-section 내의 h2 요소 스타일을 설정합니다. */
        #story-section h2 {
            margin-bottom: 4rem;
        }

        /* story-list 클래스의 스타일을 설정합니다. */
        .story-list {
            display: flex;
            gap: 1rem;
        }

        /* story 클래스의 스타일을 설정합니다. */
        .story {
            width: calc(100% / 3);
            height: 10rem;
            position: relative;
        }

        /* story-profile 클래스의 스타일을 설정합니다. */
        .story-profile {
            width: 4rem;
            height: 4rem;
            background-color: #1659bd;
            position: absolute;
            left: 50%;
            top: -2rem;
            transform: translateX(-50%);
            border-radius: 50%;
        }

        /* story-thumbnail 클래스의 스타일을 설정합니다. */
        .story-thumbnail {
            width: 100%;
            height: 100%;
            background-color: #eee;
            border-radius: 5px;
        }

        /* story-description 클래스의 스타일을 설정합니다. */
        .story-description {
            position: absolute;
            bottom: 0;
            width: 100%;
            padding: 16px;
        }

        /* story-description 클래스 안의 p 요소 스타일을 설정합니다. */
        .story-description p {
            width: 100%;
            display: -webkit-box;
            -webkit-line-clamp: 1;
            -webkit-box-orient: vertical;
            text-overflow: ellipsis;
            overflow: hidden;
        }

        /* 첫 번째가 아닌 모든 section 요소에 대한 스타일입니다. */
        section:not(:first-child) {
            margin-top: 32px;
        }

        /* 추가된 세 번째 section 요소의 스타일을 설정합니다. */
        #article-section {
            background-color: #f0f0f0;
            padding: 32px;
        }
    </style>
</head>
<body>
    <!-- header 요소: 페이지 상단의 고정 헤더입니다. -->
    <header>
        <div class="left">
            <div class="logo">logo</div>
            <input type="text">
        </div>
        <div class="center">
            <div class="icons">
                <div class="icon">home</div>
                <div class="icon">friends</div>
                <div class="icon">camera</div>
                <div class="icon">game</div>
            </div>
        </div>
        <div class="right">
            <div class="profile">profile</div>
        </div>
    </header>

    <!-- aside 요소: 페이지의 사이드바입니다. -->
    <aside></aside>

    <!-- main 요소: 페이지의 주요 콘텐츠 영역입니다. -->
    <main>
        <!-- 첫 번째 section: 이야기 섹션입니다. -->
        <section id="story-section">
            <h2>Stories</h2>
            <div class="story-list">
                <!-- story 클래스: 각각의 이야기 카드입니다. -->
                <div class="story">
                    <div class="story-profile"></div>
                    <div class="story-thumbnail"></div>
                    <div class="story-description">
                        <p>
                            Lorem Ipsum Dolor asdfjkasdjflkads
                            loremasdjfajsdlfajds
                        </p>
                    </div>
                </div>

                <div class="story">
                    <div class="story-profile"></div>
                    <div class="story-thumbnail"></div>
                    <div class="story-description">
                        Lorem Ipsum Dolor
                    </div>
                </div>

                <div class="story">
                    <div class="story-profile"></div>
                    <div class="story-thumbnail"></div>
                    <div class="story-description">
                        Lorem Ipsum Dolor
                    </div>
                </div>
            </div>
        </section>

        <!-- 두 번째 section: 포스트 폼 섹션입니다. -->
        <section id="post-form-section">
            b
        </section>

        <!-- 세 번째 section: 기사 섹션입니다. -->
        <section id="article-section">
            c
        </section>
    </main>
</body>
</html>
```

## 단계15: 최종 HTML 구조 확인 및 정리

```js
<!-- 단계 15: 최종 HTML 구조 확인 및 정리 -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <style>
        /* 모든 요소의 기본 여백과 패딩을 제거하고, 박스 크기 설정을 변경합니다. */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        /* body 요소의 배경 색상을 설정합니다. */
        body {
            background-color: #f5f5f5;
        }

        /* header 요소의 스타일을 설정합니다. */
        header {
            width: 100%;
            height: 90px;
            background-color: #fff;
            border-bottom: 2px solid #ddd;
            position: fixed;
            top: 0;
            left: 0;
            display: flex;
            justify-content: space-between;
            padding: 16px;
        }

        /* header 안의 left 클래스에 대한 스타일입니다. */
        header .left {
            display: flex;
            align-self: center;
        }

        /* header 안의 left 클래스 안의 input 요소에 대한 스타일입니다. */
        header .left input {
            height: 40px;
            margin-left: 20px;
        }

        /* header 안의 center 클래스에 대한 스타일입니다. */
        header .center {
            display: flex;
            align-items: center;
        }

        /* header 안의 center 클래스 안의 icons 클래스에 대한 스타일입니다. */
        header .center .icons {
            display: flex;
        }

        /* header 안의 center 클래스 안의 icons 클래스 안의 icon 클래스에 대한 스타일입니다. */
        header .center .icons .icon {
            padding: 0 16px;
        }

        /* header 안의 right 클래스에 대한 스타일입니다. */
        header .right {
            display: flex;
            align-items: center;
        }

        /* aside 요소의 스타일을 설정합니다. */
        aside {
            width: 240px;
            height: 100%;
            position: fixed;
            background-color: red;
            top: 90px;
            left: 0;
            border-right: 2px solid #ddd;
            opacity: 0.5;
        }

        /* main 요소의 스타일을 설정합니다. */
        main {
            width: 100%;
            padding-top: 122px;
            padding-bottom: 32px;
            padding-left: 272px;
            padding-right: 32px;
        }

        /* section 요소의 기본 스타일을 설정합니다. */
        section {
            padding: 32px;
            background-color: #fff;
        }

        /* #story-section 내의 h2 요소 스타일을 설정합니다. */
        #story-section h2 {
            margin-bottom: 4rem;
        }

        /* story-list 클래스의 스타일을 설정합니다. */
        .story-list {
            display: flex;
            gap: 1rem;
        }

        /* story 클래스의 스타일을 설정합니다. */
        .story {
            width: calc(100% / 3);
            height: 10rem;
            position: relative;
        }

        /* story-profile 클래스의 스타일을 설정합니다. */
        .story-profile {
            width: 4rem;
            height: 4rem;
            background-color: #1659bd;
            position: absolute;
            left: 50%;
            top: -2rem;
            transform: translateX(-50%);
            border-radius: 50%;
        }

        /* story-thumbnail 클래스의 스타일을 설정합니다. */
        .story-thumbnail {
            width: 100%;
            height: 100%;
            background-color: #eee;
            border-radius: 5px;
        }

        /* story-description 클래스의 스타일을 설정합니다. */
        .story-description {
            position: absolute;
            bottom: 0;
            width: 100%;
            padding: 16px;
        }

        /* story-description 클래스 안의 p 요소 스타일을 설정합니다. */
        .story-description p {
            width: 100%;
            display: -webkit-box;
            -webkit-line-clamp: 1;
            -webkit-box-orient: vertical;
            text-overflow: ellipsis;
            overflow: hidden;
        }

        /* 첫 번째가 아닌 모든 section 요소에 대한 스타일입니다. */
        section:not(:first-child) {
            margin-top: 32px;
        }

        /* 추가된 세 번째 section 요소의 스타일을 설정합니다. */
        #article-section {
            background-color: #f0f0f0;
            padding: 32px;
        }
    </style>
</head>
<body>
    <!-- header 요소: 페이지 상단의 고정 헤더입니다. -->
    <header>
        <div class="left">
            <div class="logo">logo</div>
            <input type="text">
        </div>
        <div class="center">
            <div class="icons">
                <div class="icon">home</div>
                <div class="icon">friends</div>
                <div class="icon">camera</div>
                <div class="icon">game</div>
            </div>
        </div>
        <div class="right">
            <div class="profile">profile</div>
        </div>
    </header>

    <!-- aside 요소: 페이지의 사이드바입니다. -->
    <aside></aside>

    <!-- main 요소: 페이지의 주요 콘텐츠 영역입니다. -->
    <main>
        <!-- 첫 번째 section: 이야기 섹션입니다. -->
        <section id="story-section">
            <h2>Stories</h2>
            <div class="story-list">
                <!-- story 클래스: 각각의 이야기 카드입니다. -->
                <div class="story">
                    <div class="story-profile"></div>
                    <div class="story-thumbnail"></div>
                    <div class="story-description">
                        <p>
                            Lorem Ipsum Dolor asdfjkasdjflkads
                            loremasdjfajsdlfajds
                        </p>
                    </div>
                </div>

                <div class="story">
                    <div class="story-profile"></div>
                    <div class="story-thumbnail"></div>
                    <div class="story-description">
                        Lorem Ipsum Dolor
                    </div>
                </div>

                <div class="story">
                    <div class="story-profile"></div>
                    <div class="story-thumbnail"></div>
                    <div class="story-description">
                        Lorem Ipsum Dolor
                    </div>
                </div>
            </div>
        </section>

        <!-- 두 번째 section: 포스트 폼 섹션입니다. -->
        <section id="post-form-section">
            b
        </section>

        <!-- 세 번째 section: 기사 섹션입니다. -->
        <section id="article-section">
            c
        </section>
    </main>
</body>
</html>
```
