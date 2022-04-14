- [Materials](#materials)
- [Basic](#basic)
  - [Unity Addressable Asset System](#unity-addressable-asset-system)
  - [Task](#task)
  - [UniTask](#unitask)
  - [2D Character Setup](#2d-character-setup)
  - [Pixel Perfect](#pixel-perfect)

---

# Materials

* [Awesome Unity Open Source on GitHub (800+) @ github](https://github.com/baba-s/awesome-unity-open-source-on-github)

# Basic

## Unity Addressable Asset System

* [Unity Addressable Asset System 기본 개념](https://young-94.tistory.com/47)

## Task

* [C# Tasks vs. Unity Jobs](https://www.jacksondunstan.com/articles/4926)

---

`System.threading.Tasks.Task` 는 `.net4.0` 부터 지원된다. thread 를 추상화한 class 이다. asynchronous programming 을 위해 필요하다. 그러나 Task 는 무겁다. 가벼운 [UniTask](https://github.com/Cysharp/UniTask.git) 를 사용한다.

## UniTask

[UniTask](https://github.com/Cysharp/UniTask.git) Provides an efficient allocation free async/await integration for Unity.

## 2D Character Setup

[2D Character Setup](unity_2d_char_setup.md)

## Pixel Perfect

* [[개발일지] Frostory 픽셀 퍼펙트 세팅](https://gall.dcinside.com/mgallery/board/view/?id=game_dev&no=55170)

Runtime 에 Image 가 정상적으로 보이지않는 현상을 Pixel Perfect 설정으로 해결해야 한다.
