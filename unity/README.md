- [Materials](#materials)
- [Basic](#basic)
  - [Unity Addressable Asset System](#unity-addressable-asset-system)
  - [Task](#task)
  - [UniTask](#unitask)
  - [2D Character Setup](#2d-character-setup)
  - [Pixel Perfect](#pixel-perfect)
- [Advanced](#advanced)
  - [Right to left TextMesh Pro](#right-to-left-textmesh-pro)

---

# Materials

* [Awesome Unity Open Source on GitHub (800+) @ github](https://github.com/baba-s/awesome-unity-open-source-on-github)

# Basic

## Unity Addressable Asset System

[Addressables](unity_addressables.md)

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
* [유나이트 서울 2020 - 픽셀 + 탑다운 + 칼부림 액션게임을 개발하고 싶을 땐 이걸 보고 참아보자. Track1-3 | youtube](https://www.youtube.com/watch?v=J-cfVwYNQSk)

Runtime 에 Image 가 정상적으로 보이지않는 현상을 Pixel Perfect 설정으로 해결해야 한다.

Pixel Perfect 설정은 어렵다. 다양한 방법을 연구해야 한다.

# Advanced

## Right to left TextMesh Pro

[RTLTMPro | github](https://github.com/pnarimani/RTLTMPro#installation)

Text Mesh Pro 가 Arabic 을 지원하지 못한다. 그것을 보완한 것.
