- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
  - [Overview](#overview)
  - [Apple Developer Program](#apple-developer-program)
  - [Xcode Application Tutorial](#xcode-application-tutorial)
  - [Pod Application Tutorial](#pod-application-tutorial)
  - [Publish iOS Library](#publish-ios-library)
  - [Distribution](#distribution)
- [Advanced](#advanced)
  - [Background Tasks](#background-tasks)

----

# Abstract

ios app development 에 대해 정리한다.

# Materials

* [ Do it! 스위프트로 아이폰 앱 만들기 입문 - 개정 5판](http://www.yes24.com/Product/Goods/96825837)
  * [src](https://github.com/doitswift/example)
* [Awesome iOS](https://github.com/vsouza/awesome-ios)
* [iOS Example](https://iosexample.com/)

# Basic

## Overview

다음과 같은 순서대로 prerequisites 가 필요하다.

* [Developer | apple](https://developer.apple.com/account/) 에 가입해야 한다.
* [Swift](/swift/README.md) 를 학습해야 한다.
* [xcode](/xcode/README.md) 를 사용해야 한다.

## Apple Developer Program

> [Developer | apple](https://developer.apple.com/account/)

무료가입은 가능하지만 Simulator Test 만 가능하다. Device Test 는 못한다.

다음은 iOS Simulator 가 지원하는 기능의 목록이다. 

| iOS Simulator can | iOS Simulator can't |
|---|---|
| 좌, 우 회전 | GPS 의 실제값 |
| 흔들기 | 전화착신 이벤트 |
| 멀티 터치 | 카메라 |
| GPS 의 가상값 | 가속도 센서 |

## Xcode Application Tutorial

[ios Hello World](iosappdevelopment_helloworld.md)

## Pod Application Tutorial

[cocoapods](/cocoapods/README.md#using-pod-application-create)

## Publish iOS Library

[cocoapods](/cocoapods/README.md#using-pod-lib-create)

## Distribution

ios application 배포 방법을 이해하기 위해서는 다음과 같은 개념들을 알아야 한다.

* DeviceID
  * iPhone 마다 부여된 고유한 ID 를 말한다.
* AppID
* Certificate
  * ios application 을 Certificate 의 비밀키로 signing 해야 App Store 에 제출할
    수 있다. 
* Provisioning Profile
  * AppID, DeviceID, Certificate 의 매핑정보가 들어있다.
  * Enterprise Profile 을 만들면 AppID, Certificate 만 필요하다. 즉, DeviceID
    없이 ios app 을 배포할 수 있다.

ios application (ipa) 배포하는 방법은 다음과 같다.

* Submitting to App Store 
* Copy ipa to specific devices
* Firebase enterprise distribution
* Testflight
  * App Store 심사가 필요하다. 불편하다.

Firebase 를 이용하면 Enterprise Provisioning Profile 을 생성하여 DeviceID 없이
배포할 수 있다. 주로 test app 배포 용도로 사용한다.

# Advanced

## Background Tasks

* [[iOS] BackgroundTasks Framework 간단 정리](https://lemon-dev.tistory.com/entry/iOS-BackgroundTask-Framework-%EA%B0%84%EB%8B%A8-%EC%A0%95%EB%A6%AC)
* [How to manage background tasks with the Task Scheduler in iOS 13?](https://snow.dog/blog/how-to-manage-background-tasks-with-the-task-scheduler-in-ios-13)

----

iOS 13 부터 `BGAppRefreshTask`, `BGProcessingTask` 를 이용하면 Background Task 구현이 가능하다. `BGAppRefreshTask` 는 가벼운 것 `BGProcessingTask` 는 무거운 것에 실행하자???

`BGAppRefreshTask` - An object representing a short task typically used to refresh content that’s run while the app is in the background.

`BGProcessingTask` - A time-consuming processing task that runs while the app is in the background.

iOS 가 언제 background task 를 취소할지 예측할 수 없다. 우선순위가 낮아 언제 실행될지 예측할 수 없다. UX 를 신경써야 한다.
