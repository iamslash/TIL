- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
  - [Overview](#overview)
  - [Apple Developer Program](#apple-developer-program)
  - [Xcode Application Tutorial](#xcode-application-tutorial)
  - [Pod Application Tutorial](#pod-application-tutorial)
  - [Publish iOS Library](#publish-ios-library)

----

# Abstract

ios app development 에 대해 정리한다.

# Materials

* [ Do it! 스위프트로 아이폰 앱 만들기 입문 - 개정 5판](http://www.yes24.com/Product/Goods/96825837)
  * [src](https://github.com/doitswift/example)
* [Awesome iOS](https://github.com/vsouza/awesome-ios)

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
