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
  - [Modular Architecture](#modular-architecture)
  - [Dynamic Library vs Staic Library](#dynamic-library-vs-staic-library)

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
  * 앱에 부여할 고유한 ID 를 말한다. (ex. `com.iamslash.awsome`)
* Certificate
  * ios application 을 Certificate 의 비밀키로 signing 해야 App Store 에 제출할
    수 있다. 
* Provisioning Profile
  * AppID, DeviceID, Certificate 의 매핑정보가 들어있다.
  * Enterprise Profile 을 만들면 AppID, Certificate 만 필요하다. 즉, DeviceID
    없이 ios app 을 배포할 수 있다.

ios application (ipa) 배포하는 방법들은 다음과 같다.

* Submitting to App Store 
* Copy ipa to specific devices
* Firebase enterprise distribution
* Testflight
  * App Store 심사가 필요하다. 불편하다.

Firebase 를 이용하면 Enterprise Provisioning Profile 을 생성하여 DeviceID 없이
배포할 수 있다. 주로 test app 배포 용도로 사용한다.

# Advanced

## Background Tasks

* [Background Tasks | apple](https://developer.apple.com/documentation/backgroundtasks)
* [Advances in App Background Execution | wwdc2019](https://developer.apple.com/videos/play/wwdc2019/707/)
  * [src](https://developer.apple.com/documentation/backgroundtasks/refreshing_and_maintaining_your_app_using_background_tasks)
* [Background execution demystified | wwdc2020](https://developer.apple.com/videos/play/wwdc2020/10063)
* [[iOS] BackgroundTasks Framework 간단 정리](https://lemon-dev.tistory.com/entry/iOS-BackgroundTask-Framework-%EA%B0%84%EB%8B%A8-%EC%A0%95%EB%A6%AC)
* [How to manage background tasks with the Task Scheduler in iOS 13?](https://snow.dog/blog/how-to-manage-background-tasks-with-the-task-scheduler-in-ios-13)

----

iOS 는 `Background Task Completion` 을 제공한다. iOS 13 이전에도 있었던 것 같다.
foreground 의 app 이 background 로 바뀌면 하던 일을 마무리할 수 있다. foreground
에서 background 로 바뀔 때 background 에서 한번 실행된다.

iOS 13 부터 `BGAppRefreshTask`, `BGProcessingTask` 를 제공한다. 

`BGAppRefreshTask` - 비교적 가벼운 logic 이 적당하다. app 이 다음 번에
foreground 가 되었을 때 UI 를 미리 업데이트하는 logic 에 적당하다. 예를 들어
user 가 획득한 점수를 원격으로부터 받아오는 것이 해당된다.

`BGProcessingTask` - 비교적 무거운 logic 이 적당하다. 예를 들어 아주 긴 파일을
다운로드하는 것이 해당된다. 

두 가지 방식에 대해 cancel 조건이 다를 것이다. iOS 가 언제 background task 를
취소할지 예측할 수 없다. 언제 실행될지도 예측할 수 없다. UX 를 신경써야 한다.

테스트 방법은 [Starting and Terminating Tasks During Development |
apple](https://developer.apple.com/documentation/backgroundtasks/starting_and_terminating_tasks_during_development)
을 참고한다. 

`BGTaskScheduler.shared.submit()` 에 break point 를 설정한다. app 의 실행이 멈출
때 LLDB prompt 에 다음과 같은 command line 을 입력하여 background task 를 시작
혹은 종료할 수 있다. test 를 위해 AppStore 제출과 관계없는 code 를 작성할 필요가 있다.

```
LLDB> e -l objc -- (void)[[BGTaskScheduler sharedScheduler] _simulateLaunchForTaskWithIdentifier:@"TASK_IDENTIFIER"]

LLDB> e -l objc -- (void)[[BGTaskScheduler sharedScheduler] _simulateExpirationForTaskWithIdentifier:@"TASK_IDENTIFIER"]
```

## Modular Architecture

- [Modular Architecture on iOS and macOS | github](https://github.com/CyrilCermak/modular_architecture_on_ios/tree/master)
  - `Modular Architecture on iOS and macOS` ebook src 

## Dynamic Library vs Staic Library

Apple의 생태계에서 라이브러리는 크게 두 가지 방식으로 만들 수 있습니다: **정적 링크 라이브러리(Static Library)**와 **동적 링크 라이브러리(Dynamic Library)**입니다. 동적 링크 라이브러리는 이전에 Cocoa Touch Framework로 불렸지만, 이제는 단순히 Framework로 불립니다.

Swift 생태계에서 사용되는 Swift Package와 **Swift Package Manager(SPM)**도 주목할 필요가 있습니다. SPM은 주로 소스 코드를 공유하는 데 사용되며, 정적 링크를 기본으로 하지만 개발자가 선택하면 동적 링크도 가능합니다. 또한, SPM을 통해 XCFramework를 공유하는 것도 일반적이며, 이 경우 SPM은 컴파일된 바이너리를 감싸는 역할을 합니다.

1. 동적 링크 라이브러리
- Dylib: Mach-O 형식의 바이너리를 가지는 동적 라이브러리 (`.dylib`).
- Framework: 바이너리와 실행 중에 필요한 추가 리소스를 포함하는 번들 (`.framework`).
- TBDs: 텍스트 기반의 동적 라이브러리 스텁으로, 기기의 시스템에 존재하는 바이너리를 참조하며, 가벼운 SDK로 제공됨 (`.tbd`).
- XCFramework: Xcode 11부터 도입된 라이브러리로, macOS, iOS, iOS 시뮬레이터, watchOS 등 여러 플랫폼용 프레임워크를 묶을 수 있음 (`.xcframework`).

1. 정적 링크 라이브러리
- Archive: 컴파일된 객체 파일을 포함하는 정적 아카이브 (`.a`).
- Framework: 정적 바이너리 또는 정적 아카이브와 필요한 리소스를 포함하는 프레임워크 (`.framework`).
- XCFramework: 동적 링크와 동일하게 정적 링크에서도 XCFramework를 사용할 수 있음 (`.xcframework`).
