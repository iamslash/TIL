- [Materials](#materials)
- [Basic](#basic)
  - [Android Application Tutorial](#android-application-tutorial)
  - [Android Libary Tutorial](#android-libary-tutorial)
  - [Android Application Release](#android-application-release)
- [Advanced](#advanced)
  - [Adnroid Device Connection](#adnroid-device-connection)
  - [Android UI Debugging](#android-ui-debugging)
  - [Android Device Screen Mirroring](#android-device-screen-mirroring)

----

# Materials

* [Codelab Android Compose | github](https://github.com/android/codelab-android-compose)
* [Do it! 깡샘의 안드로이드 앱 프로그래밍 with 코틀린 [개정판]](http://easyspub.co.kr/20_Menu/BookView/489/PUB)
  * [src](https://kkangsnote.tistory.com/138?category=1048735)
  * [youtube](https://www.youtube.com/watch?v=zP5rl8NtZ6U&list=PLYlZbv3fX7WvaWMB9zRgbO7Hzf3MRgrIf&index=2)
  * [ssams](https://ssamz.com/lecture_view.php?LectureStep1=51&LectureSeq=18)
* [Compose codelab](https://developer.android.com/courses/android-basics-compose/course?hl=ko)
* [android | github](https://github.com/android)
* [now in android | github](https://github.com/android/nowinandroid)
  * A fully functional Android app built entirely with Kotlin and Jetpack Compose
* [compose samples | github](https://github.com/android/compose-samples)
  * Official Jetpack Compose samples.

# Basic

## Android Application Tutorial

Android Studio 로 Application Project 를 만들어 보자.

[Hello World Tutorial](androidappdevelopment_helloworld.md)

## Android Libary Tutorial

* [안드로이드 내가 만든 라이브러리 배포하는법1(로컬)](https://dog-footprint.tistory.com/4?category=857506)
  * [안드로이드 내가 만든 라이브러리 배포하는법2(원격)](https://dog-footprint.tistory.com/5?category=857506)

[MyLib Tutorial](androidappdevelopment_library.md)

## Android Application Release

Android Application 을 PlayStore 에 출시해보자.

[Android Application Release](androidapprelease.md)

# Advanced

## Adnroid Device Connection

* Check developer mode in a device.
* Check USB Debugging mode in a device.
* Check USB cables.
* Restart AndroidStudio.
* Restart adb servers.
  ```bash
  $ adb kill-server
  $ adb start-server
  ``` 
* Reboot macbook.

## Android UI Debugging

- [Flipper](https://github.com/facebook/flipper)

## Android Device Screen Mirroring

- [scrcpy](https://github.com/Genymobile/scrcpy)
