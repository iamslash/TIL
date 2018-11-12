- [Abstract](#abstract)
- [Materials](#materials)
- [Concepts](#concepts)
    - [Overview](#overview)
    - [Androidmanifest.xml](#androidmanifestxml)
    - [Activity](#activity)
    - [Fragment](#fragment)
    - [Intent](#intent)
    - [Service](#service)
    - [Broadcast receiver](#broadcast-receiver)
    - [Dangerous permissions](#dangerous-permissions)
- [Tools](#tools)
    - [jarsigner](#jarsigner)

-------------------------------------------------------------------------------

# Abstract

android studio 에 대해 정리한다.

# Materials

* [doit 안드로이드](https://www.youtube.com/playlist?list=PLG7te9eYUi7sq701GghpoSKe-jbkx9NIF)
  * 킹왕짱 한글 동영상
  * [src](http://147.46.109.80:9090/board/board-list.do?boardId=doitandroid)
* [될 때까지 안드로이드 @ yes24](http://www.yes24.com/24/goods/59298937)
  * [video @ youtube](https://www.youtube.com/watch?v=MjtlPTUUL74&list=PLxTmPHxRH3VWSF7kMcsIaTglWUJZpWeQ9)
  * [src @ github](https://github.com/junsuk5/android-first-book)
* [sample android apps @ github](https://github.com/codepath/android_guides/wiki/Sample-Android-Apps)
  * 유용한 샘플 모음
* [android-architecture-components @ github](https://github.com/googlesamples/android-architecture-components)
  * 구글에서 제공하는 샘플
* [awesome android @ github](https://github.com/JStumpp/awesome-android)
* [awesome android-ui @ github](https://github.com/wasabeef/awesome-android-ui)

# Concepts

## Overview

## Androidmanifest.xml

app이 설치될때 android os가 알아야할 정보들이 저장된 xml.

## Activity

화면을 표현하는 class.

## Fragment

Activity보다 작은 단위의 class.

## Intent

Activity, Service등이 서로 통신할 수 있는 수단. 일종의 메시지이다.

## Service

화면은 없고 로직을 표현하는 class. android os가 특별 관리 해주기
때문에 종료하면 다시 실행된다.

## Broadcast receiver

android os로 부터 global event를 수신할 수 있는 클래스. 역시 UI는
없다.

## Dangerous permissions

normal permission보다 위험하여 유저의 허가가 필요한 권한이다. 
[리스트](https://developer.android.com/guide/topics/permissions/overview.html#permission-groups)

# Tools

## jarsigner

* a.apk signing하기

```
jarsigner -verbose -keystore dma.keystore -storepass xxxxxx -keypass xxxxxx a.apk iamslash
```

* a.apk가 signining되었는지 확인하기

```
jarsigner -verify -verbose -certs a.apk
```

* a.apk를 재서명 하기

```
zip d a.apk META-INF/\*
jarsigner -verbose -keystore dma.keystore -storepass xxxxxx -keypass xxxxxx a.apk iamslash
```

* a.keystore의 fingerprint출력하기

```
keytool -list -v -keystore a.keystore -alias iamslash -storepass xxxxx -keypass xxxxx
```

* zip algin하기

```
zipalign -v 4 a.apk a.apk
```
