<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Abstract](#abstract)
- [Materials](#materials)
- [Concepts](#concepts)
    - [Activity](#activity)
    - [Intent](#intent)
    - [Service](#service)
    - [Androidmanifest.xml](#androidmanifestxml)

<!-- markdown-toc end -->

-------------------------------------------------------------------------------

# Abstract

android studio 에 대해 정리한다.

# Materials

* [doit 안드로이드](https://www.youtube.com/playlist?list=PLG7te9eYUi7sq701GghpoSKe-jbkx9NIF)
  * 킹왕짱 한글 동영상
  * [src](http://147.46.109.80:9090/board/board-list.do?boardId=doitandroid)

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
