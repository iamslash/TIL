- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
  - [Install](#install)
  - [Using Pod Application Create](#using-pod-application-create)
  - [Using Pod Lib Create](#using-pod-lib-create)
  - [Create POD](#create-pod)

----

# Abstract

CocoaPods is a dependency manager for Swift and Objective-C Cocoa projects. 

# Materials

* [CocoaPods Guides](https://guides.cocoapods.org/)
* [An Introduction to CocoaPods (Route 85) | youtube](https://www.youtube.com/watch?v=iEAjvNRdZa0)
  * 명쾌하게 cocoapods 사용방법을 설명한다.

# Basic

## Install

```bash
$ sudo gem install cocoapods
$ pod setup
$ pod search AFNetwork
```

## Using Pod Application Create

xcode 로 `testHello.xcodeproject` 를 생성한다.

다음과 같이 `Podfile` 을 생성한다.

```bash
$ pod init

$ vim Podfile
platform :ios, '8.0'
use_frameworks!

target 'testHello' do
  pod 'AFNetworking', '~> 2.6'
  pod 'ORStackView', '~> 3.0'
end

$ tree .
.
├── Podfile
├── testHello
│   ├── AppDelegate.swift
│   ├── Assets.xcassets
│   │   ├── AccentColor.colorset
│   │   │   └── Contents.json
│   │   ├── AppIcon.appiconset
│   │   │   └── Contents.json
│   │   └── Contents.json
│   ├── Base.lproj
│   │   ├── LaunchScreen.storyboard
│   │   └── Main.storyboard
│   ├── Info.plist
│   ├── SceneDelegate.swift
│   └── ViewController.swift
└── testHello.xcodeproj
    ├── project.pbxproj
    ├── project.xcworkspace
    │   ├── contents.xcworkspacedata
    │   ├── xcshareddata
    │   │   └── IDEWorkspaceChecks.plist
    │   └── xcuserdata
    │       └── david.s.xcuserdatad
    │           └── UserInterfaceState.xcuserstate
    └── xcuserdata
        └── david.s.xcuserdatad
            └── xcschemes
                └── xcschememanagement.plist
```

다음과 같이 dependencies 를 설치한다.

```bash
# Install dependencies
$ pod install

# Open your application workspace file
$ open App.xcworkspace

$ tree .
.
├── Podfile
├── Podfile.lock
├── Pods
│   ├── AFNetworking
│   │   ├── AFNetworking
│   │   │   ├── AFHTTPRequestOperation.h
│   │   │   ├── AFHTTPRequestOperation.m
│   │   │   ├── AFHTTPRequestOperationManager.h
│   │   │   ├── AFHTTPRequestOperationManager.m
│   │   │   ├── AFHTTPSessionManager.h
│   │   │   ├── AFHTTPSessionManager.m
│   │   │   ├── AFNetworkReachabilityManager.h
│   │   │   ├── AFNetworkReachabilityManager.m
│   │   │   ├── AFNetworking.h
│   │   │   ├── AFSecurityPolicy.h
│   │   │   ├── AFSecurityPolicy.m
│   │   │   ├── AFURLConnectionOperation.h
│   │   │   ├── AFURLConnectionOperation.m
│   │   │   ├── AFURLRequestSerialization.h
│   │   │   ├── AFURLRequestSerialization.m
│   │   │   ├── AFURLResponseSerialization.h
│   │   │   ├── AFURLResponseSerialization.m
│   │   │   ├── AFURLSessionManager.h
│   │   │   └── AFURLSessionManager.m
│   │   ├── LICENSE
│   │   ├── README.md
│   │   └── UIKit+AFNetworking
│   │       ├── AFNetworkActivityIndicatorManager.h
│   │       ├── AFNetworkActivityIndicatorManager.m
│   │       ├── UIActivityIndicatorView+AFNetworking.h
│   │       ├── UIActivityIndicatorView+AFNetworking.m
│   │       ├── UIAlertView+AFNetworking.h
│   │       ├── UIAlertView+AFNetworking.m
│   │       ├── UIButton+AFNetworking.h
│   │       ├── UIButton+AFNetworking.m
│   │       ├── UIImage+AFNetworking.h
│   │       ├── UIImageView+AFNetworking.h
│   │       ├── UIImageView+AFNetworking.m
│   │       ├── UIKit+AFNetworking.h
│   │       ├── UIProgressView+AFNetworking.h
│   │       ├── UIProgressView+AFNetworking.m
│   │       ├── UIRefreshControl+AFNetworking.h
│   │       ├── UIRefreshControl+AFNetworking.m
│   │       ├── UIWebView+AFNetworking.h
│   │       └── UIWebView+AFNetworking.m
│   ├── FLKAutoLayout
│   │   ├── FLKAutoLayout
│   │   │   ├── FLKAutoLayoutPredicateList.h
│   │   │   ├── FLKAutoLayoutPredicateList.m
│   │   │   ├── NSLayoutConstraint+FLKAutoLayoutDebug.h
│   │   │   ├── NSLayoutConstraint+FLKAutoLayoutDebug.m
│   │   │   ├── UIView+FLKAutoLayout.h
│   │   │   ├── UIView+FLKAutoLayout.m
│   │   │   ├── UIView+FLKAutoLayoutDebug.h
│   │   │   ├── UIView+FLKAutoLayoutDebug.m
│   │   │   ├── UIView+FLKAutoLayoutPredicate.h
│   │   │   └── UIView+FLKAutoLayoutPredicate.m
│   │   ├── LICENSE
│   │   └── README.md
│   ├── Headers
│   ├── Local\ Podspecs
│   ├── Manifest.lock
│   ├── ORStackView
│   │   ├── Classes
│   │   │   └── ios
│   │   │       ├── ORSplitStackView.h
│   │   │       ├── ORSplitStackView.m
│   │   │       ├── ORStackScrollView.h
│   │   │       ├── ORStackScrollView.m
│   │   │       ├── ORStackView.h
│   │   │       ├── ORStackView.m
│   │   │       ├── ORStackViewController.h
│   │   │       ├── ORStackViewController.m
│   │   │       ├── ORTagBasedAutoStackView.h
│   │   │       ├── ORTagBasedAutoStackView.m
│   │   │       └── private
│   │   │           ├── ORStackView+Private.h
│   │   │           └── ORStackView+Private.m
│   │   ├── LICENSE
│   │   └── README.md
│   ├── Pods.xcodeproj
│   │   ├── project.pbxproj
│   │   └── xcuserdata
│   │       └── david.s.xcuserdatad
│   │           └── xcschemes
│   │               ├── AFNetworking.xcscheme
│   │               ├── FLKAutoLayout.xcscheme
│   │               ├── ORStackView.xcscheme
│   │               ├── Pods-testHello.xcscheme
│   │               └── xcschememanagement.plist
│   └── Target\ Support\ Files
│       ├── AFNetworking
│       │   ├── AFNetworking-Info.plist
│       │   ├── AFNetworking-dummy.m
│       │   ├── AFNetworking-prefix.pch
│       │   ├── AFNetworking-umbrella.h
│       │   ├── AFNetworking.debug.xcconfig
│       │   ├── AFNetworking.modulemap
│       │   └── AFNetworking.release.xcconfig
│       ├── FLKAutoLayout
│       │   ├── FLKAutoLayout-Info.plist
│       │   ├── FLKAutoLayout-dummy.m
│       │   ├── FLKAutoLayout-prefix.pch
│       │   ├── FLKAutoLayout-umbrella.h
│       │   ├── FLKAutoLayout.debug.xcconfig
│       │   ├── FLKAutoLayout.modulemap
│       │   └── FLKAutoLayout.release.xcconfig
│       ├── ORStackView
│       │   ├── ORStackView-Info.plist
│       │   ├── ORStackView-dummy.m
│       │   ├── ORStackView-prefix.pch
│       │   ├── ORStackView-umbrella.h
│       │   ├── ORStackView.debug.xcconfig
│       │   ├── ORStackView.modulemap
│       │   └── ORStackView.release.xcconfig
│       └── Pods-testHello
│           ├── Pods-testHello-Info.plist
│           ├── Pods-testHello-acknowledgements.markdown
│           ├── Pods-testHello-acknowledgements.plist
│           ├── Pods-testHello-dummy.m
│           ├── Pods-testHello-frameworks-Debug-input-files.xcfilelist
│           ├── Pods-testHello-frameworks-Debug-output-files.xcfilelist
│           ├── Pods-testHello-frameworks-Release-input-files.xcfilelist
│           ├── Pods-testHello-frameworks-Release-output-files.xcfilelist
│           ├── Pods-testHello-frameworks.sh
│           ├── Pods-testHello-umbrella.h
│           ├── Pods-testHello.debug.xcconfig
│           ├── Pods-testHello.modulemap
│           └── Pods-testHello.release.xcconfig
├── testHello
│   ├── AppDelegate.swift
│   ├── Assets.xcassets
│   │   ├── AccentColor.colorset
│   │   │   └── Contents.json
│   │   ├── AppIcon.appiconset
│   │   │   └── Contents.json
│   │   └── Contents.json
│   ├── Base.lproj
│   │   ├── LaunchScreen.storyboard
│   │   └── Main.storyboard
│   ├── Info.plist
│   ├── SceneDelegate.swift
│   └── ViewController.swift
├── testHello.xcodeproj
│   ├── project.pbxproj
│   ├── project.xcworkspace
│   │   ├── contents.xcworkspacedata
│   │   ├── xcshareddata
│   │   │   └── IDEWorkspaceChecks.plist
│   │   └── xcuserdata
│   │       └── david.s.xcuserdatad
│   │           └── UserInterfaceState.xcuserstate
│   └── xcuserdata
│       └── david.s.xcuserdatad
│           └── xcschemes
│               └── xcschememanagement.plist
└── testHello.xcworkspace
    ├── contents.xcworkspacedata
    ├── xcshareddata
    │   └── IDEWorkspaceChecks.plist
    └── xcuserdata
        └── david.s.xcuserdatad
            └── UserInterfaceState.xcuserstate

$ open testHello.xcworkspace
```

`testHello.xcodeproj` 말고 `Pods.xcodeproj` 가 생성되었다. 이 것은 library 들을 위한 project 이다. 또한 `testHello.xcworkspace` 파일도 생성되었다. 이것은 `testHello.xcodeproj, Pods.xcodeproj` 를 포함한 workspace file 이다. library 도 함께 build 하려면 `testHello.xcworkspace` 를 build 해야 한다.

다음과 같이 header file 을 import 하고 library 를 사용하는 code 를 작성한다.

Import your dependencies.

```swift
#import <Reachability/Reachability.h>
```

## Using Pod Lib Create

> * [Using Pod Lib Create](https://guides.cocoapods.org/making/using-pod-lib-create.html)

```bash
# Use default template from https://github.com/CocoaPods/pod-template.git
$ pod lib create MyLib
Cloning `https://github.com/CocoaPods/pod-template.git` into `MyLib`.
> iOS
> Swift
> yes
> None
> No
# After this xcode project will be opened.

$ tree MyLib -L 2
MyLib
├── Example
│   ├── MyLib.xcodeproj
│   ├── MyLib.xcworkspace
│   ├── Podfile
│   ├── Podfile.lock
│   ├── Pods
│   └── Tests
├── LICENSE
├── MyLib
│   ├── Assets
│   └── Classes
├── MyLib.podspec
├── README.md
└── _Pods.xcodeproj -> Example/Pods/Pods.xcodeproj
```

주요 파일은 다음과 같다.

| File | Description |
|--|--|
| `.travis.yml` | a setup file for travis-ci. | 
| `_Pods.xcproject` | a symlink to your Pod's project for Carthage support | 
| `LICENSE` | defaulting to the MIT License. | 
| `MyLib.podspec` | the Podspec for your Library. | 
| `README.md` | a default README in markdown. | 
| `RemoveMe.swift/m` | a single file to to ensure compilation works initially | 
| `MyLib` | your library's classes | 

## Create POD

```bash
$ pod spec create Peanut
$ edit Peanut.podspec
$ pod spec lint Peanut.podspec
```
