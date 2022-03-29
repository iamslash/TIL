- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
  - [Install](#install)
  - [Get Started](#get-started)
  - [Create POD](#create-pod)
  - [Publish CocoaPod](#publish-cocoapod)

----

# Abstract

CocoaPods is a dependency manager for Swift and Objective-C Cocoa projects. 

# Materials

* [CocoaPods Guides](https://guides.cocoapods.org/)

# Basic

## Install

```bash
$ sudo gem install cocoapods
```

## Get Started

`$ pod init` 으로 Podfile 을 생성한다.

```ruby
platform :ios, '8.0'
use_frameworks!

target 'MyApp' do
  pod 'AFNetworking', '~> 2.6'
  pod 'ORStackView', '~> 3.0'
  pod 'SwiftyJSON', '~> 2.3'
end
```

```bash
# Install dependencies
$ pod install

# Open your application workspace file
$ open App.xcworkspace
```

Import your dependencies.

```swift
#import <Reachability/Reachability.h>
```

## Create POD

```bash
$ pod spec create Peanut
$ edit Peanut.podspec
$ pod spec lint Peanut.podspec
```

## Publish CocoaPod

* [[iOS] Library를 CocoaPods에 배포하는 방법](https://jinnify.tistory.com/61)

Create GitHub Repository and Clone it.

```bash
$ git clone git@github.com:iamslash/FooHelp.git
$ cd FooHelp
$ pod lib create pod name
> iOS
> Swift
> yes
> None
> No
```
