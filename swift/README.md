
<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Abstract](#abstract)
- [Materials](#materials)
- [Usage](#usage)
    - [Collections compared c++ container](#collections-compared-c-container)
    - [Collections](#collections)
    - [Class](#class)
- [Type](#type)
- [Constants](#constants)
- [Variables](#variables)

<!-- markdown-toc end -->

-------------------------------------------------------------------------------

# Abstract

swift에 대해 정리한다.

# Materials

* [Introduction to Swift (for C#.NET developers)](https://www.jbssolutions.com/introduction-swift-c-net-developers/)
* [swift basic](http://minsone.github.io/mac/ios/swift-the-basic-summary)

# Usage

## Collections compared c++ container

## Collections

## Class

```csharp
// c#
public class HelloWorld {
    private string _message;
     
    HelloWorld(string message) {
        this._message = message;
    }
     
    void display() {
        System.Diagnostics.Debug.WriteLine(_message);
    }
     
    public string Message {
        get {
            return _message;
        }
        set {
            _message = value;
        }
    }
}

// C#
HelloWorld a;
var b = new HelloWorld("hello");
```

```swift
// Swift
public class HelloWorld {
    private var _message: String
     
    init(message: String) {
        self._message = message
    }
     
    func display() {
        debugPrint(_message)
    }
     
    public var Message: String {
        get {
            return _message
        }
        set {
            _message = newValue
        }
    }
}

// Swift
var a : HelloWorld
var b = HelloWorld(message: "hello")
```

# Type

* Int
* Double, Float
* Boolean
* String
* Array, Dictionary
* Tuple

# Constants

```swift
let a = 10
```

# Variables

```swift
var a = 10
```
