- [Abstract](#abstract)
- [Materials](#materials)
- [Language Summary](#language-summary)
    - [Messages](#messages)
    - [Defined Types](#defined-types)
    - [Preprocessor Directives](#preprocessor-directives)
    - [Compiler Directives](#compiler-directives)
    - [Classes](#classes)
    - [Categories](#categories)
    - [Formal Protocols](#formal-protocols)
    - [Methods](#methods)
    - [Deprecation Syntax](#deprecation-syntax)
    - [Naming Conventions](#naming-conventions)
- [Objects, Classes and Methods](#objects-classes-and-methods)
- [Allocating and Initializing Objects](#allocating-and-initializing-objects)
- [Declared Properties](#declared-properties)
- [Categories and Exensions](#categories-and-exensions)
- [Blocks](#blocks)
- [Protocols](#protocols)
- [Fast Enumeration](#fast-enumeration)
- [Enabling Handling](#enabling-handling)
- [Threading](#threading)
- [Remote Messaging](#remote-messaging)
- [Using C++ With Objective-C](#using-c-with-objective-c)
- [Deprecation Syntax](#deprecation-syntax)
- [Naming Conventions](#naming-conventions)

-------------------------------------------------------------------------------

# Abstract

objc를 정리한다. 

# Materials

* objective-c 2.0 programming language pdf
* [Programming with Objective-C @ apple](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/ProgrammingWithObjectiveC/Introduction/Introduction.html#//apple_ref/doc/uid/TP40011210-CH1-SW1)
* [developer doc](https://developer.apple.com/documentation)
  * developer documents
  * 일부 문서는 검색이 되지 않는다. 왜???
* [apple opensource](https://opensource.apple.com/source/)
  * apple opensource
  * Framework 의 source 를 확인 할 수 있다.
* [CFString.c](https://opensource.apple.com/source/CF/CF-744.18/CFString.c)
  * NSString source
* [iOS 11 & Swift 4 @ udemy](https://www.udemy.com/ios-11-app-development-bootcamp/learn/v4/overview)
  * 유료이지만 알차다. src는 이메일 인증으로 다운받을 수 있다.
* [Objective-C 강좌 - 12개 앱 만들면서 배우는 iOS 아이폰 앱 개발 @ inflearn](https://www.inflearn.com/course/objective-c-%EA%B0%95%EC%A2%8C/)
  * 유료이다. src는 강좌안 링크로 다운받을 수 있다..
* [effective objective-c 2.0](https://www.amazon.com/Effective-Objective-C-2-0-Specific-Development-ebook/dp/B00CUG5MZA)

# Language Summary

## Messages

message 는 다음과 같은 형식을 갖는다.

```
[receiver message]
```

`receiver` 는 `nil, self, super, object, class` 가 가능하다. `message` 는 method 와 arguments 의 모음이다.

## Defined Types

다음과 같은 주요 타입이 `objc/objc.h` 에 정의 되어 있다.

| Type | Definition |
|:-----|:-----------|
| `id` | An object. `void*` 와 유사하다. |
| `Class` | A class object |
| `SEL` | A selector. 컴파일러에서 메소드를 구분하기 위한 타입 |
| `IMP` | A pointer to a method implementation that returns an id. |
| `BOOL` | A Boolean value. `YES` 혹은 `NO`이다. `BOOL` 의 타입은 `char` 이다. |

다음은 `objc.h` 에 정의된 주요 값들이다.

| `nil` | A null object pointer |
| `Nil` | A null class Pointer |
| `NO` | A Boolean false value |
| `YES` | A boolean true value |


## Preprocessor Directives

| Notation | Definition |
:----------|:-----------|
| `#import` | `#include` 와 같다. 중복포함 문제를 해결해 준다. |
| `//` | 주석 |

## Compiler Directives

| Directive | Definition |
:----------|:-----------|
| `@interface` | 클래스 혹은 카테고리 선언 |
| `@implementation` | 클래스 혹은 카테고리 구현 |
| `@protocol` | 프로토콜 선언 |
| `@end` | 클래스, 카테고리, 프로토콜의 선언 및 구현 종료 |
| `@private` |  |
| `@protected` |  |
| `@public` |  |
| `@try` |  |
| `@throw` |  |
| `@catch()` |  |
| `@finally` |  |
| `@property` | 프라퍼티 선언 |
| `@synthesize` | 프라퍼티 구현 |
| `@dynamic` | 프라퍼티의 구현이 없어도 컴파일 경고를 출력금지. 동적으로 구현하겠다는 표시? |
| `@class` | Declares the names of classes defined elsewhere |
| `@selector(method_name)` | 컴파일된 실렉터(`SEL`)를 리턴한다 |
| `@protocol(protocol_name` | forward declaration 을 위해 프로토콜을 리턴한다. |
| `@encode(type_spec` | Yields a character string that encodes the type structure of type_spec |
| `@"string"` | NSString object 를 리턴한다 |
| `@"string1" @"string2" ... @"stringN"` | 문자열 상수를 결합하여 NSString object 를 리턴한다 |
| `@synchronized` | 멀티쓰레드 환경에서 동기화를 보장해준다 |

## Classes

일반적인 클래스 선언은 다음과 같다. 주로 `.h` 에 저장한다.

```objc
#import "ItsSuperclass.h"
          @interface ClassName : ItsSuperclass < protocol_list >
          {
              instance variable declarations
          }
          method declarations
          @end
```

일반적인 클래스 구현은 다음과 같다. 주로 `.m` 에 저장한다.

```objc
  #import “ClassName.h”
          @implementation ClassName
          method definitions
          @end
```

## Categories

카테고리는 c# 의 extension method 와 유사하다. 특정 클래스의 소스 코드 수정없이 메소드를 추가할 수 있다. 일반적인 카테고리 선언은 다음과 같다.

```objc
 #import "ClassName.h"
          @interface ClassName ( CategoryName ) < protocol list >
          method declarations
          @end
```

일반적인 카테고리 구현은 다음과 같다.

```objc
 #import "CategoryName.h"
          @implementation ClassName ( CategoryName )
          method definitions
          @end
```

## Formal Protocols

프로토콜은 c# 의 interface 와 같다. 일반적인 프로토콜 선언은 다음과 같다.

```objc
 @protocol ProtocolName < protocol list >
          declarations of required methods
          @optional
          declarations of optional methods
          @required
          declarations of required methods
          @end
```

프로토콜 선언에서 다음과 같은 타입 한정자들은 remote messaing 을 위해 사용한다. 

| Type Qualifier | Definition |
|:---------------|:-----------|
| `oneway` | The method is for asynchronous messages and has no valid return type. |
| `in` | The argument passes information to the remote receiver. |
| `out` | The argument gets information returned by reference. |
| `inout` | The argument both passes information and gets information. |
| `bycopy` | A copy of the object, not a proxy, should be passed or returned. |
| `byref` | A reference to the object, not a copy, should be passed or returned. |

## Methods

objc는 함수를 호출하는 형식이 특이하다. 메소드 앞의 `+` 는 `class method` 를 의미한다. 메소드 앞의 `-` 는 `instance method` 를 의미한다. 메소드의 argument 는 c-style 의 형변환 문법을 이용해서 표기한다. 메소드를 호출하는 것은 메시지를 전달한다는 의미이기
때문에 c#과 달리 `[nil print]`를 허용한다. 인자는 `:` 다음에 따라온다. 인자가 여러개인 경우
두번째 인자부터 label을 사용한다. label은 생략할 수 있지만 추천하지 않는다. label 역시 메소드의 구성요소 이기 때문에 컴파일러는 label 을 포함하여 메소드를 구별한다. 메소드의 리턴, 인자 타입은 기본이 `int` 가 아니고 `id` 이다.

모든 메소드는 `self, _cmd` 를 숨겨진 인자로 제공한다. `self` 는 receiving object 이고 `_cmd` 는 호출되는 메소드의 실렉터이다.

다음은 메소드를 c# 과 objc 로 구현한 예이다.

```csharp
Document document = new Document();
document.Print();

public class Document : IPrintable
{
    public bool SaveAs(string fileName, string filePath)
    {
        return true;
    }
}
...
Document document = new Document();
bool success = document.SaveAs("MyFile.txt", "C:\Temp");
```

```objc
Document *document = [[Document alloc] init];
[document print];
[document release];

// Document.h
@interface Document : NSObject<Printing>
 
-(BOOL)saveAs:(NSString *)fileName toPath:(NSString *)filePath;
 
@end
 
// Document.m
@implementation Document
...
- (BOOL)saveAs:(NSString *)fileName toLocation:(NSString *)filePath {
    // Add code to save file to path...
    return YES;
}
@end
...
Document *document = [[Document alloc] init];
BOOL success = [document saveAs:@"MyFile.txt" toLocation:@"~/Temp"];
[document release];
```

## Deprecation Syntax

특정 함수를 폐기할 때 다음과 같이 표현한다.

```objc
 @interface SomeClass
          -method __attribute__((deprecated));
```

```objc
#include <AvailabilityMacros.h>
@interface SomeClass
-method DEPRECATED_ATTRIBUTE; // or some other deployment-target-specific macro @end

```

## Naming Conventions

클래스, 카테고리, 프로토콜의 선언은 주로 `.h` 에 저장하고 구현은 주로 `.m` 에 저장한다.

클래스, 카테고리, 프로토콜의 이름은 주로 대문자로 시작한다. 그러나 메소드, 인스턴스 변수의 이름은 소문자로 시작한다. 인스턴서를 저장한 변수 역시 소문자로 시작한다. 


# Objects, Classes and Methods

objc는 `a.h`에서 `@interface`를 이용하여 class를 선언하고 `a.m`에서
`@implementation`를 이용하여 class를 구현한다.  `id` type은 `void*`와
유사하다. 함수이름 앞의 `-`는 instance method를 의미하고 `+`는 class
method를 의미한다.

```c#
// Document.cs
public class Document
{
    public Document(string title)
    {
        // Initialise properties, etc.
    }
}
```

```
// Document.h
@interface Document : NSObject
 
- (id)initWithTitle:(NSString *)aTitle;
@end
 
// Document.m
@implementation Document
 
- (id)initWithTitle:(NSString *)aTitle {
    if((self = [super init])) {
        // Initialise properties, etc.
    }
    return self;
}
@end
```

# Allocating and Initializing Objects

objc에서 메모리는 `alloc` 함수를 호출하여 할당한다. `dealloc` 함수를
호출하여 해제할 수 있다. `alloc`, `retain` 함수를 호출하면 reference
count가 증가하고 `release`를 호출하면 reference count가
감소한다. reference count가 0이되면 `dealloc`이 호출된다.

`autorelease`를 이용하면 자동으로 해제할 수 있지만 의도치 않는 일이
발생할 수 있기 때문에 추천하지 않는다.

```csharp
Document document = new Document("My New Document");
```

```objc
Document *document = [[Document alloc] initWithTitle:@"My New Document"];

Document *document = [[Document alloc] initWithTitle:@"My New Document"];
// Do some stuff...
[document release];

Document *document = [[[Document alloc] initWithTitle:@"My New Document"] autorelease];
// Do some stuff...
// No need to manually release
```

# Declared Properties

c#의 property와 유사하다. `@property`로 선언하고 `@synthesize`로
구현한다. `nonatomic`은 thread safe하지 않다는 의미하고 `copy`는
문자열을 복사한다는 의미이다.

```csharp
public class Document : IPrintable
{
    public string Title { get; set; }
 
    public Document(string title)
    {
        this.Title = title;
    }
}
```

```objc
// Document.h
@interface Document : NSObject<Printing>
 
- (id)initWithTitle:(NSString *)aTitle;
@property (nonatomic, copy) NSString *title;
 
@end
 
// Document.m
@implementation Document
 
@synthesize title;
 
- (id)initWithTitle:(NSString *)aTitle {
    if((self = [super init])) {
        self.title = aTitle;
    }
    return self;
}
 
- (void)dealloc {
    [title release];
    [super dealloc];
}
@end
```

# Categories and Exensions

c#의 extension method와 비슷하다. class의 상속없이 기능을 확장할 수
있다.

```csharp
public static class StringExtensions
{
    public static string Reverse(this string s)
    {
        char[] arr = s.ToCharArray();
        Array.Reverse(arr);
        return new string(arr);
    }
};

Document document = new Document("My New Document");
Console.WriteLine(document.title.Reverse());
```

```objc
// NSString+Reverse.h
@interface NSString (Reverse)
 
- (NSString *)reverse;
 
@end
 
// NSString+Reverse.m
@implementation NSString (Reverse)
 
- (NSString *)reverse {
    NSMutableString *reversedStr;
    int len = [self length];
 
    reversedStr = [NSMutableString stringWithCapacity:len];
 
    while (len > 0)
        [reversedStr appendString:
         [NSString stringWithFormat:@"%C", [self characterAtIndex:--len]]];
 
    return reversedStr;
}
@end

#import "NSString+Reverse.h"
...
Document *document = [[Document alloc] initWithTitle:@"My New Document"];
NSLog(@"Reversed title: %@", [document.title reverse]);
[document release];
```

# Blocks

apple 이 `lambda expression` 을 지원하기 위해 clang implementation 에 추가한 비표준 기능이다.

[blocks @ wikipedia](https://en.wikipedia.org/wiki/Blocks_(C_language_extension))

[blocks @ apple](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/Blocks/Articles/00_Introduction.html)

```objc

```

# Protocols

objc의 protocol은 c#의 interface와 비슷하다.

```csharp
// IPrintable.cs
public interface IPrintable
{
    Print();
}
 
// Document.cs
public class Document : IPrintable
{
    public void Print()
    {
        Console.WriteLine(@"Printing...{0}", this.Title);
    }
}
```

```objc
// Printable.h
@protocol Printing <NSObject>
 
- (void)print;
@optional
- (void)preview;
 
@end
 
// Document.h
@interface Document : NSObject<Printing>
 
@end
 
// Document.m
@implementation Document
 
- (void)print {
    NSLog(@"Printing %@", self.title);
}
@end
```

# Fast Enumeration

# Enabling Handling

# Threading

# Remote Messaging

# Using C++ With Objective-C

# Deprecation Syntax

# Naming Conventions

