- [Abstract](#abstract)
- [Materials](#materials)
- [Basic Usages](#basic-usages)
  - [Collections compared to c++ containers](#collections-compared-to-c-containers)
  - [Collections](#collections)
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
- [Advanced Usages](#advanced-usages)
  - [Objects, Classes and Methods](#objects-classes-and-methods)
  - [Allocating and Initializing Objects](#allocating-and-initializing-objects)
  - [Declared Properties](#declared-properties)
  - [Categories and Exensions](#categories-and-exensions)
  - [Blocks](#blocks)
  - [Protocols](#protocols)
  - [Fast Enumeration](#fast-enumeration)
  - [Enabling Staic Behavior](#enabling-staic-behavior)
  - [Selectors](#selectors)
  - [Exception Handling](#exception-handling)
  - [Threading](#threading)
  - [Remote Messaging](#remote-messaging)
  - [Using C++ With Objective-C](#using-c-with-objective-c)

-------------------------------------------------------------------------------

# Abstract

[objective-c 2.0 programming language pdf](http://cagt.bu.edu/w/images/b/b6/Objective-C_Programming_Language.pdf) 를 읽고 정리하자.
# Materials

* [objective-c 2.0 programming language pdf](http://cagt.bu.edu/w/images/b/b6/Objective-C_Programming_Language.pdf)
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
* [iOS Programming](https://eezytutorials.com/ios/)

# Basic Usages

## Collections compared to c++ containers

* [이곳](https://developer.apple.com/library/archive/documentation/General/Conceptual/DevPedia-CocoaCore/Collection.html) 에 의하면 objc 의 collection 은 `NSArray, NSSet, NSDictionary, NSPointerArray, NSHashTable, NSMapTable, NSMutableArray, NSMutableDictionary, NSMutableSet, NSCountedSet` 등이 있다.
  
| c++                  | objc                  |
|:---------------------|:----------------------|
| `if, else`           | `if, else`            |
| `for, while`         | `for, while`          |
| `array`              | `NSArray`             |
| `vector`             | `NSMutableArray`      |
| `deque`              | ``                    |
| `forward_list`       | ``                    |
| `list`               | ``                    |
| `stack`              | ``                    |
| `queue`              | ``                    |
| `priority_queue`     | ``                    |
| `set`                | ``                    |
| `multiset`           | ``                    |
| `map`                | ``                    |
| `multimap`           | ``                    |
| `unordered_set`      | `NSMutableSet`        |
| `unordered_multiset` | `NSCountedSet`        |
| `unordered_map`      | `NSMutableDictionary` |
| `unordered_multimap` | ``                    |

## Collections

* NSArray

```objc
#import <Foundation/Foundation.h>

int main(void) {
  // initialization
  NSArray *array = @[@"FOO", @"BAR"];
  // access value at index 0
  NSString *value = array[0];

  // + array
  NSArray *array = [NSArray array];   
  NSLog(@"%@",array);

  // + arrayWithObject:
  NSArray *array =  [NSArray arrayWithObject:@"Foo"];
  NSLog(@"%@",array);

  // + arrayWithObjects:
  NSArray *array =  [NSArray arrayWithObjects:@"Eezy",@"Tutorials"];  
  NSLog(@"%@",array);

  // + arrayWithArray:
  NSArray *tempArray = [NSArray arrayWithObjects:@"Eezy",@"Tutorials"]; 
  NSArray *array =  [NSArray arrayWithArray:tempArray];   
  NSLog(@"%@",array);

  // + arrayWithContentsOfFile:
  //
  // plist
/*
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
 <array>
    <string>Foo</string>
    <string>Bar</string>
</array>
</plist>
*/
  NSString *file = [[NSBundle mainBundle] pathForResource:@"Data" ofType:@"plist"];
  NSArray *array = [NSArray arrayWithContentsOfFile:file];
  NSLog(@"%@", array);      
}

  // + arrayWithContentsOfURL:
  NSURL *url = [NSURL URLWithString:@"http://www.foo.com/a.plist"];
  NSArray *array = [NSArray arrayWithContentsOfURL:url];
  NSLog(@"%@", array]);

  // + arrayWithObjects:count:
  NSString *values[3];
  values[0] = @"Foo";
  values[1] = @"Bar";
  values[2] = @"Baz"; // Baz ignored since count is 2 
  NSArray *array = [NSArray arrayWithObjects:values count:2];
  NSLog(@"%@",array ]);

  // - init
  NSArray *array = [[NSArray alloc]init];
  NSLog(@"%@",array ]);

  // - initWithArray:
  var tempArray:NSArray = NSArray(array: ["Foo","Bar"])  
  var array = NSArray(array: tempArray)
  println(array)

  // - initWithArray:copyItems:
  NSArray *tempArray = [NSArray arrayWithObjects:@"Foo",@"Bar"];
// Check if the two array objects refer to same objects. It shouldn't.
  NSArray *array =  [[NSArray alloc]initWithArray:tempArray copyItems:YES];   
  NSLog(@"%@",array);

  // - initWithContentsOfFile:
  // plist
  /*
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
 <array>
    <string>Foo</string>
    <string>Bar</string>
</array>
</plist>  
  */
  NSString *file = [[NSBundle mainBundle] pathForResource:@"Data" ofType:@"plist"];
  NSArray *array = [[NSArray alloc]initWithContentsOfFile:file];
  NSLog(@"%@", array);

  // - initWithContentsOfURL:
  NSURL *url = [NSURL URLWithString:@"http://www.foo.com/a.plist"];
  NSArray *array = [[NSArray alloc]initWithContentsOfURL:url];
  NSLog(@"%@",array ]);

  // - initWithObjects:
  NSArray *array =  [[NSArray alloc] initWithObjects:@"Foo",@"Bar"];       
  NSLog(@"%@",array)

  // - initWithObjects:count:
  NSString *values[3];
  values[0] = @"Foo";
  values[1] = @"Bar";
  values[2] = @"Baz"; // Baz ignored since count is 2 
  NSArray *array = [[NSArray alloc] initWithObjects:values count:2];
  NSLog(@"%@",array ]);

  // - containsObject:
  NSArray *array =  [NSArray arrayWithObjects:@"Foo", @"Bar"];
  BOOL containsObject = [array containsObject:@"Bar"];
  NSLog(@"Contains Object Bar: ",containsObject)
  containsObject = [array containsObject:@"Foo Bar"];
  NSLog(@"Contains Object Foo Bar: ",containsObject)

  // - count
  NSArray *array =  [NSArray arrayWithObjects:@"Foo",@"Bar"];
  NSLog(@"Count: %d",[array count]);

  // - getObjects:range:
  NSArray *tempArray = [NSArray arrayWithObjects:@"Foo",@"Bar", @"Baz"];
  id *array;
  NSRange range = NSMakeRange(1, 2);
  objects = malloc(sizeof(id) * range.length);
  [tempArray getObjects:array range:range];
  for (index = 0; index < range.length; index++) {
    NSLog(@"Array object index %d: %@",index, objects[index]);
  }

  // - firstObject  
  NSArray *array = [NSArray arrayWithObjects:@"Foo",@"Bar", @"Baz"];
  NSLog(@"First Object: %@", [array firstObject])

  // - lastObject
  NSArray *array = [NSArray arrayWithObjects:@"Foo",@"Bar", @"Baz"];
  NSLog(@"Last Object: %@", [array lastObject])

  // - objectAtIndex:
  NSArray *array = [NSArray arrayWithObjects:@"Foo",@"Bar", @"Baz"];
  NSLog(@"Object at index 2: %@", [array objectAtIndex:2])

  // - objectAtIndexedSubscript:
  NSArray *array = [NSArray arrayWithObjects:@"Foo",@"Bar", @"Baz"];
  NSLog(@"Object at Indexed Subscript 2: %@", [array objectAtIndexedSubscript:2])
  NSLog(@"Object at Indexed Subscript 3: %@", [array objectAtIndexedSubscript:3])

  // - objectsAtIndexes:
  NSArray *tempArray = [NSArray arrayWithObjects:@"Foo",@"Bar", @"Baz"];
  array = [tempArray objectsAtIndexes:[NSIndexSet indexSetWithIndexesInRange:NSMakeRange(1, 2)]]; 
  NSLog(@"%@",array ]);

  // - objectEnumerator
  NSArray *array = [NSArray arrayWithObjects:@"Foo",@"Bar", @"Baz"];
  NSEnumerator *enumerator = [array objectEnumerator];
  id anObject;
  while (anObject = [enumerator nextObject]) {
    NSLog(@"%@",anObject ]);
  }

  // - reverseObjectEnumerator
  NSArray *array = [NSArray arrayWithObjects:@"Foo",@"Bar", @"Baz"];
  NSEnumerator *enumerator = [array reverseObjectEnumerator];
  id anObject;
  while (anObject = [enumerator nextObject]) {
    NSLog(@"%@",anObject ]);
  }

  // - indexOfObject:
  NSArray *array = [NSArray arrayWithObjects:@"Foo",@"Bar", @"Baz"];
  NSLog(@"Index of Baz is %d",[array indexOfObject:@"Baz"]);

  // - indexOfObject:inRange:
  NSArray *array = [NSArray arrayWithObjects:@"Foo",@"Bar", @"Baz",@"Foo"];
  NSRange range = NSMakeRange(1, 2);
  NSLog(@"Index of Foo in range 1,3 is %d",[array indexOfObject:@"Foo" inRange:range ]);
  range = NSMakeRange(0, 3);
  NSLog(@"Index of Foo in range 0,3 is %d",[array indexOfObject:@"Foo" inRange:range ]);

  // - indexOfObjectIdenticalTo:
  NSString *str = @"Foo";
  NSArray *array = [NSArray arrayWithObjects:str,@"Bar", @"Baz"];
  NSLog(@"Index of Eezy identical",[array indexOfObject:str]);
  NSLog(@"Index of Eezy identical",[array indexOfObject:@"Foo"]);

  // - indexOfObjectIdenticalTo:inRange:
  NSString *str = @"Foo";
  NSArray *array = [NSArray arrayWithObjects:@"Foo",@"Bar", @"Baz", str];
  NSRange range = NSMakeRange(1, 3);
  NSLog(@"Index of Foo identical",[array indexOfObject:str inRange:range ]);
  range = NSMakeRange(0, 3);
  NSLog(@"Index of Foo identical",[array indexOfObject:str inRange:range ]);

  // - indexOfObjectPassingTest:
  NSArray *array = [NSArray arrayWithObjects:@"Foo",@"Bar", @"Baz"];
  int index = [array indexOfObjectPassingTest:^BOOL(id element,NSUInteger idx,BOOL *stop){
     return [(NSArray *)element containsObject:@"Foo"];
  }];
  NSLog(@"Index is %d",index);
  if (index >= 0 && index < [arrayWithArray count]){
    [arrayWithArray removeObjectAtIndex:index];
  }
```

* NSset
  * [NSSet by example](https://eezytutorials.com/ios/nsset-by-example.php)
  * [NSset @ apple](https://developer.apple.com/documentation/foundation/nsset?language=objc)

* NSDictionary

  * [NSDictionary by example](https://eezytutorials.com/ios/nsdictionary-by-example.php)
  * [NSDictionary @ apple](https://developer.apple.com/documentation/foundation/nsdictionary?language=objc)

* NSPointerArray

  * [NSPointerArray @ apple](https://developer.apple.com/documentation/foundation/nspointerarray?language=objc)


* NSHashTable
    
    * [NSHashTable @ apple](https://developer.apple.com/documentation/foundation/nshashtable?language=objc)


* NSMapTable

  * [NSMapTable @ apple](https://developer.apple.com/documentation/foundation/nsmaptable?language=objc)


* NSMutableArray
  * [NSMutableArray by example](https://eezytutorials.com/ios/nsmutablearray-by-example.php)
  * [NSMutableArray @ apple](https://developer.apple.com/documentation/foundation/nsmaptable?language=objc)

* NSMutableDictionary
  
  * [NSMutableDictionary by example](https://eezytutorials.com/ios/nsmutabledictionary-by-example.php)
  * [NSMutableDictionary @ apple](https://developer.apple.com/documentation/foundation/nsmutabledictionary?language=objc)

* NSMutableSet
  * [NSMutableSet by example](https://eezytutorials.com/ios/nsmutableset-by-example.php)
  * [NSMutableSet @ apple](https://developer.apple.com/documentation/foundation/nsmutableset?language=objc)

* NSCountedSet
  * [NSCountedSet by example](https://eezytutorials.com/ios/nscountedset-by-example.php)
  * [NSCountedSet @ apple](https://developer.apple.com/documentation/foundation/nscountedset?language=objc)

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

| Value | |
|:-----|:-----------|
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

objc 는 함수를 호출하는 형식이 특이하다. 메소드 앞의 `+` 는 `class method` 를
의미한다. 메소드 앞의 `-` 는 `instance method` 를 의미한다. 메소드의 argument 는
c-style 의 형변환 문법을 이용해서 표기한다. 메소드를 호출하는 것은 메시지를
전달한다는 의미이기 때문에 c#과 달리 `[nil print]`를 허용한다. 인자는 `:` 다음에
따라온다. 인자가 여러개인 경우 두번째 인자부터 label을 사용한다. label은 생략할
수 있지만 추천하지 않는다. label 역시 메소드의 구성요소 이기 때문에 컴파일러는
label 을 포함하여 메소드를 구별한다. 메소드의 리턴, 인자 타입은 기본이 `int` 가
아니고 `id` 이다.

모든 메소드는 `self, _cmd` 를 숨겨진 인자로 제공한다. `self` 는 receiving object
이고 `_cmd` 는 호출되는 메소드의 실렉터이다.

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

```c
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

클래스, 카테고리, 프로토콜의 선언은 주로 `.h` 에 저장하고 구현은 주로 `.m` 에
저장한다.

클래스, 카테고리, 프로토콜의 이름은 주로 대문자로 시작한다. 그러나 메소드,
인스턴스 변수의 이름은 소문자로 시작한다. 인스턴스를 저장한 변수 역시 소문자로
시작한다. 

# Advanced Usages

## Objects, Classes and Methods

objc는 `a.h` 에서 `@interface` 를 이용하여 class 를 선언하고 `a.m` 에서
`@implementation` 를 이용하여 class 를 구현한다.  `id` type은 `void*` 와
유사하다. 함수이름 앞의 `-` 는 instance method를 의미하고 `+` 는 class method를
의미한다.

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

```c
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

## Allocating and Initializing Objects

objc에서 메모리는 `alloc` 함수를 호출하여 할당한다. `dealloc` 함수를 호출하여
해제할 수 있다. `alloc`, `retain` 함수를 호출하면 `reference count` 가 증가하고
`release` 를 호출하면 `reference count 가 감소한다. `reference count` 가 `0`
이되면 `dealloc` 이 호출된다.

`autorelease` 를 이용하면 자동으로 해제할 수 있지만 의도치 않는 일이 발생할 수
있기 때문에 추천하지 않는다.

```csharp
Document document = new Document("My New Document");
```

```c
Document *document = [[Document alloc] initWithTitle:@"My New Document"];

Document *document = [[Document alloc] initWithTitle:@"My New Document"];
// Do some stuff...
[document release];

Document *document = [[[Document alloc] initWithTitle:@"My New Document"] autorelease];
// Do some stuff...
// No need to manually release
```

## Declared Properties

c# 의 property 와 유사하다. `@property` 로 선언하고 `@synthesize` 로 구현한다.
`nonatomic` 은 thread safe 하지 않다는 의미하고 `copy` 는 문자열을 복사한다는
의미이다.

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

```c
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

## Categories and Exensions

c# 의 extension method 와 비슷하다. class 의 상속없이 기능을 확장할 수 있다.

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

```c
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

## Blocks

[blocks @ wikipedia](https://en.wikipedia.org/wiki/Blocks_(C_language_extension))

[blocks @ apple](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/Blocks/Articles/00_Introduction.html)

apple 이 `lambda expression` 을 지원하기 위해 clang implementation 에 추가한
비표준 기능이다. 다음은 예이다.

```c
int multiplier = 7;
int (^myBlock)(int) = ^(int num) {
    return num * multiplier;
};
```

block 의 외부와 block 의 내부에서 변수를 공유하기 위해 `__block` 을 사용한다.

```c
NSArray *stringsArray = @[ @"string 1",
                          @"String 21", // <-
                          @"string 12",
                          @"String 11",
                          @"Strîng 21", // <-
                          @"Striñg 21", // <-
                          @"String 02" ];
 
NSLocale *currentLocale = [NSLocale currentLocale];
__block NSUInteger orderedSameCount = 0;
 
NSArray *diacriticInsensitiveSortArray = [stringsArray sortedArrayUsingComparator:^(id string1, id string2) {
 
    NSRange string1Range = NSMakeRange(0, [string1 length]);
    NSComparisonResult comparisonResult = [string1 compare:string2 options:NSDiacriticInsensitiveSearch range:string1Range locale:currentLocale];
 
    if (comparisonResult == NSOrderedSame) {
        orderedSameCount++;
    }
    return comparisonResult;
}];
 
NSLog(@"diacriticInsensitiveSortArray: %@", diacriticInsensitiveSortArray);
NSLog(@"orderedSameCount: %d", orderedSameCount);
 
/*
Output:
 
diacriticInsensitiveSortArray: (
    "String 02",
    "string 1",
    "String 11",
    "string 12",
    "String 21",
    "Str\U00eeng 21",
    "Stri\U00f1g 21"
)
orderedSameCount: 2
*/
```

## Protocols

objc 의 protocol 은 c# 의 interface 와 비슷하다.

```cs
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

```c
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

## Fast Enumeration

collection 을 편리하게 순회하는 방법이다. 다음과 같은 방법으로 사용한다.

```c
for ( Type newvar in expression ) { statements }
```

```c
Type existingvar;
for ( Type existingvar in expression ) { statements }
```

다음은 fast enumeration 을 사용한 예이다.

```c
NSArray *array = [NSArray arrayWithObjects:@"One", @"Two", @"Three", @"Four", nil];
for (NSString *element in array) {
  NSLog(@"element: %@", element);
}

NSDictionary *dictionary = [NSDictionary dictionaryWithObjectsAndKeys:@"quattuor", @"four", @"quinque", @"five", @"sex", @"six", nil];
NSString *key;
for (key in dictionary) {
  NSLog(@"English: %@, Latin: %@", key, [dictionary valueForKey:key]);
}
```

## Enabling Staic Behavior

objc 는 기본적으로 dynamic behavior 이다. 즉 compile time 보다는 run time 에
결정되는 것들이 많다. 그러나 다음과 같이 class 의 인스턴스를 `id` 형으로
저장하지 않는 다면 `static typing, type checking` 을 할 수 있다. 곧, static
behavior 가 가능해진다.

```objc
// static typing
Rectangle *thisObject = [[Square alloc] init];

// type checking
Shape     *aShape;
Rectangle *aRect;
aRect = [[Rectangle alloc] init];
aShape = aRect;
```

## Selectors

컴파일러는 메소드의 이름과 유니크한 아이디를 테이블 형태로 관리한다. 유니크한
아이디가 곧 실렉터이고 다음과 같이 선언할 수 있다.

```objc
SEL setWidthHeight;
setWidthHeight = @selector(setWidth:height:);
...
setWidthHeight = NSSelectorFromString(aBuffer);
...
NSString *method;
method = NSStringFromSelector(setWidthHeight);
```

실렉터는 `performSelector:, performSelector:withObject:,
performSelector:withObject:withObject:` 과 같이 NSObject의 메소드로 호출 할 수
있다. 다음은 호출의 예이다.

```objc
[friend performSelector:@selector(gossipAbout:)
withObject:aNeighbor];
//[friend gossipAbout:aNeighbor];
...
id helper = getTheReceiver();
SEL request = getTheSelector();
[helper performSelector:request];
```

## Exception Handling

`@try, @catch, @throw, @finally` 등으로 exception handling 한다.

```c
// Basic Exception
Cup *cup = [[Cup alloc] init];
@try {
    [cup fill];
} @catch (NSException *ex) {
    NSLog(@"main: Caught %@: %@", [ex name], [ex reason]);
} @finally {
    [cup release];
}

// Catching Different Types of Exception
@try {
    ...
} @catch (CustomException *ce) {
    ...
} @catch (NSException * ne) {
    ...
} @catch (id ue) {
    ...
} @finally {
    ...
}

// Throwing Exceptions
NSException *ex = [NSException exceptionWithName:@"HotTeaException" reason:@"The tea is too hot" userInfo:nil];
@throw exception;
```

## Threading

```objc
// locking a method using self
- (void)Foo {
    @synchronized(self) {
        // ciritical code
        ...
    }
}

// locking a method using a custom semaphore
Account* account = [Account accountFromString:[ccountField stringValue]];
id accountSemaphore = [Account semaphore];
@synchronized(accountSemaphore) {
    // ciritical code.
    ...
}
```

## Remote Messaging

updating...

## Using C++ With Objective-C

objc 에서 c++ 를 사용하고 싶다면 `*.mm` 를 작성해야 한다. objc 는 c++ 의 virtual
function 을 지원하지 않는다. 그리고 c++ 의 constructor, destructor 를 호출하지
않는다.

```c
/* Hello.mm
* Compile with: g++ -x objective-c++ -framework Foundation Hello.mm  -o hello
*/
#import <Foundation/Foundation.h>
class Hello {
    private:
        id greeting_text;  // holds an NSString
    public:
        Hello() {
            greeting_text = @"Hello, world!";
        }
        Hello(const char* initial_greeting_text) {
            greeting_text = [[NSString alloc]
initWithUTF8String:initial_greeting_text];
        }
        void say_hello() {
            printf("%s\n", [greeting_text UTF8String]);
        } 
};

@interface Greeting : NSObject {
    @private
        Hello *hello;
}
- (id)init;
- (void)dealloc;
- (void)sayGreeting;
- (void)sayGreeting:(Hello*)greeting;
@end

@implementation Greeting
- (id)init {
    if (self = [super init]) {
        hello = new Hello();
}
    return self;
}
- (void)dealloc {
    delete hello;
    [super dealloc];
}
- (void)sayGreeting {
    hello->say_hello();
}
- (void)sayGreeting:(Hello*)greeting {
    greeting->say_hello();
}
@end
int main() {
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    Greeting *greeting = [[Greeting alloc] init];
    [greeting sayGreeting];
    Hello *hello = new Hello("Bonjour, monde!");
    [greeting sayGreeting:hello];
    delete hello;
    [greeting release];
    [pool release];
    return 0;
}
```