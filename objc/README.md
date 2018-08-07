- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
    - [Classes](#classes)
    - [Objects and Memory](#objects-and-memory)
    - [Protocols](#protocols)
    - [Methods](#methods)
    - [Properties](#properties)
    - [Categories](#categories)

-------------------------------------------------------------------------------

# Abstract

objc를 c#과 비교하여 정리한다. 

# Materials

* [objective-C 2.0 Programming Language](http://cagt.bu.edu/w/images/b/b6/Objective-C_Programming_Language.pdf)
  * manual
* [api doc](https://developer.apple.com/documentation)
  * api 검색
* [iOS 11 & Swift 4 @ udemy](https://www.udemy.com/ios-11-app-development-bootcamp/learn/v4/overview)
  * 유료이지만 알차다. src는 이메일 인증으로 다운받을 수 있다.
* [Objective-C 강좌 - 12개 앱 만들면서 배우는 iOS 아이폰 앱 개발 @ inflearn](https://www.inflearn.com/course/objective-c-%EA%B0%95%EC%A2%8C/)
  * 유료이다. src는 강좌안 링크로 다운받을 수 있다..
* [objc for dotnet developer](https://timross.wordpress.com/2011/06/12/an-introduction-to-objective-c-for-net-developers/)
* [learn objc](http://cocoadevcentral.com/d/learn_objectivec/)
* [objc faq](http://www.faqs.org/faqs/computer-lang/Objective-C/faq/)
* [effective objective-c 2.0](https://www.amazon.com/Effective-Objective-C-2-0-Specific-Development-ebook/dp/B00CUG5MZA)

# Basic

## Classes

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

## Objects and Memory

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

## Protocols

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

## Methods

objc는 함수를 호출하는 형식이 특이하다. 메시지를 전달한다는 의미이기
때문에 c#과 달리 `[nil print]`를 허용한다. 인자가 여러개인 경우
두번째 인자부터 label을 사용한다. label은 생략할 수도 있다.

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
 
- (BOOL)saveAs:(NSString *)fileName toPath:(NSString *)filePath;
 
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

## Properties

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

## Categories

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

