- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
  - [Compile, Execution](#compile-execution)
  - [Reserved Words](#reserved-words)
  - [Useful Keywords](#useful-keywords)
  - [min, max values](#min-max-values)
  - [abs vs fabs](#abs-vs-fabs)
  - [Bit Maniulation](#bit-maniulation)
  - [String](#string)
  - [Random](#random)
  - [Sort](#sort)
  - [Search](#search)
  - [virtual function](#virtual-function)
  - [vector vs deque vs list](#vector-vs-deque-vs-list)
  - [vector](#vector)
    - [pros](#pros)
    - [cons](#cons)
  - [deque (double ended queue)](#deque-double-ended-queue)
    - [pros](#pros-1)
    - [cons](#cons-1)
  - [list](#list)
    - [pros](#pros-2)
    - [cons](#cons-2)
  - [priority_queue](#priority_queue)
  - [How to choose a container](#how-to-choose-a-container)
- [Advanced](#advanced)
  - [RAII (Resource Acquisition Is Initialzation)](#raii-resource-acquisition-is-initialzation)
  - [Compiler Generated Code](#compiler-generated-code)
  - [Disallow the use of compiler generated functions](#disallow-the-use-of-compiler-generated-functions)
  - [Declare a destructor virtual in polymorphic base classes](#declare-a-destructor-virtual-in-polymorphic-base-classes)
  - [Never call virtual functions in constructor or destructor](#never-call-virtual-functions-in-constructor-or-destructor)
  - [Named Parameter Idiom](#named-parameter-idiom)
  - [new delete](#new-delete)
  - [casting](#casting)
  - [const](#const)
  - [lvalue and rvalue](#lvalue-and-rvalue)
  - [ADL(Argument Dependent Lookup)](#adlargument-dependent-lookup)
  - [typename vs class in template](#typename-vs-class-in-template)
  - [size() infinite loop](#size-infinite-loop)
  - [Concurrent Programming](#concurrent-programming)
  - [C++ Unit Test](#c-unit-test)
  - [Boost Library](#boost-library)
- [Quiz](#quiz)
- [STL](#stl)
- [C++11](#c11)
- [Effective CPP](#effective-cpp)
- [Design Patterns](#design-patterns)

-----

# Abstract

c++에 대해 정리한다.

# Materials

* [C++ Tips of the Week](https://abseil.io/tips/)
  * Abseil is a c++ library. They provide tips of the week.
* [cracking the coding interview](http://www.crackingthecodinginterview.com/)
  * c/c++ quiz 
- [c++ programming](http://boqian.weebly.com/c-programming.html)
  - boqian의 동영상 강좌
- [혼자 연구하는 c/c++](http://soen.kr/)
  - 김상형님의 강좌
- [프로그래밍 대회: C++11 이야기 @ slideshare](https://www.slideshare.net/JongwookChoi/c11-draft?ref=https://www.acmicpc.net/blog/view/46)
- [c++ language](http://en.cppreference.com/w/cpp/language)
- [cplusplus.com](https://www.cplusplus.com)
- [c++11FAQ](http://pl.pusan.ac.kr/~woogyun/cpp11/C++11FAQ_ko.html)

# Basic

## Compile, Execution

```bash
$ g++ -std=c++11 -o a.out a.cpp
$ ./a.out
```

## Reserved Words

* [c++ keywords](https://en.cppreference.com/w/cpp/keyword)
  
----

```cpp
alignas (since C++11)
alignof (since C++11)
and
and_eq
asm
atomic_cancel (TM TS)
atomic_commit (TM TS)
atomic_noexcept (TM TS)
auto (1)
bitand
bitor
bool
break
case
catch
char
char8_t (since C++20)
char16_t (since C++11)
char32_t (since C++11)
class (1)
compl
concept (since C++20)
const
consteval (since C++20)
constexpr (since C++11)
constinit (since C++20)
const_cast
continue
co_await (since C++20)
co_return (since C++20)
co_yield (since C++20)
decltype (since C++11)
default (1)
delete (1)
do
double
dynamic_cast
else
enum
explicit
export (1) (3)
extern (1)
false
float
for
friend
goto
if
inline (1)
int
long
mutable (1)
namespace
new
noexcept (since C++11)
not
not_eq
nullptr (since C++11)
operator
or
or_eq
private
protected
public
reflexpr (reflection TS)
register (2)
reinterpret_cast
requires (since C++20)
return
short
signed
sizeof (1)
static
static_assert (since C++11)
static_cast
struct (1)
switch
synchronized (TM TS)
template
this
thread_local (since C++11)
throw
true
try
typedef
typeid
typename
union
unsigned
using (1)
virtual
void
volatile
wchar_t
while
xor
xor_eq
```

## Useful Keywords

WIP

## min, max values

```cpp
// int
printf("%d\n", INT_MAX);
print("%d\n", std::numeric_limits<int>::max());
printf("%d\n", INT_MIN);
print("%d\n", std::numeric_limits<int>::min());

// float
print("%f\n", std::numeric_limits<float>::max());
print("%f\n", std::numeric_limits<float>::min());
```

## abs vs fabs

`abs(int n)` 는 `cstdlib` 에 정의되어 있고 `fabs(double n)` 는 `cmath` 에 정의되어 있다.

## Bit Maniulation

```cpp
```

## String

```cpp
string a = "Hello World";
// Sub string
string b = a.substr(0, 5); // Hello
// Convert string, int
int n = stoi("12");
string s = to_string(n);
```

## Random

```cpp
int num = random();
```

## Sort

```cpp
vector<int> a = {5, 4, 3, 2, 1};
sort(a.begin(), a.end());  // 1 2 3 4 5
sort(a.begin(), a.end(), [](int a, int b) {
  return a < b;
});  // 5 4 3 2 1
```

## Search

```cpp
// Search Position
itr = lower_bound(vec.begin(), vec.end(), 9);  // vec[1]  
// Find the first position where 9 could be inserted and still keep the sorting.

itr = upper_bound(vec.begin(), vec.end(), 9);  // vec[4] 
// Find the last position where 9 could be inserted and still keep the sorting.
```

## virtual function

![](img/virtualfunction.png)

virtual function 은 vptr, vtable 에 의해 구현된다. `vtable` 은 virtual function 주소들의 배열이다. `vptr` 은 `vtable` 을 가리키는 포인터이다. 임의의 class 가 virtual function 이 하나라도 있다면 runtime 에서 vptr 이 만들어진다. 아래와 같은 예에서 `vptr` 이 4 byte 이면 `sizeof(Instrument) == 4` 이다.

그리고 vptr 은 vtable 을 가리킨다. 따라서 아래의 예에서 `wp->play()` 를 호출하면 `wp->vptr->Brass::vtable->Brass::play()` 를 호출하게 된다.

다음은 위 그림의 구현이다.

```cpp
class Instrument {
public:
  virtual void play() {
  }
  virtual void what() {
  }
  virtual void adjust() {
  }
};

class Wind : public Instrument {
public:
  virtual void play() {
  }
  virtual void what() {
  }
  virtual void adjust() {
  }
}

class Percussion : public Instrument {
public:
  virtual void play() {
  }
  virtual void what() {
  }
  virtual void adjust() {
  }
}
class Stringed : public Instrument {
public:
  virtual void play() {
  }
  virtual void what() {
  }
  virtual void adjust() {
  }
}

class Brass : public Instrument {
public:
  virtual void play() {
  }
  virtual void what() {
  }
  virtual void adjust() {
  }
}

int main() {
  WindPercussion* wp;
  Brass br;
  Instrument inst;

  wp = &br;
  //wp = &inst;
  // sizeof(br) is 4 because of vtpr
}
```

## vector vs deque vs list

|     | vector | deque | list |
|:---:|:---:|:---:|:---:|
| 인덱스접근 | o | o | x |
| 확장방법 | 전체재할당 | chunk 추가할당 | 불필요 |
| 중간삽입 | O(n) | O(n) | O(1)|

## vector

### pros

- 동적으로 확장 및 축소가 가능하다. dynamic array 로 구현되어 있다. 재할당 방식이다. 메모리가 연속으로 할당되어 있어 포인터 연산이 가능하다.
- index로 접근 가능하다. O(1)

### cons

- 끝이 아닌 위치에 삽입 및 제거시 성능이 떨어진다.
- 동적으로 확장 및 축소할때 전체를 재할당 하므로 비용이 크다.

## deque (double ended queue)

### pros

- index로 접근 가능하다. O(1)
- 끝이 아닌 위치에 삽입 및 제거시 성능이 좋다. O(1)
- 동적으로 확장 될때 일정한 크기만큼 chuck가 하나 더 할당되는 방식이다.
  저장 원소가 많거나 원소의 크기가 클때 즉 메모리 할당이 큰 경우 
  vector 에 비해 확장 비용이 적다.

### cons

- 메모리가 연속으로 할당되어 있지 않아 vector 와 달리 포인터 연산이 불가능하다.

## list

### pros

- vector, deque와 달리 임의의 위치에 삽입 및 제거시 성능이 좋다. `O(1)`

### cons

- index 로 접근 불가능하다. 

## priority_queue

다음의 내용을 기억한다.

* `Comparator 의 operator() 가 참을 리턴하면 아래값이다` 를 기억하자.
* `less<int>` 이면 작은 값이 아래 값이다. 기본으로 사용한다.
* `greater<int>` 이면 큰 값이 아래 값이다.

다음은 `int` 가 element 이고 element 에 `-` 부호를 사용한 경우이다.

```cpp
class Solution {
public:
  int connectSticks(vector<int>& sticks) {
    priority_queue<int> pq;
    for (int len : sticks) {
      pq.push(-len);
    }
    int ans = 0;
    while (pq.size() > 1) {
      int sum = -pq.top(); pq.pop();
      sum += -pq.top(); pq.pop();
      ans += sum;
      pq.push(-sum);
    }
    return ans;
  }
};
```

다음은 `int` 가 element 이고 `greater<int>` 를 사용한 경우이다.

```cpp
class Solution {
public:
  int connectSticks(vector<int>& sticks) {
    priority_queue<int, vector<int>, greater<int>> pq;
    for (int len : sticks) {
      pq.push(len);
    }
    int ans = 0;
    while (pq.size() > 1) {
      int sum = pq.top(); pq.pop();
      sum += pq.top(); pq.pop();
      ans += sum;
      pq.push(sum);
    }
    return ans;
  }
};
```

다음은 `int` 가 element 이고 `Comparator` 를 정의한 경우이다.

```cpp
// greater<int>
struct Comparator {
  bool operator() (int a, int b) {
    return a > b;
  }
};
class Solution {
public:
  int connectSticks(vector<int>& sticks) {
    priority_queue<int, vector<int>, Comparator> pq;
    for (int len : sticks) {
      pq.push(len);
    }
    int ans = 0;
    while (pq.size() > 1) {
      int sum = pq.top(); pq.pop();
      sum += pq.top(); pq.pop();
      ans += sum;
      pq.push(sum);
    }
    return ans;
  }
};
```

다음은 `pair<string, int>` 가 element 인 경우이다.

```cpp
#include <cstdio>
#include <queue>
#include <string>

using namespace std;

struct Compare {
  bool operator() (const pair<string, int>& a,
                   const pair<string, int>& b) {
    if (a.second == b.second) {
      return a.first > b.first;
    }
    return a.second < b.second;
  }
};
int main() {
  priority_queue<pair<string, int>,
                 vector<pair<string, int>>,
                 Compare> pq;
  pq.push({"Hello", 1});
  pq.push({"World", 2});
  pq.push({"Foo", 3});
  pq.push({"Bar", 3});
  // Bar Foo World Hello
  while (!pq.empty()) {
    printf("%s\n", pq.top().first.c_str());
    pq.pop();
  }
  return 0;
}
```

## How to choose a container

[C++ Containers Cheat Sheet](http://homepages.e3.net.nz/~djm/cppcontainers.html)

![](img/containerchoice.png)

# Advanced

## RAII (Resource Acquisition Is Initialzation)

* [RAII는 무엇인가](https://blog.seulgi.kim/2014/01/raii.html)

----

resource 생성이 곧 초기화이다 라는 의미이다. scope 을 벗어나면 자동으로 초기화되도록 구현하는 것이다. 

c++ 는 finally 가 없다. RAII 때문이다. C++의 아버지이자 RAII 라는 용어를 처음 만든 Bjarne Stroustrub 는 "RAII가 있는데 굳이 있을 필요가 없다." 라고 말했다.

아래는 `unique_ptr` 를 이용하여 RAII 을 구현한 예이다.

```cpp
void unsafeFunction() {
  Resource* resource = new Resource();
  /* Do something with resource */
  thisFunctionCanThrow예외();
  /* Do something else with resource */
  delete resource;
}

void unmainta
nableFunct i on(

) {
  Resource* resource = nullptr;
  try {
 
    resource 
  = n
  ew Resource();
    /* Do something with resource */
    thisFunctionCanThrowException();
    /* Do something else with resource */
    delete resource;
  catch(std::exception& e) {
    delete resource;
    throw e;
  }
}

void safeFunction() {
  std::unique_ptr<Resource> resource(new Resource());
  /* Do something with resource */
  thisFunctionCanThrowException();
  /* Do something else with resource */
}
```

## Compiler Generated Code

compiler 는 경우에 따라 `Copy constructor, Copy Assignment Operator, Destructor, Default Constructor` 를 생성해 준다.

```cpp
/*
Compiler silently writes 4 functions if they are not explicitly declared:
1. Copy constructor.
2. Copy Assignment Operator.
3. Destructor.
4. Default constructor (only if there is no constructor declared).
*/

class dog {};

/* equivalent to */

class dog {
	public:
		dog(const dog& rhs) {...};   // Member by member initialization

		dog& operator=(const dog& rhs) {...}; // Member by member copying

		dog() {...};  // 1. Call base class's default constructor; 
		              // 2. Call data member's default constructor.

		~dog() {...}; // 1. Call base class's destructor; 
		              // 2. Call data member's destructor.
}
/*
Note:
1. They are public and inline.
2. They are generated only if they are needed.
*/
```

## Disallow the use of compiler generated functions

`delete` 을 이용하여 컴파일러가 코드를 생성하지 못하도록 할 수 있다.

```cpp
class dog {
   public:
   dog(const dog&) = delete; // Prevent copy constructor from being used.
                              // Useful when dog holds unsharable resource.
}
```

## Declare a destructor virtual in polymorphic base classes

`destructor` 를 `virtual` 로 다형성을 실현할 수 있다. 예를 들어서 `yellowdog` 클래스가 `dog` 을 상속받는다고 하자. `yellowdog` 의 `destructor` 가 호출되게 하려면 어떻게 해야할까? `virtual destructor` 를 사용하거나 `shared_ptr` 을 사용한다.

```cpp
/* Problem */
class yellowdog : public dog {
};

dog* dogFactory::createDog() {
	dog* pd = new yellowdog();
	return pd;
}

int main() {
	dog* pd = dogFactory::createDog();
	...
	delete pd;  // Only dog's destructor is invoked, not yellowdog's.
}

/*
Solution: 
*/
class dog {
      virtual ~dog() {...}
}

/* 
Note: All classes in STL have no virtual destructor, so be careful inheriting 
from them.
*/

/*
When we should use virtual destructor:
Any class with virtual functions should have a virtual destructor.

When not to use virtual destructor:
1. Size of the class needs to be small;
2. Size of the class needs to be precise, e.g. passing an object from C++ to C.
*/

/* Solution 2: 
 *    using shared_prt
 */

class Dog {
public:
   ~Dog() {  cout << "Dog is destroyed"; }
};

class Yellowdog : public Dog {
public:
   ~Yellowdog() {cout << "Yellow dog destroyed." <<endl; }
};


class DogFactory {
public:
   //static Dog* createYellowDog() { return (new Yellowdog()); }
   static shared_ptr<Dog> createYellowDog() { 
      return shared_ptr<Yellowdog>(new Yellowdog()); 
   }
   //static unique_ptr<Dog> createYellowDog() {
   //   return unique_ptr<Yellowdog>(new Yellowdog());
   //}

};

int main() {

   //Dog* pd = DogFactory::createYellowDog();
   shared_ptr<Dog> pd = DogFactory::createYellowDog();
   //unique_ptr<Dog> pd = DogFactory::createYellowDog();
   
   //delete pd;
   
	return 0;
}
/*Note: you cannot use unique_ptr for this purpose */
```

## Never call virtual functions in constructor or destructor

`Constructor` 혹은 `Destructor` 에서 `virtual function` 을 호출하지 말자. 객체의 생명주기에 따라 호출이 안될 수 있기 때문이다. 

아래의 예에서 `dog` 의 constructor 에서 `bark()` 를 호출하면 `vptr` 이 아직 만들어지지 않았기 때문에 `dog::bark()` 가 호출된다.

`yellowdog` 의 생성자에서 `dog` 의 생성자에 `color` 를 전달하는 식으로 해결할 수 있다.

```cpp
class dog {
 public:
  string m_name;
  dog(string name) {
    m_name = name; 
    bark();
  }
  virtual void bark() { 
    cout<< "Woof, I am just a dog " << m_name << endl;
  }
};

class yellowdog : public dog {
 public:
  yellowdog(string name) : dog(string name) {...}
  virtual void bark() {
    cout << "Woof, I am a yellow dog " << m_name << endl; 
  }
};

int main ()
{
  yellowdog mydog("Bob");
}

OUTPUT:
Woof, I am just a dog Bob.

/*
During the construction, all virtual function works like non-virtual function.

Why?
Base class's constructor run before derived class's constructor. So at the 
time of bark(), the yellowlog is not constructed yet.

Why Java behaves differently?

There is a fundamental difference in how Java and C++ defines an object's Life time.
Java: All members are null initialized before constructor runs. Life starts before constructor.
C++: Constructor is supposed to initialize members. Life starts after constructor is finished.

Calling down to parts of an object that haven not yet initialized is inherently dangerous.
*/

/*
solution 1:
*/
class dog {
 public:
  ...
  dog(string name, string color) { 
    m_name = name; 
    bark(color);
  }
  void bark(string str) { 
    cout<< "Woof, I am "<< str << " dog " << m_name << endl;
  }
};

class yellowdog : public dog {
 public:
  yellowdog(string name) : dog(name, "yellow") {}
};

int main ()
{
  yellowdog mydog("Bob");
}

OUTPUT:
Woof, I am yellow dog Bob

/*
solution 2:
*/
class dog {
 public:
  ...
  dog(string name, string woof) {
    m_name = name; 
    bark(woof);
  }
  dog(string name) {
    m_name = name; 
    bark( getMyColor() );
  }
  void bark(string str) { 
    cout<< "Woof, I am "<< str << "private:"; 
  }
 private:
  static string getMyColor() {
    return "just a";
  } 
};

class yellowdog : public dog {
 public:
  yellowdog(string name) : dog(name, getMyColor()) {}
 private:
  static string getMyColor() {
    return "yellow";
  }  //Why static?
};

int main ()
{
  yellowdog mydog("Bob");
}
OUTPUT:
Woof, I am yellow dog Bob
```

## Named Parameter Idiom

python 처럼 named parameter 를 흉내내보자.

```cpp

class OpenFile {
 public:
  OpenFile(
    string filename, 
    bool readonly=true, 
    bool appendWhenWriting=false, 
    int blockSize=256, 
    bool unbuffered=true, 
    bool exclusiveAccess=false);
}

OpenFile pf = OpenFile("foo.txt", true, false, 1024, true, true);
// Inconvenient to use
// Unreadable
// Inflexible

// What's ideal is:
OpenFile pf = OpenFile(.filename("foo.txt"), .blockSize(1024) );

/* Solution */
class OpenFile {
public:
  OpenFile(std::string const& filename);
  OpenFile& readonly()  { 
    readonly_ = true; 
    return *this; 
  }
  OpenFile& createIfNotExist() { 
    createIfNotExist_ = true; 
    return *this; 
  }
  OpenFile& blockSize(unsigned nbytes) { 
    blockSize_ = nbytes; 
    return *this; 
  }
  ...
};

OpenFile f = OpenFile("foo.txt")
           .readonly()
           .createIfNotExist()
           .appendWhenWriting()
           .blockSize(1024)
           .unbuffered()
           .exclusiveAccess();

OpenFile f = OpenFile("foo.txt").blockSize(1024);
```

## new delete

다음은 `new` 와 `delete` 의 under the hood 이다.

```cpp
   dog* pd = new dog();
/* 
 * Step 1. operator new is called to allocate memory.
 * Step 2. dog's constructor is called to create dog.
 * Step 3. if step 2 throws an exception, call operator delete to free the 
 *         memory allocated in step 1.
 */
   delete pd;
/* 
 * Step 1. dog's destructor is called.
 * Step 2. operator delete is called to free the memory.
 */
```

다음은 `global` `operator new` 의 구현이다. 

```cpp
/*
 * This is how the operator new may look like if you re-implement it:
 *
 * Note: new handler is a function invoked when operator new failed to allocate 
 * memory.
 *   set_new_handler() installs a new handler and returns current new handler.
 */
void* operator new(std::size_t size) throw(std::bad_alloc) {
   while (true) {
      void* pMem = malloc(size);   // Allocate memory
      if (pMem) 
         return pMem;              // Return the memory if successful

      std::new_handler Handler = std::set_new_handler(0);  // Get new handler
      std::set_new_handler(Handler);

      if (Handler)
         (*Handler)();            // Invoke new handler
      else
         throw bad_alloc();       // If new handler is null, throw exception
   }
}
```

다음은 `class` `operator new` 의 구현이다.

```cpp
/* 
 * Member Operator new
 */
class dog {
   ...
   public:
   static void* operator new(std::size_t size) throw(std::bad_alloc) 
   {
      if (size == sizeof(dog))
         customNewForDog(size);
      else
         ::operator new(size);
   }
   ...
};

class yellowdog : public dog {
   int age;
   static void* operator new(std::size_t size) throw(std::bad_alloc) 
};

int main() {
   yellowdog* py= new yellowdog();
}
```

다음은 `class` `operator delete` 의 구현이다.

```cpp
/* Similarly for operator delete */
class dog {
   static void operator delete(void* pMemory) throw() {
      cout << "Bo is deleting a dog, \n";
      customDeleteForDog();
      free(pMemory);
   }
   ~dog() {};
};

class yellowdog : public dog {
   static void operator delete(void* pMemory) throw() {
      cout << "Bo is deleting a yellowdog, \n";
      customDeleteForYellowDog();
      free(pMemory);
   }
};

int main() {
   dog* pd = new yellowdog();
   delete pd;
}

// See any problem?
//
//
// How about a virtual static operator delete?
//
//
// Solution:
//   virtual ~dog() {}
```

우리는 다음과 같은 이유로 `new, delete` 을 customizing 한다.

```cpp
/*
 * Why do we want to customize new/delete
 *
 * 1. Usage error detection: 
 *    - Memory leak detection/garbage collection. 
 *    - Array index overrun/underrun.
 * 2. Improve efficiency:
 *    a. Clustering related objects to reduce page fault.
 *    b. Fixed size allocation (good for application with many small objects).
 *    c. Align similar size objects to same places to reduce fragmentation.
 * 3. Perform additional tasks:
 *    a. Fill the deallocated memory with 0's - security.
 *    b. Collect usage statistics.
 */

/*
 * Writing a GOOD memory manager is HARD!
 *
 * Before writing your own version of new/delete, consider:
 *
 * 1. Tweak your compiler toward your needs;
 * 2. Search for memory management library, E.g. Pool library from Boost.
 */
```

## casting

* [형 변환(static_cast, const_cast, reinterpret_cast , dynamic_cast)](https://recoverlee.tistory.com/48)

----

type conversion 은  `implicit type conversion` 과 `explicit type conversion` 과 같이 두가지가 있다. 이 중 `explicit type conversion` 이 곧 `casting` 에 해당된다.

`casting` 은 `static_cast, dynamic_cast, const_cast, reinterpret_cast` 와 같이 네가지가 있다.

`static_cast` 는 한가지 차이점을 제외하고 `implicit cast` 와 같다. `static_cast` 를 사용하면 문법적 엄격함을 표현할 수 있다. 

```cpp
int  i = 1;
char c = i;  // implicit cast 
char d = static_cast<char>(i); // static_cast 
```

`implicit, static_cast` 는 pointer 에 적용할 때 데이터타입이 같아야 한다. 예를 들어 다음과 같이 `char*` 를 `int*` 로 형변환 하면 에러가 발생한다.

```cpp
  char c = 'A';
  char*d = &c;
  int* e = d;
  int* f = static_cast<int*>(d);

//   g++ -std=c++11 -o a.out a.cpp
// a.cpp:13:8: error: cannot initialize a variable of type 'int *'
//       with an lvalue of type 'char *'
//   int* e = d;
//        ^   ~
// a.cpp:14:12: error: static_cast from 'char *' to 'int *' is not
//       allowed
//   int* f = static_cast<int*>(d);
//            ^~~~~~~~~~~~~~~~~~~~
// 2 errors generated.
```

`static_cast` 는 `implicit cast` 와 한가지 차이점이 있다. 
`implicit cast` 는 `is a` 관계가 성립하는 경우만 형변환이 되지만 `static_cast` 는 `is a` 관계가 성립하는 경우도 허용하고 down cast 도 허용한다. 즉, 상속관계이면 형변환을 허용한다.

```cpp
  CFoo* pFoo = new CFoo();
  CBase* p0 = pFoo;
  CBase* p1 = static_cast<CBase*>(pFoo);

  CBase* p2 = new CFoo();
  CFoo*  p3 = p2; // error
  CFoo*  p4 = static_cast<CFoo*>(p2);
```

`reinterpret_cast` 는 관련없는 포인터 타입을 변환한다. `const, volatile` 을 형변환할 수 없는 점을 제외하고 `()` 와 같다.

```cpp
  char c = 'A';
  char*d = &c;
  // int* e = d; // error
  // int* f = static_cast<int*>(d); //error
  int* g = reinterpret_cast<int*>(d);
  int* h = (int*)d;

  int  i = 1;
  // char j = reinterpret_cast<char>(i); // error
  char k = (char)i;
```

`const_cast` 는 동일한 데이터타입의 pointer, reference 에만 적용가능하다. const 로 선언된 데이터를 수정할 필요가 있을 때 형변환 한다. 역시 `()` 와 같다. 그러나 문법적 엄격함 때문에 `const_cast` 를 사용한다.
또한 volatile 로 선언된 데이터를 형변환할 때도 사용한다.

```cpp
  const CFoo* p0  = new CFoo();
  // CFoo* p1 = p0; // error
  CFoo* p2 = (CFoo*)p0;
  p2->b = 10;
  CFoo* p3 = const_cast<CFoo*>(p0);
  p3->b = 10;

  volatile CFoo foo;
  CFoo* p4 = const_cast<CFoo*>(&foo);
```

`dynamic_cast` 는 runtime 에 형변환을 하면서 검증한다. 즉, 상속관계, is a 관계가 아니면 NULL 을 리턴한다. 다음과 같은 형식으로 사용한다.

```
dynamic_cast<타입>(표현식)
```

* 표현식은 가상함수와 RTTI(Runtime Type Information) 를 포함하는 클래스에 대한 포인터, 참조형, 객체이다.
* 타입은 가상함수와 RTTI 를 포함하는 클래스의 포인터, 참조형이다.

다음은 RTTI 로 사용하는 `type_info` 클래스의 모양이다.

```cpp
class type_info {
    public: virtual ~type_info();
    int operator==(const type_info& rhs) const;
    int operator!=(const type_info& rhs) const;
    int before(const type_info& rhs) const;
    const char* name() const;
    const char* raw_name() const;
    private: void *_m_data;
    char _m_d_name[1];
    type_info(const type_info& rhs);
    type_info& operator=(const type_info& rhs);
};
```

`type_info` 는 runtime 에 사용할 `class` 의 메타정보의 모음이다. 

예를 들어 다음과 같은 경우 `pf2` 는 NULL 이 저장된다. `pb2` 는 유효한 주소가 저장된다. `dynamic_cast` 를 이용하여 캐스트를 한 경우 실행 코드는 `dynamic_cast` 의 표현식에 기술된 객체를 이용하여 RTTI 포인터 테이블을 검색하고, 만약 RTTI 포인터 테이블 상에 일치하는 RTTI 가 존재 한다면 표현식에 기술된 객체의 타입을 변환하여 반환하고, RTTI 포인터 테이블 상에 일치하는 RTTI 가 존재 하지 않는다면 dynamic_cast 는 NULL 을 반환한다.

```cpp
class CBase {
 public:
  int a;
  virtual void fun1() {
    printf("CBase::fun1\n");
  }
  virtual void fun2() {
    printf("CBase::fun2\n");
  }
};
class CFoo : public CBase {
 public:
  int b;
  virtual void fun3() {
    printf("CFoo::fun3\n");
  }
};

int main() {

  CBase* pb1 = new CBase();
  CFoo*  pf1 = new CFoo();
  CFoo*  pf2 = dynamic_cast<CFoo*>(pb1);  // pf2 is NULL
  CBase* pb2 = dynamic_cast<CBase*>(pf1); // pb2 is valid

  printf("%p\n", pf2);
  printf("%p\n", pb2);
// 0x0
// 0x7fc49bc02ae0
  return 0;
}
```
RTTI 는 `typeid` 를 통해 얻어낼 수도 있다.

```cpp
class CBase {
 public:
  int a;
  virtual void fun1() {
    printf("CBase::fun1\n");
  }
  virtual void fun2() {
    printf("CBase::fun2\n");
  }
};
class CFoo : public CBase {
 public:
  int b;
  virtual void fun3() {
    printf("CFoo::fun3\n");
  }
};

int main() {


  CBase* p1 = new CBase();
  CFoo*  p2 = new CFoo();
  const std::type_info& t1 = typeid(p1);
  const std::type_info& t2 = typeid(*p1);

  printf("%s\n", t1.name());
  printf("%s\n", t2.name());
// P5CBase
// 5CBase
    
  return 0;
}
```

다음은 다양한 사용예이다.

```cpp
/*
 * 1. static_cast
 */
int i = 9;
float f = static_cast<float>(i);  // convert object from one type to another
dog d1 = static_cast<dog>(string("Bob"));  // Type conversion needs to be defined.
dog* pd = static_cast<dog*>(new yellowdog()); // convert pointer/reference from one type 
                                              // to a related type (down/up cast)

/*
 * 2. dynamic_cast 
 */
dog* pd = new yellowdog();
yellowdog py = dynamic_cast<yellowdog*>(pd); 
// a. It convert pointer/reference from one type to a related type (down cast)
// b. run-time type check.  If succeed, py==pd; if fail, py==0;
// c. It requires the 2 types to be polymorphic (have virtual function).

/*
 * 3. const_cast
 */                                        // only works on pointer/reference
const char* str = "Hello, world.";         // Only works on same type
char* modifiable = const_cast<char*>(str); // cast away constness of the object 
                                           // being pointed to

/*
 * 4. reinterpret_cast
 */
long p = 51110980;                   
dog* dd = reinterpret_cast<dog*>(p);  // re-interpret the bits of the object pointed to
// The ultimate cast that can cast one pointer to any other type of pointer.

/*
 * C-Style Casting:  
 */
short a = 2000;
int i = (int)a;  // c-like cast notation
int j = int(a);   // functional notation
//   A mixture of static_cast, const_cast and reinterpret_cast
```

## const

기본적인 `const` 사용법은 다음과 같다.

```cpp
int i = 1;
const int* p1 = &i; // data is const, pointer is not
int* const p2 = &i; // pointer is const, data is not
const int* const p3; // data and pointer are both const
int const *p4 = &i; // data is const, pointer is not
// left const of *, data is const
// right const of *, pointer is const
```

`const` 를 `parameters, return value, function` 에 사용해 보자.

```cpp
class Dog {
   int age;
   string name;
public:
   Dog() { age = 3; name = "dummy"; }
   
   // const parameters
   void setAge(const int& a) { age = a; }
   void setAge(int& a) { age = a; }
   
   // Const return value
   const string& getName() {return name;}
   
   // const function
   void printDogName() const { cout << name << "const" << endl; }
   void printDogName() { cout << getName() << " non-const" << endl; }
};

int main() {
   Dog d;
   d.printDogName();
   
   const Dog d2;
   d2.printDogName();
   
}
```

`const function` 이라도 `mutable` 이 사용된 멤버변수는 수정할 수 있다.

```cpp
class BigArray {
   vector<int> v; // huge vector
   mutable int accessCounter;
   
   int* v2; // another int array

public:
   int getItem(int index) const {
      accessCounter++;
      return v[index];
   }
   
    void setV2Item(int index, int x)  {
      *(v2+index) = x;
   }
    
   // Quiz:
   const int*const fun(const int*const& p)const;
 };

 int main(){
    BigArray b;
 }
```

## lvalue and rvalue

`lvalue` 는 메모리 위치를 가지고 있는 `expression` 이다. 따라서 다른 `value` 에 의해 가르켜 지거나 수정될 수 있다. `rvalue` 는 `lvalue` 가 아닌 `expression` 이다. 임시적으로 생성된 것이어서 다른 `value` 에 의해 가르켜 질 수 없고 수정될 수도 없다. `c++11` 은 `rvalue reference` 를 새롭게 소개하고 있다. 

다음은 `lvalue` 의 예이다.

```cpp
int i;        // i is a lvalue
int* p = &i;  // i's address is identifiable
i = 2;    // Memory content is modified

class dog;
dog d1;   // lvalue of user defined type (class)

         // Most variables in C++ code are lvalues
```

다음은 `rvalue` 의 예이다.

```cpp
//Rvalue Examples:
int x = 2;        // 2 is an rvalue
int x = i+2;      // (i+2) is an rvalue
int* p = &(i+2);  // Error
i+2 = 4;     // Error
2 = i;       // Error

dog d1;
d1 = dog();  // dog() is rvalue of user defined type (class)

int sum(int x, int y) { 
  return x+y; 
}
int i = sum(3, 4);  // sum(3, 4) is rvalue

//Rvalues: 2, i+2, dog(), sum(3,4), x+y
//Lvalues: x, i, d1, p
```

`lvalue reference` 는 `lvalue` 만 가르킬 수 있다. 그러나 예외적으로 `const lvalue reference` 의 경우 `rvalue` 를 가르킬 수 있다. 다음은 `lvalue reference` 의 예이다.

```cpp
//Reference (or lvalue reference):
int i;
int& r = i;

int& r = 5;      // Error

//예외: Constant lvalue reference can be assign a rvalue;
const int& r = 5;   //

int square(int& x) { 
  return x * x; 
}
square(i);   //  OK
square(40);  //  Error

//Workaround:
int square(const int& x) { 
  return x * x; 
}  
// square(40) and square(i) work
```

`lvalue` 는 `rvalue` 를 만들 때 `rvalue` 는 `lvalue` 를 만들 때 사용될 수 있다.

```cpp
/*
 * lvalue can be used to create an rvalue
 */
int i = 1;
int x = i + 2; 

int x = i;

/*
 * rvalue can be used to create an lvalue
 */
int v[3];
*(v+2) = 4;
```

function 혹은 operator 가 항상 `rvalue` 만을 리턴하지는 않는다.

```cpp
/*
 * Misconception 1: function or operator always yields rvalues.
 */
int x = i + 3;  
int y = sum(3,4);

int myglobal ;
int& foo() {
   return myglobal;
}
foo() = 50;

// A more common example:
array[3] = 50;  // Operator [] almost always generates lvalue
```

`lvalue` 가 항상 수정될 수는 없다.

```cpp
/*
 * Misconception 2: lvalues are modifiable.
 *
 * C language: lvalue means "value suitable for left-hand-side of assignment"
 */
const int c = 1;  // c is a lvalue
c = 2;   // Error, c is not modifiable.
```

`rvalue` 도 수정될 수 있다.

```cpp
/*
 * Misconception 3: rvalues are not modifiable.
 */
i + 3 = 6;    // Error
sum(3, 4) = 7; // Error

// It is not true for user defined type (class)
class dog;
dog().bark();  // bark() may change the state of the dog object.
```

## ADL(Argument Dependent Lookup)

함수 `g(x)` 를 호출할 때 인자 `x` 의 타입을 고려하여 적당한 함수 `g` 를 찾는 것을 ADL 이라고 한다.

```cpp
// Example 1:
namespace A
{
  struct X {};
  void g(X) {
    cout << " calling A::g() \n"; 
  }
}

void g(X) { 
  cout << " calling ::g() \n"; 
}

int main() {
   A::X x1;
   g(x1);   // Koenig Lookup, or Argument Dependent Lookup (ADL)
}

//Notes:
//1. Remove A:: from A::g(x);
//2. Add a global g(A::X);
// Argument Dependent Lookup (ADL)


/*
 *  Name Lookup Sequence
 *
 *  With namespaces:
 *  current scope => next enclosed scope => ... => global scope 
 *
 *  To override the sequence:
 *  1. Qualifier or using declaration
 *  2. Koenig lookup
 *
 *  With classes:
 *  current class scope => parent scope => ... => global scope
 *
 *  To override the sequence:
 *   - Qualifier or using declaration
 *
 *
 *  Name hiding
 */
```

## typename vs class in template

보통은 `typename` 과 `class` 를 교환해서 사용할 수 있다.

```cpp
template<typename T>
T square(T x) {
   return x * x;
}

template<class T>
T square(T x) {
   return x * x;
}
```

`T::A *aObj;` 과 같은 표현을 살펴보자. 컴파일러는 `*` 을 곱연산자로 해석한다. `T::A` 의 `A` 가 static member 가 아닌 type 이라는 것을 컴파일러에게 알려주기 위해 `typename` 이라는 키워드를 만들었다.

```cpp
template <class T>
class Demonstration {
 public:
  void method() {
    T::A *aObj; // oops …
    // …
  };

template <class T>
class Demonstration {
 public:
  void method() {
    typename T::A* a6; // declare pointer to T’s A
    // …
  };

```

## size() infinite loop

다음과 같은 경우 size() 의 type 이 unsigned 이므로 `` 무한 루프에 빠진다. 

```cpp
  std::vector<int> v;
  for (int i = 0; v.size() - 1 < 0; ++i) {
    printf("loop\n");
  }
```

## Concurrent Programming

* [concurrent programming](cpp_concurrent.md)

## C++ Unit Test

...

## Boost Library

...

# Quiz

* Last K Lines
* Reverse String
* Hash Table vs. STL Map
* Virtual Functions
* Shallow vs. Deep Copy
* Volatile
* Virtual Base Class
* Copy Node
* Smart Pointer
* Malloc
* 20 Alloc

# STL

* [stl](cpp_stl.md)

# C++11

* [c++11](cpp_cpp11.md)

# Effective CPP

* [Effective Cpp](cpp_effective.md)

# Design Patterns

* [Design Patterns in Cpp](cpp_design_pattern.md)
