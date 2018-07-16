- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
  - [virtual function](#virtual-function)
  - [vector vs deque vs list](#vector-vs-deque-vs-list)
  - [vector](#vector)
    - [pros](#pros)
    - [cons](#cons)
  - [deque (double ended queue)](#deque-double-ended-queue)
    - [pros](#pros)
    - [cons](#cons)
  - [list](#list)
    - [pros](#pros)
    - [cons](#cons)
  - [How to choose a container](#how-to-choose-a-container)
- [Advanced](#advanced)
  - [const](#const)
  - [lvalue and rvalue](#lvalue-and-rvalue)
- [Basic STL](#basic-stl)
- [Advanced STL](#advanced-stl)
- [Concurrent Programming](#concurrent-programming)
- [C++11](#c11)
  - [auto](#auto)
  - [range based for](#range-based-for)
  - [initializer lists](#initializer-lists)
  - [in-class member initializers](#in-class-member-initializers)
  - [tuple](#tuple)
  - [advanced STL container](#advanced-stl-container)
  - [lambda](#lambda)
  - [move semantics](#move-semantics)
  - [Value Categories](#value-categories)
  - [r-value reference](#r-value-reference)
  - [Perfect Forwarding](#perfect-forwarding)
  - [move constructor](#move-constructor)
  - [array](#array)
  - [timer](#timer)
  - [regex](#regex)
  - [random](#random)
  - [thread](#thread)
  - [to_string](#tostring)
  - [convert string](#convert-string)
- [C++ Unit Test](#c-unit-test)
- [Boost Library](#boost-library)

-----

# Abstract

c++에 대해 정리한다.

# Materials

- [c++ programming](http://boqian.weebly.com/c-programming.html)
  - boqian의 동영상 강좌
- [혼자 연구하는 c/c++](http://soen.kr/)
  - 김상형님의 강좌
- [프로그래밍 대회: C++11 이야기 @ slideshare](https://www.slideshare.net/JongwookChoi/c11-draft?ref=https://www.acmicpc.net/blog/view/46)
- [c++ language](http://en.cppreference.com/w/cpp/language)
- [cplusplus.com](https://www.cplusplus.com)
- [c++11FAQ](http://pl.pusan.ac.kr/~woogyun/cpp11/C++11FAQ_ko.html)

# Basic

## virtual function

![](img/virtualfunction.png)

virtual function 은 vptr, vtable 에 의해 구현된다. 다음과 같이 `Instrument, Wind, Percussion, Stringed, Brass` 를 정의해 보자.

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
```

각각의 child class 들의 object 들은 `vptr, vtable` 을 갖는다.

## vector vs deque vs list

|     | vector | deque | list |
|:---:|:---:|:---:|:---:|
| 인덱스접근 | o | o | x |
| 확장방법 | 전체재할당 | chunk 추가할당 | 불필요 |
| 중간삽입 | O(n) | O(n) | O(1)|

## vector

### pros

- 동적으로 확장 및 축소가 가능하다. dynamic array로 구현되어 있다.
  재할당 방식이다. 메모리가 연속으로 할당되어 있어 포인터 연산이 가능하다.
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
  vector에 비해 확장 비용이 적다.

### cons

- 메모리가 연속으로 할당되어 있지 않아 vector와 달리 포인터 연산이 불가능하다.

## list

### pros

- vector, deque와 달리 임의의 위치에 삽입 및 제거시 성능이 좋다. O(1)

### cons

- index로 접근 불가능하다. 

## How to choose a container

[C++ Containers Cheat Sheet](http://homepages.e3.net.nz/~djm/cppcontainers.html)

![](img/containerchoice.png)

# Advanced

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

`const function` 이라도 `mutable` 이 사용된 멤버변수를 수정할 수 있다.

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

int sum(int x, int y) { return x+y; }
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

//Exception: Constant lvalue reference can be assign a rvalue;
const int& r = 5;   //

int square(int& x) { return x*x; }
square(i);   //  OK
square(40);  //  Error

//Workaround:
int square(const int& x) { return x*x; }  // square(40) and square(i) work
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
sum(3,4) = 7; // Error

// It is not true for user defined type (class)
class dog;
dog().bark();  // bark() may change the state of the dog object.
```


# Basic STL

# Advanced STL

# Concurrent Programming

# C++11

## auto

- 컴파일 타임에 타입을 자동으로 추론한다.

```cpp
  std::map<std::string, std::string> M = { {"FOO", "foo"}, {"BAR", "bar"} };
  for (auto it = M.begin(); it != M.end(); ++it) {
    std::cout << it->first << " : " << it->second << std::endl;
  }
```

## range based for

```cpp
  for (auto& kv : M) {
    std::cout << kv.first << " : " << kv.second << std::endl;
  }
  // iterate 4 times because of null character
  for (char c : "RGB") {...}
  // iterate 3 times
  for (char c : string("RGB") {...}
```

## initializer lists

- container를 간단히 초기화 할 수 있다.

```cpp
//
vector<int> a = {1, 2, 3, 4};
map<string, int> b = { {"a", 1}, {"b", 2} };
pair<int, long long> c = {3, 4LL};
pair<vector<int>, pair<char, char>> d = { {1, 2, 3}, {'A', 'B'} };
tuple<int, string, int> e = {2222, "Yellow", 22};

std::pair<std::string, std::string> get_name() {
  return {"BAZ", "baz"};
}
//
struct vector3 {
  int x, y, z;
  vector3(int x = 0, int y = 0, int z = 0) : x(x), y(y), z(z) {}
};
Vector3 o = Vector3(0, 0, 0);
Vector3 V = {1, 2, 3}; 
// 생성자 없이 explicit type을 주어 값 생성
Vector3_add(Vector3{0, 0, 0}, Vector3{0, 0, 0});
// 함수 파라메터에 따라 자동으로 type추론이 가능
Vector3_add( {0, 0, 0}, {0, 0, 0});
// old
int min_val = min(x, min(y, z));
// new
int min_val = min({x, y, z});
int max_val = max({x, y, z});
tie(min_val, max_val) = minmax({p, q, r, s});
//
for (const auto & x : {2, 3, 5, 7}) {
  std::cout << x << std::endl;
}
```

## in-class member initializers

- struct, class의 field를 초기화 할 수 있다.

```cpp
class Foo {
    int sum = 0;
    int n;
};
```

## tuple

```cpp
  std::tuple<int, int, int> t_1(1, 2, 3);
  auto t_2 = std::make_tuple(1, 2, 3);
  std::cout << std::get<0>(t_1) << " " <<
      std::get<1>(t_1) << " " <<
      std::get<2>(t_1) << " " << std::endl;
  // tuple, tie
  int a = 3, b = 4;
  std::tie(b, a) = std::make_tuple(1, 2);
  std::cout << a << " " << b << std::endl;

  // tuple list sort
  std::vector<std::tuple<int, int, int> > tv;
  tv.push_back(std::make_tuple(1, 2, 3));
  tv.push_back(std::make_tuple(2, 1, 3));
  tv.push_back(std::make_tuple(1, 1, 3));  
  std::sort(tv.begin(), tv.end());
  for (const auto& x : tv) {
    std::cout << std::get<0>(x) << " " << std::get<1>(x) << " " << std::get<2>(x) << std::endl;
  }

  // // tuple example : lexicographical comparison
  // std::sort(a.begin(), a.end(), [&](const Elem& x, const Elem& y) {
  //     return std::make_tuple(x.score, -x.age, x.submission)
  //         < std::make_tuple(y.score, -y.age, y.submission);
  //   }); 
```

## advanced STL container

```cpp
  // advanced STL container
  // argument가 container의 element의 type의 생성자에 전달된다.
  std::vector<std::pair<int, int> > vvv;
  vvv.push_back(std::make_pair(3, 4));
  vvv.emplace_back(3, 4);
  std::queue<std::tuple<int, int, int> > q;
  q.emplace(1, 2, 3);

  // advanced STL container
  // unordered_set, unordered_map
  // red black tree vs hash
  std::unordered_map<long long, int> pows;
  for (int i = 0; i < 63; ++i)
    pows[1LL << i] = i;
```

## lambda

```cpp
  // lambda function
  // [captures](parameters){body}
  auto func = [](){};
  func();

  // lambda function recursive
  std::function<int(int)> f;
  f = [&f](int x) -> int {
    if (x <= 1)
      return x;
    return f(x - 1) + f(x - 2);
  };

  // lambda stl algorithms
  std::vector<int> primes = {2, 3, 5, 7, 11};
  auto is_even = [](int n){return (n & 1) == 0;};
  bool all_even = std::all_of(primes.begin(), primes.end(), is_even);
```

## move semantics

- Matrix클래스를 사용한 나쁜 예.

```cpp
// bad 
typedef vector<vector<int>> Matrix;
void multiply(const Matrix& A, const Matrix& B, Matrix& C);
Matrix C;
multiply(A, B, C);
print(C);
```
- Matrix클래스를 사용한 좋은 예. host부분을 주목하자. 위에서 처럼
  연산의 결과에 해당하는 Matrix를 미리 선언할 필요 없다.

```cpp
Matrix operator* (const Matrix* A, const Matrix& B) {
  size_t n = A.size();
  Matrix C(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k)
        C[i][j] += A[i][k] + B[k][j];
  return C;
}
Matrix E = (D * D) + I;
E = E * E;
```

- c++11에서 RVO(Return Value Optimization)덕분에 Matrix 값 복사는
  일어나지 않는다. rvalue가 return된다.
  
## Value Categories

![](_img/value_category.png)

- [Value categories @ cppreference](http://en.cppreference.com/w/cpp/language/value_category)
- lvalue
  - `std::cin`, `std::endl`
    - the name of a variable or a function in scope, regarless of type
  - `std::getline(std::cin, str)`, `std::cout << 1`, `str1 = str2`, `++it`
    - a function call or an overloaded operator expression of lvalue reference return type
  - `a = b`, `a += b`, `a %= b`
    - the built-in assignment and compound assignment expressions
  - `++a`, `--a`
    - the built-in pre-increment and pre-decrement expressions
  - `*p`
    - the built-in indirection expression
  - `a[n]`, `p[n]`
    - the built-in subscript expressions
  - `a.m`
    - the member of object expression
  - `p->m`
    - the built-in member of pointer expression
  - `a.*mp`
    - the pointer to member of object expression
  - `p->*mp`
    - the built-in pointer to member of pointer expression
  - `a, b`
    - the built-in comma expression, there b is an lvalue
  - `a ? b : c`
    - the ternary conditional expression for some b and c
  - `"Hello, world"`
    - string literal
  - `static_cast<int&>(x)`
    - a cast expression to lavalue reference type
  - 주소값이 있는 변수
  - same as glvalue
  - address of an lvalue may be taken
    - `&++i`, `&std::endl`
  - a modifiable lvalue may be used as the left-hand operand of the
    built-in assignment and compound assignment operators
  - an lvalue may be used to initialize an lvalue reference; this
    associates a new name with the object identified by the
    expression.
    
- prvalue (pure rvalue)
  - `42`, `true`, `nullptr`
    - a literal except for string literal
  - `str.substr(1, 2)`, `str1 + str2`, `it++`
    - a function call or an overloaded operator expression of non-reference return type
  - `a++`, `a--`
    - the built-in post-increment and post-decrement expressions
  - `a+b`, `a%b`, `a&b`, `a<<b`
    - the built-in arithmetic expressions
  - `a && b`, `a || b`, `!a` 
    - the built-in logical expressions
  - `&a`
    - built-in address-of expression
  - `p->m`
    - built-in member of pointer expression
  - `a.*mp`
    - the pointer to member of object expression
  - `p->*mp`
    - `a, b`
    - the built-in comman expression, where b is an rvalue
  - `a ? b : c`
    - the ternary onditional expression for some b and c
  - `static_cast<double>(x)`, `std::string{}`, `(int)42`
    - a cast expression to non-reference type
  - `this`
  - `[](int x){return x * x;}`
    - a lambda expression
  - same as rvalue
  - a prvalue cannot be polymorphic: the dynamic type of the object it
    identifies is always the type of the expression
  - a non-class non-array prvalue cannot be cv-qualified
  - a prvalue cannot have incomplete type
- xvalue (expiring value)
  - `std::move(x)`
    - a function call or an overloaded operator expression of rvalue
      reference to object return type
  - `a[n]`
    - the built-in subscript expression
  - `a.m`
    - the member of object expression
  - `a.*mp`
    - the pointer to member of object expression
  - `a ? b : c`
    - the ternary conditional expression for some b and c
  - `static_cast<char&&>(x)`
    - a cast expression to rvalue reference to object type
  - same as rvalue
  - same as glvalue
- glvalue (generalized value)
  - lvalue or xvalue
  - a glvalue may be implicitly converted to a prvalue with
    lavalue-to-rvalue, array-to-pointer, or function-to-pointer
    implicit conversion
  - a glvalue may be polymorphic: the dynamic type of the object it
    identifies is not necessarily the static type of the expression
  - a glvalue can have incomplete type, where permitted by the expression
- rvalue 
  - xvalue or prvalue
  - address of an rvalue may not be taken
    - `&int()`, `&i++`, `&42`, `&std::move(x)`
  - an rvalue can't be used as the left-hand operand of the built-in
    assignment or compound assignment operators.
  - an rvalue may be used to initialize a const lvalue reference, in
    which case the lifetime of the object identified by the rvalue is
    extended until the scope of the reference ends.
  
## r-value reference

`rvalue` 는 `move semantics` 혹은 `perfect fowarding` 을 위해 사용된다. 다음은 `rvalue` 의 예이다. `std::move` 는 `lvalue` 를 인자로 받아서 `rvalue` 로 `move semantics` 하여 리턴한다.

```cpp
int a = 5;
int& b = a;   // b is a lvalue reference, originally called reference in C++ 03

int&& c       // c is an rvalue reference

void printInt(int& i) { cout << "lvalue reference: " << i << endl; }
void printInt(int&& i) { cout << "rvalue reference: " << i << endl; } 

int main() {
   int i = 1;
   printInt(i);   // Call the first printInt
   printInt(6);   // Call the second printInt

   printInt(std::move(i));   // Call the second printInt
}
```

`rvalue reference` 는 `move semantics` 를 수행하기 위해 `copy constructor` 혹은 `copy assignment operator` 에서 유용하게 사용된다.

```cpp
/* 
 * Note 1: the most useful place for rvalue reference is overloading copy 
 * constructor and assignment operator, to achieve move semantics.
 */
X& X::operator=(X const & rhs); 
X& X::operator=(X&& rhs);
```

모든 `stl containers` 는 `move semantics` 가 구현되어 있다.

```cpp
/* Note 2: Move semantics is implemented for all STL containers, which means:
 *    a. Move to C++ 11, You code will be faster without changing a thing.
 *    b. You should use passing by value more often.
 */

vector<int> foo() { ...; return myvector; }

void goo(vector<int>& arg);   // Pass by reference if you use arg to carry
                             // data back from goo()
```

## Perfect Forwarding

어떠한 함수가  `rvalue` 를 `rvalue` 로 `lvalue` 를 `lvalue` 로 전달하는 것을 `perfect forwarding` 이라고 한다. 다음의 예에서 `relay` 는 `perfect forwarding` 이 되고 있지 않다.

```cpp
void foo( boVector arg );
// boVector has both move constructor and copy constructor

template< typename T >
void relay(T arg ) {
   foo(arg);  
}

int main() {
   boVector reusable = createBoVector();
   relay(reusable);
   ...
   relay(createBoVector());
}
```

앞서 언급한 예를 다음과 같이 수정하면 `perfect forwarding` 이 가능하다. 

```cpp
// Solution:
template< typename T >
void relay(T&& arg ) {
  foo( std::forward<T>( arg ) );
}

//* Note: this will work because type T is a template type.

/* 
 * Reference Collapsing Rules ( C++ 11 ):
 * 1.  T& &   ==>  T&
 * 2.  T& &&  ==>  T&
 * 3.  T&& &  ==>  T&
 * 4.  T&& && ==>  T&&
 */
```

`remove_reference` 는 `type T` 의 `reference` 를 제거한다.

```cpp
template< classs T >
struct remove_reference;    // It removes reference on type T

// T is int&
remove_refence<int&>::type i;  // int i;

// T is int
remove_refence<int>::type i;   // int i;
```

앞서 언급한 `reference collapsing rules` 에 의해 `relay` 로 전달된 인자가 어떻게 `perfect forwarding` 이 가능해지는지 살펴보자.

```cpp
/*
 * rvalue reference is specified with type&&.
 *
 * type&& is rvalue reference?
 */

// T&& variable is intialized with rvalue => rvalue reference
  relay(9); =>  T = int&& =>  T&& = int&& && = int&&

// T&& variable is intialized with lvalue => lvalue reference
  relay(x); =>  T = int&  =>  T&& = int& && = int&

// T&& is Universal Reference: rvalue, lvalue, const, non-const, etc...
// Conditions:
// 1. T is a template type.
// 2. Type deduction (reference collasing) happens to T.
//    - T is a function template type, not class template type.
//
```

`std:forward` 는 다음과 같이 구현되어 있다.

```cpp
template< typename T >
void relay(T&& arg ) {
  foo( std::forward<T>( arg ) );
}

// Implementation of std::forward()
template<class T>
T&& forward(typename remove_reference<T>::type& arg) {
  return static_cast<T&&>(arg);
} 
```

`std::move` 와 `std::forward` 의 차이는 다음과 같다.

```cpp

// std::move() vs std::forward()
std::move<T>(arg);    // Turn arg into an rvalue type
std::forward<T>(arg); // Turn arg to type of T&&
```

## move constructor

- 값 복사 없이 메모리가 이동하는 형태를 move semantic이라고 한다. move
  semantic이 가능한 생성자를 move constructor라고 한다.

```cpp
template<typename T>
struct vector {
  vector();
  vector(size_t size); // constructor
  vector(vector<T> &a); // copy constructor
  vector(vector<T> &&a); // move constructor
};
```

## array

```cpp
array<int, 6> a = {1, 2, 3};
```

- vector와 다르게 크기가 고정된 배열이다.
- int[]와 다른점은 무엇일까?
  - STL algorithm함수 들에 적용이 가능하다. (ex. begin, end)
  - 값 복사, 사전순 비교 등이 쉽다. 
  - assertive하게 동작 (out-of-index일때 바로 throw)
  
## timer

```cpp
using namespace std::chrono;
auto _start = system_clock::now();
auto _end = system_clock::now();
long millisecs = duration_cast<milliseconds>(_end - _start).count();
```

## regex

```cpp
if (regex_match("ABCD", regex("(A|B)C(.*)D"))) { 
  //... 
}
```

- [a.cpp](library/regex/a.cpp)

## random

```cpp
std::mt19936 eng; // Mersenne Twister
std::uniform_int_distribution<int> U(-100, 100);
for (int i = 0; i < n; ++i)
  cout << U(eng) << std;
```

## thread

```cpp
```

## to_string

```cpp
string s = std::to_string(10);
string s = std::to_string(3.1415926535);
```

## convert string

- stod, stof, stoi, stol, stold, stoll, stoul, stoull

```cpp
int x = stoi("-1");
long long y = stoll("2147483648");
long long z = stoll("1000...0000", 0, 2); // 4294967296
```

# C++ Unit Test

# Boost Library