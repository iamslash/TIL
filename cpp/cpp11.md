
- [nullptr](#nullptr)
- [enum class](#enum-class)
- [static_assert](#staticassert)
- [Delegating Contructor](#delegating-contructor)
- [override](#override)
- [final](#final)
- [default](#default)
- [delete](#delete)
- [constexpr](#constexpr)
- [String Literals](#string-literals)
- [User Defined Literals](#user-defined-literals)
- [auto](#auto)
- [range based for](#range-based-for)
- [initializer lists](#initializer-lists)
- [Uniform Initialization](#uniform-initialization)
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
- [to_string](#tostring)
- [convert string](#convert-string)
- [Variadic Template](#variadic-template)
- [Template Alias](#template-alias)
- [decltype](#decltype)
- [chrono](#chrono)

----

## nullptr

`NULL` 대신 `nullptr` 을 사용하자.

```cpp
void foo(int i) { 
  cout << "foo_int" << endl; 
}
void foo(char* pc) { 
  cout << "foo_char*" << endl; 
}

int main() {
   foo(NULL);    // Ambiguity

   // C++ 11
   foo(nullptr); // call foo(char*)
}
```

## enum class

```cpp
  // C++ 03
  enum apple {green_a, red_a};
  enum orange {big_o, small_o};
  apple a = green_a;
  orange o = big_o;

  if (a == o) 
    cout << "green apple and big orange are the same\n";
  else
    cout << "green apple and big orange are not the same\n";

  // C++ 11
  enum class apple {green, red};
  enum class orange {big, small};
  apple a = apple::green;
  orange o = orange::big;

  if (a == o) 
    cout << "green apple and big orange are the same\n";
  else
    cout << "green apple and big orange are not the same\n";

  // Compile fails because we haven't define ==(apple, orange)
```

## static_assert

```cpp
// run-time assert
  assert(myPointer != NULL);

// Compile time assert (C++ 11)
  static_assert(sizeof(int) == 4);
```

## Delegating Contructor

생성자에서 다른 생성자가 호출되게 할 수 있다.

```cpp
class Dog {
 public:
  Dog() { ... }
  Dog(int a) { 
    Dog(); 
    doOtherThings(a); 
  }
};

// C++ 03:
class Dog {
  init() { ... };
 public:
  Dog() { 
    init(); 
  }
  Dog(int a) { 
    init(); 
    doOtherThings(); 
  }
};
/* Cons:
 * 1. Cumbersome code.
 * 2. init() could be invoked by other functions.
 */

// C++ 11:
class Dog {
  int age = 9;
 public:
  Dog() { ... }
  Dog(int a) : Dog() { 
    doOtherThings(); 
  }
};
// Limitation: Dog() has to be called first.
```

## override

클래스의 멤버 함수를 오버라이딩할 때 사용한다.

```cpp
// C++ 03
class Dog {
  virtual void A(int);
  virtual void B() const;
}

class Yellowdog : public Dog {
  virtual void A(float);  // Created a new function
  virtual void B(); // Created a new function 
}

// C++ 11
class Dog {
  virtual void A(int);
  virtual void B() const;
  void C();
}

class Yellowdog : public Dog {
  virtual void A(float) override;  // Error: no function to override
  virtual void B() override;       // Error: no function to override
  void C() override;               // Error: not a virtual function
}
```

## final

더이상 오버라이딩을 못하게 할 때 함수에 사용한다. 더이상 상속을 못하게 할 때 클래스에 사용한다.

```cpp
class Dog final {    // no class can be derived from Dog
   ...
};
   
class Dog {
  virtual void bark() final;  // No class can override bark() 
};
```

## default

기본 생성자가 강제로 생성되도록 하기 위해 `default` 키워드를 사용한다.

```cpp
class Dog {
  Dog(int age) {}
};

Dog d1;  // Error: compiler will not generate the default constructor

// C++ 11:
class Dog {
  Dog(int age);
  Dog() = default;    // Force compiler to generate the default constructor
};
```

## delete

클래스의 멤버 함수를 사용하고 싶지 않을 때 쓴다.

```cpp
class Dog {
  Dog(int age) {}
}

Dog a(2);
Dog b(3.0); // 3.0 is converted from double to int
a = b;     // Compiler generated assignment operator

// C++ 11:
class Dog {
  Dog(int age) {}
  Dog(double) = delete;
  Dog& operator=(const Dog&) = delete;
}
```

## constexpr

컴파일 타임때 평가 되도록 하기 위해 함수에 사용한다.

```cpp
int arr[6];    //OK
int A() { 
  return 3; 
}
int arr[A()+3];   // Compile Error 

// C++ 11
constexpr int A() { 
  return 3; 
}  
// Forces the computation to happen 
// at compile time.
int arr[A()+3];   // Create an array of size 6

// Write faster program with constexpr
constexpr int cubed(int x) { 
  return x * x * x; 
}

int y = cubed(1789);  // computed at compile time

//Function cubed() is:
//1. Super fast. It will not consume run-time cycles
//2. Super small. It will not occupy space in binary.
```

## String Literals

utf8, utf16, utf32 등을 문자열 상수로 표현할 수 있다.

```cpp
  // C++ 03:
  char*     a = "string";  

  // C++ 11:
  char*     a = u8"string";  // to define an UTF-8 string. 
  char16_t* b = u"string";   // to define an UTF-16 string. 
  char32_t* c = U"string";   // to define an UTF-32 string. 
  char*     d = R"string \\"    // to define raw string. 
```

## User Defined Literals

`3.4cm` 와 같은 커스텀 상수를 만들 수 있다.

```cpp
// C++ went a long way to make user defined types (classes) to behave same as buildin types.
// User defined literals pushes this effort even further

//Old C++:
long double height = 3.4;

// Remember in high school physics class?
height = 3.4cm;
ratio = 3.4cm / 2.1mm; 

//Why we don't do that anymore?
// 1. No language support
// 2. Run time cost associated with the unit translation

// C++ 11:
long double operator"" _cm(long double x) {
  return x * 10; 
}
long double operator"" _m(long double x) { 
  return x * 1000; 
}
long double operator"" _mm(long double x) {
  return x; 
}

int main() {
   long double height = 3.4_cm;
   cout << height  << endl;              // 34
   cout << (height + 13.0_m)  << endl;   // 13034
   cout << (130.0_mm / 13.0_m)  << endl; // 0.01
}

//Note: add constexpr to make the translation happen in compile time.

// Restriction: it can only work with following paramters:
  char const*
  unsigned long long
  long double
  char const*, std::size_t
  wchar_t const*, std::size_t
  char16_t const*, std::size_t
  char32_t const*, std::size_t
// Note: return value can be of any types.

// Example:
int operator"" _hex(char const* str, size_t l) { 
  // Convert hexdecimal formated str to integer ret
  return ret;
}

int operator"" _oct(char const* str, size_t l) { 
  // Convert octal formated str to integer ret
  return ret;
}

int main() {
  cout << "FF"_hex << endl;  // 255
  cout << "40"_oct << endl;  // 32
}
```

## auto

- 컴파일 타임에 타입을 자동으로 추론한다.

```cpp
  std::map<std::string, std::string> M = { {"FOO", "foo"}, {"BAR", "bar"} };
  for (auto it = M.begin(); it != M.end(); ++it) {
    std::cout << it->first << " : " << it->second << std::endl;
  }

  std::vector<int> vec = {2, 3, 4, 5};

// C++ 03
for (std::vector<int>::iterator it = vec.begin(); it!=vec.end(); ++ it)
  m_vec.push_back(*it);

// C++ 11: use auto type
for (auto it = vec.begin(); it!=vec.end(); ++ it)
  m_vec.push_back(*it);

auto a = 6;    // a is a integer
auto b = 9.6;  // b is a double
auto c = a;    // c is an integer
auto const x = a;   // int const x = a
auto& y = a;        // int& y = a

// It's static type, no run-time cost, fat-free.
// It also makes code easier to maintain.

// 1. Don't use auto when type conversion is needed
// 2. IDE becomes more important
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

// C++ 03:
  for (vector<int>::iterator itr = v.begin(); itr!=v.end(); ++ itr)
    cout << (*itr);

// C++ 11:
  for (auto i : v) { // works on any class that has begin() and end()
    cout << i ;    // readonly access
   }

  for (auto& i : v) {
    i++;                 // changes the values in v
  }                       // and also avoids copy construction

  auto x = begin(v);  // Same as: int x = v.begin();

  int arr[4] = {3, 2, 4, 5};
  auto y = begin(arr); // y == 3
  auto z = end(arr);   // z == 5
  // How this worked? Because begin() and end() are defined for array.
  // Adapt your code to third party library by defining begin() and end()
  // for their containers.
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
// Define your own initializer_list constructor:
#include <initializer_list>
class BoVector {
   vector<int> m_vec;
   public:
   BoVector(const initializer_list<int>& v) {
      for (initializer_list<int>::iterator itr = v.begin(); itr!=v.end(); ++ itr)
        m_vec.push_back(*itr);
   }
};

BoVector v = {0, 2, 3, 4};
BoVector v{0, 2, 3, 4};   // effectively the same

// Automatic normal Initialization
class Rectangle {
   public:
   Rectangle(int height, int width, int length){ }
};

void draw_rect(Rectangle r);

int main() {
   draw_rect({5, 6, 9});  // Rectangle{5,6,9} is automatically called
}

// Note: use it with caution.
// 1. Not very readable, even with the help of IDE. Funcion name rarely indicates
//    the type of parameter the function takes.
// 2. Function could be overloaded with differenct parameter types.

void draw_rect(Triangle t);

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

## Uniform Initialization

타입 추론에 의해 객체를 생성할 때 3가지 순서를 따른다. 첫째 `initializer_list constructor` 를 찾는다. 둘째 적절한 `constructor` 를 찾는다. 셋째 `aggreate initialization` 을 한다.

```cpp
// C++ 03
class Dog {     // Aggregate class or struct
  public:
    int age;
    string name;
};
Dog d1 = {5, "Henry"};   // Aggregate Initialization

// C++ 11 extended the scope of curly brace initialization
class Dog {
  public:
    Dog(int age, string name) {...};
};
Dog d1 = {5, "Henry"}; 

/* Uniform Initialization Search Order:
 * 1. Initializer_list constructor
 * 2. Regular constructor that takes the appropriate parameters.
 * 3. Aggregate initializer.
 */

Dog d1{3};

class Dog {
  public:
  int age;                                // 3rd choice

  Dog(int a) {                            // 2nd choice
    age = a;
  }

  Dog(const initializer_list<int>& vec) { // 1st choice
    age = *(vec.begin());      
   }
};
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

struct Node {
  char id; 
  int value;
  Node(char i, int v) : id(i), value(v) {}
  Node() : id(0), value('z') {}
};

int main() {
  tuple<int, string, char> t(32, "Penny wise", 'a');
  tuple<int, string, char> t = {32, "Penny wise", 'a'};  // Wont compile, constructor is explicit

  cout << get<0>(t) << endl;
  cout << get<1>(t) << endl;
  cout << get<2>(t) << endl;

  get<1>(t) = "Pound foolish";
  cout << get<1>(t) << endl;

  string& s = get<1>(t);
  s = "Patience is virtue"; 
  cout << get<1>(t) << endl;   
  //get<3>(t);  // Won't compile, t only has 3 fields
  // get<1>(t) is similar to t[1] for vector

  int i = 1;
  //get<i>(t); // Won't compile, i must be a compile time constant


  tuple<int, string, char> t2;  // default construction 
  t2 = tuple<int, string, char>(12, "Curiosity kills the cat", 'd'); 
  t2 = make_tuple(12, "Curiosity kills the cat", 'd'); 

  if (t > t2) {  // Lexicographical comparison
    cout << "t is larger than t2" << endl;
  }

  t = t2;  // member by member copying

// Tuple can store references !!  STL containers such as vectors cannot.  Pair can.
  string st = "In for a penny";
  tuple<string&> t3(st);  
  //auto t3 = make_tuple(ref(st));  // Do the same thing
  get<0>(t3) = "In for a pound";  // st has "In for a pound"
  cout << st << endl;
  t2 = make_tuple(12, "Curiosity kills the cat", 'd'); 
  int x;
  string y;
  char z;
  std::make_tuple(std::ref(x), std::ref(y), std::ref(z)) = t2;  // assign t2 to x, y, z
  std::tie(x,y,z) = t2;  // same thing
  std::tie(x, std::ignore, z) = t2;  // get<1>(t2) is ignored

// Other features
  auto t4 = std::tuple_cat( t2, t3 );  // t4 is tuple<int, string, char, string>
  cout << get<3>(t4) << endl;  // "In for a pound" 

  // type traits
  cout << std::tuple_size<decltype(t4)>::value << endl;  // Output: 4
  std::tuple_element<1, decltype(t4)>::type dd; // dd is a string
   
}

// tuple vs struct

tuple<string, int> getNameAge() { 
  return make_tuple("Bob", 34);
}

int main() {
  struct Person { string name; int age; } p;
  tuple<string, int> t;

  cout << p.name << " " << p.age << endl;
  cout << get<0>(t) << " " << get<1>(t) << endl;

  // As a one-time only structure to transfer a group of data
  string name;
  int age;
  tie(name, age) = getNameAge();

  // Comparison of tuples
  tuple<int, int, int> time1, time2; // hours, minutes, seconds
  if (time1 > time2) 
    cout << " time1 is a later time";

  // Multi index map
  map<tuple<int,int,int>, string> timemap;
  timemap.insert(make_pair(make_tuple(12, 2, 3), "Game start"));
	cout << timemap[make_tuple(2,3,4)]; 
  unordered_map<tuple<int,int,int>, string> timemap;

   // Little trick
  int a, b, c;
  tie(b, c, a) = make_tuple(a, b, c);

}
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

cout << [](int x, int y){return x+y}(3,4) << endl;  // Output: 7
auto f = [](int x, int y) { return x+y; };
cout << f(3,4) << endl;   // Output: 7

template<typename func>
void filter(func f, vector<int> arr) {
  for (auto i: arr) {
    if (f(i))
      cout << i << " ";
  }
}

int main() {
  vector<int> v = {1, 2, 3, 4, 5, 6 };

  filter([](int x) {return (x>3);},  v);    // Output: 4 5 6
  ...
  filter([](int x) {return (x>2 && x<5);},  v); // Output: 3 4


  int y = 4;  
  filter([&](int x) {return (x>y);},  v);    // Output: 5 6
  //Note: [&] tells compiler that we want variable capture
}

// Lambda function works almost like a language extention
template
for_nth_item  
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
  일어나지 않는다. rvalue 가 return 된다.
  
## Value Categories

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

int&& c;       // c is an rvalue reference

void printInt(int& i) { 
  cout << "lvalue reference: " << i << endl; 
}
void printInt(int&& i) { 
  cout << "rvalue reference: " << i << endl; 
} 

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

vector<int> foo() { 
  ...; 
  return myvector; 
}

void goo(vector<int>& arg);   
// Pass by reference if you use arg to carry
// data back from goo()
```

## Perfect Forwarding

어떠한 함수가  `rvalue` 를 `rvalue` 로 `lvalue` 를 `lvalue` 로 전달하는 것을 `perfect forwarding` 이라고 한다. 다음의 예에서 `relay` 는 `perfect forwarding` 이 되고 있지 않다.

```cpp
void foo(boVector arg);
// boVector has both move constructor and copy constructor

template<typename T>
void relay(T arg) {
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
template<typename T>
void relay(T&& arg) {
  foo(std::forward<T>(arg));
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
template<classs T>
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
  foo(std::forward<T>( arg ));
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

- 값 복사 없이 메모리가 이동하는 형태를 move semantic 이라고 한다. move
  semantic 이 가능한 생성자를 move constructor 라고 한다.

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

- vector 와 다르게 크기가 고정된 배열이다.
- int[] 와 다른점은 무엇일까?
  - STL algorithm 함수 들에 적용이 가능하다. (ex. begin, end)
  - 값 복사, 사전순 비교 등이 쉽다. 
  - assertive 하게 동작 (out-of-index 일때 바로 throw)
  
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

```cpp
#include <regex>
using namespace std;

int main() {
  string str;
  while (true) {
    cin >> str;
	  //regex e("abc.", regex_constants::icase);   // .   Any character except newline
	  //regex e("abc?");               // ?       Zero or 1 preceding character
	  //regex e("abc*");               // *       Zero or more preceding character
	  //regex e("abc+");               // +       One of more preceding character
	  //regex e("ab[cd]*");            // [...]   Any character inside the square brackets
	  //regex e("ab[^cd]*");           // [...]   Any character not inside the square brackets
	  //regex e("ab[cd]{3,5}");
	  //regex e("abc|de[\]fg]");         // |       Or
	  //regex  e("(abc)de+\\1");       // \1      First group
	  //regex  e("(ab)c(de+)\\2\\1");
	  //regex e("[[:w:]]+@[[:w:]]+\.com"); // [[:w:]] word character: digit, number, or underscore

	  //regex e("abc.$");                 // $   End of the string
	  regex e("^abc.+", regex_constants::grep);                 // ^   begin of the string
	  

	  //bool match = regex_match(str, e);
	  bool match = regex_search(str, e);

	  cout << (match? "Matched" : "Not matched") << endl << endl;
  }
}

/*
Regular Expression Grammars:

  ECMAScript
  basic
  extended
  awk
  grep 
  egrep
 */

/***************  Deal with subexpression *****************/

/* 
  std::match_results<>  Store the detailed matches
  smatch                Detailed match in string

  smatch m;
  m[0].str()   The entire match (same with m.str(), m.str(0))
  m[1].str()   The substring that matches the first group  (same with m.str(1))
  m[2].str()   The substring that matches the second group
  m.prefix()   Everything before the first matched character
  m.suffix()   Everything after the last matched character
*/

int main() {
   string str;

   while (true) {
      cin >> str;
	  smatch m;        // typedef std::match_results<string>

	  regex e("([[:w:]]+)@([[:w:]]+)\.com");  

	  bool found = regex_search(str, m, e);

      cout << "m.size() " << m.size() << endl;
	  for (int n = 0; n< m.size(); n++) {
		   cout << "m[" << n << "]: str()=" << m[n].str() << endl;
		   cout << "m[" << n << "]: str()=" << m.str(n) << endl;
			cout << "m[" << n << "]: str()=" << *(m.begin()+n) << endl;
	  }
	  cout << "m.prefix().str(): " << m.prefix().str() << endl;
	  cout << "m.suffix().str(): " << m.suffix().str() << endl;
   }
}

/**************** Regex Iterator ******************/
int main() {
	cout << "Hi" << endl;

   string str;

   while (true) {
      cin >> str;

	  regex e("([[:w:]]+)@([[:w:]]+)\.com"); 
	  
	  sregex_iterator pos(str.cbegin(), str.cend(), e);
	  sregex_iterator end;  // Default constructor defines past-the-end iterator
	  for (; pos!=end; pos++) {
		  cout << "Matched:  " << pos->str(0) << endl;
		  cout << "user name: " << pos->str(1) << endl;
		  cout << "Domain: " << pos->str(2) << endl;
		  cout << endl;
	  }
	  cout << "=============================\n\n";
   }
}

/**************** Regex Token Iterator ******************/
int main() {
	cout << "Hi" << endl;

	//string str = "Apple; Orange, {Cherry}; Blueberry";
	string str = "boq@yahoo.com, boqian@gmail.com; bo@hotmail.com";

	//regex e("[[:punct:]]+");  // Printable character that is not space, digit, or letter.
	//regex e("[ [:punct:]]+"); 
	regex e("([[:w:]]+)@([[:w:]]+)\.com");
	  
	sregex_token_iterator pos(str.cbegin(), str.cend(), e, 0);
	sregex_token_iterator end;  // Default constructor defines past-the-end iterator
	for (; pos!=end; pos++) {
		cout << "Matched:  " << *pos << endl;
	}
	cout << "=============================\n\n";
		
	
	cin >> str;
}

/**************** regex_replace ******************/
int main() {
	cout << "Hi" << endl;

	string str = "boq@yahoo.com, boqian@gmail.com; bo@hotmail.com";

	regex e("([[:w:]]+)@([[:w:]]+)\.com");
	regex e("([[:w:]]+)@([[:w:]]+)\.com", regex_constants::grep|regex_constants::icase );
	  
	//cout << regex_replace(str, e, "$1 is on $2");
   cout << regex_replace(str, e, "$1 is on $2", regex_constants::format_no_copy|regex_constants::format_first_only);
	cout << regex_replace(str, e, "$1 is on $2");
		
	std::cin >> str;
}
```

- [a.cpp](library/regex/a.cpp)

## random

```cpp
std::mt19936 eng; // Mersenne Twister
std::uniform_int_distribution<int> U(-100, 100);
for (int i = 0; i < n; ++i)
  cout << U(eng) << std;

int main ()
{
  std::default_random_engine eng;
	cout << "Min: " << eng.min() << endl; 
	cout << "Max: " << eng.max() << endl;

	cout << eng() << endl;  // Generate one random value
	cout << eng() << endl;  // Generate second random value

	std::stringstream state;
	state << eng;  // Save the state

	cout << eng() << endl;  // Generate one random value
	cout << eng() << endl;  // Generate second random value

	state >> eng;  // Restore the state
	cout << eng() << endl;  // Generate one random value
	cout << eng() << endl;  // Generate second random value
}

/* More examples */
void printRandom(std::default_random_engine e) {
	for (int i=0; i<10; i++) 
		cout << e() << " ";
	cout << endl;
}

template <typename T>
void printArray(T arr) {
	for (auto v:arr) {
		cout << v << " ";
	}
	cout << endl;
}

int main ()
{
  std::default_random_engine eng;
	printRandom(eng);

	std::default_random_engine eng2;
	printRandom(eng2);

	unsigned seed = std::chrono::steady_clock::now().time_since_epoch().count();
	std::default_random_engine e3(seed);
	printRandom(e3);

	eng.seed();  // reset engine to initial state
	eng.seed(109); // set engine to a state according to seed 109

	eng2.seed(109);
	if (eng == eng2)   // will return true
		cout << "eng and eng2 have the same state" << endl;

	cout << "\n\n Shuffling:" << endl;
	int arr[] = {1,2,3,4,5,6,7,8,9};
	vector<int> d(arr, arr+9);
	printArray(d);

	vector<int> d =  {1,2,3,4,5,6,7,8,9};
	std::shuffle(d.begin(), d.end(), std::default_random_engine());
	printArray(d);
	std::shuffle(d.begin(), d.end(), std::default_random_engine());  // same order
	printArray(d);
	
	std::shuffle(d.begin(), d.end(), eng);
	printArray(d);
	std::shuffle(d.begin(), d.end(), eng);  // different order
	printArray(d);
}

/* Other random engines */

/* Distribution */

int main ()  {
	// engine only provides a source of randomness
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine e(seed);
   // How to get a random number between 0 and 5?
   //  e()%6  
	//    -- Bad quality of randomness
	//    -- Can only provide uniform distribution

	std::uniform_int_distribution<int> distr(0,5);  // range: [0,5]  -- both 1 and 5 are included
													// default param: [0, INT_MAX]
	cout << " int_distribution: " << endl; 
    for (int i=0; i<30; i++) {
        cout << distr(e) << " ";
    }

	cout << "\n\n real_distribution: " << endl;

	std::uniform_real_distribution<double> distrReal(0,5);  // half open: [1, 5)  -- 1 is included, 5 is not.
														// default param: [0, 1)
    for (int i=0; i<30; i++) {
        cout << distrReal(e) << " ";
    }

	cout << " poisson_distribution: " << endl; 
	std::poisson_distribution<int> distrP(1.0);  //  mean (double) 
    for (int i=0; i<30; i++) {
        cout << distrP(e) << " ";
    }
	cout << endl;	

	cout << " normal_distribution: " << endl; 
	std::normal_distribution<double> distrN(10.0, 3.0);  // mean and standard deviation
	vector<int> v(20);
    for (int i=0; i<800; i++) {
        int num = distrN(e); // convert double to int
		if (num >= 0 && num < 20)
			v[num]++;   // E.g., v[10] records number of times 10 appeared
    }
	for (int i=0; i<20; i++) {
		cout << i << ": " << std::string(v[i], '*') << endl;
	}
	cout << endl;

	// Stop using rand()%n; 
}
/* Other distributions */
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

## Variadic Template

가변인자 템플릿이 가능하다.

```cpp
template<typename... arg>
class BoTemplate;

BoTemplate<float> t1;
BoTemplate<int, long, double, float> t2;
BoTemplate<int, std::vector<double>> t3;

BoTemplate<> t4;

// Combination of variadic and non-variadic argument
template<typename T, typename... arg>
class BoTemplate;

BoTemplate<> t4;  // Error
BoTemplate<int, long, double, float> t2;  // OK

// Define a default template argument
template<typename T = int, typename... arg>
class BoTemplate;
```

## Template Alias

템플릿 별칭이 가능하다.

```cpp
  template<class T> class Dog { /* ... */ };
  template<class T>
    using DogVec = std::vector<T, Dog<T>>;

  DogVec<int> v;  // Same as: std::vector<int, Dog<int>>
```

## decltype

인스턴스를 인자로 받아 타입을 리턴할 때 사용한다.

```cpp
  const int& foo();      // Declare a function foo()
  decltype(foo())  x1;  //  type is const int&

  struct S { double x; };
  decltype(S::x)   x2;  //  x2 is double

  auto s = make_shared<S>();
  decltype(s->x)   x3;  //  x3 is double

  int i;
  decltype(i)      x4;  //  x4 is int  

  float f;              
  decltype(i + f)  x5;   // x5 is float

  // decltype turns out to be very useful for template generic programming
  template<type X, type Y>
  void foo(X x, Y y) {
     ...
     decltype(x+y) z;
     ...
  }

  // How about return type needs to use decltype?
  template<type X, type Y>
  decltype(x+y) goo(X x, Y y) {      // Error: x & y are undefined 
     return  x + y;
  }

  // Combining auto and decltype to implement templates with trailing return type
  template<type X, type Y>
  auto goo(X x, Y y) -> decltype(x+y) {
     return  x + y;
  }
```

## chrono

```cpp
/*
	-- A precision-neutral library for time and date
	
 * clocks:
 *
 * std::chrono::system_clock:  current time according to the system (it is not steady)
 * std::chrono::steady_clock:  goes at a uniform rate (it can't be adjusted)
 * std::chrono::high_resolution_clock: provides smallest possible tick period. 
 *                   (might be a typedef of steady_clock or system_clock)
 *
 * clock period is represented with std:ratio<>
 */

std::ratio<1,10>  r; // 
cout << r.num << "/" << r.den << endl;

cout << chrono::system_clock::period::num << "/" << chrono::system_clock::period::den << endl;
cout << chrono::steady_clock::period::num << "/" << chrono::steady_clock::period::den << endl;
cout << chrono::high_resolution_clock::period::num << "/" << chrono::high_resolution_clock::period::den << endl;

/*
 *
 * std:chrono::duration<>:  represents time duration
 *    duration<int, ratio<1,1>> --  number of seconds stored in a int  (this is the default)
 *    duration<double, ration<60,1>> -- number of minutes (60 seconds) stored in a double
 *    convenince duration typedefs in the library:
 *    nanoseconds, microseconds, milliseconds, seconds, minutes, hours
 * system_clock::duration  -- duration<T, system_clock::period>
 *                                 T is a signed arithmetic type, could be int or long or others
 */
chrono::microseconds mi(2745);
chrono::nanoseconds na = mi;
chrono::milliseconds mill = chrono::duration_cast<chrono::milliseconds>(mi);  // when information loss could happen, convert explicitly
														  // Truncation instead of rounding
	mi = mill + mi;  // 2000 + 2745 = 4745
	mill = chrono::duration_cast<chrono::milliseconds>(mill + mi);  // 6
	cout << na.count() << std::endl;
	cout << mill.count() << std::endl;
	cout << mi.count() << std::endl;

   cout << "min: " << chrono::system_clock::duration::min().count() << "\n";
   cout << "max: " << chrono::system_clock::duration::max().count() << "\n";

 /* std::chrono::time_point<>: represents a point of time
 *       -- Length of time elapsed since a spacific time in history: 
 *          00:00 January 1, 1970 (Corordinated Universal Time - UTC)  -- epoch of a clock
 * time_point<system_clock, milliseconds>:  according to system_clock, the elapsed time since epoch in milliseconds
 *
 * typdefs
  system_clock::time_point  -- time_point<system_clock, system_clock::duration>
  steady_clock::time_point  -- time_point<steady_clock, steady_clock::duration>
 */
	// Use system_clock
	chrono::system_clock::time_point tp = chrono::system_clock::now();
	cout << tp.time_since_epoch().count() << endl;  
	tp = tp + seconds(2);  // no need for cast because tp is very high resolution
	cout << tp.time_since_epoch().count() << endl;

	// Calculate time interval
	chrono::steady_clock::time_point start = chrono::steady_clock::now();
	cout << "I am bored" << endl;
	chrono::steady_clock::time_point end = chrono::steady_clock::now();
	chrono::steady_clock::duration d = end - start;
	if (d == chrono::steady_clock::duration::zero())
		cout << "no time elapsed" << endl;
	cout << duration_cast<microseconds>(d).count() << endl;
   // Using system_clock may result in incorrect value
```