- [vector](#vector)
- [deque](#deque)
- [list](#list)
- [set, multiset](#set-multiset)
- [map, multimap](#map-multimap)
- [unordered container](#unordered-container)
- [Associative Array](#associative-array)
- [array](#array)
- [Container Adaptor](#container-adaptor)
- [Iterators](#iterators)
- [Iterator Adaptor (Predefined Iterator)](#iterator-adaptor-predefined-iterator)
- [Functors (Function Objects)](#functors-function-objects)
- [Object Slicing](#object-slicing)
- [Sorting](#sorting)
- [Algorithms](#algorithms)
- [Algorithms For Sorted Data](#algorithms-for-sorted-data)
- [Algorithms For Non-Modifying](#algorithms-for-non-modifying)
- [Algorithms For Modifying](#algorithms-for-modifying)
- [Algorithms for Equality, Equivalance](#algorithms-for-equality-equivalance)
- [Algorithm vs Member Function](#algorithm-vs-member-function)
- [Modifying In A Container](#modifying-in-a-container)
- [Removing In A Container](#removing-in-a-container)
- [shared_ptr](#shared_ptr)
- [weak_ptr](#weak_ptr)
- [unique_ptr](#unique_ptr)
- [input output stream](#input-output-stream)
- [random_device](#random_device)

----

## vector

```cpp
vector<int> vec;   // vec.size() == 0
vec.push_back(4);
vec.push_back(1);
vec.push_back(8);  // vec: {4, 1, 8};    vec.size() == 3

// Vector specific operations:
cout << vec[2];     // 8  (no range check)
cout << vec.at(2);  // 8  (throw range_error exception of out of range)

for (int i; i < vec.size(); i++) 
   cout << vec[i] << " ";

for (list<int>::iterator itr = vec.begin(); itr!= vec.end(); ++itr)
   cout << *itr << " ";  

for (it : vec)    // C++ 11
   cout << it << " ";

// Vector is a dynamically allocated contiguous array in memory
int* p = &vec[0];   p[2] = 6;

// Common member functions of all containers.
// vec: {4, 1, 8}
if (vec.empty()) { 
   cout << "Not possible.\n"; 
}

cout << vec.size();   // 3
vector<int> vec2(vec);  // Copy constructor, vec2: {4, 1, 8}
vec.clear();    // Remove all items in vec;   vec.size() == 0
vec2.swap(vec);   // vec2 becomes empty, and vec has 3 items.

// Notes: No penalty of abstraction, very efficient.

vector<int> a;
vector<int> b;
// Add range
a.insert(a.end(), b.begin(), b.end();
// Compare contens of two vector
equal(a.begin(), a.end(), b.begin());

/* Properties of Vector:
 * 1. fast insert/remove at the end: O(1)
 * 2. slow insert/remove at the begining or in the middle: O(n)
 * 3. slow search: O(n)
 */
```

## deque

```cpp

deque<int> deq = { 4, 6, 7 };
deq.push_front(2);  // deq: {2, 4, 6, 7}
deq.push_back(3);   // deq: {2, 4, 6, 7, 3}

// Deque has similar interface with vector
cout << deq[1];  // 4

/* Properties:
 * 1. fast insert/remove at the begining and the end;
 * 2. slow insert/remove in the middle: O(n)
 * 3. slow search: O(n)
 */
```

## list

```cpp
list<int> mylist = {5, 2, 9 }; 
mylist.push_back(6);  // mylist: { 5, 2, 9, 6}
mylist.push_front(4); // mylist: { 4, 5, 2, 9, 6}
   
list<int>::iterator itr = find(mylist.begin(), mylist.end(), 2); // itr -> 2
mylist.insert(itr, 8);   // mylist: {4, 5, 8, 2, 9, 6}  
                         // O(1), faster than vector/deque
itr++;                   // itr -> 9
mylist.erase(itr);       // mylist: {4, 8, 5, 2, 6}   O(1)

/* Properties:
 * 1. fast insert/remove at any place: O(1)
 * 2. slow search: O(n)
 * 3. no random access, no [] operator.
 */

mylist1.splice(itr, mylist2, itr_a, itr_b );   // O(1)
```

## set, multiset

```cpp
  set<int> myset;
  myset.insert(3);    // myset: {3}
  myset.insert(1);    // myset: {1, 3}
  myset.insert(7);    // myset: {1, 3, 7},  O(log(n))

  set<int>::iterator it;
  it = myset.find(7);  // O(log(n)), it points to 7
                  // Sequence containers don't even have find() member function
  pair<set<int>::iterator, bool> ret;
  ret = myset.insert(3);  // no new element inserted
  if (ret.second==false) 
     it=ret.first;       // "it" now points to element 3

  myset.insert(it, 9);  // myset:  {1, 3, 7, 9}   O(log(n)) => O(1)
                         // it points to 3
  myset.erase(it);         // myset:  {1, 7, 9}

  myset.erase(7);   // myset:  {1, 9}
  // Note: none of the sequence containers provide this kind of erase.

// multiset is a set that allows duplicated items
multiset<int> myset;

// set/multiset: value of the elements cannot be modified
*it = 10;  // *it is read-only

/* Properties:
* 1. Fast search: O(log(n))
* 2. Traversing is slow (compared to vector & deque)
* 3. No random access, no [] operator.
*/
```

## map, multimap

```cpp
map<char, int> mymap;
mymap.insert ( pair<char,int>('a',100) );
mymap.insert ( make_pair('z',200) );

map<char,int>::iterator it = mymap.begin();
mymap.insert(it, pair<char,int>('b',300));  // "it" is a hint

it = mymap.find('z');  // O(log(n))

// map with descending order
map<int, int, greater<int>> mymap;
mymap.insert({10, 15});
mymap.insert({20, 25});
for (auto& pr : mymap) {
  printf("[%d, %d] ", pr.first, pr.second);
}
printf("\n");
// [20, 25] [10, 15]

// showing contents:
for ( it=mymap.begin() ; it != mymap.end(); it++ )
  cout << (*it).first << " => " << (*it).second << endl;

// multimap is a map that allows duplicated keys
multimap<char,int> mymap;

// map/multimap: 
//  -- keys cannot be modified
//     type of *it:   pair<const char, int>
     (*it).first = 'd';  // Error

// Associative Containers: set, multiset, map, multimap
//
// What does "Associative" mean?
```

## unordered container

```cpp
  unordered_set<string> myset = { "red","green","blue" };
  unordered_set<string>::const_iterator itr = myset.find ("green"); // O(1)
  if (itr != myset.end())   // Important check 
     cout << *itr << endl;
  myset.insert("yellow");  // O(1)

  vector<string> vec = {"purple", "pink"};
  myset.insert(vec.begin(), vec.end());

// Hash table specific APIs:
  cout << "load_factor = " << myset.load_factor() << endl;
  string x = "red";
  cout << x << " is in bucket #" << myset.bucket(x) << endl;
  cout << "Total bucket #" << myset.bucket_count() << endl;

/* Properties of Unordered Containers:
 * 1. Fastest search/insert at any place: O(1)
 *     Associative Container takes O(log(n))
 *     vector, deque takes O(n)
 *     list takes O(1) to insert, O(n) to search
 * 2. Unorderd set/multiset: element value cannot be changed.
 *    Unorderd map/multimap: element key cannot be changed.
 */  
```

## Associative Array

```cpp
unordered_map<char, string> day = {{'S',"Sunday"}, {'M',"Monday"}};

cout << day['S'] << endl;    // No range check
cout << day.at('S') << endl; // Has range check

vector<int> vec = {1, 2, 3};
vec[5] = 5;   // Compile Error

day['W'] = "Wednesday";  // Inserting {'W', "Wednesday}
day.insert(make_pair('F', "Friday"));  // Inserting {'F', "Friday"}

day.insert(make_pair('M', "MONDAY"));  // Fail to modify, it's an unordered_map
day['M'] = "MONDAY";                   // Succeed to modify

void foo(const unordered_map<char, string>& m) {
   //m['S'] = "SUNDAY";
   //cout << m['S'] << endl;
   auto itr = m.find('S');
   if (itr != m.end())
      cout << *itr << endl;
}
foo(day);

//   cout << m['S'] << endl;
//   auto itr = m.find('S');
//   if (itr != m.end() )
//      cout << itr->second << endl;

//Notes about Associative Array: 
//1. Search time: unordered_map, O(1); map, O(log(n));
//2. Unordered_map may degrade to O(n);
//3. Can't use multimap and unordered_multimap, they don't have [] operator.
```

## array

```cpp
int a[3] = {3, 4, 5};
array<int, 3> a = {3, 4, 5};
a.begin();
a.end();
a.size();
a.swap();
array<int, 4> b = {3, 4, 5};
```

## Container Adaptor

```cpp
/*
 * Container Adaptor
 *  - Provide a restricted interface to meet special needs
 *  - Implemented with fundamental container classes
 *
 *  1. stack:  LIFO, push(), pop(), top()
 *
 *  2. queue:  FIFO, push(), pop(), front(), back() 
 *
 *  3. priority queue: first item always has the greatest priority
 *                   push(), pop(), top()
 */
```

## Iterators

```cpp
// 1. Random Access Iterator:  vector, deque, array
vector<int> itr;
itr = itr + 5;  // advance itr by 5
itr = itr - 4;  
if (itr2 > itr1) ...
++itr;   // faster than itr++
--itr;

// 2. Bidirectional Iterator: list, set/multiset, map/multimap
list<int> itr;
++itr;
--itr;

// 3. Forward Iterator: forward_list
forward_list<int> itr;
++itr;

// Unordered containers provide "at least" forward iterators.

// 4. Input Iterator: read and process values while iterating forward.
int x = *itr;

// 5. Output Iterator: output values while iterating forward.
*itr = 100;

// Every container has a iterator and a const_iterator
set<int>::iterator itr;
set<int>::const_iterator citr;  // Read_only access to container elements

set<int> myset = {2,4,5,1,9};
for (citr = myset.begin(); citr != myset.end(); ++citr) {
   cout << *citr << endl;
   //*citr = 3;
}
for_each(myset.cbegin(), myset.cend(), MyFunction);  // Only in C++ 11

// Iterator Functions:
advance(itr, 5);       // Move itr forward 5 spots.   itr += 5;
distance(itr1, itr2);  // Measure the distance between itr1 and itr2

// Two ways to declare a reverse iterator
reverse_iterator<vector<int>::iterator> ritr;
vector<int>::reverse_iterator ritr;


// Traversing with reverse iterator
vector<int> vec = {4,5,6,7};
reverse_iterator<vector<int>::iterator> ritr;
for (ritr = vec.rbegin(); ritr != vec.rend(); ritr++)
   cout << *ritr << endl;   // prints: 7 6 5 4

/*
 * Reverse Iterator and Iterator 
 */   

vector<int>::iterator itr;
vector<int>::reverse_iterator ritr;

ritr = vector<int>::reverse_iterator(itr);

itr = vector<int>::iterator(ritr);  // Compile Error
itr = ritr.base();  

// C++ Standard: base() returns current iterator
// 

vector<int> vec = {1,2,3,4,5};
vector<int>::reverse_iterator ritr = find(vec.rbegin(), vec.rend(), 3);

cout << (*ritr) << endl;   // 3

vector<int>::iterator itr = ritr.base();  

cout << (*itr) << endl;   // 4 

vec = {1,2,3,4,5};
ritr = find(vec.rbegin(), vec.rend(), 3);

//Inserting
vec.insert(ritr, 9);         // vec: {1,2,3,9,4,5}
// or
vec.insert(ritr.base(), 9);  // vec: {1,2,3,9,4,5}

vec = {1,2,3,4,5};
ritr = find(vec.rbegin(), vec.rend(), 3);

// Erasing
vec.erase(ritr);    // vec: {1,2,4,5}  
// or
vec.erase(ritr.base());    // vec: {1,2,3,5}  
```

## Iterator Adaptor (Predefined Iterator)

```cpp

/* Iterator Adaptor (Predefined Iterator)
 *  - A special, more powerful iterator
 * 1. Insert iterator
 * 2. Stream iterator
 * 3. Reverse iterator
 * 4. Move iterator (C++ 11)
 */

// 1. Insert Iterator:
vector<int> vec1 = {4,5};
vector<int> vec2 = {12, 14, 16, 18};
vector<int>::iterator it = find(vec2.begin(), vec2.end(), 16);
insert_iterator<vector<int>> i_itr(vec2, it);
copy(vec1.begin(),vec1.end(),  // source
     i_itr);                   // destination
     //vec2: {12, 14, 4, 5, 16, 18}
// Other insert iterators: back_insert_iterator, front_insert_iterator

// 2. Stream Iterator:
vector<string> vec4;
copy(istream_iterator<string>(cin), istream_iterator<string>(), 
            back_inserter(vec4));

copy(vec4.begin(), vec4.end(), ostream_iterator<string>(cout, " "));

// Make it terse:
copy(istream_iterator<string>(cin), istream_iterator<string>(), 
            ostream_iterator<string>(cout, " "));

// 3. Reverse Iterator:
vector<int> vec = {4,5,6,7};
reverse_iterator<vector<int>::iterator> ritr;
for (ritr = vec.rbegin(); ritr != vec.rend(); ritr++)
   cout << *ritr << endl;   // prints: 7 6 5 4
```

## Functors (Function Objects)

```cpp
class X {
   public:
   void operator()(string str) { 
      cout << "Calling functor X with parameter " << str<< endl;
   }  
};

int main()
{
   X foo;
   foo("Hi");    // Calling functor X with parameter Hi
}
/*
 * Benefits of functor:
 * 1. Smart function: capabilities beyond operator()
 * 	It can remember state.
 * 2. It can have its own type.
 */

// 
//   operator string () const { return "X"; }

/*
 * Parameterized Function
 */
class X {
   public:
   X(int i) {}
   void operator()(string s) { 
      cout << "Calling functor X with parameter " << s << endl;
   }
};

int main()
{
   X(8)("Hi");
}

void add2(int i) {
   cout << i+2 << endl;
}

template<int val>
void addVal(int i) {
   cout << val+i << endl;
}

class AddValue {
   int val;
   public:
   AddValue(int j) : val(j) { }
   void operator()(int i) {
      cout << i+val << endl;
   }
};

int main()
{
   vector<int> vec = { 2, 3, 4, 5};   
   //for_each(vec.begin(), vec.end(), add2); // {4, 5, 6, 7}
   int x = 2;
   //for_each(vec.begin(), vec.end(), addVal<x>); // {4, 5, 6, 7}
   for_each(vec.begin(), vec.end(), AddValue(x)); // {4, 5, 6, 7}
}

/*
 * Build-in Functors
 */
less greater  greater_equal  less_equal  not_equal_to
logical_and  logical_not  logical_or
multiplies minus  plus  divide  modulus  negate

int x = multiplies<int>()(3,4);  //  x = 3 * 4 

if (not_equal_to<int>()(x, 10))   // if (x != 10)
   cout << x << endl;

/*
 * Parameter Binding
 */
set<int> myset = { 2, 3, 4, 5};   
vector<int> vec;

int x = multiplies<int>()(3,4);  //  x = 3 * 4 

// Multiply myset's elements by 10 and save in vec:
transform(myset.begin(), myset.end(),    // source
	      back_inserter(vec),              // destination
			bind(multiplies<int>(), placeholders::_1, 10));  // functor
    // First parameter of multiplies<int>() is substituted with myset's element
    // vec: {20, 30, 40, 50}

void addVal(int i, int val) {
   cout << i+val << endl;
}
for_each(vec.begin(), vec.end(), bind(addVal, placeholders::_1, 2));

// C++ 03: bind1st, bind2nd

// Convert a regular function to a functor
double Pow(double x, double y) {
	return pow(x, y);
}

int main()
{
  set<int> myset = {3, 1, 25, 7, 12};
  deque<int> d;
  auto f = function<double (double,double)>(Pow);   //C++ 11
  transform(myset.begin(), myset.end(),     // source
		      back_inserter(d),              // destination
				bind(f, placeholders::_1, 2));  // functor
            //  d: {1, 9, 49, 144, 625}
}
// C++ 03 uses ptr_fun 

set<int> myset = {3, 1, 25, 7, 12};
// when (x > 20) || (x < 5),  copy from myset to d
deque<int> d;

bool needCopy(int x){
   return (x>20)||(x<5);
}


transform(myset.begin(), myset.end(),     // source
          back_inserter(d),               // destination
          needCopy
          );

// C++ 11 lambda function:
transform(myset.begin(), myset.end(),     // source
          back_inserter(d),              // destination
          [](int x){return (x>20)||(x<5);}
          );

/*
          bind(logical_or<bool>, 
              bind(greater<int>(), placeholders::_1, 20),
              bind(less<int>(), placeholders::_1, 5))

// C++ 11 lambda function:
transform(myset.begin(), myset.end(),     // source
          back_inserter(d),              // destination
          [](int x){return (x>20)||(x<5);}
          );

bool needCopy(int x){
   return (x>20)||(x<5);
}
*/

/*
 * Why do we need functor in STL?
 *
 */

set<int> myset = {3, 1, 25, 7, 12}; // myset: {1, 3, 7, 12, 25}
// same as:
set<int, less<int> > myset = {3, 1, 25, 7, 12};

bool lsb_less(int x, int y) {
      return (x%10)<(y%10);
}

class Lsb_less {
   public:
   bool operator()(int x, int y) {
      return (x%10)<(y%10);
   }
};
int main()
{
  set<int, Lsb_less> myset = {3, 1, 25, 7, 12};  // myset: {1,12,3,25,7}
  ...
}

/*
 * Predicate
 *
 * A functor or function that:
 * 1. Returns a boolean
 * 2. Does not modify data
 */

class NeedCopy {
   bool operator()(int x){   
      return (x>20)||(x<5);  
   }
};

transform(myset.begin(), myset.end(),     // source
          back_inserter(d),               // destination
          NeedCopy()
          );

// Predicate is used for comparison or condition check
```

## Object Slicing

자식 클래스의 인스턴스 value를 부모 클래스의 인스턴스 value 로 형변환 할 때 자식 클래스의 정보가 날아가는 현상이다.

```cpp
#include <iostream>
using namespace std;
 
class Base
{
protected:
    int i;
public:
    Base(int a)     { i = a; }
    virtual void display()
    { cout << "I am Base class object, i = " << i << endl; }
};
 
class Derived : public Base
{
    int j;
public:
    Derived(int a, int b) : Base(a) { j = b; }
    virtual void display()
    { cout << "I am Derived class object, i = "
           << i << ", j = " << j << endl;  }
};
 
// Global method, Base class object is passed by value
void somefunc (Base obj)
{
    obj.display();
}
 
int main()
{
    Base b(33);
    Derived d(45, 54);
    somefunc(b);
    somefunc(d);  // Object Slicing, the member j of d is sliced off
    return 0;
}

// I am Base class object, i = 33
// I am Base class object, i = 45
```

pointer 혹은 reference 를 사용하여 피할 수 있다.

```cpp
// rest of code is similar to above
void somefunc (Base &obj)
{
    obj.display();
}           
// rest of code is similar to above
...
// rest of code is similar to above
void somefunc (Base *objp)
{
    objp->display();
}
 
int main()
{
    Base *bp = new Base(33) ;
    Derived *dp = new Derived(45, 54);
    somefunc(bp);
    somefunc(dp);  // No Object Slicing
    return 0;
}

// I am Base class object, i = 33
// I am Derived class object, i = 45, j = 54
```

## Sorting

```cpp
// use functor or function as sorting function
class Cmp {
 public:
  bool operator()(int a, int b) {
    return a < b;
  }
};

bool cmp(int a, int b) {
  return a < b;
}

int main(void)
{
  vector<int> vec = {4, 3, 2, 6, 4, 1};

  std::sort(vec.begin(), vec.end(), Cmp());
  std::sort(vec.begin(), vec.end(), cmp);
   
  return 0;
}

// Sorting algorithm requires random access iterators:
//    vector, deque, container array, native array

vector<int> vec = {9,1,10,2,45,3,90,4,9,5,8};

sort(vec.begin(), vec.end());  // sort with operator <
// vec:  1 2 3 4 5 8 9 9 10 45 90

bool lsb_less(int x, int y) {
      return (x%10)<(y%10);
}	
sort(vec.begin(), vec.end(), lsb_less);  // sort with lsb_less()
// vec: 10 90 1 2 3 4 45 5 8 9 9

// Sometime we don't need complete sorting.

// Problem #1: Finding top 5 students according to their test scores.
//
//  -  partial sort
vector<int> vec = {9,60,70,8,45,87,90,69,69,55,7};

partial_sort(vec.begin(), vec.begin()+5, vec.end(), greater<int>());
// vec: 90 87 70 69 69 8 9 45 60 55 7

// Overloaded:
partial_sort(vec.begin(), vec.begin()+5, vec.end());
// vec: 7 8 9 45 55 90 60 87 70 69 69

// Problem #2: Finding top 5 students according to their score, but I don't 
// care their order.
vector<int> vec = {9,60,70,8,45,87,90,69,69,55,7};

nth_element(vec.begin(), vec.begin()+5, vec.end(), greater<int>());
// vec: 69 87 70 90 69 60 55 45 9 8 7

// Problem #3: Move the students whose score is less than 10 to the front
vector<int> vec = {9,60,70,8,45,87,90,69,69,55,7};

bool lessThan10(int i) {
	return (i<10);
}
partition(vec.begin(),  vec.end(), lessThan10);
// vec: 8 7 9 90 69 60 55 45 70 87 69

// To preserve the original order within each partition:
stable_partition(vec.begin(),  vec.end(), lessThan10);
// vec: 9 8 7 60 70 45 87 90 69 69 55

// Heap Algorithms
//
// Heap:
// 1. First element is always the largest
// 2. Add/remove takes O(log(n)) time

vector<int> vec = {9,1,10,2,45,3,90,4,9,5,8};

make_heap(vec.begin(), vec.end());
// vec: 90 45 10 9 8 3 9 4 2 5 1

// Remove the largest element:
pop_heap(vec.begin(), vec.end());  // 1. Swap vec[0] with last item vec[size-1]
                                   // 2. Heapify [vec.begin(), vec.end()-1)
// vec:  45 9 10 4 8 3 9 1 2 5 90
vec.pop_back();  // Remove the last item (the largest one)
// vec:  45 9 10 4 8 3 9 1 2 5

// Add a new element:
vec.push_back(100);
push_heap(vec.begin(), vec.end());  // Heapify the last item in vec
// vec:  100 45 10 4 9 3 9 1 2 5 8

// Heap Sorting:
vector<int> vec = {9,1,10,2,45,3,90,4,9,5,8};
make_heap(vec.begin(), vec.end());

sort_heap(vec.begin(), vec.end());
// vec: 1 2 3 4 5 8 9 9 10 45 100
// Note: sort_heap can only work on a heap.
```

## Algorithms

```cpp
vector<int> vec = { 4, 2, 5, 1, 3, 9};   
vector<int>::iterator itr = min_element(vec.begin(), vec.end()); // itr -> 1

// Note 1: Algorithm always process ranges in a half-open way: [begin, end)
sort(vec.begin(), itr);  // vec: { 2, 4, 5, 1, 3, 9}

reverse(itr, vec.end());  // vec: { 2, 4, 5, 9, 3, 1}   itr => 9

// Note 2:
vector<int> vec2(3);
copy(itr, vec.end(),  // Source
     vec2.begin());   // Destination
     //vec2 needs to have at least space for 3 elements.

// Note 3:
vector<int> vec3;
copy(itr, vec.end(), back_inserter(vec3));  // Inserting instead of overwriting 
                  // back_insert_iterator      Not efficient

vec3.insert(vec3.end(), itr, vec.end());  // Efficient and safe

// Note 4: Algorithm with function
bool isOdd(int i) {
   return i%2;
}

int main() {
   vector<int> vec = {2, 4, 5, 9, 2}
   vector<int>::iterator itr = find_if(vec.begin(), vec.end(), isOdd); 
   	                             // itr -> 5
}

// Note 5: Algorithm with native C++ array
int arr[4] = {6,3,7,4};
sort(arr, arr+4);
```

## Algorithms For Sorted Data

```cpp
vector<int> vec = {8,9,9,9,45,87,90};     // 7 items

// 1. Binary Search
// Search Elements
bool found = binary_search(vec.begin(), vec.end(), 9);  // check if 9 is in vec

vector<int> s = {9, 45, 66};
bool found = includes(vec.begin(), vec.end(),     // Range #1
		                s.begin(), s.end());        // Range #2
// Return true if all elements of s is included in vec
// Both vec and s must be sorted      

// Search Position
itr = lower_bound(vec.begin(), vec.end(), 9);  // vec[1]  
// Find the first position where 9 could be inserted and still keep the sorting.

itr = upper_bound(vec.begin(), vec.end(), 9);  // vec[4] 
// Find the last position where 9 could be inserted and still keep the sorting.

pair_of_itr = equal_range(vec.begin(), vec.end(), 9); 
// Returns both first and last position

// 2. Merge
vector<int> vec = {8,9,9,10}; 
vector<int> vec2 = {7,9,10}; 
merge(vec.begin(), vec.end(),      // Input Range #1
		vec2.begin(), vec2.end(),    // input Range #2
		vec_out.begin());               // Output 
      // Both vec and vec2 should be sorted (same for the set operation)
      // Nothing is dropped, all duplicates are kept.
// vec_out: {7,8,9,9,9,10,10}

vector<int> vec = {1,2,3,4,1,2,3,4,5}  // Both part of vec are already sorted 
inplace_merge(vec.begin(), vec.begin()+4, vec.end());  
// vec: {1,1,2,2,3,3,4,4,5}  - One step of merge sort

// 3. Set operations
//    - Both vec and vec3 should be sorted 
//    - The resulted data is also sorted
vector<int> vec = {8,9,9,10}; 
vector<int> vec2 = {7,9,10}; 
vector<int> vec_out[5]; 
set_union(vec.begin(), vec.end(),      // Input Range #1
		    vec2.begin(), vec2.end(),    // input Range #2
		    vec_out.begin());               // Output 
// if X is in both vec and vec2, only one X is kept in vec_out
// vec_out: {7,8,9,9,10}

set_intersection(vec.begin(), vec.end(),      // Input Range #1
		           vec2.begin(), vec2.end(),    // input Range #2
		           vec_out.begin());               // Output 
// Only the items that are in both vec and vec2 are saved in vec_out
// vec_out: {9,10,0,0,0}

vector<int> vec = {8,9,9,10}; 
vector<int> vec2 = {7,9,10}; 
vector<int> vec_out[5]; 
set_difference(vec.begin(), vec.end(),      // Input Range #1
		         vec2.begin(), vec2.end(),    // input Range #2
		         vec_out.begin());               // Output 
// Only the items that are in vec but not in vec2 are saved in vec_out
// vec_out: {8,9,0,0,0}

set_symmetric_difference(vec.begin(), vec.end(),      // Input Range #1
		         vec2.begin(), vec2.end(),       // input Range #2
		         vec_out.begin());               // Output 
// vec_out has items from either vec or vec2, but not from both
// vec_out: {7,8,9,0,0}

/*
 *  Numeric Algorithms (in <numeric>)
 *   - Accumulate, inner product, partial sum, adjacent difference
 */

// 1. Accumulate

int x = accumulate(vec.begin(), vec.end(), 10); 
// 10 + vec[0] + vec[1] + vec[2] + ...

int x = accumulate(vec.begin(), vec.end(), 10, multiplies<int>());
// 10 * vec[0] * vec[1] * vec[2] * ...

// 2. Inner Product
//vector<int> vec = {9,60,70,8,45,87,90};     // 7 items
int x = inner_product(vec.begin(), vec.begin()+3,  // Range #1
		               vec.end()-3,                 // Range #2
				         10);                         // Init Value
// 10 + vec[0]*vec[4] + vec[2]*vec[5] + vec[3]*vec[6]
		
int x = inner_product(vec.begin(), vec.begin()+3,  // Range #1
		                vec.end()-3,                 // Range #2
				          10,                          // Init Value
				          multiplies<int>(),
				          plus<int>());
// 10 * (vec[0]+vec[4]) * (vec[2]+vec[5]) * (vec[3]+vec[6])
              
// 3. Partial Sum
partial_sum(vec.begin(), vec.end(), vec2.begin());
// vec2[0] = vec[0]
// vec2[1] = vec[0] + vec[1];
// vec2[2] = vec[0] + vec[1] + vec[2]; 
// vec2[3] = vec[0] + vec[1] + vec[2] + vec[3]; 
// ...

partial_sum(vec.begin(), vec.end(), vec2.begin(), multiplies<int>());


// 4. Adjacent Difference
adjacent_difference(vec.begin(), vec.end(), vec2.begin());
// vec2[0] = vec[0]
// vec2[1] = vec[1] - vec[0];
// vec2[2] = vec[2] - vec[1]; 
// vec2[3] = vec[3] - vec[2]; 
// ...

adjacent_difference(vec.begin(), vec.end(), vec2.begin(), plus<int>());
```

## Algorithms For Non-Modifying

```cpp
// C++ 11 Lambda Function:
num = count_if(vec.begin(), vec.end(), [](int x){return x<10;});  

bool lessThan10(int x) {
   return x<10;
}

vector<int> vec = {9,60,90,8,45,87,90,69,69,55,7};
vector<int> vec2 = {9,60,70,8,45,87};
vector<int>::iterator itr, itr2;
pair<vector<int>::iterator, vector<int>::iterator> pair_of_itr;

// C++ 03: some algorithms can be found in tr1 or boost

vector<int> vec = {9,60,90,8,45,87,90,69,69,55,7};

// 1. Counting
//     Algorithm   Data              Operation
int n = count(vec.begin()+2, vec.end()-1, 69);   // 2
int m = count_if(vec.begin(), vec.end(), [](int x){return x==69;}); // 3  
int m = count_if(vec.begin(), vec.end(), [](int x){return x<10;}); // 3  

// 2.  Min and Max
itr = max_element(vec.begin()+2, vec.end());  // 90
// It returns the first max value
itr = max_element(vec.begin(), vec.end(), 
                  [](int x, int y){ return (x%10)<(y%10);}); // 9 
														 
// Most algorithms have a simple form and a generalized form

itr = min_element(vec.begin(), vec.end());  // 7 
// Generalized form: min_element()

pair_of_itr = minmax_element(vec.begin(), vec.end(),  // {60, 69} 
		                      [](int x, int y){ return (x%10)<(y%10);}); 
// returns a pair, which contains first of min and last of max

// 3. Linear Searching (used when data is not sorted)
//    Returns the first match
itr = find(vec.begin(), vec.end(), 55);

itr = find_if(vec.begin(), vec.end(), [](int x){ return x>80; });

itr = find_if_not(vec.begin(), vec.end(), [](int x){ return x>80; });

itr = search_n(vec.begin(), vec.end(), 2, 69);  // Consecutive 2 items of 69
// Generalized form: search_n()

// Search subrange
vector<int> sub = {45, 87, 90};
itr = search( vec.begin(), vec.end(), sub.begin(), sub.end()); 
      // search first subrange 
itr = find_end( vec.begin(), vec.end(), sub.begin(), sub.end());
      // search last subrange 
// Generalized form: search(), find_end()

// Search any_of
vector<int> items  = {87, 69};
itr = find_first_of(vec.begin(), vec.end(), items.begin(), items.end()); 
      // Search any one of the item in items
itr = find_first_of(vec.begin(), vec.end(), items.begin(), items.end(),
		              [](int x, int y) { return x==y*4;}); 
      // Search any one of the item in items that satisfy: x==y*4;

// Search Adjacent
itr = adjacent_find(vec.begin(), vec.end());  // find two adjacent items that 
                                              // are same
itr = adjacent_find(vec.begin(), vec.end(), [](int x, int y){ return x==y*4;}); 
	     // find two adjacent items that satisfy: x==y*4;

// 4. Comparing Ranges
if (equal(vec.begin(), vec.end(), vec2.begin())) {
  cout << "vec and vec2 are same.\n";	
}

if (is_permutation(vec.begin(), vec.end(), vec2.begin())) {
	cout << "vec and vec2 have same items, but in differenct order.\n";	
}

pair_of_itr = mismatch(vec.begin(), vec.end(), vec2.begin());  
// find first difference
// pair_of_itr.first is an iterator of vec 
// pair_of_itr.second is an iterator of vec2

//Lexicographical Comparison: one-by-one comparison with "less than"
lexicographical_compare(vec.begin(), vec.end(), vec2.begin(), vec2.end());
// {1,2,3,5} < {1,2,4,5}
// {1,2}     < {1,2,3}

// Generalized forms: 
//   equal(), is_permutation(), mismatch(), lexicographical_compare()

// 5. Check Attributes
is_sorted(vec.begin(), vec.end());  // Check if vec is sorted

itr = is_sorted_until(vec.begin(), vec.end()); 
// itr points to first place to where elements are no longer sorted
// Generalized forms: is_sorted(), is_sorted_until()

is_partitioned(vec.begin(), vec.end(), [](int x){return x>80;} );
			// Check if vec is partitioned according to the condition of (x>80)

is_heap(vec.begin(), vec.end());  // Check if vec is a heap
itr = is_heap_until(vec.begin(), vec.end());  // find the first place where it 
                                              // is no longer a heap
// Generalized forms: is_heap(), is_heap_until()

// All, any, none
all_of(vec.begin(), vec.end(), [](int x) {return x>80} );  
// If all of vec is bigger than 80 

any_of(vec.begin(), vec.end(), [](int x) {return x>80} );  
// If any of vec is bigger than 80 

none_of(vec.begin(), vec.end(), [](int x) {return x>80} );  
// If none of vec is bigger than 80 
```

## Algorithms For Modifying

```cpp
/*
 * Algorithm Walkthrough: 
 *   Value-changing Algorithm - Changes the element values
 *   copy, move, transform, swap, fill, replace, remove
 */

vector<int> vec = {9,60,70,8,45,87,90};     // 7 items
vector<int> vec2 = {0,0,0,0,0,0,0,0,0,0,0}; // 11 items
vector<int>::iterator itr, itr2;
pair<vector<int>::iterator, vector<int>::iterator> pair_of_itr;

vector<int> vec = {9,60,70,8,45,87,90};     // 7 items
vector<int> vec2 = {0,0,0,0,0,0,0,0,0,0,0}; // 11 items

// 1. Copy
copy(vec.begin(), vec.end(), // Source
	  vec2.begin());          // Destination

copy_if(vec.begin(), vec.end(),      // Source
		  vec2.begin(),                // Destination
		  [](int x){ return x>80;});   // Condition 
// vec2: {87, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0}

copy_n(vec.begin(),  4, vec2.begin());  
// vec2: {9, 60, 70, 8, 0, 0, 0, 0, 0, 0, 0}

copy_backward(vec.begin(),  vec.end(),  // Source
		        vec2.end());              // Destination 
// vec2: {0, 0, 0, 0, 9, 60, 70, 8, 45, 87, 90}

// 2. Move 
vector<string> vec = {"apple", "orange", "pear", "grape"}; // 4 items
vector<string> vec2 = {"", "", "", "", "", ""};            // 6 items

move(vec.begin(), vec.end(), vec2.begin());
// vec:  {"", "", "", ""}  // Undefined
// vec2: {"apple", "orange", "pear", "grape", "", ""};
//
// If move semantics are defined for the element type, elements are moved over, 
// otherwise they are copied over with copy constructor, just like copy().

move_backward(vec.begin(), vec.end(), vec2.end());
// vec2: {"", "", "apple", "orange", "pear", "grape"};

vector<int> vec = {9,60,70,8,45,87,90};     // 7 items
vector<int> vec2 = {9,60,70,8,45,87,90};     // 7 items
vector<int> vec3 = {0,0,0,0,0,0,0,0,0,0,0}; // 11 items

// 3. Transform
transform(vec.begin(), vec.end(),    // Source
		    vec3.begin(),              // Destination
			 [](int x){ return x-1;});  // Operation 

transform(vec.begin(), vec.end(),           // Source #1
          vec2.begin(),                     // Source #2
		    vec3.begin(),                     // Destination
  	       [](int x, int y){ return x+y;});  // Operation
         // Add items from vec and vec2 and save in vec3 
         // vec3[0] = vec[0] + vec2[0]
         // vec3[1] = vec[1] + vec2[1]
         // ...

// 4. Swap - two way copying
swap_ranges(vec.begin(), vec.end(), vec2.begin());

// 5. Fill
vector<int> vec = {0, 0, 0, 0, 0};

fill(vec.begin(), vec.end(), 9); // vec: {9, 9, 9, 9, 9}

fill_n(vec.begin(), 3, 9);       // vec: {9, 9, 9, 0, 0}

generate(vec.begin(), vec.end(), rand); 

generate_n(vec.begin(), 3, rand); 

// 6. Replace
replace(vec.begin(), vec.end(),  // Data Range
		  6,                       // Old value condition
		  9);                      // new value                    

replace_if(vec.begin(), vec.end(),     // Data Range
			  [](int x){return x>80;},    // Old value condition
			  9);                         // new value                    

replace_copy(vec.begin(), vec.end(),  // Source
			  vec2.begin(),              // Destination
			  6,                         // Old value condition
			  9);                        // new value                    
  // Generalized form: replace_copy_if()

// 7. Remove
remove(vec.begin(), vec.end(), 3);   // Remove all 3's
remove_if(vec.begin(), vec.end(), [](int x){return x>80;});  
	 // Remove items bigger than 80

remove_copy(vec.begin(), vec.end(),  // Source
		      vec2.begin(),            // Destination
				6);                      // Condition 
   // Remove all 6's, and copy the remain items to vec2
   // Generalized form: remove_copy_if()

unique(vec.begin(), vec.end());   // Remove consecutive equal elements

unique(vec.begin(), vec.end(), less<int>());   
        // Remove elements whose previous element is less than itself

unique_copy(vec.begin(), vec.end(), vec2.begin());   
// Remove consecutive equal elements, and then copy the uniquified items to vec2
// Generalized form: unique_copy()
	
/*
 * Order-Changing Algorithms:
 *   - reverse, rotate, permute, shuffle 
 *   
 * They changes the order of elements in container, but not necessarily the
 * elements themselves.
 */

vector<int> vec =  {9,60,70,8,45,87,90};     // 7 items
vector<int> vec2 = {0,0,0,0,0,0,0};     // 7 items

// 1. Reverse
reverse(vec.begin()+1, vec.end()-1);
// vec: {9,87,45,8,70,60,90};     // 7 items

reverse_copy(vec.begin()+1, vec.end()-1, vec2.begin());
// vec2: {87,45,8,70,60,0,0};     



// 2. Rotate
rotate(vec.begin(), vec.begin()+3, vec.end());
// vec: {8,45,87,90,9,60,70};     // 7 items

rotate_copy(vec.begin(), vec.begin()+3, vec.end(),  // Source
		 vec2.begin());                               // Destination
       // Copy vec to vec2 in rotated order
       // vec is unchanged

// 3. Permute
next_permutation(vec.begin(), vec.end()); 
                           //Lexicographically next greater permutation
prev_permutation(vec.begin(), vec.end()); 
                           //Lexicographically next smaller permutation
// {1,2,3,5} < {1,2,4,4}
// {1,2}     < {1,2,3}

//Sorted in ascending order:  {8, 9, 45, 60, 70, 87, 90} 
//                            - Lexicographically smallest
//
//Sorted in descending order: {90, 87, 70, 60, 45, 9, 8} 
//                            - Lexicographically greatest

// Generalized form: next_permutation(), prev_permutation()

// 4. Shuffle  
//    - Rearrange the elements randomly 
//      (swap each element with a randomly selected element)
random_shuffle(vec.begin(), vec.end());
random_shuffle(vec.begin(), vec.end(), rand);

// C++ 11
shuffle(vec.begin(), vec.end(), default_random_engine()); 
// Better random number generation
```

## Algorithms for Equality, Equivalance

컨테이너에서 특정한 값과 동일한 값을 찾는 알고리즘을 algorithm for equality 이라 하고 다음과 같은 종류가 있다. 알고리즘에 함수가 사용되는 경우 `==` operator 를 사용한다.

```cpp
search
find_end
find_first_of
adjacent_search
```

컨테이너에서 특정한 값과 유사한 값을 찾는 알고리즘을 algorithm for equivalance 라 하고 다음과 같은 종류가 있다. 알고리즘에 함수가 사용되는 경우 `<` 와 같은 크기 비교 operator 를 사용한다. associative container (set, multiset, map, multimap) 와 같이 원소들이 정렬되어 저장된 경우만 사용가능하다.

```cpp
binary_search   // simple forms
includes
lower_bound
upper_bound
```

다음은 예이다.

```cpp
set<int> s = {21, 23, 26, 27};

/*
 * Algorithm find() looks for equality: if (x == y)  
 */

itr1 = find(s.begin(), s.end(), 36);  // itr1 points to s.end()

/*
 * set<int>::find() looks for equivalence: if ( !(x<y) && !(y<x) )
 */

itr2 = s.find(36);  // itr2 points to s.end()
```

## Algorithm vs Member Function

컨테이너 클래스에 알고리즘과 똑같은 일을 수행하는 멤버 함수들이 있다. 그러한 경우 알고리즘 대신 멤버 함수를 사용하자. 

```cpp
// Functions with same name:
// List:
void remove(const T); template<class Comp> void remove_if(Comp);
void unique();        template<class Comp> void unique(Comp);
void sort();          template<class Comp> void sort(Comp);
void merge(list&);    template<class Comp> void merge(Comp);
void reverse();

// Associative Container:
size_type count(const T&) const;
iterator find(const T&) const;
iterator lower_bound(const T&) const;
iterator upper_bound(const T&) const;
pair<iterator,iterator> equal_range (const T&) const;
// Note: they don't have generalized form, because comparison is defined by
//       the container.

// Unordered Container:
size_type count(const T&) const;
iterator find(const T&);
std::pair<iterator, iterator> equal_range(const T&);
// Note: No generalized form; use hash function to search

unordered_set<int> s = {2,4,1,8,5,9};  // Hash table 
unordered_set<int>::iterator itr;

// Using member function
itr = s.find(4);                      // O(1)

// Using Algorithm
itr = find(s.begin(), s.end(), 4);    // O(n)

// How about map/multimap?
map<char, string> mymap = {{'S',"Sunday"}, {'M',"Monday"}, {'W', "Wendesday"}, ...};

// Using member function
itr = mymap.find('F');                                           // O(log(n))

// Using Algorithm
itr = find(mymap.begin(), mymap.end(), make_pair('F', "Friday")); // O(n)

// How about list?
list<int> s = {2,1,4,8,5,9};

// Using member function
s.remove(4);                    // O(n)
// s: {2,1,8,5,9}

// Using Algorithm
itr = remove(s.begin(), s.end(), 4);  // O(n)
// s: {2,1,8,5,9,9}
s.erase(itr, s.end());
// s: {2,1,8,5,9}

// Sort
//
// Member function
s.sort();

// Algorithm
sort(s.begin(), s.end());   // Undefined behavior

// s: {2,4,1,8,5,9}
// s: {2,1,8,5,9,9}
/*
list<int>::iterator itr = remove(s.begin(), s.end(), 4);  // O(n)
s.erase(itr, s.end());
// Similarly for algorithm: remove_if() and unique()
*/

// Using member function
s.sort();

// Using Algorithm
sort(s.begin(), s.end());   // Undefined Behavior
```

## Modifying In A Container

assosiative container(set, map) 의 경우 키값을 바로 수정할 수는 없다. 수정하고 싶다면 제거한후 다시 삽입하자. 

```cpp
vector<int> vec = {1,2,3,4,5};
vec[2] = 9;   // vec: {1,2,9,4,5}

list<int> mylist = {1,2,3,4,5};
list<int>::iterator itr = mylist.find(3);
if (itr != mylist.end())
	*itr = 9;   // mylist: {1,2,9,4,5}

// How about modifying a set?
set<int> myset = {1,2,3,4,5};
set<int>::iterator itr = myset.find(3);
if (itr != myset.end()) {
	*itr = 9;     // Many STL implementation won't compile
	const_cast<int&>(*itr) = 9;  // {1,2,9,4,5} ???
}

// What about map
map<char,int> m;
m.insert ( make_pair('a',100) );
m.insert ( make_pair('b',200) );
m.insert ( make_pair('c',300) );
...
map<char,int>::iterator itr = m.find('b');
if (itr != m.end()) {
	itr->second = 900;   // OK
	itr->first = 'd';    // Error
}

// Same thing for multimap, multiset, unordered set/multiset, unordered map/multimap
/*
 * How to modify an element of associative container or unordered container?
 */
map<char,int> m;
m.insert ( make_pair('a',100) );
m.insert ( make_pair('b',200) );
m.insert ( make_pair('c',300) );
...
map<char,int>::iterator itr = m.find('b');
if (itr != m.end()) {
	pair<char,int> orig(*itr);
	orig.first = 'd';   
	m.insert(orig);
}
```

## Removing In A Container

컨테이너 클래스의 원소를 삭제하기 위해 알고리즘을 사용한다면 `remove` 하고 `erase` 해야 한다.

```cpp
/*
 * Remove from Vector or Deque
 */
  vector<int> vec = {1, 4, 1, 1, 1, 12, 18, 16}; // To remove all '1'
  for (vector<int>::iterator itr = vec.begin(); itr != vec.end(); ++itr) {
     if ( *itr == 1 ) {
        vec.erase(itr);
     }
  }   // vec: { 4, 12, 18, 16}
  // Complexity: O(n*m)

  remove(vec.begin(), vec.end(), 1);  // O(n) 
                                      // vec: {4, 12, 18, 16, ?, ?, ?, ?}
  
  vector<int>::iterator newEnd = remove(vec.begin(), vec.end(), 1);   // O(n)
  vec.erase(newEnd, vec.end());  

  // Similarly for algorithm: remove_if() and unique()

  // vec still occupy 8 int space: vec.capacity() == 8
  vec.shrink_to_fit();   // C++ 11
  // Now vec.capacity() == 4 

  // For C++ 03:
  vector<int>(vec).swap(vec); // Release the vacant memory
/*
 * Remove from List
 */
  list<int> mylist = {1, 4, 1, 1, 1, 12, 18, 16};

  list<int>::iterator newEnd = remove(mylist.begin(), mylist.end(), 1);  
  mylist.erase(newEnd, mylist.end());

  mylist.remove(1);  // faster

/*
 * Remove from associative containers or unordered containers
 */
  multiset<int> myset = {1, 4, 1, 1, 1, 12, 18, 16};

  multiset<int>::iterator newEnd = remove(myset.begin(), myset.end(), 1);  
  myset.erase(newEnd, myset.end()); // O(n)

  myset.erase(1); // O(log(n)) or O(1)

/*
 * Remove and do something else
 */

// Associative Container:
multiset<int> s = {1, 4, 1, 1, 1, 12, 18, 16};;

multiset<int>::iterator itr;
for (itr=s.begin(); itr!=s.end(); itr++) {
   if (*itr == 1) {
      s.erase(itr);      
      cout << "Erase one item of " << *itr << endl;
   } 
}

// First erase OK; second one is undefined behavior

//Solution:
multiset<int>::iterator itr;
for (itr=s.begin(); itr!=s.end(); ) {
   if (*itr == 1) {
      cout << "Erase one item of " << *itr << endl;
      s.erase(itr++);
   } else {
      itr++;
   }
}

// Sequence Container:
vector<int> v = {1, 4, 1, 1, 1, 12, 18, 16};
vector<int>::iterator itr2;
for (itr2=v.begin(); itr2!=v.end(); ) {
   if (*itr2 == 1) {
      cout << "Erase one item of " << *itr2 << endl;
      v.erase(itr2++);
   } else {
      itr2++;
   }
}

// Sequence container and unordered container's erase() returns  
// iterator pointing to next item after the erased item.

//Solution:
for (itr2=v.begin(); itr2!=v.end(); ) {
   if (*itr2 == 1) {
      cout << "Erase one item of " << *itr2 << endl;
      itr2 = v.erase(itr2);
   } else {
      itr2++;
   }
}

// 1. Sequence container and unordered container's erase() returns the next 
//    iterator after the erased item.
// 2. Associative container's erase() returns nothing.
// 
// A thing about efficiency: v.end()

vector<int> c = {1, 4, 1, 1, 1, 12, 18, 16};

// Use Algorithm
bool equalOne(int e) {
   if (e == 1) {
      cout << e << " will be removed" << endl;
      return true;
   }
   return false;
}
auto itr = remove_if(c.begin(), c.end(), equalOne);
c.erase(itr, c.end());

// Use bind():
bool equalOne(int e, int pattern) {
   if (e == pattern) {
      cout << e << " will be removed" << endl;
      return true;
   }
   return false;
}
remove_if(v.begin(), v.end(), bind(equalOne, placeholders::_1, 1));

// Lambda:
auto itr = remove_if(v.begin(), v.end(), 
      [](int e){ 
         if(e == 1) {
            cout << e << " will be removed" <<endl; return true; 
         } 
      } 
   );

```

## shared_ptr

`share_ptr` 는 `strong refCnt, weak refCnt` 를 갖고 있다.  `strong refCnt` 를 이용하여 객체의 수명을 관리한다. `shared_ptr` 은 `strong refCnt` 를 이용하여 객체의 수명을 관리하는 똑똑한 포인터이다.

```cpp
class Dog {
    string m_name;
  public:
      void bark() { cout << "Dog " << m_name << " rules!" << endl; }
      Dog(string name) { cout << "Dog is created: " << name << endl; m_name = name; }
      Dog() { cout << "Nameless dog created." << endl; m_name = "nameless"; }
     ~Dog() { cout << "dog is destroyed: " << m_name << endl; }
	  //void enter(DogHouse* h) { h->setDog(shared_from_this()); }  // Dont's call shared_from_this() in constructor
};

class DogHouse {
    shared_ptr<Dog> m_pD;
public:
    void setDog(shared_ptr<Dog> p) { m_pD = p; cout << "Dog entered house." << endl;}
};

int main ()
{
    shared_ptr<Dog> pD(new Dog("Gunner"));
    shared_ptr<Dog> pD = make_shared<Dog>(new Dog("Gunner")); // faster and safer
    
    pD->bark();
    
    (*pD).bark();
    
    //DogHouse h;
//    DogHouse* ph = new DogHouse();
//    ph->setDog(pD);
//    delete ph;    
    
    //auto pD2 = make_shared<Dog>( Dog("Smokey") ); // Don't use shared pointer for object on stack.
//    auto pD2 = make_shared<Dog>( *(new Dog("Smokey")) ); 
//    pD2->bark();
//
//    Dog* p = new Dog();
//    shared_ptr<int> p1(p);
//    shared_ptr<int> p2(p);  // Erroneous
    
    shared_ptr<Dog> pD3;
    pD3.reset(new Dog("Tank"));
    pD3.reset();  // Dog destroyed. Same effect as: pD3 = nullptr;
//    
    //pD3.reset(pD.get());  // crashes
    
    /********** Custom Deleter ************/
    shared_ptr<Dog> pD4( new Dog("Victor"), 
                        [](Dog* p) {cout << "deleting a dog.\n"; delete p;}
                        );
                        // default deleter is operator delete.
                        
    //shared_ptr<Dog> pDD(new Dog[3]);
    shared_ptr<Dog> pDD(new Dog[3], [](Dog* p) {delete[] p;} );
```

## weak_ptr

`shared_ptr` 를 `weak_ptr` 로 할당 해도 `shared_ptr` 의 `strong refCnt` 는 변하지 않고 `weak refCnt` 만 증가한다.

```cpp
class Dog {
  //shared_ptr<Dog> m_pFriend;
  weak_ptr<Dog> m_pFriend;
 public:
  string m_name;
  void bark() { 
     cout << "Dog " << m_name << " rules!" << endl; 
  }
  Dog(string name) { 
    cout << "Dog is created: " << name << endl; m_name = name; 
  }
  ~Dog() {
     cout << "dog is destroyed: " << m_name << endl; 
  }
  void makeFriend(shared_ptr<Dog> f) { m_pFriend = f; }
  void showFriend() { 
    //cout << "My friend is: " << m_pFriend.lock()->m_name << endl;
    if (!m_pFriend.expired()) 
      cout << "My friend is: " << m_pFriend.lock()->m_name << endl;
    cout << " He is owned by " << m_pFriend.use_count() << " pointers." << endl; 
  }
};

int main ()
{
    shared_ptr<Dog> pD(new Dog("Gunner"));
    shared_ptr<Dog> pD2(new Dog("Smokey"));
    pD->makeFriend(pD2);
    pD2->makeFriend(pD);
    
    pD->showFriend();
}
```

## unique_ptr

단 하나의 주인만 허용하는 똑똑한 포인터이다. `unique_ptr` 가 `scope` 를 벗어나면 객체를 파괴한다.

```cpp
// Unique Pointers: exclusive owenership

class Dog {
  //Bone* pB;
  unique_ptr<Bone> pB;  // This prevents memory leak even constructor fails.
 public:
  string m_name;
  void bark() { 
    cout << "Dog " << m_name << " rules!" << endl; 
  }
  Dog() { 
    pB = new Bone(); 
    cout << "Nameless dog created." << endl; m_name = "nameless"; 
  }
  Dog(string name) { 
    cout << "Dog is created: " << name << endl; m_name = name; 
  }
  ~Dog() { 
    delete pB; 
    cout << "dog is destroyed: " << m_name << endl; 
  }
};

void test() {    
  //Dog* pD = new Dog("Gunner");
  unique_ptr<Dog> pD(new Dog("Gunner"));

  pD->bark();
  /*pD does a bunch of different things*/

  //Dog* p = pD.release();
  pD = nullptr;
  //pD.reset(new Dog("Smokey"));

  if (!pD) {
    cout << "pD is empty.\n";
  }
//delete pD;   
}

void f(unique_ptr<Dog> p) {
  p->bark();
}

unique_ptr<Dog> getDog() {
  unique_ptr<Dog> p(new Dog("Smokey"));
  return p;
}

void test2() {
   unique_ptr<Dog> pD(new Dog("Gunner"));
   unique_ptr<Dog> pD2(new Dog("Smokey"));
   pD2 = move(pD);
   // 1. Smokey is destroyed
   // 2. pD becomes empty.
   // 3. pD2 owns Gunner.

   pD2->bark();
   //    f(move(pD));
   //    if (!pD) {
   //        cout << "pD is empty.\n";
   //    }
   //    
   //    unique_ptr<Dog> pD2 = getDog();
   //    pD2->bark();

   unique_ptr<Dog[]> dogs(new Dog[3]);
   dogs[1].bark();
   //(*dogs).bark(); // * is not defined
}

void test3() {
    // prevent resource leak even when constructor fails
}

int main ()
{
    test2();
}
```

## input output stream

* [Input/Output @ cplusplus](http://www.cplusplus.com/reference/iolibrary/)

----

* 다음은 stringstream 을 이용하여 word 를 가져오는 예이다.

```cpp
float num; 
stringstream ss; 
string s = "25 1 3 .235\n1111111\n222222";	
ss.str(s); 
while(ss >> num)
  cout << "num: " << num << endl;
// num: 25
// num: 1
// num: 3
// num: 0.235
// num: 1.11111e+06
// num: 222222  
```

* 다음은 getline 을 이용하여 word 를 가져오는 예이다.

```cpp
int main(void)
{
  std::istringstream iss("foo bar baz");
  std::string word;
 
  while (std::getline(iss, word, ' '))
  {
    printf("word: %s\n", word.c_str());
  }  
  return 0;
}
// word: goo
// word: bar
// word: baz
```

## random_device

```cpp
// random_device example
#include <iostream>
#include <random>

int main ()
{
  std::random_device rd;

  std::cout << "default random_device characteristics:" << std::endl;
  std::cout << "minimum: " << rd.min() << std::endl;
  std::cout << "maximum: " << rd.max() << std::endl;
  std::cout << "entropy: " << rd.entropy() << std::endl;
  std::cout << "a random number: " << rd() << std::endl;

  return 0;
}
```
