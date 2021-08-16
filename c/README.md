# Materials

* [c faq kor](http://cinsk.github.io/cfaqs/html/cfaqs-ko.html)

# Basic Usage

## Declaration

선언문의 구조는 다음과 같다.
declarator는 initializer를 포함할 수 있다.

```
storage class, base type, type qualifier, declarator 
```

* storage class: static, extern, auto, typedef
* type qualifier: const

## Designated Initializers (C99)

* [6.29 Designated Initializers @ gcc](http://gcc.gnu.org/onlinedocs/gcc/Designated-Inits.html)
* [What does dot (.) mean in a struct initializer? @ stackoverflow](https://stackoverflow.com/questions/8047261/what-does-dot-mean-in-a-struct-initializer)

----

다음과 같은 struct 를 선언해보자.

```c
struct Foo {
  int     first;
  int     second;
  int     third;
};
```

다음의 방법들로 초기화할 수 있다.

```c
struct Foo foo = { 1, 2, 3 };

struct Foo foo = { 
    .first = 1, 
    .second = 2, 
    .third = 3 
};

struct Foo foo = { 
    .first = 1, 
    .third = 3,
    .second = 2
};
```
