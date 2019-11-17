- [Abstract](#abstract)
- [Intro](#intro)
- [References](#references)
- [make Usages](#make-usages)
  - [`--dry-run`](#dry-run)
- [Makefile Usages](#makefile-usages)
  - [Special Built-in Target Names](#special-built-in-target-names)
    - [`.PHONY`](#phony)
    - [`.SUFFIXES`](#suffixes)
  - [Conditional Parts of Makefiles](#conditional-parts-of-makefiles)
  - [Automatic Variables](#automatic-variables)
    - [`$*`](#)
    - [`$@`](#)
    - [`$<`, `<$?`](#)
  - [Functions for Transforming Text](#functions-for-transforming-text)
    - [shell](#shell)
    - [patsubst](#patsubst)
    - [notdir](#notdir)
    - [wildcard](#wildcard)
    - [abspath](#abspath)
  - [Pattern Rules](#pattern-rules)
  - [Directives](#directives)
  - [variable assignment](#variable-assignment)

----

# Abstract

make 에 대해 정리한다.

# Intro

Makefile 의 규칙은 보통 다음과 같다.

```make
target … : prerequisites …
        recipe
        …
        …
```

# References

* [make tutorial](https://makefiletutorial.com/)
  * the most wonderful tutorial
* [GNU Make 강좌](https://wiki.kldp.org/KoreanDoc/html/GNU-Make/GNU-Make.html#toc3)
  * 오래되긴 하였지만 한글 이다.
* [GNU Make @ kgnu](http://korea.gnu.org/manual/release/make/make-sjp/make-ko_toc.html)  
  * 오래되긴 하였지만 한글 이다.
* [make manual @ gnu](https://www.gnu.org/software/make/manual/make.html#Automatic-Variables)

# make Usages

## `--dry-run`

해당 target을 실행하지는 않고 recipe를 출력한다.
Makefile을 옳바로 작성했는지 검사할 때 유용하다.

```
make --dry-run proto
```

# Makefile Usages

## Special Built-in Target Names

### `.PHONY`

.PHONY의 prerequisites에 등록된 target들은 해당 target이름과 같은
file이 존재해도 동작 한다.

```
clean :
    rm -rf *.bin
    
.PHONY : clean
```

### `.SUFFIXES`

.SUFFIXES의 prerequisites에 등록된
 확장자들은
 [suffix rule](https://www.gnu.org/software/make/manual/make.html#Suffix-Rules)을
 따른다. 그러나 suffix rule은 하위 호환성 때문에 여전히 작동하지만
 현대의 make에서 추천하지 않는다. suffix rule보다는 pattern rule을
 사용하자.
 
suffixe rule은 prerequisites가 없다. 
 
```
.c.o:
        $(CC) -c $(CFLAGS) $(CPPFLAGS) -o $@ $<
```

## Conditional Parts of Makefiles

ifeq else endif 를 이용하여 conditional expression이 가능하다.

```make
libs_for_gcc = -lgnu
normal_libs =

foo: $(objects)
ifeq ($(CC),gcc)
        $(CC) -o foo $(objects) $(libs_for_gcc)
else
        $(CC) -o foo $(objects) $(normal_libs)
endif
```

다음은 TARGET변수를 이용하여 플래폼별로 설정을 달리한다.

```make
TARGET  := $(shell uname -s | tr '[A-Z]' '[a-z]' 2>/dev/null || echo unknown)

ifeq ($(TARGET), sunos)
	CFLAGS += -D_PTHREADS -D_POSIX_C_SOURCE=200112L
	LIBS   += -lsocket
else ifeq ($(TARGET), darwin)
	LDFLAGS += -pagezero_size 10000 -image_base 100000000
else ifeq ($(TARGET), linux)
	CFLAGS  += -D_POSIX_C_SOURCE=200112L -D_BSD_SOURCE
	LIBS    += -ldl
	LDFLAGS += -Wl,-E
else ifeq ($(TARGET), freebsd)
	CFLAGS  += -D_DECLARE_C99_LDBL_MATH
	LDFLAGS += -Wl,-E
endif
```

## Automatic Variables

implicit rule이 적용되었 을 때 특별한 의미와 함께 recipe에 사용할 수
있는 변수 들이다. 주로 `$`로 시작한다.

### `$*` ###

확장자가 없는 현재의 target

```make
main.o : main.c io.h
gcc -c $*.c
```

위의 예제를 살펴보자. `$*`는 `main`과 같다.

### `$@` ###

현재의 target

```make
test : $(OBJS)
gcc -o $@ $*.c
```

위의 예제를 살펴보자. `$@`는 test와 같다.

### `$<`, `<$?` ###

현재의 target보다 더 최근에 갱신된 파일 이름. `.o` 파일보다 더 최근에
갱신된 .c 파일은 자동적으로 컴파일이 된다. 가령 main.o를 만들고 난
다음에 main.c를 갱신하게 되면 main.c는 $<의 작용에 의해 새롭게
컴파일이 된다.

## Functions for Transforming Text

### shell ###

```make
contents := $(shell cat foo)
```

### patsubst ###

pattern substitution이다. 아래의 결과는 `x.c.o bar.o`이다.

```make
$(patsubst %.c,%.o,x.c.c bar.c)
```

### notdir ###

디렉토리를 제거한 파일이름을 얻어온다. 다음의 결과는
`foo.c hacks`이다.

```make
$(notdir src/foo.c hacks)
```

### wildcard ###

파일들의 목록을 얻어온다.

```make
objects := $(wildcard *.o)
```

### abspath ###

절대 경로를 얻어온다.

```make
objects := $(abspath names...)
```

## Pattern Rules

패턴 룰은 반드시 `%`를 포함한다. 패턴 룰은 보통 다음과 같다. 

```
%.o : %.c 
    recipe…
```

## Directives

* vpath

현재 디렉토리에서 prerequites의 파일들을 검색할 수 없을때
추가로 검색할 디렉토리를 정해주자.

```
vpath %.h ../headers
```

## variable assignment

* Lazy Set

VARIABLE이 사용될 때 마다 evaluation된다

```
VARIABLE = value
```

* Immediate Set

VARIABLE이 선언될 때 evaluation된다.

```
VARIABLE := value
```

* Conditional Set

VARIABLE이 값을 가지고 있지 않을때만 값을 배정하자.

```
VARIABLE ?= value
```

* Append Set

```
VARIABLE += value
```
