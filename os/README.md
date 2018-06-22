

# Abstract

운영체제 (Operating System) 에 대해 정리한다.

# Materials

* [Windows 구조와 원리](http://www.hanbit.co.kr/store/books/look.php?p_code=B6822670083)
  * 오래전에 출간되어 절판되었지만 한글로 된 책들중 최강이다.
* [Write Great Code I](http://www.plantation-productions.com/Webster/www.writegreatcode.com/)
  * 킹왕짱
* [google interview university @ github](https://github.com/jwasham/coding-interview-university)
  * operating system 관련된 링크를 참고하자. 쓸만한 비디오가 잘 정리되어 있다.

# Computer System Architecture 

프로세서의 역사를 다음과 같이 간략히 표현할 수 있다.

| 프로세서 | 도입년도 | 레지스터 크기 | 데이터 버스 | 어드레스 버스 | clock speed |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 8008 | 1972 | 8 | 8 | 8 | |
| 8080 | 1974 | 8 | 8 | 16 | |
| 8086/88 | 1978 | 16/16 | 16/8 | 20 | 8Mhz |
| 80286 | 1982 | 16 | 16 | 24 | 12.5Mhz |
| 80386 | 1985 | 32 | 32 | 32 | 20Mhz |
| 80486 | 1989 | 32 | 32 | 32 | 25Mhz |
| pentium | 1993 | 32 | 64 | 32 | 60Mhz |
| pentium pro | 1995 | 32 | 64 | 36 | 200Mhz |
| pentium II | 1997 | 32 | 64 | 36 | 266Mhz |
| pentium III | 1999 | 32 | 64 | 36 | 500Mhz |

## ENIAC (Electronic Numerical Integrator And Computer)

최초의 디지털 컴퓨터는 1946년 완성된 ENIAC이다. 펜실베니아 대학의 전기 공학 무어 스쿨 (Moore School of Electrical Engineering) 에서 제작되었다. 높이가 18 피트, 길이가 80 피트, 무게가 30 톤 이상이었다. 그러나 프로그래밍을 위해 6,000 여개의 스위치를 조작하고 각종 케이블을 연결해야 했다.

## Von Neumann Architecture

ENIAC 프로젝트의 고문이었던 수학자 폰 노이만 (John Von Neumann) 은 프로그램 내장 방식 (Stored-program Concept) 을 고안했다. 이것을 Von Neumann Machine 이라고 부른다. 

Von Neumann Machine 은 프로그램과 데이터를 실행 되기 전에 메모리에 올려 놓고 프로그램이 실행 될 때에는 프로그램의 명렁어와 데이터들을 메모리로부터 불러들이고 프로그램에 대한 저장 또는 변경 역시 기억장치에 저장되어 있는 프로그램을 변경함으로써 가능하게 한다. 현대의 컴퓨터와 거의 똑같은 모델이다.

## IAS (Institute for Advanced Study) Machine

1952 년 프린스턴 대학에서 Von Neumann Architecture 을 IAS 라는 이름으로 구현한다. Von Neumann Machine 이라고 부르기도 한다.

IAS 의 메모리는 1,000 개의 저장소로 이루어져 있었다. 각 저장소는 40비트로 구성되었다. 저장소는 데이터, 명령어들을 표현할 수 있었다.

```
Data

0 1                                        39
+-------------------------------------------+
| |                                         |
+-------------------------------------------+
 ^
 |
 sign bit

Commands

0        8          19 20     28           39
+-------------------------------------------+
|        |            |        |            |
+-------------------------------------------+
     ^          ^          ^          ^
  opcode     address     opcode     address

```

IAS 는 다읕과 같이 7개의 레지스터를 가지고 있다.

| name | description |
:-----:|:------------:
| MBR (Memory Buffer Register) | 메모리로부터 읽어들인 데이터를 저장하고 있다. |
| MAR (Memory Address Register) | MBR 로 읽어들일 메모리 주소를 저장한다. |
| IR (Instruction Register) | 실행될 명령어를 저장한다. |
| IBR (Instruction Buffer Register) | 메모리로부터 읽어들인 명령어의 내용을 임시로 저장한다. |
| PC (Program Counter) | 다음번에 실행될 명렁어를 가져올 메모리 주소를 저장한다. |
| AC (Accumulator) | ALU 로부터 계산된 내용을 임시로 저장한다. |
| MQ (Multiplier Quotient) | ALU 로부터 계산된 내용을 임시로 저장한다. |

IAS 의 명령어 사이클은 패치 사이클 (Fetch Cycle), 실행 사이클 (Execution Cycle) 로 구성된다.

다음은 IAS 의 구조를 표현한 그림이다.

![]()

다음은 IAS 의 명령어를 처리하는 과정을 표현한 그림이다.

![]()

[이것](https://www.youtube.com/watch?v=mVbxrQE4f90)은 IAS 의 명령어 사이클을 설명한 동영상이다.

# Data Representation

## Floating Point 

## Byte Order

# machine language

# Procedure and Stack

# Process and Thread

# Thread Scheduling

# Thread synchronization

# Memory Management

# Segmentation

# Paging

# Page Management

# Cache Management

# Windows Cache Management

# Userlevel and Kernellevel

# Execution file and Loader

* [Portable Executable Format](../pef/README.md)
  * 윈도우즈의 실행파일 포맷

* [Elf](../elf/README.md)
  * 리눅스의 실행파일 포맷