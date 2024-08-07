
- [Abstract](#abstract)
- [Materials](#materials)
- [Computer System Architecture Overview](#computer-system-architecture-overview)
- [The Operating System](#the-operating-system)
- [The Kernel](#the-kernel)
- [Stored Program Concept](#stored-program-concept)
- [Bus Overview](#bus-overview)
- [문자셋의 종류와 특징](#문자셋의-종류와-특징)
- [MBCS, WBCS 동시 지원](#mbcs-wbcs-동시-지원)
- [32Bit vs 64Bit](#32bit-vs-64bit)
- [Design Minimal CPU Instruction Set](#design-minimal-cpu-instruction-set)
- [Direct Address Mode and Indirect Address Mode](#direct-address-mode-and-indirect-address-mode)
- [Process](#process)
- [Scheduler](#scheduler)
- [Process Status](#process-status)
- [Procedure And Stack](#procedure-and-stack)
- [Process And Thread](#process-and-thread)
- [User Level Thread vs Kernel Level Thread](#user-level-thread-vs-kernel-level-thread)
- [Thread Scheduling](#thread-scheduling)
  - [Thread Status](#thread-status)
- [Thread synchronization](#thread-synchronization)
  - [Critical Section](#critical-section)
  - [Mutex](#mutex)
  - [Semaphore](#semaphore)
  - [Event](#event)
  - [Kernel Object](#kernel-object)
- [Memory Management](#memory-management)
- [Segmentation](#segmentation)
- [Paging](#paging)
- [Page Management](#page-management)
- [Processor Cache Management](#processor-cache-management)
- [Windows Cache Management](#windows-cache-management)
- [User mode, Kernel mode](#user-mode-kernel-mode)
- [Virtual Memory Control](#virtual-memory-control)
- [Heap Control](#heap-control)
- [MMF (Memory Mapped File)](#mmf-memory-mapped-file)
- [DLL (Dynamic Link Library)](#dll-dynamic-link-library)
- [Execution file and Loader](#execution-file-and-loader)
- [File System](#file-system)
- [Journaling File System](#journaling-file-system)
- [Quiz](#quiz)

----

# Abstract

운영체제 (Operating System) 에 대해 정리한다. Windows, Linux 를 구분해서 정리한다. [Computer System Architecture](/csa/README.md) 의 내용을 먼저 이해해야 한다.  [linux kernal](/linuxkernel/README.md) 에서 linux kernel 을 정리한다. [osimpl](/osimpl/README.md) 에서 실제로 구현해본다.

# Materials

- [Dive into Systems](https://diveintosystems.org/)
  - 킹왕짱 Computer System Architecture, Operating System 책
* [Kernel of Linux | OLC](https://olc.kr/course/course_online_view.jsp?id=35&s_keyword=Kernel&x=0&y=0)
  * 고건 교수 동영상 강좌 
  * 가장 깊이 있는 한글 동영상 강좌
  * [pdf](https://olc.kr/classroom/library.jsp?cuid=283080)
* [혼자 공부하는 컴퓨터구조 + 운영체제](https://github.com/kangtegong/self-learning-cs)
  * [src](https://github.com/kangtegong/self-learning-cs)
  * [video](https://www.youtube.com/playlist?list=PLVsNizTWUw7FCS83JhC1vflK8OcLRG0Hl)
* [뇌를 자극하는 윈도우즈 시스템 프로그래밍 | youtube](https://www.youtube.com/playlist?list=PLVsNizTWUw7E2KrfnsyEjTqo-6uKiQoxc)
  * [book | yes24](https://www.yes24.com/Product/Goods/2502445)
  * 이해가 쉬운 동영상 강좌 
* [linux 0.01](https://github.com/zavg/linux-0.01)
  * 토발즈가 릴리즈한 최초 리눅스 커널
  * gcc 1.x 에서 빌드가 된다.
* [linux 0.01 remake](http://draconux.free.fr/os_dev/linux0.01_news.html)
  * gcc 4.x 에서 빌드가 되도록 수정된 fork
  * [src](https://github.com/issamabd/linux-0.01-remake)
* [linux 0.01 commentary](https://www.academia.edu/5267870/The_Linux_Kernel_0.01_Commentary)
* [linux 0.01 running on qemu](https://iamhjoo.tistory.com/11)
* [C++로 나만의 운영체제 만들기](http://www.yes24.com/Product/goods/64574002)
  * [src](https://github.com/pdpdds/SkyOS)
* [Source code listing for the Lions' Commentary in PDF and PostScript](http://v6.cuzuco.com/)
  * 역사가 깊은 유닉스 코드와 주석
  * [src](https://github.com/zrnsm/lions-source)
* [cracking the coding interview](http://www.crackingthecodinginterview.com/)
  * threads and lock quiz 가 볼만함
* [linux-insides](https://0xax.gitbooks.io/linux-insides/content/index.html)
  * [src](https://github.com/0xAX/linux-insides)
  * [번역](https://github.com/junsooo/linux-insides-ko) 
  * 리눅스 커널에 대해 설명한다.
* [밑바닥부터 만드는 컴퓨팅 시스템](https://www.nand2tetris.org/)
  * [번역서](http://www.yes24.com/Product/Goods/71129079?scode=032&OzSrank=1)
  * 블리언로직부터 운영체제까지 제작해보기
* [Windows Internals, Part 1: System architecture, processes, threads, memory management, and more (7th Edition)](https://www.amazon.com/Windows-Internals-Part-architecture-management/dp/0735684189)
  * [번역서](http://www.yes24.com/24/Goods/57905305?Acode=101#contentsConstitution)
  * 윈도우즈 커널 추천 도서
* [Linux Kernel Development (3rd Edition)](https://www.amazon.com/Linux-Kernel-Development-Robert-Love/dp/0672329468)
  * [번역서](http://www.yes24.com/24/Goods/7351874?Acode=101)
  * 리눅스 커널 추천 도서
* [Understanding the Linux Kernel, 3rd Edition](http://shop.oreilly.com/product/9780596005658.do)
  * [번역서](http://www.yes24.com/24/Goods/2157231?Acode=101)
  * 리눅스 커널 추천 도서
* [Understanding the Linux Virtual Memory Manager](https://www.amazon.com/Understanding-Linux-Virtual-Memory-Manager/dp/0131453483)
  * 리눅스 커널 추천 도서
* [Intel 80386 Reference Programmer's Manual](https://pdos.csail.mit.edu/6.828/2005/readings/i386/toc.htm)
* [Paging and Segmentation | youtube](https://www.youtube.com/watch?v=5ioo7ezWn1U&list=PLWi7UcbOD_0uhZqGfWbpQ_Ym30ehQCeyq)
  *  virtual address to physical address translation 은 설명이 좋다.
* [Memory Management : Segmentation 2](http://anster.egloos.com/2138204)
  * 메모리 관리를 잘 정리한 블로그
* [High Performance Computer Architecture | udacity](https://www.udacity.com/course/high-performance-computer-architecture--ud007)
  * 체계적인 인강 그러나 virtual address to physical address translation 은 설명이 부족하다.
* [Introduction to Operating Systems](https://classroom.udacity.com/courses/ud923)
  * Kernel. vs User-level threads 가 정말 좋았음
  * [wiki](https://www.udacity.com/wiki/ud923)
* [Windows 구조와 원리](http://www.hanbit.co.kr/store/books/look.php?p_code=B6822670083)
  * 오래전에 출간되어 절판되었지만 한글로 된 책들중 최강이다.
* [Write Great Code I](http://www.plantation-productions.com/Webster/www.writegreatcode.com/)
  * 킹왕짱
* [google interview university @ github](https://github.com/jwasham/coding-interview-university)
  * operating system 관련된 링크를 참고하자. 쓸만한 비디오가 잘 정리되어 있다.

# Computer System Architecture Overview

![](computerhwsystem.png)

# The Operating System

- [13. The Operating System | DiveIntoSystems](https://diveintosystems.org/book/C13-OS/index.html)

운영체제(Operating System, OS)는 컴퓨터 시스템에서 하드웨어와 소프트웨어 리소스를 관리하고, 사용자와 컴퓨터 사이의 인터페이스 역할을 하는 시스템 소프트웨어입니다. 운영체제의 주요 목적은 사용자가 컴퓨터와 편리하게 상호 작용할 수 있도록 하드웨어 리소스를 효과적으로 관리하는 것입니다.

![](img/2024-01-07-16-39-45.png)

# The Kernel

- [13. The Operating System | DiveIntoSystems](https://diveintosystems.org/book/C13-OS/index.html)

커널(Kernel)은 운영체제의 핵심 구성 요소로, 하드웨어와 소프트웨어 사이에서 가장 기본적이면서 핵심적인 기능들을 수행하는 중요한 부분입니다. 커널은 시스템의 성능, 안정성 및 보안을 담당하며, 컴퓨터 하드웨어와 상호 작용하여 프로세스, 메모리, 파일 시스템, 입출력 장치 등의 자원을 효과적으로 관리하는 역할을 합니다.

![](img/2024-01-07-16-41-30.png)

# Stored Program Concept

![](storedprogramconcept.png)

[Von Neumann Architecture](/csa/README.md#von-neumann-architecture) 라고도 한다. fetch, decode, execute 과정으로 프로그램을 실행한다.

* fetch
  * CPU 내부로 명령어 이동
* decode
  * 명령어 해석
  * 컨트롤 유닛
* execution
  * 연산을 진행
  * 보통은 ALU 를 생각

# Bus Overview

![](busoverview.png)

Bus 의 종류는 **Data Bus**, **Address Bus**, **Control Bus** 가 있다.

* Data Bus
  * 데이터 이동
* Address Bus
  * 메모리 주소 이동
* Control Bus
  * 컨트롤 신호 이동   

# 문자셋의 종류와 특징

문자셋은 다음과 같은 종류가 있다.

* SBCS (single byte character set)
  * 문자를 표현하는데 1 바이트 사용
  * 아스키코드
* MBCS (multi byte character set)
  * 한글은 2 바이트, 영문은 1 바이트를 사용
* WBCS (wide byte character set)
  * 문자를 표현하는데 SBCS 의 두배 즉 2 바이트를 사용하겠다. 
  * 유니코드

한글은 MBCS 로 구현하면 `strlen` 이 제대로 동작하지 않는다. WBCS 를 이용하자. WBCS 를 이용하려면 SBCS API 에 대응되는 WBCS API 를 사용한다. (`strlen vs wcslen`)  

application 을 WBCS 로 구현하려면 `int main(int argc, char* argv[])` 대신 `int wmain(int argc, wchar_t* argv[])` 를 사용해야 한다.

# MBCS, WBCS 동시 지원

`char, wchar_t` 를 동시에 지원하기 위해 `windows.h` 에 다음과 같이 정의되어 있다. `UNICODE` 를 선언하면 WBCS 를 구현할 수 있다.

```c
#ifdef UNICODE
  typedef WCHAR   TCHAR;
  typedef LPWSTR  LPTSTR;
  typedef LPCWSTR LPCTSTR;  
#else
  typedef CHAR   TCHAR;
  typedef LPSTR  LPTSTR;
  typedef LPCSTR LPCTSTR;  
#endif
```

* `UNICODE` 가 선언되어 있다면 

```c
TCHAR arr[10] => WCHAR arr[10] => wchar_t arr[10] 
```

* `UNICODE` 가 선언되어 있지 않다면

```c
TCHAR arr[10] => CHAR arr[10] => char arr[10] 
```

또한 `"", L""` 을 동시에 지원하기 위해 `windows.h` 에 다음과 같이 정의되어 있다.

```c
#ifdef _UNICODE
  #define __T(x)  L ## x
#else
  #define __T(x)  x
#endif

#define _T(x)  __T(x)
#define _TEXT(x)  __T(x)
```

따라서 `UNICODE` 가 정의되어 있으면 `_T("A") => __T("A") => L"A"` 이다. 그리고 `UNICODE` 가 정의되어 있지 않으면 `_T("A") => __T("A") => "A"` 이다.

또한 `strlen, wcslen` 과 같은 API 를 동시에 지원하기 위해 `windows.h` 에 다음과 같이 정의도어 있다.

```c
#ifdef _UNICODE
  #define _tmain wmain
  #define _tcslen wcslen
  #define _tprintf wprintf
  #define _tscanf wscanf
#else
  #define _tmain main
  #define _tcslen strlen
  #define _tprintf printf
  #define _tscanf scanf
#endif
```

# 32Bit vs 64Bit

다음과 같은 차이점에 따라 32bit CPU 인지 64bit CPU 인지 구분이 가능하다.

* 한번에 송수신 가능한 데이터의 크기가 32bit 이냐 64bit 이냐
* CPU 가 한번에 처리하는 데이터의 크기가 32bit 이냐 64bit 이냐

64bit application 을 제작하더라도 32bit application 을 코딩할 때와 별 차이가 없다. 대신 데이터 모델을 유의하자. CPU, OS 에 다음과 같이 데이터 모델이 달라진다. [참고](https://dojang.io/mod/page/view.php?id=737)

| 데이터 모델     | short | int | long | long long | pointer | CPU, OS                               |
| --------------- | ----- | --- | ---- | --------- | ------- | ------------------------------------- |
| LLP64 / IL32P64 | 2     | 4   | 4    | 8         | 8       | x86_64: Windows                       |
| LP64 / I32LP64  | 2     | 4   | 8    | 8         | 8       | x86_64: UNIX, Linux, SUN OS, BSD, OSX |

# Design Minimal CPU Instruction Set

* 사칙연산을 위한 `ADD, SUB, MUL, DIV`
  * 피연산자는 register 뿐 이다. memory 를 접근할 instruction 이 필요하다.
* 메모리 접근을 위한 `LOAD, STORE`
  * 메모리의 정보를 레지스터로 가져오거나 레지스터의 정보를 메모리에 저장한다.

# Direct Address Mode and Indirect Address Mode

* `LOAD r1, 0x10`
  * Memory 의 `0x10` 주소에서 한바이트 값을 가져와서 register `r1` 에 저장한다. 이것을 direct address mode 라고 한다.
* `LOAD r1, 0x30`
  * Memory 의 `0x30` 주소의 값은 `0x10` 이다. 메모리의 `0x10` 주소에서 한바이트 값을 가져와서 register `r1` 에 저장한다. 이렇게 한번더 참조해서 값을 가져오는 것을 indirect address mode 라고 한다.

# Process

메인 메모리로 이동하여 실행중인 프로그램을 프로세스라고 한다.

# Scheduler

둘 이상의 프로세스가 적절히 실행되도록 컨트롤하는 소프트웨어이다. OS 의 부분 요소이다.

**General OS** 는 preemptive (선점형) 방식으로 scheduling 한다. time slice 가 길다. 프로세스가
생성될 때 마다 priority 를 봐서 기존의 것보다 높으면 새로운 프로세스의 상태를 running 으로 바꾼다.

**Realtime OS** 는 non-preemptive (비선점형) 방식으로 scheduling 한다. time slice 가 짧다. 프로세스가
생성되더라도 기존의 프로세스가 모두 끝날 때까지 기다린다.

# Process Status

- [13.2. Processes](https://diveintosystems.org/book/C13-OS/processes.html)

Process 는 Scheduler 에 의해 다음과 같이 상태가 변화한다. Blocked 은 I/O 처리를 위해 잠을 자야하는 상태이다.

![](img/2024-01-07-16-44-14.png)

# Procedure And Stack

쓰레드가 태어나면 Virtual Memory 에 stack 을 위한 공간이 마련된다. 함수가 호출되면 그 함수의 parameter 들이 오른쪽에서 왼쪽으로 stack 에 저장된다. 이후 return address, old EBP 등이 stack 에 저장된다. 

![](stack_frame.png)

함수가 호출될 때 parameter 들을 어떻게 처리하는지에 대한 규약을 calling convention 이라고 하고 `__cdecl, __stdcall, __fastcall` 등이 있다. `__cdecl` 은 함수를 호출한 쪽에서 parameter 들을 해제한다. `__stdcall` 은 호출된 함수 쪽에서 parameter 들을 해제한다. `__fastcall` 은 두개까지의 parameter 들은 ECX, EDX 레지스터에 저장하고 호출된 함수 쪽에서 parameter 들을 해제한다.

compiler 는 linker 에게 산출물을 전달할 때 함수, 변수 등의 이름을 일정한 규칙을 가지고 변경하게 되는데 이것을 name mangling 혹은 name decoration 이라고 한다. 일반적으로 함수의 이름, 파라미터 타입, calling convention 등을 사용하여 이름을 만들어 낸다. name mangling 은 compiler 마다 다르기 때문에 각 메뉴얼을 참고하자.

다음은 다양한 calling convention 들을 비교한 것이다.

| Segment word size | Calling Convention      | Parameters in registers | Parameter order on stack | Stack cleanup by |
| ----------------- | ----------------------- | ----------------------- | ------------------------ | ---------------- |
| 32bit             | `__cdecl`               |                         | `C`                      | `Caller`         |
|                   | `__stdcall`             |                         | `C`                      | `Callee`         |
|                   | `__fastcall`            | `ecx, edx`              | `C`                      | `Callee`         |
|                   | `__thiscall`            | `ecx`                   | `C`                      | `Callee`         |
| 64bit             | Windows(MS, Intel)      | `rcx/xmm0`              | `C`                      | `Caller`         |
|                   |                         | `rdx/xmm1`              | `C`                      | `Caller`         |
|                   |                         | `r8/xmm2`               | `C`                      | `Caller`         |
|                   |                         | `r9/xmm3`               | `C`                      | `Caller`         |
|                   | Linux, BSD (GNU, Intel) | `rdi, rsi`              | `C`                      | `Caller`         |
|                   |                         | `rdx, rcx, r8`          | `C`                      | `Caller`         |
|                   |                         | `r9, xmm0-7`            | `C`                      | `Caller`         |

# Process And Thread

윈도우즈의 유저레벨 프로세스는 다음과 같이 EPROCESS 구조체로 구현한다. [참고](https://www.nirsoft.net/kernel_struct/vista/EPROCESS.html)

```c	
struct EPROCESS
typedef struct _EPROCESS
{
     KPROCESS Pcb;
     EX_PUSH_LOCK ProcessLock;
     LARGE_INTEGER CreateTime;
     LARGE_INTEGER ExitTime;
     EX_RUNDOWN_REF RundownProtect;
     PVOID UniqueProcessId;
     LIST_ENTRY ActiveProcessLinks;
     ULONG QuotaUsage[3];
     ULONG QuotaPeak[3];
     ULONG CommitCharge;
     ULONG PeakVirtualSize;
     ULONG VirtualSize;
     LIST_ENTRY SessionProcessLinks;
     PVOID DebugPort;
     union
     {
          PVOID ExceptionPortData;
          ULONG ExceptionPortValue;
          ULONG ExceptionPortState: 3;
     };
     PHANDLE_TABLE ObjectTable;
     EX_FAST_REF Token;
     ULONG WorkingSetPage;
     EX_PUSH_LOCK AddressCreationLock;
     PETHREAD RotateInProgress;
     PETHREAD ForkInProgress;
     ULONG HardwareTrigger;
     PMM_AVL_TABLE PhysicalVadRoot;
     PVOID CloneRoot;
     ULONG NumberOfPrivatePages;
     ULONG NumberOfLockedPages;
     PVOID Win32Process;
     PEJOB Job;
     PVOID SectionObject;
     PVOID SectionBaseAddress;
     _EPROCESS_QUOTA_BLOCK * QuotaBlock;
     _PAGEFAULT_HISTORY * WorkingSetWatch;
     PVOID Win32WindowStation;
     PVOID InheritedFromUniqueProcessId;
     PVOID LdtInformation;
     PVOID VadFreeHint;
     PVOID VdmObjects;
     PVOID DeviceMap;
     PVOID EtwDataSource;
     PVOID FreeTebHint;
     union
     {
          HARDWARE_PTE PageDirectoryPte;
          UINT64 Filler;
     };
     PVOID Session;
     UCHAR ImageFileName[16];
     LIST_ENTRY JobLinks;
     PVOID LockedPagesList;
     LIST_ENTRY ThreadListHead;
     PVOID SecurityPort;
     PVOID PaeTop;
     ULONG ActiveThreads;
     ULONG ImagePathHash;
     ULONG DefaultHardErrorProcessing;
     LONG LastThreadExitStatus;
     PPEB Peb;
     EX_FAST_REF PrefetchTrace;
     LARGE_INTEGER ReadOperationCount;
     LARGE_INTEGER WriteOperationCount;
     LARGE_INTEGER OtherOperationCount;
     LARGE_INTEGER ReadTransferCount;
     LARGE_INTEGER WriteTransferCount;
     LARGE_INTEGER OtherTransferCount;
     ULONG CommitChargeLimit;
     ULONG CommitChargePeak;
     PVOID AweInfo;
     SE_AUDIT_PROCESS_CREATION_INFO SeAuditProcessCreationInfo;
     MMSUPPORT Vm;
     LIST_ENTRY MmProcessLinks;
     ULONG ModifiedPageCount;
     ULONG Flags2;
     ULONG JobNotReallyActive: 1;
     ULONG AccountingFolded: 1;
     ULONG NewProcessReported: 1;
     ULONG ExitProcessReported: 1;
     ULONG ReportCommitChanges: 1;
     ULONG LastReportMemory: 1;
     ULONG ReportPhysicalPageChanges: 1;
     ULONG HandleTableRundown: 1;
     ULONG NeedsHandleRundown: 1;
     ULONG RefTraceEnabled: 1;
     ULONG NumaAware: 1;
     ULONG ProtectedProcess: 1;
     ULONG DefaultPagePriority: 3;
     ULONG PrimaryTokenFrozen: 1;
     ULONG ProcessVerifierTarget: 1;
     ULONG StackRandomizationDisabled: 1;
     ULONG Flags;
     ULONG CreateReported: 1;
     ULONG NoDebugInherit: 1;
     ULONG ProcessExiting: 1;
     ULONG ProcessDelete: 1;
     ULONG Wow64SplitPages: 1;
     ULONG VmDeleted: 1;
     ULONG OutswapEnabled: 1;
     ULONG Outswapped: 1;
     ULONG ForkFailed: 1;
     ULONG Wow64VaSpace4Gb: 1;
     ULONG AddressSpaceInitialized: 2;
     ULONG SetTimerResolution: 1;
     ULONG BreakOnTermination: 1;
     ULONG DeprioritizeViews: 1;
     ULONG WriteWatch: 1;
     ULONG ProcessInSession: 1;
     ULONG OverrideAddressSpace: 1;
     ULONG HasAddressSpace: 1;
     ULONG LaunchPrefetched: 1;
     ULONG InjectInpageErrors: 1;
     ULONG VmTopDown: 1;
     ULONG ImageNotifyDone: 1;
     ULONG PdeUpdateNeeded: 1;
     ULONG VdmAllowed: 1;
     ULONG SmapAllowed: 1;
     ULONG ProcessInserted: 1;
     ULONG DefaultIoPriority: 3;
     ULONG SparePsFlags1: 2;
     LONG ExitStatus;
     WORD Spare7;
     union
     {
          struct
          {
               UCHAR SubSystemMinorVersion;
               UCHAR SubSystemMajorVersion;
          };
          WORD SubSystemVersion;
     };
     UCHAR PriorityClass;
     MM_AVL_TABLE VadRoot;
     ULONG Cookie;
     ALPC_PROCESS_CONTEXT AlpcContext;
} EPROCESS, *PEPROCESS;
```

EPROCESS의 중요한 멤버는 다음과 같다.

| member field                 | description                                                       |
| :--------------------------- | :---------------------------------------------------------------- |
| DirectoryTableBase           | 가상 메모리의 CR3 레지스터값                                      |
| LdtDescriptor                | 16비트 애플리케이션에서 사용되는 LDT 디스크립터                   |
| Int21Descriptor              | 16비트 애플리케이션에서 인터럽트 21의 디스크립터                  |
| IopmOffset                   | IO 허용 비트의 Offset                                             |
| Iopl                         | IO 특권레벨 (0일 경우 커널모드만 허용, 3일 경우 유저모드까지 허용 |
| ActiveProcessors             | 현재 활성화 되어있는 CPU 개수                                     |
| KernelTime                   | 이 프로세스가 커널레벨에서 소비한 시간 단위 개수                  |
| UserTime                     | 이 프로세스가 유저레벨에서 소비한 시간 단위 개수                  |
| ReadyListHead                | 현재 준비 상태에 있는 쓰레드의 리스트                             |
| SwapListEntry                | 현재 스와핑이 되고 있는 쓰레드의 리스트                           |
| ThreadListHead               | 이 프로세스가 가지고 있는 쓰레드 목록                             |
| ProcessLock                  | 이 프로세스가 접근시 사용하는 동기화 객체                         |
| Affinity                     | 멀티 코어 CPU 에서 이 프로세스의 Affinity                         |
| StackCount                   | 이 프로세스에서 사용하는 스택 개수                                |
| BasePriority                 | 프로세스 우선순위 (0~15)                                          |
| ThreadQuantum                | 이 프로세스에서 생성되는 쓰레드의 기본 퀀텀 값                    |
| CreateTime                   | 이 프로세스의 생성 시간                                           |
| UniqueProcessId              | 이 프로세스의 고유 아이디                                         |
| ActiveProcessLinks           | 모든 프로세스의 목록                                              |
| CommitCharge                 | 이 프로세스가 사용하는 물리 메모리 크기                           |
| PeakPagefileUsage            | 최고 페이지파일 사용 크기                                         |
| PeakVirtualSize              | 최고 가상 메모리 크기                                             |
| VirtualSize                  | 가상 메모리 크기                                                  |
| WorkingSetSize               | 이 프로세스의 워킹세트 크기                                       |
| DebugPort                    | 디버깅 상태일 때 LPC 포트                                         |
| Token                        | 프로세스의 토큰 정보                                              |
| WorkingSetLock               | Working Set 조정 시 사용되는 Lock                                 |
| WorkingSetPage               | Working Set에 의한 Page 개수                                      |
| AddressCreationLock          | 이 프로세스에서 메모리 생성시 사용되는 Lock                       |
| VadRoot                      | 유저 메모리 영역을 설명하는 VAD pointer                           |
| NumberOfPriatePages          | 이 프로세스의 프라이빗 페이지 개수                                |
| NumberOfLockedPages          | 락 되어진 페이지 개수                                             |
| Peb                          | Process Environment Block                                         |
| SectionBaseAddress           | 프로세스 세션 베이스 주소, 주로 이미지의 베이스 주소              |
| WorkingSetWatch              | 페이지 폴트 발생 시 저장되는 히스토리                             |
| Win32WindowStation           | 현재 실행되는 프로세스의 Window Station ID                        |
| InheritedFromUniqueProcessId | 부모 프로세스의 ID                                                |
| LdtInformation               | 이 프로세스의 LDT 정보를 지시                                     |
| VdmObjects                   | 16비트 프로그램일 때 사용됨                                       |
| DeviceMap                    | 이 프로세스에서 사용할 수 있는 DOS의 디바이스 맵                  |
| SessionId                    | 터미널 서비스의 세션 ID                                           |
| ImageFileName                | 이 프로세스의 이름                                                |
| PriorityClass                | 이 프로세스의 우선순위                                            |
| SubSystemMinorVersion        | 서브시스템의 마이너 버전                                          |
| SubSystemMajorVersion        | 서브시스템의 메이저 버전                                          |
| SubSystemVersion             | 서브시스템 버전                                                   |
| LockedPageList               | 이 페이지에서 락 되어진 페이지의 리스트                           |
| ReadOperationCount           | I/O Read 개수                                                     |
| WriteOperationCount          | I/O Write 개수                                                    |
| CommitChargeLimit            | 최대로 사용할 수 있는 물리 메모리 크기                            |
| CommitChargePeak             | 최대로 사용된 물리 메모리 크기                                    |

윈도우즈의 커널레벨 프로세스는 다음과 같이 KPROCESS 로 구현한다. [참고](https://www.nirsoft.net/kernel_struct/vista/KPROCESS.html)

```c
typedef struct _KPROCESS
{
     DISPATCHER_HEADER Header;
     LIST_ENTRY ProfileListHead;
     ULONG DirectoryTableBase;
     ULONG Unused0;
     KGDTENTRY LdtDescriptor;
     KIDTENTRY Int21Descriptor;
     WORD IopmOffset;
     UCHAR Iopl;
     UCHAR Unused;
     ULONG ActiveProcessors;
     ULONG KernelTime;
     ULONG UserTime;
     LIST_ENTRY ReadyListHead;
     SINGLE_LIST_ENTRY SwapListEntry;
     PVOID VdmTrapcHandler;
     LIST_ENTRY ThreadListHead;
     ULONG ProcessLock;
     ULONG Affinity;
     union
     {
          ULONG AutoAlignment: 1;
          ULONG DisableBoost: 1;
          ULONG DisableQuantum: 1;
          ULONG ReservedFlags: 29;
          LONG ProcessFlags;
     };
     CHAR BasePriority;
     CHAR QuantumReset;
     UCHAR State;
     UCHAR ThreadSeed;
     UCHAR PowerState;
     UCHAR IdealNode;
     UCHAR Visited;
     union
     {
          KEXECUTE_OPTIONS Flags;
          UCHAR ExecuteOptions;
     };
     ULONG StackCount;
     LIST_ENTRY ProcessListEntry;
     UINT64 CycleTime;
} KPROCESS, *PKPROCESS;
```

윈도우즈의 유저레벨 쓰레드는 다음과 같이 ETHREAD 구조체로 구현한다. [참고](https://www.nirsoft.net/kernel_struct/vista/ETHREAD.html)

```c
typedef struct _ETHREAD
{
     KTHREAD Tcb;
     LARGE_INTEGER CreateTime;
     union
     {
          LARGE_INTEGER ExitTime;
          LIST_ENTRY KeyedWaitChain;
     };
     union
     {
          LONG ExitStatus;
          PVOID OfsChain;
     };
     union
     {
          LIST_ENTRY PostBlockList;
          struct
          {
               PVOID ForwardLinkShadow;
               PVOID StartAddress;
          };
     };
     union
     {
          PTERMINATION_PORT TerminationPort;
          PETHREAD ReaperLink;
          PVOID KeyedWaitValue;
          PVOID Win32StartParameter;
     };
     ULONG ActiveTimerListLock;
     LIST_ENTRY ActiveTimerListHead;
     CLIENT_ID Cid;
     union
     {
          KSEMAPHORE KeyedWaitSemaphore;
          KSEMAPHORE AlpcWaitSemaphore;
     };
     PS_CLIENT_SECURITY_CONTEXT ClientSecurity;
     LIST_ENTRY IrpList;
     ULONG TopLevelIrp;
     PDEVICE_OBJECT DeviceToVerify;
     _PSP_RATE_APC * RateControlApc;
     PVOID Win32StartAddress;
     PVOID SparePtr0;
     LIST_ENTRY ThreadListEntry;
     EX_RUNDOWN_REF RundownProtect;
     EX_PUSH_LOCK ThreadLock;
     ULONG ReadClusterSize;
     LONG MmLockOrdering;
     ULONG CrossThreadFlags;
     ULONG Terminated: 1;
     ULONG ThreadInserted: 1;
     ULONG HideFromDebugger: 1;
     ULONG ActiveImpersonationInfo: 1;
     ULONG SystemThread: 1;
     ULONG HardErrorsAreDisabled: 1;
     ULONG BreakOnTermination: 1;
     ULONG SkipCreationMsg: 1;
     ULONG SkipTerminationMsg: 1;
     ULONG CopyTokenOnOpen: 1;
     ULONG ThreadIoPriority: 3;
     ULONG ThreadPagePriority: 3;
     ULONG RundownFail: 1;
     ULONG SameThreadPassiveFlags;
     ULONG ActiveExWorker: 1;
     ULONG ExWorkerCanWaitUser: 1;
     ULONG MemoryMaker: 1;
     ULONG ClonedThread: 1;
     ULONG KeyedEventInUse: 1;
     ULONG RateApcState: 2;
     ULONG SelfTerminate: 1;
     ULONG SameThreadApcFlags;
     ULONG Spare: 1;
     ULONG StartAddressInvalid: 1;
     ULONG EtwPageFaultCalloutActive: 1;
     ULONG OwnsProcessWorkingSetExclusive: 1;
     ULONG OwnsProcessWorkingSetShared: 1;
     ULONG OwnsSystemWorkingSetExclusive: 1;
     ULONG OwnsSystemWorkingSetShared: 1;
     ULONG OwnsSessionWorkingSetExclusive: 1;
     ULONG OwnsSessionWorkingSetShared: 1;
     ULONG OwnsProcessAddressSpaceExclusive: 1;
     ULONG OwnsProcessAddressSpaceShared: 1;
     ULONG SuppressSymbolLoad: 1;
     ULONG Prefetching: 1;
     ULONG OwnsDynamicMemoryShared: 1;
     ULONG OwnsChangeControlAreaExclusive: 1;
     ULONG OwnsChangeControlAreaShared: 1;
     ULONG PriorityRegionActive: 4;
     UCHAR CacheManagerActive;
     UCHAR DisablePageFaultClustering;
     UCHAR ActiveFaultCount;
     ULONG AlpcMessageId;
     union
     {
          PVOID AlpcMessage;
          ULONG AlpcReceiveAttributeSet;
     };
     LIST_ENTRY AlpcWaitListEntry;
     ULONG CacheManagerCount;
} ETHREAD, *PETHREAD;
```

ETHREAD 의 중요한 멤버는 다음과 같다.

| member field        | description                                                                                     |
| :------------------ | :---------------------------------------------------------------------------------------------- |
| InitialStack        | 커널 스택의 낮은 주소                                                                           |
| StackLimit          | 커널 스택의 높은 주소                                                                           |
| Kernel Stack        | 커널 모드에서 현재 스택 포인터 (ESP)                                                            |
| DebugActive         | 디버깅 중인가?                                                                                  |
| State               | 현재 쓰레드 상태                                                                                |
| Iopl                | IOPL                                                                                            |
| NpxState            | Floating Point 상태 정보                                                                        |
| Priority            | 우선순위                                                                                        |
| ContextSwitches     | 쓰레드 스위칭 횟수                                                                              |
| WaitIrql            | 현재 Wait 상태에서 IRQL                                                                         |
| WaitListEntry       | 현재 상태가 Wait인 쓰레드 목록                                                                  |
| BasePriority        | 이 쓰레드의 베이스 우선순위                                                                     |
| Quantum             | 이 쓰레드의 퀀컴 값                                                                             |
| ServiceTable        | 서비스 테이블                                                                                   |
| Affinity            | 커널에서의 쓰레드 Affinity                                                                      |
| Preempted           | 선점 여부                                                                                       |
| KernelStackResident | 쓰레드 커널 스택이 쓰레드 종료 후에도 메모리에 있는가                                           |
| NextProcessor       | 스케줄러에 의해 결정된 다음번 실행시 사용될 CPU                                                 |
| TrapFrame           | Exception 발생시 사용될 트랩 프레임 포인터                                                      |
| PreviousMode        | 이전의 모드가 유저모드인가 커널모드인가, 시스템 함수 호출에서 유효성을 체크하는데 사용되어진다. |
| KernelTime          | 커널모드에서 이 쓰레드가 수행된 시간                                                            |
| UserTime            | 유저모드에서 이 쓰레드가 수행된 시간                                                            |
| Alertable           | Alertable 상태                                                                                  |
| StackBase           | 이 쓰레드의 스택 베이스 주소                                                                    |
| ThreadListEntry     | 프로세서가 가지고 있는 모든 쓰레드들의 목록                                                     |
| CreateTime          | 생성시간                                                                                        |
| ExitTime            | 종료시간                                                                                        |
| ExitStatus          | exit status                                                                                     |
| PostBlockList       | 이 쓰레드가 참조하는 모든 Object들의 리스트                                                     |
| ActiveTimerListHead | 이 쓰레드에 활성화된 타이머 리스트                                                              |
| UniqueThread        | 이 쓰레드의 고유한 번호                                                                         |
| ImpersonationInfo   | 임퍼스네이션 정보                                                                               |

다음은 커널 쓰레드를 구현한 KTHREAD 이다. [참고](https://www.nirsoft.net/kernel_struct/vista/KTHREAD.html)

```c
typedef struct _KTHREAD
{
     DISPATCHER_HEADER Header;
     UINT64 CycleTime;
     ULONG HighCycleTime;
     UINT64 QuantumTarget;
     PVOID InitialStack;
     PVOID StackLimit;
     PVOID KernelStack;
     ULONG ThreadLock;
     union
     {
          KAPC_STATE ApcState;
          UCHAR ApcStateFill[23];
     };
     CHAR Priority;
     WORD NextProcessor;
     WORD DeferredProcessor;
     ULONG ApcQueueLock;
     ULONG ContextSwitches;
     UCHAR State;
     UCHAR NpxState;
     UCHAR WaitIrql;
     CHAR WaitMode;
     LONG WaitStatus;
     union
     {
          PKWAIT_BLOCK WaitBlockList;
          PKGATE GateObject;
     };
     union
     {
          ULONG KernelStackResident: 1;
          ULONG ReadyTransition: 1;
          ULONG ProcessReadyQueue: 1;
          ULONG WaitNext: 1;
          ULONG SystemAffinityActive: 1;
          ULONG Alertable: 1;
          ULONG GdiFlushActive: 1;
          ULONG Reserved: 25;
          LONG MiscFlags;
     };
     UCHAR WaitReason;
     UCHAR SwapBusy;
     UCHAR Alerted[2];
     union
     {
          LIST_ENTRY WaitListEntry;
          SINGLE_LIST_ENTRY SwapListEntry;
     };
     PKQUEUE Queue;
     ULONG WaitTime;
     union
     {
          struct
          {
               SHORT KernelApcDisable;
               SHORT SpecialApcDisable;
          };
          ULONG CombinedApcDisable;
     };
     PVOID Teb;
     union
     {
          KTIMER Timer;
          UCHAR TimerFill[40];
     };
     union
     {
          ULONG AutoAlignment: 1;
          ULONG DisableBoost: 1;
          ULONG EtwStackTraceApc1Inserted: 1;
          ULONG EtwStackTraceApc2Inserted: 1;
          ULONG CycleChargePending: 1;
          ULONG CalloutActive: 1;
          ULONG ApcQueueable: 1;
          ULONG EnableStackSwap: 1;
          ULONG GuiThread: 1;
          ULONG ReservedFlags: 23;
          LONG ThreadFlags;
     };
     union
     {
          KWAIT_BLOCK WaitBlock[4];
          struct
          {
               UCHAR WaitBlockFill0[23];
               UCHAR IdealProcessor;
          };
          struct
          {
               UCHAR WaitBlockFill1[47];
               CHAR PreviousMode;
          };
          struct
          {
               UCHAR WaitBlockFill2[71];
               UCHAR ResourceIndex;
          };
          UCHAR WaitBlockFill3[95];
     };
     UCHAR LargeStack;
     LIST_ENTRY QueueListEntry;
     PKTRAP_FRAME TrapFrame;
     PVOID FirstArgument;
     union
     {
          PVOID CallbackStack;
          ULONG CallbackDepth;
     };
     PVOID ServiceTable;
     UCHAR ApcStateIndex;
     CHAR BasePriority;
     CHAR PriorityDecrement;
     UCHAR Preempted;
     UCHAR AdjustReason;
     CHAR AdjustIncrement;
     UCHAR Spare01;
     CHAR Saturation;
     ULONG SystemCallNumber;
     ULONG Spare02;
     ULONG UserAffinity;
     PKPROCESS Process;
     ULONG Affinity;
     PKAPC_STATE ApcStatePointer[2];
     union
     {
          KAPC_STATE SavedApcState;
          UCHAR SavedApcStateFill[23];
     };
     CHAR FreezeCount;
     CHAR SuspendCount;
     UCHAR UserIdealProcessor;
     UCHAR Spare03;
     UCHAR Iopl;
     PVOID Win32Thread;
     PVOID StackBase;
     union
     {
          KAPC SuspendApc;
          struct
          {
               UCHAR SuspendApcFill0[1];
               CHAR Spare04;
          };
          struct
          {
               UCHAR SuspendApcFill1[3];
               UCHAR QuantumReset;
          };
          struct
          {
               UCHAR SuspendApcFill2[4];
               ULONG KernelTime;
          };
          struct
          {
               UCHAR SuspendApcFill3[36];
               PKPRCB WaitPrcb;
          };
          struct
          {
               UCHAR SuspendApcFill4[40];
               PVOID LegoData;
          };
          UCHAR SuspendApcFill5[47];
     };
     UCHAR PowerState;
     ULONG UserTime;
     union
     {
          KSEMAPHORE SuspendSemaphore;
          UCHAR SuspendSemaphorefill[20];
     };
     ULONG SListFaultCount;
     LIST_ENTRY ThreadListEntry;
     LIST_ENTRY MutantListHead;
     PVOID SListFaultAddress;
     PVOID MdlForLockedTeb;
} KTHREAD, *PKTHREAD;
```

# User Level Thread vs Kernel Level Thread

* [11장. 커널 레벨 쓰레드와 유저 레벨 쓰레드 | youtube](https://www.youtube.com/watch?v=sOt80Kw0Ols&list=PLVsNizTWUw7E2KrfnsyEjTqo-6uKiQoxc&index=30)
* [Lesson 3: 11. OS Protection Boundary](https://classroom.udacity.com/courses/ud923/lessons/3014898657/concepts/30606385900923)
* [Lesson 3: 12. OS System Call Flowchart](https://classroom.udacity.com/courses/ud923/lessons/3014898657/concepts/34183989490923)
* [Lesson 3: 13. Crossing the OS Boundary](https://classroom.udacity.com/courses/ud923/lessons/3014898657/concepts/34183989500923)

----

![](userlevelvskernellevelthread.png)

kernel level thread 는 kernel level 에서 scheduling 된다. 따라서 하나의 process 가 두개 이상의 kernel level thread 를 소유하고 있을 때 그 중 하나가 I/O block 되더라도 다른 thread 는 계속 실행할 수 있다. 또한 kernel 에서 직접 제공해주기 때문에 안전성과 기능의 다양성이 장점이다. 그러나 O/S 가 kernel level thread 를 context switching 하기 위해서는 user level 에서 kernel level 로 전환되야 하기 때문에 느리다. 

user level thread 는 user level 에서 scheduling 된다. kernel 은 user level thread 를 포함한 process 단위로 scheduling 한다. kernel 은 user level thread 를 알 수 없다. 따라서 user level thread 중 하나가 I/O 블록이 되면 kernel 은 그 thread 를 소유한 process 의 상태를 running 에서 ready 로 바꾼다. user level thread 는 context switching 될 때 O/S 가 user level 에서 kernel level 로 전환할 필요가 없다. 따라서 user level thread 는 context switching 이 kernel level thread 보다 빠르다.

multithreading model 은 user level thread 와 kernel level thread 의 mapping 방법에 따라 `1:1`, `N:1`, `N:M` 방법이 있다. c++ 의 pthread, JVM 은 `1:1` 이다. goroutine 은 `N:M` 이다. [참고](https://classroom.udacity.com/courses/ud923/lessons/3065538763/concepts/34341886380923)

Linux kernel 은 2.6 이전에 process 단위로 scheduling 되었다. [참고](https://en.wikipedia.org/wiki/Native_POSIX_Thread_Library). pthread 는 NPTL (Native Posix Thread Library) 이다. 따라서 `1:1 thread library` 이고 `pthread_create` 을 통해서 kernel level thread 를 만들어 낼 수 있다.

# Thread Scheduling

## Thread Status

![](Windows_thread_states-f4.14.png)

쓰레드가 프로세서를 선점하고 있을 때 다른 쓰레드가 우선순위에 의해 프로세서를 선점하는 스케줄링 방식을 preemptive scheduling 이라고 한다. 쓰레드가 프로세서를 선점하고 있는 동안 다른 쓰레드가 프로세서를 선점하지 못하는 스케줄링 방식을 Nonpreemptive Shceduling 이라고 한다. 윈도우즈는 기본적으로 preemptive shceduling 을 사용한다. 

윈도우즈는 선점형 스케줄링을 구현하기 위하여 타이머 인터럽트를 사용한다. 윈도우즈가 타이머 인터럽트를 받을 때 마다 실행되고 있는 쓰레드의 퀀텀을 감소시키고 이 값이 0 이하로 떨어지면 이 쓰레드의 할당 시간은 만료된 것으로 처리하고 DPC (Defered Procedure Call) 큐에 DPC 를 하나 삽입한다. IRQL (Interrupt Request Level) 이 DPC/dispatch level 이하로 떨어지면 DPC 인터럽트가 발생하고 윈도우즈의 Dispatcher 가 깨어나서 DPC 큐에서 DPC 를 하나 꺼내 실행한다. DPC 가 하나 실행될 때 마다 쓰레드가 실행되겠지?

![](irql.png)

# Thread synchronization

## Critical Section

유저레벨에서 간단히 사용할 수 있는 동기화 방법이다. 커널 객체를 바로 사용하지 않기 때문에 속도가 빠르다. 동일한 프로세스에서 사용 가능하다.

```cpp
// Global variable
CRITICAL_SECTION CriticalSection; 

int main( void )
{
    ...

    // Initialize the critical section one time only.
    if (!InitializeCriticalSectionAndSpinCount(&CriticalSection, 
        0x00000400) ) 
        return;
    ...

    // Release resources used by the critical section object.
    DeleteCriticalSection(&CriticalSection);
}

DWORD WINAPI ThreadProc( LPVOID lpParameter )
{
    ...

    // Request ownership of the critical section.
    EnterCriticalSection(&CriticalSection); 

    // Access the shared resource.

    // Release ownership of the critical section.
    LeaveCriticalSection(&CriticalSection);

    ...
return 1;
}
```

## Mutex

커널객체를 사용하는 동기방법중 하나이다. 동일한 프로세스에서 사용해야 하는 제한이 없다.

```cpp
#include <windows.h>
#include <stdio.h>

#define THREADCOUNT 2

HANDLE ghMutex; 

DWORD WINAPI WriteToDatabase( LPVOID );

int main( void )
{
    HANDLE aThread[THREADCOUNT];
    DWORD ThreadID;
    int i;

    // Create a mutex with no initial owner

    ghMutex = CreateMutex( 
        NULL,              // default security attributes
        FALSE,             // initially not owned
        NULL);             // unnamed mutex

    if (ghMutex == NULL) 
    {
        printf("CreateMutex error: %d\n", GetLastError());
        return 1;
    }

    // Create worker threads

    for( i=0; i < THREADCOUNT; i++ )
    {
        aThread[i] = CreateThread( 
                     NULL,       // default security attributes
                     0,          // default stack size
                     (LPTHREAD_START_ROUTINE) WriteToDatabase, 
                     NULL,       // no thread function arguments
                     0,          // default creation flags
                     &ThreadID); // receive thread identifier

        if( aThread[i] == NULL )
        {
            printf("CreateThread error: %d\n", GetLastError());
            return 1;
        }
    }

    // Wait for all threads to terminate

    WaitForMultipleObjects(THREADCOUNT, aThread, TRUE, INFINITE);

    // Close thread and mutex handles

    for( i=0; i < THREADCOUNT; i++ )
        CloseHandle(aThread[i]);

    CloseHandle(ghMutex);

    return 0;
}

DWORD WINAPI WriteToDatabase( LPVOID lpParam )
{ 
    // lpParam not used in this example
    UNREFERENCED_PARAMETER(lpParam);

    DWORD dwCount=0, dwWaitResult; 

    // Request ownership of mutex.

    while( dwCount < 20 )
    { 
        dwWaitResult = WaitForSingleObject( 
            ghMutex,    // handle to mutex
            INFINITE);  // no time-out interval
 
        switch (dwWaitResult) 
        {
            // The thread got ownership of the mutex
            case WAIT_OBJECT_0: 
                __try { 
                    // TODO: Write to the database
                    printf("Thread %d writing to database...\n", 
                            GetCurrentThreadId());
                    dwCount++;
                } 

                __finally { 
                    // Release ownership of the mutex object
                    if (! ReleaseMutex(ghMutex)) 
                    { 
                        // Handle error.
                    } 
                } 
                break; 

            // The thread got ownership of an abandoned mutex
            // The database is in an indeterminate state
            case WAIT_ABANDONED: 
                return FALSE; 
        }
    }
    return TRUE; 
}
```

## Semaphore

커널객체를 사용하는 동기방법중 하나이다. 동일한 프로세스에서 사용해야 하는 제한이 없다. Mutex 는 한번에 하나의 스레드만이 자원에 접근할 수 있지만 semaphore 는 지정한 개수만큼의 스레드가 자원에 접근할 수 있다.

```cpp
#include <windows.h>
#include <stdio.h>

#define MAX_SEM_COUNT 10
#define THREADCOUNT 12

HANDLE ghSemaphore;

DWORD WINAPI ThreadProc( LPVOID );

int main( void )
{
    HANDLE aThread[THREADCOUNT];
    DWORD ThreadID;
    int i;

    // Create a semaphore with initial and max counts of MAX_SEM_COUNT

    ghSemaphore = CreateSemaphore( 
        NULL,           // default security attributes
        MAX_SEM_COUNT,  // initial count
        MAX_SEM_COUNT,  // maximum count
        NULL);          // unnamed semaphore

    if (ghSemaphore == NULL) 
    {
        printf("CreateSemaphore error: %d\n", GetLastError());
        return 1;
    }

    // Create worker threads

    for( i=0; i < THREADCOUNT; i++ )
    {
        aThread[i] = CreateThread( 
                     NULL,       // default security attributes
                     0,          // default stack size
                     (LPTHREAD_START_ROUTINE) ThreadProc, 
                     NULL,       // no thread function arguments
                     0,          // default creation flags
                     &ThreadID); // receive thread identifier

        if( aThread[i] == NULL )
        {
            printf("CreateThread error: %d\n", GetLastError());
            return 1;
        }
    }

    // Wait for all threads to terminate

    WaitForMultipleObjects(THREADCOUNT, aThread, TRUE, INFINITE);

    // Close thread and semaphore handles

    for( i=0; i < THREADCOUNT; i++ )
        CloseHandle(aThread[i]);

    CloseHandle(ghSemaphore);

    return 0;
}

DWORD WINAPI ThreadProc( LPVOID lpParam )
{

    // lpParam not used in this example
    UNREFERENCED_PARAMETER(lpParam);

    DWORD dwWaitResult; 
    BOOL bContinue=TRUE;

    while(bContinue)
    {
        // Try to enter the semaphore gate.

        dwWaitResult = WaitForSingleObject( 
            ghSemaphore,   // handle to semaphore
            0L);           // zero-second time-out interval

        switch (dwWaitResult) 
        { 
            // The semaphore object was signaled.
            case WAIT_OBJECT_0: 
                // TODO: Perform task
                printf("Thread %d: wait succeeded\n", GetCurrentThreadId());
                bContinue=FALSE;            

                // Simulate thread spending time on task
                Sleep(5);

                // Release the semaphore when task is finished

                if (!ReleaseSemaphore( 
                        ghSemaphore,  // handle to semaphore
                        1,            // increase count by one
                        NULL) )       // not interested in previous count
                {
                    printf("ReleaseSemaphore error: %d\n", GetLastError());
                }
                break; 

            // The semaphore was nonsignaled, so a time-out occurred.
            case WAIT_TIMEOUT: 
                printf("Thread %d: wait timed out\n", GetCurrentThreadId());
                break; 
        }
    }
    return TRUE;
}
```

## Event

커널객체를 사용하는 동기화 방법중 하나이다. 동일한 프로세스에서 사용해야 하는 제한이 없다. 스레드가 시작되는 시점을 이벤트를 통해 제어한다.

```cpp
#include <windows.h>
#include <stdio.h>

#define THREADCOUNT 4 

HANDLE ghWriteEvent; 
HANDLE ghThreads[THREADCOUNT];

DWORD WINAPI ThreadProc(LPVOID);

void CreateEventsAndThreads(void) 
{
    int i; 
    DWORD dwThreadID; 

    // Create a manual-reset event object. The write thread sets this
    // object to the signaled state when it finishes writing to a 
    // shared buffer. 

    ghWriteEvent = CreateEvent( 
        NULL,               // default security attributes
        TRUE,               // manual-reset event
        FALSE,              // initial state is nonsignaled
        TEXT("WriteEvent")  // object name
        ); 

    if (ghWriteEvent == NULL) 
    { 
        printf("CreateEvent failed (%d)\n", GetLastError());
        return;
    }

    // Create multiple threads to read from the buffer.

    for(i = 0; i < THREADCOUNT; i++) 
    {
        // TODO: More complex scenarios may require use of a parameter
        //   to the thread procedure, such as an event per thread to  
        //   be used for synchronization.
        ghThreads[i] = CreateThread(
            NULL,              // default security
            0,                 // default stack size
            ThreadProc,        // name of the thread function
            NULL,              // no thread parameters
            0,                 // default startup flags
            &dwThreadID); 

        if (ghThreads[i] == NULL) 
        {
            printf("CreateThread failed (%d)\n", GetLastError());
            return;
        }
    }
}

void WriteToBuffer(VOID) 
{
    // TODO: Write to the shared buffer.
    
    printf("Main thread writing to the shared buffer...\n");

    // Set ghWriteEvent to signaled

    if (! SetEvent(ghWriteEvent) ) 
    {
        printf("SetEvent failed (%d)\n", GetLastError());
        return;
    }
}

void CloseEvents()
{
    // Close all event handles (currently, only one global handle).
    
    CloseHandle(ghWriteEvent);
}

int main( void )
{
    DWORD dwWaitResult;

    // TODO: Create the shared buffer

    // Create events and THREADCOUNT threads to read from the buffer

    CreateEventsAndThreads();

    // At this point, the reader threads have started and are most
    // likely waiting for the global event to be signaled. However, 
    // it is safe to write to the buffer because the event is a 
    // manual-reset event.
    
    WriteToBuffer();

    printf("Main thread waiting for threads to exit...\n");

    // The handle for each thread is signaled when the thread is
    // terminated.
    dwWaitResult = WaitForMultipleObjects(
        THREADCOUNT,   // number of handles in array
        ghThreads,     // array of thread handles
        TRUE,          // wait until all are signaled
        INFINITE);

    switch (dwWaitResult) 
    {
        // All thread objects were signaled
        case WAIT_OBJECT_0: 
            printf("All threads ended, cleaning up for application exit...\n");
            break;

        // An error occurred
        default: 
            printf("WaitForMultipleObjects failed (%d)\n", GetLastError());
            return 1;
    } 
            
    // Close the events to clean up

    CloseEvents();

    return 0;
}

DWORD WINAPI ThreadProc(LPVOID lpParam) 
{
    // lpParam not used in this example.
    UNREFERENCED_PARAMETER(lpParam);

    DWORD dwWaitResult;

    printf("Thread %d waiting for write event...\n", GetCurrentThreadId());
    
    dwWaitResult = WaitForSingleObject( 
        ghWriteEvent, // event handle
        INFINITE);    // indefinite wait

    switch (dwWaitResult) 
    {
        // Event object was signaled
        case WAIT_OBJECT_0: 
            //
            // TODO: Read from the shared buffer
            //
            printf("Thread %d reading from buffer\n", 
                   GetCurrentThreadId());
            break; 

        // An error occurred
        default: 
            printf("Wait error (%d)\n", GetLastError()); 
            return 0; 
    }

    // Now that we are done reading the buffer, we could use another
    // event to signal that this thread is no longer reading. This
    // example simply uses the thread handle for synchronization (the
    // handle is signaled when the thread terminates.)

    printf("Thread %d exiting\n", GetCurrentThreadId());
    return 1;
}

```

## Kernel Object

다음은 커널 동기화 오브젝트들이다. `KEVENT, KSEMAPHORE, KMUTANT` 모두 `DISPATCHER_HEADER` 를 멤버변수로 가지고 있다. `DISPATCHER_HEADER` 의 `WaitListHead` 를 이용하면 해당 동기화 객체에 대기하고 있는 쓰레드들의 목록을 얻어올 수 있다. `WaitListHead` 는 `KWAIT_BLOCK` 을 가리키고 `KWAIT_BLOCK` 의 `Thread` 는 `KTHREAD` 를 가리킨다.

```cpp
typedef struct _DISPATCHER_HEADER
{
     union
     {
          struct
          {
               UCHAR Type;
               union
               {
                    UCHAR Abandoned;
                    UCHAR Absolute;
                    UCHAR NpxIrql;
                    UCHAR Signalling;
               };
               union
               {
                    UCHAR Size;
                    UCHAR Hand;
               };
               union
               {
                    UCHAR Inserted;
                    UCHAR DebugActive;
                    UCHAR DpcActive;
               };
          };
          LONG Lock;
     };
     LONG SignalState;
     LIST_ENTRY WaitListHead;
} DISPATCHER_HEADER, *PDISPATCHER_HEADER;

typedef struct _KEVENT
{
     DISPATCHER_HEADER Header;
} KEVENT, *PKEVENT;

typedef struct _KSEMAPHORE
{
     DISPATCHER_HEADER Header;
     LONG Limit;
} KSEMAPHORE, *PKSEMAPHORE;

typedef struct _KMUTANT
{
     DISPATCHER_HEADER Header;
     LIST_ENTRY MutantListEntry;
     PKTHREAD OwnerThread;
     UCHAR Abandoned;
     UCHAR ApcDisable;
} KMUTANT, *PKMUTANT;
```

![](dispatcher_header.png)

# Memory Management

모든 프로세스들은 자신만의 독립적인 메모리 공간을 갖는다. 이것을 Virtual Memory 라 하고 Virtual Memory Address 에 의해 접근한다. Virtual Memory 는 4KB 단위로 분할하여 물리 메모리로 이동되어야 프로세스가 접근할 수 있다. 4KB 단위를 페이지라고 부른다.

CPU 는 instruction 을 실행할 때 virtual memory address 를 사용한다. 이때 이것을
physical memory address 로 전환해주는 일을 OS 의 
**MMU (Memory Management Unit)** 가 수행한다. 

가상 메모리의 페이지들중 물리메모리에 상주하는 것들을 working set 이라고 한다.

페이지는 `Free, Reserved, Commited` 와 같이 총 3가지 상태를 갖는다. 

Logical Address(Virtual Memory Address) 는 세그먼트 레지스터 (CS, DS, ES, SS, FS, GS) 의 visible part 인 segment selector(16bit) 와 offset(32bit) 으로 구성된다.

Logical Address 는 Segementation 을 통해서 Linear Address 로 변환된다. Linear Address 는 다시 Paging 을 통해서 Physical Address 로 변환되야 물리 메모리 접근이 가능하다.


![](address_tralsation_overview.png)

페이징이 도입되기 전에는 Linear Address 가 곧 Physical Address 였다. 페이징은 80386 부터 도입되었다.

![](80386_Addr_Mech.png)

# Segmentation

![](memory_registers.jpg)

GDT (Global Descriptor Table) 은 8byte 의 Segment Descriptor 들을 가지고 있는 자료구조이다. GDTR (Global Descriptor Table Register) 은 GDT를 가리킨다. 

LDT (Local Descriptor Table) 는 8byte 의 Segment Descriptor 들을 가지고 있는 자료구조이다. LDTR (Local Descriptor Table Register) 은 LDT를 가리킨다.

IDTR, TR (Task Register) 는 어디에 쓰는 걸까?

![](segment_selector.jpg)

Logical Address 는 16bit 의 Segment Selector 와 32bit 의 offset 으로 구성된다. 아래 그림과 같이 Segment Selector 는 여러 세그먼트 레지스터의 visible part 에 해당한다. hidden part 는 해당 세그먼트 레지스터의 Segment Selector 가 가리키는 Segment Descriptor Table Entry 의 일부값들이 저장된다.

![](segment_registers.jpg)

Segment Selector 의 상위 13bit 는 `2^13` 즉 `8192` 와 같다. Segment Selector 의 Index 는 13bit 이고 이것은 TI 가 0혹은 1일 때에 따라서 GDT 혹은 LDT 의 항목의 인덱스를 저장한다. GDTR 의 베이스 어드레스로부터, Segment Selctor 의 인덱스 x 8 만큼의 주소를 더하면 특정 세그먼트 디스크립터에 접근 가능하다.

RPL 은 0부터 3까지 특권레벨을 의미한다. 0은 커널레벨이고 3은 유저레벨이다.

![](GDT_LDT.jpg)

다음은 접근한 세그먼트 디스크립터의 자세한 내용이다.

![](segment_descriptor.jpg)

세그먼트 디스크립터의 Base Address(16bit) 와 Logical Address 의 offset(32bit) 을 더하면 Linear Address(32bit) 을 얻을 수 있다. 이렇게 만들어진 Linear Address 에서 어떻게 PDE, PTE, offset 을 얻을 수 있는 걸까? Segment Descriptor 의 Base Address 가 이미 PDE, PTE 를 포함하고 있는 건가?

![](logical_addr_to_linear_addr.jpg)

# Paging

Segmentation 과정을 통해서 만들어진 Linear Address 의 형태는 다음과 같다.

![](linearaddr_format.png)

CR3 는 Page Directory 를 가리킨다. Linear Address 의  `DIR` 는 Page Directory Entry 하나를 가리킨다. Page Directory Entry 는 Page Table 을 가리킨다. `PAGE` 는 Page Table Entry 하나를 가리킨다. Page Table Entry 는 Page Frame (4KB) 를 하나 가리킨다. `OFFSET` 은 Page Frame 의 특정 주소를 가리킨다.

![](page_translation.png)

CR3, PDE, PTE 의 비트별 세부내역은 생략한다. 

# Page Management

프로세서가 특정 프로세스 가상메모리의 페이지를 요청했을 때 그 페이지가 물리메모리에 없다면 페이지 폴트 예외가 발생한다. 이후 그 페이지는 디스크에서 물리메모리로 이동하는 데 이것을 페이지인이라고 한다.

대부분의 경우 한번 사용된 Page 의 근처 Page 들을 다시 참조하는 경향이 있는데 이것을 Locality 라고 한다. OS 가 Locality 때문에 특정 페이지를 Page-in 할 때 그 페이지 근처의 다른 페이지들도 함께 Page-in 하는 것을 Prepaging 이라고 한다. 

물리 메모리에 상주하는 페이지들을 Working Set 이라고 한다. 당장 작업할 수 있는 것들의 집합이라는 의미이다.

페이지는 LRU 혹은 FIFO 방식으로 교체한다. LRU 는 가장 최근에 사용한 페이지는 다시 사용할 가능성이 있으므로 덜 Page-Out 하는 방법이다.

프로세스가 잦은 Page fault Exception 때문에 Page-in, Page-out 을 하느라 CPU 이용률은 줄어들고 디스크 I/O 작업을 기다리는데 소비하는 시간이 많아지게 되면 시스템 전체가 성능 저하를 가져온다. 이러한 현상을 Thrashing (스레싱) 이라고 한다.

예를 들어서 한 시스템에 프로세스의 개수가 점점 더 많아진다면 프로세스당 사용할 수 있는 물리 메모리 공간이 줄어들게 되어 Thrashing 이 발생할 테고 대부분의 프로세스는 CPU 이용률이 줄어들고 디스크 Page-In, Page-out 을 하기 위해 DISK I/O 작업 대기 시간이 늘어날 것이다. CPU 이용률이 낮아 지기 때문에 시스템의 성능 저하를 가져온다.

# Processor Cache Management

cache(캐시)는 다음에 사용할 것을 미리 저장하여 저장된 값을 로드하는 시간을 아낄 수 있는 것이다. CPU 는 저장된 것의 성격에 따라 Instruction Cache 와 Data Cache 두 종류를 가지고 있고 Processor 로 부터 접근 거리에 따라 L1 Cache, L2 Cache 등을 가지고 있다. 

![](cpu_cache_overview.png)

cache 가 주소를 매핑하는 방식은 Direct Mapping, Associative Mapping, Set-Associative Mapping 과 같이 3가지 방법이 있다. 주로 Set-Associative Mapping 을 사용한다.

![](direct_mapping.png)

![](associative_mapping.png)

![](set_associative_mapping.png)

cache 는 주로 LRU(least recently used) 방식을 사용한다.

cache 는 line 단위로 정보를 저장하는데 line 의 크기는 Processor 에 따라 다르다. 주로 32B 이다. cache 의 line 크기를 고려하여 프로그래밍 하면 성능을 개선시킬 수 있다.

cache 의 쓰기정책은 정보를 메모리에 저장하는 방법은 시점에 따라 Write Through, Write Back 과 같이 두가지 종류가 있다. cache 의 정보가 변경될 때마다 메모리에 저장하는 방식이 Write Through 이고 cache 의 정보가 변경될 때마다 표기해 놓고 해당 line 이 cache 에서 제거될 때 메모리에 반영하는 방식이 Write Back 이다.

CR0 의 CD 는 Cache Disable 을 의미한다. 1 이면 cache 를 비활성화 하여 시스템의 성능이 저하된다. 디버깅을 할 때를 제외하고는 권장되지 않는다. NW 는 Non-Cache Write-Through 를 의미한다. 1 이면 Write Back 을 0 이면 Write Through 를 사용한다.

![](cache_cr0.png)

또한 CR3, PDE, PTE 의 비트들을 이용하여 페이지 단위로 cache 를 제어할 수 있다. PCD 는 Page Cache Disable 이다. PWT 는 Page Write Through 이다.

![](cache_cr3.png)

![](cache_pde_pte.png)

cache 의 속도는 processor 와 거리가 가까울 수록 빠르다.

![](cache_speed.png)

다음은 space localtity 를 이용하여 최적화를 수행한 예이다.

```cpp
int v[10][10];
void main() {
    int r = 0;
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            r += v[i][j];
            // r += v[j][i]; 
        }
    }
}
```

`r += [i][j]` 대신 `r += [j][i]` 를 사용했다면 연속적으로 저장된 메모리를 사용하는 것이 아니므로 성능이 저하된다.

다음은 미리 cache 에 로딩하여 최적화를 수행한 예이다. `PREFETCH0 [ecx]` 는 인텔 컴파일러만 지원하는 명령어이다.

```cpp
void __fastcall PREFETCH(void * p) {
    // PREFETCH0 [ecx]
    _asm _emit 0x0f;
    _asm _emit 0x18;
    _asm _emit 0x09;
}

int v[1615];
int Sum(int * p) {
    int r = 0;
    for (int i = 0; i < 16; ++i) {
        r += *(p + n);
    }
    return r;
}

void main() {
    int r;
    for (int i = 0; i < 100; ++i) {
        PREFETCH(v + (16 * (i + 1)));
        r = Sum(v + (16 * i));
    }
    r = Sum(v + (16 * 100));
}
```

다음은 데이터이 정렬을 활용하여 최적화를 수행한 예이다.

```cpp
typedef struct _PHONE_BOOK_ENTRY {
    _PHONE_BOOK_ENTRY* pnext;
    char Name[20];
    char Email[10];
    char Phone[16];
    //_PHONE_BOOK_ENTRY* pnext;
} PHONE_BOOK_ENTRY, *PPHONE_BOOK_ENTRY;

PHONE_BOOK_ENTRY SearchName(char* pname) {
    PHONE_BOOK_ENTRY phead;
    ...
    while (phead != NULL) {
        if (stricmp(pname, phead->Name) == 0) 
            return phead;
        phead = phead->Next;
    }
    return NULL;
}
```

만약 위의 코드에서 `pnext` 를 `Phone` 아래에 위치시키면 `if (stricmp(pname, phead->Name) == 0)` 에서 space locality 가 지켜지지 않기 때문에 성능이 저하된다. 즉 `_PHONE_BOOK_ENTRY` 의 `pnext` 와 `Name` 은 가까이 있어야 한다. 

TLB (Translate Lookaside Buffer) 는 virtual memory address 를 physical memory address 로 traslation 할 때 성능을 향상시키위해 존재하는 cache 이다.

![](cache_tlb.png)

# Windows Cache Management

윈도우즈는 하나의 파일을 메모리로 읽어 들일 때 캐시를 활용하여 성능을 향상한다.

![](cache_windows_read_file_overview.png)

file 의 내용은 

![](cache_system_cache.png)

# User mode, Kernel mode

* [커널 모드와 유저 모드](https://www.youtube.com/watch?v=4y5BgddMY7o&list=PLVsNizTWUw7E2KrfnsyEjTqo-6uKiQoxc&index=32&t=0s)
* [User Mode vs Kernel Mode](https://www.tutorialspoint.com/User-Mode-vs-Kernel-Mode)

----

application 이 실행하면 32bit OS 는 virtual memory 4 GB 를 할당한다. 2 GB 는 user space 이고 2 GB 는 kernel space 이다. user space 에는 application 의 instruction 들이 저장된다. kernel space 에는 OS 의 instruction 들이 저장된다.  정확하게 얘기하면 mapping 정보가 저장된다. 여러 application 들은 각각의 virtual memeory 의 kenrnel space 에 OS instruction 들을 중복해서 들고 있는 것이 아니고 OS instruction 들을 가리키고 있는 것이다. virtual memeory 에는 instruction 만 저장되는 것은 아니다. data 를 포함한 여러가지가 저장된다.

OS 가 특정 application 의 instruction 들을 하나씩 실행하는 경우를 생각해 보자. virtual memory 의 user space 의 instruction 들을 fetch, decode, execute 하다가 kernel space 의 instruction 들을 fetch, decode, execute 하고 다시 user space 의 instruction 들을 fetch, decode, execute 할 것이다. 이것을 OS 가 application 을 user mode, kernel mode, user mode 로 실행된다고 말한다. 이때 user space 의 instruction 에서 kernel space 를 접근할 수 있다면 얼마든지 system 을 엉망으로 만들 수 있다. 따라서 OS 는 user space 의 instruction 은 kernel space 를 접근할 수 없도록 통제해야한다. 그러나 kernel space 의 instruction 들은 user space 를 접근할 수 있다.

또한 OS 가 user mode, kernel mode 를 전환할 때 register 들을 바꿔치는 것을 포함해서 CPU 에게 상당한 부담이다. 앞서 언급한 Threading model 에서 `1:1` 의 경우 thread 가 context switching 이 될때 마다 user mode, kernel mode 의 전환이 필요하기 때문에 CPU 에 부담이 된다. `N:1` 의 경우는 process 의 context switching 이 될때만 user mode, kernel mode 의 전환이 필요하다. 따라서 user level thread 가 얼마든지 context switching 이 되더라도 빠르다. 그러나 process 의 thread 중 하나라도 I/O block 이 되는 경우 process 가 통째로 block 된다.

# Virtual Memory Control

virtual memory 는 page 단위로 physical memory 와 mapping 된다.
그리고 page 단위로 할당된다. virtual memory 는 page 단위로 `commit, free, reserve`
의 상태로 둘 수 있다. `commit` 은 physical memory 와 연결됨을 의미한다.
`free` 는 physical memory 와 연결되어 있지 않음을 의미한다. 메모리 단편화를
방지하기 위해 연속된 공간을 `reserve` 해 놓으면 caching 이 될 수 있기 때문에
효율적이다.

메모리 할당은 `malloc` 으로 할 수 있지만 Windows 의 경우 `VirtualAlloc` 을 사용하면
더욱 많은 기능을 이용할 수 있다. 메모리는 `Allocation Granularity Boundary` 배수를 
시작주소로 `page size` 배수 만큼씩 할당된다. 

다음과 같이 `GetSystemInfo(&si)` 를 이용하면 `Allocation Granularity Boundary, page size`
를 알 수 있다.

```c
GetSystemInfo(&si);
pageSize         = si.dwPageSize // 4k
allocGranularity = si.dwAllocationGranularity // 64k
```

예를 들어서 최초 메모리를 `VirtualAlloc` 을 이용해 `4k` 할당하면 
`64k` 부터 `4k` 가 할당된다. 그 다음 `8k` 를 할당하면 `128k`
부터 `8k` 가 할당된다.

# Heap Control

Windows 는 `default heap` 을 제공한다. 그러나 별도의 heap 을 사용하면
메모리 단편화 등등 장점이 있다. `HeapCreate, HeapDestroy` 로 additional heap
을 만들어서 사용해보자. 그러나 메모리 할당을 위해 `malloc, free` 대신 `HeapAlloc, HeapFree` 
를 사용해야 한다.

# MMF (Memory Mapped File)

Virtual memory 에 mapping 한 FILE 을 MMF (Memory Mapped File) 이라 한다.
mapping 된 Virtual memory 에 write 하면 mapping 된 File 에 쓰기가 된다.

아주 큰 파일의 내용을 sorting 한다고 해보자. 먼저 Memory 로 FILE 을 읽어 들이고
sorting 한 다음 다시 FILE 에 써야 한다. 이때 FILE I/O 가 발생한다. 그러나
MMF 를 사용하면 Virtual memory 의 내용을 sorting 하기만 하면 된다.

Windows 에서 다음과 같은 순서대로 MMF 를 생성한다.

```cpp
// Create file
HANDLE hFile = CreateFile(...);

// Create mapped file object
HANDLE hMapFile = CreateFileMapping(hFile, ...);

// Map the file to virtual memory
TCHAR* pWrite = (TCHAR*)MapViewOfFile(hMapFile, ...);
``` 

# DLL (Dynamic Link Library)

A.exe 와 B.exe 가 a.lib 을 static library link 를 했다면 A.exe 도 a.lib 을 가지고 있고 B.exe 도 a.lib 를 가지고 있다. A.exe 의 virtual memory 가 만들어지고 physical memory 에 paging in 될때 a.lib 역시 같이 포함된다. B.exe 의 virtual memory 가 만들어지고 physical memory 에 paging in 될 때 역시 a.lib 역시 같이 포함된다. 똑같은 a.lib 이지만 각각 paging in 된다. 비효율적이다.

A.exe 와 B.exe 가 a.dll 을 dynamic libary link 를 했다면 A.exe 의 virtual memory 가 만들어지고 physcial memory 에 paging in 될때 a.dll 가 어딘가에 paging in 되고 mapping 정보가 A.exe 의 virtual memory 에 저장된다. 이때 B.exe 의 virtual memory 가 만들어지고 physical memory 에 paging in 될때 a.dll 은 이미 어딘가에 paging in 되어 있고 mapping 정보가 B.exe 의 virtual memory 에 저장된다. 

A.exe 에서 B.exe 로 process context switching 이 발생해도 a.dll 은 physical memory 혹은 swap disk 에서 load 되어있고 unload 되지 않는다.

# Execution file and Loader

* [Portable Executable Format](/pef/README.md)
  * 윈도우즈의 실행파일 포맷

* [Elf](/elf/README.md)
  * 리눅스의 실행파일 포맷

# File System

* [유닉스 파일시스템과 i-node 구조체](https://jiming.tistory.com/359)

----

ext4 file system 의 [i-node structure](https://github.com/torvalds/linux/blob/d2f8825ab78e4c18686f3e1a756a30255bb00bf3/fs/ext4/ext4.h) 는 다음과 같다.

```c
/*
 * Structure of an inode on the disk
 */
struct ext4_inode {
	__le16	i_mode;		/* File mode */
	__le16	i_uid;		/* Low 16 bits of Owner Uid */
	__le32	i_size_lo;	/* Size in bytes */
	__le32	i_atime;	/* Access time */
	__le32	i_ctime;	/* Inode Change time */
	__le32	i_mtime;	/* Modification time */
	__le32	i_dtime;	/* Deletion Time */
	__le16	i_gid;		/* Low 16 bits of Group Id */
	__le16	i_links_count;	/* Links count */
	__le32	i_blocks_lo;	/* Blocks count */
	__le32	i_flags;	/* File flags */
	union {
		struct {
			__le32  l_i_version;
		} linux1;
		struct {
			__u32  h_i_translator;
		} hurd1;
		struct {
			__u32  m_i_reserved1;
		} masix1;
	} osd1;				/* OS dependent 1 */
	__le32	i_block[EXT4_N_BLOCKS];/* Pointers to blocks */
	__le32	i_generation;	/* File version (for NFS) */
	__le32	i_file_acl_lo;	/* File ACL */
	__le32	i_size_high;
	__le32	i_obso_faddr;	/* Obsoleted fragment address */
	union {
		struct {
			__le16	l_i_blocks_high; /* were l_i_reserved1 */
			__le16	l_i_file_acl_high;
			__le16	l_i_uid_high;	/* these 2 fields */
			__le16	l_i_gid_high;	/* were reserved2[0] */
			__le16	l_i_checksum_lo;/* crc32c(uuid+inum+inode) LE */
			__le16	l_i_reserved;
		} linux2;
		struct {
			__le16	h_i_reserved1;	/* Obsoleted fragment number/size which are removed in ext4 */
			__u16	h_i_mode_high;
			__u16	h_i_uid_high;
			__u16	h_i_gid_high;
			__u32	h_i_author;
		} hurd2;
		struct {
			__le16	h_i_reserved1;	/* Obsoleted fragment number/size which are removed in ext4 */
			__le16	m_i_file_acl_high;
			__u32	m_i_reserved2[2];
		} masix2;
	} osd2;				/* OS dependent 2 */
	__le16	i_extra_isize;
	__le16	i_checksum_hi;	/* crc32c(uuid+inum+inode) BE */
	__le32  i_ctime_extra;  /* extra Change time      (nsec << 2 | epoch) */
	__le32  i_mtime_extra;  /* extra Modification time(nsec << 2 | epoch) */
	__le32  i_atime_extra;  /* extra Access time      (nsec << 2 | epoch) */
	__le32  i_crtime;       /* File Creation time */
	__le32  i_crtime_extra; /* extra FileCreationtime (nsec << 2 | epoch) */
	__le32  i_version_hi;	/* high 32 bits for 64-bit version */
	__le32	i_projid;	/* Project ID */
};
```

`i_mode` 는 16 bit 로 다음과 같은 구조를 갖는다.

```
 bit:      4 1 1 1 1 1 1 1 1 1 1 1
desc: type u g s r w x r w x r w x
```

다음은 type (4 bit) 의 종류이다.

| ls 표기 | 종류       | Value    |
| ------- | ---------- | -------- |
| `-`     | 정규파일   | S_IFREG  |
| d       | 디렉터리   | S_IFDIR  |
| c       | 문자장치   | S_IFCHR  |
| b       | 블록장치   | S_IFBLK  |
| l       | 링크파일   | S_IFLNK  |
| p       | 파이프파일 | S_IFFIFO |
| s       | 소켓파일   | S_IFSOCK |

다음은 그 다음 3 bit `u g r` 의 내용이다.

|     | 종류       | Value | 내용 |
| --- | ---------- | ----- | ---- |
| u   | SETUID     | 4000  | EUID (유효 사용자 아이디)가 RUID (실행 사용자 아이디)에서 파일의 소유자 아이디로 변경된다.  |
| g   | SETGID     | 2000  | EGID (유효 그룹 아이디)가 RGID (실행 그룹 아이디)에서 파일의 소유 그룹 아이디로 변경된다. |
| s   | Sticky Bit | 1000  | file, directory permission handling ??? |

`EXT4_N_BLOCKS` 는 [linux/fs/ext4/ext4.h](https://github.com/torvalds/linux/blob/d2f8825ab78e4c18686f3e1a756a30255bb00bf3/fs/ext4/ext4.h) 에 다음과 같이 정의되어 있다.

```c
/*
 * Constants relative to the data blocks
 */
#define	EXT4_NDIR_BLOCKS		12
#define	EXT4_IND_BLOCK			EXT4_NDIR_BLOCKS
#define	EXT4_DIND_BLOCK			(EXT4_IND_BLOCK + 1)
#define	EXT4_TIND_BLOCK			(EXT4_DIND_BLOCK + 1)
#define	EXT4_N_BLOCKS			(EXT4_TIND_BLOCK + 1)
```

![](https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/Ext2-inode.svg/1024px-Ext2-inode.svg.png)

하나의 data block 의 크기는 page frame 과 동일하다. 일반적으로 4 KB 이다. direct block 는 12 개 뿐이다. 따라서 하나의 i-node 에서 direct block 만으로는 48 KB 밖에 저장할 수 없다.

single indirect block 을 사용하여 더욱 많은 data block 을 하나의 i-node 에 저장할 수 있다. 하나의 주소는 4 byte 라고 하자. 하나의 data block 이 4 KB 이므로 하나의 block 으로 1 K (1024) 개의 포인터를 저장할 수 있다. 

따라서 하나의 i-node 에서 single indirect block 으로 `4 KB * 1 K = 4 MB` 를 저장할 수 있다. double indirect block 으로는 `4 MB * 1 K = 4 GB` 를 저장할 수 있다. triple indirect block 으로는 `4 GB * 1 K = 4 TB` 를 저장할 수 있다.

결국 하나의 i-node 로 저장할 수 있는 data block 은 다음과 같다.

```
         direct blocks: 48 KB
single indirect blocks:  4 MB
double indirect blocks:  4 GB
triple indirect blocks:  4 TB ------------------------------------
          Total blocks:  4 TB
```

그러나 32-bit linux 에서는 4 GB 만 지원한다. linux kernel 의 file function 들이 사용하느 variable, argument 들이 32 bit 로 구현되어 있기 때문이다.

Hard Disk 의 하나의 sector 는 512 byte 이다. i-node 의 하나의 block 은 4 KB 일 때 이것은 8 개의 sector 에 대응한다.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Disk-structure2.svg/800px-Disk-structure2.svg.png)

ext4 file system 의 경우 하나의 파일 혹은 디렉토리는 `ext4_dir_entry` structure 로 다음과 같이 표현한다.

```c
/*
 * Structure of a directory entry
 */
#define EXT4_NAME_LEN 255

struct ext4_dir_entry {
	__le32	inode;			/* Inode number */
	__le16	rec_len;		/* Directory entry length */
	__le16	name_len;		/* Name length */
	char	name[EXT4_NAME_LEN];	/* File name */
};
```

`rec_len, name_len` 를 제외한 세부적인 정보는 `inode` 가 가리키는 `ext4_inode` structure 를 이용한다.

[ext4 disk layout](https://ext4.wiki.kernel.org/index.php/Ext4_Disk_Layout) 을 참고하면 하나의 disk 가 어떻게 구성되어 있는지 알 수 있다.
* [INODE STRUCTURE IN EXT4 FILESYSTEM](https://selvamvasu.wordpress.com/2014/08/01/inode-vs-ext4/)

![](https://selvamvasu.files.wordpress.com/2014/08/ext4.jpg)

# Journaling File System

저널링 파일시스템(**Journaling File System**)은 컴퓨터 시스템에서 파일 시스템의
안정성과 데이터 무결성을 보장하기 위한 기술 중 하나로, 변경 사항을 기록하는
저널이라는 로그를 사용하여 파일 시스템의 메타데이터를 관리한다.

파일 시스템에서 파일 및 디렉터리의 구성 정보나 권한 등을 나타내는 정보를 메타데이터라고 합니다. 저널링 파일 시스템에서 발생하는 모든 메타데이터 변경 사항은 먼저 저널에 기록된 후, 실제 파일 시스템에 적용된다.

저널링 파일 시스템의 주요 이점은 다음과 같다.

- 데이터 무결성 유지: 시스템의 갑작스러운 전원 장애나 충돌 등으로 인해 발생할 수
  있는 메타데이터의 손실과 데이터 무결성의 손상을 방지한다.
- 빠른 복구 시간: 전원이 복구되거나 시스템이 재시작될 때, 저널링 파일 시스템은
  저널에서 메타데이터의 변경 사항을 확인하여 파일 시스템 복구 과정을 빠르게
  수행한다.
- 데이터 일관성: 저널링 파일 시스템은 저널로의 변경 사항을 확인하는 동안
  적용하지 못한 변경 사항이 있으면 이를 처리하여 데이터 일관성을 유지한다.

주로 사용되는 저널링 파일 시스템의 예로는 Linux의 **Ext3**, **Ext4**, **XFS**, **ReiserFS** 등이 있으며, Windows의 **NTFS**와 macOS의 **HFS+**도 일종의 저널링 파일 시스템으로 간주된다. 이러한 파일 시스템들은 디스크에 저장된 데이터의 안정성과 무결성을 보장하며, 시스템의 복구 시간을 최소화하는데 도움을 준다.

# Quiz

* Thread vs. Process
* Context Switch
* Dining Philosophers
* Deadlock-Free Class
* Call In Order
* Synchronized Methods
* FizzBuzz
