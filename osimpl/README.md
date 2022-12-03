- [Abstract](#abstract)
- [Essentials](#essentials)
- [Materials](#materials)
- [Prerequisites](#prerequisites)
- [MikanOS on macos](#mikanos-on-macos)
  - [edk2](#edk2)
  - [qemu](#qemu)
  - [llvm](#llvm)
  - [etc](#etc)
  - [Hello World](#hello-world)
  - [Build Kernel](#build-kernel)
  - [Run Loader With Kernel](#run-loader-with-kernel)
  - [Draw Pixel On Kernel](#draw-pixel-on-kernel)
  - [Make Basic](#make-basic)
  - [Print Characters On Console](#print-characters-on-console)
  - [Mouse Input](#mouse-input)
  - [Final](#final)
  - [How To Boot Real Computer With USB](#how-to-boot-real-computer-with-usb)

----

# Abstract

os 를 구현해보자. 바탁부터 구현하는 것은 어려워서 UEFI 를 이용해서 제작해 보자. [0부터 시작하는 OS 자작 입문 [내가 만드는 OS 세계의 모든 것]](http://www.acornpub.co.kr/book/operating-system) 를 읽고 따라해본다.

[fourbucks @ github](https://github.com/iamslash/fourbucks) 에서 실제로 구현해 본다.

# Essentials

* [0부터 시작하는 OS 자작 입문 [내가 만드는 OS 세계의 모든 것]](http://www.acornpub.co.kr/book/operating-system)
  * [src](https://github.com/uchan-nos/mikanos)
  * [mikanos-build](https://github.com/uchan-nos/mikanos-build)
  * [os 구조와 원리 | hanbit](https://www.hanbit.co.kr/store/books/look.php?p_code=B9833754652) 를 계승한 책이다.
* [만들면서 배우는 OS 커널의 구조와 원리](http://www.hanbit.co.kr/store/books/look.php?p_code=B1271180320)
  * [src](https://github.com/iamroot-kernel-13th-x86/book_os_kernel_structure_principle)
* [C++로 나만의 운영체제 만들기](http://acornpub.co.kr/book/cplus-os-development)
  * visual studio 와 c++ 를 주로 사용하여 os 를 개발한다. 부트로더는 grub 를 사용함.
  * [src](https://github.com/pdpdds/SkyOS)
* [64비트 멀티코어 OS 원리와 구조](http://www.mint64os.pe.kr)
  * [ebook](http://www.yes24.com/Product/Goods/65061299?scode=032&OzSrank=1)
  * [src](https://github.com/kkamagui/mint64os-examples)

# Materials

* [os 구조와 원리 | hanbit](https://www.hanbit.co.kr/store/books/look.php?p_code=B9833754652)
  * [src](https://github.com/hide27k/haribote-os)
* [만들면서 배우는 OS 커널의 구조와 원리 해설 @ naver](https://blog.naver.com/hyuga777/80125530101)
  * 책을 잘 요약함
* [iamroot](http://www.iamroot.org/)
  * 커널 오프라인 스터디

# Prerequisites

* [NASM](https://www.nasm.us/)
* qemu
* Intel_Architecture_Software_Developers_Manual_Volume_3_Sys.pdf

# MikanOS on macos

## edk2

* [mikanos macosx build](https://qiita.com/yamoridon/items/4905765cc6e4f320c9b5)
  * Build & Run on macosx 
* [0부터 시작하는 os 자작 입문 [에러 및 해결]](http://www.kudryavka.me/?p=1056)
  * Trouble shooting edk2

```bash
$ cd ~/my/etc
$ git clone --recursive https://github.com/tianocore/edk2.git
$ cd edk2/BaseTools/Source/C
$ git checkout edk2-stable202111
$ make
```

## qemu

```bash
$ brew install qemu
```

## llvm

```bash
$ brew install llvm
$ export PATH=/usr/local/opt/llvm/bin:$PATH
```

## etc

```bash
$ brew install nasm dosfstools binutils
$ export PATH=/usr/local/opt/binutils/bin:$PATH
```

## Hello World

```bash
$ cd ~/my/cpp/
$ git clone https://github.com/uchan-nos/mikanos.git
$ cd mikanos
$ git checkout osbook_day02a
$ cd ~/my/c/edk2
$ ln -s $HOME/workspace/mikanos/MikanLoaderPkg .
$ source edksetup.sh
$ vim Conf/target.txt
ACTIVE_PLATFORM       = MikanLoaderPkg/MikanLoaderPkg.dsc
TARGET                = DEBUG
TARGET_ARCH           = X64
TOOL_CHAIN_CONF       = Conf/tools_def.txt
TOOL_CHAIN_TAG        = CLANGPDB
BUILD_RULE_CONF = Conf/build_rule.txt

$ vim /Users/david.s/my/c/edk2/MdePkg/Library/BaseLib/BaseLib.inf
#[LibraryClasses.X64, LibraryClasses.IA32]
#  RegisterFilterLib

$ build

$ ~/my/etc/mikanos-build/devenv/run_qemu.sh Build/MikanLoaderX64/DEBUG_CLANGPDB/X64/Loader.efi
```

## Build Kernel

```bash
$ cd ~/my/c/mikanos
$ git checkout osbook_day03a
$ cd kernel

$ clang++ -O2 -Wall -g --target=x86_64-elf -ffreestanding -mno-red-zone \
  -fno-exceptions -fno-rtti -std=c++17 -c main.cpp
$ ld.lld --entry KernelMain -z norelro --image-base 0x100000 --static \
  -o kernel.elf main.o
```

## Run Loader With Kernel

```bash
$ cd ~/my/etc/mikanos-buld/edk2
$ build
$ ~/my/etc/mikanos-build/devenv/run_qemu.sh \
    Build/MikanLoaderX64/DEBUG_CLANGPDB/X64/Loader.efi \
    ~/my/cpp/mikanos/kernel/kernel.elf
```

## Draw Pixel On Kernel

* [mikanos macosx build](https://qiita.com/yamoridon/items/4905765cc6e4f320c9b5)
  * Build & Run on macosx 

We need to update some code for macos

```c
// ~/my/cpp/mikanos/MikanLoaderPkg/Main.c
// AsIs
typedef void EntryPointType(UINT64, UINT64);
// ToBe
typedef void __attribute__((sysv_abi)) EntryPointType(UINT64, UINT64);
```

We need to change link commandline

```bash
# AsIs
$ ld.lld $LDFLAGS --entry KernelMain -z norelro --image-base 0x100000 --static -o kernel.elf main.o
# ToBe
$ ld.lld $LDFLAGS --entry=KernelMain -z norelro --image-base 0x100000 --static -o kernel.elf -z separate-code main.o
```

```bash
$ ln -s ~/my/etc/mikanos-build/ ~/osbook
$ vim ~/my/etc/mikanos-build/devenv/buildenv.sh
BASEDIR="$HOME/my/etc/mikanos-build/devenv/x86_64-elf"
EDK2DIR="$HOME/my/c/edk2"

$ source ~/my/etc/mikanos-build/devenv/buildenv.sh
$ echo $CPPFLAGS
$ echo $LDFLAGS

$ cd ~/my/cpp/mikanos/kernel
$ git checkout osbook_day03c
$ ~/my/cpp/mikanos/MikanLoaderPkg/Main.c

$ clang++ $CPPFLAGS -O2 -Wall -g --target=x86_64-elf -ffreestanding -mno-red-zone \
  -fno-exceptions -fno-rtti -std=c++17 -c main.cpp
#$ clang++ -I/Users/david.s/my/etc/mikanos-build/devenv/x86_64-elf/include/c++/v1 -I/Users/david.s/my/etc/mikanos-build/devenv/x86_64-elf/include -I/Users/david.s/my/etc/mikanos-build/devenv/x86_64-elf/include/freetype2     -I/Users/david.s/my/c/edk2/MdePkg/Include -I/Users/david.s/my/c/edk2/MdePkg/Include/X64     -nostdlibinc -D__ELF__ -D_LDBL_EQ_DBL -D_GNU_SOURCE -D_POSIX_TIMERS     -DEFIAPI='__attribute__((ms_abi))' -O2 -Wall -g --target=x86_64-elf -ffreestanding -mno-red-zone -fno-exceptions -fno-rtti -std=c++17 -c main.cpp
$ ld.lld $LDFLAGS --entry=KernelMain -z norelro --image-base 0x100000 --static -o kernel.elf -z separate-code main.o

# Build and run edk2
$ cd ~/my/c/edk2
$ source edksetup.sh 
$ build
$ ~/my/etc/mikanos-build/devenv/run_qemu.sh \
    Build/MikanLoaderX64/DEBUG_CLANGPDB/X64/Loader.efi \
    ~/my/cpp/mikanos/kernel/kernel.elf
```

## Make Basic

We need to update Makefile for macos

```bash
# Build kernel
$ cd ~/my/cpp/mikanos
$ git checkout osbook_day04a

$ vim kernel/Makefile # Add -z separate-code
LDFLAGS += --entry KernelMain -z norelro --image-base 0x100000 --static -z separate-code

$ vim MikanLoaderPkg/Main.c
//typedef void EntryPointType(UINT64, UINT64);
typedef void __attribute__((sysv_abi)) EntryPointType(UINT64, UINT64);

$ source ~/my/etc/mikanos-build/devenv/buildenv.sh
$ make

# Build edk2
$ cd ~/my/c/edk2
$ source edksetup.sh 
$ build

# Run BootLoader and Kernel
$ ~/my/etc/mikanos-build/devenv/run_qemu.sh \
    Build/MikanLoaderX64/DEBUG_CLANGPDB/X64/Loader.efi \
    ~/my/cpp/mikanos/kernel/kernel.elf
```

## Print Characters On Console

```bash
# Build kernel
$ cd ~/my/cpp/mikanos
$ git checkout osbook_day05f

$ vim kernel/Makefile # Add -z separate-code
LDFLAGS += --entry KernelMain -z norelro --image-base 0x100000 --static -z separate-code

$ vim kernel/Makefile # Update sed for macos sed
    sed -I '' -e 's|$(notdir $(OBJ))|$(OBJ)|' $@

$ vim MikanLoaderPkg/Main.c
//typedef void EntryPointType(const struct FrameBufferConfig*);
typedef void __attribute__((sysv_abi)) EntryPointType(const struct FrameBufferConfig*);

$ source ~/my/etc/mikanos-build/devenv/buildenv.sh
$ make

# Build edk2
$ cd ~/my/c/edk2
$ source edksetup.sh 
$ build

# Run BootLoader and Kernel
$ ~/my/etc/mikanos-build/devenv/run_qemu.sh \
    Build/MikanLoaderX64/DEBUG_CLANGPDB/X64/Loader.efi \
    ~/my/cpp/mikanos/kernel/kernel.elf
```

## Mouse Input

```bash
# Build kernel
$ cd ~/my/cpp/mikanos
$ git checkout osbook_day06c

$ vim kernel/Makefile # Add -z separate-code
LDFLAGS += --entry KernelMain -z norelro --image-base 0x100000 --static -z separate-code

$ vim kernel/Makefile # Update sed for macos sed
    sed -I '' -e 's|$(notdir $(OBJ))|$(OBJ)|' $@

$ vim MikanLoaderPkg/Main.c
//typedef void EntryPointType(const struct FrameBufferConfig*);
typedef void __attribute__((sysv_abi)) EntryPointType(const struct FrameBufferConfig*);

$ source ~/my/etc/mikanos-build/devenv/buildenv.sh
$ make clean && make

# Build edk2
$ cd ~/my/c/edk2
$ source edksetup.sh 
$ build clean && build

# Run BootLoader and Kernel
$ ~/my/etc/mikanos-build/devenv/run_qemu.sh \
    Build/MikanLoaderX64/DEBUG_CLANGPDB/X64/Loader.efi \
    ~/my/cpp/mikanos/kernel/kernel.elf
```

## Final

```bash
# Build kernel
$ cd ~/my/cpp/mikanos
$ git checkout master

$ vim kernel/Makefile # Add -z separate-code
LDFLAGS += --entry KernelMain -z norelro --image-base 0x100000 --static -z separate-code

$ vim kernel/Makefile # Update sed for macos sed
    sed -I '' -e 's|$(notdir $(OBJ))|$(OBJ)|' $@

$ vim kernel/main.cpp # Add this for macos
extern "C" void __cxa_pure_virtual() {
  while (1) __asm__("hlt");
}

$ vim MikanLoaderPkg/Main.c
//  typedef void EntryPointType(const struct FrameBufferConfig*,
//                              const struct MemoryMap*,
//                              const VOID*,
//                              VOID*,
//                              EFI_RUNTIME_SERVICES*);
  typedef void __attribute__((sysv_abi)) EntryPointType(const struct FrameBufferConfig*,
                                                        const struct MemoryMap*,
                                                        const VOID*,
                                                        VOID&,
                                                        EFI_RUNTIME_SERVICES*);

$ source ~/my/etc/mikanos-build/devenv/buildenv.sh
$ make clean && make

# Build edk2
$ cd ~/my/c/edk2
$ source edksetup.sh 
$ build clean && build

# Run BootLoader and Kernel
$ ~/my/etc/mikanos-build/devenv/run_qemu.sh \
    Build/MikanLoaderX64/DEBUG_CLANGPDB/X64/Loader.efi \
    ~/my/cpp/mikanos/kernel/kernel.elf
```

## How To Boot Real Computer With USB

Copy files to USB like these. `BOOTX64.EFI` is same with `Build/MikanLoaderX64/DEBUG_CLANGPDB/X64/Loader.efi`

```
/kernel.elf
/EFI/BOOT/BOOTX64.EFI
```

Turn on computer inserting USB. It works. (2022.12.03)
