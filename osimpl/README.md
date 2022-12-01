- [Abstract](#abstract)
- [Essentials](#essentials)
- [Materials](#materials)
- [Prerequisites](#prerequisites)
- [MikanOS on macosx](#mikanos-on-macosx)
  - [edk2](#edk2)
  - [qemu](#qemu)
  - [llvm](#llvm)
  - [etc](#etc)
  - [Hello World](#hello-world)

----

# Abstract

os 를 구현해보자. [fourbucks @ github](https://github.com/iamslash/fourbucks) 에서 실제로 구현해 본다.

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

# MikanOS on macosx

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

