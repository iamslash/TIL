# ELF (Executable and Linkable Format)

Linux/Unix 실행 파일의 내부 구조를 이해하고 실무에서 활용하는 방법을 설명합니다.

## 목차

- [Overview](#overview)
- [ELF 구조](#elf-구조)
- [Headers](#headers)
- [Dynamic Linking](#dynamic-linking)
- [ELF Loading 과정](#elf-loading-과정)
- [보안 기능](#보안-기능)
- [실무 활용](#실무-활용)
- [유용한 도구](#유용한-도구)
- [트러블슈팅](#트러블슈팅)
- [참고 자료](#참고-자료)

----

## Overview

### ELF란?

ELF(Executable and Linkable Format)는 Linux/Unix에서 실행 파일의 표준 포맷입니다.
- 실행 파일 (executable)
- 공유 라이브러리 (`.so` - Shared Object)
- 오브젝트 파일 (`.o` - Object file)
- 코어 덤프 (core dump)

### 실무에서 왜 필요한가?

**디버깅 시나리오:**
```
프로그램이 0x00401234 주소에서 Segmentation Fault 발생

ELF 구조를 알면:
→ 0x00401234가 어느 섹션인지 파악
→ .text 섹션(코드)인지 .data 섹션(데이터)인지 확인
→ 심볼 테이블로 함수 이름 추적
```

**성능 최적화:**
- 공유 라이브러리의 PLT/GOT 호출 오버헤드 이해
- Strip으로 불필요한 심볼 제거하여 파일 크기 감소
- 메모리 정렬 최적화

**보안 분석:**
- PIE, RELRO, Stack Canary 등 보호 기능 확인
- GOT overwrite 공격 벡터 이해
- 동적 링커 동작 방식 파악

**배포 문제 해결:**
- ldd로 필요한 공유 라이브러리 파악
- RPATH/RUNPATH 설정으로 라이브러리 경로 관리
- 정적 링킹 vs 동적 링킹 선택

### File vs Memory

ELF 파일은 disk와 memory에서 다른 형태로 존재합니다:

| Aspect | File (Disk) | Memory (Loaded) |
|--------|-------------|-----------------|
| 단위 | Section (linking 관점) | Segment (loading 관점) |
| 헤더 | Section Headers | Program Headers |
| .bss 섹션 | 파일에 없음 | 메모리에 할당됨 |
| GOT/PLT | 초기값 | 함수 주소로 채워짐 |
| 주소 | File offset | Virtual Address |

**왜 중요한가:**
- Section: 링커가 사용 (컴파일 타임)
- Segment: 로더가 사용 (런타임)
- 디버거는 VA를 표시, objdump는 파일 오프셋을 표시

----

## ELF 구조

### High-level 구조

```
┌─────────────────────┐
│   ELF Header        │ ← Magic number, 아키텍처, Entry point
├─────────────────────┤
│  Program Headers    │ ← 로더가 사용 (Segments)
│  (optional)         │   • LOAD, DYNAMIC, INTERP 등
├─────────────────────┤
│   .text Section     │ ← 실행 코드
├─────────────────────┤
│   .rodata Section   │ ← 읽기 전용 데이터
├─────────────────────┤
│   .data Section     │ ← 초기화된 전역 변수
├─────────────────────┤
│   .bss Section      │ ← 초기화 안된 전역 변수 (파일에 없음)
├─────────────────────┤
│   .plt Section      │ ← Procedure Linkage Table
├─────────────────────┤
│   .got Section      │ ← Global Offset Table
├─────────────────────┤
│   .dynamic Section  │ ← 동적 링킹 정보
├─────────────────────┤
│   .symtab Section   │ ← 심볼 테이블
├─────────────────────┤
│   .strtab Section   │ ← 문자열 테이블
├─────────────────────┤
│  Section Headers    │ ← 링커가 사용 (Sections)
└─────────────────────┘
```

### 주요 섹션

| 섹션 | 용도 | 권한 | 비고 |
|------|------|------|------|
| `.text` | 실행 코드 | R-X | Read, Execute |
| `.rodata` | 읽기 전용 데이터 (상수) | R-- | Read only |
| `.data` | 초기화된 전역 변수 | RW- | Read, Write |
| `.bss` | 초기화 안된 전역 변수 | RW- | 파일에 없음 (0으로 초기화) |
| `.plt` | Procedure Linkage Table | R-X | 함수 호출 트램폴린 |
| `.got` | Global Offset Table | RW- | 함수/변수 주소 저장 |
| `.got.plt` | PLT용 GOT | RW- | Lazy binding 사용 |
| `.dynamic` | 동적 링킹 정보 | R-- | 필요한 라이브러리 목록 |
| `.dynsym` | 동적 심볼 테이블 | R-- | 런타임 심볼 해석 |
| `.symtab` | 전체 심볼 테이블 | -- | strip으로 제거 가능 |
| `.strtab` | 문자열 테이블 | -- | 심볼 이름 저장 |
| `.rel.text` | 재배치 정보 | -- | 링커가 사용 |

### 주소 변환

ELF 파일에서 사용되는 주소 표현 방식:

**VA (Virtual Address):**
- 메모리의 가상 주소
- 예: `VA = 0x00401234`

**File Offset:**
- 파일에서의 위치 (바이트 오프셋)
- 예: `Offset = 0x1234`

#### 주소 변환 공식

**VA → File Offset:**
```
1. 올바른 세그먼트/섹션 찾기
   p_vaddr ≤ VA < p_vaddr + p_memsz

2. 오프셋 계산
   Offset = VA - p_vaddr + p_offset
```

#### 실전 예제

`/bin/ls`의 `.text` 섹션 정보:
```
VirtAddr  = 0x00401000  (메모리에서 섹션 시작)
FileOff   = 0x00001000  (파일에서 섹션 시작)
Size      = 0x00010000
```

**문제 1: VA → File Offset 변환**

`VA = 0x00401234` 주소가 파일의 어느 위치에 있는가?

```
1. 올바른 섹션 찾기
   0x00401000 ≤ 0x00401234 < 0x00401000 + 0x10000
   → .text 섹션에 속함 ✓

2. 섹션 내 오프셋 계산
   Offset_in_section = 0x00401234 - 0x00401000 = 0x234

3. 파일 오프셋 계산
   FileOff = 0x234 + 0x1000 = 0x1234

답: 파일의 0x1234 위치
```

**문제 2: File Offset → VA 변환**

파일의 `Offset = 0x1234` 위치가 메모리의 어느 주소인가?

```
1. 올바른 섹션 찾기
   0x1000 ≤ 0x1234 < 0x1000 + 0x10000
   → .text 섹션에 속함 ✓

2. 섹션 내 오프셋 계산
   Offset_in_section = 0x1234 - 0x1000 = 0x234

3. VA 계산
   VA = 0x234 + 0x00401000 = 0x00401234

답: VA = 0x00401234
```

#### 베이스 주소 (Base Address)

**PIE (Position Independent Executable) 비활성화:**
- 실행 파일은 고정 주소에 로드 (보통 0x00400000)
- 공유 라이브러리만 랜덤 주소에 로드

**PIE 활성화:**
- 실행 파일도 랜덤 주소에 로드 (ASLR)
- 모든 주소는 베이스 주소를 기준으로 재계산

```bash
# PIE 확인
readelf -h ./app | grep Type
# Type: EXEC (비활성화) 또는 DYN (활성화)

# 런타임 베이스 주소 확인
cat /proc/$(pidof app)/maps
# 출력:
# 00400000-00401000 r-xp ... /path/to/app  (PIE 비활성화)
# 55f8a2c00000-... r-xp ... /path/to/app  (PIE 활성화, 랜덤)
```

----

## Headers

### ELF Header

**실무 필수 필드:**

```c
typedef struct {
  unsigned char e_ident[16];  // Magic number, class, endian
  uint16_t e_type;            // 파일 타입 (EXEC, DYN, REL)
  uint16_t e_machine;         // 아키텍처 (x86, x64, ARM)
  uint32_t e_version;         // ELF 버전
  uint64_t e_entry;           // Entry point 주소
  uint64_t e_phoff;           // Program Header 오프셋
  uint64_t e_shoff;           // Section Header 오프셋
  uint32_t e_flags;           // 프로세서별 플래그
  uint16_t e_ehsize;          // ELF Header 크기
  uint16_t e_phentsize;       // Program Header 엔트리 크기
  uint16_t e_phnum;           // Program Header 개수
  uint16_t e_shentsize;       // Section Header 엔트리 크기
  uint16_t e_shnum;           // Section Header 개수
  uint16_t e_shstrndx;        // Section name string table 인덱스
} Elf64_Ehdr;
```

| Field | 설명 | 실무 팁 |
|-------|------|---------|
| `e_ident[0-3]` | Magic number | `0x7F 'E' 'L' 'F'` 확인<br>다르면 ELF 아님 |
| `e_ident[4]` | Class | `1` = 32bit (ELF32)<br>`2` = 64bit (ELF64) |
| `e_ident[5]` | Endian | `1` = Little endian<br>`2` = Big endian |
| `e_type` | 파일 타입 | `2` = EXEC (실행 파일)<br>`3` = DYN (공유 라이브러리/PIE)<br>`1` = REL (오브젝트 파일) |
| `e_machine` | CPU 타입 | `0x3E` = x86-64<br>`0x03` = x86<br>`0xB7` = ARM64 |
| `e_entry` | 진입점 주소 | 프로그램 시작 주소<br>디버거 BP 설정 시 사용 |

**확인 방법:**
```bash
# ELF Header 확인
readelf -h ./app

# Magic number 확인
xxd -l 16 ./app
# 출력: 7f 45 4c 46 02 01 01 00 ...
#       ^  E  L  F  64 LE
```

### Program Headers (Segments)

**개념:**
- 로더가 사용하는 정보
- 어떤 데이터를 메모리의 어디에 로드할지 지정

```c
typedef struct {
  uint32_t p_type;    // 세그먼트 타입
  uint32_t p_flags;   // 권한 (R/W/X)
  uint64_t p_offset;  // 파일 오프셋
  uint64_t p_vaddr;   // 가상 주소
  uint64_t p_paddr;   // 물리 주소 (무시)
  uint64_t p_filesz;  // 파일 크기
  uint64_t p_memsz;   // 메모리 크기
  uint64_t p_align;   // 정렬
} Elf64_Phdr;
```

**주요 세그먼트 타입:**

| 타입 | 값 | 설명 | 실무 활용 |
|------|-----|------|-----------|
| `PT_NULL` | 0 | 사용 안함 | 무시 |
| `PT_LOAD` | 1 | 로딩 가능 세그먼트 | 실제 메모리에 매핑<br>권한에 따라 분리 (R-X, RW-) |
| `PT_DYNAMIC` | 2 | 동적 링킹 정보 | 필요한 라이브러리<br>RPATH, RUNPATH |
| `PT_INTERP` | 3 | 인터프리터 경로 | 동적 링커 경로<br>(예: `/lib64/ld-linux-x86-64.so.2`) |
| `PT_NOTE` | 4 | 부가 정보 | Build ID 등 |
| `PT_TLS` | 7 | Thread Local Storage | 스레드별 변수 |
| `PT_GNU_STACK` | 0x6474e551 | 스택 권한 | NX bit 확인 |
| `PT_GNU_RELRO` | 0x6474e552 | Read-only 재배치 | RELRO 확인 |

**확인 방법:**
```bash
# Program Headers 확인
readelf -l ./app

# 출력 예시:
# Type      Offset   VirtAddr           FileSiz  MemSiz   Flg Align
# LOAD      0x000000 0x0000000000400000 0x001000 0x001000 R E 0x1000
# LOAD      0x001000 0x0000000000601000 0x000500 0x000600 RW  0x1000
# DYNAMIC   0x001200 0x0000000000601200 0x000200 0x000200 RW  0x8
# GNU_STACK 0x000000 0x0000000000000000 0x000000 0x000000 RW  0x10
```

### Section Headers (Sections)

**개념:**
- 링커가 사용하는 정보
- 코드, 데이터, 심볼 등을 논리적으로 구분

```c
typedef struct {
  uint32_t sh_name;       // 섹션 이름 (string table 인덱스)
  uint32_t sh_type;       // 섹션 타입
  uint64_t sh_flags;      // 섹션 플래그
  uint64_t sh_addr;       // 가상 주소
  uint64_t sh_offset;     // 파일 오프셋
  uint64_t sh_size;       // 섹션 크기
  uint32_t sh_link;       // 관련 섹션 인덱스
  uint32_t sh_info;       // 추가 정보
  uint64_t sh_addralign;  // 정렬
  uint64_t sh_entsize;    // 엔트리 크기 (테이블인 경우)
} Elf64_Shdr;
```

**섹션 플래그:**

```c
#define SHF_WRITE     0x1    // 쓰기 가능
#define SHF_ALLOC     0x2    // 메모리에 할당
#define SHF_EXECINSTR 0x4    // 실행 가능
```

**확인 방법:**
```bash
# Section Headers 확인
readelf -S ./app

# 특정 섹션 내용 보기
readelf -x .rodata ./app   # Hex dump
readelf -p .rodata ./app   # String dump

# 섹션 크기 확인
size ./app
# 출력:
#    text    data     bss     dec     hex filename
#   12345    1234    5678   19257    4b39 ./app
```

----

## Dynamic Linking

### PLT (Procedure Linkage Table)

**개념:**
- 외부 함수 호출을 위한 트램폴린 코드
- Lazy binding: 첫 호출 시에만 주소 해석

**동작 방식:**

```asm
; printf@plt (PLT 엔트리)
printf@plt:
    jmp    [printf@got.plt]    ; GOT에서 주소 점프
    push   0                    ; 심볼 인덱스
    jmp    PLT0                 ; 링커 호출

; 첫 호출 시:
;   GOT에는 다음 명령 주소가 저장 (push 0)
;   → 동적 링커 호출 → printf 주소 해석 → GOT에 저장
;
; 이후 호출:
;   GOT에는 실제 printf 주소가 저장
;   → 직접 점프
```

**실전 예제:**

```bash
# PLT 확인
objdump -d ./app | grep @plt
# 출력:
# 0000000000401030 <printf@plt>:
# 0000000000401040 <malloc@plt>:

# PLT를 통한 함수 호출 확인
objdump -d ./app | grep "call.*@plt"
# 출력:
# 401234:  e8 f7 fd ff ff    call   401030 <printf@plt>
```

### GOT (Global Offset Table)

**개념:**
- 외부 함수/변수의 주소를 저장하는 테이블
- 동적 링커가 런타임에 채워줌

**GOT vs GOT.PLT:**

| GOT | GOT.PLT |
|-----|---------|
| 전역 변수 주소 | 함수 주소 |
| 로드 시 해석 | Lazy binding (첫 호출 시) |
| RELRO로 보호 가능 | 쓰기 권한 필요 (GOT overwrite 취약점) |

**GOT overwrite 공격:**

```c
// 취약한 코드
char buf[8];
gets(buf);  // 버퍼 오버플로우

// 공격 시나리오:
// 1. GOT에서 exit@got 주소 확인
// 2. 버퍼 오버플로우로 exit@got를 system 주소로 덮어쓰기
// 3. exit() 호출 시 system()이 실행됨
```

**확인 방법:**

```bash
# GOT 확인
readelf -r ./app

# 출력 예시:
# Relocation section '.rela.plt':
# Offset         Info           Type           Sym. Value    Sym. Name + Addend
# 000000601018  000100000007 R_X86_64_JUMP_SLO 0000000000000000 printf + 0
# 000000601020  000200000007 R_X86_64_JUMP_SLO 0000000000000000 malloc + 0

# GOT 주소 확인
objdump -R ./app
```

### 동적 링커

**동적 링커란:**
- 프로그램 실행 시 공유 라이브러리를 로드하고 심볼을 해석
- Linux: `/lib64/ld-linux-x86-64.so.2`
- 환경 변수: `LD_LIBRARY_PATH`, `LD_PRELOAD`

**동적 링커 경로 확인:**

```bash
# 인터프리터 확인
readelf -l ./app | grep interpreter
# 출력: [Requesting program interpreter: /lib64/ld-linux-x86-64.so.2]

# 필요한 라이브러리 확인
ldd ./app
# 출력:
#   linux-vdso.so.1 (0x00007fff...)
#   libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f...)
#   /lib64/ld-linux-x86-64.so.2 (0x00007f...)
```

**라이브러리 검색 순서:**

```
1. RPATH (컴파일 타임, DT_RPATH)
2. LD_LIBRARY_PATH (환경 변수)
3. RUNPATH (컴파일 타임, DT_RUNPATH)
4. /etc/ld.so.cache (시스템 캐시)
5. /lib, /usr/lib (기본 경로)
```

**RPATH vs RUNPATH:**

```bash
# RPATH 설정 (보안 취약 - LD_LIBRARY_PATH보다 우선)
gcc -Wl,-rpath,/custom/lib app.c -o app

# RUNPATH 설정 (권장 - LD_LIBRARY_PATH가 우선)
gcc -Wl,--enable-new-dtags,-rpath,/custom/lib app.c -o app

# 확인
readelf -d ./app | grep PATH
# 출력:
#  (RPATH)    Library rpath: [/custom/lib]
#  (RUNPATH)  Library runpath: [/custom/lib]

# patchelf로 수정
patchelf --set-rpath /new/path ./app
patchelf --remove-rpath ./app
```

----

## ELF Loading 과정

프로그램을 실행하면 커널과 동적 링커가 다음 순서로 동작합니다.

### 1. 커널의 ELF 로딩

```c
// 커널 내부 동작 (의사코드)

// ELF 검증
if (memcmp(ehdr->e_ident, "\x7fELF", 4) != 0) {
    return -ENOEXEC;
}

// Program Headers 순회
for (i = 0; i < ehdr->e_phnum; i++) {
    Elf64_Phdr *phdr = &phdrs[i];

    if (phdr->p_type == PT_LOAD) {
        // LOAD 세그먼트를 메모리에 매핑
        void *addr = mmap(
            (void *)phdr->p_vaddr,
            phdr->p_memsz,
            prot_from_flags(phdr->p_flags),
            MAP_FIXED | MAP_PRIVATE,
            fd,
            phdr->p_offset
        );
    }

    if (phdr->p_type == PT_INTERP) {
        // 동적 링커 경로 읽기
        char interp[PATH_MAX];
        read_interp(fd, phdr->p_offset, interp);
        // /lib64/ld-linux-x86-64.so.2
    }
}

// 동적 링커로 제어 이전
exec_interp(interp_path, argv, envp);
```

### 2. 동적 링커의 초기화

```c
// 동적 링커 내부 동작 (의사코드)

// 필요한 라이브러리 로드
for (dep in PT_DYNAMIC) {
    if (dep->d_tag == DT_NEEDED) {
        char *libname = get_string(dep->d_val);
        void *handle = load_library(libname);
    }
}

// 심볼 해석 (Non-lazy binding)
for (rel in RELA_DYN) {
    void *sym_addr = resolve_symbol(rel->r_sym);
    *(void **)rel->r_offset = sym_addr;
}

// PLT 초기화 (Lazy binding)
// GOT.PLT에 링커 호출 코드 주소 저장
setup_plt();
```

### 3. 재배치 (Relocation)

```c
// 재배치 수행
for (i = 0; i < num_relocations; i++) {
    Elf64_Rela *rela = &rela_table[i];

    switch (ELF64_R_TYPE(rela->r_info)) {
    case R_X86_64_GLOB_DAT:
        // 전역 변수 재배치
        *(uint64_t *)(base + rela->r_offset) =
            symbol_value + rela->r_addend;
        break;

    case R_X86_64_JUMP_SLOT:
        // 함수 주소 재배치 (GOT.PLT)
        *(uint64_t *)(base + rela->r_offset) =
            symbol_value + rela->r_addend;
        break;

    case R_X86_64_RELATIVE:
        // PIE/PIC 재배치
        *(uint64_t *)(base + rela->r_offset) =
            base + rela->r_addend;
        break;
    }
}
```

### 4. 초기화 함수 실행

```c
// .init_array 실행
void **init_array = find_section(".init_array");
for (i = 0; init_array[i]; i++) {
    ((void (*)())init_array[i])();
}

// 생성자 실행 (__attribute__((constructor)))
call_constructors();
```

### 5. Entry Point 실행

```c
// main() 호출
typedef int (*main_t)(int, char **, char **);
main_t main_func = (main_t)ehdr->e_entry;

int exitcode = main_func(argc, argv, envp);

// 종료 처리
exit(exitcode);
```

### 6. 종료 처리

```c
// .fini_array 실행
void **fini_array = find_section(".fini_array");
for (i = num_fini - 1; i >= 0; i--) {
    ((void (*)())fini_array[i])();
}

// 소멸자 실행 (__attribute__((destructor)))
call_destructors();

// 프로세스 종료
_exit(exitcode);
```

### 로딩 과정 요약

```
1. 커널 ELF 로딩    : PT_LOAD 세그먼트를 메모리에 매핑
2. 동적 링커 호출   : PT_INTERP로 지정된 링커 실행
3. 라이브러리 로드  : DT_NEEDED 라이브러리들 로드
4. 심볼 해석        : GOT 채우기
5. 재배치 수행      : PIE/PIC 주소 보정
6. 초기화 실행      : .init_array, constructors
7. Entry Point      : main() 함수 실행
8. 종료 처리        : .fini_array, destructors
```

**Lazy Binding 동작:**

```
첫 번째 printf 호출:
  1. call printf@plt
  2. jmp [printf@got.plt]  → 아직 해석 안됨, PLT 다음 명령으로
  3. push 심볼 인덱스
  4. 동적 링커 호출
  5. printf 주소 해석
  6. GOT.PLT 업데이트
  7. printf 실행

두 번째 printf 호출:
  1. call printf@plt
  2. jmp [printf@got.plt]  → 해석된 printf 주소로 직접 점프 ✓
```

----

## 보안 기능

### ASLR (Address Space Layout Randomization)

**개념:**
- 프로그램을 실행할 때마다 메모리 주소를 랜덤하게 배치
- 공격자가 주소를 예측하기 어렵게 만듦

**시스템 설정:**
```bash
# ASLR 상태 확인
cat /proc/sys/kernel/randomize_va_space
# 0 = 비활성화
# 1 = 스택/힙/라이브러리만
# 2 = 스택/힙/라이브러리/실행파일 (PIE 필요)

# ASLR 활성화 (권장)
echo 2 | sudo tee /proc/sys/kernel/randomize_va_space
```

**PIE (Position Independent Executable):**
- ASLR을 실행 파일에도 적용하려면 PIE로 컴파일 필요

```bash
# PIE 컴파일
gcc -fPIE -pie app.c -o app

# PIE 확인
readelf -h app | grep Type
# Type: DYN (Shared object file) ← PIE 활성화
# Type: EXEC (Executable file)  ← PIE 비활성화

# checksec 도구 사용
checksec --file=./app
# PIE enabled: Yes
```

### NX (No-Execute)

**개념:**
- 데이터 영역(.data, stack, heap)의 코드 실행 금지
- DEP와 동일한 개념

**활성화 방법:**
```bash
# NX 컴파일 (기본 활성화)
gcc -z noexecstack app.c -o app

# NX 확인
readelf -l app | grep GNU_STACK
# GNU_STACK ... RW  ← NX 활성화 (실행 권한 없음)
# GNU_STACK ... RWE ← NX 비활성화 (실행 권한 있음)

# checksec 도구
checksec --file=./app
# NX enabled: Yes
```

### RELRO (Relocation Read-Only)

**개념:**
- GOT 영역을 읽기 전용으로 만들어 GOT overwrite 공격 방지

**Partial RELRO:**
- GOT는 여전히 쓰기 가능
- .init_array, .fini_array는 읽기 전용

**Full RELRO:**
- 모든 심볼을 로드 시 해석 (lazy binding 비활성화)
- GOT도 읽기 전용
- 시작 시간 증가

```bash
# Full RELRO 컴파일 (권장)
gcc -Wl,-z,relro,-z,now app.c -o app

# Partial RELRO 컴파일
gcc -Wl,-z,relro app.c -o app

# RELRO 확인
readelf -l app | grep GNU_RELRO
# GNU_RELRO ... ← RELRO 활성화

readelf -d app | grep BIND_NOW
# (BIND_NOW) ← Full RELRO (lazy binding 비활성화)
# 없으면 Partial RELRO

# checksec 도구
checksec --file=./app
# RELRO: Full RELRO
```

### Stack Canary

**개념:**
- 스택 버퍼 오버플로우 탐지
- 함수 반환 전 canary 값 검증

```bash
# Stack Canary 컴파일 (기본 활성화)
gcc -fstack-protector-strong app.c -o app

# 확인
objdump -d app | grep stack_chk_fail
# call   <__stack_chk_fail@plt>

# checksec 도구
checksec --file=./app
# Canary: Yes
```

### FORTIFY_SOURCE

**개념:**
- 위험한 함수 (strcpy, sprintf 등)를 안전한 버전으로 치환
- 버퍼 크기 검사 추가

```bash
# FORTIFY_SOURCE 컴파일
gcc -D_FORTIFY_SOURCE=2 -O2 app.c -o app

# 확인
objdump -T app | grep _chk
# __strcpy_chk
# __sprintf_chk
```

### 보안 체크리스트

```bash
# checksec 설치
sudo apt install checksec   # Debian/Ubuntu
brew install checksec       # macOS

# 전체 보안 기능 확인
checksec --file=./app

# 권장 출력:
# PIE:      PIE enabled
# NX:       NX enabled
# RELRO:    Full RELRO
# Canary:   Canary found
# FORTIFY:  Enabled

# 안전한 컴파일 플래그
gcc -fPIE -pie \
    -fstack-protector-strong \
    -D_FORTIFY_SOURCE=2 \
    -Wl,-z,relro,-z,now \
    -Wl,-z,noexecstack \
    -O2 \
    app.c -o app
```

----

## 실무 활용

### 시나리오 1: Segmentation Fault 분석

**문제:**
```
Segmentation fault (core dumped)
```

**분석 과정:**

```bash
# 1. Core dump 활성화
ulimit -c unlimited

# 2. 프로그램 재실행
./app
# Segmentation fault (core dumped)

# 3. GDB로 분석
gdb ./app core
(gdb) bt            # 백트레이스
(gdb) info registers
(gdb) x/i $rip      # Crash 지점 명령어

# 4. 어느 섹션에서 발생했는지 확인
readelf -S ./app
# .text 섹션 범위 확인

# 5. 심볼 정보로 함수 추적
readelf -s ./app | grep function_name
```

### 시나리오 2: 라이브러리 의존성 문제

**문제:**
```
./app: error while loading shared libraries: libfoo.so.1: cannot open shared object file
```

**해결 과정:**

```bash
# 1. 필요한 라이브러리 확인
ldd ./app
# libfoo.so.1 => not found

# 2. 라이브러리 검색
find /usr -name "libfoo.so*"
# /usr/local/lib/libfoo.so.1.2.3

# 3. 해결 방법 A: 심볼릭 링크
sudo ln -s /usr/local/lib/libfoo.so.1.2.3 /usr/lib/libfoo.so.1

# 4. 해결 방법 B: LD_LIBRARY_PATH 설정
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
./app

# 5. 해결 방법 C: RPATH 수정 (권장)
patchelf --set-rpath /usr/local/lib ./app

# 6. 해결 방법 D: /etc/ld.so.conf 수정
echo "/usr/local/lib" | sudo tee -a /etc/ld.so.conf.d/local.conf
sudo ldconfig
```

### 시나리오 3: 심볼 충돌

**문제:**
```
Symbol `foo' causes multiple definition error
```

**진단:**

```bash
# 1. 심볼 확인
nm -D app | grep foo
nm -D libbar.so | grep foo

# 2. 어느 라이브러리에서 정의되는지 확인
readelf -s app | grep foo
readelf -s libbar.so | grep foo

# 3. 해결: 심볼 숨기기 (visibility)
# foo.c
__attribute__((visibility("hidden")))
void foo() {
    // ...
}

# 4. 재컴파일
gcc -fvisibility=hidden app.c -o app
```

### 시나리오 4: 바이너리 크기 최적화

**문제:**
```
바이너리가 너무 큼 (배포 용량 문제)
```

**최적화:**

```bash
# 1. 심볼 테이블 제거
strip ./app
# 크기 감소: ~30-50%

# 2. 크기 비교
ls -lh app
# Before: 2.5M
# After:  1.2M

# 3. 부분 strip (디버깅 심볼만 제거)
strip --strip-debug ./app

# 4. 최적화 컴파일
gcc -Os app.c -o app          # 크기 최적화
gcc -O3 app.c -o app          # 속도 최적화

# 5. LTO (Link Time Optimization)
gcc -flto app.c lib.c -o app  # 더 공격적인 최적화

# 6. UPX 압축 (런타임 압축 해제)
upx --best ./app
# 크기 감소: ~50-70%
```

----

## 유용한 도구

### readelf

**기본 사용:**
```bash
# 전체 정보
readelf -a app

# ELF Header
readelf -h app

# Program Headers
readelf -l app

# Section Headers
readelf -S app

# 심볼 테이블
readelf -s app

# 동적 심볼
readelf --dyn-syms app

# 재배치 정보
readelf -r app

# 동적 섹션
readelf -d app
```

### objdump

**기본 사용:**
```bash
# 디스어셈블
objdump -d app

# 특정 섹션 디스어셈블
objdump -d -j .text app

# 소스 코드와 함께 (디버그 정보 필요)
objdump -S app

# 심볼 테이블
objdump -t app

# 동적 심볼
objdump -T app

# 재배치 정보
objdump -R app

# 섹션 헤더
objdump -h app
```

### nm

**기본 사용:**
```bash
# 심볼 목록
nm app

# 동적 심볼만
nm -D app

# C++ 이름 복원
nm -C app

# 정렬
nm -n app          # 주소순
nm -S app          # 크기 표시
nm --size-sort app # 크기순

# 특정 심볼 찾기
nm app | grep printf
```

### ldd

**기본 사용:**
```bash
# 의존 라이브러리 확인
ldd app
# 출력:
#   linux-vdso.so.1 (0x00007fff...)
#   libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6
#   /lib64/ld-linux-x86-64.so.2

# 상세 정보
ldd -v app

# 사용하지 않는 의존성 확인
ldd -u app

# 주의: 실행 파일을 로드하므로 신뢰할 수 없는 파일에 사용 금지
```

### patchelf

**설치:**
```bash
sudo apt install patchelf   # Debian/Ubuntu
brew install patchelf       # macOS
```

**사용 예제:**
```bash
# RPATH 확인
patchelf --print-rpath app

# RPATH 설정
patchelf --set-rpath /custom/lib app

# RPATH 제거
patchelf --remove-rpath app

# 인터프리터 변경
patchelf --set-interpreter /lib64/ld-linux-x86-64.so.2 app

# RUNPATH 설정
patchelf --set-rpath /custom/lib --force-rpath app
```

### checksec

**설치:**
```bash
sudo apt install checksec
brew install checksec
```

**사용 예제:**
```bash
# 단일 파일 확인
checksec --file=./app

# 디렉토리 전체 확인
checksec --dir=/usr/bin

# 실행 중인 프로세스 확인
checksec --proc-all
```

### GDB

**ELF 디버깅:**
```bash
# 실행
gdb ./app

# 명령어
(gdb) info files         # 섹션 정보
(gdb) info functions     # 함수 목록
(gdb) info variables     # 변수 목록
(gdb) info sharedlibrary # 로드된 라이브러리

# 메모리 매핑
(gdb) info proc mappings

# GOT 확인
(gdb) x/10gx 0x601000    # GOT 주소

# PLT 확인
(gdb) disas printf@plt
```

----

## 트러블슈팅

### 1. "cannot execute binary file: Exec format error"

**원인:**
- 다른 아키텍처 바이너리
- 손상된 ELF 파일

**진단:**
```bash
# 아키텍처 확인
file app
# ELF 64-bit LSB executable, x86-64

readelf -h app | grep Machine
# Machine: Advanced Micro Devices X86-64

# Magic number 확인
xxd -l 4 app
# 00000000: 7f45 4c46  .ELF
```

**해결:**
- 올바른 아키텍처로 재컴파일
- 크로스 컴파일 도구 사용

### 2. "symbol lookup error: undefined symbol"

**원인:**
- 필요한 라이브러리 누락
- 잘못된 라이브러리 버전

**해결:**
```bash
# 1. 필요한 심볼 확인
nm -D app | grep symbol_name

# 2. 어느 라이브러리에 있는지 찾기
for lib in /usr/lib/*.so*; do
    nm -D "$lib" 2>/dev/null | grep -q symbol_name && echo "$lib"
done

# 3. 라이브러리 링크
gcc app.c -o app -lfoo

# 4. 런타임 경로 설정
export LD_LIBRARY_PATH=/path/to/lib
```

### 3. Segmentation Fault

**원인:**
- Null pointer 접근
- 스택 오버플로우
- 힙 손상
- 잘못된 메모리 접근

**디버깅:**
```bash
# 1. Core dump 활성화
ulimit -c unlimited

# 2. GDB로 분석
gdb ./app core
(gdb) bt                # 백트레이스
(gdb) info registers    # 레지스터
(gdb) x/10i $rip-10     # Crash 전후 명령어

# 3. Valgrind로 메모리 오류 찾기
valgrind --leak-check=full ./app

# 4. AddressSanitizer 사용
gcc -fsanitize=address -g app.c -o app
./app
```

### 4. GLIBC 버전 불일치

**증상:**
```
./app: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found
```

**진단:**
```bash
# 필요한 GLIBC 버전 확인
objdump -T app | grep GLIBC
# 0000000000000000      DF *UND* GLIBC_2.34 ...

# 시스템 GLIBC 버전 확인
ldd --version
# ldd (Ubuntu GLIBC 2.31-0ubuntu9) 2.31
```

**해결:**
```bash
# 방법 1: 정적 링크 (권장)
gcc -static app.c -o app

# 방법 2: 낮은 버전 GLIBC를 타겟으로 컴파일
# 이전 버전 Ubuntu/Debian에서 컴파일

# 방법 3: Docker 사용
docker run -v $(pwd):/work ubuntu:20.04 bash
cd /work
gcc app.c -o app
```

### 5. "Text file busy" 오류

**증상:**
```
./app: Text file busy
```

**원인:**
- 실행 중인 프로세스가 있음
- 파일이 쓰기로 열려있음

**해결:**
```bash
# 1. 실행 중인 프로세스 확인
ps aux | grep app
lsof app

# 2. 프로세스 종료
killall app

# 3. 강제 종료
killall -9 app
```

----

## 참고 자료

### 공식 문서

- [ELF Specification @ SCO](http://www.sco.com/developers/gabi/latest/contents.html)
- [System V ABI](https://refspecs.linuxfoundation.org/elf/elf.pdf)
- [x86-64 ABI](https://refspecs.linuxbase.org/elf/x86_64-abi-0.99.pdf)

### 추천 도구

- [patchelf](https://github.com/NixOS/patchelf) - ELF 수정
- [checksec](https://github.com/slimm609/checksec.sh) - 보안 기능 확인
- [radare2](https://github.com/radareorg/radare2) - 리버스 엔지니어링
- [GDB](https://www.gnu.org/software/gdb/) - 디버거

### 학습 자료

**입문:**
- [teensy @ muppetlabs](http://www.muppetlabs.com/~breadbox/software/tiny/teensy.html) - build-up 방식으로 ELF 설명
- [ELF101 @ github](https://github.com/corkami/pics/tree/master/binary) - ELF 구조 다이어그램

**중급:**
- [How Programs Get Run @ LWN](https://lwn.net/Articles/630727/) - ELF 로딩 과정
- [Understanding the ELF](https://medium.com/@MrJamesFisher/understanding-the-elf-4bd60daac571)

**고급:**
- [A Whirlwind Tutorial on Creating Really Teensy ELF Executables](http://www.muppetlabs.com/~breadbox/software/tiny/)
- [The Linux Programming Interface](https://man7.org/tlpi/) - Chapter 41: Shared Libraries

### Python 라이브러리

```bash
# ELF 파싱
pip install pyelftools

# 사용 예제
from elftools.elf.elffile import ELFFile

with open('app', 'rb') as f:
    elf = ELFFile(f)
    print(f"Entry point: {hex(elf.header['e_entry'])}")

    for section in elf.iter_sections():
        print(f"{section.name}: {hex(section['sh_addr'])}")
```

----

## 부록: 빠른 참조

### 주소 계산 공식

```
VA (Virtual Address) = 메모리 가상 주소
File Offset = VA - Segment_VAddr + Segment_FileOffset
```

### readelf 치트시트

```bash
# 전체 정보
readelf -a app

# 특정 정보만
readelf -h app        # ELF Header
readelf -l app        # Program Headers
readelf -S app        # Section Headers
readelf -s app        # Symbol table
readelf -d app        # Dynamic section
readelf -r app        # Relocations

# 필터링
readelf -h app | grep Entry
readelf -l app | grep LOAD
readelf -s app | grep FUNC
```

### 보안 플래그 체크

```bash
# checksec 사용
checksec --file=./app

# 권장 출력:
# PIE:      PIE enabled
# NX:       NX enabled
# RELRO:    Full RELRO
# Canary:   Canary found

# 안전한 컴파일 명령
gcc -fPIE -pie \
    -fstack-protector-strong \
    -D_FORTIFY_SOURCE=2 \
    -Wl,-z,relro,-z,now \
    -Wl,-z,noexecstack \
    -O2 \
    app.c -o app
```

### GDB 디버깅 치트시트

```bash
# 시작
gdb ./app

# 유용한 명령어
(gdb) info files                # 섹션 정보
(gdb) info functions            # 함수 목록
(gdb) info sharedlibrary        # 로드된 라이브러리
(gdb) info proc mappings        # 메모리 맵

# PLT/GOT 확인
(gdb) disas 'printf@plt'        # PLT 디스어셈블
(gdb) x/10gx 0x601000           # GOT 내용 확인

# 백트레이스
(gdb) bt                        # 전체
(gdb) bt full                   # 로컬 변수 포함
```
