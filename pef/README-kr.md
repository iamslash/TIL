# PE (Portable Executable) Format

Windows 실행 파일(.exe, .dll)의 내부 구조를 이해하고 실무에서 활용하는 방법을 설명합니다.

## 목차

- [Overview](#overview)
- [PE 구조](#pe-구조)
- [Headers](#headers)
- [Import & Export](#import--export)
- [PE Loading 과정](#pe-loading-과정)
- [보안 기능](#보안-기능)
- [실무 활용](#실무-활용)
- [유용한 도구](#유용한-도구)
- [트러블슈팅](#트러블슈팅)
- [참고 자료](#참고-자료)

----

## Overview

### PE란?

PE(Portable Executable)는 Windows에서 실행 파일의 표준 포맷입니다.
- `.exe` - 실행 파일
- `.dll` - Dynamic Link Library
- `.sys` - Device Driver
- `.ocx` - ActiveX Control

### 실무에서 왜 필요한가?

**디버깅 시나리오:**
```
프로그램이 0x00401234 주소에서 Crash 발생

PE 구조를 알면:
→ 0x00401234가 어느 모듈의 어느 섹션인지 파악
→ ImageBase가 0x00400000이면 RVA는 0x1234
→ .text 섹션(코드)인지 .data 섹션(데이터)인지 확인
```

**성능 최적화:**
- DLL의 Import table이 크면 loading time 증가
- Delay-load로 필요한 DLL만 로딩하여 Startup 시간 개선

**보안 분석:**
- Malware가 PE header를 조작하는 방식 이해
- DEP, ASLR 등 보호 기능 활성화 여부 확인

**배포 문제 해결:**
- Dependency Walker로 필요한 DLL 파악
- 32bit/64bit 혼용 문제 진단

### File vs Memory

PE 파일은 disk와 memory에서 다른 형태로 존재합니다:

| Aspect | File (Disk) | Memory (Loaded) |
|--------|-------------|-----------------|
| Alignment | FileAlignment (512B) | SectionAlignment (4KB) |
| .bss 섹션 | 파일에 없음 | 메모리에 할당됨 |
| Import table | Function names | Function addresses (IAT) |
| 주소 | File offset (RAW) | Virtual Address (RVA) |

**왜 중요한가:**
- Debugger는 memory address를 표시
- Static analyzer는 file offset을 표시
- RVA ↔ File offset 변환 필요

----

## PE 구조

### High-level 구조

```
┌─────────────────────┐
│   DOS Header        │ ← "MZ" 서명 (하위 호환용)
├─────────────────────┤
│   DOS Stub          │ ← "This program cannot be run..."
├─────────────────────┤
│   PE Header         │ ← "PE" 서명
│   - Signature       │
│   - File Header     │   • CPU 종류 (x86/x64)
│   - Optional Header │   • Entry Point
│                     │   • 섹션 정보
├─────────────────────┤
│  Section Header 1   │ ← .text (코드)
│  Section Header 2   │ ← .data (데이터)
│  Section Header 3   │ ← .rsrc (리소스)
├─────────────────────┤
│   .text Section     │ ← 실제 코드
├─────────────────────┤
│   .data Section     │ ← 전역 변수
├─────────────────────┤
│   .rsrc Section     │ ← 아이콘, 문자열
└─────────────────────┘
```

### 주요 섹션

| 섹션 | 용도 | 권한 |
|------|------|------|
| `.text` | 실행 코드 | Read, Execute |
| `.data` | 초기화된 전역 변수 | Read, Write |
| `.rdata` | 읽기 전용 데이터 (상수) | Read |
| `.bss` | 초기화 안된 전역 변수 | Read, Write |
| `.rsrc` | 리소스 (아이콘, 문자열) | Read |
| `.idata` | Import table | Read |
| `.edata` | Export table | Read |
| `.reloc` | Base relocation | Read |

### 주소 변환

PE 파일에서 사용되는 3가지 주소 표현 방식:

**RVA (Relative Virtual Address):**
- ImageBase로부터의 **상대 주소**
- 예: `RVA = 0x1234`

**VA (Virtual Address):**
- 메모리의 **절대 주소**
- 계산: `VA = ImageBase + RVA`
- 예: `VA = 0x00400000 + 0x1234 = 0x00401234`

**RAW (File Offset):**
- **파일에서의 위치** (바이트 오프셋)
- 계산: `RAW = RVA - SectionRVA + SectionRAW`

#### ImageBase란?

**ImageBase**는 PE 파일이 프로세스의 가상 메모리에 로드될 때 **시작하기를 원하는 가상 주소**입니다. Optional Header에 저장되어 있는 **선호 로딩 주소**입니다.

**기본값 (Default ImageBase):**

32bit:
```
EXE 파일:  0x00400000
DLL 파일:  0x10000000, 0x50000000, 0x60000000 등 (충돌 방지)
SYS 파일:  0x00010000
```

64bit:
```
EXE 파일:  0x0000000140000000
DLL 파일:  0x0000000180000000 등 (다양)
```

**왜 DLL마다 다른 ImageBase를 쓰는가?**

한 프로세스에 여러 DLL이 로드되므로 주소 충돌을 방지하기 위함:

```
Process A의 Virtual Memory 공간:
┌─────────────────────────┐
│ app.exe                 │ 0x00400000
├─────────────────────────┤
│ kernel32.dll            │ 0x77000000
├─────────────────────────┤
│ user32.dll              │ 0x76000000
├─────────────────────────┤
│ mydll.dll               │ 0x10000000
└─────────────────────────┘
```

**프로세스 간 독립성:**

각 프로세스는 독립적인 가상 주소 공간을 가지므로, 다른 프로세스에서 같은 ImageBase를 사용해도 충돌하지 않습니다:

```
Process A (notepad.exe):  ImageBase = 0x00400000 ✓
Process B (notepad.exe):  ImageBase = 0x00400000 ✓ (다른 Virtual Memory)
Process C (calc.exe):     ImageBase = 0x00400000 ✓ (다른 Virtual Memory)
```

**ASLR의 영향:**

- **ASLR 비활성화**: PE는 항상 ImageBase 주소에 로드 (예측 가능 → 보안 취약)
- **ASLR 활성화**: 실행할 때마다 **랜덤한 주소**에 로드 (ImageBase는 "희망 사항"일 뿐)

```cpp
// ASLR 활성화 시
실행 1: app.exe가 0x00400000에 로드
실행 2: app.exe가 0x00D20000에 로드 (랜덤)
실행 3: app.exe가 0x01A40000에 로드 (랜덤)
```

**실제 로드 주소가 ImageBase와 다를 때:**

Base Relocation이 필요합니다 (자세한 내용은 [PE Loading 과정](#pe-loading-과정) 참조).

```cpp
// 충돌 예시
ImageBase = 0x10000000 (희망)
실제 로드 = 0x20000000 (다른 DLL이 0x10000000 사용 중)
→ Relocation 수행: 모든 절대 주소를 +0x10000000 보정
```

**확인 방법:**

```bash
# ImageBase 확인
dumpbin /headers app.exe | findstr "image base"
# 출력: 400000 image base (32bit)
# 출력: 140000000 image base (64bit)

# 런타임 실제 로드 주소 확인 (디버거)
# WinDbg: lm (loaded modules)
# x64dbg: Symbols tab
```

#### 주소 변환 공식

**중요:** `IMAGE_SECTION_HEADER`의 `VirtualAddress` 필드는 이름과 달리 **RVA 값**입니다!

```cpp
// Section Header 구조체
typedef struct _IMAGE_SECTION_HEADER {
  DWORD VirtualAddress;      // 섹션 시작 RVA (상대 주소!)
  DWORD PointerToRawData;    // 섹션 시작 RAW (파일 오프셋)
  ...
} IMAGE_SECTION_HEADER;
```

**변환 공식:**
```
RAW = RVA - VirtualAddress + PointerToRawData
      └─────────┬─────────┘
          섹션 내 오프셋
```

#### 실전 예제

`notepad.exe`의 `.text` 섹션 정보:
```
VirtualAddress   = 0x1000  (메모리에서 섹션 시작 RVA)
PointerToRawData = 0x400   (파일에서 섹션 시작)
ImageBase        = 0x00400000
```

**문제 1: RVA → RAW 변환**

`RVA = 0x1234` 주소가 파일의 어느 위치에 있는가?

```
1. 올바른 섹션 찾기
   0x1000 ≤ 0x1234 < 0x1000 + SectionSize
   → .text 섹션에 속함 ✓

2. 섹션 내 오프셋 계산
   Offset = 0x1234 - 0x1000 = 0x234

3. 파일 오프셋 계산
   RAW = 0x234 + 0x400 = 0x634

답: 파일의 0x634 위치
```

**문제 2: VA → RAW 변환**

`VA = 0x00401234` 주소가 파일의 어느 위치에 있는가?

```
1. RVA 계산
   RVA = VA - ImageBase
   RVA = 0x00401234 - 0x00400000 = 0x1234

2. RAW 변환 (위 문제 1과 동일)
   RAW = 0x1234 - 0x1000 + 0x400 = 0x634

답: 파일의 0x634 위치
```

**문제 3: RAW → RVA 변환**

파일의 `RAW = 0x634` 위치가 메모리의 어느 주소인가?

```
1. 올바른 섹션 찾기
   0x400 ≤ 0x634 < 0x400 + SizeOfRawData
   → .text 섹션에 속함 ✓

2. 섹션 내 오프셋 계산
   Offset = 0x634 - 0x400 = 0x234

3. RVA 계산
   RVA = 0x234 + 0x1000 = 0x1234

4. VA 계산 (선택)
   VA = 0x00400000 + 0x1234 = 0x00401234

답: RVA = 0x1234, VA = 0x00401234
```

#### 헤더 영역 예외

RVA가 첫 섹션보다 앞(헤더 영역)에 있으면:
```
RAW = RVA (변환 불필요)

예: RVA = 0x100 (DOS Header 내부)
    → RAW = 0x100
```

----

## Headers

### DOS Header

**실무 필수 필드:**

```cpp
typedef struct _IMAGE_DOS_HEADER {
    WORD  e_magic;      // "MZ" (0x5A4D)
    // ... 14개 필드 생략 ...
    DWORD e_lfanew;     // PE Header 위치
} IMAGE_DOS_HEADER;
```

| Field | 설명 | 실무 팁 |
|-------|------|---------|
| `e_magic` | "MZ" 서명 | 파일 검증 시 첫 확인 항목<br>0x5A4D가 아니면 PE 아님 |
| `e_lfanew` | PE Header offset | 보통 0x80-0x100 사이<br>비정상적으로 크면 의심 |

### NT Headers

```cpp
typedef struct _IMAGE_NT_HEADERS {
  DWORD Signature;                    // "PE\0\0" (0x4550)
  IMAGE_FILE_HEADER FileHeader;
  IMAGE_OPTIONAL_HEADER OptionalHeader;
} IMAGE_NT_HEADERS;
```

### File Header

**실무 필수 필드:**

```cpp
typedef struct _IMAGE_FILE_HEADER {
  WORD  Machine;              // CPU 아키텍처
  WORD  NumberOfSections;     // 섹션 개수
  DWORD TimeDateStamp;        // 빌드 시간
  WORD  SizeOfOptionalHeader; // Optional Header 크기
  WORD  Characteristics;      // 파일 속성
} IMAGE_FILE_HEADER;
```

| Field | 설명 | 실무 활용 |
|-------|------|-----------|
| `Machine` | CPU 타입 | `0x14C` = x86 (32bit)<br>`0x8664` = x64 (64bit)<br>아키텍처 불일치 오류 진단 |
| `NumberOfSections` | 섹션 개수 | 정상: 3-6개<br>Packer 사용 시: 10개 이상<br>1-2개: 의심스러움 |
| `TimeDateStamp` | 빌드 시간 | PDB 파일 매칭 시 사용<br>0이면 reproducible build |
| `Characteristics` | 파일 속성 | `0x0002` = Executable<br>`0x2000` = DLL<br>`0x0001` = Relocation 정보 제거됨 |

### Optional Header

**실무 핵심 필드:**

```cpp
typedef struct _IMAGE_OPTIONAL_HEADER {
  WORD  Magic;                    // 0x10B (32bit) / 0x20B (64bit)
  DWORD AddressOfEntryPoint;      // 진입점 RVA
  DWORD ImageBase;                // 로딩 기본 주소
  DWORD SectionAlignment;         // 메모리 섹션 정렬 (보통 4KB)
  DWORD FileAlignment;            // 파일 섹션 정렬 (보통 512B)
  DWORD SizeOfImage;              // 메모리 크기
  DWORD SizeOfHeaders;            // 헤더 크기
  WORD  Subsystem;                // GUI(2) / CUI(3)
  WORD  DllCharacteristics;       // 보안 플래그
  DWORD NumberOfRvaAndSizes;      // DataDirectory 개수
  IMAGE_DATA_DIRECTORY DataDirectory[16];
} IMAGE_OPTIONAL_HEADER;
```

| Field | 설명 | 디버깅/분석 활용 |
|-------|------|------------------|
| `AddressOfEntryPoint` | 시작점 RVA | Debugger에서 진입점 BP 설정<br>섹션 밖이면 Packer 의심 |
| `ImageBase` | 로딩 기본 주소 | x86: `0x00400000` (exe), `0x10000000` (dll)<br>x64: `0x140000000`<br>ASLR로 변경될 수 있음 |
| `SectionAlignment` | 메모리 정렬 | 보통 `0x1000` (4KB)<br>주소 계산 시 필요 |
| `FileAlignment` | 파일 정렬 | 보통 `0x200` (512B)<br>RVA→File offset 변환 시 사용 |
| `Subsystem` | 실행 타입 | `2` = GUI (창 표시)<br>`3` = CUI (콘솔)<br>`1` = Native (드라이버) |
| `DllCharacteristics` | 보안 설정 | **중요!** ASLR, DEP, CFG 확인 |

**DllCharacteristics - 보안 체크리스트:**

```cpp
// 반드시 설정되어야 할 플래그
#define IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE    0x0040  // ASLR 활성화
#define IMAGE_DLLCHARACTERISTICS_NX_COMPAT       0x0100  // DEP 활성화
#define IMAGE_DLLCHARACTERISTICS_TERMINAL_SERVER 0x8000  // Terminal Server

// 권장 플래그
#define IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA 0x0020  // 64bit ASLR
#define IMAGE_DLLCHARACTERISTICS_GUARD_CF        0x4000  // Control Flow Guard
```

**확인 방법:**
```bash
# dumpbin으로 확인
dumpbin /headers app.exe | findstr "DLL characteristics"

# 출력 예시:
8160 DLL characteristics
       Dynamic base          # ASLR ✓
       NX compatible         # DEP ✓
       Terminal Server Aware
```

### Section Headers

```cpp
typedef struct _IMAGE_SECTION_HEADER {
  BYTE  Name[8];              // 섹션 이름 (예: ".text")
  DWORD VirtualSize;          // 메모리 크기
  DWORD VirtualAddress;       // 메모리 시작 주소 (RVA)
  DWORD SizeOfRawData;        // 파일 크기
  DWORD PointerToRawData;     // 파일 시작 위치
  DWORD Characteristics;      // 섹션 속성 (권한)
} IMAGE_SECTION_HEADER;
```

| Field | 설명 | 실무 팁 |
|-------|------|---------|
| `VirtualSize` | 메모리 크기 | 초기화 안된 데이터 포함<br>SizeOfRawData보다 클 수 있음 |
| `VirtualAddress` | 메모리 시작 RVA | SectionAlignment 배수<br>주소 계산의 기준점 |
| `SizeOfRawData` | 파일 크기 | FileAlignment 배수<br>0이면 파일에 데이터 없음 (.bss) |
| `PointerToRawData` | 파일 위치 | RVA → RAW 변환 시 사용 |
| `Characteristics` | 권한 | Execute, Read, Write 조합 |

----

## Import & Export

### Import Address Table (IAT)

**개념:**
- DLL의 함수를 사용하기 위한 테이블
- Import = "다른 DLL의 함수를 가져와서 사용"

**Import 방식:**

1. **Implicit Linking (정적 링크)**
   - 프로그램 시작 시 DLL 로딩
   - Import table에 명시
   ```cpp
   // kernel32.dll의 CreateFileW 사용
   HANDLE h = CreateFileW(...);
   ```

2. **Explicit Linking (동적 링크)**
   - 필요할 때 DLL 로딩
   ```cpp
   HMODULE hDll = LoadLibrary(L"mydll.dll");
   FARPROC pFunc = GetProcAddress(hDll, "MyFunction");
   ```

3. **Delay-load**
   - 첫 호출 시 DLL 로딩
   ```cpp
   #pragma comment(linker, "/DELAYLOAD:user32.dll")
   MessageBoxW(...);  // 여기서 처음 user32.dll 로딩
   ```

**IMAGE_IMPORT_DESCRIPTOR:**

```cpp
typedef struct _IMAGE_IMPORT_DESCRIPTOR {
  DWORD OriginalFirstThunk;  // INT (Import Name Table) RVA
  DWORD TimeDateStamp;       // Binding 정보
  DWORD ForwarderChain;      // Forwarding 정보
  DWORD Name;                // DLL 이름 RVA
  DWORD FirstThunk;          // IAT (Import Address Table) RVA
} IMAGE_IMPORT_DESCRIPTOR;
```

**Import 동작 과정:**

```
1. OriginalFirstThunk (INT) 읽기
   → 함수 이름 확인: "CreateFileW"

2. DLL 로딩
   → LoadLibrary("kernel32.dll")

3. 함수 주소 얻기
   → GetProcAddress(hKernel32, "CreateFileW")

4. FirstThunk (IAT)에 주소 저장
   → IAT[0] = 0x77A12340

5. 프로그램이 함수 호출
   → call [IAT + 0]  // 0x77A12340 호출
```

**실무 팁:**
```bash
# Import table 확인
dumpbin /imports app.exe

# Delay-load DLL 확인
dumpbin /headers app.exe | findstr /C:"delay load"

# Import 함수 개수 (성능 영향)
dumpbin /imports app.exe | findstr "    "
```

### Export Address Table (EAT)

**개념:**
- DLL이 제공하는 함수 목록
- Export = "내 함수를 다른 프로그램이 사용하도록 공개"

**IMAGE_EXPORT_DIRECTORY:**

```cpp
typedef struct _IMAGE_EXPORT_DIRECTORY {
  DWORD Name;                    // DLL 이름 RVA
  DWORD Base;                    // Ordinal base (보통 1)
  DWORD NumberOfFunctions;       // 함수 개수
  DWORD NumberOfNames;           // 이름 있는 함수 개수
  DWORD AddressOfFunctions;      // 함수 주소 배열 RVA
  DWORD AddressOfNames;          // 함수 이름 배열 RVA
  DWORD AddressOfNameOrdinals;   // Ordinal 배열 RVA
} IMAGE_EXPORT_DIRECTORY;
```

**Export 방식:**

1. **Name Export (권장)**
   ```cpp
   // mydll.cpp
   __declspec(dllexport) void MyFunction() {
       // ...
   }
   ```

2. **Ordinal Export**
   ```cpp
   // mydll.def
   EXPORTS
       MyFunction @1
   ```

**실무 활용:**
```bash
# Export 함수 목록 확인
dumpbin /exports kernel32.dll

# 특정 함수 찾기
dumpbin /exports kernel32.dll | findstr CreateFile

# 출력:
    123   7C CreateFileA
    124   7D CreateFileW
```

----

## PE Loading 과정

프로그램을 실행하면 Windows Loader가 다음 순서로 동작합니다.

### 1. File Mapping

```cpp
// Loader 내부 동작 (의사코드)
HANDLE hFile = CreateFile(L"app.exe", ...);
HANDLE hMapping = CreateFileMapping(hFile, ...);
LPVOID pBase = MapViewOfFile(hMapping, ...);

// DOS Header 검증
if (*(WORD*)pBase != 0x5A4D) { // "MZ"
    return ERROR_BAD_FORMAT;
}

// PE Header 검증
DWORD peOffset = *(DWORD*)((BYTE*)pBase + 0x3C); // e_lfanew
if (*(DWORD*)((BYTE*)pBase + peOffset) != 0x4550) { // "PE"
    return ERROR_BAD_FORMAT;
}
```

### 2. Memory Allocation

```cpp
// ImageBase에 메모리 할당 시도
LPVOID pImageBase = VirtualAlloc(
    (LPVOID)pOptHeader->ImageBase,
    pOptHeader->SizeOfImage,
    MEM_RESERVE | MEM_COMMIT,
    PAGE_READWRITE
);

// 실패 시 (ASLR 또는 충돌) 다른 주소에 할당
if (!pImageBase) {
    pImageBase = VirtualAlloc(NULL, ...);
    // → Base Relocation 필요!
}
```

### 3. Section Loading

```cpp
for (int i = 0; i < pFileHeader->NumberOfSections; i++) {
    IMAGE_SECTION_HEADER* pSection = &pSectionHeaders[i];

    // 파일에서 메모리로 복사
    memcpy(
        (BYTE*)pImageBase + pSection->VirtualAddress,
        (BYTE*)pBase + pSection->PointerToRawData,
        pSection->SizeOfRawData
    );

    // 섹션 권한 설정
    DWORD protect = 0;
    if (pSection->Characteristics & IMAGE_SCN_MEM_EXECUTE)
        protect = PAGE_EXECUTE_READ;
    else if (pSection->Characteristics & IMAGE_SCN_MEM_WRITE)
        protect = PAGE_READWRITE;
    else
        protect = PAGE_READONLY;

    VirtualProtect(
        (BYTE*)pImageBase + pSection->VirtualAddress,
        pSection->Misc.VirtualSize,
        protect,
        &oldProtect
    );
}
```

### 4. Import Resolution

```cpp
// Import Directory 순회
IMAGE_IMPORT_DESCRIPTOR* pImport = ...;

while (pImport->Name != 0) {
    // DLL 로딩
    char* dllName = (char*)pImageBase + pImport->Name;
    HMODULE hDll = LoadLibrary(dllName);

    // IAT에 함수 주소 채우기
    IMAGE_THUNK_DATA* pIAT = (IMAGE_THUNK_DATA*)
        ((BYTE*)pImageBase + pImport->FirstThunk);
    IMAGE_THUNK_DATA* pINT = (IMAGE_THUNK_DATA*)
        ((BYTE*)pImageBase + pImport->OriginalFirstThunk);

    while (pINT->u1.AddressOfData != 0) {
        if (pINT->u1.Ordinal & IMAGE_ORDINAL_FLAG) {
            // Ordinal import
            WORD ordinal = IMAGE_ORDINAL(pINT->u1.Ordinal);
            pIAT->u1.Function = (DWORD_PTR)
                GetProcAddress(hDll, (LPCSTR)ordinal);
        } else {
            // Name import
            IMAGE_IMPORT_BY_NAME* pName = (IMAGE_IMPORT_BY_NAME*)
                ((BYTE*)pImageBase + pINT->u1.AddressOfData);
            pIAT->u1.Function = (DWORD_PTR)
                GetProcAddress(hDll, pName->Name);
        }
        pINT++;
        pIAT++;
    }
    pImport++;
}
```

### 5. Base Relocation (필요 시)

ASLR이 활성화되거나 ImageBase가 사용 중일 때 필요합니다.

```cpp
if (pImageBase != pOptHeader->ImageBase) {
    LONGLONG delta = (BYTE*)pImageBase - pOptHeader->ImageBase;

    IMAGE_BASE_RELOCATION* pReloc = ...;

    while (pReloc->VirtualAddress != 0) {
        DWORD numEntries = (pReloc->SizeOfBlock - 8) / 2;
        WORD* pEntry = (WORD*)(pReloc + 1);

        for (DWORD i = 0; i < numEntries; i++) {
            int type = pEntry[i] >> 12;
            int offset = pEntry[i] & 0xFFF;

            if (type == IMAGE_REL_BASED_DIR64) {
                ULONGLONG* pAddr = (ULONGLONG*)
                    ((BYTE*)pImageBase + pReloc->VirtualAddress + offset);
                *pAddr += delta;
            }
        }
        pReloc = (IMAGE_BASE_RELOCATION*)
            ((BYTE*)pReloc + pReloc->SizeOfBlock);
    }
}
```

### 6. TLS Callbacks 실행

```cpp
// TLS (Thread Local Storage) callbacks
if (pDataDir[IMAGE_DIRECTORY_ENTRY_TLS].Size > 0) {
    IMAGE_TLS_DIRECTORY* pTls = ...;
    PIMAGE_TLS_CALLBACK* pCallback =
        (PIMAGE_TLS_CALLBACK*)pTls->AddressOfCallBacks;

    while (*pCallback) {
        (*pCallback)((LPVOID)pImageBase, DLL_PROCESS_ATTACH, NULL);
        pCallback++;
    }
}
```

### 7. Entry Point 실행

```cpp
// 마지막으로 Entry Point 호출
typedef int (WINAPI *ENTRY_POINT)();
ENTRY_POINT pEntry = (ENTRY_POINT)
    ((BYTE*)pImageBase + pOptHeader->AddressOfEntryPoint);

int exitCode = pEntry();
```

### 로딩 과정 요약

```
1. File Mapping       : 파일을 메모리에 매핑
2. Memory Allocation  : ImageBase에 공간 확보
3. Section Loading    : 섹션을 메모리로 복사
4. Import Resolution  : DLL 로딩 & IAT 채우기
5. Base Relocation    : 주소 재배치 (ASLR 시)
6. TLS Callbacks      : 초기화 코드 실행
7. Entry Point        : main() 함수 실행
```

----

## 보안 기능

### ASLR (Address Space Layout Randomization)

**개념:**
- 프로그램을 실행할 때마다 메모리 주소를 랜덤하게 배치
- 공격자가 주소를 예측하기 어렵게 만듦

**활성화 방법:**
```cpp
// Linker 옵션
/DYNAMICBASE

// DllCharacteristics 플래그
IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE (0x0040)
```

**확인:**
```bash
dumpbin /headers app.exe | findstr "Dynamic base"
```

### DEP (Data Execution Prevention)

**개념:**
- 데이터 영역(.data, stack, heap)의 코드 실행 금지
- 버퍼 오버플로우로 주입된 코드 실행 차단

**활성화 방법:**
```cpp
// Linker 옵션
/NXCOMPAT

// DllCharacteristics 플래그
IMAGE_DLLCHARACTERISTICS_NX_COMPAT (0x0100)
```

**확인:**
```bash
dumpbin /headers app.exe | findstr "NX compatible"
```

### CFG (Control Flow Guard)

**개념:**
- 간접 함수 호출의 대상 주소를 검증
- ROP 공격 방어

**활성화 방법:**
```cpp
// Linker 옵션 (Visual Studio 2015+)
/GUARD:CF

// DllCharacteristics 플래그
IMAGE_DLLCHARACTERISTICS_GUARD_CF (0x4000)
```

**확인:**
```bash
dumpbin /headers app.exe | findstr "Guard"
```

----

## 실무 활용

### 시나리오 1: Crash Dump 분석

**문제:**
```
프로그램이 0x00007FF6A2C51234 주소에서 Access Violation
```

**분석 과정:**

```bash
# 1. ImageBase 확인
dumpbin /headers app.exe | findstr "image base"
# 출력: 140000000 image base

# 2. RVA 계산
# RVA = 0x00007FF6A2C51234 - 0x00007FF6A2C50000
# RVA = 0x1234

# 3. 어느 섹션인지 확인
dumpbin /headers app.exe
# 출력:
# .text  VirtualAddress: 0x00001000, Size: 0x00050000
# .data  VirtualAddress: 0x00051000, Size: 0x00010000
# → 0x1234는 .text 섹션 (코드 영역)
```

### 시나리오 2: DLL Hell 해결

**문제:**
```
"The procedure entry point XXX could not be located in the DLL YYY.dll"
```

**해결 과정:**

```bash
# 1. 필요한 DLL 버전 확인
dumpbin /imports app.exe | findstr YYY.dll

# 2. Export 비교
dumpbin /exports C:\Windows\System32\YYY.dll | findstr XXX
dumpbin /exports C:\MyApp\YYY.dll | findstr XXX

# 3. 해결:
# - 올바른 DLL을 app.exe와 같은 폴더에 배포
```

### 시나리오 3: 32bit/64bit 혼용 오류

**문제:**
```
"Bad Image" 오류 또는 0xc000007b
```

**진단:**

```bash
# app.exe 아키텍처 확인
dumpbin /headers app.exe | findstr machine
# 출력: 14C machine (x86)

# DLL 아키텍처 확인
dumpbin /headers somelib.dll | findstr machine
# 출력: 8664 machine (x64)

# → 아키텍처 불일치!
```

**해결:**
- 모두 x86으로 또는 모두 x64로 통일

----

## 유용한 도구

### dumpbin (Visual Studio 내장)

**기본 사용:**
```bash
# 전체 요약
dumpbin /summary app.exe

# Headers
dumpbin /headers app.exe

# Import table
dumpbin /imports app.exe

# Export table (DLL)
dumpbin /exports mydll.dll

# Dependencies
dumpbin /dependents app.exe
```

### Dependency Walker

**다운로드:** http://www.dependencywalker.com/

**사용 시나리오:**
1. Missing DLL 찾기 (빨간색 = 없는 DLL)
2. Function not found 진단
3. Architecture 불일치 확인

### CFF Explorer

**다운로드:** https://ntcore.com/?page_id=388

**주요 기능:**
1. Header 편집
2. Import 수정
3. Resource 편집
4. Digital signature 확인

### Python pefile

**설치:**
```bash
pip install pefile
```

**사용 예제:**
```python
import pefile

pe = pefile.PE('app.exe')

# Header 정보
print(f"Entry Point: {hex(pe.OPTIONAL_HEADER.AddressOfEntryPoint)}")
print(f"ImageBase: {hex(pe.OPTIONAL_HEADER.ImageBase)}")

# Sections
for section in pe.sections:
    print(f"{section.Name.decode().strip('\\x00')}: "
          f"VA={hex(section.VirtualAddress)}")

# Security features
dll_characteristics = pe.OPTIONAL_HEADER.DllCharacteristics
print(f"ASLR: {bool(dll_characteristics & 0x0040)}")
print(f"DEP:  {bool(dll_characteristics & 0x0100)}")
```

----

## 트러블슈팅

### 1. "This program cannot be run in DOS mode"

**원인:**
- PE가 아닌 파일을 실행
- 파일 손상

**진단:**
```bash
# 첫 2바이트 확인
xxd -l 2 file.exe
# 출력: 4d 5a (MZ) 가 아니면 손상됨
```

**해결:**
- 파일 재다운로드
- 바이너리 모드로 전송

### 2. "The application was unable to start correctly (0xc000007b)"

**원인:**
- 32bit/64bit 불일치
- Missing MSVC runtime

**해결:**
```bash
# 아키텍처 확인
dumpbin /headers app.exe | findstr machine
dumpbin /headers dependency.dll | findstr machine

# Visual C++ Redistributable 설치
```

### 3. Missing DLL 오류

**증상:**
```
"VCRUNTIME140.dll was not found"
```

**해결:**
```bash
# 방법 1: Local 배포 (권장)
# 필요한 DLL을 exe와 같은 폴더에 복사

# 방법 2: Dependency 확인
dumpbin /dependents app.exe

# 방법 3: Static linking
# /MT 옵션으로 재컴파일
```

### 4. Import by ordinal 실패

**증상:**
```
"Ordinal 123 not found in YYY.dll"
```

**해결:**
- Name import 사용 (권장)
- 올바른 DLL 버전 사용

### 5. Base relocation 오류

**증상:**
- ASLR 활성화 시 crash

**해결:**
```cpp
// Linker 옵션 수정
/FIXED:NO            // Relocation 정보 포함
/DYNAMICBASE         // ASLR 지원
```

----

## 참고 자료

### 공식 문서

- [PE Format Specification @ MSDN](https://learn.microsoft.com/en-us/windows/win32/debug/pe-format)
- [Peering Inside the PE @ MSDN](https://msdn.microsoft.com/en-us/library/ms809762.aspx)

### 추천 도구

- [CFF Explorer](https://ntcore.com/?page_id=388) - PE Editor
- [Dependency Walker](http://www.dependencywalker.com/) - Dependency Analyzer
- [Process Explorer](https://learn.microsoft.com/en-us/sysinternals/downloads/process-explorer) - Runtime Analysis
- [PEDUMP @ github](https://github.com/martell/pedump) - PE Dumper

### 학습 자료

**입문:**
- [corkami/pics @ github](https://github.com/corkami/pics/tree/master/binary) - PE 구조 다이어그램

**중급:**
- [PE @ tistory](http://www.reversecore.com/25?category=216978) - "리버싱 핵심 기술"의 저자가 설명한 PE

**고급:**
- [Malware Theory - Memory Mapping of PE Files](https://www.youtube.com/watch?v=cc1tX1t_bLg&list=PLynb9SXC4yETaQYYBSg696V77Ku8TOM8-&index=3) - PE가 Virtual Memory에 어떻게 매핑되는지

### Python 라이브러리

```bash
# PE 파싱
pip install pefile

# PE 수정
pip install lief
```

----

## 부록: 빠른 참조

### 주소 계산 공식

```
VA (Virtual Address) = ImageBase + RVA
RVA (Relative VA) = VA - ImageBase
RAW (File Offset) = RVA - VirtualAddress + PointerToRawData
```

### dumpbin 치트시트

```bash
# 전체 정보
dumpbin /all app.exe

# 특정 정보만
dumpbin /headers app.exe       # Headers
dumpbin /imports app.exe        # Import table
dumpbin /exports app.exe        # Export table
dumpbin /dependents app.exe     # Dependencies

# 필터링
dumpbin /headers app.exe | findstr machine
dumpbin /headers app.exe | findstr "DLL characteristics"
```

### 보안 플래그 체크

```bash
# 최소 보안 요구사항 확인
dumpbin /headers app.exe | findstr /C:"Dynamic base" /C:"NX compatible"

# 출력 예시 (안전):
       Dynamic base
       NX compatible
```
