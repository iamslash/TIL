- [Go 런타임이 시작해서 `main` 함수를 호출하기까지의 과정](#go-런타임이-시작해서-main-함수를-호출하기까지의-과정)
  - [Go 런타임 시작부터 main() 호출까지](#go-런타임-시작부터-main-호출까지)
    - [**1. 프로그램 시작 (OS 레벨)**](#1-프로그램-시작-os-레벨)
      - [**OS가 프로그램 로딩**](#os가-프로그램-로딩)
      - [**진입점 (Entry Point)**](#진입점-entry-point)
    - [**2. 런타임 초기화 시작**](#2-런타임-초기화-시작)
      - [**rt0\_go 함수**](#rt0_go-함수)
    - [**3. 인수 처리**](#3-인수-처리)
      - [**args 함수**](#args-함수)
    - [**4. OS 초기화**](#4-os-초기화)
      - [**osinit 함수**](#osinit-함수)
    - [**5. 스케줄러 초기화**](#5-스케줄러-초기화)
      - [**schedinit 함수**](#schedinit-함수)
      - [**procresize 함수**](#procresize-함수)
    - [**6. 메인 고루틴 생성**](#6-메인-고루틴-생성)
      - [**newproc 함수**](#newproc-함수)
      - [**newproc1 함수**](#newproc1-함수)
    - [**7. 스케줄링 시작**](#7-스케줄링-시작)
      - [**mstart 함수**](#mstart-함수)
      - [**mstart1 함수**](#mstart1-함수)
    - [**8. 스케줄링 루프**](#8-스케줄링-루프)
      - [**schedule 함수**](#schedule-함수)
    - [**9. 메인 고루틴 실행**](#9-메인-고루틴-실행)
      - [**execute 함수**](#execute-함수)
      - [**gogo 함수**](#gogo-함수)
    - [**10. main 함수 호출**](#10-main-함수-호출)
      - [**main 함수 진입**](#main-함수-진입)
  - [요약](#요약)
    - [**전체 과정**](#전체-과정)
    - [**핵심 함수들**](#핵심-함수들)

-----

# Go 런타임이 시작해서 `main` 함수를 호출하기까지의 과정

## Go 런타임 시작부터 main() 호출까지

### **1. 프로그램 시작 (OS 레벨)**

#### **OS가 프로그램 로딩**
```bash
# OS가 a.out 실행
./a.out

# OS가 프로세스 생성
# PID: 12345
# 가상 메모리 공간 할당
# 코드 세그먼트, 데이터 세그먼트 로딩
```

#### **진입점 (Entry Point)**
```go
// runtime/rt0_linux_amd64.s (어셈블리 코드)
TEXT _rt0_amd64_linux(SB),NOSPLIT,$-8
    MOVQ    0(SP), DI  // argc
    LEAQ    8(SP), SI  // argv
    JMP     runtime·rt0_go(SB)
```

### **2. 런타임 초기화 시작**

#### **rt0_go 함수**
```go
// runtime/asm_amd64.s
TEXT runtime·rt0_go(SB),NOSPLIT,$0
    // 1. argc, argv 설정
    MOVQ    DI, AX     // argc
    MOVQ    SI, BX     // argv
    
    // 2. 스택 설정
    MOVQ    SP, BP
    
    // 3. 런타임 초기화 호출
    CALL    runtime·args(SB)
    CALL    runtime·osinit(SB)
    CALL    runtime·schedinit(SB)
    
    // 4. 메인 고루틴 생성
    CALL    runtime·newproc(SB)
    
    // 5. 스케줄링 시작
    CALL    runtime·mstart(SB)
```

### **3. 인수 처리**

#### **args 함수**
```go
// runtime/runtime1.go
func args(c int32, v **byte) {
    argc = c
    argv = v
    sysargs(c, v)
}
```

### **4. OS 초기화**

#### **osinit 함수**
```go
// runtime/os_linux.go
func osinit() {
    // 1. CPU 코어 수 확인
    ncpu = getproccount()
    
    // 2. 페이지 크기 설정
    physPageSize = getPageSize()
    
    // 3. OS별 초기화
    osArchInit()
}
```

### **5. 스케줄러 초기화**

#### **schedinit 함수**
```go
// runtime/proc.go
func schedinit() {
    // 1. 스케줄러 락 초기화
    lockInit(&sched.lock, lockRankSched)
    
    // 2. 고루틴 ID 초기화
    sched.lastpoll = uint64(nanotime())
    
    // 3. P 개수 설정
    procs := ncpu
    if n, ok := atoi32(gogetenv("GOMAXPROCS")); ok && n > 0 {
        procs = n
    }
    procresize(procs)
    
    // 4. 메인 고루틴 초기화
    mainStarted = true
}
```

#### **procresize 함수**
```go
// runtime/proc.go
func procresize(nprocs int32) *p {
    // 1. P 배열 크기 조정
    old := gomaxprocs
    gomaxprocs = nprocs
    
    // 2. P 생성
    for i := int32(0); i < nprocs; i++ {
        pp := allp[i]
        if pp == nil {
            pp = new(p)
            pp.id = i
            pp.status = _Pgcstop
            pp.sudogcache = pp.sudogbuf[:0]
            for i := range pp.deferpool {
                pp.deferpool[i] = pp.deferpoolbuf[i][:0]
            }
            pp.wbBuf.reset()
            atomicstorep(unsafe.Pointer(&allp[i]), unsafe.Pointer(pp))
        }
    }
    
    // 3. P 상태 설정
    for i := int32(0); i < nprocs; i++ {
        p := allp[i]
        p.status = _Pidle
        runqinit(p)
    }
    
    return runnablePs[0]
}
```

### **6. 메인 고루틴 생성**

#### **newproc 함수**
```go
// runtime/proc.go
func newproc(fn *funcval) {
    // 1. 현재 고루틴 가져오기
    gp := getg()
    pc := getcallerpc()
    
    // 2. 새로운 고루틴 생성
    systemstack(func() {
        newg := newproc1(fn, gp, pc)
        
        // 3. P에 고루틴 추가
        _p_ := getg().m.p.ptr()
        runqput(_p_, newg, true)
        
        // 4. 스케줄러 깨우기
        if atomic.Load(&sched.npidle) != 0 && atomic.Load(&sched.nmspinning) == 0 {
            wakep()
        }
    })
}
```

#### **newproc1 함수**
```go
// runtime/proc.go
func newproc1(fn *funcval, callergp *g, callerpc uintptr) *g {
    // 1. 고루틴 구조체 할당
    _p_ := getg().m.p.ptr()
    newg := gfget(_p_)
    if newg == nil {
        newg = malg(_StackMin)
        casgstatus(newg, _Gidle, _Gdead)
        allgadd(newg)
    }
    
    // 2. 고루틴 초기화
    totalSize := 4*sys.RegSize + sys.MinFrameSize
    totalSize += -totalSize & (sys.SpAlign - 1)
    sp := newg.stack.hi - totalSize
    
    // 3. 스택 설정
    newg.sched.sp = sp
    newg.sched.pc = funcPC(goexit) + sys.PCQuantum
    newg.sched.g = guintptr(unsafe.Pointer(newg))
    gostartcallfn(&newg.sched, fn)
    newg.gopc = callerpc
    newg.ancestors = saveAncestors(callergp)
    newg.startpc = fn.fn
    
    // 4. 상태 설정
    casgstatus(newg, _Gdead, _Grunnable)
    
    return newg
}
```

### **7. 스케줄링 시작**

#### **mstart 함수**
```go
// runtime/proc.go
func mstart() {
    // 1. M 초기화
    _g_ := getg()
    
    // 2. 스택 설정
    osStack := _g_.stack.lo
    if osStack == 0 {
        osStack = _g_.stack.hi
    }
    
    // 3. 스케줄링 루프 시작
    mstart1()
    
    // 4. M 종료
    mexit(osStack)
}
```

#### **mstart1 함수**
```go
// runtime/proc.go
func mstart1() {
    // 1. M 초기화
    _g_ := getg()
    
    // 2. P 가져오기
    _p_ := _g_.m.p.ptr()
    
    // 3. 스케줄링 루프 시작
    schedule()
}
```

### **8. 스케줄링 루프**

#### **schedule 함수**
```go
// runtime/proc.go
func schedule() {
    // 1. 현재 고루틴 가져오기
    _g_ := getg()
    
    // 2. P 가져오기
    _p_ := _g_.m.p.ptr()
    
    // 3. 고루틴 선택
    var gp *g
    var inheritTime bool
    
    // 4. P의 로컬 큐에서 고루틴 선택
    if gp == nil {
        gp, inheritTime = runqget(_p_)
    }
    
    // 5. 전역 큐에서 고루틴 선택
    if gp == nil {
        gp, inheritTime = findrunnable()
    }
    
    // 6. 고루틴 실행
    if gp != nil {
        execute(gp, inheritTime)
    }
}
```

### **9. 메인 고루틴 실행**

#### **execute 함수**
```go
// runtime/proc.go
func execute(gp *g, inheritTime bool) {
    // 1. 고루틴 상태 변경
    _g_ := getg()
    _p_ := _g_.m.p.ptr()
    
    casgstatus(gp, _Grunnable, _Grunning)
    gp.waitsince = 0
    gp.preempt = false
    gp.stackguard0 = gp.stack.lo + _StackGuard
    
    // 2. 고루틴 실행
    gogo(&gp.sched)
}
```

#### **gogo 함수**
```go
// runtime/asm_amd64.s
TEXT runtime·gogo(SB),NOSPLIT,$0-8
    MOVQ    buf+0(FP), BX    // gobuf
    MOVQ    gobuf_g(BX), DX
    MOVQ    0(DX), CX        // make sure g != nil
    JMP     gogo<>(SB)

TEXT gogo<>(SB),NOSPLIT,$0
    get_tls(CX)
    MOVQ    DX, g(CX)
    MOVQ    DX, R14          // set the g register
    MOVQ    gobuf_sp(BX), SP // restore SP
    MOVQ    gobuf_ret(BX), AX
    MOVQ    gobuf_ctxt(BX), DX
    MOVQ    gobuf_bp(BX), BP
    MOVQ    $0, gobuf_sp(BX) // clear to help garbage collector
    MOVQ    $0, gobuf_ret(BX)
    MOVQ    $0, gobuf_ctxt(BX)
    MOVQ    $0, gobuf_bp(BX)
    MOVQ    gobuf_pc(BX), BX
    JMP     BX
```

### **10. main 함수 호출**

#### **main 함수 진입**
```go
// 사용자 코드
func main() {
    // 1. 스택 프레임 생성
    // 2. 변수 할당
    a := 1
    b := new(int)
    *b = 2
    
    // 3. fmt.Println 호출
    fmt.Println("Hello World")
}
```

## 요약

### **전체 과정**
1. **OS가 프로그램 로딩**
2. **rt0_go 진입점 실행**
3. **args, osinit, schedinit 호출**
4. **P 생성 및 초기화**
5. **메인 고루틴 생성**
6. **스케줄링 루프 시작**
7. **메인 고루틴 실행**
8. **main() 함수 호출**

### **핵심 함수들**
- **rt0_go**: 런타임 진입점
- **schedinit**: 스케줄러 초기화
- **newproc**: 고루틴 생성
- **mstart**: 스케줄링 시작
- **schedule**: 스케줄링 루프
- **execute**: 고루틴 실행

**결론**: Go 런타임은 **OS 진입점부터 시작**하여 **스케줄러 초기화**, **고루틴 생성**, **스케줄링 루프 시작**을 거쳐 **main() 함수를 호출**합니다.
