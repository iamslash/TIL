
- [Go 고루틴 관리: M, G, P 완전 이해](#go-고루틴-관리-m-g-p-완전-이해)
  - [**1. M, G, P가 무엇인가?**](#1-m-g-p가-무엇인가)
    - [**G (Goroutine) - 고루틴**](#g-goroutine---고루틴)
    - [**M (Machine) - 머신**](#m-machine---머신)
    - [**P (Processor) - 프로세서**](#p-processor---프로세서)
  - [**2. 아주 쉬운 예제 코드**](#2-아주-쉬운-예제-코드)
  - [**3. 실행 과정 단계별 분석**](#3-실행-과정-단계별-분석)
    - [**1단계: 프로그램 시작**](#1단계-프로그램-시작)
    - [**2단계: 첫 번째 고루틴 생성**](#2단계-첫-번째-고루틴-생성)
    - [**3단계: 두 번째 고루틴 생성**](#3단계-두-번째-고루틴-생성)
    - [**4단계: 스케줄링 시작**](#4단계-스케줄링-시작)
    - [**5단계: 고루틴 전환**](#5단계-고루틴-전환)
  - [**4. 실제 메모리 구조**](#4-실제-메모리-구조)
    - [**M, G, P 구조체**](#m-g-p-구조체)
  - [**5. 스케줄링 과정**](#5-스케줄링-과정)
    - [**스케줄링 루프**](#스케줄링-루프)
    - [**고루틴 실행**](#고루틴-실행)
  - [**6. 실제 동작 시뮬레이션**](#6-실제-동작-시뮬레이션)
    - [**시간순 실행 과정**](#시간순-실행-과정)
  - [**7. 고루틴 상태 전환**](#7-고루틴-상태-전환)
    - [**고루틴 상태**](#고루틴-상태)
    - [**상태 전환 과정**](#상태-전환-과정)
  - [**8. 실제 확인 방법**](#8-실제-확인-방법)
    - [**런타임 정보 확인**](#런타임-정보-확인)
- [요약](#요약)
  - [**M, G, P의 역할**](#m-g-p의-역할)
  - [**고루틴 관리 과정**](#고루틴-관리-과정)
  - [**핵심 포인트**](#핵심-포인트)

-----

Go 고루틴이 어떻게 관리되는지 M, G, P를 이용해 아주 쉬운 예제로 설명해드리겠습니다.

## Go 고루틴 관리: M, G, P 완전 이해

### **1. M, G, P가 무엇인가?**

#### **G (Goroutine) - 고루틴**
```go
// G는 실행할 작업 단위
// 함수처럼 실행되는 코드 블록
// 매우 가벼움 (2KB 스택)

// 예시:
go func() {
    fmt.Println("Hello from goroutine!")
}()
```

#### **M (Machine) - 머신**
```go
// M은 OS 스레드와 1:1 매핑
// 실제로 고루틴을 실행하는 실행기
// CPU 코어에서 실제 작업 수행

// 예시:
// M0 ←→ OS Thread 0
// M1 ←→ OS Thread 1
```

#### **P (Processor) - 프로세서**
```go
// P는 논리적 프로세서
// 고루틴 실행 큐를 관리
// 메모리 할당 캐시를 가짐

// 예시:
// P0: [G1, G2, G3] 실행 큐
// P1: [G4, G5, G6] 실행 큐
```

### **2. 아주 쉬운 예제 코드**

```go
package main

import (
    "fmt"
    "time"
)

func main() {
    // 메인 고루틴 (G0)
    fmt.Println("Main goroutine started")
    
    // 새로운 고루틴 생성 (G1)
    go func() {
        fmt.Println("Goroutine 1: Hello!")
        time.Sleep(1 * time.Second)
        fmt.Println("Goroutine 1: Done!")
    }()
    
    // 또 다른 고루틴 생성 (G2)
    go func() {
        fmt.Println("Goroutine 2: World!")
        time.Sleep(2 * time.Second)
        fmt.Println("Goroutine 2: Done!")
    }()
    
    // 메인 고루틴 대기
    time.Sleep(3 * time.Second)
    fmt.Println("Main goroutine finished")
}
```

### **3. 실행 과정 단계별 분석**

#### **1단계: 프로그램 시작**
```go
// 프로그램 시작 시
// M0 (메인 머신) 생성
// P0 (메인 프로세서) 생성
// G0 (메인 고루틴) 생성

// 초기 상태:
M0 → P0 → G0 (main 함수)
```

#### **2단계: 첫 번째 고루틴 생성**
```go
go func() {
    fmt.Println("Goroutine 1: Hello!")
    time.Sleep(1 * time.Second)
    fmt.Println("Goroutine 1: Done!")
}()

// G1 생성 후:
M0 → P0 → G0 (main 함수)
P0.runq = [G1]  // P0의 실행 큐에 G1 추가
```

#### **3단계: 두 번째 고루틴 생성**
```go
go func() {
    fmt.Println("Goroutine 2: World!")
    time.Sleep(2 * time.Second)
    fmt.Println("Goroutine 2: Done!")
}()

// G2 생성 후:
M0 → P0 → G0 (main 함수)
P0.runq = [G1, G2]  // P0의 실행 큐에 G2 추가
```

#### **4단계: 스케줄링 시작**
```go
// M0이 P0을 가져와서 고루틴 실행
// P0의 실행 큐에서 G1 선택

// 실행 중:
M0 → P0 → G1 (첫 번째 고루틴 실행)
P0.runq = [G2]  // G1이 실행 중이므로 큐에서 제거
```

#### **5단계: 고루틴 전환**
```go
// G1이 time.Sleep()으로 블록됨
// M0이 P0을 가져와서 G2 실행

// 실행 중:
M0 → P0 → G2 (두 번째 고루틴 실행)
P0.runq = []  // G2가 실행 중이므로 큐가 비어있음
```

### **4. 실제 메모리 구조**

#### **M, G, P 구조체**
```go
// M (Machine) 구조체
type m struct {
    mid     int64    // 머신 ID
    curg    *g       // 현재 실행 중인 고루틴
    p       *p       // 연결된 프로세서
    // ... 기타 필드들
}

// G (Goroutine) 구조체
type g struct {
    goid    int64    // 고루틴 ID
    status  uint32   // 상태 (running, waiting, etc.)
    stack   stack    // 스택
    // ... 기타 필드들
}

// P (Processor) 구조체
type p struct {
    pid     int32    // 프로세서 ID
    runq    [256]guintptr  // 실행 큐
    runqhead uint32  // 큐 헤드
    runqtail uint32  // 큐 테일
    // ... 기타 필드들
}
```

### **5. 스케줄링 과정**

#### **스케줄링 루프**
```go
// M이 실행하는 스케줄링 루프
func schedule() {
    for {
        // 1. P를 가져옴
        _p_ := getp()
        
        // 2. P의 실행 큐에서 고루틴 선택
        gp := runqget(_p_)
        
        // 3. 고루틴이 없으면 전역 큐에서 가져옴
        if gp == nil {
            gp = globrunqget(_p_, 0)
        }
        
        // 4. 고루틴 실행
        if gp != nil {
            execute(gp, false)
        }
        
        // 5. 루프 반복
    }
}
```

#### **고루틴 실행**
```go
// 고루틴 실행
func execute(gp *g, inheritTime bool) {
    // 1. 고루틴 상태 변경
    gp.status = _Grunning
    
    // 2. 고루틴 실행
    gogo(&gp.sched)
    
    // 3. 고루틴이 완료되면 루프로 돌아감
}
```

### **6. 실제 동작 시뮬레이션**

#### **시간순 실행 과정**
```go
// t=0: 프로그램 시작
M0 → P0 → G0 (main 함수 시작)
P0.runq = []

// t=1: 첫 번째 고루틴 생성
M0 → P0 → G0 (go func() 호출)
P0.runq = [G1]

// t=2: 두 번째 고루틴 생성
M0 → P0 → G0 (go func() 호출)
P0.runq = [G1, G2]

// t=3: G1 실행 시작
M0 → P0 → G1 (첫 번째 고루틴 실행)
P0.runq = [G2]

// t=4: G1이 time.Sleep()으로 블록
M0 → P0 → G2 (두 번째 고루틴 실행)
P0.runq = []

// t=5: G2가 time.Sleep()으로 블록
M0 → P0 → G0 (main 함수 계속 실행)
P0.runq = []

// t=6: G1 완료
M0 → P0 → G0 (main 함수 계속 실행)
P0.runq = []

// t=7: G2 완료
M0 → P0 → G0 (main 함수 계속 실행)
P0.runq = []

// t=8: main 함수 완료
프로그램 종료
```

### **7. 고루틴 상태 전환**

#### **고루틴 상태**
```go
const (
    _Gidle    = iota  // 유휴 상태
    _Grunnable        // 실행 가능
    _Grunning         // 실행 중
    _Gsyscall         // 시스템 콜 중
    _Gwaiting         // 대기 중
    _Gdead            // 종료됨
)
```

#### **상태 전환 과정**
```go
// 고루틴 생성
G1: _Gidle → _Grunnable

// 고루틴 실행
G1: _Grunnable → _Grunning

// time.Sleep() 호출
G1: _Grunning → _Gwaiting

// time.Sleep() 완료
G1: _Gwaiting → _Grunnable

// 고루틴 완료
G1: _Grunning → _Gdead
```

### **8. 실제 확인 방법**

#### **런타임 정보 확인**
```go
import "runtime"

func printRuntimeInfo() {
    fmt.Printf("Goroutines: %d\n", runtime.NumGoroutine())
    fmt.Printf("CPU cores: %d\n", runtime.NumCPU())
    fmt.Printf("GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))
}
```

## 요약

### **M, G, P의 역할**
- **G (Goroutine)**: 실행할 작업 단위
- **M (Machine)**: 실제로 고루틴을 실행하는 실행기
- **P (Processor)**: 고루틴 실행 큐를 관리하는 논리적 프로세서

### **고루틴 관리 과정**
1. **고루틴 생성**: P의 실행 큐에 추가
2. **스케줄링**: M이 P를 가져와서 고루틴 실행
3. **상태 전환**: 고루틴 상태에 따라 스케줄링
4. **완료**: 고루틴이 완료되면 큐에서 제거

### **핵심 포인트**
- **M이 실제 실행기**
- **P가 실행 큐 관리**
- **G가 실행할 작업**
- **스케줄링 루프가 고루틴 관리**

**결론**: Go 고루틴은 **M, G, P 모델**로 관리되며, **M이 P를 가져와서 P의 큐에서 G를 선택하여 실행**하는 구조입니다.
