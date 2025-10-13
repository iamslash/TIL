- [상황 요약](#상황-요약)
  - [1) `runtime.GC()` → 마킹 준비 \& 루트 작업 등록](#1-runtimegc--마킹-준비--루트-작업-등록)
  - [2) 루트·스택·글로벌 스캔 → “회색 큐(grey queue)”에 푸시](#2-루트스택글로벌-스캔--회색-큐grey-queue에-푸시)
  - [3) 마킹 중에도 안전: **하이브리드 write barrier** (삭제+삽입)](#3-마킹-중에도-안전-하이브리드-write-barrier-삭제삽입)
  - [4) 마킹 종료 → **스윕(sweep)** 으로 미표시(white) 회수](#4-마킹-종료--스윕sweep-으로-미표시white-회수)
  - [5) “왜 순환인데도 수거되나?” — 개념 정리](#5-왜-순환인데도-수거되나--개념-정리)
  - [6) 코드 레벨에서 “당신의 예제”에 매핑](#6-코드-레벨에서-당신의-예제에-매핑)
  - [한 줄 결론](#한-줄-결론)

-----

아래는 **코드**에서 만들어진 원형(cycle)이 **Go 1.24 런타임의 실제 GC 코드 흐름**을 따라 **어떻게 수거되는지**를 단계별로 설명한 것입니다. 포인트는 “**루트에서 도달 가능한 것만 산다**”는 추적형(tracing) 마크-스윕의 원칙과, 마킹 도중에도 안전성을 보장하는 **하이브리드 write barrier** 입니다.

---

# 상황 요약 

```go
// ...생략...
box1.next = box2
box2.next = box3
box3.next = box1   // <- 순환 형성

fmt.Println(box2.content)
fmt.Println(box3.content)

box1, box2, box3 = nil  // <- 모든 외부 루트(스택 변수) 참조 제거
runtime.GC()            // <- 데모용 강제 GC
```

이 시점에서 힙에는 `(box1)->(box2)->(box3)->(box1)` **사이클만 남아** 있고, **스택/전역에 그 사이클을 가리키는 포인터가 전혀 없습니다.** 따라서 루트에서 이 사이클로 “첫 발을” 디딜 수 없으므로 **전부 가비지**가 됩니다. 이 결론이 어떻게 런타임 코드에서 구현되는지 아래 단계로 보세요.

---

## 1) `runtime.GC()` → 마킹 준비 & 루트 작업 등록

강제 GC는 짧은 STW로 **마킹 모드 전환, write barrier 활성화, 루트 마킹 잡 등록**을 수행합니다(“루트”는 전역/스택/특수 스팬 등). 이 준비 단계는 `mgc.go`에 요약되어 있습니다. ([tip.golang.org][1])

* 요지: “마크 단계 준비 → write barrier 켜기 → 루트 마크 job 큐잉. 모든 P가 배리어를 켤 때까지 STW.” ([tip.golang.org][1])

당신의 코드에서는 이미 `box1, box2, box3 = nil`로 **루트에 사이클 진입점이 없습니다.** 즉, 루트 스캔이 시작돼도 이 사이클을 가리키는 최초의 회색화(회색 큐 등록)가 발생하지 않습니다.

---

## 2) 루트·스택·글로벌 스캔 → “회색 큐(grey queue)”에 푸시

마킹 워커는 루트에서 시작해 포인터를 따라가며 **흰→회색→검은**으로 칠합니다. **객체 스캔의 핵심 루틴**은 다음과 같습니다.

* `scanObject(b, gcw)`
  객체의 타입 포인터 맵을 따라 필드들을 훑고, 힙 포인터를 발견하면
  `findObject(...)`로 실제 힙 객체인지 식별 → **`greyobject(...)`** 로 **회색 큐**에 넣음.
  (큰 객체는 oblet 단위로 분할) ([tip.golang.org][2])

해당 소스의 실제 라인(요약):

* `scanObject` 내부에서 `findObject`로 대상 확인 후 `greyobject(obj, ...)` 호출. ([tip.golang.org][2])

> 당신의 예제에서는 루트 스캔 단계에서 **그 사이클로 들어가는 “첫 객체”가 회색화되지 않습니다.**
> 이유: 지역 변수 `box1/2/3`를 이미 `nil`로 만들었으므로 **루트 그래프에 진입점이 없기 때문**입니다.

---

## 3) 마킹 중에도 안전: **하이브리드 write barrier** (삭제+삽입)

마킹이 진행되는 동안에도 사용자 코드(뮤테이터)가 포인터를 바꿀 수 있으므로, 컴파일러가 삽입한 write barrier가 정확성을 보장합니다. 런타임 `mbarrier.go`의 주석/유사코드가 **Go의 하이브리드 배리어**(Yuasa 삭제 + Dijkstra 삽입)를 명확히 설명합니다:

```go
writePointer(slot, ptr):
    shade(*slot)            // 삭제 배리어: 기존 대상 음영화
    if current stack is grey:
        shade(ptr)          // 삽입 배리어: 새 대상 음영화 (스택이 회색일 때)
    *slot = ptr             // 실제 쓰기
```

* 목적: 마킹 도중 포인터 이동으로 인해 “루트에서 보이던 객체가 숨어버리는” 것을 막음.
* 런타임 주석에 이 논리와 안전성(메모리 오더링 포함)이 자세히 적혀 있습니다. ([tip.golang.org][3])

> 당신의 케이스에서는 **루트 끊김이 이미 끝난 뒤** `runtime.GC()`를 호출하므로, 배리어가 개입해 사이클을 구해낼 기회 자체가 거의 없습니다(새 포인터 출현이 없기 때문). 즉, **사이클은 계속 “미도달(white)” 상태**로 남습니다.

---

## 4) 마킹 종료 → **스윕(sweep)** 으로 미표시(white) 회수

마킹이 끝나면 스윕 단계가 시작되고, 여전히 **흰색으로 남은 객체**를 포함한 스팬을 훑어 **살아있는 비트가 없는 객체(또는 스팬 전체)** 를 회수합니다. 스윕의 진입과 실행은 `mgcsweep.go`/`mspan.sweep` 경로에 정리돼 있습니다. ([tip.golang.org][4])

* 문서 요지: “스윕은 결국 **`mspan.sweep`** 이 단위 작업이며, 배경/비례 스위퍼가 스팬을 하나씩 쓸어간다.” ([tip.golang.org][4])

> 결과: 사이클을 이루던 `Box` 3개는 **루트 미도달 → 마킹되지 않음 → 스윕에서 회수**.

---

## 5) “왜 순환인데도 수거되나?” — 개념 정리

Go 1.24의 GC는 **동시 tri-color 마크-스윕**으로, **“도달성(reachability)” 기준**입니다.
**순환이든 아니든 루트에서 닿지 않으면(첫 회색화가 없으면) 끝까지 white → 스윕으로 해제**됩니다. 이 원칙은 공식 **GC 가이드**에도 요약되어 있습니다. ([tip.golang.org][5])

---

## 6) 코드 레벨에서 “당신의 예제”에 매핑

1. `box3.next = box1` 로 **사이클 형성**

   * 마킹 중이었다면 write barrier가 **shade(\*slot)**/**shade(ptr)** 로 안전을 보장합니다. ([tip.golang.org][3])

2. `box1, box2, box3 = nil` 로 **루트 제거**

   * 이제 스택/글로벌 등 어떤 루트에서도 사이클 내부 객체로 **첫 진입이 불가**.

3. `runtime.GC()`

   * 마킹 준비/루트 큐잉(Write barrier on) → 루트/스택/글로벌 스캔에서 **해당 사이클로 진입 불가** →
     마킹 워커의 `scanObject`는 루트에서 회색화된 객체들만 따라가며 `findObject`→`greyobject` 호출(당신의 사이클은 여기에 오르지 못함). ([tip.golang.org][2])

4. **스윕**

   * 흰색으로 남은 사이클 객체들이 속한 스팬을 `mspan.sweep` 경로에서 회수. ([tip.golang.org][4])

---

## 한 줄 결론

**순환 자체는 문제가 아닙니다.** Go 1.24의 **추적형 GC**는 **루트에서 도달 불가한 전체 사이클**을 **표식되지 않은(white)** 상태로 남겨 두었다가 **스윕 단계에서 정상적으로 회수**합니다. ([tip.golang.org][5])

원하면 `GODEBUG=gctrace=1`을 켜고 이 코드로 **마킹/스윕 로그**를 읽는 법까지 같이 보여줄게요(“heap goal/heap live, mark termination, sweep” 순으로 출력).

[1]: https://tip.golang.org/src/runtime/mgc.go?utm_source=chatgpt.com "- The Go Programming Language"
[2]: https://tip.golang.org/src/runtime/mgcwork.go?utm_source=chatgpt.com "- The Go Programming Language"
[3]: https://tip.golang.org/src/runtime/mbarrier.go " - The Go Programming Language"
[4]: https://tip.golang.org/src/runtime/mgcsweep.go?utm_source=chatgpt.com "- The Go Programming Language"
[5]: https://tip.golang.org/doc/gc-guide?utm_source=chatgpt.com "A Guide to the Go Garbage Collector"
