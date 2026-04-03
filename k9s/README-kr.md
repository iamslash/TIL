# k9s 한글 가이드

- [개요](#개요)
- [학습 자료](#학습-자료)
- [설치 및 실행](#설치-및-실행)
- [화면 구성](#화면-구성)
- [리소스 이동 (Command Mode)](#리소스-이동-command-mode)
- [리소스 조작](#리소스-조작)
  - [조회/탐색](#조회탐색)
  - [로그](#로그)
  - [접속/실행](#접속실행)
  - [편집/삭제](#편집삭제)
  - [포트 포워딩](#포트-포워딩)
  - [정렬/필터](#정렬필터)
- [일반 조작](#일반-조작)
- [네비게이션](#네비게이션)
- [자주 쓰는 시나리오](#자주-쓰는-시나리오)
  - [Pod 이 CrashLoopBackOff 일 때](#pod-이-crashloopbackoff-일-때)
  - [특정 서비스에 로컬에서 접속하고 싶을 때](#특정-서비스에-로컬에서-접속하고-싶을-때)
  - [Deployment 이미지를 업데이트할 때](#deployment-이미지를-업데이트할-때)
  - [여러 Pod 을 한꺼번에 삭제할 때](#여러-pod-을-한꺼번에-삭제할-때)
  - [문제 있는 리소스만 빠르게 찾을 때](#문제-있는-리소스만-빠르게-찾을-때)
  - [특정 Pod 의 소유자(Deployment)를 확인할 때](#특정-pod-의-소유자deployment를-확인할-때)

---

# 개요

k9s 는 터미널에서 Kubernetes 클러스터를 관리하는 TUI(Text User Interface) 도구이다. `kubectl` 명령어를 일일이 타이핑하는 대신, 단축키로 빠르게 리소스를 조회하고 조작할 수 있다.

```
kubectl get pods → k9s 에서 화살표로 선택
kubectl logs -f  → k9s 에서 l 키 하나
kubectl exec -it → k9s 에서 s 키 하나
kubectl describe → k9s 에서 d 키 하나
```

# 학습 자료

- [k9s 공식 사이트](https://k9scli.io/)
- [k9s | github](https://github.com/derailed/k9s)
- [k9s 설치 및 사용법](https://1minute-before6pm.tistory.com/18)

---

# 설치 및 실행

```bash
# 설치
brew install k9s

# 기본 실행
k9s

# 특정 네임스페이스로 시작
k9s -n myns

# 특정 리소스 뷰로 시작
k9s -c pod         # Pod 목록으로 시작
k9s -c dp          # Deployment 목록으로 시작
k9s -c svc         # Service 목록으로 시작

# 특정 컨텍스트로 시작
k9s --context my-production-cluster

# 읽기 전용 모드 (실수 방지)
k9s --readonly

# 버전 확인
k9s version
```

---

# 화면 구성

```
┌─────────────────────────────────────────────┐
│ Context: k3d-harness-local [RW]             │  ← 현재 클러스터, 읽기/쓰기 모드
│ Cluster: k3d-harness-local                  │
│ User:    admin@k3d-harness-local            │
│ K9s Rev: v0.50.18                           │
│ K8s Rev: v1.33.6+k3s1                       │
│ CPU: 1%   MEM: 12%                          │  ← 클러스터 리소스 사용률
├─────────────────────────────────────────────┤
│ NAMESPACE  NAME           READY  STATUS     │  ← 리소스 목록
│ default    my-app-xxx     1/1    Running    │
│ default    my-db-xxx      1/1    Running    │
│ kube-sys   coredns-xxx    1/1    Running    │
├─────────────────────────────────────────────┤
│ <pod> <help>                                │  ← 현재 보고 있는 리소스 타입
└─────────────────────────────────────────────┘
```

---

# 리소스 이동 (Command Mode)

`:` 를 누르면 Command Mode 가 된다. 리소스 단축어를 입력하여 화면을 전환한다.

| 명령어 | 이동 대상 | kubectl 대응 |
|--------|----------|-------------|
| `:pod` 또는 `:po` | Pod 목록 | `kubectl get pods` |
| `:dp` | Deployment 목록 | `kubectl get deployments` |
| `:svc` | Service 목록 | `kubectl get services` |
| `:ns` | Namespace 목록 | `kubectl get namespaces` |
| `:no` | Node 목록 | `kubectl get nodes` |
| `:rs` | ReplicaSet 목록 | `kubectl get replicasets` |
| `:sts` | StatefulSet 목록 | `kubectl get statefulsets` |
| `:ds` | DaemonSet 목록 | `kubectl get daemonsets` |
| `:cm` | ConfigMap 목록 | `kubectl get configmaps` |
| `:sec` | Secret 목록 | `kubectl get secrets` |
| `:ing` | Ingress 목록 | `kubectl get ingress` |
| `:pv` | PersistentVolume 목록 | `kubectl get pv` |
| `:pvc` | PersistentVolumeClaim 목록 | `kubectl get pvc` |
| `:cj` | CronJob 목록 | `kubectl get cronjobs` |
| `:job` | Job 목록 | `kubectl get jobs` |
| `:hpa` | HPA 목록 | `kubectl get hpa` |
| `:sa` | ServiceAccount 목록 | `kubectl get serviceaccounts` |
| `:rb` | RoleBinding 목록 | `kubectl get rolebindings` |
| `:crb` | ClusterRoleBinding 목록 | `kubectl get clusterrolebindings` |
| `:ctx` | Context 전환 | `kubectl config use-context` |
| `:alias` | 모든 단축어 보기 | — |

> 단축어가 기억나지 않으면 `Ctrl+a` 로 전체 Alias 목록을 볼 수 있다.

---

# 리소스 조작

Pod 목록에서 화살표로 선택한 뒤 아래 키를 누른다.

## 조회/탐색

| 키 | 기능 | kubectl 대응 | 예시 |
|---|---|---|---|
| `Enter` | View (상세 진입) | — | Pod 선택 → `Enter` → 컨테이너 목록 |
| `d` | Describe | `kubectl describe pod my-app` | 이벤트, 상태, 볼륨 마운트 등 상세 정보 |
| `o` | Show Node | — | 이 Pod 이 어느 노드에서 실행 중인지 표시 |
| `Shift+j` | Jump Owner | — | Pod → 이 Pod 을 관리하는 Deployment/ReplicaSet 으로 이동 |
| `Ctrl+w` | Toggle Wide | `kubectl get pod -o wide` | IP, 노드 등 추가 컬럼 표시 |
| `0` | 전체 네임스페이스 | `kubectl get pod -A` | 모든 네임스페이스의 리소스 보기 |
| `1` | default 네임스페이스 | `kubectl get pod -n default` | default 만 보기 |

## 로그

| 키 | 기능 | kubectl 대응 | 예시 |
|---|---|---|---|
| `l` | Logs | `kubectl logs -f my-app` | 실시간 로그 스트리밍 |
| `p` | Logs Previous | `kubectl logs --previous my-app` | **CrashLoopBackOff 시 이전 컨테이너 로그** |

## 접속/실행

| 키 | 기능 | kubectl 대응 | 예시 |
|---|---|---|---|
| `s` | Shell | `kubectl exec -it my-app -- /bin/sh` | 컨테이너 안에 쉘 접속 |
| `a` | Attach | `kubectl attach -it my-app` | 메인 프로세스의 stdin/stdout 에 직접 연결 |

**`s` vs `a` 차이:**

```
s (Shell):   컨테이너 안에 /bin/sh 를 새로 실행하고 거기에 접속
             → 자유롭게 명령어 실행 가능 (ls, cat, curl 등)
             → 디버깅할 때 사용. 실무에서 99% 이것을 쓴다

a (Attach):  이미 실행 중인 메인 프로세스(PID 1)에 직접 연결
             → 새 프로세스를 띄우지 않음
             → 대화형 앱(Python REPL, Node.js REPL)이 메인일 때만 의미 있음
             → 대부분의 웹서버/워커 컨테이너에서는 stdout 만 보임 (그건 l 로그가 낫다)
```

## 편집/삭제

| 키 | 기능 | kubectl 대응 | 예시 |
|---|---|---|---|
| `e` | Edit | `kubectl edit pod my-app` | vim 에서 YAML 편집 |
| `Ctrl+d` | Delete | `kubectl delete pod my-app` | 확인 후 삭제 |
| `Ctrl+k` | Kill | `kubectl delete pod --force --grace-period=0` | 강제 즉시 종료 |
| `c` | Copy | — | 리소스 YAML 을 클립보드에 복사 |
| `n` | Copy Namespace | — | 네임스페이스 이름을 클립보드에 복사 |
| `Ctrl+s` | Save | — | 리소스 YAML 을 로컬 파일로 저장 |

## 포트 포워딩

| 키 | 기능 | kubectl 대응 | 예시 |
|---|---|---|---|
| `Shift+f` | Port-Forward | `kubectl port-forward my-app 8080:80` | 로컬 포트 → Pod 포트 연결 |
| `f` | Show PortForward | — | 현재 활성화된 포트 포워딩 목록 |

## 정렬/필터

| 키 | 기능 | 예시 |
|---|---|---|
| `/term` | Filter | `/nginx` → 이름에 "nginx" 포함된 것만 표시 |
| `Ctrl+z` | Toggle Faults | **문제 있는 리소스만 표시** (장애 대응 시 가장 먼저) |
| `z` | Sanitize | 사용하지 않는 리소스 표시 (빈 ConfigMap 등) |
| `Shift+a` | Sort Age | 생성 시간순 정렬 |
| `Shift+n` | Sort Name | 이름순 정렬 |
| `Shift+p` | Sort Namespace | 네임스페이스순 정렬 |
| `Shift+s` | Sort Status | 상태순 정렬 (Running, Pending, Error) |
| `Shift+o` | Sort Selected Column | 현재 선택된 컬럼 기준 정렬 |

---

# 일반 조작

| 키 | 기능 | 설명 |
|---|---|---|
| `Ctrl+a` | Aliases | 리소스 단축어 전체 목록 |
| `q` | Back | 이전 화면 |
| `Esc` | Back + Clear | 뒤로 가기 + 필터 초기화 |
| `:cmd` | Command mode | 리소스 이동 (`:pod`, `:dp`, `:svc` 등) |
| `?` | Help | 단축키 목록 (스크린샷 화면) |
| `Space` | Mark | 리소스 선택 (여러 개 선택 가능) |
| `Ctrl+\` | Mark Clear | 선택 해제 |
| `Ctrl+Space` | Mark Range | 범위 선택 |
| `Ctrl+r` | Reload | 설정 리로드 |
| `Ctrl+g` | Toggle Crumbs | 상단 경로 표시줄 토글 |
| `Ctrl+e` | Toggle Header | 헤더 행 토글 |
| `:q` | Quit | k9s 종료 |

---

# 네비게이션

| 키 | 기능 |
|---|---|
| `j` / `k` | 위/아래 이동 (vim 스타일) |
| `h` / `l` | 좌/우 이동 |
| `g` | 맨 위로 |
| `Shift+g` | 맨 아래로 |
| `Ctrl+f` | Page Down |
| `Ctrl+b` | Page Up |
| `[` / `]` | History Back / Forward |
| `-` | 마지막으로 사용한 명령어 |

---

# 자주 쓰는 시나리오

## Pod 이 CrashLoopBackOff 일 때

```
1. Ctrl+z        → 문제 있는 리소스만 보기 (정상 Pod 숨김)
2. 화살표로 문제 Pod 선택
3. p             → 이전 컨테이너 로그 확인 (죽기 전 마지막 로그)
4. d             → Describe 로 이벤트 확인 (OOM? ImagePullBackOff?)
```

> `l` (현재 로그)이 아닌 `p` (이전 로그)를 눌러야 한다. 컨테이너가 계속 재시작되므로 현재 로그는 비어있을 수 있다.

## 특정 서비스에 로컬에서 접속하고 싶을 때

```
1. :svc          → Service 목록으로 이동
2. /my-api       → 이름 필터링
3. Shift+f       → Port Forward 설정
                    Local Port: 8080, Container Port: 80
4. 브라우저에서 localhost:8080 접속
5. f             → 활성 포트 포워딩 목록 확인
```

## Deployment 이미지를 업데이트할 때

```
1. :dp           → Deployment 목록
2. 선택 후 e     → vim 에서 YAML 편집
3. /image 로 검색 → 이미지 태그 수정 (예: v1.0 → v1.1)
4. :wq           → 저장하면 자동으로 Rolling Update 시작
5. :pod          → Pod 목록으로 가서 새 Pod 이 뜨는지 확인
```

## 여러 Pod 을 한꺼번에 삭제할 때

```
1. Space 로 Pod 들을 하나씩 선택 (좌측에 체크 표시)
2. 또는 Ctrl+Space 로 범위 선택
3. Ctrl+d        → 선택된 Pod 전체 삭제
```

## 문제 있는 리소스만 빠르게 찾을 때

```
1. 0             → 전체 네임스페이스 보기
2. Ctrl+z        → Faults 토글 (에러 상태만 필터)
3. Shift+s       → Status 순 정렬
→ Error, CrashLoopBackOff, Pending 등이 위로 올라온다
```

## 특정 Pod 의 소유자(Deployment)를 확인할 때

```
1. Pod 선택
2. Shift+j       → Owner 로 점프
→ Pod → ReplicaSet → Deployment 순으로 올라갈 수 있다
```
