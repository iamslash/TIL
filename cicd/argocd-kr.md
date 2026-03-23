# ArgoCD (CD)

## 개요

ArgoCD는 Kubernetes를 위한 **GitOps 기반 CD 도구**이다. Git repo에 정의된
desired state와 K8s 클러스터의 actual state를 비교하여 동기화한다.

## 핵심 원칙: GitOps

```
Git repo (desired state)  ←→  K8s cluster (actual state)
         │                          │
         └── ArgoCD가 둘을 비교 ────┘
              차이가 있으면 동기화
```

- Git이 유일한 진실의 원천이다
- 모든 배포는 Git commit으로 추적된다
- 수동으로 K8s를 변경하면 ArgoCD가 감지하고 Git 상태로 되돌린다 (selfHeal)

## 아키텍처

```
┌──────────────────────────────────────┐
│ K8s Cluster                          │
│                                      │
│  ┌──────────────────────────────┐   │
│  │ ArgoCD (argocd namespace)    │   │
│  │                              │   │
│  │  ┌────────────────────────┐  │   │
│  │  │ Application Controller │  │   │       ┌──────────┐
│  │  │ - Git repo 감시 (poll) │◀─┼───┼──pull──│ Git Repo │
│  │  │ - desired vs actual    │  │   │       └──────────┘
│  │  │ - 동기화 실행          │  │   │
│  │  └────────────────────────┘  │   │
│  │                              │   │
│  │  ┌──────────┐ ┌───────────┐  │   │
│  │  │ API      │ │ Repo      │  │   │
│  │  │ Server   │ │ Server    │  │   │
│  │  └──────────┘ └───────────┘  │   │
│  └──────────────────────────────┘   │
│                                      │
│  ┌──────────────────────────────┐   │
│  │ my-app namespace             │   │
│  │  ┌─────┐ ┌─────┐ ┌───────┐  │   │
│  │  │ Pod │ │ Svc │ │Ingress│  │   │
│  │  └─────┘ └─────┘ └───────┘  │   │
│  └──────────────────────────────┘   │
└──────────────────────────────────────┘
```

ArgoCD가 K8s 클러스터 **내부**에서 실행되므로 외부에서 K8s credential이
필요 없다.

## 설치

```bash
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# CLI 설치 (macOS)
brew install argocd

# 초기 admin 비밀번호 확인
argocd admin initial-password -n argocd
```

## 핵심 리소스: Application

ArgoCD의 배포 단위이다. "어떤 Git repo의 어떤 경로를 어떤 클러스터에
배포할지" 정의한다.

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: my-app
  namespace: argocd
spec:
  project: default

  source:
    repoURL: https://github.com/myorg/manifests
    targetRevision: main
    path: my-app/prod            # repo 내 manifest 경로

  destination:
    server: https://kubernetes.default.svc
    namespace: my-app-prod

  syncPolicy:
    automated:                   # 자동 동기화 (없으면 수동)
      prune: true                # 삭제된 리소스도 반영
      selfHeal: true             # 수동 변경 시 Git 상태로 복원
    syncOptions:
      - CreateNamespace=true
```

## 동기화 모드

| 모드 | 설명 | 사용 환경 |
|------|------|----------|
| **Manual** | 사람이 Sync 버튼/CLI로 배포 | production |
| **Auto Sync** | Git 변경 감지 시 자동 배포 | dev, staging |

```bash
# 수동 동기화
argocd app sync my-app

# 상태 확인
argocd app get my-app
```

## Sync Window

특정 시간대에만 배포를 허용/차단한다.

```yaml
spec:
  syncWindows:
    - kind: allow
      schedule: "0 9 * * 1-5"    # 월~금 09시부터
      duration: 9h                # 9시간 동안만 허용
```

## Sync Wave & Hook

배포 순서를 제어한다.

```yaml
# wave 숫자가 낮은 것부터 순서대로 배포
metadata:
  annotations:
    argocd.argoproj.io/sync-wave: "1"   # DB migration 먼저

---
metadata:
  annotations:
    argocd.argoproj.io/sync-wave: "2"   # App 다음
```

```yaml
# PreSync Hook: 배포 전에 Job 실행
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migrate
  annotations:
    argocd.argoproj.io/hook: PreSync
    argocd.argoproj.io/hook-delete-policy: HookSucceeded
spec:
  template:
    spec:
      containers:
        - name: migrate
          image: myorg/migrate:latest
          command: ["./migrate", "up"]
      restartPolicy: Never
```

## 롤백

```bash
# 이전 버전으로 롤백
argocd app rollback my-app

# 특정 Git revision으로 동기화
argocd app sync my-app --revision abc1234
```

Git에서 revert commit을 하면 ArgoCD가 자동으로 이전 상태로 배포한다.

## 다중 환경 관리

### 방법 1: 디렉토리 분리

```
manifests/
├── my-app/
│   ├── dev/
│   │   ├── deployment.yaml
│   │   └── kustomization.yaml
│   ├── staging/
│   │   ├── deployment.yaml
│   │   └── kustomization.yaml
│   └── prod/
│       ├── deployment.yaml
│       └── kustomization.yaml
```

### 방법 2: Kustomize overlays

```
manifests/
├── base/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── kustomization.yaml
├── overlays/
│   ├── dev/
│   │   └── kustomization.yaml     # replicas: 1
│   ├── staging/
│   │   └── kustomization.yaml     # replicas: 2
│   └── prod/
│       └── kustomization.yaml     # replicas: 3
```

### 방법 3: Helm values

```yaml
spec:
  source:
    repoURL: https://github.com/myorg/charts
    chart: my-app
    helm:
      valueFiles:
        - values-prod.yaml
```

## ApplicationSet

여러 Application을 템플릿으로 한번에 생성한다.

```yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: my-app-set
spec:
  generators:
    - list:
        elements:
          - env: dev
            namespace: my-app-dev
          - env: staging
            namespace: my-app-staging
          - env: prod
            namespace: my-app-prod
  template:
    metadata:
      name: my-app-{{env}}
    spec:
      source:
        repoURL: https://github.com/myorg/manifests
        path: my-app/{{env}}
      destination:
        server: https://kubernetes.default.svc
        namespace: "{{namespace}}"
```

## Argo Rollouts (고급 배포 전략)

ArgoCD 자체는 kubectl apply 수준의 배포만 지원한다. Canary, Blue-Green 등
고급 배포 전략은 **Argo Rollouts**를 추가 설치하여 사용한다.

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: my-app
spec:
  strategy:
    canary:
      steps:
        - setWeight: 10       # 10% 트래픽
        - pause: { duration: 5m }
        - setWeight: 50       # 50% 트래픽
        - pause: { duration: 5m }
        - setWeight: 100      # 전체 트래픽
```
