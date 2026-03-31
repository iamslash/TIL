# Kubernetes 기초

- [Kubernetes 란?](#kubernetes-란)
- [아키텍처](#아키텍처)
- [설치](#설치)
- [kubectl 기본 명령어](#kubectl-기본-명령어)
- [Pod](#pod)
- [ReplicaSet](#replicaset)
- [Deployment](#deployment)
- [Service](#service)
- [Namespace](#namespace)

---

## Kubernetes 란?

### 컨테이너 오케스트레이션이 왜 필요한가

서버 10대에 컨테이너 100개를 수동으로 관리한다고 상상해 보자.

- 어떤 서버에 어떤 컨테이너가 떠 있는지 파악하기 어렵다
- 컨테이너 하나가 죽으면 직접 재시작해야 한다
- 트래픽이 급증할 때 컨테이너를 몇 개 더 띄울지, 어느 서버에 띄울지 사람이 판단해야 한다
- 배포할 때 서비스 중단 없이 구버전을 새버전으로 교체하기가 복잡하다
- 컨테이너끼리 통신하려면 IP 를 직접 관리해야 하는데, 컨테이너가 재시작되면 IP 가 바뀐다

이런 문제들이 쌓이면 운영팀은 정작 기능 개발보다 인프라 관리에 더 많은 시간을 쓰게 된다.

### Kubernetes 가 해결하는 것

Kubernetes (이하 k8s) 는 컨테이너 오케스트레이션 플랫폼이다. 다음 문제를 자동으로 처리해 준다.

| 문제 | Kubernetes 의 해결책 |
|------|---------------------|
| 컨테이너가 죽었다 | 자동으로 재시작 (자기 치유, Self-healing) |
| 트래픽이 늘었다 | 선언한 replicas 수만큼 Pod 를 자동 증감 (스케일링) |
| 새 버전을 배포해야 한다 | 롤링 업데이트로 무중단 배포 (배포 전략) |
| 컨테이너 IP 가 바뀌었다 | Service 오브젝트가 고정 IP/DNS 를 제공 (서비스 디스커버리) |
| 여러 컨테이너로 트래픽을 분산해야 한다 | Service 가 로드 밸런싱 처리 |

핵심 개념은 **선언형(Declarative) 관리**다. "nginx 3개를 항상 유지해라"라고 YAML 파일로 선언하면, Kubernetes 가 그 상태를 계속 감시하면서 현실을 선언에 맞게 유지한다.

---

## 아키텍처

Kubernetes 클러스터는 크게 두 종류의 노드로 구성된다.

```
+---------------------------+       +------------------+
|      Control Plane        |       |   Worker Node    |
|  (클러스터의 두뇌)           |       |  (실제 일꾼)      |
|                           |       |                  |
|  kube-apiserver  <--------+-------+--> kubelet       |
|  etcd                     |       |    kube-proxy    |
|  kube-scheduler           |       |    container     |
|  kube-controller-manager  |       |    runtime       |
+---------------------------+       +------------------+
```

### Control Plane 컴포넌트

**kube-apiserver**
- 클러스터의 정문(게이트웨이)이다
- `kubectl` 명령어는 모두 apiserver 를 통해 처리된다
- 인증, 인가, 유효성 검사를 수행한 뒤 etcd 에 상태를 저장한다
- 비유: 회사의 접수 창구. 모든 요청은 이 창구를 통해서만 처리된다

**etcd**
- 클러스터의 모든 상태 정보를 저장하는 분산 key-value 저장소다
- Pod 목록, 노드 정보, 설정값 등이 여기에 저장된다
- 비유: 회사의 문서 보관함. 모든 기록이 여기에 있다

**kube-scheduler**
- 새로 생성된 Pod 를 어느 Worker Node 에 배치할지 결정한다
- CPU/메모리 여유, affinity 규칙, taint/toleration 등을 고려한다
- 비유: 인사팀 배치 담당자. 빈 자리(노드)를 보고 신입(Pod)을 배정한다

**kube-controller-manager**
- 여러 컨트롤러를 하나의 프로세스로 실행한다
- ReplicaSet 컨트롤러, Deployment 컨트롤러, Node 컨트롤러 등이 포함된다
- 각 컨트롤러는 "현재 상태"와 "원하는 상태"를 비교하여 차이를 좁힌다
- 비유: 관리감독팀. 선언된 규칙대로 현장이 돌아가는지 계속 감시한다

### Worker Node 컴포넌트

**kubelet**
- 각 Worker Node 에서 실행되는 에이전트다
- apiserver 에서 받은 PodSpec 을 읽어 컨테이너를 실행하고 상태를 보고한다
- 비유: 현장 관리자. 본사(Control Plane) 지시를 받아 현장에서 실행한다

**kube-proxy**
- 각 노드의 네트워크 규칙을 관리한다
- Service 오브젝트가 동작할 수 있도록 iptables 규칙을 설정한다
- 비유: 교환원. 외부 전화(트래픽)를 올바른 내선(Pod)으로 연결한다

**container runtime**
- 실제 컨테이너를 실행하는 소프트웨어다
- containerd, CRI-O 등이 있다 (Docker 는 1.24 이후 기본 런타임에서 제거됨)

### 아키텍처 다이어그램

```
kubectl
  |
  v
kube-apiserver  <-->  etcd
  |
  +---> kube-scheduler
  |
  +---> kube-controller-manager
  |
  v
kubelet (Worker Node)
  |
  v
container runtime  -->  Pod (컨테이너 1개 이상)
```

참고 이미지: `img/KubernetesArchitecturalOverview.png`

---

## 설치

### minikube (로컬 학습용)

로컬 머신에서 단일 노드 Kubernetes 클러스터를 실행한다. 학습 목적에 적합하다.

```bash
# macOS 설치
brew install minikube

# 클러스터 시작
minikube start

# 상태 확인
minikube status

# 클러스터 중지
minikube stop

# 대시보드 실행
minikube dashboard
```

### kind (Kubernetes IN Docker, CI/CD 테스트용)

Docker 컨테이너 안에서 Kubernetes 노드를 실행한다. CI 파이프라인 테스트에 적합하다.

```bash
# macOS 설치
brew install kind

# 클러스터 생성
kind create cluster --name my-cluster

# 클러스터 목록 확인
kind get clusters

# 클러스터 삭제
kind delete cluster --name my-cluster
```

### EKS (프로덕션)

AWS 의 관리형 Kubernetes 서비스다. Control Plane 을 AWS 가 관리해 준다.

```bash
# eksctl 설치 (macOS)
brew tap weaveworks/tap
brew install weaveworks/tap/eksctl

# 클러스터 생성 (약 15-20분 소요)
eksctl create cluster \
  --name my-cluster \
  --region ap-northeast-2 \
  --nodegroup-name my-nodes \
  --node-type t3.medium \
  --nodes 2

# kubeconfig 업데이트
aws eks update-kubeconfig --region ap-northeast-2 --name my-cluster

# 노드 확인
kubectl get nodes
```

---

## kubectl 기본 명령어

`kubectl` 은 Kubernetes 클러스터를 조작하는 CLI 도구다.

### 핵심 명령어

```bash
# 리소스 목록 조회
kubectl get pods
kubectl get nodes
kubectl get deployments
kubectl get services

# 리소스 상세 정보 조회
kubectl describe pod <pod-name>
kubectl describe node <node-name>

# YAML/JSON 파일로 리소스 생성 또는 업데이트
kubectl apply -f my-pod.yaml

# 리소스 삭제
kubectl delete pod <pod-name>
kubectl delete -f my-pod.yaml

# 컨테이너 로그 확인
kubectl logs <pod-name>
kubectl logs -f <pod-name>                    # 실시간 스트리밍
kubectl logs <pod-name> -c <container-name>   # 특정 컨테이너 로그

# 컨테이너 내부 쉘 접속
kubectl exec -it <pod-name> -- /bin/bash
kubectl exec -it <pod-name> -c <container-name> -- /bin/sh
```

### 자주 쓰는 옵션

```bash
# 넓은 형식으로 출력 (IP, 노드 정보 포함)
kubectl get pods -o wide

# YAML 형식으로 출력 (현재 상태를 파일로 저장할 때 유용)
kubectl get pod <pod-name> -o yaml

# 모든 네임스페이스의 리소스 조회
kubectl get pods --all-namespaces
kubectl get pods -A

# 특정 네임스페이스의 리소스 조회
kubectl get pods -n kube-system

# 레이블 표시
kubectl get pods --show-labels

# 레이블로 필터링
kubectl get pods -l app=nginx
```

### 실무 팁: alias 와 bash completion

매번 `kubectl` 을 타이핑하는 것은 번거롭다. 다음 설정을 `~/.bashrc` 또는 `~/.zshrc` 에 추가한다.

```bash
# alias 설정
alias k='kubectl'
alias kgp='kubectl get pods'
alias kgs='kubectl get services'
alias kgd='kubectl get deployments'
alias kdp='kubectl describe pod'

# bash completion 활성화 (bash)
source <(kubectl completion bash)

# zsh completion 활성화
source <(kubectl completion zsh)

# alias 에도 completion 적용
complete -F __start_kubectl k
```

---

## Pod

### Pod 이란

Pod 는 Kubernetes 에서 배포할 수 있는 가장 작은 단위다. 하나 이상의 컨테이너를 담는 껍데기(wrapper)라고 이해하면 된다.

Pod 안의 컨테이너들은 다음을 공유한다.
- 네트워크 네임스페이스 (같은 IP, 같은 포트 공간)
- 스토리지 (볼륨 공유 가능)

직접 Pod 를 만들어 쓰는 경우는 드물다. 보통 Deployment 나 ReplicaSet 을 통해 Pod 를 관리한다. 직접 만든 Pod 는 노드 장애 시 자동 복구되지 않는다.

### 기본 Pod YAML 예제

```yaml
# my-pod.yaml
apiVersion: v1          # 이 리소스가 사용하는 API 버전
kind: Pod               # 리소스 종류
metadata:
  name: my-nginx        # Pod 이름 (클러스터 내 고유해야 함)
  labels:
    app: nginx          # 레이블: Service 나 ReplicaSet 이 Pod 를 찾을 때 사용
spec:
  containers:
  - name: nginx-container    # 컨테이너 이름
    image: nginx:1.25        # 사용할 Docker 이미지
    ports:
    - containerPort: 80      # 컨테이너가 사용하는 포트 (문서 목적, 강제성 없음)
    resources:
      requests:
        memory: "64Mi"       # 최소 요청 메모리
        cpu: "250m"          # 최소 요청 CPU (1000m = 1 코어)
      limits:
        memory: "128Mi"      # 최대 메모리 (초과 시 OOMKilled)
        cpu: "500m"          # 최대 CPU
```

```bash
# Pod 생성
kubectl apply -f my-pod.yaml

# Pod 상태 확인
kubectl get pods

# Pod 상세 정보 (이벤트, 조건 등 포함)
kubectl describe pod my-nginx

# Pod 삭제
kubectl delete pod my-nginx
```

### 멀티 컨테이너 Pod (Sidecar 패턴)

Sidecar 패턴은 메인 컨테이너 옆에 보조 컨테이너를 붙이는 방식이다. 로그 수집, 프록시, 설정 파일 동기화 등에 사용된다.

```yaml
# sidecar-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-with-sidecar
  labels:
    app: my-app
spec:
  containers:
  - name: main-app              # 메인 컨테이너: 실제 애플리케이션
    image: nginx:1.25
    ports:
    - containerPort: 80
    volumeMounts:
    - name: shared-logs         # 사이드카와 공유할 볼륨을 마운트
      mountPath: /var/log/nginx

  - name: log-collector         # 사이드카 컨테이너: 로그 수집
    image: busybox
    command: ["/bin/sh", "-c", "tail -f /logs/access.log"]
    volumeMounts:
    - name: shared-logs         # 메인 컨테이너와 같은 볼륨을 마운트
      mountPath: /logs

  volumes:
  - name: shared-logs           # 두 컨테이너가 공유하는 emptyDir 볼륨
    emptyDir: {}
```

```bash
# Pod 생성
kubectl apply -f sidecar-pod.yaml

# 특정 컨테이너 로그 확인
kubectl logs app-with-sidecar -c main-app
kubectl logs app-with-sidecar -c log-collector

# 특정 컨테이너에 접속
kubectl exec -it app-with-sidecar -c main-app -- /bin/bash
```

### Pod 라이프사이클

Pod 는 다음 단계를 거친다.

| 단계 | 설명 |
|------|------|
| `Pending` | Pod 가 수락되었지만 컨테이너가 아직 시작되지 않은 상태. 이미지 다운로드 중이거나 스케줄링 대기 중 |
| `Running` | Pod 가 노드에 배치되고 하나 이상의 컨테이너가 실행 중인 상태 |
| `Succeeded` | Pod 의 모든 컨테이너가 정상 종료된 상태 (Job 에서 주로 발생) |
| `Failed` | Pod 의 모든 컨테이너가 종료되었고 하나 이상이 실패로 종료된 상태 |
| `Unknown` | Pod 상태를 알 수 없는 상태. 주로 노드와 통신 불가 시 발생 |

---

## ReplicaSet

### 왜 필요한가

Pod 를 직접 만들면 그 Pod 가 죽었을 때 아무도 다시 살려주지 않는다. ReplicaSet 은 "이 Pod 를 항상 N개 유지해라"라는 규칙을 지키는 컨트롤러다.

- Pod 가 죽으면 새 Pod 를 자동으로 생성한다
- 노드 장애로 Pod 가 사라지면 다른 노드에 새 Pod 를 생성한다
- Pod 수를 줄이면 초과분을 자동으로 삭제한다

실무에서는 ReplicaSet 을 직접 쓰지 않고 Deployment 를 통해 간접적으로 사용한다. Deployment 가 ReplicaSet 을 관리하기 때문이다.

### ReplicaSet YAML 예제

```yaml
# my-replicaset.yaml
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: nginx-replicaset
spec:
  replicas: 3             # 유지할 Pod 개수
  selector:
    matchLabels:
      app: nginx          # 이 레이블을 가진 Pod 를 관리 대상으로 삼는다
  template:               # 새 Pod 를 만들 때 사용할 템플릿
    metadata:
      labels:
        app: nginx        # selector 의 matchLabels 와 반드시 일치해야 한다
    spec:
      containers:
      - name: nginx
        image: nginx:1.25
        ports:
        - containerPort: 80
```

```bash
# ReplicaSet 생성
kubectl apply -f my-replicaset.yaml

# ReplicaSet 확인
kubectl get replicaset
kubectl get rs

# Pod 확인 (3개가 생성되어 있어야 함)
kubectl get pods --show-labels

# Pod 하나를 강제 삭제 -> ReplicaSet 이 자동으로 새 Pod 생성
kubectl delete pod <pod-name>
kubectl get pods   # 곧 다시 3개가 됨

# ReplicaSet 삭제 (관리 중인 Pod 도 함께 삭제됨)
kubectl delete replicaset nginx-replicaset
```

### selector 와 label 매칭

ReplicaSet 의 `spec.selector.matchLabels` 와 `spec.template.metadata.labels` 는 반드시 일치해야 한다. ReplicaSet 은 selector 로 자신이 관리할 Pod 를 찾는다. label 이 일치하는 Pod 가 이미 존재하면 그것도 관리 대상에 포함된다.

```
ReplicaSet
  selector.matchLabels:
    app: nginx    <---- 이 레이블로 Pod 를 찾는다
  template.metadata.labels:
    app: nginx    <---- 새 Pod 를 만들 때 이 레이블을 붙인다
```

---

## Deployment

### ReplicaSet 과의 관계

Deployment 는 ReplicaSet 위에서 동작하는 상위 개념이다.

```
Deployment
  └── ReplicaSet (v1)  [구버전, replicas=0]
  └── ReplicaSet (v2)  [현재버전, replicas=3]
```

Deployment 가 ReplicaSet 을 직접 생성하고 관리한다. 새 버전을 배포할 때 새 ReplicaSet 을 만들고 구 ReplicaSet 의 replicas 를 점차 줄인다. 이것이 롤링 업데이트다. 문제가 생기면 구 ReplicaSet 의 replicas 를 다시 늘려 롤백할 수 있다.

### Deployment YAML 예제

```yaml
# my-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment    # Deployment 이름
  labels:
    app: nginx
spec:
  replicas: 3               # 유지할 Pod 개수
  selector:
    matchLabels:
      app: nginx            # 관리할 Pod 의 레이블
  strategy:
    type: RollingUpdate     # 배포 전략: RollingUpdate 또는 Recreate
    rollingUpdate:
      maxSurge: 1           # 업데이트 중 최대로 초과 생성 가능한 Pod 수
      maxUnavailable: 1     # 업데이트 중 최대로 중단 가능한 Pod 수
  minReadySeconds: 5        # 새 Pod 가 Ready 상태가 된 후 대기 시간(초)
  template:                 # Pod 템플릿
    metadata:
      labels:
        app: nginx          # selector 와 일치해야 함
    spec:
      containers:
      - name: nginx
        image: nginx:1.25   # 배포할 이미지
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "64Mi"
            cpu: "250m"
          limits:
            memory: "128Mi"
            cpu: "500m"
```

### 주요 필드 설명

| 필드 | 설명 |
|------|------|
| `replicas` | 유지할 Pod 수. 기본값은 1 |
| `strategy.type` | `RollingUpdate`: 점진적 교체 (기본값). `Recreate`: 전체 중단 후 재시작 |
| `rollingUpdate.maxSurge` | 업데이트 중 replicas 초과 허용 Pod 수 (숫자 또는 %) |
| `rollingUpdate.maxUnavailable` | 업데이트 중 중단 허용 Pod 수 (숫자 또는 %) |
| `minReadySeconds` | 새 Pod 가 Ready 후 안정적이라고 판단하기까지 대기 초 |
| `template` | 생성할 Pod 의 스펙. Pod YAML 의 `metadata`, `spec` 과 동일 |

### 배포 및 스케일링 명령어

```bash
# Deployment 생성
kubectl apply -f my-deployment.yaml

# Deployment 확인
kubectl get deployments
kubectl get deploy

# Pod 확인
kubectl get pods

# 스케일링: replicas 를 5개로 변경
kubectl scale deployment nginx-deployment --replicas=5

# 이미지 업데이트 (롤링 업데이트 시작)
kubectl set image deployment/nginx-deployment nginx=nginx:1.26

# 롤아웃 상태 확인
kubectl rollout status deployment/nginx-deployment

# 롤아웃 히스토리 확인
kubectl rollout history deployment/nginx-deployment

# 이전 버전으로 롤백
kubectl rollout undo deployment/nginx-deployment

# 특정 버전으로 롤백
kubectl rollout undo deployment/nginx-deployment --to-revision=2

# Deployment 삭제 (Pod, ReplicaSet 모두 삭제됨)
kubectl delete deployment nginx-deployment
```

---

## Service

### 왜 필요한가

Pod 는 IP 를 갖고 있지만 그 IP 는 고정되어 있지 않다. Pod 가 재시작되면 IP 가 바뀐다. 또한 ReplicaSet 으로 Pod 가 3개 있다면 그 중 어떤 Pod 에 요청을 보낼지 결정해야 한다.

Service 는 이 두 가지 문제를 해결한다.
- Pod IP 가 바뀌어도 항상 동일한 IP/DNS 로 접근 가능하게 한다
- 여러 Pod 에 로드 밸런싱을 해준다

Service 는 `selector` 로 대상 Pod 를 찾는다. label 이 일치하는 Pod 의 IP 를 Endpoints 오브젝트에 자동으로 등록하고 관리한다.

### ClusterIP: 클러스터 내부 통신

기본 Service 타입이다. 클러스터 내부에서만 접근 가능한 가상 IP 를 할당한다.

```yaml
# clusterip-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-clusterip-svc    # Service 이름 (DNS 이름으로도 사용됨)
spec:
  type: ClusterIP              # 생략하면 기본값이 ClusterIP
  selector:
    app: nginx                 # 이 레이블을 가진 Pod 로 트래픽을 전달
  ports:
  - name: http
    port: 80                   # Service 가 노출하는 포트 (클라이언트가 접근하는 포트)
    targetPort: 80             # 트래픽을 전달할 Pod 의 포트
    protocol: TCP
```

```bash
# Service 생성
kubectl apply -f clusterip-service.yaml

# Service 확인
kubectl get services
kubectl get svc

# 엔드포인트 확인 (트래픽이 전달되는 Pod IP 목록)
kubectl get endpoints nginx-clusterip-svc

# 클러스터 내부에서 DNS 로 접근 테스트
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl http://nginx-clusterip-svc:80
```

### NodePort: 외부 노출 (개발용)

각 노드의 특정 포트를 열어 외부에서 접근 가능하게 한다. 포트 범위는 30000-32767 이다. 개발 환경이나 테스트 목적으로 주로 사용한다.

```yaml
# nodeport-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-nodeport-svc
spec:
  type: NodePort
  selector:
    app: nginx
  ports:
  - name: http
    port: 80           # 클러스터 내부에서 접근하는 포트 (ClusterIP 로도 접근 가능)
    targetPort: 80     # Pod 의 포트
    nodePort: 30080    # 노드에서 열리는 포트 (30000-32767 범위)
    protocol: TCP
```

```bash
# Service 생성
kubectl apply -f nodeport-service.yaml

# Service 확인 (EXTERNAL-IP 는 <none>, PORT(S) 에 30080 표시)
kubectl get svc nginx-nodeport-svc

# 노드 IP 확인
kubectl get nodes -o wide

# 외부에서 접근 (minikube 사용 시)
minikube service nginx-nodeport-svc --url

# 외부에서 직접 접근
curl http://<노드-IP>:30080
```

### LoadBalancer: 외부 노출 (프로덕션)

클라우드 환경(AWS, GCP, Azure)에서 외부 로드 밸런서를 자동으로 프로비저닝한다. 실제 프로덕션 환경에서 외부 트래픽을 받을 때 사용한다.

```yaml
# loadbalancer-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-lb-svc
  annotations:
    # AWS 환경에서 NLB 를 사용하려면 아래 annotation 추가
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
spec:
  type: LoadBalancer
  selector:
    app: nginx
  ports:
  - name: http
    port: 80           # 외부 로드 밸런서가 노출하는 포트
    targetPort: 80     # Pod 의 포트
    protocol: TCP
```

```bash
# Service 생성
kubectl apply -f loadbalancer-service.yaml

# EXTERNAL-IP 가 할당될 때까지 대기 (클라우드 환경에서 수십 초 소요)
kubectl get svc nginx-lb-svc -w

# EXTERNAL-IP 가 할당되면 해당 주소로 접근
curl http://<EXTERNAL-IP>:80
```

### ExternalName

클러스터 내부에서 외부 서비스를 DNS 이름으로 참조할 때 사용한다. 실제 트래픽 전달 없이 DNS CNAME 으로 동작한다.

```yaml
# externalname-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: my-database-svc       # 클러스터 내부에서 사용할 이름
spec:
  type: ExternalName
  externalName: my-db.example.com   # 실제 외부 DNS 이름으로 CNAME 처리
```

```bash
# 클러스터 내부에서 my-database-svc 로 접근하면 my-db.example.com 으로 해석됨
kubectl apply -f externalname-service.yaml
```

### Service 타입 비교

| 타입 | 접근 범위 | 사용 목적 |
|------|-----------|-----------|
| `ClusterIP` | 클러스터 내부 only | 마이크로서비스 간 통신 |
| `NodePort` | 노드 IP + 포트로 외부 접근 | 개발/테스트 환경 |
| `LoadBalancer` | 외부 LB IP 로 접근 | 프로덕션 외부 트래픽 |
| `ExternalName` | DNS CNAME | 외부 서비스 참조 |

---

## Namespace

### 용도

Namespace 는 하나의 클러스터를 논리적으로 분리하는 방법이다. 다음 목적으로 사용한다.

- **환경 분리**: `dev`, `staging`, `production` 네임스페이스로 환경을 격리한다
- **팀 분리**: 팀별로 네임스페이스를 할당해 리소스 충돌을 방지한다
- **리소스 쿼터 적용**: 네임스페이스별로 CPU, 메모리 사용량을 제한할 수 있다

같은 네임스페이스 안에서는 리소스 이름이 고유해야 하지만, 다른 네임스페이스에서는 같은 이름을 사용해도 된다.

### 기본 Namespace

Kubernetes 설치 시 기본으로 생성되는 네임스페이스들이다.

| 네임스페이스 | 용도 |
|-------------|------|
| `default` | 네임스페이스를 지정하지 않으면 여기에 리소스가 생성됨 |
| `kube-system` | Kubernetes 시스템 컴포넌트 (apiserver, scheduler 등) |
| `kube-public` | 인증 없이 접근 가능. 주로 클러스터 정보 공개용 |
| `kube-node-lease` | 노드 하트비트 정보 저장 |

```bash
# 기본 네임스페이스 목록 확인
kubectl get namespaces
kubectl get ns

# 특정 네임스페이스의 리소스 조회
kubectl get pods -n kube-system

# 모든 네임스페이스의 Pod 조회
kubectl get pods -A
```

### Namespace 생성과 리소스 할당

```yaml
# my-namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: development    # 네임스페이스 이름
  labels:
    env: dev           # 레이블 (선택 사항)
```

```bash
# 네임스페이스 생성 (명령어 방식)
kubectl create namespace development

# 또는 YAML 파일 방식
kubectl apply -f my-namespace.yaml

# 네임스페이스 확인
kubectl get ns
```

특정 네임스페이스에 Pod 를 배포하는 예제다.

```yaml
# pod-in-namespace.yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-dev
  namespace: development    # 배포할 네임스페이스 지정
  labels:
    app: nginx
spec:
  containers:
  - name: nginx
    image: nginx:1.25
    ports:
    - containerPort: 80
```

```bash
# 네임스페이스를 지정하여 Pod 생성
kubectl apply -f pod-in-namespace.yaml

# development 네임스페이스의 Pod 조회
kubectl get pods -n development

# 특정 네임스페이스를 기본값으로 설정 (매번 -n 옵션을 붙이지 않아도 됨)
kubectl config set-context --current --namespace=development

# 현재 컨텍스트 확인
kubectl config view --minify | grep namespace
```

### ResourceQuota 로 네임스페이스 리소스 제한

팀별 네임스페이스에 CPU, 메모리 사용량 상한을 걸어두면 한 팀이 클러스터 전체 자원을 독점하는 것을 막을 수 있다.

```yaml
# resource-quota.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: dev-quota
  namespace: development    # 적용할 네임스페이스
spec:
  hard:
    requests.cpu: "2"          # CPU 요청 합계 상한
    requests.memory: 2Gi       # 메모리 요청 합계 상한
    limits.cpu: "4"            # CPU 한도 합계 상한
    limits.memory: 4Gi         # 메모리 한도 합계 상한
    pods: "10"                 # Pod 총 개수 상한
```

```bash
# ResourceQuota 적용
kubectl apply -f resource-quota.yaml

# 쿼터 사용량 확인
kubectl describe resourcequota dev-quota -n development
```
