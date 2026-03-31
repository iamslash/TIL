# Kubernetes 고급

- [스케줄링](#스케줄링)
  - [nodeSelector](#nodeselector)
  - [Node Affinity / Anti-Affinity](#node-affinity--anti-affinity)
  - [Pod Affinity / Anti-Affinity](#pod-affinity--anti-affinity)
  - [Taint 와 Toleration](#taint-와-toleration)
  - [Topology Spread Constraints](#topology-spread-constraints)
- [Custom Resource Definition (CRD)](#custom-resource-definition-crd)
- [Operator 패턴](#operator-패턴)
- [Admission Webhook](#admission-webhook)
- [Custom Metrics 와 KEDA](#custom-metrics-와-keda)
- [Custom Scheduler](#custom-scheduler)
- [Helm](#helm)

-----

## 스케줄링

kube-scheduler 는 Pod 를 어느 Node 에 배치할지 결정한다. 내부적으로 두 단계를 거친다.

1. **Filtering** (필터링): 조건을 만족하지 않는 Node 를 후보에서 제거한다.
2. **Scoring** (점수 계산): 남은 Node 들에 점수를 매겨 가장 적합한 Node 를 선택한다.

nodeSelector, nodeAffinity, podAffinity, Taint/Toleration 은 모두 Filtering 단계에서 동작한다.

### nodeSelector

가장 단순한 Node 선택 방법이다. Node 에 Label 을 붙이고 Pod 의 `nodeSelector` 에 해당 Label 을 지정하면 그 Node 에만 Pod 가 배치된다.

**Node 에 Label 추가:**

```bash
# GPU 노드에 레이블 추가
kubectl label nodes gpu-node-1 accelerator=nvidia-tesla-v100

# 레이블 확인
kubectl get nodes --show-labels

# 레이블 삭제 (키 뒤에 - 를 붙임)
kubectl label nodes gpu-node-1 accelerator-
```

**nodeSelector 를 사용하는 Pod YAML:**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-training-pod
spec:
  nodeSelector:
    accelerator: nvidia-tesla-v100   # 이 레이블을 가진 노드에만 배치
  containers:
  - name: training
    image: tensorflow/tensorflow:latest-gpu
    resources:
      limits:
        nvidia.com/gpu: 1            # GPU 1개 요청
```

**한계:** `nodeSelector` 는 정확히 일치하는 Label 만 지원한다. `In`, `NotIn`, `Exists` 같은 연산자가 필요하면 `nodeAffinity` 를 사용해야 한다.

---

### Node Affinity / Anti-Affinity

`nodeAffinity` 는 `nodeSelector` 보다 풍부한 조건 표현식을 지원한다. 두 가지 정책이 있다.

| 정책 | 의미 |
|------|------|
| `requiredDuringSchedulingIgnoredDuringExecution` | 스케줄링 시 **반드시** 만족해야 한다. 만족하는 Node 가 없으면 Pod 는 Pending 상태가 된다. |
| `preferredDuringSchedulingIgnoredDuringExecution` | 스케줄링 시 **최대한** 만족하려 한다. 만족하는 Node 가 없어도 다른 Node 에 배치된다. |

"IgnoredDuringExecution" 은 Pod 가 이미 실행 중일 때 Node 의 Label 이 바뀌어도 Pod 를 퇴거시키지 않는다는 의미다.

**실무 예제: GPU 노드에만 ML 워크로드 배치**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-training
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-training
  template:
    metadata:
      labels:
        app: ml-training
    spec:
      affinity:
        nodeAffinity:
          # 필수 조건: gpu=true 레이블이 있는 노드에만 배치
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: gpu
                operator: In          # gpu 값이 "true" 인 노드만 선택
                values:
                - "true"
          # 선호 조건: nvidia 계열 GPU 를 우선 선택하되 필수는 아님
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 80                # 가중치 1~100, 높을수록 더 선호
            preference:
              matchExpressions:
              - key: gpu-type
                operator: In
                values:
                - nvidia-a100
                - nvidia-v100
          - weight: 20
            preference:
              matchExpressions:
              - key: gpu-type
                operator: In
                values:
                - nvidia-t4
      containers:
      - name: training
        image: tensorflow/tensorflow:latest-gpu
        resources:
          limits:
            nvidia.com/gpu: 1
```

**operator 종류:**

| operator | 의미 |
|----------|------|
| `In` | values 목록 중 하나와 일치 |
| `NotIn` | values 목록 어디에도 없음 |
| `Exists` | 해당 키가 존재함 (values 불필요) |
| `DoesNotExist` | 해당 키가 존재하지 않음 |
| `Gt` | 값이 지정한 숫자보다 큼 |
| `Lt` | 값이 지정한 숫자보다 작음 |

---

### Pod Affinity / Anti-Affinity

`podAffinity` 는 특정 Pod 와 **같은 Node** 에 배치되도록 한다. `podAntiAffinity` 는 특정 Pod 와 **다른 Node** 에 배치되도록 한다.

`topologyKey` 가 핵심 개념이다. 어떤 단위로 "같은 곳" / "다른 곳" 을 판단할지 결정한다.

| topologyKey | 의미 |
|-------------|------|
| `kubernetes.io/hostname` | 같은 Node (호스트) |
| `topology.kubernetes.io/zone` | 같은 가용 영역(AZ) |
| `topology.kubernetes.io/region` | 같은 리전 |

**실무 예제: 웹서버와 Redis 캐시를 같은 Node 에, 웹서버 replica 를 다른 Node 에 분산**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web-server
  template:
    metadata:
      labels:
        app: web-server
    spec:
      affinity:
        podAffinity:
          # 필수 조건: redis 파드와 같은 노드에 배치 (네트워크 지연 최소화)
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - redis-cache
            topologyKey: kubernetes.io/hostname   # 같은 노드 기준
        podAntiAffinity:
          # 필수 조건: 다른 web-server 파드와 다른 노드에 배치 (고가용성)
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - web-server
            topologyKey: kubernetes.io/hostname   # 각 노드에 최대 1개만
      containers:
      - name: web
        image: nginx:1.25
        ports:
        - containerPort: 80
```

**Redis 캐시 Deployment (web-server 의 podAffinity 대상):**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis-cache
spec:
  replicas: 3
  selector:
    matchLabels:
      app: redis-cache
  template:
    metadata:
      labels:
        app: redis-cache      # web-server 가 이 레이블을 기준으로 affinity 설정
    spec:
      containers:
      - name: redis
        image: redis:7.2
        ports:
        - containerPort: 6379
```

---

### Taint 와 Toleration

Taint 는 Node 에 "기피 조건" 을 설정한다. Pod 는 기본적으로 Taint 가 있는 Node 에 배치되지 않는다. Toleration 은 Pod 가 특정 Taint 를 "참겠다" 고 선언하는 것이다. Toleration 이 있는 Pod 만 해당 Node 에 배치될 수 있다.

**Taint effect 종류:**

| effect | 의미 |
|--------|------|
| `NoSchedule` | 새 Pod 를 이 Node 에 스케줄하지 않는다. 기존 Pod 는 유지된다. |
| `PreferNoSchedule` | 가능하면 이 Node 에 스케줄하지 않는다. 다른 Node 가 없으면 배치될 수 있다. |
| `NoExecute` | 새 Pod 를 스케줄하지 않고, 기존 실행 중인 Pod 도 퇴거(evict)시킨다. |

**Taint 명령어:**

```bash
# Taint 추가: key=value:effect 형식
kubectl taint nodes gpu-node-1 dedicated=gpu-workload:NoSchedule

# Taint 확인
kubectl describe node gpu-node-1 | grep Taints

# Taint 삭제 (effect 뒤에 - 를 붙임)
kubectl taint nodes gpu-node-1 dedicated=gpu-workload:NoSchedule-
```

**실무 예제: GPU 노드에 Taint 걸고 GPU 워크로드만 Toleration**

1단계: GPU 노드에 Taint 추가

```bash
kubectl taint nodes gpu-node-1 dedicated=gpu-workload:NoSchedule
kubectl taint nodes gpu-node-2 dedicated=gpu-workload:NoSchedule
```

2단계: 일반 Pod 는 GPU 노드에 배치되지 않음 (Toleration 없음)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: normal-web-pod
spec:
  # tolerations 없음 -> gpu-node-1, gpu-node-2 에 배치 불가
  containers:
  - name: web
    image: nginx:1.25
```

3단계: GPU 워크로드 Pod 는 Toleration 을 선언하여 GPU 노드에 배치

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-training-pod
spec:
  tolerations:
  - key: dedicated
    operator: Equal               # key 의 값이 value 와 정확히 일치할 때 허용
    value: gpu-workload
    effect: NoSchedule            # NoSchedule taint 를 허용
  nodeSelector:
    accelerator: nvidia-tesla-v100  # nodeSelector 와 함께 사용하여 GPU 노드 지정
  containers:
  - name: training
    image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
    resources:
      limits:
        nvidia.com/gpu: 2
```

**operator 종류:**

| operator | 의미 |
|----------|------|
| `Equal` | key 와 value 가 모두 일치해야 허용 |
| `Exists` | key 만 일치하면 허용 (value 는 무시) |

**tolerationSeconds 예제 (NoExecute 에서만 유효):**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-with-toleration-timeout
spec:
  tolerations:
  - key: node.kubernetes.io/not-ready
    operator: Exists
    effect: NoExecute
    tolerationSeconds: 300    # 노드가 NotReady 상태가 된 후 300초 뒤에 evict
  containers:
  - name: app
    image: nginx:1.25
```

---

### Topology Spread Constraints

Pod 를 여러 가용 영역(AZ) 또는 Node 에 균등하게 분배할 때 사용한다. `podAntiAffinity` 보다 더 세밀한 분산 제어가 가능하다.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
spec:
  replicas: 6
  selector:
    matchLabels:
      app: api-server
  template:
    metadata:
      labels:
        app: api-server
    spec:
      topologySpreadConstraints:
      # 제약 1: 가용 영역(AZ) 간 파드 수 차이를 최대 1개로 유지
      - maxSkew: 1                              # 허용되는 최대 편차
        topologyKey: topology.kubernetes.io/zone  # AZ 기준으로 분산
        whenUnsatisfiable: DoNotSchedule        # 조건 만족 불가시 스케줄 거부
        labelSelector:
          matchLabels:
            app: api-server
      # 제약 2: 노드 간 파드 수 차이를 최대 1개로 유지
      - maxSkew: 1
        topologyKey: kubernetes.io/hostname     # 노드 기준으로 분산
        whenUnsatisfiable: ScheduleAnyway       # 조건 만족 불가시에도 스케줄 허용
        labelSelector:
          matchLabels:
            app: api-server
      containers:
      - name: api
        image: my-api:1.0.0
        ports:
        - containerPort: 8080
```

**whenUnsatisfiable 옵션:**

| 옵션 | 의미 |
|------|------|
| `DoNotSchedule` | 조건을 만족할 수 없으면 Pod 를 스케줄하지 않는다 (strict) |
| `ScheduleAnyway` | 조건을 만족할 수 없어도 최선의 Node 에 스케줄한다 (soft) |

---

## Custom Resource Definition (CRD)

CRD(Custom Resource Definition) 는 Kubernetes API 를 확장하는 방법이다. 기본 제공되는 Pod, Deployment, Service 외에 사용자가 직접 새로운 리소스 타입을 정의할 수 있다. CRD 를 등록하면 `kubectl` 과 Kubernetes API 를 통해 커스텀 리소스를 CRUD 할 수 있다.

**CRD 정의 YAML (validation schema 포함):**

```yaml
# certificate-crd.yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: certificates.tls.example.com   # <plural>.<group> 형식이어야 함
spec:
  group: tls.example.com               # API 그룹명
  versions:
  - name: v1
    served: true                       # 이 버전으로 API 요청을 처리할지 여부
    storage: true                      # 이 버전으로 etcd 에 저장할지 여부 (한 버전만 true)
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            required:
            - domain
            - secretName
            properties:
              domain:
                type: string
                description: TLS 인증서를 발급할 도메인
              secretName:
                type: string
                description: 인증서를 저장할 Secret 이름
              issuer:
                type: string
                enum:
                - letsencrypt-prod
                - letsencrypt-staging
                - self-signed
                default: letsencrypt-prod
              renewBefore:
                type: string
                description: "만료 며칠 전에 갱신할지 (예: 30d)"
                default: "30d"
          status:
            type: object
            properties:
              phase:
                type: string
                enum:
                - Pending
                - Issuing
                - Ready
                - Failed
              expiryDate:
                type: string
              message:
                type: string
    additionalPrinterColumns:           # kubectl get 출력에 추가할 열
    - name: Domain
      type: string
      jsonPath: .spec.domain
    - name: Status
      type: string
      jsonPath: .status.phase
    - name: Expiry
      type: string
      jsonPath: .status.expiryDate
    - name: Age
      type: date
      jsonPath: .metadata.creationTimestamp
  scope: Namespaced                    # Namespaced 또는 Cluster
  names:
    plural: certificates               # kubectl get certificates
    singular: certificate              # kubectl get certificate
    kind: Certificate                  # YAML 에서 사용하는 Kind
    shortNames:
    - cert                             # kubectl get cert (단축 이름)
```

**CRD 적용 및 확인:**

```bash
# CRD 등록
kubectl apply -f certificate-crd.yaml

# 등록된 CRD 확인
kubectl get crds
# NAME                          CREATED AT
# certificates.tls.example.com   2024-01-15T10:00:00Z

# CRD 상세 정보
kubectl describe crd certificates.tls.example.com
```

**커스텀 리소스(CR) 생성:**

```yaml
# my-certificate.yaml
apiVersion: tls.example.com/v1
kind: Certificate
metadata:
  name: my-app-cert
  namespace: production
spec:
  domain: myapp.example.com            # 인증서 발급 대상 도메인
  secretName: myapp-tls-secret        # 인증서를 저장할 Secret 이름
  issuer: letsencrypt-prod             # 발급 기관
  renewBefore: "30d"                   # 만료 30일 전 자동 갱신
```

```bash
# 커스텀 리소스 생성
kubectl apply -f my-certificate.yaml

# 조회 (shortName 사용 가능)
kubectl get cert -n production
# NAME           DOMAIN                STATUS    EXPIRY       AGE
# my-app-cert    myapp.example.com     Pending              5s

# 상세 조회
kubectl describe cert my-app-cert -n production
```

CRD 만 등록해서는 실제로 아무 동작도 하지 않는다. 커스텀 리소스의 상태를 감시하고 원하는 동작을 수행하는 **Controller** 가 필요하다. 이것이 Operator 패턴이다.

---

## Operator 패턴

Operator 는 CRD + Custom Controller 의 조합이다. 운영자(Operator)가 수동으로 하는 반복적인 운영 작업(배포, 스케일링, 백업, 장애 복구 등)을 자동화한다.

### 동작 원리: Reconciliation Loop

```
Watch (감시) -> Compare (비교) -> Act (행동)
```

1. **Watch**: Controller 가 특정 리소스(CRD 포함)의 변경을 감시한다.
2. **Compare**: 현재 상태(Current State) 와 원하는 상태(Desired State) 를 비교한다.
3. **Act**: 차이가 있으면 현재 상태를 원하는 상태로 맞추는 작업을 수행한다.

이 루프를 **Reconciliation Loop** 라고 한다. 멱등성(idempotent)을 가져야 한다. 즉, 여러 번 실행해도 결과가 같아야 한다.

**Controller 의사코드 (Go 스타일):**

```go
// Reconcile 함수는 Certificate 리소스가 생성/변경/삭제될 때마다 호출된다
func (r *CertificateReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    // 1. 커스텀 리소스 조회
    cert := &tlsv1.Certificate{}
    if err := r.Get(ctx, req.NamespacedName, cert); err != nil {
        // 리소스가 삭제된 경우 무시
        return ctrl.Result{}, client.IgnoreNotFound(err)
    }

    // 2. 현재 상태 확인: Secret 이 이미 존재하는지 확인
    secret := &corev1.Secret{}
    err := r.Get(ctx, types.NamespacedName{
        Name:      cert.Spec.SecretName,
        Namespace: cert.Namespace,
    }, secret)

    if errors.IsNotFound(err) {
        // 3. Secret 이 없으면 인증서 발급 시작
        if err := r.issueCertificate(ctx, cert); err != nil {
            return ctrl.Result{}, err
        }
        // 상태 업데이트
        cert.Status.Phase = "Issuing"
        r.Status().Update(ctx, cert)
        // 1분 후 재확인
        return ctrl.Result{RequeueAfter: time.Minute}, nil
    }

    // 4. 인증서 만료 여부 확인
    if r.shouldRenew(secret, cert.Spec.RenewBefore) {
        r.renewCertificate(ctx, cert, secret)
        return ctrl.Result{RequeueAfter: 24 * time.Hour}, nil
    }

    // 5. 정상 상태
    cert.Status.Phase = "Ready"
    r.Status().Update(ctx, cert)
    return ctrl.Result{RequeueAfter: 24 * time.Hour}, nil
}
```

### 유명한 Operator 들

| Operator | 용도 | 설치 |
|----------|------|------|
| cert-manager | TLS 인증서 자동 발급 및 갱신 | `helm install cert-manager jetstack/cert-manager` |
| prometheus-operator | Prometheus + Alertmanager + Grafana 관리 | kube-prometheus-stack |
| Strimzi | Kafka 클러스터 관리 | `helm install strimzi-kafka-operator strimzi/strimzi-kafka-operator` |
| ArgoCD | GitOps 배포 자동화 | `kubectl apply -f install.yaml` |
| Zalando Postgres Operator | PostgreSQL 클러스터 관리 | `helm install postgres-operator postgres-operator-charts/postgres-operator` |

### Operator SDK 와 KubeBuilder

Operator 를 직접 개발할 때 두 가지 주요 프레임워크가 있다.

**KubeBuilder** (권장, CNCF 공식):

```bash
# KubeBuilder 설치
curl -L -o kubebuilder "https://go.kubebuilder.io/dl/latest/$(go env GOOS)/$(go env GOARCH)"
chmod +x kubebuilder && sudo mv kubebuilder /usr/local/bin/

# 새 프로젝트 생성
kubebuilder init --domain tls.example.com --repo github.com/myorg/cert-operator

# API (CRD + Controller) 생성
kubebuilder create api --group tls --version v1 --kind Certificate

# 코드 생성 (DeepCopy 등)
make generate

# CRD manifest 생성
make manifests

# 로컬에서 실행 (테스트용)
make run
```

**Operator SDK**:

```bash
# Go 기반 Operator 초기화
operator-sdk init --domain tls.example.com --repo github.com/myorg/cert-operator

# API 생성
operator-sdk create api --group tls --version v1 --kind Certificate --resource --controller
```

---

## Admission Webhook

Admission Webhook 은 리소스가 생성/수정/삭제될 때 kube-apiserver 가 외부 서버에 검증 또는 변경 요청을 보내는 메커니즘이다. 두 종류가 있다.

| 종류 | 역할 |
|------|------|
| MutatingAdmissionWebhook | 리소스를 **변경(Mutate)** 한다. 예: 사이드카 자동 주입, 기본값 설정 |
| ValidatingAdmissionWebhook | 리소스를 **검증(Validate)** 한다. 예: 필수 레이블 확인, 보안 정책 강제 |

**요청 처리 흐름:**

```
kubectl apply
    |
    v
kube-apiserver
    |
    +-> Authentication (인증)
    |
    +-> Authorization (인가: RBAC)
    |
    +-> MutatingAdmissionWebhook  <-- 외부 Webhook 서버 호출 (변경)
    |       |
    |       +-> 사이드카 컨테이너 추가
    |       +-> 기본 레이블/어노테이션 추가
    |       +-> 리소스 요청값 자동 설정
    |
    +-> Object Schema Validation
    |
    +-> ValidatingAdmissionWebhook <-- 외부 Webhook 서버 호출 (검증)
    |       |
    |       +-> 정책 위반 여부 확인
    |       +-> 필수 레이블 존재 확인
    |
    +-> etcd 에 저장
```

### MutatingWebhookConfiguration

**실무 예제: 모든 Pod 에 자동으로 로그 수집 사이드카 주입**

```yaml
# mutating-webhook.yaml
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingWebhookConfiguration
metadata:
  name: sidecar-injector-webhook
webhooks:
- name: sidecar-injector.logging.example.com
  admissionReviewVersions: ["v1"]
  clientConfig:
    service:
      name: sidecar-injector-svc     # Webhook 서버 Service 이름
      namespace: logging-system
      path: "/inject"                 # Webhook 서버의 엔드포인트
    caBundle: <base64-encoded-CA>    # Webhook 서버 TLS CA 인증서
  rules:
  - apiGroups: [""]
    apiVersions: ["v1"]
    operations: ["CREATE"]            # Pod 생성 시에만 호출
    resources: ["pods"]
  namespaceSelector:
    matchLabels:
      sidecar-injection: enabled      # 이 레이블이 있는 Namespace 에서만 동작
  failurePolicy: Fail                 # Webhook 실패 시 요청 거부 (Ignore 는 무시)
  sideEffects: None
```

**ValidatingWebhookConfiguration 예제: 필수 레이블 강제:**

```yaml
# validating-webhook.yaml
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: required-labels-validator
webhooks:
- name: validate-labels.policy.example.com
  admissionReviewVersions: ["v1"]
  clientConfig:
    service:
      name: label-validator-svc
      namespace: policy-system
      path: "/validate"
    caBundle: <base64-encoded-CA>
  rules:
  - apiGroups: ["apps"]
    apiVersions: ["v1"]
    operations: ["CREATE", "UPDATE"]   # 생성 및 수정 시 검증
    resources: ["deployments"]
  namespaceSelector:
    matchExpressions:
    - key: kubernetes.io/metadata.name
      operator: NotIn
      values:
      - kube-system                    # 시스템 네임스페이스는 제외
      - kube-public
  failurePolicy: Fail
  sideEffects: None
```

**Webhook 서버가 반환하는 AdmissionResponse 구조 (Go):**

```go
// Mutating Webhook: 사이드카 추가
func handleInject(w http.ResponseWriter, r *http.Request) {
    // ... AdmissionRequest 파싱 ...

    // JSON Patch 로 사이드카 컨테이너 추가
    patch := []map[string]interface{}{
        {
            "op":   "add",
            "path": "/spec/containers/-",
            "value": map[string]interface{}{
                "name":  "fluentd-sidecar",
                "image": "fluent/fluentd:v1.16",
                "volumeMounts": []map[string]interface{}{
                    {"name": "app-logs", "mountPath": "/var/log/app"},
                },
            },
        },
    }
    patchBytes, _ := json.Marshal(patch)
    patchType := admissionv1.PatchTypeJSONPatch

    response := &admissionv1.AdmissionResponse{
        UID:       request.UID,
        Allowed:   true,              // 요청 허용
        Patch:     patchBytes,        // 변경 내용 (JSON Patch)
        PatchType: &patchType,
    }
    // ... response 반환 ...
}
```

---

## Custom Metrics 와 KEDA

### Custom Metrics API

기본 metrics-server 는 CPU/메모리만 제공한다. HPA 에서 비즈니스 메트릭(요청 수, 큐 길이 등)을 사용하려면 Custom Metrics API 서버가 필요하다.

```
HPA
 |
 +-> /apis/metrics.k8s.io          (CPU/메모리 - metrics-server)
 |
 +-> /apis/custom.metrics.k8s.io   (커스텀 메트릭 - Prometheus Adapter 등)
 |
 +-> /apis/external.metrics.k8s.io (외부 메트릭 - Kafka lag, SQS depth 등)
```

### Prometheus Adapter 설정

Prometheus 에서 수집한 메트릭을 Kubernetes Custom Metrics API 로 노출한다.

```yaml
# prometheus-adapter-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: adapter-config
  namespace: monitoring
data:
  config.yaml: |
    rules:
    # HTTP 요청 수를 커스텀 메트릭으로 노출
    - seriesQuery: 'http_requests_total{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace:
            resource: namespace
          pod:
            resource: pod
      name:
        matches: "^(.*)_total"
        as: "${1}_per_second"        # http_requests_per_second 로 노출
      metricsQuery: 'rate(<<.Series>>{<<.LabelMatchers>>}[2m])'
```

**Custom Metrics 를 사용하는 HPA:**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa-custom
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-server
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second    # Prometheus Adapter 가 노출한 메트릭
      target:
        type: AverageValue
        averageValue: "1000"              # 파드당 초당 1000 요청이 되면 스케일 아웃
```

### KEDA (Kubernetes Event-Driven Autoscaler)

KEDA 는 Kafka, RabbitMQ, AWS SQS, Redis 등 다양한 이벤트 소스를 기반으로 Pod 를 0개까지 스케일 다운하거나 수천 개까지 스케일 아웃할 수 있는 오토스케일러다.

**KEDA 설치:**

```bash
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
helm install keda kedacore/keda --namespace keda --create-namespace
```

**KEDA 핵심 리소스:**

| 리소스 | 역할 |
|--------|------|
| ScaledObject | Deployment/StatefulSet 을 스케일링 대상으로 지정 |
| ScaledJob | Job 을 이벤트마다 생성 |
| TriggerAuthentication | Scaler 인증 정보 (API 키, 연결 문자열 등) 관리 |

**실무 예제: Kafka Consumer Lag 기반 자동 스케일링**

```yaml
# kafka-consumer-scaledobject.yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: kafka-consumer-scaler
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: order-consumer              # 스케일링 대상 Deployment
  pollingInterval: 15                 # 메트릭 폴링 간격 (초)
  cooldownPeriod: 60                  # 스케일 다운 대기 시간 (초)
  minReplicaCount: 0                  # 최소 파드 수 (0이면 유휴 시 완전히 내림)
  maxReplicaCount: 50                 # 최대 파드 수
  fallback:
    failureThreshold: 3               # 메트릭 수집 실패 3번 이후
    replicas: 5                       # 폴백으로 5개 유지
  triggers:
  - type: kafka
    metadata:
      bootstrapServers: kafka-broker-0.kafka:9092,kafka-broker-1.kafka:9092
      consumerGroup: order-processor-group   # Consumer Group ID
      topic: orders                           # 구독 중인 토픽
      lagThreshold: "100"                     # 파드당 허용 lag 임계값
      offsetResetPolicy: latest
    authenticationRef:
      name: kafka-auth                        # TriggerAuthentication 참조
```

**TriggerAuthentication (Kafka SASL/SSL 인증):**

```yaml
# kafka-auth.yaml
apiVersion: keda.sh/v1alpha1
kind: TriggerAuthentication
metadata:
  name: kafka-auth
  namespace: production
spec:
  secretTargetRef:
  - parameter: sasl                   # KEDA 파라미터명
    name: kafka-credentials           # Secret 이름
    key: sasl-mechanism               # Secret 의 키
  - parameter: username
    name: kafka-credentials
    key: username
  - parameter: password
    name: kafka-credentials
    key: password
  - parameter: tls
    name: kafka-credentials
    key: tls
```

**스케일링 동작 확인:**

```bash
# ScaledObject 상태 확인
kubectl get scaledobject -n production
# NAME                     SCALETARGET       MIN   MAX   READY   ACTIVE
# kafka-consumer-scaler    order-consumer    0     50    True    True

# 현재 메트릭 확인
kubectl get hpa -n production
# NAME                            REFERENCE                  TARGETS     MINPODS   MAXPODS
# keda-hpa-kafka-consumer-scaler  Deployment/order-consumer  850/100     0         50
```

**SQS 기반 스케일링 예제:**

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: sqs-consumer-scaler
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sqs-worker
  minReplicaCount: 0
  maxReplicaCount: 30
  triggers:
  - type: aws-sqs-queue
    metadata:
      queueURL: https://sqs.ap-northeast-2.amazonaws.com/123456789/my-queue
      queueLength: "50"               # 파드당 처리할 메시지 수
      awsRegion: ap-northeast-2
    authenticationRef:
      name: aws-keda-auth
```

---

## Custom Scheduler

기본 `kube-scheduler` 대신 자체 스케줄링 로직을 사용하고 싶을 때 Custom Scheduler 를 만들 수 있다.

### schedulerName 확인

Pod 이 어떤 스케줄러를 사용하는지 확인하려면:

```bash
$ kubectl get pod <pod-name> -o yaml | grep schedulerName
schedulerName: default-scheduler   # default-scheduler = kube-scheduler
```

### Custom Scheduler 를 사용하는 Pod

Pod 의 `schedulerName` 필드에 커스텀 스케줄러 이름을 지정하면 된다.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: custom-scheduled-pod
spec:
  schedulerName: my-custom-scheduler   # 기본 스케줄러 대신 사용
  containers:
  - name: nginx-container
    image: nginx
```

이 Pod 은 `my-custom-scheduler` 가 실행 중이지 않으면 `Pending` 상태로 남는다.

### Custom Scheduler 의 동작 원리

커스텀 스케줄러는 다음 과정을 반복한다:

```
1. kube-apiserver 를 통해 새로 생성된 Pod 을 감시
2. Pod 의 nodeName 이 비어 있고, schedulerName 이 자신과 일치하면 스케줄링 시작
3. Node Filtering: 조건에 맞지 않는 노드 제외 (리소스 부족, taint 등)
4. Node Scoring: 남은 노드들에 점수를 매겨 최적 노드 선택
5. kube-apiserver 를 통해 Pod 의 nodeName 을 선택된 노드로 설정 (Binding)
```

### 개발 방법

| 방법 | 설명 | 난이도 |
|------|------|--------|
| **Scheduling Framework** | kube-scheduler 에 플러그인을 추가하는 공식 방법. Filter, Score, Bind 등 확장 포인트 제공 | 중간 |
| **Scheduler Extender** | kube-scheduler 가 외부 HTTP 서버에 필터링/스코어링을 위임. 기존 스케줄러를 수정하지 않아도 됨 | 낮음 |
| **kube-scheduler 소스 수정** | kube-scheduler 코드를 직접 수정하여 빌드. 가장 자유도가 높지만 유지보수가 어려움 | 높음 |

참고 자료:
- [Scheduling Framework | kubernetes.io](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/)
- [Scheduler Extender Design Proposal | github](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/scheduling/scheduler_extender.md)
- [Custom Scheduler Python 예제 | github](https://github.com/alicek106/start-docker-kubernetes/blob/master/chapter11-2/custom-scheduler-python/__main__.py)

---

## Helm

### Helm 이란

Helm 은 Kubernetes 의 패키지 매니저다. apt/yum 처럼 애플리케이션을 패키지(Chart) 단위로 관리한다. 여러 YAML 파일로 구성된 복잡한 애플리케이션을 하나의 Chart 로 묶어 설치, 업그레이드, 롤백, 삭제를 간편하게 처리한다.

**핵심 개념:**

| 개념 | 설명 |
|------|------|
| Chart | Kubernetes 리소스를 묶은 패키지 (디렉토리 구조) |
| Repository | Chart 를 배포하는 저장소 |
| Release | Chart 를 클러스터에 설치한 인스턴스 (이름 있음) |
| Values | Chart 의 기본값을 재정의하는 설정 파일 |

### Chart 구조

```
my-app/
├── Chart.yaml          # Chart 메타데이터 (이름, 버전, 설명)
├── values.yaml         # 기본 설정값
├── templates/          # Kubernetes 리소스 템플릿 (Go template)
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── _helpers.tpl    # 재사용 가능한 템플릿 함수 정의
│   └── NOTES.txt       # 설치 후 출력되는 안내 메시지
├── charts/             # 의존하는 하위 Chart (subchart)
└── .helmignore         # Helm 이 무시할 파일 목록
```

**Chart.yaml 예시:**

```yaml
# Chart.yaml
apiVersion: v2                        # Helm 3 은 v2
name: my-app
description: 내 애플리케이션 Helm Chart
type: application                     # application 또는 library
version: 1.2.0                        # Chart 버전 (SemVer)
appVersion: "2.0.1"                   # 애플리케이션 버전
dependencies:
- name: postgresql
  version: "12.x.x"
  repository: https://charts.bitnami.com/bitnami
  condition: postgresql.enabled        # values.yaml 의 postgresql.enabled 가 true 일 때만 설치
```

**values.yaml 예시:**

```yaml
# values.yaml - 기본값 정의
replicaCount: 2

image:
  repository: my-registry/my-app
  tag: "2.0.1"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: false                       # 기본적으로 Ingress 비활성화
  host: ""

resources:
  requests:
    cpu: 100m
    memory: 128Mi
  limits:
    cpu: 500m
    memory: 512Mi

postgresql:
  enabled: true                        # PostgreSQL subchart 활성화
  auth:
    database: myapp
    username: myapp
```

**templates/deployment.yaml 예시:**

```yaml
# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "my-app.fullname" . }}    # _helpers.tpl 에 정의된 함수
  labels:
    {{- include "my-app.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}        # values.yaml 의 값 참조
  selector:
    matchLabels:
      {{- include "my-app.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "my-app.selectorLabels" . | nindent 8 }}
    spec:
      containers:
      - name: {{ .Chart.Name }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        imagePullPolicy: {{ .Values.image.pullPolicy }}
        ports:
        - containerPort: 8080
        resources:
          {{- toYaml .Values.resources | nindent 10 }}
```

### 기본 명령어

**Repository 관리:**

```bash
# 공식 stable 저장소 추가
helm repo add stable https://charts.helm.sh/stable

# Bitnami 저장소 추가
helm repo add bitnami https://charts.bitnami.com/bitnami

# 저장소 목록 확인
helm repo list

# 저장소 업데이트 (최신 Chart 목록 동기화)
helm repo update

# Chart 검색
helm search repo nginx
helm search hub postgresql        # Artifact Hub 에서 검색
```

**설치 (install):**

```bash
# 기본 설치
helm install my-release bitnami/nginx

# 네임스페이스 지정 및 자동 생성
helm install my-nginx bitnami/nginx \
  --namespace web \
  --create-namespace

# values.yaml 파일로 설정 재정의
helm install my-nginx bitnami/nginx \
  --values custom-values.yaml

# 개별 값 재정의 (--set)
helm install my-nginx bitnami/nginx \
  --set replicaCount=3 \
  --set service.type=LoadBalancer

# 설치 전 렌더링 결과 확인 (dry-run)
helm install my-nginx bitnami/nginx --dry-run --debug

# 설치된 release 목록
helm list -A                       # 모든 네임스페이스
```

**업그레이드 (upgrade):**

```bash
# Chart 버전 업그레이드
helm upgrade my-nginx bitnami/nginx

# 없으면 설치, 있으면 업그레이드 (--install)
helm upgrade --install my-nginx bitnami/nginx \
  --namespace web \
  --create-namespace \
  --values custom-values.yaml

# 변경 내용 확인 (diff - helm-diff 플러그인 필요)
helm diff upgrade my-nginx bitnami/nginx --values custom-values.yaml
```

**롤백 (rollback):**

```bash
# 릴리즈 히스토리 확인
helm history my-nginx
# REVISION  STATUS      CHART          APP VERSION  DESCRIPTION
# 1         superseded  nginx-15.0.0   1.25.0       Install complete
# 2         superseded  nginx-15.1.0   1.25.1       Upgrade complete
# 3         deployed    nginx-15.2.0   1.25.2       Upgrade complete

# 특정 리비전으로 롤백
helm rollback my-nginx 2

# 바로 이전 리비전으로 롤백
helm rollback my-nginx
```

**삭제 및 정보 조회:**

```bash
# 릴리즈 삭제
helm uninstall my-nginx -n web

# 릴리즈 상세 정보
helm status my-nginx -n web

# 실제로 렌더링된 YAML 확인
helm get manifest my-nginx -n web

# 적용된 values 확인
helm get values my-nginx -n web
```

**Chart 개발:**

```bash
# 새 Chart 생성
helm create my-app

# Chart 문법 검사
helm lint my-app/

# 로컬 Chart 설치
helm install my-release ./my-app/

# Chart 패키징 (.tgz 생성)
helm package my-app/

# 특정 values 파일로 렌더링 결과 확인
helm template my-release ./my-app/ --values prod-values.yaml
```

**환경별 values 파일 관리 패턴:**

```bash
# values.yaml          - 공통 기본값
# values-dev.yaml      - 개발 환경 재정의
# values-staging.yaml  - 스테이징 환경 재정의
# values-prod.yaml     - 프로덕션 환경 재정의

# 개발 환경 배포
helm upgrade --install my-app ./my-app \
  --values values.yaml \
  --values values-dev.yaml \
  -n development

# 프로덕션 환경 배포
helm upgrade --install my-app ./my-app \
  --values values.yaml \
  --values values-prod.yaml \
  -n production
```
