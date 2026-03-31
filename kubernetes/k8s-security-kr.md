# Kubernetes 보안 (Korean)

- [보안 개요](#보안-개요)
- [인증 (Authentication)](#인증-authentication)
- [ServiceAccount](#serviceaccount)
- [RBAC (Role-Based Access Control)](#rbac-role-based-access-control)
- [PodSecurityStandard (PSS)](#podsecuritystandard-pss)
- [SecurityContext](#securitycontext)
- [Secret 관리 모범 사례](#secret-관리-모범-사례)
- [NetworkPolicy 로 네트워크 보안](#networkpolicy-로-네트워크-보안)
- [Image Security](#image-security)

---

## 보안 개요

Kubernetes 보안은 **4C 모델**로 설명한다. 각 계층은 하위 계층의 보안을 전제로 한다.

| 계층 | 설명 |
|---|---|
| Cloud | 클라우드 인프라 (IAM, 네트워크 ACL, 스토리지 암호화) |
| Cluster | Kubernetes 클러스터 (API 서버, etcd, kubelet 설정) |
| Container | 컨테이너 런타임, 이미지, SecurityContext |
| Code | 애플리케이션 코드, 의존성, 시크릿 처리 |

### 주요 공격 표면

- **API 서버**: 모든 요청의 진입점. 인증/인가 미흡 시 클러스터 전체가 위험
- **etcd**: 클러스터 전체 상태 저장. 암호화 없이 노출되면 모든 Secret 탈취 가능
- **kubelet**: 각 노드에서 실행. 인증 없이 노출되면 임의 컨테이너 실행 가능
- **네트워크**: Pod 간 기본 통신은 모두 허용. NetworkPolicy 없으면 측면 이동 가능

---

## 인증 (Authentication)

Kubernetes 는 여러 인증 방식을 지원한다. 클러스터가 요청을 받으면 설정된 인증 방식을 순서대로 시도한다.

### X.509 인증서

사람 사용자(human user)를 위한 가장 일반적인 인증 방식이다. `~/.kube/config` 에 인증서가 내장된다.

```bash
# 새 사용자 키와 CSR 생성
openssl genrsa -out developer.key 2048
openssl req -new -key developer.key -out developer.csr \
  -subj "/CN=developer/O=dev-team"

# Kubernetes CertificateSigningRequest 리소스로 서명 요청
cat developer.csr | base64 | tr -d '\n'
```

```yaml
# developer-csr.yaml
apiVersion: certificates.k8s.io/v1
kind: CertificateSigningRequest
metadata:
  name: developer                          # CSR 리소스 이름
spec:
  request: <base64-encoded-csr>            # 위에서 생성한 base64 CSR 값
  signerName: kubernetes.io/kube-apiserver-client
  expirationSeconds: 86400                 # 1일 유효기간
  usages:
  - client auth                            # 클라이언트 인증 용도
```

```bash
# CSR 승인
kubectl certificate approve developer

# 인증서 추출
kubectl get csr developer -o jsonpath='{.status.certificate}' | base64 -d > developer.crt

# kubeconfig 에 자격증명 등록
kubectl config set-credentials developer \
  --client-certificate=developer.crt \
  --client-key=developer.key
```

### ServiceAccount 토큰

Pod 가 API 서버에 접근할 때 사용하는 인증 방식이다. Kubernetes 1.24+ 부터는 시간 제한이 있는 projected token 을 사용한다.

```bash
# ServiceAccount 토큰 수동 확인 (디버깅 용도)
kubectl create token my-serviceaccount --duration=1h
```

### OIDC (OpenID Connect)

기업 환경에서 SSO 를 통해 Kubernetes 에 접근할 때 사용한다. GitHub, Google, Dex 같은 외부 IDP 와 연동한다.

```yaml
# kube-apiserver 설정 플래그 (예: /etc/kubernetes/manifests/kube-apiserver.yaml)
spec:
  containers:
  - command:
    - kube-apiserver
    - --oidc-issuer-url=https://accounts.google.com   # OIDC 발급자 URL
    - --oidc-client-id=kubernetes                      # 클라이언트 ID
    - --oidc-username-claim=email                      # 사용자 이름으로 쓸 클레임
    - --oidc-groups-claim=groups                       # 그룹으로 쓸 클레임
```

### kubeconfig 파일 구조

```yaml
# ~/.kube/config
apiVersion: v1
kind: Config
clusters:
- cluster:
    certificate-authority-data: LS0t...    # CA 인증서 (base64)
    server: https://k8s-api.example.com:6443
  name: production-cluster
contexts:
- context:
    cluster: production-cluster
    user: developer
    namespace: default                     # 기본 namespace 지정 가능
  name: dev-context
current-context: dev-context               # 현재 활성 컨텍스트
users:
- name: developer
  user:
    client-certificate-data: LS0t...       # 클라이언트 인증서 (base64)
    client-key-data: LS0t...               # 클라이언트 키 (base64)
```

```bash
# kubeconfig 조작 명령어
kubectl config get-contexts                # 모든 컨텍스트 조회
kubectl config use-context dev-context     # 컨텍스트 전환
kubectl config set-context --current --namespace=dev  # 현재 컨텍스트의 기본 namespace 변경
```

---

## ServiceAccount

ServiceAccount 는 Pod 가 Kubernetes API 에 접근할 때 사용하는 아이덴티티다. 사람이 아닌 프로세스를 위한 계정이다.

### 기본 ServiceAccount

모든 namespace 에는 `default` ServiceAccount 가 자동 생성된다. Pod 에 ServiceAccount 를 지정하지 않으면 `default` 가 자동 마운트된다.

```bash
# 기본 ServiceAccount 확인
kubectl get serviceaccount default -o yaml

# Pod 내부에서 자동 마운트된 토큰 위치
# /var/run/secrets/kubernetes.io/serviceaccount/token
```

### 커스텀 ServiceAccount 생성

```yaml
# app-serviceaccount.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: app-sa                             # ServiceAccount 이름
  namespace: production                    # 특정 namespace 에 생성
  labels:
    app: my-app
```

```bash
kubectl apply -f app-serviceaccount.yaml
kubectl get serviceaccount -n production
```

### Pod 에 ServiceAccount 할당

```yaml
# pod-with-sa.yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
  namespace: production
spec:
  serviceAccountName: app-sa              # 위에서 만든 ServiceAccount 할당
  containers:
  - name: app
    image: nginx:1.25
```

### automountServiceAccountToken 비활성화

Pod 가 API 서버에 접근할 필요가 없다면 토큰 자동 마운트를 꺼서 공격 표면을 줄인다.

```yaml
# secure-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
  namespace: production
spec:
  serviceAccountName: app-sa
  automountServiceAccountToken: false      # 토큰 자동 마운트 비활성화 (Pod 레벨)
  containers:
  - name: app
    image: nginx:1.25
    securityContext:
      runAsNonRoot: true
      runAsUser: 1000
```

ServiceAccount 레벨에서도 비활성화할 수 있다.

```yaml
# no-automount-sa.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: no-mount-sa
  namespace: production
automountServiceAccountToken: false        # 이 SA 를 사용하는 모든 Pod 에 적용
```

---

## RBAC (Role-Based Access Control)

RBAC 는 누가(Subject) 무엇을(Resource) 어떻게(Verb) 할 수 있는지를 정의한다.

### Role vs ClusterRole

| 구분 | 적용 범위 | 사용 시점 |
|---|---|---|
| Role | 특정 namespace | 특정 팀/서비스에 namespace 범위 권한 부여 |
| ClusterRole | 전체 cluster | 운영자, 모니터링 도구, namespace 에 속하지 않는 리소스(Node, PV 등) |

### RoleBinding vs ClusterRoleBinding

| 구분 | 대상 | 사용 시점 |
|---|---|---|
| RoleBinding | 특정 namespace | Role 또는 ClusterRole 을 특정 namespace 에서만 부여 |
| ClusterRoleBinding | 전체 cluster | ClusterRole 을 클러스터 전체에 부여 |

> 팁: ClusterRole 을 여러 namespace 에서 재사용하되 범위를 제한하려면 `RoleBinding` 으로 ClusterRole 을 바인딩한다.

### 실무 예제 1: 특정 namespace 에서 Pod 만 읽을 수 있는 개발자 Role

```yaml
# developer-role.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader                         # Role 이름
  namespace: development                   # 이 namespace 에서만 유효
rules:
- apiGroups: [""]                          # "" 는 core API group (Pod, Service, ConfigMap 등)
  resources: ["pods", "pods/log"]          # 접근 허용할 리소스 목록
  verbs: ["get", "list", "watch"]          # 허용할 동작 (읽기 전용)
```

```yaml
# developer-rolebinding.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: developer-pod-reader               # RoleBinding 이름
  namespace: development                   # 같은 namespace 지정
subjects:
- kind: User                               # 일반 사용자에게 부여
  name: alice                              # 사용자 이름 (X.509 CN 또는 OIDC 클레임)
  apiGroup: rbac.authorization.k8s.io
- kind: ServiceAccount                     # ServiceAccount 에도 동시 부여 가능
  name: ci-runner
  namespace: development
roleRef:
  kind: Role                               # Role 참조
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```

```bash
kubectl apply -f developer-role.yaml
kubectl apply -f developer-rolebinding.yaml

# 권한 확인
kubectl auth can-i list pods --namespace=development --as=alice
# yes

kubectl auth can-i delete pods --namespace=development --as=alice
# no
```

### 실무 예제 2: 모든 namespace 에서 Deployment 를 관리할 수 있는 운영자 ClusterRole

```yaml
# operator-clusterrole.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: deployment-manager                 # ClusterRole 이름
rules:
- apiGroups: ["apps"]                      # Deployment, ReplicaSet, DaemonSet 등은 "apps" group
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: [""]                          # Pod 조회는 core group
  resources: ["pods", "pods/log", "events"]
  verbs: ["get", "list", "watch"]
```

```yaml
# operator-clusterrolebinding.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: operator-deployment-manager       # ClusterRoleBinding 이름
subjects:
- kind: Group                              # 그룹 단위로 권한 부여 (OIDC 그룹 클레임 활용)
  name: ops-team
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole                        # ClusterRole 참조
  name: deployment-manager
  apiGroup: rbac.authorization.k8s.io
```

```bash
kubectl apply -f operator-clusterrole.yaml
kubectl apply -f operator-clusterrolebinding.yaml

# 권한 확인 - 모든 namespace 에서 가능한지 테스트
kubectl auth can-i create deployments --namespace=production --as-group=ops-team --as=dummy
# yes

kubectl auth can-i delete nodes --as-group=ops-team --as=dummy
# no
```

### kubectl auth can-i 로 권한 확인

```bash
# 현재 사용자의 권한 확인
kubectl auth can-i create pods
kubectl auth can-i delete deployments -n production

# 특정 사용자로 가장(impersonation)해서 확인
kubectl auth can-i list secrets -n kube-system --as=alice

# 특정 ServiceAccount 로 확인
kubectl auth can-i get pods --as=system:serviceaccount:default:app-sa

# 현재 사용자가 할 수 있는 모든 동작 조회
kubectl auth can-i --list
kubectl auth can-i --list -n development
```

---

## PodSecurityStandard (PSS)

PodSecurityStandard 는 Kubernetes 1.25 에서 PodSecurityPolicy 를 대체한 기능이다. namespace 레이블 하나로 보안 프로필을 적용한다.

### 세 가지 프로필

| 프로필 | 대상 | 특징 |
|---|---|---|
| Privileged | 시스템 컴포넌트, 레거시 앱 | 제한 없음. 기본값 |
| Baseline | 일반 워크로드 | 알려진 위험 설정(hostNetwork, hostPID, privileged 컨테이너)만 차단 |
| Restricted | 보안 민감 워크로드 | 엄격한 제한. runAsNonRoot, seccompProfile 필수 |

### 세 가지 모드

| 모드 | 동작 |
|---|---|
| enforce | 위반 Pod 생성 거부 |
| audit | 허용하되 감사 로그에 기록 |
| warn | 허용하되 경고 메시지 출력 |

### namespace 레이블로 적용

```yaml
# namespace-pss.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: secure-app
  labels:
    # enforce 모드로 restricted 프로필 적용
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: latest

    # 감사 로그에 baseline 위반 기록
    pod-security.kubernetes.io/audit: baseline
    pod-security.kubernetes.io/audit-version: latest

    # warn 으로 baseline 위반 경고 출력
    pod-security.kubernetes.io/warn: baseline
    pod-security.kubernetes.io/warn-version: latest
```

```bash
kubectl apply -f namespace-pss.yaml

# 기존 namespace 에 레이블 추가
kubectl label namespace production \
  pod-security.kubernetes.io/enforce=baseline \
  pod-security.kubernetes.io/warn=restricted

# 위반 여부 dry-run 으로 사전 점검
kubectl apply --dry-run=server -f my-pod.yaml -n secure-app
```

### 단계적 도입 전략

신규 클러스터가 아니라면 바로 enforce 를 적용하면 기존 워크로드가 중단될 수 있다.

```bash
# 1단계: warn 으로 시작해서 위반 목록 파악
kubectl label namespace my-ns pod-security.kubernetes.io/warn=restricted

# 2단계: audit 추가 (감사 로그 수집)
kubectl label namespace my-ns pod-security.kubernetes.io/audit=restricted

# 3단계: 워크로드 수정 완료 후 enforce 적용
kubectl label namespace my-ns pod-security.kubernetes.io/enforce=restricted
```

---

## SecurityContext

SecurityContext 는 Pod 또는 컨테이너 수준에서 보안 관련 설정을 지정한다.

### 설정 적용 범위

| 필드 | Pod 레벨 | 컨테이너 레벨 |
|---|---|---|
| runAsUser | 가능 (기본값) | 가능 (개별 재정의) |
| runAsGroup | 가능 | 가능 |
| runAsNonRoot | 가능 | 가능 |
| fsGroup | 가능 | 불가 |
| readOnlyRootFilesystem | 불가 | 가능 |
| capabilities | 불가 | 가능 |
| seccompProfile | 가능 | 가능 |
| allowPrivilegeEscalation | 불가 | 가능 |

### 완전한 YAML 예제

```yaml
# secure-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: secure-app
  namespace: production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: secure-app
  template:
    metadata:
      labels:
        app: secure-app
    spec:
      # Pod 레벨 SecurityContext
      securityContext:
        runAsUser: 1000                    # UID 1000 으로 실행 (root 금지)
        runAsGroup: 3000                   # GID 3000 으로 실행
        fsGroup: 2000                      # 볼륨을 GID 2000 으로 소유
        runAsNonRoot: true                 # root(UID 0)로 실행 시 Pod 시작 거부
        seccompProfile:
          type: RuntimeDefault             # 컨테이너 런타임 기본 seccomp 프로파일 적용

      serviceAccountName: secure-app-sa
      automountServiceAccountToken: false  # API 접근 불필요 시 토큰 마운트 차단

      containers:
      - name: app
        image: nginx:1.25@sha256:a484...   # digest 로 이미지 고정 (태그 변경 공격 방지)

        # 컨테이너 레벨 SecurityContext (Pod 레벨보다 우선)
        securityContext:
          runAsUser: 1000                  # Pod 레벨과 동일하게 명시적 지정
          runAsNonRoot: true
          readOnlyRootFilesystem: true     # 루트 파일시스템 읽기 전용 (악성코드 기록 차단)
          allowPrivilegeEscalation: false  # setuid/setgid 를 통한 권한 상승 차단
          capabilities:
            drop:
            - ALL                          # 기본 Linux capability 전부 제거
            add:
            - NET_BIND_SERVICE             # 1024 미만 포트 바인딩만 허용 (nginx 필요)

        ports:
        - containerPort: 8080

        # readOnlyRootFilesystem: true 로 인해 쓰기가 필요한 경로는 tmpfs 볼륨 사용
        volumeMounts:
        - name: tmp-dir
          mountPath: /tmp
        - name: cache-dir
          mountPath: /var/cache/nginx

        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 256Mi

      volumes:
      - name: tmp-dir
        emptyDir:
          medium: Memory                   # 메모리 기반 tmpfs (디스크 I/O 차단)
      - name: cache-dir
        emptyDir: {}
```

### 최소 권한 원칙 적용 요약

```bash
# Pod 보안 설정 확인
kubectl get pod secure-app-xxx -o jsonpath='{.spec.securityContext}' | jq .

# 컨테이너가 실제로 어떤 UID 로 실행 중인지 확인
kubectl exec secure-app-xxx -- id
# uid=1000 gid=3000 groups=3000,2000
```

---

## Secret 관리 모범 사례

### etcd 암호화 (EncryptionConfiguration)

기본적으로 Secret 은 etcd 에 평문으로 저장된다. EncryptionConfiguration 으로 암호화해야 한다.

```yaml
# /etc/kubernetes/enc/encryption-config.yaml (kube-apiserver 노드에 위치)
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
- resources:
  - secrets                                # Secret 리소스 암호화 대상
  - configmaps                             # ConfigMap 도 암호화 가능
  providers:
  - aescbc:                                # AES-CBC 암호화 (권장)
      keys:
      - name: key1
        secret: <base64-encoded-32byte-key> # openssl rand -base64 32 로 생성
  - identity: {}                           # 암호화 안 함 (복호화용 fallback 으로 마지막에 위치)
```

```bash
# kube-apiserver 에 암호화 설정 활성화 (kubeadm 클러스터 기준)
# /etc/kubernetes/manifests/kube-apiserver.yaml 에 추가:
# --encryption-provider-config=/etc/kubernetes/enc/encryption-config.yaml

# 기존 Secret 을 암호화된 상태로 재저장
kubectl get secrets --all-namespaces -o json | kubectl replace -f -

# etcd 에서 직접 확인 (암호화 전: plaintext, 암호화 후: k8s:enc:aescbc 로 시작)
ETCDCTL_API=3 etcdctl get /registry/secrets/default/my-secret --print-value-only
```

### 외부 Secret 관리

etcd 암호화만으로는 부족할 때 외부 Secret 관리 시스템과 연동한다.

**External Secrets Operator (ESO)**

AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault 같은 외부 저장소의 Secret 을 Kubernetes Secret 으로 동기화한다.

```yaml
# external-secret.yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: app-secret
  namespace: production
spec:
  refreshInterval: 1h                      # 1시간마다 외부 저장소에서 갱신
  secretStoreRef:
    name: aws-secretsmanager               # SecretStore 이름 (별도 설정 필요)
    kind: ClusterSecretStore
  target:
    name: app-secret                       # 생성될 Kubernetes Secret 이름
    creationPolicy: Owner
  data:
  - secretKey: db-password                 # Kubernetes Secret 의 키 이름
    remoteRef:
      key: prod/app/database               # AWS Secrets Manager 경로
      property: password                   # JSON 값에서 가져올 필드
```

**Sealed Secrets**

GitOps 환경에서 암호화된 Secret 을 Git 에 안전하게 저장한다.

```bash
# kubeseal 로 Secret 암호화
kubectl create secret generic db-cred \
  --from-literal=password=mysecret \
  --dry-run=client -o yaml \
  | kubeseal --format yaml > sealed-db-cred.yaml

# 암호화된 SealedSecret 을 Git 에 커밋하고 클러스터에 적용
kubectl apply -f sealed-db-cred.yaml
```

### Secret 을 환경변수 vs 볼륨으로 주입하는 차이

```yaml
# secret-injection-comparison.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: secret-demo
spec:
  replicas: 1
  selector:
    matchLabels:
      app: secret-demo
  template:
    metadata:
      labels:
        app: secret-demo
    spec:
      containers:
      - name: app
        image: busybox:1.36

        # 방법 1: 환경변수로 주입 (단점: 자식 프로세스로 전파, 로그에 노출 위험)
        env:
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials         # Secret 이름
              key: password                # Secret 의 키
              optional: false              # Secret 없으면 Pod 시작 실패

        # 방법 2: 볼륨으로 마운트 (권장: 런타임에 갱신 반영, 파일 권한 제어 가능)
        volumeMounts:
        - name: secret-vol
          mountPath: /etc/secrets          # 마운트 경로
          readOnly: true                   # 읽기 전용 마운트

      volumes:
      - name: secret-vol
        secret:
          secretName: db-credentials       # 마운트할 Secret 이름
          defaultMode: 0400                # 파일 권한 설정 (소유자 읽기 전용)
          items:
          - key: password
            path: db-password              # /etc/secrets/db-password 로 접근
```

| 비교 항목 | 환경변수 | 볼륨 마운트 |
|---|---|---|
| 런타임 갱신 | 불가 (재시작 필요) | 가능 (kubelet 이 주기적 갱신) |
| 자식 프로세스 전파 | 자동 전파 (위험) | 없음 |
| 로그 노출 위험 | 높음 (`env` 출력 시) | 낮음 |
| 파일 권한 제어 | 불가 | 가능 |

### RBAC 으로 Secret 접근 제한

```yaml
# secret-reader-role.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: secret-reader
  namespace: production
rules:
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["get"]                           # list/watch 는 모든 Secret 이름이 노출되므로 제외
  resourceNames: ["app-secret"]            # 특정 Secret 만 접근 허용 (이름 제한)
```

```yaml
# secret-reader-binding.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: app-secret-reader
  namespace: production
subjects:
- kind: ServiceAccount
  name: app-sa
  namespace: production
roleRef:
  kind: Role
  name: secret-reader
  apiGroup: rbac.authorization.k8s.io
```

```bash
# Secret 접근 감사
kubectl auth can-i get secrets/app-secret -n production \
  --as=system:serviceaccount:production:app-sa
# yes

kubectl auth can-i list secrets -n production \
  --as=system:serviceaccount:production:app-sa
# no
```

---

## NetworkPolicy 로 네트워크 보안

> 자세한 NetworkPolicy 설명은 [k8s-networking-kr.md](k8s-networking-kr.md) 를 참조한다.

Kubernetes 의 기본 Pod 간 통신은 모두 허용(allow-all)이다. NetworkPolicy 로 필요한 통신만 허용하는 최소 권한 네트워크를 구성한다.

### 기본 deny-all 정책 적용

```yaml
# default-deny-all.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production                    # 이 namespace 에 적용
spec:
  podSelector: {}                          # {} = 모든 Pod 선택
  policyTypes:
  - Ingress                                # 인바운드 트래픽 규칙 정의
  - Egress                                 # 아웃바운드 트래픽 규칙 정의
  # ingress, egress 항목이 없으므로 모든 트래픽 차단
```

```bash
kubectl apply -f default-deny-all.yaml

# deny-all 적용 후 필요한 통신만 추가로 허용
```

### 특정 서비스만 허용하는 예제

```yaml
# allow-frontend-to-backend.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-to-backend
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend                         # 이 정책이 적용될 Pod (백엔드)
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend                    # 프론트엔드 Pod 에서 오는 트래픽만 허용
    ports:
    - protocol: TCP
      port: 8080                           # 백엔드 포트
```

---

## Image Security

### 신뢰할 수 있는 레지스트리만 사용

OPA Gatekeeper 또는 Kyverno 를 사용해 허용된 레지스트리의 이미지만 배포 가능하도록 정책을 강제한다.

```yaml
# kyverno-allow-registry-policy.yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: restrict-image-registries
spec:
  validationFailureAction: Enforce         # 위반 시 배포 거부
  rules:
  - name: validate-registries
    match:
      any:
      - resources:
          kinds: ["Pod"]
    validate:
      message: "허용된 레지스트리(registry.company.com)의 이미지만 사용 가능합니다."
      pattern:
        spec:
          containers:
          - image: "registry.company.com/*" # 허용 레지스트리 패턴
```

### 이미지 태그 대신 digest 사용

이미지 태그(`latest`, `1.25`)는 다른 내용으로 덮어쓸 수 있다. digest 는 불변이므로 공급망 공격을 방지한다.

```bash
# 이미지 digest 확인
docker inspect nginx:1.25 --format='{{index .RepoDigests 0}}'
# nginx@sha256:a484819eb60211f5299034ac80f6a681b06f89e65866ce91f356ed7c72af059c

# 또는 skopeo 로 레지스트리에서 직접 확인
skopeo inspect docker://nginx:1.25 | jq '.Digest'
```

```yaml
# pod-with-digest.yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-secure
spec:
  containers:
  - name: nginx
    # 태그 대신 digest 로 이미지 고정
    image: nginx@sha256:a484819eb60211f5299034ac80f6a681b06f89e65866ce91f356ed7c72af059c
    securityContext:
      runAsNonRoot: true
      runAsUser: 101                       # nginx 공식 이미지의 nginx 사용자 UID
      readOnlyRootFilesystem: false        # nginx 는 /var/cache 등에 쓰기 필요
      allowPrivilegeEscalation: false
```

### imagePullSecrets 설정

Private 레지스트리에서 이미지를 가져올 때 인증 정보를 Secret 으로 관리한다.

```bash
# Docker 레지스트리 인증 Secret 생성
kubectl create secret docker-registry registry-auth \
  --docker-server=registry.company.com \
  --docker-username=deployer \
  --docker-password=<PASSWORD> \
  --docker-email=deployer@company.com \
  --namespace=production
```

```yaml
# pod-with-pull-secret.yaml
apiVersion: v1
kind: Pod
metadata:
  name: private-image-pod
  namespace: production
spec:
  imagePullSecrets:
  - name: registry-auth                   # 위에서 만든 Secret 이름
  containers:
  - name: app
    image: registry.company.com/team/app:1.0.0
```

ServiceAccount 에 imagePullSecrets 를 등록하면 해당 SA 를 사용하는 모든 Pod 에 자동 적용된다.

```yaml
# sa-with-pull-secret.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: app-sa
  namespace: production
imagePullSecrets:
- name: registry-auth                     # 이 SA 를 쓰는 모든 Pod 에 자동 적용
```

### 이미지 취약점 스캔

```bash
# Trivy 로 이미지 취약점 스캔 (CI/CD 파이프라인에 통합 권장)
trivy image nginx:1.25

# 심각도 HIGH, CRITICAL 만 출력
trivy image --severity HIGH,CRITICAL nginx:1.25

# Kubernetes 클러스터 내 실행 중인 이미지 스캔
trivy k8s --report summary cluster
```

---

## 참고 자료

- [Kubernetes 공식 문서 - 보안](https://kubernetes.io/docs/concepts/security/)
- [Kubernetes RBAC](https://kubernetes.io/docs/reference/access-authn-authz/rbac/)
- [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
- [Security Context](https://kubernetes.io/docs/tasks/configure-pod-container/security-context/)
- [Secrets Management](https://kubernetes.io/docs/concepts/configuration/secret/)
- [Network Policies](https://kubernetes.io/docs/concepts/services-networking/network-policies/)
- [External Secrets Operator](https://external-secrets.io/)
- [Sealed Secrets](https://github.com/bitnami-labs/sealed-secrets)
- [Trivy 취약점 스캐너](https://trivy.dev/)
- [Kyverno 정책 엔진](https://kyverno.io/)
