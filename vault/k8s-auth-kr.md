# Kubernetes Auth 연동

## 개요

Vault의 Kubernetes Auth Method는 K8s Pod가 별도 credential 없이 Vault에
인증하는 방법이다. K8s ServiceAccount Token을 사용하며, Vault가 K8s API Server에
토큰 유효성을 확인한다.

## Secret 주입 방식

K8s Pod에 Vault secret을 전달하는 세 가지 방식이 있다.

### 1. Vault Agent Sidecar Injector (권장)

Vault Agent가 sidecar 컨테이너로 Pod에 주입되어 secret을 파일로 마운트한다.

```
┌─────────────────────────────────────┐
│  Pod                                 │
│                                      │
│  ┌──────────┐   ┌────────────────┐  │
│  │ App      │   │ Vault Agent    │  │
│  │ Container│   │ (Sidecar)      │  │
│  │          │   │                │  │
│  │ /vault/  │◀──│ ① SA Token으로  │  │
│  │ secrets/ │파일│   Vault 로그인  │  │
│  │ config   │   │ ② secret 가져옴│  │
│  │          │   │ ③ 파일로 기록   │  │
│  │          │   │ ④ TTL 만료 시   │  │
│  │          │   │   자동 갱신     │  │
│  └──────────┘   └────────────────┘  │
│                                      │
│  ServiceAccount: my-app-sa           │
└─────────────────────────────────────┘
```

#### 설치 (Vault Agent Injector)

```bash
helm repo add hashicorp https://helm.releases.hashicorp.com
helm install vault hashicorp/vault \
  --set "injector.enabled=true" \
  --set "server.enabled=false"   # 외부 Vault 사용 시
```

#### Deployment 설정

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  namespace: prod
spec:
  replicas: 2
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
      annotations:
        # Vault Agent Injector 어노테이션
        vault.hashicorp.com/agent-inject: "true"
        vault.hashicorp.com/role: "my-app-prod"

        # secret을 파일로 주입
        vault.hashicorp.com/agent-inject-secret-db: "secret/data/my-app/prod"

        # 파일 내용의 포맷 지정 (Go template)
        vault.hashicorp.com/agent-inject-template-db: |
          {{- with secret "secret/data/my-app/prod" -}}
          DB_HOST={{ .Data.data.DB_HOST }}
          DB_USER={{ .Data.data.DB_USER }}
          DB_PASSWORD={{ .Data.data.DB_PASSWORD }}
          {{- end }}
    spec:
      serviceAccountName: my-app-sa
      containers:
        - name: my-app
          image: my-registry/my-app:v1.0.0
          command: ["/bin/sh", "-c"]
          args:
            - source /vault/secrets/db && ./app
```

주입된 파일 경로: `/vault/secrets/db`

#### 어노테이션 옵션

| 어노테이션 | 설명 |
|-----------|------|
| `agent-inject` | Vault Agent 주입 활성화 |
| `role` | Vault K8s auth role |
| `agent-inject-secret-{name}` | secret 경로 (파일명: `/vault/secrets/{name}`) |
| `agent-inject-template-{name}` | 파일 내용 템플릿 (Go template) |
| `agent-inject-status` | `update`로 설정하면 재배포 없이 갱신 |
| `agent-pre-populate-only` | `true`면 init container만 (sidecar 없음) |
| `agent-revoke-on-shutdown` | `true`면 Pod 종료 시 토큰 폐기 |

### 2. Vault CSI Provider

Kubernetes CSI (Container Storage Interface) 드라이버로 secret을 마운트한다.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app
spec:
  serviceAccountName: my-app-sa
  containers:
    - name: my-app
      image: my-registry/my-app:v1.0.0
      volumeMounts:
        - name: vault-secrets
          mountPath: "/vault/secrets"
          readOnly: true
  volumes:
    - name: vault-secrets
      csi:
        driver: secrets-store.csi.k8s.io
        readOnly: true
        volumeAttributes:
          secretProviderClass: "my-app-vault"
---
apiVersion: secrets-store.csi.x-k8s.io/v1
kind: SecretProviderClass
metadata:
  name: my-app-vault
spec:
  provider: vault
  parameters:
    roleName: "my-app-prod"
    vaultAddress: "https://vault.internal:8200"
    objects: |
      - objectName: "db-password"
        secretPath: "secret/data/my-app/prod"
        secretKey: "DB_PASSWORD"
```

### 3. App이 직접 Vault API 호출

App 코드에서 직접 Vault에 인증하고 secret을 가져온다.

#### Python 예시

```python
import hvac
import os

# K8s ServiceAccount Token 읽기
with open("/var/run/secrets/kubernetes.io/serviceaccount/token") as f:
    sa_token = f.read()

# Vault 클라이언트 생성
client = hvac.Client(url="https://vault.internal:8200")

# K8s Auth로 로그인
client.auth.kubernetes.login(
    role="my-app-prod",
    jwt=sa_token
)

# Secret 읽기
secret = client.secrets.kv.v2.read_secret_version(
    path="my-app/prod"
)

db_password = secret["data"]["data"]["DB_PASSWORD"]
```

#### Node.js 예시

```javascript
const vault = require("node-vault")({
  endpoint: "https://vault.internal:8200",
});

const fs = require("fs");

async function getSecrets() {
  // K8s SA Token 읽기
  const jwt = fs.readFileSync(
    "/var/run/secrets/kubernetes.io/serviceaccount/token",
    "utf8"
  );

  // K8s Auth 로그인
  await vault.kubernetesLogin({
    role: "my-app-prod",
    jwt: jwt,
  });

  // Secret 읽기
  const result = await vault.read("secret/data/my-app/prod");
  return result.data.data;
}
```

### 방식 비교

| 항목 | Agent Sidecar | CSI Provider | 직접 API 호출 |
|------|--------------|-------------|-------------|
| 코드 변경 | 불필요 | 불필요 | 필요 |
| 자동 갱신 | O (Agent가 처리) | 제한적 | 직접 구현 |
| 리소스 오버헤드 | sidecar 메모리 | CSI 드라이버 | 없음 |
| 복잡도 | 어노테이션만 추가 | CRD 설정 | Vault SDK 의존 |
| 권장 환경 | 범용 | Secret을 K8s Secret으로 동기화 필요 시 | 세밀한 제어 필요 시 |

## 템플릿 패턴

### 환경변수 파일

```
vault.hashicorp.com/agent-inject-template-env: |
  {{- with secret "secret/data/my-app/prod" -}}
  export DB_HOST="{{ .Data.data.DB_HOST }}"
  export DB_PORT="{{ .Data.data.DB_PORT }}"
  export DB_PASSWORD="{{ .Data.data.DB_PASSWORD }}"
  {{- end }}
```

### JSON 설정 파일

```
vault.hashicorp.com/agent-inject-template-config: |
  {{- with secret "secret/data/my-app/prod" -}}
  {
    "database": {
      "host": "{{ .Data.data.DB_HOST }}",
      "password": "{{ .Data.data.DB_PASSWORD }}"
    }
  }
  {{- end }}
```

### 여러 secret 경로 합치기

```
vault.hashicorp.com/agent-inject-secret-db: "secret/data/my-app/prod/db"
vault.hashicorp.com/agent-inject-secret-api: "secret/data/my-app/prod/api"
```

→ `/vault/secrets/db` 와 `/vault/secrets/api` 두 파일이 생성된다.

## ServiceAccount Token Projection

K8s 1.21+ 에서는 bound ServiceAccount Token을 사용한다. 기존 SA Token과 달리
audience, TTL이 제한된다.

```yaml
spec:
  containers:
    - name: my-app
      volumeMounts:
        - name: vault-token
          mountPath: /var/run/secrets/vault
  volumes:
    - name: vault-token
      projected:
        sources:
          - serviceAccountToken:
              audience: vault
              expirationSeconds: 3600
              path: token
```

이 토큰은 `vault` audience에만 유효하고 1시간 후 만료된다. K8s API Server에
직접 사용할 수 없으므로 탈취 시 피해가 제한된다.

## 트러블슈팅

### 흔한 에러

| 증상 | 원인 | 해결 |
|------|------|------|
| `permission denied` | Policy가 secret 경로를 허용하지 않음 | `vault policy read` 로 경로 확인 |
| `service account not authorized` | SA 이름 또는 namespace 불일치 | Vault role의 바인딩 확인 |
| `token expired` | TTL 만료 | Agent sidecar가 자동 갱신하는지 확인 |
| Pod 시작 안 됨 (init 대기) | Vault Agent Injector가 Vault에 연결 불가 | Vault 주소, 네트워크 확인 |

### 디버깅

```bash
# Vault Agent 로그 확인
kubectl logs my-app-pod -c vault-agent-init
kubectl logs my-app-pod -c vault-agent

# SA Token 확인
kubectl exec my-app-pod -c my-app -- \
  cat /var/run/secrets/kubernetes.io/serviceaccount/token

# Vault에서 직접 검증
vault write auth/kubernetes/login \
  role=my-app-prod \
  jwt=$(cat sa-token)
```
