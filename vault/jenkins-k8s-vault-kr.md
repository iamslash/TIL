# Jenkins CI/CD + K8s + Vault 구성

## 개요

Jenkins로 빌드하고 K8s에 배포하는 파이프라인에서 Vault를 사용하여 secret을
안전하게 관리하는 구성이다. 핵심 원칙은 **어떤 단계에서도 secret이 평문으로
노출되지 않는 것**이다.

## 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        Jenkins CI/CD                             │
│                                                                  │
│  ① Build Image  ② Push to Registry  ③ kubectl apply             │
│     (이미지에 secret 없음)            (manifest에 secret 없음)   │
└──────────────────────────────┬──────────────────────────────────┘
                               │ kubeconfig (제한된 권한)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                            │
│                                                                  │
│  ┌─────────────────────────────────────────────┐                │
│  │  Namespace: my-app-prod                      │                │
│  │                                              │                │
│  │  ┌──────────────┐     ┌───────────────────┐  │                │
│  │  │ App Pod      │     │ Vault Agent       │  │                │
│  │  │              │◀────│ (Sidecar)         │  │                │
│  │  │ /vault/      │ 파일 │                   │  │                │
│  │  │ secrets/     │ 마운트│ SA Token으로      │  │                │
│  │  │ config       │     │ Vault 인증        │  │                │
│  │  └──────────────┘     └────────┬──────────┘  │                │
│  │                                │              │                │
│  │  ServiceAccount: my-app-sa     │              │                │
│  └────────────────────────────────┼──────────────┘                │
│                                   │                               │
└───────────────────────────────────┼───────────────────────────────┘
                                    │ K8s Auth
                                    ▼
                            ┌───────────────┐
                            │    Vault       │
                            │                │
                            │ Policy:        │
                            │ my-app-prod    │
                            │ → secret/      │
                            │   my-app/prod  │
                            └───────────────┘
```

## Step 1: Vault 설정

### Secret 저장

```bash
vault kv put secret/my-app/prod \
  DB_HOST="db.internal:5432" \
  DB_USER="app_user" \
  DB_PASSWORD="s3cret_p@ssw0rd" \
  API_KEY="abc-123-def-456"
```

### Policy 생성

```hcl
# my-app-prod.hcl
path "secret/data/my-app/prod" {
  capabilities = ["read"]
}
```

```bash
vault policy write my-app-prod my-app-prod.hcl
```

### K8s Auth Role 생성

```bash
vault auth enable kubernetes

vault write auth/kubernetes/config \
  kubernetes_host="https://k8s-api.internal:6443" \
  kubernetes_ca_cert=@/path/to/k8s-ca.crt

vault write auth/kubernetes/role/my-app-prod \
  bound_service_account_names=my-app-sa \
  bound_service_account_namespaces=my-app-prod \
  policies=my-app-prod \
  ttl=1h \
  max_ttl=4h
```

## Step 2: Kubernetes 리소스

### Namespace + ServiceAccount

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: my-app-prod
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: my-app-sa
  namespace: my-app-prod
```

### App Pod의 K8s RBAC (최소 권한)

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: my-app-role
  namespace: my-app-prod
rules: []  # K8s API 접근 권한 없음
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: my-app-binding
  namespace: my-app-prod
subjects:
  - kind: ServiceAccount
    name: my-app-sa
roleRef:
  kind: Role
  name: my-app-role
  apiGroup: rbac.authorization.k8s.io
```

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  namespace: my-app-prod
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
        vault.hashicorp.com/agent-inject: "true"
        vault.hashicorp.com/role: "my-app-prod"
        vault.hashicorp.com/agent-inject-secret-config: "secret/data/my-app/prod"
        vault.hashicorp.com/agent-inject-template-config: |
          {{- with secret "secret/data/my-app/prod" -}}
          export DB_HOST="{{ .Data.data.DB_HOST }}"
          export DB_USER="{{ .Data.data.DB_USER }}"
          export DB_PASSWORD="{{ .Data.data.DB_PASSWORD }}"
          export API_KEY="{{ .Data.data.API_KEY }}"
          {{- end }}
        vault.hashicorp.com/agent-revoke-on-shutdown: "true"
    spec:
      serviceAccountName: my-app-sa
      containers:
        - name: my-app
          image: registry.internal/my-app:latest  # Jenkins가 태그 업데이트
          command: ["/bin/sh", "-c"]
          args:
            - source /vault/secrets/config && ./app
          ports:
            - containerPort: 3000
          securityContext:
            runAsNonRoot: true
            runAsUser: 1000
            readOnlyRootFilesystem: true
            allowPrivilegeEscalation: false
            capabilities:
              drop: ["ALL"]
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "256Mi"
              cpu: "200m"
          livenessProbe:
            httpGet:
              path: /health
              port: 3000
            periodSeconds: 30
```

### Network Policy (선택, 권장)

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: my-app-netpol
  namespace: my-app-prod
spec:
  podSelector:
    matchLabels:
      app: my-app
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - port: 3000
  egress:
    # Vault 접근 허용
    - to:
        - ipBlock:
            cidr: 10.0.0.10/32  # Vault IP
      ports:
        - port: 8200
    # 외부 API 접근 허용 (필요한 경우만)
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - port: 443
    # DNS
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - port: 53
          protocol: UDP
```

## Step 3: Jenkins Pipeline

### Jenkins의 K8s 접근 권한 (최소화)

```yaml
# Jenkins가 사용하는 K8s ServiceAccount
apiVersion: v1
kind: ServiceAccount
metadata:
  name: jenkins-deployer
  namespace: my-app-prod
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: jenkins-deployer-role
  namespace: my-app-prod
rules:
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["get", "patch"]        # image tag 변경만 가능
    resourceNames: ["my-app"]      # 이 deployment만
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: jenkins-deployer-binding
  namespace: my-app-prod
subjects:
  - kind: ServiceAccount
    name: jenkins-deployer
roleRef:
  kind: Role
  name: jenkins-deployer-role
  apiGroup: rbac.authorization.k8s.io
```

Jenkins는 **deployment의 image tag를 바꾸는 것만** 할 수 있다. Secret 읽기,
Pod exec, 다른 리소스 접근은 불가능하다.

### Jenkinsfile

```groovy
pipeline {
    agent any

    environment {
        REGISTRY = 'registry.internal'
        IMAGE    = 'my-app'
        K8S_NS   = 'my-app-prod'
    }

    stages {
        stage('Test') {
            steps {
                sh 'npm test'
            }
        }

        stage('Build') {
            steps {
                sh "docker build --no-cache -t ${REGISTRY}/${IMAGE}:${BUILD_NUMBER} ."
            }
        }

        stage('Image Scan') {
            steps {
                // 이미지에 secret이나 취약점이 없는지 스캔
                sh "trivy image --severity HIGH,CRITICAL --exit-code 1 ${REGISTRY}/${IMAGE}:${BUILD_NUMBER}"
            }
        }

        stage('Push') {
            steps {
                sh "docker push ${REGISTRY}/${IMAGE}:${BUILD_NUMBER}"
            }
        }

        stage('Deploy') {
            steps {
                // Jenkins는 secret을 모른다. image tag만 업데이트한다.
                sh """
                  kubectl -n ${K8S_NS} set image \
                    deployment/my-app \
                    my-app=${REGISTRY}/${IMAGE}:${BUILD_NUMBER}
                """
            }
        }

        stage('Verify') {
            steps {
                sh """
                  kubectl -n ${K8S_NS} rollout status \
                    deployment/my-app --timeout=120s
                """
            }
        }
    }

    post {
        failure {
            sh "kubectl -n ${K8S_NS} rollout undo deployment/my-app"
        }
    }
}
```

### Dockerfile (Secret이 들어가지 않도록)

```dockerfile
FROM node:20-slim

WORKDIR /app

# 의존성 설치 (캐시 레이어)
COPY package*.json ./
RUN npm ci --production

# 소스 복사
COPY . .

# .env, credential 파일이 포함되지 않도록
# .dockerignore에 명시
EXPOSE 3000
USER 1000

CMD ["node", "app.js"]
```

```gitignore
# .dockerignore
.env
.env.*
*.pem
*.key
credentials*
vault-token
```

## 각 단계에서 차단되는 공격

```
공격 시나리오                              차단 지점
──────────────────────────────────────────────────────
① Git repo에서 secret 탈취               → repo에 secret 없음
② Docker image에서 secret 추출           → image에 secret 없음 + trivy 스캔
③ Jenkins에서 secret 탈취                → Jenkins는 secret을 모름
④ K8s manifest에서 secret 탈취           → manifest에 secret 없음
⑤ 다른 namespace Pod에서 Vault 접근      → SA + namespace 바인딩으로 차단
⑥ 같은 namespace 다른 Pod에서 접근       → SA 이름 바인딩으로 차단
⑦ App Pod에서 다른 secret 경로 접근      → Vault policy로 경로 제한
⑧ kubectl exec로 환경변수 확인           → Jenkins RBAC에 exec 없음
⑨ 토큰 탈취 후 장기간 사용               → TTL 1시간, 자동 만료
⑩ 컨테이너 탈출 시도                     → securityContext 강화
⑪ Pod 종료 후 토큰 재사용                → agent-revoke-on-shutdown
⑫ 네트워크 스니핑                        → Network Policy로 통신 제한
```

## Secret 로테이션

### 수동 로테이션

```bash
# Vault에서 secret 업데이트
vault kv put secret/my-app/prod \
  DB_PASSWORD="new_p@ssw0rd" \
  ...

# Vault Agent가 자동으로 새 값을 파일에 반영
# App이 재시작 없이 새 값을 읽으려면 파일 감시 로직 필요
```

### 자동 로테이션 (Dynamic Secret)

Vault의 Database Secret Engine을 사용하면 DB credential이 자동 생성/폐기된다.

```bash
# App이 요청할 때마다 새 DB 계정 발급
vault read database/creds/my-app-role
# username: v-k8s-my-app-abc123
# password: A1b2C3d4...
# lease_duration: 1h
# → 1시간 후 자동 삭제
```

## 환경 분리

```
Vault:
  secret/my-app/dev     → Policy: my-app-dev
  secret/my-app/staging → Policy: my-app-staging
  secret/my-app/prod    → Policy: my-app-prod

K8s:
  ns: my-app-dev     → SA: my-app-sa → Vault Role: my-app-dev
  ns: my-app-staging → SA: my-app-sa → Vault Role: my-app-staging
  ns: my-app-prod    → SA: my-app-sa → Vault Role: my-app-prod
```

각 namespace의 SA는 자기 환경의 secret만 읽을 수 있다. dev 환경의 Pod가 prod
secret에 접근하는 것은 불가능하다.
