# 실전 예제: GitHub Actions + ArgoCD

## 시나리오

간단한 Node.js API 서버를 GitHub Actions로 빌드하고 ArgoCD로 K8s에 배포한다.

## 레포 구조

두 개의 repo를 사용한다 (소스와 manifest 분리).

```
myorg/my-app          (소스 코드 + CI workflow)
├── src/
│   └── index.js
├── Dockerfile
├── package.json
└── .github/
    └── workflows/
        └── ci.yml

myorg/my-app-manifests  (K8s manifest + ArgoCD가 감시)
├── base/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── kustomization.yaml
└── overlays/
    ├── dev/
    │   └── kustomization.yaml
    └── prod/
        └── kustomization.yaml
```

## 1. 소스 코드

### src/index.js

```javascript
const express = require("express");
const app = express();

app.get("/health", (req, res) => res.json({ status: "ok" }));
app.get("/", (req, res) => res.json({ message: "Hello World" }));

app.listen(3000, () => console.log("Listening on :3000"));
```

### Dockerfile

```dockerfile
FROM node:20-slim
WORKDIR /app
COPY package*.json ./
RUN npm ci --production
COPY src/ src/
EXPOSE 3000
USER 1000
CMD ["node", "src/index.js"]
```

## 2. GitHub Actions (CI)

### .github/workflows/ci.yml

```yaml
name: CI

on:
  push:
    branches: [main]

permissions:
  id-token: write      # OIDC
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci
      - run: npm test

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    outputs:
      image-tag: ${{ github.sha }}
    steps:
      - uses: actions/checkout@v4

      # OIDC로 AWS 인증 (credential 없음)
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/ci-deployer
          aws-region: ap-northeast-2

      - uses: aws-actions/amazon-ecr-login@v2

      - name: Build and push image
        env:
          REGISTRY: 123456789012.dkr.ecr.ap-northeast-2.amazonaws.com
          IMAGE: my-app
          TAG: ${{ github.sha }}
        run: |
          docker build -t $REGISTRY/$IMAGE:$TAG .
          docker push $REGISTRY/$IMAGE:$TAG

  update-manifest:
    needs: build-and-push
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
        with:
          repository: myorg/my-app-manifests
          token: ${{ secrets.MANIFEST_REPO_TOKEN }}

      - name: Update image tag
        env:
          NEW_TAG: ${{ needs.build-and-push.outputs.image-tag }}
          REGISTRY: 123456789012.dkr.ecr.ap-northeast-2.amazonaws.com
        run: |
          cd overlays/prod
          kustomize edit set image $REGISTRY/my-app=$REGISTRY/my-app:$NEW_TAG
          git config user.name "github-actions"
          git config user.email "actions@github.com"
          git add .
          git commit -m "Deploy my-app:$NEW_TAG"
          git push
```

## 3. K8s Manifest

### base/deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
        - name: my-app
          image: 123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:latest
          ports:
            - containerPort: 3000
          livenessProbe:
            httpGet:
              path: /health
              port: 3000
            periodSeconds: 30
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "256Mi"
              cpu: "200m"
```

### base/service.yaml

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-app
spec:
  selector:
    app: my-app
  ports:
    - port: 80
      targetPort: 3000
```

### base/kustomization.yaml

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
  - service.yaml
```

### overlays/prod/kustomization.yaml

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
namespace: my-app-prod
replicas:
  - name: my-app
    count: 3
images:
  - name: 123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app
    newTag: latest    # CI가 이 값을 업데이트
```

### overlays/dev/kustomization.yaml

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
namespace: my-app-dev
replicas:
  - name: my-app
    count: 1
```

## 4. ArgoCD Application

### argocd-app-prod.yaml

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: my-app-prod
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/myorg/my-app-manifests
    targetRevision: main
    path: overlays/prod
  destination:
    server: https://kubernetes.default.svc
    namespace: my-app-prod
  syncPolicy:
    syncOptions:
      - CreateNamespace=true
    # prod는 Manual Sync (자동 배포 안 함)
```

### argocd-app-dev.yaml

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: my-app-dev
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/myorg/my-app-manifests
    targetRevision: main
    path: overlays/dev
  destination:
    server: https://kubernetes.default.svc
    namespace: my-app-dev
  syncPolicy:
    automated:              # dev는 Auto Sync
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

```bash
# ArgoCD에 Application 등록
kubectl apply -f argocd-app-prod.yaml
kubectl apply -f argocd-app-dev.yaml
```

## 5. 전체 배포 흐름

```
① 개발자: git push (main branch)
     │
     ▼
② GitHub Actions (CI):
     ├── npm test (테스트)
     ├── docker build & push (OIDC로 ECR 인증)
     └── manifest repo에 image tag commit
     │
     ▼
③ ArgoCD (CD):
     ├── Git 변경 감지 (3분 간격 poll)
     ├── dev: Auto Sync → 자동 배포
     └── prod: OutOfSync 표시 → 대기
     │
     ▼
④ 운영자: ArgoCD UI에서 prod Sync 클릭
     │
     ▼
⑤ ArgoCD: prod 배포 실행
     ├── kustomize build
     └── kubectl apply
```

## credential 정리

| 구간 | credential | 방식 |
|------|-----------|------|
| GitHub Actions → ECR | 없음 | OIDC |
| GitHub Actions → manifest repo | `MANIFEST_REPO_TOKEN` | GitHub Secret |
| ArgoCD → Git repo | Deploy Key (SSH) | ArgoCD Secret |
| ArgoCD → K8s | 없음 | 클러스터 내부 |
| K8s → ECR | IAM Role for SA | IRSA |
