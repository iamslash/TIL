# OIDC (Workload Identity Federation)

## 개요

OIDC (OpenID Connect)는 **credential 없이** GitHub Actions workflow가 외부
서비스에 인증하는 방법이다. GitHub가 "이 workflow는 어떤 org/repo/branch에서
실행되고 있다"는 증명서(OIDC 토큰)를 발행하고, 외부 서비스가 이를 검증한다.

## 일상 비유

```
여권으로 호텔 체크인:
  ① 정부가 여권을 발행한다
  ② 호텔에 여권을 보여준다
  ③ 호텔이 여권을 확인한다
  ④ 호텔이 방 키를 준다
  → 호텔에 미리 비밀번호를 등록할 필요 없음

OIDC로 registry 인증:
  ① GitHub가 OIDC 토큰을 발행한다
  ② Registry에 토큰을 보여준다
  ③ Registry가 GitHub에 "이 토큰 진짜냐?" 확인한다
  ④ Registry가 접근 권한을 준다
  → Registry에 미리 비밀번호를 저장할 필요 없음
```

## OIDC 토큰의 정체

JWT (JSON Web Token)이다. GitHub의 비밀키로 서명되어 위조가 불가능하다.

```json
{
  "iss": "https://token.actions.githubusercontent.com",
  "sub": "repo:MyOrg/my-app:ref:refs/heads/main",
  "aud": "https://mycompany.jfrog.io",
  "ref": "refs/heads/main",
  "repository": "MyOrg/my-app",
  "repository_owner": "MyOrg",
  "job_workflow_ref": "MyOrg/my-app/.github/workflows/build.yml@refs/heads/main",
  "exp": 1711234567,
  "iat": 1711234267
}
```

| 필드 | 의미 |
|------|------|
| `iss` | 발행자 (GitHub) |
| `sub` | 주체 (어떤 org/repo/branch) |
| `aud` | 대상 (누구에게 보여줄 것인지) |
| `exp` | 만료 시간 (수분 후) |
| `repository` | 실행 중인 repo |
| `job_workflow_ref` | 실행 중인 workflow 파일 경로 |

## 인증 흐름

```
GitHub Actions                    외부 서비스 (JFrog, AWS, GCP 등)
    │                                    │
    │ ① OIDC 토큰 자동 발행               │
    │    sub: "repo:MyOrg/my-app:        │
    │          ref:refs/heads/main"      │
    │    서명: GitHub 비밀키               │
    │                                    │
    │ ② "이 토큰으로 인증할게"             │
    │ ──────────────────────────────────▶ │
    │                                    │
    │                                    │ ③ GitHub 공개키로 서명 검증
    │                                    │   → sub 확인 (허용된 repo인가)
    │                                    │   → exp 확인 (만료되지 않았나)
    │                                    │
    │                ④ 단기 access token  │
    │ ◀────────────────────────────────── │
    │   (수분 유효)                        │
    │                                    │
    │ ⑤ access token으로 작업 수행         │
    │ ──────────────────────────────────▶ │
    │                                    │
    │ ⑥ 수분 후 access token 자동 만료     │
```

핵심: **credential이 어디에도 저장되지 않는다.**

## JFrog Artifactory + OIDC 설정

### Step 1: JFrog에서 OIDC Provider 등록

JFrog Platform > Administration > Security > OpenID Connect

```
Provider Name:  github-actions
Provider URL:   https://token.actions.githubusercontent.com
Audience:       jfrog
```

### Step 2: JFrog에서 Identity Mapping 설정

어떤 repo가 어떤 권한을 갖는지 매핑한다.

```json
{
  "name": "my-app-ci",
  "priority": 1,
  "claims": {
    "sub": "repo:MyOrg/my-app:ref:refs/heads/main"
  },
  "token_spec": {
    "scope": "applied-permissions/groups:ci-deployers",
    "expires_in": 300
  }
}
```

`MyOrg/my-app`의 `main` branch에서 실행된 workflow만 `ci-deployers` 권한을
받는다. 다른 repo나 다른 branch는 차단된다.

### Step 3: GitHub Actions Workflow

```yaml
name: Build and Push

on:
  push:
    branches: [main]

permissions:
  id-token: write    # OIDC 토큰 발행 허용
  contents: read

jobs:
  build:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4

      - name: Login to JFrog (OIDC)
        uses: jfrog/setup-jfrog-cli@v4
        env:
          JF_URL: https://mycompany.jfrog.io
        with:
          oidc-provider-name: github-actions
          oidc-audience: jfrog

      - name: Build and push
        run: |
          jf docker build -t mycompany.jfrog.io/docker/my-app:${{ github.sha }} .
          jf docker push mycompany.jfrog.io/docker/my-app:${{ github.sha }}
```

**`secrets.XXX`가 없다.** credential이 존재하지 않는다.

## AWS ECR + OIDC 설정

### Step 1: AWS에서 OIDC Provider 등록

```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

### Step 2: IAM Role 생성

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
          "token.actions.githubusercontent.com:sub": "repo:MyOrg/my-app:ref:refs/heads/main"
        }
      }
    }
  ]
}
```

### Step 3: Workflow

```yaml
permissions:
  id-token: write
  contents: read

jobs:
  build:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/ci-deployer
          aws-region: ap-northeast-2

      - uses: aws-actions/amazon-ecr-login@v2

      - run: |
          docker build -t 123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:${{ github.sha }} .
          docker push 123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app:${{ github.sha }}
```

## GCP Artifact Registry + OIDC 설정

### Step 1: Workload Identity Federation 설정

```bash
# Workload Identity Pool 생성
gcloud iam workload-identity-pools create github-pool \
  --location="global"

# Provider 생성
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository"

# Service Account에 권한 부여
gcloud iam service-accounts add-iam-policy-binding \
  ci-pusher@my-project.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/123/locations/global/workloadIdentityPools/github-pool/attribute.repository/MyOrg/my-app"
```

### Step 2: Workflow

```yaml
permissions:
  id-token: write
  contents: read

jobs:
  build:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4

      - uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: "projects/123/locations/global/workloadIdentityPools/github-pool/providers/github-provider"
          service_account: "ci-pusher@my-project.iam.gserviceaccount.com"

      - run: |
          gcloud auth configure-docker us-docker.pkg.dev
          docker build -t us-docker.pkg.dev/my-project/repo/my-app:${{ github.sha }} .
          docker push us-docker.pkg.dev/my-project/repo/my-app:${{ github.sha }}
```

## Identity Mapping 보안

OIDC의 보안은 **Identity Mapping (claim 조건)** 설정에 달려있다.

```
안전한 설정:
  sub: "repo:MyOrg/my-app:ref:refs/heads/main"
  → 특정 org, 특정 repo, 특정 branch만 허용

위험한 설정:
  sub: "repo:MyOrg/*"
  → org 내 모든 repo가 접근 가능

매우 위험한 설정:
  sub: "*"
  → 모든 GitHub repo가 접근 가능
```

가능한 한 repo + branch 단위로 제한해야 한다.

## OIDC의 남는 위험

| 위험 | 설명 |
|------|------|
| 단기 access token 메모리 탈취 | Step ④~⑤ 사이에 수분짜리 토큰이 메모리에 존재 |
| GitHub 자체 침해 | OIDC 서명 키 유출 시 토큰 위조 가능 (확률 극히 낮음) |
| Identity Mapping 설정 오류 | `sub` claim을 넓게 설정하면 의도하지 않은 접근 허용 |
| workflow 파일 변조 | main branch의 workflow 파일을 수정하면 권한 획득 가능 → branch protection 필수 |

그래도 **수분 후 만료되는 토큰**과 **수년간 유효한 정적 토큰**의 위험 차이는
비교할 수 없을 만큼 크다.

## OIDC 지원 서비스

| 서비스 | 지원 여부 |
|--------|----------|
| AWS (ECR, S3 등) | O |
| GCP (Artifact Registry 등) | O |
| Azure (ACR 등) | O |
| JFrog Artifactory | O |
| HashiCorp Vault | O (JWT auth) |
| Docker Hub | O (limited) |
| Harbor | X (직접 구현 필요) |
| Nexus | X |

OIDC를 지원하지 않는 서비스는 Vault를 중간에 두고 Vault JWT auth + Dynamic
Secret으로 유사한 효과를 얻을 수 있다.
