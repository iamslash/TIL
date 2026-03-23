# Vault 인증 방법 (Auth Methods)

## 개요

Vault에 접근하려면 먼저 인증(Authentication)을 거쳐야 한다. 인증에 성공하면
Vault Token이 발급되고, 이 토큰으로 secret에 접근한다.

**핵심 원칙**: 사람이 관리하는 credential을 줄일수록 안전하다.

## 인증 방법 비교

```
                    보안 수준
Token (하드코딩)     █░░░░░░░░░  토큰이 코드/설정에 평문 존재
환경변수             ██░░░░░░░░  ps, /proc 등으로 노출 가능
UserPass            ███░░░░░░░  ID/PW 관리 필요
AppRole             ███████░░░  두 조각 분리 + 일회용 Secret ID
K8s Auth            █████████░  관리할 credential 자체가 없음
AWS/GCP IAM Auth    █████████░  클라우드 플랫폼 identity 활용
```

## AppRole

credential을 **두 조각으로 분리**하여 서로 다른 경로로 전달하는 방식이다.
CI/CD 파이프라인이나 VM 기반 서버에서 주로 사용한다.

### 구조

```
┌─────────────┐                          ┌───────────┐
│  CI/CD      │ ── Role ID (고정, 덜 민감) ──▶│           │
│  (배포 시)   │                          │           │
└─────────────┘                          │   App     │──▶ Vault 로그인
┌─────────────┐                          │ (런타임)  │    (Role ID + Secret ID)
│  Vault      │ ── Secret ID (일회용) ────▶│           │
│  (API 호출)  │                          └───────────┘
└─────────────┘
```

| 조각 | 특성 | 전달 방식 |
|------|------|----------|
| **Role ID** | 고정값, 단독으로는 쓸모없음 | CI/CD 파이프라인, 환경변수 |
| **Secret ID** | 일회용, TTL 있음, 사용 횟수 제한 | Vault API로 배포 시점에 발급 |

### 설정

```bash
# AppRole auth 활성화
vault auth enable approle

# Role 생성
vault write auth/approle/role/my-app \
  token_policies="my-app-prod" \
  token_ttl=1h \
  token_max_ttl=4h \
  secret_id_ttl=10m \
  secret_id_num_uses=1   # 일회용

# Role ID 조회 (고정값)
vault read auth/approle/role/my-app/role-id
# role_id: db02de05-fa39-4855-059b-67221c5c2f63

# Secret ID 발급 (일회용)
vault write -f auth/approle/role/my-app/secret-id
# secret_id: 6a174c20-f6de-a53c-74d2-6018fcceff64
# secret_id_ttl: 10m
# secret_id_num_uses: 1
```

### 로그인

```bash
vault write auth/approle/login \
  role_id="db02de05-fa39-4855-059b-67221c5c2f63" \
  secret_id="6a174c20-f6de-a53c-74d2-6018fcceff64"
# token: hvs.CAESI...
```

### 보안 분석

두 조각이 **동시에** 탈취되어야만 위험하다. Secret ID는 한 번 쓰면 소멸하므로
탈취 창(window)이 극히 짧다.

| 공격 시나리오 | 결과 |
|-------------|------|
| Role ID만 탈취 | Secret ID 없이 로그인 불가 |
| Secret ID만 탈취 | Role ID 없이 로그인 불가 |
| 둘 다 탈취했지만 이미 사용됨 | Secret ID 소진, 로그인 불가 |
| 둘 다 탈취, 아직 미사용 | 위험 (10분 이내 공격 필요) |

## Kubernetes Auth

credential 자체가 필요 없다. K8s가 이미 증명한 신원(identity)을 Vault가
신뢰하는 방식이다.

### 구조

```
┌──────────────────────────────────────────────────┐
│  Kubernetes Cluster                               │
│                                                   │
│  ┌───────────┐    ① ServiceAccount Token          │
│  │  App Pod  │ ──(자동 마운트, K8s가 발급)──────┐  │
│  └───────────┘                                 │  │
│                                                │  │
└────────────────────────────────────────────────┘  │
                                                    │
       ② "나는 ns=prod, sa=my-app 이다"             │
                        │                           │
                        ▼                           │
                 ┌───────────┐                      │
                 │  Vault    │◀─────────────────────┘
                 │           │
                 │           │  ③ K8s API에 토큰 검증 요청
                 │           │──▶ K8s API Server
                 │           │◀── "유효하다"
                 │           │
                 │           │  ④ policy에 따라 secret 반환
                 └───────────┘
```

1. **Pod 생성 시** K8s가 ServiceAccount Token을 자동 마운트한다
2. **App**이 이 토큰으로 Vault에 로그인 요청한다
3. **Vault**가 K8s API Server에 토큰 유효성을 확인한다
4. **검증 통과 시** 해당 ServiceAccount에 매핑된 policy로 secret을 반환한다

### 설정

```bash
# K8s auth 활성화
vault auth enable kubernetes

# K8s API 연결 설정
vault write auth/kubernetes/config \
  kubernetes_host="https://k8s-api.internal:6443" \
  kubernetes_ca_cert=@/path/to/k8s-ca.crt

# Role 생성 (어떤 SA가 어떤 policy를 받는지)
vault write auth/kubernetes/role/my-app-prod \
  bound_service_account_names=my-app-sa \
  bound_service_account_namespaces=prod \
  policies=my-app-prod \
  ttl=1h \
  max_ttl=4h
```

### 바인딩 제약

| 파라미터 | 설명 |
|---------|------|
| `bound_service_account_names` | 허용할 SA 이름 |
| `bound_service_account_namespaces` | 허용할 namespace |
| `policies` | 부여할 Vault policy |
| `ttl` | 발급되는 토큰의 수명 |

다른 namespace의 다른 SA로는 이 role에 로그인할 수 없다.

### 보안 분석

| 공격 시나리오 | 결과 |
|-------------|------|
| 다른 Pod에서 접근 시도 | SA가 다르므로 role 매칭 실패 |
| 다른 namespace에서 같은 SA 이름 | namespace 바인딩으로 차단 |
| SA Token 탈취 | K8s 토큰 projection 사용 시 TTL 제한 |
| Pod 삭제 시 | SA Token 자동 무효화 |

## AWS IAM Auth

AWS 환경에서 EC2 Instance Profile 또는 IAM Role을 Vault 인증에 사용한다.

```bash
vault auth enable aws

vault write auth/aws/role/my-app \
  auth_type=iam \
  bound_iam_principal_arn="arn:aws:iam::123456789012:role/my-app-role" \
  policies=my-app-prod \
  ttl=1h
```

EC2/Lambda에서 별도 credential 없이 자신의 IAM Role로 Vault에 인증한다.

## GCP Auth

GCP 환경에서 Service Account를 Vault 인증에 사용한다.

```bash
vault auth enable gcp

vault write auth/gcp/role/my-app \
  type=iam \
  bound_service_accounts="my-app@project-id.iam.gserviceaccount.com" \
  policies=my-app-prod \
  ttl=1h
```

## 환경별 권장 Auth Method

| 환경 | 권장 Auth Method | 이유 |
|------|-----------------|------|
| Kubernetes | `kubernetes` | 관리할 credential 없음 |
| AWS EC2/Lambda | `aws` | IAM Role 활용 |
| GCP GCE/GKE | `gcp` | Service Account 활용 |
| VM (on-prem) | `approle` | 플랫폼 identity 없으므로 |
| CI/CD (Jenkins) | `approle` | 파이프라인에서 Secret ID 발급 |
| 관리자 | `userpass` + MFA | 사람이 직접 로그인 |

핵심은 **플랫폼이 제공하는 identity를 최대한 활용**하는 것이다. 플랫폼 identity가
없는 환경에서만 AppRole을 사용한다.
