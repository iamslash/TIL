# Vault 개요 및 기본 개념

## Vault 란?

HashiCorp Vault는 secret(토큰, 비밀번호, 인증서, API 키 등)을 안전하게 저장하고
접근을 제어하는 도구이다. 단순한 Key-Value 저장소가 아니라 동적 secret 생성,
암호화, 접근 감사 등을 제공한다.

## 핵심 개념

### Secret Engine

Secret을 저장하거나 동적으로 생성하는 백엔드이다.

| Engine | 설명 |
|--------|------|
| `kv` (v1/v2) | 정적 Key-Value 저장소. v2는 버전 관리 지원 |
| `database` | DB 접속 credential을 동적으로 생성/폐기 |
| `aws` | AWS IAM credential을 동적으로 생성 |
| `pki` | TLS 인증서를 동적으로 발급 |
| `transit` | 데이터 암/복호화 (Encryption as a Service) |

```bash
# KV v2 engine 활성화
vault secrets enable -path=secret kv-v2

# secret 저장
vault kv put secret/my-app/prod DB_PASSWORD="s3cret" API_KEY="abc123"

# secret 읽기
vault kv get secret/my-app/prod
```

### Auth Method

Vault에 로그인하는 방법이다. 로그인에 성공하면 Vault Token이 발급된다.

| Method | 설명 | 적합한 환경 |
|--------|------|------------|
| `token` | Vault Token 직접 사용 | 초기 설정, 관리자 |
| `userpass` | ID/PW 기반 | 사람이 직접 로그인 |
| `approle` | Role ID + Secret ID | CI/CD, 서버 애플리케이션 |
| `kubernetes` | K8s ServiceAccount Token | K8s Pod |
| `aws` | AWS IAM Role/Instance Profile | EC2, Lambda |
| `gcp` | GCP Service Account | GCE, GKE |
| `jwt/oidc` | JWT 토큰 검증 | SSO 연동 |

### Policy

누가 어떤 secret에 접근할 수 있는지 정의하는 규칙이다. HCL로 작성한다.

```hcl
# my-app-prod.hcl
# my-app의 prod secret만 읽기 가능
path "secret/data/my-app/prod" {
  capabilities = ["read"]
}

# my-app의 dev secret은 읽기/쓰기 가능
path "secret/data/my-app/dev" {
  capabilities = ["read", "create", "update"]
}

# 다른 모든 경로는 기본적으로 거부 (deny by default)
```

Capabilities:

| Capability | 설명 |
|------------|------|
| `create` | 새 secret 생성 |
| `read` | secret 읽기 |
| `update` | secret 수정 |
| `delete` | secret 삭제 |
| `list` | 경로 목록 조회 |
| `deny` | 명시적 거부 (다른 policy보다 우선) |

```bash
# policy 생성
vault policy write my-app-prod my-app-prod.hcl

# policy 확인
vault policy read my-app-prod
```

### Token과 Lease

Vault Token에는 수명(TTL)이 있다. 만료되면 해당 토큰으로 발급된 모든 secret도
함께 폐기된다.

```
Token 발급 (TTL: 1h)
    │
    ├── 0m: secret 접근 가능
    ├── 30m: renew 가능 (TTL 연장)
    ├── 60m: 만료 → secret 접근 불가
    │
    └── max_ttl: renew로도 연장 불가한 최대 수명
```

```bash
# 토큰 정보 확인
vault token lookup

# 토큰 갱신
vault token renew
```

## Vault 아키텍처

```
┌─────────────────────────────────────────────┐
│                Vault Server                  │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Auth     │  │ Secret   │  │ Audit     │  │
│  │ Methods  │  │ Engines  │  │ Devices   │  │
│  └────┬─────┘  └────┬─────┘  └─────┬─────┘  │
│       │              │              │         │
│  ┌────▼──────────────▼──────────────▼─────┐  │
│  │           Core (Policy Engine)          │  │
│  └────────────────┬───────────────────────┘  │
│                   │                           │
│  ┌────────────────▼───────────────────────┐  │
│  │        Storage Backend                  │  │
│  │  (Consul, Raft, S3, DynamoDB 등)       │  │
│  └────────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### Storage Backend

Vault 데이터가 물리적으로 저장되는 곳이다. 데이터는 항상 암호화되어 저장된다.

| Backend | 특징 |
|---------|------|
| **Raft** (Integrated Storage) | Vault 자체 내장. 별도 의존성 없음. 권장 |
| **Consul** | HashiCorp Consul 사용. HA 지원 |
| **S3** | AWS S3 사용. HA 미지원 (단독으로 비권장) |

### Seal / Unseal

Vault는 시작 시 **sealed** 상태이다. Unseal 해야 사용 가능하다.

```
Sealed (잠김)                    Unsealed (사용 가능)
  │                                  │
  │  unseal key 입력 (Shamir's       │  모든 기능 동작
  │  Secret Sharing: 5개 중 3개)     │
  │──────────────────────────────▶   │
  │                                  │
  │  재시작, 장애 시                   │
  │◀──────────────────────────────   │
```

프로덕션에서는 **Auto Unseal**을 사용한다:

| 방법 | 설명 |
|------|------|
| AWS KMS | AWS KMS 키로 자동 unseal |
| GCP CKMS | GCP Cloud KMS로 자동 unseal |
| Azure Key Vault | Azure KMS로 자동 unseal |
| HSM | 하드웨어 보안 모듈 사용 (Enterprise) |

## 동적 Secret 예시 (Database)

정적 비밀번호 대신 Vault가 필요할 때 DB 계정을 생성하고 TTL 만료 시 자동
삭제한다.

```bash
# Database secret engine 활성화
vault secrets enable database

# PostgreSQL 연결 설정
vault write database/config/mydb \
  plugin_name=postgresql-database-plugin \
  connection_url="postgresql://{{username}}:{{password}}@db.internal:5432/mydb" \
  allowed_roles="my-app-role" \
  username="vault_admin" \
  password="admin_password"

# Role 설정 (생성할 DB 유저의 SQL)
vault write database/roles/my-app-role \
  db_name=mydb \
  creation_statements="CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}'; GRANT SELECT ON ALL TABLES IN SCHEMA public TO \"{{name}}\";" \
  default_ttl="1h" \
  max_ttl="24h"

# 동적 credential 발급
vault read database/creds/my-app-role
# username: v-approle-my-app-r-abc123
# password: A1b2C3d4E5f6...
# lease_duration: 1h
```

1시간 후 Vault가 자동으로 이 DB 유저를 삭제한다. 유출되더라도 피해가 제한된다.

## Audit Log

모든 Vault 접근을 기록한다. 프로덕션에서 반드시 활성화해야 한다.

```bash
# 파일 기반 audit log
vault audit enable file file_path=/var/log/vault/audit.log

# syslog
vault audit enable syslog
```

로그 예시:
```json
{
  "type": "response",
  "auth": {
    "token_type": "service",
    "policies": ["my-app-prod"]
  },
  "request": {
    "operation": "read",
    "path": "secret/data/my-app/prod",
    "remote_address": "10.0.1.5"
  },
  "response": {
    "data": {
      "keys": ["DB_PASSWORD", "API_KEY"]
    }
  }
}
```

> 응답 데이터의 값은 HMAC으로 해시 처리되어 기록된다. 평문 secret이 audit log에
> 남지 않는다.
