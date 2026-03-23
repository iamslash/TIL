# Vault Dynamic Secret

## 개요

Vault Dynamic Secret은 CI가 실행될 때마다 **새로운 credential을 생성**하고,
일정 시간 후 **자동으로 폐기**하는 방식이다. 정적 토큰과 달리 탈취되어도 피해
시간이 제한된다.

## 정적 Secret vs Dynamic Secret

```
정적 Secret:
──────────────────────────────────────────────▶ 시간
  abc123          (수년간 동일한 값)
  생성                                    수동 폐기


Dynamic Secret:
──────────────────────────────────────────────▶ 시간
  xyz789  (1시간 후 자동 폐기)
  ├──────┤
          pqr456  (1시간 후 자동 폐기)
          ├──────┤
                   mno321  (1시간 후 자동 폐기)
                   ├──────┤
  CI 실행①  CI 실행②  CI 실행③
```

| 항목 | 정적 Secret | Dynamic Secret |
|------|------------|----------------|
| 값 | 항상 동일 | 매번 다름 |
| 수명 | 수동 교체 전까지 영구 | TTL 만료 시 자동 폐기 |
| 탈취 시 | 교체 전까지 무제한 사용 | 최대 TTL 시간만 사용 가능 |
| 로테이션 | 수동 | 자동 |

## GitHub Actions + Vault 연동

### 사전 조건

- Vault 서버가 운영 중이어야 한다
- Vault에 AppRole 또는 JWT auth가 설정되어 있어야 한다

### Vault 설정

```bash
# Secret Engine 활성화
vault secrets enable -path=ci kv-v2

# CI용 credential 저장
vault kv put ci/registry \
  username="ci-user" \
  password="s3cret"

# Policy 생성 (읽기만 허용)
vault policy write ci-deployer - <<EOF
path "ci/data/registry" {
  capabilities = ["read"]
}
EOF

# AppRole 설정
vault auth enable approle

vault write auth/approle/role/ci-deployer \
  token_policies="ci-deployer" \
  token_ttl=15m \
  token_max_ttl=30m \
  secret_id_ttl=5m \
  secret_id_num_uses=1
```

### Workflow

```yaml
# .github/workflows/build.yml
name: Build and Push

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4

      - name: Get credentials from Vault
        uses: hashicorp/vault-action@v3
        with:
          url: https://vault.internal:8200
          method: approle
          roleId: ${{ secrets.VAULT_ROLE_ID }}
          secretId: ${{ secrets.VAULT_SECRET_ID }}
          secrets: |
            ci/data/registry username | REGISTRY_USER ;
            ci/data/registry password | REGISTRY_PASS

      - name: Build and push
        run: |
          echo "$REGISTRY_PASS" | \
            docker login registry.internal -u "$REGISTRY_USER" --password-stdin
          docker build -t registry.internal/my-app:${{ github.sha }} .
          docker push registry.internal/my-app:${{ github.sha }}

      - name: Cleanup
        if: always()
        run: docker logout registry.internal
```

### Database Dynamic Secret (진정한 Dynamic)

위 예시는 KV에 저장된 정적 credential을 Vault를 경유해서 읽는 것이다.
진정한 Dynamic Secret은 Vault가 **매번 새 계정을 생성하고 자동 폐기**하는 것이다.

```bash
# Database Secret Engine 설정
vault secrets enable database

vault write database/config/registry-db \
  plugin_name=postgresql-database-plugin \
  connection_url="postgresql://{{username}}:{{password}}@db.internal:5432/registry" \
  allowed_roles="ci-role" \
  username="vault_admin" \
  password="admin_pass"

vault write database/roles/ci-role \
  db_name=registry-db \
  creation_statements="CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}'; GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA public TO \"{{name}}\";" \
  default_ttl="30m" \
  max_ttl="1h"
```

```bash
# CI가 요청할 때마다 새 DB 계정 발급
vault read database/creds/ci-role
# username: v-approle-ci-role-abc123    ← 이 순간 생성됨
# password: A1b2C3d4E5
# lease_duration: 30m                   ← 30분 후 자동 삭제
```

## Vault 방식의 트레이드오프

### 장점

- 탈취 시 피해가 TTL로 제한된다
- 자동 로테이션 (Dynamic Secret 사용 시)
- Audit Log로 누가 언제 접근했는지 추적 가능
- 세밀한 Policy로 접근 경로 제한

### 단점

- Vault 서버를 직접 운영해야 한다 (가용성, 백업, 업그레이드)
- 복잡도 증가 (AppRole 설정, Policy 관리)
- Vault 장애 시 CI 전체가 중단될 수 있다
- GitHub Secrets에 여전히 `VAULT_ROLE_ID`, `VAULT_SECRET_ID`를 저장해야 한다

### "결국 GitHub Secrets가 필요한 거 아닌가?"

맞다. AppRole 방식은 Vault에 로그인하기 위한 credential이 GitHub Secrets에
저장된다. 하지만 차이가 있다:

```
GitHub Secrets만 사용:
  REGISTRY_TOKEN 탈취 → 무제한 접근

Vault + AppRole:
  VAULT_SECRET_ID 탈취 → Vault 로그인 가능
    → 그러나 Secret ID는 일회용, 5분 후 만료
    → Vault Policy로 접근 범위 제한
    → Audit Log로 탐지 가능
```

이 문제를 완전히 해결하려면 OIDC를 사용한다.
