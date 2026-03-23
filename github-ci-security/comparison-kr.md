# 방식별 비교 및 권장 사항

## 보안 수준 비교

```
방식                    보안 수준          credential 존재 여부
──────────────────────────────────────────────────────────
환경변수 (VM 직접)      █░░░░░░░░░        VM에 평문 상주
GitHub Secrets          ████░░░░░░        GitHub에 암호화 저장, CI 메모리에 평문
Vault (KV + AppRole)    ██████░░░░        Vault에 암호화, CI 메모리에 평문, TTL 제한
Vault (Dynamic Secret)  ████████░░        매번 새 credential 생성, 자동 폐기
OIDC                    ██████████        credential 자체가 없음
```

## 상세 비교

| 항목 | 환경변수 | GitHub Secrets | Vault | OIDC |
|------|---------|----------------|-------|------|
| credential 저장 위치 | VM | GitHub 서버 | Vault 서버 | **없음** |
| 로그 마스킹 | 없음 | 자동 (`***`) | 자동 | 해당 없음 |
| 로테이션 | 수동 | 수동 | 자동 가능 | **불필요** |
| VM 침해 시 | 영구 토큰 노출 | 정적 토큰 노출 | 단기 토큰 노출 | **수분 토큰만 노출** |
| 탈취 후 유효 기간 | 무제한 | 수동 교체까지 | TTL (1시간 등) | **수분** |
| 감사 추적 | 없음 | 없음 | Audit Log | 클라우드 Audit Log |
| 추가 인프라 | 없음 | 없음 | Vault 운영 필요 | 없음 |
| 설정 복잡도 | 낮음 | 낮음 | 높음 | 중간 |
| repo 격리 | 없음 | repo/env별 | Policy별 | claim별 자동 격리 |

## 탈취 시나리오별 피해 비교

### 시나리오: CI machine이 침해되어 /proc에서 credential 탈취

```
환경변수:
  abc123 탈취 → 수년간 유효 → registry 무제한 접근
  피해 기간: 발견 및 수동 교체까지 (수일~수주)

GitHub Secrets:
  abc123 탈취 → 수년간 유효 → registry 무제한 접근
  피해 기간: 발견 및 수동 교체까지 (수일~수주)
  (환경변수와 동일 — 정적 토큰이라는 본질이 같으므로)

Vault Dynamic Secret:
  xyz789 탈취 → 1시간 후 만료 → 1시간만 접근 가능
  피해 기간: 최대 1시간 (자동)

OIDC:
  단기 토큰 탈취 → 5분 후 만료 → 5분만 접근 가능
  재발급 불가 (GitHub의 OIDC claim이 필요)
  피해 기간: 최대 5분 (자동)
```

## 의사결정 플로우차트

```
외부 서비스가 OIDC를 지원하는가?
  │
  ├── Yes → OIDC 사용 (최선)
  │
  └── No → Vault를 운영하고 있는가?
            │
            ├── Yes → Vault Dynamic Secret 사용
            │         (불가능하면 Vault KV + AppRole)
            │
            └── No → GitHub Secrets + Environment Secret 사용
                      (최소한의 보안)
                      │
                      └── 환경변수 직접 관리 → 하지 마라
```

## 실제 환경별 권장

### 클라우드 환경 (AWS/GCP/Azure)

```yaml
# OIDC 사용 — credential 없음
permissions:
  id-token: write
steps:
  - uses: aws-actions/configure-aws-credentials@v4
    with:
      role-to-assume: arn:aws:iam::123456789012:role/ci-deployer
```

### JFrog Artifactory

```yaml
# OIDC 사용 — credential 없음
permissions:
  id-token: write
steps:
  - uses: jfrog/setup-jfrog-cli@v4
    with:
      oidc-provider-name: github-actions
```

### OIDC 미지원 사내 Registry + Vault 있음

```yaml
# Vault Dynamic Secret
steps:
  - uses: hashicorp/vault-action@v3
    with:
      url: https://vault.internal:8200
      method: approle
      roleId: ${{ secrets.VAULT_ROLE_ID }}
      secretId: ${{ secrets.VAULT_SECRET_ID }}
      secrets: |
        ci/data/registry token | REGISTRY_TOKEN
```

### OIDC 미지원 + Vault 없음

```yaml
# GitHub Secrets (Environment Secret 권장)
jobs:
  deploy:
    environment: production
    steps:
      - run: |
          echo "${{ secrets.REGISTRY_TOKEN }}" | \
            docker login --password-stdin ...
```

이 경우 반드시:
- Environment Secret을 사용하여 branch/승인 제한
- 정기적으로 토큰 로테이션 (90일 권장)
- runner를 repo 단위로 격리

## 마이그레이션 경로

기존에 GitHub Secrets로 운영하고 있다면 단계적으로 전환할 수 있다.

```
현재                    단기                    장기
GitHub Secrets     →   Environment Secret  →   OIDC
(repo secret)          (branch/승인 제한)       (credential 없음)
```

1단계는 설정 변경만으로 즉시 가능하다. OIDC 전환은 외부 서비스 측 설정이
필요하므로 계획을 세워 진행한다.
