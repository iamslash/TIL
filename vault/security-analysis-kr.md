# 보안 위협 분석 및 대응

## Secret 관리 방식별 보안 수준

```
방식                        보안 수준          주요 위험
─────────────────────────────────────────────────────────
코드 하드코딩               █░░░░░░░░░        git history에 영구 잔류
.env 파일                  ██░░░░░░░░        파일 유출, 백업에 포함
K8s Secret (base64)        ███░░░░░░░        etcd에 평문 저장 (기본값)
환경변수                    ███░░░░░░░        ps, /proc, 로그 노출
Sealed Secrets             █████░░░░░        클러스터 내 복호화 키 의존
External Secrets Operator  ██████░░░░        외부 저장소 연동, 동기화 지연
Vault + K8s Auth           █████████░        런타임만 메모리에 존재
Vault + Dynamic Secret     ██████████        자동 생성/폐기, 유출 시 피해 제한
```

## 위협 모델 (Threat Model)

### 1. 코드/저장소 레벨

| 위협 | 설명 | 대응 |
|------|------|------|
| Secret이 git에 커밋됨 | git history에 영구 잔류. force push로도 완전 삭제 어려움 | repo에 secret을 넣지 않음. git-secrets, gitleaks로 pre-commit 스캔 |
| Docker image에 secret 포함 | 레지스트리 접근 시 추출 가능 | multi-stage build, .dockerignore, trivy 스캔 |
| CI/CD 변수에 secret 저장 | Jenkins credential store 침해 시 전체 노출 | CI/CD는 secret을 모르는 구조로 설계 |

### 2. 런타임 레벨

| 위협 | 설명 | 대응 |
|------|------|------|
| 환경변수에서 secret 노출 | `/proc/{pid}/environ`, `ps auxe` | Vault Agent가 파일로 주입, 환경변수 미사용 |
| 메모리 덤프에서 추출 | core dump, 디버거 접근 | securityContext 강화, core dump 비활성화 |
| 로그에 secret 출력 | 실수로 로깅 | 로깅 필터 적용, secret 값 마스킹 |
| 사이드카 컨테이너 취약점 | Vault Agent 이미지 취약점 | 이미지 최신 유지, 취약점 스캔 |

### 3. 네트워크 레벨

| 위협 | 설명 | 대응 |
|------|------|------|
| Pod 간 트래픽 스니핑 | 같은 노드에서 패킷 캡처 | mTLS (service mesh), Network Policy |
| Vault 통신 도청 | Vault ↔ Pod 통신 | Vault은 항상 TLS (HTTPS) |
| DNS 스푸핑 | Vault 주소 위조 | Vault TLS 인증서 검증 |

### 4. 접근 제어 레벨

| 위협 | 설명 | 대응 |
|------|------|------|
| 다른 Pod에서 secret 접근 | namespace/SA 우회 | Vault K8s Auth의 바인딩 제약 |
| kubectl exec로 침입 | Pod 내부에서 파일/메모리 접근 | RBAC으로 exec 권한 제한 |
| 권한 상승 (Privilege Escalation) | 컨테이너 탈출 | securityContext, PodSecurityPolicy/Standards |
| 내부자 위협 | Vault 관리자가 secret 열람 | Vault audit log, 관리자 MFA, policy 분리 |

### 5. 시간 레벨

| 위협 | 설명 | 대응 |
|------|------|------|
| 장기 유효 토큰 탈취 | 한번 탈취하면 오래 사용 가능 | TTL 설정 (1h), 자동 만료 |
| 폐기된 Pod의 토큰 재사용 | Pod 종료 후에도 토큰 유효 | `agent-revoke-on-shutdown`, token bound CIDR |
| 유출된 DB 비밀번호 장기 사용 | 정적 비밀번호는 수동 변경 전까지 유효 | Dynamic Secret (자동 생성/폐기) |

## Defense in Depth (심층 방어)

단일 방어가 아닌 여러 레이어를 구성한다. 한 레이어가 뚫려도 다음 레이어가 방어한다.

```
Layer 1: 코드에 secret 없음
  │  ↓ 뚫리면 (실수로 커밋)
Layer 2: git-secrets / gitleaks가 차단
  │  ↓ 뚫리면 (스캔 누락)
Layer 3: Docker image에 포함 안 됨 (.dockerignore)
  │  ↓ 뚫리면 (image에 포함)
Layer 4: trivy 이미지 스캔이 감지
  │  ↓ 뚫리면 (스캔 우회)
Layer 5: K8s RBAC으로 접근 제한
  │  ↓ 뚫리면 (RBAC 우회)
Layer 6: Network Policy로 통신 제한
  │  ↓ 뚫리면 (네트워크 우회)
Layer 7: Vault Policy로 경로 제한
  │  ↓ 뚫리면 (policy 우회)
Layer 8: TTL로 피해 시간 제한
  │  ↓ 뚫리면 (TTL 내 악용)
Layer 9: Audit Log로 탐지 및 대응
```

## 프로덕션 체크리스트

### Vault 서버

- [ ] HA 구성 (Raft 또는 Consul backend)
- [ ] Auto Unseal (KMS 사용)
- [ ] TLS 활성화
- [ ] Audit Log 활성화 (2개 이상 backend 권장)
- [ ] Root Token 폐기 (초기 설정 후)
- [ ] 관리자 접근에 MFA 적용

### Policy

- [ ] 최소 권한 원칙 적용
- [ ] 환경별 policy 분리 (dev/staging/prod)
- [ ] `deny` capability로 민감 경로 명시적 차단
- [ ] 정기적 policy 리뷰

### K8s Auth

- [ ] SA + namespace 바인딩 설정
- [ ] TTL 적절히 설정 (1h 권장)
- [ ] ServiceAccount Token Projection 사용

### K8s Deployment

- [ ] securityContext 설정 (runAsNonRoot, readOnlyRootFilesystem 등)
- [ ] Resource limits 설정
- [ ] Network Policy 적용
- [ ] RBAC 최소 권한

### CI/CD

- [ ] Jenkins가 secret을 모르는 구조
- [ ] Jenkins의 K8s 권한 최소화 (image tag 변경만)
- [ ] 이미지 스캔 (trivy)
- [ ] .dockerignore에 sensitive 파일 포함

### 모니터링

- [ ] Vault audit log 수집 및 알림
- [ ] 비정상 접근 패턴 탐지 (같은 secret 대량 조회 등)
- [ ] Secret 로테이션 일정 관리
- [ ] Dynamic Secret 사용 가능한 곳은 전환

## 완벽한 보안은 없다

모든 대응을 적용해도 **100% 안전은 불가능**하다.

- Vault 서버 자체가 침해되면 모든 secret이 위험하다
- K8s control plane이 침해되면 SA Token이 위조될 수 있다
- 충분한 권한을 가진 내부자는 우회할 수 있다

목표는 **완벽한 보안이 아니라, 공격 비용을 최대화하고 피해를 최소화하는 것**이다.
TTL, Dynamic Secret, Audit Log는 "뚫려도 피해를 제한"하는 마지막 방어선이다.
