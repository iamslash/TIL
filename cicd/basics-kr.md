# CI/CD 개요

## CI/CD 란?

| 구분 | 의미 | 하는 일 |
|------|------|--------|
| **CI** (Continuous Integration) | 지속적 통합 | 코드 변경 시 자동으로 빌드, 테스트, 이미지 생성 |
| **CD** (Continuous Delivery/Deployment) | 지속적 배포 | 빌드된 결과물을 환경에 자동/수동으로 배포 |

## 현대 CI/CD 아키텍처 (권장)

```
Developer
    │
    ▼ git push
┌──────────────────┐
│ GitHub           │
│  ├── Source Code  │
│  └── Manifests   │
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐  ┌────────────────────────────┐
│ GitHub │  │ K8s Cluster                │
│Actions │  │  ┌────────┐  ┌──────────┐ │
│ (CI)   │  │  │ ArgoCD │  │ App Pod  │ │
│        │  │  │ (CD)   │──▶          │ │
│ Build  │  │  └───┬────┘  └──────────┘ │
│ Test   │  │      │ git pull            │
│ Push   │  └──────┼────────────────────┘
└───┬────┘         │
    │              │
    ▼              │
┌────────┐         │
│ Image  │         │
│Registry│◀────────┘ (image pull)
└────────┘
```

### CI (GitHub Actions)

1. 개발자가 코드를 push 한다
2. GitHub Actions가 자동으로 빌드, 테스트를 실행한다
3. Docker image를 빌드하여 registry에 push 한다
4. Manifest repo의 image tag를 업데이트한다

### CD (ArgoCD)

1. ArgoCD가 Git repo의 변경을 감지한다
2. 현재 K8s 상태와 Git의 desired state를 비교한다
3. 차이가 있으면 동기화하여 배포한다

### 이 조합의 장점

- **CI**: GitHub Actions OIDC로 credential 없이 registry 인증
- **CD**: ArgoCD가 K8s 내부에서 동작하므로 K8s credential 불필요
- **GitOps**: Git이 유일한 진실의 원천 (배포 상태가 항상 Git과 일치)
- **감사**: 모든 배포가 Git commit으로 추적 가능
