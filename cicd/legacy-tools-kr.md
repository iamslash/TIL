# 레거시 및 대안 CI/CD 도구

## 개요

아래 도구들은 모두 CI/CD 영역에서 사용된 적이 있거나 현재도 일부 사용 중이다.
하지만 **새로 구축하는 환경에서는 거의 선택되지 않는 추세**이다. 각 도구별로
사용이 줄어드는 이유를 정리한다.

---

## Jenkins

| 항목 | 내용 |
|------|------|
| 유형 | 자체 호스팅 CI/CD 서버 |
| 전성기 | 2010~2020 |
| 현재 | 레거시 유지, 신규 도입 거의 없음 |

### 사용이 줄어드는 이유

| 이유 | 설명 |
|------|------|
| **서버 운영 부담** | Jenkins 자체를 직접 설치/관리/업그레이드해야 한다. 플러그인 호환성 문제가 빈번하다 |
| **보안** | OIDC IdP 미내장. credential을 Jenkins 내부에 저장해야 하므로 Vault 같은 추가 인프라가 필요하다 |
| **확장성** | agent 스케일링, 큐 관리, 리소스 할당을 직접 구성해야 한다 |
| **코드와 CI 분리** | 소스 코드는 GitHub에, CI 설정은 Jenkins에 따로 있어 관리 포인트가 분산된다 |
| **느린 피드백** | GitHub Actions는 push 즉시 트리거되지만 Jenkins는 webhook 설정이 추가로 필요하다 |
| **UI가 오래됨** | Blue Ocean 프로젝트가 중단되었고, 기본 UI는 현대적이지 않다 |

---

## Spinnaker

| 항목 | 내용 |
|------|------|
| 유형 | 멀티 클라우드 CD 플랫폼 (Netflix 개발) |
| 전성기 | 2017~2021 |
| 현재 | 대규모 멀티 클라우드 환경에서만 유지 |

### 사용이 줄어드는 이유

| 이유 | 설명 |
|------|------|
| **운영 복잡도 극심** | 10개 이상의 마이크로서비스(Deck, Gate, Orca, Clouddriver 등)를 직접 운영해야 한다 |
| **리소스 소비** | 최소 16GB RAM, 4 CPU 이상 필요. 소규모 팀에게는 과도하다 |
| **K8s credential 필요** | Push 방식이므로 외부에서 K8s에 접근하는 credential이 필요하다 |
| **GitOps 미지원** | Git을 진실의 원천으로 사용하지 않는다. 파이프라인 상태가 중심이다 |
| **커뮤니티 축소** | Netflix가 Spinnaker 팀 규모를 줄였다. 오픈소스 기여와 업데이트가 둔화되었다 |
| **학습 곡선** | 설정과 운영이 복잡하여 전담 인력이 필요하다 |

---

## Travis CI

| 항목 | 내용 |
|------|------|
| 유형 | SaaS CI |
| 전성기 | 2013~2019 (오픈소스 무료 CI의 대명사) |
| 현재 | 거의 사용되지 않음 |

### 사용이 줄어드는 이유

| 이유 | 설명 |
|------|------|
| **Idera 인수 후 악화** | 2019년 Idera에 인수된 후 무료 플랜 축소, 서비스 품질 저하 |
| **오픈소스 무료 빌드 제거** | 오픈소스 프로젝트의 무료 빌드 크레딧이 사실상 사라짐 |
| **보안 사고** | 2021년 사용자 secret 유출 사고 발생 |
| **GitHub Actions 등장** | GitHub에 내장된 CI가 Travis CI의 존재 이유를 대체함 |

---

## CircleCI

| 항목 | 내용 |
|------|------|
| 유형 | SaaS CI/CD |
| 전성기 | 2016~2022 |
| 현재 | 사용 중이나 신규 도입 감소 |

### 사용이 줄어드는 이유

| 이유 | 설명 |
|------|------|
| **2023년 보안 사고** | 전체 사용자의 secret 로테이션을 권고하는 대규모 보안 침해 발생 |
| **비용** | GitHub Actions 대비 가격 경쟁력이 떨어진다 |
| **별도 서비스** | GitHub와 별도로 CircleCI를 관리해야 하는 오버헤드 |
| **OIDC 지원** | CircleCI도 OIDC를 지원하지만, GitHub Actions가 이미 GitHub과 통합되어 있어 더 간편하다 |

---

## GitLab CI

| 항목 | 내용 |
|------|------|
| 유형 | GitLab 내장 CI/CD |
| 현재 | GitLab을 사용하는 조직에서는 여전히 주력 |

### GitHub Actions 대비 약점

| 이유 | 설명 |
|------|------|
| **GitHub 생태계** | GitHub이 소스 코드 호스팅 시장 점유율 1위. GitHub을 쓰면 GitLab CI를 쓸 이유가 없다 |
| **Action Marketplace** | GitHub Actions의 재사용 Action 생태계가 더 크다 |
| **OIDC** | 둘 다 지원하지만 GitHub Actions쪽 클라우드 연동 Action이 더 풍부하다 |

> GitLab을 이미 사용하는 조직이라면 GitLab CI는 좋은 선택이다.
> "잘 안 쓰인다"기보다는 "GitHub 사용자에게는 선택지가 아니다"에 가깝다.

---

## TeamCity (JetBrains)

| 항목 | 내용 |
|------|------|
| 유형 | 자체 호스팅 CI/CD (JetBrains) |
| 현재 | JetBrains 생태계 사용 조직에서 일부 사용 |

### 사용이 줄어드는 이유

| 이유 | 설명 |
|------|------|
| **자체 호스팅** | Jenkins와 동일한 서버 운영 부담 |
| **라이선스 비용** | 무료 플랜이 제한적이고, 에이전트 수에 따라 비용 증가 |
| **생태계** | GitHub Actions나 GitLab CI 대비 통합 Action/플러그인이 적다 |

---

## Bamboo (Atlassian)

| 항목 | 내용 |
|------|------|
| 유형 | 자체 호스팅 CI/CD (Atlassian) |
| 현재 | Atlassian이 신규 판매 중단 발표 |

### 사용이 줄어드는 이유

| 이유 | 설명 |
|------|------|
| **EOL 선언** | Atlassian이 Bamboo Server/Data Center의 신규 라이선스 판매를 중단했다 |
| **Bitbucket Pipelines로 대체** | Atlassian은 Bitbucket Pipelines(SaaS)로 전환을 유도하고 있다 |
| **자체 호스팅** | 서버 운영 부담 |

---

## AWS CodePipeline / CodeBuild

| 항목 | 내용 |
|------|------|
| 유형 | AWS 네이티브 CI/CD |
| 현재 | AWS에 깊이 묶인 조직에서 사용 |

### 범용적으로 안 쓰이는 이유

| 이유 | 설명 |
|------|------|
| **AWS 종속** | AWS 외 환경에서는 사용할 수 없다 |
| **설정 복잡** | CodePipeline + CodeBuild + CodeDeploy 조합이 복잡하다 |
| **UI/UX** | GitHub Actions 대비 개발자 경험이 떨어진다 |
| **생태계** | 재사용 Action이나 커뮤니티가 상대적으로 작다 |

---

## 요약

```
도구                  상태              신규 도입 권장
──────────────────────────────────────────────────
Jenkins              레거시 유지        X
Spinnaker            레거시 유지        X (멀티 클라우드 제외)
Travis CI            사실상 사망        X
CircleCI             유지 중           △ (보안 사고 이후 신뢰 하락)
GitLab CI            활발              O (GitLab 사용 시)
TeamCity             유지 중           △
Bamboo               EOL              X
AWS CodePipeline     유지 중           △ (AWS only)
─────────────────────────────────────────────────
GitHub Actions       성장 중           O (권장)
ArgoCD               성장 중           O (K8s CD 권장)
```
