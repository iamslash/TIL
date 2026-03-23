# GitHub Actions Secrets

## 개요

GitHub Actions Secrets는 credential을 GitHub 서버에 암호화하여 저장하고, workflow
실행 시 CI machine에 전달하는 방식이다. 환경변수 직접 관리보다 안전하지만,
credential이 존재한다는 근본적 한계가 있다.

## Secret 등록

**Settings > Secrets and variables > Actions > New repository secret**

```
Name:  REGISTRY_TOKEN
Value: abc123...
```

등록된 값은 다시 조회할 수 없다. 수정과 삭제만 가능하다.

## Workflow에서 사용

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

      - name: Login to registry
        run: |
          echo "${{ secrets.REGISTRY_TOKEN }}" | \
            docker login registry.internal \
              -u ci-user --password-stdin

      - name: Build and push
        run: |
          docker build -t registry.internal/my-app:${{ github.sha }} .
          docker push registry.internal/my-app:${{ github.sha }}

      - name: Cleanup
        if: always()
        run: docker logout registry.internal
```

`${{ secrets.XXX }}`로 참조하면 된다.

## Secret 전달 흐름

```
GitHub Server                         CI Machine (Self-hosted Runner)
┌──────────────┐                      ┌──────────────────────┐
│ Encrypted    │                      │ Runner Agent (상주)   │
│ Secrets      │                      │                      │
│ Storage      │  ① workflow 트리거    │                      │
│              │                      │                      │
│  ② 실행 직전 │  ③ HTTPS로           │  ④ 환경변수로         │
│    복호화    │ ──job payload 전달──▶ │    프로세스에 주입     │
│              │  (secret 값 포함)     │                      │
│              │                      │  ⑤ step 실행         │
│              │                      │                      │
│              │                      │  ⑥ job 종료 시       │
│              │                      │    프로세스와 함께     │
│              │                      │    메모리에서 제거     │
└──────────────┘                      └──────────────────────┘
```

## 보안 특성

```
✓ 저장 시 암호화 (libsodium sealed box)
✓ 로그에 자동 마스킹 (값이 ***로 치환)
✓ Fork된 repo의 PR workflow에서 접근 불가
✓ API로 값 조회 불가 (쓰기/삭제만 가능)
✗ CI machine 메모리에는 평문으로 존재
✗ /proc/{pid}/environ으로 읽기 가능 (VM 침해 시)
```

## Secret 범위 (Scope)

| 범위 | 설정 위치 | 적용 대상 |
|------|----------|----------|
| **Repository** | repo Settings > Secrets | 해당 repo만 |
| **Environment** | repo Settings > Environments > Secrets | 특정 environment만 |
| **Organization** | org Settings > Secrets | 선택한 repo들에 공유 |

### Environment Secret (권장)

환경별로 secret을 분리하고 승인 절차를 추가할 수 있다.

```
Settings > Environments > "production" 생성
  → Required reviewers: @team-lead   (배포 전 승인 필요)
  → Deployment branches: main만 허용
  → Environment secrets 등록
```

```yaml
jobs:
  deploy:
    runs-on: self-hosted
    environment: production    # 이 environment의 secret만 사용
    steps:
      - run: |
          echo "${{ secrets.PROD_REGISTRY_TOKEN }}" | \
            docker login registry.internal --password-stdin ...
```

이렇게 하면 `main` branch에서만, reviewer 승인 후에만 production secret에
접근할 수 있다.

## 로그 마스킹의 한계

GitHub Actions는 secret 값이 로그에 출력되면 `***`로 치환한다. 하지만
값을 변환하면 마스킹이 우회된다.

```yaml
# 나쁜 예 — base64 인코딩하면 원본과 달라 마스킹 안 됨
- run: echo "${{ secrets.TOKEN }}" | base64

# 나쁜 예 — 글자 사이에 공백을 넣으면 마스킹 안 됨
- run: echo "${{ secrets.TOKEN }}" | sed 's/./& /g'

# 좋은 예 — secret을 stdout에 출력하지 않는다
- run: |
    echo "${{ secrets.TOKEN }}" | docker login --password-stdin ...
```

## Self-hosted Runner 주의사항

GitHub-hosted runner는 job마다 새 VM을 만들고 끝나면 폐기한다. 하지만
self-hosted runner는 VM이 재사용되므로 추가 주의가 필요하다.

| 위험 | 대응 |
|------|------|
| workflow 종료 후 credential이 디스크에 잔류 | `docker logout`, 임시 파일 정리 |
| 다른 repo가 같은 runner 공유 | runner를 repo/org 단위로 격리 |
| 이전 job의 docker credential 잔류 | `~/.docker/config.json` 정리 |

```yaml
- name: Cleanup
  if: always()
  run: |
    docker logout registry.internal
    rm -f ~/.docker/config.json
```

## 본질적 한계

GitHub Secrets는 **credential 관리를 GitHub에 위임**하는 방식이다.

- GitHub 서버가 침해되면 secret이 노출될 수 있다
- 정적 토큰이므로 유출 시 수동으로 교체해야 한다
- 교체하지 않으면 유출된 토큰이 영구적으로 유효하다
- CI machine 메모리에 평문이 존재하는 것은 피할 수 없다

이 한계를 근본적으로 해결하려면 OIDC로 credential 자체를 없애야 한다.
