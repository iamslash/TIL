# GitHub Actions (CI)

## 개요

GitHub Actions는 GitHub에 내장된 CI/CD 플랫폼이다. 코드를 push 하면
자동으로 빌드, 테스트, 이미지 생성 등을 수행한다.

## 핵심 개념

### Workflow

`.github/workflows/` 디렉토리에 YAML로 정의하는 자동화 파이프라인이다.

### Trigger (on)

workflow를 실행하는 조건이다.

```yaml
on:
  push:
    branches: [main]        # main push 시
  pull_request:
    branches: [main]        # main으로 PR 시
  schedule:
    - cron: "0 9 * * *"     # 매일 09:00 UTC
  workflow_dispatch:          # 수동 실행
```

### Job & Step

```yaml
jobs:
  build:                     # Job 이름
    runs-on: ubuntu-latest   # 실행 환경
    steps:                   # Job 안의 단계들
      - uses: actions/checkout@v4    # Action 사용
      - run: npm test                # 셸 명령 실행
```

### Runner

workflow가 실행되는 머신이다.

| 종류 | 설명 |
|------|------|
| **GitHub-hosted** | GitHub이 제공하는 VM. job마다 새 VM 생성 후 폐기 |
| **Self-hosted** | 직접 운영하는 서버. 사내 네트워크 접근 가능 |

## OIDC 인증

GitHub Actions는 OIDC IdP가 내장되어 있어 credential 없이 외부 서비스에
인증할 수 있다.

```yaml
permissions:
  id-token: write    # OIDC 토큰 발행 허용

steps:
  - uses: aws-actions/configure-aws-credentials@v4
    with:
      role-to-assume: arn:aws:iam::123456789012:role/ci-deployer
      aws-region: ap-northeast-2
  # 이후 AWS CLI 사용 가능 — credential 없음
```

## Workflow 구성 요소

### Action

재사용 가능한 workflow 단위이다. Marketplace에서 찾거나 직접 만들 수 있다.

```yaml
# 공식 Action 사용
- uses: actions/checkout@v4
- uses: actions/setup-node@v4
  with:
    node-version: 20

# 서드파티 Action
- uses: docker/build-push-action@v5
```

### Secret

```yaml
# repo Settings > Secrets에서 등록
- run: echo "${{ secrets.MY_TOKEN }}" | docker login --password-stdin ...
```

### Environment

```yaml
jobs:
  deploy:
    environment: production   # 승인 필요, branch 제한 가능
```

### Matrix

여러 조건을 동시에 테스트한다.

```yaml
strategy:
  matrix:
    node-version: [18, 20, 22]
    os: [ubuntu-latest, macos-latest]
```

### Artifact

job 간 파일을 전달하거나 빌드 결과를 보관한다.

```yaml
- uses: actions/upload-artifact@v4
  with:
    name: build-output
    path: dist/
```

### Cache

의존성 설치 시간을 줄인다.

```yaml
- uses: actions/cache@v4
  with:
    path: node_modules
    key: ${{ runner.os }}-node-${{ hashFiles('package-lock.json') }}
```

## CI Workflow 예시 (전체)

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

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
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: write

    steps:
      - uses: actions/checkout@v4

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/ci-deployer
          aws-region: ap-northeast-2

      - uses: aws-actions/amazon-ecr-login@v2

      - name: Build and push
        run: |
          IMAGE=123456789012.dkr.ecr.ap-northeast-2.amazonaws.com/my-app
          docker build -t $IMAGE:${{ github.sha }} .
          docker push $IMAGE:${{ github.sha }}

      - name: Update manifest
        run: |
          cd manifests
          sed -i "s|image:.*|image: $IMAGE:${{ github.sha }}|" deployment.yaml
          git config user.name "github-actions"
          git config user.email "actions@github.com"
          git add .
          git commit -m "Update image to ${{ github.sha }}"
          git push
```
