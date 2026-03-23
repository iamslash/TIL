# 환경변수의 문제점

## 개요

CI machine (self-hosted runner)에 환경변수로 credential을 설정하는 방식은 가장
단순하지만, 보안상 가장 취약하다.

```bash
# VM에 직접 설정
export REGISTRY_TOKEN=abc123
```

## 노출 경로

사내 VM이라도 다음 경로로 credential이 노출될 수 있다.

### 1. /proc/{pid}/environ

같은 VM에서 root 또는 같은 user 권한을 가진 프로세스가 읽을 수 있다.

```bash
# 공격자가 Runner 프로세스의 환경변수를 읽음
cat /proc/1234/environ | tr '\0' '\n'
# REGISTRY_TOKEN=abc123  ← 노출
```

### 2. ps 명령어

```bash
ps auxe | grep runner
# 환경변수가 프로세스 목록에 노출
```

### 3. CI 로그

실수로 환경변수를 로깅하면 로그 파일에 영구 기록된다.

```yaml
# 실수로 출력
- run: echo "Token is $REGISTRY_TOKEN"
# 또는 debug mode에서 전체 환경변수 출력
- run: env
```

### 4. VM 스냅샷 / 백업

VM 스냅샷을 생성하면 메모리 상태가 그대로 포함된다. 스냅샷에 접근 가능한
사람은 환경변수를 복원할 수 있다.

### 5. Runner 공유 환경

하나의 VM에서 여러 repo의 workflow가 실행되면, 다른 repo의 workflow가
환경변수에 접근할 수 있다.

```
VM (Self-hosted Runner)
├── Repo A workflow → REGISTRY_TOKEN 접근 가능
├── Repo B workflow → REGISTRY_TOKEN 접근 가능  ← 의도하지 않은 접근
└── Repo C workflow → REGISTRY_TOKEN 접근 가능  ← 의도하지 않은 접근
```

### 6. SSH 접근

VM에 SSH 접근 가능한 사람은 환경변수를 자유롭게 조회할 수 있다.

```bash
ssh ci-machine
export  # 모든 환경변수 출력
```

## "사내 VM이니까 안전하다"는 착각

```
외부 공격:   방화벽, VPN으로 어느 정도 차단 가능
내부자 위협: VM 접근 권한이 있는 모든 직원이 잠재적 위험
실수:       로그 출력, 설정 파일 커밋 등 인적 오류
```

사내 네트워크가 외부 공격을 줄여주지만, 내부자 위협과 인적 오류는 전혀
해결하지 못한다.

## 결론

환경변수 직접 관리는 어떤 환경에서든 권장하지 않는다. 최소한 GitHub Actions
Secrets를 사용하고, 가능하면 OIDC로 credential 자체를 없애는 것이 바람직하다.
