# 서브에이전트와 역할 분리

> 커리큘럼 05-CH01 대응 문서

## 목차

- [1. 서브에이전트는 언제 단일 에이전트보다 유리한가](#1-서브에이전트는-언제-단일-에이전트보다-유리한가)
- [2. 탐색·구현·검증 역할 나누기](#2-탐색구현검증-역할-나누기)
  - [2.1 탐색 담당 (Explorer)](#21-탐색-담당-explorer)
  - [2.2 구현 담당 (Executor)](#22-구현-담당-executor)
  - [2.3 검증 담당 (Verifier)](#23-검증-담당-verifier)
- [3. 결과 합치기와 메인 컨텍스트 관리](#3-결과-합치기와-메인-컨텍스트-관리)
  - [3.1 서브에이전트 결과 요약 패턴](#31-서브에이전트-결과-요약-패턴)
  - [3.2 run_in_background 활용법](#32-run_in_background-활용법)
- [4. 실전 워크플로우: 버그 수정 시 3개 서브에이전트 협업](#4-실전-워크플로우-버그-수정-시-3개-서브에이전트-협업)

---

## 1. 서브에이전트는 언제 단일 에이전트보다 유리한가

### 판단 기준 체크리스트

다음 조건 중 **3개 이상**에 해당하면 서브에이전트 분리를 고려한다.

```
[ ] 작업이 독립적으로 병렬 실행 가능한 부분으로 나뉜다
    예) 프론트엔드 작업 + 백엔드 작업이 동시에 가능할 때

[ ] 각 단계의 컨텍스트(파일, 정보)가 서로 거의 겹치지 않는다
    예) 탐색 단계에서 읽는 파일 vs 구현 단계에서 수정하는 파일이 다를 때

[ ] 한 에이전트가 모든 과정을 처리하면 컨텍스트 창이 80% 이상 찰 것으로 예상된다
    예) 대규모 리팩토링, 여러 모듈을 동시에 다루는 작업

[ ] 역할별로 전문화된 프롬프트/시스템 지시가 필요하다
    예) 보안 리뷰어는 취약점 관점, 성능 리뷰어는 병목 관점으로 봐야 할 때

[ ] 한 단계의 결과가 다음 단계의 입력이 되지만, 결과 요약본만 전달해도 충분하다
    예) "탐색 결과: 문제는 auth.ts 132번째 줄에 있음" → 이 요약만 구현자에게 전달

[ ] 실패 격리가 중요하다 (한 서브에이전트가 실패해도 다른 에이전트가 독립적으로 동작)
    예) 테스트 실행 중 오류가 나도 문서 작성 에이전트는 계속 진행
```

### 단일 에이전트가 더 나은 경우

```
[ ] 작업이 2-3개의 파일 수정으로 끝나는 단순한 경우
[ ] 각 단계가 강하게 결합되어 있어 상세한 컨텍스트 전달이 필요한 경우
[ ] 서브에이전트 조율 오버헤드가 실제 작업 시간보다 클 것으로 예상되는 경우
[ ] 작업 시간이 30초 미만으로 예상되는 경우
```

### 의사결정 흐름도

```
작업 수신
    │
    ▼
작업을 독립적인 부분으로 나눌 수 있는가?
    │
    ├─ 아니오 ──→ 단일 에이전트로 처리
    │
    └─ 예
        │
        ▼
    컨텍스트가 80% 이상 찰 것으로 예상되는가?
        │
        ├─ 아니오 ──→ 단일 에이전트로 처리 (단, 병렬 실행 활용)
        │
        └─ 예
            │
            ▼
        서브에이전트 분리 적용
        (explore → execute → verify)
```

---

## 2. 탐색·구현·검증 역할 나누기

Claude Code의 `Agent` 도구를 사용하면 메인 에이전트가 서브에이전트를 생성하여 특정 작업을 위임할 수 있다. 각 서브에이전트는 독립적인 컨텍스트를 가지고 동작한다.

### 2.1 탐색 담당 (Explorer)

탐색 에이전트의 역할: **읽기 전용**으로 코드베이스를 분석하고 구현에 필요한 정보를 수집한다.

#### Explorer 서브에이전트 프롬프트 예시

```
당신은 코드베이스 탐색 전문가입니다. 다음 규칙을 엄격히 따르세요:

규칙:
1. 파일을 읽고 분석만 한다. 어떤 파일도 수정하지 않는다.
2. 발견한 내용을 구조화된 형식으로 보고한다.
3. 추측하지 않는다. 실제로 확인한 사실만 보고한다.

탐색 목표:
- 버그 위치: `UserService.authenticate()` 메서드에서 JWT 검증이 실패하는 원인
- 조사 범위: src/auth/, src/services/user.service.ts, tests/auth/

보고 형식:
## 탐색 결과

### 버그 위치
- 파일: <정확한 파일 경로>
- 라인: <라인 번호>
- 코드: <문제가 되는 코드 스니펫>

### 근본 원인
<왜 이 버그가 발생하는지 설명>

### 영향 범위
- 영향받는 파일 목록
- 영향받는 테스트 목록

### 구현 시 주의사항
<구현 에이전트에게 전달할 핵심 정보>
```

#### Claude Code에서 Explorer 에이전트 호출

```python
# Claude Code의 Agent 도구 사용 예시 (메인 에이전트에서 실행)

result = Agent(
    prompt="""
    당신은 코드베이스 탐색 전문가입니다. 파일을 수정하지 않고 읽기만 합니다.

    탐색 목표:
    - src/auth/jwt.service.ts에서 토큰 만료 검증 로직을 찾는다
    - 해당 로직을 사용하는 모든 파일을 식별한다
    - 현재 구현의 문제점을 파악한다

    사용 가능한 도구: Read, Glob, Grep (Write, Edit 사용 금지)

    보고 형식:
    - 문제 위치 (파일:라인)
    - 근본 원인 1-2문장 요약
    - 수정이 필요한 파일 목록
    - 테스트 파일 위치
    """,
    tools=["Read", "Glob", "Grep"]  # 읽기 도구만 허용
)
```

#### 실제 Explorer 출력 예시

```
## 탐색 결과

### 버그 위치
- 파일: src/auth/jwt.service.ts
- 라인: 47-52
- 코드:
  ```typescript
  const decoded = jwt.verify(token, process.env.JWT_SECRET);
  // 문제: exp 필드를 별도로 검사하지 않음
  // jwt.verify()는 만료를 자동 검사하지만,
  // iat(발급 시간)이 미래인 경우를 처리하지 않음
  return decoded as JwtPayload;
  ```

### 근본 원인
jwt.verify()가 exp(만료 시간) 검증은 하지만, iat(발급 시간)이 미래 시점으로
설정된 토큰을 거부하지 않는다. 서버 시간과 클라이언트 시간이 어긋날 때 발생.

### 영향 범위
- src/auth/jwt.service.ts (수정 필요)
- src/middleware/auth.middleware.ts (의존)
- tests/auth/jwt.service.spec.ts (테스트 추가 필요)

### 구현 시 주의사항
- clockTolerance 옵션을 30초로 설정하면 해결 가능
- JwtPayload 타입 정의는 src/types/auth.types.ts에 있음
- 기존 테스트 3개가 이 함수를 직접 테스트함 (수정 시 영향 있음)
```

---

### 2.2 구현 담당 (Executor)

구현 에이전트의 역할: 탐색 결과를 받아 **최소한의 변경**으로 수정을 구현한다.

#### Executor 서브에이전트 프롬프트 예시

```
당신은 정밀한 코드 구현 전문가입니다.

탐색 에이전트가 전달한 분석 결과:
---
[탐색 결과를 여기에 붙여넣기]
---

구현 규칙:
1. 탐색 결과에서 명시된 파일만 수정한다.
2. 범위를 벗어난 "개선"은 하지 않는다.
3. 각 변경 후 변경 이유를 주석으로 남긴다.
4. 수정 완료 후 다음 형식으로 보고한다:

## 구현 결과

### 수정된 파일
- <파일 경로>: <변경 요약 1줄>

### 변경 내용 상세
<각 변경의 이전/이후 코드>

### 검증 에이전트에게 전달할 정보
- 테스트해야 할 시나리오 목록
- 엣지 케이스 목록
- 변경으로 인해 깨질 수 있는 기존 테스트
```

#### Claude Code에서 Executor 에이전트 호출

```python
# 탐색 결과를 Executor에게 전달
explorer_output = """
버그 위치: src/auth/jwt.service.ts:47-52
근본 원인: iat 미래 시점 토큰을 거부하지 않음
수정 방법: clockTolerance: 30 옵션 추가
"""

result = Agent(
    prompt=f"""
    당신은 정밀한 코드 구현 전문가입니다.

    탐색 결과:
    {explorer_output}

    구현 지시:
    1. src/auth/jwt.service.ts의 jwt.verify() 호출에 clockTolerance: 30 옵션을 추가한다.
    2. 변경 이유를 코드 주석으로 명시한다.
    3. 명시된 파일 외에는 수정하지 않는다.
    4. 수정 후 변경 내용을 보고한다.

    사용 가능한 도구: Read, Edit, Write, Bash (테스트 실행용)
    """,
    tools=["Read", "Edit", "Write", "Bash"]
)
```

#### 실제 Executor 출력 예시

```
## 구현 결과

### 수정된 파일
- src/auth/jwt.service.ts: jwt.verify()에 clockTolerance 옵션 추가

### 변경 내용 상세

이전:
```typescript
const decoded = jwt.verify(token, process.env.JWT_SECRET);
```

이후:
```typescript
const decoded = jwt.verify(token, process.env.JWT_SECRET, {
  // 서버-클라이언트 시간 차이 최대 30초 허용
  // 관련 이슈: GitHub #234 - JWT iat 미래 시점 버그
  clockTolerance: 30,
});
```

### 검증 에이전트에게 전달할 정보
테스트해야 할 시나리오:
1. 정상 토큰 - 인증 성공해야 함
2. 만료된 토큰 - 인증 실패해야 함
3. iat가 29초 미래인 토큰 - clockTolerance 내이므로 성공해야 함
4. iat가 31초 미래인 토큰 - 허용 범위 초과이므로 실패해야 함
5. JWT_SECRET이 없는 경우 - 에러를 던져야 함

깨질 수 있는 기존 테스트:
- tests/auth/jwt.service.spec.ts: 3번 테스트 케이스 (옵션 없이 호출하는 mock 테스트)
```

---

### 2.3 검증 담당 (Verifier)

검증 에이전트의 역할: 구현 결과를 독립적으로 검토하고 **증거 기반으로 완료 여부를 판정**한다.

#### Verifier 서브에이전트 프롬프트 예시

```
당신은 독립적인 검증 전문가입니다. 구현 에이전트의 결과를 비판적으로 검토합니다.

구현 에이전트의 보고:
---
[구현 결과를 여기에 붙여넣기]
---

검증 절차:
1. 수정된 파일을 직접 읽어 변경 내용이 보고와 일치하는지 확인한다.
2. 테스트를 실행하고 결과를 확인한다.
3. 구현 에이전트가 제시한 시나리오를 모두 검증한다.
4. 변경으로 인한 사이드 이펙트가 없는지 확인한다.

판정 기준:
- PASS: 모든 시나리오가 예상대로 동작하고 기존 테스트가 통과함
- PARTIAL: 일부 시나리오가 실패하거나 개선이 필요한 부분이 있음
- FAIL: 핵심 기능이 동작하지 않거나 기존 테스트가 깨짐

보고 형식:
## 검증 결과

### 판정: [PASS / PARTIAL / FAIL]

### 테스트 실행 결과
[실제 테스트 출력]

### 시나리오별 검증
| 시나리오 | 예상 결과 | 실제 결과 | 통과 여부 |
|---------|---------|---------|---------|

### 발견된 문제 (있는 경우)
[문제 설명과 위치]

### 권장 사항
[추가 수정 또는 모니터링이 필요한 사항]
```

#### Claude Code에서 Verifier 에이전트 호출

```python
executor_output = """
수정 파일: src/auth/jwt.service.ts
변경: clockTolerance: 30 옵션 추가
테스트 시나리오: [5가지 시나리오 목록]
"""

result = Agent(
    prompt=f"""
    당신은 독립적인 검증 전문가입니다.

    구현 결과:
    {executor_output}

    검증 지시:
    1. src/auth/jwt.service.ts를 읽어 변경이 올바른지 확인한다.
    2. npm test -- --testPathPattern=auth 로 테스트를 실행한다.
    3. 테스트 결과를 바탕으로 PASS/PARTIAL/FAIL을 판정한다.
    4. 실제 테스트 출력을 보고서에 포함한다.

    중요: 직접 코드를 수정하지 않는다. 판정만 내린다.
    """,
    tools=["Read", "Bash", "Grep"]  # 쓰기 도구 제외
)
```

#### 실제 Verifier 출력 예시

```
## 검증 결과

### 판정: PASS

### 테스트 실행 결과
```
PASS tests/auth/jwt.service.spec.ts
  JwtService
    ✓ 정상 토큰 검증 (12ms)
    ✓ 만료된 토큰 거부 (8ms)
    ✓ clockTolerance 29초 이내 허용 (15ms)
    ✓ clockTolerance 31초 초과 거부 (9ms)
    ✓ JWT_SECRET 없을 때 에러 발생 (6ms)

Test Suites: 1 passed, 1 total
Tests:       5 passed, 5 total
```

### 시나리오별 검증
| 시나리오 | 예상 결과 | 실제 결과 | 통과 여부 |
|---------|---------|---------|--------|
| 정상 토큰 | 성공 | 성공 | ✓ |
| 만료 토큰 | 실패 | 실패 | ✓ |
| iat+29초 | 성공 | 성공 | ✓ |
| iat+31초 | 실패 | 실패 | ✓ |
| 시크릿 없음 | 에러 | 에러 | ✓ |

### 발견된 문제
없음

### 권장 사항
- clockTolerance 값(30초)을 환경변수로 추출하면 환경별 설정이 가능함
- 운영 환경 모니터링: 향후 1주일간 JWT 인증 실패율 추이 확인 권장
```

---

## 3. 결과 합치기와 메인 컨텍스트 관리

### 3.1 서브에이전트 결과 요약 패턴

서브에이전트가 작업을 완료하면 전체 내용이 아닌 **구조화된 요약**만 메인 컨텍스트로 가져온다. 이렇게 하면 메인 에이전트의 컨텍스트 창이 낭비되지 않는다.

#### 요약 추출 프롬프트 패턴

```
서브에이전트의 작업이 완료되었습니다. 다음 형식으로 결과를 요약해주세요:

요약 형식 (전체 응답의 마지막에 포함):

=== AGENT_RESULT_SUMMARY ===
STATUS: [SUCCESS | PARTIAL | FAILED]
MODIFIED_FILES: [수정된 파일 목록, 없으면 NONE]
KEY_FINDINGS: [핵심 발견사항 3줄 이내]
NEXT_AGENT_CONTEXT: [다음 에이전트에게 전달할 핵심 정보]
BLOCKERS: [차단 요소, 없으면 NONE]
=== END_SUMMARY ===
```

#### 메인 에이전트에서 요약 파싱 예시

```python
def parse_agent_summary(agent_output: str) -> dict:
    """서브에이전트 출력에서 구조화된 요약을 추출한다."""
    import re

    pattern = r'=== AGENT_RESULT_SUMMARY ===\n(.*?)\n=== END_SUMMARY ==='
    match = re.search(pattern, agent_output, re.DOTALL)

    if not match:
        return {"STATUS": "UNKNOWN", "error": "요약 블록을 찾을 수 없음"}

    summary_text = match.group(1)
    result = {}

    for line in summary_text.strip().split('\n'):
        if ': ' in line:
            key, value = line.split(': ', 1)
            result[key.strip()] = value.strip()

    return result

# 사용 예시
explorer_result = Agent(prompt="탐색 프롬프트...")
summary = parse_agent_summary(explorer_result)

# 요약만 다음 에이전트에게 전달 (전체 탐색 내용 대신)
executor_prompt = f"""
탐색 결과 요약:
- 상태: {summary['STATUS']}
- 핵심 발견: {summary['KEY_FINDINGS']}
- 전달 정보: {summary['NEXT_AGENT_CONTEXT']}

위 정보를 바탕으로 수정을 구현하세요.
"""
```

#### 컨텍스트 크기 비교

```
방식 A (전체 전달):
  탐색 에이전트 전체 출력: ~8,000 토큰
  + 구현 에이전트 전체 출력: ~6,000 토큰
  메인 에이전트 컨텍스트: ~14,000 토큰 소비

방식 B (요약만 전달):
  탐색 에이전트 요약: ~200 토큰
  + 구현 에이전트 요약: ~150 토큰
  메인 에이전트 컨텍스트: ~350 토큰 소비

절약: 약 97.5%
```

#### 파일 기반 결과 전달 (대용량 결과 처리)

```bash
# 서브에이전트가 결과를 파일에 저장하도록 지시
AGENT_PROMPT="탐색 완료 후 결과를 /tmp/explorer-result.json에 저장하세요."

# 메인 에이전트는 파일의 요약 섹션만 읽음
cat /tmp/explorer-result.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
# 요약 필드만 출력
print(json.dumps({
    'status': data['status'],
    'bug_location': data['bug_location'],
    'affected_files': data['affected_files'][:5],  # 최대 5개만
    'recommendation': data['recommendation']
}, ensure_ascii=False, indent=2))
"
```

---

### 3.2 run_in_background 활용법

`run_in_background`를 사용하면 서브에이전트를 병렬로 실행하여 전체 작업 시간을 단축할 수 있다.

#### 병렬 실행이 유효한 패턴

```
직렬 실행 (느림):
  탐색 (30초) → 구현 (60초) → 검증 (20초) = 110초

병렬 실행 (빠름):
  탐색(프론트엔드 30초) ─┐
  탐색(백엔드 30초)      ─┤→ 구현(합쳐서 60초) → 검증 = 90초
  탐색(DB 30초)          ─┘
```

#### Bash 도구에서 run_in_background 사용

```python
# Claude Code에서 병렬 작업 실행 예시

# 1. 두 탐색 에이전트를 동시에 시작 (백그라운드)
frontend_explore = Bash(
    command="claude-agent --prompt '프론트엔드 auth 컴포넌트 탐색' > /tmp/fe-result.txt",
    run_in_background=True
)

backend_explore = Bash(
    command="claude-agent --prompt '백엔드 auth API 탐색' > /tmp/be-result.txt",
    run_in_background=True
)

# 2. 두 결과가 모두 완료되면 합쳐서 구현 에이전트에게 전달
# (백그라운드 작업 완료 알림을 받으면 진행)
```

#### 실제 병렬 테스트 실행 패턴

```bash
#!/usr/bin/env bash
# parallel-verify.sh - 여러 테스트 스위트를 병렬로 실행한다.

# 각 테스트를 백그라운드로 실행하고 PID를 기록한다.
npm test -- --testPathPattern=auth > /tmp/test-auth.log 2>&1 &
AUTH_PID=$!

npm test -- --testPathPattern=user > /tmp/test-user.log 2>&1 &
USER_PID=$!

npm test -- --testPathPattern=payment > /tmp/test-payment.log 2>&1 &
PAYMENT_PID=$!

echo "테스트 실행 중... (PIDs: $AUTH_PID, $USER_PID, $PAYMENT_PID)"

# 모든 테스트가 완료될 때까지 대기한다.
wait $AUTH_PID
AUTH_STATUS=$?

wait $USER_PID
USER_STATUS=$?

wait $PAYMENT_PID
PAYMENT_STATUS=$?

# 결과를 합쳐서 보고한다.
echo "=== 병렬 테스트 결과 ==="
echo "Auth 테스트: $([ $AUTH_STATUS -eq 0 ] && echo PASS || echo FAIL)"
echo "User 테스트: $([ $USER_STATUS -eq 0 ] && echo PASS || echo FAIL)"
echo "Payment 테스트: $([ $PAYMENT_STATUS -eq 0 ] && echo PASS || echo FAIL)"

# 실패한 테스트 로그만 출력한다.
if [ $AUTH_STATUS -ne 0 ]; then
  echo "--- Auth 실패 로그 ---"
  cat /tmp/test-auth.log
fi
```

#### Claude Code 프롬프트에서 병렬 실행 지시

```
다음 두 작업을 동시에(병렬로) 실행해줘:

작업 1: src/auth/ 디렉토리의 모든 TypeScript 파일을 분석하고
       함수별 복잡도를 /tmp/auth-complexity.txt에 저장

작업 2: tests/auth/ 디렉토리의 테스트 커버리지를 분석하고
       미테스트 함수 목록을 /tmp/coverage-gaps.txt에 저장

두 작업이 모두 완료되면 결과를 합쳐서 우선순위가 높은 개선 항목 상위 5개를 알려줘.
```

---

## 4. 실전 워크플로우: 버그 수정 시 3개 서브에이전트 협업

실제 버그 수정 시나리오를 통해 탐색 → 구현 → 검증 에이전트가 어떻게 협력하는지 보여준다.

### 시나리오

**버그**: 사용자가 로그아웃 후 브라우저 뒤로 가기를 누르면 여전히 대시보드에 접근할 수 있다.

**이슈**: GitHub Issue #456 - "로그아웃 후 뒤로 가기 버튼으로 대시보드 접근 가능"

### 전체 워크플로우 코드

```python
#!/usr/bin/env python3
"""
bug-fix-workflow.py
3개 서브에이전트를 사용한 버그 수정 워크플로우 오케스트레이터
"""

import subprocess
import json
from pathlib import Path

def run_agent(role: str, prompt: str, allowed_tools: list) -> str:
    """서브에이전트를 실행하고 결과를 반환한다."""
    print(f"\n{'='*50}")
    print(f"[{role.upper()} 에이전트] 시작")
    print(f"{'='*50}")

    # 실제로는 Claude Code Agent 도구를 통해 실행
    # 여기서는 예시 구조를 보여줌
    result = f"[{role} 에이전트 실행 결과]"

    print(f"[{role.upper()} 에이전트] 완료")
    return result

def extract_summary(agent_output: str) -> dict:
    """에이전트 출력에서 요약을 추출한다."""
    # AGENT_RESULT_SUMMARY 블록 파싱
    lines = agent_output.split('\n')
    summary = {}
    in_summary = False

    for line in lines:
        if '=== AGENT_RESULT_SUMMARY ===' in line:
            in_summary = True
            continue
        if '=== END_SUMMARY ===' in line:
            break
        if in_summary and ': ' in line:
            key, value = line.split(': ', 1)
            summary[key.strip()] = value.strip()

    return summary


# ============================================================
# 1단계: 탐색 에이전트
# ============================================================
EXPLORER_PROMPT = """
당신은 버그 탐색 전문가입니다. 다음 버그를 분석하세요.

버그 설명:
사용자가 로그아웃 후 브라우저 뒤로 가기를 누르면 여전히 대시보드에 접근 가능.

탐색 범위:
- src/middleware/auth.middleware.ts
- src/pages/dashboard/
- src/auth/session.service.ts
- src/auth/logout.handler.ts

탐색 규칙:
- 파일을 읽고 분석만 한다. 수정하지 않는다.
- HTTP 캐시 헤더 설정을 특히 확인한다.
- 세션 무효화 로직을 확인한다.
- 클라이언트 사이드 라우팅 가드를 확인한다.

=== AGENT_RESULT_SUMMARY ===
STATUS: [SUCCESS|PARTIAL|FAILED]
MODIFIED_FILES: NONE
KEY_FINDINGS: [핵심 발견 3줄]
NEXT_AGENT_CONTEXT: [구현 에이전트에게 전달할 정보]
BLOCKERS: [없으면 NONE]
=== END_SUMMARY ===
"""

# 탐색 에이전트 실행 (실제 Claude Code 환경에서)
explorer_output = """
버그를 분석한 결과:

1. src/auth/logout.handler.ts:
   - 세션은 올바르게 무효화됨
   - 그러나 HTTP 응답에 Cache-Control 헤더가 없음

2. src/pages/dashboard/index.tsx:
   - getServerSideProps에서 세션 검사를 하지만
   - 브라우저가 캐시된 페이지를 반환할 때는 서버에 요청하지 않음

3. src/middleware/auth.middleware.ts:
   - 인증된 페이지에 Cache-Control: no-store 헤더가 없음

근본 원인: 대시보드 페이지 응답에 캐시 방지 헤더가 없어
브라우저가 이전 페이지를 캐시하여 재사용함.

=== AGENT_RESULT_SUMMARY ===
STATUS: SUCCESS
MODIFIED_FILES: NONE
KEY_FINDINGS: 대시보드 응답에 Cache-Control 헤더 없음. auth.middleware.ts에 추가 필요. dashboard/index.tsx의 getServerSideProps에도 헤더 추가 필요.
NEXT_AGENT_CONTEXT: auth.middleware.ts에 protected routes용 no-cache 헤더 추가. dashboard/index.tsx getServerSideProps에 res.setHeader('Cache-Control', 'no-store') 추가.
BLOCKERS: NONE
=== END_SUMMARY ===
"""

explorer_summary = extract_summary(explorer_output)
print("탐색 완료:", explorer_summary)


# ============================================================
# 2단계: 구현 에이전트
# ============================================================
EXECUTOR_PROMPT = f"""
당신은 정밀한 코드 구현 전문가입니다.

탐색 결과:
- 핵심 발견: {explorer_summary.get('KEY_FINDINGS', '')}
- 구현 정보: {explorer_summary.get('NEXT_AGENT_CONTEXT', '')}

구현 지시:
1. src/middleware/auth.middleware.ts에 protected routes용 Cache-Control 헤더를 추가한다.
   추가할 헤더:
   - Cache-Control: no-store, no-cache, must-revalidate
   - Pragma: no-cache
   - Expires: 0

2. src/pages/dashboard/index.tsx의 getServerSideProps에 헤더를 추가한다.

구현 규칙:
- 명시된 2개 파일만 수정한다.
- 기존 코드 스타일을 유지한다.
- 각 변경에 이유를 주석으로 남긴다.

=== AGENT_RESULT_SUMMARY ===
STATUS: [SUCCESS|PARTIAL|FAILED]
MODIFIED_FILES: [수정된 파일 목록]
KEY_FINDINGS: [구현 내용 요약]
NEXT_AGENT_CONTEXT: [검증 에이전트에게 전달할 테스트 시나리오]
BLOCKERS: [없으면 NONE]
=== END_SUMMARY ===
"""

# 구현 에이전트 실행 (실제 환경에서)
executor_output = """
다음 두 파일을 수정했습니다:

src/middleware/auth.middleware.ts 수정:
```typescript
// 인증이 필요한 페이지는 브라우저 캐시를 방지한다.
// 관련 버그: GitHub #456 - 로그아웃 후 뒤로 가기로 접근 가능
if (isProtectedRoute(req.path)) {
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate');
  res.setHeader('Pragma', 'no-cache');
  res.setHeader('Expires', '0');
}
```

src/pages/dashboard/index.tsx 수정:
```typescript
export async function getServerSideProps(context: GetServerSidePropsContext) {
  // 로그아웃 후 브라우저 뒤로 가기 버튼으로 접근 방지
  context.res.setHeader('Cache-Control', 'no-store');

  const session = await getSession(context);
  if (!session) {
    return { redirect: { destination: '/login', permanent: false } };
  }
  // ... 기존 코드
}
```

=== AGENT_RESULT_SUMMARY ===
STATUS: SUCCESS
MODIFIED_FILES: src/middleware/auth.middleware.ts, src/pages/dashboard/index.tsx
KEY_FINDINGS: 두 파일에 Cache-Control no-store 헤더 추가 완료. 기존 테스트 구조 유지.
NEXT_AGENT_CONTEXT: 테스트 시나리오: 1)로그인→대시보드→로그아웃→뒤로가기 시 로그인 페이지 리다이렉트, 2)직접 URL 접근 시 리다이렉트, 3)유효 세션 시 대시보드 정상 접근
BLOCKERS: NONE
=== END_SUMMARY ===
"""

executor_summary = extract_summary(executor_output)
print("구현 완료:", executor_summary)


# ============================================================
# 3단계: 검증 에이전트
# ============================================================
VERIFIER_PROMPT = f"""
당신은 독립적인 검증 전문가입니다.

구현 결과:
- 수정 파일: {executor_summary.get('MODIFIED_FILES', '')}
- 핵심 변경: {executor_summary.get('KEY_FINDINGS', '')}
- 검증 시나리오: {executor_summary.get('NEXT_AGENT_CONTEXT', '')}

검증 지시:
1. 수정된 두 파일을 직접 읽어 변경이 올바른지 확인한다.
2. npm test -- --testPathPattern=auth|dashboard 를 실행한다.
3. 구현 에이전트가 제시한 3가지 시나리오를 모두 검증한다.
4. HTTP 응답 헤더를 curl로 직접 확인한다:
   curl -I http://localhost:3000/dashboard

판정 기준:
- PASS: 모든 테스트 통과 + Cache-Control 헤더 확인
- PARTIAL: 일부 테스트 실패 또는 헤더 누락
- FAIL: 핵심 버그가 수정되지 않음

=== AGENT_RESULT_SUMMARY ===
STATUS: [PASS|PARTIAL|FAIL]
MODIFIED_FILES: NONE
KEY_FINDINGS: [검증 결과 요약]
NEXT_AGENT_CONTEXT: [후속 조치 권장사항]
BLOCKERS: [없으면 NONE]
=== END_SUMMARY ===
"""

# 검증 에이전트 실행 (실제 환경에서)
verifier_output = """
## 검증 결과

### 파일 검증
src/middleware/auth.middleware.ts - Cache-Control 헤더 추가 확인 ✓
src/pages/dashboard/index.tsx - getServerSideProps 헤더 추가 확인 ✓

### 테스트 실행 결과
```
PASS tests/auth/auth.middleware.spec.ts
  AuthMiddleware
    ✓ protected routes에 Cache-Control 헤더 추가 (18ms)
    ✓ public routes는 헤더 없음 (5ms)

PASS tests/pages/dashboard.spec.ts
  Dashboard
    ✓ 세션 없을 때 로그인 리다이렉트 (22ms)
    ✓ 유효 세션으로 접근 성공 (15ms)

Tests: 4 passed, 4 total
```

### HTTP 헤더 확인
```
HTTP/1.1 200 OK
Cache-Control: no-store, no-cache, must-revalidate
Pragma: no-cache
Expires: 0
```

=== AGENT_RESULT_SUMMARY ===
STATUS: PASS
MODIFIED_FILES: NONE
KEY_FINDINGS: 모든 테스트 통과. Cache-Control 헤더 올바르게 설정됨. 브라우저 캐시 버그 수정 확인.
NEXT_AGENT_CONTEXT: CHANGELOG 업데이트 및 PR 생성 권장. QA에서 실제 브라우저로 뒤로 가기 시나리오 수동 검증 권장.
BLOCKERS: NONE
=== END_SUMMARY ===
"""

verifier_summary = extract_summary(verifier_output)

# 최종 결과 보고
print("\n" + "="*60)
print("버그 수정 워크플로우 완료")
print("="*60)
print(f"탐색 상태: {explorer_summary.get('STATUS')}")
print(f"구현 상태: {executor_summary.get('STATUS')}")
print(f"검증 판정: {verifier_summary.get('STATUS')}")
print(f"수정 파일: {executor_summary.get('MODIFIED_FILES')}")
print(f"후속 조치: {verifier_summary.get('NEXT_AGENT_CONTEXT')}")
```

### Claude Code 대화 형식 워크플로우 예시

위 워크플로우를 Claude Code 대화 인터페이스에서 실행하는 실제 프롬프트 시퀀스:

```
# 사용자 → Claude (메인 에이전트)
GitHub Issue #456을 해결해줘:
로그아웃 후 브라우저 뒤로 가기를 누르면 대시보드에 접근 가능한 버그.

탐색 에이전트, 구현 에이전트, 검증 에이전트 순서로 처리해줘.
각 에이전트 결과를 요약해서 다음 에이전트에게 전달하는 방식으로.
```

```
# Claude (메인 에이전트) 응답
3단계 서브에이전트 워크플로우를 시작합니다.

[1/3] 탐색 에이전트 시작...
Agent 도구를 사용하여 코드베이스를 분석하겠습니다.

[탐색 에이전트 실행 중...]
→ 근본 원인 발견: Cache-Control 헤더 누락

[2/3] 구현 에이전트 시작...
탐색 결과를 바탕으로 수정을 구현합니다.

[구현 에이전트 실행 중...]
→ 2개 파일 수정 완료

[3/3] 검증 에이전트 시작...
독립적으로 구현 결과를 검증합니다.

[검증 에이전트 실행 중...]
→ 판정: PASS (4/4 테스트 통과)

워크플로우 완료:
- 수정 파일: src/middleware/auth.middleware.ts, src/pages/dashboard/index.tsx
- 검증 결과: 모든 테스트 통과
- 권장 사항: PR 생성 후 QA 수동 검증 요청
```

---

## 요약

| 에이전트 | 역할 | 허용 도구 | 출력 형식 |
|---------|------|---------|---------|
| Explorer | 읽기 전용 코드 분석 | Read, Glob, Grep | 구조화된 탐색 보고서 |
| Executor | 최소 변경 구현 | Read, Edit, Write, Bash | 변경 내용 + 검증 시나리오 |
| Verifier | 독립 검증 및 판정 | Read, Bash, Grep | PASS/PARTIAL/FAIL 판정 |

**서브에이전트 분리의 핵심 이점**:
1. 각 에이전트의 컨텍스트가 깨끗하게 유지됨 (역할에 집중)
2. 요약만 메인 컨텍스트로 전달하여 컨텍스트 창 절약 (~97%)
3. 탐색/구현 실패가 서로 격리됨
4. 병렬 실행으로 전체 작업 시간 단축 가능
5. 검증 에이전트의 독립성으로 구현 편향 없는 검토 가능
