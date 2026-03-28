# 테스트와 검증 루프 만들기

> 03-CH04: "된 것 같다"는 추측 버리기 — 정확한 완료 기준과 자동 검증 루프

## 목차

- ["된 것 같다"를 버리는 이유](#된-것-같다를-버리는-이유)
- [정확한 완료 기준 세우기 (TDD 원칙)](#정확한-완료-기준-세우기-tdd-원칙)
- [검증 루프 구성: 테스트 + 린트 + 타입체크](#검증-루프-구성-테스트--린트--타입체크)
- [Claude Code에서 TDD 워크플로우](#claude-code에서-tdd-워크플로우)
- [Codex에서 자동 검증 설정](#codex에서-자동-검증-설정)
- [CLAUDE.md에 검증 규칙 넣기](#claudemd에-검증-규칙-넣기)
- [AI가 틀렸을 때 디버깅 질문 패턴](#ai가-틀렸을-때-디버깅-질문-패턴)
- [hooks로 자동 테스트 실행 설정](#hooks로-자동-테스트-실행-설정)

---

## "된 것 같다"를 버리는 이유

AI가 코드를 작성하면 자연스럽게 "잘 됐겠지"라고 넘어가고 싶어진다. 하지만 이것이 AI 활용에서 가장 흔한 실패 패턴이다.

**전형적인 실패 시나리오:**

```
사용자: parseDate 함수가 UTC 타임존도 처리하도록 수정해줘.
AI: 네, 수정했습니다. (코드 제시)
사용자: 고마워. (PR 병합)
... 이틀 후 ...
QA: UTC+9 환경에서 날짜가 하루씩 밀려요.
```

AI는 논리적으로 옳아 보이는 코드를 작성한다. 하지만 "논리적으로 옳아 보이는 것"과 "실제로 동작하는 것"은 다르다. 그 차이를 메우는 것이 **검증 루프**다.

완료의 정의는 명확해야 한다:

```
완료 = 테스트가 통과한다 + 린트 오류가 없다 + 타입 오류가 없다
```

---

## 정확한 완료 기준 세우기 (TDD 원칙)

TDD(Test-Driven Development)의 핵심은 코드보다 테스트를 먼저 작성하는 것이다. AI 협업에서는 이 원칙이 더욱 중요하다. AI에게 "완료의 증거"를 먼저 정의해주면 AI는 그 증거를 충족시키는 코드를 작성한다.

### AI와 TDD하는 올바른 순서

```
1. 실패하는 테스트 먼저 작성 (사람이 or AI가)
2. AI에게 "이 테스트를 통과시켜줘" 라고 요청
3. 테스트 실행으로 완료 확인
4. 리팩터링 (테스트는 계속 통과해야 함)
```

### 구체적인 완료 기준 예시

모호한 요청:
```
"parseDate가 여러 형식을 지원하도록 해줘."
```

명확한 완료 기준:
```python
# 이 테스트가 모두 통과하면 완료다.

def test_parse_date_formats():
    assert parse_date("2026-03-28") == date(2026, 3, 28)
    assert parse_date("28/03/2026") == date(2026, 3, 28)
    assert parse_date("March 28, 2026") == date(2026, 3, 28)
    assert parse_date("20260328") == date(2026, 3, 28)

def test_parse_date_invalid_raises():
    with pytest.raises(ValueError, match="지원하지 않는 날짜 형식"):
        parse_date("not-a-date")

def test_parse_date_timezone_utc():
    result = parse_datetime("2026-03-28T12:00:00Z")
    assert result.tzinfo == timezone.utc
    assert result.hour == 12
```

---

## 검증 루프 구성: 테스트 + 린트 + 타입체크

세 가지 검증을 하나의 명령으로 묶어서 "검증 게이트"를 만든다.

### Python 프로젝트 검증 스크립트

```bash
#!/bin/bash
# scripts/verify.sh
set -e

echo "=== 타입 체크 ==="
mypy src/ --strict

echo "=== 린트 ==="
ruff check src/ tests/

echo "=== 포맷 ==="
ruff format --check src/ tests/

echo "=== 테스트 ==="
pytest tests/ -v --tb=short

echo "=== 모두 통과 ==="
```

```bash
chmod +x scripts/verify.sh
```

### TypeScript/Node.js 프로젝트 검증 스크립트

```bash
#!/bin/bash
# scripts/verify.sh
set -e

echo "=== 타입 체크 ==="
tsc --noEmit

echo "=== 린트 ==="
eslint src/ --ext .ts,.tsx --max-warnings 0

echo "=== 포맷 체크 ==="
prettier --check "src/**/*.{ts,tsx}"

echo "=== 단위 테스트 ==="
jest --passWithNoTests

echo "=== 모두 통과 ==="
```

### `package.json`에 통합

```json
{
  "scripts": {
    "verify": "bash scripts/verify.sh",
    "test": "jest",
    "test:watch": "jest --watch",
    "typecheck": "tsc --noEmit",
    "lint": "eslint src/ --ext .ts,.tsx",
    "lint:fix": "eslint src/ --ext .ts,.tsx --fix"
  }
}
```

### `Makefile`로 통합 (언어 무관)

```makefile
# Makefile
.PHONY: verify test lint typecheck

verify: typecheck lint test
	@echo "모든 검증 통과"

typecheck:
	tsc --noEmit

lint:
	eslint src/ --ext .ts,.tsx --max-warnings 0

test:
	jest --passWithNoTests

# AI 작업 후 빠른 확인용
quick-check:
	tsc --noEmit && jest --onlyFailures
```

---

## Claude Code에서 TDD 워크플로우

### pytest 기반 Python 예시

**1단계: 실패하는 테스트를 먼저 작성하도록 지시**

```
사용자: src/utils/email.py에 validate_email 함수를 추가해야 해.
        먼저 pytest 테스트만 작성해줘. 구현은 아직 하지 마.

        검증해야 할 케이스:
        - user@example.com → True
        - invalid-email → False
        - user@.com → False
        - @domain.com → False
        - 빈 문자열 → False
        - 255자 초과 → False
```

Claude가 작성한 테스트:

```python
# tests/utils/test_email.py
import pytest
from src.utils.email import validate_email

class TestValidateEmail:
    def test_valid_email(self):
        assert validate_email("user@example.com") is True

    def test_missing_at_sign(self):
        assert validate_email("invalid-email") is False

    def test_empty_local_part(self):
        assert validate_email("@domain.com") is False

    def test_empty_domain(self):
        assert validate_email("user@.com") is False

    def test_empty_string(self):
        assert validate_email("") is False

    def test_too_long(self):
        long_email = "a" * 250 + "@example.com"
        assert validate_email(long_email) is False

    def test_subdomains(self):
        assert validate_email("user@mail.example.co.kr") is True
```

**2단계: 테스트가 실패함을 확인**

```bash
pytest tests/utils/test_email.py -v
# ImportError: cannot import name 'validate_email' from 'src.utils.email'
# 예상된 실패 — 아직 구현이 없으니 정상
```

**3단계: 구현 요청**

```
사용자: 이제 이 테스트를 모두 통과시키는 validate_email 구현해줘.
        외부 라이브러리 없이 순수 Python만 써.
```

**4단계: 테스트로 완료 확인**

```bash
pytest tests/utils/test_email.py -v
# 7 passed in 0.12s
```

### Jest 기반 TypeScript 예시

```
사용자: src/lib/currency.ts에 formatCurrency 함수 TDD로 만들어줘.

        조건:
        - 숫자를 한국 원화 형식으로 포맷 (예: 1500000 → "₩1,500,000")
        - 음수 지원 (예: -500 → "-₩500")
        - 소수점 입력 시 반올림 (예: 1500.7 → "₩1,501")
        - 0 처리 (예: 0 → "₩0")
```

Claude가 작성하는 테스트 먼저:

```typescript
// src/lib/__tests__/currency.test.ts
import { formatCurrency } from "../currency";

describe("formatCurrency", () => {
  it("정수를 원화 형식으로 포맷한다", () => {
    expect(formatCurrency(1500000)).toBe("₩1,500,000");
  });

  it("음수를 처리한다", () => {
    expect(formatCurrency(-500)).toBe("-₩500");
  });

  it("소수점은 반올림한다", () => {
    expect(formatCurrency(1500.7)).toBe("₩1,501");
    expect(formatCurrency(1500.4)).toBe("₩1,500");
  });

  it("0을 처리한다", () => {
    expect(formatCurrency(0)).toBe("₩0");
  });

  it("큰 숫자에 천 단위 쉼표를 넣는다", () => {
    expect(formatCurrency(1000000000)).toBe("₩1,000,000,000");
  });
});
```

테스트 실행 → 실패 확인 → 구현 요청 → 테스트 통과 확인.

---

## Codex에서 자동 검증 설정

Codex 최신 실습에서는 `codex.md`보다 `AGENTS.md` 중심으로 검증 규칙을 두는 편이 안전하다.

### `AGENTS.md` 검증 규칙 설정

```markdown
# AGENTS.md

## 검증 게이트 (필수)

모든 코드 변경 후 반드시 다음 순서로 검증을 실행한다:

1. 타입 체크: `npm run typecheck` (오류 0개여야 함)
2. 린트: `npm run lint` (경고 포함 0개여야 함)
3. 테스트: `npm test` (모든 테스트 통과해야 함)

검증이 실패하면 코드를 수정하고 다시 검증한다. 검증 통과 전에는 완료로 보고하지 않는다.

## TDD 원칙

- 새 기능 추가 시: 테스트 먼저, 구현 나중
- 버그 수정 시: 버그를 재현하는 테스트 먼저, 수정 나중
- 리팩터링 시: 기존 테스트가 통과하는지 확인하면서 진행

## 금지 사항

- 테스트를 삭제하거나 skip 처리하는 것
- `// @ts-ignore` 또는 `// @ts-expect-error` 임시 사용
- `any` 타입 남용 (부득이한 경우 주석 필수)
```

### Codex CLI 실행 시 검증 포함

```bash
# 비대화형 실행 + 자동 검증 포함
codex exec --full-auto "
  src/api/users.ts에 deleteUser 함수를 추가해줘.

  완료 기준:
  1. tests/api/users.test.ts의 deleteUser 관련 테스트 모두 통과
  2. tsc --noEmit 오류 없음
  3. eslint 경고 없음

  검증 명령: npm run verify
"
```

필요하면 승인 정책과 샌드박스를 직접 지정할 수도 있다:

```bash
codex exec -a never -s workspace-write "
  src/api/users.ts에 deleteUser 함수를 추가하고
  완료 후 npm run verify를 실행해줘.
"
```

---

## CLAUDE.md에 검증 규칙 넣기

`CLAUDE.md`는 Claude Code가 프로젝트를 시작할 때 읽는 설정 파일이다. 여기에 검증 규칙을 명시하면 매번 지시하지 않아도 된다.

### 실제 `CLAUDE.md` 검증 섹션 예시

```markdown
# CLAUDE.md

## 프로젝트: MyApp API

### 기술 스택
- Runtime: Node.js 20 + TypeScript 5
- Framework: Express 4
- Test: Jest + Supertest
- Lint: ESLint + Prettier
- Type check: tsc

### 검증 게이트 (MANDATORY)

코드를 수정한 후에는 반드시 다음을 실행하고 모두 통과해야 한다:

```bash
npm run verify
# 내부적으로 다음을 순서대로 실행:
# 1. tsc --noEmit
# 2. eslint src/ --max-warnings 0
# 3. jest --passWithNoTests
```

**예외 없음.** 검증이 실패하면 코드를 수정하고 재실행한다.

### TDD 워크플로우

1. 새 기능: 테스트 파일에 실패 케이스 먼저 작성 → 구현 → 검증
2. 버그 수정: 버그 재현 테스트 작성 → 수정 → 검증
3. 리팩터링: 테스트 먼저 실행하여 현재 상태 확인 → 수정 → 재검증

### 테스트 파일 위치 규칙

- 단위 테스트: `src/**/__tests__/*.test.ts`
- 통합 테스트: `tests/integration/*.test.ts`
- E2E: `tests/e2e/*.test.ts`

### 완료의 정의

- [ ] 요청한 기능이 동작한다
- [ ] 관련 테스트가 추가되었다
- [ ] `npm run verify`가 통과한다
- [ ] 기존 테스트가 깨지지 않는다
```

---

## AI가 틀렸을 때 디버깅 질문 패턴

AI가 작성한 코드가 테스트를 통과하지 못할 때, 막연하게 "고쳐줘"라고 하면 AI는 같은 실수를 반복하거나 다른 방향으로 잘못 간다. 정확한 컨텍스트를 제공하는 것이 핵심이다.

### 패턴 1: 실패 출력을 그대로 붙여넣기

```
다음 테스트가 실패해:

=== FAILURES ===
FAILED tests/utils/test_email.py::TestValidateEmail::test_subdomains
AssertionError: assert False is True

validate_email("user@mail.example.co.kr")이 False를 반환하고 있어.
현재 구현의 어떤 부분이 서브도메인을 잘못 처리하는지 찾아서 고쳐줘.
수정 후 반드시 pytest tests/utils/test_email.py -v 실행해서 확인해.
```

### 패턴 2: 예상값과 실제값을 명시하기

```
formatCurrency(-500)의 결과가 다음과 같이 다르게 나와:

예상: "-₩500"
실제: "₩-500"

음수 기호의 위치가 문제야. 현재 구현 코드를 보고, 음수 처리 로직을 찾아서
기호가 ₩ 앞에 오도록 수정해줘. 수정 후 jest currency.test.ts 실행해서 확인해.
```

### 패턴 3: 타입 오류를 그대로 제공하기

```
다음 TypeScript 오류가 발생해:

src/api/users.ts:45:18 - error TS2345:
Argument of type 'string | undefined' is not assignable to
parameter of type 'string'.
  Type 'undefined' is not assignable to type 'string'.

45    const user = await findUser(req.params.id);

req.params.id가 undefined일 수 있는데 findUser는 string만 받아.
null/undefined 처리를 추가해서 이 오류를 고쳐줘.
단, findUser 함수 시그니처는 바꾸지 마.
```

### 패턴 4: 근본 원인 찾기를 요청하기

```
deleteUser API가 처음에는 동작하는데 두 번째 호출부터 404를 반환해.

테스트:
it("여러 번 삭제 요청해도 첫 번째만 204, 나머지는 404", async () => {
  const res1 = await request(app).delete("/users/1");
  expect(res1.status).toBe(204); // 통과

  const res2 = await request(app).delete("/users/1");
  expect(res2.status).toBe(404); // 통과 (이미 삭제됨, 정상)
});

근데 실제로는 첫 번째도 404가 나와. 다음을 순서대로 해줘:
1. deleteUser 핸들러에서 DB 쿼리가 어떻게 동작하는지 설명해줘
2. 왜 첫 번째 요청도 404를 반환하는지 원인 찾아줘
3. 수정하고 테스트 돌려서 확인해줘
```

### 패턴 5: 이전 시도가 실패했음을 알리기

```
방금 네가 수정한 parseDate가 UTC 테스트에서 여전히 실패해:

FAILED test_parse_date_timezone_utc
AssertionError: assert datetime(2026, 3, 28, 12, 0, tzinfo=<UTC>) ==
               datetime(2026, 3, 28, 3, 0, tzinfo=<UTC>)

네가 timezone.utc를 붙이는 방식이 잘못됐어.
입력 문자열이 "Z"로 끝나면 UTC+0을 의미하는데,
현재 코드가 로컬 타임존으로 파싱한 뒤 UTC로 변환하고 있어.

python-dateutil의 parser.parse를 쓰지 말고,
datetime.fromisoformat() 또는 직접 파싱해서 처리해줘.
수정 후 pytest tests/utils/test_date.py::test_parse_date_timezone_utc -v 실행해.
```

### 패턴 6: 경계 조건 실패를 구체적으로 명시하기

```
validate_password가 대부분 통과하는데 딱 하나가 실패해:

FAILED: test_password_exactly_8_chars
입력: "Ab1!defg" (정확히 8자)
예상: True (8자 이상이면 유효)
실제: False

현재 구현이 >= 8이 아니라 > 8을 쓰는 것 같아.
경계값 조건을 확인하고 수정해줘.
min_length=8이면 8자는 유효해야 해.
```

### 패턴 7: 여러 테스트가 같은 이유로 실패할 때

```
다음 5개 테스트가 모두 같은 오류로 실패해:

TypeError: Cannot read properties of undefined (reading 'id')
  at formatUser (src/utils/user.ts:12)

실패한 테스트들:
- test_format_user_basic
- test_format_user_with_avatar
- test_format_user_admin
- test_format_user_guest
- test_format_user_inactive

formatUser가 user.id에 접근하는데 user가 undefined인 경우를 처리 안 하는 것 같아.
방어 코드를 추가하되, user가 null/undefined면 null을 반환하도록 해줘.
타입도 User | null | undefined를 받도록 수정해.
```

---

## hooks로 자동 테스트 실행 설정

Claude Code의 hooks 기능을 사용하면 특정 이벤트 발생 시 자동으로 검증을 실행할 수 있다.

### `~/.claude/settings.json` hooks 설정

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "bash -c 'cd \"$CLAUDE_PROJECT_DIR\" && npm run typecheck 2>&1 | tail -5'",
            "timeout": 30000
          }
        ]
      }
    ]
  }
}
```

이 설정은 Claude Code가 파일을 편집할 때마다 타입 체크를 자동 실행한다.

### 프로젝트별 hooks 설정 (`.claude/settings.json`)

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "bash /Users/yourname/myproject/scripts/on-file-change.sh",
            "timeout": 60000
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash /Users/yourname/myproject/scripts/verify.sh",
            "timeout": 120000,
            "blocking": true
          }
        ]
      }
    ]
  }
}
```

### 자동 실행 스크립트: `scripts/on-file-change.sh`

```bash
#!/bin/bash
# 파일 변경 시 빠른 검증 (전체 테스트 대신 빠른 체크)
set -e

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
CHANGED_FILE="${CLAUDE_TOOL_INPUT_FILE_PATH:-}"

echo "변경된 파일: $CHANGED_FILE"

# TypeScript 타입 체크 (빠름)
cd "$PROJECT_DIR"
tsc --noEmit 2>&1 | head -20

# 변경된 파일과 관련된 테스트만 실행
if [[ "$CHANGED_FILE" == *".ts" ]]; then
  # 파일명에서 테스트 패턴 추출
  BASE=$(basename "$CHANGED_FILE" .ts)
  TEST_PATTERN="${BASE}.test.ts"

  if jest --testPathPattern="$TEST_PATTERN" --passWithNoTests 2>&1; then
    echo "관련 테스트 통과"
  else
    echo "테스트 실패 — 수정 필요"
    exit 1
  fi
fi
```

### Stop hook: 대화 종료 전 전체 검증

```bash
#!/bin/bash
# scripts/verify.sh — Claude Code가 Stop 이벤트에서 실행
set -e

cd "${CLAUDE_PROJECT_DIR:-$(pwd)}"

echo "=== 최종 검증 시작 ==="

# 1. 타입 체크
echo "1/3 타입 체크..."
if ! tsc --noEmit 2>&1; then
  echo "FAIL: 타입 오류가 있습니다."
  exit 1
fi

# 2. 린트
echo "2/3 린트..."
if ! eslint src/ --ext .ts,.tsx --max-warnings 0 2>&1; then
  echo "FAIL: 린트 오류가 있습니다."
  exit 1
fi

# 3. 테스트
echo "3/3 테스트..."
if ! jest --passWithNoTests 2>&1; then
  echo "FAIL: 테스트가 실패했습니다."
  exit 1
fi

echo "=== 모든 검증 통과 ==="
```

### Pre-commit hook으로 추가 보호

```bash
#!/bin/bash
# .git/hooks/pre-commit
set -e

echo "커밋 전 검증..."

npm run typecheck || { echo "타입 오류 수정 후 커밋하세요"; exit 1; }
npm run lint || { echo "린트 오류 수정 후 커밋하세요"; exit 1; }
npm test -- --passWithNoTests || { echo "테스트 통과 후 커밋하세요"; exit 1; }

echo "검증 완료"
```

```bash
chmod +x .git/hooks/pre-commit
```

### GitHub Actions와 연동

```yaml
# .github/workflows/verify.yml
name: 검증

on: [push, pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"
      - run: npm ci
      - name: 타입 체크
        run: npm run typecheck
      - name: 린트
        run: npm run lint
      - name: 테스트
        run: npm test -- --coverage
      - name: 커버리지 리포트
        uses: codecov/codecov-action@v4
```

---

## 정리: 검증 루프 체크리스트

AI와 작업할 때마다 이 순서를 지킨다:

```
[ ] 완료 기준을 테스트로 먼저 정의했는가?
[ ] AI가 변경한 후 npm run verify (또는 동등한 명령)를 실행했는가?
[ ] 테스트가 모두 통과하는가?
[ ] 기존 테스트가 깨지지 않았는가?
[ ] 타입 오류가 없는가?
[ ] 린트 경고가 없는가?
```

이 체크리스트를 `CLAUDE.md`에 넣어두면 AI도 참조하면서 작업한다.

**핵심 원칙: AI의 말을 믿지 말고, 테스트 결과를 믿어라.** "됐습니다"는 정보가 아니다. `7 passed in 0.23s`가 정보다.
