# 나만의 Agent Harness 시작하기

> 커리큘럼 05-CH03 대응 문서

## 목차

- [하네스가 왜 모델보다 더 중요한가](#하네스가-왜-모델보다-더-중요한가)
- [최소 하네스 구성](#최소-하네스-구성)
- [SDD로 Spec.md 작성하기](#sdd로-specmd-작성하기)
- [작업 흐름을 문서·훅·스킬·평가로 묶기](#작업-흐름을-문서훅스킬평가로-묶기)
- [하네스 자기개선 루프](#하네스-자기개선-루프)

---

## 하네스가 왜 모델보다 더 중요한가

### 모델은 바뀌어도 하네스는 남는다

2023년 GPT-4 → Claude 3 → Claude 4 → Gemini Ultra 순으로 "최고 모델"이 교체되는 데 걸린 시간은 평균 6개월이다. 반면 잘 설계된 하네스는 몇 년간 사용된다.

```
시간 축
────────────────────────────────────────────►
GPT-4    Claude-3    Claude-4    Gemini-2    ...
  │           │           │           │
  └───────────┴───────────┴───────────┘
                하네스 (불변)
  ┌─────────────────────────────────────────┐
  │ CLAUDE.md + Hooks + Skills + Specs      │
  │ verify.sh + 평가 기준                    │
  └─────────────────────────────────────────┘
```

### 하네스가 없으면 생기는 문제

| 문제 | 증상 | 하네스 해결책 |
|------|------|---------------|
| 재현 불가 | 같은 프롬프트로 다른 결과 | Spec.md + 평가 스크립트 |
| 컨텍스트 소실 | 매 세션마다 재설명 필요 | CLAUDE.md + notepad |
| 검증 부재 | 에이전트 출력 신뢰 불가 | verify.sh 자동 실행 |
| 패턴 미축적 | 좋은 작업 방식이 사라짐 | Skills + Hooks |
| 에이전트 교체 비용 | 모델 변경 시 재작업 | 추상화된 인터페이스 |

### 하네스의 정의

```
하네스 = 모델에게 일관된 환경을 제공하는 모든 것의 합

구성 요소:
  1. 컨텍스트 문서 (CLAUDE.md, Spec.md, Design.md)
  2. 자동화 훅 (Pre/Post 도구 실행 훅)
  3. 재사용 스킬 (/slash-commands)
  4. 완료 검증 스크립트 (verify.sh)
  5. 평가 기준 (Eval 스위트)
```

---

## 최소 하네스 구성

### 최소 하네스 디렉토리 구조

```
my-project/
├── CLAUDE.md                 # 에이전트에게 보여주는 프로젝트 컨텍스트
├── .claude/
│   ├── settings.json         # 훅, MCP 서버, 권한 설정
│   └── commands/             # 커스텀 슬래시 명령
│       ├── spec.md           # /spec 명령 정의
│       ├── verify.md         # /verify 명령 정의
│       └── deploy.md         # /deploy 명령 정의
├── .omc/
│   ├── plans/                # 계획 문서 (읽기 전용)
│   ├── specs/                # Spec.md 파일들
│   │   └── auth-feature.md
│   └── evals/                # 평가 스위트
│       └── auth-eval.py
├── scripts/
│   ├── verify.sh             # 완료 기준 검증 스크립트
│   ├── eval.py               # 에이전트 출력 평가
│   └── setup-harness.sh      # 하네스 초기 설정
└── src/                      # 실제 소스 코드
```

### CLAUDE.md 최소 템플릿

```markdown
# 프로젝트명

## 빠른 컨텍스트
- 언어: TypeScript, Node.js 20
- 프레임워크: Express 4, Prisma 5
- 테스트: Vitest, Supertest
- 빌드: `npm run build`
- 테스트: `npm test`
- 린트: `npm run lint`

## 완료 기준
모든 작업은 다음을 충족해야 합니다:
1. `npm run build` 통과
2. `npm test` 통과 (커버리지 80% 이상)
3. `npm run lint` 경고 0개
4. `scripts/verify.sh` 통과

## 코딩 규칙
- 함수형 스타일 선호 (클래스 최소화)
- 모든 외부 API 호출은 타임아웃 설정 필수
- 에러는 반드시 로깅 후 상위로 전파

## 금지 사항
- `console.log`를 프로덕션 코드에 남기지 말 것
- `any` 타입 사용 금지
- 테스트 없이 PR 금지
```

### verify.sh 스크립트 예시

```bash
#!/usr/bin/env bash
# scripts/verify.sh
# 용도: 에이전트 작업 완료 기준 자동 검증
# 실행: bash scripts/verify.sh [feature-name]

set -euo pipefail

FEATURE=${1:-"all"}
PASS=0
FAIL=0
ERRORS=()

# 색상 출력
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; PASS=$((PASS+1)); }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; FAIL=$((FAIL+1)); ERRORS+=("$1"); }
log_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

# ─── 1. 빌드 검증 ────────────────────────────────────────────
log_info "빌드 검증 중..."
if npm run build --silent 2>/dev/null; then
    log_pass "빌드 성공"
else
    log_fail "빌드 실패"
fi

# ─── 2. 테스트 검증 ──────────────────────────────────────────
log_info "테스트 실행 중..."
COVERAGE_OUTPUT=$(npm test -- --coverage --reporter=json 2>/dev/null || echo "TEST_FAILED")

if echo "$COVERAGE_OUTPUT" | grep -q "TEST_FAILED"; then
    log_fail "테스트 실패"
else
    # 커버리지 80% 이상 검증
    COVERAGE=$(echo "$COVERAGE_OUTPUT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
total = data.get('total', {})
lines = total.get('lines', {}).get('pct', 0)
print(f'{lines:.1f}')
" 2>/dev/null || echo "0")

    if (( $(echo "$COVERAGE >= 80" | bc -l) )); then
        log_pass "테스트 통과 (커버리지: ${COVERAGE}%)"
    else
        log_fail "커버리지 미달: ${COVERAGE}% (기준: 80%)"
    fi
fi

# ─── 3. 린트 검증 ────────────────────────────────────────────
log_info "린트 검사 중..."
LINT_WARNINGS=$(npm run lint --silent 2>&1 | grep -c "warning" || echo "0")
LINT_ERRORS=$(npm run lint --silent 2>&1 | grep -c "error" || echo "0")

if [ "$LINT_ERRORS" -eq "0" ] && [ "$LINT_WARNINGS" -eq "0" ]; then
    log_pass "린트 클린"
elif [ "$LINT_ERRORS" -eq "0" ]; then
    log_fail "린트 경고 ${LINT_WARNINGS}개 (기준: 0개)"
else
    log_fail "린트 에러 ${LINT_ERRORS}개"
fi

# ─── 4. 타입 검사 ────────────────────────────────────────────
log_info "TypeScript 타입 검사 중..."
if npx tsc --noEmit --strict 2>/dev/null; then
    log_pass "타입 검사 통과"
else
    log_fail "타입 에러 존재"
fi

# ─── 5. 보안 검사 ────────────────────────────────────────────
log_info "보안 취약점 검사 중..."
AUDIT_HIGH=$(npm audit --json 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
vulns = data.get('metadata', {}).get('vulnerabilities', {})
print(vulns.get('high', 0) + vulns.get('critical', 0))
" 2>/dev/null || echo "0")

if [ "$AUDIT_HIGH" -eq "0" ]; then
    log_pass "보안 취약점 없음"
else
    log_fail "High/Critical 취약점 ${AUDIT_HIGH}개"
fi

# ─── 6. 피처별 통합 테스트 ───────────────────────────────────
if [ "$FEATURE" != "all" ] && [ -f "scripts/integration/${FEATURE}.sh" ]; then
    log_info "${FEATURE} 통합 테스트 실행 중..."
    if bash "scripts/integration/${FEATURE}.sh"; then
        log_pass "${FEATURE} 통합 테스트 통과"
    else
        log_fail "${FEATURE} 통합 테스트 실패"
    fi
fi

# ─── 결과 요약 ───────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════"
echo "검증 결과: PASS=${PASS} FAIL=${FAIL}"
echo "══════════════════════════════════════"

if [ ${#ERRORS[@]} -gt 0 ]; then
    echo "실패 항목:"
    for err in "${ERRORS[@]}"; do
        echo "  - $err"
    done
    echo ""
    echo "에이전트에게 전달할 메시지:"
    echo "verify.sh 실패. 위 항목을 수정 후 재시도하세요."
    exit 1
fi

echo -e "${GREEN}모든 완료 기준 통과${NC}"
exit 0
```

### .claude/settings.json 최소 설정

```json
{
  "permissions": {
    "allow": [
      "Bash(npm run *)",
      "Bash(bash scripts/verify.sh*)",
      "Bash(git status)",
      "Bash(git diff*)",
      "Bash(git add *)",
      "Bash(git commit*)",
      "Read(**)",
      "Write(src/**)",
      "Write(.omc/specs/**)"
    ],
    "deny": [
      "Bash(git push --force*)",
      "Bash(rm -rf *)",
      "Bash(curl * | bash)"
    ]
  },
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "command",
            "command": "bash scripts/verify.sh",
            "description": "파일 저장 후 자동 검증"
          }
        ]
      }
    ]
  }
}
```

---

## SDD로 Spec.md 작성하기

### SDD(Spec-Driven Development)란

```
전통적 방식:
  PRD → 바로 구현 → 에이전트가 추측으로 채움 → 버그

SDD 방식:
  PRD → Spec.md (명확한 계약) → 구현 → verify.sh로 검증
         │
         └── 에이전트가 추측할 여지 없음
```

### Spec.md 전체 템플릿

```markdown
# Spec: [기능명]

> 버전: 1.0.0
> 작성일: YYYY-MM-DD
> 상태: DRAFT | APPROVED | IMPLEMENTED | VERIFIED

---

## 1. 목적 (Why)

[이 기능이 왜 필요한지 한 문단으로 설명]

비즈니스 가치:
- [ ] [측정 가능한 성과 1]
- [ ] [측정 가능한 성과 2]

---

## 2. 범위 (Scope)

### 포함 (In Scope)
- [구체적으로 구현할 것 1]
- [구체적으로 구현할 것 2]

### 제외 (Out of Scope)
- [이번 구현에서 하지 않을 것 1]
- [이번 구현에서 하지 않을 것 2]

---

## 3. 인터페이스 명세 (Interface Contract)

### API 엔드포인트

```
POST /api/v1/auth/login
Content-Type: application/json

Request:
{
  "email": string,      // RFC 5322 형식
  "password": string,   // 8자 이상, 대소문자+숫자+특수문자
  "rememberMe": boolean // optional, default: false
}

Response (200 OK):
{
  "accessToken": string,   // JWT, 1시간 유효
  "refreshToken": string,  // JWT, 30일 유효 (rememberMe=true 시)
  "user": {
    "id": string,          // UUID v4
    "email": string,
    "role": "user" | "admin"
  }
}

Response (401 Unauthorized):
{
  "error": "INVALID_CREDENTIALS",
  "message": string
}

Response (429 Too Many Requests):
{
  "error": "RATE_LIMITED",
  "retryAfter": number  // 초 단위
}
```

### 데이터 모델

```sql
-- users 테이블
CREATE TABLE users (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email       VARCHAR(255) UNIQUE NOT NULL,
  password    VARCHAR(255) NOT NULL,  -- bcrypt hash, cost=12
  role        VARCHAR(50) DEFAULT 'user',
  created_at  TIMESTAMP DEFAULT NOW(),
  updated_at  TIMESTAMP DEFAULT NOW()
);

-- login_attempts 테이블 (rate limiting)
CREATE TABLE login_attempts (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email       VARCHAR(255) NOT NULL,
  ip_address  INET NOT NULL,
  success     BOOLEAN NOT NULL,
  created_at  TIMESTAMP DEFAULT NOW()
);
```

---

## 4. 동작 명세 (Behavior)

### 정상 흐름

```gherkin
Feature: 사용자 로그인

  Scenario: 유효한 자격증명으로 로그인
    Given 이메일 "user@example.com", 비밀번호 "Pass123!"로 등록된 사용자가 있을 때
    When POST /api/v1/auth/login {"email": "user@example.com", "password": "Pass123!"}
    Then HTTP 200 응답
    And 응답에 accessToken, refreshToken, user 필드가 있을 것
    And accessToken은 유효한 JWT일 것

  Scenario: 잘못된 비밀번호
    Given 이메일 "user@example.com"으로 등록된 사용자가 있을 때
    When POST /api/v1/auth/login {"email": "user@example.com", "password": "WrongPass!"}
    Then HTTP 401 응답
    And 응답에 "INVALID_CREDENTIALS" 에러 코드가 있을 것
```

### Rate Limiting 규칙

- 동일 이메일로 5분 내 5회 실패 시 15분 잠금
- 동일 IP로 5분 내 20회 실패 시 1시간 잠금
- 429 응답의 retryAfter는 잠금 해제까지 남은 초

### 보안 요구사항

- [ ] 비밀번호는 bcrypt (cost=12) 이상으로 저장
- [ ] accessToken 만료: 1시간
- [ ] refreshToken 만료: 30일 (rememberMe=false 시 세션)
- [ ] HTTPS only (HTTP redirect)
- [ ] CORS: 허용 도메인 목록만

---

## 5. 완료 기준 (Definition of Done)

에이전트는 다음 모든 항목을 충족해야 구현 완료로 간주합니다:

### 코드 기준
- [ ] `npm run build` 에러 0개
- [ ] `npm run lint` 경고 0개
- [ ] `npx tsc --noEmit` 에러 0개

### 테스트 기준
- [ ] 단위 테스트 커버리지 ≥ 85%
- [ ] 통합 테스트: 위 Gherkin 시나리오 전부 통과
- [ ] 경계값 테스트: 이메일 형식, 비밀번호 최소/최대 길이

### 보안 기준
- [ ] 비밀번호 평문 로그 없음 (grep -r "password" logs/)
- [ ] JWT 시크릿 하드코딩 없음 (환경변수 사용)
- [ ] npm audit: high/critical 취약점 0개

### 성능 기준
- [ ] 로그인 API p95 응답 시간 < 500ms (로컬 DB 기준)

---

## 6. 의존성 및 제약사항

### 외부 의존성
- Node.js >= 20.0.0
- PostgreSQL >= 15
- Redis >= 7 (rate limiting 저장소)

### 환경변수
```
JWT_SECRET=<32자 이상 랜덤 문자열>
JWT_REFRESH_SECRET=<32자 이상 랜덤 문자열>
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
```

---

## 7. 테스트 픽스처

```typescript
// tests/fixtures/auth.fixtures.ts
export const validUser = {
  email: "test@example.com",
  password: "Test123!@#",
  hashedPassword: "$2b$12$...",  // bcrypt hash
};

export const invalidCredentials = [
  { email: "wrong@example.com", password: "Test123!@#" },
  { email: "test@example.com", password: "wrongpassword" },
  { email: "", password: "" },
];
```

---

## 8. 변경 이력

| 날짜 | 버전 | 변경 내용 | 작성자 |
|------|------|-----------|--------|
| YYYY-MM-DD | 1.0.0 | 초안 작성 | [이름] |
```

### PRD → Spec.md 변환 프롬프트

Claude Code에 붙여넣기:

```
당신은 소프트웨어 스펙 작성 전문가입니다.

다음 PRD(Product Requirements Document)를 엄격한 Spec.md로 변환하세요.

변환 규칙:
1. 모호한 표현은 구체적 수치로 대체
   - "빠르게" → "p95 < 200ms"
   - "안전하게" → "bcrypt cost=12, HTTPS only"
   - "많은 사용자" → "동시 1000 요청"

2. 모든 API는 요청/응답 스키마를 JSON으로 명시

3. 동작은 Gherkin (Given/When/Then) 형식으로

4. 완료 기준은 자동 검증 가능한 항목만

5. 다음 섹션을 반드시 포함:
   - 범위 (In Scope / Out of Scope)
   - 인터페이스 명세
   - 동작 명세
   - 완료 기준 (체크리스트)
   - 환경변수 목록

PRD:
---
[여기에 PRD 내용 붙여넣기]
---

위 PRD를 Spec.md 형식으로 변환하세요.
불명확한 부분은 [CLARIFICATION NEEDED: 질문] 형식으로 표시하세요.
```

---

## 작업 흐름을 문서·훅·스킬·평가로 묶기

### 전체 하네스 구조도

```
                    ┌─────────────────────────────────────────┐
                    │           나만의 Agent Harness           │
                    └─────────────────────────────────────────┘

  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │   문서 층    │   │    훅 층     │   │   스킬 층    │   │   평가 층    │
  │              │   │              │   │              │   │              │
  │ CLAUDE.md    │   │ PreToolUse   │   │ /spec        │   │ verify.sh    │
  │ Spec.md      │   │ PostToolUse  │   │ /verify      │   │ eval.py      │
  │ Design.md    │   │ PreCompact   │   │ /deploy      │   │ coverage     │
  │ AGENTS.md    │   │ Stop         │   │ /review      │   │ benchmark    │
  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
         │                  │                  │                  │
         └──────────────────┴──────────────────┴──────────────────┘
                                     │
                              에이전트 실행
                                     │
                              ┌──────▼──────┐
                              │   모델      │
                              │ (교체 가능) │
                              └─────────────┘
```

### 훅 설정 (.claude/settings.json 완성본)

```json
{
  "permissions": {
    "allow": [
      "Bash(npm *)",
      "Bash(npx *)",
      "Bash(git status)",
      "Bash(git diff*)",
      "Bash(git add src/*)",
      "Bash(git commit*)",
      "Bash(bash scripts/*)",
      "Read(**)",
      "Write(src/**)",
      "Write(.omc/**)",
      "Write(tests/**)"
    ],
    "deny": [
      "Bash(git push --force*)",
      "Bash(rm -rf /)",
      "Bash(curl * | sh)",
      "Bash(sudo *)"
    ]
  },
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "bash scripts/hooks/pre-bash.sh",
            "description": "위험 명령 사전 차단"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "command",
            "command": "bash scripts/hooks/post-write.sh \"$TOOL_INPUT_FILE_PATH\"",
            "description": "파일 저장 후 린트 + 타입 체크"
          }
        ]
      },
      {
        "matcher": "Bash(git commit*)",
        "hooks": [
          {
            "type": "command",
            "command": "bash scripts/verify.sh",
            "description": "커밋 전 전체 검증"
          }
        ]
      }
    ],
    "PreCompact": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash scripts/hooks/save-context.sh",
            "description": "컨텍스트 압축 전 중요 상태 저장"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash scripts/hooks/session-end.sh",
            "description": "세션 종료 시 작업 상태 저장"
          }
        ]
      }
    ]
  }
}
```

### 훅 스크립트 예시

`scripts/hooks/post-write.sh`:

```bash
#!/usr/bin/env bash
# 파일 저장 후 즉시 실행되는 훅

FILE_PATH="${1:-}"

# TypeScript 파일이면 타입 체크
if [[ "$FILE_PATH" == *.ts || "$FILE_PATH" == *.tsx ]]; then
    echo "타입 체크 중: $FILE_PATH"
    npx tsc --noEmit 2>&1 | head -20 || true
fi

# 테스트 파일이면 해당 테스트만 실행
if [[ "$FILE_PATH" == *.test.ts || "$FILE_PATH" == *.spec.ts ]]; then
    echo "관련 테스트 실행 중..."
    npx vitest run "$FILE_PATH" --reporter=verbose 2>&1 | tail -20 || true
fi
```

`scripts/hooks/save-context.sh`:

```bash
#!/usr/bin/env bash
# 컨텍스트 압축 전 현재 작업 상태를 저장

mkdir -p .omc/state

# 현재 변경 파일 목록 저장
git diff --name-only > .omc/state/changed-files.txt

# 현재 TODO 상태 추출
grep -r "TODO\|FIXME\|HACK" src/ > .omc/state/todos.txt 2>/dev/null || true

# verify.sh 결과 저장
bash scripts/verify.sh > .omc/state/last-verify.txt 2>&1 || true

echo "컨텍스트 저장 완료: .omc/state/"
```

### 커스텀 스킬 예시

`.claude/commands/spec.md` (/spec 명령):

```markdown
---
description: 새 기능의 Spec.md를 대화식으로 작성합니다
---

사용자가 제공한 기능 설명을 바탕으로 Spec.md를 작성하세요.

1. 먼저 기능의 핵심 목적을 한 문장으로 요약하세요.
2. 다음 질문에 답하며 Spec을 채우세요:
   - API 엔드포인트가 있는가? → 요청/응답 스키마 명시
   - DB 변경이 있는가? → 마이그레이션 SQL 포함
   - 외부 서비스 연동이 있는가? → 타임아웃/재시도 정책 명시
   - 보안 요구사항이 있는가? → 구체적 기술 스펙으로 변환

3. .omc/specs/[기능명].md 에 저장하세요.
4. 저장 후 "Spec이 불명확한 부분 목록"을 출력하세요.

$ARGUMENTS
```

`.claude/commands/verify.md` (/verify 명령):

```markdown
---
description: 현재 구현이 Spec의 완료 기준을 충족하는지 검증합니다
---

다음 순서로 검증을 실행하세요:

1. bash scripts/verify.sh 실행
2. 실패 항목 분석
3. 각 실패의 근본 원인 파악
4. 수정 우선순위 제시 (Critical → High → Medium)
5. 수정 완료 후 verify.sh 재실행

검증 대상 Spec: $ARGUMENTS
```

### eval.py - 에이전트 출력 평가 스크립트

```python
#!/usr/bin/env python3
"""
eval.py: 에이전트 출력을 자동으로 평가하는 스크립트
사용법: python scripts/eval.py --spec .omc/specs/auth-feature.md --output output.txt
"""

import argparse
import json
import re
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

@dataclass
class EvalResult:
    criterion: str
    passed: bool
    score: float  # 0.0 ~ 1.0
    details: str

@dataclass
class EvalSuite:
    results: List[EvalResult] = field(default_factory=list)

    @property
    def total_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def passed(self) -> bool:
        return self.total_score >= 0.8

def check_build() -> EvalResult:
    result = subprocess.run(
        ["npm", "run", "build", "--silent"],
        capture_output=True, text=True
    )
    passed = result.returncode == 0
    return EvalResult(
        criterion="빌드 성공",
        passed=passed,
        score=1.0 if passed else 0.0,
        details=result.stderr[:200] if not passed else "OK"
    )

def check_test_coverage(threshold: float = 0.8) -> EvalResult:
    result = subprocess.run(
        ["npm", "test", "--", "--coverage", "--reporter=json"],
        capture_output=True, text=True
    )
    try:
        coverage_data = json.loads(result.stdout)
        pct = coverage_data["total"]["lines"]["pct"] / 100
        passed = pct >= threshold
        return EvalResult(
            criterion=f"테스트 커버리지 >= {threshold*100:.0f}%",
            passed=passed,
            score=min(pct / threshold, 1.0),
            details=f"실제 커버리지: {pct*100:.1f}%"
        )
    except (json.JSONDecodeError, KeyError):
        return EvalResult(
            criterion="테스트 커버리지",
            passed=False,
            score=0.0,
            details="커버리지 데이터 파싱 실패"
        )

def check_no_any_types(src_dir: str = "src") -> EvalResult:
    result = subprocess.run(
        ["grep", "-r", ": any", src_dir, "--include=*.ts"],
        capture_output=True, text=True
    )
    count = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
    passed = count == 0
    score = max(0.0, 1.0 - count * 0.1)
    return EvalResult(
        criterion="any 타입 사용 없음",
        passed=passed,
        score=score,
        details=f"any 타입 {count}개 발견" if not passed else "OK"
    )

def check_spec_coverage(spec_path: str, src_dir: str = "src") -> EvalResult:
    """Spec의 완료 기준 체크리스트 항목이 구현됐는지 확인"""
    spec_content = Path(spec_path).read_text()

    # 완료 기준 추출
    dod_section = re.search(r"## 5\. 완료 기준.*?(?=##|\Z)", spec_content, re.DOTALL)
    if not dod_section:
        return EvalResult("Spec 커버리지", False, 0.0, "완료 기준 섹션 없음")

    checklist = re.findall(r"- \[ \] (.+)", dod_section.group())
    total = len(checklist)

    if total == 0:
        return EvalResult("Spec 커버리지", True, 1.0, "체크리스트 항목 없음")

    # 간단한 키워드 기반 검증 (실제로는 더 정교하게)
    passed_count = 0
    for item in checklist:
        # 빌드/린트/타입 항목은 별도 검증으로 처리
        if any(kw in item for kw in ["build", "lint", "tsc"]):
            passed_count += 1

    score = passed_count / total
    return EvalResult(
        criterion="Spec 완료 기준 충족",
        passed=score >= 0.8,
        score=score,
        details=f"{passed_count}/{total} 항목 통과"
    )

def run_eval(spec_path: str = None) -> EvalSuite:
    suite = EvalSuite()
    suite.results.append(check_build())
    suite.results.append(check_test_coverage(0.8))
    suite.results.append(check_no_any_types())
    if spec_path and Path(spec_path).exists():
        suite.results.append(check_spec_coverage(spec_path))
    return suite

def main():
    parser = argparse.ArgumentParser(description="에이전트 출력 평가")
    parser.add_argument("--spec", help="Spec.md 경로")
    parser.add_argument("--json", action="store_true", help="JSON 출력")
    args = parser.parse_args()

    suite = run_eval(args.spec)

    if args.json:
        output = {
            "total_score": suite.total_score,
            "passed": suite.passed,
            "results": [
                {
                    "criterion": r.criterion,
                    "passed": r.passed,
                    "score": r.score,
                    "details": r.details,
                }
                for r in suite.results
            ],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"\n{'='*50}")
        print(f"평가 결과: {'PASS' if suite.passed else 'FAIL'}")
        print(f"총점: {suite.total_score*100:.1f}%")
        print(f"{'='*50}")
        for r in suite.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"[{status}] {r.criterion}: {r.details}")
        print()

if __name__ == "__main__":
    main()
```

---

## 하네스 자기개선 루프

### 자기개선 루프 설계

```
┌─────────────────────────────────────────────────────────┐
│                   하네스 자기개선 루프                    │
└─────────────────────────────────────────────────────────┘

  [실제 작업 실행]
       │
       ▼
  [eval.py로 평가] ──── 점수 기록 (.omc/evals/history.json)
       │
       ▼
  [패턴 분석]
  - 어떤 작업 유형에서 점수가 낮은가?
  - 어떤 프롬프트가 더 좋은 결과를 냈는가?
  - 어떤 훅이 가장 자주 실패를 잡았는가?
       │
       ▼
  [하네스 개선 제안 생성]
  - CLAUDE.md 보완
  - Spec 템플릿 개선
  - 새 스킬 추가
  - 새 훅 추가
       │
       ▼
  [개선 사항 적용] ──────────────────────────────────────┐
       │                                                   │
       └─────────────── 다음 작업 실행 ────────────────────┘
```

### 평가 이력 수집 스크립트

```bash
#!/usr/bin/env bash
# scripts/record-eval.sh
# 매 작업 완료 후 평가 결과를 기록

mkdir -p .omc/evals

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
TASK=${1:-"unknown"}
SPEC=${2:-""}

# 평가 실행
EVAL_OUTPUT=$(python3 scripts/eval.py --spec "$SPEC" --json 2>/dev/null)
SCORE=$(echo "$EVAL_OUTPUT" | python3 -c "import json,sys; print(json.load(sys.stdin)['total_score'])" 2>/dev/null || echo "0")

# 이력 파일에 추가
HISTORY_FILE=".omc/evals/history.json"

if [ ! -f "$HISTORY_FILE" ]; then
    echo "[]" > "$HISTORY_FILE"
fi

# 새 항목 추가
python3 -c "
import json, sys
from pathlib import Path

history_path = Path('$HISTORY_FILE')
history = json.loads(history_path.read_text())

history.append({
    'timestamp': '$TIMESTAMP',
    'task': '$TASK',
    'spec': '$SPEC',
    'score': $SCORE,
    'eval': $EVAL_OUTPUT
})

# 최근 100개만 유지
history = history[-100:]
history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False))
print(f'평가 기록됨: score={$SCORE}')
"
```

### 메타 프롬프트: 하네스 자기개선

Claude Code에 붙여넣기:

```
당신은 Agent Harness 개선 전문가입니다.

현재 하네스 구조를 분석하고 개선점을 찾아주세요.

## 분석할 파일들
- CLAUDE.md: 현재 에이전트 컨텍스트 문서
- .claude/settings.json: 현재 훅 및 권한 설정
- .omc/evals/history.json: 최근 100개 작업 평가 이력
- scripts/verify.sh: 현재 완료 기준 검증 스크립트

## 분석 기준
1. 낮은 점수 패턴: 어떤 작업 유형에서 반복적으로 실패하는가?
2. 누락된 훅: 어떤 실수가 자동으로 잡히지 않고 있는가?
3. 불명확한 지시: CLAUDE.md에서 에이전트가 잘못 해석하는 부분은?
4. 비효율적 워크플로: 반복 수작업이 스킬로 자동화될 수 있는 부분은?
5. 검증 격차: verify.sh가 잡지 못하는 품질 문제는?

## 출력 형식
### 발견된 문제점
[문제 1]: [심각도 High/Medium/Low]
  - 증상: [구체적 증상]
  - 근본 원인: [분석]
  - 개선안: [구체적 변경 내용]

### 즉시 적용 가능한 개선 (오늘)
[파일명] 수정:
```변경 전```
```변경 후```

### 중기 개선 (이번 주)
- [ ] [새 스킬 추가 계획]
- [ ] [새 훅 추가 계획]

### 장기 개선 (이번 달)
- [ ] [하네스 구조 변경 계획]

분석을 시작하세요.
```

### 하네스 품질 지표 대시보드

```python
#!/usr/bin/env python3
# scripts/harness-dashboard.py
# 하네스 건강 상태를 한눈에 보여주는 대시보드

import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

def load_history(days: int = 30) -> list:
    history_path = Path(".omc/evals/history.json")
    if not history_path.exists():
        return []

    history = json.loads(history_path.read_text())
    cutoff = datetime.utcnow() - timedelta(days=days)

    return [
        h for h in history
        if datetime.fromisoformat(h["timestamp"].replace("Z", "+00:00")).replace(tzinfo=None) > cutoff
    ]

def calculate_metrics(history: list) -> dict:
    if not history:
        return {}

    scores = [h["score"] for h in history]
    avg_score = sum(scores) / len(scores)

    # 실패 패턴 분석
    fail_reasons = defaultdict(int)
    for h in history:
        eval_data = h.get("eval", {})
        for result in eval_data.get("results", []):
            if not result["passed"]:
                fail_reasons[result["criterion"]] += 1

    # 개선 추세 (최근 10개 vs 이전 10개)
    recent = scores[-10:] if len(scores) >= 10 else scores
    previous = scores[-20:-10] if len(scores) >= 20 else []
    trend = (sum(recent)/len(recent) - sum(previous)/len(previous)) if previous else 0

    return {
        "total_tasks": len(history),
        "avg_score": avg_score,
        "pass_rate": sum(1 for s in scores if s >= 0.8) / len(scores),
        "trend": trend,
        "top_failures": sorted(fail_reasons.items(), key=lambda x: -x[1])[:5],
    }

def main():
    history = load_history(30)
    metrics = calculate_metrics(history)

    if not metrics:
        print("평가 이력 없음. scripts/record-eval.sh 실행 후 재시도하세요.")
        return

    print("\n" + "="*55)
    print("   Agent Harness 품질 대시보드 (최근 30일)")
    print("="*55)
    print(f"총 작업 수:    {metrics['total_tasks']:>6}개")
    print(f"평균 점수:     {metrics['avg_score']*100:>5.1f}%")
    print(f"통과율:        {metrics['pass_rate']*100:>5.1f}%")
    trend_sign = "+" if metrics["trend"] > 0 else ""
    print(f"개선 추세:     {trend_sign}{metrics['trend']*100:>4.1f}% (최근 10개 기준)")
    print()

    if metrics["top_failures"]:
        print("자주 실패하는 기준:")
        for criterion, count in metrics["top_failures"]:
            print(f"  {count:>3}회 - {criterion}")

    print()
    if metrics["avg_score"] >= 0.9:
        print("상태: EXCELLENT - 하네스가 잘 동작하고 있습니다")
    elif metrics["avg_score"] >= 0.8:
        print("상태: GOOD - 소폭 개선 여지 있음")
    elif metrics["avg_score"] >= 0.6:
        print("상태: NEEDS IMPROVEMENT - 메타 프롬프트로 하네스 점검 권장")
    else:
        print("상태: CRITICAL - 즉시 하네스 개선 필요")
    print()

if __name__ == "__main__":
    main()
```

### 실제 프로젝트 하네스 설정 완성 예시

```bash
# 새 프로젝트에 하네스 설치 (setup-harness.sh)
#!/usr/bin/env bash

set -euo pipefail

echo "Agent Harness 설정 중..."

# 디렉토리 생성
mkdir -p .claude/commands
mkdir -p .omc/{specs,evals,plans,state}
mkdir -p scripts/{hooks,integration}

# verify.sh 설치
curl -sL https://your-harness-repo.example.com/verify.sh -o scripts/verify.sh
chmod +x scripts/verify.sh

# eval.py 설치
curl -sL https://your-harness-repo.example.com/eval.py -o scripts/eval.py

# 훅 스크립트 설치
for hook in pre-bash post-write save-context session-end; do
    curl -sL "https://your-harness-repo.example.com/hooks/${hook}.sh" \
         -o "scripts/hooks/${hook}.sh"
    chmod +x "scripts/hooks/${hook}.sh"
done

# .claude/settings.json 생성
cat > .claude/settings.json << 'SETTINGS'
{
  "permissions": {
    "allow": ["Bash(npm *)", "Bash(bash scripts/*)", "Read(**)", "Write(src/**)"],
    "deny": ["Bash(git push --force*)", "Bash(rm -rf /)"]
  },
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [{"type": "command", "command": "bash scripts/hooks/post-write.sh \"$TOOL_INPUT_FILE_PATH\""}]
      }
    ],
    "Stop": [
      {"hooks": [{"type": "command", "command": "bash scripts/hooks/session-end.sh"}]}
    ]
  }
}
SETTINGS

# CLAUDE.md 템플릿 생성 (없으면)
if [ ! -f CLAUDE.md ]; then
    cat > CLAUDE.md << 'CLAUDE'
# 프로젝트명

## 빠른 컨텍스트
- 언어: [언어]
- 빌드: `npm run build`
- 테스트: `npm test`

## 완료 기준
bash scripts/verify.sh 통과

## 코딩 규칙
- [규칙 1]
- [규칙 2]
CLAUDE
fi

echo "하네스 설정 완료!"
echo ""
echo "다음 단계:"
echo "1. CLAUDE.md를 프로젝트에 맞게 수정하세요"
echo "2. scripts/verify.sh의 완료 기준을 조정하세요"
echo "3. Claude Code를 실행하고 '/spec [기능명]'으로 시작하세요"
```

---

## 요약

**하네스의 핵심 원칙:**

1. 모델은 교체되지만 하네스는 남는다. 하네스에 투자하는 것이 모델 튜닝보다 ROI가 높다.

2. 최소 하네스 = `CLAUDE.md` + `verify.sh` + `.claude/settings.json`. 이 세 파일만 있어도 에이전트 품질이 크게 올라간다.

3. `Spec.md`는 에이전트와의 계약서다. 모호한 요구사항은 모호한 구현을 낳는다.

4. 훅은 에이전트의 습관을 형성한다. Pre/Post 훅으로 반복 실수를 자동으로 방지한다.

5. 하네스 자체도 에이전트로 개선한다. 메타 프롬프트와 평가 이력을 활용해 하네스가 스스로 진화하도록 설계한다.
