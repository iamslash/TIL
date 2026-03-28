# 문서화 자동화 & 팀 단위 AI 코딩 에이전트 운영 규칙

> 커리큘럼 04-CH03 대응 문서

## 목차

- [1. AI가 코드를 바꾸면 문서도 같이 바뀌게 만들기](#1-ai가-코드를-바꾸면-문서도-같이-바뀌게-만들기)
  - [1.1 PostToolUse 훅으로 README/CHANGELOG 자동 업데이트](#11-posttooluse-훅으로-readmechangelog-자동-업데이트)
  - [1.2 CLAUDE.md에 문서 동기화 규칙 넣기](#12-claudemd에-문서-동기화-규칙-넣기)
  - [1.3 JSDoc/Docstring 자동 생성 프롬프트 패턴](#13-jsdocdocstring-자동-생성-프롬프트-패턴)
- [2. 사람과 AI가 함께 일하는 Git Repository 운영 규칙](#2-사람과-ai가-함께-일하는-git-repository-운영-규칙)
  - [2.1 AI 커밋 메시지 컨벤션](#21-ai-커밋-메시지-컨벤션)
  - [2.2 브랜치 네이밍 규칙](#22-브랜치-네이밍-규칙)
  - [2.3 PR 템플릿에 AI 사용 여부 표기](#23-pr-템플릿에-ai-사용-여부-표기)
  - [2.4 .gitignore에 AI 임시 파일 추가](#24-gitignore에-ai-임시-파일-추가)
  - [2.5 CODEOWNERS와 AI 작업 범위 제한](#25-codeowners와-ai-작업-범위-제한)

---

## 1. AI가 코드를 바꾸면 문서도 같이 바뀌게 만들기

핵심 원칙: **코드와 문서의 동기화를 자동화**하면 "코드는 최신인데 문서는 3개월 전 상태"라는 고전적인 문제를 방지한다.

### 1.1 PostToolUse 훅으로 README/CHANGELOG 자동 업데이트

Claude Code는 도구 실행 후 훅(hook)을 실행할 수 있다. `PostToolUse` 훅을 설정하면 파일 편집이 발생할 때마다 문서 업데이트 스크립트를 자동으로 실행한다.

#### 훅 설정 파일: `.claude/settings.json`

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/update-docs.sh"
          }
        ]
      }
    ]
  }
}
```

#### 훅 스크립트: `.claude/hooks/update-docs.sh`

```bash
#!/usr/bin/env bash
# update-docs.sh - 코드 변경 후 자동으로 문서를 업데이트한다.
set -euo pipefail

# 환경변수에서 변경된 파일 경로를 읽는다.
# Claude Code는 CLAUDE_TOOL_INPUT_FILE_PATH 환경변수로 편집된 파일 경로를 전달한다.
CHANGED_FILE="${CLAUDE_TOOL_INPUT_FILE_PATH:-}"

if [[ -z "$CHANGED_FILE" ]]; then
  exit 0
fi

echo "[doc-hook] 변경 감지: $CHANGED_FILE"

# 소스 파일이 변경된 경우에만 동작한다.
if [[ "$CHANGED_FILE" =~ \.(ts|tsx|js|jsx|py|go|rs|java)$ ]]; then

  # 1. CHANGELOG.md 자동 업데이트
  TODAY=$(date +%Y-%m-%d)
  CHANGELOG="CHANGELOG.md"

  if [[ -f "$CHANGELOG" ]]; then
    # 오늘 날짜 섹션이 없으면 추가한다.
    if ! grep -q "## \[$TODAY\]" "$CHANGELOG"; then
      # 파일 맨 위에 새 섹션을 삽입한다.
      TMPFILE=$(mktemp)
      {
        head -1 "$CHANGELOG"
        echo ""
        echo "## [$TODAY] - Auto-updated"
        echo ""
        echo "### Changed"
        echo "- \`$CHANGED_FILE\` modified by AI agent"
        echo ""
        tail -n +2 "$CHANGELOG"
      } > "$TMPFILE"
      mv "$TMPFILE" "$CHANGELOG"
      echo "[doc-hook] CHANGELOG.md 업데이트 완료"
    fi
  fi

  # 2. 모듈 README가 있으면 "Last Updated" 타임스탬프를 갱신한다.
  MODULE_DIR=$(dirname "$CHANGED_FILE")
  MODULE_README="$MODULE_DIR/README.md"

  if [[ -f "$MODULE_README" ]]; then
    # "Last Updated:" 줄을 현재 날짜로 교체한다.
    sed -i.bak "s/Last Updated: .*/Last Updated: $TODAY/" "$MODULE_README"
    rm -f "$MODULE_README.bak"
    echo "[doc-hook] $MODULE_README 타임스탬프 갱신 완료"
  fi
fi
```

#### 훅 동작 확인

```bash
# 훅 파일에 실행 권한을 부여한다.
chmod +x .claude/hooks/update-docs.sh

# Claude Code를 통해 파일을 편집하면 훅이 자동으로 실행된다.
# 수동으로 테스트하려면 환경변수를 설정하고 직접 실행한다.
CLAUDE_TOOL_INPUT_FILE_PATH="src/api/user.ts" bash .claude/hooks/update-docs.sh
```

#### 더 정교한 훅: CHANGELOG에 변경 내용 요약 포함

```bash
#!/usr/bin/env bash
# update-changelog-with-summary.sh
set -euo pipefail

CHANGED_FILE="${CLAUDE_TOOL_INPUT_FILE_PATH:-}"
TOOL_NAME="${CLAUDE_TOOL_NAME:-}"
TODAY=$(date +%Y-%m-%d)

# Edit 또는 Write 도구가 사용된 경우에만 실행한다.
if [[ "$TOOL_NAME" != "Edit" && "$TOOL_NAME" != "Write" ]]; then
  exit 0
fi

# 소스 파일 변경만 추적한다.
if [[ ! "$CHANGED_FILE" =~ \.(ts|js|py|go)$ ]]; then
  exit 0
fi

# git diff로 변경 요약을 생성한다.
DIFF_SUMMARY=$(git diff --stat HEAD -- "$CHANGED_FILE" 2>/dev/null || echo "새 파일")

# CHANGELOG.md에 항목을 추가한다.
cat >> CHANGELOG.md << EOF

### [$TODAY] AI Agent Change
- File: \`$CHANGED_FILE\`
- Summary: $DIFF_SUMMARY
EOF

echo "[doc-hook] CHANGELOG에 변경 내용 추가 완료"
```

---

### 1.2 CLAUDE.md에 문서 동기화 규칙 넣기

`CLAUDE.md`는 Claude Code가 프로젝트에서 작업할 때 항상 읽는 지시 파일이다. 여기에 문서 동기화 규칙을 명시하면 AI가 코드를 변경할 때마다 문서를 함께 수정하도록 강제할 수 있다.

#### 실제 CLAUDE.md 규칙 예시

```markdown
# CLAUDE.md

## 문서 동기화 규칙 (MANDATORY)

### 코드 변경 시 반드시 함께 수정해야 하는 파일들

1. **함수/클래스 추가 또는 변경 시**
   - 해당 파일의 JSDoc/docstring을 업데이트한다.
   - 해당 모듈의 `README.md`에 API 변경 사항을 반영한다.
   - `CHANGELOG.md`에 변경 내용을 기록한다.

2. **API 엔드포인트 추가/변경/삭제 시**
   - `docs/api.md` 또는 `openapi.yaml`을 반드시 업데이트한다.
   - 엔드포인트 삭제 시 deprecation 공지를 추가한다.

3. **환경변수 추가 시**
   - `.env.example`에 해당 변수와 설명을 추가한다.
   - `docs/configuration.md`에 설명을 추가한다.

4. **데이터베이스 스키마 변경 시**
   - `docs/schema.md`를 업데이트한다.
   - 마이그레이션 파일 이름을 `CHANGELOG.md`에 기록한다.

### 문서 작성 형식

- 모든 공개(public) 함수에는 JSDoc/docstring이 있어야 한다.
- JSDoc 형식: `@param`, `@returns`, `@throws`, `@example` 태그를 포함한다.
- 한국어로 설명을 작성하되 코드와 타입은 영어를 유지한다.

### 자동화된 문서 검사

코드 변경 후 다음 명령어를 실행하여 문서 누락을 확인한다:
\`\`\`bash
npm run docs:check   # JSDoc 커버리지 확인
npm run docs:build   # 문서 빌드 및 링크 검사
\`\`\`

### 예외 사항

다음 파일 변경은 문서 업데이트가 면제된다:
- `*.test.ts`, `*.spec.ts` (테스트 파일)
- `*.config.js` (빌드 설정 파일)
- `scripts/` 디렉토리 내 일회성 스크립트
```

#### 규칙을 Claude에게 확인시키는 프롬프트

```
이 파일을 수정하기 전에 CLAUDE.md의 문서 동기화 규칙을 확인하고,
코드 변경 후 어떤 문서를 업데이트해야 하는지 목록을 먼저 알려줘.
```

---

### 1.3 JSDoc/Docstring 자동 생성 프롬프트 패턴

AI에게 일관된 문서 주석을 생성하도록 프롬프트를 구성하는 패턴이다.

#### TypeScript JSDoc 생성 프롬프트

```
다음 TypeScript 함수에 JSDoc 주석을 추가해줘.

요구사항:
- @param: 각 파라미터의 타입과 한국어 설명
- @returns: 반환값 타입과 한국어 설명
- @throws: 발생 가능한 에러 타입과 조건
- @example: 실제 사용 예시 코드 (실행 가능해야 함)
- @since: 현재 날짜 (YYYY-MM-DD 형식)

함수:
\`\`\`typescript
async function fetchUserProfile(userId: string, options?: FetchOptions): Promise<UserProfile> {
  const response = await api.get(`/users/${userId}`, options);
  if (!response.ok) throw new ApiError(response.status);
  return response.json();
}
\`\`\`
```

#### 생성 결과 예시

```typescript
/**
 * 사용자 프로필 정보를 API에서 가져온다.
 *
 * @param userId - 조회할 사용자의 고유 식별자
 * @param options - API 요청 옵션 (타임아웃, 헤더 등)
 * @returns 사용자 프로필 데이터를 담은 Promise
 * @throws {ApiError} API 응답이 실패한 경우 (4xx, 5xx 상태 코드)
 * @throws {NetworkError} 네트워크 연결 문제가 발생한 경우
 * @example
 * ```typescript
 * const profile = await fetchUserProfile('user-123');
 * console.log(profile.name); // "홍길동"
 *
 * // 타임아웃 옵션과 함께 사용
 * const profile = await fetchUserProfile('user-123', { timeout: 5000 });
 * ```
 * @since 2024-01-15
 */
async function fetchUserProfile(userId: string, options?: FetchOptions): Promise<UserProfile> {
  const response = await api.get(`/users/${userId}`, options);
  if (!response.ok) throw new ApiError(response.status);
  return response.json();
}
```

#### Python Docstring 생성 프롬프트

```
다음 Python 함수에 Google 스타일 docstring을 추가해줘.

요구사항:
- 함수 목적 요약 (한국어, 1-2문장)
- Args 섹션: 각 인자의 타입과 한국어 설명
- Returns 섹션: 반환값 설명
- Raises 섹션: 발생 가능한 예외
- Example 섹션: 실행 가능한 예시

함수:
\`\`\`python
def calculate_discount(price: float, user_tier: str, coupon_code: str = None) -> float:
    tier_rates = {"bronze": 0.05, "silver": 0.10, "gold": 0.20}
    rate = tier_rates.get(user_tier, 0)
    if coupon_code and validate_coupon(coupon_code):
        rate += 0.05
    return price * (1 - rate)
\`\`\`
```

#### 생성 결과 예시

```python
def calculate_discount(price: float, user_tier: str, coupon_code: str = None) -> float:
    """사용자 등급과 쿠폰 코드를 기반으로 할인된 가격을 계산한다.

    사용자 등급에 따른 기본 할인율을 적용하고, 유효한 쿠폰이 있으면
    추가 5% 할인을 더한다.

    Args:
        price: 할인 전 원래 가격 (0 이상의 양수)
        user_tier: 사용자 등급 ('bronze', 'silver', 'gold' 중 하나)
        coupon_code: 쿠폰 코드 문자열. None이면 쿠폰 할인이 적용되지 않는다.

    Returns:
        할인이 적용된 최종 가격 (float)

    Raises:
        ValueError: price가 음수인 경우
        KeyError: user_tier가 유효하지 않은 값인 경우 (단, 기본값 0으로 처리)

    Example:
        >>> calculate_discount(10000, 'gold')
        8000.0
        >>> calculate_discount(10000, 'silver', 'SAVE5')
        8500.0
    """
    tier_rates = {"bronze": 0.05, "silver": 0.10, "gold": 0.20}
    rate = tier_rates.get(user_tier, 0)
    if coupon_code and validate_coupon(coupon_code):
        rate += 0.05
    return price * (1 - rate)
```

#### 일괄 Docstring 생성 스크립트

```bash
#!/usr/bin/env bash
# generate-docstrings.sh - 프로젝트 전체의 미문서화 함수를 찾아 Claude에게 docstring 생성을 요청한다.

# docstring이 없는 Python 함수를 찾는다.
UNDOCUMENTED=$(grep -rn "^def \|^    def " src/ --include="*.py" | \
  python3 -c "
import sys
import re

for line in sys.stdin:
    filepath, lineno, content = line.split(':', 2)
    print(f'{filepath}:{lineno}: {content.strip()}')
" | head -20)

echo "문서화가 필요한 함수 목록:"
echo "$UNDOCUMENTED"
echo ""
echo "Claude Code에서 다음 명령어로 일괄 생성하세요:"
echo "claude 'src/ 디렉토리의 모든 Python 파일에서 docstring이 없는 public 함수를 찾아 Google 스타일 docstring을 추가해줘'"
```

---

## 2. 사람과 AI가 함께 일하는 Git Repository 운영 규칙

AI 에이전트가 코드를 수정하고 커밋하는 환경에서는 "누가 어떤 변경을 했는가"를 추적하고, AI의 작업 범위를 명확히 제한하는 것이 중요하다.

### 2.1 AI 커밋 메시지 컨벤션

#### 표준 AI 커밋 메시지 형식

```
<type>(<scope>): <subject>

<body>

Co-Authored-By: Claude <noreply@anthropic.com>
AI-Generated: true
AI-Tool: Claude Code (claude-sonnet-4-5)
AI-Session: <session-id>
```

#### 실제 커밋 예시

```
feat(auth): JWT 갱신 로직에 만료 시간 검증 추가

사용자 토큰이 만료된 후에도 API 호출이 성공하는 버그를 수정.
만료 시간을 30초 여유 있게 검사하여 엣지 케이스를 처리.

- TokenService.refresh()에 isExpiringSoon() 헬퍼 추가
- 만료 임박 토큰에 대한 사전 갱신 로직 구현
- 관련 단위 테스트 3개 추가

Co-Authored-By: Claude <noreply@anthropic.com>
AI-Generated: true
AI-Tool: Claude Code (claude-sonnet-4-5)
```

#### CLAUDE.md에 커밋 규칙 추가

```markdown
## Git 커밋 규칙

### AI 작업 시 필수 포함 사항

모든 AI 생성 커밋에는 다음을 포함해야 한다:

\`\`\`
Co-Authored-By: Claude <noreply@anthropic.com>
\`\`\`

### Commit Type 목록

- feat: 새로운 기능
- fix: 버그 수정
- docs: 문서만 변경
- refactor: 기능 변경 없는 코드 리팩토링
- test: 테스트 추가 또는 수정
- chore: 빌드 프로세스 또는 보조 도구 변경

### 커밋 메시지 작성 방법

커밋하기 전에 반드시 git diff를 확인하고,
변경 내용을 정확히 반영하는 메시지를 작성한다.
자동화된 "auto-commit" 스타일의 모호한 메시지는 금지한다.
```

#### git 훅으로 AI 커밋 메시지 형식 강제

```bash
#!/usr/bin/env bash
# .git/hooks/commit-msg
# AI 생성 커밋에 필수 태그가 있는지 검사한다.

COMMIT_MSG_FILE=$1
COMMIT_MSG=$(cat "$COMMIT_MSG_FILE")

# AI 생성 커밋인지 확인 (환경변수로 표시)
if [[ "${CLAUDE_CODE_SESSION:-}" != "" ]]; then
  # Co-Authored-By 태그가 없으면 자동으로 추가한다.
  if ! grep -q "Co-Authored-By: Claude" "$COMMIT_MSG_FILE"; then
    echo "" >> "$COMMIT_MSG_FILE"
    echo "Co-Authored-By: Claude <noreply@anthropic.com>" >> "$COMMIT_MSG_FILE"
    echo "AI-Generated: true" >> "$COMMIT_MSG_FILE"
  fi
fi
```

---

### 2.2 브랜치 네이밍 규칙

AI가 생성한 브랜치와 사람이 생성한 브랜치를 구별하면 코드 리뷰 시 맥락을 빠르게 파악할 수 있다.

#### 브랜치 네이밍 컨벤션

```
# 사람이 만든 브랜치
feature/user-auth-oauth2
fix/login-redirect-bug
chore/upgrade-node-20

# AI(Claude Code)가 만든 브랜치
ai/feat/user-auth-oauth2
ai/fix/login-redirect-bug
claude/refactor/extract-auth-service

# AI + 사람이 페어로 작업하는 브랜치
pair/feat/payment-integration
```

#### CLAUDE.md에 브랜치 규칙 추가

```markdown
## 브랜치 네이밍 규칙

### AI 단독 작업 브랜치

AI가 독립적으로 작업하는 브랜치는 `ai/` 접두사를 사용한다:

\`\`\`
ai/<type>/<short-description>
\`\`\`

예시:
- `ai/feat/add-user-search`
- `ai/fix/null-pointer-in-auth`
- `ai/refactor/simplify-payment-flow`

### 브랜치 생성 명령어

새 작업을 시작할 때 반드시 새 브랜치를 만든다:

\`\`\`bash
git checkout -b ai/feat/$(date +%Y%m%d)-<description>
\`\`\`

### 직접 main/master에 커밋 금지

AI는 절대 main, master, develop 브랜치에 직접 커밋하지 않는다.
반드시 새 브랜치를 만들고 PR을 통해 머지한다.
```

#### 브랜치 보호 설정 (GitHub CLI)

```bash
# main 브랜치를 보호하고 직접 push를 차단한다.
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["ci/tests","ci/lint"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":1}' \
  --field restrictions='{"users":[],"teams":[]}'

# AI 생성 PR에 자동 레이블을 추가하는 규칙 설정
gh label create "ai-generated" --color "#0075ca" --description "AI 에이전트가 생성한 변경사항"
```

---

### 2.3 PR 템플릿에 AI 사용 여부 표기

#### `.github/PULL_REQUEST_TEMPLATE.md`

```markdown
## 변경 사항 요약

<!-- 이 PR에서 무엇을 변경했는지 간략히 설명하세요 -->

## 변경 유형

- [ ] 새로운 기능 (feat)
- [ ] 버그 수정 (fix)
- [ ] 문서 수정 (docs)
- [ ] 리팩토링 (refactor)
- [ ] 테스트 추가/수정 (test)
- [ ] 기타 (chore)

## AI 도구 사용 여부

- [ ] 이 PR은 AI 도구(Claude Code, Copilot 등)를 사용하여 생성되었습니다.

### AI 사용 시 추가 정보

| 항목 | 내용 |
|------|------|
| 사용 도구 | <!-- 예: Claude Code (claude-sonnet-4-5) --> |
| AI 작업 비율 | <!-- 예: 80% AI / 20% 수동 수정 --> |
| 검토 여부 | <!-- AI 생성 코드를 직접 검토했는지 여부 --> |

## 테스트 방법

```bash
# 변경 사항을 테스트하는 명령어를 작성하세요
npm test
```

## 체크리스트

- [ ] 코드가 프로젝트 컨벤션을 따릅니다
- [ ] 관련 문서를 업데이트했습니다 (README, CHANGELOG 등)
- [ ] AI 생성 코드를 직접 검토하고 이해했습니다
- [ ] 테스트가 추가되었거나 기존 테스트가 통과합니다

## 관련 이슈

Closes #
```

#### AI PR 자동 레이블 워크플로우 (`.github/workflows/label-ai-pr.yml`)

```yaml
name: AI PR 레이블 자동 부착

on:
  pull_request:
    types: [opened, edited]

jobs:
  label-ai-pr:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
    steps:
      - name: PR 본문에서 AI 사용 여부 확인
        id: check-ai
        uses: actions/github-script@v7
        with:
          script: |
            const body = context.payload.pull_request.body || '';
            const isAI = body.includes('[x] 이 PR은 AI 도구') ||
                         body.includes('AI-Generated: true');

            if (isAI) {
              await github.rest.issues.addLabels({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: context.payload.pull_request.number,
                labels: ['ai-generated']
              });
              console.log('AI 생성 레이블 추가 완료');
            }
```

---

### 2.4 .gitignore에 AI 임시 파일 추가

Claude Code와 다른 AI 도구들이 생성하는 임시 파일들을 git에서 제외한다.

#### `.gitignore` AI 관련 항목

```gitignore
# ============================================
# AI 도구 임시 파일 및 설정
# ============================================

# Claude Code
.claude/cache/
.claude/tmp/
.claude/sessions/
*.claude-tmp
*.claude-session

# oh-my-claudecode 상태 파일 (팀 공유 불필요)
.omc/state/
.omc/notepad.md
# 단, plans와 project-memory는 팀 공유를 위해 포함할 수 있음
# .omc/plans/       <- 이 줄을 주석 처리하면 plans를 공유함
# .omc/project-memory.json

# GitHub Copilot
.github/copilot-cache/

# Cursor AI
.cursor/rules/.cursor-tutor
.cursorignore.cache

# Codeium
.codeium/

# AI 생성 임시 파일 패턴
*.ai-draft
*.ai-backup
*.ai-temp
*_ai_generated_*.tmp

# AI 실험 파일 (실제 코드와 구분)
experiments/ai-*/
sandbox/ai-*/

# LLM 응답 캐시
.llm-cache/
llm-responses/
```

#### `.gitignore` 적용 후 확인

```bash
# 현재 추적되고 있는 AI 관련 파일을 확인한다.
git ls-files | grep -E "\.claude|\.omc|\.cursor|\.ai-"

# 이미 추적 중인 파일이 있다면 제거한다.
git rm --cached .omc/state/autopilot-state.json
git rm --cached -r .claude/cache/

# .gitignore 규칙이 올바르게 동작하는지 테스트한다.
git check-ignore -v .omc/state/test.json
```

---

### 2.5 CODEOWNERS와 AI 작업 범위 제한

`CODEOWNERS` 파일로 AI 에이전트가 수정할 수 있는 파일 범위를 명시적으로 제한한다.

#### `.github/CODEOWNERS`

```
# CODEOWNERS - 파일별 코드 소유자 및 리뷰 담당자 지정
# 형식: <파일패턴> <소유자>

# ============================================
# 보안 민감 파일 - AI 단독 수정 금지, 시니어 엔지니어 리뷰 필수
# ============================================
*.env                          @security-team
.env.*                         @security-team
config/secrets.*               @security-team @backend-lead
src/auth/                      @security-team @backend-lead
src/crypto/                    @security-team

# ============================================
# 인프라 및 배포 설정 - DevOps 팀 리뷰 필수
# ============================================
.github/workflows/             @devops-team
docker-compose*.yml            @devops-team
Dockerfile*                    @devops-team
terraform/                     @devops-team
k8s/                           @devops-team

# ============================================
# 데이터베이스 마이그레이션 - DBA 리뷰 필수
# ============================================
migrations/                    @dba-team @backend-lead
db/schema/                     @dba-team

# ============================================
# AI 에이전트가 자유롭게 수정 가능한 영역
# (리뷰어가 지정되지 않은 파일은 AI 작업 허용 범위)
# ============================================
src/components/                @frontend-team
src/utils/                     @frontend-team
tests/unit/                    @qa-team
docs/                          @tech-writers

# ============================================
# Claude Code 설정 파일 - AI 에이전트 소유
# ============================================
CLAUDE.md                      @ai-ops-team
.claude/settings.json          @ai-ops-team
.omc/plans/                    @ai-ops-team
```

#### AI 작업 범위를 CLAUDE.md에 명시

```markdown
## AI 에이전트 작업 제한 범위

### 수정 금지 파일 (절대 변경하지 않는다)

다음 파일들은 CODEOWNERS에 의해 보호되며, AI 에이전트는 수정을 요청받더라도
이 파일들을 직접 편집하지 않고 사람 담당자에게 리뷰를 요청한다:

- `*.env`, `.env.*` - 환경 변수 파일
- `config/secrets.*` - 시크릿 설정
- `src/auth/**` - 인증/인가 로직
- `.github/workflows/**` - CI/CD 파이프라인
- `migrations/**` - 데이터베이스 마이그레이션

### 수정 가능 파일 (자유롭게 작업)

- `src/components/**` - UI 컴포넌트
- `src/utils/**` - 유틸리티 함수
- `tests/unit/**` - 단위 테스트
- `docs/**` - 문서

### 불확실한 경우

파일이 어느 범주에 속하는지 불분명하면 먼저 확인을 요청한다.
"이 파일을 수정해도 되는가요?" 라고 물어보는 것이 더 안전하다.
```

#### PR 리뷰 시 AI 작업 범위 초과 감지 스크립트

```bash
#!/usr/bin/env bash
# check-ai-scope.sh - AI가 수정 금지 파일을 변경했는지 확인한다.

PROTECTED_PATTERNS=(
  "*.env"
  ".env.*"
  "config/secrets*"
  "src/auth/*"
  ".github/workflows/*"
  "migrations/*"
)

# PR에서 변경된 파일 목록을 가져온다.
CHANGED_FILES=$(git diff --name-only origin/main...HEAD)

echo "변경된 파일 목록:"
echo "$CHANGED_FILES"
echo ""

VIOLATIONS=0
for pattern in "${PROTECTED_PATTERNS[@]}"; do
  matches=$(echo "$CHANGED_FILES" | grep -E "$pattern" || true)
  if [[ -n "$matches" ]]; then
    echo "경고: 보호된 파일이 변경되었습니다 (패턴: $pattern):"
    echo "$matches"
    VIOLATIONS=$((VIOLATIONS + 1))
  fi
done

if [[ $VIOLATIONS -gt 0 ]]; then
  echo ""
  echo "총 $VIOLATIONS 개의 보호된 파일 변경이 감지되었습니다."
  echo "AI 생성 PR인 경우 담당 팀의 추가 리뷰가 필요합니다."
  exit 1
fi

echo "보호된 파일 변경 없음. 검사 통과."
```

#### CI에 범위 검사 통합 (`.github/workflows/ai-scope-check.yml`)

```yaml
name: AI 작업 범위 검사

on:
  pull_request:
    branches: [main, develop]

jobs:
  scope-check:
    runs-on: ubuntu-latest
    # AI 생성 PR에만 적용한다.
    if: contains(github.event.pull_request.labels.*.name, 'ai-generated')
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: AI 작업 범위 초과 검사
        run: |
          chmod +x scripts/check-ai-scope.sh
          bash scripts/check-ai-scope.sh
```

---

## 요약

| 항목 | 설정 위치 | 핵심 효과 |
|------|-----------|-----------|
| 문서 자동 업데이트 | `.claude/settings.json` 훅 | 코드 변경 시 CHANGELOG 자동 갱신 |
| 문서 동기화 규칙 | `CLAUDE.md` | AI가 코드 변경 시 관련 문서도 수정 |
| JSDoc 생성 패턴 | 프롬프트 템플릿 | 일관된 형식의 코드 주석 |
| 커밋 컨벤션 | `CLAUDE.md` + git 훅 | AI 생성 커밋 추적 가능 |
| 브랜치 규칙 | `CLAUDE.md` | AI 작업 브랜치 식별 |
| PR 템플릿 | `.github/PULL_REQUEST_TEMPLATE.md` | AI 사용 여부 공개 |
| gitignore | `.gitignore` | AI 임시 파일 제외 |
| 작업 범위 제한 | `.github/CODEOWNERS` + CI | 민감 파일 보호 |
