# 훅과 정책으로 워크플로우 고정하기

> 커리큘럼 04-CH02 | Claude Code 훅 시스템 완전 가이드

---

## 목차

- [훅이 프롬프트보다 강한 이유](#훅이-프롬프트보다-강한-이유)
- [settings.json에서 훅 설정하기](#settingsjson에서-훅-설정하기)
- [PreToolUse Hook: 위험한 명령 차단](#pretooluse-hook-위험한-명령-차단)
- [PostToolUse Hook: 자동 포맷 및 테스트 실행](#posttooluse-hook-자동-포맷-및-테스트-실행)
- [Notification Hook](#notification-hook)
- [팀 개발 규칙 자동화](#팀-개발-규칙-자동화)
- [훅 디버깅 방법](#훅-디버깅-방법)

---

## 훅이 프롬프트보다 강한 이유

프롬프트로 AI에게 규칙을 알려주면 AI는 "최선을 다해" 따른다. 그러나 확률적으로 따르기 때문에 예외가 생긴다.

```
# 이 방식은 AI가 "때로는" 따르지 않을 수 있다
프롬프트: "절대로 git push --force를 실행하지 마세요"
```

훅은 다르다. 훅은 **AI가 특정 도구를 호출하기 전후에 운영체제 수준에서 실행되는 스크립트**다.

```
훅: PreToolUse → bash 스크립트 실행 → exit code 1이면 도구 실행 차단
결과: AI의 의지와 무관하게 명령이 차단됨
```

### 프롬프트 vs 훅 비교

| 비교 항목 | 프롬프트 기반 규칙 | 훅 기반 규칙 |
|-----------|-------------------|-------------|
| 준수율 | ~95% (확률적) | 100% (확정적) |
| 우회 가능성 | 있음 (특이한 문맥) | 없음 (OS 수준 차단) |
| 실행 시점 | AI가 판단 | AI 판단 이전에 실행 |
| 감사 로그 | 없음 | 스크립트로 기록 가능 |
| 팀 적용 | 개인 프롬프트 관리 | settings.json 버전 관리 |

훅은 "AI에게 부탁"하는 것이 아니라 "시스템에 정책을 심는 것"이다.

---

## settings.json에서 훅 설정하기

훅 설정은 `.claude/settings.json` (프로젝트) 또는 `~/.claude/settings.json` (글로벌)에 작성한다.

### 기본 구조

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "도구이름 또는 패턴",
        "hooks": [
          {
            "type": "command",
            "command": "실행할 bash 명령어"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "도구이름 또는 패턴",
        "hooks": [
          {
            "type": "command",
            "command": "실행할 bash 명령어"
          }
        ]
      }
    ],
    "Notification": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "실행할 bash 명령어"
          }
        ]
      }
    ]
  }
}
```

### 훅 이벤트 종류

| 이벤트 | 실행 시점 | 용도 |
|--------|-----------|------|
| `PreToolUse` | 도구 실행 직전 | 차단, 검증, 로깅 |
| `PostToolUse` | 도구 실행 직후 | 포맷, 테스트, 알림 |
| `Notification` | Claude가 알림 전송 시 | 외부 알림 연동 |

### matcher 패턴

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{ "type": "command", "command": "..." }]
      },
      {
        "matcher": "Write|Edit",
        "hooks": [{ "type": "command", "command": "..." }]
      },
      {
        "matcher": ".*",
        "hooks": [{ "type": "command", "command": "..." }]
      }
    ]
  }
}
```

- `"Bash"` : Bash 도구 호출 시에만
- `"Write|Edit"` : Write 또는 Edit 도구 호출 시
- `".*"` : 모든 도구 호출 시

### 훅 입력 데이터 (stdin)

훅 스크립트는 stdin으로 JSON 데이터를 받는다.

```json
{
  "tool_name": "Bash",
  "tool_input": {
    "command": "rm -rf /tmp/cache",
    "description": "캐시 삭제"
  },
  "session_id": "abc123",
  "cwd": "/Users/dev/my-project"
}
```

PostToolUse에서는 `tool_response`도 포함된다:

```json
{
  "tool_name": "Bash",
  "tool_input": { "command": "ls -la" },
  "tool_response": {
    "output": "total 48\ndrwxr-xr-x ...",
    "exit_code": 0
  }
}
```

---

## PreToolUse Hook: 위험한 명령 차단

### rm -rf 차단 훅

**`.claude/settings.json`:**

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "bash ~/.claude/hooks/block-dangerous-commands.sh"
          }
        ]
      }
    ]
  }
}
```

**`~/.claude/hooks/block-dangerous-commands.sh`:**

```bash
#!/bin/bash
# 위험한 명령어를 차단하는 PreToolUse 훅

# stdin에서 JSON 읽기
INPUT=$(cat)

# 명령어 추출
COMMAND=$(echo "$INPUT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('tool_input', {}).get('command', ''))
")

# 차단할 패턴 목록
BLOCKED_PATTERNS=(
  "rm -rf /"
  "rm -rf \$HOME"
  "rm -rf ~"
  "> /dev/sda"
  "dd if=/dev/zero"
  "chmod -R 777 /"
  ":(){ :|:& };:"  # fork bomb
)

for pattern in "${BLOCKED_PATTERNS[@]}"; do
  if echo "$COMMAND" | grep -qF "$pattern"; then
    echo "차단됨: 위험한 명령어가 감지됐습니다." >&2
    echo "명령어: $COMMAND" >&2
    echo "패턴: $pattern" >&2

    # 감사 로그 기록
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] BLOCKED: $COMMAND" >> ~/.claude/logs/blocked-commands.log

    exit 1  # exit 1 = 도구 실행 차단
  fi
done

exit 0  # exit 0 = 도구 실행 허용
```

**테스트:**

```bash
# 훅 스크립트 실행 권한 부여
chmod +x ~/.claude/hooks/block-dangerous-commands.sh

# 로그 디렉토리 생성
mkdir -p ~/.claude/logs

# 수동 테스트
echo '{"tool_name":"Bash","tool_input":{"command":"rm -rf /"}}' | \
  bash ~/.claude/hooks/block-dangerous-commands.sh
# 출력: 차단됨: 위험한 명령어가 감지됐습니다.
# exit code: 1

echo '{"tool_name":"Bash","tool_input":{"command":"ls -la"}}' | \
  bash ~/.claude/hooks/block-dangerous-commands.sh
# exit code: 0 (허용됨)
```

### git push --force 차단 훅

**`.claude/settings.json`에 추가:**

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/block-force-push.sh"
          }
        ]
      }
    ]
  }
}
```

**`.claude/hooks/block-force-push.sh`:**

```bash
#!/bin/bash
# git push --force 차단 훅

INPUT=$(cat)

COMMAND=$(echo "$INPUT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('tool_input', {}).get('command', ''))
")

# git push --force 또는 git push -f 감지
if echo "$COMMAND" | grep -qE "git push.*(--force|-f)"; then

  # main 또는 master 브랜치에 대한 force push는 완전 차단
  if echo "$COMMAND" | grep -qE "(main|master)"; then
    echo "[BLOCKED] main/master 브랜치에 force push는 절대 허용되지 않습니다." >&2
    echo "명령어: $COMMAND" >&2
    exit 1
  fi

  # 다른 브랜치는 경고 후 확인 요청
  echo "[WARNING] force push가 감지됐습니다: $COMMAND" >&2
  echo "force push는 협업 브랜치에서 다른 팀원의 작업을 덮어쓸 수 있습니다." >&2
  echo "계속하려면 Claude에게 'force push 허용' 이라고 명시적으로 말해주세요." >&2

  # 환경 변수로 명시적 허용 여부 확인
  if [ "$ALLOW_FORCE_PUSH" != "yes" ]; then
    exit 1
  fi
fi

exit 0
```

**`.claude/settings.json` 전체 예시 (여러 훅 조합):**

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/block-dangerous-commands.sh"
          },
          {
            "type": "command",
            "command": "bash .claude/hooks/block-force-push.sh"
          },
          {
            "type": "command",
            "command": "bash .claude/hooks/log-all-commands.sh"
          }
        ]
      }
    ]
  }
}
```

여러 훅이 배열로 나열되면 순서대로 실행되고, 하나라도 exit 1을 반환하면 도구 실행이 차단된다.

---

## PostToolUse Hook: 자동 포맷 및 테스트 실행

### 파일 저장 후 자동 포맷

**`.claude/settings.json`:**

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/auto-format.sh"
          }
        ]
      }
    ]
  }
}
```

**`.claude/hooks/auto-format.sh`:**

```bash
#!/bin/bash
# 파일 저장 후 자동 포맷 실행

INPUT=$(cat)

# 저장된 파일 경로 추출
FILE_PATH=$(echo "$INPUT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
tool_input = data.get('tool_input', {})
# Write 도구는 file_path, Edit 도구도 file_path
print(tool_input.get('file_path', ''))
")

if [ -z "$FILE_PATH" ]; then
  exit 0
fi

echo "포맷 실행: $FILE_PATH"

# 확장자별 포맷터 선택
case "$FILE_PATH" in
  *.ts|*.tsx)
    if command -v npx &>/dev/null; then
      npx prettier --write "$FILE_PATH" 2>/dev/null
      npx eslint --fix "$FILE_PATH" 2>/dev/null
      echo "TypeScript 포맷 완료: $FILE_PATH"
    fi
    ;;
  *.js|*.jsx)
    if command -v npx &>/dev/null; then
      npx prettier --write "$FILE_PATH" 2>/dev/null
      echo "JavaScript 포맷 완료: $FILE_PATH"
    fi
    ;;
  *.py)
    if command -v black &>/dev/null; then
      black "$FILE_PATH" 2>/dev/null
      echo "Python 포맷 완료: $FILE_PATH"
    fi
    if command -v isort &>/dev/null; then
      isort "$FILE_PATH" 2>/dev/null
    fi
    ;;
  *.go)
    if command -v gofmt &>/dev/null; then
      gofmt -w "$FILE_PATH"
      echo "Go 포맷 완료: $FILE_PATH"
    fi
    ;;
  *.json)
    if command -v jq &>/dev/null; then
      tmp=$(mktemp)
      jq . "$FILE_PATH" > "$tmp" && mv "$tmp" "$FILE_PATH"
      echo "JSON 포맷 완료: $FILE_PATH"
    fi
    ;;
esac

exit 0  # PostToolUse는 exit code가 실행을 차단하지 않음
```

### 파일 저장 후 관련 테스트 자동 실행

**`.claude/hooks/auto-test.sh`:**

```bash
#!/bin/bash
# 소스 파일 변경 후 관련 테스트 자동 실행

INPUT=$(cat)

FILE_PATH=$(echo "$INPUT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('tool_input', {}).get('file_path', ''))
")

# 테스트 파일 자체가 변경된 경우 제외
if echo "$FILE_PATH" | grep -qE "(\.test\.|\.spec\.|__tests__)"; then
  exit 0
fi

# 소스 파일에 대응하는 테스트 파일 찾기
find_test_file() {
  local src_file="$1"
  local dir=$(dirname "$src_file")
  local base=$(basename "$src_file" | sed 's/\.[^.]*$//')
  local ext="${src_file##*.}"

  # 테스트 파일 후보 경로들
  local candidates=(
    "$dir/__tests__/$base.test.$ext"
    "$dir/__tests__/$base.spec.$ext"
    "$dir/$base.test.$ext"
    "$dir/$base.spec.$ext"
  )

  for candidate in "${candidates[@]}"; do
    if [ -f "$candidate" ]; then
      echo "$candidate"
      return
    fi
  done
}

TEST_FILE=$(find_test_file "$FILE_PATH")

if [ -n "$TEST_FILE" ]; then
  echo "관련 테스트 실행: $TEST_FILE"

  # 프로젝트 루트로 이동
  PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo ".")
  cd "$PROJECT_ROOT" || exit 0

  # 테스트 실행 (백그라운드로 실행해서 Claude 응답을 블록하지 않음)
  if [ -f "package.json" ]; then
    # Jest 또는 Vitest 감지
    if grep -q '"jest"' package.json; then
      npx jest "$TEST_FILE" --no-coverage 2>&1 | tail -20 &
    elif grep -q '"vitest"' package.json; then
      npx vitest run "$TEST_FILE" 2>&1 | tail -20 &
    fi
  elif [ -f "pytest.ini" ] || [ -f "pyproject.toml" ]; then
    pytest "$TEST_FILE" -v 2>&1 | tail -20 &
  fi
fi

exit 0
```

**`.claude/settings.json` PostToolUse 완전 예시:**

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/auto-format.sh"
          },
          {
            "type": "command",
            "command": "bash .claude/hooks/auto-test.sh"
          }
        ]
      },
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/log-bash-output.sh"
          }
        ]
      }
    ]
  }
}
```

---

## Notification Hook

### Slack 알림 훅

Claude Code가 중요한 알림을 보낼 때 Slack으로 전달한다.

**`.claude/settings.json`:**

```json
{
  "hooks": {
    "Notification": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash ~/.claude/hooks/notify-slack.sh"
          }
        ]
      }
    ]
  }
}
```

**`~/.claude/hooks/notify-slack.sh`:**

```bash
#!/bin/bash
# Claude Code 알림을 Slack으로 전달

INPUT=$(cat)

# 알림 내용 추출
MESSAGE=$(echo "$INPUT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('message', ''))
")

TITLE=$(echo "$INPUT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('title', 'Claude Code 알림'))
")

# Slack Webhook URL (환경 변수에서 읽기)
SLACK_WEBHOOK="${SLACK_WEBHOOK_URL}"

if [ -z "$SLACK_WEBHOOK" ]; then
  exit 0  # Webhook이 없으면 무시
fi

# Slack 메시지 전송
curl -s -X POST "$SLACK_WEBHOOK" \
  -H 'Content-type: application/json' \
  -d "{
    \"text\": \"*${TITLE}*\n${MESSAGE}\",
    \"blocks\": [
      {
        \"type\": \"section\",
        \"text\": {
          \"type\": \"mrkdwn\",
          \"text\": \"*${TITLE}*\n${MESSAGE}\"
        }
      }
    ]
  }" &

exit 0
```

**환경 변수 설정:**

```bash
# ~/.zshrc 또는 ~/.bashrc에 추가
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/T.../B.../..."
```

### 데스크탑 알림 훅 (macOS)

```bash
#!/bin/bash
# ~/.claude/hooks/notify-desktop.sh

INPUT=$(cat)

MESSAGE=$(echo "$INPUT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('message', 'Claude Code 완료')[:100])
")

# macOS 알림 센터
osascript -e "display notification \"$MESSAGE\" with title \"Claude Code\" sound name \"Glass\""

exit 0
```

---

## 팀 개발 규칙 자동화

### 완성된 팀 훅 워크플로우

실제 팀 프로젝트에서 사용하는 `.claude/settings.json` 완전판이다.

**`.claude/settings.json`:**

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/pre-bash.sh"
          }
        ]
      },
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/pre-write.sh"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/post-write.sh"
          }
        ]
      }
    ],
    "Notification": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/notify.sh"
          }
        ]
      }
    ]
  }
}
```

**`.claude/hooks/pre-bash.sh`:**

```bash
#!/bin/bash
# 모든 Bash 명령어 실행 전 검사

INPUT=$(cat)

COMMAND=$(echo "$INPUT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('tool_input', {}).get('command', ''))
")

# 1. 위험한 명령어 차단
DANGEROUS_PATTERNS=(
  "rm -rf /"
  "rm -rf \$HOME"
  "git push --force.*main"
  "git push --force.*master"
  "git push -f.*main"
  "git push -f.*master"
  "DROP TABLE"
  "DROP DATABASE"
)

for pattern in "${DANGEROUS_PATTERNS[@]}"; do
  if echo "$COMMAND" | grep -qiE "$pattern"; then
    echo "[BLOCKED] 위험한 명령 차단: $pattern" >&2
    echo "명령어: $COMMAND" >&2

    # 감사 로그
    LOG_DIR=".claude/logs"
    mkdir -p "$LOG_DIR"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] BLOCKED | $COMMAND" >> "$LOG_DIR/blocked.log"

    exit 1
  fi
done

# 2. 모든 명령어 감사 로그
LOG_DIR=".claude/logs"
mkdir -p "$LOG_DIR"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] EXEC | $COMMAND" >> "$LOG_DIR/audit.log"

exit 0
```

**`.claude/hooks/pre-write.sh`:**

```bash
#!/bin/bash
# 파일 쓰기 전 검사 (보안 파일 보호)

INPUT=$(cat)

FILE_PATH=$(echo "$INPUT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('tool_input', {}).get('file_path', ''))
")

# 쓰기 금지 파일 패턴
PROTECTED_PATTERNS=(
  "\.env$"
  "\.env\.production$"
  "secrets\."
  "credentials\."
  "private-key"
  "\.pem$"
  "\.key$"
)

for pattern in "${PROTECTED_PATTERNS[@]}"; do
  if echo "$FILE_PATH" | grep -qE "$pattern"; then
    echo "[BLOCKED] 보호된 파일 수정 시도: $FILE_PATH" >&2
    echo "이 파일은 Claude Code가 수정할 수 없도록 보호됩니다." >&2
    exit 1
  fi
done

exit 0
```

**`.claude/hooks/post-write.sh`:**

```bash
#!/bin/bash
# 파일 저장 후 lint-staged + prettier 자동 실행

INPUT=$(cat)

FILE_PATH=$(echo "$INPUT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('tool_input', {}).get('file_path', ''))
")

if [ -z "$FILE_PATH" ] || [ ! -f "$FILE_PATH" ]; then
  exit 0
fi

PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
if [ -z "$PROJECT_ROOT" ]; then
  exit 0
fi

cd "$PROJECT_ROOT" || exit 0

# Prettier 자동 실행
if [ -f ".prettierrc" ] || [ -f ".prettierrc.json" ] || [ -f "prettier.config.js" ]; then
  case "$FILE_PATH" in
    *.ts|*.tsx|*.js|*.jsx|*.css|*.scss|*.json|*.md)
      npx prettier --write "$FILE_PATH" 2>/dev/null
      echo "Prettier: $FILE_PATH 포맷 완료"
      ;;
  esac
fi

# ESLint 자동 수정 (TypeScript/JavaScript만)
if [ -f ".eslintrc.js" ] || [ -f ".eslintrc.json" ] || [ -f "eslint.config.js" ]; then
  case "$FILE_PATH" in
    *.ts|*.tsx|*.js|*.jsx)
      npx eslint --fix "$FILE_PATH" 2>/dev/null
      echo "ESLint: $FILE_PATH 수정 완료"
      ;;
  esac
fi

# Stylelint 자동 수정 (CSS/SCSS만)
if command -v npx &>/dev/null && [ -f ".stylelintrc.json" ]; then
  case "$FILE_PATH" in
    *.css|*.scss)
      npx stylelint --fix "$FILE_PATH" 2>/dev/null
      echo "Stylelint: $FILE_PATH 수정 완료"
      ;;
  esac
fi

exit 0
```

**`.claude/hooks/notify.sh`:**

```bash
#!/bin/bash
# 알림 훅 - 데스크탑 + 선택적 Slack 전송

INPUT=$(cat)

MESSAGE=$(echo "$INPUT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('message', '')[:200])
" 2>/dev/null)

# macOS 데스크탑 알림
if command -v osascript &>/dev/null; then
  osascript -e "display notification \"$MESSAGE\" with title \"Claude Code\"" 2>/dev/null
fi

# Linux 데스크탑 알림
if command -v notify-send &>/dev/null; then
  notify-send "Claude Code" "$MESSAGE" 2>/dev/null
fi

exit 0
```

### 훅 디렉토리 초기화 스크립트

팀에 새로 합류한 팀원이 훅을 바로 사용할 수 있도록 초기화 스크립트를 제공한다.

**`.claude/hooks/install.sh`:**

```bash
#!/bin/bash
# Claude Code 훅 초기화 스크립트

echo "Claude Code 훅 설치 중..."

HOOKS_DIR=".claude/hooks"
LOGS_DIR=".claude/logs"

# 디렉토리 생성
mkdir -p "$HOOKS_DIR" "$LOGS_DIR"

# 실행 권한 부여
chmod +x "$HOOKS_DIR"/*.sh

# .gitignore에 로그 디렉토리 추가
if ! grep -q ".claude/logs" .gitignore 2>/dev/null; then
  echo ".claude/logs/" >> .gitignore
  echo ".gitignore에 .claude/logs/ 추가됨"
fi

echo "설치 완료!"
echo ""
echo "현재 활성화된 훅:"
echo "  PreToolUse  → pre-bash.sh, pre-write.sh"
echo "  PostToolUse → post-write.sh (prettier, eslint 자동 실행)"
echo "  Notification → notify.sh (데스크탑 알림)"
```

**`package.json`에 초기화 스크립트 등록:**

```json
{
  "scripts": {
    "claude:setup": "bash .claude/hooks/install.sh",
    "postinstall": "npm run claude:setup"
  }
}
```

---

## 훅 디버깅 방법

### 1. 훅 스크립트 직접 실행

Claude Code를 통하지 않고 훅 스크립트를 직접 테스트한다.

```bash
# PreToolUse 훅 테스트 (Bash 도구 시뮬레이션)
echo '{
  "tool_name": "Bash",
  "tool_input": {
    "command": "rm -rf /tmp/test",
    "description": "테스트 삭제"
  },
  "session_id": "test-session",
  "cwd": "/Users/dev/my-project"
}' | bash .claude/hooks/pre-bash.sh

echo "Exit code: $?"

# PostToolUse 훅 테스트 (Write 도구 시뮬레이션)
echo '{
  "tool_name": "Write",
  "tool_input": {
    "file_path": "/Users/dev/my-project/src/test.ts",
    "content": "export const x = 1;"
  },
  "tool_response": {
    "success": true
  }
}' | bash .claude/hooks/post-write.sh

echo "Exit code: $?"
```

### 2. 훅 로그 확인

```bash
# 차단된 명령어 로그 확인
cat .claude/logs/blocked.log

# 전체 감사 로그 확인
cat .claude/logs/audit.log

# 실시간 로그 모니터링
tail -f .claude/logs/audit.log

# 오늘 차단된 명령어만 필터링
grep "$(date +%Y-%m-%d)" .claude/logs/blocked.log
```

### 3. 훅 비활성화 방법

특정 상황에서 훅을 일시적으로 비활성화해야 할 때:

```bash
# 환경 변수로 훅 우회 (훅 스크립트에서 지원하는 경우)
ALLOW_FORCE_PUSH=yes claude

# settings.json에서 특정 훅 주석처리 (JSON은 주석 불가, 훅을 빈 배열로 교체)
# "hooks": []

# 훅 스크립트를 임시로 항상 성공하도록 변경
echo '#!/bin/bash\nexit 0' > .claude/hooks/pre-bash.sh
```

### 4. 훅 stderr/stdout 동작

```
stdout (exit 0): Claude에게 정보 전달 (표시될 수 있음)
stderr (exit 0): Claude 터미널에 경고 출력
stderr (exit 1): Claude 터미널에 에러 출력 + 도구 실행 차단
stdout (exit 1): 무시됨
```

```bash
#!/bin/bash
# 올바른 출력 방법
echo "이 메시지는 Claude에게 전달됩니다" # stdout
echo "이 메시지는 터미널에 경고로 표시됩니다" >&2  # stderr

exit 0  # 허용
# 또는
exit 1  # 차단
```

### 5. Python으로 JSON 파싱 대안

bash에서 JSON을 파싱할 때 `python3` 외에 `jq`를 사용할 수도 있다.

```bash
# python3 방식
COMMAND=$(echo "$INPUT" | python3 -c "import json,sys; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))")

# jq 방식 (jq가 설치된 경우)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // ""')

# node.js 방식
COMMAND=$(echo "$INPUT" | node -e "let d='';process.stdin.on('data',c=>d+=c).on('end',()=>console.log(JSON.parse(d).tool_input?.command||''))")
```

### 6. 훅 체인 디버깅

여러 훅이 연결된 경우 각 훅의 실행 순서를 추적한다.

```bash
#!/bin/bash
# 디버그 모드 훅 래퍼
# .claude/hooks/debug-wrapper.sh

HOOK_NAME="$1"
HOOK_SCRIPT="$2"

INPUT=$(cat)

echo "[DEBUG] 훅 시작: $HOOK_NAME at $(date)" >&2
echo "[DEBUG] 입력: $(echo "$INPUT" | python3 -m json.tool 2>/dev/null || echo "$INPUT")" >&2

echo "$INPUT" | bash "$HOOK_SCRIPT"
EXIT_CODE=$?

echo "[DEBUG] 훅 종료: $HOOK_NAME, exit_code=$EXIT_CODE" >&2

exit $EXIT_CODE
```

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "bash .claude/hooks/debug-wrapper.sh pre-bash .claude/hooks/pre-bash.sh"
          }
        ]
      }
    ]
  }
}
```

---

## 요약

| 훅 유형 | 실행 시점 | exit 0 | exit 1 |
|---------|-----------|--------|--------|
| `PreToolUse` | 도구 실행 전 | 허용 | 차단 |
| `PostToolUse` | 도구 실행 후 | 계속 | 경고만 (차단 안 함) |
| `Notification` | 알림 발생 시 | 계속 | 알림 억제 |

훅의 핵심은 **"AI에게 부탁하는 것"에서 "시스템에 정책을 심는 것"으로 전환**이다. 팀의 개발 규칙을 훅으로 자동화하면 코드 리뷰에서 "이거 prettier 돌렸어요?" 같은 질문이 사라진다.

첫 번째 훅으로는 PostToolUse에서 prettier 자동 실행부터 시작하는 것을 권장한다. 효과가 즉각적이고 팀원 모두가 체감할 수 있다.
