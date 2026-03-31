# Claude Code Best Practice — 실전편 (Git/PR, Debugging, Utilities, Daily)

> **출처**: [claude-code-best-practice](https://github.com/shanraisshan/claude-code-best-practice) 레포지토리 기반.
> Boris Cherny, Cat Wu, Dex 등의 공식 팁.

---

## 목차

- [Git / PR (5개)](#git--pr-5개)
  - [51. PR을 작게 유지하라](#51-pr을-작게-유지하라)
  - [52. 항상 squash merge](#52-항상-squash-merge)
  - [53. 시간당 1회 이상 commit](#53-시간당-1회-이상-commit)
  - [54. @claude 태그로 PR에 lint 규칙 자동 생성](#54-claude-태그로-pr에-lint-규칙-자동-생성)
  - [55. `/code-review`로 multi-agent PR 분석](#55-code-review로-multi-agent-pr-분석)
- [Debugging (7개)](#debugging-7개)
  - [56. 스크린샷을 Claude에게 공유](#56-스크린샷을-claude에게-공유)
  - [57. Chrome MCP로 콘솔 로그 직접 확인](#57-chrome-mcp로-콘솔-로그-직접-확인)
  - [58. 터미널을 background task로 실행](#58-터미널을-background-task로-실행)
  - [59. `/doctor`로 진단](#59-doctor로-진단)
  - [60. compaction 에러 해결](#60-compaction-에러-해결)
  - [61. Cross-model로 QA](#61-cross-model로-qa)
  - [62. Agentic search (glob + grep) >> RAG](#62-agentic-search-glob--grep--rag)
- [Utilities (5개)](#utilities-5개)
  - [63. iTerm/Ghostty/tmux >> IDE 터미널](#63-itermghostytmux--ide-터미널)
  - [64. 음성 프롬프팅 (Voice Dictation)](#64-음성-프롬프팅-voice-dictation)
  - [65. claude-code-hooks 커뮤니티](#65-claude-code-hooks-커뮤니티)
  - [66. Status line으로 상태 인지](#66-status-line으로-상태-인지)
  - [67. settings.json 기능 탐색](#67-settingsjson-기능-탐색)
- [Daily (3개)](#daily-3개)
  - [68. 매일 업데이트하고 changelog 읽기](#68-매일-업데이트하고-changelog-읽기)
  - [69. Reddit 커뮤니티 팔로우](#69-reddit-커뮤니티-팔로우)
  - [70. 공식 소스 팔로우 (X/Twitter)](#70-공식-소스-팔로우-xtwitter)

---

## Git / PR (5개)

### 51. PR을 작게 유지하라

Boris의 하루: **141개 PR, 총 45,032줄 변경, PR 중간값 118줄**.

```
| 지표 | 값     | 의미                          |
|------|--------|-------------------------------|
| p50  | 118줄  | PR 절반이 118줄 이하          |
| p90  | 498줄  | 90%가 500줄 미만              |
| p99  | 2,978줄| 3천줄 넘는 PR은 1~2개뿐       |
```

작은 PR의 장점:
- 리뷰가 쉽다
- merge conflict 위험 감소
- 문제 발생 시 revert가 간단

### 52. 항상 squash merge

```bash
# squash merge: 브랜치의 모든 커밋을 하나로 합쳐서 머지
git merge --squash feature-branch
```

장점:
- 깔끔한 linear history
- PR 1개 = commit 1개 → `git revert` 한 번으로 전체 기능 롤백
- `git bisect`가 단순해짐
- AI 워크플로우에서 나오는 "fix lint", "try this" 같은 노이즈 커밋 제거

> Boris: "141 PRs, always squashed, median 118 lines"

### 53. 시간당 1회 이상 commit

```
# 작업 완료 즉시 커밋하는 습관
"이 기능 구현 완료했으면 커밋해줘"
```

자주 커밋하면:
- 되돌릴 지점이 많아짐
- 작업 손실 최소화
- Claude가 git history를 참고할 수 있음

### 54. @claude 태그로 PR에 lint 규칙 자동 생성

GitHub에 Claude Code app을 설치하면, PR 코멘트에서 `@claude`를 태그할 수 있다.

```
# 동료의 PR에서:
@claude 이 PR의 패턴을 기반으로 CLAUDE.md에 lint 규칙을 추가해줘
```

반복되는 리뷰 피드백을 자동화하는 방법. Boris는 이를 "Compounding Engineering"이라 부른다.

> Boris (Pragmatic Engineer 팟캐스트): "tag @claude on a coworker's PR to auto-generate lint rules for recurring review feedback — automate yourself out of code review"

### 55. `/code-review`로 multi-agent PR 분석

```
/code-review
```

agent 팀이 PR을 분석한다:
- 버그 탐지
- 보안 취약점 스캔
- 리그레션 확인

Anthropic 내부에서 먼저 사용 — 엔지니어당 코드 출력이 연 200% 증가했고, 리뷰가 병목이 되자 만들어졌다.

> Boris: "I've been using it for a few weeks and found it catches many real bugs I would not have noticed otherwise"

---

## Debugging (7개)

### 56. 스크린샷을 Claude에게 공유

시각적 이슈가 있을 때 텍스트로 설명하지 말고 **스크린샷**을 준다.

```
# macOS:
Cmd+Shift+4 로 캡처 → Claude에게 경로 전달
"이 스크린샷을 보고 UI 문제를 진단해줘: /tmp/screenshot.png"
```

Claude는 이미지를 직접 읽을 수 있다 (multimodal).

### 57. Chrome MCP로 콘솔 로그 직접 확인

프론트엔드 작업 시 Claude가 직접 브라우저를 보게 한다.

```json
// .mcp.json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp"]
    }
  }
}
```

사용 가능한 MCP:
- **Claude in Chrome**: 실제 Chrome 브라우저 연결 — 콘솔, 네트워크, DOM 검사
- **Playwright**: 자동화된 브라우저 테스트
- **Chrome DevTools**: DevTools Protocol 직접 접근

> Boris: "The most important tip: give Claude a way to verify its output. Give Claude a browser and it will iterate until it looks good."

### 58. 터미널을 background task로 실행

로그를 봐야 하는 서버를 background로 실행하면 로그 가시성이 좋아진다.

```
"app.py를 background task로 실행해줘. 로그를 보면서 문제를 진단하자."
```

Claude가 서버를 foreground로 실행하면 터미널이 잠겨서 다른 작업을 못 한다.

### 59. `/doctor`로 진단

설치, 인증, 설정 문제가 의심될 때 첫 번째로 시도할 것.

```
/doctor
```

Claude Code 설치 상태, 인증, MCP 연결, 설정 파일 등을 자동 진단.

### 60. compaction 에러 해결

context가 너무 클 때 compaction이 실패할 수 있다.

```
# 해결 방법:
/model
# → 1M context 모델 선택 (claude-opus-4-6 등)

/compact
# → 이제 더 큰 context에서 compaction 가능
```

### 61. Cross-model로 QA

구현은 하나의 모델로, QA는 다른 모델로.

```
# Claude (Opus)로 구현
# → 결과를 Codex CLI로 리뷰
codex "이 diff를 리뷰해줘: $(git diff)"
```

같은 모델의 같은 세션에서 자기 코드를 리뷰하는 것보다, 다른 모델이나 다른 세션에서 리뷰하는 것이 효과적 (test time compute).

### 62. Agentic search (glob + grep) >> RAG

Claude Code 팀은 벡터 데이터베이스를 시도했다가 **포기**했다.

이유:
- 코드가 빠르게 변해서 임베딩이 금방 stale해짐
- 파일 권한 관리가 복잡
- glob + grep의 agentic search가 더 정확하고 최신 상태를 반영

```
# Claude가 내부적으로 하는 것:
Glob("**/*.py")  → 파일 목록 탐색
Grep("DiaConfig") → 코드 내 패턴 검색
Read("dia/config.py") → 파일 직접 읽기
```

> Boris (Pragmatic Engineer 팟캐스트): "agentic search beats RAG — we tried and discarded vector databases"

---

## Utilities (5개)

### 63. iTerm/Ghostty/tmux >> IDE 터미널

Claude Code 팀은 IDE 터미널보다 **독립 터미널**을 선호한다.

```
# 추천 터미널:
- Ghostty: synchronized rendering, 24-bit color, unicode 지원
- iTerm2: macOS에서 가장 인기, 알림 지원
- tmux: 여러 세션 관리, 탭/패널 분할
```

여러 Claude를 동시에 돌릴 때 tmux가 특히 유용:
```bash
tmux new-session -s dia-main
tmux new-window -t dia-main -n feature-a
tmux new-window -t dia-main -n feature-b
```

### 64. 음성 프롬프팅 (Voice Dictation)

말하는 속도가 타이핑의 3배. 프롬프트가 더 상세해진다.

```
# 방법 1: Claude Code 내장
/voice
# → Space bar 누르고 말하기

# 방법 2: macOS 시스템
# fn 두 번 누르기 → 받아쓰기 시작

# 방법 3: Wispr Flow (서드파티)
# 어디서든 음성 입력 가능
```

> Boris: "Fun fact: I do most of my coding by speaking to Claude, rather than typing."

### 65. claude-code-hooks 커뮤니티

[claude-code-hooks](https://github.com/shanraisshan/claude-code-hooks) 레포지토리에서 커뮤니티가 만든 hook 예제를 참고할 수 있다.

### 66. Status line으로 상태 인지

```
/statusline
```

Claude가 `.bashrc`/`.zshrc`를 분석하여 맞춤 status line을 생성한다.
표시 항목: 모델명, context 사용량, 현재 branch, 비용 등.

context 사용량이 한눈에 보이므로 `/compact` 타이밍을 놓치지 않는다.

### 67. settings.json 기능 탐색

```json
{
  "model": "opus",
  "language": "korean",
  "alwaysThinkingEnabled": true,
  "plansDirectory": "./plans",
  "spinnerVerbs": ["분석 중", "코딩 중", "검증 중"]
}
```

60+ 설정과 100+ 환경변수가 있다. `env` 필드로 환경변수를 설정하면 wrapper script가 필요 없다.

> Boris: "with 37 settings and 84 environment variables, there's a good chance any behavior you want is configurable"
> (현재 v2.1.86 기준 60+ 설정으로 증가)

---

## Daily (3개)

### 68. 매일 업데이트하고 changelog 읽기

```bash
# 업데이트
claude update

# changelog 확인
# https://github.com/anthropics/claude-code/blob/main/CHANGELOG.md
```

Claude Code는 빠르게 진화한다. 매일 새 기능이 추가되므로 changelog를 읽는 습관이 중요.

### 69. Reddit 커뮤니티 팔로우

- [r/ClaudeAI](https://www.reddit.com/r/ClaudeAI/) — 전체 Claude 관련
- [r/ClaudeCode](https://www.reddit.com/r/ClaudeCode/) — Claude Code 특화

실전 팁, 문제 해결, 워크플로우 공유가 활발하다.

### 70. 공식 소스 팔로우 (X/Twitter)

| 누구 | 역할 |
|------|------|
| [Boris Cherny](https://x.com/bcherny) | Claude Code 창시자 |
| [Thariq](https://x.com/trq212) | Skills 시스템 |
| [Cat Wu](https://x.com/_catwu) | 엔지니어링 |
| [Lydia Hallie](https://x.com/lydiahallie) | DevRel |
| [Noah Zweben](https://x.com/noahzweben) | 제품 |
| [Claude](https://x.com/claudeai) | 공식 계정 |
