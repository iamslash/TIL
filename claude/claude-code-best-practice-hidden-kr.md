# Claude Code Best Practice — 숨겨진 기능편 (Boris 15 Hidden Features + Customization)

> **출처**: [claude-code-best-practice](https://github.com/shanraisshan/claude-code-best-practice) 레포지토리 기반.
> Boris Cherny 15 Hidden & Under-Utilized Features (2026-03-30) + 12 Ways to Customize (2026-02-12).

---

## 목차

- [Boris 15 Hidden Features](#boris-15-hidden-features)
  - [71. 모바일 앱으로 코딩](#71-모바일-앱으로-코딩)
  - [72. 세션을 디바이스 간 이동](#72-세션을-디바이스-간-이동)
  - [73. `/loop` + `/schedule` — 가장 강력한 기능](#73-loop--schedule--가장-강력한-기능)
  - [74. Hooks로 결정적 로직 실행](#74-hooks로-결정적-로직-실행)
  - [75. Cowork Dispatch — 비코딩 작업 위임](#75-cowork-dispatch--비코딩-작업-위임)
  - [76. Chrome extension — 프론트엔드 검증의 핵심](#76-chrome-extension--프론트엔드-검증의-핵심)
  - [77. Desktop 앱으로 웹서버 자동 시작/테스트](#77-desktop-앱으로-웹서버-자동-시작테스트)
  - [78. `/branch`로 세션 포크](#78-branch로-세션-포크)
  - [79. `/btw`로 사이드 질문](#79-btw로-사이드-질문)
  - [80. Git worktrees로 수십 개 Claude 병렬 실행](#80-git-worktrees로-수십-개-claude-병렬-실행)
  - [81. `/batch`로 대규모 changeset fan out](#81-batch로-대규모-changeset-fan-out)
  - [82. `--bare`로 SDK 시작 10x 빠르게](#82---bare로-sdk-시작-10x-빠르게)
  - [83. `--add-dir`로 여러 repo 접근](#83---add-dir로-여러-repo-접근)
  - [84. `--agent`로 커스텀 시스템 프롬프트](#84---agent로-커스텀-시스템-프롬프트)
  - [85. `/voice`로 음성 입력](#85-voice로-음성-입력)
- [Customization (1개)](#customization-1개--위-항목과-중복-제외)
  - [86. settings.json을 git에 커밋하여 팀 공유](#86-settingsjson을-git에-커밋하여-팀-공유)

---

## Boris 15 Hidden Features

### 71. 모바일 앱으로 코딩

Claude 모바일 앱(iOS/Android)의 **Code 탭**에서 코드 작성, 변경 리뷰, PR 승인이 가능하다.

```
1. Claude 앱 다운로드 (iOS/Android)
2. 좌측 메뉴 → Code 탭
3. 노트북 없이도 코드 변경 가능
```

> Boris: "I write a lot of my code from the iOS app"

### 72. 세션을 디바이스 간 이동

```bash
# 클라우드 세션을 로컬 터미널로 가져오기:
claude --teleport
# 또는 세션 내에서:
/teleport

# 로컬 세션을 폰/웹에서 제어:
/remote-control
# 또는:
/rc
```

Boris는 `/config`에서 **"Enable Remote Control for all sessions"**를 켜두고 사용한다.

### 73. `/loop` + `/schedule` — 가장 강력한 기능

로컬에서 최대 3일간 반복 실행하거나, 클라우드에서 컴퓨터가 꺼져도 실행.

```
# Boris의 실제 사용 예:
/loop 5m /babysit           # 코드리뷰 자동 대응, 리베이스, 머지까지
/loop 30m /slack-feedback    # Slack 피드백 수집 → PR 생성
/loop /post-merge-sweeper    # 머지 후 놓친 리뷰 코멘트 처리
/loop 1h /pr-pruner          # 오래된 불필요 PR 자동 닫기
```

핵심: **워크플로우를 skill로 만들고 → loop으로 자동화**하는 패턴.

> Boris: "Experiment with turning workflows into skills + loops. It's powerful."

### 74. Hooks로 결정적 로직 실행

Hook 이벤트별 실용 예시:

```
SessionStart  → 세션 시작 시 동적으로 context 로드
PreToolUse    → 모든 bash 명령을 로깅
PermissionRequest → WhatsApp으로 승인/거부 라우팅
Stop          → Claude가 멈출 때 계속하도록 nudge
```

Hooks는 Claude의 agentic loop 밖에서 **결정적으로** 실행된다 — 확률적인 CLAUDE.md 지시와 다르다.

### 75. Cowork Dispatch — 비코딩 작업 위임

Dispatch는 Claude Desktop 앱의 **보안 리모트 컨트롤**.

```
- Slack, 이메일 확인 및 정리
- 파일 관리
- 브라우저 작업
- MCP를 통한 다양한 도구 사용
```

코딩이 아닌 작업을 Claude에게 위임할 때 사용.

> Boris: "When I'm not coding, I'm dispatching."

### 76. Chrome extension — 프론트엔드 검증의 핵심

**가장 중요한 원칙**: Claude에게 출력을 검증할 방법을 줘라.

```
# 비유: 누군가에게 웹사이트를 만들라고 하면서 브라우저를 못 쓰게 하면?
# → 결과가 좋을 리 없다.

# Chrome extension을 주면:
# → Claude가 코드를 쓰고, 브라우저에서 확인하고, 수정하고, 반복
```

Boris는 웹 코드 작업 시 **항상** Chrome extension을 사용한다.

> Boris: "give Claude a way to verify its output. Once you do that, Claude will iterate until the result is great."

### 77. Desktop 앱으로 웹서버 자동 시작/테스트

Desktop 앱에는 **내장 브라우저**가 포함되어 있어, 웹서버를 자동으로 시작하고 테스트할 수 있다.

```
# CLI나 VSCode에서도 Chrome extension으로 유사하게 설정 가능
# Desktop 앱은 이 과정이 통합되어 있음
```

### 78. `/branch`로 세션 포크

현재 대화를 분기하여 다른 접근을 시도한다.

```
# 방법 1: 세션 내에서
/branch

# 방법 2: CLI에서
claude --resume <session-id> --fork-session
```

`/branch` 후에는 분기된 세션에 있다. 원래 세션으로 돌아가려면:
```bash
claude -r <original-session-id>
```

계획 A와 계획 B를 각각 시도해보고 비교할 때 유용하다.

### 79. `/btw`로 사이드 질문

작업 중인 Claude를 방해하지 않고 빠른 질문을 한다.

```
/btw dachshund 스펠링이 뭐야?
> dachshund — 독일어로 "badger dog" (dachs=오소리, hund=개)
> ↑/↓ 스크롤 · Space/Enter/Escape로 닫기
```

메인 context에 추가되지 않으므로 오염이 없다.

> Boris: "I use this all the time to answer quick questions while the agent works."

### 80. Git worktrees로 수십 개 Claude 병렬 실행

```bash
# 새 worktree에서 Claude 시작:
claude -w

# 또는 수동으로:
git worktree add ../dia-experiment experiment-branch
cd ../dia-experiment
claude
```

Desktop 앱에는 "worktree" 체크박스가 있다.
비-git VCS 사용자는 `WorktreeCreate` hook으로 커스텀 로직 추가 가능.

> Boris: "I have dozens of Claudes running at all times, and this is how I do it."

### 81. `/batch`로 대규모 changeset fan out

```
/batch
```

Claude가 인터뷰를 통해 작업을 파악한 뒤, **수십~수백 개의 worktree agent**를 생성하여 병렬 처리.

사용 사례:
- 대규모 코드 마이그레이션
- API 시그니처 일괄 변경
- 린트 규칙 전체 적용

각 worktree agent는 독립적으로 자신만의 코드 카피에서 작업한다.

### 82. `--bare`로 SDK 시작 10x 빠르게

비대화형 SDK 사용 시 불필요한 CLAUDE.md, settings, MCP 스캔을 건너뛴다.

```bash
claude -p "summarize this codebase" \
    --output-format=stream-json \
    --verbose \
    --bare
```

향후 버전에서 기본값이 `--bare`로 바뀔 예정. 현재는 명시적 opt-in.

> Boris: "this was a design oversight — for now, opt in with the flag to get up to 10x faster startup"

### 83. `--add-dir`로 여러 repo 접근

```bash
# 시작 시:
claude --add-dir /path/to/other-repo

# 세션 중:
/add-dir /path/to/other-repo
```

다른 레포를 추가하면 Claude가 해당 레포의 파일을 읽고 수정할 **권한**도 함께 부여된다.

팀 설정으로도 가능:
```json
// .claude/settings.json
{
  "additionalDirectories": ["/path/to/shared-lib"]
}
```

> Boris: "this not only tells Claude about the repo, but also gives it permissions to work in the repo"

### 84. `--agent`로 커스텀 시스템 프롬프트

```bash
# .claude/agents/에 agent 정의 후:
claude --agent=voice-clone-reviewer
```

agent별로 다른 도구, 권한, 모델, 설명을 지정할 수 있다.
예: read-only agent, 리뷰 전용 agent, 도메인 특화 agent.

```markdown
<!-- .claude/agents/read-only-explorer.md -->
---
name: read-only-explorer
description: Read-only codebase exploration
tools: Read, Glob, Grep
model: haiku
---
코드를 수정하지 않고 탐색만 한다.
```

### 85. `/voice`로 음성 입력

```
# CLI:
/voice
# → Space bar 길게 눌러서 말하기

# Desktop 앱:
# → 음성 버튼 클릭

# iOS:
# → 설정에서 Dictation 활성화
```

20개 언어 지원. 활성화 키 변경 가능.

> Boris: "Fun fact: I do most of my coding by speaking to Claude, rather than typing."

---

## Customization (1개 — 위 항목과 중복 제외)

### 86. settings.json을 git에 커밋하여 팀 공유

```json
// .claude/settings.json — git에 커밋
{
  "model": "opus",
  "permissions": {
    "allow": ["Bash(uv run *)", "Bash(ruff *)"]
  },
  "attribution": {
    "commit": "Co-Authored-By: Claude <noreply@anthropic.com>"
  }
}
```

```json
// .claude/settings.local.json — git-ignored, 개인 설정
{
  "model": "sonnet",
  "alwaysThinkingEnabled": true
}
```

설정 계층 우선순위:
1. **Managed** (조직 강제) — 절대 오버라이드 불가
2. **CLI 인자** — 단일 세션 오버라이드
3. **settings.local.json** — 개인 프로젝트 설정
4. **settings.json** — 팀 공유 설정
5. **~/.claude/settings.json** — 글로벌 개인 기본값

> Boris: "customize, check your settings.json into git so your team can benefit too"
