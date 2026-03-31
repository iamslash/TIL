# Claude Code Best Practice — 자동화편 (Hooks, Workflows)

> **출처**: [claude-code-best-practice](https://github.com/shanraisshan/claude-code-best-practice) 레포지토리 기반.
> Boris Cherny, Thariq 등의 공식 팁.

---

## 목차

- [Hooks (5개)](#hooks-5개)
  - [33. On-demand hooks — skill별 위험 차단](#33-on-demand-hooks--skill별-위험-차단)
  - [34. PreToolUse hook으로 skill 사용량 측정](#34-pretooluse-hook으로-skill-사용량-측정)
  - [35. PostToolUse hook으로 자동 format](#35-posttooluse-hook으로-자동-format)
  - [36. PermissionRequest를 Opus에 라우팅](#36-permissionrequest를-opus에-라우팅)
  - [37. Stop hook으로 계속 작업하게 nudge](#37-stop-hook으로-계속-작업하게-nudge)
- [Workflows (7개)](#workflows-7개)
  - [38. 50%에서 수동 `/compact` — agent dumb zone 회피](#38-50에서-수동-compact--agent-dumb-zone-회피)
  - [39. 작은 작업은 vanilla Claude Code가 최고](#39-작은-작업은-vanilla-claude-code가-최고)
  - [40. 유용한 슬래시 커맨드들](#40-유용한-슬래시-커맨드들)
  - [41. thinking mode + Explanatory output style](#41-thinking-mode--explanatory-output-style)
  - [42. ultrathink 키워드로 고강도 추론](#42-ultrathink-키워드로-고강도-추론)
  - [43. 세션 이름 지정 + 재개](#43-세션-이름-지정--재개)
  - [44. Esc Esc 또는 `/rewind`로 되돌리기](#44-esc-esc-또는-rewind로-되돌리기)
- [Workflows Advanced (6개)](#workflows-advanced-6개)
  - [45. ASCII 다이어그램으로 아키텍처 이해](#45-ascii-다이어그램으로-아키텍처-이해)
  - [46. `/loop`과 `/schedule`](#46-loop과-schedule)
  - [47. Ralph Wiggum plugin으로 자율 실행](#47-ralph-wiggum-plugin으로-자율-실행)
  - [48. `/permissions` 와일드카드 >> `--dangerously-skip-permissions`](#48-permissions-와일드카드----dangerously-skip-permissions)
  - [49. `/sandbox`로 84% 권한 프롬프트 감소](#49-sandbox로-84-권한-프롬프트-감소)
  - [50. Product verification skill에 1주일 투자할 가치](#50-product-verification-skill에-1주일-투자할-가치)

---

## Hooks (5개)

Hooks는 Claude의 agentic loop **밖에서** 결정적으로 실행되는 사용자 정의 핸들러다.
`.claude/settings.json`에서 설정한다.

### 33. On-demand hooks — skill별 위험 차단

skill에 hooks를 등록하면, 그 skill이 활성화된 동안에만 hook이 작동한다.

```markdown
<!-- .claude/skills/careful/SKILL.md -->
---
name: careful
description: Block destructive commands during sensitive operations
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: |
            echo "$TOOL_INPUT" | grep -qiE '(rm -rf|DROP TABLE|force-push|kubectl delete)' && echo "BLOCKED: destructive command" && exit 1 || exit 0
---
```

`/careful` 을 실행하면 세션 동안 위험한 명령이 차단된다.
`/freeze` 패턴도 유사: 특정 디렉토리 외부 편집을 차단.

> Thariq: "on-demand hooks in skills — /careful blocks destructive commands, /freeze blocks edits outside a directory"

### 34. PreToolUse hook으로 skill 사용량 측정

어떤 skill이 얼마나 자주 트리거되는지 로깅한다.

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Skill",
        "hooks": [
          {
            "type": "command",
            "command": "echo \"$(date +%Y-%m-%dT%H:%M:%S) skill=$TOOL_INPUT\" >> ~/.claude/skill-usage.log"
          }
        ]
      }
    ]
  }
}
```

인기 skill과 기대보다 덜 트리거되는 skill을 파악할 수 있다.

> Thariq: "use a PreToolUse hook to find popular or undertriggering skills"

### 35. PostToolUse hook으로 자동 format

Claude가 코드를 작성하면 자동으로 포맷터를 실행한다. Claude가 90%는 잘 포맷하지만, 나머지 10%를 hook이 처리해서 CI 실패를 방지한다.

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "ruff format . || true"
          }
        ]
      }
    ]
  }
}
```

`|| true`는 포맷터가 실패해도 Claude의 작업 흐름을 중단하지 않기 위함.

> Boris: "Claude generates well-formatted code, the hook handles the last 10% to avoid CI failures"

### 36. PermissionRequest를 Opus에 라우팅

권한 요청을 사람이 직접 승인하는 대신, Opus 모델에 위임해서 안전한 것은 자동 승인한다.

```json
{
  "hooks": {
    "PermissionRequest": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "scripts/auto-approve-safe.sh"
          }
        ]
      }
    ]
  }
}
```

스크립트 내부에서 Opus API를 호출하여 공격 패턴을 스캔하고, 안전하면 승인, 위험하면 사람에게 전달.

> Boris: 🚫👶 — "route permission requests to Opus via a hook — let it scan for attacks and auto-approve safe ones"

### 37. Stop hook으로 계속 작업하게 nudge

Claude가 턴을 마칠 때 Stop hook이 발동한다. 이를 활용해 검증을 강제하거나 계속 작업하게 만든다.

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "prompt",
            "prompt": "작업이 완료되었는지 확인하라. 미완료 항목이 있으면 계속 진행하라."
          }
        ]
      }
    ]
  }
}
```

agent를 시작하거나, 프롬프트로 계속할지 결정하게 할 수도 있다.

> Boris: "use a Stop hook to nudge Claude to keep going or verify its work at the end of a turn"

---

## Workflows (7개)

### 38. 50%에서 수동 `/compact` — agent dumb zone 회피

context window가 차면 Claude의 성능이 저하된다 ("agent dumb zone").

```
# context 사용량 확인
/context

# 50%에 도달하면:
/compact

# 새로운 작업으로 전환할 때:
/clear
```

`/compact`는 대화를 요약하고 context를 절약한다. 포커스 지시도 가능:

```
/compact 오디오 생성 파라미터 관련 내용에 집중해서 요약해줘
```

### 39. 작은 작업은 vanilla Claude Code가 최고

워크플로우, agent, skill은 강력하지만, **작은 작업에는 오버헤드**다.

```
# 이럴 때는 그냥 Claude에게 직접:
"이 함수의 타입 힌트 추가해줘"
"이 버그 고쳐줘"

# 이럴 때 워크플로우 사용:
"전체 인증 시스템을 리팩토링해줘"
"새 기능을 설계하고 구현해줘"
```

> "vanilla cc is better than any workflows with smaller tasks"

### 40. 유용한 슬래시 커맨드들

```
/model     — 모델 선택 + reasoning 레벨 (Opus for plan, Sonnet for code)
/context   — context 사용량 컬러 그리드
/usage     — plan 사용 한도 확인
/extra-usage — 오버플로 과금 설정
/config    — 전체 설정 조정
/cost      — 현재 세션 토큰 사용량
```

> Cat Wu: Opus는 plan mode, Sonnet은 코드 작성에 사용하면 효율적.

### 41. thinking mode + Explanatory output style

학습 목적이라면 두 가지를 켜라:

```
/config
→ Thinking: true (Claude의 추론 과정을 볼 수 있음)
→ Output Style: Explanatory (변경 이유를 ★ Insight 박스로 설명)
```

또는 Learning output style: Claude가 코드 변경을 코칭하듯 설명한다.

> Boris: "Use Opus with thinking for everything. It's the best coding model."

### 42. ultrathink 키워드로 고강도 추론

프롬프트에 `ultrathink`를 포함하면 Claude가 extended thinking을 high effort로 수행한다.

```
"이 race condition을 분석해줘. ultrathink"
```

Anthropic 공식 문서의 extended thinking 팁에서 소개된 키워드.

### 43. 세션 이름 지정 + 재개

여러 Claude를 동시에 실행할 때 세션을 구분한다.

```
/rename dia-voice-clone-refactor

# 나중에 이어서:
/resume dia-voice-clone-refactor

# 또는 세션 목록에서 선택:
/resume
```

Boris 팁: `[TODO - refactor task]` 같은 상태 표시 이름을 사용하면 어떤 세션이 미완료인지 한눈에 파악.

### 44. Esc Esc 또는 `/rewind`로 되돌리기

Claude가 잘못된 방향으로 갈 때, 같은 context에서 고치려 하지 말고 **되돌려라**.

```
# Esc 두 번 누르기 → 코드와 대화를 이전 시점으로 되돌림
# 또는:
/rewind
```

같은 context에서 "아니 그게 아니라..."라고 수정하면 이전 잘못된 시도가 context를 오염시킨다.
되돌린 후 깨끗한 상태에서 다시 지시하는 것이 더 효과적.

---

## Workflows Advanced (6개)

### 45. ASCII 다이어그램으로 아키텍처 이해

```
"이 프로젝트의 아키텍처를 ASCII 다이어그램으로 그려줘"
```

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Text Input │────▶│   Encoder    │────▶│  Cross-Attn  │
│  [S1]...[S2]│     │  12 layers   │     │             │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                                                 ▼
                                         ┌──────────────┐     ┌──────────┐
                                         │   Decoder    │────▶│  DAC     │
                                         │  18 layers   │     │  Decode  │
                                         └──────────────┘     └──────────┘
```

> Boris: "use ASCII diagrams a lot to understand your architecture"

### 46. `/loop`과 `/schedule`

```
# 로컬에서 반복 실행 (최대 3일):
/loop 5m /lint-fix          # 5분마다 린트 수정
/loop 30m /slack-feedback   # 30분마다 Slack 피드백 수집

# 클라우드에서 반복 실행 (컴퓨터 꺼도 동작):
/schedule                   # Claude가 대화형으로 설정 안내
```

Boris의 실제 사용 예:
- `/loop 5m /babysit` — PR 코드리뷰 자동 대응, 리베이스, 머지까지
- `/loop 1h /pr-pruner` — 오래된 불필요 PR 자동 닫기

> Boris: "Experiment with turning workflows into skills + loops. It's powerful."

### 47. Ralph Wiggum plugin으로 자율 실행

장기 실행 작업에 사용. Claude가 작업 완료까지 스스로 반복한다.

```
# 방법 1: 프롬프트에서 검증 요청
"이 작업을 끝나면 background agent로 결과를 검증해줘"

# 방법 2: Stop hook으로 자동 검증
# 방법 3: Ralph Wiggum plugin 설치
```

> Boris: "for very long-running tasks, use the ralph-wiggum plugin"

### 48. `/permissions` 와일드카드 >> `--dangerously-skip-permissions`

```
# 나쁜 방법:
claude --dangerously-skip-permissions  # 모든 권한 우회 — 위험

# 좋은 방법:
/permissions
# Allow: Bash(uv run *), Bash(ruff *), Edit(/Users/david.sun/my/py/dia/**)
```

와일드카드로 안전한 명령만 사전 승인. `.claude/settings.json`에 저장하여 팀과 공유.

```json
{
  "permissions": {
    "allow": [
      "Bash(uv run *)",
      "Bash(ruff *)",
      "Bash(python example/*)",
      "Edit(/Users/david.sun/my/py/dia/**)"
    ]
  }
}
```

### 49. `/sandbox`로 84% 권한 프롬프트 감소

```
/sandbox
```

파일 시스템과 네트워크를 격리하여 안전한 환경에서 실행. Anthropic 내부에서 84% 권한 프롬프트 감소 효과.

### 50. Product verification skill에 1주일 투자할 가치

verification skill은 Claude의 출력이 **실제로 올바른지** 자동 검증한다.

```markdown
<!-- .claude/skills/audio-verifier/SKILL.md -->
---
name: audio-verifier
description: When audio generation completes, verify output quality
---
1. 출력 파일이 존재하고 0 byte가 아닌지 확인
2. 샘플레이트가 44100Hz인지 확인
3. 오디오 길이가 예상 범위(5~35초) 내인지 확인
4. NaN이나 무음 구간이 없는지 확인
```

> Thariq: "Verification skills are extremely useful. It can be worth having an engineer spend a week just making your verification skills excellent."
