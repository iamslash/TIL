# Claude Code Best Practice — 확장편 (Agents, Commands, Skills)

> **출처**: [claude-code-best-practice](https://github.com/shanraisshan/claude-code-best-practice) 레포지토리 기반.
> Boris Cherny, Thariq, Lydia Hallie 등의 공식 팁.

---

## 목차

- [Agents (4개)](#agents-4개)
  - [17. 기능 특화 subagent + skill >> 범용 agent](#17-기능-특화-subagent--skill--범용-agent)
  - [18. "use subagents"를 붙여서 compute 더 투입](#18-use-subagents를-붙여서-compute-더-투입)
  - [19. Agent teams + tmux + git worktrees](#19-agent-teams--tmux--git-worktrees)
  - [20. Test time compute — 별도 context window의 힘](#20-test-time-compute--별도-context-window의-힘)
- [Commands (3개)](#commands-3개)
  - [21. workflow는 command로 만들어라](#21-workflow는-command로-만들어라)
  - [22. 매일 반복하는 inner loop → slash command](#22-매일-반복하는-inner-loop--slash-command)
  - [23. 하루에 2번 이상 하면 skill이나 command로](#23-하루에-2번-이상-하면-skill이나-command로)
- [Skills (9개)](#skills-9개)
  - [24. `context: fork`로 격리된 subagent에서 실행](#24-context-fork로-격리된-subagent에서-실행)
  - [25. monorepo에서 subfolder skill 사용](#25-monorepo에서-subfolder-skill-사용)
  - [26. Skill은 폴더다 — progressive disclosure](#26-skill은-폴더다--progressive-disclosure)
  - [27. Gotchas 섹션 필수](#27-gotchas-섹션-필수)
  - [28. description은 트리거다](#28-description은-트리거다)
  - [29. 당연한 걸 쓰지 마라](#29-당연한-걸-쓰지-마라)
  - [30. Claude를 조종하지 마라](#30-claude를-조종하지-마라)
  - [31. script와 라이브러리를 skill에 포함](#31-script와-라이브러리를-skill에-포함)
  - [32. `` !`command` ``으로 동적 shell 출력 주입](#32-command으로-동적-shell-출력-주입)

---

## Agents (4개)

### 17. 기능 특화 subagent + skill >> 범용 agent

"QA agent", "backend engineer agent" 같은 범용 agent보다 **기능 특화** agent가 훨씬 효과적이다.

```markdown
<!-- .claude/agents/voice-clone-reviewer.md -->
---
name: voice-clone-reviewer
description: Review voice cloning output quality and suggest parameter adjustments
model: sonnet
skills:
  - audio-quality-check
tools: Read, Bash, Glob, Grep
---
Voice cloning 결과물의 품질을 평가하고 파라미터 조정을 제안하는 전문 에이전트.
```

skill을 preload하면 agent가 시작할 때 해당 지식을 갖고 시작한다 (**progressive disclosure**).

> Boris: "have feature specific sub-agents with skills instead of general qa, backend engineer"

### 18. "use subagents"를 붙여서 compute 더 투입

프롬프트 끝에 "use subagents"를 추가하면 Claude가 작업을 subagent에 분산시킨다.

```
"이 코드를 리팩토링해줘. use subagents"
```

효과: 메인 agent의 context window를 깨끗하게 유지하면서, subagent가 개별 작업을 처리.

> Boris: 🚫👶 — subagent에 맡기고 babysit 하지 마라.

### 19. Agent teams + tmux + git worktrees

병렬 개발의 핵심 조합:

```bash
# git worktree로 독립 작업 공간 생성
git worktree add ../dia-feature-a feature-a
git worktree add ../dia-feature-b feature-b

# 각 worktree에서 별도 Claude 세션 실행
cd ../dia-feature-a && claude
cd ../dia-feature-b && claude
```

tmux로 여러 세션을 한 화면에서 관리. Boris는 **수십 개의 Claude를 동시에** 실행한다.

### 20. Test time compute — 별도 context window의 힘

같은 모델이라도 **별도 context window**에서 실행하면 다른 관점을 제공한다.

```
# Agent A: 코드를 작성
# Agent B (별도 세션): 같은 코드를 리뷰
```

하나의 agent가 만든 버그를 동일 모델의 다른 agent가 찾을 수 있다.
엔지니어링 팀과 같은 원리: 작성자보다 리뷰어가 버그를 더 잘 발견한다.

> Boris: "using separate context windows makes the result even better — one agent can cause bugs and another (same model) can find them"

---

## Commands (3개)

### 21. workflow는 command로 만들어라

agent가 아닌 **command**로 워크플로우를 정의한다. command는 기존 context에 지식을 주입하는 방식이라 가볍다.

```markdown
<!-- .claude/commands/commit-push-pr.md -->
---
name: commit-push-pr
description: Commit changes, push to remote, and create a PR
---
1. 변경사항을 커밋한다 (conventional commit 형식)
2. 현재 브랜치를 push한다
3. gh CLI로 PR을 생성한다
```

```
# 사용:
/commit-push-pr
```

Command는 `.claude/commands/`에 위치하며 git에 커밋하여 팀 전체가 공유한다.

### 22. 매일 반복하는 inner loop → slash command

하루에 여러 번 하는 작업은 slash command로 만든다.

```markdown
<!-- .claude/commands/lint-fix.md -->
---
name: lint-fix
description: Run linter and auto-fix issues
---
ruff check . --fix && ruff format .
결과를 보고하고, 고치지 못한 이슈가 있으면 수동 수정을 제안한다.
```

Boris 예시들:
- `/commit-push-pr` — 커밋, 푸시, PR 생성
- `/techdebt` — 세션 끝에 중복 코드 찾아서 제거
- `/slack-feedback` — Slack 피드백을 수집해서 PR 생성

### 23. 하루에 2번 이상 하면 skill이나 command로

```
# 매일 하는 것들을 관찰하고:
"모델 로드 → 텍스트 입력 → 생성 → 품질 확인"

# command로:
/dia-test-generate   ← 한 번에 전체 워크플로우 실행
```

> Boris: "if you do something more than once a day, turn it into a skill or command"

---

## Skills (9개)

### 24. `context: fork`로 격리된 subagent에서 실행

skill에 `context: fork`를 설정하면 별도 subagent에서 실행된다. 메인 context에는 최종 결과만 전달.

```markdown
<!-- .claude/skills/heavy-analysis/SKILL.md -->
---
name: heavy-analysis
description: Run deep code analysis
context: fork
agent: general-purpose
---
코드를 깊이 분석하고 결과 요약만 반환한다.
```

메인 context가 중간 tool call들로 오염되지 않아서, 긴 분석 작업에 유용하다.

> Lydia Hallie: "main context only sees the final result, not intermediate tool calls"

### 25. monorepo에서 subfolder skill 사용

```
packages/
  frontend/
    .claude/skills/react-patterns/SKILL.md    # frontend에서만 활성화
  backend/
    .claude/skills/api-conventions/SKILL.md   # backend에서만 활성화
```

`paths` frontmatter로 특정 파일 패턴에서만 skill이 활성화되게 할 수도 있다:

```yaml
---
paths: "packages/frontend/**/*.tsx"
---
```

### 26. Skill은 폴더다 — progressive disclosure

skill은 단일 markdown이 아니라 **폴더**다. 하위에 참조자료, 스크립트, 예제를 둔다.

```
.claude/skills/audio-generation/
  SKILL.md              # 메인 지시사항
  references/
    api.md              # 상세 API 시그니처
    parameters.md       # 파라미터 가이드
  scripts/
    validate_output.py  # 출력 검증 스크립트
  examples/
    basic.md            # 기본 사용 예시
    voice_clone.md      # 음성 클론 예시
```

Claude는 필요할 때 하위 파일을 읽는다. 처음부터 전부 로드하지 않아서 context를 절약한다.

> Thariq: "Think of the entire file system as a form of context engineering and progressive disclosure"

### 27. Gotchas 섹션 필수

skill에서 **가장 가치 있는 콘텐츠**는 Gotchas 섹션이다. Claude가 실패한 지점을 누적한다.

```markdown
## Gotchas

- Mac에서 `torch.compile`을 사용하면 크래시남. 항상 `use_torch_compile=False` 사용
- `[S1]`으로 시작하지 않으면 오디오 품질이 급격히 저하됨
- 5초 미만 오디오 프롬프트는 voice cloning 품질이 나쁨, 5~10초가 최적
- float16은 MPS에서 간헐적 NaN 발생, float32 사용 권장
```

시간이 지나면서 이 섹션이 가장 가치 있는 지식이 된다.

> Thariq: "The highest-signal content in any skill is the Gotchas section"

### 28. description은 트리거다

skill의 `description` 필드는 사람용 설명이 아니라, **모델이 "이 skill을 사용할지" 판단하는 트리거**다.

```yaml
# 나쁜 예 (사람용 설명):
description: "오디오 생성 관련 유틸리티"

# 좋은 예 (트리거):
description: "When generating audio, cloning voices, or adjusting TTS parameters"
```

Claude Code 시작 시 모든 skill의 description을 스캔해서 "이 요청에 맞는 skill이 있는가?" 판단한다.

> Thariq: "the description field is not a summary — it's a description of when to trigger this skill. Write it for the model."

### 29. 당연한 걸 쓰지 마라

Claude는 이미 코드와 프레임워크에 대해 많이 알고 있다. skill에는 **기본 행동과 다른 것**만 쓴다.

```markdown
<!-- 나쁜 예: Claude가 이미 아는 것 -->
Python에서 함수를 정의할 때 def 키워드를 사용합니다.

<!-- 좋은 예: Claude가 모르는 프로젝트 특화 지식 -->
DAC 코덱은 9채널을 사용하며, delay_pattern [0,8,9,10,11,12,13,14,15]로
채널 간 지연을 적용한다. 이 패턴을 변경하면 오디오가 깨진다.
```

> Thariq: 🚫👶 — "focus on information that pushes Claude out of its normal way of thinking"

### 30. Claude를 조종하지 마라

단계별 지시가 아닌, **목표와 제약**을 준다.

```markdown
<!-- 나쁜 예: railroading -->
1. 먼저 config.py를 열어라
2. DecoderConfig 클래스를 찾아라
3. num_channels 필드를 수정해라
4. 저장해라

<!-- 좋은 예: 목표 + 제약 -->
Goal: decoder의 채널 수를 변경 가능하게 만들기
Constraints:
- 기존 API 호환성 유지
- DiaConfig의 frozen=True 패턴 따르기
- 기본값은 현재 값(9) 유지
```

> Thariq: 🚫👶 — "give goals and constraints, not prescriptive step-by-step instructions"

### 31. script와 라이브러리를 skill에 포함

Claude가 boilerplate를 재구성하는 대신 **기존 코드를 조합**하게 한다.

```
.claude/skills/benchmark/
  SKILL.md
  scripts/
    run_benchmark.py      # 벤치마크 실행 스크립트
    parse_results.py      # 결과 파싱
    generate_report.py    # 보고서 생성
```

```markdown
<!-- SKILL.md -->
벤치마크 실행 시 scripts/ 디렉토리의 스크립트를 조합하여 사용한다.
직접 벤치마크 코드를 작성하지 말 것.
```

> Thariq: "Giving Claude scripts and libraries lets Claude spend its turns on composition"

### 32. `` !`command` ``으로 동적 shell 출력 주입

SKILL.md 안에서 backtick-bang 구문으로 실행 시점에 shell 명령 결과를 주입한다.

```markdown
## 현재 환경 정보

!`python --version`
!`uv pip list | grep torch`
!`git branch --show-current`
```

Claude가 skill을 호출할 때 명령이 실행되고, 모델은 **결과만** 본다.
동적으로 변하는 환경 정보를 skill에 포함할 때 유용하다.

> Lydia Hallie: "embed !`command` in SKILL.md to inject dynamic shell output into the prompt"
