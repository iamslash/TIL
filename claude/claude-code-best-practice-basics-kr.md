# Claude Code Best Practice — 기초편

> **출처**: [claude-code-best-practice](https://github.com/shanraisshan/claude-code-best-practice) 레포지토리의 tips/, best-practice/ 디렉토리 내용을 기반으로 정리.
> Boris Cherny (Claude Code 창시자), Thariq, Dex (HumanLayer) 등의 공식 팁.

---

## 목차

- [Prompting (3개)](#prompting-3개)
  - [1. Claude에게 도전하게 하라](#1-claude에게-도전하게-하라)
  - [2. 평범한 수정 후에는 갈아엎어라](#2-평범한-수정-후에는-갈아엎어라)
  - [3. 버그는 붙여넣고 "fix" 한마디](#3-버그는-붙여넣고-fix-한마디)
- [Planning / Specs (6개)](#planning--specs-6개)
  - [4. 항상 plan mode로 시작하라](#4-항상-plan-mode로-시작하라)
  - [5. Claude에게 인터뷰시키고, 새 세션에서 실행](#5-claude에게-인터뷰시키고-새-세션에서-실행)
  - [6. phase-wise gated plan](#6-phase-wise-gated-plan)
  - [7. 두 번째 Claude가 staff engineer로 plan 리뷰](#7-두-번째-claude가-staff-engineer로-plan-리뷰)
  - [8. 상세한 스펙으로 모호함 줄이기](#8-상세한-스펙으로-모호함-줄이기)
  - [9. PRD보다 prototype — 많이 만들어보기](#9-prd보다-prototype--많이-만들어보기)
- [CLAUDE.md (7개)](#claudemd-7개)
  - [10. 200줄 이하로 유지](#10-200줄-이하로-유지)
  - [11. `<important if="...">` 태그로 규칙 강조](#11-important-if-태그로-규칙-강조)
  - [12. monorepo용 다중 CLAUDE.md](#12-monorepo용-다중-claudemd)
  - [13. `.claude/rules/`로 큰 지시사항 분리](#13-clauderules로-큰-지시사항-분리)
  - [14. memory.md, constitution.md는 보장이 안 된다](#14-memorymd-constitutionmd는-보장이-안-된다)
  - [15. "run the tests"가 첫 시도에 통과해야](#15-run-the-tests가-첫-시도에-통과해야)
  - [16. 코드베이스를 깨끗하게 유지하고 마이그레이션 완료하기](#16-코드베이스를-깨끗하게-유지하고-마이그레이션-완료하기)

---

## Prompting (3개)

### 1. Claude에게 도전하게 하라

Claude에게 수동적으로 "이거 해줘"가 아니라 **능동적으로 검증하게** 프롬프팅한다.

```
"grill me on these changes and don't make a PR until I pass your test."
```

또는 브랜치 간 차이를 검증하게:

```
"prove to me this works — diff behavior between main and my feature branch"
```

> Boris: 🚫👶 (babysit 하지 마라) — Claude가 스스로 판단하게 두라.

### 2. 평범한 수정 후에는 갈아엎어라

첫 시도가 그저그런 결과일 때, 거기서 패치하지 말고:

```
"knowing everything you know now, scrap this and implement the elegant solution"
```

같은 context에서 이미 문제를 이해했으므로, 두 번째 시도가 훨씬 깔끔하다.

### 3. 버그는 붙여넣고 "fix" 한마디

```
# Slack 버그 스레드를 복사해서:
"fix"

# CI 실패 로그를 보고:
"Go fix the failing CI tests."
```

어떻게 고칠지 마이크로매니징하지 말 것. Claude가 코드베이스를 탐색하고 스스로 해결한다.
Docker 로그를 가리키면 분산 시스템 트러블슈팅도 가능하다.

---

## Planning / Specs (6개)

### 4. 항상 plan mode로 시작하라

Boris가 가장 많이 반복하는 조언. 복잡한 작업은 바로 코딩하지 말고 계획부터.

```
# 방법 1: Shift+Tab 두 번
# 방법 2: /plan 슬래시 커맨드
# 방법 3: 프롬프트에 직접
"plan mode로 이 기능을 설계해줘"
```

> "Pour your energy into the plan so Claude can 1-shot the implementation."
> — Boris

잘 안 되면 **코드를 더 고치지 말고 plan mode로 돌아가서 재계획**한다.
검증 단계에서도 plan mode를 사용한다 — 빌드만이 아니라 검증도 계획이 필요하다.

### 5. Claude에게 인터뷰시키고, 새 세션에서 실행

최소한의 스펙이나 프롬프트로 시작하여 Claude의 `AskUserQuestion` 도구로 인터뷰를 받는다.

```
"이 기능의 요구사항을 정리해줘. 부족한 부분은 나에게 질문해."
```

Claude가 질문을 통해 스펙을 완성하면, **새 세션**을 열어서 그 스펙을 실행한다.
이유: 인터뷰 과정의 context가 실행 context를 오염시키지 않게.

> Thariq의 팁. AskUserQuestion 도구는 구조화된 multiple choice 질문도 지원한다.

### 6. phase-wise gated plan

각 phase에 게이트(통과 조건)를 두고, 각 게이트에 테스트를 포함한다.

```
Phase 1: 데이터 모델 설계
  - Gate: unit test 통과
Phase 2: API 엔드포인트 구현
  - Gate: integration test 통과
Phase 3: UI 연동
  - Gate: E2E test 통과
```

한 phase가 게이트를 통과하지 못하면 다음 phase로 넘어가지 않는다.

### 7. 두 번째 Claude가 staff engineer로 plan 리뷰

plan을 작성한 후, 별도 세션(또는 cross-model)에서 리뷰한다.

```
# 세션 1에서 plan 작성 후 export
/export plan.md

# 세션 2에서:
"이 plan을 staff engineer 관점에서 리뷰해줘. 빠진 edge case, 성능 이슈, 보안 문제를 지적해."
```

같은 모델이라도 **별도 context window**에서 보면 다른 관점이 나온다 (test time compute 원리).

### 8. 상세한 스펙으로 모호함 줄이기

```
# 나쁜 예:
"로그인 기능 추가해줘"

# 좋은 예:
"이메일+비밀번호 기반 로그인. bcrypt 해싱, JWT 토큰 반환,
 실패 시 429 rate limit, 5회 시도 후 15분 잠금.
 기존 /api/v2/ 패턴을 따를 것."
```

> Boris: "Write detailed specs and reduce ambiguity before handing work off. The more specific you are, the better the output."

### 9. PRD보다 prototype — 많이 만들어보기

```
"20~30개 버전을 빠르게 만들어보자. 비용은 낮으니 많은 shot을 쏘자."
```

> Boris (Pragmatic Engineer 팟캐스트): "prototype > PRD — build 20-30 versions instead of writing specs, the cost of building is low so take many shots"

---

## CLAUDE.md (7개)

### 10. 200줄 이하로 유지

CLAUDE.md가 길어지면 Claude가 규칙을 무시하기 시작한다.

```
# 이상적: 60줄 (HumanLayer 기준)
# 최대: 200줄 (Boris 권장)
# 그래도 100% 보장은 아님 (Reddit에서도 보고됨)
```

> Boris: "keep CLAUDE.md under 200 lines per file for reliable adherence"

### 11. `<important if="...">` 태그로 규칙 강조

CLAUDE.md가 길어질수록 Claude가 일부 규칙을 무시한다. 도메인 특화 규칙은 조건부 태그로 감싸서 강조한다.

```markdown
<important if="editing database models">
마이그레이션 파일을 반드시 생성하고, 기존 데이터 호환성을 테스트할 것.
</important>
```

> Dex (HumanLayer): [stop-claude-from-ignoring-your-claude-md](https://www.hlyr.dev/blog/stop-claude-from-ignoring-your-claude-md)

### 12. monorepo용 다중 CLAUDE.md

```
/mymonorepo/
├── CLAUDE.md          # 루트: 공통 규칙 (항상 로드)
├── frontend/
│   └── CLAUDE.md      # frontend 파일 작업 시에만 lazy 로드
├── backend/
│   └── CLAUDE.md      # backend 파일 작업 시에만 lazy 로드
```

**Ancestor loading** (상위): 현재 디렉토리에서 루트까지 올라가며 모든 CLAUDE.md 로드 (시작 시 즉시).
**Descendant loading** (하위): 하위 디렉토리의 CLAUDE.md는 해당 디렉토리 파일을 읽을 때만 lazy 로드.
**Sibling**: frontend에서 작업 중이면 backend의 CLAUDE.md는 절대 로드되지 않음.

### 13. `.claude/rules/`로 큰 지시사항 분리

CLAUDE.md가 200줄을 넘기면 규칙 파일로 분리한다.

```
.claude/
  rules/
    formatting.md      # 코드 포맷 규칙
    testing.md          # 테스트 컨벤션
    architecture.md     # 아키텍처 패턴
```

CLAUDE.md에는 핵심만 남기고, 상세 규칙은 rules/ 디렉토리에 분리.

### 14. memory.md, constitution.md는 보장이 안 된다

CLAUDE.md 자체도 100% 준수가 보장되지 않는데, 별도 파일은 더 약하다.
결정적으로 강제해야 하는 동작은 **settings.json**을 사용한다.

```json
// settings.json — 확실한 강제
{
  "attribution": { "commit": "" }  // Co-Authored-By 제거
}
```

```markdown
<!-- CLAUDE.md — 가이드라인이지 강제가 아님 -->
NEVER add Co-Authored-By to commits  ← 가끔 무시될 수 있음
```

> davila7: "settings.json은 결정적(deterministic), CLAUDE.md는 확률적(probabilistic)"

### 15. "run the tests"가 첫 시도에 통과해야

CLAUDE.md의 완성도 기준: 아무 개발자가 Claude를 실행하고 "run the tests"라고 했을 때 첫 시도에 통과하는가?

```markdown
## Quick Commands
​```bash
# 테스트 실행
uv run pytest tests/ -v

# 린트
uvx ruff check . && uvx ruff format --check .

# 개발 서버
uv run python app.py
​```
```

통과하지 못하면 CLAUDE.md에 **필수 설정/빌드/테스트 명령어가 빠져있다**는 뜻이다.

> Dex: "any developer should be able to launch Claude, say 'run the tests' and it works on the first try"

### 16. 코드베이스를 깨끗하게 유지하고 마이그레이션 완료하기

절반만 마이그레이션된 프레임워크는 Claude를 가장 혼란스럽게 만든다.

```
# 나쁜 상태: 두 가지 패턴이 공존
src/auth/old_middleware.py   # Express 스타일
src/auth/new_middleware.py   # FastAPI 스타일
```

Claude가 어느 패턴을 따를지 모른다. 마이그레이션은 **완전히 끝내거나**, CLAUDE.md에 "새 코드는 반드시 new_middleware 패턴을 따를 것"이라고 명시한다.

> Boris (Pragmatic Engineer 팟캐스트): "keep codebases clean and finish migrations — partially migrated frameworks confuse models"
