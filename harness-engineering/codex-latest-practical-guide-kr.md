# Codex 최신 실습 가이드

> 2026-03-28 기준 로컬 `codex --help`, `codex exec --help`, `codex mcp add --help`로 교차 확인한 실습 가이드

## 목차

- [현재 Codex CLI 표면 이해하기](#현재-codex-cli-표면-이해하기)
- [AGENTS.md로 기본 작업 규칙 고정하기](#agentsmd로-기본-작업-규칙-고정하기)
- [승인 정책과 샌드박스 조합](#승인-정책과-샌드박스-조합)
- [codex exec로 비대화형 실행하기](#codex-exec로-비대화형-실행하기)
- [review, resume, fork 실전 패턴](#review-resume-fork-실전-패턴)
- [MCP 연결 최신 흐름](#mcp-연결-최신-흐름)
- [실무용 체크리스트](#실무용-체크리스트)

---

## 현재 Codex CLI 표면 이해하기

최신 Codex 실습에서 먼저 고쳐야 할 오해는 다음 둘이다.

1. 비대화형 자동화는 `codex exec`가 중심이다.
2. 승인 정책은 `--approval-mode`가 아니라 `-a/--ask-for-approval`와 `--full-auto` 조합으로 이해하는 편이 맞다.

대표 명령:

```bash
codex
codex exec "..."
codex review
codex resume --last
codex fork --last
codex mcp list
codex mcp add <name> --url <url>
```

실무에서 자주 보는 옵션:

| 목적 | 옵션 |
|------|------|
| 작업 디렉토리 지정 | `-C, --cd` |
| 승인 정책 지정 | `-a, --ask-for-approval` |
| 샌드박스 지정 | `-s, --sandbox` |
| 저마찰 자동 실행 | `--full-auto` |
| 추가 쓰기 가능 디렉토리 | `--add-dir` |
| 비대화형 결과 JSONL 출력 | `--json` |
| 마지막 응답 저장 | `-o, --output-last-message` |

---

## AGENTS.md로 기본 작업 규칙 고정하기

Codex 최신 실습에서는 `codex.md`보다 `AGENTS.md` 중심으로 설명하는 편이 안전하다.

프로젝트 루트 `AGENTS.md` 예시:

```markdown
# Project Rules for Codex

## Verification
- Every code change must pass `npm run verify`
- Do not claim completion before reading test output
- If verification fails, fix and rerun

## Editing Rules
- Smallest viable diff
- Reuse existing utilities before adding new abstractions
- Do not disable tests to make the build pass

## TypeScript Rules
- No `any` without a short justification comment
- Prefer narrowing over type assertions

## Reporting
- Final response must include changed files
- Final response must include verification commands actually run
```

핵심은 프롬프트마다 검증 규칙을 반복하는 대신, 저장소 기본 규칙을 `AGENTS.md`에 고정하는 것이다.

---

## 승인 정책과 샌드박스 조합

최신 Codex에서 승인/실행 조합은 이렇게 이해하면 된다.

| 상황 | 추천 조합 | 의미 |
|------|-----------|------|
| 안전한 읽기 중심 탐색 | `-a untrusted -s read-only` | 읽기 위주, 위험 명령은 승인 필요 |
| 일반적인 자동 구현 | `--full-auto` | `on-request + workspace-write` 축약 |
| CI/배치형 비대화 실행 | `-a never -s workspace-write` | 승인 없이 워크스페이스 범위 내 실행 |
| 외부 샌드박스가 이미 있는 특수 환경 | `--dangerously-bypass-approvals-and-sandbox` | 매우 위험, 별도 격리 환경에서만 |

예시:

```bash
# 읽기 중심 조사
codex exec -a untrusted -s read-only \
  "src/auth/ 디렉토리 구조와 로그인 흐름을 요약해줘"

# 일반적인 자동 구현
codex exec --full-auto \
  "tests/api/users.test.ts를 읽고 deleteUser 구현을 완료해줘"

# CI용 비대화형 실행
codex exec -a never -s workspace-write \
  -o .codex-last.txt \
  "npm run verify를 통과하도록 타입 오류를 수정해줘"
```

---

## codex exec로 비대화형 실행하기

`codex exec`는 스크립트, tmux, 배치 자동화의 기본 표면이다.

### 단일 작업 실행

```bash
codex exec --full-auto -C /path/to/repo "
SPEC-006을 구현해줘.

요구사항:
- 파일: src/components/retro/RetroForm.tsx
- KPT 동적 목록
- react-hook-form + zod 사용
- 완료 후 npm run verify 실행
"
```

### 결과를 파일로 저장

```bash
codex exec --full-auto \
  -o .codex/last-message.txt \
  "src/api/users.ts의 deleteUser를 구현하고 검증 결과까지 보고해줘"
```

### JSONL 이벤트로 로그 수집

```bash
codex exec --full-auto --json \
  "src/lib/date.ts의 UTC 파싱 버그를 수정해줘" \
  > .codex/logs/date-fix.jsonl
```

### 프롬프트를 stdin으로 전달

```bash
cat specs/SPEC-010.md | codex exec --full-auto -
```

### 컨텍스트 전달 원칙

최신 CLI 예시에서는 존재가 불확실한 임의 옵션 대신 아래 방식을 우선 사용한다.

1. 작업 루트를 `-C`로 고정한다.
2. 추가 쓰기 범위가 있으면 `--add-dir`를 쓴다.
3. 핵심 파일 경로를 프롬프트 본문에 명시한다.
4. 공통 규칙은 `AGENTS.md`에 둔다.

예시:

```bash
codex exec --full-auto \
  -C /Users/yourname/myapp \
  --add-dir /Users/yourname/shared-packages \
  "
  다음 파일을 기준으로 작업해줘:
  - src/types/index.ts
  - src/lib/supabase/client.ts
  - src/components/retro/RetroForm.tsx

  RetroForm 구현 후 npm run verify 실행
  "
```

---

## review, resume, fork 실전 패턴

### 코드 리뷰

```bash
codex review
```

리뷰용 실습은 보통 다음처럼 설명하면 충분하다.

- 현재 변경사항 기준으로 리뷰한다
- 버그, 리스크, 빠진 테스트를 우선 찾는다
- 구현보다 검증 관점으로 읽는다

### 최근 세션 재개

```bash
codex resume --last
```

긴 작업을 이어서 할 때는 새 프롬프트로 처음부터 설명하기보다 `resume`이 낫다.

### 기존 세션을 복제해서 다른 방향으로 실험

```bash
codex fork --last
```

같은 컨텍스트에서 다른 접근을 비교할 때는 `fork`가 유용하다.

---

## MCP 연결 최신 흐름

Codex 최신 실습에서는 MCP를 직접 CLI에서 추가하고 확인하는 흐름을 먼저 가르치는 편이 낫다.

### Streamable HTTP MCP 서버 추가

```bash
codex mcp add openaiDeveloperDocs \
  --url https://developers.openai.com/mcp
```

### 등록된 MCP 확인

```bash
codex mcp list
```

### stdio 기반 MCP 서버 추가

```bash
codex mcp add github -- npx -y @modelcontextprotocol/server-github
```

환경변수가 필요한 경우:

```bash
codex mcp add github \
  --env GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_TOKEN \
  -- npx -y @modelcontextprotocol/server-github
```

핵심은 실습자가 먼저 `codex mcp add/list/get` 흐름을 익히고, 그다음 프로젝트별 `.mcp.json`이나 다른 도구와의 설정 공유로 넘어가게 만드는 것이다.

---

## 실무용 체크리스트

최신 Codex 실습 문서를 쓸 때 다음 기준을 유지하면 덜 낡는다.

- 명령 자동화 예시는 `codex exec` 기준으로 쓴다.
- 승인 설명은 `--ask-for-approval`, `--sandbox`, `--full-auto` 기준으로 쓴다.
- 프로젝트 규칙은 `AGENTS.md`에 둔다.
- 세션 재개와 분기는 `resume`, `fork`로 설명한다.
- MCP는 `codex mcp add/list` 흐름을 먼저 보여준다.
- 존재가 불확실한 옵션은 예시에서 빼고, 확인한 옵션만 쓴다.

## 참고 문서

- Codex CLI: https://developers.openai.com/codex/cli
- Codex Cloud: https://developers.openai.com/codex/cloud
- OpenAI Docs MCP: https://developers.openai.com/learn/docs-mcp
