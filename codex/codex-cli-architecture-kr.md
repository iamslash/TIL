# Codex CLI 동작 원리

> **근거**: `source-inspected` (`~/.codex/config.toml`, `~/.codex/AGENTS.md`, `~/.codex/prompts/executor.md`, `~/.codex/agents/executor.toml`, `codex --help`, `codex mcp list/get`, `codex features list`), `runtime-observed` (`~/.codex` 디렉터리 구조, session JSONL, shell snapshot 파일), `inferred` (일부 내부 결합 순서와 우선순위는 외부 구조로부터 추정)
>
> **관측 시점**: 2026-03-26, `codex-cli 0.116.0`.

Codex CLI는 단순한 "프롬프트를 보내는 쉘"이 아니라, 다음 다섯 층이 겹쳐진 로컬 에이전트 런타임이다.

1. CLI 바이너리와 TUI/exec 진입점
2. 모델/기본 지침 계층
3. 제어 계층 (`AGENTS.md`, prompts, agents, skills)
4. 도구 계층 (shell, apply_patch, MCP, review/apply/fork/resume)
5. 영속화 계층 (sessions, sqlite, logs, shell snapshots)

## 1. 디렉터리 구조

로컬 `~/.codex/`는 Codex 런타임의 홈 디렉터리이다. (`runtime-observed`)

```text
~/.codex/
├── AGENTS.md
├── config.toml
├── agents/*.toml
├── prompts/*.md
├── skills/
├── sessions/YYYY/MM/DD/*.jsonl
├── archived_sessions/*.jsonl
├── shell_snapshots/*.sh
├── history.jsonl
├── logs_1.sqlite
├── state_5.sqlite
├── models_cache.json
├── cloud-requirements-cache.json
├── version.json
└── auth.json
```

## 2. 제어 흐름의 핵심 개념

Codex의 제어는 한 파일이 독점하지 않는다.

| 계층 | 역할 | 로컬 근거 |
|------|------|------|
| base instructions | 모델 기본 성격/행동 규칙 | `models_cache.json`, session `base_instructions` |
| developer instructions | 설치/환경이 주입하는 추가 규칙 | `config.toml`, session JSONL |
| AGENTS.md | 작업 디렉터리 범위 운영 규칙 | `~/.codex/AGENTS.md`, 프로젝트 `AGENTS.md` |
| role prompt | 특정 역할별 세부 posture | `~/.codex/prompts/*.md` |
| agent TOML | 역할의 모델/effort/metadata | `~/.codex/agents/*.toml` |
| skills | 워크플로우 묶음 | `~/.codex/skills/` |

즉, Codex는 "모델 하나"가 아니라 "기본 지침 + 설치 레이어 + 저장소 규칙 + 역할 프롬프트 + 스킬"의 합성체로 움직인다.

## 3. `AGENTS.md`, `prompts/`, `agents/`의 관계

### `AGENTS.md`

`AGENTS.md`는 작업 디렉터리 범위의 최상위 운영 계약이다. (`source-inspected`)

```text
~/.codex/AGENTS.md
  └── 오케스트레이션 원칙, delegation 규칙, verification 루프, skill 트리거
```

이 파일은 "무슨 자세로 일할지"를 정한다.

### `prompts/*.md`

`prompts/executor.md` 같은 파일은 역할별 작업 surface를 좁힌다. (`source-inspected`)

```text
executor
  - Explore, implement, verify, finish
  - Default: ask last
  - Success criteria / verification checklist
```

즉, prompt 파일은 역할별 행동 절차를 정의한다.

### `agents/*.toml`

`agents/executor.toml`은 역할의 모델/effort/메타데이터를 담는다. (`source-inspected`)

```text
name = "executor"
model = "gpt-5.4-mini"
model_reasoning_effort = "high"
developer_instructions = """ ... """
```

즉, TOML은 "어떤 모델로 어떤 prompt를 실을지"를 정하는 실행 메타데이터에 가깝다.

## 4. 도구 계층

Codex CLI는 셸과 MCP를 모두 도구 평면으로 다룬다.

### 셸/파일/패치

`codex --help`와 현재 실행 환경을 보면 Codex는 최소한 다음 축을 가진다. (`source-inspected` + `runtime-observed`)

- shell command 실행
- patch 적용 (`codex apply`)
- 세션 이어쓰기 (`resume`)
- 세션 fork (`fork`)
- 비대화형 실행 (`exec`)
- 코드 리뷰 (`review`)

### MCP

`codex mcp list/get`에서 두 종류의 전송이 관찰됐다. (`runtime-observed`)

| 전송 방식 | 예시 | 의미 |
|------|------|------|
| `stdio` | `omx_state`, `omx_memory`, `omx_code_intel` | 로컬 프로세스를 자식으로 실행 |
| `streamable_http` | `slack` | 원격 MCP endpoint 사용 |

예:

```text
omx_state
  transport: stdio
  command: node

slack
  transport: streamable_http
  url: https://.../mcp/slack
```

## 5. 설정 계층

`~/.codex/config.toml`은 런타임의 중심 설정 파일이다. (`source-inspected`)

핵심 필드:
- `model = "gpt-5.4"`
- `model_reasoning_effort = "high"`
- `model_context_window = 800000`
- `model_auto_compact_token_limit = 650000`
- `[mcp_servers.*]`
- `[projects."..."].trust_level`
- `[features]`
- `[agents]`
- `[tui]`

즉, Codex는 대부분의 동작을 플래그가 아니라 TOML 기반 구성으로 조절한다.

## 6. 세션/상태 영속화 계층

Codex는 세션과 로그를 여러 저장소로 나눈다. (`runtime-observed`)

| 파일/디렉터리 | 역할 |
|------|------|
| `sessions/YYYY/MM/DD/*.jsonl` | 세션별 상세 이벤트 로그 |
| `archived_sessions/*.jsonl` | 과거 세션 보관 |
| `history.jsonl` | 명령/대화 히스토리 |
| `logs_1.sqlite` | 로그 DB |
| `state_5.sqlite` | 내부 상태 DB |
| `shell_snapshots/*.sh` | 셸 함수/alias/env 스냅샷 |

`shell_snapshots`에 실제 zsh 함수 정의와 alias 정리 코드가 저장되는 것으로 보아, Codex는 도구 실행 전에 셸 상태를 재구성하거나 안정화할 수 있다. (`runtime-observed` + `inferred`)

## 7. feature flag 계층

`codex features list`는 기능의 stage와 effective state를 출력한다. (`runtime-observed`)

관측 예시:
- `multi_agent = true`
- `child_agents_md = true`
- `enable_request_compression = true`
- `shell_snapshot = true`
- `unified_exec = true`
- `voice_transcription = true`

즉, Codex는 "단일 고정 제품"이라기보다 feature gate로 진화하는 런타임이다.

## 8. oh-my-codex(OMX) overlay

이 설치 환경에서는 Codex 위에 OMX가 올라가 있다. (`source-inspected` + `runtime-observed`)

`config.toml`과 `AGENTS.md` 기준으로 OMX는 다음을 추가한다.
- 상위 orchestration 지침
- prompt/agent/skill 체계
- MCP 서버 번들 (`omx_state`, `omx_memory`, `omx_code_intel`, `omx_trace`, `omx_team_run`)
- notify hook
- status line 구성

따라서 현재 로컬 Codex는 "순정 Codex CLI"가 아니라 "Codex CLI + OMX overlay"로 이해하는 것이 정확하다.

## 요약

```text
Codex CLI
├── model + base instructions
├── config.toml
├── AGENTS.md / prompts / agents / skills
├── shell + MCP tool plane
└── sessions / sqlite / logs / shell snapshots
```

이 구조 덕분에 Codex는 단순 채팅이 아니라 "지침 합성 + 도구 실행 + 세션 재개 + 로컬 오케스트레이션"이 가능한 코딩 에이전트로 동작한다.
