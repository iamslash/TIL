# Codex CLI 설정 파일 구조

> **근거**: `source-inspected` (`~/.codex/config.toml`, `codex mcp list/get`, `codex features list`, `~/.codex/version.json`, `~/.codex/models_cache.json`, `~/.codex/cloud-requirements-cache.json`), `runtime-observed` (`~/.codex` 디렉터리, session 파일 경로, shell snapshot 파일), `inferred` (일부 필드의 내부 사용 방식은 이름과 동작으로부터 추정)
>
> **관측 시점**: 2026-03-26, `codex-cli 0.116.0`.

Codex CLI는 설정, 세션, 캐시, 상태를 `~/.codex/` 아래에 분산 저장한다. 이 문서는 각 파일의 역할과 실제 관찰된 주요 필드를 정리한다.

## 파일 전체 맵

```text
~/.codex/
├── config.toml                    ← 핵심 사용자 설정
├── AGENTS.md                      ← 상위 운영 지침
├── prompts/*.md                   ← 역할별 prompt
├── agents/*.toml                  ← 역할별 모델/메타데이터
├── skills/                        ← 워크플로우 skill
├── sessions/YYYY/MM/DD/*.jsonl    ← 세션 이벤트 로그
├── archived_sessions/*.jsonl      ← 과거 세션 보관
├── history.jsonl                  ← 히스토리
├── shell_snapshots/*.sh           ← 셸 상태 스냅샷
├── logs_1.sqlite                  ← 로그 DB
├── state_5.sqlite                 ← 내부 상태 DB
├── models_cache.json              ← 모델 메타데이터 캐시
├── cloud-requirements-cache.json  ← 클라우드 요구사항 캐시
├── version.json                   ← 최신 버전 체크 캐시
└── auth.json                      ← 인증 정보
```

## `~/.codex/config.toml` — 핵심 설정 파일

사용자가 의도적으로 수정하는 가장 중요한 설정 파일이다. (`source-inspected`)

### 상단 공통 설정

```toml
notify = ["node", ".../notify-hook.js"]
model_reasoning_effort = "high"
developer_instructions = "You have oh-my-codex installed..."
model = "gpt-5.4"
personality = "pragmatic"
model_context_window = 800000
model_auto_compact_token_limit = 650000
```

핵심 의미:

| 키 | 의미 |
|------|------|
| `model` | 기본 모델 |
| `model_reasoning_effort` | 기본 reasoning effort |
| `personality` | 기본 성격 프리셋 |
| `developer_instructions` | 설치 레이어가 추가로 주입하는 developer instructions |
| `model_context_window` | 설정상 최대 컨텍스트 창 |
| `model_auto_compact_token_limit` | 자동 compact 시작 임계값 |
| `notify` | 작업 완료/이벤트 알림 훅 |

### MCP 서버

`[mcp_servers.*]` 블록에 서버별 transport 정보가 저장된다. (`source-inspected`)

```toml
[mcp_servers.omx_state]
command = "node"
args = [".../state-server.js"]
startup_timeout_sec = 5.0

[mcp_servers.slack]
url = "https://.../mcp/slack"
```

여기서 다음 두 가지가 보인다.
- 로컬 stdio 서버는 `command` + `args`
- 원격 MCP 서버는 `url`

`codex mcp get` 출력도 이 설정을 반영한다. (`runtime-observed`)

### 프로젝트 trust

```toml
[projects."/Users/david.sun/prj/github"]
trust_level = "trusted"
```

즉, Codex는 작업 경로별 trust level을 따로 저장한다.

### feature flag

```toml
[features]
voice_transcription = true
multi_agent = true
child_agents_md = true
```

실제 effective state는 `codex features list`로 다시 확인할 수 있다. 설정 파일과 런타임 effective state는 완전히 같다고 단정할 수는 없다. (`runtime-observed` + `inferred`)

### agent/thread/TUI

```toml
[agents]
max_threads = 6
max_depth = 2

[tui]
status_line = ["model-with-reasoning", "git-branch", ...]
```

즉, Codex는 멀티 에이전트 깊이/스레드 수와 TUI 상태줄까지 TOML에서 제어한다.

## `~/.codex/AGENTS.md`

이 파일은 홈 디렉터리 범위의 운영 계약이다. (`source-inspected`)

역할:
- 오케스트레이션 원칙
- delegation 규칙
- skill 트리거
- verification 루프
- model routing

즉, `config.toml`이 "기계적 설정"이라면 `AGENTS.md`는 "행동 규칙"에 가깝다.

## `prompts/*.md`와 `agents/*.toml`

이 둘은 짝으로 이해하는 편이 쉽다.

| 위치 | 역할 |
|------|------|
| `prompts/executor.md` | 역할별 작업 절차와 출력 계약 |
| `agents/executor.toml` | 역할 이름, 모델, reasoning effort, metadata |

관측된 예:
- `prompts/executor.md`는 explore → implement → verify 절차를 정의
- `agents/executor.toml`은 `model = "gpt-5.4-mini"`, `model_reasoning_effort = "high"`를 지정

## `sessions/` — 세션 이벤트 로그

기본 실행은 세션을 JSONL로 저장한다. (`runtime-observed`)

```text
~/.codex/sessions/2026/03/26/rollout-2026-03-26T14-48-59-....jsonl
```

`codex exec --help`에 `--ephemeral`가 별도로 있는 점도 이 동작을 뒷받침한다. (`source-inspected`)

## `shell_snapshots/` — 셸 상태 스냅샷

Codex는 셸 함수/alias/환경 일부를 별도 `.sh` 파일로 저장한다. (`runtime-observed`)

관측된 snapshot 파일에는 다음이 포함된다.
- `unalias -a`
- zsh 함수 정의
- `nvm`, `sdkman`, `vcs_info` 관련 함수

즉, Codex는 셸 환경을 단순 문자열이 아니라 "재구성 가능한 스냅샷"으로 다룬다.

## `history.jsonl`

이 파일은 세션 단위 JSONL과는 별개로, 더 넓은 히스토리를 축적하는 파일로 보인다. (`runtime-observed` + `inferred`)

세션별 상세 복원은 `sessions/`, 전반적 사용 히스토리는 `history.jsonl`로 나뉜다.

## sqlite 파일들

관측된 sqlite 계열 파일:

```text
logs_1.sqlite
logs_1.sqlite-shm
logs_1.sqlite-wal
state_5.sqlite
```

의미:
- `logs_1.sqlite`: 로그 저장소
- `state_5.sqlite`: 내부 상태 저장소
- `-shm`, `-wal`: sqlite WAL 모드 보조 파일

이 DB들의 구체적 테이블 구조는 이번 문서에서 직접 분석하지 않았다. (`runtime-observed`)

## `models_cache.json`

모델 카탈로그와 base instructions 캐시가 들어 있다. (`source-inspected`)

관측된 주요 필드:

| 키 | 의미 |
|------|------|
| `fetched_at` | 캐시 시점 |
| `client_version` | 모델 목록을 받은 클라이언트 버전 |
| `models[].slug` | 모델 식별자 |
| `models[].supported_reasoning_levels` | reasoning effort 목록 |
| `models[].base_instructions` | 모델 기본 지침 텍스트 |
| `models[].context_window` | 모델 메타데이터상의 context window |

이 파일은 "선택 가능한 모델 메타데이터 캐시" 역할을 한다.

## `cloud-requirements-cache.json`

클라우드 작업 조건 또는 entitlement 검증 결과 캐시로 보인다. (`source-inspected` + `inferred`)

관측된 필드:

```jsonc
{
  "signed_payload": {
    "cached_at": "...",
    "expires_at": "...",
    "chatgpt_user_id": "...",
    "account_id": "..."
  },
  "signature": "..."
}
```

즉, Codex는 일부 클라우드 관련 요구사항을 서명된 payload 형태로 로컬에 캐시한다.

## `version.json`

버전 체크 캐시이다. (`source-inspected`)

```jsonc
{
  "latest_version": "0.116.0",
  "last_checked_at": "...",
  "dismissed_version": null
}
```

즉, Codex는 최신 버전 확인 결과를 로컬에 보관한다.

## `auth.json`

인증 자격 정보 파일이다. (`runtime-observed`)

이 파일은 민감 정보가 포함되므로 수동 편집/공유 대상이 아니다.

## 요약

| 파일 | 분류 | 사용자가 직접 수정하는가 |
|------|------|------|
| `config.toml` | 핵심 설정 | O |
| `AGENTS.md` | 운영 지침 | O |
| `prompts/*.md` | 역할 프롬프트 | O |
| `agents/*.toml` | 역할 메타데이터 | O |
| `sessions/*.jsonl` | 세션 로그 | X |
| `shell_snapshots/*.sh` | 셸 스냅샷 | X |
| `models_cache.json` | 모델 캐시 | 보통 X |
| `cloud-requirements-cache.json` | 클라우드 요구사항 캐시 | X |
| `version.json` | 버전 캐시 | X |
| `auth.json` | 인증 정보 | X |
