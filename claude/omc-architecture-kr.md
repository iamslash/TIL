# OMC (oh-my-claudecode) 동작 원리

> **근거**: `source-inspected` (OMC 플러그인 소스 코드 직접 확인 — hooks.json, .mcp.json, scripts/*.mjs, bridge/*.py), `runtime-observed` (실행 중 훅 주입 및 MCP 도구 호출 관찰), `inferred` (일부 내부 동작은 코드 구조로부터 추정)
>
> **관측 시점**: 2026-03-26, OMC 4.4.4. 버전에 따라 경로, 훅 구성, 도구 목록이 달라질 수 있다.

OMC는 Claude Code의 플러그인 시스템 위에서 동작하는 멀티 에이전트 오케스트레이션 레이어이다. 코드를 직접 실행하는 것이 아니라, **Claude의 행동을 제어하는 메타-레이어**이다.

## 핵심 개념

OMC는 "프롬프트 인젝션을 체계적으로 자동화한 시스템"이다. 훅으로 컨텍스트를 주입하여 Claude가 특정 패턴으로 행동하도록 유도하고, MCP 서버로 추가 도구를 제공하며, 스킬 마크다운으로 복잡한 워크플로우의 지침을 전달한다. (`inferred`: 소스 코드 분석 기반 종합 해석)

## 플러그인 등록

`~/.claude/settings.json`: (`runtime-observed`)
```jsonc
{
  "enabledPlugins": {
    "oh-my-claudecode@omc": true
  }
}
```

## 디렉토리 구조

```text
~/.claude/plugins/cache/omc/oh-my-claudecode/{version}/
├── hooks/hooks.json          # 훅 정의 (이벤트별 셸 커맨드)
├── .mcp.json                 # MCP 서버 정의 (도구 제공)
├── skills/                   # 스킬 정의 (워크플로우 마크다운)
│   ├── autopilot/SKILL.md
│   ├── ralph/SKILL.md
│   ├── cancel/SKILL.md
│   └── ...
├── scripts/                  # 훅 실행 스크립트 (Node.js)
│   ├── keyword-detector.mjs  # 매직 키워드 감지
│   ├── skill-injector.mjs    # 학습된 스킬 주입
│   ├── pre-tool-enforcer.mjs # 도구 호출 전 컨텍스트 주입
│   ├── post-tool-verifier.mjs
│   ├── persistent-mode.cjs   # Stop 시 모드 유지 강제
│   └── ...
├── bridge/                   # MCP 서버 + 외부 런타임
│   ├── mcp-server.cjs        # 메인 MCP 도구 서버 ("t" 네임스페이스)
│   ├── team-mcp.cjs          # Team MCP 서버
│   ├── gyoshu_bridge.py      # Python REPL 브릿지
│   └── runtime-cli.cjs       # Codex/Gemini CLI 런타임
└── dist/                     # 컴파일된 TypeScript 코드
```

## 3가지 확장 메커니즘

### 1. 훅 (Hooks)

이벤트 발생 시 Node.js 스크립트를 실행하여 stdout을 `<system-reminder>`로 Claude에 주입. (`source-inspected`: hooks/hooks.json)

`hooks/hooks.json`에 정의된 이벤트들:

| 이벤트 | 훅 스크립트 | 역할 |
|--------|------------|------|
| `UserPromptSubmit` | `keyword-detector.mjs` | "autopilot", "ralph" 등 매직 키워드 감지 |
| `UserPromptSubmit` | `skill-injector.mjs` | `omc-learned/` 스킬 트리거 매칭 → 주입 |
| `PreToolUse` | `pre-tool-enforcer.mjs` | 도구 호출 전 컨텍스트 주입 |
| `PostToolUse` | `post-tool-verifier.mjs` | 도구 호출 후 검증 컨텍스트 주입 |
| `PostToolUseFailure` | `post-tool-use-failure.mjs` | 실패 시 "fix and continue" 주입 |
| `Stop` | `persistent-mode.cjs` | 모드 활성 시 "계속 작업하라" 강제 |
| `PreCompact` | `pre-compact.mjs` | 컨텍스트 압축 전 중요 정보 보존 |
| `SessionStart` | `session-start.mjs` | 세션 초기화, 프로젝트 메모리 로드 |
| `SessionEnd` | `session-end.mjs` | 세션 종료 정리 |
| `SubagentStart/Stop` | `subagent-tracker.mjs` | 서브에이전트 추적 |
| `PermissionRequest` | `permission-handler.mjs` | Bash 권한 요청 처리 |

훅의 동작 방식: (`source-inspected` + `runtime-observed`)
```text
Claude Code가 이벤트 발생 감지
  → stdin으로 이벤트 데이터를 Node.js 프로세스에 전달
  → Node.js 스크립트가 stdout으로 텍스트 출력
  → Claude Code가 stdout을 <system-reminder>로 감싸서 대화에 주입
  → Claude가 주입된 지침을 읽고 행동
```

### 2. MCP 도구 서버

`.mcp.json`에 정의된 2개의 MCP 서버: (`source-inspected`: .mcp.json)

**`t` (메인 도구 서버)**:
- `state_read/write/clear/list_active` — 모드 상태 관리
- `notepad_read/write` — 세션 메모리
- `project_memory_read/write` — 프로젝트 메모리
- `lsp_*` — LSP 통합 (hover, goto definition, references...)
- `ast_grep_search/replace` — AST 기반 코드 검색/변환
- `python_repl` — Python 실행

**`team` (Team 모드 전용)**:
- `omc_run_team_start/wait/status/cleanup`

### 3. 스킬 (Skills)

마크다운 파일로 된 워크플로우 지침서. 두 가지 경로에서 로드: (`source-inspected`: skill-injector.mjs)

| 종류 | 경로 | 호출 방법 |
|------|------|----------|
| 플러그인 스킬 | `skills/autopilot/SKILL.md` 등 | `/oh-my-claudecode:autopilot` (Skill tool) |
| 학습된 스킬 | `~/.claude/skills/omc-learned/*/SKILL.md` | 트리거 키워드 자동 매칭 (skill-injector 훅) |

## Python 코드 실행 원리

`python_repl` MCP 도구 호출 시: (`source-inspected`: bridge/gyoshu_bridge.py)

```text
Claude → mcp__plugin_oh-my-claudecode_t__python_repl(code="...")
  │
  ▼
bridge/mcp-server.cjs (Node.js MCP 서버)
  │
  ▼
bridge/gyoshu_bridge.py (별도 Python 프로세스)
  │
  │ Unix Socket으로 JSON-RPC 2.0 통신:
  │ → {"jsonrpc":"2.0","id":"req_001","method":"execute","params":{"code":"..."}}
  │ ← {"jsonrpc":"2.0","id":"req_001","result":{"output":"...","variables":[...]}}
  │
  │ 특징:
  │ - 영속적 네임스페이스 (변수 유지)
  │ - stdout/stderr 캡처
  │ - 타임아웃 지원
  │
  ▼
결과를 Claude에 반환
```

## 상태 관리

OMC는 `.omc/` 디렉토리에 모드별 상태를 JSON 파일로 저장: (`source-inspected` + `runtime-observed`)

```text
{worktree}/.omc/
├── state/
│   ├── sessions/{sessionId}/    # 세션별 상태
│   │   ├── autopilot-state.json
│   │   ├── ralph-state.json
│   │   └── ...
│   └── team-state.json          # 레거시 호환
├── notepad.md                   # 세션 메모리
├── project-memory.json          # 프로젝트 영속 메모리
├── plans/                       # 계획 문서
└── logs/                        # 감사 로그
```

## Autopilot/Ralph 모드의 지속 실행 원리

(`source-inspected`: keyword-detector.mjs, persistent-mode.cjs / `runtime-observed`: Stop 훅 메시지 확인)

1. 사용자가 "autopilot" 입력
2. `keyword-detector.mjs`가 `[MAGIC KEYWORD: AUTOPILOT]` 주입
3. Claude가 Skill tool로 `oh-my-claudecode:autopilot` 실행
4. `state_write(mode="autopilot", data={active: true, ...})` 호출
5. 이후 매 도구 호출마다 `PreToolUse` 훅이 "The boulder never stops" 주입
6. Claude가 멈추려 하면 `Stop` 훅의 `persistent-mode.cjs`가:
   - `.omc/state/sessions/{id}/autopilot-state.json` 확인
   - `active: true`이면 "Autopilot not complete. Continue working." 출력
   - Claude가 이를 읽고 작업 재개
7. 모든 작업 완료 시 Claude가 `/oh-my-claudecode:cancel` 호출하여 상태 정리

## `<mnemosyne>` 태그

Mnemosyne(므네모시네)는 그리스 신화의 기억의 여신. OMC에서 skill-injector가 학습된 스킬을 주입할 때 이 태그로 감싼다: (`source-inspected`: skill-injector.mjs / `runtime-observed`: 대화 중 system-reminder에서 확인)

```xml
<system-reminder>
UserPromptSubmit hook additional context:
<mnemosyne>
## Relevant Learned Skills
### hello-slack (user)
...스킬 내용...
</mnemosyne>
</system-reminder>
```

"이전 세션에서 기억해둔 지식을 불러온다"는 시맨틱 힌트.
