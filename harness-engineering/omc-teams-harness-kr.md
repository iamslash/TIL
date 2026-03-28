# oh-my-claudecode로 배우는 Teams-first Agent Harness

> **커리큘럼**: 07-CH01 | **난이도**: 중급 | **작성일**: 2026-03-28

## 목차

- [루트부터 읽기: CLAUDE.md, AGENTS.md, .mcp.json의 역할 분담](#루트부터-읽기)
- [agents, hooks, skills, templates 폴더별 역할 분해](#폴더별-역할-분해)
- [Team / Autopilot / Ultrawork: 오케스트레이션 모드 비교](#오케스트레이션-모드-비교)
- [Team 패턴을 내 Agent Harness로 이식하기](#team-패턴-이식하기)

---

## 루트부터 읽기

oh-my-claudecode(이하 OMC)가 Claude를 제어하는 방식은 세 파일의 협력에서 시작된다. 이 세 파일은 각각 다른 레이어에서 동작하며 서로를 보완한다.

### CLAUDE.md — 행동 지침서

CLAUDE.md는 Claude Code가 세션을 시작할 때 자동으로 읽는 마크다운 파일이다. OMC는 이 파일을 두 위치에 배치한다.

```
~/.claude/CLAUDE.md          # 전역 지침 (모든 프로젝트에 적용)
{worktree}/CLAUDE.md         # 프로젝트별 지침 (해당 프로젝트에만 적용)
```

실제 OMC의 전역 CLAUDE.md 구조는 다음 섹션으로 구성된다:

```markdown
# oh-my-claudecode - Intelligent Multi-Agent Orchestration

<operating_principles>
- 전문화된 작업은 가장 적합한 에이전트에게 위임
- 증거 기반 결론 (가정이 아닌 검증)
- 최소 경로 원칙 (직접 실행 > tmux 워커 > 에이전트)
</operating_principles>

<delegation_rules>
- 다중 파일 구현, 리팩토링, 디버깅: 위임
- 사소한 조작, 단일 명령: 직접 실행
- 실질적 코드 변경: executor 에이전트로 라우팅
</delegation_rules>

<agent_catalog>
# 빌드/분석 레인
- explore (haiku): 코드베이스 탐색, 심볼/파일 매핑
- analyst (opus): 요구사항 분석, 수락 기준
- planner (opus): 작업 시퀀싱, 실행 계획
- architect (opus): 시스템 설계, 경계 정의
- executor (sonnet): 코드 구현, 리팩토링
- verifier (sonnet): 완료 검증, 테스트 적합성
</agent_catalog>

<skills>
- autopilot: 아이디어 → 동작하는 코드 전체 자동 실행
- ralph: 검증 포함 자기 참조 루프
- ultrawork: 최대 병렬성, 독립 작업 동시 실행
- team: N개 에이전트 단계별 파이프라인
</skills>
```

**핵심 설계 원칙**: CLAUDE.md는 Claude의 "성격"을 정의한다. 에이전트 카탈로그, 위임 규칙, 스킬 목록이 모두 이 파일에서 선언된다.

### AGENTS.md — 컨텍스트 계층

AGENTS.md는 디렉토리별로 배치되어 계층적 컨텍스트를 형성한다. 루트 AGENTS.md가 전체 규칙을 정의하고, 하위 디렉토리의 AGENTS.md가 지역 규칙을 추가한다.

```
{worktree}/
├── AGENTS.md                # 전체 프로젝트 규칙
├── src/
│   ├── AGENTS.md            # src 디렉토리 전용 규칙
│   └── components/
│       └── AGENTS.md        # 컴포넌트 전용 규칙
└── tests/
    └── AGENTS.md            # 테스트 전용 규칙
```

실제 루트 AGENTS.md 예시:

```markdown
# Project Agent Rules

## Architecture
- 모든 상태 변경은 Redux store를 통해서만 수행
- API 호출은 services/ 레이어에서만 허용

## Code Style
- TypeScript strict mode 필수
- 함수형 컴포넌트 우선, 클래스 컴포넌트 금지

## Testing
- 새 기능에는 반드시 단위 테스트 포함
- 커버리지 임계값: 80% 이상
```

src/AGENTS.md 예시:

```markdown
# Source Directory Rules

## Component Guidelines
- props는 interface로 정의, type alias 사용 금지
- 사이드 이펙트는 useEffect로만 처리
- 컴포넌트당 최대 200줄
```

### .mcp.json — 도구 공급자

.mcp.json은 Model Context Protocol(MCP) 서버를 정의한다. Claude에게 파일 시스템, 코드 인텔리전스, 상태 관리 도구를 제공한다.

```json
{
  "mcpServers": {
    "plugin:oh-my-claudecode:t": {
      "command": "node",
      "args": ["~/.claude/plugins/cache/omc/oh-my-claudecode/4.4.4/bridge/mcp-server.cjs"],
      "env": {
        "OMC_WORKTREE": "${workspaceFolder}"
      }
    },
    "plugin:oh-my-claudecode:team": {
      "command": "node",
      "args": ["~/.claude/plugins/cache/omc/oh-my-claudecode/4.4.4/bridge/team-mcp.cjs"]
    }
  }
}
```

**"t" 네임스페이스 도구 목록**:

| 도구 | 역할 |
|------|------|
| `lsp_diagnostics` | TypeScript/Python 타입 오류 검사 |
| `lsp_goto_definition` | 심볼 정의 위치 탐색 |
| `notepad_read/write` | 세션 메모 읽기/쓰기 |
| `state_read/write` | 오케스트레이션 상태 영속화 |
| `ast_grep_search` | 구조적 코드 패턴 검색 |
| `python_repl` | 지속 Python REPL |

**"team" 네임스페이스 도구**:

| 도구 | 역할 |
|------|------|
| `omc_run_team_start` | 팀 실행 시작 |
| `omc_run_team_wait` | 팀 완료 대기 |
| `omc_run_team_status` | 팀 상태 조회 |
| `omc_run_team_cleanup` | 팀 리소스 정리 |

### 세 파일의 협력 관계

```
CLAUDE.md ──→ "어떤 에이전트를 어떻게 사용할지" (행동 규칙)
    │
AGENTS.md ──→ "이 코드베이스의 규칙은 무엇인지" (컨텍스트)
    │
.mcp.json ──→ "어떤 도구를 쓸 수 있는지" (능력)
```

Claude는 세션 시작 시 이 세 파일을 모두 읽고 통합하여 행동 방침을 결정한다.

---

## 폴더별 역할 분해

OMC 플러그인 디렉토리 구조와 각 폴더의 역할을 분석한다.

```
~/.claude/plugins/cache/omc/oh-my-claudecode/4.4.4/
├── agents/           # 에이전트 정의 (역할별 프롬프트)
├── hooks/            # 이벤트 훅 (자동 컨텍스트 주입)
├── skills/           # 스킬 정의 (워크플로우 마크다운)
├── scripts/          # 훅 실행 스크립트 (Node.js)
├── bridge/           # MCP 서버 및 런타임 브릿지
├── dist/             # 컴파일된 TypeScript
└── src/              # TypeScript 소스
    ├── agents/       # 에이전트 정의 TypeScript
    ├── hooks/        # 훅 핸들러
    └── skills/       # 스킬 로더
```

### agents/ 폴더 — 역할별 에이전트 정의

각 에이전트는 독립된 정의 파일을 가진다. 핵심 에이전트들의 실제 구조:

**explore 에이전트** (haiku 모델, 코드 탐색 전용):

```typescript
// src/agents/definitions.ts (일부)
export const AGENT_DEFINITIONS: AgentDefinition[] = [
  {
    id: "explore",
    name: "Explorer",
    model: "haiku",
    description: "내부 코드베이스 탐색, 심볼/파일 매핑",
    preamble: `
      You are Explorer. Your mission is to map the codebase quickly.
      Use Glob, Grep, and Read tools to discover structure.
      Report findings concisely. Do NOT implement changes.
      Focus: file locations, symbol definitions, dependency graphs.
    `,
    tools: ["Glob", "Grep", "Read", "Bash"],
    maxTokens: 4096
  },
  {
    id: "executor",
    name: "Executor",
    model: "sonnet",
    description: "코드 구현, 리팩토링, 기능 개발",
    preamble: `
      You are Executor. Your mission is to implement precisely.
      Smallest viable diff. Verify with lsp_diagnostics after each change.
      No new abstractions for single-use logic.
      Always show fresh build/test output before claiming completion.
    `,
    tools: ["Edit", "Write", "Bash", "Read", "Glob", "Grep", "lsp_diagnostics"],
    maxTokens: 16384
  },
  {
    id: "verifier",
    name: "Verifier",
    model: "sonnet",
    description: "완료 증거 수집, 클레임 검증, 테스트 적합성",
    preamble: `
      You are Verifier. Your mission is evidence-backed confidence.
      Run tests, check diagnostics, validate claims with actual output.
      Report: pass/fail with specific evidence. No assumptions.
    `,
    tools: ["Bash", "Read", "lsp_diagnostics"],
    maxTokens: 8192
  }
];
```

**architect 에이전트** (opus 모델, 설계 전담):

```typescript
{
  id: "architect",
  name: "Architect",
  model: "opus",
  description: "시스템 설계, 경계 정의, 장기 트레이드오프",
  preamble: `
    You are Architect. Your mission is long-horizon system design.
    Think in terms of: interfaces, boundaries, data flow, failure modes.
    Produce: ADRs (Architecture Decision Records), interface contracts.
    Avoid: implementation details, short-term optimizations.
  `,
  tools: ["Read", "Glob", "Grep"],
  maxTokens: 32768
}
```

### hooks/ 폴더 — 자동 컨텍스트 주입

훅은 Claude Code의 이벤트 시스템에 연결되어 자동으로 컨텍스트를 주입한다.

```json
// hooks/hooks.json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "node ~/.claude/plugins/cache/omc/.../scripts/pre-tool-enforcer.mjs"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "command",
            "command": "node ~/.claude/plugins/cache/omc/.../scripts/post-tool-verifier.mjs"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "node ~/.claude/plugins/cache/omc/.../scripts/persistent-mode.cjs"
          }
        ]
      }
    ]
  }
}
```

훅의 실제 동작 — `pre-tool-enforcer.mjs` 핵심 로직:

```javascript
// scripts/pre-tool-enforcer.mjs (개념적 재현)
import { readFileSync } from 'fs';
import { join } from 'path';

const input = JSON.parse(process.stdin.read());
const { tool_name, tool_input, session_id, cwd } = input;

// 1. 병렬 실행 힌트 주입
if (tool_name === 'Bash') {
  const hint = "Use parallel execution for independent tasks. " +
               "Use run_in_background for long operations.";
  console.log(JSON.stringify({
    type: "system_reminder",
    content: `PreToolUse:Bash hook additional context: ${hint}`
  }));
}

// 2. 상태 파일 존재 확인 → ralph/ultrawork 모드 지속
const statePath = join(cwd, '.omc/state/ralph-state.json');
try {
  const state = JSON.parse(readFileSync(statePath, 'utf8'));
  if (state.active) {
    console.log(JSON.stringify({
      type: "system_reminder",
      content: "The boulder never stops rolling. Continue working."
    }));
  }
} catch { /* 상태 없음 = 정상 */ }
```

### skills/ 폴더 — 워크플로우 마크다운

각 스킬은 SKILL.md 파일로 정의된 워크플로우 지침이다.

```
skills/
├── autopilot/SKILL.md
├── ralph/SKILL.md
├── team/SKILL.md
├── ultrawork/SKILL.md
├── plan/SKILL.md
└── cancel/SKILL.md
```

실제 team/SKILL.md 구조 (개념적 재현):

```markdown
# Team Skill

## Trigger
Keywords: "team", "coordinated team", "team ralph"

## Pipeline
team-plan → team-prd → team-exec → team-verify → team-fix (loop)

## Stage: team-plan
Agents: explore (haiku) + planner (opus)
Output: execution plan, file list, dependency graph

## Stage: team-prd
Agents: analyst (opus)
Output: acceptance criteria, scope definition

## Stage: team-exec
Agents: executor (sonnet) + specialists as needed
Specialists: designer, build-fixer, writer, test-engineer, deep-executor
Output: implemented code changes

## Stage: team-verify
Agents: verifier (sonnet) + security-reviewer/code-reviewer as needed
Decision: complete | team-fix | failed

## Stage: team-fix (loop, max 3 attempts)
Agents: executor/build-fixer/debugger (by defect type)
→ returns to team-exec or team-verify

## State Persistence
state_write(mode="team") tracks:
- current_phase
- team_name
- fix_loop_count
- linked_ralph
- stage_history
```

---

## 오케스트레이션 모드 비교

### Team 모드

단계별 파이프라인으로 N개 에이전트가 협력한다. 각 단계는 명확한 입출력을 가지며 이전 단계의 결과를 다음 단계로 전달한다.

```
사용자 요청
    ↓
team-plan (explore + planner)
    → 실행 계획, 파일 목록
    ↓
team-prd (analyst)
    → 수락 기준, 범위 정의
    ↓
team-exec (executor + specialists)
    → 구현된 코드 변경
    ↓
team-verify (verifier + reviewers)
    → 검증 결과
    ↓
[complete] 또는 team-fix (loop)
```

실행 예시:

```bash
# Claude Code에서 /team 스킬 호출
/oh-my-claudecode:team

# 또는 트리거 키워드로 자동 활성화
"team: 사용자 인증 시스템 구현해줘"
```

### Autopilot 모드

아이디어에서 동작하는 코드까지 전체를 자율 실행한다. 사용자 개입 없이 탐색→계획→구현→검증→수정을 반복한다.

```
아이디어 입력
    ↓
탐색 (codebase 이해)
    ↓
계획 (무엇을 변경할지)
    ↓
구현 (executor)
    ↓
검증 (verifier)
    ↓
수정 (필요 시 루프)
    ↓
완료 보고
```

실행 예시:

```bash
# 트리거 키워드
"autopilot: JWT 기반 로그인 API 만들어줘"
"build me: 실시간 채팅 기능"
"I want a: 파일 업로드 컴포넌트"
```

### Ultrawork 모드

독립적인 작업들을 최대 병렬로 동시 실행한다. 의존성이 없는 작업은 동시에 에이전트를 생성하여 처리한다.

```
작업 목록
├── 작업A ──→ executor-1 (병렬)
├── 작업B ──→ executor-2 (병렬)
├── 작업C ──→ executor-3 (병렬)
└── 작업D ──→ executor-4 (병렬)
    ↓ (모두 완료 후)
verifier (통합 검증)
```

실행 예시:

```bash
# 트리거 키워드
"ulw: 다음 5개 버그 모두 수정해줘: #1 #2 #3 #4 #5"
"ultrawork: 모든 컴포넌트에 TypeScript 타입 추가"
```

### 비교표

| 항목 | Team | Autopilot | Ultrawork |
|------|------|-----------|-----------|
| **주요 패턴** | 단계별 파이프라인 | 자율 루프 | 최대 병렬성 |
| **에이전트 수** | 역할별 N개 | 1개 (자율) | 작업당 1개 |
| **사용자 개입** | 단계 전환 시 선택 | 최소 | 없음 |
| **적합한 작업** | 복잡한 기능 개발 | 명확한 단일 목표 | 독립적 다수 작업 |
| **오버헤드** | 높음 (단계 조율) | 중간 | 낮음 (직렬 없음) |
| **오류 복구** | team-fix 루프 | 자체 수정 | 개별 에이전트 재실행 |
| **상태 영속화** | 단계별 state 파일 | 없음 | 없음 |
| **모델 비용** | opus + sonnet 혼합 | sonnet 위주 | sonnet 다수 |

### 선택 기준

```
작업 유형에 따른 모드 선택:

복잡한 기능 (다수 파일, 아키텍처 결정 필요)
    → Team 모드

명확한 단일 목표 (e.g., "이 버그 고쳐줘")
    → Autopilot 모드

독립적인 다수 작업 (e.g., "이 10개 파일 리팩토링")
    → Ultrawork 모드

지속적 완료가 필요한 장기 작업
    → Ralph 모드 (Autopilot + persistence loop)
```

---

## Team 패턴 이식하기

OMC의 Team 패턴을 최소화하여 자신의 Agent Harness로 이식하는 단계별 가이드다. 전체 OMC를 설치하지 않고 핵심 패턴만 가져온다.

### 이식 목표

```
최소 Team Harness:
├── CLAUDE.md          # 에이전트 카탈로그 + 위임 규칙
├── AGENTS.md          # 프로젝트 컨텍스트
├── .claude/
│   └── settings.json  # 훅 설정
└── harness/
    ├── agents.md      # 에이전트 역할 정의
    ├── pipeline.md    # 파이프라인 정의
    └── state.json     # 실행 상태
```

### 1단계: 에이전트 카탈로그 정의

프로젝트 루트에 CLAUDE.md를 작성한다:

```markdown
# My Agent Harness

<agent_catalog>
## 사용 가능한 에이전트

### planner (opus)
역할: 작업 분해, 실행 계획 작성
트리거: 복잡한 기능 요청 시
출력: 단계별 TODO 목록, 파일 변경 계획

### executor (sonnet)
역할: 코드 구현, 파일 수정
트리거: 명확한 구현 작업
출력: 변경된 파일, 빌드 결과

### verifier (sonnet)
역할: 구현 검증, 테스트 실행
트리거: 구현 완료 후
출력: 통과/실패 증거, 진단 결과
</agent_catalog>

<delegation_rules>
- 2개 이상 파일 변경: executor에게 위임
- 설계 결정: planner에게 먼저 물어보기
- 최종 확인: 항상 verifier로 검증
</delegation_rules>

<pipeline>
plan → exec → verify → [fix → verify]*
</pipeline>
```

### 2단계: 파이프라인 상태 관리

간단한 상태 파일로 단계 추적:

```bash
# harness/state.json 초기화
cat > harness/state.json << 'EOF'
{
  "phase": "plan",
  "task": "",
  "history": [],
  "fix_count": 0,
  "max_fixes": 3
}
EOF
```

파이프라인 전환 스크립트 (`harness/pipeline.sh`):

```bash
#!/bin/bash
# harness/pipeline.sh - 파이프라인 단계 전환

set -euo pipefail

STATE_FILE="harness/state.json"
PHASE=$(jq -r '.phase' "$STATE_FILE")

transition() {
  local next_phase="$1"
  local reason="$2"

  # 이력 추가
  jq --arg phase "$PHASE" \
     --arg next "$next_phase" \
     --arg reason "$reason" \
     --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
     '.history += [{"from": $phase, "to": $next, "reason": $reason, "at": $ts}] |
      .phase = $next' \
     "$STATE_FILE" > /tmp/state.tmp && mv /tmp/state.tmp "$STATE_FILE"

  echo "Phase transition: $PHASE → $next_phase ($reason)"
}

case "$PHASE" in
  plan)
    echo "=== PLAN PHASE ==="
    echo "Planner: 작업을 분해하고 실행 계획을 작성하세요."
    echo "완료 후: transition exec '계획 완료'"
    ;;
  exec)
    echo "=== EXEC PHASE ==="
    echo "Executor: 계획에 따라 구현하세요."
    echo "완료 후: transition verify '구현 완료'"
    ;;
  verify)
    echo "=== VERIFY PHASE ==="
    echo "Verifier: 구현을 검증하세요."
    fix_count=$(jq '.fix_count' "$STATE_FILE")
    max_fixes=$(jq '.max_fixes' "$STATE_FILE")
    echo "Fix 시도 횟수: $fix_count / $max_fixes"
    ;;
  fix)
    fix_count=$(jq '.fix_count' "$STATE_FILE")
    max_fixes=$(jq '.max_fixes' "$STATE_FILE")
    if [ "$fix_count" -ge "$max_fixes" ]; then
      echo "최대 fix 횟수 초과. 수동 개입 필요."
      transition failed "max fixes exceeded"
    else
      jq '.fix_count += 1' "$STATE_FILE" > /tmp/state.tmp && mv /tmp/state.tmp "$STATE_FILE"
      echo "Fix #$((fix_count + 1)): 문제를 수정하세요."
    fi
    ;;
esac
```

### 3단계: 훅으로 자동 컨텍스트 주입

`.claude/settings.json`에 훅 등록:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "node harness/hooks/pre-bash.mjs"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "node harness/hooks/post-edit.mjs"
          }
        ]
      }
    ]
  }
}
```

`harness/hooks/pre-bash.mjs`:

```javascript
// harness/hooks/pre-bash.mjs
import { readFileSync } from 'fs';

const input = JSON.parse(await new Promise(resolve => {
  let data = '';
  process.stdin.on('data', chunk => data += chunk);
  process.stdin.on('end', () => resolve(data));
}));

// 현재 파이프라인 단계 읽기
let phase = 'unknown';
try {
  const state = JSON.parse(readFileSync('harness/state.json', 'utf8'));
  phase = state.phase;
} catch {}

// 단계별 힌트 주입
const hints = {
  plan: "현재 PLAN 단계: 구현하지 말고 계획만 세우세요.",
  exec: "현재 EXEC 단계: 계획대로 최소한의 변경만 하세요.",
  verify: "현재 VERIFY 단계: 테스트를 실행하고 증거를 수집하세요.",
  fix: "현재 FIX 단계: 실패한 테스트의 근본 원인을 수정하세요."
};

if (hints[phase]) {
  process.stdout.write(JSON.stringify({
    type: "system_reminder",
    content: `Harness hint: ${hints[phase]}`
  }));
}
```

`harness/hooks/post-edit.mjs`:

```javascript
// harness/hooks/post-edit.mjs
const input = JSON.parse(await new Promise(resolve => {
  let data = '';
  process.stdin.on('data', chunk => data += chunk);
  process.stdin.on('end', () => resolve(data));
}));

const { tool_name, tool_response } = input;

// 편집 후 항상 LSP 검사 권장
process.stdout.write(JSON.stringify({
  type: "system_reminder",
  content: `PostEdit: lsp_diagnostics를 실행하여 타입 오류를 확인하세요.`
}));
```

### 4단계: 실제 실행 예시

```bash
# 1. 새 작업 시작
echo '{"phase":"plan","task":"JWT 인증 추가","history":[],"fix_count":0,"max_fixes":3}' \
  > harness/state.json

# 2. Claude에게 작업 지시 (PLAN 단계)
# Claude는 CLAUDE.md의 규칙에 따라 planner 역할로 동작
# "JWT 인증 시스템을 추가하려고 합니다. 계획을 세워주세요."

# 3. 계획 완료 후 EXEC 단계로 전환
bash harness/pipeline.sh  # 현재 단계 확인
jq '.phase = "exec"' harness/state.json > /tmp/s.tmp && mv /tmp/s.tmp harness/state.json

# 4. 구현 (EXEC 단계)
# Claude는 executor 역할로 동작

# 5. VERIFY 단계로 전환
jq '.phase = "verify"' harness/state.json > /tmp/s.tmp && mv /tmp/s.tmp harness/state.json

# 6. 검증 실패 시 FIX 단계
jq '.phase = "fix"' harness/state.json > /tmp/s.tmp && mv /tmp/s.tmp harness/state.json
bash harness/pipeline.sh  # fix count 자동 증가
jq '.phase = "verify"' harness/state.json > /tmp/s.tmp && mv /tmp/s.tmp harness/state.json
```

### 이식 결과 디렉토리 구조

```
my-project/
├── CLAUDE.md                    # 에이전트 카탈로그 + 위임 규칙
├── AGENTS.md                    # 프로젝트 컨텍스트 규칙
├── .claude/
│   └── settings.json            # 훅 설정
├── harness/
│   ├── agents.md                # 에이전트 역할 상세 정의
│   ├── pipeline.sh              # 파이프라인 단계 전환
│   ├── state.json               # 현재 실행 상태
│   └── hooks/
│       ├── pre-bash.mjs         # Bash 호출 전 컨텍스트 주입
│       └── post-edit.mjs        # 파일 편집 후 검사 권장
└── src/
    └── ...
```

### 검증 체크리스트

이식이 제대로 됐는지 확인하는 항목:

```bash
# 1. 훅이 등록됐는지 확인
cat .claude/settings.json | jq '.hooks'

# 2. 상태 파일이 올바른지 확인
cat harness/state.json | jq '.'

# 3. Claude에게 테스트 요청
# "harness 상태를 확인하고 현재 단계를 알려줘"
# → Claude가 state.json을 읽고 단계를 보고해야 함

# 4. 훅이 동작하는지 확인
# Bash 명령 실행 시 "Harness hint:" 메시지가 시스템 리마인더로 표시돼야 함
```

---

## 핵심 정리

**OMC Team 패턴의 본질**은 세 가지다:

1. **역할 분리**: 에이전트마다 명확한 역할과 제약 (planner는 계획만, executor는 구현만, verifier는 검증만)
2. **단계 게이팅**: 이전 단계의 출력이 다음 단계의 입력. 건너뛰지 않음
3. **상태 영속화**: 세션이 끊겨도 어느 단계에 있는지 파일로 추적

이 세 원칙만 지키면 OMC의 복잡한 인프라 없이도 효과적인 Agent Harness를 운영할 수 있다.
