# Claude Code 설정 파일 구조

> **근거**: `runtime-observed` (실제 파일 내용 확인), `inferred` (일부 필드의 용도는 이름과 값으로부터 추정)
>
> **관측 시점**: 2026-03-26. Claude Code 버전 업데이트에 따라 필드가 추가/변경될 수 있다.

Claude Code는 여러 JSON 파일에 설정과 상태를 분산 저장한다. 이 문서는 각 파일의 역할과 주요 필드를 정리한다.

## 파일 전체 맵

```text
~/.claude.json                          ← 로컬 상태 + MCP 서버 등록
~/.claude/
├── settings.json                       ← 사용자 설정 (모델, 플러그인, 환경변수)
├── .mcp.json                           ← 글로벌 MCP 서버 (존재 시)
├── plugins/
│   └── installed_plugins.json          ← 설치된 플러그인 목록
{프로젝트루트}/
├── .claude/
│   └── settings.local.json             ← 프로젝트별 권한 설정
├── CLAUDE.md                           ← 프로젝트별 지침 (system 메시지에 포함)
└── .mcp.json                           ← 프로젝트별 MCP 서버 (존재 시)
```

## ~/.claude.json — 로컬 상태 파일

Claude Code의 **런타임 상태**를 저장하는 메인 파일. 사용자가 직접 편집하는 것은 권장되지 않는다.

### MCP 서버 등록

사용자가 `claude mcp add` 명령으로 추가한 MCP 서버가 저장된다.

```jsonc
{
  "mcpServers": {
    "glean_default": {
      "type": "http",
      "url": "https://tinder-be.glean.com/mcp/default"
    },
    "grafana": {
      "type": "http",
      "url": "https://agentisgateway.agentis.ue1az.tinderops.net/mcp/grafana"
    }
  }
}
```

`/mcp` 명령으로 보이는 **Local MCPs**와 **User MCPs**가 이 파일에서 온다. (`runtime-observed`)

### 사용자 계정/구독

```jsonc
{
  "oauthAccount": {
    "accountUuid": "...",
    "emailAddress": "...",
    "organizationUuid": "...",
    "billingType": "...",
    "hasExtraUsageEnabled": true
    // ... 11개 필드
  },
  "userID": "64자 해시",
  "hasAvailableSubscription": false
}
```

### 사용 통계

```jsonc
{
  "numStartups": 96,                     // 총 실행 횟수
  "promptQueueUseCount": 202,            // 프롬프트 큐 사용 횟수
  "toolUsage": {                         // 도구별 사용 횟수
    "Bash": "...", "Read": "...", "Edit": "...", "WebSearch": "..."
  },
  "skillUsage": {                        // 스킬별 사용 횟수
    "oh-my-claudecode:omc-setup": "...", "hud": "..."
  }
}
```

### 피쳐 플래그 / A/B 테스트

Anthropic 서버에서 내려받은 실험 설정 캐시. 기능의 활성화/비활성화를 서버 측에서 제어한다. (`inferred`)

```jsonc
{
  "cachedStatsigGates": {                // 12개 게이트
    "tengu_disable_bypass_permissions_mode": "...",
    "tengu_tool_pear": "..."
  },
  "cachedGrowthBookFeatures": {          // 170개 피쳐 플래그
    "tengu_streaming_tool_execution2": "...",
    "tengu_orchid_trellis": "..."
  }
}
```

### 마이그레이션 상태

모델 버전 업그레이드 시 한 번만 실행되는 마이그레이션 완료 여부.

```jsonc
{
  "sonnet45MigrationComplete": true,
  "opus45MigrationComplete": true,
  "opus46MigrationComplete": true,       // inferred: opusProMigrationComplete 포함
  "thinkingMigrationComplete": true
}
```

### 프로젝트 히스토리

```jsonc
{
  "projects": {                          // 작업한 프로젝트 19개
    "/Users/.../TIL": "...",
    "/Users/.../TinderAndroid": "..."
  },
  "githubRepoPaths": {                   // 연결된 GitHub 저장소 9개
    "iamslash/til": "...",
    "anthropics/claude-code": "..."
  }
}
```

### UI/UX 상태

```jsonc
{
  "tipsHistory": {},                     // 표시된 팁 37개
  "hasCompletedOnboarding": true,
  "lastReleaseNotesSeen": "2.1.84",
  "shiftEnterKeyBindingInstalled": true,
  "deepLinkTerminal": "iTerm"
}
```

### 카테고리 요약

| 카테고리 | 용도 | 사용자가 수정하는가 |
|----------|------|-------------------|
| mcpServers | MCP 서버 등록 | O (`claude mcp add`) |
| oauthAccount | 인증 정보 | X (자동) |
| 사용 통계 | 도구/스킬 사용 횟수 | X (자동) |
| 피쳐 플래그 | Anthropic 서버 측 실험 캐시 | X (서버에서 관리) |
| 마이그레이션 | 모델 업그레이드 완료 여부 | X (자동) |
| 프로젝트 | 작업 프로젝트 기록 | X (자동) |
| UI 상태 | 팁, 온보딩, 키바인딩 | X (자동) |

## ~/.claude/settings.json — 사용자 설정

사용자가 의도적으로 수정하는 설정 파일. (`runtime-observed`)

```jsonc
{
  "env": {
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000"    // 환경변수
  },
  "model": "opus[1m]",                          // 사용 모델
  "enabledPlugins": {                            // 활성화된 플러그인
    "oh-my-claudecode@omc": true,
    "slack@claude-plugins-official": true,
    "telegram@claude-plugins-official": true
  },
  "alwaysThinkingEnabled": true,                 // extended thinking 항상 활성화
  "statusLine": {                                // 상태바 설정
    "type": "command",
    "command": "node .../omc-hud.mjs"
  }
}
```

## {프로젝트}/.claude/settings.local.json — 프로젝트별 권한

프로젝트별로 자동 허용되는 도구/명령어 목록. 사용자가 도구 사용을 승인할 때 자동으로 추가된다. (`runtime-observed`)

```jsonc
{
  "permissions": {
    "allow": [
      "Bash(npm install:*)",
      "Bash(python3:*)",
      "Bash(git add:*)",
      "Bash(git commit:*)",
      "Bash(git push:*)",
      "mcp__glean_default__search",
      "mcp__grafana__search_dashboards",
      "mcp__github__search_code",
      "mcp__atlassian__jira_search"
      // ... 도구 사용 시 자동 누적
    ]
  }
}
```

## MCP 서버 등록 위치 총정리

`/mcp` 명령에 표시되는 서버들의 출처: (`runtime-observed`)

| /mcp 카테고리 | 설정 파일 | 예시 |
|--------------|----------|------|
| **Local MCPs** | `~/.claude.json` → `mcpServers` | atlassian, context7, databricks, figma, filesystem, github, grafana, kibana, phoenix, slack |
| **User MCPs** | `~/.claude.json` → `mcpServers` (동일 파일) | glean_default |
| **Built-in MCPs** | 각 플러그인의 `.mcp.json` | plugin:oh-my-claudecode:t, plugin:oh-my-claudecode:team, plugin:slack:slack, plugin:telegram:telegram |

> Local MCPs와 User MCPs가 같은 파일(`~/.claude.json`)에 있지만 `/mcp` 화면에서 분리되어 표시되는 기준은 명확하지 않다. (`inferred`: 프로젝트 스코프 여부로 구분되는 것으로 추정)

## CLAUDE.md — 프로젝트별 지침

매 API 요청의 `system` 메시지에 포함되는 마크다운 파일. 프로젝트별 코딩 규칙, 빌드 방법, 주의사항 등을 기록한다. Claude Code가 자동으로 읽어서 system prompt에 주입한다. (`runtime-observed`)

탐색 위치 (우선순위는 `inferred` — 정확한 순서는 Claude Code 내부 구현에 따라 다를 수 있음):
- `{프로젝트루트}/CLAUDE.md` — 프로젝트 공유 지침
- `~/.claude/CLAUDE.md` — 글로벌 개인 지침
- `~/.claude/projects/{프로젝트경로}/CLAUDE.md` — 프로젝트별 개인 지침 (git에 포함되지 않음)
