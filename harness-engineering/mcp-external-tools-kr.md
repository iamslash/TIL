# MCP로 외부 도구 연결하기

> 03-CH03: AI를 코드 밖으로 꺼내기 — Model Context Protocol 실전 가이드

## 목차

- [MCP가 왜 중요한가](#mcp가-왜-중요한가)
- [Claude Code에서 Codex 호출하는 MCP 도구 만들기](#claude-code에서-codex-호출하는-mcp-도구-만들기)
- [GitHub MCP 서버 연결](#github-mcp-서버-연결)
- [DB 연결 MCP 예시 (PostgreSQL)](#db-연결-mcp-예시-postgresql)
- [브라우저 MCP 연결](#브라우저-mcp-연결)
- [Claude Code와 Codex에서 MCP 설정 공유하기](#claude-code와-codex에서-mcp-설정-공유하기)
- [커스텀 MCP 서버 만들기 (Node.js)](#커스텀-mcp-서버-만들기-nodejs)

---

## MCP가 왜 중요한가

AI 모델은 기본적으로 텍스트를 읽고 쓰는 것만 할 수 있다. 하지만 실제 개발 작업은 그보다 훨씬 다양하다.

- GitHub에서 PR을 열거나 이슈를 조회한다
- PostgreSQL에서 실제 데이터를 쿼리한다
- 브라우저를 자동화해 화면을 캡처한다
- 외부 API를 호출해 결과를 받아온다

**MCP(Model Context Protocol)**는 이 모든 것을 가능하게 하는 표준 인터페이스다. AI 모델과 외부 도구 사이의 "USB 포트" 역할을 한다. MCP 서버는 JSON-RPC 위에서 동작하며, 모델이 `tools/call` 메시지를 보내면 서버가 실제 작업을 수행하고 결과를 돌려준다.

```
Claude Code / Codex
       |
    MCP 클라이언트
       |  (stdio / SSE / HTTP)
    MCP 서버
       |
  실제 도구 (GitHub API, PostgreSQL, 브라우저 ...)
```

MCP가 없으면 AI는 코드를 제안하는 데 그친다. MCP가 있으면 AI가 직접 코드를 실행하고, 결과를 확인하고, 다음 단계를 결정한다.

---

## Claude Code에서 Codex 호출하는 MCP 도구 만들기

### 목표

Claude Code 세션 안에서 Codex CLI를 MCP 도구로 호출해, 두 모델을 협력시킨다.

### 1단계: 프로젝트 루트에 `.mcp.json` 생성

```json
{
  "mcpServers": {
    "codex-runner": {
      "type": "stdio",
      "command": "node",
      "args": ["/Users/yourname/.mcp/codex-runner/index.js"],
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

> `.mcp.json`은 프로젝트 루트에 두면 Claude Code가 자동으로 읽는다.
> `~/.claude/settings.json`의 `mcpServers` 블록에 넣으면 전역 적용된다.

### 2단계: Codex Runner MCP 서버 작성

```bash
mkdir -p ~/.mcp/codex-runner
cd ~/.mcp/codex-runner
npm init -y
npm install @modelcontextprotocol/sdk
```

```javascript
// ~/.mcp/codex-runner/index.js
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { execSync } from "child_process";

const server = new McpServer({
  name: "codex-runner",
  version: "1.0.0",
});

server.tool(
  "run_codex",
  "Codex CLI를 사용해 코드를 생성하거나 수정한다",
  {
    prompt: z.string().describe("Codex에게 전달할 지시사항"),
    workdir: z.string().optional().describe("작업 디렉토리 (기본값: 현재 디렉토리)"),
    approvalPolicy: z.enum(["untrusted", "on-request", "never"]).default("on-request"),
    sandbox: z.enum(["read-only", "workspace-write", "danger-full-access"]).default("workspace-write"),
    fullAuto: z.boolean().default(false),
  },
  async ({ prompt, workdir, approvalPolicy, sandbox, fullAuto }) => {
    const cwd = workdir ?? process.cwd();
    const escaped = prompt.replace(/"/g, '\\"');
    const mode = fullAuto
      ? `codex exec --full-auto "${escaped}"`
      : `codex exec -a ${approvalPolicy} -s ${sandbox} "${escaped}"`;

    try {
      const output = execSync(mode, {
        cwd,
        encoding: "utf-8",
        timeout: 120_000,
        env: { ...process.env },
      });
      return {
        content: [{ type: "text", text: output }],
      };
    } catch (err) {
      return {
        content: [{ type: "text", text: `오류: ${err.message}` }],
        isError: true,
      };
    }
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
```

### 3단계: Claude Code에서 사용하기

Claude Code 대화 중에 다음처럼 사용한다:

```
사용자: src/utils/parser.ts에 있는 parseDate 함수를 ISO 8601도 지원하도록
        Codex한테 시켜줘.

Claude: run_codex 도구를 호출합니다.
  prompt: "src/utils/parser.ts의 parseDate 함수가 ISO 8601 형식도
           파싱할 수 있도록 수정해줘. 기존 테스트는 깨지면 안 돼."
  workdir: "/Users/yourname/myproject"
  approvalPolicy: "on-request"
  sandbox: "workspace-write"
  fullAuto: false
```

승인/샌드박스 직접 조합 대신 저마찰 자동 실행을 원하면:

```
Claude: run_codex 도구를 호출합니다.
  prompt: "tests/date.test.ts를 기준으로 parseDate 버그를 수정하고 검증해줘."
  workdir: "/Users/yourname/myproject"
  fullAuto: true
```

## Codex CLI에서 MCP를 직접 연결하는 최신 흐름

Claude Code 안에서 Codex를 호출하는 패턴과 별개로, Codex 자체에 MCP를 붙일 때는 CLI 흐름을 먼저 익히는 편이 좋다.

### Streamable HTTP MCP 추가

```bash
codex mcp add openaiDeveloperDocs \
  --url https://developers.openai.com/mcp
```

### stdio 기반 MCP 추가

```bash
codex mcp add github --env GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_TOKEN \
  -- npx -y @modelcontextprotocol/server-github
```

### 현재 등록 상태 확인

```bash
codex mcp list
```

이 흐름을 먼저 설명한 뒤 프로젝트별 `.mcp.json` 또는 다른 도구와의 설정 공유로 넘어가면 실습자가 덜 헷갈린다.

---

## GitHub MCP 서버 연결

### 설치

```bash
# GitHub MCP 서버는 공식 패키지로 제공된다
npm install -g @modelcontextprotocol/server-github
```

### `.mcp.json` 설정

```json
{
  "mcpServers": {
    "github": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

환경변수 설정:

```bash
export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### PR 생성 예시

Claude Code 대화:

```
사용자: feature/add-login 브랜치를 main으로 PR 열어줘.
        제목은 "feat: 로그인 기능 추가"이고,
        본문에는 변경사항 요약 넣어줘.

Claude: GitHub MCP의 create_pull_request 도구를 호출합니다.
  owner: "myorg"
  repo: "myapp"
  title: "feat: 로그인 기능 추가"
  head: "feature/add-login"
  base: "main"
  body: |
    ## 변경사항
    - JWT 기반 인증 추가
    - 로그인/로그아웃 엔드포인트 구현
    - 세션 만료 처리
```

### 이슈 조회 예시

```
사용자: 이번 주에 열린 버그 이슈 목록 보여줘.

Claude: list_issues 도구를 호출합니다.
  owner: "myorg"
  repo: "myapp"
  labels: ["bug"]
  state: "open"
  since: "2026-03-24T00:00:00Z"
```

실제 응답 (MCP가 반환하는 JSON):

```json
[
  {
    "number": 142,
    "title": "로그인 후 리다이렉트 루프 발생",
    "created_at": "2026-03-25T09:12:00Z",
    "labels": ["bug", "priority:high"]
  },
  {
    "number": 138,
    "title": "토큰 만료 메시지가 한국어로 표시 안 됨",
    "created_at": "2026-03-24T14:30:00Z",
    "labels": ["bug", "i18n"]
  }
]
```

---

## DB 연결 MCP 예시 (PostgreSQL)

### 설치

```bash
npm install -g @modelcontextprotocol/server-postgres
```

### `.mcp.json` 설정

```json
{
  "mcpServers": {
    "postgres": {
      "type": "stdio",
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-postgres",
        "postgresql://myuser:mypassword@localhost:5432/mydb"
      ]
    }
  }
}
```

보안 권고: 프로덕션 DB에는 읽기 전용 계정을 사용한다.

```sql
-- 읽기 전용 역할 생성
CREATE ROLE mcp_readonly LOGIN PASSWORD 'safe_password';
GRANT CONNECT ON DATABASE mydb TO mcp_readonly;
GRANT USAGE ON SCHEMA public TO mcp_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO mcp_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO mcp_readonly;
```

### Claude Code에서 쿼리 실행

```
사용자: 지난 7일간 가장 많이 주문한 상품 top 5 알려줘.

Claude: postgres MCP의 query 도구를 호출합니다.
  sql: |
    SELECT
      p.name AS product_name,
      SUM(oi.quantity) AS total_ordered
    FROM order_items oi
    JOIN products p ON p.id = oi.product_id
    JOIN orders o ON o.id = oi.order_id
    WHERE o.created_at >= NOW() - INTERVAL '7 days'
    GROUP BY p.id, p.name
    ORDER BY total_ordered DESC
    LIMIT 5;
```

### 스키마 탐색

```
사용자: users 테이블 구조 보여줘.

Claude: query 도구를 호출합니다.
  sql: |
    SELECT column_name, data_type, is_nullable, column_default
    FROM information_schema.columns
    WHERE table_name = 'users'
    ORDER BY ordinal_position;
```

---

## 브라우저 MCP 연결

### Playwright 기반 MCP 서버 설치

```bash
npm install -g @executeautomation/playwright-mcp-server
```

### `.mcp.json` 설정

```json
{
  "mcpServers": {
    "playwright": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@executeautomation/playwright-mcp-server"],
      "env": {
        "BROWSER": "chromium",
        "HEADLESS": "true"
      }
    }
  }
}
```

### 실제 사용 예시

스크린샷 캡처:

```
사용자: https://staging.myapp.com/login 페이지 스크린샷 찍어줘.

Claude: playwright_screenshot 도구를 호출합니다.
  url: "https://staging.myapp.com/login"
  path: "/tmp/login-screenshot.png"
  fullPage: true
```

폼 자동화:

```
사용자: 스테이징 환경에서 회원가입 플로우 테스트해줘.
        test@example.com / TestPassword123 으로 가입해봐.

Claude: playwright_navigate → playwright_fill → playwright_click 순서로 호출합니다.

  1. navigate: url="https://staging.myapp.com/signup"
  2. fill: selector="#email", value="test@example.com"
  3. fill: selector="#password", value="TestPassword123"
  4. click: selector="button[type=submit]"
  5. screenshot: path="/tmp/after-signup.png"
```

---

## Claude Code와 Codex에서 MCP 설정 공유하기

### 공유 설정 패턴

두 도구가 같은 MCP 서버를 사용하려면 설정 파일을 공유하거나 공통 위치를 지정한다.

**방법 1: 프로젝트 루트 `.mcp.json` (권장)**

```
myproject/
├── .mcp.json          ← Claude Code와 Codex 둘 다 여기서 읽는다
├── src/
└── ...
```

`.mcp.json` 파일 하나로 두 도구가 같은 MCP 서버 집합을 사용한다. Claude Code는 프로젝트 루트를 자동으로 탐지하고, Codex도 동일한 규칙을 따른다.

**방법 2: 전역 설정 분리**

Claude Code 전역 설정 (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "github": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "postgres": {
      "type": "stdio",
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-postgres",
        "${DATABASE_URL}"
      ]
    }
  }
}
```

Codex 전역 설정 (`~/.codex/config.yaml`):

```yaml
mcp_servers:
  - name: github
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "${GITHUB_TOKEN}"
  - name: postgres
    command: npx
    args: ["-y", "@modelcontextprotocol/server-postgres", "${DATABASE_URL}"]
```

**방법 3: 심볼릭 링크로 단일 소스 유지**

```bash
# 공통 설정 파일을 한 곳에 두고
mkdir -p ~/.config/mcp
cat > ~/.config/mcp/shared.json << 'EOF'
{
  "mcpServers": {
    "github": { ... },
    "postgres": { ... }
  }
}
EOF

# Claude Code용
ln -sf ~/.config/mcp/shared.json ~/.claude/mcp-shared.json

# 프로젝트별 .mcp.json에서 참조 (Claude Code는 이를 merge함)
```

### 환경변수로 민감 정보 분리

```bash
# ~/.zshrc 또는 ~/.bashrc
export GITHUB_TOKEN="ghp_..."
export DATABASE_URL="postgresql://user:pass@host:5432/db"
export OPENAI_API_KEY="sk-..."

# 선택적으로 direnv 사용 (프로젝트별 환경변수)
# .envrc 파일에
export DATABASE_URL="postgresql://localhost:5432/myproject_dev"
```

---

## 커스텀 MCP 서버 만들기 (Node.js)

실무에서 가장 자주 필요한 패턴은 사내 API나 독자적인 도구를 MCP로 감싸는 것이다.

### 예시: 사내 배포 시스템 MCP 서버

```bash
mkdir my-deploy-mcp
cd my-deploy-mcp
npm init -y
npm install @modelcontextprotocol/sdk zod node-fetch
```

```javascript
// index.js
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({
  name: "deploy-system",
  version: "1.0.0",
});

const DEPLOY_API = process.env.DEPLOY_API_URL ?? "https://deploy.internal/api";
const DEPLOY_TOKEN = process.env.DEPLOY_TOKEN;

// 도구 1: 배포 상태 조회
server.tool(
  "get_deploy_status",
  "특정 서비스의 현재 배포 상태를 조회한다",
  {
    service: z.string().describe("서비스 이름 (예: api-gateway, auth-service)"),
    environment: z.enum(["dev", "staging", "prod"]),
  },
  async ({ service, environment }) => {
    const res = await fetch(
      `${DEPLOY_API}/status?service=${service}&env=${environment}`,
      { headers: { Authorization: `Bearer ${DEPLOY_TOKEN}` } }
    );
    const data = await res.json();
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(data, null, 2),
        },
      ],
    };
  }
);

// 도구 2: 배포 트리거
server.tool(
  "trigger_deploy",
  "서비스를 특정 환경에 배포한다",
  {
    service: z.string(),
    environment: z.enum(["dev", "staging"]), // prod는 의도적으로 제외
    version: z.string().describe("배포할 버전 태그 (예: v1.2.3)"),
    reason: z.string().describe("배포 이유 (감사 로그용)"),
  },
  async ({ service, environment, version, reason }) => {
    const res = await fetch(`${DEPLOY_API}/deploy`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${DEPLOY_TOKEN}`,
      },
      body: JSON.stringify({ service, environment, version, reason }),
    });
    const data = await res.json();
    return {
      content: [
        {
          type: "text",
          text: `배포 시작: ${data.deploy_id}\n상태: ${data.status}\n예상 완료: ${data.estimated_duration}초`,
        },
      ],
    };
  }
);

// 리소스: 배포 이력 (읽기 전용 데이터)
server.resource(
  "deploy-history",
  "deploy://history/{service}",
  async (uri) => {
    const service = uri.pathname.replace("/history/", "");
    const res = await fetch(`${DEPLOY_API}/history/${service}`, {
      headers: { Authorization: `Bearer ${DEPLOY_TOKEN}` },
    });
    const history = await res.json();
    return {
      contents: [
        {
          uri: uri.href,
          mimeType: "application/json",
          text: JSON.stringify(history, null, 2),
        },
      ],
    };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
```

### `.mcp.json`에 등록

```json
{
  "mcpServers": {
    "deploy-system": {
      "type": "stdio",
      "command": "node",
      "args": ["/path/to/my-deploy-mcp/index.js"],
      "env": {
        "DEPLOY_API_URL": "https://deploy.internal/api",
        "DEPLOY_TOKEN": "${DEPLOY_TOKEN}"
      }
    }
  }
}
```

### Claude Code에서 사용

```
사용자: auth-service 스테이징 배포 상태 확인하고,
        문제 없으면 v2.1.0 배포해줘.

Claude: 1. get_deploy_status 호출
           service: "auth-service", environment: "staging"
        → 현재 버전: v2.0.8, 상태: healthy, 마지막 배포: 3일 전

        2. 상태가 정상이므로 trigger_deploy 호출
           service: "auth-service"
           environment: "staging"
           version: "v2.1.0"
           reason: "사용자 요청에 의한 배포 - 로그인 버그 수정 포함"
        → 배포 시작: deploy-20260328-001
           예상 완료: 180초
```

### MCP 서버 디버깅

```bash
# Inspector로 서버 동작 확인
npx @modelcontextprotocol/inspector node index.js

# 수동으로 JSON-RPC 메시지 테스트
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | node index.js

# 특정 도구 호출 테스트
echo '{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "get_deploy_status",
    "arguments": {
      "service": "auth-service",
      "environment": "staging"
    }
  }
}' | node index.js
```

---

## 정리

| 도구 | 패키지 | 주요 용도 |
|------|--------|-----------|
| GitHub | `@modelcontextprotocol/server-github` | PR, 이슈, 코드 탐색 |
| PostgreSQL | `@modelcontextprotocol/server-postgres` | DB 쿼리, 스키마 탐색 |
| Playwright | `@executeautomation/playwright-mcp-server` | 브라우저 자동화, 스크린샷 |
| 커스텀 | 직접 작성 | 사내 API, 독자적 도구 |

MCP의 핵심 원칙: **AI가 도구를 호출하고, 결과를 보고, 다음 행동을 결정한다.** 도구가 많을수록 AI가 자율적으로 완수할 수 있는 작업의 범위가 넓어진다.
