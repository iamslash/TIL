# OpenCode 개요

## OpenCode 란?

OpenCode는 **오픈소스 AI 코딩 에이전트**이다. Claude Code와 유사한 터미널
기반 AI 코딩 도우미로, 다양한 LLM 프로바이더(Anthropic, OpenAI, Google,
AWS Bedrock 등)를 지원한다.

## 설치

```bash
# 스크립트
curl -fsSL https://opencode.ai/install | bash

# 패키지 매니저
npm i -g opencode-ai@latest
brew install anomalyco/tap/opencode    # macOS/Linux
```

## 초기 설정

설치 후 LLM 프로바이더의 API 키를 설정해야 한다.

```bash
# OpenAI 사용 시
export OPENAI_API_KEY=sk-...

# Anthropic 사용 시
export ANTHROPIC_API_KEY=sk-ant-...

# 또는 opencode.json 설정 파일에서 지정
```

프로젝트 디렉토리에서 `opencode`를 실행하면 바로 대화를 시작할 수 있다.

```bash
cd my-project
opencode
```

## 핵심 특징

| 특징 | 설명 |
|------|------|
| 오픈소스 | MIT 라이선스. 코드가 모두 공개되어 있다 |
| 멀티 프로바이더 | Anthropic, OpenAI, Google, AWS Bedrock, Azure 등 지원 |
| TUI (Terminal UI) | 터미널에서 동작하는 인터랙티브 UI |
| Desktop App | Electron 기반 데스크톱 앱도 제공 (BETA) |
| TypeScript | Bun 런타임 기반 TypeScript로 작성됨 |
| SQLite 저장소 | 세션, 메시지, 프로젝트 등을 로컬 SQLite에 저장 |

## 프로젝트 구조

```
opencode/
├── packages/
│   ├── opencode/        # 핵심 패키지 (CLI, 에이전트, 스토리지)
│   │   ├── src/
│   │   │   ├── session/     # 세션/메시지/파트 관리
│   │   │   ├── storage/     # SQLite DB + JSON 파일 스토리지
│   │   │   ├── project/     # 프로젝트 관리
│   │   │   ├── provider/    # LLM 프로바이더 연동
│   │   │   ├── cli/         # CLI 명령어 및 TUI
│   │   │   └── ...
│   │   └── migration/       # DB 마이그레이션 SQL
│   ├── console/             # 웹 콘솔
│   ├── enterprise/          # 엔터프라이즈 기능
│   ├── ui/                  # 공통 UI 컴포넌트
│   └── web/                 # 웹사이트
├── sdks/                    # SDK
└── specs/                   # 스펙 문서
```

## 기술 스택

| 구분 | 기술 |
|------|------|
| 런타임 | Bun (Node.js 호환) |
| 언어 | TypeScript |
| DB | SQLite (drizzle-orm) |
| TUI | Ink (React for CLI) |
| 빌드 | Turbo (monorepo) |
| 패키지 매니저 | Bun |
