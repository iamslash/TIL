# System Prompt & Instructions (세션 간 지속 기억)

## 개요

SQLite에 저장되는 세션/메시지는 **세션 내 기억**이다. 세션을 새로 만들면
이전 대화는 포함되지 않는다. 하지만 **프로젝트 지시사항(Instructions)**은
세션과 무관하게 매 LLM 호출마다 system prompt에 주입되어 **세션을 넘어
지속되는 기억** 역할을 한다.

## System Prompt 구성

매 LLM 호출 시 system prompt는 다음 요소들이 조합된다:

```
┌─────────────────────────────────────────────────────┐
│                  System Prompt                       │
│                                                      │
│  ① Provider Prompt (모델별 기본 프롬프트)             │
│     - anthropic.txt (Claude)                        │
│     - beast.txt (GPT-4o, o1, o3)                    │
│     - codex.txt (기타 GPT 모델)                      │
│     - gemini.txt (Gemini)                           │
│     - default.txt (기타)                            │
│                                                      │
│  ② Environment (환경 정보)                           │
│     - 모델 이름, 작업 디렉토리, git 여부              │
│     - 플랫폼, 오늘 날짜                              │
│                                                      │
│  ③ Skills (사용 가능한 스킬 목록)                     │
│                                                      │
│  ④ Instructions (프로젝트/글로벌 지시사항) ← 핵심     │
│     - AGENTS.md / CLAUDE.md 파일 내용                │
│     - config에서 지정한 추가 instruction 파일          │
│     - URL로 지정한 원격 instruction                   │
└─────────────────────────────────────────────────────┘
```

```typescript
// packages/opencode/src/session/prompt.ts — prompt() 내부
// 참고: Provider Prompt는 이보다 앞서 LLM.stream()에서 별도로 주입됨
// 아래는 추가 system prompt를 조합하는 부분
const system = [
  ...(await SystemPrompt.environment(model)),   // ② Environment
  ...(skills ? [skills] : []),                  // ③ Skills
  ...(await InstructionPrompt.system()),        // ④ Instructions
]
```

Provider Prompt(①)는 `LLM.stream()` 내부에서 별도로 주입된다:

```typescript
// packages/opencode/src/session/llm.ts — LLM.stream() 내부
const system: string[] = []
system.push([
  ...(input.agent.prompt ? [input.agent.prompt] : SystemPrompt.provider(input.model)),
  ...input.system,          // ← 위에서 조합한 ②③④가 여기로 전달
  ...(input.user.system ? [input.user.system] : []),
].filter(x => x).join("\n"))
```

## Instructions 상세

### 검색 파일

OpenCode는 다음 파일을 자동으로 찾아 system prompt에 포함한다:

```typescript
// packages/opencode/src/session/instruction.ts
const FILES = [
  "AGENTS.md",
  "CLAUDE.md",
  "CONTEXT.md",   // deprecated
]
```

### 검색 순서

```
① 프로젝트 로컬 (작업 디렉토리 → 워크트리 루트까지 상위 탐색)
   ./AGENTS.md → ../AGENTS.md → ../../AGENTS.md → ... → {worktree}/AGENTS.md

② 글로벌
   ~/.config/opencode/AGENTS.md
   또는
   ~/.claude/CLAUDE.md  (Claude Code 호환)

③ Config에서 지정한 추가 파일
   config.instructions: ["./docs/rules.md", "~/my-rules.md", "https://example.com/rules.md"]
```

### 실제 주입 형식

```
Instructions from: /Users/me/my-app/AGENTS.md
# My Project Rules

- TypeScript strict mode 사용
- 모든 함수에 JSDoc 작성
- 테스트 커버리지 80% 이상 유지

Instructions from: /Users/me/.config/opencode/AGENTS.md
# Global Rules

- 한국어로 답변
- 코드 변경 전 반드시 확인 질문
```

### 디렉토리별 Instructions (계층적)

파일을 읽을 때 해당 디렉토리에 AGENTS.md가 있으면 **추가로** 로드한다.
이미 system prompt에 포함된 파일이나 이미 읽은 파일은 중복 로드하지 않는다.

```
my-app/
├── AGENTS.md                 ← system prompt에 항상 포함
├── src/
│   ├── AGENTS.md             ← src/ 내 파일을 Read할 때 추가 로드
│   └── routes/
│       ├── AGENTS.md         ← src/routes/ 내 파일을 Read할 때 추가 로드
│       └── api.ts
```

```typescript
// packages/opencode/src/session/instruction.ts — InstructionPrompt.resolve()
export async function resolve(messages, filepath, messageID) {
  // filepath의 디렉토리부터 워크트리 루트까지 올라가며
  // AGENTS.md를 찾아 아직 로드하지 않은 것만 반환
  let current = path.dirname(target)
  while (current.startsWith(root) && current !== root) {
    const found = await find(current)
    if (found && !system.has(found) && !already.has(found) && !isClaimed(messageID, found)) {
      claim(messageID, found)
      results.push({ filepath: found, content: "Instructions from: " + found + "\n" + content })
    }
    current = path.dirname(current)
  }
}
```

## Provider별 System Prompt

모델에 따라 다른 기본 프롬프트가 사용된다:

```typescript
// packages/opencode/src/session/system.ts
export function provider(model: Provider.Model) {
  if (model.api.id.includes("gpt-4") || model.api.id.includes("o1") || model.api.id.includes("o3"))
    return [PROMPT_BEAST]           // OpenAI 고성능 모델
  if (model.api.id.includes("gpt")) return [PROMPT_CODEX]
  if (model.api.id.includes("gemini-")) return [PROMPT_GEMINI]
  if (model.api.id.includes("claude")) return [PROMPT_ANTHROPIC]
  return [PROMPT_DEFAULT]
}
```

## ConfigMarkdown 파싱

AGENTS.md 파일은 YAML frontmatter를 가질 수 있다. 또한 특수 문법을 지원한다:

```typescript
// packages/opencode/src/config/markdown.ts
export const FILE_REGEX = /(?<![\w`])@(\.?[^\s`,.]*(?:\.[^\s`,.]+)*)/g   // @파일경로
export const SHELL_REGEX = /!`([^`]+)`/g                                  // !`셸 명령`
```

| 문법 | 예시 | 설명 |
|------|------|------|
| `@파일경로` | `@src/schema.ts` | 해당 파일 내용을 컨텍스트에 포함 |
| `` !`명령` `` | `` !`git branch` `` | 셸 명령 실행 결과를 컨텍스트에 포함 |

### 예시: AGENTS.md

```markdown
---
model: openai/gpt-4o
---

# My Project

이 프로젝트는 Express.js + TypeScript API 서버입니다.

## 규칙
- @tsconfig.json 의 설정을 준수하라
- 현재 브랜치: !`git branch --show-current`
- 모든 변경 후 `npm test` 실행
```

## Long-term Memory 관점에서의 의미

```
세션 내 기억 (SQLite):
  → 이 세션에서 어떤 대화를 했는가
  → session/message/part 테이블

세션 간 기억 (Instructions):
  → 이 프로젝트에서 항상 지켜야 할 규칙은 무엇인가
  → AGENTS.md, CLAUDE.md 파일
  → 세션이 바뀌어도 항상 system prompt에 포함
  → 사용자가 파일을 수정하면 다음 호출부터 즉시 반영

글로벌 기억 (Global Instructions):
  → 모든 프로젝트에서 공통으로 적용할 규칙
  → ~/.config/opencode/AGENTS.md
  → ~/.claude/CLAUDE.md
```
