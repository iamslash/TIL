# Codex CLI API 및 세션 이벤트 구조

> **근거**: `source-inspected` (OpenAI Responses API 문서, `codex --help`, 최신 session JSONL), `runtime-observed` (`codex exec --help`, `pwd` 요청이 남긴 session JSONL), `inferred` (Codex 내부 네트워크 요청의 정확한 페이로드는 직접 캡처하지 않았으므로 일부는 외부 구조로부터 추정)
>
> **관측 시점**: 2026-03-26, `codex-cli 0.116.0`.
>
> **주의**: 아래 구조 설명은 실제 패킷 캡처가 아니라 로컬 session JSONL과 공식 문서를 기반으로 재구성한 것이다.

Codex CLI는 OpenAI의 Responses 스타일 아이템 모델과 매우 유사한 구조로 세션을 기록한다. 로컬 session JSONL에는 `message`, `reasoning`, `function_call`, `function_call_output` 같은 타입이 그대로 남는다.

## 큰 그림

공식 Responses 문서는 입력과 출력의 기본 단위를 `Items`로 설명한다. 로컬 Codex session JSONL도 같은 방향을 따른다:

```text
session_meta
event_msg(task_started)
response_item(message: developer)
response_item(message: user)
response_item(reasoning)
response_item(function_call)
response_item(function_call_output)
response_item(message: assistant)
event_msg(token_count)
```

즉, Codex는 "긴 문장 하나"보다 "타입이 있는 이벤트/아이템들의 열"로 세션을 다룬다.

## 세션 파일 포맷

Codex는 기본적으로 세션을 JSONL로 저장한다. (`runtime-observed`)

```text
~/.codex/sessions/YYYY/MM/DD/rollout-<timestamp>-<uuid>.jsonl
```

각 줄은 독립된 JSON 객체이며, 크게 3종류가 보인다.

| 상위 `type` | 의미 | 예시 |
|------|------|------|
| `session_meta` | 세션 시작 메타데이터 | `cwd`, `cli_version`, `model_provider`, `base_instructions` |
| `event_msg` | 런타임 이벤트 | `task_started`, `user_message`, `agent_message`, `token_count` |
| `response_item` | 모델/도구 아이템 | `message`, `reasoning`, `function_call`, `function_call_output` |

## `session_meta`

세션 첫머리에는 고정 메타데이터가 남는다. (`runtime-observed`)

```jsonc
{
  "type": "session_meta",
  "payload": {
    "id": "019d28b0-...",
    "cwd": "/Users/david.sun/prj/github",
    "originator": "codex_cli_rs",
    "cli_version": "0.116.0",
    "model_provider": "openai",
    "base_instructions": {
      "text": "You are Codex, a coding agent based on GPT-5.4..."
    }
  }
}
```

여기서 중요한 점:
- base instructions 자체가 세션 파일에 저장된다.
- 현재 작업 디렉터리와 CLI 버전이 함께 남는다.
- 모델 제공자(`openai`)가 명시된다.

## `event_msg`

`event_msg`는 세션 진행 상태를 나타내는 메타 이벤트이다.

### 1. `task_started`

```jsonc
{
  "type": "event_msg",
  "payload": {
    "type": "task_started",
    "turn_id": "...",
    "model_context_window": 760000,
    "collaboration_mode_kind": "default"
  }
}
```

`config.toml`에는 `model_context_window = 800000`이 있지만, 실제 turn 시작 이벤트에는 `760000`이 기록되었다. (`runtime-observed`) 정확한 차감 규칙은 내부 구현에 따라 달라질 수 있다. (`inferred`)

### 2. `user_message`

```jsonc
{
  "type": "event_msg",
  "payload": {
    "type": "user_message",
    "message": "pwd"
  }
}
```

### 3. `agent_message`

Codex는 중간 진행 상황을 별도 이벤트로도 남긴다.

```jsonc
{
  "type": "event_msg",
  "payload": {
    "type": "agent_message",
    "message": "현재 작업 디렉터리를 바로 확인하겠습니다.",
    "phase": "commentary"
  }
}
```

### 4. `token_count`

세션 중간중간 누적 토큰 통계가 기록된다.

```jsonc
{
  "type": "event_msg",
  "payload": {
    "type": "token_count",
    "info": {
      "total_token_usage": {
        "input_tokens": 562526,
        "cached_input_tokens": 358144,
        "output_tokens": 9923
      }
    }
  }
}
```

이 `cached_input_tokens` 필드는 Codex가 upstream 입력 캐시를 사용하고 있음을 보여주는 강한 관측 근거이다. (`runtime-observed`)

## `response_item`

`response_item`은 모델과 도구의 실제 작업 단위이다.

| `payload.type` | 의미 | 관측 예시 |
|------|------|------|
| `message` | developer/user/assistant 메시지 | `role: "user"`, `content: [{type:"input_text", ...}]` |
| `reasoning` | reasoning 요약 또는 암호화된 추론 블록 | `encrypted_content` 필드 포함 |
| `function_call` | 도구 호출 요청 | `name: "exec_command"` |
| `function_call_output` | 도구 결과 | 명령 출력, exit code, stdout 요약 |

### `message`

```jsonc
{
  "type": "response_item",
  "payload": {
    "type": "message",
    "role": "user",
    "content": [
      {"type": "input_text", "text": "pwd"}
    ]
  }
}
```

### `function_call`

```jsonc
{
  "type": "response_item",
  "payload": {
    "type": "function_call",
    "name": "exec_command",
    "arguments": "{\"cmd\":\"pwd\",\"workdir\":\"/Users/david.sun/prj/github\"}",
    "call_id": "call_PM0klhVXe8tubs2pLvVKq5UD"
  }
}
```

### `function_call_output`

```jsonc
{
  "type": "response_item",
  "payload": {
    "type": "function_call_output",
    "call_id": "call_PM0klhVXe8tubs2pLvVKq5UD",
    "output": "Command: /bin/zsh -lc pwd\nOutput:\n/Users/david.sun/prj/github"
  }
}
```

## `pwd` 예시

`pwd` 요청은 아래 순서로 session JSONL에 남았다. (`runtime-observed`)

```text
1. user message: "pwd"
2. event_msg(user_message)
3. agent_message(commentary): "현재 작업 디렉터리를 바로 확인하겠습니다."
4. response_item(function_call): exec_command(cmd="pwd")
5. response_item(function_call_output): /Users/david.sun/prj/github
6. assistant final message: "/Users/david.sun/prj/github"
```

Codex는 단순 질문도 "설명 → 도구 호출 → 결과 주입 → 최종 응답" 패턴으로 처리한다.

## CLI 명령 표면

`codex --help` 기준 주요 진입점은 다음과 같다. (`source-inspected` + `runtime-observed`)

| 명령 | 역할 |
|------|------|
| `codex` | 대화형 CLI |
| `codex exec` | 비대화형 실행 |
| `codex review` | 코드 리뷰 전용 실행 |
| `codex resume` | 이전 세션 이어서 실행 |
| `codex fork` | 이전 세션 fork |
| `codex apply` | 최신 diff를 `git apply` 형태로 적용 |
| `codex mcp` | MCP 서버 관리 |
| `codex features` | feature flag 조회/변경 |

## 세션 파일과 Responses API의 관계

공식 Responses 문서는 `message`, `function_call`, `function_call_output`, `reasoning` 같은 아이템 타입을 설명한다. 로컬 Codex session JSONL도 거의 같은 타입명을 사용한다. (`source-inspected` + `runtime-observed`)

따라서 안전한 결론은 다음 정도이다.
- Codex의 세션 기록 모델은 Responses API 아이템 구조와 강하게 닮아 있다.
- 로컬에서 직접 보이는 것은 "세션 기록 포맷"이지, 원시 HTTP 요청/응답은 아니다.
- 네트워크 레벨의 최종 페이로드 세부 사항은 직접 캡처하지 않았으므로 일부는 추정으로 남겨야 한다.
