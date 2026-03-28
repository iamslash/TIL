# Codex CLI 실행 흐름 다이어그램

> **근거**: `runtime-observed` (최신 session JSONL의 `pwd` 입력, `function_call`, `function_call_output`, `agent_message`, `token_count`), `source-inspected` (`codex --help`, `~/.codex/sessions` 경로 구조), `inferred` (모델 호출의 내부 네트워크 경계는 세션 로그를 기반으로 재구성)
>
> **관측 시점**: 2026-03-26, `codex-cli 0.116.0`.
>
> 아래 다이어그램은 실제 HTTP 패킷 캡처가 아니라, session JSONL에 남은 이벤트 순서를 기반으로 재구성한 것이다.

`pwd` 요청이 Codex 내부에서 어떻게 흘러가는지 가장 작은 사례로 정리한다.

## 1. 전체 시퀀스 다이어그램

```text
User              Codex CLI         Session JSONL         Model Loop         Shell
 |                    |                   |                   |                |
 | "pwd"              |                   |                   |                |
 |------------------->|                   |                   |                |
 |                    | write user item   |                   |                |
 |                    |------------------>| response_item     |                |
 |                    |                   | role=user         |                |
 |                    |                   | text="pwd"        |                |
 |                    |                   |                   |                |
 |                    | write event       |                   |                |
 |                    |------------------>| event_msg         |                |
 |                    |                   | type=user_message |                |
 |                    |                   |                   |                |
 |                    |                   |                   | reasoning       |
 |                    |                   |                   | tool needed     |
 |                    |                   |                   | exec_command    |
 |                    |                   |                   |                |
 |                    | commentary        |                   |                |
 |<-------------------|                   |                   |                |
 | "현재 작업 디렉터리를  |                   |                   |                |
 |  바로 확인..."      |                   |                   |                |
 |                    | write agent msg   |                   |                |
 |                    |------------------>| event_msg         |                |
 |                    |                   | type=agent_message|                |
 |                    |                   |                   |                |
 |                    | write call item   |                   |                |
 |                    |------------------>| response_item     |                |
 |                    |                   | function_call     |                |
 |                    |                   | name=exec_command |                |
 |                    |                   | args: cmd=pwd     |                |
 |                    |                   |                   |                |
 |                    | run shell ------------------------------------------->|
 |                    |                                                 pwd    |
 |                    |<------------------------------------------------------|
 |                    |                   |                   |                |
 |                    | write call output |                   |                |
 |                    |------------------>| response_item     |                |
 |                    |                   | function_call_out |                |
 |                    |                   | /Users/.../github |                |
 |                    |                   |                   |                |
 |                    | final message     |                   |                |
 |<-------------------|                   |                   |                |
 | "/Users/.../github"|                   |                   |                |
```

## 2. session JSONL에 실제로 남는 순서

관측된 핵심 줄은 아래와 같다. (`runtime-observed`)

```text
1. response_item(message, role=user, input_text="pwd")
2. event_msg(user_message, message="pwd")
3. event_msg(agent_message, phase=commentary)
4. response_item(function_call, name="exec_command", arguments={"cmd":"pwd", ...})
5. response_item(function_call_output, output="/Users/david.sun/prj/github")
6. response_item(message, role=assistant)  // 최종 응답
7. event_msg(token_count, cached_input_tokens=...)
```

즉, Codex는 "화면에 보여준 문장"과 "모델/도구 내부 아이템"을 둘 다 로그에 남긴다.

## 3. `pwd` 예시 상세

실제 관찰된 핵심 필드:

```text
user input_text
  "pwd"

function_call
  name = "exec_command"
  arguments = {"cmd":"pwd","workdir":"/Users/david.sun/prj/github",...}

function_call_output
  Command: /bin/zsh -lc pwd
  Output:
  /Users/david.sun/prj/github
```

이건 Codex가 단순 문자열 답변을 추측하지 않고, 현재 환경에서는 실제 명령 실행을 통해 확인한 뒤 응답한다는 뜻이다.

## 4. 시작 이벤트

세션이 시작될 때는 `task_started` 이벤트가 먼저 남는다.

```jsonc
{
  "type": "event_msg",
  "payload": {
    "type": "task_started",
    "model_context_window": 760000,
    "collaboration_mode_kind": "default"
  }
}
```

따라서 흐름은 사실상 아래처럼 시작한다.

```text
session_meta
→ task_started
→ developer/user messages
→ tool loop
→ final answer
→ token_count
```

## 5. 누적 구조

Codex 세션은 turn마다 "이전 아이템 + 새 아이템"이 쌓이는 방식으로 이해하는 것이 자연스럽다. (`inferred`)

```text
Turn 1
  session_meta
  task_started
  user: "pwd"

Turn 2
  이전 전체
  agent_message(commentary)
  function_call(exec_command)

Turn 3
  이전 전체
  function_call_output("/Users/david.sun/prj/github")
  assistant final message
```

실제 OpenAI 서버로 어떤 형태로 직렬화되는지는 로컬에서 직접 보이지 않지만, session JSONL 수준에서는 아이템 누적 구조가 분명하다.

## 6. `--ephemeral`가 아닌 기본 동작

`codex exec --help`에는 `--ephemeral` 옵션이 따로 존재한다. 즉, 기본값은 세션 파일을 디스크에 남기는 쪽이다. (`source-inspected`)

```text
--ephemeral
  Run without persisting session files to disk
```

따라서 지금 본 `~/.codex/sessions/...jsonl` 파일은 기본 persistence 동작의 결과라고 볼 수 있다.

## 7. 셸 스냅샷과의 연결

같은 session id를 prefix로 가지는 shell snapshot 파일이 별도로 존재한다. (`runtime-observed`)

```text
session:
  ~/.codex/sessions/2026/03/26/rollout-2026-03-26T14-48-59-019d28b0-....jsonl

shell snapshot:
  ~/.codex/shell_snapshots/019d28b0-....1774504140360075000.sh
```

즉, Codex는 대화 로그와 셸 재현 정보를 분리 저장한다.

## 8. 요약

```text
사용자 입력
→ session JSONL 기록
→ 모델이 tool call 결정
→ 셸 실행
→ function_call_output 기록
→ 최종 assistant 메시지
→ token_count 갱신
```

이 루프가 Codex의 가장 작은 실행 단위이다.
