# Slash Commands

## Slash Command 란?

`/` 로 시작하는 명령어를 Slack 채팅창에 입력하면 App Server가 처리하는 기능이다.
예: `/deploy my-service v1.2.3`, `/status`, `/help`.

## 등록 방법

### Slack App 설정에서 등록

1. **Slash Commands** 페이지로 이동.
2. **Create New Command** 클릭.
3. 다음 항목을 입력:

| 항목 | 설명 | 예시 |
|------|------|------|
| Command | 명령어 이름 | `/deploy` |
| Request URL | 요청을 받을 URL | `https://example.com/slack/events` |
| Short Description | 명령어 설명 | `서비스를 배포합니다` |
| Usage Hint | 사용법 힌트 | `[service] [version]` |
| Escape channels, users, and links | 자동 변환 여부 | 체크 권장 |

### App Manifest로 등록

```yaml
features:
  slash_commands:
    - command: /deploy
      url: https://example.com/slack/events
      description: 서비스를 배포합니다
      usage_hint: "[service] [version]"
      should_escape: true
    - command: /status
      url: https://example.com/slack/events
      description: 서비스 상태를 확인합니다
      usage_hint: "[service]"
```

## 수신 Payload

Slack이 App Server로 보내는 데이터:

```json
{
  "command": "/deploy",
  "text": "my-service v1.2.3",
  "trigger_id": "1234567890.123456",
  "user_id": "U0123ABC",
  "user_name": "david",
  "team_id": "T0123ABC",
  "team_domain": "myteam",
  "channel_id": "C0123ABC",
  "channel_name": "general",
  "response_url": "https://hooks.slack.com/commands/T0123/..."
}
```

### 주요 필드

| 필드 | 설명 |
|------|------|
| `command` | 실행된 명령어 |
| `text` | 명령어 뒤에 입력된 텍스트 |
| `trigger_id` | 모달을 열 때 필요한 ID (3초 이내 사용) |
| `user_id` | 명령어를 실행한 사용자 ID |
| `channel_id` | 명령어를 실행한 채널 ID |
| `response_url` | 지연 응답을 보낼 URL (30분 유효, 최대 5회) |

## 응답 방법

### 1. 즉시 응답 (3초 이내)

```javascript
app.command("/hello", async ({ ack }) => {
  await ack("안녕하세요!");
});
```

`ack()`에 텍스트를 전달하면 해당 사용자에게만 보이는 임시 메시지로 응답된다.

### 2. 채널에 보이는 즉시 응답

```javascript
app.command("/announce", async ({ ack }) => {
  await ack({
    response_type: "in_channel",
    text: "공지사항입니다!",
  });
});
```

| `response_type` | 설명 |
|-----------------|------|
| `ephemeral` (기본값) | 명령어를 실행한 사용자에게만 보인다 |
| `in_channel` | 채널의 모든 사용자에게 보인다 |

### 3. 지연 응답 (3초 초과)

시간이 오래 걸리는 작업은 먼저 `ack()`로 응답한 후 `respond()`로 결과를 전송한다.

```javascript
app.command("/deploy", async ({ command, ack, respond }) => {
  // 3초 이내에 반드시 ack
  await ack("배포를 시작합니다...");

  // 시간이 걸리는 작업
  const result = await deployService(command.text);

  // response_url로 결과 전송
  await respond({
    response_type: "in_channel",
    text: `배포 완료: ${result}`,
  });
});
```

### 4. 모달로 응답

```javascript
app.command("/create-ticket", async ({ command, ack, client }) => {
  await ack();

  await client.views.open({
    trigger_id: command.trigger_id,
    view: {
      type: "modal",
      callback_id: "ticket_modal",
      title: { type: "plain_text", text: "티켓 생성" },
      submit: { type: "plain_text", text: "생성" },
      blocks: [
        {
          type: "input",
          block_id: "title_block",
          label: { type: "plain_text", text: "제목" },
          element: {
            type: "plain_text_input",
            action_id: "title_input",
          },
        },
        {
          type: "input",
          block_id: "desc_block",
          label: { type: "plain_text", text: "설명" },
          element: {
            type: "plain_text_input",
            action_id: "desc_input",
            multiline: true,
          },
        },
      ],
    },
  });
});
```

## 텍스트 파싱 패턴

```javascript
app.command("/deploy", async ({ command, ack, respond }) => {
  await ack();

  const args = command.text.trim().split(/\s+/);

  if (args.length < 2) {
    await respond("사용법: /deploy [service] [version]");
    return;
  }

  const [service, version] = args;
  await respond(`배포: ${service} ${version}`);
});
```

## 주의 사항

- **3초 규칙**: `ack()`를 3초 이내에 호출하지 않으면 Slack이 "이 명령어가 동작하지
  않았습니다" 에러를 표시한다.
- **trigger_id 유효 기간**: 모달을 열려면 3초 이내에 `views.open`을 호출해야 한다.
- **response_url 제한**: 30분 동안 유효하며 최대 5번 사용할 수 있다.
- **명령어 충돌**: 다른 App과 같은 명령어를 사용하면 사용자가 선택해야 한다.
  고유한 명령어 이름을 사용하는 것이 좋다.
