# Events API

## 개요

Events API는 Slack에서 발생하는 이벤트(메시지, 리액션, 채널 변경 등)를 App
Server로 실시간으로 전달하는 구독 기반 API이다.

## 이벤트 전달 방식

### HTTP Mode

Slack이 설정된 Request URL로 HTTP POST 요청을 보낸다.

```
POST https://your-app.example.com/slack/events
Content-Type: application/json
```

### Socket Mode

App이 Slack에 WebSocket 연결을 맺어 이벤트를 수신한다. 공인 URL이 필요 없다.

## URL Verification

HTTP Mode에서 Request URL을 설정하면 Slack이 검증 요청을 보낸다. 이 요청에
`challenge` 값을 그대로 반환해야 한다.

```json
// Slack이 보내는 요청
{
  "type": "url_verification",
  "token": "Jhj5dZrVaK7ZwHHjRyZWjbDl",
  "challenge": "3eZbrw1aBm2rZgRNFdxV2595E9CY3gmdALWMmHkvFXO7tYXAYM8P"
}

// App이 반환해야 하는 응답
{
  "challenge": "3eZbrw1aBm2rZgRNFdxV2595E9CY3gmdALWMmHkvFXO7tYXAYM8P"
}
```

> Bolt 프레임워크를 사용하면 자동으로 처리된다.

## 주요 이벤트 목록

### 메시지 관련

| 이벤트 | 설명 | 필요 Scope |
|--------|------|-----------|
| `message.channels` | 공개 채널 메시지 | `channels:history` |
| `message.groups` | 비공개 채널 메시지 | `groups:history` |
| `message.im` | DM 메시지 | `im:history` |
| `message.mpim` | 그룹 DM 메시지 | `mpim:history` |

### 앱 관련

| 이벤트 | 설명 | 필요 Scope |
|--------|------|-----------|
| `app_mention` | 봇이 멘션됨 | `app_mentions:read` |
| `app_home_opened` | App Home 탭 열림 | - |
| `app_uninstalled` | App 제거됨 | - |

### 채널 관련

| 이벤트 | 설명 |
|--------|------|
| `channel_created` | 채널 생성 |
| `channel_deleted` | 채널 삭제 |
| `channel_rename` | 채널 이름 변경 |
| `member_joined_channel` | 멤버가 채널에 참여 |
| `member_left_channel` | 멤버가 채널을 떠남 |

### 리액션 관련

| 이벤트 | 설명 | 필요 Scope |
|--------|------|-----------|
| `reaction_added` | 리액션 추가 | `reactions:read` |
| `reaction_removed` | 리액션 제거 | `reactions:read` |

## 이벤트 Payload 구조

```json
{
  "token": "XXYYZZ",
  "team_id": "T0123ABC",
  "api_app_id": "A0123ABC",
  "event": {
    "type": "message",
    "subtype": null,
    "text": "안녕하세요",
    "user": "U0123ABC",
    "channel": "C0123ABC",
    "ts": "1234567890.123456",
    "event_ts": "1234567890.123456",
    "channel_type": "channel"
  },
  "type": "event_callback",
  "event_id": "Ev0123ABC",
  "event_time": 1234567890
}
```

## Bolt에서 이벤트 처리

### 메시지 이벤트

```javascript
// 특정 텍스트가 포함된 메시지
app.message("hello", async ({ message, say }) => {
  await say(`안녕하세요, <@${message.user}>!`);
});

// 정규식 패턴 매칭
app.message(/배포\s+(.+)\s+(.+)/, async ({ context, say }) => {
  const service = context.matches[1];
  const version = context.matches[2];
  await say(`${service} ${version} 배포를 시작합니다.`);
});

// 모든 메시지 (패턴 없이)
app.message(async ({ message }) => {
  console.log(`Message: ${message.text}`);
});
```

### 앱 멘션

```javascript
app.event("app_mention", async ({ event, say }) => {
  await say(`무엇을 도와드릴까요, <@${event.user}>?`);
});
```

### App Home 열기

```javascript
app.event("app_home_opened", async ({ event, client }) => {
  await client.views.publish({
    user_id: event.user,
    view: {
      type: "home",
      blocks: [
        {
          type: "header",
          text: { type: "plain_text", text: "Deploy Bot 대시보드" },
        },
        {
          type: "section",
          text: {
            type: "mrkdwn",
            text: "최근 배포 현황을 확인하세요.",
          },
        },
        {
          type: "actions",
          elements: [
            {
              type: "button",
              text: { type: "plain_text", text: "새 배포" },
              action_id: "new_deploy",
            },
          ],
        },
      ],
    },
  });
});
```

### 리액션 이벤트

```javascript
app.event("reaction_added", async ({ event, client }) => {
  // 특정 이모지에만 반응
  if (event.reaction === "ticket") {
    await client.chat.postMessage({
      channel: event.item.channel,
      thread_ts: event.item.ts,
      text: `<@${event.user}>님이 티켓을 생성했습니다.`,
    });
  }
});
```

## 메시지 서브타입

메시지 이벤트에는 다양한 서브타입이 있다.

| subtype | 설명 |
|---------|------|
| `null` | 일반 사용자 메시지 |
| `bot_message` | 봇이 보낸 메시지 |
| `message_changed` | 메시지 편집 |
| `message_deleted` | 메시지 삭제 |
| `file_share` | 파일 공유 |
| `thread_broadcast` | 스레드 메시지가 채널에도 공유됨 |
| `channel_join` | 채널 입장 알림 |

봇 메시지에 다시 반응하면 무한 루프가 발생할 수 있으므로 주의한다.

```javascript
app.message(async ({ message, say }) => {
  // 봇 메시지는 무시
  if (message.subtype === "bot_message") return;

  await say("응답합니다.");
});
```

## 이벤트 재전송 (Retry)

App Server가 3초 이내에 HTTP 200을 반환하지 않으면 Slack이 이벤트를 재전송한다.
`X-Slack-Retry-Num`, `X-Slack-Retry-Reason` 헤더로 재전송 여부를 확인할 수 있다.

재전송에 의한 중복 처리를 방지하려면 `event_id`로 중복 검사를 수행한다.

```javascript
const processedEvents = new Set();

app.use(async ({ body, next }) => {
  if (body.event_id && processedEvents.has(body.event_id)) {
    return; // 중복 이벤트 무시
  }
  processedEvents.add(body.event_id);
  await next();
});
```
