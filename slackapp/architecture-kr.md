# Slack App 아키텍처 및 구성 요소

## 전체 아키텍처

```
┌──────────────────────────────────────────────────┐
│                   Slack Platform                  │
│                                                   │
│  ┌─────────┐  ┌──────────┐  ┌─────────────────┐  │
│  │ Events  │  │ Slash    │  │ Interactive     │  │
│  │ API     │  │ Commands │  │ Components      │  │
│  └────┬────┘  └────┬─────┘  └───────┬─────────┘  │
│       │            │                │             │
└───────┼────────────┼────────────────┼─────────────┘
        │            │                │
        ▼            ▼                ▼
┌──────────────────────────────────────────────────┐
│              App Server (Your Code)               │
│                                                   │
│  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
│  │ Event    │  │ Command  │  │ Interaction    │  │
│  │ Handler  │  │ Handler  │  │ Handler        │  │
│  └────┬─────┘  └────┬─────┘  └───────┬────────┘  │
│       │              │                │           │
│       ▼              ▼                ▼           │
│  ┌──────────────────────────────────────────┐    │
│  │           Business Logic Layer            │    │
│  └──────────────────┬───────────────────────┘    │
│                     │                             │
│  ┌──────────────────▼───────────────────────┐    │
│  │         Slack Web API Client              │    │
│  │  (chat.postMessage, views.open, etc.)     │    │
│  └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
```

## 핵심 구성 요소

### 1. Slack API 서버 (Slack 측)

Slack이 운영하는 서버. App Server와 통신하는 두 가지 방식을 제공한다.

- **Request URL (HTTP)**: Slack이 이벤트를 App Server의 URL로 POST 한다.
- **WebSocket (Socket Mode)**: App이 Slack에 WebSocket 연결을 맺는다.

### 2. App Server (개발자 측)

개발자가 운영하는 서버. 이벤트를 수신하고 처리한다.

#### 수신하는 요청의 종류

| 요청 유형 | Endpoint 예시 | 설명 |
|-----------|---------------|------|
| Event Subscription | `/slack/events` | 메시지, 채널 변경 등 이벤트 |
| Slash Command | `/slack/events` | `/deploy`, `/status` 등 커맨드 |
| Interactive Payload | `/slack/events` | 버튼 클릭, 모달 제출 등 |
| Options Load | `/slack/events` | 동적 셀렉트 메뉴의 옵션 로드 |
| Shortcut | `/slack/events` | 글로벌/메시지 단축키 |

> Bolt 프레임워크를 사용하면 모든 요청이 단일 endpoint(`/slack/events`)로 수신된다.

### 3. Slack Web API

App Server에서 Slack으로 요청을 보내는 REST API이다.

#### 주요 API 메서드

| 메서드 | 설명 |
|--------|------|
| `chat.postMessage` | 메시지 전송 |
| `chat.update` | 메시지 수정 |
| `chat.postEphemeral` | 특정 사용자에게만 보이는 임시 메시지 전송 |
| `views.open` | 모달 열기 |
| `views.update` | 모달 업데이트 |
| `views.push` | 모달 스택에 새 뷰 추가 |
| `conversations.list` | 채널 목록 조회 |
| `users.info` | 사용자 정보 조회 |
| `files.upload` | 파일 업로드 |
| `reactions.add` | 리액션 추가 |

### 4. App Manifest

Slack App의 설정을 선언적으로 정의하는 YAML/JSON 파일이다. 코드로 App 설정을
관리할 수 있다.

```yaml
display_information:
  name: My App
  description: 배포 자동화 봇
  background_color: "#2c2d30"

features:
  bot_user:
    display_name: deploy-bot
    always_online: true
  slash_commands:
    - command: /deploy
      url: https://my-app.example.com/slack/events
      description: 서비스 배포
      usage_hint: "[service] [version]"

oauth_config:
  scopes:
    bot:
      - chat:write
      - commands
      - channels:read

settings:
  event_subscriptions:
    request_url: https://my-app.example.com/slack/events
    bot_events:
      - message.channels
      - app_mention
  interactivity:
    is_enabled: true
    request_url: https://my-app.example.com/slack/events
  socket_mode_enabled: false
```

## 요청/응답 흐름

### Slash Command 흐름

```
사용자: /deploy my-service v1.2.3
        │
        ▼
Slack ──POST──▶ App Server
  payload: {
    command: "/deploy",
    text: "my-service v1.2.3",
    trigger_id: "...",
    user_id: "U123",
    channel_id: "C456"
  }
        │
        ▼
App Server: 3초 이내에 HTTP 200 응답 (필수)
        │
        ▼
App Server ──chat.postMessage──▶ Slack API
  (비동기로 결과 메시지 전송)
```

### Interactive Component 흐름

```
사용자: [Rollback 버튼 클릭]
        │
        ▼
Slack ──POST──▶ App Server
  payload: {
    type: "block_actions",
    actions: [{action_id: "rollback_action"}],
    trigger_id: "...",
    user: {id: "U123"},
    channel: {id: "C456"}
  }
        │
        ▼
App Server: views.open (확인 모달 표시)
        │
        ▼
사용자: [모달에서 확인 클릭]
        │
        ▼
Slack ──POST──▶ App Server
  payload: {
    type: "view_submission",
    view: {callback_id: "rollback_confirm", ...}
  }
        │
        ▼
App Server: 롤백 실행 후 chat.postMessage로 결과 전송
```

## Rate Limiting

Slack API는 요청 빈도를 제한한다.

| Tier | 초당 요청 수 | 대표 메서드 |
|------|-------------|------------|
| Tier 1 | 1 | `admin.*`, `migration.*` |
| Tier 2 | 20 | `channels.list`, `users.list` |
| Tier 3 | 50 | `reactions.add`, `pins.add` |
| Tier 4 | 100 | `chat.postMessage` |
| Special | 다양 | `files.upload`, `views.open` |

Rate limit에 걸리면 `429 Too Many Requests` 응답과 함께 `Retry-After` 헤더가
반환된다. 이 값을 활용하여 재시도 로직을 구현해야 한다.
