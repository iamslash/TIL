# Slack App 개요 및 기본 개념

## Slack App 이란?

Slack App은 Slack 워크스페이스에 기능을 추가하는 애플리케이션이다. 메시지 전송,
채널 관리, 외부 서비스 연동 등 다양한 작업을 자동화하거나 확장할 수 있다.

## Slack App vs Bot vs Integration

| 구분 | 설명 |
|------|------|
| **Slack App** | Slack Platform 위에서 동작하는 애플리케이션의 총칭. Bot, Slash Command, Workflow 등을 포함한다. |
| **Bot** | Slack App의 한 유형. 사용자처럼 메시지를 보내고 받을 수 있는 자동화된 사용자이다. `bot` scope와 Bot Token을 사용한다. |
| **Integration** | 외부 서비스(GitHub, Jira 등)를 Slack과 연결하는 것. Slack App으로 구현된다. |
| **Workflow** | Slack의 Workflow Builder로 만든 자동화 흐름. 코드 없이도 구성 가능하다. |

## 토큰의 종류

Slack App은 다양한 토큰을 사용한다.

### Bot Token (`xoxb-`)

- Bot 사용자로서 API를 호출할 때 사용한다.
- App이 설치된 워크스페이스에서 동작한다.
- 가장 일반적으로 사용되는 토큰이다.

### User Token (`xoxp-`)

- 특정 사용자를 대신하여 API를 호출할 때 사용한다.
- 사용자의 권한 범위 내에서 동작한다.
- 예: 사용자의 프로필 변경, 사용자 대신 메시지 전송.

### App-Level Token (`xapp-`)

- WebSocket 연결(Socket Mode)에 사용한다.
- 워크스페이스가 아닌 App 레벨의 토큰이다.
- `connections:write` scope가 필요하다.

### Configuration Token (`xoxe-`)

- App의 manifest를 관리할 때 사용한다.
- 드물게 사용된다.

## OAuth Scopes

Slack App이 수행할 수 있는 작업의 범위를 정의한다. 최소 권한 원칙을 따라 필요한
scope만 요청해야 한다.

### 주요 Bot Scopes

| Scope | 설명 |
|-------|------|
| `chat:write` | 메시지 전송 |
| `channels:read` | 공개 채널 정보 조회 |
| `channels:history` | 공개 채널 메시지 이력 조회 |
| `groups:read` | 비공개 채널 정보 조회 |
| `im:read` | DM 정보 조회 |
| `users:read` | 사용자 정보 조회 |
| `commands` | Slash Command 등록 |
| `reactions:read` | 리액션 조회 |
| `files:read` | 파일 조회 |

### Scope 요청 방법

App 설정 페이지 > **OAuth & Permissions** > **Scopes** 에서 추가한다. 새로운
scope를 추가하면 App을 재설치해야 한다.

## Slack App의 동작 방식

```
사용자 액션 (메시지, 버튼 클릭, 슬래시 커맨드)
        │
        ▼
   Slack Platform
        │
        ▼
  App Server (이벤트 수신)
        │
        ▼
  비즈니스 로직 처리
        │
        ▼
  Slack API 호출 (응답)
```

1. **사용자가 Slack에서 액션을 수행한다** (메시지 전송, 버튼 클릭 등).
2. **Slack이 App Server로 이벤트를 전달한다** (HTTP POST 또는 WebSocket).
3. **App Server가 이벤트를 처리한다** (비즈니스 로직).
4. **App Server가 Slack API를 호출하여 응답한다** (메시지 전송, 모달 표시 등).

## Socket Mode vs HTTP Mode

### HTTP Mode

- Slack이 App Server의 공개 URL로 HTTP POST 요청을 보낸다.
- 공인 IP와 HTTPS가 필요하다.
- 프로덕션 환경에서 일반적으로 사용한다.

### Socket Mode

- App이 Slack에 WebSocket 연결을 맺는다.
- 공인 IP가 필요 없다. 방화벽 뒤에서도 동작한다.
- 개발 환경이나 사내 서버에 적합하다.
- App-Level Token (`xapp-`)이 필요하다.

## Block Kit

Slack 메시지의 UI를 구성하는 프레임워크이다. JSON 기반으로 레이아웃을 정의한다.

### Block 유형

| Block | 설명 |
|-------|------|
| `section` | 텍스트와 accessory(버튼, 이미지 등)를 포함하는 기본 블록 |
| `actions` | 버튼, 셀렉트 메뉴 등 인터랙티브 요소의 컨테이너 |
| `divider` | 구분선 |
| `header` | 제목 텍스트 |
| `image` | 이미지 |
| `context` | 보조 텍스트/이미지 (작은 글씨) |
| `input` | 모달에서 사용하는 입력 필드 |
| `rich_text` | 서식이 적용된 텍스트 |

### Block Kit 예시

```json
{
  "blocks": [
    {
      "type": "header",
      "text": {
        "type": "plain_text",
        "text": "배포 알림"
      }
    },
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "*서비스:* my-service\n*버전:* v1.2.3\n*환경:* production"
      }
    },
    {
      "type": "actions",
      "elements": [
        {
          "type": "button",
          "text": {
            "type": "plain_text",
            "text": "Rollback"
          },
          "style": "danger",
          "action_id": "rollback_action"
        }
      ]
    }
  ]
}
```

[Block Kit Builder](https://app.slack.com/block-kit-builder)에서 시각적으로 블록을
구성하고 미리보기할 수 있다.
