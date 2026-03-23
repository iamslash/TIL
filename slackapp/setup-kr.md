# Slack App 생성 및 설정

## App 생성

### 1. Slack API 사이트에서 생성

1. [api.slack.com/apps](https://api.slack.com/apps)에 접속한다.
2. **Create New App** 클릭.
3. 두 가지 방법 중 선택:
   - **From scratch**: 수동으로 설정.
   - **From an app manifest**: YAML/JSON manifest로 한번에 설정.

### 2. From Scratch

1. **App Name** 입력 (예: `deploy-bot`).
2. **Workspace** 선택 (개발용 워크스페이스 권장).
3. **Create App** 클릭.

### 3. From an App Manifest

```yaml
display_information:
  name: deploy-bot
features:
  bot_user:
    display_name: deploy-bot
    always_online: true
oauth_config:
  scopes:
    bot:
      - chat:write
      - commands
settings:
  socket_mode_enabled: true
```

manifest를 붙여넣고 생성하면 모든 설정이 한번에 적용된다.

## 필수 설정

### Bot User 설정

**App Home** 페이지에서 Bot User를 활성화한다.

- **Display Name**: Slack에서 보이는 이름.
- **Default Username**: 봇의 username.
- **Always Show My Bot as Online**: 항상 온라인으로 표시할지 여부.

### OAuth Scopes 설정

**OAuth & Permissions** 페이지에서 필요한 scope를 추가한다.

```
Bot Token Scopes:
  - chat:write          (메시지 전송)
  - commands            (Slash Command)
  - app_mentions:read   (멘션 이벤트 수신)
```

### Event Subscriptions 설정

**Event Subscriptions** 페이지에서 수신할 이벤트를 선택한다.

1. **Enable Events** 토글을 켠다.
2. **Request URL** 입력 (HTTP Mode인 경우).
3. **Subscribe to bot events** 에서 이벤트 선택:
   - `app_mention` — 봇이 멘션되었을 때
   - `message.channels` — 공개 채널 메시지
   - `message.im` — DM 메시지

### Interactivity 설정

버튼, 모달 등 인터랙티브 기능을 사용하려면:

1. **Interactivity & Shortcuts** 페이지로 이동.
2. **Interactivity** 토글을 켠다.
3. **Request URL** 입력.

## App 설치

### 워크스페이스에 설치

1. **Install App** 페이지로 이동.
2. **Install to Workspace** 클릭.
3. OAuth 권한을 승인한다.
4. **Bot User OAuth Token** (`xoxb-...`)이 발급된다.

### 토큰 저장

발급된 토큰은 환경변수로 관리한다. 코드에 직접 넣지 않는다.

```bash
export SLACK_BOT_TOKEN=xoxb-your-token
export SLACK_SIGNING_SECRET=your-signing-secret
export SLACK_APP_TOKEN=xapp-your-app-token  # Socket Mode 사용 시
```

## Socket Mode 설정

방화벽 뒤에서 개발하거나 공인 URL이 없을 때 사용한다.

1. **Basic Information** > **App-Level Tokens** > **Generate Token and Scopes**.
2. Token Name 입력, `connections:write` scope 추가.
3. **Generate** 클릭. `xapp-` 토큰이 발급된다.
4. **Socket Mode** 페이지에서 **Enable Socket Mode** 토글을 켠다.

## 개발 환경 설정 (로컬)

### ngrok을 이용한 로컬 개발 (HTTP Mode)

```bash
# 로컬 서버 실행 (3000 포트)
npm start

# 다른 터미널에서 ngrok 실행
ngrok http 3000
```

ngrok이 제공하는 HTTPS URL을 Slack App의 Request URL로 설정한다.

```
https://abc123.ngrok-free.app/slack/events
```

> ngrok 무료 플랜은 URL이 재시작마다 바뀌므로 주의한다.

### Socket Mode를 이용한 로컬 개발

Socket Mode를 사용하면 ngrok이 필요 없다. App-Level Token만 있으면 된다.

```bash
export SLACK_APP_TOKEN=xapp-your-app-token
npm start
```

## 다중 워크스페이스 배포 (Distribution)

여러 워크스페이스에 App을 배포하려면 OAuth 흐름을 구현해야 한다.

1. **Manage Distribution** 페이지에서 체크리스트를 완료한다.
2. **Activate Public Distribution** 을 활성화한다.
3. OAuth Install URL을 사용자에게 제공한다.
4. 설치 시 발급되는 토큰을 워크스페이스별로 저장/관리한다.

```
OAuth Install URL:
https://slack.com/oauth/v2/authorize?client_id=YOUR_CLIENT_ID&scope=chat:write,commands&user_scope=
```
