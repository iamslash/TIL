# Bolt 프레임워크

## Bolt 란?

Bolt는 Slack이 공식 제공하는 Slack App 개발 프레임워크이다. JavaScript, Python,
Java 버전이 있다. 이벤트 라우팅, 인증, 미들웨어 등을 내장하고 있어 보일러플레이트
코드를 크게 줄여준다.

## 설치

### JavaScript (Node.js)

```bash
npm init -y
npm install @slack/bolt
```

### Python

```bash
pip install slack-bolt
```

## 기본 구조

### JavaScript

```javascript
const { App } = require("@slack/bolt");

const app = new App({
  token: process.env.SLACK_BOT_TOKEN,
  signingSecret: process.env.SLACK_SIGNING_SECRET,
  // Socket Mode 사용 시
  socketMode: true,
  appToken: process.env.SLACK_APP_TOKEN,
});

// 메시지 리스너
app.message("hello", async ({ message, say }) => {
  await say(`안녕하세요, <@${message.user}>!`);
});

// Slash Command 리스너
app.command("/deploy", async ({ command, ack, respond }) => {
  await ack();
  await respond(`배포 요청: ${command.text}`);
});

// 버튼 액션 리스너
app.action("approve_button", async ({ body, ack, say }) => {
  await ack();
  await say(`<@${body.user.id}>님이 승인했습니다.`);
});

// 모달 제출 리스너
app.view("submit_modal", async ({ ack, body, view }) => {
  await ack();
  const values = view.state.values;
  // 값 처리
});

(async () => {
  await app.start(process.env.PORT || 3000);
  console.log("Bolt app is running!");
})();
```

### Python

```python
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

app = App(token=os.environ["SLACK_BOT_TOKEN"])

@app.message("hello")
def handle_hello(message, say):
    say(f"안녕하세요, <@{message['user']}>!")

@app.command("/deploy")
def handle_deploy(ack, command, respond):
    ack()
    respond(f"배포 요청: {command['text']}")

@app.action("approve_button")
def handle_approve(ack, body, say):
    ack()
    say(f"<@{body['user']['id']}>님이 승인했습니다.")

@app.view("submit_modal")
def handle_submission(ack, body, view):
    ack()
    values = view["state"]["values"]
    # 값 처리

if __name__ == "__main__":
    # Socket Mode
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()

    # HTTP Mode
    # app.start(port=3000)
```

## 미들웨어

Bolt는 요청 처리 파이프라인에 미들웨어를 삽입할 수 있다.

### 글로벌 미들웨어

모든 요청에 적용된다.

```javascript
// 모든 요청을 로깅
app.use(async ({ next, body }) => {
  console.log(`Event type: ${body.type}`);
  await next();
});
```

### 리스너 미들웨어

특정 리스너에만 적용된다.

```javascript
// 특정 채널에서만 동작하는 미들웨어
async function onlyInGeneral({ message, next }) {
  if (message.channel === "C0123456") {
    await next();
  }
}

app.message("deploy", onlyInGeneral, async ({ say }) => {
  await say("배포를 시작합니다.");
});
```

## 에러 핸들링

```javascript
app.error(async (error) => {
  console.error(`Error: ${error.message}`);
  // 에러 알림 전송, 로깅 등
});
```

## `say` vs `respond` vs `client`

| 함수 | 설명 | 사용 시점 |
|------|------|----------|
| `say` | 이벤트가 발생한 채널에 메시지를 전송한다 | 메시지 이벤트, 액션 핸들러 |
| `respond` | `response_url`을 통해 응답한다. 임시 메시지 가능 | Slash Command, Interactive |
| `client` | Slack Web API를 직접 호출한다 | 어디서든 사용 가능 |

### `client` 사용 예시

```javascript
app.command("/notify", async ({ command, ack, client }) => {
  await ack();

  // 다른 채널에 메시지 전송
  await client.chat.postMessage({
    channel: "C0OTHER",
    text: `<@${command.user_id}>님의 알림: ${command.text}`,
  });

  // 모달 열기
  await client.views.open({
    trigger_id: command.trigger_id,
    view: {
      type: "modal",
      callback_id: "notify_modal",
      title: { type: "plain_text", text: "알림 설정" },
      blocks: [
        /* ... */
      ],
    },
  });
});
```

## 프로젝트 구조 (권장)

```
my-slack-app/
├── app.js              # App 초기화 및 시작
├── listeners/
│   ├── commands/       # Slash Command 핸들러
│   │   └── deploy.js
│   ├── events/         # Event 핸들러
│   │   └── app-mention.js
│   ├── actions/        # Interactive Action 핸들러
│   │   └── approve.js
│   └── views/          # Modal View 핸들러
│       └── submit.js
├── services/           # 비즈니스 로직
│   └── deploy-service.js
├── utils/
│   └── blocks.js       # Block Kit 헬퍼
├── package.json
└── .env
```

```javascript
// listeners/commands/deploy.js
module.exports = function registerDeployCommand(app) {
  app.command("/deploy", async ({ command, ack, respond }) => {
    await ack();
    // ...
  });
};

// app.js
const registerDeployCommand = require("./listeners/commands/deploy");
registerDeployCommand(app);
```
