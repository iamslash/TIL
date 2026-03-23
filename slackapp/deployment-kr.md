# 배포 및 운영

## 배포 옵션

### 1. 클라우드 서버 (VM / Container)

AWS EC2, GCP Compute Engine, Azure VM 등에 직접 배포한다.

```bash
# Docker로 빌드
docker build -t my-slack-app .
docker run -d \
  -e SLACK_BOT_TOKEN=xoxb-... \
  -e SLACK_SIGNING_SECRET=... \
  -p 3000:3000 \
  my-slack-app
```

```dockerfile
# Dockerfile (Node.js)
FROM node:20-slim
WORKDIR /app
COPY package*.json ./
RUN npm ci --production
COPY . .
EXPOSE 3000
CMD ["node", "app.js"]
```

### 2. Serverless (AWS Lambda)

Bolt는 AWS Lambda용 어댑터를 제공한다.

```bash
npm install @slack/bolt aws-lambda-rig
```

```javascript
const { App, AwsLambdaReceiver } = require("@slack/bolt");

const awsLambdaReceiver = new AwsLambdaReceiver({
  signingSecret: process.env.SLACK_SIGNING_SECRET,
});

const app = new App({
  token: process.env.SLACK_BOT_TOKEN,
  receiver: awsLambdaReceiver,
});

app.message("hello", async ({ say }) => {
  await say("Hello!");
});

module.exports.handler = async (event, context, callback) => {
  const handler = await awsLambdaReceiver.start();
  return handler(event, context, callback);
};
```

### 3. Google Cloud Functions

```javascript
const { App, ExpressReceiver } = require("@slack/bolt");

const receiver = new ExpressReceiver({
  signingSecret: process.env.SLACK_SIGNING_SECRET,
  processBeforeResponse: true, // Cloud Functions에서 필요
});

const app = new App({
  token: process.env.SLACK_BOT_TOKEN,
  receiver,
});

app.command("/hello", async ({ ack }) => {
  await ack("Hello from Cloud Functions!");
});

module.exports.slack = receiver.app;
```

### 4. Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: slack-bot
spec:
  replicas: 2
  selector:
    matchLabels:
      app: slack-bot
  template:
    metadata:
      labels:
        app: slack-bot
    spec:
      containers:
        - name: slack-bot
          image: my-registry/slack-bot:latest
          ports:
            - containerPort: 3000
          env:
            - name: SLACK_BOT_TOKEN
              valueFrom:
                secretKeyRef:
                  name: slack-secrets
                  key: bot-token
            - name: SLACK_SIGNING_SECRET
              valueFrom:
                secretKeyRef:
                  name: slack-secrets
                  key: signing-secret
          livenessProbe:
            httpGet:
              path: /health
              port: 3000
            periodSeconds: 30
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "256Mi"
              cpu: "200m"
```

### 5. Socket Mode (공인 URL 불필요)

사내 네트워크에서 운영할 때 적합하다.

```javascript
const { App } = require("@slack/bolt");

const app = new App({
  token: process.env.SLACK_BOT_TOKEN,
  socketMode: true,
  appToken: process.env.SLACK_APP_TOKEN,
});

(async () => {
  await app.start();
  console.log("Socket Mode app running");
})();
```

## 환경별 구성

### 개발 / 스테이징 / 프로덕션 분리

환경별로 별도의 Slack App을 생성하는 것을 권장한다.

```
deploy-bot-dev      → 개발 워크스페이스
deploy-bot-staging  → 스테이징 워크스페이스
deploy-bot          → 프로덕션 워크스페이스
```

### 환경변수 관리

```bash
# .env.development
SLACK_BOT_TOKEN=xoxb-dev-token
SLACK_SIGNING_SECRET=dev-secret
LOG_LEVEL=debug

# .env.production
SLACK_BOT_TOKEN=xoxb-prod-token
SLACK_SIGNING_SECRET=prod-secret
LOG_LEVEL=info
```

## 모니터링

### Health Check 엔드포인트

```javascript
const { App, ExpressReceiver } = require("@slack/bolt");

const receiver = new ExpressReceiver({
  signingSecret: process.env.SLACK_SIGNING_SECRET,
});

// Health check
receiver.router.get("/health", (req, res) => {
  res.status(200).json({ status: "ok" });
});

const app = new App({
  token: process.env.SLACK_BOT_TOKEN,
  receiver,
});
```

### 로깅

```javascript
const { App, LogLevel } = require("@slack/bolt");

const app = new App({
  token: process.env.SLACK_BOT_TOKEN,
  signingSecret: process.env.SLACK_SIGNING_SECRET,
  logLevel: LogLevel.INFO,
});
```

### 에러 알림

```javascript
app.error(async (error) => {
  console.error(`Unhandled error: ${error.message}`);

  // 에러를 별도 채널에 알림
  await app.client.chat.postMessage({
    token: process.env.SLACK_BOT_TOKEN,
    channel: "#bot-alerts",
    text: `:warning: Bot Error: ${error.message}`,
  });
});
```

## 고가용성 고려 사항

### HTTP Mode

- 여러 인스턴스를 로드밸런서 뒤에 배치할 수 있다.
- 이벤트는 하나의 인스턴스에만 전달되므로 상태를 외부 저장소(Redis, DB)에
  저장해야 한다.

### Socket Mode

- WebSocket 연결은 단일 인스턴스가 유지한다.
- 최대 10개의 동시 WebSocket 연결을 지원한다.
- 페일오버를 위해 여러 인스턴스가 연결을 맺어둘 수 있다.
