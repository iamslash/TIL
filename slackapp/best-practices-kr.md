# 보안 및 모범 사례

## 보안

### 요청 검증 (Signing Secret)

Slack은 모든 요청에 서명을 포함한다. App Server는 이 서명을 검증하여 요청이
Slack에서 온 것인지 확인해야 한다.

```
X-Slack-Signature: v0=a2114d57b48eac39b9ad189dd8316235a7b4a8d21a10bd27519666489c69b503
X-Slack-Request-Timestamp: 1531420618
```

검증 방법:

```
base_string = "v0:{timestamp}:{request_body}"
signature = "v0=" + HMAC-SHA256(signing_secret, base_string)
```

> Bolt 프레임워크를 사용하면 자동으로 검증된다.

### 토큰 관리

- **환경변수**로 관리한다. 코드에 하드코딩하지 않는다.
- **Secret Manager** 사용을 권장한다 (AWS Secrets Manager, GCP Secret Manager 등).
- **토큰 로테이션**: 토큰이 유출되면 즉시 재발급한다
  (App 설정 > OAuth & Permissions > Regenerate).
- `.env` 파일은 `.gitignore`에 추가한다.

```gitignore
# .gitignore
.env
.env.*
```

### 최소 권한 원칙

필요한 scope만 요청한다. 예를 들어 메시지를 보내기만 한다면 `chat:write`만
필요하다. `admin.*` scope는 정말 필요한 경우에만 요청한다.

### 입력 검증

사용자 입력을 신뢰하지 않는다.

```javascript
app.command("/deploy", async ({ command, ack, respond }) => {
  await ack();

  const text = command.text.trim();

  // 허용된 서비스 목록 검증
  const allowedServices = ["web", "api", "worker"];
  const [service, version] = text.split(/\s+/);

  if (!allowedServices.includes(service)) {
    await respond(`허용되지 않는 서비스: ${service}`);
    return;
  }

  // 버전 형식 검증
  if (!/^v\d+\.\d+\.\d+$/.test(version)) {
    await respond("버전 형식이 올바르지 않습니다. 예: v1.2.3");
    return;
  }

  // 안전한 처리
});
```

## 사용자 경험

### 응답 시간

- 3초 이내에 `ack()`를 호출한다.
- 시간이 걸리는 작업은 먼저 진행 상태를 알리고 나중에 결과를 전송한다.

```javascript
app.command("/report", async ({ command, ack, respond }) => {
  await ack("리포트를 생성 중입니다... :hourglass_flowing_sand:");

  const report = await generateReport();

  await respond({
    replace_original: true,
    text: `리포트가 준비되었습니다: ${report.url}`,
  });
});
```

### 에러 메시지

사용자에게 친절한 에러 메시지를 제공한다.

```javascript
app.command("/deploy", async ({ command, ack, respond }) => {
  await ack();

  try {
    await deploy(command.text);
    await respond(":white_check_mark: 배포가 완료되었습니다.");
  } catch (error) {
    await respond(
      `:x: 배포에 실패했습니다.\n원인: ${error.message}\n문의: #deploy-support`
    );
  }
});
```

### Ephemeral vs 공개 메시지

- **에러 메시지**: ephemeral (실행한 사용자에게만 표시).
- **성공 결과**: in_channel (팀원에게도 공유가 필요한 경우).
- **중간 상태**: ephemeral (불필요한 노이즈 방지).

### 메시지 업데이트

진행 상태를 보여줄 때 새 메시지를 계속 보내지 말고 기존 메시지를 업데이트한다.

```javascript
// 초기 메시지
const result = await client.chat.postMessage({
  channel: channelId,
  text: ":hourglass: 배포 시작...",
});

// 진행 상태 업데이트
await client.chat.update({
  channel: channelId,
  ts: result.ts,
  text: ":arrows_counterclockwise: 빌드 중...",
});

// 완료 업데이트
await client.chat.update({
  channel: channelId,
  ts: result.ts,
  text: ":white_check_mark: 배포 완료!",
});
```

### 스레드 활용

관련 메시지는 스레드로 묶어 채널 노이즈를 줄인다.

```javascript
await client.chat.postMessage({
  channel: channelId,
  thread_ts: originalMessageTs,
  text: "배포 로그입니다.",
});
```

## 코드 구조

### 핸들러 분리

하나의 파일에 모든 핸들러를 넣지 않는다.

```
listeners/
  commands/
    deploy.js
    status.js
  events/
    app-mention.js
    message.js
  actions/
    approve.js
  views/
    deploy-modal.js
```

### 비즈니스 로직 분리

Slack 핸들러와 비즈니스 로직을 분리한다. 이렇게 하면 테스트가 쉬워진다.

```javascript
// services/deploy-service.js
async function deploy(service, version) {
  // 순수한 비즈니스 로직 (Slack 의존성 없음)
}

// listeners/commands/deploy.js
const { deploy } = require("../../services/deploy-service");

module.exports = function (app) {
  app.command("/deploy", async ({ command, ack, respond }) => {
    await ack();
    const result = await deploy(service, version);
    await respond(`결과: ${result}`);
  });
};
```

## Rate Limit 대응

```javascript
const { WebClient } = require("@slack/web-api");

const client = new WebClient(process.env.SLACK_BOT_TOKEN, {
  retryConfig: {
    retries: 3,
    factor: 2, // exponential backoff
  },
});
```

## 테스트

### 단위 테스트

비즈니스 로직을 분리했다면 Slack 없이 테스트할 수 있다.

```javascript
const { deploy } = require("../services/deploy-service");

test("deploy returns success", async () => {
  const result = await deploy("my-service", "v1.2.3");
  expect(result.status).toBe("success");
});
```

### Bolt 테스트 유틸리티

Bolt는 테스트 헬퍼를 제공하지 않으므로, `ack`, `say`, `respond` 등을 mock하여
테스트한다.

```javascript
test("deploy command", async () => {
  const ack = jest.fn();
  const respond = jest.fn();

  await handleDeploy({
    command: { text: "my-service v1.2.3" },
    ack,
    respond,
  });

  expect(ack).toHaveBeenCalled();
  expect(respond).toHaveBeenCalledWith(
    expect.stringContaining("배포")
  );
});
```
