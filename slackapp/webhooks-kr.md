# Webhooks

## 개요

Webhook은 외부 시스템에서 Slack으로 메시지를 보내거나, Slack에서 외부 시스템으로
데이터를 전달하는 간단한 방법이다.

## Incoming Webhooks

외부 시스템 → Slack으로 메시지를 보내는 단방향 통신이다.

### 설정 방법

1. Slack App 설정 > **Incoming Webhooks** > 활성화.
2. **Add New Webhook to Workspace** 클릭.
3. 메시지를 보낼 채널을 선택한다.
4. Webhook URL이 발급된다.

```
https://hooks.slack.com/services/TXXXXX/BXXXXX/XXXXXXXXXX
```

### 메시지 전송

```bash
# 간단한 텍스트
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"배포가 완료되었습니다."}' \
  https://hooks.slack.com/services/TXXXXX/BXXXXX/XXXXXXXXXX

# Block Kit 사용
curl -X POST -H 'Content-type: application/json' \
  --data '{
    "blocks": [
      {
        "type": "section",
        "text": {
          "type": "mrkdwn",
          "text": "*배포 완료* :white_check_mark:\n서비스: `my-service`\n버전: `v1.2.3`"
        }
      }
    ]
  }' \
  https://hooks.slack.com/services/TXXXXX/BXXXXX/XXXXXXXXXX
```

### 프로그래밍 언어별 예시

#### Python

```python
import requests

webhook_url = "https://hooks.slack.com/services/TXXXXX/BXXXXX/XXXXXXXXXX"

payload = {
    "text": "배포가 완료되었습니다.",
    "blocks": [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*my-service* `v1.2.3` 배포 완료"
            }
        }
    ]
}

response = requests.post(webhook_url, json=payload)
```

#### JavaScript

```javascript
const payload = {
  text: "배포가 완료되었습니다.",
};

await fetch("https://hooks.slack.com/services/TXXXXX/BXXXXX/XXXXXXXXXX", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload),
});
```

### 제한 사항

- 특정 채널에만 메시지를 보낼 수 있다 (URL마다 채널 고정).
- 메시지 전송만 가능하다 (읽기, 수정, 삭제 불가).
- 인터랙티브 컴포넌트(버튼 등)의 응답을 받을 수 없다.
- Rate limit: 초당 1개 요청 권장.

## Workflow Webhooks

Slack의 Workflow Builder와 연동되는 Webhook이다. Incoming Webhook보다 유연하다.

### 특징

- Workflow Builder에서 Webhook trigger를 생성한다.
- 커스텀 변수를 정의하고 Workflow의 후속 단계에서 사용할 수 있다.
- 채널 선택이 Workflow 단계에서 이루어지므로 더 유연하다.

### 사용 예시

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  --data '{
    "service": "my-service",
    "version": "v1.2.3",
    "status": "success"
  }' \
  https://hooks.slack.com/triggers/TXXXXX/BXXXXX/XXXXXXXXXX
```

## response_url

Slash Command나 Interactive Component에서 제공되는 일회성 응답 URL이다.

### 특징

| 항목 | 값 |
|------|-----|
| 유효 기간 | 30분 |
| 최대 사용 횟수 | 5회 |
| 응답 타입 | `ephemeral` 또는 `in_channel` |

### 사용 예시

```javascript
app.command("/slow-task", async ({ command, ack }) => {
  await ack("처리 중...");

  // 시간이 걸리는 작업
  const result = await longRunningTask();

  // response_url로 결과 전송
  await fetch(command.response_url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      response_type: "in_channel",
      replace_original: true,
      text: `완료: ${result}`,
    }),
  });
});
```

### response_url Payload 옵션

| 옵션 | 설명 |
|------|------|
| `replace_original: true` | 원래 메시지를 교체한다 |
| `delete_original: true` | 원래 메시지를 삭제한다 |
| `response_type: "in_channel"` | 모든 사용자에게 보이게 한다 |
| `response_type: "ephemeral"` | 요청한 사용자에게만 보이게 한다 |

## Incoming Webhook vs Web API 비교

| 항목 | Incoming Webhook | Web API (`chat.postMessage`) |
|------|-----------------|------------------------------|
| 설정 | 간단 (URL만 필요) | 토큰 + 스코프 필요 |
| 채널 | URL당 1개 고정 | 자유롭게 지정 |
| 기능 | 메시지 전송만 | 전체 API 사용 가능 |
| 인터랙티브 | 표시만 가능 (응답 불가) | 완전 지원 |
| 적합한 경우 | CI/CD 알림, 모니터링 | 완전한 Slack App |

간단한 알림용도라면 Incoming Webhook으로 충분하다. 양방향 상호작용이 필요하면
Web API를 사용한다.
