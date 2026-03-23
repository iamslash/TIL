# Interactive Components (모달, 버튼 등)

## 개요

Interactive Components는 사용자가 클릭, 선택, 입력 등으로 App과 상호작용할 수
있는 UI 요소이다. 버튼, 셀렉트 메뉴, 날짜 선택기, 모달 등이 포함된다.

## 버튼 (Button)

### 메시지에 버튼 추가

```javascript
app.command("/approve", async ({ command, ack, client }) => {
  await ack();

  await client.chat.postMessage({
    channel: command.channel_id,
    text: "승인 요청",
    blocks: [
      {
        type: "section",
        text: {
          type: "mrkdwn",
          text: `*승인 요청*\n<@${command.user_id}>님이 배포 승인을 요청했습니다.`,
        },
      },
      {
        type: "actions",
        elements: [
          {
            type: "button",
            text: { type: "plain_text", text: "승인" },
            style: "primary",
            action_id: "approve_action",
            value: "deploy_123",
          },
          {
            type: "button",
            text: { type: "plain_text", text: "거절" },
            style: "danger",
            action_id: "reject_action",
            value: "deploy_123",
          },
        ],
      },
    ],
  });
});
```

### 버튼 클릭 핸들링

```javascript
app.action("approve_action", async ({ body, ack, say, client }) => {
  await ack();

  // 원본 메시지 업데이트 (버튼 제거)
  await client.chat.update({
    channel: body.channel.id,
    ts: body.message.ts,
    text: `<@${body.user.id}>님이 승인했습니다.`,
    blocks: [
      {
        type: "section",
        text: {
          type: "mrkdwn",
          text: `~승인 요청~ → *승인됨* by <@${body.user.id}>`,
        },
      },
    ],
  });
});
```

## 셀렉트 메뉴 (Select Menu)

### 정적 셀렉트

```json
{
  "type": "actions",
  "elements": [
    {
      "type": "static_select",
      "action_id": "env_select",
      "placeholder": { "type": "plain_text", "text": "환경 선택" },
      "options": [
        {
          "text": { "type": "plain_text", "text": "Development" },
          "value": "dev"
        },
        {
          "text": { "type": "plain_text", "text": "Staging" },
          "value": "staging"
        },
        {
          "text": { "type": "plain_text", "text": "Production" },
          "value": "prod"
        }
      ]
    }
  ]
}
```

### 동적 셀렉트 (External Data Source)

```javascript
// 옵션 로드 핸들러
app.options("service_select", async ({ ack, options }) => {
  const keyword = options.value; // 사용자가 입력한 검색어

  const services = await fetchServices(keyword);
  const opts = services.map((s) => ({
    text: { type: "plain_text", text: s.name },
    value: s.id,
  }));

  await ack({ options: opts });
});
```

### 사용자 셀렉트 / 채널 셀렉트

```json
{
  "type": "users_select",
  "action_id": "assignee_select",
  "placeholder": { "type": "plain_text", "text": "담당자 선택" }
}
```

```json
{
  "type": "channels_select",
  "action_id": "channel_select",
  "placeholder": { "type": "plain_text", "text": "채널 선택" }
}
```

## 모달 (Modal)

모달은 팝업 형태의 입력 폼이다. `trigger_id`가 필요하며 3초 이내에 열어야 한다.

### 모달 열기

```javascript
app.command("/ticket", async ({ command, ack, client }) => {
  await ack();

  await client.views.open({
    trigger_id: command.trigger_id,
    view: {
      type: "modal",
      callback_id: "ticket_submit",
      title: { type: "plain_text", text: "티켓 생성" },
      submit: { type: "plain_text", text: "생성" },
      close: { type: "plain_text", text: "취소" },
      blocks: [
        {
          type: "input",
          block_id: "title_block",
          label: { type: "plain_text", text: "제목" },
          element: {
            type: "plain_text_input",
            action_id: "title_input",
            placeholder: { type: "plain_text", text: "티켓 제목을 입력하세요" },
          },
        },
        {
          type: "input",
          block_id: "priority_block",
          label: { type: "plain_text", text: "우선순위" },
          element: {
            type: "static_select",
            action_id: "priority_input",
            options: [
              {
                text: { type: "plain_text", text: "긴급" },
                value: "urgent",
              },
              {
                text: { type: "plain_text", text: "높음" },
                value: "high",
              },
              {
                text: { type: "plain_text", text: "보통" },
                value: "medium",
              },
              {
                text: { type: "plain_text", text: "낮음" },
                value: "low",
              },
            ],
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
          optional: true,
        },
      ],
    },
  });
});
```

### 모달 제출 핸들링

```javascript
app.view("ticket_submit", async ({ ack, body, view, client }) => {
  const values = view.state.values;
  const title = values.title_block.title_input.value;
  const priority = values.priority_block.priority_input.selected_option.value;
  const desc = values.desc_block.desc_input.value || "";

  // 유효성 검사 실패 시 에러 반환
  if (title.length < 5) {
    await ack({
      response_action: "errors",
      errors: {
        title_block: "제목은 5자 이상이어야 합니다.",
      },
    });
    return;
  }

  await ack();

  // 티켓 생성 후 알림
  await client.chat.postMessage({
    channel: body.user.id,
    text: `티켓이 생성되었습니다: *${title}* (${priority})`,
  });
});
```

### 모달 업데이트 / 스택 push

```javascript
// 현재 모달을 업데이트
await client.views.update({
  view_id: body.view.id,
  view: {
    /* 새로운 view 정의 */
  },
});

// 모달 스택에 새 뷰를 push (뒤로가기 가능)
await client.views.push({
  trigger_id: body.trigger_id,
  view: {
    /* 새로운 view 정의 */
  },
});
```

## Overflow Menu

항목이 많을 때 `...` 메뉴로 표시한다.

```json
{
  "type": "overflow",
  "action_id": "overflow_menu",
  "options": [
    {
      "text": { "type": "plain_text", "text": "편집" },
      "value": "edit"
    },
    {
      "text": { "type": "plain_text", "text": "삭제" },
      "value": "delete"
    },
    {
      "text": { "type": "plain_text", "text": "로그 보기" },
      "value": "view_logs"
    }
  ]
}
```

## Date Picker / Time Picker

```json
{
  "type": "input",
  "block_id": "date_block",
  "label": { "type": "plain_text", "text": "마감일" },
  "element": {
    "type": "datepicker",
    "action_id": "deadline_input",
    "initial_date": "2026-04-01",
    "placeholder": { "type": "plain_text", "text": "날짜 선택" }
  }
}
```

## Confirm Dialog

버튼 클릭 시 확인 대화상자를 표시한다.

```json
{
  "type": "button",
  "text": { "type": "plain_text", "text": "삭제" },
  "style": "danger",
  "action_id": "delete_action",
  "confirm": {
    "title": { "type": "plain_text", "text": "정말 삭제하시겠습니까?" },
    "text": { "type": "mrkdwn", "text": "이 작업은 되돌릴 수 없습니다." },
    "confirm": { "type": "plain_text", "text": "삭제" },
    "deny": { "type": "plain_text", "text": "취소" }
  }
}
```
