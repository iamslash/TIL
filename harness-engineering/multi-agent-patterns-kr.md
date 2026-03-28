# 멀티 에이전트 패턴과 선택 기준

> 커리큘럼 05-CH02 대응 문서

## 목차

- [왜 멀티 에이전트인가](#왜-멀티-에이전트인가)
- [패턴 비교표](#패턴-비교표)
- [Fan-out / Fan-in 패턴](#fan-out--fan-in-패턴)
- [Pipeline (순차 체이닝) 패턴](#pipeline-순차-체이닝-패턴)
- [Orchestrator-Worker 패턴](#orchestrator-worker-패턴)
- [Debate / Consensus 패턴](#debate--consensus-패턴)
- [Supervisor 패턴](#supervisor-패턴)
- [멀티 에이전트가 오히려 손해인 경우](#멀티-에이전트가-오히려-손해인-경우)
- [하이브리드 AI 코딩 에이전트 전략](#하이브리드-ai-코딩-에이전트-전략)

---

## 왜 멀티 에이전트인가

단일 LLM 호출의 컨텍스트 창은 유한하다. 복잡한 소프트웨어 작업은 다음 세 가지 이유로 단일 에이전트를 넘어선다.

1. **병렬 처리**: 독립적인 하위 작업을 동시에 실행하면 벽시계 시간이 줄어든다.
2. **전문화**: 탐색 에이전트, 구현 에이전트, 리뷰 에이전트는 각자의 역할에 특화된 프롬프트를 받는다.
3. **검증 루프**: 한 에이전트의 출력을 다른 에이전트가 독립적으로 검증한다.

---

## 패턴 비교표

| 패턴 | 구조 | 주 사용처 | 강점 | 약점 | 비용 배율 |
|------|------|-----------|------|------|-----------|
| Fan-out/Fan-in | 1 오케스트레이터 → N 워커 → 집계 | 대규모 병렬 분석 | 최고 속도 | 집계 복잡성 | N배 |
| Pipeline | A → B → C → D (순차) | 단계별 변환 워크플로 | 단순·예측 가능 | 속도 느림 | 직렬 합산 |
| Orchestrator-Worker | 동적 작업 할당 | 크기 미지의 작업 목록 | 유연성 | 조율 오버헤드 | 가변 |
| Debate/Consensus | 에이전트 A vs B → 중재자 | 아키텍처·설계 결정 | 품질 최대화 | 가장 비쌈 | 3N배+ |
| Supervisor | 감독자가 서브에이전트 재시도 | 고신뢰 자동화 | 내결함성 | 복잡한 상태 관리 | 가변 |

---

## Fan-out / Fan-in 패턴

### 개념도

```
         ┌── Worker A (파일 분석) ──┐
입력 ──> │── Worker B (테스트 생성) ─│──> Aggregator ──> 최종 결과
         └── Worker C (문서 작성) ──┘
```

### 언제 쓰는가

- 작업이 독립적으로 분할 가능할 때
- 각 조각이 완전한 컨텍스트를 자체적으로 가질 때
- 결과를 단순 병합할 수 있을 때

### Claude Code 구현 예시

오케스트레이터 프롬프트:

```
당신은 코드 리뷰 오케스트레이터입니다.
아래 파일 목록을 받아 각 파일을 독립 에이전트에게 할당하고
결과를 수집해 종합 리뷰 보고서를 만드세요.

파일 목록: src/auth.ts, src/db.ts, src/api.ts

각 에이전트에게 이 지시를 전달하세요:
"[파일명]을 분석하고 다음 형식으로 보고하세요:
SECURITY: [보안 문제]
PERF: [성능 문제]
STYLE: [스타일 문제]"
```

Agent tool 호출 (TypeScript SDK 기준):

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

async function fanOutReview(files: string[]) {
  // Fan-out: 병렬로 각 파일 에이전트 실행
  const workerPromises = files.map((file) =>
    client.messages.create({
      model: "claude-opus-4-5",
      max_tokens: 2048,
      system: `당신은 코드 리뷰 전문가입니다. ${file} 파일만 분석하세요.`,
      messages: [
        {
          role: "user",
          content: `다음 파일을 분석하고 SECURITY/PERF/STYLE 형식으로 보고하세요:\n\n${file}`,
        },
      ],
    })
  );

  const results = await Promise.all(workerPromises);

  // Fan-in: 집계 에이전트로 결과 종합
  const aggregated = results
    .map((r, i) => `=== ${files[i]} ===\n${r.content[0].text}`)
    .join("\n\n");

  const summary = await client.messages.create({
    model: "claude-opus-4-5",
    max_tokens: 4096,
    messages: [
      {
        role: "user",
        content: `다음 개별 리뷰들을 종합해 우선순위별 액션 아이템을 정리하세요:\n\n${aggregated}`,
      },
    ],
  });

  return summary.content[0].text;
}

fanOutReview(["src/auth.ts", "src/db.ts", "src/api.ts"]).then(console.log);
```

### 실제 oh-my-claudecode 활용

```bash
# ultrawork 모드로 Fan-out 실행
/oh-my-claudecode:ultrawork "src/ 디렉토리의 모든 TypeScript 파일을 병렬로 리뷰하고 보안 취약점 보고서 작성"
```

---

## Pipeline (순차 체이닝) 패턴

### 개념도

```
요구사항 ──> [Analyst] ──> Spec ──> [Architect] ──> Design ──> [Executor] ──> Code ──> [Verifier] ──> 완료
```

### 언제 쓰는가

- 각 단계가 이전 단계의 출력에 의존할 때
- 중간 산출물이 검토·승인 필요할 때
- 단계별 전문성이 다를 때

### Claude Code 구현 예시

oh-my-claudecode pipeline 프롬프트:

```bash
/oh-my-claudecode:pipeline "
단계 1 (analyst): 사용자 인증 기능 요구사항을 분석하고 Spec.md 작성
단계 2 (architect): Spec.md를 읽고 시스템 설계 문서 작성
단계 3 (executor): 설계 문서를 바탕으로 코드 구현
단계 4 (verifier): 구현 코드가 Spec.md 요구사항을 충족하는지 검증
"
```

직접 구현 (Python):

```python
import anthropic

client = anthropic.Anthropic()

def pipeline_run(requirement: str) -> dict:
    stages = {}

    # 단계 1: 분석
    analyst_result = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        system="당신은 소프트웨어 분석가입니다. 요구사항을 명확한 Spec으로 변환하세요.",
        messages=[{"role": "user", "content": requirement}]
    )
    stages["spec"] = analyst_result.content[0].text

    # 단계 2: 설계 (이전 단계 출력 포함)
    architect_result = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        system="당신은 소프트웨어 아키텍트입니다.",
        messages=[{
            "role": "user",
            "content": f"다음 Spec을 바탕으로 시스템 설계를 작성하세요:\n\n{stages['spec']}"
        }]
    )
    stages["design"] = architect_result.content[0].text

    # 단계 3: 구현
    executor_result = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        system="당신은 시니어 개발자입니다. 설계 문서를 보고 코드를 구현하세요.",
        messages=[{
            "role": "user",
            "content": f"설계:\n{stages['design']}\n\n위 설계를 Python으로 구현하세요."
        }]
    )
    stages["code"] = executor_result.content[0].text

    # 단계 4: 검증
    verifier_result = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system="당신은 QA 엔지니어입니다.",
        messages=[{
            "role": "user",
            "content": f"Spec:\n{stages['spec']}\n\n코드:\n{stages['code']}\n\n코드가 Spec을 충족하는지 검증하세요."
        }]
    )
    stages["verification"] = verifier_result.content[0].text

    return stages

result = pipeline_run("JWT 기반 사용자 인증 시스템 구현")
print(result["verification"])
```

---

## Orchestrator-Worker 패턴

### 개념도

```
Orchestrator
    │
    ├── Task Queue: [task1, task2, task3, ...]
    │
    ├── Worker 1 ──> task1 완료 ──> 다음 작업 요청
    ├── Worker 2 ──> task2 완료 ──> 다음 작업 요청
    └── Worker 3 ──> task3 완료 ──> 다음 작업 요청
```

### 언제 쓰는가

- 작업 목록의 크기를 사전에 알 수 없을 때
- 작업이 실행 중에 동적으로 생성될 때
- 워커 수를 탄력적으로 조정해야 할 때

### Claude Code 구현 예시 (Agent tool 사용)

오케스트레이터 시스템 프롬프트:

```
당신은 코드베이스 마이그레이션 오케스트레이터입니다.

도구 목록:
- list_migration_tasks(): 남은 마이그레이션 작업 목록 반환
- assign_task(worker_id, task_id): 워커에게 작업 할당
- check_task_status(task_id): 작업 상태 확인
- merge_results(): 완료된 작업 결과 통합

작업 흐름:
1. list_migration_tasks()로 작업 목록 조회
2. 사용 가능한 워커에게 작업 할당
3. 완료된 작업 결과 수집
4. 모든 작업 완료 시 merge_results() 호출

워커 에이전트 지시:
"할당된 파일의 CommonJS import를 ESM으로 변환하고 완료 보고하세요."
```

실제 구현 (Claude Code 네이티브 Team API):

```typescript
// oh-my-claudecode Team API 활용
import { TeamCreate, TaskCreate, SendMessage } from "@anthropic-ai/claude-code";

async function orchestratorWorkerPattern(tasks: string[]) {
  // 팀 생성
  const team = await TeamCreate({ name: "migration-team" });

  // 작업 큐 생성
  const taskIds = await Promise.all(
    tasks.map((task) =>
      TaskCreate({
        team_name: team.name,
        title: task,
        description: `파일 마이그레이션: ${task}`,
      })
    )
  );

  // 워커 에이전트들이 작업 큐에서 가져가서 실행
  // (실제로는 Task() 호출로 에이전트 스폰)
  console.log(`${taskIds.length}개 작업 생성 완료`);

  return team.name;
}
```

---

## Debate / Consensus 패턴

### 개념도

```
질문 ──> Agent A (찬성 측) ──┐
                              ├──> Mediator (중재자) ──> 최종 결정
질문 ──> Agent B (반대 측) ──┘
              ↑ 반복 (최대 N라운드)
```

### 언제 쓰는가

- 아키텍처 결정처럼 정답이 하나가 아닐 때
- 편향 없는 평가가 필요할 때
- 트레이드오프를 명시적으로 탐색해야 할 때

### Claude Code 구현 예시

```python
import anthropic

client = anthropic.Anthropic()

def debate_consensus(question: str, rounds: int = 2) -> str:
    history_a = []
    history_b = []

    for round_num in range(rounds):
        # Agent A: 마이크로서비스 옹호
        response_a = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            system="""당신은 마이크로서비스 아키텍처 전문가입니다.
항상 마이크로서비스의 장점을 강조하고 상대방 주장에 반박하세요.""",
            messages=history_a + [{"role": "user", "content": question if round_num == 0
                                   else f"상대방 주장: {history_b[-1]['content']}\n반박하세요."}]
        )
        history_a.append({"role": "user", "content": question})
        history_a.append({"role": "assistant", "content": response_a.content[0].text})

        # Agent B: 모놀리스 옹호
        response_b = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            system="""당신은 모놀리식 아키텍처 전문가입니다.
항상 모놀리스의 단순성을 강조하고 상대방 주장에 반박하세요.""",
            messages=history_b + [{"role": "user", "content": f"상대방 주장: {response_a.content[0].text}\n반박하세요."}]
        )
        history_b.append({"role": "user", "content": question})
        history_b.append({"role": "assistant", "content": response_b.content[0].text})

    # 중재자: 최종 결론
    mediator = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        system="당신은 공정한 기술 중재자입니다. 양측 주장을 듣고 상황에 맞는 결론을 도출하세요.",
        messages=[{
            "role": "user",
            "content": f"""질문: {question}

마이크로서비스 측 최종 주장:
{history_a[-1]['content']}

모놀리스 측 최종 주장:
{history_b[-1]['content']}

위 논쟁을 바탕으로 팀 규모 5명, 스타트업 초기 단계 상황에서의 최적 결론을 내리세요."""
        }]
    )

    return mediator.content[0].text

result = debate_consensus("우리 팀이 마이크로서비스와 모놀리스 중 무엇을 선택해야 하나요?")
print(result)
```

oh-my-claudecode에서의 활용:

```bash
# ralplan 명령: Planner + Architect + Critic이 합의에 이를 때까지 토론
/oh-my-claudecode:ralplan "신규 결제 시스템 아키텍처 결정: 이벤트 소싱 vs 전통적 CRUD"
```

---

## Supervisor 패턴

### 개념도

```
Supervisor
    │
    ├── Sub-agent 실행 ──> 성공? ──> 완료
    │                 └──> 실패? ──> 재시도 (전략 수정)
    │                           └──> N회 실패 ──> 에스컬레이션
```

### 언제 쓰는가

- 외부 시스템 연동처럼 실패 가능성이 있을 때
- 재시도 전략이 필요할 때
- 고신뢰 자동화 파이프라인을 구성할 때

### Claude Code 구현 예시

```python
import anthropic
import time

client = anthropic.Anthropic()

def supervisor_with_retry(task: str, max_retries: int = 3) -> str:
    """Supervisor 패턴: 실패 시 전략을 수정해 재시도"""

    previous_failures = []

    for attempt in range(max_retries):
        # 서브에이전트 실행
        failure_context = ""
        if previous_failures:
            failure_context = f"\n\n이전 시도 실패 내역:\n" + "\n".join(
                f"시도 {i+1}: {f}" for i, f in enumerate(previous_failures)
            )

        result = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2048,
            system=f"""당신은 코드 실행 에이전트입니다.
작업을 수행하고 결과를 보고하세요.
실패 시 반드시 'FAILED: [이유]' 형식으로 보고하세요.{failure_context}""",
            messages=[{"role": "user", "content": task}]
        )

        response_text = result.content[0].text

        # Supervisor가 결과 검증
        if "FAILED:" not in response_text:
            return f"성공 (시도 {attempt + 1}): {response_text}"

        # 실패 기록 및 전략 수정
        failure_reason = response_text.split("FAILED:")[1].strip()
        previous_failures.append(failure_reason)

        # Supervisor가 재시도 전략 결정
        strategy = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            system="당신은 장애 분석 전문가입니다. 실패 원인을 분석하고 다음 시도 전략을 한 문장으로 제시하세요.",
            messages=[{"role": "user", "content": f"실패 원인: {failure_reason}"}]
        )
        new_strategy = strategy.content[0].text
        task = f"{task}\n\n[감독자 지시] {new_strategy}"

        time.sleep(2 ** attempt)  # 지수 백오프

    return f"최대 재시도 {max_retries}회 초과. 에스컬레이션 필요."

result = supervisor_with_retry("프로덕션 DB 마이그레이션 스크립트 실행 및 검증")
print(result)
```

---

## 멀티 에이전트가 오히려 손해인 경우

### 판단 기준 플로우차트

```
작업이 있다
    │
    ▼
단순한 단일 질문인가?
    │ YES ──> 단일 LLM 호출 사용 (멀티 에이전트 불필요)
    │ NO
    ▼
작업 결과물이 2000 토큰 미만인가?
    │ YES ──> 단일 LLM 호출 사용
    │ NO
    ▼
하위 작업들이 독립적인가?
    │ NO ──> Pipeline 고려, 단 단계가 2개면 단일 호출 고려
    │ YES
    ▼
각 하위 작업이 30초 이상 걸리는가?
    │ NO ──> 단일 LLM 호출 또는 단순 루프 사용
    │ YES
    ▼
멀티 에이전트 패턴 적용
```

### 멀티 에이전트가 손해인 구체적 상황

| 상황 | 이유 | 대안 |
|------|------|------|
| 단어 교정 | 조율 오버헤드 > 실제 작업 | 단일 호출 |
| 간단한 코드 한 줄 수정 | 에이전트 스폰 비용 | 직접 편집 |
| 결과가 순서 의존적 | Fan-out 불가 | 단순 Pipeline |
| 컨텍스트 공유 필수 | 에이전트 간 전달 비용 높음 | 단일 긴 컨텍스트 |
| 예산이 $0.10 미만 | 에이전트 조율 비용이 주 비용 초과 | 단일 호출 |
| 실시간 응답 필요 (< 2초) | 에이전트 스폰 지연 | 캐시된 단일 호출 |

### 비용 계산 예시

```python
# 단일 에이전트 비용
single_agent_cost = 1000 * 0.000003  # 1000 토큰 @ claude-sonnet
# = $0.003

# Fan-out 5개 에이전트 비용 (각 300 토큰 + 집계 500 토큰)
fanout_cost = (5 * 300 + 500) * 0.000003
# = $0.006

# Fan-out이 유리한 조건:
# - 병렬 실행으로 시간 단축이 가치 있을 때
# - 각 에이전트 품질이 단일 에이전트보다 현저히 높을 때
# - 작업 당 토큰이 단순 합산 이상으로 줄어들 때
```

---

## 하이브리드 AI 코딩 에이전트 전략

### Claude Code vs Codex 역할 분담

| 작업 유형 | Claude Code | Codex CLI |
|-----------|-------------|-----------|
| 코드베이스 탐색 | 최적 (파일 읽기 + 심볼 추적) | 보통 |
| 아키텍처 계획 | 최적 (장문 추론) | 미흡 |
| 반복 구현 | 양호 | 최적 (빠른 편집) |
| 유닛 테스트 생성 | 양호 | 최적 (패턴 학습) |
| PR 리뷰 | 최적 (전체 컨텍스트) | 보통 |
| 리팩토링 | 최적 | 양호 |
| 보안 감사 | 최적 | 미흡 |

### 권장 하이브리드 패턴

```
Claude Code (탐색/계획/리뷰)
         │
         ├── 1. 코드베이스 탐색 → 컨텍스트 문서 생성
         │
         ├── 2. 아키텍처 결정 → Design.md 출력
         │
         ▼
Codex CLI (구현/테스트) ← Design.md 입력
         │
         ├── 3. 코드 구현 (빠른 반복)
         │
         ├── 4. 테스트 생성
         │
         ▼
Claude Code (리뷰/검증)
         │
         └── 5. PR 리뷰 + 보안 감사
```

### MCP로 Codex를 Claude Code에서 호출하는 예시

`.claude/settings.json` 설정:

```json
{
  "mcpServers": {
    "codex": {
      "command": "npx",
      "args": ["-y", "@openai/codex-mcp"],
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

Claude Code 세션에서 호출:

```
사용자: 이 TypeScript 프로젝트의 모든 any 타입을 찾아 Codex에게 구체적 타입으로 교체하도록 지시해줘.

Claude Code 내부 처리:
1. [Glob] **/*.ts 파일 목록 수집
2. [Grep] "any" 타입 사용 위치 파악
3. [컨텍스트 생성] 각 파일의 인터페이스 구조 파악
4. [MCP: codex] 다음 작업 지시:
   "파일: src/api.ts, 라인 42: `data: any`를
    실제 API 응답 구조에 맞게 구체적 타입으로 교체하시오.
    관련 인터페이스: ApiResponse { status: number; payload: unknown }"
5. [검증] Codex 수정 결과를 TypeScript 컴파일러로 검증
```

MCP 도구 직접 호출 예시 (CLAUDE.md에서 Codex MCP 활성화 후):

```bash
# Claude Code 프롬프트에서:
"codex MCP 도구를 사용해 src/utils.ts의 버블 정렬을 퀵 정렬로 교체해줘.
 교체 후 기존 테스트가 통과하는지 확인해줘."
```

### oh-my-claudecode ccg 명령 (Claude + Codex + Gemini 삼중 병렬)

```bash
# 백엔드는 Codex, 프론트엔드는 Gemini, 조율은 Claude
/oh-my-claudecode:ccg "
백엔드 (Codex): 사용자 인증 REST API 엔드포인트 구현
프론트엔드 (Gemini): React 로그인 폼 컴포넌트 구현
조율 (Claude): 두 구현이 서로 호환되는지 검증하고 통합
"
```

### 작업 유형별 도구 선택 가이드

```
새 기능 요청이 왔다
         │
         ▼
[Claude Code] 요구사항 분석 + 설계 문서 작성
         │
         ├── 구현이 단순한 CRUD인가?
         │       YES ──> [Codex] 직접 구현 (빠름)
         │       NO ──>  [Claude Code] 구현 (정확도 우선)
         │
         ├── UI 컴포넌트 작업인가?
         │       YES ──> [Gemini] Figma 해석 + 컴포넌트 생성
         │       NO ──>  계속
         │
         ├── 보안 관련 코드인가?
         │       YES ──> [Claude Code] 반드시 Claude로 (보안 추론)
         │       NO ──>  계속
         │
         └── 대규모 리팩토링인가?
                 YES ──> [Claude Code 계획] + [Codex 실행]
                 NO ──>  [Claude Code] 단독 처리
```

---

## 요약

멀티 에이전트 패턴은 **병렬화 가능성**, **작업 복잡도**, **비용 허용치**에 따라 선택해야 한다.

- 30초 미만 단순 작업: 단일 에이전트
- 독립 분할 가능 + 각 조각 >30초: Fan-out/Fan-in
- 단계 의존성 강함: Pipeline
- 동적 작업 목록: Orchestrator-Worker
- 아키텍처 결정: Debate/Consensus
- 고신뢰 자동화: Supervisor

하이브리드 전략에서 Claude Code는 **탐색·계획·리뷰** 역할을, Codex는 **빠른 구현·반복** 역할을 맡는 것이 비용 대비 효과가 가장 높다.
