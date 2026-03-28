# oh-my-openagent로 배우는 Agent Harness 설계

> **커리큘럼**: 07-CH02 | **난이도**: 중급-고급 | **작성일**: 2026-03-28

## 목차

- [/init-deep과 계층형 AGENTS.md 구성](#init-deep과-계층형-agentsmd)
- [Sisyphus / Prometheus / Oracle / Librarian: 역할 기반 오케스트레이터](#역할-기반-오케스트레이터)
- [Skill-Embedded MCP, Hooks, Hash-Anchored Edit: 하네스의 차별점](#하네스의-차별점)
- [나만의 하네스로 축소 이식하기: Spec, Rules, Verify만 남기기](#축소-이식하기)

---

## /init-deep과 계층형 AGENTS.md

### /init-deep이란

`/init-deep`은 코드베이스를 깊이 분석하여 계층형 AGENTS.md 구조를 자동 생성하는 명령이다. 단순히 루트 AGENTS.md 하나를 만드는 것이 아니라, 디렉토리별로 맥락화된 규칙 파일을 생성한다.

실행:

```bash
# Claude Code에서 실행
/oh-my-claudecode:deepinit

# 또는 트리거 키워드
"deepinit: 이 프로젝트의 AGENTS.md 계층을 생성해줘"
```

### /init-deep이 생성하는 파일들

실제 Python 웹 API 프로젝트를 예시로:

```
my-api/
├── AGENTS.md                          # 루트: 전체 프로젝트 규칙
├── src/
│   ├── AGENTS.md                      # src: 소스 코드 규칙
│   ├── api/
│   │   ├── AGENTS.md                  # api: REST 엔드포인트 규칙
│   │   ├── v1/
│   │   │   └── AGENTS.md              # v1: 버전별 규칙
│   │   └── middleware/
│   │       └── AGENTS.md              # middleware: 미들웨어 규칙
│   ├── models/
│   │   └── AGENTS.md                  # models: ORM 모델 규칙
│   └── services/
│       └── AGENTS.md                  # services: 비즈니스 로직 규칙
├── tests/
│   └── AGENTS.md                      # tests: 테스트 작성 규칙
└── scripts/
    └── AGENTS.md                      # scripts: 스크립트 규칙
```

### 각 계층의 AGENTS.md 내용

**루트 AGENTS.md** — 프로젝트 전체 적용:

```markdown
# Project: my-api Agent Rules

## Tech Stack
- Language: Python 3.11+
- Framework: FastAPI 0.104+
- ORM: SQLAlchemy 2.0 (async)
- Database: PostgreSQL 15
- Testing: pytest + pytest-asyncio

## Architecture Decisions
- 모든 I/O는 async/await 사용 (sync 함수 금지)
- 비즈니스 로직은 services/ 레이어에서만
- 데이터베이스 직접 쿼리는 models/ 레이어에서만
- 외부 API 호출은 반드시 timeout 설정 (기본 30초)

## Code Conventions
- 함수/변수: snake_case
- 클래스: PascalCase
- 상수: UPPER_SNAKE_CASE
- 타입 힌트 필수 (모든 public 함수)
- Docstring: Google style

## Error Handling
- 모든 예외는 커스텀 예외 클래스 사용 (exceptions.py)
- HTTP 에러: raise HTTPException (코드 직접 반환 금지)
- 로깅: structlog 사용, print() 금지

## Testing Requirements
- 새 기능: 단위 테스트 + 통합 테스트 필수
- 커버리지 임계값: 80%
- 테스트 DB: SQLite in-memory (PostgreSQL 불필요)
```

**src/api/AGENTS.md** — API 레이어 전용:

```markdown
# API Layer Rules (src/api/)

## Router Rules
- 각 도메인은 별도 router 파일 (user_router.py, auth_router.py 등)
- 모든 라우터는 prefix와 tags 지정 필수
- 응답 모델은 반드시 Pydantic schema 사용

## Request/Response Pattern
```python
# 올바른 패턴
@router.post("/users", response_model=UserResponse, status_code=201)
async def create_user(
    body: UserCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> UserResponse:
    return await user_service.create(db, body)

# 금지 패턴
@router.post("/users")
async def create_user(body: dict):  # dict 금지, Pydantic 사용
    ...
```

## Validation Rules
- 입력 검증: Pydantic validator 사용 (직접 if 검사 금지)
- 인증 필요 엔드포인트: Depends(get_current_user) 반드시 포함
```

**src/models/AGENTS.md** — ORM 모델 전용:

```markdown
# Models Layer Rules (src/models/)

## SQLAlchemy Rules
- Base: DeclarativeBase 사용 (구버전 declarative_base() 금지)
- 모든 모델: id, created_at, updated_at 필드 필수
- Relationship: lazy="selectin" 기본값

## Migration Rules
- 스키마 변경 시 Alembic 마이그레이션 파일 생성 필수
- 마이그레이션 메시지: "add_{column}_to_{table}" 형식

## Example Model Pattern
```python
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        default=func.now(), onupdate=func.now()
    )
```
```

**tests/AGENTS.md** — 테스트 전용:

```markdown
# Testing Rules (tests/)

## Test Structure
- 파일명: test_{module}.py
- 함수명: test_{동작}_{조건}_{기대결과}
  예: test_create_user_with_valid_data_returns_201

## Fixtures
- DB 픽스처: conftest.py의 async_session 사용
- 인증 픽스처: conftest.py의 auth_headers 사용

## Async Tests
```python
@pytest.mark.asyncio
async def test_create_user_success(client: AsyncClient, db: AsyncSession):
    response = await client.post(
        "/api/v1/users",
        json={"email": "test@example.com", "password": "secure123"}
    )
    assert response.status_code == 201
    assert response.json()["email"] == "test@example.com"
```

## Mocking Rules
- 외부 API: pytest-httpx로 모킹
- 시간: freezegun 사용
- 절대 실제 외부 서비스 호출 금지
```

### 계층 컨텍스트 자동 주입 원리

Claude Code는 파일을 편집할 때 해당 파일의 디렉토리부터 루트까지 모든 AGENTS.md를 자동으로 읽는다.

```
src/api/v1/user_router.py 편집 시 로드 순서:
1. AGENTS.md (루트)
2. src/AGENTS.md
3. src/api/AGENTS.md
4. src/api/v1/AGENTS.md  ← 가장 구체적인 규칙
```

이 "좁혀지는 컨텍스트" 패턴이 정확한 역할 기반 행동을 가능하게 한다.

---

## 역할 기반 오케스트레이터

oh-my-openagent의 핵심은 신화적 캐릭터 이름을 가진 네 역할의 에이전트다. 각 역할은 명확한 책임 경계를 가진다.

### Sisyphus — 끝없는 실행자 (Executor/Persistence)

**목적**: 반복적이고 지속적인 작업 실행. 실패해도 포기하지 않고 다시 시작.

**설계 의도**: 그리스 신화의 시시포스처럼 끝없이 바위를 굴린다. OMC의 ralph 모드에 해당하는 개념이다.

```python
# Sisyphus 에이전트 정의 (개념적 구현)
class SisyphusAgent:
    """
    역할: 지속적 실행, 실패 복구
    모델: sonnet (비용 효율)
    루프: 무제한 (외부 종료 신호 대기)
    """

    PREAMBLE = """
    You are Sisyphus. Your mission is persistent execution.

    Core behaviors:
    1. Execute the assigned task completely
    2. If blocked or failed, analyze the failure and retry
    3. Never give up unless explicitly told to stop
    4. Track attempts in state file
    5. Escalate to Oracle after 3 failed attempts

    You do NOT decide WHAT to do — that's Prometheus's job.
    You ONLY execute and persist.
    """

    def __init__(self, max_attempts: int = 10):
        self.max_attempts = max_attempts
        self.state_file = ".harness/sisyphus-state.json"

    def execute_loop(self, task: str):
        attempt = 0
        while attempt < self.max_attempts:
            try:
                result = self._execute_task(task)
                if result.success:
                    self._update_state({"status": "complete", "attempts": attempt + 1})
                    return result
                attempt += 1
                self._update_state({"status": "retrying", "attempt": attempt, "error": result.error})
            except Exception as e:
                attempt += 1
                if attempt >= 3:
                    self._escalate_to_oracle(task, str(e))
```

실제 사용 패턴:

```bash
# Sisyphus에게 반복 작업 할당
cat > .harness/tasks/sisyphus-001.json << 'EOF'
{
  "id": "sisyphus-001",
  "agent": "sisyphus",
  "task": "테스트 스위트가 모두 통과할 때까지 실패한 테스트를 수정하라",
  "success_criteria": "pytest --tb=short 출력에서 PASSED만 보임",
  "max_attempts": 10
}
EOF
```

### Prometheus — 전략적 계획자 (Planner/Architect)

**목적**: 높은 수준의 전략 수립, 작업 분해, 리소스 배분. 직접 코드를 작성하지 않는다.

**설계 의도**: 불을 훔쳐 인류에게 준 프로메테우스처럼, 고차원 지식을 실행 가능한 계획으로 변환한다.

```python
class PrometheusAgent:
    """
    역할: 전략 수립, 작업 분해
    모델: opus (복잡한 추론 필요)
    출력: 실행 계획, 에이전트 할당
    """

    PREAMBLE = """
    You are Prometheus. Your mission is strategic planning.

    Core behaviors:
    1. Analyze the high-level goal
    2. Decompose into atomic tasks (each <2 hours of work)
    3. Assign each task to the right agent (Sisyphus/Oracle/Librarian)
    4. Identify dependencies and parallelizable tasks
    5. Define success criteria for each task

    You do NOT write code or fix bugs.
    You ONLY plan, decompose, and coordinate.

    Output format (always JSON):
    {
      "plan_id": "unique-id",
      "goal": "original goal",
      "tasks": [
        {
          "id": "task-001",
          "agent": "sisyphus|oracle|librarian",
          "description": "...",
          "depends_on": [],
          "success_criteria": "...",
          "estimated_complexity": "low|medium|high"
        }
      ],
      "parallel_groups": [["task-001", "task-002"], ["task-003"]]
    }
    """
```

실제 Prometheus 계획 출력 예시:

```json
{
  "plan_id": "plan-2026-03-28-001",
  "goal": "PostgreSQL을 MongoDB로 마이그레이션",
  "tasks": [
    {
      "id": "task-001",
      "agent": "oracle",
      "description": "현재 PostgreSQL 스키마 분석 및 MongoDB 문서 구조 설계",
      "depends_on": [],
      "success_criteria": "MongoDB 스키마 설계 문서 생성 완료",
      "estimated_complexity": "high"
    },
    {
      "id": "task-002",
      "agent": "librarian",
      "description": "motor (MongoDB async driver) 공식 문서 수집",
      "depends_on": [],
      "success_criteria": "motor API 참조 문서 .harness/docs/motor.md에 저장",
      "estimated_complexity": "low"
    },
    {
      "id": "task-003",
      "agent": "sisyphus",
      "description": "SQLAlchemy 모델을 Motor Document 모델로 변환",
      "depends_on": ["task-001", "task-002"],
      "success_criteria": "모든 모델 파일이 Motor 패턴으로 변환됨",
      "estimated_complexity": "medium"
    }
  ],
  "parallel_groups": [
    ["task-001", "task-002"],
    ["task-003"]
  ]
}
```

### Oracle — 지식 분석가 (Analyst/Debugger)

**목적**: 복잡한 문제 분석, 근본 원인 파악, 아키텍처 결정. 실행하지 않고 통찰을 제공한다.

**설계 의도**: 델포이의 신탁처럼, 불확실한 상황에서 방향을 제시한다.

```python
class OracleAgent:
    """
    역할: 깊은 분석, 근본 원인 파악
    모델: opus (복잡한 추론)
    트리거: Sisyphus 실패 3회, 복잡한 아키텍처 질문
    """

    PREAMBLE = """
    You are Oracle. Your mission is deep analysis and insight.

    You are called when:
    1. Sisyphus has failed 3+ times on the same task
    2. An architectural decision needs careful analysis
    3. Root cause of a complex bug is unknown

    Core behaviors:
    1. Read ALL relevant context (state files, logs, code)
    2. Identify the root cause or hidden constraint
    3. Provide specific, actionable guidance
    4. Do NOT implement — provide the insight for Sisyphus to act on

    Output format:
    - Root cause analysis
    - Recommended approach (with rationale)
    - Specific steps for Sisyphus to follow
    - Risks and mitigations
    """

    def analyze_sisyphus_failure(self, task: dict, failure_history: list) -> dict:
        """Sisyphus 실패 이력을 받아 근본 원인 분석"""
        context = {
            "task": task,
            "failures": failure_history,
            "relevant_files": self._gather_context(task)
        }
        return self._deep_analyze(context)
```

Oracle 분석 출력 예시:

```markdown
# Oracle Analysis: task-003 반복 실패

## 근본 원인
SQLAlchemy의 `relationship()` 사용 시 Motor에는 직접 대응하는 개념이 없음.
Motor는 관계형 JOIN을 지원하지 않으며, 임베디드 문서나 수동 참조를 사용해야 함.

## 숨겨진 제약
현재 코드베이스의 User-Order 관계가 6단계 깊이로 중첩됨.
단순 변환 불가 — 데이터 접근 패턴 재설계 필요.

## Sisyphus를 위한 구체적 단계
1. `src/models/` 의 모든 `relationship()` 호출 목록화
2. 각 관계에 대해: 임베디드 문서(1:1) vs 수동 참조(1:N) 결정
3. 접근 빈도가 높은 관계는 임베디드로 변환
4. 변환 후 각 컬렉션의 쿼리 패턴 테스트

## 위험 요소
- Order 컬렉션이 임베디드 시 무제한 성장 가능 → 배열 크기 제한 설정 필요
- 트랜잭션 원자성 손실 → 보상 트랜잭션 패턴 고려
```

### Librarian — 지식 수집가 (Document Specialist/Researcher)

**목적**: 공식 문서, API 레퍼런스, 외부 지식 수집 및 정리. 코드 작성이나 분석 없이 정보만 수집한다.

**설계 의도**: 알렉산드리아 도서관의 사서처럼, 정확한 정보를 체계적으로 수집하고 보관한다.

```python
class LibrarianAgent:
    """
    역할: 공식 문서 수집, 지식 정리
    모델: haiku (정보 수집은 추론 불필요)
    출력: .harness/docs/ 에 마크다운 파일
    """

    PREAMBLE = """
    You are Librarian. Your mission is accurate knowledge collection.

    Core behaviors:
    1. Fetch official documentation (WebFetch, WebSearch)
    2. Verify accuracy against the official source
    3. Summarize in Korean with English code examples preserved
    4. Save to .harness/docs/{library-name}.md
    5. Include: API signatures, common patterns, gotchas, version requirements

    You do NOT analyze, implement, or make decisions.
    You ONLY collect and organize knowledge accurately.

    Anti-patterns to avoid:
    - Do NOT guess API signatures (always verify with official docs)
    - Do NOT include deprecated patterns
    - Do NOT summarize without reading the actual source
    """

    DOCS_DIR = ".harness/docs"

    def collect(self, library: str, topics: list[str]) -> str:
        """공식 문서에서 특정 주제 수집"""
        docs = []
        for topic in topics:
            url = self._resolve_official_url(library, topic)
            content = self._fetch_and_parse(url)
            docs.append(content)

        output_path = f"{self.DOCS_DIR}/{library}.md"
        self._save_organized(output_path, docs)
        return output_path
```

Librarian이 생성하는 문서 예시 (`.harness/docs/motor.md`):

```markdown
# Motor (MongoDB Async Driver) 레퍼런스

> 출처: https://motor.readthedocs.io/en/stable/
> 버전: Motor 3.3+, Python 3.8+, PyMongo 4.5+
> 수집일: 2026-03-28

## 설치

```bash
pip install motor
```

## 기본 연결

```python
import motor.motor_asyncio

# 비동기 클라이언트
client = motor.motor_asyncio.AsyncIOMotorClient("mongodb://localhost:27017")
db = client["mydb"]
collection = db["users"]
```

## CRUD 패턴

### 삽입
```python
# 단건 삽입
result = await collection.insert_one({"name": "Alice", "email": "alice@example.com"})
doc_id = result.inserted_id

# 다건 삽입
result = await collection.insert_many([
    {"name": "Bob"},
    {"name": "Carol"}
])
```

### 조회
```python
# 단건 조회
doc = await collection.find_one({"email": "alice@example.com"})

# 다건 조회 (비동기 이터레이터)
async for doc in collection.find({"active": True}):
    print(doc)

# 필드 제한
doc = await collection.find_one(
    {"email": "alice@example.com"},
    {"_id": 0, "name": 1, "email": 1}
)
```

## FastAPI 통합 패턴

```python
# lifespan으로 연결 관리
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.mongodb_client = AsyncIOMotorClient(settings.MONGODB_URL)
    app.mongodb = app.mongodb_client[settings.DB_NAME]
    yield
    app.mongodb_client.close()

app = FastAPI(lifespan=lifespan)

# 의존성
async def get_db(request: Request) -> AsyncIOMotorDatabase:
    return request.app.mongodb
```

## 주의사항 (Gotchas)
- `find_one` 결과는 `None` 가능 — 항상 None 체크 필요
- `_id` 필드는 `ObjectId` 타입 — JSON 직렬화 시 `str(doc["_id"])` 변환 필요
- 트랜잭션은 replica set 필요 — 단일 노드에서 불가
```

### 네 역할 비교

| 역할 | 신화 | 모델 | 핵심 행동 | 절대 하지 않는 것 |
|------|------|------|-----------|------------------|
| **Sisyphus** | 시시포스 | sonnet | 실행, 재시도, 지속 | 계획, 분석 |
| **Prometheus** | 프로메테우스 | opus | 전략, 분해, 조율 | 코드 작성, 실행 |
| **Oracle** | 신탁 | opus | 분석, 진단, 통찰 | 구현, 계획 |
| **Librarian** | 사서 | haiku | 수집, 정리, 보관 | 분석, 결정 |

---

## 하네스의 차별점

### Skill-Embedded MCP

일반적인 MCP는 외부 서버로 운영된다. Skill-Embedded MCP는 스킬(마크다운 지침) 안에 MCP 도구 호출 패턴을 직접 내장하는 방식이다.

**일반 패턴** — MCP가 외부에서 도구를 제공:

```
Claude ─→ MCP Server ─→ 도구 실행 ─→ 결과 반환
```

**Skill-Embedded 패턴** — 스킬이 MCP 도구 사용 방법을 지정:

```markdown
# skill: collect-docs

## Steps
1. Invoke `mcp__librarian__fetch_official_docs` with:
   - library: {library_name}
   - version: {version}
   - save_path: ".harness/docs/{library_name}.md"

2. After fetch, invoke `mcp__librarian__validate_completeness` with:
   - doc_path: ".harness/docs/{library_name}.md"
   - required_sections: ["installation", "basic_usage", "api_reference"]

3. If validation fails, re-fetch with expanded scope
```

실제 구현 — MCP 서버와 스킬의 결합:

```javascript
// bridge/librarian-mcp.cjs
const MCP_TOOLS = {
  fetch_official_docs: {
    description: "공식 문서를 가져와 파일로 저장",
    inputSchema: {
      type: "object",
      properties: {
        library: { type: "string", description: "라이브러리 이름" },
        version: { type: "string", description: "대상 버전" },
        save_path: { type: "string", description: "저장 경로" }
      },
      required: ["library", "save_path"]
    },
    handler: async ({ library, version, save_path }) => {
      const url = resolveOfficialUrl(library, version);
      const content = await fetchAndParse(url);
      await fs.writeFile(save_path, content, 'utf8');
      return { success: true, path: save_path, chars: content.length };
    }
  },

  validate_completeness: {
    description: "수집된 문서의 완전성 검사",
    inputSchema: {
      type: "object",
      properties: {
        doc_path: { type: "string" },
        required_sections: { type: "array", items: { type: "string" } }
      }
    },
    handler: async ({ doc_path, required_sections }) => {
      const content = await fs.readFile(doc_path, 'utf8');
      const missing = required_sections.filter(s => !content.includes(`## ${s}`));
      return { complete: missing.length === 0, missing_sections: missing };
    }
  }
};
```

스킬 파일 (`skills/collect-docs/SKILL.md`):

```markdown
# Collect Docs Skill

## Trigger
"문서 수집", "collect docs for", "fetch {library} docs"

## Agent: Librarian

## Embedded MCP Calls
이 스킬은 다음 MCP 도구를 순서대로 호출한다:

1. `mcp__harness_librarian__fetch_official_docs`
   ```json
   {
     "library": "{parsed_library_name}",
     "version": "{parsed_version_or_latest}",
     "save_path": ".harness/docs/{library_name}.md"
   }
   ```

2. `mcp__harness_librarian__validate_completeness`
   ```json
   {
     "doc_path": ".harness/docs/{library_name}.md",
     "required_sections": ["installation", "basic_usage", "api_reference", "gotchas"]
   }
   ```

3. 검증 실패 시 — `mcp__harness_librarian__fetch_official_docs` 재호출
   (expanded scope: include examples and tutorials)

## Success Criteria
- 문서 파일 생성됨
- 모든 required_sections 존재
- 공식 출처 URL 헤더에 기록됨
```

**Skill-Embedded MCP의 장점**:
- 스킬 마크다운만 읽어도 어떤 도구가 어떤 순서로 호출되는지 파악 가능
- 도구 호출 파라미터가 스킬 정의에 문서화됨
- 스킬 단위로 테스트 및 디버깅 가능

### Hash-Anchored Edit

대규모 파일 편집 시 가장 흔한 실패는 잘못된 위치를 수정하는 것이다. Hash-Anchored Edit은 편집 대상을 해시값으로 고정하여 정확도를 높인다.

**문제 상황**: 3000줄 파일에서 특정 함수를 수정할 때, 유사한 패턴이 여러 개 있으면 잘못된 위치를 수정할 수 있다.

**Hash-Anchored Edit 방식**:

```python
# harness/tools/anchored_edit.py
import hashlib
import re
from pathlib import Path

def compute_anchor(content: str, start_line: int, end_line: int) -> str:
    """편집 대상 범위의 해시값 계산"""
    lines = content.split('\n')
    target = '\n'.join(lines[start_line-1:end_line])
    return hashlib.sha256(target.encode()).hexdigest()[:12]

def anchored_edit(
    file_path: str,
    old_string: str,
    new_string: str,
    anchor: str  # 해시값
) -> dict:
    """해시 앵커로 검증된 안전한 편집"""
    content = Path(file_path).read_text()

    # 1. old_string이 실제로 존재하는지 확인
    if old_string not in content:
        return {"success": False, "error": "old_string not found in file"}

    # 2. 해시 앵커로 올바른 위치인지 검증
    idx = content.index(old_string)
    actual_hash = hashlib.sha256(old_string.encode()).hexdigest()[:12]

    if actual_hash != anchor:
        return {
            "success": False,
            "error": f"Anchor mismatch: expected {anchor}, got {actual_hash}",
            "hint": "File may have changed since anchor was computed"
        }

    # 3. 고유성 확인 (동일 문자열이 여러 곳에 있으면 거부)
    occurrences = content.count(old_string)
    if occurrences > 1:
        return {
            "success": False,
            "error": f"old_string appears {occurrences} times. Provide more context.",
            "locations": find_all_locations(content, old_string)
        }

    # 4. 안전하게 교체
    new_content = content.replace(old_string, new_string, 1)
    Path(file_path).write_text(new_content)

    return {
        "success": True,
        "replaced_at": idx,
        "new_anchor": hashlib.sha256(new_string.encode()).hexdigest()[:12]
    }
```

MCP 도구로 노출:

```javascript
// bridge/mcp-server.cjs 내 도구 정의
anchored_edit: {
  description: "해시 앵커로 검증된 안전한 파일 편집",
  inputSchema: {
    type: "object",
    properties: {
      file_path: { type: "string" },
      old_string: { type: "string", description: "교체할 원본 텍스트" },
      new_string: { type: "string", description: "교체될 새 텍스트" },
      anchor: {
        type: "string",
        description: "old_string의 SHA256 해시 앞 12자리 (compute_anchor로 생성)"
      }
    },
    required: ["file_path", "old_string", "new_string", "anchor"]
  }
}
```

실제 사용 워크플로우:

```bash
# Step 1: 편집 전 앵커 계산
python3 -c "
import hashlib
old = '''def get_user(user_id: int) -> User:
    return db.query(User).filter(User.id == user_id).first()'''
print(hashlib.sha256(old.encode()).hexdigest()[:12])
"
# 출력: a3f9b2c1d4e7

# Step 2: Claude에게 anchored_edit 호출 지시
# "src/services/user_service.py 파일의 get_user 함수를
#  앵커 a3f9b2c1d4e7를 사용해서 수정해줘"

# Step 3: MCP 도구 호출 (Claude가 자동 실행)
# anchored_edit(
#   file_path="src/services/user_service.py",
#   old_string="def get_user...",
#   new_string="async def get_user...",
#   anchor="a3f9b2c1d4e7"
# )
```

**Hash-Anchored Edit의 장점**:
- 유사한 코드 패턴이 많은 파일에서 정확한 위치 편집 보장
- 파일 변경 감지 (다른 에이전트가 먼저 수정했는지 확인)
- 편집 이력 추적 (before anchor → after anchor)

---

## 축소 이식하기

oh-my-openagent의 전체 복잡성 없이 핵심 3요소만 추출하여 최소 하네스를 구성한다.

### 최소 하네스 3요소

```
최소 하네스 (Minimal Harness):
├── Spec.md        # 무엇을 만드는지 (명세)
├── Rules          # 어떻게 만드는지 (규칙)
│   ├── AGENTS.md  # 에이전트별 행동 규칙
│   └── CLAUDE.md  # 전체 행동 지침
└── verify.sh      # 올바르게 만들었는지 (검증)
```

**Spec.md** — 구현 명세서:

```markdown
# Feature Spec: 사용자 인증 API

## Goal
JWT 기반 사용자 인증 시스템 구현

## Requirements

### 기능 요구사항
- POST /auth/register: 이메일 + 비밀번호로 회원가입
- POST /auth/login: 로그인 후 JWT access/refresh 토큰 발급
- POST /auth/refresh: refresh 토큰으로 access 토큰 갱신
- POST /auth/logout: refresh 토큰 무효화

### 비기능 요구사항
- Access 토큰 만료: 15분
- Refresh 토큰 만료: 30일
- 비밀번호 해싱: bcrypt (rounds=12)
- JWT 알고리즘: RS256

## Out of Scope
- OAuth2 소셜 로그인
- MFA (다단계 인증)
- 이메일 검증

## Acceptance Criteria
- [ ] `pytest tests/test_auth.py` 전체 통과
- [ ] `mypy src/` 오류 없음
- [ ] 모든 엔드포인트 OpenAPI 문서 자동 생성
- [ ] 응답 시간 p99 < 200ms (로컬 기준)
```

**AGENTS.md** — 에이전트 행동 규칙:

```markdown
# Agent Rules for: 사용자 인증 API

## Sisyphus (Executor) Rules
- 구현 시작 전 Spec.md의 Requirements 섹션 반드시 읽기
- 각 엔드포인트 구현 후 verify.sh 실행
- 테스트 실패 시 최대 3회 재시도, 초과 시 Oracle에게 에스컬레이션

## Prometheus (Planner) Rules
- 구현 순서: models → services → routes → tests 순서
- 의존성 우선: 다른 것이 의존하는 것부터 구현

## Oracle (Analyst) Rules
- 보안 결정 시 OWASP JWT 가이드라인 참조
- 토큰 저장 방식은 httpOnly 쿠키 vs Authorization 헤더 트레이드오프 분석

## Librarian Rules
- 사용 전 python-jose, passlib 공식 문서 반드시 수집
- 수집 경로: .harness/docs/python-jose.md, .harness/docs/passlib.md

## All Agents
- 절대 하지 않는 것: print() 사용, plain text 비밀번호 저장, 하드코딩 시크릿
- 항상 하는 것: 타입 힌트, 에러 처리, 로깅 (structlog)
```

**verify.sh** — 자동 검증 스크립트:

```bash
#!/bin/bash
# verify.sh - 구현 완료 검증

set -euo pipefail

PASS=0
FAIL=0
ERRORS=()

check() {
  local name="$1"
  local cmd="$2"

  if eval "$cmd" > /tmp/verify-output.txt 2>&1; then
    echo "  PASS: $name"
    ((PASS++)) || true
  else
    echo "  FAIL: $name"
    echo "    $(head -5 /tmp/verify-output.txt)"
    ERRORS+=("$name")
    ((FAIL++)) || true
  fi
}

echo "=== Verification Start ==="

# 1. 타입 검사
check "mypy type check" "mypy src/ --ignore-missing-imports"

# 2. 린트
check "ruff lint" "ruff check src/"

# 3. 단위 테스트
check "pytest unit tests" "pytest tests/unit/ -v --tb=short"

# 4. 통합 테스트
check "pytest integration tests" "pytest tests/integration/ -v --tb=short"

# 5. 커버리지
check "coverage >= 80%" "pytest --cov=src --cov-fail-under=80 -q"

# 6. OpenAPI 문서 생성 확인
check "openapi schema valid" "python -c \"
from src.main import app
import json
schema = app.openapi()
assert '/auth/register' in str(schema)
assert '/auth/login' in str(schema)
print('OpenAPI schema valid')
\""

# 7. 시크릿 하드코딩 검사
check "no hardcoded secrets" "! grep -rn 'secret_key\s*=\s*[\"'\''][^{]' src/ --include='*.py'"

echo ""
echo "=== Results ==="
echo "PASS: $PASS, FAIL: $FAIL"

if [ ${#ERRORS[@]} -gt 0 ]; then
  echo ""
  echo "Failed checks:"
  for err in "${ERRORS[@]}"; do
    echo "  - $err"
  done
  exit 1
fi

echo "All checks passed!"
exit 0
```

### 축소 이식 단계별 가이드

**1단계: 디렉토리 초기화**

```bash
mkdir -p .harness/{docs,state,tasks,hooks}

# Spec 작성 (비어있는 템플릿)
cat > Spec.md << 'EOF'
# Feature Spec: {기능명}

## Goal
{목표 한 문장}

## Requirements
### 기능 요구사항
-

### 비기능 요구사항
-

## Acceptance Criteria
- [ ]
EOF

# 상태 초기화
cat > .harness/state/current.json << 'EOF'
{
  "phase": "plan",
  "spec": "Spec.md",
  "started_at": null,
  "completed_at": null,
  "verify_results": null
}
EOF
```

**2단계: CLAUDE.md 작성 (역할 정의)**

```bash
cat > CLAUDE.md << 'EOF'
# Minimal Agent Harness

<roles>
## Planner
작업: Spec.md를 읽고 구현 계획 수립
출력: .harness/tasks/ 에 JSON 파일
제약: 코드 작성 금지

## Executor
작업: Planner의 계획에 따라 구현
출력: 변경된 소스 파일
제약: Spec.md 범위 이탈 금지, 각 변경 후 verify.sh 실행

## Verifier
작업: verify.sh 실행 및 결과 분석
출력: .harness/state/current.json 업데이트
제약: 테스트 수정 금지 (실패하면 구현 수정)
</roles>

<workflow>
1. Spec.md 읽기
2. Planner: 구현 계획 → .harness/tasks/
3. Executor: 구현 → 각 단계 후 verify.sh
4. Verifier: 최종 verify.sh → 완료 또는 수정
</workflow>

<rules>
- Spec.md의 Out of Scope 항목은 절대 구현하지 않음
- verify.sh가 통과해야만 "완료" 선언 가능
- 3회 연속 verify.sh 실패 시 중단하고 사용자에게 보고
</rules>
EOF
```

**3단계: verify.sh 작성 (프로젝트별 커스터마이즈)**

```bash
cat > verify.sh << 'EOF'
#!/bin/bash
# 이 파일을 프로젝트에 맞게 수정하세요

PASS=0; FAIL=0

check() {
  if eval "$2" &>/dev/null; then
    echo "  PASS: $1"; ((PASS++)) || true
  else
    echo "  FAIL: $1"; ((FAIL++)) || true
  fi
}

# ===== 여기서부터 프로젝트별 검사 추가 =====
check "syntax check"    "python -m py_compile src/**/*.py"
check "tests pass"      "pytest tests/ -q"
check "no type errors"  "mypy src/"
# ===== 여기까지 =====

echo "PASS: $PASS, FAIL: $FAIL"
[ $FAIL -eq 0 ] && exit 0 || exit 1
EOF
chmod +x verify.sh
```

**4단계: 훅으로 자동 verify 트리거**

```bash
cat > .claude/settings.json << 'EOF'
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "node .harness/hooks/post-edit.mjs"
          }
        ]
      }
    ]
  }
}
EOF

cat > .harness/hooks/post-edit.mjs << 'EOF'
// 파일 편집 후 verify 실행 권장 메시지 출력
process.stdout.write(JSON.stringify({
  type: "system_reminder",
  content: "파일이 수정되었습니다. verify.sh를 실행하여 Acceptance Criteria를 확인하세요."
}));
EOF
```

**5단계: 실행 및 검증**

```bash
# Spec.md 작성 완료 후 Claude에게:
# "Spec.md를 읽고 구현 계획을 세워줘"

# 계획 확인 후:
# "계획에 따라 구현해줘. 각 단계 후 verify.sh 실행해줘"

# 완료 확인:
bash verify.sh
# Expected: "PASS: N, FAIL: 0" and exit 0
```

### 최종 결과물 디렉토리 구조

```
my-project/
├── CLAUDE.md                    # 역할 정의 + 워크플로우 규칙
├── AGENTS.md                    # 프로젝트별 에이전트 행동 규칙
├── Spec.md                      # 구현 명세 (인수 조건 포함)
├── verify.sh                    # 자동 검증 스크립트
├── .claude/
│   └── settings.json            # PostToolUse 훅 설정
├── .harness/
│   ├── docs/                    # Librarian이 수집한 공식 문서
│   │   ├── python-jose.md
│   │   └── passlib.md
│   ├── state/
│   │   └── current.json         # 현재 파이프라인 단계
│   ├── tasks/                   # Planner가 생성한 작업 파일
│   │   ├── task-001.json
│   │   └── task-002.json
│   └── hooks/
│       └── post-edit.mjs        # 편집 후 verify 권장 훅
└── src/
    └── ...
```

### 하네스 효과 측정

```bash
# 하네스 도입 전후 비교 측정 스크립트
cat > .harness/measure.sh << 'EOF'
#!/bin/bash

echo "=== Harness Health Check ==="

# 1. Spec.md 존재 여부
[ -f Spec.md ] && echo "  PASS: Spec.md 존재" || echo "  FAIL: Spec.md 없음"

# 2. Acceptance Criteria 모두 체크됐는지
UNCHECKED=$(grep -c "^- \[ \]" Spec.md 2>/dev/null || echo 0)
CHECKED=$(grep -c "^- \[x\]" Spec.md 2>/dev/null || echo 0)
echo "  Criteria: $CHECKED 완료 / $((CHECKED + UNCHECKED)) 전체"

# 3. verify.sh 마지막 실행 결과
if [ -f .harness/state/last-verify.json ]; then
  LAST=$(jq -r '.result' .harness/state/last-verify.json)
  TS=$(jq -r '.timestamp' .harness/state/last-verify.json)
  echo "  Last verify: $LAST ($TS)"
else
  echo "  Last verify: 아직 실행 안 됨"
fi

# 4. 수집된 문서
DOC_COUNT=$(ls .harness/docs/*.md 2>/dev/null | wc -l | tr -d ' ')
echo "  Collected docs: $DOC_COUNT 개"
EOF
chmod +x .harness/measure.sh
```

---

## 핵심 정리

oh-my-openagent에서 배울 수 있는 Agent Harness 설계 원칙:

1. **계층형 컨텍스트**: AGENTS.md를 디렉토리별로 배치하면 에이전트가 편집하는 파일의 "지역 규칙"을 자동으로 읽는다. 단일 AGENTS.md보다 훨씬 정밀한 제어가 가능하다.

2. **역할의 명확한 경계**: Sisyphus(실행), Prometheus(계획), Oracle(분석), Librarian(수집)은 서로의 영역을 침범하지 않는다. 이 경계가 에이전트의 예측 가능성을 높인다.

3. **Skill-Embedded MCP**: 스킬 마크다운 안에 MCP 도구 호출 순서를 명시하면, 스킬 파일만 읽어도 전체 워크플로우를 파악할 수 있다. 문서화와 실행이 동일한 파일에 있다.

4. **Hash-Anchored Edit**: 대규모 파일 편집 시 해시 앵커는 "올바른 위치를 수정했는가"를 기계적으로 검증한다. 편집 실수를 사전에 방지한다.

5. **최소 하네스 3요소**: Spec, Rules, Verify만 있으면 동작하는 하네스를 구성할 수 있다. 복잡한 인프라 없이도 역할 분리와 검증이 가능하다.
