# 스킬로 반복 작업 줄이기

> 커리큘럼 04-CH01 | Claude Code & Codex 실전 스킬 패턴

---

## 목차

- [스킬이란 무엇인가](#스킬이란-무엇인가)
- [Claude Code 스킬 만들기](#claude-code-스킬-만들기)
  - [디렉토리 구조](#디렉토리-구조)
  - [코드 리뷰 스킬](#코드-리뷰-스킬)
  - [커밋 메시지 생성 스킬](#커밋-메시지-생성-스킬)
  - [테스트 작성 스킬](#테스트-작성-스킬)
- [Codex 스킬 만들기](#codex-스킬-만들기)
  - [AGENTS.md 기반 스킬 패턴](#agentsmd-기반-스킬-패턴)
- [스킬 재사용 패턴 설계](#스킬-재사용-패턴-설계)
- [$ARGUMENTS로 인자 전달하기](#arguments로-인자-전달하기)
- [프로젝트별 vs 글로벌 스킬](#프로젝트별-vs-글로벌-스킬)

---

## 스킬이란 무엇인가

스킬(Skill)은 **반복적으로 입력하는 프롬프트를 파일로 저장해 슬래시 명령어로 호출하는 재사용 가능한 템플릿**이다.

매번 이런 프롬프트를 타이핑하는 대신:

```
이 코드를 리뷰해줘. SOLID 원칙 위반 여부 확인하고,
보안 취약점 체크하고, 성능 이슈도 봐줘.
한국어로 답변해줘.
```

스킬 파일을 한 번 작성해두면 `/review`만 입력하면 된다.

### 스킬이 해결하는 문제

| 문제 | 스킬 없이 | 스킬 사용 시 |
|------|-----------|-------------|
| 일관성 | 매번 다른 프롬프트 | 동일한 기준 보장 |
| 생산성 | 프롬프트 작성 시간 | 명령어 한 줄 |
| 팀 공유 | 개인 노하우에 갇힘 | 저장소에서 공유 |
| 온보딩 | 신규 팀원이 프롬프트 패턴 학습 필요 | 명령어 목록 공유로 즉시 활용 |

---

## Claude Code 스킬 만들기

### 디렉토리 구조

Claude Code 스킬은 `.claude/commands/` 디렉토리에 마크다운 파일로 저장된다.

```
프로젝트 루트/
├── .claude/
│   └── commands/
│       ├── review.md          # /review 명령어
│       ├── commit.md          # /commit 명령어
│       ├── test.md            # /test 명령어
│       ├── refactor.md        # /refactor 명령어
│       └── explain.md         # /explain 명령어
├── src/
└── ...
```

글로벌 스킬은 `~/.claude/commands/`에 저장한다.

```
~/.claude/
└── commands/
    ├── standup.md             # /standup - 모든 프로젝트에서 사용
    ├── docs.md                # /docs - 문서 생성
    └── debug.md               # /debug - 디버깅 도우미
```

---

### 코드 리뷰 스킬

**파일 경로:** `.claude/commands/review.md`

```markdown
현재 선택된 코드 또는 $ARGUMENTS에서 지정한 파일을 코드 리뷰해줘.

## 리뷰 기준

### 1. 코드 품질
- SOLID 원칙 준수 여부
- DRY(Don't Repeat Yourself) 위반 탐지
- 함수/클래스 단일 책임 원칙

### 2. 보안
- SQL 인젝션, XSS, CSRF 취약점
- 하드코딩된 시크릿 또는 민감한 정보
- 입력값 검증 누락

### 3. 성능
- N+1 쿼리 문제
- 불필요한 반복 연산
- 메모리 누수 가능성

### 4. 가독성
- 변수/함수 이름의 명확성
- 주석 품질 (과도하거나 부족한 주석)
- 복잡도가 높은 로직

## 출력 형식

각 이슈에 대해 다음 형식으로 보고해줘:

**[심각도: 높음/중간/낮음]** 파일명:줄번호
- 문제: (무엇이 잘못됐는지)
- 이유: (왜 문제인지)
- 개선 방법: (구체적인 코드 예시 포함)

마지막에 전체 요약과 우선순위별 개선 사항 목록을 작성해줘.
한국어로 답변해줘.
```

**사용 방법:**

```bash
# 현재 열린 파일 리뷰
/review

# 특정 파일 리뷰
/review src/auth/login.ts

# 여러 파일 리뷰
/review src/auth/login.ts src/auth/session.ts
```

**실제 동작 예시:**

```
사용자: /review src/auth/login.ts

Claude: src/auth/login.ts 파일을 분석했습니다.

**[심각도: 높음]** src/auth/login.ts:42
- 문제: 패스워드를 평문으로 비교하고 있음
- 이유: 패스워드는 반드시 해시 비교해야 함
- 개선 방법:
  ```typescript
  // 변경 전
  if (user.password === inputPassword) { ... }

  // 변경 후
  import bcrypt from 'bcrypt';
  if (await bcrypt.compare(inputPassword, user.password)) { ... }
  ```

**[심각도: 중간]** src/auth/login.ts:78
- 문제: 로그인 실패 시 'Invalid password'라고 구체적으로 알려줌
- 이유: 계정 존재 여부를 노출하는 보안 취약점
- 개선 방법: 'Invalid credentials'처럼 일반적인 메시지 사용
```

---

### 커밋 메시지 생성 스킬

**파일 경로:** `.claude/commands/commit.md`

```markdown
현재 스테이징된 변경사항을 분석해서 Conventional Commits 형식의
커밋 메시지를 생성해줘.

## 분석 절차

1. `git diff --staged`를 실행해서 스테이징된 변경사항 확인
2. 변경의 성격 파악 (기능 추가, 버그 수정, 리팩토링, 문서, 테스트 등)
3. 영향받는 모듈/컴포넌트 식별

## Conventional Commits 형식

```
<type>(<scope>): <subject>

<body>

<footer>
```

### 타입 선택 기준
- `feat`: 새로운 기능 추가
- `fix`: 버그 수정
- `docs`: 문서만 변경
- `style`: 코드 의미 변경 없는 포맷 변경
- `refactor`: 버그 수정도 기능 추가도 아닌 코드 변경
- `test`: 테스트 추가 또는 수정
- `chore`: 빌드 프로세스 또는 보조 도구 변경
- `perf`: 성능 개선

## 출력

커밋 메시지 3가지 옵션을 제시해줘 (간결한 버전, 상세한 버전, 팀 협업용 버전).
각 옵션에 대해 선택 이유도 설명해줘.

$ARGUMENTS가 있으면 해당 내용을 커밋 메시지의 맥락으로 활용해줘.
(예: /commit "JIRA-123 관련 작업")
```

**사용 방법:**

```bash
# 스테이징된 변경사항으로 커밋 메시지 생성
git add src/auth/login.ts
/commit

# JIRA 이슈 번호 포함
/commit "PROJ-456 소셜 로그인 기능"

# 실제 커밋까지 자동화
/commit "PROJ-456" --auto-commit
```

**실제 동작 예시:**

```
사용자: /commit "PROJ-789 장바구니 기능"

Claude: git diff --staged를 분석했습니다.

변경된 파일: src/cart/CartService.ts, src/cart/CartController.ts

**옵션 1 (간결한 버전)**
```
feat(cart): 장바구니 수량 변경 기능 추가
```

**옵션 2 (상세한 버전)**
```
feat(cart): 장바구니 수량 변경 기능 추가

- CartService에 updateQuantity 메서드 구현
- 재고 초과 시 에러 처리 추가
- CartController에 PATCH /cart/:itemId 엔드포인트 연결

Closes PROJ-789
```

**옵션 3 (팀 협업용)**
```
feat(cart): implement cart quantity update [PROJ-789]

Add updateQuantity method to CartService with inventory validation.
Controller exposes PATCH /cart/:itemId endpoint.

Reviewed-by: (팀원 이름)
Closes: PROJ-789
```
```

---

### 테스트 작성 스킬

**파일 경로:** `.claude/commands/test.md`

```markdown
$ARGUMENTS에서 지정한 파일 또는 현재 파일에 대한 테스트를 작성해줘.

## 테스트 작성 원칙

### TDD 사이클 준수
1. 실패하는 테스트 먼저 작성 (Red)
2. 최소한의 코드로 통과 (Green)
3. 리팩토링 (Refactor)

### 테스트 분류
- **단위 테스트**: 함수/메서드 단위 격리 테스트
- **통합 테스트**: 모듈 간 상호작용 테스트
- **엣지 케이스**: 경계값, 에러 케이스, null/undefined 처리

### 테스트 명명 규칙
```
describe('컴포넌트/함수명', () => {
  describe('메서드명', () => {
    it('조건 + 기대 동작', () => { ... })
  })
})
```

## 테스트 파일 구조

대상 파일의 언어/프레임워크에 맞는 테스트 파일을 생성해줘:
- TypeScript/Node.js → Jest + ts-jest
- Python → pytest
- Go → testing 패키지

## 출력 형식

1. 테스트 파일 전체 코드 (바로 사용 가능한 형태)
2. 각 테스트 케이스에 대한 설명
3. 목(Mock) 전략 설명 (외부 의존성이 있는 경우)
4. 테스트 실행 명령어

커버리지 목표: 80% 이상 (핵심 로직 100%)
```

**사용 방법:**

```bash
# 현재 파일 테스트 생성
/test

# 특정 파일 테스트 생성
/test src/services/PaymentService.ts

# 특정 메서드에 집중한 테스트
/test src/services/PaymentService.ts processPayment 메서드
```

**생성된 테스트 파일 예시:**

```typescript
// src/services/__tests__/PaymentService.test.ts

import { PaymentService } from '../PaymentService';
import { PaymentGateway } from '../../gateways/PaymentGateway';

jest.mock('../../gateways/PaymentGateway');

describe('PaymentService', () => {
  let paymentService: PaymentService;
  let mockGateway: jest.Mocked<PaymentGateway>;

  beforeEach(() => {
    mockGateway = new PaymentGateway() as jest.Mocked<PaymentGateway>;
    paymentService = new PaymentService(mockGateway);
  });

  describe('processPayment', () => {
    it('유효한 카드 정보로 결제 성공 시 트랜잭션 ID를 반환한다', async () => {
      const mockTxId = 'txn_123456';
      mockGateway.charge.mockResolvedValue({ txId: mockTxId, status: 'success' });

      const result = await paymentService.processPayment({
        amount: 10000,
        cardNumber: '4111111111111111',
        currency: 'KRW',
      });

      expect(result.txId).toBe(mockTxId);
      expect(result.status).toBe('success');
    });

    it('잔액 부족 시 InsufficientFundsError를 던진다', async () => {
      mockGateway.charge.mockRejectedValue(new Error('insufficient_funds'));

      await expect(
        paymentService.processPayment({ amount: 999999999, cardNumber: '4111111111111111', currency: 'KRW' })
      ).rejects.toThrow('InsufficientFundsError');
    });

    it('금액이 0 이하이면 ValidationError를 던진다', async () => {
      await expect(
        paymentService.processPayment({ amount: -100, cardNumber: '4111111111111111', currency: 'KRW' })
      ).rejects.toThrow('ValidationError');
    });
  });
});
```

---

## Codex 스킬 만들기

### AGENTS.md 기반 스킬 패턴

Codex 최신 실습에서는 별도 `codex.md`보다 `AGENTS.md`를 중심 규칙 파일로 두는 편이 안정적이다. 스킬 패턴은 특정 작업 유형에 대한 지침 섹션으로 구성한다.

**파일 경로:** `AGENTS.md` (프로젝트 루트)

```markdown
# AGENTS.md

## 코드 리뷰 에이전트 (skill: review)

이 섹션은 `/review` 작업 요청 시 적용되는 지침이다.

### 리뷰 범위
- 변경된 파일만 분석 (git diff 기준)
- 테스트 파일은 별도 섹션으로 분리

### 리뷰 기준
1. 타입 안전성 (TypeScript strict 모드 기준)
2. 에러 처리 완전성
3. 비동기 처리 패턴 (async/await vs Promise chain)

### 출력 형식
Markdown 테이블로 이슈 목록 출력:
| 파일 | 줄 | 심각도 | 이슈 | 권장사항 |

---

## 커밋 에이전트 (skill: commit)

### 커밋 메시지 규칙
- Conventional Commits 1.0.0 준수
- 제목 50자 이내
- 본문은 72자에서 줄바꿈

### 자동화 범위
- `git diff --staged` 자동 읽기
- 영향받는 패키지/모듈 자동 감지
- Breaking change 감지 시 `!` 표기 추가

---

## 테스트 에이전트 (skill: test)

### 테스트 프레임워크 감지
- package.json의 `jest`, `vitest`, `mocha` 설정 자동 감지
- pytest.ini 또는 pyproject.toml 기반 Python 테스트

### 테스트 생성 규칙
1. 기존 테스트 파일이 있으면 해당 파일에 추가
2. 없으면 `__tests__/` 디렉토리에 새 파일 생성
3. 목(Mock)은 최소화 - 실제 로직 검증 우선

### 커버리지 목표
- 신규 코드: 90% 이상
- 핵심 비즈니스 로직: 100%
```

**Codex에서 스킬 호출:**

```bash
# Codex CLI에서 AGENTS.md 지침 기반 작업 실행
codex exec "review 작업: src/auth/login.ts 파일 리뷰해줘"

codex exec "commit 작업: 현재 스테이징된 변경사항으로 커밋 메시지 생성"

codex exec "test 작업: src/services/UserService.ts 테스트 파일 생성"
```

읽기 전용 리뷰라면 승인/샌드박스를 더 보수적으로 둘 수 있다:

```bash
codex exec -a untrusted -s read-only \
  "review 작업: src/auth/login.ts 파일 리뷰해줘"
```

---

## 스킬 재사용 패턴 설계

Claude Code와 Codex에서 공통으로 사용할 수 있는 스킬을 설계하는 방법이다.

### 공유 스킬 저장소 구조

```
.skills/                          # 공유 스킬 정의 (버전 관리됨)
├── definitions/
│   ├── review.yaml               # 스킬 메타데이터
│   ├── commit.yaml
│   └── test.yaml
├── prompts/
│   ├── review.md                 # 실제 프롬프트 내용
│   ├── commit.md
│   └── test.md
└── adapters/
    ├── claude-code/              # Claude Code용 링크
    │   └── sync.sh               # .claude/commands/ 로 복사
    └── codex/                    # Codex용 변환
        └── sync.sh               # AGENTS.md 로 통합

.claude/
└── commands/                     # Claude Code 스킬 (sync.sh로 생성)
    ├── review.md -> ../../.skills/prompts/review.md
    ├── commit.md -> ../../.skills/prompts/commit.md
    └── test.md -> ../../.skills/prompts/test.md
```

**동기화 스크립트:** `.skills/adapters/claude-code/sync.sh`

```bash
#!/bin/bash
# .skills/prompts/ 의 스킬을 .claude/commands/ 로 동기화

SKILLS_DIR="$(git rev-parse --show-toplevel)/.skills/prompts"
COMMANDS_DIR="$(git rev-parse --show-toplevel)/.claude/commands"

mkdir -p "$COMMANDS_DIR"

for skill_file in "$SKILLS_DIR"/*.md; do
  skill_name=$(basename "$skill_file")
  target="$COMMANDS_DIR/$skill_name"

  if [ ! -L "$target" ]; then
    ln -sf "$skill_file" "$target"
    echo "Linked: $skill_name -> .claude/commands/"
  fi
done

echo "Sync complete. Skills available:"
ls "$COMMANDS_DIR"
```

**동기화 스크립트:** `.skills/adapters/codex/sync.sh`

```bash
#!/bin/bash
# .skills/prompts/ 의 스킬을 AGENTS.md 로 통합

SKILLS_DIR="$(git rev-parse --show-toplevel)/.skills/prompts"
AGENTS_FILE="$(git rev-parse --show-toplevel)/AGENTS.md"

# 기존 자동생성 섹션 제거
if [ -f "$AGENTS_FILE" ]; then
  sed -i '/<!-- SKILLS:AUTO-START -->/,/<!-- SKILLS:AUTO-END -->/d' "$AGENTS_FILE"
fi

# 스킬 섹션 추가
echo "<!-- SKILLS:AUTO-START -->" >> "$AGENTS_FILE"
echo "# 자동 생성된 스킬 섹션" >> "$AGENTS_FILE"
echo "" >> "$AGENTS_FILE"

for skill_file in "$SKILLS_DIR"/*.md; do
  skill_name=$(basename "$skill_file" .md)
  echo "## Skill: $skill_name" >> "$AGENTS_FILE"
  cat "$skill_file" >> "$AGENTS_FILE"
  echo "" >> "$AGENTS_FILE"
done

echo "<!-- SKILLS:AUTO-END -->" >> "$AGENTS_FILE"
echo "AGENTS.md 업데이트 완료"
```

**package.json에 스크립트 등록:**

```json
{
  "scripts": {
    "skills:sync": "bash .skills/adapters/claude-code/sync.sh && bash .skills/adapters/codex/sync.sh",
    "skills:list": "ls .claude/commands/",
    "postinstall": "npm run skills:sync"
  }
}
```

---

## $ARGUMENTS로 인자 전달하기

스킬 파일 안에서 `$ARGUMENTS`는 슬래시 명령어 뒤에 입력한 모든 텍스트로 치환된다.

### 기본 인자 사용

**파일 경로:** `.claude/commands/explain.md`

```markdown
$ARGUMENTS 코드를 분석하고 설명해줘.

설명 수준: 초보자도 이해할 수 있도록
언어: 한국어
포함 내용:
1. 이 코드가 하는 일
2. 핵심 알고리즘 또는 패턴
3. 주의해야 할 부분
4. 개선 가능한 부분

코드 스니펫을 포함한 단계별 설명으로 작성해줘.
```

```bash
# 사용 예시
/explain quicksort 알고리즘

/explain src/utils/debounce.ts 파일의 debounce 함수

/explain 이 코드의 시간 복잡도
```

### 조건부 인자 활용

**파일 경로:** `.claude/commands/refactor.md`

```markdown
다음 코드 또는 $ARGUMENTS에서 지정한 파일을 리팩토링해줘.

## 리팩토링 목표

$ARGUMENTS가 비어있으면 일반적인 개선을 수행하고,
특정 목표가 명시되면 해당 목표에 집중해줘.

가능한 목표 예시:
- "성능" → 알고리즘 최적화, 캐싱 추가
- "가독성" → 함수 분리, 변수명 개선
- "타입" → TypeScript 타입 강화
- "에러처리" → try-catch 추가, 에러 타입 정의

## 출력 형식
1. 변경 전 코드
2. 변경 후 코드
3. 변경 사항 목록 (bullet points)
4. 각 변경의 이유
```

```bash
# 일반 리팩토링
/refactor src/utils/api.ts

# 성능에 집중한 리팩토링
/refactor src/utils/api.ts 성능

# 타입 개선에 집중
/refactor src/utils/api.ts 타입 안전성 강화
```

### 다중 인자 파싱 패턴

**파일 경로:** `.claude/commands/translate.md`

```markdown
$ARGUMENTS 형식: "파일경로 대상언어"

예시: /translate src/comments.ts 영어→한국어

주어진 파일의 주석과 문자열을 지정한 언어로 번역해줘.
코드 로직은 변경하지 말고 주석과 사용자에게 표시되는 문자열만 번역해.

변환 후 전체 파일을 출력해줘.
```

---

## 프로젝트별 vs 글로벌 스킬

### 글로벌 스킬 (`~/.claude/commands/`)

모든 프로젝트에서 공통으로 사용하는 스킬이다.

```
~/.claude/commands/
├── standup.md          # 스탠드업 미팅 요약
├── docs.md             # JSDoc/docstring 생성
├── debug.md            # 범용 디버깅 도우미
├── optimize.md         # 성능 최적화 분석
└── security.md         # 보안 취약점 스캔
```

**파일 예시:** `~/.claude/commands/standup.md`

```markdown
오늘 작업한 내용을 스탠드업 미팅용으로 요약해줘.

## 분석 대상
- 오늘 수정된 파일 (git log --since=today 기준)
- 완료된 TODO 주석
- 닫힌 GitHub 이슈 (있는 경우)

## 출력 형식 (Slack 메시지 형식)
*어제 한 일:*
• (bullet points)

*오늘 할 일:*
• (bullet points)

*블로커:*
• 없음 (또는 구체적인 블로커)

이모지 적절히 사용, 2-3줄로 간결하게.
```

### 프로젝트별 스킬 (`.claude/commands/`)

특정 프로젝트의 맥락에 맞는 스킬이다.

```
my-ecommerce-app/
└── .claude/
    └── commands/
        ├── review.md          # 이커머스 도메인 특화 리뷰 기준
        ├── migration.md       # DB 마이그레이션 체크리스트
        ├── api-test.md        # REST API 테스트 생성
        └── seed.md            # 테스트 데이터 시드 생성
```

**프로젝트 특화 스킬 예시:** `.claude/commands/migration.md`

```markdown
$ARGUMENTS 또는 현재 변경된 DB 스키마를 분석해서
안전한 마이그레이션 플랜을 작성해줘.

## 이 프로젝트 DB 규칙
- PostgreSQL 14 사용
- 모든 테이블에 `created_at`, `updated_at` 컬럼 필수
- soft delete는 `deleted_at` 컬럼으로 구현
- 마이그레이션 파일명: `YYYYMMDD_HHMMSS_description.sql`

## 체크리스트
- [ ] 롤백 스크립트 포함
- [ ] 대용량 테이블 변경 시 배치 처리
- [ ] 인덱스 생성은 CONCURRENTLY 옵션
- [ ] 외래키 제약조건 순서 확인

## 출력
1. 마이그레이션 SQL 파일 (up)
2. 롤백 SQL 파일 (down)
3. 예상 실행 시간
4. 주의사항
```

### 스킬 우선순위

동일한 이름의 스킬이 여러 위치에 있을 때:

```
우선순위 (높음 → 낮음):
1. .claude/commands/ (프로젝트 루트)
2. ~/.claude/commands/ (글로벌)
```

프로젝트 스킬이 항상 글로벌 스킬을 덮어쓴다. 이를 이용해 글로벌 스킬을 기본값으로 두고 프로젝트마다 커스터마이즈할 수 있다.

### 스킬 관리 베스트 프랙티스

```bash
# 현재 사용 가능한 스킬 목록 확인 (Claude Code 내에서)
/help

# 글로벌 스킬과 프로젝트 스킬 동시 확인
ls ~/.claude/commands/ && ls .claude/commands/

# 스킬 파일을 에디터로 바로 열기
code .claude/commands/review.md

# 팀에 새 스킬 배포
git add .claude/commands/new-skill.md
git commit -m "feat(skills): add new-skill for X workflow"
git push
```

---

## 요약

| 항목 | Claude Code | Codex |
|------|-------------|-------|
| 스킬 위치 | `.claude/commands/*.md` | `AGENTS.md` 섹션 |
| 글로벌 스킬 | `~/.claude/commands/` | `~/.codex/AGENTS.md` |
| 인자 전달 | `$ARGUMENTS` | 프롬프트 내 자연어 |
| 팀 공유 | Git으로 관리 | Git으로 관리 |
| 우선순위 | 프로젝트 > 글로벌 | 로컬 > 글로벌 |

스킬을 도입하면 프롬프트 작성 시간을 줄이고, 팀 전체가 일관된 기준으로 AI를 활용할 수 있다. 처음에는 가장 자주 쓰는 작업 3개(리뷰, 커밋, 테스트)부터 스킬로 만들어 보자.
