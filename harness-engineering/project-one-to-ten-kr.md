# 프로젝트 2 - 1 to 10: 동작하는 코드를 좋은 코드로

> 커리큘럼 06-CH02 대응 문서

## 목차

- [1. AI가 만든 코드 품질 리뷰](#1-ai가-만든-코드-품질-리뷰)
- [2. 리팩터링 루프 만들기](#2-리팩터링-루프-만들기)
- [3. 기능 추가 요청을 받아 다시 계획 세우기](#3-기능-추가-요청을-받아-다시-계획-세우기)
- [4. 프로젝트 회고: 무엇을 맡기고 무엇을 직접 할 것인가](#4-프로젝트-회고-무엇을-맡기고-무엇을-직접-할-것인가)

---

## 1. AI가 만든 코드 품질 리뷰

AI가 생성한 코드는 "동작하지만 좋지 않은" 경우가 많다. 타입 안전성 부족, 에러 핸들링 누락, 관심사 미분리가 주요 패턴이다.

### 1.1 코드 리뷰 프롬프트 패턴 5가지

**패턴 1 - 전체 파일 종합 리뷰**

```
아래 코드를 시니어 엔지니어 관점에서 리뷰해 주세요.

**리뷰 기준** (각 항목에 심각도: Critical / Major / Minor로 표시):
1. 타입 안전성: any 타입, 불필요한 타입 단언, 런타임 오류 가능성
2. 에러 핸들링: try-catch 누락, 에러 전파 방식, 사용자 피드백
3. 성능: 불필요한 리렌더, 메모이제이션 누락, N+1 쿼리
4. 보안: 입력 검증, 인증 검사, SQL 인젝션 가능성
5. 가독성: 함수 크기, 변수명, 중복 코드
6. React 관행: 훅 규칙, 부수 효과 관리, 컴포넌트 책임 범위

**출력 형식**:
## 요약
전체 코드 품질 평가 (A/B/C/D)

## 발견된 문제
### Critical
- [파일명:줄번호] 문제 설명 → 수정 방향

### Major
- ...

### Minor
- ...

## 즉시 수정 권장 사항 TOP 3
1. ...

**대상 코드**:
```typescript
{여기에 코드 붙여넣기}
```
```

**패턴 2 - 보안 집중 리뷰**

```
아래 코드에서 보안 취약점을 찾아주세요.
Next.js API Route / Supabase Row Level Security 맥락입니다.

**확인 항목**:
1. 인증 없이 접근 가능한 엔드포인트가 있는가?
2. 사용자 입력이 DB 쿼리에 직접 삽입되는가?
3. 다른 사용자의 데이터에 접근할 수 있는 경로가 있는가?
4. 환경변수가 클라이언트에 노출되는가? (NEXT_PUBLIC_ 여부)
5. CORS, CSRF 관련 설정이 누락되었는가?

발견된 각 취약점에 대해:
- 취약점 유형 (OWASP Top 10 기준)
- 공격 시나리오 (어떻게 악용될 수 있는가)
- 수정 코드 예시

**대상 코드**:
```typescript
{여기에 코드 붙여넣기}
```
```

**패턴 3 - 성능 리뷰**

```
아래 React 컴포넌트의 성능 문제를 분석해 주세요.

**확인 항목**:
1. 불필요한 리렌더링을 유발하는 패턴
   - 인라인 객체/함수 생성
   - Context 남용
   - 의존성 배열 오류
2. 메모이제이션이 필요한 곳 (useMemo, useCallback, React.memo)
3. 데이터 패칭 최적화 (폭포식 패칭, 중복 요청)
4. 번들 크기 영향 (동적 임포트 기회)

각 문제에 대해 Before/After 코드를 보여주세요.

**대상 컴포넌트**:
```typescript
{여기에 코드 붙여넣기}
```
```

**패턴 4 - 타입 안전성 리뷰**

```
아래 TypeScript 코드의 타입 안전성을 강화해 주세요.

**현재 문제점을 찾고 수정**:
1. `any` 타입을 구체적 타입으로 교체
2. `as` 타입 단언을 타입 가드로 교체
3. `undefined` / `null` 처리 누락
4. 제네릭으로 재사용성을 높일 수 있는 곳
5. Zod 스키마와 TypeScript 타입이 일치하지 않는 곳

수정 전 / 수정 후 코드를 나란히 보여주고,
각 변경의 이유를 한 줄로 설명해 주세요.

**대상 코드**:
```typescript
{여기에 코드 붙여넣기}
```
```

**패턴 5 - 구조/아키텍처 리뷰**

```
아래 파일 구조와 코드를 보고 아키텍처 관점에서 리뷰해 주세요.

**프로젝트 구조**:
```
{tree 명령어 출력 붙여넣기}
```

**핵심 파일들**:
```typescript
// {파일명 1}
{내용}

// {파일명 2}
{내용}
```

**확인 항목**:
1. 관심사 분리: UI / 비즈니스 로직 / 데이터 접근이 혼재되어 있는가?
2. 의존성 방향: 상위 레이어가 하위 레이어에 올바르게 의존하는가?
3. 코드 중복: 공통 로직이 여러 곳에 복사되어 있는가?
4. 확장성: 기능 추가 시 어떤 파일을 수정해야 하는가? 너무 많은가?

개선 방향을 구체적 리팩터링 제안으로 작성해 주세요.
```

---

### 1.2 흔한 AI 코드 문제점 체크리스트

Codex / Claude Code가 생성한 코드를 병합하기 전에 이 목록을 확인한다.

```markdown
## AI 생성 코드 사전 검토 체크리스트

### 타입 안전성
- [ ] `any` 타입 사용 없음 (grep -r ": any" src/)
- [ ] `as unknown as` 이중 단언 없음
- [ ] API 응답에 Zod 런타임 검증 적용
- [ ] `undefined` 반환 가능한 함수에 옵셔널 체이닝 처리

### 에러 핸들링
- [ ] 모든 async 함수에 try-catch 또는 .catch() 처리
- [ ] API 에러를 사용자에게 적절히 표시 (토스트, 에러 메시지)
- [ ] 로딩 상태 처리 (버튼 비활성화, 스피너)
- [ ] 네트워크 오류와 서버 오류를 구분하여 처리

### 보안
- [ ] 모든 API Route에 인증 검사 포함
- [ ] 사용자가 자신의 데이터만 접근 가능한지 검증
- [ ] 환경변수: 서버 전용 키가 NEXT_PUBLIC_ 아닌 것으로 확인
- [ ] 사용자 입력을 DB 쿼리에 직접 삽입하지 않음

### React 관행
- [ ] 훅은 최상위 레벨에서만 호출
- [ ] useEffect 의존성 배열 완전성 (eslint-plugin-react-hooks)
- [ ] 이벤트 핸들러에서 불필요한 useState 대신 useRef 사용
- [ ] 큰 컴포넌트를 200줄 이하로 분리

### 코드 품질
- [ ] 함수 하나의 책임이 하나인가 (단일 책임 원칙)
- [ ] 매직 넘버/문자열을 상수로 정의
- [ ] console.log 제거 (빌드 후 잔존하지 않음)
- [ ] TODO/FIXME 주석을 이슈로 등록하고 코드에서 제거

### 테스트 가능성
- [ ] 비즈니스 로직이 UI 컴포넌트에서 분리되어 있음
- [ ] 외부 의존성(Supabase, fetch)이 주입 가능하거나 모킹 가능
- [ ] 순수 함수는 테스트 파일이 존재
```

---

## 2. 리팩터링 루프 만들기

### 2.1 리팩터링 요청 프롬프트 예시

**에러 핸들링 추가**

```
아래 함수에 프로덕션 수준의 에러 핸들링을 추가해 주세요.

**현재 코드**:
```typescript
async function createRetro(data: RetroFormData) {
  const response = await fetch('/api/retros', {
    method: 'POST',
    body: JSON.stringify(data),
  });
  const result = await response.json();
  return result;
}
```

**추가할 사항**:
1. HTTP 상태 코드 검사 (response.ok)
2. 네트워크 오류와 API 오류 구분
3. 커스텀 에러 클래스 `ApiError` 정의 (status, message 포함)
4. 함수 반환 타입을 `Promise<Result<Retro, ApiError>>` 형태로 변경
   (Result 타입: { data: T, error: null } | { data: null, error: E })
5. 호출부에서 에러를 처리하는 예시 코드도 함께 작성

수정된 전체 코드와 사용 예시를 보여주세요.
```

**타입 강화**

```
아래 코드에서 `any`를 제거하고 완전한 타입 안전성을 확보해 주세요.

**현재 코드**:
```typescript
{여기에 코드 붙여넣기}
```

**요구사항**:
1. 모든 `any`를 구체적 타입으로 교체
2. 타입 단언(`as`)을 타입 가드 함수로 교체
3. API 응답은 Zod 스키마로 런타임 검증 추가
4. 변경된 타입과 함께 전체 파일을 다시 작성

Zod 스키마는 `src/lib/schemas.ts`에 분리하여 내보내세요.
```

**컴포넌트 분리**

```
아래 컴포넌트가 너무 많은 책임을 지고 있습니다.
단일 책임 원칙에 따라 분리해 주세요.

**현재 파일**: src/app/(dashboard)/retros/[id]/page.tsx

**현재 코드**:
```typescript
{여기에 코드 붙여넣기}
```

**분리 기준**:
1. 데이터 패칭 로직 → `src/hooks/useRetro.ts` 커스텀 훅
2. KPT 섹션 표시 → `src/components/retro/KptSection.tsx`
3. 액션 아이템 목록 → `src/components/retro/ActionItemList.tsx`
4. 공유 버튼 로직 → `src/components/retro/ShareButton.tsx`

각 파일의 전체 내용을 작성하고,
메인 페이지 파일이 어떻게 단순해지는지 보여주세요.
```

---

### 2.2 단계적 리팩터링 워크플로우

리팩터링은 한 번에 모두 하지 않는다. 단계를 나누면 각 단계에서 검증이 가능하고 롤백이 쉽다.

```
# 단계별 리팩터링 진행 순서 (Claude Code에 단계별로 요청)

## 1단계: 타입 추가 (기능 변경 없음)
목표: 런타임 동작을 바꾸지 않고 타입만 강화

프롬프트:
"src/lib/retros.ts의 모든 함수에 완전한 TypeScript 타입을 추가해 주세요.
로직은 변경하지 마세요. 타입만 추가합니다.
변경 후 `npx tsc --noEmit`이 오류 없이 통과해야 합니다."

검증:
$ npx tsc --noEmit
$ git diff --stat   # 타입만 변경되었는지 확인

## 2단계: 에러 핸들링 추가 (로직 일부 변경)
목표: try-catch 및 에러 상태 추가

프롬프트:
"src/lib/retros.ts의 모든 async 함수에 에러 핸들링을 추가해 주세요.
- Result 타입 패턴 사용 (성공/실패를 반환값으로 표현)
- throw 대신 return { error } 패턴
- 호출부(컴포넌트)도 새 패턴에 맞게 함께 수정"

검증:
$ npx tsc --noEmit
$ npm run build
$ # 브라우저에서 에러 시나리오 수동 테스트

## 3단계: 테스트 추가 (코드 변경 없음)
목표: 현재 동작을 테스트로 문서화

프롬프트:
"src/lib/retros.ts의 각 함수에 대한 단위 테스트를 작성해 주세요.
파일 위치: src/lib/__tests__/retros.test.ts
테스트 프레임워크: Vitest
Supabase 클라이언트는 vi.mock으로 모킹
성공 케이스 + 에러 케이스 각각 테스트"

검증:
$ npx vitest run src/lib/__tests__/retros.test.ts

## 4단계: 구조 개선 (선택적, 테스트 통과 후)
목표: 관심사 분리 및 모듈 경계 명확화

프롬프트:
"아래 파일들의 의존성 구조를 개선해 주세요.
현재: 컴포넌트가 Supabase 직접 호출
목표: 컴포넌트 → lib/retros.ts → Supabase
모든 Supabase 호출을 lib/ 레이어로 이동하고
컴포넌트는 lib 함수만 사용하도록 수정하세요."

검증:
$ grep -r "supabase" src/components/  # 결과 없어야 함
$ npx vitest run
$ npm run build
```

**리팩터링 루프 자동화 스크립트**:

```bash
#!/usr/bin/env bash
# refactor-loop.sh
# 리팩터링 각 단계를 안전하게 실행하고 검증

set -euo pipefail

STEP=$1
BRANCH="refactor/step-${STEP}-$(date +%Y%m%d-%H%M%S)"

echo "=== 리팩터링 Step ${STEP} 시작 ==="

# 현재 상태를 브랜치로 저장 (롤백 지점)
git checkout -b "$BRANCH"
echo "브랜치 생성: $BRANCH"

# 단계별 검증 함수
verify_step() {
  echo ""
  echo "--- 검증 시작 ---"

  echo "1. TypeScript 타입 검사..."
  if ! npx tsc --noEmit; then
    echo "TypeScript 오류 발생. 수정 후 재시도하세요."
    exit 1
  fi

  echo "2. ESLint 검사..."
  if ! npx next lint --dir src --quiet; then
    echo "ESLint 오류 발생."
    exit 1
  fi

  echo "3. 빌드 검사..."
  if ! npm run build; then
    echo "빌드 실패."
    exit 1
  fi

  echo "4. 테스트 실행..."
  if ls src/**/*.test.{ts,tsx} 2>/dev/null | head -1 | grep -q .; then
    npx vitest run
  else
    echo "테스트 파일 없음 (건너뜀)"
  fi

  echo "--- 검증 완료 ---"
  echo ""
}

case "$STEP" in
  "1")
    echo "Step 1: 타입 강화"
    echo "Claude Code에서 타입 추가 작업을 완료한 후 Enter를 누르세요..."
    read -r
    verify_step
    git add -A && git commit -m "refactor(step1): add TypeScript types"
    echo "Step 1 완료. 커밋 생성됨."
    ;;

  "2")
    echo "Step 2: 에러 핸들링 추가"
    echo "Claude Code에서 에러 핸들링 작업을 완료한 후 Enter를 누르세요..."
    read -r
    verify_step
    git add -A && git commit -m "refactor(step2): add error handling with Result pattern"
    echo "Step 2 완료."
    ;;

  "3")
    echo "Step 3: 테스트 추가"
    echo "Claude Code에서 테스트 작성을 완료한 후 Enter를 누르세요..."
    read -r
    npx vitest run
    git add -A && git commit -m "test: add unit tests for lib/retros"
    echo "Step 3 완료."
    ;;

  "4")
    echo "Step 4: 구조 개선"
    echo "Claude Code에서 구조 개선을 완료한 후 Enter를 누르세요..."
    read -r

    # 컴포넌트에 Supabase 직접 의존 없는지 확인
    DIRECT_IMPORTS=$(grep -r "supabase" src/components/ --include="*.tsx" --include="*.ts" | wc -l)
    if [ "$DIRECT_IMPORTS" -gt 0 ]; then
      echo "경고: 컴포넌트에 Supabase 직접 임포트 ${DIRECT_IMPORTS}곳 발견"
      grep -r "supabase" src/components/ --include="*.tsx" --include="*.ts" -n
    fi

    verify_step
    git add -A && git commit -m "refactor(step4): separate concerns, move DB calls to lib layer"
    echo "Step 4 완료."
    ;;

  *)
    echo "사용법: ./refactor-loop.sh [1|2|3|4]"
    exit 1
    ;;
esac

echo ""
echo "=== 리팩터링 Step ${STEP} 완료 ==="
echo "브랜치: $BRANCH"
echo "다음 단계: ./refactor-loop.sh $((STEP + 1))"
```

---

## 3. 기능 추가 요청을 받아 다시 계획 세우기

MVP를 출시한 후 기능 추가 요청이 들어온다. 기존 코드베이스에 안전하게 추가하는 패턴을 익혀야 한다.

### 3.1 Explore → Plan → Implement 패턴

**Explore 단계 - 현재 코드 이해**

```
새 기능을 추가하기 전에 현재 코드베이스를 분석해 주세요.

**추가할 기능**: 회고에 팀원이 이모지 반응(👍❤️😂)을 남길 수 있는 기능

**분석 요청**:
1. 현재 코드베이스의 파일 구조를 보고, 새 기능이 영향을 줄 파일을 모두 나열해 주세요.
2. 데이터 모델(Supabase 테이블 스키마)에서 어떤 변경이 필요한지 파악해 주세요.
3. 기존 컴포넌트 중 재사용 가능한 것을 찾아주세요.
4. 예상되는 기술적 위험이나 복잡도를 평가해 주세요.

**현재 프로젝트 구조**:
```
{tree -L 3 src/ 출력 붙여넣기}
```

**현재 데이터 모델** (Supabase 테이블):
```sql
{현재 테이블 스키마 붙여넣기}
```

탐색 결과를 바탕으로 Plan 단계에서 사용할 구현 계획을 제안해 주세요.
```

**Plan 단계 - 구현 계획 수립**

```
아래 탐색 결과를 바탕으로 이모지 반응 기능의 구현 계획을 작성해 주세요.

**탐색 결과**:
{Explore 단계 결과 붙여넣기}

**계획 요구사항**:
1. DB 마이그레이션 SQL 작성 (새 테이블 또는 컬럼)
2. 영향 받는 파일 목록과 각 파일에서 할 변경 요약
3. SPEC 단위로 분해 (각 SPEC은 독립적으로 실행 가능하게)
4. 기존 기능이 깨지지 않음을 보장하는 방법
5. 구현 순서 (의존성 고려)

**제약**:
- 기존 retros, retro_items 테이블 스키마는 변경 불가
- 공유 링크로 접근한 비로그인 사용자도 반응 추가 가능 (이름 없이)
- 실시간 업데이트는 MVP에서 제외

출력: 마크다운 계획서 + SPEC 목록
```

**Implement 단계 - 단계별 구현**

```
아래 SPEC을 구현해 주세요.

**SPEC-020: 이모지 반응 DB 마이그레이션**

```sql
-- 구현할 마이그레이션:
-- retro_reactions 테이블 생성
-- 컬럼: id, retro_id, emoji, user_id(nullable), created_at
-- RLS 정책: 누구나 조회 가능, 인증된 사용자만 삽입, 자신의 것만 삭제
```

Supabase 마이그레이션 파일로 작성해 주세요:
파일 경로: supabase/migrations/20240120000001_add_retro_reactions.sql

마이그레이션 파일 작성 후 로컬 테스트:
$ supabase db reset
$ supabase db push
```

---

### 3.2 SPEC 업데이트 프롬프트

기존 SPEC이 있을 때 기능 추가로 인해 업데이트가 필요한 경우:

```
기존 SPEC-006(회고 생성 폼)에 이모지 반응 기능을 추가해야 합니다.

**기존 SPEC-006 내용**:
{기존 SPEC 붙여넣기}

**추가 요구사항**:
- 회고 상세 페이지에서 각 KPT 항목 옆에 이모지 반응 버튼 표시
- 현재 반응 현황 표시 (👍 3, ❤️ 1 형식)
- 반응 추가/취소 토글 동작

**업데이트 요청**:
1. 기존 SPEC-006에서 변경이 필요한 부분을 명시
2. 새 SPEC-021(반응 UI 컴포넌트) 초안 작성
3. 두 SPEC 간 의존성 명시
4. 기존 RetroForm 컴포넌트가 영향을 받는지 확인

기존 SPEC은 최대한 보존하고, 변경이 불가피한 경우에만 수정 범위를 최소화하세요.
```

---

### 3.3 기능 추가 전 영향 분석 자동화

```bash
#!/usr/bin/env bash
# impact-analysis.sh
# 새 기능 추가 전 영향 받는 파일을 자동으로 분석
# 사용법: ./impact-analysis.sh "retro_reactions"

set -euo pipefail

FEATURE_KEYWORD="${1:-}"

if [ -z "$FEATURE_KEYWORD" ]; then
  echo "사용법: ./impact-analysis.sh <키워드>"
  echo "예시: ./impact-analysis.sh retro"
  exit 1
fi

echo "=== 영향 분석: '$FEATURE_KEYWORD' ==="
echo ""

echo "1. 관련 타입 정의:"
grep -r "interface\|type " src/types/ --include="*.ts" -l 2>/dev/null || echo "  없음"

echo ""
echo "2. 관련 기존 코드:"
grep -r "$FEATURE_KEYWORD" src/ --include="*.tsx" --include="*.ts" -l 2>/dev/null | head -20

echo ""
echo "3. 관련 API Routes:"
find src/app/api -name "route.ts" 2>/dev/null | xargs grep -l "$FEATURE_KEYWORD" 2>/dev/null || echo "  없음"

echo ""
echo "4. 관련 테스트 파일:"
find src -name "*.test.ts" -o -name "*.test.tsx" 2>/dev/null | xargs grep -l "$FEATURE_KEYWORD" 2>/dev/null || echo "  없음"

echo ""
echo "5. 현재 파일 수 통계:"
echo "  컴포넌트: $(find src/components -name "*.tsx" 2>/dev/null | wc -l)개"
echo "  페이지: $(find src/app -name "page.tsx" 2>/dev/null | wc -l)개"
echo "  API Routes: $(find src/app/api -name "route.ts" 2>/dev/null | wc -l)개"
echo "  훅: $(find src/hooks -name "*.ts" 2>/dev/null | wc -l)개"
echo "  lib: $(find src/lib -name "*.ts" 2>/dev/null | wc -l)개"

echo ""
echo "6. 최근 변경 파일 (git):"
git log --oneline -5
```

---

## 4. 프로젝트 회고: 무엇을 맡기고 무엇을 직접 할 것인가

MVP를 완성한 후 돌아보면, AI에게 맡긴 것 중 잘 된 것과 직접 해야 했던 것이 보인다. 이 패턴을 정리하면 다음 프로젝트가 빨라진다.

### 4.1 AI 위임 적합도 매트릭스

아래 표는 작업 유형별로 AI 위임이 얼마나 적합한지 정리한 것이다.

| 작업 유형 | AI 위임 적합도 | 이유 | 권장 방식 |
|-----------|---------------|------|-----------|
| 보일러플레이트 코드 (CRUD API, 폼 컴포넌트) | **매우 높음** | 패턴이 명확, 컨텍스트가 단순 | Codex full-auto |
| UI 컴포넌트 (shadcn/ui 기반) | **매우 높음** | 컴포넌트 라이브러리가 정해져 있음 | Claude Code |
| TypeScript 타입 정의 | **높음** | 구조가 명확하면 잘 생성 | Claude Code |
| 단위 테스트 작성 | **높음** | 기존 코드 기반, 패턴 반복 | Claude Code |
| DB 마이그레이션 SQL | **높음** | 요구사항이 명확한 경우 | Claude Code |
| 에러 핸들링 추가 | **높음** | 기존 패턴을 적용하는 작업 | Claude Code |
| 리팩터링 (단순 분리) | **보통** | 컨텍스트 제공이 중요 | Claude Code + 검토 필수 |
| 아키텍처 결정 | **낮음** | 시스템 전체 맥락 이해 필요 | 직접 (AI 자문 가능) |
| 보안 정책 설계 | **낮음** | 비즈니스 요구사항과 결합 | 직접 (AI 검토 보조) |
| 데이터 모델 설계 | **낮음** | 도메인 이해와 장기 설계 필요 | 직접 (AI 자문 가능) |
| 성능 최적화 (복잡) | **낮음** | 프로파일링 + 실제 병목 이해 필요 | 직접 |
| 비즈니스 로직 (핵심) | **낮음** | 명세가 모호하거나 예외 많음 | 직접 |
| 외부 API 연동 (복잡) | **보통** | 공식 문서와 대조 필요 | Claude Code + 검토 필수 |
| 배포 설정 (첫 설정) | **보통** | 환경마다 다름, 디버깅 어려움 | 직접 |
| 문서 작성 (내부) | **높음** | 코드 기반 문서화 | Claude Code |

---

### 4.2 위임 결정 가이드라인

**위임해도 되는 조건 (3개 이상 해당 시)**

```
체크리스트:
[ ] 요구사항을 100단어 이내로 명확히 설명할 수 있다
[ ] 성공/실패 기준을 객관적으로 정의할 수 있다
[ ] 생성된 코드를 30분 안에 검토할 수 있다
[ ] 틀려도 되돌리기 쉽다 (git revert 가능)
[ ] 기존 패턴이 있어 참고할 수 있다
[ ] 테스트로 정확성을 검증할 수 있다
```

**직접 해야 하는 조건 (1개라도 해당 시)**

```
체크리스트:
[ ] 요구사항이 여러 해석 가능하고 비즈니스 판단이 필요하다
[ ] 잘못되면 보안 취약점이나 데이터 손실로 이어진다
[ ] 시스템 전체 구조를 바꾸는 결정이다
[ ] 실제 사용자 데이터를 조작하거나 삭제한다
[ ] 외부 서비스 계약/비용과 관련된 설정이다
```

---

### 4.3 실제 프로젝트 회고 템플릿

이 템플릿을 채워 다음 프로젝트에서 반복 실수를 줄인다.

```markdown
# 프로젝트 회고: {프로젝트명}

**완료일**: {날짜}
**총 소요 시간**: {시간}
**AI 도구**: Claude Code, Codex

---

## 1. AI 위임 성공 사례

| 작업 | 도구 | 절약된 시간 | 품질 |
|------|------|------------|------|
| RetroForm 컴포넌트 초안 | Claude Code | 2시간 | 90% 완성, 타입 보완 필요 |
| CRUD API 5개 | Codex | 4시간 | 80% 완성, 에러 핸들링 추가 |
| 단위 테스트 15개 | Claude Code | 3시간 | 95% 완성, mock 수정 필요 |

## 2. AI 위임 실패 사례 (직접 다시 해야 했던 것)

| 작업 | 실패 이유 | 교훈 |
|------|---------|------|
| Supabase RLS 정책 | 비즈니스 규칙 이해 부족 | 데이터 접근 정책은 직접 설계 |
| 실시간 구독 로직 | 상태 관리 복잡도 과소평가 | 복잡한 비동기는 먼저 PoC 직접 작성 |

## 3. 다음 프로젝트를 위한 개선

### 프롬프트 개선
- [ ] SPEC에 "영향 받는 파일" 섹션 추가 (컨텍스트 명확화)
- [ ] 완료 기준에 TypeScript 타입 오류 0개 명시
- [ ] 컴포넌트 프롬프트에 반응형 브레이크포인트 항상 포함

### 워크플로우 개선
- [ ] Codex 실행 전 타입 파일 먼저 확정 (의존성 해결)
- [ ] 병렬 실행 후 verify-specs.sh 자동 실행 추가
- [ ] 리팩터링은 기능 개발 직후, 다음 기능 전에 완료

### 위임 전략 개선
- [ ] 데이터 모델 설계는 위임하지 않음 (30분 직접)
- [ ] 보안 관련 코드는 위임 후 반드시 패턴 2(보안 리뷰 프롬프트) 실행
- [ ] 테스트는 구현 직후 바로 위임 (나중으로 미루지 않음)

## 4. 속도 측정

| 단계 | 예상 시간 | 실제 시간 | AI 기여도 |
|------|---------|---------|---------|
| PRD 작성 | 4시간 | 1시간 | 75% |
| SPEC 분해 | 2시간 | 30분 | 85% |
| M1 골격 구현 | 8시간 | 3시간 | 70% |
| M2 핵심 기능 | 16시간 | 6시간 | 65% |
| M3 마무리 | 8시간 | 4시간 | 50% |
| **합계** | **38시간** | **14.5시간** | **약 62%** |
```

---

### 4.4 위임 결정 순서도

실제 작업이 들어왔을 때 아래 순서로 판단한다.

```
새 작업 도착
    │
    ▼
요구사항을 명확히 설명할 수 있는가?
    │ NO  → 먼저 직접 명세를 명확화 (30분)
    │ YES
    ▼
보안/데이터 손실 위험이 있는가?
    │ YES → 직접 구현 (AI는 검토 보조)
    │ NO
    ▼
기존 코드베이스 패턴이 있는가?
    │ YES → AI에게 위임 (기존 파일을 컨텍스트로 제공)
    │ NO
    ▼
유사한 오픈소스 예시가 있는가?
    │ YES → AI에게 위임 (예시 URL 프롬프트에 포함)
    │ NO
    ▼
작업 크기가 4시간 이내인가?
    │ YES → AI에게 위임 (명세를 최대한 구체화)
    │ NO  → SPEC으로 더 작게 분해 후 반복
```

---

### 4.5 팀 위임 가이드라인 (협업 환경)

팀으로 일할 때 AI 위임을 일관되게 유지하는 규칙이다.

```markdown
## 팀 AI 위임 규칙 (CONTRIBUTING.md에 추가)

### AI 생성 코드 병합 요건
1. `npx tsc --noEmit` 오류 0개
2. `npm run build` 성공
3. 변경된 파일의 ESLint 오류 0개
4. AI 생성 코드임을 PR 설명에 명시
5. 보안 관련 변경사항은 반드시 패턴 2(보안 리뷰) 실행 후 결과 첨부

### AI에게 절대 위임하지 않는 것
- 프로덕션 DB 마이그레이션 실행 (로컬 테스트 후 직접 실행)
- 환경변수 및 시크릿 값 결정
- 외부 서비스 API 키 발급 및 권한 설정
- 사용자 데이터를 삭제하는 스크립트

### AI 위임 시 컨텍스트 제공 표준
- 관련 타입 파일 항상 포함
- 영향 받는 기존 파일의 현재 내용 포함
- 완료 기준 체크리스트 포함
- 기술 스택 버전 명시 (Next.js 14, Supabase v2 등)
```

---

*이 문서는 커리큘럼 06-CH02에 대응합니다. 이전 단계는 [project-zero-to-one-kr.md](./project-zero-to-one-kr.md)를 참고하세요.*
