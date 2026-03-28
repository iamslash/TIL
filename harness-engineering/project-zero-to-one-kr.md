# 프로젝트 1 - 0 to 1: 아이디어에서 MVP까지

> 커리큘럼 06-CH01 대응 문서

## 목차

- [1. PRD 작성과 MVP 범위 결정](#1-prd-작성과-mvp-범위-결정)
- [2. ROADMAP과 SPEC으로 작업 분해](#2-roadmap과-spec으로-작업-분해)
- [3. Claude Code로 화면 골격과 컴포넌트 만들기](#3-claude-code로-화면-골격과-컴포넌트-만들기)
- [4. Codex로 병렬 구현 및 자동 검증](#4-codex로-병렬-구현-및-자동-검증)

---

## 1. PRD 작성과 MVP 범위 결정

### 1.1 PRD 템플릿 전문

아래는 AI 보조 개발에 최적화된 PRD(Product Requirements Document) 템플릿이다. Claude Code에 그대로 붙여넣어 초안을 채울 수 있다.

```markdown
# PRD: {제품명}

**버전**: 0.1
**작성일**: {날짜}
**작성자**: {이름}
**상태**: Draft / Review / Approved

---

## 1. 문제 정의 (Problem Statement)

**핵심 문제**:
> 한 문장으로 작성. 예: "소규모 팀이 스프린트 회고를 체계적으로 기록하고 공유하지 못해 반복 실수가 줄지 않는다."

**영향 받는 사용자**:
- 주요 페르소나: {이름, 역할, 핵심 불편}
- 부차 페르소나: {이름, 역할, 핵심 불편}

**현재 해결 방식과 한계**:
| 현재 방법 | 한계 |
|----------|------|
| Notion 수동 정리 | 포맷 제각각, 검색 불가 |
| Slack 쓰레드 | 유실, 맥락 단절 |

---

## 2. 목표 (Goals)

**비즈니스 목표**:
- [ ] {측정 가능한 목표 1} — 성공 지표: {KPI}
- [ ] {측정 가능한 목표 2} — 성공 지표: {KPI}

**사용자 목표**:
- [ ] {사용자가 달성하려는 결과 1}
- [ ] {사용자가 달성하려는 결과 2}

**비목표 (Non-goals)**:
- {MVP에서 의도적으로 제외하는 것}
- {이유: 왜 지금은 아닌가}

---

## 3. 기능 요구사항 (Functional Requirements)

### 3.1 핵심 기능 (Must-have)
| ID | 기능 | 설명 | 우선순위 |
|----|------|------|----------|
| F01 | 회고 생성 | 템플릿 선택 후 회고 항목 입력 | P0 |
| F02 | 회고 공유 | 링크 또는 이메일로 팀에 공유 | P0 |
| F03 | 액션 아이템 추적 | 다음 스프린트까지 완료 여부 체크 | P1 |

### 3.2 선택 기능 (Nice-to-have)
| ID | 기능 | 설명 | 이유 |
|----|------|------|------|
| F10 | AI 요약 | 회고 내용 자동 요약 | 추후 검토 |

---

## 4. 비기능 요구사항 (Non-functional Requirements)

- **성능**: 페이지 초기 로드 < 2초 (LCP 기준)
- **보안**: 팀 외부 접근 차단, 링크 만료 기능
- **접근성**: WCAG 2.1 AA 준수
- **브라우저 지원**: Chrome, Safari, Firefox 최신 2버전

---

## 5. 사용자 여정 (User Journey)

```
[랜딩] → [로그인/가입] → [대시보드] → [회고 생성] → [작성] → [공유] → [액션 추적]
```

각 단계별 주요 행동:
1. **랜딩**: CTA 클릭 → 가입 전환
2. **가입**: 이메일 + 팀 이름 입력
3. **회고 생성**: 템플릿 선택 → 스프린트 번호 입력
4. **작성**: KPT / 4Ls / Mad-Sad-Glad 형식
5. **공유**: 팀 슬랙 채널 또는 이메일 전송
6. **액션 추적**: 담당자 지정, 마감일 설정

---

## 6. 기술 스택 제안

| 레이어 | 선택 | 근거 |
|--------|------|------|
| Frontend | Next.js 14 (App Router) | SSR, 파일 기반 라우팅 |
| Backend | Next.js Route Handlers | 초기 단순성 |
| DB | Supabase (PostgreSQL) | 인증 내장, 빠른 셋업 |
| 배포 | Vercel | Next.js 최적화 |
| 스타일 | Tailwind CSS + shadcn/ui | 빠른 컴포넌트 조립 |

---

## 7. 마일스톤

| 단계 | 목표 | 기간 |
|------|------|------|
| M1 - 골격 | 화면 라우팅 + 인증 | 1주 |
| M2 - 핵심 기능 | 회고 CRUD + 공유 | 2주 |
| M3 - MVP 완성 | 액션 추적 + 배포 | 1주 |

---

## 8. 열린 질문 (Open Questions)

- [ ] 팀 초대 방식: 이메일 초대 vs. 링크 공유?
- [ ] 회고 데이터 보존 기간은?
- [ ] 무료/유료 플랜 구분이 필요한가?

---

## 9. 참고 자료

- 경쟁사 분석: {URL}
- 사용자 인터뷰 요약: {URL}
- 디자인 목업: {Figma URL}
```

---

### 1.2 Claude Code에게 PRD 작성 맡기는 프롬프트 예시

아래 프롬프트를 Claude Code 대화창에 그대로 붙여넣으면 PRD 초안을 받을 수 있다.

**프롬프트 A - 아이디어 설명 후 PRD 생성**

```
당신은 시니어 Product Manager입니다.

아래 아이디어를 바탕으로 PRD 초안을 작성해 주세요.

**아이디어**: 소규모 개발팀(3~10명)이 스프린트 회고를 구조화된 형식으로 기록하고,
팀원과 손쉽게 공유하며, 도출된 액션 아이템을 다음 스프린트까지 추적할 수 있는 웹 앱.

**작성 지침**:
1. 문제 정의 섹션에서 핵심 고통점을 한 문장으로 날카롭게 표현하세요.
2. 기능 요구사항은 P0(없으면 출시 불가) / P1(중요하지만 이후 추가 가능)으로 분류하세요.
3. MVP는 2주 안에 혼자 구현 가능한 수준으로 제한하세요.
4. 비목표(Non-goals)를 명시적으로 적어 범위 팽창을 막으세요.
5. 기술 스택은 Next.js + Supabase + Vercel 기준으로 제안하세요.

출력 형식: 마크다운, 위의 PRD 템플릿 구조 준수.
```

**프롬프트 B - 기존 PRD 검토 및 개선**

```
아래 PRD 초안을 검토하고 다음 기준으로 피드백 및 개선안을 제시하세요.

**검토 기준**:
1. 문제가 충분히 구체적인가? (모호한 표현 지적)
2. 성공 지표(KPI)가 측정 가능한가?
3. 비목표가 명시되어 있는가?
4. MVP 범위가 2주 1인 개발 기준으로 적절한가?
5. 기술적 위험 요소가 빠진 것은 없는가?

**PRD 초안**:
{여기에 PRD 내용 붙여넣기}

출력: 섹션별 평점(1-5) + 구체적 개선 제안.
```

---

### 1.3 MVP 범위 결정 매트릭스

아이디어가 많을 때 MVP에 넣을지 뺄지 판단하는 기준표다.

| 기능 | 사용자 가치 (1-5) | 구현 복잡도 (1-5, 높을수록 어려움) | MVP 포함 여부 |
|------|------------------|-----------------------------------|---------------|
| 회고 생성 (KPT 형식) | 5 | 2 | **Yes** |
| 팀원 초대 (이메일) | 4 | 3 | **Yes** |
| AI 자동 요약 | 3 | 5 | No |
| 슬랙 알림 연동 | 3 | 4 | No |
| 다국어 지원 | 2 | 4 | No |
| 공유 링크 생성 | 5 | 1 | **Yes** |
| 액션 아이템 추적 | 4 | 2 | **Yes** |
| 대시보드 차트 | 2 | 4 | No |

**판단 규칙**:
- 사용자 가치 >= 4 AND 구현 복잡도 <= 3 → MVP 포함
- 사용자 가치 >= 5 AND 구현 복잡도 <= 4 → MVP 포함 검토
- 나머지 → 백로그로 이동, ROADMAP의 M2 이후로 예약

**Claude Code에게 매트릭스 채우기 요청하는 프롬프트**:

```
다음 기능 목록에 대해 MVP 범위 결정 매트릭스를 채워주세요.

**기능 목록**:
{기능1, 기능2, 기능3 ...}

**평가 기준**:
- 사용자 가치: 이 기능 없이 핵심 문제가 해결되는가? (1=없어도 됨, 5=없으면 제품이 아님)
- 구현 복잡도: Next.js + Supabase 스택 기준 1인 개발 시간 (1=반나절, 5=1주+)

마크다운 표 형식으로 출력하고, MVP 포함 여부에 대한 한 줄 근거를 함께 작성하세요.
```

---

## 2. ROADMAP과 SPEC으로 작업 분해

### 2.1 ROADMAP.md 예시

```markdown
# ROADMAP - RetroSync

> 마지막 업데이트: 2024-01-15
> 현재 상태: M1 진행 중

## 비전
소규모 팀이 회고를 습관으로 만들 수 있도록 마찰을 제거한다.

---

## M1 - 골격 (1주차, 1/15 ~ 1/21)

**목표**: 사용자가 가입하고 빈 대시보드를 볼 수 있다.

- [ ] SPEC-001: Next.js 프로젝트 초기화 + 기본 레이아웃
- [ ] SPEC-002: Supabase 프로젝트 생성 + 인증 설정 (이메일/비밀번호)
- [ ] SPEC-003: 로그인 / 회원가입 페이지 UI
- [ ] SPEC-004: 인증 미들웨어 + 보호된 라우트
- [ ] SPEC-005: 기본 대시보드 페이지 (빈 상태 UI 포함)

**완료 기준**: Vercel에 배포된 URL에서 가입 → 로그인 → 대시보드 진입 가능

---

## M2 - 핵심 기능 (2-3주차, 1/22 ~ 2/4)

**목표**: 회고를 만들고 팀과 공유할 수 있다.

- [ ] SPEC-006: 회고 생성 폼 (KPT 템플릿)
- [ ] SPEC-007: 회고 저장 API (Supabase)
- [ ] SPEC-008: 회고 목록 페이지
- [ ] SPEC-009: 회고 상세 페이지
- [ ] SPEC-010: 공유 링크 생성 (비인증 접근 가능한 공개 URL)
- [ ] SPEC-011: 팀 생성 + 멤버 초대 (이메일)

**완료 기준**: 회고 생성 → 링크 복사 → 비로그인 사용자가 내용 열람 가능

---

## M3 - MVP 완성 (4주차, 2/5 ~ 2/11)

**목표**: 액션 아이템을 추적하고 프로덕션 준비를 마친다.

- [ ] SPEC-012: 액션 아이템 CRUD
- [ ] SPEC-013: 담당자 지정 + 마감일 설정
- [ ] SPEC-014: 완료 체크 기능
- [ ] SPEC-015: 에러 바운더리 + 로딩 상태 전체 적용
- [ ] SPEC-016: 기본 SEO + OG 태그
- [ ] SPEC-017: 프로덕션 환경변수 설정 + 최종 배포

**완료 기준**: 실제 팀이 1회 회고를 완주할 수 있다.

---

## 백로그 (M4 이후)

- AI 회고 요약
- Slack 알림 연동
- 대시보드 차트 (팀 건강도 트렌드)
- 다국어 지원
```

---

### 2.2 작업 단위 SPEC 예시

각 SPEC은 Claude Code 또는 Codex가 독립적으로 실행할 수 있는 최소 단위로 작성한다.

```markdown
# SPEC-006: 회고 생성 폼 (KPT 템플릿)

**마일스톤**: M2
**예상 소요**: 2-3시간
**의존성**: SPEC-003 (UI 컴포넌트), SPEC-002 (Supabase 클라이언트)

## 목표
사용자가 KPT(Keep / Problem / Try) 형식으로 회고를 작성하고 저장할 수 있다.

## 구현 상세

### 파일 경로
- `app/retros/new/page.tsx` — 회고 생성 페이지
- `components/retro/RetroForm.tsx` — KPT 폼 컴포넌트
- `lib/retros.ts` — 회고 저장 함수

### UI 요구사항
- 스프린트 번호 입력 필드 (숫자)
- Keep 섹션: 텍스트 영역 + 항목 추가 버튼 (동적 목록)
- Problem 섹션: 동일 패턴
- Try 섹션: 동일 패턴
- 저장 버튼 (로딩 상태 포함)
- 취소 버튼 → 대시보드로 이동

### 데이터 구조
```typescript
interface RetroItem {
  id: string;
  content: string;
}

interface RetroFormData {
  sprintNumber: number;
  keep: RetroItem[];
  problem: RetroItem[];
  try: RetroItem[];
}
```

### 완료 기준
- [ ] 폼 제출 시 Supabase `retros` 테이블에 저장
- [ ] 저장 성공 시 해당 회고 상세 페이지로 리다이렉트
- [ ] 저장 실패 시 에러 토스트 표시
- [ ] 각 섹션에 최소 1개 항목 없으면 제출 불가 (유효성 검사)
- [ ] 모바일(375px) 반응형 레이아웃
```

---

### 2.3 SPEC 분해 프롬프트

```
당신은 시니어 풀스택 개발자입니다.

아래 ROADMAP 마일스톤을 Claude Code / Codex가 독립적으로 실행 가능한
최소 작업 단위(SPEC)로 분해해 주세요.

**ROADMAP 마일스톤**:
{여기에 마일스톤 내용 붙여넣기}

**SPEC 분해 규칙**:
1. 각 SPEC은 1명이 2-4시간 안에 완료 가능한 크기로 작성
2. 다른 SPEC과 의존성이 최소화되도록 설계 (병렬 실행 가능한 것을 최대화)
3. 각 SPEC에 포함할 내용:
   - 목표 (한 문장)
   - 영향 받는 파일 목록
   - UI/로직 상세 요구사항
   - TypeScript 인터페이스 (필요시)
   - 완료 기준 체크리스트
4. 기술 스택: Next.js 14 App Router + Supabase + Tailwind CSS + shadcn/ui

**출력**: SPEC-XXX 형식으로 번호를 붙여 마크다운으로 작성
```

---

## 3. Claude Code로 화면 골격과 컴포넌트 만들기

### 3.1 Next.js 프로젝트 초기 셋업 프롬프트

**1단계 - 프로젝트 생성**

```
새 Next.js 14 프로젝트를 아래 스펙으로 초기화해 주세요.

**프로젝트명**: retro-sync
**기술 스택**:
- Next.js 14 (App Router, TypeScript)
- Tailwind CSS
- shadcn/ui (초기 컴포넌트: Button, Input, Textarea, Card, Toast)
- Supabase JS 클라이언트

**실행할 명령어 순서**:
1. `npx create-next-app@latest retro-sync --typescript --tailwind --app --src-dir`
2. shadcn/ui 초기화: `npx shadcn-ui@latest init`
3. 필요한 shadcn 컴포넌트 추가
4. Supabase 클라이언트 설치: `npm install @supabase/supabase-js @supabase/ssr`

**초기 폴더 구조 생성**:
```
src/
├── app/
│   ├── (auth)/
│   │   ├── login/page.tsx
│   │   └── signup/page.tsx
│   ├── (dashboard)/
│   │   ├── dashboard/page.tsx
│   │   └── retros/
│   │       ├── new/page.tsx
│   │       └── [id]/page.tsx
│   ├── share/[token]/page.tsx
│   └── layout.tsx
├── components/
│   ├── ui/          (shadcn 자동 생성)
│   ├── auth/
│   ├── retro/
│   └── layout/
├── lib/
│   ├── supabase/
│   │   ├── client.ts
│   │   └── server.ts
│   └── utils.ts
└── types/
    └── index.ts
```

각 파일에 기본 보일러플레이트를 작성하고,
`src/types/index.ts`에 공통 타입을 정의해 주세요.
```

**2단계 - 레이아웃 및 네비게이션 셋업**

```
아래 요구사항으로 앱 레이아웃과 네비게이션을 구현해 주세요.

**파일**: `src/app/(dashboard)/layout.tsx`

**요구사항**:
- 좌측 사이드바 (데스크톱) / 하단 탭바 (모바일) 네비게이션
- 네비게이션 항목: 대시보드, 회고 목록, 새 회고
- 현재 경로에 따라 활성 상태 표시 (usePathname 사용)
- 사용자 아바타 + 이름 표시 (우상단 또는 사이드바 하단)
- 로그아웃 버튼

**디자인 참고**:
- 색상: 중립 회색 배경, 인디고 포인트 컬러
- 스타일: 미니멀, 화이트 카드 기반

현재 파일 내용:
{파일 내용 붙여넣기 또는 "파일이 없습니다"}
```

---

### 3.2 컴포넌트 생성 프롬프트 패턴

**패턴 1 - 신규 컴포넌트 생성**

```
`src/components/retro/RetroCard.tsx` 컴포넌트를 만들어 주세요.

**역할**: 회고 목록 페이지에서 각 회고를 카드 형태로 표시

**Props**:
```typescript
interface RetroCardProps {
  id: string;
  sprintNumber: number;
  createdAt: string;
  keepCount: number;
  problemCount: number;
  tryCount: number;
  authorName: string;
}
```

**UI 요구사항**:
- shadcn Card 컴포넌트 사용
- 스프린트 번호를 큰 글씨로 (예: "Sprint #12")
- KPT 각 섹션 항목 수를 배지로 표시
- 작성일 (상대 시간: "3일 전")
- 카드 클릭 시 `/retros/{id}`로 이동
- 호버 시 그림자 효과

**제약**:
- 'use client' 불필요 (서버 컴포넌트로 작성)
- date-fns 라이브러리로 날짜 포맷
```

**패턴 2 - 기존 컴포넌트에 기능 추가**

```
`src/components/retro/RetroForm.tsx`에 아래 기능을 추가해 주세요.

**현재 파일**:
{현재 파일 내용 전체}

**추가할 기능**:
1. 자동 저장 (30초마다 localStorage에 임시 저장)
2. 페이지 진입 시 localStorage에 임시 저장된 내용이 있으면 복원 여부 토스트로 확인
3. 저장 성공 후 localStorage 초기화

**제약**:
- 기존 폼 로직(useForm, validation)을 건드리지 마세요
- 자동 저장 로직을 커스텀 훅 `useAutoSave`로 분리해 `src/hooks/useAutoSave.ts`에 저장
- 토스트는 shadcn/ui Toast 사용
```

**패턴 3 - 페이지 전체 구현**

```
`src/app/(dashboard)/dashboard/page.tsx`를 구현해 주세요.

**이 페이지의 역할**: 사용자의 최근 회고 목록과 빠른 액션을 보여주는 홈

**데이터 패칭**:
- Supabase 서버 클라이언트로 현재 사용자의 회고 최신 5개 조회
- `src/lib/supabase/server.ts`의 `createServerClient` 사용

**UI 섹션**:
1. 환영 메시지 (사용자 이름 포함)
2. "새 회고 시작" CTA 버튼 (눈에 띄는 크기)
3. 최근 회고 카드 목록 (RetroCard 컴포넌트 사용)
4. 회고가 없을 때 빈 상태 UI ("첫 회고를 시작해보세요" 일러스트 + CTA)

**서버 컴포넌트로 작성** (async/await 사용, useEffect 없음)
```

---

## 4. Codex로 병렬 구현 및 자동 검증

### 4.1 Codex exec + full-auto 기본 사용법

```bash
# 단일 SPEC 실행 (비대화형 자동 실행)
codex exec --full-auto "SPEC-006을 구현해 주세요: 회고 생성 폼 (KPT 템플릿)

요구사항:
- 파일: src/components/retro/RetroForm.tsx
- KPT 형식 (Keep/Problem/Try) 동적 목록 입력
- shadcn/ui Card, Button, Input, Textarea 사용
- react-hook-form + zod 유효성 검사
- 저장 시 /api/retros POST 호출
- 완료 후 npm run verify 실행"

# 작업 루트를 명시하고 추가 디렉토리 쓰기 권한 부여
codex exec --full-auto \
  -C /Users/yourname/myapp \
  --add-dir /Users/yourname/shared-packages \
  "
  다음 파일을 기준으로 작업해 주세요:
  - src/types/index.ts
  - src/lib/supabase/client.ts
  - src/components/retro/RetroForm.tsx

  RetroForm 컴포넌트를 구현하고 src/components/retro/RetroForm.tsx에 저장해 주세요.
  완료 후 npm run verify를 실행해 주세요.
  "
```

---

### 4.2 여러 터미널에서 Codex 병렬 실행 스크립트

여러 SPEC을 동시에 실행하여 개발 속도를 높이는 스크립트다.

```bash
#!/usr/bin/env bash
# run-parallel-specs.sh
# 사용법: ./run-parallel-specs.sh

set -euo pipefail

PROJECT_DIR="$(pwd)"
LOG_DIR="$PROJECT_DIR/.codex-logs"
mkdir -p "$LOG_DIR"

# 실행할 SPEC 정의 (독립적인 것만 병렬 실행)
declare -A SPECS=(
  ["SPEC-006"]="src/components/retro/RetroForm.tsx를 구현하세요. KPT 형식 동적 목록, react-hook-form + zod 유효성 검사, shadcn/ui 컴포넌트 사용."
  ["SPEC-008"]="src/app/(dashboard)/retros/page.tsx를 구현하세요. 회고 목록 페이지, 서버 컴포넌트, Supabase에서 데이터 조회, RetroCard 컴포넌트 사용."
  ["SPEC-010"]="src/app/api/retros/share/route.ts를 구현하세요. 공유 링크 생성 API, UUID 토큰 발급, Supabase retro_shares 테이블에 저장."
)

PIDS=()
SPEC_NAMES=()

echo "=== Codex 병렬 실행 시작: $(date) ==="
echo "실행할 SPEC 수: ${#SPECS[@]}"
echo ""

for spec_id in "${!SPECS[@]}"; do
  prompt="${SPECS[$spec_id]}"
  log_file="$LOG_DIR/${spec_id}.log"

  echo "시작: $spec_id"

  # 각 SPEC을 백그라운드로 실행
  codex exec --full-auto "$prompt" \
    > "$log_file" 2>&1 &

  PIDS+=($!)
  SPEC_NAMES+=("$spec_id")
done

echo ""
echo "모든 SPEC 실행 중... (PID: ${PIDS[*]})"
echo ""

# 완료 대기 및 결과 수집
FAILED=()
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  spec="${SPEC_NAMES[$i]}"

  if wait "$pid"; then
    echo "완료: $spec"
  else
    echo "실패: $spec (로그: $LOG_DIR/${spec}.log)"
    FAILED+=("$spec")
  fi
done

echo ""
echo "=== 실행 완료: $(date) ==="

if [ ${#FAILED[@]} -eq 0 ]; then
  echo "모든 SPEC 성공적으로 완료"
else
  echo "실패한 SPEC: ${FAILED[*]}"
  echo "로그를 확인하세요: $LOG_DIR/"
  exit 1
fi
```

**tmux로 각 SPEC을 별도 창에서 실시간 모니터링하는 스크립트**:

```bash
#!/usr/bin/env bash
# run-specs-tmux.sh
# 사용법: ./run-specs-tmux.sh
# 요구사항: tmux 설치 필요

SESSION="codex-parallel"

# 기존 세션 제거
tmux kill-session -t "$SESSION" 2>/dev/null || true

# 새 세션 생성
tmux new-session -d -s "$SESSION" -n "SPEC-006"

# SPEC-006: RetroForm
tmux send-keys -t "$SESSION:SPEC-006" \
  "codex exec --full-auto 'RetroForm 컴포넌트를 src/components/retro/RetroForm.tsx에 구현: KPT 동적 목록, react-hook-form, shadcn/ui'" \
  Enter

# SPEC-008: 회고 목록 페이지
tmux new-window -t "$SESSION" -n "SPEC-008"
tmux send-keys -t "$SESSION:SPEC-008" \
  "codex exec --full-auto '회고 목록 페이지를 src/app/(dashboard)/retros/page.tsx에 구현: 서버 컴포넌트, Supabase 조회, RetroCard 사용'" \
  Enter

# SPEC-010: 공유 링크 API
tmux new-window -t "$SESSION" -n "SPEC-010"
tmux send-keys -t "$SESSION:SPEC-010" \
  "codex exec --full-auto '공유 링크 API를 src/app/api/retros/share/route.ts에 구현: UUID 토큰, Supabase retro_shares 테이블'" \
  Enter

# 첫 번째 창으로 포커스
tmux select-window -t "$SESSION:SPEC-006"

# tmux 세션 연결
echo "tmux 세션 '$SESSION'에 연결합니다..."
echo "창 전환: Ctrl+B + 숫자  |  종료: Ctrl+B + d"
tmux attach-session -t "$SESSION"
```

---

### 4.3 자동 검증 스크립트

Codex가 생성한 코드를 자동으로 검증하는 스크립트다.

```bash
#!/usr/bin/env bash
# verify-specs.sh
# Codex 작업 완료 후 실행하여 품질 검증
# 사용법: ./verify-specs.sh [SPEC-ID]

set -euo pipefail

PROJECT_DIR="$(pwd)"
SPEC_ID="${1:-all}"
ERRORS=0
WARNINGS=0

print_header() {
  echo ""
  echo "=========================================="
  echo " $1"
  echo "=========================================="
}

print_check() {
  local status=$1
  local message=$2
  if [ "$status" = "pass" ]; then
    echo "  PASS  $message"
  elif [ "$status" = "fail" ]; then
    echo "  FAIL  $message"
    ERRORS=$((ERRORS + 1))
  else
    echo "  WARN  $message"
    WARNINGS=$((WARNINGS + 1))
  fi
}

# 1. TypeScript 타입 검사
print_header "1. TypeScript 타입 검사"
if npx tsc --noEmit 2>/dev/null; then
  print_check "pass" "TypeScript 타입 오류 없음"
else
  print_check "fail" "TypeScript 타입 오류 발견"
  npx tsc --noEmit 2>&1 | head -20
fi

# 2. ESLint 검사
print_header "2. ESLint 정적 분석"
if npx next lint --dir src 2>/dev/null; then
  print_check "pass" "ESLint 오류 없음"
else
  print_check "fail" "ESLint 오류 발견"
fi

# 3. 빌드 검사
print_header "3. Next.js 빌드 검사"
if npm run build 2>/dev/null; then
  print_check "pass" "빌드 성공"
else
  print_check "fail" "빌드 실패"
  npm run build 2>&1 | tail -20
fi

# 4. 파일 존재 검사 (예상 파일이 실제로 생성되었는지)
print_header "4. 예상 파일 생성 검사"
EXPECTED_FILES=(
  "src/components/retro/RetroForm.tsx"
  "src/app/(dashboard)/retros/page.tsx"
  "src/app/api/retros/share/route.ts"
)

for file in "${EXPECTED_FILES[@]}"; do
  if [ -f "$PROJECT_DIR/$file" ]; then
    print_check "pass" "$file 존재"
  else
    print_check "fail" "$file 미생성"
  fi
done

# 5. 코드 품질 기본 검사
print_header "5. 코드 품질 기본 검사"

# console.log 잔존 여부
CONSOLE_LOGS=$(grep -r "console\.log" src/ --include="*.tsx" --include="*.ts" -l 2>/dev/null | wc -l)
if [ "$CONSOLE_LOGS" -eq 0 ]; then
  print_check "pass" "console.log 없음"
else
  print_check "warn" "console.log 발견: ${CONSOLE_LOGS}개 파일"
fi

# any 타입 사용 여부
ANY_TYPES=$(grep -r ": any" src/ --include="*.tsx" --include="*.ts" | wc -l)
if [ "$ANY_TYPES" -eq 0 ]; then
  print_check "pass" "any 타입 없음"
else
  print_check "warn" "any 타입 사용: ${ANY_TYPES}곳"
fi

# TODO/FIXME 잔존 여부
TODOS=$(grep -r "TODO\|FIXME\|HACK" src/ --include="*.tsx" --include="*.ts" | wc -l)
if [ "$TODOS" -eq 0 ]; then
  print_check "pass" "미완성 주석 없음"
else
  print_check "warn" "TODO/FIXME 발견: ${TODOS}개"
fi

# 6. 테스트 실행 (테스트 파일이 있는 경우)
print_header "6. 테스트 실행"
if ls src/**/*.test.{ts,tsx} 2>/dev/null | head -1 | grep -q .; then
  if npm test -- --passWithNoTests 2>/dev/null; then
    print_check "pass" "테스트 통과"
  else
    print_check "fail" "테스트 실패"
  fi
else
  print_check "warn" "테스트 파일 없음 (나중에 추가 권장)"
fi

# 최종 결과
print_header "검증 결과 요약"
echo "  오류: $ERRORS개"
echo "  경고: $WARNINGS개"
echo ""

if [ $ERRORS -eq 0 ]; then
  echo "  검증 완료: 모든 필수 검사 통과"
  exit 0
else
  echo "  검증 실패: ${ERRORS}개 오류를 수정한 후 재실행하세요"
  exit 1
fi
```

**스크립트 실행 방법**:

```bash
# 실행 권한 부여
chmod +x run-parallel-specs.sh run-specs-tmux.sh verify-specs.sh

# 병렬 실행 (백그라운드)
./run-parallel-specs.sh

# tmux 실시간 모니터링
./run-specs-tmux.sh

# 완료 후 검증
./verify-specs.sh
```

---

### 4.4 Codex SPEC 프롬프트 작성 모범 사례

Codex에 넘기는 프롬프트는 짧고 명확해야 한다. 아래 형식을 따른다.

```
# 형식
[동사] [파일 경로]에 [기능]을 구현하세요.

컨텍스트:
- 사용하는 타입: {타입명} (src/types/index.ts 참고)
- 의존하는 함수: {함수명} (src/lib/xxx.ts)
- 사용하는 컴포넌트: {컴포넌트명} (src/components/xxx.tsx)

요구사항:
1. {구체적 요구사항 1}
2. {구체적 요구사항 2}

완료 기준:
- {체크 가능한 조건 1}
- {체크 가능한 조건 2}
```

**실제 예시**:

```
src/app/api/retros/route.ts에 회고 생성 API를 구현하세요.

컨텍스트:
- 사용하는 타입: RetroFormData (src/types/index.ts)
- Supabase 서버 클라이언트: createServerClient (src/lib/supabase/server.ts)

요구사항:
1. POST /api/retros 엔드포인트
2. 요청 바디에서 sprintNumber, keep, problem, try 배열 추출
3. 현재 로그인 사용자 ID를 Supabase 세션에서 조회
4. retros 테이블에 삽입 후 생성된 ID 반환
5. 인증되지 않은 요청은 401 반환

완료 기준:
- TypeScript 타입 오류 없음
- 인증 검사 로직 포함
- 성공 시 { id: string } JSON 반환
- 실패 시 적절한 HTTP 상태 코드와 에러 메시지 반환
```

---

*이 문서는 커리큘럼 06-CH01에 대응합니다. 다음 단계는 [project-one-to-ten-kr.md](./project-one-to-ten-kr.md)를 참고하세요.*
