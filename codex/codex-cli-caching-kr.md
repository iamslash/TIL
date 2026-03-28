# Codex CLI 캐시 구조

> **근거**: `source-inspected` (`~/.codex/models_cache.json`, `~/.codex/cloud-requirements-cache.json`, 최신 session JSONL의 `token_count`, OpenAI Responses/모델 문서), `runtime-observed` (`cached_input_tokens` 필드, 로컬 cache 파일 존재), `inferred` (원격 prompt cache의 정확한 서버 내부 구현은 공개되지 않았으므로 일부는 추정)
>
> **관측 시점**: 2026-03-26.

Codex에는 한 종류의 캐시만 있는 것이 아니다. 현재 로컬 환경에서 확인되는 캐시는 크게 두 층이다.

1. OpenAI 쪽 입력 prefix 재사용 캐시
2. 로컬 메타데이터/요구사항 캐시

## 1. upstream 입력 캐시

최신 session JSONL의 `token_count` 이벤트에는 `cached_input_tokens` 필드가 들어 있다. (`runtime-observed`)

```jsonc
{
  "total_token_usage": {
    "input_tokens": 562526,
    "cached_input_tokens": 358144,
    "output_tokens": 9923
  }
}
```

이건 Codex가 같은 입력 prefix를 다시 보낼 때, upstream에서 일부 입력이 캐시 히트된다는 뜻이다.

## 무엇이 반복 prefix가 되는가

Codex 세션에서는 아래 요소가 여러 turn에 걸쳐 반복되기 쉽다. (`inferred`)

```text
- base instructions
- developer instructions
- AGENTS.md / prompt overlay
- 이전 turn까지의 누적 세션 컨텍스트
- tool schema 또는 host runtime metadata
```

즉, turn이 진행될수록 "앞부분은 같고 끝부분만 새로 추가"되는 구조가 만들어지기 쉽다.

## `pwd` 예시에서의 의미

`pwd`처럼 짧은 질문도 실제 세션에는 다음이 함께 실린다. (`runtime-observed` + `inferred`)

```text
base instructions
+ developer messages
+ 사용자 입력 "pwd"
+ tool call / tool result
+ 최종 응답
```

질문은 짧아도 세션 prefix는 짧지 않다. 그래서 같은 세션이 길어질수록 `cached_input_tokens`가 의미 있게 커질 수 있다.

## 비용/지연 관점

OpenAI 모델 문서는 일부 모델에서 `cached input` 가격을 별도로 표기한다. (`source-inspected`) 즉, 캐시 히트는 단순한 내부 최적화가 아니라 과금/지연에도 반영되는 메커니즘이다.

안전한 결론은 다음 정도이다.
- 캐시 히트된 입력은 일반 입력과 별도로 집계된다.
- 긴 공통 prefix가 반복될수록 비용과 TTFT에 유리할 가능성이 높다.
- 정확한 할인율과 적용 조건은 모델/시점에 따라 달라질 수 있으므로 문서 상단의 관측 시점을 함께 봐야 한다.

## 서버 내부 구현에 대해 확실히 말할 수 있는 범위

확실한 것:
- Codex telemetry에 `cached_input_tokens`가 존재한다.
- OpenAI 모델 문서가 `cached input` 가격 축을 공개한다.

추정인 것:
- 실제 서버 내부에서 KV cache나 그와 유사한 prefix 재사용 메커니즘이 동작할 가능성이 높다.
- 다만 Codex 로컬만으로는 exact matching 규칙, TTL, invalidation 규칙을 볼 수 없다.

따라서 "입력 캐시가 있다"는 것은 강하게 말할 수 있지만, "정확히 어떤 자료구조로 구현됐다"는 부분은 추정으로 남겨야 한다.

## 2. 로컬 모델 메타데이터 캐시

`~/.codex/models_cache.json`은 모델 카탈로그 캐시이다. (`source-inspected`)

```jsonc
{
  "fetched_at": "...",
  "etag": "...",
  "client_version": "0.117.0",
  "models": [
    {
      "slug": "gpt-5.4",
      "supported_reasoning_levels": [...],
      "base_instructions": "...",
      "context_window": 272000
    }
  ]
}
```

이 파일은 다음 역할을 한다.
- 모델 목록 캐싱
- reasoning level/도구 지원 여부 캐싱
- base instructions 캐싱

즉, Codex는 실행 때마다 모든 모델 메타데이터를 새로 받지 않는다.

## 3. 클라우드 요구사항 캐시

`~/.codex/cloud-requirements-cache.json`도 별도 캐시이다. (`source-inspected`)

```jsonc
{
  "signed_payload": {
    "cached_at": "...",
    "expires_at": "...",
    "chatgpt_user_id": "...",
    "account_id": "...",
    "contents": null
  },
  "signature": "..."
}
```

역할:
- 클라우드 entitlement/requirements 검증 결과 임시 저장
- 만료 시각 기반 재검증
- 서명 기반 무결성 검증

즉, 이 파일은 "모델 입력 캐시"가 아니라 "권한/요구사항 메타데이터 캐시"에 가깝다.

## 4. 버전 캐시

`version.json`도 넓은 의미의 캐시이다. (`source-inspected`)

```jsonc
{
  "latest_version": "0.116.0",
  "last_checked_at": "...",
  "dismissed_version": null
}
```

이 파일은 버전 체크 결과를 재사용한다.

## 5. shell snapshot은 캐시인가

`shell_snapshots/*.sh`는 엄밀히 말해 캐시보다 "재현용 스냅샷"에 가깝다. (`runtime-observed`)

하지만 실행 때마다 셸 상태를 다시 추론하지 않고 재사용 가능한 형태로 저장한다는 점에서, 넓은 의미의 실행 컨텍스트 캐시로 볼 수는 있다. (`inferred`)

## 요약

| 종류 | 파일/신호 | 역할 |
|------|------|------|
| upstream 입력 캐시 | `cached_input_tokens` | 반복 prefix 재사용 |
| 모델 메타데이터 캐시 | `models_cache.json` | 모델 목록/기본 지침 재사용 |
| 클라우드 요구사항 캐시 | `cloud-requirements-cache.json` | entitlement/requirements 재사용 |
| 버전 캐시 | `version.json` | 최신 버전 체크 결과 재사용 |
| 실행 스냅샷 | `shell_snapshots/*.sh` | 셸 상태 재현 |
