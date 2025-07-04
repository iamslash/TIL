



---

# 🧱 1. Top-level Query Clause란?

Elasticsearch에서 `query` 안에 직접 들어가는 clause들을 말합니다. 예:

```json
{
  "query": {
    "match": {
      "title": "Elasticsearch"
    }
  }
}
```

여기서 `match`는 **top-level clause**입니다. 이 자리에 올 수 있는 모든 후보들을 하나씩 설명하겠습니다.

---

# Top-level Query Clauses — 전체 목록과 설명

## 1. match

**설명**: 분석기(analyzer)를 통해 텍스트를 나누고, 각 토큰이 문서에 포함되어 있는지 검사하는 가장 일반적인 전체 텍스트 검색.

**예제**:

```json
{
  "query": {
    "match": {
      "title": "machine learning"
    }
  }
}
```

---

## 2. match\_phrase

**설명**: 분석된 단어들의 순서와 위치까지 고려하여 완전한 문장을 그대로 포함하는 문서를 찾음.

**예제**:

```json
{
  "query": {
    "match_phrase": {
      "title": "machine learning"
    }
  }
}
```

---

## 3. term

**설명**: 분석되지 않은 키워드나 숫자 필드에 대해 정확한 값을 검색.

**예제**:

```json
{
  "query": {
    "term": {
      "status": "published"
    }
  }
}
```

---

## 4. terms

**설명**: 하나의 필드가 여러 후보 값 중 하나와 일치하면 검색.

**예제**:

```json
{
  "query": {
    "terms": {
      "status": ["draft", "review"]
    }
  }
}
```

---

## 5. range

**설명**: 숫자, 날짜, 문자열 등에 대해 범위를 조건으로 검색.

**예제**:

```json
{
  "query": {
    "range": {
      "price": {
        "gte": 100,
        "lt": 500
      }
    }
  }
}
```

---

## 6. bool

**설명**: 여러 쿼리를 논리적으로 조합. must (AND), must\_not (NOT), should (OR), filter 등을 사용 가능.

**예제**:

```json
{
  "query": {
    "bool": {
      "must": [
        { "term": { "status": "active" } }
      ],
      "must_not": [
        { "term": { "banned": true } }
      ],
      "should": [
        { "match": { "title": "AI" } }
      ]
    }
  }
}
```

---

## 7. constant\_score

**설명**: 조건을 만족하는 문서에 동일한 점수를 부여. 점수 계산을 무시하고 필터링에만 집중할 때 사용.

**예제**:

```json
{
  "query": {
    "constant_score": {
      "filter": {
        "term": {
          "status": "active"
        }
      }
    }
  }
}
```

---

## 8. function\_score

**설명**: 쿼리 결과에 대해 점수를 수식 또는 함수로 조정. 예: 거리, 날짜 기반 가중치, 사용자 정의 스크립트 등.

**예제**:

```json
{
  "query": {
    "function_score": {
      "query": {
        "match": { "title": "robotics" }
      },
      "functions": [
        {
          "field_value_factor": {
            "field": "popularity",
            "factor": 1.5
          }
        }
      ],
      "score_mode": "multiply",
      "boost_mode": "sum"
    }
  }
}
```

---

## 9. script\_score

**설명**: 스크립트를 이용하여 점수를 직접 계산. `function_score`의 내부 요소로도 사용되지만 단독 top-level clause로도 가능.

**예제**:

```json
{
  "query": {
    "script_score": {
      "query": {
        "match_all": {}
      },
      "script": {
        "source": "doc['clicks'].value * 2"
      }
    }
  }
}
```

---

## 10. match\_all

**설명**: 모든 문서와 일치. 필터링 없이 전체 조회할 때 사용.

**예제**:

```json
{
  "query": {
    "match_all": {}
  }
}
```

---

## 11. match\_none

**설명**: 항상 문서가 일치하지 않음. 테스트나 조건부 쿼리에서 사용 가능.

**예제**:

```json
{
  "query": {
    "match_none": {}
  }
}
```

---

## 12. exists

**설명**: 특정 필드가 존재하는 문서만 조회.

**예제**:

```json
{
  "query": {
    "exists": {
      "field": "user_id"
    }
  }
}
```

---

## 13. prefix

**설명**: 문자열의 시작 부분이 특정 값으로 시작하는 경우.

**예제**:

```json
{
  "query": {
    "prefix": {
      "username": "dav"
    }
  }
}
```

---

## 14. wildcard

**설명**: 와일드카드(`*`, `?`)를 사용한 문자열 매칭.

**예제**:

```json
{
  "query": {
    "wildcard": {
      "email": "david*@gmail.com"
    }
  }
}
```

---

## 15. fuzzy

**설명**: 오타나 유사한 문자열도 허용하여 검색.

**예제**:

```json
{
  "query": {
    "fuzzy": {
      "name": {
        "value": "david",
        "fuzziness": 2
      }
    }
  }
}
```

---

## 16. multi\_match

**설명**: 여러 필드를 동시에 검색할 때 사용.

**예제**:

```json
{
  "query": {
    "multi_match": {
      "query": "neural networks",
      "fields": ["title", "description"]
    }
  }
}
```

---

## 17. dis\_max

**설명**: 여러 쿼리 중 가장 높은 점수를 가진 결과만 유지. 중복 조건의 최적화에 유용.

**예제**:

```json
{
  "query": {
    "dis_max": {
      "queries": [
        { "match": { "title": "AI" } },
        { "match": { "body": "AI" } }
      ],
      "tie_breaker": 0.3
    }
  }
}
```

---

## 18. query\_string

**설명**: Lucene 문법으로 복잡한 쿼리를 작성할 수 있음. 사용자가 직접 쿼리문을 작성하는 경우에 주로 사용.

**예제**:

```json
{
  "query": {
    "query_string": {
      "query": "title:AI AND body:robot",
      "default_field": "title"
    }
  }
}
```

---

## 19. simple\_query\_string

**설명**: query\_string보다 안전하며, 오류를 덜 발생시키는 구문. 사용자가 입력하는 쿼리에 적합.

**예제**:

```json
{
  "query": {
    "simple_query_string": {
      "query": "\"machine learning\"~5",
      "fields": ["title", "description"]
    }
  }
}
```

# FAQ

## How gauss makes scores?

Elasticsearch의 `gauss` 함수는 **가우시안 분포(정규분포)**를 기반으로 점수를 계산합니다. 수식은 다음과 같습니다:

### 가우시안 함수의 수식

```
score = exp(-0.5 * ((distance / sigma)^2))
```

여기서:
- `distance` = |현재 값 - origin|
- `sigma` = scale 매개변수에서 계산된 표준편차

### 매개변수 설명

선택하신 예제에서:
```json
"gauss": {
  "last_login": {
    "origin": "now",        // 최적값 (현재 시간)
    "scale": "7d",          // 7일
    "decay": 0.5            // 감쇠율 50%
  }
}
```

- **origin**: 점수가 최대(1.0)가 되는 기준점
- **scale**: `origin`에서 얼마나 멀어질 때 `decay` 값만큼 감쇠할지 결정
- **decay**: `scale` 거리에서 점수가 얼마나 감쇠될지 (0.5 = 50%)

### 실제 계산 예시

현재 시간을 기준으로 사용자의 `last_login`이 얼마나 오래됐는지에 따른 점수:

- **지금 로그인**: 거리 = 0일 → 점수 = 1.0
- **3.5일 전**: 거리 = 3.5일 → 점수 ≈ 0.7
- **7일 전**: 거리 = 7일 → 점수 = 0.5 (decay 값)
- **14일 전**: 거리 = 14일 → 점수 ≈ 0.1
- **21일 전**: 거리 = 21일 → 점수 ≈ 0.02

### 시각적 그래프

```
점수
 1.0 ┤
     │   ●
 0.8 ┤    ╲
     │     ╲
 0.6 ┤      ╲
     │       ╲
 0.4 ┤        ●  ← 7일 지점 (decay=0.5)
     │         ╲
 0.2 ┤          ╲
     │           ╲●
 0.0 ┤            ╲___
     └────────────────────
      0   7   14   21   일수
```

이렇게 **최근 접속일수록 높은 점수**를 부여하고, 시간이 지날수록 **부드럽게 감소**하는 점수 분포를 만들어줍니다.

## match_all is mandatory of script_score?

### 오류 발생 예시

```json
{
  "query": {
    "script_score": {
      "script": {
        "source": "doc['purchase_count'].value * 0.5 + doc['review_score'].value * 2"
      }
    }
  }
}
```

이렇게 하면 다음과 같은 오류가 발생합니다:

```json
{
  "error": {
    "type": "parsing_exception",
    "reason": "[script_score] query malformed, no start_object after query name"
  }
}
```

### script_score의 동작 원리

`script_score`는 **2단계 프로세스**로 동작합니다:

1. **1단계**: `query` 부분으로 문서 필터링
2. **2단계**: 필터링된 문서들에 대해 `script`로 점수 계산

```json
{
  "script_score": {
    "query": {          // 1단계: 어떤 문서들을 대상으로 할 것인가?
      "match_all": {}    // → 모든 문서
    },
    "script": {          // 2단계: 선택된 문서들의 점수를 어떻게 계산할 것인가?
      "source": "doc['purchase_count'].value * 0.5 + doc['review_score'].value * 2"
    }
  }
}
```

### 다른 query 예시들

#### 특정 조건 문서만 점수 계산
```json
{
  "script_score": {
    "query": {
      "term": { "status": "active" }    // active 사용자만 대상
    },
    "script": {
      "source": "doc['purchase_count'].value * 0.5 + doc['review_score'].value * 2"
    }
  }
}
```

#### 복합 조건
```json
{
  "script_score": {
    "query": {
      "bool": {
        "must": [
          { "range": { "age": { "gte": 18 } } },
          { "term": { "verified": true } }
        ]
      }
    },
    "script": {
      "source": "doc['purchase_count'].value * 0.5 + doc['review_score'].value * 2"
    }
  }
}
```

**결론**: `query` 부분이 없으면 Elasticsearch가 어떤 문서들을 대상으로 점수를 계산할지 알 수 없기 때문에 **구문 오류**가 발생합니다. `match_all`은 "모든 문서를 대상으로 하겠다"는 의미입니다.

## How boost_mode makes scores?

`boost_mode`는 **원래 쿼리의 점수**와 **함수들의 점수**를 어떻게 결합할지 결정하는 매개변수입니다.

### function_score 점수 계산 과정

```
1단계: 원래 쿼리 점수 계산
2단계: 각 함수의 점수 계산  
3단계: score_mode로 함수 점수들 결합
4단계: boost_mode로 원래 점수와 함수 점수 최종 결합
```

### boost_mode 옵션들

#### 1. `multiply` (예제에서 사용)
```
최종점수 = 원래점수 × 함수점수
```

#### 2. `sum`
```
최종점수 = 원래점수 + 함수점수
```

#### 3. `avg`
```
최종점수 = (원래점수 + 함수점수) / 2
```

#### 4. `max`
```
최종점수 = max(원래점수, 함수점수)
```

#### 5. `min`
```
최종점수 = min(원래점수, 함수점수)
```

#### 6. `replace`
```
최종점수 = 함수점수 (원래점수 무시)
```

### 구체적인 계산 예시

선택하신 예제에서:

```json
{
  "query": { "term": { "status": "active" } },  // 원래 쿼리
  "functions": [
    { "gauss": { "last_login": {...} } },       // 함수1: 0.8점
    { "field_value_factor": { "activity_score": {...} } }  // 함수2: 1.5점
  ],
  "score_mode": "sum",        // 함수점수 결합: 0.8 + 1.5 = 2.3
  "boost_mode": "multiply"    // 최종 결합 방식
}
```

**가정**: 
- 원래 쿼리 점수: 1.2
- gauss 함수 점수: 0.8
- field_value_factor 점수: 1.5

**계산 과정**:
1. **함수점수 결합** (score_mode="sum"): 0.8 + 1.5 = 2.3
2. **최종점수 계산** (boost_mode="multiply"): 1.2 × 2.3 = **2.76**

### boost_mode별 결과 비교

같은 조건에서 boost_mode만 바꾼 경우:

| boost_mode | 계산식 | 결과 |
|------------|--------|------|
| multiply   | 1.2 × 2.3 | 2.76 |
| sum        | 1.2 + 2.3 | 3.5  |
| avg        | (1.2 + 2.3) / 2 | 1.75 |
| max        | max(1.2, 2.3) | 2.3  |
| min        | min(1.2, 2.3) | 1.2  |
| replace    | 2.3 | 2.3  |

**결론**: `boost_mode="multiply"`는 원래 쿼리와 함수가 모두 높은 점수를 가질 때 **상승효과**를 만들어 더 높은 점수를 부여합니다.

## function_score vs script_score?

`function_score`와 `script_score`는 모두 **점수 계산을 위한 쿼리**이지만, 접근 방식과 사용 목적이 다릅니다.

### 주요 차이점

| 구분 | function_score | script_score |
|------|----------------|--------------|
| **계산 방식** | 미리 정의된 함수들 조합 | 직접 스크립트 작성 |
| **복잡성** | 구조화된 설정 | 자유로운 로직 구현 |
| **성능** | 최적화된 내장 함수 | 스크립트 실행 오버헤드 |
| **유연성** | 제한적 | 무제한 |

### 1. function_score - 구조화된 접근

```json
{
  "function_score": {
    "query": { "term": { "status": "active" } },
    "functions": [
      {
        "gauss": {
          "last_login": {
            "origin": "now",
            "scale": "7d",
            "decay": 0.5
          }
        }
      },
      {
        "field_value_factor": {
          "field": "activity_score",
          "factor": 1.2
        }
      }
    ],
    "score_mode": "sum",
    "boost_mode": "multiply"
  }
}
```

**특징**:
- 미리 정의된 함수들 (`gauss`, `field_value_factor`, `random_score` 등)
- 여러 함수를 조합 가능
- `score_mode`, `boost_mode`로 점수 결합 방식 제어
- 성능 최적화됨

### 2. script_score - 자유로운 로직

```json
{
  "script_score": {
    "query": { "term": { "status": "active" } },
    "script": {
      "source": """
        double login_score = Math.exp(-0.5 * Math.pow((System.currentTimeMillis() - doc['last_login'].value.millis) / (7 * 24 * 60 * 60 * 1000), 2));
        double activity_score = doc['activity_score'].value * 1.2;
        return login_score + activity_score;
      """
    }
  }
}
```

**특징**:
- 직접 스크립트로 점수 계산 로직 작성
- 복잡한 수학적 계산 가능
- 조건문, 반복문 등 프로그래밍 로직 사용
- 더 느릴 수 있음

### 언제 어떤 것을 사용할까?

#### function_score 사용 권장
- **일반적인 점수 조정**: 거리, 날짜, 필드값 기반
- **성능이 중요한 경우**
- **여러 함수 조합이 필요한 경우**
- **유지보수가 중요한 경우**

```json
{
  "function_score": {
    "functions": [
      { "gauss": { "location": {...} } },     // 거리 기반
      { "gauss": { "date": {...} } },         // 날짜 기반
      { "field_value_factor": { "field": "popularity" } }  // 인기도 기반
    ]
  }
}
```

#### script_score 사용 권장
- **복잡한 비즈니스 로직**
- **조건부 계산**
- **function_score로 불가능한 계산**

```json
{
  "script_score": {
    "script": {
      "source": """
        if (doc['user_type'].value == 'premium') {
          return doc['base_score'].value * 2;
        } else if (doc['purchase_count'].value > 10) {
          return doc['base_score'].value * 1.5;
        } else {
          return doc['base_score'].value;
        }
      """
    }
  }
}
```

### 성능 비교

```
function_score: 빠름 ⚡⚡⚡
script_score:   느림 ⚡
```

**결론**: 
- **단순한 점수 조정**: `function_score` 사용
- **복잡한 로직**: `script_score` 사용
- **성능 우선**: `function_score` 선택


## How tie_breaker makes scores?

`tie_breaker`는 **dis_max 쿼리에서 최고 점수 외의 다른 쿼리 점수들을 얼마나 반영할지** 결정하는 매개변수입니다.

### dis_max의 기본 동작

`dis_max`는 기본적으로 **가장 높은 점수만** 사용합니다:

```json
{
  "dis_max": {
    "queries": [
      { "match": { "title": "chatbot" } },    // 점수: 2.0
      { "match": { "body": "chatbot" } }      // 점수: 1.5
    ]
  }
}
```

**tie_breaker 없이**: 최종 점수 = 2.0 (최고점수만)

### tie_breaker의 역할

### 계산 공식
```
최종점수 = 최고점수 + (다른점수들의 합 × tie_breaker)
```

#### 예제 계산

선택하신 예제에서:
- title 매칭 점수: 2.0
- body 매칭 점수: 1.5
- tie_breaker: 0.3

```
최종점수 = 2.0 + (1.5 × 0.3) = 2.0 + 0.45 = 2.45
```

### tie_breaker 값에 따른 차이

| tie_breaker | 계산 | 결과 | 의미 |
|-------------|------|------|------|
| 0.0 (기본값) | 2.0 + (1.5 × 0.0) | 2.0 | 최고점수만 사용 |
| 0.3 | 2.0 + (1.5 × 0.3) | 2.45 | 다른 점수 30% 반영 |
| 0.5 | 2.0 + (1.5 × 0.5) | 2.75 | 다른 점수 50% 반영 |
| 1.0 | 2.0 + (1.5 × 1.0) | 3.5 | 모든 점수 합계 (bool should와 동일) |

### 실제 사용 시나리오

#### 1. 제목 우선, 본문 보조 고려

```json
{
  "dis_max": {
    "queries": [
      { "match": { "title": "machine learning" } },
      { "match": { "content": "machine learning" } }
    ],
    "tie_breaker": 0.3
  }
}
```

**의도**: 제목 매칭이 우선이지만, 본문 매칭도 어느 정도 반영

#### 2. 여러 필드 검색에서 균형 조정

```json
{
  "dis_max": {
    "queries": [
      { "match": { "name": "david" } },
      { "match": { "email": "david" } },
      { "match": { "description": "david" } }
    ],
    "tie_breaker": 0.2
  }
}
```

**의도**: 가장 잘 매칭되는 필드 우선, 다른 필드들도 소량 반영

### tie_breaker vs bool should 비교

#### dis_max (tie_breaker=0.3)
```
점수 = max(점수들) + (나머지점수들 × 0.3)
```

#### bool should
```
점수 = 모든 점수들의 합
```

**언제 사용할까?**

- **dis_max + tie_breaker**: 하나의 필드가 주요하고 다른 필드들은 보조적일 때
- **bool should**: 모든 필드가 동등하게 중요할 때

**결론**: `tie_breaker`는 **주요 매칭 외에 다른 매칭들도 약간씩 고려**하여 더 정교한 관련성 점수를 만들어줍니다.

# 현실적인 시나리오를 기반으로 한 실습 예제

---

## 1. `match` – 사용자가 입력한 검색어로 블로그 제목 찾기

**시나리오**: 사용자가 "deep learning"이라는 키워드로 블로그 글을 검색함

```json
{
  "query": {
    "match": {
      "title": "deep learning"
    }
  }
}
```

**설명**: 텍스트는 분석되어 "deep"과 "learning" 두 단어로 분리되고, 이 단어들을 모두 포함한 문서가 검색됩니다.

---

## 2. `term` – 특정 상태인 상품 검색

**시나리오**: 상품 상태가 `"available"`인 상품만 검색하고 싶음

```json
{
  "query": {
    "term": {
      "status": "available"
    }
  }
}
```

**설명**: status 필드는 키워드(keyword) 타입이어야 하고, 분석 없이 정확히 일치해야 합니다.

---

## 3. `terms` – 특정 카테고리에 속한 게시물들 찾기

**시나리오**: 카테고리가 "tech", "science", 또는 "math"인 글 목록 필터링

```json
{
  "query": {
    "terms": {
      "category": ["tech", "science", "math"]
    }
  }
}
```

---

## 4. `range` – 특정 가격대 상품 검색

**시나리오**: 10,000원 이상 50,000원 미만인 상품을 찾음

```json
{
  "query": {
    "range": {
      "price": {
        "gte": 10000,
        "lt": 50000
      }
    }
  }
}
```

---

## 5. `bool` – 여러 조건을 조합

**시나리오**: 제목에 "machine"이 포함되고, 상태가 "active"이며, 금지 사용자는 제외

```json
{
  "query": {
    "bool": {
      "must": [
        { "match": { "title": "machine" } },
        { "term": { "status": "active" } }
      ],
      "must_not": [
        { "term": { "banned": true } }
      ]
    }
  }
}
```

---

## 6. `constant_score` – 점수 무시하고 단순 필터

**시나리오**: 광고용 사용자 필터링, relevance 점수는 필요 없음

```json
{
  "query": {
    "constant_score": {
      "filter": {
        "term": {
          "user_type": "premium"
        }
      }
    }
  }
}
```

---

## 7. `function_score` – 최근 접속 시간과 활동 점수로 사용자 순위 결정

**시나리오**: 조건에 맞는 사용자 중, 최근 접속한 사람과 활동 점수가 높은 사람에게 가중치 부여

```json
{
  "query": {
    "function_score": {
      "query": {
        "term": {
          "status": "active"
        }
      },
      "functions": [
        {
          "gauss": {
            "last_login": {
              "origin": "now",
              "scale": "7d",
              "decay": 0.5
            }
          }
        },
        {
          "field_value_factor": {
            "field": "activity_score",
            "factor": 1.2
          }
        }
      ],
      "score_mode": "sum",
      "boost_mode": "multiply"
    }
  }
}
```

---

## 8. `script_score` – 사용자 맞춤 점수 계산

**시나리오**: 사용자의 구매 수에 따라 맞춤 점수 계산

```json
{
  "query": {
    "script_score": {
      "query": {
        "match_all": {}
      },
      "script": {
        "source": "doc['purchase_count'].value * 0.5 + doc['review_score'].value * 2"
      }
    }
  }
}
```

---

## 9. `match_phrase` – 정확한 문장 포함 검색

**시나리오**: "artificial intelligence"라는 문장이 포함된 글 검색

```json
{
  "query": {
    "match_phrase": {
      "content": "artificial intelligence"
    }
  }
}
```

---

## 10. `exists` – 특정 필드가 존재하는 문서만 필터링

**시나리오**: 이메일 주소가 입력된 사용자만 찾기

```json
{
  "query": {
    "exists": {
      "field": "email"
    }
  }
}
```

---

## 11. `prefix` – 문자열 시작 조건 검색

**시나리오**: 이름이 "da"로 시작하는 사용자 검색

```json
{
  "query": {
    "prefix": {
      "name": "da"
    }
  }
}
```

---

## 12. `wildcard` – 유연한 패턴 검색

**시나리오**: "david"로 시작하고 "@gmail.com"으로 끝나는 이메일 검색

```json
{
  "query": {
    "wildcard": {
      "email": "david*@gmail.com"
    }
  }
}
```

---

## 13. `fuzzy` – 오타 허용 검색

**시나리오**: "david"를 오타 포함하여 검색 (예: "dvid", "david", "dvaid")

```json
{
  "query": {
    "fuzzy": {
      "name": {
        "value": "david",
        "fuzziness": 2
      }
    }
  }
}
```

---

## 14. `multi_match` – 여러 필드에서 키워드 검색

**시나리오**: 제목이나 설명에서 "quantum computing" 검색

```json
{
  "query": {
    "multi_match": {
      "query": "quantum computing",
      "fields": ["title", "description"]
    }
  }
}
```

---

## 15. `dis_max` – 여러 쿼리 중 최고 점수만 사용

**시나리오**: 제목 또는 본문에 "chatbot"이 있을 때, 하나라도 일치하면 문서를 포함시키되 더 잘 맞는 쪽의 점수로 정렬

```json
{
  "query": {
    "dis_max": {
      "queries": [
        { "match": { "title": "chatbot" } },
        { "match": { "body": "chatbot" } }
      ],
      "tie_breaker": 0.3
    }
  }
}
```

---

## 16. `query_string` – 복잡한 수식 검색

**시나리오**: 사용자가 직접 `"AI AND robot"`과 같은 구문을 입력했을 때

```json
{
  "query": {
    "query_string": {
      "query": "title:AI AND body:robot",
      "default_field": "body"
    }
  }
}
```

---

## 17. `simple_query_string` – 안전한 수식 검색

**시나리오**: 검색어에 문법 오류가 있을 수 있으므로 에러 발생 없이 검색 실행

```json
{
  "query": {
    "simple_query_string": {
      "query": "\"deep learning\"~5",
      "fields": ["title", "body"]
    }
  }
}
```

---

## 18. `match_all` – 전체 문서 조회

**시나리오**: 모든 문서를 점수와 무관하게 가져오기

```json
{
  "query": {
    "match_all": {}
  }
}
```

---

## 19. `match_none` – 조건에 따라 결과가 없어야 하는 경우

**시나리오**: 필터 조건이 맞지 않는 경우, 강제로 빈 결과를 리턴

```json
{
  "query": {
    "match_none": {}
  }
}
```

---

# 시나리오 1: 사용자의 선호 장르 기반 영화 추천

## 1-1. 기본 terms 필터링

```json
{
  "query": {
    "terms": {
      "genre": ["sci-fi", "action"]
    }
  }
}
```

## 1-2. genre + 최신 개봉일 순 정렬

```json
{
  "query": {
    "bool": {
      "must": [
        { "terms": { "genre": ["sci-fi", "action"] } }
      ]
    }
  },
  "sort": [
    { "release_date": "desc" }
  ]
}
```

## 1-3. genre 필터 + 평점 기반 가중치 부여

```json
{
  "query": {
    "function_score": {
      "query": {
        "terms": {
          "genre": ["sci-fi", "action"]
        }
      },
      "functions": [
        {
          "field_value_factor": {
            "field": "rating",
            "factor": 1.5,
            "missing": 3.0
          }
        }
      ],
      "score_mode": "sum",
      "boost_mode": "multiply"
    }
  }
}
```

---

# 시나리오 2: 사용자 위치 기반 근처 이벤트 추천

## 2-1. 반경 10km 이벤트 필터

```json
{
  "query": {
    "geo_distance": {
      "distance": "10km",
      "location": {
        "lat": 37.4979,
        "lon": 127.0276
      }
    }
  }
}
```

## 2-2. 가까운 거리일수록 높은 점수 (gauss 사용)

```json
{
  "query": {
    "function_score": {
      "query": {
        "geo_distance": {
          "distance": "10km",
          "location": {
            "lat": 37.4979,
            "lon": 127.0276
          }
        }
      },
      "functions": [
        {
          "gauss": {
            "location": {
              "origin": "37.4979,127.0276",
              "scale": "2km",
              "decay": 0.5
            }
          }
        }
      ]
    }
  }
}
```

## 2-3. 거리 + 인기 점수 조합

```json
{
  "query": {
    "function_score": {
      "query": {
        "geo_distance": {
          "distance": "10km",
          "location": {
            "lat": 37.4979,
            "lon": 127.0276
          }
        }
      },
      "functions": [
        {
          "gauss": {
            "location": {
              "origin": "37.4979,127.0276",
              "scale": "2km",
              "decay": 0.5
            }
          }
        },
        {
          "field_value_factor": {
            "field": "popularity_score",
            "factor": 1.2
          }
        }
      ],
      "score_mode": "sum",
      "boost_mode": "multiply"
    }
  }
}
```

---

# 시나리오 3: 최근 접속 사용자 우선 추천

## 3-1. 최근 7일 내 접속 필터

```json
{
  "query": {
    "range": {
      "last_login": {
        "gte": "now-7d/d"
      }
    }
  }
}
```

## 3-2. 최근 접속일이 가까울수록 점수 높임 (gauss)

```json
{
  "query": {
    "function_score": {
      "query": {
        "match_all": {}
      },
      "functions": [
        {
          "gauss": {
            "last_login": {
              "origin": "now",
              "scale": "7d",
              "decay": 0.3
            }
          }
        }
      ]
    }
  }
}
```

## 3-3. 최근 접속 + 활동점수 조합

```json
{
  "query": {
    "function_score": {
      "query": {
        "term": { "status": "active" }
      },
      "functions": [
        {
          "gauss": {
            "last_login": {
              "origin": "now",
              "scale": "7d",
              "decay": 0.3
            }
          }
        },
        {
          "field_value_factor": {
            "field": "activity_score",
            "factor": 1.1
          }
        }
      ],
      "score_mode": "sum",
      "boost_mode": "multiply"
    }
  }
}
```

---

# 시나리오 4: 활동 점수 기반 정렬

## 4-1. 단순한 `activity_score` 기준 정렬

```json
{
  "query": {
    "match_all": {}
  },
  "sort": [
    { "activity_score": "desc" }
  ]
}
```

## 4-2. 점수 없을 경우 기본값 부여

```json
{
  "query": {
    "function_score": {
      "query": {
        "match_all": {}
      },
      "functions": [
        {
          "field_value_factor": {
            "field": "activity_score",
            "missing": 0.1
          }
        }
      ]
    }
  }
}
```

## 4-3. 활동 점수 + 평판 점수 스크립트 기반 조합

```json
{
  "query": {
    "script_score": {
      "query": {
        "match_all": {}
      },
      "script": {
        "source": "doc['activity_score'].value * 0.7 + doc['reputation_score'].value * 0.3"
      }
    }
  }
}
```

---

# 시나리오 5: 사용자 태그와 아이템 태그 매칭

## 5-1. 사용자 선호 태그와 일치하는 태그 검색

```json
{
  "query": {
    "terms": {
      "tags": ["eco", "vegan", "organic"]
    }
  }
}
```

## 5-2. 일치 개수에 따라 점수 증가 (스크립트)

```json
{
  "query": {
    "script_score": {
      "query": {
        "terms": {
          "tags": ["eco", "vegan", "organic"]
        }
      },
      "script": {
        "source": """
          int matches = 0;
          for (tag in params.user_tags) {
            if (doc['tags'].contains(tag)) {
              matches += 1;
            }
          }
          return matches;
        """,
        "params": {
          "user_tags": ["eco", "vegan", "organic"]
        }
      }
    }
  }
}
```

## 5-3. 태그 일치 수 + 평점 조합 점수

```json
{
  "query": {
    "script_score": {
      "query": {
        "terms": {
          "tags": ["eco", "vegan", "organic"]
        }
      },
      "script": {
        "source": """
          int matches = 0;
          for (tag in params.user_tags) {
            if (doc['tags'].contains(tag)) {
              matches += 1;
            }
          }
          return matches * 2 + doc['rating'].value;
        """,
        "params": {
          "user_tags": ["eco", "vegan", "organic"]
        }
      }
    }
  }
}
```

---

# 시나리오 1: 사용자 프로필 기반 상대 매칭

**상황**
사용자 A는 다음과 같은 프로필을 가짐:

* 성별: 남성
* 관심 성별: 여성
* 나이: 30세
* 선호 나이대: 25\~35세
* 위치: 서울

이 정보를 기반으로 조건에 맞는 사용자(상대)를 찾고 점수화.

## 1-1. 기본 필터 기반 매칭

```json
{
  "query": {
    "bool": {
      "must": [
        { "term": { "gender": "female" } },
        { "range": { "age": { "gte": 25, "lte": 35 } } },
        {
          "geo_distance": {
            "distance": "50km",
            "location": {
              "lat": 37.5665,
              "lon": 126.9780
            }
          }
        }
      ]
    }
  }
}
```

## 1-2. 여기에 최근 접속 시간 가중치 추가

```json
{
  "query": {
    "function_score": {
      "query": {
        "bool": {
          "must": [
            { "term": { "gender": "female" } },
            { "range": { "age": { "gte": 25, "lte": 35 } } }
          ]
        }
      },
      "functions": [
        {
          "gauss": {
            "last_active": {
              "origin": "now",
              "scale": "3d"
            }
          }
        }
      ]
    }
  }
}
```

## 1-3. 프로필 유사도 기반 스크립트 점수화

```json
{
  "query": {
    "script_score": {
      "query": {
        "bool": {
          "must": [
            { "term": { "gender": "female" } }
          ]
        }
      },
      "script": {
        "source": """
          double age_score = 1 - Math.abs(doc['age'].value - params.user_age) / 10.0;
          return age_score * doc['activity_score'].value;
        """,
        "params": {
          "user_age": 30
        }
      }
    }
  }
}
```

---

# 시나리오 2: 실시간 A/B 추천 실험 (예: 알고리즘 버전 테스트)

**상황**
A/B 실험에서 서로 다른 추천 알고리즘 버전을 테스트하려고 함.
각 사용자에게 실험 그룹을 랜덤 부여하고 추천 결과를 다르게 구성.

## 2-1. 그룹 A에게는 단순 평점 기반 추천

```json
{
  "query": {
    "function_score": {
      "query": {
        "term": { "experiment_group": "A" }
      },
      "functions": [
        {
          "field_value_factor": {
            "field": "rating"
          }
        }
      ]
    }
  }
}
```

## 2-2. 그룹 B는 활동 점수 + 최근성 가중치 조합

```json
{
  "query": {
    "function_score": {
      "query": {
        "term": { "experiment_group": "B" }
      },
      "functions": [
        {
          "gauss": {
            "last_active": {
              "origin": "now",
              "scale": "5d"
            }
          }
        },
        {
          "field_value_factor": {
            "field": "activity_score"
          }
        }
      ],
      "score_mode": "sum"
    }
  }
}
```

## 2-3. 실험 그룹을 다르게 처리하고, 실험 이름으로 구분

```json
{
  "query": {
    "bool": {
      "filter": [
        { "term": { "experiment_name": "rec_algo_test" } },
        {
          "terms": {
            "experiment_group": ["A", "B"]
          }
        }
      ]
    }
  }
}
```

> 참고: 실험 로그는 Elasticsearch 외부에서 수집 및 분석 (예: BigQuery, Redshift, 로그 분석 플랫폼 등)

---

# 시나리오 3: 관심사 벡터 기반 유사도 필터링 (Cosine similarity 등)

**상황**
사용자와 아이템 모두 고차원 관심사 벡터 (예: `dense_vector`)를 보유.
두 벡터 간 cosine similarity를 계산해 유사도 높은 아이템 추천.

## 3-1. dense\_vector 필드를 이용한 cosine similarity 기반 정렬

```json
{
  "query": {
    "script_score": {
      "query": {
        "match_all": {}
      },
      "script": {
        "source": "cosineSimilarity(params.query_vector, 'item_vector') + 1.0",
        "params": {
          "query_vector": [0.1, 0.2, 0.3, 0.4]
        }
      }
    }
  }
}
```

## 3-2. 유사도 + 인기도 점수 조합

```json
{
  "query": {
    "script_score": {
      "query": {
        "match_all": {}
      },
      "script": {
        "source": """
          double sim = cosineSimilarity(params.query_vector, 'item_vector') + 1.0;
          return sim * doc['popularity'].value;
        """,
        "params": {
          "query_vector": [0.1, 0.2, 0.3, 0.4]
        }
      }
    }
  }
}
```

## 3-3. 유사도 임계값 필터링

```json
{
  "query": {
    "script_score": {
      "query": {
        "match_all": {}
      },
      "script": {
        "source": """
          double sim = cosineSimilarity(params.query_vector, 'item_vector');
          return sim > 0.7 ? sim : 0;
        """,
        "params": {
          "query_vector": [0.1, 0.2, 0.3, 0.4]
        }
      }
    }
  }
}
```

> `dense_vector` 필드 사용을 위해서는 `index: false`, `similarity: cosine` 설정이 필요합니다.

---

# 시나리오 4: 인기 사용자 자동 노출 제어

**상황**
너무 자주 노출되는 인기 사용자의 노출 빈도를 제어하거나 제한하고 싶음.

## 4-1. 인기 사용자만 노출 (기본 점수)

```json
{
  "query": {
    "range": {
      "popularity": {
        "gte": 80
      }
    }
  }
}
```

## 4-2. 인기 사용자일수록 점수를 의도적으로 감소시킴

```json
{
  "query": {
    "script_score": {
      "query": {
        "range": {
          "popularity": {
            "gte": 50
          }
        }
      },
      "script": {
        "source": "1 / Math.pow(doc['popularity'].value, 0.5)"
      }
    }
  }
}
```

## 4-3. 사용자별로 노출 횟수 기반 점수 조정 (데이터에 노출 카운트가 저장된 경우)

```json
{
  "query": {
    "script_score": {
      "query": {
        "match_all": {}
      },
      "script": {
        "source": """
          double base_score = doc['rating'].value;
          double penalty = Math.log(1 + doc['exposure_count'].value);
          return base_score / penalty;
        """
      }
    }
  }
}
```

> 노출 횟수 기반 노멀라이징을 통해 자주 노출된 사용자는 점수를 낮추고 새로운 사용자를 위로 올릴 수 있음.

---

이 시나리오들은 모두 Elasticsearch를 기반으로 **실시간 사용자 맞춤형 추천을 구현하는 데 필수적인 전략들**입니다.
특정 환경 (Tinder, Netflix, 전자상거래 등)에 맞춰 튜닝된 예제를 원하신다면 도메인과 데이터를 기준으로 더욱 구체화해드릴 수 있습니다. 원하시는 방향이 있을까요?
