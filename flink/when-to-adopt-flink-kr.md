# Flink 도입 시점 판단 가이드

## 개요

이 문서는 Go Kafka Consumer로 운영 중인 스트림 처리 시스템에서 Apache Flink로 전환할 시점을 판단하는 체크리스트와 단계적 전환 전략을 제공한다.

---

## 1. Go Consumer로 충분한 경우

아래 조건을 모두 만족한다면 Flink 도입은 오버엔지니어링이다.

### 처리량 기준

- 전체 Kafka 이벤트 처리량이 **10K events/sec 이하**
- 단일 Go Consumer 프로세스가 CPU 50% 미만에서 소화 가능
- Redis 메모리 사용률이 70% 미만으로 유지됨

### 패턴 복잡도 기준

- 모든 처리 패턴이 **단순 카운팅/집계** (슬라이딩 윈도우, 텀블링 윈도우)
- **이벤트 순서**에 의존하는 패턴 없음 (예: "A 이후 B가 없으면 C")
- **이벤트 타임스탬프 정확도**가 중요하지 않음 (처리 시간 기준 허용)
- 여러 Kafka 토픽을 **조인**하여 처리하는 패턴 없음

### 신뢰성 기준

- **at-least-once** 처리로 충분 (exactly-once 불필요)
- 일부 이벤트 유실 또는 중복 처리가 비즈니스적으로 허용됨
- 장애 복구 시 몇 분의 데이터 재처리가 허용됨

### 운영 기준

- Flink 클러스터를 운영할 DevOps 인력이 없음
- Java/Scala 개발 역량이 팀에 없음
- 이미 Redis를 운영 중이고 추가 인프라 도입 비용 부담이 큼

### 구체적인 예시

```go
// 이런 패턴들은 Go Consumer로 충분하다

// 패턴 1: 단순 카운터
func handleEvent(ctx context.Context, event Event) error {
    count, _ := redis.Incr(ctx, "count:"+event.UserID).Result()
    redis.Expire(ctx, "count:"+event.UserID, time.Minute)
    if count >= threshold {
        sendAlert(event.UserID)
    }
    return nil
}

// 패턴 2: 슬라이딩 윈도우 카운터
func handleWithSlidingWindow(ctx context.Context, event Event) error {
    key := "window:" + event.UserID
    now := time.Now().UnixMilli()
    redis.ZAdd(ctx, key, redis.Z{Score: float64(now), Member: now})
    redis.ZRemRangeByScore(ctx, key, "0", fmt.Sprintf("%d", now-windowMs))
    count, _ := redis.ZCard(ctx, key).Result()
    // ... 알림 발송
    return nil
}
```

---

## 2. Flink가 필요해지는 시점

아래 항목 중 하나라도 해당된다면 Flink 도입을 진지하게 검토한다.

### 임계 조건 (하나라도 해당 시 즉시 검토)

**CEP 패턴 발생**
```
"A 이벤트 이후 N분 내에 B 이벤트가 없으면 C를 실행"
→ Go로 신뢰성 있게 구현 불가 (Redis TTL + cron 조합은 at-most-once)
→ Flink CEP notFollowedBy가 결정적 우위

예: 매치 후 30분 내 대화 미시작, 결제 후 10분 내 확인 미완료
```

**복수 Kafka 토픽 스트림 조인**
```
"swipe.recorded + match.created를 조인하여 매치율 계산"
→ Go로 구현 시 상태 관리 복잡도가 매우 높음
→ Flink DataStream API에서 connect() + CoProcessFunction

예: 이벤트 퍼널 분석, A/B 테스트 결과 실시간 집계
```

**이벤트 타임스탬프 정확도 요구**
```
"이벤트가 생성된 시각 기준으로 5분 윈도우를 계산해야 함"
→ Go Consumer는 이벤트 도착 시각(처리 시간) 기준으로만 윈도우 가능
→ 네트워크 지연, 소비자 지연이 있으면 윈도우 오차 발생
→ Flink 워터마크 기반 이벤트 타임 윈도우가 필요

예: 금융 이상 탐지, SLA 준수율 계산
```

### 처리량 기반 조건

**10K~50K events/sec**
```
Redis 병목 징후:
- Redis CPU 사용률 지속 70% 초과
- Redis 명령어 지연 P99 > 10ms
- Redis 메모리 부족으로 LRU 방출 발생

Flink 전환 효과:
- 상태를 Flink 내부(RocksDB)에 보관 → Redis 부하 제거
- TaskManager 추가로 선형 확장
```

**50K events/sec 초과**
```
Go Consumer 한계:
- 단일 Consumer 그룹의 병렬성 = Kafka 파티션 수
- 파티션 수 증가 = 재배치 오버헤드
- Redis CLUSTER 구성 필요 → 운영 복잡도 급증

Flink 전환 효과:
- 처리량에 비례한 TaskManager 추가
- 상태 분산 자동 관리
```

### 비즈니스 요구사항 기반 조건

**Exactly-once 처리 요구**
```
상황: 결제 이벤트, 포인트 적립, 구독 상태 변경
요구: 장애 발생 시 중복 처리 또는 미처리 모두 불가
Go 구현: 멱등성 키 + DB 트랜잭션으로 근사 가능 (복잡)
Flink: 체크포인트 + Kafka 트랜잭션 싱크로 exactly-once 내장
```

**복잡한 상태 공유**
```
상황: 유저 A의 이벤트가 유저 B의 처리에 영향
예: "친구 네트워크 내 이상 행동 전파 탐지"
Go 구현: Redis를 통한 간접 공유 (경쟁 조건 위험)
Flink: Broadcast State로 공유 상태 안전하게 배포
```

---

## 3. 판단 체크리스트

아래 10개 질문에 답하여 Flink 도입 여부를 판단한다.

```
[ ] Q1. 현재 처리량이 10K events/sec를 초과하거나, 6개월 내 초과할 것으로 예상되는가?
[ ] Q2. Redis 지연(P99 > 10ms)이나 메모리 부족 경고가 지속적으로 발생하는가?
[ ] Q3. "A 이후 B가 없으면 C" 형태의 복합 이벤트 패턴이 필요한가?
[ ] Q4. 두 개 이상의 Kafka 토픽을 조인하여 처리하는 패턴이 있는가?
[ ] Q5. 이벤트 생성 시각(이벤트 타임) 기준의 정확한 윈도우 계산이 필요한가?
[ ] Q6. 결제, 포인트, 구독 등 exactly-once가 필요한 이벤트를 처리하는가?
[ ] Q7. 단일 유저가 아닌 유저 그룹 또는 네트워크 수준의 상태를 추적하는가?
[ ] Q8. 3일 비활동, 구독 만료 7일 전 등 장시간 타이머가 10개 이상 존재하는가?
[ ] Q9. 팀에 Java 개발 역량이 있거나, 확보 계획이 있는가?
[ ] Q10. Flink 클러스터를 운영할 DevOps 또는 플랫폼 팀이 있는가?
```

**판단 기준**

| 체크 수 | 권장 |
|---------|------|
| 0~2개 | Go Consumer 유지. 현재 아키텍처로 충분 |
| 3~5개 (Q9, Q10 포함) | Kafka Streams 도입 검토 (JVM 기반이지만 Flink보다 단순) |
| 3~5개 (Q9, Q10 미포함) | Go Consumer 개선 (Redis 최적화, cron 개선) |
| 6개 이상 (Q9, Q10 포함) | Flink 도입 강력 권장 |
| 6개 이상 (Q9, Q10 미포함) | Java 역량 확보 후 Flink 도입, 또는 Managed Flink 서비스 검토 |

---

## 4. 단계적 도입 전략

Go Consumer에서 Flink로 바로 전환하지 말고 단계적으로 검증한다.

### Phase 0: Go Consumer (현재)

```
대상: 단순 카운팅, Redis 기반 슬라이딩 윈도우
처리량: < 10K events/sec
신뢰성: at-least-once (Redis 장애 시 카운터 유실 허용)
운영: Redis + Go 프로세스 관리
```

```go
// 현재 구조: 토픽별 Consumer + Redis 상태 관리
type FraudConsumer struct {
    reader  *kafka.Reader
    redis   *redis.Client
    alertCh chan Alert
}
```

### Phase 1: Go Consumer 개선 (필요 시)

```
목표: Redis 부하 절감, 신뢰성 향상
방법:
  - Redis 파이프라인 적극 활용 (RTT 감소)
  - Lua 스크립트로 atomic 연산 보장
  - 로컬 인메모리 캐시 + Redis 배치 플러시
  - Kafka 파티션 증설 + Consumer 그룹 확장
```

```go
// 개선: 로컬 버퍼 + 배치 플러시
type ImprovedConsumer struct {
    localBuffer map[string]int64  // 로컬 카운터
    flushTicker *time.Ticker      // 100ms마다 Redis에 배치 전송
    mu          sync.Mutex
}

func (c *ImprovedConsumer) flush(ctx context.Context) {
    c.mu.Lock()
    batch := c.localBuffer
    c.localBuffer = make(map[string]int64)
    c.mu.Unlock()

    pipe := c.redis.Pipeline()
    for key, delta := range batch {
        pipe.IncrBy(ctx, key, delta)
    }
    pipe.Exec(ctx)
}
```

### Phase 2: Kafka Streams (선택적 중간 단계)

```
대상: Java 역량이 있고 Flink의 운영 복잡도가 부담스러운 팀
특징:
  - 별도 클러스터 없음 (라이브러리 형태)
  - Kafka 기반 상태 관리 (RocksDB + Kafka 체인지로그)
  - Flink보다 단순, Go Consumer보다 강력
  - exactly-once 지원

한계:
  - CEP 패턴 미지원 (Flink CEP 없음)
  - 복잡한 윈도우 조인 제한
  - Kafka 의존도 높음
```

```java
// Kafka Streams 예시: 5분 슬라이딩 윈도우 카운터
KStream<String, SwipeEvent> swipes = builder.stream("swipe.recorded");
swipes
    .groupByKey()
    .windowedBy(SlidingWindows.ofTimeDifferenceAndGrace(
        Duration.ofMinutes(5), Duration.ofSeconds(30)
    ))
    .count()
    .filter((key, count) -> count >= 50)
    .toStream()
    .to("fraud.alert.triggered");
```

### Phase 3: Flink 도입 (점진적 전환)

```
전략: 새 패턴부터 Flink로 구현, 기존 Go Consumer는 유지
이유: 빅뱅 전환은 위험. 기능별로 검증 후 점진적 이전
```

**단계 3-1: CEP 패턴만 Flink로 구현**

```
- 기존 Go Consumer: 단순 집계 패턴 유지
- 신규 Flink Job: CEP 패턴 (매치 후 대화 미시작 등)
- 기간: 4~8주
- 검증: A/B 비교 (Go cron vs Flink CEP 결과 일치 여부)
```

**단계 3-2: 고처리량 집계 패턴 이전**

```
- Redis 병목이 발생하는 패턴부터 Flink로 이전
- Feature store sync, 피처 집계 등
- 기간: 4~6주
- 검증: Flink 처리 결과를 Redis에 쓰고 Go Consumer 결과와 비교
```

**단계 3-3: 타이머 기반 패턴 이전**

```
- cron 기반 Go 구현을 Flink 타이머로 교체
- 이탈 위험 감지, 갱신 알림 등
- 기간: 4~6주
- 검증: 알림 발송 시각 정확도 비교
```

**단계 3-4: Go Consumer 단종**

```
- 모든 패턴이 Flink로 이전된 후
- Go Consumer 코드 아카이브
- Redis 사용 패턴 재검토 (Flink가 상태 관리 → Redis 역할 축소)
- 기간: 전체 전환 완료 후 1개월 모니터링 후 결정
```

### 전환 시 주의사항

**오프셋 관리**
```
Go Consumer와 Flink가 같은 토픽을 동시에 소비할 때:
- Consumer 그룹 ID를 다르게 설정 (병렬 소비)
- Go: "fraud-consumer-swipe"
- Flink: "flink-fraud-swipe"
- 중복 알림 방지를 위해 Alert 중복 제거 로직 추가
```

**상태 초기화**
```
Flink Job 최초 시작 시 과거 이력이 없음
→ 워밍업 기간 동안 카운터 과소 추정 가능
→ 해결: Kafka earliest offset부터 읽어 상태 재구성 (초기 실행 1회)
```

**롤백 계획**
```
Flink Job 장애 시:
1. Flink Job 중지
2. Go Consumer 재시작 (Consumer 그룹 오프셋 확인 후)
3. 장애 기간 이벤트는 재처리 또는 스킵 결정
```

---

## 5. 관리형 Flink 서비스 옵션

자체 Flink 클러스터 운영이 부담스러운 경우 관리형 서비스를 검토한다.

| 서비스 | 제공사 | 특징 | 비용 |
|--------|--------|------|------|
| Amazon Managed Flink (Kinesis Data Analytics) | AWS | Flink 1.18, 자동 스케일링, S3 체크포인트 | KPU 단위 과금, 비쌈 |
| Confluent Cloud for Apache Flink | Confluent | Kafka 통합 최적화, SQL API 지원 | CFU 단위 과금 |
| Ververica Platform | Ververica (Flink 창시자) | 엔터프라이즈 운영 도구, Kubernetes 지원 | 라이선스 비용 |
| 자체 운영 (Kubernetes Operator) | 직접 | 최저 비용, 최고 운영 부담 | 인프라 비용만 |

**권장**: 초기 도입 시 자체 운영(Flink Kubernetes Operator)으로 시작하여 운영 패턴을 익힌 후 필요 시 관리형으로 전환.

---

## 6. 요약

```
현재 상황                          권장 결정
────────────────────────────────────────────────────────
단순 집계, < 10K/s, Redis 여유     → Go Consumer 유지
Redis 병목 시작, 복잡도 낮음       → Go Consumer 최적화 (배치 플러시)
CEP 패턴 1개 이상 필요             → Flink 도입 (CEP만 먼저)
처리량 > 10K/s, Java 역량 있음     → Flink 도입 (점진적)
처리량 > 10K/s, Java 역량 없음     → Kafka Streams 검토
모든 패턴 복잡, 대규모             → Flink 전체 전환
```

**핵심 원칙**: Flink는 강력하지만 운영 복잡도가 높다. Go Consumer로 충분히 해결 가능한 패턴까지 Flink로 구현하면 오히려 개발/운영 비용이 증가한다. CEP 패턴, 대규모 처리량, 이벤트 타임 정확도 중 하나라도 필요해지는 시점이 Flink 도입의 적기다.
