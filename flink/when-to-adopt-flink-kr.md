# Flink 도입 시점 판단 가이드

## 목차

- [개요](#개요)
- [1. Go Consumer로 충분한 경우](#1-go-consumer로-충분한-경우)
- [2. Flink가 필요해지는 시점](#2-flink가-필요해지는-시점)
- [3. 판단 체크리스트](#3-판단-체크리스트)
- [4. 단계적 도입 전략](#4-단계적-도입-전략)
- [5. 관리형 Flink 서비스 옵션](#5-관리형-flink-서비스-옵션)
- [6. 실제 도입 사례](#6-실제-도입-사례)
- [7. 비용 비교 분석](#7-비용-비교-분석)
- [8. 팀 준비도 평가](#8-팀-준비도-평가)
- [9. 실패 사례와 안티패턴](#9-실패-사례와-안티패턴)
- [10. 요약](#10-요약)

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

## 6. 실제 도입 사례

실제 서비스에서 Go Consumer를 Flink로 전환한 세 가지 사례를 살펴본다. 각 사례는 전환 전 문제, 전환 방법, 결과, 교훈으로 구성된다.

### 사례 1: 스팸 탐지 시스템

**배경**

데이팅 앱에서 단시간에 과도한 스와이프를 보내는 스팸 계정을 탐지하는 시스템이다.

**초기 구현: Go Consumer + Redis Sorted Set**

```go
// 5분 슬라이딩 윈도우 내 스와이프 카운트
func (c *SpamConsumer) handleSwipe(ctx context.Context, event SwipeEvent) error {
    key := fmt.Sprintf("swipe:5m:%s", event.SenderID)
    now := time.Now().UnixMilli()
    windowStart := now - 5*60*1000

    pipe := c.redis.Pipeline()
    pipe.ZAdd(ctx, key, redis.Z{Score: float64(now), Member: now})
    pipe.ZRemRangeByScore(ctx, key, "0", fmt.Sprintf("%d", windowStart))
    pipe.ZCard(ctx, key)
    pipe.Expire(ctx, key, 6*time.Minute)
    results, err := pipe.Exec(ctx)
    if err != nil {
        return err
    }
    count := results[2].(*redis.IntCmd).Val()
    if count >= 100 {
        c.alertCh <- SpamAlert{UserID: event.SenderID, Count: count}
    }
    return nil
}
```

**문제**

처리량이 10K events/sec를 넘으면서 Redis Sorted Set의 `ZRANGEBYSCORE` 연산이 병목이 됐다. Redis CPU가 지속적으로 80%를 초과했고, P99 지연이 50ms를 넘었다. 유저 수가 늘수록 키 수가 선형으로 증가했고, Redis 메모리도 빠르게 소진됐다.

**전환: Flink Sliding Window**

```java
DataStream<SwipeEvent> swipes = env.fromSource(kafkaSource, ...);

swipes
    .keyBy(SwipeEvent::getSenderID)
    .window(SlidingEventTimeWindows.of(
        Time.minutes(5), Time.seconds(30)
    ))
    .aggregate(new CountAggregator())
    .filter(result -> result.getCount() >= 100)
    .sinkTo(alertSink);
```

**결과**

| 항목 | 전환 전 | 전환 후 |
|------|---------|---------|
| 처리 지연 (P99) | 10초 | 500ms |
| Redis 의존성 | 있음 | 없음 (상태는 RocksDB) |
| 코드량 | 약 250줄 | 약 150줄 |
| 인프라 | Redis 클러스터 + Go | Flink TaskManager x3 |

**교훈**

Redis Sorted Set 기반 슬라이딩 윈도우는 유저 수에 비례하여 키가 증가하기 때문에 처리량 한계가 있다. Flink는 상태를 내부 RocksDB에 파티션 단위로 분산 저장하므로 수평 확장이 자연스럽다. Redis를 제거하면 운영해야 할 인프라도 줄어든다.

---

### 사례 2: 리인게이지먼트 알림

**배경**

매치가 발생한 후 30분 내에 어느 쪽도 메시지를 보내지 않으면 "첫 메시지를 보내세요" 알림을 발송하는 시스템이다.

**초기 구현: Go Consumer + cron job**

```go
// cron: 1분마다 실행
func (j *ReengageJob) run(ctx context.Context) error {
    threshold := time.Now().Add(-30 * time.Minute)
    matches, err := j.db.QueryMatchesWithoutMessage(ctx, threshold)
    if err != nil {
        return err
    }
    for _, match := range matches {
        j.notifier.Send(match.UserA, match.UserB)
        j.db.MarkNotified(ctx, match.ID)
    }
    return nil
}
```

**문제**

cron 주기가 1분이기 때문에 알림 발송 시각이 최대 1분 늦을 수 있었다. 더 심각한 문제는 매 분마다 전체 매치 테이블을 스캔하여 DB에 부하가 집중됐다는 점이다. 유저가 늘수록 스캔 대상이 선형으로 증가했다.

**전환: Flink CEP notFollowedBy**

```java
Pattern<AppEvent, ?> pattern = Pattern.<AppEvent>begin("match")
    .where(new SimpleCondition<AppEvent>() {
        public boolean filter(AppEvent e) {
            return e.getType().equals("MATCH_CREATED");
        }
    })
    .notFollowedBy("message")
    .where(new SimpleCondition<AppEvent>() {
        public boolean filter(AppEvent e) {
            return e.getType().equals("MESSAGE_SENT");
        }
    })
    .within(Time.minutes(30));

PatternStream<AppEvent> patternStream = CEP.pattern(
    events.keyBy(AppEvent::getMatchID), pattern
);

patternStream
    .select(new ReengageAlertFunction())
    .sinkTo(notificationSink);
```

**결과**

| 항목 | 전환 전 | 전환 후 |
|------|---------|---------|
| 알림 타이밍 오차 | 최대 1분 | 수 초 이내 |
| DB 부하 | 전체 스캔 (1분 주기) | 없음 (이벤트 기반) |
| cron 인프라 | 필요 | 불필요 |
| 패턴 표현력 | 제한적 | CEP 패턴 확장 용이 |

**교훈**

"A 이후 N분 내에 B가 없으면 C" 패턴은 cron + DB 스캔으로도 구현할 수 있지만, 정확도와 DB 부하 면에서 한계가 있다. Flink CEP의 `notFollowedBy`는 이 패턴을 이벤트 기반으로 정확하게 처리한다. DB 스캔이 사라지면 운영 DB의 부하가 줄어들고 알림 정확도가 높아진다.

---

### 사례 3: 실시간 피처 스토어

**배경**

추천 모델에 공급할 유저 행동 피처(최근 1시간 스와이프 수, 최근 24시간 매치율 등)를 실시간으로 집계하는 시스템이다.

**초기 구현: Spark 배치 파이프라인**

```
S3 (이벤트 로그) -> Spark Job (1시간 주기) -> Redis 피처 스토어 -> 추천 모델
```

1시간 단위 배치로 피처를 업데이트했다.

**문제**

피처가 최대 1시간 지연되었다. 유저가 방금 전 활발하게 활동했어도 추천 모델은 1시간 전 피처를 사용했다. 추천 품질이 실제 유저 상태를 반영하지 못했다.

**전환: Flink Sliding Window -> Redis 피처 스토어**

```java
DataStream<UserEvent> events = env.fromSource(kafkaSource, ...);

// 1시간 슬라이딩 윈도우, 5분 슬라이드
events
    .keyBy(UserEvent::getUserID)
    .window(SlidingEventTimeWindows.of(Time.hours(1), Time.minutes(5)))
    .aggregate(new FeatureAggregator())
    .addSink(new RedisFeatureSink());
```

```java
// RedisFeatureSink: 집계 결과를 Redis 피처 스토어에 기록
public class RedisFeatureSink extends RichSinkFunction<UserFeature> {
    public void invoke(UserFeature feature, Context context) {
        jedis.hset("feature:" + feature.getUserID(),
            "swipe_1h", String.valueOf(feature.getSwipeCount1h()),
            "match_rate_24h", String.valueOf(feature.getMatchRate24h())
        );
        jedis.expire("feature:" + feature.getUserID(), 86400);
    }
}
```

**결과**

| 항목 | 전환 전 | 전환 후 |
|------|---------|---------|
| 피처 지연 | 최대 1시간 | 최대 5분 |
| 인프라 | Spark 클러스터 + S3 | Flink + Kafka |
| 추천 CTR | 기준값 | 15% 향상 |
| 배치 운영 | 실패 시 수동 재실행 | 자동 체크포인트 복원 |

**교훈**

배치 파이프라인의 피처 지연은 추천 품질에 직접 영향을 준다. Flink로 실시간 집계하면 피처 신선도가 크게 높아진다. Flink가 상태를 관리하므로 배치 실패 시 수동 재실행 없이 자동으로 복원된다. 단, Redis 피처 스토어에 쓰는 Sink는 멱등성을 보장하도록 설계해야 한다.

---

## 7. 비용 비교 분석

Flink 도입은 기술적 결정인 동시에 비용 결정이다. 처리량별 손익분기점을 이해하고 도입 시점을 판단한다.

### Go Consumer + Redis 구성 (기준)

```
구성:
  Go Consumer: EC2 c5.xlarge x2 (온디맨드, 서울 리전)
  Redis:       ElastiCache r6g.large x1 (단일 노드)
  Kafka:       MSK m5.large x3 (별도 계산)

월간 비용 추정 (2024년 기준):
  EC2 c5.xlarge x2:    약 $280/월
  ElastiCache r6g.large: 약 $130/월
  합계:                약 $410/월
```

처리량이 늘면 Redis를 수직 확장(r6g.xlarge: $260/월)하거나 수평 확장(Cluster Mode)해야 하므로 비용이 급격히 늘어난다.

### Flink on Kubernetes 구성

```
구성:
  JobManager: EC2 c5.large x1
  TaskManager: EC2 c5.2xlarge x2 (슬롯 8개)
  S3: 체크포인트 저장 (소량)

월간 비용 추정:
  c5.large x1:     약 $70/월
  c5.2xlarge x2:   약 $560/월
  S3 (체크포인트): 약 $10/월
  합계:            약 $640/월
```

### 처리량별 손익분기점

| 처리량 | Go + Redis 비용 | Flink 비용 | 권장 |
|--------|----------------|-----------|------|
| < 5K/s | $410/월 | $640/월 | Go Consumer |
| 5K~10K/s | $410~540/월 | $640/월 | Go Consumer (최적화) |
| 10K~30K/s | $540~900/월 | $640~900/월 | 손익분기점. Flink 검토 |
| 30K/s 초과 | $900+/월 (Redis 급증) | $900~1200/월 | Flink 우위 |

위 수치는 추정값이며 실제 환경에 따라 달라진다. 핵심은 처리량이 높아질수록 Redis 비용이 급증하는 반면, Flink는 TaskManager 추가 비용이 예측 가능하다는 점이다.

### 운영 인력 비용

비용 비교에서 인프라 비용만큼 중요한 것이 운영 인력 비용이다.

**Go Consumer 직접 구현**
- 새 패턴 추가 시 Go 코드 작성, Redis 설계, 테스트 필요
- 패턴당 개발 기간: 1~2주
- Redis 장애 대응, 메모리 튜닝, LRU 설정 등 운영 오버헤드 지속 발생

**Flink 프레임워크 활용**
- 초기 학습 비용: 1~3개월
- 새 패턴 추가 시 Flink API 활용, 보일러플레이트 적음
- 패턴당 개발 기간: 3~5일 (숙련 후)
- Flink 클러스터 운영 오버헤드 (모니터링, 업그레이드) 추가

처리량이 높고 패턴이 복잡할수록 Flink의 개발 생산성 우위가 인프라 비용 차이를 상쇄한다.

---

## 8. 팀 준비도 평가

Flink 도입 성공 여부는 기술적 요인만큼 팀 준비도에 달려 있다. 도입 전에 아래 항목을 점검한다.

### 기술 역량 체크리스트

**필수 역량**

```
[ ] Java 또는 Scala 코드를 읽고 작성할 수 있다
[ ] Kafka Consumer Group, 오프셋 관리를 이해한다
[ ] 분산 시스템의 장애 유형(네트워크 파티션, 노드 장애)을 설명할 수 있다
[ ] Kubernetes Pod, Deployment, Service 기본 개념을 안다
[ ] JVM 힙 메모리, GC 기본 개념을 안다
```

**권장 역량**

```
[ ] RocksDB의 기본 동작 원리를 안다
[ ] Kafka Exactly-once semantics(트랜잭션 프로듀서)를 이해한다
[ ] Prometheus + Grafana로 메트릭을 조회한 경험이 있다
[ ] Flink 공식 문서를 읽고 예제를 실행해 본 적이 있다
```

필수 역량을 갖춘 팀원이 1명 이상 있어야 프로덕션 운영이 가능하다. 권장 역량은 운영 안정성을 높인다.

### 학습 곡선 예상

**1개월차: 기초**
- Flink 공식 문서 완독 (DataStream API, Windowing, State)
- 로컬 환경에서 예제 Job 실행
- Flink UI로 Job 모니터링 방법 파악
- 기대 수준: 간단한 집계 Job 작성 가능

**3개월차: 프로덕션 진입**
- 체크포인트, 세이브포인트 운영 경험
- Kafka Source/Sink 연동, 오프셋 관리
- CEP 패턴 1~2개 프로덕션 적용
- 기대 수준: 주요 패턴 독립 개발 가능, 장애 대응 가능

**6개월차: 최적화**
- RocksDB 상태 백엔드 튜닝 (블록 캐시, 압축 설정)
- 역압(Backpressure) 진단 및 해소
- TaskManager 리소스 최적화
- 기대 수준: 성능 이슈 독립 해결, 신규 팀원 온보딩 가능

### 최소 인력 구성

Flink를 프로덕션에서 안정적으로 운영하려면 최소 2명이 필요하다.

**개발 담당 1명**
- Flink Job 개발, 테스트, 배포
- 새 비즈니스 패턴 구현
- 코드 리뷰, 문서화

**운영 담당 1명**
- Flink 클러스터 모니터링 (Grafana 대시보드)
- 체크포인트 상태 점검, 세이브포인트 관리
- Kubernetes 리소스 관리, 업그레이드 계획

두 역할을 한 명이 겸임하면 장애 대응 시 부담이 집중된다. 팀 규모가 작다면 Amazon Managed Flink와 같은 관리형 서비스를 활용하여 운영 부담을 줄이는 방법을 검토한다.

---

## 9. 실패 사례와 안티패턴

Flink를 도입한 팀이 자주 겪는 실수를 정리한다. 같은 실수를 반복하지 않도록 한다.

### 안티패턴 1: 단순 필터링에 Flink 도입

**상황**

Kafka 이벤트를 읽어서 특정 조건에 맞는 이벤트만 다른 토픽으로 라우팅하는 파이프라인에 Flink를 도입했다.

```java
// 이런 작업에 Flink는 과하다
events
    .filter(e -> e.getCountry().equals("KR"))
    .sinkTo(krEventsSink);
```

**문제**

상태가 없는 단순 필터링이다. Go Consumer 10줄로 충분히 구현할 수 있는 작업에 Flink 클러스터 운영 복잡도를 추가했다. 장애가 나면 Flink를 모르는 팀원이 대응하기 어렵다.

**해결**

상태가 없고 패턴이 단순한 변환은 Go Consumer로 충분하다. Flink는 상태 관리, 윈도우 집계, CEP 패턴이 필요할 때 도입한다.

---

### 안티패턴 2: 상태 없는 변환에 RocksDB 사용

**상황**

Flink를 도입하면서 모든 Job에 RocksDB 상태 백엔드를 기본으로 설정했다.

```java
// 모든 Job에 무조건 RocksDB 설정
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
env.setStateBackend(new EmbeddedRocksDBStateBackend());
```

**문제**

RocksDB는 디스크 기반이라 상태 읽기/쓰기가 HashMap 백엔드보다 느리다. 상태가 없거나 소량인 Job에서는 HashMap 백엔드가 더 빠르다. 불필요한 디스크 I/O가 성능을 저하시켰다.

**해결**

상태 크기에 따라 백엔드를 선택한다.

```
상태 없음 또는 소량 (< 1GB): HashMapStateBackend (기본값)
대규모 상태 (> 1GB), 프로덕션: EmbeddedRocksDBStateBackend
```

---

### 안티패턴 3: UID 미설정으로 세이브포인트 복원 실패

**상황**

Flink Job을 개발할 때 operator UID를 설정하지 않고 배포했다. 이후 코드를 수정하고 세이브포인트에서 복원을 시도했더니 실패했다.

```java
// 잘못된 예: UID 없음
events
    .keyBy(Event::getUserID)
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .aggregate(new CountAggregator())  // UID 없음
    .sinkTo(alertSink);               // UID 없음
```

**문제**

Flink는 UID가 없으면 코드 구조를 기반으로 operator ID를 자동 생성한다. 코드를 수정하면 ID가 바뀌어 세이브포인트와 매핑이 깨진다. 프로덕션에서 Job 업그레이드 시 세이브포인트 복원이 불가능하여 상태를 처음부터 재구성해야 했다.

**해결**

모든 stateful operator에 명시적 UID를 부여한다.

```java
// 올바른 예: 명시적 UID 설정
events
    .keyBy(Event::getUserID)
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .aggregate(new CountAggregator())
    .uid("swipe-count-aggregator")       // UID 설정
    .name("Swipe Count Aggregator")
    .sinkTo(alertSink)
    .uid("alert-sink")                   // UID 설정
    .name("Alert Kafka Sink");
```

UID는 한번 정하면 바꾸지 않는다. 변경하면 세이브포인트 복원이 깨진다.

---

### 안티패턴 4: 체크포인트 간격을 너무 짧게 설정

**상황**

데이터 유실을 최소화하려고 체크포인트 간격을 10초로 설정했다.

```java
// 잘못된 예: 너무 짧은 체크포인트 간격
env.enableCheckpointing(10_000); // 10초
```

**문제**

체크포인트는 모든 TaskManager의 상태를 S3(또는 HDFS)에 저장하는 작업이다. 간격이 짧으면 체크포인트 오버헤드가 실제 데이터 처리 시간을 침범한다. 특히 RocksDB 상태가 크면 체크포인트 완료 시간이 10초를 넘어 연속 체크포인트가 쌓이며 역압(Backpressure)이 발생했다.

**해결**

체크포인트 간격은 복구 목표 시간(RTO)에 맞게 설정한다. 대부분의 스트리밍 애플리케이션에서 1~5분이 적절하다.

```java
// 올바른 예
env.enableCheckpointing(60_000);  // 1분
env.getCheckpointConfig().setCheckpointTimeout(30_000);  // 30초 내 완료 강제
env.getCheckpointConfig().setMaxConcurrentCheckpoints(1); // 동시 체크포인트 1개 제한
env.getCheckpointConfig().setMinPauseBetweenCheckpoints(30_000); // 체크포인트 간 최소 30초 대기
```

체크포인트 완료 시간을 Grafana로 모니터링하고, 완료 시간이 간격의 80%를 초과하면 간격을 늘린다.

---

## 10. 요약

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
