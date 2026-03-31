# Go Consumer vs Apache Flink — 실시간 스트림 처리 패턴 10가지 비교

## 개요

데이팅 서비스의 실시간 이벤트 처리 요구사항을 Go Kafka Consumer와 Apache Flink 1.18+ Java API로 구현했을 때의 차이를 실제 코드와 함께 비교한다.

### 공통 환경

- **Kafka 토픽**: `swipe.recorded`, `message.sent`, `report.created`, `login.attempted`, `match.created`, `subscription.changed`, `user.activity`, `location.updated`
- **Go 라이브러리**: `github.com/segmentio/kafka-go v0.4`, `github.com/redis/go-redis/v9`
- **Flink**: 1.18+ Java API, `flink-connector-kafka 3.x`
- **상태 저장소**: Redis (Go), Flink State Backend (Flink)

---

## Pattern 1: 5분 내 스와이프 50회 — 스팸 스와이프 탐지

### 패턴 설명

사용자가 5분 슬라이딩 윈도우 안에서 스와이프를 50회 이상 수행하면 스팸으로 탐지하여 사기 탐지 알림을 발생시킨다. 봇 또는 자동화 도구를 이용한 어뷰징을 차단하는 데 사용된다.

### 입력 이벤트

- Kafka 토픽: `swipe.recorded`
- 페이로드: `{ user_id, target_user_id, direction, timestamp }`

### 출력

- Kafka 토픽: `fraud.alert.triggered`
- 페이로드: `{ user_id, type: "spam_swipe", count, detected_at }`

---

### Go Consumer 구현

```go
package consumer

import (
    "context"
    "fmt"
    "time"

    "github.com/redis/go-redis/v9"
    "github.com/segmentio/kafka-go"
)

type SwipeEvent struct {
    UserID       string `json:"user_id"`
    TargetUserID string `json:"target_user_id"`
    Direction    string `json:"direction"`
    Timestamp    int64  `json:"timestamp"`
}

type FraudAlert struct {
    UserID string
    Type   string
    Count  int64
}

type FraudConsumer struct {
    reader  *kafka.Reader
    redis   *redis.Client
    alertCh chan FraudAlert
}

// handleSwipe: Redis Sorted Set으로 5분 슬라이딩 윈도우 카운터 구현
// 파이프라인으로 4개 커맨드를 단일 RTT에 실행
func (c *FraudConsumer) handleSwipe(ctx context.Context, event SwipeEvent) error {
    key := fmt.Sprintf("swipe_count:%s", event.UserID)
    now := time.Now().UnixMilli()
    windowStart := now - 5*60*1000 // 5분 전 (밀리초)

    pipe := c.redis.Pipeline()
    // 현재 타임스탬프를 score와 member 모두로 사용 (유일성 보장)
    pipe.ZAdd(ctx, key, redis.Z{
        Score:  float64(now),
        Member: fmt.Sprintf("%d", now),
    })
    // 5분 윈도우 바깥의 오래된 항목 제거
    pipe.ZRemRangeByScore(ctx, key, "0", fmt.Sprintf("%d", windowStart))
    // 현재 윈도우 내 카운트 조회
    pipe.ZCard(ctx, key)
    // 키 TTL 갱신 (윈도우보다 1분 더 유지하여 경계 처리)
    pipe.Expire(ctx, key, 6*time.Minute)

    results, err := pipe.Exec(ctx)
    if err != nil {
        return fmt.Errorf("redis pipeline exec: %w", err)
    }

    count := results[2].(*redis.IntCmd).Val()
    if count >= 50 {
        c.alertCh <- FraudAlert{
            UserID: event.UserID,
            Type:   "spam_swipe",
            Count:  count,
        }
    }
    return nil
}

func NewFraudConsumer(brokers []string, redisAddr string) *FraudConsumer {
    return &FraudConsumer{
        reader: kafka.NewReader(kafka.ReaderConfig{
            Brokers: brokers,
            Topic:   "swipe.recorded",
            GroupID: "fraud-consumer-swipe",
        }),
        redis:   redis.NewClient(&redis.Options{Addr: redisAddr}),
        alertCh: make(chan FraudAlert, 1000),
    }
}

func (c *FraudConsumer) Run(ctx context.Context) error {
    for {
        msg, err := c.reader.ReadMessage(ctx)
        if err != nil {
            return err
        }
        var event SwipeEvent
        if err := json.Unmarshal(msg.Value, &event); err != nil {
            continue
        }
        if err := c.handleSwipe(ctx, event); err != nil {
            log.Printf("handleSwipe error: %v", err)
        }
    }
}
```

**코드량**: 약 60줄 (handleSwipe 로직만 25줄)

---

### Flink 구현

```java
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.assigners.SlidingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;

public class SpamSwipeDetector {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.enableCheckpointing(60_000); // 1분마다 체크포인트

        KafkaSource<SwipeEvent> source = KafkaSource.<SwipeEvent>builder()
            .setBootstrapServers("kafka:9092")
            .setTopics("swipe.recorded")
            .setGroupId("flink-fraud-swipe")
            .setValueOnlyDeserializer(new SwipeEventDeserializer())
            .build();

        DataStream<SwipeEvent> swipes = env.fromSource(
            source,
            WatermarkStrategy.<SwipeEvent>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                .withTimestampAssigner((e, ts) -> e.getTimestamp()),
            "swipe-source"
        );

        swipes
            .keyBy(SwipeEvent::getUserId)
            // 5분 슬라이딩 윈도우, 1분마다 슬라이드
            .window(SlidingEventTimeWindows.of(Time.minutes(5), Time.minutes(1)))
            .aggregate(new CountAggregate<SwipeEvent>())
            .filter(result -> result.getCount() >= 50)
            .map(result -> FraudAlert.of(result.getKey(), "spam_swipe", result.getCount()))
            .sinkTo(KafkaSink.<FraudAlert>builder()
                .setBootstrapServers("kafka:9092")
                .setRecordSerializer(new FraudAlertSerializer("fraud.alert.triggered"))
                .build());

        env.execute("Spam Swipe Detector");
    }
}

// 재사용 가능한 범용 카운터
class CountAggregate<T> implements AggregateFunction<T, Long, CountResult> {
    @Override public Long createAccumulator() { return 0L; }
    @Override public Long add(T value, Long acc) { return acc + 1; }
    @Override public CountResult getResult(Long acc) { return new CountResult(acc); }
    @Override public Long merge(Long a, Long b) { return a + b; }
}
```

**코드량**: 약 35줄 (핵심 파이프라인 8줄)

---

### 비교 분석

| 항목 | Go Consumer | Flink |
|------|-------------|-------|
| 핵심 로직 코드량 | ~25줄 | ~8줄 |
| 외부 의존성 | Redis (필수) | 없음 (내부 상태) |
| 윈도우 정밀도 | 이벤트 도착 시각 기준 | 이벤트 타임스탬프 기준 (정확) |
| Redis 장애 시 | 카운터 유실 가능 | 체크포인트에서 복원 |
| 처리 지연 | 이벤트마다 즉시 | 윈도우 슬라이드 주기(1분)마다 |
| 운영 복잡도 | Redis 관리 필요 | Flink 클러스터 관리 필요 |

**결론**: 1K events/sec 이하이고 Redis를 이미 운영 중이라면 Go Consumer로 충분. 처리량이 높거나 이벤트 타임스탬프 정확도가 중요하면 Flink가 유리.

---

## Pattern 2: 1분 내 동일 메시지 10회 — 스팸 메시지 탐지

### 패턴 설명

같은 발신자가 동일한 내용의 메시지를 1분 안에 10회 이상 전송하면 스팸으로 탐지한다. 메시지 내용의 SHA-256 해시를 키로 사용하여 중복을 감지한다.

### 입력 이벤트

- Kafka 토픽: `message.sent`
- 페이로드: `{ sender_id, receiver_id, content, timestamp }`

### 출력

- Kafka 토픽: `fraud.alert.triggered`
- 페이로드: `{ user_id, type: "spam_message", count, detected_at }`

---

### Go Consumer 구현

```go
package consumer

import (
    "context"
    "crypto/sha256"
    "fmt"
    "time"

    "github.com/redis/go-redis/v9"
)

type MessageEvent struct {
    SenderID   string `json:"sender_id"`
    ReceiverID string `json:"receiver_id"`
    Content    string `json:"content"`
    Timestamp  int64  `json:"timestamp"`
}

func (c *FraudConsumer) handleMessage(ctx context.Context, event MessageEvent) error {
    // 메시지 내용의 SHA-256 해시 앞 8바이트를 키로 사용
    hash := sha256.Sum256([]byte(event.Content))
    key := fmt.Sprintf("msg_dup:%s:%x", event.SenderID, hash[:8])

    // INCR + 만료 설정 (atomic하지 않지만 카운터 목적으로 충분)
    count, err := c.redis.Incr(ctx, key).Result()
    if err != nil {
        return fmt.Errorf("redis incr: %w", err)
    }
    // 첫 번째 증가 시에만 만료 설정 (1분 tumbling window 근사)
    if count == 1 {
        c.redis.Expire(ctx, key, time.Minute)
    }
    if count >= 10 {
        c.alertCh <- FraudAlert{
            UserID: event.SenderID,
            Type:   "spam_message",
            Count:  count,
        }
    }
    return nil
}
```

**코드량**: 약 20줄

**주의**: `Incr`와 `Expire`가 atomic하지 않아 레이스 컨디션 가능성이 있다. Lua 스크립트로 해결할 수 있으나 복잡도가 올라간다.

---

### Flink 구현

```java
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;

public class SpamMessageDetector {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<MessageEvent> messages = env.fromSource(
            buildKafkaSource("message.sent", new MessageEventDeserializer()),
            WatermarkStrategy.<MessageEvent>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                .withTimestampAssigner((e, ts) -> e.getTimestamp()),
            "message-source"
        );

        messages
            // (발신자ID + 메시지 해시)로 그룹화 — 동일 내용 중복 집계
            .keyBy(msg -> msg.getSenderId() + ":" + sha256Hex(msg.getContent()))
            // 1분 텀블링 윈도우
            .window(TumblingEventTimeWindows.of(Time.minutes(1)))
            .aggregate(new CountAggregate<MessageEvent>())
            .filter(result -> result.getCount() >= 10)
            .map(result -> FraudAlert.of(
                result.getKey().split(":")[0], // sender_id 추출
                "spam_message",
                result.getCount()
            ))
            .sinkTo(buildKafkaSink("fraud.alert.triggered"));

        env.execute("Spam Message Detector");
    }

    private static String sha256Hex(String content) {
        // MessageDigest로 SHA-256 계산 후 hex string 반환
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] hash = md.digest(content.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(hash, 0, 8); // 앞 8바이트만
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    }
}
```

**코드량**: 약 30줄 (핵심 파이프라인 10줄)

---

### 비교 분석

| 항목 | Go Consumer | Flink |
|------|-------------|-------|
| 구현 복잡도 | 낮음 | 낮음 |
| Atomic 보장 | Lua 스크립트 필요 | 내장 (윈도우 집계) |
| 윈도우 방식 | 첫 이벤트 기준 만료 (근사) | 정확한 텀블링 윈도우 |
| 메모리 사용 | Redis 메모리 | Flink 힙/RocksDB |

**결론**: 이 패턴은 Go로 충분히 구현 가능. 단, atomic 보장이 필요하면 Lua 스크립트 추가 필요.

---

## Pattern 3: 24시간 내 신고 3회 — 자동 제재

### 패턴 설명

특정 사용자가 24시간 안에 3회 이상 신고되면 자동으로 계정을 제재한다. 제재는 fraud-service API를 통해 적용된다.

### 입력 이벤트

- Kafka 토픽: `report.created`
- 페이로드: `{ reporter_id, target_user_id, reason, timestamp }`

### 출력

- fraud-service API 호출: `POST /internal/users/{id}/suspend`

---

### Go Consumer 구현

```go
package consumer

import (
    "context"
    "fmt"
    "time"

    "github.com/redis/go-redis/v9"
)

type ReportEvent struct {
    ReporterID   string `json:"reporter_id"`
    TargetUserID string `json:"target_user_id"`
    Reason       string `json:"reason"`
    Timestamp    int64  `json:"timestamp"`
}

type FraudServiceClient interface {
    AutoSuspend(ctx context.Context, userID, reason string) error
}

func (c *FraudConsumer) handleReport(ctx context.Context, event ReportEvent) error {
    key := fmt.Sprintf("report_count:%s", event.TargetUserID)

    count, err := c.redis.Incr(ctx, key).Result()
    if err != nil {
        return fmt.Errorf("redis incr: %w", err)
    }
    // 첫 신고 시 24시간 만료 설정
    if count == 1 {
        c.redis.Expire(ctx, key, 24*time.Hour)
    }
    if count >= 3 {
        if err := c.fraudClient.AutoSuspend(ctx, event.TargetUserID, "24시간 내 3회 이상 신고"); err != nil {
            return fmt.Errorf("auto suspend user %s: %w", event.TargetUserID, err)
        }
    }
    return nil
}
```

**코드량**: 약 20줄

---

### Flink 구현

```java
public class AutoSuspendDetector {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<ReportEvent> reports = env.fromSource(
            buildKafkaSource("report.created", new ReportEventDeserializer()),
            WatermarkStrategy.<ReportEvent>forBoundedOutOfOrderness(Duration.ofMinutes(1))
                .withTimestampAssigner((e, ts) -> e.getTimestamp()),
            "report-source"
        );

        reports
            .keyBy(ReportEvent::getTargetUserId)
            // 24시간 텀블링 윈도우
            .window(TumblingEventTimeWindows.of(Time.hours(24)))
            .aggregate(new CountAggregate<ReportEvent>())
            .filter(result -> result.getCount() >= 3)
            // HTTP 클라이언트로 fraud-service API 호출
            .process(new AutoSuspendFunction())
            .name("auto-suspend-sink");

        env.execute("Auto Suspend Detector");
    }
}

class AutoSuspendFunction extends ProcessFunction<CountResult, Void> {
    private transient OkHttpClient httpClient;

    @Override
    public void open(Configuration parameters) {
        this.httpClient = new OkHttpClient();
    }

    @Override
    public void processElement(CountResult result, Context ctx, Collector<Void> out) throws Exception {
        String userId = result.getKey();
        RequestBody body = RequestBody.create(
            String.format("{\"reason\": \"24시간 내 %d회 신고\"}", result.getCount()),
            MediaType.get("application/json")
        );
        Request request = new Request.Builder()
            .url("http://fraud-service/internal/users/" + userId + "/suspend")
            .post(body)
            .build();
        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new RuntimeException("suspend failed: " + response.code());
            }
        }
    }
}
```

**코드량**: 약 45줄

---

### 비교 분석

| 항목 | Go Consumer | Flink |
|------|-------------|-------|
| 코드량 | 적음 | 많음 (ProcessFunction 필요) |
| 윈도우 정확도 | 첫 신고 기준 만료 (근사) | 이벤트 타임스탬프 기준 정확 |
| 외부 API 호출 | Go HTTP 클라이언트 | Flink ProcessFunction 내 HTTP |
| 장애 복구 | Redis 기반 | 체크포인트 기반 |

**결론**: 이 패턴은 Go Consumer가 더 간결. 단순 카운터 + API 호출은 Go에 적합.

---

## Pattern 4: 10분 내 5개 디바이스 로그인 — 계정 탈취 탐지

### 패턴 설명

같은 계정이 10분 안에 5개 이상의 서로 다른 디바이스에서 로그인 시도가 발생하면 계정 탈취로 탐지한다. 단순 카운트가 아니라 **고유 디바이스 수**를 계산해야 한다.

### 입력 이벤트

- Kafka 토픽: `login.attempted`
- 페이로드: `{ user_id, device_id, device_type, ip_address, timestamp }`

### 출력

- Kafka 토픽: `fraud.alert.triggered`
- 페이로드: `{ user_id, type: "account_takeover", unique_devices, detected_at }`

---

### Go Consumer 구현

```go
package consumer

import (
    "context"
    "fmt"
    "time"

    "github.com/redis/go-redis/v9"
)

type LoginEvent struct {
    UserID     string `json:"user_id"`
    DeviceID   string `json:"device_id"`
    DeviceType string `json:"device_type"`
    IPAddress  string `json:"ip_address"`
    Timestamp  int64  `json:"timestamp"`
}

func (c *FraudConsumer) handleLogin(ctx context.Context, event LoginEvent) error {
    key := fmt.Sprintf("login_devices:%s", event.UserID)
    now := time.Now().UnixMilli()
    windowStart := now - 10*60*1000 // 10분 전

    pipe := c.redis.Pipeline()
    // DeviceID를 member로, 타임스탬프를 score로 저장 — 자연스럽게 중복 제거
    // 같은 DeviceID가 오면 score만 업데이트됨 (Sorted Set 특성)
    pipe.ZAdd(ctx, key, redis.Z{
        Score:  float64(now),
        Member: event.DeviceID,
    })
    // 10분 윈도우 바깥 항목 제거
    pipe.ZRemRangeByScore(ctx, key, "0", fmt.Sprintf("%d", windowStart))
    // 윈도우 내 고유 디바이스 수 = Sorted Set의 카디널리티
    pipe.ZCard(ctx, key)
    pipe.Expire(ctx, key, 11*time.Minute)

    results, err := pipe.Exec(ctx)
    if err != nil {
        return fmt.Errorf("redis pipeline: %w", err)
    }

    uniqueDevices := results[2].(*redis.IntCmd).Val()
    if uniqueDevices >= 5 {
        c.alertCh <- FraudAlert{
            UserID: event.UserID,
            Type:   "account_takeover",
            Count:  uniqueDevices,
        }
    }
    return nil
}
```

**코드량**: 약 30줄

**핵심**: Redis Sorted Set에서 `member`를 `DeviceID`로 설정하면 같은 디바이스가 여러 번 로그인해도 하나로 카운트된다. `ZCard`가 곧 고유 디바이스 수다.

---

### Flink 구현

```java
import org.apache.flink.api.common.functions.AggregateFunction;
import java.util.HashSet;
import java.util.Set;

public class AccountTakeoverDetector {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<LoginEvent> logins = env.fromSource(
            buildKafkaSource("login.attempted", new LoginEventDeserializer()),
            WatermarkStrategy.<LoginEvent>forBoundedOutOfOrderness(Duration.ofSeconds(10))
                .withTimestampAssigner((e, ts) -> e.getTimestamp()),
            "login-source"
        );

        logins
            .keyBy(LoginEvent::getUserId)
            // 10분 슬라이딩 윈도우, 1분마다 슬라이드
            .window(SlidingEventTimeWindows.of(Time.minutes(10), Time.minutes(1)))
            .aggregate(new DistinctDeviceCounter())
            .filter(result -> result.getUniqueDevices() >= 5)
            .map(result -> FraudAlert.of(result.getUserId(), "account_takeover", result.getUniqueDevices()))
            .sinkTo(buildKafkaSink("fraud.alert.triggered"));

        env.execute("Account Takeover Detector");
    }
}

// 고유 DeviceID Set을 누산기로 사용
class DistinctDeviceCounter
    implements AggregateFunction<LoginEvent, Set<String>, DeviceCountResult> {

    @Override
    public Set<String> createAccumulator() {
        return new HashSet<>();
    }

    @Override
    public Set<String> add(LoginEvent event, Set<String> accumulator) {
        accumulator.add(event.getDeviceId());
        return accumulator;
    }

    @Override
    public DeviceCountResult getResult(Set<String> accumulator) {
        return new DeviceCountResult(accumulator.size());
    }

    @Override
    public Set<String> merge(Set<String> a, Set<String> b) {
        a.addAll(b);
        return a;
    }
}
```

**코드량**: 약 40줄 (`DistinctDeviceCounter` 포함)

---

### 비교 분석

| 항목 | Go Consumer | Flink |
|------|-------------|-------|
| 고유 카운터 구현 | Redis Sorted Set member 활용 (우아함) | `HashSet<String>` 누산기 |
| 메모리 사용 | Redis에 분산 | Flink 힙 (큰 Set이면 RocksDB로) |
| 슬라이딩 윈도우 | 수동 ZRemRangeByScore | 내장 지원 |
| 확장성 | Redis 메모리 한계 | TaskManager 병렬화 |

**결론**: Go의 Redis Sorted Set + DeviceID를 member로 사용하는 방식은 우아하고 효율적. 중간 난이도에서 Go가 충분히 경쟁력 있음.

---

## Pattern 5: 최근 1시간 스와이프 수 집계 — 추천 피처

### 패턴 설명

추천 알고리즘에서 사용하는 실시간 피처를 계산한다. 각 사용자의 최근 1시간 스와이프 수를 실시간으로 집계하여 피처 스토어(Redis Hash)에 저장한다.

### 입력 이벤트

- Kafka 토픽: `swipe.recorded`
- 페이로드: `{ user_id, target_user_id, direction, timestamp }`

### 출력

- Redis Hash 업데이트: `features:{user_id}` → `swipe_count_1h = N`

---

### Go Consumer 구현

```go
func (c *FeatureConsumer) handleSwipe(ctx context.Context, event SwipeEvent) error {
    key := fmt.Sprintf("feature:swipe_1h:%s", event.UserID)
    now := time.Now().UnixMilli()
    windowStart := now - 3600*1000 // 1시간 전

    pipe := c.redis.Pipeline()
    // 현재 타임스탬프를 score/member로 저장
    pipe.ZAdd(ctx, key, redis.Z{
        Score:  float64(now),
        Member: fmt.Sprintf("%d", now),
    })
    // 1시간 윈도우 바깥 항목 제거
    pipe.ZRemRangeByScore(ctx, key, "0", fmt.Sprintf("%d", windowStart))
    // 윈도우 내 카운트
    pipe.ZCard(ctx, key)
    pipe.Expire(ctx, key, 2*time.Hour)
    results, err := pipe.Exec(ctx)
    if err != nil {
        return fmt.Errorf("redis pipeline: %w", err)
    }

    count := results[2].(*redis.IntCmd).Val()
    // 피처 스토어(Redis Hash)에 실시간 업데이트
    featureKey := fmt.Sprintf("features:%s", event.UserID)
    return c.redis.HSet(ctx, featureKey, "swipe_count_1h", count).Err()
}
```

**코드량**: 약 22줄

**문제점**: 이벤트가 올 때마다 Redis 5개 커맨드 실행. 1K events/sec이면 초당 5K Redis 커맨드 발생.

---

### Flink 구현

```java
public class FeatureStoreSync {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<SwipeEvent> swipes = env.fromSource(
            buildKafkaSource("swipe.recorded", new SwipeEventDeserializer()),
            WatermarkStrategy.<SwipeEvent>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                .withTimestampAssigner((e, ts) -> e.getTimestamp()),
            "swipe-source"
        );

        swipes
            .keyBy(SwipeEvent::getUserId)
            // 1시간 슬라이딩 윈도우, 1분마다 갱신
            .window(SlidingEventTimeWindows.of(Time.hours(1), Time.minutes(1)))
            .aggregate(new CountAggregate<SwipeEvent>())
            // 윈도우 종료마다 Redis에 1번만 쓰기
            .addSink(new RedisSink<>(
                new FlinkJedisPoolConfig.Builder().setHost("redis").build(),
                new FeatureStoreMapper()
            ));

        env.execute("Feature Store Sync");
    }
}

class FeatureStoreMapper implements RedisMapper<CountResult> {

    @Override
    public RedisCommandDescription getCommandDescription() {
        return new RedisCommandDescription(RedisCommand.HSET, null);
    }

    @Override
    public String getKeyFromData(CountResult result) {
        return "features:" + result.getKey(); // features:{user_id}
    }

    @Override
    public String getValueFromData(CountResult result) {
        return String.valueOf(result.getCount()); // swipe_count_1h 값
    }

    // 필드명 지정 (HSET key field value)
    public String getFieldFromData(CountResult result) {
        return "swipe_count_1h";
    }
}
```

**코드량**: 약 40줄

---

### 비교 분석

| 항목 | Go Consumer | Flink |
|------|-------------|-------|
| Redis 쓰기 빈도 | 이벤트마다 5 커맨드 | 윈도우 종료마다 1 커맨드 |
| 피처 신선도 | 즉시 반영 | 최대 1분 지연 |
| 처리량 확장성 | Redis 병목 가능 | Flink 병렬화로 선형 확장 |
| 코드 복잡도 | 보통 | 보통 (RedisMapper 필요) |

**결론**: 피처 신선도가 실시간(초 단위)이어야 하면 Go, 1분 지연이 허용되면 Flink가 Redis 부하 측면에서 유리. 10K events/sec 이상이면 Flink로 전환 권장.

---

## Pattern 6: 매치 후 30분 내 대화 미시작 — 리인게이지먼트

### 패턴 설명

두 사용자가 매치된 후 30분 안에 대화를 시작하지 않으면 리인게이지먼트 푸시 알림을 발송한다. 이 패턴은 **"A 이벤트 이후 B 이벤트가 없으면 C를 실행"** 구조로, Go에서는 구현이 복잡하고 Flink CEP가 결정적 우위를 보이는 핵심 패턴이다.

### 입력 이벤트

- Kafka 토픽 1: `match.created` → `{ user1_id, user2_id, match_id, timestamp }`
- Kafka 토픽 2: `message.sent` → `{ sender_id, receiver_id, match_id, timestamp }`

### 출력

- notification-service API 호출: 리인게이지먼트 푸시 알림

---

### Go Consumer 구현

```go
package consumer

import (
    "context"
    "fmt"
    "time"

    "github.com/redis/go-redis/v9"
    "github.com/segmentio/kafka-go"
)

type GrowthConsumer struct {
    matchReader   *kafka.Reader
    messageReader *kafka.Reader
    redis         *redis.Client
    notifClient   NotificationClient
}

// 1단계: match.created 이벤트 → Redis에 "대기 중" 등록 (TTL = 30분)
func (c *GrowthConsumer) handleMatch(ctx context.Context, event MatchEvent) error {
    // 두 방향 모두 키 생성 (message.sent에서 방향을 모름)
    key1 := fmt.Sprintf("match_no_chat:%s:%s", event.User1ID, event.User2ID)
    key2 := fmt.Sprintf("match_no_chat:%s:%s", event.User2ID, event.User1ID)

    pipe := c.redis.Pipeline()
    pipe.Set(ctx, key1, event.MatchID, 30*time.Minute)
    pipe.Set(ctx, key2, event.MatchID, 30*time.Minute)
    _, err := pipe.Exec(ctx)
    return err
}

// 2단계: message.sent 이벤트 → "대기 중" 해제
func (c *GrowthConsumer) handleMessage(ctx context.Context, event MessageEvent) error {
    key1 := fmt.Sprintf("match_no_chat:%s:%s", event.SenderID, event.ReceiverID)
    key2 := fmt.Sprintf("match_no_chat:%s:%s", event.ReceiverID, event.SenderID)
    return c.redis.Del(ctx, key1, key2).Err()
}

// 3단계: 별도 폴링 goroutine — Redis keyspace notification은 신뢰성 보장 안 됨
// Redis 6.x에서 keyspace notification 설정 필요: CONFIG SET notify-keyspace-events Ex
// 대안: 매 30초마다 만료된 키를 스캔하는 cron goroutine
// 문제: Redis SCAN은 전체 키를 순회 — 유저 100만이면 성능 저하
func (c *GrowthConsumer) pollExpiredMatches(ctx context.Context) {
    ticker := time.NewTicker(30 * time.Second)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            // SCAN으로 만료 예정 키 찾기 (실제로는 만료 전 미리 알 수 없음)
            // keyspace notification subscribe로 대체 필요
            // 아래는 keyspace notification 방식
            pubsub := c.redis.PSubscribe(ctx, "__keyevent@0__:expired")
            for msg := range pubsub.Channel() {
                key := msg.Payload
                if strings.HasPrefix(key, "match_no_chat:") {
                    parts := strings.Split(key, ":")
                    if len(parts) == 3 {
                        userID := parts[1]
                        c.notifClient.SendReengagement(ctx, userID, "매치 후 대화를 시작해보세요!")
                    }
                }
            }
        }
    }
}

// RunMultiTopicConsumer: 두 토픽을 동시에 소비
func (c *GrowthConsumer) Run(ctx context.Context) error {
    eg, ctx := errgroup.WithContext(ctx)

    eg.Go(func() error { return c.consumeMatches(ctx) })
    eg.Go(func() error { return c.consumeMessages(ctx) })
    eg.Go(func() error { c.pollExpiredMatches(ctx); return nil })

    return eg.Wait()
}
```

**코드량**: 약 65줄 이상

**문제점**:
1. Redis keyspace notification은 at-most-once 보장 — 알림 유실 가능
2. 두 Kafka 토픽을 별도 goroutine으로 소비하는 복잡한 구조
3. 매치-메시지 간의 ordering 보장 없음 (메시지가 먼저 도착할 수 있음)
4. 실제로 신뢰성 있는 구현을 하려면 PostgreSQL 폴링 cron job이 필요

---

### Flink 구현 (CEP)

```java
import org.apache.flink.cep.CEP;
import org.apache.flink.cep.PatternSelectFunction;
import org.apache.flink.cep.pattern.Pattern;
import org.apache.flink.cep.pattern.conditions.SimpleCondition;

public class ReengagementDetector {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // 두 토픽을 하나의 통합 이벤트 스트림으로 병합
        DataStream<AppEvent> matches = env
            .fromSource(buildKafkaSource("match.created", new MatchEventDeserializer()), ...)
            .map(e -> AppEvent.fromMatch(e));

        DataStream<AppEvent> messages = env
            .fromSource(buildKafkaSource("message.sent", new MessageEventDeserializer()), ...)
            .map(e -> AppEvent.fromMessage(e));

        DataStream<AppEvent> events = matches.union(messages);

        // CEP 패턴: "match 이벤트 이후 30분 내에 message 이벤트가 없으면"
        Pattern<AppEvent, ?> noConversationPattern = Pattern
            .<AppEvent>begin("match")
            .where(new SimpleCondition<AppEvent>() {
                @Override
                public boolean filter(AppEvent event) {
                    return event.getType().equals("match.created");
                }
            })
            .notFollowedBy("message") // 핵심: 이후에 message가 없으면
            .where(new SimpleCondition<AppEvent>() {
                @Override
                public boolean filter(AppEvent event) {
                    return event.getType().equals("message.sent");
                }
            })
            .within(Time.minutes(30)); // 30분 윈도우 내

        PatternStream<AppEvent> patternStream = CEP.pattern(
            events.keyBy(AppEvent::getUserId), // 유저별 패턴 매칭
            noConversationPattern
        );

        patternStream
            .select(new PatternSelectFunction<AppEvent, ReengagementAlert>() {
                @Override
                public ReengagementAlert select(Map<String, List<AppEvent>> pattern) {
                    AppEvent matchEvent = pattern.get("match").get(0);
                    return new ReengagementAlert(
                        matchEvent.getUserId(),
                        "매치 후 대화를 시작해보세요!",
                        matchEvent.getMatchId()
                    );
                }
            })
            .process(new SendPushNotificationFunction()); // notification-service 호출

        env.execute("Reengagement Detector");
    }
}
```

**코드량**: 약 50줄 (CEP 패턴 선언 15줄)

---

### 비교 분석

| 항목 | Go Consumer | Flink CEP |
|------|-------------|-----------|
| 코드량 | 65줄+ | 50줄 |
| 신뢰성 | Redis keyspace notification (at-most-once) | Flink 체크포인트 (exactly-once) |
| 구현 복잡도 | 매우 높음 (멀티 토픽 + cron + Redis TTL) | 낮음 (선언적 패턴) |
| 정확한 30분 보장 | 어려움 (Redis TTL 근사) | 이벤트 타임스탬프 기준 정확 |
| 이벤트 순서 처리 | 수동 구현 필요 | 내장 (워터마크 기반) |
| 운영 복잡도 | 낮음 | 높음 (Flink 클러스터) |

**결론**: 이 패턴이 **Flink CEP의 핵심 가치를 보여주는 사례**다. Go로 신뢰성 있게 구현하려면 PostgreSQL 폴링 cron job까지 필요하여 복잡도가 크게 높아진다. 이 단일 패턴 때문에 Flink 도입을 검토할 만하다.

---

## Pattern 7: 3일 연속 비활동 — 이탈 위험 감지

### 패턴 설명

사용자가 3일 이상 앱을 사용하지 않으면 이탈 위험 신호로 분류하고 리인게이지먼트 캠페인을 발동한다. 모든 활동 이벤트(스와이프, 메시지, 로그인 등)를 수신하여 마지막 활동 시각을 추적한다.

### 입력 이벤트

- Kafka 토픽: `user.activity` (모든 활동 이벤트 통합)
- 페이로드: `{ user_id, activity_type, timestamp }`

### 출력

- growth-service API: 이탈 위험 세그먼트에 사용자 추가

---

### Go Consumer 구현

```go
package consumer

import (
    "context"
    "fmt"
    "time"

    "github.com/redis/go-redis/v9"
)

// 1단계: 모든 활동 이벤트에서 마지막 활동 시각 업데이트
func (c *GrowthConsumer) handleActivity(ctx context.Context, event ActivityEvent) error {
    key := fmt.Sprintf("last_active:%s", event.UserID)
    // 7일 TTL — 7일 이상 비활동이면 키 자체가 사라짐
    return c.redis.Set(ctx, key, time.Now().Unix(), 7*24*time.Hour).Err()
}

// 2단계: 별도 cron goroutine — 매시간 전체 유저 스캔
// 문제: 유저 100만 명이면 SCAN에 수십 초 소요
// 문제: SCAN은 순간 스냅샷이 아니라 커서 기반 — 스캔 중 변경된 키 놓칠 수 있음
func (j *ChurnDetectionJob) Run(ctx context.Context) error {
    ticker := time.NewTicker(time.Hour)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        case <-ticker.C:
            if err := j.scanAndDetect(ctx); err != nil {
                log.Printf("churn scan error: %v", err)
            }
        }
    }
}

func (j *ChurnDetectionJob) scanAndDetect(ctx context.Context) error {
    threshold := time.Now().Unix() - int64(3*24*time.Hour.Seconds()) // 3일 전
    cursor := uint64(0)

    for {
        keys, nextCursor, err := j.redis.Scan(ctx, cursor, "last_active:*", 1000).Result()
        if err != nil {
            return fmt.Errorf("redis scan: %w", err)
        }

        for _, key := range keys {
            val, err := j.redis.Get(ctx, key).Int64()
            if err != nil {
                continue
            }
            if val < threshold {
                userID := strings.TrimPrefix(key, "last_active:")
                j.growthClient.AddChurnRisk(ctx, userID)
            }
        }

        cursor = nextCursor
        if cursor == 0 {
            break
        }
    }
    return nil
}
```

**코드량**: 약 50줄

**문제점**:
- 이벤트 기반이 아닌 cron 기반 → 탐지 지연 최대 1시간
- 유저 100만이면 Redis SCAN에 수십 초 소요
- SCAN 중 변경된 키를 놓칠 수 있음

---

### Flink 구현

```java
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;

public class ChurnDetector {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<ActivityEvent> activities = env.fromSource(
            buildKafkaSource("user.activity", new ActivityEventDeserializer()),
            WatermarkStrategy.<ActivityEvent>forBoundedOutOfOrderness(Duration.ofMinutes(5))
                .withTimestampAssigner((e, ts) -> e.getTimestamp()),
            "activity-source"
        );

        activities
            .keyBy(ActivityEvent::getUserId)
            // 이벤트 기반 타이머 — 정확한 3일 감지
            .process(new ChurnDetectionFunction())
            .name("churn-detector");

        env.execute("Churn Detector");
    }
}

class ChurnDetectionFunction
    extends KeyedProcessFunction<String, ActivityEvent, ChurnAlert> {

    // 유저별 마지막 활동 시각 (Flink 관리 상태)
    private ValueState<Long> lastActivityState;
    // 등록된 타이머 시각 (중복 타이머 방지)
    private ValueState<Long> timerState;

    @Override
    public void open(Configuration parameters) {
        lastActivityState = getRuntimeContext().getState(
            new ValueStateDescriptor<>("lastActivity", Long.class)
        );
        timerState = getRuntimeContext().getState(
            new ValueStateDescriptor<>("timer", Long.class)
        );
    }

    @Override
    public void processElement(ActivityEvent event, Context ctx, Collector<ChurnAlert> out)
        throws Exception {

        long eventTime = event.getTimestamp();
        lastActivityState.update(eventTime);

        // 기존 타이머 취소 후 새 타이머 등록 (활동이 생기면 타이머 리셋)
        Long existingTimer = timerState.value();
        if (existingTimer != null) {
            ctx.timerService().deleteEventTimeTimer(existingTimer);
        }

        // 현재 이벤트로부터 정확히 3일 후 타이머 등록
        long timerTime = eventTime + Duration.ofDays(3).toMillis();
        ctx.timerService().registerEventTimeTimer(timerTime);
        timerState.update(timerTime);
    }

    @Override
    public void onTimer(long timestamp, OnTimerContext ctx, Collector<ChurnAlert> out)
        throws Exception {

        Long lastActivity = lastActivityState.value();
        if (lastActivity == null) {
            return;
        }

        // 타이머 발동 시 실제로 3일 이상 지났는지 재확인 (중간에 활동이 있었을 수 있음)
        if (timestamp - lastActivity >= Duration.ofDays(3).toMillis()) {
            out.collect(new ChurnAlert(
                ctx.getCurrentKey(),
                "3일 이상 비활동",
                timestamp
            ));
        }
        timerState.clear();
    }
}
```

**코드량**: 약 65줄

---

### 비교 분석

| 항목 | Go Consumer | Flink |
|------|-------------|-------|
| 탐지 방식 | cron 기반 (최대 1시간 지연) | 이벤트 타임스탬프 기반 타이머 (정확) |
| 확장성 | Redis SCAN → 유저 수에 비례한 부하 | 유저별 상태 분산, 선형 확장 |
| 타이머 정확도 | 근사 (cron 주기 내 오차) | 이벤트 타임스탬프 기준 정확 |
| 활동 시 타이머 리셋 | Redis TTL 갱신 (간단) | timerState 관리 (복잡) |
| 장애 복구 | Redis 데이터 보존 | Flink 체크포인트 |

**결론**: Go는 구현이 단순하지만 cron 기반이라 탐지 지연과 확장성 문제가 있음. Flink는 타이머 관리 코드가 복잡하지만 정확하고 확장성이 좋음. 유저 수가 100만 이상이면 Flink 권장.

---

## Pattern 8: 구독 만료 7일 전 — 갱신 알림

### 패턴 설명

사용자의 구독(PLUS/GOLD/PLATINUM)이 만료되기 7일 전에 갱신 안내 푸시 알림을 발송한다. 구독 변경 이벤트를 수신하여 만료일 기반 타이머를 등록한다.

### 입력 이벤트

- Kafka 토픽: `subscription.changed`
- 페이로드: `{ user_id, plan, expires_at, timestamp }`

### 출력

- notification-service API: 갱신 안내 푸시 알림

---

### Go Consumer 구현

```go
package consumer

import (
    "context"
    "time"
    "fmt"
)

type SubscriptionEvent struct {
    UserID    string    `json:"user_id"`
    Plan      string    `json:"plan"`
    ExpiresAt time.Time `json:"expires_at"`
    Timestamp int64     `json:"timestamp"`
}

// 구독 이벤트 수신 시 만료일을 Redis에 저장
func (c *SubscriptionConsumer) handleSubscriptionChanged(ctx context.Context, event SubscriptionEvent) error {
    if event.Plan == "FREE" {
        return nil // 무료 플랜은 만료 없음
    }
    key := fmt.Sprintf("sub_expires:%s", event.UserID)
    // 만료일까지의 TTL로 키 저장
    ttl := time.Until(event.ExpiresAt)
    if ttl <= 0 {
        return nil
    }
    return c.redis.Set(ctx, key, event.ExpiresAt.Unix(), ttl).Err()
}

// 별도 cron — 매일 자정에 7일 후 만료 예정 유저 스캔
// 문제: 구독자 50만이면 DB 스캔이 무거움
// 문제: Redis TTL은 7일 전 정확 알림을 보내기 어려움 (만료 시점만 알 수 있음)
func (j *RenewalReminderJob) Run(ctx context.Context) error {
    ticker := time.NewTicker(24 * time.Hour)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        case <-ticker.C:
            // DB에서 7일 후 만료 예정 구독자 조회
            users, err := j.db.QueryContext(ctx,
                `SELECT user_id FROM subscriptions
                 WHERE expires_at BETWEEN NOW() + INTERVAL '6 days' AND NOW() + INTERVAL '7 days'
                 AND plan != 'FREE'`)
            if err != nil {
                log.Printf("db query: %v", err)
                continue
            }
            defer users.Close()
            for users.Next() {
                var userID string
                users.Scan(&userID)
                j.notifClient.SendRenewalReminder(ctx, userID)
            }
        }
    }
}
```

**코드량**: 약 45줄

**문제점**: Redis TTL로는 "만료 7일 전" 시점을 정확히 알기 어려움. 결국 DB 폴링 cron이 필요.

---

### Flink 구현

```java
public class RenewalReminderDetector {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<SubscriptionEvent> subscriptions = env.fromSource(
            buildKafkaSource("subscription.changed", new SubscriptionEventDeserializer()),
            WatermarkStrategy.<SubscriptionEvent>forBoundedOutOfOrderness(Duration.ofMinutes(1))
                .withTimestampAssigner((e, ts) -> e.getTimestamp()),
            "subscription-source"
        );

        subscriptions
            .filter(e -> !e.getPlan().equals("FREE")) // 무료 플랜 제외
            .keyBy(SubscriptionEvent::getUserId)
            .process(new RenewalReminderFunction())
            .process(new SendPushNotificationFunction());

        env.execute("Renewal Reminder");
    }
}

class RenewalReminderFunction
    extends KeyedProcessFunction<String, SubscriptionEvent, RenewalReminder> {

    private ValueState<Long> timerState;
    private ValueState<String> planState;

    @Override
    public void open(Configuration parameters) {
        timerState = getRuntimeContext().getState(
            new ValueStateDescriptor<>("renewalTimer", Long.class)
        );
        planState = getRuntimeContext().getState(
            new ValueStateDescriptor<>("plan", String.class)
        );
    }

    @Override
    public void processElement(SubscriptionEvent event, Context ctx, Collector<RenewalReminder> out)
        throws Exception {

        // 기존 타이머 취소
        Long existing = timerState.value();
        if (existing != null) {
            ctx.timerService().deleteEventTimeTimer(existing);
        }

        // 만료 7일 전 타이머 등록
        long reminderTime = event.getExpiresAt() - Duration.ofDays(7).toMillis();
        if (reminderTime > ctx.timestamp()) {
            ctx.timerService().registerEventTimeTimer(reminderTime);
            timerState.update(reminderTime);
            planState.update(event.getPlan());
        }
    }

    @Override
    public void onTimer(long timestamp, OnTimerContext ctx, Collector<RenewalReminder> out)
        throws Exception {
        out.collect(new RenewalReminder(ctx.getCurrentKey(), planState.value()));
        timerState.clear();
        planState.clear();
    }
}
```

**코드량**: 약 55줄

---

### 비교 분석

| 항목 | Go Consumer | Flink |
|------|-------------|-------|
| 알림 방식 | DB 폴링 cron (하루 1회) | 이벤트 타이머 (정확한 7일 전) |
| DB 부하 | 매일 전체 구독자 스캔 | 없음 (이벤트 기반) |
| 구독 갱신 시 처리 | cron에서 자동 처리 (재조회) | 타이머 취소 후 재등록 |
| 알림 정확도 | 최대 24시간 오차 | 이벤트 타임스탬프 기준 정확 |

**결론**: 알림 정확도가 중요하지 않고 DB를 이미 운영 중이라면 Go cron이 단순. 이벤트 기반 정확한 시점이 필요하면 Flink 타이머.

---

## Pattern 9: 1분 이벤트 유실률 1% 초과 — 파이프라인 장애 감지

### 패턴 설명

Kafka 파이프라인에서 1분 동안 처리된 이벤트 수와 예상 이벤트 수를 비교하여 유실률이 1%를 초과하면 알림을 발생시킨다. 데이터 파이프라인의 신뢰성을 모니터링하는 메타 패턴이다.

### 입력 이벤트

- Kafka 토픽: `swipe.recorded` (또는 모든 주요 토픽)
- 카운터 소스: Kafka offset 기반 예상 이벤트 수

### 출력

- PagerDuty/Slack 알림: 파이프라인 장애 경보

---

### Go Consumer 구현

```go
package consumer

import (
    "context"
    "fmt"
    "time"
    "strconv"
)

// 1분 텀블링 윈도우 카운터 (Redis INCR + TTL)
func (c *MonitorConsumer) handleEvent(ctx context.Context, event interface{}) error {
    // 현재 분을 윈도우 키로 사용 (YYYY-MM-DD-HH-MM)
    minute := time.Now().UTC().Format("2006-01-02-15-04")
    key := fmt.Sprintf("event_count:%s", minute)

    count, err := c.redis.Incr(ctx, key).Result()
    if err != nil {
        return err
    }
    if count == 1 {
        c.redis.Expire(ctx, key, 5*time.Minute) // 5분 보존
    }
    return nil
}

// 별도 goroutine: 매 분 이전 분의 카운트와 예상값 비교
func (c *MonitorConsumer) checkDropRate(ctx context.Context) {
    ticker := time.NewTicker(time.Minute)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            prev := time.Now().UTC().Add(-time.Minute).Format("2006-01-02-15-04")
            key := fmt.Sprintf("event_count:%s", prev)

            val, err := c.redis.Get(ctx, key).Result()
            if err != nil {
                continue
            }
            actualCount, _ := strconv.ParseInt(val, 10, 64)

            // 예상 이벤트 수 = Kafka 파티션별 offset 증분 합산 (별도 조회 필요)
            expectedCount := c.getKafkaExpectedCount(ctx, prev)
            if expectedCount == 0 {
                continue
            }

            dropRate := float64(expectedCount-actualCount) / float64(expectedCount)
            if dropRate > 0.01 { // 1% 초과
                c.alerter.Send(ctx, fmt.Sprintf(
                    "파이프라인 유실률 %.2f%% (실제: %d, 예상: %d)",
                    dropRate*100, actualCount, expectedCount,
                ))
            }
        }
    }
}

// Kafka Admin API로 예상 이벤트 수 조회 (구현 복잡)
func (c *MonitorConsumer) getKafkaExpectedCount(ctx context.Context, minute string) int64 {
    // kafka-go Admin Client로 각 파티션의 offset 증분 조회
    // 복잡한 구현 생략 — 실제로는 Kafka offset API 활용
    return 0
}
```

**코드량**: 약 55줄 (Kafka offset 조회 구현 제외)

---

### Flink 구현

```java
public class PipelineHealthMonitor {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<SwipeEvent> events = env.fromSource(
            buildKafkaSource("swipe.recorded", new SwipeEventDeserializer()),
            WatermarkStrategy.<SwipeEvent>forBoundedOutOfOrderness(Duration.ofSeconds(10))
                .withTimestampAssigner((e, ts) -> e.getTimestamp()),
            "swipe-source"
        );

        // 1분 텀블링 윈도우로 실제 처리 이벤트 수 집계
        DataStream<WindowResult> actualCounts = events
            .windowAll(TumblingEventTimeWindows.of(Time.minutes(1)))
            .aggregate(new GlobalCountAggregate());

        // 예상 이벤트 수 = 직전 1분의 카운트에서 기준값 유지
        // 실제로는 별도 소스(Kafka metadata)에서 offset 증분 조회
        actualCounts
            .process(new DropRateCalculator(0.01)) // 1% 임계값
            .filter(alert -> alert.isAlert())
            .addSink(new PagerDutySink());

        env.execute("Pipeline Health Monitor");
    }
}

class DropRateCalculator extends ProcessFunction<WindowResult, DropRateAlert> {
    private final double threshold;
    private ValueState<Long> prevCountState;

    public DropRateCalculator(double threshold) {
        this.threshold = threshold;
    }

    @Override
    public void open(Configuration parameters) {
        prevCountState = getRuntimeContext().getState(
            new ValueStateDescriptor<>("prevCount", Long.class)
        );
    }

    @Override
    public void processElement(WindowResult result, Context ctx, Collector<DropRateAlert> out)
        throws Exception {

        Long prevCount = prevCountState.value();
        if (prevCount != null && prevCount > 0) {
            // 직전 분 대비 감소율 계산
            double dropRate = (double)(prevCount - result.getCount()) / prevCount;
            if (dropRate > threshold) {
                out.collect(new DropRateAlert(
                    dropRate,
                    result.getCount(),
                    prevCount,
                    result.getWindowEnd()
                ));
            }
        }
        prevCountState.update(result.getCount());
    }
}
```

**코드량**: 약 50줄

---

### 비교 분석

| 항목 | Go Consumer | Flink |
|------|-------------|-------|
| 윈도우 구현 | Redis INCR + 시간 기반 키 | 내장 텀블링 윈도우 |
| Kafka offset 조회 | Admin API 별도 구현 필요 | Flink source metrics 활용 가능 |
| 직전 분 비교 | Redis에서 이전 키 조회 | ProcessFunction ValueState |
| 복잡도 | 보통 | 보통 |

**결론**: 두 방식 모두 비슷한 복잡도. Go가 이미 있다면 굳이 Flink로 전환할 필요 없음.

---

## Pattern 10: 특정 지역 동시 접속 급증 — 트래픽 이상 감지

### 패턴 설명

특정 지역(도시/국가)에서 5분 안에 동시 접속자가 기준값의 3배를 초과하면 트래픽 이상으로 탐지한다. DDoS, 이벤트성 트래픽 급증, 또는 봇 트래픽을 감지하는 데 사용된다.

### 입력 이벤트

- Kafka 토픽: `location.updated` 또는 `login.attempted`
- 페이로드: `{ user_id, latitude, longitude, region_code, timestamp }`

### 출력

- Kafka 토픽: `traffic.anomaly.detected`
- 페이로드: `{ region_code, count, baseline, ratio, detected_at }`

---

### Go Consumer 구현

```go
package consumer

import (
    "context"
    "fmt"
    "time"

    "github.com/redis/go-redis/v9"
)

type LocationEvent struct {
    UserID     string  `json:"user_id"`
    Latitude   float64 `json:"latitude"`
    Longitude  float64 `json:"longitude"`
    RegionCode string  `json:"region_code"`
    Timestamp  int64   `json:"timestamp"`
}

func (c *TrafficMonitor) handleLocationUpdate(ctx context.Context, event LocationEvent) error {
    now := time.Now().UnixMilli()
    windowStart := now - 5*60*1000 // 5분 전
    key := fmt.Sprintf("region_traffic:%s", event.RegionCode)

    pipe := c.redis.Pipeline()
    // 유저ID를 member, 타임스탬프를 score로 저장 (중복 접속 1번으로 카운트)
    pipe.ZAdd(ctx, key, redis.Z{
        Score:  float64(now),
        Member: event.UserID,
    })
    pipe.ZRemRangeByScore(ctx, key, "0", fmt.Sprintf("%d", windowStart))
    pipe.ZCard(ctx, key)
    pipe.Expire(ctx, key, 6*time.Minute)

    results, err := pipe.Exec(ctx)
    if err != nil {
        return err
    }

    currentCount := results[2].(*redis.IntCmd).Val()

    // 기준값: 해당 지역의 평균 5분 접속자 수 (별도 집계 또는 하드코딩)
    baseline := c.getRegionBaseline(ctx, event.RegionCode)
    if baseline == 0 {
        return nil
    }

    ratio := float64(currentCount) / float64(baseline)
    if ratio >= 3.0 { // 기준값의 3배 초과
        c.alertCh <- TrafficAnomaly{
            RegionCode: event.RegionCode,
            Count:      currentCount,
            Baseline:   baseline,
            Ratio:      ratio,
        }
    }
    return nil
}

// 기준값: 직전 7일 같은 시간대 평균 (Redis에 사전 계산하여 저장)
func (c *TrafficMonitor) getRegionBaseline(ctx context.Context, regionCode string) int64 {
    key := fmt.Sprintf("region_baseline:%s", regionCode)
    val, err := c.redis.Get(ctx, key).Int64()
    if err != nil {
        return 0
    }
    return val
}
```

**코드량**: 약 45줄

---

### Flink 구현

```java
public class TrafficAnomalyDetector {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<LocationEvent> locations = env.fromSource(
            buildKafkaSource("location.updated", new LocationEventDeserializer()),
            WatermarkStrategy.<LocationEvent>forBoundedOutOfOrderness(Duration.ofSeconds(10))
                .withTimestampAssigner((e, ts) -> e.getTimestamp()),
            "location-source"
        );

        locations
            .keyBy(LocationEvent::getRegionCode)
            // 5분 슬라이딩 윈도우, 1분마다 슬라이드
            .window(SlidingEventTimeWindows.of(Time.minutes(5), Time.minutes(1)))
            .aggregate(new DistinctUserCounter()) // 고유 유저 수 집계
            .process(new AnomalyDetectionFunction(3.0)) // 기준값의 3배 임계값
            .filter(TrafficAnomaly::isAnomaly)
            .sinkTo(buildKafkaSink("traffic.anomaly.detected"));

        env.execute("Traffic Anomaly Detector");
    }
}

class AnomalyDetectionFunction
    extends KeyedProcessFunction<String, RegionCount, TrafficAnomaly> {

    private final double threshold;
    // 슬라이딩 윈도우 히스토리로 기준값 계산
    private ListState<Long> historicalCounts;

    public AnomalyDetectionFunction(double threshold) {
        this.threshold = threshold;
    }

    @Override
    public void open(Configuration parameters) {
        historicalCounts = getRuntimeContext().getListState(
            new ListStateDescriptor<>("historicalCounts", Long.class)
        );
    }

    @Override
    public void processElement(RegionCount current, Context ctx, Collector<TrafficAnomaly> out)
        throws Exception {

        List<Long> history = StreamSupport
            .stream(historicalCounts.get().spliterator(), false)
            .collect(Collectors.toList());

        // 최근 7개 윈도우(7분)의 평균을 기준값으로 사용
        if (history.size() >= 7) {
            double baseline = history.stream().mapToLong(Long::longValue).average().orElse(0);
            double ratio = baseline > 0 ? current.getCount() / baseline : 0;

            if (ratio >= threshold) {
                out.collect(new TrafficAnomaly(
                    current.getRegionCode(),
                    current.getCount(),
                    (long) baseline,
                    ratio
                ));
            }
            history.remove(0); // 가장 오래된 항목 제거
        }

        history.add(current.getCount());
        historicalCounts.update(history);
    }
}

// 지역별 고유 유저 수 집계
class DistinctUserCounter
    implements AggregateFunction<LocationEvent, Set<String>, RegionCount> {

    @Override
    public Set<String> createAccumulator() { return new HashSet<>(); }

    @Override
    public Set<String> add(LocationEvent e, Set<String> acc) {
        acc.add(e.getUserId());
        return acc;
    }

    @Override
    public RegionCount getResult(Set<String> acc) {
        return new RegionCount(acc.size()); // 키는 윈도우 컨텍스트에서 제공
    }

    @Override
    public Set<String> merge(Set<String> a, Set<String> b) {
        a.addAll(b);
        return a;
    }
}
```

**코드량**: 약 65줄

---

### 비교 분석

| 항목 | Go Consumer | Flink |
|------|-------------|-------|
| 고유 유저 카운트 | Redis ZAdd + ZCard | HashSet 누산기 |
| 기준값 계산 | 외부 저장소 (사전 계산) | Flink ListState (자동 히스토리) |
| 슬라이딩 윈도우 | 수동 ZRemRangeByScore | 내장 지원 |
| 지역 수 확장 | Redis 키 공간 비례 | Flink 파티션 병렬화 |

**결론**: 지역 수가 적고(~100개) 기준값을 사전 계산할 수 있다면 Go로 충분. 동적 기준값 계산이나 지역이 많다면 Flink가 유리.

---

## 전체 패턴 비교 요약

| 패턴 | 핵심 기법 | Go 추천 | Flink 추천 | 전환 시점 |
|------|-----------|---------|-----------|----------|
| 1. 스팸 스와이프 | 슬라이딩 카운터 | O (1K/s 이하) | 이벤트 타임 정확도 필요 시 | 10K events/s 초과 |
| 2. 스팸 메시지 | 해시 기반 중복 | O | - | 거의 불필요 |
| 3. 자동 제재 | 24h 카운터 | O | - | 거의 불필요 |
| 4. 계정 탈취 | 고유 디바이스 | O (Redis ZAdd 우아함) | - | 10K events/s 초과 |
| 5. 피처 집계 | 1시간 슬라이딩 | O (즉시 반영 필요 시) | Redis 부하 줄이려면 | 10K events/s 초과 |
| 6. 리인게이지먼트 | CEP notFollowedBy | 어려움 (cron 필요) | **O (결정적 우위)** | CEP 패턴 있으면 즉시 |
| 7. 이탈 위험 | 3일 타이머 | 보통 (cron 근사) | O (이벤트 타이머) | 100만 유저 초과 |
| 8. 갱신 알림 | 만료 타이머 | 보통 (DB cron) | O (이벤트 타이머) | 정확도 요구 시 |
| 9. 유실률 감지 | 1분 카운터 비교 | O | O | 동등 |
| 10. 트래픽 급증 | 지역별 집계 | O (지역 < 100개) | O (동적 기준값) | 동적 기준값 필요 시 |
