# Flink 코딩 가이드 — 고급편

## 목차

- [CEP (Complex Event Processing)](#cep-complex-event-processing)
  - [CEP 란 무엇인가](#cep-란-무엇인가)
  - [Pattern 정의](#pattern-정의)
  - [within (시간 제한)](#within-시간-제한)
  - [실무 예제 1: 계정 탈취 의심 감지](#실무-예제-1-계정-탈취-의심-감지)
  - [실무 예제 2: 결제 미완료 알림](#실무-예제-2-결제-미완료-알림)
  - [CEP 전체 코드](#cep-전체-코드)
- [Async I/O (비동기 외부 호출)](#async-io-비동기-외부-호출)
  - [왜 필요한가](#왜-필요한가)
  - [unorderedWait vs orderedWait](#unorderedwait-vs-orderedwait)
  - [실무 예제: 유저 정보 비동기 조회 enrichment](#실무-예제-유저-정보-비동기-조회-enrichment)
  - [타임아웃 처리](#타임아웃-처리)
  - [Async I/O 전체 코드](#async-io-전체-코드)
- [Side Output (부가 출력)](#side-output-부가-출력)
  - [OutputTag 정의](#outputtag-정의)
  - [용도](#용도)
  - [실무 예제: 이벤트 라우팅](#실무-예제-이벤트-라우팅)
  - [Side Output 전체 코드](#side-output-전체-코드)
- [Broadcast State 패턴](#broadcast-state-패턴)
  - [사용 시나리오](#사용-시나리오)
  - [BroadcastProcessFunction 구현](#broadcastprocessfunction-구현)
  - [실무 예제: 동적 차단 규칙 적용](#실무-예제-동적-차단-규칙-적용)
  - [Broadcast State 전체 코드](#broadcast-state-전체-코드)
- [Custom Serialization](#custom-serialization)
  - [왜 기본 직렬화가 느린가](#왜-기본-직렬화가-느린가)
  - [TypeInformation 과 TypeSerializer](#typeinformation-과-typeserializer)
  - [Avro, Protobuf 연동](#avro-protobuf-연동)
  - [실무 예제: Protobuf 메시지 처리](#실무-예제-protobuf-메시지-처리)
- [테스트 작성](#테스트-작성)
  - [MiniClusterWithClientResource](#miniclusterwithclientresource)
  - [TestHarness 단위 테스트](#testharness-단위-테스트)
  - [실무 예제: 타이머 로직 단위 테스트](#실무-예제-타이머-로직-단위-테스트)
  - [테스트 전체 코드](#테스트-전체-코드)
- [성능 최적화 팁](#성능-최적화-팁)
  - [Operator Chaining 제어](#operator-chaining-제어)
  - [Serialization 최적화](#serialization-최적화)
  - [State 접근 최소화 패턴](#state-접근-최소화-패턴)
  - [Network Buffer 튜닝](#network-buffer-튜닝)

---

## CEP (Complex Event Processing)

### CEP 란 무엇인가

단순 집계(aggregation)는 "최근 1분간 이벤트 개수"처럼 개별 이벤트를 수치로 요약한다. CEP 는 그보다 한 단계 위로, **이벤트들 사이의 순서와 시간 관계**에서 패턴을 찾는다.

예를 들어 "로그인 실패가 3회 연속 발생한 직후 성공이 오면 계정 탈취로 의심한다"는 규칙은 단순 카운터로 표현하기 어렵다. 이벤트의 순서, 동일 사용자 키 범위, 시간 창을 동시에 고려해야 하기 때문이다. Flink CEP 라이브러리는 이런 복잡한 시퀀스 패턴을 선언적으로 표현하게 해 준다.

의존성 추가 (Maven):

```xml
<dependency>
    <groupId>org.apache.flink</groupId>
    <artifactId>flink-cep</artifactId>
    <version>1.18.1</version>
</dependency>
```

### Pattern 정의

`Pattern` 은 이벤트 시퀀스를 설명하는 빌더 객체이다. 주요 연결자는 다음과 같다.

| 연결자 | 의미 |
|---|---|
| `next(name)` | 바로 다음 이벤트 (사이에 다른 이벤트 없음) |
| `followedBy(name)` | 이후 어느 시점의 이벤트 (사이에 다른 이벤트 허용) |
| `followedByAny(name)` | followedBy 의 비결정론적 버전 |
| `notNext(name)` | 바로 다음 이벤트가 해당 조건이면 안 됨 |
| `notFollowedBy(name)` | 이후 어느 시점에도 해당 조건 이벤트가 오면 안 됨 |

각 패턴 노드에는 `.where(condition)` 으로 필터 조건을 붙인다.

```java
// "failFirst 라는 이름의 이벤트 노드 정의 — type 이 FAIL 인 이벤트"
Pattern<LoginEvent, ?> pattern = Pattern
    .<LoginEvent>begin("failFirst")
        .where(new SimpleCondition<LoginEvent>() {
            @Override
            public boolean filter(LoginEvent event) {
                return "FAIL".equals(event.getType());
            }
        });
```

### within (시간 제한)

패턴 전체에 시간 창을 걸어 "이 시간 안에 시퀀스가 완성되지 않으면 버린다"고 지정한다.

```java
pattern.within(Time.minutes(10)); // 10분 안에 전체 패턴이 매칭되어야 함
```

`within` 은 Pattern 체인의 마지막에 한 번만 붙인다. 중간 노드별로 다른 시간 창이 필요하면 `AfterMatchSkipStrategy` 와 조합한다.

### 실무 예제 1: 계정 탈취 의심 감지

**시나리오**: 동일 사용자 ID 에 대해 로그인 실패가 3회 이상 연속 발생한 뒤 10분 이내에 성공이 오면 경보를 발생시킨다.

패턴 설계:

```
failFirst -> failSecond -> failThird -> success
       (모두 같은 userId, within 10분)
```

### 실무 예제 2: 결제 미완료 알림

**시나리오**: ORDER_CREATED 이벤트 후 30분 이내에 동일 orderId 의 PAYMENT_COMPLETED 가 오지 않으면 알림을 보낸다.

패턴 설계:

```
orderCreated  notFollowedBy(paymentCompleted)  within 30분
```

`notFollowedBy` 로 끝나는 패턴은 **타임아웃 이벤트**로 처리한다. 정상 매칭(payment 가 온 경우)이 아니라 시간이 초과됐을 때 결과가 나온다.

### CEP 전체 코드

```java
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.cep.CEP;
import org.apache.flink.cep.PatternSelectFunction;
import org.apache.flink.cep.PatternStream;
import org.apache.flink.cep.PatternTimeoutFunction;
import org.apache.flink.cep.pattern.Pattern;
import org.apache.flink.cep.pattern.conditions.SimpleCondition;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.util.OutputTag;

import java.time.Duration;
import java.util.List;
import java.util.Map;

// ── 도메인 객체 ──────────────────────────────────────────────────────────────

class LoginEvent {
    public String userId;
    public String type;   // "FAIL" or "SUCCESS"
    public long   timestamp;

    public LoginEvent() {}
    public LoginEvent(String userId, String type, long timestamp) {
        this.userId    = userId;
        this.type      = type;
        this.timestamp = timestamp;
    }
    public String getUserId()  { return userId; }
    public String getType()    { return type; }
    public long   getTimestamp() { return timestamp; }
}

class SuspiciousLoginAlert {
    public String userId;
    public String message;
    public SuspiciousLoginAlert(String userId, String message) {
        this.userId  = userId;
        this.message = message;
    }
}

class OrderEvent {
    public String orderId;
    public String type;       // "ORDER_CREATED" or "PAYMENT_COMPLETED"
    public long   timestamp;

    public OrderEvent() {}
    public OrderEvent(String orderId, String type, long timestamp) {
        this.orderId   = orderId;
        this.type      = type;
        this.timestamp = timestamp;
    }
    public String getOrderId()  { return orderId; }
    public String getType()     { return type; }
    public long   getTimestamp() { return timestamp; }
}

// ── 예제 1: 계정 탈취 의심 감지 ──────────────────────────────────────────────

public class CepAccountTakeoverDetection {

    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // 이벤트 시간 기반 처리를 위해 Watermark 전략 설정
        // 최대 5초 지연을 허용하는 단조 증가 타임스탬프 전략 사용
        DataStream<LoginEvent> loginStream = env
            .fromElements(
                new LoginEvent("user1", "FAIL",    1_000L),
                new LoginEvent("user1", "FAIL",    2_000L),
                new LoginEvent("user1", "FAIL",    3_000L),
                new LoginEvent("user1", "SUCCESS", 4_000L),
                new LoginEvent("user2", "FAIL",    1_500L)  // 패턴 미완성
            )
            .assignTimestampsAndWatermarks(
                WatermarkStrategy
                    .<LoginEvent>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                    .withTimestampAssigner((event, ts) -> event.getTimestamp())
            );

        // 패턴 정의: 실패 3회 연속 후 성공 — 10분 이내
        Pattern<LoginEvent, ?> loginPattern = Pattern
            .<LoginEvent>begin("failFirst")
                .where(new SimpleCondition<LoginEvent>() {
                    @Override
                    public boolean filter(LoginEvent e) {
                        return "FAIL".equals(e.getType());
                    }
                })
            // next: 사이에 다른 이벤트 없이 바로 다음이어야 함
            .next("failSecond")
                .where(new SimpleCondition<LoginEvent>() {
                    @Override
                    public boolean filter(LoginEvent e) {
                        return "FAIL".equals(e.getType());
                    }
                })
            .next("failThird")
                .where(new SimpleCondition<LoginEvent>() {
                    @Override
                    public boolean filter(LoginEvent e) {
                        return "FAIL".equals(e.getType());
                    }
                })
            // followedBy: 바로 다음이 아니어도 되지만 10분 창 안에 있어야 함
            .followedBy("success")
                .where(new SimpleCondition<LoginEvent>() {
                    @Override
                    public boolean filter(LoginEvent e) {
                        return "SUCCESS".equals(e.getType());
                    }
                })
            .within(Time.minutes(10)); // 전체 시퀀스는 10분 안에 완성

        // userId 를 키로 파티셔닝 후 CEP 적용
        PatternStream<LoginEvent> patternStream = CEP.pattern(
            loginStream.keyBy(LoginEvent::getUserId),
            loginPattern
        );

        // 매칭된 시퀀스에서 경보 생성
        DataStream<SuspiciousLoginAlert> alerts = patternStream.select(
            new PatternSelectFunction<LoginEvent, SuspiciousLoginAlert>() {
                @Override
                public SuspiciousLoginAlert select(Map<String, List<LoginEvent>> match) {
                    // match 맵의 키는 pattern 에서 지정한 이름
                    LoginEvent lastFail = match.get("failThird").get(0);
                    LoginEvent success  = match.get("success").get(0);
                    return new SuspiciousLoginAlert(
                        lastFail.getUserId(),
                        "로그인 실패 3회 후 성공 감지 — 계정 탈취 의심 (성공 시각: "
                            + success.getTimestamp() + ")"
                    );
                }
            }
        );

        alerts.print();
        env.execute("Account Takeover Detection");
    }
}

// ── 예제 2: 결제 미완료 알림 ─────────────────────────────────────────────────

public class CepPaymentTimeoutAlert {

    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<OrderEvent> orderStream = env
            .fromElements(
                new OrderEvent("order-1", "ORDER_CREATED",      1_000L),
                new OrderEvent("order-2", "ORDER_CREATED",      2_000L),
                new OrderEvent("order-2", "PAYMENT_COMPLETED",  5_000L),  // order-2 는 정상
                new OrderEvent("order-1", "ORDER_CREATED",     35 * 60 * 1_000L) // 워터마크 진행용
            )
            .assignTimestampsAndWatermarks(
                WatermarkStrategy
                    .<OrderEvent>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                    .withTimestampAssigner((e, ts) -> e.getTimestamp())
            );

        // notFollowedBy 로 끝나는 패턴 — 타임아웃 시 결과가 나온다
        Pattern<OrderEvent, ?> paymentPattern = Pattern
            .<OrderEvent>begin("orderCreated")
                .where(new SimpleCondition<OrderEvent>() {
                    @Override
                    public boolean filter(OrderEvent e) {
                        return "ORDER_CREATED".equals(e.getType());
                    }
                })
            // notFollowedBy: 이후 결제가 오면 이 패턴을 무효화
            .notFollowedBy("paymentCompleted")
                .where(new SimpleCondition<OrderEvent>() {
                    @Override
                    public boolean filter(OrderEvent e) {
                        return "PAYMENT_COMPLETED".equals(e.getType());
                    }
                })
            .within(Time.minutes(30));

        // 타임아웃된 이벤트를 담을 OutputTag
        OutputTag<OrderEvent> timeoutTag = new OutputTag<OrderEvent>("payment-timeout") {};

        PatternStream<OrderEvent> patternStream = CEP.pattern(
            orderStream.keyBy(OrderEvent::getOrderId),
            paymentPattern
        );

        // select 의 두 번째 인자로 타임아웃 핸들러를 등록
        SingleOutputStreamOperator<String> mainStream = patternStream.select(
            timeoutTag,
            // 타임아웃 핸들러: 30분 안에 결제가 없었을 때 호출
            new PatternTimeoutFunction<OrderEvent, OrderEvent>() {
                @Override
                public OrderEvent timeout(Map<String, List<OrderEvent>> pattern,
                                          long timeoutTimestamp) {
                    return pattern.get("orderCreated").get(0);
                }
            },
            // 정상 매칭 핸들러: notFollowedBy 패턴은 정상 매칭이 발생하지 않으므로 빈 문자열 반환
            new PatternSelectFunction<OrderEvent, String>() {
                @Override
                public String select(Map<String, List<OrderEvent>> match) {
                    return "";
                }
            }
        );

        // 타임아웃된 주문만 side output 에서 꺼내 처리
        DataStream<OrderEvent> unpaidOrders = mainStream.getSideOutput(timeoutTag);
        unpaidOrders.map(e -> "미결제 알림: orderId=" + e.getOrderId()).print();

        env.execute("Payment Timeout Alert");
    }
}
```

---

## Async I/O (비동기 외부 호출)

### 왜 필요한가

스트림 처리 중 외부 DB 나 REST API 를 동기 방식으로 호출하면 처리 스레드가 응답을 기다리는 동안 블로킹된다. 예를 들어 DB 조회 지연이 10ms 이고 Slot 이 1개라면 초당 최대 100개 이벤트밖에 처리하지 못한다.

Async I/O 는 하나의 스레드에서 수백 개의 요청을 동시에 in-flight 상태로 유지하면서 완료된 것부터 결과를 처리한다. 처리량이 수십 배 향상되는 경우가 많다.

내부 동작:

```
이벤트 수신 → 비동기 요청 큐에 등록 → 다음 이벤트 수신 → ...
                  ↓ 응답 도착 시 콜백
              결과를 ResultFuture 에 완성
```

### unorderedWait vs orderedWait

| 메서드 | 순서 보장 | 처리량 | 용도 |
|---|---|---|---|
| `unorderedWait` | 보장 안 함 | 높음 | 순서 무관한 enrichment |
| `orderedWait` | 입력 순서 유지 | 낮음 | 순서에 의존하는 downstream 연산 |

대부분의 enrichment 작업은 이벤트 순서가 중요하지 않으므로 `unorderedWait` 를 권장한다.

### 실무 예제: 유저 정보 비동기 조회 enrichment

이벤트에는 `userId` 만 있고, DB 에서 `userName`, `userTier` 를 조회해 이벤트에 붙이는 시나리오이다.

### 타임아웃 처리

비동기 요청이 지정 시간 안에 완료되지 않으면 `AsyncFunction.timeout()` 이 호출된다. 기본 동작은 예외를 던져 job 을 실패시키므로, 반드시 오버라이드하여 빈 결과 또는 기본값을 반환해야 한다.

### Async I/O 전체 코드

```java
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.AsyncDataStream;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.async.AsyncFunction;
import org.apache.flink.streaming.api.functions.async.ResultFuture;
import org.apache.flink.streaming.api.functions.async.RichAsyncFunction;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.Collections;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

// ── 도메인 객체 ──────────────────────────────────────────────────────────────

class ClickEvent {
    public String userId;
    public String page;
    public long   timestamp;

    public ClickEvent() {}
    public ClickEvent(String userId, String page, long timestamp) {
        this.userId    = userId;
        this.page      = page;
        this.timestamp = timestamp;
    }
}

class EnrichedClickEvent {
    public String userId;
    public String userName;
    public String userTier; // "FREE", "PREMIUM", "VIP"
    public String page;
    public long   timestamp;
}

// ── AsyncFunction 구현 ───────────────────────────────────────────────────────

/**
 * RichAsyncFunction 을 상속하면 open/close 에서 커넥션 풀 등 리소스를 관리할 수 있다.
 * 실제 프로덕션에서는 HikariCP 나 AsyncHttpClient 같은 비동기 클라이언트를 사용한다.
 * 여기서는 설명을 위해 스레드 풀로 동기 JDBC 를 비동기처럼 감싸는 패턴을 보여 준다.
 */
public class UserEnrichmentFunction
        extends RichAsyncFunction<ClickEvent, EnrichedClickEvent> {

    private transient ExecutorService executor;
    private transient Connection       dbConn;

    private final String jdbcUrl;

    public UserEnrichmentFunction(String jdbcUrl) {
        this.jdbcUrl = jdbcUrl;
    }

    @Override
    public void open(Configuration parameters) throws Exception {
        // TaskManager 가 시작될 때 한 번 호출 — 커넥션과 스레드 풀을 초기화
        executor = Executors.newFixedThreadPool(10);
        dbConn   = DriverManager.getConnection(jdbcUrl, "user", "pass");
    }

    @Override
    public void asyncInvoke(ClickEvent input, ResultFuture<EnrichedClickEvent> resultFuture) {
        // CompletableFuture 를 사용해 DB 조회를 스레드 풀에서 실행
        CompletableFuture.supplyAsync(() -> {
            try {
                PreparedStatement stmt = dbConn.prepareStatement(
                    "SELECT user_name, user_tier FROM users WHERE user_id = ?"
                );
                stmt.setString(1, input.userId);
                ResultSet rs = stmt.executeQuery();

                EnrichedClickEvent enriched = new EnrichedClickEvent();
                enriched.userId    = input.userId;
                enriched.page      = input.page;
                enriched.timestamp = input.timestamp;

                if (rs.next()) {
                    // DB 조회 성공 시 필드 채우기
                    enriched.userName = rs.getString("user_name");
                    enriched.userTier = rs.getString("user_tier");
                } else {
                    // 유저를 찾지 못한 경우 기본값 설정
                    enriched.userName = "UNKNOWN";
                    enriched.userTier = "FREE";
                }
                return enriched;
            } catch (Exception e) {
                throw new RuntimeException("DB 조회 실패: userId=" + input.userId, e);
            }
        }, executor).thenAccept(resultFuture::complete);
        // thenAccept: 결과가 준비되면 ResultFuture 에 넘겨 Flink 가 downstream 으로 방출하게 함
    }

    @Override
    public void timeout(ClickEvent input, ResultFuture<EnrichedClickEvent> resultFuture) {
        // 타임아웃 발생 시 빈 결과를 반환해 job 을 계속 진행
        // 예외를 던지면 job 이 실패하므로, 기본값으로 처리하는 것이 안전
        EnrichedClickEvent fallback = new EnrichedClickEvent();
        fallback.userId    = input.userId;
        fallback.page      = input.page;
        fallback.timestamp = input.timestamp;
        fallback.userName  = "TIMEOUT";
        fallback.userTier  = "FREE";
        resultFuture.complete(Collections.singleton(fallback));
    }

    @Override
    public void close() throws Exception {
        // TaskManager 가 종료될 때 리소스 해제
        if (executor != null) executor.shutdown();
        if (dbConn   != null) dbConn.close();
    }
}

// ── 메인 잡 ─────────────────────────────────────────────────────────────────

public class AsyncIoJob {

    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<ClickEvent> clickStream = env.fromElements(
            new ClickEvent("u1", "/home",    1_000L),
            new ClickEvent("u2", "/product", 2_000L),
            new ClickEvent("u3", "/cart",    3_000L)
        );

        // unorderedWait: 응답 순서 보장 없음, 처리량 최대화
        // 두 번째 인자: 타임아웃 값, 세 번째 인자: 타임아웃 단위
        // 네 번째 인자: 동시 in-flight 요청 수 상한 (기본 100)
        DataStream<EnrichedClickEvent> enrichedStream = AsyncDataStream.unorderedWait(
            clickStream,
            new UserEnrichmentFunction("jdbc:mysql://localhost:3306/mydb"),
            1_000,          // 타임아웃 1000ms
            TimeUnit.MILLISECONDS,
            50              // 최대 50개 요청을 동시에 in-flight
        );

        enrichedStream.print();
        env.execute("Async IO Enrichment Job");
    }
}
```

---

## Side Output (부가 출력)

### OutputTag 정의

`OutputTag` 는 부가 출력 스트림에 이름과 타입을 부여하는 핸들이다. 익명 클래스로 생성해야 제네릭 타입 정보가 런타임에 보존된다.

```java
// 올바른 방법: 익명 클래스로 생성 (타입 정보 보존)
OutputTag<String> errorTag = new OutputTag<String>("parse-error") {};

// 잘못된 방법: 직접 생성하면 제네릭 타입이 소거됨
// OutputTag<String> errorTag = new OutputTag<>("parse-error"); // 컴파일은 되지만 런타임 오류 가능
```

### 용도

- **늦은 데이터 분리**: 워터마크를 지나서 도착한 이벤트를 별도 스트림으로 수집
- **에러 이벤트 분리**: 파싱 실패, 유효성 검사 실패 이벤트를 격리해 dead-letter 토픽으로 전송
- **조건별 라우팅**: 이벤트 속성에 따라 서로 다른 Kafka 토픽이나 싱크로 분기

하나의 `ProcessFunction` 에서 여러 `OutputTag` 에 동시에 방출할 수 있으므로, `if-else` 로 분기하는 것보다 유연하게 스트림을 나눌 수 있다.

### 실무 예제: 이벤트 라우팅

**시나리오**: Kafka 에서 JSON 이벤트를 읽어 다음 세 가지로 분류한다.

1. 정상 이벤트 → `events-normal` 토픽
2. JSON 파싱 실패 이벤트 → `events-error` 토픽
3. 워터마크보다 늦게 도착한 이벤트 → `events-late` 토픽

### Side Output 전체 코드

```java
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.time.Duration;

// ── 도메인 객체 ──────────────────────────────────────────────────────────────

class NormalEvent {
    public String userId;
    public String action;
    public long   eventTime;

    public NormalEvent(String userId, String action, long eventTime) {
        this.userId    = userId;
        this.action    = action;
        this.eventTime = eventTime;
    }
    public long getEventTime() { return eventTime; }
}

class ErrorEvent {
    public String rawJson;
    public String reason;

    public ErrorEvent(String rawJson, String reason) {
        this.rawJson = rawJson;
        this.reason  = reason;
    }
}

// ── ProcessFunction 구현 ─────────────────────────────────────────────────────

public class EventRoutingFunction extends ProcessFunction<String, NormalEvent> {

    // 각 OutputTag 를 static final 로 선언해 직렬화 시 안정적으로 참조되게 함
    public static final OutputTag<ErrorEvent>  ERROR_TAG =
        new OutputTag<ErrorEvent>("parse-error") {};

    public static final OutputTag<NormalEvent> LATE_TAG =
        new OutputTag<NormalEvent>("late-event") {};

    private final ObjectMapper mapper = new ObjectMapper();

    @Override
    public void processElement(
            String rawJson,
            Context ctx,
            Collector<NormalEvent> out) throws Exception {

        NormalEvent event;

        // 1단계: JSON 파싱 시도
        try {
            JsonNode node = mapper.readTree(rawJson);
            event = new NormalEvent(
                node.get("userId").asText(),
                node.get("action").asText(),
                node.get("eventTime").asLong()
            );
        } catch (Exception e) {
            // 파싱 실패: 에러 side output 으로 방출
            ctx.output(ERROR_TAG, new ErrorEvent(rawJson, e.getMessage()));
            return; // 메인 스트림으로는 방출하지 않음
        }

        // 2단계: 늦은 도착 여부 확인
        // ctx.timerService().currentWatermark() 는 현재 워터마크 타임스탬프를 반환
        if (event.getEventTime() < ctx.timerService().currentWatermark()) {
            // 워터마크보다 이전 시간 이벤트 → 늦은 데이터 side output
            ctx.output(LATE_TAG, event);
            return;
        }

        // 3단계: 정상 이벤트는 메인 스트림으로 방출
        out.collect(event);
    }
}

// ── 메인 잡 ─────────────────────────────────────────────────────────────────

public class SideOutputRoutingJob {

    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // 실제 환경에서는 KafkaSource 를 사용한다
        DataStream<String> rawStream = env.fromElements(
            "{\"userId\":\"u1\",\"action\":\"click\",\"eventTime\":1000}",
            "this is not json",                                                 // 파싱 실패 예시
            "{\"userId\":\"u2\",\"action\":\"purchase\",\"eventTime\":2000}",
            "{\"userId\":\"u3\",\"action\":\"view\",\"eventTime\":500}"         // 늦은 데이터 예시
        );

        // Watermark 전략: 최대 1초 지연 허용
        DataStream<String> timedStream = rawStream.assignTimestampsAndWatermarks(
            WatermarkStrategy
                .<String>forBoundedOutOfOrderness(Duration.ofSeconds(1))
                // 문자열 스트림이므로 파싱 전 타임스탬프 추출은 생략하고
                // ProcessFunction 내부에서 직접 처리
                .withTimestampAssigner((str, ts) -> {
                    try {
                        return new ObjectMapper().readTree(str).get("eventTime").asLong();
                    } catch (Exception e) {
                        return Long.MIN_VALUE; // 파싱 실패 이벤트는 최솟값 반환
                    }
                })
        );

        // ProcessFunction 적용 — SingleOutputStreamOperator 를 받아야 side output 에 접근 가능
        SingleOutputStreamOperator<NormalEvent> mainStream = timedStream
            .process(new EventRoutingFunction());

        // 각 스트림을 꺼내 별도 싱크로 연결
        DataStream<NormalEvent> normalStream = mainStream;
        DataStream<ErrorEvent>  errorStream  = mainStream.getSideOutput(EventRoutingFunction.ERROR_TAG);
        DataStream<NormalEvent> lateStream   = mainStream.getSideOutput(EventRoutingFunction.LATE_TAG);

        // 실제 환경에서는 KafkaSink 로 각각 다른 토픽에 전송
        normalStream.print("NORMAL");
        errorStream .print("ERROR");
        lateStream  .print("LATE");

        env.execute("Event Routing with Side Output");
    }
}
```

---

## Broadcast State 패턴

### 사용 시나리오

Broadcast State 는 **모든 병렬 Task 에 동일한 상태를 공유**해야 할 때 사용한다. 가장 전형적인 사례는 런타임 중에 변경될 수 있는 규칙(rule) 을 모든 Task 에 동기화하는 것이다.

예를 들어 IP 차단 목록이나 사기 탐지 규칙은 주기적으로 업데이트된다. 이 규칙을 Kafka 토픽에 넣고, Flink 가 실시간으로 읽어서 모든 병렬 Task 의 로컬 상태를 갱신하면 재배포 없이 규칙을 바꿀 수 있다.

동작 원리:

```
규칙 스트림 (낮은 QPS)
        |
        ↓ broadcast
모든 Task 인스턴스의 BroadcastState (동일한 복사본)
        |
        + 이벤트 스트림 (높은 QPS)
        ↓
BroadcastProcessFunction.processElement()
```

### BroadcastProcessFunction 구현

`BroadcastProcessFunction<IN1, IN2, OUT>` 을 상속한다.

- `processBroadcastElement()`: 규칙 스트림 이벤트 처리 — BroadcastState 를 **쓸 수 있다**
- `processElement()`: 일반 이벤트 스트림 처리 — BroadcastState 를 **읽기만 할 수 있다**

이 비대칭 접근 규칙은 병렬 Task 간 일관성을 보장하기 위한 것이다. 일반 이벤트 처리 중에는 쓰기가 금지된다.

### 실무 예제: 동적 차단 규칙 적용

**시나리오**: 관리자가 특정 IP 를 차단하는 규칙을 실시간으로 추가하면, 진행 중인 트래픽 스트림에 즉시 반영해 차단된 IP 의 이벤트를 필터링한다.

### Broadcast State 전체 코드

```java
import org.apache.flink.api.common.state.BroadcastState;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.common.state.ReadOnlyBroadcastState;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.streaming.api.datastream.BroadcastStream;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.co.BroadcastProcessFunction;
import org.apache.flink.util.Collector;

// ── 도메인 객체 ──────────────────────────────────────────────────────────────

class TrafficEvent {
    public String ip;
    public String path;
    public long   timestamp;

    public TrafficEvent() {}
    public TrafficEvent(String ip, String path, long timestamp) {
        this.ip        = ip;
        this.path      = path;
        this.timestamp = timestamp;
    }
}

class BlockRule {
    public String ip;
    public String reason;
    public boolean block; // true: 차단 추가, false: 차단 해제

    public BlockRule() {}
    public BlockRule(String ip, String reason, boolean block) {
        this.ip     = ip;
        this.reason = reason;
        this.block  = block;
    }
}

class FilteredEvent {
    public String ip;
    public String path;
    public String status; // "ALLOWED" or "BLOCKED"

    public FilteredEvent(String ip, String path, String status) {
        this.ip     = ip;
        this.path   = path;
        this.status = status;
    }
}

// ── BroadcastProcessFunction 구현 ────────────────────────────────────────────

public class IpBlockBroadcastFunction
        extends BroadcastProcessFunction<TrafficEvent, BlockRule, FilteredEvent> {

    // Broadcast State 의 스키마를 정의하는 Descriptor
    // 이 Descriptor 는 connect() 와 BroadcastProcessFunction 양쪽에서 동일하게 사용해야 함
    public static final MapStateDescriptor<String, BlockRule> BLOCK_RULES_DESCRIPTOR =
        new MapStateDescriptor<>(
            "block-rules",   // state 이름
            Types.STRING,    // 키 타입: IP 주소
            Types.POJO(BlockRule.class)  // 값 타입: 차단 규칙
        );

    @Override
    public void processBroadcastElement(
            BlockRule rule,
            Context ctx,
            Collector<FilteredEvent> out) throws Exception {

        // 규칙 스트림 처리 — BroadcastState 를 여기서만 쓸 수 있음
        BroadcastState<String, BlockRule> state = ctx.getBroadcastState(BLOCK_RULES_DESCRIPTOR);

        if (rule.block) {
            // 차단 규칙 추가
            state.put(rule.ip, rule);
        } else {
            // 차단 규칙 해제
            state.remove(rule.ip);
        }
    }

    @Override
    public void processElement(
            TrafficEvent event,
            ReadOnlyContext ctx,
            Collector<FilteredEvent> out) throws Exception {

        // 일반 이벤트 처리 — BroadcastState 읽기만 허용
        ReadOnlyBroadcastState<String, BlockRule> state =
            ctx.getBroadcastState(BLOCK_RULES_DESCRIPTOR);

        if (state.contains(event.ip)) {
            // 차단된 IP: 차단 이벤트를 방출 (모니터링용으로 기록)
            out.collect(new FilteredEvent(event.ip, event.path, "BLOCKED"));
        } else {
            // 정상 트래픽: 허용
            out.collect(new FilteredEvent(event.ip, event.path, "ALLOWED"));
        }
    }
}

// ── 메인 잡 ─────────────────────────────────────────────────────────────────

public class BroadcastStateJob {

    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // 일반 트래픽 스트림 (높은 처리량)
        DataStream<TrafficEvent> trafficStream = env.fromElements(
            new TrafficEvent("1.2.3.4", "/api/login",   1_000L),
            new TrafficEvent("5.6.7.8", "/api/products", 2_000L),
            new TrafficEvent("1.2.3.4", "/api/payment",  3_000L)  // 이 시점에는 이미 차단됨
        );

        // 규칙 스트림 (낮은 처리량 — 관리자 업데이트)
        DataStream<BlockRule> ruleStream = env.fromElements(
            new BlockRule("1.2.3.4", "스팸 IP", true),   // 1.2.3.4 차단 추가
            new BlockRule("9.9.9.9", "테스트",  true)    // 9.9.9.9 차단 추가
        );

        // 규칙 스트림을 BroadcastStream 으로 변환
        // Descriptor 는 IpBlockBroadcastFunction 과 동일한 것을 사용해야 함
        BroadcastStream<BlockRule> broadcastRules =
            ruleStream.broadcast(IpBlockBroadcastFunction.BLOCK_RULES_DESCRIPTOR);

        // 트래픽 스트림과 broadcast 스트림을 connect
        DataStream<FilteredEvent> result = trafficStream
            .connect(broadcastRules)
            .process(new IpBlockBroadcastFunction());

        result.print();
        env.execute("Dynamic IP Block with Broadcast State");
    }
}
```

---

## Custom Serialization

### 왜 기본 직렬화가 느린가

Flink 의 기본 직렬화 전략은 다음 순서로 적용된다.

1. Flink 내장 타입 (Integer, String, Tuple, POJO 등) → 최적화된 TypeSerializer 사용
2. Kryo 등록된 타입 → Kryo 직렬화
3. 나머지 → Kryo 기본 직렬화 (가장 느림)

Kryo 는 범용이기 때문에 스키마 정보를 함께 직렬화하고 리플렉션을 사용한다. 이벤트가 초당 수백만 건인 환경에서는 직렬화 CPU 비용이 전체 처리 시간의 30% 이상을 차지하기도 한다.

**POJO 최적화 규칙**: Flink 가 최적화된 TypeSerializer 를 생성하려면 POJO 가 다음 조건을 모두 만족해야 한다.

- public 클래스
- 인수 없는 public 기본 생성자
- 모든 필드가 public 이거나 getter/setter 를 가짐
- 필드 타입 역시 지원되는 타입 (재귀 조건)

### TypeInformation 과 TypeSerializer

`TypeInformation<T>` 은 Flink 타입 시스템의 핵심이다. `TypeSerializer<T>` 를 생성하는 팩토리 역할을 한다.

```java
// 타입 정보 명시적 획득
TypeInformation<MyEvent> typeInfo = TypeInformation.of(MyEvent.class);

// 제네릭 타입의 경우 TypeHint 로 타입 소거 우회
TypeInformation<Tuple2<String, Long>> tupleType =
    TypeInformation.of(new TypeHint<Tuple2<String, Long>>() {});
```

제네릭 클래스를 반환하는 함수에는 `getProducedType()` 을 오버라이드해 타입 정보를 명시해야 Kryo 로 폴백되는 것을 방지할 수 있다.

### Avro, Protobuf 연동

Flink 는 `flink-avro` 모듈로 Avro 를 공식 지원한다. Protobuf 는 공식 모듈이 없으므로 직접 `DeserializationSchema` / `SerializationSchema` 를 구현한다.

의존성:

```xml
<!-- Avro -->
<dependency>
    <groupId>org.apache.flink</groupId>
    <artifactId>flink-avro</artifactId>
    <version>1.18.1</version>
</dependency>

<!-- Protobuf (구글 공식 라이브러리) -->
<dependency>
    <groupId>com.google.protobuf</groupId>
    <artifactId>protobuf-java</artifactId>
    <version>3.25.1</version>
</dependency>
```

### 실무 예제: Protobuf 메시지 처리

Protobuf 스키마 정의 (`user_event.proto`):

```protobuf
syntax = "proto3";
package com.example;

message UserEvent {
  string user_id  = 1;
  string action   = 2;
  int64  event_time = 3;
}
```

```java
import com.example.UserEventProto.UserEvent;
import org.apache.flink.api.common.serialization.DeserializationSchema;
import org.apache.flink.api.common.serialization.SerializationSchema;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

// ── Protobuf DeserializationSchema ───────────────────────────────────────────

/**
 * Kafka 메시지 바이트를 Protobuf UserEvent 로 역직렬화한다.
 * Flink 가 타입 정보를 알 수 있도록 getProducedType() 을 반드시 오버라이드한다.
 */
public class UserEventDeserializer implements DeserializationSchema<UserEvent> {

    @Override
    public UserEvent deserialize(byte[] message) throws Exception {
        // Protobuf parseFrom 은 바이트 배열에서 메시지를 복원
        return UserEvent.parseFrom(message);
    }

    @Override
    public boolean isEndOfStream(UserEvent nextElement) {
        // 스트림이 끝이 없으므로 항상 false 반환
        return false;
    }

    @Override
    public TypeInformation<UserEvent> getProducedType() {
        // TypeInformation.of 로 명시해 Kryo 폴백 방지
        return TypeInformation.of(UserEvent.class);
    }
}

// ── Protobuf SerializationSchema ─────────────────────────────────────────────

public class UserEventSerializer implements SerializationSchema<UserEvent> {

    @Override
    public byte[] serialize(UserEvent element) {
        // Protobuf toByteArray 는 효율적인 이진 직렬화
        return element.toByteArray();
    }
}

// ── 메인 잡 ─────────────────────────────────────────────────────────────────

public class ProtobufKafkaJob {

    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // Protobuf DeserializationSchema 를 KafkaSource 에 주입
        KafkaSource<UserEvent> source = KafkaSource.<UserEvent>builder()
            .setBootstrapServers("localhost:9092")
            .setTopics("user-events-proto")
            .setGroupId("flink-proto-consumer")
            .setStartingOffsets(OffsetsInitializer.latest())
            .setValueOnlyDeserializer(new UserEventDeserializer())
            .build();

        DataStream<UserEvent> events = env.fromSource(
            source,
            org.apache.flink.api.common.eventtime.WatermarkStrategy.noWatermarks(),
            "Kafka Protobuf Source"
        );

        // Protobuf 필드 접근: 생성된 클래스의 getter 사용
        events
            .filter(e -> !e.getUserId().isEmpty())          // 빈 userId 필터
            .map(e -> e.getUserId() + ":" + e.getAction()) // userId:action 형식으로 변환
            .print();

        env.execute("Protobuf Kafka Job");
    }
}
```

---

## 테스트 작성

### MiniClusterWithClientResource

통합 테스트에서는 `MiniClusterWithClientResource` 를 JUnit 4 Rule 로 등록해 인프라 없이 실제 Flink 클러스터를 인메모리로 구동한다.

```java
@ClassRule
public static MiniClusterWithClientResource flinkCluster =
    new MiniClusterWithClientResource(
        new MiniClusterResourceConfiguration.Builder()
            .setNumberSlotsPerTaskManager(2) // TaskManager 당 슬롯 수
            .setNumberTaskManagers(1)
            .build()
    );
```

통합 테스트는 실제 DAG 전체를 실행하므로 느리다. 핵심 로직(특히 타이머, 상태 관리)은 `TestHarness` 단위 테스트로 빠르게 검증한 뒤, 통합 테스트는 엔드투엔드 동작 확인에만 사용한다.

### TestHarness 단위 테스트

`KeyedOneInputStreamOperatorTestHarness` 는 `KeyedProcessFunction` 을 클러스터 없이 단위 테스트할 수 있게 해 준다. 핵심 기능:

- `harness.processElement(record, timestamp)`: 이벤트 주입
- `harness.setProcessingTime(ts)`: Processing Time 전진 (타이머 트리거)
- `harness.processWatermark(ts)`: Watermark 전진 (Event Time 타이머 트리거)
- `harness.extractOutputStreamRecords()`: 방출된 결과 수집

`ProcessFunctionTestHarnesses` 유틸리티 클래스를 사용하면 harness 생성 코드를 줄일 수 있다.

### 실무 예제: 타이머 로직 단위 테스트

**시나리오**: 첫 번째 이벤트 수신 후 5초 이내에 두 번째 이벤트가 없으면 알림을 보내는 `KeyedProcessFunction` 을 테스트한다.

### 테스트 전체 코드

```java
// ── 프로덕션 코드 ────────────────────────────────────────────────────────────

import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;

/**
 * 첫 번째 이벤트가 오면 타이머를 등록하고,
 * 5초 이내에 두 번째 이벤트가 오면 타이머를 취소한다.
 * 타이머가 발동하면 (두 번째 이벤트 미도착) 알림을 방출한다.
 */
public class InactivityAlertFunction
        extends KeyedProcessFunction<String, String, String> {

    // 타이머 등록 여부를 추적하는 상태
    private transient ValueState<Long> timerState;

    @Override
    public void open(Configuration parameters) {
        timerState = getRuntimeContext().getState(
            new ValueStateDescriptor<>("timer-ts", Types.LONG)
        );
    }

    @Override
    public void processElement(String event, Context ctx, Collector<String> out)
            throws Exception {

        Long existingTimer = timerState.value();

        if (existingTimer == null) {
            // 첫 번째 이벤트: 현재 처리 시간 기준 5초 후 타이머 등록
            long timerTs = ctx.timerService().currentProcessingTime() + 5_000L;
            ctx.timerService().registerProcessingTimeTimer(timerTs);
            timerState.update(timerTs);
        } else {
            // 두 번째 이벤트: 기존 타이머 취소 및 상태 초기화
            ctx.timerService().deleteProcessingTimeTimer(existingTimer);
            timerState.clear();
        }
    }

    @Override
    public void onTimer(long timestamp, OnTimerContext ctx, Collector<String> out)
            throws Exception {
        // 타이머 발동: 5초 동안 두 번째 이벤트가 없었음
        out.collect("ALERT: key=" + ctx.getCurrentKey() + " 비활성 감지");
        timerState.clear();
    }
}

// ── 테스트 코드 ──────────────────────────────────────────────────────────────

import org.apache.flink.streaming.api.operators.KeyedProcessOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.util.KeyedOneInputStreamOperatorTestHarness;
import org.apache.flink.streaming.util.ProcessFunctionTestHarnesses;

import org.junit.Assert;
import org.junit.Test;

import java.util.List;

public class InactivityAlertFunctionTest {

    /**
     * 시나리오: 첫 이벤트 후 5초 이내에 두 번째 이벤트가 없으면 알림 방출
     */
    @Test
    public void testAlertFiredWhenNoSecondEvent() throws Exception {

        InactivityAlertFunction function = new InactivityAlertFunction();

        // ProcessFunctionTestHarnesses 로 harness 생성
        // 세 번째 인자: 키 추출 함수, 네 번째: 키 타입
        KeyedOneInputStreamOperatorTestHarness<String, String, String> harness =
            ProcessFunctionTestHarnesses.forKeyedProcessFunction(
                function,
                event -> "user1",       // 모든 이벤트는 "user1" 키
                Types.STRING
            );

        harness.open();

        // t=0: 첫 번째 이벤트 주입 (처리 시간 0으로 설정)
        harness.setProcessingTime(0L);
        harness.processElement("event-1", 0L);

        // t=3000: 아직 타이머 미발동 (5초 전)
        harness.setProcessingTime(3_000L);
        Assert.assertTrue("3초 시점에는 알림이 없어야 함",
            harness.extractOutputStreamRecords().isEmpty());

        // t=5001: 타이머 발동 시점 초과
        harness.setProcessingTime(5_001L);

        List<StreamRecord<? extends String>> output = harness.extractOutputStreamRecords();
        Assert.assertEquals("알림 1건이 방출되어야 함", 1, output.size());
        Assert.assertTrue("알림 메시지에 ALERT 포함 확인",
            output.get(0).getValue().contains("ALERT"));

        harness.close();
    }

    /**
     * 시나리오: 5초 이내에 두 번째 이벤트가 오면 알림이 발생하지 않아야 함
     */
    @Test
    public void testNoAlertWhenSecondEventArrives() throws Exception {

        InactivityAlertFunction function = new InactivityAlertFunction();

        KeyedOneInputStreamOperatorTestHarness<String, String, String> harness =
            ProcessFunctionTestHarnesses.forKeyedProcessFunction(
                function,
                event -> "user1",
                Types.STRING
            );

        harness.open();

        // t=0: 첫 번째 이벤트
        harness.setProcessingTime(0L);
        harness.processElement("event-1", 0L);

        // t=2000: 두 번째 이벤트 (5초 이내) — 타이머 취소
        harness.setProcessingTime(2_000L);
        harness.processElement("event-2", 2_000L);

        // t=6000: 타이머가 취소됐으므로 발동 없음
        harness.setProcessingTime(6_000L);

        List<StreamRecord<? extends String>> output = harness.extractOutputStreamRecords();
        Assert.assertTrue("두 번째 이벤트가 오면 알림이 없어야 함", output.isEmpty());

        harness.close();
    }

    /**
     * 통합 테스트 예시: MiniCluster 를 사용해 전체 잡 실행
     */
    @org.junit.ClassRule
    public static org.apache.flink.test.util.MiniClusterWithClientResource flinkCluster =
        new org.apache.flink.test.util.MiniClusterWithClientResource(
            new org.apache.flink.runtime.testutils.MiniClusterResourceConfiguration.Builder()
                .setNumberSlotsPerTaskManager(2)
                .setNumberTaskManagers(1)
                .build()
        );

    @Test
    public void testIntegration() throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(1);

        // 실제 잡 그래프를 구성하고 실행
        // 통합 테스트는 전체 파이프라인의 동작을 검증하는 데 사용
        List<String> results = new java.util.ArrayList<>();

        env.fromElements("event-1")
            .keyBy(e -> "user1")
            .process(new InactivityAlertFunction())
            .addSink(new org.apache.flink.streaming.api.functions.sink.SinkFunction<String>() {
                @Override
                public void invoke(String value, Context context) {
                    results.add(value);
                }
            });

        // 통합 테스트에서 ProcessingTime 타이머를 정확히 제어하려면
        // AutoWatermarkInterval 을 0 으로 설정하고 ManualClock 을 사용해야 한다
        // 여기서는 구조만 보여 주며, 실제 타이머 검증은 TestHarness 로 수행한다
        env.execute("Integration Test");
    }
}
```

---

## 성능 최적화 팁

### Operator Chaining 제어

Flink 는 기본적으로 연속된 Operator 들을 하나의 Task (스레드) 로 묶는다. 이를 **Operator Chaining** 이라고 하며, 직렬화·역직렬화 비용과 스레드 컨텍스트 전환을 줄여 준다.

그러나 특정 Operator 가 CPU 를 독점하거나 독립적으로 스케일링이 필요하다면 체이닝을 끊는 것이 유리하다.

```java
DataStream<String> stream = source
    .map(heavyComputeFunction)
        .startNewChain()        // 이 Operator 부터 새 체인 시작
    .filter(filterFunction)
        .disableChaining()      // 이 Operator 는 앞뒤 모든 체인에서 분리
    .map(lightFunction);

// 잡 전체의 체이닝을 끄려면 (거의 사용하지 않음 — 성능 저하 위험)
env.disableOperatorChaining();
```

언제 체이닝을 끊는가:

- `heavyComputeFunction` 이 CPU 를 많이 쓰고, 뒤에 오는 `filter` 를 별도 병렬도로 실행하고 싶을 때
- 프로파일링 결과 특정 Operator 가 병목일 때 분리해 독립적으로 스케일링

### Serialization 최적화

**POJO 조건 준수**: 앞서 설명한 POJO 규칙을 지키면 Flink 가 바이트 오프셋 기반 TypeSerializer 를 생성해 Kryo 대비 2~5배 빠르다.

```java
// Kryo 폴백 경고를 로그에서 확인하는 방법
// 잡 시작 시 로그에 "Type ... is using a Kryo serializer" 가 찍히면 최적화 필요

// Kryo 를 명시적으로 비활성화해 문제를 조기에 발견
env.getConfig().disableGenericTypes(); // Kryo 폴백 시 예외 발생 (개발 환경에서 유용)
```

**타입 정보 명시**: 람다나 익명 클래스를 반환하는 연산자에는 `.returns()` 로 타입을 명시한다.

```java
stream
    .map(event -> new Tuple2<>(event.userId, event.score))
    .returns(new TypeHint<Tuple2<String, Double>>() {})  // 타입 힌트 없으면 Kryo 폴백
    .keyBy(t -> t.f0);
```

### State 접근 최소화 패턴

State 는 로컬 메모리(HeapStateBackend) 또는 RocksDB 에 저장된다. RocksDB 는 읽기/쓰기가 디스크 I/O 를 포함하므로 불필요한 접근은 피해야 한다.

```java
// 나쁜 패턴: 매 이벤트마다 state 를 두 번 읽음
public void processElement(Event event, Context ctx, Collector<String> out) throws Exception {
    Long count = countState.value();                    // 읽기 1회
    if (count == null) count = 0L;
    countState.update(count + 1);                      // 쓰기 1회
    Long latestCount = countState.value();              // 불필요한 읽기 2회
    if (latestCount > 100) out.collect("threshold");
}

// 좋은 패턴: 로컬 변수에 캐시하고 state 는 최소한으로 접근
public void processElement(Event event, Context ctx, Collector<String> out) throws Exception {
    Long count = countState.value();
    if (count == null) count = 0L;
    count++;                                            // 로컬 변수로 계산
    countState.update(count);                           // 쓰기 1회만
    if (count > 100) out.collect("threshold");
}
```

대규모 맵 상태가 필요하다면 `MapState` 보다 RocksDB 의 Column Family 를 직접 활용하는 `RocksDBStateBackend` 세밀 조정을 검토한다.

### Network Buffer 튜닝

Task 간 데이터 전송은 Network Buffer 를 통해 이루어진다. 버퍼가 너무 작으면 잦은 플러시로 처리량이 낮아지고, 너무 크면 지연(latency)이 증가한다.

주요 설정 (`flink-conf.yaml`):

```yaml
# 네트워크 버퍼 메모리 전체 크기 (기본: 총 JVM 힙의 일부)
taskmanager.network.memory.fraction: 0.1
taskmanager.network.memory.min: 64mb
taskmanager.network.memory.max: 1gb

# 버퍼 크기 (기본 32KB): 줄이면 latency 개선, 늘리면 throughput 개선
taskmanager.memory.segment-size: 32kb

# 출력 버퍼 플러시 간격 (기본: 100ms)
# 낮추면 latency 개선되지만 CPU 비용 증가
execution.buffer-timeout: 100ms
```

코드에서 동적으로 설정:

```java
// 지연 시간 우선: 버퍼 타임아웃을 0 으로 설정 (즉시 플러시)
env.setBufferTimeout(0);

// 처리량 우선: 버퍼 타임아웃을 늘림
env.setBufferTimeout(200); // 200ms 마다 플러시
```

버퍼 튜닝은 프로파일링 결과를 보고 조정한다. 일반적인 권장값은 latency 요건이 100ms 미만이면 `buffer-timeout=0`, 처리량 최대화가 목표면 기본값(100ms)을 유지한다.
