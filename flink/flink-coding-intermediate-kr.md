# Flink 코딩 가이드 — 중급편

## 목차

- [Window 연산](#window-연산)
  - [Tumbling Window](#tumbling-window)
  - [Sliding Window](#sliding-window)
  - [Session Window](#session-window)
  - [Global Window + Custom Trigger](#global-window--custom-trigger)
- [시간 개념 (Time Semantics)](#시간-개념-time-semantics)
  - [Event Time vs Processing Time vs Ingestion Time](#event-time-vs-processing-time-vs-ingestion-time)
  - [WatermarkStrategy 설정](#watermarkstrategy-설정)
  - [늦게 도착한 이벤트 처리](#늦게-도착한-이벤트-처리)
- [상태 관리 (State Management)](#상태-관리-state-management)
  - [ValueState](#valuestate)
  - [ListState](#liststate)
  - [MapState](#mapstate)
  - [StateTTL](#statettl)
- [KeyedProcessFunction](#keyedprocessfunction)
- [체크포인트 프로그래밍](#체크포인트-프로그래밍)
- [여러 스트림 합치기](#여러-스트림-합치기)

---

## Window 연산

Window 연산은 무한한 스트림 데이터를 유한한 묶음으로 나누어 집계하거나 분석할 때 사용한다.
Flink는 Tumbling, Sliding, Session, Global 네 가지 기본 윈도우 유형을 제공한다.

### Tumbling Window

고정 크기의 겹치지 않는 윈도우다. 예를 들어 "5분마다 클릭 수를 집계"할 때 사용한다.
각 이벤트는 정확히 하나의 윈도우에만 속한다.

```java
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;

import java.time.Duration;

public class TumblingWindowExample {

    // 클릭 이벤트를 표현하는 POJO
    public static class ClickEvent {
        public String userId;   // 유저 ID
        public String pageId;   // 페이지 ID
        public long timestamp;  // 이벤트 발생 시각 (epoch millis)

        public ClickEvent() {}

        public ClickEvent(String userId, String pageId, long timestamp) {
            this.userId = userId;
            this.pageId = pageId;
            this.timestamp = timestamp;
        }
    }

    // 페이지별 클릭 수를 집계하는 AggregateFunction
    // IN=ClickEvent, ACC=Long(누적 카운트), OUT=Tuple2<String, Long>(pageId, count)
    public static class ClickCountAggregator
            implements AggregateFunction<ClickEvent, Long, Tuple2<String, Long>> {

        @Override
        public Long createAccumulator() {
            return 0L; // 초기 누적값
        }

        @Override
        public Long add(ClickEvent event, Long accumulator) {
            return accumulator + 1; // 이벤트가 들어올 때마다 카운트 증가
        }

        @Override
        public Tuple2<String, Long> getResult(Long accumulator) {
            // 실제 사용 시에는 WindowFunction으로 pageId를 함께 넘겨야 한다
            // 여기서는 단순화를 위해 placeholder 사용
            return Tuple2.of("page", accumulator);
        }

        @Override
        public Long merge(Long a, Long b) {
            return a + b; // 세션 윈도우 병합 시 사용
        }
    }

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // 예시 데이터 스트림 (실제 환경에서는 Kafka 소스를 사용)
        DataStream<ClickEvent> clicks = env.fromElements(
            new ClickEvent("user1", "home", 1_000L),
            new ClickEvent("user2", "home", 2_000L),
            new ClickEvent("user1", "product", 3_000L),
            new ClickEvent("user3", "home", 4_000L),
            new ClickEvent("user2", "product", 310_000L) // 5분 이후 이벤트
        ).assignTimestampsAndWatermarks(
            // 이벤트 타임스탬프를 기준으로 워터마크를 설정한다
            // 5초 지연을 허용한다 (네트워크 지연 등을 고려)
            WatermarkStrategy
                .<ClickEvent>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                .withTimestampAssigner((event, recordTimestamp) -> event.timestamp)
        );

        DataStream<Long> result = clicks
            .keyBy(event -> event.pageId)       // 페이지별로 키 분류
            .window(TumblingEventTimeWindows.of(Time.minutes(5))) // 5분 단위 텀블링 윈도우
            .aggregate(new ClickCountAggregator()); // 윈도우 내 이벤트 집계

        result.print();
        env.execute("Tumbling Window Click Count");
    }
}
```

### Sliding Window

윈도우가 일정 간격으로 슬라이딩하면서 겹치는 구간이 생긴다.
"5분 윈도우를 1분마다 갱신"하면 각 이벤트는 최대 5개의 윈도우에 속한다.
실시간 이동 평균, 최근 N분 집계 같은 유즈케이스에 적합하다.

```java
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.assigners.SlidingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;

import java.time.Duration;

public class SlidingWindowExample {

    // 서버 메트릭 이벤트
    public static class MetricEvent {
        public String serverId;   // 서버 ID
        public double cpuUsage;   // CPU 사용률 (0.0 ~ 1.0)
        public long timestamp;    // 이벤트 발생 시각

        public MetricEvent() {}

        public MetricEvent(String serverId, double cpuUsage, long timestamp) {
            this.serverId = serverId;
            this.cpuUsage = cpuUsage;
            this.timestamp = timestamp;
        }

        @Override
        public String toString() {
            return String.format("server=%s, cpu=%.2f, ts=%d", serverId, cpuUsage, timestamp);
        }
    }

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<MetricEvent> metrics = env.fromElements(
            new MetricEvent("server-1", 0.4, 0L),
            new MetricEvent("server-1", 0.8, 60_000L),   // 1분 후
            new MetricEvent("server-1", 0.6, 120_000L),  // 2분 후
            new MetricEvent("server-1", 0.9, 180_000L),  // 3분 후
            new MetricEvent("server-1", 0.5, 240_000L)   // 4분 후
        ).assignTimestampsAndWatermarks(
            WatermarkStrategy
                .<MetricEvent>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                .withTimestampAssigner((event, ts) -> event.timestamp)
        );

        // 5분 윈도우를 1분마다 슬라이딩 — 최근 5분간 최대 CPU 사용률 추적
        DataStream<MetricEvent> maxCpuPerWindow = metrics
            .keyBy(event -> event.serverId) // 서버별로 분리
            .window(SlidingEventTimeWindows.of(
                Time.minutes(5), // 윈도우 크기: 5분
                Time.minutes(1)  // 슬라이드 간격: 1분마다 새 윈도우 생성
            ))
            .reduce((a, b) -> {
                // 두 이벤트 중 CPU 사용률이 높은 것을 남긴다
                return a.cpuUsage > b.cpuUsage ? a : b;
            });

        maxCpuPerWindow.print();
        env.execute("Sliding Window CPU Max");
    }
}
```

### Session Window

이벤트 사이의 간격(Gap)을 기준으로 윈도우를 나눈다. 유저가 일정 시간 동안 아무 행동을 하지 않으면
세션이 종료된다. 유저 세션 분석, 장바구니 이탈 감지 같은 행동 분석에 자주 쓰인다.

```java
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.assigners.EventTimeSessionWindows;
import org.apache.flink.streaming.api.windowing.time.Time;

import java.time.Duration;

public class SessionWindowExample {

    // 유저 액션 이벤트
    public static class UserAction {
        public String userId;    // 유저 ID
        public String action;    // 액션 종류 (click, scroll, purchase 등)
        public long timestamp;   // 이벤트 발생 시각

        public UserAction() {}

        public UserAction(String userId, String action, long timestamp) {
            this.userId = userId;
            this.action = action;
            this.timestamp = timestamp;
        }
    }

    // 세션 내 총 액션 수를 집계한다
    public static class SessionActionCounter
            implements AggregateFunction<UserAction, Integer, Integer> {

        @Override
        public Integer createAccumulator() {
            return 0;
        }

        @Override
        public Integer add(UserAction value, Integer accumulator) {
            return accumulator + 1; // 액션 하나당 카운트 증가
        }

        @Override
        public Integer getResult(Integer accumulator) {
            return accumulator;
        }

        @Override
        public Integer merge(Integer a, Integer b) {
            return a + b;
        }
    }

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // user1 은 0~2분에 3번 액션 후 30분 이상 비활동 -> 세션 2개 생성됨
        DataStream<UserAction> actions = env.fromElements(
            new UserAction("user1", "click",    0L),
            new UserAction("user1", "scroll",   30_000L),  // 30초 후
            new UserAction("user1", "click",    90_000L),  // 1분 30초 후
            new UserAction("user1", "purchase", 3_600_000L) // 60분 후 — 새 세션
        ).assignTimestampsAndWatermarks(
            WatermarkStrategy
                .<UserAction>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                .withTimestampAssigner((event, ts) -> event.timestamp)
        );

        // 30분(1800초) 비활동이 있으면 세션 종료
        DataStream<Integer> sessionActionCounts = actions
            .keyBy(event -> event.userId) // 유저별로 독립적인 세션 관리
            .window(EventTimeSessionWindows.withGap(Time.minutes(30))) // 30분 Gap
            .aggregate(new SessionActionCounter());

        sessionActionCounts.print();
        env.execute("Session Window User Actions");
    }
}
```

### Global Window + Custom Trigger

Global Window는 모든 이벤트를 하나의 윈도우에 담는다. 기본적으로는 아무것도 트리거하지 않으므로
반드시 Custom Trigger를 함께 사용해야 한다. "이벤트가 N개 쌓이면 처리" 같은 카운트 기반 로직에 유용하다.

```java
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.assigners.GlobalWindows;
import org.apache.flink.streaming.api.windowing.triggers.Trigger;
import org.apache.flink.streaming.api.windowing.triggers.TriggerResult;
import org.apache.flink.streaming.api.windowing.windows.GlobalWindow;

import java.time.Duration;

public class GlobalWindowCustomTriggerExample {

    public static class OrderEvent {
        public String customerId; // 고객 ID
        public double amount;     // 주문 금액
        public long timestamp;    // 주문 시각

        public OrderEvent() {}

        public OrderEvent(String customerId, double amount, long timestamp) {
            this.customerId = customerId;
            this.amount = amount;
            this.timestamp = timestamp;
        }
    }

    // 이벤트가 3개 쌓이면 윈도우를 트리거하는 커스텀 트리거
    public static class CountTrigger extends Trigger<OrderEvent, GlobalWindow> {

        private final int maxCount; // 트리거 임계값

        public CountTrigger(int maxCount) {
            this.maxCount = maxCount;
        }

        @Override
        public TriggerResult onElement(
                OrderEvent element,
                long timestamp,
                GlobalWindow window,
                TriggerContext ctx) throws Exception {

            // 현재까지 누적된 이벤트 수를 상태에서 읽는다
            ValueState<Integer> countState = ctx.getPartitionedState(
                new ValueStateDescriptor<>("count", Types.INT)
            );

            int count = countState.value() == null ? 0 : countState.value();
            count++;
            countState.update(count);

            if (count >= maxCount) {
                // 임계값에 도달했으면 상태를 초기화하고 FIRE_AND_PURGE 반환
                countState.clear();
                return TriggerResult.FIRE_AND_PURGE; // 결과 방출 후 윈도우 데이터 삭제
            }
            return TriggerResult.CONTINUE; // 아직 임계값 미달 — 계속 누적
        }

        @Override
        public TriggerResult onProcessingTime(long time, GlobalWindow window, TriggerContext ctx) {
            return TriggerResult.CONTINUE; // Processing Time 타이머는 사용 안 함
        }

        @Override
        public TriggerResult onEventTime(long time, GlobalWindow window, TriggerContext ctx) {
            return TriggerResult.CONTINUE; // Event Time 타이머는 사용 안 함
        }

        @Override
        public void clear(GlobalWindow window, TriggerContext ctx) throws Exception {
            // 윈도우가 제거될 때 상태도 함께 정리
            ctx.getPartitionedState(
                new ValueStateDescriptor<>("count", Types.INT)
            ).clear();
        }
    }

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<OrderEvent> orders = env.fromElements(
            new OrderEvent("cust1", 10.0, 1_000L),
            new OrderEvent("cust1", 20.0, 2_000L),
            new OrderEvent("cust1", 15.0, 3_000L), // 3번째 — 트리거 발동
            new OrderEvent("cust1", 30.0, 4_000L),
            new OrderEvent("cust1", 25.0, 5_000L),
            new OrderEvent("cust1", 50.0, 6_000L)  // 3번째 — 트리거 발동
        ).assignTimestampsAndWatermarks(
            WatermarkStrategy
                .<OrderEvent>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                .withTimestampAssigner((event, ts) -> event.timestamp)
        );

        // 고객별 3건 주문이 쌓일 때마다 합산 금액을 출력한다
        DataStream<Double> batchTotal = orders
            .keyBy(event -> event.customerId)
            .window(GlobalWindows.create())          // 글로벌 윈도우 — 끝이 없음
            .trigger(new CountTrigger(3))            // 3개마다 트리거
            .reduce((a, b) -> {
                // 두 주문의 금액을 합산 (customerId는 동일하므로 a를 기준으로)
                return new OrderEvent(a.customerId, a.amount + b.amount, a.timestamp);
            })
            .map(event -> event.amount);             // 금액만 추출

        batchTotal.print();
        env.execute("Global Window Batch Order Total");
    }
}
```

---

## 시간 개념 (Time Semantics)

### Event Time vs Processing Time vs Ingestion Time

Flink는 세 가지 시간 개념을 지원한다.

| 시간 종류 | 의미 | 장점 | 단점 |
|---|---|---|---|
| Event Time | 이벤트가 실제로 발생한 시각 | 결과의 정확성 보장 | 늦게 도착한 이벤트 처리 필요 |
| Processing Time | Flink가 이벤트를 처리하는 시각 | 구현이 단순함 | 네트워크 지연에 따라 결과가 달라짐 |
| Ingestion Time | 이벤트가 Flink 소스에 도착한 시각 | Processing Time보다 일관성 있음 | Event Time보다 부정확 |

실무에서 Event Time을 써야 하는 이유는 다음과 같다.

- Kafka 파티션마다 이벤트 도달 순서가 다를 수 있다.
- 모바일 앱처럼 오프라인 상태에서 이벤트가 생성된 뒤 나중에 전송되는 경우가 있다.
- 재처리(Reprocessing) 시 Processing Time을 쓰면 과거 데이터를 올바르게 집계할 수 없다.
- A/B 테스트나 지표 집계에서 시간 기준이 흔들리면 의사결정이 틀어진다.

### WatermarkStrategy 설정

워터마크는 "이 시각 이전의 이벤트는 모두 도착했다"는 신호다.
Flink는 워터마크를 기준으로 윈도우를 닫는다.

```java
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.eventtime.SerializableTimestampAssigner;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;

import java.time.Duration;

public class WatermarkStrategyExample {

    public static class SensorReading {
        public String sensorId;  // 센서 ID
        public double value;     // 측정값
        public long eventTime;   // 센서에서 측정한 실제 시각

        public SensorReading() {}

        public SensorReading(String sensorId, double value, long eventTime) {
            this.sensorId = sensorId;
            this.value = value;
            this.eventTime = eventTime;
        }
    }

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<SensorReading> readings = env.fromElements(
            new SensorReading("sensor-A", 23.5, 1_000L),
            new SensorReading("sensor-A", 24.1, 5_000L),
            new SensorReading("sensor-A", 22.8, 3_000L), // 순서 뒤바뀐 이벤트
            new SensorReading("sensor-B", 18.0, 2_000L)
        );

        // 방법 1: BoundedOutOfOrderness — 최대 N초 지연을 허용
        // 가장 일반적인 전략. 실무에서는 95th percentile latency를 기준으로 설정한다.
        WatermarkStrategy<SensorReading> boundedStrategy =
            WatermarkStrategy
                .<SensorReading>forBoundedOutOfOrderness(Duration.ofSeconds(10))
                .withTimestampAssigner(
                    (SerializableTimestampAssigner<SensorReading>)
                        (event, recordTimestamp) -> event.eventTime
                );

        // 방법 2: Monotonously Increasing — 이벤트가 항상 순서대로 온다고 가정
        // 지연 없이 즉시 워터마크를 발행하므로 레이턴시가 낮지만, 순서가 보장될 때만 사용한다.
        WatermarkStrategy<SensorReading> monotonicStrategy =
            WatermarkStrategy
                .<SensorReading>forMonotonousTimestamps()
                .withTimestampAssigner(
                    (SerializableTimestampAssigner<SensorReading>)
                        (event, recordTimestamp) -> event.eventTime
                );

        // 실제 파이프라인에 적용
        DataStream<SensorReading> withWatermarks = readings
            .assignTimestampsAndWatermarks(boundedStrategy);

        // 1분 텀블링 윈도우로 센서별 최대 측정값 계산
        withWatermarks
            .keyBy(r -> r.sensorId)
            .window(TumblingEventTimeWindows.of(Time.minutes(1)))
            .reduce((a, b) -> a.value > b.value ? a : b)
            .print();

        env.execute("Watermark Strategy Example");
    }
}
```

### 늦게 도착한 이벤트 처리

워터마크 이후에도 이벤트가 늦게 도착할 수 있다. Flink는 두 가지 방법으로 처리한다.

- `allowedLateness`: 윈도우 종료 후에도 N시간 동안 윈도우를 메모리에 유지해서 늦은 이벤트를 재반영한다.
- `sideOutputLateData`: 늦은 이벤트를 별도 스트림으로 빼서 모니터링하거나 별도 처리한다.

```java
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.util.OutputTag;

import java.time.Duration;

public class LateDataHandlingExample {

    public static class PurchaseEvent {
        public String userId;  // 유저 ID
        public double amount;  // 구매 금액
        public long eventTime; // 구매 발생 시각

        public PurchaseEvent() {}

        public PurchaseEvent(String userId, double amount, long eventTime) {
            this.userId = userId;
            this.amount = amount;
            this.eventTime = eventTime;
        }

        @Override
        public String toString() {
            return String.format("userId=%s, amount=%.2f, ts=%d", userId, amount, eventTime);
        }
    }

    // 늦게 도착한 이벤트를 담을 사이드 출력 태그
    static final OutputTag<PurchaseEvent> LATE_OUTPUT_TAG =
        new OutputTag<PurchaseEvent>("late-purchases") {};

    // 구매 금액 합산
    public static class AmountSumAggregator
            implements AggregateFunction<PurchaseEvent, Double, Double> {

        @Override
        public Double createAccumulator() {
            return 0.0;
        }

        @Override
        public Double add(PurchaseEvent value, Double accumulator) {
            return accumulator + value.amount;
        }

        @Override
        public Double getResult(Double accumulator) {
            return accumulator;
        }

        @Override
        public Double merge(Double a, Double b) {
            return a + b;
        }
    }

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<PurchaseEvent> purchases = env.fromElements(
            new PurchaseEvent("user1", 100.0, 0L),
            new PurchaseEvent("user1", 200.0, 30_000L),   // 30초 후
            new PurchaseEvent("user1", 50.0,  65_000L),   // 1분 5초 후 — 2번째 윈도우
            new PurchaseEvent("user1", 300.0, 55_000L)    // 55초 — 첫 윈도우 종료 후 늦게 도착
        ).assignTimestampsAndWatermarks(
            WatermarkStrategy
                .<PurchaseEvent>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                .withTimestampAssigner((event, ts) -> event.eventTime)
        );

        SingleOutputStreamOperator<Double> windowResult = purchases
            .keyBy(event -> event.userId)
            .window(TumblingEventTimeWindows.of(Time.minutes(1)))
            // 워터마크 통과 후 최대 30초까지 늦은 이벤트를 윈도우에 반영한다
            // 이 기간이 지나면 sideOutput으로 빠진다
            .allowedLateness(Time.seconds(30))
            // allowedLateness 기간도 지난 이벤트는 LATE_OUTPUT_TAG 스트림으로 분리
            .sideOutputLateData(LATE_OUTPUT_TAG)
            .aggregate(new AmountSumAggregator());

        // 정상 결과 출력
        windowResult.print("NORMAL");

        // 너무 늦게 도착한 이벤트를 별도로 처리 (예: 알람, DLQ 저장)
        DataStream<PurchaseEvent> lateEvents = windowResult.getSideOutput(LATE_OUTPUT_TAG);
        lateEvents.print("LATE");

        env.execute("Late Data Handling");
    }
}
```

---

## 상태 관리 (State Management)

Flink의 상태(State)는 키 단위로 관리되며 체크포인트를 통해 내결함성(fault tolerance)을 보장한다.
상태는 반드시 `RichFunction` 혹은 `ProcessFunction` 안에서 선언해야 한다.

### ValueState

단일 값을 저장한다. 유저별 마지막 로그인 시각처럼 "현재 값 하나"를 추적할 때 사용한다.

```java
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;

import java.time.Duration;

public class ValueStateExample {

    public static class LoginEvent {
        public String userId;   // 유저 ID
        public long timestamp;  // 로그인 시각

        public LoginEvent() {}

        public LoginEvent(String userId, long timestamp) {
            this.userId = userId;
            this.timestamp = timestamp;
        }
    }

    // 유저별 마지막 로그인 시각을 ValueState로 추적
    public static class LastLoginTracker
            extends KeyedProcessFunction<String, LoginEvent, String> {

        // ValueState 선언 — open() 에서 초기화한다
        private ValueState<Long> lastLoginTimeState;

        @Override
        public void open(Configuration parameters) {
            // 상태 디스크립터: 이름과 타입을 지정한다
            ValueStateDescriptor<Long> descriptor =
                new ValueStateDescriptor<>("lastLoginTime", Types.LONG);
            lastLoginTimeState = getRuntimeContext().getState(descriptor);
        }

        @Override
        public void processElement(
                LoginEvent event,
                Context ctx,
                Collector<String> out) throws Exception {

            Long previousLogin = lastLoginTimeState.value(); // 이전 로그인 시각 읽기

            if (previousLogin == null) {
                // 첫 로그인
                out.collect(String.format("[%s] 첫 로그인: %d", event.userId, event.timestamp));
            } else {
                long gapSeconds = (event.timestamp - previousLogin) / 1000;
                out.collect(String.format(
                    "[%s] 재로그인 — 이전 로그인으로부터 %d초 경과",
                    event.userId, gapSeconds
                ));
            }

            // 현재 로그인 시각으로 상태 갱신
            lastLoginTimeState.update(event.timestamp);
        }
    }

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<LoginEvent> logins = env.fromElements(
            new LoginEvent("user1", 1_000L),
            new LoginEvent("user2", 2_000L),
            new LoginEvent("user1", 3_600_000L), // user1 이 1시간 후 재로그인
            new LoginEvent("user2", 7_200_000L)  // user2 가 2시간 후 재로그인
        ).assignTimestampsAndWatermarks(
            WatermarkStrategy
                .<LoginEvent>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                .withTimestampAssigner((e, ts) -> e.timestamp)
        );

        logins
            .keyBy(event -> event.userId)
            .process(new LastLoginTracker())
            .print();

        env.execute("ValueState Last Login");
    }
}
```

### ListState

리스트를 저장한다. 유저의 최근 행동 이력처럼 "N개의 항목을 누적"할 때 사용한다.

```java
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.List;

public class ListStateExample {

    public static class PageViewEvent {
        public String userId;  // 유저 ID
        public String pageId;  // 조회한 페이지 ID
        public long timestamp; // 조회 시각

        public PageViewEvent() {}

        public PageViewEvent(String userId, String pageId, long timestamp) {
            this.userId = userId;
            this.pageId = pageId;
            this.timestamp = timestamp;
        }
    }

    // 유저별 최근 5개 페이지 뷰 이력을 ListState에 유지한다
    public static class RecentPageViewTracker
            extends KeyedProcessFunction<String, PageViewEvent, String> {

        private ListState<String> recentPagesState; // 최근 페이지 ID 목록 저장
        private static final int MAX_HISTORY = 5;  // 최대 유지 건수

        @Override
        public void open(Configuration parameters) {
            ListStateDescriptor<String> descriptor =
                new ListStateDescriptor<>("recentPages", Types.STRING);
            recentPagesState = getRuntimeContext().getListState(descriptor);
        }

        @Override
        public void processElement(
                PageViewEvent event,
                Context ctx,
                Collector<String> out) throws Exception {

            // 기존 이력 읽기
            List<String> history = new ArrayList<>();
            for (String page : recentPagesState.get()) {
                history.add(page);
            }

            // 새 페이지 추가
            history.add(event.pageId);

            // 최대 건수 초과 시 가장 오래된 항목 제거
            if (history.size() > MAX_HISTORY) {
                history.remove(0);
            }

            // 상태 업데이트
            recentPagesState.update(history);

            out.collect(String.format("[%s] 최근 방문 페이지: %s", event.userId, history));
        }
    }

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<PageViewEvent> pageViews = env.fromElements(
            new PageViewEvent("user1", "home",    1_000L),
            new PageViewEvent("user1", "search",  2_000L),
            new PageViewEvent("user1", "product", 3_000L),
            new PageViewEvent("user1", "cart",    4_000L),
            new PageViewEvent("user1", "checkout",5_000L),
            new PageViewEvent("user1", "confirm", 6_000L) // 6번째 — home 이 밀려남
        );

        pageViews
            .keyBy(event -> event.userId)
            .process(new RecentPageViewTracker())
            .print();

        env.execute("ListState Recent Pages");
    }
}
```

### MapState

키-값 맵을 저장한다. 유저별 카테고리 방문 횟수처럼 "키별 카운터"가 필요할 때 사용한다.

```java
import org.apache.flink.api.common.state.MapState;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;

import java.util.HashMap;
import java.util.Map;

public class MapStateExample {

    public static class ProductViewEvent {
        public String userId;    // 유저 ID
        public String category;  // 상품 카테고리 (예: electronics, fashion)
        public long timestamp;   // 조회 시각

        public ProductViewEvent() {}

        public ProductViewEvent(String userId, String category, long timestamp) {
            this.userId = userId;
            this.category = category;
            this.timestamp = timestamp;
        }
    }

    // 유저별로 카테고리 조회 횟수를 MapState에 누적한다
    public static class CategoryViewCounter
            extends KeyedProcessFunction<String, ProductViewEvent, String> {

        // MapState<카테고리, 조회횟수>
        private MapState<String, Integer> categoryCountState;

        @Override
        public void open(Configuration parameters) {
            MapStateDescriptor<String, Integer> descriptor =
                new MapStateDescriptor<>("categoryCount", Types.STRING, Types.INT);
            categoryCountState = getRuntimeContext().getMapState(descriptor);
        }

        @Override
        public void processElement(
                ProductViewEvent event,
                Context ctx,
                Collector<String> out) throws Exception {

            // 해당 카테고리의 현재 카운트 조회 (없으면 0)
            Integer currentCount = categoryCountState.get(event.category);
            if (currentCount == null) {
                currentCount = 0;
            }

            // 카운트 증가 후 저장
            categoryCountState.put(event.category, currentCount + 1);

            // 전체 카테고리 분포 출력
            Map<String, Integer> snapshot = new HashMap<>();
            for (Map.Entry<String, Integer> entry : categoryCountState.entries()) {
                snapshot.put(entry.getKey(), entry.getValue());
            }
            out.collect(String.format("[%s] 카테고리별 조회 수: %s", event.userId, snapshot));
        }
    }

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<ProductViewEvent> events = env.fromElements(
            new ProductViewEvent("user1", "electronics", 1_000L),
            new ProductViewEvent("user1", "fashion",     2_000L),
            new ProductViewEvent("user1", "electronics", 3_000L),
            new ProductViewEvent("user1", "books",       4_000L),
            new ProductViewEvent("user1", "fashion",     5_000L)
        );

        events
            .keyBy(event -> event.userId)
            .process(new CategoryViewCounter())
            .print();

        env.execute("MapState Category Counter");
    }
}
```

### StateTTL

상태가 무한히 쌓이면 메모리가 부족해진다. StateTTL을 설정하면 마지막으로 접근한 이후 일정 시간이
지난 상태를 자동으로 삭제한다. 오래된 세션 데이터, 캐시 등에 유용하다.

```java
import org.apache.flink.api.common.state.StateTtlConfig;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.time.Time;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;

public class StateTtlExample {

    public static class UserEvent {
        public String userId;
        public String eventType; // "login", "action", "logout"
        public long timestamp;

        public UserEvent() {}

        public UserEvent(String userId, String eventType, long timestamp) {
            this.userId = userId;
            this.eventType = eventType;
            this.timestamp = timestamp;
        }
    }

    // 유저 세션 상태 — 30분 비활동 시 자동 만료
    public static class SessionStateTracker
            extends KeyedProcessFunction<String, UserEvent, String> {

        private ValueState<String> sessionState; // 현재 세션 상태 (active/expired)

        @Override
        public void open(Configuration parameters) {
            // TTL 설정: 마지막 갱신 후 30분이 지나면 상태 삭제
            StateTtlConfig ttlConfig = StateTtlConfig
                .newBuilder(Time.minutes(30))
                // 읽기 및 쓰기 시 TTL 갱신 (접근할 때마다 만료 시간 리셋)
                .setUpdateType(StateTtlConfig.UpdateType.OnCreateAndWrite)
                // 만료된 상태를 읽으면 null을 반환한다
                .setStateVisibility(StateTtlConfig.StateVisibility.NeverReturnExpired)
                .build();

            ValueStateDescriptor<String> descriptor =
                new ValueStateDescriptor<>("sessionStatus", Types.STRING);

            // TTL 설정을 디스크립터에 적용한다
            descriptor.enableTimeToLive(ttlConfig);

            sessionState = getRuntimeContext().getState(descriptor);
        }

        @Override
        public void processElement(
                UserEvent event,
                Context ctx,
                Collector<String> out) throws Exception {

            String currentStatus = sessionState.value(); // TTL 만료 시 null 반환

            if (currentStatus == null) {
                // 상태가 없거나 만료됨 — 새 세션 시작
                sessionState.update("active");
                out.collect(String.format("[%s] 새 세션 시작 (이벤트: %s)", event.userId, event.eventType));
            } else {
                // 기존 세션 유지 — 상태에 쓰는 것만으로 TTL이 갱신된다
                sessionState.update("active");
                out.collect(String.format("[%s] 세션 유지 중 (이벤트: %s)", event.userId, event.eventType));
            }
        }
    }

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<UserEvent> events = env.fromElements(
            new UserEvent("user1", "login",  1_000L),
            new UserEvent("user1", "action", 2_000L),
            new UserEvent("user1", "action", 3_000L)
            // 실제로 30분 이상 경과한 이벤트가 오면 TTL로 상태가 사라진다
        );

        events
            .keyBy(event -> event.userId)
            .process(new SessionStateTracker())
            .print();

        env.execute("StateTTL Session Tracking");
    }
}
```

---

## KeyedProcessFunction

`KeyedProcessFunction`은 Flink에서 가장 강력한 저수준 API다.
이벤트 처리(`processElement`)와 타이머(`onTimer`)를 함께 사용해 복잡한 비즈니스 로직을 구현할 수 있다.

- `processElement`: 이벤트가 도착할 때마다 호출된다.
- `onTimer`: 등록한 타이머가 만료될 때 호출된다.
  - Event Time Timer: 워터마크가 특정 시각을 지날 때 발동
  - Processing Time Timer: 현재 시계 기준으로 N초 후 발동

실무 예제: "매칭 후 30분 이내에 대화 메시지를 보내지 않으면 푸시 알림 발송"

```java
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;

import java.time.Duration;

public class MatchReminderExample {

    // 이벤트 타입 상수
    public static final String EVENT_MATCH   = "MATCH";   // 매칭 성사
    public static final String EVENT_MESSAGE = "MESSAGE"; // 메시지 전송

    public static class UserActivityEvent {
        public String userId;    // 유저 ID
        public String eventType; // MATCH or MESSAGE
        public long timestamp;   // 이벤트 발생 시각

        public UserActivityEvent() {}

        public UserActivityEvent(String userId, String eventType, long timestamp) {
            this.userId = userId;
            this.eventType = eventType;
            this.timestamp = timestamp;
        }
    }

    // 매칭 후 30분 이내 메시지 미발송 시 알림을 생성하는 ProcessFunction
    public static class MatchReminderFunction
            extends KeyedProcessFunction<String, UserActivityEvent, String> {

        // 타이머 등록 여부와 등록된 시각을 저장
        // null이면 타이머 없음, 값이 있으면 그 시각에 타이머가 등록됨
        private ValueState<Long> timerState;

        // 매칭 후 알림까지의 대기 시간 (30분 = 밀리초)
        private static final long REMINDER_DELAY_MS = 30 * 60 * 1000L;

        @Override
        public void open(Configuration parameters) {
            ValueStateDescriptor<Long> descriptor =
                new ValueStateDescriptor<>("matchTimer", Types.LONG);
            timerState = getRuntimeContext().getState(descriptor);
        }

        @Override
        public void processElement(
                UserActivityEvent event,
                Context ctx,
                Collector<String> out) throws Exception {

            if (EVENT_MATCH.equals(event.eventType)) {
                // 매칭 이벤트 도착 — 30분 후에 타이머를 등록한다
                long reminderTime = event.timestamp + REMINDER_DELAY_MS;

                // 기존에 등록된 타이머가 있으면 취소한다
                Long existingTimer = timerState.value();
                if (existingTimer != null) {
                    ctx.timerService().deleteEventTimeTimer(existingTimer);
                }

                // 새 타이머 등록
                ctx.timerService().registerEventTimeTimer(reminderTime);
                timerState.update(reminderTime);

                out.collect(String.format(
                    "[%s] 매칭 성사. %d ms 후 미발송 알림 예약",
                    event.userId, REMINDER_DELAY_MS
                ));

            } else if (EVENT_MESSAGE.equals(event.eventType)) {
                // 메시지 이벤트 도착 — 등록된 타이머를 취소한다
                Long registeredTimer = timerState.value();
                if (registeredTimer != null) {
                    ctx.timerService().deleteEventTimeTimer(registeredTimer);
                    timerState.clear();
                    out.collect(String.format("[%s] 메시지 전송됨. 알림 취소.", event.userId));
                }
            }
        }

        @Override
        public void onTimer(
                long timestamp,
                OnTimerContext ctx,
                Collector<String> out) throws Exception {

            // 30분이 지났는데도 메시지가 없었다 — 알림을 발송한다
            // getCurrentKey()로 현재 유저 ID를 알 수 있다
            String userId = ctx.getCurrentKey();
            out.collect(String.format(
                "[PUSH] [%s] 매칭 후 30분이 지났어요! 먼저 인사를 건네보세요.",
                userId
            ));

            // 상태 정리 (타이머는 이미 발동했으므로 상태만 지우면 됨)
            timerState.clear();
        }
    }

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        long baseTime = 0L;
        long thirtyOneMin = 31 * 60 * 1000L; // 31분 (밀리초)

        DataStream<UserActivityEvent> events = env.fromElements(
            // user1: 매칭 후 31분이 지나도 메시지 없음 -> 알림 발송
            new UserActivityEvent("user1", EVENT_MATCH,   baseTime),
            new UserActivityEvent("user1", EVENT_MESSAGE, baseTime + thirtyOneMin),

            // user2: 매칭 후 10분 안에 메시지 전송 -> 알림 취소
            new UserActivityEvent("user2", EVENT_MATCH,   baseTime),
            new UserActivityEvent("user2", EVENT_MESSAGE, baseTime + 10 * 60 * 1000L)
        ).assignTimestampsAndWatermarks(
            WatermarkStrategy
                .<UserActivityEvent>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                .withTimestampAssigner((e, ts) -> e.timestamp)
        );

        events
            .keyBy(event -> event.userId) // 유저별로 독립적으로 타이머를 관리한다
            .process(new MatchReminderFunction())
            .print();

        env.execute("Match Reminder with KeyedProcessFunction");
    }
}
```

---

## 체크포인트 프로그래밍

체크포인트는 Flink 잡의 상태를 주기적으로 영구 저장소에 스냅샷으로 저장한다.
장애가 발생하면 마지막 체크포인트 시점으로 복구해 데이터 처리를 재개한다.

### 체크포인트 설정 코드

```java
import org.apache.flink.api.common.restartstrategy.RestartStrategies;
import org.apache.flink.api.common.time.Time;
import org.apache.flink.runtime.state.filesystem.FsStateBackend;
import org.apache.flink.streaming.api.CheckpointingMode;
import org.apache.flink.streaming.api.environment.CheckpointConfig;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public class CheckpointConfigExample {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // 1. 체크포인트 활성화 및 기본 설정
        env.enableCheckpointing(60_000L); // 60초마다 체크포인트 수행

        CheckpointConfig checkpointConfig = env.getCheckpointConfig();

        // Exactly-once: 각 레코드가 정확히 한 번 처리됨을 보장한다
        // At-least-once보다 레이턴시가 높지만 데이터 정합성이 필요할 때 사용한다
        checkpointConfig.setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE);

        // 두 체크포인트 사이의 최소 간격 (ms)
        // 이전 체크포인트가 끝난 후 최소 30초 뒤에 다음 체크포인트를 시작한다
        checkpointConfig.setMinPauseBetweenCheckpoints(30_000L);

        // 체크포인트 타임아웃 — 이 시간 안에 완료되지 않으면 취소한다
        checkpointConfig.setCheckpointTimeout(120_000L);

        // 동시에 진행 가능한 체크포인트 수 (기본 1)
        checkpointConfig.setMaxConcurrentCheckpoints(1);

        // 잡이 취소되어도 체크포인트를 삭제하지 않는다 (수동 복구에 유용)
        checkpointConfig.enableExternalizedCheckpoints(
            CheckpointConfig.ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION
        );

        // 2. 상태 백엔드 설정
        // FsStateBackend: 상태를 JVM 힙에 두고 체크포인트만 파일 시스템에 저장
        // 상태가 크면 RocksDB 백엔드(EmbeddedRocksDBStateBackend)를 권장한다
        env.setStateBackend(new FsStateBackend("s3://my-bucket/flink/checkpoints"));

        // 3. 재시작 전략 설정
        // 3번까지 재시작, 각 시도 사이에 10초 대기
        env.setRestartStrategy(
            RestartStrategies.fixedDelayRestart(3, Time.seconds(10))
        );

        // 또는 지수 백오프 전략 (실무에서 더 자주 사용)
        // env.setRestartStrategy(
        //     RestartStrategies.exponentialDelayRestart(
        //         Time.seconds(1),   // 초기 대기 시간
        //         Time.minutes(5),   // 최대 대기 시간
        //         2.0,               // 지수 기저
        //         Time.hours(1),     // 이 시간 동안 장애 없으면 재시도 카운터 리셋
        //         0.1                // 지터 (무작위 오차) 비율
        //     )
        // );

        // 이후 파이프라인 구성 및 실행
        env.execute("Checkpoint Config Example");
    }
}
```

### Exactly-once vs At-least-once

```
Exactly-once
  - 장애 복구 시 각 레코드가 정확히 한 번만 결과에 반영된다.
  - Kafka 소스/싱크와 함께 쓰면 트랜잭션을 활용해 end-to-end exactly-once를 달성할 수 있다.
  - 체크포인트 배리어가 모든 파이프라인 오퍼레이터를 정렬할 때까지 기다려야 하므로 레이턴시가 증가한다.

At-least-once
  - 장애 복구 시 일부 레코드가 중복 처리될 수 있다.
  - 배리어 정렬을 건너뛰므로 레이턴시가 낮다.
  - 멱등(idempotent)한 싱크나 중복을 허용할 수 있는 경우에 적합하다.
```

### 체크포인트와 Kafka 트랜잭션 연동

Kafka Sink에 EXACTLY_ONCE 시맨틱을 설정하면 Flink 체크포인트와 Kafka 트랜잭션이 연동된다.
체크포인트가 완료될 때만 Kafka 트랜잭션을 커밋하므로 싱크까지 exactly-once가 보장된다.

```java
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.connector.kafka.sink.KafkaRecordSerializationSchema;
import org.apache.flink.connector.kafka.sink.KafkaSink;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.streaming.api.CheckpointingMode;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public class KafkaExactlyOnceExample {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // Exactly-once 체크포인트 활성화 — Kafka 트랜잭션 연동의 전제 조건
        env.enableCheckpointing(30_000L, CheckpointingMode.EXACTLY_ONCE);

        // Kafka 소스 설정
        KafkaSource<String> source = KafkaSource.<String>builder()
            .setBootstrapServers("kafka:9092")
            .setTopics("input-topic")
            .setGroupId("flink-consumer-group")
            .setStartingOffsets(OffsetsInitializer.latest())
            .setValueOnlyDeserializer(new SimpleStringSchema())
            .build();

        // Kafka 싱크 — EXACTLY_ONCE 시맨틱 설정
        KafkaSink<String> sink = KafkaSink.<String>builder()
            .setBootstrapServers("kafka:9092")
            // EXACTLY_ONCE: 체크포인트 완료 시에만 Kafka 트랜잭션 커밋
            // transactional.id.prefix를 고유하게 설정해야 재시작 시 이전 트랜잭션과 충돌하지 않는다
            .setDeliveryGuarantee(
                org.apache.flink.connector.kafka.sink.DeliveryGuarantee.EXACTLY_ONCE
            )
            .setTransactionalIdPrefix("flink-txn-")
            .setRecordSerializer(
                KafkaRecordSerializationSchema.builder()
                    .setTopic("output-topic")
                    .setValueSerializationSchema(new SimpleStringSchema())
                    .build()
            )
            .build();

        DataStream<String> stream = env.fromSource(
            source,
            org.apache.flink.api.common.eventtime.WatermarkStrategy.noWatermarks(),
            "Kafka Source"
        );

        // 간단한 변환 후 Kafka 싱크에 쓴다
        stream
            .map(value -> "[processed] " + value)
            .sinkTo(sink);

        env.execute("Kafka Exactly-Once Pipeline");
    }
}
```

---

## 여러 스트림 합치기

실무에서는 여러 소스에서 오는 스트림을 합쳐야 할 때가 많다.
Flink는 `union`과 `connect` 두 가지 방법을 제공한다.

### union: 같은 타입 스트림 합치기

`union`은 동일한 데이터 타입의 스트림 여러 개를 하나로 합친다.
단순 병합이므로 두 스트림을 구별하는 별도 로직은 없다.

```java
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.time.Duration;

public class UnionStreamExample {

    public static class AppEvent {
        public String source;    // 이벤트 발생 앱 (android/ios)
        public String userId;    // 유저 ID
        public String eventType; // 이벤트 종류
        public long timestamp;   // 발생 시각

        public AppEvent() {}

        public AppEvent(String source, String userId, String eventType, long timestamp) {
            this.source = source;
            this.userId = userId;
            this.eventType = eventType;
            this.timestamp = timestamp;
        }

        @Override
        public String toString() {
            return String.format("[%s] %s: %s at %d", source, userId, eventType, timestamp);
        }
    }

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // Android 앱 이벤트 스트림
        DataStream<AppEvent> androidEvents = env.fromElements(
            new AppEvent("android", "user1", "click", 1_000L),
            new AppEvent("android", "user3", "purchase", 3_000L)
        ).assignTimestampsAndWatermarks(
            WatermarkStrategy
                .<AppEvent>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                .withTimestampAssigner((e, ts) -> e.timestamp)
        );

        // iOS 앱 이벤트 스트림
        DataStream<AppEvent> iosEvents = env.fromElements(
            new AppEvent("ios", "user2", "click", 2_000L),
            new AppEvent("ios", "user4", "purchase", 4_000L)
        ).assignTimestampsAndWatermarks(
            WatermarkStrategy
                .<AppEvent>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                .withTimestampAssigner((e, ts) -> e.timestamp)
        );

        // 두 스트림을 union으로 병합 — 같은 타입이기 때문에 바로 합칠 수 있다
        DataStream<AppEvent> allEvents = androidEvents.union(iosEvents);

        allEvents
            .filter(e -> "purchase".equals(e.eventType)) // 구매 이벤트만 필터
            .print();

        env.execute("Union Stream Example");
    }
}
```

### connect: 다른 타입 스트림 합치기 (CoProcessFunction)

`connect`는 타입이 다른 두 스트림을 연결한다. 각 스트림에 대해 별도의 처리 로직을 정의할 수 있어
"설정 스트림 + 데이터 스트림" 같은 패턴에 유용하다.

실무 예제: 클릭 스트림과 구매 스트림을 합쳐 전환율(클릭 -> 구매 비율) 계산

```java
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.co.CoProcessFunction;
import org.apache.flink.util.Collector;

import java.time.Duration;

public class ConnectStreamExample {

    // 클릭 이벤트
    public static class ClickEvent {
        public String userId;  // 유저 ID
        public String itemId;  // 클릭한 상품 ID
        public long timestamp; // 클릭 시각

        public ClickEvent() {}

        public ClickEvent(String userId, String itemId, long timestamp) {
            this.userId = userId;
            this.itemId = itemId;
            this.timestamp = timestamp;
        }
    }

    // 구매 이벤트
    public static class PurchaseEvent {
        public String userId;  // 유저 ID
        public String itemId;  // 구매한 상품 ID
        public long timestamp; // 구매 시각

        public PurchaseEvent() {}

        public PurchaseEvent(String userId, String itemId, long timestamp) {
            this.userId = userId;
            this.itemId = itemId;
            this.timestamp = timestamp;
        }
    }

    // 전환율 계산 결과
    public static class ConversionStats {
        public String userId;    // 유저 ID
        public int clickCount;   // 총 클릭 수
        public int purchaseCount; // 총 구매 수

        public ConversionStats(String userId, int clickCount, int purchaseCount) {
            this.userId = userId;
            this.clickCount = clickCount;
            this.purchaseCount = purchaseCount;
        }

        @Override
        public String toString() {
            double rate = clickCount > 0 ? (double) purchaseCount / clickCount * 100 : 0;
            return String.format("[%s] 클릭: %d, 구매: %d, 전환율: %.1f%%",
                userId, clickCount, purchaseCount, rate);
        }
    }

    // 두 스트림을 처리하는 CoProcessFunction
    // 첫 번째 타입: ClickEvent, 두 번째 타입: PurchaseEvent, 출력 타입: ConversionStats
    public static class ConversionRateFunction
            extends CoProcessFunction<ClickEvent, PurchaseEvent, ConversionStats> {

        private ValueState<Integer> clickCountState;    // 유저별 클릭 수
        private ValueState<Integer> purchaseCountState; // 유저별 구매 수

        @Override
        public void open(Configuration parameters) {
            clickCountState = getRuntimeContext().getState(
                new ValueStateDescriptor<>("clickCount", Types.INT)
            );
            purchaseCountState = getRuntimeContext().getState(
                new ValueStateDescriptor<>("purchaseCount", Types.INT)
            );
        }

        // 첫 번째 스트림(ClickEvent)의 이벤트 처리
        @Override
        public void processElement1(
                ClickEvent click,
                Context ctx,
                Collector<ConversionStats> out) throws Exception {

            Integer clicks = clickCountState.value();
            clicks = clicks == null ? 0 : clicks;
            clicks++;
            clickCountState.update(clicks);

            Integer purchases = purchaseCountState.value();
            purchases = purchases == null ? 0 : purchases;

            // 클릭이 발생할 때마다 현재 전환율 출력
            out.collect(new ConversionStats(click.userId, clicks, purchases));
        }

        // 두 번째 스트림(PurchaseEvent)의 이벤트 처리
        @Override
        public void processElement2(
                PurchaseEvent purchase,
                Context ctx,
                Collector<ConversionStats> out) throws Exception {

            Integer purchases = purchaseCountState.value();
            purchases = purchases == null ? 0 : purchases;
            purchases++;
            purchaseCountState.update(purchases);

            Integer clicks = clickCountState.value();
            clicks = clicks == null ? 0 : clicks;

            // 구매가 발생할 때마다 현재 전환율 출력
            out.collect(new ConversionStats(purchase.userId, clicks, purchases));
        }
    }

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // 클릭 스트림 — user1 이 5번 클릭
        DataStream<ClickEvent> clicks = env.fromElements(
            new ClickEvent("user1", "item-A", 1_000L),
            new ClickEvent("user1", "item-B", 2_000L),
            new ClickEvent("user1", "item-A", 3_000L),
            new ClickEvent("user1", "item-C", 4_000L),
            new ClickEvent("user1", "item-B", 5_000L)
        ).assignTimestampsAndWatermarks(
            WatermarkStrategy
                .<ClickEvent>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                .withTimestampAssigner((e, ts) -> e.timestamp)
        );

        // 구매 스트림 — user1 이 2번 구매
        DataStream<PurchaseEvent> purchases = env.fromElements(
            new PurchaseEvent("user1", "item-A", 3_500L),
            new PurchaseEvent("user1", "item-B", 6_000L)
        ).assignTimestampsAndWatermarks(
            WatermarkStrategy
                .<PurchaseEvent>forBoundedOutOfOrderness(Duration.ofSeconds(5))
                .withTimestampAssigner((e, ts) -> e.timestamp)
        );

        // connect 로 두 스트림을 연결한 후 CoProcessFunction 적용
        // keyBy 를 먼저 적용해야 상태를 유저별로 분리할 수 있다
        clicks
            .keyBy(e -> e.userId)
            .connect(purchases.keyBy(e -> e.userId))
            .process(new ConversionRateFunction())
            .print();

        env.execute("Connect Stream Conversion Rate");
    }
}
```

---

## 마무리

이 문서에서 다룬 중급 개념 요약:

| 주제 | 핵심 내용 |
|---|---|
| Window 연산 | Tumbling/Sliding/Session/Global 윈도우 용도와 차이점 |
| 시간 개념 | Event Time 사용 이유, WatermarkStrategy, 늦은 이벤트 처리 |
| 상태 관리 | ValueState/ListState/MapState 선택 기준, StateTTL로 메모리 관리 |
| KeyedProcessFunction | 타이머 등록/취소로 복잡한 비즈니스 로직 구현 |
| 체크포인트 | Exactly-once 설정, Kafka 트랜잭션 연동 |
| 스트림 합치기 | union(동일 타입), connect+CoProcessFunction(다른 타입) |

다음 단계로는 다음 주제를 학습하는 것을 권장한다.

- Flink SQL과 Table API를 이용한 선언적 스트림 처리
- RocksDB 상태 백엔드와 대용량 상태 최적화
- Flink CEP(Complex Event Processing)로 패턴 감지
- Async I/O로 외부 시스템 조회 성능 개선
