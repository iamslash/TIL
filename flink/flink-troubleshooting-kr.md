# Flink 주요 장애 상황과 대처 방안

- [개요](#개요)
- [1. 체크포인트 타임아웃](#1-체크포인트-타임아웃)
  - [증상](#증상)
  - [원인 분석](#원인-분석)
  - [대처 방법](#대처-방법)
  - [예방 설정](#예방-설정)
- [2. Back Pressure (배압)](#2-back-pressure-배압)
  - [증상](#증상-1)
  - [원인 분석](#원인-분석-1)
  - [진단 방법](#진단-방법)
  - [대처 방법](#대처-방법-1)
- [3. Out of Memory (OOM)](#3-out-of-memory-oom)
  - [증상](#증상-2)
  - [OOM 유형별 원인과 대처](#oom-유형별-원인과-대처)
  - [메모리 튜닝 체크리스트](#메모리-튜닝-체크리스트)
- [4. Kafka Consumer Lag 급증](#4-kafka-consumer-lag-급증)
  - [증상](#증상-3)
  - [원인 분석](#원인-분석-2)
  - [단계별 대처](#단계별-대처)
- [5. 데이터 쏠림 (Data Skew)](#5-데이터-쏠림-data-skew)
  - [증상](#증상-4)
  - [원인 분석](#원인-분석-3)
  - [대처 방법](#대처-방법-2)
- [6. 세이브포인트 복원 실패](#6-세이브포인트-복원-실패)
  - [증상](#증상-5)
  - [원인 분석](#원인-분석-4)
  - [올바른 UID 설정 방법](#올바른-uid-설정-방법)
  - [병렬도 변경 시 주의사항](#병렬도-변경-시-주의사항)
- [7. TaskManager 연결 끊김](#7-taskmanager-연결-끊김)
  - [증상](#증상-6)
  - [원인 분석](#원인-분석-5)
  - [대처 방법](#대처-방법-3)
- [8. Watermark 가 진행되지 않음](#8-watermark-가-진행되지-않음)
  - [증상](#증상-7)
  - [원인 분석](#원인-분석-6)
  - [대처 방법](#대처-방법-4)
- [장애 대응 플로우차트](#장애-대응-플로우차트)
- [운영 필수 모니터링 항목](#운영-필수-모니터링-항목)

---

# 개요

이 문서는 Flink 를 프로덕션에서 운영할 때 자주 발생하는 장애 상황과 대처 방법을 정리한다. 각 상황에 대해 증상, 원인, 진단 방법, 코드 수준의 해결책을 포함한다.

> 대상 독자: Flink 를 처음 운영하는 Junior Software Engineer

---

# 1. 체크포인트 타임아웃

## 증상

- Flink Web UI 의 Checkpoints 탭에서 체크포인트가 반복적으로 실패한다
- 로그에 `Checkpoint was declined` 또는 `Checkpoint expired before completing` 메시지가 나타난다
- 체크포인트 소요 시간이 설정한 타임아웃(기본 10분)을 초과한다

## 원인 분석

체크포인트가 완료되려면 **모든 Task 가 Checkpoint Barrier 를 받고, 상태 스냅샷을 저장소에 쓰고, ACK 를 보내야** 한다. 이 과정 중 어느 단계에서든 병목이 생기면 타임아웃이 발생한다.

흔한 원인:

**1) Back Pressure 로 Barrier 전파 지연**

Barrier 는 데이터 레코드와 함께 흐르므로, 하류 연산자에 Back Pressure 가 걸리면 Barrier 도 막힌다. 이것이 가장 흔한 원인이다.

```
Source → Map → [Back Pressure 여기서 발생] → Window → Sink
                Barrier 가 Window 까지 전달되지 못함
                → 체크포인트 타임아웃
```

**2) 상태 크기가 너무 큼**

윈도우에 대량의 데이터가 쌓이거나, `MapState` 에 수백만 개의 키가 들어가면 스냅샷 자체에 시간이 오래 걸린다.

**3) 저장소 I/O 병목**

체크포인트를 S3 나 HDFS 에 저장하는데, 네트워크가 느리거나 저장소 응답이 지연되는 경우.

## 대처 방법

**증분 체크포인팅 활성화** (RocksDB 사용 시)

```java
// 전체 상태가 아닌 변경분만 저장하므로 체크포인트 크기가 대폭 줄어든다
env.setStateBackend(new EmbeddedRocksDBStateBackend(true)); // true = incremental
```

**Unaligned Checkpoint 활성화** (Flink 1.11+)

일반 체크포인트는 Barrier Alignment 을 기다리지만, Unaligned Checkpoint 는 기다리지 않는다. Back Pressure 가 심한 환경에서 효과적이다.

```java
env.getCheckpointConfig().enableUnalignedCheckpoints();
```

단, Unaligned Checkpoint 는 체크포인트 크기가 커질 수 있다. 진행 중이던 버퍼 데이터도 함께 저장하기 때문이다.

**체크포인트 간격 조정**

```java
// 너무 짧으면 이전 체크포인트가 끝나기도 전에 다음 것이 시작된다
env.enableCheckpointing(60_000); // 60초 간격

// 체크포인트 간 최소 간격 설정 (이전 체크포인트 완료 후 최소 30초 대기)
env.getCheckpointConfig().setMinPauseBetweenCheckpoints(30_000);

// 타임아웃 설정
env.getCheckpointConfig().setCheckpointTimeout(300_000); // 5분
```

## 예방 설정

프로덕션에서 권장하는 체크포인트 설정:

```java
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

// 기본 설정
env.enableCheckpointing(60_000); // 60초 간격
env.getCheckpointConfig().setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE);
env.getCheckpointConfig().setCheckpointTimeout(300_000); // 5분 타임아웃
env.getCheckpointConfig().setMinPauseBetweenCheckpoints(30_000); // 최소 30초 간격
env.getCheckpointConfig().setMaxConcurrentCheckpoints(1); // 동시 체크포인트 1개

// RocksDB 증분 체크포인팅
env.setStateBackend(new EmbeddedRocksDBStateBackend(true));

// 체크포인트 저장소
env.getCheckpointConfig().setCheckpointStorage("s3://my-bucket/flink/checkpoints");

// 실패해도 Job 을 중단하지 않음
env.getCheckpointConfig().setTolerableCheckpointFailureNumber(3);
```

---

# 2. Back Pressure (배압)

## 증상

- Flink Web UI 에서 특정 연산자의 Back Pressure 지표가 **HIGH** (빨간색)로 표시된다
- 처리 지연(latency)이 점점 늘어난다
- Kafka Consumer Lag 이 함께 증가한다

## 원인 분석

Back Pressure 는 하류 연산자가 상류 연산자보다 느릴 때 발생한다. 파이프라인에서 가장 느린 연산자가 전체 처리 속도를 결정한다.

```
Source (1000 msg/s) → Map (1000 msg/s) → DB Sink (200 msg/s)
                                          ↑ 병목!
                                          Back Pressure 가 상류로 전파
```

흔한 원인:

| 원인 | 설명 |
|------|------|
| **느린 외부 I/O** | DB 쓰기, HTTP 호출, Redis 조회가 느려짐 |
| **무거운 연산** | 복잡한 정규식, 대량 JSON 파싱, 머신러닝 추론 |
| **데이터 쏠림** | 특정 키에 이벤트가 집중되어 일부 Slot 만 과부하 |
| **부족한 병렬도** | 처리해야 할 데이터량 대비 Slot 수가 부족 |

## 진단 방법

**1) Flink Web UI 확인**

Web UI → Job → Task → Back Pressure 탭에서 각 연산자의 Back Pressure 상태를 확인한다. **빨간색 연산자의 바로 다음(하류) 연산자**가 진짜 병목이다.

```
Source [OK] → Map [HIGH] → Sink [HIGH]
                            ↑ 진짜 병목은 Sink
                Map 이 HIGH 인 이유는 Sink 가 느려서 Map 의 출력 버퍼가 찬 것
```

**2) Metrics 확인**

```
# 각 연산자의 처리율 확인
flink_taskmanager_job_task_numRecordsInPerSecond
flink_taskmanager_job_task_numRecordsOutPerSecond

# 버퍼 사용률 확인 (1.0 에 가까우면 Back Pressure)
flink_taskmanager_job_task_buffers_outPoolUsage
```

## 대처 방법

**1) 비동기 I/O 사용 (외부 I/O 병목 시)**

동기 I/O 는 응답을 기다리는 동안 스레드가 놀지만, 비동기 I/O 는 여러 요청을 동시에 보낸다.

```java
// Before: 동기 방식 — DB 응답 올 때까지 스레드가 대기
public class SyncDbLookup extends RichMapFunction<Event, EnrichedEvent> {
    @Override
    public EnrichedEvent map(Event event) throws Exception {
        // 이 호출이 50ms 걸리면, 처리량은 1스레드당 20 msg/s 로 제한된다
        UserInfo user = dbClient.getUser(event.getUserId());
        return new EnrichedEvent(event, user);
    }
}

// After: 비동기 방식 — 동시에 여러 요청을 보내고 결과가 오면 처리
public class AsyncDbLookup extends RichAsyncFunction<Event, EnrichedEvent> {
    private transient AsyncDatabaseClient asyncClient;

    @Override
    public void open(Configuration parameters) {
        asyncClient = new AsyncDatabaseClient(/* config */);
    }

    @Override
    public void asyncInvoke(Event event, ResultFuture<EnrichedEvent> resultFuture) {
        CompletableFuture<UserInfo> future = asyncClient.getUserAsync(event.getUserId());
        future.thenAccept(user -> {
            resultFuture.complete(Collections.singleton(new EnrichedEvent(event, user)));
        });
    }

    @Override
    public void timeout(Event event, ResultFuture<EnrichedEvent> resultFuture) {
        resultFuture.complete(Collections.singleton(new EnrichedEvent(event, null)));
    }
}

// 사용: 동시 요청 100개, 타임아웃 5초
DataStream<EnrichedEvent> result = AsyncDataStream.unorderedWait(
    eventStream,
    new AsyncDbLookup(),
    5, TimeUnit.SECONDS,    // 타임아웃
    100                      // 동시 요청 수
);
```

**2) Sink 배치 쓰기**

레코드 하나씩 DB 에 쓰는 대신, 모아서 한꺼번에 쓰면 I/O 횟수가 줄어든다.

```java
public class BatchingSink extends RichSinkFunction<Record> {
    private final List<Record> buffer = new ArrayList<>();
    private static final int BATCH_SIZE = 500;

    @Override
    public void invoke(Record value, Context context) throws Exception {
        buffer.add(value);
        if (buffer.size() >= BATCH_SIZE) {
            flush();
        }
    }

    @Override
    public void close() throws Exception {
        if (!buffer.isEmpty()) {
            flush();
        }
    }

    private void flush() throws Exception {
        dbClient.batchInsert(buffer); // 500개를 한 번의 쿼리로 저장
        buffer.clear();
    }
}
```

**3) 병렬도 올리기**

병목 연산자의 병렬도만 선택적으로 올릴 수 있다.

```java
stream
    .keyBy(event -> event.getUserId())
    .map(new LightweightMap()).setParallelism(4)      // 가벼운 연산: 4
    .addSink(new HeavyDbSink()).setParallelism(16);   // 무거운 Sink: 16
```

---

# 3. Out of Memory (OOM)

## 증상

- TaskManager 프로세스가 갑자기 죽는다
- 로그에 `java.lang.OutOfMemoryError: Java heap space` 또는 `Direct buffer memory` 가 나타난다
- Kubernetes 환경에서 Pod 이 `OOMKilled` 상태로 재시작된다

## OOM 유형별 원인과 대처

### JVM Heap OOM

**원인**: 사용자 코드에서 JVM 힙에 큰 객체를 직접 올린 경우.

```java
// BAD: 상태를 일반 HashMap 에 저장 — 힙에 무한정 쌓인다
public class BadFunction extends RichMapFunction<Event, Result> {
    // 이 HashMap 은 Flink 가 관리하지 않으므로
    // 체크포인트에도 포함되지 않고, 메모리 제한도 없다
    private final Map<String, List<Event>> cache = new HashMap<>();

    @Override
    public Result map(Event event) {
        cache.computeIfAbsent(event.getUserId(), k -> new ArrayList<>()).add(event);
        // cache 가 계속 커지다가 OOM 발생
        return process(cache.get(event.getUserId()));
    }
}

// GOOD: Flink State API 를 사용 — Flink 가 메모리를 관리한다
public class GoodFunction extends KeyedProcessFunction<String, Event, Result> {
    // RocksDB 사용 시 디스크에 저장되므로 힙 부담 없음
    private ListState<Event> eventState;

    @Override
    public void open(Configuration parameters) {
        ListStateDescriptor<Event> descriptor =
            new ListStateDescriptor<>("events", Event.class);
        eventState = getRuntimeContext().getListState(descriptor);
    }

    @Override
    public void processElement(Event event, Context ctx, Collector<Result> out) throws Exception {
        eventState.add(event);
        List<Event> events = new ArrayList<>();
        eventState.get().forEach(events::add);
        out.collect(process(events));
    }
}
```

**핵심**: Flink 에서 상태를 저장할 때는 반드시 `ValueState`, `ListState`, `MapState` 등 Flink State API 를 사용해야 한다. 일반 Java 자료구조(`HashMap`, `ArrayList`)를 필드로 선언하면 Flink 가 관리할 수 없다.

### Direct Memory OOM

**원인**: 네트워크 버퍼가 부족하거나, 프레임워크 Off-heap 메모리가 부족한 경우.

```
# 로그 예시
java.lang.OutOfMemoryError: Direct buffer memory
```

**대처**:

```yaml
# flink-conf.yaml
# 네트워크 버퍼 메모리를 늘린다
taskmanager.memory.network.fraction: 0.15       # 기본 0.1
taskmanager.memory.network.min: 128mb
taskmanager.memory.network.max: 1gb
```

### Metaspace OOM

**원인**: 클래스 로딩이 반복되는 경우. 주로 Session Mode 에서 Job 을 반복 제출할 때 발생한다.

```yaml
# flink-conf.yaml
taskmanager.memory.jvm-metaspace.size: 512mb  # 기본 256mb
```

### Kubernetes OOMKilled (JVM 밖의 메모리 초과)

**원인**: JVM 이 사용하는 총 메모리가 Pod 의 메모리 limit 을 초과한 경우. Flink 의 JVM Overhead 설정이 부족하면 발생한다.

```yaml
# flink-conf.yaml
# JVM Overhead 비율을 늘린다 (기본 0.1)
taskmanager.memory.jvm-overhead.fraction: 0.15
taskmanager.memory.jvm-overhead.min: 256mb
taskmanager.memory.jvm-overhead.max: 1gb
```

## 메모리 튜닝 체크리스트

```yaml
# 프로덕션 권장 설정 예시 (TaskManager 총 메모리 4GB 기준)
taskmanager.memory.process.size: 4096mb

# Task Heap: 사용자 코드용 (RocksDB 사용 시 상대적으로 작게)
taskmanager.memory.task.heap.size: 1024mb

# Managed Memory: RocksDB 캐시용 (RocksDB 사용 시 넉넉하게)
taskmanager.memory.managed.fraction: 0.4

# Network: Task 간 데이터 교환용
taskmanager.memory.network.fraction: 0.1
taskmanager.memory.network.min: 128mb
taskmanager.memory.network.max: 512mb

# JVM Overhead: JVM 내부 + 네이티브 라이브러리
taskmanager.memory.jvm-overhead.fraction: 0.15
```

---

# 4. Kafka Consumer Lag 급증

## 증상

- Kafka 모니터링에서 Consumer Group 의 Lag 이 지속적으로 증가한다
- Flink Job 은 정상 동작하지만, 처리하는 이벤트의 타임스탬프가 현재 시각보다 점점 뒤처진다
- 실시간 알림이 지연되거나, 대시보드 집계가 과거 데이터를 보여준다

## 원인 분석

Lag = Kafka 에 쌓인 메시지 수 - Flink 가 읽은 메시지 수. Lag 이 늘어난다는 것은 **유입 속도 > 처리 속도**라는 뜻이다.

| 원인 | 진단 방법 |
|------|----------|
| **트래픽 급증** | Kafka 토픽의 메시지 유입율 확인 |
| **Flink 처리 느려짐** | Web UI 에서 Back Pressure 확인 |
| **체크포인트 실패 → 재시작 반복** | 로그에서 restart 횟수 확인 |
| **Kafka 파티션 대비 병렬도 부족** | 파티션 수 vs 병렬도 비교 |

## 단계별 대처

**1단계: 원인 파악**

```bash
# Kafka Consumer Lag 확인
kafka-consumer-groups.sh --bootstrap-server kafka:9092 \
  --group flink-my-job --describe

# 출력 예시:
# TOPIC      PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
# my-topic   0          1000000         1500000         500000  ← Lag 50만
# my-topic   1          1000000         1000100         100     ← 거의 없음
# my-topic   2          1000000         1500000         500000  ← Lag 50만
```

위 예시에서 파티션 0, 2 에만 Lag 이 쌓여 있다면 데이터 쏠림이 원인이다.

**2단계: 빠른 대응 — 병렬도 올리기**

```bash
# 1. 세이브포인트 생성
./bin/flink stop <JobID>

# 2. 병렬도를 올려서 재시작
./bin/flink run -s <savepoint-path> -p 12 my-job.jar
```

**3단계: Lag 해소 후 원래 병렬도로 복귀**

일시적 트래픽 급증이었다면, Lag 이 해소된 후 원래 병렬도로 되돌려 리소스를 절약한다.

**4단계: 근본 원인 해결**

```java
// 처리 로직이 무거운 경우 — 불필요한 연산 제거
stream
    // BAD: 모든 이벤트에 대해 무거운 JSON 파싱
    .map(raw -> objectMapper.readValue(raw, Event.class))
    // GOOD: 필요한 필드만 추출 (Jackson Streaming API 등)
    .map(raw -> extractFields(raw))

// keyBy 가 불필요하게 많은 경우
stream
    // BAD: keyBy 를 두 번 하면 네트워크 셔플이 두 번 발생
    .keyBy(e -> e.getUserId())
    .map(new SomeMap())
    .keyBy(e -> e.getUserId())  // 같은 키로 다시 keyBy → 불필요한 셔플
    .process(new SomeProcess())

    // GOOD: 한 번의 keyBy 에서 모두 처리
    .keyBy(e -> e.getUserId())
    .process(new CombinedProcess())  // map + process 를 합침
```

---

# 5. 데이터 쏠림 (Data Skew)

## 증상

- Flink Web UI 에서 동일 연산자의 Subtask 별 처리량이 크게 차이난다
  - Subtask 0: 10,000 records/s
  - Subtask 1: 100 records/s
  - Subtask 2: 50 records/s
- 일부 Slot 의 CPU 사용률이 100% 인데 나머지는 한가하다
- 전체 처리량이 기대보다 낮다

## 원인 분석

`keyBy()` 는 키의 해시값으로 데이터를 분배한다. 키 분포가 불균등하면 특정 Slot 에 데이터가 집중된다.

실제 사례:

```
데이팅 서비스에서 keyBy(userId) 로 스와이프 이벤트를 처리한다고 가정.
봇 계정 "bot-001" 이 전체 스와이프의 30% 를 생성한다면,
"bot-001" 을 담당하는 Slot 하나가 전체 부하의 30% 를 감당해야 한다.
나머지 Slot 들은 70% 를 나눠 가진다.
```

## 대처 방법

### 방법 1: 핫 키 사전 필터링

가장 간단한 방법. 비정상적으로 많은 이벤트를 생성하는 키를 미리 걸러낸다.

```java
// 봇이나 비정상 유저를 사이드 아웃풋으로 분리
public class HotKeyFilter extends ProcessFunction<Event, Event> {
    // 별도 처리할 핫 키 목록 (Broadcast State 로 동적 업데이트 가능)
    private static final Set<String> HOT_KEYS = Set.of("bot-001", "bot-002");

    private static final OutputTag<Event> HOT_KEY_TAG =
        new OutputTag<Event>("hot-key") {};

    @Override
    public void processElement(Event event, Context ctx, Collector<Event> out) {
        if (HOT_KEYS.contains(event.getUserId())) {
            ctx.output(HOT_KEY_TAG, event); // 별도 경로로 분리
        } else {
            out.collect(event); // 정상 경로
        }
    }
}

// 사용
SingleOutputStreamOperator<Event> mainStream = inputStream
    .process(new HotKeyFilter());

DataStream<Event> hotKeyStream = mainStream.getSideOutput(HOT_KEY_TAG);

// 정상 트래픽: keyBy 로 처리
mainStream.keyBy(e -> e.getUserId()).process(new NormalProcess());

// 핫 키: 별도 로직으로 처리 (더 높은 병렬도, 또는 단순 집계)
hotKeyStream.process(new HotKeyProcess());
```

### 방법 2: Key Salting (키에 소금 뿌리기)

키에 랜덤 접미사를 붙여 여러 Slot 에 분산시킨 뒤, 나중에 원래 키로 재집계한다.

```java
// 1단계: 키에 salt 를 붙여 분산
int saltBuckets = 10;

DataStream<Tuple3<String, Integer, Long>> saltedStream = inputStream
    .map(event -> {
        int salt = ThreadLocalRandom.current().nextInt(saltBuckets);
        String saltedKey = event.getUserId() + "#" + salt;
        return Tuple3.of(saltedKey, event.getCount(), 1L);
    })
    .keyBy(t -> t.f0)  // "user-001#3" 같은 salted key 로 분산
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .reduce((a, b) -> Tuple3.of(a.f0, a.f1 + b.f1, a.f2 + b.f2));
    // → 1차 집계: salt 별 부분합

// 2단계: salt 를 제거하고 원래 키로 재집계
saltedStream
    .map(t -> {
        String originalKey = t.f0.substring(0, t.f0.lastIndexOf("#"));
        return Tuple3.of(originalKey, t.f1, t.f2);
    })
    .keyBy(t -> t.f0)  // 원래 "user-001" 키로
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .reduce((a, b) -> Tuple3.of(a.f0, a.f1 + b.f1, a.f2 + b.f2));
    // → 2차 집계: 최종 결과
```

이 방법은 2단계 집계가 필요하므로 코드가 복잡해지지만, 핫 키를 미리 알 수 없을 때 유용하다.

### 방법 3: rebalance() 활용 (keyBy 전 단계)

`keyBy()` 앞에서 데이터가 이미 불균등하다면 `rebalance()` 로 균등 분배한 뒤 가벼운 전처리를 수행한다. 단, `keyBy()` 자체의 쏠림은 해결하지 못한다.

```java
stream
    .rebalance()                          // 라운드 로빈으로 균등 분배
    .map(new HeavyPreprocessing())        // 무거운 전처리를 균등하게 분산
    .keyBy(event -> event.getUserId())    // 이후 keyBy (쏠림은 여전히 가능)
    .process(new LightweightProcess());   // keyBy 이후는 가벼운 처리만
```

---

# 6. 세이브포인트 복원 실패

## 증상

- 세이브포인트에서 Job 을 재시작하면 오류가 발생한다
- 로그에 `StateMigrationException` 또는 `The operator state ... is not part of the savepoint` 메시지가 나타난다
- 일부 연산자의 상태가 유실된다

## 원인 분석

Flink 는 세이브포인트의 상태를 연산자의 **UID** 를 기준으로 매핑한다. UID 를 명시하지 않으면 Flink 가 **연산자 그래프의 위치**를 기반으로 자동 생성한다. 코드를 조금이라도 변경하면 (연산자 추가/삭제/순서 변경) 자동 생성 UID 가 달라져서 매핑이 실패한다.

```
세이브포인트 생성 시:                  복원 시 (코드 변경 후):
Source (uid: auto-123)               Source (uid: auto-123)
  → Map (uid: auto-456)               → Filter (uid: auto-456)  ← 새로 추가된 연산자
  → Sink (uid: auto-789)              → Map (uid: auto-789)     ← UID 가 밀림!
                                       → Sink (uid: auto-012)

Map 의 상태가 Filter 에 매핑되고, Sink 의 상태가 Map 에 매핑된다 → 실패
```

**이것은 Flink 를 처음 운영하는 팀에서 거의 반드시 한 번은 겪는 실수이다.**

## 올바른 UID 설정 방법

**모든 상태를 가진 연산자에 `.uid()` 를 명시적으로 지정한다.**

```java
// BAD: UID 가 없어서 코드 변경 시 복원 실패
stream
    .keyBy(e -> e.getUserId())
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .sum("count")
    .addSink(new MySink());

// GOOD: 모든 연산자에 UID 명시
stream
    .keyBy(e -> e.getUserId())
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .sum("count")
    .uid("user-count-5min-window")     // ← 고유하고 의미 있는 이름
    .name("User Count 5min Window")     // ← Web UI 에 표시되는 이름
    .addSink(new MySink())
    .uid("user-count-db-sink")
    .name("User Count DB Sink");
```

UID 작성 규칙:

| 규칙 | 설명 |
|------|------|
| **고유해야 한다** | 같은 Job 안에서 중복 불가 |
| **코드 변경 시 유지** | 연산자의 로직이 바뀌어도 UID 는 유지해야 상태 복원 가능 |
| **의미 있는 이름** | `uid-1` 보다 `user-count-5min-window` 가 낫다 |
| **삭제하면 안 된다** | 한 번 배포한 후에는 UID 를 변경하거나 삭제하면 안 된다 |

## 병렬도 변경 시 주의사항

세이브포인트에서 복원하면서 병렬도를 변경할 수 있다. 하지만 상태 유형에 따라 제약이 있다.

| 상태 유형 | 병렬도 변경 가능? | 설명 |
|----------|-----------------|------|
| **Keyed State** | O | 키 기반으로 자동 재분배 |
| **Operator State (List)** | O | 리스트를 균등 분배 |
| **Operator State (Union)** | O | 모든 인스턴스에 전체 복사 |
| **Operator State (Broadcast)** | O | 모든 인스턴스에 동일 복사 |

```bash
# 병렬도 6 → 12 로 변경하며 복원
./bin/flink run -s /savepoints/savepoint-abc123 -p 12 my-job.jar
# Keyed State 는 키 해시에 따라 새 Slot 에 자동 재분배된다
```

---

# 7. TaskManager 연결 끊김

## 증상

- 로그에 `TaskManager is no longer reachable` 메시지가 나타난다
- JobManager 가 TaskManager 의 하트비트를 받지 못해 해당 TaskManager 의 Task 를 실패 처리한다
- Kubernetes 환경에서 Pod Eviction 이 발생한다

## 원인 분석

| 원인 | 설명 |
|------|------|
| **OOM 으로 프로세스 사망** | 위의 OOM 섹션 참고 |
| **Full GC 로 일시 정지** | 오래 걸리는 GC 동안 하트비트를 보내지 못함 |
| **네트워크 일시 단절** | 클라우드 환경에서 간헐적 네트워크 이슈 |
| **Kubernetes Pod Eviction** | 노드의 리소스 부족으로 Pod 이 퇴거됨 |

## 대처 방법

**하트비트 타임아웃 조정**

네트워크가 불안정하거나 GC pause 가 긴 환경에서는 타임아웃을 늘린다.

```yaml
# flink-conf.yaml
# 하트비트 간격 (기본 10초)
heartbeat.interval: 10000

# 하트비트 타임아웃 (기본 50초) — 이 시간 동안 하트비트가 없으면 연결 끊김 처리
heartbeat.timeout: 180000  # 3분으로 늘림
```

**GC 튜닝**

```yaml
# flink-conf.yaml
# G1GC 사용 (큰 힙에서 pause 시간 최소화)
env.java.opts.taskmanager: >-
  -XX:+UseG1GC
  -XX:MaxGCPauseMillis=100
  -XX:+PrintGCDetails
  -XX:+PrintGCDateStamps
  -Xloggc:/opt/flink/log/gc.log
```

**재시작 전략 설정**

일시적 장애에 대비하여 자동 재시작 전략을 설정한다.

```java
// 고정 지연 재시작: 최대 3회, 10초 간격
env.setRestartStrategy(RestartStrategies.fixedDelayRestart(3, Time.seconds(10)));

// 실패율 기반 재시작: 5분 내 3회까지 허용, 10초 간격
env.setRestartStrategy(RestartStrategies.failureRateRestart(
    3,                       // 최대 실패 횟수
    Time.minutes(5),         // 측정 구간
    Time.seconds(10)         // 재시작 간 대기 시간
));
```

---

# 8. Watermark 가 진행되지 않음

## 증상

- 윈도우 연산의 결과가 나오지 않는다 (윈도우가 트리거되지 않음)
- Flink Web UI 의 Watermark 지표가 특정 값에서 멈춰 있다
- 일부 파티션의 Watermark 만 진행되고 나머지는 멈춰 있다

## 원인 분석

Flink 의 Watermark 는 **모든 입력 채널 중 가장 느린 것**을 기준으로 진행된다. 하나의 Kafka 파티션이라도 이벤트가 오지 않으면 전체 Watermark 가 멈춘다.

```
Partition 0: Watermark = 10:05:00  (이벤트 계속 들어옴)
Partition 1: Watermark = 10:05:00  (이벤트 계속 들어옴)
Partition 2: Watermark = 10:01:00  (이벤트가 안 들어옴!)  ← 전체 Watermark 가 여기에 멈춤

전체 Watermark = min(10:05:00, 10:05:00, 10:01:00) = 10:01:00
→ 10:01:00 이후의 윈도우가 트리거되지 않음
```

흔한 원인:
- 트래픽이 적어서 일부 Kafka 파티션에 이벤트가 없음
- Kafka 파티션 수 > Flink 병렬도 → 일부 Source Task 가 여러 파티션을 담당하는데 그 중 하나가 비어 있음
- Source 연산자가 이벤트 없이도 Watermark 를 전진시키는 설정이 빠져 있음

## 대처 방법

**Idle Source 감지 설정**

이벤트가 없는 파티션을 idle 로 표시하여 Watermark 계산에서 제외한다.

```java
// Flink 1.11+: WatermarkStrategy 에서 idle 타임아웃 설정
WatermarkStrategy<Event> strategy = WatermarkStrategy
    .<Event>forBoundedOutOfOrderness(Duration.ofSeconds(5))
    .withTimestampAssigner((event, timestamp) -> event.getTimestamp())
    .withIdleness(Duration.ofMinutes(1));  // ← 1분간 이벤트가 없으면 idle 처리

DataStream<Event> stream = env
    .fromSource(kafkaSource, strategy, "Kafka Source");
```

`withIdleness(Duration.ofMinutes(1))` 가 핵심이다. 이 설정이 없으면 이벤트가 없는 파티션이 전체 Watermark 를 영원히 막는다.

**Kafka 파티션 수 = Flink 병렬도 맞추기**

하나의 Source Task 가 여러 파티션을 담당하면 관리가 복잡해진다. 가능하면 1:1 로 맞춘다.

---

# 장애 대응 플로우차트

```
Job 이 비정상이다
│
├─ Job 이 죽었는가?
│  ├─ YES → 로그 확인
│  │  ├─ OOM → [3. OOM] 참고
│  │  ├─ TaskManager 끊김 → [7. TaskManager 연결 끊김] 참고
│  │  └─ 복원 실패 → [6. 세이브포인트 복원 실패] 참고
│  │
│  └─ NO → 성능 문제
│     ├─ Lag 증가 → [4. Kafka Consumer Lag] 참고
│     ├─ Back Pressure → [2. Back Pressure] 참고
│     ├─ 체크포인트 실패 → [1. 체크포인트 타임아웃] 참고
│     ├─ 윈도우 안 나옴 → [8. Watermark] 참고
│     └─ 일부 Slot 만 느림 → [5. Data Skew] 참고
```

---

# 운영 필수 모니터링 항목

| 항목 | 메트릭 | 임계값 예시 |
|------|--------|------------|
| **체크포인트 소요 시간** | `lastCheckpointDuration` | > 60초 이면 경고 |
| **체크포인트 실패** | `numberOfFailedCheckpoints` | > 0 이면 확인 |
| **Back Pressure** | `backPressuredTimeMsPerSecond` | > 500ms 이면 경고 |
| **Kafka Lag** | `records-lag-max` (Consumer 메트릭) | > 10,000 이면 경고 |
| **TaskManager 메모리** | `Status.JVM.Memory.Heap.Used` | > 80% 이면 경고 |
| **처리 지연** | `currentInputWatermark` vs 현재 시각 | > 5분 차이면 경고 |
| **재시작 횟수** | `fullRestarts` | > 3회/시간 이면 조사 |

> Grafana + Prometheus 조합으로 위 메트릭을 대시보드로 구성하는 것을 권장한다. Flink 는 Prometheus Reporter 를 내장하고 있어 설정만 추가하면 메트릭을 노출할 수 있다.

```yaml
# flink-conf.yaml
metrics.reporter.prom.factory.class: org.apache.flink.metrics.prometheus.PrometheusReporterFactory
metrics.reporter.prom.port: 9249
```
