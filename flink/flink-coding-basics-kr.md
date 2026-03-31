# Flink 코딩 가이드 — 기초편

## 목차

- [프로젝트 설정](#프로젝트-설정)
- [ExecutionEnvironment 이해하기](#executionenvironment-이해하기)
- [데이터 소스 (Source)](#데이터-소스-source)
- [기본 변환 (Transformation)](#기본-변환-transformation)
- [싱크 (Sink)](#싱크-sink)
- [첫 번째 완전한 예제: Word Count](#첫-번째-완전한-예제-word-count)
- [첫 번째 실무 예제: 이벤트 필터링 파이프라인](#첫-번째-실무-예제-이벤트-필터링-파이프라인)
- [디버깅 팁](#디버깅-팁)

---

## 프로젝트 설정

Flink 프로젝트를 시작하려면 Maven 또는 Gradle 프로젝트를 생성하고 Flink 의존성을 추가해야 한다.

### Maven 의존성

아래는 Flink 1.18 기준 최소한의 `pom.xml` 예시다. Kafka 소스/싱크도 함께 포함했다.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
             http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.example</groupId>
    <artifactId>flink-tutorial</artifactId>
    <version>1.0-SNAPSHOT</version>
    <packaging>jar</packaging>

    <properties>
        <!-- Java 11 이상 권장 -->
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
        <!-- Flink 버전은 클러스터와 동일하게 맞춰야 한다 -->
        <flink.version>1.18.1</flink.version>
    </properties>

    <dependencies>
        <!-- Flink Java API — 스트리밍 작업의 핵심 라이브러리 -->
        <dependency>
            <groupId>org.apache.flink</groupId>
            <artifactId>flink-streaming-java</artifactId>
            <version>${flink.version}</version>
        </dependency>

        <!-- Flink 클라이언트 — 로컬 실행과 jar 제출에 필요 -->
        <dependency>
            <groupId>org.apache.flink</groupId>
            <artifactId>flink-clients</artifactId>
            <version>${flink.version}</version>
        </dependency>

        <!-- Kafka 커넥터 — Kafka 소스/싱크를 사용할 때 필요 -->
        <dependency>
            <groupId>org.apache.flink</groupId>
            <artifactId>flink-connector-kafka</artifactId>
            <version>${flink.version}</version>
        </dependency>

        <!-- JDBC 커넥터 — DB 싱크를 사용할 때 필요 -->
        <dependency>
            <groupId>org.apache.flink</groupId>
            <artifactId>flink-connector-jdbc</artifactId>
            <version>3.1.2-1.17</version>
        </dependency>

        <!-- Jackson — JSON 파싱에 사용 -->
        <dependency>
            <groupId>com.fasterxml.jackson.core</groupId>
            <artifactId>jackson-databind</artifactId>
            <version>2.15.2</version>
        </dependency>

        <!-- MySQL JDBC 드라이버 — JDBC 싱크에서 MySQL 사용 시 필요 -->
        <dependency>
            <groupId>com.mysql</groupId>
            <artifactId>mysql-connector-j</artifactId>
            <version>8.3.0</version>
        </dependency>

        <!-- SLF4J + Logback — 로컬 실행 시 로그를 보기 위해 필요 -->
        <dependency>
            <groupId>org.slf4j</groupId>
            <artifactId>slf4j-api</artifactId>
            <version>1.7.36</version>
        </dependency>
        <dependency>
            <groupId>ch.qos.logback</groupId>
            <artifactId>logback-classic</artifactId>
            <version>1.2.11</version>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <!-- fat jar 생성 — 클러스터에 제출할 때 모든 의존성을 포함한 단일 jar 를 만든다 -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-shade-plugin</artifactId>
                <version>3.5.1</version>
                <executions>
                    <execution>
                        <phase>package</phase>
                        <goals><goal>shade</goal></goals>
                        <configuration>
                            <transformers>
                                <transformer implementation=
                                    "org.apache.maven.plugins.shade.resource.ManifestResourceTransformer">
                                    <!-- main 클래스를 여기에 지정 -->
                                    <mainClass>com.example.WordCountJob</mainClass>
                                </transformer>
                            </transformers>
                        </configuration>
                    </execution>
                </executions>
            </plugin>
        </plugins>
    </build>
</project>
```

### Gradle 의존성

Gradle 을 선호한다면 `build.gradle` 에 아래와 같이 추가한다.

```groovy
plugins {
    id 'java'
    id 'com.github.johnrengelman.shadow' version '8.1.1' // fat jar 생성 플러그인
}

ext {
    flinkVersion = '1.18.1'
}

dependencies {
    implementation "org.apache.flink:flink-streaming-java:${flinkVersion}"
    implementation "org.apache.flink:flink-clients:${flinkVersion}"
    implementation "org.apache.flink:flink-connector-kafka:${flinkVersion}"
    implementation "org.apache.flink:flink-connector-jdbc:3.1.2-1.17"
    implementation 'com.fasterxml.jackson.core:jackson-databind:2.15.2'
    implementation 'com.mysql:mysql-connector-j:8.3.0'
    runtimeOnly 'ch.qos.logback:logback-classic:1.2.11'
}

// fat jar 의 메인 클래스 지정
shadowJar {
    manifest {
        attributes 'Main-Class': 'com.example.WordCountJob'
    }
}
```

---

## ExecutionEnvironment 이해하기

Flink 잡은 반드시 `StreamExecutionEnvironment` 에서 시작한다. 이 객체는 Flink 잡의 "설계도"를 관리하며, `env.execute()` 를 호출하는 순간 그 설계도를 클러스터(또는 로컬 JVM)에 제출한다.

### StreamExecutionEnvironment 생성

```java
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public class EnvironmentExample {
    public static void main(String[] args) throws Exception {

        // 실행 환경을 가져온다.
        // - 로컬에서 실행하면 로컬 미니 클러스터가 자동으로 뜬다.
        // - 클러스터(YARN, Kubernetes 등)에서 실행하면 클러스터 환경에 연결된다.
        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();

        // 이 아래에 소스 → 변환 → 싱크 파이프라인을 정의한다.
        // ...

        // 파이프라인 정의가 끝나면 execute() 로 잡을 제출한다.
        // - 이 호출은 블로킹(blocking)이다. 잡이 끝날 때까지 메인 스레드가 기다린다.
        // - 잡 이름은 Flink Web UI 에서 구분하는 데 사용된다.
        env.execute("EnvironmentExample");
    }
}
```

### 병렬도(Parallelism) 설정

```java
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public class ParallelismExample {
    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();

        // 잡 전체의 기본 병렬도를 설정한다.
        // 예: 4 로 설정하면 각 오퍼레이터(map, filter 등)가 4개의 병렬 태스크로 실행된다.
        // 클러스터의 Task Slot 수보다 크게 설정하면 안 된다.
        env.setParallelism(4);

        // 특정 오퍼레이터만 병렬도를 다르게 설정할 수도 있다.
        // (뒤쪽 섹션에서 예제와 함께 설명한다)

        env.execute("ParallelismExample");
    }
}
```

### 체크포인트(Checkpoint) 설정

```java
import org.apache.flink.streaming.api.CheckpointingMode;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.time.Duration;

public class CheckpointExample {
    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();

        // 체크포인트를 10초마다 수행한다.
        // 체크포인트란 현재 처리 상태(오프셋, 상태 데이터 등)를 영구 저장소에 저장하는 것이다.
        // 잡이 실패하면 마지막 체크포인트 시점부터 재처리한다.
        env.enableCheckpointing(10_000); // 단위: 밀리초

        // EXACTLY_ONCE: 각 이벤트가 정확히 한 번만 처리되도록 보장한다 (기본값).
        // AT_LEAST_ONCE: 최소 한 번 처리를 보장한다 (성능이 더 좋지만 중복 가능).
        env.getCheckpointConfig()
           .setCheckpointingConsistencyMode(CheckpointingMode.EXACTLY_ONCE);

        // 체크포인트 타임아웃: 60초 안에 완료되지 않으면 중단한다.
        env.getCheckpointConfig()
           .setCheckpointTimeout(60_000);

        // 두 체크포인트 사이에 최소 5초 간격을 둔다.
        // 체크포인트가 너무 자주 발생해 처리 성능을 떨어뜨리는 것을 방지한다.
        env.getCheckpointConfig()
           .setMinPauseBetweenCheckpoints(5_000);

        env.execute("CheckpointExample");
    }
}
```

### env.execute() 의 의미

`env.execute()` 를 호출하기 전까지는 어떤 데이터도 처리되지 않는다. Flink 는 "지연 실행(lazy evaluation)" 방식이기 때문이다.

- `fromElements(...)`, `map(...)`, `filter(...)` 등은 단지 실행 계획(DAG)을 만들 뿐이다.
- `env.execute()` 를 호출하는 순간 비로소 클러스터에 잡이 제출되고 데이터가 흐르기 시작한다.
- `execute()` 는 `JobExecutionResult` 를 반환한다. 잡이 끝나면 처리한 레코드 수 등의 통계를 담고 있다.

---

## 데이터 소스 (Source)

소스는 Flink 파이프라인에서 데이터가 들어오는 입구다.

### fromElements — 테스트용 인라인 데이터

```java
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public class FromElementsExample {
    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();

        // 코드 안에 데이터를 직접 넣는 방식이다.
        // 유닛 테스트나 로직 확인 시 편리하다.
        // 실무 잡에는 사용하지 않는다.
        DataStream<String> stream = env.fromElements(
            "hello world",
            "flink is fast",
            "hello flink"
        );

        // 각 요소를 그대로 콘솔에 출력한다.
        stream.print();

        env.execute("FromElementsExample");
    }
}
```

### fromCollection — 컬렉션 소스

```java
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.util.Arrays;
import java.util.List;

public class FromCollectionExample {
    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();

        // Java List 를 소스로 사용한다.
        // 테스트 데이터를 동적으로 만들 때 유용하다.
        List<String> data = Arrays.asList(
            "click,user_001,2024-01-01T10:00:00",
            "purchase,user_002,2024-01-01T10:01:00",
            "click,user_001,2024-01-01T10:02:00"
        );

        DataStream<String> stream = env.fromCollection(data);

        stream.print();

        env.execute("FromCollectionExample");
    }
}
```

### Socket 소스 — 간단한 실습용

```java
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public class SocketSourceExample {
    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();

        // 터미널에서 `nc -lk 9999` 를 실행한 뒤 이 잡을 시작한다.
        // nc 에 타이핑하는 내용이 Flink 로 들어온다.
        // 호스트명, 포트, 구분자(여기서는 줄바꿈)를 지정한다.
        DataStream<String> stream = env.socketTextStream("localhost", 9999);

        stream.print();

        env.execute("SocketSourceExample");
        // 실행 후 터미널에서 "hello flink" 를 입력하면 콘솔에 출력된다.
    }
}
```

### Kafka 소스 — 실무에서 가장 많이 사용

Kafka 는 실무 Flink 잡의 99%가 사용하는 소스다. 아래는 완전한 예제다.

```java
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public class KafkaSourceExample {
    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();

        // KafkaSource 를 빌더 패턴으로 생성한다.
        KafkaSource<String> kafkaSource = KafkaSource.<String>builder()
            // Kafka 브로커 주소 (여러 개면 콤마로 구분)
            .setBootstrapServers("kafka-broker1:9092,kafka-broker2:9092")

            // 읽을 토픽 이름
            .setTopics("user-events")

            // 컨슈머 그룹 ID — 오프셋 커밋을 추적하는 데 사용된다.
            // 같은 그룹 ID 로 재시작하면 이전에 읽던 위치부터 이어서 읽는다.
            .setGroupId("flink-tutorial-consumer")

            // 오프셋 시작 위치 설정
            // - earliest(): 토픽의 가장 처음부터 읽는다 (재처리 시 사용)
            // - latest(): 지금부터 새로 들어오는 메시지만 읽는다
            // - committedOffsets(): 마지막으로 커밋된 오프셋부터 읽는다 (기본 장애 복구)
            .setStartingOffsets(OffsetsInitializer.earliest())

            // 메시지를 String 으로 역직렬화한다.
            // JSON 메시지라면 이후 map 에서 파싱한다.
            .setValueOnlyDeserializer(new SimpleStringSchema())

            .build();

        // WatermarkStrategy.noWatermarks() 는 이벤트 시간을 사용하지 않을 때 지정한다.
        // 이벤트 시간 기반 윈도우를 쓰려면 다른 전략을 사용해야 한다.
        DataStream<String> stream = env.fromSource(
            kafkaSource,
            WatermarkStrategy.noWatermarks(),
            "Kafka Source"  // 이 이름이 Flink Web UI 에 표시된다
        );

        stream.print();

        env.execute("KafkaSourceExample");
    }
}
```

#### Kafka 소스 — 커스텀 역직렬화

메시지가 JSON 이고 특정 POJO 로 바로 변환하고 싶다면 커스텀 역직렬화를 사용한다.

```java
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.serialization.DeserializationSchema;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

// 이벤트를 담는 간단한 POJO
class UserEvent {
    public String eventType; // 이벤트 종류 (예: "click", "purchase")
    public String userId;    // 사용자 ID
    public long   timestamp; // 이벤트 발생 시각 (Unix epoch ms)

    // Jackson 이 JSON 을 역직렬화할 때 기본 생성자가 필요하다.
    public UserEvent() {}

    @Override
    public String toString() {
        return "UserEvent{eventType='" + eventType
            + "', userId='" + userId
            + "', timestamp=" + timestamp + "}";
    }
}

// JSON 바이트 배열을 UserEvent 객체로 변환하는 역직렬화 클래스
class UserEventDeserializer implements DeserializationSchema<UserEvent> {

    // ObjectMapper 는 스레드 안전(thread-safe)하다. static final 로 재사용한다.
    private static final ObjectMapper mapper = new ObjectMapper();

    @Override
    public UserEvent deserialize(byte[] message) throws Exception {
        // Kafka 메시지 바이트를 JSON 으로 파싱해 UserEvent 객체로 변환한다.
        return mapper.readValue(message, UserEvent.class);
    }

    @Override
    public boolean isEndOfStream(UserEvent nextElement) {
        // 스트리밍 소스는 끝이 없으므로 항상 false 를 반환한다.
        return false;
    }

    @Override
    public TypeInformation<UserEvent> getProducedType() {
        // Flink 가 타입 정보를 추론할 수 있도록 명시적으로 반환한다.
        return TypeInformation.of(UserEvent.class);
    }
}

public class KafkaSourceWithDeserializerExample {
    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();

        KafkaSource<UserEvent> kafkaSource = KafkaSource.<UserEvent>builder()
            .setBootstrapServers("localhost:9092")
            .setTopics("user-events")
            .setGroupId("flink-deserializer-example")
            .setStartingOffsets(OffsetsInitializer.latest())
            // 커스텀 역직렬화 클래스를 사용한다.
            .setValueOnlyDeserializer(new UserEventDeserializer())
            .build();

        DataStream<UserEvent> stream = env.fromSource(
            kafkaSource,
            WatermarkStrategy.noWatermarks(),
            "Kafka UserEvent Source"
        );

        // UserEvent 의 toString() 이 호출되어 콘솔에 출력된다.
        stream.print();

        env.execute("KafkaSourceWithDeserializerExample");
    }
}
```

---

## 기본 변환 (Transformation)

소스에서 읽어온 데이터를 가공하는 단계다. Flink 는 다양한 변환 오퍼레이터를 제공한다.

### map — 1:1 변환

`map` 은 입력 하나를 받아 출력 하나를 만든다. 이벤트에서 특정 필드를 추출하거나 형식을 바꿀 때 사용한다.

```java
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

// CSV 로 인코딩된 이벤트에서 userId 필드만 추출하는 예제
// 입력 형식: "click,user_001,2024-01-01T10:00:00"
public class MapExample {
    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<String> rawStream = env.fromElements(
            "click,user_001,2024-01-01T10:00:00",
            "purchase,user_002,2024-01-01T10:01:00",
            "click,user_003,2024-01-01T10:02:00"
        );

        // MapFunction<입력타입, 출력타입>
        // 각 CSV 줄에서 두 번째 필드(userId)만 꺼낸다.
        DataStream<String> userIds = rawStream.map(
            new MapFunction<String, String>() {
                @Override
                public String map(String line) throws Exception {
                    // "click,user_001,2024-01-01T10:00:00" 를 콤마로 분리
                    String[] parts = line.split(",");
                    // 인덱스 1 이 userId
                    return parts[1];
                }
            }
        );

        // 람다로 더 간결하게 작성할 수도 있다.
        // DataStream<String> userIds = rawStream.map(line -> line.split(",")[1]);

        userIds.print();
        // 출력 예시:
        // user_001
        // user_002
        // user_003

        env.execute("MapExample");
    }
}
```

### flatMap — 1:N 변환

`flatMap` 은 입력 하나를 받아 0개 이상의 출력을 만든다. 문장을 단어로 분리하거나, 하나의 이벤트에서 여러 파생 이벤트를 만들 때 사용한다.

```java
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.Collector;

// 문장을 단어 단위로 분리하는 예제
public class FlatMapExample {
    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<String> sentences = env.fromElements(
            "hello world",
            "flink is fast and scalable",
            "hello flink"
        );

        // FlatMapFunction<입력타입, 출력타입>
        // Collector 로 출력 요소를 0개 이상 내보낼 수 있다.
        DataStream<String> words = sentences.flatMap(
            new FlatMapFunction<String, String>() {
                @Override
                public void flatMap(String sentence, Collector<String> out) {
                    // 문장을 공백으로 분리해 단어마다 하나씩 emit 한다.
                    for (String word : sentence.split("\\s+")) {
                        out.collect(word);
                    }
                }
            }
        );

        words.print();
        // 출력 예시 (순서는 병렬도에 따라 달라질 수 있다):
        // hello
        // world
        // flink
        // is
        // ...

        env.execute("FlatMapExample");
    }
}
```

### filter — 조건 필터링

`filter` 는 조건에 맞는 요소만 통과시킨다. `true` 를 반환하면 통과, `false` 이면 버린다.

```java
import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

// "purchase" 이벤트만 통과시키는 예제
public class FilterExample {
    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<String> events = env.fromElements(
            "click,user_001",
            "purchase,user_002",
            "click,user_003",
            "purchase,user_004",
            "view,user_005"
        );

        // FilterFunction<입력타입>
        // true 를 반환하는 요소만 다운스트림으로 전달된다.
        DataStream<String> purchases = events.filter(
            new FilterFunction<String>() {
                @Override
                public boolean filter(String event) {
                    // 첫 번째 필드가 "purchase" 인 이벤트만 통과
                    return event.startsWith("purchase");
                }
            }
        );

        // 람다로 더 간결하게: events.filter(e -> e.startsWith("purchase"))

        purchases.print();
        // 출력 예시:
        // purchase,user_002
        // purchase,user_004

        env.execute("FilterExample");
    }
}
```

### keyBy — 키 기반 그룹핑

`keyBy` 는 동일한 키를 가진 이벤트들이 반드시 같은 병렬 태스크(Slot)로 전달되도록 라우팅한다. 집계(sum, count 등)나 상태 저장 처리를 올바르게 하려면 반드시 `keyBy` 를 먼저 해야 한다.

**왜 keyBy 가 중요한가?**

Flink 는 여러 태스크가 병렬로 실행된다. `keyBy` 없이 집계하면 각 태스크가 자신이 받은 데이터만 집계하므로 결과가 부정확하다. `keyBy` 를 쓰면 같은 키(예: 같은 userId)는 항상 같은 태스크로 가기 때문에 정확한 집계가 가능하다.

```java
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.KeyedStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

// 이벤트 타입별로 그룹핑한 뒤 카운팅하는 예제
public class KeyByExample {
    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<String> events = env.fromElements(
            "click,user_001",
            "purchase,user_002",
            "click,user_003",
            "purchase,user_004",
            "click,user_005"
        );

        // 먼저 (이벤트타입, 카운트) 형태의 Tuple2 로 변환한다.
        // Tuple2<String, Long> 에서 f0 은 이벤트 타입, f1 은 1 (카운트 초기값)
        DataStream<Tuple2<String, Long>> typeWithCount = events.map(
            new MapFunction<String, Tuple2<String, Long>>() {
                @Override
                public Tuple2<String, Long> map(String event) {
                    String eventType = event.split(",")[0]; // "click" 또는 "purchase"
                    return Tuple2.of(eventType, 1L);        // (타입, 1)
                }
            }
        );

        // keyBy 로 f0 (이벤트 타입) 기준으로 그룹핑한다.
        // 같은 이벤트 타입은 항상 같은 태스크로 전달된다.
        KeyedStream<Tuple2<String, Long>, String> keyed =
            typeWithCount.keyBy(tuple -> tuple.f0);

        // sum(1) 은 f1 필드(카운트)를 키별로 누적 합산한다.
        DataStream<Tuple2<String, Long>> result = keyed.sum(1);

        result.print();
        // 출력 예시 (누적 카운트이므로 이벤트가 들어올 때마다 갱신된다):
        // (click,1)
        // (purchase,1)
        // (click,2)
        // (purchase,2)
        // (click,3)

        env.execute("KeyByExample");
    }
}
```

---

## 싱크 (Sink)

싱크는 처리한 데이터를 외부로 내보내는 출구다.

### print — 디버깅용 콘솔 출력

```java
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public class PrintSinkExample {
    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<String> stream = env.fromElements("a", "b", "c");

        // 각 요소를 표준 출력(stdout)에 출력한다.
        // 병렬로 실행 중이면 앞에 태스크 번호가 붙는다. 예: "2> b"
        stream.print();

        // 접두사를 붙이면 여러 스트림을 동시에 디버깅할 때 구분이 쉽다.
        stream.print("DEBUG");

        env.execute("PrintSinkExample");
    }
}
```

### Kafka 싱크

```java
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.connector.base.DeliveryGuarantee;
import org.apache.flink.connector.kafka.sink.KafkaRecordSerializationSchema;
import org.apache.flink.connector.kafka.sink.KafkaSink;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public class KafkaSinkExample {
    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();

        // 체크포인트가 활성화되어 있어야 EXACTLY_ONCE 가 동작한다.
        env.enableCheckpointing(10_000);

        DataStream<String> results = env.fromElements(
            "result_1",
            "result_2",
            "result_3"
        );

        KafkaSink<String> kafkaSink = KafkaSink.<String>builder()
            // 출력할 Kafka 브로커 주소
            .setBootstrapServers("localhost:9092")

            // 각 요소를 어떻게 직렬화해서 어느 토픽에 쓸지 정의한다.
            .setRecordSerializer(
                KafkaRecordSerializationSchema.builder()
                    .setTopic("output-topic")           // 출력 토픽
                    .setValueSerializationSchema(
                        new SimpleStringSchema()        // String 을 바이트로 직렬화
                    )
                    .build()
            )

            // 전달 보장 수준 설정
            // - AT_LEAST_ONCE: 최소 한 번 (중복 가능, 성능 우선)
            // - EXACTLY_ONCE: 정확히 한 번 (Kafka 트랜잭션 사용, 체크포인트 필요)
            // - NONE: 보장 없음 (테스트용)
            .setDeliveryGuarantee(DeliveryGuarantee.AT_LEAST_ONCE)

            .build();

        // sinkTo() 로 싱크를 연결한다.
        results.sinkTo(kafkaSink);

        env.execute("KafkaSinkExample");
    }
}
```

### JDBC 싱크 — 데이터베이스 저장

```java
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.connector.jdbc.JdbcConnectionOptions;
import org.apache.flink.connector.jdbc.JdbcExecutionOptions;
import org.apache.flink.connector.jdbc.JdbcSink;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

// (이벤트타입, 카운트) 를 MySQL 에 저장하는 예제
// 사전에 아래 테이블이 생성되어 있어야 한다:
// CREATE TABLE event_counts (event_type VARCHAR(50), count BIGINT);
public class JdbcSinkExample {
    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();

        DataStream<Tuple2<String, Long>> eventCounts = env.fromElements(
            Tuple2.of("click", 100L),
            Tuple2.of("purchase", 50L),
            Tuple2.of("view", 200L)
        );

        // JdbcSink.sink() 는 deprecated 됐지만 입문용으로는 여전히 간단하다.
        // 실무에서는 JdbcSink.exactlyOnceSink() 사용을 검토한다.
        eventCounts.addSink(
            JdbcSink.sink(
                // INSERT 쿼리 (또는 UPSERT)
                "INSERT INTO event_counts (event_type, count) VALUES (?, ?)",

                // PreparedStatement 에 값을 채우는 방법 정의
                (preparedStatement, tuple) -> {
                    preparedStatement.setString(1, tuple.f0); // 이벤트 타입
                    preparedStatement.setLong(2, tuple.f1);   // 카운트
                },

                // 실행 옵션: 몇 개 모아서 한 번에 배치로 쓸지 설정
                JdbcExecutionOptions.builder()
                    .withBatchSize(200)          // 200개마다 한 번에 DB 에 쓴다
                    .withBatchIntervalMs(2_000)  // 또는 2초마다 쓴다 (둘 중 먼저 도달하는 조건)
                    .withMaxRetries(3)           // 실패 시 최대 3번 재시도
                    .build(),

                // DB 연결 정보
                new JdbcConnectionOptions.JdbcConnectionOptionsBuilder()
                    .withUrl("jdbc:mysql://localhost:3306/mydb")
                    .withDriverName("com.mysql.cj.jdbc.Driver")
                    .withUsername("root")
                    .withPassword("password")
                    .build()
            )
        );

        env.execute("JdbcSinkExample");
    }
}
```

---

## 첫 번째 완전한 예제: Word Count

Kafka 에서 문장을 읽어 단어별로 카운팅하고 콘솔에 출력하는 전통적인 Word Count 예제다.

```java
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.util.Collector;

/**
 * Kafka Word Count 잡
 *
 * 파이프라인:
 *   Kafka("sentences") → flatMap(단어 분리) → keyBy(단어) → sum(카운트) → print
 *
 * 실행 전 준비:
 *   1. Kafka 에 "sentences" 토픽 생성
 *   2. 해당 토픽에 문장을 보내는 프로듀서 실행
 *      예: kafka-console-producer.sh --topic sentences --bootstrap-server localhost:9092
 */
public class WordCountJob {

    public static void main(String[] args) throws Exception {

        // 1. 실행 환경 설정
        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();

        // 병렬도 2: 두 개의 태스크 슬롯을 사용한다.
        // 로컬 실행 시에는 논리적 병렬 스레드가 2개 생긴다.
        env.setParallelism(2);

        // 10초마다 체크포인트를 수행한다.
        // 잡이 재시작되면 마지막 체크포인트의 Kafka 오프셋부터 이어서 읽는다.
        env.enableCheckpointing(10_000);

        // 2. Kafka 소스 정의
        KafkaSource<String> source = KafkaSource.<String>builder()
            .setBootstrapServers("localhost:9092")
            .setTopics("sentences")           // 문장이 들어오는 토픽
            .setGroupId("word-count-job")
            .setStartingOffsets(OffsetsInitializer.earliest())
            .setValueOnlyDeserializer(new SimpleStringSchema())
            .build();

        DataStream<String> sentences = env.fromSource(
            source,
            WatermarkStrategy.noWatermarks(),
            "Kafka Sentences Source"
        );

        // 3. 변환 파이프라인 정의

        DataStream<Tuple2<String, Long>> wordCounts = sentences

            // 3-1. 각 문장을 단어로 분리한다. (1:N 변환)
            // 입력: "hello flink"
            // 출력: ("hello", 1), ("flink", 1)
            .flatMap(new FlatMapFunction<String, Tuple2<String, Long>>() {
                @Override
                public void flatMap(String sentence, Collector<Tuple2<String, Long>> out) {
                    // 빈 문자열과 공백만 있는 줄은 무시한다.
                    if (sentence == null || sentence.trim().isEmpty()) {
                        return;
                    }
                    for (String word : sentence.toLowerCase().split("\\s+")) {
                        // 각 단어를 (단어, 1) 형태로 내보낸다.
                        out.collect(Tuple2.of(word, 1L));
                    }
                }
            })

            // 3-2. 같은 단어가 같은 태스크로 가도록 단어(f0)를 키로 그룹핑한다.
            .keyBy(tuple -> tuple.f0)

            // 3-3. 키별로 f1(카운트) 필드를 누적 합산한다.
            .sum(1);

        // 4. 결과를 콘솔에 출력한다.
        wordCounts.print("WORD_COUNT");
        // 출력 예시:
        // WORD_COUNT:1> (hello,1)
        // WORD_COUNT:2> (flink,1)
        // WORD_COUNT:1> (hello,2)    <- "hello" 가 두 번째로 등장했을 때 카운트 갱신

        // 5. 잡 제출
        env.execute("Kafka Word Count");
    }
}
```

---

## 첫 번째 실무 예제: 이벤트 필터링 파이프라인

Kafka 에서 JSON 이벤트를 읽어 특정 조건으로 필터링한 뒤 다른 Kafka 토픽으로 출력하는 파이프라인이다. 실무에서 가장 자주 만나는 가장 단순한 형태의 Flink 잡이다.

```java
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.serialization.DeserializationSchema;
import org.apache.flink.api.common.serialization.SerializationSchema;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.connector.base.DeliveryGuarantee;
import org.apache.flink.connector.kafka.sink.KafkaRecordSerializationSchema;
import org.apache.flink.connector.kafka.sink.KafkaSink;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

/**
 * 이벤트 필터링 파이프라인
 *
 * 파이프라인:
 *   Kafka("raw-events") → JSON 파싱 → purchase 이벤트만 필터링
 *                       → JSON 직렬화 → Kafka("purchase-events")
 *
 * 입력 JSON 형식 (raw-events 토픽):
 *   {"eventType":"click","userId":"user_001","amount":0}
 *   {"eventType":"purchase","userId":"user_002","amount":9900}
 *
 * 출력 JSON 형식 (purchase-events 토픽):
 *   {"eventType":"purchase","userId":"user_002","amount":9900}
 */
public class EventFilterPipelineJob {

    // ---- 도메인 모델 ----

    // 입력 이벤트를 담는 POJO
    static class RawEvent {
        public String eventType; // 이벤트 종류
        public String userId;    // 사용자 ID
        public long   amount;    // 금액 (구매 이벤트에서만 유효)

        public RawEvent() {} // Jackson 역직렬화용 기본 생성자
    }

    // ---- 역직렬화: JSON 바이트 → RawEvent ----

    static class RawEventDeserializer implements DeserializationSchema<RawEvent> {
        private static final ObjectMapper mapper = new ObjectMapper();

        @Override
        public RawEvent deserialize(byte[] message) throws Exception {
            return mapper.readValue(message, RawEvent.class);
        }

        @Override
        public boolean isEndOfStream(RawEvent nextElement) {
            return false; // 스트리밍은 끝이 없다.
        }

        @Override
        public TypeInformation<RawEvent> getProducedType() {
            return TypeInformation.of(RawEvent.class);
        }
    }

    // ---- 직렬화: RawEvent → JSON 바이트 ----

    static class RawEventSerializer implements SerializationSchema<RawEvent> {
        private static final ObjectMapper mapper = new ObjectMapper();

        @Override
        public byte[] serialize(RawEvent event) {
            try {
                return mapper.writeValueAsBytes(event);
            } catch (Exception e) {
                // 직렬화 실패 시 null 을 반환하면 해당 레코드를 건너뛴다.
                // 실무에서는 별도 에러 토픽에 기록하거나 메트릭을 남기는 것이 좋다.
                throw new RuntimeException("직렬화 실패: " + e.getMessage(), e);
            }
        }
    }

    // ---- 메인 잡 ----

    public static void main(String[] args) throws Exception {

        // 1. 실행 환경 설정
        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.getExecutionEnvironment();

        env.setParallelism(4);
        env.enableCheckpointing(15_000); // 15초마다 체크포인트

        // 2. Kafka 소스: raw-events 토픽에서 JSON 이벤트를 읽는다.
        KafkaSource<RawEvent> source = KafkaSource.<RawEvent>builder()
            .setBootstrapServers("localhost:9092")
            .setTopics("raw-events")
            .setGroupId("event-filter-pipeline")
            .setStartingOffsets(OffsetsInitializer.latest())
            .setValueOnlyDeserializer(new RawEventDeserializer())
            .build();

        DataStream<RawEvent> rawEvents = env.fromSource(
            source,
            WatermarkStrategy.noWatermarks(),
            "raw-events Kafka Source"
        );

        // 3. 변환: purchase 이벤트만 필터링한다.
        DataStream<RawEvent> purchaseEvents = rawEvents.filter(
            new FilterFunction<RawEvent>() {
                @Override
                public boolean filter(RawEvent event) {
                    // eventType 이 null 이면 NPE 방지를 위해 false 를 반환한다.
                    return "purchase".equals(event.eventType);
                }
            }
        );

        // (선택) 로컬 디버깅 시 필터 결과를 콘솔에서 확인하고 싶다면 주석 해제
        // purchaseEvents.print("PURCHASE");

        // 4. Kafka 싱크: purchase-events 토픽으로 필터링된 이벤트를 출력한다.
        KafkaSink<RawEvent> sink = KafkaSink.<RawEvent>builder()
            .setBootstrapServers("localhost:9092")
            .setRecordSerializer(
                KafkaRecordSerializationSchema.builder()
                    .setTopic("purchase-events")
                    .setValueSerializationSchema(new RawEventSerializer())
                    .build()
            )
            .setDeliveryGuarantee(DeliveryGuarantee.AT_LEAST_ONCE)
            .build();

        purchaseEvents.sinkTo(sink);

        // 5. 잡 제출
        env.execute("Event Filter Pipeline");
    }
}
```

---

## 디버깅 팁

### 로컬에서 실행하는 방법

Flink 잡은 IDE(IntelliJ IDEA 등)에서 바로 `main()` 을 실행할 수 있다. `getExecutionEnvironment()` 는 로컬에서 호출되면 자동으로 로컬 미니 클러스터를 시작한다. 별도로 Flink 클러스터를 설치할 필요가 없다.

```java
// 이 코드는 IDE 에서 main() 으로 직접 실행해도 동작한다.
StreamExecutionEnvironment env =
    StreamExecutionEnvironment.getExecutionEnvironment();

// 로컬 실행 시 병렬도를 1로 설정하면 출력 순서를 예측할 수 있어 디버깅이 쉽다.
env.setParallelism(1);
```

Kafka 소스를 포함한 잡을 로컬에서 테스트할 때는 `localhost:9092` 에 Kafka 가 떠 있어야 한다. Docker Compose 로 간단히 구성할 수 있다.

```yaml
# docker-compose.yml (Kafka 로컬 테스트 환경)
version: "3"
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    ports:
      - "9092:9092"
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
```

### print() 와 로깅

```java
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.flink.api.common.functions.MapFunction;

// map 함수 안에서 로거를 사용하는 방법
class LoggingMapFunction implements MapFunction<String, String> {

    // static final Logger 는 직렬화 문제를 피하는 표준 방식이다.
    private static final Logger LOG =
        LoggerFactory.getLogger(LoggingMapFunction.class);

    @Override
    public String map(String value) throws Exception {
        // 처리량이 많을 때 모든 이벤트를 로깅하면 성능이 크게 떨어진다.
        // 샘플링하거나 오류 상황에서만 로깅하는 것이 좋다.
        LOG.info("처리 중인 값: {}", value);
        return value.toUpperCase();
    }
}
```

`print()` 는 간단한 디버깅에는 충분하지만, 처리량이 높은 실무 환경에서는 반드시 제거해야 한다. 출력 자체가 병목이 될 수 있다.

### Flink Web UI 활용

로컬에서 잡을 실행하면 기본적으로 `http://localhost:8081` 에 Web UI 가 열린다.

Web UI 에서 확인할 수 있는 것들:

- **Job Graph**: 파이프라인 DAG 시각화. 각 오퍼레이터가 몇 개의 태스크로 나뉘었는지 볼 수 있다.
- **처리 지연 (Latency)**: 각 오퍼레이터의 처리 속도.
- **레코드 처리 수**: 각 오퍼레이터가 지금까지 처리한 레코드 수.
- **체크포인트 상태**: 마지막 체크포인트가 성공했는지, 얼마나 걸렸는지.
- **예외(Exception)**: 태스크가 실패한 경우 스택 트레이스를 확인할 수 있다.

Web UI 에서 백압(backpressure) 표시가 빨간색으로 나타나면 해당 오퍼레이터가 다운스트림보다 데이터를 빠르게 생성하고 있다는 뜻이다. 병렬도를 높이거나 로직을 최적화해야 한다.
