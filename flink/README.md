- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
  - [Install](#install)
  - [Flink Architecture](#flink-architecture)
  - [Memory Configurations](#memory-configurations)
    - [Job Manager](#job-manager)
  - [Task Manager](#task-manager)
  - [Deployment Modes](#deployment-modes)
  - [Deploy Flink App In Application Mode](#deploy-flink-app-in-application-mode)
  - [Logging](#logging)
  - [Watermars](#watermars)
  - [Message Propagation Methods](#message-propagation-methods)
  - [File System](#file-system)
  - [AWS S3](#aws-s3)
  - [Check Point](#check-point)
    - [Saving Check Point Process](#saving-check-point-process)
    - [Loading Check Point Process](#loading-check-point-process)
  - [Save Point](#save-point)
  - [Check Point vs Save Point](#check-point-vs-save-point)
  - [State Backend](#state-backend)
  - [RocksDB](#rocksdb)
  - [Sharing Data Among Task Managers](#sharing-data-among-task-managers)
  - [Graceful Shutdown](#graceful-shutdown)

----

# Abstract

Apache Flink is a framework and distributed processing engine for stateful computations over unbounded and bounded data streams

# Materials

* [글로벌 기업이 더 주목하는 스트림 프로세싱 프레임워크- (플링크Flink) 이해하기 | samsungsds](https://www.samsungsds.com/kr/insights/flink.html)
* [스트림 프로세싱의 긴 여정을 위한 이정표 (w. Apache Flink) | medium](https://medium.com/rate-labs/%EC%8A%A4%ED%8A%B8%EB%A6%BC-%ED%94%84%EB%A1%9C%EC%84%B8%EC%8B%B1%EC%9D%98-%EA%B8%B4-%EC%97%AC%EC%A0%95%EC%9D%84-%EC%9C%84%ED%95%9C-%EC%9D%B4%EC%A0%95%ED%91%9C-with-flink-8e3953f97986)
  * Stream framework 의 jardon 을 포함하여 설명
* [Apache Flink Training Excercises | github](https://github.com/apache/flink-training)
* [Stream Processing with Apache Flink: Fundamentals, Implementation, and Operation of Streaming Applications | amazon](https://www.amazon.com/Stream-Processing-Apache-Flink-Implementation-ebook/dp/B07QM3DSB7)
  * [src-java](https://github.com/streaming-with-flink/examples-java)
  * [src-scala](https://github.com/streaming-with-flink/examples-scala)
  * [src-kotlin](https://github.com/rockmkd/flink-examples-kotlin)
* [Demystifying Flink Memory Allocation and tuning - Roshan Naik | youtube](https://www.youtube.com/watch?v=aq1Whga-RJ4)
  * [Flink Memory Tuning Calculator | googledoc](https://docs.google.com/spreadsheets/d/1DMUnHXNdoK1BR9TpTTpqeZvbNqvXGO7PlNmTojtaStU/edit#gid=0)

# Basic

## Install

Download and decompress files

* [flink downloads](https://flink.apache.org/downloads.html)

## Flink Architecture

- [Flink Architecture | flink](https://nightlies.apache.org/flink/flink-docs-master/docs/concepts/flink-architecture/)

![](img/2023-11-04-18-17-04.png)

The Flink runtime has two main types of processes: **JobManager** and **TaskManager**.

**JobManager** is responsible for coordinating the distributed execution of
Flink applications. Its main tasks include scheduling tasks, reacting to
finished tasks or execution failures, coordinating checkpoints, and coordinating
recovery on failures. The JobManager contains three components:
**ResourceManager** (responsible for resource allocation and provisioning),
**Dispatcher** (providing a REST interface to submit Flink applications and
starting a new JobMaster for each submitted job), and **JobMaster** (responsible
for managing the execution of a single JobGraph).

**TaskManager**, also known as workers, is responsible for executing the tasks
of a dataflow and buffering and exchanging data streams. There must be at least
one TaskManager, and it consists of task slots, which are the smallest unit of
scheduling for a TaskManager. Task slots indicate the number of concurrent
processing tasks that can be executed, and multiple operators may be executed
within a single task slot.

In Flink, tasks are formed by **chaining** together operator subtasks for
distributed execution. This chaining process is an optimization technique aimed
at reducing the overhead caused by thread-to-thread handover and buffering,
resulting in increased overall throughput and decreased latency. 
**Each task is executed by a single thread.**

one **task** is executed by one thread. Each **task** comprises one or more
**chained operator subtasks**, and these **subtasks** are executed within the
same thread to improve efficiency and reduce overheads associated with
thread-to-thread handover and buffering.

**Chaining operators** into tasks makes the execution process more efficient, but it
can be configured based on specific needs, as described in the 
[chaining documentation](https://nightlies.apache.org/flink/flink-docs-release-1.18/docs/dev/datastream/operators/overview/#task-chaining-and-resource-groups).

![](img/2023-11-04-18-21-59.png)

a **task slot** in Flink can hold **multiple tasks**, which means it can
effectively manage **multiple threads**. Each TaskManager has a defined number
of task slots, and each slot represents a fixed subset of the TaskManager's
resources. By allocating different tasks to the same task slot, these tasks (and
their corresponding threads) can share resources, such as managed memory, TCP
connections (via multiplexing), and heartbeat messages. As a result, a single
task slot can support the execution of multiple tasks with multiple threads.

![](img/2023-11-04-18-34-07.png)

Including many tasks in one task slot offers several advantages:

- **Resource Utilization**: Combining multiple tasks within a single task slot
  allows for better resource utilization. Shared resources, such as 
  **managed memory**, **TCP connections** (via multiplexing), and 
  **heartbeat messages**, are used more efficiently, leading to overall 
  increased productivity and reduced overhead.
- **Optimization**: By chaining multiple tasks together into a single task slot,
  Flink reduces the overhead associated with thread-to-thread handovers and
  buffering, which in turn increases overall throughput and decreases latency.
- **Simplified Cluster Management**: Allowing slot sharing means that a Flink
  cluster needs exactly as many task slots as the highest parallelism used in
  the job. This simplifies cluster management, eliminating the need to calculate
  the total number of tasks (with varying parallelism) in a program.
- **Flexibility**: By adjusting the number of task slots and the distribution of
  tasks within them, users can control the degree of isolation between subtasks,
  maximizing efficiency based on specific workload requirements.
- **Scalability**: Including many tasks in a single task slot enables greater
  parallelism, making it easier to handle complex pipelines without compromising
  resource distribution or creating bottlenecks in computation.

In summary, including multiple tasks in a single task slot enables more
efficient resource utilization, improved performance, easier management of Flink
clusters, flexibility in workload distribution, and better scalability.

![](img/2023-11-04-18-34-21.png)

## Memory Configurations

- [Set up Flink’s Process Memory | flink](https://nightlies.apache.org/flink/flink-docs-release-1.18/docs/deployment/memory/mem_setup/)

### Job Manager

![](img/2023-11-04-19-18-19.png)

**Off-heap memory** refers to the memory that is allocated outside the JVM heap
space and is managed directly by Flink using native memory management libraries.
It is used for various purposes such as managing buffers for network
communication, managing memory for state backends, and other internal data
structures. Off-heap memory is not subject to Java's garbage collection, which
means it can provide more predictable latency and performance behavior compared
to on-heap memory. In the context of Apache Flink, the total Flink memory
consumption includes usage of both JVM heap and off-heap memory.

**Metaspace** is a non-heap memory pool introduced in Java 8 for class metadata
storage. It replaced the "Permanent Generation" (PermGen) space that was used in
earlier Java versions. **Metaspace** is allocated outside the Java heap and can
resize dynamically. It stores metadata like class definitions, method data, and
other constant pool information. When the metaspace runs out of memory, the JVM
triggers a Full GC, or garbage collection, to free up space by removing unused
classes and classloader data.

**JVM Overhead** is the additional memory that the JVM needs to operate but is
not directly attributable to Flink's application data or Flink components. This
overhead includes the JVM code cache, the memory used by the Just-In-Time (JIT)
compiler, and other miscellaneous JVM overheads. It is important to allocate
enough JVM overhead memory to ensure the smooth operation of the JVM and Flink
application. By default, the JVM overhead is a fraction of the total memory of
the Flink process, with a minimum and maximum value.

## Task Manager

![](img/2023-11-04-19-20-01.png)

**Direct memory** refers to the memory in the Java Virtual Machine (JVM) that is
allocated outside the Java heap, also known as off-heap memory. In Flink's
memory model, direct memory plays a significant role in managing components such
as network memory, framework off-heap memory, and task off-heap memory.

The advantage of using **direct memory** is that it can be accessed without
going through the Java garbage collector, which can lead to improved performance
as data in direct memory isn't subject to garbage collection pauses. However,
developers should be cautious when using direct memory since it could lead to
memory leaks or OutOfMemoryError situations if not managed properly. In Flink,
the direct memory limit of the JVM is typically configured using the JVM
parameters provided during the setup.

![](img/2023-11-04-19-20-23.png)

The **framework heap memory** is a component of Flink's memory model that is
dedicated to the Flink framework's internal data structures and operations. It
is a part of the JVM Heap memory and is configured using the option
`taskmanager.memory.framework.heap.size`. Generally, you should not change the
**framework heap memory** without a good reason, as it should only be adjusted
if you are sure that Flink needs more memory for certain internal processes or
in specific deployment scenarios with high parallelism.

**Task Heap Memory** is a part of the JVM Heap memory dedicated to running
Flink's application, including operators and user code. It is configured using
the option `taskmanager.memory.task.heap.size`.

**Managed Memory** is native memory (off-heap) managed by Flink and used for
workloads such as RocksDB state backend in streaming jobs, sorting, hash tables,
and caching of intermediate results in both streaming and batch jobs. It is
configured using the options `taskmanager.memory.managed.size` and
`taskmanager.memory.managed.fraction`.

**Framework Off-heap Memory** is off-heap direct (or native) memory dedicated to
the Flink framework for internal data structures or operations. It is an
advanced option and configured using the option
`taskmanager.memory.framework.off-heap.size`.

**Task Off-heap Memory** is off-heap direct (or native) memory dedicated to
running Flink's application, including operators. It is configured using the
option `taskmanager.memory.task.off-heap.size`.

**Network Memory** is direct memory reserved for data record exchange between
tasks, such as buffering data for transfer over the network. It is a capped
fractionated component of the total Flink memory and is used for allocating
network buffers. It is configured using the options
`taskmanager.memory.network.min`, `taskmanager.memory.network.max`, and
`taskmanager.memory.network.fraction`.

## Deployment Modes

Flink execute applications in one of three ways.

- in Application Mode,
- in Session Mode,
- in a Per-Job Mode (deprecated wince flink 1.15).

![](img/2023-11-04-13-58-23.png)

## Deploy Flink App In Application Mode

- [Application Mode Deployment | flink](https://nightlies.apache.org/flink/flink-docs-master/docs/deployment/resource-providers/standalone/overview/#application-mode)
----

- Download flink binary from [download page](https://flink.apache.org/downloads/)

```bash
# Copy application jar
$ cp ./examples/streaming/TopSpeedWindowing.jar lib/

# Run Job Manager
$ ./bin/standalone-job.sh start --job-classname org.apache.flink.streaming.examples.windowing.TopSpeedWindowing

# Run Task Manager
$ ./bin/taskmanager.sh start
# Open browser localhost:8081

# Stop all
$ ./bin/taskmanager.sh stop
$ ./bin/standalone-job.sh stop
```

## Logging

- [Deployment - Advanced - Logging | flink](https://nightlies.apache.org/flink/flink-docs-master/docs/deployment/advanced/logging/)

```bash
$ cd ~/flink-1.18.0/log/
```

`F1` shows command palette in Flink Web UI Logs.

![](img/2023-11-04-23-09-47.png)

## Watermars

- [이벤트 시간 처리(Event Time Processing)와 워터마크(Watermark) | tistory](https://seamless.tistory.com/99)

**Watermarks** in Flink are used for determining event time progression and handling
out-of-order events in a data stream. They are essentially a timestamp that
moves monotonically in the event time dimension, allowing Flink to determine
when it has received all events up to a certain point in time. This helps Flink
decide when to trigger time-based window computations or when to discard late
events.

```java
import org.apache.flink.api.common.eventtime.*;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.TimeCharacteristic;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.timestamps.BoundedOutOfOrdernessTimestampExtractor;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;

public class FlinkWatermarkExample {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // Set the time characteristic to event time
        env.setStreamTimeCharacteristic(TimeCharacteristic.EventTime);

        // Create a simple data stream of Tuple2<Long, Integer> to emulate (timestamp, value)
        DataStream<Tuple2<Long, Integer>> inputStream = env.fromElements(
                Tuple2.of(1000L, 1),
                Tuple2.of(1500L, 2),
                Tuple2.of(2900L, 3),
                Tuple2.of(3500L, 4),
                Tuple2.of(7500L, 5)
        );

        // Assign timestamps and generate watermarks
        DataStream<Tuple2<Long, Integer>> timestampedStream = inputStream.assignTimestampsAndWatermarks(
                new BoundedOutOfOrdernessTimestampExtractor<Tuple2<Long, Integer>>(Time.milliseconds(500)) {
                    @Override
                    public long extractTimestamp(Tuple2<Long, Integer> element) {
                        return element.f0;
                    }
                });

        // Define a tumbling event time window of length 2 seconds
        DataStream<Tuple2<Long, Integer>> windowedStream = timestampedStream
                .keyBy(e -> 1)
                .window(TumblingEventTimeWindows.of(Time.seconds(2)))
                .sum(1);

        // Print the windowed stream results
        windowedStream.print();

        // Start execution of the data processing
        env.execute("Flink Watermark Example");
    }
}
```

In this example, we have a simple data stream of `Tuple2 (timestamp, value)` to
emulate a stream with timestamps. We set the time characteristic to event time
using `env.setStreamTimeCharacteristic()`.

We then assign timestamps and generate watermarks using a
`BoundedOutOfOrdernessTimestampExtractor`. This extractor allows events to be
out of order up to a certain bound (in this case, 500 milliseconds). The
`extractTimestamp()` method returns the timestamp component of our input
`Tuple2`.

Next, we define a tumbling event time window of length 2 seconds using
`TumblingEventTimeWindows.of(Time.seconds(2))`. The window will sum the values of
input events received within each 2-second window.

Finally, we print the results of the windowed stream and start the execution
using `env.execute()`. This example demonstrates how to use watermarks in Flink
to handle out-of-order events in an event-time based streaming scenario using
Java APIs.

## Message Propagation Methods

- [Data Exchange inside Apache Flink](https://flink.apache.org/2020/03/24/advanced-flink-application-patterns-vol.2-dynamic-updates-of-application-logic/#data-exchange-inside-apache-flink)

Inside Apache Flink, message propagation methods include `FORWARD`, `HASH`, `REBALANCE`, `BROADCAST`.

```java
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.state.BroadcastState;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.common.typeinfo.BasicTypeInfo;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.BroadcastStream;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.co.BroadcastProcessFunction;

public class FlinkDataExchangePatternsExample {

    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // Create a simple data stream of Integer values
        DataStream<Integer> integerStream = env.fromElements(1, 2, 3, 4, 5, 6);

        // FORWARD data exchange
        DataStream<Integer> forwardStream = integerStream.map(new MapFunction<Integer, Integer>() {
            @Override
            public Integer map(Integer value) throws Exception {
                return value * 2;
            }
        });

        // HASH data exchange
        DataStream<Tuple2<Integer, Integer>> keyedStream = integerStream
                .map(i -> Tuple2.of(i, i % 2)).keyBy(1).name("hash-data-exchange");

        // REBALANCE data exchange
        DataStream<Integer> rebalanceStream = integerStream.rebalance().map(new MapFunction<Integer, Integer>() {
            @Override
            public Integer map(Integer value) throws Exception {
                return value * 3;
            }
        }).name("rebalance-data-exchange");

        // Create a simple data stream of String values for broadcasting
        DataStream<String> broadcastStringStream = env.fromElements("A", "B", "C");

        // Create a descriptor for the broadcast state
        MapStateDescriptor<Void, String> descriptor = new MapStateDescriptor<>(
                "BroadcastState",
                TypeInformation.of(Void.class),
                BasicTypeInfo.STRING_TYPE_INFO);

        // Create a broadcast stream with the string values and descriptor
        BroadcastStream<String> broadcastStream = broadcastStringStream.broadcast(descriptor);

        // BROADCAST data exchange
        DataStream<String> broadcastResultStream = integerStream.connect(broadcastStream)
                .process(new BroadcastProcessFunction<Integer, String, String>() {

                    // Main data processing logic for unicast elements (integer stream)
                    @Override
                    public void processElement(Integer value, ReadOnlyContext ctx, Collector<String> out) throws Exception {
                        // Just emit the original integer as a string
                        out.collect(String.valueOf(value));
                    }

                    // Broadcast data processing logic for broadcast elements (string stream)
                    @Override
                    public void processBroadcastElement(String value, Context ctx, Collector<String> out) throws Exception {
                        // Retrieve the broadcast state handle
                        BroadcastState<Void, String> broadcastState = ctx.getBroadcastState(descriptor);

                        // Update the broadcast state with the new string value
                        broadcastState.put(null, value);

                        // Emit the updated broadcast value
                        out.collect("Broadcasted: " + value);
                    }
                }).name("broadcast-data-exchange");

        forwardStream.print("Forward:");
        keyedStream.print("Keyed (Hash):");
        rebalanceStream.print("Rebalance:");
        broadcastResultStream.print("Broadcast:");

        // Start execution of the data processing
        env.execute("Flink Data Exchange Patterns Example");
    }
}
```

In this example, we first create a simple data stream of integer values and
apply different data exchange patterns:

- **FORWARD**: The `map()` function doubles the input value, and the output is
  directly sent to the next operator.
- **HASH**: We create a `keyBy()` function that assigns a key based on the integer
  value's remainder when divided by 2. This distributes the data based on the
  hash of the key.
- **REBALANCE**: We call the `rebalance()` function before the `map()` function,
  which ensures a round-robin data distribution to the next operator that
  triples the input value.
- **BROADCAST**: We create an additional `broadcastStringStream`, create a
  `MapStateDescriptor` for the broadcast state, and create a `BroadcastStream`
  using the `broadcast()` function. We connect the `integerStream` with the
  `broadcastStream` and process their elements using a
  `BroadcastProcessFunction`.

Finally, we start the execution by calling `env.execute()`. This code demonstrates
how different data exchange patterns are achieved in Apache Flink using Java
APIs.

## File System

Flink requires a file system to handle various aspects of its distributed data
processing tasks. The file system plays a crucial role in managing input data,
output data, and intermediate data generated during the processing of a Flink
job. It also serves as a storage backend for checkpoints and savepoints, which
are critical for ensuring fault-tolerance and state consistency in Flink
applications.

Here are some reasons why Flink needs a file system:

- **Data Ingestion and Output**: Flink reads data from various sources,
  processes it and writes the resulting data to one or more sinks. The file
  system can be used as a source or sink, allowing Flink to read or write data
  from formats like text files, Parquet, Avro, or ORC. Integrations with various
  file systems enable Flink to work seamlessly with diverse data storage
  options.
- **Storage Backend for Checkpoints and Savepoints**: Flink uses checkpoints and
  savepoints to save the state of a job during its execution, enabling
  fault-tolerance and resumability. A distributed file system serves as a
  reliable storage backend for Flink's checkpoints and savepoints, ensuring that
  they are durable, globally accessible, and can be safely restored in the event
  of failures.
- **Intermediate Data Storage**: While processing and transforming data within
  Flink pipelines, intermediate data may need to be persisted to disk for
  operations like sorting, joining, and windowing that involve large volumes of
  data. A file system helps Flink manage and persist intermediate data during
  these stages.
- **File-based Connectors**: Flink provides connectors for various data storage
  systems, which allows Flink to interface with both distributed and local file
  systems such as HDFS, S3, GCS, and local disk, enabling seamless integration
  with a broad range of data storage solutions for different use cases.
- **Data Exchange Between Operators**: Flink needs to exchange data between
  parallel tasks running on different nodes. Although Flink primarily uses
  communication via the network for data exchange, it also employs the file
  system to exchange data (using files) between tasks if network resources are
  insufficient for certain operations like sorting or shuffling.

## AWS S3

Flink supports file system including AWS S3.

For a Flink application that uploads media files to S3, using the **AWS S3 SDK**
would be more appropriate. The primary reason is that 
**Flink's AWS S3 connector** is designed for Flink's specific use cases, 
such as reading and writing large datasets for distributed processing or 
storing Flink checkpoints, rather than handling generic file operations 
like uploading media files.

## Check Point

A checkpointis a mechanism to ensure **fault-tolerance** and **consistency** in
a streaming application. It periodically captures the state of a running Flink
job and saves it to a distributed, persistent storage system (such as HDFS or
S3). In case of failures, the system can resume the job from the latest
successful checkpoint, ensuring that the application's state is consistent and
no data is lost. Checkpoints help Flink to recover the state of an application
and maintain **exactly-once processing** semantics, providing reliable and
accurate results even in case of failures.

There are several reasons why Flink needs checkpoints:

- **Fault Tolerance**: Flink operates in distributed environments, which are
  inherently prone to various types of failures, such as node failures, network
  issues, or software errors. Checkpoints help preserve the state of an
  application during such failures, enabling it to recover and continue
  processing from where it left off, ensuring no data loss.
- **State Consistency**: Streaming applications often maintain stateful
  information (e.g., counts, windowed aggregates, or machine-learning models) to
  perform various computations. Checkpoints ensure the consistency of this state
  across the application, even in case of failures or scaling events. When a
  recovery is necessary, Flink rolls back to the latest checkpoint, restoring
  the consistent state.
- **Exactly-Once Processing**: To guarantee that each record in a data stream is
  processed exactly once, even in the face of failures, Flink uses checkpoints
  to maintain a clear boundary between what has been processed and what remains
  to be processed. By recovering from the latest checkpoint and using log-based
  sources like Kafka, Flink can ensure that each record is processed only once,
  without any duplicates or missed records.
- **Scalability**: In a distributed system, operators can be dynamically scaled
  up or down according to the workloads. Checkpoints and state snapshots provide
  a consistent view of the application, allowing Flink to rescale its tasks and
  redistribute its state to new operators, ensuring smooth scaling operations.
- **Savepoints**: Apart from fault tolerance, checkpoints are also the
  foundation for Flink savepoints. Savepoints are manually triggered snapshots
  that are used for versioning, upgrading an application to a new version, or
  changing the application's parallelism without losing its state.

Flink can save checkpoints by periodically storing the state of an application
to a durable, distributed storage system. This process involves a series of
coordinated actions between the Flink runtime, the tasks running the job, and
the chosen storage system.

### Saving Check Point Process

Here is an overview of how Flink saves checkpoints:

- **Configuration**: To enable checkpoints, you need to configure them in your
  Flink application by setting a checkpointing interval, specifying the
  checkpoint storage system, and defining the checkpoint consistency level (at
  least once or exactly once). For example, in Java:
    ```java 
    StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
    env.enableCheckpointing(60000); // Enable checkpointing and set the interval to 60 seconds
    env.getCheckpointConfig().setCheckpointStorage("hdfs:///flink/checkpoints"); // Define the storage location
    env.getCheckpointConfig().setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE); // Set the consistency level
    ```
- **Checkpoint Coordinator**: Flink designates one of the JobManager instances
  in the cluster as the Checkpoint Coordinator. It is responsible for triggering
  checkpoints at the specified interval and coordinating the checkpoint process
  across the tasks.
- **Task State Snapshot**: When the checkpoint coordinator triggers a
  checkpoint, all the tasks running the Flink application start taking a
  snapshot of their operator states in a distributed, parallel manner. To ensure
  consistency, Flink uses a mechanism called "barrier alignment" where input
  streams are divided into pre-checkpoint and post-checkpoint records using
  checkpoint barriers. Tasks make sure that all their operator states correspond
  to the pre-checkpoint records.
- **Checkpoint Storage**: Once all tasks have successfully taken their state
  snapshots, they store their state in the specified distributed storage system.
  This can be a file system like HDFS, AWS S3, or a database like Apache
  Cassandra or RocksDB. If configured, the state backends, such as
  RocksDBStateBackend, perform incremental checkpointing to save the differences
  since the last completed checkpoint, reducing the amount of data that needs to
  be stored.
- **Acknowledgment**: After storing the state to the storage system, each task
  sends an acknowledgment to the Checkpoint Coordinator. If the coordinator
  receives acknowledgments from all tasks, the checkpoint is considered
  successful, and the metadata about this checkpoint is kept. If any task fails
  to save its state or send an acknowledgment, the checkpoint is considered
  failed, and the coordinator discards the checkpoints in the storage system.

### Loading Check Point Process

Restoring a checkpoint in Flink is essential for recovering the state of a job
in case of failures or restarts. Flink handles checkpoint restoration
automatically when restarting a failed job or when you want to resume a job from
a savepoint.

Here is an outline of the steps involved in restoring a checkpoint:

- **Configure Checkpoints**: Ensure that your Flink application has checkpoints
  enabled and properly configured by specifying the checkpoint interval, storage
  location, and checkpointing mode (at least once or exactly once). This
  configuration is required for Flink to store its periodic snapshots and use
  them to restore its state during recovery.
- **Detect Failure**: When a failure occurs, Flink detects the failed tasks and
  triggers a recovery process. This process involves canceling the tasks,
  cleaning up resources, and initiating the restart of the application.
- **Identify Latest Checkpoint**: Flink accesses the checkpoint storage and
  locates the latest successfully completed checkpoint. The metadata of this
  checkpoint contains information about how to access the snapshot of the
  operator states and connections to log-based sources like Kafka.
- **Restart Application**: Flink restarts the tasks of the failed application.
  During the initialization of each task, Flink restores the state of the
  corresponding operator from the snapshot data in the identified checkpoint.
- **Resume Processing**: With the recovered state in place, Flink's tasks resume
  processing from the point of the checkpoint. If you are using log-based
  sources (like Kafka), the non-committed records will be replayed, ensuring
  guaranteed exactly-once processing.

Remember that the checkpoint restoration process is automatic. However, if you
want to restore a Flink job from a savepoint (a manual checkpoint), you need to
provide the savepoint path while submitting the job. Here's an example using the
Flink CLI for a job with JobID <jobId>:

```bash
./bin/flink run -s <savepoint-path> -j <path-to-job-jar> -c <job-class> [<job arguments>...]
```

## Save Point

A savepoint is a **point-in-time snapshot** of the state of a running Flink job,
similar to a **checkpoint**. However, savepoints are created manually on-demand by
the user, rather than periodically and automatically like checkpoints.
Savepoints are stored in a distributed and durable storage system (e.g., HDFS or
S3), allowing users to manage the application's state.

The primary use of savepoints is for managing the lifecycle of a Flink
application, enabling features such as job versioning, upgrades, and rollbacks.
Users can stop a job, take a savepoint, and then restart the job from that
savepoint while making changes to the job's code or configuration. This enables
users to update their application, revert to an earlier version, or migrate the
job to a new cluster without losing any job state.

## Check Point vs Save Point

- **Triggering Mechanism**: 
  - Checkpoints are taken automatically and periodically by the Flink system,
    according to the specified checkpointing interval.
  - Savepoints are triggered manually by the user on-demand, using Flink's
    command-line interface or REST API.
- **Primary Use-Case**: 
  - Checkpoints are primarily used for fault-tolerance, ensuring that a
    streaming application can automatically recover from failures and resume
    processing from a consistent state with minimal data loss.
  - Savepoints are primarily used for managing the lifecycle of a Flink
    application. They allow users to upgrade, downgrade, or modify the
    application's code or configuration and support tasks such as versioning,
    job migration, and rollbacks.
- **Retention Policy**:
  - Checkpoints are usually retained only for a limited time, depending on the
    configured checkpointing and retention settings. Old checkpoints are
    discarded automatically once they are no longer needed for recovery.
  - Savepoints are retained until explicitly deleted by the user. They persist
    even after a job is canceled or completed, enabling users to manage the
    job's lifecycle more flexibly.
- **Compatibility**:
  - Checkpoints are generally tied to the specific version and configuration of
    the Flink job. They may not be compatible across different Flink versions or
    significant job changes.
  - Savepoints provide a more stable and compatible framework for managing job
    state across different Flink versions or job updates. Flink offers migration
    tools and guidelines to help users maintain compatibility between job
    versions when using savepoints.

## State Backend

The **state backend** is a component responsible for managing and persisting the
state of a streaming application. The state backend handles **state storage**,
**checkpointing**, **recovery**, and **other state-related operations**,
ensuring fault-tolerance and exactly-once processing semantics. Flink provides
several built-in state backends, each with different performance characteristics
and tradeoffs.

Flink currently supports the following state backends:

- **MemoryStateBackend**: This backend stores the state in the JobManager's Java
  heap memory. Checkpoints are also stored in memory and can be configured to
  spill over to disk when they reach a certain size. MemoryStateBackend is
  primarily used for development and testing purposes, as it is not suitable for
  very large state sizes or high-availability scenarios.
- **FileSystemStateBackend**: This backend stores the state on a distributed
  file system such as HDFS, S3, or another supported storage system. Checkpoints
  are also written to the file system, ensuring durability and fault-tolerance.
  FileSystemStateBackend is suitable for production environments and can handle
  large state sizes.
- **RocksDBStateBackend**: This backend stores the state in an embedded,
  RocksDB-based key-value store. Checkpoints are taken incrementally, reducing
  the impact on application performance and are written to a distributed file
  system for durability. RocksDBStateBackend can handle very large state sizes
  with low-latency access and is recommended for most production use-cases.

The choice of state backend depends on factors such as state size, access
patterns, performance requirements, and fault-tolerance needs. Users can
configure the state backend in their Flink application, ensuring that the system
meets their requirements for scalability and reliability.

## RocksDB

RocksDB is not a file system but a high-performance embedded key-value store
library developed by Facebook. In Flink, RocksDB is predominantly used as a
state backend. The state backend is responsible for managing and storing the
state of an application, including key-value states and windows, and plays a
crucial role in Flink's stateful computations and fault-tolerance mechanisms.

The RocksDB State Backend offers a range of benefits for Flink applications:

- **Efficient State Management**: RocksDB is optimized for high read and write
  performance on SSDs or persistent memory, which ensures efficient state
  management even for large stateful applications.
- **Embeddable Store**: RocksDB is an embedded library for C++, which means it
  can be embedded within the Flink application without the need for a separate
  service or infrastructure. This simplifies deployment and ensures low access
  latency for state updates and queries.
- **Incremental Checkpoints**: The RocksDB State Backend supports incremental
  checkpoints, storing only the differences between two subsequent checkpoints
  instead of saving the full state in each checkpoint. This saves time and
  reduces the I/O and storage overhead during checkpointing.
- **Asynchronous State Snapshots**: RocksDB State Backend can take asynchronous
  state snapshots, ensuring that state updates and queries are not blocked
  during checkpointing. This minimizes the impact of checkpoints on the
  application's performance.
- **Out-of-Core State Support**: RocksDB is able to spill state data to disk if
  it doesn't fit in memory, allowing Flink to manage large state sizes that
  surpass the available main memory. This makes it suitable for use cases with
  large stateful computations.

## Sharing Data Among Task Managers

If you'd like to avoid using an additional component like
[Redis](/redis/README.md), you can utilize Flink's built-in 
**Broadcast State pattern** to share data among task managers efficiently. 
Broadcast State is a read-only state shared among parallel instances 
of an operator in a Flink job. It is especially useful for sharing configuration settings, 
rules, reference data, or model updates without needing an external component.

Here's an example of how to use Broadcast State in Flink:

Define the data types for the main data and broadcast data:

```java
import org.apache.flink.api.java.tuple.Tuple2;

public class MainData {
    // your main data fields
}

public class BroadcastData {
    // your broadcast data fields
}
```

Implement a custom `BroadcastProcessFunction`:

```java
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.common.state.ReadOnlyBroadcastState;
import org.apache.flink.streaming.api.functions.co.BroadcastProcessFunction;
import org.apache.flink.util.Collector;

public class CustomBroadcastProcessFunction extends BroadcastProcessFunction<MainData, BroadcastData, Tuple2<String, MainData>> {

    private final MapStateDescriptor<String, BroadcastData> broadcastStateDescriptor;

    public CustomBroadcastProcessFunction(MapStateDescriptor<String, BroadcastData> broadcastStateDescriptor) {
        this.broadcastStateDescriptor = broadcastStateDescriptor;
    }

    @Override
    public void processElement(MainData value, ReadOnlyContext ctx, Collector<Tuple2<String, MainData>> out) throws Exception {
        ReadOnlyBroadcastState<String, BroadcastData> broadcastState = ctx.getBroadcastState(broadcastStateDescriptor);

        // Access the shared broadcastState according to your use case
        BroadcastData sharedData = broadcastState.get("some_key");
        
        // Perform your desired processing
    }

    @Override
    public void processBroadcastElement(BroadcastData value, Context ctx, Collector<Tuple2<String, MainData>> out) throws Exception {
        ctx.getBroadcastState(broadcastStateDescriptor).put("some_key", value);
    }
}
```

Set up the Broadcast State in your Flink program:

```java
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.api.java.typeutils.TypeInformation;
import org.apache.flink.streaming.api.datastream.BroadcastConnectedStream;
import org.apache.flink.streaming.api.datastream.DataStream;

DataStream<MainData> mainDataStream = ... // your main data stream
DataStream<BroadcastData> broadcastDataStream = ... // your broadcast data stream

MapStateDescriptor<String, BroadcastData> broadcastStateDescriptor = new MapStateDescriptor<>(
        "broadcastState", TypeInformation.of(String.class), TypeInformation.of(BroadcastData.class));

BroadcastConnectedStream<MainData, BroadcastData> connectedStream = mainDataStream
        .connect(broadcastDataStream.broadcast(broadcastStateDescriptor));

DataStream<Tuple2<String, MainData>> resultStream = connectedStream.process(new CustomBroadcastProcessFunction(broadcastStateDescriptor));
```

In this example, the Broadcast State pattern is used to share `BroadcastData`
among parallel instances of a Flink operator that processes MainData. The
`broadcastDataStream` contains the data you want to share, and the
mainDataStream contains the data you want to process using the shared broadcast
data.

Using the **Broadcast State pattern**, you can eliminate the need for an additional
component like [Redis](/redis/README.md) and maintain efficient data sharing directly within Flink's
built-in capabilities.

## Graceful Shutdown

Flink supports **graceful shutdown**, which allows the streaming application to
stop processing new data while processing the remaining data in the pipeline. A
graceful shutdown can be triggered using the "stop" command from the
command-line interface or REST API. During graceful shutdown, Flink ensures that
all ongoing checkpoints, state updates, and already processed data are handled
correctly to maintain consistency and no data is lost.

Here's an example of how to perform a graceful shutdown using the command-line interface:

```bash
./bin/flink stop <JobID>
```

Replace `<JobID>` with your Flink job's actual job ID.

You can also perform a graceful shutdown using Flink's REST API. Here's an
example of a curl command that triggers a graceful shutdown:

```bash
curl -X PATCH http://<JobManagerAddress>:<Port>/jobs/<JobID>/stop
```

Replace `<JobManagerAddress>` and `<Port>` with the actual address and port of
your Flink JobManager, and `<JobID>` with your job's ID.

When a Flink job is stopped gracefully, the following steps occur:

Flink stops accepting new input data from the sources and waits for all data
that is currently being processed to complete. Flink initiates a new savepoint,
which persists the final state of the job. The job is marked as "stopped" and
the user can restart the job from the latest savepoint, ensuring a consistent
state. Using graceful shutdown in Flink, you can stop your streaming application
in a controlled and safe manner, maintain exactly-once processing semantics, and
have the ability to resume the job from a consistent state if needed.
