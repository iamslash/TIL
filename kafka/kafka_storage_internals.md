- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)

----

# Abstract

Kafka Storage Internals 를 정리한다.

# Materials

* [Kafka Internals](https://developer.confluent.io/learn-kafka/architecture/get-started/)
  * Videos
* [How Kafka’s Storage Internals Work](https://thehoard.blog/how-kafkas-storage-internals-work-3a29b02e026)
* [A Practical Introduction to Kafka Storage Internals @ medium](https://medium.com/@durgaswaroop/a-practical-introduction-to-kafka-storage-internals-d5b544f6925f)

# Basic

다음과 같이 Kafka topic 을 만들어 보자. Kafka topic 은 3 개의 partition 으로 나누어진다. 하나의 partition 은 여러개의 segment file 로 저장된다. 

```bash
$ kafka-topics.bat --create --topic freblogg --partitions 3 --replication-factor 1 --zookeeper localhost:2181
```

그리고 다음과 같이 segment file 들을 확인해 보자.

```
$ tree freblogg*
freblogg-0
|-- 00000000000000000000.index
|-- 00000000000000000000.log
|-- 00000000000000000000.timeindex
`-- leader-epoch-checkpoint
freblogg-1
|-- 00000000000000000000.index
|-- 00000000000000000000.log
|-- 00000000000000000000.timeindex
`-- leader-epoch-checkpoint
freblogg-2
|-- 00000000000000000000.index
|-- 00000000000000000000.log
|-- 00000000000000000000.timeindex
`-- leader-epoch-checkpoint

$ ls -lh freblogg-0
total 20M
- freblogg 197121 10M Aug  5 08:26 00000000000000000000.index
- freblogg 197121   0 Aug  5 08:26 00000000000000000000.log
- freblogg 197121 10M Aug  5 08:26 00000000000000000000.timeindex
- freblogg 197121   0 Aug  5 08:26 leader-epoch-checkpoint
```

`freblogg-0,freblogg-1,freblogg-2` 는 각각 partition 을 의미한다. 하나의 segment 는 `*.index, *.log, *.timeindex, leader-epoch-checkpoint` 파일들로 구성된다. segment file 의 이름은 `*.log` 에 보관된 첫번째 Kafka Record Data 의 offset 을 의미한다. `00000000000000000000.log` 의 첫번째 Kafka Record Data 의 offset 은 `0` 이라는 말이다.

이제 다음과 같이 Message 를 두개 보내 보자.

```bash
$ kafka-console-producer.bat --topic freblogg --broker-list localhost:9092
> Hello World
> Hello World
> amazon

$ ls -lh freblogg*
freblogg-0:
total 20M
- freblogg 197121 10M Aug  5 08:26 00000000000000000000.index
- freblogg 197121   0 Aug  5 08:26 00000000000000000000.log
- freblogg 197121 10M Aug  5 08:26 00000000000000000000.timeindex
- freblogg 197121   0 Aug  5 08:26 leader-epoch-checkpoint

freblogg-1:
total 21M
- freblogg 197121 10M Aug  5 08:26 00000000000000000000.index
- freblogg 197121  68 Aug  5 10:15 00000000000000000000.log
- freblogg 197121 10M Aug  5 08:26 00000000000000000000.timeindex
- freblogg 197121  11 Aug  5 10:15 leader-epoch-checkpoint

freblogg-2:
total 21M
- freblogg 197121 10M Aug  5 08:26 00000000000000000000.index
- freblogg 197121  79 Aug  5 09:59 00000000000000000000.log
- freblogg 197121 10M Aug  5 08:26 00000000000000000000.timeindex
- freblogg 197121  11 Aug  5 09:59 leader-epoch-checkpoint
```

첫번째 메시지는 `freblogg-2` partition 에 저장되었다. 두번째 메시지는 `freblogg-1` partition 에 저장되었다.

메시지가 어떻게 저장되어 있는지 확인해 보자. "Hello World" 는 확실히 보인다.

```bash
$ cat freblogg-2/*.log
@^@^BÂ°Â£Ã¦Ãƒ^@^K^XÃ¿Ã¿Ã¿Ã¿Ã¿Ã¿^@^@^@^A"^@^@^A^VHello World^@
```

이번에는 다음과 같은 방법으로 메시지를 확인해 보자.

```bash
$ kafka-run-class.bat kafka.tools.DumpLogSegments --deep-iteration --print-data-log --files logs\freblogg-2\00000000000000000000.log

Dumping logs\freblogg-2\00000000000000000000.log
Starting offset: 0

offset: 0 position: 0 CreateTime: 1533443377944 isvalid: true keysize: -1 valuesize: 11 producerId: -1 headerKeys: [] payload: Hello World

offset: 1 position: 79 CreateTime: 1533462689974 isvalid: true keysize: -1 valuesize: 6 producerId: -1 headerKeys: [] payload: amazon
```

다음은 `*.index` 와 `*.log` 의 구조이다. `*.index` 을 이용하면 offset 별로 Kafka Record Data 의 position 을 알 수 있다. 

| Offset | Position |
|---|---|
| 0 | 0 |
| 1 | 79 |

| Offset | Position | Time | Message |
|---|---|---|---|
| 0 | 0 | 1533443377944 | Hello World |
| 1 | 79 | 1533462689974 | amazon |

`*.index` 의 offset 들이 메모리에서 정렬되어 있다면 특정 offset 을 binary search 할 수 있다. 그리고 position 을 알아내서 `*.log` 파일을 열어 `O(1)` 으로 Kafka Record Data 를 읽어올 수 있다. 
