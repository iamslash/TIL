- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)

----

# Abstract

Log Compacted Topic 을 만들면 동일한 Key 를 갖는 데이터를 줄일 수 있다.

# Materials

* [Log Compacted Topics in Apache Kafka](https://towardsdatascience.com/log-compacted-topics-in-apache-kafka-b1aa1e4665a7)
  
# Basic

다음과 같은 방법으로 Log Compacted Topic 을 제작한다. `--config "cleanup.policy=compact"` 를 사용한다.

```bash
$ kafka-topics --create --zookeeper zookeeper:2181 --topic latest-product-price --replication-factor 1 --partitions 1 --config "cleanup.policy=compact" --config "delete.retention.ms=100"  --config "segment.ms=100" --config "min.cleanable.dirty.ratio=0.01"
```

Log Compacted Topic 은 동일한 Key 를 갖는 Kafka Record Data 들중 가장 마지막 Kafka Record Data 를 유지하고 나머지는 지운다. 

하나의 partition 은 tail, head 로 나누어 진다. tail 부분의 Record Data 는 background thread 에 의해 
compact 된다. head 부분의 Record Data 는 동일한 Key 의 Kafka Record Data 가 존재할 수 있다.

이제 Kafka Record Data 를 producing 하자. key, value 를 `:` 로 구분했다.

```bash
$ kafka-console-producer --broker-list localhost:9092 --topic latest-product-price --property parse.key=true --property key.separator=:
>p3:10$
>p5:7$
>p3:11$
>p6:25$
>p6:12$
>p5:14$
>p5:17$
```

이제 Kafka Record Data 를 consuming 해보자. compact 되었다. key `p5` 를 갖는 Record Data 가 2 개이다. 이것은 Active Segment 의 head 에 있기 때문이다. tail 로 이동한다면 compact 될 것이다. 

```bash
$ kafka-console-consumer --bootstrap-server localhost:9092 --topic latest-product-price --property  print.key=true --property key.separator=: --from-beginning
p3:11$
p6:12$
p5:14$
p5:17$
```

한 발 더 깊게 들어가 보자. 다음은 segment file 들의 목록이다. 하나의 segment file 은 `*.index, *.log, *.timeindex` 로 구성되어 있다. segment file 의 이름은 그 file 의 가장 첫번째 Kafka Record Data 의 offset 과 같다. 예를 들어 `00000000000000000006.log` segment file 의 첫번째 Kafka Recrod Data 는 offset 이 `6` 이다.

```
$ ls /var/lib/kafka/data/latest-product-price-0/
00000000000000000000.index 
00000000000000000000.log 
00000000000000000000.timeindex 
00000000000000000005.snapshot 
00000000000000000006.index
00000000000000000006.log
00000000000000000006.snapshot
00000000000000000006.timeindex
leader-epoch-checkpoint
```

Kafka 는 몇개의 cleaner threads 를 생성한다. cleaner threads 는 background 에서 log compaction 을 수행한다.
