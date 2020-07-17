# Materials

* [Cassnadra 의 기본 특징 정리](https://nicewoong.github.io/development/2018/02/11/cassandra-feature/)
* [Apache Cassandra 톺아보기 - 1편](https://meetup.toast.com/posts/58)
  * [Apache Cassandra 톺아보기 - 2편](https://meetup.toast.com/posts/60)
  * [Apache Cassandra 톺아보기 - 3편](https://meetup.toast.com/posts/65)
* [Cassandra(카산드라) 내부 구조](https://nicewoong.github.io/development/2018/02/11/cassandra-internal/)
* [아파치 분산 데이타 베이스 Cassandra 소개 @ bcho](https://bcho.tistory.com/440)
  * [Cassandra Node CRUD Architecture @ bcho](https://bcho.tistory.com/657?category=431286)
  * []()
* [How To Install Cassandra and Run a Single-Node Cluster on Ubuntu 14.04 @ digitalocean](https://www.digitalocean.com/community/tutorials/how-to-install-cassandra-and-run-a-single-node-cluster-on-ubuntu-14-04)

# Install with Docker

* [cassandra @ docker-hub](https://hub.docker.com/_/cassandra)

```console
$ docker run --name my-cassandra -d cassandra

$ docker exec -it my-cassandra bash
> cqlsh
```

# Basics

## Features

### Partitioner

Partitioner is a module which transform "Row key" to "token". There 3 partitioners such as "RandomPartitioner, Murmur3Partitioner, ByteOrderedPartitioner".

`ByteOrderedPartitioner` can make hot spot situation.

![](https://image.toast.com/aaaadh/real/2016/techblog/apache1%282%29.png)

`RandomPartiioner` convert "Row Key" to "token" using "MD5".

`Murmur3Partitioner` convert "Row Key" to "token" using "Murmur5".

### Data Consistency

* [Switching snitches @ datastax](https://docs.datastax.com/en/archived/cassandra/3.0/cassandra/operations/opsSwitchSnitch.html)
* [Data consistency @ datastax](https://docs.datastax.com/en/archived/cassandra/3.0/cassandra/dml/dmlDataConsistencyTOC.html)

몇개의 Replication 을 통해서 어느 정도 수준의 데이터 일관성을 확보할 것인지 설정할 수 있다.

### Data Replication

Keyspace 를 생성할 때 Replication 의 배치전략, 복제개수, 위치등을 결정할 수 있다. 

### Snitch

`/opt/cassandra/conf/cassandra.yaml` 의 snitch 설정을 통해서 다수의 Data Center 설정등을 할 수 있다.

### How to write data

![](cassandra_data_write.png)

* Client 는 임의의 Cassandra node 에게 Write request 한다. 이때 Write request 를 수신한 node 를 Coordinator 라고 한다.
* Coordinator 는 수신한 data 의 Row key 를 token 으로 변환하고 어떤 node 에 data 를 write 해야 하는지 판단한다. 그리고 Query 에 저장된 Consistency level 에 따라 몇 개의 node 에 write 할지 참고하여 data 를 write 할 node 들의 status 를 확인한다.
* 이때 특정한 node 의 status 가 정상이 아니라면 consistency level 에 따라 coordinator 의 `hint hand off` 라는 local 의 임시 저장공간에 write 할 data 를 저장한다.
  * 만약 후에 비정상인 node 의 상태가 정상으로 회복되면 Coordinator 가 data 를 write 해줄 수 있다.
  * `hint hand off` 에 저장하고 coordinator 가 죽어버리면 방법이 없다.
* Coordinator 는 `hint and off` 에 data 를 backup 하고 Cassandra 의 topology 를 확인하여 어느 데이터 센터의 어느 렉에 있는 노드에 먼저 접근할 것인지 결정한다. 그리고 그 node 에 Write request 한다.

![](cassandra_data_write_2.png)

* Target node 는 Coordinator 로 부터 Write request 를 수신하면 `CommitLog` 라는 local disk 에 저장한다. 
* Target node 는 `MemTable` 이라는 memory 에 data 를 write 하고 reponse 를 Coordinator 에게 보낸다.
* Target node 는 `MemTable` 에 data 가 충분히 쌓이면 `SSTable` 이라는 local disk 에 data 를 flush 한다.
  * `SSTable` 은 immutalbe 하고 sequential 하다.
  * Cassandra 는 다수의 `SSTable` 을 정기적으로 Compaction 한다. 예를 들어 n 개의 `SSTable` 에 `a` 하는 데이터가 존재한다면 Compaction 할 때 가장 최신의 버전으로 merge 한다.

### How to read data

![](cassandra_data_read.png)

### How to update data


## Useful Queries 

* [Cassandra @ tutorialpoint](https://www.tutorialspoint.com/cassandra/index.htm)

```bash
# Create keyspace
> CREATE KEYSPACE iamslash
WITH replication = {'class': 'SimpleStrategy', 'replication_factor' : 3};

# Describe cluster
> DESCRIBE cluster;

# Describe keyspaces;
> DESCRIBE keyspaces;

# Describe tables;
> DESCRIBE tables;

# Create a table
CREATE TABLE iamslash.person ( 
    code text, 
    location text, 
    sequence text, 
    description text, 
    PRIMARY KEY (code, location)
);

# Describe a table
> DESCRIBE iamslash.person;

# Insert data
INSERT INTO iamslash.person (code, location, sequence, description ) VALUES ('N1', 'Seoul', 'first', 'AA');
INSERT INTO iamslash.person (code, location, sequence, description ) VALUES ('N1', 'Gangnam', 'second', 'BB');
INSERT INTO iamslash.person (code, location, sequence, description ) VALUES ('N2', 'Seongnam', 'third', 'CC');
INSERT INTO iamslash.person (code, location, sequence, description ) VALUES ('N2', 'Pangyo', 'fourth', 'DD');
INSERT INTO iamslash.person (code, location, sequence, description ) VALUES ('N2', 'Jungja', 'fifth', 'EE');

INSERT INTO iamslash.person (code, location, sequence, description ) VALUES ('N3', 'Songpa', 'seventh', 'FF');

# Select data
> select * from iamslash.person;

# Update data
> update iamslash.person SET sequence='sixth' WHERE code='N3' AND location='Songpa';
> select * from iamslash.person;

# Delete a column (Range delettions)
>  Delete sequence FROM iamslash.person WHERE code='N3' AND location='Songpa';
> select * from iamslash.person;

# Delete a row
> Delete FROM iamslash.person WHERE code='N3';
```

## Basic Schema Design

* [Basic Rules of Cassandra Data Modeling](https://www.datastax.com/blog/2015/02/basic-rules-cassandra-data-modeling)
* [Data Modeling in Cassandra @ baeldung](https://www.baeldung.com/cassandra-data-modeling)
