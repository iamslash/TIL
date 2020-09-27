- [Abstract](#abstract)
- [Materials](#materials)
- [Install with Docker](#install-with-docker)
- [Basics](#basics)
  - [Features](#features)
    - [Partitioner](#partitioner)
    - [Data Consistency](#data-consistency)
    - [Data Replication](#data-replication)
    - [Snitch](#snitch)
    - [How to write data](#how-to-write-data)
    - [How to read data](#how-to-read-data)
    - [How to delete data](#how-to-delete-data)
    - [How to update data](#how-to-update-data)
    - [Light-weight transaction](#light-weight-transaction)
    - [Secondary Index](#secondary-index)
    - [Batch](#batch)
    - [Collection](#collection)
  - [Good Pattern](#good-pattern)
    - [Time Sequencial Data](#time-sequencial-data)
    - [Denormalize](#denormalize)
    - [Paging](#paging)
  - [Anti Pattern](#anti-pattern)
    - [Unbounded Data](#unbounded-data)
    - [Secondary Index](#secondary-index-1)
    - [Delete Data](#delete-data)
    - [Memory Overflow](#memory-overflow)
  - [Data Types](#data-types)
  - [Useful Queries](#useful-queries)
  - [Basic Schema Design](#basic-schema-design)

----

# Abstract

Cassandra's write performance is good. It is perfect for write heavy jobs. For example, Messages, Logs by time. Cassandra's update performcne is bad. If you want to update records many times, consider to use MongoDB, dynamoDB.

# Materials

* [Apache Cassandra Documentation](https://cassandra.apache.org/doc/latest/)
  * This is mandatory.
* [Cassnadra 의 기본 특징 정리](https://nicewoong.github.io/development/2018/02/11/cassandra-feature/)
* [Apache Cassandra 톺아보기 - 1편](https://meetup.toast.com/posts/58)
  * [Apache Cassandra 톺아보기 - 2편](https://meetup.toast.com/posts/60)
  * [Apache Cassandra 톺아보기 - 3편](https://meetup.toast.com/posts/65)
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
  * `hint and off` 덕분에 Coordinator 는 비정상이었던 target node 의 상태가 정상으로 회복되면 write request 를 다시 보낼 수 있다.
  * 그러나 `hint hand off` 에 저장하고 coordinator 가 죽어버리면 방법이 없다.
* Coordinator 는 `hint and off` 에 data 를 backup 하고 Cassandra 의 topology 를 확인하여 어느 데이터 센터의 어느 렉에 있는 노드에 먼저 접근할 것인지 결정한다. 그리고 그 node 에 write request 한다.

![](cassandra_data_write_2.png)

* Target node 는 Coordinator 로 부터 write request 를 수신하면 `CommitLog` 라는 local disk 에 저장한다. 
* Target node 는 `MemTable` 이라는 memory 에 data 를 write 하고 response 를 Coordinator 에게 보낸다.
* Target node 는 `MemTable` 에 data 가 충분히 쌓이면 `SSTable` 이라는 local disk 에 data 를 flush 한다.
  * `SSTable` 은 immutalbe 하고 sequential 하다.
  * Cassandra 는 다수의 `SSTable` 을 정기적으로 Compaction 한다. 예를 들어 n 개의 `SSTable` 에 `a` 라는 데이터가 여러개 존재한다면 Compaction 할 때 가장 최신의 버전으로 merge 한다.

### How to read data

![](cassandra_data_read.png)

* Client 는 임의의 Cassandra node 에게 read request 한다. 이때 read request 를 수신한 node 를 Coordinator 라고 한다.
* Coordinator 는 수신한 data 의 Row key 를 token 으로 변환하고 어떤 node 에 data 를 read 해야 하는지 판단한다. 그리고 Query 에 저장된 Consistency level 에 따라 몇 개의 Replication 을 확인할지 결정한다. 그리고 data 가 있는 가장 가까운 node 에 data request 를 보내고 그 다음 가까운 node 들에게는 data digest request 를 보낸다.
* Coordinator 는 이렇게 가져온 data response 와 data digest response 를 참고하여 data 가 일치하지 않으면 일치하지 않는 node 들로 부터 full data 를 가져와서 그들 중 가장 최신 버전의 data 를 client 에게 보낸다. 
  * 그리고 그 data 를 이용하여 target node 들의 data 를 보정한다.

![](cassandra_data_read_2.png)

* Target node 는 read request 를 수신하면 가장 먼저 `MemTable` 을 확인한다. 
* 만약 없다면 `SSTable` 을 확인해야 한다. 그러나 그 전에 `SStable` 과 짝을 이루는 `Bloom Filter` 와 `Index` 를 먼저 확인 한다.
  * `Bloom Filter` 는 확률적 자료구조이다. 데이터가 없는 걸 있다고 거짓말 할 수 있지만 있는 걸 없다고 거짓말 하지는 못한다.
  * `Bloom Filter` 가 있다고 하면 `SSTable` 에 반드시 데이터가 있음을 의미한다.
* `Bloom Filter` 를 통해 `SSTable` 에 data 가 있다는 것을 알았다면 메모리에 저장되어 있는 `Summary Index` 를 통해 disk 에 저장되어 있는 원본 index 를 확인하여 `SSTable` 의 data 위치에 대한 offset 을 알게된다. 그리고 `SSTable` 에서 원하는 data 를 가져올 수 있다.
  * 이러한 과정은 최근에 생성된 `SSTable` 부터 차례대로 이루어진다.

### How to delete data

* Cassandra 는 delete 을 바로 수행하지 않는다. 모든 data 는 `Tombstone` 이라는 marker 를 갖는다. delete request 를 수신하면 그 data 의 `Tombstone` 에 marking 을 하고 주기적인 `Garbage Collection` 이나 `SSTable` 의 compaction 등이 발생할 때 data 를 삭제한다.

### How to update data

* Cassandra 는 update request 를 수신하면 `delete` process 를 수행하고 `write` process 를 수행한다.

### Light-weight transaction

### Secondary Index

### Batch

### Collection

## Good Pattern

### Time Sequencial Data

### Denormalize

### Paging

## Anti Pattern

### Unbounded Data

### Secondary Index

### Delete Data

### Memory Overflow

## Data Types

* [CQL data types](https://docs.datastax.com/en/cql-oss/3.x/cql/cql_reference/cql_data_types_c.html)

-----

* `text` is a alias for `varchar.`

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
* [Data Modeling @ Cassandra](https://cassandra.apache.org/doc/latest/data_modeling/index.html)

This is a hotel keyspace.

```sql
CREATE KEYSPACE hotel WITH replication =
  {‘class’: ‘SimpleStrategy’, ‘replication_factor’ : 3};

CREATE TYPE hotel.address (
  street text,
  city text,
  state_or_province text,
  postal_code text,
  country text );

CREATE TABLE hotel.hotels_by_poi (
  poi_name text,
  hotel_id text,
  name text,
  phone text,
  address frozen<address>,
  PRIMARY KEY ((poi_name), hotel_id) )
  WITH comment = ‘Q1. Find hotels near given poi’
  AND CLUSTERING ORDER BY (hotel_id ASC) ;

CREATE TABLE hotel.hotels (
  id text PRIMARY KEY,
  name text,
  phone text,
  address frozen<address>,
  pois set )
  WITH comment = ‘Q2. Find information about a hotel’;

CREATE TABLE hotel.pois_by_hotel (
  poi_name text,
  hotel_id text,
  description text,
  PRIMARY KEY ((hotel_id), poi_name) )
  WITH comment = Q3. Find pois near a hotel’;

CREATE TABLE hotel.available_rooms_by_hotel_date (
  hotel_id text,
  date date,
  room_number smallint,
  is_available boolean,
  PRIMARY KEY ((hotel_id), date, room_number) )
  WITH comment = ‘Q4. Find available rooms by hotel date’;

CREATE TABLE hotel.amenities_by_room (
  hotel_id text,
  room_number smallint,
  amenity_name text,
  description text,
  PRIMARY KEY ((hotel_id, room_number), amenity_name) )
  WITH comment = ‘Q5. Find amenities for a room’;
```

This is a reservation keyspace

```sql
CREATE KEYSPACE reservation WITH replication = {‘class’:
  ‘SimpleStrategy’, ‘replication_factor’ : 3};

CREATE TYPE reservation.address (
  street text,
  city text,
  state_or_province text,
  postal_code text,
  country text );

CREATE TABLE reservation.reservations_by_confirmation (
  confirm_number text,
  hotel_id text,
  start_date date,
  end_date date,
  room_number smallint,
  guest_id uuid,
  PRIMARY KEY (confirm_number) )
  WITH comment = ‘Q6. Find reservations by confirmation number’;

CREATE TABLE reservation.reservations_by_hotel_date (
  hotel_id text,
  start_date date,
  end_date date,
  room_number smallint,
  confirm_number text,
  guest_id uuid,
  PRIMARY KEY ((hotel_id, start_date), room_number) )
  WITH comment = ‘Q7. Find reservations by hotel and date’;

CREATE TABLE reservation.reservations_by_guest (
  guest_last_name text,
  hotel_id text,
  start_date date,
  end_date date,
  room_number smallint,
  confirm_number text,
  guest_id uuid,
  PRIMARY KEY ((guest_last_name), hotel_id) )
  WITH comment = ‘Q8. Find reservations by guest name’;

CREATE TABLE reservation.guests (
  guest_id uuid PRIMARY KEY,
  first_name text,
  last_name text,
  title text,
  emails set,
  phone_numbers list,
  addresses map<text, frozen<address>,
  confirm_number text )
  WITH comment = ‘Q9. Find guest by ID’;
```
