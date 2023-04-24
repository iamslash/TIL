- [Requirements](#requirements)
  - [Functional Requirements](#functional-requirements)
  - [Non-Functional Requirements](#non-functional-requirements)
- [High Level Design](#high-level-design)
  - [Components](#components)
  - [Data Model](#data-model)
    - [Data Access Pattern](#data-access-pattern)
    - [Data Storage System](#data-storage-system)
  - [High Level Architecture](#high-level-architecture)
- [High Level Design Deep Dive](#high-level-design-deep-dive)
  - [Metrics Collection](#metrics-collection)
  - [Scale Metrics Transmission Pipeline](#scale-metrics-transmission-pipeline)
  - [Where Aggregations can happen](#where-aggregations-can-happen)
  - [Query Service](#query-service)
  - [Storage Layer](#storage-layer)
    - [Time-Series DataBase](#time-series-database)
    - [Space Optimization](#space-optimization)
  - [Alerting System](#alerting-system)
  - [Visualization](#visualization)

-----

# Requirements

## Functional Requirements

* `100 million` DAU
* `1,000` server pools, `100` machines per pool, `100` metrics per machine
  * `10 million` metrics
* `1 year` data retention
* raw from for `7 days`, `1 minute` resolution for `30 days`, `1 hour` resolution for `1 year`.
* metrics include CPU usage, request count, memory usage, message count in message queues.

## Non-Functional Requirements

* Scalability
* Low latency
* Reliablility
* Flexibility
  * The system should be integrated with new tech stack in the future easily.

# High Level Design

## Components 

* Data collection
  * Collect metric data from different sources.
* Data transmission
  * Transfer data from sources to the metrics monitoring system.
* Data storage
  * Organize and store incomming data.
* Alerting 
  * Generate alerts for anomalies. (Pager duty)
* Visualization
  * Present data in graphs. [Grafana](/grafana/README.md)  

## Data Model

What is the average CPU load across all web servers in the us-west region for
the last 10 minutes? Ths is an example of metrics.

```
CPU.load host=webserver01,region=us-west 1613707265 50
CPU.load host=webserver01,region=us-west 1613707265 62
CPU.load host=webserver02,region=us-west 1613707265 43
CPU.load host=webserver02,region=us-west 1613707265 53
...
CPU.load host=webserver01,region=us-west 1613707265 76
CPU.load host=webserver01,region=us-west 1613707265 83
```

| name | type |
|--|--|
| metric name | String |
| tags/labels | List of `<key:value>` pairs |
| array of values, timestamp | array of `value, timestamp>` pairs | 

### Data Access Pattern

The **write load** is heavy. The **read load** is spiky.

### Data Storage System

General purpose database such as [MySQL](/mysql/README.md) is not a good
solution for time-series data. 

Time-series database is a good solution. OpenTSDB is good but is based on Hadoop
and HBase. It is too complicated. Twitter uses MetricsDB and Amazon offers
Timestream. InfluxDB, [Prometheus](/prometheus/README.md) are good solutions.

InfluxDB (8 cores, 32 GB RAM) can handle over 250,000 writes per second. This is
a benchmarking for InfluxDB.

| vCPU | RAM | IOPS | Writes per sec | Queries per sec | Unique series |
|--|--|--|--|--|--|
| 2-4 | 2-4 GB | `500` | `< 5000` | `< 5` | `< 100,000` |
| 4-6 | 8-32 GB | `500 - 1000` | `< 250,000` | `< 25` | `< 100,000` |
| 8+ | 32+ GB | `1000+` | `> 250,000` | `> 25` | `> 1,000,000` |

## High Level Architecture 

![](img/2023-03-21-20-05-45.png)

# High Level Design Deep Dive

## Metrics Collection

Pull vs Push

* Easy Debugging
  * Pull: You can debug using `/metrics` endpoint even on your laptop.
* Health Check
  * Pull: If an application server does not respond for pulling, you can check the health.
* Short-lived jobs
  * Push: The batch jobs will be short-lived.
* Firewall or complicated network setups
  * Pull: All `/metrics` endpoint should be reachable.
  * Push: If metric collector is behind loadbalancers, Push is easier.
* Performance
  * Pull: Use TCP.
  * Push: Use UDP. It's more efficient.
* Data authenticity
  * Pull: API Key???
  * Push: Whitelist.

## Scale Metrics Transmission Pipeline

**Scale through kafka**

* Kafka decouples between the data collection and data processing.

**alternative to kafka**

* [m3](/m3/README.md)

## Where Aggregations can happen

**Collction agent**

It will run on the client-side.

**Ingestion pipeline**

It will run before writing to storage. [Flink](/flink/README.md) is a good example.

**Query side**

It will run after writing to storage.

## Query Service

**Cache Layer**

It will cache query results.

**Time-series Database Query**

It is hard to build SQL to query time-searies data.

```sql
  SELECT id,
         temp,
         avg(temp) over (partition by group_nr order by time_read) AS rolling_avg         
    FROM (
           SELECT id, temp, time_read, interval_group
                  id - row_number() over (partition by interval_group ORDER BY time_read) AS group_nr
             FROM (...) t1            
         ) t2
ORDER BY time_read;  
```

Flux (InfluxDB) is simpler than SQL.

```
from(db:"telegraf")
  |> range(start:-1h)
  |> filter(fn: (r) => r._measurement == "foo")
  |> expoentialMovingAverage(size:-10s)
```

## Storage Layer

### Time-Series DataBase

[According to the Facebook](https://www.vldb.org/pvldb/vol8/p1816-teller.pdf), At least 85% of all queries are for past 26 hours. InFlux
ia a good solution.

### Space Optimization

**Data encoding and compression**

```
32 bits for ts
1610087371  1610087381  1610087391  1610087400  1610087411 

under 32 bits for ts with delta
1610087371          10          10           9          11
```

**Downsampling**

* Retention: 7 days no sampling
* Retention: 30 days downsample to 1 min resolution
* Retention: 1 year, downsample to 1 hr resolution

10-sec resolution data

| metric | ts | hostname | metric_value |
|--|--|--|--|
| cpu | 2021-10-24T19:00:00Z | iamslash | 10 |
| cpu | 2021-10-24T19:00:10Z | iamslash | 16 |
| cpu | 2021-10-24T19:00:20Z | iamslash | 20 |
| cpu | 2021-10-24T19:00:30Z | iamslash | 30 |
| cpu | 2021-10-24T19:00:40Z | iamslash | 20 |
| cpu | 2021-10-24T19:00:50Z | iamslash | 30 |

30-sec resolution data

| metric | ts | hostname | metric_value(avg) |
|--|--|--|--|
| cpu | 2021-10-24T19:00:00Z | iamslash | 19 |
| cpu | 2021-10-24T19:00:30Z | iamslash | 25 |

**Cold storage**

The storage for inactive data.

## Alerting System

Pagerduty is a good solution.

## Visualization

[Grafana](/grafana/README.md) is a good solution.
