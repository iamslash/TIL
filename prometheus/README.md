
- [Abstract](#abstract)
- [Architecture](#architecture)
- [Materials](#materials)
- [Install](#install)
  - [Install on osx](#install-on-osx)
  - [Deploy with docker on OSX](#deploy-with-docker-on-osx)
  - [Deploy Prometheus, Grafana, NodeExporter with docker-compose](#deploy-prometheus-grafana-nodeexporter-with-docker-compose)
  - [Deploy GitHub Monitoring Stack with docker-compose](#deploy-github-monitoring-stack-with-docker-compose)
- [PromQL](#promql)
  - [Data Model](#data-model)
  - [Time Series](#time-series)
  - [Data Types](#data-types)
  - [Examples](#examples)
  - [Basics](#basics)
    - [Time series Sslectors](#time-series-sslectors)
      - [Instant vector selectors](#instant-vector-selectors)
      - [Range Vector Selectors](#range-vector-selectors)
      - [Time Duration](#time-duration)
      - [Offset modifier](#offset-modifier)
  - [Operators](#operators)
    - [Binary operators](#binary-operators)
      - [Arithmetic binary operators](#arithmetic-binary-operators)
      - [Comparison binary operators](#comparison-binary-operators)
      - [Logical/set binary operators](#logicalset-binary-operators)
    - [Vector Matching (join)](#vector-matching-join)
    - [Aggregation operators](#aggregation-operators)
    - [Binary operator precedence](#binary-operator-precedence)
  - [Functions](#functions)
  - [HTTP API](#http-api)
    - [Expression queries](#expression-queries)
      - [Instance queries](#instance-queries)
      - [Range queries](#range-queries)
    - [prometheus http request duration seconds](#prometheus-http-request-duration-seconds)
    - [grpc server handling seconds](#grpc-server-handling-seconds)
  - [Summary vs Histogram](#summary-vs-histogram)
  - [1. 메트릭 타입 개념](#1-메트릭-타입-개념)
    - [**Histogram**](#histogram)
    - [**Summary**](#summary)
  - [2. Go 코드 예시 (구조체)](#2-go-코드-예시-구조체)
    - [**Histogram 구조체 예시**](#histogram-구조체-예시)
      - [예시 데이터 (5개의 버킷)](#예시-데이터-5개의-버킷)
    - [**Summary 구조체 예시**](#summary-구조체-예시)
      - [예시 데이터 (p90, p99만 기록)](#예시-데이터-p90-p99만-기록)
  - [3. Prometheus에 저장되는 시계열 차이](#3-prometheus에-저장되는-시계열-차이)
    - [Histogram](#histogram-1)
    - [Summary](#summary-1)
  - [4. 쿼리 측면의 차이](#4-쿼리-측면의-차이)
  - [5. 요약](#5-요약)
  - [irate vs rate](#irate-vs-rate)
  - [1. irate(redis\_operation\_seconds\_bucket\[5m\])](#1-irateredis_operation_seconds_bucket5m)
  - [2. rate(redis\_operation\_seconds\_bucket\[5m\])](#2-rateredis_operation_seconds_bucket5m)
  - [3. 예시 (15초마다 데이터가 저장되는 상황)](#3-예시-15초마다-데이터가-저장되는-상황)
    - [**정리**](#정리)
  - [rate deep dive](#rate-deep-dive)
  - [예시로 설명](#예시로-설명)
    - [상황1. 정상적으로 값이 계속 증가하는 경우](#상황1-정상적으로-값이-계속-증가하는-경우)
    - [상황2. 카운터가 중간에 리셋(값이 0으로 떨어짐)된 경우](#상황2-카운터가-중간에-리셋값이-0으로-떨어짐된-경우)
    - [요약](#요약)
  - [incrase vs rate](#incrase-vs-rate)
- [Metric Types](#metric-types)
- [How to Develop Prometheus Client](#how-to-develop-prometheus-client)
  - [Simple Instrumentation](#simple-instrumentation)
- [Advanced](#advanced)
  - [How to reload configuration](#how-to-reload-configuration)
  - [Prometheus High Availability](#prometheus-high-availability)
  - [How to delete metrics](#how-to-delete-metrics)
  - [How to drop metrics](#how-to-drop-metrics)
  - [How to relabel](#how-to-relabel)
  - [PromQL for overall metrics](#promql-for-overall-metrics)
  - [PromQL for kubernetes](#promql-for-kubernetes)

----

# Abstract

서버를 모니터링하는 시스템이다. 모니터링을 원하는 서버에 Exporter 를 설치한다. Prometheus 는 여러 Exporter 들에게 접속하여 데이터를 얻어온다. 즉 pulling 한다. 알림을 받을 규칙을 만들어서 Alert Manager 로 보내면 Alert Manager 가 규칙에 따라 알림을 보낸다.

# Architecture

![](https://prometheus.io/assets/architecture.png)

# Materials

* [Prometheus란?](https://medium.com/finda-tech/prometheus%EB%9E%80-cf52c9a8785f)
* [Prometheus 를 알아보자](https://gompangs.tistory.com/entry/Prometheus-%EB%A5%BC-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90)
* [오픈소스 모니터링 시스템 Prometheus #1](https://blog.outsider.ne.kr/1254)
  * [오픈소스 모니터링 시스템 Prometheus #2](https://blog.outsider.ne.kr/1255)
* [How To Install Prometheus on Ubuntu 16.04](https://www.digitalocean.com/community/tutorials/how-to-install-prometheus-on-ubuntu-16-04)
* [Prometheus with Kubernetes](https://www.slideshare.net/jinsumoon33/kubernetes-prometheus-monitoring)

# Install

## Install on osx

```bash
$ brew install prometheus
```

## Deploy with docker on OSX

```bash
$ docker pull prom/prometheus
$ vim ~/tmp/prometheus.yml
$ docker run \
    --rm \
    --name my-prometheus \
    -p 9090:9090 \
    -v ~/tmp/prometheus.yml:/etc/prometheus/prometheus.yml \
    prom/prometheus
```

## Deploy Prometheus, Grafana, NodeExporter with docker-compose

* [Prometheus-Grafana | github](https://github.com/Einsteinish/Docker-Compose-Prometheus-and-Grafana)

```bash
$ cd my/docker/
$ git clone https://github.com/Einsteinish/Docker-Compose-Prometheus-and-Grafana.git
$ cd Docker-Compose-Prometheus-and-Grafana
$ docker-compose up -d
# Open browser http://localhost:3000 with admin / admin

$ docker-compose down
```

## Deploy GitHub Monitoring Stack with docker-compose

* [github-monitoring](https://github.com/vegasbrianc/github-monitoring)
  * [A Prometheus & Grafana docker-compose stack](https://github.com/vegasbrianc/prometheus) 를 이용한 project 이다.
  * 특정 github repo 들을 등록하면 github 통계를 grafana 로 확인할 수 있다.
  * github exporter, prometheus, grafana 로 구성된다.
  
----  

```bash
$ git clone git@github.com:vegasbrianc/github-monitoring.git
$ cd github-monitoring
# Create Private Access Token in GitHub and paste it to docker-compose.yml
$ vim docker-compose.yml
$ docker-compose up -d
# Open browser http://localhost:3000 with admin / foobar
$ docker-compose down 
```

# PromQL

* [Prometheus Query(PromQL) 기본 이해하기](https://devthomas.tistory.com/15)
* [QUERY EXAMPLES](https://prometheus.io/docs/prometheus/latest/querying/examples/)
* [QUERYING PROMETHEUS](https://prometheus.io/docs/prometheus/latest/querying/basics/)
* [Prometheus Blog Series (Part 2): Metric types](https://blog.pvincent.io/2017/12/prometheus-blog-series-part-2-metric-types/)
* [Go gRPC Interceptors for Prometheus monitoring](https://github.com/grpc-ecosystem/go-grpc-prometheus)
* [TRACKING REQUEST DURATION WITH PROMETHEUS](https://povilasv.me/prometheus-tracking-request-duration/)

----

## Data Model

다음은 Prometheus 가 metric 을 춣력하는 형태이다.

```
http_requests_total{method="POST", handler="/hello"} 1037
```

다음과 같은 형식이다.

```
<metric-name>{<label-name>=<label-value>, ...} <metric-value> [<timestamp>]
```

## Time Series

* Time series is a series of time and value. For example, `[[1m,0.1],[2m,0.2],[3m,0.3]]` is time series.
* Sample is one of time series members. For example, `[1m,0.1]` is a sample.

## Data Types

* **Instant vector**
  * a set of time series containing a single sample for each time series, all sharing the same timestamp
  * multiple samples per one time slot.
    ```c
    http_requests_total{method="POST", handler="/messages"} 1037
    http_requests_total{method="GET", handler="/messages"} 500
    ```
* **Range vector**
  * a set of time series containing a range of data points over time for each time series
    ```c
    http_requests_total{method="POST", handler="/messages"}[5m]
    [1037 @1551242271.728, 1038 @1551242331.728, 1040 @1551242391.728]

    http_requests_total{method="GET", handler="/messages"}[5m]
    [500 @1551242484.013, 501 @1551242544.013, 502 @1551242604.013]
    ```
* **Scalar**
  * a simple numeric floating point value
  
* **String (Deprecated)**
  * a simple string value; currently unused

## Examples

* `http_request_total` 
  * return all time series with the metric.

* `prometheus_http_requests_total{handler="/metrics"}`
  * return all time series in graph
  * return the last data in console

* `prometheus_http_requests_total{handler="/metrics"}[5m]`
  * error because the result is range vector in graph
  * return the last 5min data in console

* `http_request_total` 
  * return all time series with the metric

* `http_requests_total{job="apiserver", handler="/api/comments"}`
  * return all time series with the metric and the given job and handler lables

* `http_requests_total{job=~".*server"}`
  * `~` means REGEX
  * `=, !=, =~, !~`
  * use `.+` instead of `.*`

    ```
    {job=~".*"}              # Bad!
    {job=~".+"}              # Good!
    {job=~".*",method="get"} # Good!
    ```
* `http_requests_total{environment=~"staging|testing|development",method!="GET"}`
  * or 

* `{__name__="http_requests_total"}` is same with `http_requests_total`

* `{__name__=~"job:.*"}`
  * return all time series starts with `job:`

* `rate(http_requests_total[5m])`
  * subquery
  * return 5-minute rate of the metric
  * `s, m, h, d, w, y` means each `seconds, minutes, hours, days, weeks, years`

* `max_over_time(deriv(rate(distance_covered_total[5s])[30s:5s])[10m:])`
  * this is an example of nested subquery but too complicated.

* `rate(http_requests_total[5m])`
  * Return the per-second rate for all time series with the http_requests_total metric name, as measured over the last 5 minutes.

* `sum by (job) (rate(http_requests_total[5m]))`
  * get sum of rates grouped by job

* `(instance_memory_limit_bytes - instance_memory_usage_bytes) / 1024 / 1024`
  * get the unused memory in MiB for every instance

* `sum by (app, proc) (instance_memory_limit_bytes - instance_memory_usage_bytes) / 1024 / 1024`
  * get the sum of the unused memory in MiB grouped by app, proc

* `topk(3, sum by (app, proc) (rate(instance_cpu_time_ns[5m])))`
  * top 3 CPU users grouped by app, proc
  * These are datas
    ```
    instance_cpu_time_ns{app="lion", proc="web", rev="34d0f99", env="prod", job="cluster-manager"}
    instance_cpu_time_ns{app="elephant", proc="worker", rev="34d0f99", env="prod", job="cluster-manager"}
    instance_cpu_time_ns{app="turtle", proc="api", rev="4d3a513", env="prod", job="cluster-manager"}
    instance_cpu_time_ns{app="fox", proc="widget", rev="4d3a513", env="prod", job="cluster-manager"}
    ...
    ```

* `http_requests_total offset 5m`
  * returns the value of http_requests_total 5 minutes in the past relative to the current query evaluation time

* `sum(http_requests_total{method="GET"} offset 5m)`
  * subquery sum with offset
  * `sum(http_requests_total{method="GET"}) offset 5m // INVALID.`

* `rate(http_requests_total[5m] offset 1w)`
  * subquery rate with offset

* `count by (app) (instance_cpu_time_ns)`
  * get the count of the running instances grouped by app

## Basics

### Time series Sslectors

#### Instant vector selectors

return instant vector

```bash
http_requests_total
http_requests_total{job="prometheus",group="canary"}
http_requests_total{environment=~"staging|testing|development",method!="GET"}
{job=~".*"}              # Bad!
{job=~".+"}              # Good!
{job=~".*",method="get"} # Good!
{__name__=~"job:.*"}
on{} # Bad!
{__name__="on"} # Good!
```

#### Range Vector Selectors

return range vector

```
http_requests_total{job="prometheus"}[5m]
```

#### Time Duration

```
ms - milliseconds
s - seconds
m - minutes
h - hours
d - days - assuming a day has always 24h
w - weeks - assuming a week has always 7d
y - years - assuming a year has always 365d
```

```
5h
1h30m
5m
10s
```

#### Offset modifier

```c
http_requests_total offset 5m
sum(http_requests_total{method="GET"} offset 5m) // GOOD.
sum(http_requests_total{method="GET"}) offset 5m // INVALID.
rate(http_requests_total[5m] offset 1w)
```

## Operators

### Binary operators

#### Arithmetic binary operators

```
+ (addition)
- (subtraction)
* (multiplication)
/ (division)
% (modulo)
^ (power/exponentiation)
```

#### Comparison binary operators

```
== (equal)
!= (not-equal)
> (greater-than)
< (less-than)
>= (greater-or-equal)
<= (less-or-equal)
```

#### Logical/set binary operators

```
and (intersection)
or (union)
unless (complement)
```

### Vector Matching (join)

> * [Vector matching @ prometheus.io](https://prometheus.io/docs/prometheus/latest/querying/operators/#vector-matching)
> * [Left joins in PromQL](https://www.robustperception.io/left-joins-in-promql)

There are vector matching such as `one to one`. `one to many, many to one`. There is no `many to many`.

**One-to-one vector matches**

`ignoring(code) method:http_requests:rate5m` 는 `method:http_request:rate5m` 에서 `code` label 은 무시하고 `method_code:http_errors:rate5m` 과 label 이 모두 match 되는 것을 의미한다.

```js
Format:

<vector expr> <bin-op> ignoring(<label list>) <vector expr>
<vector expr> <bin-op> on(<label list>) <vector expr>

Example Input:

method_code:http_errors:rate5m{method="get", code="500"}  24
method_code:http_errors:rate5m{method="get", code="404"}  30
method_code:http_errors:rate5m{method="put", code="501"}  3
method_code:http_errors:rate5m{method="post", code="500"} 6
method_code:http_errors:rate5m{method="post", code="404"} 21

method:http_requests:rate5m{method="get"}  600
method:http_requests:rate5m{method="del"}  34
method:http_requests:rate5m{method="post"} 120

Example Query:

method_code:http_errors:rate5m{code="500"} / ignoring(code) method:http_requests:rate5m

Example Output:

{method="get"}  0.04            //  24 / 600
{method="post"} 0.05            //   6 / 120
```

**Many-to-one and one-to-many vector matches**

데이터의 개수 즉 cardinality 가 많은 쪽이 many 이다. `group_left` 는 왼쪽이 cardinality 가 높다는 의미이다. `group_right` 는 오른쪽이 cardinality 가 높다는 의미이다. 이것을 잘못 사용하면 error 가 발생한다???

```
Format:

<vector expr> <bin-op> ignoring(<label list>) group_left(<label list>) <vector expr>
<vector expr> <bin-op> ignoring(<label list>) group_right(<label list>) <vector expr>
<vector expr> <bin-op> on(<label list>) group_left(<label list>) <vector expr>
<vector expr> <bin-op> on(<label list>) group_right(<label list>) <vector expr>

Example Input:

method_code:http_errors:rate5m{method="get", code="500"}  24
method_code:http_errors:rate5m{method="get", code="404"}  30
method_code:http_errors:rate5m{method="put", code="501"}  3
method_code:http_errors:rate5m{method="post", code="500"} 6
method_code:http_errors:rate5m{method="post", code="404"} 21

method:http_requests:rate5m{method="get"}  600
method:http_requests:rate5m{method="del"}  34
method:http_requests:rate5m{method="post"} 120

Example Query:

method_code:http_errors:rate5m / ignoring(code) group_left method:http_requests:rate5m

Example Output:

{method="get", code="500"}  0.04            //  24 / 600
{method="get", code="404"}  0.05            //  30 / 600
{method="post", code="500"} 0.05            //   6 / 120
{method="post", code="404"} 0.175           //  21 / 120
```

### Aggregation operators

It is same with group by of SQL.

* sum (calculate sum over dimensions)
* min (select minimum over dimensions)
* max (select maximum over dimensions)
* avg (calculate the average over dimensions)
* stddev (calculate population standard deviation over dimensions)
* stdvar (calculate population standard variance over dimensions)
* count (count number of elements in the vector)
* count_values (count number of elements with the same value)
* bottomk (smallest k elements by sample value)
* topk (largest k elements by sample value)
* quantile (calculate φ-quantile (0 ≤ φ ≤ 1) over dimensions)

### Binary operator precedence

```
^
*, /, %
+, -
==, !=, <=, <, >=, >
and, unless
or
```

## Functions

* `abs()`
* `rate()`

## HTTP API

### Expression queries

#### Instance queries

```
GET /api/v1/query
POST /api/v1/query
```

`resultType` is `vector`.

```console
$ curl 'http://localhost:9090/api/v1/query?query=up&time=2015-07-01T20:10:51.781Z'
{
   "status" : "success",
   "data" : {
      "resultType" : "vector",
      "result" : [
         {
            "metric" : {
               "__name__" : "up",
               "job" : "prometheus",
               "instance" : "localhost:9090"
            },
            "value": [ 1435781451.781, "1" ]
         },
         {
            "metric" : {
               "__name__" : "up",
               "job" : "node",
               "instance" : "localhost:9100"
            },
            "value" : [ 1435781451.781, "0" ]
         }
      ]
   }
}
```

#### Range queries

```
GET /api/v1/query_range
POST /api/v1/query_range
```

`resultType` is `matrix`.

```
$ curl 'http://localhost:9090/api/v1/query_range?query=up&start=2015-07-01T20:10:30.781Z&end=2015-07-01T20:11:00.781Z&step=15s'
{
   "status" : "success",
   "data" : {
      "resultType" : "matrix",
      "result" : [
         {
            "metric" : {
               "__name__" : "up",
               "job" : "prometheus",
               "instance" : "localhost:9090"
            },
            "values" : [
               [ 1435781430.781, "1" ],
               [ 1435781445.781, "1" ],
               [ 1435781460.781, "1" ]
            ]
         },
         {
            "metric" : {
               "__name__" : "up",
               "job" : "node",
               "instance" : "localhost:9091"
            },
            "values" : [
               [ 1435781430.781, "0" ],
               [ 1435781445.781, "0" ],
               [ 1435781460.781, "1" ]
            ]
         }
      ]
   }
}
```

### prometheus http request duration seconds

* data

  ```json
  # HELP prometheus_http_request_duration_seconds Histogram of latencies for HTTP requests.
  # TYPE prometheus_http_request_duration_seconds histogram
  prometheus_http_request_duration_seconds_bucket{handler="/",le="0.1"} 25547
  prometheus_http_request_duration_seconds_bucket{handler="/",le="0.2"} 26688
  prometheus_http_request_duration_seconds_bucket{handler="/",le="0.4"} 27760
  prometheus_http_request_duration_seconds_bucket{handler="/",le="1"} 28641
  prometheus_http_request_duration_seconds_bucket{handler="/",le="3"} 28782
  prometheus_http_request_duration_seconds_bucket{handler="/",le="8"} 28844
  prometheus_http_request_duration_seconds_bucket{handler="/",le="20"} 28855
  prometheus_http_request_duration_seconds_bucket{handler="/",le="60"} 28860
  prometheus_http_request_duration_seconds_bucket{handler="/",le="120"} 28860
  prometheus_http_request_duration_seconds_bucket{handler="/",le="+Inf"} 28860
  prometheus_http_request_duration_seconds_sum{handler="/"} 1863.80491025699
  prometheus_http_request_duration_seconds_count{handler="/"} 28860
  ```

* the number of observations per second over the last five minutes on average
  * `rate(prometheus_http_request_duration_seconds_sum[5m])`
* how long they took per second on average
  * `rate(prometheus_http_request_duration_seconds_count[5m])`
* the average duration of one observation
  * `rate(prometheus_http_request_duration_seconds_sum[5m] / rate(prometheus_http_request_duration_seconds_count[5m])`
* 0.9 quantile (the 90th percentile) of seconds
  * `histogram_quantile(0.9, rate(prometheus_http_request_duration_seconds_bucket[5m]))`

### grpc server handling seconds

* data

  ```json
  grpc_server_handling_seconds_bucket{grpc_code="OK",grpc_method="PingList",grpc_service="mwitkow.testproto.TestService",grpc_type="server_stream",le="0.005"} 1
  grpc_server_handling_seconds_bucket{grpc_code="OK",grpc_method="PingList",grpc_service="mwitkow.testproto.TestService",grpc_type="server_stream",le="0.01"} 1
  grpc_server_handling_seconds_bucket{grpc_code="OK",grpc_method="PingList",grpc_service="mwitkow.testproto.TestService",grpc_type="server_stream",le="0.025"} 1
  grpc_server_handling_seconds_bucket{grpc_code="OK",grpc_method="PingList",grpc_service="mwitkow.testproto.TestService",grpc_type="server_stream",le="0.05"} 1
  grpc_server_handling_seconds_bucket{grpc_code="OK",grpc_method="PingList",grpc_service="mwitkow.testproto.TestService",grpc_type="server_stream",le="0.1"} 1
  grpc_server_handling_seconds_bucket{grpc_code="OK",grpc_method="PingList",grpc_service="mwitkow.testproto.TestService",grpc_type="server_stream",le="0.25"} 1
  grpc_server_handling_seconds_bucket{grpc_code="OK",grpc_method="PingList",grpc_service="mwitkow.testproto.TestService",grpc_type="server_stream",le="0.5"} 1
  grpc_server_handling_seconds_bucket{grpc_code="OK",grpc_method="PingList",grpc_service="mwitkow.testproto.TestService",grpc_type="server_stream",le="1"} 1
  grpc_server_handling_seconds_bucket{grpc_code="OK",grpc_method="PingList",grpc_service="mwitkow.testproto.TestService",grpc_type="server_stream",le="2.5"} 1
  grpc_server_handling_seconds_bucket{grpc_code="OK",grpc_method="PingList",grpc_service="mwitkow.testproto.TestService",grpc_type="server_stream",le="5"} 1
  grpc_server_handling_seconds_bucket{grpc_code="OK",grpc_method="PingList",grpc_service="mwitkow.testproto.TestService",grpc_type="server_stream",le="10"} 1
  grpc_server_handling_seconds_bucket{grpc_code="OK",grpc_method="PingList",grpc_service="mwitkow.testproto.TestService",grpc_type="server_stream",le="+Inf"} 1
  grpc_server_handling_seconds_sum{grpc_code="OK",grpc_method="PingList",grpc_service="mwitkow.testproto.TestService",grpc_type="server_stream"} 0.0003866430000000001
  grpc_server_handling_seconds_count{grpc_code="OK",grpc_method="PingList",grpc_service="mwitkow.testproto.TestService",grpc_type="server_stream"} 1
  ```

* request inbound rate
  * `sum(rate(grpc_server_started_total{job="foo"}[1m])) by (grpc_service)`
* unary request error rate
  * `sum(rate(grpc_server_handled_total{job="foo",grpc_type="unary",grpc_code!="OK"}[1m])) by (grpc_service)`
* unary request error percentage
  ```json
  sum(rate(grpc_server_handled_total{job="foo",grpc_type="unary",grpc_code!="OK"}[1m])) by (grpc_service)
  / 
  sum(rate(grpc_server_started_total{job="foo",grpc_type="unary"}[1m])) by (grpc_service)
  * 100.0
  ```
* average response stream size
  ```json
  sum(rate(grpc_server_msg_sent_total{job="foo",grpc_type="server_stream"}[10m])) by (grpc_service)
  /
  sum(rate(grpc_server_started_total{job="foo",grpc_type="server_stream"}[10m])) by (grpc_service)
  ``` 
* 99%-tile latency of unary requests
  * `histogram_quantile(0.99, sum(rate(grpc_server_handling_seconds_bucket{job="foo",grpc_type="unary"}[5m])) by (grpc_service,le))`
* percentage of slow unary queries (>250ms)
  * `100.0 - (sum(rate(grpc_server_handling_seconds_bucket{job="foo",grpc_type="unary",le="0.25"}[5m])) by (grpc_service) / sum(rate(grpc_server_handling_seconds_count{job="foo",grpc_type="unary"}[5m])) by (grpc_service)) * 100.0`

## Summary vs Histogram

## 1. 메트릭 타입 개념

### **Histogram**
- 값의 분포(분포 구간별 개수)를 여러 **버킷(bucket)**에 나눠 기록
- sum, count, 각 버킷별 누적 개수를 기록
- **쿼리 시점에 원하는 분위수(p90, p99 등)를 자유롭게 계산**할 수 있음

### **Summary**
- 애플리케이션이 직접 분위수(p90, p99 등)를 계산해서 기록
- sum, count, 그리고 미리 정한 분위수 값만 기록
- **추가적인 분위수는 쿼리에서 뽑을 수 없음**

---

## 2. Go 코드 예시 (구조체)

### **Histogram 구조체 예시**
```go
type Histogram struct {
    Buckets map[float64]uint64 // 버킷 경계값(le)별 누적 카운트
    Count   uint64             // 전체 샘플 개수
    Sum     float64            // 전체 값의 합
}
```
#### 예시 데이터 (5개의 버킷)
```go
hist := Histogram{
    Buckets: map[float64]uint64{
        0.1:  3,  // 0.1초 이하 값이 3개
        0.5:  7,  // 0.5초 이하 값이 7개
        1.0:  10, // 1초 이하 값이 10개
        2.0:  12, // 2초 이하 값이 12개
        +Inf: 15, // 전체 값 15개
    },
    Count: 15,
    Sum:   8.3,
}
```
- Prometheus는 **각 버킷별 시계열**과 sum, count를 저장합니다.
- 쿼리에서는 이런 데이터를 조합해 **원하는 분위수(예: p95, p99 등)를 자유롭게 계산**할 수 있습니다.

---

### **Summary 구조체 예시**
```go
type Summary struct {
    Count     uint64              // 전체 샘플 개수
    Sum       float64             // 전체 값의 합
    Quantiles map[float64]float64 // 미리 정한 분위수 값(예: 0.9, 0.99 등)
}
```
#### 예시 데이터 (p90, p99만 기록)
```go
summary := Summary{
    Count: 15,
    Sum:   8.3,
    Quantiles: map[float64]float64{
        0.90: 0.85,  // 90% 분위수 값
        0.99: 1.8,   // 99% 분위수 값
    },
}
```
- Prometheus는 **sum, count, quantile별 시계열**을 저장합니다.
- **애플리케이션에서 미리 정한 분위수만 기록**하므로,  
  쿼리에서 새로운 분위수를 계산할 수 없습니다.

---

## 3. Prometheus에 저장되는 시계열 차이

### Histogram
```text
redis_query_seconds_bucket{le="0.1"}  3
redis_query_seconds_bucket{le="0.5"}  7
redis_query_seconds_bucket{le="1.0"}  10
redis_query_seconds_bucket{le="2.0"}  12
redis_query_seconds_bucket{le="+Inf"} 15
redis_query_seconds_sum               8.3
redis_query_seconds_count             15
```
- **여러 버킷** + sum + count → 다양한 분위수 계산 가능

### Summary
```text
redis_query_seconds{quantile="0.9"}  0.85
redis_query_seconds{quantile="0.99"} 1.8
redis_query_seconds_sum              8.3
redis_query_seconds_count            15
```
- **미리 정한 분위수만** 기록 → 쿼리에서 추가 분위수 불가

---

## 4. 쿼리 측면의 차이

- **Histogram:**  
  `histogram_quantile(0.95, sum(rate(redis_query_seconds_bucket[5m])) by (le))` 처럼  
  원하는 분위수를 쿼리에서 지정해 계산 가능

- **Summary:**  
  `redis_query_seconds{quantile="0.99"}`  
  → 미리 정한 분위수 값만 조회 가능

---

## 5. 요약

| 구분        | Histogram                        | Summary                               |
|-------------|----------------------------------|---------------------------------------|
| 저장 데이터 | 여러 버킷, sum, count             | quantile(미리 정함), sum, count       |
| 분위수      | 쿼리에서 자유롭게 계산 가능       | 미리 정한 값만 사용 가능              |
| 집계        | 레이블 집계(aggregate) 가능      | 집계 불가(합산하면 분위수 의미 없음)  |

## irate vs rate

## 1. irate(redis_operation_seconds_bucket[5m])
- **의미:**  
  지정된 5분 구간 안에서 **가장 최근 두 개의 데이터 포인트**만 사용해서 변화율을 계산합니다.
- **계산 방식:**  
  ```
  (가장 최근 값 - 그 전 값) / (두 값의 시간 차이)
  ```
  즉, 거의 "순간 변화율(instantaneous rate)"을 구합니다.
- **특징:**  
  - 최근 값 두 개로만 계산하므로, 데이터가 자주 저장된다면 거의 실시간 변화량을 반영합니다.
  - 노이즈가 심할 수 있음(최근 두 점의 변화만 반영).

---

## 2. rate(redis_operation_seconds_bucket[5m])
- **의미:**  
  지정된 5분 구간 안에 **포함된 모든 데이터 포인트**를 이용해서 평균 변화율을 계산합니다.
- **계산 방식:**  
  ```
  (5분 구간 내의 마지막 값 - 처음 값) / (두 값의 시간 차이)
  ```
  즉, 5분 동안의 "평균 변화율"을 구합니다.
- **특징:**  
  - 구간 전체의 변화를 평균내므로 값이 더 부드럽고 안정적임.
  - 갑작스런 변화(스파이크)는 완만하게 반영됨.

---

## 3. 예시 (15초마다 데이터가 저장되는 상황)
- 5분 동안 약 20개 데이터 포인트가 있음.
- **irate**는 그 중 마지막 2개만 사용 → 최신 변화만 반영, 노이즈에 민감.
- **rate**는 5분 전체를 사용 → 전체 평균, 값이 부드럽고 트렌드 파악에 유리.

---

### **정리**
| 함수   | 사용 데이터 포인트            | 결과 성격      | 용도                        |
|--------|------------------------------|----------------|-----------------------------|
| irate  | 최근 2개                     | 순간 변화율    | 실시간 변화 감지, 알람 등   |
| rate   | 5분 내 모든 데이터           | 평균 변화율    | 트렌드, 그래프, 리포트 등   |

## rate deep dive

"구간 내 모든 데이터 포인트를 활용한다"는 말은, Prometheus의 `rate()` 함수가 단순히 처음과 끝 값만 사용하는 것이 아니라,  
**구간 내에 있는 모든 값(데이터 포인트)**을 보고 **카운터의 리셋(값이 갑자기 0으로 떨어지는 경우)** 등 특별한 상황을 처리한다는 뜻입니다.

---

## 예시로 설명

### 상황1. 정상적으로 값이 계속 증가하는 경우

구간: `[5m]`  
샘플 데이터(시간, 값):

| 시간      | 값  |
|-----------|-----|
| 10:00:00  | 100 |
| 10:01:00  | 120 |
| 10:02:00  | 140 |
| 10:03:00  | 160 |
| 10:04:00  | 180 |

- **rate(metric[5m])**을 계산하면:
  ```
  (180 - 100) / (10:04:00 - 10:00:00)
  = 80 / 240초
  = 0.333.../sec
  ```
- 여기서, "처음과 끝 값"만 써도 결과는 맞음.

---

### 상황2. 카운터가 중간에 리셋(값이 0으로 떨어짐)된 경우

| 시간      | 값  |
|-----------|-----|
| 10:00:00  | 100 |
| 10:01:00  | 120 |
| 10:02:00  | 10  |  ← 리셋!
| 10:03:00  | 30  |
| 10:04:00  | 50  |

- 단순히 `50 - 100`으로 계산하면 -50이 되어버림. (잘못된 값!)
- **Prometheus의 rate()는**  
  1. 중간에 값이 갑자기 줄어드는(리셋되는) 경우를 감지  
  2. **리셋 전까지 증가분:** 120 - 100 = 20  
  3. **리셋 후 증가분:** 50 - 10 = 40  
  4. **총 증가량:** 20 + 40 = 60  
  5. **총 시간:** 10:04:00 - 10:00:00 = 240초  
  6. **rate:** 60 / 240 = 0.25/sec

**이렇게 "구간 내 모든 데이터 포인트"를 살펴야 리셋/이상치가 있어도 정확한 변화량을 구할 수 있습니다.**

---

### 요약
- 값이 정상적으로 증가만 하면 처음과 끝 값만 써도 문제 없음.
- **하지만 중간에 리셋, 이상치가 있으면 모든 데이터 포인트를 다 살펴야 한다.**
- Prometheus의 `rate()` 함수는 이런 상황을 자동으로 처리해줍니다.

## incrase vs rate

* [프로메테우스 지표 rate와 increase의 차이점](https://blog.voidmainvoid.net/449)

`increase()` 는 일정시간동안 늘어난 count.

`rate()` 는 초당 늘어난 count.

# Metric Types

* [The 4 Types Of Prometheus Metrics](https://tomgregory.com/the-four-types-of-prometheus-metrics/)

-----

* Counter
  * 오로지 늘어나기만 하는 수치
  * `rate(request_count[5m])` 와 같이 `rate()` 을 사용할 수 있다.
* Gauge
  * 늘어나거나 줄어드는 수치
  * `avg_over_time(queue_size[5m])` 와 같이 `avg_over_time()` 을 사용할 수 있다.
* Histogram
  * Server 에서 Quantile 별 집계를 수행함.
  * 다음은 `4.467s, 9.213s, and 9.298s` 의 예이다.
    ```
    # HELP request_duration Time for HTTP request.
    # TYPE request_duration histogram
    request_duration_bucket{le="0.005",} 0.0
    request_duration_bucket{le="0.01",} 0.0
    request_duration_bucket{le="0.025",} 0.0
    request_duration_bucket{le="0.05",} 0.0
    request_duration_bucket{le="0.075",} 0.0
    request_duration_bucket{le="0.1",} 0.0
    request_duration_bucket{le="0.25",} 0.0
    request_duration_bucket{le="0.5",} 0.0
    request_duration_bucket{le="0.75",} 0.0
    request_duration_bucket{le="1.0",} 0.0
    request_duration_bucket{le="2.5",} 0.0
    request_duration_bucket{le="5.0",} 1.0
    request_duration_bucket{le="7.5",} 1.0
    request_duration_bucket{le="10.0",} 3.0
    request_duration_bucket{le="+Inf",} 3.0
    request_duration_count 3.0
    request_duration_sum 22.978489699999997
    ```

  * 다음과 같이 query 한다.
    ```
    rate(request_duration_sum[5m])
    /
    rate(request_duration_count[5m])  
    ```

  * `histogram_quantile(0.95, sum(rate(request_duration_bucket[5m])) by (le))` 와 같이 `historgram_quantile()` 을 사용가능하다.

* Summary
  * Client 에서 Quantile 별 집계를 수행함. 미리 정해진 Quantile 만 사용할 수 있음. `histogram_quantile()` 을 사용할 수 없다.
    ```
    # HELP request_duration_summary Time for HTTP request.
    # TYPE request_duration_summary summary
    request_duration_summary{quantile="0.95",} 7.4632192
    request_duration_summary_count 5.0
    request_duration_summary_sum 27.338737899999998
    ```
  * 다음과 같이 query 한다.
  
    ```
    rate(request_duration_summary_sum[5m])
    /
    rate(request_duration_summary_count[5m])
    ```
    
**Metric Type Comparison Table**

|	 |Counter |	Gauge |	Histogram |	Summary |
|--|--|--|--|--|
| **General**				| | | |  |
| Can go up and down |	✗ |	✓ |	✓ |	✓ |
| Is a complex type (publishes multiple values per metric)	| ✗ |	✗ |	✓ |	✓ |
| Is an approximation	| ✗ |	✗ |	✓ |	✓ |
| **Querying**		| | | |	|
| Can query with rate function	| ✓	| ✗	| ✗	| ✗ |
| Can calculate percentiles	| ✗	| ✗	| ✓	| ✓ |
| Can query with histogram_quantile function	| ✗	| ✗	| ✓	| ✗ |

# How to Develop Prometheus Client

## Simple Instrumentation

* [INSTRUMENTING A GO APPLICATION FOR PROMETHEUS](https://prometheus.io/docs/guides/go-application/)

----

* Application Server
  * main.go

    ```go
    package main

    import (
      "math/rand"
      "net/http"
      "time"

      "github.com/prometheus/client_golang/prometheus"
      "github.com/prometheus/client_golang/prometheus/promauto"
      "github.com/prometheus/client_golang/prometheus/promhttp"
    )

    func recordMetrics() {
      go func() {
        for {
          opsProcessed.Inc()

          if n := rand.Intn(100); n%2 == 0 {
            httpReqs.WithLabelValues("200 OK", "GET").Inc()
          } else {
            httpReqs.WithLabelValues("200 OK", "POST").Inc()
          }

          apiLatency.WithLabelValues("myapp").Observe(float64(rand.Intn(100)))

          time.Sleep(2 * time.Second)
        }
      }()
    }

    var (
      opsProcessed = promauto.NewCounter(prometheus.CounterOpts{
        Name: "myapp_processed_ops_total",
        Help: "The total number of processed events",
      })

      httpReqs = prometheus.NewCounterVec(
        prometheus.CounterOpts{
          Name: "myapp_http_requests_total",
          Help: "How many HTTP requests processed, partitioned by status code and HTTP method.",
        },
        []string{"code", "method"},
      )

      apiLatency = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
          Name:    "myapp_api_latency",
          Help:    "api latency",
          Buckets: prometheus.LinearBuckets(0, 5, 20),
        },
        []string{"target_group"},
      )
    )

    func init() {
      //prometheus.MustRegister(opsProcessed)
      prometheus.MustRegister(httpReqs)
      prometheus.MustRegister(apiLatency)
    }

    func main() {
      recordMetrics()

      http.Handle("/metrics", promhttp.Handler())
      http.ListenAndServe(":2112", nil)
    }
    ```
  * Run application server

    ```bash
    $ go mod init main
    $ go build
    $ go run main.go
    $ curl http://localhost:2112/metrics
    ```

* Prometheus
  * prometheus.yml

    ```yml
    # my global config
    global:
      scrape_interval:     15s # Set the scrape interval to every 15 seconds. Default is every 1 minute.
      evaluation_interval: 15s # Evaluate rules every 15 seconds. The default is every 1 minute.
      # scrape_timeout is set to the global default (10s).

    # Alertmanager configuration
    alerting:
      alertmanagers:
      - static_configs:
        - targets:
          # - alertmanager:9093

    # Load rules once and periodically evaluate them according to the global 'evaluation_interval'.
    rule_files:
      # - "first_rules.yml"
      # - "second_rules.yml"

    # A scrape configuration containing exactly one endpoint to scrape:
    # Here it's Prometheus itself.
    scrape_configs:
      # The job name is added as a label `job=<job_name>` to any timeseries scraped from this config.
      - job_name: 'prometheus'
        # metrics_path defaults to '/metrics'
        # scheme defaults to 'http'.
        static_configs:
        - targets: ['localhost:9090']
      - job_name: myapp
        scrape_interval: 10s
        static_configs:
        - targets:
          - host.docker.internal:2112
    ```
  * Run Prometheus

    ```bash
    $ docker pull prom/prometheus
    $ vim ~/tmp/prometheus.yml
    $ docker run \
        --rm -d \
        -p 9090:9090 \
        -v ~/tmp/prometheus.yml:/etc/prometheus/prometheus.yml \
        --name my-prometheus \
        prom/prometheus
    ```
  * Open browser
    * status page: `localhost:9090`
    * metrics endpoint: `localhost:9090/metrics`

# Advanced

## How to reload configuration

* [Reloading Prometheus’ Configuration](https://www.robustperception.io/reloading-prometheus-configuration)

----

```bash
$ kill -HUP 1234
```

## Prometheus High Availability

* [Prometheus High Availability](https://ssup2.github.io/theory_analysis/Prometheus_High_Availability/)

----

multiple prometheus and ALB with sticky session

## How to delete metrics

* [프로메테우스 시계열 데이터 삭제하기](https://kangwoo.github.io/devops/prometheus/delete-metrics-from-prometheus/)
  
-----

You should turn on the argument `--web.enable-admin-api`.

```bash
# Delete metrics with specific labels
curl -X POST \
	-g 'http://localhost:9090/api/v1/admin/tsdb/delete_series?match[]={foo="bar"}'

# Delete metrics with specific labels
curl -X POST \
	-g 'http://localhost:9090/api/v1/admin/tsdb/delete_series?match[]={job="node_exporter"}'

# Delete metrics with specific labels
curl -X POST \
	-g 'http://localhost:9090/api/v1/admin/tsdb/delete_series?match[]={instance="172.22.0.1:9100"}'

# Delete all metrics 
curl -X POST \
	-g 'http://localhost:9090/api/v1/admin/tsdb/delete_series?match[]={__name__=~".+"}'

# Let's do compaction right now
curl -X POST -g 'http://localhost:9090/api/v1/admin/tsdb/clean_tombstones'
```

## How to drop metrics

* [Dropping metrics at scrape time with Prometheus](https://www.robustperception.io/dropping-metrics-at-scrape-time-with-prometheus)

-----

Check top 10 metrics

```yaml
topk(20, count by (__name__, job)({__name__=~".+"}))
```

**metric_relabel_configs** 를 이용하면 metric data 를 drop 할 수 있다. **relabel_configs** 과 다름을 유의하자.

다음의 설정은 metric data 의 이름이 `go_memstat_(.*)` 혹은 `prometheus_engine_(.*)` 인 것을 drop 하고 나머지는 keep 한다.

```yaml
scrape_configs:
  - job_name: 'prometheus'
    scrape_interval: 5s
    static_configs:
      - targets: ['localhost:9090']
    metric_relabel_configs:
    - source_labels: [__name__]
      regex: 'go_memstats_(.*)|prometheus_engine_(.*)'
      action: drop
```

## How to relabel

* [170. [Prometheus] 2편. PromQL 사용하기, k8s에서 Meta Label과 relabel을 활용한 기본 라벨 설정, 그 외의 이야기 @ naverblog](https://blog.naver.com/PostView.nhn?blogId=alice_k106&logNo=221535575875)
* [relabel_config @ prometheus](https://prometheus.io/docs/prometheus/latest/configuration/configuration/#relabel_config)
* [Prometheus Relabel Rules and the ‘action’ Parameter](https://blog.freshtracks.io/prometheus-relabel-rules-and-the-action-parameter-39c71959354a)
* [Life of a Label @ robustperception](https://www.robustperception.io/life-of-a-label)
  * [PromCon 2016: Life of a Label - Brian Brazil @ youtube](https://www.youtube.com/watch?v=b5-SvvZ7AwI)

----

metric data 의 label 을 교체 및 추가하는 것을 relabel 이라고 한다. relabel 의 조건이 맞으면 action 을 수행할 수 있다. relabel_configs 의 항목은 순서대로 처리되는 것 같다???

많이 사용하는 action 은 **keep**, **drop**, **replace** 와 **labelmap** 이다. **replace** 는 교체하는 것이고 **labelmap** 은 새로운 label 을 만들어서 값을 복사하는 것이다. **labelmap** 은 항상 처음에 와야 한다???

Prometheus 는 metric data 의 특정 label 을 보고 조건이 맞으면 metric data 를 **keep** 할 수도 있고 **drop** 할 수도 있다. 

다음은 `must` 라는 label 의 값이 `foo` 인 metric data 를 keep 하고 나머지는 drop 하라는 의미이다.

```yml
scrape_configs:
  - job_name: 'prometheus'
    scrape_interval: 5s
    static_configs:
      - targets: ['localhost:9090']
    relabel_configs:
    - source_labels: [must]
      regex: foo
      action: keep
```

다음은 `__address__` 라는 label 이 있는 metric data 를 keep 하고 나머지는 drop 하라는 의미이다.

```yml
scrape_configs:
  - job_name: 'prometheus'
    scrape_interval: 5s
    static_configs:
      - targets: ['localhost:9090']
    relabel_configs:
    - source_labels: [__address__]
      regex: (.*)
      action: keep
```

다음은 `__address__` 라는 label 이 있는 metric data 를 drop 하고 나머지는 keep 하라는 의미이다.

```yml
scrape_configs:
  - job_name: 'prometheus'
    scrape_interval: 5s
    static_configs:
      - targets: ['localhost:9090']
    relabel_configs:
    - source_labels: [__address__]
      regex: (.*)
      action: drop
```

기본적으로 `__address__, job` 와 같은 label 은 `instance, job` 으로 relabel 된다. `__` 로 시작하는 label 은 meta label 이라고 한다.

[relabel_config @ prometheus](https://prometheus.io/docs/prometheus/latest/configuration/configuration/#relabel_config) 를 참고하면 다양한 action 들을 확인할 수 있다.

`__` 로 시작하는 label 을 meta label 이라고 한다. meta label 들은 relabel process 후 제거된다.

다음과 같이 `prometheus.yml` 을 만들어서 docker 를 이용하여 실험할 수 있다.

```yml
global:
  scrape_interval:     15s # By default, scrape targets every 15 seconds.
  external_labels:
    monitor: 'codelab-monitor'

scrape_configs:
  - job_name: 'prometheus'
    scrape_interval: 5s
    static_configs:
      - targets: ['localhost:9090']

    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__scheme__]
        action: replace
        target_label: hello
      - action: labelmap
        regex: __scheme__
        replacement: world      
```

![](prometheus_target.png)

## PromQL for overall metrics

```bash
# Count of timeseries
count({__name__=~".+"}) 

# Count of timeseries by job
count({__name__=~".+"}) by (job)

{job="cluster-autoscaler"}	5
{job="kubernetes-apiservers"}	21797
{job="kubernetes-nodes-kubelet"}	1413
{job="kubernetes-nodes-cadvisor"}	2897
{job="kubernetes-service-endpoints"}	1082

# List of unique metrics by job, name
count({__name__=~".+", job="cluster-autoscaler"}) by (__name__)
count({__name__=~".+", job="kubernetes-apiservers"}) by (__name__)
count({__name__=~".+", job="kubernetes-nodes-kubelet"}) by (__name__)
count({__name__=~".+", job="kubernetes-nodes-cadvisor"}) by (__name__)
count({__name__=~".+", job="kubernetes-service-endpoints", kubernetes_name="kube-state-metrics"}) by (__name__)

# Count of unique metrics by job, name
count(count({__name__=~".+", job="cluster-autoscaler"}) by (__name__))
```

## PromQL for kubernetes

> * CPU usage milicore

```c
// container 가 사용한 cpu 량을 milicore 로 보여다오.
// 소수점 단위이다. (ex, 0.001)
// 1 은 1000 milicore 를 의미한다.
container_cpu_usage_seconds_total
// 최근 2 분동안 container_cpu_usage_seconds_total 을 
// 모아서 초당 증감을 보여다오.
rate(container_cpu_usage_seconds_total[2m]
// 값이 너무 작으니 적당히 100 을 곱하자.
// 모든 팟 마다 rate(container_cpu_usage_seconds_total[2m]) * 100 의
// 평균을 보여다오. 평균은 하나의 pod 에 포함된 container 들의 
// rate(container_cpu_usage_seconds_total[2m]) * 100 를 모두 더하고
// container 의 개수로 나눈 값이다.
// 이 값이 90 을 넘으면 pod 이 cpu 를 많이 점유하고 있구나라고
// 생각할 수 있다.
avg(rate(container_cpu_usage_seconds_total[2m]) * 100) by (pod)
```
