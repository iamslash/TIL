
- [Abstract](#abstract)
- [Architecture](#architecture)
- [Materials](#materials)
- [Install](#install)
  - [Install on osx](#install-on-osx)
  - [Install with docker](#install-with-docker)
  - [Install with docker-compose](#install-with-docker-compose)
- [Basic usages](#basic-usages)
- [PromQL](#promql)
  - [Histogram](#histogram)
    - [prometheus http request duration seconds](#prometheus-http-request-duration-seconds)
    - [grpc server handling seconds](#grpc-server-handling-seconds)
  - [Summary](#summary)
  - [Summary vs Histogram](#summary-vs-histogram)
- [Client](#client)
  - [Simple Instrumentation](#simple-instrumentation)
  - [Metric types](#metric-types)

----

# Abstract

서버를 모니터링하는 시스템이다. 모니터링을 원하는 서버에 Exporter 를 설치한다. Prometheus 는 여러 Exporter 들에게 접속하여 데이터를 얻어온다. 즉 pulling 한다. 알림을 받을 규칙을 만들어서 Alert Manager 로 보내면 Alert Manager 가 규칙에 따라 알림을 보낸다.

# Architecture

![](https://prometheus.io/assets/architecture.png)

# Materials

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

## Install with docker

```bash
$ docker pull prom/prometheus
$ vim ~/tmp/prometheus.yml
$ docker run \
    -p 9090:9090 \
    -v ~/tmp/prometheus.yml:/etc/prometheus/prometheus.yml \
    prom/prometheus
```

## Install with docker-compose

* [A Prometheus & Grafana docker-compose stack](https://github.com/vegasbrianc/prometheus)
  * Prometheus, Grafana

----

```bash
$ cd my/docker/
$ git clone git@github.com:vegasbrianc/prometheus.git
$ HOSTNAME=$(hostname) docker stack deploy -c docker-stack.yml prom
# open Grafana dashboard http://192.168.10.1:3000
# admin / foobar (/grafana/config.monitoring)
$ docker stack ps prom
$ docker service ls
$ docker service logs prom_<service_name>
```

# Basic usages

- run prometheus

  - ```$ prometheus --config.file=prometheus.yml```
  - prometheus.yml

    ```yml
    global:
      scrape_interval:     15s # By default, scrape targets every 15 seconds.

      # Attach these labels to any time series or alerts when communicating with
      # external systems (federation, remote storage, Alertmanager).
      external_labels:
        monitor: 'codelab-monitor'

    # A scrape configuration containing exactly one endpoint to scrape:
    # Here it's Prometheus itself.
    scrape_configs:
      # The job name is added as a label `job=<job_name>` to any timeseries scraped from this config.
      - job_name: 'prometheus'

        # Override the global default and scrape targets from this job every 5 seconds.
        scrape_interval: 5s

        static_configs:
          - targets: ['localhost:9090']
      - job_name: "node"
        static_configs:
          - targets:
              - "localhost:9100"
            labels:
              resin_app: RESIN_APP_ID
              resin_device_uuid: RESIN_DEVICE_UUID
    ```

- exporter
  - 특정 machine 에서 자료를 수집한다. URL 을 통해 prometheus 가 pulling 한다.
  - [node_exporter](https://github.com/prometheus/node_exporter) 는
    linux 의 cpu 등등의 정보를 수집한다. 실행후
    `http://<your-device-ip>:9100/metrics` 를 브라우저로 접속한다.

- scraping
  - config file 의 scrape_configs 를 설정하여 prometheus 가 targeting 할 수 있도록 하자.
  - prometheus.yml
  
    ```yaml
    ...
    scrape_configs:  
      - job_name: "node"
        static_configs:
        - targets:
            - "localhost:9100"
          labels:
            resin_app: RESIN_APP_ID
            resin_device_uuid: RESIN_DEVICE_UUID
    ...
    ```
  
- alert
  - alertmanager 를 이용하여 관리자에게 email 등등의 알람을 전송 할 수 있다.
  
  - prometheus rulefile, a.rules

    ```
    ALERT cpu_threshold_exceeded  
      IF (100 * (1 - avg by(job)(irate(node_cpu{mode='idle'}[5m])))) > THRESHOLD_CPU
      ANNOTATIONS {
        summary = "Instance {{ $labels.instance }} CPU usage is dangerously high",
        description = "This device's CPU usage has exceeded the threshold with a value of {{ $value }}.",
      }
    ```
  
  - alertmanager configfile, a.yml
    
    ```yml
    route:  
      group_by: [Alertname]
      # Send all notifications to me.
      receiver: email-me
      # When a new group of alerts is created by an incoming alert, wait at
      # least 'group_wait' to send the initial notification.
      # This way ensures that you get multiple alerts for the same group that start
      # firing shortly after another are batched together on the first
      # notification.
      group_wait: 30s

      # When the first notification was sent, wait 'group_interval' to send a batch
      # of new alerts that started firing for that group.
      group_interval: 5m

      # If an alert has successfully been sent, wait 'repeat_interval' to
      # resend them.
      repeat_interval: 3h

    templates:  
    - '/etc/ALERTMANAGER_PATH/default.tmpl'

    receivers:  
    - name: email-me
      email_configs:
      - to: GMAIL_ACCOUNT
        from: GMAIL_ACCOUNT
        smarthost: smtp.gmail.com:587
        html: '{{ template "email.default.html" . }}'
        auth_username: "GMAIL_ACCOUNT"
        auth_identity: "GMAIL_ACCOUNT"
        auth_password: "GMAIL_AUTH_TOKEN"
    ```

# PromQL

* [QUERY EXAMPLES](https://prometheus.io/docs/prometheus/latest/querying/examples/)
* [QUERYING PROMETHEUS](https://prometheus.io/docs/prometheus/latest/querying/basics/)
* [Prometheus Blog Series (Part 2): Metric types](https://blog.pvincent.io/2017/12/prometheus-blog-series-part-2-metric-types/)
* [Go gRPC Interceptors for Prometheus monitoring](https://github.com/grpc-ecosystem/go-grpc-prometheus)
* [TRACKING REQUEST DURATION WITH PROMETHEUS](https://povilasv.me/prometheus-tracking-request-duration/)

----

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

* `sum by (job) (
  rate(http_requests_total[5m])
)`
  * get sum of rates grouped by job

* `(instance_memory_limit_bytes - instance_memory_usage_bytes) / 1024 / 1024`
  * get the unused memory in MiB for every instance

* `sum by (app, proc) (
  instance_memory_limit_bytes - instance_memory_usage_bytes
) / 1024 / 1024`
  * get the sum of the unused memory grouped by app, proc

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

## Histogram

* [Prometheus monitoring for your gRPC Go servers.](https://github.com/grpc-ecosystem/go-grpc-prometheus)
* [How does a Prometheus Histogram work?](https://www.robustperception.io/how-does-a-prometheus-histogram-work)

----

Usually measure the latency. Can adjust time period when make the range vector. But Summary can't.

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
  * `rate(prometheus_http_request_duration_seconds_sum[5m]`
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

## Summary

* [How does a Prometheus Summary work?](https://www.robustperception.io/how-does-a-prometheus-summary-work)

----

* data

  ```json
  # HELP prometheus_rule_evaluation_duration_seconds The duration for a rule to execute.
  # TYPE prometheus_rule_evaluation_duration_seconds summary
  prometheus_rule_evaluation_duration_seconds{quantile="0.5"} 6.4853e-05
  prometheus_rule_evaluation_duration_seconds{quantile="0.9"} 0.00010102
  prometheus_rule_evaluation_duration_seconds{quantile="0.99"} 0.000177367
  prometheus_rule_evaluation_duration_seconds_sum 1.623860968846092e+06
  prometheus_rule_evaluation_duration_seconds_count 1.112293682e+09
  ```

* the number of observations per second over the last five minutes on average
  * `rate(prometheus_rule_evaluation_duration_seconds_count[5m])`
* how long they took per second on average  
  * `rate(prometheus_rule_evaluation_duration_seconds_sum[5m])`
* the average duration of one observation  
  * `rate(prometheus_rule_evaluation_duration_seconds_sum[5m] / rate(prometheus_rule_evaluation_duration_seconds_count[5m])`
  
## Summary vs Histogram

* Histogram can adjust time period when make the range vector. But Summary can't.
* Histogram costs more than Summary in server-side.

# Client

## Simple Instrumentation

* [INSTRUMENTING A GO APPLICATION FOR PROMETHEUS](https://prometheus.io/docs/guides/go-application/)

----

* run client
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
  * run client

    ```bash
    $ go mod init main
    $ go build
    $ go run main.go
    $ curl http://localhost:2112/metrics
    ```

* run server
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
  * run server

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
  * open browser
    * status page: `localhost:9090`
    * metrics endpoint: `localhost:9090/metrics`

## Metric types

* [METRIC TYPES](https://prometheus.io/docs/concepts/metric_types/)

----

There 4 kinds of metric types such as Counter, Gauge, Histogram, Summary.

**Counter**

This is a cumulative metric. For example, the number of requests served, tasks completed, http total send bytes, http total request, running time or errors. There is just increment no decrement. 

[Counter doc](https://godoc.org/github.com/prometheus/client_golang/prometheus#Counter)

```go
httpReqs := prometheus.NewCounterVec(
    prometheus.CounterOpts{
        Name: "http_requests_total",
        Help: "How many HTTP requests processed, partitioned by status code and HTTP method.",
    },
    []string{"code", "method"},
)
prometheus.MustRegister(httpReqs)

httpReqs.WithLabelValues("404", "POST").Add(42)

// If you have to access the same set of labels very frequently, it
// might be good to retrieve the metric only once and keep a handle to
// it. But beware of deletion of that metric, see below!
m := httpReqs.WithLabelValues("200", "GET")
for i := 0; i < 1000000; i++ {
    m.Inc()
}
// Delete a metric from the vector. If you have previously kept a handle
// to that metric (as above), future updates via that handle will go
// unseen (even if you re-create a metric with the same label set
// later).
httpReqs.DeleteLabelValues("200", "GET")
// Same thing with the more verbose Labels syntax.
httpReqs.Delete(prometheus.Labels{"method": "GET", "code": "200"})
```

**Gauge**

This is a metric that represents a single numerical value that can arbitrarily go up and down. For example, memory, disk usage, temperature, current cpu usage, current thread count.

[Gauge doc](https://godoc.org/github.com/prometheus/client_golang/prometheus#Gauge)

```go
opsQueued := prometheus.NewGauge(prometheus.GaugeOpts{
    Namespace: "our_company",
    Subsystem: "blob_storage",
    Name:      "ops_queued",
    Help:      "Number of blob storage operations waiting to be processed.",
})
prometheus.MustRegister(opsQueued)

// 10 operations queued by the goroutine managing incoming requests.
opsQueued.Add(10)
// A worker goroutine has picked up a waiting operation.
opsQueued.Dec()
// And once more...
opsQueued.Dec()
```

**Histogram**

A histogram samples observations (usually things like request durations or response sizes) and counts them in configurable buckets. It also provides a sum of all observed values. For example, TPS.

[Histogram doc](https://godoc.org/github.com/prometheus/client_golang/prometheus#Histogram)

```go
temps := prometheus.NewHistogram(prometheus.HistogramOpts{
    Name:    "pond_temperature_celsius",
    Help:    "The temperature of the frog pond.", // Sorry, we can't measure how badly it smells.
    Buckets: prometheus.LinearBuckets(20, 5, 5),  // 5 buckets, each 5 centigrade wide.
})

// Simulate some observations.
for i := 0; i < 1000; i++ {
    temps.Observe(30 + math.Floor(120*math.Sin(float64(i)*0.1))/10)
}

// Just for demonstration, let's check the state of the histogram by
// (ab)using its Write method (which is usually only used by Prometheus
// internally).
metric := &dto.Metric{}
temps.Write(metric)
fmt.Println(proto.MarshalTextString(metric))
```

```
histogram: <
  sample_count: 1000
  sample_sum: 29969.50000000001
  bucket: <
    cumulative_count: 192
    upper_bound: 20
  >
  bucket: <
    cumulative_count: 366
    upper_bound: 25
  >
  bucket: <
    cumulative_count: 501
    upper_bound: 30
  >
  bucket: <
    cumulative_count: 638
    upper_bound: 35
  >
  bucket: <
    cumulative_count: 816
    upper_bound: 40
  >
>
```

**Summary**

[Summary doc](https://godoc.org/github.com/prometheus/client_golang/prometheus#Summary)

```go
temps := prometheus.NewSummary(prometheus.SummaryOpts{
    Name:       "pond_temperature_celsius",
    Help:       "The temperature of the frog pond.",
    Objectives: map[float64]float64{0.5: 0.05, 0.9: 0.01, 0.99: 0.001},
})

// Simulate some observations.
for i := 0; i < 1000; i++ {
    temps.Observe(30 + math.Floor(120*math.Sin(float64(i)*0.1))/10)
}

// Just for demonstration, let's check the state of the summary by
// (ab)using its Write method (which is usually only used by Prometheus
// internally).
metric := &dto.Metric{}
temps.Write(metric)
fmt.Println(proto.MarshalTextString(metric))
```

```
summary: <
  sample_count: 1000
  sample_sum: 29969.50000000001
  quantile: <
    quantile: 0.5
    value: 31.1
  >
  quantile: <
    quantile: 0.9
    value: 41.3
  >
  quantile: <
    quantile: 0.99
    value: 41.9
  >
>
```
