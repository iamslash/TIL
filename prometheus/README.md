
- [Abstract](#abstract)
- [Architecture](#architecture)
- [Materials](#materials)
- [Install](#install)
  - [Install on osx](#install-on-osx)
  - [Install with docker](#install-with-docker)
  - [Install with docker-compose](#install-with-docker-compose)
- [Basic usages](#basic-usages)
- [PromQL](#promql)
- [Client](#client)
  - [Simple Instrumentation](#simple-instrumentation)
  - [Metric types](#metric-types)

----

# Abstract

서버를 모니터링하는 시스템이다. 모니터링을 원하는 서버에 Exporter 를 설치한다. Prometheus 는 여러 Exporter 들에게 접속하여 데이터를 얻어온다. 즉 pulling 한다. 알림을 받을 규칙을 만들어서 Alert Manager 로 보내면 Alert Manager 가 규칙에 따라 알림을 보낸다.

# Architecture

![](https://prometheus.io/assets/architecture.png)


# Materials

* [오픈소스 모니터링 시스템 Prometheus #1](https://blog.outsider.ne.kr/1254)
  * [오픈소스 모니터링 시스템 Prometheus #2](https://blog.outsider.ne.kr/1255)
* [How To Install Prometheus on Ubuntu 16.04](https://www.digitalocean.com/community/tutorials/how-to-install-prometheus-on-ubuntu-16-04)

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
* [169. [Prometheus] 1편 : Prometheus (프로메테우스) 사용 방법, 기본 개념, 데이터 구조](https://blog.naver.com/PostView.nhn?blogId=alice_k106&logNo=221535163599)
  * [170. [Prometheus] 2편. PromQL 사용하기, k8s에서 Meta Label과 relabel을 활용한 기본 라벨 설정, 그 외의 이야기](https://blog.naver.com/PostView.nhn?blogId=alice_k106&logNo=221535575875)

----

* `http_request_total` 
  * get last http_request_total value

* `http_requests_total{job="apiserver", handler="/api/comments"}`
  * get last http_request_total filtered with labels

* `http_requests_total{job="apiserver", handler="/api/comments"}[5m]`
  * get last 5m http_request_total filtered with labels and with 10 sec granularity

* `http_requests_total{job=~".*server"}`
  * `~` means REGEX
  * `=, !=, =~, !~`
  * use `.+` instead of `.*`

    ```
    {job=~".*"} # Bad!
    {job=~".+"}              # Good!
    {job=~".*",method="get"} # Good!
    ```

* `rate(http_requests_total[5m])[30m:1m]`
  * subquery
  * get the last 5-min rate of the http_requests_total for the past 30 min, with a resolution of 1m

* `max_over_time(deriv(rate(distance_covered_total[5s])[30s:5s])[10m:])`
  * too complicated

* `rate(http_requests_total[5m])`
  * get the per-second rate for all time series with the http_requests_total, as measured over the last 5 min.

* `sum by (job) (
  rate(http_requests_total[5m])
)`
  * get summation grouped by job

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

  * `count by (app) (instance_cpu_time_ns)`
    * get the count of the running instances per application

# Client

## Simple Instrumentation

* [INSTRUMENTING A GO APPLICATION FOR PROMETHEUS](https://prometheus.io/docs/guides/go-application/)

----

* run client
  * main.go

    ```go
    package main

    import (
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
          time.Sleep(2 * time.Second)
        }
      }()
    }

    var (
      opsProcessed = promauto.NewCounter(prometheus.CounterOpts{
        Name: "myapp_processed_ops_total",
        Help: "The total number of processed events",
      })
    )

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

* Counter

This is a cumulative metric. For example, the number of requests served, tasks completed, or errors.

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

* Gauge

This is a metric that represents a single numerical value that can arbitrarily go up and down.

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

* Histogram

A histogram samples observations (usually things like request durations or response sizes) and counts them in configurable buckets. It also provides a sum of all observed values.

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

* Summary

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
