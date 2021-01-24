# Abstract

Metric 에 관련된 내용들을 정리한다.

# Basic

## Basic Metrics Handler

* [INSTRUMENTING A GO APPLICATION FOR PROMETHEUS @ prometheus](https://prometheus.io/docs/guides/go-application/)
* [Prometheus instrumentation library for Go applications @ github](https://github.com/prometheus/client_golang)

----

`main.go`

```go
package main

import (
        "net/http"

        "github.com/prometheus/client_golang/prometheus/promhttp"
)

func main() {
        http.Handle("/metrics", promhttp.Handler())
        http.ListenAndServe(":2112", nil)
}
```

Command line for build

```bash
$ go get github.com/prometheus/client_golang/prometheus/promhttp
$ go run main.go
$ curl localhost:2112/metrics
```

## Counter, Gauge, Histogram, Summary

* [Gauge example @ gist](https://gist.github.com/tembleking/0b8968dbdf36dfef6227fbfdd9bb1a82)

-----

`main.go`

```go
package main

import (
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
	counter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Namespace: "golang",
			Name:      "my_counter",
			Help:      "This is my counter",
		})

	gauge = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Namespace: "golang",
			Name:      "my_gauge",
			Help:      "This is my gauge",
		})

	histogram = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Namespace: "golang",
			Name:      "my_histogram",
			Help:      "This is my histogram",
		})

	summary = prometheus.NewSummary(
		prometheus.SummaryOpts{
			Namespace: "golang",
			Name:      "my_summary",
			Help:      "This is my summary",
		})
)

func main() {
	rand.Seed(time.Now().Unix())

	histogramVec := prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name: "prom_request_time",
		Help: "Time it has taken to retrieve the metrics",
	}, []string{"time"})

	prometheus.Register(histogramVec)

	http.Handle("/metrics", newHandlerWithHistogram(promhttp.Handler(), histogramVec))

	prometheus.MustRegister(counter)
	prometheus.MustRegister(gauge)
	prometheus.MustRegister(histogram)
	prometheus.MustRegister(summary)

	go func() {
		for {
			counter.Add(rand.Float64() * 5)
			gauge.Add(rand.Float64()*15 - 5)
			histogram.Observe(rand.Float64() * 10)
			summary.Observe(rand.Float64() * 10)

			time.Sleep(time.Second)
		}
	}()

	log.Fatal(http.ListenAndServe(":8080", nil))
}

func newHandlerWithHistogram(handler http.Handler, histogram *prometheus.HistogramVec) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		start := time.Now()
		status := http.StatusOK

		defer func() {
			histogram.WithLabelValues(fmt.Sprintf("%d", status)).Observe(time.Since(start).Seconds())
		}()

		if req.Method == http.MethodGet {
			handler.ServeHTTP(w, req)
			return
		}
		status = http.StatusBadRequest

		w.WriteHeader(status)
	})
}
```

## Metric for versions

* [Exposing the software version to Prometheus](https://www.robustperception.io/exposing-the-software-version-to-prometheus)
* [exemple for _build_info metric #693](https://github.com/prometheus/client_golang/issues/693)
* [Expose version and other build information as metric similar to Prometheus itself @ github](https://github.com/prometheus/node_exporter/pull/176/files)

-----

`main.go`

```go
package main

import (
  "net/http"

  "github.com/prometheus/client_golang/prometheus"
  "github.com/prometheus/client_golang/prometheus/promhttp"
)

var (
  Version  string
  Revision string
  Branch   string

  metric = prometheus.NewGaugeVec(
    prometheus.GaugeOpts{
      Name: "app_build_info",
      Help: "A metric with a constant '1' value labeled by version, revision, and branch from which the app was built.",
    },
    []string{"version", "revision", "branch"},
  )
)

func main() {
  Version = "1.0.1"
  Revision = "1a2s3d3"
  Branch = "master"
  metric.WithLabelValues(Version, Revision, Branch).Set(1)

  prometheus.MustRegister(metric)
  http.Handle("/metrics", promhttp.Handler())
  http.ListenAndServe(":2112", nil)
}
```

Command line for build

```bash
$ go get github.com/prometheus/client_golang/prometheus/promhttp
$ go run main.go
$ curl localhost:2112/metrics
# TYPE app_build_info gauge
app_build_info{branch="master",revision="1a2s3d3",version="1.0.1"} 1
```
