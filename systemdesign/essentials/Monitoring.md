# Abstract

This is about a constructing Monitoring system. We should have a Metric, Logger,Alert, Tracing system for robust monitoring.

* Applications should route logs to somewhere and Log-viewer shows routed logs.
* Applications should route metrics to somewhere and Metric-viewer such as [Grafana](/grafana/README.md) shows routed metrics.
* If an incident happened, Alterting solution should email or call.
* We should trace API calls for debugging.

# References

* [m3 @ TIL](/m3/README.md)
  * Distributed TSDB and Query Engine, Prometheus Sidecar, Metrics Aggregator, and more such as Graphite storage and query engine.
* [Grafana @ TIL](/grafana/README.md)
  * The open-source platform for monitoring and observability.
* [Observability in Distributed Systems @ baeldung](https://www.baeldung.com/distributed-systems-observability)

# Materials

* [Observability at Scale: Building Uber’s Alerting Ecosystem @ uber](https://eng.uber.com/observability-at-scale/)

# Logging

* [Logging @ TIL](LogViewer.md)
* [graylog @ TIL](/graylog/README.md)

# Metric Solution

* [Metric @ TIL](Metric.md)
* [grafana @ TIL](/grafana/README.md)
* [graphite @ TIL](/graphite/README.md)
* [m3 @ TIL](/m3/README.md)

# Major Metrics

* [System in grafana @ TIL](/grafana/README.md#system)
* [JVM in grafana @ TIL](/grafana/README.md#jvm)
* [Request in grafana @ TIL](/grafana/README.md#request)

# Alert Solution

* [Alerting @ TIL](Alerting.md)
* [grafana @ TIL](/grafana/README.md)

# Trace Solution

* [opentracing](https://opentracing.io/)
* [Jaeger](https://www.jaegertracing.io/)
  