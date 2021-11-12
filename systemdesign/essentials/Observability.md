- [Abstract](#abstract)
- [References](#references)
- [Materials](#materials)
- [Monitoring](#monitoring)
  - [Major Metrics](#major-metrics)
- [Logging](#logging)
- [Tracing](#tracing)
- [Alerting](#alerting)

----

# Abstract

This is about a constructing Observability system. We should have a Monitoring, Logging, Tracing, Alerting, Auditing system for robust Observability.

* Applications should route logs to somewhere and Log-viewer shows routed logs.
* Applications should route metrics to somewhere and Metric-viewer such as [Grafana](/grafana/README.md) shows routed metrics.
* We should trace API calls for debugging.
* If an incident happened, Alterting solution should email or call.

# References

* [m3 @ TIL](/m3/README.md)
  * Distributed TSDB and Query Engine, Prometheus Sidecar, Metrics Aggregator, and more such as Graphite storage and query engine.
* [Grafana @ TIL](/grafana/README.md)
  * The open-source platform for monitoring and observability.
* [Observability in Distributed Systems @ baeldung](https://www.baeldung.com/distributed-systems-observability)
  * Distributed System 에서 Observability 를 정리한 글이다.

# Materials

* [Observability at Scale: Building Uber’s Alerting Ecosystem @ uber](https://eng.uber.com/observability-at-scale/)

# Monitoring

* [Monitoring @ TIL](Monitoring.md)
* [grafana @ TIL](/grafana/README.md)
* [graphite @ TIL](/graphite/README.md)
* [m3 @ TIL](/m3/README.md)

## Major Metrics

* [System in grafana @ TIL](/grafana/README.md#system)
* [JVM in grafana @ TIL](/grafana/README.md#jvm)
* [Request in grafana @ TIL](/grafana/README.md#request)

# Logging

* [graylog @ TIL](/graylog/README.md)

# Tracing

* [opentracing](https://opentracing.io/)
* [Jaeger](https://www.jaegertracing.io/)

# Alerting

* [Alerting @ TIL](Alerting.md)
* [grafana @ TIL](/grafana/README.md)
  