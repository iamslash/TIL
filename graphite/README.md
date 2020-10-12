# Materials

* [그라파이트와 그라파나로 메트릭스 모니터링 시스템 구축하기](https://www.44bits.io/ko/post/monitoring-system-with-graphite-and-grafana)
* [Using Graphite in Grafana](https://grafana.com/docs/grafana/latest/features/datasources/graphite/)
* [Intro to Dropwizard Metrics](https://www.baeldung.com/dropwizard-metrics)
  * Metric types including Couter Meter, Gauge, Counter, Histogram and Timer, and Reporter to output metrics' values.

# Metric Types

## Meter

## Gauge

## Counter

## Histogram

## Timer

## Reporter

# Queries

* CPU Usage Percentage
  * offset, absolute, aliasByNode

    ```js
    aliasByNode(absolute(offset(sortByName(hosts.$service.$role.$host.system.cpu.percent-idle), -100)), 3)
    ```

* System Load Average
  * aliasSub, brace expansion

    ```js
    aliasSub(aliasByNode(hosts.$service.$role.$host.system.load.load.{longterm,shortterm,midterm}, 3, 7), "xxooxxooxx", "")
    ```

* Memory Usage Bytes

    ```js
    aliasByNode(hosts.$service.$role.$host.system.memory.memory-used, 3, 6)
    ```

* Disk Usage Bytes (Root)

    ```js
    aliasByNode(hosts.$service.$role.$host.system.disk.<disk-name>.root.{free,used}, 3, 8)
    ```

* Network Tx Bytes
  * sortByName, perSecond

    ```js
    aliasByNode(sortByName(perSecond(hosts.*.$role.$host.system.netlink-<interface-name>.if_octets.tx)), 3, 5)
    ```

* Network Rx Bytes

```js
aliasByNode(sortByName(perSecond(hosts.*.$role.$host.system.netlink-<interface-name>.if_octets.rx)), 3, 5)
```
