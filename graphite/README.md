# Materials

* [그라파이트와 그라파나로 메트릭스 모니터링 시스템 구축하기](https://www.44bits.io/ko/post/monitoring-system-with-graphite-and-grafana)
* [Using Graphite in Grafana](https://grafana.com/docs/grafana/latest/features/datasources/graphite/)
* [Intro to Dropwizard Metrics](https://www.baeldung.com/dropwizard-metrics)
  * Metric types including Couter Meter, Gauge, Counter, Histogram and Timer, and Reporter to output metrics' values.

# Metric Types

## Meter

WIP

## Gauge

매 순간 순간의 변하는 수치이다. 예를 들어 Tomcat Thread Count 같은 것들이 해당된다.

## Counter

한번 태어나면 숫자가 평생 증가하는 수치이다. 만약 시간별로 증가한 수치를 보고싶다면 graphite 에서 `nonNegativeDerivative()` 를 이용해야 한다.

## Histogram

Key 별로 수치가 존재한다. 분위수가 미리 계산된다. 예를 들어 API 의 Latency 가 해당된다.

## Timer

WIP

## Reporter

WIP

# Queries

* CPU Usage Percentage
  * offset, absolute, aliasByNode

    ```js
    aliasByNode(absolute(offset(sortByName(hosts.$host.system.cpu.percent-idle), -100)), 3)
    ```

* System Load Average
  * aliasSub, brace expansion

    ```js
    aliasSub(aliasByNode(hosts.$host.system.load.load.{longterm,shortterm,midterm}, 3, 7), "xxooxxooxx", "")
    ```

* Memory Usage Bytes

    ```js
    aliasByNode(hosts.$host.system.memory.memory-used, 3, 6)
    ```

* Disk Usage Bytes (Root)

    ```js
    aliasByNode(hosts.$host.system.disk.<disk-name>.root.{free,used}, 3, 8)
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
