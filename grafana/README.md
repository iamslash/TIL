- [Materials](#materials)
- [Install](#install)
  - [Install GitHub Monitoring Stack with docker-compose](#install-github-monitoring-stack-with-docker-compose)
- [Tutorial](#tutorial)
- [Major Metrics for Grafana Graphite EC2](#major-metrics-for-grafana-graphite-ec2)
  - [System](#system)
  - [Request](#request)
  - [JVM](#jvm)
- [Major Metrics for Grafana M3 Kubernetes](#major-metrics-for-grafana-m3-kubernetes)
  - [System](#system-1)

------

# Materials

* [grafana dashboards](https://grafana.com/grafana/dashboards)
  * 멋진 grafana dashboards 들을 검색하고 import 할 수 있다.

# Install

## Install GitHub Monitoring Stack with docker-compose

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

# Tutorial

* [grafana dashboards](https://grafana.com/grafana/dashboards) 에서 좋은 dashboard 를 골라서 import 해보고 학습한다.

# Major Metrics for Grafana Graphite EC2

## System

```c
* CPU
aliasByNode(sortByName(absolute(offset(hosts.$host.system.cpu.percent-idle, -100))), 3)

* Load
aliasByNode(sortByName(hosts.$host.system.load.load.shortterm), 3)

* Mem
aliasByNode(sortByName(hosts.$host.system.memory.memory-used), 3)

* Network Rx Per Sec
aliasByNode(sortByName(perSecond(hosts.$host.system.interface-{em1,eth0}.if_octets.rx)), 3)

* Network Tx Per Sec
aliasByNode(sortByName(perSecond(hosts.$host.system.interface-{em1,eth0}.if_octets.tx)), 3)

* Swap
aliasByNode(scale(sortByName(hosts.$host.system.swap.swap-used), 1024), 3)

* Disk
sumSeries(summarize(hosts.i-*.system.disk.*.*.used, '24h', 'avg', false))
```

## Request

```c
* Request Sum Per Min
aliasByNode(sortByName(nonNegativeDerivative(hosts.$host.$instanceId.tomcat.global-request-processor.http-nio.requestCount)), 3, 4)

* Request Per Min
aliasByNode(sortByName(nonNegativeDerivative(hosts.$host.$instanceId.tomcat.global-request-processor.http-nio.requestCount)), 3, 4)

* Request Error Per Min
aliasByNode(sortByName(nonNegativeDerivative(hosts.$host.$instanceId.tomcat.global-request-processor.http-nio.errorCount)), 3, 4)

* Nginx 4xx Error
aliasByNode(sortByName(sumSeriesWithWildcards(nonNegativeDerivative(hosts.$host.system.tail-nginx*.counter-4xx), 5)), 3)

* Nginx 5xx Error
aliasByNode(sortByName(sumSeriesWithWildcards(nonNegativeDerivative(hosts.$host.system.tail-nginx*.counter-5xx), 5)), 3)

* Tomcat Error
aliasByNode(sortByName(sumSeriesWithWildcards(nonNegativeDerivative(hosts.*.system.tail-tomcat-error-log-10001.counter-all), 5)), 3)

* Current Thread Busy
aliasByNode(sortByName(hosts.$host.$instanceId.tomcat.thread-pool.http-nio.currentThreadsBusy), 3)
```

## JVM

```c
* Heap Usage
aliasByNode(sortByName(hosts.$host.$instanceId.jvm.memory.HeapMemoryUsage.used), 3, 4)

* Non-Heap Usage
aliasByNode(sortByName(hosts.$host.$instanceId.jvm.memory.NonHeapMemoryUsage.used), 3, 4)

* Full GC Time
aliasByNode(sortByName(nonNegativeDerivative(hosts.$host.$instanceId.jvm.gc.{PS_MarkSweep,G1_Old_Generation}.CollectionTime)), 3, 4)

* Minor GC Time
aliasByNode(sortByName(nonNegativeDerivative(hosts.$host.$instanceId.jvm.gc.{PS_Scavenge,G1_Young_Generation}.CollectionTime)), 3, 4)

* DB Active Connections
aliasByNode(sortByName(hosts.$host.$instanceId.datasource.*.NumActive), 3, 6)

* DB Idle Connections
aliasByNode(sortByName(hosts.$host.$instanceId.datasource.*.NumIdle), 3, 6)

* DB Mean Borrow Wait Time
aliasByNode(nonNegativeDerivative(sortByName(hosts.$host.$instanceId.datasource.*.MeanBorrowWaitTimeMillis)), 3, 6)

* DB Max Borrow Wait Time
aliasByNode(nonNegativeDerivative(sortByName(hosts.$host.$instanceId.datasource.*.MaxBorrowWaitTimeMillis)), 3, 6)
```

# Major Metrics for Grafana M3 Kubernetes

## System

```c
* CPU Usage
avg(irate(container_cpu_usage_seconds_total{pod!="", namespace="$namespace"}[1m]) * 100) by (pod)

* Memory Usage
sum (container_memory_working_set_bytes{pod!="", namespace="$namespace"}) by (pod)

* Network RX
sum(irate(container_network_receive_bytes_total{pod!="",namespace="$namespace"}[1m])) by (pod)

* Network TX
sum(irate(container_network_transmit_bytes_total{pod!="",namespace="$namespace"}[1m])) by (pod)

* Throttling
rate(container_cpu_cfs_throttled_seconds_total{pod!="",namespace="$namespace"}[5m])
```
