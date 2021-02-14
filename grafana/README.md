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

# Major Metrics

## System

## NginX

## JVM

