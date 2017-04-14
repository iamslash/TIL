# intro

- 머신의 cpu, rem, diskio 등등의 상태를 모니터링하는 프로그램.
- graphite, influxDB, OpenTSDB, Nagios, Sensu등과 비슷하다.
- go로 제작되었다. time series db, web service 등등 all in one이다.
- sql을 사용하지 않고 전용 쿼리를 이용한다.
- grafana를 활용하면 여러 prometheus instance들의 상태를 하나의 대쉬보드로 모니터링 할 수 있다.

# install

```bash
brew install prometheus
```

# usage

- run prometheus

  - a.yml

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
```

  - run

```bash
prometheus -config.file=a.yml
```

- exporter
  - 특정 머신에서 exporter를 활용하면 prometheus server의 URL을 활용하여 데이터를 전송 할 수 있다.
  - [node_exporter](https://github.com/prometheus/node_exporter)는
    linux의 cpu등등의 정보를 수집한다. 실행후 http://<your-device-ip>:9100/metrics

- scraping

- alert
  - alertmanager를 이용하여 관리자에게 email등등의 알람을 전송 할 수 있다.

# conclusion

- 특정 머신에 prometheus를 띄우고 모니터링을 원하는 머신마다 exporter를 실행하자. prometheus를 웹으로 접속하여 모니터링하자.
- application server에 prometheus로 리포팅하는 코드를 구현하자.

# reference

- [Monitoring linux stats with Prometheus.io](https://resin.io/blog/monitoring-linux-stats-with-prometheus-io/)
- [Fleet-wide Machine Metrics Monitoring in 20mins](https://resin.io/blog/prometheusv2/)
