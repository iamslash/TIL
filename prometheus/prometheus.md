# prometheus intro

- 머신의 상태를 모니터링하는 프로그램.
- graphite, influxDB, OpenTSDB, Nagios, Sensu등과 비슷하다.
- go로 제작되었다. time series db, web service 등등 all in one이다.
- sql을 사용하지 않고 전용 쿼리를 이용한다.
- grafana를 활용하면 더욱 멋진 대쉬보드를 이용할 수 있다.

# prometheus install

```bash
brew install prometheus
```

# prometheus usage

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
