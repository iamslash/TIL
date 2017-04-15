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

  - prometheus config file, a.yml

```
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
  - 특정 machine에서 자료를 수집하여 URL을 통해 prometheus가 접근하도록 한다.
  - [node_exporter](https://github.com/prometheus/node_exporter)는
    linux의 cpu등등의 정보를 수집한다. 실행후
    http://<your-device-ip>:9100/metrics

- scraping
  - config file의 scrape_configs를 설정하여 prometheus가 targeting할 수 있도록 하자.
  
```
scrape_configs:  
  - job_name: "node"
    static_configs:
    - targets:
        - "localhost:9100"
      labels:
        resin_app: RESIN_APP_ID
        resin_device_uuid: RESIN_DEVICE_UUID
```
  
- alert
  - alertmanager를 이용하여 관리자에게 email등등의 알람을 전송 할 수 있다.
  
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
  
  ```
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

# conclusion

- 특정 머신에 prometheus를 띄우고 모니터링을 원하는 머신마다 exporter를 실행하자. prometheus를 웹으로 접속하여 모니터링하자.
- application server에 exporter기능을 추가하자. exporter port가 필요하다.

# reference

- [Monitoring linux stats with Prometheus.io](https://resin.io/blog/monitoring-linux-stats-with-prometheus-io/)
- [Fleet-wide Machine Metrics Monitoring in 20mins](https://resin.io/blog/prometheusv2/)
