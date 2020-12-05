# Abstract

Prometheus exporter for hardware and OS metrics exposed by *NIX kernels, written in Go with pluggable metric collectors.

# Materials

* [node-exporter @ github](https://github.com/prometheus/node_exporter)

# Install

## Install native binary

* [Prometheus 모니터링](https://teamsmiley.github.io/2020/01/17/prometheus/)

----

It's better to run without docker for system reasons.

```bash
$ wget https://github.com/prometheus/node_exporter/releases/download/v0.18.1/node_exporter-0.18.1.linux-amd64.tar.gz
$ tar -xvzf node_exporter-0.18.1.linux-amd64.tar.gz
$ cd node_exporter-0.18.1.linux-amd64/
$ mv node_exporter /usr/local/bin/

$ sudo vim /etc/systemd/system/node_exporter.service
[Unit]
Description=Node Exporter
After=network.target
 
[Service]
User=root
Group=root
Type=simple
ExecStart=/usr/local/bin/node_exporter
 
[Install]
WantedBy=multi-user.target

$ sudo systemctl daemon-reload
$ sudo systemctl enable node_exporter.service
$ sudo systemctl start node_exporter.service
$ sudo systemctl status node_exporter.service
 
# Open browser with localhost:9100
$ curl localhost:9100/metrics

# Set prometheus's config for scaping metrics from node_exporter
$ vim ~/m3/scripts/development/m3_stack/prometheus.yml
scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['prometheus:9090']
 
  - job_name: 'node_exporter'
    static_configs:
      - targets: ['xxx.xxx.xxx.xxx:9100']
```

Import node exporter full from grafana labs. https://grafana.com/grafana/dashboards/1860
