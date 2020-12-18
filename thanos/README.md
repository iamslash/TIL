# Abstract

Open source, highly available Prometheus setup with long term storage capabilities.

# Materials

* [thanos.io](https://thanos.io/)
* [Quickstart for Thanos @ github.com](git@github.com:dbluxo/quickstart-thanos.git)

# Components

* **Sidecar**: connects to Prometheus, reads its data for query and/or uploads it to cloud storage.
* **Store Gateway**: serves metrics inside of a cloud storage bucket.
* **Compactor**: compacts, downsamples and applies retention on the data stored in cloud storage bucket.
* **Receiver**: receives data from Prometheus’ remote-write WAL, exposes it and/or upload it to cloud storage.
* **Ruler/Rule**: evaluates recording and alerting rules against data in Thanos for exposition and/or upload.
* **Querier/Query**: implements Prometheus’ v1 API to aggregate data from the underlying components.

# Architecture

* [Prometheus HA with Thanos](https://medium.com/@mail2ramunakerikanti/thanos-for-prometheus-f7f111e3cb75)

----

![](https://miro.medium.com/max/770/1*l_5E_Ap4Ps5Ys6zDR73x3Q.png)

# Install

## Install prometheus, grafana, sidecar, store, compactor, querier, minio, bucket-web with docker-compose

* [Quickstart for Thanos @ github.com](git@github.com:dbluxo/quickstart-thanos.git)

----

```bash
$ git clone https://github.com/dbluxo/quickstart-thanos
$ cd quickstart-thanos
$ docker-compose up -d

# Open browser with localhost:3000 for grafana. admin / foobar
# Open browser with localhost:9000 for minio. smth / Need8Chars
```

## Install receiver, querier

* [Thanos Receive example with docker-compose](https://gist.github.com/blockloop/637bd0c8c9295178b67812f43e661419)

----

```bash
$ mkdir thanos-receive-docker-compose
$ cd thanos-receive-docker-compose
$ vim docker-compose.yaml
$ vim hashring.json
$ vim prometheus.yaml
$ mkdir thanos
$ vim thanos/bucket_config.yaml
$ chmod 755 prometheus.yaml
 
$ docker-compose up -d
$ docker-compose down
```

* `docker-compose.yaml`

```yaml
version: "3"

networks:
  default:
    driver: bridge
    ipam:
      config:
        - subnet: 172.16.57.0/24

services:
  prometheus:
    image: prom/prometheus:v2.15.2
    ports:
      - 9090:9090
    volumes:
      - ${PWD}/prometheus.yaml:/prometheus.yaml
    command:
      - '--config.file=/prometheus.yaml'

  query:
    image: quay.io/blockloop/thanos@sha256:f6dfe89483ed9f8d9d256d5da9928bcfbd7c41dfd73e9c4ce140b0b01ccf24a8
    expose:
      - "20902"
    ports:
      - 20902:10902
    command: >
      query
      --log.format=logfmt
      --query.replica-label=replica
      --store=thanos-01:10901
      --store=thanos-02:10901
      --store=thanos-03:10901
      --store=thanos-store-gateway:10091

  thanos-store-gateway:
    image: quay.io/thanos/thanos:v0.17.2
    volumes:
        - ./thanos/:/etc/thanos/
        - /home/iamslash/data/thanos-store:/data/thanos/store
    command:
        - 'store'
        - '--grpc-address=0.0.0.0:10091'
        - '--http-address=0.0.0.0:10902'
        - '--data-dir=/data/thanos/store'
        - '--objstore.config-file=/etc/thanos/bucket_config.yaml'
    restart: always

  thanos-bucket-web:
    image: quay.io/thanos/thanos:v0.17.2
    volumes:
        - ./thanos/:/etc/thanos/
    command:
        - 'tools'
        - 'bucket'
        - 'web'
        - '--http-address=0.0.0.0:10902'
        - '--log.level=debug'
        - '--objstore.config-file=/etc/thanos/bucket_config.yaml'
        - '--refresh=5m'
        - '--timeout=2m'
        - '--label=replica'
    ports:
        - 10904:10902
    restart: always    

  thanos-01:
    image: quay.io/blockloop/thanos@sha256:f6dfe89483ed9f8d9d256d5da9928bcfbd7c41dfd73e9c4ce140b0b01ccf24a8
    expose:
      - 10902
      - 10901
      - 19291
    ports:
      - 10902:10902
      - 10901:10901
      - 19291:19291
    volumes:
      - ${PWD}/hashring.json:/hashring.json
      - /home/iamslash/data/thanos-receive/data-01:/data
      - ./thanos/:/etc/thanos/
    command: >
      receive
      --log.level=debug
      --log.format=logfmt
      --remote-write.address=0.0.0.0:19291
      --tsdb.retention=24h
      --tsdb.path=/data
      --tsdb.min-block-duration=2h
      --tsdb.max-block-duration=32h
      --receive.replication-factor=2
      --receive.hashrings-file=/hashring.json
      --label=replica='"thanos-01"'
      --receive.local-endpoint="thanos-01:10901"
      --objstore.config-file=/etc/thanos/bucket_config.yaml

  thanos-02:
    image: quay.io/blockloop/thanos@sha256:f6dfe89483ed9f8d9d256d5da9928bcfbd7c41dfd73e9c4ce140b0b01ccf24a8
    ports:
      - 10902
      - 10901
      - 19291
    volumes:
      - ${PWD}/hashring.json:/hashring.json
      - /home/iamslash/data/thanos-receive/data-02:/data
      - ./thanos/:/etc/thanos/
    command: >
      receive
      --log.level=debug
      --log.format=logfmt
      --remote-write.address=0.0.0.0:19291
      --tsdb.path=/data
      --tsdb.retention=24h
      --tsdb.min-block-duration=2h
      --tsdb.max-block-duration=32h
      --receive.replication-factor=2
      --receive.hashrings-file=/hashring.json
      --label=replica='"thanos-02"'
      --receive.local-endpoint="thanos-02:10901"
      --objstore.config-file=/etc/thanos/bucket_config.yaml

  thanos-03:
    image: quay.io/blockloop/thanos@sha256:f6dfe89483ed9f8d9d256d5da9928bcfbd7c41dfd73e9c4ce140b0b01ccf24a8
    ports:
      - 10902
      - 10901
      - 19291
    volumes:
      - ${PWD}/hashring.json:/hashring.json
      - /home/iamslash/data/thanos-receive/data-03:/data
      - ./thanos/:/etc/thanos/
    command: >
      receive
      --log.level=debug
      --log.format=logfmt
      --remote-write.address=0.0.0.0:19291
      --tsdb.path=/data
      --tsdb.retention=24h
      --tsdb.min-block-duration=2h
      --tsdb.max-block-duration=32h
      --receive.replication-factor=2
      --receive.hashrings-file=/hashring.json
      --label=replica='"thanos-03"'
      --receive.local-endpoint="thanos-03:10901"
      --objstore.config-file=/etc/thanos/bucket_config.yaml
```

* `hashring.json`

```json
[
    {
        "hashring": "default",
        "endpoints": ["thanos-01:10901", "thanos-02:10901", "thanos-03:10901"],
        "tenants": []
    }
]
```

* `prometheus.yaml`

```yaml
# vim set syntax=yaml
global:
  scrape_interval:     10s
  evaluation_interval: 60s

remote_write:
  - url: "http://thanos-01:19291/api/v1/receive"
    queue_config:
      max_samples_per_send: 500
      batch_send_deadline: 5s
      min_backoff: 50ms
      max_backoff: 500ms
scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['127.0.0.1:9090']
  - job_name: 'thanos'
    static_configs:
      - targets: ['thanos-01:10902', 'thanos-02:10902', 'thanos-03:10902']
```

* `thanos/bucket_config.yaml`

```yaml
type: S3
config:
  bucket: iamslash.thanos-receive.dev
  signature_version2: false
  endpoint: s3-website.ap-northeast-2.amazonaws.com
  insecure: true
```
