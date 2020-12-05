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

## Install with docker-compose

* [Quickstart for Thanos @ github.com](git@github.com:dbluxo/quickstart-thanos.git)

----

```bash
$ git clone https://github.com/dbluxo/quickstart-thanos
$ cd quickstart-thanos
$ docker-compose up -d

# Open browser with localhost:3000 for grafana. admin / foobar
# Open browser with localhost:9000 for minio. smth / Need8Chars
```




