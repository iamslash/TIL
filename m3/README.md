- [Abstract](#abstract)
- [Materials](#materials)
- [Concepts](#concepts)
- [Compoments](#compoments)
- [Basic](#basic)
  - [Creating a Single Node M3DB Cluster with Docker](#creating-a-single-node-m3db-cluster-with-docker)
  - [M3DB Cluster Deployment, Manually](#m3db-cluster-deployment-manually)
  - [setting up M3DB on Kubernetes](#setting-up-m3db-on-kubernetes)

----

# Abstract

M3 is a cluster for long-term logging solution. As documentation for Prometheus states, it is limited by single nodes in its scalability and durability.

# Materials

* [M3 Community Meetup - July 10, 2020 @ vimeo](https://vimeo.com/user120001164/review/440449118/00d5aa7216?sort=lastUserActionEventDate&direction=desc)
* [An introduction to M3](https://aiven.io/blog/an-introduction-to-m3)
* [M3 and Prometheus, Monitoring at Planet Scale for Everyone - Rob Skillington, Uber @ youtube](https://www.youtube.com/watch?v=EFutyuIpFXQ)
  * [pdf](https://static.sched.com/hosted_files/kccnceu19/e0/M3%20and%20Prometheus%2C%20Monitoring%20at%20Planet%20Scale%20for%20Everyone.pdf)
  * [event](https://kccnceu19.sched.com/event/MPbX)
* [M3 Documentation](https://m3db.io/docs/)
* [M3 Media](https://m3db.io/docs/overview/media/)

# Concepts

* Placement: Mapping of Node to Shards
* Namespace: Similar with table of DataBase

# Compoments 

* [M3 Coordinator](m3coordinator.md)
  * service that coordinates reads and writes between upstream systems, such as Prometheus, and M3DB.
* [M3DB](m3db.md)
  * distributed time series database that provides scalable storage and a reverse index of time series
* [M3 Query](m3query.md)
  * service that houses a distributed query engine for querying both realtime and historical metrics, supporting several different query languages.
* **M3 Aggregator**
  * a service that runs as a dedicated metrics aggregator and provides stream based downsampling, based on dynamic rules stored in etcd.

# Basic

## Creating a Single Node M3DB Cluster with Docker

* [Creating a Single Node M3DB Cluster with Docker](https://m3db.io/docs/quickstart/docker/)

```bash
$ docker run --rm -d -p 7201:7201 -p 7203:7203 --name m3db -v $(pwd)/m3db_data:/var/lib/m3db quay.io/m3db/m3dbnode:latest
```

## M3DB Cluster Deployment, Manually

* [m3 stack](https://github.com/m3db/m3/tree/master/scripts/development/m3_stack)

```bash
# Install go
#  https://www.systutorials.com/how-to-install-go-1-13-x-on-ubuntu-18-04/
$ wget https://dl.google.com/go/go1.13.9.linux-amd64.tar.gz
$ tar xf go1.13.9.linux-amd64.tar.gz
$ sudo mv go /usr/local/go-1.13
$ vim ~/.profile
export GOROOT=/usr/local/go-1.13
export PATH=$GOROOT/bin:$PATH
 
# Clone
$ git clone https://github.com/m3db/m3.git
 
# Build
$ cd m3
$ make m3dbnode
 
# Start it
$ cd scripts/development/m3_stack/
$ chmod 644 prometheus.yml
$ ./start_m3.sh
 
# Stop it
$ ./stop_m3.sh
# Open browser with xxx.xxx.xxx.xxx:3000 for grafana
# Open browser with xxx.xxx.xxx.xxx:9090 for prometheus
```

## setting up M3DB on Kubernetes

* [M3DB on Kubernetes @ github](https://m3db.github.io/m3/how_to/kubernetes/)

----

```bash
# Set StorageClss of fast with AWS EBS (class io1)
$ kubectl apply -f https://raw.githubusercontent.com/m3db/m3/master/kube/storage-fast-aws.yaml

# Download bundle
$ wget https://raw.githubusercontent.com/m3db/m3/master/kube/bundle.yaml

# Update image tag from latest to m3dbnode:v1.0.0-rc.0 
$ vim bundle.yaml
image: quay.io/m3/m3dbnode:v1.0.0-rc.0 #latest

# Apply bundle
$ kubectl apply -f bundle.yaml

# Port forward
$ kubectl -n m3db port-forward svc/m3coordinator 7201

# Set placement
$ curl -sSf -X POST localhost:7201/api/v1/services/m3db/placement/init -d '{
  "num_shards": 1024,
  "replication_factor": 3,
  "instances": [
    {
      "id": "m3dbnode-0",
      "isolation_group": "pod0",
      "zone": "embedded",
      "weight": 100,
      "endpoint": "m3dbnode-0.m3dbnode:9000",
      "hostname": "m3dbnode-0.m3dbnode",
      "port": 9000
    },
    {
      "id": "m3dbnode-1",
      "isolation_group": "pod1",
      "zone": "embedded",
      "weight": 100,
      "endpoint": "m3dbnode-1.m3dbnode:9000",
      "hostname": "m3dbnode-1.m3dbnode",
      "port": 9000
    },
    {
      "id": "m3dbnode-2",
      "isolation_group": "pod2",
      "zone": "embedded",
      "weight": 100,
      "endpoint": "m3dbnode-2.m3dbnode:9000",
      "hostname": "m3dbnode-2.m3dbnode",
      "port": 9000
    }
  ]
}'

# Set namespace
$ curl -X POST localhost:7201/api/v1/services/m3db/namespace -d '{
 "name": "default",
 "options": {
  "bootstrapEnabled": true,
  "flushEnabled": true,
  "writesToCommitLog": true,
  "cleanupEnabled": true,
  "snapshotEnabled": true,
  "repairEnabled": false,
  "retentionOptions": {
   "retentionPeriodDuration": "720h",
   "blockSizeDuration": "12h",
   "bufferFutureDuration": "1h",
   "bufferPastDuration": "1h",
   "blockDataExpiry": true,
   "blockDataExpiryAfterNotAccessPeriodDuration": "5m"
  },
  "indexOptions": {
   "enabled": true,
   "blockSizeDuration": "12h"
  }
 }
}'

# Port forward
$ kubectl -n m3db port-forward svc/m3dbnode 9003

# Write metrics
$ curl -sSf -X POST localhost:9003/writetagged -d '{
 "namespace": "default",
 "id": "foo",
 "tags": [
  {
   "name": "city",
   "value": "new_york"
  },
  {
   "name": "endpoint",
   "value": "/request"
  }
 ],
 "datapoint": {
  "timestamp": '"$(date "+%s")"',
  "value": 42.123456789
 }
}'

# Read metrics
$ curl -sSf -X POST http://localhost:9003/query -d '{
 "namespace": "default",
 "query": {
  "regexp": {
   "field": "city",
   "regexp": ".*"
  }
 },
 "rangeStart": 0,
 "rangeEnd": '"$(date "+%s")"'
}' | jq .

# Add m3dbnode
$ curl -sSf -X POST localhost:7201/api/v1/services/m3db/placement -d '{
  "instances": [
    {
      "id": "m3dbnode-3",
      "isolation_group": "pod3",
      "zone": "embedded",
      "weight": 100,
      "endpoint": "m3dbnode-3.m3dbnode:9000",
      "hostname": "m3dbnode-3.m3dbnode",
      "port": 9000
    }
  ]
}'

# Provision grafana
$ kubectl create namespace grafana
$ helm install grafana grafana/grafana \
  --namespace grafana \
  --set persistence.storageClassName="gp2" \
  --set persistence.enabled=true \
  --set adminPassword='EKS!sAWSome' \
  --values /home/ec2-user/eks/eksctl_config/m3db/first/grafana.yaml \
  --set service.type=LoadBalancer

# Provision prometheus
#   https://www.eksworkshop.com/intermediate/240_monitoring/
$ helm install prometheus prometheus-12.0.1.tgz --namespace prometheus --set alertmanager.persistentVolume.storageClass="gp2" --set server.persistentVolume.storageClass="gp2"
```
