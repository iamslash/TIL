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

* [m3db-operator @ github](https://github.com/m3db/m3db-operator)


