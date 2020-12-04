# Abstract

Etcd 는 go 로 만들어진 Distributed Key/Value Store 이다. raft 라는 consensus algorithm 을 사용한다.
etcdctl 은 command line client 이다.

# Materials

* [etcd.io](https://etcd.io/docs/v3.4.0/)
  * [demo @ etcd.io](https://etcd.io/docs/v3.4.0/demo/)
* [etcd @ joinc](https://www.joinc.co.kr/w/man/12/etcd)
* [etcd @ github](https://github.com/etcd-io/etcd)

# Install

## Install with Docker

* [Running etcd under Docker](https://etcd.io/docs/v2/docker_guide/)

```bash
# Running etcd in standalone mode
$ export HostIP="192.168.12.50"
$ docker run -d -v /usr/share/ca-certificates/:/etc/ssl/certs -p 4001:4001 -p 2380:2380 -p 2379:2379 \
 --name etcd quay.io/coreos/etcd:v2.3.8 \
 -name etcd0 \
 -advertise-client-urls http://${HostIP}:2379,http://${HostIP}:4001 \
 -listen-client-urls http://0.0.0.0:2379,http://0.0.0.0:4001 \
 -initial-advertise-peer-urls http://${HostIP}:2380 \
 -listen-peer-urls http://0.0.0.0:2380 \
 -initial-cluster-token etcd-cluster-1 \
 -initial-cluster etcd0=http://${HostIP}:2380 \
 -initial-cluster-state new

$ docker -it etcd bash
> etcdctl -C http://192.168.12.50:2379 member list
> etcdctl cluster-health
```

```bash
# Running a 3 node etcd cluster
$ docker run --rm -d -v /usr/share/ca-certificates/:/etc/ssl/certs -p 4001:4001 -p 2380:2380 -p 2379:2379 \
 --name etcd quay.io/coreos/etcd:v2.3.8 \
 -name etcd0 \
 -advertise-client-urls http://192.168.12.50:2379,http://192.168.12.50:4001 \
 -listen-client-urls http://0.0.0.0:2379,http://0.0.0.0:4001 \
 -initial-advertise-peer-urls http://192.168.12.50:2380 \
 -listen-peer-urls http://0.0.0.0:2380 \
 -initial-cluster-token etcd-cluster-1 \
 -initial-cluster etcd0=http://192.168.12.50:2380,etcd1=http://192.168.12.51:2380,etcd2=http://192.168.12.52:2380 \
 -initial-cluster-state new

$ docker run --rm -d -v /usr/share/ca-certificates/:/etc/ssl/certs -p 4001:4001 -p 2380:2380 -p 2379:2379 \
 --name etcd quay.io/coreos/etcd:v2.3.8 \
 -name etcd1 \
 -advertise-client-urls http://192.168.12.51:2379,http://192.168.12.51:4001 \
 -listen-client-urls http://0.0.0.0:2379,http://0.0.0.0:4001 \
 -initial-advertise-peer-urls http://192.168.12.51:2380 \
 -listen-peer-urls http://0.0.0.0:2380 \
 -initial-cluster-token etcd-cluster-1 \
 -initial-cluster etcd0=http://192.168.12.50:2380,etcd1=http://192.168.12.51:2380,etcd2=http://192.168.12.52:2380 \
 -initial-cluster-state new

$ docker run --rm -d -v /usr/share/ca-certificates/:/etc/ssl/certs -p 4001:4001 -p 2380:2380 -p 2379:2379 \
 --name etcd quay.io/coreos/etcd:v2.3.8 \
 -name etcd2 \
 -advertise-client-urls http://192.168.12.52:2379,http://192.168.12.52:4001 \
 -listen-client-urls http://0.0.0.0:2379,http://0.0.0.0:4001 \
 -initial-advertise-peer-urls http://192.168.12.52:2380 \
 -listen-peer-urls http://0.0.0.0:2380 \
 -initial-cluster-token etcd-cluster-1 \
 -initial-cluster etcd0=http://192.168.12.50:2380,etcd1=http://192.168.12.51:2380,etcd2=http://192.168.12.52:2380 \
 -initial-cluster-state new
```

## Install with docker-compose

* [bitnami/etcd @ dockerhub](https://hub.docker.com/r/bitnami/etcd/)

-----

* Run etcd cluster

```bash
$ curl -LO https://raw.githubusercontent.com/bitnami/bitnami-docker-etcd/master/docker-compose-cluster.yml
$ docker-compose up

$ docker exec -it a_etcd2_1 bash
```

* `docker-compose-cluster.yml`

```yml
version: '2'

services:
  etcd1:
    networks:
      - backend  
    image: docker.io/bitnami/etcd:3-debian-10
    environment:
      - ALLOW_NONE_AUTHENTICATION=yes
      - ETCD_NAME=etcd1
      - ETCD_INITIAL_ADVERTISE_PEER_URLS=http://etcd1:2380
      - ETCD_LISTEN_PEER_URLS=http://0.0.0.0:2380
      - ETCD_LISTEN_CLIENT_URLS=http://0.0.0.0:2379
      - ETCD_ADVERTISE_CLIENT_URLS=http://etcd1:2379
      - ETCD_INITIAL_CLUSTER_TOKEN=etcd-cluster
      - ETCD_INITIAL_CLUSTER=etcd1=http://etcd1:2380,etcd2=http://etcd2:2380,etcd3=http://etcd3:2380
      - ETCD_INITIAL_CLUSTER_STATE=new
    expose:
      - "2379"
      - "2380"
    ports:
      - "0.0.0.0:2379:2379"
      - "0.0.0.0:2380:2380"
  etcd2:
    networks:
      - backend
    image: docker.io/bitnami/etcd:3-debian-10
    environment:
      - ALLOW_NONE_AUTHENTICATION=yes
      - ETCD_NAME=etcd2
      - ETCD_INITIAL_ADVERTISE_PEER_URLS=http://etcd2:2380
      - ETCD_LISTEN_PEER_URLS=http://0.0.0.0:2380
      - ETCD_LISTEN_CLIENT_URLS=http://0.0.0.0:2379
      - ETCD_ADVERTISE_CLIENT_URLS=http://etcd2:2379
      - ETCD_INITIAL_CLUSTER_TOKEN=etcd-cluster
      - ETCD_INITIAL_CLUSTER=etcd1=http://etcd1:2380,etcd2=http://etcd2:2380,etcd3=http://etcd3:2380
      - ETCD_INITIAL_CLUSTER_STATE=new
  etcd3:
    networks:
      - backend
    image: docker.io/bitnami/etcd:3-debian-10
    environment:
      - ALLOW_NONE_AUTHENTICATION=yes
      - ETCD_NAME=etcd3
      - ETCD_INITIAL_ADVERTISE_PEER_URLS=http://etcd3:2380
      - ETCD_LISTEN_PEER_URLS=http://0.0.0.0:2380
      - ETCD_LISTEN_CLIENT_URLS=http://0.0.0.0:2379
      - ETCD_ADVERTISE_CLIENT_URLS=http://etcd3:2379
      - ETCD_INITIAL_CLUSTER_TOKEN=etcd-cluster
      - ETCD_INITIAL_CLUSTER=etcd1=http://etcd1:2380,etcd2=http://etcd2:2380,etcd3=http://etcd3:2380
      - ETCD_INITIAL_CLUSTER_STATE=new
networks:
  backend:
```

# Basic

## Client Usages

* [etcd 기본사용](https://arisu1000.tistory.com/27782)
* [api_grpc_gateway.md @ github](https://github.com/etcd-io/etcd/blob/master/Documentation/dev-guide/api_grpc_gateway.md)

------

Please use `-L` option to redirect to master server.

```bash
$ docker exec -it a_etcd2_1 bash

# Health check
$ curl -L http://localhost:2379/health
{"health":"true"}
$ etcdctl cluster-health

# Version
$ curl -L http://localhost:2379/version
{"etcdserver":"3.4.14","etcdcluster":"3.4.0"}

# Insert key
$ curl -L http://localhost:2379/v3/kv/put -X POST -d '{"key": "Zm9v", "value": "YmFy"}'
{"header":{"cluster_id":"10316109323310759371","member_id":"15168875803774599630","revision":"2","raft_term":"2"}}

#
```
