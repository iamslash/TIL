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

* [Run etcd clusters inside containers @ etcd.io](https://etcd.io/docs/v3.4.0/op-guide/container/)

-----

* `vim run_etcd.sh`

```bash
#!/usr/bin/env bash
if [[ $# -ne 2 ]]; then
  echo "Invalid Arguments"
  echo "  bash run_etcd.sh etcd-node-0 10.xxx.xxx.xxx"
  echo "  bash run_etcd.sh etcd-node-1 10.yyy.yyy.yyy"
  echo "  bash run_etcd.sh etcd-node-2 10.zzz.zzz.zzz"
  exit 2
fi
 
THIS_NAME=$1
THIS_IP=$2
 
#echo $THIS_NAME
#echo $THIS_IP
#exit
 
REGISTRY=quay.io/coreos/etcd
 
# For each machine
ETCD_VERSION=latest
TOKEN=etcd-token
CLUSTER_STATE=new
NAME_1=etcd-node-0
NAME_2=etcd-node-1
NAME_3=etcd-node-2
HOST_1=10.xxx.xxx.xxx
HOST_2=10.yyy.yyy.yyy
HOST_3=10.zzz.zzz.zzz
CLUSTER=${NAME_1}=http://${HOST_1}:2380,${NAME_2}=http://${HOST_2}:2380,${NAME_3}=http://${HOST_3}:2380
DATA_DIR=/var/lib/etcd
 
docker run --rm -d \
  -p 2379:2379 \
  -p 2380:2380 \
  --volume=${DATA_DIR}:/etcd-data \
  --name etcd ${REGISTRY}:${ETCD_VERSION} \
  /usr/local/bin/etcd \
  --data-dir=/etcd-data --name ${THIS_NAME} \
  --initial-advertise-peer-urls http://${THIS_IP}:2380 --listen-peer-urls http://0.0.0.0:2380 \
  --advertise-client-urls http://${THIS_IP}:2379 --listen-client-urls http://0.0.0.0:2379 \
  --initial-cluster ${CLUSTER} \
  --initial-cluster-state ${CLUSTER_STATE} --initial-cluster-token ${TOKEN}
```

* Run in each machine

```bash
# Install etcd
#   https://docs.openstack.org/ko_KR/install-guide/environment-etcd-ubuntu.html
$ sudo apt-get update
$ sudo apt-get install etcd

# Run in machine 1
$ vim run_etcd.sh
$ chmod 755 run_etcd.sh
$ ./run_etcd.sh etcd-node-0 10.xxx.xxx.xxx
 
$ etcdctl --endpoints=http://localhost:2379 member list

# Run in machine 2
$ vim run_etcd.sh
$ chmod 755 run_etcd.sh
$ ./run_etcd.sh etcd-node-0 10.yyy.yyy.yyy
 
$ etcdctl --endpoints=http://localhost:2379 member list

# Run in machine 3
$ vim run_etcd.sh
$ chmod 755 run_etcd.sh
$ ./run_etcd.sh etcd-node-0 10.zzz.zzz.zzz
 
$ etcdctl --endpoints=http://localhost:2379 member list
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
