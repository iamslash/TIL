# Abstract

Etcd 는 go 로 만들어진 Distributed Key/Value Store 이다. raft 라는 consensus algorithm 을 사용한다.
etcdctl 은 command line client 이다.

# Materials

* [etcd @ joinc](https://www.joinc.co.kr/w/man/12/etcd)
* [etcd @ github](https://github.com/etcd-io/etcd)

# Install with Docker

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

# Basic
