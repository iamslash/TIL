- [Abstract](#abstract)
- [Materials](#materials)
- [Install with Docker](#install-with-docker)
- [Basic](#basic)
  - [Create znodes (persistent, ephemeral, sequential)](#create-znodes-persistent-ephemeral-sequential)
  - [Get data](#get-data)
  - [Remove / Delete a znode](#remove--delete-a-znode)
  - [Set data](#set-data)
  - [Watch znode for changes](#watch-znode-for-changes)
  - [Check Status](#check-status)
  - [Ephemeral nodes](#ephemeral-nodes)
- [Case Studies](#case-studies)
- [Caution](#caution)

----

# Abstract

consensus algorithm 중 하나인 zab 을 구현한 coordinator 이다. 주로 다음과 같은
특징을 갖는다.

* Strong Consistency 를 지원하므로 Global Locking 에 사용한다.
* Configuration Management System 으로 이용한다. 예를 들어 A/B test 를 수행할 때
  IOS client 를 한번 build 하고 configuration 의 내용에 따라 runtime 에 기능이
  달라지도록 한다.
* Strong Consistency 를 보장하기 때문에 Write 연산이 비싸다.

Consensus 란 분산 시스템에서 노드 간의 상태를 공유하는 알고리즘을 말한다. 가장
유명한 알고리즘으로 `Paxos` 가 있다. `Raft` 는 이해하기 어려운 기존의 알고리즘과
달리 쉽게 이해하고 구현하기 위해 설계되었다.

# Materials

* [ZooKeeper를 활용한 Redis Cluster 관리](https://d2.naver.com/helloworld/294797)
* [아이펀팩토리 게임서버개발 세미나 2부 zookeeper 를 이용한 분산 서버 만들기 @ youtube](https://www.youtube.com/watch?v=8yGHlHm0h6g)
  * [ppt](https://www.slideshare.net/iFunFactory/apache-zookeeper-55966566)
* [Zookeeper Tutorial @ joinc](https://www.joinc.co.kr/w/man/12/zookeeper/tutorial)
* [How To Install and Configure an Apache ZooKeeper Cluster on Ubuntu 18.04](https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-an-apache-zookeeper-cluster-on-ubuntu-18-04)
* [Zookeeper Tutorial @ tutorialpoint](https://www.tutorialspoint.com/zookeeper/index.htm)

# Install with Docker

* [zookeeper @ dockerhub](https://hub.docker.com/_/zookeeper)

----

```bash
$ docker pull zookeeper
$ docker run --rm --name my-zookeeper -p 2181:2181 -p 2888:2888 -p 3888:3888 -p 8080:8080 --restart always -d zookeeper
# EXPOSE 2181 2888 3888 8080 (the zookeeper client port, follower port, election port, AdminServer port respectively)
$ docker exec -it my-zookeeper /bin/bash
$ cd bin
$ ./zkCli.sh
# start zookeeper server
# $ bin/zkServer.sh start
# start zookeeper cli
# $ bin/zkCli.sh
```

* browser 로 `http://localhost:8080/commands` 를 접속해 본다.
  * [참고](http://www.mtitek.com/tutorials/zookeeper/http-admin-interface.php)

# Basic

## Create znodes (persistent, ephemeral, sequential)

node 는 File System 처럼 tree structure 이고 각 node 마다 meta data 를 갖는다.

node 의 종류는 ephemeral, persistent 와 같이 두가지가 있다. 

* Ephemeral: session 이 끊어지면 삭제되는 node 이다. Service Discovery 에 사용할 수 있다.
* Persistent: session 이 끊어져도 삭제되지 않는 node 이다.

또한 node 는 Sequential 속성을 가질 수 있다. Sequential 속성을 갖는 node 는 node 의 이름뒤에 4 byte 크기의 숫자가 부착된다. 이 숫자는 0 부터 2,147,483,647 개의 unique node 를 만들 수 있다.

```bash
$ help

$ ls /

# Create persistent node
$ create /docker-cluster docker_cluster
$ create /docker-cluster/0001 cluster_0001
$ create /docker-cluster/0002 cluster_0002

# Create sequential persistent node
$ create -s /docker-cluster/0001/mynode node1
Created /docker-cluster/0001/mynode0000000000
$ create -s /docker-cluster/0001/mynode node2
Created /docker-cluster/0001/mynode0000000001
```

## Get data

```bash
$ get -s /docker-cluster/0001/mynode0000000000
node1
cZxid = 0x10
ctime = Sat Sep 12 11:27:15 UTC 2020
mZxid = 0x10
mtime = Sat Sep 12 11:27:15 UTC 2020
pZxid = 0x10
cversion = 0
dataVersion = 0
aclVersion = 0
ephemeralOwner = 0x0
dataLength = 5
numChildren = 0
```

| key            | description         |
| -------------- | ------------------- |
| data           | node's data         |
| cZxid          | node created id     |
| ctime          | node created time   |
| mzXid          | node modified id    |
| mtime          | node modified time  |
| pZxid          |                     |
| cversion       |                     |
| dataVersion    | node data's version |
| aclVersion     |                     |
| ephemeralOwner |                     |
| dataLength     |                     |
| numChildren    | children's number   |

## Remove / Delete a znode

```bash
$ delete /docker-cluster/0001/mynode0000000000
```

## Set data

```bash
$ set /docker-cluster/0001/mynode0000000001 iamslash
$ get -s /docker-cluster/0001/mynode0000000001
```

## Watch znode for changes

terminal b 을 실행해서 접속한다. 다음과 같이 `get -w` 를 이용해서 watch 를 등록한다.

```bash
$ get -w /docker-cluster/0001/mynode0000000001
```

terminal a 에서 다음을 실행한다.

```console
$ set /docker-cluster/0001/mynode0000000001 hello
```

watch event 는 한번 뿐이다. event 를 받고나서 다시 등록해야 한다.

## Check Status

```bash
$ stat /docker-cluster/0001/mynode0000000001
```

## Ephemeral nodes

terminal b 에서 다음을 실행하여 Ephermeral node 를 등록하자.

```console
$ create -e -s /docker-cluster/0001/mynode node2
$ create -e -s /docker-cluster/0001/mynode node3
$ ls /docker-cluster/0001
[mynode0000000000, mynode0000000001, mynode0000000002, mynode0000000003]
$ get -s /docker-cluster/0001/mynode0000000002
cZxid = 0x1a
ctime = Sat Sep 12 11:45:59 UTC 2020
mZxid = 0x1a
mtime = Sat Sep 12 11:45:59 UTC 2020
pZxid = 0x1a
cversion = 0
dataVersion = 0
aclVersion = 0
ephemeralOwner = 0x10000da40f20005
dataLength = 5
numChildren = 0
```

terminal b 를 종료한다. 그리고 terminal a 에서 다음을 실행한다. 삭제되는데 시간이 걸릴 수 있다.

```console
$ ls /docker-cluster/0001
[mynode0000000000, mynode0000000001]
```

# Case Studies

* 로비 서버가 게임서버 리스트를 유저에게 동적으로 전달하고 싶다. 
  * 게임 서버는 프로세스가 뜰 때 자신의 uid 를 생성한다.
  * `/servers/{uid}` 형태로 노드를 생성한다. 노드 data 에 자신의 ip, port 를 등록한다.
  * `/servers` 를 순회하면서 기존에 등록된 ip, port 들을 얻어낸다.
  * `/servers` 에 watcher 를 등록해서 이후에 등록된 게임 서버정보를 수신한다.
  * `/servers/{uid}` 에 ephemeral 을 등록하여 자신이 죽었을 때 다른 서버들에게 알려준다.
  * `/servers/{groupname}` 형태로 그루핑 할 수도 있다.
* 게임 서버가 여러대일 때 유저의 중복 로그인을 체크하고 싶다.
  * 유저가 접속하면 해당 게임서버는 `/users/{name}` 형태로 노드를 생성한다. 노드 데이터로 서버의 `uid` 를 기록한다.
  * 다른 서버에서 유저접속 여부를 확인하고 싶으면 `/users/{name}` 의 존재여부를 확인한다.
  * 유저가 로그아웃하면 `/users/{name}` 을 삭제한다.
  * 해당 게임서버가 죽었을 때 `ephemeral` 로 등록되어 있었다면 `/users/{name}` 역시 사라질 것이다.
  * DB server 에 저장하는 것보다 빠르다.

# Caution

* 빈번히 갱신되는 데이터 저장소로 사용하면 절대 안된다.
  * Concensus algorithm 덕분에 Strong Consistency 를 지원한다. Concensus algorithm 는 비싼 연산이다. 특히 "쓰기".
  * Write bound job 은 zookeeper 를 이용하지 말자.
