# Abstract

consensus algorithm 중 하나인 zab 을 구현한 coordinator 이다.

Consensus 란 분산 시스템에서 노드 간의 상태를 공유하는 알고리즘을 말한다. 가장 유명한 알고리즘으로 `Paxos` 가 있다. `Raft` 는 이해하기 어려운 기존의 알고리즘과 달리 쉽게 이해하고 구현하기 위해 설계되었다.

# Materials

* [아이펀팩토리 게임서버개발 세미나 2부 zookeeper 를 이용한 분산 서버 만들기 @ youtube](https://www.youtube.com/watch?v=8yGHlHm0h6g)
  * [ppt](https://www.slideshare.net/iFunFactory/apache-zookeeper-55966566)
* [Zookeeper Tutorial @ joinc](https://www.joinc.co.kr/w/man/12/zookeeper/tutorial)
* [How To Install and Configure an Apache ZooKeeper Cluster on Ubuntu 18.04](https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-an-apache-zookeeper-cluster-on-ubuntu-18-04)
* [Zookeeper Tutorial @ tutorialpoint](https://www.tutorialspoint.com/zookeeper/index.htm)

# Install with Docker

* [zookeeper @ dockerhub](https://hub.docker.com/_/zookeeper)

----

```bash
docker pull zookeeper
docker run --name my-zookeeper -p 2181:2181 -p 2888:2888 -p 3888:3888 -p 8080:8080 --restart always -d zookeeper
# EXPOSE 2181 2888 3888 8080 (the zookeeper client port, follower port, election port, AdminServer port respectively)
docker exec -it my-zookeeper /bin/bash
cd bin
./zkCli.sh
# start zookeeper server
#bin/zkServer.sh start
# start zookeeper cli
#bin/zkCli.sh
```

* browser 로 `http://localhost:8080/commands` 를 접속해 본다.
  * [참고](http://www.mtitek.com/tutorials/zookeeper/http-admin-interface.php)

# Basic

## Create znodes (persistent, ephemeral, sequential)

ephemeral node 를 등록한 client 가 접속이 끊어지면 phemeral node 는 사라진다. service discovery 에 이용할 수 있다. watch 를 등록해 두면 해당 데이터가 변경되었을 때 알림을 수신할 수 있다.

```bash
# create persistent znode
create /FirstZnode "Myfirstzookeeper-app"

# create sequential znode
create -s /FirstZnode second-data
# [zk: localhost:2181(CONNECTED) 2] create -s /FirstZnode “second-data”
# Created /FirstZnode0000000023

# create ephemeral znode
create -e /SecondZnode "Ephemeral-data"
```

## Get data

```bash
get /FirstZnode
# [zk: localhost:2181(CONNECTED) 1] get /FirstZnode
# “Myfirstzookeeper-app”
# cZxid = 0x7f
# ctime = Tue Sep 29 16:15:47 IST 2015
# mZxid = 0x7f
# mtime = Tue Sep 29 16:15:47 IST 2015
# pZxid = 0x7f
# cversion = 0
# dataVersion = 0
# aclVersion = 0
# ephemeralOwner = 0x0
# dataLength = 22
# numChildren = 0

get /FirstZnode0000000023
# [zk: localhost:2181(CONNECTED) 1] get /FirstZnode0000000023
# “Second-data”
# cZxid = 0x80
# ctime = Tue Sep 29 16:25:47 IST 2015
# mZxid = 0x80
# mtime = Tue Sep 29 16:25:47 IST 2015
# pZxid = 0x80
# cversion = 0
# dataVersion = 0
# aclVersion = 0
# ephemeralOwner = 0x0
# dataLength = 13
# numChildren = 0
```

## Watch znode for changes

watch 는 get command 로 등록할 수 있다.

```bash
#get /FirstZnode watch 1 # deprecated
get -w /FirstZnode
```

## Set data

watch 가 등록되어 있다면 알림이 날아갈 것이다. 

```bash
set /SecondZnode Data-updated
```

## Create children of a znode

```bash
create /FirstZnode/Child1 firstchildren
```

## List children of a znode

```bash
ls /MyFirstZnode
```

## Check Status

```bash
stat /FirstZnode
```

## Remove / Delete a znode

```bash
rmr /FirstZnode
```

# Case Studies

* 로비 서버가 유저에게 접근 가능한 게임서버 리스트를 동적으로 전달하고 싶다. 
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
  * concensus 는 비싼 연산이다. 특히 "쓰기"
  * Write bound job 은 zookeeper 를 이용하지 말자.

