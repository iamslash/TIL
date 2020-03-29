- [Abstract](#abstract)
- [Materials](#materials)
- [Basics](#basics)
  - [Permission](#permission)
  - [Hello Docker](#hello-docker)
  - [How to build a image](#how-to-build-a-image)
  - [Dockerhub](#dockerhub)
  - [Private docker registry](#private-docker-registry)
  - [Basic Docker Commands](#basic-docker-commands)
  - [Dockerfile Instruction](#dockerfile-instruction)
  - [Advanced Docker Commands](#advanced-docker-commands)
- [Advanced](#advanced)
  - [Network](#network)
  - [User of docker container](#user-of-docker-container)

----

# Abstract

vmware, virtualbox 보다 훨씬 성능이 좋은 가상화 기술이다. 

# Materials

- [초보를 위한 도커 안내서 - 이미지 만들고 배포하기](https://subicura.com/2017/02/10/docker-guide-for-beginners-create-image-and-deploy.html)
  - building image, docker registry, deployment
- [도커 Docker 기초 확실히 다지기](https://futurecreator.github.io/2018/11/16/docker-container-basics/index.html)
  - 명쾌한 한장 요약
- [가장 빨리 만나는 Docker](http://pyrasis.com/docker.html)
  - 쉬운 한글 책
  - [src](https://github.com/pyrasis/dockerbook)

# Basics

## Permission

* [초보를 위한 도커 안내서 - 설치하고 컨테이너 실행하기](https://subicura.com/2017/01/19/docker-guide-for-beginners-2.html)

----

docker 는 root 권한이 필요하다. root 가 아닌 사용자가 sudo 없이 사용하려면 해당 사용자를 docker 그룹에 추가한다.

```bash
sudo usermod -aG docker $USER
sudo usermod -aG docker iamslash
```

## Hello Docker

Sinatra 를 사용한 ruby web app 을 docker 로 실행해보자. 먼저 `~/my/ruby/a` 에 `Gemfile, app.rb` 를 제작한다.

* Gemfile

```
source 'https://rubygems.org'
gem 'sinatra'
```

* app.rb

```ruby
require 'sinatra'
require 'socket'

get '/' do
  Socket.gethostname
end
```

그리고 실행한다.

```bash
cd ~/my/ruby/a
bundle install
bundle exec ruby app.rb
```

ruby 및 개발환경이 갖추어져 있지 않다면 실행이 불가능하다. 그러나 docker 를 사용하면 ruby 개발환경이 갖추어진 image 를 container 로 간단히 실행할 수 있다. `--rm` 옵션때문에 실행이 종료되면 container 가 자동으로 삭제된다.

```bash
docker run --rm \
-p 4567:4567 \
-v $PWD:/usr/src/app \
-w /usr/src/app \
ruby \
bash -c "bundle install && bundle exec ruby app.rb -o 0.0.0.0"
```

이제 브라우저로 `http://localhost:4567` 를 접속해본다.

## How to build a image

이제 앞에서 실행한 image 를 build 해보자. 먼저 다음과 같이 `my/docker/a/Dockerfile` 을 제작한다. [이곳](https://docs.docker.com/engine/reference/builder/) 은 Dockerfile reference 이다.

```Dockerfile
# install ubuntu
FROM       ubuntu:16.04
MAINTAINER iamslash@gmail.com
RUN        apt-get -y update

# install ruby
RUN apt-get -y install ruby
RUN gem install bundler

# copy sources
COPY . /usr/src/app

# install Gemfile
WORKDIR /usr/src/app
RUN     bundle install

# export port, run Sinatra
EXPOSE 4567
CMD    bundle exec ruby app.rb -o 0.0.0.0
```

그리고 다음과 같이 image 를 build 한다.

```bash
> cd ~/my/docker/a/
> docker build -t app .
> docker images
```

위의 Dockerfile 을 더욱 최적화 해보자. `ruby:2.3` image 를 pull 할 수 있기 때문에 `ubuntu` 부터 설치할 필요는 없다. docker 가 Dockerfile 의 instruction 을 실행할 때 결과가 이전과 같으면 cache hit 되어 빠르다. 따라서 이전과 같은 결과가 발생하도록 instruction 의 위치를 조절하거나 standard out 을 제거하도록 한다.

```Dockerfile
# install ubuntu, ruby
FROM ruby:2.3
MAINTAINER iamslash@gmail.com

# copy Gemfile, install packages
# optimized with cache
COPY Gemfile* /usr/src/app/
WORKDIR /usr/src/app
RUN bundle install --no-rdoc --no-ri

# copy sources
COPY . /usr/src/app

# export port, run Sinatra
EXPOSE 4567
CMD bundle exec ruby app.rb -o 0.0.0.0
```

```bash
docker run --rm \
-p 4567:4567 \
--name my-app
app 
```

## Dockerhub 

docker image 를 dockerhub 에 push 해보자.

```bash
> docker login
# image file specification:
#   [registry url]/[user id]/[image name]:[tag] 
# rename name and tag
# > docker src_image_name[:TAG] dst_image_name[:TAG]
docker tag app iamslash/iamslash-app:1
# push image
docker push iamslash/iamslash-app:1
```

## Private docker registry

dockerhub 는 private repository 가 유료이다. 무료 private docker registry 를 운영해보자.

```bash
> docker run -d \
-v c:\my\dockerregistry:/var/lib/registry \
-p 5000:5000 \
distribution/registry:2.6.0

> docker tag app localhost:5000/iamslash/iamslash-app:1
> docker push localhost:5000/iamslash/iamslash-app:1
> tree c:\my\docker\registry
```

## Basic Docker Commands

```bash
# print version 
> docker version
# pull hello-world image, run container from hello-world image
> docker run hello-world
# help
> docker help
# search image
> docker search <image-name>
# pull image
> docker pull <image-name>:<tag>
# list image
> docker images
# docker run <option> <execution-filename>
> docker run -it --name ubuntu ubuntu:latest /bin/bash
# list containers
> docker ps -a
# list containers with format
> docker ps -a --format "table {{.ID}}\t{{.Status}}\t{{.Names}}"
> docker ps -a --no-trunc --format "table {{.Names}}\t{{.Command}}"
# start container
> docker start <container-name>
# restart container
> docker restart <container-name>

# attach to container
# > docker attach <container-name>
# attach to container with command, argument
# > docker attach <container-name> <command> <argument>
# run echo command on ubuntu container
> docker attach ubuntu echo "hello world"

# stop container
#   sends a SIGTERM signal
> docker stop <container-name>
# remove container
> docker rm <container-name>
# remove image
# > docker rmi <image-name>:<tag>
> docker rmi ubuntu:latest

# build image
> docker build <option> <build-path>
  > mkdir A
  > cd A
  > emacs Dockerfile
  > docker build tag A:0.1 .
  > docker run --name A -d -p 80:80 -v /root/data:data A:0.1
# commit container  
> docker commit
  > docker commit -a "iamslash <iamslash@gmail.com>" -m "vim installed" ubuntu ubuntu:latest

# run /bin/bash on container and get a terminal
# > docker exec
> docker exec -it myubuntu /bin/bash
# get a terminal as root user
> docker exec -it -u root jenkins /bin/bash
```

## Dockerfile Instruction

```Dockerfile
## FROM
# 어떤 이미지를 기반으로 이미지를 생성할지 설정
FROM ubuntu:14.04

## MAINTAINER
# 이미지를 생성한 사람의 정보
MAINTAINER    David, Sun <iamslash@gmail.com>

## RUN
# 이미지를 빌드할때 스크립트 혹은 명령을 실행
# /bin/sh 로 실행
RUN apt-get install -y nginx
RUN echo "Hello Docker" > /tmp/hello
RUN curl -sSL https://golang.org/dl/go1.3.1.src.tar.gz | tar -v -C /usr/local -xz
RUN git clone https://github.com/docker/docker.git
# shell 없이 실행
RUN ["apt-get", "install", "-y", "nginx"]
RUN ["/user/local/bin/hello", "--help"]

## CMD
# 컨테이너를 시작할때 스크립트 혹은 명령을 실행한다. 딱 한번만 사용 가능하다.
# /bin/sh 로 실행
CMD touch /home/hello/hello.txt
# shell 없이 실행
CMD ["redis-server"]
CMD ["mysqld", "--datadir=/var/lib/mysql", "--user=mysql"]

## ENTRYPOINT
# 컨테이너를 시작할때 스크립트 혹은 명령을 실행한다. 딱 한번만 사용 가능하다.
# ENTRYPOINT 에 설정한 명령에 매개 변수를 전달하여 실행
# ENTRYPOINT 와 CMD 를 동시에 사용하면 CMD 는 agument 역할만 한다
ENTRYPOINT ["echo"]
CMD ["hello"]

## EXPOSE
# 호스트에 노출할 포트
EXPOSE 80
EXPOSE 443
EXPOSE 80 443

## ENV
# 환경 변수를 설정
ENV GOPATH /go
ENV PATH /go/bin:$PATH
ENV HELLO 1234
CMD echo $HELLO
# docker run 에서도 설정할 수 있다.
# > docker run -e HELLO=1234 app

## ADD
# 파일을 이미지에 추가
# 압축파일 해제, 파일 URL 도 사용가능하다
ADD hello-entrypoint.sh /entrypoint.sh
ADD hello-dir /hello-dir
ADD zlib-1.2.8.tar.gz /
ADD hello.zip /
ADD http://example.com/hello.txt /hello.txt
ADD *.txt /root/

## COPY
# 파일을 이미지에 추가
# ADD 와는 달리 COPY 는 압축 파일을 추가할 때 압축을 해제하지 않고 파일 URL 도 사용할 수 없다.
COPY hello-entrypoint.sh /entrypoint.sh
COPY hello-dir /hello-dir
COPY zlib-1.2.8.tar.gz /zlib-1.2.8.tar.gz
COPY *.txt /root/

## VOLUME
# 디렉터리의 내용을 컨테이너에 저장하지 않고 호스트에 저장하도록 설정
VOLUME /data
VOLUME ["/data", "/var/log/hello"]

# docker run 에서도 사용가능
# > docker run -v /prj/data:/data app

## USER
# 명령을 실행할 사용자 계정을 설정. RUN, CMD, ENTRYPOINT 가 USER 로 실행된다.
USER nobody
RUN touch /tmp/hello.txt

USER root
RUN touch /hello.txt
ENTRYPOINT /hello-entrypoint.sh

## WORKDIR
# RUN, CMD, ENTRYPOINT의 명령이 실행될 디렉터리를 설정
WORKDIR /root
RUN touch hello.txt

WORKDIR /tmp
RUN touch hello.txt

## ONBUILD
# 생성한 이미지를 기반으로 다른 이미지가 생성될 때 명령을 실행
# build event handler 이다.
ONBUILD RUN touch /hello.txt
ONBUILD ADD world.txt /world.txt
```

## Advanced Docker Commands

```bash
## attach
# 실행되고 있는 컨테이너에 표준 입력(stdin)과 표준 출력(stdout)을 연결
# docker attach <옵션> <컨테이너 이름, ID>
> docker run -it -d --name hello ubuntu:14.01 /bin/bash
> docker attach hello

## build
#  Dockerfile로 이미지를 생성
# docker build <옵션> <Dockerfile 경로>
$ docker build -t hello .
$ docker build -t hello /opt/hello
$ docker build -t hello ../../
$ docker build -t hello https://raw.githubusercontent.com/kstaken/dockerfile-examples/master/apache/Dockerfile
$ echo -e "FROM ubuntu:14.04\nRUN apt-get update" | sudo docker build -t hello -
$ cat Dockerfile | sudo docker build -t hello -
$ docker build -t hello - < Dockerfile

## commit
# 컨테이너의 변경 사항을 이미지로 생성
# docker commit <옵션> <컨테이너 이름, ID> <저장소 이름>/<이미지 이름>:<태그>
$ docker commit -a "iamslash <iamslash@gmail.com>" -m "add hello.txt" hello hello:0.2

## cp
# 컨테이너의 디렉터리나 파일을 호스트로 복사
# docker cp <컨테이너 이름>:<경로> <호스트 경로>
$ docker cp hello:/hello.txt .

## create
# 이미지로 컨테이너를 생성
# docker create <옵션> <이미지 이름, ID> <명령> <매개 변수>
$ docker create -it --name hello ubuntu:14.04 /bin/bash
$ docker start hello
$ docker attach hello

## diff
# 컨테이너에서 변경된 파일을 확인
# docker diff <컨테이너 이름, ID>
$ docker diff hello

## events
# Docker 서버에서 일어난 이벤트를 실시간으로 출력
# docker events
$ docker events

## exec
# 외부에서 컨테이너 안의 명령을 실행
# docker export <옵션> <컨테이너 이름, ID> <명령> <매개 변수>
$ docker exec -it hello /bin/bash
$ docker exec hello apt-get update
$ docker exec hello apt-get install -y redis-server
$ docker exec -d hello redis-server
$ sudo docker top hello ax

## export
# 컨테이너의 파일시스템을 tar 파일로 저장
# docker export <컨테이너 이름, ID>
$ docker export hello > hello.tar

## history
# 이미지의 히스토리를 출력
# docker history <옵션> <이미지 이름, ID>
$ docker history hello

## images
# 이미지 목록을 출력
# docker images <옵션> <이미지 이름>
docker images ubuntu
echo -e "FROM ubuntu:14.04\nRUN apt-get update" | sudo docker build -
# 이름이 없는 이미지 출력
docker images -f "dangling=true"
# 이름 없는 이미지를 모두 삭제
sudo docker rmi $(sudo docker images -f "dangling=true" -q)

## import
# tar 파일(.tar, .tar.gz, .tgz, .bzip, .tar.xz, .txz)로 압축된 파일시스템으로부터 이미지를 생성
# docker import <tar 파일의 URL 또는 -> <저장소 이름>/<이미지 이름>:<태그>
$ docker import http://example.com/hello.tar.gz hello
$ cat hello.tar | docker import - hello
# 현재 디렉토리의 내용을 바로 이미지로 생성
$ sudo tar -c . | sudo docker import - hello

## info
# 현재 시스템 정보와 Docker 컨테이너, 이미지 개수, 설정 등을 출력
# docker info
$ docker info

## inspect
# 컨테이너와 이미지의 세부 정보를 JSON 형태로 출력
# docker inspect <옵션> <컨테이너 또는 이미지 이름, ID>
# 이미지의 세부 정보에서 아키텍처와 OS를 출력
$ docker inspect -f "{{ .Architecture }} {{ .Os }}" ubuntu:14.04
# 컨테이너의 IP 주소를 출력
$ docker inspect -f "{{ .NetworkSettings.IPAddress }}" hello
# 세부 정보의 일부 내용을 JSON 형태로 출력
$ docker inspect -f "{{json .NetworkSettings}}" hello
# 컨테이너의 세부 정보에서 특정 부분만 추출하여 원하는 포맷으로 출력
$ docker inspect -f '{{range $p, $conf := .NetworkSettings.Ports}} {{$p}} -> {{(index $conf 0).HostPort}} {{end}}' hello
80/tcp -> 80  8080/tcp -> 8080
# .NetworkSettings.Ports
# "Ports": {
#     "80/tcp": [
#         {
#             "HostIp": "0.0.0.0",
#             "HostPort": "80"
#         }
#     ],
#     "8080/tcp": [
#         {
#             "HostIp": "0.0.0.0",
#             "HostPort": "8080"
#         }
#     ]
# }
# {{range $p, $conf := .NetworkSettings.Ports}} 으로 .NetworkSettings.Ports 의 내용을 순회하면서 $p, $conf 에 저장. 그리고 $p는 그대로 출력하고, $conf 배열에서 첫 번째 항목(index $conf 0) 의 .HostPort 를 출력

## kill
# sends a SIGKILL signal.
# docker kill <옵션> <컨테이너 이름, ID>
$ docker kill hello

## load
# tar 파일로 이미지를 생성
# docker load <옵션>
$ sudo docker load < ubuntu.tar

## login
# Docker 레지스트리에 로그인
# docker login <옵션> <Docker 레지스트리 URL>
$ docker login

## logout
# Docker 레지스트리에서 로그아웃
# docker logout <Docker 레지스트리 서버 URL>
$ docker logout

## logs
# 컨테이너의 로그를 출력
# docker logs <컨테이너 이름, ID>
$ docker logs hello
$ docker logs hello -f

## port
# 컨테이너에서 포트가 열려 있는지 확인
# docker port <컨테이너 이름, ID> <포트>
$ docker port hello 80

## pause
# 컨테이너에서 실행되고 있는 모든 프로세스를 일시 정지
# docker pause <컨테이너 이름, ID>
$ docker pause hello

## ps
# 컨테이너 목록을 출력
# docker ps <옵션>
$ docker ps -a
# --no-trunc: 정보가 끊기지 않고 모두 출력
$ docker ps -a --no-trunc

## pull
#  Docker 레지스트리에서 이미지를 다운로드
# docker pull <옵션> <저장소 이름>/<이미지 이름>:<태그>
$ docker pull centos
$ docker pull ubuntu:14.04
$ docker pull registry.hub.docker.com/ubuntu:14.04
$ docker pull exampleuser/hello:0.1
$ docker pull 192.168.0.39:5000/hello:0.1
$ docker pull exampleregistry.com:5000/hello:0.1

## push
# Docker 레지스트리에 이미지를 업로드
# docker push <저장소 이름>/<이미지 이름>:<태그>
$ docker tag hello:0.1 exampleuser/hello:0.1
$ docker pull exampleuser/hello:0.1
$ docker tag hello:0.1 192.168.0.39:5000/hello:0.1
$ docker pull 192.168.0.39:5000/hello:0.1
$ docker tag hello:0.1 exampleregistry.com:5000/hello:0.1
$ docker pull exampleregistry.com:5000/hello:0.1

## restart
# 컨테이너를 재시작
# docker restart <옵션> <컨테이너 이름, ID>
$ docker restart hello

## rm
# 컨테이너를 삭제
# docker rm <옵션> <컨테이너 이름, ID>
$ docker rm -l hello/db

## rmi
# 이미지를 삭제
# docker rmi <저장소 이름>/<이미지 이름, ID>:<태그>
$ sudo docker rmi hello
$ sudo docker rmi hello:0.1
$ sudo docker rmi exampleuser/hello:0.1
$ sudo docker rmi 192.168.0.39:5000/hello:0.1
$ sudo docker rmi exampleregistry.com:5000/hello:0.1
# 실행되고 있는 이미지를 강제로 삭제
$ docker run -i -t -d --name hello ubuntu:14.04 /bin/bash
$ docker rmi -f hello
# 한번에 삭제
$ docker rmi `sudo docker images -aq`
$ docker rmi $(sudo docker images -aq)

## run
# 이미지로 컨테이너를 생성
# docker run <옵션> <이미지 이름, ID> <명령> <매개 변수>
$ docker run -i -t ubuntu:14.04 /bin/bash
# --cap-add 옵션을 사용하여 컨테이너에서 SYS_ADMIN Capability 를 사용
$ docker run -it --rm --name hello --cap-add SYS_ADMIN ubuntu:14.04 bash
$ docker -p 192.168.0.10:80:8080 ubuntu:14.04 bash
# --expose 옵션을 사용하여 80 포트를 호스트에만 연결하고 외부에 노출하지 않는다. 호스트와 --link 옵션으로 연결한 컨테이너에서만 접속가능.
$ docker run --expose 80 ubuntu:14.04 bash
# 환경변수 설정
$ docker run -it -e HELLO_VAR="Hello World" ubuntu:14.04 bash
# 환경변수 파일 설정, -e 옵션이 파일보다 우선순위가 높다
$ docker run -it --env-file ./example-env.sh -e HELLO="Hello World" ubuntu:14.04 bash
# bash 환경변수 설정
$ EXAMPLE=10 docker run -it --env-file ./example-env.sh ubuntu:14.04 bash
# --link 를 사용하여 Redis 컨테이너와 연결
$ docker run -d --name cache redis:latest
$ docker run -it --link cache:cache ubuntu:14.04 bash

## save
# 이미지를 tar 파일로 저장
# docker save <옵션> <이미지 이름>:<태그>
$ docker save -o nginx.tar nginx:latest
$ docker save -o redis.tar redis:latest
$ docker save ubuntu:14.04 > ubuntu14.04.tar
$ docker save ubuntu > ubuntu.tar

## search
# Docker Hub에서 이미지를 검색
# docker search <옵션> <검색어>
$ docker search -s 10 ubuntu

## start
# 컨테이너를 시작
# docker start <옵션> <컨테이너 이름, ID>
$ docker run -d --name hello ubuntu:14.04 /bin/bash -c "while true; do echo Hello World; sleep 1; done"

## stop
# 컨테이너를 정지
# docker stop <옵션> <컨테이너 이름, ID>
$ docker run -d --name hello ubuntu:14.04 /bin/bash -c "while true; do echo Hello World; sleep 1; done"
$ docker stop -t 0 hello

## tag
# 이미지에 태그를 설정
# docker tag <옵션> <이미지 이름>:<태그> <저장소 주소, 사용자명>/<이미지 이름>:<태그>
$ echo "FROM ubuntu:14.04" | docker build -t hello:latest -
$ docker tag hello:latest hello:0.1
$ docker tag hello:latest exampleuser/hello:0.1
$ docker tag hello:latest 192.168.0.39/hello:0.1
$ docker images

## top
# 컨테이너에서 실행되고 있는 프로세스 목록을 출력
# docker top <컨테이너 이름, ID> <ps 옵션>
$ docker top hello aux

## unpause
# 일시 정지된 컨테이너를 다시 시작
# docker unpause <컨테이너 이름, ID>
$ docker run -i -t -d --name hello ubuntu:14.04 /bin/bash
$ docker pause hello
$ docker unpause hello

## version
# Docker 버전을 출력
# docker version
$ docker version

## wait
# 컨테이너가 정지될 때까지 기다린 뒤 Exit Code를 출력
# docker wait <컨테이너 이름, ID>
$ docker run -d --name hello redis:latest
$ docker wait hello
```

# Advanced

```bash
# remove containers which stopped
> sudo docker ps -a | grep Exit | cut -d ' ' -f 1 | xargs sudo docker rm

# 정지한 컨테이너, 연결이 끊긴 볼륨, 연결이 끊긴 네트워크, dangling 이미지가 삭제.
> docker system prune
# docker disk free
> docker system df

# get dangling image id
> docker images -f "dangling=true" -q
# remove image
> docker rmi $(docker images -f "dangling=true" -q)

# save image
> docker save img_name > img_name.tar
> docker save -o img_name.tar img_name
# load image
> docker load < img_name.tar
> docker load -i img_name.tar

# export container
> docker export container_name > container_name.tar
# import container
> docker import container_name.tar

# restart policy (no | on-failure | always | )
#   https://rampart81.github.io/post/docker_commands/
#   docker container 가 멈추었을때 다시시작하는 옵션
#   --restart=on-failure:5
#     optional restart count
#   --restart unless-stopped
#     restart the container unless it is explicitly stopped or Docker itself is stopped or restarted.
#   --restart always
#     always restart the container if it stops.
sudo docker run --detach \
  --hostname gitlab.example.com \
  --publish 443:443 --publish 80:80 --publish 22:22 \
  --name gitlab \
  --restart always \
  --volume /srv/gitlab/config:/etc/gitlab \
  --volume /srv/gitlab/logs:/var/log/gitlab \
  --volume /srv/gitlab/data:/var/opt/gitlab \
  gitlab/gitlab-ce:latest
```

## Network

* [Docker Network 구조(1) - docker0와 container network 구조](https://bluese05.tistory.com/15)

----

There are 4 kinds of network options in docker. (host, bridge, container, null)

* bridge (default)
  * docker daemon create a network bridge "docker0" and docker containers make a network interface binded the bridge "docker0"

```bash
$ docker network inspect bridge
[
    {
        "Name": "bridge",
        "Id": "988a07e544f2202a05fe010539d909f26e60f0c0013af07ae2a5c44f157fc9f5",
        "Created": "2019-12-27T23:10:10.381976258Z",
        "Scope": "local",
        "Driver": "bridge",
        "EnableIPv6": false,
        "IPAM": {
            "Driver": "default",
            "Options": null,
            "Config": [
                {
                    "Subnet": "172.17.0.0/16",
                    "Gateway": "172.17.0.1"
                }
            ]
        },
        "Internal": false,
        "Attachable": false,
        "Ingress": false,
        "ConfigFrom": {
            "Network": ""
        },
        "ConfigOnly": false,
        "Containers": {},
        "Options": {
            "com.docker.network.bridge.default_bridge": "true",
            "com.docker.network.bridge.enable_icc": "true",
            "com.docker.network.bridge.enable_ip_masquerade": "true",
            "com.docker.network.bridge.host_binding_ipv4": "0.0.0.0",
            "com.docker.network.bridge.name": "docker0",
            "com.docker.network.driver.mtu": "1500"
        },
        "Labels": {}
    }
]
```

* host
  * use same network with host OS.

```bash
$ docker run --net=host httpd web01
# show network interfaces same with host OS
$ docker exec web01 ip addr show
# there is no binding with the bridge "docker0"
$ brctl show
# There is no IP address for web01 because it does not have network environments. 
$ docker network inspect host
```

* container
  * can communicate with other containers.

```bash
$ docker run --name web02 -d httpd
$ docker ps -a
# web03 can communicate with web02 with net option
$ docker run --name web03 --net=container:e1b4a085348e -d httpd
$ docker ps -a
# web03 has same IP, MAC with web02 
$ docker exec web02 ip addr show
$ docker exec web03 ip addr show
# There is no inteface for web03 because it uses the interface of web02
$ brctl show
$ docker network inspect bridge
```

* null
  * make isolated network environments but no network interface.

```bash
$ docker run --name web04 --net=none -d httpd
# There is just loopback no eth0 interface. So web04 cannot communicate with it's outside
$ docker exec web04 ip addr show
```

* connect to host from the container
  * [Access MacOS host from a docker container](https://medium.com/@balint_sera/access-macos-host-from-a-docker-container-e0c2d0273d7f)
  * use `host.docker.internal` as ip addr in a container


## User of docker container

* [How to set user of Docker container](https://codeyarns.com/2017/07/21/how-to-set-user-of-docker-container/)

