- [Abstract](#abstract)
- [Materials](#materials)
- [Basics](#basics)
  - [Permission](#permission)
  - [Install](#install)
  - [Versioning](#versioning)
  - [Making a image](#making-a-image)
  - [Dockderizing moniwiki](#dockderizing-moniwiki)
  - [Upload to Docker Hub](#upload-to-docker-hub)
  - [Hello Docker](#hello-docker)
  - [Private docker registry](#private-docker-registry)
  - [Dockerizing again](#dockerizing-again)
  - [How to build a image](#how-to-build-a-image)
  - [Basic Docker Commands](#basic-docker-commands)
  - [Dockerfile Instruction](#dockerfile-instruction)
  - [Advanced Docker Commands](#advanced-docker-commands)
- [Advanced](#advanced)
  - [Process ID of Docker container is 1](#process-id-of-docker-container-is-1)
  - [Useful commands](#useful-commands)
  - [Network](#network)
  - [User of docker container](#user-of-docker-container)

----

# Abstract

vmware, virtualbox 보다 훨씬 성능이 좋은 가상화 기술이다. 

# Materials

- [도커(Docker) 입문편 컨테이너 기초부터 서버 배포까지](https://www.44bits.io/ko/post/easy-deploy-with-docker#%EC%8B%A4%EC%A0%84-%EB%8F%84%EC%BB%A4-%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%A1%9C-%EC%84%9C%EB%B2%84-%EC%95%A0%ED%94%8C%EB%A6%AC%EC%BC%80%EC%9D%B4%EC%85%98-%EB%B0%B0%ED%8F%AC%ED%95%98%EA%B8%B0)
  - [도커 컨테이너는 가상머신인가요? 프로세스인가요?](https://www.44bits.io/ko/post/is-docker-container-a-virtual-machine-or-a-process)
  - [컨테이너 기초 - chroot를 사용한 프로세스의 루트 디렉터리 격리](https://www.44bits.io/ko/post/change-root-directory-by-using-chroot)
  - [정적 링크 프로그램을 chroot와 도커(Docker) scratch 이미지로 실행하기](https://www.44bits.io/ko/post/static-compile-program-on-chroot-and-docker-scratch-image)
  - [만들면서 이해하는 도커(Docker) 이미지의 구조,도커 이미지 빌드 원리와 Overayfs](https://www.44bits.io/ko/post/how-docker-image-work)
  - [도커 컴포즈를 활용하여 완벽한 개발 환경 구성하기, 컨테이너 시대의 Django 개발환경 구축하기](https://www.44bits.io/ko/post/almost-perfect-development-environment-with-docker-and-docker-compose)
  - [아마존 엘라스틱 컨테이너 서비스(ECS) 입문, 도커(Docker) 컨테이너 오케스트레이션](https://www.44bits.io/ko/post/container-orchestration-101-with-docker-and-aws-elastic-container-service)
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

## Install

* [도커(Docker) 입문편 컨테이너 기초부터 서버 배포까지](https://www.44bits.io/ko/post/easy-deploy-with-docker#%EC%8B%A4%EC%A0%84-%EB%8F%84%EC%BB%A4-%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%A1%9C-%EC%84%9C%EB%B2%84-%EC%95%A0%ED%94%8C%EB%A6%AC%EC%BC%80%EC%9D%B4%EC%85%98-%EB%B0%B0%ED%8F%AC%ED%95%98%EA%B8%B0)

----

```bash
$ curl -s https://get.docker.com | sudo sh
$ docker -v
$ cat /etc/apt/sources.list.d/docker.list
$ dpkg --get-selections | grep docker
```

docker 는 docker-ce, docker-ce-cli 로 구성된다. ce 는 community edition 을 의미한다. docker-ce 는 server 이고 REST api 도 제공한다. command line 은 docker-ce-cli 가 실행한다. docker-ce-cli 는 docker-ce server 와 통신한다.

```bash
$ docker ps
Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Get http://%2Fvar%2Frun%2Fdocker.sock/v1.40/containers/json: dial unix /var/run/docker.sock: connect: permission denied

# To prevent error use sudo
# If you want to use docker without sudo, Add $USER to docker group.
$ sudo docker ps
$ sudo usermod -aG docker $USER
$ sudo su - $USER
$ docker ps

# Find out registry
$ docker info | grep Registry
 Registry: https://index.docker.io/v1/
```

한가지 명심할 것은  이미지는 파일들의 집합이고, 컨테이너는 이 파일들의 집합 위에서 실행된 특별한 프로세스이다. 즉, Docker container 는 process 이다.

## Versioning

Docker container 의 변경사항을 image 에 반영할 수 있다.

```bash
$ docker pull ubuntu:bionic
$ docker run -it --rm ubuntu:bionic bash
> git --version
```

Host OS 의 shell 에서 docker 의 변경사항을 확인해 보자.

```bash
$ docker ps
$ docker diff 3bc6d0c2d284
...
```

Guest OS 의 shell 에서 git 을 설치해보자.

```bash
$ apt update
$ apt install -y git
$ git --version
```

다시 Host OS 의 shell 에서 docker 의 변경사항을 확인해 보자.

```bash
$ docker diff 3bc6d0c2d284 | head
...
# Let's commit modification
$ docker commit 65d60d3dd306 ubuntu:git

$ docker run -it --rm ubuntu:git bash
> git --version
> exit
$ docker rmi ubuntu:git
```

## Making a image

먼저 Dockerfile 을 만들어 보자.

```bash
$ mkdir A
$ cd A
$ vim Dockerfile
FROM ubuntu:bionic
RUN apt-get update
RUN apt-get install -y git

$ docker build -t ubuntu:git-from-dockerfile .
$ docker run -it ubuntu:git-from-dockerfile bash
> git --version
```

## Dockderizing moniwiki

```bash
$ git clone https://github.com/nacyot/docker-moniwiki.git
$ cd docker-moniwiki/moniwiki
$ cat Dockerfile
FROM ubuntu:14.04

RUN apt-get update &&\
  apt-get -qq -y install git curl build-essential apache2 php5 libapache2-mod-php5 rcs

WORKDIR /tmp
RUN \
  curl -L -O https://github.com/wkpark/moniwiki/archive/v1.2.5p1.tar.gz &&\
  tar xf /tmp/v1.2.5p1.tar.gz &&\
  mv moniwiki-1.2.5p1 /var/www/html/moniwiki &&\
  chown -R www-data:www-data /var/www/html/moniwiki &&\
  chmod 777 /var/www/html/moniwiki/data/ /var/www/html/moniwiki/ &&\
  chmod +x /var/www/html/moniwiki/secure.sh &&\
  /var/www/html/moniwiki/secure.sh

RUN a2enmod rewrite

ENV APACHE_RUN_USER www-data
ENV APACHE_RUN_GROUP www-data
ENV APACHE_LOG_DIR /var/log/apache2

EXPOSE 80

CMD bash -c "source /etc/apache2/envvars && /usr/sbin/apache2 -D FOREGROUND"

$ docker build -t iamslash/moniwiki:latest .
$ docker run -d -p 9999:80 iamslash/moniwiki:latest
# http://127.0.0.1:9999/moniwiki/monisetup.php

# docker 를 build 하는 과정에서 생성된 중간 image 들을 확인할 수 있다.
$ docker hisotry moniwiki:latest
IMAGE               CREATED             CREATED BY                                      SIZE                COMMENT
408c35f3d162        7 hours ago         /bin/sh -c #(nop)  CMD ["/bin/sh" "-c" "bash...   0B
f68911b22856        7 hours ago         /bin/sh -c #(nop)  EXPOSE 80                    0B
df4bd9e4dd7e        7 hours ago         /bin/sh -c #(nop)  ENV APACHE_LOG_DIR=/var/l...   0B
dee2cb60f1cc        7 hours ago         /bin/sh -c #(nop)  ENV APACHE_RUN_GROUP=www-...   0B
a1f1247b98cb        7 hours ago         /bin/sh -c #(nop)  ENV APACHE_RUN_USER=www-d...   0B
98a0ed3df283        7 hours ago         /bin/sh -c a2enmod rewrite                      30B
48926b3b3da0        7 hours ago         /bin/sh -c curl -L -O https://github.com/wkp...   7.32MB
dbdc86a08299        7 hours ago         /bin/sh -c #(nop) WORKDIR /tmp                  0B
becdcac5d788        7 hours ago         /bin/sh -c apt-get update &&  apt-get -qq -y...   184MB
6e4f1fe62ff1        3 months ago        /bin/sh -c #(nop)  CMD ["/bin/bash"]            0B
<missing>           3 months ago        /bin/sh -c mkdir -p /run/systemd && echo 'do...   7B
<missing>           3 months ago        /bin/sh -c set -xe   && echo '#!/bin/sh' > /...   195kB
<missing>           3 months ago        /bin/sh -c [ -z "$(apt-get indextargets)" ]     0B
<missing>           3 months ago        /bin/sh -c #(nop) ADD file:276b5d943a4d284f8...   196MB
```

images 가 <missing> 인 것은 base image 의 내용이다. 중간 image 는 local machine 에서 build 한 것만 확인할 수 있다.

## Upload to Docker Hub

먼저 docker registry 를 확인해 본다.

```bash
$ docker info | grep Registry
 Registry: https://index.docker.io/v1/
```

`https://index.docker.io/v1/` 가 docker-hub 의 API server 주소이다. image 의 full-name 에는 docker registry 의 주소가 포함되어 있따. 예를들어 `ubuntu:bionic` image 의 full-name 은 `docker.io/library/ubuntu:bionic` 이다. 

```bash
$ docker pull docker.io.library/ubuntu:bionic
```

image 의 full-name 은 다음과 같이 4 부분으로 구성된다.

* docker-hub api server address `docker.io`
* name-space `library`
* image name `ubuntu`
* tag `bionic`

docker push 를 하려면 docker registry 에 login 해야 한다.

```bash
$ docker login
# change image name including namespace
$ docker tag likechad/moniwiki:latest iamslash/miniwiki:latest

# push it
$ docker push iamslash/moniwiki:latest
```

## Hello Docker

다음은 간단한 command line 으로 centos 에서 bash 를 실행하는 방법이다.

```bash
$ docker run -it --rm centos:latest bash
Unable to find image 'centos:latest' locally
latest: Pulling from library/centos
8d30e94188e7: Pull complete
Digest: sha256:2ae0d2c881c7123870114fb9cc7afabd1e31f9888dac8286884f6cf59373ed9b
Status: Downloaded newer image for centos:latest

# show centos version
$ cat /etc/redhat-release
CentOS Linux release 8.1.1911 (Core)
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

## Dockerizing again

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

## Process ID of Docker container is 1

```bash
$ echo $$
5673

$ docker run -it --rm ubuntu:latest bash
> echo $$
> 1
# Can't kill process 1 in container
> kill -9 1
>
```


## Useful commands

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

