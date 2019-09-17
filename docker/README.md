# Abstract

기존의 vmware, virtualbox 보다 훨씬 성능이 좋은 가상화 기술이다.

# Materials

- [초보를 위한 도커 안내서 - 이미지 만들고 배포하기](https://subicura.com/2017/02/10/docker-guide-for-beginners-create-image-and-deploy.html)
  - building image, docker registry, deployment
- [도커 Docker 기초 확실히 다지기](https://futurecreator.github.io/2018/11/16/docker-container-basics/index.html)
  - 명쾌한 한장 요약
- [가장 빨리 만나는 Docker](http://pyrasis.com/docker.html)
  - 쉬운 한글 책

# Usage

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

# Dockerhub 

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

# Private docker registry

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

## Useful Commands

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
# container 를 하나 실행하자.
# my-ubuntu container 를 실행하고 bash shell 을 획득하자.
# docker run <option> <execution-filename>
> docker run -i -t --name my-ubuntu ubuntu:latest /bin/bash
# list containers
> docker ps -a
# start container
> docker start <container-name>
# restart container
> docker restart <container-name>

# attatch to container
# > docker attach <container-name>
# attatch to container with command, argument
# > docker attach <container-name> <command> <argument>
# my-ubuntu container 에서 echo 명령어를 실행해보자.
> docker attach my-ubuntu echo "hello world"

# stop container
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
  > docker run --name myA -d -p 80:80 -v /root/data:data A:0.1
# commit container  
> docker commit
  > docker commit -a "iamslash <iamslash@gmail.com>" -m "vim installed" my-ubuntu ubuntu:latest

# get a bash from a alive container
# > docker exec
> docker exec -it myubuntu /bin/bash
```

# Advanced

```bash
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
```