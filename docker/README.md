# Abstract

- 가상화 기술의 하나이다. 기존의 vmware, virtualbox보다 훨씬 성능이 좋다.

# Materials

- [도커 Docker 기초 확실히 다지기](https://futurecreator.github.io/2018/11/16/docker-container-basics/index.html)
  - 명쾌한 한장 요약
- [가장 빨리 만나는 Docker](http://pyrasis.com/docker.html)

# Usage

```bash
> docker version
> docker run hello-world
> docker help
> docker search <image-name>
> docker pull <image-name>:<tag>
> docker images
# container 를 하나 실행하자.
> docker run <option> <execution-filename>
# my-ubuntu container 를 실행하고 bash shell 을 획득하자.
> docker run -i -t --name my-ubuntu ubuntu:latest /bin/bash
> docker ps -a
> docker start <container-name>
> docker restart <container-name>
> docker attach <container-name>
> docker attach <container-name> <command> <argument>
# my-ubuntu container 에서 echo 명령어를 실행해보자.
> docker attach my-ubuntu echo "hello world"
> docker stop <container-name>
# container 를 삭제하자
> docker rm <container-name>
> docker rmi <image-name>:<tag>
# image 를 삭제하자
> docker build <option> <build-path>
  > mkdir A
  > cd A
  > emacs Dockerfile
  > docker build tag A:0.1 .
  > docker run --name myA -d -p 80:80 -v /root/data:data A:0.1
> docker commit
  > docker commit -a "iamslash <iamslash@gmail.com>" -m "vim installed" myubuntu ubuntu:latest
> docker exec
  # 기존에 뭔가 실행하고 있는 container에 terminal을 획득해보자.
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