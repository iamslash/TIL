# intro

- 가상화 기술의 하나이다. 기존의 vmware, virtualbox보다 훨씬 성능이 좋다.

# usage

- docker help
- docker search <image-name>
- docker pull <image-name>:<tag>
- docker images
- docker run <option> <execution-filename>
  - docker run -i -t --name myubuntu ubuntu:latest /bin/bash
- docker ps -a
- docker start <container-name>
- docker restart <container-name>
- docker attach <container-name>
- docker attach <container-name> <command> <argument>
  - docker attach myubuntu echo "hello world"
- docker stop <container-name>
- docker rm <container-name>
- docker rmi <image-name>:<tab>
- docker build <option> <build-path>
  - mkdir A
  - cd A
  - emacs Dockerfile
  - docker build tag A:0.1 .
  - docker run --name myA -d -p 80:80 -v /root/data:data A:0.1
- docker commit
  - docker commit -a "iamslash <iamslash@gmail.com>" -m "vim installed" myubuntu ubuntu:latest
- docker exec
  - docker exec -it myubuntu /bin/bash
    - 기존에 뭔가 실행하고 있는 container에 terminal을 획득해보자.

# reference


- [가장 빨리 만나는 Docker](http://pyrasis.com/docker.html)
