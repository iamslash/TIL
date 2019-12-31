# Materials

* [Docker Swarm 을 이용한 Container Orchestration 환경 만들기](https://tech.osci.kr/2019/02/13/59736201/)
* [Swarm mode overview](https://docs.docker.com/engine/swarm/)
* [Docker Swarm Mode Tutorials](https://github.com/docker/labs/tree/master/swarm-mode)

# Architecture 

Docker-swarm is consisted of Master and Worker nodes. Docker Swarm is installed automatically when using Docker for Mac or Docker for Windows.

![](https://docs.docker.com/engine/swarm/images/swarm-diagram.png)

# Basics

* init swarm, create the service

```bash
# init docker swarm
$ docker swarm init --advertise-addr 192.168.0.176

# add master node
$ docker swarm join-token manager

# add worker node
$ docker swarm join \
>     --token SWMTKN-1-2m3tqsm8ly45vpd5i80p4bkor5zaohfmultu4cdnvfpg8yxmuk-9ghru6puwdvqms3bn7zqtiyvt \
>     192.168.0.176:2377
$ docker swarm join \
>     --token SWMTKN-1-2m3tqsm8ly45vpd5i80p4bkor5zaohfmultu4cdnvfpg8yxmuk-9ghru6puwdvqms3bn7zqtiyvt \
>     192.168.0.176:2377

# show worker nodes
$ docker node ls

# create a service
$ docker service create --name web httpd
$ docker service ls

# show service status
$ docker service ps web

# scale out
$ docker service scale web=3
$ docker service ls
$ for i in $(cat /etc/hosts | grep manager| awk '{print $1}')
> do
> ssh root@$i "docker ps -a"
> done
$ docker service ps web

# remove service
$ docker service rm web
$ docker service ls
$ docker service ps web
```

* run php docker image

```bash
# modify config for private docker registry
$ vi /etc/docker/daemon.json
{
  "insecure-registries" : ["manager1.iamslash.com:5000"]
}

# create a docker registry service
$ docker service create --name registry -p 5000:5000 registry
z3gl3pie7xm9vjfyetot9zi3q

# prepare DOCKERFILE stuff
$ tree
.
├── Dockerfile
├── README.md
└── htdocs
    └── index.php

1 directory, 3 files

$ vi Dockerfile
FROM php:7.2-apache
MAINTAINER chhanz <chhan@osci.kr>

ADD htdocs/index.php /var/www/html/index.php

EXPOSE 80

$ cat htdocs/index.php
<html>
<body>
<center>
<b>
<?php
$host=gethostname();
echo "Container Name : ";
echo $host;
?>
<p> Image Version : orignal</p>
</b>
</center>
</body>
</html>

# build a docker image
$ docker build -t phpdemo:v1 .

# push the docker image
$ docker images
$ docker tag phpdemo:v1 manager1.iamslash.com:5000/phpdemo:v1
$ docker images
$ docker push manager1.iamslash.com:5000/phpdemo:v1

# deploy the service
$ docker service create --name phpdemo -p 80:80 manager1.iamslash.com:5000/phpdemo:v1
$ docker service ls
$ docker service ps phpdemo
$ docker service ls
$ docker service ps

# replicate the docker service
$ docker service scale phpdemo=3
$ docker service ls
$ docker service ps phpdemo

# update the index.php
$ cat htdocs/index.php
<html>
<body>
<center>
<b>
<?php
$host=gethostname();
echo "Container Name : ";
echo $host;
?>
<p> Image Version : Update Version v2</p>
</b>
</center>
</body>
</html>

# build, push the docker image again
$ docker build -t phpdemo:v2 .
$ docker images
$ docker tag phpdemo:v2 manager1.iamslash.com:5000/phpdemo:v2
$ docker images
$ docker push manager1.iamslash.com:5000/phpdemo:v2

# rolling update
$ docker service update --update-parallelism 1 --image manager1.iamslash.com:5000/phpdemo:v2 phpdemo
$ docker service ps phpdemo

# rollback
$ docker service update --rollback phpdemo
$ docker service ps phpdemo
```
