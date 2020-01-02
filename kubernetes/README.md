- [Abstract](#abstract)
- [Materials](#materials)
- [Architecture](#architecture)
  - [Overview](#overview)
  - [Kubernetes Components](#kubernetes-components)
    - [Master](#master)
    - [Node](#node)
    - [Addons](#addons)
- [Install](#install)
  - [Install on Win64](#install-on-win64)
  - [Install on macOS](#install-on-macos)
- [Basic](#basic)
  - [Useful Commands](#useful-commands)
  - [Launch Single Pod](#launch-single-pod)
  - [Lunach Pods with livnessprobe, readynessprobe](#lunach-pods-with-livnessprobe-readynessprobe)
    - [key commands](#key-commands)
    - [Launch Simple Pod](#launch-simple-pod)
    - [Launch Simple Pod with LivenessProbe](#launch-simple-pod-with-livenessprobe)
    - [Launch Simple Pod with ReadinessProbe](#launch-simple-pod-with-readinessprobe)
    - [Launch Simple Pod with HealthCheck](#launch-simple-pod-with-healthcheck)
    - [Launch Simple Pod with Multi Containers](#launch-simple-pod-with-multi-containers)
    - [Delete All resources](#delete-all-resources)
  - [Launch Replicaset](#launch-replicaset)
    - [Launch Simple Replicaset](#launch-simple-replicaset)
    - [Launch ReplicaSet Scale out](#launch-replicaset-scale-out)
  - [Launch Deployment](#launch-deployment)
    - [Launch Simple Deployment](#launch-simple-deployment)
    - [Launch Deployment with RollingUpdate](#launch-deployment-with-rollingupdate)
  - [Launch Service](#launch-service)
    - [Launch Simple Service](#launch-simple-service)
    - [Launch Service with NodePort](#launch-service-with-nodeport)
  - [Launch LoadBalancer](#launch-loadbalancer)
    - [Launch Simple LoadBalancer](#launch-simple-loadbalancer)
    - [????](#)
    - [???](#)
    - [???](#1)
  - [Launch Ingress](#launch-ingress)
    - [Launch Simple Ingress](#launch-simple-ingress)
    - [????](#1)
  - [Launch Horizontal Pod Autoscaler](#launch-horizontal-pod-autoscaler)
    - [Launch Simple Horizontal Pod Autoscaler](#launch-simple-horizontal-pod-autoscaler)
  - [Launch Kubernetes Dashboard](#launch-kubernetes-dashboard)
- [Authorization](#authorization)
- [AWS EKS](#aws-eks)
- [Dive Deep](#dive-deep)
  - [controller](#controller)

----

# Abstract

Kubernetes 는 여러개의 Container 들을 협업시킬 수 있는 도구이다. 

# Materials

* [Kubernetes in Action](http://acornpub.co.kr/book/k8s-in-action)
  * [src](https://github.com/luksa/kubernetes-in-action?files=1)
* [CNCF @ youtube](https://www.youtube.com/channel/UCvqbFHwN-nwalWPjPUKpvTA)
  * Cloud Native Computing Foundation
* [Kubernetes Blog](https://kubernetes.io/blog/)
* [EKS workshop](https://eksworkshop.com/010_introduction/basics/concepts_nodes/)
  * This explains about K8s
* [[토크ON세미나] 쿠버네티스 살펴보기 @ youtube](https://www.youtube.com/watch?v=xZ3tcFvbUGc&list=PLinIyjMcdO2SRxI4VmoU6gwUZr1XGMCyB&index=2)
  * 한글로 설명한 자세한 강의
  * [Workshop 개발 환경 셋팅하기 @ github](https://github.com/subicura/workshop-init)
  * [kubernetes 기본 가이드 @ github](https://github.com/subicura/workshop-k8s-basic)
* [AWS Kubernetes 서비스 자세히 살펴보기 - 정영준 솔루션즈 아키텍트(AWS), 이창수 솔루션즈 아키텍트(AWS)](https://www.youtube.com/watch?v=iAP_pTrm4Eo)
  * [slide](https://www.slideshare.net/awskorea/aws-kubernetes-aws-aws-devday2018)
* [Kubernetes Deconstructed: Understanding Kubernetes by Breaking It Down - Carson Anderson, DOMO](https://www.youtube.com/watch?v=90kZRyPcRZw&fbclid=IwAR0InYUnMQD-t-o8JhS5U5KRRaJvSQQc1fBDeBCb8cv6eRk62vsG2Si_Ijo)
  * [slide](http://kube-decon.carson-anderson.com/Layers/3-Network.sozi.html#frame4156)

# Architecture

## Overview

Kubernetes 가 사용하는 큰 개념은 객체 (Object) 와 그것을 관리하는 컨트롤러 (Controller) 두가지가 있다. Object 의 종류는 **Pod, Service, Volume, Namespace** 등이 있다. Controller 의 종류는 **ReplicaSet, Deployment, StatefulSet, DaemonSet, Job** 등이 있다. Kubernetes 는 yaml 파일을 사용하여 설정한다.

```yaml
apiVersion : v1
Kind : Pod
```

Kind 의 값에 따라 설정파일이 어떤 Object 혹은 controller 에 대한 작업인지 알 수 있다.

Kubernetes Cluster 는 Master 와 Node 두가지 종류가 있다. 

Master 는 **etcd, kube-apiserver, kube-scheduler, kube-controller-manager, kubelet, kube-proxy, docker** 등이 실행된다. Master 장비 1 대에 앞서 언급한 프로세스들 한 묶음을 같이 실행하는게 일반적인 구성이다. Master 는 일반적으로 High Availibility 를 위해 3 대 실행한다. 평소 1 대를 활성시키고 나머지 2 대는 대기시킨다.

Node 는 초기에 미니언(minion) 이라고 불렀다. Node 는 **kubelet, kube-proxy, docker** 등이 실행된다. 대부분의 컨테이너들은 Node 에서 실행된다.

![](https://upload.wikimedia.org/wikipedia/commons/b/be/Kubernetes.png)

위의 그림은 Kubernetes System Diagram 이다. Master 와 여러개의 Node 들로 구성된다. Operator 는 오로지 Master 의 API Server 와 통신한다. Node 들 역시 마찬가지이다.

## Kubernetes Components

### Master

* ETCD
  * key-value 저장소
* kube-apiserver
  * kubernetes cluster api 를 사용할 수 있게 해주는 gateway 이다. 들어오는 요청의 유효성을 검증하고 다른 곳으로 전달한다.
* kube-scheduler
  * 현재 cluster 안에서 자원할당이 가능한 Node 를 하나 선택하여 그곳에 pod 를 실행한다. Pod 가 하나 실행할 때 여러가지 조건이 지정되는데 kube-scheduler 가 그 조건에 맞는 Node 를 찾아준다. 예를 들어 필요한 하드웨어 요구사항, affinity, anti-affinity, 특정 데이터가 있는가 등이 해당된다.
* kube-controller-manager
  * kubernetes 는 controller 들이 Pod 들을 관리한다. kube-controller-manager 는 controller 들을 실행한다.
* cloud-controller-manager
  * 또 다른 cloud 와 연동할 때 사용한다. 
  * Node Controller, Route Controller, Service Controller, Volume Controler 등이 관련되어 있다.

### Node

* kubelet
  * 모든 Node 에서 실행되는 agent 이다. Pod 의 Container 가 실행되는 것을 관리한다. PodSpecs 라는 설정을 받아서 그 조건에 맞게 Container 를 실행하고 Container 가 정상적으로 실행되고 있는지 상태 체크를 한다.
* kube-proxy 
  * kubernetes 는 cluster 안의 virtual network 를 설정하고 관리한다. kube-proxy 는 virtual network 가 동작할 수 있도록하는 process 이다. host 의 network 규칙을 관리하거나 connection forwarding 을 한다.
* container runtime
  * container 를 실행한다. 가장 많이 알려진 container runtime 은 docker 이다. container 에 관한 표준을 제정하는 [OCI(Open Container Initiative)](https://www.opencontainers.org/) 의 runtime-spec 을 구현하는 container runtime 이라면 kubernetes 에서 사용할 수 있다.
* cAdvisor (container advisor)
  * 리소스 사용, 성능 통계를 제공


### Addons

cluster 안에서 필요한 기능들을 위해 실행되는 Pod 들이다. 주로 Deployment Controller, Replication Controller 에 의해 관리된다. Addon 이 사용하는 namespace 는 kub-system 이다.

* Networking Addon
* DNS Addon
* Dashboard Addon
* Container resource monitoring
* cluster logging

# Install

## Install on Win64

* Install docker, enable kubernetes. That's all.
* If you meet the issue like following, set env `%KUBECONFIG%` as `c:\Users\iamslash\.kube\config`

```bash
> kubectl config get-contexts
CURRENT   NAME      CLUSTER   AUTHINFO   NAMESPACE
> kubectl version
...
Unable to connect to the server: dial tcp [::1]:8080: connectex: No connection could be made because the target machine actively refused it.
```

## Install on macOS

* [도커(Docker), 쿠버네티스(Kubernetes) 통합 도커 데스크톱을 스테이블 채널에 릴리즈 @ 44bits](https://www.44bits.io/ko/post/news--release-docker-desktop-with-kubernetes-to-stable-channel)

* Install docker, enable kubernetes. That's all.

# Basic

## Useful Commands

* [workshop-k8s-basic/guide/guide-03/task-01.md](https://github.com/subicura/workshop-k8s-basic/blob/master/guide/guide-03/task-01.md)
  * [[토크ON세미나] 쿠버네티스 살펴보기 6강 - Kubernetes(쿠버네티스) 실습 1 | T아카데미](https://www.youtube.com/watch?v=G0-VoHbunks&list=PLinIyjMcdO2SRxI4VmoU6gwUZr1XGMCyB&index=6)
* [kubectl 치트 시트](https://kubernetes.io/ko/docs/reference/kubectl/cheatsheet/)

----

* api-resources

```bash
# show all objects including node, pod, replicaset, deployemnt,
# service, loadbalancer, ingress, volume, configmap, secret,
# namespace
$ kubectl api-resources
```

* get

```bash
# show recent pod, replicaset, deployment, service not all
$ kubectl get all

# show nodes
$ kubectl get no
$ kubectl get node
$ kubectl get nodes

# change result format
$ kubectl get nodes -o wide
$ kubectl get nodes -o yaml
$ kubectl get nodes -o json
$ kubectl get nodes -o json |
      jq ".items[].metadata.name"
$ kubectl get nodes -o json |
      jq ".items[] | {name:.metadata.name} + .status.capacity"

# show pods with the namespace
$ k get pods --all-namespace
$ k get pods --namespace kube-system
```

* describe

```bash
# kubectl describe type/name
# kubectl describe type name
kubectl describe node <node name>
kubectl describe node/<node name>
```

* etc

```bash
kubectl exec -it <POD_NAME>
kubectl logs -f <POD_NAME|TYPE/NAME>

kubectl apply -f <FILENAME>
kubectl delete -f <FILENAME>
```

## Launch Single Pod

```bash
# Create my-nginx-* pod and my-nginx deployment
> kubectl run my-nginx --image nginx --port=80
# Show running pods
> kubectl get pods
# Show deployments. Deployment is a specification for deploying pods.
> kubectl get dployments
# Scale out my-nginx deployment.
> kubectl scale deploy my-nginx --replicas=2
# Create a service to expose my-nginx pods. These are kinds of services. ClusterIP, NodePort, LoadBalancer, ExteralName
> kubectl expose deployment my-nginx --type=NodePort
# show services
> kubectl get services
# show details of my-nginx service
> kubectl describe service my-nginx
# Delete my-nginx deployment including pods.
> kubectl delete deployment my-nginx
# Delete my-nginx service
> kubectl delete service my-nginx
```

## Lunach Pods with livnessprobe, readynessprobe

* [workshop-k8s-basic/guide/guide-03/task-02.md](https://github.com/subicura/workshop-k8s-basic/blob/master/guide/guide-03/task-02.md)
  * [[토크ON세미나] 쿠버네티스 살펴보기 6강 - Kubernetes(쿠버네티스) 실습 1 | T아카데미](https://www.youtube.com/watch?v=G0-VoHbunks&list=PLinIyjMcdO2SRxI4VmoU6gwUZr1XGMCyB&index=6)

----

### key commands

```bash
$ kubectl run whoami --image subicura/whoami:1 
$ kubectl get po
$ kubectl get pod
$ kubectl get pods
$ kubectl get pods -o wide
$ kubectl get pods -o yaml
$ kubectl get pods -o json
$ kubectl logs whoami-<xxxx>
$ kubectl logs -f whoami-<xxxx>
$ kubectl exec -it whoami-<xxxx> sh
$ kubectl describe pods whoami-<xxxx>
$ kubectl delete pods whoami-<xxxx>
$ kubectl get pods
$ kubectl get all
$ kubectl delete deployment/whoami
```

### Launch Simple Pod

* whoami-pod.yml
   
```yml
apiVersion: v1
kind: Pod
metadata:
  name: whoami
  labels:
    type: app
spec:
  containers:
  - name: app
    image: subicura/whoami:1
```
* launch

```bash
$ kubectl apply -f whoami-pod.yml
```

### Launch Simple Pod with LivenessProbe

* whoami-pod-lp.yml

```yml
apiVersion: v1
kind: Pod
metadata:
  name: whoami-lp
  labels:
    type: app
spec:
  containers:
  - name: app
    image: subicura/whoami:1
    livenessProbe:
      httpGet:
        path: /not/exist
        port: 8080
      initialDelaySeconds: 5
      timeoutSeconds: 2 # Default 1
      periodSeconds: 5 # Defaults 10
      failureThreshold: 1 # Defaults 3
```

* launch

```bash
$ kubectl apply -f whoami-pod-lp.yml
```

### Launch Simple Pod with ReadinessProbe

* whoami-pod-rp.yml

```yml
apiVersion: v1
kind: Pod
metadata:
  name: whoami-rp
  labels:
    type: app
spec:
  containers:
  - name: app
    image: subicura/whoami:1
    readinessProbe:
      httpGet:
        path: /not/exist
        port: 8080
      initialDelaySeconds: 5
      timeoutSeconds: 2 # Default 1
      periodSeconds: 5 # Defaults 10
      failureThreshold: 1 # Defaults 3
```

* launch

```bash
$ kubectl apply -f whoami-pod-rp.yml
```

### Launch Simple Pod with HealthCheck

* whoami-pod-health.yml

```yml
apiVersion: v1
kind: Pod
metadata:
  name: whoami-health
  labels:
    type: app
spec:
  containers:
  - name: app
    image: subicura/whoami:1
    livenessProbe:
      httpGet:
        path: /
        port: 4567
    readinessProbe:
      httpGet:
        path: /
        port: 4567
```

* launch

```bash
$ kubectl apply -f whoami-pod-health.yml
```

### Launch Simple Pod with Multi Containers

* whoami-pod-redis.yml

```yml
apiVersion: v1
kind: Pod
metadata:
  name: whoami-redis
  labels:
    type: stack
spec:
  containers:
  - name: app
    image: subicura/whoami-redis:1
    env:
    - name: REDIS_HOST
      value: "localhost"
  - name: db
    image: redis
```

* launch

```bash
$ kubectl apply -f whoami-pod-redis.yml
$ kubectl get all
$ kubectl logs whoami-redis
$ kubectl logs whoami-redis app
$ kubectl logs whoami-redis db
$ kubectl exec -it whoami-redis
$ kubectl exec -it whoami-redis -c db sh
$ kubectl exec -it whoami-redis -c app sh
  apk add curl busybox-extras # install telnet
  curl localhost:4567
  telnet localhost 6379
    dbsize
    KEYS *
    GET count
    quit
$ kubectl get pod/whoami-redis
$ kubectl get pod/whoami-redis -o yaml
$ kubectl get pod/whoami-redis -o jsonpath="{.spec.containers[0].name}"
$ kubectl get pod/whoami-redis -o jsonpath="{.spec.containers[*].name}"
$ kubectl describe pod/whoami-redis
```

### Delete All resources

```bash
$ kubectl delete -f ./
```

## Launch Replicaset

* [workshop-k8s-basic/guide/guide-03/task-03.md](https://github.com/subicura/workshop-k8s-basic/blob/master/guide/guide-03/task-03.md)
  * [[토크ON세미나] 쿠버네티스 살펴보기 6강 - Kubernetes(쿠버네티스) 실습 1 | T아카데미](https://www.youtube.com/watch?v=G0-VoHbunks&list=PLinIyjMcdO2SRxI4VmoU6gwUZr1XGMCyB&index=6)

----

### Launch Simple Replicaset

We use Deployment more than Replicaset. ReplicaSet is used in Deployment.

* whoami-rs.yml
  * ReplicaSet is still beta.
  * If there is no pod such as selector, Launch pod with template. 

```yml
apiVersion: apps/v1beta2
kind: ReplicaSet
metadata:
  name: whoami-rs
spec:
  replicas: 1
  selector:
    matchLabels:
      type: app
      service: whoami
  template:
    metadata:
      labels:
        type: app
        service: whoami
    spec:
      containers:
      - name: whoami
        image: subicura/whoami:1
        livenessProbe:
          httpGet:
            path: /
            port: 4567
```

* launch

```bash
$ kubectl apply -f whoami-rs.yml
$ kubectl get pods --show-labels
# If remove service from label, ReplicatSet launch another pod
$ kubectl label pod/whoami-rs-<xxxx> service-
# set label
$ kubectl label pod/whoami-rs-<xxxx> service=whoami
# modify replicas as 3 and apply again
#   It is same with kubectl scale --replicas=3 -f whoami-rs.yml
$ kubectl apply -f whoami-rs.yml
```

### Launch ReplicaSet Scale out

* whoami-rs-scaled.yml

```yml
apiVersion: apps/v1beta2
kind: ReplicaSet
metadata:
  name: whoami-rs
spec:
  replicas: 4
  selector:
    matchLabels:
      type: app
      service: whoami
  template:
    metadata:
      labels:
        type: app
        service: whoami
    spec:
      containers:
      - name: whoami
        image: subicura/whoami:1
        livenessProbe:
          httpGet:
            path: /
            port: 4567
```

* launch

```bash
$ kubectl apply -f whoami-rs-scaled.yml
```

## Launch Deployment

* [workshop-k8s-basic/guide/guide-03/task-04.md](https://github.com/subicura/workshop-k8s-basic/blob/master/guide/guide-03/task-04.md)
  * [[토크ON세미나] 쿠버네티스 살펴보기 6강 - Kubernetes(쿠버네티스) 실습 1 | T아카데미](https://www.youtube.com/watch?v=G0-VoHbunks&list=PLinIyjMcdO2SRxI4VmoU6gwUZr1XGMCyB&index=6)

----

### Launch Simple Deployment

* whoami-deploy.yml
  * It is almost same with ReplicaSet.
  * Deployment manages versions of ReplicaSet.

```yml
apiVersion: apps/v1beta2
kind: Deployment
metadata:
  name: whoami-deploy
spec:
  replicas: 3
  selector:
    matchLabels:
      type: app
      service: whoami
  template:
    metadata:
      labels:
        type: app
        service: whoami
    spec:
      containers:
      - name: whoami
        image: subicura/whoami:1
        livenessProbe:
          httpGet:
            path: /
            port: 4567
```

* launch

```bash
$ kubectl apply -f whoami-deploy.yml
$ kubectl set image deploy/whoami-deploy whoami=subicura/whoami:2
$ kubectl apply -f whoami-deploy.yml
# watch continuously
$ kubectl get rs -w
$ kubectl describe deploy/whoami-deploy
# show history
$ kubectl rollout history -f whoami-deploy.yml
$ kubectl set image deploy/whoami-deploy whoami=subicura/whoami:1 --record=true
$ kubectl rollout history -f whoami-deploy.yml
$ kubectl rollout history -f whoami-deploy.yml --revision=2
$ kubectl rollout status deploy/whoami-deploy
$ kubectl rollout undo deploy/whoami-deploy
$ kubectl rollout undo deploy/whoami-deploy --to-revision=3
```

### Launch Deployment with RollingUpdate

* whoami-deploy-strategy.yml

```yml
apiVersion: apps/v1beta2
kind: Deployment
metadata:
  name: whoami-deploy
spec:
  replicas: 3
  selector:
    matchLabels:
      type: app
      service: whoami
  minReadySeconds: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
  template:
    metadata:
      labels:
        type: app
        service: whoami
    spec:
      containers:
      - name: whoami
        image: subicura/whoami:1
        livenessProbe:
          httpGet:
            path: /
            port: 4567
```

* launch

```bash
$ kubectl apply -f whoami-deploy-strategy
$ kubectl describe deploy/whoami-deploy
$ kubectl set image deploy/whoami-deploy whoami=subicura/whoami:2
$ kubectl get rs -w
```

## Launch Service

* [workshop-k8s-basic/guide/guide-03/task-05.md](https://github.com/subicura/workshop-k8s-basic/blob/master/guide/guide-05/task-05.md)
  * [[토크ON세미나] 쿠버네티스 살펴보기 7강 - Kubernetes(쿠버네티스) 실습 2 | T아카데미](https://www.youtube.com/watch?v=v6TUgqfV3Fo&list=PLinIyjMcdO2SRxI4VmoU6gwUZr1XGMCyB&index=7)
* [Kubernetes NodePort vs LoadBalancer vs Ingress? When should I use what?](https://medium.com/google-cloud/kubernetes-nodeport-vs-loadbalancer-vs-ingress-when-should-i-use-what-922f010849e0)

----

* ClusterIP is used for internal communication.
* NodePort is used for external communication???

### Launch Simple Service

* redis-app.yml

```yml
apiVersion: apps/v1beta2
kind: Deployment
metadata:
  name: redis
spec:
  selector:
    matchLabels:
      type: db
      service: redis
  template:
    metadata:
      labels:
        type: db
        service: redis
    spec:
      containers:
      - name: redis
        image: redis
        ports:
        - containerPort: 6379
          protocol: TCP
---
# This is for ClusterIP
# ClusterIP is used for internal communication
apiVersion: v1
kind: Service
metadata:
  name: redis
spec:
  ports:
  - port: 6379
    protocol: TCP
  selector:
    type: db
    service: redis
```

* whoami.yml

```yml
apiVersion: apps/v1beta2
kind: Deployment
metadata:
  name: whoami
spec:
  selector:
    matchLabels:
      type: app
      service: whoami
  template:
    metadata:
      labels:
        type: app
        service: whoami
    spec:
      containers:
      - name: whoami
        image: subicura/whoami-redis:1
        env:
        - name: REDIS_HOST
          value: "redis"
        - name: REDIS_PORT
          value: "6379"
```

* launch

```bash
$ kubectl apply -f redis-app.yml
$ kubectl apply -f whoami.yml
$ kubectl get ep
$ kubectl exec -it whoami-<xxxxx> sh
  apk add curl busybox-extras # install telnet
  curl localhost:4567
  curl localhost:4567
  telnet localhost 6379
  telnet redis 6379
    dbsize
    KEYS *
    GET count
    quit
```

### Launch Service with NodePort

* whoami-svc.yml

```yml
apiVersion: v1
kind: Service
metadata:
  name: whoami
spec:
  type: NodePort
  ports:
  - port: 4567
    protocol: TCP
  selector:
    type: app
    service: whoami
```

## Launch LoadBalancer

* [workshop-k8s-basic/guide/guide-03/task-06.md](https://github.com/subicura/workshop-k8s-basic/blob/master/guide/guide-05/task-06.md)
  * [[토크ON세미나] 쿠버네티스 살펴보기 7강 - Kubernetes(쿠버네티스) 실습 2 | T아카데미](https://www.youtube.com/watch?v=v6TUgqfV3Fo&list=PLinIyjMcdO2SRxI4VmoU6gwUZr1XGMCyB&index=7)

----

### Launch Simple LoadBalancer

* whoami-app.yml
  * If you launch this on AWS, ELB will attached to service.
  * NodePort is just a external port of Node But LoadBalancer is external reousrce to load balances.

```yml
apiVersion: apps/v1beta2
kind: Deployment
metadata:
  name: whoami
spec:
  selector:
    matchLabels:
      type: app
      service: whoami
  template:
    metadata:
      labels:
        type: app
        service: whoami
    spec:
      containers:
      - name: whoami
        image: subicura/whoami-redis:1
        env:
        - name: REDIS_HOST
          value: "redis"
        - name: REDIS_PORT
          value: "6379"
---

apiVersion: v1
kind: Service
metadata:
  name: whoami
spec:
  type: LoadBalancer
  ports:
  - port: 8000
    targetPort: 4567
    protocol: TCP
  selector:
    type: app
    service: whoami

---

apiVersion: apps/v1beta2
kind: Deployment
metadata:
  name: redis
spec:
  selector:
    matchLabels:
      type: db
      service: redis
  template:
    metadata:
      labels:
        type: db
        service: redis
    spec:
      containers:
      - name: redis
        image: redis
        ports:
        - containerPort: 6379
          protocol: TCP
---

apiVersion: v1
kind: Service
metadata:
  name: redis
spec:
  ports:
  - port: 6379
    protocol: TCP
  selector:
    type: db
    service: redis
```

* launch

```bash
```

### ????

* whoami-svc-v1-v2.yml

```yml
apiVersion: apps/v1beta2
kind: Deployment
metadata:
  name: whoami-v1
spec:
  replicas: 3
  selector:
    matchLabels:
      type: app
      service: whoami
      version: v1
  template:
    metadata:
      labels:
        type: app
        service: whoami
        version: v1
    spec:
      containers:
      - name: whoami
        image: subicura/whoami:1
        livenessProbe:
          httpGet:
            path: /
            port: 4567

---

apiVersion: apps/v1beta2
kind: Deployment
metadata:
  name: whoami-v2
spec:
  replicas: 3
  selector:
    matchLabels:
      type: app
      service: whoami
      version: v2
  template:
    metadata:
      labels:
        type: app
        service: whoami
        version: v2
    spec:
      containers:
      - name: whoami
        image: subicura/whoami:2
        livenessProbe:
          httpGet:
            path: /
            port: 4567
```

* launch

```bash
```

### ???

* whoami-svc-v1.yml

```yml
apiVersion: v1
kind: Service
metadata:
  name: whoami
spec:
  type: LoadBalancer
  ports:
  - port: 8000
    targetPort: 4567
    protocol: TCP
  selector:
    type: app
    service: whoami
    version: v1
```

* launch

```bash
```

### ???

* whoami-svc-all.yml

```yml
apiVersion: v1
kind: Service
metadata:
  name: whoami
spec:
  type: LoadBalancer
  ports:
  - port: 8000
    targetPort: 4567
    protocol: TCP
  selector:
    type: app
    service: whoami
```

* launch

```bash
```

## Launch Ingress

* [workshop-k8s-basic/guide/guide-03/bonus.md](https://github.com/subicura/workshop-k8s-basic/blob/master/guide/guide-05/bonus.md)
  * [[토크ON세미나] 쿠버네티스 살펴보기 7강 - Kubernetes(쿠버네티스) 실습 2 | T아카데미](https://www.youtube.com/watch?v=v6TUgqfV3Fo&list=PLinIyjMcdO2SRxI4VmoU6gwUZr1XGMCyB&index=7)

----

* Ingress is a mapping between DNS to internals.
* You can use IP as DNS
  * [sslip.io](https://sslip.io/)
  * [nip.io](https://nip.io/)

```bash
10.0.0.1.nip.io maps to 10.0.0.1
192-168-1-250.nip.io maps to 192.168.1.250
app.10.8.0.1.nip.io maps to 10.8.0.1
app-37-247-48-68.nip.io maps to 37.247.48.68
customer1.app.10.0.0.1.nip.io maps to 10.0.0.1
customer2-app-127-0-0-1.nip.io maps to 127.0.0.1
```

### Launch Simple Ingress

* whoami-v1.yml

```yml
apiVersion: extensions/v1beta1
kind: Ingress
metadata:
  name: whoami-v1
  annotations:
    ingress.kubernetes.io/rewrite-target: "/"
    ingress.kubernetes.io/ssl-redirect: "false"
spec:
  rules:
  - host: v1.whoami.13.125.41.102.sslip.io
    http:
      paths: 
      - path: /
        backend:
          serviceName: whoami-v1
          servicePort: 4567

---

apiVersion: apps/v1beta2
kind: Deployment
metadata:
  name: whoami-v1
spec:
  replicas: 3
  selector:
    matchLabels:
      type: app
      service: whoami
      version: v1
  template:
    metadata:
      labels:
        type: app
        service: whoami
        version: v1
    spec:
      containers:
      - name: whoami
        image: subicura/whoami:1
        livenessProbe:
          httpGet:
            path: /
            port: 4567

---

apiVersion: v1
kind: Service
metadata:
  name: whoami-v1
spec:
  ports:
  - port: 4567
    protocol: TCP
  selector:
    type: app
    service: whoami
    version: v1
```

* launch

```bash
$ kubectl get ingress
```

### ????

* whoami-v2.yml

```yml
apiVersion: extensions/v1beta1
kind: Ingress
metadata:
  name: whoami-v2
  annotations:
    ingress.kubernetes.io/rewrite-target: "/"
    ingress.kubernetes.io/ssl-redirect: "false"
spec:
  rules:
  - host: v2.whoami.13.125.41.102.sslip.io
    http:
      paths: 
      - path: /
        backend:
          serviceName: whoami-v2
          servicePort: 4567

---

apiVersion: apps/v1beta2
kind: Deployment
metadata:
  name: whoami-v2
spec:
  replicas: 3
  selector:
    matchLabels:
      type: app
      service: whoami
      version: v2
  template:
    metadata:
      labels:
        type: app
        service: whoami
        version: v2
    spec:
      containers:
      - name: whoami
        image: subicura/whoami:2
        livenessProbe:
          httpGet:
            path: /
            port: 4567

---

apiVersion: v1
kind: Service
metadata:
  name: whoami-v2
spec:
  ports:
  - port: 4567
    protocol: TCP
  selector:
    type: app
    service: whoami
    version: v2
```

## Launch Horizontal Pod Autoscaler

* [workshop-k8s-basic/guide/guide-03/task-06.md](https://github.com/subicura/workshop-k8s-basic/blob/master/guide/guide-05/task-06.md)
  * [[토크ON세미나] 쿠버네티스 살펴보기 7강 - Kubernetes(쿠버네티스) 실습 2 | T아카데미](https://www.youtube.com/watch?v=v6TUgqfV3Fo&list=PLinIyjMcdO2SRxI4VmoU6gwUZr1XGMCyB&index=7)

----


### Launch Simple Horizontal Pod Autoscaler

* hpa-example-deploy.yml.yml

```yml
apiVersion: apps/v1beta2
kind: Deployment
metadata:
  name: hpa-example-deploy
spec:
  selector:
    matchLabels:
      type: app
      service: hpa-example
  template:
    metadata:
      labels:
        type: app
        service: hpa-example
    spec:
      containers:
      - name: hpa-example
        image: k8s.gcr.io/hpa-example
        resources:
            limits:
              cpu: "0.5"
            requests:
              cpu: "0.25"
---

apiVersion: v1
kind: Service
metadata:
  name: hpa-example
spec:
  ports:
  - port: 80
    protocol: TCP
  selector:
    type: app
    service: hpa-example
```

* hpa.yml

```yml
apiVersion: autoscaling/v1
kind: HorizontalPodAutoscaler
metadata:
  name: hpa-example
spec:
  maxReplicas: 4
  minReplicas: 1
  scaleTargetRef:
    apiVersion: extensions/v1
    kind: Deployment
    name: hpa-example-deploy
  targetCPUUtilizationPercentage: 10
```

## Launch Kubernetes Dashboard

```bash
# show k8s client, server version
$ kubectl version --output yaml
# show contexts, default context is docker-for-desktop
$ kubectl config get-contexts
# show nodes
$ kubectl get nodes
# show pods
$ kubectl get pods --all-namespaces
# launch k8s dashbaord
$ kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v1.10.1/src/deploy/recommended/kubernetes-dashboard.yaml
# show services
$ kubectl get services --all-namespaces
# launch proxy server to connect k8s dashboard
$ kubectl proxy
# open http://localhost:8001/api/v1/namespaces/kube-system/services/https:kubernetes-dashboard:/proxy/
# create sample user and login. 
# - https://github.com/kubernetes/dashboard/blob/master/docs/user/access-control/creating-sample-user.md
```

* `https://raw.githubusercontent.com/kubernetes/dashboard/v1.10.1/src/deploy/recommended/kubernetes-dashboard.yaml`

```yaml
# Copyright 2017 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------------------- Dashboard Secret ------------------- #

apiVersion: v1
kind: Secret
metadata:
  labels:
    k8s-app: kubernetes-dashboard
  name: kubernetes-dashboard-certs
  namespace: kube-system
type: Opaque

---
# ------------------- Dashboard Service Account ------------------- #

apiVersion: v1
kind: ServiceAccount
metadata:
  labels:
    k8s-app: kubernetes-dashboard
  name: kubernetes-dashboard
  namespace: kube-system

---
# ------------------- Dashboard Role & Role Binding ------------------- #

kind: Role
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: kubernetes-dashboard-minimal
  namespace: kube-system
rules:
  # Allow Dashboard to create 'kubernetes-dashboard-key-holder' secret.
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["create"]
  # Allow Dashboard to create 'kubernetes-dashboard-settings' config map.
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["create"]
  # Allow Dashboard to get, update and delete Dashboard exclusive secrets.
- apiGroups: [""]
  resources: ["secrets"]
  resourceNames: ["kubernetes-dashboard-key-holder", "kubernetes-dashboard-certs"]
  verbs: ["get", "update", "delete"]
  # Allow Dashboard to get and update 'kubernetes-dashboard-settings' config map.
- apiGroups: [""]
  resources: ["configmaps"]
  resourceNames: ["kubernetes-dashboard-settings"]
  verbs: ["get", "update"]
  # Allow Dashboard to get metrics from heapster.
- apiGroups: [""]
  resources: ["services"]
  resourceNames: ["heapster"]
  verbs: ["proxy"]
- apiGroups: [""]
  resources: ["services/proxy"]
  resourceNames: ["heapster", "http:heapster:", "https:heapster:"]
  verbs: ["get"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: kubernetes-dashboard-minimal
  namespace: kube-system
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: kubernetes-dashboard-minimal
subjects:
- kind: ServiceAccount
  name: kubernetes-dashboard
  namespace: kube-system

---
# ------------------- Dashboard Deployment ------------------- #

kind: Deployment
apiVersion: apps/v1
metadata:
  labels:
    k8s-app: kubernetes-dashboard
  name: kubernetes-dashboard
  namespace: kube-system
spec:
  replicas: 1
  revisionHistoryLimit: 10
  selector:
    matchLabels:
      k8s-app: kubernetes-dashboard
  template:
    metadata:
      labels:
        k8s-app: kubernetes-dashboard
    spec:
      containers:
      - name: kubernetes-dashboard
        image: k8s.gcr.io/kubernetes-dashboard-amd64:v1.10.1
        ports:
        - containerPort: 8443
          protocol: TCP
        args:
          - --auto-generate-certificates
          # Uncomment the following line to manually specify Kubernetes API server Host
          # If not specified, Dashboard will attempt to auto discover the API server and connect
          # to it. Uncomment only if the default does not work.
          # - --apiserver-host=http://my-address:port
        volumeMounts:
        - name: kubernetes-dashboard-certs
          mountPath: /certs
          # Create on-disk volume to store exec logs
        - mountPath: /tmp
          name: tmp-volume
        livenessProbe:
          httpGet:
            scheme: HTTPS
            path: /
            port: 8443
          initialDelaySeconds: 30
          timeoutSeconds: 30
      volumes:
      - name: kubernetes-dashboard-certs
        secret:
          secretName: kubernetes-dashboard-certs
      - name: tmp-volume
        emptyDir: {}
      serviceAccountName: kubernetes-dashboard
      # Comment the following tolerations if Dashboard must not be deployed on master
      tolerations:
      - key: node-role.kubernetes.io/master
        effect: NoSchedule

---
# ------------------- Dashboard Service ------------------- #

kind: Service
apiVersion: v1
metadata:
  labels:
    k8s-app: kubernetes-dashboard
  name: kubernetes-dashboard
  namespace: kube-system
spec:
  ports:
    - port: 443
      targetPort: 8443
  selector:
    k8s-app: kubernetes-dashboard
```

# Authorization

* [쿠버네티스 권한관리(Authorization)](https://arisu1000.tistory.com/27848)

# AWS EKS

* [Kubernetes On AWS | AWS Kubernetes Tutorial | AWS EKS Tutorial | AWS Training | Edureka](https://www.youtube.com/watch?v=6H5sXQoJiso)
  * [Getting Started with the AWS Management Console](https://docs.aws.amazon.com/eks/latest/userguide/getting-started-console.html#w243aac11b9b7c11b7b1)
* [Amazon EKS 시작하기](https://aws.amazon.com/ko/eks/getting-started/)
* [Amazon EKS Starter: Docker on AWS EKS with Kubernetes @ udemy](https://www.udemy.com/course/amazon-eks-starter-kubernetes-on-aws/)
* [AWS 기반 Kubernetes 정복하기 - 정영준 솔루션즈 아키텍트(AWS)](https://www.youtube.com/watch?v=lGw2y-GLBbs)
* [Getting Started with Amazon EKS](https://docs.aws.amazon.com/en_en/eks/latest/userguide/getting-started.html)

----

![](img/eks_arch.png)

* create IAM role "`eks-role`" 
  * with policies "`AmazonEKSClusterPolicy, AmazonEKSServicePolicy`"
* create Network (VPC, subnets, security groups) "`eks-net`" with CloudFormation
  * with the template body `https://amazon-eks.s3-us-west-2.amazonaws.com/cloudformation/2019-11-15/amazon-eks-vpc-sample.yaml`
* create EKS cluster "`nginx-cluster`"
* install kubectl

  ```bash
  $ kubectl version --short --client
  ```

* install aws cli

  ```bash
  $ aws --version
  ```

* install aws-iam-authenticator

   ```bash
   $ brew install aws-iam-authenticator
   $ aws-iam-authenticator --help
   ```

* Create a kubeconfig File

  ```bash
  $ aws --region ap-northeast-2 eks update-kubeconfig --name nginx-cluster
  Added new context arn:aws:eks:ap-northeast-2:612149981322:cluster/nginx-cluster to /Users/davidsun/.kube/config
  ```

* create worker nodes "`nginx-cluster-worker-nodes`" with CloudFormation
  * with the template body `https://amazon-eks.s3-us-west-2.amazonaws.com/cloudformation/2019-11-15/amazon-eks-nodegroup-role.yaml`

* create k8s ConfigMap and connect `nginx-cluster-worker-nodes` to `nginx-cluster`
  
  * `aws-iam-authenticator.yaml`

    ```yaml
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: aws-auth
      namespace: kube-system
    data:
      mapRoles:
        - rolearn: <NodeInstanceRole of CloudFormation nginx-cluster-worker-nodes>
          username: system:node:{{EC2PrivateDNSName}}
          groups:
            - system:bootstrappers
            - system:nodes
    ```

  * apply

    ```bash
    $ kubectl apply -f aws-iam-authenticator.yaml
    $ kubectl get nodes
    ```

* create k8s Deployment, Service 

  * `nginx-deploy.yaml`

    ```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: nginx
    spec:
      selector:
        matchlabels:
          run: nginx
      replicas: 2
      template:
        metadata:
          labels:
            run: nginx
        spec:
          containers:
          - name: nginx
            image: nginx:1.7.9
            ports:
            - containerPort: 80 
    ```

  * `nginx-service.yaml`

    ```yaml
    apiVersion: v1
    kind: Service
    metadata:
      name: nginx
      labels:
        run: nginx
    spec:
      ports:
      - port: 80
        protocol: TCP
      selector:
        run: nginx
      type: LoadBalancer
    ```

  * apply

    ```bash
    $ kubectl create -f nginx-deploy.yaml
    $ kubectl create -f nginx-service.yaml
    $ kubectl get services -o wide
    # copy LoadBalancer Ingress
    $ kubectl describe svc nginx
    ```

* open browser copied url

# Dive Deep

## controller

* [A deep dive into Kubernetes controllers](https://engineering.bitnami.com/articles/a-deep-dive-into-kubernetes-controllers.html)
  * [kubewatch](https://github.com/bitnami-labs/kubewatch)
    * controller which sends slack messages
  * [sample-controller](https://github.com/kubernetes/sample-controller)

----

* kubernetes 의 controller 는 특정한 kubernetes resource 를 지켜보고 있다가 desired state 를 적용한다.
  ```go
  for {
    desired := getDesiredState()
    current := getCurrentState()
    makeChanges(desired, current)
  }
  ```
* controller 의 주요 컴포넌트로 Informer/SharedInformer 와 Workqueue 가 있다.
* Informer/SharedInformer 는 desired state 를 발견하면 Workqueue 에 아이템을 하나 삽입한다.
* kube-controller-manager 는 많은 수의 controller 들을 포함한다. 각 controller 는 자신들이 담당하는 특정 resource 에 대해서만 polling 하고 caching 한다. 이 cache 는 controller 들에 의해 공유된다. SharedInformer 는 이와 같이 공유된 cache 를 사용한다. 따라서 SharedInformer 를 Informer 보다 더 많이 사용한다. 
* Worker thread 는 Workqueue 에서 아이템을 하나 꺼내어 처리한다.


