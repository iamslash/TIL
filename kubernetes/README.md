# Abstract

Kubernetes 는 여러개의 Container 들을 협업시킬 수 있는 도구이다. 

# Materials

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

# Basic

## Overview

Kubernetes 가 사용하는 큰 개념은 객체 (Object) 와 그것을 관리하는 컨트롤러 (Controller) 두가지가 있다. Object 의 종류는 Pod, Service, Volume, Namespace 등이 있다. Controller 의 종류는 ReplicaSet, Deployment, StatefulSet, DaemonSet, Job 등이 있다. Kubernetes 는 yaml 파일을 사용하여 설정한다.

```yaml
apiVersion : v1
Kind : Pod
```

Kind 의 값에 따라 설정파일이 어떤 Object 혹은 controller 에 대한 작업인지 알 수 있다.

Kubernetes Cluster 는 Master 와 Node 두가지 종류가 있다. 

Master 는 etcd, kube-apiserver, kube-scheduler, kube-controller-manager, kubelet, kube-proxy, docker 등이 실행된다. Master 장비 1대에 앞서 언급한 프로세스들 한 묶음을 같이 실행하는게 일반적인 구성이다. Master 는 일반적으로 High Availibility 를 위해 3 대 실행한다. 평소 1 대를 활성시키고 나머지 2 대는 대기시킨다.

Node 는 초기에 미니언(minion) 이라고 불렀다. Node 는 kubelet, kube-proxy, docker 등이 실행된다. 대부분의 컨테이너들은 Node 에서 실행된다.

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

# Usages

```bash
# 하나의 pod 에 my-nginx container 를 실행하자
> kubectl run my-nginx --image nginx --port=80
# 제대로 실행되었는지 pod 들의 목록을 보자
> kubectl get pods
# 실행중인 pod 들의 목록을 보자
> kubectl get dployments
# my-nginx pod 의 개수를 늘려보자.
> kubectl scale deploy my-nginx --replicas=2
# 서비스를 외부에 노출하기 위해서는 service 를 실행해야 한다. 서비스타입의 종류는 다음과 같다. ClusterIP, NodePort, LoadBalancer, ExteralName
> kubectl expose deployment my-nginx --type=NodePort
# 서비스들의 목록을 얻어오자.
> kubectl get services
# my-nginx service 의 자세한 정보를 보자
> kubectl describe service my-nginx
# my-ngnix pod 를 삭제하자.
> kubectl delete deployment my-nginx
# my-nginx service 를 삭제하자.
> kubectl delete service my-nginx
```

