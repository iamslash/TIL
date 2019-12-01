- [Abstract](#abstract)
- [Materials](#materials)
- [Architecture](#architecture)
  - [Overview](#overview)
  - [Kubernetes Components](#kubernetes-components)
    - [Master](#master)
    - [Node](#node)
    - [Addons](#addons)
- [Install](#install)
  - [Install on macOS](#install-on-macos)
- [Basic](#basic)
  - [Launch Kubernetes Dashboard](#launch-kubernetes-dashboard)
  - [Launch Single Pod](#launch-single-pod)
  - [](#)

----

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

# Architecture

## Overview

Kubernetes 가 사용하는 큰 개념은 객체 (Object) 와 그것을 관리하는 컨트롤러 (Controller) 두가지가 있다. Object 의 종류는 **Pod, Service, Volume, Namespace** 등이 있다. Controller 의 종류는 **ReplicaSet, Deployment, StatefulSet, DaemonSet, Job** 등이 있다. Kubernetes 는 yaml 파일을 사용하여 설정한다.

```yaml
apiVersion : v1
Kind : Pod
```

Kind 의 값에 따라 설정파일이 어떤 Object 혹은 controller 에 대한 작업인지 알 수 있다.

Kubernetes Cluster 는 Master 와 Node 두가지 종류가 있다. 

Master 는 **etcd, kube-apiserver, kube-scheduler, kube-controller-manager, kubelet, kube-proxy, docker** 등이 실행된다. Master 장비 1대에 앞서 언급한 프로세스들 한 묶음을 같이 실행하는게 일반적인 구성이다. Master 는 일반적으로 High Availibility 를 위해 3 대 실행한다. 평소 1 대를 활성시키고 나머지 2 대는 대기시킨다.

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

## Install on macOS

* [도커(Docker), 쿠버네티스(Kubernetes) 통합 도커 데스크톱을 스테이블 채널에 릴리즈 @ 44bits](https://www.44bits.io/ko/post/news--release-docker-desktop-with-kubernetes-to-stable-channel)

* Install docker, enable kubernetes. That's all.

# Basic

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

## 