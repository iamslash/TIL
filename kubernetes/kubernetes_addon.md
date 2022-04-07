- [Abstract](#abstract)
- [kube-state-metric](#kube-state-metric)
- [metric-server](#metric-server)
- [kube-state-metrics vs metric-server](#kube-state-metrics-vs-metric-server)
- [core-dns](#core-dns)

----

# Abstract

This is for Kubernetes Addons including kube-state-metric, metric-server, external-dns, etc.

# kube-state-metric

* [kube-state-metrics에 대해서 @ medium](https://medium.com/finda-tech/kube-state-metrics%EC%97%90-%EB%8C%80%ED%95%B4%EC%84%9C-1303b10fb8f8)
* [kube-state-metrics @ github](https://github.com/kubernetes/kube-state-metrics)

----

**kube-state-metrics** is a simple service that listens to the Kubernetes API server and generates metrics about the state of the objects.

It is not focused on the health of the individual Kubernetes components, but rather on the health of the various objects inside, such as deployments, nodes and pods.

쿠버네티스 클러스터 워커노드로 사용중인 서버의 CPU, 메모리, 디스크 뿐만 아니라 쿠버네티스 클러스터 내부의 Pod가 사용중인 리소스 매트릭과 네트워크 I/O, Deployments 갯수, Pod 갯수 등의 다양한 정보를 수집

# metric-server

* [Kubernetes Metrics Server @ github](https://github.com/kubernetes-sigs/metrics-server)
* [쿠버네티스 모니터링 : metrics-server (kubernetes monitoring : metrics-server)](https://arisu1000.tistory.com/27856)

----

**Metrics Server** collects resource metrics from Kubelets and exposes them in Kubernetes apiserver through Metrics API for use by **Horizontal Pod Autoscaler** and **Vertical Pod Autoscaler**. Metrics API can also be accessed by kubectl top, making it easier to debug autoscaling pipelines.

# kube-state-metrics vs metric-server

* [kube-state-metrics vs. metrics-server](https://github.com/kubernetes/kube-state-metrics#kube-state-metrics-vs-metrics-server)

-----

# core-dns

> * [[Kubernetes/Networking] CoreDNS
[출처] [Kubernetes/Networking] CoreDNS|작성자 kangdorr](https://blog.naver.com/PostView.naver?blogId=kangdorr&logNo=222597241716&parentCategoryNo=&categoryNo=7&viewDate=&isShowPopularPosts=false&from=postList)
> * [서비스 디스커버리를 위해 CoreDNS 사용하기](https://kubernetes.io/ko/docs/tasks/administer-cluster/coredns/)
> * [[K8S] CoreDNS의 동작 흐름 @ tistory](https://ba-reum.tistory.com/39)

[coredns](https://coredns.io/) 는 go 로 만들어진 dns server 이다. kubernetes 에 사용된다. 

Kubernetes Service 를 생성하면
`<service-name>.<namespace-name>.svc.cluster.local` 이 CoreDNS 에 등록된다. 같은
cluster 인 경우 `<service-name>.<namespace-name>` 를 사용해도 된다. 같은
namespace 인 경우 `<service-name>` 을 사용해도 된다.

임의의 Pod 에서 DNS Query 를 수행하면 다음의 과정을 거쳐서 응답을 얻어온다.

```
Pod (etc/resolv) -> CoreDNS -> Node (etc/resolv) -> External DNS Server
```

좀더 Dive Deep 해보자.

다음과 같이 coredns pod 를 확인해 보자.

```bash
$ kubectl -n kube-system get pods
NAME                               READY   STATUS    RESTARTS   AGE
coredns-558bd4d5db-hgsw8           1/1     Running   0          24h
```

coredns Service 에게 DNS query 를 요청하여 IP 를 확인할 수 있다. 즉, Service
Discovery 가 가능하다. 다음과 같이 coredns Service 를 확인해 보자.

```bash
$ kubectl -n kube-system get svc kube-dns
NAME       TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)                  AGE
kube-dns   ClusterIP   10.96.0.10   <none>        53/UDP,53/TCP,9153/TCP   24h
```

`53/UDP,53/TCP` 로 DNS Query 를 보낼 수 있다. `9153/TCP` 는 metric port 이다. 

이번에는 coredns 의 설정을 살펴보자. 다음과 같이 Configmap 을 확인해 보자.

```bash
$ kubectl -n kube-system describe configmaps coredns
Name:         coredns
Namespace:    kube-system
Labels:       <none>
Annotations:  <none>

Data
====
Corefile:
----
.:53 {
    errors
    health {
       lameduck 5s
    }
    ready
    kubernetes cluster.local in-addr.arpa ip6.arpa {
       pods insecure
       fallthrough in-addr.arpa ip6.arpa
       ttl 30
    }
    prometheus :9153
    hosts {
       192.168.65.2 host.minikube.internal
       fallthrough
    }
    forward . /etc/resolv.conf {
       max_concurrent 1000
    }
    cache 30
    loop
    reload
    loadbalance
}

Events:  <none>
```

주요 항목은 다음과 같다.

* `error`: Error 를 stdout 으로 출력
* `health`: CoreDNS 의 liveness probe 를 확인한다. `http://<CoreDNS-Pod-IP>:8080/health` 로 확인
* `ready`: CoreDNS 의 readiness probe 를 확인한다. `http://<CoreDNS-Pod-IP>:8181/health` 로 확인
* `kubernetes`: 
  * `pods`: Pod 응답 설정 A Record 를 처리하기 위한 모드를 설정한다.
    * `disabled`: 기본 값. Pod 요청을 처리하지 않고 NXDOMAIN 반환
    * `insecure`: A Record 를 반환 (보안에 취약하지만 kube-dns 와 하위호환성 제공)
    * `verified`: 동일한 Namespace 의 Pod 만 A Record 를 반환
  * `fallthrough`: Discovery 가 실패한 경우 동작을 설정한다.
  * `ttl`: 요청한 Query 의 Caching 을 설정한다. 
* `prometheus` 모니터링이 활성화 된 경우 Prometheus plugin 을 통해 metric 을 전송한다.
* `forward`: upstream Nameserver 또는 stubDomain 을 지정한다.
* `cache`: CoreDNS 가 Query 에 대한 Cache 유지 시간
* `loop`: loop 가 발견되면 CoreDNS 를 중단
* `reload`: Corefile 이 변경되는 경우 자동으로 로드한다.
* `loadbalance`: Query 의 응답 값에 대해 A, AAAA, MX Record 의 순서로 무작위 선정한다.

이제 DNS Query 가 어떻게 CoreDNS 혹은 Upstream Nameserver 까지 전달되는지 살펴보자.

먼저 다음과 같이 임의의 Pod 의 `/etc/resolv.conf` 를 살펴보자.

```bash
$ kubectl run -it --rm netdebug --image=nicolaka/netshoot --restart=Never -- cat /etc/resolv.conf
nameserver 10.96.0.10
search default.svc.cluster.local svc.cluster.local cluster.local
options ndots:5

$ kubectl get svc,ep -A
NAMESPACE     NAME                 TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)                  AGE
default       service/kubernetes   ClusterIP   10.96.0.1    <none>        443/TCP                  24h
kube-system   service/kube-dns     ClusterIP   10.96.0.10   <none>        53/UDP,53/TCP,9153/TCP   24h

NAMESPACE     NAME                                 ENDPOINTS                                     AGE
default       endpoints/kubernetes                 192.168.49.2:8443                             24h
kube-system   endpoints/k8s.io-minikube-hostpath   <none>                                        24h
kube-system   endpoints/kube-dns                   172.17.0.2:53,172.17.0.2:53,172.17.0.2:9153   24h
```

다음은 `/etc/resolv.conf` 의 주요한 항목의 설명이다.

* `nameserver`: netdebug Pod 이 DNS Query 를 요청할 DNS Server 의 IP 이다. 즉,
  CoreDns Service 의 IP 이다.
* `search`: DNS Server 에 DNS Query 를 요청할 때 Domain Name 에 부착할 Suffix 이다.
* `ndots:5`: dot 개수가 5 이상이면 FQDN 처럼 Query 한다. dot 개수가 5 미만이면
  `search` 의 항목들을 접미사로 추가하여 DNS Query 를 요청한다. 

이제 netdebug Pod 에서 `google.com` 을 DNS Query 해보자. 

```bash
$ kubectl run -it --rm netdebug --image=nicolaka/netshoot --restart=Never -- nslookup -type=A google.com -debug | grep QUESTIONS -A1
    QUESTIONS:
	google.com.default.svc.cluster.local, type = A, class = IN
--
    QUESTIONS:
	google.com.svc.cluster.local, type = A, class = IN
--
    QUESTIONS:
	google.com.cluster.local, type = A, class = IN
--
    QUESTIONS:
	google.com, type = A, class = IN 
```

참고로 `google.com.` 으로 DNS Query 하면
CoreDNS 를 거치지 않고 Upstream DNS Server 에게 전달된다.

```bash
$ kubectl run -it --rm netdebug --image=nicolaka/netshoot --restart=Never -- nslookup -type=A google.com. -debug | grep QUESTIONS -A1
    QUESTIONS:
	google.com, type = A, class = IN
```

`google.com` 에 대해 `default.svc.cluster.local, svc.cluster.local,
cluster.local` 접미사가 부착되어 DNS Query 한 것을 확인할 수 있다. 모두 CoreDNS 에서
검색을 실패하였다. External DNS Server 로 DNS Query 하였다.

다음과 같이 Node 의 `/etc/resolv.conf` 및 packet dump 를 확인해 보자.

```bash
$ tcpdump -i $CoreDNSveth -nn udp port 53
```
