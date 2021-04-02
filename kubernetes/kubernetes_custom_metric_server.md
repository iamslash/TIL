- [Abstract](#abstract)
- [Materials](#materials)
- [Prerequisites](#prerequisites)
- [HPA with metrics-server](#hpa-with-metrics-server)
- [HPA with Custom Metric Server](#hpa-with-custom-metric-server)

-----

# Abstract

HorizontalPodAutoscaler 에서 Custom Metric Server 와 통신하여 Custom Metric 에 대한 Scaling 을 수행할 수 있다.

# Materials

> * [167. [Kubernetes] 쿠버네티스 모니터링 : Prometheus Adapter와 Opencensus를 이용한 Custom Metrics 수집 및 HPA 적용 @ naverblog](https://blog.naver.com/PostView.nhn?blogId=alice_k106&logNo=221521978267&categoryNo=20&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView)
> * [Custom Metrics Adapter Server Boilerplate @ github](https://github.com/kubernetes-sigs/custom-metrics-apiserver)
> * [Building Your Own Custom Metrics API for Kubernetes Horizontal Pod Autoscaler](https://medium.com/swlh/building-your-own-custom-metrics-api-for-kubernetes-horizontal-pod-autoscaler-277473dea2c1)

# Prerequisites

[Kubernetes Design Documents and Proposals @ github](https://github.com/kubernetes/community/tree/master/contributors/design-proposals) 를 살펴보면 Kubernetes 의 여러가지 Design Proposals 을 확인할 수 있다. 

특히 [Kubernetes monitoring architecture](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/instrumentation/monitoring_architecture.md#architecture) 를 읽어보면 Metric 을 **System Metric** 과 **Service Metric** 으로 분리하고 Metric Pipeline 을 **Core Metric Pipeline** 과 **Monitoring Piepeline** 으로 분리하는 것을 발견할 수 있다. [쿠버네티스 모니터링 아키텍처(kubernetes monitoring architecture) @ tistory](https://arisu1000.tistory.com/27855?category=787056) 는 번역글이다.

Kubernetes 의 Metric 은 **System Metric** 과 **Service Metric** 으로 나뉘어진다.

**System Metric** 은 usage of CPU and memory by container and node 이다. 다시 nore-metrics 와 non-core-metrics 로 나뉘어진다.

* core-metrics
  * metrics that Kubernetes understands and uses for operation of its internal components and core utilities
  * for example, metrics used for scheduling
  * for example, cpu cumulative usage, memory instantaneous usage, disk usage of pods, disk usage of containers. We can check those with `kubectl top`.

* non-core-metrics
  * which are not interpreted by Kubernetes???

**Service Metric** 은 application 을 monitoring 하는데 필요한 metrics 이다. 다시 those produced by Kubernetes infrastructure components, those produced by user applications 으로 나뉘어진다. 예를 들어 nginx application 의 response time 이나 nginx application 5xx count 등이 해당된다.

Kubernetes 의 Metric Pipeline 은 **Core Metric Pipeline** 과 **Monitoring Piepeline** 으로 나뉘어진다.

**Core Metric Piepeline** 은 core system metrics 를 모으는 것이다. cadvisor 는 kubelet 안에 포함되어 있다. 이것은 node/pod/container 의 metrics 을 수집한다. [metric-server @ github](https://github.com/kubernetes-sigs/metrics-server) 는 kubelet 으로 부터 metrics 을 수집한다. 그리고 kube-apiserver 를 통해 metric-server 를 노출한다. 다른 component 들은 maser metrics api (`/apis/metrics.k8s.io`) 를 통해 core system metrics 를 얻을 수 있다. 예를 들어 HorizontalPodAutoscaler 는 CPU, memory metric 을 이용하여 auto scaling 할 수 있다.

**Monitoring Piepeline** 은 Cluster user 들이 application 들을 monitoring 할 때 사용할 수 있는 pipeline 이다. Kubernetes 가 책임지지 않는다. **System Metric**, **Service Metric** 을 monitoring 할 수 있어야 한다. 여러가지 방법을 제안한다. advisor + Prometheus 를 추천한다. 

이런 방법은 어떨까? Kubernetes Cluster 에 Prometheus 가 실행된다. Prometheus 는 metric server 로 부터 metric 을 수집한다. 그리고 [m3](/m3/README.md) 에 metric 을 remote write 한다. Cluster user 들은 Graphana 로 monitoring 한다.

다음은 Kubernetes 가 제안한 Monitoring Architecture Diagram 이다. 파란색은 **Monitoring Piepeline** 을 의미한다.

![](https://raw.githubusercontent.com/kubernetes/community/master/contributors/design-proposals/instrumentation/monitoring_architecture.png)

# HPA with metrics-server

[metric-server @ github](https://github.com/kubernetes-sigs/metrics-server) 설치하고 HPA 와 연동해 보자.

먼저 [metric-server @ github](https://github.com/kubernetes-sigs/metrics-server) 가 설치되어있는지 API 를 통해서 확인해 보자.

```bash
$ kubectl get --raw /apis/metrics.k8s.io
Error from server (NotFound): the server could not find the requested resource
```

다음과 같이 설치한다. 클러스터 내에서 사용하는 인증서가 신뢰되지 않은 경우와 호스트 이름을 찾지 못하는 경우를 방지하기 위해 `metrics-server-deployment.yaml` 를 수정한다. [Metrics server issue with hostname resolution of kubelet and apiserver unable to communicate with metric-server clusterIP](https://github.com/kubernetes-sigs/metrics-server/issues/131)

```bash
$ git clone https://github.com/kubernetes-incubator/metrics-server.git

$ vim metrics-server/deploy/1.8+/metrics-server-deployment.yaml
...
        emptyDir: {}
      containers:
      - name: metrics-server
        args:
        - --kubelet-insecure-tls
        - --kubelet-preferred-address-types=InternalIP
        image: k8s.gcr.io/metrics-server-amd64:v0.3.1
        imagePullPolicy: Always
        volumeMounts:
        - name: tmp-dir
          mountPath: /tmp
...
```

WIP...

# HPA with Custom Metric Server

Custom Metric Server 를 제작하여 HPA 와 연동해 보자. 

먼저 Custom Metric Server 를 제작한다. 그리고 endpoint 를 kube-apiserver 에 등록한다. 이제 kube-apiserver 의 `/apis/custom.metrics.k8s.io` 를 이용하여 Custom Metric Server 에게 request 할 수 있다. HorizontalPodAutoscaler 는 언급한 Custom Metrics Api 를 이용하여 scaling 할 수 있다.

WIP...
