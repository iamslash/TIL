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

다음과 같이 metric-server 를 설치한다. 클러스터 내에서 사용하는 인증서가 신뢰되지 않은 경우와 호스트 이름을 찾지 못하는 경우를 방지하기 위해 `components.yaml` 에 `--kubelet-insecure-tls` 를 추가한다. [Metrics server issue with hostname resolution of kubelet and apiserver unable to communicate with metric-server clusterIP](https://github.com/kubernetes-sigs/metrics-server/issues/131)

```bash
$ minikube start

$ wget https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
$ vim components.yaml
...
    spec:
      containers:
      - args:
        - --cert-dir=/tmp
        - --secure-port=4443
        - --kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname
        - --kubelet-use-node-status-port
        - --kubelet-insecure-tls
...

$ kubectl apply -f components.yaml
serviceaccount/metrics-server created
clusterrole.rbac.authorization.k8s.io/system:aggregated-metrics-reader created
clusterrole.rbac.authorization.k8s.io/system:metrics-server created
rolebinding.rbac.authorization.k8s.io/metrics-server-auth-reader created
clusterrolebinding.rbac.authorization.k8s.io/metrics-server:system:auth-delegator created
clusterrolebinding.rbac.authorization.k8s.io/system:metrics-server created
service/metrics-server created
deployment.apps/metrics-server created
apiservice.apiregistration.k8s.io/v1beta1.metrics.k8s.io created

# Check metric api registered
$ kubectl get --raw /apis/metrics.k8s.io | jq .
{
  "kind": "APIGroup",
  "apiVersion": "v1",
  "name": "metrics.k8s.io",
  "versions": [
    {
      "groupVersion": "metrics.k8s.io/v1beta1",
      "version": "v1beta1"
    }
  ],
  "preferredVersion": {
    "groupVersion": "metrics.k8s.io/v1beta1",
    "version": "v1beta1"
  }
}

$ kubectl top node
NAME       CPU(cores)   CPU%   MEMORY(bytes)   MEMORY%
minikube   321m         4%     781Mi           39%

# The result is empty because there is nod pod yet.
$ kubectl top pod
```

이제 Deployment, Service, HPA 를 apply 해보자.

> `deploy-normal`

```yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iamslash-blog
spec:
  replicas: 1
  selector:
    matchLabels:
      app: blog-service-label
  template:
    metadata:
      labels:
        app: blog-service-label
    spec:
      containers:
      - name: flask-web-server
        image: alicek106/flask-opencensus-example
        ports:
        - name: web-service
          containerPort: 8080
        - name: healthz-checker
          containerPort: 8088
        resources:
          requests:
            memory: "16Mi"
            cpu: "64m"
          limits:
            memory: "256Mi"
            cpu: "100m"
---
kind: Service
apiVersion: v1
metadata:
  name: iamslash-blog
  labels:
    app: service-monitor-label
spec:
  selector:
    app: blog-service-label # same to deployment selector
  ports:
  - name: web-service
    port: 8080
  - name: healthz-checker
    port: 8088
```

> `hpa-normal.yaml`

```yml
apiVersion: autoscaling/v2beta1
kind: HorizontalPodAutoscaler
metadata:
  name: hpa-normal
spec:
  scaleTargetRef:
    apiVersion: extensions/v1beta1
    kind: Deployment
    name: iamslash-blog
  minReplicas: 1
  maxReplicas: 4
  metrics:
  - type: Resource
    resource:
      name: cpu
      targetAverageUtilization: 5
  - type: Resource
    resource:
      name: memory
      targetAverageValue: 200Mi
```

HPA 가 cpu 를 이용하여 어떻게 scaling 하는지 알아보자.

`reousrces.requests.cpu == 64m` 이고 `resource.targetAverageUtilization == 5` 이다.
`64m` 을 `100%` 라고 하면 `3.2` 은 `5%` 이다. HPA 는 주기적으로 메트릭값을 얻어온다. 그리고 
desired-replica 개수를 계산하고 pod 의 개수를 desired-replica 개수 만큼 유지한다.

desired-replica 개수는 다음과 같이 계산한다.

```
desiredReplicas = ceil[currentReplicas * ( currentMetricValue / desiredMetricValue )]
```

currentMetricValue 는 container 당 평균값이다. desiredMetricValue 는 HPA manifest 에
표기한 값이다. 즉, pod 에 포함된 container 들의 cpu 사용량의
평균이 `3.2` 를 넘어가면 scale-out 하고 `3.2` 보다 작으면 scale-in 하라는 의미이다.

```bash
$ kubectl apply -f deploy-normal.yaml,hpa-normal.yaml
$ kubectl get hpa
NAME         REFERENCE                  TARGETS                          MINPODS   MAXPODS   REPLICAS   AGE
hpa-normal   Deployment/iamslash-blog   <unknown>/200Mi, <unknown>/50%   1         4         0          5m8s
$ kubectl get pods
NAME                            READY   STATUS    RESTARTS   AGE
iamslash-blog-b499ddfc5-zb4hb   1/1     Running   0          2m26s
$ kubectl top pod
NAME                            CPU(cores)   MEMORY(bytes)
iamslash-blog-b499ddfc5-zb4hb   1m           21Mi
```

이제 stress 를 이용해서 CPU 사용량을 늘려보자. 실패함.

```bash
$ kubectl exec -it iamslash-blog-b499ddfc5-zb4hb -c flask-web-server -- bash
> apt-get install stress
> stress

# Check CPU usage in other session
$ watch -n 1 kubectl top pod
```

이제 삭제하자.

```bash
$ kubectl delete -f deploy-normal.yaml,hpa-normal.yaml
```

# HPA with Custom Metric Server

Custom Metric Server 를 제작하여 HPA 와 연동해 보자. 

먼저 Custom Metric Server 를 제작한다. 그리고 endpoint 를 kube-apiserver 에 등록한다. 이제 kube-apiserver 의 `/apis/custom.metrics.k8s.io` 를 이용하여 Custom Metric Server 에게 request 할 수 있다. HorizontalPodAutoscaler 는 언급한 Custom Metrics Api 를 이용하여 scaling 할 수 있다.

WIP...
