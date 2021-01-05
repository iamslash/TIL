- [Abstract](#abstract)
- [kube-state-metric](#kube-state-metric)
- [metric-server](#metric-server)
- [kube-state-metrics vs metric-server](#kube-state-metrics-vs-metric-server)
- [node-exporter](#node-exporter)
- [external-dns](#external-dns)

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

# node-exporter

# external-dns

