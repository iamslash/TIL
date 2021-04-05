- [Monitor Cluster Components](#monitor-cluster-components)
- [Managing Application Logs](#managing-application-logs)
- [Monitoring Pipeline](#monitoring-pipeline)

----

# Monitor Cluster Components

Metrics Server

```bash
$ minikube addons enable metrics-server

$ git clone https://github.com/kubernetes-incubator/metrics-server.git
$ kubectl create -f deploy/1.8+/
$ kubectl top node
$ kubectl top pod
```

# Managing Application Logs

* `event-sumulator.yaml` Single Container

```yaml
apiVersion: v1
kind: Pod
metadata:
name: event-simulator-pod
spec:
containers:
- name: event-simulator
image: kodekloud/event-simulator
```

```bash
# logs of docker container
$ docker run -d kodekloud/event-simulator
$ docker logs -f ecf

# logs of kubernetes container
$ kubectl create -f event-simulator.yaml
$ kubectl logs -f event-simulator-pod
```

* `event-simulator.yaml` multiple containers

```yaml
apiVersion: v1
kind: Pod
metadata:
name: event-simulator-pod
spec:
containers:
- name: event-simulator
image: kodekloud/event-simulator
event-simulator.yaml
- name: image-processor
image: some-image-processor
```

```bash
$ kubectl logs -f event-simulator-pod event-simulator
```

# Monitoring Pipeline

We can collect metrics using [prometheus](/prometheus/README.md) from `kube-apiservers`, `kubelet`, `kubelet/cadvisor`, [node-exporter](/node_exporter/README.md). kubelet includes cadvisor.

[PromQL for overall metrics](/prometheus/README.md#promql-for-overall-metrics)
