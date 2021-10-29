- [Memory Metrics](#memory-metrics)
  - [container_memory_rss](#container_memory_rss)
  - [container_memory_working_set_bytes](#container_memory_working_set_bytes)
- [Monitor Cluster Components](#monitor-cluster-components)
- [Managing Application Logs](#managing-application-logs)
- [Monitoring Pipeline](#monitoring-pipeline)
----

# Memory Metrics

> [What is the difference between “container_memory_working_set_bytes” and “container_memory_rss” metric on the container](https://stackoverflow.com/questions/65428558/what-is-the-difference-between-container-memory-working-set-bytes-and-contain)
> [Kubernetes Pod Memory Monitoring — RSS, Working Set](https://www.getoutsidedoor.com/2021/03/30/kubernetes-pod-memory-monitoring/)

Kubernetes Pod Container 의 memory metrics 를 이해하려면 다음과 같은 것들이 중요하다. container_memory_rss, container_memory_working_set_bytes 는 physical memory 를 의미한다. 

* **container_memory_rss** : The amount of anonymous and swap cache memory
* **container_memory_working_set_bytes** : The amount of working set memory, this includes recently accessed memory, dirty memory, and kernel memory
* **container_memory_usage_bytes** : this metric also includes cached (think filesystem cache) items that can be evicted under memory pressure.
* **kube_pod_container_resource_requests** : `kube_pod_container_resource_requests{job="kube-state-metrics",resource="memory"}` 는 pod container 의 request memory 를 말한다.

`container_memory_working_set_bytes{...} / kube_pod_container_resource_requests{job="kube-state-metrics",resource="memory",...}" >= 90%` 이면 alarming 하는 것이 좋다.

`container_memory_working_set_bytes` 가 pod container memory request 에 가까워지면 Linux 의 OOM Killer 가 해당 process 를 kill 할 것이다.

## container_memory_rss ##

total_rss from `/sys/fs/cgroups/memory/memory.status file` 의 `total_rss` 와 같다.

```go
// The amount of anonymous and swap cache memory (includes transparent
// hugepages).
// Units: Bytes.
RSS uint64 `json:"rss"`
```

swapped out 된 pages 는 포함하지 않는다. physical memory 에 거주하는 shared libraries 의 memory 는 포함한다. 또한 physical memeory 에 거주하는 stack, heap 도 포함한다.

## container_memory_working_set_bytes ##

the total usage 에서 inactive file 을 제거한 것이다. Memory Pressure 상황일 때도 eviction 되지 않을 것들을 의미한다.

```go
// The amount of working set memory, this includes recently accessed memory,
// dirty memory, and kernel memory. Working set is <= "usage".
// Units: Bytes.
WorkingSet uint64 `json:"working_set"`
```

Working Set is the current size, in bytes, of the Working Set of this process. The Working Set is the set of memory pages touched recently by the threads in the process.

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
