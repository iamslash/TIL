- [Memory Metrics](#memory-metrics)
  - [container_memory_usage_bytes](#container_memory_usage_bytes)
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

`container_memory_rss` 가 Pod Container Memory Request 에 가까워지면 Linux 의 OOM Killer 가 해당 process 를 kill 할 것이다.

`container_memory_rss{...} / kube_pod_container_resource_requests{job="kube-state-metrics",resource="memory",...}" >= 90%` 이면 alarming 하는 것이 좋다.

## container_memory_usage_bytes

Container 가 사용하고 있는 모든 메모리를 의미한다. Kernel 에 의해 reclaimed 될
수 있는 Cache 도 포함된다. 이 것이 Limit 에 도달한다고 OOM 이 발생하지는 않는다.

다음은 cadvisor 의 `Usage` 정의이다.

```go
// cadvisor/info/v1/container.go
type MemoryStats struct {
    // Current memory usage, this includes all memory regardless of when it was
    // accessed.
    // Units: Bytes.
    Usage uint64 `json:"usage"`
    ...
}
```

## container_memory_rss ##

Container 가 사용하는 메모리중 RAM 에 존재하는 것의 크기이다. 즉, Physical
Memory 를 얼만큼 사용하는지를 나타낸다. 당연히 Swapped Out 된 Pages 는 포함하지
않는다.

다음은 cadvisor 의 `RSS` 정의이다.

```go
// cadvisor/info/v1/container.go
type MemoryStats struct {
    // The amount of anonymous and swap cache memory (includes transparent
    // hugepages).
    // Units: Bytes.
    RSS uint64 `json:"rss"`
    ...
}
```

cgroup 이 `/sys/fs/cgroups/memory/memory.status` 에 `RSS` 를 기록한다. 그리고
cAdvisor 는 그것을 읽어서 export 한다.

`container_memory_rss` 는 OOM Killer 와 관련이 있다. OOM Killer 는
`oom_badness()` 가 return 하는 수치가 높으면 해당 프로세스를 kill 한다.

아래는 Linux Kernel `oom_badness` 의 구현이다.

```c
long oom_badness(struct task_struct *p, unsigned long totalpages)
{
	long points;
	long adj;

	if (oom_unkillable_task(p))
		return LONG_MIN;

	p = find_lock_task_mm(p);
	if (!p)
		return LONG_MIN;

	/*
	 * Do not even consider tasks which are explicitly marked oom
	 * unkillable or have been already oom reaped or the are in
	 * the middle of vfork
	 */
	adj = (long)p->signal->oom_score_adj;
	if (adj == OOM_SCORE_ADJ_MIN ||
			test_bit(MMF_OOM_SKIP, &p->mm->flags) ||
			in_vfork(p)) {
		task_unlock(p);
		return LONG_MIN;
	}

	/*
	 * The baseline for the badness score is the proportion of RAM that each
	 * task's rss, pagetable and swap space use.
	 */
	points = get_mm_rss(p->mm) + get_mm_counter(p->mm, MM_SWAPENTS) +
		mm_pgtables_bytes(p->mm) / PAGE_SIZE;
	task_unlock(p);

	/* Normalize to oom_score_adj units */
	adj *= totalpages / 1000;
	points += adj;

	return points;
}
```

Linux 는 다음과 같이 `RSS` 를 나누고 있다.

| Name | Description |
|--|--|
| `MM_FILEPAGES` | Disk 의 내용을 Page 단위로 RAM 에 Caching 한 것 |
| `MM_ANONPAGES` | 동적, 정적으로 RAM 에 할당된 것??? |
| `MM_SWAPENTS` | Swapped Out 된 것??? |
| `MM_SHMEMPAGES` | RAM 에서 모든 Process 들이 공유해서 사용하는 것. |

다음은 Linux Kernel 의 구현이다.

```c
// linux/mm/mm_types_task.h
/*
 * When updating this, please also update struct resident_page_types[] in
 * kernel/fork.c
 */
enum {
	MM_FILEPAGES,	/* Resident file mapping pages */
	MM_ANONPAGES,	/* Resident anonymous pages */
	MM_SWAPENTS,	/* Anonymous swap entries */
	MM_SHMEMPAGES,	/* Resident shared memory pages */
	NR_MM_COUNTERS
};
```

`get_mm_rss()` 는 `RSS` 를 return 한다. `get_mm_rss()` 를 살펴보자.
`MM_FILEPAGES` 와 `MM_ANONPAGES` 를 합한 값을 리턴한다.

```c
static inline unsigned long get_mm_rss(struct mm_struct *mm)
{
	return get_mm_counter(mm, MM_FILEPAGES) +
		get_mm_counter(mm, MM_ANONPAGES) +
		get_mm_counter(mm, MM_SHMEMPAGES);
}
```

`oom_badness()` 가 리턴하는 숫자는 `RSS` 즉, `container_memory_rss` 와
관계가 깊다는 것을 알 수 있다. 

## container_memory_working_set_bytes ##

Container 가 최근 접근한 메모리의 크기를 나타낸다.

다음은 [cAdvisor](/cadvisor/README.md) 의 `WorkingSet` 정의이다.

```go
// The amount of working set memory, this includes recently accessed memory,
// dirty memory, and kernel memory. Working set is <= "usage".
// Units: Bytes.
WorkingSet uint64 `json:"working_set"`
```

cAdvisor 는 `setMemoryStats()` 에서 workingSet 을 `Usage - Inactive File` 로
계산한다. `Inactive File` 은 File Pages 중 Reclaimed 될만한 것을 나타낸다. 즉,
Disk 를 Page 단위로 읽어서 RAM 에 Caching 해 놓은 것들중 Swapped Out 될만한
것들이다.

다음은 cAdvisor 의 `setMemoryStats()` 구현이다.

```c
// cadvisor/handler.go
func setMemoryStats(s *cgroups.Stats, ret *info.ContainerStats) {
    ...
    workingSet := ret.Memory.Usage
    if v, ok := s.MemoryStats.Stats[inactiveFileKeyName]; ok {
        if workingSet < v {
            workingSet = 0
        } else {
            workingSet -= v
        }
    }
}
```

`container_memory_working_set_bytes` 은 `container_memory_rss` 와 달리 Physical
Memory 만을 포함하지는 않는 것 같다. 따라서 OOM Killer 의 badness 에 영향이
있다고 생각하기 어렵다. 이 것 보다는 `container_memory_rss` 가 Container 의
Memory Alterting 에 사용되는 것이 맞다고 생각한다.

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
