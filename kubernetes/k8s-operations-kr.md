# Kubernetes 클러스터 운영

- [노드 관리](#노드-관리)
  - [kubectl cordon / uncordon](#kubectl-cordon--uncordon)
  - [kubectl drain](#kubectl-drain)
  - [PodDisruptionBudget 으로 안전한 drain](#poddisruptionbudget-으로-안전한-drain)
  - [노드 유지보수 시나리오](#노드-유지보수-시나리오)
- [클러스터 업그레이드 전략](#클러스터-업그레이드-전략)
  - [업그레이드 원칙](#업그레이드-원칙)
  - [Control Plane 업그레이드](#control-plane-업그레이드)
  - [Worker Node 업그레이드](#worker-node-업그레이드)
- [etcd 백업과 복원](#etcd-백업과-복원)
  - [etcd 란](#etcd-란)
  - [스냅샷 백업](#스냅샷-백업)
  - [스냅샷 복원](#스냅샷-복원)
  - [자동 백업 CronJob](#자동-백업-cronjob)
- [모니터링](#모니터링)
  - [Kubernetes 모니터링 아키텍처](#kubernetes-모니터링-아키텍처)
  - [metrics-server](#metrics-server)
  - [kube-state-metrics](#kube-state-metrics)
  - [Prometheus + Grafana 구성](#prometheus--grafana-구성)
  - [컨테이너 메모리 메트릭 이해](#컨테이너-메모리-메트릭-이해)
  - [kubectl top 명령어](#kubectl-top-명령어)
  - [주요 모니터링 대상 정리](#주요-모니터링-대상-정리)
- [로깅](#로깅)
  - [kubectl logs 기본 사용법](#kubectl-logs-기본-사용법)
  - [클러스터 수준 로깅 아키텍처](#클러스터-수준-로깅-아키텍처)
  - [DaemonSet 으로 로그 수집](#daemonset-으로-로그-수집)
  - [사이드카 패턴 로깅](#사이드카-패턴-로깅)
- [실무 도구](#실무-도구)
  - [k9s](#k9s)
  - [kubectx / kubens](#kubectx--kubens)
  - [stern](#stern)
  - [Telepresence](#telepresence)
  - [Lens](#lens)

----

# 노드 관리

## kubectl cordon / uncordon

`cordon` 은 특정 노드에 새로운 Pod 가 스케줄링되지 않도록 막는다. 이미 실행 중인 Pod 는 영향받지 않는다.

```bash
# 노드를 스케줄링 불가 상태로 변경 (새 Pod 배치 거부)
kubectl cordon worker-node-1

# 노드 상태 확인 (SchedulingDisabled 가 표시됨)
kubectl get nodes
# NAME            STATUS                     ROLES    AGE
# worker-node-1   Ready,SchedulingDisabled   <none>   10d

# 노드를 다시 스케줄링 가능 상태로 변경
kubectl uncordon worker-node-1
```

## kubectl drain

`drain` 은 노드에서 실행 중인 Pod 를 다른 노드로 옮기고, 해당 노드를 스케줄링 불가 상태로 만든다. `cordon` + Pod 퇴거(eviction) 를 동시에 수행한다.

```bash
# 노드에서 Pod 를 안전하게 퇴거시킨다
# --ignore-daemonsets: DaemonSet 으로 생성된 Pod 는 무시 (DaemonSet Pod 는 퇴거 불가)
# --delete-emptydir-data: emptyDir 볼륨을 사용하는 Pod 도 퇴거 허용
kubectl drain worker-node-1 --ignore-daemonsets --delete-emptydir-data

# 강제로 퇴거 (ReplicaSet 등으로 관리되지 않는 단독 Pod 도 삭제)
kubectl drain worker-node-1 --ignore-daemonsets --delete-emptydir-data --force

# 유지보수가 끝난 뒤 노드를 다시 활성화
kubectl uncordon worker-node-1
```

drain 실행 시 Kubernetes 는 PodDisruptionBudget 을 확인한다. 최소 가용 Pod 수를 위반하면 drain 이 차단된다.

## PodDisruptionBudget 으로 안전한 drain

PodDisruptionBudget (PDB) 은 자발적 중단(voluntary disruption) 시 동시에 내려갈 수 있는 Pod 수를 제한한다.

```yaml
# pdb-web.yaml
# web-deployment 의 Pod 가 drain 중에도 최소 2개는 항상 실행되도록 보장한다
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: web-pdb
  namespace: production
spec:
  minAvailable: 2          # 항상 최소 2개의 Pod 가 실행 중이어야 한다
  selector:
    matchLabels:
      app: web-server
```

```yaml
# pdb-api.yaml
# 전체 Pod 중 최대 20% 만 동시에 중단 허용
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: api-pdb
  namespace: production
spec:
  maxUnavailable: "20%"    # 전체 Pod 의 20% 이상은 동시에 중단 불가
  selector:
    matchLabels:
      app: api-server
```

```bash
# PDB 적용
kubectl apply -f pdb-web.yaml

# PDB 상태 확인
kubectl get pdb -n production
# NAME      MIN AVAILABLE   MAX UNAVAILABLE   ALLOWED DISRUPTIONS   AGE
# web-pdb   2               N/A               1                     5m
```

## 노드 유지보수 시나리오

OS 패치나 커널 업그레이드 등 노드 재부팅이 필요한 경우의 전체 흐름이다.

```bash
# 1단계: 노드를 스케줄링 불가 + Pod 퇴거
kubectl drain worker-node-1 --ignore-daemonsets --delete-emptydir-data

# 2단계: 노드에 SSH 접속하여 유지보수 작업 수행
ssh worker-node-1
sudo apt-get update && sudo apt-get upgrade -y
sudo reboot

# 3단계: 노드 재부팅 완료 후 상태 확인
kubectl get node worker-node-1
# STATUS 가 Ready 가 될 때까지 대기

# 4단계: 노드를 다시 스케줄링 가능 상태로 변경
kubectl uncordon worker-node-1

# 5단계: Pod 가 정상적으로 재배치되었는지 확인
kubectl get pods -o wide | grep worker-node-1
```

----

# 클러스터 업그레이드 전략

## 업그레이드 원칙

- Control Plane (API Server, etcd, Controller Manager, Scheduler) 을 먼저 업그레이드한 뒤 Worker Node 를 업그레이드한다.
- 한 번에 한 마이너 버전씩 올린다. (예: 1.27 → 1.28 → 1.29)
- 마이너 버전을 두 단계 이상 건너뛰는 업그레이드는 지원하지 않는다.
- 업그레이드 전에 반드시 etcd 백업을 수행한다.

## Control Plane 업그레이드

`kubeadm` 을 사용하는 클러스터 기준이다.

```bash
# Control Plane 노드에서 실행

# 1단계: kubeadm 업그레이드 (예: 1.28.x -> 1.29.x)
sudo apt-mark unhold kubeadm
sudo apt-get update
sudo apt-get install -y kubeadm=1.29.0-00
sudo apt-mark hold kubeadm

# 2단계: 업그레이드 계획 확인 (실제 변경 없이 시뮬레이션)
sudo kubeadm upgrade plan

# 3단계: 실제 업그레이드 적용
sudo kubeadm upgrade apply v1.29.0

# 4단계: Control Plane 노드 drain
kubectl drain control-plane-1 --ignore-daemonsets --delete-emptydir-data

# 5단계: kubelet, kubectl 업그레이드
sudo apt-mark unhold kubelet kubectl
sudo apt-get install -y kubelet=1.29.0-00 kubectl=1.29.0-00
sudo apt-mark hold kubelet kubectl

# 6단계: kubelet 재시작
sudo systemctl daemon-reload
sudo systemctl restart kubelet

# 7단계: Control Plane 노드 uncordon
kubectl uncordon control-plane-1

# 8단계: 버전 확인
kubectl get nodes
kubectl version
```

## Worker Node 업그레이드

Worker Node 는 Control Plane 업그레이드 완료 후 하나씩 진행한다.

```bash
# Control Plane 노드에서 Worker Node drain
kubectl drain worker-node-1 --ignore-daemonsets --delete-emptydir-data

# Worker Node 에 SSH 접속하여 실행
ssh worker-node-1

# kubeadm 업그레이드
sudo apt-mark unhold kubeadm
sudo apt-get update
sudo apt-get install -y kubeadm=1.29.0-00
sudo apt-mark hold kubeadm

# Worker Node 설정 업그레이드
sudo kubeadm upgrade node

# kubelet, kubectl 업그레이드
sudo apt-mark unhold kubelet kubectl
sudo apt-get install -y kubelet=1.29.0-00 kubectl=1.29.0-00
sudo apt-mark hold kubelet kubectl

# kubelet 재시작
sudo systemctl daemon-reload
sudo systemctl restart kubelet

# Control Plane 노드로 돌아와서 uncordon
kubectl uncordon worker-node-1

# 노드 상태 확인
kubectl get nodes
# NAME               STATUS   ROLES           VERSION
# control-plane-1    Ready    control-plane   v1.29.0
# worker-node-1      Ready    <none>          v1.29.0
```

----

# etcd 백업과 복원

## etcd 란

etcd 는 Kubernetes 클러스터의 모든 상태를 저장하는 분산 키-값 저장소다. Pod, Service, ConfigMap, Secret, Deployment 등 클러스터의 모든 오브젝트 정보가 etcd 에 저장된다. etcd 가 손상되면 클러스터 전체가 동작 불능 상태가 된다.

```
클러스터 상태 저장 구조:
  kubectl apply → API Server → etcd (영구 저장)
                            ← etcd (조회)
```

## 스냅샷 백업

`etcdctl` 은 etcd 를 관리하는 CLI 도구다.

```bash
# etcd Pod 에서 인증서 경로 확인 (kubeadm 기준)
kubectl -n kube-system describe pod etcd-control-plane-1

# 스냅샷 백업 (etcd 가 실행 중인 Control Plane 노드에서 수행)
ETCDCTL_API=3 etcdctl snapshot save /backup/etcd-snapshot-$(date +%Y%m%d%H%M%S).db \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key

# 스냅샷 상태 확인
ETCDCTL_API=3 etcdctl snapshot status /backup/etcd-snapshot-20240101120000.db \
  --write-out=table
# +----------+----------+------------+------------+
# |   HASH   | REVISION | TOTAL KEYS | TOTAL SIZE |
# +----------+----------+------------+------------+
# | 59b6e992 |   123456 |       1200 |     4.2 MB |
# +----------+----------+------------+------------+
```

## 스냅샷 복원

```bash
# 1단계: 스냅샷으로 데이터 복원 (새 데이터 디렉토리에 복원)
ETCDCTL_API=3 etcdctl snapshot restore /backup/etcd-snapshot-20240101120000.db \
  --data-dir=/var/lib/etcd-restored \
  --initial-cluster=control-plane-1=https://127.0.0.1:2380 \
  --initial-advertise-peer-urls=https://127.0.0.1:2380 \
  --name=control-plane-1

# 2단계: etcd static pod manifest 수정 (복원된 데이터 디렉토리를 가리키도록)
# /etc/kubernetes/manifests/etcd.yaml 의 --data-dir 과 hostPath 를 수정
sudo vi /etc/kubernetes/manifests/etcd.yaml
# volumes.hostPath.path 를 /var/lib/etcd-restored 로 변경
# containers.volumeMounts 의 mountPath 도 일치시킨다

# 3단계: kubelet 이 etcd Pod 를 재시작할 때까지 대기
kubectl -n kube-system get pod etcd-control-plane-1 -w

# 4단계: 클러스터 상태 확인
kubectl get nodes
kubectl get pods -A
```

## 자동 백업 CronJob

```yaml
# etcd-backup-cronjob.yaml
# 매일 오전 2시에 etcd 스냅샷을 자동으로 생성한다
apiVersion: batch/v1
kind: CronJob
metadata:
  name: etcd-backup
  namespace: kube-system
spec:
  schedule: "0 2 * * *"          # 매일 02:00 에 실행
  successfulJobsHistoryLimit: 7  # 최근 7개의 성공 이력 보관
  failedJobsHistoryLimit: 3      # 최근 3개의 실패 이력 보관
  jobTemplate:
    spec:
      template:
        spec:
          hostNetwork: true       # etcd 에 직접 접근하기 위해 호스트 네트워크 사용
          nodeSelector:
            node-role.kubernetes.io/control-plane: ""  # Control Plane 노드에서만 실행
          tolerations:
            - key: node-role.kubernetes.io/control-plane
              effect: NoSchedule
          containers:
            - name: etcd-backup
              image: registry.k8s.io/etcd:3.5.9-0
              command:
                - /bin/sh
                - -c
                - |
                  # 오늘 날짜로 스냅샷 파일명 생성
                  BACKUP_FILE="/backup/etcd-$(date +%Y%m%d%H%M%S).db"
                  etcdctl snapshot save "$BACKUP_FILE" \
                    --endpoints=https://127.0.0.1:2379 \
                    --cacert=/etc/kubernetes/pki/etcd/ca.crt \
                    --cert=/etc/kubernetes/pki/etcd/server.crt \
                    --key=/etc/kubernetes/pki/etcd/server.key
                  echo "Backup saved: $BACKUP_FILE"
                  # 7일 이상 된 백업 파일 삭제
                  find /backup -name "etcd-*.db" -mtime +7 -delete
              env:
                - name: ETCDCTL_API
                  value: "3"
              volumeMounts:
                - name: etcd-certs
                  mountPath: /etc/kubernetes/pki/etcd
                  readOnly: true
                - name: backup-storage
                  mountPath: /backup
          volumes:
            - name: etcd-certs
              hostPath:
                path: /etc/kubernetes/pki/etcd
            - name: backup-storage
              hostPath:
                path: /var/etcd-backup   # 호스트의 백업 디렉토리
          restartPolicy: OnFailure
```

```bash
# CronJob 적용
kubectl apply -f etcd-backup-cronjob.yaml

# 수동으로 즉시 백업 실행 (테스트 목적)
kubectl create job --from=cronjob/etcd-backup etcd-backup-manual -n kube-system

# 백업 Job 상태 확인
kubectl get jobs -n kube-system
kubectl logs job/etcd-backup-manual -n kube-system
```

----

# 모니터링

## Kubernetes 모니터링 아키텍처

```
노드 레벨 메트릭 수집 흐름:
  컨테이너 런타임 (containerd)
    └─ cAdvisor (kubelet 내장)
         └─ Kubelet Metrics API (/metrics/cadvisor)
              ├─ metrics-server   → kubectl top / HPA
              └─ Prometheus       → Grafana 대시보드

클러스터 오브젝트 상태 수집 흐름:
  Kubernetes API Server
    └─ kube-state-metrics
         └─ Prometheus → Grafana 대시보드
```

## metrics-server

metrics-server 는 각 노드의 kubelet 에서 CPU/메모리 사용량을 수집하여 Kubernetes Metrics API 로 노출한다. HPA (Horizontal Pod Autoscaler) 와 VPA (Vertical Pod Autoscaler) 가 이 API 를 사용한다.

```bash
# metrics-server 설치 (kubectl apply)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# minikube 에서 활성화
minikube addons enable metrics-server

# 설치 확인
kubectl -n kube-system get deployment metrics-server
kubectl -n kube-system get pods -l k8s-app=metrics-server

# metrics-server 가 준비되면 kubectl top 을 사용할 수 있다
kubectl top nodes
kubectl top pods -A
```

metrics-server 는 짧은 주기(15초)로 메트릭을 수집하며, 과거 데이터를 저장하지 않는다. 장기 이력을 위해서는 Prometheus 가 필요하다.

## kube-state-metrics

kube-state-metrics 는 Kubernetes API Server 를 감시하여 클러스터 내 오브젝트(Deployment, Pod, Node 등)의 상태를 Prometheus 형식의 메트릭으로 노출한다.

metrics-server 와의 차이점:

| 항목 | metrics-server | kube-state-metrics |
|------|----------------|--------------------|
| 수집 대상 | CPU, 메모리 사용량 | 오브젝트 상태, 개수, 설정값 |
| 데이터 소스 | kubelet (cAdvisor) | Kubernetes API Server |
| 주요 용도 | kubectl top, HPA | Prometheus 모니터링, 알람 |
| 과거 데이터 | 저장 안 함 | Prometheus 가 저장 |

```bash
# kube-state-metrics 설치
kubectl apply -f https://github.com/kubernetes/kube-state-metrics/releases/latest/download/kube-state-metrics.yaml

# 설치 확인
kubectl -n kube-system get pods -l app.kubernetes.io/name=kube-state-metrics

# 주요 메트릭 예시 (Prometheus 쿼리)
# Deployment 중 사용 불가 레플리카 수
# kube_deployment_status_replicas_unavailable{namespace="production"}

# Pod 가 재시작된 횟수
# kube_pod_container_status_restarts_total{namespace="production"}

# Node 의 할당 가능한 메모리
# kube_node_status_allocatable{resource="memory"}
```

## Prometheus + Grafana 구성

kube-prometheus-stack 헬름 차트를 사용하면 Prometheus, Grafana, kube-state-metrics, node-exporter 를 한 번에 설치할 수 있다.

```bash
# Helm repo 추가
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# kube-prometheus-stack 설치 (monitoring 네임스페이스)
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.adminPassword=your-secure-password

# 설치된 컴포넌트 확인
kubectl -n monitoring get pods
# NAME                                              READY   STATUS
# kube-prometheus-stack-grafana-xxx                 3/3     Running
# kube-prometheus-stack-kube-state-metrics-xxx      1/1     Running
# kube-prometheus-stack-operator-xxx                1/1     Running
# kube-prometheus-stack-prometheus-node-exporter-xxx 1/1   Running
# prometheus-kube-prometheus-stack-prometheus-0     2/2     Running

# Grafana 포트 포워딩 (로컬에서 접근)
kubectl -n monitoring port-forward svc/kube-prometheus-stack-grafana 3000:80
# 브라우저에서 http://localhost:3000 접속 (admin / your-secure-password)
```

```yaml
# prometheus-additional-scrape.yaml
# 커스텀 애플리케이션 메트릭을 스크랩하는 ServiceMonitor 예시
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: my-app-monitor
  namespace: monitoring
  labels:
    release: kube-prometheus-stack  # Prometheus 가 이 레이블로 ServiceMonitor 를 선택
spec:
  selector:
    matchLabels:
      app: my-app                   # 모니터링할 Service 의 레이블
  namespaceSelector:
    matchNames:
      - production
  endpoints:
    - port: metrics                 # Service 에서 메트릭을 노출하는 포트 이름
      interval: 30s                 # 30초마다 스크랩
      path: /metrics
```

## 컨테이너 메모리 메트릭 이해

Kubernetes 에서 컨테이너 메모리를 모니터링할 때 세 가지 메트릭이 자주 사용된다. 각각의 의미를 정확히 이해해야 올바른 알람을 설정할 수 있다.

| 메트릭 | 설명 | OOM 관련 |
|--------|------|----------|
| `container_memory_usage_bytes` | 캐시(파일시스템 캐시)를 포함한 전체 메모리 사용량 | Limit 에 도달해도 OOM 이 바로 발생하지 않음 |
| `container_memory_rss` | RAM 에 실제로 올라와 있는 메모리 (익명 페이지 + 스왑 캐시) | OOM Killer 의 badness 점수와 직접 연관 |
| `container_memory_working_set_bytes` | 최근 접근된 메모리 (Usage - Inactive File Cache) | Limit 초과 시 OOM 발생 기준으로 사용됨 |

**container_memory_rss** 는 cgroup 의 `memory.stat` 에서 읽어오며, Linux OOM Killer 의 `oom_badness()` 계산에 사용된다. RSS 가 메모리 Request 에 근접하면 OOM 위험이 높다.

```
# Prometheus 알람 규칙 예시
# container_memory_rss 가 Request 의 90% 를 초과하면 알람

container_memory_rss{container!="", container!="POD"}
  /
kube_pod_container_resource_requests{job="kube-state-metrics", resource="memory", container!=""}
  >= 0.9
```

**container_memory_working_set_bytes** 는 cAdvisor 에서 다음과 같이 계산된다:

```
working_set = usage - inactive_file_cache
```

Kubernetes 는 `working_set_bytes` 가 메모리 Limit 을 초과하면 OOM 을 발생시킨다.

## kubectl top 명령어

```bash
# 모든 노드의 CPU/메모리 사용량 확인
kubectl top nodes
# NAME            CPU(cores)   CPU%   MEMORY(bytes)   MEMORY%
# worker-node-1   250m         6%     1800Mi          45%
# worker-node-2   180m         4%     2100Mi          52%

# 특정 네임스페이스의 Pod 리소스 사용량 확인
kubectl top pods -n production

# 컨테이너별 메트릭 포함
kubectl top pods -n production --containers
# POD                    NAME          CPU(cores)   MEMORY(bytes)
# web-deploy-abc12       web-server    15m          128Mi
# web-deploy-abc12       sidecar       5m           32Mi

# 메모리 사용량 기준으로 정렬
kubectl top pods -A --sort-by=memory

# CPU 사용량 기준으로 정렬
kubectl top pods -A --sort-by=cpu
```

## 주요 모니터링 대상 정리

| 모니터링 대상 | 도구 | 주요 메트릭/확인 항목 |
|---------------|------|-----------------------|
| 노드 CPU/메모리 | metrics-server, node-exporter | `kubectl top nodes`, `node_cpu_seconds_total` |
| Pod CPU/메모리 | metrics-server, cAdvisor | `kubectl top pods`, `container_memory_rss` |
| Deployment 상태 | kube-state-metrics | `kube_deployment_status_replicas_unavailable` |
| Pod 재시작 횟수 | kube-state-metrics | `kube_pod_container_status_restarts_total` |
| Node 상태 | kube-state-metrics | `kube_node_status_condition` |
| API Server 응답시간 | Prometheus | `apiserver_request_duration_seconds` |
| etcd 상태 | etcd metrics | `etcd_server_is_leader`, `etcd_disk_wal_fsync_duration_seconds` |
| HPA 동작 | kube-state-metrics | `kube_horizontalpodautoscaler_status_current_replicas` |

----

# 로깅

## kubectl logs 기본 사용법

```bash
# 단일 컨테이너 Pod 의 로그 확인
kubectl logs my-pod

# 특정 네임스페이스의 Pod 로그
kubectl logs my-pod -n production

# 로그 스트리밍 (실시간으로 출력)
kubectl logs -f my-pod

# 이전 컨테이너 인스턴스의 로그 (OOM 이나 크래시로 재시작된 경우 유용)
kubectl logs my-pod --previous

# 여러 컨테이너가 있는 Pod 에서 특정 컨테이너 로그 확인
kubectl logs my-pod -c my-container

# 최근 100줄만 출력
kubectl logs my-pod --tail=100

# 특정 시간 이후의 로그만 출력
kubectl logs my-pod --since=1h
kubectl logs my-pod --since-time="2024-01-01T10:00:00Z"

# 레이블 셀렉터로 여러 Pod 의 로그 확인
kubectl logs -l app=web-server -n production --tail=50
```

```yaml
# single-container-pod.yaml
# 단일 컨테이너 Pod 예시
apiVersion: v1
kind: Pod
metadata:
  name: event-simulator-pod
spec:
  containers:
    - name: event-simulator
      image: kodekloud/event-simulator
```

```yaml
# multi-container-pod.yaml
# 여러 컨테이너가 있는 Pod 에서는 컨테이너 이름을 명시해야 한다
apiVersion: v1
kind: Pod
metadata:
  name: event-simulator-pod
spec:
  containers:
    - name: event-simulator
      image: kodekloud/event-simulator
    - name: image-processor
      image: some-image-processor
```

```bash
# 여러 컨테이너 Pod 에서 특정 컨테이너 로그
kubectl logs -f event-simulator-pod -c event-simulator
```

## 클러스터 수준 로깅 아키텍처

컨테이너가 재시작되거나 노드가 삭제되면 `kubectl logs` 로 로그를 볼 수 없게 된다. 클러스터 수준 로깅은 로그를 외부 저장소에 영구적으로 보관한다.

```
클러스터 수준 로깅 흐름:

  컨테이너
    └─ 표준출력(stdout) / 표준에러(stderr)
         └─ 컨테이너 런타임 → /var/log/containers/
              └─ 로그 수집기 (Fluentd / Fluent Bit)
                   └─ 로그 저장소 (Elasticsearch / Loki / CloudWatch)
                        └─ 시각화 (Kibana / Grafana)
```

## DaemonSet 으로 로그 수집

Fluent Bit 을 DaemonSet 으로 배포하면 모든 노드에서 컨테이너 로그를 자동으로 수집한다.

```yaml
# fluent-bit-daemonset.yaml
# 모든 노드에 Fluent Bit 를 배포하여 /var/log/containers 의 로그를 수집한다
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluent-bit
  namespace: logging
  labels:
    app: fluent-bit
spec:
  selector:
    matchLabels:
      app: fluent-bit
  template:
    metadata:
      labels:
        app: fluent-bit
    spec:
      serviceAccountName: fluent-bit
      tolerations:
        - key: node-role.kubernetes.io/control-plane
          effect: NoSchedule
      containers:
        - name: fluent-bit
          image: fluent/fluent-bit:2.2
          ports:
            - containerPort: 2020  # Prometheus 메트릭 포트
          volumeMounts:
            - name: varlog
              mountPath: /var/log       # 호스트의 로그 디렉토리 마운트
            - name: varlibdockercontainers
              mountPath: /var/lib/docker/containers
              readOnly: true
            - name: fluent-bit-config
              mountPath: /fluent-bit/etc/
          resources:
            requests:
              cpu: 50m
              memory: 64Mi
            limits:
              cpu: 200m
              memory: 256Mi
      volumes:
        - name: varlog
          hostPath:
            path: /var/log
        - name: varlibdockercontainers
          hostPath:
            path: /var/lib/docker/containers
        - name: fluent-bit-config
          configMap:
            name: fluent-bit-config
---
# Fluent Bit 설정 ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-config
  namespace: logging
data:
  fluent-bit.conf: |
    [SERVICE]
        Flush         5
        Log_Level     info
        Parsers_File  parsers.conf

    # 컨테이너 로그 입력
    [INPUT]
        Name              tail
        Tag               kube.*
        Path              /var/log/containers/*.log
        Parser            docker
        DB                /var/log/flb_kube.db
        Mem_Buf_Limit     5MB
        Skip_Long_Lines   On
        Refresh_Interval  10

    # Kubernetes 메타데이터 보강 (Pod 이름, 네임스페이스, 레이블 추가)
    [FILTER]
        Name                kubernetes
        Match               kube.*
        Kube_URL            https://kubernetes.default.svc:443
        Kube_CA_File        /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        Kube_Token_File     /var/run/secrets/kubernetes.io/serviceaccount/token
        Merge_Log           On

    # Elasticsearch 로 로그 전송
    [OUTPUT]
        Name            es
        Match           *
        Host            elasticsearch.logging.svc.cluster.local
        Port            9200
        Logstash_Format On
        Logstash_Prefix kubernetes
        Retry_Limit     False
```

## 사이드카 패턴 로깅

애플리케이션이 파일에 로그를 기록하는 경우, 사이드카 컨테이너가 해당 파일을 읽어 표준출력으로 내보낼 수 있다.

```yaml
# sidecar-logging.yaml
# app 컨테이너는 /var/log/app.log 파일에 로그를 기록한다
# log-sidecar 컨테이너는 그 파일을 읽어서 stdout 으로 출력한다
apiVersion: v1
kind: Pod
metadata:
  name: app-with-log-sidecar
  namespace: production
spec:
  containers:
    - name: app
      image: my-app:latest
      volumeMounts:
        - name: log-volume
          mountPath: /var/log
      # 애플리케이션이 /var/log/app.log 에 로그를 기록한다고 가정

    - name: log-sidecar
      image: busybox:1.36
      command:
        - /bin/sh
        - -c
        - |
          # app 컨테이너의 로그 파일이 생성될 때까지 대기
          while [ ! -f /var/log/app.log ]; do sleep 1; done
          # 로그 파일을 실시간으로 stdout 으로 출력
          tail -f /var/log/app.log
      volumeMounts:
        - name: log-volume
          mountPath: /var/log
      resources:
        requests:
          cpu: 10m
          memory: 16Mi
        limits:
          cpu: 50m
          memory: 64Mi

  volumes:
    - name: log-volume
      emptyDir: {}   # app 과 log-sidecar 가 공유하는 임시 볼륨
```

```bash
# 사이드카 컨테이너의 로그 확인
kubectl logs app-with-log-sidecar -c log-sidecar -f
```

----

# 실무 도구

## k9s

k9s 는 터미널에서 Kubernetes 클러스터를 관리하는 인터랙티브 UI 도구다. kubectl 명령어 없이 키보드로 Pod, Deployment, Service 등을 탐색하고 관리할 수 있다.

```bash
# macOS 설치
brew install k9s

# Linux 설치 (바이너리 직접 다운로드)
curl -sS https://webinstall.dev/k9s | bash

# k9s 실행
k9s

# 특정 네임스페이스로 시작
k9s -n production

# 읽기 전용 모드로 실행 (실수로 삭제 방지)
k9s --readonly
```

k9s 주요 단축키:

| 단축키 | 동작 |
|--------|------|
| `:pods` | Pod 목록 보기 |
| `:deployments` | Deployment 목록 보기 |
| `:services` | Service 목록 보기 |
| `:nodes` | Node 목록 보기 |
| `l` | 선택한 Pod 의 로그 보기 |
| `e` | 선택한 리소스 편집 |
| `d` | 선택한 리소스 describe |
| `ctrl+d` | 선택한 리소스 삭제 |
| `s` | 선택한 Pod 에 shell 접속 |
| `/` | 검색 필터 |
| `?` | 도움말 |

## kubectx / kubens

kubectx 는 여러 클러스터 간 컨텍스트를 빠르게 전환하는 도구다. kubens 는 네임스페이스를 전환한다.

```bash
# macOS 설치
brew install kubectx

# 사용 가능한 컨텍스트 목록 확인
kubectx
# minikube
# production-cluster
# staging-cluster

# 다른 클러스터로 전환
kubectx production-cluster

# 이전 컨텍스트로 돌아가기
kubectx -

# 현재 컨텍스트 확인
kubectx -c

# 사용 가능한 네임스페이스 목록 확인
kubens
# default
# kube-system
# production
# monitoring

# 네임스페이스 전환
kubens production

# 이전 네임스페이스로 돌아가기
kubens -
```

## stern

stern 은 여러 Pod 의 로그를 동시에 tail 할 수 있는 도구다. Pod 이름의 일부나 레이블로 여러 Pod 를 선택할 수 있다.

```bash
# macOS 설치
brew install stern

# 이름에 "web" 이 포함된 모든 Pod 의 로그를 동시에 출력
stern web

# 특정 네임스페이스에서 레이블로 Pod 선택
stern -n production -l app=api-server

# 특정 컨테이너만 출력
stern web --container nginx

# 특정 시간 이후의 로그만 출력
stern web --since 30m

# 로그에서 특정 패턴 필터링 (정규식 지원)
stern web --include "ERROR|WARN"

# 출력 포맷 지정
stern web --output json

# 여러 네임스페이스에서 동시 조회
stern web --all-namespaces
```

stern 은 각 Pod 의 로그를 다른 색상으로 출력하므로 여러 Pod 의 로그를 동시에 추적할 때 유용하다.

## Telepresence

Telepresence 는 로컬 개발 환경에서 원격 Kubernetes 클러스터의 서비스에 직접 접근할 수 있게 해주는 도구다. 로컬에서 코드를 수정하면서 실제 클러스터 환경에서 테스트할 수 있다.

```bash
# macOS 설치
brew install datawire/blackbird/telepresence

# 클러스터에 Traffic Manager 설치
telepresence helm install

# 클러스터에 연결
telepresence connect

# 연결 상태 확인
telepresence status

# 특정 서비스를 로컬 프로세스로 인터셉트
# 클러스터의 api-service 로 오는 트래픽을 로컬 8080 포트로 전달
telepresence intercept api-service --port 8080:http

# 인터셉트 해제
telepresence leave api-service

# 클러스터 연결 해제
telepresence quit
```

```bash
# 인터셉트 중 로컬에서 서비스 실행 예시
# 클러스터의 api-service 로 오는 요청이 로컬 Node.js 서버로 전달된다
telepresence intercept api-service --port 3000:http
node app.js  # 로컬에서 서버 실행
```

## Lens

Lens 는 GUI 기반 Kubernetes 관리 도구(IDE)다. 여러 클러스터를 시각적으로 관리하고, 실시간 메트릭, 로그 스트리밍, 터미널 접속 등을 제공한다.

```bash
# macOS 설치 (Homebrew Cask)
brew install --cask lens

# 또는 공식 사이트에서 다운로드
# https://k8slens.dev/
```

Lens 주요 기능:

| 기능 | 설명 |
|------|------|
| 멀티 클러스터 관리 | kubeconfig 의 모든 컨텍스트를 자동으로 인식 |
| 실시간 메트릭 | CPU, 메모리 사용량 그래프 (metrics-server 필요) |
| 로그 스트리밍 | GUI 에서 Pod 로그 실시간 확인 |
| 터미널 접속 | Pod 쉘에 브라우저 내장 터미널로 접속 |
| 리소스 편집 | YAML 편집기로 리소스 직접 수정 |
| Helm 차트 관리 | 설치된 Helm 릴리스 조회 및 관리 |

```bash
# Lens 에서 사용할 kubeconfig 경로 (기본값)
# macOS/Linux: ~/.kube/config
# 여러 클러스터 설정이 있으면 모두 자동으로 인식된다

# 여러 kubeconfig 파일을 병합하는 방법
export KUBECONFIG=~/.kube/config:~/.kube/production-config:~/.kube/staging-config
kubectl config view --flatten > ~/.kube/merged-config
```

---

실무 도구 비교 요약:

| 도구 | 유형 | 주요 용도 |
|------|------|-----------|
| k9s | 터미널 UI | 빠른 클러스터 탐색 및 관리 |
| kubectx/kubens | CLI | 클러스터/네임스페이스 전환 |
| stern | CLI | 여러 Pod 로그 동시 tail |
| Telepresence | CLI | 로컬 개발과 원격 클러스터 연동 |
| Lens | GUI | 시각적 클러스터 관리 |
