# Kubernetes 한글 가이드

- [개요](#개요)
- [학습 자료](#학습-자료)
- [문서 구성](#문서-구성)
  - [기초](#기초)
  - [설정과 스토리지](#설정과-스토리지)
  - [네트워크](#네트워크)
  - [워크로드 관리](#워크로드-관리)
  - [보안](#보안)
  - [클러스터 운영](#클러스터-운영)
  - [고급](#고급)
- [학습 순서 가이드](#학습-순서-가이드)

---

# 개요

Kubernetes(K8s)는 컨테이너화된 어플리케이션의 배포, 스케일링, 관리를 자동화하는 오픈소스 플랫폼이다.

서버 10대에 컨테이너 100개를 수동으로 관리한다고 상상해 보자. 어떤 서버에 여유가 있는지 파악하고, 컨테이너가 죽으면 다시 띄우고, 트래픽이 늘면 수동으로 늘려야 한다. Kubernetes 는 이 모든 것을 자동으로 해준다.

| Kubernetes 가 해결하는 것 | 설명 |
|--------------------------|------|
| **배포 자동화** | 선언적으로 원하는 상태를 정의하면 K8s 가 알아서 배포 |
| **자기 치유** | 컨테이너가 죽으면 자동으로 재시작 |
| **스케일링** | 트래픽에 따라 자동으로 Pod 수를 늘리고 줄임 |
| **서비스 디스커버리** | Pod IP 가 바뀌어도 DNS 이름으로 접근 가능 |
| **로드 밸런싱** | 여러 Pod 에 트래픽을 균등하게 분배 |
| **롤링 업데이트** | 무중단으로 새 버전 배포, 문제 시 롤백 |

# 학습 자료

* [Kubernetes 공식 문서](https://kubernetes.io/docs/)
* [Certified Kubernetes Administrator (CKA) with Practice Tests | udemy](https://www.udemy.com/course/certified-kubernetes-administrator-with-practice-tests/)
* [subicura Kubernetes 안내서](https://subicura.com/k8s/)
* [쿠버네티스 안내서 - 설치부터 배포까지 | youtube](https://www.youtube.com/watch?v=6n5obRKsCRQ&list=PLIUCBpI1B4hULaWsBnq_SIq6jGCZZikiw)
* [토크ON세미나 쿠버네티스 | youtube](https://www.youtube.com/watch?v=xZ3tcFvbUGc&list=PLinIYkTblag5Yak-xmh_RQR4uhBbyrb8u)
* [Kubernetes The Hard Way | github](https://github.com/kelseyhightower/kubernetes-the-hard-way)
* [KodeKloud CKA Practice | kodekloud](https://kodekloud.com/courses/certified-kubernetes-administrator-cka/)

---

# 문서 구성

쉬운 내용부터 어려운 내용 순으로 정리되어 있다. 위에서 아래로 순서대로 읽으면 된다.

## 기초

| 문서 | 주요 내용 |
|------|----------|
| [k8s-basics-kr.md](./k8s-basics-kr.md) | Kubernetes 란, 아키텍처 (Control Plane, Worker Node), 설치 (minikube, kind, EKS), kubectl 기본 명령어, Pod, ReplicaSet, Deployment, Service (ClusterIP, NodePort, LoadBalancer), Namespace |

다루는 리소스: `Pod`, `ReplicaSet`, `Deployment`, `Service`, `Namespace`

## 설정과 스토리지

| 문서 | 주요 내용 |
|------|----------|
| [k8s-config-storage-kr.md](./k8s-config-storage-kr.md) | ConfigMap (환경변수, 볼륨 마운트), Secret (Opaque, docker-registry, tls), Volume (emptyDir, hostPath), PersistentVolume, PersistentVolumeClaim, StorageClass (동적 프로비저닝), 실무 패턴 |

다루는 리소스: `ConfigMap`, `Secret`, `PersistentVolume`, `PersistentVolumeClaim`, `StorageClass`

## 네트워크

| 문서 | 주요 내용 |
|------|----------|
| [k8s-networking-kr.md](./k8s-networking-kr.md) | Pod 네트워킹 원리 (veth pair, bridge, overlay, CNI), Service 네트워킹 (kube-proxy, iptables, IPVS), DNS (CoreDNS, Corefile), Headless Service, Ingress (호스트/경로 기반 라우팅, TLS), NetworkPolicy |

다루는 리소스: `Ingress`, `NetworkPolicy`, Headless `Service`

## 워크로드 관리

| 문서 | 주요 내용 |
|------|----------|
| [k8s-workload-kr.md](./k8s-workload-kr.md) | Rolling Update / Rollback, StatefulSet, DaemonSet, Job / CronJob, HPA (Horizontal Pod Autoscaler), InitContainer, Sidecar 패턴, Probe (liveness, readiness, startup), Resource Requests / Limits (QoS, LimitRange, ResourceQuota) |

다루는 리소스: `StatefulSet`, `DaemonSet`, `Job`, `CronJob`, `HorizontalPodAutoscaler`, `LimitRange`, `ResourceQuota`

## 보안

| 문서 | 주요 내용 |
|------|----------|
| [k8s-security-kr.md](./k8s-security-kr.md) | 보안 4C 모델, 인증 (X.509, OIDC), ServiceAccount, RBAC (Role, ClusterRole, Binding), PodSecurityStandard, SecurityContext, Secret 관리 모범 사례 (etcd 암호화, External Secrets Operator), Image Security |

다루는 리소스: `ServiceAccount`, `Role`, `ClusterRole`, `RoleBinding`, `ClusterRoleBinding`

## 클러스터 운영

| 문서 | 주요 내용 |
|------|----------|
| [k8s-operations-kr.md](./k8s-operations-kr.md) | 노드 관리 (drain, cordon, PodDisruptionBudget), 클러스터 업그레이드 전략 (kubeadm), etcd 백업과 복원, 모니터링 (metrics-server, kube-state-metrics, Prometheus + Grafana, 메모리 메트릭), 로깅 (DaemonSet, Sidecar), 실무 도구 (k9s, kubectx, stern, Telepresence, Lens) |

다루는 리소스: `PodDisruptionBudget`

## 고급

| 문서 | 주요 내용 |
|------|----------|
| [k8s-advanced-kr.md](./k8s-advanced-kr.md) | 스케줄링 (nodeSelector, Node/Pod Affinity, Taint/Toleration, Topology Spread Constraints), CRD (Custom Resource Definition), Operator 패턴 (KubeBuilder, Operator SDK), Admission Webhook (Mutating, Validating), Custom Metrics 와 KEDA, Helm (Chart, values, rollback) |

다루는 리소스: `CustomResourceDefinition`, `MutatingWebhookConfiguration`, `ValidatingWebhookConfiguration`, `ScaledObject` (KEDA)

---

# 학습 순서 가이드

```
1. 기초         ← Pod, Deployment, Service 를 이해하면 K8s 의 50% 를 아는 것
   ↓
2. 설정/스토리지  ← ConfigMap, Secret, PVC 로 실무 어플리케이션 구성
   ↓
3. 네트워크      ← Ingress, DNS, NetworkPolicy 로 트래픽 관리
   ↓
4. 워크로드      ← StatefulSet, HPA, Probe 로 안정적 운영
   ↓
5. 보안          ← RBAC, SecurityContext 로 프로덕션 보안 강화
   ↓
6. 운영          ← 업그레이드, 모니터링, 백업으로 클러스터 관리
   ↓
7. 고급          ← CRD, Operator, Helm 으로 확장
```

> Junior 개발자라면 1~3 까지만 먼저 익히고, 나머지는 필요할 때 찾아보면 된다.

