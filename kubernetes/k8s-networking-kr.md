# Kubernetes 네트워크

- [Pod 네트워킹 원리](#pod-네트워킹-원리)
  - [모든 Pod 은 고유 IP 를 갖는다](#모든-pod-은-고유-ip-를-갖는다)
  - [같은 노드의 Pod 간 통신](#같은-노드의-pod-간-통신)
  - [다른 노드의 Pod 간 통신](#다른-노드의-pod-간-통신)
  - [CNI 란](#cni-란)
  - [주요 CNI 플러그인](#주요-cni-플러그인)
- [Service 네트워킹 원리](#service-네트워킹-원리)
  - [kube-proxy 의 역할](#kube-proxy-의-역할)
  - [iptables 모드 vs IPVS 모드](#iptables-모드-vs-ipvs-모드)
  - [Service 에서 Pod 으로 트래픽이 전달되는 과정](#service-에서-pod-으로-트래픽이-전달되는-과정)
  - [Endpoints 와 EndpointSlice](#endpoints-와-endpointslice)
- [DNS (CoreDNS)](#dns-coredns)
  - [클러스터 내부 DNS 해석 규칙](#클러스터-내부-dns-해석-규칙)
  - [Service DNS](#service-dns)
  - [Pod DNS](#pod-dns)
  - [CoreDNS 설정 (Corefile)](#coredns-설정-corefile)
  - [DNS 디버깅](#dns-디버깅)
- [Headless Service](#headless-service)
  - [ClusterIP None 의 의미](#clusterip-none-의-의미)
  - [StatefulSet 과 Headless Service 조합](#statefulset-과-headless-service-조합)
  - [DNS 레코드 차이](#dns-레코드-차이)
  - [완전한 YAML 예제](#headless-service-yaml-예제)
- [Ingress](#ingress)
  - [Ingress 란](#ingress-란)
  - [Ingress Controller](#ingress-controller)
  - [호스트 기반 라우팅](#호스트-기반-라우팅)
  - [경로 기반 라우팅](#경로-기반-라우팅)
  - [TLS 설정](#tls-설정)
- [NetworkPolicy](#networkpolicy)
  - [기본 동작](#기본-동작)
  - [Ingress 정책](#ingress-정책)
  - [Egress 정책](#egress-정책)
  - [실무 예제](#실무-예제)

----

## Pod 네트워킹 원리

### 모든 Pod 은 고유 IP 를 갖는다

Kubernetes 의 네트워크 모델은 세 가지 기본 규칙을 따른다.

- 모든 Pod 은 클러스터 전체에서 고유한 IP 주소를 갖는다.
- 모든 Pod 은 NAT(Network Address Translation) 없이 다른 모든 Pod 과 통신할 수 있다.
- 노드의 에이전트(kubelet, 시스템 데몬 등)는 해당 노드의 모든 Pod 과 통신할 수 있다.

이 모델 덕분에 Pod 내부의 컨테이너들은 `localhost` 를 통해 서로 통신할 수 있다. Pod 하나는 하나의 네트워크 네임스페이스(Network Namespace)를 공유하기 때문이다.

```
Pod A (IP: 10.244.1.5)
  +-----------------------+
  | container-1 (port 8080)|  <-- localhost:8080 으로 container-2 접근 가능
  | container-2 (port 3306)|
  +-----------------------+
```

### 같은 노드의 Pod 간 통신

같은 노드에 있는 Pod 들은 **veth pair** 와 **Linux Bridge** 를 통해 통신한다.

```
Node
  +--------------------------------------------------+
  |  Pod A (10.244.1.5)    Pod B (10.244.1.6)        |
  |  +----------+          +----------+              |
  |  | veth0    |          | veth0    |              |
  +--|----------|----------|----------|--            |
  |  vethA                 vethB                     |
  |       \                 /                        |
  |        +---[cbr0 bridge (10.244.1.1/24)]---+    |
  |                         |                        |
  |                      eth0 (노드 NIC)             |
  +--------------------------------------------------+
```

- **veth pair**: 한쪽 끝은 Pod 내부(컨테이너의 eth0), 반대쪽 끝은 호스트 네트워크 네임스페이스에 위치하는 가상 이더넷 인터페이스 쌍이다.
- **cbr0 (Linux Bridge)**: 같은 노드의 veth 인터페이스들을 연결하는 소프트웨어 스위치다. Pod A 에서 Pod B 로 패킷이 전달될 때 이 브리지를 경유한다.

### 다른 노드의 Pod 간 통신

다른 노드에 있는 Pod 간 통신은 **오버레이 네트워크(Overlay Network)** 또는 **BGP 라우팅** 을 통해 이루어진다.

```
Node 1 (192.168.1.10)          Node 2 (192.168.1.11)
  +----------------------+       +----------------------+
  |  Pod A (10.244.1.5) |       |  Pod C (10.244.2.3) |
  |  cbr0 (10.244.1.1)  |       |  cbr0 (10.244.2.1)  |
  |  eth0               |       |  eth0               |
  +------|--------------+       +------|--------------+
         |                             |
         +--------[물리 네트워크]------+
```

오버레이 네트워크 방식(예: Flannel VXLAN)에서는 패킷이 물리 네트워크를 통과할 때 원본 패킷을 UDP 캡슐화(encapsulation)하여 전달한 뒤 목적지 노드에서 역캡슐화(decapsulation)한다. BGP 라우팅 방식(예: Calico BGP)에서는 캡슐화 없이 라우팅 테이블을 통해 직접 전달한다.

### CNI 란

**CNI(Container Network Interface)** 는 컨테이너 런타임과 네트워크 플러그인 사이의 표준 인터페이스다. kubelet 은 Pod 를 생성할 때 CNI 플러그인을 호출하여 네트워크 인터페이스 설정, IP 할당, 라우팅 규칙 추가를 위임한다.

CNI 플러그인이 담당하는 작업은 다음과 같다.

- Pod 용 veth pair 생성
- Pod 에 IP 주소 할당 (IPAM: IP Address Management)
- 브리지 또는 라우팅 테이블에 규칙 추가
- Pod 삭제 시 네트워크 자원 해제

CNI 설정 파일은 노드의 `/etc/cni/net.d/` 디렉터리에 위치하고, 플러그인 바이너리는 `/opt/cni/bin/` 에 위치한다.

### 주요 CNI 플러그인

| 플러그인 | 방식 | 특징 |
|---------|------|------|
| Flannel | VXLAN 오버레이 | 설정이 단순하여 입문에 적합 |
| Calico | BGP 라우팅 또는 VXLAN | NetworkPolicy 지원, 성능 우수 |
| Cilium | eBPF 기반 | L7 정책, 고성능, 관측 가능성 |
| Weave Net | 메쉬 오버레이 | 멀티캐스트 지원 |

----

## Service 네트워킹 원리

Pod 의 IP 는 Pod 가 재시작되거나 재스케줄링되면 바뀐다. **Service** 는 변하지 않는 가상 IP(ClusterIP)와 DNS 이름을 제공하여 클라이언트가 안정적으로 Pod 에 접근할 수 있도록 한다.

### kube-proxy 의 역할

**kube-proxy** 는 모든 노드에서 DaemonSet 형태로 실행되며, API 서버를 감시하여 Service 와 Endpoints 변경 사항을 감지하고 그에 맞는 네트워크 규칙을 노드에 적용한다.

```
[kubectl apply -f service.yaml]
        |
        v
  kube-apiserver
        |
        v
  kube-proxy (각 노드에서 실행)
        |
        v
  iptables / IPVS 규칙 업데이트
```

### iptables 모드 vs IPVS 모드

**iptables 모드** (기본값):

- Service 의 ClusterIP 로 향하는 패킷을 실제 Pod IP 로 DNAT(Destination NAT) 한다.
- 규칙이 체인 형태로 순차 평가되므로 Service 수가 많아지면 성능이 저하된다.
- 로드밸런싱은 랜덤(random) 방식이다.

**IPVS 모드**:

- 리눅스 커널의 IPVS(IP Virtual Server)를 사용하여 해시 테이블 기반으로 패킷을 처리한다.
- Service 수가 수천 개가 넘는 대규모 클러스터에서 iptables 보다 성능이 좋다.
- Round Robin, Least Connection, Source Hashing 등 다양한 로드밸런싱 알고리즘을 지원한다.

IPVS 모드를 활성화하려면 kube-proxy ConfigMap 에서 `mode: "ipvs"` 로 설정하고 노드에 `ip_vs` 커널 모듈이 로드되어 있어야 한다.

### Service 에서 Pod 으로 트래픽이 전달되는 과정

```
클라이언트 Pod
  --> ClusterIP:Port (가상 IP)
  --> iptables/IPVS 규칙 (DNAT)
  --> 실제 Pod IP:Port
```

1. 클라이언트가 Service 의 ClusterIP(예: 10.96.80.10)로 요청을 보낸다.
2. 해당 노드의 iptables 또는 IPVS 규칙이 패킷을 가로채어 목적지 IP 를 실제 Pod IP 로 변환(DNAT)한다.
3. 패킷이 실제 Pod 에 도달한다.
4. 응답 패킷은 역방향으로 SNAT 되어 클라이언트에게 돌아온다.

ClusterIP 는 어떤 노드에도 실제로 할당되지 않는 가상 IP 이다. iptables 규칙에 의해서만 존재한다.

### Endpoints 와 EndpointSlice

Service 의 `selector` 와 일치하는 Pod 의 IP 목록을 **Endpoints** 오브젝트가 저장한다.

```bash
# Service 와 Endpoints 함께 확인
kubectl get svc,endpoints -n default

# Endpoints 상세 확인
kubectl describe endpoints my-service
```

대규모 클러스터에서 Endpoints 오브젝트 하나에 수천 개의 IP 가 담기면 etcd 와 네트워크 부하가 커진다. 이를 해결하기 위해 Kubernetes 1.17 부터 **EndpointSlice** 가 도입되었다. EndpointSlice 는 최대 100개 단위로 Endpoints 를 분할하여 관리한다.

```bash
# EndpointSlice 확인
kubectl get endpointslices -n default
```

----

## DNS (CoreDNS)

### 클러스터 내부 DNS 해석 규칙

Kubernetes 클러스터는 **CoreDNS** 를 DNS 서버로 사용한다. 각 Pod 의 `/etc/resolv.conf` 는 kubelet 에 의해 자동으로 설정된다.

```bash
# Pod 내부에서 /etc/resolv.conf 확인
kubectl run -it --rm netdebug --image=nicolaka/netshoot --restart=Never -- cat /etc/resolv.conf
```

출력 예시:

```
nameserver 10.96.0.10          # CoreDNS Service 의 ClusterIP
search default.svc.cluster.local svc.cluster.local cluster.local
options ndots:5
```

- `nameserver`: Pod 가 DNS 쿼리를 보낼 CoreDNS Service 의 IP 이다.
- `search`: 짧은 이름(예: `my-svc`)으로 쿼리할 때 자동으로 붙이는 접미사 목록이다.
- `ndots:5`: 도메인 이름에 점(`.`)이 5개 미만이면 `search` 의 접미사를 먼저 붙여서 시도한다. 5개 이상이면 FQDN 으로 바로 쿼리한다.

DNS 쿼리 흐름:

```
Pod (/etc/resolv.conf) --> CoreDNS --> 노드 (/etc/resolv.conf) --> 외부 DNS 서버
```

### Service DNS

Service 에는 다음과 같은 FQDN(Fully Qualified Domain Name)이 자동으로 부여된다.

```
<service-name>.<namespace>.svc.cluster.local
```

| 호출 위치 | 사용 가능한 이름 |
|----------|----------------|
| 같은 namespace | `my-svc` |
| 다른 namespace | `my-svc.my-namespace` |
| 어디서든 | `my-svc.my-namespace.svc.cluster.local` |

예시: `default` namespace 의 `frontend` Service 에 접근하는 경우:

```bash
# 같은 namespace 에서
curl http://frontend

# 다른 namespace 에서
curl http://frontend.default

# FQDN
curl http://frontend.default.svc.cluster.local
```

### Pod DNS

Pod 에도 DNS 레코드가 생성된다. Pod IP 의 점(`.`)을 하이픈(`-`)으로 바꾼 형태다.

```
<pod-ip-with-dashes>.<namespace>.pod.cluster.local
```

예를 들어 IP 가 `10.244.1.5` 인 Pod 의 DNS 이름은 다음과 같다.

```
10-244-1-5.default.pod.cluster.local
```

Pod DNS 는 일반적으로 직접 사용하지 않는다. StatefulSet 과 Headless Service 를 함께 쓸 때 개별 Pod 를 식별하는 용도로 활용한다.

### CoreDNS 설정 (Corefile)

CoreDNS 의 설정은 `kube-system` namespace 의 ConfigMap `coredns` 에 저장된다.

```bash
kubectl -n kube-system describe configmaps coredns
```

Corefile 예시:

```
.:53 {
    errors                          # 에러를 stdout 으로 출력
    health {
       lameduck 5s                  # 셧다운 전 5초 유예시간
    }
    ready                           # readiness probe 엔드포인트 활성화
    kubernetes cluster.local in-addr.arpa ip6.arpa {
       pods insecure                # Pod A 레코드 응답 (insecure: kube-dns 하위호환)
       fallthrough in-addr.arpa ip6.arpa
       ttl 30                       # 쿼리 캐시 TTL (초)
    }
    prometheus :9153                # Prometheus 메트릭 노출 포트
    forward . /etc/resolv.conf {    # 클러스터 외부 쿼리를 노드 DNS 서버로 전달
       max_concurrent 1000
    }
    cache 30                        # DNS 응답 캐시 시간 (초)
    loop                            # 루프 감지 시 CoreDNS 중단
    reload                          # Corefile 변경 시 자동 리로드
    loadbalance                     # A/AAAA/MX 레코드 응답 순서를 랜덤하게 섞음
}
```

주요 지시어 설명:

| 지시어 | 설명 |
|-------|------|
| `errors` | 에러 로그를 표준 출력으로 보냄 |
| `health` | `http://<CoreDNS-Pod-IP>:8080/health` 로 헬스 체크 |
| `ready` | `http://<CoreDNS-Pod-IP>:8181/ready` 로 준비 상태 체크 |
| `kubernetes` | 클러스터 내부 DNS 쿼리 처리 (Service, Pod 레코드) |
| `forward` | 외부 도메인 쿼리를 업스트림 DNS 서버로 전달 |
| `cache` | 쿼리 결과를 지정된 시간(초) 동안 캐시 |
| `reload` | ConfigMap 변경을 자동 감지하여 재설정 |

특정 도메인에 대한 쿼리를 다른 DNS 서버로 보내려면 `forward` 블록을 추가한다.

```
# 예: acme.local 도메인은 사내 DNS 서버 1.2.3.4 로 전달
acme.local:53 {
    forward . 1.2.3.4
}
```

### DNS 디버깅

```bash
# 디버그 Pod 실행 (nicolaka/netshoot 이미지 사용)
kubectl run -it --rm netdebug --image=nicolaka/netshoot --restart=Never -- bash

# nslookup 으로 Service 조회
nslookup my-service.default.svc.cluster.local

# dig 으로 상세 조회
dig my-service.default.svc.cluster.local

# 외부 도메인 조회 시 search 접미사 적용 과정 확인
nslookup -type=A google.com -debug

# CoreDNS Pod 로그 확인
kubectl -n kube-system logs -l k8s-app=kube-dns

# CoreDNS Service 확인
kubectl -n kube-system get svc kube-dns
```

CoreDNS 가 특정 도메인을 해석하지 못하는 경우 체크 포인트:

1. `kubectl get pods -n kube-system` 으로 CoreDNS Pod 가 Running 상태인지 확인한다.
2. `kubectl -n kube-system logs <coredns-pod>` 로 에러 메시지를 확인한다.
3. Pod 의 `/etc/resolv.conf` 가 올바른 CoreDNS ClusterIP 를 가리키는지 확인한다.
4. NetworkPolicy 가 UDP/TCP 포트 53 트래픽을 차단하고 있지 않은지 확인한다.

----

## Headless Service

### ClusterIP None 의 의미

일반 Service 는 kube-proxy 가 만든 가상 IP(ClusterIP)를 통해 여러 Pod 에 로드밸런싱한다. **Headless Service** 는 `spec.clusterIP: None` 을 설정하여 가상 IP 를 갖지 않는 Service 다.

Headless Service 에 DNS 쿼리를 보내면:

- 일반 Service: ClusterIP 하나가 반환된다.
- Headless Service: 셀렉터에 매칭되는 모든 Pod 의 IP 목록이 A 레코드로 반환된다.

```
일반 Service DNS 쿼리:
  my-svc.default.svc.cluster.local --> 10.96.80.10 (ClusterIP)

Headless Service DNS 쿼리:
  my-svc.default.svc.cluster.local --> 10.244.1.5, 10.244.2.3, 10.244.3.7 (Pod IP 목록)
```

로드밸런싱이 필요 없거나 클라이언트가 특정 Pod 에 직접 접근해야 할 때 Headless Service 를 사용한다. 대표적인 사례가 StatefulSet 기반 데이터베이스 클러스터다.

### StatefulSet 과 Headless Service 조합

StatefulSet 은 Pod 에 안정적인 순서 번호(0, 1, 2, ...)와 고정 호스트명을 부여한다. Headless Service 와 함께 사용하면 각 Pod 에 예측 가능한 DNS 이름이 부여된다.

```
<pod-name>.<headless-svc-name>.<namespace>.svc.cluster.local
```

예를 들어 `web` StatefulSet 과 `nginx` Headless Service 가 있다면:

```
web-0.nginx.default.svc.cluster.local  --> Pod web-0 의 IP
web-1.nginx.default.svc.cluster.local  --> Pod web-1 의 IP
web-2.nginx.default.svc.cluster.local  --> Pod web-2 의 IP
```

이 덕분에 MySQL 이나 Kafka 같은 클러스터 소프트웨어가 특정 노드(리더, 팔로워)에 직접 접근할 수 있다.

### DNS 레코드 차이

```
일반 Service (ClusterIP: 10.96.80.10)
  DNS 조회: my-svc.default.svc.cluster.local
  응답:     A 10.96.80.10

Headless Service (ClusterIP: None)
  DNS 조회: my-svc.default.svc.cluster.local
  응답:     A 10.244.1.5
            A 10.244.2.3
            A 10.244.3.7

StatefulSet Pod 직접 접근
  DNS 조회: web-0.my-svc.default.svc.cluster.local
  응답:     A 10.244.1.5
```

### Headless Service YAML 예제

```yaml
# headless-statefulset.yaml
# Headless Service: ClusterIP 없이 Pod IP 를 직접 반환
apiVersion: v1
kind: Service
metadata:
  name: nginx                          # StatefulSet 의 serviceName 과 일치해야 함
  namespace: default
  labels:
    app: nginx
spec:
  clusterIP: None                      # Headless Service 설정의 핵심
  selector:
    app: nginx                         # 이 레이블을 가진 Pod 의 IP 를 DNS 로 반환
  ports:
  - port: 80
    name: web

---

# StatefulSet: 순서 있는 Pod 을 생성하며 각 Pod 에 고정 DNS 이름 부여
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: web
  namespace: default
spec:
  selector:
    matchLabels:
      app: nginx                       # Service selector 와 동일한 레이블
  serviceName: "nginx"                 # 위에서 만든 Headless Service 이름
  replicas: 3                          # web-0, web-1, web-2 세 Pod 생성
  template:
    metadata:
      labels:
        app: nginx
    spec:
      terminationGracePeriodSeconds: 10
      containers:
      - name: nginx
        image: nginx:1.25
        ports:
        - containerPort: 80
          name: web
        volumeMounts:
        - name: www
          mountPath: /usr/share/nginx/html  # 각 Pod 마다 독립적인 볼륨 마운트
  volumeClaimTemplates:                # Pod 마다 개별 PVC 를 자동 생성
  - metadata:
      name: www
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: "standard"
      resources:
        requests:
          storage: 1Gi
```

배포 및 확인:

```bash
kubectl apply -f headless-statefulset.yaml

# Pod 가 순서대로 생성되는 것을 확인 (web-0 -> web-1 -> web-2)
kubectl get pods -w

# Headless Service 확인: CLUSTER-IP 가 None 임을 확인
kubectl get svc nginx

# DNS 조회: 모든 Pod IP 가 반환되어야 함
kubectl run -it --rm netdebug --image=nicolaka/netshoot --restart=Never -- \
  nslookup nginx.default.svc.cluster.local

# 특정 Pod DNS 조회
kubectl run -it --rm netdebug --image=nicolaka/netshoot --restart=Never -- \
  nslookup web-0.nginx.default.svc.cluster.local
```

----

## Ingress

### Ingress 란

**Ingress** 는 클러스터 외부에서 들어오는 HTTP/HTTPS 트래픽을 클러스터 내부 Service 로 라우팅하는 L7(애플리케이션 계층) 로드밸런서 규칙 오브젝트다.

Ingress 가 없다면 외부에 서비스를 노출하려면 Service 타입을 LoadBalancer 또는 NodePort 로 설정해야 한다. 서비스마다 별도의 로드밸런서가 필요해지므로 비용이 증가한다. Ingress 는 하나의 진입점에서 호스트 이름과 URL 경로를 기준으로 여러 Service 로 트래픽을 분배한다.

```
외부 클라이언트
      |
      v
[Ingress Controller (nginx, ALB 등)]
      |
      +-- app.example.com/api  --> api-service:8080
      |
      +-- app.example.com/web  --> web-service:80
      |
      +-- admin.example.com    --> admin-service:9090
```

Ingress 오브젝트 자체는 규칙 선언일 뿐이다. 실제로 트래픽을 처리하는 것은 **Ingress Controller** 다.

### Ingress Controller

**Ingress Controller** 는 Ingress 오브젝트를 감시하고, 선언된 규칙에 따라 실제 프록시(nginx, Envoy 등)를 설정하는 컨트롤러다. 클러스터에 기본으로 설치되지 않으므로 별도로 설치해야 한다.

주요 Ingress Controller:

| 컨트롤러 | 특징 |
|---------|------|
| ingress-nginx | nginx 기반. 가장 널리 사용됨 |
| AWS ALB Ingress Controller | AWS Application Load Balancer 사용 |
| Traefik | 자동 설정, Let's Encrypt 내장 |
| HAProxy Ingress | 고성능, 세밀한 제어 |

ingress-nginx 설치:

```bash
# Helm 으로 설치
helm upgrade --install ingress-nginx ingress-nginx \
  --repo https://kubernetes.github.io/ingress-nginx \
  --namespace ingress-nginx --create-namespace

# 설치 확인
kubectl get pods -n ingress-nginx
kubectl get svc -n ingress-nginx
```

### 호스트 기반 라우팅

호스트 이름(도메인)에 따라 다른 Service 로 트래픽을 전달한다.

```yaml
# ingress-host-based.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: host-based-ingress
  namespace: default
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /  # 요청 경로를 / 로 재작성
spec:
  ingressClassName: nginx                           # 사용할 Ingress Controller 지정
  rules:
  - host: app.example.com                           # 이 도메인으로 들어오는 요청
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: app-service                       # app-service 로 전달
            port:
              number: 80
  - host: admin.example.com                         # 이 도메인으로 들어오는 요청
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: admin-service                     # admin-service 로 전달
            port:
              number: 9090
```

### 경로 기반 라우팅

같은 호스트에서 URL 경로에 따라 다른 Service 로 트래픽을 전달한다.

```yaml
# ingress-path-based.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: path-based-ingress
  namespace: default
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /$2  # 캡처 그룹 $2 로 경로 재작성
spec:
  ingressClassName: nginx
  rules:
  - host: app.example.com
    http:
      paths:
      - path: /api(/|$)(.*)                           # /api 로 시작하는 경로
        pathType: ImplementationSpecific
        backend:
          service:
            name: api-service                         # api-service:8080 으로 전달
            port:
              number: 8080
      - path: /web(/|$)(.*)                           # /web 으로 시작하는 경로
        pathType: ImplementationSpecific
        backend:
          service:
            name: web-service                         # web-service:80 으로 전달
            port:
              number: 80
      - path: /                                       # 나머지 모든 경로 (기본값)
        pathType: Prefix
        backend:
          service:
            name: default-service
            port:
              number: 80
```

`pathType` 값 설명:

| pathType | 동작 |
|----------|------|
| `Exact` | 경로가 정확히 일치해야 함 |
| `Prefix` | `/` 로 구분된 접두사가 일치하면 됨 |
| `ImplementationSpecific` | Ingress Controller 가 자체적으로 해석 |

### TLS 설정

HTTPS 를 활성화하려면 TLS 인증서를 Secret 으로 저장하고 Ingress 에서 참조한다.

```bash
# 자체 서명 인증서 생성 (개발/테스트용)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key \
  -out tls.crt \
  -subj "/CN=app.example.com/O=my-org"

# Secret 생성
kubectl create secret tls app-tls-secret \
  --key tls.key \
  --cert tls.crt \
  --namespace default
```

```yaml
# ingress-tls.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tls-ingress
  namespace: default
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"  # HTTP -> HTTPS 자동 리다이렉트
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - app.example.com                                  # TLS 를 적용할 도메인
    secretName: app-tls-secret                         # 인증서가 담긴 Secret 이름
  rules:
  - host: app.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: app-service
            port:
              number: 80
```

배포 및 확인:

```bash
kubectl apply -f ingress-tls.yaml

# Ingress 상태 확인 (ADDRESS 필드에 로드밸런서 IP 가 나타나야 함)
kubectl get ingress tls-ingress

# HTTPS 요청 테스트 (-k 는 자체 서명 인증서 검증 무시)
curl -k https://app.example.com/

# HTTP 는 자동으로 HTTPS 로 리다이렉트되는지 확인
curl -v http://app.example.com/
```

----

## NetworkPolicy

### 기본 동작

Kubernetes 는 기본적으로 **모든 Pod 간 트래픽을 허용**한다. NetworkPolicy 오브젝트를 적용하면 특정 Pod 에 대한 트래픽을 제어할 수 있다.

중요한 점은 NetworkPolicy 는 **Pod 를 격리하는 것이 아니라 허용 규칙을 선언**한다는 것이다.

- 특정 Pod 에 NetworkPolicy 가 하나라도 선택되면, 그 Policy 에서 명시적으로 허용하지 않은 트래픽은 모두 차단된다.
- NetworkPolicy 가 없는 Pod 는 모든 트래픽이 허용된다.

NetworkPolicy 는 CNI 플러그인이 지원해야 동작한다. Calico, Cilium, Weave Net 등은 지원하지만 Flannel 기본 설정은 지원하지 않는다.

### Ingress 정책

Pod 로 **들어오는** 트래픽을 제어한다.

```yaml
# netpol-ingress.yaml
# web Pod 로 들어오는 트래픽: frontend namespace 의 Pod 에서 80 포트만 허용
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-to-web
  namespace: default
spec:
  podSelector:
    matchLabels:
      app: web                              # 이 Policy 가 적용될 Pod 선택
  policyTypes:
  - Ingress                                 # Ingress 규칙만 적용 (Egress 는 제한 없음)
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: frontend                    # frontend namespace 에서 오는 트래픽만 허용
    - podSelector:
        matchLabels:
          role: frontend-pod               # 또는 role=frontend-pod 레이블을 가진 Pod 허용
    ports:
    - protocol: TCP
      port: 80                              # 80 포트만 허용
```

주의: `from` 아래에 여러 항목을 나열하면 **OR** 조건이다. `namespaceSelector` 와 `podSelector` 를 같은 항목 안에 쓰면 **AND** 조건이다.

```yaml
# AND 조건: frontend namespace 이면서 role=frontend-pod 인 Pod 만 허용
ingress:
- from:
  - namespaceSelector:
      matchLabels:
        name: frontend
    podSelector:                            # 같은 리스트 항목 안에 있으면 AND
      matchLabels:
        role: frontend-pod
```

### Egress 정책

Pod 에서 **나가는** 트래픽을 제어한다.

```yaml
# netpol-egress.yaml
# api Pod 에서 나가는 트래픽: database namespace 의 5432 포트와 DNS(53) 만 허용
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-api-egress
  namespace: default
spec:
  podSelector:
    matchLabels:
      app: api                              # 이 Policy 가 적용될 Pod 선택
  policyTypes:
  - Egress                                  # Egress 규칙만 적용
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: database                    # database namespace 로 나가는 트래픽 허용
    ports:
    - protocol: TCP
      port: 5432                            # PostgreSQL 포트만 허용
  - to:
    - namespaceSelector: {}                 # 모든 namespace 포함 (DNS 용)
    ports:
    - protocol: UDP
      port: 53                              # DNS 쿼리 허용 (이것이 없으면 DNS 해석 실패)
    - protocol: TCP
      port: 53
```

Egress 정책을 적용할 때 DNS 포트(53)를 허용하지 않으면 Pod 가 DNS 해석을 못해 외부 서비스와 통신이 불가능해지는 흔한 실수가 발생한다.

### 실무 예제

실무에서 자주 사용되는 패턴: **특정 namespace 에서만 DB Pod 에 접근 허용**

시나리오: `database` namespace 에 PostgreSQL Pod 가 있고, `backend` namespace 의 Pod 만 접근을 허용하고 싶다.

```bash
# namespace 에 레이블 추가 (NetworkPolicy 에서 namespaceSelector 로 참조하기 위해 필요)
kubectl label namespace backend name=backend
kubectl label namespace database name=database
```

```yaml
# netpol-db-access.yaml
# database namespace 의 postgres Pod: backend namespace 의 Pod 에서 5432 포트만 허용
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: postgres-access-policy
  namespace: database                       # 이 Policy 는 database namespace 에 적용
spec:
  podSelector:
    matchLabels:
      app: postgres                         # postgres Pod 에 이 Policy 적용
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: backend                     # backend namespace 에서 오는 트래픽만 허용
      podSelector:
        matchLabels:
          app: backend-api                  # backend-api 레이블을 가진 Pod 만 허용 (AND 조건)
    ports:
    - protocol: TCP
      port: 5432                            # PostgreSQL 포트만 허용
---

# 모든 트래픽 차단 기준선 (baseline deny-all)
# database namespace 에 적용하여 명시적으로 허용하지 않은 모든 Ingress 트래픽 차단
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-ingress
  namespace: database
spec:
  podSelector: {}                           # namespace 의 모든 Pod 에 적용
  policyTypes:
  - Ingress                                 # 모든 Ingress 트래픽 차단 (허용 규칙 없음)
```

배포 및 검증:

```bash
kubectl apply -f netpol-db-access.yaml

# NetworkPolicy 확인
kubectl get networkpolicy -n database
kubectl describe networkpolicy postgres-access-policy -n database

# backend namespace 의 Pod 에서 postgres 접근 시도 (성공해야 함)
kubectl run -it --rm test-allowed --image=nicolaka/netshoot \
  --namespace=backend --restart=Never -- \
  nc -zv postgres.database.svc.cluster.local 5432

# default namespace 의 Pod 에서 postgres 접근 시도 (차단되어야 함)
kubectl run -it --rm test-denied --image=nicolaka/netshoot \
  --namespace=default --restart=Never -- \
  nc -zv postgres.database.svc.cluster.local 5432
```

NetworkPolicy 는 방화벽 규칙이 아니라 선언적 허용 목록이다. 정책 적용 시 항상 deny-all 을 기준선으로 두고 필요한 트래픽만 허용하는 방식을 권장한다.
