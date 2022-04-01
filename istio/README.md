- [Abstract](#abstract)
- [References](#references)
- [Materials](#materials)
- [Basic](#basic)
  - [Install](#install)
  - [Architecture](#architecture)
  - [Traffic Management](#traffic-management)
    - [Virtual services](#virtual-services)
    - [Destination rules](#destination-rules)
    - [Gateways](#gateways)
    - [Service entries](#service-entries)
    - [Sidecars](#sidecars)
  - [Bookinfo Application](#bookinfo-application)
  - [Dive Deep Into Istio Traffics](#dive-deep-into-istio-traffics)
    - [Request from Client to POD](#request-from-client-to-pod)
    - [Response from POD to Client](#response-from-pod-to-client)
    - [Request from POD to External Server](#request-from-pod-to-external-server)
    - [Response from External to POD](#response-from-external-to-pod)
    - [Optimization Of Traffics](#optimization-of-traffics)
      - [Merbridge](#merbridge)
      - [Cilium CNI](#cilium-cni)

----

# Abstract

**Istio** is a completely open source service mesh that layers transparently onto existing distributed applications.

A **service mesh** is a dedicated infrastructure layer for handling service-to-service communication. A service mesh is consisted of **control plane** and **data plane**.

* The **data plane** is composed of a set of **intelligent proxies (Envoy)** deployed as sidecars.
* The **control plane** manages and configures the proxies to route traffic.

Istio provides these features. [Feature Status](https://istio.io/latest/docs/releases/feature-stages/)

* Traffic Management
  * Request Routing
  * Fault Injection
  * Traffic Shifting
  * TCP Traffic Shifting
  * Request Timeouts
  * Circuit Breaking
  * Mirroring
  * Locality Load Balancing
  * Ingress
  * Egress
* Security
  * Certificate Management
  * Authentication
  * Authorization
* Observability
  * Telemetry API
  * Metrics
  * Logs
  * Distributed Tracing
  * Visualizing Your Mesh
  * Remotely Accessing Telemetry Addons 
* Extensibility
  * WebAssembly

# References

* [Documentation](https://istio.io/latest/docs/)
  * [src](https://github.com/istio/istio)

# Materials

* [Istio 트래픽 흐름 @ youtube](https://www.youtube.com/playlist?list=PLDoAIZhHTMvPIY7PHDtMaqbUUWEbE6-6H)
  * [Istio 🌶️ 트래픽 흐름 Life of a packet @ notion](https://gasidaseo.notion.site/Istio-Life-of-a-packet-6ad9808e14594296bf854dcc203cab71)
* [Getting Started @ istio.io](https://istio.io/latest/docs/setup/getting-started/)
* [Service Mesh Comparison](https://servicemesh.es/)
  * comparison of Istio, Linkerd, Kuma
* [istio @ eksworkshop](https://www.eksworkshop.com/advanced/310_servicemesh_with_istio/)
* [Istio Service Mesh](https://daddyprogrammer.org/post/13721/istio-service-mesh/)

# Basic

## Install

[Getting Started @ istio.io](https://istio.io/latest/docs/setup/getting-started/) 를 참고하여 minikube 에 install 해보자.

## Architecture

> [Architecture](https://istio.io/latest/docs/ops/deployment/architecture/)

![](img/istio_arch.png)

## Traffic Management

Istio 가 제공하는 기능중 Traffic Management 는 기본적인 것이다. 잘 알아두자.
Istio 는 Traffic Management 를 위해 다음과 같은 Resource 들을 이용한다.

* Virtual services
* Destination rules
* Gateways
* Service entries
* Sidecars

### Virtual services
### Destination rules
### Gateways
### Service entries
### Sidecars

## Bookinfo Application

* [Bookinfo Application @ istio.io](https://istio.io/latest/docs/examples/bookinfo/)
  * [src](https://github.com/istio/istio/tree/master/samples/bookinfo) 

[Bookinfo Application @ istio.io](https://istio.io/latest/docs/examples/bookinfo/) 를 참고로 중요한 manifest file 을 익혀본다.

istio 를 적용하기 전 system architecture 는 다음과 같다. 

![](img/bookinfo_architecture_without_istio.png)

다음은 istio 를 적용한 system architecture 이다. 

![](img/bookinfo_architecture_with_istio.png)

각 POD 에 istio proxy conntainer 가 side 로 inject 되어 있다. 이렇게 istio proxy container 가 포함된 POD 들을 **data plane** 이라고 한다. 또한 istiod 와 같이 istio proxy 와 통신하면서 service mesh 의 control tower 역할을 하는 것을 **control plane** 이라고 한다.

이제 [Bookinfo Application @ istio.io](https://istio.io/latest/docs/examples/bookinfo/)를 참고하여 bookinfo service 를 설치해보자.

[samples/bookinfo/platform/kube/bookinfo.yaml](https://raw.githubusercontent.com/istio/istio/release-1.10/samples/bookinfo/platform/kube/bookinfo.yaml) 을 이용하여 Service, Deployment 를 설치한다.

[samples/bookinfo/networking/bookinfo-gateway.yaml](https://raw.githubusercontent.com/istio/istio/release-1.10/samples/bookinfo/networking/bookinfo-gateway.yaml) 을 이용하여 **Gateway, VirtualService** 를 설치한다. 이제 외부 트래픽을 Pod 에서 받을 수 있다.

```yml
# samples/bookinfo/networking/bookinfo-gateway.yaml
apiVersion: networking.istio.io/v1alpha3
kind: Gateway
metadata:
  name: bookinfo-gateway
spec:
  selector:
    istio: ingressgateway # use istio default controller
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "*"
---
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: bookinfo
spec:
  hosts:
  - "*"
  gateways:
  - bookinfo-gateway
  http:
  - match:
    - uri:
        exact: /productpage
    - uri:
        prefix: /static
    - uri:
        exact: /login
    - uri:
        exact: /logout
    - uri:
        prefix: /api/v1/products
    route:
    - destination:
        host: productpage
        port:
          number: 9080
```

[samples/bookinfo/networking/destination-rule-all.yaml](https://raw.githubusercontent.com/istio/istio/release-1.10/samples/bookinfo/networking/destination-rule-all.yaml) 을 이용하여 **DestinationRule** 을 설치한다. version 에 따라 traffic 을 management 하는 rule 을 생성할 수 있다.

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: productpage
spec:
  host: productpage
  subsets:
  - name: v1
    labels:
      version: v1
---
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: reviews
spec:
  host: reviews
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
  - name: v3
    labels:
      version: v3
---
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: ratings
spec:
  host: ratings
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
  - name: v2-mysql
    labels:
      version: v2-mysql
  - name: v2-mysql-vm
    labels:
      version: v2-mysql-vm
---
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: details
spec:
  host: details
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
---
```

istio 를 적용한 bookinfo example 의 network traffic 흐름은 다음과 같다.

![](img/bookinfo_network_traffic.png)

## Dive Deep Into Istio Traffics

> * [Istio 트래픽 흐름 @ youtube](https://www.youtube.com/playlist?list=PLDoAIZhHTMvPIY7PHDtMaqbUUWEbE6-6H)
>   * [Istio 🌶️ 트래픽 흐름 Life of a packet @ notion](https://gasidaseo.notion.site/Istio-Life-of-a-packet-6ad9808e14594296bf854dcc203cab71)
> * [How eBPF will solve Service Mesh - Goodbye Sidecars](https://isovalent.com/blog/post/2021-12-08-ebpf-servicemesh)
> * [Try eBPF-powered Cilium Service Mesh - join the beta program!](https://cilium.io/blog/2021/12/01/cilium-service-mesh-beta)

다음은 Kubernetes Node 의 Traffic 흐름을 표현한 것이다. Node 의 `eth0` network
interface 으로 넘어온 패킷이 `veth, veth, loopback` network interface 를 거쳐
app 으로 전달된다. 많은 수의 Network Interface 를 지나기 때문에 비효율적이다.

![](img/service_mesh_traffics.png)

다음은 Kubernetes Node 의 POD 에서 외부로 Request 를 요청했을 때 Traffic 의 흐름이다. 역시 많은 수의 Network Interface 를 지나기 때문에 비효율적이다.

![](img/cost_of_sidecar_injection.png)

### Request from Client to POD

* [1.1 클라이언트(요청) → 파드(인입)](https://gasidaseo.notion.site/Istio-Life-of-a-packet-6ad9808e14594296bf854dcc203cab71#5ed7095cfbf74fe3b89d8c96f66d780b)

### Response from POD to Client

* [1.2 파드(리턴 트래픽) → 클라이언트](https://gasidaseo.notion.site/Istio-Life-of-a-packet-6ad9808e14594296bf854dcc203cab71#710f224348d2435e806bb1bc4d14a5f5)

### Request from POD to External Server

* [2.1 파드(요청) → 외부 웹서버](https://gasidaseo.notion.site/Istio-Life-of-a-packet-6ad9808e14594296bf854dcc203cab71#d51cdb24177c4c25952e08f4486132b7)

### Response from External to POD

* [2.2 외부 웹서버(리턴 트래픽) → 파드](https://gasidaseo.notion.site/Istio-Life-of-a-packet-6ad9808e14594296bf854dcc203cab71#97bfb642beea4cdab6daa87b4c962763)

### Optimization Of Traffics

#### Merbridge

[Merbridge](https://istio.io/latest/blog/2022/merbridge/) 를 사용하면 [eBPF](/bpf/README.md) 을 이용하여 Traffic 을 최적화 할 수 있다고 한다.

아래는 [Merbridge](https://istio.io/latest/blog/2022/merbridge/) 를 사용하기 전의 모습이다.

![](img/merbridge_iptables.png)

아래는 [Merbridge](https://istio.io/latest/blog/2022/merbridge/) 를 사용한 모습이다.

![](img/merbridge_ebpf.png)

아래는 [Merbridge](https://istio.io/latest/blog/2022/merbridge/) 를 사용하고 같은 Node 위에서 실행된 POD 들의 모습이다.

![](img/merbridge_ebpf_same_machine.png)

#### Cilium CNI

[Cilium CNI](https://cilium.io/blog/2021/12/01/cilium-service-mesh-beta) 을 사용하면 다음과 같이 sidecar 없이 traffic routing 이 가능하다고 한다. [Cilium CNI](https://cilium.io/blog/2021/12/01/cilium-service-mesh-beta) 은 [eBPF](/bpf/README.md) 를 이용한다.

![](img/service_mesh_sidecarless.png)
