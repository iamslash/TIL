- [Materials](#materials)
- [Concept](#concept)
- [Aggregation Layer](#aggregation-layer)
- [Dynamic Admission Control](#dynamic-admission-control)
- [Custom Resource](#custom-resource)
- [Custom Controller](#custom-controller)
- [Custom Scheduler](#custom-scheduler)
- [Custom Metric Server](#custom-metric-server)

-----

# Materials

* [Kubernetes Documentation / Concepts / Extending Kubernetes @ kubernetes.io](https://kubernetes.io/docs/concepts/extend-kubernetes/)

# Concept

![](https://docs.google.com/drawings/d/e/2PACX-1vQBRWyXLVUlQPlp7BvxvV9S1mxyXSM6rAc_cbLANvKlu6kCCf-kGTporTMIeG5GZtUdxXz1xowN7RmL/pub?w=960&h=720)

# Aggregation Layer

> * [Extending the Kubernetes API with the aggregation layer @ kubernetes.io](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/apiserver-aggregation/)

# Dynamic Admission Control

> * [172. [Kubernetes] Admission Controller의 이해 및 Python으로 Mutate Webhook 작성 예제 @ naverblog](https://blog.naver.com/alice_k106/221546328906)

# Custom Resource

`CustomResourceDefinition` 을 이용하여 Kubernetes 의 Resource 를 정의할수 있다.

* `my-crd-example.yaml`

```yml
apiVersion: apiextensions.k8s.io/v1beta1
kind: CustomResourceDefinition
metadata:
  name: alices.iamslash.com  # 1. CRD의 이름
spec:
  group: iamslash.com      # 2. 커스텀 리소스의 API 그룹
  version: v1alpha1     #    커스텀 리소스의 API 버전
  scope: Namespaced   #    커스텀 리소스가 네임스페이스에 속하는지 여부
  names:
    plural: alices        # 3. 커스텀 리소스의 이름 (복수형)
    singular: alice      #    커스텀 리소스의 이름 (단수형)
    kind: Alice         #    YAML 파일 등에서 사용될 커스텀 리소스의 Kind
    shortNames: ["ac"] #    커스텀 리소스 이름의 줄임말
  validation:
    openAPIV3Schema:    # 4. 커스텀 리소스의 데이터를 정의
      required: ["spec"]   #   커스텀 리소스에는 반드시 "spec" 이 존재해야 함.
      properties:         #    커스텀 리소스에 저장될 데이터 형식을 정의
        spec:
          required: ["myvalue"]
          properties:
            myvalue:
              type: "string"
              minimum: 1
```

```bash
$ kubectl apply -f my-crd-example.yaml
$ kubectl get crds
```

이제 custom resource 를 적용해 보자.

* `my-cr-example.yaml`

```yml
apiVersion: iamslash.com/v1alpha1
kind: Alice
metadata:
  name: my-custom-resource
spec:
  myvalue: "This is my value"
```

```bash
$ kubectl apply -f my-cr-example.yaml
$ kubectl get alices
$ kubectl get ac
$ kubectl describe ac my-custom-resource
```

CustomResourceDefinition 을 정의하고 CustomResource 를 생성했다면 그것을
Reconcile 할 수 있는 Custom Controller 가 필요하다.

Reconcile 을 위해 CustomResourceDefinition 을 사용하고 Controller 를 구현하는
방법을 Operator 패턴이라고 한다. [operatorhub.io](https://operatorhub.io/) 은
다양한 Operator (CRD,Controller) 를 제공하고 있다.

[OperatorSDK](https://github.com/operator-framework/operator-sdk), [KubeBuilder](https://github.com/kubernetes-sigs/kubebuilder) 를 이용하여 Operator 를 구현할 수도 있다. [178. [Kubernetes] 2편: Operator SDK 기본 예제 및 활용 : Hadoop CRD를 통한 Hadoop 클러스터 구축하기 @ naverblog](https://blog.naver.com/alice_k106/221586279079)

또한 [code-generator @ github](https://github.com/kubernetes/code-generator) 를 이용하면 go code 를 생성하여 Custom Controller 를 제작할 수 있다. 

* [Kubernetes Deep Dive: Code Generation for CustomResources](https://www.openshift.com/blog/kubernetes-deep-dive-code-generation-customresources) 
* [Kubernetes Controller 구현해보기](https://getoutsidedoor.com/2020/05/09/kubernetes-controller-%EA%B5%AC%ED%98%84%ED%95%B4%EB%B3%B4%EA%B8%B0/)
* [Programming Kubernetes CRDs](https://insujang.github.io/2020-02-13/programming-kubernetes-crd/)

# Custom Controller

> * [178. [Kubernetes] 2편: Operator SDK 기본 예제 및 활용 : Hadoop CRD를 통한 Hadoop 클러스터 구축하기 @ naverblog](https://blog.naver.com/alice_k106/221586279079)
> * [A deep dive into Kubernetes controllers](https://engineering.bitnami.com/articles/a-deep-dive-into-kubernetes-controllers.html)
> * [sample-controller @ github](https://github.com/kubernetes/sample-controller)
>   * pretty simple custom controller
> * [kubewatch @ github](https://github.com/bitnami-labs/kubewatch)
>   * controller which sends slack messages

![](img/client-go-controller-interaction.jpeg)

* kubernetes 의 controller 는 특정한 kubernetes resource 를 지켜보고 있다가 desired state 를 적용한다.
  ```go
  for {
    desired := getDesiredState()
    current := getCurrentState()
    makeChanges(desired, current)
  }
  ```
* controller 의 주요 컴포넌트로 Informer/SharedInformer 와 Workqueue 가 있다.
* Informer/SharedInformer 는 desired state 를 발견하면 Workqueue 에 아이템을 하나 삽입한다.
* kube-controller-manager 는 많은 수의 controller 들을 포함한다. 각 controller 는 자신들이 담당하는 특정 resource 에 대해서만 polling 하고 caching 한다. 이 cache 는 controller 들에 의해 공유된다. SharedInformer 는 이와 같이 공유된 cache 를 사용한다. 따라서 SharedInformer 를 Informer 보다 더 많이 사용한다. 
* Worker thread 는 Workqueue 에서 아이템을 하나 꺼내어 처리한다.

# Custom Scheduler

pod 의 schedulerName 을 확인하면 어떤 scheduler 를 사용하고 있는지 알 수 있다.
`default-scheduler` 는 `kube-scheduler` 를 의미한다.

```bash
$ kubectl get pod <pod-name> -o yaml | grep scheduler
schedulerName: default-scheduler
```

다음은 `default-schduler` 대신 custom scheduler 를 사용한 예이다.

* `custom-scheduled-pod.yaml`

```yml
apiVersion: v1
kind: Pod
metadata:
  name: custom-scheduled-pod
spec:
  schedulerName: my-custom-scheduler
  containers:
  - name: nginx-container
    image: nginx
```

custom scheduler 는 다음과 같은 것들을 차례로 구현해야 한다.

* kube-apiserver 를 통해서 새롭게 생성된 pod data 를 가져온다.
* pod data 중 nodeName 이 설정되어 있지 않으면 schedulerName 이 custom scheduler
  와 일치하는지 검사한다.
* node filtering, node scoring 등을 수행하고 kube-apiserver 를 통해서 pod data
  의 nodeName 을 worker-node 의 이름으로 채운다.

다음은 customer scheduler 의 예이다.

* [custom-scheduler-python @ github](https://github.com/alicek106/start-docker-kubernetes/blob/master/chapter11-2/custom-scheduler-python/__main__.py)
  * python 으로 제작한 예
* [Advanced Scheduling in Kubernetes @ kubernetes.io](https://kubernetes.io/blog/2017/03/advanced-scheduling-in-kubernetes/)
  * bash 로 제작한 예

다음은 custom scheduler 의 개발 방법이다.

* kube-scheduler 의 code 를 수정해서 build 할 수도 있다.
* [Scheduling Framework @ kubernetes.io](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/) 를 사용해서 개발할 수도 있다.
* [community/contributors/design-proposals/scheduling/scheduler_extender.md @ github](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/scheduling/scheduler_extender.md) 를 이용해서
  kube-scheduler 에 logic 을 추가할 수도 있다.

# Custom Metric Server

> [Custom Metric Server @ TIL](kubernetes_custom_metric_server.md)

