- [Materials](#materials)
- [Concept](#concept)
- [Dynamic Admission Control](#dynamic-admission-control)
- [Custom Controller](#custom-controller)
- [Custom Scheduler](#custom-scheduler)

-----

# Materials

* [Kubernetes Documentation / Concepts / Extending Kubernetes @ kubernetes.io](https://kubernetes.io/docs/concepts/extend-kubernetes/)

# Concept

![](https://docs.google.com/drawings/d/e/2PACX-1vQBRWyXLVUlQPlp7BvxvV9S1mxyXSM6rAc_cbLANvKlu6kCCf-kGTporTMIeG5GZtUdxXz1xowN7RmL/pub?w=960&h=720)

# Dynamic Admission Control

> * [172. [Kubernetes] Admission Controller의 이해 및 Python으로 Mutate Webhook 작성 예제 @ naverblog](https://blog.naver.com/alice_k106/221546328906)

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

> * [start-docker-kubernetes/chapter11-2/custom-scheduler-python @ github](https://github.com/alicek106/start-docker-kubernetes/blob/master/chapter11-2/custom-scheduler-python/__main__.py)
> * [Advanced Scheduling in Kubernetes @ kubernetes.io](https://kubernetes.io/blog/2017/03/advanced-scheduling-in-kubernetes/)
> * [Scheduling Framework @ kubernetes.io](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/)
> * [community/contributors/design-proposals/scheduling/scheduler_extender.md @ github](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/scheduling/scheduler_extender.md)
