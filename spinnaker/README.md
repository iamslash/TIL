- [Abstract](#abstract)
- [Basic](#basic)
  - [Install with minikube, helm v3](#install-with-minikube-helm-v3)
  - [Architecture](#architecture)
  - [Spinnaker Pipeline Stages](#spinnaker-pipeline-stages)

----

# Abstract

Spinnaker 에 대해 정리한다.

# Basic

## Install with minikube, helm v3

```bash
$ minikube delete

$ minikube start --memory 5120 --cpus=4

$ helm repo add spinnaker https://helmcharts.opsmx.com/

# This will take very long time
$ helm install my-spinnaker spinnaker/spinnaker --timeout 600s

$ kubectl get pods -A

$ helm get notes my-spinnaker
NOTES:
1. You will need to create 2 port forwarding tunnels in order to access the Spinnaker UI:
  export DECK_POD=$(kubectl get pods --namespace default -l "cluster=spin-deck" -o jsonpath="{.items[0].metadata.name}")
  kubectl port-forward --namespace default $DECK_POD 9000

  export GATE_POD=$(kubectl get pods --namespace default -l "cluster=spin-gate" -o jsonpath="{.items[0].metadata.name}")
  kubectl port-forward --namespace default $GATE_POD 8084

2. Visit the Spinnaker UI by opening your browser to: http://127.0.0.1:9000

To customize your Spinnaker installation. Create a shell in your Halyard pod:

  kubectl exec --namespace default -it my-spinnaker-spinnaker-halyard-0 bash

For more info on using Halyard to customize your installation, visit:
  https://www.spinnaker.io/reference/halyard/

For more info on the Kubernetes integration for Spinnaker, visit:
  https://www.spinnaker.io/reference/providers/kubernetes-v2/

```

## Architecture 

> * [Spinnaker Achitecture](https://spinnaker.io/docs/reference/architecture/)
>   * [Spinnaker Architecture Overview](https://spinnaker.io/docs/reference/architecture/microservices-overview/)

Spinnaker 는 여러개의 마이크로서비스들로 이루어져있다.

* **Deck** is the browser-based UI.
* **Gate** is the API gateway.
* **Orca** is the orchestration engine. It handles all ad-hoc operations and pipelines. Read more on the Orca Service Overview .
* **Clouddriver** is responsible for all mutating calls to the cloud providers and for indexing/caching all deployed resources.
* **Front50** is used to persist the metadata of applications, pipelines, projects and notifications.
* **Rosco** is the bakery. It produces immutable VM images (or image templates) for various cloud providers.
* **Igor** is used to trigger pipelines via continuous integration jobs in systems like Jenkins and Travis CI, and it allows Jenkins/Travis stages to be used in pipelines.
* **Echo** is Spinnaker’s eventing bus.
* **Fiat** is Spinnaker’s authorization service.
* **Kayenta** provides automated canary analysis for Spinnaker.
* **Keel** powers Managed Delivery .
* **Halyard** is Spinnaker’s configuration service.

## Spinnaker Pipeline Stages

> * [Spinnaker Pipeline Stages](https://spinnaker.io/docs/reference/pipeline/stages/)
> * [Pipeline Stage Plugin Walkthrough](https://spinnaker.io/docs/community/contributing/code/developer-guides/plugin-creators/stage-plugin-walkthrough/)

Spinnaker Pipeline 에서 지원하는 Stage 의 종류는 정해져 있다. 주요 Stage 는 Bake, Deploy, Jenkins 가 있다.

* Bake
  * Helm3 templating 을 할 수 있다.
  * remote template, values 파일들을 지정하면 Kubernetes Manifest Files 들을 templating 할 수 있다.
* Deploy
  * Bake stage 를 통해 만들어진 Kubernetes Manifest Files 들을 Kubernetes Cluster 에 apply 한다.
* Jenkins
  * Jenkins 의 특정 Job 을 실행할 수 있다.

code 는 어디있는 걸까? [pf4jStagePlugin](https://github.com/spinnaker-plugin-examples/pf4jStagePlugin) 는 예제로 보인다.
