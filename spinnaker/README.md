- [Abstract](#abstract)
- [Install with minikube, helm v3](#install-with-minikube-helm-v3)

----

# Abstract

Spinnaker 에 대해 정리한다.

# Install with minikube, helm v3

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
