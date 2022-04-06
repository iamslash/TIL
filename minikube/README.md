# Abstract

minikube is local Kubernetes, focusing on making it easy to learn and develop for Kubernetes.

# Materials

* [minikube start](https://minikube.sigs.k8s.io/docs/start/)
* [minikube 설치 및 활용 @ medium](https://medium.com/@cratios48/minikube-%EC%84%A4%EC%B9%98-%EB%B0%8F-%ED%99%9C%EC%9A%A9-4a63ddbc7fcb)

# Basic

## Commands

```bash
# Start your cluster
$ minikube start

# Pause Kubernetes without impacting deployed applications:
$ minikube pause
$ minikube unpause

# Halt the cluster
$ minikube stop

# Increase the default memory limit (requires a restart)
$ minikube config set memory 16384

# Browse the catalog of easily installed Kubernetes services:
$ minikube addons list

# Create a second cluster running an older Kubernetes release
$ minikube start -p aged --kubernetes-version=v1.16.1

# Delete all of the minikube clusters
$ minikube delete --all
```
