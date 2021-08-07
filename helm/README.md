# Abstract

The package manager for kubernetes

# Materials

* [helm.sh](https://helm.sh/)
* [helm @ eksworkshop](https://www.eksworkshop.com/beginner/060_helm/)

# Install

## Install with script

```bash
$ curl -sSL https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 | bash

$ helm version --short

# Add repo
$ helm repo add stable https://charts.helm.sh/stable

# Search charts from repo
$ helm search repo stable

# Regiter autocomplete
$ helm completion bash >> ~/.bash_completion
. /etc/profile.d/bash_completion.sh
. ~/.bash_completion
source <(helm completion bash)
```

# Basic

# Basic Commands

```bash
# first, add the default repository, then update
$ helm repo add stable https://charts.helm.sh/stable
$ helm repo update

$ helm search repo

$ helm search repo nginx

# Add another repo
$ helm repo add bitnami https://charts.bitnami.com/bitnami

# Search bitnami from repo
$ helm search repo bitnami
$ helm search repo bitnami/nginx

# Install bitnami/nginx
$ helm install mywebserver bitnami/nginx

$ kubectl get svc,po,deploy
$ kubectl describe deployment mywebserver
$ kubectl get pods -l app.kubernetes.io/name=nginx
$ kubectl get service mywebserver-nginx -o wide
# Open browser with EXTERNAL-IP

# List helm charts
$ helm list

# Uninstall 
$ helm uninstall mywebserver

# No longer available
$ kubectl get pods -l app.kubernetes.io/name=nginx
$ kubectl get service mywebserver-nginx -o wide
```

# Basic helm

```bash
```
