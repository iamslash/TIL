- [Abstract](#abstract)
- [Materials](#materials)
- [Install](#install)
  - [Install with script](#install-with-script)
- [Tutorial](#tutorial)
- [Basic](#basic)
  - [Basic Commands](#basic-commands)
  - [Debugging Templates](#debugging-templates)
  - [Basic Templates](#basic-templates)
- [Advanced](#advanced)
  - [Render manifests from helm chart](#render-manifests-from-helm-chart)

---

# Abstract

The package manager for kubernetes

# Materials

* [Getting Started @ hlm.sh](https://helm.sh/docs/chart_template_guide/getting_started/)
  * `helm create myhelm` 을 실행후 문서를 따라서 실습해 보자.
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

# Tutorial

* [Getting Started @ hlm.sh](https://helm.sh/docs/chart_template_guide/getting_started/)
  * `helm create myhelm` 을 실행후 문서를 따라서 실습해 보자.

myhelm directory 를 생성하자. 

```bash
$ helm create myhelm
```

대부분의 파일은 지우고 다음 파일만 남기자.

```bash
$ tree myhelm
myhelm
├── Chart.yaml
├── charts
├── templates
└── values.yaml
```

그리고 `myhelm/templates/configmap.yaml` 을 다음과 같이 생성한다.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mychart-configmap
data:
  myvalue: "Hello World"
```

```bash
$ tree myhelm
myhelm
├── Chart.yaml
├── charts
├── templates
│   └── configmap.yaml
└── values.yaml
```

이제 myhelm 을 install 해보자.

```bash
$ helm install mypkg myhelm
NAME: mypkg
LAST DEPLOYED: Mon Aug  9 09:41:26 2021
NAMESPACE: default
STATUS: deployed
REVISION: 1
TEST SUITE: None

$ helm list
NAME 	NAMESPACE	REVISION	UPDATED                             	STATUS  	CHART       	APP VERSION
mypkg	default  	1       	2021-08-09 09:41:26.028545 +0900 KST	deployed	myhelm-0.1.0	1.16.0
```

설치가 된 상태에서 manifest 를 얻어올 수도 있다.

```bash
$ helm get manifest mypkg
---
# Source: myhelm/templates/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mychart-configmap
data:
  myvalue: "Hello World"
```

이제 mypkg 을 uninstall 해보자.

```bash
$ helm uninstall mypkg
release "mypkg" uninstalled
```

또한 다음과 같이 dry-run install 할 수 있다.

```bash
$ helm install --dry-run mypkg myhelm
NAME: mypkg
LAST DEPLOYED: Mon Aug  9 09:58:35 2021
NAMESPACE: default
STATUS: pending-install
REVISION: 1
TEST SUITE: None
HOOKS:
MANIFEST:
---
# Source: myhelm/templates/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mychart-configmap
data:
  myvalue: "Hello World"
```

또한 다음과 같이 debug 과 함께 dry-run install 할 수 있다. 

```bash
$ helm install --dry-run --debug mypkg myhelm
install.go:173: [debug] Original chart version: ""
install.go:190: [debug] CHART PATH: /Users/david.s/my/helm/myhelm

NAME: mypkg
LAST DEPLOYED: Mon Aug  9 10:00:01 2021
NAMESPACE: default
STATUS: pending-install
REVISION: 1
TEST SUITE: None
USER-SUPPLIED VALUES:
{}

COMPUTED VALUES:
{}

HOOKS:
MANIFEST:
---
# Source: myhelm/templates/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mychart-configmap
data:
  myvalue: "Hello World"
```

# Basic

## Basic Commands

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

## Debugging Templates

* [Debugging Templates @ helm.sh](https://helm.sh/docs/chart_template_guide/debugging/)

```bash
# Create and init myhelm directory
$ helm create myhelm

# Lint myhelm directory
$ helm lint myhelm

# Install with dry-run
$ helm install --dry-run --generate-name myhelm

# Get template
$ helm template myhelm
```

## Basic Templates

* `{{- `
  * [Controlling Whitespace](https://helm.sh/docs/chart_template_guide/control_structures/#controlling-whitespace)
  * 왼쪽의 whitespaces 를 제거한다. newline 도 포함이다.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ .Release.Name }}-configmap
data:
  myvalue: "Hello World"
  drink: {{ .Values.favorite.drink | default "tea" | quote }}
  food: {{ .Values.favorite.food | upper | quote }}
  {{- if eq .Values.favorite.drink "coffee" }}
  mug: "true"
  {{- end }}
```

위의 yaml 은 templating 후 아래와 같이 변환된다.

```yaml
# Source: mychart/templates/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: clunky-cat-configmap
data:
  myvalue: "Hello World"
  drink: "coffee"
  food: "PIZZA"
  mug: "true"
```

* ` -}}`
  * [Controlling Whitespace](https://helm.sh/docs/chart_template_guide/control_structures/#controlling-whitespace)
  * 오른쪽의 whitespaces 를 제거한다. newline 도 포함이다.

# Advanced

## Render manifests from helm chart

* [schelm @ github](https://github.com/databus23/schelm)

```bash
$ go get -u github.com/databus23/schelm

$ helm install --dry-run mypkg myhelm | schelm myhelm_output

$ tree myhelm_output
myhelm_output
└── myhelm
    └── templates
        └── configmap.yaml
```
