- [Basic](#basic)
  - [Launch Ingress](#launch-ingress)
    - [Simple Ingress](#simple-ingress)
    - [Ingress with Annotation](#ingress-with-annotation)
    - [Ingress with SSL/TLS](#ingress-with-ssltls)
    - [Ingress with many Ingress Controllers](#ingress-with-many-ingress-controllers)
  - [Launch Persistent Volume, Persistent Claim](#launch-persistent-volume-persistent-claim)
  - [Launch ServiceAccount, RBAC](#launch-serviceaccount-rbac)
  - [Launch Horizontal Pod Autoscaler](#launch-horizontal-pod-autoscaler)
    - [Launch Simple Horizontal Pod Autoscaler](#launch-simple-horizontal-pod-autoscaler)
  - [Launch Kubernetes Dashboard](#launch-kubernetes-dashboard)
- [Dive Deep](#dive-deep)
  - [controller](#controller)
  - [API server](#api-server)
  - [Monitoring](#monitoring)

------

# Basic

## Launch Ingress

* [workshop-k8s-basic/guide/guide-03/bonus.md](https://github.com/subicura/workshop-k8s-basic/blob/master/guide/guide-05/bonus.md)
  * [[토크ON세미나] 쿠버네티스 살펴보기 7강 - Kubernetes(쿠버네티스) 실습 2 | T아카데미](https://www.youtube.com/watch?v=v6TUgqfV3Fo&list=PLinIyjMcdO2SRxI4VmoU6gwUZr1XGMCyB&index=7)

----

Ingress maps DNS to Services and include settings such as TLS.
You need Ingress Controller to use Ingress such as Nginx Web Server Ingress Controller.

### Simple Ingress 

* `ingress-example.yaml`
  * Service with ClusterIp type???

```yml
apiVersion: networking.k8s.io/v1beta1
kind: Ingress
metadata:
  name: ingress-example
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    kubernetes.io/ingress.class: "nginx"
spec:
  rules:
  - host: alicek106.example.com                   # [1]
    http:
      paths:
      - path: /echo-hostname                     # [2]
        backend:
          serviceName: hostname-service          # [3]
          servicePort: 80
```

```bash
$ kubectl appy -f ingress-example.yaml
$ kubectl get ingress
```

* Install ingress controller

```bash
$ kubectl appy -f \
  https://raw.githubusercontent.com/kubernetes/ingress-nginx/mast/deploy/static/mandatory.yaml

$ kubectl get pods,deployment -n ingress-nginx  
```

* `ingress-service-lb.yaml` 
  * Service object with LoadBalancer type

```yml
kind: Service
apiVersion: v1
metadata:
  name: ingress-nginx
  namespace: ingress-nginx
spec:
  type: LoadBalancer
  selector:
    app.kubernetes.io/name: ingress-nginx
    app.kubernetes.io/part-of: ingress-nginx
  ports:
    - name: http
      port: 80
      targetPort: http
    - name: https
      port: 443
      targetPort: https
```

```bash
$ kubectl apply -f ingress-service-lb.yaml
# Show including EXTERNAL-IP.
$ kubectl get svc -n ingress-nginx
$ kubectl apply -f hostname-deployment.yaml
$ kubectl apply -f hostname-service.yaml
$ kubectl get pods,services

# You can set temporal DNS name with --resolve.
$ curl --resolve alicek106.example.com:80:<LoadBalancer-IP> alicek106.example.comecho-hostname
$ curl --resolve alicek106.example.com:80:1.2.3.4 alicek106.example.comecho-hostname

# You can use AWS private DNS name with spec.host.
$ kubectl edit ingress ingress-example
...
spec:
  rules:
  - host: a206556.ap-northeast-2.elb.amazonaws.com
...
$ curl a206556.ap-northeast-2.elb.amazonaws.com/echo-hostname
```

* `minimal-ingress.yaml`

```yml
apiVersion: networking.k8s.io/v1beta1
kind: Ingress
metadata:
  name: minimal-ingress
spec:
  backend:
    serviceName: hostname-service
    servicePort: 80
```

### Ingress with Annotation

There are useful annotations such as `kubernetes.io/ingress.class`, `nginx.ingress.kubernetes.io/rewrite-target`.

* `ingress-example.yaml`
  * `metadata.annotations.kubernetes.io/ingress.class` means nginx Ingress Controller.

```yml
apiVersion: networking.k8s.io/v1beta1
kind: Ingress
metadata:
  name: ingress-example
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    kubernetes.io/ingress.class: "nginx"
spec:
  rules:
  - host: alicek106.example.com                   # [1]
    http:
      paths:
      - path: /echo-hostname                     # [2]
        backend:
          serviceName: hostname-service          # [3]
          servicePort: 80
```

* `ingress-rewrite-target.yaml`
  * `/echo-hostname/color/red` -> `/color/red`
  * `/echo-hostname/color` -> `/color`
  * `/echo-hostname` -> `/`

```yaml
apiVersion: networking.k8s.io/v1beta1
kind: Ingress
metadata:
  name: ingress-example
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /$2 # path의 (.*) 에서 획득한 경로로 전달합니다.
    kubernetes.io/ingress.class: "nginx"
spec:
  rules:
  - host: <여러분이 Nginx 컨트롤러에 접근하기 위한 도메인 이름을 입력합니다>
  #- host: a2cbfefcfbcfd48f8b4c15039fbb6d0a-1976179327.ap-northeast-2.elb.amazonaws.com
    http:
      paths:
      - path: /echo-hostname(/|$)(.*)          # (.*) 을 통해 경로를 얻습니다.
        backend:
          serviceName: hostname-service
          servicePort: 80
```

```bash
$ kubectl apply -f ingress-rewrite-target.yaml
```

### Ingress with SSL/TLS

```bash
# Create public, private keys
$ openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key -out tls.crt -subj "/CN/=iamslash.example.com/O=iamslash"

$ ls
tls.crt tls.key
$ kubectl create secret tls tls-secret --key tls.key --cert tls.crt  
```

* `ingress-tls.yaml`

```yml
apiVersion: networking.k8s.io/v1beta1
kind: Ingress
metadata:
  name: ingress-example
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    kubernetes.io/ingress.class: "nginx"
spec:
  tls:
  - hosts:
    - iamslash.example.com            
    secretName: tls-secret
  rules:
  - host: iamslash.example.com         
    http:
      paths:
      - path: /echo-hostname
        backend:
          serviceName: hostname-service
          servicePort: 80
```

```bash
$ kubectl apply -f ingress-tls.yaml
$ curl https://iamslash.example.com/echo-hostname -k
$ curl http://iamslash.example.com/echo-hostname -k
```

### Ingress with many Ingress Controllers

```bash
$ wget https://raw.githubusercontent.com/kubernetes/ingress-nginx/mast/deploy/static/mandatory.yaml

$ vim mandatory.yaml
...
            --ingress-class=alicek106-nginx

$ kubectl apply -f mandatory.yaml            
```

* `ingress-custom-class.yaml`

```yml
apiVersion: networking.k8s.io/v1beta1
kind: Ingress
metadata:
  name: ingress-example
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    kubernetes.io/ingress.class: "alicek106-nginx"
spec:
  rules:
  - host: alicek106.example.com                   # [1]
    http:
      paths:
      - path: /echo-hostname                     # [2]
        backend:
          serviceName: hostname-service          # [3]
          servicePort: 80
```

```bash
$ kubectl delete -f ./
$ kubectl delete -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/mast/deploy/static/mandatory.yaml
```

## Launch Persistent Volume, Persistent Claim

## Launch ServiceAccount, RBAC

## Launch Horizontal Pod Autoscaler

* [workshop-k8s-basic/guide/guide-03/task-06.md](https://github.com/subicura/workshop-k8s-basic/blob/master/guide/guide-05/task-06.md)
  * [[토크ON세미나] 쿠버네티스 살펴보기 7강 - Kubernetes(쿠버네티스) 실습 2 | T아카데미](https://www.youtube.com/watch?v=v6TUgqfV3Fo&list=PLinIyjMcdO2SRxI4VmoU6gwUZr1XGMCyB&index=7)

----

### Launch Simple Horizontal Pod Autoscaler

* hpa-example-deploy.yml.yml

```yml
apiVersion: apps/v1beta2
kind: Deployment
metadata:
  name: hpa-example-deploy
spec:
  selector:
    matchLabels:
      type: app
      service: hpa-example
  template:
    metadata:
      labels:
        type: app
        service: hpa-example
    spec:
      containers:
      - name: hpa-example
        image: k8s.gcr.io/hpa-example
        resources:
            limits:
              cpu: "0.5"
            requests:
              cpu: "0.25"
---

apiVersion: v1
kind: Service
metadata:
  name: hpa-example
spec:
  ports:
  - port: 80
    protocol: TCP
  selector:
    type: app
    service: hpa-example
```

* hpa.yml

```yml
apiVersion: autoscaling/v1
kind: HorizontalPodAutoscaler
metadata:
  name: hpa-example
spec:
  maxReplicas: 4
  minReplicas: 1
  scaleTargetRef:
    apiVersion: extensions/v1
    kind: Deployment
    name: hpa-example-deploy
  targetCPUUtilizationPercentage: 10
```

## Launch Kubernetes Dashboard

```bash
# show k8s client, server version
$ kubectl version --output yaml
# show contexts, default context is docker-for-desktop
$ kubectl config get-contexts
# show nodes
$ kubectl get nodes
# show pods
$ kubectl get pods --all-namespaces
# launch k8s dashbaord
$ kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v1.10.1/src/deploy/recommended/kubernetes-dashboard.yaml
# show services
$ kubectl get services --all-namespaces
# launch proxy server to connect k8s dashboard
$ kubectl proxy
# open http://localhost:8001/api/v1/namespaces/kube-system/services/https:kubernetes-dashboard:/proxy/
# create sample user and login. 
# - https://github.com/kubernetes/dashboard/blob/master/docs/user/access-control/creating-sample-user.md
```

* `https://raw.githubusercontent.com/kubernetes/dashboard/v1.10.1/src/deploy/recommended/kubernetes-dashboard.yaml`

```yaml
# Copyright 2017 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------------------- Dashboard Secret ------------------- #

apiVersion: v1
kind: Secret
metadata:
  labels:
    k8s-app: kubernetes-dashboard
  name: kubernetes-dashboard-certs
  namespace: kube-system
type: Opaque

---
# ------------------- Dashboard Service Account ------------------- #

apiVersion: v1
kind: ServiceAccount
metadata:
  labels:
    k8s-app: kubernetes-dashboard
  name: kubernetes-dashboard
  namespace: kube-system

---
# ------------------- Dashboard Role & Role Binding ------------------- #

kind: Role
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: kubernetes-dashboard-minimal
  namespace: kube-system
rules:
  # Allow Dashboard to create 'kubernetes-dashboard-key-holder' secret.
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["create"]
  # Allow Dashboard to create 'kubernetes-dashboard-settings' config map.
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["create"]
  # Allow Dashboard to get, update and delete Dashboard exclusive secrets.
- apiGroups: [""]
  resources: ["secrets"]
  resourceNames: ["kubernetes-dashboard-key-holder", "kubernetes-dashboard-certs"]
  verbs: ["get", "update", "delete"]
  # Allow Dashboard to get and update 'kubernetes-dashboard-settings' config map.
- apiGroups: [""]
  resources: ["configmaps"]
  resourceNames: ["kubernetes-dashboard-settings"]
  verbs: ["get", "update"]
  # Allow Dashboard to get metrics from heapster.
- apiGroups: [""]
  resources: ["services"]
  resourceNames: ["heapster"]
  verbs: ["proxy"]
- apiGroups: [""]
  resources: ["services/proxy"]
  resourceNames: ["heapster", "http:heapster:", "https:heapster:"]
  verbs: ["get"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: kubernetes-dashboard-minimal
  namespace: kube-system
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: kubernetes-dashboard-minimal
subjects:
- kind: ServiceAccount
  name: kubernetes-dashboard
  namespace: kube-system

---
# ------------------- Dashboard Deployment ------------------- #

kind: Deployment
apiVersion: apps/v1
metadata:
  labels:
    k8s-app: kubernetes-dashboard
  name: kubernetes-dashboard
  namespace: kube-system
spec:
  replicas: 1
  revisionHistoryLimit: 10
  selector:
    matchLabels:
      k8s-app: kubernetes-dashboard
  template:
    metadata:
      labels:
        k8s-app: kubernetes-dashboard
    spec:
      containers:
      - name: kubernetes-dashboard
        image: k8s.gcr.io/kubernetes-dashboard-amd64:v1.10.1
        ports:
        - containerPort: 8443
          protocol: TCP
        args:
          - --auto-generate-certificates
          # Uncomment the following line to manually specify Kubernetes API server Host
          # If not specified, Dashboard will attempt to auto discover the API server and connect
          # to it. Uncomment only if the default does not work.
          # - --apiserver-host=http://my-address:port
        volumeMounts:
        - name: kubernetes-dashboard-certs
          mountPath: /certs
          # Create on-disk volume to store exec logs
        - mountPath: /tmp
          name: tmp-volume
        livenessProbe:
          httpGet:
            scheme: HTTPS
            path: /
            port: 8443
          initialDelaySeconds: 30
          timeoutSeconds: 30
      volumes:
      - name: kubernetes-dashboard-certs
        secret:
          secretName: kubernetes-dashboard-certs
      - name: tmp-volume
        emptyDir: {}
      serviceAccountName: kubernetes-dashboard
      # Comment the following tolerations if Dashboard must not be deployed on master
      tolerations:
      - key: node-role.kubernetes.io/master
        effect: NoSchedule

---
# ------------------- Dashboard Service ------------------- #

kind: Service
apiVersion: v1
metadata:
  labels:
    k8s-app: kubernetes-dashboard
  name: kubernetes-dashboard
  namespace: kube-system
spec:
  ports:
    - port: 443
      targetPort: 8443
  selector:
    k8s-app: kubernetes-dashboard
```

# Dive Deep

## controller

* [A deep dive into Kubernetes controllers](https://engineering.bitnami.com/articles/a-deep-dive-into-kubernetes-controllers.html)
  * [sample-controller @ github](https://github.com/kubernetes/sample-controller)
    * pretty simple custom controller
  * [kubewatch @ github](https://github.com/bitnami-labs/kubewatch)
    * controller which sends slack messages

----

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

## API server

* [Kubernetes API Reference](https://kubernetes.io/docs/reference/)

## Monitoring

* [Prometheus with Kubernetes](https://www.slideshare.net/jinsumoon33/kubernetes-prometheus-monitoring)

