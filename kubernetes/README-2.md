- [Materials](#materials)
- [Basic](#basic)
  - [Launch Ingress](#launch-ingress)
    - [Simple Ingress](#simple-ingress)
    - [Ingress with Annotation](#ingress-with-annotation)
    - [Ingress with SSL/TLS](#ingress-with-ssltls)
    - [Ingress with many Ingress Controllers](#ingress-with-many-ingress-controllers)
  - [Launch Persistent Volume, Persistent Claim](#launch-persistent-volume-persistent-claim)
    - [Local Volume : hostPath, emptyDir](#local-volume--hostpath-emptydir)
    - [Network Volume](#network-volume)
    - [Volume management with PV, PVC](#volume-management-with-pv-pvc)
  - [Authorization](#authorization)
  - [RBAC(Role-based access control)](#rbacrole-based-access-control)
    - [Role](#role)
    - [ClusterRole](#clusterrole)
    - [RoleBinding](#rolebinding)
    - [ClusterRoleBinding](#clusterrolebinding)
    - [Setting Secrets on Service Account for Image Registry](#setting-secrets-on-service-account-for-image-registry)
    - [Setting Secrets on kubeconfig](#setting-secrets-on-kubeconfig)
    - [User, Group](#user-group)
    - [User authentication with X509](#user-authentication-with-x509)
  - [Resource Limit of Pods](#resource-limit-of-pods)
  - [Major Kubernetes Objects](#major-kubernetes-objects)
    - [Limit](#limit)
    - [Request](#request)
    - [CPU Limit](#cpu-limit)
    - [QoS class, Memory Limit](#qos-class-memory-limit)
    - [ResourceQuota](#resourcequota)
    - [LimitRange](#limitrange)
    - [Admission Controller](#admission-controller)
  - [Kubernetes Scheduling](#kubernetes-scheduling)
    - [nodeName, nodeSelector, nodeAffinity, podAffinity, podAntiAffinity](#nodename-nodeselector-nodeaffinity-podaffinity-podantiaffinity)
    - [Taints, Tolerations](#taints-tolerations)
    - [Cordon, Drain, PodDisruptionBudget](#cordon-drain-poddisruptionbudget)
    - [Custom Scheduler](#custom-scheduler)
    - [Static Pods vs DaemonSets](#static-pods-vs-daemonsets)
    - [configuring Scheduler](#configuring-scheduler)
  - [Kubernetes Application Status, Deployment](#kubernetes-application-status-deployment)
    - [Rolling update with Deployment](#rolling-update-with-deployment)
    - [BlueGreen update](#bluegreen-update)
    - [LifeCyle](#lifecyle)
    - [LivenessProbe, ReadinessProbe](#livenessprobe-readinessprobe)
    - [Terminating status](#terminating-status)
  - [Custom Resource Definition](#custom-resource-definition)
  - [Kubernetes Objects using Pod Objects](#kubernetes-objects-using-pod-objects)
    - [Jobs](#jobs)
    - [CronJobs](#cronjobs)
    - [DaemonSets](#daemonsets)
    - [StatefulSets](#statefulsets)
  - [Launch Horizontal Pod Autoscaler](#launch-horizontal-pod-autoscaler)
    - [Launch Simple Horizontal Pod Autoscaler](#launch-simple-horizontal-pod-autoscaler)
- [Advanced](#advanced)
  - [Launch Kubernetes Dashboard](#launch-kubernetes-dashboard)
  - [Process of Pod Termination](#process-of-pod-termination)
  - [Kubernetes Extension](#kubernetes-extension)
  - [API server](#api-server)
  - [Monitoring](#monitoring)
- [Continue...](#continue)

------

# Materials

* [시작하세요! 도커/쿠버네티스](http://www.yes24.com/Product/Goods/84927385)
  * [src](https://github.com/alicek106/start-docker-kubernetes)
  * 한글책 중 최고

# Basic

## Launch Ingress

* [workshop-k8s-basic/guide/guide-03/bonus.md](https://github.com/subicura/workshop-k8s-basic/blob/master/guide/guide-05/bonus.md)
  * [[토크ON세미나] 쿠버네티스 살펴보기 7강 - Kubernetes(쿠버네티스) 실습 2 | T아카데미](https://www.youtube.com/watch?v=v6TUgqfV3Fo&list=PLinIyjMcdO2SRxI4VmoU6gwUZr1XGMCyB&index=7)

----

Ingress maps DNS to Services, include settings such as TLS and rewrite HTTP URL.
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

PV (Persistent Volume) 은 POD 의 mount 대상이되는 physical volume 을 의미한다. PVC (Persistent Volume Claim) 은 POD 와 PV 를 연결해주는 역할을 한다. PVC 에 기록된 spec 에 맞는 PV 가 없다면 POD 는 physical volume 을 mount 할 수 없다.

Kubernetes supports NFS, AWS EBS, Ceph, GlusterFS as Network Persistent Volumes.

Storage Class 는 physical volume 의 dynamic provision 을 위해 필요하다. POD 와 함께 생성된 PVC 에 맞는 PV 가 없다면 Storage Class 에 미리 설정된 spec 대로 AWS EBS 를 하나 만들고 POD 는 그 physical volume 을 mount 할 수 있다.

POD 가 worker-node 의 path 에 mount 하고 싶다면 `hostPath` 를 사용한다. 이것은 pod 가 delete 되도라도 보전된다. 그러나 pod 가 delete 됬을 때 보존이 필요 없다면 `emptyDir` 을 이용하여 임시디렉토리를 생성한다. 또한 모든 container 들이 공유할 수 있다.

### Local Volume : hostPath, emptyDir

* `hostpath-pod.yaml`
  * worker node's `/tmp` mount to pod's `/etc/data`.
  * This is useful for specific pod to run on specific worker-node like CAdvisor.

```yml
apiVersion: v1
kind: Pod
metadata:
  name: hostpath-pod
spec:
  containers:
    - name: my-container
      image: busybox
      args: [ "tail", "-f", "/dev/null" ]
      volumeMounts:
      - name: my-hostpath-volume
        mountPath: /etc/data
  volumes:
    - name: my-hostpath-volume
      hostPath:
        path: /tmp
```

```bash
$ kubectl apply -f hostpath-pod.yaml
$ kubectl exec -it hostpath-pod touch /etc/data/mydata/
# After connecting to worker-node
$ ls /tmp/mydata
```

* `emptydir-pod.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: emptydir-pod
spec:
  containers:
  - name: content-creator
    image: alicek106/alpine-wget:latest
    args: ["tail", "-f", "/dev/null"]
    volumeMounts:
    - name: my-emptydir-volume
      mountPath: /data                      # 1. 이 컨테이너가 /data 에 파일을 생성하면

  - name: apache-webserver
    image: httpd:2
    volumeMounts:
    - name: my-emptydir-volume
      mountPath: /usr/local/apache2/htdocs/  # 2. 아파치 웹 서버에서 접근 가능합니다.

  volumes:
    - name: my-emptydir-volume
      emptyDir: {}                             # 포드 내에서 파일을 공유하는 emptyDir
```

* Launch

```bash
$ kubectl apply -f emptydir-pod.yaml
$ kubectl exec -it emptydir-pod -c content-creator sh
> echo Hello World >> /data/test.html
# exit
$ kubectl describe pod emptydir-pod | grep IP
$ kubectl run -it --rm debug --image=alicek106/ubuntu:curl --restart=Never -- curl 172.xx.0.8.test.html
```

### Network Volume

* `nfs-deployment.yaml`

```yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nfs-server
spec:
  selector:
    matchLabels:
      role: nfs-server
  template:
    metadata:
      labels:
        role: nfs-server
    spec:
      containers:
      - name: nfs-server
        image: gcr.io/google_containers/volume-nfs:0.8
        ports:
          - name: nfs
            containerPort: 2049
          - name: mountd
            containerPort: 20048
          - name: rpcbind
            containerPort: 111
        securityContext:
          privileged: true
```

* `nfs-service.yaml`

```yml
apiVersion: v1
kind: Service
metadata:
  name: nfs-service
spec:
  ports:
  - name: nfs
    port: 2049
  - name: mountd
    port: 20048
  - name: rpcbind
    port: 111
  selector:
    role: nfs-server
```

* Launch

```bash
$ kubectl apply -f nfs-deployment.yaml
$ kubectl apply -f nfs-service.yaml
```

* `nfs-pod.yaml`

```yml
apiVersion: v1
kind: Pod
metadata:
  name: nfs-pod
spec:
  containers:
    - name: nfs-mount-container
      image: busybox
      args: [ "tail", "-f", "/dev/null" ]
      volumeMounts:
      - name: nfs-volume
        mountPath: /mnt           # 포드 컨테이너 내부의 /mnt 디렉터리에 마운트합니다.
  volumes:
  - name : nfs-volume
    nfs:                            # NFS 서버의 볼륨을 포드의 컨테이너에 마운트합니다.
      path: /
      server: {NFS_SERVICE_IP}
```

* Launch

```bash
$ export NFS_CLUSTER_IP=$(kubectl get svc/nfs-service -o jsonpath='{.spec.clusterIP}')
$ cat nfs-pod.yaml | sed "s/{NFS_SERVICE_IP}/$NFS_CLUSTER_IP/g" | kubectl apply -f
$ kubectl get pod nfs-pod
$ kubectl exec -it nfs-pod sh
> df -h
```

### Volume management with PV, PVC

* Check legacy pv, pvc

```bash
$ kubectl get persistentvolume,persistentvolumeclaim
$ kubectl get pv,pvc
```

* Create AWs EBS volume

```bash
$ export BOLUME_ID=$(aws ec2 create-volume --size 5 \
  --region ap-northeast-2 \
  --availability-zone ap-northeast-2a \
  --volume-type gp2 \
  --tag-specifications \
  'ResourceType=volume,Tags=[{Key=KubernetesCluster,Value=mycluster.k8s.local}]' \
  | jq '.VolumeId' -r)
```

* `ebs-pv.yaml`

```yml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: ebs-pv
spec:
  capacity:
    storage: 5Gi         # 이 볼륨의 크기는 5G입니다.
  accessModes:
    - ReadWriteOnce    # 하나의 포드 (또는 인스턴스) 에 의해서만 마운트 될 수 있습니다.
  awsElasticBlockStore:
    fsType: ext4
    volumeID: <VOLUME_ID>
```

* Launch

```bash
$ cat ebs-pv.yaml | sed "s/<VOLUME_ID>/$VOLUME_ID/G" | kubectl apply -f -
$ kubectl get pv
```

* `ebs-pod-pvc.yaml`

```yml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-ebs-pvc                  # 1. my-ebs-pvc라는 이름의 pvc 를 생성합니다.
spec:
  storageClassName: ""
  accessModes:
    - ReadWriteOnce       # 2.1 속성이 ReadWriteOnce인 퍼시스턴트 볼륨과 연결합니다.
  resources:
    requests:
      storage: 5Gi          # 2.2 볼륨 크기가 최소 5Gi인 퍼시스턴트 볼륨과 연결합니다.
---
apiVersion: v1
kind: Pod
metadata:
  name: ebs-mount-container
spec:
  containers:
    - name: ebs-mount-container
      image: busybox
      args: [ "tail", "-f", "/dev/null" ]
      volumeMounts:
      - name: ebs-volume
        mountPath: /mnt
  volumes:
  - name : ebs-volume
    persistentVolumeClaim:
      claimName: my-ebs-pvc    # 3. my-ebs-pvc라는 이름의 pvc를 사용합니다.
```

* Launch

```bash
$ kubectl apply -f ebs-pod-pvc.yaml
$ kubectl get pv, pvc
$ kubectl get pods
$ kubectl exec ebs-mount-container -- df -h | grep /mnt
```

* ebs-pv-storageclass.yaml

```yml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: ebs-pv-custom-cs
spec:
  capacity:
    storage: 5Gi
  accessModes:
    - ReadWriteOnce
  storageClassName: my-ebs-volume
  awsElasticBlockStore:
    fsType: ext4
    volumeID: <VOLUME_ID> # 여러분의 EBS 볼륨 ID로 대신합니다.
    # volumeID: vol-0390f3a601e58ce9b
```

* ebs-pod-pvc-custom-sc.yaml

```yml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-ebs-pvc-custom-sc
spec:
  storageClassName: my-ebs-volume
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
---
apiVersion: v1
kind: Pod
metadata:
  name: ebs-mount-container-custom-sc
spec:
  containers:
    - name: ebs-mount-container
      image: busybox
      args: [ "tail", "-f", "/dev/null" ]
      volumeMounts:
      - name: ebs-volume
        mountPath: /mnt
  volumes:
  - name : ebs-volume
    persistentVolumeClaim:
      claimName: my-ebs-pvc-custom-sc
```

* ebs-pv-label.yaml

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: ebs-pv-label
  labels:
    region: ap-northeast-2a
spec:
  capacity:
    storage: 5Gi
  accessModes:
    - ReadWriteOnce
  awsElasticBlockStore:
    fsType: ext4
    # volumeID: vol-025c52fbd39d35417
    volumeID: <여러분의 VOLUME ID를 입력합니다> 
```

* ebs-pod-pvc-label-selector.yaml

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-ebs-pvc-selector
spec:
  selector:
    matchLabels:
      region: ap-northeast-2a
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
---
apiVersion: v1
kind: Pod
metadata:
  name: ebs-mount-container-label
spec:
  containers:
    - name: ebs-mount-container
      image: busybox
      args: [ "tail", "-f", "/dev/null" ]
      volumeMounts:
      - name: ebs-volume
        mountPath: /mnt
  volumes:
  - name : ebs-volume
    persistentVolumeClaim:
      claimName: my-ebs-pvc-selector
```

* `ebs-pv-delete.yaml`

```yml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: ebs-pv-delete
spec:
  capacity:
    storage: 5Gi
  accessModes:
    - ReadWriteOnce
  awsElasticBlockStore:
    fsType: ext4
    volumeID: <VOLUME_ID>
  persistentVolumeReclaimPolicy: Delete
```

```bash
$ cat ebs-pv-delete.yaml | sed "s/<VOLUME_ID>/$VOLUME_ID/g" | kubectl apply -f -
$ kubectl get pv
$ kubectl apply -f ebs-pod-pvc.yaml
$ kubectl get pods
$ kubectl get pv,pvc
$ kubectl delete -f ebs-pod-pvc.yaml
$ kubectl get pv,pvc
```

* `storageclass-slow.yaml`

```yml
kind: StorageClass
apiVersion: storage.k8s.io/v1
metadata:
  name: slow
provisioner: kubernetes.io/aws-ebs
parameters:
  type: st1
  fsType: ext4
  zones: ap-northeast-2a  # 여러분의 쿠버네티스 클러스터가 위치한 가용 영역을 입력합니다.
```

* `storageclass-fast.yaml`

```yml
kind: StorageClass
apiVersion: storage.k8s.io/v1
metadata:
  name: fast
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp2
  fsType: ext4
  zones: ap-northeast-2a # 여러분의 쿠버네티스 클러스터가 위치한 가용 영역을 입력합니다.
```

```bash
$ kubectl apply -f storageclass-slow.yaml
$ kubectl apply -f storageclass-fast.yaml
$ kubectl get sc
```

* `pvc-fast-sc.yaml`

```yml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-fast-sc
spec:
  storageClassName: fast
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

```bash
$ kubectl apply -f pvc-fast-sc.yaml
$ kubectl get pv,pvc
$ kubectl get sc fast -o yaml
```

* `storageclass-default.yaml`

```yml
kind: StorageClass
apiVersion: storage.k8s.io/v1
metadata:
  name: generic
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp2
  fsType: ext4
  zones: ap-northeast-2a # 여러분의 쿠버네티스 클러스터가 위치한 가용 영역을 입력합니다.
```

```bash
$ kubectl apply -f storageclass-default.yaml
$ kubectl get storageclass
```

## Authorization

* [쿠버네티스 권한관리(Authorization)](https://arisu1000.tistory.com/27848)

------

Kubernetes 는 ABAC(Attribute-based access control) 혹은 RBAC(Role-based access control) 를 이용하여 Authorization 을 수행할 수 있다. Master-Node 의 API 호출을 권한관리할 수 있다.

ABAC(Attribute-based access control) 는 사용자(user), 그룹(group), 요청 경로(request path), 요청 동사(request verb), namespace, 자원 등으로 권한을 설정한다. 설정은 파일로 관리한다. 설정이 변경될 때 마다 Master-Node 를 rebooting 해야 한다. 매우 불편하여 사용되지 않는다.

RBAC(Role-based access control) 는 사용자(user), 역할(role) 을 각각 선언하고 두가지를 묶어서(binding) 하여 사용자(user) 에게 권한을 부여해 준다. Master-Node 에 접근할 필요 없이 kubectl 혹은 API 로 권한 설정이 가능하다. 매우 유용하다.

## RBAC(Role-based access control)

* [INTRO TO RBAC @ eksworkshop](https://www.eksworkshop.com/beginner/090_rbac/)

----

RBAC 는 다음과 같은 논리적 요소들로 구성된다.

* **Entity**
  * Group, User, Service Account 와 같이 Kubernetes Resource 를 접근할 수 있는 권한의 주체이다.
* **Resource**
  * Pod, Service, Secret 과 같이 Entity 가 접근할 수 있는 Kubernetes 의 자원이다.
* **Role**
  * Action 의 Rule 들을 모아 놓은 것이다.
* **RoleBinding**
  * Role 과 Entity 의 묶음이다.
* **Namespace**
  * 하나의 Physical Cluster 를 Namespace 를 정하여 별도의 Virtual cluster 로 나누어 사용할 수 있다.

### Role

Role은 특정 API 혹은 리소스에 대한 권한들을 명시해둔 규칙들의 집합이다. 특정 namespace 만 적용된다.

* `read-role.yml`

```yml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: default
  name: read-role
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"]
```

다음은 verbs 의 종류이다.

| verbs | description|
|---|--|
| create | create resource |
| get | get resource |
| list | list resource |
| update | update resource |
| patch | patch resource  |
| delete | delete resource |
| deletecollection | delete resources |

### ClusterRole

ClusterRole은 역시 특정 API 혹은 리소스에 대한 권한들을 명시해둔 규칙들의 집합이다. 그러나 특정 namespace 가 아닌 전체 cluster 에 대해 적용된다.

* `read-clusterrole.yml`

```yml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: read-clusterrole
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"]
```

* `clusterrole-aggregation.yml`
  * 다른 ClusterRole 들로 부터 label 이 `kubernetes.io/bootstrapping: rbac-defaults` 와 match 되는 것들의 rule 들을 가져온다.

```yml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: admin-aggregation
aggregationRule:
  clusterRoleSelectors:
  - matchLabels:
      kubernetes.io/bootstrapping: rbac-defaults
rules: []
```

다음과 같이 url 에 대해 권한 설정을 할 수도 있다.

```yml
rules:
- nonResourceURLs: ["/healthcheck”, “/metrics/*"]
  verbs: [“get”, “post"]
```

### RoleBinding

Role 과 User 를 묶어주는 역할을 한다. 특정 namespace 만 적용된다.

* 다음과 같이 `serviceaccount-myuser.yaml` 를 선언하여 유저를 생성한다.

```yml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: myuser
  namespace: default
```

* `read-rolebinding.yml`

```yml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-rolebinding
  namespace: default
subjects:
- kind: ServiceAccount
  name: myuser
  apiGroup: ""
roleRef:
  kind: Role
  name: read-role
  apiGroup: rbac.authorization.k8s.io
```

### ClusterRoleBinding

Role 과 User 를 묶어주는 역할을 한다. 특정 namespace 가 아닌 전체 cluster 에 적용된다.

* `read-clusterrolebinding.yml`

```yml
kind: ClusterRoleBinding
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: read-clusterrolebinding
subjects:
- kind: ServiceAccount
  name: myuser
  namespace: default
  apiGroup: ""
roleRef:
  kind: ClusterRole
  name: read-clusterrole
  apiGroup: rbac.authorization.k8s.io
```

### Setting Secrets on Service Account for Image Registry

`imagePullSecrets` 를 이용하면 Service Account 가 private docker registry 에서
docker pull 할 때 Secret 을 사용할 수 있게 할 수 있다.

* `sa-reg-auth.yaml`

```yml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: reg-auth-iamslash
  namespace: default
imagePullSecrets:
- name: registry-auth
```

```bash
$ kubectl apply -f sa-reg-auth.yaml
$ kubectl describe sa reg-auth-iamslash | grep Image
```

### Setting Secrets on kubeconfig

kubeconfig file 은 일반 적으로 `~/.kube/config` 에 저장된다. `clusters, users`
가 각각 정의되어 있다. `context` 는 `clusters, users` 를 짝지어 grouping 한
것이다. `context` 를 바꿔가면서 다향한 `clusters, users` 설정을 이용할 수 있다.

* `~/.kube/config`

```yml
apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: LS0t...
    server: https://kubernetes.docker.internal:6443
  name: docker-desktop
contexts:
- context:
    cluster: docker-desktop
    user: docker-desktop
  name: docker-desktop
- context:
    cluster: docker-desktop
    user: docker-desktop
  name: docker-for-desktop
current-context: docker-desktop
kind: Config
preferences: {}
users:
- name: docker-desktop
  user:
    client-certificate-data: LS0t...
    client-key-data: LS0tL...
```

```bash
$ kubectl get secrets
$ export secret_name=iamslash-token-gfg41
$ export decoded_token=$(kubectl get secret $secret_name -o jsonpath='{.data.token}' | base64 -d)
$ kubectl config set-credentials iamslash-user --token=$decoded_token
$ kubectl config get-clusters
$ kubectl config set-context my-new-context --cluste=kubernetes --user=iamslash-user
$ kubectl config get-contexts
$ kubectl get deployment
$ kubectl get pods
$ kubectl get service
$ kubectl config use-context kubernetes-admin@kubernetes
```

### User, Group

Kubernetes 는 `ServiceAccount` 말고도 `User, Group` 에 권한을 부여할 수 있다.
예를 들어 `RoleBinding, ClusterRoleBinding` 에 다음과 같이 `ServiceAccount` 대신
`User, Group` 을 사용할 수 있다.

```yml
...
subjects:
- kind: User
  name: iamslash
--  
...
subjects:
- kind: Group
  name: devops-team
```

이번에는 `iamslash` 라는 user 를 이용하여 `ServiceAccount` 를 만들어 사용해
보자. `ServiceAccount` 형식은 `system:serviceaccount:<namespace>:<username>`
의 형태이다.

```bash
$ kubectl get services --as system:serviceaccount:default:iamslash
```

다음과 같이 `RoleBinding` 에도 `user` 를 사용한 `ServiceAccount` 를 이용할 수
있다.

```yml
...
subjects:
- kind: User
  name: system:serviceaccount:default:iamslash
  namespace: default
roleRef:
kind: Role
...  
```

다음은 `Group` 을 사용한 `RoleBinding` 의 예이다.

* `service-read-role-all-sa.yaml`

```yml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: service-reader-rolebinding
subjects:
- kind: Group
  name: system:serviceaccounts
roleRef:
  kind: ClusterRole  # 클러스터 롤 바인딩에서 연결할 권한은 클러스터 롤이여야 합니다.
  name: service-reader
  apiGroup: rbac.authorization.k8s.io
```

Kubernetes 에서 `~/.kube/config` 를 이용하여 인증하는 것을 `X.509` 인증이라고
한다. 별도의 `IDP` 를 이용하여 authentication 을 수행할 수 있다. [181. [Kubernetes] 쿠버네티스 인증 3편: Dex와 Github OAuth (OIDC) 를 이용한 사용자 인증하기 @ naverblog](https://blog.naver.com/alice_k106/221598325656)

### User authentication with X509

kubenetes 는 master node 의 `/etc/ca-certificates` 에 certificates 가 설치된다.

```bash
C:\Users\iamslash\.kube>kubectl -n kube-system describe pod kube-apiserver-docker-desktop
Name:               kube-apiserver-docker-desktop
Namespace:          kube-system
Priority:           2000000000
PriorityClassName:  system-cluster-critical
Node:               docker-desktop/192.168.65.3
Start Time:         Fri, 26 Mar 2021 06:55:35 +0900
Labels:             component=kube-apiserver
                    tier=control-plane
Annotations:        kubernetes.io/config.hash: b1dff398070b11d23d8d2653b78d430e
                    kubernetes.io/config.mirror: b1dff398070b11d23d8d2653b78d430e
                    kubernetes.io/config.seen: 2020-11-28T12:37:52.9356967Z
                    kubernetes.io/config.source: file
Status:             Running
IP:                 192.168.65.3
Containers:
  kube-apiserver:
    Container ID:  docker://0ef5c301524a57d15cae2330e7d7f51a94d3c5e0ac379b4975a1b5bf16d9a677
    Image:         k8s.gcr.io/kube-apiserver:v1.14.8
    Image ID:      docker-pullable://k8s.gcr.io/kube-apiserver@sha256:03cb15b3c4c7c5bca518bd07c1731ce939f8608d25d220af82e28e0ac447472a
    Port:          <none>
    Host Port:     <none>
    Command:
      kube-apiserver
      --advertise-address=192.168.65.3
      --allow-privileged=true
      --authorization-mode=Node,RBAC
      --client-ca-file=/run/config/pki/ca.crt
      --enable-admission-plugins=NodeRestriction
      --enable-bootstrap-token-auth=true
      --etcd-cafile=/run/config/pki/etcd/ca.crt
      --etcd-certfile=/run/config/pki/apiserver-etcd-client.crt
      --etcd-keyfile=/run/config/pki/apiserver-etcd-client.key
      --etcd-servers=https://127.0.0.1:2379
      --insecure-port=0
      --kubelet-client-certificate=/run/config/pki/apiserver-kubelet-client.crt
      --kubelet-client-key=/run/config/pki/apiserver-kubelet-client.key
      --kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname
      --proxy-client-cert-file=/run/config/pki/front-proxy-client.crt
      --proxy-client-key-file=/run/config/pki/front-proxy-client.key
      --requestheader-allowed-names=front-proxy-client
      --requestheader-client-ca-file=/run/config/pki/front-proxy-ca.crt
      --requestheader-extra-headers-prefix=X-Remote-Extra-
      --requestheader-group-headers=X-Remote-Group
      --requestheader-username-headers=X-Remote-User
      --secure-port=6443
      --service-account-key-file=/run/config/pki/sa.pub
      --service-cluster-ip-range=10.96.0.0/12
      --tls-cert-file=/run/config/pki/apiserver.crt
      --tls-private-key-file=/run/config/pki/apiserver.key
    State:          Running
      Started:      Fri, 26 Mar 2021 06:55:37 +0900
    Last State:     Terminated
      Reason:       Error
      Exit Code:    255
      Started:      Thu, 25 Mar 2021 21:08:48 +0900
      Finished:     Fri, 26 Mar 2021 06:55:22 +0900
    Ready:          True
    Restart Count:  21
    Requests:
      cpu:        250m
    Liveness:     http-get https://192.168.65.3:6443/healthz delay=15s timeout=15s period=10s #success=1 #failure=8
    Environment:  <none>
    Mounts:
      /etc/ca-certificates from etc-ca-certificates (ro)
      /etc/ssl/certs from ca-certs (ro)
      /run/config/pki from k8s-certs (ro)
      /usr/local/share/ca-certificates from usr-local-share-ca-certificates (ro)
      /usr/share/ca-certificates from usr-share-ca-certificates (ro)
Conditions:
  Type              Status
  Initialized       True
  Ready             True
  ContainersReady   True
  PodScheduled      True
Volumes:
  ca-certs:
    Type:          HostPath (bare host directory volume)
    Path:          /etc/ssl/certs
    HostPathType:  DirectoryOrCreate
  etc-ca-certificates:
    Type:          HostPath (bare host directory volume)
    Path:          /etc/ca-certificates
    HostPathType:  DirectoryOrCreate
  k8s-certs:
    Type:          HostPath (bare host directory volume)
    Path:          /run/config/pki
    HostPathType:  DirectoryOrCreate
  usr-local-share-ca-certificates:
    Type:          HostPath (bare host directory volume)
    Path:          /usr/local/share/ca-certificates
    HostPathType:  DirectoryOrCreate
  usr-share-ca-certificates:
    Type:          HostPath (bare host directory volume)
    Path:          /usr/share/ca-certificates
    HostPathType:  DirectoryOrCreate
QoS Class:         Burstable
Node-Selectors:    <none>
Tolerations:       :NoExecute
Events:            <none>

$ kubectl -n kube-system exec -it kube-apiserver-docker-desktop -c kube-apiserver -- sh

> pwd
/run/config/pki
> ls -alh
total 60K
drwxr-xr-x 3 root root  340 Mar 26 07:13 .
drwxr-xr-x 3 root root 4.0K Mar 25 21:55 ..
-rw-r--r-- 1 root root 1.1K Mar 26 07:13 apiserver-etcd-client.crt
-rw-r--r-- 1 root root 1.7K Mar 26 07:13 apiserver-etcd-client.key
-rw-r--r-- 1 root root 1.1K Mar 26 07:13 apiserver-kubelet-client.crt
-rw-r--r-- 1 root root 1.7K Mar 26 07:13 apiserver-kubelet-client.key
-rw-r--r-- 1 root root 1.4K Mar 26 07:13 apiserver.crt
-rw-r--r-- 1 root root 1.7K Mar 26 07:13 apiserver.key
-rw-r--r-- 1 root root 1.1K Mar 26 07:13 ca.crt
-rw-r--r-- 1 root root 1.7K Mar 26 07:13 ca.key
drwxr-xr-x 2 root root  200 Mar 26 07:13 etcd
-rw-r--r-- 1 root root 1.1K Mar 26 07:13 front-proxy-ca.crt
-rw-r--r-- 1 root root 1.7K Mar 26 07:13 front-proxy-ca.key
-rw-r--r-- 1 root root 1.1K Mar 26 07:13 front-proxy-client.crt
-rw-r--r-- 1 root root 1.7K Mar 26 07:13 front-proxy-client.key
-rw-r--r-- 1 root root 1.7K Mar 26 07:13 sa.key
-rw-r--r-- 1 root root  451 Mar 26 07:13 sa.pub
```

`ca.key` 가 private key 이고 `ca.crt` 가 root certificate 이다. `apiserver.crt`
는 `ca.crt` 로 부터 발급된 하위 인증서이다. kubernetes 는 user 는 하위 인증서를
이용하여 인증할 수 있다. 하위 인증서를 만들어서 `~/.kube/config` 에 저장하여
인증에 사용한다. 이것을 `x.509 Authentication` 이라고 한다.

다음과 같이 root certificate 으로 부터 하위 certificate 를 만들어 보자. 

```bash
# Create private key for iamslash user
$ openssl genrsa -out iamslash.key 2048

# Create CSR (Certificate Signing Request) file
# O means a group
# CN means a user
$ openssl req -new -key iamslash.key -out iamslash.csr -subj "/O=iamslash-org/CN=iamslash-cert"
```

kubeconfig file 에 기본적으로 설정된 인증서는 Organization 이 `system:master` 로
설정되어 있다. `system:masters` group 에 `cluster-admin` Clusterrole 이 할당되어
있기 때문이다. 

```bash
# kubenetes-admin.crt 는 kubeconfig 에 기본적으로 설정된 관리자 사용자의
# 인증서를 base64 로 디코딩한 뒤 저장한 파일이다.
$ openssl x509 -in kubernetes-admin.crt -noout -text

# Show ClusterRoleBinding of cluster-admin
$ kubectl describe clusterrolebinding cluster-admin
Name:         cluster-admin
Labels:       kubernetes.io/bootstrapping=rbac-defaults
Annotations:  rbac.authorization.kubernetes.io/autoupdate: true
Role:
  Kind:  ClusterRole
  Name:  cluster-admin
Subjects:
  Kind   Name            Namespace
  ----   ----            ---------
  Group  system:masters
```

이제 `iamslash.key` 라는 private key 로 `iamslash.csr` 에 서명하자. private key
file 을 이용하는 것은 위험하다. api 를 이용하여 간접적으로 서명해보자. 다음과
같이 `CerticateSigningRequest` object 를 만들어 본다.

* `iamslash-csr.yaml`

```yml
apiVersion: certificates.k8s.io/v1beta1
kind: CertificateSigningRequest
metadata:
  name: iamslash-csr
spec:
  groups:
  - system:authenticated
  request: <CSR>
  usages:
  - digital signature
  - key encipherment
  - client auth
```

`<CSR>` 에 `iamslash.csr` file 의 내용을 base64 encoding 해서 넣어보자.

```bash
$ export CSR=$(cat iamslash.csr | base64 | tr -d '\n')
$ sed -i -o "s/<CSR>/$CSR/g" iamslash-csr.yaml
```

이제 `iamslash-csr.yaml` 을 apply 하자.

```bash
# Apply CSR
$ kubectl apply -f iamslash-csr.yaml
# CONDITION will be pending
$ kubectl get csr
# Approve CSR, then CONDITION will be Approved,Issued
$ kubectl certificate approve iamslash-csr
# Extract Sub Certificate from CSR
$ kubectl get csr iamslash-csr -o jsonpath='{.status.certificate}' | base64 -d > iamslash.crt
```

이제 하위 인증서 `iamslash.crt` 를 추출했다. 하위 인증서와 비밀키로 새로운 user `iamslash-x509-user`
를 kubeconfig 에 등록해 보자. 그리고 새로운 context 도 만들어 본다.

```bash
$ kubectl config set-credentials iamslash-x509-user --client-certificate=iamslash.crt --client-key=iamslash.key
# Create context
$ kubectl config get-clusters
$ kubectl config set-context iamslash-x509-context --cluster kubernetes --user iamslash-x509-user
$ kubectl config use-context iamslash-x509-context
$ kubcctl get svc
```

그러나 permission error 발생할 것이다. 다음과 같이 RoleBinding 을 만들어 적용하자.

* `x509-cert-rolebinding-user.yaml`

```yml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: service-reader-rolebinding-user
  namespace: default
subjects:
- kind: User
  name: iamslash-cert
roleRef:
  kind: Role
  name: service-reader 
  apiGroup: rbac.authorization.k8s.io
```

```bash
$ kubectl config get-contexts
# You should use other context such as kubernetes-admin@kubernetes
# because iamslash-x509-context does not have permissions right now
$ kubectl apply -f x509-cert-rolebinding-user.yaml --context kubernetes-admin@kubernetes
$ kubectl get svc
```

## Resource Limit of Pods

## Major Kubernetes Objects

Reqeust, Limit, 
Guaranteed, BestEffort, Bursatable, 
ResourceQuota, LimitRange

### Limit

Limit 은 이 만큼의 자원이 컨테이너에게 제한된다는 것을 의미한다.

다음은 docker 에서 resource 의 limit 을 설정하는 방법이다.

```bash
$ docker run -it --name memory_1gb --memory 1g ubuntu:16.04
$ docker run -it --name cpu_1_alloc --cpu 1 ubuntu:16.04
$ docker run -it --name cpu_shares_example --cpu-shares 1024 ubuntu:16.04
# Unlimited
$ docker run -it --name unlimited_blade ubuntu:16.04
```

다음은 Limit 의 예이다.

* `resource-limit-pod.yaml`

```yml
apiVersion: v1
kind: Pod
metadata:
  name: resource-limit-pod
  labels:
    name: resource-limit-pod
spec:
  containers:
  - name: nginx
    image: nginx:latest
    resources:
      limits:
        memory: "256Mi"
        cpu: "1000m"
```

```bash
$ kubectl apply -f resource-limit-pod.yaml
$ kubectl get pods -o wide
NAME                 READY   STATUS    RESTARTS   AGE    IP          NODE             NOMINATED NODE   READINESS GATES
resource-limit-pod   1/1     Running   0          108s   10.1.1.86   docker-desktop   <none>           <none>
$ kubectl describe node ip-xxx
Non-terminated Pods:         (10 in total)
  Namespace                  Name                                      CPU Requests  CPU Limits  Memory Requests  Memory Limits  AGE
  ---------                  ----                                      ------------  ----------  ---------------  -------------  ---
  default                    resource-limit-pod                        1 (50%)       1 (50%)     256Mi (13%)      256Mi (13%)    2m12s
  docker                     compose-6c67d745f6-tzn4w                  0 (0%)        0 (0%)      0 (0%)           0 (0%)         121d
  docker                     compose-api-57ff65b8c7-89rxs              0 (0%)        0 (0%)      0 (0%)           0 (0%)         121d
  kube-system                coredns-6dcc67dcbc-gq8zk                  100m (5%)     0 (0%)      70Mi (3%)        170Mi (8%)     121d
  kube-system                coredns-6dcc67dcbc-v8nh8                  100m (5%)     0 (0%)      70Mi (3%)        170Mi (8%)     121d
  kube-system                etcd-docker-desktop                       0 (0%)        0 (0%)      0 (0%)           0 (0%)         121d
  kube-system                kube-apiserver-docker-desktop             250m (12%)    0 (0%)      0 (0%)           0 (0%)         121d
  kube-system                kube-controller-manager-docker-desktop    200m (10%)    0 (0%)      0 (0%)           0 (0%)         121d
  kube-system                kube-proxy-t57rq                          0 (0%)        0 (0%)      0 (0%)           0 (0%)         121d
  kube-system                kube-scheduler-docker-desktop             100m (5%)     0 (0%)      0 (0%)           0 (0%)         121d
Allocated resources:
  (Total limits may be over 100 percent, i.e., overcommitted.)
  Resource           Requests     Limits
  --------           --------     ------
  cpu                1750m (87%)  1 (50%)
  memory             396Mi (20%)  596Mi (31%)
  ephemeral-storage  0 (0%)       0 (0%)
Events:              <none>
```

### Request

Request 는 적어도 이 만큼의 자원은 컨테이너에게 보장돼야 하는 것을 의미한다.

다음은 Request 의 예이다.

* `resource-limit-with-request-pod.yaml`

```yml
apiVersion: v1
kind: Pod
metadata:
  name: resource-limit-with-request-pod
  labels:
    name: resource-limit-with-request-pod
spec:
  containers:
  - name: nginx
    image: nginx:latest
    resources:
      limits:
        memory: "256Mi"
        cpu: "1000m"
      requests:
        memory: "128Mi"
        cpu: "500m"
```

### CPU Limit

> [185. [Kubernetes] CPU Affinity를 위한 CPU Manager 사용 방법, 구현 원리 Deep Dive @ naverblog](https://blog.naver.com/alice_k106/221633530545)

특정 POD 가 특정 NODE 의 CPU 만을 이용하도록 제한하기 위해서는 CPU Manager 를
이용해야 한다. CPU Manager 는 kubelet 의 실행옵션을 변경해야 한다.

### QoS class, Memory Limit

Linux 에서 OOM (Out Of Memory) Killer 는 process 별로 점수를 매기고
MemoryPressure 가 True 인 상황에 점수가 가장 높은 process 를 kill 한다. 다음과
같이 process 별로 oom_score_adj 를 확인할 수 있다.

```bash
$ ps aux | grep dockerd
$ ls /proc/1234/
oom_adj  oom_score  oom_score_adj
$ cat /proc/1234/oom_score_adj
-999
```

kubernetes 는 pod 의 limit, request 값에 따라 pod 의 qos class 를 정한다. QoS
class 는 BestEffort, Burstable, Guaranteed 와 같이 총 3 가지가 있다.

Kubernetes 는 memory 가 부족하면 우선순위가 가장 낮은 POD 를 특정 node 에서
퇴거시킨다. 만약 memory 가 갑작스럽게 높아지면 OOM Killer 가 oom_score_adj 가
가장 낮은 process 를 강제로 종료한다. 그리고 pod 의 restart policy 에 의해 다시
시작된다.

pod 의 우선순위는 BestEffort, Burstable, Guaranteed 순으로 높아진다.

**Guaranteed**

Limit 과 Request 가 같은 POD 는 QosClass 가 Guaranteed 이다.

* `resource-lmiit-pod.yaml`

```yml
apiVersion: v1
kind: Pod
metadata:
  name: resource-limit-pod
  labels:
    name: resource-limit-pod
spec:
  containers:
  - name: nginx
    image: nginx:latest
    resources:
      limits:
        memory: "256Mi"
        cpu: "1000m"
```

* `resource-limit-pod-guaranteed.yaml`

```yml
apiVersion: v1
kind: Pod
metadata:
  name: resource-limit-pod-guaranteed
  labels:
    name: resource-limit-pod-guaranteed
spec:
  containers:
  - name: nginx
    image: nginx:latest
    resources:
      limits:
        memory: "256Mi"
        cpu: "1000m"
      requests:
        memory: "256Mi"
        cpu: "1000m"
```

QoS class 가 Guaranteed 이면 oom_score_adj 가 -998 이다.

**BestEffort**

Request, Limit 을 설정하지 않는 POD 는 QoS class 가 BestEffort 이다.

* `nginx-besteffort-pod.yaml`

```yml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-besteffort-pod
spec:
  containers:
  - name: nginx-besteffort-pod
    image: nginx:latest
```

node 에 존재하는 모든 자원을 사용할 수도 있지만 자원을 전혀 사용하지 못할 수도 있다.

**Burstable**

Request 가 Limit 보다 작은 POD 는 QoS class 가 Burstable 이다.

* `resource-limit-with-request-pod.yaml`

```yml
apiVersion: v1
kind: Pod
metadata:
  name: resource-limit-with-request-pod
  labels:
    name: resource-limit-with-request-pod
spec:
  containers:
  - name: nginx
    image: nginx:latest
    resources:
      limits:
        memory: "256Mi"
        cpu: "1000m"
      requests:
        memory: "128Mi"
        cpu: "500m"
```

### ResourceQuota

ResourceQuota 는 namespace 의 resource (cpu, memory, pvc size,
ephemeral-storage) 를 제한한다.

```bash
$ kubectl get quota
$ kubectl get resourcequota
```

* `resource-quota.yaml`

```yml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: resource-quota-example
  namespace: default
spec:
  hard:
    requests.cpu: "1000m"
    requests.memory: "500Mi"
    limits.cpu: "1500m"
    limits.memory: "1000Mi"
```

```bash
$ kubectl apply -f resource-quota.yaml
$ kubectl describe quota
Name:            resource-quota-example
Namespace:       default
Resource         Used  Hard
--------         ----  ----
limits.cpu       0     1500m
limits.memory    0     1000Mi
requests.cpu     0     1
requests.memory  0     500Mi

# This will fail to run
$ kubectl run memory-over-pod --image=nginx --generator=run-pod/v1 --request='cpu=200m,memory=300Mi' --limmits='cpu=200m,memory=3000Mi'
```

다음은 ResourceQuota 를 이용하여 cpu, memory, pods count, services count 를 제한하는 예이다.

* `quota-limit-pod-svc.yaml`

```yml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: resource-quota-example
  namespace: default
spec:
  hard:
    requests.cpu: "1000m"
    requests.memory: "500Mi"
    limits.cpu: "1500m"
    limits.memory: "1000Mi"
    count/pods: 3
    count/services: 5
```

다음은 ResourceQuota 를 이용하여 BestEffort Qos class 의 Pod 개수를 제한하는 예이다.

* `quota-limit-besteffort.yaml`

```yml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: besteffort-quota
  namespace: default
spec:
  hard:
    count/pods: 1
  scopes:
    - BestEffort
```

### LimitRange

LimitRange 는 네임스페이스에 할당되는 resource 의 범위 또는 기본값등을 설정한다.

다음은 LimitRange 의 예이다.

* `limitrange-example.yaml`

```yml
apiVersion: v1
kind: LimitRange
metadata:
  name: mem-limit-range
spec:
  limits:
  - default:                # 1. 자동으로 설정될 기본 Limit 값
      memory: 256Mi
      cpu: 200m
    defaultRequest:        # 2. 자동으로 설정될 기본 Request 값
      memory: 128Mi
      cpu: 100m
    max:                   # 3. 자원 할당량의 최대값
      memory: 1Gi
      cpu: 1000m
    min:                   # 4. 자원 할당량의 최소값
      memory: 16Mi
      cpu: 50m
    type: Container        # 5. 각 컨테이너에 대해서 적용
```

```bash
$ kubectl get limitranges
$ kubectl get limits
$ kubectl apply -f limitrange-example.yaml
```

다음은 value 대신 ratio 를 사용한 에이다.

* `limitrange-ratio.yaml`

```yml
apiVersion: v1
kind: LimitRange
metadata:
  name: limitrange-ratio
spec:
  limits:
  - maxLimitRequestRatio:
      memory: 1.5
      cpu: 1
    type: Container
```

다음은 pod 의 resource 범위를 제한하는 예이다. pod 의 resource usage
는 모든 container resource usage 의 합과 같다.

* `limitrange-example-pod.yaml`

```yml
apiVersion: v1
kind: LimitRange
metadata:
  name: pod-limit-range
spec:
  limits:
  - max:                   
      memory: 1Gi
    min:                   
      memory: 200Mi
    type: Pod       
```

### Admission Controller

다음은 Kubernetes API Flow 이다. Admission Controller 는 API 요청이 적절한지 검증하고 필요에 따라 API 요청을 변형한다. API 요청을 검증하는 것을 **Validating** 단계라고 한다. API 요청을 변형하는 것을 **Mutating** 단계라고 한다.

![](https://d33wubrfki0l68.cloudfront.net/af21ecd38ec67b3d81c1b762221b4ac777fcf02d/7c60e/images/blog/2019-03-21-a-guide-to-kubernetes-admission-controllers/admission-controller-phases.png)

ResourceQuota, LimitRange Admission Controller 는 다음과 같이 동작한다.

* user 가 `kubectl apply -f pod.yaml` 를 수행한다.
* x509 certificate, Service Account 등을 통해 Authentication 을 거친다.
* Role, Clusterrole 등을 통해 Authorization 을 거친다.
* ResourceQuota Admission Controller 는 POD 의 자원 할당 요청이 적절한지 Validating 한다. 만약 POD 로 인해서 해당 ResourceQuota 로 설정된 namespace resource 제한을 넘어선 다면 API 요청은 거절된다.
* LimitRange Admission Controller 는 cpu, memory 할당의 기본값을 추가한다. 즉, 원래 API 요청을 변형한다.

Custom Admission Controller 를 만들 수도 있다. [kubernetes extension @ TIL](kubernetes_extension.md). 예를 들어 nginx pod 을 생성할 때 실수로 적혀진 port number 를 Custom Admission Controller 에서 수정할 수도 있다.

Istio 는 Admission Controller 를 통해서 pod 에 proxy side car container 를 Injection 한다.

## Kubernetes Scheduling

### nodeName, nodeSelector, nodeAffinity, podAffinity, podAntiAffinity

kube-scheduler 는 node filtering, node scoring 의 과정을 통해 scheduling 한다. 즉, worker-node 에 pod 을 할당 한다. 이것은 etcd 에 저장된 pod data 의 nodeName 을 특정 worker-node 의 이름으로 변경하는 것을 의미한다. [kubernetes/pkg/scheduler/framework/plugins/ @ github](https://github.com/kubernetes/kubernetes/tree/master/pkg/scheduler/framework/plugins) 에서 filtering, scoring 의 code 를 확인할 수 있다.

node scoring 은 customizing 할 이유가 거의 없다. node filtering 은 nodeName, nodeSelector, nodeAffinity, podAffinity, taints, tolerations, cordon, drain 등을 통해서 가능하다.

가장 간단한 scheduling 방법은 nodeName 을 이용하는 것이다. 그러나 권장하지 않는다.

> `nodename-nginx.yaml`

```yml
apiVersion: v1
kind: Pod
metadata:
  name: nginx
spec:
  nodeName: ip-10-43-0-30.ap-northeast-2.compute.internal
  containers:
  - name: nginx
    image: nginx:latest
```

```bash
$ kubectl apply -f nodename-nginx.yaml

$ kubectl get pods -o wide
```

이번에는 NodeSelector 를 이용하여 특정 label 를 갖는 node 에 pod 를 할당하자. 

> `nodeselector-nginx.yaml`

```yml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-nodeselector
spec:
  nodeSelector:
    mylabel/disk: hdd
  containers:
  - name: nginx
    image: nginx:latest
```

node 의 label 은 다음과 같은 방법으로 CRUD 할 수 있다.

```bash
$ kubectl get nodes --show-labels

# Add the label to node
$ kubectl label nodes xxx.xxx.xxx.xxx mylabel/disk=ssd
$ kubectl label nodes xxx.xxx.xxx.xxx mylabel/disk=hdd

# Del the label from node
$ kubectl label nodes xxx.xxx.xxx.xxx mylabel/disk=hdd-
```

nodeAffinity 를 이용하면 nodeSelector 보다 조건을 정밀하게 설정할 수 있다.

> `nodeaffinity-required.yaml`

```yml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-nodeaffinity-required
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: mylabel/disk         
            operator: In          # values의 값 중 하나만 만족하면 됩니다.
            values:
            - ssd
            - hdd 
  containers:
  - name: nginx
    image: nginx:latest
```

requiredDuringSchedulingIgnoredDuringExecution 은 scheduling 하기 전에는 필수로 적용하고 일단 scheduling 되고 나면 무시하라는 의미이다.

preferredDuringSchedulingIgnoredDuringExecution 은 scheduling 하기 전에는 최대한 적용하고 일단 scheduling 되고 나면 무시하라는 의미이다. 적용안될 수도 있다.

> `nodeaffinity-preferred.yaml`

```yml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-nodeaffinity-preferred
spec:
  affinity:
    nodeAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 80               # 조건을 만족하는 노드에 1~100까지의 가중치를 부여
        preference:
          matchExpressions:
          - key: mylabel/disk
            operator: In
            values:
            - ssd
  containers:
  - name: nginx
    image: nginx:latest
```

podAffinity 를 이용하여 조건을 만족하는 pod 와 함께 실행되게 할 수 있다.

> `podaffinity-required.yaml`

```yml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-pod-antiaffinity
spec:
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
          - key: mylabel/database
            operator: In
            values:
            - mysql
        topologyKey: failure-domain.beta.kubernetes.io/zone
  containers:
  - name: nginx
    image: nginx:latest
```

topologyKey 는 해당 라벨을 가지는 worker-node 에서 조건을 수행하라는 의미이다. 예를 들어 kubernetes.io/zone 은 같은 avilability zone 의 worker-node 에서 조건을 수행하라는 의미이다.

topologyKey 를 `kubernetes.io/hostname` 으로 설정해 두면 matchExpressions 를 만족하는 worker-node 에서 pod 을 실행한다. 모든 worker-node 는 자신만의 hostname 을 갖기 때문이다.

> `podaffinity-hostname-topology.yaml`

```yml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-podaffinity-hostname
spec:
  affinity:
    podAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
          - key: mylabel/database
            operator: In
            values:
            - mysql
        topologyKey: kubernetes.io/hostname
  containers:
  - name: nginx
    image: nginx:latest
```

podAntiAffinity 는 podAffinity 와 반대의 의미를 갖는다. 예를 들어 다음과 같은 경우 matchExpressions 조건을 만족하는 pod 가 위치한 node 와 다른 topology 의 node 에 pod 를 할당한다.

> `pod-anitiaffinity-required.yaml`

```yml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-pod-antiaffinity
spec:
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
          - key: mylabel/database
            operator: In
            values:
            - mysql
        topologyKey: failure-domain.beta.kubernetes.io/zone
  containers:
  - name: nginx
    image: nginx:latest
```

### Taints, Tolerations

kube-scheduler 는 taints, tolerations 을 통해 node filtering 을 수행할 수 있다.
taints 는 node 에 표식을 하여 pod 이 할당되지 않게 하는 것이다. tolerations 는
node 의 taints 가 있음에도 불구하고 pod 가 할당되도록 하는 것이다.

다음과 같은 방법으로 taints 를 설정할 수 있다.

```bash
# Create taint
$ kubectl taint node xxx.xxx.xxx.xxx iamslash/my-taint=dirty:NoSchedule

# Remove taint
$ kubectl taint node xxx.xxx.xxx.xxx iamslash/my-tain:NoSchedule-
```

taint value 의 형식은 label 과 비슷하다. `<key>=<value>:<effect>` 이다.

`<effect>` 는 `NoSchedule, NoExecute, PreferNoSchedule` 과 같이 3 가지가 있다.

* **NoSchedule** : pod 를 스케줄하지 말자
* **NoExecute** : pod 를 스케줄도 하지 말고 실행된 pod 이 있으면 퇴거 (evict) 시키자. 
* **PreferNoSchedule** : 가능하면 pod 를 스케줄하지 말자

이번에는 tolertaions 을 이용하여 taint 가 부착된 node 에 pod 을 할당해 보자.

> `tolertation-test.yaml`

```yml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-toleration-test
spec:
  tolerations:
  - key: iamslash/my-taint 
    value: dirty              
    operator: Equal          # iamslash/my-taint 키의 값이 dirty이며 (Equal)
    effect: NoSchedule       # Taint 효과가 NoSchedule인 경우 해당 Taint를 용인합니다.
  containers:
  - name: nginx
    image: nginx:latest
```

일반적으로 pod 는 master-node 에는 할당되지 않는다. worker-node 에 할당된다. 이것은 master-node 에 taint 가 있기 때문이다. master-node 의 taint 는 `<key>=<value>:<effect>` 형식에서 `=<value>` 가 생략되어 있음을 주의 하자. 이 경우는 `<value>` 가 비어 있는 것이다.

```bash
$ kubectl describe node <master-node-name>
Taints:             node-role.kubernetes.io/master:NoSchedule
Unschedulable:      false
```

다음은 master-node 의 taint 에도 불구하고 tolerations 를 이용하여 pod 를 할당하는 예이다.

> `toleration-maser.yaml`

```yml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-master-toleration
spec:
  tolerations:
  - key: node-role.kubernetes.io/master  
    effect: NoSchedule                  
    operator: Equal
    value: ""
  nodeSelector:
    node-role.kubernetes.io/master: ""   # 마스터 노드에서도 포드가 생성되도록 지정합니다.
  containers:
  - name: nginx
    image: nginx:latest
```

그렇다면 master-node 에서 실행 중인 pod 는 어떤 tolerations 가 있는지 확인해 보자. `:NoExecute` 를 확인할 수 있다. taint 의 형식 `<key>=<value>:<effect>` 에서 `<effect>` 부분만 일치하는 taint 가 부착된 worker-node 에 pod 을 할당할 수 있다.

```bash
$ kubectl get pods -n kube-system | grep api
$ kubectl -n kube-system describe pod kube-apiserver-docker-desktop
QoS Class:         Burstable
Node-Selectors:    <none>
Tolerations:       :NoExecute
Events:            <none>

$ kubectl -n kube-system get pod kube-apiserver-docker-desktop -o yaml | grep -F2 toleration
...
  terminationGracePeriodSeconds: 30
  tolerations:
  - effect: NoExecute
    operator: Exists
...    
```

Kubernetes 는 worker-node 에 문제가 발생하면 taint 를 부착하여 pod 의 scheduling 을 막는다. 다음과 같은 taint 들이 있다. [Taint based Evictions @ kubernetes.io](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/#taint-based-evictions)

* **NotReady**: work-node 가 아직 준비되지 않은 상태
* **Unreachable**: Network 가 불안한 상태
* **Memory-Pressure**: Memory 가 부족한 상태
* **Disk-Pressure**: Disk 가 부족한 상태

또한 `tolerationSeconds: 300` 을 사용하면 taint 를 허용하는 시간을 정할 수 있다. 즉, `300 s` 가 지나도 taint 가 그대로 부착되어 있으면 pod 을 evict 한다.

```
$ kubectl get pod <pod-name> -o yaml | grep -F4 tolerationSeconds
```

### Cordon, Drain, PodDisruptionBudget

**cordon** 을 이용하면 worker-node 에 pod 을 scheduling 되지 않도록 할 수 있다.

```bash
$ kubectl cordon <node-name>
$ kubectl uncordon <node-name>
$ kubectl get nodes
$ kubectl describe node <node-name>
Taints:         node.kubernetes.io/unschedulable:NoSchedule
Unschedulable:  true
```

cordon 을 수행한 worker-node 에 `node.kubernetes.io/unschedulable:NoSchedule`
taint 와 `Unschedulable` 이 true 가 되어있다. `NoSchedule` 이기 때문에 실행중인
pod 이 evict 되지는 않는다. 그러나 ingress 의 traffic 은 pod 으로 전달 되지 않는다.
[kubectl cordon causes downtime of ingress(nginx) @ github](https://github.com/kubernetes/kubernetes/issues/65013)

**drain** 은 scheduling 도 하지 않고 실행중인 pod 을 evict 한다.

```bash
$ kubectl drain <node-name>
# If there are daemonsets, you need to use --ignore-daemonset option
$ kubectl drain <node-name> --ignore-daemonsets
$ kubectl get nodes
```

Deployment, ReplicaSet, Job, StatefulSet 에 의해 생성되지 않은 pod 가 있다면
drain 은 실패한다. `--force` 옵션을 사용하면 drain 할 수 있다.

**PodDisruptionBudget** 은 drain 이 수행되었을 때 evict 되는 pod 의 개수를
조정하는 것이다. pod 제공하는 service 를 유지하면서 evict 할 수 있다.

* `simple-pdb-example.yaml`

```yml
apiVersion: policy/v1beta1
kind: PodDisruptionBudget
metadata:
  name: simple-pdb-example
spec:
maxUnavailable: 1        # 비활성화될 수 있는 포드의 최대 갯수 또는 비율 (%) 
  # minAvailable: 2
  selector:                 # PDB의 대상이 될 포드를 선택하는 라벨 셀렉터 
    matchLabels:
      app: webserver
```

PodDisruptionBudget 는 maxUnavailable 혹은 minAvailable 중 하나만 사용할 수
있다. maxUnavilable 은 비활성화 될 수 있는 pod 의 최대 개수 혹은 비율이다.
minAvailable 은 활성화 될 수 있는 pod 의 최소 개수 혹은 비율이다. maxUnavailable
을 0% 혹은 minAvailable 을 100% 로 하면 evict 가 안된다.

이때 select 의 lable 은 Deployment 의 label 이 아닌 pod 의 label 이어야 한다.

* `deployment-pdb-test.yaml`

```yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deployment-pdb-test
spec:
  replicas: 3
  selector:
    matchLabels:
      app: webserver
  template:
    metadata:
      name: my-webserver
      labels:
        app: webserver
    spec:
      containers:
      - name: my-webserver
        image: alicek106/rr-test:echo-hostname
        ports:
        - containerPort: 80
```

### Custom Scheduler

* [Custom Scheduler @ TIL](kubernetes_extension.md#custom-scheduler)

### Static Pods vs DaemonSets

Static pods 는 Kube-api server 를 이용하지 않고 kubelet 이 실행하는 pods 이다. 주로 Master Node 의 kube-system component 들이 해당된다. 주로 `/etc/kubernetes/manifests` 의 pod manifestfile 들을 실행한다. (`etcd.yaml`, `kube-apiserver.yaml`, `kube-controller-manager.yaml`, `kube-scheduler.yaml`)

| Static PODs | DaemonSets |
|--|--|
| Created by the Kubelet | Created by Kube-API server (DaemonSet Controller) |
| Deploy Control Plane components as Static Pods | Deploy Monitoring Agents, Logging Agents on nodes |
| Ignored by the Kube-Scheuler | Ignored by the Kube-Scheuler |

* `kubelet.service` 를 살펴보면 pods manifestfile 의 path 를 알 수 있다.

```systemd
ExecStart=/usr/local/bin/kubelet \\
  --container-runtime=remote \\
  --container-runtime-endpoint=unix://var/run/containerd/containerd.sock \\
  --config=kubeconfig.yaml \\
  --kubeconfig=/var/lib/kubelet/kubeconfig \\
  --network-plugin=cni \\
  --register-node=true \\
  --v=2
```

* `kubeconfig.yaml`

```yaml
staticPodPath: /etc/kubernetes/manifests
```

### configuring Scheduler

Deploy Additional Scheduler

```bash
$ wget https://storage.googleapis.com/kubernetes-release/release/v1.12.0/bin/linux/amd64/kube-scheduler
```

* `kube-scheduler.service`
  ```
  ExecStart=/usr/local/bin/kube-scheduler \\
    --config=/etc/kubernetes/config/kube-scheduler.yaml \\  
    --scheduler-name= default-scheduler
  ``` 

* `my-custom-scheduler.service`
  ```
  ExecStart=/usr/local/bin/kube-scheduler \\
    --config=/etc/kubernetes/config/kube-scheduler.yaml \\
    --scheduler-name= my-custom-scheduler
  ```

## Kubernetes Application Status, Deployment

### Rolling update with Deployment

`--record` 를 이용하면 이전에 사용했던 replicaSet 이 deployment history 에
기록된다.

* `deployment-v1.yaml`

```yml  
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      name: nginx
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.15
        ports:
        - containerPort: 80
```

* `deployment-v2.yaml`

```yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deployment-recreate
spec:
  replicas: 3
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      name: nginx
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.16
        ports:
        - containerPort: 80
```

```bash
$ kubectl apply -f deployment-v1.yaml --record
$ kubectl get pods
$ kubectl apply -f deployment-v2.yaml --record
$ kubectl rollout history deployment nginx-deployment
```

기본적으로 replicaSet 의 revision 은 10 개까지 저장된다. 그러나 revisionHistoryLimit
을 설정하여 변경할 수 있다.

* `deployment-history-limit.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deployment-history-limit
spec:
  revisionHistoryLimit: 3
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      name: nginx
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.17
        ports:
        - containerPort: 80
```

Recreate Strategy 를 이용하면 기존 pod 를 삭제하고 새로운 pod 를 생성하기 때문에
서비스 중단이 발생할 수 있다.

* `deployment-recreate-v1.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deployment-recreate
spec:
  replicas: 3
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      name: nginx
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.15
        ports:
        - containerPort: 80
```

* `deployment-recreate-v2.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deployment-recreate
spec:
  replicas: 3
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      name: nginx
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.16
        ports:
        - containerPort: 80
```

```bash
$ kubectl apply -f deployment-recreate-v1.yaml
$ kubectl get pods
$ kubectl apply -f deployment-recreate-v2.yaml
$ kubectl get pods
```

RollingUpdate strategy 를 사용하면 서비스 중단 없이 pod 를 교체할 수 있다.

* `deployment-rolling-update.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deployment-rolling-update
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 2
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      name: nginx
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.15
        ports:
        - containerPort: 80
```

* maxUnavailable 은 rollingUpdate 도중 사용이 불가능한 pod 의 최대 개수 혹은
  비율이다. 이때 replicas 의 수를 기준으로 비율이 계산된다. 소수점 값은
  버려진다. 예를 들어 maxUnavailable 이 25% 이라면 75% 의 pod 는 서비스가
  가능해야 한다.
* maxSurge 는 rollingUpdate 도중 새롭게 생성된 pod 의 최대 개수 혹은 비율이다.
  이때 replicas 의 수를 기준으로 비율이 계산된다. 소수점 값은 반올림된다. 예를
  들어 maxSurge 가 25% 이라면 legacy pod, new pod 의 개수는 replicas 값 대비
  최대 125% 까지 늘어날 수 있다.

### BlueGreen update 

다음과 같은 방법으로 BlueGreen 배포를 사용한다.

* 기존 버전(v1) 의 Deployment 가 생성되어 있다. 서비스는 v1 의 pod 로 Request 를
  전달한다.
* 새로운 버전(v2) 의 Deployment 를 생성한다.
* Service 의 label 을 변경하여 Request 를 v2 의 pod 으로 전달한다.
* v1 Deployment 를 삭제한다.

### LifeCyle

다음은 pod 의 life cycle 이다.

* **Pending**: kube-apiserver 로 pod 생성 request 가 도착했지만 아직 worker-node 에
  생성되지 않았다.
* **Running**: pod 의 container 들이 모두 생성되었다.
* **Completed**: pod 의 container 들이 모두 실행되었다.
* **Error**: pod 의 container 들중 몇몇이 정상적으로 종료되지 못했다.
* **Terminating**: pod 가 delete 혹은 evict 되기 위해 머물러 있는 상태이다.

다음은 pod 가 다시실행되는 경우이다.

* restartPolicy 가 Always 이면 pod 가 Completed 일 때 다시 실행된다. 
* restartPolicy 가 Always 혹은 OnFailure 이면 pod 가 Error 일 때 다시 실행된다.
* 다시 실행될때는 Completed 혹은 Error 에서 CrashLoopBackOff 상태로 변한다.
* 다시 실행될때 마다 CrashLoopBackOff 의 유지기간은 지수시간만큼 늘어난다.

pod 는 init container, post start 가 제대로 실행이 완료되어야 Running 상태로 전환할 수 있다.

* `init-container-example.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: init-container-example
spec:
  initContainers: # 초기화 컨테이너를 이 항목에 정의합니다.
  - name: my-init-container
    image: busybox
    command: ["sh", "-c", "echo Hello World!"]
  containers: # 애플리케이션 컨테이너를 이 항목에 정의합니다.
  - name: nginx
    image: nginx
```

```bash
$ kubectl apply -f init-container-example.yaml
```

pod 는 init container 를 이용하여 resource 의 dependency 를 설정할 수 있다.
다음은 myservice 가 만들어질 때까지 pod 이 기다리는 예이다.

* `init-container-uppercase.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: init-container-usecase
spec:
  containers:
  - name: nginx
    image: nginx
  initContainers:
  - name: wait-other-service
    image: busybox
    command: ['sh', '-c', 'until nslookup myservice; do echo waiting..; sleep 1; done;']
```

postStart 는 container 가 시작하고 수행하는 hook 이다. 반면에 preStop 은 container 가
종료되기 전에 수행하는 hook 이다. postStart 는 HTTP, Exec 가 가능하다.

* `poststart-hook.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: poststart-hook
spec:
  containers:
  - name: poststart-hook
    image: nginx
    lifecycle:
      postStart:
        exec:
          command: ["sh", "-c", "touch /myfile"]
```

```bash
$ kubectl apply -f poststart-hook.yaml
$ kubectl exec poststart-hook ls /myfile
```

### LivenessProbe, ReadinessProbe

LivenessProbe 는 container 가 살아있는지 검사하는 것이다. 실패하면 pod 의
restartPolicy 에 의해 재시작한다. httpGet, exec, tcpSocket 이 가능하다.

ReadinessProbe 는 container 가 준비되어있는지 검사하는 것이다. 실패하면 Service
의 routing 대상에서 제외된다.

다음은 livenessProbe 의 예이다.

* `livenessprobe-pod.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: livenessprobe-pod
spec:
  containers:
  - name: livenessprobe-pod
    image: nginx
    livenessProbe:  # 이 컨테이너에 대해 livenessProbe를 정의합니다.
      httpGet:      # HTTP 요청을 통해 애플리케이션의 상태를 검사합니다.
        port: 80    # <포드의 IP>:80/ 경로를 통해 헬스 체크 요청을 보냅니다. 
        path: /
```

```bash
$ kubectl apply -f livenessprobe-pod.yaml
$ kubectl get pods
$ kubectl exec livenessprobe-pod -- rm /usr/share/nginx/html/index.html
$ kubectl describe po livenessprobe-pod
$ kubectl get events --sort-by=.metadata.creationTimestamp
```

다음은 readinessProbe 의 예이다.

* `readinessprobe-pod-svc.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: readinessprobe-pod
  labels:
    my-readinessprobe: test
spec:
  containers:
  - name: readinessprobe-pod
    image: nginx       # Nginx 서버 컨테이너를 생성합니다.
    readinessProbe:    # <포드의 IP>:80/ 로 상태 검사 요청을 전송합니다.
      httpGet:
        port: 80
        path: /
---
apiVersion: v1
kind: Service
metadata:
  name: readinessprobe-svc
spec:
  ports:
    - name: nginx
      port: 80
      targetPort: 80
  selector:
    my-readinessprobe: test
  type: ClusterIP
```

```bash
$ kubectl apply -f readinessprobe-pod-svc.yaml
$ kubectl get pods -w
$ kubectl run -it --rm debug --image=alicek106/ubuntu:curl --restart=Never -- curl readinessprobe-svc
$ kubectl get endpoints
$ kubectl exec readinessprobe-pod -- rm /usr/share/nginx/html/index.html
$ kubectl get pods -w
$ kubectl run -it --rm debug --image=alicek106/ubuntu:curl --restart=Never -- curl --connect-timeout 5 readinessprobe-svc
$ kubectl get endpoints
```

readinessProbe 를 이용하기 어려운 경우는 minReadySeconds 를 이용하여 일정시간
지난 다음 pod 을 delete 하거나 create 한다.

* `minreadyseconds-v1.yaml`

```yaml
# Reference : https://github.com/kubernetes/kubernetes/issues/51671
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minreadyseconds-v1
spec:
  replicas: 1
  minReadySeconds: 30
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: minready-test
  template:
    metadata:
      labels:
        app: minready-test
    spec:
      containers:
      - name: minreadyseconds-v1
        image: alicek106/rr-test:echo-hostname
---
apiVersion: v1
kind: Service
metadata:
  name: myservice
spec:
  ports:
  - name: web-port
    port: 80
    targetPort: 80
  selector:
    app: minready-test
  type: NodePort
```

livenessProbe, readinessProbe 는 다음과 같은 option 들을 갖는다.

* **periodSeconds**: 검사 주기. 기본값은 10 초이다.
* **initialDelaySeconds**: 검사 전 대기시간. 기본값은 없다.
* **timeoutSeconds**: 요청의 타임아웃 시간. 기본값은 1 초이다.
* **successThreshold**: 검사가 성공하기 위한 횟수이다. 기본값은 1 이다.
* **failureThreshold**: 검사가 실패하기 위한 횟수이다. 기본값은 3 이다.

다음은 option 을 포함한 readinessProbe 의 예이다.

* `probe-options.yaml`

```yml
apiVersion: v1
kind: Pod
metadata:
  name: probe-options-example
  labels:
    my-readinessprobe: test
spec:
  containers:
  - name: probe-options-example
    image: nginx
    readinessProbe:
      httpGet:
        port: 80
        path: /
      periodSeconds: 5
      initialDelaySeconds: 10
      timeoutSeconds: 1
      successThreshold: 1
      failureThreshold: 3
```

### Terminating status

* [Process of Pod Termination](#process-of-pod-termination)

## Custom Resource Definition

## Kubernetes Objects using Pod Objects

### Jobs

Job 은 batch 와 같이 수행후 종료하는 Kubernetes object 이다.

### CronJobs

CronJob 은 주기적으로 실행되는 Kubernetes object 이다.

### DaemonSets

DaemonSet 은 모든 Node 에 동일한 Pod 를 하나씩 생성하는 Kubernetes object 이다.

### StatefulSets

StatefulSet 은 state 를 갖는 Pod 를 관리하는 Kubernetes object 이다.

* [1 스테이트풀셋(Statefulset)이란? @ naverblog](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=isc0304&logNo=221885403537)

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

# Advanced

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

## Process of Pod Termination

```
preStop       SIGTERM             SIGKILL
  |              |                   |
  |              |                   |
  |              |                   |
------------------------------------------->  
 t0              t1                  t2

t0: Pod will execute preStop when deletionTimestamp is added to pod data of ETCD.
t1: As soon as prestop is finished Pod will send SIGTERM to containers.
t2: What if containers are not terminated after SIGTERM during terminationGracePeriodSeconds, pod will send SIGKILL.

terminationGracePeriodSeconds's default value is 30s
```

## Kubernetes Extension

* [Kubernees Extension @ TIL](kubernetes_extension.md)

## API server

* [Kubernetes API Reference](https://kubernetes.io/docs/reference/)

## Monitoring

* [Prometheus with Kubernetes](https://www.slideshare.net/jinsumoon33/kubernetes-prometheus-monitoring)

# Continue...

* [README-3.md](README-3.md)
