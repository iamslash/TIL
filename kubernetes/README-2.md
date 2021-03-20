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
    - [kubeconfig](#kubeconfig)
    - [User, Group](#user-group)
    - [User authentication with X509](#user-authentication-with-x509)
  - [Resource Limit of Pods](#resource-limit-of-pods)
    - [Limit](#limit)
    - [Request](#request)
    - [CPU Limit](#cpu-limit)
    - [QoS class, Memory Limit](#qos-class-memory-limit)
    - [ResourceQuota](#resourcequota)
    - [LimitRange](#limitrange)
    - [Admission Controller](#admission-controller)
  - [Kubernetes Scheduling](#kubernetes-scheduling)
    - [NodeSelector, Node Affinity, Pod Affinity](#nodeselector-node-affinity-pod-affinity)
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

### kubeconfig

### User, Group

### User authentication with X509

## Resource Limit of Pods

Reqeust, Limit, 
Guaranteed, BestEffort, Bursatable, 
ResourceQuota, LimitRanger

### Limit

### Request

### CPU Limit

### QoS class, Memory Limit

### ResourceQuota

### LimitRange

### Admission Controller

## Kubernetes Scheduling

### NodeSelector, Node Affinity, Pod Affinity

nodeAffinity, podAffinity, topologyKey, reqruiedDuringSchedulingIgnoredDuringExecution, preferredDuringSchedulingIgnoredDuringExecution

### Taints, Tolerations

### Cordon, Drain, PodDisruptionBudget

### Custom Scheduler

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

### BlueGreen update 

### LifeCyle

### LivenessProbe, ReadinessProbe

### Terminating status

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
