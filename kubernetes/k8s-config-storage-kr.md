# Kubernetes 설정과 스토리지

- [Materials](#materials)
- [ConfigMap](#configmap)
  - [ConfigMap 생성](#configmap-생성)
  - [환경변수로 주입](#환경변수로-주입)
  - [볼륨 마운트로 주입](#볼륨-마운트로-주입)
  - [커맨드 인자로 주입](#커맨드-인자로-주입)
- [Secret](#secret)
  - [ConfigMap 과의 차이](#configmap-과의-차이)
  - [Secret 종류](#secret-종류)
  - [Opaque Secret 생성과 사용](#opaque-secret-생성과-사용)
  - [docker-registry Secret](#docker-registry-secret)
  - [tls Secret](#tls-secret)
  - [주의사항](#주의사항)
- [Volume](#volume)
  - [emptyDir](#emptydir)
  - [hostPath](#hostpath)
- [PersistentVolume](#persistentvolume)
  - [PV 란](#pv-란)
  - [accessModes](#accessmodes)
  - [reclaimPolicy](#reclaimpolicy)
  - [PV YAML 예제](#pv-yaml-예제)
- [PersistentVolumeClaim](#persistentvolumeclaim)
  - [PVC 란](#pvc-란)
  - [PV 와 PVC 의 바인딩 과정](#pv-와-pvc-의-바인딩-과정)
  - [PVC YAML 예제](#pvc-yaml-예제)
  - [Pod 에서 PVC 사용](#pod-에서-pvc-사용)
- [StorageClass](#storageclass)
  - [동적 프로비저닝이란](#동적-프로비저닝이란)
  - [StorageClass YAML 예제](#storageclass-yaml-예제)
  - [PVC 에서 StorageClass 지정](#pvc-에서-storageclass-지정)
  - [기본 StorageClass 설정](#기본-storageclass-설정)
- [실무 패턴](#실무-패턴)
  - [StatefulSet 과 PVC 템플릿](#statefulset-과-pvc-템플릿)
  - [ConfigMap 변경 시 Pod 재시작 전략](#configmap-변경-시-pod-재시작-전략)

---

# Materials

- [시작하세요! 도커/쿠버네티스](http://www.yes24.com/Product/Goods/84927385)
  - 한글책 중 최고
- [Kubernetes 공식 문서 - ConfigMap](https://kubernetes.io/docs/concepts/configuration/configmap/)
- [Kubernetes 공식 문서 - Secrets](https://kubernetes.io/docs/concepts/configuration/secret/)
- [Kubernetes 공식 문서 - Persistent Volumes](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)
- [Kubernetes 공식 문서 - Storage Classes](https://kubernetes.io/docs/concepts/storage/storage-classes/)

---

# ConfigMap

ConfigMap 은 설정 데이터를 Pod 와 분리하여 관리하는 오브젝트다. 환경마다 다른 설정값(데이터베이스 주소, 로그 레벨 등)을 컨테이너 이미지에 하드코딩하지 않고 외부에서 주입할 수 있다.

## ConfigMap 생성

kubectl 명령어로 직접 생성하거나 YAML 파일을 작성해서 생성할 수 있다.

```bash
# 리터럴 값으로 생성
kubectl create configmap app-config \
  --from-literal=LOG_LEVEL=debug \
  --from-literal=DB_HOST=mysql-service

# 파일로부터 생성 (파일 이름이 key 가 된다)
kubectl create configmap app-config --from-file=app.properties

# 디렉터리 내 파일 전체로부터 생성
kubectl create configmap app-config --from-file=./config-dir/
```

`configmap.yaml` - YAML 로 직접 정의하는 방법:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config          # ConfigMap 이름
  namespace: default
data:
  # 단순 키-값 쌍
  LOG_LEVEL: "debug"
  DB_HOST: "mysql-service"
  DB_PORT: "3306"
  # 파일 내용을 통째로 값으로 저장할 수도 있다
  app.properties: |
    spring.datasource.url=jdbc:mysql://mysql-service:3306/mydb
    spring.datasource.username=admin
    logging.level.root=DEBUG
```

```bash
kubectl apply -f configmap.yaml
kubectl get configmap app-config
kubectl describe configmap app-config
```

## 환경변수로 주입

Pod 의 컨테이너에 ConfigMap 의 특정 키를 환경변수로 주입하는 방법이다.

`pod-env-from-configmap.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: busybox
    args: ["env"]             # 환경변수 출력 후 종료
    env:
    # 특정 키 하나만 가져오기
    - name: LOG_LEVEL         # 컨테이너 내부에서 사용할 환경변수 이름
      valueFrom:
        configMapKeyRef:
          name: app-config    # ConfigMap 이름
          key: LOG_LEVEL      # ConfigMap 에서 가져올 키
    - name: DATABASE_HOST
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: DB_HOST
    # ConfigMap 전체를 한번에 환경변수로 로드하기
    envFrom:
    - configMapRef:
        name: app-config      # app-config 의 모든 키가 환경변수로 들어온다
```

```bash
kubectl apply -f pod-env-from-configmap.yaml
kubectl logs app-pod          # 출력된 환경변수 확인
```

## 볼륨 마운트로 주입

ConfigMap 의 각 키가 파일 이름이 되고, 값이 파일 내용이 되어 컨테이너 내부에 마운트된다. 설정 파일이 필요한 애플리케이션에 적합하다.

`pod-volume-from-configmap.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-pod-volume
spec:
  containers:
  - name: app
    image: busybox
    args: ["cat", "/etc/config/app.properties"]  # 마운트된 파일 읽기
    volumeMounts:
    - name: config-volume
      mountPath: /etc/config    # 이 경로에 ConfigMap 내용이 파일로 마운트된다
      readOnly: true
  volumes:
  - name: config-volume
    configMap:
      name: app-config          # 사용할 ConfigMap 이름
      # 특정 키만 마운트하고 싶을 때 items 를 지정한다
      items:
      - key: app.properties     # ConfigMap 의 키
        path: app.properties    # 마운트될 파일 이름
```

```bash
kubectl apply -f pod-volume-from-configmap.yaml
kubectl logs app-pod-volume
```

볼륨 마운트 방식은 ConfigMap 이 업데이트되면 일정 시간(기본 약 60초) 후 파일 내용이 자동으로 갱신된다는 장점이 있다. 반면 환경변수 방식은 Pod 를 재시작해야 반영된다.

## 커맨드 인자로 주입

환경변수를 경유하여 컨테이너 실행 인자에 ConfigMap 값을 넘길 수 있다.

`pod-args-from-configmap.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-pod-args
spec:
  containers:
  - name: app
    image: busybox
    env:
    - name: LOG_LEVEL
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: LOG_LEVEL
    # $(환경변수명) 문법으로 args 에서 참조한다
    command: ["/bin/sh"]
    args: ["-c", "echo Log level is $(LOG_LEVEL)"]
```

```bash
kubectl apply -f pod-args-from-configmap.yaml
kubectl logs app-pod-args
# 출력: Log level is debug
```

---

# Secret

Secret 은 비밀번호, API 키, TLS 인증서처럼 민감한 데이터를 저장하는 오브젝트다.

## ConfigMap 과의 차이

| 항목 | ConfigMap | Secret |
|------|-----------|--------|
| 용도 | 일반 설정값 | 민감한 데이터 |
| 저장 방식 | 평문 | base64 인코딩 |
| etcd 암호화 | 기본 비활성화 | 활성화 가능 (`EncryptionConfiguration`) |
| 메모리 저장 | 디스크 | tmpfs (메모리) |

## Secret 종류

| 타입 | 용도 |
|------|------|
| `Opaque` | 일반적인 임의 데이터 (기본값) |
| `kubernetes.io/dockerconfigjson` | 프라이빗 이미지 레지스트리 인증 |
| `kubernetes.io/tls` | TLS 인증서와 키 |
| `kubernetes.io/service-account-token` | 서비스 어카운트 토큰 |

## Opaque Secret 생성과 사용

```bash
# 리터럴 값으로 생성 (kubectl 이 자동으로 base64 인코딩한다)
kubectl create secret generic db-secret \
  --from-literal=DB_PASSWORD=mysecretpassword \
  --from-literal=DB_USERNAME=admin
```

YAML 로 직접 작성할 때는 값을 미리 base64 로 인코딩해야 한다.

```bash
# base64 인코딩 방법
echo -n "mysecretpassword" | base64
# 출력: bXlzZWNyZXRwYXNzd29yZA==
```

`secret-opaque.yaml`

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
  namespace: default
type: Opaque
data:
  # 값은 반드시 base64 로 인코딩된 문자열이어야 한다
  DB_USERNAME: YWRtaW4=              # admin
  DB_PASSWORD: bXlzZWNyZXRwYXNzd29yZA==  # mysecretpassword
```

Pod 에서 Secret 을 환경변수로 사용하는 예제:

`pod-with-secret-env.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-with-secret
spec:
  containers:
  - name: app
    image: busybox
    args: ["env"]
    env:
    - name: DB_PASSWORD             # 컨테이너 내부 환경변수 이름
      valueFrom:
        secretKeyRef:
          name: db-secret           # Secret 이름
          key: DB_PASSWORD          # Secret 의 키
    - name: DB_USERNAME
      valueFrom:
        secretKeyRef:
          name: db-secret
          key: DB_USERNAME
```

```bash
kubectl apply -f secret-opaque.yaml
kubectl apply -f pod-with-secret-env.yaml
kubectl logs app-with-secret
```

## docker-registry Secret

프라이빗 컨테이너 레지스트리에서 이미지를 pull 할 때 필요한 인증 정보를 저장한다.

```bash
kubectl create secret docker-registry registry-secret \
  --docker-server=my-registry.example.com \
  --docker-username=myuser \
  --docker-password=mypassword \
  --docker-email=myemail@example.com
```

`pod-with-registry-secret.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-private-image
spec:
  containers:
  - name: app
    image: my-registry.example.com/myapp:v1  # 프라이빗 레지스트리 이미지
  imagePullSecrets:
  - name: registry-secret   # 위에서 생성한 docker-registry Secret
```

## tls Secret

Ingress 에서 HTTPS 를 설정할 때 TLS 인증서와 키를 저장한다.

```bash
# 자체 서명 인증서 생성 (테스트용)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout tls.key -out tls.crt -subj "/CN=myapp.example.com/O=myorg"

# TLS Secret 생성
kubectl create secret tls tls-secret \
  --key tls.key \
  --cert tls.crt
```

`ingress-tls.yaml` 에서 사용하는 예:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
spec:
  tls:
  - hosts:
    - myapp.example.com
    secretName: tls-secret        # TLS Secret 이름
  rules:
  - host: myapp.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: app-service
            port:
              number: 80
```

## 주의사항

base64 는 암호화가 아니라 인코딩이다. base64 로 인코딩된 값은 누구든 쉽게 디코딩할 수 있다.

```bash
# base64 디코딩 예시 - 이처럼 쉽게 원래 값을 볼 수 있다
echo "bXlzZWNyZXRwYXNzd29yZA==" | base64 -d
# 출력: mysecretpassword
```

Secret 을 실제 환경에서 안전하게 사용하려면 다음을 고려해야 한다.

- etcd 저장 시 암호화 활성화 (`EncryptionConfiguration`)
- RBAC 로 Secret 접근 권한을 최소화
- HashiCorp Vault, AWS Secrets Manager 같은 외부 시크릿 관리 도구 연동 검토
- Secret YAML 파일을 Git 에 커밋하지 않기

---

# Volume

Pod 안의 컨테이너가 데이터를 저장하거나 컨테이너 간에 데이터를 공유할 때 Volume 을 사용한다.

## emptyDir

Pod 가 노드에 할당될 때 생성되고 Pod 가 삭제될 때 함께 삭제되는 임시 디렉터리다. 같은 Pod 내의 여러 컨테이너가 파일을 공유할 때 사용한다.

`pod-emptydir.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: shared-data-pod
spec:
  containers:
  - name: writer                        # 파일을 생성하는 컨테이너
    image: busybox
    command: ["/bin/sh", "-c"]
    args:
    - while true; do
        echo "$(date) hello" >> /data/log.txt;
        sleep 5;
      done
    volumeMounts:
    - name: shared-volume
      mountPath: /data                  # writer 컨테이너는 /data 에 마운트

  - name: reader                        # 파일을 읽는 컨테이너
    image: busybox
    command: ["/bin/sh", "-c"]
    args:
    - tail -f /log/log.txt
    volumeMounts:
    - name: shared-volume
      mountPath: /log                   # reader 컨테이너는 /log 에 마운트

  volumes:
  - name: shared-volume
    emptyDir: {}                        # Pod 내부 컨테이너들이 공유하는 임시 볼륨
```

```bash
kubectl apply -f pod-emptydir.yaml
kubectl logs shared-data-pod -c reader  # reader 컨테이너 로그 확인
```

emptyDir 은 기본적으로 노드의 디스크 공간을 사용한다. 메모리를 사용하고 싶다면 `medium: Memory` 를 지정한다.

```yaml
volumes:
- name: cache-volume
  emptyDir:
    medium: Memory      # 메모리(tmpfs) 사용 - 빠르지만 Pod 메모리 사용량에 포함됨
    sizeLimit: 128Mi    # 최대 크기 제한
```

## hostPath

노드(호스트)의 파일 시스템 경로를 Pod 에 마운트한다. Pod 가 삭제되어도 노드의 데이터는 유지된다. 주로 노드의 시스템 파일에 접근해야 하는 DaemonSet 용도로 사용한다.

`pod-hostpath.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hostpath-pod
spec:
  containers:
  - name: app
    image: busybox
    args: ["tail", "-f", "/dev/null"]
    volumeMounts:
    - name: host-volume
      mountPath: /host-data             # 컨테이너 내부에서 접근할 경로
  volumes:
  - name: host-volume
    hostPath:
      path: /tmp                        # 노드의 /tmp 디렉터리를 마운트
      type: DirectoryOrCreate           # 디렉터리가 없으면 생성
```

hostPath 의 `type` 값:

| type | 설명 |
|------|------|
| `""` | 사전 검사 없음 |
| `DirectoryOrCreate` | 디렉터리가 없으면 생성 |
| `Directory` | 디렉터리가 이미 존재해야 함 |
| `FileOrCreate` | 파일이 없으면 생성 |
| `File` | 파일이 이미 존재해야 함 |

```bash
kubectl apply -f pod-hostpath.yaml
kubectl exec -it hostpath-pod -- touch /host-data/test.txt
# 노드에 직접 접속하면 /tmp/test.txt 가 생성된 것을 확인할 수 있다
```

hostPath 는 Pod 가 특정 노드에 종속되므로 프로덕션 환경에서는 가급적 사용하지 않는 것이 좋다.

---

# PersistentVolume

## PV 란

PersistentVolume(PV) 은 클러스터 관리자가 미리 프로비저닝해 둔 스토리지 리소스다. NFS, AWS EBS, GCP Persistent Disk 같은 네트워크 스토리지를 추상화하여 Kubernetes 오브젝트로 표현한 것이다.

Pod 의 라이프사이클과 독립적으로 존재하므로 Pod 가 삭제되어도 PV 에 저장된 데이터는 유지된다.

## accessModes

PV 가 지원하는 접근 모드를 정의한다. 실제 지원 여부는 스토리지 종류에 따라 다르다.

| 모드 | 약어 | 설명 |
|------|------|------|
| `ReadWriteOnce` | RWO | 하나의 노드에서 읽기/쓰기 가능 |
| `ReadOnlyMany` | ROX | 여러 노드에서 읽기 전용으로 접근 가능 |
| `ReadWriteMany` | RWX | 여러 노드에서 읽기/쓰기 가능 |
| `ReadWriteOncePod` | RWOP | 하나의 Pod 에서만 읽기/쓰기 가능 (Kubernetes 1.22+) |

## reclaimPolicy

PVC 가 삭제된 후 PV 를 어떻게 처리할지 결정한다.

| 정책 | 설명 |
|------|------|
| `Retain` | PVC 삭제 후 PV 를 보존한다. 관리자가 수동으로 처리해야 한다. |
| `Delete` | PVC 삭제 시 PV 와 실제 스토리지(EBS 등)도 함께 삭제된다. |
| `Recycle` | 데이터를 초기화(`rm -rf`) 후 다시 사용 가능 상태로 만든다. 현재는 deprecated. |

## PV YAML 예제

`pv-nfs.yaml` - NFS 를 사용하는 PV 예제:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-nfs-data             # PV 이름
spec:
  capacity:
    storage: 10Gi               # 스토리지 용량
  accessModes:
  - ReadWriteMany               # NFS 는 다중 노드 읽기/쓰기를 지원한다
  persistentVolumeReclaimPolicy: Retain  # PVC 삭제 후 PV 보존
  nfs:
    server: 192.168.1.100       # NFS 서버 IP 또는 호스트명
    path: /exports/data         # NFS 서버에서 공유하는 경로
```

`pv-hostpath.yaml` - 로컬 테스트용 hostPath PV:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-local-test
spec:
  capacity:
    storage: 1Gi
  accessModes:
  - ReadWriteOnce               # hostPath 는 단일 노드만 지원한다
  persistentVolumeReclaimPolicy: Delete
  hostPath:
    path: /mnt/data             # 노드의 실제 경로
    type: DirectoryOrCreate
```

```bash
kubectl apply -f pv-nfs.yaml
kubectl get pv                  # STATUS 가 Available 이어야 한다
kubectl describe pv pv-nfs-data
```

---

# PersistentVolumeClaim

## PVC 란

PersistentVolumeClaim(PVC) 은 개발자가 스토리지를 요청하는 방법이다. 개발자는 실제 스토리지의 구현 세부 사항을 몰라도 "10Gi 의 ReadWriteOnce 스토리지가 필요하다"고 선언하면 된다.

Kubernetes 가 조건에 맞는 PV 를 찾아서 PVC 와 바인딩해준다.

## PV 와 PVC 의 바인딩 과정

```
개발자가 PVC 생성
       |
       v
Kubernetes 가 PVC 조건(용량, accessMode)에 맞는 PV 탐색
       |
  조건 충족 PV 발견
       |
       v
PV 와 PVC 가 1:1 바인딩 (STATUS: Bound)
       |
       v
Pod 에서 PVC 를 볼륨으로 마운트하여 사용
```

바인딩 조건:
- PVC 요청 용량 <= PV 용량
- PVC 의 accessMode 가 PV 의 accessModes 에 포함
- storageClassName 이 일치 (또는 둘 다 비어 있음)

## PVC YAML 예제

`pvc-data.yaml`

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-data                # PVC 이름
  namespace: default
spec:
  accessModes:
  - ReadWriteOnce               # 필요한 접근 모드
  resources:
    requests:
      storage: 5Gi              # 필요한 용량 (PV 용량 이하여야 바인딩 가능)
  # storageClassName 을 지정하지 않으면 기본 StorageClass 를 사용한다
  # 정적 프로비저닝 PV 를 사용하려면 storageClassName: "" 로 설정한다
  storageClassName: ""
```

```bash
kubectl apply -f pvc-data.yaml
kubectl get pvc                 # STATUS 가 Bound 이면 PV 와 연결된 것이다
kubectl describe pvc pvc-data
```

## Pod 에서 PVC 사용

`pod-with-pvc.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-with-pvc
spec:
  containers:
  - name: app
    image: busybox
    args: ["tail", "-f", "/dev/null"]
    volumeMounts:
    - name: persistent-storage
      mountPath: /app/data      # 컨테이너 내부에서 접근할 경로
  volumes:
  - name: persistent-storage
    persistentVolumeClaim:
      claimName: pvc-data       # 사용할 PVC 이름
```

```bash
kubectl apply -f pod-with-pvc.yaml
kubectl exec -it app-with-pvc -- sh
# 컨테이너 내부
> echo "persistent data" > /app/data/test.txt
> exit

# Pod 를 삭제하고 다시 생성해도 /app/data/test.txt 가 유지된다
kubectl delete pod app-with-pvc
kubectl apply -f pod-with-pvc.yaml
kubectl exec -it app-with-pvc -- cat /app/data/test.txt
# 출력: persistent data
```

---

# StorageClass

## 동적 프로비저닝이란

정적 프로비저닝은 관리자가 PV 를 미리 만들어두고 개발자가 PVC 로 요청하는 방식이다. 이 방식은 관리 부담이 크고 유연하지 않다.

동적 프로비저닝은 PVC 가 생성될 때 StorageClass 에 정의된 프로비저너가 자동으로 스토리지(EBS, GCP PD 등)를 생성하고 PV 를 만들어 바인딩한다. 개발자는 PVC 만 작성하면 된다.

```
개발자가 PVC 생성 (storageClassName 지정)
       |
       v
StorageClass 의 프로비저너가 실제 스토리지 자동 생성
       |
       v
PV 가 자동 생성되고 PVC 와 바인딩
       |
       v
Pod 에서 즉시 사용 가능
```

## StorageClass YAML 예제

`storageclass-aws-ebs.yaml` - AWS EBS 프로비저너:

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: ebs-sc                          # StorageClass 이름
provisioner: ebs.csi.aws.com            # AWS EBS CSI 드라이버 프로비저너
parameters:
  type: gp3                             # EBS 볼륨 타입 (gp2, gp3, io1 등)
  fsType: ext4                          # 파일 시스템 타입
reclaimPolicy: Delete                   # PVC 삭제 시 EBS 볼륨도 삭제
allowVolumeExpansion: true              # PVC 용량 확장 허용
volumeBindingMode: WaitForFirstConsumer # Pod 가 스케줄링될 때 볼륨 생성 (AZ 맞춤)
```

`storageclass-gcp-pd.yaml` - GCP Persistent Disk 프로비저너:

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: gcp-pd-sc
provisioner: pd.csi.storage.gke.io      # GCP PD CSI 드라이버 프로비저너
parameters:
  type: pd-ssd                          # 디스크 타입 (pd-standard, pd-ssd, pd-balanced)
  fstype: ext4
reclaimPolicy: Delete
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
```

```bash
kubectl apply -f storageclass-aws-ebs.yaml
kubectl get storageclass
kubectl describe storageclass ebs-sc
```

## PVC 에서 StorageClass 지정

`pvc-dynamic.yaml`

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-dynamic-ebs
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi               # 요청하는 용량 - EBS 볼륨이 이 크기로 생성된다
  storageClassName: ebs-sc        # 위에서 만든 StorageClass 이름
```

```bash
kubectl apply -f pvc-dynamic.yaml
# PVC 상태가 Pending -> Bound 로 바뀌는 것을 확인
kubectl get pvc -w
```

## 기본 StorageClass 설정

클러스터에 기본 StorageClass 를 지정해두면 PVC 에서 `storageClassName` 을 생략했을 때 자동으로 사용된다.

```bash
# 현재 기본 StorageClass 확인 (PROVISIONER 옆에 (default) 표시)
kubectl get storageclass

# 기존 StorageClass 를 기본값으로 지정
kubectl patch storageclass ebs-sc \
  -p '{"metadata": {"annotations": {"storageclass.kubernetes.io/is-default-class": "true"}}}'

# 기본값 해제
kubectl patch storageclass ebs-sc \
  -p '{"metadata": {"annotations": {"storageclass.kubernetes.io/is-default-class": "false"}}}'
```

YAML 로 기본 StorageClass 선언:

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: ebs-sc
  annotations:
    storageclass.kubernetes.io/is-default-class: "true"   # 기본 StorageClass 로 지정
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
reclaimPolicy: Delete
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
```

---

# 실무 패턴

## StatefulSet 과 PVC 템플릿

Deployment 는 모든 Pod 가 동일한 PVC 를 공유하거나 PVC 를 사용하지 않는다. StatefulSet 은 `volumeClaimTemplates` 를 사용하여 각 Pod 에 독립적인 PVC 를 자동 생성한다. 데이터베이스처럼 각 인스턴스가 고유한 데이터를 가져야 하는 상태 있는 워크로드에 적합하다.

`statefulset-with-pvc.yaml`

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mysql
spec:
  selector:
    matchLabels:
      app: mysql
  serviceName: mysql            # Headless Service 이름 (StatefulSet 에 필수)
  replicas: 3                   # 3개의 Pod 생성
  template:
    metadata:
      labels:
        app: mysql
    spec:
      containers:
      - name: mysql
        image: mysql:8.0
        env:
        - name: MYSQL_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mysql-secret
              key: root-password
        ports:
        - containerPort: 3306
        volumeMounts:
        - name: mysql-data
          mountPath: /var/lib/mysql   # 각 Pod 가 독립적인 볼륨을 마운트한다
  volumeClaimTemplates:               # 각 Pod 에 대해 PVC 를 자동으로 생성한다
  - metadata:
      name: mysql-data                # volumeMounts 의 name 과 일치해야 한다
    spec:
      accessModes:
      - ReadWriteOnce
      storageClassName: ebs-sc
      resources:
        requests:
          storage: 10Gi
```

StatefulSet 이 생성되면 `mysql-data-mysql-0`, `mysql-data-mysql-1`, `mysql-data-mysql-2` 라는 이름의 PVC 가 자동 생성된다.

```bash
kubectl apply -f statefulset-with-pvc.yaml
kubectl get statefulset mysql
kubectl get pvc                        # 각 Pod 별 PVC 가 생성된 것을 확인
kubectl get pods -l app=mysql          # mysql-0, mysql-1, mysql-2 순서로 생성
```

StatefulSet 을 삭제해도 PVC 는 자동으로 삭제되지 않는다. 데이터 보호를 위한 의도적인 동작이다.

```bash
# StatefulSet 삭제 후에도 PVC 는 남아 있다
kubectl delete statefulset mysql
kubectl get pvc                        # PVC 가 여전히 존재한다

# PVC 를 명시적으로 삭제해야 데이터가 삭제된다
kubectl delete pvc mysql-data-mysql-0 mysql-data-mysql-1 mysql-data-mysql-2
```

## ConfigMap 변경 시 Pod 재시작 전략

ConfigMap 을 업데이트해도 환경변수로 주입한 경우에는 Pod 를 재시작해야 변경이 반영된다. 볼륨 마운트로 주입한 경우에는 자동으로 파일이 갱신되지만 애플리케이션이 파일 변경을 감지하고 재로드하는 로직이 없다면 실질적 효과가 없다.

**방법 1: Deployment 롤링 재시작 (권장)**

```bash
# ConfigMap 업데이트
kubectl edit configmap app-config
# 또는
kubectl apply -f configmap.yaml

# Deployment 롤링 재시작 - 기존 Pod 를 순차적으로 교체한다
kubectl rollout restart deployment/app-deployment

# 재시작 진행 상황 확인
kubectl rollout status deployment/app-deployment
```

**방법 2: ConfigMap 버전 관리로 자동 재시작 트리거**

ConfigMap 이름에 버전이나 해시를 포함하고 Deployment 의 어노테이션을 업데이트하면 Kubernetes 가 자동으로 롤링 업데이트를 수행한다.

`deployment-with-configmap-hash.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-deployment
spec:
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
      annotations:
        # ConfigMap 내용의 checksum 을 어노테이션으로 기록한다
        # 값이 바뀌면 Kubernetes 가 Pod 를 재시작한다
        checksum/config: "abc123def456"   # ConfigMap 내용의 sha256 해시 (CI/CD 에서 자동 계산)
    spec:
      containers:
      - name: app
        image: myapp:v1
        envFrom:
        - configMapRef:
            name: app-config
```

실무에서는 Helm 의 `tpl` 함수나 CI/CD 파이프라인에서 ConfigMap 내용의 sha256 해시를 계산하여 어노테이션을 자동으로 업데이트하는 방식을 많이 사용한다.

```bash
# ConfigMap 내용의 sha256 해시를 계산하는 예시
kubectl get configmap app-config -o yaml | sha256sum
```

**방법 3: Reloader 같은 전용 도구 사용**

[Stakater Reloader](https://github.com/stakater/Reloader) 는 ConfigMap 또는 Secret 이 변경될 때 자동으로 관련 Deployment 를 재시작해주는 Kubernetes 컨트롤러다.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-deployment
  annotations:
    reloader.stakater.com/auto: "true"  # ConfigMap 또는 Secret 변경 시 자동 재시작
spec:
  ...
```
