# Kubernetes 워크로드 관리

- [Rolling Update / Rollback](#rolling-update--rollback)
- [StatefulSet](#statefulset)
- [DaemonSet](#daemonset)
- [Job / CronJob](#job--cronjob)
- [Horizontal Pod Autoscaler (HPA)](#horizontal-pod-autoscaler-hpa)
- [InitContainer](#initcontainer)
- [Sidecar 패턴](#sidecar-패턴)
- [Probe (헬스 체크)](#probe-헬스-체크)
- [Resource Requests / Limits](#resource-requests--limits)

----

## Rolling Update / Rollback

Deployment 는 파드를 업데이트할 때 두 가지 전략을 지원한다.

| 전략 | 설명 |
|---|---|
| RollingUpdate | 기존 파드를 점진적으로 교체한다. 서비스 중단 없이 업데이트된다. 기본값이다. |
| Recreate | 기존 파드를 모두 삭제한 뒤 새 파드를 생성한다. 잠깐의 다운타임이 발생한다. |

### maxSurge / maxUnavailable

RollingUpdate 전략에서 두 가지 파라미터로 업데이트 속도와 가용성을 조절한다.

- `maxSurge`: 업데이트 중 원래 replicas 수보다 최대 몇 개까지 더 생성할 수 있는지 지정한다. 숫자 또는 퍼센트로 지정한다.
- `maxUnavailable`: 업데이트 중 동시에 사용 불가능한 파드가 최대 몇 개까지 허용되는지 지정한다. 숫자 또는 퍼센트로 지정한다.

### Deployment YAML 예제

```yaml
# deployment-web.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app               # Deployment 이름
  labels:
    app: web-app
spec:
  replicas: 3                 # 유지할 파드 수
  selector:
    matchLabels:
      app: web-app
  strategy:
    type: RollingUpdate       # 롤링 업데이트 전략 사용
    rollingUpdate:
      maxSurge: 1             # 업데이트 중 최대 1개 초과 생성 허용 (3+1=4개까지)
      maxUnavailable: 0       # 업데이트 중 사용 불가 파드 허용 수 (0이면 항상 3개 유지)
  template:
    metadata:
      labels:
        app: web-app
    spec:
      containers:
      - name: web
        image: nginx:1.21     # 처음 배포 버전
        ports:
        - containerPort: 80
        resources:
          requests:
            cpu: "100m"
            memory: "128Mi"
          limits:
            cpu: "200m"
            memory: "256Mi"
```

### 이미지 버전 업데이트 시나리오

```bash
# Deployment 배포
kubectl apply -f deployment-web.yaml

# 현재 rollout 상태 확인
kubectl rollout status deployment/web-app

# 이미지를 1.21 -> 1.25 로 업데이트
kubectl set image deployment/web-app web=nginx:1.25

# 업데이트 진행 상태 확인 (완료될 때까지 대기)
kubectl rollout status deployment/web-app

# 업데이트 히스토리 확인
kubectl rollout history deployment/web-app

# 특정 리비전의 상세 내용 확인
kubectl rollout history deployment/web-app --revision=2

# 직전 버전으로 롤백
kubectl rollout undo deployment/web-app

# 특정 리비전으로 롤백
kubectl rollout undo deployment/web-app --to-revision=1

# 롤백 후 상태 확인
kubectl rollout status deployment/web-app
kubectl get pods
```

### Recreate 전략 예제

```yaml
# deployment-recreate.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: batch-processor
spec:
  replicas: 2
  selector:
    matchLabels:
      app: batch-processor
  strategy:
    type: Recreate            # 기존 파드 전체 삭제 후 새 파드 생성
  template:
    metadata:
      labels:
        app: batch-processor
    spec:
      containers:
      - name: processor
        image: myapp:2.0
```

----

## StatefulSet

StatefulSet 은 상태를 가지는 애플리케이션(데이터베이스, 메시지 큐 등)을 위한 워크로드 리소스다.

### Deployment 와의 차이

| 항목 | Deployment | StatefulSet |
|---|---|---|
| 파드 이름 | 랜덤 해시 (web-5f7b9c) | 순번 고정 (web-0, web-1) |
| 네트워크 ID | 매번 변경 | 안정적으로 유지 |
| 스토리지 | 공유 볼륨 사용 | 파드마다 독립 PVC |
| 배포 순서 | 동시에 | 0번부터 순서대로 |
| 삭제 순서 | 무작위 | 마지막 번호부터 역순 |

### Headless Service 와 StatefulSet YAML 예제

StatefulSet 은 반드시 Headless Service 와 함께 사용해야 한다. Headless Service 는 `clusterIP: None` 으로 설정하며, 개별 파드에 DNS 이름을 부여한다.

파드의 DNS 형식: `<파드이름>.<서비스이름>.<네임스페이스>.svc.cluster.local`

예) `web-0.nginx.default.svc.cluster.local`

```yaml
# nginx-statefulset.yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx                 # StatefulSet 의 serviceName 과 일치해야 함
  labels:
    app: nginx
spec:
  clusterIP: None             # Headless Service: 개별 파드에 직접 DNS 부여
  selector:
    app: nginx
  ports:
  - port: 80
    name: web
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: web                   # 파드 이름의 접두사가 됨: web-0, web-1, web-2
spec:
  serviceName: "nginx"        # 위에서 정의한 Headless Service 이름
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      terminationGracePeriodSeconds: 10   # 파드 종료 전 대기 시간(초)
      containers:
      - name: nginx
        image: nginx:1.21
        ports:
        - containerPort: 80
          name: web
        volumeMounts:
        - name: www
          mountPath: /usr/share/nginx/html  # PVC 를 이 경로에 마운트
  volumeClaimTemplates:                   # 파드마다 PVC 를 자동으로 생성
  - metadata:
      name: www
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: "standard"
      resources:
        requests:
          storage: 1Gi         # 파드당 1Gi PVC 생성
```

### 배포/삭제 동작 확인

```bash
# 배포 (web-0, web-1, web-2 순서대로 생성됨)
kubectl apply -f nginx-statefulset.yaml

# 파드 확인 - 이름이 web-0, web-1, web-2 로 고정됨
kubectl get pods -o wide

# PVC 확인 - 파드마다 www-web-0, www-web-1, www-web-2 생성됨
kubectl get pvc

# 스케일 업 (web-3, web-4 순서대로 추가)
kubectl scale statefulset web --replicas=5

# 스케일 다운 (web-4, web-3 순서대로 삭제, PVC 는 유지됨)
kubectl scale statefulset web --replicas=3

# StatefulSet 삭제 (PVC 는 삭제되지 않음)
kubectl delete -f nginx-statefulset.yaml

# PVC 는 수동으로 삭제해야 함
kubectl delete pvc --all
```

----

## DaemonSet

DaemonSet 은 클러스터의 모든 노드(또는 특정 노드)에 파드를 하나씩 실행하는 리소스다.

주요 사용 사례:
- 로그 수집 에이전트 (Fluentd, Filebeat)
- 노드 모니터링 에이전트 (Prometheus Node Exporter)
- 네트워크 플러그인 (CNI)
- 스토리지 데몬

### Fluentd 로그 수집기 DaemonSet 예제

```yaml
# fluentd-daemonset.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd                   # 모든 노드에서 실행될 로그 수집 에이전트
  namespace: kube-system
  labels:
    app: fluentd
spec:
  selector:
    matchLabels:
      app: fluentd
  template:
    metadata:
      labels:
        app: fluentd
    spec:
      # master 노드의 taint(NoSchedule) 를 허용하여 master 노드에도 배포
      tolerations:
      - key: node-role.kubernetes.io/master
        effect: NoSchedule
      - key: node-role.kubernetes.io/control-plane
        effect: NoSchedule
      # worker 노드 중에서 로그 수집이 필요한 노드만 선택 (선택 사항)
      # nodeSelector:
      #   logging: "true"
      containers:
      - name: fluentd
        image: fluent/fluentd-kubernetes-daemonset:v1-debian-elasticsearch
        env:
        - name: FLUENT_ELASTICSEARCH_HOST   # Elasticsearch 주소
          value: "elasticsearch.logging.svc.cluster.local"
        - name: FLUENT_ELASTICSEARCH_PORT
          value: "9200"
        resources:
          requests:
            cpu: "100m"
            memory: "200Mi"
          limits:
            cpu: "200m"
            memory: "400Mi"
        volumeMounts:
        - name: varlog
          mountPath: /var/log          # 노드의 로그 디렉토리 마운트
        - name: varlibdockercontainers
          mountPath: /var/lib/docker/containers
          readOnly: true
      terminationGracePeriodSeconds: 30
      volumes:
      - name: varlog
        hostPath:
          path: /var/log               # 노드의 실제 경로를 컨테이너에 마운트
      - name: varlibdockercontainers
        hostPath:
          path: /var/lib/docker/containers
```

### nodeSelector 로 특정 노드만 지정

```yaml
# 특정 레이블이 있는 노드에만 배포하는 DaemonSet
spec:
  template:
    spec:
      nodeSelector:
        disk: ssd                # disk=ssd 레이블이 있는 노드에만 배포
```

```bash
# 노드에 레이블 추가
kubectl label node worker-1 disk=ssd

# DaemonSet 확인
kubectl get daemonset fluentd -n kube-system
kubectl get pods -n kube-system -l app=fluentd -o wide
```

----

## Job / CronJob

### Job

Job 은 한 번 실행하고 완료되는 작업을 위한 리소스다. 파드가 성공적으로 종료될 때까지 재시도한다.

주요 파라미터:
- `completions`: 성공적으로 완료해야 할 파드 수 (기본값: 1)
- `parallelism`: 동시에 실행할 파드 수 (기본값: 1)
- `backoffLimit`: 실패 시 재시도 횟수 (기본값: 6)
- `activeDeadlineSeconds`: Job 의 최대 실행 시간(초)

```yaml
# db-migration-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration             # 데이터베이스 마이그레이션 Job
spec:
  completions: 1                 # 1번 성공하면 완료
  parallelism: 1                 # 한 번에 1개 파드 실행
  backoffLimit: 3                # 실패 시 최대 3번 재시도
  activeDeadlineSeconds: 300     # 5분 내에 완료되지 않으면 강제 종료
  template:
    spec:
      restartPolicy: OnFailure   # Job 에서는 Never 또는 OnFailure 만 허용
      containers:
      - name: migration
        image: myapp:1.0
        command: ["python", "manage.py", "migrate"]
        env:
        - name: DB_HOST
          value: "postgres.default.svc.cluster.local"
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-secret    # Secret 에서 비밀번호를 가져옴
              key: password
```

### 병렬 Job 예제

```yaml
# parallel-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: image-processing         # 이미지를 병렬로 처리하는 Job
spec:
  completions: 10                # 총 10개 작업 완료 필요
  parallelism: 3                 # 동시에 3개 파드 실행
  backoffLimit: 2
  template:
    spec:
      restartPolicy: Never       # 실패해도 재시작하지 않고 새 파드 생성
      containers:
      - name: worker
        image: image-processor:1.0
        command: ["python", "process.py"]
```

```bash
# Job 실행
kubectl apply -f db-migration-job.yaml

# Job 상태 확인
kubectl get jobs

# Job 로그 확인
kubectl logs job/db-migration

# 완료된 Job 삭제
kubectl delete job db-migration
```

### CronJob

CronJob 은 지정한 스케줄에 따라 Job 을 주기적으로 실행한다. Linux 의 cron 표현식을 사용한다.

cron 표현식 형식: `분 시 일 월 요일`

```yaml
# backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: db-backup                # 매일 새벽 2시에 DB 백업
spec:
  schedule: "0 2 * * *"         # 매일 02:00 에 실행 (분 시 일 월 요일)
  concurrencyPolicy: Forbid      # 이전 Job 이 실행 중이면 새 Job 을 건너뜀
  successfulJobsHistoryLimit: 3  # 성공한 Job 기록 3개 보관
  failedJobsHistoryLimit: 1      # 실패한 Job 기록 1개 보관
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: backup
            image: postgres:15
            command:
            - /bin/sh
            - -c
            - pg_dump -h $DB_HOST -U $DB_USER $DB_NAME | gzip > /backup/$(date +%Y%m%d).sql.gz
            env:
            - name: DB_HOST
              value: "postgres.default.svc.cluster.local"
            - name: DB_USER
              value: "admin"
            - name: DB_NAME
              value: "mydb"
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc   # 백업 파일을 저장할 PVC
```

| concurrencyPolicy | 설명 |
|---|---|
| Allow | 이전 Job 이 실행 중이어도 새 Job 을 시작한다 (기본값) |
| Forbid | 이전 Job 이 실행 중이면 새 Job 을 건너뛴다 |
| Replace | 이전 Job 을 중단하고 새 Job 을 시작한다 |

```bash
# CronJob 확인
kubectl get cronjobs

# 수동으로 CronJob 즉시 실행
kubectl create job --from=cronjob/db-backup manual-backup-$(date +%s)

# CronJob 일시 중지
kubectl patch cronjob db-backup -p '{"spec":{"suspend":true}}'
```

----

## Horizontal Pod Autoscaler (HPA)

HPA 는 CPU, 메모리 사용률 또는 커스텀 메트릭에 따라 Deployment 의 파드 수를 자동으로 조절한다.

HPA 가 동작하려면 `metrics-server` 가 클러스터에 설치되어 있어야 한다.

### metrics-server 설치

```bash
# metrics-server 설치
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# 설치 확인
kubectl get --raw /apis/metrics.k8s.io

# 노드/파드 리소스 사용량 확인
kubectl top nodes
kubectl top pods
```

### HPA YAML 예제 (CPU / 메모리 기반)

```yaml
# web-app-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  replicas: 2                    # 초기 파드 수
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      containers:
      - name: web
        image: nginx:1.21
        ports:
        - containerPort: 80
        resources:
          requests:
            cpu: "100m"          # HPA 가 이 값을 기준으로 퍼센트를 계산
            memory: "128Mi"
          limits:
            cpu: "200m"
            memory: "256Mi"
---
# web-app-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app                # 스케일 대상 Deployment
  minReplicas: 2                 # 최소 파드 수
  maxReplicas: 10                # 최대 파드 수
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 50   # CPU 사용률 50% 초과 시 스케일 아웃
  - type: Resource
    resource:
      name: memory
      target:
        type: AverageValue
        averageValue: 200Mi      # 파드당 평균 메모리 200Mi 초과 시 스케일 아웃
```

### desiredReplicas 계산 공식

```
desiredReplicas = ceil[ currentReplicas * (currentMetricValue / desiredMetricValue) ]
```

예) requests.cpu=100m, targetAverageUtilization=50% 인 경우:
- 기준값 = 100m * 50% = 50m
- 현재 파드 2개의 평균 CPU = 80m 이면
- desiredReplicas = ceil[ 2 * (80m / 50m) ] = ceil[3.2] = 4

### kubectl autoscale 명령어

```bash
# 명령어로 HPA 생성 (CPU 50% 기준, 최소 2개 최대 10개)
kubectl autoscale deployment web-app --cpu-percent=50 --min=2 --max=10

# HPA 상태 확인
kubectl get hpa

# HPA 상세 정보 (현재 메트릭 값 포함)
kubectl describe hpa web-app-hpa
```

### Custom Metrics 기반 HPA

CPU/메모리 외에 애플리케이션 메트릭(초당 요청 수, 큐 길이 등)으로 스케일링할 수 있다.

**Prometheus Adapter**: Prometheus 에서 수집한 메트릭을 Kubernetes Custom Metrics API 에 노출한다.

**KEDA (Kubernetes Event-Driven Autoscaling)**: 이벤트 소스(Kafka, RabbitMQ, Redis 등) 기반으로 스케일링을 지원한다. 0으로 스케일 인도 가능하다.

```yaml
# keda-scaledobject.yaml (KEDA 사용 예제)
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: kafka-scaledobject
spec:
  scaleTargetRef:
    name: consumer-deployment    # 스케일 대상 Deployment
  minReplicaCount: 0             # KEDA 는 0으로 스케일 인 가능
  maxReplicaCount: 20
  triggers:
  - type: kafka
    metadata:
      bootstrapServers: kafka.default.svc.cluster.local:9092
      consumerGroup: my-group
      topic: my-topic
      lagThreshold: "100"        # lag 이 100 초과 시 스케일 아웃
```

----

## InitContainer

InitContainer 는 메인 컨테이너가 시작되기 전에 순서대로 실행되는 컨테이너다. 초기화 작업이 완료된 후에만 메인 컨테이너가 시작된다.

주요 사용 사례:
- 의존 서비스(DB, 캐시)가 준비될 때까지 대기
- 설정 파일 다운로드 또는 생성
- 데이터베이스 마이그레이션
- 퍼미션 설정

```yaml
# web-with-init.yaml
apiVersion: v1
kind: Pod
metadata:
  name: web-with-init
spec:
  # InitContainer 목록: 순서대로 실행되며 모두 성공해야 메인 컨테이너 시작
  initContainers:
  - name: wait-for-db            # 1번째: DB 가 준비될 때까지 대기
    image: busybox:1.35
    command:
    - /bin/sh
    - -c
    - |
      until nc -z postgres.default.svc.cluster.local 5432; do
        echo "DB 연결 대기 중..."
        sleep 2
      done
      echo "DB 연결 확인 완료"
  - name: run-migration          # 2번째: DB 마이그레이션 실행
    image: myapp:1.0
    command: ["python", "manage.py", "migrate"]
    env:
    - name: DB_HOST
      value: "postgres.default.svc.cluster.local"
  - name: download-config        # 3번째: 설정 파일 다운로드
    image: busybox:1.35
    command:
    - /bin/sh
    - -c
    - wget -O /config/app.conf http://config-server/app.conf
    volumeMounts:
    - name: config-volume
      mountPath: /config          # 다운로드한 설정 파일을 볼륨에 저장

  # 메인 컨테이너: InitContainer 가 모두 완료된 후 시작
  containers:
  - name: web
    image: myapp:1.0
    ports:
    - containerPort: 8080
    volumeMounts:
    - name: config-volume
      mountPath: /config          # InitContainer 가 준비한 설정 파일 사용

  volumes:
  - name: config-volume
    emptyDir: {}                  # InitContainer 와 메인 컨테이너가 공유하는 임시 볼륨
```

```bash
# InitContainer 포함 파드 상태 확인
kubectl get pod web-with-init

# InitContainer 로그 확인
kubectl logs web-with-init -c wait-for-db
kubectl logs web-with-init -c run-migration

# 파드 상세 정보에서 InitContainer 상태 확인
kubectl describe pod web-with-init
```

----

## Sidecar 패턴

Sidecar 패턴은 메인 컨테이너와 같은 파드 안에서 보조 역할을 하는 컨테이너를 함께 실행하는 패턴이다. 두 컨테이너는 localhost 로 통신하고 볼륨을 공유할 수 있다.

주요 사용 사례:
- 로그 수집 및 전송 (Fluentd, Filebeat)
- 프록시 / 서비스 메시 (Envoy, Istio)
- 인증서 갱신 (cert-manager)
- 설정 동기화

### 로그 전송 Sidecar 예제

```yaml
# app-with-log-sidecar.yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-with-log-sidecar
spec:
  containers:
  # 메인 컨테이너: 애플리케이션 로그를 파일에 기록
  - name: app
    image: myapp:1.0
    volumeMounts:
    - name: log-volume
      mountPath: /var/log/app    # 로그 파일 저장 경로

  # Sidecar 컨테이너: 로그 파일을 읽어서 중앙 로그 시스템으로 전송
  - name: log-shipper
    image: fluent/fluentd:v1.16
    volumeMounts:
    - name: log-volume
      mountPath: /var/log/app    # 메인 컨테이너와 같은 볼륨을 공유
    env:
    - name: FLUENT_ELASTICSEARCH_HOST
      value: "elasticsearch.logging.svc.cluster.local"

  volumes:
  - name: log-volume
    emptyDir: {}                 # 메인 컨테이너와 Sidecar 가 공유하는 볼륨
```

### Envoy Proxy Sidecar 예제

```yaml
# app-with-envoy-sidecar.yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-with-envoy
  labels:
    app: myapp
spec:
  containers:
  # 메인 컨테이너: 8080 포트로 서비스
  - name: app
    image: myapp:1.0
    ports:
    - containerPort: 8080

  # Sidecar: Envoy 프록시 - 인바운드/아웃바운드 트래픽 처리
  - name: envoy-proxy
    image: envoyproxy/envoy:v1.27-latest
    ports:
    - containerPort: 9090        # Envoy 가 외부 트래픽을 수신하는 포트
    - containerPort: 9901        # Envoy 관리자 포트
    volumeMounts:
    - name: envoy-config
      mountPath: /etc/envoy

  volumes:
  - name: envoy-config
    configMap:
      name: envoy-config         # Envoy 설정 ConfigMap
```

----

## Probe (헬스 체크)

Probe 는 컨테이너의 상태를 주기적으로 확인하는 메커니즘이다.

| Probe 종류 | 역할 | 실패 시 동작 |
|---|---|---|
| livenessProbe | 컨테이너가 살아있는지 확인 | 컨테이너 재시작 |
| readinessProbe | 트래픽을 받을 준비가 됐는지 확인 | 서비스 엔드포인트에서 제거 |
| startupProbe | 컨테이너 시작이 완료됐는지 확인 | 완료 전까지 liveness/readiness 비활성화 |

### Probe 공통 설정 파라미터

| 파라미터 | 설명 | 기본값 |
|---|---|---|
| initialDelaySeconds | 컨테이너 시작 후 첫 번째 probe 까지 대기 시간 | 0 |
| periodSeconds | probe 실행 주기 | 10 |
| timeoutSeconds | probe 응답 대기 시간 | 1 |
| successThreshold | 성공으로 판단하기 위한 연속 성공 횟수 | 1 |
| failureThreshold | 실패로 판단하기 위한 연속 실패 횟수 | 3 |

### 세 가지 Probe 방식

- `httpGet`: HTTP GET 요청을 보내 200-399 응답이면 성공
- `tcpSocket`: TCP 연결이 가능하면 성공
- `exec`: 커맨드 실행 후 종료 코드가 0이면 성공

### 완전한 Probe 예제

```yaml
# app-with-probes.yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-with-probes
spec:
  containers:
  - name: web
    image: myapp:1.0
    ports:
    - containerPort: 8080

    # startupProbe: 애플리케이션 시작 완료 확인
    # failureThreshold * periodSeconds = 최대 시작 대기 시간: 30 * 10 = 300초
    startupProbe:
      httpGet:
        path: /healthz/startup   # 시작 완료 여부를 응답하는 엔드포인트
        port: 8080
      initialDelaySeconds: 5     # 5초 후 첫 번째 확인
      periodSeconds: 10          # 10초마다 확인
      failureThreshold: 30       # 30번 실패하면 컨테이너 재시작 (최대 300초 대기)

    # livenessProbe: 컨테이너 정상 동작 여부 확인
    livenessProbe:
      httpGet:
        path: /healthz/live      # 컨테이너 생존 여부를 응답하는 엔드포인트
        port: 8080
      initialDelaySeconds: 0     # startupProbe 완료 후 즉시 시작
      periodSeconds: 10          # 10초마다 확인
      timeoutSeconds: 5          # 5초 내에 응답 없으면 실패
      failureThreshold: 3        # 3번 연속 실패 시 컨테이너 재시작

    # readinessProbe: 트래픽 수신 준비 여부 확인
    readinessProbe:
      httpGet:
        path: /healthz/ready     # 트래픽 수신 준비 여부를 응답하는 엔드포인트
        port: 8080
      initialDelaySeconds: 0
      periodSeconds: 5           # 5초마다 확인 (liveness 보다 자주 확인)
      failureThreshold: 3        # 3번 연속 실패 시 서비스 엔드포인트에서 제거
```

### tcpSocket Probe 예제 (데이터베이스 등 HTTP 없는 서비스)

```yaml
# postgres-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: postgres
spec:
  containers:
  - name: postgres
    image: postgres:15
    ports:
    - containerPort: 5432
    env:
    - name: POSTGRES_PASSWORD
      valueFrom:
        secretKeyRef:
          name: postgres-secret
          key: password
    livenessProbe:
      tcpSocket:
        port: 5432               # 5432 포트로 TCP 연결 시도
      initialDelaySeconds: 30    # PostgreSQL 시작에 시간이 걸리므로 30초 대기
      periodSeconds: 10
      failureThreshold: 3
    readinessProbe:
      exec:
        command:                 # pg_isready 명령으로 DB 준비 여부 확인
        - pg_isready
        - -U
        - postgres
      initialDelaySeconds: 5
      periodSeconds: 5
```

----

## Resource Requests / Limits

### Requests 와 Limits 차이

- `requests`: 파드 스케줄링 기준. 이 만큼의 리소스가 있는 노드에만 스케줄링된다. 컨테이너에게 최소한 보장되는 리소스다.
- `limits`: 컨테이너가 사용할 수 있는 최대 리소스. 초과 시 CPU 는 스로틀링되고 메모리는 OOMKilled 된다.

### QoS (Quality of Service) 클래스

Kubernetes 는 requests/limits 설정에 따라 파드에 QoS 클래스를 자동으로 부여한다. 노드 리소스가 부족할 때 우선순위에 따라 파드를 종료한다.

| QoS 클래스 | 조건 | 우선순위 |
|---|---|---|
| Guaranteed | requests == limits (모든 컨테이너, CPU+메모리 모두) | 가장 높음 (마지막에 종료) |
| Burstable | requests < limits (또는 일부만 설정) | 중간 |
| BestEffort | requests 와 limits 모두 미설정 | 가장 낮음 (가장 먼저 종료) |

### Resource Requests / Limits YAML 예제

```yaml
# resource-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: resource-example
spec:
  containers:
  - name: app
    image: myapp:1.0
    resources:
      requests:
        cpu: "250m"              # 스케줄링 기준: 0.25 코어 이상 여유 있는 노드에 배치
        memory: "128Mi"          # 스케줄링 기준: 128Mi 이상 여유 있는 노드에 배치
      limits:
        cpu: "500m"              # 최대 0.5 코어까지 사용 가능 (초과 시 스로틀링)
        memory: "256Mi"          # 최대 256Mi 까지 사용 가능 (초과 시 OOMKilled)
```

CPU 단위: `1000m = 1 코어`, `250m = 0.25 코어`
메모리 단위: `Ki(키비바이트)`, `Mi(메비바이트)`, `Gi(기비바이트)`

### LimitRange

LimitRange 는 네임스페이스 안의 파드/컨테이너에 적용되는 기본값과 최대/최솟값을 정의한다. requests/limits 를 명시하지 않은 파드에 기본값이 자동으로 적용된다.

```yaml
# default-limitrange.yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: default-limits
  namespace: default             # 이 네임스페이스에만 적용
spec:
  limits:
  - type: Container
    default:                     # limits 미설정 시 적용되는 기본 limits
      cpu: "500m"
      memory: "256Mi"
    defaultRequest:              # requests 미설정 시 적용되는 기본 requests
      cpu: "100m"
      memory: "128Mi"
    max:                         # 컨테이너가 설정할 수 있는 최댓값
      cpu: "2"
      memory: "1Gi"
    min:                         # 컨테이너가 설정해야 하는 최솟값
      cpu: "50m"
      memory: "64Mi"
```

### ResourceQuota

ResourceQuota 는 네임스페이스 전체에서 사용할 수 있는 총 리소스 양을 제한한다. 팀별 리소스 할당에 활용한다.

```yaml
# team-resourcequota.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: team-quota
  namespace: team-a              # team-a 네임스페이스의 전체 리소스 제한
spec:
  hard:
    requests.cpu: "4"            # 네임스페이스 전체 CPU requests 합계 최대 4 코어
    requests.memory: 8Gi         # 네임스페이스 전체 메모리 requests 합계 최대 8Gi
    limits.cpu: "8"              # 네임스페이스 전체 CPU limits 합계 최대 8 코어
    limits.memory: 16Gi          # 네임스페이스 전체 메모리 limits 합계 최대 16Gi
    pods: "20"                   # 네임스페이스 내 최대 파드 수 20개
    services: "10"               # 네임스페이스 내 최대 서비스 수 10개
    persistentvolumeclaims: "5"  # 네임스페이스 내 최대 PVC 수 5개
```

```bash
# LimitRange 확인
kubectl get limitrange -n default
kubectl describe limitrange default-limits -n default

# ResourceQuota 확인
kubectl get resourcequota -n team-a
kubectl describe resourcequota team-quota -n team-a

# 네임스페이스 전체 리소스 사용량 확인
kubectl top pods -n team-a
```
