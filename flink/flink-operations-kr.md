# Flink 운영 가이드 — Kubernetes 환경

## 목차

- [1. Flink Kubernetes Operator 설치](#1-flink-kubernetes-operator-설치)
- [2. 체크포인트와 세이브포인트](#2-체크포인트와-세이브포인트)
- [3. 상태 백엔드 선택](#3-상태-백엔드-선택)
- [4. 병렬성 조정](#4-병렬성parallelism-조정)
- [5. Prometheus 메트릭 모니터링](#5-prometheus-메트릭-모니터링)
- [6. 장애 복구 시나리오](#6-장애-복구-시나리오)
- [7. Go Consumer 대비 운영 비용 비교](#7-go-consumer-대비-운영-비용-비교)
- [8. Kubernetes Operator 배포 상세](#8-kubernetes-operator-배포-상세)
- [9. Prometheus 모니터링 설정](#9-prometheus-모니터링-설정)
- [10. Grafana 대시보드](#10-grafana-대시보드)
- [11. 로그 수집](#11-로그-수집)
- [12. 실무 운영 체크리스트](#12-실무-운영-체크리스트)

---

## 개요

Apache Flink를 Kubernetes에서 운영하는 방법과 Go Kafka Consumer 대비 운영 비용을 비교한다. Flink Kubernetes Operator 1.8+를 기준으로 한다.

---

## 1. Flink Kubernetes Operator 설치

### Helm으로 설치

```bash
# cert-manager 먼저 설치 (Operator webhook 의존성)
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.14.0/cert-manager.yaml
kubectl wait --for=condition=Available deployment/cert-manager -n cert-manager --timeout=120s

# Flink Kubernetes Operator Helm chart 추가
helm repo add flink-operator-repo https://downloads.apache.org/flink/flink-kubernetes-operator-1.8.0/
helm install flink-kubernetes-operator flink-operator-repo/flink-kubernetes-operator \
  --namespace flink-operator \
  --create-namespace \
  --set webhook.create=true
```

### FlinkDeployment CRD로 Flink Job 배포

```yaml
# deploy/flink/spam-swipe-detector.yaml
apiVersion: flink.apache.org/v1beta1
kind: FlinkDeployment
metadata:
  name: spam-swipe-detector
  namespace: flink-jobs
spec:
  image: registry.example.com/flink-jobs:1.0.0
  flinkVersion: v1_18
  flinkConfiguration:
    taskmanager.numberOfTaskSlots: "4"
    state.backend: rocksdb
    state.backend.incremental: "true"
    execution.checkpointing.interval: "60s"
    execution.checkpointing.min-pause: "30s"
    execution.checkpointing.timeout: "120s"
    execution.checkpointing.externalized-checkpoint-retention: RETAIN_ON_CANCELLATION
    state.checkpoints.dir: s3://my-bucket/flink/checkpoints/spam-swipe
    state.savepoints.dir: s3://my-bucket/flink/savepoints/spam-swipe
    metrics.reporters: prom
    metrics.reporter.prom.class: org.apache.flink.metrics.prometheus.PrometheusReporter
    metrics.reporter.prom.port: "9249"
  serviceAccount: flink-service-account
  jobManager:
    resource:
      memory: "2048m"
      cpu: 1
    replicas: 1 # HA 구성 시 ZooKeeper 또는 Kubernetes HA 필요
  taskManager:
    resource:
      memory: "4096m"
      cpu: 2
    replicas: 3
  job:
    jarURI: local:///opt/flink/jobs/fraud-jobs.jar
    entryClass: com.harness.fraud.SpamSwipeDetector
    args:
      - --kafka-brokers
      - kafka.kafka.svc.cluster.local:9092
      - --redis-host
      - redis.redis.svc.cluster.local
    parallelism: 6 # TaskManager 3개 × slot 2개
    upgradeMode: savepoint # 업그레이드 시 savepoint에서 재시작
    state: running
```

### Job 상태 확인

```bash
kubectl get flinkdeployment -n flink-jobs
kubectl describe flinkdeployment spam-swipe-detector -n flink-jobs

# Job Manager 로그 확인
kubectl logs deployment/spam-swipe-detector -n flink-jobs -c flink-main-container
```

---

## 2. 체크포인트와 세이브포인트

### 개념 구분

| 항목 | 체크포인트 | 세이브포인트 |
|------|-----------|-------------|
| 목적 | 장애 복구 (자동) | 계획된 업그레이드/마이그레이션 (수동) |
| 트리거 | Flink 자동 (주기적) | 사용자/오퍼레이터 수동 |
| 형식 | 내부 최적화 포맷 | 이식 가능한 표준 포맷 |
| 삭제 | 완료 후 자동 삭제 (기본) | 수동 삭제 필요 |
| 비용 | 낮음 (증분 가능) | 높음 (전체 상태 스냅샷) |

### 체크포인트 설정

```yaml
# FlinkDeployment.spec.flinkConfiguration
execution.checkpointing.interval: "60s"         # 60초마다 체크포인트
execution.checkpointing.min-pause: "30s"         # 체크포인트 간 최소 간격
execution.checkpointing.timeout: "120s"          # 체크포인트 타임아웃
execution.checkpointing.max-concurrent: "1"      # 동시 체크포인트 수
execution.checkpointing.mode: EXACTLY_ONCE       # exactly-once 보장
state.checkpoints.num-retained: "5"             # 보존할 체크포인트 수
state.checkpoints.dir: s3://bucket/checkpoints  # S3에 저장
```

### 세이브포인트 수동 생성

```bash
# FlinkDeployment CRD로 세이브포인트 트리거
kubectl patch flinkdeployment spam-swipe-detector -n flink-jobs \
  --type=merge \
  -p '{"spec":{"job":{"savepointTriggerNonce":1}}}'

# 세이브포인트 경로 확인
kubectl get flinkdeployment spam-swipe-detector -n flink-jobs \
  -o jsonpath='{.status.jobStatus.savepointInfo}'
```

### 세이브포인트에서 Job 재시작

```yaml
spec:
  job:
    upgradeMode: savepoint
    initialSavepointPath: s3://my-bucket/flink/savepoints/spam-swipe/savepoint-abc123
    allowNonRestoredState: false # 상태 불일치 시 실패 (안전)
```

---

## 3. 상태 백엔드 선택

### HashMapStateBackend (힙 기반)

```yaml
state.backend: hashmap
```

- 모든 상태를 JVM 힙에 보관
- 접근 속도: 매우 빠름 (나노초 단위)
- 상태 크기 제한: TaskManager 힙 메모리 이내
- 체크포인트: 전체 상태를 원격 저장소에 스냅샷
- **적합한 경우**: 상태 크기가 작음 (유저당 몇 KB 이내), 지연 시간이 중요한 패턴

### EmbeddedRocksDBStateBackend

```yaml
state.backend: rocksdb
state.backend.incremental: "true"   # 증분 체크포인트 (권장)
state.backend.rocksdb.memory.managed: "true"  # Flink 관리 메모리 사용
```

- 상태를 로컬 RocksDB (디스크 + 메모리 캐시)에 보관
- 접근 속도: 힙보다 느림 (마이크로초 단위)
- 상태 크기 제한: 거의 없음 (디스크 기반)
- 체크포인트: 증분 방식으로 변경분만 업로드 → 체크포인트 시간 단축
- **적합한 경우**: 대용량 상태 (유저 수백만, 긴 윈도우), Pattern 7~8의 타이머

### 선택 기준

```
상태 크기 < TaskManager 힙의 50% → HashMapStateBackend
상태 크기가 크거나 유저 수 > 100만 → EmbeddedRocksDBStateBackend
체크포인트 시간이 느리다 → incremental=true 활성화
```

---

## 4. 병렬성(Parallelism) 조정

### 기본 개념

```
총 처리 능력 = parallelism × 단일 슬롯 처리량
최대 parallelism = TaskManager 수 × taskmanager.numberOfTaskSlots
```

### 병렬성 설정 방법

```yaml
# FlinkDeployment에서 설정
spec:
  job:
    parallelism: 6  # 전체 Job 기본 병렬성
  taskManager:
    replicas: 3     # TaskManager 3개
  flinkConfiguration:
    taskmanager.numberOfTaskSlots: "2"  # 슬롯 2개 → 최대 parallelism 6
```

### 런타임 스케일링 (Reactive Mode)

```yaml
# Reactive Mode: TaskManager 수 변경 시 자동으로 parallelism 조정
spec:
  flinkConfiguration:
    scheduler-mode: reactive
```

```bash
# TaskManager 수 조정 (Reactive Mode에서 자동 반영)
kubectl scale deployment spam-swipe-detector-taskmanager -n flink-jobs --replicas=5
```

### Kafka 파티션과 병렬성 정렬

```
권장: parallelism = Kafka 파티션 수
이유: 각 Flink 서브태스크가 1개 이상의 파티션을 균등하게 담당
주의: parallelism > Kafka 파티션 수이면 일부 서브태스크가 유휴 상태
```

---

## 5. Prometheus 메트릭 모니터링

### Flink 내장 메트릭 활성화

```yaml
# FlinkDeployment.spec.flinkConfiguration
metrics.reporters: prom
metrics.reporter.prom.class: org.apache.flink.metrics.prometheus.PrometheusReporter
metrics.reporter.prom.port: "9249"
metrics.reporter.prom.interval: "10s"
```

### Prometheus ServiceMonitor (Prometheus Operator 사용 시)

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: flink-jobs
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: spam-swipe-detector
  namespaceSelector:
    matchNames:
      - flink-jobs
  endpoints:
    - port: metrics
      path: /
      interval: 15s
```

### 핵심 메트릭

| 메트릭 | 설명 | 알림 임계값 |
|--------|------|-------------|
| `flink_taskmanager_job_task_numRecordsInPerSecond` | 초당 수신 레코드 수 | 0으로 떨어지면 알림 |
| `flink_jobmanager_job_lastCheckpointDuration` | 마지막 체크포인트 소요 시간 | > 30s 알림 |
| `flink_jobmanager_job_numberOfFailedCheckpoints` | 실패한 체크포인트 수 | > 3 알림 |
| `flink_taskmanager_Status_JVM_Memory_Heap_Used` | JVM 힙 사용량 | > 80% 알림 |
| `flink_taskmanager_job_task_buffers_inputQueueLength` | 입력 버퍼 큐 길이 | > 100 알림 (백프레셔) |
| `flink_jobmanager_job_uptime` | Job 가동 시간 | 급감 시 재시작 감지 |
| `flink_taskmanager_job_task_numRecordsOut` | 출력 레코드 수 | 0이면 처리 정지 의심 |

### Grafana 대시보드 쿼리 예시

```promql
# 초당 처리 레코드 수 (전체 Job)
sum(rate(flink_taskmanager_job_task_numRecordsInPerSecond[1m]))
  by (job_name)

# 체크포인트 지연 P95
histogram_quantile(0.95,
  rate(flink_jobmanager_job_lastCheckpointDuration_bucket[5m])
)

# 백프레셔 감지 (inputQueueLength가 높으면 처리 속도보다 수신 속도가 빠름)
max(flink_taskmanager_job_task_buffers_inputQueueLength)
  by (task_name) > 50
```

---

## 6. 장애 복구 시나리오

### 시나리오 1: TaskManager 프로세스 종료

```
발생: TaskManager OOMKilled 또는 노드 장애
자동 복구:
  1. Flink JobManager가 TaskManager 실패 감지
  2. Kubernetes가 TaskManager Pod 재시작
  3. JobManager가 마지막 체크포인트에서 상태 복원
  4. Kafka 오프셋을 체크포인트 시점으로 되감기 (at-least-once / exactly-once)
소요 시간: 체크포인트 주기 + Pod 재시작 시간 (통상 30~120초)
데이터 손실: 없음 (체크포인트 이후 이벤트만 재처리)
```

### 시나리오 2: JobManager 프로세스 종료

```
발생: JobManager OOMKilled
기본 구성(단일 JM)에서:
  1. Kubernetes가 JobManager Pod 재시작
  2. 마지막 체크포인트에서 Job 재시작
HA 구성(Kubernetes HA) 필요:
  - Leader 선출을 Kubernetes ConfigMap으로 관리
  - 스탠바이 JobManager 준비 (즉시 전환)
```

HA 설정:
```yaml
flinkConfiguration:
  high-availability: kubernetes
  high-availability.storageDir: s3://bucket/ha/spam-swipe
  kubernetes.cluster-id: spam-swipe-detector
```

### 시나리오 3: Kafka 연결 중단

```
발생: Kafka 브로커 일시 중단
Flink 동작:
  1. KafkaSource가 재연결 시도 (exponential backoff)
  2. Job은 백프레셔 상태로 대기 (처리 정지)
  3. Kafka 복구 후 자동으로 마지막 커밋 오프셋부터 재처리
  4. 체크포인트 실패 시 이전 체크포인트로 롤백
설정: kafka.consumer.retry.backoff.ms = 1000 (기본값)
```

### 시나리오 4: 상태 불일치 (스키마 변경 후 복원 실패)

```
발생: Flink 상태 스키마 변경 후 이전 체크포인트로 복원 시도
증상: "Could not restore from checkpoint" 오류
해결:
  1. 새 버전의 Job을 allowNonRestoredState: true로 배포 (상태 초기화 허용)
  2. 또는 상태 마이그레이션 코드 작성 후 세이브포인트에서 복원
```

```yaml
spec:
  job:
    allowNonRestoredState: true  # 일부 상태 유실 허용 (임시 조치)
```

---

## 7. Go Consumer 대비 운영 비용 비교

### 인프라 비용 비교 (월간, AWS EKS 기준)

| 항목 | Go Consumer | Flink |
|------|-------------|-------|
| 컴퓨팅 | t3.medium × 2 (Go app) = ~$60 | m5.xlarge × 3 (TaskManager) + m5.large × 1 (JobManager) = ~$370 |
| Redis | cache.r6g.large = ~$150 | 선택적 (상태 백엔드에 따라) |
| S3 (체크포인트) | 없음 | ~$10~30/월 |
| 모니터링 | Prometheus (기존 활용) | Prometheus (기존 활용) |
| **월간 합계** | **~$210** | **~$400~430** |
| **비율** | 1× | **~2×** |

### 개발/운영 공수 비교

| 항목 | Go Consumer | Flink |
|------|-------------|-------|
| 초기 구현 (패턴 1~5) | 1~2주 | 2~3주 (Java 숙련도 필요) |
| 초기 구현 (패턴 6~8) | 3~4주 (cron + Redis 복잡도) | 1~2주 (CEP/타이머) |
| 운영 학습 곡선 | 낮음 (Redis 숙련 가정) | 높음 (Flink 아키텍처 이해 필요) |
| 디버깅 | Go 표준 도구 | Flink Web UI + 로그 분석 |
| 스키마 변경 비용 | Kafka 스키마 변경만 | 상태 스키마 마이그레이션 추가 |
| 장애 대응 속도 | 빠름 (단순 프로세스) | 느림 (체크포인트 복구 과정) |

### 처리량별 권장 구성

| 처리량 | 권장 | 이유 |
|--------|------|------|
| < 1K events/sec | Go Consumer | 충분, Flink 오버엔지니어링 |
| 1K ~ 10K events/sec | Go Consumer (Redis 모니터링 필요) | Redis 병목 주시 |
| > 10K events/sec | Flink | Redis 병목 회피, 선형 확장 |
| CEP 패턴 존재 | Flink (처리량 무관) | Go로 신뢰성 있게 구현 불가 |
| 이벤트 타임 정확도 필수 | Flink | 워터마크 기반 정확한 윈도우 |

### 운영 체크리스트

**Go Consumer 운영 시**
- [ ] Redis 메모리 사용률 모니터링 (70% 초과 시 알림)
- [ ] Kafka consumer lag 모니터링
- [ ] Redis TTL 만료로 인한 카운터 유실 허용 범위 정의
- [ ] cron 기반 패턴의 지연 허용 범위 문서화

**Flink 운영 시**
- [ ] 체크포인트 성공률 모니터링 (> 99%)
- [ ] 체크포인트 소요 시간 모니터링 (< 30초)
- [ ] JVM 힙 사용률 모니터링 (< 80%)
- [ ] 백프레셔 모니터링 (inputQueueLength < 100)
- [ ] S3 체크포인트 저장소 정기 정리
- [ ] Job 업그레이드 시 세이브포인트 생성 후 진행
- [ ] TaskManager 재시작 시 체크포인트 복원 검증

---

## 8. Kubernetes Operator 배포 상세

### Flink Kubernetes Operator 설치 (Helm 상세)

기본 설치 외에 운영 환경에서 사용할 추가 옵션을 함께 설정한다.

```bash
# Helm values 파일 생성
cat > flink-operator-values.yaml <<'EOF'
# 운영 환경 Helm values

# Webhook 활성화 (FlinkDeployment 유효성 검사)
webhook:
  create: true

# Operator 리소스 제한
operatorPod:
  resources:
    requests:
      cpu: "200m"
      memory: "512Mi"
    limits:
      cpu: "1"
      memory: "1Gi"

# 기본 Flink 설정 (모든 FlinkDeployment에 상속)
defaultConfiguration:
  create: true
  append: true
  flink-conf.yaml: |+
    # 기본 체크포인트 설정
    execution.checkpointing.interval: 60s
    execution.checkpointing.mode: EXACTLY_ONCE
    # 기본 메트릭 설정
    metrics.reporters: prom
    metrics.reporter.prom.class: org.apache.flink.metrics.prometheus.PrometheusReporter
    metrics.reporter.prom.port: "9249"
    metrics.reporter.prom.interval: "10s"

# RBAC 설정
rbac:
  create: true

# 네임스페이스 감시 범위 (비어있으면 전체 클러스터)
watchNamespaces:
  - flink-jobs
  - flink-staging
EOF

# Operator 설치
helm install flink-kubernetes-operator flink-operator-repo/flink-kubernetes-operator \
  --namespace flink-operator \
  --create-namespace \
  --values flink-operator-values.yaml \
  --version 1.8.0

# 설치 확인
kubectl get pods -n flink-operator
kubectl get crd | grep flink
```

### 완전한 FlinkDeployment YAML (Application Mode)

Application Mode는 Job별로 독립적인 JobManager와 TaskManager를 실행한다. 각 Job이 완전히 격리되므로 운영 환경에서 권장한다.

```yaml
# deploy/flink/application-mode.yaml
apiVersion: flink.apache.org/v1beta1
kind: FlinkDeployment
metadata:
  name: fraud-detector                      # Job 식별 이름
  namespace: flink-jobs                     # Job 실행 네임스페이스
  labels:
    app: fraud-detector
    team: platform
    env: production
spec:
  # Flink 이미지 (커스텀 JAR 포함)
  image: registry.example.com/flink-jobs:2.1.0
  imagePullPolicy: IfNotPresent
  flinkVersion: v1_18                       # Flink 버전 (v1_17, v1_18, v1_19)

  # Flink 설정 (flink-conf.yaml 내용)
  flinkConfiguration:
    # --- TaskManager 슬롯 ---
    taskmanager.numberOfTaskSlots: "4"      # TM 1개당 슬롯 수

    # --- 상태 백엔드 ---
    state.backend: rocksdb                  # 대용량 상태에는 RocksDB 권장
    state.backend.incremental: "true"       # 증분 체크포인트 (S3 전송량 감소)
    state.backend.rocksdb.memory.managed: "true"  # Flink managed memory 사용

    # --- 체크포인트 ---
    execution.checkpointing.interval: "60s"         # 60초마다 체크포인트
    execution.checkpointing.min-pause: "30s"         # 체크포인트 간 최소 대기
    execution.checkpointing.timeout: "300s"          # 5분 타임아웃
    execution.checkpointing.max-concurrent: "1"      # 동시 체크포인트 1개
    execution.checkpointing.mode: EXACTLY_ONCE       # 정확히 한 번 처리
    execution.checkpointing.externalized-checkpoint-retention: RETAIN_ON_CANCELLATION
    state.checkpoints.num-retained: "5"              # 최근 5개 보존
    state.checkpoints.dir: s3://my-bucket/flink/checkpoints/fraud-detector

    # --- 세이브포인트 ---
    state.savepoints.dir: s3://my-bucket/flink/savepoints/fraud-detector

    # --- 메트릭 ---
    metrics.reporters: prom
    metrics.reporter.prom.class: org.apache.flink.metrics.prometheus.PrometheusReporter
    metrics.reporter.prom.port: "9249"
    metrics.reporter.prom.interval: "10s"

    # --- HA (고가용성) ---
    high-availability: kubernetes
    high-availability.storageDir: s3://my-bucket/flink/ha/fraud-detector
    kubernetes.cluster-id: fraud-detector   # 클러스터 식별자 (Job 이름과 동일 권장)

    # --- 네트워크 버퍼 ---
    taskmanager.network.memory.fraction: "0.1"       # 네트워크 버퍼 비율
    taskmanager.network.memory.min: "64mb"
    taskmanager.network.memory.max: "1gb"

    # --- 로그 ---
    env.java.opts.taskmanager: "-XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/tmp"

  # Kubernetes ServiceAccount (S3 접근 IAM 권한 연결)
  serviceAccount: flink-service-account

  # --- JobManager 설정 ---
  jobManager:
    resource:
      memory: "2048m"                       # JM 메모리 (힙 + 오버헤드 포함)
      cpu: 1
    replicas: 1                             # HA 없이는 1, HA 사용 시 1 (리더 선출)

  # --- TaskManager 설정 ---
  taskManager:
    resource:
      memory: "4096m"                       # TM 메모리
      cpu: 2
    replicas: 3                             # TM 3개 (parallelism 12 지원)

  # --- Job 설정 ---
  job:
    jarURI: local:///opt/flink/jobs/fraud-jobs.jar  # 이미지 내 JAR 경로
    entryClass: com.example.fraud.FraudDetector     # main 클래스
    args:
      - --kafka-brokers
      - kafka.kafka.svc.cluster.local:9092
      - --input-topic
      - transactions
      - --output-topic
      - fraud-alerts
      - --redis-host
      - redis.redis.svc.cluster.local
    parallelism: 12                         # TM 3개 × slot 4개 = 최대 12
    upgradeMode: savepoint                  # 업그레이드 시 savepoint 사용
    state: running                          # 목표 상태 (running / suspended)
    savepointTriggerNonce: 0               # 세이브포인트 수동 트리거 시 증가
    allowNonRestoredState: false            # 상태 불일치 시 실패 (안전)

  # --- Pod 템플릿 (공통) ---
  podTemplate:
    metadata:
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9249"
    spec:
      securityContext:
        runAsUser: 9999                     # non-root 실행
        fsGroup: 9999
      containers:
        - name: flink-main-container
          env:
            - name: AWS_REGION
              value: ap-northeast-2
          resources: {}                     # jobManager/taskManager에서 오버라이드

  # --- Ingress (Flink Web UI 외부 노출, 선택사항) ---
  ingress:
    template: "/{{namespace}}/{{name}}(/|$)(.*)"
    className: nginx
    annotations:
      nginx.ingress.kubernetes.io/rewrite-target: "/$2"
```

### 완전한 FlinkSessionJob YAML (Session Mode)

Session Mode는 여러 Job이 하나의 Flink 클러스터를 공유한다. 개발/스테이징 환경이나 단발성 배치 Job에 적합하다.

```yaml
# deploy/flink/session-cluster.yaml
# 1단계: Session 클러스터 생성 (Job 없음)
apiVersion: flink.apache.org/v1beta1
kind: FlinkDeployment
metadata:
  name: flink-session-cluster              # Session 클러스터 이름
  namespace: flink-staging
spec:
  image: flink:1.18-scala_2.12
  flinkVersion: v1_18
  flinkConfiguration:
    taskmanager.numberOfTaskSlots: "4"
    metrics.reporters: prom
    metrics.reporter.prom.class: org.apache.flink.metrics.prometheus.PrometheusReporter
    metrics.reporter.prom.port: "9249"
  serviceAccount: flink-service-account
  jobManager:
    resource:
      memory: "1024m"
      cpu: 0.5
  taskManager:
    resource:
      memory: "2048m"
      cpu: 1
    replicas: 2
  # job 섹션 없음 = Session 클러스터
---
# 2단계: Session 클러스터에 Job 제출
apiVersion: flink.apache.org/v1beta1
kind: FlinkSessionJob
metadata:
  name: report-generator                   # Job 이름
  namespace: flink-staging
spec:
  deploymentName: flink-session-cluster    # 대상 Session 클러스터
  job:
    jarURI: https://repo.example.com/flink/report-jobs-1.0.jar  # 원격 JAR도 가능
    entryClass: com.example.report.ReportGenerator
    args:
      - --date
      - "2024-01-15"
    parallelism: 4
    upgradeMode: stateless                 # 상태 없는 배치 Job은 stateless
    state: running
```

### Job 업그레이드 전략 (Stateful Upgrade with Savepoint)

운영 중인 Job을 중단 없이 새 버전으로 교체하는 절차다.

```bash
# 1단계: 현재 상태 확인
kubectl get flinkdeployment fraud-detector -n flink-jobs -o yaml

# 2단계: 세이브포인트 수동 생성 (nonce를 1 증가)
kubectl patch flinkdeployment fraud-detector -n flink-jobs \
  --type=merge \
  -p '{"spec":{"job":{"savepointTriggerNonce":1}}}'

# 3단계: 세이브포인트 완료 대기 (lastSavepointPath 필드 확인)
kubectl get flinkdeployment fraud-detector -n flink-jobs \
  -o jsonpath='{.status.jobStatus.savepointInfo.lastSavepointPath}'
# 출력 예: s3://my-bucket/flink/savepoints/fraud-detector/savepoint-abc123-def456

# 4단계: FlinkDeployment에 새 이미지와 세이브포인트 경로 적용
kubectl patch flinkdeployment fraud-detector -n flink-jobs \
  --type=merge \
  -p '{
    "spec": {
      "image": "registry.example.com/flink-jobs:2.2.0",
      "job": {
        "initialSavepointPath": "s3://my-bucket/flink/savepoints/fraud-detector/savepoint-abc123-def456",
        "upgradeMode": "savepoint"
      }
    }
  }'

# 5단계: 업그레이드 진행 확인
kubectl rollout status deployment/fraud-detector -n flink-jobs
kubectl get flinkdeployment fraud-detector -n flink-jobs -w

# 6단계: Job 재기동 후 체크포인트 정상 동작 확인
kubectl logs deployment/fraud-detector -n flink-jobs -c flink-main-container | \
  grep -E "Completed checkpoint|checkpoint failed"
```

---

## 9. Prometheus 모니터링 설정

### flink-conf.yaml 메트릭 Reporter 설정

Flink Kubernetes Operator를 통해 배포할 때는 `flinkConfiguration` 섹션에 아래 설정을 추가한다.

```yaml
# FlinkDeployment.spec.flinkConfiguration 에 추가
flinkConfiguration:
  # Prometheus Reporter 활성화
  metrics.reporters: prom
  metrics.reporter.prom.factory.class: org.apache.flink.metrics.prometheus.PrometheusReporterFactory
  metrics.reporter.prom.port: "9249"           # 메트릭 수집 포트
  metrics.reporter.prom.interval: "10s"        # 메트릭 갱신 주기

  # 추가 메트릭 활성화 (선택)
  metrics.latency.history-size: "128"          # 지연 메트릭 히스토리 크기
  metrics.latency.interval: "0"               # 지연 메트릭 비활성화 (0 = off, 성능 영향 있음)
```

Pod에 Prometheus scrape 어노테이션을 추가한다.

```yaml
# FlinkDeployment.spec.podTemplate.metadata.annotations
annotations:
  prometheus.io/scrape: "true"    # Prometheus가 이 Pod를 수집
  prometheus.io/port: "9249"      # 수집 포트
  prometheus.io/path: "/"         # 메트릭 경로 (기본값)
```

### ServiceMonitor YAML (Prometheus Operator)

Prometheus Operator를 사용하는 환경에서 ServiceMonitor로 Flink 메트릭을 자동 수집한다.

먼저 Flink Pod에 메트릭 포트를 Service로 노출해야 한다.

```yaml
# deploy/monitoring/flink-metrics-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: fraud-detector-metrics              # 메트릭 수집용 Service
  namespace: flink-jobs
  labels:
    app: fraud-detector
    metrics: "true"                         # ServiceMonitor selector 대상
spec:
  selector:
    app: fraud-detector                     # FlinkDeployment Pod 선택
  ports:
    - name: metrics                         # ServiceMonitor에서 이름으로 참조
      port: 9249
      targetPort: 9249
      protocol: TCP
  clusterIP: None                           # Headless Service (Pod별 수집 가능)
---
# deploy/monitoring/flink-service-monitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: flink-jobs-monitor                  # ServiceMonitor 이름
  namespace: monitoring                     # Prometheus Operator 네임스페이스
  labels:
    release: prometheus                     # Prometheus Operator selector와 일치해야 함
spec:
  # 대상 Service가 있는 네임스페이스
  namespaceSelector:
    matchNames:
      - flink-jobs
      - flink-staging

  # 대상 Service 선택 조건
  selector:
    matchLabels:
      metrics: "true"                       # 위 Service의 label과 일치

  # 수집 설정
  endpoints:
    - port: metrics                         # Service의 포트 이름
      path: /                               # Flink Prometheus 엔드포인트
      interval: 15s                         # 15초마다 수집
      scrapeTimeout: 10s                    # 10초 타임아웃
      honorLabels: true                     # Flink 레이블 우선
      relabelings:
        # Pod 이름을 instance 레이블로 설정
        - sourceLabels: [__meta_kubernetes_pod_name]
          targetLabel: instance
        # Job 이름 추출
        - sourceLabels: [__meta_kubernetes_service_label_app]
          targetLabel: flink_job
```

### 주요 Prometheus Alerting Rules 5개

```yaml
# deploy/monitoring/flink-alerting-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: flink-alerting-rules
  namespace: monitoring
  labels:
    release: prometheus                     # Prometheus Operator가 인식하는 레이블
spec:
  groups:
    - name: flink.critical                  # 규칙 그룹 이름
      interval: 30s                         # 평가 주기
      rules:

        # 규칙 1: 체크포인트 연속 실패
        # 체크포인트가 3회 연속 실패하면 상태 복구 불가 상태가 될 수 있다.
        - alert: FlinkCheckpointFailure
          expr: |
            increase(
              flink_jobmanager_job_numberOfFailedCheckpoints[10m]
            ) > 3
          for: 0m                           # 즉시 알림 (체크포인트 실패는 즉각 대응 필요)
          labels:
            severity: critical
            team: platform
          annotations:
            summary: "Flink Job {{ $labels.job_name }} 체크포인트 실패"
            description: |
              Job {{ $labels.job_name }}에서 지난 10분간 체크포인트 실패가
              {{ $value }}회 발생했습니다. 상태 저장소(S3) 접근 또는
              체크포인트 타임아웃을 확인하세요.
            runbook_url: "https://wiki.example.com/flink/runbook#checkpoint-failure"

        # 규칙 2: Job 재시작 과다
        # Job이 빠르게 반복 재시작하면 처리가 중단되고 Kafka lag이 증가한다.
        - alert: FlinkJobRestartingTooOften
          expr: |
            increase(
              flink_jobmanager_job_numRestarts[30m]
            ) > 5
          for: 5m
          labels:
            severity: warning
            team: platform
          annotations:
            summary: "Flink Job {{ $labels.job_name }} 반복 재시작"
            description: |
              Job {{ $labels.job_name }}이 지난 30분간 {{ $value }}회 재시작했습니다.
              TaskManager OOMKilled 또는 코드 버그를 확인하세요.
              kubectl logs deployment/{{ $labels.job_name }} -n flink-jobs
            runbook_url: "https://wiki.example.com/flink/runbook#restart-loop"

        # 규칙 3: Back Pressure 지속
        # Back Pressure는 처리 속도보다 수신 속도가 빨라 버퍼가 가득 찬 상태다.
        - alert: FlinkBackPressureHigh
          expr: |
            max by (job_name, task_name) (
              flink_taskmanager_job_task_isBackPressured
            ) == 1
          for: 10m                          # 10분 이상 지속 시 알림
          labels:
            severity: warning
            team: platform
          annotations:
            summary: "Flink Task {{ $labels.task_name }} Back Pressure 지속"
            description: |
              Job {{ $labels.job_name }}의 Task {{ $labels.task_name }}에서
              10분 이상 Back Pressure가 감지됩니다.
              다운스트림 병목(Sink 느림, DB 부하)이나 parallelism 부족을 확인하세요.
            runbook_url: "https://wiki.example.com/flink/runbook#back-pressure"

        # 규칙 4: Kafka Consumer Lag 과다
        # Kafka lag이 높으면 실시간 처리가 지연되어 이벤트 처리 시간이 늘어난다.
        - alert: FlinkKafkaConsumerLagHigh
          expr: |
            max by (job_name, topic, partition) (
              flink_taskmanager_job_task_operator_KafkaSourceReader_KafkaConsumer_records_lag_max
            ) > 100000
          for: 15m                          # 15분 이상 lag이 높으면 알림
          labels:
            severity: warning
            team: platform
          annotations:
            summary: "Flink Kafka Consumer Lag 과다 ({{ $labels.job_name }})"
            description: |
              Job {{ $labels.job_name }}의 토픽 {{ $labels.topic }}에서
              Kafka Consumer Lag이 {{ $value }}개를 초과했습니다.
              parallelism을 늘리거나 처리 로직 성능을 확인하세요.
            runbook_url: "https://wiki.example.com/flink/runbook#kafka-lag"

        # 규칙 5: JVM 힙 메모리 고갈 위험
        # 힙 사용률이 85%를 초과하면 OOMKilled로 인한 재시작 위험이 높다.
        - alert: FlinkJvmHeapCritical
          expr: |
            (
              flink_taskmanager_Status_JVM_Memory_Heap_Used
              /
              flink_taskmanager_Status_JVM_Memory_Heap_Max
            ) * 100 > 85
          for: 5m                           # 5분 이상 지속 시 알림
          labels:
            severity: critical
            team: platform
          annotations:
            summary: "Flink TaskManager JVM 힙 사용률 위험 ({{ $labels.instance }})"
            description: |
              TaskManager {{ $labels.instance }}의 JVM 힙 사용률이
              {{ printf "%.1f" $value }}%에 도달했습니다.
              GC 로그를 확인하고 taskManager.resource.memory 증가를 검토하세요.
              kubectl top pod -n flink-jobs
            runbook_url: "https://wiki.example.com/flink/runbook#oom"
```

---

## 10. Grafana 대시보드

### 핵심 패널 10개와 PromQL 쿼리

아래 10개 패널로 Flink Job의 전반적인 건강 상태를 한눈에 파악할 수 있다.

#### 패널 1: Job Uptime (Job 가동 시간)

Job이 마지막으로 재시작한 이후 얼마나 지속 실행되고 있는지 보여준다. 값이 갑자기 0으로 초기화되면 재시작이 발생한 것이다.

```promql
# PromQL
flink_jobmanager_job_uptime / 1000

# 패널 설정
# - 시각화 타입: Stat (현재값 강조)
# - 단위: seconds
# - 임계값: 초록(> 3600), 노랑(> 60), 빨강(< 60)
```

#### 패널 2: Checkpoint Duration (체크포인트 소요 시간)

마지막 체크포인트 완료에 걸린 시간이다. 이 값이 길어지면 체크포인트 타임아웃 위험이 증가한다.

```promql
# PromQL: 마지막 체크포인트 소요 시간 (밀리초)
flink_jobmanager_job_lastCheckpointDuration

# PromQL: Job별 최대값
max by (job_name) (flink_jobmanager_job_lastCheckpointDuration)

# 패널 설정
# - 시각화 타입: Time series
# - 단위: milliseconds
# - 임계값: 초록(< 10000), 노랑(< 30000), 빨강(> 30000)
```

#### 패널 3: Checkpoint Failure Count (체크포인트 실패 횟수)

누적 체크포인트 실패 횟수다. 증가하면 상태 저장소(S3) 문제 또는 Job 불안정을 의미한다.

```promql
# PromQL: 10분간 체크포인트 실패 증가율
increase(flink_jobmanager_job_numberOfFailedCheckpoints[10m])

# 패널 설정
# - 시각화 타입: Stat
# - 단위: short (count)
# - 임계값: 초록(0), 빨강(> 0)
```

#### 패널 4: Records In/Out Per Second (초당 레코드 처리량)

Job의 실제 처리 처리량이다. 두 값의 차이가 커지면 처리 지연 발생을 의미한다.

```promql
# PromQL: 초당 수신 레코드 수 (전체 Task 합산)
sum by (job_name) (
  rate(flink_taskmanager_job_task_numRecordsIn[1m])
)

# PromQL: 초당 출력 레코드 수 (전체 Task 합산)
sum by (job_name) (
  rate(flink_taskmanager_job_task_numRecordsOut[1m])
)

# 패널 설정
# - 시각화 타입: Time series (두 쿼리를 같은 패널에)
# - 단위: rps (records per second)
```

#### 패널 5: Back Pressure Time (Back Pressure 비율)

Task가 Back Pressure 상태인 시간의 비율이다. 0%가 정상이고, 높을수록 처리 병목을 의미한다.

```promql
# PromQL: Task별 Back Pressure 비율 (0~1, 1이 100%)
max by (job_name, task_name) (
  flink_taskmanager_job_task_backPressuredTimeMsPerSecond
) / 1000

# 패널 설정
# - 시각화 타입: Time series (task_name별 legend)
# - 단위: percentunit (0.0 ~ 1.0)
# - 임계값: 초록(< 0.1), 노랑(< 0.5), 빨강(> 0.5)
```

#### 패널 6: Kafka Consumer Lag (Kafka 처리 지연)

Kafka 파티션별 미처리 메시지 수다. 지속적으로 증가하면 처리량 부족을 의미한다.

```promql
# PromQL: 파티션별 Kafka Consumer Lag 합산
sum by (job_name, topic) (
  flink_taskmanager_job_task_operator_KafkaSourceReader_KafkaConsumer_records_lag_max
)

# 패널 설정
# - 시각화 타입: Time series
# - 단위: short (records)
# - 임계값: 초록(< 1000), 노랑(< 50000), 빨강(> 100000)
```

#### 패널 7: JVM Heap / Non-Heap Usage (JVM 메모리 사용량)

TaskManager의 JVM 힙과 비힙(메타스페이스, 코드 캐시) 메모리 사용량이다.

```promql
# PromQL: 힙 사용률 (%)
(
  sum by (instance) (flink_taskmanager_Status_JVM_Memory_Heap_Used)
  /
  sum by (instance) (flink_taskmanager_Status_JVM_Memory_Heap_Max)
) * 100

# PromQL: 비힙 사용량 (bytes)
sum by (instance) (flink_taskmanager_Status_JVM_Memory_NonHeap_Used)

# 패널 설정
# - 시각화 타입: Time series
# - 단위: 힙은 percent (0~100), 비힙은 bytes
```

#### 패널 8: TaskManager 수 (운영 중인 TaskManager 수)

현재 실행 중인 TaskManager Pod의 수다. 예상 수와 다르면 Pod 장애를 의미한다.

```promql
# PromQL: 등록된 TaskManager 수
flink_jobmanager_numRegisteredTaskManagers

# 패널 설정
# - 시각화 타입: Stat
# - 단위: short (count)
# - 임계값: 예상 TM 수 기준 설정 (예: 초록(>= 3), 빨강(< 3))
```

#### 패널 9: Network Buffer Usage (네트워크 버퍼 사용률)

Task간 데이터 전송에 사용되는 네트워크 버퍼 사용률이다. 100%에 가까워지면 Back Pressure 발생 가능성이 높다.

```promql
# PromQL: 입력 버퍼 사용률 (%)
(
  sum by (job_name, task_name) (
    flink_taskmanager_job_task_buffers_inputQueueLength
  )
  /
  sum by (job_name, task_name) (
    flink_taskmanager_job_task_buffers_inputPoolUsage
  )
) * 100

# PromQL: 출력 버퍼 큐 길이 (절대값)
sum by (job_name, task_name) (
  flink_taskmanager_job_task_buffers_outputQueueLength
)

# 패널 설정
# - 시각화 타입: Time series
# - 단위: percent
```

#### 패널 10: GC Pause Time / Restart Count

GC 일시 정지 시간이 길면 처리 지연의 원인이 된다. 재시작 횟수는 Job 안정성을 나타낸다.

```promql
# PromQL: GC 일시 정지 시간 (ms/초)
rate(
  flink_taskmanager_Status_JVM_GarbageCollector_G1_Old_Generation_Time[1m]
)

# PromQL: Job 재시작 횟수 증가율 (1시간 기준)
increase(flink_jobmanager_job_numRestarts[1h])

# 패널 설정 (GC 패널)
# - 시각화 타입: Time series
# - 단위: ms (milliseconds per second)

# 패널 설정 (재시작 패널)
# - 시각화 타입: Stat
# - 임계값: 초록(0), 노랑(> 0), 빨강(> 3)
```

---

## 11. 로그 수집

### Flink 로그 구조 (log4j2)

Flink는 기본적으로 log4j2를 사용한다. 로그는 `STDOUT`과 파일(`/opt/flink/log/`) 양쪽에 출력된다.

```
Flink 로그 레벨 구조:
- ERROR: 즉각 대응이 필요한 오류 (체크포인트 실패, OOM 등)
- WARN:  잠재적 문제 (재연결 시도, 타임아웃 등)
- INFO:  정상 운영 이벤트 (체크포인트 완료, Job 시작 등)
- DEBUG: 상세 처리 흐름 (운영 환경에서는 비활성화 권장)

주요 로그 파일:
- flink-main-container: JobManager 또는 TaskManager 메인 프로세스
- stdout: Flink 운영 이벤트
- flink-*.log: 상세 로그
```

### log4j2.properties 커스터마이징

커스텀 log4j2 설정을 ConfigMap으로 배포한다.

```yaml
# deploy/flink/log4j2-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: flink-log4j2-config
  namespace: flink-jobs
data:
  log4j2.properties: |
    # 루트 로거 설정
    rootLogger.level = INFO
    rootLogger.appenderRef.console.ref = ConsoleAppender
    rootLogger.appenderRef.file.ref = FileAppender

    # 콘솔 출력 (JSON 형식 - 로그 수집 파이프라인과 호환)
    appender.console.name = ConsoleAppender
    appender.console.type = CONSOLE
    appender.console.layout.type = JsonTemplateLayout
    appender.console.layout.eventTemplateUri = classpath:EcsLayout.json

    # 파일 출력 (Fluent Bit가 수집)
    appender.file.name = FileAppender
    appender.file.type = RollingFile
    appender.file.fileName = /opt/flink/log/flink.log
    appender.file.filePattern = /opt/flink/log/flink-%d{yyyy-MM-dd}-%i.log.gz
    appender.file.layout.type = JsonTemplateLayout
    appender.file.layout.eventTemplateUri = classpath:EcsLayout.json
    appender.file.policies.type = Policies
    appender.file.policies.size.type = SizeBasedTriggeringPolicy
    appender.file.policies.size.size = 100MB
    appender.file.strategy.type = DefaultRolloverStrategy
    appender.file.strategy.max = 10

    # Flink 내부 패키지는 WARN 이상만 출력 (로그 노이즈 감소)
    logger.akka.name = akka
    logger.akka.level = WARN
    logger.kafka.name = org.apache.kafka
    logger.kafka.level = WARN
    logger.hadoop.name = org.apache.hadoop
    logger.hadoop.level = WARN
    logger.zookeeper.name = org.apache.zookeeper
    logger.zookeeper.level = WARN

    # 애플리케이션 패키지는 INFO 유지
    logger.app.name = com.example
    logger.app.level = INFO
```

ConfigMap을 FlinkDeployment의 podTemplate에 마운트한다.

```yaml
# FlinkDeployment.spec.podTemplate 추가
podTemplate:
  spec:
    volumes:
      - name: flink-log4j2-config         # ConfigMap 볼륨
        configMap:
          name: flink-log4j2-config
    containers:
      - name: flink-main-container
        volumeMounts:
          - name: flink-log4j2-config
            mountPath: /opt/flink/conf/log4j2.properties
            subPath: log4j2.properties    # ConfigMap의 특정 키만 마운트
```

### Kubernetes 환경에서 로그 수집 (Fluent Bit Sidecar)

Fluent Bit를 Sidecar로 배포하여 Flink 로그를 중앙 수집 시스템(Elasticsearch, Loki 등)으로 전송한다.

```yaml
# deploy/flink/fluent-bit-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-flink-config
  namespace: flink-jobs
data:
  fluent-bit.conf: |
    [SERVICE]
        Flush         5
        Log_Level     info
        Daemon        off
        Parsers_File  parsers.conf

    # Flink 로그 파일 읽기
    [INPUT]
        Name              tail
        Path              /opt/flink/log/*.log
        Parser            json                      # JSON 형식 로그 파싱
        Tag               flink.*
        Refresh_Interval  10
        Mem_Buf_Limit     10MB
        Skip_Long_Lines   On

    # Kubernetes 메타데이터 추가 (namespace, pod_name 등)
    [FILTER]
        Name                kubernetes
        Match               flink.*
        Kube_URL            https://kubernetes.default.svc:443
        Kube_CA_File        /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        Kube_Token_File     /var/run/secrets/kubernetes.io/serviceaccount/token
        Kube_Tag_Prefix     flink.
        Merge_Log           On
        Keep_Log            Off
        Annotations         Off
        Labels              On

    # Elasticsearch 전송 (또는 Loki, CloudWatch 등으로 변경 가능)
    [OUTPUT]
        Name            es
        Match           flink.*
        Host            elasticsearch.monitoring.svc.cluster.local
        Port            9200
        Index           flink-logs
        Type            _doc
        Suppress_Type_Name On

  parsers.conf: |
    [PARSER]
        Name        json
        Format      json
        Time_Key    @timestamp
        Time_Format %Y-%m-%dT%H:%M:%S.%LZ
---
# FlinkDeployment podTemplate에 Fluent Bit Sidecar 추가
# deploy/flink/application-mode-with-logging.yaml (일부)
#
# spec.podTemplate.spec.containers 에 아래 내용 추가:
#
#   - name: fluent-bit                      # Sidecar 컨테이너
#     image: fluent/fluent-bit:3.0
#     resources:
#       requests:
#         cpu: "50m"
#         memory: "50Mi"
#       limits:
#         cpu: "200m"
#         memory: "200Mi"
#     volumeMounts:
#       - name: flink-logs                  # Flink 로그 공유 볼륨
#         mountPath: /opt/flink/log
#       - name: fluent-bit-config
#         mountPath: /fluent-bit/etc
#
# spec.podTemplate.spec.volumes 에 아래 내용 추가:
#
#   - name: flink-logs
#     emptyDir: {}                          # Flink와 Fluent Bit가 공유
#   - name: fluent-bit-config
#     configMap:
#       name: fluent-bit-flink-config
```

---

## 12. 실무 운영 체크리스트

### 배포 전 체크리스트

배포 전에 아래 항목을 순서대로 확인한다. 체크포인트와 리소스 설정이 잘못되면 운영 중 장애로 이어진다.

**코드 및 설정 검증**
- [ ] FlinkDeployment YAML에 `serviceAccount` 필드가 설정되어 있고, 해당 ServiceAccount에 S3 접근 권한(IAM Role)이 연결되어 있다.
- [ ] `state.checkpoints.dir`과 `state.savepoints.dir`이 실제 존재하는 S3 버킷 경로를 가리킨다.
- [ ] Kafka 브로커 주소와 토픽 이름이 배포 환경(dev/staging/prod)에 맞게 설정되어 있다.
- [ ] Job의 `parallelism`이 TaskManager 수 × slot 수 이하인지 확인한다.
- [ ] 상태 스키마 변경 여부를 확인하고, 변경 시 마이그레이션 계획을 수립한다.

**리소스 설정 검증**
- [ ] TaskManager `memory` 설정이 실제 처리할 상태 크기를 수용할 수 있는지 계산한다.
  - RocksDB: 상태 크기의 2배 이상 (디스크 + 메모리 캐시)
  - HashMapStateBackend: JVM 힙의 50% 이하를 상태에 할당
- [ ] Kubernetes Node의 리소스가 충분한지 확인한다.
  - `kubectl describe nodes | grep -A 5 "Allocated resources"`
- [ ] `cpu` 설정이 너무 낮으면 체크포인트 타임아웃이 발생할 수 있다.

**체크포인트 설정 검증**
- [ ] `execution.checkpointing.timeout`이 `execution.checkpointing.interval`보다 크다.
- [ ] `state.checkpoints.num-retained`가 최소 3 이상이다 (이전 체크포인트로 롤백 가능).
- [ ] RocksDB 사용 시 `state.backend.incremental: "true"`가 설정되어 있다.

**모니터링 준비**
- [ ] Prometheus ServiceMonitor가 배포되어 있고 메트릭 수집이 동작한다.
- [ ] Grafana 대시보드가 새 Job을 인식하는지 확인한다.
- [ ] Alerting Rule이 활성화되어 있고 알림 채널(Slack, PagerDuty)과 연결되어 있다.

---

### 배포 중 체크리스트

**신규 Job 배포 (첫 배포)**

```bash
# 1. FlinkDeployment 적용
kubectl apply -f deploy/flink/application-mode.yaml

# 2. Pod 기동 확인
kubectl get pods -n flink-jobs -w

# 3. Job 상태 확인 (RUNNING이 되어야 정상)
kubectl get flinkdeployment -n flink-jobs

# 4. 첫 번째 체크포인트 완료 확인
kubectl logs deployment/fraud-detector -n flink-jobs -c flink-main-container | \
  grep "Completed checkpoint"
# 출력 예: Completed checkpoint 1 for job abc123 (12345 bytes in 5678 ms)

# 5. Kafka 처리 확인 (lag이 감소하는지)
kubectl exec -it kafka-client -n kafka -- \
  kafka-consumer-groups.sh --bootstrap-server kafka:9092 \
  --describe --group flink-fraud-detector
```

**기존 Job 업그레이드 (Stateful)**

- [ ] 업그레이드 전 수동 세이브포인트를 생성하고 경로를 기록한다.
- [ ] 세이브포인트 경로를 `initialSavepointPath`에 설정한다.
- [ ] 새 이미지 배포 후 Job이 RUNNING 상태로 복귀하는지 확인한다. (통상 2~5분)
- [ ] 업그레이드 후 첫 체크포인트가 성공하는지 확인한다.
- [ ] Kafka lag이 정상 범위로 돌아오는지 확인한다.
- [ ] 문제 발생 시 이전 이미지와 세이브포인트로 롤백한다.

```bash
# 롤백 절차
kubectl patch flinkdeployment fraud-detector -n flink-jobs \
  --type=merge \
  -p '{
    "spec": {
      "image": "registry.example.com/flink-jobs:2.1.0",
      "job": {
        "initialSavepointPath": "s3://my-bucket/flink/savepoints/fraud-detector/savepoint-이전경로"
      }
    }
  }'
```

---

### 장애 시 체크리스트

장애가 발생하면 아래 순서로 진단한다.

**1단계: 현재 상태 파악 (5분 이내)**

```bash
# Job 상태 확인
kubectl get flinkdeployment -n flink-jobs
# READY 컬럼이 false이거나 STATE가 FAILED이면 문제 있음

# Pod 상태 확인
kubectl get pods -n flink-jobs
# CrashLoopBackOff, OOMKilled, Evicted 상태 확인

# 최근 이벤트 확인
kubectl describe flinkdeployment fraud-detector -n flink-jobs | tail -30
kubectl get events -n flink-jobs --sort-by='.lastTimestamp' | tail -20
```

**2단계: 로그 확인 (5~15분)**

```bash
# JobManager 로그 (전체 Job 흐름)
kubectl logs deployment/fraud-detector -n flink-jobs -c flink-main-container \
  --since=30m | grep -E "ERROR|WARN|checkpoint|restart|exception"

# 이전 컨테이너 로그 (OOMKilled 등 재시작 전 상태)
kubectl logs deployment/fraud-detector -n flink-jobs -c flink-main-container \
  --previous | tail -100

# TaskManager 로그 (특정 Pod 지정)
kubectl logs fraud-detector-taskmanager-0 -n flink-jobs -c flink-main-container \
  --since=10m | grep -E "ERROR|exception"
```

**3단계: 메트릭 확인 (Grafana)**

- [ ] Grafana에서 장애 발생 시각 전후 메트릭을 확인한다.
- [ ] 체크포인트 실패 횟수 증가 여부를 확인한다.
- [ ] JVM 힙 사용률이 장애 직전 급증했는지 확인한다.
- [ ] Back Pressure 지속 여부를 확인한다.
- [ ] Kafka Consumer Lag 추이를 확인한다.

**4단계: 복구 절차**

| 장애 원인 | 복구 방법 |
|-----------|-----------|
| TaskManager OOMKilled | `taskManager.resource.memory` 증가 후 재배포 |
| 체크포인트 타임아웃 | S3 접근 권한 확인, `timeout` 값 증가 |
| Kafka 연결 실패 | Kafka 브로커 상태 확인, 네트워크 정책 확인 |
| 상태 복원 실패 | `allowNonRestoredState: true`로 임시 배포 후 원인 파악 |
| Back Pressure 지속 | `parallelism` 증가 또는 다운스트림 병목 해결 |
| Job 반복 재시작 | 로그에서 예외 메시지 확인, 코드 버그 수정 후 재배포 |

```bash
# 긴급 복구: 마지막 성공 체크포인트에서 재시작
# 1. 사용 가능한 체크포인트 확인
aws s3 ls s3://my-bucket/flink/checkpoints/fraud-detector/ --recursive | \
  grep "_metadata" | sort | tail -5

# 2. Job 일시 중지
kubectl patch flinkdeployment fraud-detector -n flink-jobs \
  --type=merge -p '{"spec":{"job":{"state":"suspended"}}}'

# 3. 특정 체크포인트에서 재시작
kubectl patch flinkdeployment fraud-detector -n flink-jobs \
  --type=merge \
  -p '{
    "spec": {
      "job": {
        "state": "running",
        "initialSavepointPath": "s3://my-bucket/flink/checkpoints/fraud-detector/chk-42/_metadata"
      }
    }
  }'
```
