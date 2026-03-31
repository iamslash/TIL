# Flink 운영 가이드 — Kubernetes 환경

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
