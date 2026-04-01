# Flink 소스 코드 투어

- [개요](#개요)
- [학습 자료](#학습-자료)
- [전체 흐름 요약](#전체-흐름-요약)
- [1. 클러스터 시작](#1-클러스터-시작)
  - [쉘 스크립트 진입점](#쉘-스크립트-진입점)
  - [JobManager 시작 코드](#jobmanager-시작-코드)
  - [JobManager 가 시작하는 핵심 컴포넌트](#jobmanager-가-시작하는-핵심-컴포넌트)
- [2. TaskManager 시작](#2-taskmanager-시작)
- [3. TaskManager 등록](#3-taskmanager-등록)
  - [ResourceManager 에 등록](#resourcemanager-에-등록)
  - [SlotManager 에 Slot 등록](#slotmanager-에-slot-등록)
- [4. Job 제출](#4-job-제출)
  - [CLI 에서 Dispatcher 까지](#cli-에서-dispatcher-까지)
  - [Dispatcher 의 Job 수신](#dispatcher-의-job-수신)
- [5. 실행 그래프와 상태 관리](#5-실행-그래프와-상태-관리)
- [핵심 클래스 요약](#핵심-클래스-요약)
- [디버깅 환경 설정](#디버깅-환경-설정)
  - [Remote JVM Debugging](#remote-jvm-debugging)

---

# 개요

이 문서는 Apache Flink 소스 코드의 핵심 흐름을 따라간다. 클러스터가 시작되고, TaskManager 가 등록되고, Job 이 제출되어 실행되기까지의 과정을 **실제 코드**와 함께 설명한다.

> 대상 독자: Flink 아키텍처를 이해한 후, 내부 구현이 궁금한 개발자

# 학습 자료

- [Flink 소스 코드 | github](https://github.com/apache/flink)

---

# 전체 흐름 요약

```
[1] 클러스터 시작
    start-cluster.sh
    ├── jobmanager.sh → StandaloneSessionClusterEntrypoint.main()
    │   ├── Dispatcher 시작        (Job 수신/관리)
    │   ├── ResourceManager 시작   (TaskManager/Slot 관리)
    │   └── SlotManager 시작       (Slot 할당)
    └── taskmanager.sh → TaskManagerRunner.main()

[2] TaskManager 등록
    TaskManager → ResourceManager.registerTaskExecutor()
                → SlotManager.registerTaskManager()

[3] Job 제출
    flink run WordCount.jar
    → REST API → PekkoRpcActor.handleMessage()
    → Dispatcher.submitJob()
    → JobMaster 생성 → ExecutionGraph 구성 → Task 배포

[4] 실행 중 상태 관리
    Task 상태 변경 → JobMaster.updateTaskExecutionState()
                   → ExecutionGraph.updateState()
```

---

# 1. 클러스터 시작

## 쉘 스크립트 진입점

`start-cluster.sh` 를 실행하면 JobManager 와 TaskManager 가 차례로 시작된다.

```bash
./bin/start-cluster.sh
  ├── ./bin/jobmanager.sh start
  │     └── ./bin/flink-console.sh standalonesession
  └── ./bin/taskmanager.sh start
        └── ./bin/flink-console.sh taskexecutor
```

`flink-console.sh` 는 `$SERVICE` 변수에 따라 다른 Java 클래스를 실행한다.

```bash
case $SERVICE in
    (standalonesession)
        # JobManager (Session Mode)
        CLASS_TO_RUN=org.apache.flink.runtime.entrypoint.StandaloneSessionClusterEntrypoint
    ;;
    (standalonejob)
        # JobManager (Application Mode)
        CLASS_TO_RUN=org.apache.flink.container.entrypoint.StandaloneApplicationClusterEntryPoint
    ;;
    (taskexecutor)
        # TaskManager
        CLASS_TO_RUN=org.apache.flink.runtime.taskexecutor.TaskManagerRunner
    ;;
    (kubernetes-session)
        CLASS_TO_RUN=org.apache.flink.kubernetes.entrypoint.KubernetesSessionClusterEntrypoint
    ;;
    (kubernetes-application)
        CLASS_TO_RUN=org.apache.flink.kubernetes.entrypoint.KubernetesApplicationClusterEntrypoint
    ;;
    # ... historyserver, zookeeper, sql-gateway 등
esac
```

> 핵심: 배포 모드(Session/Application/Kubernetes)에 따라 진입점 클래스가 달라지지만, 내부 구조는 동일하다.

## JobManager 시작 코드

`StandaloneSessionClusterEntrypoint.main()` 이 JobManager 의 시작점이다.

```java
// org.apache.flink.runtime.entrypoint.StandaloneSessionClusterEntrypoint
public class StandaloneSessionClusterEntrypoint extends SessionClusterEntrypoint {

    public static void main(String[] args) {
        // 1. 환경 정보 로깅, 시그널 핸들러 등록
        EnvironmentInformation.logEnvironmentInfo(LOG, "StandaloneSessionClusterEntrypoint", args);
        SignalHandler.register(LOG);
        JvmShutdownSafeguard.installAsShutdownHook(LOG);

        // 2. 커맨드라인 인자 파싱 → Flink 설정(Configuration) 로드
        final EntrypointClusterConfiguration entrypointClusterConfiguration =
                ClusterEntrypointUtils.parseParametersOrExit(args, ...);
        Configuration configuration = loadConfiguration(entrypointClusterConfiguration);

        // 3. 엔트리포인트 생성 후 클러스터 시작
        StandaloneSessionClusterEntrypoint entrypoint =
                new StandaloneSessionClusterEntrypoint(configuration);
        ClusterEntrypoint.runClusterEntrypoint(entrypoint);
        // ↑ 이 안에서 Dispatcher, ResourceManager, SlotManager 가 모두 시작된다
    }
}
```

## JobManager 가 시작하는 핵심 컴포넌트

`ClusterEntrypoint.runClusterEntrypoint()` 가 호출되면 세 가지 핵심 컴포넌트가 시작된다.

### Dispatcher (Job 수신/관리)

외부에서 제출된 Job 을 수신하고, JobMaster 를 생성하여 실행을 위임한다.

```java
// org.apache.flink.runtime.dispatcher.runner.SessionDispatcherLeaderProcess
public class SessionDispatcherLeaderProcess extends AbstractDispatcherLeaderProcess {

    @Override
    protected void onStart() {
        startServices();
        // 이전에 실행 중이던 Job 이 있으면 복구한다 (장애 복구)
        onGoingRecoveryOperation =
                createDispatcherBasedOnRecoveredJobGraphsAndRecoveredDirtyJobResults();
    }
}
```

### ResourceManager (TaskManager/Slot 관리)

TaskManager 의 등록을 받고, 리소스(Slot)를 관리한다. Leader Election 을 통해 HA 를 지원한다.

```java
// org.apache.flink.runtime.resourcemanager.ResourceManagerServiceImpl
public class ResourceManagerServiceImpl implements ResourceManagerService, LeaderContender {

    @Override
    public void start() throws Exception {
        synchronized (lock) {
            if (running) {
                LOG.debug("Resource manager service has already started.");
                return;
            }
            running = true;
        }
        LOG.info("Starting resource manager service.");
        // Leader Election 시작 — HA 환경에서 리더가 되면 실제 ResourceManager 가 동작한다
        leaderElection.startLeaderElection(this);
    }
}
```

### SlotManager (Slot 할당)

ResourceManager 내부에서 동작하며, TaskManager 가 보고한 Slot 을 추적하고 Job 의 리소스 요청에 따라 Slot 을 할당한다.

```java
// org.apache.flink.runtime.resourcemanager.slotmanager.FineGrainedSlotManager
public class FineGrainedSlotManager implements SlotManager {

    @Override
    public void start(
            ResourceManagerId newResourceManagerId,
            Executor newMainThreadExecutor,
            ResourceAllocator newResourceAllocator,
            ResourceEventListener newResourceEventListener,
            BlockedTaskManagerChecker newBlockedTaskManagerChecker) {
        LOG.info("Starting the slot manager.");

        resourceManagerId = Preconditions.checkNotNull(newResourceManagerId);
        mainThreadExecutor = Preconditions.checkNotNull(newMainThreadExecutor);
        resourceAllocator = Preconditions.checkNotNull(newResourceAllocator);
        // ... 초기화 ...

        started = true;

        // 주기적으로 클러스터 리소스 상태를 확인하는 스케줄러 등록
        if (resourceAllocator.isSupported()) {
            clusterReconciliationCheck =
                    scheduledExecutor.scheduleWithFixedDelay(
                            () -> mainThreadExecutor.execute(this::checkClusterReconciliation),
                            0L,
                            taskManagerTimeout.toMilliseconds(),
                            TimeUnit.MILLISECONDS);
        }
        registerSlotManagerMetrics();
    }
}
```

> 정리: `runClusterEntrypoint()` 하나로 **Dispatcher + ResourceManager + SlotManager** 가 모두 시작된다. 이 세 컴포넌트가 JobManager 프로세스의 핵심이다.

---

# 2. TaskManager 시작

`TaskManagerRunner.main()` 이 TaskManager 의 시작점이다.

```java
// org.apache.flink.runtime.taskexecutor.TaskManagerRunner
public class TaskManagerRunner implements FatalErrorHandler {

    public static void main(String[] args) throws Exception {
        // 환경 로깅, 시그널 핸들러
        EnvironmentInformation.logEnvironmentInfo(LOG, "TaskManager", args);
        SignalHandler.register(LOG);
        JvmShutdownSafeguard.installAsShutdownHook(LOG);

        long maxOpenFileHandles = EnvironmentInformation.getOpenFileHandlesLimit();
        if (maxOpenFileHandles != -1L) {
            LOG.info("Maximum number of open file descriptors is {}.", maxOpenFileHandles);
        }

        // TaskManager 프로세스 시작
        runTaskManagerProcessSecurely(args);
    }

    public void start() throws Exception {
        synchronized (lock) {
            startTaskManagerRunnerServices();
            // TaskExecutor 서비스 시작 → ResourceManager 에 자신을 등록한다
            taskExecutorService.start();
        }
    }
}
```

`taskExecutorService.start()` 가 호출되면 TaskManager 는 ResourceManager 에 자신을 등록하러 간다.

---

# 3. TaskManager 등록

## ResourceManager 에 등록

TaskManager 가 시작되면 ResourceManager 에 RPC 로 등록 요청을 보낸다. ResourceManager 는 이를 수신하여 내부 맵에 등록한다.

```java
// org.apache.flink.runtime.resourcemanager.ResourceManager
public abstract class ResourceManager<WorkerType extends ResourceIDRetrievable>
        extends FencedRpcEndpoint<ResourceManagerId> {

    private RegistrationResponse registerTaskExecutorInternal(
            TaskExecutorGateway taskExecutorGateway,
            TaskExecutorRegistration taskExecutorRegistration) {

        ResourceID taskExecutorResourceId = taskExecutorRegistration.getResourceId();

        // 이미 등록된 TaskManager 가 재연결한 경우 → 기존 등록 제거
        WorkerRegistration<WorkerType> oldRegistration =
                taskExecutors.remove(taskExecutorResourceId);
        if (oldRegistration != null) {
            slotManager.unregisterTaskManager(oldRegistration.getInstanceID(), ...);
        }

        // TaskManager 를 수용할 수 있는지 확인
        final Optional<WorkerType> newWorkerOptional =
                getWorkerNodeIfAcceptRegistration(taskExecutorResourceId);

        if (!newWorkerOptional.isPresent()) {
            // 인식되지 않는 TaskManager → 거부
            return new TaskExecutorRegistrationRejection("...");
        } else {
            // 등록 성공
            WorkerRegistration<WorkerType> registration = new WorkerRegistration<>(...);

            LOG.info("Registering TaskManager with ResourceID {} at ResourceManager",
                    taskExecutorResourceId.getStringWithMetadata());

            // 내부 맵에 등록
            taskExecutors.put(taskExecutorResourceId, registration);

            // 하트비트 모니터링 시작
            taskManagerHeartbeatManager.monitorTarget(
                    taskExecutorResourceId,
                    new TaskExecutorHeartbeatSender(taskExecutorGateway));

            return new TaskExecutorRegistrationSuccess(
                    registration.getInstanceID(), resourceId, clusterInformation, ...);
        }
    }
}
```

핵심 포인트:
- `taskExecutors` 맵에 TaskManager 를 등록한다
- **하트비트 모니터링**을 시작한다 (끊기면 TaskManager 연결 끊김 처리)
- 이미 등록된 TaskManager 가 재연결하면 기존 등록을 교체한다

## SlotManager 에 Slot 등록

ResourceManager 등록이 완료되면, TaskManager 는 자신이 보유한 Slot 정보를 SlotManager 에 보고한다.

```java
// org.apache.flink.runtime.resourcemanager.slotmanager.FineGrainedSlotManager
public class FineGrainedSlotManager implements SlotManager {

    @Override
    public RegistrationResult registerTaskManager(
            final TaskExecutorConnection taskExecutorConnection,
            SlotReport initialSlotReport,
            ResourceProfile totalResourceProfile,
            ResourceProfile defaultSlotResourceProfile) {

        checkInit();

        // 이미 등록된 TaskManager 인지 확인
        if (taskManagerTracker
                .getRegisteredTaskManager(taskExecutorConnection.getInstanceID())
                .isPresent()) {
            // 이미 등록됨 → 무시
            reportSlotStatus(taskExecutorConnection.getInstanceID(), initialSlotReport);
            return RegistrationResult.IGNORED;
        }

        // 클러스터 리소스 상한 초과 확인
        if (isMaxTotalResourceExceededAfterAdding(totalResourceProfile)) {
            LOG.info("Can not register task manager {}. Max total resource limitation reached.",
                    taskExecutorConnection.getResourceID());
            return RegistrationResult.REJECTED;
        }

        // TaskManager 를 트래커에 추가
        taskManagerTracker.addTaskManager(
                taskExecutorConnection, totalResourceProfile, defaultSlotResourceProfile);

        // 대기 중인 Slot 요청이 있으면 즉시 할당
        // (예: Job 이 먼저 제출되었는데 TaskManager 가 아직 없었던 경우)
        if (matchedPendingTaskManagerOptional.isPresent()) {
            allocateSlotsForRegisteredPendingTaskManager(...);
            return RegistrationResult.SUCCESS;
        }

        checkResourceRequirementsWithDelay();
        return RegistrationResult.SUCCESS;
    }
}
```

> 이 과정이 완료되면 **Slot 이 사용 가능 상태**가 된다. 이제 Job 이 제출되면 이 Slot 에 Task 를 배치할 수 있다.

---

# 4. Job 제출

## CLI 에서 Dispatcher 까지

```bash
./bin/flink run examples/streaming/WordCount.jar
```

이 명령어는 REST API 를 통해 JobGraph 를 JobManager 에 전송한다. JobManager 내부에서는 Pekko(구 Akka) RPC 프레임워크가 메시지를 라우팅한다.

```java
// org.apache.flink.runtime.rpc.pekko.PekkoRpcActor
// 모든 RPC 메시지가 이 메서드를 통과한다
class PekkoRpcActor<T extends RpcEndpoint & RpcGateway> extends AbstractActor {

    private void handleMessage(final Object message) {
        if (state.isRunning()) {
            mainThreadValidator.enterMainThread();
            try {
                handleRpcMessage(message);
                // ↑ 여기서 Dispatcher.submitJob() 이 호출된다
            } finally {
                mainThreadValidator.exitMainThread();
            }
        } else {
            // 아직 시작되지 않은 엔드포인트 → 메시지 버림
            sendErrorIfSender(new EndpointNotStartedException(...));
        }
    }
}
```

> `mainThreadValidator` 에 주목하자. Flink 의 RPC 엔드포인트는 **싱글 스레드 모델**이다. 모든 메시지가 하나의 메인 스레드에서 순차 처리되므로 동기화 문제가 없다.

## Dispatcher 의 Job 수신

```java
// org.apache.flink.runtime.dispatcher.Dispatcher
public abstract class Dispatcher extends FencedRpcEndpoint<DispatcherId>
        implements DispatcherGateway {

    @Override
    public CompletableFuture<Acknowledge> submitJob(JobGraph jobGraph, Time timeout) {
        final JobID jobID = jobGraph.getJobID();
        log.info("Received JobGraph submission '{}' ({}).", jobGraph.getName(), jobID);

        try {
            // 이미 완료된 Job 인가?
            if (isInGloballyTerminalState(jobID)) {
                return FutureUtils.completedExceptionally(
                        DuplicateJobSubmissionException.ofGloballyTerminated(jobID));
            }
            // 이미 실행 중인 Job 인가?
            else if (jobManagerRunnerRegistry.isRegistered(jobID)) {
                return FutureUtils.completedExceptionally(
                        DuplicateJobSubmissionException.of(jobID));
            }
            // 정상 제출
            else {
                return internalSubmitJob(jobGraph);
                // ↑ JobMaster 를 생성하고, ExecutionGraph 를 구성하고,
                //   SlotManager 에 Slot 을 요청하고, Task 를 배포한다
            }
        } catch (FlinkException e) {
            return FutureUtils.completedExceptionally(e);
        }
    }
}
```

`internalSubmitJob()` 이후의 흐름:
1. **JobMaster 생성** — Job 하나당 JobMaster 하나
2. **ExecutionGraph 구성** — 논리적 JobGraph → 물리적 ExecutionGraph 로 변환
3. **Slot 요청** — SlotManager 에 필요한 Slot 수만큼 요청
4. **Task 배포** — 할당받은 Slot 의 TaskManager 에 Task 를 전송

---

# 5. 실행 그래프와 상태 관리

Task 가 실행되는 동안 상태가 변경되면 (RUNNING → FINISHED, FAILED 등) TaskManager 가 JobMaster 에 보고한다.

```java
// org.apache.flink.runtime.jobmaster.JobMaster
@Override
public CompletableFuture<Acknowledge> updateTaskExecutionState(
        final TaskExecutionState taskExecutionState) {
    try {
        checkNotNull(taskExecutionState);
        // ExecutionGraph 에 상태 변경을 전달
        if (schedulerNG.updateTaskExecutionState(taskExecutionState)) {
            return CompletableFuture.completedFuture(Acknowledge.get());
        } else {
            // 해당 실행 시도를 찾을 수 없음
            return FutureUtils.completedExceptionally(
                    new ExecutionGraphException("Execution attempt not found."));
        }
    } catch (Exception e) {
        handleJobMasterError(e);
        return FutureUtils.completedExceptionally(e);
    }
}
```

```java
// org.apache.flink.runtime.executiongraph.DefaultExecutionGraph
@Override
public boolean updateState(TaskExecutionStateTransition state) {
    assertRunningInJobMasterMainThread();
    // 현재 실행 중인 시도(Attempt) 를 찾는다
    final Execution attempt = currentExecutions.get(state.getID());

    if (attempt != null) {
        try {
            // 상태 업데이트 수행
            final boolean stateUpdated = updateStateInternal(state, attempt);
            // 완료된 Task 의 파티션 해제
            maybeReleasePartitionGroupsFor(attempt);
            return stateUpdated;
        } catch (Throwable t) {
            ExceptionUtils.rethrowIfFatalErrorOrOOM(t);
            // 상태 업데이트 중 오류 → 전체 Job 실패 처리
            failGlobal(t);
            return false;
        }
    }
    return false;
}
```

> `failGlobal(t)` 에 주목하자. 상태 업데이트 중 예외가 발생하면 ExecutionGraph 가 비일관 상태가 될 수 있으므로 **Job 전체를 실패 처리**한다. 이후 재시작 전략에 따라 체크포인트에서 복구된다.

---

# 핵심 클래스 요약

| 클래스 | 패키지 | 역할 |
|--------|--------|------|
| `StandaloneSessionClusterEntrypoint` | `runtime.entrypoint` | JobManager 진입점 (Session Mode) |
| `SessionDispatcherLeaderProcess` | `runtime.dispatcher.runner` | Dispatcher 시작 및 장애 복구 |
| `Dispatcher` | `runtime.dispatcher` | Job 수신, JobMaster 생성 |
| `ResourceManagerServiceImpl` | `runtime.resourcemanager` | ResourceManager 시작, Leader Election |
| `FineGrainedSlotManager` | `runtime.resourcemanager.slotmanager` | Slot 등록/할당/해제 관리 |
| `TaskManagerRunner` | `runtime.taskexecutor` | TaskManager 진입점 |
| `ResourceManager` | `runtime.resourcemanager` | TaskManager 등록, 하트비트 관리 |
| `PekkoRpcActor` | `runtime.rpc.pekko` | RPC 메시지 라우팅 (싱글 스레드) |
| `JobMaster` | `runtime.jobmaster` | 개별 Job 의 실행 관리 |
| `DefaultExecutionGraph` | `runtime.executiongraph` | 물리적 실행 계획, Task 상태 관리 |

---

# 디버깅 환경 설정

## Remote JVM Debugging

Flink 소스 코드를 직접 디버깅하려면 Remote JVM Debugging 이 가장 편하다.

**1단계: Flink 바이너리 다운로드**

[Flink 다운로드 페이지](https://flink.apache.org/downloads/) 에서 바이너리를 받아 압축을 푼다.

**2단계: JVM 디버그 옵션 추가**

`~/flink-1.18/bin/jobmanager.sh` 에 디버그 에이전트를 추가한다.

```bash
# 수정 전
export FLINK_ENV_JAVA_OPTS="${FLINK_ENV_JAVA_OPTS} ${FLINK_ENV_JAVA_OPTS_JM}"

# 수정 후 — 5005 포트로 디버거 연결을 기다린다
export FLINK_ENV_JAVA_OPTS="${FLINK_ENV_JAVA_OPTS} ${FLINK_ENV_JAVA_OPTS_JM} -agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=*:5005"
```

**3단계: Flink JobManager 실행**

```bash
./bin/standalone-job.sh start \
  --job-classname org.apache.flink.streaming.examples.windowing.TopSpeedWindowing
```

**4단계: IntelliJ 에서 Remote Debug 연결**

1. Flink 소스를 clone 하고 동일 버전 태그로 checkout
2. Run > Edit Configurations > + > Remote JVM Debug
3. Host: `localhost`, Port: `5005`
4. 원하는 클래스에 브레이크포인트 설정 후 Debug 실행

**5단계: 브라우저에서 확인**

http://localhost:8081 에서 Flink Web UI 를 열고, 실행 중인 Job 을 확인하면서 디버깅한다.
