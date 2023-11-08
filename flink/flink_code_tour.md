- [Materials](#materials)
- [Build on IntelliJ](#build-on-intellij)
- [Remote JVM Debugging](#remote-jvm-debugging)
- [Job Manager REST API Handlers](#job-manager-rest-api-handlers)
- [Stand Alone Cluster Entry Point](#stand-alone-cluster-entry-point)
- [Dispatching a Job From Command Line](#dispatching-a-job-from-command-line)
- [Task Manaager Entry Point](#task-manaager-entry-point)
- [Registering Task Manager To Resource Manager](#registering-task-manager-to-resource-manager)
- [Registering Task Executor to Slot Manager](#registering-task-executor-to-slot-manager)
- [Execution Graph](#execution-graph)

---

# Materials

- [flink | github](https://github.com/apache/flink)

# Build on IntelliJ

- JDK 8
- scala 2.12.17

**Install Scala Plugin**

![](img/2023-11-02-20-46-07.png)

**Setup JDK 8**

![](img/2023-11-02-20-43-54.png)

**Setup Scala SDK**

Open Module Settings...

![](img/2023-11-03-07-58-44.png)

Platform Settings / Global Libraries

![](img/2023-11-03-08-00-46.png)

Project Settings / Libraries

![](img/2023-11-03-08-01-53.png)

Select All Modules

![](img/2023-11-03-08-02-44.png)

**Build flink-runtime**

![](img/2023-11-02-21-18-58.png)

**Run JobManager**

- [Application Mode Deployment | flink](https://nightlies.apache.org/flink/flink-docs-master/docs/deployment/resource-providers/standalone/overview/#application-mode)

- To run the Flink JobManager, you need to create a run configuration in IntelliJ:
- Go to Run > Edit Configurations.
- Click the + button to add a new configuration.
- Select Application.
- Set a name for the run configuration.
- In the Main class field, enter the following:
- `org.apache.flink.runtime.entrypoint.StandaloneSessionClusterEntrypoint`
  - ![](img/2023-11-02-20-51-35.png)
  - ![](img/2023-11-02-21-29-31.png)

- In the Program arguments field, provide the required arguments, for example:

```
-c /path/to/your/flink/conf/directory -Djobmanager.rpc.address=localhost
```

Make sure to replace `/path/to/your/flink/conf/`directory with the actual path to the Flink configuration directory you want to use.

Set the appropriate module in the Use classpath of module field.

Click OK to save the run configuration.

Now you can run the Flink JobManager by selecting the new configuration and clicking the Run button.

**Trouble Shooting**

- slf4j issue
  - ![](img/2023-11-03-08-08-42.png)

# Remote JVM Debugging

It's difficult to debug Flink JobManager on IntelliJ. but It's easy to debug
remotely Flink JobManager on IntelliJ using downloaded Flink 1.18.

**Download Flink-1.18 and decompress it**

Download flink binary from [download page](https://flink.apache.org/downloads/)

**Clone Flink repo and reset flink-1.18 tag**

![](img/2023-11-04-14-50-29.png)

**Add to Edit Configurations**

![](img/2023-11-04-14-47-57.png)

![](img/2023-11-04-14-49-40.png)

**Fix JVM Pamaters of Flink**

Add JVM Parameter
`-agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=*:5005"` on
`~/flink-1.18/bin/jobmanager.sh`.

```bash
# AsIs
ENTRYPOINT=standalonesession

if [[ $STARTSTOP == "start" ]] || [[ $STARTSTOP == "start-foreground" ]]; then
    # Add JobManager-specific JVM options
    export FLINK_ENV_JAVA_OPTS="${FLINK_ENV_JAVA_OPTS} ${FLINK_ENV_JAVA_OPTS_JM}"
    parseJmArgsAndExportLogs "${args[@]}"


# ToBe
ENTRYPOINT=standalonesession

if [[ $STARTSTOP == "start" ]] || [[ $STARTSTOP == "start-foreground" ]]; then
    # Add JobManager-specific JVM options
    export FLINK_ENV_JAVA_OPTS="${FLINK_ENV_JAVA_OPTS} ${FLINK_ENV_JAVA_OPTS_JM} -agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=*:5005"
    parseJmArgsAndExportLogs "${args[@]}"
```

![](img/2023-11-04-15-17-53.png)

**Run Flink-1.18 JobManager**

```bash
# Run Job Manager
$ ./bin/standalone-job.sh start --job-classname org.apache.flink.streaming.examples.windowing.TopSpeedWindowing
```

**Set BreakPoints**

![](img/2023-11-04-14-55-47.png)

**Attach IntelliJ Debugger to Flink-1.18 JobManager**

![](img/2023-11-04-14-54-42.png)

![](img/2023-11-04-14-56-19.png)

**Open the browser**

Open http://localhost:8081/#/job/running

![](img/2023-11-04-14-57-01.png)

**Have some fun**

![](img/2023-11-04-14-57-43.png)

# Job Manager REST API Handlers

- [REST API | flink](https://nightlies.apache.org/flink/flink-docs-master/docs/ops/rest_api/)

The core class is `org.apache.flink.runtime.webmonitor.WebMonitorEndpoint`.

# Stand Alone Cluster Entry Point

This command line will execute the `StandaloneSessionClusterEntrypoint`.

```bash
./bin/start-cluster.sh
  ./bin/jobmanager.sh start
    ./bin/flink-console.sh standalonesession
  ./bin/taskmanager.sh start  
    ./bin/flink-console.sh taskexecutor
```

`./bin/flink-console.sh` has many entry points depending on the `$SERVICE`
variable.

```bash
case $SERVICE in
    (taskexecutor)
        CLASS_TO_RUN=org.apache.flink.runtime.taskexecutor.TaskManagerRunner
    ;;

    (historyserver)
        CLASS_TO_RUN=org.apache.flink.runtime.webmonitor.history.HistoryServer
    ;;

    (zookeeper)
        CLASS_TO_RUN=org.apache.flink.runtime.zookeeper.FlinkZooKeeperQuorumPeer
    ;;

    (standalonesession)
        CLASS_TO_RUN=org.apache.flink.runtime.entrypoint.StandaloneSessionClusterEntrypoint
    ;;

    (standalonejob)
        CLASS_TO_RUN=org.apache.flink.container.entrypoint.StandaloneApplicationClusterEntryPoint
    ;;

    (kubernetes-session)
        CLASS_TO_RUN=org.apache.flink.kubernetes.entrypoint.KubernetesSessionClusterEntrypoint
    ;;

    (kubernetes-application)
        CLASS_TO_RUN=org.apache.flink.kubernetes.entrypoint.KubernetesApplicationClusterEntrypoint
    ;;

    (kubernetes-taskmanager)
        CLASS_TO_RUN=org.apache.flink.kubernetes.taskmanager.KubernetesTaskExecutorRunner
    ;;

    (sql-gateway)
        CLASS_TO_RUN=org.apache.flink.table.gateway.SqlGateway
        SQL_GATEWAY_CLASSPATH="`findSqlGatewayJar`":"`findFlinkPythonJar`"
    ;;

    (*)
        echo "Unknown service '${SERVICE}'. $USAGE."
        exit 1
    ;;
esac
```

```java
// org.apache.flink.runtime.entrypoint.StandaloneSessionClusterEntrypoint
public class StandaloneSessionClusterEntrypoint extends SessionClusterEntrypoint {
...
    public static void main(String[] args) {
        // startup checks and logging
        EnvironmentInformation.logEnvironmentInfo(
                LOG, StandaloneSessionClusterEntrypoint.class.getSimpleName(), args);
        SignalHandler.register(LOG);
        JvmShutdownSafeguard.installAsShutdownHook(LOG);

        final EntrypointClusterConfiguration entrypointClusterConfiguration =
                ClusterEntrypointUtils.parseParametersOrExit(
                        args,
                        new EntrypointClusterConfigurationParserFactory(),
                        StandaloneSessionClusterEntrypoint.class);
        Configuration configuration = loadConfiguration(entrypointClusterConfiguration);

        StandaloneSessionClusterEntrypoint entrypoint =
                new StandaloneSessionClusterEntrypoint(configuration);

        ClusterEntrypoint.runClusterEntrypoint(entrypoint);
    }

// ClusterEntrypoint.runClusterEntrypoint(entrypoint) will start dispatcher
// org.apache.flink.runtime.dispatcher.runner.SessionDispatcherLeaderProcess
public class SessionDispatcherLeaderProcess extends AbstractDispatcherLeaderProcess
...
    @Override
    protected void onStart() {
        startServices();

        onGoingRecoveryOperation =
                createDispatcherBasedOnRecoveredJobGraphsAndRecoveredDirtyJobResults();
    }

// ClusterEntrypoint.runClusterEntrypoint(entrypoint) will start resource manager
// org.apache.flink.runtime.resourcemanager.ResourceManagerServiceImpl
public class ResourceManagerServiceImpl implements ResourceManagerService, LeaderContender {
...
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

        leaderElection.startLeaderElection(this);
    }


// ClusterEntrypoint.runClusterEntrypoint(entrypoint) will start slot manager
// org.apache.flink.runtime.resourcemanager.slotmanager.FineGrainedSlotManager
public class FineGrainedSlotManager implements SlotManager {
...
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
        resourceEventListener = Preconditions.checkNotNull(newResourceEventListener);
        slotStatusSyncer.initialize(
                taskManagerTracker, resourceTracker, resourceManagerId, mainThreadExecutor);
        blockedTaskManagerChecker = Preconditions.checkNotNull(newBlockedTaskManagerChecker);

        started = true;

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
```

# Dispatching a Job From Command Line

Run this command `./bin/flink run examples/streaming/WordCount.jar`.

```java
// org/apache/flink/runtime/rpc/pekko/PekkoRpcActor.java
class PekkoRpcActor<T extends RpcEndpoint & RpcGateway> extends AbstractActor {
...
    private void handleMessage(final Object message) {
        if (state.isRunning()) {
            mainThreadValidator.enterMainThread();

            try {
                handleRpcMessage(message);
            } finally {
                mainThreadValidator.exitMainThread();
            }
        } else {
            log.info(
                    "The rpc endpoint {} has not been started yet. Discarding message {} until processing is started.",
                    rpcEndpoint.getClass().getName(),
                    message);

            sendErrorIfSender(
                    new EndpointNotStartedException(
                            String.format(
                                    "Discard message %s, because the rpc endpoint %s has not been started yet.",
                                    message, rpcEndpoint.getAddress())));
        }
    }
...

// org/apache/flink/runtime/dispatcher/Dispatcher.java
public abstract class Dispatcher extends FencedRpcEndpoint<DispatcherId>
        implements DispatcherGateway {
...
    @Override
    public CompletableFuture<Acknowledge> submitJob(JobGraph jobGraph, Time timeout) {
        final JobID jobID = jobGraph.getJobID();
        log.info("Received JobGraph submission '{}' ({}).", jobGraph.getName(), jobID);

        try {
            if (isInGloballyTerminalState(jobID)) {
                log.warn(
                        "Ignoring JobGraph submission '{}' ({}) because the job already reached a globally-terminal state (i.e. {}) in a previous execution.",
                        jobGraph.getName(),
                        jobID,
                        Arrays.stream(JobStatus.values())
                                .filter(JobStatus::isGloballyTerminalState)
                                .map(JobStatus::name)
                                .collect(Collectors.joining(", ")));
                return FutureUtils.completedExceptionally(
                        DuplicateJobSubmissionException.ofGloballyTerminated(jobID));
            } else if (jobManagerRunnerRegistry.isRegistered(jobID)
                    || submittedAndWaitingTerminationJobIDs.contains(jobID)) {
                // job with the given jobID is not terminated, yet
                return FutureUtils.completedExceptionally(
                        DuplicateJobSubmissionException.of(jobID));
            } else if (isPartialResourceConfigured(jobGraph)) {
                return FutureUtils.completedExceptionally(
                        new JobSubmissionException(
                                jobID,
                                "Currently jobs is not supported if parts of the vertices have "
                                        + "resources configured. The limitation will be removed in future versions."));
            } else {
                return internalSubmitJob(jobGraph);
            }
        } catch (FlinkException e) {
            return FutureUtils.completedExceptionally(e);
        }
    }
```

# Task Manaager Entry Point

```java
// org.apache.flink.runtime.taskexecutor.TaskManagerRunner
public class TaskManagerRunner implements FatalErrorHandler {
...
    public static void main(String[] args) throws Exception {
        // startup checks and logging
        EnvironmentInformation.logEnvironmentInfo(LOG, "TaskManager", args);
        SignalHandler.register(LOG);
        JvmShutdownSafeguard.installAsShutdownHook(LOG);

        long maxOpenFileHandles = EnvironmentInformation.getOpenFileHandlesLimit();

        if (maxOpenFileHandles != -1L) {
            LOG.info("Maximum number of open file descriptors is {}.", maxOpenFileHandles);
        } else {
            LOG.info("Cannot determine the maximum number of open file descriptors");
        }

        runTaskManagerProcessSecurely(args);
    }

...

// org.apache.flink.runtime.taskexecutor.TaskManagerRunner
public class TaskManagerRunner implements FatalErrorHandler {
...
    public void start() throws Exception {
        synchronized (lock) {
            startTaskManagerRunnerServices();
            taskExecutorService.start();
        }
    }
```

# Registering Task Manager To Resource Manager

```java
// org.apache.flink.runtime.resourcemanager.ResourceManager
public abstract class ResourceManager<WorkerType extends ResourceIDRetrievable>
        extends FencedRpcEndpoint<ResourceManagerId>
        implements DelegationTokenManager.Listener, ResourceManagerGateway {
...
    private RegistrationResponse registerTaskExecutorInternal(
            TaskExecutorGateway taskExecutorGateway,
            TaskExecutorRegistration taskExecutorRegistration) {
        ResourceID taskExecutorResourceId = taskExecutorRegistration.getResourceId();
        WorkerRegistration<WorkerType> oldRegistration =
                taskExecutors.remove(taskExecutorResourceId);
        if (oldRegistration != null) {
            // TODO :: suggest old taskExecutor to stop itself
            log.debug(
                    "Replacing old registration of TaskExecutor {}.",
                    taskExecutorResourceId.getStringWithMetadata());

            // remove old task manager registration from slot manager
            slotManager.unregisterTaskManager(
                    oldRegistration.getInstanceID(),
                    new ResourceManagerException(
                            String.format(
                                    "TaskExecutor %s re-connected to the ResourceManager.",
                                    taskExecutorResourceId.getStringWithMetadata())));
        }

        final Optional<WorkerType> newWorkerOptional =
                getWorkerNodeIfAcceptRegistration(taskExecutorResourceId);

        String taskExecutorAddress = taskExecutorRegistration.getTaskExecutorAddress();
        if (!newWorkerOptional.isPresent()) {
            log.warn(
                    "Discard registration from TaskExecutor {} at ({}) because the framework did "
                            + "not recognize it",
                    taskExecutorResourceId.getStringWithMetadata(),
                    taskExecutorAddress);
            return new TaskExecutorRegistrationRejection(
                    "The ResourceManager does not recognize this TaskExecutor.");
        } else {
            WorkerType newWorker = newWorkerOptional.get();
            WorkerRegistration<WorkerType> registration =
                    new WorkerRegistration<>(
                            taskExecutorGateway,
                            newWorker,
                            taskExecutorRegistration.getDataPort(),
                            taskExecutorRegistration.getJmxPort(),
                            taskExecutorRegistration.getHardwareDescription(),
                            taskExecutorRegistration.getMemoryConfiguration(),
                            taskExecutorRegistration.getTotalResourceProfile(),
                            taskExecutorRegistration.getDefaultSlotResourceProfile(),
                            taskExecutorRegistration.getNodeId());

            log.info(
                    "Registering TaskManager with ResourceID {} ({}) at ResourceManager",
                    taskExecutorResourceId.getStringWithMetadata(),
                    taskExecutorAddress);
            taskExecutors.put(taskExecutorResourceId, registration);

            taskManagerHeartbeatManager.monitorTarget(
                    taskExecutorResourceId, new TaskExecutorHeartbeatSender(taskExecutorGateway));

            return new TaskExecutorRegistrationSuccess(
                    registration.getInstanceID(),
                    resourceId,
                    clusterInformation,
                    latestTokens.get());
        }
    }
```

# Registering Task Executor to Slot Manager

```java
// org.apache.flink.runtime.resourcemanager.slotmanager.FineGrainedSlotManager
public class FineGrainedSlotManager implements SlotManager {
...
    @Override
    public RegistrationResult registerTaskManager(
            final TaskExecutorConnection taskExecutorConnection,
            SlotReport initialSlotReport,
            ResourceProfile totalResourceProfile,
            ResourceProfile defaultSlotResourceProfile) {
        checkInit();
        LOG.info(
                "Registering task executor {} under {} at the slot manager.",
                taskExecutorConnection.getResourceID(),
                taskExecutorConnection.getInstanceID());

        // we identify task managers by their instance id
        if (taskManagerTracker
                .getRegisteredTaskManager(taskExecutorConnection.getInstanceID())
                .isPresent()) {
            LOG.debug(
                    "Task executor {} was already registered.",
                    taskExecutorConnection.getResourceID());
            reportSlotStatus(taskExecutorConnection.getInstanceID(), initialSlotReport);
            return RegistrationResult.IGNORED;
        } else {
            Optional<PendingTaskManager> matchedPendingTaskManagerOptional =
                    initialSlotReport.hasAllocatedSlot()
                            ? Optional.empty()
                            : findMatchingPendingTaskManager(
                                    totalResourceProfile, defaultSlotResourceProfile);

            if (!matchedPendingTaskManagerOptional.isPresent()
                    && isMaxTotalResourceExceededAfterAdding(totalResourceProfile)) {

                LOG.info(
                        "Can not register task manager {}. The max total resource limitation <{}, {}> is reached.",
                        taskExecutorConnection.getResourceID(),
                        maxTotalCpu,
                        maxTotalMem.toHumanReadableString());
                return RegistrationResult.REJECTED;
            }

            taskManagerTracker.addTaskManager(
                    taskExecutorConnection, totalResourceProfile, defaultSlotResourceProfile);

            if (initialSlotReport.hasAllocatedSlot()) {
                slotStatusSyncer.reportSlotStatus(
                        taskExecutorConnection.getInstanceID(), initialSlotReport);
            }

            if (matchedPendingTaskManagerOptional.isPresent()) {
                PendingTaskManager pendingTaskManager = matchedPendingTaskManagerOptional.get();
                allocateSlotsForRegisteredPendingTaskManager(
                        pendingTaskManager, taskExecutorConnection.getInstanceID());
                taskManagerTracker.removePendingTaskManager(
                        pendingTaskManager.getPendingTaskManagerId());
                return RegistrationResult.SUCCESS;
            }

            checkResourceRequirementsWithDelay();
            return RegistrationResult.SUCCESS;
        }
    }
```

# Execution Graph

```java
//org/apache/flink/runtime/jobmaster/JobMaster.java
    @Override
    public CompletableFuture<Acknowledge> updateTaskExecutionState(
            final TaskExecutionState taskExecutionState) {
        FlinkException taskExecutionException;
        try {
            checkNotNull(taskExecutionState, "taskExecutionState");

            if (schedulerNG.updateTaskExecutionState(taskExecutionState)) {
                return CompletableFuture.completedFuture(Acknowledge.get());
            } else {
                taskExecutionException =
                        new ExecutionGraphException(
                                "The execution attempt "
                                        + taskExecutionState.getID()
                                        + " was not found.");
            }
        } catch (Exception e) {
            taskExecutionException =
                    new JobMasterException(
                            "Could not update the state of task execution for JobMaster.", e);
            handleJobMasterError(taskExecutionException);
        }
        return FutureUtils.completedExceptionally(taskExecutionException);
    }

// org/apache/flink/runtime/executiongraph/DefaultExecutionGraph.java
    @Override
    public boolean updateState(TaskExecutionStateTransition state) {
        assertRunningInJobMasterMainThread();
        final Execution attempt = currentExecutions.get(state.getID());

        if (attempt != null) {
            try {
                final boolean stateUpdated = updateStateInternal(state, attempt);
                maybeReleasePartitionGroupsFor(attempt);
                return stateUpdated;
            } catch (Throwable t) {
                ExceptionUtils.rethrowIfFatalErrorOrOOM(t);

                // failures during updates leave the ExecutionGraph inconsistent
                failGlobal(t);
                return false;
            }
        } else {
            return false;
        }
    }
```
