- [Materials](#materials)
- [Build on IntelliJ](#build-on-intellij)
- [Remote JVM Debugging](#remote-jvm-debugging)
- [Job Manager REST API Handlers](#job-manager-rest-api-handlers)
- [Task Manager Task Handlers](#task-manager-task-handlers)

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

# Task Manager Task Handlers

WIP...
