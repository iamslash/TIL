# Abstract

jvm application parameters 에 대해 정리한다.

# Materials

* [JVM Parameters InitialRAMPercentage, MinRAMPercentage, and MaxRAMPercentage](https://www.baeldung.com/java-jvm-parameters-rampercentage)

# Major Parameters

```
                -XX:+UseContainerSupport
                -XX:InitialRAMPercentage=60
                -XX:MinRAMPercentage=60
                -XX:MaxRAMPercentage=60
                -XX:MaxGCPauseMillis=200
                -XX:MaxMetaspaceSize=256m
                -XX:MetaspaceSize=256m
```

* `-XX:+UseContainerSupport`

[Improve docker container detection and resource configuration usage](https://bugs.openjdk.org/browse/JDK-8146115)

container 에 할당된 CPU, MEM 을 참고한다.

* `-XX:ActiveProcessorCount=1`

[JVM + Container 환경에서 수상한 Memory 사용량 증가 현상 분석하기](https://hyperconnect.github.io/2022/07/19/suspicious-jvm-memory-in-container.html)

Kubernetes 에서 `resources.requests.cpu=1, resources.limits.cpu=null` 를 사용한다면 thread 의 개수가 예상과 달리 32 개로 늘어날 수 있다. `availableProcessors()` 가 Worker Node 의 CPU 개수를 반환하는 것이 문제이다. 이때 `-XX:ActiveProcessorCount=1` 를 사용해야 제대로 된 thread 개수를 얻어온다.

* `-XX:+UseG1GC`
  * [JVM 메모리 구조와 GC](https://johngrib.github.io/wiki/jvm-memory/)

* `-XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=${BASEPATH}/heapdump_$(date '+%Y%m%d%H%M').hprof"`
  * JVM 이 OutOfMemoryError 로 종료되었을 때 지정된 경로로 heap dump 가 생성된다.
  * [Java Memory Analysis](https://kwonnam.pe.kr/wiki/java/memory)

* `-Xms1024m -Xmx1024m`
  * memory start size and memory max size
  * [JVM 메모리 관련 설정](https://epthffh.tistory.com/entry/JVM-%EB%A9%94%EB%AA%A8%EB%A6%AC-%EA%B4%80%EB%A0%A8-%EC%84%A4%EC%A0%95)

* `-XX:InitialRAMPercentage / -XX:MaxRAMPercentage`
  * System available memory 중 init heap percentage / max heap percentage
  * `-XX:+UseContainerSupport` 가 포함되어 있다면 System available memory 가 Container available memory 로 바뀐다.
  * `-Xms, -Xmx` 가 설정되었다면 무시된다.
  * [JVM Parameters InitialRAMPercentage, MinRAMPercentage, and MaxRAMPercentage](https://www.baeldung.com/java-jvm-parameters-rampercentage)
  * [-XX:InitialRAMPercentage / -XX:MaxRAMPercentage](https://www.eclipse.org/openj9/docs/xxinitialrampercentage/)
