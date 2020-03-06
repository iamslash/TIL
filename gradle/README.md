# Abstract

java진영의 build tool인 gradle에 대해 정리한다. gradle은 maven보다
성능이 좋다.

# Materials

* [gradle DSL reference](https://docs.gradle.org/current/dsl/)
* [Creating New Gradle Builds](https://guides.gradle.org/creating-new-gradle-builds/)

# Basic

* [2. Gradle의 기본 구조 살펴보기](https://gmind.tistory.com/entry/Gradle%EC%9D%98-%EA%B8%B0%EB%B3%B8-%EA%B8%B0%EB%8A%A5-%EB%A7%9B%EB%B3%B4%EA%B8%B0?category=655027)

```bash
$ gradle version
$ cd my-app

# Generates gradlew, gradlew.bat, .gradle, gradle
# Those are for system Gradle is not installed on. 
$ gradle wrapper
$ tree -a .
.
├── .gradle
│   ├── 6.1.1
│   │   ├── executionHistory
│   │   │   ├── executionHistory.bin
│   │   │   └── executionHistory.lock
│   │   ├── fileChanges
│   │   │   └── last-build.bin
│   │   ├── fileHashes
│   │   │   ├── fileHashes.bin
│   │   │   └── fileHashes.lock
│   │   ├── gc.properties
│   │   └── vcsMetadata-1
│   ├── buildOutputCleanup
│   │   ├── buildOutputCleanup.lock
│   │   ├── cache.properties
│   │   └── outputFiles.bin
│   ├── checksums
│   │   └── checksums.lock
│   └── vcs-1
│       └── gc.properties
├── gradle
│   └── wrapper
│       ├── gradle-wrapper.jar
│       └── gradle-wrapper.properties
├── gradlew
└── gradlew.bat

# Generates build.gradle, settings.gradle, etc... interactively.
$ gradle init

Select type of project to generate:
  1: basic
  2: application
  3: library
  4: Gradle plugin
Enter selection (default: basic) [1..4] 2

Select implementation language:
  1: C++
  2: Groovy
  3: Java
  4: Kotlin
  5: Swift
Enter selection (default: Java) [1..5] 3

Select build script DSL:
  1: Groovy
  2: Kotlin
Enter selection (default: Groovy) [1..2] 1

Select test framework:
  1: JUnit 4
  2: TestNG
  3: Spock
  4: JUnit Jupiter
Enter selection (default: JUnit 4) [1..4] 4

Project name (default: my-app):
Source package (default: my.app):

> Task :init
Get more help with your project: https://docs.gradle.org/6.1.1/userguide/tutorial_java_projects.html

BUILD SUCCESSFUL in 30s
2 actionable tasks: 1 executed, 1 up-to-date

$ tree -a .
.
├── .gitattributes
├── .gitignore
├── .gradle
│   ├── 6.1.1
│   │   ├── executionHistory
│   │   │   ├── executionHistory.bin
│   │   │   └── executionHistory.lock
│   │   ├── fileChanges
│   │   │   └── last-build.bin
│   │   ├── fileHashes
│   │   │   ├── fileHashes.bin
│   │   │   └── fileHashes.lock
│   │   ├── gc.properties
│   │   └── vcsMetadata-1
│   ├── buildOutputCleanup
│   │   ├── buildOutputCleanup.lock
│   │   ├── cache.properties
│   │   └── outputFiles.bin
│   ├── checksums
│   │   └── checksums.lock
│   └── vcs-1
│       └── gc.properties
├── build.gradle
├── gradle
│   └── wrapper
│       ├── gradle-wrapper.jar
│       └── gradle-wrapper.properties
├── gradlew
├── gradlew.bat
├── settings.gradle
└── src
    ├── main
    │   ├── java
    │   │   └── my
    │   │       └── app
    │   │           └── App.java
    │   └── resources
    └── test
        ├── java
        │   └── my
        │       └── app
        │           └── AppTest.java
        └── resources
```

다음은 생성된 `settings.gralde` 이다. 

```groovy
rootProject.name = 'my-app'
```

include 를 사용하여 sub-project 를 설정할 수 있다.

```groovy
rootProject.name = 'my-app'
include 'backend'
include 'frontend'
```

다음은 생성된 `build.graddle` 이다. 

```groovy
plugins {
    // Apply the java plugin to add support for Java
    id 'java'

    // Apply the application plugin to add support for building a CLI application.
    id 'application'
}

repositories {
    // Use jcenter for resolving dependencies.
    // You can declare any Maven/Ivy/file repository here.
    jcenter()
}

dependencies {
    // This dependency is used by the application.
    implementation 'com.google.guava:guava:28.1-jre'

    // Use JUnit Jupiter API for testing.
    testImplementation 'org.junit.jupiter:junit-jupiter-api:5.5.2'

    // Use JUnit Jupiter Engine for testing.
    testRuntimeOnly 'org.junit.jupiter:junit-jupiter-engine:5.5.2'
}

application {
    // Define the main class for the application.
    mainClassName = 'my.app.App'
}

test {
    // Use junit platform for unit tests
    useJUnitPlatform()
}
```

`build.gradle` 의 내용은 단순하다. 그러나 Gradle 은 기본적으로 다음과 같은 task graph 를 생성해 준다.

![](img/javaPluginTasks.png)

다음과 같이 `gradle build` 를 실행하면 수행되는 task 들을 알 수 있다.

```bash
$ bradle build

```

다음과 같이 새로운 task 를 기존의 task graph 에 삽입할 수 있다. 

* [Gradle-workshop](http://azquelt.github.io/gradle-workshop/)

![](img/tasks1.svg)

![](img/tasks2.svg)

```groovy
task integrationTest(type: Test) {
	dependsOn integrationTestClasses
	dependsOn installDist

	testClassesDir = sourceSets.integrationTest.output.classesDir
	classpath = sourceSets.integrationTest.runtimeClasspath
}
```


# Tutorial

## Hello World

```bash
$ make Hello
$ cd Hello
$ gradle init

├── build.gradle  
├── gradle
│   └── wrapper
│       ├── gradle-wrapper.jar  
│       └── gradle-wrapper.properties  
├── gradlew  
├── gradlew.bat  
└── settings.gradle  

# build.gradle : Gradle build script for configuring the current project
# gradle-wrapper.java : Gradle Wrapper executable JAR
# gradle-wrapper.properties : Gradle Wrapper configuration properties
# gradlew : Gradle Wrapper script for Unix-based systems
# gradlew.bat : Gradle Wrapper script for Windows
# settings.gradle : Gradle settings script for configuring the Gradle build
```

* create a task

```groovy
task copy(type: Copy, group: "Custom", description: "Copies sources to the dest directory") {
    from "src"
    into "dest"
}
```

```bash
$ gradle copy
```

* apply a plugin

```groovy
plugins {
    id "base"
}
...
task zip(type: Zip, group: "Archive", description: "Archives sources in a zip file") {
    from "src"
    setArchiveName "basic-demo-1.0.zip"
}
```

```bash
$ gradlew zip
$ gradlew tasks
$ gradlew properties
```