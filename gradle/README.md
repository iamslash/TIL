# Abstract

Gradle 은 task runner 이다. Gradle 은 maven 보다 성능이 좋다.

# Materials
 
* [Gradle for Android and Java @ udacity](https://classroom.udacity.com/courses/ud867) 
  * Great materials
  * [src](https://github.com/udacity/ud867)
* [Command-Line Interface @ Gradle](https://docs.gradle.org/current/userguide/command_line_interface.html)
* [gradle DSL reference @ Gradle](https://docs.gradle.org/current/dsl/)
* [Creating New Gradle Builds @ Gradle](https://guides.gradle.org/creating-new-gradle-builds/)
* [Chapter 01 그레이드의 시작 - 1. 그레이들이란? @ youtube](https://www.youtube.com/watch?v=s-XZ5B15ZJ0&list=PL7mmuO705dG2pdxCYCCJeAgOeuQN1seZz)

# Basic

* [2. Gradle의 기본 구조 살펴보기](https://gmind.tistory.com/entry/Gradle%EC%9D%98-%EA%B8%B0%EB%B3%B8-%EA%B8%B0%EB%8A%A5-%EB%A7%9B%EB%B3%B4%EA%B8%B0?category=655027)

```bash
$ gradle version
$ cd my-app

# Generates gradlew, gradlew.bat, .gradle, gradle
# Those are for system Gradle is not installed on. 
# Those should be tracked by SCM.
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

다음은 생성된 `build.gradle` 이다. 

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
$ gradle build -i

Initialized native services in: /Users/davidsun/.gradle/native
The client will now receive all logging from the daemon (pid: 89198). The daemon log file: /Users/davidsun/.gradle/daemon/6.1.1/daemon-89198.out.log
Starting 8th build in daemon [uptime: 2 mins 42.693 secs, performance: 99%, non-heap usage: 15% of 268.4 MB]
Using 16 worker leases.
Starting Build
Settings evaluated using settings file '/Users/davidsun/my/gradle/my-app/settings.gradle'.
Projects loaded. Root project using build file '/Users/davidsun/my/gradle/my-app/build.gradle'.
Included projects: [root project 'my-app']

> Configure project :
Evaluating root project 'my-app' using build file '/Users/davidsun/my/gradle/my-app/build.gradle'.
All projects evaluated.
Selected primary task 'build' from project :
Tasks to be executed: [task ':compileJava', task ':processResources', task ':classes', task ':jar', task ':startScripts', task ':distTar', task ':distZip', task ':assemble', task ':compileTestJava', task ':processTestResources', task ':testClasses', task ':test', task ':check', task ':build']
:compileJava (Thread[Execution worker for ':',5,main]) started.

...

BUILD SUCCESSFUL in 602ms
7 actionable tasks: 7 up-to-date
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

gradle deamon 은 gradle task 수행 속도를 빠르게 도와 준다. `gradle --stop` 은 gradle daemon 을 멈춘다.

`println` 은 `System.out.println` 의 shortcut 이다. groovy 에서 `closure` 는 anonymous block of code 를 말한다. 다음은 `closure` 의 예이다.

```groovy
task groovy {}

def foo = "One million dollars"
def myClosure = {
    println "Hello from a closure"
    println "The value of foo is $foo"
}

myClosure()
def bar = myClosure
def baz = bar
baz()
```

`closure` 의 delegate 을 특정 instance 로 assign 하면 `closure` 의 variables, methods 들은 `closure.delegate` 의 variables, methods 와 같다.

```groovy
class GroovyGreeter {
    String greeting = "Default greeting"
    def printGreeting() { println "Greeting: $greeting" }
}
def myGroovyGreeter = new GroovyGreeter()

myGroovyGreeter.printGreeting()
myGroovyGreeter.greeting = "My custom greeting"
myGroovyGreeter.printGreeting()

def greetingClosure = {
  greeting = "Setting the greeting from a closure"
  printGreeting()
}

// greetingClosure() // This doesn't work, because `greeting` isn't defined
greetingClosure.delegate = myGroovyGreeter
greetingClosure() // This works as `greeting` is a property of the delegate
```

`-b` option 은 특정 build file 을 실행한다.

```bash
$ gradle -b a.gradle HelloWorld
```

closure 가 delgate 을 가질 수 있는 것 처럼 build.gradle 은 project 가 delegate 이다. 

```groovy
// 
project.task("myTask1")
// This is same with above
task("myTask2")
// We can leave off the parentheses.
task "myTask3"
```

다음은 task object 의 properties 인 description, group, doLast 를 assign 하는 예이다.

```groovy
task myTask7 {
    description("Description") // Function call works
    //description "Description" // This is identical to the line above
    group = "Some group" // Assignment also works
    doLast { // We can also omit the parentheses, because Groovy syntax
        println "Here's the action"
    }
}
```

또한 다음과 같이 task object 의 properties 를 argument 로 assign 할 수도 있다.

```groovy
task myTask8(description: "Another description") {
    doLast {
        println "Doing something"
    }
}
```

task 들은 `dependsOn`, `finalizedBy`, `mustRunAfter` 를 통해서 의존관계를 형성할 수 있다.

```groovy

task putOnSocks {
    doLast {
        println "Putting on Socks."
    }
}

task putOnShoes {
    dependsOn "putOnSocks"
    doLast {
        println "Putting on Shoes."
    }
}
```

`$ gradle tasks` 는 putOnSocks 를 보여주지는 않는다. 그러나 `$ gradle tasks --all` 을 통해 볼 수 있다.

```groovy

task eatBreakfast {
    finalizedBy "brushYourTeeth"
    doLast{
        println "Om nom nom breakfast!"
    }
}

task brushYourTeeth {
    doLast {
        println "Brushie Brushie Brushie."
    }
}
```

`gradle putOnFragrance takeShower` 를 실행하면 `takeShower` task 를 실행하고 `putOnFragrance` task 를 실행한다.

```grooyv
task takeShower {
    doLast {
        println "Taking a shower."
    }
}

task putOnFragrance {
    shouldRunAfter "takeShower"
    doLast{
        println "Smellin' fresh!"
    }
}
```



# Advanced 

## `compile` vs `implementation`

* [What's the difference between implementation and compile in Gradle? @ stackoverflow](https://stackoverflow.com/questions/44493378/whats-the-difference-between-implementation-and-compile-in-gradle)

다음의 것들은 각각 변경되었다.

| previous             | now                         |
| -------------------- | --------------------------- |
| `compile`            | `implementation`            |
| `testCompile`        | `testImplementation`        |
| `debugCompile`       | `debugImplementation`       |
| `androidTestCompile` | `androidTestImplementation` |

## `implementation` vs `api`

다음과 같이 라이브러리의 의존성이 설정되어 있다고 해보자.

```
A(implementation) -> B -> C -> D
```

만약 `A` library 의 코드가 수정되어 rebuild 된다면 `implementation` 의 경우는 `A, B` 만 rebuild 된다. 그러나 `api` 의 경우는 `A, B, C, D` 모두 rebuild 된다.
