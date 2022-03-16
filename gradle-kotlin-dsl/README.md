- [Abstract](#abstract)
- [Materials](#materials)
- [Basics](#basics)
  - [gradle commandline](#gradle-commandline)
  - [Script File Naming](#script-file-naming)
  - [Implicit Imports](#implicit-imports)
  - [Applying Plugins](#applying-plugins)
  - [Configuring plugins](#configuring-plugins)
  - [Configuration avoidance](#configuration-avoidance)
  - [Configuring tasks](#configuring-tasks)
  - [Creating tasks](#creating-tasks)
  - [Configurations and dependencies](#configurations-and-dependencies)
  - [Type-safe model accessors](#type-safe-model-accessors)
  - [Multi-project builds](#multi-project-builds)
  - [When you can’t use the plugins {} block](#when-you-cant-use-the-plugins--block)
  - [Working with container objects](#working-with-container-objects)
  - [Working with runtime properties](#working-with-runtime-properties)
  - [The Kotlin DSL Plugin](#the-kotlin-dsl-plugin)
  - [The embedded Kotlin](#the-embedded-kotlin)
- [Advanced](#advanced)
  - [Maven BOM, POM](#maven-bom-pom)
  - [Publish Plugin](#publish-plugin)

----

# Abstract

kotlin DSL 로 [gradle](/gradle/README.md) 을 이용하는 방법에 대해 정리한다.

# Materials

* [Migrating build logic from Groovy to Kotlin @ gradle](https://docs.gradle.org/current/userguide/migrating_from_groovy_to_kotlin_dsl.html)
* [Gradle Kotlin DSL Primer @ gradle](https://docs.gradle.org/current/userguide/kotlin_dsl.html)
* [gradle/kotlin-dsl-samples @ github](https://github.com/gradle/kotlin-dsl-samples/tree/master/samples)

# Basics

## gradle commandline

```bash
# Show commandline help
$ ./gradlew --help
# Show tasks help
$ ./gradlew help

# Show a list of available tasks
$ ./gradlew tasks
# Show all tasks
$ ./gradlew tasks --all

# Run <task>
$ ./gradlew <task>
# Run <task> and log errors only
$ ./gradle -q <task>

# Show more detail about a task
$ ./gradlew help --task <task>
```

## Script File Naming

Kotlin DSL 을 이용하여 Build Sript 를 제작한다면 `build.gradle.kts` 를 생성한다. Settings Script 는 `settings.gradle.kts` 를 생성한다. Init Script 는 `init.gradle.kts` 를 생성한다.

Groovy DSL 을 이용하여 Script 를 제작한다면 `build.gradle, settings.gradle, init.gradle` 를 생성한다.

## Implicit Imports

Gradle Kotiln DSL Script 는 기본적으로 [default Gradle API imports](https://docs.gradle.org/current/userguide/writing_build_scripts.html#script-default-imports) 를 import 한다. 

또한 Kotlin DSL API 를 import 한다. Kotlin DSL API 는 `org.gradle.kotlin.dsl`, `org.gradle.kotlin.dsl.plugins.dsl` 패키지의 모든 type 들을 말한다???

## Applying Plugins

plugin 은 `plugins {}` 를 이용하여 적용할 수 있다. 이것을 declaratively 라고 한다. 또한 `apply(..)` 을 이용하여 적용할 수도 있다. 이것은 오래된 방식이다. imperatively 라고도 한다.

다음은 `plugins {}` 을 이용하여 plugin 을 적용한 예이다.

```kotlin
plugins {
    java
    jacoco
    `maven-publish`
    id("org.springframework.boot") version "2.4.1"
    id("org.springframework.boot") version "2.4.1" apply false
}
```

`jacoco` 는 `BuiltinPluginIdExtensions.kt` 에 다음과 같이 정의되어 있다.  package level 의 extention property 이다. `id("org.gradle.jacoco")` 를 호출하여 `PluginDependencySpec` object 를 리턴한다.

```kotlin
// BuiltinPluginIdExtensions.kt
/**
 * The builtin Gradle plugin implemented by [org.gradle.testing.jacoco.plugins.JacocoPlugin].
 *
 * Visit the [plugin user guide](https://docs.gradle.org/current/userguide/jacoco_plugin.html) for additional information.
 *
 * @see org.gradle.testing.jacoco.plugins.JacocoPlugin
 */
inline val org.gradle.plugin.use.PluginDependenciesSpec.`jacoco`: org.gradle.plugin.use.PluginDependencySpec
    get() = id("org.gradle.jacoco")
```

Kotlin DSL 은 `jacoco, java, maven-publish` 와 같은 **Gradle Core Plugin** 들을 extention property 로 제공한다. 모두 `BuiltinPluginIdExtensions.kt` 에 정의되어 있다.

`id("org.springframework.boot")` 의 `id` 는 `PluginDependenciesSpec.java` 에 다음과 같이 정의되어 있다. `PluginDependencySpec` 을 리턴한다.
 
```java
// PluginDependenciesSpec.java
public interface PluginDependenciesSpec {
    PluginDependencySpec id(String id);
}
```

`version, apply` 는 `PluginDependenciesSpecScope.kt` 에 다음과 같이 정의되어 있다. `apply false` 는 해당 plugin 은 적용하지 말라는 의미이다.

```kotlin
// PluginDependenciesSpecScope.kt
infix fun PluginDependencySpec.version(version: String?): PluginDependencySpec = version(version)

infix fun PluginDependencySpec.apply(apply: Boolean): PluginDependencySpec = apply(apply)
```

`plugins {}` 말고 `apply(..)` 를 이용한 imperative style 도 가능하다. 그러나 type safe accessor 를 사용하지 못한다. IntelliJ 의 자동완성 기능을 이용할 수 없다. 추천하지 않는다. 

```kotlin
buildscript {
    repositories {
        gradlePluginPortal()
    }
    dependencies {
        classpath("org.springframework.boot:spring-boot-gradle-plugin:2.4.1")
    }
}

apply(plugin = "java")
apply(plugin = "jacoco")
apply(plugin = "org.springframework.boot")
```

## Configuring plugins

`jacoco` 를 `plugins {}` 를 이용하여 적용했다면 다음과 같은 방법으로 설정할 수 있다.

```kotlin
plugins {
    jacoco
}

jacoco {
    toolVersion = "0.8.1"
}
```

`jacoco` 와 같은 것을 configuration element 라고 한다. 대부분의 Plugin 들은 Project 에 configuration element 를 추가한다. 만약 Plugin 들이 제공하는 configuration element 들을 알고 싶다면 다음과 같은 command line 으로 확인할 수 있다.

```bash
$ ./gradlew kotlinDslAccessorsReport
```

물론 `apply ()` 를 이용하여 다음과 같이 설정할 수도 있다. 그러나 추천하지 않는다.

```kotlin
apply(plugin = "checkstyle")

configure<CheckstyleExtension> {
    maxErrors = 10
}
```

## Configuration avoidance

Gradle build phase 에서 당장 필요없는 task 의 configuration 을 뒤로 미루는 것이다. 예를 들어 `compile` task 를 실행할 때 `code quality, testing, publishing` task 들은 설정할 필요가 없다. 이렇게 하면 효율적이다. 그렇다면 가급적이면 Configuration avoidance API 를 사용하라는 말인가???

[Configuration avoidance @ gradle](https://docs.gradle.org/current/userguide/migrating_from_groovy_to_kotlin_dsl.html#configuration-avoidance)

## Configuring tasks

task 를 설정하고 싶다면 다음과 같이 `tasks` Container 를 이용한다.

```kotlin
tasks.jar {
    archiveFileName.set("foo.jar")
}
```

`tasks` container 에 task 가 포함되어 있다.

또한 다음과 같이 `tasks` container API 를 이용하여 설정할 수도 있다.

```kotlin
tasks.named<Jar>("jar") {
    archiveFileName.set("foo.jar")
}
```

그러나 task 의 type 을 알아야 한다. task 의 type 은 다음과 같은 command line 으로 알아낼 수 있다.

```bash
$ ./gradlew help --task jar
```

또한 다음과 같이 `tasks` container eager API 를 이용하여 설정할 수도 있다. configuration avoidance 를 하지 않는다.

```kotlin
tasks.getByName<Jar>("jar") {
    archiveFileName.set("foo.jar")
}
```

이제 지금까지 얘기한 것들을 종합하여 `bootJar, bootRun` task 들을 decalaritive style 로 설정해보자. 즉, configuration avoidance 해보자.

```kotlin
plugins {
    java
    id("org.springframework.boot") version "2.4.5"
}

tasks.bootJar {
    archiveFileName.set("app.jar")
    mainClassName = "com.example.demo.Demo"
}

tasks.bootRun {
    mainClass.set("com.example.demo.Demo")
    args("--spring.profiles.active=demo")
}
```

또한 imperative style 로 설정해보자. task 의 type 을 알아야 해서 불편하다. 그리고 configuration avoidance 를 하지 않는다. 

```bash
$ ./gradlew help --task bootJar
...
Type
     BootJar (org.springframework.boot.gradle.tasks.bundling.BootJar)
$ ./gradlew help --task bootRun
...
Type
     BootRun (org.springframework.boot.gradle.tasks.run.BootRun)
```

```kotlin
import org.springframework.boot.gradle.tasks.bundling.BootJar
import org.springframework.boot.gradle.tasks.run.BootRun

plugins {
    java
    id("org.springframework.boot") version "2.4.5"
}

tasks.named<BootJar>("bootJar") {
    archiveFileName.set("app.jar")
    mainClassName = "com.example.demo.Demo"
}

tasks.named<BootRun>("bootRun") {
    mainClass.set("com.example.demo.Demo")
    args("--spring.profiles.active=demo")
}
```

## Creating tasks

task 는 top level funtion 인 `task(..)` 를 이용하여 생성한다. 이것을 eager api 라고 한다. 즉, configuration avoidance 하지 않는다.

```kotlin
task("greeting") {
    doLast { println("Hello, World!") }
}
```

다음과 같이 configuration avoidance API 인 `tasks.register` 로 생성할 수도 있다.

```kotlin
tasks.register("greeting") {
    doLast { println("Hello, World!") }
}
```

한편 다음과 같이 eager API 인 `tasks.create` 로 생성할 수도 있다.

```kotlin
tasks.create("greeting") {
    doLast { println("Hello, World!") }
}
```

## Configurations and dependencies

만약 `plugins {}` 를 이용하여 plugin 들을 적용했다면 dependencies 는 다음과 같이 선언한다.

```kotlin
plugins {
    `java-library`
}
dependencies {
    implementation("com.example:lib:1.1")
    runtimeOnly("com.example:runtime:1.0")
    testImplementation("com.example:test-support:1.3") {
        exclude(module = "junit")
    }
    testRuntimeOnly("com.example:test-junit-jupiter-runtime:1.3")
}
```

그러나 `apply (..)` 를 이용하여 plugin 들을 적용했다면 dependencies 는 다음과 같이 작성한다.

```kotlin
apply(plugin = "java-library")
dependencies {
    "implementation"("com.example:lib:1.1")
    "runtimeOnly"("com.example:runtime:1.0")
    "testImplementation"("com.example:test-support:1.3") {
        exclude(module = "junit")
    }
    "testRuntimeOnly"("com.example:test-junit-jupiter-runtime:1.3")
}
```

type-safe accesor 가 없기 때문에 추천하지 않는다. 훨씬 복잡하다. 반드시 `plugins {}` 를 이용하여 plugin 들을 적용하자.

만약 custom dependency 가 필요하다면 다음과 같이 구현한다. 다음은 `db, integTestImplementation` custom configuration 을 선언하고 사용한 예이다. 

```kotlin
val db by configurations.creating
val integTestImplementation by configurations.creating {
    extendsFrom(configurations["testImplementation"])
}

dependencies {
    db("org.postgresql:postgresql")
    integTestImplementation("com.example:integ-test-support:1.3")
}
```

만약 위와 같은 경우 `testRuntimeOnly` 와 같은 기존의 configuration 을 사용하고 싶다면 다음과 같이 referencing 해서 작성한다.

```kotlin
// get the existing 'testRuntimeOnly' configuration
val testRuntimeOnly by configurations

dependencies {
    testRuntimeOnly("com.example:test-junit-jupiter-runtime:1.3")
    "db"("org.postgresql:postgresql")
    "integTestImplementation"("com.example:integ-test-support:1.3")
}
```

## Type-safe model accessors

Gradle Kotlin DSL 은 다음과 같은 configuration 들에 대해 type-safe model accessors 를 제공한다. IDE 를 통해 자동완성기능을 이용할 수 있다.

* Java plugin 이 제공하는 `implementation, runtimeOnly` configuration
* Project extension and conventions such as `sourceSets`
* `tasks, configurations` container 의 element 들
* Project extension container 들의 element 들. 예를 들면 sourceSets 의 element 들

각 항목에 대한 자세한 사항은 [Type-safe model accessors @ gradle](https://docs.gradle.org/current/userguide/kotlin_dsl.html#type-safe-accessors) 를 참고한다.

Plugin 이 제공하는 type model accessor 는 `plugins {}` 이후 사용가능하다. 

```kotlin
plugins {
    `java-library`
}

dependencies {                              
    api("junit:junit:4.13")
    implementation("junit:junit:4.13")
    testImplementation("junit:junit:4.13")
}

configurations {                            
    implementation {
        resolutionStrategy.failOnVersionConflict()
    }
}

sourceSets {                                
    main {                                  
        java.srcDir("src/core/java")
    }
}

java {                                      
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
}

tasks {
    test {                                  
        testLogging.showExceptions = true
    }
}
```

또한 다음과 같은 방법으로 configuration avoidance 할 수 있다.

```kotlin
tasks.test {
    // lazy configuration
}

// Lazy reference
val testProvider: TaskProvider<Test> = tasks.test

testProvider {
    // lazy configuration
}

// Eagerly realized Test task, defeat configuration avoidance if done out of a lazy context
val test: Test = tasks.test.get()
```

`apply(...)` 를 사용하여 plugin 을 적용하면 type-safe model accessor 를 사용할 수 없다.

```kotlin
// build.gradle.kts
apply(plugin = "java-library")

dependencies {
    "api"("junit:junit:4.13")
    "implementation"("junit:junit:4.13")
    "testImplementation"("junit:junit:4.13")
}

configurations {
    "implementation" {
        resolutionStrategy.failOnVersionConflict()
    }
}

configure<SourceSetContainer> {
    named("main") {
        java.srcDir("src/core/java")
    }
}

configure<JavaPluginConvention> {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
}

tasks {
    named<Test>("test") {
        testLogging.showExceptions = true
    }
}
```

다음과 같은 경우는 type-safe model accessor 를 사용할 수 없다.

* `apply(plugin = "id")` 를 사용하여 plugin 이 적용되었을 때
* The project build script
* `apply(from = "script-plugin.gradle.kts")` 를 사용하여 plugin 이 적용되었을 때
* [cross-project configuration](https://docs.gradle.org/current/userguide/kotlin_dsl.html#sec:kotlin_cross_project_configuration) 이 적용되었을 때

## Multi-project builds

다수의 프로젝트의 build script 는 다음과 같이 작성한다.

```kotlin
// settings.gradle.kts
rootProject.name = "multi-project-build"
include("domain", "infra", "http")

// build.gradle.kts
plugins {
    id("com.github.johnrengelman.shadow") version "4.0.1" apply false
    id("io.ratpack.ratpack-java") version "1.8.2" apply false
}

// domain/build.gradle.kts
plugins {
    `java-library`
}

dependencies {
    api("javax.measure:unit-api:1.0")
    implementation("tec.units:unit-ri:1.0.3")
}

// infra/build.gradle.kts
plugins {
    `java-library`
    id("com.github.johnrengelman.shadow")
}

shadow {
    applicationDistribution.from("src/dist")
}

tasks.shadowJar {
    minimize()
}

// http/build.gradle.kts
plugins {
    java
    id("io.ratpack.ratpack-java")
}

dependencies {
    implementation(project(":domain"))
    implementation(project(":infra"))
    implementation(ratpack.dependency("dropwizard-metrics"))
}

application {
    mainClass.set("example.App")
}

ratpack.baseDir = file("src/ratpack/baseDir")
```

만약에 gradlePluginPortal 보다 높은 우선순위의 repository 가 필요하다면 다음과 같이 `pluginManagement` 애 repositories 를 선언한다. 

```kotlin
pluginManagement {
    repositories {
        mavenCentral()
        gradlePluginPortal()
    }
}
```

artifact 가 repository 에 없다면 `plugins {}` 쓸 수 없다. 왜지???

```kotlin
// settings.gradle.kts
include("lib", "app")

// build.gradle.kts
buildscript {
    repositories {
        google()
        gradlePluginPortal()
    }
    dependencies {
        classpath("com.android.tools.build:gradle:4.1.2")
    }
}

// lib/build.gradle.kts
plugins {
    id("com.android.library")
}

android {
    // ...
}

// app/build.gradle.kts
plugins {
    id("com.android.application")
}

android {
    // ...
}
```

하나의 project build script 에서 다른 project build script 를 포함하는 것을 Cross-configuring 이라고 한다. 다음은 root project build script 에서 `domain, infra, http` project 의 build script 를 구현한 것이다. type-safe model accessor 는 사용할 수 없음을 주의하자.

```kotlin
// settings.gradle.kts
rootProject.name = "multi-project-build"
include("domain", "infra", "http")

// build.gradle.kts
import com.github.jengelman.gradle.plugins.shadow.ShadowExtension
import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar
import ratpack.gradle.RatpackExtension

plugins {
    id("com.github.johnrengelman.shadow") version "4.0.1" apply false
    id("io.ratpack.ratpack-java") version "1.8.2" apply false
}

project(":domain") {
    apply(plugin = "java-library")
    dependencies {
        "api"("javax.measure:unit-api:1.0")
        "implementation"("tec.units:unit-ri:1.0.3")
    }
}

project(":infra") {
    apply(plugin = "java-library")
    apply(plugin = "com.github.johnrengelman.shadow")
    configure<ShadowExtension> {
        applicationDistribution.from("src/dist")
    }
    tasks.named<ShadowJar>("shadowJar") {
        minimize()
    }
}

project(":http") {
    apply(plugin = "java")
    apply(plugin = "io.ratpack.ratpack-java")
    repositories { mavenCentral() }
    val ratpack = the<RatpackExtension>()
    dependencies {
        "implementation"(project(":domain"))
        "implementation"(project(":infra"))
        "implementation"(ratpack.dependency("dropwizard-metrics"))
        "runtimeOnly"("org.slf4j:slf4j-simple:1.7.25")
    }
    configure<JavaApplication> {
        mainClass.set("example.App")
    }
    ratpack.baseDir = file("src/ratpack/baseDir")
}
```

## When you can’t use the plugins {} block

[Gradle Plugin Portal](https://plugins.gradle.org/) 에 등록된 plugin 은 `plugin {}` 를 통해서 사용가능하다.

```kotlin
// build.gradle.kts
plugins {
    java
    id("io.ratpack.ratpack-java")
}
...
```

그러나 publishing 방법에 따라 `plgin {}` 를 통해서 사용할 수 없는 plugin 도 있다. 이유는 [When you can’t use the plugins {} block @ gradle](https://docs.gradle.org/current/userguide/kotlin_dsl.html#sec:plugins_resolution_strategy) 를 참고하자.

그때는 다음과 같이 `settings.gradle.kts` 에 `PluginManagement {}` 를 사용하여 `id` 와 `module` 을 mapping 한다. `id("com.android.application")` 는 `useModule("com.android.tools.build:gradle:${requested.version}"` 에 mapping 되어 있고 `google()` 을 통해 다운로드한다.

```kotlin
// settings.gradle.kts
pluginManagement {
    repositories {
        google()
        gradlePluginPortal()
    }
    resolutionStrategy {
        eachPlugin {
            if(requested.id.namespace == "com.android") {
                useModule("com.android.tools.build:gradle:${requested.version}")
            }
        }
    }
}

// build.gradle.kts
plugins {
    id("com.android.application") version "4.1.2"
}

android {
    // ...
}
```

## Working with container objects

Container object 는 다른 object 를 포함하는 object 를 말한다. 예를 들면 `Configurations` object 는 `Configuration` object 를 포함한다. 또한 `Tasks` object 는 `Task` object 를 포함한다.

대부분의 Gradle Container Object 들은 ` NamedDomainObjectContainer<DomainObjectType>` 를 구현한다. 몇몇은 `PolymorphicDomainObjectContainer<BaseType>` 를 구현한다. 다음은 Container Object 의 `named(), register()` 를 사용한 예이다. `named()` 는 Container element object 를 수정할 수 있고 `register()` 는 Container element object 를 생성할 수 있다.

```kotlin
// build.gradle.kts
tasks.named("check")                    
tasks.register("myTask1")               

tasks.named<JavaCompile>("compileJava") 
tasks.register<Copy>("myCopy1")         

tasks.named("assemble") {               
    dependsOn(":myTask1")
}
tasks.register("myTask2") {             
    description = "Some meaningful words"
}

tasks.named<Test>("test") {             
    testLogging.showStackTraces = true
}
tasks.register<Copy>("myCopy2") {       
    from("source")
    into("destination")
}
```

또한 Kotlin delegate Properties 를 통해 Container element 의 reference 를 얻어올 수 있다. `existing(), registering()` 은 avoidance API 이다. eager API 를 사용하고 싶다면 `getting(), creating()` 을 사용한다.

```kotlin
val check by tasks.existing
val myTask1 by tasks.registering

val compileJava by tasks.existing(JavaCompile::class)
val myCopy1 by tasks.registering(Copy::class)

val assemble by tasks.existing {
    dependsOn(myTask1)  
}
val myTask2 by tasks.registering {
    description = "Some meaningful words"
}

val test by tasks.existing(Test::class) {
    testLogging.showStackTraces = true
}
val myCopy2 by tasks.registering(Copy::class) {
    from("source")
    into("destination")
}
```

하나의 Container object 에서 다수의 Container element object 를 접근할 예정이라면 다음과 같은 형태로 구현한다. 반복해서 Container object 를 표기할 필요가 없다. 다음은 container element object 중 `test, check` 를 설정하고 `myCheck, myHelp` container element object 를 생성하는 예이다. 

```kotlin
tasks {
    test {
        testLogging.showStackTraces = true
    }
    val myCheck by registering {
        doLast { /* assert on something meaningful */ }
    }
    check {
        dependsOn(myCheck)
    }
    register("myHelp") {
        doLast { /* do something helpful */ }
    }
}
```

## Working with runtime properties

Gradle 은 runtime 에 정의되는 2 가지 properties 가 있다. 그것은 **project properties**, **extra properties** 를 말한다.

project properties 는 다음과 같이 Kotlin delegated properteis 로 접근이 가능하다. 

```kotlin
val myProperty: String by project  
val myNullableProperty: String? by project 
```

extra properties 는 다음과 같이 Kotiln delegated properties 로 접근이 가능하다.

```kotlin
val myNewProperty by extra("initial value")  
val myOtherNewProperty by extra { "calculated initial value" }  

val myProperty: String by extra  
val myNullableProperty: String? by extra  
```

이렇게 project propeties, extra properties 를 사용하는 것은 project build scripts, script plugins, settings scripts and initialization scripts 에서 가능하다.

만약 sub-project 에서 root-project 의 extra properties 를 접근하고 싶다면 다음과 같이 한다.

```kotiln
val myNewProperty: String by rootProject.extra 
```

extra properties 는 주로 Project 에 추가해서 사용지만 Task 에도 추가할 수 있다. Task 는 ExtensionAware 를 상속하기 때문이다.

```kotlin
tasks {
    test {
        val reportType by extra("dev")  
        doLast {
            // Use 'suffix' for post processing of reports
        }
    }

    register<Zip>("archiveTestReports") {
        val reportType: String by test.get().extra  
        archiveAppendix.set(reportType)
        from(test.get().reports.html.destination)
    }
}
```

만약 eager configuration 을 하고 싶다면 다음과 같이 한다.

```kotlin
tasks.test.doLast { ... }

val testReportType by tasks.test.get().extra("dev")  

tasks.create<Zip>("archiveTestReports") {
    archiveAppendix.set(testReportType)  
    from(test.get().reports.html.destination)
}
```

extra properties 는 다음과 같이 map 을 이용할 수도 있다.

```kotlin
extra["myNewProperty"] = "initial value"  

tasks.create("myTask") {
    doLast {
        println("Property: ${project.extra["myNewProperty"]}")  
    }
}
```

## The Kotlin DSL Plugin

Kotlin DSL Plugin 은 다음과 같이 사용한다. plugin 의 version 은 가급적 사용하지 않는다. Gradle version 과 호환되는 Kotlin DSL version 이 사용될 것이다.

```gradle
plugins {
    `kotlin-dsl`
}

repositories {
    // The org.jetbrains.kotlin.jvm plugin requires a repository
    // where to download the Kotlin compiler dependencies from.
    mavenCentral()
}
```

Kotlin DSL Plugin 을 적용하면 다음과 같은 것을 할 수 있다.

* Kotlin source 를 compile 하기 위해 [Kotlin Plugin](https://kotlinlang.org/docs/gradle.html#targeting-the-jvm) 을 적용한다.
* `kotlin-stdlib-jdk8, kotlin-reflect and gradleKotlinDsl()` dependencies 를 `compileOnly, testImplementation` 에 적용한다. Kotlin libraries 와 Gradle API 를 Kotlin code 에서 사용할 수 있다. 
* Kotlin DSL scripts 에서 사용된 것과 같은 것으로 Kotiln compiler 를 설정한다. 
* [Precompiled script plugins](https://docs.gradle.org/current/userguide/custom_plugins.html#sec:precompiled_plugins) 를 사용할 수 있다.

## The embedded Kotlin

Gradle 은 Kotlin 을 포함하고 있다. 예를 들어 `Gradle 4.3` 은 `Kotlin DSL v0.12.1` 를 포함하고 있다. `Kotlin DSL v0.12.1` 를 다음과 같은 library 들을 포함한다.

* `kotlin-compiler-embeddable 1.1.51`
* `kotlin-stdlib 1.1.51`
* `kotlin-reflect 1.1.51`

# Advanced

## Maven BOM, POM

* [Spring with Maven BOM](https://www.baeldung.com/spring-maven-bom)

----

POM 은 Project Object Model 를 말한다. Project 의 설정이 담겨진 xml 파일이다. 

BOM 은 Bill Of Materials 를 말한다. POM 의 특별한 형태이다. 주로 third party library 들의 버전관리를 담당한다.

```xml
<project ...>
	
    <modelVersion>4.0.0</modelVersion>
    <groupId>baeldung</groupId>
    <artifactId>Baeldung-BOM</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <packaging>pom</packaging>
    <name>BaelDung-BOM</name>
    <description>parent pom</description>
    <dependencyManagement>
        <dependencies>
            <dependency>
                <groupId>test</groupId>
                <artifactId>a</artifactId>
                <version>1.2</version>
            </dependency>
            <dependency>
                <groupId>test</groupId>
                <artifactId>b</artifactId>
                <version>1.0</version>
                <scope>compile</scope>
            </dependency>
            <dependency>
                <groupId>test</groupId>
                <artifactId>c</artifactId>
                <version>1.0</version>
                <scope>compile</scope>
            </dependency>
        </dependencies>
    </dependencyManagement>
</project>
```

`<dependencyManagement>` 안에 artifact 들이 선언되어 있다.

POM 파일에서 2 가지 방법으로 BOM 을 사용할 수 있다.

첫번째 방법은 POM 파일에서 BOM 을 상속하는 것이다. `<parent>` 를 이용한다.

```xml
<project ...>
    <modelVersion>4.0.0</modelVersion>
    <groupId>baeldung</groupId>
    <artifactId>Test</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <packaging>pom</packaging>
    <name>Test</name>
    <parent>
        <groupId>baeldung</groupId>
        <artifactId>Baeldung-BOM</artifactId>
        <version>0.0.1-SNAPSHOT</version>
    </parent>
</project>
```

두번째 방법은 POM 파일에서 BOM 을 import 하는 것이다. `<dependencyManagement>` 를 이용한다.

```xml
<project ...>
    <modelVersion>4.0.0</modelVersion>
    <groupId>baeldung</groupId>
    <artifactId>Test</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <packaging>pom</packaging>
    <name>Test</name>
    
    <dependencyManagement>
        <dependencies>
            <dependency>
                <groupId>baeldung</groupId>
                <artifactId>Baeldung-BOM</artifactId>
                <version>0.0.1-SNAPSHOT</version>
                <type>pom</type>
                <scope>import</scope>
            </dependency>
        </dependencies>
    </dependencyManagement>
</project>
```

Spring 의 경우는 다음과 같이 `<dependencyManagement>` 를 이용하여 BOM 을 import 할 수 있다.

```xml
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>org.springframework</groupId>
            <artifactId>spring-framework-bom</artifactId>
            <version>4.3.8.RELEASE</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>
```

따라서 다음과 같이 version 을 생략해서 사용해도 artifact 들의 version 이 BOM 에 표기된대로 관리된다. 

```xml
<dependencies>
    <dependency>
        <groupId>org.springframework</groupId>
        <artifactId>spring-context</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework</groupId>
        <artifactId>spring-web</artifactId>
    </dependency>
<dependencies>
```

## Publish Plugin

* [publish plugin](publish-plugin.md)
