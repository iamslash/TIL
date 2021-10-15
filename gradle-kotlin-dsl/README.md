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
  - [](#)
  - [Type-safe model accessors](#type-safe-model-accessors)
  - [Multi-project builds](#multi-project-builds)
  - [When you can’t use the plugins {} block](#when-you-cant-use-the-plugins--block)
  - [Working with container objects](#working-with-container-objects)
  - [Working with runtime properties](#working-with-runtime-properties)
  - [The Kotlin DSL Plugin](#the-kotlin-dsl-plugin)
  - [The embedded Kotlin](#the-embedded-kotlin)
- [Advanced](#advanced)
  - [Maven BOM, POM](#maven-bom-pom)

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

또한 Kotlin DSL API 를 import 한다. Kotlin DSL API 는 org.gradle.kotlin.dsl and org.gradle.kotlin.dsl.plugins.dsl 패키지의 모든 type 들을 말한다???

## Applying Plugins

## Configuring plugins

## Configuration avoidance

## Configuring tasks

## Creating tasks

## Configurations and dependencies

## 

## Type-safe model accessors


## Multi-project builds



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

또한 Kotlin delegate Properties 를 통해 Container element 의 reference 를 얻어올 수 있다. `existing(), registering()` 은 avoidance api 이다. 피하고 싶다면 `getting(), creating()` 을 사용한다.

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

## The Kotlin DSL Plugin

## The embedded Kotlin

Gradle 은 Kotlin 을 포함하고 있다. 예를 들어 `Gradle 4.3` 은 `Kotlin DSL v0.12.1` 를 포함하고 있다. `Kotlin DSL v0.12.1` 를 다음과 같은 library 들을 포함한다.

* `kotlin-compiler-embeddable 1.1.51`
* `kotlin-stdlib 1.1.51`
* `kotlin-reflect 1.1.51`

# Advanced

## Maven BOM, POM

* [Spring with Maven BOM](https://www.baeldung.com/spring-maven-bom)
