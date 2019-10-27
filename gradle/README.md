# Abstract

java진영의 build tool인 gradle에 대해 정리한다. gradle은 maven보다
성능이 좋다.

# Materials

* [gradle DSL reference](https://docs.gradle.org/current/dsl/)
* [Creating New Gradle Builds](https://guides.gradle.org/creating-new-gradle-builds/)

# Basic

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