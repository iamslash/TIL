- [Materials](#materials)
- [Basic](#basic)
  - [Install](#install)
  - [Hello World](#hello-world)
  - [Directory Structures](#directory-structures)
  - [POM](#pom)
    - [project](#project)
    - [properties](#properties)
    - [dependencies](#dependencies)
    - [build](#build)
    - [repositories](#repositories)
    - [pluginRepositories](#pluginrepositories)
    - [distributionManagement](#distributionmanagement)
  - [Goal](#goal)
  - [Repository](#repository)
  - [Version](#version)
  - [Scope](#scope)
  - [Super POM](#super-pom)
  - [Plugin](#plugin)
- [Advanced](#advanced)
  - [BOM](#bom)

-----

# Materials

* [Maven 정복](https://wikidocs.net/17298)
* [maven archetype plugin - 템플릿에서 메이븐 프로젝트 생성하기](https://www.lesstif.com/pages/viewpage.action?pageId=21430332)
* [메이븐(Maven) 강의 @ youtube](https://www.youtube.com/watch?v=VAp0n9DmeEA&list=PLq8wAnVUcTFWRRi_JWLArMND_PnZM6Yja)
* [Maven in 5 Minutes @ apache](https://maven.apache.org/guides/getting-started/maven-in-five-minutes.html)

# Basic

## Install

[sdkman](/sdkman/README.md) 참고.

```bash
$ sdk list maven

$ sdk install maven 3.8.6

$ java --version
openjdk 17.0.5 2022-10-18 LTS
OpenJDK Runtime Environment Corretto-17.0.5.8.1 (build 17.0.5+8-LTS)
OpenJDK 64-Bit Server VM Corretto-17.0.5.8.1 (build 17.0.5+8-LTS, mixed mode, sharing)

$ mvn --version
Apache Maven 3.8.6 (84538c9988a25aec085021c365c560670ad80f63)
Maven home: /Users/david.s/.sdkman/candidates/maven/current
Java version: 17.0.5, vendor: Amazon.com Inc., runtime: /Users/david.s/.sdkman/candidates/java/17.0.5-amzn
Default locale: ko_KR, platform encoding: UTF-8
OS name: "mac os x", version: "12.6.1", arch: "x86_64", family: "mac"
```

## Hello World

다음과 같이 `pom.xml` 을 생성한다.

```bash
$ mkdir my-maven && cd my-mave
$ vim pom.xml
$ touch src/main/java/HelloWorld.java
```

```xml
<!-- pom/.xml -->
<project>
    <groupId>com.iamslash</groupId>
    <artifactId>HelloWorld</artifactId>
    <modelVersion>4.0.0</modelVersion>
    <version>1.0-SNAPSHOT</version>
    <packaging>jar</packaging>
    <build>
        <plugins>
        <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-compiler-plugin</artifactId>
            <version>3.7.0</version>
            <configuration>
                <source>1.8</source>
                <target>1.8</target>
            </configuration>
        </plugin>
        </plugins>
    </build>    
</project>
```

```java
// src/main/java/HelloWorld.java
public class HelloWorld{
    public static void main(String args[]){
        System.out.println("Maven Hello World!");
    }
}
```

```bash
$ mvn --version

# Clean the project
$ mvn clean

# Compile the project
$ mvn compile

# Execute from classes
$ java -cp target/classes HelloWorld

# Package the project
$ mvn package

# Execute from jars
$ java -cp target/HelloWorld-1.0-SNAPSHOT.jar HelloWorld
```

## Directory Structures

```bash
$ tree my-maven
.
├── pom.xml
├── src
│   └── main
│       └── java
│           └── HelloWorld.java
└── target
    ├── HelloWorld-1.0-SNAPSHOT.jar
    ├── classes
    │   └── HelloWorld.class
    ├── generated-sources
    │   └── annotations
    ├── maven-archiver
    │   └── pom.properties
    └── maven-status
        └── maven-compiler-plugin
            └── compile
                └── default-compile
                    ├── createdFiles.lst
                    └── inputFiles.lst
```

## POM

* [04. Maven pom.xml 파일 구조](https://wikidocs.net/18340)

---

POM stands for project of model. Maven 은 `pom.xml` 을 통해 task 를 정의한다.

### project

project meta data

* modelVersion
* groupId
* artifactId
* version
* packaging

### properties

maven 에서 사용할 속성들

### dependencies

의존성 libary 들의 모음. `groupId, artifactId, version` 는 필수임. 

### build

build meta data

### repositories

library 의 repository

### pluginRepositories

maven plugin 의 repository

### distributionManagement

`mvn deploy` 를 실행했을 때 배포될 repository

## Goal

[gradle](/gradle/README.md) 의 task 와 같다.

주요 goal 은 다음과 같다.

* clean
* compile
* package
* install
* deploy

## Repository

maven 은 다음과 같은 repository 를 갖는다.

* local repository
  * `~/.m2/` 에 `groupId, artifactId, versions` 로 directory 가 구성된다.
* remote repository
  * library repository
* plugin remote repository
  * maven plugin repository

## Version

* `1.0-SNAPSHOT` 은 개발진행을 표현한다.
* `1.0-RC1` 의 RC 는 Release Candidate 를 의미한다.
* `1.0-M1` 의 M 는 Milestone Release 를 의미한다.

## Scope

* compile
* provided
* runtime
* test
* system
* import

## Super POM

Project 의 pom.xml 은 super pom.xml 을 상속한다. super pom.xml 은
`/Users/david.s/.sdkman/candidates/maven/3.8.6/lib/maven-model-builder-3.8.6.jar`
의 `org/apache/maven/model/pom-4.0.0.xml` 이다.

## Plugin

주요 plugin 은 다음과 같다.

* clean
* maven-compiler-plugin
* test
* install
* package
* deploy
* maven-jar-plugin
* maven-source-plugin
* maven-javadoc-plugin
* maven-war-plugin

# Advanced

## BOM 

* [Maven의 Transitive Dependency 길들이기](https://blog.sapzil.org/2018/01/21/taming-maven-transitive-dependencies/)
  
Bom stands for bill of materials. Bom 을 적용하면 version 을 강제할 수 있다.
