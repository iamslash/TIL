- [Abstract](#abstract)
- [Materials](#materials)
- [Install](#install)
  - [Install with 2.190.1 docker on windows10](#install-with-21901-docker-on-windows10)
  - [Install Jenkins 2.190.1 with docker on maxOS](#install-jenkins-21901-with-docker-on-maxos)
- [Basic](#basic)
  - [jenkin_home structure](#jenkinhome-structure)
    - [directories](#directories)
    - [files](#files)
  - [Build Now Process](#build-now-process)
  - [Setting](#setting)
    - [Locale](#locale)
    - [GitHub secret text](#github-secret-text)
    - [Simple Job with Pipeline value](#simple-job-with-pipeline-value)
    - [Simple Job with pipeline script from scm](#simple-job-with-pipeline-script-from-scm)
  - [Contribution](#contribution)
  - [JenkinsCLI](#jenkinscli)
  - [Pipeline as a code](#pipeline-as-a-code)
  - [Declaritive pipeline](#declaritive-pipeline)
  - [Scripted pipeline](#scripted-pipeline)
- [How to make a Jenkins-plugin](#how-to-make-a-jenkins-plugin)
- [Script Console](#script-console)

----

# Abstract

jenkins 에 대해 정리한다.

# Materials

* [XECon2016 - GitHub + Jenkins + Docker로 자동배포 시스템 구축하기. 조정현 @ youtube](https://www.youtube.com/watch?v=ZM9sU3nqCMM)
* [도커(Docker) 활용 및 배포 자동화 실전 초급 @ youtube](https://www.youtube.com/playlist?list=PLRx0vPvlEmdChjc6N3JnLaX-Gihh5pHcx)
* [Jenkins Handbook](https://jenkins.io/doc/book/)
  * 꼭 읽어야할 필독서
  * [pdf](https://jenkins.io/user-handbook.pdf)
* [Jenkins doc](https://jenkins.io/doc/)
* [jenkins @ github](https://github.com/jenkinsci/jenkins)
* [Jenkins World 2017: Mastering the Jenkins Script Console @ youtube](https://www.youtube.com/watch?v=qaUPESDcsGg)
  * jenkins script console

# Install

## Install with 2.190.1 docker on windows10

```bash
$ docker pull jenkins/jenkins:lts
$ docker run -d -p 50000:50000 -p 8080:8080 -v D:\my\dockervolume\jenkins_home:/var/jenkins_home --name jenkins jenkins/jenkins:lts
$ docker logs jenkins -f
```

browser 로 `localhost:8080` 으로 접속한다. docker 실행창에 출력된 key 를 입력한다. install suggested plugins 하면 끝. 플러그인 설치를 실패할 때가 있다. 그렇다면 `C:\my\dockervolume\jenkins_home/*` 를 모두 지우고 `docker stop, rm` 이후 다시 실행해본다. 잘 된다.

## Install Jenkins 2.190.1 with docker on maxOS

```bash
$ docker pull jenkins/jenkins:lts
$ docker run -d -p 50000:50000 -p 8080:8080 -v /Users/davidsun/my/> dockervolume/jenkins_home:/var/jenkins_home --name jenkins jenkins/jenkins:lts
$ docker logs jenkins -f
```

# Basic

## jenkin_home structure

### directories

| directory | description |
|-----------|-------------|
| `/jobs` | list of jobs |
| `/nodes` | list of nodes |
| `/logs` | log files |
| `/plugins` | list of plugins |
| `/workspace` | list of job worksspacees |

### files

| directory | description |
|-----------|-------------|
| `/config.xml` | global configuration |

## Build Now Process

* make workspace direcotry at `/jenkins_home/worksspace/<job-name>`
* execute Pipeline script

## Setting

### Locale 

* MENU | Manage Plugins | Install Locale plugin
* MENU | Configure System
* Locale | Default Language | en or ENGLISH
* check `Ignore browser preference and force this language to all users`

### GitHub secret text

* [gitHub와 Jenkins 연결하기](https://webcache.googleusercontent.com/search?q=cache:P6VRZNmJqRkJ:https://bcho.tistory.com/1237+&cd=1&hl=ko&ct=clnk&gl=kr)


### Simple Job with Pipeline value

* Jenkins | New Item | "HelloWorld.pipeline" with Pipeline template
* Pipeline | Pipeline script

```groovy
node {
    def hello = 'Hello World'
    stage ('clone') {
        git 'https://github.com/welearntocode/HelloWorld.git'
    }
    dir ('sh') {
        stage ('sh/execute') {
            sh './a.sh'
        }
    }
    stage ('print') {
        print(hello) 
    }
}

// void for no return
// def for return
void print(message) {
    echo "${message}"
}
```

* Build Now

### Simple Job with pipeline script from scm

* Jenkinsfile

```groovy
node {
    def hello = 'Hello World'
    stage ('clone') {
        git 'https://github.com/welearntocode/HelloWorld.git'
    }
    dir ('sh') {
        stage ('sh/execute') {
            sh './a.sh'
        }
    }
    stage ('print') {
        print(hello) 
    }
}

// void for no return
// def for return
void print(message) {
    echo "${message}"
}
```

* Jenkins | New Item | "HelloWorld.pipeline" with Pipeline template
* Pipeline | Pipeline script from SCM
* Build Now

## Contribution

* [Beginners Guide to Contributing](https://wiki.jenkins.io/display/JENKINS/Beginners+Guide+to+Contributing)

## JenkinsCLI

```bash
$ java -jar jenkins-cli.jar -s http://localhost:8080/ -auth iamslash:?????? help
```

## Pipeline as a code

* [Learning Jenkins Pipeline @ github](https://github.com/mcpaint/learning-jenkins-pipeline)
  * declaritive pipeline script 연구
* [Jenkinsfile 을 이용한 젠킨스 Pipeline 설정](https://limsungmook.github.io/2016/11/09/jenkins-pipeline/)
* [Pipeline as Code with Jenkins](https://jenkins.io/solutions/pipeline/)
* [Using a Jenkinsfile ](https://jenkins.io/doc/book/pipeline/jenkinsfile/)

----

`Jenkinsfile` 이라는 이름의 text file 이다. repository root 에 groovy 로 작성한다. `Declarative Pipeline`, `Scripted Pipeline` 과 같은 두가지 형식으로 작성한다. 

browser 로 `http://localhost:8080/pipeline-syntax` 를 접속하면 자세한 reference 들을 확인할 수 있다.

`Jenkinsfile` 의 문법은 [DSL with Groovy](/groovy/README.md) 를 참고하자.

## Declaritive pipeline

* [Pipeline Syntax](https://jenkins.io/doc/book/pipeline/syntax)
* [Pipeline Steps Reference](https://jenkins.io/doc/pipeline/steps/)

----

Declarative Pipeline follow the same rules.

* The top-level of the Pipeline must be a block, specifically: `pipeline { }`
* No semicolons as statement separators. Each statement has to be on its own line
* Blocks must only consist of **Sections**, **Directives**, **Steps**, or assignment statements.
* A property reference statement is treated as no-argument method invocation. So for example, `input` is treated as `input()`

Sections in Declarative Pipeline typically contain one or more Directives or Steps.

* agent, post, stages, steps

Directives are consisted of these.

* environment, options, parameters, triggers, jenkins cron syntax, stage, tools, input, when

Sequential stages is a list of nested stages to be run within them in sequential order. 

```groovy
// Jenkinsfile (Declarative Pipeline)
pipeline {
    agent none
    stages {
        stage('Non-Sequential Stage') {
            agent {
                label 'for-non-sequential'
            }
            steps {
                echo "On Non-Sequential Stage"
            }
        }
        stage('Sequential') {
            agent {
                label 'for-sequential'
            }
            environment {
                FOR_SEQUENTIAL = "some-value"
            }
            stages {
                stage('In Sequential 1') {
                    steps {
                        echo "In Sequential 1"
                    }
                }
                stage('In Sequential 2') {
                    steps {
                        echo "In Sequential 2"
                    }
                }
                stage('Parallel In Sequential') {
                    parallel {
                        stage('In Parallel 1') {
                            steps {
                                echo "In Parallel 1"
                            }
                        }
                        stage('In Parallel 2') {
                            steps {
                                echo "In Parallel 2"
                            }
                        }
                    }
                }
            }
        }
    }
}
```

Parallel block will be executed in parallel.

```groovy
// Jenkinsfile (Declarative Pipeline)
pipeline {
    agent any
    stages {
        stage('Non-Parallel Stage') {
            steps {
                echo 'This stage will be executed first.'
            }
        }
        stage('Parallel Stage') {
            when {
                branch 'master'
            }
            failFast true
            parallel {
                stage('Branch A') {
                    agent {
                        label "for-branch-a"
                    }
                    steps {
                        echo "On Branch A"
                    }
                }
                stage('Branch B') {
                    agent {
                        label "for-branch-b"
                    }
                    steps {
                        echo "On Branch B"
                    }
                }
                stage('Branch C') {
                    agent {
                        label "for-branch-c"
                    }
                    stages {
                        stage('Nested 1') {
                            steps {
                                echo "In stage Nested 1 within Branch C"
                            }
                        }
                        stage('Nested 2') {
                            steps {
                                echo "In stage Nested 2 within Branch C"
                            }
                        }
                    }
                }
            }
        }
    }
}
```

Declarative Pipelines may use all the avilable steps in the [Pipeline Steps Reference](https://jenkins.io/doc/pipeline/steps/). The `script` step is only supported in Declaritive Piepline.

```groovy
// Jenkinsfile (Declarative Pipeline)
pipeline {
    agent any
    stages {
        stage('Example') {
            steps {
                echo 'Hello World'

                script {
                    def browsers = ['chrome', 'firefox']
                    for (int i = 0; i < browsers.size(); ++i) {
                        echo "Testing the ${browsers[i]} browser"
                    }
                }
            }
        }
    }
}
```


## Scripted pipeline

* [Pipeline Examples](https://jenkins.io/doc/pipeline/examples/)
  * 킹왕짱 예제들
* [젠킨스 파이프라인 정리 - 2. Scripted 문법 소개 @ tistory](https://jojoldu.tistory.com/356)
* [scripted-pipeline @ jenkins](https://jenkins.io/doc/book/pipeline/syntax/#scripted-pipeline)

----


Scripted Pipeline should start `node {...}` block. And It is effectively a general pupose DSL built with Groovy.

This is an example of Flow Control

```groovy
// Jenkinsfile (Scripted Pipeline)
node {
    stage('Example') {
        if (env.BRANCH_NAME == 'master') {
            echo 'I only execute on the master branch'
        } else {
            echo 'I execute elsewhere'
        }
    }
}
```

This is an example of try/catch. 

```groovy
// Jenkinsfile (Scripted Pipeline)
node {
    stage('Example') {
        try {
            sh 'exit 1'
        }
        catch (exc) {
            echo 'Something failed, I should sound the klaxons!'
            throw
        }
    }
}
```

# How to make a Jenkins-plugin

* [IntelliJ setup for Jenkins Plugin Development](https://medium.com/@baymac/setting-up-intellij-idea-for-jenkins-plugin-development-66a074bbe4a9)

----

* Configure Maven Settings
  * vim `~/.m2/settings.xml`

```xml
<settings>
  <pluginGroups>
    <pluginGroup>org.jenkins-ci.tools</pluginGroup>
  </pluginGroups>
 
  <profiles>
    <!-- Give access to Jenkins plugins -->
    <profile>
      <id>jenkins</id>
      <activation>
        <activeByDefault>true</activeByDefault> <!-- change this to false, if you don't like to have it on per default -->
      </activation>
      <repositories>
        <repository>
          <id>repo.jenkins-ci.org</id>
          <url>https://repo.jenkins-ci.org/public/</url>
        </repository>
      </repositories>
      <pluginRepositories>
        <pluginRepository>
          <id>repo.jenkins-ci.org</id>
          <url>https://repo.jenkins-ci.org/public/</url>
        </pluginRepository>
      </pluginRepositories>
    </profile>
  </profiles>
  <mirrors>
    <mirror>
      <id>repo.jenkins-ci.org</id>
      <url>https://repo.jenkins-ci.org/public/</url>
      <mirrorOf>m.g.o-public</mirrorOf>
    </mirror>
  </mirrors>
</settings>
```

* Generate an empty skeleton plugin

```bash
$ mvn archetype:generate -Dfilter=io.jenkins.archetypes:empty-plugin
$ cd <artiface-id>
$ idea pom.xml
```

* Add run/debug configuration

```bash
$ mvn install
$ mvn hpi:run
```

open browser with url `http://localhost:8080/jenkins`

* Next time you open your project

```bash
$ cd my/java/aaa
$ idea pom.xml
```

* Plugins to aid development
  * Stapler
  * Jenkins Control Plugin

# Script Console

* [Jenkins World 2017: Mastering the Jenkins Script Console](https://www.youtube.com/watch?v=qaUPESDcsGg)
* [Jenkins Area Meetup - Hacking on Jenkins Internals - Jenkins Script Console](https://www.youtube.com/watch?v=T1x2kCGRY1w)

----

* open browser with url `http://localhost:8080/script`

```groovy

```