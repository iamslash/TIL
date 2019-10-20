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

# jenkin_home structure

## directories

| directory | description |
|-----------|-------------|
| `/jobs` | list of jobs |
| `/nodes` | list of nodes |
| `/logs` | log files |
| `/plugins` | list of plugins |
| `/workspace` | list of job worksspacees |

## files

| directory | description |
|-----------|-------------|
| `/config.xml` | global configuration |

# Build Now Process

* make workspace direcotry at `/jenkins_home/worksspace/<job-name>`
* execute Pipeline script

# Setting

## Locale 

* MENU | Manage Plugins | Install Locale plugin
* MENU | Configure System
* Locale | Default Language | en or ENGLISH

## GitHub secret text

* [gitHub와 Jenkins 연결하기](https://webcache.googleusercontent.com/search?q=cache:P6VRZNmJqRkJ:https://bcho.tistory.com/1237+&cd=1&hl=ko&ct=clnk&gl=kr)

# Basic

## Simple Job with Pipeline value

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

## Simple Job with pipeline script from scm

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

# Pipeline as a code

* [Learning Jenkins Pipeline @ github](https://github.com/mcpaint/learning-jenkins-pipeline)
  * declaritive pipeline script 연구
* [Jenkinsfile 을 이용한 젠킨스 Pipeline 설정](https://limsungmook.github.io/2016/11/09/jenkins-pipeline/)
* [Pipeline as Code with Jenkins](https://jenkins.io/solutions/pipeline/)
* [Using a Jenkinsfile ](https://jenkins.io/doc/book/pipeline/jenkinsfile/)

----

`Jenkinsfile` 이라는 이름의 text file 이다. repository root 에 groovy 로 작성한다. `Declarative Pipeline`, `Scripted Pipeline` 과 같은 두가지 형식으로 작성한다. 

browser 로 `http://localhost:8080/pipeline-syntax` 를 접속하면 자세한 reference 들을 확인할 수 있다.

# Declaritive pipeline

## skeleton

```groovy
pipeline {
    agent {}
    triggers {}
    tools {}
    environment {}
    options {}
    parameters {}
    stages {
        stage('stage1') {}
        stage('stage2') {}
        
        parallel { 
            stage('parallel_1') {}
            stage('parallel_2') {}
        }
    }
    
    // execute after stages
    post {
      always {}
      changed {}
      fixed {}
      regression {}
      aborted {}
      failure {}
      success {}
      unstable {}
      cleanup {}
    }
}
```

## example 1


# Scripted pipeline

* [Pipeline Examples](https://jenkins.io/doc/pipeline/examples/)
  * 킹왕짱 예제들
* [젠킨스 파이프라인 정리 - 2. Scripted 문법 소개 @ tistory](https://jojoldu.tistory.com/356)
* [scripted-pipeline @ jenkins](https://jenkins.io/doc/book/pipeline/syntax/#scripted-pipeline)

----

# Contribution

* [Beginners Guide to Contributing](https://wiki.jenkins.io/display/JENKINS/Beginners+Guide+to+Contributing)