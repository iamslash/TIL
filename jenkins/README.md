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

# Install with docker on windows10

```bash
docker pull jenkins/jenkins:lts
docker run -d -p 50000:50000 -p 8080:8080 -v C:/my/dockervolume/var/jenkins_home:/var/jenkins_home --name my-jenkins jenkins/jenkins:lts
```

browser 로 `localhost:8080` 으로 접속한다. docker 실행창에 출력된 key 를 입력한다. install suggested plugins 하면 끝. 플러그인 설치를 실패할 때가 있다. 그렇다면 `C:/my/dockervolume/var/jenkins_home/*` 를 모두 지우고 `docker stop, rm` 이후 다시 실행해본다. 잘 된다.

# Pipeline

* [Using a Jenkinsfile ](https://jenkins.io/doc/book/pipeline/jenkinsfile/)

----

`Jenkinsfile` 이라는 이름의 text file 이다. groovy 로 작성한다. `Declarative Pipeline`, `Scripted Pipeline` 과 같은 두가지 형식으로 작성한다.

```groovy
// Jenkinsfile (Declarative Pipeline)
pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                echo 'Building..'
            }
        }
        stage('Test') {
            steps {
                echo 'Testing..'
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying....'
            }
        }
    }
}
// Jenkinsfile (Scripted Pipeline)
node {
    checkout scm 
    /* .. snip .. */
}
```

다음은 Build Automation 을 위한 Jenkinsfile 의 예이다.

```groovy
Jenkinsfile (Declarative Pipeline)
pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                sh 'make' 
                archiveArtifacts artifacts: '**/target/*.jar', fingerprint: true 
            }
        }
    }
}
```

다음은 Test Automation 을 위한 Jenkinsfile 의 예이다.

```groovy
// Jenkinsfile (Declarative Pipeline)
pipeline {
    agent any

    stages {
        stage('Test') {
            steps {
                /* `make check` returns non-zero on test failures,
                * using `true` to allow the Pipeline to continue nonetheless
                */
                sh 'make check || true' 
                junit '**/target/*.xml' 
            }
        }
    }
}
```

다음은 Deploy Automation 을 위한 Jenkinsfile 의 예이다.

```groovy
// Jenkinsfile (Declarative Pipeline)
pipeline {
    agent any

    stages {
        stage('Deploy') {
            when {
              expression {
                currentBuild.result == null || currentBuild.result == 'SUCCESS' 
              }
            }
            steps {
                sh 'make publish'
            }
        }
    }
}
```