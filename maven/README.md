# Materials

* [Maven 정복](https://wikidocs.net/17298)
* [maven archetype plugin - 템플릿에서 메이븐 프로젝트 생성하기](https://www.lesstif.com/pages/viewpage.action?pageId=21430332)
* [메이븐(Maven) 강의 @ youtube](https://www.youtube.com/watch?v=VAp0n9DmeEA&list=PLq8wAnVUcTFWRRi_JWLArMND_PnZM6Yja)
* [Maven in 5 Minutes @ apache](https://maven.apache.org/guides/getting-started/maven-in-five-minutes.html)

# Basic

## Install

```bash
$ sdk install maven
```

## How to build and run

```bash
$ mvn --version
# create a project
$ mvn archetype:generate -DgroupId=com.iamslash -DartifacId=foo -DarchetypeArtifactId=maven-archetype-quickstart
# compile the project
$ mvn compile
# build the project
$ mvn package
# run the project
$ java -cp target/Hello-1.0-SNAPSHOT.jar com.iamslash.App
```

## BOM 

* [Maven의 Transitive Dependency 길들이기](https://blog.sapzil.org/2018/01/21/taming-maven-transitive-dependencies/)
  
Bom stands for bill of materials. Bom 을 적용하면 version 을 강제할 수 있다.

## POM

POM stands for project of model. Maven 은 pom.xml 을 통해 task 를 정의한다.

* [메이븐(Maven)은 알고 스프링(Spring)을 쓰는가? (pom.xml 분석하며 가볍게 정리하는 빌드 툴, Maven)](https://jeong-pro.tistory.com/168)
