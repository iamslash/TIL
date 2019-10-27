# Materials

* [maven archetype plugin - 템플릿에서 메이븐 프로젝트 생성하기](https://www.lesstif.com/pages/viewpage.action?pageId=21430332)
* [메이븐(Maven) 강의 @ youtube](https://www.youtube.com/watch?v=VAp0n9DmeEA&list=PLq8wAnVUcTFWRRi_JWLArMND_PnZM6Yja)
* [Maven in 5 Minutes @ apache](https://maven.apache.org/guides/getting-started/maven-in-five-minutes.html)

# Basic

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