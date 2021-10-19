- [Abstract](#abstract)
- [Materials](#materials)
- [Basics](#basics)

----

# Abstract

Maven Plugin 을 publish 하는 방법을 정리한다.

# Materials

* [Publishing a project as module @ gradle](https://docs.gradle.org/current/userguide/publishing_setup.html)
* [Maven Publish Plugin](https://docs.gradle.org/current/userguide/publishing_maven.html)

# Basics

먼저 `build.gradle.kts` 에 `maven-publish-plugin` 을 적용한다.

```kotlin
// build.gradle.kts
plugins {
    `maven-publish`
}
```

다음과 같은 task 들이 생성된다.

* generatePomFileForPubNamePublication
* publishPubNamePublicationToRepoNameRepository
* publishPubNamePublicationToMavenLocal
* publish
* publishToMavenLocal

또한 `publications` 라는 project extention 을 제공한다. `publications` 는 `publications, repositories` 와 같은 container 들을 제공한다.

이제 다음과 같이 `publications` 를 customizing 해보자. `groupId, artifactId, version` 을 customizing 해보자.

```kotlin
// build.gradle.kts
publishing {
    publications {
        create<MavenPublication>("maven") {
            groupId = "org.gradle.sample"
            artifactId = "library"
            version = "1.1"

            from(components["java"])
        }
    }
}
```

또한 다음과 같이 `POM` 을 customizing 할 수도 있다.

```kotlin
// build.gradle.kts
publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            pom {
                name.set("My Library")
                description.set("A concise description of my library")
                url.set("http://www.iamslash.com/library")
                properties.set(mapOf(
                    "myProp" to "value",
                    "prop.with.dots" to "anotherValue"
                ))
                licenses {
                    license {
                        name.set("The Apache License, Version 2.0")
                        url.set("http://www.apache.org/licenses/LICENSE-2.0.txt")
                    }
                }
                developers {
                    developer {
                        id.set("davidsun")
                        name.set("David Sun")
                        email.set("iamslash@gmail.com")
                    }
                }
                scm {
                    connection.set("scm:git:git://iamslash.com/my-library.git")
                    developerConnection.set("scm:git:ssh://iamslash.com/my-library.git")
                    url.set("http://iamslash.com/my-library/")
                }
            }
        }
    }
}
```

resolved versions 는 뭐지???

```kotlin
// build.gradle.kts
publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            versionMapping {
                usage("java-api") {
                    fromResolutionOf("runtimeClasspath")
                }
                usage("java-runtime") {
                    fromResolutionResult()
                }
            }
        }
    }
}
```

이제 `repositories` 를 customizing 해보자. 먼저 배포할 url 을 설정한다.

```kotlin
// build.gradle.kts
publishing {
    repositories {
        maven {
            // change to point to your repo, e.g. http://my.org/repo
            url = uri(layout.buildDirectory.dir("repo"))
        }
    }
}
```

보통 libary 의 version 은 snapshot, release 와 같이 두가지 형태로 관리한다. 다음은 version 문자열이 `SNAPSHOT` 으로 끝나면 snapshot repo 에 publishing 하고 그렇지 않으면 release repo 에 publishing 하는 예이다.

```kotlin
// build.gradle.kts
publishing {
    repositories {
        maven {
            val releasesRepoUrl = layout.buildDirectory.dir("repos/releases")
            val snapshotsRepoUrl = layout.buildDirectory.dir("repos/snapshots")
            url = uri(if (version.toString().endsWith("SNAPSHOT")) snapshotsRepoUrl else releasesRepoUrl)
        }
    }
}
```

또는 project 혹은 system property 에 release 라는 property 가 있는지에 따라 repo url 을 다르게 설정할 수도 있다.

```kotlin
// publishing {
    repositories {
        maven {
            val releasesRepoUrl = layout.buildDirectory.dir("repos/releases")
            val snapshotsRepoUrl = layout.buildDirectory.dir("repos/snapshots")
            url = uri(if (project.hasProperty("release")) releasesRepoUrl else snapshotsRepoUrl)
        }
    }
}
```

만약 Maven Local 에 publishing 하면 보통 `$USER_HOME/.m2/repository` 에 저장된다.

```bash
./gradlew publishMavenToLocal
```

다음은 java library 를 publishing 하는 예이다. sources, JavaDoc, customized POM 이 포함된다. 

```kotlin
// build.gradle.kts
plugins {
    `java-library`
    `maven-publish`
    signing
}

group = "com.example"
version = "1.0"

java {
    withJavadocJar()
    withSourcesJar()
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            artifactId = "my-library"
            from(components["java"])
            versionMapping {
                usage("java-api") {
                    fromResolutionOf("runtimeClasspath")
                }
                usage("java-runtime") {
                    fromResolutionResult()
                }
            }
            pom {
                name.set("My Library")
                description.set("A concise description of my library")
                url.set("http://www.iamslash.com/library")
                properties.set(mapOf(
                    "myProp" to "value",
                    "prop.with.dots" to "anotherValue"
                ))
                licenses {
                    license {
                        name.set("The Apache License, Version 2.0")
                        url.set("http://www.apache.org/licenses/LICENSE-2.0.txt")
                    }
                }
                developers {
                    developer {
                        id.set("davidsun")
                        name.set("David Sun")
                        email.set("iamslash@gmail.com")
                    }
                }
                scm {
                    connection.set("scm:git:git://iamslash.com/my-library.git")
                    developerConnection.set("scm:git:ssh://iamslash.com/my-library.git")
                    url.set("http://iamslash.com/my-library/")
                }
            }
        }
    }
    repositories {
        maven {
            // change URLs to point to your repos, e.g. http://my.org/repo
            val releasesRepoUrl = uri(layout.buildDirectory.dir("repos/releases"))
            val snapshotsRepoUrl = uri(layout.buildDirectory.dir("repos/snapshots"))
            url = if (version.toString().endsWith("SNAPSHOT")) snapshotsRepoUrl else releasesRepoUrl
        }
    }
}

signing {
    sign(publishing.publications["mavenJava"])
}

tasks.javadoc {
    if (JavaVersion.current().isJava9Compatible) {
        (options as StandardJavadocDocletOptions).addBooleanOption("html5", true)
    }
}
```

위의 build script 는 다음과 같은 파일들을 생성할 것이다.

* The POM: `my-libary-1.0.pom`
* The primary JAR artifact: `my-library-1.0.jar`
* The sources JAR artifact: `my-library-1.0-sources.jar`
* The Javadoc JAR artifact: `my-library-1.0-javadoc.jar`
