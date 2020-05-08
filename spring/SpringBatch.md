# Materials

* Jojoldu Spring Batch Guide
  * [src](https://github.com/jojoldu/spring-batch-in-action)
  * [1. Spring Batch 가이드 - 배치 어플리케이션이란?](https://jojoldu.tistory.com/324?category=902551)
  * [2. Spring Batch 가이드 - Batch Job 실행해보기](https://jojoldu.tistory.com/325)
  * [3. Spring Batch 가이드 - 메타테이블엿보기](https://jojoldu.tistory.com/326?category=902551)
  * [4. Spring Batch 가이드 - Spring Batch Job Flow](https://jojoldu.tistory.com/328?category=902551)
  * [5. Spring Batch 가이드 - Spring Batch Scope & Job Parameter](https://jojoldu.tistory.com/330?category=902551)
  * [6. Spring Batch 가이드 - Chunk 지향 처리](https://jojoldu.tistory.com/331?category=902551)
    * [Spring Batch에서 영속성 컨텍스트 문제 (processor에서 lazyException 발생할때)](https://jojoldu.tistory.com/146)
    * [Spring Batch Paging Reader 사용시 같은 조건의 데이터를 읽고 수정할때 문제](https://jojoldu.tistory.com/337?category=902551)
  * [7. Spring Batch 가이드 - ItemReader](https://jojoldu.tistory.com/336?category=902551)
  * [8. Spring Batch 가이드 - ItemWriter](https://jojoldu.tistory.com/339?category=902551)
  * [9. Spring Batch 가이드 - ItemProcessor](https://jojoldu.tistory.com/347?category=902551)
    * [Spring batch & JPA에서 N+1 문제 해결](https://jojoldu.tistory.com/414?category=902551)
    * [Spring Batch 공통 설정 관리하기 (feat. 젠킨스 Environment variables)](https://jojoldu.tistory.com/445?category=902551)
    * [3. AWS Code Deploy로 배포 Jenkins에서 배치 Jenkins로 Spring Batch 배포하기 - 젠킨스 연동](https://jojoldu.tistory.com/445?category=902551)
    * [Spring Batch의 멱등성 유지하기](https://jojoldu.tistory.com/451?category=902551)
  * [10. Spring Batch 가이드 - Spring Batch 테스트 코드](https://jojoldu.tistory.com/455?category=902551)
    * [10.1. Spring Batch 단위 테스트 코드 - Reader 편](https://jojoldu.tistory.com/456?category=902551)
  * [Spring Batch와 QuerydslItemReader](https://jojoldu.tistory.com/473?category=902551)
  * [Spring Batch의 유니크 Job Parameter 활용하기](https://jojoldu.tistory.com/487?category=902551)
  * [Spring Batch 관리 도구로서의 Jenkins](https://jojoldu.tistory.com/489?category=902551)
  * [JobParameter 활용 방법 (feat. LocalDate 파라미터 사용하기)](https://jojoldu.tistory.com/490?category=902551)

* [Spring Batch - Reference Documentation](https://docs.spring.io/spring-batch/docs/current/reference/html/index.html)

# Basic

## Concept

Spring batch 는 job 과 step 으로 구성된다. 하나의 job 은 여러 step 들로 구성된다. 하나의 step 은 여러 tasklet 으로 구성된다. 특히 ChuckOrientedTasklet 은 chuck 단위로 처리하고 ItemReader, ItemWriter, ItemProcessor 로 구성된다. 이때 ItemProcessor 는 생략할 수 있다.

![](img/spring-batch-reference-model.png)

## Simple Spring Batch

* [exbatch](https://github.com/iamslash/spring-examples/exbatch)

-----

build.gradle

```gradle
plugins {
	id 'org.springframework.boot' version '2.2.7.RELEASE'
	id 'io.spring.dependency-management' version '1.0.9.RELEASE'
	id 'java'
	id 'eclipse'
}

group = 'com.iamslash'
version = '0.0.1-SNAPSHOT'
sourceCompatibility = '1.8'

repositories {
	mavenCentral()
}

dependencies {
	implementation 'org.springframework.boot:spring-boot-starter-batch'
	implementation 'org.springframework.boot:spring-boot-starter-data-jpa'
	implementation 'org.springframework.boot:spring-boot-starter-data-jdbc'
	implementation 'com.h2database:h2'
	implementation 'mysql:mysql-connector-java'
	implementation 'org.projectlombok:lombok'
	testImplementation('org.springframework.boot:spring-boot-starter-test') {
		exclude group: 'org.junit.vintage', module: 'junit-vintage-engine'
	}
	testImplementation('org.springframework.batch:spring-batch-test')
}

test {
	useJUnitPlatform()
}
```

다음과 같이 Application class 에 `@EnableBatchProcessing` 을 추가한다.

```java
@EnableBatchProcessing
@SpringBootApplication
public class ExbatchApplication {

	public static void main(String[] args) {
		SpringApplication.run(ExbatchApplication.class, args);
	}

}
```


