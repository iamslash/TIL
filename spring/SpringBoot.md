# Materials

* [스프링 부트 개념과 활용 @ inflearn](https://www.inflearn.com/course/%EC%8A%A4%ED%94%84%EB%A7%81%EB%B6%80%ED%8A%B8)

## Tutorial of STS

- [Spring Tool Suite](https://spring.io/tools)를 설치한다.
- STS를 시작하고 File | New | Spring Starter Project를 선택하고 적당히 설정하자.
  - com.iamslash.firstspring
- 다음과 같은 파일을 com.iamslash.firstspring에 추가하자.

```java
package com.iamslash.firstspring;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class A {
	@RequestMapping("/")
	public String index() {
		return "helloworld!";
	}
}
```
- firstspring을 R-click후에 Run As | Spring Boot App선택해서 실행하자.
- 브라우저를 이용하여 http://localhost:8080으로 접속하자.

## Tutorial of springboot 

- [springboot manual installation](https://docs.spring.io/spring-boot/docs/current/reference/html/getting-started-installing-spring-boot.html#getting-started-manual-cli-installation)
  에서 zip을 다운받아 `d:\local\src\`에 압축을 해제하자.
- 환경설정변수 SPRING_HOME을 만들자. `d:\local\src\D:\local\src\spring-1.5.6.RELEASE`
- 환경설정변수 PATH에 `%SPRING_HOME%\bin`을 추가하자.
- command shell에서 `spring version`을 실행한다.
- command shell에서 `spring init'을 이용하여 새로운 프로젝트를 제작할 수 있다.


# 스프링 부트 원리

## 의존성 관리 이해

## 의존성 관리 응용

## 자동 설정 이해

## 자동설정 만들기 1 부 : Starter 와 AutoConfigure

## 자동설정 만들기 2 부 : @ConfigurationProperties

## 내장 웹 서버 이해

## 내장 웹 서버 응용 1 부: 컨테이너와 포트

## 내장 웹 서버 응용 2 부: HTTPS 와 HTTP2

## 톰캣 HTTP2

## 독립적으로 실행가능한 JAR

## 스프링 부트 원리 정리

# 스프링 부트 활용

## 스프링 부트 활용 소개

## SpringApplication 1 부

## SpringApplication 2 부

## 외부 설정 1 부

## 외부 설정 2 부 (1)

## 외부 설정 2 부 (2)

## 프로파일

## 로깅 1부 : 스프링 부트 기본 로거설정

## 로깅 2부 : 커스터마이징

## 테스트
## 테스트 유틸
## Spring-Boot-Devtools
## 스프링 웹 MVC 1 부: 소개
## 스프링 웹 MVC 2 부: HttpMessageconverters
## 스프링 웹 MVC 3 부: ViewResolve
## 스프링 웹 MVC 4 부: 정적 리소스 지원
## 스프링 웹 MVC 5 부: 웹 JAR
## 스프링 웹 MVC 6 부: index 페이지와 파비콘
## 스프링 웹 MVC 7 부: Thymeleaf
## 스프링 웹 MVC 8 부: HtmlUnit
## 스프링 웹 MVC 9 부: ExceptionHandler
## 스프링 웹 MVC 10 부: Spring HATEOAS
## 스프링 웹 MVC 11 부: CORS

## 스프링 데이터 1 부: 소개
## 스프링 데이터 2 부: 인메모리 데이터베이스
## 스프링 데이터 3 부: MySQL
## 스프링 데이터 4 부: PostgreSQL
## 스프링 데이터 5 부: 스프링 데이터 JPA 소개
## 스프링 데이터 6 부: 스프링 데이터 JPA 연동
## 스프링 데이터 7 부: 데이터베이스 초기화
## 스프링 데이터 8 부: 데이터베이스 마이그레이션
## 스프링 데이터 9 부: Redis
## 스프링 데이터 10 부: MongoDB
## 스프링 데이터 11 부: Neo4J
## 스프링 데이터 12 부: 정리

## 스프링 시큐리티 1 부: StarterSecurity
## 스프링 시큐리티 2 부: 시큐리티 설정 커스터마이징

## 스프링 REST 클라이언트 1 부: RestTemplate vs WebClient
## 스프링 REST 클라이언트 2 부: Customizing
## 그밖에 다양한 기술 연동

# 스프링 부트 운영

## 스프링 부트 Actuator 1 부: 소개

## 스프링 부트 Actuator 2 부: JMX 와 HTTP

## 스프링 부트 Actuator 3 부: 스프링 부트 어드민

