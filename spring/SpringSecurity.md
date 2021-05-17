# Materials

* [스프링 프레임워크 핵심 기술 @ inflearn](https://www.inflearn.com/course/spring-framework_core/)
  * [src](https://github.com/keesun/spring-security-basic)
* [Spring boot - Spring Security(스프링 시큐리티) 란? 완전 해결!](https://coding-start.tistory.com/153)
* [Spring Security 를 유닛테스트 하기](https://velog.io/@jj362/Spring-Security-%EB%A5%BC-%EC%9C%A0%EB%8B%9B%ED%85%8C%EC%8A%A4%ED%8A%B8-%ED%95%98%EA%B8%B0)


# 스프링 시큐리티: 폼 인증

## 폼 인증 예제 살펴보기

다음과 같은 api 를 구현해 본다.

* `/` : 인증 여부에 따라 보여지는 화면이 다르다.
* `/info` : 인증과 상관없이 접근할 수 있다.
* `/dashboard` : 인증이 되어야만 볼 수 있다. 인증이 되어있지 않으면 login 화면이 등장한다.
* `/admin` : admin 권한이 있어야만 볼 수 있다.

## 스프링 웹 프로젝트 만들기

* [Created web project @ github](https://github.com/keesun/spring-security-basic/commit/9592611b3c297c476c4d17ee6aea74ab1a7c0ccf)

## 스프링 시큐리티 연동

* [Add Spring Security @ github](https://github.com/keesun/spring-security-basic/commit/ca7fc00e43e4bbe63957b071a6dce05a71075fcc)

## 스프링 시큐리티 설정하기

* [Add Spring Security Config @ github](https://github.com/keesun/spring-security-basic/commit/eb52f4cf88d7369a8503b200ea7955b3727f0f23)

## 스프링 시큐리티 커스터마이징: 인메모리 유저 추가

* [Add InMemory Users](https://github.com/keesun/spring-security-basic/commit/858354ef00218d482fd2185230e3304f7a09f1c0)

---

spring security 가 제공해주는 `user / xxxx` 의 password 는 매번 바뀐다. userid, password 를 hard coding 해보자.

## 스프링 시큐리티 커스터마이징: JPA 연동

* [Integrate JPA @ github](https://github.com/keesun/spring-security-basic/commit/9a242cc045c45f8e04fef121451f5c2f7e83c43f)

----

Account entity 를 정의하고 JPA 를 통해 읽어오자.

## 스프링 시큐리티 커스터마이징: PasswordEncoder

* [Add password encoder @ github](https://github.com/keesun/spring-security-basic/commit/e60160e42fafa8d8b5ef4045487748e9a93f5658)

-----

bcrypt PasswordEncoder Bean 을 생성하여 password 를 bcrypt 로 암호화하여 저장하자.

## 스프링 시큐리티 테스트 1부, 2부

* [Spring Security Test @ github](https://github.com/keesun/spring-security-basic/commit/6e3cf6ec62300e1b5a999cd89d1655c506ac6c0f)

----

`@RunWith(SpringRunner.class) @SpringBootTest @AutoConfigureMockMvc` 를 이용하여 spring security test code 를 작성한다.

`@WithMockUser(username = "iamslash", roles = "USER")` 가 반복되서 사용된다면 `public @interface WithUser` 를 정의하여 code duplicates 를 피할 수 있다.

# 스프링 시큐리티: 아키텍처

## SecurityContextHolder와 Authentication

* **SecurityContextHolder**
  * **SecurityContext** 를 제공한다.
* **SecurityContext**
  * **Authentication** 을 제공한다.
* **Authentication**
  * **Principal**, **GrantedAuthority** 를 제공한다.
* **Principal**
  * 누구에 해당하는 정보이다. **UserDetailsService** 에서 return 한 객체이다. 즉, **UserDetails** 와 같다.
* **GrantedAuthority**
  * **Principal** 이 가지고 있는 권한을 나타낸다. 예를 들어 "ROLE_USER, ROLE_ADMIN" 등이 있다.
* **UserDetailsService** 
  * **UserDetails** 를 return 하는 DAO interface 이다.

## AuthenticationManager와 Authentication

## ThreadLocal

## Authentication과 SecurityContextHodler

## 스프링 시큐리티 필터와 FilterChainProxy

## DelegatingFilterProxy와 FilterChainProxy

## AccessDecisionManager 1부

## AccessDecisionManager 2부

## FilterSecurityInterceptor

## ExceptionTranslationFilter

## 스프링 시큐리티 아키텍처 정리

* [Spring Security Architecture @ spring.io](https://spring.io/guides/topicals/spring-security-architecture)
* [8. Architecture and Implementation @ Spring Security Reference](https://docs.spring.io/spring-security/site/docs/5.1.5.RELEASE/reference/htmlsingle/#overall-architecture)

----

* **SecurityContextHolder**, to provide access to the **SecurityContext**.
* **SecurityContext**, to hold the **Authentication** and possibly request-specific security information.
* **Authentication**, to represent the principal in a Spring Security-specific manner.
* **GrantedAuthority**, to reflect the application-wide permissions granted to a principal.
* **UserDetails**, to provide the necessary information to build an **Authentication** object from your application’s DAOs or other source of security data.
* **UserDetailsService**, to create a **UserDetails** when passed in a String-based username (or certificate ID or the like).

# 웹 애플리케이션 시큐리티

## 스프링 시큐리티 ignoring() 1부
## 스프링 시큐리티 ignoring() 2부
## Async 웹 MVC를 지원하는 필터: WebAsyncManagerIntegrationFilter
## 스프링 시큐리티와 @Async
## SecurityContext 영속화 필터: SecurityContextPersistenceFilter
## 시큐리티 관련 헤더 추가하는 필터: HeaderWriterFilter
## CSRF 어택 방지 필터: CsrfFilter
## CSRF 토큰 사용 예제
## 로그아웃 처리 필터: LogoutFilter
## 폼 인증 처리 필터: UsernamePasswordAuthenticationFilter
## 로그인/로그아웃 폼 페이지 생성해주는 필터: DefaultLogin/LogoutPageGeneratingFilter
## 로그인/로그아웃 폼 커스터마이징
## Basic 인증 처리 필터: BasicAuthenticationFilter
## 요청 캐시 필터: RequestCacheAwareFilter
## 시큐리티 관련 서블릿 스팩 구현 필터: SecurityContextHolderAwareRequestFilter
## 익명 인증 필터: AnonymousAuthenticationFilter
## 세션 관리 필터: SessionManagementFilter
## 인증/인가 예외 처리 필터: ExceptionTranslationFilter
## 인가 처리 필터: FilterSecurityInterceptor
## 토큰 기반 인증 필터 : RememberMeAuthenticationFilter
## 커스텀 필터 추가하기

# 스프링 시큐리티 그밖에

## 타임리프 스프링 시큐리티 확장팩
## sec 네임스페이스
## 메소드 시큐리티
## @AuthenticationPrincipal
## 스프링 데이터 연동
## 스프링 시큐리티 마무리
