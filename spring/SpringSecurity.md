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

----

## 스프링 시큐리티 연동
## 스프링 시큐리티 설정하기
## 스프링 시큐리티 커스터마이징: 인메모리 유저 추가
## 스프링 시큐리티 커스터마이징: JPA 연동
## 스프링 시큐리티 커스터마이징: PasswordEncoder
## 스프링 시큐리티 테스트 1부
## 스프링 시큐리티 테스트 2부

# 스프링 시큐리티: 아키텍처

## SecurityContextHolder와 Authentication
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
