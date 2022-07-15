- [Materials](#materials)
- [Prerequisites](#prerequisites)
	- [Spring Security Architecture](#spring-security-architecture)
		- [A Review of Filters](#a-review-of-filters)
		- [DelegatingFilterProxy](#delegatingfilterproxy)
		- [FilterChainProxy](#filterchainproxy)
		- [SecurityFilterChain](#securityfilterchain)
		- [Security Filters](#security-filters)
		- [Handling Security Exceptions](#handling-security-exceptions)
	- [Servlet Authentication Architecture](#servlet-authentication-architecture)
		- [SecurityContextHolder](#securitycontextholder)
		- [AuthenticationManager](#authenticationmanager)
		- [ProviderManager](#providermanager)
		- [**Request Credentials with `AuthenticationEntryPoint`**](#request-credentials-withauthenticationentrypoint)
		- [AbstractAuthenticationProcessingFilter](#abstractauthenticationprocessingfilter)
	- [Servlet Authorization Architecture](#servlet-authorization-architecture)
		- [Authorities](#authorities)
		- [Delegate-based AuthorizationManager implements](#delegate-based-authorizationmanager-implements)
		- [Adapting AccessDecisionManager and AccessDecisionVoters](#adapting-accessdecisionmanager-and-accessdecisionvoters)
		- [Hierarchical Roles](#hierarchical-roles)
- [스프링 시큐리티: 폼 인증](#스프링-시큐리티-폼-인증)
	- [폼 인증 예제 살펴보기](#폼-인증-예제-살펴보기)
	- [스프링 웹 프로젝트 만들기](#스프링-웹-프로젝트-만들기)
	- [스프링 시큐리티 연동](#스프링-시큐리티-연동)
	- [스프링 시큐리티 설정하기](#스프링-시큐리티-설정하기)
	- [스프링 시큐리티 커스터마이징: 인메모리 유저 추가](#스프링-시큐리티-커스터마이징-인메모리-유저-추가)
	- [스프링 시큐리티 커스터마이징: JPA 연동](#스프링-시큐리티-커스터마이징-jpa-연동)
	- [스프링 시큐리티 커스터마이징: PasswordEncoder](#스프링-시큐리티-커스터마이징-passwordencoder)
	- [스프링 시큐리티 테스트 1부, 2부](#스프링-시큐리티-테스트-1부-2부)
- [스프링 시큐리티: 아키텍처](#스프링-시큐리티-아키텍처)
	- [SecurityContextHolder와 Authentication](#securitycontextholder와-authentication)
	- [AuthenticationManager와 Authentication](#authenticationmanager와-authentication)
	- [ThreadLocal](#threadlocal)
	- [Authentication과 SecurityContextHodler](#authentication과-securitycontexthodler)
	- [스프링 시큐리티 필터와 FilterChainProxy](#스프링-시큐리티-필터와-filterchainproxy)
	- [DelegatingFilterProxy와 FilterChainProxy](#delegatingfilterproxy와-filterchainproxy)
	- [AccessDecisionManager 1부](#accessdecisionmanager-1부)
	- [AccessDecisionManager 2부](#accessdecisionmanager-2부)
	- [FilterSecurityInterceptor](#filtersecurityinterceptor)
	- [ExceptionTranslationFilter](#exceptiontranslationfilter)
	- [스프링 시큐리티 아키텍처 정리](#스프링-시큐리티-아키텍처-정리)
- [웹 애플리케이션 시큐리티](#웹-애플리케이션-시큐리티)
	- [스프링 시큐리티 ignoring() 1부](#스프링-시큐리티-ignoring-1부)
	- [스프링 시큐리티 ignoring() 2부](#스프링-시큐리티-ignoring-2부)
	- [Async 웹 MVC를 지원하는 필터: WebAsyncManagerIntegrationFilter](#async-웹-mvc를-지원하는-필터-webasyncmanagerintegrationfilter)
	- [스프링 시큐리티와 @Async](#스프링-시큐리티와-async)
	- [SecurityContext 영속화 필터: SecurityContextPersistenceFilter](#securitycontext-영속화-필터-securitycontextpersistencefilter)
	- [시큐리티 관련 헤더 추가하는 필터: HeaderWriterFilter](#시큐리티-관련-헤더-추가하는-필터-headerwriterfilter)
	- [CSRF 어택 방지 필터: CsrfFilter](#csrf-어택-방지-필터-csrffilter)
	- [CSRF 토큰 사용 예제](#csrf-토큰-사용-예제)
	- [로그아웃 처리 필터: LogoutFilter](#로그아웃-처리-필터-logoutfilter)
	- [폼 인증 처리 필터: UsernamePasswordAuthenticationFilter](#폼-인증-처리-필터-usernamepasswordauthenticationfilter)
	- [로그인/로그아웃 폼 페이지 생성해주는 필터: DefaultLogin/LogoutPageGeneratingFilter](#로그인로그아웃-폼-페이지-생성해주는-필터-defaultloginlogoutpagegeneratingfilter)
	- [로그인/로그아웃 폼 커스터마이징](#로그인로그아웃-폼-커스터마이징)
	- [Basic 인증 처리 필터: BasicAuthenticationFilter](#basic-인증-처리-필터-basicauthenticationfilter)
	- [요청 캐시 필터: RequestCacheAwareFilter](#요청-캐시-필터-requestcacheawarefilter)
	- [시큐리티 관련 서블릿 스팩 구현 필터: SecurityContextHolderAwareRequestFilter](#시큐리티-관련-서블릿-스팩-구현-필터-securitycontextholderawarerequestfilter)
	- [익명 인증 필터: AnonymousAuthenticationFilter](#익명-인증-필터-anonymousauthenticationfilter)
	- [세션 관리 필터: SessionManagementFilter](#세션-관리-필터-sessionmanagementfilter)
	- [인증/인가 예외 처리 필터: ExceptionTranslationFilter](#인증인가-예외-처리-필터-exceptiontranslationfilter)
	- [인가 처리 필터: FilterSecurityInterceptor](#인가-처리-필터-filtersecurityinterceptor)
	- [토큰 기반 인증 필터 : RememberMeAuthenticationFilter](#토큰-기반-인증-필터--remembermeauthenticationfilter)
	- [커스텀 필터 추가하기](#커스텀-필터-추가하기)
- [스프링 시큐리티 그밖에](#스프링-시큐리티-그밖에)
	- [타임리프 스프링 시큐리티 확장팩](#타임리프-스프링-시큐리티-확장팩)
	- [sec 네임스페이스](#sec-네임스페이스)
	- [메소드 시큐리티](#메소드-시큐리티)
	- [@AuthenticationPrincipal](#authenticationprincipal)
	- [스프링 데이터 연동](#스프링-데이터-연동)

----

# Materials

* [Spring Security | baeldung](https://www.baeldung.com/category/spring/spring-security/)
  * [Spring Security Authorization | baeldung](https://www.baeldung.com/category/spring-security/tag/authorization/)
* [스프링 시큐리티 - Spring Boot 기반으로 개발하는 Spring Security | inflearn](https://www.inflearn.com/course/%EC%BD%94%EC%96%B4-%EC%8A%A4%ED%94%84%EB%A7%81-%EC%8B%9C%ED%81%90%EB%A6%AC%ED%8B%B0)
  * [src](https://github.com/onjsdnjs/corespringsecurityfinal)
* [Spring Security | spring.io](https://docs.spring.io/spring-security/reference/index.html)
* [스프링 프레임워크 핵심 기술 | inflearn](https://www.inflearn.com/course/spring-framework_core/)
  * [src](https://github.com/keesun/spring-security-basic)
* [Spring boot - Spring Security(스프링 시큐리티) 란? 완전 해결!](https://coding-start.tistory.com/153)
* [Spring Security 를 유닛테스트 하기](https://velog.io/@jj362/Spring-Security-%EB%A5%BC-%EC%9C%A0%EB%8B%9B%ED%85%8C%EC%8A%A4%ED%8A%B8-%ED%95%98%EA%B8%B0)

# Prerequisites

다음의 문서를 잘 읽고 Spring Security Architecture, Authentication Architecture, Authorization Architecture 항목을 그림위주로 이해한다.

- [Spring Security | spring.io](https://docs.spring.io/spring-security/reference/index.html)
    - [samples](https://docs.spring.io/spring-security/reference/samples.html)

## Spring Security Architecture

### A Review of Filters

Spring 은 다음과 같이 HTTP Request 를 Filter 들을 지나 Servlet 으로 전달된다.

![](https://docs.spring.io/spring-security/reference/_images/servlet/architecture/filterchain.png)

Filter 의 doFilter 는 다음과 같이 구현한다. `chain.doFilter()` 를 호출하여 다음 Filter 로 HTTP Request 를 전달한다. 

```java
fun doFilter(request: ServletRequest, response: ServletResponse, chain: FilterChain) {
    // do something before the rest of the application
    chain.doFilter(request, response) // invoke the rest of the application
    // do something after the rest of the application
}
```

### DelegatingFilterProxy

DelegatingFilterProxy 를 통해 Filter 로 전달된 HTTP Request 를 다른 곳으로 전달할 수 있다.

![](https://docs.spring.io/spring-security/reference/_images/servlet/architecture/delegatingfilterproxy.png)

다음은 `DelegatingFilterProxy` 의 pseudo code 이다.

```java
fun doFilter(request: ServletRequest, response: ServletResponse, chain: FilterChain) {
	// Lazily get Filter that was registered as a Spring Bean
	// For the example in DelegatingFilterProxy 
	// delegate is an instance of Bean Filter0
	val delegate: Filter = getFilterBean(someBeanName)
	// delegate work to the Spring Bean
	delegate.doFilter(request, response)
}
```

### FilterChainProxy

FilterChainProxy 는 Spring Security 가 제공하는 특별한 Filter 이다. FilterChainProxy 는 HTTP Request 를 SecurityFilterChain 의 Filter 들로 전달한다.

![](https://docs.spring.io/spring-security/reference/_images/servlet/architecture/filterchainproxy.png)


### SecurityFilterChain

SecurityFilterChain 은 Security Filter 들로 구성된다.

![](https://docs.spring.io/spring-security/reference/_images/servlet/architecture/securityfilterchain.png)

다음과 같이 Multiple SecurityFilterChain 도 가능하다.

![](https://docs.spring.io/spring-security/reference/_images/servlet/architecture/multi-securityfilterchain.png)

### Security Filters

다음은 Security Filter 들이다.

* ForceEagerSessionCreationFilter
* ChannelProcessingFilter
* WebAsyncManagerIntegrationFilter
* SecurityContextPersistenceFilter
* HeaderWriterFilter
* CorsFilter
* CsrfFilter
* LogoutFilter
* OAuth2AuthorizationRequestRedirectFilter
* Saml2WebSsoAuthenticationRequestFilter
* X509AuthenticationFilter
* AbstractPreAuthenticatedProcessingFilter
* CasAuthenticationFilter
* OAuth2LoginAuthenticationFilter
* Saml2WebSsoAuthenticationFilter
* UsernamePasswordAuthenticationFilter
* OpenIDAuthenticationFilter
* DefaultLoginPageGeneratingFilter
* DefaultLogoutPageGeneratingFilter
* ConcurrentSessionFilter
* DigestAuthenticationFilter
* BearerTokenAuthenticationFilter
* BasicAuthenticationFilter
* RequestCacheAwareFilter
* SecurityContextHolderAwareRequestFilter
* JaasApiIntegrationFilter
* RememberMeAuthenticationFilter
* AnonymousAuthenticationFilter
* OAuth2AuthorizationCodeGrantFilter
* SessionManagementFilter
* ExceptionTranslationFilter
* FilterSecurityInterceptor
* SwitchUserFilter

### Handling Security Exceptions

ExceptionTranslationFilter 는 Exception 을 처리한다.

![](https://docs.spring.io/spring-security/reference/_images/servlet/architecture/exceptiontranslationfilter.png)

다음은 ExceptionTranslationFilter 의 pseudo code 이다.

```java
try {
	filterChain.doFilter(request, response);
} catch (AccessDeniedException | AuthenticationException ex) {
	if (!authenticated || ex instanceof AuthenticationException) {
		startAuthentication();
	} else {
		accessDenied();
	}
}
```

## Servlet Authentication Architecture

### SecurityContextHolder

SecurityContextHolder 는 SecurityContext, Authentication, Principal, Credentials, Authorities 를 소유한다.

![](https://docs.spring.io/spring-security/reference/_images/servlet/authentication/architecture/securitycontextholder.png)

### AuthenticationManager

**AuthenticationManager** 는 인증을 담당하는 interface 이다. 주로 **ProviderManager** 가 구현한다.

### ProviderManager

하나의 ProviderManager 에 여러개의 AuthenticationProvider 를 inject 할 수 있다.

![](https://docs.spring.io/spring-security/reference/_images/servlet/authentication/architecture/providermanager.png)

![](https://docs.spring.io/spring-security/reference/_images/servlet/authentication/architecture/providermanager-parent.png)

![](https://docs.spring.io/spring-security/reference/_images/servlet/authentication/architecture/providermanagers-parent.png)

### **Request Credentials with `AuthenticationEntryPoint`**

Client 가 Credentials 를 전송할 수 있도록 Redirect 한다.

### AbstractAuthenticationProcessingFilter

![](https://docs.spring.io/spring-security/reference/_images/servlet/authentication/architecture/abstractauthenticationprocessingfilter.png)

## Servlet Authorization Architecture

### Authorities

Authorities 는 `getAuthority()` 로 구별하자. **GrantedAuthority** 는 다음과 같은 함수를 갖는다.

```java
String getAuthority();
```

### Delegate-based AuthorizationManager implements

![](https://docs.spring.io/spring-security/reference/_images/servlet/authorization/authorizationhierarchy.png)

### Adapting AccessDecisionManager and AccessDecisionVoters

AuthorizationManager 이 추천인 듯. AccessDecisionManager, AccessDecisionVoters 를 위한 Adapter 예제가 있음.

```java
@Component
public class AccessDecisionManagerAuthorizationManagerAdapter implements AuthorizationManager {
    private final AccessDecisionManager accessDecisionManager;
    private final SecurityMetadataSource securityMetadataSource;

    @Override
    public AuthorizationDecision check(Supplier<Authentication> authentication, Object object) {
        try {
            Collection<ConfigAttributes> attributes = this.securityMetadataSource.getAttributes(object);
            this.accessDecisionManager.decide(authentication.get(), object, attributes);
            return new AuthorizationDecision(true);
        } catch (AccessDeniedException ex) {
            return new AuthorizationDecision(false);
        }
    }

    @Override
    public void verify(Supplier<Authentication> authentication, Object object) {
        Collection<ConfigAttributes> attributes = this.securityMetadataSource.getAttributes(object);
        this.accessDecisionManager.decide(authentication.get(), object, attributes);
    }
}

@Component
public class AccessDecisionVoterAuthorizationManagerAdapter implements AuthorizationManager {
    private final AccessDecisionVoter accessDecisionVoter;
    private final SecurityMetadataSource securityMetadataSource;

    @Override
    public AuthorizationDecision check(Supplier<Authentication> authentication, Object object) {
        Collection<ConfigAttributes> attributes = this.securityMetadataSource.getAttributes(object);
        int decision = this.accessDecisionVoter.vote(authentication.get(), object, attributes);
        switch (decision) {
        case ACCESS_GRANTED:
            return new AuthorizationDecision(true);
        case ACCESS_DENIED:
            return new AuthorizationDecision(false);
        }
        return null;
    }
}
```

### Hierarchical Roles

Authorities 를 계층구조로 해석할 수 있다. 예를 들어 ROLE_ADMIN 은 ROLE_STAFF 의 권한을 포함할 수 있다.

```java
@Bean
AccessDecisionVoter hierarchyVoter() {
    RoleHierarchy hierarchy = new RoleHierarchyImpl();
    hierarchy.setHierarchy("ROLE_ADMIN > ROLE_STAFF\n" +
            "ROLE_STAFF > ROLE_USER\n" +
            "ROLE_USER > ROLE_GUEST");
    return new RoleHierarchyVoter(hierarchy);
}
```

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

* [SecurityContextHolder and Authentication](https://github.com/keesun/spring-security-basic/commit/a9854bc80c2e6b3026b4b00432e290dafef3b45f)

----

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

```java
// src/main/java/com/iamslash/exsecurity/form/SampleService.java
@Service
public class SampleService {

    public void dashboard() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        Object principal = authentication.getPrincipal();
        Collection<? extends GrantedAuthority> authorities = authentication.getAuthorities();
        Object credentials = authentication.getCredentials();
        boolean authenticated = authentication.isAuthenticated();
    }
}
```

## AuthenticationManager와 Authentication

authentication 은 `org\springframework\security\authentication\AuthenticationManager.java` 에서 처리된다.

```java
// org\springframework\security\authentication\AuthenticationManager.java
public interface AuthenticationManager {
	Authentication authenticate(Authentication authentication)
			throws AuthenticationException;  
}

// org\springframework\security\authentication\ProviderManager.java
public class ProviderManager implements AuthenticationManager, MessageSourceAware, InitializingBean {
	public Authentication authenticate(Authentication authentication)
			throws AuthenticationException {
		Class<? extends Authentication> toTest = authentication.getClass();
		AuthenticationException lastException = null;
		AuthenticationException parentException = null;
		Authentication result = null;
		Authentication parentResult = null;
		boolean debug = logger.isDebugEnabled();
    ...
		for (AuthenticationProvider provider : getProviders()) {
			if (!provider.supports(toTest)) {
				continue;
			}

			if (debug) {
				logger.debug("Authentication attempt using "
						+ provider.getClass().getName());
			}

			try {
				result = provider.authenticate(authentication);

				if (result != null) {
					copyDetails(authentication, result);
					break;
				}
			}
			catch (AccountStatusException | InternalAuthenticationServiceException e) {
				prepareException(e, authentication);
				// SEC-546: Avoid polling additional providers if auth failure is due to
				// invalid account status
				throw e;
			} catch (AuthenticationException e) {
				lastException = e;
			}
		}
    ...
  }
}
```

## ThreadLocal

* [ThreadLocal @ github](https://github.com/keesun/spring-security-basic/commit/4eae7a7cf0a80b53dc3f7075cb23314fa5090cac)

----

`SecurityContext` 는 `private static final ThreadLocal<SecurityContext> contextHolder = new ThreadLocal<>()` 에 저장되어 있다. 같은 thread 는 `SecurityContext` 를 argument 로 주고받을 필요가 없다. `SecurityContextHolder.getContext()` 를 호출하여 `SecurityContext` 를 얻어올 수 있다.

```java
// org\springframework\security\core\context\SecurityContextHolder.java
public class SecurityContextHolder {
  ...
  public static SecurityContext getContext() {
		return strategy.getContext();
	}
  ...
	private static void initialize() {
		if (!StringUtils.hasText(strategyName)) {
			// Set default
			strategyName = MODE_THREADLOCAL;
		}

		if (strategyName.equals(MODE_THREADLOCAL)) {
			strategy = new ThreadLocalSecurityContextHolderStrategy();
		}
    ...
  }  
}

// org\springframework\security\core\context\ThreadLocalSecurityContextHolderStrategy.java
final class ThreadLocalSecurityContextHolderStrategy implements
		SecurityContextHolderStrategy {

	private static final ThreadLocal<SecurityContext> contextHolder = new ThreadLocal<>();

	public void clearContext() {
		contextHolder.remove();
	}

	public SecurityContext getContext() {
		SecurityContext ctx = contextHolder.get();

		if (ctx == null) {
			ctx = createEmptyContext();
			contextHolder.set(ctx);
		}

		return ctx;
	}

	public void setContext(SecurityContext context) {
		Assert.notNull(context, "Only non-null SecurityContext instances are permitted");
		contextHolder.set(context);
	}

	public SecurityContext createEmptyContext() {
		return new SecurityContextImpl();
	}
}
```

## Authentication과 SecurityContextHodler

* [setAuthentication @ github](https://github.com/keesun/spring-security-basic/commit/2072f851e4c4206061b48155fc280f18a0b371a8)

----

`AuthenticationManager::authenticate` 는 인증을 마치고 `Authenticate` object 를 리턴한다. 

`UsernamePasswordAuthenticationFilter` filter 는 form authentication 을 처리한다. `AuthenticationManager::authenticate` 에 의해 리턴된 `Authentication` object 를 `SecurityContextHolder` 에 저장한다.

```java
// org\springframework\security\web\authentication\UsernamePasswordAuthenticationFilter.java
public class UsernamePasswordAuthenticationFilter extends
		AbstractAuthenticationProcessingFilter {
	public Authentication attemptAuthentication(HttpServletRequest request,
			HttpServletResponse response) throws AuthenticationException {
		if (postOnly && !request.getMethod().equals("POST")) {
			throw new AuthenticationServiceException(
					"Authentication method not supported: " + request.getMethod());
		}

		String username = obtainUsername(request);
		String password = obtainPassword(request);

		if (username == null) {
			username = "";
		}

		if (password == null) {
			password = "";
		}

		username = username.trim();

		UsernamePasswordAuthenticationToken authRequest = new UsernamePasswordAuthenticationToken(
				username, password);

		// Allow subclasses to set the "details" property
		setDetails(request, authRequest);

		return this.getAuthenticationManager().authenticate(authRequest);
	}  
}      
```

`SecurityContextPersistenceFilter` 는 여러 HTTP request 에서 `SecurityContext` 를 공유할 수 있게 한다???

```java
// org\springframework\security\web\context\SecurityContextPersistenceFilter.java
public class SecurityContextPersistenceFilter extends GenericFilterBean {
  ...
	public void doFilter(ServletRequest req, ServletResponse res, FilterChain chain)
			throws IOException, ServletException {
		HttpServletRequest request = (HttpServletRequest) req;
		HttpServletResponse response = (HttpServletResponse) res;

		if (request.getAttribute(FILTER_APPLIED) != null) {
			// ensure that filter is only applied once per request
			chain.doFilter(request, response);
			return;
		}

		final boolean debug = logger.isDebugEnabled();

		request.setAttribute(FILTER_APPLIED, Boolean.TRUE);

		if (forceEagerSessionCreation) {
			HttpSession session = request.getSession();

			if (debug && session.isNew()) {
				logger.debug("Eagerly created session: " + session.getId());
			}
		}

		HttpRequestResponseHolder holder = new HttpRequestResponseHolder(request,
				response);
		SecurityContext contextBeforeChainExecution = repo.loadContext(holder);

		try {
			SecurityContextHolder.setContext(contextBeforeChainExecution);

			chain.doFilter(holder.getRequest(), holder.getResponse());

		}
		finally {
			SecurityContext contextAfterChainExecution = SecurityContextHolder
					.getContext();
			// Crucial removal of SecurityContextHolder contents - do this before anything
			// else.
			SecurityContextHolder.clearContext();
			repo.saveContext(contextAfterChainExecution, holder.getRequest(),
					holder.getResponse());
			request.removeAttribute(FILTER_APPLIED);

			if (debug) {
				logger.debug("SecurityContextHolder now cleared, as request processing completed");
			}
		}
	}  
  ...
}
```

## 스프링 시큐리티 필터와 FilterChainProxy

Spring security 는 다음과 같은 filter 를 제공한다.

1. WebAsyncManagerIntergrationFilter
2. **SecurityContextPersistenceFilter**
3. HeaderWriterFilter
4. CsrfFilter
5. LogoutFilter
6. **UsernamePasswordAuthenticationFilter**
7. DefaultLoginPageGeneratingFilter
8. DefaultLogoutPageGeneratingFilter
9. BasicAuthenticationFilter
10. RequestCacheAwareFtiler
11. SecurityContextHolderAwareReqeustFilter
12. AnonymouseAuthenticationFilter
13. SessionManagementFilter
14. **ExeptionTranslationFilter**
15. FilterSecurityInterceptor

언급한 Filter 들은 `FilterChainProxy` 에 의해 호출된다.

```java
// org\springframework\security\web\FilterChainProxy.java
public class FilterChainProxy extends GenericFilterBean {
	private static final Log logger = LogFactory.getLog(FilterChainProxy.class);
	private final static String FILTER_APPLIED = FilterChainProxy.class.getName().concat(
			".APPLIED");
	private List<SecurityFilterChain> filterChains;
	private FilterChainValidator filterChainValidator = new NullFilterChainValidator();
	private HttpFirewall firewall = new StrictHttpFirewall();
...
	@Override
	public void doFilter(ServletRequest request, ServletResponse response,
			FilterChain chain) throws IOException, ServletException {
		boolean clearContext = request.getAttribute(FILTER_APPLIED) == null;
		if (clearContext) {
			try {
				request.setAttribute(FILTER_APPLIED, Boolean.TRUE);
				doFilterInternal(request, response, chain);
			}
			finally {
				SecurityContextHolder.clearContext();
				request.removeAttribute(FILTER_APPLIED);
			}
		}
		else {
			doFilterInternal(request, response, chain);
		}
	}
...
}  
```

## DelegatingFilterProxy와 FilterChainProxy

`DelegatingFilterProxy` 가 `FilterChainProxy` 를 호출한다. `FilterChainProxy` 는 등록된 여러 Filter 들을 호출한다. `FilterChainProxy` 는 `springSecurityFilterChain` 라는 이름의 Bean 으로 등록된다.

```java
// org\springframework\security\web\context\AbstractSecurityWebApplicationInitializer.java
public abstract class AbstractSecurityWebApplicationInitializer
		implements WebApplicationInitializer {

	private static final String SERVLET_CONTEXT_PREFIX = "org.springframework.web.servlet.FrameworkServlet.CONTEXT.";

	public static final String DEFAULT_FILTER_NAME = "springSecurityFilterChain";
...  
}

// org\springframework\web\filter\DelegatingFilterProxy.java
public class DelegatingFilterProxy extends GenericFilterBean {
...
	public DelegatingFilterProxy(String targetBeanName) {
		this(targetBeanName, null);
	}  
	public DelegatingFilterProxy(String targetBeanName, @Nullable WebApplicationContext wac) {
		Assert.hasText(targetBeanName, "Target Filter bean name must not be null or empty");
		this.setTargetBeanName(targetBeanName);
		this.webApplicationContext = wac;
		if (wac != null) {
			this.setEnvironment(wac.getEnvironment());
		}
	}  
	@Override
	public void doFilter(ServletRequest request, ServletResponse response, FilterChain filterChain)
			throws ServletException, IOException {

		// Lazily initialize the delegate if necessary.
		Filter delegateToUse = this.delegate;
		if (delegateToUse == null) {
			synchronized (this.delegateMonitor) {
				delegateToUse = this.delegate;
				if (delegateToUse == null) {
					WebApplicationContext wac = findWebApplicationContext();
					if (wac == null) {
						throw new IllegalStateException("No WebApplicationContext found: " +
								"no ContextLoaderListener or DispatcherServlet registered?");
					}
					delegateToUse = initDelegate(wac);
				}
				this.delegate = delegateToUse;
			}
		}

		// Let the delegate perform the actual doFilter operation.
		invokeDelegate(delegateToUse, request, response, filterChain);
	}
...
	protected void invokeDelegate(
			Filter delegate, ServletRequest request, ServletResponse response, FilterChain filterChain)
			throws ServletException, IOException {

		delegate.doFilter(request, response, filterChain);
	}    
}
```

## AccessDecisionManager 1부

* [AccessDicisionManager @ github](https://github.com/keesun/spring-security-basic/commit/75cca02c6481a33ff4e4e23c750004f30e05ddd9)

-----

`AccessDecisionManager` 는 Authorization 를 처리한다. `AffirmativeBased, ConsensusBased, UnanimousBased` 가 `AccessDecisionManager` 를 implement 한다.

* **AffirmativeBased**: 여러 `AccessDecisionVoter<S>` 중 하나만 찬성해도 허용한다. 기본값.
* **ConsensusBased**: 다수결.
* **UnanimousBased**: 만장일치.

`RoleVoter, ExpressionVoter` 등이 `AccessDecisionVoter` 를 구현한다.

```java
// org\springframework\security\access\AccessDecisionManager.java
public interface AccessDecisionManager {
	void decide(Authentication authentication, Object object,
			Collection<ConfigAttribute> configAttributes) throws AccessDeniedException,
			InsufficientAuthenticationException;
	boolean supports(ConfigAttribute attribute);
	boolean supports(Class<?> clazz);  
}

// org\springframework\security\access\vote\AffirmativeBased.java
public class AffirmativeBased extends AbstractAccessDecisionManager {
	public void decide(Authentication authentication, Object object,
			Collection<ConfigAttribute> configAttributes) throws AccessDeniedException {
		int deny = 0;

		for (AccessDecisionVoter voter : getDecisionVoters()) {
			int result = voter.vote(authentication, object, configAttributes);

			if (logger.isDebugEnabled()) {
				logger.debug("Voter: " + voter + ", returned: " + result);
			}

			switch (result) {
			case AccessDecisionVoter.ACCESS_GRANTED:
				return;

			case AccessDecisionVoter.ACCESS_DENIED:
				deny++;

				break;

			default:
				break;
			}
		}

		if (deny > 0) {
			throw new AccessDeniedException(messages.getMessage(
					"AbstractAccessDecisionManager.accessDenied", "Access is denied"));
		}

		// To get this far, every AccessDecisionVoter abstained
		checkAllowIfAllAbstainDecisions();
	}  
}

// org\springframework\security\access\vote\ConsensusBased.java
public class ConsensusBased extends AbstractAccessDecisionManager {
	public void decide(Authentication authentication, Object object,
			Collection<ConfigAttribute> configAttributes) throws AccessDeniedException {
		int grant = 0;
		int deny = 0;

		for (AccessDecisionVoter voter : getDecisionVoters()) {
			int result = voter.vote(authentication, object, configAttributes);

			if (logger.isDebugEnabled()) {
				logger.debug("Voter: " + voter + ", returned: " + result);
			}

			switch (result) {
			case AccessDecisionVoter.ACCESS_GRANTED:
				grant++;

				break;

			case AccessDecisionVoter.ACCESS_DENIED:
				deny++;

				break;

			default:
				break;
			}
		}

		if (grant > deny) {
			return;
		}

		if (deny > grant) {
			throw new AccessDeniedException(messages.getMessage(
					"AbstractAccessDecisionManager.accessDenied", "Access is denied"));
		}

		if ((grant == deny) && (grant != 0)) {
			if (this.allowIfEqualGrantedDeniedDecisions) {
				return;
			}
			else {
				throw new AccessDeniedException(messages.getMessage(
						"AbstractAccessDecisionManager.accessDenied", "Access is denied"));
			}
		}

		// To get this far, every AccessDecisionVoter abstained
		checkAllowIfAllAbstainDecisions();
	}  
}

// org\springframework\security\access\vote\UnanimousBased.java
public class UnanimousBased extends AbstractAccessDecisionManager {
	public void decide(Authentication authentication, Object object,
			Collection<ConfigAttribute> attributes) throws AccessDeniedException {

		int grant = 0;

		List<ConfigAttribute> singleAttributeList = new ArrayList<>(1);
		singleAttributeList.add(null);

		for (ConfigAttribute attribute : attributes) {
			singleAttributeList.set(0, attribute);

			for (AccessDecisionVoter voter : getDecisionVoters()) {
				int result = voter.vote(authentication, object, singleAttributeList);

				if (logger.isDebugEnabled()) {
					logger.debug("Voter: " + voter + ", returned: " + result);
				}

				switch (result) {
				case AccessDecisionVoter.ACCESS_GRANTED:
					grant++;

					break;

				case AccessDecisionVoter.ACCESS_DENIED:
					throw new AccessDeniedException(messages.getMessage(
							"AbstractAccessDecisionManager.accessDenied",
							"Access is denied"));

				default:
					break;
				}
			}
		}

		// To get this far, there were no deny votes
		if (grant > 0) {
			return;
		}

		// To get this far, every AccessDecisionVoter abstained
		checkAllowIfAllAbstainDecisions();
	}  
}

// org\springframework\security\access\AccessDecisionVoter.java
public interface AccessDecisionVoter<S> {
	int ACCESS_GRANTED = 1;
	int ACCESS_ABSTAIN = 0;
	int ACCESS_DENIED = -1;
	int vote(Authentication authentication, S object,
			Collection<ConfigAttribute> attributes);  
}
```

## AccessDecisionManager 2부

* [AccessDicisionManager @ github](https://github.com/keesun/spring-security-basic/commit/75cca02c6481a33ff4e4e23c750004f30e05ddd9)

----

`RoleHierarchyImpl` class 를 이용하면 계층형으로 Role 을 관리할 수 있다. 예를 들어 `Role_Admin` 이 `Role_User`
 의 권한을 포함한다. `roleHierarchy.setHierarchy("ROLE_ADMIN > ROLE_USER");`

```java
// org\springframework\security\access\hierarchicalroles\RoleHierarchyImpl.java
public class RoleHierarchyImpl implements RoleHierarchy {
...
	public void setHierarchy(String roleHierarchyStringRepresentation) {
		this.roleHierarchyStringRepresentation = roleHierarchyStringRepresentation;

		if (logger.isDebugEnabled()) {
			logger.debug("setHierarchy() - The following role hierarchy was set: "
					+ roleHierarchyStringRepresentation);
		}

		buildRolesReachableInOneStepMap();
		buildRolesReachableInOneOrMoreStepsMap();
	}
...  
}
```

## FilterSecurityInterceptor

`FilterSecurityInterceptor` 는 `FilterChainproxy` 의 마지막 filter 로 삽입된다. `FilterSecurityInterceptor` 는 `AccessDecisionManager` 를 이용하여 Authorization 을 처리한다. 

```java
// org\springframework\security\web\access\intercept\FilterSecurityInterceptor.java
public class FilterSecurityInterceptor extends AbstractSecurityInterceptor implements Filter {
...
	public void doFilter(ServletRequest request, ServletResponse response,
			FilterChain chain) throws IOException, ServletException {
		FilterInvocation fi = new FilterInvocation(request, response, chain);
		invoke(fi);
	}
...

	public void invoke(FilterInvocation fi) throws IOException, ServletException {
		if ((fi.getRequest() != null)
				&& (fi.getRequest().getAttribute(FILTER_APPLIED) != null)
				&& observeOncePerRequest) {
			// filter already applied to this request and user wants us to observe
			// once-per-request handling, so don't re-do security checking
			fi.getChain().doFilter(fi.getRequest(), fi.getResponse());
		}
		else {
			// first time this request being called, so perform security checking
			if (fi.getRequest() != null && observeOncePerRequest) {
				fi.getRequest().setAttribute(FILTER_APPLIED, Boolean.TRUE);
			}

			InterceptorStatusToken token = super.beforeInvocation(fi);

			try {
				fi.getChain().doFilter(fi.getRequest(), fi.getResponse());
			}
			finally {
				super.finallyInvocation(token);
			}

			super.afterInvocation(token, null);
		}
	}
...
}
```

## ExceptionTranslationFilter

`AuthenticationException, AccessDeniedException` 을 처리해주는 filter 이다.

```java
// org\springframework\security\web\access\ExceptionTranslationFilter.java
public class ExceptionTranslationFilter extends GenericFilterBean {
...
	private AccessDeniedHandler accessDeniedHandler = new AccessDeniedHandlerImpl();
	private AuthenticationEntryPoint authenticationEntryPoint;
	private AuthenticationTrustResolver authenticationTrustResolver = new AuthenticationTrustResolverImpl();
	private ThrowableAnalyzer throwableAnalyzer = new DefaultThrowableAnalyzer();

	private RequestCache requestCache = new HttpSessionRequestCache();

	private final MessageSourceAccessor messages = SpringSecurityMessageSource.getAccessor();
...  

	public void doFilter(ServletRequest req, ServletResponse res, FilterChain chain)
			throws IOException, ServletException {
		HttpServletRequest request = (HttpServletRequest) req;
		HttpServletResponse response = (HttpServletResponse) res;

		try {
			chain.doFilter(request, response);

			logger.debug("Chain processed normally");
		}
		catch (IOException ex) {
			throw ex;
		}
		catch (Exception ex) {
			// Try to extract a SpringSecurityException from the stacktrace
			Throwable[] causeChain = throwableAnalyzer.determineCauseChain(ex);
			RuntimeException ase = (AuthenticationException) throwableAnalyzer
					.getFirstThrowableOfType(AuthenticationException.class, causeChain);

			if (ase == null) {
				ase = (AccessDeniedException) throwableAnalyzer.getFirstThrowableOfType(
						AccessDeniedException.class, causeChain);
			}

			if (ase != null) {
				if (response.isCommitted()) {
					throw new ServletException("Unable to handle the Spring Security Exception because the response is already committed.", ex);
				}
				handleSpringSecurityException(request, response, chain, ase);
			}
			else {
				// Rethrow ServletExceptions and RuntimeExceptions as-is
				if (ex instanceof ServletException) {
					throw (ServletException) ex;
				}
				else if (ex instanceof RuntimeException) {
					throw (RuntimeException) ex;
				}

				// Wrap other Exceptions. This shouldn't actually happen
				// as we've already covered all the possibilities for doFilter
				throw new RuntimeException(ex);
			}
		}
	}
...  
}
```

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

* [Ingoring() @ github](https://github.com/keesun/spring-security-basic/commit/6b8396a11117f313549cb74dc02ba3d4bfde9662)

----

`https://www.iamslash.com/favicon.ico` 와 같은 static resource 들은 authentication 의 대상이 되지 않도록 하자. spring security filter 가 전혀 적용되지 않는다.

```java
@Configuration
@EnableWebSecurity
public class SecurityConfig extends WebSecurityConfigurerAdapter {
    ...
    @Override
    public void configure(WebSecurity web) throws Exception {
        web.ignoring().requestMatchers(PathRequest.toStaticResources().atCommonLocations());
    }
    ...
}    
```

## 스프링 시큐리티 ignoring() 2부

다음과 같이 `http.authorizeRequests()` 를 이용하면 spring security filter 가 적용된다.

```java
http.authorizeRequests()
.requestMatchers(PathRequest.toStaticResources().atCommonLocations()).permitAll()
```

* dynamic resource 는 `http.authorizeRequests()` 를 이용하자.
* static resource 는 `web.ignoring()` 를 이용하자.

## Async 웹 MVC를 지원하는 필터: WebAsyncManagerIntegrationFilter

* [WebAsynManagerIntegrationFilter @ github](https://github.com/keesun/spring-security-basic/commit/ceb4b5263f7d67b1b8ff55240114496100dfc028)

-----

`Principal` 는 thread local 이다. 한편 `WebAsynManagerIntegrationFilter` filter 는 tomcat thread 와 logic thread 에서 같은 `Principal` 을 공유할 수 있도록 해준다. 

`asyncHandler()` 는 tomcat thread 에서 실행된다. tomcat thread 가 return 한 `Callable<String>` 은 다른 thread 에서 실행된다. `WebAsynManagerIntegrationFilter` filter 덕분에 tomcat thread 와 logic thread 는 같은 `Principal` 을 공유한다.

```java
// src/main/java/com/iamslash/exsecurity/common/SecurityLogger.java
public class SecurityLogger {

    public static void log(String message) {
        System.out.println(message);
        Thread thread = Thread.currentThread();
        System.out.println("Thread: " + thread.getName());
        Object principal = SecurityContextHolder.getContext().getAuthentication().getPrincipal();
        System.out.println("Principal: " + principal);
    }

}

// src/main/java/com/iamslash/exsecurity/form/SampleController.java
@Controller
public class SampleController {
  ...
    @GetMapping("/async-handler")
    @ResponseBody
    public Callable<String> asyncHandler() {
        SecurityLogger.log("MVC");
        return () -> {
            SecurityLogger.log("Callable");
            return "Async Handler";
        };
    }  
}
```

## 스프링 시큐리티와 @Async

`@Async` 를 Method 에 부착하면 Async Method 를 호출할 수 있다. `@EnableAsync` 를 `ExsecurityApplication` class 에 부착해야 한다. `@Async` 가 부착된 Method 는 다른 thread 에서 실행될 것이다. `SecurityContextHolder.setStrategyName(SecurityContextHolder.MODE_INHERITABLETHREADLOCAL);` 를 호출하여 다른 thread 에 SecurityContext 가 공유되게 해야 한다.

```java
// src/main/java/com/iamslash/exsecurity/ExsecurityApplication.java
@SpringBootApplication
@EnableAsync
public class ExsecurityApplication {

	@Bean
	public PasswordEncoder passwordEncoder() {
		return PasswordEncoderFactories.createDelegatingPasswordEncoder();
	}
	public static void main(String[] args) {
		SpringApplication.run(ExsecurityApplication.class, args);
	}
}

// src/main/java/com/iamslash/exsecurity/config/SecurityConfig.java
@Configuration
@EnableWebSecurity
public class SecurityConfig extends WebSecurityConfigurerAdapter {
    public SecurityExpressionHandler expressionHandler() {
        RoleHierarchyImpl roleHierarchy = new RoleHierarchyImpl();
        roleHierarchy.setHierarchy("ROLE_ADMIN > ROLE_USER");
        DefaultWebSecurityExpressionHandler handler = new DefaultWebSecurityExpressionHandler();
        handler.setRoleHierarchy(roleHierarchy);
        return handler;
    }
    @Override
    public void configure(WebSecurity web) throws Exception {
        web.ignoring().requestMatchers(PathRequest.toStaticResources().atCommonLocations());
    }
    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http.authorizeRequests()
                .mvcMatchers("/", "/info", "/account/**").permitAll()
                .mvcMatchers("/admin").hasRole("ADMIN")
                .mvcMatchers("/user").hasRole("USER")
                .anyRequest().authenticated()
                .expressionHandler(expressionHandler());
        http.formLogin();
        http.httpBasic();

        SecurityContextHolder.setStrategyName(SecurityContextHolder.MODE_INHERITABLETHREADLOCAL);
    }

}

// src/main/java/com/iamslash/exsecurity/form/SampleController.java
@Controller
public class SampleController {
    @Autowired SampleService sampleService;
    @Autowired AccountRepository accountRepository;
    @GetMapping("/async-service")
    @ResponseBody
    public String asyncService() {
        SecurityLogger.log("MVC, before async service");
        sampleService.asyncService();
        SecurityLogger.log("MVC, after async service");
        return "Async Service";
    }		
}

// src/main/java/com/iamslash/exsecurity/form/SampleService.java
@Service
public class SampleService {
    public void dashboard() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        UserDetails userDetails = (UserDetails) authentication.getPrincipal();
        System.out.println("===============");
        System.out.println(authentication);
        System.out.println(userDetails.getUsername());
    }

    @Async
    public void asyncService() {
        SecurityLogger.log("Async Service");
        System.out.println("Async service is called.");
    }
}
```

## SecurityContext 영속화 필터: SecurityContextPersistenceFilter

SecurityContext 는 `SecurityContextPersistenceFilter` 에 의해 `SecurityContextRepository` object 에 저장된다. `HttpSessionSecurityContextRepository` 는 `SecurityContextRepository` 를 구현한다.

```java
// org\springframework\security\web\context\SecurityContextPersistenceFilter.java
public class SecurityContextPersistenceFilter extends GenericFilterBean {
	static final String FILTER_APPLIED = "__spring_security_scpf_applied";

	private SecurityContextRepository repo;

	private boolean forceEagerSessionCreation = false;	
...
	public void doFilter(ServletRequest req, ServletResponse res, FilterChain chain)
			throws IOException, ServletException {
...
		try {
			SecurityContextHolder.setContext(contextBeforeChainExecution);

			chain.doFilter(holder.getRequest(), holder.getResponse());

		}
		finally {
			SecurityContext contextAfterChainExecution = SecurityContextHolder
					.getContext();
			// Crucial removal of SecurityContextHolder contents - do this before anything
			// else.
			SecurityContextHolder.clearContext();
			repo.saveContext(contextAfterChainExecution, holder.getRequest(),
					holder.getResponse());
			request.removeAttribute(FILTER_APPLIED);

			if (debug) {
				logger.debug("SecurityContextHolder now cleared, as request processing completed");
			}
		}
	}  
  ...
}

// org\springframework\security\web\context\SecurityContextRepository.java
public interface SecurityContextRepository {
	SecurityContext loadContext(HttpRequestResponseHolder requestResponseHolder);
	void saveContext(SecurityContext context, HttpServletRequest request,
			HttpServletResponse response);
	boolean containsContext(HttpServletRequest request);
}

// org\springframework\security\web\context\HttpSessionSecurityContextRepository.java
public class HttpSessionSecurityContextRepository implements SecurityContextRepository {
	public static final String SPRING_SECURITY_CONTEXT_KEY = "SPRING_SECURITY_CONTEXT";

	protected final Log logger = LogFactory.getLog(this.getClass());
...
	public void saveContext(SecurityContext context, HttpServletRequest request,
			HttpServletResponse response) {
		SaveContextOnUpdateOrErrorResponseWrapper responseWrapper = WebUtils
				.getNativeResponse(response,
						SaveContextOnUpdateOrErrorResponseWrapper.class);
		if (responseWrapper == null) {
			throw new IllegalStateException(
					"Cannot invoke saveContext on response "
							+ response
							+ ". You must use the HttpRequestResponseHolder.response after invoking loadContext");
		}
		if (!responseWrapper.isContextSaved()) {
			responseWrapper.saveContext(context);
		}
	}
...
}
```

## 시큐리티 관련 헤더 추가하는 필터: HeaderWriterFilter

`HeaderWriterFilter` 는 HTTP Security Header 를 추가해 준다. 다음의 class 들운 `HeaderWriter` interface 를 구현한다.

* CacheControlHeaderWriter
  * Cache History Vulnerability 방어
  * `Cache-Control: no-cache, no-store, max-age=0, must-revalidate`
  * `Expires: 0`
  * `Pragma: no-cache`
* ClearSiteDataHeaderWriter
* CompositeHeaderWriter
* ContentSecurityPolicyHeaderWriter
* DelegatingRequestMatcherHeaderWriter
* FeaturePolicyHeaderWriter
* HpkpHeaderWriter
* HstsHeaderWriter
  * HTTPS only
* ReferrerPolicyHeaderWriter
* StaticHeadersWriter
* XContentTypeOptionsHeaderWriter
  * Mime-type Sniffing 방어
  * `X-Content-Type-Options: nosniff`
* XXssProtectionHeaderWriter
  * Browser 의 XSS Filter 사용
  * `X-XSS-Protection: 1; mode=block`
* XFrameOptionsHeaderWriter
  * ClickJacking 방어
  * `X-Frame-Options: DENY`

```java
// org\springframework\security\web\header\HeaderWriterFilter.java
public class HeaderWriterFilter extends OncePerRequestFilter {
	private final List<HeaderWriter> headerWriters;

// org\springframework\security\web\header\HeaderWriter.java
public interface HeaderWriter {
	void writeHeaders(HttpServletRequest request, HttpServletResponse response);
}

// org\springframework\security\web\header\writers\XContentTypeOptionsHeaderWriter.java
public final class XContentTypeOptionsHeaderWriter extends StaticHeadersWriter {
	public XContentTypeOptionsHeaderWriter() {
		super("X-Content-Type-Options", "nosniff");
	}
}

// org\springframework\security\web\header\writers\XXssProtectionHeaderWriter.java
public final class XXssProtectionHeaderWriter implements HeaderWriter {
	private static final String XSS_PROTECTION_HEADER = "X-XSS-Protection";
...
	private void updateHeaderValue() {
		if (!enabled) {
			this.headerValue = "0";
			return;
		}
		this.headerValue = "1";
		if (block) {
			this.headerValue += "; mode=block";
		}
	}
...
}

// org\springframework\security\web\header\writers\CacheControlHeadersWriter.java
public final class CacheControlHeadersWriter implements HeaderWriter {
	private static final String EXPIRES = "Expires";
	private static final String PRAGMA = "Pragma";
	private static final String CACHE_CONTROL = "Cache-Control";

	private final HeaderWriter delegate;
...
	private static List<Header> createHeaders() {
		List<Header> headers = new ArrayList<>(3);
		headers.add(new Header(CACHE_CONTROL,
				"no-cache, no-store, max-age=0, must-revalidate"));
		headers.add(new Header(PRAGMA, "no-cache"));
		headers.add(new Header(EXPIRES, "0"));
		return headers;
	}
...
}	

// org\springframework\security\web\header\writers\frameoptions\XFrameOptionsHeaderWriter.java
public final class XFrameOptionsHeaderWriter implements HeaderWriter {

	public static final String XFRAME_OPTIONS_HEADER = "X-Frame-Options";

	private final AllowFromStrategy allowFromStrategy;
	private final XFrameOptionsMode frameOptionsMode;
...

...
}
```

## CSRF 어택 방지 필터: CsrfFilter

`CsrfFilter` 는 [CSRF (Cross Site Request Forgery)](#csrf-cross-site-request-forgery) Attack 을 방어한다.

Spring boot application 은 `<input name="_csrf" type="hidden" value="a-b-c-d-e"` 를 포함한 HTML 을 HTTP Reponse 로 전송한다. Browser 에서 form 을 전송할 때 Spring boot application 은 CsrfFilter 에서 `_csrf` 의 값을 validte 한다.

```java
// org\springframework\security\web\csrf\CsrfFilter.java
public final class CsrfFilter extends OncePerRequestFilter {
	public static final RequestMatcher DEFAULT_CSRF_MATCHER = new DefaultRequiresCsrfMatcher();
...
	@Override
	protected void doFilterInternal(HttpServletRequest request,
			HttpServletResponse response, FilterChain filterChain)
					throws ServletException, IOException {
		request.setAttribute(HttpServletResponse.class.getName(), response);

		CsrfToken csrfToken = this.tokenRepository.loadToken(request);
		final boolean missingToken = csrfToken == null;
		if (missingToken) {
			csrfToken = this.tokenRepository.generateToken(request);
			this.tokenRepository.saveToken(csrfToken, request, response);
		}
		request.setAttribute(CsrfToken.class.getName(), csrfToken);
		request.setAttribute(csrfToken.getParameterName(), csrfToken);

		if (!this.requireCsrfProtectionMatcher.matches(request)) {
			filterChain.doFilter(request, response);
			return;
		}

		String actualToken = request.getHeader(csrfToken.getHeaderName());
		if (actualToken == null) {
			actualToken = request.getParameter(csrfToken.getParameterName());
		}
		if (!csrfToken.getToken().equals(actualToken)) {
			if (this.logger.isDebugEnabled()) {
				this.logger.debug("Invalid CSRF token found for "
						+ UrlUtils.buildFullRequestUrl(request));
			}
			if (missingToken) {
				this.accessDeniedHandler.handle(request, response,
						new MissingCsrfTokenException(actualToken));
			}
			else {
				this.accessDeniedHandler.handle(request, response,
						new InvalidCsrfTokenException(csrfToken, actualToken));
			}
			return;
		}

		filterChain.doFilter(request, response);
	}

...
}
```

## CSRF 토큰 사용 예제

* [CSRF example /signup @ github](https://github.com/keesun/spring-security-basic/commit/5cd43f0cd36b4447c46d8cdf277f0207e1042907)

----

CSRF 를 사용한 `SignUpController` class 를 정의하고 `SignUpControllerTest` class 를 작성해 보자. CSRF 는 `csrf()` 를 주입하자.

```java
// src/main/java/com/iamslash/exsecurity/account/SignUpController.java
@Controller
@RequestMapping("/signup")
public class SignUpController {

    @Autowired AccountService accountService;

    @GetMapping
    public String signupForm(Model model) {
        model.addAttribute("account", new Account());
        return "signup";
    }

    @PostMapping
    public String processSignUp(@ModelAttribute Account account) {
        account.setRole("USER");
        accountService.createNew(account);
        return "redirect:/";
    }

}

// src/test/java/com/iamslash/exsecurity/account/SignUpControllerTest.java
@RunWith(SpringRunner.class)
@SpringBootTest
@AutoConfigureMockMvc
public class SignUpControllerTest {

    @Autowired
    MockMvc mockMvc;

    @Test
    public void signUpForm() throws Exception {
        mockMvc.perform(get("/signup"))
                .andDo(print())
                .andExpect(status().isOk())
                .andExpect(content().string(containsString("_csrf")));
    }

    @Test
    public void processSignUp() throws Exception {
        mockMvc.perform(post("/signup")
                .param("username", "keesun")
                .param("password", "123")
                .with(csrf()))
                .andDo(print())
                .andExpect(status().is3xxRedirection());
    }

} 
```

## 로그아웃 처리 필터: LogoutFilter

* [LogoutFilter @ github](https://github.com/keesun/spring-security-basic/commit/736beda713b58138bc56c76ab3851ac04d7adfac)

----

`LogoutFilter` 는 logout 처리를 한다. `LogoutHandler handler` 는 logout 를 처리한다. `LogoutSuccessHandler logoutSuccessHandler` 는 logout 성공 후 처리를 한다.

```java
// org\springframework\security\web\authentication\logout\LogoutFilter.java
public class LogoutFilter extends GenericFilterBean {
	private RequestMatcher logoutRequestMatcher;
	private final LogoutHandler handler;
	private final LogoutSuccessHandler logoutSuccessHandler;
...
	public void doFilter(ServletRequest req, ServletResponse res, FilterChain chain)
			throws IOException, ServletException {
		HttpServletRequest request = (HttpServletRequest) req;
		HttpServletResponse response = (HttpServletResponse) res;

		if (requiresLogout(request, response)) {
			Authentication auth = SecurityContextHolder.getContext().getAuthentication();

			if (logger.isDebugEnabled()) {
				logger.debug("Logging out user '" + auth
						+ "' and transferring to logout destination");
			}

			this.handler.logout(request, response, auth);

			logoutSuccessHandler.onLogoutSuccess(request, response, auth);

			return;
		}

		chain.doFilter(request, response);
	}
...
}
```

별다른 설정이 없다면 logout page 는 `DefaultLogoutPageGeneratingFilter` 가 rendering 해준다.

```java
// org\springframework\security\web\authentication\ui\DefaultLogoutPageGeneratingFilter.java
public class DefaultLogoutPageGeneratingFilter extends OncePerRequestFilter {
	private RequestMatcher matcher = new AntPathRequestMatcher("/logout", "GET");
...
	@Override
	protected void doFilterInternal(HttpServletRequest request,
			HttpServletResponse response, FilterChain filterChain)
			throws ServletException, IOException {
		if (this.matcher.matches(request)) {
			renderLogout(request, response);
		} else {
			filterChain.doFilter(request, response);
		}
	}

	private void renderLogout(HttpServletRequest request, HttpServletResponse response)
			throws IOException {
		String page =  "<!DOCTYPE html>\n"
				+ "<html lang=\"en\">\n"
				+ "  <head>\n"
				+ "    <meta charset=\"utf-8\">\n"
				+ "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1, shrink-to-fit=no\">\n"
				+ "    <meta name=\"description\" content=\"\">\n"
				+ "    <meta name=\"author\" content=\"\">\n"
				+ "    <title>Confirm Log Out?</title>\n"
				+ "    <link href=\"https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta/css/bootstrap.min.css\" rel=\"stylesheet\" integrity=\"sha384-/Y6pD6FV/Vv2HJnA6t+vslU6fwYXjCFtcEpHbNJ0lyAFsXTsjBbfaDjzALeQsN6M\" crossorigin=\"anonymous\">\n"
				+ "    <link href=\"https://getbootstrap.com/docs/4.0/examples/signin/signin.css\" rel=\"stylesheet\" crossorigin=\"anonymous\"/>\n"
				+ "  </head>\n"
				+ "  <body>\n"
				+ "     <div class=\"container\">\n"
				+ "      <form class=\"form-signin\" method=\"post\" action=\"" + request.getContextPath() + "/logout\">\n"
				+ "        <h2 class=\"form-signin-heading\">Are you sure you want to log out?</h2>\n"
				+ renderHiddenInputs(request)
				+ "        <button class=\"btn btn-lg btn-primary btn-block\" type=\"submit\">Log Out</button>\n"
				+ "      </form>\n"
				+ "    </div>\n"
				+ "  </body>\n"
				+ "</html>";

		response.setContentType("text/html;charset=UTF-8");
		response.getWriter().write(page);
	}
...
}
```

다음과 같이 `http.logout().logoutSuccessUrl("/")` 를 호출하여 logout 후에 특정 url 로 이동시킬 수 있다.

```java
// src/main/java/com/iamslash/exsecurity/config/SecurityConfig.java
@Configuration
@EnableWebSecurity
public class SecurityConfig extends WebSecurityConfigurerAdapter {
    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http.authorizeRequests()
                .mvcMatchers("/", "/info", "/account/**", "/signup").permitAll()
                .mvcMatchers("/admin").hasRole("ADMIN")
                .mvcMatchers("/user").hasRole("USER")
                .anyRequest().authenticated()
                .expressionHandler(expressionHandler());
        http.formLogin();
        http.httpBasic();

        http.logout().logoutSuccessUrl("/");

        SecurityContextHolder.setStrategyName(SecurityContextHolder.MODE_INHERITABLETHREADLOCAL);
    }
}
```

## 폼 인증 처리 필터: UsernamePasswordAuthenticationFilter

`UsernamePasswordAuthenticationFilter` filter 는 form authentication 을 처리한다. 
* `UsernamePasswordAuthenticationFilter` 는 `AbstractAuthenticationProcessingFilter` 를 상속한다. 
* `AbstractAuthenticationProcessingFilter` 는 `AuthenticationManager authenticationManager` 를 갖는다.
* `ProviderManager` 는 `AuthenticationManager` 를 상속한다.
* `ProviderManager` 는 `List<AuthenticationProvider> providers` 를 갖는다.
* `DaoAuthenticationProvider` 는 `AuthenticationProvider` 를 상속한다.
* `DaoAuthenticationProvider` 는 `UserDetailsService userDetailsService` 를 갖는다.
* `AuthenticationManager::authenticate` 는 인증을 마치고 `Authenticate` object 를 리턴한다.  
* `AuthenticationManager::authenticate` 에 의해 리턴된 `Authentication` object 를 `SecurityContextHolder` 에 저장한다.

```java
// org\springframework\security\web\authentication\AbstractAuthenticationProcessingFilter.java
public abstract class AbstractAuthenticationProcessingFilter extends GenericFilterBean
		implements ApplicationEventPublisherAware, MessageSourceAware {
	protected ApplicationEventPublisher eventPublisher;
	protected AuthenticationDetailsSource<HttpServletRequest, ?> authenticationDetailsSource = new WebAuthenticationDetailsSource();
	private AuthenticationManager authenticationManager;			
}

// org\springframework\security\web\authentication\UsernamePasswordAuthenticationFilter.java
public class UsernamePasswordAuthenticationFilter extends
		AbstractAuthenticationProcessingFilter {
	public Authentication attemptAuthentication(HttpServletRequest request,
			HttpServletResponse response) throws AuthenticationException {
		if (postOnly && !request.getMethod().equals("POST")) {
			throw new AuthenticationServiceException(
					"Authentication method not supported: " + request.getMethod());
		}

		String username = obtainUsername(request);
		String password = obtainPassword(request);

		if (username == null) {
			username = "";
		}

		if (password == null) {
			password = "";
		}

		username = username.trim();

		UsernamePasswordAuthenticationToken authRequest = new UsernamePasswordAuthenticationToken(
				username, password);

		// Allow subclasses to set the "details" property
		setDetails(request, authRequest);

		return this.getAuthenticationManager().authenticate(authRequest);
	}  
}      

// org\springframework\security\authentication\AuthenticationManager.java
public interface AuthenticationManager {
	Authentication authenticate(Authentication authentication)
			throws AuthenticationException;
}

// org\springframework\security\authentication\ProviderManager.java
public class ProviderManager implements AuthenticationManager, MessageSourceAware,
		InitializingBean {
	private static final Log logger = LogFactory.getLog(ProviderManager.class);
	private AuthenticationEventPublisher eventPublisher = new NullEventPublisher();
	private List<AuthenticationProvider> providers = Collections.emptyList();
...	
}	

// org\springframework\security\authentication\AuthenticationProvider.java
public interface AuthenticationProvider {
	Authentication authenticate(Authentication authentication)
			throws AuthenticationException;
...
}

// org\springframework\security\authentication\dao\DaoAuthenticationProvider.java
public class DaoAuthenticationProvider extends AbstractUserDetailsAuthenticationProvider {
...	
	private UserDetailsService userDetailsService;
	public void setUserDetailsService(UserDetailsService userDetailsService) {
		this.userDetailsService = userDetailsService;
	}
...	
}
```

## 로그인/로그아웃 폼 페이지 생성해주는 필터: DefaultLogin/LogoutPageGeneratingFilter

`DefaultLoginPageGeneratingFilter` 는 login page 를 rendering 해준다. `DefaultLogoutPageGeneratingFilter` 는 logout page 를 rendering 해준다. `http.formLogin().loginPage("/login")` 를 호출하면 `DefaultLoginPageGeneratingFilter, DefaultLogoutPageGeneratingFilter` 는 Filter chain 에서 사라진다.

```java
// org\springframework\security\web\authentication\ui\DefaultLoginPageGeneratingFilter.java
public class DefaultLoginPageGeneratingFilter extends GenericFilterBean {
...

	private String generateLoginPageHtml(HttpServletRequest request, boolean loginError,
			boolean logoutSuccess) {
		String errorMsg = "Invalid credentials";

		if (loginError) {
			HttpSession session = request.getSession(false);

			if (session != null) {
				AuthenticationException ex = (AuthenticationException) session
						.getAttribute(WebAttributes.AUTHENTICATION_EXCEPTION);
				errorMsg = ex != null ? ex.getMessage() : "Invalid credentials";
			}
		}

		StringBuilder sb = new StringBuilder();

		sb.append("<!DOCTYPE html>\n"
				+ "<html lang=\"en\">\n"
				+ "  <head>\n"
				+ "    <meta charset=\"utf-8\">\n"
				+ "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1, shrink-to-fit=no\">\n"
				+ "    <meta name=\"description\" content=\"\">\n"
				+ "    <meta name=\"author\" content=\"\">\n"
				+ "    <title>Please sign in</title>\n"
				+ "    <link href=\"https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta/css/bootstrap.min.css\" rel=\"stylesheet\" integrity=\"sha384-/Y6pD6FV/Vv2HJnA6t+vslU6fwYXjCFtcEpHbNJ0lyAFsXTsjBbfaDjzALeQsN6M\" crossorigin=\"anonymous\">\n"
				+ "    <link href=\"https://getbootstrap.com/docs/4.0/examples/signin/signin.css\" rel=\"stylesheet\" crossorigin=\"anonymous\"/>\n"
				+ "  </head>\n"
				+ "  <body>\n"
				+ "     <div class=\"container\">\n");

		String contextPath = request.getContextPath();
		if (this.formLoginEnabled) {
			sb.append("      <form class=\"form-signin\" method=\"post\" action=\"" + contextPath + this.authenticationUrl + "\">\n"
					+ "        <h2 class=\"form-signin-heading\">Please sign in</h2>\n"
					+ createError(loginError, errorMsg)
					+ createLogoutSuccess(logoutSuccess)
					+ "        <p>\n"
					+ "          <label for=\"username\" class=\"sr-only\">Username</label>\n"
					+ "          <input type=\"text\" id=\"username\" name=\"" + this.usernameParameter + "\" class=\"form-control\" placeholder=\"Username\" required autofocus>\n"
					+ "        </p>\n"
					+ "        <p>\n"
					+ "          <label for=\"password\" class=\"sr-only\">Password</label>\n"
					+ "          <input type=\"password\" id=\"password\" name=\"" + this.passwordParameter + "\" class=\"form-control\" placeholder=\"Password\" required>\n"
					+ "        </p>\n"
					+ createRememberMe(this.rememberMeParameter)
					+ renderHiddenInputs(request)
					+ "        <button class=\"btn btn-lg btn-primary btn-block\" type=\"submit\">Sign in</button>\n"
					+ "      </form>\n");
		}

		if (openIdEnabled) {
			sb.append("      <form name=\"oidf\" class=\"form-signin\" method=\"post\" action=\"" + contextPath + this.openIDauthenticationUrl + "\">\n"
					+ "        <h2 class=\"form-signin-heading\">Login with OpenID Identity</h2>\n"
					+ createError(loginError, errorMsg)
					+ createLogoutSuccess(logoutSuccess)
					+ "        <p>\n"
					+ "          <label for=\"username\" class=\"sr-only\">Identity</label>\n"
					+ "          <input type=\"text\" id=\"username\" name=\"" + this.openIDusernameParameter + "\" class=\"form-control\" placeholder=\"Username\" required autofocus>\n"
					+ "        </p>\n"
					+ createRememberMe(this.openIDrememberMeParameter)
					+ renderHiddenInputs(request)
					+ "        <button class=\"btn btn-lg btn-primary btn-block\" type=\"submit\">Sign in</button>\n"
					+ "      </form>\n");
		}

		if (oauth2LoginEnabled) {
			sb.append("<h2 class=\"form-signin-heading\">Login with OAuth 2.0</h2>");
			sb.append(createError(loginError, errorMsg));
			sb.append(createLogoutSuccess(logoutSuccess));
			sb.append("<table class=\"table table-striped\">\n");
			for (Map.Entry<String, String> clientAuthenticationUrlToClientName : oauth2AuthenticationUrlToClientName.entrySet()) {
				sb.append(" <tr><td>");
				String url = clientAuthenticationUrlToClientName.getKey();
				sb.append("<a href=\"").append(contextPath).append(url).append("\">");
				String clientName = HtmlUtils.htmlEscape(clientAuthenticationUrlToClientName.getValue());
				sb.append(clientName);
				sb.append("</a>");
				sb.append("</td></tr>\n");
			}
			sb.append("</table>\n");
		}

		if (this.saml2LoginEnabled) {
			sb.append("<h2 class=\"form-signin-heading\">Login with SAML 2.0</h2>");
			sb.append(createError(loginError, errorMsg));
			sb.append(createLogoutSuccess(logoutSuccess));
			sb.append("<table class=\"table table-striped\">\n");
			for (Map.Entry<String, String> relyingPartyUrlToName : saml2AuthenticationUrlToProviderName.entrySet()) {
				sb.append(" <tr><td>");
				String url = relyingPartyUrlToName.getKey();
				sb.append("<a href=\"").append(contextPath).append(url).append("\">");
				String partyName = HtmlUtils.htmlEscape(relyingPartyUrlToName.getValue());
				sb.append(partyName);
				sb.append("</a>");
				sb.append("</td></tr>\n");
			}
			sb.append("</table>\n");
		}
		sb.append("</div>\n");
		sb.append("</body></html>");

		return sb.toString();
	}
...
}	

// org\springframework\security\web\authentication\ui\DefaultLogoutPageGeneratingFilter.java
public class DefaultLogoutPageGeneratingFilter extends OncePerRequestFilter {
	private RequestMatcher matcher = new AntPathRequestMatcher("/logout", "GET");
...

	private void renderLogout(HttpServletRequest request, HttpServletResponse response)
			throws IOException {
		String page =  "<!DOCTYPE html>\n"
				+ "<html lang=\"en\">\n"
				+ "  <head>\n"
				+ "    <meta charset=\"utf-8\">\n"
				+ "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1, shrink-to-fit=no\">\n"
				+ "    <meta name=\"description\" content=\"\">\n"
				+ "    <meta name=\"author\" content=\"\">\n"
				+ "    <title>Confirm Log Out?</title>\n"
				+ "    <link href=\"https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta/css/bootstrap.min.css\" rel=\"stylesheet\" integrity=\"sha384-/Y6pD6FV/Vv2HJnA6t+vslU6fwYXjCFtcEpHbNJ0lyAFsXTsjBbfaDjzALeQsN6M\" crossorigin=\"anonymous\">\n"
				+ "    <link href=\"https://getbootstrap.com/docs/4.0/examples/signin/signin.css\" rel=\"stylesheet\" crossorigin=\"anonymous\"/>\n"
				+ "  </head>\n"
				+ "  <body>\n"
				+ "     <div class=\"container\">\n"
				+ "      <form class=\"form-signin\" method=\"post\" action=\"" + request.getContextPath() + "/logout\">\n"
				+ "        <h2 class=\"form-signin-heading\">Are you sure you want to log out?</h2>\n"
				+ renderHiddenInputs(request)
				+ "        <button class=\"btn btn-lg btn-primary btn-block\" type=\"submit\">Log Out</button>\n"
				+ "      </form>\n"
				+ "    </div>\n"
				+ "  </body>\n"
				+ "</html>";

		response.setContentType("text/html;charset=UTF-8");
		response.getWriter().write(page);
	}
...
}
```

## 로그인/로그아웃 폼 커스터마이징

* [Custom login/logout page @ github](https://github.com/keesun/spring-security-basic/commit/bd7e867f547fb648cc2be3a20eb633eafafc717a)

----

login, logout page 를 customizing 해보자. `login.html` 에서 submit 를 하면 Spring boot application 의 `UsernamePasswordAuthenticationFilter` 에서 login 처리 즉 `authenticate` 를 처리한다. `logout.html` 에서 submit 를 하면 Spring boot application 의 `LogoutFilter` 에서 logout 처리를 한다.

```java
// src/main/java/com/iamslash/exsecurity/account/LogInOutController.java
@Controller
public class LogInOutController {

    @GetMapping("/login")
    public String loginForm() {
        return "login";
    }

    @GetMapping("/logout")
    public String logoutForm() {
        return "logout";
    }

}

//src/main/java/com/iamslash/exsecurity/config/SecurityConfig.java
@Configuration
@EnableWebSecurity
public class SecurityConfig extends WebSecurityConfigurerAdapter {
    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http.authorizeRequests()
                .mvcMatchers("/", "/info", "/account/**", "/signup").permitAll()
                .mvcMatchers("/admin").hasRole("ADMIN")
                .mvcMatchers("/user").hasRole("USER")
                .anyRequest().authenticated()
                .expressionHandler(expressionHandler());
        http.formLogin()
                .loginPage("/login")
                .permitAll();
        http.httpBasic();

        http.logout()
				        .logoutSuccessUrl("/");

        SecurityContextHolder.setStrategyName(SecurityContextHolder.MODE_INHERITABLETHREADLOCAL);
    }
}
```

## Basic 인증 처리 필터: BasicAuthenticationFilter

* [The 'Basic' HTTP Authentication Scheme @ rfc](https://datatracker.ietf.org/doc/html/rfc7617)

----

`BasicAuthenticationFilter` 는 Basic authentication 을 지원한다. Basic authentication 은 `Authorization: Basic QWxhZGRpbjpPcGVuU2VzYW1l` 와 같은 Header 를 이용하여 authentication 을 처리한다. `QWxhZGRpbjpPcGVuU2VzYW1l` 은 username, password 를 base64 encoding 한 것이다. 보안에 매우 취약하다. HTTPS 를 사용해야 한다.
`SecurityConfig` class 의 `void configure()` 에서 `http.httpBasic()` 를 호출하여 BasicAuthenticationFilter 를 추가한다.

Basic authentication 은 HTTP request 할 때 마다 HTTP header `Authorization: Basic QWxhZGRpbjpPcGVuU2VzYW1l` 를 전송해야 한다.

```java
@Configuration
@EnableWebSecurity
public class SecurityConfig extends WebSecurityConfigurerAdapter {
    ...
    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http.authorizeRequests()
                .mvcMatchers("/", "/info", "/account/**", "/signup").permitAll()
                .mvcMatchers("/admin").hasRole("ADMIN")
                .mvcMatchers("/user").hasRole("USER")
                .anyRequest().authenticated()
                .expressionHandler(expressionHandler());
        http.formLogin();
        http.httpBasic();

        http.logout().logoutSuccessUrl("/");

        SecurityContextHolder.setStrategyName(SecurityContextHolder.MODE_INHERITABLETHREADLOCAL);
    }
    ...
}    
```

## 요청 캐시 필터: RequestCacheAwareFilter

`RequestCacheAwareFilter` 는 HTTP Request 를 caching 한다. 예를 들어 `/dashboard` 를 request 했을 때 login 이 되지 않은 상태이면 Spring boot application 은 `/login` page 를 rendering 한다. username, password 를 입력후 submit 하면 Spring boot application 은 이전에 caching 되있던 `/dashboard` request 를 이용한다.

```java
// org\springframework\security\web\savedrequest\RequestCacheAwareFilter.java
public class RequestCacheAwareFilter extends GenericFilterBean {

	private RequestCache requestCache;

	public RequestCacheAwareFilter() {
		this(new HttpSessionRequestCache());
	}

	public RequestCacheAwareFilter(RequestCache requestCache) {
		Assert.notNull(requestCache, "requestCache cannot be null");
		this.requestCache = requestCache;
	}

	public void doFilter(ServletRequest request, ServletResponse response,
			FilterChain chain) throws IOException, ServletException {

		HttpServletRequest wrappedSavedRequest = requestCache.getMatchingRequest(
				(HttpServletRequest) request, (HttpServletResponse) response);

		chain.doFilter(wrappedSavedRequest == null ? request : wrappedSavedRequest,
				response);
	}

}
```

## 시큐리티 관련 서블릿 스팩 구현 필터: SecurityContextHolderAwareRequestFilter

`SecurityContextHolderAwareRequestFilter` 는 Servlet API 구현한다???

```java
// org\springframework\security\web\servletapi\SecurityContextHolderAwareRequestFilter.java
public class SecurityContextHolderAwareRequestFilter extends GenericFilterBean {
	private String rolePrefix = "ROLE_";
	private HttpServletRequestFactory requestFactory;
	private AuthenticationEntryPoint authenticationEntryPoint;
	private AuthenticationManager authenticationManager;
	private List<LogoutHandler> logoutHandlers;
	private AuthenticationTrustResolver trustResolver = new AuthenticationTrustResolverImpl();

	public void setRolePrefix(String rolePrefix) {
		Assert.notNull(rolePrefix, "Role prefix must not be null");
		this.rolePrefix = rolePrefix;
		updateFactory();
	}

	public void setAuthenticationEntryPoint(
			AuthenticationEntryPoint authenticationEntryPoint) {
		this.authenticationEntryPoint = authenticationEntryPoint;
	}

	public void setAuthenticationManager(AuthenticationManager authenticationManager) {
		this.authenticationManager = authenticationManager;
	}

	public void setLogoutHandlers(List<LogoutHandler> logoutHandlers) {
		this.logoutHandlers = logoutHandlers;
	}

	public void doFilter(ServletRequest req, ServletResponse res, FilterChain chain)
			throws IOException, ServletException {
		chain.doFilter(this.requestFactory.create((HttpServletRequest) req,
				(HttpServletResponse) res), res);
	}

	@Override
	public void afterPropertiesSet() throws ServletException {
		super.afterPropertiesSet();
		updateFactory();
	}

	private void updateFactory() {
		String rolePrefix = this.rolePrefix;
		this.requestFactory = createServlet3Factory(rolePrefix);
	}

	public void setTrustResolver(AuthenticationTrustResolver trustResolver) {
		Assert.notNull(trustResolver, "trustResolver cannot be null");
		this.trustResolver = trustResolver;
		updateFactory();
	}

	private HttpServletRequestFactory createServlet3Factory(String rolePrefix) {
		HttpServlet3RequestFactory factory = new HttpServlet3RequestFactory(rolePrefix);
		factory.setTrustResolver(this.trustResolver);
		factory.setAuthenticationEntryPoint(this.authenticationEntryPoint);
		factory.setAuthenticationManager(this.authenticationManager);
		factory.setLogoutHandlers(this.logoutHandlers);
		return factory;
	}

}
```

## 익명 인증 필터: AnonymousAuthenticationFilter

`AnonymousAuthenticationFilter` 는 `Authentication` object 가 null 이면 Anonymous Authentication object 를 만들어 Security Context 에 넣어준다. null 이 아니면 아무일도 하지 않는다.

```java
// org\springframework\security\web\authentication\AnonymousAuthenticationFilter.java
public class AnonymousAuthenticationFilter extends GenericFilterBean implements
		InitializingBean {
...
	public void doFilter(ServletRequest req, ServletResponse res, FilterChain chain)
			throws IOException, ServletException {

		if (SecurityContextHolder.getContext().getAuthentication() == null) {
			SecurityContextHolder.getContext().setAuthentication(
					createAuthentication((HttpServletRequest) req));

			if (logger.isDebugEnabled()) {
				logger.debug("Populated SecurityContextHolder with anonymous token: '"
						+ SecurityContextHolder.getContext().getAuthentication() + "'");
			}
		}
		else {
			if (logger.isDebugEnabled()) {
				logger.debug("SecurityContextHolder not populated with anonymous token, as it already contained: '"
						+ SecurityContextHolder.getContext().getAuthentication() + "'");
			}
		}

		chain.doFilter(req, res);
	}
	protected Authentication createAuthentication(HttpServletRequest request) {
		AnonymousAuthenticationToken auth = new AnonymousAuthenticationToken(key,
				principal, authorities);
		auth.setDetails(authenticationDetailsSource.buildDetails(request));

		return auth;
	}
...
}

// src/main/java/com/iamslash/exsecurity/config/SecurityConfig.java
@Configuration
@EnableWebSecurity
public class SecurityConfig extends WebSecurityConfigurerAdapter {
    public SecurityExpressionHandler expressionHandler() {
        RoleHierarchyImpl roleHierarchy = new RoleHierarchyImpl();
        roleHierarchy.setHierarchy("ROLE_ADMIN > ROLE_USER");
        DefaultWebSecurityExpressionHandler handler = new DefaultWebSecurityExpressionHandler();
        handler.setRoleHierarchy(roleHierarchy);
        return handler;
    }
    @Override
    public void configure(WebSecurity web) throws Exception {
        web.ignoring().requestMatchers(PathRequest.toStaticResources().atCommonLocations());
    }
    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http.authorizeRequests()
                .mvcMatchers("/", "/info", "/account/**").permitAll()
                .mvcMatchers("/admin").hasRole("ADMIN")
                .mvcMatchers("/user").hasRole("USER")
                .anyRequest().authenticated()
                .expressionHandler(expressionHandler());
        http.formLogin();
        http.httpBasic();

        http.anonymous()
						.principal()
						.authorities()
						.key()

        SecurityContextHolder.setStrategyName(SecurityContextHolder.MODE_INHERITABLETHREADLOCAL);
    }

}
```

## 세션 관리 필터: SessionManagementFilter

`SessionManagementFilter` 는 다음과 같은 기능들을 제공한다.

* [Session Fixation Attack](https://owasp.org/www-community/attacks/Session_fixation) 을 방지한다. 즉, Session 이 조작되는 것을 방지한다.
* 동시에 접근가능한 Client 의 숫자를 제어한다.
* Session 생성 전략 (sessionCreationPolicy) 을 제어한다.

[Session Fixation Attacks](https://secureteam.co.uk/articles/web-application-security-articles/understanding-session-fixation-attacks/) 은 다음과 같은 과정으로 이루어진다.

![](https://secureteam.co.uk/gage/wp-content/uploads/2018/03/image1-768x650.png)

1. Attacker 가 Server 에 log-in 한다.
2. Server 는 sessionid 를 Cookie 에 저장하여 Attacker 에게 전송한다.
3. Attacker 는 Victim 에게 hijacking link 를 보내준다.
4. Victim 은 hijacking link 를 선택하고 hijacking link 에 저장된 sessionid 를 이용하여 Server 에 login 한다.
5. Attacker 는 sessionid 를 이용하여 Victim 의 resource 들을 접근할 수 있다.

다음과 같이 `SecurityConfig::configure(HttpSecurity http)` 에서 `http.sessionManagement()` 호출 하여 설정할 수 있다.

```java
// src/main/java/com/iamslash/exsecurity/config/SecurityConfig.java
@Configuration
@EnableWebSecurity
public class SecurityConfig extends WebSecurityConfigurerAdapter {
...
    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http.authorizeRequests()
                .mvcMatchers("/", "/info", "/account/**").permitAll()
                .mvcMatchers("/admin").hasRole("ADMIN")
                .mvcMatchers("/user").hasRole("USER")
                .anyRequest().authenticated()
                .expressionHandler(expressionHandler());
        http.formLogin();
        http.httpBasic();

        http.sessionManagement()
				    .sessionFixation()
						.changeSessionId()
						.invalidSessionUrl("/login");

        SecurityContextHolder.setStrategyName(SecurityContextHolder.MODE_INHERITABLETHREADLOCAL);
    }

}
```

다음과 같이 `maxSessionPreventsLogin(false)` 를 호출하면 같은 user 에대해 최대 동시 session 은 1 개로 제한한다. 또한 새로운 user 는 login 할 수 있고 기존 user 의 session 은 제거된다. 기존 user 는 다시 login 할 수 있다. 만약 `maxSessionPreventsLogin(true)` 를 호출하면 새로운 user 는 login 할 수 없다.

```java
// src/main/java/com/iamslash/exsecurity/config/SecurityConfig.java
@Configuration
@EnableWebSecurity
public class SecurityConfig extends WebSecurityConfigurerAdapter {
...
    @Override
    protected void configure(HttpSecurity http) throws Exception {
			...
        http.sessionManagement()
				    .sessionFixation()
						.changeSessionId()
						.maximumSession(1)
						.maxSessionPreventsLogin(false);
			...
		}
...								
}
```

다음과 같이 `sessionCreationPolicy(SessionCreationPolicy.STATELESS)` 를 호출하면 session 을 사용하지 않는다. api 호출할 때마다 login 해야 한다. `RequestCacheAwareFilter` 역시 previous http request 를 session 에 caching 한다. 따라서 `/login` 하고 `/dashboard` 로 이동하는 logic 은 동작하지 않는다.

```java
// src/main/java/com/iamslash/exsecurity/config/SecurityConfig.java
@Configuration
@EnableWebSecurity
public class SecurityConfig extends WebSecurityConfigurerAdapter {
...
    @Override
    protected void configure(HttpSecurity http) throws Exception {
			...
        http.sessionManagement()
				    .sessionCreationPolicy(SessionCreationPolicy.STATELESS);
			...
		}
...								
}

// org\springframework\security\config\http\SessionCreationPolicy.java
public enum SessionCreationPolicy {
	/** Always create an {@link HttpSession} */
	ALWAYS,
	/**
	 * Spring Security will never create an {@link HttpSession}, but will use the
	 * {@link HttpSession} if it already exists
	 */
	NEVER,
	/** Spring Security will only create an {@link HttpSession} if required */
	IF_REQUIRED,
	/**
	 * Spring Security will never create an {@link HttpSession} and it will never use it
	 * to obtain the {@link SecurityContext}
	 */
	STATELESS
}
```

## 인증/인가 예외 처리 필터: ExceptionTranslationFilter

* [ExceptionTranslatorFilter](https://github.com/keesun/spring-security-basic/commit/38ebb45a3ab68ff89f1809d035957c25fbc4af47)

----

* `ExceptionTranslationFilter` 는 `FilterSecurityInterceptor` 전에 처리된다.
* `ExceptionTranslationFilter` 는 `AuthenticationException, AccessDeniedException` 을 처리한다.
* `FilterSecurityInterceptor` 는 `AccessDecisionManager, AffirmativeBased` 를 이용하여 Authorization 을 처리한다.
* `AuthenticationException` 는 `AuthenticationEntryPoint, Http403ForbiddenEntryPoint` 에서 처리된다.
* `AccessDeniedException` 는 `AccessDeniedHandler, ` 에서 처리된다.

```java
// org\springframework\security\web\access\ExceptionTranslationFilter.java
public class ExceptionTranslationFilter extends GenericFilterBean {
...
	private AccessDeniedHandler accessDeniedHandler = new AccessDeniedHandlerImpl();
	private AuthenticationEntryPoint authenticationEntryPoint;
	private AuthenticationTrustResolver authenticationTrustResolver = new AuthenticationTrustResolverImpl();
	private ThrowableAnalyzer throwableAnalyzer = new DefaultThrowableAnalyzer();

	private RequestCache requestCache = new HttpSessionRequestCache();

	private final MessageSourceAccessor messages = SpringSecurityMessageSource.getAccessor();
...  

	public void doFilter(ServletRequest req, ServletResponse res, FilterChain chain)
			throws IOException, ServletException {
...
				handleSpringSecurityException(request, response, chain, ase);
...  
	}
...
	private void handleSpringSecurityException(HttpServletRequest request,
			HttpServletResponse response, FilterChain chain, RuntimeException exception)
			throws IOException, ServletException {
		if (exception instanceof AuthenticationException) {
			logger.debug(
					"Authentication exception occurred; redirecting to authentication entry point",
					exception);

			sendStartAuthentication(request, response, chain,
					(AuthenticationException) exception);
		}
		else if (exception instanceof AccessDeniedException) {
			Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
			if (authenticationTrustResolver.isAnonymous(authentication) || authenticationTrustResolver.isRememberMe(authentication)) {
				logger.debug(
						"Access is denied (user is " + (authenticationTrustResolver.isAnonymous(authentication) ? "anonymous" : "not fully authenticated") + "); redirecting to authentication entry point",
						exception);

				sendStartAuthentication(
						request,
						response,
						chain,
						new InsufficientAuthenticationException(
							messages.getMessage(
								"ExceptionTranslationFilter.insufficientAuthentication",
								"Full authentication is required to access this resource")));
			}
			else {
				logger.debug(
						"Access is denied (user is not anonymous); delegating to AccessDeniedHandler",
						exception);

				accessDeniedHandler.handle(request, response,
						(AccessDeniedException) exception);
			}
		}
	}
...
}

// org\springframework\security\web\AuthenticationEntryPoint.java
public interface AuthenticationEntryPoint {
	void commence(HttpServletRequest request, HttpServletResponse response,
			AuthenticationException authException) throws IOException, ServletException;
}

// org\springframework\security\web\authentication\Http403ForbiddenEntryPoint.java
public class Http403ForbiddenEntryPoint implements AuthenticationEntryPoint {
	private static final Log logger = LogFactory.getLog(Http403ForbiddenEntryPoint.class);
	public void commence(HttpServletRequest request, HttpServletResponse response,
			AuthenticationException arg2) throws IOException {
		if (logger.isDebugEnabled()) {
			logger.debug("Pre-authenticated entry point called. Rejecting access");
		}
		response.sendError(HttpServletResponse.SC_FORBIDDEN, "Access Denied");
	}
}

// org\springframework\security\web\access\AccessDeniedHandler.java
public interface AccessDeniedHandler {
	void handle(HttpServletRequest request, HttpServletResponse response,
			AccessDeniedException accessDeniedException) throws IOException,
			ServletException;
}

// org\springframework\security\web\session\InvalidSessionAccessDeniedHandler.java
public final class InvalidSessionAccessDeniedHandler implements AccessDeniedHandler {
	private final InvalidSessionStrategy invalidSessionStrategy;

	public InvalidSessionAccessDeniedHandler(InvalidSessionStrategy invalidSessionStrategy) {
		Assert.notNull(invalidSessionStrategy, "invalidSessionStrategy cannot be null");
		this.invalidSessionStrategy = invalidSessionStrategy;
	}

	public void handle(HttpServletRequest request, HttpServletResponse response,
			AccessDeniedException accessDeniedException) throws IOException,
			ServletException {
		invalidSessionStrategy.onInvalidSessionDetected(request, response);
	}
}
```

## 인가 처리 필터: FilterSecurityInterceptor

`FilterSecurityInterceptor` 는 `AccessDecisionManager, AffirmativeBased` 를 이용하여 Authorization 을 처리한다.

다음과 같이 `http.authorizeRequests()` 를 호출하여 autorization 을 설정할 수 있다. `mvcMatchers(), regexMachers()` 등을 이용하여 url 을 제어할 수 있다. 

* `permitAll()` 을 이용하여 특정 url 에 대해 모든 권한을 허용할 수 있다. `hasRole()` 을 이용하여 특정 url 에 대해 일부 권한을 허용할 수 있다. 
* `hashAuthority("ROLE_USEr")` 는 `ROLE_` prefix 를 사용해야 하는 것을 제외하고 `hasRole()` 과 같다.
* `anyRequest().anonymous()` 는 특정 url 에 대해 authenticated 가 되지 않아야 접근이 가능하다.
* `anyRequest().authenticated()` 는 특정 url 에 대해 authenticated 되어야 접근이 가능하다.
* `anyRequest().rememberMe()` 는 특정 url 에 대해 rememberMe() 로 authenticated 되어야 접근이 가능하다.
* `anyRequest().fullyAuthentiated()` 는 특정 url 에 대해 rememberMe() 로 authenticated 된 user 가 다시 login 해야 접근이 가능하다.

```java
// src/main/java/com/iamslash/exsecurity/config/SecurityConfig.java
@Configuration
@EnableWebSecurity
public class SecurityConfig extends WebSecurityConfigurerAdapter {
...
    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http.authorizeRequests()
                .mvcMatchers("/", "/info", "/account/**").permitAll()
                .mvcMatchers("/admin").hasRole("ADMIN")
                .mvcMatchers("/user").hasRole("USER")
                .anyRequest().authenticated()
                .expressionHandler(expressionHandler());
        http.formLogin();
        http.httpBasic();

        http.sessionManagement()
				    .sessionFixation()
						.changeSessionId()
						.invalidSessionUrl("/login");

        SecurityContextHolder.setStrategyName(SecurityContextHolder.MODE_INHERITABLETHREADLOCAL);
    }

}
```

## 토큰 기반 인증 필터 : RememberMeAuthenticationFilter

* [RememberMeAuthenticationFilter @ github](https://github.com/keesun/spring-security-basic/commit/397955014faeb515595b7cb3c2121bd6055742c4)

----

`RememberMeAuthenticationFilter` 는 cookie 를 이용해서 authentication 을 기억하는 기능을 제공한다. [EditThisCookie @ crhome](https://chrome.google.com/webstore/detail/editthiscookie/fngmhnnpilhplaeedifhccceomclgfbg?hl=ko) 을 chrome 에 설치하면 cookie 를 편집하여 debugging 이 가능하다.

`http.rememberMe()` 없이 Spring boot application 을 실행하고 login 해보자. `JSESSIONID` 가 cookie 에 저장되어 있을 것이다. [EditThisCookie @ crhome](https://chrome.google.com/webstore/detail/editthiscookie/fngmhnnpilhplaeedifhccceomclgfbg?hl=ko) 로 `JSESSIONID` 를 지워보자. 다시 login 해야 한다.

``http.rememberMe()` 를 포함해서  Spring boot application 을 실행하고 login 해보자. `JSESSIONID, remember-me` 가 cookie 에 저장되어 있을 것이다. `JSESSIONID` 를 지우고 인증이 필요한 page 를 request 해보자. 다시 login 할 필요가 없다. `remember-me` 에 의해 `JSESSIONID` 가 다시 cookie 에 저장된다.

```java
// src/main/java/com/iamslash/exsecurity/config/SecurityConfig.java
@Configuration
@EnableWebSecurity
public class SecurityConfig extends WebSecurityConfigurerAdapter {
...
    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http.authorizeRequests()
                .mvcMatchers("/", "/info", "/account/**").permitAll()
                .mvcMatchers("/admin").hasRole("ADMIN")
                .mvcMatchers("/user").hasRole("USER")
                .anyRequest().authenticated()
                .expressionHandler(expressionHandler());
        http.formLogin();
        http.httpBasic();

        http.rememberMe()
                .userDetailsService(accountService)
                .key("remember-me");

        SecurityContextHolder.setStrategyName(SecurityContextHolder.MODE_INHERITABLETHREADLOCAL);
    }

}


// org\springframework\security\web\authentication\rememberme\RememberMeAuthenticationFilter.java
public class RememberMeAuthenticationFilter extends GenericFilterBean implements
		ApplicationEventPublisherAware {
...
	public void doFilter(ServletRequest req, ServletResponse res, FilterChain chain)
			throws IOException, ServletException {
		HttpServletRequest request = (HttpServletRequest) req;
		HttpServletResponse response = (HttpServletResponse) res;

		if (SecurityContextHolder.getContext().getAuthentication() == null) {
			Authentication rememberMeAuth = rememberMeServices.autoLogin(request,
					response);

			if (rememberMeAuth != null) {
				// Attempt authenticaton via AuthenticationManager
				try {
					rememberMeAuth = authenticationManager.authenticate(rememberMeAuth);

					// Store to SecurityContextHolder
					SecurityContextHolder.getContext().setAuthentication(rememberMeAuth);

					onSuccessfulAuthentication(request, response, rememberMeAuth);

					if (logger.isDebugEnabled()) {
						logger.debug("SecurityContextHolder populated with remember-me token: '"
								+ SecurityContextHolder.getContext().getAuthentication()
								+ "'");
					}

					// Fire event
					if (this.eventPublisher != null) {
						eventPublisher
								.publishEvent(new InteractiveAuthenticationSuccessEvent(
										SecurityContextHolder.getContext()
												.getAuthentication(), this.getClass()));
					}

					if (successHandler != null) {
						successHandler.onAuthenticationSuccess(request, response,
								rememberMeAuth);

						return;
					}

				}
				catch (AuthenticationException authenticationException) {
					if (logger.isDebugEnabled()) {
						logger.debug(
								"SecurityContextHolder not populated with remember-me token, as "
										+ "AuthenticationManager rejected Authentication returned by RememberMeServices: '"
										+ rememberMeAuth
										+ "'; invalidating remember-me token",
								authenticationException);
					}

					rememberMeServices.loginFail(request, response);

					onUnsuccessfulAuthentication(request, response,
							authenticationException);
				}
			}

			chain.doFilter(request, response);
		}
		else {
			if (logger.isDebugEnabled()) {
				logger.debug("SecurityContextHolder not populated with remember-me token, as it already contained: '"
						+ SecurityContextHolder.getContext().getAuthentication() + "'");
			}

			chain.doFilter(request, response);
		}
	}
...	
}
```

## 커스텀 필터 추가하기

* [LoggingFilter @ github](https://github.com/keesun/spring-security-basic/commit/e2642defe5ad5d40e225b34c75e09392ef4b8a24)

----

`LoggingFilter` 를 만들어서 Filter chain 에 추가해 보자.

```java
// src/main/java/com/iamslash/exsecurity/common/LoggingFilter.java
public class LoggingFilter extends GenericFilterBean {

    private Logger logger = LoggerFactory.getLogger(this.getClass());

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) throws IOException, ServletException {
        StopWatch stopWatch = new StopWatch();
        stopWatch.start(((HttpServletRequest)request).getRequestURI());

        chain.doFilter(request, response);

        stopWatch.stop();
        logger.info(stopWatch.prettyPrint());
    }
}

// src/main/java/com/iamslash/exsecurity/config/SecurityConfig.java
    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http.addFilterBefore(new LoggingFilter(), WebAsyncManagerIntegrationFilter.class);
				...
		}
```

# 스프링 시큐리티 그밖에

## 타임리프 스프링 시큐리티 확장팩

`thymeleaf-extras-springsecurity5` 를 이용하면 thymeleaf 의 다양한 기능을 이용할 수 있다. `build.gradle` 에 다음과 같은 dependency 를 추가한다.

```groovy
implementation 'org.thymeleaf.extras:thymeleaf-extras-springsecurity5'
```

authentication 이 처리된 상태에 따라 `/logout` 혹은 `/login` link 가 보여진다. 그러나 `"${#authorization.expr('isAuthenticated()')}"` type-safe 하지 못하다. 실행해야 오류를 발견할 수 있다.

```html
<!DOCTYPE html>
<html lang="en" xmlns:th="http://www.thymeleaf.org">
<head>
    <meta charset="UTF-8">
    <title>Index</title>
</head>
<body>
    <h1 th:text="${message}">Hello</h1>
    <div th:if="${#authorization.expr('isAuthenticated()')}">
        <h2 th:text="${#authentication.name}">Name</h2>
        <a href="/logout" th:href="@{/logout}">Logout</a>
    </div>
    <div th:unless="${#authorization.expr('isAuthenticated()')}">
        <a href="/login" th:href="@{/login}">Login</a>
    </div>
</body>
</html> 
```

## sec 네임스페이스

`xmlns:sec="http://www.thymeleaf.org/extras/spring-security"` 를 추가하면 type-safe 하게 작성할 수 있다. IntelliJ ultimate version 의 경우 intelli-sense 를 지원한다.

```html
<!DOCTYPE html>
<html lang="en" xmlns:th="http://www.thymeleaf.org" xmlns:sec="http://www.thymeleaf.org/extras/spring-security">
<head>
    <meta charset="UTF-8">
    <title>Index</title>
</head>
<body>
    <h1 th:text="${message}">Hello</h1>
    <div th:if="${#authorization.expr('isAuthenticated()')}">
        <h2 th:text="${#authentication.name}">Name</h2>
        <a href="/logout" th:href="@{/logout}">Logout</a>
    </div>
    <div th:unless="${#authorization.expr('isAuthenticated()')}">
        <a href="/login" th:href="@{/login}">Login</a>
    </div>
</body>
</html> 
```

```html
<div sec:authorize="isAuthenticated()">
<h2 sec:authentication="name">Name</h2>
<a href="/logout" th:href="@{/logout}">Logout</a>
</div>
<div sec:authorize="!isAuthenticated()">
<a href="/login" th:href="@{/login}">Login</a>
</div>
```

## 메소드 시큐리티

* [Method Security @ github](https://github.com/keesun/spring-security-basic/commit/a7c6ccc52dfe53c8f92ce601152dc693800337f6)

-----

Service 의 Method 에 `@Secured, @RolesAllowed` 등을 사용하여 Authorization 을 수행할 수 있다. 즉, 인가된 user 만 Method 를 호출할 수 있다.

```java
// src/main/java/com/iamslash/exsecurity/config/MethodSecurityConfig.java
@Configuration
@EnableGlobalMethodSecurity(securedEnabled = true, prePostEnabled = true, jsr250Enabled = true)
public class MethodSecurityConfig extends GlobalMethodSecurityConfiguration {

    @Override
    protected AccessDecisionManager accessDecisionManager() {
        RoleHierarchyImpl roleHierarchy = new RoleHierarchyImpl();
        roleHierarchy.setHierarchy("ROLE_ADMIN > ROLE_USER");
        AffirmativeBased accessDecisionManager = (AffirmativeBased) super.accessDecisionManager();
        accessDecisionManager.getDecisionVoters().add(new RoleHierarchyVoter(roleHierarchy));
        return accessDecisionManager;
    }
}

// src/main/java/com/iamslash/exsecurity/config/SecurityConfig.java
// src/main/java/com/iamslash/exsecurity/config/SecurityConfig.java
@Configuration
@EnableWebSecurity
public class SecurityConfig extends WebSecurityConfigurerAdapter {
...
    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http.authorizeRequests()
                .mvcMatchers("/", "/info", "/account/**").permitAll()
                .mvcMatchers("/admin").hasRole("ADMIN")
                .mvcMatchers("/user").hasRole("USER")
                .anyRequest().authenticated()
                .expressionHandler(expressionHandler());
        http.formLogin();
        http.httpBasic();

        http.rememberMe()
                .userDetailsService(accountService)
                .key("remember-me");

        SecurityContextHolder.setStrategyName(SecurityContextHolder.MODE_INHERITABLETHREADLOCAL);
    }

    @Bean
    @Override
    public AuthenticationManager authenticationManagerBean() throws Exception {
        return super.authenticationManagerBean();
    }
}

// src/main/java/com/iamslash/exsecurity/form/SampleService.java
@Service
public class SampleService {

    @Secured({"ROLE_USER", "ROLE_ADMIN"})
    public void dashboard() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        UserDetails userDetails = (UserDetails) authentication.getPrincipal();

// src/test/java/com/iamslash/exsecurity/form/SampleServiceTest.java				
@RunWith(SpringRunner.class)
@SpringBootTest
public class SampleServiceTest {

    @Autowired
    SampleService sampleService;

    @Autowired
    AccountService accountService;

    @Autowired
    AuthenticationManager authenticationManager;

    @Test
    public void dashboard() {
        Account account = new Account();
        account.setRole("ADMIN");
        account.setUsername("keesun");
        account.setPassword("123");
        accountService.createNew(account);

        UserDetails userDetails = accountService.loadUserByUsername("keesun");

        UsernamePasswordAuthenticationToken token = new UsernamePasswordAuthenticationToken(userDetails, "123");
        Authentication authentication = authenticationManager.authenticate(token);

        SecurityContextHolder.getContext().setAuthentication(authentication);

        sampleService.dashboard();
    }

} 
```

## @AuthenticationPrincipal

* [@AuthenticationPrincipal @ github](https://github.com/keesun/spring-security-basic/commit/863d4bbe009e4ea1fcf385fdb5775bd19ab64eb8)

-----

`@AuthenticationPrincipal` 을 Controller class 의 Method 에 argument 로 사용하자. Spring security 의 Principal object 를 argument 로 전달받을 수 있다.

```java
// src/main/java/com/iamslash/exsecurity/form/SampleController.java
@Controller
public class SampleController {
	@GetMapping("/")
	public String index(Mode model, @AuthenticationPrincipal UserAccount userAccount) {
		if (userAccount == null) {
			model.addAttribute("message", "Hello Spring Security");
		} else {
			model.addAttribute("message", "Hello, " + userAccount.getUserName());
		}
		return "index";
	}
}
```

## 스프링 데이터 연동

* [Spring Security Data @ github](https://github.com/keesun/spring-security-basic/commit/4ed44a25712c442141f8876a0494e018ab6733ed)

----

`spring-security-data` 를 이용하면 다양한 Spring security data 기능을 이용할 수 있다. 

다음과 같이 `build.gradle` 에 dependency 를 추가한다.

```groovy
implementation 'org.springframework.security:spring-security-data'
```

`@Query` 에서 `SpEL` 로 principal 을 접근한다.

```java
@Query("select b from Book b where b.author.id = ?#{principal.account.id}")
List<Book> findCurrentUserBooks();
```
