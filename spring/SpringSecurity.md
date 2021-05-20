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
