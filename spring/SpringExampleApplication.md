# Materials

* [스프링과 JPA 기반 웹 애플리케이션 개발](https://www.inflearn.com/course/%EC%8A%A4%ED%94%84%EB%A7%81-JPA-%EC%9B%B9%EC%95%B1)
  * [src](https://github.com/hackrslab/studyolle)

# 시작하기

# 회원가입

## 3. Account Domain Class

* [Add Account domain class](https://github.com/hackrslab/studyolle/commit/440218ca109cf915cd8ad2593a47e367c16eaeea)

## 4. 회원 가입 컨트롤러

* [Added handler for sign up form](https://github.com/hackrslab/studyolle/commit/11eb65278dbb0dc3b99cf9d1352e9b497aa08106)

## 5. 회원 가입 뷰

* [SignUp view](https://github.com/hackrslab/studyolle/commit/bc88d9e4d7f892b09146c94260dcf144b05cd1c6)

## 6.	회원 가입: 폼 서브밋 검증

* [Handling the signUp form submit](https://github.com/hackrslab/studyolle/commit/7b1942959cf985c263d29033123ab07f4c6c4e91)

## 7.	회원 가입: 폼 서브밋 처리

## 8.	회원 가입: 리팩토링 및 테스트

* [Refactoring and write tests for the signup](https://github.com/hackrslab/studyolle/commit/4c08cde524baf3f0888ba2e2358b5df03e83b017)


```java
@AutoConfigureMockMvc
class AccountControllerTest {
    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private AccountRepository accountRepository;

    @MockBean
    JavaMailSender javaMailSender;
    ...
    @DisplayName("회원 가입 처리 - 입력값 정상")
    @Test
    void signUpSubmit_with_correct_input() throws Exception {
        mockMvc.perform(post("/sign-up")
                .param("nickname", "keesun")
                .param("email", "keesun@email.com")
                .param("password", "12345678")
                .with(csrf()))
                .andExpect(status().is3xxRedirection())
                .andExpect(view().name("redirect:/"));

        assertTrue(accountRepository.existsByEmail("keesun@email.com"));
        // javaMailSender 의 send 가 SimpleMailMessage argument 와 함께 호출됬어야 한다.
        then(javaMailSender).should().send(any(SimpleMailMessage.class));
    }    
}
```

## 9.	회원 가입: 패스워드 인코더

* [Encoding password](https://github.com/hackrslab/studyolle/commit/9b77bc945127f2375888ea8bc1580691a4b34a37)

password 는 bcrypt 로 random 한 salt 와 함께 hash value 를 생성하여 Data Base 에 저장한다. 유저가 password 를 입력하면
bcrypt 로 random 한 salt 와 함께 hash value 를 생성하여 DB 의 password 와 match 되는지 확인한다. bcrypt algorithm 의 특성때문에 random salt 에 대해서 match 가 가능하다???

## 10.	회원 가입: 인증 메일 확인

* [Check email token](https://github.com/hackrslab/studyolle/commit/78eb1c67701ef1bbcb13c156034b8f89e77b8be2)

## 11.	회원 가입: 인증 메일 확인 테스트 및 리팩토링

* [refactoring](https://github.com/hackrslab/studyolle/commit/cef68c6d000e4168a97f653bde79cf7d169a3617)

## 12.	회원 가입: 가입 완료 후 자동 로그인

* [Login after successful signup and checked email](https://github.com/hackrslab/studyolle/commit/e18d38988a618598f1d1477a8fa55cacadd0dbc1)

## 13.	회원 가입: 메인 네비게이션 메뉴 변경

* [View change according to authentication](https://github.com/hackrslab/studyolle/commit/bf53a8ed0afa6cd596a49b392a620b4451cb0e2a)

다음과 같이 thymeleaf 의 security 기능을 이용해 보자.

build.gradle 에 `org.thymeleaf.extras:thymeleaf-extras-springsecurity5` dependency 를 추가한다.

HTML template 에 다음과 같이 xmlns:nec 를 html 의 attribute 로 추가하고 `sec` namespace 의 attribute 를 사용할 수 있다.

```html
<html lang="en"
      xmlns:th="http://www.thymeleaf.org"
      xmlns:sec="http://www.thymeleaf.org/extras/spring-security">
...
        <ul class="navbar-nav justify-content-end">
            <li class="nav-item" sec:authorize="!isAuthenticated()">
                <a class="nav-link" th:href="@{/login}">로그인</a>
            </li>
...                  
```

## 14.	프론트엔드 라이브러리 설정

* [Integrate NPM into Spring Boot](https://github.com/hackrslab/studyolle/commit/efa269199ddb207c8e12b2ead125d28dba7f0968)

`/node_modules/**` HTTP Request 는 Security Filter 를 적용되지 않도록 한다.

```java
    @Override
    public void configure(WebSecurity web) throws Exception {
        web.ignoring()
                .mvcMatchers("/node_modules/**")
                .requestMatchers(PathRequest.toStaticResources().atCommonLocations());
    }
}
```

## 15.	뷰 중복 코드 제거

* [Use fragments](https://github.com/hackrslab/studyolle/commit/cb63f07249647605a5606bb7c0be88b56e61d260)

## 16.	첫 페이지 보완

* [Updated home page](https://github.com/hackrslab/studyolle/commit/34b002c212c674fcf353b1cb21c5df54f0764123)

*	네비게이션 바에 Fontawesome으로 아이콘 추가
*	이메일 인증을 마치지 않은 사용자에게 메시지 보여주기
*	jdenticon으로 프로필 기본 이미지 생성하기

## 17.	현재 인증된 사용자 정보 참조

* [Added @currentuser](https://github.com/hackrslab/studyolle/commit/390b0b10b39ee3e5767c3cd029bf4f833916c394)

## 18.	가입 확인 이메일 재전송

* [Resend confirm email](https://github.com/hackrslab/studyolle/commit/18276fffcbf0aa8ecf5a0de236d7ab8cff8d50c7)

## 19.	로그인 / 로그아웃

* [Login and Logout](https://github.com/hackrslab/studyolle/commit/fa7e83a6259b4d1215c62d1c84d04c513d37252e)

## 20.	로그인 / 로그아웃 테스트

* [Login and Logout test](https://github.com/hackrslab/studyolle/commit/fb43cf720bb7c83cbb488eb0b8f55d86893fed4e)

테스트 코드를 작성할 때 Test Reqeust 에 CSRF token 을 포함시키기 위해`csrf()` 를 꼭 추가해야 한다.

```java
    @DisplayName("이메일로 로그인 성공")
    @Test
    void login_with_email() throws Exception {
        mockMvc.perform(post("/login")
                .param("username", "keesun@email.com")
                .param("password", "12345678")
                .with(csrf()))
                .andExpect(status().is3xxRedirection())
                .andExpect(redirectedUrl("/"))
                .andExpect(authenticated().withUsername("keesun"));
    }
```

## 21.	로그인 기억하기 (RememberMe)

* [Support RememberMe](https://github.com/hackrslab/studyolle/commit/6aa99e82436a216cbcd8d3d8d7e05953b09112f1)

## 22.	프로필 뷰

* [Profile view](https://github.com/hackrslab/studyolle/commit/f573b43a5f8abf039e2b2eb65928e0d451e6d920)

## 23.	Open EntityManager (또는 Session) In View 필터

* [Bug fix for checking email token](https://github.com/hackrslab/studyolle/commit/e98b3f3233dd643a58cc24b9c7912cb0cc1a7d81)

# 계정설정

## 24.	프로필 수정 폼

* [Profile update form](https://github.com/hackrslab/studyolle/commit/dabd12e4b5df6646b260f705c5b555a33d0208e0)

## 25.	프로필 수정 처리

* [Update Profile](https://github.com/hackrslab/studyolle/commit/3492f156446312b59e3c97679d3ee8bb2bbf14c3)

## 26.	프로필 수정 테스트

* [Testing update profile](https://github.com/hackrslab/studyolle/commit/6e6c1571e51176590acb44a2d13e911582430036)

----

인증된 사용자를 `@WithAccount` 으로 mocking 할 수 없다. 실제 DB 에 저장된 정보와 같은 Authentication 이 필요하기 때문이다.

다음과 같은 방법으로 Mocking 한다.

* `@WithAccount` 를 제작한다.

```java
@Retention(RetentionPolicy.RUNTIME)
@WithSecurityContext(factory = WithAccountSecurityContextFacotry.class)
public @interface WithAccount {
  String value();
}
```

* SecurityContextFactory 를 implement 한 `WithAccountSecurityContextFactory` 를 구현한다..

```java
@RequiredArgsConstructor
public class WithAccountSecurityContextFacotry implements WithSecurityContextFactory<WithAccount> {

  private final AccountService accountService;

  @Override
  public SecurityContext createSecurityContext(WithAccount withAccount) {
    String nickname = withAccount.value();

    SignUpForm signUpForm = new SignUpForm();
    signUpForm.setNickname(nickname);
    signUpForm.setEmail(nickname + "@email.com");
    signUpForm.setPassword("12345678");
    accountService.processNewAccount(signUpForm);

    UserDetails principal = accountService.loadUserByUsername(nickname);
    Authentication authentication = new UsernamePasswordAuthenticationToken(principal, principal.getPassword(), principal.getAuthorities());
    SecurityContext context = SecurityContextHolder.createEmptyContext();
    context.setAuthentication(authentication);
    return context;
  }
}
```

* 테스트 코드에서 `@WithAccount` 를 사용한다.

```java
    @WithAccount("keesun")
    @DisplayName("프로필 수정 폼")
    @Test
    void updateProfileForm() throws Exception {
        mockMvc.perform(get(SettingsController.SETTINGS_PROFILE_URL))
                .andExpect(status().isOk())
                .andExpect(model().attributeExists("account"))
                .andExpect(model().attributeExists("profile"));
    }
```

## 27.	프로필 이미지 변경

* [Updat profile image](https://github.com/hackrslab/studyolle/commit/518789bba4a0bcff726494a9af52d67b0a4f429e)

## 28.	패스워드 수정

* [Update password](https://github.com/hackrslab/studyolle/commit/7b964084be8234022f52bea52137bcfd03a09de6)

## 29.	패스워드 수정 테스트 

* [Update password test](https://github.com/hackrslab/studyolle/commit/db8e04574329a1dc851976a6a068e125fa590bc2)

## 30.	알림 설정

* [Update notifications](https://github.com/hackrslab/studyolle/commit/d24b4181f04b5de850b91764353c0b4aef2df2a0)

----

다음은 [Thymeleaf 의 5 가지 expression](https://www.thymeleaf.org/doc/articles/standarddialect5minutes.html) 이다.

* `${...}` : Variable expressions. These are OGNL expressions (or Spring EL if you have spring integrated)
* `*{...}` : Selection expressions. Same as above, excepted it will be executed on a previously selected object only
* `#{...}` : Message (i18n) expressions. Used to retrieve locale-specific messages from external sources
* `@{...}` : Link (URL) expressions. Used to build URLs
* `~{...}` : Fragment expressions. Represent fragments of markup and move them around templates

## 31.	ModelMapper 적용

* [Apply ModelMapper](https://github.com/hackrslab/studyolle/commit/7d80e2802bd4d59a5cb2be38a078ec10a4cd6580)
* [Refactoring](https://github.com/hackrslab/studyolle/commit/3de4df955d19e202a86b3b7c8f770af5852349f3)

----

`org.modelmapper:modelmapper` 는 object 의 properties 를 다른 object 의 properties 로 mapping 해준다.

## 32.	닉네임 수정

* [Update nickname](https://github.com/hackrslab/studyolle/commit/c17d1c83b6fe982285d6e8c67a5320be42214d85)

## 33.	패스워드를 잊어버렸습니다

* [Forgot password](https://github.com/hackrslab/studyolle/commit/d655c1652469bf8f7320b30a6847909510d63a30)

# 관심주제와 지역정보











# DB 와 Email 설정

# Study

# Meetup

# Package and Refactoring

# Notification

# Search 

# Handling errors and Deployment

