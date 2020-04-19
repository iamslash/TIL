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

## 34.	관심 주제와 지역 정보 관리 기능 미리보기

## 35.	관심 주제 도메인

## 36.	관심 주제 등록 뷰

## 37.	관심 주제 등록 기능 구현

* [Add tag](https://github.com/hackrslab/studyolle/commit/511ab7d87c757dcab47942c35673284bc4a88ede)

다음과 같이 develop 단계의 DB 설정을 application.properties 에 설정한다.

```conf
spring.profiles.active=local

# 개발할 때에만 create-drop 또는 update를 사용하고 운영 환경에서는 validate를 사용합니다.
spring.jpa.hibernate.ddl-auto=create-drop

# 개발시 SQL 로깅을 하여 어떤 값으로 어떤 SQL이 실행되는지 확인합니다.
spring.jpa.properties.hibernate.format_sql=true
logging.level.org.hibernate.SQL=DEBUG
logging.level.org.hibernate.type.descriptor.sql.BasicBinder=TRACE
```

## 38.	관심 주제 조회

* [Read Tag](https://github.com/hackrslab/studyolle/commit/2f6fe2174940cedda2c1efd72773f80b15d4af96)

## 39.	관심 주제 삭제

* [Remove Tag](https://github.com/hackrslab/studyolle/commit/18f4091300e894f4c067ba861777acc0e06cfa6e)

## 40.	관심 주제 자동완성

* [Tag suggestion](https://github.com/hackrslab/studyolle/commit/3c95d0d13739353410a9da19594e134f11b35ae2)

## 41.	관심 주제 테스트

* [Test updating tags](https://github.com/hackrslab/studyolle/commit/3a6bfc286847b06aa248563fc80a15f125262f79)

## 42.	 지역 도메인

* [Zone domain and init data](https://github.com/hackrslab/studyolle/commit/411b7f030e10b735711487183346f919782ead5d)
* [URL mapping refactoring](https://github.com/hackrslab/studyolle/commit/fd5c16f60789007b56f25bf8106b4ea7b5bd3fc2)
* [Update CurrentUser to CurrentAccount](https://github.com/hackrslab/studyolle/commit/cd6fe53531892490f31bc2d66ebc8e4b92a55c73)

## 43.	 지역 정보 추가/삭제/테스트

* [Read, Add, Remove and Test Zones](https://github.com/hackrslab/studyolle/commit/3d94d3bd3ca23efef3be0b7da0ccbae51851cba0)

# DB 와 Email 설정

## 44.	PostgreSQL 설치 및 설정

## 45.	인텔리J 데이터베이스 탭

## 46.	SMTP 설정

Gmail 을 SMTP server 로 사용

* application.properties

```conf
spring.mail.host=smtp.gmail.com
spring.mail.port=587
spring.mail.username=iamslash@gmail.com
spring.mail.password=xxxxxxxx
spring.mail.properties.mail.smtp.auth=true
spring.mail.properties.mail.smtp.timeout=5000
spring.mail.properties.mail.smtp.starttls.enable=true
```

## 47.	EmailService 추상화

* [Design and implmented EmailService](https://github.com/hackrslab/studyolle/commit/55325fb0f71b11cfeaa81afcad71e4d359feb25f)

## 48.	HTML 이메일 전송하기

* [Sending HTML email](https://github.com/hackrslab/studyolle/commit/caae965ce8076c9ad1a67059185da58966537fae)

# Study

## 49.	스터디 관기 기능 미리보기
## 50.	스터디 도메인

* [Study domain](https://github.com/hackrslab/studyolle/commit/309b0611effafa9ef0b292035058f8fab9c5188f)
* [JPA 까먹지 말자! (2)](http://wonwoo.ml/index.php/post/category/jpa)
  * @Basic LAZY, EAGER 참고
  
## 51.	스터디 개설

* [Add Study](https://github.com/hackrslab/studyolle/commit/3d0e258bf89ab856fa53e17510ba4586ca8d672f)
* [[Spring] Transactional 정리 및 예제](https://goddaehee.tistory.com/167)
  * isolation 참고
  
## 52.	스터디 조회

* [View Study](https://github.com/hackrslab/studyolle/commit/1a5abcf6725d8482b97c90d62935c88aaf9d893e)
* [Added tests for creating and querying study](https://github.com/hackrslab/studyolle/commit/09113e70a098e13a9812ebd2b3d147da4ba49df1)
* [[Spring JPA #23] Spring JPA EntityGraph](https://engkimbs.tistory.com/835)
  * EntityGraph 참고

Fetch strategy 를 grouping 해서 적용할 수 있다.

## 53.	스터디 구성원 조회

* [View study members](https://github.com/hackrslab/studyolle/commit/bc92f41c8cb2f3b6d11c47cc700c97a50fdae4b1)

## 54.	스터디 설정 - 소개 수정

* [Study settings - description](https://github.com/hackrslab/studyolle/commit/a00c9aac49e80e645104538364bb9c52a860dec3)

## 55.	스터디 설정 - 배너

* [Study settings - banner](https://github.com/hackrslab/studyolle/commit/920185dcf2e0004dce20df8a9cc334749efe28cb)

## 56.	스터디 설정 - 태그/지역

* [Study settings - tags and zones](https://github.com/hackrslab/studyolle/commit/4afe1c31d4954cc97812033ab797a696709cd95a)

## 57.	스터디 설정 - 상태 변경

* [Study settings - status](https://github.com/hackrslab/studyolle/commit/9921038977c2e10dc2c9353ea84b34f4b930d763)

## 58.	스터디 설정 - 경로 및 이름 수정

* [Study settings - path and title](https://github.com/hackrslab/studyolle/commit/31b0caa9756bf3c42b021a38eca187c5ed3177a0)

## 59.	스터디 삭제

* [Study delete](https://github.com/hackrslab/studyolle/commit/05525b6b52b735456219f848fdd0370c3c4870f1)

## 60.	스터디 참여 및 탈퇴

* [Study join and leave](https://github.com/hackrslab/studyolle/commit/a7b6aec37d410c624c52f9a3c2a798cf18b72824)
* [Add tests for StudyControllers](https://github.com/hackrslab/studyolle/commit/a7cb861c9cf3a7995a28a0ea76531c0176046762)

# Meetup

## 61.	모임 관리 기능 미리보기
## 62.	모임 도메인



## 63.	모임 만들기 뷰
## 64.	모임 만들기 폼 서브밋
## 65.	모임 조회
## 66.	모임 목록 조회
## 67.	모임 수정
## 68.	모임 취소
## 69.	모임 참가 신청 및 취소
## 70.	모임 참가 신청 수락 취소 및 출석 체크

# Package and Refactoring

## 71.	패키지 구조 정리
## 72.	테스트 클래스 정리
## 73.	테스트 DB를 PostgreSQL로 전환

# Notification

## 74.	알림 기능 미리보기
## 75.	알림 도메인
## 76.	알림 인프라 설정
## 77.	스터디 개설 알림
## 78.	알림 아이콘 변경
## 79.	알림 목록 조회 및 삭제
## 80.	관심있는 스터디 변경 알림
## 81.	모임 관련 알림

# Search 

## 82.	검색 기능 미리보기
## 83.	검색 기능 구현
## 84.	N+1 Select 문제 해결
## 85.	페이징 적용
## 86.	검색 뷰 개선
## 87.	로그인 전 첫 페이지
## 88.	로그인 후 첫 페이지

# Handling errors and Deployment

## 89.	에러 핸들러 및 뷰 추가
## 90.	배포시 고려할 것

