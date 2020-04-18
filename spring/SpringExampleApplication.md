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



# 계정설정

# 관심주제와 지역정보

# DB 와 Email 설정

# Study

# Meetup

# Package and Refactoring

# Notification

# Search 

# Handling errors and Deployment

