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

# 계정설정

# 관심주제와 지역정보

# DB 와 Email 설정

# Study

# Meetup

# Package and Refactoring

# Notification

# Search 

# Handling errors and Deployment

