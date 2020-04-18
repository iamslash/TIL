# Materials

* [스프링 프레임워크 핵심 기술 @ inflearn](https://www.inflearn.com/course/spring-framework_core/)
  * [원리 src](https://github.com/keesun/javaservletdemo)
  * [설정 src](https://github.com/keesun/demo-boot-web)
  * [활용 src](https://github.com/keesun/demo-web-mvc)

# 스프링 MVC 동작원리

## 스프링 MVC 소개

## 서블릿 소개

* [1: HelloServlet](https://github.com/keesun/javaservletdemo/commit/cca099c7d0e44cb46d4fb991d7d1b996450ac03f)

## 서블릿 애플리케이션 개발

## 서블릿 리스너와 필터

리스너는 웹 애플리케이션에서 발생하는 주요 이벤트를 핸들링할 수 있다. 필터는 Servlet Container 가 특정 Servlet 에게 request 혹은 response 를 전달할 때 호출된다. 필터는 체이닝된다.

* [2. Servlet Listener and filter](https://github.com/keesun/javaservletdemo/commit/c998afe114442244f9c0cc0c3d7a001cb5828174)

## 스프링 IoC 컨테이너 연동

## 스프링 MVC 연동

* [3. Spring MVC](https://github.com/keesun/javaservletdemo/commit/88b840eec71f0e63ddc77766b123a324688ce5b8)

## DispatcherServlet 1 부

* [4. Debugging DispatcherServlet](https://github.com/keesun/javaservletdemo/commit/b985936194a8f7714b360e2c9aaf3c43dfa38863)

## DispatcherServlet 2 부

## DispatcherServlet 3 부

* [5. Adding a ViewResolver bean](https://github.com/keesun/javaservletdemo/commit/8713ecba00d01e148098c8642727dfcbb1239a10)

## 스프링 MVC 구성요소

## 스프링 MVC 동작 원리 마무리

# 스프링 MVC 설정

## 스프링 MVC 빈 설정

* [6. Defatul Strategies](https://github.com/keesun/javaservletdemo/commit/c17051f2134b29757a46f0e5f4165f3dcebed853)
* [7. Without web.xml](https://github.com/keesun/javaservletdemo/commit/11612db08c42e42803ebf1254e609049374ec789)

## @EnableWebMvc

* [8. @EnableWebMVC](https://github.com/keesun/javaservletdemo/commit/aa356f9707b794a9617cfd17831ea05ef7525046)

## WebMvcConfigurer

## 스프링 부트의 스프링 MVC 설정

## 스프링 부트 JSP

## WAR 파일 배포하기

## WebMvcConfigurer 1부 Formatter

* [1. Formatter 설정](https://github.com/keesun/demo-boot-web/commit/183fc75ee425799ffba22e987cbfa37085cfc9cf)

## 도메인 클래스 컨버터

* [2. Domain Class Converter 설정](https://github.com/keesun/demo-boot-web/commit/20eb8b478e08b062b9dbdf18e3f1dc02fa7949d8)

## 핸들러 인터셉터

## 핸들러 인터셉터 구현

* [3. HandlerInterceptor 구현 및 등록](https://github.com/keesun/demo-boot-web/commit/63a8c0935902e9e5f8455f1ea39e1131adce1fa5)

## 리소스 핸들러

## HTTP 메시지 컨버터

* [5. HttpMessageConverter 문자열](https://github.com/keesun/demo-boot-web/commit/7f0c90a177e2c39406a2a372539dcf479d00a951)

## HTTP 메시지 킨버터 2 부 JSON

Spring boot 를 사용한다면 기본적으로 JacksonJSON 2 가 dependency 설정되어 있다.

* [6. HttpMessageConverter JSON](https://github.com/keesun/demo-boot-web/commit/0f3a45d4b6e7be7d231f0ee02a7387520f27eee4)

## HTTP 메시지 컨버터 XML

XML converter 를 사용하고 싶다면 JacksonXML 혹은 JAXB 를 dependency 설정하자.

* [7. HttpMessageConverter XML](https://github.com/keesun/demo-boot-web/commit/f4d0ec00156eccbdfe1d34b982f6af589ef5455e)

## 기타 WebMvcConfigurer 설정

## 스프링 MVC 설정 마무리

# 스프링 MVC 활용

## 스프링 MVC 활용 소개

## 요청 매핑하기 1부 HTTP Method

* [HttpMethod mapping](https://github.com/keesun/demo-web-mvc/commit/b0cf48332b8e7859c0561eb8f6ed08a9f832cea5)

## 요청 매핑하기 2부 URI 패턴

* [URI Pattern Mapping](https://github.com/keesun/demo-web-mvc/commit/f3dd5bce25b01456f718eb84d3f39538dc5f7844)

## 요청 매핑하기 3부 미디어 타입

* [Consumes and Produces](https://github.com/keesun/demo-web-mvc/commit/271e4cccfd24bba420c7f82290c9c2ab6c3a6860)

## 요청 매핑하기 4부 헤더와 매개변수

* [Headers and Request Parameters](https://github.com/keesun/demo-web-mvc/commit/8a96ede102cb030cee707a40a82a5f6930d4bbc2)

## 요청 매핑하기 5부 HEAD 와 OPTIONS

* [OPTIONS and HEADER](https://github.com/keesun/demo-web-mvc/commit/9985caf224ae467fdf961f076acf247e9bfc7f00)

특별한 구현을 하지 않아도 Spring Web MVC 는 `HEAD, OPTIONS` 를 handling 한다.

* `HEAD`
  * GET 요청과 동일하다. Request Body 는 없고 Response Header 만 온다.
* `OPTIONS`
  * 사용할 수 있는 HTTP Method 의 목록을 Allow header 에 얻어온다.

## 요청 매핑하기 6부 커스텀 애노테이션

* [Custom annotation](https://github.com/keesun/demo-web-mvc/commit/b8e99082018b10882f4f53dfe8b2d0b6f3e79910)

## 요청 매핑하기 7부 연습문제

* [Mapping Quiz](https://github.com/keesun/demo-web-mvc/commit/44700614e1cd91598d8e494c3567c18a82fc2c6a)

## 핸들러 메소드 1 부 아규먼트와 리턴타입

* Argument
  * `WebRequest, NativeWebRequest, ServletRequest, ServletResponse, HttpServletRequest, HttpServletResponse`
  * `InputStream, Reader, OutputStream, Writer` : Request 본문 읽을때, Response 본문 쓸때 사용
  * `PushBuilder` : Spring 5, HTTP/2 resource push 에 사용
  * `HttpMethod` : Get, POST 등의 정보
  * `Locale, TimeZone, ZoneId` : LocaleResolver 가 분석한 Request 의 Locale 정보
  * `@PathVariable` : URI 템플릿 변수 읽을 때 사용
  * `@MatrixVariable` : URI 결로중 키에 해당하는 값을 읽어 올 때 사용
  * `RequestParam` : Request Parameter 를 읽을 때 사용
  * `RequestHeader` : Header 를 읽을 때 사용
* Return
  * `@ResponseBody` : HttpMessageConverter 를 사용해 Response 본문으로 사용
  * `HttpEntity, RepsponseEntity` : Response 본문 뿐 아니라 Header 까지 포함한 Response 를 만들 때 사용
  * `String` : ViewResolver 를 이용하여 View 를 검색함. View 이름에 해당한다.
  * `View` : 암묵적인 모델 정보를 렌더링할 View instance
  * `Map, Model` : RequestToViewNameTranslator 를 통해서 암묵적으로 판단한 View Rendering 할 때 사용할 모델 정보
  * `@ModelAttribute` : RequestToViewNameTranslator 를 통해서 암묵적으로 판단한 View Rendering 할 때 사용할 모델 정보에 추가한다.

## 핸들러 메소드 2 부 URI 패턴

* [@PathVariable & @MatrixVariable](https://github.com/keesun/demo-web-mvc/commit/44db6a379c5c76af9a6e7601d69164e960d8a763)

## 핸들러 메소드 3 부 매개변수 (단순타입)

* [@RequestParam](https://github.com/keesun/demo-web-mvc/commit/3956a0ebe320ed0a57c47a11d8445f781f821e61)

## 핸들러 메소드 4 부 폼 서브밋

* [Thymeleaf Form](https://github.com/keesun/demo-web-mvc/commit/25fc8fa13c31e1e33bf85ae007d013808b15bb85)

## 핸들러 메소드 5 부 @ModelAttribute

* [@ModelAttribute](https://github.com/keesun/demo-web-mvc/commit/3de65495d486b9143febf84261f567e02eb85f4f)

## 핸들러 메소드 6 부 @Validated

* [@Validated](https://github.com/keesun/demo-web-mvc/commit/9fb7d3aa95c7559dbce6015c92164c70fe19c93a)

## 핸들러 메소드 7 부 폼 서브밋 에러처리

* [Form submit with error handling](https://github.com/keesun/demo-web-mvc/commit/115924b0596b2a83f26ea899418968d94f0c5399)

## 핸들러 메소드 8부 @SessionAttributes

* [@SessionAttributes](https://github.com/keesun/demo-web-mvc/commit/fbb37cca93303de5755bace2178925eecf047911)

## 핸들러 메소드 9부 멀티 폼 서브밋

* [Multi-form submit with @SessionAttributes and SessionState](https://github.com/keesun/demo-web-mvc/commit/031746c29ef742e11235252dd3d4f40df04203ed)

## 핸들러 메소드 10 부 @SessionAttribute

* [@SessionAttribute](https://github.com/keesun/demo-web-mvc/commit/30d30fcf4f2af876bb25c7c25ae8cdc7e940b0fb)

## 핸들러 메소드 11부 RedirectAttributes

* [RedirectAttributes](https://github.com/keesun/demo-web-mvc/commit/5a9b24a4e200bc18876e498ffef42dd15330f7db)

## 핸들러 메소드 12 부 FlashAttributes

* [Flash Attributes](https://github.com/keesun/demo-web-mvc/commit/dce1ad589769df7cd8d4b71bd2f2a386441a4964)

## 핸들러 메소드 13부 MultipartFile

* [MultipartFile](https://github.com/keesun/demo-web-mvc/commit/f02ad5b17e0042e1d1900052c880cb7b73df9f27)
* [File Download](https://github.com/keesun/demo-web-mvc/commit/7a3b6fb047694a4745b1bf972d184c86054a2802)

## 핸들러 메소드 14부 ResponseEntity

## 핸들러 메소드 15부 @RequestBody & HttpEntity

* [@requestbody & HttpEntity](https://github.com/keesun/demo-web-mvc/commit/8490538297a631b2d58a29e1e7abf955ce916c9c)

## 핸들러 메소드 16부 @ResponseBody & ResponseEntity

* [@responsebody & ResponseEntity](https://github.com/keesun/demo-web-mvc/commit/ed34b64823547829abdab0d0f86b7df6518e52ef)

## 핸들러 메소드 17부 정리 및 과제

* [spring-petclinic](https://github.com/spring-projects/spring-petclinic)

## 모델 @ModelAttribute

* [@ModelAttribute Method](https://github.com/keesun/demo-web-mvc/commit/f8d87bc70d558d87694cdff8b881d796eb72a64c)

## 데이터바인더 @InitBinder

* [WebDataBinder @InitBinder](https://github.com/keesun/demo-web-mvc/commit/ccb7ceb63b17546e8b3695ba4e7b744820d6f95c)

## 예외 처리 핸들러  @ExceptionHandler

* [@ExceptionHandler](https://github.com/keesun/demo-web-mvc/commit/176f9b6d86ae7db8f509dccfe56495146fbc5ee8)

## 전역 컨트롤러 @ControllerAdvice

* [@ControllerAdvice](https://github.com/keesun/demo-web-mvc/commit/c6b95af66da2749f2c7a92f1ad4ff566982bf2fa)

## 스프링 MVC 강좌 마무리
