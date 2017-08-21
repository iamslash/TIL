# Abstract

- spring framework에 대해 적는다.

# Tutorial of STS

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

# Tutorial of springboot 

- [springboot manual installation](https://docs.spring.io/spring-boot/docs/current/reference/html/getting-started-installing-spring-boot.html#getting-started-manual-cli-installation)
  에서 zip을 다운받아 `d:\local\src\`에 압축을 해제하자.
- 환경설정변수 SPRING_HOME을 만들자. `d:\local\src\D:\local\src\spring-1.5.6.RELEASE`
- 환경설정변수 PATH에 `%SPRING_HOME%\bin`을 추가하자.
- command shell에서 `spring version`을 실행한다.
- command shell에서 `spring init'을 이용하여 새로운 프로젝트를 제작할 수 있다.
