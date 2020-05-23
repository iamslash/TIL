# Abstract

This is about annotations of Spring Boot Framework.

# @ConfigurationProperties

특정 Bean 은 Configuration class 를 이용하여 생성한다. 이때 그 Bean 의 설정을 넘겨줘야 한다. 이 설정을 Properties class 라고 한다. `@ConfigurationProperties` 는 Properties class 에 생성할 Bean 의 이름과 함께 attach 한다.

```java
@ConfigurationProperties("user")
public class UserProperties {
	private String name;
	private int age;
	...
}
```

# @EnableConfigurationProperties

Configuration class 에 생성할 Bean 의 Properties class 를 넘겨줘야 한다. `@EnableConfigurationProperties` 는 Configuration class 에 넘겨줄 Properties class 와 함께 attach 한다.

```java
@Configuration
@EnableConfigurationProperties(UserProperties.class)
public class UserConfiguration {
	@Bean
	@ConditionalOnMissingBean
	public User user(UserProperties properties) {
		User user = new User();
		user.setAge(properties.getAge());
		user.setName(properties.getName());
		return user;
	}
}
```

# @TestPropertySource



