- [Annotation Under The Hood](#annotation-under-the-hood)
- [Reading Merged Annotions Flow](#reading-merged-annotions-flow)
- [`@SpringBootApplication`](#springbootapplication)
- [`@SpringBootConfiguration`](#springbootconfiguration)
- [`@Configuration`](#configuration)
- [`@PropertySource`](#propertysource)
- [`@ComponentScan`](#componentscan)
- [`@Import`](#import)
- [`@ImportResource`](#importresource)
- [`@Bean`](#bean)
- [`@EnableAutoConfiguration`](#enableautoconfiguration)
- [`@Repository`](#repository)
- [`@Transactional`](#transactional)
- [`@Service`](#service)
- [`@Component`](#component)
- [`@Controller`](#controller)
- [`@RestController`](#restcontroller)
- [`@Validation`](#validation)

----

# Annotation Under The Hood

Spring Application 은 어떻게 Annotation 을 다루고 있는지 살펴보자. Annotation 을
다루는 방법은 다음과 같은 것들이 있다.

* `Processor` Class 를 상속한 Class 를 정의해서 특정 Annotation 을 구현한다.
  * 아래의 예는 `@Foo` Class 를 처리하고 있다. [java_annotation](/java/java_annotation.md#annotation-processing) 참고.

```java
public class FooProcessor implements Processor {
  @Override
  public Set<String> getSupportedAnnotationTypes() {
    return Set.of(Foo.class.getName());
  }
  @Override
  public SourceVersion getSupportedSourceVersion() {
    return SourceVersion.latestSupported();
  }
  @Override
  public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
    Set<? extends Element> elements = roundEnv.getElementsAnnotationWith(Foo.class);
    for (Element el : elements) {
      Name elName = el.getSimpleName();
      if (el.getKind() != ElementKind.INTERFACE) {
        processingEnv.getMessager().printMessage(Diagnostic.Kind.ERROR, "Foo annotation can not be used on " + elName);
      } else {
        processingEnv.getMessager().printMessage(Diagnostic.Kind.NOTE, "Processing " + elName);
      }
    }
    return true;
  }
}
```

* Class 에 부착된 Annotation 을 읽어와서 특정 Annotation 에 약속된 일을
  수행한다. 
  * 아래의 예는 `@ComponentScan` Annotation 을 `ComponentScan.class` 로 특정하고 있다.

```java
// org.springframework.context.annotation.ConfigurationClassParser
class ConfigurationClassParser {
...
	@Nullable
	protected final SourceClass doProcessConfigurationClass(
			ConfigurationClass configClass, SourceClass sourceClass, Predicate<String> filter)
		// Process any @ComponentScan annotations
		Set<AnnotationAttributes> componentScans = AnnotationConfigUtils.attributesForRepeatable(
				sourceClass.getMetadata(), ComponentScans.class, ComponentScan.class);
```


# Reading Merged Annotions Flow

기본적으로 Class 의 Annotation 을 `getAnnotations()` 로 읽어오면 그 Class 에
부착된 Annotation 만 읽어온다. 부착된 Annotation 에 부착된 Annotation 을
읽어오려면 Annotation Graph 를 만들어서 읽어야 한다. [JavaAnnotations](/java/java_annotation.md#merged-annotations) 참고

`@SpringBootApplication` 은 다음과 같이 정의되어 있다. `@interface SpringBootApplication` 에 부착된 Annotation 들을 어떻게 읽어오는 걸까?

```java
// org.springframework.boot.autoconfigure.SpringBootApplication
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Inherited
@SpringBootConfiguration
@EnableAutoConfiguration
@ComponentScan(excludeFilters = { @Filter(type = FilterType.CUSTOM, classes = TypeExcludeFilter.class),
		@Filter(type = FilterType.CUSTOM, classes = AutoConfigurationExcludeFilter.class) })
public @interface SpringBootApplication {
```

다음과 같이 MergedAnnotation 을 이용하는 걸까?

```java
// org.springframework.boot.test.context.SpringBootContextLoader
public class SpringBootContextLoader extends AbstractContextLoader {
...
	protected String[] getArgs(MergedContextConfiguration config) {
		return MergedAnnotations.from(config.getTestClass(), SearchStrategy.TYPE_HIERARCHY).get(SpringBootTest.class)
				.getValue("args", String[].class).orElse(NO_ARGS);
	}
```

# `@SpringBootApplication`

`@SpringBootApplication` 를 부착하면 부착된 Class 는 다음과 같은 기능을 갖는다.

* `@SpringBootConfiguration` 때문에 `@Configuration` Class 가 된다.
* `@EnableAutoConfiguration` 때문에 auto-configuration 을 실행한다.
* `@ComponentScan` 때문에 component scanning 을 실행한다.

```java
// org.springframework.boot.autoconfigure.SpringBootApplication
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Inherited
@SpringBootConfiguration
@EnableAutoConfiguration
@ComponentScan(excludeFilters = { @Filter(type = FilterType.CUSTOM, classes = TypeExcludeFilter.class),
		@Filter(type = FilterType.CUSTOM, classes = AutoConfigurationExcludeFilter.class) })
public @interface SpringBootApplication {
```

# `@SpringBootConfiguration`

`@SpringBootConfiguration` 를 부착하면 부착된 Class 는 다음과 같은 기능을 갖는다.

* `@Configuration` 때문에 `@Configuration` Class 가 된다. 즉, `@Bean` Method 는 Bean 을 생성할 수 있다. 

```java
// org.springframework.boot.SpringBootConfiguration
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Configuration
public @interface SpringBootConfiguration {
```

# `@Configuration`

`@Bean` Method 가 return 하는 Object 를 Bean 으로 등록한다.

```java
// org.springframework.context.annotation.ConfigurationClassParser
class ConfigurationClassParser {
...
	@Nullable
	protected final SourceClass doProcessConfigurationClass(
			ConfigurationClass configClass, SourceClass sourceClass, Predicate<String> filter)
			throws IOException {
	...
		// Process any @ComponentScan annotations
		Set<AnnotationAttributes> componentScans = AnnotationConfigUtils.attributesForRepeatable(
				sourceClass.getMetadata(), ComponentScans.class, ComponentScan.class);
		if (!componentScans.isEmpty() &&
				!this.conditionEvaluator.shouldSkip(sourceClass.getMetadata(), ConfigurationPhase.REGISTER_BEAN)) {
			for (AnnotationAttributes componentScan : componentScans) {
				// The config class is annotated with @ComponentScan -> perform the scan immediately
				Set<BeanDefinitionHolder> scannedBeanDefinitions =
						this.componentScanParser.parse(componentScan, sourceClass.getMetadata().getClassName());
				// Check the set of scanned definitions for any further config classes and parse recursively if needed
				for (BeanDefinitionHolder holder : scannedBeanDefinitions) {
					BeanDefinition bdCand = holder.getBeanDefinition().getOriginatingBeanDefinition();
					if (bdCand == null) {
						bdCand = holder.getBeanDefinition();
					}
					if (ConfigurationClassUtils.checkConfigurationClassCandidate(bdCand, this.metadataReaderFactory)) {
						parse(bdCand.getBeanClassName(), holder.getBeanName());
					}
				}
			}
		}				

```

# `@PropertySource`

```java
// org.springframework.context.annotation.ConfigurationClassParser
class ConfigurationClassParser {
...
	@Nullable
	protected final SourceClass doProcessConfigurationClass(
			ConfigurationClass configClass, SourceClass sourceClass, Predicate<String> filter)
			throws IOException {
	...
		// Process any @PropertySource annotations
		for (AnnotationAttributes propertySource : AnnotationConfigUtils.attributesForRepeatable(
				sourceClass.getMetadata(), PropertySources.class,
				org.springframework.context.annotation.PropertySource.class)) {
			if (this.environment instanceof ConfigurableEnvironment) {
				processPropertySource(propertySource);
			}
			else {
				logger.info("Ignoring @PropertySource annotation on [" + sourceClass.getMetadata().getClassName() +
						"]. Reason: Environment must implement ConfigurableEnvironment");
			}
		}
```


# `@ComponentScan`

특정 package 안에 존재하는 `@Component` Class 들을 Bean 으로 등록한다.

```java
// org.springframework.context.annotation.ConfigurationClassParser
class ConfigurationClassParser {
...
	@Nullable
	protected final SourceClass doProcessConfigurationClass(
			ConfigurationClass configClass, SourceClass sourceClass, Predicate<String> filter)
		// Process any @ComponentScan annotations
		Set<AnnotationAttributes> componentScans = AnnotationConfigUtils.attributesForRepeatable(
				sourceClass.getMetadata(), ComponentScans.class, ComponentScan.class);
		if (!componentScans.isEmpty() &&
				!this.conditionEvaluator.shouldSkip(sourceClass.getMetadata(), ConfigurationPhase.REGISTER_BEAN)) {
			for (AnnotationAttributes componentScan : componentScans) {
				// The config class is annotated with @ComponentScan -> perform the scan immediately
				Set<BeanDefinitionHolder> scannedBeanDefinitions =
						this.componentScanParser.parse(componentScan, sourceClass.getMetadata().getClassName());
				// Check the set of scanned definitions for any further config classes and parse recursively if needed
				for (BeanDefinitionHolder holder : scannedBeanDefinitions) {
					BeanDefinition bdCand = holder.getBeanDefinition().getOriginatingBeanDefinition();
					if (bdCand == null) {
						bdCand = holder.getBeanDefinition();
					}
					if (ConfigurationClassUtils.checkConfigurationClassCandidate(bdCand, this.metadataReaderFactory)) {
						parse(bdCand.getBeanClassName(), holder.getBeanName());
					}
				}
			}
		}		
```

# `@Import`

`@Import` 의 Arguement 로 `@Configuration` Class 를 넘기면 그 `@Configuration`
Class 의 Instance 를 Bean 으로 등록한다. 즉, `@Configuration` Class 의 `@Bean` Method 가
return 하는 Object 를 Bean 으로 등록한다.

```java
// org.springframework.context.annotation.ConfigurationClassParser
class ConfigurationClassParser {
...
	@Nullable
	protected final SourceClass doProcessConfigurationClass(
			ConfigurationClass configClass, SourceClass sourceClass, Predicate<String> filter)
			throws IOException {
	...
		// Process any @Import annotations
		processImports(configClass, sourceClass, getImports(sourceClass), filter, true);

```

# `@ImportResource`

```java
// org.springframework.context.annotation.ConfigurationClassParser
class ConfigurationClassParser {
...
	@Nullable
	protected final SourceClass doProcessConfigurationClass(
			ConfigurationClass configClass, SourceClass sourceClass, Predicate<String> filter)
			throws IOException {
	...
		// Process any @ImportResource annotations
		AnnotationAttributes importResource =
				AnnotationConfigUtils.attributesFor(sourceClass.getMetadata(), ImportResource.class);
		if (importResource != null) {
			String[] resources = importResource.getStringArray("locations");
			Class<? extends BeanDefinitionReader> readerClass = importResource.getClass("reader");
			for (String resource : resources) {
				String resolvedResource = this.environment.resolveRequiredPlaceholders(resource);
				configClass.addImportedResource(resolvedResource, readerClass);
			}
		}
```

# `@Bean`

`@Bean` Method 가 return 하는 Object 는 Bean 으로 등록된다.

```java
// org.springframework.context.annotation.ConfigurationClassParser
class ConfigurationClassParser {
...
	@Nullable
	protected final SourceClass doProcessConfigurationClass(
			ConfigurationClass configClass, SourceClass sourceClass, Predicate<String> filter)
			throws IOException {
	...
		// Process individual @Bean methods
		Set<MethodMetadata> beanMethods = retrieveBeanMethodMetadata(sourceClass);
		for (MethodMetadata methodMetadata : beanMethods) {
			configClass.addBeanMethod(new BeanMethod(methodMetadata, configClass));
		}
```

# `@EnableAutoConfiguration`

`FACTORIES_RESOURCE_LOCATION` 에 `spring.factories` 파일 경로가 hard coding 되어 있다.

```java
// org.springframework.core.io.support.SpringFactoriesLoader
public final class SpringFactoriesLoader {

	/**
	 * The location to look for factories.
	 * <p>Can be present in multiple JAR files.
	 */
	public static final String FACTORIES_RESOURCE_LOCATION = "META-INF/spring.factories";

...
}
```

`loadFactoryNames` 에서 classpath 에 포함된 `spring.factories` 파일들을 로딩한다.
 
```java
// org.springframework.core.io.support.SpringFactoriesLoader
public final class SpringFactoriesLoader {
...	
	public static List<String> loadFactoryNames(Class<?> factoryType, @Nullable ClassLoader classLoader) {
		ClassLoader classLoaderToUse = classLoader;
		if (classLoaderToUse == null) {
			classLoaderToUse = SpringFactoriesLoader.class.getClassLoader();
		}
		String factoryTypeName = factoryType.getName();
		return loadSpringFactories(classLoaderToUse).getOrDefault(factoryTypeName, Collections.emptyList());
	}

	private static Map<String, List<String>> loadSpringFactories(ClassLoader classLoader) {
		Map<String, List<String>> result = cache.get(classLoader);
		if (result != null) {
			return result;
		}

		result = new HashMap<>();
		try {
			Enumeration<URL> urls = classLoader.getResources(FACTORIES_RESOURCE_LOCATION);
```

# `@Repository`

# `@Transactional`

# `@Service`

# `@Component`

# `@Controller`

# `@RestController`

# `@Validation`
