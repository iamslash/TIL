- [Reading `application.yml` Flow](#reading-applicationyml-flow)
- [How Event Handler Works](#how-event-handler-works)
- [How ApplicationRunner Works](#how-applicationrunner-works)
- [Processing Bean](#processing-bean)
- [Getting Bean Instance](#getting-bean-instance)
- [Registering Bean Definition](#registering-bean-definition)
- [Creating Bean Instance](#creating-bean-instance)
- [Instanciating SingleTon Bean Instances](#instanciating-singleton-bean-instances)
- [Reading `spring.factories`](#reading-springfactories)

----

# Reading `application.yml` Flow

```java
// org.springframework.context.ConfigurableApplicationContext
public interface ConfigurableApplicationContext extends ApplicationContext, Lifecycle, Closeable {
...
	public ConfigurableApplicationContext run(String... args) {
		StopWatch stopWatch = new StopWatch();
		stopWatch.start();
		ConfigurableApplicationContext context = null;
		Collection<SpringBootExceptionReporter> exceptionReporters = new ArrayList<>();
		configureHeadlessProperty();
		SpringApplicationRunListeners listeners = getRunListeners(args);
		listeners.starting();
		try {
			ApplicationArguments applicationArguments = new DefaultApplicationArguments(args);
			ConfigurableEnvironment environment = prepareEnvironment(listeners, applicationArguments);

// org.springframework.boot.context.config.ConfigFileApplicationListener
public class ConfigFileApplicationListener implements EnvironmentPostProcessor, SmartApplicationListener, Ordered {
...
	@Override
	public void onApplicationEvent(ApplicationEvent event) {
		if (event instanceof ApplicationEnvironmentPreparedEvent) {
			onApplicationEnvironmentPreparedEvent((ApplicationEnvironmentPreparedEvent) event);
		}
		if (event instanceof ApplicationPreparedEvent) {
			onApplicationPreparedEvent(event);
		}
	}

// org.springframework.boot.context.config.ConfigFileApplicationListener
public class ConfigFileApplicationListener implements EnvironmentPostProcessor, SmartApplicationListener, Ordered {
...
		void load() {
			FilteredPropertySource.apply(this.environment, DEFAULT_PROPERTIES, LOAD_FILTERED_PROPERTY,
					(defaultProperties) -> {
						this.profiles = new LinkedList<>();
						this.processedProfiles = new LinkedList<>();
						this.activatedProfiles = false;
						this.loaded = new LinkedHashMap<>();
						initializeProfiles();
						while (!this.profiles.isEmpty()) {
							Profile profile = this.profiles.poll();
							if (isDefaultProfile(profile)) {
								addProfileToEnvironment(profile.getName());
							}
							load(profile, this::getPositiveProfileFilter,
									addToLoaded(MutablePropertySources::addLast, false));
							this.processedProfiles.add(profile);
						}
						load(null, this::getNegativeProfileFilter, addToLoaded(MutablePropertySources::addFirst, true));
						addLoadedPropertySources();
						applyActiveProfiles(defaultProperties);
					});
		}
// org.springframework.boot.context.config.ConfigFileApplicationListener
public class ConfigFileApplicationListener implements EnvironmentPostProcessor, SmartApplicationListener, Ordered {
...
		private void load(String location, String name, Profile profile, DocumentFilterFactory filterFactory,
				DocumentConsumer consumer) {
			if (!StringUtils.hasText(name)) {
				for (PropertySourceLoader loader : this.propertySourceLoaders) {
					if (canLoadFileExtension(loader, location)) {
						load(loader, location, profile, filterFactory.getDocumentFilter(profile), consumer);
						return;
					}
				}
				throw new IllegalStateException("File extension of config file location '" + location
						+ "' is not known to any PropertySourceLoader. If the location is meant to reference "
						+ "a directory, it must end in '/'");
			}
			Set<String> processed = new HashSet<>();
			for (PropertySourceLoader loader : this.propertySourceLoaders) {
				for (String fileExtension : loader.getFileExtensions()) {
					if (processed.add(fileExtension)) {
						loadForFileExtension(loader, location + name, "." + fileExtension, profile, filterFactory,
								consumer);
					}
				}
			}
		}

// org.springframework.boot.context.config.ConfigFileApplicationListener
public class ConfigFileApplicationListener implements EnvironmentPostProcessor, SmartApplicationListener, Ordered {

	private static final String DEFAULT_PROPERTIES = "defaultProperties";

	// Note the order is from least to most specific (last one wins)
	private static final String DEFAULT_SEARCH_LOCATIONS = "classpath:/,classpath:/config/,file:./,file:./config/";

	private static final String DEFAULT_NAMES = "application";
...
	private void loadForFileExtension(PropertySourceLoader loader, String prefix, String fileExtension,
			Profile profile, DocumentFilterFactory filterFactory, DocumentConsumer consumer) {
		DocumentFilter defaultFilter = filterFactory.getDocumentFilter(null);
		DocumentFilter profileFilter = filterFactory.getDocumentFilter(profile);
		if (profile != null) {
			// Try profile-specific file & profile section in profile file (gh-340)
			String profileSpecificFile = prefix + "-" + profile + fileExtension;
			load(loader, profileSpecificFile, profile, defaultFilter, consumer);
			load(loader, profileSpecificFile, profile, profileFilter, consumer);
			// Try profile specific sections in files we've already processed
			for (Profile processedProfile : this.processedProfiles) {
				if (processedProfile != null) {
					String previouslyLoaded = prefix + "-" + processedProfile + fileExtension;
					load(loader, previouslyLoaded, profile, profileFilter, consumer);
				}
			}
		}
		// Also try the profile-specific section (if any) of the normal file
		load(loader, prefix + fileExtension, profile, profileFilter, consumer);
	}

// org.springframework.boot.context.config.ConfigFileApplicationListener
public class ConfigFileApplicationListener implements EnvironmentPostProcessor, SmartApplicationListener, Ordered {
...
		private void load(PropertySourceLoader loader, String location, Profile profile, DocumentFilter filter,
				DocumentConsumer consumer) {
			try {
				Resource resource = this.resourceLoader.getResource(location);
				if (resource == null || !resource.exists()) {
					if (this.logger.isTraceEnabled()) {
						StringBuilder description = getDescription("Skipped missing config ", location, resource,
								profile);
						this.logger.trace(description);
					}
					return;
				}
				if (!StringUtils.hasText(StringUtils.getFilenameExtension(resource.getFilename()))) {
					if (this.logger.isTraceEnabled()) {
						StringBuilder description = getDescription("Skipped empty config extension ", location,
								resource, profile);
						this.logger.trace(description);
					}
					return;
				}
				String name = "applicationConfig: [" + location + "]";
				List<Document> documents = loadDocuments(loader, name, resource);
				if (CollectionUtils.isEmpty(documents)) {
					if (this.logger.isTraceEnabled()) {
						StringBuilder description = getDescription("Skipped unloaded config ", location, resource,
								profile);
						this.logger.trace(description);
					}
					return;
				}
				List<Document> loaded = new ArrayList<>();
				for (Document document : documents) {
					if (filter.match(document)) {
						addActiveProfiles(document.getActiveProfiles());
						addIncludedProfiles(document.getIncludeProfiles());
						loaded.add(document);
					}
				}
				Collections.reverse(loaded);
				if (!loaded.isEmpty()) {
					loaded.forEach((document) -> consumer.accept(profile, document));
					if (this.logger.isDebugEnabled()) {
						StringBuilder description = getDescription("Loaded config file ", location, resource, profile);
						this.logger.debug(description);
					}
				}
			}
			catch (Exception ex) {
				throw new IllegalStateException("Failed to load property source from location '" + location + "'", ex);
			}
		}

```

# How Event Handler Works

```java
// org.springframework.boot.SpringApplication
public class SpringApplication {
...
	public ConfigurableApplicationContext run(String... args) {
...
		try {
			listeners.running(context);

// org.springframework.boot.context.event.EventPublishingRunListener
public class EventPublishingRunListener implements SpringApplicationRunListener, Ordered {
...
	@Override
	public void running(ConfigurableApplicationContext context) {
		context.publishEvent(new ApplicationReadyEvent(this.application, this.args, context));
	}

// org.springframework.context.event.ApplicationListenerMethodAdapter
public class ApplicationListenerMethodAdapter implements GenericApplicationListener {
...
	@Override
	public void onApplicationEvent(ApplicationEvent event) {
		processEvent(event);
	}
```

# How ApplicationRunner Works

```java
// org.springframework.boot.SpringApplication
public class SpringApplication {
...
	public ConfigurableApplicationContext run(String... args) {
    ...
			listeners.started(context);
			callRunners(context, applicationArguments);

// org.springframework.boot.SpringApplication
public class SpringApplication {
...
	private void callRunners(ApplicationContext context, ApplicationArguments args) {
		List<Object> runners = new ArrayList<>();
		runners.addAll(context.getBeansOfType(ApplicationRunner.class).values());
		runners.addAll(context.getBeansOfType(CommandLineRunner.class).values());
		AnnotationAwareOrderComparator.sort(runners);
		for (Object runner : new LinkedHashSet<>(runners)) {
			if (runner instanceof ApplicationRunner) {
				callRunner((ApplicationRunner) runner, args);
			}
			if (runner instanceof CommandLineRunner) {
				callRunner((CommandLineRunner) runner, args);
			}
		}
	}
```

# Processing Bean

* [Spring IoC Container를 까보자 #Bean 등록은 어떻게 될까?](https://blog.woniper.net/336?category=699184)

-----

* Register Bean Definition to `Map<String, BeanDefinition>`.
* `getBean` instantiate Bean with `BeanDefinition` and save to `Map<String, Object>` for cache.
* Context Object has Beans.

Bean Processing 을 이해하기 위해서는 다음의 Class Diagram 을 이해해야 한다. 자꾸
보면 외워진다.

![](img/classdiagrambeans.png)

```plantuml
@startuml
class BeanDefinitionHolder
BeanDefinitionHolder "1" *-- "1" BeanDefinition
DefaultListableBeanFactory .. BeanDefinitionHolder : create

interface AliasRegistry {}
interface BeanDefinitionRegistry implements AliasRegistry {}

interface BeanDefinitionRegistry {}

interface BeanFactory {}

interface ListableBeanFactory extends BeanFactory {}

interface ApplicationContext extends ListableBeanFactory {}

interface BeanDefinition 

ApplicationContext "1" --* "1" AbstractApplicationContext
class AbstractApplicationContext extends DefaultResourceLoader

BeanDefinitionRegistry "1" --* "1" AnnotatedBeanDefinitionReader
AnnotatedBeanDefinitionReader "1" --* "1" AnnotationConfigApplicationContext

class SimpleAliasRegistry implements AliasRegistry
interface SingletonBeanRegistry

class DefaultSingletonBeanRegistry extends SimpleAliasRegistry implements SingletonBeanRegistry

abstract class FactoryBeanRegistrySupport extends DefaultSingletonBeanRegistry

abstract class AbstractBeanFactory extends FactoryBeanRegistrySupport implements ConfigurableBeanFactory

abstract class AbstractAutowireCapableBeanFactory extends AbstractBeanFactory

class DefaultListableBeanFactory extends AbstractAutowireCapableBeanFactory implements BeanDefinitionRegistry
DefaultListableBeanFactory "1" --* "1" GenericApplicationContext

class GenericApplicationContext extends AbstractApplicationContext implements BeanDefinitionRegistry {}

' AnnotationConfigApplicationContext
interface AnnotationConfigRegistry {}

class AnnotationConfigApplicationContext extends GenericApplicationContext implements AnnotationConfigRegistry {}

@enduml
```

# Getting Bean Instance

`ListableBeanFactory::getBeansWithAnnotation()` 으로 특정 Bean 을 읽어올 수 있다.

[Getting Bean Instance](SpringAnnotationsCodeTour.md#getting-bean-instance) 참고.

# Registering Bean Definition

* `DefaultListableBeanFactory.registerBeanDefinition` register bean-name,
  bean-definition to `Map<String, BeanDefinition>`.

[Registering Bean Flow](SpringAnnotationsCodeTour.md#registering-bean-flow)
참고.

* [A process of Spring Bean from scratch](https://programming.vip/docs/a-process-of-spring-bean-from-scratch.html)

![](img/registeringbean.png)

# Creating Bean Instance

* `AnnotationConfigApplicationContext::getBean()` starts to instantiate Bean.
* `AbstractBeanFactory.doGetBean` instantiate Bean.
  
[Creating Bean Intstance](SpringAnnotationsCodeTour.md#creating-bean-instance) 참고.

# Instanciating SingleTon Bean Instances

`ConfigurableListableBeanFactory::preInstantiateSingletons()` 에서 SingleTon Bean Instances 를 생성한다.

```java
// org.springframework.beans.factory.config.ConfigurableListableBeanFactory
public interface ConfigurableListableBeanFactory
		extends ListableBeanFactory, AutowireCapableBeanFactory, ConfigurableBeanFactory {
...
	@Override
	public void preInstantiateSingletons() throws BeansException {
```

다음은 SpringApplication 이
`ConfigurableListableBeanFactory::preInstantiateSingletons()` 를 호출하기 까지
흐름을 추적한 것이다. `AbstractBeanFactory::getBean()` 를 호출하면 `DefaultSingletonBeanRegistry::singletonObjects`
에 없으면 Bean Instance 를 생성한다. [Creating Bean Instance](SpringAnnotationsCodeTour.md#creating-bean-instance) 참고.

```java
// com.iamslash.exannotation.ExAnnotationApp
@SpringBootApplication
public class ExAnnotationApp {

    public static void main(String[] args) {
        new SpringApplicationBuilder()
            .sources(ExAnnotationApp.class)
            .run(args);

// org.springframework.boot.builder.SpringApplicationBuilder
public class SpringApplicationBuilder {
...
	public ConfigurableApplicationContext run(String... args) {
		if (this.running.get()) {
			// If already created we just return the existing context
			return this.context;
		}
		configureAsChildIfNecessary(args);
		if (this.running.compareAndSet(false, true)) {
			synchronized (this.running) {
				// If not already running copy the sources over and then run.
				this.context = build().run(args);

// org.springframework.boot.SpringApplication
public class SpringApplication {
...
	public ConfigurableApplicationContext run(String... args) {
		StopWatch stopWatch = new StopWatch();
		stopWatch.start();
		ConfigurableApplicationContext context = null;
		Collection<SpringBootExceptionReporter> exceptionReporters = new ArrayList<>();
		configureHeadlessProperty();
		SpringApplicationRunListeners listeners = getRunListeners(args);
		listeners.starting();
		try {
			ApplicationArguments applicationArguments = new DefaultApplicationArguments(args);
			ConfigurableEnvironment environment = prepareEnvironment(listeners, applicationArguments);
			configureIgnoreBeanInfo(environment);
			Banner printedBanner = printBanner(environment);
			context = createApplicationContext();
			exceptionReporters = getSpringFactoriesInstances(SpringBootExceptionReporter.class,
					new Class[] { ConfigurableApplicationContext.class }, context);
			prepareContext(context, environment, listeners, applicationArguments, printedBanner);
			refreshContext(context);

// org.springframework.boot.SpringApplication
public class SpringApplication {
...
	private void refreshContext(ConfigurableApplicationContext context) {
		refresh(context);

// org.springframework.boot.SpringApplication
public class SpringApplication {
...
	protected void refresh(ApplicationContext applicationContext) {
		Assert.isInstanceOf(AbstractApplicationContext.class, applicationContext);
		((AbstractApplicationContext) applicationContext).refresh();

// org.springframework.boot.web.servlet.context.ServletWebServerApplicationContext
public class ServletWebServerApplicationContext extends GenericWebApplicationContext
		implements ConfigurableWebServerApplicationContext {
...
	@Override
	public final void refresh() throws BeansException, IllegalStateException {
		try {
			super.refresh();

// org.springframework.context.support.AbstractApplicationContext
public abstract class AbstractApplicationContext extends DefaultResourceLoader
		implements ConfigurableApplicationContext {
...
	@Override
	public void refresh() throws BeansException, IllegalStateException {
		synchronized (this.startupShutdownMonitor) {
			// Prepare this context for refreshing.
			prepareRefresh();

			// Tell the subclass to refresh the internal bean factory.
			ConfigurableListableBeanFactory beanFactory = obtainFreshBeanFactory();

			// Prepare the bean factory for use in this context.
			prepareBeanFactory(beanFactory);

			try {
				// Allows post-processing of the bean factory in context subclasses.
				postProcessBeanFactory(beanFactory);

				// Invoke factory processors registered as beans in the context.
				invokeBeanFactoryPostProcessors(beanFactory);

				// Register bean processors that intercept bean creation.
				registerBeanPostProcessors(beanFactory);

				// Initialize message source for this context.
				initMessageSource();

				// Initialize event multicaster for this context.
				initApplicationEventMulticaster();

				// Initialize other special beans in specific context subclasses.
				onRefresh();

				// Check for listener beans and register them.
				registerListeners();

				// Instantiate all remaining (non-lazy-init) singletons.
				finishBeanFactoryInitialization(beanFactory);

// org.springframework.context.support.AbstractApplicationContext
public abstract class AbstractApplicationContext extends DefaultResourceLoader
		implements ConfigurableApplicationContext {
...
	protected void finishBeanFactoryInitialization(ConfigurableListableBeanFactory beanFactory) {
		// Initialize conversion service for this context.
		if (beanFactory.containsBean(CONVERSION_SERVICE_BEAN_NAME) &&
				beanFactory.isTypeMatch(CONVERSION_SERVICE_BEAN_NAME, ConversionService.class)) {
			beanFactory.setConversionService(
					beanFactory.getBean(CONVERSION_SERVICE_BEAN_NAME, ConversionService.class));
		}

		// Register a default embedded value resolver if no bean post-processor
		// (such as a PropertyPlaceholderConfigurer bean) registered any before:
		// at this point, primarily for resolution in annotation attribute values.
		if (!beanFactory.hasEmbeddedValueResolver()) {
			beanFactory.addEmbeddedValueResolver(strVal -> getEnvironment().resolvePlaceholders(strVal));
		}

		// Initialize LoadTimeWeaverAware beans early to allow for registering their transformers early.
		String[] weaverAwareNames = beanFactory.getBeanNamesForType(LoadTimeWeaverAware.class, false, false);
		for (String weaverAwareName : weaverAwareNames) {
			getBean(weaverAwareName);
		}

		// Stop using the temporary ClassLoader for type matching.
		beanFactory.setTempClassLoader(null);

		// Allow for caching all bean definition metadata, not expecting further changes.
		beanFactory.freezeConfiguration();

		// Instantiate all remaining (non-lazy-init) singletons.
		beanFactory.preInstantiateSingletons();

// org.springframework.beans.factory.config.ConfigurableListableBeanFactory
public interface ConfigurableListableBeanFactory
		extends ListableBeanFactory, AutowireCapableBeanFactory, ConfigurableBeanFactory {
...
	@Override
	public void preInstantiateSingletons() throws BeansException {
		if (logger.isTraceEnabled()) {
			logger.trace("Pre-instantiating singletons in " + this);
		}

		// Iterate over a copy to allow for init methods which in turn register new bean definitions.
		// While this may not be part of the regular factory bootstrap, it does otherwise work fine.
		List<String> beanNames = new ArrayList<>(this.beanDefinitionNames);

		// Trigger initialization of all non-lazy singleton beans...
		for (String beanName : beanNames) {
			RootBeanDefinition bd = getMergedLocalBeanDefinition(beanName);
			if (!bd.isAbstract() && bd.isSingleton() && !bd.isLazyInit()) {
				if (isFactoryBean(beanName)) {
					Object bean = getBean(FACTORY_BEAN_PREFIX + beanName);
					if (bean instanceof FactoryBean) {
						final FactoryBean<?> factory = (FactoryBean<?>) bean;
						boolean isEagerInit;
						if (System.getSecurityManager() != null && factory instanceof SmartFactoryBean) {
							isEagerInit = AccessController.doPrivileged((PrivilegedAction<Boolean>)
											((SmartFactoryBean<?>) factory)::isEagerInit,
									getAccessControlContext());
						}
						else {
							isEagerInit = (factory instanceof SmartFactoryBean &&
									((SmartFactoryBean<?>) factory).isEagerInit());
						}
						if (isEagerInit) {
							getBean(beanName);
						}
					}
				}
				else {
					getBean(beanName);		
```

# Reading `spring.factories`

`spring.factories` 에 Interface 혹은 Abstract Class 를 key 로 Implementation Class 를 comma-separated value 로 넣어두면 Spring Application 이 시작할 때 필요한 Implementation Class 의 Instances 를 생성할 수 있다. 이것은 `SpringFactoriesLoader` Class 에 의해 이루어 진다. `factories` 라는 단어를 사용한 것을 보니 Bean 생성 로직과는 무관하게 Class Instance 를 생성하는 mechanism 같다.

`SpringFactoriesLoader` 는 `META-INF/spring.factories` 를 읽고 Instance 를 생성하는 Class 이다. 내가 발견한 `spring.factories` 는 다음과 같다.

```
spring-beans-5.2.5.RELEASE.jar!/META-INF/spring.factories
spring-boot-2.2.6.RELEASE.jar!/META-INF/spring.factories
spring-boot-autoconfigure-2.2.6.RELEASE.jar!/META-INF/spring.factories
```

`spring.factores` 의 형식은 Properties format 을 따른다. 즉, key 와 comma-separated values 로 구성된다. key 는
Interface 혹은 Abstract Class 이름이다. value 는 key 를 구현한 Implementation Class 의 이름이다.

```conf
# Auto Configure
org.springframework.boot.autoconfigure.EnableAutoConfiguration=\
org.springframework.boot.autoconfigure.admin.SpringApplicationAdminJmxAutoConfiguration,\
org.springframework.boot.autoconfigure.aop.AopAutoConfiguration,\
org.springframework.boot.autoconfigure.amqp.RabbitAutoConfiguration,\
org.springframework.boot.autoconfigure.batch.BatchAutoConfiguration,\
org.springframework.boot.autoconfigure.cache.CacheAutoConfiguration,\
...
```

`Map<String, List<String>> SpringFactoriesLoader::loadSpringFactories(@Nullable ClassLoader classLoader)` 는 모든 `spring.factories` 파일들을 읽고 그것을 Map 에 저장하여 리턴한다. 이 Map 은 caching 된다. 즉, 두번째 호출부터 이미 생성한 Map 을 바로 리턴한다. 또한 `SpringFactoriesLoader::FACTORIES_RESOURCE_LOCATION` 에 spring factory file (`spring.factories`)의 경로가 hard coding 되어 있다.

```java
// org.springframework.core.io.support.SpringFactoriesLoader
public final class SpringFactoriesLoader {
...
	public static final String FACTORIES_RESOURCE_LOCATION = "META-INF/spring.factories";

// org.springframework.core.io.support.SpringFactoriesLoader
	private static Map<String, List<String>> loadSpringFactories(@Nullable ClassLoader classLoader) {
		MultiValueMap<String, String> result = cache.get(classLoader);
		if (result != null) {
			return result;
		}

		try {
			Enumeration<URL> urls = (classLoader != null ?
					classLoader.getResources(FACTORIES_RESOURCE_LOCATION) :
					ClassLoader.getSystemResources(FACTORIES_RESOURCE_LOCATION));
			result = new LinkedMultiValueMap<>();
			while (urls.hasMoreElements()) {
				URL url = urls.nextElement();
				UrlResource resource = new UrlResource(url);
				Properties properties = PropertiesLoaderUtils.loadProperties(resource);
				for (Map.Entry<?, ?> entry : properties.entrySet()) {
					String factoryTypeName = ((String) entry.getKey()).trim();
					for (String factoryImplementationName : StringUtils.commaDelimitedListToStringArray((String) entry.getValue())) {
						result.add(factoryTypeName, factoryImplementationName.trim());
					}
				}
			}
			cache.put(classLoader, result);
			return result;
		}
		catch (IOException ex) {
			throw new IllegalArgumentException("Unable to load factories from location [" +
					FACTORIES_RESOURCE_LOCATION + "]", ex);
		}
	}
```

`<T> T SpringFactoriesLoader::instantiateFactory(String factoryImplementationName, Class<T> factoryType, ClassLoader classLoader)` 는 Interface 혹은 Abstract Class 가 factoryType 이고 Implementation Class 가 factoryImplementationName 인 것을 생성하여 리턴한다. 즉, factoryImplementationName 이름을 갖는 Class 의 Instance 를 생성하여 리턴한다. 

```java
// org.springframework.core.io.support.SpringFactoriesLoader
public final class SpringFactoriesLoader {
...
	private static <T> T instantiateFactory(String factoryImplementationName, Class<T> factoryType, ClassLoader classLoader) {
		try {
			Class<?> factoryImplementationClass = ClassUtils.forName(factoryImplementationName, classLoader);
			if (!factoryType.isAssignableFrom(factoryImplementationClass)) {
				throw new IllegalArgumentException(
						"Class [" + factoryImplementationName + "] is not assignable to factory type [" + factoryType.getName() + "]");
			}
			return (T) ReflectionUtils.accessibleConstructor(factoryImplementationClass).newInstance();
		}
		catch (Throwable ex) {
			throw new IllegalArgumentException(
				"Unable to instantiate factory class [" + factoryImplementationName + "] for factory type [" + factoryType.getName() + "]",
				ex);
		}
	}
```

`List<String> SpringFactoriesLoader::loadFactoryNames(Class<?> factoryType, @Nullable ClassLoader classLoader)` 는 Interface 혹은 Abstract Class 가
factoryType 인 것의 Implementation Class Names 을 리턴한다. 즉, `spring.factories` 를 읽고
`factoryType` 을 key 로 하는 value 들을 리턴한다.

```java
// org.springframework.core.io.support.SpringFactoriesLoader
public final class SpringFactoriesLoader {
...
	public static List<String> loadFactoryNames(Class<?> factoryType, @Nullable ClassLoader classLoader) {
		String factoryTypeName = factoryType.getName();
		return loadSpringFactories(classLoader).getOrDefault(factoryTypeName, Collections.emptyList());
	}
```

`List<T> SpringFactoriesLoader::loadFactories(Class<T> factoryType, @Nullable ClassLoader classLoader)` 는 Interface 혹은 Abstract Class 가 factoryType 인 Implementation Class 의 Instances 를 생성하여 리턴한다.

```java
// org.springframework.core.io.support.SpringFactoriesLoader
public final class SpringFactoriesLoader {
...
	public static <T> List<T> loadFactories(Class<T> factoryType, @Nullable ClassLoader classLoader) {
		Assert.notNull(factoryType, "'factoryType' must not be null");
		ClassLoader classLoaderToUse = classLoader;
		if (classLoaderToUse == null) {
			classLoaderToUse = SpringFactoriesLoader.class.getClassLoader();
		}
		List<String> factoryImplementationNames = loadFactoryNames(factoryType, classLoaderToUse);
		if (logger.isTraceEnabled()) {
			logger.trace("Loaded [" + factoryType.getName() + "] names: " + factoryImplementationNames);
		}
		List<T> result = new ArrayList<>(factoryImplementationNames.size());
		for (String factoryImplementationName : factoryImplementationNames) {
			result.add(instantiateFactory(factoryImplementationName, factoryType, classLoaderToUse));
		}
		AnnotationAwareOrderComparator.sort(result);
		return result;
	}
```

Spring Application 은 다음과 같은 부분에서 `spring.factories` 의 Implementation Class 의 Instance 를 생성하고 있다. 이때 Implementation Class 의 Instance 를 생성하는 logic 은 `SpringFactoriesLoader` 를 이용하지 않는다는 것을 유의하자.

```java
// org.springframework.boot.SpringApplication
public class SpringApplication {
...
		setInitializers((Collection) getSpringFactoriesInstances(ApplicationContextInitializer.class));
		setListeners((Collection) getSpringFactoriesInstances(ApplicationListener.class));

...
			exceptionReporters = getSpringFactoriesInstances(SpringBootExceptionReporter.class,
					new Class[] { ConfigurableApplicationContext.class }, context);
...
		return new SpringApplicationRunListeners(logger,
				getSpringFactoriesInstances(SpringApplicationRunListener.class, types, this, args));

// org.springframework.boot.SpringApplication
public class SpringApplication {
...
	private <T> Collection<T> getSpringFactoriesInstances(Class<T> type) {
		return getSpringFactoriesInstances(type, new Class<?>[] {});
	}

// org.springframework.boot.SpringApplication
public class SpringApplication {
...
	private <T> Collection<T> getSpringFactoriesInstances(Class<T> type, Class<?>[] parameterTypes, Object... args) {
		ClassLoader classLoader = getClassLoader();
		// Use names and ensure unique to protect against duplicates
		Set<String> names = new LinkedHashSet<>(SpringFactoriesLoader.loadFactoryNames(type, classLoader));
		List<T> instances = createSpringFactoriesInstances(type, parameterTypes, classLoader, args, names);

// org.springframework.boot.SpringApplication
public class SpringApplication {
...
	private <T> List<T> createSpringFactoriesInstances(Class<T> type, Class<?>[] parameterTypes,
			ClassLoader classLoader, Object[] args, Set<String> names) {
		List<T> instances = new ArrayList<>(names.size());
		for (String name : names) {
			try {
				Class<?> instanceClass = ClassUtils.forName(name, classLoader);
				Assert.isAssignable(type, instanceClass);
				Constructor<?> constructor = instanceClass.getDeclaredConstructor(parameterTypes);
				T instance = (T) BeanUtils.instantiateClass(constructor, args);
				instances.add(instance);
			}
			catch (Throwable ex) {
				throw new IllegalArgumentException("Cannot instantiate " + type + " : " + name, ex);
			}
		}
		return instances;
	}
```
