- [Annotation Under The Hood](#annotation-under-the-hood)
- [Reading Merged Annotions Flow](#reading-merged-annotions-flow)
- [Registering Bean Definition Flow](#registering-bean-definition-flow)
- [Creating Bean Instance](#creating-bean-instance)
- [Getting Bean Instance](#getting-bean-instance)
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
* Class 에 부착된 Annotation 을 읽어와서 특정 Annotation 과 약속한 일을
  수행한다. 
  * `@ComponentScan` 를 사용하면 그 Annotation 을 읽어와서 attributes 를 얻어온다. 그리고 component scanning 대상을 확인한다. [@ComponentScan](#componentscan) 참고.
* 특정 Class 의 Instance 를 Bean 으로 등록할 때 Proxy Class 의 Instance 를 만들어서
  그것을 Bean 으로 등록한다. (AOP)  
  * `@Repository` 을 사용하면 그것이 부착된 Class 를 Proxy Class 로 감싸서 AOP 를 구현한다. [@Repository](#repository) 참고.
 
아래의 예는 `@Foo` Class 를 Compile Time 에 조작하는 것이다. [JAVA Annotation Processing](/java/java_annotation.md#annotation-processing) 참고.

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

# Reading Merged Annotions Flow

기본적으로 Class 의 Annotation 을 `getAnnotations()` 로 읽어오면 그 Class 에
부착된 Annotation 만 읽어온다. 부착된 Annotation 에 다시 부착된 Annotation 을
읽어오려면 Annotation Graph 를 만들어서 읽어야 한다. [Java Annotations Merged
Annotations](/java/java_annotation.md#merged-annotations) 참고.

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

`MergedAnnotation::from()` 을 이용하면 특정 Class 에 특정 Annotation 이 있는지 검색할 수 있다. 아래의 예는 
`BeanDefinitionLoader::isComponent()` 의 구현이다. 특정 Class 에 `@Component` 가 부착됬는지 확인한다.

```java
// org.springframework.boot.BeanDefinitionLoader
class BeanDefinitionLoader {
...
	private boolean isComponent(Class<?> type) {
		// This has to be a bit of a guess. The only way to be sure that this type is
		// eligible is to make a bean definition out of it and try to instantiate it.
		if (MergedAnnotations.from(type, SearchStrategy.TYPE_HIERARCHY).isPresent(Component.class)) {

// org.springframework.core.annotation.MergedAnnotations
public interface MergedAnnotations extends Iterable<MergedAnnotation<Annotation>> {
...
	static MergedAnnotations from(AnnotatedElement element, SearchStrategy searchStrategy) {
		return from(element, searchStrategy, RepeatableContainers.standardRepeatables());

// org.springframework.core.annotation.MergedAnnotations
public interface MergedAnnotations extends Iterable<MergedAnnotation<Annotation>> {
...
	static MergedAnnotations from(AnnotatedElement element, SearchStrategy searchStrategy,
			RepeatableContainers repeatableContainers) {

		return TypeMappedAnnotations.from(element, searchStrategy, repeatableContainers, AnnotationFilter.PLAIN);

// org.springframework.core.annotation.TypeMappedAnnotations
final class TypeMappedAnnotations implements MergedAnnotations {
...	
	static MergedAnnotations from(AnnotatedElement element, SearchStrategy searchStrategy,
			RepeatableContainers repeatableContainers, AnnotationFilter annotationFilter) {

		if (AnnotationsScanner.isKnownEmpty(element, searchStrategy)) {
			return NONE;
		}
		return new TypeMappedAnnotations(element, searchStrategy, repeatableContainers, annotationFilter);

// org.springframework.core.annotation.TypeMappedAnnotations
final class TypeMappedAnnotations implements MergedAnnotations {
...	
	private TypeMappedAnnotations(AnnotatedElement element, SearchStrategy searchStrategy,
			RepeatableContainers repeatableContainers, AnnotationFilter annotationFilter) {

		this.source = element;
		this.element = element;
		this.searchStrategy = searchStrategy;
		this.annotations = null;
		this.repeatableContainers = repeatableContainers;
		this.annotationFilter = annotationFilter;
	}

// org.springframework.core.annotation.TypeMappedAnnotations
final class TypeMappedAnnotations implements MergedAnnotations {
...
	@Override
	public <A extends Annotation> boolean isPresent(Class<A> annotationType) {
		if (this.annotationFilter.matches(annotationType)) {
			return false;
		}
		return Boolean.TRUE.equals(scan(annotationType,
				IsPresent.get(this.repeatableContainers, this.annotationFilter, false)));
	}
```

# Registering Bean Definition Flow

`BeanDefinitionLoader` Class 는 `AnnotatedBeanDefinitionReader annotatedReader`
와 `XmlBeanDefinitionReader xmlReader` 를 갖는다. Annotation 혹은 XML 로 부터
Bean Definition 을 읽어들일 수 있다.

```java
// org.springframework.boot.BeanDefinitionLoader
class BeanDefinitionLoader {

	private final Object[] sources;

	private final AnnotatedBeanDefinitionReader annotatedReader;

	private final XmlBeanDefinitionReader xmlReader;
```

`DefaultListableBeanFactory::registerBeanDefinition(String beanName, BeanDefinition beanDefinition)` 를 호출하여 `DefaultListableBeanFactory::beanDefinitionMap` 에 Bean 의 이름과 BeanDefinition 을 저장한다. 그리고 `DefaultListableBeanFactory::beanDefinitionNames` 에 Bean 의 이름을 추가한다.

```java
// org.springframework.beans.factory.support.DefaultListableBeanFactory
public class DefaultListableBeanFactory extends AbstractAutowireCapableBeanFactory
		implements ConfigurableListableBeanFactory, BeanDefinitionRegistry, Serializable {
...
	private final Map<String, BeanDefinition> beanDefinitionMap = new ConcurrentHashMap<>(256);
...
	private volatile List<String> beanDefinitionNames = new ArrayList<>(256);
...
	@Override
	public void registerBeanDefinition(String beanName, BeanDefinition beanDefinition)
			throws BeanDefinitionStoreException {
...
				// Still in startup registration phase
				this.beanDefinitionMap.put(beanName, beanDefinition);
				this.beanDefinitionNames.add(beanName);				
```

`BeanDefinitionRegistry::registerBeanDefinition(String beanName, BeanDefinition beanDefinition)` 가 Bean 의 이름과 BeanDefinition 을 저장한다. `DefaultListableBeanFactory` 는 `BeanDefinitionRegistry` 를 implement 한다. 

```java
// org.springframework.boot.BeanDefinitionLoader
class BeanDefinitionLoader {
...	
	private int load(Object source) {
		Assert.notNull(source, "Source must not be null");
		if (source instanceof Class<?>) {
			return load((Class<?>) source);

// org.springframework.boot.BeanDefinitionLoader
class BeanDefinitionLoader {
...	
	private int load(Class<?> source) {
		if (isGroovyPresent() && GroovyBeanDefinitionSource.class.isAssignableFrom(source)) {
			// Any GroovyLoaders added in beans{} DSL can contribute beans here
			GroovyBeanDefinitionSource loader = BeanUtils.instantiateClass(source, GroovyBeanDefinitionSource.class);
			load(loader);
		}
		if (isComponent(source)) {
			this.annotatedReader.register(source);		

// org.springframework.context.annotation.AnnotatedBeanDefinitionReader
public class AnnotatedBeanDefinitionReader {
...
	public void register(Class<?>... componentClasses) {
		for (Class<?> componentClass : componentClasses) {
			registerBean(componentClass);

// org.springframework.context.annotation.AnnotatedBeanDefinitionReader
public class AnnotatedBeanDefinitionReader {
...
	public void registerBean(Class<?> beanClass) {
		doRegisterBean(beanClass, null, null, null, null);

// org.springframework.context.annotation.AnnotatedBeanDefinitionReader
public class AnnotatedBeanDefinitionReader {
...
	private <T> void doRegisterBean(Class<T> beanClass, @Nullable String name,
			@Nullable Class<? extends Annotation>[] qualifiers, @Nullable Supplier<T> supplier,
			@Nullable BeanDefinitionCustomizer[] customizers) {

		AnnotatedGenericBeanDefinition abd = new AnnotatedGenericBeanDefinition(beanClass);
		if (this.conditionEvaluator.shouldSkip(abd.getMetadata())) {
			return;
		}

		abd.setInstanceSupplier(supplier);
		ScopeMetadata scopeMetadata = this.scopeMetadataResolver.resolveScopeMetadata(abd);
		abd.setScope(scopeMetadata.getScopeName());
		String beanName = (name != null ? name : this.beanNameGenerator.generateBeanName(abd, this.registry));

		AnnotationConfigUtils.processCommonDefinitionAnnotations(abd);
		if (qualifiers != null) {
			for (Class<? extends Annotation> qualifier : qualifiers) {
				if (Primary.class == qualifier) {
					abd.setPrimary(true);
				}
				else if (Lazy.class == qualifier) {
					abd.setLazyInit(true);
				}
				else {
					abd.addQualifier(new AutowireCandidateQualifier(qualifier));
				}
			}
		}
		if (customizers != null) {
			for (BeanDefinitionCustomizer customizer : customizers) {
				customizer.customize(abd);
			}
		}

		BeanDefinitionHolder definitionHolder = new BeanDefinitionHolder(abd, beanName);
		definitionHolder = AnnotationConfigUtils.applyScopedProxyMode(scopeMetadata, definitionHolder, this.registry);
		BeanDefinitionReaderUtils.registerBeanDefinition(definitionHolder, this.registry);
...	

// org.springframework.beans.factory.support.BeanDefinitionReaderUtils
public abstract class BeanDefinitionReaderUtils {
...
	public static void registerBeanDefinition(
			BeanDefinitionHolder definitionHolder, BeanDefinitionRegistry registry)
			throws BeanDefinitionStoreException {

		// Register bean definition under primary name.
		String beanName = definitionHolder.getBeanName();
		registry.registerBeanDefinition(beanName, definitionHolder.getBeanDefinition());

// org.springframework.context.support.GenericApplicationContext
public class GenericApplicationContext extends AbstractApplicationContext implements BeanDefinitionRegistry {
...
	@Override
	public void registerBeanDefinition(String beanName, BeanDefinition beanDefinition)
			throws BeanDefinitionStoreException {

		this.beanFactory.registerBeanDefinition(beanName, beanDefinition);

// org.springframework.beans.factory.support.DefaultListableBeanFactory
public class DefaultListableBeanFactory extends AbstractAutowireCapableBeanFactory
		implements ConfigurableListableBeanFactory, BeanDefinitionRegistry, Serializable {
...
	private final Map<String, BeanDefinition> beanDefinitionMap = new ConcurrentHashMap<>(256);
...
	private volatile List<String> beanDefinitionNames = new ArrayList<>(256);
...
	@Override
	public void registerBeanDefinition(String beanName, BeanDefinition beanDefinition)
			throws BeanDefinitionStoreException {
...
				// Still in startup registration phase
				this.beanDefinitionMap.put(beanName, beanDefinition);
				this.beanDefinitionNames.add(beanName);
```

# Creating Bean Instance

`Object AbstractBeanFactory::createBean(String beanName, RootBeanDefinition mbd, @Nullable Object[] args)` 이 호출되고 `<T> T BeanUtils::instantiateClass(Constructor<T> ctor, Object... args)` 에서 `ctor.newInstance()` 를 호출하여 Bean Instance 를 생성한다.

```java
// org.springframework.beans.factory.support.AbstractBeanFactory
public abstract class AbstractBeanFactory extends FactoryBeanRegistrySupport implements ConfigurableBeanFactory {
...
	protected abstract Object createBean(String beanName, RootBeanDefinition mbd, @Nullable Object[] args)
			throws BeanCreationException;

// org.springframework.beans.factory.support.SimpleInstantiationStrategy
public class SimpleInstantiationStrategy implements InstantiationStrategy {
...
	@Override
	public Object instantiate(RootBeanDefinition bd, @Nullable String beanName, BeanFactory owner) {
```

다음은 `AbstractBeanFactory::createBean()` 을 추적한 것이다.

```java
// org.springframework.beans.factory.support.AbstractBeanFactory
public abstract class AbstractBeanFactory extends FactoryBeanRegistrySupport implements ConfigurableBeanFactory {
...
	protected abstract Object createBean(String beanName, RootBeanDefinition mbd, @Nullable Object[] args)
			throws BeanCreationException;

// org.springframework.beans.factory.support.AbstractAutowireCapableBeanFactory
public abstract class AbstractAutowireCapableBeanFactory extends AbstractBeanFactory
		implements AutowireCapableBeanFactory {
...
	@Override
	protected Object createBean(String beanName, RootBeanDefinition mbd, @Nullable Object[] args)
			throws BeanCreationException {

		if (logger.isTraceEnabled()) {
			logger.trace("Creating instance of bean '" + beanName + "'");
		}
		RootBeanDefinition mbdToUse = mbd;

		// Make sure bean class is actually resolved at this point, and
		// clone the bean definition in case of a dynamically resolved Class
		// which cannot be stored in the shared merged bean definition.
		Class<?> resolvedClass = resolveBeanClass(mbd, beanName);
		if (resolvedClass != null && !mbd.hasBeanClass() && mbd.getBeanClassName() != null) {
			mbdToUse = new RootBeanDefinition(mbd);
			mbdToUse.setBeanClass(resolvedClass);
		}

		// Prepare method overrides.
		try {
			mbdToUse.prepareMethodOverrides();
		}
		catch (BeanDefinitionValidationException ex) {
			throw new BeanDefinitionStoreException(mbdToUse.getResourceDescription(),
					beanName, "Validation of method overrides failed", ex);
		}

		try {
			// Give BeanPostProcessors a chance to return a proxy instead of the target bean instance.
			Object bean = resolveBeforeInstantiation(beanName, mbdToUse);
			if (bean != null) {
				return bean;
			}
		}
		catch (Throwable ex) {
			throw new BeanCreationException(mbdToUse.getResourceDescription(), beanName,
					"BeanPostProcessor before instantiation of bean failed", ex);
		}

		try {
			Object beanInstance = doCreateBean(beanName, mbdToUse, args);

// org.springframework.beans.factory.support.AbstractAutowireCapableBeanFactory
public abstract class AbstractAutowireCapableBeanFactory extends AbstractBeanFactory
		implements AutowireCapableBeanFactory {
...
	protected Object doCreateBean(final String beanName, final RootBeanDefinition mbd, final @Nullable Object[] args)
			throws BeanCreationException {

		// Instantiate the bean.
		BeanWrapper instanceWrapper = null;
		if (mbd.isSingleton()) {
			instanceWrapper = this.factoryBeanInstanceCache.remove(beanName);
		}
		if (instanceWrapper == null) {
			instanceWrapper = createBeanInstance(beanName, mbd, args);

// org.springframework.beans.factory.support.AbstractAutowireCapableBeanFactory
public abstract class AbstractAutowireCapableBeanFactory extends AbstractBeanFactory
		implements AutowireCapableBeanFactory {
...
	protected BeanWrapper createBeanInstance(String beanName, RootBeanDefinition mbd, @Nullable Object[] args) {
		// Make sure bean class is actually resolved at this point.
		Class<?> beanClass = resolveBeanClass(mbd, beanName);

		if (beanClass != null && !Modifier.isPublic(beanClass.getModifiers()) && !mbd.isNonPublicAccessAllowed()) {
			throw new BeanCreationException(mbd.getResourceDescription(), beanName,
					"Bean class isn't public, and non-public access not allowed: " + beanClass.getName());
		}

		Supplier<?> instanceSupplier = mbd.getInstanceSupplier();
		if (instanceSupplier != null) {
			return obtainFromSupplier(instanceSupplier, beanName);
		}

		if (mbd.getFactoryMethodName() != null) {
			return instantiateUsingFactoryMethod(beanName, mbd, args);
		}

		// Shortcut when re-creating the same bean...
		boolean resolved = false;
		boolean autowireNecessary = false;
		if (args == null) {
			synchronized (mbd.constructorArgumentLock) {
				if (mbd.resolvedConstructorOrFactoryMethod != null) {
					resolved = true;
					autowireNecessary = mbd.constructorArgumentsResolved;
				}
			}
		}
		if (resolved) {
			if (autowireNecessary) {
				return autowireConstructor(beanName, mbd, null, null);
			}
			else {
				return instantiateBean(beanName, mbd);
			}
		}

		// Candidate constructors for autowiring?
		Constructor<?>[] ctors = determineConstructorsFromBeanPostProcessors(beanClass, beanName);
		if (ctors != null || mbd.getResolvedAutowireMode() == AUTOWIRE_CONSTRUCTOR ||
				mbd.hasConstructorArgumentValues() || !ObjectUtils.isEmpty(args)) {
			return autowireConstructor(beanName, mbd, ctors, args);
		}

		// Preferred constructors for default construction?
		ctors = mbd.getPreferredConstructors();
		if (ctors != null) {
			return autowireConstructor(beanName, mbd, ctors, null);
		}

		// No special handling: simply use no-arg constructor.
		return instantiateBean(beanName, mbd);

// org.springframework.beans.factory.support.AbstractAutowireCapableBeanFactory
public abstract class AbstractAutowireCapableBeanFactory extends AbstractBeanFactory
		implements AutowireCapableBeanFactory {
...
	protected BeanWrapper instantiateBean(final String beanName, final RootBeanDefinition mbd) {
		try {
			Object beanInstance;
			final BeanFactory parent = this;
			if (System.getSecurityManager() != null) {
				beanInstance = AccessController.doPrivileged((PrivilegedAction<Object>) () ->
						getInstantiationStrategy().instantiate(mbd, beanName, parent),
						getAccessControlContext());
			}
			else {
				beanInstance = getInstantiationStrategy().instantiate(mbd, beanName, parent);	

// org.springframework.beans.factory.support.SimpleInstantiationStrategy
public class SimpleInstantiationStrategy implements InstantiationStrategy {
...
	@Override
	public Object instantiate(RootBeanDefinition bd, @Nullable String beanName, BeanFactory owner) {
		// Don't override the class with CGLIB if no overrides.
		if (!bd.hasMethodOverrides()) {
			Constructor<?> constructorToUse;
			synchronized (bd.constructorArgumentLock) {
				constructorToUse = (Constructor<?>) bd.resolvedConstructorOrFactoryMethod;
				if (constructorToUse == null) {
					final Class<?> clazz = bd.getBeanClass();
					if (clazz.isInterface()) {
						throw new BeanInstantiationException(clazz, "Specified class is an interface");
					}
					try {
						if (System.getSecurityManager() != null) {
							constructorToUse = AccessController.doPrivileged(
									(PrivilegedExceptionAction<Constructor<?>>) clazz::getDeclaredConstructor);
						}
						else {
							constructorToUse = clazz.getDeclaredConstructor();
						}
						bd.resolvedConstructorOrFactoryMethod = constructorToUse;
					}
					catch (Throwable ex) {
						throw new BeanInstantiationException(clazz, "No default constructor found", ex);
					}
				}
			}
			return BeanUtils.instantiateClass(constructorToUse);
```

# Getting Bean Instance

`Object BeanFactory::getBean(String name)` 이 특정한 이름의 Bean Instance 를
얻어온다. `DefaultSingletonBeanRegistry::singletonObjects` 에 있으면 가져오고
없으면 생성한다. Bean 을 생성하는 것은 [Creating Bean
Instance](#creating-bean-instance) 를 참고하자.

```java
// org.springframework.beans.factory.BeanFactory
public interface BeanFactory {
...
	Object getBean(String name) throws BeansException;

// org.springframework.beans.factory.support.DefaultSingletonBeanRegistry
public class DefaultSingletonBeanRegistry extends SimpleAliasRegistry implements SingletonBeanRegistry {

	/** Cache of singleton objects: bean name to bean instance. */
	private final Map<String, Object> singletonObjects = new ConcurrentHashMap<>(256);
```

다음은 `BeanFactory::getBean()` 의 흐름을 추적한 것이다.

```java
// org.springframework.beans.factory.BeanFactory
public interface BeanFactory {
...
	Object getBean(String name) throws BeansException;

// org.springframework.beans.factory.support.AbstractBeanFactory
public abstract class AbstractBeanFactory extends FactoryBeanRegistrySupport implements ConfigurableBeanFactory {
...
	protected <T> T doGetBean(final String name, @Nullable final Class<T> requiredType,
			@Nullable final Object[] args, boolean typeCheckOnly) throws BeansException {

		final String beanName = transformedBeanName(name);
		Object bean;

		// Eagerly check singleton cache for manually registered singletons.
		Object sharedInstance = getSingleton(beanName);
		if (sharedInstance != null && args == null) {
			if (logger.isTraceEnabled()) {
				if (isSingletonCurrentlyInCreation(beanName)) {
					logger.trace("Returning eagerly cached instance of singleton bean '" + beanName +
							"' that is not fully initialized yet - a consequence of a circular reference");
				}
				else {
					logger.trace("Returning cached instance of singleton bean '" + beanName + "'");
				}
			}
			bean = getObjectForBeanInstance(sharedInstance, name, beanName, null);
		}

		else {
			// Fail if we're already creating this bean instance:
			// We're assumably within a circular reference.
			if (isPrototypeCurrentlyInCreation(beanName)) {
				throw new BeanCurrentlyInCreationException(beanName);
			}

			// Check if bean definition exists in this factory.
			BeanFactory parentBeanFactory = getParentBeanFactory();
			if (parentBeanFactory != null && !containsBeanDefinition(beanName)) {
				// Not found -> check parent.
				String nameToLookup = originalBeanName(name);
				if (parentBeanFactory instanceof AbstractBeanFactory) {
					return ((AbstractBeanFactory) parentBeanFactory).doGetBean(
							nameToLookup, requiredType, args, typeCheckOnly);
				}
				else if (args != null) {
					// Delegation to parent with explicit args.
					return (T) parentBeanFactory.getBean(nameToLookup, args);
				}
				else if (requiredType != null) {
					// No args -> delegate to standard getBean method.
					return parentBeanFactory.getBean(nameToLookup, requiredType);
				}
				else {
					return (T) parentBeanFactory.getBean(nameToLookup);
				}
			}

			if (!typeCheckOnly) {
				markBeanAsCreated(beanName);
			}

			try {
				final RootBeanDefinition mbd = getMergedLocalBeanDefinition(beanName);
				checkMergedBeanDefinition(mbd, beanName, args);

				// Guarantee initialization of beans that the current bean depends on.
				String[] dependsOn = mbd.getDependsOn();
				if (dependsOn != null) {
					for (String dep : dependsOn) {
						if (isDependent(beanName, dep)) {
							throw new BeanCreationException(mbd.getResourceDescription(), beanName,
									"Circular depends-on relationship between '" + beanName + "' and '" + dep + "'");
						}
						registerDependentBean(dep, beanName);
						try {
							getBean(dep);
						}
						catch (NoSuchBeanDefinitionException ex) {
							throw new BeanCreationException(mbd.getResourceDescription(), beanName,
									"'" + beanName + "' depends on missing bean '" + dep + "'", ex);
						}
					}
				}

				// Create bean instance.
				if (mbd.isSingleton()) {
					sharedInstance = getSingleton(beanName, () -> {

// org.springframework.beans.factory.support.DefaultSingletonBeanRegistry
public class DefaultSingletonBeanRegistry extends SimpleAliasRegistry implements SingletonBeanRegistry {
...
	public Object getSingleton(String beanName, ObjectFactory<?> singletonFactory) {
		Assert.notNull(beanName, "Bean name must not be null");
		synchronized (this.singletonObjects) {
			Object singletonObject = this.singletonObjects.get(beanName);
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

`@SpringBootConfiguration` 를 부착하면 `@Configuration` Class 가 된다.
`@SpringBootConfiguration` 에 `@Configuration` 이 부착되어 있기 때문이다. 즉,
`@Bean` Method 는 Bean Instance 을 생성할 수 있다. 

```java
// org.springframework.boot.SpringBootConfiguration
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Configuration
public @interface SpringBootConfiguration {
```

# `@Configuration`

`ConfigurationClassParser::processConfigurationClass(ConfigurationClass configClass, Predicate<String> filter)` 를 호출하여 처리한다. `configClass` 는 `@Configuration` Class 와 같다. 주로 `@Bean` Method 가 return 하는 Object 를 Bean Instance 로 등록한다. 

```java		
// org.springframework.context.annotation.ConfigurationClassParser
class ConfigurationClassParser {
...
	protected void processConfigurationClass(ConfigurationClass configClass, Predicate<String> filter) throws IOException {
		if (this.conditionEvaluator.shouldSkip(configClass.getMetadata(), ConfigurationPhase.PARSE_CONFIGURATION)) {
			return;
		}

		ConfigurationClass existingClass = this.configurationClasses.get(configClass);
		if (existingClass != null) {
			if (configClass.isImported()) {
				if (existingClass.isImported()) {
					existingClass.mergeImportedBy(configClass);
				}
				// Otherwise ignore new imported config class; existing non-imported class overrides it.
				return;
			}
			else {
				// Explicit bean definition found, probably replacing an import.
				// Let's remove the old one and go with the new one.
				this.configurationClasses.remove(configClass);
				this.knownSuperclasses.values().removeIf(configClass::equals);
			}
		}

		// Recursively process the configuration class and its superclass hierarchy.
		SourceClass sourceClass = asSourceClass(configClass, filter);
		do {
			sourceClass = doProcessConfigurationClass(configClass, sourceClass, filter);
		}
		while (sourceClass != null);

		this.configurationClasses.put(configClass, configClass);
	}

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

# `@Import`

`@Import` 의 Arguement 로 특별한 Class 를 넘기면 그 Class 의 Instance 를 Bean
Instance 으로 등록한다. 특별한 Class 를 Importee Class 라고 하자. 종류는 다음과 같다.

* `candidate.isAssignable(ImportSelector.class) == true` 인 Class
* `candidate.isAssignable(ImportBeanDefinitionRegistrar.class) == true` 인 Class
* `@Configuration` Class

```java
// org.springframework.context.annotation.ConfigurationClassParser
//        configClass: @Import 가 부착된 Class
// currentSourceClass: configClass 의 SourceClass
//   importCandidates: @Import 의 대상인 Importee Classes
class ConfigurationClassParser {
...
	private void processImports(ConfigurationClass configClass, SourceClass currentSourceClass,
			Collection<SourceClass> importCandidates, Predicate<String> exclusionFilter,
			boolean checkForCircularImports) {

		if (importCandidates.isEmpty()) {
			return;
		}

		if (checkForCircularImports && isChainedImportOnStack(configClass)) {
			this.problemReporter.error(new CircularImportProblem(configClass, this.importStack));
		}
		else {
			this.importStack.push(configClass);
			try {
				for (SourceClass candidate : importCandidates) {
					if (candidate.isAssignable(ImportSelector.class)) {
						// Candidate class is an ImportSelector -> delegate to it to determine imports
						Class<?> candidateClass = candidate.loadClass();
						ImportSelector selector = ParserStrategyUtils.instantiateClass(candidateClass, ImportSelector.class,
								this.environment, this.resourceLoader, this.registry);
						Predicate<String> selectorFilter = selector.getExclusionFilter();
						if (selectorFilter != null) {
							exclusionFilter = exclusionFilter.or(selectorFilter);
						}
						if (selector instanceof DeferredImportSelector) {
							this.deferredImportSelectorHandler.handle(configClass, (DeferredImportSelector) selector);
						}
						else {
							String[] importClassNames = selector.selectImports(currentSourceClass.getMetadata());
							Collection<SourceClass> importSourceClasses = asSourceClasses(importClassNames, exclusionFilter);
							processImports(configClass, currentSourceClass, importSourceClasses, exclusionFilter, false);
						}
					}
					else if (candidate.isAssignable(ImportBeanDefinitionRegistrar.class)) {
						// Candidate class is an ImportBeanDefinitionRegistrar ->
						// delegate to it to register additional bean definitions
						Class<?> candidateClass = candidate.loadClass();
						ImportBeanDefinitionRegistrar registrar =
								ParserStrategyUtils.instantiateClass(candidateClass, ImportBeanDefinitionRegistrar.class,
										this.environment, this.resourceLoader, this.registry);
						configClass.addImportBeanDefinitionRegistrar(registrar, currentSourceClass.getMetadata());
					}
					else {
						// Candidate class not an ImportSelector or ImportBeanDefinitionRegistrar ->
						// process it as an @Configuration class
						this.importStack.registerImport(
								currentSourceClass.getMetadata(), candidate.getMetadata().getClassName());
						processConfigurationClass(candidate.asConfigClass(configClass), exclusionFilter);
					}
				}
			}
			catch (BeanDefinitionStoreException ex) {
				throw ex;
			}
			catch (Throwable ex) {
				throw new BeanDefinitionStoreException(
						"Failed to process import candidates for configuration class [" +
						configClass.getMetadata().getClassName() + "]", ex);
			}
			finally {
				this.importStack.pop();
			}
		}
	}
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

`@Bean` Method 가 return 하는 Object 는 Bean 으로 등록된다. `@Configuration`
Class 안에서 선언되야 생성되는 Bean Instance 의 Single-ton 이 보장된다. 주로
`@Configuration` Class 안에서만 사용된다. [@Configuration](#configuration) 참고.

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

`AutoConfigurationImportSelector.class` 를 Import 하고 있다. 또한 `AutoConfigurationImportSelector.class` 는 `candidate.isAssignable(ImportSelector.class) == true` 인 Class 이다. [`@Import`](#import) 참고.

```java
// org.springframework.boot.autoconfigure.EnableAutoConfiguration
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Inherited
@AutoConfigurationPackage
@Import(AutoConfigurationImportSelector.class)
public @interface EnableAutoConfiguration {

// org.springframework.boot.autoconfigure.AutoConfigurationImportSelector
public class AutoConfigurationImportSelector implements DeferredImportSelector, BeanClassLoaderAware,
		ResourceLoaderAware, BeanFactoryAware, EnvironmentAware, Ordered {

// org.springframework.context.annotation.DeferredImportSelector
public interface DeferredImportSelector extends ImportSelector {
```

`AutoConfigurationImportSelector` 는 `spring.factories` 에 저장된 key 와 value 들을 읽어서
`@Configuration` Class 처럼 처리한다. 즉, value 를 `ConfigurationClassParser::processConfigurationClass(ConfigurationClass configClass, Predicate<String> filter)` 의 configClass 로 넘겨서 호출한다.

```java
```

`SpringFactoriesLoader::FACTORIES_RESOURCE_LOCATION` 에 spring factory file (`spring.factories`)의 경로가 hard coding 되어 있다.

```java
// org.springframework.core.io.support.SpringFactoriesLoader
public final class SpringFactoriesLoader {
...
	public static final String FACTORIES_RESOURCE_LOCATION = "META-INF/spring.factories";

// META-INF/spring.factories
...
# Auto Configure
org.springframework.boot.autoconfigure.EnableAutoConfiguration=\
org.springframework.boot.autoconfigure.admin.SpringApplicationAdminJmxAutoConfiguration,\
org.springframework.boot.autoconfigure.aop.AopAutoConfiguration,\
org.springframework.boot.autoconfigure.amqp.RabbitAutoConfiguration,\
org.springframework.boot.autoconfigure.batch.BatchAutoConfiguration,\
org.springframework.boot.autoconfigure.cache.CacheAutoConfiguration,\
...
```

`List<String> SpringFactoriesLoader::loadFactoryNames(Class<?> factoryType, @Nullable ClassLoader classLoader)` 에서 `spring.factories` 의 key 들중
`factoryType` 의 Class Path 를 고르고 그 값들을 return 한다. 다음은
`spring.factories` 파일의 `EnableAutoConfiguration` key 의 값들을 return 하는
흐름이다. 
 
```java
// org.springframework.boot.autoconfigure.AutoConfigurationImportSelector
public class AutoConfigurationImportSelector implements DeferredImportSelector, BeanClassLoaderAware,
		ResourceLoaderAware, BeanFactoryAware, EnvironmentAware, Ordered {
...
	protected List<String> getCandidateConfigurations(AnnotationMetadata metadata, AnnotationAttributes attributes) {
		List<String> configurations = SpringFactoriesLoader.loadFactoryNames(getSpringFactoriesLoaderFactoryClass(),
				getBeanClassLoader());

// org.springframework.boot.autoconfigure.AutoConfigurationImportSelector
public class AutoConfigurationImportSelector implements DeferredImportSelector, BeanClassLoaderAware,
		ResourceLoaderAware, BeanFactoryAware, EnvironmentAware, Ordered {
...
	protected Class<?> getSpringFactoriesLoaderFactoryClass() {
		return EnableAutoConfiguration.class;
	}

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
```

# `@Repository`

`@Repository` Class 의 Proxy Class 를 Bean 으로 등록한다. Proxy Class 는 다음과 같이 생성한다.

```java
// org.springframework.data.repository.core.support.RepositoryFactorySupport
@Slf4j
public abstract class RepositoryFactorySupport implements BeanClassLoaderAware, BeanFactoryAware {
...
	public <T> T getRepository(Class<T> repositoryInterface, RepositoryFragments fragments) {

		if (LOG.isDebugEnabled()) {
			LOG.debug("Initializing repository instance for {}…", repositoryInterface.getName());
		}

		Assert.notNull(repositoryInterface, "Repository interface must not be null!");
		Assert.notNull(fragments, "RepositoryFragments must not be null!");

		RepositoryMetadata metadata = getRepositoryMetadata(repositoryInterface);
		RepositoryComposition composition = getRepositoryComposition(metadata, fragments);
		RepositoryInformation information = getRepositoryInformation(metadata, composition);

		validate(information, composition);

		Object target = getTargetRepository(information);

		// Create proxy
		ProxyFactory result = new ProxyFactory();
		result.setTarget(target);
		result.setInterfaces(repositoryInterface, Repository.class, TransactionalProxy.class);

		if (MethodInvocationValidator.supports(repositoryInterface)) {
			result.addAdvice(new MethodInvocationValidator());
		}

		result.addAdvisor(ExposeInvocationInterceptor.ADVISOR);

		postProcessors.forEach(processor -> processor.postProcess(result, information));

		if (DefaultMethodInvokingMethodInterceptor.hasDefaultMethods(repositoryInterface)) {
			result.addAdvice(new DefaultMethodInvokingMethodInterceptor());
		}

		ProjectionFactory projectionFactory = getProjectionFactory(classLoader, beanFactory);
		result.addAdvice(new QueryExecutorMethodInterceptor(information, projectionFactory));

		composition = composition.append(RepositoryFragment.implemented(target));
		result.addAdvice(new ImplementationMethodExecutionInterceptor(composition));

		T repository = (T) result.getProxy(classLoader);

		if (LOG.isDebugEnabled()) {
			LOG.debug("Finished creation of repository instance for {}.", repositoryInterface.getName());
		}

		return repository;
	}
```

# `@Transactional`

`TransactionManager::commit()` 이 호출되는 부분을 중심으로 이해하자.

Spring Application 이 실행되었을 때 다음과 같이 `transactionInterceptor` 가 등록된다. 이후 Transaction 이 발생할 때 마다 `TransactionIterceptor` class 의 `invokeWithinTransaction()` 가 호출된다.

```java
// org.springframework.data.repository.core.support.TransactionalRepositoryProxyPostProcessor
class TransactionalRepositoryProxyPostProcessor implements RepositoryProxyPostProcessor {
...
	public void postProcess(ProxyFactory factory, RepositoryInformation repositoryInformation) {

		TransactionInterceptor transactionInterceptor = new TransactionInterceptor();
		transactionInterceptor.setTransactionAttributeSource(
				new RepositoryAnnotationTransactionAttributeSource(repositoryInformation, enableDefaultTransactions));
		transactionInterceptor.setTransactionManagerBeanName(transactionManagerName);
		transactionInterceptor.setBeanFactory(beanFactory);
		transactionInterceptor.afterPropertiesSet();

		factory.addAdvice(transactionInterceptor);
	}
...
}
```

다음은 Transaction 이 발생할 때 마다 실제로 호출되는 흐름이다. `h2` 로 실습했다.

```java
// org.springframework.transaction.interceptor.TransactionInterceptor
public abstract class TransactionAspectSupport implements BeanFactoryAware, InitializingBean {
...
	@Nullable
	protected Object invokeWithinTransaction(Method method, @Nullable Class<?> targetClass,
			final InvocationCallback invocation) throws Throwable {
	...
		if (txAttr == null || !(ptm instanceof CallbackPreferringPlatformTransactionManager)) {
			// Standard transaction demarcation with getTransaction and commit/rollback calls.
			TransactionInfo txInfo = createTransactionIfNecessary(ptm, txAttr, joinpointIdentification);

			Object retVal;
			try {
				// This is an around advice: Invoke the next interceptor in the chain.
				// This will normally result in a target object being invoked.
				retVal = invocation.proceedWithInvocation();
...				
}

// com.iamslash.exjpa.post.PostService
@Service
public class PostService {

    PostRepository postsRepository;

    public PostService(PostRepository postsRepository) {
        this.postsRepository = postsRepository;
    }

    @Transactional
    public void save(Post post) {
        postsRepository.save(post);
    }
}

// org.springframework.data.jpa.repository.support.SimpleJpaRepository
@Repository
@Transactional(readOnly = true)
public class SimpleJpaRepository<T, ID> implements JpaRepositoryImplementation<T, ID> {
...
	@Transactional
	@Override
	public <S extends T> S save(S entity) {

		if (entityInformation.isNew(entity)) {
			em.persist(entity);
			return entity;
		} else {
			return em.merge(entity);
		}
	}
...
}

// org.springframework.transaction.interceptor.TransactionInterceptor
public abstract class TransactionAspectSupport implements BeanFactoryAware, InitializingBean {
...
	@Nullable
	protected Object invokeWithinTransaction(Method method, @Nullable Class<?> targetClass,
			final InvocationCallback invocation) throws Throwable {
		...
		if (txAttr == null || !(ptm instanceof CallbackPreferringPlatformTransactionManager)) {
			...
			commitTransactionAfterReturning(txInfo);
			return retVal;
		}
...	
}

// org.springframework.transaction.interceptor.TransactionInterceptor
public abstract class TransactionAspectSupport implements BeanFactoryAware, InitializingBean {
...
	protected void commitTransactionAfterReturning(@Nullable TransactionInfo txInfo) {
		if (txInfo != null && txInfo.getTransactionStatus() != null) {
			if (logger.isTraceEnabled()) {
				logger.trace("Completing transaction for [" + txInfo.getJoinpointIdentification() + "]");
			}
			txInfo.getTransactionManager().commit(txInfo.getTransactionStatus());
		}
	}
...
}	

// org.springframework.transaction.support.AbstractPlatformTransactionManager
public abstract class AbstractPlatformTransactionManager implements PlatformTransactionManager, Serializable {
...
	@Override
	public final void commit(TransactionStatus status) throws TransactionException {
		if (status.isCompleted()) {
			throw new IllegalTransactionStateException(
					"Transaction is already completed - do not call commit or rollback more than once per transaction");
		}

		DefaultTransactionStatus defStatus = (DefaultTransactionStatus) status;
		if (defStatus.isLocalRollbackOnly()) {
			if (defStatus.isDebug()) {
				logger.debug("Transactional code has requested rollback");
			}
			processRollback(defStatus, false);
			return;
		}

		if (!shouldCommitOnGlobalRollbackOnly() && defStatus.isGlobalRollbackOnly()) {
			if (defStatus.isDebug()) {
				logger.debug("Global transaction is marked as rollback-only but transactional code requested commit");
			}
			processRollback(defStatus, true);
			return;
		}

		processCommit(defStatus);
...
}

// org.springframework.transaction.support.AbstractPlatformTransactionManager
public abstract class AbstractPlatformTransactionManager implements PlatformTransactionManager, Serializable {
...
	private void processCommit(DefaultTransactionStatus status) throws TransactionException {
		try {
			boolean beforeCompletionInvoked = false;

			try {
				boolean unexpectedRollback = false;
				prepareForCommit(status);
				triggerBeforeCommit(status);
				triggerBeforeCompletion(status);
				beforeCompletionInvoked = true;

				if (status.hasSavepoint()) {
					if (status.isDebug()) {
						logger.debug("Releasing transaction savepoint");
					}
					unexpectedRollback = status.isGlobalRollbackOnly();
					status.releaseHeldSavepoint();
				}
				else if (status.isNewTransaction()) {
					if (status.isDebug()) {
						logger.debug("Initiating transaction commit");
					}
					unexpectedRollback = status.isGlobalRollbackOnly();
					doCommit(status);

// org.springframework.orm.jpa.JpaTransactionManager
public class JpaTransactionManager extends AbstractPlatformTransactionManager
		implements ResourceTransactionManager, BeanFactoryAware, InitializingBean {
...
	@Override
	protected void doCommit(DefaultTransactionStatus status) {
		JpaTransactionObject txObject = (JpaTransactionObject) status.getTransaction();
		if (status.isDebug()) {
			logger.debug("Committing JPA transaction on EntityManager [" +
					txObject.getEntityManagerHolder().getEntityManager() + "]");
		}
		try {
			EntityTransaction tx = txObject.getEntityManagerHolder().getEntityManager().getTransaction();
			tx.commit();

// org.hibernate.engine.transaction.internal.TransactionImpl
public class TransactionImpl implements TransactionImplementor {
...
	@Override
	public void commit() {
		if ( !isActive( true ) ) {
			// allow MARKED_ROLLBACK to propagate through to transactionDriverControl
			// the boolean passed to isActive indicates whether MARKED_ROLLBACK should be
			// considered active
			//
			// essentially here we have a transaction that is not active and
			// has not been marked for rollback only
			throw new IllegalStateException( "Transaction not successfully started" );
		}

		LOG.debug( "committing" );

		try {
			internalGetTransactionDriverControl().commit();

// org.hibernate.resource.transaction.backend.jdbc.internal.JdbcResourceLocalTransactionCoordinatorImpl
public class JdbcResourceLocalTransactionCoordinatorImpl implements TransactionCoordinator {
...
		@Override
		public void commit() {
			try {
				if ( rollbackOnly ) {
					log.debugf( "On commit, transaction was marked for roll-back only, rolling back" );

					try {
						rollback();

						if ( jpaCompliance.isJpaTransactionComplianceEnabled() ) {
							log.debugf( "Throwing RollbackException on roll-back of transaction marked rollback-only on commit" );
							throw new RollbackException( "Transaction was marked for rollback-only" );
						}

						return;
					}
					catch (RollbackException e) {
						throw e;
					}
					catch (RuntimeException e) {
						log.debug( "Encountered failure rolling back failed commit", e );
						throw e;
					}
				}

				JdbcResourceLocalTransactionCoordinatorImpl.this.beforeCompletionCallback();
				jdbcResourceTransaction.commit();

// org.hibernate.resource.jdbc.internal.AbstractLogicalConnectionImplementor
public abstract class AbstractLogicalConnectionImplementor implements LogicalConnectionImplementor, PhysicalJdbcTransaction {
...
	@Override
	public void commit() {
		try {
			log.trace( "Preparing to commit transaction via JDBC Connection.commit()" );
			getConnectionForTransactionManagement().commit();

// com.zaxxer.hikari.pool.ProxyConnection
public abstract class ProxyConnection implements Connection
{
...
   @Override
   public void commit() throws SQLException
   {
      delegate.commit();

// org.h2.jdbc.JdbcConnection
public class JdbcConnection extends TraceObject implements Connection, JdbcConnectionBackwardsCompat,
        CastDataProvider {
...
    @Override
    public synchronized void commit() throws SQLException {
        try {
            debugCodeCall("commit");
            checkClosedForWrite();
            if (SysProperties.FORCE_AUTOCOMMIT_OFF_ON_COMMIT
                    && getAutoCommit()) {
                throw DbException.get(ErrorCode.METHOD_DISABLED_ON_AUTOCOMMIT_TRUE, "commit()");
            }
            commit = prepareCommand("COMMIT", commit);
            commit.executeUpdate(null);
```

# `@Service`

# `@Component`

# `@Controller`

# `@RestController`

# `@Validation`
