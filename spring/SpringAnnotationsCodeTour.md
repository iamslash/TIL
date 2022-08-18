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

다음과 같이 `MergedAnnotation` 을 이용하는 걸까?

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
