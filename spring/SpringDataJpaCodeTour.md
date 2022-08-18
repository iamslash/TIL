- [Abstract](#abstract)
- [Materials](#materials)
- [JPA (Java Persistence API)](#jpa-java-persistence-api)
- [JpaRepository Class](#jparepository-class)
- [@Transactional Class](#transactional-class)

----

# Abstract

Spring Data Jpa 의 code 를 분석해 본다. 

# Materials

* [Spring Data JPA | github](https://github.com/spring-projects/spring-data-jpa)
* [Understanding Spring JPA native query under the hood | stackoverflow](https://stackoverflow.com/questions/58784625/understanding-spring-jpa-native-query-under-the-hood)

# JPA (Java Persistence API)

* [JPA는 도대체 뭘까? (orm, 영속성, hibernate, spring-data-jpa) | velog](https://velog.io/@adam2/JPA%EB%8A%94-%EB%8F%84%EB%8D%B0%EC%B2%B4-%EB%AD%98%EA%B9%8C-orm-%EC%98%81%EC%86%8D%EC%84%B1-hibernate-spring-data-jpa)

JPA 는 Persistence, ORM 을 위한 API Specification 이다. JPA repo 는 [Jakarta Persistence project | github](https://github.com/eclipse-ee4j/jpa-api) 이다.

[hibernate-orm | github](https://github.com/hibernate/hibernate-orm) 는 JPA 의 implementation 이다. [Spring Data JPA](SpringDataJpa.md) 는 JPA 를 더욱 쉽게 사용하라고 Spring 에서 만든 Framework 이다. 

일반적으로 우리는 Spring Application 으로 ORM 을 이용할 때 JPA 혹은 [hibernate-orm | github](https://github.com/hibernate/hibernate-orm) 를 호출하는 것보다 [Spring Data JPA](SpringDataJpa.md) 를 사용한다.

# JpaRepository Class

* find and extract the query from the annotation.
* possibly create a count query from that.
* replace spell expression with parameters.
* add ordering if applicable.
* prepare the query with the EntityManager.
* register parameters.
* add pagination limits.
* execute the query.
* transform result.

`EntityManager::persist()` 가 호출되는 분을 중심으로 흐름을 이해하자. [How are Spring Data repositories actually implemented? | stackoverflow](https://stackoverflow.com/questions/38509882/how-are-spring-data-repositories-actually-implemented) 를 참고.

```java
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

// org.springframework.data.repository.core.support.RepositoryFactorySupport
@Slf4j
public abstract class RepositoryFactorySupport implements BeanClassLoaderAware, BeanFactoryAware {
...
		@Nullable
		private Object doInvoke(MethodInvocation invocation) throws Throwable {

			Method method = invocation.getMethod();

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

// org.hibernate.persister.entity.AbstractEntityPersister
public abstract class AbstractEntityPersister
		implements OuterJoinLoadable, Queryable, ClassMetadata, UniqueKeyLoadable,
		SQLLoadable, LazyPropertyInitializer, PostInsertIdentityPersister, Lockable {
...
	public Serializable insert(Object[] fields, Object object, SharedSessionContractImplementor session)
			throws HibernateException {
		// apply any pre-insert in-memory value generation
		preInsertInMemoryValueGeneration( fields, object, session );

		final int span = getTableSpan();
		final Serializable id;
		if ( entityMetamodel.isDynamicInsert() ) {
			// For the case of dynamic-insert="true", we need to generate the INSERT SQL
			boolean[] notNull = getPropertiesToInsert( fields );
			id = insert( fields, notNull, generateInsertString( true, notNull ), object, session );
			for ( int j = 1; j < span; j++ ) {
				insert( id, fields, notNull, j, generateInsertString( notNull, j ), object, session );
			}
		}
		else {
			// For the case of dynamic-insert="false", use the static SQL
			id = insert( fields, getPropertyInsertability(), getSQLIdentityInsertString(), object, session );
```

`AbstractEntityPersister::sqlIdentityInsertString` 는 insert SQL 를 갖고 있다. Spring Application 이 시작할 때 `AbstractEntityPersister::sqlIdentityInsertString` 를 채운다.

```java
// org.hibernate.persister.entity.AbstractEntityPersister
public abstract class AbstractEntityPersister
		implements OuterJoinLoadable, Queryable, ClassMetadata, UniqueKeyLoadable,
		SQLLoadable, LazyPropertyInitializer, PostInsertIdentityPersister, Lockable {
...
	public final void postInstantiate() throws MappingException {
		doLateInit();

// org.hibernate.persister.entity.AbstractEntityPersister
public abstract class AbstractEntityPersister
		implements OuterJoinLoadable, Queryable, ClassMetadata, UniqueKeyLoadable,
		SQLLoadable, LazyPropertyInitializer, PostInsertIdentityPersister, Lockable {
...
	private void doLateInit() {
		//insert/update/delete SQL
		final int joinSpan = getTableSpan();
		sqlDeleteStrings = new String[joinSpan];
		sqlInsertStrings = new String[joinSpan];
		sqlUpdateStrings = new String[joinSpan];
		sqlLazyUpdateStrings = new String[joinSpan];

		sqlUpdateByRowIdString = rowIdName == null ?
				null :
				generateUpdateString( getPropertyUpdateability(), 0, true );
		sqlLazyUpdateByRowIdString = rowIdName == null ?
				null :
				generateUpdateString( getNonLazyPropertyUpdateability(), 0, true );

		for ( int j = 0; j < joinSpan; j++ ) {
			sqlInsertStrings[j] = customSQLInsert[j] == null ?
					generateInsertString( getPropertyInsertability(), j ) :
						substituteBrackets( customSQLInsert[j]);
			sqlUpdateStrings[j] = customSQLUpdate[j] == null ?
					generateUpdateString( getPropertyUpdateability(), j, false ) :
						substituteBrackets( customSQLUpdate[j]);
			sqlLazyUpdateStrings[j] = customSQLUpdate[j] == null ?
					generateUpdateString( getNonLazyPropertyUpdateability(), j, false ) :
						substituteBrackets( customSQLUpdate[j]);
			sqlDeleteStrings[j] = customSQLDelete[j] == null ?
					generateDeleteString( j ) :
						substituteBrackets( customSQLDelete[j]);
		}

		tableHasColumns = new boolean[joinSpan];
		for ( int j = 0; j < joinSpan; j++ ) {
			tableHasColumns[j] = sqlUpdateStrings[j] != null;
		}

		//select SQL
		sqlSnapshotSelectString = generateSnapshotSelectString();
		sqlLazySelectStringsByFetchGroup = generateLazySelectStringsByFetchGroup();
		sqlVersionSelectString = generateSelectVersionString();
		if ( hasInsertGeneratedProperties() ) {
			sqlInsertGeneratedValuesSelectString = generateInsertGeneratedValuesSelectString();
		}
		if ( hasUpdateGeneratedProperties() ) {
			sqlUpdateGeneratedValuesSelectString = generateUpdateGeneratedValuesSelectString();
		}
		if ( isIdentifierAssignedByInsert() ) {
			identityDelegate = ( (PostInsertIdentifierGenerator) getIdentifierGenerator() )
					.getInsertGeneratedIdentifierDelegate( this, getFactory().getDialect(), useGetGeneratedKeys() );
			sqlIdentityInsertString = customSQLInsert[0] == null
					? generateIdentityInsertString( getPropertyInsertability() )

// org.hibernate.persister.entity.AbstractEntityPersister
public abstract class AbstractEntityPersister
		implements OuterJoinLoadable, Queryable, ClassMetadata, UniqueKeyLoadable,
		SQLLoadable, LazyPropertyInitializer, PostInsertIdentityPersister, Lockable {
...
	public String generateIdentityInsertString(boolean[] includeProperty) {
		Insert insert = identityDelegate.prepareIdentifierGeneratingInsert();
		insert.setTableName( getTableName( 0 ) );

		// add normal properties except lobs
		for ( int i = 0; i < entityMetamodel.getPropertySpan(); i++ ) {
			if ( isPropertyOfTable( i, 0 ) && !lobProperties.contains( i ) ) {
				final InDatabaseValueGenerationStrategy generationStrategy = entityMetamodel.getInDatabaseValueGenerationStrategies()[i];

				if ( includeProperty[i] ) {
					insert.addColumns(
							getPropertyColumnNames( i ),
							propertyColumnInsertable[i],
							propertyColumnWriters[i]
					);
				}
				else if ( generationStrategy != null &&
						generationStrategy.getGenerationTiming().includesInsert() &&
						generationStrategy.referenceColumnsInSql() ) {

					final String[] values;

					if ( generationStrategy.getReferencedColumnValues() == null ) {
						values = propertyColumnWriters[i];
					}
					else {
						values = new String[propertyColumnWriters[i].length];

						for ( int j = 0; j < values.length; j++ ) {
							values[j] = ( generationStrategy.getReferencedColumnValues()[j] != null ) ?
									generationStrategy.getReferencedColumnValues()[j] :
									propertyColumnWriters[i][j];
						}
					}
					insert.addColumns(
							getPropertyColumnNames( i ),
							propertyColumnInsertable[i],
							values
					);
				}
			}
		}

		// HHH-4635 & HHH-8103
		// Oracle expects all Lob properties to be last in inserts
		// and updates.  Insert them at the end.
		for ( int i : lobProperties ) {
			if ( includeProperty[i] && isPropertyOfTable( i, 0 ) ) {
				insert.addColumns( getPropertyColumnNames( i ), propertyColumnInsertable[i], propertyColumnWriters[i] );
			}
		}

		// add the discriminator
		addDiscriminatorToInsert( insert );

		// delegate already handles PK columns

		if ( getFactory().getSessionFactoryOptions().isCommentsEnabled() ) {
			insert.setComment( "insert " + getEntityName() );
		}

		return insert.toStatementString();
	}
```

`@Repository` Class 의 Proxy Class 는 다음과 같이 만들어 진다.

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

# @Transactional Class

[Spring Annotations Code Tour @Transactional](SpringAnnotationsCodeTour.md#transactional)
