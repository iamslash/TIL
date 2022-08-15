# Abstract

Spring Data Jpa 의 code 를 분석해 본다. 

# Materials

* [Spring Data JPA | github](https://github.com/spring-projects/spring-data-jpa)
* [Understanding Spring JPA native query under the hood | stackoverflow](https://stackoverflow.com/questions/58784625/understanding-spring-jpa-native-query-under-the-hood)

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

`EntityManager::persist()` 가 호출되는 부분가지 흐름을 이해하자. 

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

# @Transactional Class

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
