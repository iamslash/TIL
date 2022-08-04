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

# @Transactional Class

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
