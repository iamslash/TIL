# Abstract

Spring Kafka 를 정리한다. 


# Code Tour of consumer

다음과 같이 listener 를 정의했다고 하자. 

```java

@Component
@Slf4j
public class RetryableKafkaListener {

  @RetryableTopic(
      attempts = "4",
      backoff = @Backoff(delay = 1000, multiplier = 2.0),
      autoCreateTopics = "false",
      topicSuffixingStrategy = TopicSuffixingStrategy.SUFFIX_WITH_INDEX_VALUE)
  @KafkaListener(topics = "orders")
  public void listen(String in, @Header(KafkaHeaders.RECEIVED_TOPIC) String topic) {
    log.info(in + " from " + topic);
    throw new RuntimeException("test");
  }

  @DltHandler
  public void dlt(String in, @Header(KafkaHeaders.RECEIVED_TOPIC) String topic) {
    log.info(in + " from " + topic);
  }
}
```

최초 Kafka 로 부터 message 를 얻어오는 부분은 `org.springframework.kafka.listener.KafkaMessageListenerContainer::run` 의 `pollAndInvoke()` 이다.

```java
public class KafkaMessageListenerContainer<K, V> // NOSONAR line count
		extends AbstractMessageListenerContainer<K, V> {
...

		@Override // NOSONAR complexity
		public void run() {
			ListenerUtils.setLogOnlyMetadata(this.containerProperties.isOnlyLogRecordMetadata());
			publishConsumerStartingEvent();
			this.consumerThread = Thread.currentThread();
			setupSeeks();
			KafkaUtils.setConsumerGroupId(this.consumerGroupId);
			this.count = 0;
			this.last = System.currentTimeMillis();
			initAssignedPartitions();
			publishConsumerStartedEvent();
			Throwable exitThrowable = null;
			while (isRunning()) {
				try {
					pollAndInvoke();
				}
...
		}
...
}
```

`@KafkaListener` annotation 처리는 `org.springframework.kafka.annotation.KafkaListenerAnnotationBeanPostProcessor::postProcessAfterInitialization` 에서 처리한다.

```java
public class KafkaListenerAnnotationBeanPostProcessor<K, V>
		implements BeanPostProcessor, Ordered, BeanFactoryAware, SmartInitializingSingleton {
...
	@Override
	public Object postProcessAfterInitialization(final Object bean, final String beanName) throws BeansException {
		if (!this.nonAnnotatedClasses.contains(bean.getClass())) {
...
			if (annotatedMethods.isEmpty()) {
				this.nonAnnotatedClasses.add(bean.getClass());
				this.logger.trace(() -> "No @KafkaListener annotations found on bean type: " + bean.getClass());
			}
			else {
				// Non-empty set of methods
				for (Map.Entry<Method, Set<KafkaListener>> entry : annotatedMethods.entrySet()) {
					Method method = entry.getKey();
					for (KafkaListener listener : entry.getValue()) {
						processKafkaListener(listener, method, bean, beanName);
					}
				}
				this.logger.debug(() -> annotatedMethods.size() + " @KafkaListener methods processed on bean '"
							+ beanName + "': " + annotatedMethods);
			}
			if (hasClassLevelListeners) {
				processMultiMethodListeners(classLevelListeners, multiMethods, bean, beanName);
			}
		}
		return bean;
	}
...
	protected void processKafkaListener(KafkaListener kafkaListener, Method method, Object bean, String beanName) {
		Method methodToUse = checkProxy(method, bean);
		MethodKafkaListenerEndpoint<K, V> endpoint = new MethodKafkaListenerEndpoint<>();
		endpoint.setMethod(methodToUse);

		if (!processMainAndRetryListeners(kafkaListener, bean, beanName, methodToUse, endpoint)) {
			processListener(endpoint, kafkaListener, bean, beanName);
		}
	}
...
	private boolean processMainAndRetryListeners(KafkaListener kafkaListener, Object bean, String beanName,
												Method methodToUse, MethodKafkaListenerEndpoint<K, V> endpoint) {

		RetryTopicConfiguration retryTopicConfiguration = new RetryTopicConfigurationProvider(this.beanFactory, this.resolver, this.expressionContext)
						.findRetryConfigurationFor(kafkaListener.topics(), methodToUse, bean);

		if (retryTopicConfiguration == null) {
			this.logger.debug("No retry topic configuration found for topics " + Arrays.asList(kafkaListener.topics()));
			return false;
		}

		RetryTopicConfigurer.EndpointProcessor endpointProcessor = endpointToProcess ->
				this.doProcessKafkaListenerAnnotation(endpointToProcess, kafkaListener, bean);

		String beanRef = kafkaListener.beanRef();
		this.listenerScope.addListener(beanRef, bean);

		KafkaListenerContainerFactory<?> factory =
				resolveContainerFactory(kafkaListener, resolve(kafkaListener.containerFactory()), beanName);

		getRetryTopicConfigurer()
				.processMainAndRetryListeners(endpointProcessor, endpoint, retryTopicConfiguration,
						this.registrar, factory, this.defaultContainerFactoryBeanName);

		this.listenerScope.removeListener(beanRef);
		return true;
	}
...        
	public RetryTopicConfiguration findRetryConfigurationFor(String[] topics, Method method, Object bean) {
		RetryableTopic annotation = AnnotationUtils.findAnnotation(method, RetryableTopic.class);
		return annotation != null
				? new RetryableTopicAnnotationProcessor(this.beanFactory, this.resolver, this.expressionContext)
						.processAnnotation(topics, method, annotation, bean)
				: maybeGetFromContext(topics);
	}
...    
}    
```

`@RetryableTopic` annotation 처리는 위의 code 의 `new RetryableTopicAnnotationProcessor(this.beanFactory, this.resolver, this.expressionContext).processAnnotation(topics, method, annotation, bean)` 에서 처리한다.

```java
// org.springframework.kafka.annotation.RetryableTopicAnnotationProcessor
public class RetryableTopicAnnotationProcessor {
...
	public RetryTopicConfiguration processAnnotation(String[] topics, Method method, RetryableTopic annotation,
			Object bean) {

		Long resolvedTimeout = resolveExpressionAsLong(annotation.timeout(), "timeout", false);
		long timeout = RetryTopicConstants.NOT_SET;
		if (resolvedTimeout != null) {
			timeout = resolvedTimeout;
		}
		List<Class<? extends Throwable>> includes = resolveClasses(annotation.include(), annotation.includeNames(),
				"include");
		List<Class<? extends Throwable>> excludes = resolveClasses(annotation.exclude(), annotation.excludeNames(),
				"exclude");
		boolean traverse = false;
		if (StringUtils.hasText(annotation.traversingCauses())) {
			Boolean traverseResolved = resolveExpressionAsBoolean(annotation.traversingCauses(), "traversingCauses");
			if (traverseResolved != null) {
				traverse = traverseResolved;
			}
			else {
				traverse = includes.size() > 0 || excludes.size() > 0;
			}
		}
		return RetryTopicConfigurationBuilder.newInstance()
				.maxAttempts(resolveExpressionAsInteger(annotation.attempts(), "attempts", true))
				.customBackoff(createBackoffFromAnnotation(annotation.backoff(), this.beanFactory))
				.retryTopicSuffix(resolveExpressionAsString(annotation.retryTopicSuffix(), "retryTopicSuffix"))
				.dltSuffix(resolveExpressionAsString(annotation.dltTopicSuffix(), "dltTopicSuffix"))
				.dltHandlerMethod(getDltProcessor(method, bean))
				.includeTopics(Arrays.asList(topics))
				.listenerFactory(annotation.listenerContainerFactory())
				.autoCreateTopics(resolveExpressionAsBoolean(annotation.autoCreateTopics(), "autoCreateTopics"),
						resolveExpressionAsInteger(annotation.numPartitions(), "numPartitions", true),
						resolveExpressionAsShort(annotation.replicationFactor(), "replicationFactor", true))
				.retryOn(includes)
				.notRetryOn(excludes)
				.traversingCauses(traverse)
				.useSingleTopicForFixedDelays(annotation.fixedDelayTopicStrategy())
				.dltProcessingFailureStrategy(annotation.dltStrategy())
				.setTopicSuffixingStrategy(annotation.topicSuffixingStrategy())
				.timeoutAfter(timeout)
				.create(getKafkaTemplate(annotation.kafkaTemplate(), topics));
	}
...
}
```
