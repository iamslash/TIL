- [Summary](#summary)
- [Spring Application Socket Connection Flow](#spring-application-socket-connection-flow)
- [Spring Application HTTP Request Flow](#spring-application-http-request-flow)
- [Spring Application Handler Mapping Flow](#spring-application-handler-mapping-flow)
- [Spring Application View Rendering Flow](#spring-application-view-rendering-flow)
- [Spring Application Exception Rendering Flow](#spring-application-exception-rendering-flow)
- [Spring Application RestController Class HTTP Response Flow](#spring-application-restcontroller-class-http-response-flow)

----

# Summary

* [SPRING MVC REQUEST LIFE CYCLE](https://justforchangesake.wordpress.com/2014/05/07/spring-mvc-request-life-cycle/)

----

다음은 Spring Framework 가 어떻게 HTTP Request 를 처리하는지를 표현한 그림이다. `DispatcherServlet` 의 code 를 가장 먼저 이해하는 것이 좋다.

![](https://justforchangesake.files.wordpress.com/2014/05/spring-request-lifecycle.jpg)

다음과 같은 Component 들을 먼저 이해하는 것이 중요하다.

* **Filter**: `Filter` 는 HTTP Request 를 가장 먼저 처리하는 것이다. Filter 들은 여러개가 chaining 되어 있다. HTTP Request 는 Filter 들을 지나서 `DispatchServlet` 으로 향한다.
* **Dispatcher servlet**: `DispatchServlet` instance 는 HTTP Request URL 에 대응하는 `@Controller Class` 의 `@RequestMapping Method` 를 호출한다. 
* **Common services**: `DispatchServlet` 은 `MultipartResolver, LocaleResolver, ThemeResolver` instance 와 같은 resolver 들을 가지고 있다. 이러한 resolver 들을 Common services 라고 한다. 이 것들의 설정은 `DispatchServlet` 의 부모인 `FrameworkServlet` 의 `private WebApplicationContext webApplicationContext` 에 저장되어 있다. 
* **Handler mapping**: `handlerMappings` 는 다양한 방법으로 HTTP Request 를 `@Controller Class` 의 method 에 매핑해준다. `DispatchServlet` 이 `private List<HandlerMapping> handlerMappings` 를 가지고 있다.
* **Handler interceptor**: `Handler interceptor` 는 모든 `@Controller Class` 의 `@RequestMapping Method` 가 호출되기 전 공통으로 처리되는 것이다.
* **Handler exception resolver**: `Handler exception resolver` 는 Exception 을 처리하는 것이 이다. `DispatchServlet` 은 `private List<HandlerExceptionResolver> handlerExceptionResolvers` 를 가지고 있다.
* **View Resolver**: `View Resolver` 는 HTML Rendering 을 처리하는 것이다. `DispatchServlet` 은 `private List<ViewResolver> viewResolvers` 를 가지고 있다.

# Spring Application Socket Connection Flow

* [스프링부트는 어떻게 다중 유저 요청을 처리할까? (Tomcat9.0 Thread Pool)](https://velog.io/@sihyung92/how-does-springboot-handle-multiple-requests)

Spring Boot 의 embedded tomcat 은 task queue 를 갖고 있다. task queue 에 쌓여진 HTTP Request 를 worker thread 들이 가져가서 처리한다. 다음과 같이 embedded tomcat 의 max thread 등을 설정할 수 있다.

```yaml
server:
  tomcat:
    threads:
      max: 200 
      min-spare: 10 
    max-connections: 8192 
    accept-count: 100 # task queue size
    connection-timeout: 20000 # time out secs
  port: 8080
```

BIO (Blocking I/O) 대신 NIO (NonBlocking I/O) 를 사용한다.

embedded tomcat 은 다음과 같이 생성된다.

```java
// org.springframework.boot.web.embedded.tomcat.TomcatServletWebServerFactory
public class TomcatServletWebServerFactory extends AbstractServletWebServerFactory
		implements ConfigurableTomcatWebServerFactory, ResourceLoaderAware {
...
	@Override
	public WebServer getWebServer(ServletContextInitializer... initializers) {
		if (this.disableMBeanRegistry) {
			Registry.disableRegistry();
		}
		Tomcat tomcat = new Tomcat();
		File baseDir = (this.baseDirectory != null) ? this.baseDirectory : createTempDir("tomcat");
		tomcat.setBaseDir(baseDir.getAbsolutePath());
		Connector connector = new Connector(this.protocol);
		connector.setThrowOnFailure(true);
		tomcat.getService().addConnector(connector);
		customizeConnector(connector);
		tomcat.setConnector(connector);
		tomcat.getHost().setAutoDeploy(false);
		configureEngine(tomcat.getEngine());
		for (Connector additionalConnector : this.additionalTomcatConnectors) {
			tomcat.getService().addConnector(additionalConnector);
		}
		prepareContext(tomcat.getHost(), initializers);
		return getTomcatWebServer(tomcat);
	}
```

embedded tomcat 은 다음과 같이 시작한다.

```java
// org.springframework.boot.web.embedded.tomcat.TomcatWebServer
public class TomcatWebServer implements WebServer {
...
	private void initialize() throws WebServerException {
		logger.info("Tomcat initialized with port(s): " + getPortsDescription(false));
		synchronized (this.monitor) {
			try {
				addInstanceIdToEngineName();

				Context context = findContext();
				context.addLifecycleListener((event) -> {
					if (context.equals(event.getSource()) && Lifecycle.START_EVENT.equals(event.getType())) {
						// Remove service connectors so that protocol binding doesn't
						// happen when the service is started.
						removeServiceConnectors();
					}
				});

				// Start the server to trigger initialization listeners
				this.tomcat.start();
...
```

main thread 는 `startDaemonAwaitThread()` 에서 embedded tomcat 을 wait 한다.

```java
// org.springframework.boot.web.embedded.tomcat.TomcatWebServer
public class TomcatWebServer implements WebServer {
...
	private void initialize() throws WebServerException {
		logger.info("Tomcat initialized with port(s): " + getPortsDescription(false));
		synchronized (this.monitor) {
			try {
				addInstanceIdToEngineName();

				Context context = findContext();
				context.addLifecycleListener((event) -> {
					if (context.equals(event.getSource()) && Lifecycle.START_EVENT.equals(event.getType())) {
						// Remove service connectors so that protocol binding doesn't
						// happen when the service is started.
						removeServiceConnectors();
					}
				});

				// Start the server to trigger initialization listeners
				this.tomcat.start();

				// We can re-throw failure exception directly in the main thread
				rethrowDeferredStartupExceptions();

				try {
					ContextBindings.bindClassLoader(context, context.getNamingToken(), getClass().getClassLoader());
				}
				catch (NamingException ex) {
					// Naming is not enabled. Continue
				}

				// Unlike Jetty, all Tomcat threads are daemon threads. We create a
				// blocking non-daemon to stop immediate shutdown
				startDaemonAwaitThread();
...       

public class TomcatWebServer implements WebServer {
...
	private void startDaemonAwaitThread() {
		Thread awaitThread = new Thread("container-" + (containerCounter.get())) {

			@Override
			public void run() {
				TomcatWebServer.this.tomcat.getServer().await();
			}

		};
		awaitThread.setContextClassLoader(getClass().getClassLoader());
		awaitThread.setDaemon(false);
		awaitThread.start();
	}
```

다음은 `TaskQueue` 와 `TaskThreadFactory` 를 생성하는 부분이다. `TaskThreadFactory` 에
의해 생성되는 thread 의 이름은 `http-nio-8080-*` 과 같다.

```java
// org.apache.tomcat.util.net.AbstractEndpoint
public abstract class AbstractEndpoint<S,U> {
...
    public void createExecutor() {
        internalExecutor = true;
        TaskQueue taskqueue = new TaskQueue();
        TaskThreadFactory tf = new TaskThreadFactory(getName() + "-exec-", daemon, getThreadPriority());
        executor = new ThreadPoolExecutor(getMinSpareThreads(), getMaxThreads(), 60, TimeUnit.SECONDS,taskqueue, tf);
        taskqueue.setParent( (ThreadPoolExecutor) executor);
    }
```

다음은 Client 가 접속하면 Server 에서 `Runnable` instance 을 만들어 `TaskQueue` 에
삽입하는 부분이다. (`workQueue.offer(command)`)

```java
// java.util.concurrent.ThreadPoolExecutor
public class ThreadPoolExecutor extends java.util.concurrent.ThreadPoolExecutor {
...    
    public void execute(Runnable command) {
        if (command == null)
            throw new NullPointerException();
        int c = ctl.get();
        if (workerCountOf(c) < corePoolSize) {
            if (addWorker(command, true))
                return;
            c = ctl.get();
        }
        if (isRunning(c) && workQueue.offer(command)) {
            int recheck = ctl.get();
            if (! isRunning(recheck) && remove(command))
                reject(command);
            else if (workerCountOf(recheck) == 0)
                addWorker(null, false);
        }
        else if (!addWorker(command, false))
            reject(command);
    }
```

다음은 `Worker` instance 가 `TaskQueue` 에서 Socket Connection 을 가져오는 부분이다. (`workQueue.poll()`)

```java
// java.util.concurrent.ThreadPoolExecutor
    private final class Worker
        extends AbstractQueuedSynchronizer
        implements Runnable
    {
...
        public void run() {
            runWorker(this);
        }
...
    }     

public class ThreadPoolExecutor extends AbstractExecutorService {
...    
    final void runWorker(Worker w) {
        Thread wt = Thread.currentThread();
        Runnable task = w.firstTask;
        w.firstTask = null;
        w.unlock(); // allow interrupts
        boolean completedAbruptly = true;
        try {
            while (task != null || (task = getTask()) != null) {   

public class ThreadPoolExecutor extends AbstractExecutorService {
...    
    private Runnable getTask() {
        boolean timedOut = false; // Did the last poll() time out?

        for (;;) {
            int c = ctl.get();

            // Check if queue empty only if necessary.
            if (runStateAtLeast(c, SHUTDOWN)
                && (runStateAtLeast(c, STOP) || workQueue.isEmpty())) {
                decrementWorkerCount();
                return null;
            }

            int wc = workerCountOf(c);

            // Are workers subject to culling?
            boolean timed = allowCoreThreadTimeOut || wc > corePoolSize;

            if ((wc > maximumPoolSize || (timed && timedOut))
                && (wc > 1 || workQueue.isEmpty())) {
                if (compareAndDecrementWorkerCount(c))
                    return null;
                continue;
            }

            try {
                Runnable r = timed ?
                    workQueue.poll(keepAliveTime, TimeUnit.NANOSECONDS) :
                    workQueue.take();
```

`Worker` instance 는 다음과 같이 `ThreadPoolExecutor::addWorker()` 를 호출하여 생성한다.

```java
// java.util.concurrent.ThreadPoolExecutor
public class ThreadPoolExecutor extends AbstractExecutorService {
...    
    public void execute(Runnable command) {
        if (command == null)
            throw new NullPointerException();
        int c = ctl.get();
        if (workerCountOf(c) < corePoolSize) {
            if (addWorker(command, true))
                return;
            c = ctl.get();
        }
        if (isRunning(c) && workQueue.offer(command)) {
            int recheck = ctl.get();
            if (! isRunning(recheck) && remove(command))
                reject(command);
            else if (workerCountOf(recheck) == 0)
                addWorker(null, false);
        }
        else if (!addWorker(command, false))
            reject(command);
    }

public class ThreadPoolExecutor extends AbstractExecutorService {
...  
    private boolean addWorker(Runnable firstTask, boolean core) {
        retry:
        for (int c = ctl.get();;) {
            // Check if queue empty only if necessary.
            if (runStateAtLeast(c, SHUTDOWN)
                && (runStateAtLeast(c, STOP)
                    || firstTask != null
                    || workQueue.isEmpty()))
                return false;

            for (;;) {
                if (workerCountOf(c)
                    >= ((core ? corePoolSize : maximumPoolSize) & COUNT_MASK))
                    return false;
                if (compareAndIncrementWorkerCount(c))
                    break retry;
                c = ctl.get();  // Re-read ctl
                if (runStateAtLeast(c, SHUTDOWN))
                    continue retry;
                // else CAS failed due to workerCount change; retry inner loop
            }
        }

        boolean workerStarted = false;
        boolean workerAdded = false;
        Worker w = null;
        try {
            w = new Worker(firstTask);
            final Thread t = w.thread;
            if (t != null) {
                final ReentrantLock mainLock = this.mainLock;
                mainLock.lock();
                try {
                    // Recheck while holding lock.
                    // Back out on ThreadFactory failure or if
                    // shut down before lock acquired.
                    int c = ctl.get();

                    if (isRunning(c) ||
                        (runStateLessThan(c, STOP) && firstTask == null)) {
                        if (t.getState() != Thread.State.NEW)
                            throw new IllegalThreadStateException();
                        workers.add(w);
                        workerAdded = true;
                        int s = workers.size();
                        if (s > largestPoolSize)
                            largestPoolSize = s;
                    }
                } finally {
                    mainLock.unlock();
                }
                if (workerAdded) {
                    t.start();
                    workerStarted = true;
                }
            }
        } finally {
            if (! workerStarted)
                addWorkerFailed(w);
        }
        return workerStarted;
    }
...        
```

# Spring Application HTTP Request Flow

`org.apache.tomcat.embed:tomcat-embed-core` 의 `SocketProcessorBase<S>` 는 HTTP Request 를 처리하는 `Concrete Runnable Class` 이다. 

```java
// org.apache.tomcat.util.net.SocketProcessorBase
public abstract class SocketProcessorBase<S> implements Runnable {
...
    @Override
    public final void run() {
        synchronized (socketWrapper) {
            // It is possible that processing may be triggered for read and
            // write at the same time. The sync above makes sure that processing
            // does not occur in parallel. The test below ensures that if the
            // first event to be processed results in the socket being closed,
            // the subsequent events are not processed.
            if (socketWrapper.isClosed()) {
                return;
            }
            doRun();
        }
    }
...
}

// org.apache.tomcat.util.net.NioEndpoint
public class NioEndpoint extends AbstractJsseEndpoint<NioChannel,SocketChannel> {
...
    protected class SocketProcessor extends SocketProcessorBase<NioChannel> {
...
        @Override
        protected void doRun() {
...
                if (handshake == 0) {
                    SocketState state = SocketState.OPEN;
                    // Process the request from this socket
                    if (event == null) {
                        state = getHandler().process(socketWrapper, SocketEvent.OPEN_READ);
                    } else {
                        state = getHandler().process(socketWrapper, event);
...			
		}
...		
	}
...
}

// org.apache.coyote.AbstractProtocol
public abstract class AbstractProtocol<S> implements ProtocolHandler,
        MBeanRegistration {
...
                SocketState state = SocketState.CLOSED;
                do {
                    state = processor.process(wrapper, status);
...
}

// org.apache.coyote.AbstractProcessorLight
public abstract class AbstractProcessorLight implements Processor {
...
    @Override
    public SocketState process(SocketWrapperBase<?> socketWrapper, SocketEvent status)
            throws IOException {

        SocketState state = SocketState.CLOSED;
        Iterator<DispatchType> dispatches = null;
        do {
            if (dispatches != null) {
                DispatchType nextDispatch = dispatches.next();
                if (getLog().isDebugEnabled()) {
                    getLog().debug("Processing dispatch type: [" + nextDispatch + "]");
                }
                state = dispatch(nextDispatch.getSocketStatus());
                if (!dispatches.hasNext()) {
                    state = checkForPipelinedData(state, socketWrapper);
                }
            } else if (status == SocketEvent.DISCONNECT) {
                // Do nothing here, just wait for it to get recycled
            } else if (isAsync() || isUpgrade() || state == SocketState.ASYNC_END) {
                state = dispatch(status);
                state = checkForPipelinedData(state, socketWrapper);
            } else if (status == SocketEvent.OPEN_WRITE) {
                // Extra write event likely after async, ignore
                state = SocketState.LONG;
            } else if (status == SocketEvent.OPEN_READ) {
                state = service(socketWrapper);
...
}

// org.apache.coyote.http11.Http11Processor
public class Http11Processor extends AbstractProcessor {
...
    @Override
    public SocketState service(SocketWrapperBase<?> socketWrapper)
        throws IOException {
...
            // Process the request in the adapter
            if (getErrorState().isIoAllowed()) {
                try {
                    rp.setStage(org.apache.coyote.Constants.STAGE_SERVICE);
                    getAdapter().service(request, response);
...
}

// org.apache.catalina.connector.CoyoteAdapter
public class CoyoteAdapter implements Adapter {
...
    @Override
    public void service(org.apache.coyote.Request req, org.apache.coyote.Response res) throws Exception {
...
        try {
            // Parse and set Catalina and configuration specific
            // request parameters
            postParseSuccess = postParseRequest(req, request, res, response);
            if (postParseSuccess) {
                //check valves if we support async
                request.setAsyncSupported(
                        connector.getService().getContainer().getPipeline().isAsyncSupported());
                // Calling the container
                connector.getService().getContainer().getPipeline().getFirst().invoke(request, response);
            }
...
	}
...
}

// org.apache.catalina.core.StandardEngineValve
final class StandardEngineValve extends ValveBase {
...
    @Override
    public final void invoke(Request request, Response response)
        throws IOException, ServletException {

        // Select the Host to be used for this Request
        Host host = request.getHost();
        if (host == null) {
            // HTTP 0.9 or HTTP 1.0 request without a host when no default host
            // is defined. This is handled by the CoyoteAdapter.
            return;
        }
        if (request.isAsyncSupported()) {
            request.setAsyncSupported(host.getPipeline().isAsyncSupported());
        }

        // Ask this Host to process this request
        host.getPipeline().getFirst().invoke(request, response);
    }
...
}

// org.apache.catalina.valves.ErrorReportValve
public class ErrorReportValve extends ValveBase {
...
    @Override
    public void invoke(Request request, Response response) throws IOException, ServletException {

        // Perform the request
        getNext().invoke(request, response);
...
}

// org.apache.catalina.core.StandardHostValve
final class StandardHostValve extends ValveBase {
...
    @Override
    public final void invoke(Request request, Response response)
        throws IOException, ServletException {

        // Select the Context to be used for this Request
        Context context = request.getContext();
        if (context == null) {
            return;
        }

        if (request.isAsyncSupported()) {
            request.setAsyncSupported(context.getPipeline().isAsyncSupported());
        }

        boolean asyncAtStart = request.isAsync();

        try {
            context.bind(Globals.IS_SECURITY_ENABLED, MY_CLASSLOADER);

            if (!asyncAtStart && !context.fireRequestInitEvent(request.getRequest())) {
                // Don't fire listeners during async processing (the listener
                // fired for the request that called startAsync()).
                // If a request init listener throws an exception, the request
                // is aborted.
                return;
            }

            // Ask this Context to process this request. Requests that are
            // already in error must have been routed here to check for
            // application defined error pages so DO NOT forward them to the the
            // application for processing.
            try {
                if (!response.isErrorReportRequired()) {
                    context.getPipeline().getFirst().invoke(request, response);
...
}

// org.apache.catalina.authenticator.AuthenticatorBase
public abstract class AuthenticatorBase extends ValveBase
        implements Authenticator, RegistrationListener {
...
    @Override
    public void invoke(Request request, Response response) throws IOException, ServletException {

        if (log.isDebugEnabled()) {
            log.debug("Security checking request " + request.getMethod() + " " +
                    request.getRequestURI());
        }

        // Have we got a cached authenticated Principal to record?
        if (cache) {
            Principal principal = request.getUserPrincipal();
            if (principal == null) {
                Session session = request.getSessionInternal(false);
                if (session != null) {
                    principal = session.getPrincipal();
                    if (principal != null) {
                        if (log.isDebugEnabled()) {
                            log.debug("We have cached auth type " + session.getAuthType() +
                                    " for principal " + principal);
                        }
                        request.setAuthType(session.getAuthType());
                        request.setUserPrincipal(principal);
                    }
                }
            }
        }

        boolean authRequired = isContinuationRequired(request);

        Realm realm = this.context.getRealm();
        // Is this request URI subject to a security constraint?
        SecurityConstraint[] constraints = realm.findSecurityConstraints(request, this.context);

        AuthConfigProvider jaspicProvider = getJaspicProvider();
        if (jaspicProvider != null) {
            authRequired = true;
        }

        if (constraints == null && !context.getPreemptiveAuthentication() && !authRequired) {
            if (log.isDebugEnabled()) {
                log.debug("Not subject to any constraint");
            }
            getNext().invoke(request, response);
...
}

// org.apache.catalina.core.StandardContextValve
final class StandardContextValve extends ValveBase {
...
    @Override
    public final void invoke(Request request, Response response)
        throws IOException, ServletException {
...
        if (request.isAsyncSupported()) {
            request.setAsyncSupported(wrapper.getPipeline().isAsyncSupported());
        }
        wrapper.getPipeline().getFirst().invoke(request, response);
...
}

// org.apache.catalina.core.StandardWrapperValve
// Servlet instance 가 생성된다.
// ApplicationFilterChain instance 가 생성된다.
// 첫번째 ApplicationFilterChain instance 의 doFilter 를 호출한다.
final class StandardWrapperValve
    extends ValveBase {
...
    @Override
    public final void invoke(Request request, Response response)
        throws IOException, ServletException {

        // Initialize local variables we may need
        boolean unavailable = false;
        Throwable throwable = null;
        // This should be a Request attribute...
        long t1=System.currentTimeMillis();
        requestCount.incrementAndGet();
        StandardWrapper wrapper = (StandardWrapper) getContainer();
        Servlet servlet = null;
        Context context = (Context) wrapper.getParent();

        // Check for the application being marked unavailable
        if (!context.getState().isAvailable()) {
            response.sendError(HttpServletResponse.SC_SERVICE_UNAVAILABLE,
                           sm.getString("standardContext.isUnavailable"));
            unavailable = true;
        }

        // Check for the servlet being marked unavailable
        if (!unavailable && wrapper.isUnavailable()) {
            container.getLogger().info(sm.getString("standardWrapper.isUnavailable",
                    wrapper.getName()));
            long available = wrapper.getAvailable();
            if ((available > 0L) && (available < Long.MAX_VALUE)) {
                response.setDateHeader("Retry-After", available);
                response.sendError(HttpServletResponse.SC_SERVICE_UNAVAILABLE,
                        sm.getString("standardWrapper.isUnavailable",
                                wrapper.getName()));
            } else if (available == Long.MAX_VALUE) {
                response.sendError(HttpServletResponse.SC_NOT_FOUND,
                        sm.getString("standardWrapper.notFound",
                                wrapper.getName()));
            }
            unavailable = true;
        }

        // Allocate a servlet instance to process this request
        try {
            if (!unavailable) {
                servlet = wrapper.allocate();
            }
        } catch (UnavailableException e) {
            container.getLogger().error(
                    sm.getString("standardWrapper.allocateException",
                            wrapper.getName()), e);
            long available = wrapper.getAvailable();
            if ((available > 0L) && (available < Long.MAX_VALUE)) {
                response.setDateHeader("Retry-After", available);
                response.sendError(HttpServletResponse.SC_SERVICE_UNAVAILABLE,
                           sm.getString("standardWrapper.isUnavailable",
                                        wrapper.getName()));
            } else if (available == Long.MAX_VALUE) {
                response.sendError(HttpServletResponse.SC_NOT_FOUND,
                           sm.getString("standardWrapper.notFound",
                                        wrapper.getName()));
            }
        } catch (ServletException e) {
            container.getLogger().error(sm.getString("standardWrapper.allocateException",
                             wrapper.getName()), StandardWrapper.getRootCause(e));
            throwable = e;
            exception(request, response, e);
        } catch (Throwable e) {
            ExceptionUtils.handleThrowable(e);
            container.getLogger().error(sm.getString("standardWrapper.allocateException",
                             wrapper.getName()), e);
            throwable = e;
            exception(request, response, e);
            servlet = null;
        }

        MessageBytes requestPathMB = request.getRequestPathMB();
        DispatcherType dispatcherType = DispatcherType.REQUEST;
        if (request.getDispatcherType()==DispatcherType.ASYNC) dispatcherType = DispatcherType.ASYNC;
        request.setAttribute(Globals.DISPATCHER_TYPE_ATTR,dispatcherType);
        request.setAttribute(Globals.DISPATCHER_REQUEST_PATH_ATTR,
                requestPathMB);
        // Create the filter chain for this request
        ApplicationFilterChain filterChain =
                ApplicationFilterFactory.createFilterChain(request, wrapper, servlet);

        // Call the filter chain for this request
        // NOTE: This also calls the servlet's service() method
        Container container = this.container;
        try {
            if ((servlet != null) && (filterChain != null)) {
                // Swallow output if needed
                if (context.getSwallowOutput()) {
                    try {
                        SystemLogHandler.startCapture();
                        if (request.isAsyncDispatching()) {
                            request.getAsyncContextInternal().doInternalDispatch();
                        } else {
                            filterChain.doFilter(request.getRequest(),
                                    response.getResponse());
                        }
                    } finally {
                        String log = SystemLogHandler.stopCapture();
                        if (log != null && log.length() > 0) {
                            context.getLogger().info(log);
                        }
                    }
                } else {
                    if (request.isAsyncDispatching()) {
                        request.getAsyncContextInternal().doInternalDispatch();
                    } else {
                        filterChain.doFilter
                            (request.getRequest(), response.getResponse());
...
}

// org.apache.catalina.core.ApplicationFilterChain
public final class ApplicationFilterChain implements FilterChain {
...
    @Override
    public void doFilter(ServletRequest request, ServletResponse response)
        throws IOException, ServletException {

        if( Globals.IS_SECURITY_ENABLED ) {
            final ServletRequest req = request;
            final ServletResponse res = response;
            try {
                java.security.AccessController.doPrivileged(
                    new java.security.PrivilegedExceptionAction<Void>() {
                        @Override
                        public Void run()
                            throws ServletException, IOException {
                            internalDoFilter(req,res);
                            return null;
                        }
                    }
                );
            } catch( PrivilegedActionException pe) {
                Exception e = pe.getException();
                if (e instanceof ServletException)
                    throw (ServletException) e;
                else if (e instanceof IOException)
                    throw (IOException) e;
                else if (e instanceof RuntimeException)
                    throw (RuntimeException) e;
                else
                    throw new ServletException(e.getMessage(), e);
            }
        } else {
            internalDoFilter(request,response);
        }
    }
...
    private void internalDoFilter(ServletRequest request,
                                  ServletResponse response)
        throws IOException, ServletException {

        // Call the next filter if there is one
        if (pos < n) {
            ApplicationFilterConfig filterConfig = filters[pos++];
            try {
                Filter filter = filterConfig.getFilter();

                if (request.isAsyncSupported() && "false".equalsIgnoreCase(
                        filterConfig.getFilterDef().getAsyncSupported())) {
                    request.setAttribute(Globals.ASYNC_SUPPORTED_ATTR, Boolean.FALSE);
                }
                if( Globals.IS_SECURITY_ENABLED ) {
                    final ServletRequest req = request;
                    final ServletResponse res = response;
                    Principal principal =
                        ((HttpServletRequest) req).getUserPrincipal();

                    Object[] args = new Object[]{req, res, this};
                    SecurityUtil.doAsPrivilege ("doFilter", filter, classType, args, principal);
                } else {
                    filter.doFilter(request, response, this);
                }
...
}

// org.springframework.web.filter.OncePerRequestFilter
public abstract class OncePerRequestFilter extends GenericFilterBean {
...
	@Override
	public final void doFilter(ServletRequest request, ServletResponse response, FilterChain filterChain)
			throws ServletException, IOException {

		if (!(request instanceof HttpServletRequest) || !(response instanceof HttpServletResponse)) {
			throw new ServletException("OncePerRequestFilter just supports HTTP requests");
		}
		HttpServletRequest httpRequest = (HttpServletRequest) request;
		HttpServletResponse httpResponse = (HttpServletResponse) response;

		String alreadyFilteredAttributeName = getAlreadyFilteredAttributeName();
		boolean hasAlreadyFilteredAttribute = request.getAttribute(alreadyFilteredAttributeName) != null;

		if (skipDispatch(httpRequest) || shouldNotFilter(httpRequest)) {

			// Proceed without invoking this filter...
			filterChain.doFilter(request, response);
		}
		else if (hasAlreadyFilteredAttribute) {

			if (DispatcherType.ERROR.equals(request.getDispatcherType())) {
				doFilterNestedErrorDispatch(httpRequest, httpResponse, filterChain);
				return;
			}

			// Proceed without invoking this filter...
			filterChain.doFilter(request, response);
		}
		else {
			// Do invoke this filter...
			request.setAttribute(alreadyFilteredAttributeName, Boolean.TRUE);
			try {
				doFilterInternal(httpRequest, httpResponse, filterChain);
...
	}
...
...	
}

// org.springframework.web.filter.CharacterEncodingFilter
public class CharacterEncodingFilter extends OncePerRequestFilter {
...
	@Override
	protected void doFilterInternal(
			HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
			throws ServletException, IOException {

		String encoding = getEncoding();
		if (encoding != null) {
			if (isForceRequestEncoding() || request.getCharacterEncoding() == null) {
				request.setCharacterEncoding(encoding);
			}
			if (isForceResponseEncoding()) {
				response.setCharacterEncoding(encoding);
			}
		}
		filterChain.doFilter(request, response);
	}
...
}

// org.apache.catalina.core.ApplicationFilterChain
public final class ApplicationFilterChain implements FilterChain {
...
    @Override
    public void doFilter(ServletRequest request, ServletResponse response)
        throws IOException, ServletException {

        if( Globals.IS_SECURITY_ENABLED ) {
            final ServletRequest req = request;
            final ServletResponse res = response;
            try {
                java.security.AccessController.doPrivileged(
                    new java.security.PrivilegedExceptionAction<Void>() {
                        @Override
                        public Void run()
                            throws ServletException, IOException {
                            internalDoFilter(req,res);
                            return null;
                        }
                    }
                );
            } catch( PrivilegedActionException pe) {
                Exception e = pe.getException();
                if (e instanceof ServletException)
                    throw (ServletException) e;
                else if (e instanceof IOException)
                    throw (IOException) e;
                else if (e instanceof RuntimeException)
                    throw (RuntimeException) e;
                else
                    throw new ServletException(e.getMessage(), e);
            }
        } else {
            internalDoFilter(request,response);
        }
    }
    private void internalDoFilter(ServletRequest request,
                                  ServletResponse response)
        throws IOException, ServletException {

        // Call the next filter if there is one
        if (pos < n) {
            ApplicationFilterConfig filterConfig = filters[pos++];
            try {
                Filter filter = filterConfig.getFilter();

                if (request.isAsyncSupported() && "false".equalsIgnoreCase(
                        filterConfig.getFilterDef().getAsyncSupported())) {
                    request.setAttribute(Globals.ASYNC_SUPPORTED_ATTR, Boolean.FALSE);
                }
                if( Globals.IS_SECURITY_ENABLED ) {
                    final ServletRequest req = request;
                    final ServletResponse res = response;
                    Principal principal =
                        ((HttpServletRequest) req).getUserPrincipal();

                    Object[] args = new Object[]{req, res, this};
                    SecurityUtil.doAsPrivilege ("doFilter", filter, classType, args, principal);
                } else {
                    filter.doFilter(request, response, this);
...
}

// org.springframework.web.filter.OncePerRequestFilter
public abstract class OncePerRequestFilter extends GenericFilterBean {
...
	@Override
	public final void doFilter(ServletRequest request, ServletResponse response, FilterChain filterChain)
			throws ServletException, IOException {

		if (!(request instanceof HttpServletRequest) || !(response instanceof HttpServletResponse)) {
			throw new ServletException("OncePerRequestFilter just supports HTTP requests");
		}
		HttpServletRequest httpRequest = (HttpServletRequest) request;
		HttpServletResponse httpResponse = (HttpServletResponse) response;

		String alreadyFilteredAttributeName = getAlreadyFilteredAttributeName();
		boolean hasAlreadyFilteredAttribute = request.getAttribute(alreadyFilteredAttributeName) != null;

		if (skipDispatch(httpRequest) || shouldNotFilter(httpRequest)) {

			// Proceed without invoking this filter...
			filterChain.doFilter(request, response);
		}
		else if (hasAlreadyFilteredAttribute) {

			if (DispatcherType.ERROR.equals(request.getDispatcherType())) {
				doFilterNestedErrorDispatch(httpRequest, httpResponse, filterChain);
				return;
			}

			// Proceed without invoking this filter...
			filterChain.doFilter(request, response);
		}
		else {
			// Do invoke this filter...
			request.setAttribute(alreadyFilteredAttributeName, Boolean.TRUE);
			try {
				doFilterInternal(httpRequest, httpResponse, filterChain);
...
	}
}

// org.springframework.web.filter.FormContentFilter
public class FormContentFilter extends OncePerRequestFilter {
...
	@Override
	protected void doFilterInternal(
			HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
			throws ServletException, IOException {

		MultiValueMap<String, String> params = parseIfNecessary(request);
		if (!CollectionUtils.isEmpty(params)) {
			filterChain.doFilter(new FormContentRequestWrapper(request, params), response);
		}
		else {
			filterChain.doFilter(request, response);
		}
	}
...
}

// org.apache.catalina.core.ApplicationFilterChain
public final class ApplicationFilterChain implements FilterChain {
...
    @Override
    public void doFilter(ServletRequest request, ServletResponse response)
        throws IOException, ServletException {

        if( Globals.IS_SECURITY_ENABLED ) {
            final ServletRequest req = request;
            final ServletResponse res = response;
            try {
                java.security.AccessController.doPrivileged(
                    new java.security.PrivilegedExceptionAction<Void>() {
                        @Override
                        public Void run()
                            throws ServletException, IOException {
                            internalDoFilter(req,res);
                            return null;
                        }
                    }
                );
            } catch( PrivilegedActionException pe) {
                Exception e = pe.getException();
                if (e instanceof ServletException)
                    throw (ServletException) e;
                else if (e instanceof IOException)
                    throw (IOException) e;
                else if (e instanceof RuntimeException)
                    throw (RuntimeException) e;
                else
                    throw new ServletException(e.getMessage(), e);
            }
        } else {
            internalDoFilter(request,response);
        }
    }
    private void internalDoFilter(ServletRequest request,
                                  ServletResponse response)
        throws IOException, ServletException {

        // Call the next filter if there is one
        if (pos < n) {
            ApplicationFilterConfig filterConfig = filters[pos++];
            try {
                Filter filter = filterConfig.getFilter();

                if (request.isAsyncSupported() && "false".equalsIgnoreCase(
                        filterConfig.getFilterDef().getAsyncSupported())) {
                    request.setAttribute(Globals.ASYNC_SUPPORTED_ATTR, Boolean.FALSE);
                }
                if( Globals.IS_SECURITY_ENABLED ) {
                    final ServletRequest req = request;
                    final ServletResponse res = response;
                    Principal principal =
                        ((HttpServletRequest) req).getUserPrincipal();

                    Object[] args = new Object[]{req, res, this};
                    SecurityUtil.doAsPrivilege ("doFilter", filter, classType, args, principal);
                } else {
                    filter.doFilter(request, response, this);
...
}


// org.springframework.web.filter.OncePerRequestFilter
public abstract class OncePerRequestFilter extends GenericFilterBean {
...
	@Override
	public final void doFilter(ServletRequest request, ServletResponse response, FilterChain filterChain)
			throws ServletException, IOException {

		if (!(request instanceof HttpServletRequest) || !(response instanceof HttpServletResponse)) {
			throw new ServletException("OncePerRequestFilter just supports HTTP requests");
		}
		HttpServletRequest httpRequest = (HttpServletRequest) request;
		HttpServletResponse httpResponse = (HttpServletResponse) response;

		String alreadyFilteredAttributeName = getAlreadyFilteredAttributeName();
		boolean hasAlreadyFilteredAttribute = request.getAttribute(alreadyFilteredAttributeName) != null;

		if (skipDispatch(httpRequest) || shouldNotFilter(httpRequest)) {

			// Proceed without invoking this filter...
			filterChain.doFilter(request, response);
		}
		else if (hasAlreadyFilteredAttribute) {

			if (DispatcherType.ERROR.equals(request.getDispatcherType())) {
				doFilterNestedErrorDispatch(httpRequest, httpResponse, filterChain);
				return;
			}

			// Proceed without invoking this filter...
			filterChain.doFilter(request, response);
		}
		else {
			// Do invoke this filter...
			request.setAttribute(alreadyFilteredAttributeName, Boolean.TRUE);
			try {
				doFilterInternal(httpRequest, httpResponse, filterChain);
...
	}
}

// org.springframework.web.filter.RequestContextFilter
public class RequestContextFilter extends OncePerRequestFilter {
...
	@Override
	protected void doFilterInternal(
			HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
			throws ServletException, IOException {

		ServletRequestAttributes attributes = new ServletRequestAttributes(request, response);
		initContextHolders(request, attributes);

		try {
			filterChain.doFilter(request, response);
...
}


// org.apache.catalina.core.ApplicationFilterChain
public final class ApplicationFilterChain implements FilterChain {
...
    @Override
    public void doFilter(ServletRequest request, ServletResponse response)
        throws IOException, ServletException {

        if( Globals.IS_SECURITY_ENABLED ) {
            final ServletRequest req = request;
            final ServletResponse res = response;
            try {
                java.security.AccessController.doPrivileged(
                    new java.security.PrivilegedExceptionAction<Void>() {
                        @Override
                        public Void run()
                            throws ServletException, IOException {
                            internalDoFilter(req,res);
                            return null;
                        }
                    }
                );
            } catch( PrivilegedActionException pe) {
                Exception e = pe.getException();
                if (e instanceof ServletException)
                    throw (ServletException) e;
                else if (e instanceof IOException)
                    throw (IOException) e;
                else if (e instanceof RuntimeException)
                    throw (RuntimeException) e;
                else
                    throw new ServletException(e.getMessage(), e);
            }
        } else {
            internalDoFilter(request,response);
        }
    }
    private void internalDoFilter(ServletRequest request,
                                  ServletResponse response)
        throws IOException, ServletException {

        // Call the next filter if there is one
        if (pos < n) {
            ApplicationFilterConfig filterConfig = filters[pos++];
            try {
                Filter filter = filterConfig.getFilter();

                if (request.isAsyncSupported() && "false".equalsIgnoreCase(
                        filterConfig.getFilterDef().getAsyncSupported())) {
                    request.setAttribute(Globals.ASYNC_SUPPORTED_ATTR, Boolean.FALSE);
                }
                if( Globals.IS_SECURITY_ENABLED ) {
                    final ServletRequest req = request;
                    final ServletResponse res = response;
                    Principal principal =
                        ((HttpServletRequest) req).getUserPrincipal();

                    Object[] args = new Object[]{req, res, this};
                    SecurityUtil.doAsPrivilege ("doFilter", filter, classType, args, principal);
                } else {
                    filter.doFilter(request, response, this);
...
}


// org.springframework.web.filter.OncePerRequestFilter
public abstract class OncePerRequestFilter extends GenericFilterBean {
...
	@Override
	public final void doFilter(ServletRequest request, ServletResponse response, FilterChain filterChain)
			throws ServletException, IOException {

		if (!(request instanceof HttpServletRequest) || !(response instanceof HttpServletResponse)) {
			throw new ServletException("OncePerRequestFilter just supports HTTP requests");
		}
		HttpServletRequest httpRequest = (HttpServletRequest) request;
		HttpServletResponse httpResponse = (HttpServletResponse) response;

		String alreadyFilteredAttributeName = getAlreadyFilteredAttributeName();
		boolean hasAlreadyFilteredAttribute = request.getAttribute(alreadyFilteredAttributeName) != null;

		if (skipDispatch(httpRequest) || shouldNotFilter(httpRequest)) {

			// Proceed without invoking this filter...
			filterChain.doFilter(request, response);
		}
		else if (hasAlreadyFilteredAttribute) {

			if (DispatcherType.ERROR.equals(request.getDispatcherType())) {
				doFilterNestedErrorDispatch(httpRequest, httpResponse, filterChain);
				return;
			}

			// Proceed without invoking this filter...
			filterChain.doFilter(request, response);
		}
		else {
			// Do invoke this filter...
			request.setAttribute(alreadyFilteredAttributeName, Boolean.TRUE);
			try {
				doFilterInternal(httpRequest, httpResponse, filterChain);
...
	}
}

// org.apache.tomcat.websocket.server.WsFilter
public class WsFilter extends GenericFilter {
...

    @Override
    public void doFilter(ServletRequest request, ServletResponse response,
            FilterChain chain) throws IOException, ServletException {

        // This filter only needs to handle WebSocket upgrade requests
        if (!sc.areEndpointsRegistered() ||
                !UpgradeUtil.isWebSocketUpgradeRequest(request, response)) {
            chain.doFilter(request, response);
...
}

// org.apache.catalina.core.ApplicationFilterChain
// 드디어 Server instance 의 service method 를 호출한다.
public final class ApplicationFilterChain implements FilterChain {
...
    @Override
    public void doFilter(ServletRequest request, ServletResponse response)
        throws IOException, ServletException {

        if( Globals.IS_SECURITY_ENABLED ) {
            final ServletRequest req = request;
            final ServletResponse res = response;
            try {
                java.security.AccessController.doPrivileged(
                    new java.security.PrivilegedExceptionAction<Void>() {
                        @Override
                        public Void run()
                            throws ServletException, IOException {
                            internalDoFilter(req,res);
                            return null;
                        }
                    }
                );
            } catch( PrivilegedActionException pe) {
                Exception e = pe.getException();
                if (e instanceof ServletException)
                    throw (ServletException) e;
                else if (e instanceof IOException)
                    throw (IOException) e;
                else if (e instanceof RuntimeException)
                    throw (RuntimeException) e;
                else
                    throw new ServletException(e.getMessage(), e);
            }
        } else {
            internalDoFilter(request,response);
        }
    }
...

    private void internalDoFilter(ServletRequest request,
                                  ServletResponse response)
        throws IOException, ServletException {
...
        // We fell off the end of the chain -- call the servlet instance
        try {
            if (ApplicationDispatcher.WRAP_SAME_OBJECT) {
                lastServicedRequest.set(request);
                lastServicedResponse.set(response);
            }

            if (request.isAsyncSupported() && !servletSupportsAsync) {
                request.setAttribute(Globals.ASYNC_SUPPORTED_ATTR,
                        Boolean.FALSE);
            }
            // Use potentially wrapped request from this point
            if ((request instanceof HttpServletRequest) &&
                    (response instanceof HttpServletResponse) &&
                    Globals.IS_SECURITY_ENABLED ) {
                final ServletRequest req = request;
                final ServletResponse res = response;
                Principal principal =
                    ((HttpServletRequest) req).getUserPrincipal();
                Object[] args = new Object[]{req, res};
                SecurityUtil.doAsPrivilege("service",
                                           servlet,
                                           classTypeUsedInService,
                                           args,
                                           principal);
            } else {
                servlet.service(request, response);
...
}

// javax.servlet.http.HttpServlet
public abstract class HttpServlet extends GenericServlet {
...
    @Override
    public void service(ServletRequest req, ServletResponse res)
        throws ServletException, IOException {

        HttpServletRequest  request;
        HttpServletResponse response;

        try {
            request = (HttpServletRequest) req;
            response = (HttpServletResponse) res;
        } catch (ClassCastException e) {
            throw new ServletException(lStrings.getString("http.non_http"));
        }
        service(request, response);
    }
...
}

// org.springframework.web.servlet.FrameworkServlet
public abstract class FrameworkServlet extends HttpServletBean implements ApplicationContextAware {
...
	@Override
	protected void service(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {

		HttpMethod httpMethod = HttpMethod.resolve(request.getMethod());
		if (httpMethod == HttpMethod.PATCH || httpMethod == null) {
			processRequest(request, response);
		}
		else {
			super.service(request, response);
		}
	}
...
}

// javax.servlet.http.HttpServlet
public abstract class HttpServlet extends GenericServlet {
...
    protected void service(HttpServletRequest req, HttpServletResponse resp)
        throws ServletException, IOException {

        String method = req.getMethod();

        if (method.equals(METHOD_GET)) {
            long lastModified = getLastModified(req);
            if (lastModified == -1) {
                // servlet doesn't support if-modified-since, no reason
                // to go through further expensive logic
                doGet(req, resp);
...
}

// org.springframework.web.servlet.FrameworkServlet
public abstract class FrameworkServlet extends HttpServletBean implements ApplicationContextAware {
...
	@Override
	protected final void doGet(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {

		processRequest(request, response);
	}
...
	protected final void processRequest(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {

		long startTime = System.currentTimeMillis();
		Throwable failureCause = null;

		LocaleContext previousLocaleContext = LocaleContextHolder.getLocaleContext();
		LocaleContext localeContext = buildLocaleContext(request);

		RequestAttributes previousAttributes = RequestContextHolder.getRequestAttributes();
		ServletRequestAttributes requestAttributes = buildRequestAttributes(request, response, previousAttributes);

		WebAsyncManager asyncManager = WebAsyncUtils.getAsyncManager(request);
		asyncManager.registerCallableInterceptor(FrameworkServlet.class.getName(), new RequestBindingInterceptor());

		initContextHolders(request, localeContext, requestAttributes);

		try {
			doService(request, response);
		}
...		
}

// org.springframework.web.servlet.DispatcherServlet
// HTTP Request url 에 해당하는 @RestController Class method 로 향한다.
public class DispatcherServlet extends FrameworkServlet {
...
	@Override
	protected void doService(HttpServletRequest request, HttpServletResponse response) throws Exception {
		logRequest(request);

		// Keep a snapshot of the request attributes in case of an include,
		// to be able to restore the original attributes after the include.
		Map<String, Object> attributesSnapshot = null;
		if (WebUtils.isIncludeRequest(request)) {
			attributesSnapshot = new HashMap<>();
			Enumeration<?> attrNames = request.getAttributeNames();
			while (attrNames.hasMoreElements()) {
				String attrName = (String) attrNames.nextElement();
				if (this.cleanupAfterInclude || attrName.startsWith(DEFAULT_STRATEGIES_PREFIX)) {
					attributesSnapshot.put(attrName, request.getAttribute(attrName));
				}
			}
		}

		// Make framework objects available to handlers and view objects.
		request.setAttribute(WEB_APPLICATION_CONTEXT_ATTRIBUTE, getWebApplicationContext());
		request.setAttribute(LOCALE_RESOLVER_ATTRIBUTE, this.localeResolver);
		request.setAttribute(THEME_RESOLVER_ATTRIBUTE, this.themeResolver);
		request.setAttribute(THEME_SOURCE_ATTRIBUTE, getThemeSource());

		if (this.flashMapManager != null) {
			FlashMap inputFlashMap = this.flashMapManager.retrieveAndUpdate(request, response);
			if (inputFlashMap != null) {
				request.setAttribute(INPUT_FLASH_MAP_ATTRIBUTE, Collections.unmodifiableMap(inputFlashMap));
			}
			request.setAttribute(OUTPUT_FLASH_MAP_ATTRIBUTE, new FlashMap());
			request.setAttribute(FLASH_MAP_MANAGER_ATTRIBUTE, this.flashMapManager);
		}

		try {
			doDispatch(request, response);
...
	protected void doDispatch(HttpServletRequest request, HttpServletResponse response) throws Exception {
		HttpServletRequest processedRequest = request;
		HandlerExecutionChain mappedHandler = null;
		boolean multipartRequestParsed = false;

		WebAsyncManager asyncManager = WebAsyncUtils.getAsyncManager(request);

		try {
			ModelAndView mv = null;
			Exception dispatchException = null;

			try {
				processedRequest = checkMultipart(request);
				multipartRequestParsed = (processedRequest != request);

				// Determine handler for the current request.
				mappedHandler = getHandler(processedRequest);
				if (mappedHandler == null) {
					noHandlerFound(processedRequest, response);
					return;
				}

				// Determine handler adapter for the current request.
				HandlerAdapter ha = getHandlerAdapter(mappedHandler.getHandler());

				// Process last-modified header, if supported by the handler.
				String method = request.getMethod();
				boolean isGet = "GET".equals(method);
				if (isGet || "HEAD".equals(method)) {
					long lastModified = ha.getLastModified(request, mappedHandler.getHandler());
					if (new ServletWebRequest(request, response).checkNotModified(lastModified) && isGet) {
						return;
					}
				}

				if (!mappedHandler.applyPreHandle(processedRequest, response)) {
					return;
				}

				// Actually invoke the handler.
				mv = ha.handle(processedRequest, response, mappedHandler.getHandler());
	}
...			
}

// org.springframework.web.servlet.mvc.method.AbstractHandlerMethodAdapter
public abstract class AbstractHandlerMethodAdapter extends WebContentGenerator implements HandlerAdapter, Ordered {
...
	@Override
	@Nullable
	public final ModelAndView handle(HttpServletRequest request, HttpServletResponse response, Object handler)
			throws Exception {

		return handleInternal(request, response, (HandlerMethod) handler);
	}
...
}

// org.springframework.web.servlet.mvc.method.annotation.RequestMappingHandlerAdapter
public class RequestMappingHandlerAdapter extends AbstractHandlerMethodAdapter
		implements BeanFactoryAware, InitializingBean {
...
	@Override
	protected ModelAndView handleInternal(HttpServletRequest request,
			HttpServletResponse response, HandlerMethod handlerMethod) throws Exception {

		ModelAndView mav;
		checkRequest(request);

		// Execute invokeHandlerMethod in synchronized block if required.
		if (this.synchronizeOnSession) {
			HttpSession session = request.getSession(false);
			if (session != null) {
				Object mutex = WebUtils.getSessionMutex(session);
				synchronized (mutex) {
					mav = invokeHandlerMethod(request, response, handlerMethod);
				}
			}
			else {
				// No HttpSession available -> no mutex necessary
				mav = invokeHandlerMethod(request, response, handlerMethod);
			}
		}
		else {
			// No synchronization on session demanded at all...
			mav = invokeHandlerMethod(request, response, handlerMethod);
		}
...
    // handlerMethod.method 는 "public java.lang.String com.iamslash.exbasic.hello.HelloController.hello()" 이다.
	@Nullable
	protected ModelAndView invokeHandlerMethod(HttpServletRequest request,
			HttpServletResponse response, HandlerMethod handlerMethod) throws Exception {

		ServletWebRequest webRequest = new ServletWebRequest(request, response);
		try {
			WebDataBinderFactory binderFactory = getDataBinderFactory(handlerMethod);
			ModelFactory modelFactory = getModelFactory(handlerMethod, binderFactory);

			ServletInvocableHandlerMethod invocableMethod = createInvocableHandlerMethod(handlerMethod);
			if (this.argumentResolvers != null) {
				invocableMethod.setHandlerMethodArgumentResolvers(this.argumentResolvers);
			}
			if (this.returnValueHandlers != null) {
				invocableMethod.setHandlerMethodReturnValueHandlers(this.returnValueHandlers);
			}
			invocableMethod.setDataBinderFactory(binderFactory);
			invocableMethod.setParameterNameDiscoverer(this.parameterNameDiscoverer);

			ModelAndViewContainer mavContainer = new ModelAndViewContainer();
			mavContainer.addAllAttributes(RequestContextUtils.getInputFlashMap(request));
			modelFactory.initModel(webRequest, mavContainer, invocableMethod);
			mavContainer.setIgnoreDefaultModelOnRedirect(this.ignoreDefaultModelOnRedirect);

			AsyncWebRequest asyncWebRequest = WebAsyncUtils.createAsyncWebRequest(request, response);
			asyncWebRequest.setTimeout(this.asyncRequestTimeout);

			WebAsyncManager asyncManager = WebAsyncUtils.getAsyncManager(request);
			asyncManager.setTaskExecutor(this.taskExecutor);
			asyncManager.setAsyncWebRequest(asyncWebRequest);
			asyncManager.registerCallableInterceptors(this.callableInterceptors);
			asyncManager.registerDeferredResultInterceptors(this.deferredResultInterceptors);

			if (asyncManager.hasConcurrentResult()) {
				Object result = asyncManager.getConcurrentResult();
				mavContainer = (ModelAndViewContainer) asyncManager.getConcurrentResultContext()[0];
				asyncManager.clearConcurrentResult();
				LogFormatUtils.traceDebug(logger, traceOn -> {
					String formatted = LogFormatUtils.formatValue(result, !traceOn);
					return "Resume with async result [" + formatted + "]";
				});
				invocableMethod = invocableMethod.wrapConcurrentResult(result);
			}

			invocableMethod.invokeAndHandle(webRequest, mavContainer);
...
}

// org.springframework.web.servlet.mvc.method.annotation.ServletInvocableHandlerMethod
public class ServletInvocableHandlerMethod extends InvocableHandlerMethod {
...
	public void invokeAndHandle(ServletWebRequest webRequest, ModelAndViewContainer mavContainer,
			Object... providedArgs) throws Exception {

		Object returnValue = invokeForRequest(webRequest, mavContainer, providedArgs);
...
}

// org.springframework.web.method.support.InvocableHandlerMethod
public class InvocableHandlerMethod extends HandlerMethod {
...
	@Nullable
	public Object invokeForRequest(NativeWebRequest request, @Nullable ModelAndViewContainer mavContainer,
			Object... providedArgs) throws Exception {

		Object[] args = getMethodArgumentValues(request, mavContainer, providedArgs);
		if (logger.isTraceEnabled()) {
			logger.trace("Arguments: " + Arrays.toString(args));
		}
		return doInvoke(args);
	}
	@Nullable
	protected Object doInvoke(Object... args) throws Exception {
		ReflectionUtils.makeAccessible(getBridgedMethod());
		try {
			return getBridgedMethod().invoke(getBean(), args);
		}
...
}

// com.iamslash.exbasic.hello.HelloController
@RestController
public class HelloController {

  @GetMapping("/hello")
  public String hello() {
    return "world";
  }
...
}
```

# Spring Application Handler Mapping Flow

HTTP Request 가 도착하면 `DispatcherServlet::handlerMappings` 의 `RequestMappingHandlerMapping` 은 `@Control Class` 의 `@RequestMapping Method` 들 중 적당한 것을 골라 `HandlerExecutionChain` instance 를 생성하여 리턴한다.

```java
// org.springframework.web.servlet.DispatcherServlet
public class DispatcherServlet extends FrameworkServlet {
	@Nullable
	protected HandlerExecutionChain getHandler(HttpServletRequest request) throws Exception {
		if (this.handlerMappings != null) {
			for (HandlerMapping mapping : this.handlerMappings) {
				HandlerExecutionChain handler = mapping.getHandler(request);
				if (handler != null) {
					return handler;
				}
			}
		}
		return null;
	}    
}    
```

URL 과 짝이 되는 `@Control Class` 의 `@RequestMapping Method` 의 모음은 이미 `AbstractHandlerMethodMapping` 의 `private final MappingRegistry mappingRegistry = new MappingRegistry()` 에 저장되어 있다. `AbstractHandlerMethodMapping` 은 `RequestMappingHandlerMapping` 의 부모이다.

# Spring Application View Rendering Flow

`DispatchServlet` 은 `@Control Class` 의 `@RequestMapping Method` 를 호출하고 나면 결과를 `processDispatchResult` 에서 rendering 한다.

```java
// org.springframework.web.servlet.DispatcherServlet
public class DispatcherServlet extends FrameworkServlet {
	protected void doDispatch(HttpServletRequest request, HttpServletResponse response) throws Exception {
...
				// Actually invoke the handler.
				mv = ha.handle(processedRequest, response, mappedHandler.getHandler());

				if (asyncManager.isConcurrentHandlingStarted()) {
					return;
				}

				applyDefaultViewName(processedRequest, mv);
				mappedHandler.applyPostHandle(processedRequest, response, mv);
			}
			catch (Exception ex) {
				dispatchException = ex;
			}
			catch (Throwable err) {
				// As of 4.3, we're processing Errors thrown from handler methods as well,
				// making them available for @ExceptionHandler methods and other scenarios.
				dispatchException = new NestedServletException("Handler dispatch failed", err);
			}
			processDispatchResult(processedRequest, response, mappedHandler, mv, dispatchException);
...            
    }
    private void processDispatchResult(HttpServletRequest request, HttpServletResponse response,
			@Nullable HandlerExecutionChain mappedHandler, @Nullable ModelAndView mv,
			@Nullable Exception exception) throws Exception {
...

        // mv != null 이 아니면 적당한 view 를 골라서 rendering 한다.
		if (mv != null && !mv.wasCleared()) {
			render(mv, request, response);
			if (errorView) {
				WebUtils.clearErrorRequestAttributes(request);
			}
		}
		else {
			if (logger.isTraceEnabled()) {
				logger.trace("No view rendering, null ModelAndView returned.");
			}
		}
...        
    }    
}   
```

`@RestController Class` 를 사용했다면 `mv == null` 이다.

```java
// com.iamslash.exbasic.hello.HelloController
@RestController
public class HelloController {

  @GetMapping("/hello")
  public String hello() {
    return "world";
  }
...
}
```

# Spring Application Exception Rendering Flow

`DispatchServlet` 은 `@Control Class` 의 `@RequestMapping Method` 를 호출하고 나면 결과를 `processDispatchResult` 에서 Exception 을 rendering 한다.

```java
// org.springframework.web.servlet.DispatcherServlet
public class DispatcherServlet extends FrameworkServlet {
	protected void doDispatch(HttpServletRequest request, HttpServletResponse response) throws Exception {
...
				// Actually invoke the handler.
				mv = ha.handle(processedRequest, response, mappedHandler.getHandler());

				if (asyncManager.isConcurrentHandlingStarted()) {
					return;
				}

				applyDefaultViewName(processedRequest, mv);
				mappedHandler.applyPostHandle(processedRequest, response, mv);
			}
			catch (Exception ex) {
				dispatchException = ex;
			}
			catch (Throwable err) {
				// As of 4.3, we're processing Errors thrown from handler methods as well,
				// making them available for @ExceptionHandler methods and other scenarios.
				dispatchException = new NestedServletException("Handler dispatch failed", err);
			}
			processDispatchResult(processedRequest, response, mappedHandler, mv, dispatchException);
...            
    }
    // exception != null 이 아니면 Exception 을 resolving 한다.
    private void processDispatchResult(HttpServletRequest request, HttpServletResponse response,
			@Nullable HandlerExecutionChain mappedHandler, @Nullable ModelAndView mv,
			@Nullable Exception exception) throws Exception {

		boolean errorView = false;

		if (exception != null) {
			if (exception instanceof ModelAndViewDefiningException) {
				logger.debug("ModelAndViewDefiningException encountered", exception);
				mv = ((ModelAndViewDefiningException) exception).getModelAndView();
			}
			else {
				Object handler = (mappedHandler != null ? mappedHandler.getHandler() : null);
				mv = processHandlerException(request, response, handler, exception);
				errorView = (mv != null);
			}
		}
...        
    }    
}   
```

# Spring Application RestController Class HTTP Response Flow

다음과 같은 `@RequestController Class` 를 사용하면 다른 방식으로 "world" 문자열이 HTTP Response Body 에 담겨진다.

```java
// com.iamslash.exbasic.hello.HelloController
@RestController
public class HelloController {

  @GetMapping("/hello")
  public String hello() {
    return "world";
  }
...
}
```

결국 `RequestResponseBodyMethodProcessor::writeWithMessageConverters()` 에서 "world" 를 HTTP Response Body 에 쓴다. 

```java
// org.springframework.web.servlet.mvc.method.annotation.ServletInvocableHandlerMethod
public class ServletInvocableHandlerMethod extends InvocableHandlerMethod {
...
	public void invokeAndHandle(ServletWebRequest webRequest, ModelAndViewContainer mavContainer,
			Object... providedArgs) throws Exception {

		Object returnValue = invokeForRequest(webRequest, mavContainer, providedArgs);
		setResponseStatus(webRequest);

		if (returnValue == null) {
			if (isRequestNotModified(webRequest) || getResponseStatus() != null || mavContainer.isRequestHandled()) {
				disableContentCachingIfNecessary(webRequest);
				mavContainer.setRequestHandled(true);
				return;
			}
		}
		else if (StringUtils.hasText(getResponseStatusReason())) {
			mavContainer.setRequestHandled(true);
			return;
		}

		mavContainer.setRequestHandled(false);
		Assert.state(this.returnValueHandlers != null, "No return value handlers");
		try {
			this.returnValueHandlers.handleReturnValue(
					returnValue, getReturnValueType(returnValue), mavContainer, webRequest);
		}
		catch (Exception ex) {
			if (logger.isTraceEnabled()) {
				logger.trace(formatErrorForReturnValue(returnValue), ex);
			}
			throw ex;
		}
	}
...
}

// org.springframework.web.method.support.HandlerMethodReturnValueHandlerComposite
public class HandlerMethodReturnValueHandlerComposite implements HandlerMethodReturnValueHandler {
...
	@Override
    // handler 는 RequestResponseBodyMethodProcessor 이다.
	public void handleReturnValue(@Nullable Object returnValue, MethodParameter returnType,
			ModelAndViewContainer mavContainer, NativeWebRequest webRequest) throws Exception {

		HandlerMethodReturnValueHandler handler = selectHandler(returnValue, returnType);
		if (handler == null) {
			throw new IllegalArgumentException("Unknown return value type: " + returnType.getParameterType().getName());
		}
		handler.handleReturnValue(returnValue, returnType, mavContainer, webRequest);
	}
...
}

// org.springframework.web.servlet.mvc.method.annotation.RequestResponseBodyMethodProcessor
public class RequestResponseBodyMethodProcessor extends AbstractMessageConverterMethodProcessor {
...
	@Override
	public boolean supportsReturnType(MethodParameter returnType) {
		return (AnnotatedElementUtils.hasAnnotation(returnType.getContainingClass(), ResponseBody.class) ||
				returnType.hasMethodAnnotation(ResponseBody.class));
	}

	@Override
	public void handleReturnValue(@Nullable Object returnValue, MethodParameter returnType,
			ModelAndViewContainer mavContainer, NativeWebRequest webRequest)
			throws IOException, HttpMediaTypeNotAcceptableException, HttpMessageNotWritableException {

		mavContainer.setRequestHandled(true);
		ServletServerHttpRequest inputMessage = createInputMessage(webRequest);
		ServletServerHttpResponse outputMessage = createOutputMessage(webRequest);

		// Try even with null return value. ResponseBodyAdvice could get involved.
		writeWithMessageConverters(returnValue, returnType, inputMessage, outputMessage);
	}
...
}

// org.springframework.http.converter.StringHttpMessageConverter
// RequestResponseBodyMethodProcessor 는 protected final List<HttpMessageConverter<?>> messageConverters 를 가지고 있다.
// "world" 의 경우 StringHttpMessageConverter 가 사용된다.
public class StringHttpMessageConverter extends AbstractHttpMessageConverter<String> {
...
	@Override
	protected void writeInternal(String str, HttpOutputMessage outputMessage) throws IOException {
		HttpHeaders headers = outputMessage.getHeaders();
		if (this.writeAcceptCharset && headers.get(HttpHeaders.ACCEPT_CHARSET) == null) {
			headers.setAcceptCharset(getAcceptedCharsets());
		}
		Charset charset = getContentTypeCharset(headers.getContentType());
		StreamUtils.copy(str, charset, outputMessage.getBody());
	}
...
}

// org.springframework.http.server.ServletServerHttpResponse
public class ServletServerHttpResponse implements ServerHttpResponse {
...
	@Override
	public OutputStream getBody() throws IOException {
		this.bodyUsed = true;
		writeHeaders();
		return this.servletResponse.getOutputStream();
	}
...
}
```
