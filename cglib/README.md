# Abstract

cglib 은 code generator library 이다. 

# Materials

* [다이나믹 프록시와 cglib에 대해 알아보자 | velog](https://velog.io/@jakeseo_me/%EB%8B%A4%EC%9D%B4%EB%82%98%EB%AF%B9-%ED%94%84%EB%A1%9D%EC%8B%9C%EC%97%90-%EB%8C%80%ED%95%B4-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90)

# Proxy

## Pure Proxy

Interface 가 반드시 있어야 한다.

```java
package com.iamslash.pureproxy;

public class PureProxy {

    interface Introduce {
        void printWhoYouAre();
    }

    static class IntroduceImpl implements Introduce {
        @Override
        public void printWhoYouAre() {
            System.out.println("I am Jake");
        }
    }

    static class IntroduceProxy implements Introduce {
        Introduce introduce = new IntroduceImpl();

        public void hello() {
            System.out.println("Hello");
        }

        public void bye() {
            System.out.println("Bye");
        }

        @Override
        public void printWhoYouAre() {
            hello();
            introduce.printWhoYouAre();
            bye();
        }
    }

    public static void main(String[] args) {
        Introduce introduce = new IntroduceProxy();
        introduce.printWhoYouAre();
    }
}
```

## Dynamic Proxy

Reflection 을 사용해야 한다. 느리다.

```java
package com.iamslash.dynamicproxy;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;

public class DynamicProxy {
    interface Introduce {
        void printWhoYouAre();
    }

    static class IntroduceImpl implements Introduce {
        @Override
        public void printWhoYouAre() {
            System.out.println("I am Jake");
        }
    }

    static class IntroduceProxy implements InvocationHandler {
        private final Introduce introduce;

        public IntroduceProxy(Introduce introduce) {
            this.introduce = introduce;
        }

        public void hello() {
            System.out.println("Hello");
        }

        public void bye() {
            System.out.println("Bye");
        }

        @Override
        public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
            hello();
            method.invoke(introduce, args);
            bye();
            return null;
        }
    }

    public static void main(String[] args) {
        IntroduceImpl introduceImpl = new IntroduceImpl();
        IntroduceProxy introduceProxy = new IntroduceProxy(introduceImpl);
        Introduce introduce = (Introduce) Proxy.newProxyInstance(
            Introduce.class.getClassLoader(),
            new Class[] {Introduce.class},
            introduceProxy
        );
        introduce.printWhoYouAre();
    }
}
```

## CGLib Proxy

Interface 가 없어도 된다. Dynamic Proxy 보다 빠르다.

```java
package com.iamslash.cglibproxy;

import net.sf.cglib.proxy.Enhancer;
import net.sf.cglib.proxy.MethodInterceptor;
import net.sf.cglib.proxy.MethodProxy;

import java.lang.reflect.Method;

public class CglibProxy {

    static class IntroduceImpl {
        public void printWhoYouAre() {
            System.out.println("I am Jake");
        }
    }

    static class IntroduceProxy implements MethodInterceptor {
        public void hello() {
            System.out.println("Hello");
        }

        public void bye() {
            System.out.println("Bye");
        }

        @Override
        public Object intercept(Object obj, Method method, Object[] args, MethodProxy proxy) throws Throwable {
            hello();
            Object result = proxy.invokeSuper(obj, args);
            bye();
            return result;
        }
    }

    public static void main(String[] args) {
        Enhancer enhancer = new Enhancer();
        enhancer.setSuperclass(IntroduceImpl.class);
        enhancer.setCallback(new IntroduceProxy());
        IntroduceImpl proxy = (IntroduceImpl) enhancer.create();
        proxy.printWhoYouAre();
    }
}
```
