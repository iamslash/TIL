- [Abstract](#abstract)
- [Materials](#materials)
- [Basic](#basic)
- [Annotation Processing](#annotation-processing)

----

# Abstract

java annotation 을 정리한다.

# Materials

* [더 자바, 코드를 조작하는 다양한 방법 by 백기선](https://www.inflearn.com/course/the-java-code-manipulation/dashboard)

# Basic

annotation 은 특수한 interface 이다. `@interface` 를 이용하여 다음과 같이 선언한다. class, method, field 등등에 `key=value` 형태의 추가정보를 주입할 수 있다.

```java
public @interface SimpleAnnotation {
}

public @interface SimpleAnnotationWithAttributes {
   String name();
   int order() default 0;
}
```

예를 들어 다음과 같이 `SimpleAnnotationWithValue` 를 선언하고 field `aaa` 에 사용하면 `aaa` 에 `value=new annotation` 가 추가된다.

```java
public @interface SimpleAnnotationWithValue {
   String value();
}

@SimpleAnnotationWithValue("new annotation")
public int aaa;
```

다음은 builtin annotation 들이다.

| name                   | desc                                | type  |
| :--------------------- | :---------------------------------- | -- |
| `@Deprecated`          | something deprecated | |
| `@Override`            | overrided method  |  |
| `@SuppressWarnings`    | suppress compile warnings           |  |
| `@SafeVarargs`         | suppress variable arguments warning |  |
| `@Retention`           | retention of annotation             | `SOURCE, CLASS, RUNTIME`  |
| `@Target`              | target of annotation                | `ANNOTATION_TYPE, CONSTRUCTOR, FIELD, LOCAL_VARIABLE, METHOD, PACKAGE, PARAMETER, TYPE, TYPE_PARAMETER, TYPE_USE` | 
| `@Documented`          | documented in javadoc               | |
| `@Inherited`           | this annotation will be inherited   |  |
| `@FunctionalInterface` | for functional interface            |  |
| `@Repeatable`          | repeatable annotation               | |

`@Target` 를 사용하면 다음과 같이 annotation 의 대상이 되는 자료형을 정할 수 있다.

```java
import java.lang.annotation.ElementType;
import java.lang.annotation.Target;
@Target({ElementType.FIELD, ElementType.METHOD})
public @interface AnnotationWithTarget {
}
```

annotation 은 기본적으로 상속되지 않는다. 그러나 `@Inherited` 를 사용하면 상속된다.

```java
@Target({ElementType.TYPE})
@Retention(RetentionPolicy.RUNTIME)
@Inherited
@interface InheritableAnnotation {
}

@InheritableAnnotation
public class Parent {
}

public class Child extends Parent {
}
```

다음은 `@Repeatable` 를 사용한 예이다.

```java
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface RepeatableAnnotations {
   RepeatableAnnotation[] value();
}
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Repeatable(RepeatableAnnotations.class)
public @interface RepeatableAnnotation {
   String value();
};
@RepeatableAnnotation("repeatition 1")
@RepeatableAnnotation("repeatition 2")
public void performAction() {
   // Some code here
}
```

# Annotation Processing

`@Foo` 라는 Annotation 을 구현해 보자. 먼저 `Foo.java` 가 필요하다.

* Foo.java

```java
@Target(ElementType.TYPE) 
@Retention(RetentionPolicy.SOURCE)
public @interface Foo {
}
```

그리고 Annotation processing 을 위해 `FooProcessor.java` 가 필요하다. Element 는 package, class 등등을 말한다. Annotation processing 은 round 단위로 처리된다. 특정 Annotation processor 의 process() 에서 return true 하면 다음 Annotation processing 은 되지 않는다. 마치 spring 의 filter 와 같다. 

* FooProcessor.java

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

두 개의 java 를 생성하고 `src/main/resources/META-INF.services/javax.annotation.processing.Processor` 에 다음과 같이 Annotation processor 의 full package path 를 적는다.

```
com.iamslash.FooProcessor
```

FooProcessor.class 가 존재해야 AnnotationProcessor 가 실행될 수 있다.

```
# FooProcessor.class 가 없어서 error 가 발생한다.
$ mvn install

# FooProcessor.class 가 생성된다.
$ mvn clean install
# FooProcessor.class 가 생성되었기 때문에 compile 이 된다.
$ mvn install
```

그러나 [AutoService](https://github.com/google/auto/tree/master/service) 를 사용하면 위와 같은 full package path 를 자동으로 생성해 준다.

```java
@AutoService(Processor.class)
public class FooProcessor implements Processor {
  @Override
  public Set<String> getSupportedAnnotationTypes() {
    return Set.of(Foo.class.getName());
  }
...
```

FooProcessor 에 [java poet](https://github.com/square/javapoet) 을 이용하여 코드를 조작해 보자.

```java
@AutoService(Processor.class)
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
      TypeElement typeEl = (TypeElement)el;
      ClassName className = ClassName.get(typeEl);

      MethodSpec HelloWorld = MethodSpec.methodBuilder("HelloWorld")
        .addModifiers(Modifer.PUBLIC)
        .returns(String.class)
        .addStatement("return $S", "Foo!!!")
        .build();

      TypeSpec specialFoo = TypeSpec.classBuilder("SpecialFoo")
        .addModifiers(Modifier.PUBLIC)
        .addSuperinterface(className)
        .addMethod(specialFoo)
        .build();
      Filer filer = processingEnv.getFiler();
      try {
        JavaFile.builder(className.packageName(), specialFoo)
          .build()
          .writeTo(filter);
      } catch (IOException e) {
        processingEnv.getMessage().printMessage(Diagnostic.Kind.ERROR, "ERROR: " + e);
      }
    }
    return   true;
  }
}
```

intelliJ 의 Enable annotation processing 을 check 해야 Annotation Processor 가 생성한 코드가 `target/generated-sources/annotations/com.iamslash.SpecialFoo` 가 생성된다. intelliJ 의 module 에서 `target/generated-sources` 를 source directory 로 등록해야 `main()` 에서 `SpecialFoo` 를 사용할 수 있다.

```java
public class App {
  public static void main(String[] args) {
    SpecialFoo specialFoo = new SpecialFoo();
    ...
  }
}
```
