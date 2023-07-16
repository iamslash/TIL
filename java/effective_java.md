- [Materials](#materials)
- [Creating and Destroying Objects](#creating-and-destroying-objects)
  - [item1: Consider static factory methods instead of constructors](#item1-consider-static-factory-methods-instead-of-constructors)
  - [item2: Consider a builder when faced with many constructor parameters](#item2-consider-a-builder-when-faced-with-many-constructor-parameters)
  - [item3: Enforce the singleton property with a private constructor or an enum type](#item3-enforce-the-singleton-property-with-a-private-constructor-or-an-enum-type)
  - [item4: Enforce noninstantiability with a private constructor](#item4-enforce-noninstantiability-with-a-private-constructor)
  - [item5: Prefer dependency injection to hardwiring resources](#item5-prefer-dependency-injection-to-hardwiring-resources)
  - [item6: Avoid creating unnecessary objects](#item6-avoid-creating-unnecessary-objects)
  - [item7: Eliminate obsolete object references](#item7-eliminate-obsolete-object-references)
  - [item8: Avoid finalizers and cleaners](#item8-avoid-finalizers-and-cleaners)
  - [item9: Prefer try-with-resources to try-finally](#item9-prefer-try-with-resources-to-try-finally)
- [Methods Common to All Objects](#methods-common-to-all-objects)
  - [item10: Obey the general contract when overriding equals](#item10-obey-the-general-contract-when-overriding-equals)
  - [item11: Always override hashCode when you override equals](#item11-always-override-hashcode-when-you-override-equals)
  - [item12: Always override toString](#item12-always-override-tostring)
  - [item13: Override clone judiciously](#item13-override-clone-judiciously)
  - [item14: Consider implementing Comparable](#item14-consider-implementing-comparable)
- [Classes and Interfaces](#classes-and-interfaces)
  - [item15: Minimize the accessibility of classes and members](#item15-minimize-the-accessibility-of-classes-and-members)
  - [item16: In public classes, use accessor methods, not public fields](#item16-in-public-classes-use-accessor-methods-not-public-fields)
  - [item17: Minimize mutability](#item17-minimize-mutability)
  - [item18: Favor composition over inheritance](#item18-favor-composition-over-inheritance)
  - [item19: Design and document for inheritance or else prohibit it](#item19-design-and-document-for-inheritance-or-else-prohibit-it)
  - [item20: Prefer interfaces to abstract classes](#item20-prefer-interfaces-to-abstract-classes)
  - [item21: Design interfaces for posterity](#item21-design-interfaces-for-posterity)
  - [item22: Use interfaces only to define types](#item22-use-interfaces-only-to-define-types)
  - [item23: Prefer class hierarchies to tagged classes](#item23-prefer-class-hierarchies-to-tagged-classes)
  - [item24: Favor static member classes over nonstatic](#item24-favor-static-member-classes-over-nonstatic)
  - [item25: Limit source files to a single top-level class](#item25-limit-source-files-to-a-single-top-level-class)
- [Generics](#generics)
  - [item26: Don’t use raw types](#item26-dont-use-raw-types)
  - [item27: Eliminate unchecked warnings](#item27-eliminate-unchecked-warnings)
  - [item28: Prefer lists to arrays](#item28-prefer-lists-to-arrays)
  - [item29: Favor generic types](#item29-favor-generic-types)
  - [item30: Favor generic methods](#item30-favor-generic-methods)
  - [item31: Use bounded wildcards to increase API flexibility](#item31-use-bounded-wildcards-to-increase-api-flexibility)
  - [item32: Combine generics and varargs judiciously](#item32-combine-generics-and-varargs-judiciously)
  - [item33: Consider typesafe heterogeneous containers](#item33-consider-typesafe-heterogeneous-containers)
- [Enums and Annotations](#enums-and-annotations)
  - [item34: Use enums instead of int constants](#item34-use-enums-instead-of-int-constants)
  - [item35: Use instance fields instead of ordinals](#item35-use-instance-fields-instead-of-ordinals)
  - [item36: Use EnumSet instead of bit fields](#item36-use-enumset-instead-of-bit-fields)
  - [item37: Use EnumMap instead of ordinal indexing](#item37-use-enummap-instead-of-ordinal-indexing)
  - [item38: Emulate extensible enums with interfaces](#item38-emulate-extensible-enums-with-interfaces)
  - [item39: Prefer annotations to naming patterns](#item39-prefer-annotations-to-naming-patterns)
  - [item40: Consistently use the Override annotation](#item40-consistently-use-the-override-annotation)
  - [item41: Use marker interfaces to define types](#item41-use-marker-interfaces-to-define-types)
- [Lambdas and Streams](#lambdas-and-streams)
  - [item42: Prefer lambdas to anonymous classes](#item42-prefer-lambdas-to-anonymous-classes)
  - [item43: Prefer method references to lambdas](#item43-prefer-method-references-to-lambdas)
  - [item44: Favor the use of standard functional interfaces](#item44-favor-the-use-of-standard-functional-interfaces)
  - [item45: Use streams judiciously](#item45-use-streams-judiciously)
  - [item46: Prefer side-effect-free functions in streams](#item46-prefer-side-effect-free-functions-in-streams)
  - [item47: Prefer Collection to Stream as a return type](#item47-prefer-collection-to-stream-as-a-return-type)
  - [item48: Use caution when making streams parallel](#item48-use-caution-when-making-streams-parallel)
- [Methods](#methods)
  - [item49: Check parameters for validity](#item49-check-parameters-for-validity)
  - [item50: Make defensive copies when needed](#item50-make-defensive-copies-when-needed)
  - [item51: Design method signatures carefully](#item51-design-method-signatures-carefully)
  - [item52: Use overloading judiciously](#item52-use-overloading-judiciously)
  - [item53: Use varargs judiciously](#item53-use-varargs-judiciously)
  - [item54: Return empty collections or arrays, not nulls](#item54-return-empty-collections-or-arrays-not-nulls)
  - [item55: Return optionals judiciously](#item55-return-optionals-judiciously)
  - [item56: Write doc comments for all exposed API elements](#item56-write-doc-comments-for-all-exposed-api-elements)
- [General Programming](#general-programming)
  - [item57: Minimize the scope of local variables](#item57-minimize-the-scope-of-local-variables)
  - [item58: Prefer for-each loops to traditional for loops](#item58-prefer-for-each-loops-to-traditional-for-loops)
  - [item59: Know and use the libraries](#item59-know-and-use-the-libraries)
  - [item60: Avoid float and double if exact answers are required](#item60-avoid-float-and-double-if-exact-answers-are-required)
  - [item61: Prefer primitive types to boxed primitives](#item61-prefer-primitive-types-to-boxed-primitives)
  - [item62: Avoid strings where other types are more appropriate](#item62-avoid-strings-where-other-types-are-more-appropriate)
  - [item63: Beware the performance of string concatenation](#item63-beware-the-performance-of-string-concatenation)
  - [item64: Refer to objects by their interfaces](#item64-refer-to-objects-by-their-interfaces)
  - [item65: Prefer interfaces to reflection](#item65-prefer-interfaces-to-reflection)
  - [item66: Use native methods judiciously](#item66-use-native-methods-judiciously)
  - [item67: Optimize judiciously](#item67-optimize-judiciously)
  - [item68: Adhere to generally accepted naming conventions](#item68-adhere-to-generally-accepted-naming-conventions)
- [Exceptions](#exceptions)
  - [item69: Use exceptions only for exceptional conditions](#item69-use-exceptions-only-for-exceptional-conditions)
  - [item70: Use checked exceptions for recoverable conditions and runtime exceptions for programming errors](#item70-use-checked-exceptions-for-recoverable-conditions-and-runtime-exceptions-for-programming-errors)
  - [item71: Avoid unnecessary use of checked exceptions](#item71-avoid-unnecessary-use-of-checked-exceptions)
  - [item72: Favor the use of standard exceptions](#item72-favor-the-use-of-standard-exceptions)
  - [item73: Throw exceptions appropriate to the abstraction](#item73-throw-exceptions-appropriate-to-the-abstraction)
  - [item74: Document all exceptions thrown by each method](#item74-document-all-exceptions-thrown-by-each-method)
  - [item75: Include failure-capture information in detail messages](#item75-include-failure-capture-information-in-detail-messages)
  - [item76: Strive for failure atomicity](#item76-strive-for-failure-atomicity)
  - [item77: Don’t ignore exceptions](#item77-dont-ignore-exceptions)
- [Concurrency](#concurrency)
  - [item78: Synchronize access to shared mutable data](#item78-synchronize-access-to-shared-mutable-data)
  - [item79: Avoid excessive synchronization](#item79-avoid-excessive-synchronization)
  - [item80: Prefer executors, tasks, and streams to threads](#item80-prefer-executors-tasks-and-streams-to-threads)
  - [item81: Prefer concurrency utilities to wait and notify](#item81-prefer-concurrency-utilities-to-wait-and-notify)
  - [item82: Document thread safety](#item82-document-thread-safety)
  - [item83: Use lazy initialization judiciously](#item83-use-lazy-initialization-judiciously)
  - [item84: Don’t depend on the thread scheduler](#item84-dont-depend-on-the-thread-scheduler)
- [Serialization](#serialization)
  - [item85: Prefer alternatives to Java serialization](#item85-prefer-alternatives-to-java-serialization)
  - [item86: Implement Serializable with great caution](#item86-implement-serializable-with-great-caution)
  - [item87: Consider using a custom serialized form](#item87-consider-using-a-custom-serialized-form)
  - [item88: Write readObject methods defensively](#item88-write-readobject-methods-defensively)
  - [item89: For instance control, prefer enum types to readResolve](#item89-for-instance-control-prefer-enum-types-to-readresolve)
  - [item90: Consider serialization proxies instead of serialized instances](#item90-consider-serialization-proxies-instead-of-serialized-instances)


-----

# Materials

* [Effective Java 3/E Study @ github](https://github.com/19F-Study/effective-java)
* [[책] Joshua Bloch, Effective Java 3rd Edition, 2018, Addison-Wesley (1) @ medium](https://myeongjae.kim/blog/2020/06/28/effective-java-3rd-1)
  * [[책] Joshua Bloch, Effective Java 3rd Edition, 2018, Addison-Wesley (2) @ medium](https://myeongjae.kim/blog/2020/06/28/effective-java-3rd-2)
* [Effective Java 3/E 정리](https://medium.com/@ddt1984/effective-java-3-e-%EC%A0%95%EB%A6%AC-c3fb43eec9d2)
* [『이펙티브 자바 3판』(원서: Effective Java 3rd Edition) | github](https://github.com/WegraLee/effective-java-3e-source-code)
  * [Effective Java, Third Edition](https://github.com/jbloch/effective-java-3e-source-code) 

# Creating and Destroying Objects

## item1: Consider static factory methods instead of constructors

Static factory methods are methods that return an instance of the class. They
provide advantages over constructors, like better naming, not requiring creating
new objects every time they're called, and allowing for the return of subtypes.

```java
public class MyClass {
    public static MyClass newInstance() {
        return new MyClass();
    }
}
```

## item2: Consider a builder when faced with many constructor parameters

When constructors have many parameters, it becomes difficult to read and
maintain, especially when there are optional parameters. The Builder pattern
solves this by allowing you to build a class providing only the required
parameters.

```java
public class Person {
    private final String name;
    private final int age;
    private final String address;

    private Person(Builder builder) {
        this.name = builder.name;
        this.age = builder.age;
        this.address = builder.address;
    }

    public static class Builder {
        private String name;
        private int age;
        private String address;

        public Builder(String name) {
            this.name = name;
        }

        public Builder age(int age) {
            this.age = age;
            return this;
        }

        public Builder address(String address) {
            this.address = address;
            return this;
        }

        public Person build() {
            return new Person(this);
        }
    }
}
```

## item3: Enforce the singleton property with a private constructor or an enum type

Singleton is a design pattern that restricts a class to have only one instance.
It can be enforced with a private constructor or an enum type.

```java
public class Singleton {
    private static final Singleton INSTANCE = new Singleton();

    private Singleton() {
    }

    public static Singleton getInstance() {
        return INSTANCE;
    }
}
```

## item4: Enforce noninstantiability with a private constructor

Sometimes, it's necessary to create a non-instantiable class, for example, a
utility class with static methods. To enforce noninstantiability, include a
private constructor that throws an exception.

```java
public class UtilityClass {
    private UtilityClass() {
        throw new AssertionError("Non-instantiable class");
    }
}
```

## item5: Prefer dependency injection to hardwiring resources

Dependency injection promotes flexibility and testability by providing
dependencies to an object, rather than the object being responsible for creating
or looking up its dependencies.

```java
public class MyClass {
    private final Dependency dependency;

    public MyClass(Dependency dependency) {
        this.dependency = dependency;
    }
}
```

## item6: Avoid creating unnecessary objects

Always reuse objects when it's safe, rather than creating new ones, to minimize
performance problems and memory leaks.

```java
public class ExpensiveObject {
    private static ExpensiveObject reusableInstance = new ExpensiveObject();

    public static ExpensiveObject getInstance() {
        return reusableInstance;
    }
}
```

## item7: Eliminate obsolete object references

A simple way to avoid memory leaks is removing references that are no longer
needed. It's essential to clean your resources, especially when using mutable
types.

```java
public class Stack {
    private Object[] elements;
    private int size = 0;

    public void push(Object element) {
        ensureCapacity();
        elements[size++] = element;
    }

    public Object pop() {
        if (size == 0)
            throw new EmptyStackException();
        Object result = elements[--size];
        elements[size] = null; // Eliminate obsolete reference
        return result;
    }
}
```

## item8: Avoid finalizers and cleaners

Finalizers and cleaners are often ineffective, unpredictable, and slow. They
should be avoided. Use try-finally or try-with-resources instead.

```java
try (InputStream in = new FileInputStream(file)) {
    // Process input
} catch (IOException e) {
    // Handle exception
}
```

## item9: Prefer try-with-resources to try-finally

Try-with-resources is a cleaner and safer way of handling resources, as it
automatically closes the resources and reduces the risk of error.

```java
try (InputStream in = new FileInputStream(file)) {
    // Process input
} catch (IOException e) {
    // Handle exception
}
```

# Methods Common to All Objects

## item10: Obey the general contract when overriding equals

When overriding the equals() method, ensure it's reflexive, symmetric,
transitive, consistent, and follows the general contract of overriding equals.

```java
@Override
public boolean equals(Object obj) {
    if (this == obj)
        return true;
    if (!(obj instanceof MyClass))
        return false;
    MyClass other = (MyClass) obj;
    return property1.equals(other.property1) &&
           property2.equals(other.property2);
}
```

## item11: Always override hashCode when you override equals

If you override the `equals()` method, you must also override the `hashCode()`
method to prevent inconsistent behavior in collections.

```java
@Override
public int hashCode() {
    int result = 17;
    result = 31 * result + property1.hashCode();
    result = 31 * result + property2.hashCode();
    return result;
}
```

## item12: Always override toString

Overriding `toString()` is useful for providing a human-readable representation
of your object.

```java
@Override
public String toString() {
    return String.format("MyClass[property1=%s, property2=%s]", property1, property2);
}
```

## item13: Override clone judiciously

The `clone()` method can cause various issues, such as violating the single
responsibility principle or causing subtle bugs. It's better to use copy
constructors or factory methods.

```java
public MyClass(MyClass source) {
    this.property1 = source.property1;
    this.property2 = source.property2;
}
```

## item14: Consider implementing Comparable

Implementing Comparable allows the sorting of instances based on their natural
order.

```java
public class Person implements Comparable<Person> {
    private String name;

    @Override
    public int compareTo(Person other) {
        return name.compareTo(other.name);
    }
}
```

# Classes and Interfaces

## item15: Minimize the accessibility of classes and members

Least privilege principle: assign the lowest possible access level to classes
and members.

```java
public class MyClass {
    private int privateMember;
    protected int protectedMember;
    int packagePrivateMember;
    public int publicMember;
}
```

## item16: In public classes, use accessor methods, not public fields

Encapsulate fields using getter and setter methods, providing better control
over their use.

```java
public class MyClass {
    private int value;

    public int getValue() {
        return value;
    }

    public void setValue(int value) {
        this.value = value;
    }
}
```

## item17: Minimize mutability

Immutable classes are more robust and easier to develop. Achieve immutability by
making all fields final, setting them through the constructor, and not providing
setters.

```java
public final class ImmutableClass {
    private final int value;

    public ImmutableClass(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }
}
```

## item18: Favor composition over inheritance

Composition provides better encapsulation and flexibility than inheritance.

```java
public class MyClass {
    private AnotherClass anotherClass;

    public MyClass(AnotherClass anotherClass) {
        this.anotherClass = anotherClass;
    }

    public void someMethod() {
        anotherClass.someMethod();
    }
}
```

## item19: Design and document for inheritance or else prohibit it

If a class is not designed with inheritance in mind, declare it as final or make
the constructor private to prevent unintended subclassing.

```java
public final class NonInheritableClass {
}
```

## item20: Prefer interfaces to abstract classes

Interfaces provide greater flexibility and better composition, and allow for
multiple inheritance.

```java
public interface MyInterface {
    void someMethod();
}
```

## item21: Design interfaces for posterity

To provide greater flexibility, design interfaces with the possibility of future
extension by providing default methods.

```java
public interface MyInterface {
    void method1();

    default void method2() {
        // Default implementation
    }
}
```

## item22: Use interfaces only to define types

Interfaces should only be used for defining types, not for attaching behavior or
data to a class.

```java
public interface Drawable {
    void draw();
}
```

## item23: Prefer class hierarchies to tagged classes

Tagged classes are not efficient and difficult to maintain. Prefer class
hierarchies with proper inheritance.

```java
// as-is
public class Shape {
    enum Type { RECTANGLE, CIRCLE }
    final Type type;

    double length;
    double width;
    double radius;

    Shape(Type type) {
        this.type = type;
    }
}

// to-be
public abstract class Shape {
    abstract double area();
}

public class Rectangle extends Shape {
    private final double length;
    private final double width;

    public Rectangle(double length, double width) {
        this.length = length;
        this.width = width;
    }

    @Override
    double area() {
        return length * width;
    }
}

public class Circle extends Shape {
    private final double radius;

    public Circle(double radius) {
        this.radius = radius;
    }

    @Override
    double area() {
        return Math.PI * radius * radius;
    }
}
```

## item24: Favor static member classes over nonstatic

Static member classes do not have an implicit reference to the outer class,
avoiding memory leaks and performance issues.

```java
public class OuterClass {
    public static class NestedStaticClass {
    }
}
```

## item25: Limit source files to a single top-level class

To prevent confusion and improve readability, ensure each source file contains
only one top-level class. Nested and inner classes should be related to the
top-level class.

```java
public class MyClass {
}

// Don't do this in the same source file
public class AnotherClass {
}
```

# Generics

## item26: Don’t use raw types

Raw types do not provide type checking and should be avoided. Use parameterized
types for type safety.

```java
List<String> myList = new ArrayList<>();
```

## item27: Eliminate unchecked warnings

Unchecked warnings are a potential source of bugs. Always ensure your code is
type safe.

```java
List<String> myList = new ArrayList<>(); // Safe, no warning
```

## item28: Prefer lists to arrays

Lists provide type safety and flexibility, while arrays do not. Prefer using
lists over arrays.

```java
List<String> myList = new ArrayList<>();
```

## item29: Favor generic types

Using generic types provides flexibility, reusability, and type safety.

```java
public class Box<T> {
    private T item;

    public T getItem() {
        return item;
    }

    public void setItem(T item) {
        this.item = item;
    }
}
```

## item30: Favor generic methods

Like generic types, generic methods provide flexibility, type safety, and
reusability.

```java
public static <T> boolean contains(Collection<T> c, T item) {
    for (T element : c) {
        if (element.equals(item)) {
            return true;
        }
    }
    return false;
}
```

## item31: Use bounded wildcards to increase API flexibility

Bounded wildcards make your API more flexible by allowing it to accept subtypes
of the type parameter.

```java
public void processElements(List<? extends Number> numbers) {
    // Process elements
}
```

## item32: Combine generics and varargs judiciously

Combining generics and varargs can cause issues with type safety. Use the
`@SafeVarargs` annotation when appropriate.

```java
@SafeVarargs
public static <T> List<T> asList(T... elements) {
    return Arrays.asList(elements);
}
```

## item33: Consider typesafe heterogeneous containers

A typesafe heterogeneous container allows you to store objects of different
types while maintaining type safety.

```java
// HeterogeneousContainer.java
public class HeterogeneousContainer {
    private Map<Class<?>, Object> entries = new HashMap<>();

    public <T> void put(Class<T> type, T instance) {
        entries.put(type, instance);
    }

    public <T> T get(Class<T> type) {
        return type.cast(entries.get(type));
    }
}

// MainApp.java
public class MainApp {
    public static void main(String[] args) {
        // Create a new HeterogeneousContainer instance
        HeterogeneousContainer container = new HeterogeneousContainer();

        // Store instances of different class types in the container
        container.put(String.class, "Hello, World!");
        container.put(Integer.class, 42);
        container.put(Double.class, 3.14);

        // Retrieve the instances from the container
        String strValue = container.get(String.class);
        Integer intValue = container.get(Integer.class);
        Double dblValue = container.get(Double.class);

        // Print the retrieved instances
        System.out.println("String: " + strValue);
        System.out.println("Integer: " + intValue);
        System.out.println("Double: " + dblValue);
    }
}
```

# Enums and Annotations

## item34: Use enums instead of int constants

Enums provide a type-safe way to represent fixed sets of constants. By using
enums instead of int constants, we can improve readability and safety of code.
Enums also provide useful methods like `values()` and `valueOf()`.

```java
public enum Day {
    MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY;
}
```

## item35: Use instance fields instead of ordinals

Ordinals are the numerical position of an enum constant in its type. It's better
to use instance fields instead of relying on ordinals to assign
constant-specific behavior.

```java
public enum Planet {
    MERCURY(3.303e+23, 2.4397e6),
    // ...

    private final double mass;
    private final double radius;

    Planet(double mass, double radius) {
        this.mass = mass;
        this.radius = radius;
    }

    // ...
}
```

## item36: Use EnumSet instead of bit fields

EnumSet is a compact and efficient way to represent a set of enums. It provides
better type safety and readability compared to bit fields.

```java
EnumSet<Day> weekend = EnumSet.of(Day.SATURDAY, Day.SUNDAY);
```

## item37: Use EnumMap instead of ordinal indexing

`EnumMap` is a specialized `Map` implementation for keys of enum types. They provide
type safety and performance advantages over ordinal indexing using arrays or
Lists.

```java
Map<Day, String> dayTaskMap = new EnumMap<>(Day.class);
```

## item38: Emulate extensible enums with interfaces

Although enums themselves can't be extended, you can use an interface to group
them together and achieve extensible enums.

```java
public interface Operation {
    double apply(double x, double y);
}

public enum BasicOperation implements Operation {
    PLUS("+") {
        public double apply(double x, double y) { return x + y; }
    },
    //...
}
```

## item39: Prefer annotations to naming patterns

Use annotations to provide metadata, instead of relying on naming patterns.
Annotations are more reliable and readable compared to string-based patterns.

```java
@Test
public void exampleTestMethod() {
    // ...
}
```

## item40: Consistently use the Override annotation

Using the `@Override` annotation helps to ensure you're correctly overriding a
superclass method, preventing errors due to typos or changes in the superclass.

```java
@Override
public boolean equals(Object o) {
    // ...
}
```

## item41: Use marker interfaces to define types

Marker interfaces can declare a type that carries no specific methods but
provides information about the instances that implement them.

```java
public interface Serializable {
}
```

# Lambdas and Streams

## item42: Prefer lambdas to anonymous classes

Lambdas provide a more concise and readable way to define simple function
objects compared to anonymous classes.

```java
Collections.sort(words, (s1, s2) -> Integer.compare(s1.length(), s2.length()));
```

## item43: Prefer method references to lambdas

Method references provide a more readable way to refer directly to an existing
method or constructor.

```java
Collections.sort(words, Comparator.comparingInt(String::length));
```

## item44: Favor the use of standard functional interfaces

There are standard functional interfaces available in `java.util.function` package
which can be used instead of creating custom ones, improving code readability
and interoperability.

```java
UnaryOperator<String> caseInsensitiveTrim = s -> s.toLowerCase().trim();
```

## item45: Use streams judiciously

Streams provide a better way to work with sequences of data but should not be
overused. Use them when it would result in clearer and more maintainable code.

```java
long count = primes.stream().filter(p -> p > 1000).count();
```

## item46: Prefer side-effect-free functions in streams

While using streams, try to use side-effect-free functions to avoid unexpected
behaviors due to pipeline interference.

```java
words.stream()
     .map(String::toLowerCase)
     .distinct()
     .forEach(System.out::println);
```

## item47: Prefer Collection to Stream as a return type

Using collection types (e.g., `List`, `Set`) as return types is more flexible
and user-friendly than returning streams directly.

```java
public List<String> getNames() {
    return names.stream().collect(Collectors.toList());
}
```

## item48: Use caution when making streams parallel

Parallel streams can improve performance, but they need caution when used; make
sure that the source data type, functions and terminal operations used are
thread-safe and order-independent.

```java
primes.parallelStream().filter(p -> p > 1000).count();
```

# Methods

## item49: Check parameters for validity

Checking method parameters for validity at the start of the method can help
catch bugs early and provide better error messages.

```java
public void setName(String name) {
    if (name == null || name.isEmpty()) {
        throw new IllegalArgumentException("Name cannot be null or empty");
    }
    this.name = name;
}
```

## item50: Make defensive copies when needed

Defensive copies can help to ensure that a class doesn't get violated by
malicious or accidentally incorrect code.

```java
public final class Period {
    private final Date start;
    private final Date end;

    public Period(Date start, Date end) {
        this.start = new Date(start.getTime());
        this.end = new Date(end.getTime());

        if (this.start.compareTo(this.end) > 0) {
            throw new IllegalArgumentException("Start date is after the end date");
        }
    }
}
```

## item51: Design method signatures carefully

Use appropriate parameter types, choose method names wisely and follow
conventions to make the method signatures useful and consistent with existing
practices.

```java
public List<Customer> findCustomers(String name, int minAge, int maxAge) {
    // ...
}
```

## item52: Use overloading judiciously

Overloading can lead to confusion when it's not used wisely. Do not use
overloading with similar function signatures; avoid using overloading that may
cause ambiguity.

```java
public void print(int value) {
    System.out.println(value);
}

public void print(double value) {
    System.out.println(value);
}
```

## item53: Use varargs judiciously

Varargs can be convenient for methods that require a variable number of
arguments of the same type; use them appropriately and avoid performance issues
with their excessive use.

```java
public static int sum(int... numbers) {
    int sum = 0;
    for (int number : numbers) {
        sum += number;
    }
    return sum;
}
```

## item54: Return empty collections or arrays, not nulls

Return empty collections or arrays instead of null values to avoid potential
NullPointerExceptions and reduce complexity.

```java
public List<String> getNames() {
    return names != null ? names : Collections.emptyList();
}
```

## item55: Return optionals judiciously

Use `Optional` return types when there is a clear need to represent the absence
of a value, but avoid using them unnecessarily, as they can add complexity.

```java
public Optional<String> findFirstName() {
    return names.stream().findFirst();
}
```

## item56: Write doc comments for all exposed API elements

Properly document all the methods, classes, and other API elements that are
exposed to users, using Javadoc comments.

```java
/**
 * Represents a person with a name and age.
 */
public class Person {
}
```

# General Programming

## item57: Minimize the scope of local variables

Limit the scope of local variables by declaring them where they're used and
initializing them at the time of declaration.

```java
for (int i = 0; i < 10; i++) {
    System.out.println(i);
}
```

## item58: Prefer for-each loops to traditional for loops

For-each loops are more readable and less error-prone compared to traditional
for loops, especially when iterating over collections or arrays.

```java
for (String name : names) {
    System.out.println(name);
}
```

## item59: Know and use the libraries

Familiarize yourself with the standard Java libraries and use them instead of
reinventing the wheel.

```java
List<Integer> integers = new ArrayList<>(Arrays.asList(1, 2, 3, 4, 5));
```

## item60: Avoid float and double if exact answers are required

Floating-point types like `float` and `double` should not be used if exact
answers are required, like for monetary calculations. Use `BigDecimal`, `int`,
or `long` instead.

```java
public BigDecimal calculateTotal(BigDecimal price, int quantity) {
    return price.multiply(BigDecimal.valueOf(quantity));
}
```

## item61: Prefer primitive types to boxed primitives

Prefer using primitive types rather than their boxed counterparts (e.g., `int`
over `Integer`) to avoid potential performance and correctness issues.

```java
int i = 42;
```

## item62: Avoid strings where other types are more appropriate

Strings are often overused as placeholders for other types when it's more
appropriate to use a specific type.

```java
public enum Status {
    NEW, ACTIVE, SUSPENDED, CLOSED;
}
```

## item63: Beware the performance of string concatenation

When concatenating a large number of strings, prefer to use `StringBuilder` or
`StringBuffer` over the + operator for better performance.

```java
StringBuilder sb = new StringBuilder();
sb.append("Hello").append(" ").append("World");
String result = sb.toString();
```

## item64: Refer to objects by their interfaces

When declaring a reference, use the interface type instead of the implementation
class type if possible, allowing for more flexibility in choosing the actual
implementation.

```java
List<String> names = new ArrayList<>();
```

## item65: Prefer interfaces to reflection

Reflection provides a way to access class data and its methods at runtime, but
its overuse can lead to slow, brittle and insecure code. Prefer using interfaces
when possible.

```java
public class MyClass implements MyInterface {
    // ...
}
```

## item66: Use native methods judiciously

Native methods can provide a way to call platform-specific code, but they should
be used with caution since they can lead to platform-dependent and
hard-to-maintain code.

```java
public class MyNative {
    public native void callNativeMethod();
}
```

## item67: Optimize judiciously

Optimization can improve the performance of your code, but it's important to
balance this with maintainability and readability. Focus on writing clear,
simple code and optimizing only when necessary.

```java
// Instead of writing a custom sorting algorithm:
void customSort(int[] arr) {
  // ...Complex implementation...
}

// Use built-in sort function:
Arrays.sort(arr);
```

## item68: Adhere to generally accepted naming conventions

Following standard naming conventions in Java makes the code more understandable
and maintainable, especially when working in teams.

```java
public class Student {
  private static final int MAX_AGE = 30;
  private String firstName;
  private String lastName;

  public void setFirstName(String firstName) {
    this.firstName = firstName;
  }

  public void setLastName(String lastName) {
    this.lastName = lastName;
  }
}
```

# Exceptions

## item69: Use exceptions only for exceptional conditions

Exceptions should be used for situations where the normal flow of the
application is disrupted. Using exceptions for control flow can negatively
impact performance and readability.

```java
// Bad practice
try {
  int result = divide(10, 0);
} catch (ArithmeticException e) {
  System.out.println("Cannot divide by zero");
}

// Good practice
if (divisor != 0) {
  int result = divide(10, 0);
} else {
  System.out.println("Cannot divide by zero");
}
```

## item70: Use checked exceptions for recoverable conditions and runtime exceptions for programming errors

Checked exceptions are exceptions that the program should handle, whereas
runtime exceptions result from programming errors and should not be caught.

```java
// Use checked exception for recoverable condition
public class InsufficientFundsException extends Exception {}

// Use runtime exception for programming error
public class NullPointerException extends RuntimeException {}
```

## item71: Avoid unnecessary use of checked exceptions
## item72: Favor the use of standard exceptions
## item73: Throw exceptions appropriate to the abstraction
## item74: Document all exceptions thrown by each method
## item75: Include failure-capture information in detail messages
## item76: Strive for failure atomicity
## item77: Don’t ignore exceptions
# Concurrency
## item78: Synchronize access to shared mutable data
## item79: Avoid excessive synchronization
## item80: Prefer executors, tasks, and streams to threads
## item81: Prefer concurrency utilities to wait and notify
## item82: Document thread safety
## item83: Use lazy initialization judiciously
## item84: Don’t depend on the thread scheduler
# Serialization
## item85: Prefer alternatives to Java serialization
## item86: Implement Serializable with great caution
## item87: Consider using a custom serialized form
## item88: Write readObject methods defensively
## item89: For instance control, prefer enum types to readResolve
## item90: Consider serialization proxies instead of serialized instances
