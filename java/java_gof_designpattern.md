- [References](#references)
- [Materials](#materials)
- [Creational Pattern](#creational-pattern)
  - [Abstract Factory](#abstract-factory)
  - [Builder](#builder)
    - [Basic](#basic)
    - [Lombok](#lombok)
  - [Factory Method](#factory-method)
  - [Prototype](#prototype)
  - [Singleton](#singleton)
- [Structural Pattern](#structural-pattern)
  - [Adapter](#adapter)
  - [Bridge](#bridge)
  - [Composite](#composite)
  - [Decorator](#decorator)
  - [Facade](#facade)
  - [Flyweight](#flyweight)
  - [Proxy](#proxy)
- [Behaviorial Pattern](#behaviorial-pattern)
  - [Chain of Resp.](#chain-of-resp)
  - [Command](#command)
  - [Interpreter](#interpreter)
  - [Iterator](#iterator)
  - [Mediator](#mediator)
  - [Memento](#memento)
  - [Observer](#observer)
  - [State](#state)
  - [Strategy](#strategy)
  - [Template](#template)
  - [Visitor](#visitor)

----

# References

* [GOF Design Pattern](/gofdesignpattern/README.md)

# Materials

* [Java Design Pattern | refactoring.guru](https://refactoring.guru/design-patterns/java)

# Creational Pattern

## Abstract Factory

[Abstract Factory Pattern](/gofdesignpattern/abstractfactory/abstractfactory.md)

Abstract Factory is a design pattern that provides an interface for creating
families of related or dependent objects without specifying their concrete
classes.

```java
interface Plant {}

class OrangePlant implements Plant {}

class ApplePlant implements Plant {}

interface PlantFactory {
    Plant makePlant();
}

class AppleFactory implements PlantFactory {
    public Plant makePlant() {
        return new ApplePlant();
    }
}

class OrangeFactory implements PlantFactory {
    public Plant makePlant() {
        return new OrangePlant();
    }
}

enum PlantFactoryType {
    APPLE,
    ORANGE;

    PlantFactory createFactory() {
        switch (this) {
            case APPLE:
                return new AppleFactory();
            case ORANGE:
                return new OrangeFactory();
            default:
                throw new IllegalStateException("Invalid plant factory type: " + this);
        }
    }
}

public class Main {
    public static void main(String[] args) {
        PlantFactory plantFactory = PlantFactoryType.ORANGE.createFactory();
        Plant plant = plantFactory.makePlant();
        System.out.println("Created plant: " + plant.getClass().getSimpleName());
    }
}
```

## Builder

[Builder Pattern](/gofdesignpattern/builder/builder.md)

The Builder pattern is a creational design pattern that's used when
constructing complex objects. It lets you separate the construction of objects
and their representations, so that you can create different objects using the
same construction process.

### Basic

```java
public class Person {
    private final String id;
    private final String pw;
    private final String name;
    private final String address;
    private final String email;

    private Person(Builder builder) {
        this.id = builder.id;
        this.pw = builder.pw;
        this.name = builder.name;
        this.address = builder.address;
        this.email = builder.email;
    }

    public static class Builder {
        private final String id;
        private final String pw;
        private String name;
        private String address;
        private String email;

        public Builder(String id, String pw) {
            this.id = id;
            this.pw = pw;
        }

        public Builder withName(String name) {
            this.name = name;
            return this;
        }

        public Builder withAddress(String address) {
            this.address = address;
            return this;
        }

        public Builder withEmail(String email) {
            this.email = email;
            return this;
        }

        public Person build() {
            return new Person(this);
        }
    }

    @Override
    public String toString() {
        return "Person{" +
                "id='" + id + '\'' +
                ", pw='" + pw + '\'' +
                ", name='" + name + '\'' +
                ", address='" + address + '\'' +
                ", email='" + email + '\'' +
                '}';
    }

    public static void main(String[] args) {
        Person person = new Person.Builder("AABBCCDD", "123456")
                .withName("iamslash")
                .withAddress("Irving Ave")
                .withEmail("iamslash@gmail.com")
                .build();

        System.out.println(person);
    }
}
```

### Lombok

```java
@Builder
public class Person {
    private String id;
    private String pw;
    private String name;
    private String address;
    private String email;
}

Person person = new Person.builder("AABBCCDD", "123456")
                          .name("iamslash")
                          .address("Irving Ave")
                          .email("iamslash@gmail.com")
                          .build();
```

## Factory Method

[Factory Method Pattern](/gofdesignpattern/factorymethod/factorymethod.md)

The Factory Method Pattern provides an interface for creating objects in a
superclass, but allows subclasses to alter the type of objects that will be
created. It promotes loose coupling by removing the direct dependencies between
classes and follows the concept of "programming to an interface".

```java
public interface Shape {
    void draw();
}

public class Circle implements Shape {
    @Override
    public void draw() {
        System.out.println("Drawing a circle.");
    }
}

public class Rectangle implements Shape {
    @Override
    public void draw() {
        System.out.println("Drawing a rectangle.");
    }
}

public class Triangle implements Shape {
    @Override
    public void draw() {
        System.out.println("Drawing a triangle.");
    }
}

public abstract class ShapeFactory {
    public abstract Shape createShape();
}

public class CircleFactory extends ShapeFactory {
    @Override
    public Shape createShape() {
        return new Circle();
    }
}

public class RectangleFactory extends ShapeFactory {
    @Override
    public Shape createShape() {
        return new Rectangle();
    }
}

public class TriangleFactory extends ShapeFactory {
    @Override
    public Shape createShape() {
        return new Triangle();
    }
}

public class FactoryMethodPatternDemo {
    public static void main(String[] args) {
        // Create a circle factory and create a circle
        ShapeFactory circleFactory = new CircleFactory();
        Shape circle = circleFactory.createShape();
        circle.draw();

        // Create a rectangle factory and create a rectangle
        ShapeFactory rectangleFactory = new RectangleFactory();
        Shape rectangle = rectangleFactory.createShape();
        rectangle.draw();

        // Create a triangle factory and create a triangle
        ShapeFactory triangleFactory = new TriangleFactory();
        Shape triangle = triangleFactory.createShape();
        triangle.draw();
    }
}
```

## Prototype

[Prototype Pattern](/gofdesignpattern/prototype/prototype.md)

The prototype pattern is a creational design pattern that allows you to create
new objects by copying an existing object, called the prototype. This pattern is
especially useful when creating a new object instance is expensive in terms of
resources or time, and the existing object shares most of its properties with
the new object.

```java
public abstract class Car implements Cloneable {
    private String make;
    private String model;
    private String color;

    public Car(String make, String model, String color) {
        this.make = make;
        this.model = model;
        this.color = color;
    }

    public String getMake() {
        return make;
    }

    public String getModel() {
        return model;
    }

    public String getColor() {
        return color;
    }

    public void setColor(String color) {
        this.color = color;
    }

    @Override
    protected Object clone() {
        try {
            return super.clone();
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException("Failed to clone object", e);
        }
    }
}

public class Main {
    public static void main(String[] args) {
        Car car1 = new Car("Toyota", "Camry", "Blue");

        // Create a new object using the prototype
        Car car2 = (Car) car1.clone();

        // Verify the cloned object's properties
        System.out.println("Car1 Make: " + car1.getMake());
        System.out.println("Car1 Model: " + car1.getModel());
        System.out.println("Car1 Color: " + car1.getColor());
        System.out.println("-------------");
        System.out.println("Car2 Make: " + car2.getMake());
        System.out.println("Car2 Model: " + car2.getModel());
        System.out.println("Car2 Color: " + car2.getColor());

        // Modify the cloned object's color
        car2.setColor("Red");

        // Verify that cloned object's properties are modified
        System.out.println("-------------");
        System.out.println("Car1 Color: " + car1.getColor());
        System.out.println("Car2 Color: " + car2.getColor());
    }
}
```

## Singleton

[Singleton Design Pattern](/gofdesignpattern/singleton/singleton.md)

Singleton pattern is a design pattern that restricts the instantiation of a
class to only one instance, which is globally accessible in an application. This
is useful when exactly one object is needed to coordinate actions across the
system.

```java
public class Singleton {

    // Create a private static instance of the class
    private static Singleton instance;

    // Make the constructor private to prevent instantiation from outside the class
    private Singleton() {}

    // Provide a public method to access the singleton instance
    public static Singleton getInstance() {
        if (instance == null) {
            instance = new Singleton();
        }
        return instance;
    }

    // An example of a method
    public void printMessage() {
        System.out.println("Hello from Singleton!");
    }

    // Test the Singleton
    public static void main(String[] args) {
        Singleton singleton = Singleton.getInstance();
        singleton.printMessage();
    }
}
```

# Structural Pattern

## Adapter

[Adapter Design Pattern](/gofdesignpattern/adapter/adapter.md)

The Adapter Pattern is a structural design pattern that allows two incompatible
interfaces to work together. It acts as a wrapper between old class and new
class, translating requests and responses between the old and new systems. By
creating a new class that performs the necessary translation or adaptation, the
old class can stay unchanged, and the new class can focus on fulfilling the
required functionality.

```java
interface Temperature {
    double getCelsiusTemperature();
}

class CelsiusTemperature implements Temperature {
    private double temperature;
    public CelsiusTemperature(double temperature) {
        this.temperature = temperature;
    }
    @Override
    public double getCelsiusTemperature() {
        return temperature;
    }
}

class FahrenheitTemperature {
    private double temperature;
    public FahrenheitTemperature(double temperature) {
        this.temperature = temperature;
    }
    public double getFahrenheitTemperature() {
        return temperature;
    }
}

class FahrenheitToCelsiusAdapter implements Temperature {
    private FahrenheitTemperature fahrenheitTemperature;
    public FahrenheitToCelsiusAdapter(FahrenheitTemperature fahrenheitTemperature) {
        this.fahrenheitTemperature = fahrenheitTemperature;
    }
    @Override
    public double getCelsiusTemperature() {
        return (fahrenheitTemperature.getFahrenheitTemperature() - 32) * 5.0 / 9.0;
    }
}

public class AdapterPatternDemo {
    public static void main(String[] args) {
        Temperature celsiusTemp = new CelsiusTemperature(100);
        System.out.println("Celsius Temperature: " + celsiusTemp.getCelsiusTemperature());
        
        FahrenheitTemperature fahrenheitTemp = new FahrenheitTemperature(212);
        Temperature fahrenheitToCelsiusTemp = 
                new FahrenheitToCelsiusAdapter(fahrenheitTemp);
        System.out.println("Fahrenheit Temperature in Celsius: " + 
                fahrenheitToCelsiusTemp.getCelsiusTemperature());
    }
}
```

## Bridge

[Bridge Design Pattern](/gofdesignpattern/bridge/bridge.md)

The Bridge pattern is a structural design pattern that is used to decouple an
abstraction from its implementation, allowing the two to evolve independently.
This pattern involves an interface (the bridge) that makes the functionality
available to clients, and multiple concrete implementations of that interface.
The clients interact with the abstraction provided by the interface and are
unaware of the concrete implementation.

```java
// Abstraction
interface Color {
    void applyColor();
}

// Implementor
class RedColor implements Color {
    public void applyColor() {
        System.out.println("Applying red color.");
    }
}

class BlueColor implements Color {
    public void applyColor() {
        System.out.println("Applying blue color.");
    }
}

// Bridge
abstract class Shape {
    protected Color color;

    public Shape(Color color) {
        this.color = color;
    }

    abstract void draw();
}

class Circle extends Shape {
    public Circle(Color color) {
        super(color);
    }

    void draw() {
        System.out.print("Drawing circle: ");
        color.applyColor();
    }
}

class Square extends Shape {
    public Square(Color color) {
        super(color);
    }

    void draw() {
        System.out.print("Drawing square: ");
        color.applyColor();
    }
}

// Client
public class BridgePatternDemo {
    public static void main(String[] args) {
        Shape redCircle = new Circle(new RedColor());
        Shape blueSquare = new Square(new BlueColor());

        redCircle.draw();
        blueSquare.draw();
    }
}
```

## Composite

[Composite Design Pattern](/gofdesignpattern/composite/composite.md)

Composite pattern is a structural design pattern that allows you to compose
objects into tree structures to represent part-whole hierarchies. It lets
clients treat individual objects and compositions uniformly.

```java
import java.util.ArrayList;
import java.util.List;

// Component
interface FileSystem {
    void showFileSystem();
}

// Leaf
class File implements FileSystem {
    private String fileName;

    public File(String fileName) {
        this.fileName = fileName;
    }

    @Override
    public void showFileSystem() {
        System.out.println("File: " + fileName);
    }
}

// Composite
class Directory implements FileSystem {
    private String directoryName;
    private List<FileSystem> fileSystemList = new ArrayList<>();

    public Directory(String directoryName) {
        this.directoryName = directoryName;
    }

    public void addFileSystem(FileSystem fileSystem) {
        fileSystemList.add(fileSystem);
    }

    @Override
    public void showFileSystem() {
        System.out.println("\nDirectory: " + directoryName);
        for (FileSystem fileSystem : fileSystemList) {
            fileSystem.showFileSystem();
        }
    }
}

public class CompositePatternDemo {

    public static void main(String[] args) {
        Directory rootDirectory = new Directory("Root");

        Directory dir1 = new Directory("Directory1");
        dir1.addFileSystem(new File("File1.txt"));
        dir1.addFileSystem(new File("File2.txt"));
        rootDirectory.addFileSystem(dir1);

        Directory dir2 = new Directory("Directory2");
        dir2.addFileSystem(new File("File3.txt"));
        dir2.addFileSystem(new File("File4.txt"));
        rootDirectory.addFileSystem(dir2);

        rootDirectory.showFileSystem();
    }
}
```

## Decorator

[Decorator Design Pattern](/gofdesignpattern/decorator/decorator.md)

The decorator pattern is a structural design pattern that allows adding new
functionality to an object without altering its structure. It involves a set of
decorator classes that are used to wrap concrete components. The decorator
classes mirror the type of components they can decorate but add or override
behavior.

```java
// Component interface
interface Car {
    String getDescription();
    double getCost();
}

// Concrete Component
class BasicCar implements Car {
    @Override
    public String getDescription() {
        return "Basic Car";
    }

    @Override
    public double getCost() {
        return 10000;
    }
}

// Decorator abstract class
abstract class CarDecorator implements Car {
    final Car car;

    CarDecorator(Car car) {
        this.car = car;
    }

    @Override
    public String getDescription() {
        return car.getDescription();
    }

    @Override
    public double getCost() {
        return car.getCost();
    }
}

// Concrete Decorator classes
class LuxuryPackage extends CarDecorator {
    LuxuryPackage(Car car) {
        super(car);
    }

    @Override
    public String getDescription() {
        return car.getDescription() + ", Luxury Package";
    }

    @Override
    public double getCost() {
        return car.getCost() + 5000;
    }
}

class SportPackage extends CarDecorator {
    SportPackage(Car car) {
        super(car);
    }

    @Override
    public String getDescription() {
        return car.getDescription() + ", Sport Package";
    }

    @Override
    public double getCost() {
        return car.getCost() + 3500;
    }
}

// Client code
class DecoratorPatternDemo {
    public static void main(String[] args) {
        Car basicCar = new BasicCar();
        
        Car luxuryCar = new LuxuryPackage(basicCar);
        System.out.println(luxuryCar.getDescription() +
                           " cost: " + luxuryCar.getCost());
        
        Car sportCar = new SportPackage(basicCar);
        System.out.println(sportCar.getDescription() +
                           " cost: " + sportCar.getCost());
        
        Car luxurySportCar = new LuxuryPackage(new SportPackage(basicCar));
        System.out.println(luxurySportCar.getDescription() +
                           " cost: " + luxurySportCar.getCost());
    }
}
```

## Facade

[Facade Design Pattern](/gofdesignpattern/facade/facade.md)

Facade Pattern is a structural design pattern that provides a simplified
interface to a complex subsystem. It involves a single class that represents an
entire subsystem, hiding its complexity from the client.

```java
// Subsystem classes
class CPU {
    void start() {
        System.out.println("Initialize CPU");
    }
}

class HardDrive {
    void read() {
        System.out.println("Read Hard Drive");
    }
}

class Memory {
    void load() {
        System.out.println("Load memory");
    }
}

// Facade
class ComputerFacade {
    private CPU cpu = new CPU();
    private HardDrive hardDrive = new HardDrive();
    private Memory memory = new Memory();
    
    void startComputer() {
        cpu.start();
        memory.load();
        hardDrive.read();
    }
}

// Client
public class Main {
    public static void main(String[] args) {
        ComputerFacade computerFacade = new ComputerFacade();
        computerFacade.startComputer();
    }
}
```

## Flyweight

[Flyweight Design Pattern](/gofdesignpattern/flyweight/flyweight.md)

The Flyweight pattern is a structural design pattern that minimizes memory usage
by sharing articles of data among several objects, especially when dealing with
a large number of similar objects. The main idea is to create shared objects,
called "Flyweights," which can be reused representing the same state.

```java
import java.util.HashMap;
import java.util.Map;

// The interface representing the shared objects (Flyweight)
interface Shape {
    void draw(int x, int y);
}

// Concrete Flyweight objects implementing the interface
class Circle implements Shape {
    private String color;

    public Circle(String color) { this.color = color; }

    @Override
    public void draw(int x, int y) {
        System.out.println("Drawing a " + color + " circle at (" + x + "," + y + ")");
    }
}

// Flyweight factory managing the creation and sharing of the objects
class ShapeFactory {
    private static final Map<String, Shape> shapes = new HashMap<>();

    public static Shape getCircle(String color) {
        if (!shapes.containsKey(color)) {
            shapes.put(color, new Circle(color));
            System.out.println("Creating new " + color + " circle");
        }
        return shapes.get(color);
    }
}

public class FlyweightDemo {
    public static void main(String[] args) {
        Shape redCircle = ShapeFactory.getCircle("red");
        redCircle.draw(10, 20);

        Shape blueCircle = ShapeFactory.getCircle("blue");
        blueCircle.draw(30, 40);

        Shape redCircle2 = ShapeFactory.getCircle("red");
        redCircle2.draw(50, 60);
    }
}
```

## Proxy

[Proxy Deisgn Pattern](/gofdesignpattern/proxy/proxy.md)

Proxy Pattern is a structural design pattern that involves a set of intermediary
objects, or proxies, that sit between a client and the actual target object.
Proxies are useful for controlling access, managing resource-intensive
operations, or adding extra functionality to objects without adding complexity
to the real object.

```java
interface RealObject {
    void performAction();
}

class ActualObject implements RealObject {
    public void performAction() {
        System.out.println("Performing action in the actual object");
    }
}

class ProxyObject implements RealObject {
    private RealObject realObject;

    public ProxyObject(RealObject realObject) {
        this.realObject = realObject;
    }

    public void performAction() {
        System.out.println("Performing action in the proxy object");
        realObject.performAction();
    }
}

public class Main {
    public static void main(String[] args) {
        RealObject actualObject = new ActualObject();
        RealObject proxyObject = new ProxyObject(actualObject);
        proxyObject.performAction(); // using proxy object instead of the actual object
    }
}
```

# Behaviorial Pattern

## Chain of Resp.

[Chain of Resp.](/gofdesignpattern/chainofresp/chainofresp.md)

The Chain of Responsibility design pattern is a behavioral pattern that allows
an object to send a request to a chain of potential handlers, where each handler
in the chain decides whether to process the request or pass it to the next
handler in the chain. The pattern is useful when there are multiple potential
handlers to decide which one should handle a specific request.

```java
interface Handler {
    void setNext(Handler handler);
    void handle(int request);
}

class ConcreteHandlerA implements Handler {
    private Handler next;

    public void setNext(Handler handler) {
        this.next = handler;
    }

    public void handle(int request) {
        if (request <= 10) {
            System.out.println("Handled by A");
        } else if (next != null) {
            next.handle(request);
        }
    }
}

class ConcreteHandlerB implements Handler {
    private Handler next;

    public void setNext(Handler handler) {
        this.next = handler;
    }

    public void handle(int request) {
        if (request > 10 && request <= 20) {
            System.out.println("Handled by B");
        } else if (next != null) {
            next.handle(request);
        }
    }
}

public class Main {
    public static void main(String[] args) {
        Handler handlerA = new ConcreteHandlerA();
        Handler handlerB = new ConcreteHandlerB();

        handlerA.setNext(handlerB);

        handlerA.handle(5);
        handlerA.handle(15);
        handlerA.handle(25);
    }
}
```

## Command

[Command Design Pattern](/gofdesignpattern/command/command.md)

The Command pattern is a behavioral design pattern that turns a request into a
stand-alone object that contains all information about the request. This
transformation lets you pass requests as method arguments, delay or queue a
request’s execution, and support undoable operations.

```java
interface Command {
  void execute();
}

class Receiver {
  void actionA() { System.out.println("Action A"); }
  void actionB() { System.out.println("Action B"); }
}

class ConcreteCommandA implements Command {
  private Receiver receiver;
  ConcreteCommandA(Receiver receiver) { this.receiver = receiver; }
  public void execute() { receiver.actionA(); }
}

class ConcreteCommandB implements Command {
  private Receiver receiver;
  ConcreteCommandB(Receiver receiver) { this.receiver = receiver; }
  public void execute() { receiver.actionB(); }
}

class Invoker {
  private Command command;
  void setCommand(Command command) { this.command = command; }
  void executeCommand() { command.execute(); }
}

public class Main {
  public static void main(String[] args) {
    Receiver receiver = new Receiver();
    Command commandA = new ConcreteCommandA(receiver);
    Command commandB = new ConcreteCommandB(receiver);

    Invoker invoker = new Invoker();
    invoker.setCommand(commandA);
    invoker.executeCommand();

    invoker.setCommand(commandB);
    invoker.executeCommand();
  }
}
```

## Interpreter

[Interpreter Design Pattern](/gofdesignpattern/interpreter/interpreter.md)

Interpreter pattern is a design pattern that provides a way to evaluate language
grammar or expressions. It involves building an interpreter to interpret the
expressions of a language. This pattern is used in situations where there is a
language to be interpreted or domain-specific language expression, which must be
evaluated.

```java
import java.util.HashMap;
import java.util.Map;

interface Expression {
    boolean interpret(Map<String, Boolean> context);
}

class Constant implements Expression {
    private boolean value;

    public Constant(boolean value) {
        this.value = value;
    }

    public boolean interpret(Map<String, Boolean> context) {
        return value;
    }
}

class Variable implements Expression {
    private String name;

    public Variable(String name) {
        this.name = name;
    }

    public boolean interpret(Map<String, Boolean> context) {
        return context.get(name);
    }
}

class And implements Expression {
    private Expression left, right;

    public And(Expression left, Expression right) {
        this.left = left;
        this.right = right;
    }

    public boolean interpret(Map<String, Boolean> context) {
        return left.interpret(context) && right.interpret(context);
    }
}

class Or implements Expression {
    private Expression left, right;

    public Or(Expression left, Expression right) {
        this.left = left;
        this.right = right;
    }

    public boolean interpret(Map<String, Boolean> context) {
        return left.interpret(context) || right.interpret(context);
    }
}

public class InterpreterExample {
    public static void main(String[] args) {
        Expression a = new Variable("A");
        Expression b = new Variable("B");
        Expression expression = new Or(new And(a, b), new Or(a, new And(a, b)));

        Map<String, Boolean> context = new HashMap<>();
        context.put("A", true);
        context.put("B", false);

        System.out.println("Expression result: " + expression.interpret(context));
    }
}
```

## Iterator

[Iterator Design Pattern](/gofdesignpattern/iterator/iterator.md)

Iterator design pattern is a design pattern that provides a way to access the
elements of a collection object sequentially without exposing its underlying
representation. This pattern is useful to provide a standard way to traverse
through a group of objects and helps decouple the algorithms from the data
structures they operate upon.

```java
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

class MyCollection<T> implements Iterable<T> {
    private List<T> items = new ArrayList<>();

    public void add(T item) {
        items.add(item);
    }

    public Iterator<T> iterator() {
        return new MyIterator<>(this.items);
    }
}

class MyIterator<T> implements Iterator<T> {
    private List<T> items;
    private int currentIndex = 0;

    public MyIterator(List<T> items) {
        this.items = items;
    }

    public boolean hasNext() {
        return currentIndex < items.size();
    }

    public T next() {
        return items.get(currentIndex++);
    }
}

public class IteratorExample {
    public static void main(String[] args) {
        MyCollection<Integer> numbers = new MyCollection<>();
        numbers.add(1);
        numbers.add(2);
        numbers.add(3);

        for (Integer num : numbers) {
            System.out.println(num);
        }
    }
}
```

## Mediator

[Mediator Design Pattern](/gofdesignpattern/mediator/mediator.md)

The Mediator pattern is a behavioral design pattern that defines an object that
encapsulates how objects (colleagues) interact with each other. This pattern
promotes loose coupling by avoiding direct connections between colleagues, and
the mediator object is responsible for handling their interactions.

```java
import java.util.ArrayList;
import java.util.List;

// Mediator component
interface ChatMediator {
    void sendMessage(String msg, User user);
    void addUser(User user);
}

class ChatMediatorImpl implements ChatMediator {
    private List<User> users;

    public ChatMediatorImpl() {
        this.users = new ArrayList<>();
    }

    @Override
    public void addUser(User user) {
        this.users.add(user);
    }

    @Override
    public void sendMessage(String msg, User user) {
        for (User u : this.users) {
            if (u != user) {
                u.receive(msg);
            }
        }
    }
}

// Colleague component
abstract class User {
    protected ChatMediator mediator;
    protected String name;

    public User(ChatMediator mediator, String name) {
        this.mediator = mediator;
        this.name = name;
    }

    public abstract void send(String msg);
    public abstract void receive(String msg);
}

class UserImpl extends User {
    public UserImpl(ChatMediator mediator, String name) {
        super(mediator, name);
    }

    @Override
    public void send(String msg) {
        mediator.sendMessage(msg, this);
    }

    @Override
    public void receive(String msg) {
        System.out.println(this.name + " received: " + msg);
    }
}

// Application code
public class MediatorPattern {
    public static void main(String[] args) {
        ChatMediator mediator = new ChatMediatorImpl();

        User user1 = new UserImpl(mediator, "Alice");
        User user2 = new UserImpl(mediator, "Bob");
        User user3 = new UserImpl(mediator, "Eve");

        mediator.addUser(user1);
        mediator.addUser(user2);
        mediator.addUser(user3);

        user1.send("Hello, everyone!");
    }
}
```

## Memento

[Memento Design Pattern](/gofdesignpattern/memento/memento.md)

Memento pattern is a behavioral design pattern that allows an object
(originator) to save and restore its previous state without revealing the
structure or details of its internal state, thus enabling an undo mechanism. A
memento object is used to store the state of the originator, and a caretaker is
responsible for the memento's safe keeping.

```java
// The Memento class
class Memento {
    private final String state;

    public Memento(String state) {
        this.state = state;
    }

    public String getState() {
        return state;
    }
}

// The Originator class
class Originator {
    private String state;

    public void setState(String state) {
        this.state = state;
    }

    public Memento save() {
        return new Memento(state);
    }

    public void restore(Memento memento) {
        state = memento.getState();
    }
}

// The Caretaker class
class Caretaker {
    private final Stack<Memento> mementos = new Stack<>();

    public void save(Originator originator) {
        mementos.push(originator.save());
    }

    public void undo(Originator originator) {
        if (!mementos.isEmpty()) {
            originator.restore(mementos.pop());
        }
    }
}
```

## Observer

[Observer Design Pattern](/gofdesignpattern/observer/observer.md)

Observer Pattern is a behavioral design pattern that allows an object (subject)
to maintain a list of its dependents (observers) and notify them automatically
of any changes in state. When a state change occurs, the subject updates the
observers.

```java
import java.util.ArrayList;
import java.util.List;

interface Observer {
    void update(String message);
}

class Subject {
    private List<Observer> observers = new ArrayList<>();

    void attach(Observer observer) {
        observers.add(observer);
    }

    void notifyObservers(String message) {
        for (Observer observer : observers) {
            observer.update(message);
        }
    }
}

class ConcreteObserver implements Observer {
    private String name;

    ConcreteObserver(String name) {
        this.name = name;
    }

    @Override
    public void update(String message) {
        System.out.println(name + " received the message: " + message);
    }
}

public class ObserverPatternDemo {
    public static void main(String[] args) {
        Subject subject = new Subject();
        ConcreteObserver observer1 = new ConcreteObserver("Observer1");
        ConcreteObserver observer2 = new ConcreteObserver("Observer2");

        subject.attach(observer1);
        subject.attach(observer2);

        subject.notifyObservers("Hello, Observers!");
    }
}
```

## State

[State Design Pattern](/gofdesignpattern/state/state.md)

The State pattern is a behavioral design pattern that allows an object to change
its behavior when its internal state changes.

```java
interface State {
  void handle(Context context);
}

class ConcreteStateA implements State {
  public void handle(Context context) {
    context.changeState(new ConcreteStateB());
  }
}

class ConcreteStateB implements State {
  public void handle(Context context) {
    context.changeState(new ConcreteStateA());
  }
}

class Context {
  private State state;

  public Context(State state) {
    this.state = state;
  }

  public void changeState(State newState) {
    state = newState;
  }

  public void request() {
    state.handle(this);
  }
}

public class StatePattern {
  public static void main(String[] args) {
    Context context = new Context(new ConcreteStateA());
    context.request();
    context.request();
  }
}
```

## Strategy

[Strategy Design Pattern](/gofdesignpattern/strategy/strategy.md)

The strategy pattern is a behavioral design pattern that defines a family of
algorithms, encapsulates each of them, and makes them interchangeable. It allows
the algorithm to vary independently from the clients that use it.

```java
// Strategy interface
interface Strategy {
    int doOperation(int a, int b);
}

// Concrete Strategies
class Addition implements Strategy {
    public int doOperation(int a, int b) {
        return a + b;
    }
}

class Subtraction implements Strategy {
    public int doOperation(int a, int b) {
        return a - b;
    }
}

class Multiplication implements Strategy {
    public int doOperation(int a, int b) {
        return a * b;
    }
}

// Context class
class Context {
    private Strategy strategy;

    public Context(Strategy strategy) {
        this.strategy = strategy;
    }

    public int executeStrategy(int a, int b) {
        return strategy.doOperation(a, b);
    }
}

// Client
public class Main {
    public static void main(String[] args) {
        Context context = new Context(new Addition());
        System.out.println("10 + 5 = " + context.executeStrategy(10, 5));

        context = new Context(new Subtraction());
        System.out.println("10 - 5 = " + context.executeStrategy(10, 5));

        context = new Context(new Multiplication());
        System.out.println("10 * 5 = " + context.executeStrategy(10, 5));
    }
}
```

## Template

[Template Design Pattern](/gofdesignpattern/template/template.md)

The template pattern is a behavioral design pattern that defines the program
skeleton of an algorithm in a method, called template method, which defers some
steps to subclasses. It lets subclasses redefine certain steps of an algorithm
without changing the algorithm's structure.

```java
abstract class Game {
  abstract void initialize();
  abstract void startGame();
  abstract void endGame();

  public final void play() {
    initialize();
    startGame();
    endGame();
  }
}

class Cricket extends Game {
  void initialize() { System.out.println("Cricket Game Initialized!"); }
  void startGame() { System.out.println("Cricket Game Started!"); }
  void endGame() { System.out.println("Cricket Game Finished!"); }
}

class Football extends Game {
  void initialize() { System.out.println("Football Game Initialized!"); }
  void startGame() { System.out.println("Football Game Started!"); }
  void endGame() { System.out.println("Football Game Finished!"); }
}

public class TemplatePattern {
  public static void main(String[] args) {
    Game game = new Cricket();
    game.play();

    game = new Football();
    game.play();
  }
}
```

## Visitor

[Visitor Design Pattern](/gofdesignpattern/visitor/visitor.md)

Visitor pattern is a design pattern used in object-oriented programming that
promotes separation of concerns, allowing operations to be defined independently
on objects while traversing data structures, typically through a tree traversal
algorithm. Visitor pattern is an example of the **double dispatch technique**,
where the operation's implementation depends on both the **type of the
operation** and the **type of the data**.

```java
interface ShapeVisitor {
    void visit(Circle circle);
    void visit(Rectangle rectangle);
}

interface Shape {
    void accept(ShapeVisitor visitor);
}

class Circle implements Shape {
    double radius;

    Circle(double radius) {
        this.radius = radius;
    }

    void accept(ShapeVisitor visitor) {
        visitor.visit(this);
    }
}

class Rectangle implements Shape {
    double length, width;

    Rectangle(double length, double width) {
        this.length = length;
        this.width = width;
    }

    void accept(ShapeVisitor visitor) {
        visitor.visit(this);
    }
}

class ShapeAreaVisitor implements ShapeVisitor {
    double totalArea = 0;

    void visit(Circle circle) {
        totalArea += Math.PI * Math.pow(circle.radius, 2);
    }

    void visit(Rectangle rectangle) {
        totalArea += rectangle.length * rectangle.width;
    }
}

public class VisitorPatternExample {
    public static void main(String[] args) {
        Shape[] shapes = {new Circle(5), new Rectangle(3, 4), new Circle(2)};
        ShapeAreaVisitor areaVisitor = new ShapeAreaVisitor();

        for (Shape shape : shapes) {
            shape.accept(areaVisitor);
        }

        System.out.println("Total area: " + areaVisitor.totalArea);
    }
}
```
