- [References](#references)
- [Materials](#materials)
- [Creational Pattern](#creational-pattern)
  - [Abstract Factory](#abstract-factory)
  - [Builder](#builder)
  - [Basic](#basic)
    - [Improved](#improved)
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

* [C++ Design Pattern | refactoring.guru](https://refactoring.guru/design-patterns/cpp)

# Creational Pattern

## Abstract Factory

[Abstract Factory Pattern](/gofdesignpattern/abstractfactory/abstractfactory.md)

Abstract Factory is a design pattern that provides an interface for creating
families of related or dependent objects without specifying their concrete
classes.

```c++
#include <iostream>
#include <memory>

// Plant Interface
class Plant {
 public:
  virtual ~Plant() {}
  virtual void getName() const = 0;
};

class OrangePlant : public Plant {
 public:
  void getName() const override { 
    std::cout << "OrangePlant" << std::endl; 
  }
};

class ApplePlant : public Plant {
 public:
  void getName() const override { 
    std::cout << "ApplePlant" << std::endl; 
  }
};

// PlantFactory Interface
class PlantFactory {
 public:
  virtual ~PlantFactory() {}
  virtual std::unique_ptr<Plant> makePlant() = 0;
};

class AppleFactory : public PlantFactory {
 public:
  std::unique_ptr<Plant> makePlant() override { 
    return std::make_unique<ApplePlant>(); 
  }
};

class OrangeFactory : public PlantFactory {
 public:
  std::unique_ptr<Plant> makePlant() override { 
    return std::make_unique<OrangePlant>(); 
  }
};

enum class PlantFactoryType {
  Apple,
  Orange
};

std::unique_ptr<PlantFactory> createFactory(PlantFactoryType type) {
  switch (type) {
    case PlantFactoryType::Apple:
      return std::make_unique<AppleFactory>();
    case PlantFactoryType::Orange:
      return std::make_unique<OrangeFactory>();
  }
}

int main() {
  auto plantFactory = createFactory(PlantFactoryType::Orange);
  auto plant = plantFactory->makePlant();
  plant->getName();

  return 0;
}
```

## Builder

[Builder Pattern](/gofdesignpattern/builder/builder.md)

The Builder pattern is a creational design pattern that's used when
constructing complex objects. It lets you separate the construction of objects
and their representations, so that you can create different objects using the
same construction process.

## Basic

```c++
#include <iostream>
#include <string>

class Person {
  public:
    class Builder;

  private:
    std::string id;
    std::string pw;
    std::string name;
    std::string address;
    std::string email;

    Person(const Builder& builder)
        : id(builder.id), pw(builder.pw), name(builder.name),
          address(builder.address), email(builder.email) {}

  public:
    class Builder {
        friend class Person;

    private:
        std::string id;
        std::string pw;
        std::string name;
        std::string address;
        std::string email;

    public:
        Builder(const std::string& id, const std::string& pw) : id(id), pw(pw) {}

        Builder& withName(const std::string& name) {
            this->name = name;
            return *this;
        }

        Builder& withAddress(const std::string& address) {
            this->address = address;
            return *this;
        }

        Builder& withEmail(const std::string& email) {
            this->email = email;
            return *this;
        }

        Person build() {
            return Person(*this);
        }
    };

    void print() {
        std::cout << "ID: " << id << std::endl;
        std::cout << "PW: " << pw << std::endl;
        std::cout << "Name: " << name << std::endl;
        std::cout << "Address: " << address << std::endl;
        std::cout << "Email: " << email << std::endl;
    }
};

int main() {
    Person person = Person::Builder("AABBCCDD", "123456")
                        .withName("iamslash")
                        .withAddress("Irving Ave")
                        .withEmail("iamslash@gmail.com")
                        .build();
    person.print();
```

### Improved

Improved duplicated fields.

```c++
#include <iostream>
#include <string>

class PersonBase {
protected:
    std::string id;
    std::string pw;
    std::string name;
    std::string address;
    std::string email;
};

class Person : public PersonBase {
public:
    class Builder;

private:
    Person(const Builder& builder) {
        id = builder.id;
        pw = builder.pw;
        name = builder.name;
        address = builder.address;
        email = builder.email;
    }

public:
    class Builder : public PersonBase {
    public:
        Builder(const std::string& id, const std::string& pw) {
            this->id = id;
            this->pw = pw;
        }

        Builder& withName(const std::string& name) {
            this->name = name;
            return *this;
        }

        Builder& withAddress(const std::string& address) {
            this->address = address;
            return *this;
        }

        Builder& withEmail(const std::string& email) {
            this->email = email;
            return *this;
        }
        
        Person build() {
            return Person(*this);
        }
    };

    void print() {
        std::cout << "ID: " << id << std::endl;
        std::cout << "PW: " << pw << std::endl;
        std::cout << "Name: " << name << std::endl;
        std::cout << "Address: " << address << std::endl;
        std::cout << "Email: " << email << std::endl;
    }
};

int main() {
    Person person = Person::Builder("AABBCCDD", "123456")
                        .withName("iamslash")
                        .withAddress("Irving Ave")
                        .withEmail("iamslash@gmail.com")
                        .build();
    person.print();
    return 0;
}
```

## Factory Method

[Factory Method Pattern](/gofdesignpattern/factorymethod/factorymethod.md)

The Factory Method Pattern provides an interface for creating objects in a
superclass, but allows subclasses to alter the type of objects that will be
created. It promotes loose coupling by removing the direct dependencies between
classes and follows the concept of "programming to an interface".

```c++
#include <iostream>
#include <memory>
using namespace std;

// Shape interface (pure virtual class)
class Shape {
public:
    virtual ~Shape() {}
    virtual void draw() const = 0;
};

// Concrete implementations of Shape
class Circle : public Shape {
public:
    void draw() const override {
        cout << "Drawing a circle." << endl;
    }
};

class Rectangle : public Shape {
public:
    void draw() const override {
        cout << "Drawing a rectangle." << endl;
    }
};

class Triangle : public Shape {
public:
    void draw() const override {
        cout << "Drawing a triangle." << endl;
    }
};

// Abstract factory class
class ShapeFactory {
public:
    virtual ~ShapeFactory() {}
    virtual unique_ptr<Shape> createShape() = 0;
};

// Concrete factory classes
class CircleFactory : public ShapeFactory {
public:
    unique_ptr<Shape> createShape() override {
        return make_unique<Circle>();
    }
};

class RectangleFactory : public ShapeFactory {
public:
    unique_ptr<Shape> createShape() override {
        return make_unique<Rectangle>();
    }
};

class TriangleFactory : public ShapeFactory {
public:
    unique_ptr<Shape> createShape() override {
        return make_unique<Triangle>();
    }
};

// Client code
int main() {
    // Create a circle factory and create a circle
    unique_ptr<ShapeFactory> circleFactory = make_unique<CircleFactory>();
    unique_ptr<Shape> circle = circleFactory->createShape();
    circle->draw();

    // Create a rectangle factory and create a rectangle
    unique_ptr<ShapeFactory> rectangleFactory = make_unique<RectangleFactory>();
    unique_ptr<Shape> rectangle = rectangleFactory->createShape();
    rectangle->draw();

    // Create a triangle factory and create a triangle
    unique_ptr<ShapeFactory> triangleFactory = make_unique<TriangleFactory>();
    unique_ptr<Shape> triangle = triangleFactory->createShape();
    triangle->draw();

    return 0;
}
```

## Prototype

[Prototype Pattern](/gofdesignpattern/prototype/prototype.md)

The prototype pattern is a creational design pattern that allows you to create
new objects by copying an existing object, called the prototype. This pattern is
especially useful when creating a new object instance is expensive in terms of
resources or time, and the existing object shares most of its properties with
the new object.

```c++
#include <iostream>
#include <memory>
#include <string>

// Base class
class Car {
public:
    Car(const std::string& make, const std::string& model, const std::string& color)
        : make_(make), model_(model), color_(color) {}

    // Method to create a clone of derived classes
    virtual std::unique_ptr<Car> clone() const = 0;

    void setColor(const std::string& color) {
        color_ = color;
    }

    void print() const {
        std::cout << " - Make: " << make_ << std::endl
                  << " - Model: " << model_ << std::endl
                  << " - Color: " << color_ << std::endl;
    }

protected:
    std::string make_;
    std::string model_;
    std::string color_;
};

// Derived class
class ToyotaCar : public Car {
public:
    ToyotaCar()
        : Car("Toyota", "Camry", "Blue") {}

    std::unique_ptr<Car> clone() const override {
        return std::make_unique<ToyotaCar>(*this);
    }
};

int main() {
    // Create an initial object (car1)
    std::unique_ptr<Car> car1 = std::make_unique<ToyotaCar>();

    // Create a new object by cloning car1 (car2)
    std::unique_ptr<Car> car2 = car1->clone();

    // Print properties of both objects
    std::cout << "Car1 properties:" << std::endl;
    car1->print();
    std::cout << "Car2 properties:" << std::endl;
    car2->print();

    // Modify car2's properties
    car2->setColor("Red");

    // Print properties of both objects after modification
    std::cout << "Car1 properties after changing Car2 color:" << std::endl;
    car1->print();
    std::cout << "Car2 properties after changing its color to Red:" << std::endl;
    car2->print();

    return 0;
}
```

## Singleton

[Singleton Design Pattern](/gofdesignpattern/singleton/singleton.md)

Singleton pattern is a design pattern that restricts the instantiation of a
class to only one instance, which is globally accessible in an application. This
is useful when exactly one object is needed to coordinate actions across the
system.

```c++
#include <iostream>

class Singleton {
public:
    // Method to access the singleton instance
    static Singleton& getInstance() {
        static Singleton instance;
        return instance;
    }

    // An example of a method
    void printMessage() {
        std::cout << "Hello from Singleton!" << std::endl;
    }

    // Delete copy constructor and assignment operator
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;

private:
    // Make constructor private to prevent instantiation
    Singleton() = default;
};

// Test the Singleton
int main() {
    Singleton& singleton = Singleton::getInstance();
    singleton.printMessage();
    return 0;
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

```c++
#include <iostream>

class Temperature {
public:
    virtual double getCelsiusTemperature() = 0;
};

class CelsiusTemperature : public Temperature {
private:
    double temperature;
public:
    CelsiusTemperature(double temperature) : temperature(temperature) {}
    double getCelsiusTemperature() { return temperature; }
};

class FahrenheitTemperature {
private:
    double temperature;
public:
    FahrenheitTemperature(double temperature) : temperature(temperature) {}
    double getFahrenheitTemperature() { return temperature; }
};

class FahrenheitToCelsiusAdapter : public Temperature {
private:
    FahrenheitTemperature fahrenheitTemperature;
public:
    FahrenheitToCelsiusAdapter(FahrenheitTemperature fahrenheitTemperature)
        : fahrenheitTemperature(fahrenheitTemperature) {}
    double getCelsiusTemperature() {
        return (fahrenheitTemperature.getFahrenheitTemperature() - 32) * 5.0 / 9.0;
    }
};

int main() {
    CelsiusTemperature celsiusTemp(100);
    std::cout << "Celsius Temperature: " << celsiusTemp.getCelsiusTemperature() << std::endl;

    FahrenheitTemperature fahrenheitTemp(212);
    FahrenheitToCelsiusAdapter fahrenheitToCelsiusTemp(fahrenheitTemp);
    std::cout << "Fahrenheit Temperature in Celsius: " << fahrenheitToCelsiusTemp.getCelsiusTemperature() << std::endl;

    return 0;
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

```c++
#include <iostream>
using namespace std;

// Abstraction
class Color {
public:
    virtual void applyColor() = 0;
};

// Implementor
class RedColor : public Color {
public:
    void applyColor() {
        cout << "Applying red color." << endl;
    }
};

class BlueColor : public Color {
public:
    void applyColor() {
        cout << "Applying blue color." << endl;
    }
};

// Bridge
class Shape {
protected:
    Color* color;

public:
    Shape(Color* color) : color(color) {}
    virtual void draw() = 0;
};

class Circle : public Shape {
public:
    Circle(Color* color) : Shape(color) {}

    void draw() {
        cout << "Drawing circle: ";
        color->applyColor();
    }
};

class Square : public Shape {
public:
    Square(Color* color) : Shape(color) {}

    void draw() {
        cout << "Drawing square: ";
        color->applyColor();
    }
};

// Client
int main() {
    Shape* redCircle = new Circle(new RedColor());
    Shape* blueSquare = new Square(new BlueColor());

    redCircle->draw();
    blueSquare->draw();

    delete redCircle;
    delete blueSquare;

    return 0;
}
```

## Composite

[Composite Design Pattern](/gofdesignpattern/composite/composite.md)

Composite pattern is a structural design pattern that allows you to compose
objects into tree structures to represent part-whole hierarchies. It lets
clients treat individual objects and compositions uniformly.

```c++
#include <iostream>
#include <vector>
#include <memory>

// Component
class FileSystem {
public:
    virtual void showFileSystem() const = 0;
    virtual ~FileSystem() = default;
};

// Leaf
class File : public FileSystem {
public:
    explicit File(std::string fileName) : fileName(std::move(fileName)) {}

    void showFileSystem() const override {
        std::cout << "File: " << fileName << std::endl;
    }

private:
    std::string fileName;
};

// Composite
class Directory : public FileSystem {
public:
    explicit Directory(std::string directoryName)
        : directoryName(std::move(directoryName)) {}

    void addFileSystem(const std::shared_ptr<FileSystem>& fileSystem) {
        fileSystemList.push_back(fileSystem);
    }

    void showFileSystem() const override {
        std::cout << "\nDirectory: " << directoryName << std::endl;
        for (const auto& fileSystem : fileSystemList) {
            fileSystem->showFileSystem();
        }
    }

private:
    std::string directoryName;
    std::vector<std::shared_ptr<FileSystem>> fileSystemList;
};

int main() {
    auto rootDirectory = std::make_shared<Directory>("Root");

    auto dir1 = std::make_shared<Directory>("Directory1");
    dir1->addFileSystem(std::make_shared<File>("File1.txt"));
    dir1->addFileSystem(std::make_shared<File>("File2.txt"));
    rootDirectory->addFileSystem(dir1);

    auto dir2 = std::make_shared<Directory>("Directory2");
    dir2->addFileSystem(std::make_shared<File>("File3.txt"));
    dir2->addFileSystem(std::make_shared<File>("File4.txt"));
    rootDirectory->addFileSystem(dir2);

    rootDirectory->showFileSystem();

    return 0;
}
```

## Decorator

[Decorator Design Pattern](/gofdesignpattern/decorator/decorator.md)

The decorator pattern is a structural design pattern that allows adding new
functionality to an object without altering its structure. It involves a set of
decorator classes that are used to wrap concrete components. The decorator
classes mirror the type of components they can decorate but add or override
behavior.

```c++
#include <iostream>
#include <memory>
#include <string>

// Component interface
class Car {
public:
    virtual ~Car() {}
    virtual std::string getDescription() const = 0;
    virtual double getCost() const = 0;
};

// Concrete Component
class BasicCar : public Car {
public:
    std::string getDescription() const override {
        return "Basic Car";
    }

    double getCost() const override {
        return 10000;
    }
};

// Decorator abstract class
class CarDecorator : public Car {
public:
    CarDecorator(std::unique_ptr<Car> car)
        : car_(std::move(car)) {}

    std::string getDescription() const override {
        return car_->getDescription();
    }

    double getCost() const override {
        return car_->getCost();
    }

protected:
    std::unique_ptr<Car> car_;
};

// Concrete Decorator classes
class LuxuryPackage : public CarDecorator {
public:
    LuxuryPackage(std::unique_ptr<Car> car)
        : CarDecorator(std::move(car)) {}

    std::string getDescription() const override {
        return car_->getDescription() + ", Luxury Package";
    }

    double getCost() const override {
        return car_->getCost() + 5000;
    }
};

class SportPackage : public CarDecorator {
public:
    SportPackage(std::unique_ptr<Car> car)
        : CarDecorator(std::move(car)) {}

    std::string getDescription() const override {
        return car_->getDescription() + ", Sport Package";
    }

    double getCost() const override {
        return car_->getCost() + 3500;
    }
};

// Client code
int main() {
    auto basicCar = std::make_unique<BasicCar>();
    auto luxuryCar = std::make_unique<LuxuryPackage>(std::move(basicCar));

    std::cout << luxuryCar->getDescription()
              << " cost: " << luxuryCar->getCost() << std::endl;

    auto basicCar2 = std::make_unique<BasicCar>();
    auto sportCar = std::make_unique<SportPackage>(std::move(basicCar2));

    std::cout << sportCar->getDescription()
              << " cost: " << sportCar->getCost() << std::endl;

    auto basicCar3 = std::make_unique<BasicCar>();
    auto luxurySportCar = std::make_unique<LuxuryPackage>(std::make_unique<SportPackage>(std::move(basicCar3)));

    std::cout << luxurySportCar->getDescription()
              << " cost: " << luxurySportCar->getCost() << std::endl;

    return 0;
}
```

## Facade

[Facade Design Pattern](/gofdesignpattern/facade/facade.md)

Facade Pattern is a structural design pattern that provides a simplified
interface to a complex subsystem. It involves a single class that represents an
entire subsystem, hiding its complexity from the client.

```c++
#include <iostream>

// Subsystem classes
class CPU {
public:
    void start() {
        std::cout << "Initialize CPU" << std::endl;
    }
};

class HardDrive {
public:
    void read() {
        std::cout << "Read Hard Drive" << std::endl;
    }
};

class Memory {
public:
    void load() {
        std::cout << "Load memory" << std::endl;
    }
};

// Facade
class ComputerFacade {
private:
    CPU cpu;
    HardDrive hardDrive;
    Memory memory;

public:
    void startComputer() {
        cpu.start();
        memory.load();
        hardDrive.read();
    }
};

// Client
int main() {
    ComputerFacade computerFacade;
    computerFacade.startComputer();

    return 0;
}
```

## Flyweight

[Flyweight Design Pattern](/gofdesignpattern/flyweight/flyweight.md)

The Flyweight pattern is a structural design pattern that minimizes memory usage
by sharing articles of data among several objects, especially when dealing with
a large number of similar objects. The main idea is to create shared objects,
called "Flyweights," which can be reused representing the same state.

```c++
#include <iostream>
#include <map>
#include <string>

// The interface representing the shared objects (Flyweight)
class Shape {
public:
    virtual void draw(int x, int y) = 0;
};

// Concrete Flyweight objects implementing the interface
class Circle : public Shape {
    std::string color;
public:
    Circle(std::string color) : color(color) {}
    void draw(int x, int y) override {
        std::cout << "Drawing a " << color << " circle at (" << x << "," << y << ")" << std::endl;
    }
};

// Flyweight factory managing the creation and sharing of the objects
class ShapeFactory {
    static std::map<std::string, Shape*> shapes;
public:
    static Shape* getCircle(std::string color) {
        if (shapes.find(color) == shapes.end()) {
            shapes[color] = new Circle(color);
            std::cout << "Creating new " << color << " circle" << std::endl;
        }
        return shapes[color];
    }
};
std::map<std::string, Shape*> ShapeFactory::shapes;

int main() {
    Shape* redCircle = ShapeFactory::getCircle("red");
    redCircle->draw(10, 20);

    Shape* blueCircle = ShapeFactory::getCircle("blue");
    blueCircle->draw(30, 40);

    Shape* redCircle2 = ShapeFactory::getCircle("red");
    redCircle2->draw(50, 60);

    return 0;
}
```

## Proxy

[Proxy Deisgn Pattern](/gofdesignpattern/proxy/proxy.md)

Proxy Pattern is a structural design pattern that involves a set of intermediary
objects, or proxies, that sit between a client and the actual target object.
Proxies are useful for controlling access, managing resource-intensive
operations, or adding extra functionality to objects without adding complexity
to the real object.

```c++
#include <iostream>

class RealObject {
public:
    virtual void performAction() = 0;
};

class ActualObject : public RealObject {
public:
    void performAction() {
        std::cout << "Performing action in the actual object" << std::endl;
    }
};

class ProxyObject : public RealObject {
private:
    RealObject* realObject;

public:
    ProxyObject(RealObject* realObj) : realObject(realObj) {}

    void performAction() {
        std::cout << "Performing action in the proxy object" << std::endl;
        realObject->performAction();
    }
};

int main() {
    ActualObject actualObject;
    ProxyObject proxyObject(&actualObject);
    proxyObject.performAction(); // using proxy object instead of the actual object

    return 0;
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

```c++
#include <iostream>

class Handler {
public:
    virtual void set_next(Handler* handler) = 0;
    virtual void handle(int request) = 0;
};

class ConcreteHandlerA : public Handler {
private:
    Handler* next;

public:
    void set_next(Handler* handler) override {
        next = handler;
    }

    void handle(int request) override {
        if (request <= 10) {
            std::cout << "Handled by A" << std::endl;
        } else if (next != nullptr) {
            next->handle(request);
        }
    }
};

class ConcreteHandlerB : public Handler {
private:
    Handler* next;

public:
    void set_next(Handler* handler) override {
        next = handler;
    }

    void handle(int request) override {
        if (request > 10 && request <= 20) {
            std::cout << "Handled by B" << std::endl;
        } else if (next != nullptr) {
            next->handle(request);
        }
    }
};

int main() {
    ConcreteHandlerA handlerA;
    ConcreteHandlerB handlerB;

    handlerA.set_next(&handlerB);

    handlerA.handle(5);
    handlerA.handle(15);
    handlerA.handle(25);

    return 0;
}
```

## Command

[Command Design Pattern](/gofdesignpattern/command/command.md)

The Command pattern is a behavioral design pattern that turns a request into a
stand-alone object that contains all information about the request. This
transformation lets you pass requests as method arguments, delay or queue a
request’s execution, and support undoable operations.

```c++
#include <iostream>

class Command {
public:
  virtual void execute() = 0;
};

class Receiver {
public:
  void actionA() { std::cout << "Action A" << std::endl; }
  void actionB() { std::cout << "Action B" << std::endl; }
};

class ConcreteCommandA : public Command {
public:
  ConcreteCommandA(Receiver *receiver) : m_receiver(receiver) {}
  void execute() { m_receiver->actionA(); }

private:
  Receiver *m_receiver;
};

class ConcreteCommandB : public Command {
public:
  ConcreteCommandB(Receiver *receiver) : m_receiver(receiver) {}
  void execute() { m_receiver->actionB(); }

private:
  Receiver *m_receiver;
};

class Invoker {
public:
  void setCommand(Command *command) { m_command = command; }
  void executeCommand() { m_command->execute(); }

private:
  Command *m_command;
};

int main() {
  Receiver receiver;
  ConcreteCommandA commandA(&receiver);
  ConcreteCommandB commandB(&receiver);

  Invoker invoker;
  invoker.setCommand(&commandA);
  invoker.executeCommand();

  invoker.setCommand(&commandB);
  invoker.executeCommand();

  return 0;
}
```

## Interpreter

[Interpreter Design Pattern](/gofdesignpattern/interpreter/interpreter.md)

Interpreter pattern is a design pattern that provides a way to evaluate language
grammar or expressions. It involves building an interpreter to interpret the
expressions of a language. This pattern is used in situations where there is a
language to be interpreted or domain-specific language expression, which must be
evaluated.

```c++
#include <iostream>
#include <map>
#include <string>

class Expression {
public:
    virtual bool interpret(std::map<std::string, bool>& context) = 0;
};

class Constant : public Expression {
private:
    bool value;

public:
    Constant(bool value) : value(value) {}

    bool interpret(std::map<std::string, bool>& context) override {
        return value;
    }
};

class Variable : public Expression {
private:
    std::string name;

public:
    Variable(std::string name) : name(name) {}

    bool interpret(std::map<std::string, bool>& context) override {
        return context[name];
    }
};

class And : public Expression {
private:
    Expression* left;
    Expression* right;

public:
    And(Expression* left, Expression* right) : left(left), right(right) {}

    bool interpret(std::map<std::string, bool>& context) override {
        return left->interpret(context) && right->interpret(context);
    }
};

class Or : public Expression {
private:
    Expression* left;
    Expression* right;

public:
    Or(Expression* left, Expression* right) : left(left), right(right) {}

    bool interpret(std::map<std::string, bool>& context) override {
        return left->interpret(context) || right->interpret(context);
    }
};

int main() {
    Variable a("a");
    Variable b("b");
    And a_and_b(&a, &b);
    Or a_or_b(&a, &b);
    Or expression(&a_and_b, &a_or_b);

    std::map<std::string, bool> context;
    context["a"] = true;
    context["b"] = false;

    std::cout << "Expression result: " << expression.interpret(context) << std::endl;
}
```

## Iterator

[Iterator Design Pattern](/gofdesignpattern/iterator/iterator.md)

Iterator design pattern is a design pattern that provides a way to access the
elements of a collection object sequentially without exposing its underlying
representation. This pattern is useful to provide a standard way to traverse
through a group of objects and helps decouple the algorithms from the data
structures they operate upon.

```c++
#include <iostream>
#include <vector>

template <typename T>
class MyCollection {
public:
    void add(const T& item) {
        items.push_back(item);
    }

    typename std::vector<T>::iterator begin() {
        return items.begin();
    }

    typename std::vector<T>::iterator end() {
        return items.end();
    }

private:
    std::vector<T> items;
};

int main() {
    MyCollection<int> numbers;
    numbers.add(1);
    numbers.add(2);
    numbers.add(3);

    for (auto num : numbers) {
        std::cout << num << std::endl;
    }

    return 0;
}
```

## Mediator

[Mediator Design Pattern](/gofdesignpattern/mediator/mediator.md)

The Mediator pattern is a behavioral design pattern that defines an object that
encapsulates how objects (colleagues) interact with each other. This pattern
promotes loose coupling by avoiding direct connections between colleagues, and
the mediator object is responsible for handling their interactions.

```c++
#include <iostream>
#include <vector>
#include <string>
#include <memory>

class User;

class ChatMediator {
public:
  virtual void SendMessage(const std::string& msg, User* user) = 0;
  virtual void AddUser(std::unique_ptr<User> user) = 0;
};

class User {
protected:
  ChatMediator& mediator_;
  std::string name_;
public:
  User(ChatMediator& mediator, const std::string& name)
      : mediator_(mediator), name_(name) {}
  virtual ~User() {}
  virtual void Send(const std::string& msg) = 0;
  virtual void Receive(const std::string& msg) = 0;
};

class ChatMediatorImpl : public ChatMediator {
private:
  std::vector<std::unique_ptr<User>> users_;
public:
  void SendMessage(const std::string& msg, User* user) override {
    for (const auto& u : users_) {
      if (u.get() != user) {
        u->Receive(msg);
      }
    }
  }
  void AddUser(std::unique_ptr<User> user) override {
    users_.emplace_back(std::move(user));
  }
};

class UserImpl : public User {
public:
  UserImpl(ChatMediator& mediator, const std::string& name)
      : User(mediator, name) {}

  void Send(const std::string& msg) override {
    mediator_.SendMessage(msg, this);
  }

  void Receive(const std::string& msg) override {
    std::cout << name_ << " received: " << msg << std::endl;
  }
};

int main() {
  ChatMediatorImpl mediator;

  auto user1 = std::make_unique<UserImpl>(mediator, "Alice");
  auto user2 = std::make_unique<UserImpl>(mediator, "Bob");
  auto user3 = std::make_unique<UserImpl>(mediator, "Eve");

  mediator.AddUser(std::move(user1));
  mediator.AddUser(std::move(user2));
  mediator.AddUser(std::move(user3));

  mediator.SendMessage("Hello, everyone!", nullptr);

  return 0;
}
```

## Memento

[Memento Design Pattern](/gofdesignpattern/memento/memento.md)

Memento pattern is a behavioral design pattern that allows an object
(originator) to save and restore its previous state without revealing the
structure or details of its internal state, thus enabling an undo mechanism. A
memento object is used to store the state of the originator, and a caretaker is
responsible for the memento's safe keeping.

```c++
#include <iostream>
#include <stack>

// The Memento class
class Memento {
private:
    std::string state;

public:
    Memento(std::string state) : state(state) {}
    std::string getState() { return state; }
};

// The Originator class
class Originator {
private:
    std::string state;

public:
    void setState(std::string newState) { state = newState; }
    Memento save() { return Memento(state); }
    void restore(Memento memento) { state = memento.getState(); }
};

// The Caretaker class
class Caretaker {
private:
    std::stack<Memento> mementos;

public:
    void save(Originator &originator) { mementos.push(originator.save()); }
    void undo(Originator &originator) {
        if (!mementos.empty()) {
            originator.restore(mementos.top());
            mementos.pop();
        }
    }
};
```

## Observer

[Observer Design Pattern](/gofdesignpattern/observer/observer.md)

Observer Pattern is a behavioral design pattern that allows an object (subject)
to maintain a list of its dependents (observers) and notify them automatically
of any changes in state. When a state change occurs, the subject updates the
observers.

```c++
#include <iostream>
#include <string>
#include <vector>

class Observer {
public:
    virtual void update(const std::string& message) = 0;
};

class Subject {
private:
    std::vector<Observer*> observers;

public:
    void attach(Observer* observer) {
        observers.push_back(observer);
    }

    void notifyObservers(const std::string& message) {
        for (Observer* observer : observers) {
            observer->update(message);
        }
    }
};

class ConcreteObserver : public Observer {
private:
    std::string name;

public:
    ConcreteObserver(const std::string& name) : name(name) {}

    void update(const std::string& message) override {
        std::cout << name << " received the message: " << message << std::endl;
    }
};

int main() {
    Subject subject;
    ConcreteObserver observer1("Observer1");
    ConcreteObserver observer2("Observer2");

    subject.attach(&observer1);
    subject.attach(&observer2);

    subject.notifyObservers("Hello, Observers!");
}
```

## State

[State Design Pattern](/gofdesignpattern/state/state.md)

The State pattern is a behavioral design pattern that allows an object to change
its behavior when its internal state changes.

```c++
#include <iostream>

class Context;

class State {
public:
  virtual void handle(Context &context) = 0;
};

class ConcreteStateA : public State {
public:
  void handle(Context &context) override;
};

class ConcreteStateB : public State {
public:
  void handle(Context &context) override;
};

class Context {
private:
  State *state;

public:
  Context(State *state) : state(state) {}
  void changeState(State *newState) { state = newState; }
  void request() { state->handle(*this); }
};

void ConcreteStateA::handle(Context &context) { context.changeState(new ConcreteStateB); }
void ConcreteStateB::handle(Context &context) { context.changeState(new ConcreteStateA); }

int main() {
  Context context(new ConcreteStateA());
  context.request();
  context.request();
}
```

## Strategy

[Strategy Design Pattern](/gofdesignpattern/strategy/strategy.md)

The strategy pattern is a behavioral design pattern that defines a family of
algorithms, encapsulates each of them, and makes them interchangeable. It allows
the algorithm to vary independently from the clients that use it.

```c++
#include <iostream>

class Strategy {
public:
    virtual int doOperation(int a, int b) = 0;
};

class Addition : public Strategy {
public:
    int doOperation(int a, int b) {
        return a + b;
    }
};

class Subtraction : public Strategy {
public:
    int doOperation(int a, int b) {
        return a - b;
    }
};

class Multiplication : public Strategy {
public:
    int doOperation(int a, int b) {
        return a * b;
    }
};

class Context {
private:
    Strategy* strategy;
public:
    Context(Strategy* strategy) : strategy(strategy) {}

    int executeStrategy(int a, int b) {
        return strategy->doOperation(a, b);
    }
};

int main() {
    Context context(new Addition());
    std::cout << "10 + 5 = " << context.executeStrategy(10, 5) << std::endl;

    context = Context(new Subtraction());
    std::cout << "10 - 5 = " << context.executeStrategy(10, 5) << std::endl;

    context = Context(new Multiplication());
    std::cout << "10 * 5 = " << context.executeStrategy(10, 5) << std::endl;

    return 0;
}
```

## Template

[Template Design Pattern](/gofdesignpattern/template/template.md)

The template pattern is a behavioral design pattern that defines the program
skeleton of an algorithm in a method, called template method, which defers some
steps to subclasses. It lets subclasses redefine certain steps of an algorithm
without changing the algorithm's structure.

```c++
#include <iostream>

class Game {
public:
  virtual void initialize() = 0;
  virtual void startGame() = 0;
  virtual void endGame() = 0;

  void play() {
    initialize();
    startGame();
    endGame();
  }
};

class Cricket : public Game {
  void initialize() { std::cout << "Cricket Game Initialized!\n"; }
  void startGame() { std::cout << "Cricket Game Started!\n"; }
  void endGame() { std::cout << "Cricket Game Finished!\n"; }
};

class Football : public Game {
  void initialize() { std::cout << "Football Game Initialized!\n"; }
  void startGame() { std::cout << "Football Game Started!\n"; }
  void endGame() { std::cout << "Football Game Finished!\n"; }
};

int main() {
  Game* game = new Cricket();
  game->play();

  game = new Football();
  game->play();

  delete game;
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

```c++
#include <iostream>
#include <cmath>
#include <vector>

class Circle;
class Rectangle;

class ShapeVisitor {
public:
    virtual void visit(Circle& circle) = 0;
    virtual void visit(Rectangle& rectangle) = 0;
};

class Shape {
public:
    virtual void accept(ShapeVisitor& visitor) = 0;
};

class Circle : public Shape {
public:
    double radius;

    Circle(double radius) : radius(radius) {}

    void accept(ShapeVisitor& visitor) override {
        visitor.visit(*this);
    }
};

class Rectangle : public Shape {
public:
    double length, width;

    Rectangle(double length, double width) : length(length), width(width) {}

    void accept(ShapeVisitor& visitor) override {
        visitor.visit(*this);
    }
};

class ShapeAreaVisitor : public ShapeVisitor {
public:
    double totalArea = 0;

    void visit(Circle& circle) override {
        totalArea += 3.14159 * std::pow(circle.radius, 2);
    }

    void visit(Rectangle& rectangle) override {
        totalArea += rectangle.length * rectangle.width;
    }
};

int main() {
    std::vector<Shape*> shapes = {new Circle(5), new Rectangle(3, 4), new Circle(2)};
    ShapeAreaVisitor areaVisitor;

    for (Shape* shape : shapes) {
        shape->accept(areaVisitor);
    }

    std::cout << "Total area: " << areaVisitor.totalArea << std::endl;

    return 0;
}
```
