- [References](#references)
- [Materials](#materials)
- [Creational Pattern](#creational-pattern)
  - [Abstract Factory](#abstract-factory)
  - [Builder](#builder)
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

* [Python GOF Design Pattern | refactoring.guru](https://refactoring.guru/design-patterns/python)

# Creational Pattern

## Abstract Factory

[Abstract Factory Pattern](/gofdesignpattern/abstractfactory/abstractfactory.md)

Abstract Factory is a design pattern that provides an interface for creating
families of related or dependent objects without specifying their concrete
classes.

```python
from abc import ABC, abstractmethod

class Plant(ABC):
    pass

class OrangePlant(Plant):
    pass

class ApplePlant(Plant):
    pass

class PlantFactory(ABC):
    @abstractmethod
    def make_plant(self):
        pass

class AppleFactory(PlantFactory):
    def make_plant(self):
        return ApplePlant()

class OrangeFactory(PlantFactory):
    def make_plant(self):
        return OrangePlant()

class PlantFactoryType:

    APPLE = 'apple'
    ORANGE = 'orange'

    @classmethod
    def create_factory(cls, factory_type):
        if factory_type == cls.APPLE:
            return AppleFactory()
        elif factory_type == cls.ORANGE:
            return OrangeFactory()
        else:
            raise ValueError(f"Invalid plant factory type: {factory_type}")

def main():
    plant_factory = PlantFactoryType.create_factory(PlantFactoryType.ORANGE)
    plant = plant_factory.make_plant()
    print(f"Created plant: {plant.__class__.__name__}")

if __name__ == "__main__":
    main()
```

## Builder

[Builder Pattern](/gofdesignpattern/builder/builder.md)

The Builder pattern is a creational design pattern that's used when
constructing complex objects. It lets you separate the construction of objects
and their representations, so that you can create different objects using the
same construction process.

```python
class Person:
    def __init__(self, builder):
        self.id = builder.id
        self.pw = builder.pw
        self.name = builder.name
        self.address = builder.address
        self.email = builder.email

    def __str__(self):
        return f"ID: {self.id}\nPW: {self.pw}\nName: {self.name}\nAddress: {self.address}\nEmail: {self.email}"

    class Builder:
        def __init__(self, id, pw):
            self.id = id
            self.pw = pw
            self.name = None
            self.address = None
            self.email = None

        def with_name(self, name):
            self.name = name
            return self

        def with_address(self, address):
            self.address = address
            return self

        def with_email(self, email):
            self.email = email
            return self

        def build(self):
            return Person(self)


def main():
    person = (
        Person.Builder("AABBCCDD", "123456")
        .with_name("iamslash")
        .with_address("Irving Ave")
        .with_email("iamslash@gmail.com")
        .build()
    )
    print(person)


if __name__ == "__main__":
    main()
```

## Factory Method


```py
from abc import ABC, abstractmethod


# Shape interface
class Shape(ABC):
    
    @abstractmethod
    def draw(self):
        pass


# Concrete implementations of Shape
class Circle(Shape):
    
    def draw(self):
        print("Drawing a circle.")


class Rectangle(Shape):
    
    def draw(self):
        print("Drawing a rectangle.")


class Triangle(Shape):
    
    def draw(self):
        print("Drawing a triangle.")


# Abstract factory class
class ShapeFactory(ABC):
    
    @abstractmethod
    def create_shape(self):
        pass


# Concrete factory classes
class CircleFactory(ShapeFactory):
    
    def create_shape(self):
        return Circle()


class RectangleFactory(ShapeFactory):
    
    def create_shape(self):
        return Rectangle()


class TriangleFactory(ShapeFactory):
    
    def create_shape(self):
        return Triangle()


# Client code
def main():
    # Create a circle factory and create a circle
    circle_factory = CircleFactory()
    circle = circle_factory.create_shape()
    circle.draw()

    # Create a rectangle factory and create a rectangle
    rectangle_factory = RectangleFactory()
    rectangle = rectangle_factory.create_shape()
    rectangle.draw()

    # Create a triangle factory and create a triangle
    triangle_factory = TriangleFactory()
    triangle = triangle_factory.create_shape()
    triangle.draw()


if __name__ == "__main__":
    main()
```

## Prototype

[Prototype Pattern](/gofdesignpattern/prototype/prototype.md)

The prototype pattern is a creational design pattern that allows you to create
new objects by copying an existing object, called the prototype. This pattern is
especially useful when creating a new object instance is expensive in terms of
resources or time, and the existing object shares most of its properties with
the new object.

```py
import copy

class Car:
    def __init__(self, make, model, color):
        self.make = make
        self.model = model
        self.color = color

    def clone(self):
        return copy.deepcopy(self)

class ToyotaCar(Car):
    def __init__(self):
        super().__init__("Toyota", "Camry", "Blue")

def main():
    car1 = ToyotaCar()

    # Create a new object using the prototype
    car2 = car1.clone()

    # Verify the cloned object's properties
    print("Car1 Make:", car1.make)
    print("Car1 Model:", car1.model)
    print("Car1 Color:", car1.color)
    print("-------------")
    print("Car2 Make:", car2.make)
    print("Car2 Model:", car2.model)
    print("Car2 Color:", car2.color)

    # Modify the cloned object's color
    car2.color = "Red"

    # Verify that cloned object's properties are modified
    print("-------------")
    print("Car1 Color:", car1.color)
    print("Car2 Color:", car2.color)

if __name__ == '__main__':
    main()
```

## Singleton

[Singleton Design Pattern](/gofdesignpattern/singleton/singleton.md)

Singleton pattern is a design pattern that restricts the instantiation of a
class to only one instance, which is globally accessible in an application. This
is useful when exactly one object is needed to coordinate actions across the
system.

```py
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance

    # An example of a method
    def print_message(self):
        print("Hello from Singleton!")


# Test the Singleton
if __name__ == "__main__":
    singleton = Singleton()
    singleton.print_message()
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

```py
from abc import ABC, abstractmethod

class Temperature(ABC):
    @abstractmethod
    def get_celsius_temperature(self):
        pass

class CelsiusTemperature(Temperature):
    def __init__(self, temperature):
        self.temperature = temperature
    
    def get_celsius_temperature(self):
        return self.temperature

class FahrenheitTemperature:
    def __init__(self, temperature):
        self.temperature = temperature
    
    def get_fahrenheit_temperature(self):
        return self.temperature

class FahrenheitToCelsiusAdapter(Temperature):
    def __init__(self, fahrenheit_temperature):
        self.fahrenheit_temperature = fahrenheit_temperature
    
    def get_celsius_temperature(self):
        return (self.fahrenheit_temperature.get_fahrenheit_temperature() - 32) * 5.0 / 9.0

def main():
    celsius_temp = CelsiusTemperature(100)
    print("Celsius Temperature:", celsius_temp.get_celsius_temperature())

    fahrenheit_temp = FahrenheitTemperature(212)
    fahrenheit_to_celsius_temp = FahrenheitToCelsiusAdapter(fahrenheit_temp)
    print("Fahrenheit Temperature in Celsius:", fahrenheit_to_celsius_temp.get_celsius_temperature())

if __name__ == "__main__":
    main()
```

## Bridge

[Bridge Design Pattern](/gofdesignpattern/bridge/bridge.md)

The Bridge pattern is a structural design pattern that is used to decouple an
abstraction from its implementation, allowing the two to evolve independently.
This pattern involves an interface (the bridge) that makes the functionality
available to clients, and multiple concrete implementations of that interface.
The clients interact with the abstraction provided by the interface and are
unaware of the concrete implementation.

```py
from abc import ABC, abstractmethod

# Abstraction
class Color(ABC):
    @abstractmethod
    def apply_color(self):
        pass

# Implementor
class RedColor(Color):
    def apply_color(self):
        print("Applying red color.")

class BlueColor(Color):
    def apply_color(self):
        print("Applying blue color.")

# Bridge
class Shape(ABC):
    def __init__(self, color):
        self.color = color

    @abstractmethod
    def draw(self):
        pass

class Circle(Shape):
    def draw(self):
        print("Drawing circle: ", end="")
        self.color.apply_color()

class Square(Shape):
    def draw(self):
        print("Drawing square: ", end="")
        self.color.apply_color()

# Client
if __name__ == "__main__":
    red_circle = Circle(RedColor())
    blue_square = Square(BlueColor())

    red_circle.draw()
    blue_square.draw()
```

## Composite

[Composite Design Pattern](/gofdesignpattern/composite/composite.md)

Composite pattern is a structural design pattern that allows you to compose
objects into tree structures to represent part-whole hierarchies. It lets
clients treat individual objects and compositions uniformly.

```python
from abc import ABC, abstractmethod

class FileSystem(ABC):
    @abstractmethod
    def show_file_system(self):
        pass

# Leaf
class File(FileSystem):
    def __init__(self, file_name):
        self.file_name = file_name

    def show_file_system(self):
        print(f"File: {self.file_name}")

# Composite
class Directory(FileSystem):
    def __init__(self, directory_name):
        self.directory_name = directory_name
        self.file_system_list = []

    def add_file_system(self, file_system):
        self.file_system_list.append(file_system)

    def show_file_system(self):
        print(f"\nDirectory: {self.directory_name}")
        for file_system in self.file_system_list:
            file_system.show_file_system()

def main():
    root_directory = Directory("Root")

    dir1 = Directory("Directory1")
    dir1.add_file_system(File("File1.txt"))
    dir1.add_file_system(File("File2.txt"))
    root_directory.add_file_system(dir1)

    dir2 = Directory("Directory2")
    dir2.add_file_system(File("File3.txt"))
    dir2.add_file_system(File("File4.txt"))
    root_directory.add_file_system(dir2)

    root_directory.show_file_system()

if __name__ == "__main__":
    main()
```

## Decorator

[Decorator Design Pattern](/gofdesignpattern/decorator/decorator.md)

The decorator pattern is a structural design pattern that allows adding new
functionality to an object without altering its structure. It involves a set of
decorator classes that are used to wrap concrete components. The decorator
classes mirror the type of components they can decorate but add or override
behavior.

```python
from abc import ABC, abstractmethod

class Car(ABC):
    @abstractmethod
    def get_description(self):
        pass

    @abstractmethod
    def get_cost(self):
        pass


class BasicCar(Car):

    def get_description(self):
        return "Basic Car"

    def get_cost(self):
        return 10000


class CarDecorator(Car):

    def __init__(self, car):
        self.car = car

    def get_description(self):
        return self.car.get_description()

    def get_cost(self):
        return self.car.get_cost()


class LuxuryPackage(CarDecorator):

    def get_description(self):
        return self.car.get_description() + ", Luxury Package"

    def get_cost(self):
        return self.car.get_cost() + 5000


class SportPackage(CarDecorator):

    def get_description(self):
        return self.car.get_description() + ", Sport Package"

    def get_cost(self):
        return self.car.get_cost() + 3500


# Client code
if __name__ == "__main__":
    basic_car = BasicCar()

    luxury_car = LuxuryPackage(basic_car)
    print(f"{luxury_car.get_description()} cost: {luxury_car.get_cost()}")

    sport_car = SportPackage(basic_car)
    print(f"{sport_car.get_description()} cost: {sport_car.get_cost()}")

    luxury_sport_car = LuxuryPackage(SportPackage(basic_car))
    print(f"{luxury_sport_car.get_description()} cost: {luxury_sport_car.get_cost()}")
```

## Facade

[Facade Design Pattern](/gofdesignpattern/facade/facade.md)

Facade Pattern is a structural design pattern that provides a simplified
interface to a complex subsystem. It involves a single class that represents an
entire subsystem, hiding its complexity from the client.

```py
# Subsystem classes
class CPU:
    def start(self):
        print("Initialize CPU")

class HardDrive:
    def read(self):
        print("Read Hard Drive")

class Memory:
    def load(self):
        print("Load memory")

# Facade
class ComputerFacade:
    def __init__(self):
        self.cpu = CPU()
        self.hard_drive = HardDrive()
        self.memory = Memory()

    def start_computer(self):
        self.cpu.start()
        self.memory.load()
        self.hard_drive.read()

# Client
def main():
    computer_facade = ComputerFacade()
    computer_facade.start_computer()

if __name__ == "__main__":
    main()
```

## Flyweight

[Flyweight Design Pattern](/gofdesignpattern/flyweight/flyweight.md)

The Flyweight pattern is a structural design pattern that minimizes memory usage
by sharing articles of data among several objects, especially when dealing with
a large number of similar objects. The main idea is to create shared objects,
called "Flyweights," which can be reused representing the same state.

```py
class Shape:
    def draw(self, x, y):
        pass

class Circle(Shape):
    def __init__(self, color):
        self.color = color

    def draw(self, x, y):
        print(f'Drawing a {self.color} circle at ({x},{y})')

class ShapeFactory:
    _shapes = {}

    @classmethod
    def get_circle(cls, color):
        if color not in cls._shapes:
            cls._shapes[color] = Circle(color)
            print(f'Creating new {color} circle')
        return cls._shapes[color]

red_circle = ShapeFactory.get_circle("red")
red_circle.draw(10, 20)

blue_circle = ShapeFactory.get_circle("blue")
blue_circle.draw(30, 40)

red_circle2 = ShapeFactory.get_circle("red")
red_circle2.draw(50, 60)
```

## Proxy

[Proxy Deisgn Pattern](/gofdesignpattern/proxy/proxy.md)

Proxy Pattern is a structural design pattern that involves a set of intermediary
objects, or proxies, that sit between a client and the actual target object.
Proxies are useful for controlling access, managing resource-intensive
operations, or adding extra functionality to objects without adding complexity
to the real object.

```py
from abc import ABC, abstractmethod

class RealObject(ABC):
    @abstractmethod
    def perform_action(self):
        pass

class ActualObject(RealObject):
    def perform_action(self):
        print("Performing action in the actual object")

class ProxyObject(RealObject):
    def __init__(self, real_object):
        self.real_object = real_object

    def perform_action(self):
        print("Performing action in the proxy object")
        self.real_object.perform_action()

if __name__ == "__main__":
    actual_object = ActualObject()
    proxy_object = ProxyObject(actual_object)
    proxy_object.perform_action() # using proxy object instead of the actual object
```

# Behaviorial Pattern

## Chain of Resp.

[Chain of Resp.](/gofdesignpattern/chainofresp/chainofresp.md)

The Chain of Responsibility design pattern is a behavioral pattern that allows
an object to send a request to a chain of potential handlers, where each handler
in the chain decides whether to process the request or pass it to the next
handler in the chain. The pattern is useful when there are multiple potential
handlers to decide which one should handle a specific request.

```py
from abc import ABC, abstractmethod

class Handler(ABC):
    def __init__(self):
        self.next = None

    @abstractmethod
    def handle(self, request):
        pass

    def set_next(self, handler):
        self.next = handler

class ConcreteHandlerA(Handler):
    def handle(self, request):
        if request <= 10:
            print("Handled by A")
        elif self.next is not None:
            self.next.handle(request)

class ConcreteHandlerB(Handler):
    def handle(self, request):
        if 10 < request <= 20:
            print("Handled by B")
        elif self.next is not None:
            self.next.handle(request)

handlerA = ConcreteHandlerA()
handlerB = ConcreteHandlerB()

handlerA.set_next(handlerB)

handlerA.handle(5)
handlerA.handle(15)
handlerA.handle(25)
```

## Command

[Command Design Pattern](/gofdesignpattern/command/command.md)

The Command pattern is a behavioral design pattern that turns a request into a
stand-alone object that contains all information about the request. This
transformation lets you pass requests as method arguments, delay or queue a
request’s execution, and support undoable operations.

```py
from abc import ABC, abstractmethod

class Command(ABC):
  @abstractmethod
  def execute(self): pass

class Receiver:
  def action_a(self): print("Action A")
  def action_b(self): print("Action B")

class ConcreteCommandA(Command):
  def __init__(self, receiver): self._receiver = receiver
  def execute(self): self._receiver.action_a()

class ConcreteCommandB(Command):
  def __init__(self, receiver): self._receiver = receiver
  def execute(self): self._receiver.action_b()

class Invoker:
  def set_command(self, command): self._command = command
  def execute_command(self): self._command.execute()

if __name__ == "__main__":
  receiver = Receiver()
  command_a = ConcreteCommandA(receiver)
  command_b = ConcreteCommandB(receiver)
  invoker = Invoker()

  invoker.set_command(command_a)
  invoker.execute_command()

  invoker.set_command(command_b)
  invoker.execute_command()
```

## Interpreter

## Iterator

## Mediator

## Memento

## Observer

## State

## Strategy

## Template

## Visitor

