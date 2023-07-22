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

---

# References

* [GOF Design Pattern](/gofdesignpattern/README.md)

# Materials

* [Go Gof Design Pattern | refactoring.guru](https://refactoring.guru/design-patterns/go)

# Creational Pattern

## Abstract Factory

[Abstract Factory Pattern](/gofdesignpattern/abstractfactory/abstractfactory.md)

Abstract Factory is a design pattern that provides an interface for creating
families of related or dependent objects without specifying their concrete
classes.

```go
package main

import (
	"fmt"
	"errors"
)

type Plant interface{}

type OrangePlant struct{}

type ApplePlant struct{}

type PlantFactory interface {
	MakePlant() Plant
}

type AppleFactory struct{}

func (f AppleFactory) MakePlant() Plant {
	return ApplePlant{}
}

type OrangeFactory struct{}

func (f OrangeFactory) MakePlant() Plant {
	return OrangePlant{}
}

type PlantFactoryType int

const (
	Apple PlantFactoryType = iota
	Orange
)

func CreateFactory(factoryType PlantFactoryType) (PlantFactory, error) {
	switch factoryType {
	case Apple:
		return AppleFactory{}, nil
	case Orange:
		return OrangeFactory{}, nil
	default:
		return nil, errors.New("Invalid plant factory type")
	}
}

func main() {
	plantFactory, err := CreateFactory(Orange)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	plant := plantFactory.MakePlant()
	fmt.Printf("Created plant: %T\n", plant)
}
```


## Builder

[Builder Pattern](/gofdesignpattern/builder/builder.md)

The Builder pattern is a creational design pattern that's used when
constructing complex objects. It lets you separate the construction of objects
and their representations, so that you can create different objects using the
same construction process.

### Basic

Builder class is unnecessary.

```go
package main

import (
	"fmt"
)

type Person struct {
	id      string
	pw      string
	name    string
	address string
	email   string
}

type Builder struct {
	id      string
	pw      string
	name    string
	address string
	email   string
}

func NewBuilder(id, pw string) *Builder {
	return &Builder{id: id, pw: pw}
}

func (b *Builder) WithName(name string) *Builder {
	b.name = name
	return b
}

func (b *Builder) WithAddress(address string) *Builder {
	b.address = address
	return b
}

func (b *Builder) WithEmail(email string) *Builder {
	b.email = email
	return b
}

func (b *Builder) Build() *Person {
	return &Person{
		id:      b.id,
		pw:      b.pw,
		name:    b.name,
		address: b.address,
		email:   b.email,
	}
}

func main() {
	person := NewBuilder("AABBCCDD", "123456").
		WithName("iamslash").
		WithAddress("Irving Ave").
		WithEmail("iamslash@gmail.com").
		Build()
	fmt.Printf("ID: %s\nPW: %s\nName: %s\nAddress: %s\nEmail: %s\n", person.id, person.pw, person.name, person.address, person.email)
}
```

### Improved

Removed Builder class.

```go
package main

import (
    "fmt"
)

type Person struct {
    id      string
    pw      string
    name    string
    address string
    email   string
}

type PersonBuilder func(*Person)

func WithName(name string) PersonBuilder {
    return func(p *Person) {
        p.name = name
    }
}

func WithAddress(address string) PersonBuilder {
    return func(p *Person) {
        p.address = address
    }
}

func WithEmail(email string) PersonBuilder {
    return func(p *Person) {
        p.email = email
    }
}

func NewPerson(id string, pw string, builders ...PersonBuilder) *Person {
    person := &Person{id: id, pw: pw}
    for _, builder := range builders {
        builder(person)
    }
    return person
}

func main() {
    person := NewPerson(
        "AABBCCDD",
        "123456",
        WithName("iamslash"),
        WithAddress("Irving Ave"),
        WithEmail("iamslash@gmail.com"),
    )
    fmt.Printf("ID: %s\nPW: %s\nName: %s\nAddress: %s\nEmail: %s\n", person.id, person.pw, person.name, person.address, person.email)
}
```

## Factory Method

[Factory Method Pattern](/gofdesignpattern/factorymethod/factorymethod.md)

The Factory Method Pattern provides an interface for creating objects in a
superclass, but allows subclasses to alter the type of objects that will be
created. It promotes loose coupling by removing the direct dependencies between
classes and follows the concept of "programming to an interface".

```go
package main

import "fmt"

// Shape interface
type Shape interface {
	Draw()
}

// Circle struct
type Circle struct{}

func (c Circle) Draw() {
	fmt.Println("Drawing a circle.")
}

// Rectangle struct
type Rectangle struct{}

func (r Rectangle) Draw() {
	fmt.Println("Drawing a rectangle.")
}

// Triangle struct
type Triangle struct{}

func (t Triangle) Draw() {
	fmt.Println("Drawing a triangle.")
}

// ShapeFactory interface
type ShapeFactory interface {
	CreateShape() Shape
}

// CircleFactory struct
type CircleFactory struct{}

func (cf CircleFactory) CreateShape() Shape {
	return Circle{}
}

// RectangleFactory struct
type RectangleFactory struct{}

func (rf RectangleFactory) CreateShape() Shape {
	return Rectangle{}
}

// TriangleFactory struct
type TriangleFactory struct{}

func (tf TriangleFactory) CreateShape() Shape {
	return Triangle{}
}

// main function
func main() {
	// Create a circle factory and create a circle
	circleFactory := CircleFactory{}
	circle := circleFactory.CreateShape()
	circle.Draw()

	// Create a rectangle factory and create a rectangle
	rectangleFactory := RectangleFactory{}
	rectangle := rectangleFactory.CreateShape()
	rectangle.Draw()

	// Create a triangle factory and create a triangle
	triangleFactory := TriangleFactory{}
	triangle := triangleFactory.CreateShape()
	triangle.Draw()
}
```

## Prototype

[Prototype Pattern](/gofdesignpattern/prototype/prototype.md)

The prototype pattern is a creational design pattern that allows you to create
new objects by copying an existing object, called the prototype. This pattern is
especially useful when creating a new object instance is expensive in terms of
resources or time, and the existing object shares most of its properties with
the new object.

```go
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

    # Verify the cloned object properties
    print("Car1 Make:", car1.make)
    print("Car1 Model:", car1.model)
    print("Car1 Color:", car1.color)
    print("-------------")
    print("Car2 Make:", car2.make)
    print("Car2 Model:", car2.model)
    print("Car2 Color:", car2.color)

    # Modify the cloned object color
    car2.color = "Red"

    # Verify that cloned object properties are modified
    print("-------------")
    print("Car1 Color:", car1.color)
    print("Car2 Color:", car2.color)

if __name__ == '__main__':
    main()
```

## [Singleton](go_design_pattern/singleton.md)

[Singleton Design Pattern](/gofdesignpattern/singleton/singleton.md)

Singleton pattern is a design pattern that restricts the instantiation of a
class to only one instance, which is globally accessible in an application. This
is useful when exactly one object is needed to coordinate actions across the
system.

```go
package main

import (
    "fmt"
    "sync"
)

type Singleton struct{}

var instance *Singleton
var once sync.Once

func GetInstance() *Singleton {
    once.Do(func() {
        instance = &Singleton{}
    })
    return instance
}

// An example of a method
func (s *Singleton) PrintMessage() {
    fmt.Println("Hello from Singleton!")
}

// Test the Singleton
func main() {
    singleton := GetInstance()
    singleton.PrintMessage()
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

```go
package main

import "fmt"

type Temperature interface {
	GetCelsiusTemperature() float64
}

type CelsiusTemperature struct {
	temperature float64
}

func (c CelsiusTemperature) GetCelsiusTemperature() float64 {
	return c.temperature
}

type FahrenheitTemperature struct {
	temperature float64
}

func (f FahrenheitTemperature) GetFahrenheitTemperature() float64 {
	return f.temperature
}

type FahrenheitToCelsiusAdapter struct {
	fahrenheitTemperature FahrenheitTemperature
}

func (a FahrenheitToCelsiusAdapter) GetCelsiusTemperature() float64 {
	return (a.fahrenheitTemperature.GetFahrenheitTemperature() - 32) * 5.0 / 9.0
}

func main() {
	celsiusTemp := CelsiusTemperature{temperature: 100}
	fmt.Printf("Celsius Temperature: %.2f\n", celsiusTemp.GetCelsiusTemperature())

	fahrenheitTemp := FahrenheitTemperature{temperature: 212}
	fahrenheitToCelsiusTemp := FahrenheitToCelsiusAdapter{fahrenheitTemperature: fahrenheitTemp}
	fmt.Printf("Fahrenheit Temperature in Celsius: %.2f\n", fahrenheitToCelsiusTemp.GetCelsiusTemperature())
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

```go
package main

import "fmt"

// Abstraction
type Color interface {
	applyColor()
}

// Implementor
type RedColor struct{}

func (c RedColor) applyColor() {
	fmt.Println("Applying red color.")
}

type BlueColor struct{}

func (c BlueColor) applyColor() {
	fmt.Println("Applying blue color.")
}

// Bridge
type Shape struct {
	color Color
}

func (s Shape) draw() {}

type Circle struct {
	Shape
}

func NewCircle(color Color) *Circle {
	return &Circle{Shape{color}}
}

func (c Circle) draw() {
	fmt.Print("Drawing circle: ")
	c.color.applyColor()
}

type Square struct {
	Shape
}

func NewSquare(color Color) *Square {
	return &Square{Shape{color}}
}

func (s Square) draw() {
	fmt.Print("Drawing square: ")
	s.color.applyColor()
}

// Client
func main() {
	redCircle := NewCircle(RedColor{})
	blueSquare := NewSquare(BlueColor{})

	redCircle.draw()
	blueSquare.draw()
}
```

## Composite

[Composite Design Pattern](/gofdesignpattern/composite/composite.md)

Composite pattern is a structural design pattern that allows you to compose
objects into tree structures to represent part-whole hierarchies. It lets
clients treat individual objects and compositions uniformly.

```go
package main

import (
	"fmt"
)

// Component
type FileSystem interface {
	ShowFileSystem()
}

// Leaf
type File struct {
	fileName string
}

func (f *File) ShowFileSystem() {
	fmt.Println("File:", f.fileName)
}

// Composite
type Directory struct {
	directoryName string
	fileSystemList []FileSystem
}

func (d *Directory) AddFileSystem(fs FileSystem) {
	d.fileSystemList = append(d.fileSystemList, fs)
}

func (d *Directory) ShowFileSystem() {
	fmt.Println("\nDirectory:", d.directoryName)
	for _, fileSystem := range d.fileSystemList {
		fileSystem.ShowFileSystem()
	}
}

func main() {
	rootDirectory := &Directory{directoryName: "Root"}

	dir1 := &Directory{directoryName: "Directory1"}
	dir1.AddFileSystem(&File{fileName: "File1.txt"})
	dir1.AddFileSystem(&File{fileName: "File2.txt"})
	rootDirectory.AddFileSystem(dir1)

	dir2 := &Directory{directoryName: "Directory2"}
	dir2.AddFileSystem(&File{fileName: "File3.txt"})
	dir2.AddFileSystem(&File{fileName: "File4.txt"})
	rootDirectory.AddFileSystem(dir2)

	rootDirectory.ShowFileSystem()
}
```

## Decorator

[Decorator Design Pattern](/gofdesignpattern/decorator/decorator.md)

The decorator pattern is a structural design pattern that allows adding new
functionality to an object without altering its structure. It involves a set of
decorator classes that are used to wrap concrete components. The decorator
classes mirror the type of components they can decorate but add or override
behavior.

```go
package main

import (
	"fmt"
)

type Car interface {
	GetDescription() string
	GetCost() float64
}

type BasicCar struct{}

func (c *BasicCar) GetDescription() string {
	return "Basic Car"
}

func (c *BasicCar) GetCost() float64 {
	return 10000
}

type CarDecorator struct {
	Car Car
}

func (d *CarDecorator) GetDescription() string {
	return d.Car.GetDescription()
}

func (d *CarDecorator) GetCost() float64 {
	return d.Car.GetCost()
}

type LuxuryPackage struct {
	CarDecorator
}

func NewLuxuryPackage(car Car) *LuxuryPackage {
	return &LuxuryPackage{CarDecorator{Car: car}}
}

func (l *LuxuryPackage) GetDescription() string {
	return l.Car.GetDescription() + ", Luxury Package"
}

func (l *LuxuryPackage) GetCost() float64 {
	return l.Car.GetCost() + 5000
}

type SportPackage struct {
	CarDecorator
}

func NewSportPackage(car Car) *SportPackage {
	return &SportPackage{CarDecorator{Car: car}}
}

func (s *SportPackage) GetDescription() string {
	return s.Car.GetDescription() + ", Sport Package"
}

func (s *SportPackage) GetCost() float64 {
	return s.Car.GetCost() + 3500
}

// Client code
func main() {
	basicCar := &BasicCar{}

	luxuryCar := NewLuxuryPackage(basicCar)
	fmt.Printf("%s cost: %f\n", luxuryCar.GetDescription(), luxuryCar.GetCost())

	sportCar := NewSportPackage(basicCar)
	fmt.Printf("%s cost: %f\n", sportCar.GetDescription(), sportCar.GetCost())

	luxurySportCar := NewLuxuryPackage(NewSportPackage(basicCar))
	fmt.Printf("%s cost: %f\n", luxurySportCar.GetDescription(), luxurySportCar.GetCost())
}
```

## Facade

[Facade Design Pattern](/gofdesignpattern/facade/facade.md)

Facade Pattern is a structural design pattern that provides a simplified
interface to a complex subsystem. It involves a single class that represents an
entire subsystem, hiding its complexity from the client.

```go
package main

import "fmt"

// Subsystem classes
type CPU struct{}

func (c *CPU) Start() {
	fmt.Println("Initialize CPU")
}

type HardDrive struct{}

func (h *HardDrive) Read() {
	fmt.Println("Read Hard Drive")
}

type Memory struct{}

func (m *Memory) Load() {
	fmt.Println("Load memory")
}

// Facade
type ComputerFacade struct {
	cpu       CPU
	hardDrive HardDrive
	memory    Memory
}

func (c *ComputerFacade) StartComputer() {
	c.cpu.Start()
	c.memory.Load()
	c.hardDrive.Read()
}

// Client
func main() {
	computerFacade := &ComputerFacade{}
	computerFacade.StartComputer()
}
```

## Flyweight

[Flyweight Design Pattern](/gofdesignpattern/flyweight/flyweight.md)

The Flyweight pattern is a structural design pattern that minimizes memory usage
by sharing articles of data among several objects, especially when dealing with
a large number of similar objects. The main idea is to create shared objects,
called "Flyweights," which can be reused representing the same state.

```go
package main

import "fmt"

type Shape interface {
	Draw(x, y int)
}

type Circle struct {
	color string
}

func (c Circle) Draw(x, y int) {
	fmt.Printf("Drawing a %s circle at (%d, %d)\n", c.color, x, y)
}

type ShapeFactory struct {
	shapes map[string]Shape
}

func (sf *ShapeFactory) GetCircle(color string) Shape {
	if _, ok := sf.shapes[color]; !ok {
		sf.shapes[color] = Circle{color: color}
		fmt.Printf("Creating new %s circle\n", color)
	}
	return sf.shapes[color]
}

func main() {
	shapeFactory := ShapeFactory{shapes: map[string]Shape{}}

	redCircle := shapeFactory.GetCircle("red")
	redCircle.Draw(10, 20)

	blueCircle := shapeFactory.GetCircle("blue")
	blueCircle.Draw(30, 40)

	redCircle2 := shapeFactory.GetCircle("red")
	redCircle2.Draw(50, 60)
}
```

## Proxy

[Proxy Deisgn Pattern](/gofdesignpattern/proxy/proxy.md)

Proxy Pattern is a structural design pattern that involves a set of intermediary
objects, or proxies, that sit between a client and the actual target object.
Proxies are useful for controlling access, managing resource-intensive
operations, or adding extra functionality to objects without adding complexity
to the real object.

```go
package main

import "fmt"

type RealObject interface {
    PerformAction()
}

type ActualObject struct{}

func (a *ActualObject) PerformAction() {
    fmt.Println("Performing action in the actual object")
}

type ProxyObject struct {
    realObject RealObject
}

func (p *ProxyObject) PerformAction() {
    fmt.Println("Performing action in the proxy object")
    p.realObject.PerformAction()
}

func main() {
    actualObject := &ActualObject{}
    proxyObject := &ProxyObject{realObject: actualObject}
    proxyObject.PerformAction() // using proxy object instead of the actual object
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

```go
package main

import "fmt"

type Handler interface {
	SetNext(Handler)
	Handle(int)
}

type ConcreteHandlerA struct {
	next Handler
}

func (a *ConcreteHandlerA) SetNext(handler Handler) {
	a.next = handler
}

func (a *ConcreteHandlerA) Handle(request int) {
	if request <= 10 {
		fmt.Println("Handled by A")
	} else if a.next != nil {
		a.next.Handle(request)
	}
}

type ConcreteHandlerB struct {
	next Handler
}

func (b *ConcreteHandlerB) SetNext(handler Handler) {
	b.next = handler
}

func (b *ConcreteHandlerB) Handle(request int) {
	if request > 10 && request <= 20 {
		fmt.Println("Handled by B")
	} else if b.next != nil {
		b.next.Handle(request)
	}
}

func main() {
	handlerA := &ConcreteHandlerA{}
	handlerB := &ConcreteHandlerB{}

	handlerA.SetNext(handlerB)

	handlerA.Handle(5)
	handlerA.Handle(15)
	handlerA.Handle(25)
}
```

## Command

[Command Design Pattern](/gofdesignpattern/command/command.md)

The Command pattern is a behavioral design pattern that turns a request into a
stand-alone object that contains all information about the request. This
transformation lets you pass requests as method arguments, delay or queue a
request’s execution, and support undoable operations.

```go
package main

import "fmt"

type Command interface {
	Execute()
}

type Receiver struct{}

func (r *Receiver) ActionA() {
	fmt.Println("Action A")
}

func (r *Receiver) ActionB() {
	fmt.Println("Action B")
}

type ConcreteCommandA struct {
	receiver *Receiver
}

func (c *ConcreteCommandA) Execute() {
	c.receiver.ActionA()
}

type ConcreteCommandB struct {
	receiver *Receiver
}

func (c *ConcreteCommandB) Execute() {
	c.receiver.ActionB()
}

type Invoker struct {
	command Command
}

func (i *Invoker) SetCommand(command Command) {
	i.command = command
}

func (i *Invoker) ExecuteCommand() {
	i.command.Execute()
}

func main() {
	receiver := &Receiver{}
	commandA := &ConcreteCommandA{receiver: receiver}
	commandB := &ConcreteCommandB{receiver: receiver}
	invoker := &Invoker{}

	invoker.SetCommand(commandA)
	invoker.ExecuteCommand()

	invoker.SetCommand(commandB)
	invoker.ExecuteCommand()
}
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
