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

* [GOF Design Pattern](/designpattern/README.md#gof-pattern)

# Materials

* [Swift Design Pattern | refactoring.guru](https://refactoring.guru/design-patterns/swift)
* [Design Patterns implemented in Swift 5.0 | github](https://github.com/ochococo/Design-Patterns-In-Swift)

# Creational Pattern

## Abstract Factory

[Abstract Factory Pattern](/gofdesignpattern/abstractfactory/abstractfactory.md)

Abstract Factory is a design pattern that provides an interface for creating
families of related or dependent objects without specifying their concrete
classes.

```swift
import Foundation

protocol Plant {}

class OrangePlant: Plant {}

class ApplePlant: Plant {}

protocol PlantFactory {
    func makePlant() -> Plant
}

class AppleFactory: PlantFactory {
    func makePlant() -> Plant {
        return ApplePlant()
    }
}

class OrangeFactory: PlantFactory {
    func makePlant() -> Plant {
        return OrangePlant()
    }
}

enum PlantFactoryType {
    case apple
    case orange

    func createFactory() -> PlantFactory {
        switch self {
        case .apple:
            return AppleFactory()
        case .orange:
            return OrangeFactory()
        }
    }
}

func main() {
    let plantFactory = PlantFactoryType.orange.createFactory()
    let plant = plantFactory.makePlant()
    print("Created plant: \(plant)")
}

main()
```

## Builder

[Builder Pattern](/gofdesignpattern/builder/builder.md)

The Builder pattern is a creational design pattern that's used when
constructing complex objects. It lets you separate the construction of objects
and their representations, so that you can create different objects using the
same construction process.

### Basic

There are a lot of duplicated fields.

```swift
class Person {
    let id: String
    let pw: String
    var name: String?
    var address: String?
    var email: String?
    
    private init(builder: Builder) {
        self.id = builder.id
        self.pw = builder.pw
        self.name = builder.name
        self.address = builder.address
        self.email = builder.email
    }
    
    class Builder {
        let id: String
        let pw: String
        var name: String?
        var address: String?
        var email: String?
        
        init(id: String, pw: String) {
            self.id = id
            self.pw = pw
        }
        
        func withName(_ name: String?) -> Builder {
            self.name = name
            return self
        }
        
        func withAddress(_ address: String?) -> Builder {
            self.address = address
            return self
        }
        
        func withEmail(_ email: String?) -> Builder {
            self.email = email
            return self
        }
        
        func build() -> Person {
            return Person(builder: self)
        }
    }
}

func main() {
    let person = Person.Builder(id: "AABBCCDD", pw: "123456")
        .withName("iamslash")
        .withAddress("Irving Ave")
        .withEmail("iamslash@gmail.com")
        .build()
    print(person)
}

main()
```

### Improved

Used functions and closures instead of a separate Builder class to avoid
duplicated fields

```swift
class Person {
    let id: String
    let pw: String
    var name: String?
    var address: String?
    var email: String?

    fileprivate init(
        id: String,
        pw: String,
        name: String?,
        address: String?,
        email: String?
    ) {
        self.id = id
        self.pw = pw
        self.name = name
        self.address = address
        self.email = email
    }

    typealias Builder = (
        _ id: String,
        _ pw: String
    ) -> (_ build: ((Person) -> Void)?) -> Person

    static func build(
        id: String,
        pw: String,
        _ build: ((Person) -> Void)? = nil
    ) -> Person {
        let person = Person(id: id, pw: pw, name: nil, address: nil, email: nil)
        build?(person)
        return person
    }
}

func main() {
    let person = Person.build(
        id: "AABBCCDD",
        pw: "123456"
    ) { person in
        person.name = "iamslash"
        person.address = "Irving Ave"
        person.email = "iamslash@gmail.com"
    }
    print(person)
}

main()
```

## Factory Method

[Factory Method Pattern](/gofdesignpattern/factorymethod/factorymethod.md)

The Factory Method Pattern provides an interface for creating objects in a
superclass, but allows subclasses to alter the type of objects that will be
created. It promotes loose coupling by removing the direct dependencies between
classes and follows the concept of "programming to an interface".

```swift
import Foundation

// Shape protocol
protocol Shape {
    func draw()
}

// Concrete implementations of Shape
class Circle: Shape {
    func draw() {
        print("Drawing a circle.")
    }
}

class Rectangle: Shape {
    func draw() {
        print("Drawing a rectangle.")
    }
}

class Triangle: Shape {
    func draw() {
        print("Drawing a triangle.")
    }
}

// ShapeFactory protocol
protocol ShapeFactory {
    func createShape() -> Shape
}

// Concrete factory classes
class CircleFactory: ShapeFactory {
    func createShape() -> Shape {
        return Circle()
    }
}

class RectangleFactory: ShapeFactory {
    func createShape() -> Shape {
        return Rectangle()
    }
}

class TriangleFactory: ShapeFactory {
    func createShape() -> Shape {
        return Triangle()
    }
}

// Client code
func main() {
    // Create a circle factory and create a circle
    let circleFactory = CircleFactory()
    let circle = circleFactory.createShape()
    circle.draw()

    // Create a rectangle factory and create a rectangle
    let rectangleFactory = RectangleFactory()
    let rectangle = rectangleFactory.createShape()
    rectangle.draw()

    // Create a triangle factory and create a triangle
    let triangleFactory = TriangleFactory()
    let triangle = triangleFactory.createShape()
    triangle.draw()
}

main()
```

## Prototype

[Prototype Pattern](/gofdesignpattern/prototype/prototype.md)

The prototype pattern is a creational design pattern that allows you to create
new objects by copying an existing object, called the prototype. This pattern is
especially useful when creating a new object instance is expensive in terms of
resources or time, and the existing object shares most of its properties with
the new object.

```swift
import Foundation

class Car: NSObject, NSCopying {
    var make: String
    var model: String
    var color: String

    init(make: String, model: String, color: String) {
        self.make = make
        self.model = model
        self.color = color
    }

    func copy(with zone: NSZone? = nil) -> Any {
        return Car(make: self.make, model: self.model, color: self.color)
    }
}

class ToyotaCar: Car {
    init() {
        super.init(make: "Toyota", model: "Camry", color: "Blue")
    }
}

func main() {
    let car1 = ToyotaCar()

    // Create a new object using the prototype
    let car2 = car1.copy() as! Car

    // Verify the cloned object's properties
    print("Car1 Make: \(car1.make)")
    print("Car1 Model: \(car1.model)")
    print("Car1 Color: \(car1.color)")
    print("-------------")
    print("Car2 Make: \(car2.make)")
    print("Car2 Model: \(car2.model)")
    print("Car2 Color: \(car2.color)")

    // Modify the cloned object's color
    car2.color = "Red"

    // Verify that cloned object's properties are modified
    print("-------------")
    print("Car1 Color: \(car1.color)")
    print("Car2 Color: \(car2.color)")
}

main()
```

## Singleton

[Singleton Design Pattern](/gofdesignpattern/singleton/singleton.md)

Singleton pattern is a design pattern that restricts the instantiation of a
class to only one instance, which is globally accessible in an application. This
is useful when exactly one object is needed to coordinate actions across the
system.

```swift
class Singleton {
    static let shared = Singleton()

    private init() {}

    // An example of a method
    func printMessage() {
        print("Hello from Singleton!")
    }
}

// Test the Singleton
func main() {
    let singleton = Singleton.shared
    singleton.printMessage()
}

main()
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

```swift
protocol Temperature {
    func getCelsiusTemperature() -> Double
}

class CelsiusTemperature: Temperature {
    private let temperature: Double

    init(temperature: Double) {
        self.temperature = temperature
    }

    func getCelsiusTemperature() -> Double {
        return temperature
    }
}

class FahrenheitTemperature {
    private let temperature: Double

    init(temperature: Double) {
        self.temperature = temperature
    }

    func getFahrenheitTemperature() -> Double {
        return temperature
    }
}

class FahrenheitToCelsiusAdapter: Temperature {
    private let fahrenheitTemperature: FahrenheitTemperature

    init(fahrenheitTemperature: FahrenheitTemperature) {
        self.fahrenheitTemperature = fahrenheitTemperature
    }

    func getCelsiusTemperature() -> Double {
        return ((fahrenheitTemperature.getFahrenheitTemperature() - 32) * 5.0) / 9.0
    }
}

func main() {
    let celsiusTemp = CelsiusTemperature(temperature: 100)
    print("Celsius Temperature:", celsiusTemp.getCelsiusTemperature())

    let fahrenheitTemp = FahrenheitTemperature(temperature: 212)
    let fahrenheitToCelsiusTemp = FahrenheitToCelsiusAdapter(fahrenheitTemperature: fahrenheitTemp)
    print("Fahrenheit Temperature in Celsius:", fahrenheitToCelsiusTemp.getCelsiusTemperature())
}

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

```swift
// Abstraction
protocol Color {
    func applyColor()
}

// Implementor
class RedColor: Color {
    func applyColor() {
        print("Applying red color.")
    }
}

class BlueColor: Color {
    func applyColor() {
        print("Applying blue color.")
    }
}

// Bridge
class Shape {
    var color: Color

    init(_ color: Color) {
        self.color = color
    }

    func draw() {}
}

class Circle: Shape {
    override func draw() {
        print("Drawing circle: ", terminator: "")
        color.applyColor()
    }
}

class Square: Shape {
    override func draw() {
        print("Drawing square: ", terminator: "")
        color.applyColor()
    }
}

// Client
let redCircle = Circle(RedColor())
let blueSquare = Square(BlueColor())

redCircle.draw()
blueSquare.draw()
```

## Composite

[Composite Design Pattern](/gofdesignpattern/composite/composite.md)

Composite pattern is a structural design pattern that allows you to compose
objects into tree structures to represent part-whole hierarchies. It lets
clients treat individual objects and compositions uniformly.

```swift
import Foundation

// Component
protocol FileSystem {
    func showFileSystem()
}

// Leaf
class File: FileSystem {
    private let fileName: String

    init(fileName: String) {
        self.fileName = fileName
    }

    func showFileSystem() {
        print("File: \(fileName)")
    }
}

// Composite
class Directory: FileSystem {
    private let directoryName: String
    private var fileSystemList = [FileSystem]()

    init(directoryName: String) {
        self.directoryName = directoryName
    }

    func addFileSystem(_ fileSystem: FileSystem) {
        fileSystemList.append(fileSystem)
    }

    func showFileSystem() {
        print("\nDirectory: \(directoryName)")
        for fileSystem in fileSystemList {
            fileSystem.showFileSystem()
        }
    }
}

func main() {
    let rootDirectory = Directory(directoryName: "Root")

    let dir1 = Directory(directoryName: "Directory1")
    dir1.addFileSystem(File(fileName: "File1.txt"))
    dir1.addFileSystem(File(fileName: "File2.txt"))
    rootDirectory.addFileSystem(dir1)

    let dir2 = Directory(directoryName: "Directory2")
    dir2.addFileSystem(File(fileName: "File3.txt"))
    dir2.addFileSystem(File(fileName: "File4.txt"))
    rootDirectory.addFileSystem(dir2)

    rootDirectory.showFileSystem()
}

main()
```

## Decorator

[Decorator Design Pattern](/gofdesignpattern/decorator/decorator.md)

The decorator pattern is a structural design pattern that allows adding new
functionality to an object without altering its structure. It involves a set of
decorator classes that are used to wrap concrete components. The decorator
classes mirror the type of components they can decorate but add or override
behavior.

```swift
import Foundation

protocol Car {
    func getDescription() -> String
    func getCost() -> Double
}

class BasicCar: Car {
    func getDescription() -> String {
        return "Basic Car"
    }

    func getCost() -> Double {
        return 10000.0
    }
}

class CarDecorator: Car {
    let car: Car

    init(car: Car) {
        self.car = car
    }

    func getDescription() -> String {
        return car.getDescription()
    }

    func getCost() -> Double {
        return car.getCost()
    }
}

class LuxuryPackage: CarDecorator {
    override func getDescription() -> String {
        return car.getDescription() + ", Luxury Package"
    }

    override func getCost() -> Double {
        return car.getCost() + 5000.0
    }
}

class SportPackage: CarDecorator {
    override func getDescription() -> String {
        return car.getDescription() + ", Sport Package"
    }

    override func getCost() -> Double {
        return car.getCost() + 3500.0
    }
}

let basicCar: Car = BasicCar()

let luxuryCar: Car = LuxuryPackage(car: basicCar)
print("\(luxuryCar.getDescription()) cost: \(luxuryCar.getCost())")

let sportCar: Car = SportPackage(car: basicCar)
print("\(sportCar.getDescription()) cost: \(sportCar.getCost())")

let luxurySportCar: Car = LuxuryPackage(car: SportPackage(car: basicCar))
print("\(luxurySportCar.getDescription()) cost: \(luxurySportCar.getCost())")
```

## Facade

[Facade Design Pattern](/gofdesignpattern/facade/facade.md)

Facade Pattern is a structural design pattern that provides a simplified
interface to a complex subsystem. It involves a single class that represents an
entire subsystem, hiding its complexity from the client.

```swift
// Subsystem classes
class CPU {
    func start() {
        print("Initialize CPU")
    }
}

class HardDrive {
    func read() {
        print("Read Hard Drive")
    }
}

class Memory {
    func load() {
        print("Load memory")
    }
}

// Facade
class ComputerFacade {
    private let cpu: CPU
    private let hardDrive: HardDrive
    private let memory: Memory
    
    init() {
        cpu = CPU()
        hardDrive = HardDrive()
        memory = Memory()
    }
    
    func startComputer() {
        cpu.start()
        memory.load()
        hardDrive.read()
    }
}

// Client
func main() {
    let computerFacade = ComputerFacade()
    computerFacade.startComputer()
}

main()
```

## Flyweight

[Flyweight Design Pattern](/gofdesignpattern/flyweight/flyweight.md)

The Flyweight pattern is a structural design pattern that minimizes memory usage
by sharing articles of data among several objects, especially when dealing with
a large number of similar objects. The main idea is to create shared objects,
called "Flyweights," which can be reused representing the same state.

```swift
import Foundation

protocol Shape {
    func draw(x: Int, y: Int)
}

class Circle: Shape {
    let color: String

    init(color: String) {
        self.color = color
    }

    func draw(x: Int, y: Int) {
        print("Drawing a \(color) circle at (\(x),\(y))")
    }
}

class ShapeFactory {
    // [:] is the shorthand syntax for an empty dictionary literal. 
    // the key-value pair types for the dictionary.
    private static var shapes: [String: Shape] = [:]

    static func getCircle(color: String) -> Shape {
        if let shape = shapes[color] {
            return shape
        }

        let newCircle = Circle(color: color)
        shapes[color] = newCircle
        print("Creating new \(color) circle")
        return newCircle
    }
}

let redCircle = ShapeFactory.getCircle(color: "red")
redCircle.draw(x: 10, y: 20)

let blueCircle = ShapeFactory.getCircle(color: "blue")
blueCircle.draw(x: 30, y: 40)

let redCircle2 = ShapeFactory.getCircle(color: "red")
redCircle2.draw(x: 50, y: 60)
```

## Proxy

[Proxy Deisgn Pattern](/gofdesignpattern/proxy/proxy.md)

Proxy Pattern is a structural design pattern that involves a set of intermediary
objects, or proxies, that sit between a client and the actual target object.
Proxies are useful for controlling access, managing resource-intensive
operations, or adding extra functionality to objects without adding complexity
to the real object.

```swift
protocol RealObject {
    func performAction()
}

class ActualObject: RealObject {
    func performAction() {
        print("Performing action in the actual object")
    }
}

class ProxyObject: RealObject {
    private let realObject: RealObject

    init(realObject: RealObject) {
        self.realObject = realObject
    }

    func performAction() {
        print("Performing action in the proxy object")
        realObject.performAction()
    }
}

let actualObject: RealObject = ActualObject()
let proxyObject: RealObject = ProxyObject(realObject: actualObject)
proxyObject.performAction() // using proxy object instead of the actual object
```

# Behaviorial Pattern

## Chain of Resp.

[Chain of Resp.](/gofdesignpattern/chainofresp/chainofresp.md)

The Chain of Responsibility design pattern is a behavioral pattern that allows
an object to send a request to a chain of potential handlers, where each handler
in the chain decides whether to process the request or pass it to the next
handler in the chain. The pattern is useful when there are multiple potential
handlers to decide which one should handle a specific request.

```swift
import Foundation

protocol Handler: AnyObject {
    var next: Handler? { get set }
    func handle(request: Int)
}

class ConcreteHandlerA: Handler {
    weak var next: Handler?

    func handle(request: Int) {
        if request <= 10 {
            print("Handled by A")
        } else {
            next?.handle(request: request)
        }
    }
}

class ConcreteHandlerB: Handler {
    weak var next: Handler?

    func handle(request: Int) {
        if request > 10 && request <= 20 {
            print("Handled by B")
        } else {
            next?.handle(request: request)
        }
    }
}

let handlerA = ConcreteHandlerA()
let handlerB = ConcreteHandlerB()

handlerA.next = handlerB

handlerA.handle(request: 5)
handlerA.handle(request: 15)
handlerA.handle(request: 25)
```

## Command

[Command Design Pattern](/gofdesignpattern/command/command.md)

The Command pattern is a behavioral design pattern that turns a request into a
stand-alone object that contains all information about the request. This
transformation lets you pass requests as method arguments, delay or queue a
request’s execution, and support undoable operations.

```swift
protocol Command {
  func execute()
}

class Receiver {
  func actionA() { print("Action A") }
  func actionB() { print("Action B") }
}

class ConcreteCommandA: Command {
  private let receiver: Receiver
  init(receiver: Receiver) { self.receiver = receiver }
  func execute() { receiver.actionA() }
}

class ConcreteCommandB: Command {
  private let receiver: Receiver
  init(receiver: Receiver) { self.receiver = receiver }
  func execute() { receiver.actionB() }
}

class Invoker {
  private var command: Command?
  func setCommand(command: Command) { self.command = command }
  func executeCommand() { command?.execute() }
}

let receiver = Receiver()
let commandA = ConcreteCommandA(receiver: receiver)
let commandB = ConcreteCommandB(receiver: receiver)
let invoker = Invoker()

invoker.setCommand(command: commandA)
invoker.executeCommand()

invoker.setCommand(command: commandB)
invoker.executeCommand()
```

## Interpreter

[Interpreter Design Pattern](/gofdesignpattern/interpreter/interpreter.md)

Interpreter pattern is a design pattern that provides a way to evaluate language
grammar or expressions. It involves building an interpreter to interpret the
expressions of a language. This pattern is used in situations where there is a
language to be interpreted or domain-specific language expression, which must be
evaluated.

```swift
import Foundation

protocol Expression {
    func interpret(context: [String: Bool]) -> Bool
}

class Constant: Expression {
    private var value: Bool

    init(value: Bool) {
        self.value = value
    }

    func interpret(context: [String: Bool]) -> Bool {
        return value
    }
}

class Variable: Expression {
    private var name: String

    init(name: String) {
        self.name = name
    }

    func interpret(context: [String: Bool]) -> Bool {
        return context[name] ?? false
    }
}

class And: Expression {
    private var left, right: Expression

    init(left: Expression, right: Expression) {
        self.left = left
        self.right = right
    }

    func interpret(context: [String: Bool]) -> Bool {
        return left.interpret(context: context) && right.interpret(context: context)
    }
}

class Or: Expression {
    private var left, right: Expression

    init(left: Expression, right: Expression) {
        self.left = left
        self.right = right
    }

    func interpret(context: [String: Bool]) -> Bool {
        return left.interpret(context: context) || right.interpret(context: context)
    }
}

let a = Variable(name: "A")
let b = Variable(name: "B")
let expression = Or(left: And(left: a, right: b), right: Or(left: a, right: And(left: a, right: b)))
let context = ["A": true, "B": false]

print("Expression result:", expression.interpret(context: context))
```

## Iterator

[Iterator Design Pattern](/gofdesignpattern/iterator/iterator.md)

Iterator design pattern is a design pattern that provides a way to access the
elements of a collection object sequentially without exposing its underlying
representation. This pattern is useful to provide a standard way to traverse
through a group of objects and helps decouple the algorithms from the data
structures they operate upon.

```swift
struct MyCollection<T>: Sequence {
    private var items: [T] = []

    mutating func add(_ item: T) {
        items.append(item)
    }

    func makeIterator() -> AnyIterator<T> {
        return AnyIterator(items.makeIterator())
    }
}

var numbers = MyCollection<Int>()
numbers.add(1)
numbers.add(2)
numbers.add(3)

for num in numbers {
    print(num)
}
```

## Mediator

[Mediator Design Pattern](/gofdesignpattern/mediator/mediator.md)

The Mediator pattern is a behavioral design pattern that defines an object that
encapsulates how objects (colleagues) interact with each other. This pattern
promotes loose coupling by avoiding direct connections between colleagues, and
the mediator object is responsible for handling their interactions.

```swift
protocol ChatMediator {
    func sendMessage(_ msg: String, from user: User?)
    func addUser(_ user: User)
}

class ChatMediatorImpl: ChatMediator {
    private var users: [User] = []

    func addUser(_ user: User) {
        users.append(user)
    }

    func sendMessage(_ msg: String, from user: User?) {
        for u in users where u !== user {
            u.receive(msg)
        }
    }
}

class User {
    let mediator: ChatMediator
    let name: String

    init(mediator: ChatMediator, name: String) {
        self.mediator = mediator
        self.name = name
    }

    func send(_ msg: String) {
        mediator.sendMessage(msg, from: self)
    }

    func receive(_ msg: String) {
        print("\(name) received: \(msg)")
    }
}

let mediator: ChatMediator = ChatMediatorImpl()

let user1 = User(mediator: mediator, name: "Alice")
let user2 = User(mediator: mediator, name: "Bob")
let user3 = User(mediator: mediator, name: "Eve")

mediator.addUser(user1)
mediator.addUser(user2)
mediator.addUser(user3)

user1.send("Hello, everyone!")
```

## Memento

[Memento Design Pattern](/gofdesignpattern/memento/memento.md)

Memento pattern is a behavioral design pattern that allows an object
(originator) to save and restore its previous state without revealing the
structure or details of its internal state, thus enabling an undo mechanism. A
memento object is used to store the state of the originator, and a caretaker is
responsible for the memento's safe keeping.

```swift
// The Memento class
class Memento {
    private let state: String

    init(state: String) {
        self.state = state
    }

    func getState() -> String {
        return state
    }
}

// The Originator class
class Originator {
    private var state: String = ""

    func setState(state: String) {
        self.state = state
    }

    func save() -> Memento {
        return Memento(state: state)
    }

    func restore(memento: Memento) {
        self.state = memento.getState()
    }
}

// The Caretaker class
class Caretaker {
    private var mementos: [Memento] = []

    func save(originator: Originator) {
        mementos.append(originator.save())
    }

    func undo(originator: Originator) {
        if !mementos.isEmpty {
            originator.restore(memento: mementos.removeLast())
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

```swift
import Foundation

protocol Observer: AnyObject {
    func update(message: String)
}

class Subject {
    private var observers = [Observer]()

    func attach(observer: Observer) {
        observers.append(observer)
    }

    func notifyObservers(message: String) {
        observers.forEach { $0.update(message: message) }
    }
}

class ConcreteObserver: Observer {
    private let name: String

    init(name: String) {
        self.name = name
    }

    func update(message: String) {
        print("\(name) received the message: \(message)")
    }
}

let subject = Subject()
let observer1 = ConcreteObserver(name: "Observer1")
let observer2 = ConcreteObserver(name: "Observer2")

subject.attach(observer: observer1)
subject.attach(observer: observer2)

subject.notifyObservers(message: "Hello, Observers!")
```

## State

[State Design Pattern](/gofdesignpattern/state/state.md)

The State pattern is a behavioral design pattern that allows an object to change
its behavior when its internal state changes.

```swift
protocol State {
    func handle(context: Context)
}

class ConcreteStateA: State {
    func handle(context: Context) {
        context.changeState(newState: ConcreteStateB())
    }
}

class ConcreteStateB: State {
    func handle(context: Context) {
        context.changeState(newState: ConcreteStateA())
    }
}

class Context {
    private var state: State

    init(state: State) {
        self.state = state
    }

    func changeState(newState: State) {
        state = newState
    }

    func request() {
        state.handle(context: self)
    }
}

let context = Context(state: ConcreteStateA())
context.request()
context.request()
```

## Strategy

[Strategy Design Pattern](/gofdesignpattern/strategy/strategy.md)

The strategy pattern is a behavioral design pattern that defines a family of
algorithms, encapsulates each of them, and makes them interchangeable. It allows
the algorithm to vary independently from the clients that use it.

```swift
protocol Strategy {
    func doOperation(_ a: Int, _ b: Int) -> Int
}

struct Addition: Strategy {
    func doOperation(_ a: Int, _ b: Int) -> Int {
        return a + b
    }
}

struct Subtraction: Strategy {
    func doOperation(_ a: Int, _ b: Int) -> Int {
        return a - b
    }
}

struct Multiplication: Strategy {
    func doOperation(_ a: Int, _ b: Int) -> Int {
        return a * b
    }
}

struct Context {
    private let strategy: Strategy

    init(_ strategy: Strategy) {
        self.strategy = strategy
    }

    func executeStrategy(_ a: Int, _ b: Int) -> Int {
        return strategy.doOperation(a, b)
    }
}

let context = Context(Addition())
print("10 + 5 = \(context.executeStrategy(10, 5))")

let context2 = Context(Subtraction())
print("10 - 5 = \(context2.executeStrategy(10, 5))")

let context3 = Context(Multiplication())
print("10 * 5 = \(context3.executeStrategy(10, 5))")
```

## Template

[Template Design Pattern](/gofdesignpattern/template/template.md)

The template pattern is a behavioral design pattern that defines the program
skeleton of an algorithm in a method, called template method, which defers some
steps to subclasses. It lets subclasses redefine certain steps of an algorithm
without changing the algorithm's structure.

```swift
import Foundation

protocol Game {
  func initialize()
  func startGame()
  func endGame()
  func play()
}

extension Game {
  func play() {
    initialize()
    startGame()
    endGame()
  }
}

class Cricket: Game {
  func initialize() { print("Cricket Game Initialized!") }
  func startGame() { print("Cricket Game Started!") }
  func endGame() { print("Cricket Game Finished!") }
}

class Football: Game {
  func initialize() { print("Football Game Initialized!") }
  func startGame() { print("Football Game Started!") }
  func endGame() { print("Football Game Finished!") }
}

let game: Game = Cricket()
game.play()

let game2: Game = Football()
game2.play()
```

## Visitor

[Visitor Design Pattern](/gofdesignpattern/visitor/visitor.md)

Visitor pattern is a design pattern used in object-oriented programming that
promotes separation of concerns, allowing operations to be defined independently
on objects while traversing data structures, typically through a tree traversal
algorithm. Visitor pattern is an example of the **double dispatch technique**,
where the operation's implementation depends on both the **type of the
operation** and the **type of the data**.

```swift
import Foundation

protocol ShapeVisitor {
    func visit(circle: Circle)
    func visit(rectangle: Rectangle)
}

protocol Shape {
    func accept(visitor: ShapeVisitor)
}

class Circle: Shape {
    let radius: Double

    init(radius: Double) {
        self.radius = radius
    }

    func accept(visitor: ShapeVisitor) {
        visitor.visit(circle: self)
    }
}

class Rectangle: Shape {
    let length, width: Double

    init(length: Double, width: Double) {
        self.length = length
        self.width = width
    }

    func accept(visitor: ShapeVisitor) {
        visitor.visit(rectangle: self)
    }
}

class ShapeAreaVisitor: ShapeVisitor {
    var totalArea = 0.0

    func visit(circle: Circle) {
        totalArea += Double.pi * pow(circle.radius, 2)
    }

    func visit(rectangle: Rectangle) {
        totalArea += rectangle.length * rectangle.width
    }
}

let shapes: [Shape] = [Circle(radius: 5), Rectangle(length: 3, width: 4), Circle(radius: 2)]
let areaVisitor = ShapeAreaVisitor()

for shape in shapes {
    shape.accept(visitor: areaVisitor)
}

print("Total area: \(areaVisitor.totalArea)")
```
