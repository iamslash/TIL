- [Materials](#materials)
  - [Design Patterns @ TIL](#design-patterns--til)
  - [Design Patterns In Kotlin @ github](#design-patterns-in-kotlin--github)
- [Creational Pattern](#creational-pattern)
  - [Abstract Factory](#abstract-factory)
  - [Builder](#builder)
    - [Basic](#basic)
    - [Inner Builder Class](#inner-builder-class)
    - [@JvmOverloads](#jvmoverloads)
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

-----

# Materials

## [Design Patterns @ TIL](/designpattern/README.md)
## [Design Patterns In Kotlin @ github](https://github.com/dbacinski/Design-Patterns-In-Kotlin)

# Creational Pattern

## Abstract Factory

[Abstract Factory Pattern](/gofdesignpattern/abstractfactory/abstractfactory.md)

Abstract Factory is a design pattern that provides an interface for creating
families of related or dependent objects without specifying their concrete
classes.

```kotlin
package creational

interface Plant

class OrangePlant : Plant

class ApplePlant : Plant

abstract class PlantFactory {
    abstract fun makePlant(): Plant

    companion object {
        inline fun <reified T : Plant> createFactory(): PlantFactory = when (T::class) {
            OrangePlant::class -> OrangeFactory()
            ApplePlant::class  -> AppleFactory()
            else               -> throw IllegalArgumentException()
        }
    }
}

class AppleFactory : PlantFactory() {
    override fun makePlant(): Plant = ApplePlant()
}

class OrangeFactory : PlantFactory() {
    override fun makePlant(): Plant = OrangePlant()
}

fun main() {
    val plantFactory = PlantFactory.createFactory<OrangePlant>()
    val plant = plantFactory.makePlant()
    println("Created plant: $plant")
}
```

## Builder

Kotlin 의 경우 builder pattern, DSL pattern 을 이용하여 객체를 생성하는 것보다
primary constructor 를 이용하는 편이 좋다. [Item 34: Consider a primary
constructor with named optional
arguments](#item-34-consider-a-primary-constructor-with-named-optional-arguments)

### Basic

```kotlin
package creational

import java.io.File

class Dialog {

    fun showTitle() = println("showing title")

    fun setTitle(text: String) = println("setting title text $text")

    fun setTitleColor(color: String) = println("setting title color $color")

    fun showMessage() = println("showing message")

    fun setMessage(text: String) = println("setting message $text")

    fun setMessageColor(color: String) = println("setting message color $color")

    fun showImage(bitmapBytes: ByteArray) = println("showing image with size ${bitmapBytes.size}")

    fun show() = println("showing dialog $this")
}

//Builder:
class DialogBuilder() {
    constructor(init: DialogBuilder.() -> Unit) : this() {
        init()
    }

    private var titleHolder: TextView? = null
    private var messageHolder: TextView? = null
    private var imageHolder: File? = null

    fun title(init: TextView.() -> Unit) {
        titleHolder = TextView().apply { init() }
    }

    fun message(init: TextView.() -> Unit) {
        messageHolder = TextView().apply { init() }
    }

    fun image(init: () -> File) {
        imageHolder = init()
    }

    fun build(): Dialog {
        val dialog = Dialog()

        titleHolder?.apply {
            dialog.setTitle(text)
            dialog.setTitleColor(color)
            dialog.showTitle()
        }

        messageHolder?.apply {
            dialog.setMessage(text)
            dialog.setMessageColor(color)
            dialog.showMessage()
        }

        imageHolder?.apply {
            dialog.showImage(readBytes())
        }

        return dialog
    }

    class TextView {
        var text: String = ""
        var color: String = "#00000"
    }
}

fun main() {
    // Function that creates dialog builder and builds Dialog
    fun dialog(init: DialogBuilder.() -> Unit): Dialog {
        return DialogBuilder(init).build()
    }

    val dialog: Dialog = dialog {
        title {
            text = "Dialog Title"
        }
        message {
            text = "Dialog Message"
            color = "#333333"
        }
        image {
            File.createTempFile("image", "jpg")
        }
    }

    dialog.show()
}
```

### Inner Builder Class

[Abstract Factory Pattern](/gofdesignpattern/builder/builder.md)

```kotlin
data class Person private constructor(val builder: Builder) {
    val id: String = builder.id
    val pw: String = builder.pw
    val name: String? 
    val address: String?
    val email: String?

    init {
        name = builder.name
        address = builder.address
        email = builder.email
    }

    class Builder(val id: String, val pw: String) {
        var name: String? = null
        var address: String? = null
        var email: String? = null

        fun build(): Person {
            return Person(this)
        }

        fun name(name: String?): Builder {
            this.name = name
            return this
        }

        fun address(address: String?): Builder {
            this.address = address
            return this
        }

        fun email(email: String?): Builder {
            this.email = email
            return this
        }
    }
}

fun main() {
    val person = Person
        .Builder("AABBCCDD", "123456")
        .name("iamslash")
        .address("Irving Ave")
        .email("iamslash@gmail.com")
        .build()
    println(person)
}
```

### @JvmOverloads

* [@JvmOverloads](/kotlin/README.md#jvmoverloads)

```kotlin
data class Person @JvmOverloads constructor(
    val id: String,
    val pw: String,
    var name: String? = "",
    var address: String? = "",
    var email: String? = "",
)

fun main() {
    val person = Person(
        id = "AABBCCDD", 
        pw = "123456",
        name = "iamslash",
        address = "Irving Ave",
        email = "iamslash@gmail.com",
    )
    println(person)
}    
```

## Factory Method

[Factory Method Pattern](/gofdesignpattern/factorymethod/factorymethod.md)

The Factory Method Pattern provides an interface for creating objects in a
superclass, but allows subclasses to alter the type of objects that will be
created. It promotes loose coupling by removing the direct dependencies between
classes and follows the concept of "programming to an interface".

```kotlin
// Shape interface
interface Shape {
    fun draw()
}

// Concrete implementations of Shape
class Circle : Shape {
    override fun draw() {
        println("Drawing a circle.")
    }
}

class Rectangle : Shape {
    override fun draw() {
        println("Drawing a rectangle.")
    }
}

class Triangle : Shape {
    override fun draw() {
        println("Drawing a triangle.")
    }
}

// Abstract factory class
abstract class ShapeFactory {
    abstract fun createShape(): Shape
}

// Concrete factory classes
class CircleFactory : ShapeFactory() {
    override fun createShape(): Shape {
        return Circle()
    }
}

class RectangleFactory : ShapeFactory() {
    override fun createShape(): Shape {
        return Rectangle()
    }
}

class TriangleFactory : ShapeFactory() {
    override fun createShape(): Shape {
        return Triangle()
    }
}

// Client code
fun main() {
    // Create a circle factory and create a circle
    val circleFactory = CircleFactory()
    val circle = circleFactory.createShape()
    circle.draw()

    // Create a rectangle factory and create a rectangle
    val rectangleFactory = RectangleFactory()
    val rectangle = rectangleFactory.createShape()
    rectangle.draw()

    // Create a triangle factory and create a triangle
    val triangleFactory = TriangleFactory()
    val triangle = triangleFactory.createShape()
    triangle.draw()
}
```

## Prototype

[Prototype Pattern](/gofdesignpattern/prototype/prototype.md)

The prototype pattern is a creational design pattern that allows you to create
new objects by copying an existing object, called the prototype. This pattern is
especially useful when creating a new object instance is expensive in terms of
resources or time, and the existing object shares most of its properties with
the new object.

```kotlin
abstract class Car(
    var make: String,
    var model: String,
    var color: String
): Cloneable {

    public override fun clone(): Car {
        try {
            return super.clone() as Car
        } catch (e: CloneNotSupportedException) {
            throw RuntimeException("Failed to clone object", e)
        }
    }
}

class ToyotaCar : Car("Toyota", "Camry", "Blue")

fun main() {
    val car1 = ToyotaCar()

    // Create a new object using the prototype
    val car2 = car1.clone()

    // Verify the cloned object's properties
    println("Car1 Make: ${car1.make}")
    println("Car1 Model: ${car1.model}")
    println("Car1 Color: ${car1.color}")
    println("-------------")
    println("Car2 Make: ${car2.make}")
    println("Car2 Model: ${car2.model}")
    println("Car2 Color: ${car2.color}")

    // Modify the cloned object's color
    car2.color = "Red"

    // Verify that cloned object's properties are modified
    println("-------------")
    println("Car1 Color: ${car1.color}")
    println("Car2 Color: ${car2.color}")
}
```

## Singleton

[Singleton Design Pattern](/gofdesignpattern/singleton/singleton.md)

Singleton pattern is a design pattern that restricts the instantiation of a
class to only one instance, which is globally accessible in an application. This
is useful when exactly one object is needed to coordinate actions across the
system.

```kotlin
object Singleton {

    // An example of a method
    fun printMessage() {
        println("Hello from Singleton!")
    }
}

// Test the Singleton
fun main() {
    val singleton = Singleton
    singleton.printMessage()
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

```kotlin
interface Temperature {
    fun getCelsiusTemperature(): Double
}

class CelsiusTemperature(private val temperature: Double) : Temperature {
    override fun getCelsiusTemperature() = temperature
}

class FahrenheitTemperature(private val temperature: Double) {
    fun getFahrenheitTemperature() = temperature
}

class FahrenheitToCelsiusAdapter(
    private val fahrenheitTemperature: FahrenheitTemperature
) : Temperature {
    override fun getCelsiusTemperature() =
        (fahrenheitTemperature.getFahrenheitTemperature() - 32) * 5.0 / 9.0
}

fun main() {
    val celsiusTemp = CelsiusTemperature(100.0)
    println("Celsius Temperature: ${celsiusTemp.getCelsiusTemperature()}")

    val fahrenheitTemp = FahrenheitTemperature(212.0)
    val fahrenheitToCelsiusTemp = FahrenheitToCelsiusAdapter(fahrenheitTemp)
    println("Fahrenheit Temperature in Celsius: ${fahrenheitToCelsiusTemp.getCelsiusTemperature()}")
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

```kotlin
// Abstraction
interface Color {
    fun applyColor()
}

// Implementor
class RedColor : Color {
    override fun applyColor() {
        println("Applying red color.")
    }
}

class BlueColor : Color {
    override fun applyColor() {
        println("Applying blue color.")
    }
}

// Bridge
abstract class Shape(protected val color: Color) {
    abstract fun draw()
}

class Circle(color: Color) : Shape(color) {
    override fun draw() {
        print("Drawing circle: ")
        color.applyColor()
    }
}

class Square(color: Color) : Shape(color) {
    override fun draw() {
        print("Drawing square: ")
        color.applyColor()
    }
}

// Client
fun main() {
    val redCircle: Shape = Circle(RedColor())
    val blueSquare: Shape = Square(BlueColor())

    redCircle.draw()
    blueSquare.draw()
}
```

## Composite

[Composite Design Pattern](/gofdesignpattern/composite/composite.md)

Composite pattern is a structural design pattern that allows you to compose
objects into tree structures to represent part-whole hierarchies. It lets
clients treat individual objects and compositions uniformly.

```kotlin
interface FileSystem {
    fun showFileSystem()
}

// Leaf
class File(private val fileName: String) : FileSystem {
    override fun showFileSystem() {
        println("File: $fileName")
    }
}

// Composite
class Directory(private val directoryName: String) : FileSystem {
    private val fileSystemList = mutableListOf<FileSystem>()

    fun addFileSystem(fileSystem: FileSystem) {
        fileSystemList.add(fileSystem)
    }

    override fun showFileSystem() {
        println("\nDirectory: $directoryName")
        for (fileSystem in fileSystemList) {
            fileSystem.showFileSystem()
        }
    }
}

fun main() {
    val rootDirectory = Directory("Root")

    val dir1 = Directory("Directory1")
    dir1.addFileSystem(File("File1.txt"))
    dir1.addFileSystem(File("File2.txt"))
    rootDirectory.addFileSystem(dir1)

    val dir2 = Directory("Directory2")
    dir2.addFileSystem(File("File3.txt"))
    dir2.addFileSystem(File("File4.txt"))
    rootDirectory.addFileSystem(dir2)

    rootDirectory.showFileSystem()
}
```

## Decorator

[Decorator Design Pattern](/gofdesignpattern/decorator/decorator.md)

The decorator pattern is a structural design pattern that allows adding new
functionality to an object without altering its structure. It involves a set of
decorator classes that are used to wrap concrete components. The decorator
classes mirror the type of components they can decorate but add or override
behavior.

```kotlin
interface Car {
    fun getDescription(): String
    fun getCost(): Double
}

class BasicCar : Car {
    override fun getDescription() = "Basic Car"

    override fun getCost() = 10000.0
}

abstract class CarDecorator(private val car: Car) : Car {

    override fun getDescription(): String {
        return car.getDescription()
    }

    override fun getCost(): Double {
        return car.getCost()
    }
}

class LuxuryPackage(car: Car) : CarDecorator(car) {

    override fun getDescription() = "${super.getDescription()}, Luxury Package"

    override fun getCost() = super.getCost() + 5000
}

class SportPackage(car: Car) : CarDecorator(car) {

    override fun getDescription() = "${super.getDescription()}, Sport Package"

    override fun getCost() = super.getCost() + 3500
}

fun main() {
    val basicCar: Car = BasicCar()

    val luxuryCar: Car = LuxuryPackage(basicCar)
    println("${luxuryCar.getDescription()} cost: ${luxuryCar.getCost()}")

    val sportCar: Car = SportPackage(basicCar)
    println("${sportCar.getDescription()} cost: ${sportCar.getCost()}")

    val luxurySportCar: Car = LuxuryPackage(SportPackage(basicCar))
    println("${luxurySportCar.getDescription()} cost: ${luxurySportCar.getCost()}")
}
```

## Facade

[Facade Design Pattern](/gofdesignpattern/facade/facade.md)

Facade Pattern is a structural design pattern that provides a simplified
interface to a complex subsystem. It involves a single class that represents an
entire subsystem, hiding its complexity from the client.

```kotlin
// Subsystem classes
class CPU {
    fun start() {
        println("Initialize CPU")
    }
}

class HardDrive {
    fun read() {
        println("Read Hard Drive")
    }
}

class Memory {
    fun load() {
        println("Load memory")
    }
}

// Facade
class ComputerFacade {
    private val cpu = CPU()
    private val hardDrive = HardDrive()
    private val memory = Memory()

    fun startComputer() {
        cpu.start()
        memory.load()
        hardDrive.read()
    }
}

// Client
fun main() {
    val computerFacade = ComputerFacade()
    computerFacade.startComputer()
}
```

## Flyweight

[Flyweight Design Pattern](/gofdesignpattern/flyweight/flyweight.md)

The Flyweight pattern is a structural design pattern that minimizes memory usage
by sharing articles of data among several objects, especially when dealing with
a large number of similar objects. The main idea is to create shared objects,
called "Flyweights," which can be reused representing the same state.

```kotlin
interface Shape {
    fun draw(x: Int, y: Int)
}

class Circle(private val color: String) : Shape {
    override fun draw(x: Int, y: Int) {
        println("Drawing a $color circle at ($x,$y)")
    }
}

object ShapeFactory {
    private val shapes = mutableMapOf<String, Shape>()

    fun getCircle(color: String): Shape {
        return shapes.getOrPut(color) {
            println("Creating new $color circle")
            Circle(color)
        }
    }
}

fun main() {
    val redCircle = ShapeFactory.getCircle("red")
    redCircle.draw(10, 20)

    val blueCircle = ShapeFactory.getCircle("blue")
    blueCircle.draw(30, 40)

    val redCircle2 = ShapeFactory.getCircle("red")
    redCircle2.draw(50, 60)
}
```

## Proxy

[Proxy Deisgn Pattern](/gofdesignpattern/proxy/proxy.md)

Proxy Pattern is a structural design pattern that involves a set of intermediary
objects, or proxies, that sit between a client and the actual target object.
Proxies are useful for controlling access, managing resource-intensive
operations, or adding extra functionality to objects without adding complexity
to the real object.

```kotlin
interface RealObject {
    fun performAction()
}

class ActualObject : RealObject {
    override fun performAction() {
        println("Performing action in the actual object")
    }
}

class ProxyObject(private val realObject: RealObject) : RealObject {
    override fun performAction() {
        println("Performing action in the proxy object")
        realObject.performAction()
    }
}

fun main() {
    val actualObject: RealObject = ActualObject()
    val proxyObject: RealObject = ProxyObject(actualObject)
    proxyObject.performAction() // using proxy object instead of the actual object
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

```kotlin
interface Handler {
    var next: Handler?
    
    fun handle(request: Int)

    fun setNext(next: Handler) {
        this.next = next
    }
}

class ConcreteHandlerA : Handler {
    override var next: Handler? = null

    override fun handle(request: Int) {
        if (request <= 10) {
            println("Handled by A")
        } else {
            next?.handle(request)
        }
    }
}

class ConcreteHandlerB : Handler {
    override var next: Handler? = null

    override fun handle(request: Int) {
        if (request > 10 && request <= 20) {
            println("Handled by B")
        } else {
            next?.handle(request)
        }
    }
}

fun main() {
    val handlerA = ConcreteHandlerA()
    val handlerB = ConcreteHandlerB()

    handlerA.setNext(handlerB)

    handlerA.handle(5)
    handlerA.handle(15)
    handlerA.handle(25)
}
```

## Command

[Command Design Pattern](/gofdesignpattern/command/command.md)

The Command pattern is a behavioral design pattern that turns a request into a
stand-alone object that contains all information about the request. This
transformation lets you pass requests as method arguments, delay or queue a
request’s execution, and support undoable operations.

```kotlin
interface Command {
  fun execute()
}

class Receiver {
  fun actionA() = println("Action A")
  fun actionB() = println("Action B")
}

class ConcreteCommandA(private val receiver: Receiver) : Command {
  override fun execute() = receiver.actionA()
}

class ConcreteCommandB(private val receiver: Receiver) : Command {
  override fun execute() = receiver.actionB()
}

class Invoker {
  private lateinit var command: Command
  fun setCommand(command: Command) { this.command = command }
  fun executeCommand() = command.execute()
}

fun main() {
  val receiver = Receiver()
  val commandA = ConcreteCommandA(receiver)
  val commandB = ConcreteCommandB(receiver)
  val invoker = Invoker()

  invoker.setCommand(commandA)
  invoker.executeCommand()

  invoker.setCommand(commandB)
  invoker.executeCommand()
}
```

## Interpreter

[Interpreter Design Pattern](/gofdesignpattern/interpreter/interpreter.md)

Interpreter pattern is a design pattern that provides a way to evaluate language
grammar or expressions. It involves building an interpreter to interpret the
expressions of a language. This pattern is used in situations where there is a
language to be interpreted or domain-specific language expression, which must be
evaluated.

```kotlin
interface Expression {
    fun interpret(context: Map<String, Boolean>): Boolean
}

class Constant(private val value: Boolean) : Expression {
    override fun interpret(context: Map<String, Boolean>) = value
}

class Variable(private val name: String) : Expression {
    override fun interpret(context: Map<String, Boolean>) = context[name] ?: false
}

class And(private val left: Expression, private val right: Expression) : Expression {
    override fun interpret(context: Map<String, Boolean>) = left.interpret(context) && right.interpret(context)
}

class Or(private val left: Expression, private val right: Expression) : Expression {
    override fun interpret(context: Map<String, Boolean>) = left.interpret(context) || right.interpret(context)
}

fun main() {
    val a = Variable("A")
    val b = Variable("B")
    val expression = Or(And(a, b), Or(a, And(a, b)))
    val context = mapOf("A" to true, "B" to false)

    println("Expression result: ${expression.interpret(context)}")
}
```

## Iterator

[Iterator Design Pattern](/gofdesignpattern/iterator/iterator.md)

Iterator design pattern is a design pattern that provides a way to access the
elements of a collection object sequentially without exposing its underlying
representation. This pattern is useful to provide a standard way to traverse
through a group of objects and helps decouple the algorithms from the data
structures they operate upon.

```kotlin
class MyCollection<T> : Iterable<T> {
    private val items = mutableListOf<T>()

    fun add(item: T) {
        items.add(item)
    }

    override fun iterator(): Iterator<T> = items.iterator()
}

fun main() {
    val numbers = MyCollection<Int>()
    numbers.add(1)
    numbers.add(2)
    numbers.add(3)

    for (num in numbers) {
        println(num)
    }
}
```

## Mediator

[Mediator Design Pattern](/gofdesignpattern/mediator/mediator.md)

The Mediator pattern is a behavioral design pattern that defines an object that
encapsulates how objects (colleagues) interact with each other. This pattern
promotes loose coupling by avoiding direct connections between colleagues, and
the mediator object is responsible for handling their interactions.

```kotlin
interface ChatMediator {
    fun sendMessage(msg: String, user: User?)
    fun addUser(user: User)
}

class ChatMediatorImpl : ChatMediator {
    private val users: MutableList<User> = ArrayList()

    override fun addUser(user: User) {
        users.add(user)
    }

    override fun sendMessage(msg: String, user: User?) {
        users.filter { it != user }.forEach { it.receive(msg) }
    }
}

abstract class User(
    protected val mediator: ChatMediator, 
    protected val name: String
) {
    abstract fun send(msg: String)
    abstract fun receive(msg: String)
}

class UserImpl(mediator: ChatMediator, name: String) : User(mediator, name) {
    override fun send(msg: String) {
        mediator.sendMessage(msg, this)
    }

    override fun receive(msg: String) {
        println("$name received: $msg")
    }
}

fun main() {
    val mediator: ChatMediator = ChatMediatorImpl()

    val user1 = UserImpl(mediator, "Alice")
    val user2 = UserImpl(mediator, "Bob")
    val user3 = UserImpl(mediator, "Eve")

    mediator.addUser(user1)
    mediator.addUser(user2)
    mediator.addUser(user3)

    user1.send("Hello, everyone!")
}
```

## Memento

[Memento Design Pattern](/gofdesignpattern/memento/memento.md)

Memento pattern is a behavioral design pattern that allows an object
(originator) to save and restore its previous state without revealing the
structure or details of its internal state, thus enabling an undo mechanism. A
memento object is used to store the state of the originator, and a caretaker is
responsible for the memento's safe keeping.

```kotlin
// The Memento class
data class Memento(val state: String)

// The Originator class
class Originator {
    var state: String = ""

    fun save(): Memento = Memento(state)
    fun restore(memento: Memento) {
        state = memento.state
    }
}

// The Caretaker class
class Caretaker {
    private val mementos = Stack<Memento>()

    fun save(originator: Originator) {
        mementos.push(originator.save())
    }

    fun undo(originator: Originator) {
        if (mementos.isNotEmpty()) {
            originator.restore(mementos.pop())
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

```kotlin
interface Observer {
    fun update(message: String)
}

class Subject {
    private val observers = mutableListOf<Observer>()

    fun attach(observer: Observer) {
        observers.add(observer)
    }

    fun notifyObservers(message: String) {
        observers.forEach { it.update(message) }
    }
}

class ConcreteObserver(private val name: String) : Observer {
    override fun update(message: String) {
        println("$name received the message: $message")
    }
}

fun main() {
    val subject = Subject()
    val observer1 = ConcreteObserver("Observer1")
    val observer2 = ConcreteObserver("Observer2")

    subject.attach(observer1)
    subject.attach(observer2)

    subject.notifyObservers("Hello, Observers!")
}
```

## State

[State Design Pattern](/gofdesignpattern/state/state.md)

The State pattern is a behavioral design pattern that allows an object to change
its behavior when its internal state changes.

```kotlin
interface State {
  fun handle(context: Context)
}

class ConcreteStateA : State {
  override fun handle(context: Context) {
    context.changeState(ConcreteStateB())
  }
}

class ConcreteStateB : State {
  override fun handle(context: Context) {
    context.changeState(ConcreteStateA())
  }
}

class Context(var state: State) {
  fun changeState(newState: State) {
    state = newState
  }

  fun request() {
    state.handle(this)
  }
}

fun main() {
  val context = Context(ConcreteStateA())
  context.request()
  context.request()
}
```

## Strategy

[Strategy Design Pattern](/gofdesignpattern/strategy/strategy.md)

The strategy pattern is a behavioral design pattern that defines a family of
algorithms, encapsulates each of them, and makes them interchangeable. It allows
the algorithm to vary independently from the clients that use it.

```kotlin
interface Strategy {
    fun doOperation(a: Int, b: Int): Int
}

class Addition : Strategy {
    override fun doOperation(a: Int, b: Int) = a + b
}

class Subtraction : Strategy {
    override fun doOperation(a: Int, b: Int) = a - b
}

class Multiplication : Strategy {
    override fun doOperation(a: Int, b: Int) = a * b
}

class Context(private val strategy: Strategy) {
    fun executeStrategy(a: Int, b: Int) = strategy.doOperation(a, b)
}

fun main() {
    var context = Context(Addition())
    println("10 + 5 = ${context.executeStrategy(10, 5)}")

    context = Context(Subtraction())
    println("10 - 5 = ${context.executeStrategy(10, 5)}")

    context = Context(Multiplication())
    println("10 * 5 = ${context.executeStrategy(10, 5)}")
}
```

## Template

[Template Design Pattern](/gofdesignpattern/template/template.md)

The template pattern is a behavioral design pattern that defines the program
skeleton of an algorithm in a method, called template method, which defers some
steps to subclasses. It lets subclasses redefine certain steps of an algorithm
without changing the algorithm's structure.

```kotlin
abstract class Game {
  abstract fun initialize()
  abstract fun startGame()
  abstract fun endGame()

  fun play() {
    initialize()
    startGame()
    endGame()
  }
}

class Cricket : Game() {
  override fun initialize() { println("Cricket Game Initialized!") }
  override fun startGame() { println("Cricket Game Started!") }
  override fun endGame() { println("Cricket Game Finished!") }
}

class Football : Game() {
  override fun initialize() { println("Football Game Initialized!") }
  override fun startGame() { println("Football Game Started!") }
  override fun endGame() { println("Football Game Finished!") }
}

fun main() {
  var game: Game = Cricket()
  game.play()

  game = Football()
  game.play()
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

```kotlin
interface ShapeVisitor {
    fun visit(circle: Circle)
    fun visit(rectangle: Rectangle)
}

interface Shape {
    fun accept(visitor: ShapeVisitor)
}

data class Circle(val radius: Double) : Shape {
    override fun accept(visitor: ShapeVisitor) {
        visitor.visit(this)
    }
}

data class Rectangle(val length: Double, val width: Double) : Shape {
    override fun accept(visitor: ShapeVisitor) {
        visitor.visit(this)
    }
}

class ShapeAreaVisitor : ShapeVisitor {
    var totalArea = 0.0
    
    override fun visit(circle: Circle) {
        totalArea += Math.PI * Math.pow(circle.radius, 2.0)
    }

    override fun visit(rectangle: Rectangle) {
        totalArea += rectangle.length * rectangle.width
    }
}

fun main() {
    val shapes = listOf(Shape(Circle(5.0)), Shape(Rectangle(3.0, 4.0)), Shape(Circle(2.0)));
    val areaVisitor = ShapeAreaVisitor()

    for (shape in shapes) {
        shape.accept(areaVisitor)
    }

    println("Total area: ${areaVisitor.totalArea}")
}
```
