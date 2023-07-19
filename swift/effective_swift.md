- [Materials](#materials)
- [Creating and Destroying Objects](#creating-and-destroying-objects)
  - [item1: Consider static factory methods instead of constructors](#item1-consider-static-factory-methods-instead-of-constructors)
  - [item2: Consider a builder when faced with many constructor parameters](#item2-consider-a-builder-when-faced-with-many-constructor-parameters)
  - [item3: Enforce the singleton property with a private constructor or an enum type](#item3-enforce-the-singleton-property-with-a-private-constructor-or-an-enum-type)
  - [item4: Enforce noninstantiability with a private constructor](#item4-enforce-noninstantiability-with-a-private-constructor)
  - [item5: Prefer dependency injection to hardwiring resources](#item5-prefer-dependency-injection-to-hardwiring-resources)
  - [item6: Avoid creating unnecessary ob](#item6-avoid-creating-unnecessary-ob)
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

----

# Materials

* [Effective Swift](https://theswiftists.github.io/effective-swift/)
  * Effective Java 를 Swift 로 옮긴 것

# Creating and Destroying Objects

## item1: Consider static factory methods instead of constructors

In Swift, you can use static factory methods instead of initializers to provide
more explicit and readable ways of creating objects.

```swift
class Car {
    var model: String
    
    private init(model: String) {
        self.model = model
    }
    
    static func createTesla() -> Car {
        return Car(model: "Tesla")
    }
    
    static func createBMW() -> Car {
        return Car(model: "BMW")
    }
}

let car1 = Car.createTesla()
let car2 = Car.createBMW()
```

## item2: Consider a builder when faced with many constructor parameters

The Builder pattern can be useful when dealing with many constructor parameters,
especially when some parameters are optional.

```swift
struct User {
    let firstName: String
    let lastName: String
    let age: Int?
    let address: String?
}

class UserBuilder {
    private var firstName = ""
    private var lastName = ""
    private var age: Int?
    private var address: String?
    
    func setFirstName(_ firstName: String) -> UserBuilder {
        self.firstName = firstName
        return self
    }
    
    func setLastName(_ lastName: String) -> UserBuilder {
        self.lastName = lastName
        return self
    }
    
    func setAge(_ age: Int) -> UserBuilder {
        self.age = age
        return self
    }
    
    func setAddress(_ address: String) -> UserBuilder {
        self.address = address
        return self
    }
    
    func build() -> User {
        return User(firstName: firstName, lastName: lastName, age: age, address: address)
    }
}

let user = UserBuilder()
    .setFirstName("John")
    .setLastName("Doe")
    .setAge(30)
    .setAddress("123 Main St")
    .build()
```

## item3: Enforce the singleton property with a private constructor or an enum type

Swift can enforce the Singleton pattern by using a static constant and a private
initializer.

```swift
class Singleton {
    static let shared = Singleton()
    
    private init() {}
}
```

## item4: Enforce noninstantiability with a private constructor

To prevent a class from being instantiated, Swift offers the option of using a
private initializer.

```swift
class Utility {
    private init() {
        fatalError("This class cannot be instantiated.")
    }
    
    static func someUtilityFunction() {
        print("Performing a utility task...")
    }
}

Utility.someUtilityFunction()
```

## item5: Prefer dependency injection to hardwiring resources

In order to make the code more modular and testable, you can use dependency
injection instead of tightly coupled resources.

```swift
protocol Database {
    func saveData(data: String)
}

class User {
    var database: Database
    
    init(database: Database) {
        self.database = database
    }
    
    func saveUserData(data: String) {
        database.saveData(data: data)
    }
}

class SQLDatabase: Database {
    func saveData(data: String) {
        print("Saving data to SQL Database: \(data)")
    }
}

let database = SQLDatabase()
let user = User(database: database)
user.saveUserData(data: "John Doe")
```

## item6: Avoid creating unnecessary ob

In order to reduce memory usage and improve performance, avoid creating
unnecessary objects and use existing instances or constants whenever possible.

```swift
class ExpensiveObject {
    init() {
        print("Expensive object created!")
    }
}

func processValue(value: ExpensiveObject) {
    // Do some processing
}

let obj1 = ExpensiveObject()
processValue(value: obj1)
```

## item7: Eliminate obsolete object references

Swift has **Automatic Reference Counting (ARC)** which helps to eliminate unused
objects automatically. However, you should still ensure that references are
properly cleaned up to avoid memory leaks.

```swift
class Observer {
    func observe(_ object: AnyObject) {
        NotificationCenter.default.addObserver(self, selector: #selector(handleNotification(_:)), name: Notification.Name(rawValue: "event"), object: object)
    }
    
    @objc func handleNotification(_ notification: Notification) {
        print("Notification received")
    }
    
    deinit {
        NotificationCenter.default.removeObserver(self)
    }
}
```

## item8: Avoid finalizers and cleaners

Swift has **Automatic Reference Counting (ARC)** which manages memory
automatically. Generally, you don't need to worry about deallocating memory.
However, in some cases, when you need to perform some cleanup, use
deinitialization.

```swift
class File {
    init() {
        print("File opened.")
    }
       
    deinit {
        print("File closed.")
    }
}
```

## item9: Prefer try-with-resources to try-finally

In Swift, you can use the `defer` keyword to ensure that some code is executed
right before the current scope is exited. This is similar to try-finally in
other languages.

```swift
 func readFile() {
    let file = File()
    
    defer {
        print("Cleaning up...")
    }
    
    // Read the file, throw errors if needed
}
```

# Methods Common to All Objects

## item10: Obey the general contract when overriding equals

When you override the equals method, make sure that you follow these rules: (1)
**Reflexive**, (2) **Symmetric**, (3) **Transitive**, (4) **Consistent**, and (5) **Non-nullity**.

```swift
class Person: Equatable {
    var name: String
    
    init(_ name: String) {
        self.name = name
    }
    
    static func == (lhs: Person, rhs: Person) -> Bool {
        return lhs.name == rhs.name
    }
}
```

## item11: Always override hashCode when you override equals

When you override the `equals` method, make sure to also override the `hash`
method. This ensures that the objects which are equal have the same hash code.

```swift
extension Person: Hashable {
    func hash(into hasher: inout Hasher) {
        hasher.combine(name)
    }
}
```

## item12: Always override toString

Always override the `description` computed property to provide a useful
description of the object.

```swift
extension Person: CustomStringConvertible {
    var description: String {
        return "Person[name: \(name)]"
    }
}
```

## item13: Override clone judiciously

Swift does not have a built-in `clone` method. You can create a custom method
for copying objects or conform your object to `NSCopying` protocol if you need
deep copying in classes.

```swift
class Person: NSObject, NSCopying {
    // ...
    
    func copy(with zone: NSZone? = nil) -> Any {
        return Person(name)
    }
}
```

## item14: Consider implementing Comparable

`Comparable` allows you to compare instances of your type, allowing them to be sorted or compared for equality.

```swift
extension Person: Comparable {
    static func < (lhs: Person, rhs: Person) -> Bool {
        return lhs.name < rhs.name
    }
}
```

# Classes and Interfaces

## item15: Minimize the accessibility of classes and members

Keep members private and only expose the necessary parts of the interface.

```swift
class Information {
    private var secret: String = "This is a secret"
    
    func revealSecret() -> String {
        return secret
    }
}
```

## item16: In public classes, use accessor methods, not public fields

Always use accessors instead of directly exposing class fields.

```swift
class Temperature {
    private var celsius: Double
    
    init(celsius: Double) {
        self.celsius = celsius
    }
    
    var fahrenheit: Double {
        get { return celsius * 9 / 5 + 32 }
        set { celsius = (newValue - 32) * 5 / 9 }
    }
}
```

## item17: Minimize mutability

Favor the use of `let` for properties that don't need to be changed after
initialization.

```swift
struct ImmutablePerson {
    let name: String
    let age: Int
}
```

## item18: Favor composition over inheritance

Use composition to build complex objects from simpler ones.

```swift
class Vehicle {
    let engine: Engine
    let wheels: [Wheel]
    
    init(engine: Engine, wheels: [Wheel]) {
        self.engine = engine
        self.wheels = wheels
    }
}
```

## item19: Design and document for inheritance or else prohibit it

Consider making classes final if you don't want them to be inherited.

```swift
final class NoInheritanceAllowed {
    // ...
}
```

## item20: Prefer interfaces to abstract classes

Swift doesn't have abstract classes or interfaces. Instead, use protocols to
define interfaces.

```swift
protocol Drawable {
    func draw() -> String
}

struct Circle: Drawable {
    func draw() -> String {
        return "Drawing a Circle"
    }
}
```

## item21: Design interfaces for posterity

Make sure to keep protocols simple and effective.

```swift
protocol Greeter {
    func greet(name: String) -> String
}
```

## item22: Use interfaces only to define types

Protocols should be used to define a type, not to provide implementation. You
can use protocol extensions for common implementations.

```swift
protocol Flyable {
    func fly() -> String
}

extension Flyable {
    func fly() -> String {
        return "I can fly!"
    }
}

struct Bird: Flyable {}
```

## item23: Prefer class hierarchies to tagged classes

Swift enums with associated types can achieve a similar goal.

```swift
enum Shape {
    case circle(radius: Double)
    case rectangle(width: Double, height: Double)
}

let rectangle = Shape.rectangle(width: 10, height: 5)
```

## item24: Favor static member classes over nonstatic

Static members are preferred because they don't hold a reference to their
enclosing instance.

```swift
class EnclosingClass {
    static var staticProperty: String = "I'm static"

    class InnerClass {
        static var staticProperty: String = "I'm also static"
    }
}
```

## item25: Limit source files to a single top-level class

Structure your code so that each file only contains a single top-level class.
This makes it easier to understand the purpose and responsibility of each file.

```swift
// MyClass.swift
class MyClass {
    // Implementation here
}

// AnotherClass.swift
class AnotherClass {
    // Implementation here
}
```

# Generics

## item26: Don’t use raw types

Avoid using raw types in Swift, as they can lead to runtime issues.

The concept of raw types isn't applicable to Swift like it is in Java, so no
example is necessary.

## item27: Eliminate unchecked warnings

Swift is a type-safe language, meaning it doesn't have warnings related to
unchecked type casts. Ensure your code remains type-safe by handling optional
unwrapping properly.

```swift
if let intValue = someValue as? Int {
    print("The value is an integer: \(intValue)")
} else {
    print("The value is not an integer")
}
```

## item28: Prefer lists to arrays

Prefer using Swift's `Array` and `Set` types instead of C-style arrays for
better type safety and usability.

```swift
var listOfIntegers: [Int] = [1, 2, 3]
let setOfStrings: Set<String> = ["apple", "orange", "banana"]
```

## item29: Favor generic types

Make use of generics for more reusable and type-safe code.

```swift
struct Stack<Element> {
    private var elements: [Element] = []

    mutating func push(_ item: Element) {
        elements.append(item)
    }

    mutating func pop() -> Element? {
        return elements.popLast()
    }
}
```

## item30: Favor generic methods

Prefer creating generic methods for higher levels of code reuse and type safety.

```swift
func addTwoItems<T: Numeric>(item1: T, item2: T) -> T {
    return item1 + item2
}
```

## item31: Use bounded wildcards to increase API flexibility

Swift uses associated types and generic constraints to bound generic types
instead of using wildcards.

```swift
protocol HasArea {
    associatedtype Area: Numeric
    var area: Area { get }
}

func compareAreas<T: HasArea, U: HasArea>(object1: T, object2: U) -> Bool where T.Area == U.Area {
    return object1.area == object2.area
}
```

## item32: Combine generics and varargs judiciously

Use variadic parameters in Swift to make your functions more idiomatic and
expressive.

```swift
func sum<T: Numeric>(_ elements: T...) -> T {
    return elements.reduce(0, { $0 + $1 })
}
```

## item33: Consider typesafe heterogeneous containers

Use Swift's protocol-oriented programming to define type-safe heterogeneous
containers.

```swift
protocol Storable { }

struct HeterogeneousStorage {
    private var storage = [ObjectIdentifier: Storable]()

    mutating func setValue<T: Storable>(_ value: T, forType type: T.Type) {
        storage[ObjectIdentifier(type)] = value
    }

    func getValue<T: Storable>(forType type: T.Type) -> T? {
        return storage[ObjectIdentifier(type)] as? T
    }
}
```

# Enums and Annotations

## item34: Use enums instead of int constants

Prefer using Swift's `enum` types over raw Integer constants for better type
safety and code semantics.

```swift
enum Direction {
    case north, east, south, west
}
```

## item35: Use instance fields instead of ordinals

Swift's enums don't have an ordinal automatically. Define an instance property
for an enum if an associated value is needed.

```swift
enum HttpStatusCode: Int {
    case ok = 200
    case notFound = 404
}
```

## item36: Use EnumSet instead of bit fields

Swift doesn't have an EnumSet like Java; but, you can use `OptionSet` to achieve
similar results.

```swift
struct FontStyle: OptionSet {
    let rawValue: Int

    static let bold = FontStyle(rawValue: 1 << 0)
    static let italic = FontStyle(rawValue: 1 << 1)
    static let underline = FontStyle(rawValue: 1 << 2)

    static let normal: FontStyle = []
}
```

## item37: Use EnumMap instead of ordinal indexing

Swift doesn't have a built-in concept of EnumMap as Java does, but you can use a
Dictionary with an enum as the key instead.

```swift
enum Day { case monday, tuesday, wednesday /*...*/ }

var dayEvents: [Day: [Event]] = [:]

// Populate dayEvents and access it using enum cases
```

## item38: Emulate extensible enums with interfaces

Swift doesn't have the concept of extensible enums, but you can use protocols
and structures to achieve similar results.

```swift
protocol Vehicle {
    var numberOfWheels: Int { get }
}

struct Bike: Vehicle {
    let numberOfWheels = 2
}

struct Car: Vehicle {
    let numberOfWheels = 4
}
```

## item39: Prefer annotations to naming patterns

Swift doesn't have annotations like Java, but you can use Swift's Attributes to
achieve similar results.

```swift
@available(iOS 13, *)
func doSomething() {
    // Implementation for iOS 13 and beyond
}
```

## item40: Consistently use the Override annotation

Always use the `override` keyword to denote when a method in a subclass is
intended to override a superclass's implementation.

```swift
class Base {
    func doSomething() { /* ... */ }
}

class Derived: Base {
    override func doSomething() { /* ... */ }
}
```

## item41: Use marker interfaces to define types

Use Swift protocols to define marker interfaces that help categorize and
organize types.

```swift
protocol CanFly {
    func fly()
}

class Bird: CanFly {
    func fly() { /* ... */ }
}
```

# Lambdas and Streams

## item42: Prefer lambdas to anonymous classes

In Swift, use closures instead of anonymous classes when creating inline
instances of a function-type, for example, when passing a function as an
argument to another function. Closures capture and store references to any
constants and variables from the context in which they are defined.

```swift
let numbers = [1, 2, 3, 4, 5]
let doubledNumbers = numbers.map { $0 * 2 }
```

## item43: Prefer method references to lambdas

Swift doesn't have a direct equivalent to method references in Java. When using
a method reference in Java, you can use a closure that wraps the function call
in Swift.

```swift
let words = ["apple", "banana", "cherry"]
let uppercasedWords = words.map { 
    $0.uppercased() 
}
```

## item44: Favor the use of standard functional interfaces

Utilize standard Swift function types or create custom function types that
promote code readability and reusability.

```swift
typealias Comparator<T> = (T, T) -> Bool

func sortDescending<T: Comparable>(a: T, b: T) -> Bool {
    return a > b
}

let numbers = [5, 1, 3, 2]
let sortedNumbers = numbers.sorted(by: sortDescending)
```

## item45: Use streams judiciously

In Swift, Focus on creating readable and maintainable code by using functional
programming constructs like `map`, `flatMap`, `compactMap`, `filter`, and
`reduce`.

```swift
let numbers = [0, 1, 2, 3, 4]
let evenNumbers = numbers.filter { $0 % 2 == 0 }
let totalEven = evenNumbers.reduce(0, +)
```

## item46: Prefer side-effect-free functions in streams

When using functional programming constructs, prefer using pure functions that
don't have side effects.

```swift
func double(_ number: Int) -> Int {
    return number * 2
}

let numbers = [1, 2, 3, 4, 5]
let doubledNumbers = numbers.map(double)
```

## item47: Prefer Collection to Stream as a return type

Swift doesn't have streams like Java. Use collection types, like Array, Set, and
Dictionary, as return types in functions for better clarity and code
flexibility.

```swift
func filterEvenNumbers(from numbers: [Int]) -> [Int] {
    return numbers.filter { $0 % 2 == 0 }
}
```

## item48: Use caution when making streams parallel

When working with concurrency, be cautious with parallelism to prevent race
conditions and deadlocks. Swift provides `DispatchQueue` and `OperationQueue` for
asynchronous processing.

```swift
import Dispatch

let queue = DispatchQueue(label: "com.test.parallel", attributes: .concurrent)
queue.async {
    // Perform concurrent task
}
```

# Methods

## item49: Check parameters for validity

Check input parameters before using them in your methods, use preconditions to
enforce valid inputs and prevent the method from processing invalid data.

```swift
func divide(numerator: Int, denominator: Int) -> Double {
    precondition(denominator != 0, "denominator should not be zero")
    return Double(numerator) / Double(denominator)
}
```

## item50: Make defensive copies when needed

When dealing with mutable objects, create a defensive copy to prevent unexpected modifications to the original object.

```swift
let originalArray = NSMutableArray(array: [1, 2, 3])
let defensiveCopy = originalArray.copy() as! NSArray
```

## item51: Design method signatures carefully

Design method signatures for clarity, simplicity, and flexibility. Make
parameter names and types clear and self-explanatory. Remove unnecessary
parameters and avoid varargs if possible.

```swift
func sort<T: Comparable>(array: [T], by comparator: Comparator<T>) -> [T] {
    // Implementation
}
```

## item52: Use overloading judiciously

Overload methods when it helps to provide clearer, simpler, and more flexible
APIs for different input types or use cases.

```swift
func display(message: String) {
    print(message)
}

func display(number: Int) {
    print(number)
}
```

## item53: Use varargs judiciously

Use variadic parameters in functions sparingly when it makes the API more
flexible and self-explanatory.

```swift
func sum(numbers: Int...) -> Int {
    return numbers.reduce(0, +)
}
```

## item54: Return empty collections or arrays, not nulls

Return empty collections or arrays instead of returning nulls to avoid
unexpected nil values which may cause crashes.

```swift
func getFruits() -> [String] {
    return ["apple", "banana", "cherry"]
}
```

## item55: Return optionals judiciously

Use optionals for return types in functions when it is necessary to indicate
that there might not be a valid result.

```swift
func findIndex(of element: Int, in array: [Int]) -> Int? {
    return array.firstIndex(of: element)
}
```

## item56: Write doc comments for all exposed API elements

Provide documentation comments for all public and exposed API elements to make
it easy for other developers to understand and use your code.

```swift
/// Calculates the factorial of a given integer.
///
/// - Parameter number: The integer whose factorial is to be calculated.
/// - Returns: The factorial of the given integer.
func factorial(of number: Int) -> Int {
    // Implementation
}
```

# General Programming

## item57: Minimize the scope of local variables

Declare local variables in the smallest scope possible. Initialize them where
they are declared, if possible. This makes the code more readable and reduces
errors.

```swift
// Bad practice
var sum: Int
for number in numbers {
    sum += number
}

// Good practice
var sum = 0
for number in numbers {
    sum += number
}
```

## item58: Prefer for-each loops to traditional for loops

Use for-each loops (called `for-in` loops in Swift) when iterating over
collections or ranges instead of traditional for loops, as they are more
readable and less error-prone.

```swift
let names = ["Alice", "Bob", "Charlie"]

// Bad practice
for i in 0..<names.count {
    print(names[i])
}

// Good practice
for name in names {
    print(name)
}
```

## item59: Know and use the libraries

Utilize the standard libraries provided by Swift, as they are well-tested,
promote code reusability, and lead to more efficient code.

```swift
// Using standard library functions
let numbers = [1, 2, 3, 4, 5]

let sum = numbers.reduce(0, +)
let squaredNumbers = numbers.map { $0 * $0 }
let evenNumbers = numbers.filter { $0 % 2 == 0 }
```

## item60: Avoid float and double if exact answers are required

Use the `Decimal` type when working with precise numeric values to avoid issues
with floating point inaccuracies.

```swift
// Bad practice
let a: Double = 0.1
let b: Double = 0.2
let result: Double = a + b // 0.30000000000000004

// Good practice
let a = Decimal(string: "0.1")!
let b = Decimal(string: "0.2")!
let result = a + b // 0.3
```

## item61: Prefer primitive types to boxed primitives

In Swift, this item is less relevant because there is no significant difference
between primitive types and their object representations. Swift will
automatically bridge between them.

## item62: Avoid strings where other types are more appropriate

Don't use strings to represent complex data, use custom types or enums when
appropriate. This enhances readability and maintainability.

```swift
// Bad practice
let userStatusString = "active" // Could be any string

// Good practice
enum UserStatus {
    case active, inactive, suspended
}
let userStatus = UserStatus.active
```

## item63: Beware the performance of string concatenation

When concatenating a large number of strings, use the `joined(separator:)`
method for better performance.

```swift
let words = ["Swift", "is", "awesome"]

// Bad practice
var sentence = ""
for word in words {
    sentence += " " + word
}

// Good practice
let sentence = words.joined(separator: " ")
```

## item64: Refer to objects by their interfaces

Swift uses protocols instead of interfaces. When writing code, prefer to program
to the protocol rather than a concrete implementation when possible.

```swift
protocol Drawable {
    func draw()
}

class Circle: Drawable {
    func draw() {
        print("Drawing a circle")
    }
}

class Square: Drawable {
    func draw() {
        print("Drawing a square")
    }
}

let shapes: [Drawable] = [Circle(), Square()]
for shape in shapes {
    shape.draw()
}
```

## item65: Prefer interfaces to reflection

Swift doesn't commonly use reflection like other programming languages. Instead,
utilize protocols and generics to write more type-safe and efficient code.

## item66: Use native methods judiciously

Be cautious when mixing Swift code with C or other languages, as it can
introduce performance or safety issues. Prefer Swift-native APIs when possible.

## item67: Optimize judiciously

Optimize code only when necessary and after measuring performance. Focus on
writing clear, maintainable code, and let the compiler handle optimizations.

## item68: Adhere to generally accepted naming conventions

Follow Swift naming conventions, such as UpperCamelCase for types,
lowerCamelCase for properties and methods, and clear, descriptive names for
functions.

# Exceptions

## item69: Use exceptions only for exceptional conditions

In Swift, exceptions should be used only for exceptional cases, i.e., when
something unexpected happens during the normal execution of the program. They
should not be used for normal control flow.

```swift
// Good
func divide(_ a: Int, _ b: Int) throws -> Int {
    guard b != 0 else {
        throw DivisionError.divisionByZero
    }
    return a / b
}

// Bad
func divide(_ a: Int, _ b: Int) throws -> Int {
    if b == 0 {
        throw DivisionError.divisionByZero
    } else {
        return a / b
    }
}
```

## item70: Use checked exceptions for recoverable conditions and runtime exceptions for programming errors

In Swift, there are no checked exceptions like Java. Swift uses the concept of
error handling using `throws`, `try`, and `catch`. Throw an error when there is
a recoverable condition and use Fatal Error or precondition checks for
programming errors.

```swift
enum FileError: Error {
    case fileNotFound
}

func readFile(filename: String) throws -> String {
    guard let data = FileManager.default.contents(atPath: filename) else {
        throw FileError.fileNotFound
    }
    return data
}

func main() {
    do {
        let content = try readFile(filename: "somefile.txt")
    } catch {
        print("Error occurred: \(error)")
    }
}

main()
```

## item71: Avoid unnecessary use of checked exceptions

Swift doesn't use checked exceptions. However, it's important to avoid using
error throwing unnecessarily. If there's a more appropriate way to handle an
error, like a default value or a custom result type, use that instead.

```swift
enum Result<T> {
    case success(T)
    case error(Error)
}

func readDataFromFile() -> Result<String> {
    // Reading data from file
    if (...) { // some error occurred
        return .error(FileError.fileNotFound)
    }
    return .success("File content")
}
```

## item72: Favor the use of standard exceptions

Swift provides several standard exceptions, and it's recommended to use them
instead of defining custom ones when possible. Examples of standard Swift errors
are `URLError`, `DecodingError`, and `NSError`.

```swift
import Foundation

func fetchURL() throws -> String {
    let url = URL(string: "https://www.example.com")!
    return try String(contentsOf: url)
}

do {
    try fetchURL()
} catch let error as URLError {
    print("Error: \(error.localizedDescription)")
} catch {
    print("Unknown error: \(error)")
}
```

## item73: Throw exceptions appropriate to the abstraction

When using exceptions, make sure they align with the abstraction level of the
method in which they're thrown. If a low-level exception needs to be propagated
to a higher level, it's better to create a higher-level exception and throw that
instead.

```swift
enum NetworkError: Error {
    case requestFailed
    case invalidURL
}

func fetchData() throws -> Data {
    let urlString = "https://www.example.com"
    
    guard let url = URL(string: urlString) else {
        throw NetworkError.invalidURL
    }
    
    return try Data(contentsOf: url)
}

do {
    try fetchData()
} catch NetworkError.requestFailed {
    print("Request failed")
} catch NetworkError.invalidURL {
    print("Invalid URL")
}
```

## item74: Document all exceptions thrown by each method

Document any error cases that a function or method may throw, so that users of
the function or method know what to expect and how to handle the errors.

```swift
/// Reads a file with the given filename and returns its content.
///
/// - Parameter filename: The name of the file to read.
/// - Returns: The content of the file as a string.
/// - Throws: An error of type `FileError.fileNotFound` if the file cannot be found.
func readFile(filename: String) throws -> String {
    // ...
}
```

## item75: Include failure-capture information in detail messages

When defining custom errors, include useful information for debugging purposes.

```swift
enum ConversionError: Error {
    case invalidFormat(file: String)
}

func convertToPDF(file: String) throws {
    if (...) {// Invalid file format 
        throw ConversionError.invalidFormat(file: file)
    }
    // Continue conversion
}

do {
    try convertToPDF(file: "example.docx")
} catch let error as ConversionError {
    switch error {
    case .invalidFormat(let file):
        print("Invalid format for file '\(file)'")
    }
}
```

## item76: Strive for failure atomicity

When designing methods that can fail, try to make them atomic, meaning that if
the method fails, it leaves the object in the state it was before the call.

```swift
class ArrayStack<T> {
    private var array: [T] = []
    
    func push(_ element: T) {
        array.append(element)
    }
    
    func pop() throws -> T {
        guard !array.isEmpty else {
            throw StackError.empty
        }
        return array.removeLast()
    }
}

enum StackError: Error {
    case empty
}

let stack = ArrayStack<String>()
stack.push("hello")
stack.push("world")

do {
    stack.pop()
    stack.pop()
    try stack.pop()
} catch StackError.empty {
    print("Stack is empty")
}
```

## item77: Don’t ignore exceptions

When encountering an error, handle it either by fixing the error, propagating it
up the call stack, or at least logging it. Ignoring an exception may lead to
incorrect behavior or a difficult-to-debug issue.

```swift
do {
    try fetchURL()
} catch {
    print("Error occurred: \(error)")
}
```

# Concurrency

## item78: Synchronize access to shared mutable data

In Swift, to safely access shared mutable data, use synchronization constructs like `DispatchSemaphore`, `DispatchQueue`, or `NSLock`.

```swift
private let concurrentQueue = DispatchQueue(label: "com.example.concurrentQueue", attributes: .concurrent)
private var sharedMutableData: [Int] = []

func addItem(_ item: Int) {
    concurrentQueue.async(flags: .barrier) {
        sharedMutableData.append(item)
    }
}

func getItem(at index: Int) -> Int? {
    concurrentQueue.sync {
        guard index < sharedMutableData.count else {
            return nil
        }
        return sharedMutableData[index]
    }
}
```

## item79: Avoid excessive synchronization

In Swift, limit the use of locks or dispatch queues to restrict the least amount
of code possible while still ensuring thread safety.

```swift
private let lock = NSLock()
private var fastAccessData: [Int] = []

func getData(processing: (Int) -> Int) -> [Int] {
    lock.lock()
    let data = fastAccessData
    lock.unlock()

    return data.map(processing)
}
```

## item80: Prefer executors, tasks, and streams to threads

In Swift, avoid using low-level threads directly. Use DispatchQueue and Grand
Central Dispatch (GCD) which provide powerful, high-level concurrency
primitives.

```swift
let backgroundQueue = DispatchQueue.global(qos: .background)

backgroundQueue.async {
    // Perform task in the background
    DispatchQueue.main.async {
        // Update UI on the main thread
    }
}
```

## item81: Prefer concurrency utilities to wait and notify

Use concurrency utilities like `DispatchGroup` and `DispatchSemaphore` for
synchronization rather than low-level mechanisms like `Thread` and `NSLock`.

```swift
let group = DispatchGroup()
let queue = DispatchQueue.global(qos: .userInitiated)

for i in 0..<5 {
    queue.async(group: group) {
        print("Working on task", i)
    }
}

group.notify(queue: .main) {
    // All tasks finished
}
```

## item82: Document thread safety

Clearly document the thread safety guarantees of your types. Use annotations
like `ThreadSafe` and `NotThreadSafe` to indicate your intentions.

```swift
/// A thread-safe, mutable integer value
public class ThreadSafeInt {
    private let lock = NSLock()
    private var _value: Int

    public init(value: Int) {
        _value = value
    }

    public var value: Int {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _value
        }
        set {
            lock.lock()
            _value = newValue
            lock.unlock()
        }
    }
}
```

## item83: Use lazy initialization judiciously

Use lazy initialization to defer computation of a value or creation of an object
until it's actually needed.

```swift
class Database {
    lazy var persistentStore: CoreDataPersistentStore = {
        return CoreDataPersistentStore()
    }()
}
```

## item84: Don’t depend on the thread scheduler

Do not rely on the underlying thread scheduler for the correct operation of your
application. Write non-blocking code and use Swift's concurrency utilities to
ensure responsiveness.

```swift
DispatchQueue.global(qos: .userInitiated).async {
    // Perform task in the background
    DispatchQueue.main.async {
        // Update UI or respond on the main thread
    }
}
```

# Serialization

## item85: Prefer alternatives to Java serialization

In Swift, this means that you should prefer to use Codable over NSCoding as the
way to serialize and deserialize objects. Codable is a more modern and safer
approach in comparison to NSCoding.

```swift
import Foundation

// Define a Codable struct
struct Person: Codable {
    var name: String
    var age: Int
}

let person = Person(name: "John Doe", age: 30)

let encoder = JSONEncoder()
let jsonData = try encoder.encode(person)

let decoder = JSONDecoder()
let decodedPerson = try decoder.decode(Person.self, from: jsonData)
```

## item86: Implement Serializable with great caution

Since Codable is the preferred serialization mechanism in Swift, you should
implement it cautiously to prevent vulnerabilities. Ensuring that your data
structures are simple and easy to reason will help you avoid security issues
when serializing or deserializing your objects.

## item87: Consider using a custom serialized form

Codable allows you to define custom encoding and decoding logic for your data
structures. This can be useful for backward compatibility or optimizing the
serialized format.

```swift
import Foundation

struct Person: Codable {
    var name: String
    var age: Int

    enum CodingKeys: String, CodingKey {
        case name
        case age
    }

    // Custom decoding
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.name = try container.decode(String.self, forKey: .name)
        self.age = try container.decode(Int.self, forKey: .age)
    }

    // Custom encoding
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encode(age, forKey: .age)
    }
}
```

## item88: Write readObject methods defensively

Since Swift uses Codable instead of the readObject method of Java's
Serializable, ensuring that your custom decoding logic is written defensively
will help prevent deserialization vulnerabilities.

```swift
import Foundation

struct Person: Codable {
    var name: String
    var age: Int

    // Custom decoding
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.name = try container.decode(String.self, forKey: .name)
        self.age = try container.decode(Int.self, forKey: .age)

        // Defensive checks
        if self.name.isEmpty {
            throw DecodingError.dataCorruptedError(forKey: .name, in: container, debugDescription: "Name cannot be empty")
        }
        if self.age < 0 {
            throw DecodingError.dataCorruptedError(forKey: .age, in: container, debugDescription: "Age must be greater than or equal to 0")
        }
    }
}
```

## item89: For instance control, prefer enum types to readResolve

Swift, unlike Java, does not have a direct equivalent for the readResolve
method. However, you can achieve instance control by using a custom decoding
initializer to enforce desired behavior during deserialization.

## item90: Consider serialization proxies instead of serialized instances

In Swift, you can use a separate type as a proxy for serialization and
deserialization of your main type. This can be useful for code maintenance,
backward compatibility, and protecting sensitive data.

```swift
import Foundation

struct Person: Codable {
    var name: String
    var age: Int

    // Use a separate type as proxy for serialization
    struct Proxy: Codable {
        var name: String
        var age: Int
    }

    func encode(to encoder: Encoder) throws {
        let proxy = Proxy(name: self.name, age: self.age)
        try proxy.encode(to: encoder)
    }

    init(from decoder: Decoder) throws {
        let proxy = try Proxy(from: decoder)
        self.init(name: proxy.name, age: proxy.age)
    }
}
```
